// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx_codegen.h"
#include "common.h"
#include "compile_op.h"
#include "compile_op_kernel.h"
#include "cputhread.h"
#include "group.h"
#include "ipc_mapper.h"
#include "ptx.h"

namespace moodist {

using namespace ptx;

const char* computeTarget(int computeMajor, int computeMinor) {
  if (computeMajor >= 10) {
    if (computeMinor >= 3) {
      return "sm_103a";
    } else {
      return "sm_100a";
    }
  } else if (computeMajor == 9) {
    return "sm_90a";
  } else if (computeMajor == 8 && computeMinor >= 9) {
    return "sm_89";
  } else if (computeMajor == 8) {
    return "sm_80";
  } else if (computeMajor == 7) {
    return "sm_70";
  }
  return "sm_60";
}

// compile_op_params (see paramBuffer in generateKernel): the words one launch copies (its second parameter, 31 KiB),
// and its block size
static constexpr uint32_t paramFillWords = 3968;
static constexpr uint32_t paramFillBlockSize = 256;

// The reduce loop's shared memory (see generateKernel): 2 to reduceMaxStages stages of chunkSize bytes with an mbarrier
// each, reduceResults result buffers, a ring of up to reduceMaxStages + 2 claimed items (8 bytes each), and 256 bytes
// of alignment padding
static constexpr uint32_t reduceResults = 2;
static constexpr uint32_t reduceMaxStages = 16;
static size_t reduceStageBytes(const KernelConfig& config) {
  return config.chunkSize + 8;
}
static size_t reduceFixedBytes(const KernelConfig& config) {
  return 256 + reduceResults * config.chunkSize + 8 * (reduceMaxStages + 2);
}

std::string reduceConfigError(const KernelConfig& config) {
  if (config.pipelines != 1 || config.chunkSize % (16 * config.blockSize) != 0) {
    return fmt::sprintf("kernels with reductions need one pipeline per block and chunks of a multiple of 16 bytes per "
                        "thread (block size %d, chunk size %d, %d pipelines)",
        config.blockSize, config.chunkSize, config.pipelines);
  }
  if (config.sharedMemory < reduceFixedBytes(config) + 2 * reduceStageBytes(config)) {
    return fmt::sprintf("%d bytes of shared memory are not enough for the reduce loop with chunks of %d bytes",
        config.sharedMemory, config.chunkSize);
  }
  return "";
}

struct Profiling {
  bool enabled = false;
  HashMap<std::string, size_t> nameToIndex;
  size_t nextIndex = 0;
  std::optional<u64> clockOffset;

  std::optional<u64> addrData;

  std::optional<u32> nextEntryIndex;
  std::optional<u32> dataOffset;

  static constexpr size_t maxBytesPerThread = 1024 * 128;

  size_t dataSize(const KernelConfig& config) {
    return maxBytesPerThread * config.blockSize * config.gridSize;
  }

  void start(const KernelConfig& config, u64 addrData) {
    u64 myclock = clock64();
    clockOffset = -myclock;
    enabled = true;

    this->addrData = addrData;
    nextEntryIndex = 0;
    dataOffset = maxBytesPerThread * (config.blockSize * blockIdx_x() + threadIdx_x());

    nameToIndex["<invalid>"] = nextIndex++;
  }
  void finish() {
    if (!enabled) {
      return;
    }
    u32 o = 8 + 16 * *nextEntryIndex;
    PRED(o + 16 > maxBytesPerThread) trap();
    st_global_u32(*addrData + widen(*dataOffset), *nextEntryIndex);
  }
  void enter(std::string name, std::optional<u32> arg = {}) {
    if (!enabled) {
      return;
    }
    if (!nameToIndex.contains(name)) {
      nameToIndex[name] = nextIndex++;
    }
    u32 o = 8 + 16 * *nextEntryIndex;
    u32 a = *dataOffset + o;
    st_global_u64(*addrData + widen(a), clock64() + *clockOffset + (nameToIndex[name] << 48));
    *nextEntryIndex = *nextEntryIndex + 1;
  }
  void leave() {
    enter("");
  }
};

static thread_local Profiling profiling;

// The copy engine's stage ring, run by one lane per pipeline (a block runs config.pipelines pipelines, on the first
// lane of each of its first warps). A pipeline copies chunks through a ring of numStages stages of chunkSize bytes in
// shared memory: a bulk load (TMA) of a chunk into a stage, then bulk stores of the stage to each of the chunk's
// destinations. A TMA copy is latency bound: an SM reaches its NVLink bandwidth (~52 GB/s on GB300) only with
// ~100 KB of loads in flight. So the pipeline keeps loads in flight ahead of its stores: after issuing the stores of
// a chunk, it waits until the stores of an earlier chunk have read their stage (cp.async.bulk.wait_group.read) and
// loads the next chunk into that stage (see storeLag and the loops that use the ring). A stage's record holds what
// its stores need: the chunk's offset and size, the generic address of the array of destination addresses (u64),
// and a tag for the caller. The ring's position persists across copies (the mbarrier phases of the stages
// continue).
struct PipeRing {
  static constexpr uint32_t recordSize = 32;
  uint32_t chunkSize;
  uint32_t numStages;
  u32 buffers;  // stage s at buffers + chunkSize * s
  u32 barriers; // the mbarrier of stage s at barriers + 8 * s
  u32 records;  // the record of stage s at records + recordSize * s

  u32 loadStage = 0;   // the stage of the next load
  u32 storeStage = 0;  // the stage of the next store
  u32 storeParity = 0; // the phase parity of the mbarrier of that stage

  PipeRing(uint32_t chunkSize, uint32_t numStages, const u32& buffers, const u32& barriers, const u32& records)
      : chunkSize(chunkSize), numStages(numStages), buffers(buffers), barriers(barriers), records(records) {}

  void init() {
    for (uint32_t s : range(numStages)) {
      mbarrier_init(barriers + 8 * s, 1);
    }
    fence_mbarrier_init();
  }

  // Load bytes at src + offset into the next stage, to be stored to the numDestinations addresses (plus offset) in
  // the array at dsts; tag is returned by its store(), aux kept for nextStoreAux()
  void load(const u64& src, const u64& offset, const u32& bytes, const u64& dsts, const u32& numDestinations,
      const u32& tag, const u32& aux) {
    u32 record = records + loadStage * recordSize;
    st_shared_u64(record, offset);
    st_shared_u64(record + 8, dsts);
    st_shared_u32(record + 16, bytes);
    st_shared_u32(record + 20, numDestinations);
    st_shared_u32(record + 24, tag);
    st_shared_u32(record + 28, aux);
    u32 barrier = barriers + loadStage * 8;
    mbarrier_arrive_expect_tx(barrier, bytes);
    cp_async_bulk_shared_global(buffers + loadStage * chunkSize, src + offset, bytes, barrier);
    u32 next = loadStage + 1;
    loadStage = selp_u32(0, next, next == numStages);
  }

  // The aux value of the chunk that store() stores next (its record is written when it is loaded)
  u32 nextStoreAux() {
    return ld_shared_u32(records + storeStage * recordSize + 28);
  }

  // Wait for the chunk of the next stage to arrive, and issue its stores (one bulk group). Returns its tag.
  u32 store() {
    u32 record = records + storeStage * recordSize;
    mbarrier_wait_parity_spin(barriers + storeStage * 8, storeParity);
    u64 offset = ld_shared_u64(record);
    u64 dsts = ld_shared_u64(record + 8);
    u32 bytes = ld_shared_u32(record + 16);
    u32 numDestinations = ld_shared_u32(record + 20);
    u32 tag = ld_shared_u32(record + 24);
    u32 buffer = buffers + storeStage * chunkSize;
    FOR(u32 i = 0, i < numDestinations, i += 1) {
      cp_async_bulk_global_shared(ld_u64(dsts + widen(i * 8)) + offset, buffer, bytes);
    }
    cp_async_bulk_commit_group();
    u32 next = storeStage + 1;
    Value wrap = next == numStages;
    storeStage = selp_u32(0, next, wrap);
    storeParity = storeParity ^ selp_u32(1, 0, wrap);
    return tag;
  }
};

struct KernelMappedBuffers {
  AllocatedArray& local;
  Vector<PeerArrayRef> remote;
};

struct MemoryKeyCounter {
  SpinMutex mutex;
  size_t nextKey = 1;
  Vector<size_t> freelist;

  HashMap<std::string, bool> freeMap;
  HashMap<size_t, Vector<std::string>> live;

  void put(size_t key) {
    std::lock_guard l(mutex);
    freelist.push_back(key);
    CHECK(live.contains(key));
    for (auto& s : live[key]) {
      freeMap[s] = true;
    }
    live.erase(key);
  }
  size_t get() {
    std::lock_guard l(mutex);
    size_t key;
    if (freelist.empty()) {
      key = nextKey++;
    } else {
      key = freelist.pop_back_value();
    }
    CHECK(!live.contains(key));
    live.emplace(key);
    return key;
  }

  std::string skey(const Group* group, const AllocatedArray* v) {
    return fmt::sprintf("%s-%#x\n", group->name, v->buffer.cudaPointer);
  }

  bool isFree(const Group* group, const AllocatedArray* v) {
    std::lock_guard l(mutex);
    return freeMap.contains(skey(group, v));
  }
  void add(size_t key, const Group* group, const AllocatedArray* v) {
    std::lock_guard l(mutex);
    CHECK(live.contains(key));
    live[key].push_back(skey(group, v));
    auto it = freeMap.find(skey(group, v));
    if (it != freeMap.end()) {
      freeMap.erase(it);
    }
  }
}& memoryKeyCounter = Global();

void returnKernelMemory(size_t key) {
  return memoryKeyCounter.put(key);
}
size_t acquireMemoryKey() {
  auto key = memoryKeyCounter.get();
  return key;
}

KernelMappedBuffers mapn(size_t memoryKey, compile_op::CompileContext& ctx, std::string name, Vector<uint32_t> ranks,
    size_t bytesPerConcurrencyIndex) {
  bytesPerConcurrencyIndex = std::max(std::bit_ceil(bytesPerConcurrencyIndex), (size_t)16);
  auto& x = ctx.cachedBuffers[""];
  AllocatedArray* local = nullptr;
  for (auto& v : x) {
    if (v->itembytes == bytesPerConcurrencyIndex && memoryKeyCounter.isFree(ctx.group, &*v)) {
      local = &*v;
      break;
    }
  }
  if (!local) {
    auto arr =
        std::make_unique<AllocatedArray>(ctx.group->allocateArrayDevice(bytesPerConcurrencyIndex, maxConcurrency));
    local = &*arr;
    ctx.group->buffersToReset.push_back(local);
    x.push_back(std::move(arr));
  }

  memoryKeyCounter.add(memoryKey, ctx.group, local);

  const uint32_t rank = ctx.group->rank;

  KernelMappedBuffers r(*local);

  Vector<uintptr_t> mapped(ranks.size());
  for (size_t i : indices(ranks)) {
    if (ranks[i] == rank) {
      mapped[i] = local->buffer.cudaPointer;
      continue;
    }
    ctx.group->ipcMapper->requestAddressRank(ranks[i], local->buffer.cudaPointer, local->buffer.bytes, &mapped[i]);
  }
  ctx.group->ipcMapper->wait();

  for (size_t i : indices(ranks)) {
    ctx.send(ranks[i], mapped[i], local->itembytes);
  }
  r.remote.resize(ranks.size());
  for (size_t i : indices(ranks)) {
    size_t itembytes;
    ctx.receive(ranks[i], r.remote[i].base, itembytes);
    r.remote[i].itembytes = itembytes;
  }

  return r;
}

// Quick note that PTX docs do not even mention inter-process memory semantics.
// CUDA Programming Guide talks about it a bit, but it's quite vague.
// In practice, we assume that sys scope includes the entire universe.
// We also assume that cuda has sys scope acquire fence before a kernel and
// release after a kernel.
// In practice, nvlink data is not cached, but we do not make this assumption
// (ie we still use proper fences).
// This kernel runs in the remote memory synchronization domain.

std::shared_ptr<KernelHandle> generateKernel(const Group* group, const KernelConfig& config, std::string target,
    const compile_op::Graph& graph, compile_op::CompileContext& ctx, bool tuning) {
  using namespace ptx;

  uint32_t blockSize = config.blockSize;
  uint32_t numBlocks = config.gridSize;

  size_t rank = group->rank;

  auto rv = std::make_shared<KernelHandle>();
  rv->memoryKey = acquireMemoryKey();

  Module mod;
  mod.target = target;
  mod.topLevelComments.push_back(fmt::sprintf("block size: %d  grid size: %d", config.blockSize, config.gridSize));
  mod.topLevelComments.push_back(fmt::sprintf("shared memory: %d", config.sharedMemory));
  setModule(&mod);

  CHECK(!graph.cudaEdges.empty());

  Vector<uint32_t> ranks;
  Vector<uint64_t> localNodes;
  for (auto& v : graph.cudaEdges) {
    for (auto& l : {&v.sources, &v.destinations}) {
      for (auto& x : *l) {
        if (std::ranges::find(ranks, x.rank) == ranks.end()) {
          ranks.push_back(x.rank);
        }
        if (std::ranges::find(localNodes, x.id) == localNodes.end()) {
          localNodes.push_back(x.id);
        }
      }
    }
  }
  std::ranges::sort(ranks);

  using Edge = compile_op::Graph::Edge;
  using Node = compile_op::Graph::Node;

  auto rankIndex = [&](uint32_t rank) {
    auto it = std::ranges::find(ranks, rank);
    CHECK(it != ranks.end());
    return it - ranks.begin();
  };

  size_t myIndex = rankIndex(rank);

  Vector<size_t> myPeerIndex(ranks.size());

  for (size_t i : indices(ranks)) {
    ctx.send(ranks[i], i);
  }
  for (size_t i : indices(ranks)) {
    ctx.receive(ranks[i], myPeerIndex[i]);
  }

  // The exit sync buffer holds one row of ranks.size() slots per block. Peers can have different numbers of
  // ranks (irregular graphs), so writers use the row length of the peer they write to.
  Vector<size_t> peerRanksSize(ranks.size());
  for (uint32_t r : ranks) {
    ctx.send(r, ranks.size());
  }
  for (size_t i : indices(ranks)) {
    ctx.receive(ranks[i], peerRanksSize[i]);
  }

  struct SyncIndex {
    size_t index;
    size_t stride;
  };

  HashMap<uint32_t, HashMap<uint64_t, SyncIndex>> dependencyIndices;

  for (size_t r : ranks) {
    ctx.send(r, localNodes);
  }
  for (size_t r : ranks) {
    Vector<uint64_t> nn;
    ctx.receive(r, nn);
    for (size_t i : indices(nn)) {
      uint64_t id = nn[i];
      SyncIndex si;
      si.index = i;
      si.stride = nn.size();
      CHECK(!dependencyIndices[r].contains(id));
      dependencyIndices[r][id] = si;
    }
  }

  // The nodes that edges read, as seen by this rank and its peers. An edge must signal its destinations if some
  // edge reads one of them, and that edge may be one this rank does not know (finalize gives a rank the edges
  // that touch its nodes): e.g. in a ring_write chain, rank 0 writes rank 1's node, which the edge from rank 1 to
  // rank 2 reads. The reader's executor is a peer of the writer (the node's owner, or the edge's destination,
  // which knows the edge writing the node), so the union over the peers is complete.
  HashMap<uint64_t, bool> readNodes;
  {
    Vector<uint64_t> myReadNodes;
    for (auto& v : graph.cudaEdges) {
      for (auto& n : v.sources) {
        myReadNodes.push_back(n.id);
      }
    }
    for (size_t r : ranks) {
      ctx.send(r, myReadNodes);
    }
    for (size_t r : ranks) {
      Vector<uint64_t> nn;
      ctx.receive(r, nn);
      for (uint64_t id : nn) {
        readNodes[id] = true;
      }
    }
  }

  auto map = [&](std::string name, size_t bytes) {
    return mapn(rv->memoryKey, ctx, name, ranks, bytes);
  };

  size_t addrAddrsBlockStride = graph.localCudaTensorMappings.size();
  auto addrAddrs = map("addresses", sizeof(uint64_t) * addrAddrsBlockStride * numBlocks);
  auto syncAddrs = map("syncs", sizeof(uint32_t) * ranks.size() * numBlocks);
  auto depFilled = map("depFilled", sizeof(uint32_t) * localNodes.size() * numBlocks);

  // The address arrays: the tensor addresses (8 bytes per tensor), the addresses at which the peers map this rank's
  // tensors (8 per mapping) and, with NVLS, the tensors' identities and multicast addresses (16 per tensor). Kernel
  // parameters are limited to 32764 bytes (an op with about 4000 tensor-peer mappings did not JIT), so an op whose
  // arrays could exceed paramBufferMinBytes (counting the NVLS words, decided further below) has them in a buffer
  // instead: a global array of the module with a copy per concurrency slot, which the module's second kernel,
  // compile_op_params, fills from its own parameters before each launch (launchCompileOpKernel). A slot's ops are
  // serialized on the GPU (customOp), so a launch's copy stays until its kernel has read it. MOODIST_COMPILE_OP_
  // PARAM_BUFFER=1 uses the buffer for every op (tests).
  static constexpr size_t paramBufferMinBytes = 32000;
  const size_t numTensorWords = std::max(graph.tensors.size(), (size_t)1);
  const size_t numMappingWords = std::max(graph.remoteCudaTensorMappings.size(), (size_t)1);
  const char* forceParamBuffer = std::getenv("MOODIST_COMPILE_OP_PARAM_BUFFER");
  const bool paramBuffer =
      sizeof(CompileOpCopyParameters) + 8 * (numTensorWords + numMappingWords) + 16 * numTensorWords >
          paramBufferMinBytes ||
      (forceParamBuffer && std::string_view(forceParamBuffer) == "1");

  auto* fn = mod.newFunction("compile_op_copy");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = blockSize;
    fn->addParamBytes(8, sizeof(CompileOpCopyParameters));
    if (!paramBuffer) {
      fn->addParamBytes(8, sizeof(uint64_t) * numTensorWords);
      fn->addParamBytes(8, sizeof(uint64_t) * numMappingWords);
    }

    activateNewBlock("entry");

    // --- Load kernel parameters ---
    u32 params = paramBase(0);
    u32 stepValue = loadParamField(params, 0, ValType::U32);
    u32 concurrencyIndex = loadParamField(params, 4, ValType::U32);
    static_assert(offsetof(CompileOpCopyParameters, nvlsCounterBase) == 8);

    u32 threadIndex = threadIdx_x();
    u32 blockIndex = blockIdx_x();

    u32 warpIndex = threadIndex / 32;
    u32 laneIndex = laneid();

    // Kernel profiling (debugging aid; MOODIST_COMPILE_OP_PROFILE=N): in the kernel of each op (not in the tuning
    // candidates), every thread records a timestamp at each profiling.enter(), and the op's N-th launch writes them as
    // a Chrome trace (dumpProfiling in processgroup.cc). Needs Profiling::maxBytesPerThread of device memory per
    // thread.
    profiling = Profiling();
    if (const char* env = std::getenv("MOODIST_COMPILE_OP_PROFILE"); env && !tuning) {
      int launch = std::atoi(env);
      REQUIRE(launch >= 1, "MOODIST_COMPILE_OP_PROFILE must be the number of the launch to trace (1 or more)");
      auto profilingBuffer = ctx.group->allocateDevice(profiling.dataSize(config));
      profiling = Profiling();
      profiling.start(config, profilingBuffer.cudaPointer);
      rv->profilingData = std::move(profilingBuffer);
      rv->profilingBytesPerThread = Profiling::maxBytesPerThread;
      rv->profilingCountdown = launch - 1;
    }

    auto predicated = [&](auto pred, auto f) {
      return [&, pred = std::move(pred), f = std::move(f)](auto&&... args) {
        if constexpr (std::is_same_v<decltype(f(args...)), void>) {
          PRED(pred) f(args...);
        } else {
          decltype(f(args...)) tmp;
          PRED(pred) tmp = f(args...);
          return tmp;
        }
      };
    };

    profiling.enter("parity setup");

    u64 parityAddr = addGlobalVar(4, maxConcurrency * numBlocks * 4, "parity") +
                     widen(4 * numBlocks * concurrencyIndex + 4 * blockIndex);
    u32 parityBit = 0;
    auto isThreadZero = threadIndex == 0;
    parityBit = predicated(isThreadZero, ld_global_u32)(parityAddr) ^ 1;
    profiling.enter("parity setup bar");
    auto parityPred = bar_red_or(0, blockSize, (parityBit != 0) & isThreadZero);
    profiling.enter("parity setup post-bar");
    predicated(isThreadZero, st_global_u32)(parityAddr, parityBit);
    parityBit = selp_u32(1, 0, parityPred);

    u64 parityTag = widen(parityBit) << 63;

    profiling.leave();

    auto concurrency = [&](const auto& v) {
      if constexpr (std::is_same_v<std::decay_t<decltype(v)>, PeerArrayRef>) {
        return v.base + widen(v.itembytes * concurrencyIndex);
      } else {
        return v.buffer.cudaPointer + widen(v.itembytes * concurrencyIndex);
      }
    };

    auto findLocalMapping = [&](const auto& r) {
      for (size_t i : indices(graph.localCudaTensorMappings)) {
        auto& v = graph.localCudaTensorMappings[i];
        if (v.rank == r.rank && v.tensorIndex == r.tensorIndex) {
          return i;
        }
      }
      log.error("failed to find mapping for %d %d\n", r.rank, r.tensorIndex);
      for (size_t i : indices(graph.localCudaTensorMappings)) {
        auto& v = graph.localCudaTensorMappings[i];
        log.error(" index %d: %d %d\n", i, v.rank, v.tensorIndex);
      }
      CHECK(false);
    };

    // ---- The edges this rank executes ----
    Vector<Edge> localCopies;

    for (auto& v : graph.cudaEdges) {
      if (v.sources.empty()) {
        // Relay edge: this rank's CPU thread fills the destination over RDMA
        CHECK(v.rdmaSignal >= 0 && v.destinations.size() == 1);
        CHECK(v.executorRank == rank && v.destinations[0].rank == rank);
        localCopies.push_back(v);
        continue;
      }
      // One source, or for a reduction several (see Graph::Edge)
      CHECK(v.sources.size() == 1 || compile_op::isReduction(graph.reduce));
      CHECK(v.destinations.size() >= 1);
      bool localSource = false;
      for (const Node& n : v.sources) {
        CHECK(std::ranges::find(ranks, n.rank) != ranks.end());
        localSource |= n.rank == rank;
      }
      const Node* localdst = nullptr;
      for (const Node& n : v.destinations) {
        CHECK(!n.filled);
        // gotta decide some things about the cudaEdges, rdmaEdges split
        // are these expected to be completely sepatate, such that
        // the following CHECK always passes?
        // this separation is perhaps broken by design
        //    what happens to reductions with mixed cuda and rdma sources?
        //    well, we could set up a temporary buffer with a dependency chain
        // feasible for destination, though...
        CHECK(std::ranges::find(ranks, n.rank) != ranks.end());
        if (n.rank == rank) {
          // fixme: we could have multiple outputs that map to the same cell.
          //        not really sure if that should be handled here, though.
          //        might be better to have a pass in compile_op.cc which sets
          //        up a dependency chain.
          // (the reduce loop stores to any destinations: in an all-reduce in two rounds, an executor stores its share
          // to its output and to its scratch tensor, see crossDomainReductions in compile_op.cc)
          CHECK(localdst == nullptr || compile_op::isReduction(graph.reduce));
          localdst = &n;
        }
      }
      CHECK(localSource || localdst);

      if (v.executorRank == rank) {
        localCopies.push_back(v);
      }
    }
    // reduceKernel: this rank executes reductions (edges with several sources, or parts of a reduction across NVLink
    // domains or in two phases), and runs the reduce loop below
    const bool reduceKernel = std::ranges::any_of(localCopies, [](const Edge& e) {
      return e.sources.size() > 1 || !e.sourceSignals.empty() || !e.sourceDoneSignals.empty() || e.doneSignal >= 0;
    });
    if (reduceKernel) {
      // The reduce loop's blocks take items in order, and a block waits for a source from another NVLink domain only
      // once it has finished its items before (see reduceLoop). The edges go in the order of crossDomainReductions in
      // compile_op.cc, except that the edges that compute partial results which ranks in other NVLink domains (or
      // other edges of this rank's) wait for, and wait for nothing, come first: they have all been taken before any
      // block waits. In an all-reduce in two rounds, the edges that compute shares of a part (which other ranks wait
      // for) and the copies of the other shares follow part by part, in the same order on every rank: a copy waits
      // for shares of its part, which wait only for partial results, so no rank waits for a rank that waits for it.
      // (The order of peers below is for copies: a reduction's item reads all its sources and stores to all its
      // destinations.)
      std::stable_partition(localCopies.begin(), localCopies.end(), [](const Edge& e) {
        return e.doneSignal >= 0 && e.sourceSignals.empty() && e.sourceDoneSignals.empty();
      });
    } else {
      // Relay edges first: the scan for ready edges below stops at the first ready copy, and always reaches them
      auto firstCopy = std::stable_partition(localCopies.begin(), localCopies.end(), [](const Edge& e) {
        return e.sources.empty();
      });
      // Then the copies in the order of the peer they move data to, or from, rotated to start at this rank: in rank
      // order, every rank would write to the same peer at once (in an all-to-all, all ranks to rank 0 first, then to
      // rank 1, ...), whose NVLink ingress then limits all of them while the other peers idle. An edge counts by its
      // first destination (multicast edges list their destinations rotated the same way), or by its source if it
      // reads another rank's node into this rank (the read graphs), where the source's egress limits the readers in
      // the same way (all-gather on 32 GB300 with the read graph and 16 blocks: 74 GB/s per GPU, 631 rotated by
      // source).
      std::stable_sort(firstCopy, localCopies.end(), [&](const Edge& a, const Edge& b) {
        auto key = [&](const Edge& e) {
          size_t r = e.destinations.at(0).rank;
          if (r == rank && !e.sources.empty() && e.sources[0].rank != rank) {
            r = e.sources[0].rank;
          }
          return (r + group->size - rank) % group->size;
        };
        return key(a) < key(b);
      });
    }

    // nowaits: no edge waits for another edge or for the CPU thread, so the edges run in order (the reduce loop's
    // waits for the sources that the CPU thread reads from other NVLink domains aside, see Edge::sourceSignals)
    bool nowaits = true;
    bool hasRemoteWrites = false;
    size_t maxDestinations = 0;
    for (const Edge& e : localCopies) {
      if (e.sources.empty() || !std::ranges::all_of(e.sources, &Node::filled)) {
        nowaits = false;
      }
      for (const Node& n : e.destinations) {
        if (n.rank != rank) {
          hasRemoteWrites = true;
        }
      }
      maxDestinations = std::max(maxDestinations, e.destinations.size());
    }

    // Whether another edge reads one of e's destination nodes (e.g. the next rank of a ring), and so waits for
    // e to be signaled as done.
    auto hasDependents = [&](const Edge& e) {
      for (auto& n : e.destinations) {
        if (readNodes.contains(n.id)) {
          return true;
        }
      }
      return false;
    };

    // ---- Tables ----
    // The kernel reads its edges from tables at runtime, so its code size does not depend on the number of
    // edges. All tables are in one initialized global array.
    //
    // A node address is the node's offset plus, for slot = 8 * tensorIndex | 1, this rank's tensor (a kernel
    // parameter), else the peer's tensor at byte offset slot in the exchanged address table (bit 1 marks the NVLS
    // slot of an edge; see the address table below).
    struct NodeSetup {
      uint32_t slot;
      uint32_t offsetLow;
      uint32_t offsetHigh;
    };
    // An edge record: EdgeSetup, then the node setups of the source and of the destinations, with a fixed stride
    // so that the loads of an edge do not depend on each other
    struct EdgeSetup {
      uint32_t flags;
      uint32_t signal;    // a relay edge's signal index
      uint32_t sourceDep; // depFilled index of an unfilled source node
      uint32_t bytes;
      uint32_t numDestinations;
      uint32_t signals; // signal setup index of the first depFilled slot to write when the edge is done
      uint32_t numSignals;
      uint32_t pad;
    };
    static constexpr size_t sourceSetupOffset = sizeof(EdgeSetup);
    static constexpr size_t destinationSetupsOffset = sizeof(EdgeSetup) + sizeof(NodeSetup);
    const size_t edgeStride = (destinationSetupsOffset + sizeof(NodeSetup) * maxDestinations + 15) / 16 * 16;
    // A depFilled slot on a peer (or this rank): base + itembytes * concurrencyIndex + 4 * (index + stride *
    // blockIndex)
    struct SignalSetup {
      uint64_t base;
      uint32_t itembytes;
      uint32_t index;
      uint32_t stride;
      uint32_t pad;
    };
    // Where to write the address at which a peer maps one of this rank's tensors (in the peer's address table), and
    // in NVLS ops the tensor's identity (in the peer's identity table, at the same slot)
    struct AddrSetup {
      uint64_t base;
      uint32_t itembytes;
      uint32_t paramOffset;
      uint32_t stride;
      uint32_t slot;
      uint64_t idBase;
      uint32_t idItembytes;
      uint32_t idParamOffset;
    };
    static_assert(sizeof(AddrSetup) == 40 && offsetof(AddrSetup, idBase) == 24);
    static constexpr uint32_t edgeRelay = 1;   // relay edge: waits for the CPU thread's signal, copies nothing
    static constexpr uint32_t edgeWait = 2;    // waits for its source node (relay edges: for the signal)
    static constexpr uint32_t edgeAcquire = 4; // fence.acquire.sys before the copy (source written by another agent)
    static constexpr uint32_t edgeSignal = 8;  // other edges wait for its destinations: write their depFilled slots
    static constexpr uint32_t edgeNvls = 16;   // stored with NVLS when the launch can: the NVLS slot follows the dsts
    static constexpr uint32_t edgeSourceWaits = 32; // reduce loop: sources wait for signals (PipeEdge::waits)
    static constexpr uint32_t edgeDoneSignal = 64;  // reduce loop: writes doneSignal once stored (see Edge::doneSignal)
    // A wait table word for a source that waits for a done signal of this kernel's (in cpuInBuffer), with the index
    // + 1 in the other bits (without this bit, the signal of an RDMA transfer, in rdmaSignalBuffer)
    static constexpr uint32_t waitDoneSignal = 0x80000000;

    // ---- NVLS (NVLink SHARP multicast) ----
    // An edge to every rank of the group, at the same tensor and offset on each (an all-gather's), can store each
    // chunk once, to the group's multicast mapping of that offset (see Group::Nvls), which the NVLink switches copy
    // to every GPU, instead of once per destination. One SM then reaches the GPUs' NVLink ingress, where unicast
    // stores need 12-16 (all-gather on 32 GB300: 1M 62 vs 73 us, 4M 208 vs 239, 256M 12.4 vs 12.7 ms). It needs the
    // destination tensors at the same place of one of the group's multicast objects on every rank (at the same
    // offset of every rank's pool, or of the group's communication pool), which every launch checks with the NVLS
    // identities the peers publish (see the address table below); otherwise the launch stores to each destination as
    // usual. A rank uses NVLS if every edge it executes is such an edge, every rank
    // that does stores the same number of chunks (see the step counter below), and the group has at least
    // nvlsMinRanks ranks, all in one NVLink domain: within one node (4 GPUs), multicast stores were slower from 4M
    // on (the switch also sends each chunk back to the sender, 1/4 of a GPU's ingress; 1/32 on 32 GPUs).
    //
    // Step counter: multicast stores from all ranks at full speed overload the switches' links to the GPUs (more
    // than their ingress arrives), which then deliver about 2/3 of their bandwidth (all-gather of 256M on 32 GPUs:
    // 17-19 ms instead of 12.3). So the ranks advance in step: once the stores of a chunk have completed, a rank adds
    // 1 to the op's step counter through its multicast address (one multimem.red reaches every GPU), and it stores
    // its chunk c only once its local copy of the counter shows that every rank has completed its chunks up to
    // c - nvlsWindow. This bounds the data in flight to what the links deliver, without a clock (pacing the stores
    // by time worked too, but needs the rate). The counter of each concurrency slot only grows: customOp passes its
    // value before the launch (launches times nvlsIncrement).
    static constexpr size_t nvlsMinRanks = 16;
    static constexpr uint32_t nvlsSignalLag = 1; // a chunk is signaled once the stores of the one after it are issued
    static constexpr size_t nvlsCountersStride = 64; // bytes per counter (one per concurrency slot)
    // With NVLS, a pipeline stores chunks of nvlsChunkSize through its own ring over the same stage memory (up to
    // nvlsMaxStages stages; a chunk's single store reads its stage quickly, so it reloads the stage of the chunk
    // before): per chunk, the window and the counter cost time, and 64 KiB chunks reached 674 GB/s of a GPU's
    // ingress (32 GPUs, all-gather) where 32 KiB chunks reached 654. At least 2 stages: of the 4 the unicast ring has.
    const uint32_t nvlsChunkSize = std::max(config.chunkSize, std::min(65536u, 2 * config.chunkSize));
    static constexpr uint32_t nvlsMaxStages = 4;
    static constexpr uint32_t nvlsStoreLag = 1;
    static_assert(std::tuple_size_v<decltype(KernelHandle::nvlsLaunches)> == maxConcurrency);

    // The local destination of an edge that can store with NVLS (if every rank's pool holds its destination at the
    // same offset), else nullptr
    auto nvlsLocalDestination = [&](const Edge& e) -> const Node* {
      if (e.sources.size() != 1 || !e.sources[0].filled || e.destinations.size() != group->size || hasDependents(e)) {
        return nullptr;
      }
      Vector<uint8_t> seen(group->size);
      const Node* local = nullptr;
      for (const Node& n : e.destinations) {
        const Node& first = e.destinations[0];
        if (n.rank >= group->size || seen[n.rank] || n.tensorIndex != first.tensorIndex || n.offset != first.offset ||
            n.device != DeviceType::CUDA) {
          return nullptr;
        }
        seen[n.rank] = 1;
        if (n.rank == rank) {
          local = &n;
        }
      }
      return local;
    };

    // Whether the op uses NVLS (decided by all ranks together; the same on every rank), and if so, the chunks this
    // rank stores with it per launch (0 if none), the ranks that store any (all the same number), the window and
    // the offset of the op's counters in the group's counter area
    bool nvlsOp = false;
    uint32_t nvlsChunks = 0;  // stored by this rank
    uint32_t nvlsSenders = 0; // the ranks that store any
    uint32_t nvlsWindow = 0;
    size_t nvlsCountersOffset = 0;
    if (!tuning) {
      Group* g = ctx.group;
      // (reductions: not yet, see the TODO in docs/internals/compile_op.md)
      const bool capable = !ctx.collectiveOp && !compile_op::isReduction(graph.reduce) && g->size >= nvlsMinRanks &&
                           g->supportsMulticast && g->fabricDomainRanks.size() == g->size && g->nvls.state != -1 &&
                           allocatorGetReservedBase() != 0 && ranks.size() == g->size;
      uint64_t chunks = 0;
      bool all = capable && nowaits && !localCopies.empty();
      for (const Edge& e : localCopies) {
        all &= nvlsLocalDestination(e) != nullptr;
        chunks += (e.bytes + nvlsChunkSize - 1) / nvlsChunkSize;
      }
      uint32_t myChunks = all && chunks < (1u << 30) ? (uint32_t)chunks : 0;
      for (uint32_t r : ranks) {
        ctx.send(r, capable, myChunks);
      }
      bool allCapable = true;
      bool uniform = true;
      uint32_t sendersChunks = 0;
      for (uint32_t r : ranks) {
        bool c;
        uint32_t n;
        ctx.receive(r, c, n);
        allCapable &= c;
        if (n) {
          uniform &= nvlsSenders == 0 || n == sendersChunks;
          sendersChunks = n;
          ++nvlsSenders;
        }
      }
      // A rank that does not have every rank among its peers says so (capable is false): every rank that does
      // hears it (the peers of a rank have it among theirs), so they all agree. setupNvls is collective over the
      // group, which is then all of ranks.
      nvlsOp = allCapable && uniform && nvlsSenders > 0 && g->setupNvls();
      if (nvlsOp && g->nvls.nextCounters + nvlsCountersStride * maxConcurrency > Group::Nvls::countersBytes) {
        if (rank == 0) {
          log.error("moodist: compile_op: no NVLS step counters left for more ops; this op does not use NVLS\n");
        }
        nvlsOp = false;
      }
      if (nvlsOp) {
        nvlsCountersOffset = g->nvls.nextCounters;
        g->nvls.nextCounters += nvlsCountersStride * maxConcurrency;
        nvlsChunks = myChunks;
        // A pipeline holds up to nvlsSignalLag stored but unsignaled chunks while it waits for the counter, so the
        // window must be larger than all of them (8: see the results above)
        nvlsWindow = std::max<uint32_t>(8, config.pipelines * nvlsSignalLag + 1);
        rv->nvls = true;
        rv->nvlsIncrement = nvlsSenders * sendersChunks;
        if (rank == 0) {
          log.info("compile_op: NVLS: %d ranks store %d chunks of %d bytes each with multicast stores\n", nvlsSenders,
              sendersChunks, nvlsChunkSize);
        }
      }
    }
    // Whether this rank's kernel stores with NVLS
    const bool nvlsLocal = nvlsChunks != 0;
    // In NVLS ops, every rank publishes, besides the address at which a peer maps one of its tensors (the address
    // exchange below), where the tensor is in the group's multicast objects (its "NVLS identity", see nvlsIdentity
    // in processgroup.cc), into a table of the peer's with the layout of the address table. customOp passes the
    // identities of the rank's tensors and their multicast addresses in a fourth kernel parameter (or in the parameter
    // buffer): per tensor, 8 bytes of identity (0 if none), then 8 bytes of multicast address (0 if none).
    std::optional<KernelMappedBuffers> nvlsIds;
    if (nvlsOp) {
      nvlsIds.emplace(map("nvlsids", sizeof(uint64_t) * addrAddrsBlockStride * numBlocks));
      if (!paramBuffer) {
        fn->addParamBytes(8, 16 * numTensorWords);
      }
    }

    // A word of the address arrays (see paramBuffer above): array 1 (tensor addresses), 2 (the peers' mappings) or 3
    // (NVLS words), at a byte offset. The parameter buffer holds the copies of all concurrency slots, each with the
    // three arrays in this order.
    u64 paramBufferAddr;
    if (paramBuffer) {
      rv->paramBuffer = true;
      rv->paramMappingsOffset = 8 * numTensorWords;
      rv->paramNvlsOffset = rv->paramMappingsOffset + 8 * numMappingWords;
      rv->paramBufferStride = rv->paramNvlsOffset + (nvlsOp ? 16 * numTensorWords : 0);
      CHECK((size_t)rv->paramBufferStride * maxConcurrency < (1ull << 31));
      paramBufferAddr = addGlobalVar(16, rv->paramBufferStride * maxConcurrency, "params") +
                        widen(concurrencyIndex * rv->paramBufferStride);
    }
    auto ldArrayWord = [&](int array, const u32& offset) -> u64 {
      CHECK(array >= 1 && array <= 3);
      if (paramBuffer) {
        const uint32_t base = array == 1 ? 0 : array == 2 ? rv->paramMappingsOffset : rv->paramNvlsOffset;
        return ld_global_nc_u64(paramBufferAddr + widen(offset + base));
      }
      return ld_param_u64(paramBase(array) + offset, 0);
    };

    auto nodeSetup = [&](const Node& n) {
      NodeSetup s;
      s.offsetLow = n.offset;
      s.offsetHigh = n.offset >> 32;
      if (n.rank == rank) {
        s.slot = sizeof(uint64_t) * n.tensorIndex | 1;
      } else {
        s.slot = sizeof(uint64_t) * findLocalMapping(n);
      }
      return s;
    };
    auto localNodeIndex = [&](uint64_t id) {
      auto it = std::ranges::find(localNodes, id);
      CHECK(it != localNodes.end());
      return (uint32_t)(it - localNodes.begin());
    };
    // The depFilled slots to write once e's destination nodes are filled (per block): on this rank, on the
    // destination ranks, and on the destination ranks of edges that read e's destinations.
    auto appendSignals = [&](const Edge& e, std::vector<SignalSetup>& out) {
      HashMap<uint32_t, bool> signalRanks;
      signalRanks[rank] = true;
      for (const Node& n : e.destinations) {
        signalRanks[n.rank] = true;
      }
      for (auto& ne : graph.cudaEdges) {
        bool concerned = false;
        for (auto& n : ne.sources) {
          for (auto& n2 : e.destinations) {
            if (n.id == n2.id) {
              concerned = true;
            }
          }
        }
        if (concerned) {
          for (auto& n : ne.destinations) {
            signalRanks[n.rank] = true;
          }
        }
      }
      for (auto& v : signalRanks) {
        uint32_t r = v.first;
        for (const Node& n : e.destinations) {
          auto it = dependencyIndices[r].find(n.id);
          if (it != dependencyIndices[r].end()) {
            const auto& peer = depFilled.remote.at(rankIndex(r));
            SignalSetup s = {};
            s.base = peer.base;
            s.itembytes = peer.itembytes;
            s.index = it->second.index;
            s.stride = it->second.stride;
            out.push_back(s);
          }
        }
      }
    };

    std::vector<uint8_t> edgeTable(edgeStride * localCopies.size());
    std::vector<SignalSetup> signalTable;
    for (size_t i : indices(localCopies)) {
      const Edge& e = localCopies[i];
      uint8_t* record = edgeTable.data() + edgeStride * i;
      EdgeSetup s = {};
      CHECK(e.bytes < (1ull << 32));
      s.bytes = e.bytes;
      if (e.sources.empty()) {
        s.flags = edgeRelay | edgeWait;
        s.signal = e.rdmaSignal;
      } else {
        // (the record holds one source, for the path with waits; reductions only run without waits, from the
        // PipeEdge table below, which holds all their sources)
        const Node& src = e.sources[0];
        NodeSetup source = nodeSetup(src);
        std::memcpy(record + sourceSetupOffset, &source, sizeof(source));
        if (!src.filled) {
          CHECK(e.sources.size() == 1);
          s.flags |= edgeWait | edgeAcquire;
          s.sourceDep = localNodeIndex(src.id);
        }
        for (const Node& n : e.sources) {
          if (n.rank != rank) {
            s.flags |= edgeAcquire;
          }
        }
      }
      s.numDestinations = e.destinations.size();
      for (size_t d : indices(e.destinations)) {
        NodeSetup n = nodeSetup(e.destinations[d]);
        std::memcpy(record + destinationSetupsOffset + sizeof(NodeSetup) * d, &n, sizeof(n));
      }
      if (hasDependents(e)) {
        s.flags |= edgeSignal;
        s.signals = signalTable.size();
        appendSignals(e, signalTable);
        s.numSignals = signalTable.size() - s.signals;
      }
      std::memcpy(record, &s, sizeof(s));
    }

    std::vector<AddrSetup> addrTable;
    for (size_t i : indices(graph.remoteCudaTensorMappings)) {
      auto& v = graph.remoteCudaTensorMappings[i];
      auto& r = addrAddrs.remote[rankIndex(v.rank)];

      AddrSetup s = {};
      s.base = r.base;
      s.itembytes = r.itembytes;
      s.paramOffset = sizeof(uint64_t) * i;
      s.stride = sizeof(uint64_t) * v.stride;
      s.slot = sizeof(uint64_t) * v.destinationSlot;
      if (nvlsIds) {
        auto& ir = nvlsIds->remote[rankIndex(v.rank)];
        s.idBase = ir.base;
        s.idItembytes = ir.itembytes;
        s.idParamOffset = 16 * v.tensorIndex;
      }
      addrTable.push_back(s);
    }

    // Without waits, the items of all edges form one sequence, which the pipelines of all blocks take items from
    // (see the copy loop below). An item is a chunk of an edge and a group of its destinations: an edge's
    // destinations are split into groups when the op has few chunks, so that the stores of an all-gather of a few
    // chunks to many ranks spread over the SMs too (a load per group: more local reads, which are cheap). Per edge:
    // the index of its first item in the sequence, the destination groups (item i of the edge covers chunk
    // i / groups and destination group i % groups), and the slots of its addresses in the address table that every
    // block resolves at the start (its sources (one, except in reductions), then its destinations, then with NVLS the
    // NVLS slot: the multicast address of the local destination, or 0 if the launch stores to each destination). A
    // last record (firstItem = the number of items) ends the sequence. With NVLS, the chunks of all edges also form one
    // sequence (nvlsFirst: the index of an edge's first chunk), which a launch that uses NVLS stores instead. The
    // reduce loop takes whole chunks (one destination group).
    const uint32_t chunkSize = config.chunkSize;
    const uint32_t numPipelines = config.pipelines;
    const uint32_t numClaimers = numBlocks * numPipelines;
    struct PipeEdge {
      uint32_t firstItem;
      uint32_t numItems;
      uint32_t bytes;
      uint32_t addrSlot;
      uint32_t numDestinations;
      uint32_t groups;
      uint32_t groupSize; // destinations per group (the last group may have fewer)
      uint32_t flags;
      uint32_t signals;
      uint32_t numSignals;
      uint32_t nvlsFirst;
      uint32_t numSources;
      // Reductions: the sources that an average divides after combining them (the first ones; the others are
      // partial results, see Edge::partialSources) and the divisor; with edgeSourceWaits, the index of the first
      // source's word in the wait table (per source, its signal index + 1, or waitDoneSignal | its done signal + 1,
      // or 0 if it does not wait); with edgeDoneSignal, the cpuIn slot to signal (Edge::doneSignal)
      uint32_t inputSources;
      uint32_t divisor;
      uint32_t waits;
      uint32_t doneSignal;
    };
    static_assert(sizeof(PipeEdge) == 64 && offsetof(PipeEdge, nvlsFirst) == 40 &&
                  offsetof(PipeEdge, numSources) == 44 && offsetof(PipeEdge, doneSignal) == 60);
    std::vector<PipeEdge> pipeEdges;
    std::vector<NodeSetup> pipeAddrSetups;
    std::vector<uint32_t> pipeWaits;
    uint32_t pipeNumItems = 0;
    // The NVLS edges' address slots and destination counts (for the check of every launch, see below)
    std::vector<std::array<uint32_t, 2>> nvlsEdgeTable;
    uint32_t nvlsChunksTotal = 0;
    CHECK(chunkSize >= 16 && chunkSize % 16 == 0);
    CHECK(numPipelines >= 1 && numPipelines * 32 <= blockSize);
    if (nowaits) {
      uint64_t numChunksTotal = 0;
      for (const Edge& e : localCopies) {
        numChunksTotal += (e.bytes + chunkSize - 1) / chunkSize;
      }
      // Destination groups for about 4 items per pipeline: items are taken whole (in batches), so with few items,
      // some pipelines get one more than others, e.g. with 32 chunks to 31 destinations each (an all-gather of
      // 1 MB to 32 GPUs) and 12 pipelines. More groups cost more loads (and were slower at 8 items per pipeline).
      const uint64_t groupsWanted =
          std::max<uint64_t>((4 * numClaimers + numChunksTotal - 1) / std::max<uint64_t>(numChunksTotal, 1), 1);
      for (size_t i : indices(localCopies)) {
        const Edge& e = localCopies[i];
        EdgeSetup s;
        std::memcpy(&s, edgeTable.data() + edgeStride * i, sizeof(s));
        CHECK(e.bytes > 0 && !e.destinations.empty());
        const uint32_t numChunks = (e.bytes + chunkSize - 1) / chunkSize;
        const uint32_t numDestinations = e.destinations.size();
        PipeEdge p = {};
        const uint64_t groups = reduceKernel ? 1 : std::min<uint64_t>(groupsWanted, numDestinations);
        p.groupSize = (numDestinations + groups - 1) / groups;
        p.groups = (numDestinations + p.groupSize - 1) / p.groupSize;
        const Node* nvlsDestination = nvlsLocal ? nvlsLocalDestination(e) : nullptr;
        CHECK(nvlsDestination || !nvlsLocal);
        p.firstItem = pipeNumItems;
        p.numItems = numChunks * p.groups;
        p.bytes = e.bytes;
        p.addrSlot = pipeAddrSetups.size();
        p.numDestinations = numDestinations;
        p.flags = s.flags | (nvlsDestination ? edgeNvls : 0);
        p.signals = s.signals;
        p.numSignals = s.numSignals;
        p.nvlsFirst = nvlsChunksTotal;
        p.numSources = e.sources.size();
        CHECK(e.partialSources <= e.sources.size());
        p.inputSources = e.sources.size() - e.partialSources;
        p.divisor = e.divisor != 0 ? e.divisor : e.sources.size();
        // (the tuning launches run without the CPU thread: they do not wait for its RDMA reads, but for the done
        // signals of their own edges, which they write as the op's launches do; tuneKernels resets cpuInBuffer before
        // and after them)
        if (!e.sourceDoneSignals.empty() || (!e.sourceSignals.empty() && !tuning)) {
          CHECK(e.sourceSignals.empty() || e.sourceSignals.size() == e.sources.size());
          CHECK(e.sourceDoneSignals.empty() || e.sourceDoneSignals.size() == e.sources.size());
          p.flags |= edgeSourceWaits;
          p.waits = pipeWaits.size();
          for (size_t s : indices(e.sources)) {
            const int32_t signal = e.sourceSignals.empty() || tuning ? -1 : e.sourceSignals[s];
            const int32_t doneSignal = e.sourceDoneSignals.empty() ? -1 : e.sourceDoneSignals[s];
            CHECK(signal < 0 || doneSignal < 0);
            CHECK((uint32_t)doneSignal + 1 < waitDoneSignal);
            pipeWaits.push_back(doneSignal >= 0 ? waitDoneSignal | (doneSignal + 1) : signal + 1);
          }
        }
        if (e.doneSignal >= 0) {
          p.flags |= edgeDoneSignal;
          p.doneSignal = e.doneSignal;
        }
        for (const Node& n : e.sources) {
          pipeAddrSetups.push_back(nodeSetup(n));
        }
        for (const Node& n : e.destinations) {
          pipeAddrSetups.push_back(nodeSetup(n));
        }
        if (nvlsDestination) {
          // The NVLS slot: the local destination's setup, marked by bit 1 (see the address table below)
          NodeSetup m = nodeSetup(*nvlsDestination);
          m.slot |= 2;
          pipeAddrSetups.push_back(m);
          nvlsEdgeTable.push_back({p.addrSlot, numDestinations});
          nvlsChunksTotal += (e.bytes + nvlsChunkSize - 1) / nvlsChunkSize;
        }
        pipeEdges.push_back(p);
        CHECK((uint64_t)pipeNumItems + p.numItems < (1ull << 30));
        pipeNumItems += p.numItems;
      }
      PipeEdge end = {};
      end.firstItem = pipeNumItems;
      end.nvlsFirst = nvlsChunksTotal;
      pipeEdges.push_back(end);
      CHECK(nvlsChunksTotal == nvlsChunks);
    }

    std::vector<uint8_t> tableData;
    auto appendTable = [&](const auto& table) {
      size_t offset = (tableData.size() + 7) / 8 * 8;
      size_t bytes = sizeof(table[0]) * table.size();
      tableData.resize(offset + bytes);
      if (bytes) {
        std::memcpy(tableData.data() + offset, table.data(), bytes);
      }
      return offset;
    };
    size_t edgeTableOffset = appendTable(edgeTable);
    size_t signalTableOffset = appendTable(signalTable);
    size_t addrTableOffset = appendTable(addrTable);
    size_t pipeEdgeTableOffset = appendTable(pipeEdges);
    size_t pipeAddrSetupOffset = appendTable(pipeAddrSetups);
    size_t nvlsEdgeTableOffset = appendTable(nvlsEdgeTable);
    size_t pipeWaitsOffset = appendTable(pipeWaits);
    u64 tables = addGlobalData(16, tableData.data(), tableData.size(), "tables");

    // Shared memory: with waits, the scan for ready edges keeps a done bit per edge and the selected edge, and the
    // pipelines need the destination addresses of the edge being copied; without waits, the address table (in
    // global memory if it does not fit in pipeTableSharedLimit). The stage rings take the rest.
    const uint32_t numEdges = localCopies.size();
    const uint32_t numDoneWords = (numEdges + 31) / 32;
    const uint32_t scanShared = nowaits ? 0 : (4 * numDoneWords + 8 + 15) / 16 * 16;
    static constexpr size_t pipeTableSharedLimit = 32 * 1024;
    const size_t pipeTableBytes = 8 * pipeAddrSetups.size();
    bool pipeTableInShared = false;
    uint32_t numStages = 0;
    // The reduce loop (see below): stages of chunkSize with an mbarrier each (no records), reduceResults result
    // buffers, and a ring of reduceItemSlots claimed items (8 bytes each)
    uint32_t reduceItemSlots = 0;
    if (reduceKernel) {
      CHECK(nowaits);
      // (tuneKernels checks the configurations of reductions on every rank)
      std::string error = reduceConfigError(config);
      REQUIRE(error.empty(), "compile_op: %s", error);
      const size_t perStage = reduceStageBytes(config);
      size_t fixed = reduceFixedBytes(config);
      if (pipeTableBytes <= pipeTableSharedLimit && config.sharedMemory >= fixed + pipeTableBytes + 2 * perStage) {
        pipeTableInShared = true;
        fixed += pipeTableBytes;
      }
      numStages = std::min<size_t>((config.sharedMemory - fixed) / perStage, reduceMaxStages);
      reduceItemSlots = numStages + 2;
      mod.topLevelComments.push_back(fmt::sprintf("reduce loop: chunk size %d, %d stages%s", chunkSize, numStages,
          pipeTableInShared ? "" : ", address table in global memory"));
    } else {
      const size_t perStage = chunkSize + 8 + PipeRing::recordSize;
      // 256: alignment padding; with NVLS, the check's flag and the NVLS rings' barriers and records
      size_t fixed = 256 + scanShared + (nowaits ? 0 : 8 * maxDestinations) +
                     (nvlsLocal ? 32 + numPipelines * nvlsMaxStages * (8 + PipeRing::recordSize) : 0);
      if (nowaits && pipeTableBytes <= pipeTableSharedLimit &&
          config.sharedMemory >= fixed + pipeTableBytes + 4 * numPipelines * perStage) {
        pipeTableInShared = true;
        fixed += pipeTableBytes;
      }
      REQUIRE(config.sharedMemory >= fixed + 4 * numPipelines * perStage,
          "compile_op: %d bytes of shared memory are not enough for 4 stages of %d bytes per pipeline",
          config.sharedMemory, chunkSize);
      numStages = std::min<size_t>((config.sharedMemory - fixed) / (numPipelines * perStage), 16);
      mod.topLevelComments.push_back(fmt::sprintf("chunk size %d, %d pipelines of %d stages per block%s", chunkSize,
          numPipelines, numStages, nowaits && !pipeTableInShared ? ", address table in global memory" : ""));
    }

    // Each pipeline's stage ring (the warps beyond the pipelines alias the first ring, and never use it); with the
    // reduce loop, its stages, barriers, result buffers and item ring
    std::optional<PipeRing> ring;
    Value isPipelineLane = (laneIndex == 0) & (warpIndex < numPipelines);
    u32 reduceStages;
    u32 reduceBarriers;
    u32 reduceResultBuffers;
    u32 reduceItemRing;
    if (reduceKernel) {
      reduceStages = addShared32(128, numStages * chunkSize, "reducestages");
      reduceResultBuffers = addShared32(128, reduceResults * chunkSize, "reduceresults");
      reduceBarriers = addShared32(8, numStages * 8, "reducebarriers");
      reduceItemRing = addShared32(8, reduceItemSlots * 8, "reduceitems");
      IF_D(isThreadZero) {
        for (uint32_t s : range(numStages)) {
          mbarrier_init(reduceBarriers + 8 * s, 1);
        }
        fence_mbarrier_init();
      }
    } else {
      u32 buffers = addShared32(128, numPipelines * numStages * chunkSize, "stages");
      u32 barriers = addShared32(8, numPipelines * numStages * 8, "stagebarriers");
      u32 records = addShared32(16, numPipelines * numStages * PipeRing::recordSize, "stagerecords");
      u32 w = selp_u32(warpIndex, 0, warpIndex < numPipelines);
      ring.emplace(chunkSize, numStages, buffers + w * (numStages * chunkSize), barriers + w * (numStages * 8),
          records + w * (numStages * PipeRing::recordSize));
      IF_D(isPipelineLane) {
        ring->init();
      }
    }
    // With NVLS, each pipeline's ring of nvlsChunkSize stages over the same stage memory (a launch uses one of them)
    std::optional<PipeRing> nvlsRing;
    uint32_t nvlsStages = 0;
    if (nvlsLocal) {
      nvlsStages = std::min<uint32_t>(numStages * chunkSize / nvlsChunkSize, nvlsMaxStages);
      CHECK(nvlsStages >= 2);
      u32 barriers = addShared32(8, numPipelines * nvlsStages * 8, "nvlsbarriers");
      u32 records = addShared32(16, numPipelines * nvlsStages * PipeRing::recordSize, "nvlsrecords");
      u32 w = selp_u32(warpIndex, 0, warpIndex < numPipelines);
      nvlsRing.emplace(nvlsChunkSize, nvlsStages, ring->buffers, barriers + w * (nvlsStages * 8),
          records + w * (nvlsStages * PipeRing::recordSize));
      IF_D(isPipelineLane) {
        nvlsRing->init();
      }
    }

    struct SyncSetup {
      uint64_t stbase;
      uint32_t itembytes;
      uint32_t ldoffset;
      uint32_t stBlockStride; // bytes per block row in the peer's sync buffer
    };

    std::vector<SyncSetup> syncsetup;
    for (size_t i : indices(ranks)) {
      if (ranks[i] == rank) {
        continue;
      }
      auto& r = syncAddrs.remote[i];
      SyncSetup s;
      s.stbase = r.base + sizeof(uint32_t) * myPeerIndex[i];
      s.itembytes = r.itembytes;
      s.ldoffset = sizeof(uint32_t) * i;
      s.stBlockStride = sizeof(uint32_t) * peerRanksSize[i];
      syncsetup.push_back(s);
    }

    u32 csyncaddr = addConst32(syncsetup, "syncsetup");

    profiling.enter("syncsetup");

    // Address exchange: write the addresses at which the peers map this rank's tensors into their address tables (in
    // NVLS ops, and the tensors' identities into their identity tables)
    for (uint32_t i : range((graph.remoteCudaTensorMappings.size() + blockSize - 1) / blockSize)) {
      u32 index = blockSize * i + threadIndex;
      auto pred = index < graph.remoteCudaTensorMappings.size();
      u64 caddr = tables + addrTableOffset + widen(index * sizeof(AddrSetup));
      u64 base = predicated(pred, ld_global_nc_u64)(caddr);
      u32 itembytes = predicated(pred, ld_global_nc_u32)(caddr + 8);
      u32 paramOffset = predicated(pred, ld_global_nc_u32)(caddr + 12);
      u32 stride = predicated(pred, ld_global_nc_u32)(caddr + 16);
      u32 slot = predicated(pred, ld_global_nc_u32)(caddr + 20);
      u64 addr = base + widen(itembytes * concurrencyIndex + slot + stride * blockIndex);
      predicated(pred, st_global_relaxed_sys_u64)(addr, predicated(pred, [&](const u32& o) {
        return ldArrayWord(2, o);
      })(paramOffset) | parityTag);
      if (nvlsIds) {
        u64 idBase = predicated(pred, ld_global_nc_u64)(caddr + 24);
        u32 idItembytes = predicated(pred, ld_global_nc_u32)(caddr + 32);
        u32 idParamOffset = predicated(pred, ld_global_nc_u32)(caddr + 36);
        u64 idAddr = idBase + widen(idItembytes * concurrencyIndex + slot + stride * blockIndex);
        predicated(pred, st_global_relaxed_sys_u64)(idAddr, predicated(pred, [&](const u32& o) {
          return ldArrayWord(3, o);
        })(idParamOffset) | parityTag);
      }
    }

    profiling.leave();

    profiling.enter("syncsetup warp_sync");
    warp_sync();
    profiling.leave();

    u64 addrAddrsLocalBase = concurrency(addrAddrs.local) + widen(sizeof(uint64_t) * addrAddrsBlockStride * blockIndex);
    u64 edgeTableAddr = tables + edgeTableOffset;
    u64 signalTableAddr = tables + signalTableOffset;

    // The address of a node from its setup words (the same for all threads)
    auto nodeaddr = [&](const u32& slot, const u32& offsetlo, const u32& offsethi) {
      u64 r;
      IF_D((slot & 1) != 0) {
        r = ldArrayWord(1, slot & ~3u);
      }
      ELSE {
        u64 gaddr = addrAddrsLocalBase + widen(slot);
        r = ld_global_relaxed_sys_u64(gaddr);
        WHILE((r & (1ull << 63)) != parityTag) {
          nanosleep(10);
          r = ld_global_relaxed_sys_u64(gaddr);
        }
        r &= ~(1ull << 63);
      }
      r += widen(offsetlo) | (widen(offsethi) << 32);
      return r;
    };

    // The address of a node from its setup words, for the threads where pred is true (0 for the others)
    auto nodeaddrpred = [&](Value pred, const u32& slot, const u32& offsetlo, const u32& offsethi) {
      profiling.enter("nodeaddr");
      u64 r;
      Value done = !pred;
      Value isParam = (slot & 1) != 0;
      r = predicated(!done & isParam, [&](const u32& o) {
        return ldArrayWord(1, o);
      })(slot & ~3u);
      done |= isParam;
      u64 gaddr = addrAddrsLocalBase + widen(slot);
      r = selp(r, predicated(!done, ld_global_relaxed_sys_u64)(gaddr), done);
      done |= (r & (1ull << 63)) == parityTag;
      profiling.enter("nodeaddr loop");
      WHILE(vote_sync_any(!done)) {
        nanosleep(10);
        fence_acquire_sys();
        r = selp(r, predicated(!done, ld_global_relaxed_sys_u64)(gaddr), done);
        done |= (r & (1ull << 63)) == parityTag;
      }
      profiling.enter("nodeaddr post");
      r &= ~(1ull << 63);

      r += widen(offsetlo) | (widen(offsethi) << 32);
      profiling.leave();
      return selp_u64(r, 0, pred);
    };

    // With waits, the threads resolve the destination addresses of an edge: slot i of a lane takes destination
    // gindex(i) (the same in all warps)
    const size_t numDstSlots = (maxDestinations + 31) / 32;
    auto gindex = [&](size_t i) {
      return laneIndex + 32 * i;
    };

    // The words of an edge record this thread needs: the header fields, the source's setup, and the setups of the
    // destinations of its slots
    struct EdgeWords {
      u32 flags;
      u32 bytes;
      u32 numDestinations;
      std::array<u32, 3> source;
      Vector<std::array<u32, 3>> destinations;
    };
    // Load the words of the edge at desc into w's registers (if pred, when given). Nothing uses them yet, so the
    // sequential path can have the loads of the next edge in flight while it copies the current one.
    auto loadEdge = [&](EdgeWords& w, const u64& desc, const Value* pred) {
      w.destinations.resize(numDstSlots);
      auto load = [&](const Value& d, const u64& addr) {
        if (pred) {
          PRED(*pred) ld_global_nc_u32_into(d, addr);
        } else {
          ld_global_nc_u32_into(d, addr);
        }
      };
      load(w.flags, desc);
      load(w.bytes, desc + 12);
      load(w.numDestinations, desc + 16);
      for (size_t j : range(3)) {
        load(w.source[j], desc + sourceSetupOffset + 4 * j);
      }
      for (size_t i : range(numDstSlots)) {
        u32 index = gindex(i);
        u64 caddr = desc + destinationSetupsOffset + widen(index * sizeof(NodeSetup));
        // Slots beyond the record's destinations stay unloaded
        Value valid = index < maxDestinations;
        Value p = pred ? valid & *pred : valid;
        for (size_t j : range(3)) {
          PRED(p) ld_global_nc_u32_into(w.destinations[i][j], caddr + 4 * j);
        }
      }
    };

    // Signal (this block's slots) that the destination nodes of the edge at `desc` are filled: a release fence,
    // then the depFilled stores. For the thread that has seen the edge done (thread 0 after a copy by the whole
    // block, or the lane that has seen a relay edge's signal); several lanes may run it at once, hence the
    // divergent loop.
    auto signalEdge = [&](const u64& desc) {
      u32 i = ld_global_nc_u32(desc + 20);
      u32 end = i + ld_global_nc_u32(desc + 24);
      fence_release_sys();
      Label loop;
      Label done;
      LABEL(loop);
      GOTO_IF_NOT_D(i < end, done);
      u64 s = signalTableAddr + widen(i * sizeof(SignalSetup));
      u64 base = ld_global_nc_u64(s);
      u32 itembytes = ld_global_nc_u32(s + 8);
      u32 index = ld_global_nc_u32(s + 12);
      u32 stride = ld_global_nc_u32(s + 16);
      st_global_relaxed_sys_u32(
          base + widen(itembytes * concurrencyIndex + 4 * (index + stride * blockIndex)), stepValue);
      i += 1;
      GOTO(loop);
      LABEL(done);
    };

    Vector<u64> dstaddrs(numDstSlots);
    // Resolve the addresses of an edge from its loaded words: returns the source address, sets dstaddrs
    auto resolveEdge = [&](const EdgeWords& w) {
      u64 srcaddr = nodeaddr(w.source[0], w.source[1], w.source[2]);
      for (size_t i : range(numDstSlots)) {
        auto& d = w.destinations[i];
        dstaddrs[i] = nodeaddrpred(gindex(i) < w.numDestinations, d[0], d[1], d[2]);
      }
      return srcaddr;
    };
    // After storing chunk k, a pipeline loads the next chunk into the stage of chunk k - storeLag, once the stores
    // of that chunk have read it: up to storeLag + 1 chunks' stores in flight (the stores of an all-gather chunk to
    // many GPUs take long), and numStages - storeLag loads
    static constexpr int storeLag = 2;
    CHECK(reduceKernel || storeLag < (int)numStages); // (the reduce loop does not lag, and may have 2 stages)

    // With waits: the destination addresses of the edge being copied, for the pipelines
    std::optional<u32> pipeDestinations;
    if (!nowaits) {
      pipeDestinations = addShared32(8, 8 * std::max(maxDestinations, (size_t)1), "destinations");
    }
    // With waits: copy this block's chunks of an edge, pipelined within the edge. Chunk c of an edge belongs to
    // block c % numBlocks, the same split on every rank, which the per-block depFilled slots rely on; the pipelines
    // of a block take its chunks in turn. All threads call this; it returns when this block's stores are done.
    auto pipeCopyEdge = [&](const u64& srcaddr, const u32& numDestinations, const u32& bytes) {
      IF(warpIndex == 0) {
        for (size_t i : range(numDstSlots)) {
          u32 index = gindex(i);
          PRED(index < numDestinations) st_shared_u64(*pipeDestinations + index * 8, dstaddrs[i]);
        }
      }
      barrier_sync();
      IF_D(isPipelineLane) {
        // The loads below may read data whose readiness generic loads have observed
        fence_proxy_async_global();
        u32 numChunks = (bytes - 1) / chunkSize + 1;
        u32 chunk = blockIndex + warpIndex * numBlocks;
        u64 destinations = cvta_shared(widen(*pipeDestinations));
        u32 loaded = 0;
        u32 stored = 0;
        auto loadChunk = [&]() {
          u64 offset = widen(chunk) * chunkSize;
          u32 n = min_u32(bytes - narrow(offset), chunkSize);
          ring->load(srcaddr, offset, n, destinations, numDestinations, 0, 0);
          chunk += numBlocks * numPipelines;
          loaded += 1;
        };
        WHILE((chunk < numChunks) & (loaded < numStages)) {
          loadChunk();
        }
        WHILE(stored < loaded) {
          ring->store();
          stored += 1;
          // The stage of the chunk before is free once its stores have read it
          IF((chunk < numChunks) & (stored >= storeLag + 1)) {
            cp_async_bulk_wait_group_read(storeLag);
            loadChunk();
          }
        }
        cp_async_bulk_wait_group(0);
      }
      barrier_sync();
    };

    // Copy the edge at `desc` (the same for all threads) with the whole block: flags, bytes and srcaddr taken
    // from its words (copied into registers of their own, so the words can be reloaded), dstaddrs resolved
    auto copyEdge = [&](const u64& desc, const u32& flags, const u32& bytes, const u64& srcaddr,
                        const u32& numDestinations) {
      // Every thread has read the source's address itself (remote sources), and any may issue loads
      IF((flags & edgeAcquire) != 0) {
        fence_acquire_sys();
      }
      pipeCopyEdge(srcaddr, numDestinations, bytes);
      IF_D(isThreadZero & ((flags & edgeSignal) != 0)) {
        signalEdge(desc);
      }
    };

    if (numEdges == 0) {
    } else if (nowaits) {
      // The pipelines of all blocks take the items of the sequence (see PipeEdge) from a counter: the first claim
      // of a pipeline takes enough to fill its ring (if there are enough items for all pipelines), the later ones
      // one item (two were slower: the last pipelines to finish had more left). A pipeline claims its next item when
      // it starts on one, so that the claim has returned when it needs it. Taking items dynamically keeps all SMs
      // busy until the end: the copy rate of an SM varies (with a static split, the slowest SMs finished up to 47%
      // after the others).
      // Each concurrency slot has two counters, used by alternate launches (by the launch parity): block 0 resets
      // the other one, which the previous launch used and the next one will use.
      // (the reduce loop claims one item at a time, see below)
      const uint32_t firstBatch = reduceKernel ? 1 : std::clamp<uint32_t>(pipeNumItems / numClaimers, 1, numStages);
      static constexpr uint32_t pipeBatch = 1;
      u64 claimCounters = addGlobalVar(8, 8 * maxConcurrency, "claims");
      IF_D(isThreadZero & (blockIndex == 0)) {
        st_global_u32(claimCounters + widen(concurrencyIndex * 8 + (parityBit ^ 1) * 4), 0);
      }
      u64 claimCounter = claimCounters + widen(concurrencyIndex * 8 + parityBit * 4);
      u32 nextBase; // the first item of the pending claim
      IF_D(isPipelineLane) {
        atom_global_relaxed_gpu_add_u32_into(nextBase, claimCounter, firstBatch);
      }

      // The address table: every block resolves the addresses of all edges at the start (waiting for the peers to
      // publish theirs for this launch), so that edges cost nothing to switch between
      const uint32_t numSlots = pipeAddrSetups.size();
      u64 table;
      if (pipeTableInShared) {
        table = cvta_shared(widen(addShared32(16, 8 * numSlots, "addresses")));
      } else {
        const size_t blockBytes = 8 * numSlots;
        rv->scratch = ctx.group->allocateDevice(blockBytes * numBlocks * maxConcurrency);
        u64 base = (int64_t)rv->scratch.cudaPointer;
        table = base + widen(concurrencyIndex) * (blockBytes * numBlocks) + widen(blockIndex) * blockBytes;
      }
      // NVLS: the flag the check below sets if this launch cannot use NVLS (thread 0 clears it before the resolve
      // barrier)
      const Group::Nvls& groupNvls = ctx.group->nvls;
      u32 nvlsFailedAddr;
      if (nvlsLocal) {
        nvlsFailedAddr = addShared32(4, 4, "nvlsfailed");
        IF_D(isThreadZero) {
          st_shared_u32(nvlsFailedAddr, 0);
        }
      }
      profiling.enter("resolve addresses");
      FOR(u32 first = 0, first < numSlots, first += blockSize) {
        u32 i = first + threadIndex;
        Value valid = i < numSlots;
        u64 setup = tables + pipeAddrSetupOffset + widen(i * sizeof(NodeSetup));
        u32 slot = predicated(valid, ld_global_nc_u32)(setup);
        u32 offsetLow = predicated(valid, ld_global_nc_u32)(setup + 4);
        u32 offsetHigh = predicated(valid, ld_global_nc_u32)(setup + 8);
        u64 addr = nodeaddrpred(valid, slot, offsetLow, offsetHigh);
        if (nvlsLocal) {
          // An NVLS slot (the local destination's setup, slot = 8 * tensorIndex | 3): the destination through the
          // multicast address of its tensor, if it has one, else 0
          Value isNvls = valid & ((slot & 2) != 0);
          u64 mcTensor = predicated(isNvls, [&](const u32& o) {
            return ldArrayWord(3, o);
          })((slot & ~3u) * 2 + 8);
          u64 mcAddr = selp_u64(mcTensor + (widen(offsetLow) | (widen(offsetHigh) << 32)), 0, mcTensor != 0);
          addr = selp_u64(mcAddr, addr, isNvls);
        }
        PRED(valid) st_u64(table + widen(i * 8), addr);
      }
      bool anyRemoteSources = false;
      bool anySignals = false;
      for (const PipeEdge& p : pipeEdges) {
        anyRemoteSources |= (p.flags & edgeAcquire) != 0;
        anySignals |= (p.flags & edgeSignal) != 0;
      }
      // Copies of sources on other ranks: after the reads of their addresses (see copyEdge)
      if (anyRemoteSources) {
        fence_acquire_sys();
      }
      profiling.enter("resolve addresses barrier");
      barrier_sync();
      profiling.leave();

      // NVLS check: this launch stores with NVLS if every destination of every NVLS edge has the identity of the
      // local destination (the same tensor at the same place of the same multicast object; the node offset is the
      // same on every rank): from the identities the peers publish (with the launch's parity, like the addresses).
      // Warp w checks the edges w, w + numWarps, ...; an edge that fails gets 0 in its NVLS slot and sets the flag.
      // Every block decides the same: from the same identities.
      Value nvlsMode(ValType::Pred);
      if (nvlsLocal) {
        profiling.enter("nvls check");
        const uint32_t numWarps = blockSize / 32;
        u64 idLocalBase = concurrency(nvlsIds->local) + widen(sizeof(uint64_t) * addrAddrsBlockStride * blockIndex);
        FOR(u32 e = warpIndex, e < (uint32_t)nvlsEdgeTable.size(), e += numWarps) {
          u64 record = tables + nvlsEdgeTableOffset + widen(e * 8);
          u32 addrSlot = ld_global_nc_u32(record);
          u32 numDestinations = ld_global_nc_u32(record + 4);
          u32 nvlsSlot = addrSlot + 1 + numDestinations;
          u64 nvlsAddr = ld_u64(table + widen(nvlsSlot * 8));
          u32 localSlot = ld_global_nc_u32(tables + pipeAddrSetupOffset + widen(nvlsSlot * sizeof(NodeSetup)));
          u64 identity = ldArrayWord(3, (localSlot & ~3u) * 2);
          Value ok = (nvlsAddr != 0) & (identity != 0);
          FOR(u32 d0 = 0, d0 < numDestinations, d0 += 32) {
            u32 d = d0 + laneIndex;
            Value valid = d < numDestinations;
            u32 dslot = predicated(valid, ld_global_nc_u32)(
                tables + pipeAddrSetupOffset + widen((addrSlot + 1 + d) * sizeof(NodeSetup)));
            Value remote = valid & ((dslot & 1) == 0);
            u64 idAddr = idLocalBase + widen(dslot);
            u64 published = predicated(remote, ld_global_relaxed_sys_u64)(idAddr);
            Value done = !remote | ((published & (1ull << 63)) == parityTag);
            WHILE(vote_sync_any(!done)) {
              nanosleep(10);
              published = selp(published, predicated(!done, ld_global_relaxed_sys_u64)(idAddr), done);
              done |= (published & (1ull << 63)) == parityTag;
            }
            Value bad = remote & ((published & ~(1ull << 63)) != identity);
            ok = ok & !vote_sync_any(bad);
          }
          IF_D(!ok & (laneIndex == 0)) {
            st_u64(table + widen(nvlsSlot * 8), 0);
            st_shared_u32(nvlsFailedAddr, 1);
          }
        }
        barrier_sync();
        nvlsMode = ld_shared_u32(nvlsFailedAddr) == 0;
        // Count the launch (see Group::Nvls::launches)
        IF_D(nvlsMode & isThreadZero & (blockIndex == 0)) {
          u64 launches = (int64_t)groupNvls.launches;
          u32 old;
          atom_global_relaxed_gpu_add_u32_into(old, launches, 1);
        }
        profiling.leave();
      }

      std::optional<u64> edgeDone;
      if (anySignals) {
        edgeDone = addGlobalVar(4, 4 * maxConcurrency * numEdges, "edgedone");
      }

      // The op's step counter of this concurrency slot, through the multicast and the local mapping
      u64 nvlsCounterMc;
      u64 nvlsCounterLocal;
      if (nvlsLocal) {
        u64 slotOffset = widen(concurrencyIndex * (uint32_t)nvlsCountersStride);
        nvlsCounterMc = slotOffset + (int64_t)(groupNvls.countersMc + nvlsCountersOffset);
        nvlsCounterLocal = slotOffset + (int64_t)(groupNvls.countersLocal + nvlsCountersOffset);
      }

      // Without NVLS: the pipelines of all blocks take items (see above). With NVLS edges, a launch that does not
      // use NVLS adds the chunks to the op's step counter at the end, so that its value stays the same for every
      // launch (ranks whose launch uses NVLS may wait for them).
      auto unicastCopy = [&]() {
        IF_D(isPipelineLane) {
          profiling.enter("copy");
          // The loads below may read data whose readiness generic loads have observed
          fence_proxy_async_global();
          u64 pipeEdgeTable = tables + pipeEdgeTableOffset;
          u32 item = 0;     // the next item of the current batch
          u32 batchEnd = 0; // the end of the current batch
          u32 claimSize = firstBatch;
          Value exhausted(ValType::Pred);
          exhausted = 0;
          // The edge of the last item loaded, and the first item of the edge after it
          u32 edge = 0;
          u32 nextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge));
          u32 loaded = 0;
          u32 stored = 0;
          // With NVLS edges: the NVLS chunks (of nvlsChunkSize, see PipeEdge::nvlsFirst) of the edges whose first
          // item this pipeline loaded
          u32 nvlsLoaded;
          if (nvlsLocal) {
            nvlsLoaded = 0;
          }

          // Take the next item; false if there are none left
          auto take = [&]() {
            IF(item >= batchEnd) {
              u32 base = nextBase;
              item = base;
              batchEnd = min_u32(base + claimSize, pipeNumItems);
              IF(base < pipeNumItems) {
                claimSize = pipeBatch;
                atom_global_relaxed_gpu_add_u32_into(nextBase, claimCounter, pipeBatch);
              }
            }
            return item < batchEnd;
          };
          auto loadItem = [&]() {
            // Items only increase: advance to the item's edge
            WHILE(item >= nextFirst) {
              edge += 1;
              nextFirst = ld_global_nc_u32(pipeEdgeTable + widen((edge + 1) * sizeof(PipeEdge)));
            }
            u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
            u32 edgeFirst = ld_global_nc_u32(record);
            u32 edgeBytes = ld_global_nc_u32(record + 8);
            u32 addrSlot = ld_global_nc_u32(record + 12);
            u32 numDestinations = ld_global_nc_u32(record + 16);
            u32 groups = ld_global_nc_u32(record + 20);
            u32 groupSize = ld_global_nc_u32(record + 24);
            u32 flags = ld_global_nc_u32(record + 28);
            u64 src = ld_u64(table + widen(addrSlot * 8));
            u32 index = item - edgeFirst;
            u32 chunk = index / groups;
            u32 firstDestination = (index - chunk * groups) * groupSize;
            u64 offset = widen(chunk) * chunkSize;
            u32 n = min_u32(edgeBytes - narrow(offset), chunkSize);
            u32 numGroupDestinations = min_u32(numDestinations - firstDestination, groupSize);
            // The tag: the edge, and whether it signals (bit 31)
            u32 tag = edge | ((flags & edgeSignal) << 28);
            ring->load(
                src, offset, n, table + widen((addrSlot + 1 + firstDestination) * 8), numGroupDestinations, tag, 0);
            item += 1;
            loaded += 1;
            if (nvlsLocal) {
              IF(index == 0) {
                nvlsLoaded += ld_global_nc_u32(record + sizeof(PipeEdge) + offsetof(PipeEdge, nvlsFirst)) -
                              ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
              }
            }
          };

          // Fill the ring, then store items in order, loading the next one into the stage of the item before (free
          // once its stores have read it)
          WHILE(!exhausted & (loaded < numStages)) {
            IF(take()) {
              loadItem();
            }
            ELSE {
              exhausted = 1;
            }
          }
          WHILE(stored < loaded) {
            u32 tag = ring->store();
            stored += 1;
            if (anySignals) {
              // An item of an edge that other edges (on other ranks) wait for. Its items went to any block, while the
              // waiting kernels wait for per-block slots: the last item to finish signals the slots of all blocks.
              // Per launch, every item counts once, so the counter wraps back to 0.
              IF((tag >> 31) != 0) {
                u32 e = tag & 0x7fffffff;
                u64 record = pipeEdgeTable + widen(e * sizeof(PipeEdge));
                u32 numItems = ld_global_nc_u32(record + 4);
                cp_async_bulk_wait_group(0);
                fence_release_gpu();
                u32 old =
                    atom_global_relaxed_inc_u32(*edgeDone + widen((concurrencyIndex * numEdges + e) * 4), numItems - 1);
                IF(old == numItems - 1) {
                  fence_acquire_gpu();
                  fence_release_sys();
                  u32 i = ld_global_nc_u32(record + 32);
                  u32 end = i + ld_global_nc_u32(record + 36);
                  WHILE(i < end) {
                    u64 s = signalTableAddr + widen(i * sizeof(SignalSetup));
                    u64 base = ld_global_nc_u64(s);
                    u32 itembytes = ld_global_nc_u32(s + 8);
                    u32 index = ld_global_nc_u32(s + 12);
                    u32 stride = ld_global_nc_u32(s + 16);
                    u64 slot = base + widen(itembytes * concurrencyIndex + 4 * index);
                    FOR(u32 b = 0, b < numBlocks, b += 1) {
                      st_global_relaxed_sys_u32(slot + widen(4 * stride * b), stepValue);
                    }
                    i += 1;
                  }
                }
              }
            }
            IF(!exhausted & (stored >= storeLag + 1)) {
              cp_async_bulk_wait_group_read(storeLag);
              IF(take()) {
                loadItem();
              }
              ELSE {
                exhausted = 1;
              }
            }
          }
          profiling.enter("drain");
          cp_async_bulk_wait_group(0);
          profiling.leave();
          if (nvlsLocal) {
            IF(nvlsLoaded != 0) {
              multimem_red_relaxed_sys_add_u32(nvlsCounterMc, nvlsLoaded);
            }
          }
        }
      };

      // With NVLS: the pipelines of block 0 store the chunks of the NVLS sequence (see PipeEdge), pipeline p chunks
      // p, p + numPipelines, ..., one store each, to the NVLS slot of the chunk's edge, in step with the other ranks
      // (see the step counter above: before storing chunk c, wait until the counter is at least its value before the
      // launch plus nvlsSenders * (c + 1 - nvlsWindow); once a chunk's stores have completed, add 1 to it).
      auto nvlsCopy = [&]() {
        profiling.enter("nvls copy");
        // The loads below may read data whose readiness generic loads have observed
        fence_proxy_async_global();
        u64 pipeEdgeTable = tables + pipeEdgeTableOffset;
        u32 counterBase = loadParamField(params, 8, ValType::U32);
        u32 chunk = warpIndex; // the next chunk this pipeline loads
        // The edge of the last chunk loaded, and the first chunk of the edge after it
        u32 edge = 0;
        u32 nextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge) + offsetof(PipeEdge, nvlsFirst));
        u32 loaded = 0;
        u32 stored = 0;
        u32 signaled = 0;
        auto loadChunk = [&]() {
          // Chunks only increase: advance to the chunk's edge
          WHILE(chunk >= nextFirst) {
            edge += 1;
            nextFirst =
                ld_global_nc_u32(pipeEdgeTable + widen((edge + 1) * sizeof(PipeEdge)) + offsetof(PipeEdge, nvlsFirst));
          }
          u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
          u32 edgeBytes = ld_global_nc_u32(record + 8);
          u32 addrSlot = ld_global_nc_u32(record + 12);
          u32 numDestinations = ld_global_nc_u32(record + 16);
          u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
          u64 src = ld_u64(table + widen(addrSlot * 8));
          u64 offset = widen(chunk - first) * nvlsChunkSize;
          u32 n = min_u32(edgeBytes - narrow(offset), nvlsChunkSize);
          nvlsRing->load(src, offset, n, table + widen((addrSlot + 1 + numDestinations) * 8), 1, edge, chunk);
          chunk += numPipelines;
          loaded += 1;
        };
        WHILE((chunk < nvlsChunks) & (loaded < nvlsStages)) {
          loadChunk();
        }
        WHILE(stored < loaded) {
          // The window. A wait gives up after a second: the counter only paces the stores, and a launch must not
          // hang on it (the kernels of a launch start together: each waits for all peers' addresses)
          u32 c = nvlsRing->nextStoreAux();
          IF(c >= nvlsWindow) {
            u32 target = counterBase + (c + 1 - nvlsWindow) * nvlsSenders;
            u32 value = ld_global_relaxed_sys_u32(nvlsCounterLocal);
            IF((value - target) >= 0x80000000u) {
              profiling.enter("nvls window");
              u64 t0 = globaltimer();
              u64 waited = globaltimer() - t0;
              WHILE(((value - target) >= 0x80000000u) & (waited < 1000000000ull)) {
                value = ld_global_relaxed_sys_u32(nvlsCounterLocal);
                waited = globaltimer() - t0;
              }
              profiling.leave();
            }
          }
          nvlsRing->store();
          stored += 1;
          // Signal the chunks whose stores have completed (all but the last nvlsSignalLag). Relaxed: a release
          // would wait for all of this thread's outstanding stores (1 SM then reached 1/3 of the bandwidth).
          cp_async_bulk_wait_group(nvlsSignalLag);
          IF(stored > signaled + nvlsSignalLag) {
            multimem_red_relaxed_sys_add_u32(nvlsCounterMc, stored - nvlsSignalLag - signaled);
            signaled = stored - nvlsSignalLag;
          }
          IF((chunk < nvlsChunks) & (stored >= nvlsStoreLag + 1)) {
            cp_async_bulk_wait_group_read(nvlsStoreLag);
            loadChunk();
          }
        }
        profiling.enter("drain");
        cp_async_bulk_wait_group(0);
        profiling.leave();
        IF(stored != signaled) {
          multimem_red_relaxed_sys_add_u32(nvlsCounterMc, stored - signaled);
        }
        profiling.leave();
      };

      // The reduce loop (kernels with reductions; see reductionPieces and mergeReductions in compile_op.cc). The blocks
      // take items one at a time (an item is a chunk of an edge, with all its destinations). Thread 0 loads the chunk
      // of each source of an item in turn with bulk loads, into a ring of stages, ahead across items as far as the ring
      // allows, and all threads combine each source's chunk as it arrives into their accumulators, in the order of the
      // edge's sources (the executor's own first, then the others in rank order from there, so that the ranks read
      // from different ranks at the same time). The threads then write the result into one of two result buffers, and
      // thread 0 stores it to every destination with bulk stores (the stores of the item before may still be reading
      // the other buffer). Thread t holds the bytes at 16 * (t + v * blockSize) of a chunk, v < numVectors. The 16-bit
      // floating point types are combined in f32 and rounded (to nearest even) once, when stored. An item with one
      // source (a cell that one input covers) is stored as loaded.
      auto reduceLoop = [&]() {
        profiling.enter("reduce");
        using compile_op::ReduceOp;
        const DType dtype = graph.tensors.at(0).dtype;
        const ReduceOp op = graph.reduce;
        const uint32_t numVectors = chunkSize / (16 * blockSize);
        const bool acc64 = dtype == DType::Float64 || dtype == DType::Int64;
        const bool accFloat = dtype != DType::Int32 && dtype != DType::Int64;
        const bool half = dtype == DType::Float16 || dtype == DType::BFloat16;
        const uint32_t accPerVector = half ? 8 : acc64 ? 2 : 4;
        const bool sum = op == ReduceOp::Sum || op == ReduceOp::Avg;
        const bool isMin = op == ReduceOp::Min;
        CHECK(compile_op::isReduction(op) && (accFloat || op != ReduceOp::Avg));
        // The identity: -0 for sums (x + -0 = x for every x, -0 included), +-inf or the integer extremes for min / max
        int64_t identity;
        if (sum) {
          identity = !accFloat ? 0 : acc64 ? (int64_t)0x8000000000000000ull : 0x80000000ll;
        } else if (isMin) {
          identity = !accFloat ? (acc64 ? INT64_MAX : 0x7fffffffll) : acc64 ? 0x7ff0000000000000ll : 0x7f800000ll;
        } else {
          identity = !accFloat ? (acc64 ? INT64_MIN : 0x80000000ll)
                     : acc64   ? (int64_t)0xfff0000000000000ull
                               : 0xff800000ll;
        }
        // acc = op(acc, x); floating point min / max propagate NaN (min.NaN; there is no min.NaN.f64)
        auto combine = [&](const Value& acc, const Value& x) {
          if (sum) {
            arith(accFloat ? (acc64 ? "add.rn.f64" : "add.rn.f32") : (acc64 ? "add.s64" : "add.s32"), acc, acc, x);
          } else if (!accFloat) {
            arith(isMin ? (acc64 ? "min.s64" : "min.s32") : (acc64 ? "max.s64" : "max.s32"), acc, acc, x);
          } else if (!acc64) {
            arith(isMin ? "min.NaN.f32" : "max.NaN.f32", acc, acc, x);
          } else {
            Value nan(ValType::Pred);
            setp_nan("f64", nan, acc, x);
            Value m(ValType::U64);
            arith(isMin ? "min.f64" : "max.f64", m, acc, x);
            Value nanSum(ValType::U64);
            arith("add.rn.f64", nanSum, acc, x);
            emit("selp.b64 " + acc.str() + ", " + nanSum.str() + ", " + m.str() + ", " + nan.str());
          }
        };
        // The accumulator values of a loaded vector
        auto unpack = [&](const std::array<u32, 4>& x) {
          Vector<Value> r;
          for (size_t j : range(4)) {
            if (dtype == DType::BFloat16) {
              r.push_back(x[j] << 16);
              r.push_back(x[j] & 0xffff0000ll);
            } else if (dtype == DType::Float16) {
              Value lo(ValType::U16);
              Value hi(ValType::U16);
              unpack_b32(lo, hi, x[j]);
              r.emplace_back(ValType::U32);
              arith("cvt.f32.f16", r.back(), lo);
              r.emplace_back(ValType::U32);
              arith("cvt.f32.f16", r.back(), hi);
            } else if (acc64) {
              if (j % 2 == 0) {
                r.emplace_back(ValType::U64);
                pack_b64(r.back(), x[j], x[j + 1]);
              }
            } else {
              r.push_back(x[j]);
            }
          }
          CHECK(r.size() == accPerVector);
          return r;
        };
        // The vector to store from accumulator values (16-bit floats rounded to nearest even)
        auto pack = [&](const Vector<Value>& acc) {
          std::array<u32, 4> x;
          for (size_t j : range(4)) {
            if (half) {
              // (cvt.rn.bf16x2.f32 d, a, b puts a in the upper half)
              arith(dtype == DType::BFloat16 ? "cvt.rn.bf16x2.f32" : "cvt.rn.f16x2.f32", x[j], acc[2 * j + 1],
                  acc[2 * j]);
            } else if (acc64) {
              if (j % 2 == 0) {
                unpack_b64(x[j], x[j + 1], acc[j / 2]);
              }
            } else {
              x[j] = acc[j];
            }
          }
          return x;
        };

        u64 pipeEdgeTable = tables + pipeEdgeTableOffset;

        // Reductions across NVLink domains: sources that the CPU thread reads from other domains (not in the tuning
        // launches, which run without the CPU thread), and partial results that it sends to them, or that this kernel
        // combines itself (see crossDomainReductions in compile_op.cc); in one domain in two phases, the pieces that
        // this kernel computes into its scratch tensor and then copies to their destinations (see twoPhaseReductions
        // in compile_op.cc)
        bool anyWaits = false;
        bool anyDoneSignals = false;
        for (const PipeEdge& p : pipeEdges) {
          anyWaits |= (p.flags & edgeSourceWaits) != 0;
          anyDoneSignals |= (p.flags & edgeDoneSignal) != 0;
        }
        // Per concurrency slot and edge: the items of a partial result stored so far
        std::optional<u64> partialDone;
        if (anyDoneSignals) {
          partialDone = addGlobalVar(4, 4 * maxConcurrency * numEdges, "partialdone");
        }

        // Thread 0's loads: the stage of the next load, the item being loaded (its edge, the first item of the edge
        // after it, its next source, its number of sources, its address slot, its offset in the edge and its size,
        // and with waits, its first wait word or ~0 if none), and its slot in the item ring; ldDone once no items
        // are left; the loads issued so far
        u32 ldStage = 0;
        u32 ldEdge = 0;
        u32 ldNextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge));
        u32 ldSource = 0;
        u32 ldNumSources = 0;
        u32 ldAddrSlot = 0;
        u64 ldOffset = 0;
        u32 ldBytes = 0;
        u32 ldWaits = 0;
        u32 ldSlot = 0;
        Value ldDone(ValType::Pred);
        ldDone = 0;
        u32 ldIssued = 0;
        // The stages that the threads have combined so far (the same in all threads)
        u32 consumed = 0;
        // Take the item of the claim in flight, claim the next one, and write the item (its edge and chunk) to the
        // item ring, or past the last item, the end mark (claims of one block only increase)
        auto startItem = [&]() {
          u32 item = nextBase;
          IF(item < pipeNumItems) {
            atom_global_relaxed_gpu_add_u32_into(nextBase, claimCounter, 1);
            WHILE(item >= ldNextFirst) {
              ldEdge += 1;
              ldNextFirst = ld_global_nc_u32(pipeEdgeTable + widen((ldEdge + 1) * sizeof(PipeEdge)));
            }
            u64 record = pipeEdgeTable + widen(ldEdge * sizeof(PipeEdge));
            u32 chunk = item - ld_global_nc_u32(record + offsetof(PipeEdge, firstItem));
            ldNumSources = ld_global_nc_u32(record + offsetof(PipeEdge, numSources));
            ldAddrSlot = ld_global_nc_u32(record + offsetof(PipeEdge, addrSlot));
            ldOffset = widen(chunk) * chunkSize;
            ldBytes = min_u32(ld_global_nc_u32(record + offsetof(PipeEdge, bytes)) - narrow(ldOffset), chunkSize);
            ldSource = 0;
            if (anyWaits) {
              u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
              ldWaits = selp_u32(
                  ld_global_nc_u32(record + offsetof(PipeEdge, waits)), 0xffffffffll, (flags & edgeSourceWaits) != 0);
            }
            st_shared_u32(reduceItemRing + ldSlot * 8, ldEdge);
            st_shared_u32(reduceItemRing + ldSlot * 8 + 4, chunk);
          }
          ELSE {
            ldDone = 1;
            u32 endMark = 0xffffffffll;
            st_shared_u32(reduceItemRing + ldSlot * 8, endMark);
          }
        };
        // Load the next source chunk of the item being loaded into the next stage (which the threads have combined
        // into their accumulators), and after the item's last source, take the next item
        auto issueLoad = [&]() {
          u64 src = ld_u64(table + widen((ldAddrSlot + ldSource) * 8)) + ldOffset;
          u32 barrier = reduceBarriers + ldStage * 8;
          mbarrier_arrive_expect_tx(barrier, ldBytes);
          cp_async_bulk_shared_global(reduceStages + ldStage * chunkSize, src, ldBytes, barrier);
          u32 next = ldStage + 1;
          ldStage = selp_u32(0, next, next == numStages);
          ldIssued += 1;
          ldSource += 1;
          IF(ldSource == ldNumSources) {
            u32 nextSlot = ldSlot + 1;
            ldSlot = selp_u32(0, nextSlot, nextSlot == reduceItemSlots);
            startItem();
          }
        };
        // Issue loads while the ring has room (thread 0). A source that the CPU thread reads from another NVLink
        // domain is loaded once its read has arrived (its signal in rdmaSignalBuffer), and a partial result that this
        // kernel computes for itself (in an all-reduce) once all of it has been stored (its done signal in
        // cpuInBuffer): until then, the loads stop there, and the block goes on with the stages it has. It waits for
        // the signal only when it has combined all stages loaded before (blocking, at the start of a stage): its
        // earlier items are then complete, including the partial results among them, which come before anything that
        // waits, and in an all-reduce in two rounds the shares of a part, which come before the copies of the part's
        // other shares (see localCopies), so all blocks of all ranks make progress.
        u64 signalFlags = concurrency(group->rdmaSignalBuffer);
        u64 doneFlags = concurrency(group->cpuInBuffer) + 4 * 16;
        auto fill = [&](bool blocking) {
          WHILE(!ldDone & (ldIssued - consumed < numStages)) {
            if (anyWaits) {
              IF(ldWaits != 0xffffffffll) {
                u32 wait = ld_global_nc_u32(tables + pipeWaitsOffset + widen((ldWaits + ldSource) * 4));
                IF(wait != 0) {
                  // An RDMA signal is a word per NIC, all of which get the step value (see writeRdmaSignal in
                  // cputhread.cc), a done signal one word
                  Value done = wait >= (int64_t)waitDoneSignal;
                  u32 index = (wait & (int64_t)(waitDoneSignal - 1)) - 1;
                  u64 flag = selp_u64(doneFlags + widen(index * 4), signalFlags + widen(index * Group::rdmaSignalBytes),
                      done);
                  static_assert(Group::rdmaSignalBytes == 16 && maxDevices == 4);
                  auto load = [&]() {
                    u32 v = predicated(done, ld_global_relaxed_sys_u32)(flag);
                    std::array<Value, 4> words;
                    PRED(!done) ld_global_relaxed_sys_v4_u32(words, flag);
                    return selp_u32(v, min_u32(min_u32(words[0], words[1]), min_u32(words[2], words[3])), done);
                  };
                  u32 value = load();
                  if (blocking) {
                    IF(ldIssued == consumed) {
                      profiling.enter("reduce wait");
                      WHILE(value < stepValue) {
                        value = load();
                      }
                      profiling.leave();
                    }
                  }
                  IF(value < stepValue) {
                    BREAK;
                  }
                  // The data arrived (or was stored) before the signal; the loads are bulk copies
                  fence_acquire_sys();
                  fence_proxy_async_global();
                }
              }
            }
            issueLoad();
          }
        };

        // Fill the ring
        IF_D(isThreadZero) {
          // The loads below may read data whose readiness generic loads have observed
          fence_proxy_async_global();
          startItem();
          fill(false);
        }
        barrier_sync();

        u32 consStage = 0;
        u32 consParity = 0;
        u32 consSlot = 0;
        u32 result = 0; // the result buffer of the next item
        Vector<Vector<Value>> acc(numVectors);
        for (auto& a : acc) {
          for (size_t j : range(accPerVector)) {
            a.emplace_back(acc64 ? ValType::U64 : ValType::U32);
          }
        }
        Vector<std::array<u32, 4>> raw(numVectors); // the last loaded vectors
        WHILE(true) {
          u32 entry = reduceItemRing + consSlot * 8;
          u32 edge = ld_shared_u32(entry);
          IF(edge == 0xffffffffll) {
            BREAK;
          }
          u32 chunk = ld_shared_u32(entry + 4);
          u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
          u32 numSources = ld_global_nc_u32(record + offsetof(PipeEdge, numSources));
          u32 numDestinations = ld_global_nc_u32(record + offsetof(PipeEdge, numDestinations));
          u32 addrSlot = ld_global_nc_u32(record + offsetof(PipeEdge, addrSlot));
          u64 offset = widen(chunk) * chunkSize;
          u32 n = min_u32(ld_global_nc_u32(record + offsetof(PipeEdge, bytes)) - narrow(offset), chunkSize);
          // An average divides once it has combined the inputs among the sources (the partial results of other
          // domains after them are divided already)
          u32 inputSources;
          Value divisor(acc64 ? ValType::U64 : ValType::U32);
          if (op == ReduceOp::Avg) {
            inputSources = ld_global_nc_u32(record + offsetof(PipeEdge, inputSources));
            arith(acc64 ? "cvt.rn.f64.u32" : "cvt.rn.f32.u32", divisor,
                ld_global_nc_u32(record + offsetof(PipeEdge, divisor)));
          }
          for (auto& a : acc) {
            for (Value& v : a) {
              v = identity;
            }
          }
          FOR(u32 s = 0, s < numSources, s += 1) {
            if (anyWaits) {
              // The load of this stage has not been issued: it waits for a source from another domain
              IF_D(isThreadZero & (ldIssued == consumed)) {
                fill(true);
              }
            }
            mbarrier_wait_parity_spin(reduceBarriers + consStage * 8, consParity);
            u32 stage = reduceStages + consStage * chunkSize;
            for (uint32_t v : range(numVectors)) {
              u32 off = threadIndex * 16 + 16 * blockSize * v;
              PRED(off < n) ld_shared_v4_u32_into(raw[v], stage + off);
              Vector<Value> x = unpack(raw[v]);
              for (size_t j : range(accPerVector)) {
                combine(acc[v][j], x[j]);
              }
            }
            if (op == ReduceOp::Avg) {
              IF(s + 1 == inputSources) {
                for (auto& a : acc) {
                  for (Value& v : a) {
                    arith(acc64 ? "div.rn.f64" : "div.rn.f32", v, v, divisor);
                  }
                }
              }
            }
            u32 next = consStage + 1;
            Value wrap = next == numStages;
            consStage = selp_u32(0, next, wrap);
            consParity = consParity ^ selp_u32(1, 0, wrap);
            consumed += 1;
            // Every thread has combined the stage: load the next chunk into it
            barrier_sync();
            IF_D(isThreadZero) {
              fill(false);
            }
          }
          // The result buffer is free once the stores of the item before the previous one have read it
          IF_D(isThreadZero) {
            cp_async_bulk_wait_group_read(reduceResults - 1);
          }
          barrier_sync();
          u32 buffer = reduceResultBuffers + result * chunkSize;
          Value copy = numSources == 1;
          for (uint32_t v : range(numVectors)) {
            std::array<u32, 4> x = pack(acc[v]);
            for (size_t j : range(4)) {
              x[j] = selp_u32(raw[v][j], x[j], copy);
            }
            u32 off = threadIndex * 16 + 16 * blockSize * v;
            PRED(off < n) st_shared_v4_u32(buffer + off, x);
          }
          // The bulk stores below read what the threads have written
          fence_proxy_async_shared_cta();
          barrier_sync();
          IF_D(isThreadZero) {
            FOR(u32 d = 0, d < numDestinations, d += 1) {
              u64 dst = ld_u64(table + widen((addrSlot + numSources + d) * 8)) + offset;
              cp_async_bulk_global_shared(dst, buffer, n);
            }
            cp_async_bulk_commit_group();
            if (anyDoneSignals) {
              // An edge with a done signal (a partial result that another domain reads, or a result that this kernel
              // waits for: its own domain's partial result, a piece of two phases): once all its items (of any block)
              // are stored, the last one writes the signal (per launch, every item counts once, so the counter wraps
              // back to 0)
              u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
              IF((flags & edgeDoneSignal) != 0) {
                cp_async_bulk_wait_group(0);
                fence_proxy_async_global();
                fence_release_gpu();
                u32 numItems = ld_global_nc_u32(record + offsetof(PipeEdge, numItems));
                u32 old = atom_global_relaxed_inc_u32(
                    *partialDone + widen((concurrencyIndex * numEdges + edge) * 4), numItems - 1);
                IF(old == numItems - 1) {
                  fence_acquire_gpu();
                  fence_release_sys();
                  u32 doneSignal = ld_global_nc_u32(record + offsetof(PipeEdge, doneSignal));
                  st_global_relaxed_sys_u32(concurrency(group->cpuInBuffer) + widen((16 + doneSignal) * 4), stepValue);
                }
              }
            }
          }
          result = result ^ 1;
          u32 nextSlot = consSlot + 1;
          consSlot = selp_u32(0, nextSlot, nextSlot == reduceItemSlots);
        }
        IF_D(isThreadZero) {
          cp_async_bulk_wait_group(0);
        }
        profiling.leave();
      };

      if (reduceKernel) {
        CHECK(!nvlsLocal && !anySignals);
        reduceLoop();
      } else if (nvlsLocal) {
        IF(nvlsMode) {
          IF_D(isPipelineLane & (blockIndex == 0)) {
            nvlsCopy();
          }
        }
        ELSE {
          unicastCopy();
        }
      } else {
        unicastCopy();
      }
      // All copies done before the exit's release fence
      profiling.enter("copy end barrier");
      barrier_sync();
      profiling.leave();
    } else {
      // Rounds: warp 0 scans the edges that are not done, 32 at a time (one per lane). Relay edges whose signal
      // has arrived are finished on the spot. The first ready copy edge is selected; after a block barrier, the
      // whole block copies it and thread 0 marks it done. The selection slot alternates between two words, so
      // that warp 0 can write the next round's while other warps still read this round's.
      static constexpr uint32_t selWait = 0xffffffff; // nothing ready: scan again
      static constexpr uint32_t selDone = 0xfffffffe; // all edges done
      u32 doneAddr = addShared32(4, 4 * numDoneWords, "done");
      u32 selAddr = addShared32(4, 8, "sel");
      for (uint32_t i : range((numDoneWords + blockSize - 1) / blockSize)) {
        u32 w = blockSize * i + threadIndex;
        PRED(w < numDoneWords) st_shared_u32(doneAddr + 4 * w, 0);
      }
      barrier_sync(14);

      u32 round = 0;
      WHILE(true) {
        u32 selSlot = selAddr + 4 * (round & 1);
        IF(warpIndex == 0) {
          warp_sync();
          u32 found = selWait;
          u32 pending = 0;
          FOR(u32 w = 0, w < numDoneWords, w += 1) {
            u32 doneWord = ld_shared_u32(doneAddr + 4 * w);
            u32 e = 32 * w + laneIndex;
            Value notDone = (e < numEdges) & (((doneWord >> laneIndex) & 1) == 0);
            u64 desc = edgeTableAddr + widen(e * edgeStride);
            u32 flags = predicated(notDone, ld_global_nc_u32)(desc);
            u32 signal = predicated(notDone, ld_global_nc_u32)(desc + 4);
            u32 sourceDep = predicated(notDone, ld_global_nc_u32)(desc + 8);
            Value relay = notDone & ((flags & edgeRelay) != 0);
            Value depWait = notDone & ((flags & edgeWait) != 0) & !relay;
            u32 value = stepValue;
            u64 depAddr = concurrency(depFilled.local) + widen(4 * (sourceDep + localNodes.size() * blockIndex));
            value = selp(predicated(depWait, ld_global_relaxed_sys_u32)(depAddr), value, depWait);
            if (!tuning) {
              // (the signal's words, one per NIC, see writeRdmaSignal in cputhread.cc)
              u64 flagAddr = concurrency(group->rdmaSignalBuffer) + widen(signal * Group::rdmaSignalBytes);
              std::array<Value, 4> words;
              PRED(relay) ld_global_relaxed_sys_v4_u32(words, flagAddr);
              u32 signalValue = min_u32(min_u32(words[0], words[1]), min_u32(words[2], words[3]));
              value = selp(signalValue, value, relay);
            }
            Value ready = notDone & (value >= stepValue);
            Value relayReady = ready & relay;
            u32 relayBits = vote_sync_ballot(relayReady);
            u32 copyBits = vote_sync_ballot(ready & !relay);
            pending |= vote_sync_ballot(notDone) & ~relayBits;
            // A relay edge whose signal has arrived: its node is filled; the lane that has seen the signal
            // passes it on
            IF_D(relayReady) {
              signalEdge(desc);
            }
            IF_D((laneIndex == 0) & (relayBits != 0)) {
              st_shared_u32(doneAddr + 4 * w, doneWord | relayBits);
            }
            IF(copyBits != 0) {
              found = 32 * w + lowest_bit_b32(copyBits);
              BREAK;
            }
          }
          // The lane that has seen the selected edge's source filled orders its load before the copy, which
          // the block barrier below extends to all threads
          IF(found != selWait) {
            fence_acquire_sys();
          }
          IF_D(laneIndex == 0) {
            u32 waitValue = selWait;
            u32 doneValue = selDone;
            st_shared_u32(selSlot, selp(found, selp(waitValue, doneValue, pending != 0), found != selWait));
          }
        }
        barrier_sync(14);
        u32 sel = ld_shared_u32(selSlot);
        round += 1;
        IF(sel == selDone) {
          BREAK;
        }
        IF(sel != selWait) {
          u64 desc = edgeTableAddr + widen(sel * edgeStride);
          EdgeWords w;
          loadEdge(w, desc, nullptr);
          u64 srcaddr = resolveEdge(w);
          copyEdge(desc, w.flags, w.bytes, srcaddr, w.numDestinations);
          IF_D(isThreadZero) {
            u32 wordAddr = doneAddr + 4 * (sel / 32);
            u32 one = 1;
            st_shared_u32(wordAddr, ld_shared_u32(wordAddr) | (one << (sel % 32)));
          }
        }
      }
    }

    auto execSync = [&](uint32_t stepOffset) {
      Vector<u64> ldaddrs;
      Vector<u64> staddrs;
      Vector<Value> preds;
      uint32_t numWarps = blockSize / 32;
      for (uint32_t i : range((syncsetup.size() + blockSize - 1) / blockSize)) {
        profiling.enter("sync setup");
        ldaddrs.emplace_back();
        staddrs.emplace_back();
        preds.emplace_back();
        u32 index = blockSize * i + (warpIndex + numWarps - 1) % numWarps + laneIndex * (blockSize / 32);
        u64& ldaddr = ldaddrs.back();
        u64& staddr = staddrs.back();
        Value& pred = preds.back();
        pred = index < syncsetup.size();
        PRED(pred) {
          u32 addr = csyncaddr + index * sizeof(SyncSetup);
          u64 stbase = ld_const_u64(addr);
          u32 itembytes = ld_const_u32(addr + 8);
          u32 ldoffset = ld_const_u32(addr + 12);
          u32 stBlockStride = ld_const_u32(addr + 16);
          staddr = stbase + widen(itembytes * concurrencyIndex + stBlockStride * blockIndex);
          ldaddr = concurrency(syncAddrs.local) + widen(ldoffset + sizeof(uint32_t) * ranks.size() * blockIndex);
        }
      }
      u32 targetValue = stepValue + stepOffset;
      profiling.enter("exit sync barrier");
      barrier_sync();
      profiling.enter("exit sync barrier wait");
      for (size_t i : indices(staddrs)) {
        profiling.enter("sync st");
        PRED(preds[i]) st_global_relaxed_sys_u32(staddrs[i], targetValue);
      }
      for (size_t i : indices(ldaddrs)) {
        profiling.enter("sync ld");
        u64 ldaddr = ldaddrs[i];
        Value pred = preds[i];
        Label loop;
        Label done;
        GOTO_IF_NOT(pred, done);
        LABEL(loop);
        pred = ld_global_relaxed_sys_u32(ldaddr) < targetValue;
        GOTO_IF_NOT(pred, done);
        nanosleep(10);
        GOTO(loop);
        LABEL(done);
      }
    };

    profiling.enter("exit sync entry");
    if (hasRemoteWrites) {
      IF_D(threadIndex == 0) {
        profiling.enter("fence.release.sys");
        fence_release_sys();
      }
    }
    execSync(1);

    profiling.leave();
    profiling.finish();
    rv->profilingNames.resize(profiling.nextIndex);
    for (auto& v : profiling.nameToIndex) {
      rv->profilingNames.at(v.second) = v.first;
    }
    ret();
  }

  // compile_op_params (see paramBuffer above): copies the first `words` words of its second parameter to the copy of
  // concurrency slot `slot` in the parameter buffer, from word `first` of the copy on
  if (paramBuffer) {
    auto* pf = mod.newFunction("compile_op_params");
    FunctionScope pfScope(pf);
    pf->maxThreads = paramFillBlockSize;
    pf->addParamBytes(8, 16); // slot, first, words (uint32_t each)
    pf->addParamBytes(8, 8 * paramFillWords);
    activateNewBlock("entry");
    u32 header = paramBase(0);
    u32 slot = loadParamField(header, 0, ValType::U32);
    u32 first = loadParamField(header, 4, ValType::U32);
    u32 words = loadParamField(header, 8, ValType::U32);
    u32 i = blockIdx_x() * paramFillBlockSize + threadIdx_x();
    IF_D(i < words) {
      u64 v = ld_param_u64(paramBase(1) + i * 8, 0);
      st_global_u64(globalAddr("global_params") + widen(slot * rv->paramBufferStride + (first + i) * 8), v);
    }
    ret();
  }

  std::string ptx = mod.finalize();
  rv->ptx = ptx;
  return rv;
}

void fillCompileOpParamBuffer(const KernelHandle& kh, const CompiledKernel& kernel, CUstream stream, uint32_t slot,
    std::span<const uintptr_t> tensorAddrs, std::span<const uintptr_t> mappedAddrs,
    std::span<const uint64_t> nvlsTensors) {
  CHECK(kh.paramBuffer && kernel.paramsFunction != nullptr && slot < maxConcurrency);
  CHECK(8 * tensorAddrs.size() <= kh.paramMappingsOffset);
  CHECK(kh.paramMappingsOffset + 8 * mappedAddrs.size() <= kh.paramNvlsOffset);
  CHECK(kh.paramNvlsOffset + 8 * nvlsTensors.size() <= kh.paramBufferStride);
  std::vector<uint64_t> words(kh.paramBufferStride / 8);
  std::ranges::copy(tensorAddrs, words.begin());
  std::ranges::copy(mappedAddrs, words.begin() + kh.paramMappingsOffset / 8);
  std::ranges::copy(nvlsTensors, words.begin() + kh.paramNvlsOffset / 8);
  // (the launch copies its parameters, so the arrays can be reused at once)
  struct Header {
    uint32_t slot;
    uint32_t first;
    uint32_t words;
    uint32_t pad;
  };
  static_assert(sizeof(Header) == 16);
  std::vector<uint64_t> data(paramFillWords);
  for (size_t first = 0; first < words.size(); first += paramFillWords) {
    Header header = {slot, (uint32_t)first, (uint32_t)std::min<size_t>(paramFillWords, words.size() - first), 0};
    std::copy_n(words.begin() + first, header.words, data.begin());
    std::array<void*, 2> args = {&header, data.data()};
    CHECK_CU(cuLaunchKernel(kernel.paramsFunction, (header.words + paramFillBlockSize - 1) / paramFillBlockSize, 1, 1,
        paramFillBlockSize, 1, 1, 0, stream, args.data(), nullptr));
  }
}

void launchCompileOpKernel(const KernelHandle& kh, const CompiledKernel& kernel, const KernelConfig& config,
    CUstream stream, const CompileOpCopyParameters& params, std::span<const uintptr_t> tensorAddrs,
    std::span<const uintptr_t> mappedAddrs, std::span<const uint64_t> nvlsTensors, bool fill) {
  if (kh.paramBuffer && fill) {
    fillCompileOpParamBuffer(kh, kernel, stream, params.concurrencyIndex, tensorAddrs, mappedAddrs, nvlsTensors);
  }
  // (with the parameter buffer, the kernel takes params only; without NVLS, it takes no NVLS words)
  CompileOpCopyParameters p = params;
  std::array<void*, 4> kparams = {&p, (void*)tensorAddrs.data(), (void*)mappedAddrs.data(), (void*)nvlsTensors.data()};
  CUlaunchConfig launchConfig;
  CUlaunchAttribute attr;
  std::memset(&launchConfig, 0, sizeof(launchConfig));
  launchConfig.grid = {config.gridSize, 1, 1};
  launchConfig.block = {config.blockSize, 1, 1};
  launchConfig.sharedMemBytes = 0;
  launchConfig.hStream = stream;
  launchConfig.attrs = &attr;
  launchConfig.numAttrs = 1;
  attr.id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
  attr.value.value = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE;
  CHECK_CU(cuLaunchKernelEx(&launchConfig, kernel.function, kparams.data(), nullptr));
}

} // namespace moodist
