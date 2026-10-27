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
  // the array at dsts
  void load(const u64& src, const u64& offset, const u32& bytes, const u64& dsts, const u32& numDestinations,
      const u32& tag) {
    u32 record = records + loadStage * recordSize;
    st_shared_u64(record, offset);
    st_shared_u64(record + 8, dsts);
    st_shared_u32(record + 16, bytes);
    st_shared_u32(record + 20, numDestinations);
    st_shared_u32(record + 24, tag);
    u32 barrier = barriers + loadStage * 8;
    mbarrier_arrive_expect_tx(barrier, bytes);
    cp_async_bulk_shared_global(buffers + loadStage * chunkSize, src + offset, bytes, barrier);
    u32 next = loadStage + 1;
    loadStage = selp_u32(0, next, next == numStages);
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

  auto* fn = mod.newFunction("compile_op_copy");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = blockSize;
    fn->addParamBytes(8, sizeof(CompileOpCopyParameters));
    fn->addParamBytes(8, sizeof(uint64_t) * std::max(graph.tensors.size(), (size_t)1));
    fn->addParamBytes(8, sizeof(uint64_t) * std::max(graph.remoteCudaTensorMappings.size(), (size_t)1));

    activateNewBlock("entry");

    // --- Load kernel parameters ---
    u32 params = paramBase(0);
    u32 stepValue = loadParamField(params, 0, ValType::U32);
    u32 concurrencyIndex = loadParamField(params, 4, ValType::U32);

    u32 inputAddrs = paramBase(1);
    u32 mappedAddrs = paramBase(2);

    u32 threadIndex = threadIdx_x();
    u32 blockIndex = blockIdx_x();

    u32 warpIndex = threadIndex / 32;
    u32 laneIndex = laneid();

    // Kernel profiling (debugging aid, compiled out): every thread records a timestamp at each profiling.enter(),
    // and the op's 401st launch writes them as a Chrome trace to moodist-trace-cuda-<rank>.json (dumpProfiling in
    // processgroup.cc). Needs Profiling::maxBytesPerThread of device memory per thread, also for every tuning
    // candidate.
    profiling = Profiling();
    if (false) {
      auto profilingBuffer = ctx.group->allocateDevice(profiling.dataSize(config));
      profiling = Profiling();
      profiling.start(config, profilingBuffer.cudaPointer);
      rv->profilingData = std::move(profilingBuffer);
      rv->profilingBytesPerThread = Profiling::maxBytesPerThread;
      rv->profilingCountdown = 400;
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
      CHECK(v.sources.size() == 1);
      CHECK(v.destinations.size() >= 1);
      auto& src = v.sources[0];
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
          CHECK(localdst == nullptr);
          localdst = &n;
        }
      }
      CHECK(src.rank == rank || localdst);
      CHECK(std::ranges::find(ranks, src.rank) != ranks.end());

      if (v.executorRank == rank) {
        localCopies.push_back(v);
      }
    }
    // Relay edges first: the scan for ready edges below stops at the first ready copy, and always reaches them
    auto firstCopy = std::stable_partition(localCopies.begin(), localCopies.end(), [](const Edge& e) {
      return e.sources.empty();
    });
    // Then the copies in destination rank order rotated to start at this rank: in rank order, every rank would
    // write to the same peer at once (in an all-to-all, all ranks to rank 0 first, then to rank 1, ...), whose
    // NVLink ingress then limits all of them while the other peers idle. An edge with several destinations counts
    // by its first one (multicast edges list their destinations rotated the same way).
    std::stable_sort(firstCopy, localCopies.end(), [&](const Edge& a, const Edge& b) {
      auto key = [&](const Edge& e) {
        return (e.destinations.at(0).rank + group->size - rank) % group->size;
      };
      return key(a) < key(b);
    });

    // nowaits: no edge waits for another edge or for the CPU thread, so the edges run in order
    bool nowaits = true;
    bool hasRemoteWrites = false;
    size_t maxDestinations = 0;
    for (const Edge& e : localCopies) {
      if (e.sources.empty() || !e.sources[0].filled) {
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
    // parameter), else the peer's tensor at byte offset slot in the exchanged address table.
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
    // Where to write the address at which a peer maps one of this rank's tensors (in the peer's address table)
    struct AddrSetup {
      uint64_t base;
      uint32_t itembytes;
      uint32_t paramOffset;
      uint32_t stride;
      uint32_t slot;
    };
    static constexpr uint32_t edgeRelay = 1;   // relay edge: waits for the CPU thread's signal, copies nothing
    static constexpr uint32_t edgeWait = 2;    // waits for its source node (relay edges: for the signal)
    static constexpr uint32_t edgeAcquire = 4; // fence.acquire.sys before the copy (source written by another agent)
    static constexpr uint32_t edgeSignal = 8;  // other edges wait for its destinations: write their depFilled slots

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
        const Node& src = e.sources[0];
        NodeSetup source = nodeSetup(src);
        std::memcpy(record + sourceSetupOffset, &source, sizeof(source));
        if (!src.filled) {
          s.flags |= edgeWait | edgeAcquire;
          s.sourceDep = localNodeIndex(src.id);
        }
        if (src.rank != rank) {
          s.flags |= edgeAcquire;
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

      AddrSetup s;
      s.base = r.base;
      s.itembytes = r.itembytes;
      s.paramOffset = sizeof(uint64_t) * i;
      s.stride = sizeof(uint64_t) * v.stride;
      s.slot = sizeof(uint64_t) * v.destinationSlot;
      addrTable.push_back(s);
    }

    // Without waits, the items of all edges form one sequence, which the pipelines of all blocks take items from
    // (see the copy loop below). An item is a chunk of an edge and a group of its destinations: an edge's
    // destinations are split into groups when the op has few chunks, so that the stores of an all-gather of a few
    // chunks to many ranks spread over the SMs too (a load per group: more local reads, which are cheap). Per edge:
    // the index of its first item in the sequence, the destination groups (item i of the edge covers chunk
    // i / groups and destination group i % groups), and the slots of its addresses in the address table that every
    // block resolves at the start (source, then destinations). A last record (firstItem = the number of items) ends
    // the sequence.
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
      uint32_t pad[2];
    };
    static_assert(sizeof(PipeEdge) == 48);
    std::vector<PipeEdge> pipeEdges;
    std::vector<NodeSetup> pipeAddrSetups;
    uint32_t pipeNumItems = 0;
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
        p.groupSize = (numDestinations + std::min<uint64_t>(groupsWanted, numDestinations) - 1) /
                      std::min<uint64_t>(groupsWanted, numDestinations);
        p.groups = (numDestinations + p.groupSize - 1) / p.groupSize;
        p.firstItem = pipeNumItems;
        p.numItems = numChunks * p.groups;
        p.bytes = e.bytes;
        p.addrSlot = pipeAddrSetups.size();
        p.numDestinations = numDestinations;
        p.flags = s.flags;
        p.signals = s.signals;
        p.numSignals = s.numSignals;
        pipeAddrSetups.push_back(nodeSetup(e.sources.at(0)));
        for (const Node& n : e.destinations) {
          pipeAddrSetups.push_back(nodeSetup(n));
        }
        pipeEdges.push_back(p);
        CHECK((uint64_t)pipeNumItems + p.numItems < (1ull << 30));
        pipeNumItems += p.numItems;
      }
      PipeEdge end = {};
      end.firstItem = pipeNumItems;
      pipeEdges.push_back(end);
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
    {
      const size_t perStage = chunkSize + 8 + PipeRing::recordSize;
      size_t fixed = 256 + scanShared + (nowaits ? 0 : 8 * maxDestinations); // 256: alignment padding
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

    // Each pipeline's stage ring (the warps beyond the pipelines alias the first ring, and never use it)
    std::optional<PipeRing> ring;
    Value isPipelineLane = (laneIndex == 0) & (warpIndex < numPipelines);
    {
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

    // Address exchange: write the addresses at which the peers map this rank's tensors into their address tables
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
      predicated(pred, st_global_relaxed_sys_u64)(
          addr, predicated(pred, ld_param_u64)(mappedAddrs + paramOffset, 0) | parityTag);
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
        r = ld_param_u64(inputAddrs + (slot & ~1u), 0);
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
      r = predicated(!done & isParam, ld_param_u64)(inputAddrs + (slot & ~1u), 0);
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
    CHECK(storeLag < (int)numStages);

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
          ring->load(srcaddr, offset, n, destinations, numDestinations, 0);
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
      const uint32_t firstBatch = std::clamp<uint32_t>(pipeNumItems / numClaimers, 1, numStages);
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
      profiling.enter("resolve addresses");
      FOR(u32 first = 0, first < numSlots, first += blockSize) {
        u32 i = first + threadIndex;
        Value valid = i < numSlots;
        u64 setup = tables + pipeAddrSetupOffset + widen(i * sizeof(NodeSetup));
        u32 slot = predicated(valid, ld_global_nc_u32)(setup);
        u32 offsetLow = predicated(valid, ld_global_nc_u32)(setup + 4);
        u32 offsetHigh = predicated(valid, ld_global_nc_u32)(setup + 8);
        u64 addr = nodeaddrpred(valid, slot, offsetLow, offsetHigh);
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

      std::optional<u64> edgeDone;
      if (anySignals) {
        edgeDone = addGlobalVar(4, 4 * maxConcurrency * numEdges, "edgedone");
      }

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
          ring->load(src, offset, n, table + widen((addrSlot + 1 + firstDestination) * 8), numGroupDestinations, tag);
          item += 1;
          loaded += 1;
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
              u64 flagAddr = concurrency(group->cpuOutBuffer) + widen(4 * (signal + 16));
              value = selp(predicated(relay, ld_global_relaxed_sys_u32)(flagAddr), value, relay);
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

  std::string ptx = mod.finalize();
  rv->ptx = ptx;
  return rv;
}

} // namespace moodist
