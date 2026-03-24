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

struct EmitMultiCopy {
  virtual ~EmitMultiCopy() = default;
  virtual void emit(u64 src, Vector<u64> dst, u32 ndst, u32 bytes) = 0;
};

struct EmitLockstep : EmitMultiCopy {
  const KernelConfig& config;
  u32 blockIndex;
  u32 warpIndex;
  u32 laneIndex;

  EmitLockstep(const KernelConfig& config, const u32& blockIndex, const u32& warpIndex, const u32& laneIndex)
      : config(config), blockIndex(blockIndex), warpIndex(warpIndex), laneIndex(laneIndex) {
    CHECK(numBuffers > 0);
    CHECK(bufferSize > 0);
    CHECK(config.blockSize / 64 < 15);
    CHECK(config.blockSize % 64 == 0);
  }

  uint32_t numBlocks = config.gridSize;
  uint32_t inputSharedSize = config.sharedMemory;

  uint32_t numBuffers = config.blockSize / 32;
  uint32_t blockSharedAlignment = numBuffers * 256;
  uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
  uint32_t bufferSize = blockSharedSize / numBuffers;

  u32 bufferBaseAddr = addShared32(16, numBuffers* bufferSize);
  u32 barrierBaseAddr = addShared32(8, numBuffers * 8);
  u32 sharedOffset = addShared32(4, 4);

  u32 bufferIndex = warpIndex;

  u32 syncIndex = 1 + warpIndex / 2;
  u32 pairIndex = warpIndex % 2;

  u32 buffer = bufferBaseAddr + bufferIndex * bufferSize;
  u32 barrier = barrierBaseAddr + bufferIndex * 8;

  void emit(u64 src, Vector<u64> dst, u32 ndst, u32 bytes) override {
    Label alldone;
    Value isLeader = laneIndex == 0;

    IF_D(isLeader) {
      st_shared_release_cta_u32(sharedOffset, blockIndex * bufferSize);
      mbarrier_init(barrier, 1);
    }
    barrier_sync();

    IF(pairIndex != 0) {
      GOTO_IF(bar_red_or(syncIndex, 64, 0), alldone);
    }

    PRED(bytes % 16 != 0) trap();
    PRED(src % 16 != 0) trap();
    for (size_t i : indices(dst)) {
      PRED((i < ndst) & (dst[i] % 16 != 0)) trap();
    }

    Label loop;
    LABEL(loop);

    for (uint32_t parity : range(2)) {
      u32 offset = 0;
      PRED(isLeader) offset = atom_shared_relaxed_cta_add_u32(sharedOffset, numBlocks * bufferSize);
      offset = shfl_sync_idx_b32(offset, 0, 31);
      IF(offset >= bytes) {
        bar_red_or(syncIndex, 64, 1);
        GOTO(alldone);
      }

      u64 srcaddr = src + widen(offset);

      u32 size = min_u32(bytes - offset, bufferSize);

      cp_async_bulk_wait_group_read(0);
      warp_sync();
      IF_D(isLeader) {
        mbarrier_expect_tx(barrier, size);
        mbarrier_arrive_noComplete(barrier);
        cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
      }

      bar_red_or(syncIndex, 64, 0);
      IF_D(isLeader) {
        mbarrier_wait_parity(barrier, parity);
      }
      warp_sync();
      for (size_t i : indices(dst)) {
        IF_D(i < ndst) {
          u64 dstaddr = dst[i] + widen(offset);
          cp_async_bulk_global_shared(dstaddr, buffer, size);
        }
      }
      cp_async_bulk_commit_group();

      GOTO_IF(bar_red_or(syncIndex, 64, 0), alldone);
    }

    GOTO(loop);

    LABEL(alldone);
    cp_async_bulk_wait_group(0);
    IF_D(isLeader) {
      mbarrier_inval(barrier);
    }
    barrier_sync();
  }
};

struct EmitSimple : EmitMultiCopy {
  const KernelConfig& config;
  u32 blockIndex;
  u32 warpIndex;
  u32 laneIndex;

  EmitSimple(const KernelConfig& config, const u32& blockIndex, const u32& warpIndex, const u32& laneIndex)
      : config(config), blockIndex(blockIndex), warpIndex(warpIndex), laneIndex(laneIndex) {
    CHECK(numBuffers > 0);
    CHECK(bufferSize > 0);
    CHECK(config.blockSize % 32 == 0);
  }

  uint32_t numBlocks = config.gridSize;
  uint32_t inputSharedSize = config.sharedMemory;

  uint32_t numBuffers = config.blockSize / 32;
  uint32_t blockSharedAlignment = numBuffers * 256;
  uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
  uint32_t bufferSize = blockSharedSize / numBuffers;

  u32 bufferBaseAddr = addShared32(16, numBuffers* bufferSize);
  u32 barrierBaseAddr = addShared32(8, numBuffers * 8);

  u32 bufferIndex = warpIndex;

  u32 buffer = bufferBaseAddr + bufferIndex * bufferSize;
  u32 barrier = barrierBaseAddr + bufferIndex * 8;

  void emit(u64 src, Vector<u64> dst, u32 ndst, u32 bytes) override {
    Label alldone;
    Value isLeader = laneIndex == 0;
    IF_D(isLeader) {
      mbarrier_init(barrier, 1);
    }

    PRED(bytes % 16 != 0) trap();
    PRED(src % 16 != 0) trap();
    for (size_t i : indices(dst)) {
      PRED((i < ndst) & (dst[i] % 16 != 0)) trap();
    }

    u32 offset = (warpIndex * numBlocks + blockIndex) * bufferSize;

    Label loop;
    LABEL(loop);
    for (uint32_t parity : range(2)) {
      IF(offset >= bytes) {
        GOTO(alldone);
      }
      u64 srcaddr = src + widen(offset);
      u32 size = min_u32(bytes - offset, bufferSize);
      cp_async_bulk_wait_group_read(0);
      warp_sync();
      IF_D(isLeader) {
        mbarrier_expect_tx(barrier, size);
        mbarrier_arrive_noComplete(barrier);
        cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
        mbarrier_wait_parity(barrier, parity);
      }
      warp_sync();
      for (size_t i : indices(dst)) {
        IF_D(i < ndst) {
          u64 dstaddr = dst[i] + widen(offset);
          cp_async_bulk_global_shared(dstaddr, buffer, size);
        }
      }
      cp_async_bulk_commit_group();
      offset += numBlocks * blockSharedSize;
    }
    GOTO(loop);
    LABEL(alldone);
    cp_async_bulk_wait_group(0);
    IF_D(isLeader) {
      mbarrier_inval(barrier);
    }
    barrier_sync(3);
  }
};

struct EmitBuffered : EmitMultiCopy {
  const KernelConfig& config;
  u32 blockIndex;
  u32 warpIndex;
  u32 laneIndex;

  EmitBuffered(const KernelConfig& config, const u32& blockIndex, const u32& warpIndex, const u32& laneIndex)
      : config(config), blockIndex(blockIndex), warpIndex(warpIndex), laneIndex(laneIndex) {
    CHECK(numBuffers > 0);
    CHECK(bufferSize > 0);
    CHECK(config.blockSize % 32 == 0);
  }

  uint32_t numBlocks = config.gridSize;
  uint32_t inputSharedSize = config.sharedMemory;

  uint32_t numBuffers = config.blockSize / 32 * 2;
  uint32_t blockSharedAlignment = numBuffers * 256;
  uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
  uint32_t bufferSize = blockSharedSize / numBuffers;

  u32 bufferBaseAddr = addShared32(16, numBuffers* bufferSize);
  u32 barrierBaseAddr = addShared32(8, numBuffers * 8);

  u32 bufferIndex = warpIndex * 2;

  u32 buffers[2] = {bufferBaseAddr + bufferIndex * bufferSize, bufferBaseAddr + (bufferIndex + 1) * bufferSize};
  u32 barriers[2] = {barrierBaseAddr + bufferIndex * 8, barrierBaseAddr + (bufferIndex + 1) * 8};

  void emit(u64 src, Vector<u64> dst, u32 ndst, u32 bytes) override {
    Value isLeader = laneIndex == 0;
    IF_D(isLeader) {
      mbarrier_init(barriers[0], 1);
      mbarrier_init(barriers[1], 1);
    }

    PRED(bytes % 16 != 0) trap();
    PRED(src % 16 != 0) trap();
    for (size_t i : indices(dst)) {
      PRED((i < ndst) & (dst[i] % 16 != 0)) trap();
    }

    u32 offset = (warpIndex * numBlocks + blockIndex) * bufferSize;

    WHILE(true) {
      for (int parity : range(2)) {
        for (int index : range(2)) {
          IF(offset >= bytes) {
            BREAK;
          }
          u64 srcaddr = src + widen(offset);
          u32 size = min_u32(bytes - offset, bufferSize);
          cp_async_bulk_wait_group_read(0);
          warp_sync();
          IF_D(isLeader) {
            mbarrier_expect_tx(barriers[index], size);
            mbarrier_arrive_noComplete(barriers[index]);
            cp_async_bulk_shared_global(buffers[index], srcaddr, size, barriers[index]);
            mbarrier_wait_parity(barriers[index], parity);
          }
          warp_sync();
          for (size_t i : indices(dst)) {
            IF_D(i < ndst) {
              u64 dstaddr = dst[i] + widen(offset);
              cp_async_bulk_global_shared(dstaddr, buffers[index], size);
            }
          }
          cp_async_bulk_commit_group();

          offset += numBlocks * blockSharedSize / 2;
        }
      }
    }

    cp_async_bulk_wait_group(0);
    IF_D(isLeader) {
      mbarrier_inval(barriers[0]);
      mbarrier_inval(barriers[1]);
    }
    barrier_sync();
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
  log.info("return memory key %#x\n", key);
  return memoryKeyCounter.put(key);
}
size_t acquireMemoryKey() {
  auto key = memoryKeyCounter.get();
  log.info("acquire memory key %#x\n", key);
  return key;
}

KernelMappedBuffers mapn(size_t memoryKey, compile_op::CompileContext& ctx, std::string name, Vector<uint32_t> ranks,
    size_t bytesPerConcurrencyIndex) {
  bytesPerConcurrencyIndex = std::max(std::bit_ceil(bytesPerConcurrencyIndex), (size_t)16);
  log.info("mapn %s\n", name);
  auto& x = ctx.cachedBuffers[""];
  AllocatedArray* local = nullptr;
  for (auto& v : x) {
    if (v->itembytes == bytesPerConcurrencyIndex && memoryKeyCounter.isFree(ctx.group, &*v)) {
      log.info("reuse free %d bytes for %s\n", bytesPerConcurrencyIndex, name);
      local = &*v;
      break;
    }
  }
  if (!local) {
    log.info("allocate new %d bytes for %s\n", bytesPerConcurrencyIndex, name);
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
    const compile_op::Graph& graph, compile_op::CompileContext& ctx) {
  using namespace ptx;

  uint32_t blockSize = config.blockSize;
  uint32_t numBlocks = config.gridSize;

  size_t rank = group->rank;

  auto rv = std::make_shared<KernelHandle>();
  rv->memoryKey = acquireMemoryKey();

  Module mod;
  mod.target = target;
  mod.topLevelComments.push_back(fmt::sprintf("block size: %d  grid size: %d", config.blockSize, config.gridSize));
  mod.topLevelComments.push_back(fmt::sprintf("copy engine: %s", config.copyEngine));
  mod.topLevelComments.push_back(fmt::sprintf("shared memory: %d", config.sharedMemory));
  setModule(&mod);
  if (false) {
    mod.addGlobal(".u8", 4 * 1024 * 1024, "debug_buf");
    mod.addGlobal(".u32", 1, "debug_warp_counter");
    mod.addGlobal(".u64", 1, "debug_clock_ref");
  }

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

  size_t ranksStride = 0;

  for (uint32_t r : ranks) {
    ctx.send(r, ranks.size());
  }
  for (uint32_t r : ranks) {
    size_t s;
    ctx.receive(r, s);
    ranksStride = std::max(ranksStride, s);
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

  auto map = [&](std::string name, size_t bytes) {
    return mapn(rv->memoryKey, ctx, name, ranks, bytes);
  };

  size_t addrAddrsBlockStride = graph.localCudaTensorMappings.size();
  auto addrAddrs = map("addresses", sizeof(uint64_t) * addrAddrsBlockStride * numBlocks);
  auto syncAddrs = map("syncs", sizeof(uint32_t) * ranksStride * numBlocks);
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

    u64 parityAddr = addGlobalVar(4, maxConcurrency * numBlocks * 4, "parity") +
                     widen(4 * numBlocks * concurrencyIndex + 4 * blockIndex);
    u32 parityBit = 0;
    IF_D(threadIndex == 0) {
      parityBit = ld_global_u32(parityAddr) ^ 1;
      st_global_u32(parityAddr, parityBit);
    }
    IF(bar_red_or(0, blockSize, parityBit != 0)) {
      parityBit = 1;
    }

    u64 parityTag = widen(parityBit) << 63;

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

    std::unique_ptr<EmitMultiCopy> emit;
    if (config.copyEngine == "simple") {
      emit = std::make_unique<EmitSimple>(config, blockIndex, warpIndex, laneIndex);
    } else if (config.copyEngine == "buffered") {
      emit = std::make_unique<EmitBuffered>(config, blockIndex, warpIndex, laneIndex);
    } else if (config.copyEngine == "lockstep") {
      emit = std::make_unique<EmitLockstep>(config, blockIndex, warpIndex, laneIndex);
    } else {
      CHECK(false);
    }
    CHECK(emit != nullptr);

    struct AddrSetup {
      uint64_t base;
      uint32_t itembytes;
      uint32_t paramOffset;
      uint32_t stride;
      uint32_t slot;
    };

    struct SyncSetup {
      uint64_t stbase;
      uint32_t itembytes;
      uint32_t ldoffset;
    };

    std::vector<SyncSetup> syncsetup;
    for (size_t i : indices(ranks)) {
      auto& r = syncAddrs.remote[i];
      SyncSetup s;
      s.stbase = r.base + sizeof(uint32_t) * myPeerIndex[i];
      s.itembytes = r.itembytes;
      s.ldoffset = sizeof(uint32_t) * i;
      syncsetup.push_back(s);
    }

    u32 csyncaddr = addConst32(syncsetup, "syncsetup");

    std::vector<AddrSetup> addrsetup;
    for (size_t i : indices(graph.remoteCudaTensorMappings)) {
      auto& v = graph.remoteCudaTensorMappings[i];
      auto& r = addrAddrs.remote[rankIndex(v.rank)];

      AddrSetup s;
      s.base = r.base;
      s.itembytes = r.itembytes;
      s.paramOffset = sizeof(uint64_t) * i;
      s.stride = sizeof(uint64_t) * v.stride;
      s.slot = sizeof(uint64_t) * v.destinationSlot;
      addrsetup.push_back(s);
    }
    u32 caddrsetup = addConst32(addrsetup, "addrsetup");
    for (uint32_t i : range((graph.remoteCudaTensorMappings.size() + blockSize - 1) / blockSize)) {
      u32 index = blockSize * i + threadIndex;
      IF(index < graph.remoteCudaTensorMappings.size()) {
        u32 caddr = caddrsetup + index * sizeof(AddrSetup);
        u64 base = ld_const_u64(caddr);
        u32 itembytes = ld_const_u32(caddr + 8);
        u32 paramOffset = ld_const_u32(caddr + 12);
        u32 stride = ld_const_u32(caddr + 16);
        u32 slot = ld_const_u32(caddr + 20);
        u64 addr = base + widen(itembytes * concurrencyIndex + slot + stride * blockIndex);
        st_global_relaxed_sys_u64(addr, ld_param_u64(mappedAddrs + paramOffset, 0) | parityTag);
      }
    }

    Vector<Edge> localCopies;

    for (auto& v : graph.cudaEdges) {
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

    bool nowaits = true;
    bool hasRemoteReads = false;
    bool hasRemoteWrites = false;
    for (const Edge& e : localCopies) {
      for (const Node& n : e.sources) {
        if (!n.filled) {
          nowaits = false;
        }
        if (n.rank != rank) {
          hasRemoteReads = true;
        }
      }
      for (const Node& n : e.destinations) {
        if (n.rank != rank) {
          hasRemoteWrites = true;
        }
      }
    }

    auto execSync = [&](uint32_t stepOffset) {
      for (uint32_t i : range((syncsetup.size() + blockSize - 1) / blockSize)) {
        u32 index = blockSize * i + threadIndex;
        IF_D(index < syncsetup.size()) {
          u32 addr = csyncaddr + index * sizeof(SyncSetup);
          u64 stbase = ld_const_u64(addr);
          u32 itembytes = ld_const_u32(addr + 8);
          u32 ldoffset = ld_const_u32(addr + 12);
          u32 blockStride = sizeof(uint32_t) * ranksStride * blockIndex;
          st_global_relaxed_sys_u32(stbase + widen(itembytes * concurrencyIndex + blockStride), stepValue + stepOffset);
          u64 ldaddr = concurrency(syncAddrs.local) + widen(ldoffset + blockStride);
          WHILE(ld_global_relaxed_sys_u32(ldaddr) < stepValue + stepOffset) {}
        }
      }
    };

    // auto entrySync = [&]() {
    //   execSync(0);
    //   barrier_sync();
    // };
    auto exitSync = [&]() {
      if (hasRemoteWrites) {
        // I am quite certain that this fence.release.sys is unnecessary on current archs.
        // In practice, removing it is fine - correctness tests pass.
        // Basically, because remote nvlink memory is not cached, local thread ordering 
        // (provided by cp.async.bulk.wait_group) means that nvlink data must also
        // be ordered, and we only require the order of cp.async.bulk and exitSync to be preserved.
        // However, technically, the fence is required, and it is possible that a future
        // multi-device-cache-coherent gpu could require it.
        // It has a small performance penalty, as it makes us actually wait for the writes
        // to be fully landed and visible on the remote gpu.
        IF(threadIndex == 0) {
          fence_release_sys();
        }
      }
      barrier_sync();
      execSync(1);
    };

    u64 addrAddrsLocalBase = concurrency(addrAddrs.local) + widen(sizeof(uint64_t) * addrAddrsBlockStride * blockIndex);

    auto getaddr = [&](const Node& n) {
      if (n.rank == rank) {
        return ld_param_u64(inputAddrs, sizeof(uint64_t) * n.tensorIndex) + n.offset;
      } else {
        u64 r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + sizeof(uint64_t) * findLocalMapping(n));
        WHILE((r & (1ull << 63)) != parityTag) {
          r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + sizeof(uint64_t) * findLocalMapping(n));
        }
        r &= ~(1ull << 63);
        return r + n.offset;
      }
    };

    struct NodeSetup {
      uint32_t slot;
      uint32_t offsetLow;
      uint32_t offsetHigh;
    };

    size_t maxDestinations = 0;
    for (const Edge& e : localCopies) {
      maxDestinations = std::max(maxDestinations, e.destinations.size());
    }

    std::vector<NodeSetup> nodesetup;
    for (const Edge& e : localCopies) {
      for (auto& n : e.destinations) {
        NodeSetup s;
        s.offsetLow = n.offset;
        s.offsetHigh = n.offset >> 32;
        if (n.rank == rank) {
          s.slot = sizeof(uint64_t) * n.tensorIndex | 1;
        } else {
          s.slot = sizeof(uint64_t) * findLocalMapping(n);
        }
        nodesetup.push_back(s);
      }
    }

    u32 cnodesetupaddr = addConst32(nodesetup, "nodesetup");

    auto nodeaddr = [&](u32 index) {
      u32 caddr = cnodesetupaddr + index * sizeof(NodeSetup);
      u32 slot = ld_const_u32(caddr);
      u32 offsetlo = ld_const_u32(caddr + 4);
      u32 offsethi = ld_const_u32(caddr + 8);
      u64 r;
      IF_D((slot & 1) != 0) {
        r = ld_param_u64(inputAddrs + (slot & ~1u), 0);
      }
      ELSE {
        // WHILE(((r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + widen(slot))) & (1ull << 63)) != parityTag) {}
        r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + widen(slot));
        WHILE((r & (1ull << 63)) != parityTag) {
          r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + widen(slot));
        }
        r &= ~(1ull << 63);
      }
      r += widen(offsetlo) | (widen(offsethi) << 32);
      return r;
    };

    auto setdst = [&](const Edge& e, u32 nodeoffset, Vector<u64>& dstaddrs) {
      u32 ndst = 0;
      if (e.destinations.size() == 1) {
        for (size_t i : indices(dstaddrs)) {
          u32 index = laneIndex + 32 * i;
          for (size_t z : indices(e.destinations)) {
            IF_D(index == z) {
              dstaddrs.at(i) = getaddr(e.destinations[z]);
              ndst += 1;
            }
          }
        }
      } else {
        for (size_t i : indices(dstaddrs)) {
          u32 index = laneIndex + 32 * i;
          IF_D(index < e.destinations.size()) {
            dstaddrs[i] = nodeaddr(nodeoffset + index);
            ndst += 1;
          }
        }
      }
      return ndst;
    };

    if (nowaits) {

      CHECK(localCopies.size() == 1);

      u32 nodeoffset = 0;
      for (const Edge& e : localCopies) {
        CHECK(e.sources.size() == 1);
        CHECK(e.destinations.size() >= 1);

        u64 srcaddr = getaddr(e.sources[0]);
        Vector<u64> dstaddrs((e.destinations.size() + 31) / 32);
        u32 ndst = setdst(e, nodeoffset, dstaddrs);
        nodeoffset += e.destinations.size();

        if (hasRemoteReads) {
          IF_D(threadIndex == 0) {
            fence_acquire_sys();
          }
        }

        emit->emit(srcaddr, dstaddrs, ndst, e.bytes);
      }

    } else {
      Label copy;
      u64 srcaddr = 0;
      Vector<u64> dstaddrs((maxDestinations + 31) / 32);
      u32 ndst = 0;
      u32 bytes = 0;
      u32 index = 0;

      std::vector<Label> rets;

      WHILE(true) {

        u32 numDone = 0;

        u32 nextNodeOffset = 0;
        for (const Edge& e : localCopies) {
          CHECK(e.sources.size() == 1);
          CHECK(e.destinations.size() >= 1);
          auto& src = e.sources[0];

          u32 nodeoffset = nextNodeOffset;
          nextNodeOffset += e.destinations.size();

          Label skip;
          Label sync;
          u32 pred = 0;
          IF_D(threadIndex == 0) {
            u32 ready = 1;
            if (!src.filled) {
              auto it = std::ranges::find(localNodes, src.id);
              CHECK(it != localNodes.end());
              IF(ld_global_relaxed_sys_u32(concurrency(depFilled.local) +
                                           widen(sizeof(uint32_t) * ((it - localNodes.begin()) +
                                                                        localNodes.size() * blockIndex))) < stepValue) {
                ready = 0;
              }
            }

            IF(ready != 0) {
              auto& dst0 = e.destinations[0];
              auto it = std::ranges::find(localNodes, dst0.id);
              CHECK(it != localNodes.end());
              IF(ld_global_relaxed_sys_u32(
                     concurrency(depFilled.local) +
                     widen(sizeof(uint32_t) * ((it - localNodes.begin()) + localNodes.size() * blockIndex))) ==
                  stepValue) {
                numDone += 1;
              }
              ELSE {
                pred = 1;
              }
            }
          }
          GOTO_IF_NOT(bar_red_or(15, blockSize, pred != 0), skip);

          srcaddr = getaddr(src);
          ndst = setdst(e, nodeoffset, dstaddrs);
          bytes = e.bytes;
          index = rets.size();
          //GOTO(copy);
          if (hasRemoteReads) {
            IF_D(threadIndex == 0) {
              fence_acquire_sys();
            }
          }
          emit->emit(srcaddr, dstaddrs, ndst, bytes);
          Label ret;
          LABEL(ret);
          rets.push_back(std::move(ret));

          IF_D(threadIndex == 0) {
            // fixme: this fence is only needed if signalranks has a != rank member
            fence_release_sys();
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
                  st_global_relaxed_sys_u32(
                      concurrency(depFilled.remote.at(rankIndex(r))) +
                          widen(sizeof(uint32_t) * (it->second.index + it->second.stride * blockIndex)),
                      stepValue);
                }
              }
            }
          }
          numDone += 1;
          LABEL(skip);
          // warp_sync();
        }

        IF(bar_red_or(14, blockSize, numDone == localCopies.size())) {
          BREAK;
        }
      }

      CHECK(!rets.empty());

      SKIP {
        LABEL(copy);
        if (hasRemoteReads) {
          IF_D(threadIndex == 0) {
            fence_acquire_sys();
          }
        }
        //emit->emit(srcaddr, dstaddrs, ndst, bytes);
        barrier_sync(9);

        //PRED(index != 0) trap();
        brx_idx(index, rets);
      }
    }

    exitSync();
    ret();
  }

  std::string ptx = mod.finalize();
  rv->ptx = ptx;
  return rv;
}

} // namespace moodist
