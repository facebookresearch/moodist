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
  virtual size_t groupSize() {
    return 32;
  }
  virtual ~EmitMultiCopy() = default;
  virtual void emit(u64 src, Vector<u64> dst, u32 bytes) = 0;
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

  void emit(u64 src, Vector<u64> dst, u32 bytes) override {
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
      PRED(dst[i] % 16 != 0) trap();
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
        IF_D(dst[i] != 0) {
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
  uint32_t blockSharedSize =
      std::max(inputSharedSize / blockSharedAlignment * blockSharedAlignment, blockSharedAlignment);
  uint32_t bufferSize = blockSharedSize / numBuffers;

  u32 bufferBaseAddr = addShared32(16, numBuffers* bufferSize);
  u32 barrierBaseAddr = addShared32(8, numBuffers * 8);

  u32 bufferIndex = warpIndex;

  u32 buffer = bufferBaseAddr + bufferIndex * bufferSize;
  u32 barrier = barrierBaseAddr + bufferIndex * 8;

  void sync() {
    profiling.enter("emit warp_sync");
    warp_sync();
    profiling.leave();
  }

  void emit(u64 src, Vector<u64> dst, u32 bytes) override {
    sync();
    Label alldone;
    Value isLeader = laneIndex == 0;
    IF_D(isLeader) {
      mbarrier_init(barrier, 1);
    }

    PRED(bytes % 16 != 0) trap();
    PRED(src % 16 != 0) trap();
    for (size_t i : indices(dst)) {
      PRED(dst[i] % 16 != 0) trap();
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
      profiling.enter("emit wait for write");
      cp_async_bulk_wait_group_read(0);
      sync();
      IF_D(isLeader) {
        profiling.enter("src copy", size);
        mbarrier_expect_tx(barrier, size);
        mbarrier_arrive_noComplete(barrier);
        cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
        mbarrier_wait_parity(barrier, parity);
        profiling.leave();
      }
      sync();

      for (size_t i : indices(dst)) {
        IF_D(dst[i] != 0) {
          profiling.enter("dst copy", size);
          u64 dstaddr = dst[i] + widen(offset);
          cp_async_bulk_global_shared(dstaddr, buffer, size);
          profiling.leave();
        }
      }

      cp_async_bulk_commit_group();
      offset += numBlocks * blockSharedSize;
    }
    GOTO(loop);
    LABEL(alldone);
    profiling.enter("final wait for write");
    cp_async_bulk_wait_group(0);
    profiling.leave();
    IF_D(isLeader) {
      mbarrier_inval(barrier);
    }
    profiling.enter("emit exit barrier_sync");
    barrier_sync(3);
    profiling.enter("emit exit barrier_sync wait");
    profiling.leave();
  }
};

struct EmitGrouped : EmitMultiCopy {
  const KernelConfig& config;
  u32 blockIndex;
  u32 warpIndex;
  u32 laneIndex;

  EmitGrouped(const KernelConfig& config, const u32& blockIndex, const u32& warpIndex, const u32& laneIndex)
      : config(config), blockIndex(blockIndex), warpIndex(warpIndex), laneIndex(laneIndex) {
    CHECK(numBuffers > 0);
    CHECK(bufferSize > 0);
    CHECK(config.blockSize % groupSize() == 0);
    CHECK(config.blockSize / groupSize() < 15);
  }

  size_t groupSize() override {
    return 128;
  }

  uint32_t numBlocks = config.gridSize;
  uint32_t inputSharedSize = config.sharedMemory;

  uint32_t numBuffers = config.blockSize / groupSize();
  uint32_t blockSharedAlignment = numBuffers * 256;
  uint32_t blockSharedSize =
      std::max(inputSharedSize / blockSharedAlignment * blockSharedAlignment, blockSharedAlignment);
  uint32_t bufferSize = blockSharedSize / numBuffers;

  u32 bufferBaseAddr = addShared32(16, numBuffers* bufferSize);
  u32 barrierBaseAddr = addShared32(8, numBuffers * 8);

  u32 bufferIndex = warpIndex / (groupSize() / 32);

  u32 buffer = bufferBaseAddr + bufferIndex * bufferSize;
  u32 barrier = barrierBaseAddr + bufferIndex * 8;

  void sync() {
    profiling.enter("emit sync");
    barrier_sync(bufferIndex, groupSize());
    profiling.enter("emit sync wait");
    profiling.leave();
  }

  void emit(u64 src, Vector<u64> dst, u32 bytes) override {
    sync();
    Label alldone;
    u32 myIndex = (warpIndex * 32 + laneIndex) % groupSize();
    Value isLeader = myIndex == 0;
    IF_D(isLeader) {
      mbarrier_init(barrier, 1);
    }

    PRED(bytes % 16 != 0) trap();
    PRED(src % 16 != 0) trap();
    for (size_t i : indices(dst)) {
      PRED(dst[i] % 16 != 0) trap();
    }

    u32 offset = (bufferIndex * numBlocks + blockIndex) * bufferSize;

    u64 srcaddr = src + widen(offset);
    u32 size = min_u32(bytes - offset, bufferSize);

    GOTO_IF(offset >= bytes, alldone);

    PRED(isLeader) {
      profiling.enter("initial src copy");
      mbarrier_expect_tx(barrier, size);
      mbarrier_arrive_noComplete(barrier);
      cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
      profiling.leave();
    }

    u32 parity = 0;
    WHILE(true) {
      IF_D(isLeader) {
        profiling.enter("wait for read");
        mbarrier_wait_parity(barrier, parity);
      }
      sync();
      for (size_t i : indices(dst)) {
        u64 dstaddr = dst[i] + widen(offset);
        PRED(dst[i] != 0) {
          profiling.enter("write", size);
          cp_async_bulk_global_shared(dstaddr, buffer, size);
          profiling.leave();
        }
      }
      offset += numBlocks * blockSharedSize;
      srcaddr += numBlocks * blockSharedSize;
      size = min_u32(bytes - offset, bufferSize);
      IF(offset >= bytes) {
        BREAK;
      }
      parity ^= 1;
      PRED(isLeader) {
        profiling.enter("prepare read");
        mbarrier_expect_tx(barrier, size);
        mbarrier_arrive_noComplete(barrier);
      }
      profiling.enter("wait for write");
      cp_async_bulk_commit_group();
      cp_async_bulk_wait_group_read(0);
      sync();
      PRED(isLeader) {
        profiling.enter("read");
        cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
        profiling.leave();
      }
    }

    LABEL(alldone);
    IF_D(isLeader) {
      mbarrier_inval(barrier);
    }
    profiling.enter("final wait for write");
    cp_async_bulk_wait_group(0);
    profiling.leave();
    profiling.enter("emit exit barrier_sync");
    barrier_sync(15);
    profiling.enter("emit exit barrier_sync wait");
    profiling.leave();
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

  void emit(u64 src, Vector<u64> dst, u32 bytes) override {
    Value isLeader = laneIndex == 0;
    IF_D(isLeader) {
      mbarrier_init(barriers[0], 1);
      mbarrier_init(barriers[1], 1);
    }

    PRED(bytes % 16 != 0) trap();
    PRED(src % 16 != 0) trap();
    for (size_t i : indices(dst)) {
      PRED(dst[i] % 16 != 0) trap();
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
            IF_D(dst[i] != 0) {
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
  mod.topLevelComments.push_back(fmt::sprintf("copy engine: %s", config.copyEngine));
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
    std::stable_partition(localCopies.begin(), localCopies.end(), [](const Edge& e) {
      return e.sources.empty();
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
    u64 tables = addGlobalData(16, tableData.data(), tableData.size(), "tables");

    // The scan for ready edges keeps a done bit per edge and the selected edge in shared memory, taken from the
    // copy engine's share
    const uint32_t numEdges = localCopies.size();
    const uint32_t numDoneWords = (numEdges + 31) / 32;
    const uint32_t scanShared = nowaits ? 0 : (4 * numDoneWords + 8 + 15) / 16 * 16;
    KernelConfig engineConfig = config;
    CHECK(engineConfig.sharedMemory > scanShared);
    engineConfig.sharedMemory -= scanShared;

    std::unique_ptr<EmitMultiCopy> emit;
    if (config.copyEngine == "simple") {
      emit = std::make_unique<EmitSimple>(engineConfig, blockIndex, warpIndex, laneIndex);
    } else if (config.copyEngine == "grouped") {
      emit = std::make_unique<EmitGrouped>(engineConfig, blockIndex, warpIndex, laneIndex);
    } else if (config.copyEngine == "buffered") {
      emit = std::make_unique<EmitBuffered>(engineConfig, blockIndex, warpIndex, laneIndex);
    } else if (config.copyEngine == "lockstep") {
      emit = std::make_unique<EmitLockstep>(engineConfig, blockIndex, warpIndex, laneIndex);
    } else {
      CHECK(false);
    }
    CHECK(emit != nullptr);

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

    size_t groupSize = emit->groupSize();
    const size_t numDstSlots = (maxDestinations + groupSize - 1) / groupSize;

    // Slot i of this thread takes destination gindex(i) of an edge. In groups of several warps, consecutive
    // destinations go to different warps (lane l of warp w of the group takes destination (w + 1) % warps +
    // warps * l), so that all warps of the group issue the stores of a multicast edge.
    auto gindex = [&](size_t i) {
      if (groupSize == 32) {
        return (laneIndex + groupSize * i);
      }
      u32 index = (threadIndex + 32) % groupSize;
      return (index / 32 + index % 32 * (groupSize / 32)) + groupSize * i;
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
    // Copy the edge at `desc` (the same for all threads) with the whole block: flags, bytes and srcaddr taken
    // from its words (copied into registers of their own, so the words can be reloaded), dstaddrs resolved
    auto copyEdge = [&](const u64& desc, const u32& flags, const u32& bytes, const u64& srcaddr) {
      // Every thread has read the source's address itself (remote sources), and any may issue loads
      IF((flags & edgeAcquire) != 0) {
        fence_acquire_sys();
      }
      emit->emit(srcaddr, dstaddrs, bytes);
      IF_D(isThreadZero & ((flags & edgeSignal) != 0)) {
        signalEdge(desc);
      }
    };

    if (numEdges == 0) {
    } else if (nowaits) {
      // In order, with the words of the next edge loaded while the current one is copied
      EdgeWords next;
      loadEdge(next, edgeTableAddr, nullptr);
      FOR(u32 e = 0, e < numEdges, e += 1) {
        u32 flags = next.flags;
        u32 bytes = next.bytes;
        u64 srcaddr = resolveEdge(next);
        Value more = e + 1 < numEdges;
        loadEdge(next, edgeTableAddr + widen((e + 1) * edgeStride), &more);
        copyEdge(edgeTableAddr + widen(e * edgeStride), flags, bytes, srcaddr);
      }
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
          copyEdge(desc, w.flags, w.bytes, srcaddr);
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
