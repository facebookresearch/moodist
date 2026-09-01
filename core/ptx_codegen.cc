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

  void start(const KernelConfig& config, u32 stepValue, u64 addrData) {
    u64 addr = addGlobalVar(4, 4, "profiling_atomic");
    // u64 addrClock = addGlobalVar(8, 8, "profiling_base_clock");
    u64 addrStepValue = addGlobalVar(4, 4, "profiling_step_value");
    // IF_D(threadIdx_x() == 0) {
    //   IF(atom_global_relaxed_inc_u32(addr, config.gridSize - 1) == config.gridSize - 1) {
    //     // st_global_u64(addrClock, clock64());
    //     // fence_release_gpu();
    //     st_global_relaxed_sys_u32(addrStepValue, stepValue);
    //   }
    // }
    // u64 myclock;
    // WHILE(true) {
    //   myclock = clock64();
    //   IF(ld_global_relaxed_sys_u32(addrStepValue) == stepValue) {
    //     BREAK;
    //   }
    //   nanosleep(10);
    // }
    // fence_acquire_gpu();
    u64 myclock = clock64();
    // atomicOffset = ld_global_relaxed_sys_u64(addrClock) - myclock;
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
    // PRED(o + 16 > maxBytesPerThread) trap();
    u32 a = *dataOffset + o;
    st_global_u64(*addrData + widen(a), clock64() + *clockOffset + (nameToIndex[name] << 48));
    // st_global_u32(*addrData + widen(a) + 8, nameToIndex[name]);
    // st_global_u32(*addrData + widen(a + 12), arg ? *arg : 0);
    *nextEntryIndex = *nextEntryIndex + 1;

    // fence_release_gpu();
    // fence_acquire_gpu();
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
          // cp_async_bulk_commit_group();
          // cp_async_bulk_wait_group_read(0);
          profiling.leave();
        }
      }

      // for (size_t i : indices(dst)) {
      //   for (uint32_t r : range(32)) {
      //     IF_D(laneIndex == r) {
      //       IF_D(dst[i] != 0) {
      //         profiling.enter("dst copy", size);
      //         u64 dstaddr = dst[i] + widen(offset);
      //         cp_async_bulk_global_shared(dstaddr, buffer, size);
      //         // cp_async_bulk_commit_group();
      //         // cp_async_bulk_wait_group_read(0);
      //         profiling.leave();
      //       }
      //     }
      //     warp_sync();
      //   }
      // }
      // for (size_t i : indices(dst)) {
      //   for (uint32_t r : range(32)) {
      //     IF_D(laneIndex == r) {
      //       IF_D(dst[i] != 0) {
      //         profiling.enter("dst copy", size);
      //         u64 dstaddr = dst[i] + widen(offset);
      //         cp_async_bulk_global_shared(dstaddr, buffer, size);
      //         cp_async_bulk_commit_group();
      //         cp_async_bulk_wait_group_read(0);
      //         profiling.leave();
      //       }
      //     }
      //   }
      // }

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
    // return 576 / 2;
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

  u32 writeSyncAddr = addShared32(8, 4 * numBuffers);

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
      st_shared_relaxed_u32(writeSyncAddr + 4 * bufferIndex, 0);
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
      // profiling.enter("commit write");
      cp_async_bulk_commit_group();
      cp_async_bulk_wait_group_read(0);
      sync();
      PRED(isLeader) {
        // profiling.enter("wait for write");
        // cp_async_bulk_wait_group_read(0);
        profiling.enter("read");
        cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
        profiling.leave();
      }
    }

    // Label loop;
    // LABEL(loop);
    // for (uint32_t parity : range(2)) {
    //   IF(offset >= bytes) {
    //     GOTO(alldone);
    //   }
    //   u64 srcaddr = src + widen(offset);
    //   u32 size = min_u32(bytes - offset, bufferSize);
    //   profiling.enter("emit wait for write");
    //   cp_async_bulk_wait_group_read(0);
    //   sync();
    //   IF_D(isLeader) {
    //     profiling.enter("src copy", size);
    //     mbarrier_expect_tx(barrier, size);
    //     mbarrier_arrive_noComplete(barrier);
    //     cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
    //     mbarrier_wait_parity(barrier, parity);
    //     profiling.leave();
    //   }
    //   sync();
    //   // for (size_t i : indices(dst)) {
    //   //   for (uint32_t r : range(groupSize())) {
    //   //     sync();
    //   //     IF_D(myIndex == r) {
    //   //       IF_D(dst[i] != 0) {
    //   //         profiling.enter("dst copy", size);
    //   //         u64 dstaddr = dst[i] + widen(offset);
    //   //         cp_async_bulk_global_shared(dstaddr, buffer, size);
    //   //         cp_async_bulk_commit_group();
    //   //         cp_async_bulk_wait_group_read(0);
    //   //         profiling.leave();
    //   //       }
    //   //     }
    //   //   }
    //   // }
    //   // IF_D(laneIndex == 0) {
    //   //   WHILE (ld_shared_relaxed_u32(writeSyncAddr + 4 * bufferIndex) != myIndex / 32) {
    //   //     nanosleep(10);
    //   //   }
    //   // }
    //   warp_sync();
    //   for (size_t i : indices(dst)) {
    //     IF_D(dst[i] != 0) {
    //       profiling.enter("dst copy", size);
    //       u64 dstaddr = dst[i] + widen(offset);
    //       cp_async_bulk_global_shared(dstaddr, buffer, size);
    //       // cp_async_bulk_commit_group();
    //       profiling.leave();
    //     }
    //   }
    //   // sync();
    //   warp_sync();
    //   cp_async_bulk_commit_group();
    //   offset += numBlocks * blockSharedSize;

    //   // IF_D(laneIndex == 0) {
    //   //   st_shared_relaxed_u32(writeSyncAddr + 4 * bufferIndex, (myIndex / 32 + 1) % (groupSize() / 32));
    //   // }
    // }
    // GOTO(loop);
    LABEL(alldone);
    IF_D(isLeader) {
      mbarrier_inval(barrier);
    }
    // sync();
    profiling.enter("final wait for write");
    cp_async_bulk_wait_group(0);
    profiling.leave();
    // IF_D(isLeader) {
    //   mbarrier_inval(barrier);
    // }
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

    if (false) {
      auto profilingBuffer =
          ctx.group->allocateDevice(Profiling::maxBytesPerThread * config.blockSize * config.gridSize);
      profiling = Profiling();
      profiling.start(config, stepValue, profilingBuffer.cudaPointer);
      rv->profilingData = std::move(profilingBuffer);
      rv->profilingBytesPerThread = Profiling::maxBytesPerThread;
      rv->profilingCountdown = 400;
    }

    u64 indeterminate = addGlobalVar(128, 128);

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
    // IF_D(threadIndex == 0) {
    //   parityBit = ld_global_u32(parityAddr) ^ 1;
    //   st_global_u32(parityAddr, parityBit);
    // }
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

    std::unique_ptr<EmitMultiCopy> emit;
    if (config.copyEngine == "simple") {
      emit = std::make_unique<EmitSimple>(config, blockIndex, warpIndex, laneIndex);
    } else if (config.copyEngine == "grouped") {
      emit = std::make_unique<EmitGrouped>(config, blockIndex, warpIndex, laneIndex);
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
      if (ranks[i] == rank) {
        continue;
      }
      auto& r = syncAddrs.remote[i];
      SyncSetup s;
      s.stbase = r.base + sizeof(uint32_t) * myPeerIndex[i];
      s.itembytes = r.itembytes;
      s.ldoffset = sizeof(uint32_t) * i;
      syncsetup.push_back(s);
    }

    u32 csyncaddr = addConst32(syncsetup, "syncsetup");

    profiling.enter("syncsetup");

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
      // IF_D(index < graph.remoteCudaTensorMappings.size()) {
      //   u32 caddr = caddrsetup + index * sizeof(AddrSetup);
      //   u64 base = ld_const_u64(caddr);
      //   u32 itembytes = ld_const_u32(caddr + 8);
      //   u32 paramOffset = ld_const_u32(caddr + 12);
      //   u32 stride = ld_const_u32(caddr + 16);
      //   u32 slot = ld_const_u32(caddr + 20);
      //   u64 addr = base + widen(itembytes * concurrencyIndex + slot + stride * blockIndex);
      //   st_global_relaxed_sys_u64(addr, ld_param_u64(mappedAddrs + paramOffset, 0) | parityTag);
      // }
      auto pred = index < graph.remoteCudaTensorMappings.size();
      u32 caddr = caddrsetup + index * sizeof(AddrSetup);
      u64 base = predicated(pred, ld_const_u64)(caddr);
      u32 itembytes = predicated(pred, ld_const_u32)(caddr + 8);
      u32 paramOffset = predicated(pred, ld_const_u32)(caddr + 12);
      u32 stride = predicated(pred, ld_const_u32)(caddr + 16);
      u32 slot = predicated(pred, ld_const_u32)(caddr + 20);
      u64 addr = base + widen(itembytes * concurrencyIndex + slot + stride * blockIndex);
      predicated(pred, st_global_relaxed_sys_u64)(
          addr, predicated(pred, ld_param_u64)(mappedAddrs + paramOffset, 0) | parityTag);
    }

    profiling.leave();

    profiling.enter("syncsetup warp_sync");
    warp_sync();
    profiling.leave();

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

    u64 addrAddrsLocalBase = concurrency(addrAddrs.local) + widen(sizeof(uint64_t) * addrAddrsBlockStride * blockIndex);

    auto getaddr = [&](const Node& n) {
      if (n.rank == rank) {
        return ld_param_u64(inputAddrs, sizeof(uint64_t) * n.tensorIndex) + n.offset;
      } else {
        u64 r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + sizeof(uint64_t) * findLocalMapping(n));
        WHILE((r & (1ull << 63)) != parityTag) {
          nanosleep(10);
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
      profiling.enter("nodeaddr");
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
          nanosleep(10);
          r = ld_global_relaxed_sys_u64(addrAddrsLocalBase + widen(slot));
        }
        r &= ~(1ull << 63);
      }
      r += widen(offsetlo) | (widen(offsethi) << 32);
      profiling.leave();
      return r;
    };

    auto nodeaddrpred = [&](Value pred, u32 index) {
      profiling.enter("nodeaddr");
      u32 caddr = cnodesetupaddr + index * sizeof(NodeSetup);
      u32 slot = predicated(pred, ld_const_u32)(caddr);
      u32 offsetlo = predicated(pred, ld_const_u32)(caddr + 4);
      u32 offsethi = predicated(pred, ld_const_u32)(caddr + 8);
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

    std::vector<uint32_t> shuffledIndices;
    size_t shuffledEntriesPerGroup = (maxDestinations + groupSize - 1) / groupSize * groupSize;
    CHECK(config.gridSize * config.blockSize % groupSize == 0);
    size_t numGroupsGlobally = config.gridSize * config.blockSize / groupSize;

    for (size_t i : range(numGroupsGlobally)) {
      std::vector<uint32_t> v(shuffledEntriesPerGroup);
      std::ranges::iota(v, 0);
      std::ranges::shuffle(v, getRng());
      for (auto& x : v) {
        shuffledIndices.push_back(x);
      }
    }

    // uint32_t randomizedConstant = 0;
    // uint32_t randomizedOffset = 0;

    // while (true) {
    //   uint32_t o = random((uint32_t)0, (uint32_t)0xffffffff);
    //   uint32_t x = random((uint32_t)0, (uint32_t)0xffffffff);
    //   uint32_t c = random((uint32_t)0, (uint32_t)0xffffffff);
    //   bool failed = false;
    //   std::vector<uint32_t> v;
    //   std::vector<uint32_t> collisions(shuffledEntriesPerGroup * shuffledEntriesPerGroup);
    //   for (size_t i : range(numGroupsGlobally)) {
    //     v.clear();
    //     for (size_t n : range(shuffledEntriesPerGroup)) {
    //       // //uint32_t x = ((uint64_t)(uint32_t)((((n + shuffledEntriesPerGroup * i + o) * 0x9e3779b9) ^ x) * c) *
    //       // shuffledEntriesPerGroup) >> 32; uint32_t x = ((uint64_t)(uint32_t)((n + shuffledEntriesPerGroup * i) *
    //       c) *
    //       // shuffledEntriesPerGroup) >> 32;
    //       // //uint32_t x = ((__uint128_t)((n + shuffledEntriesPerGroup * i) * 0x9E3779B97F4A7C15ul) *
    //       // shuffledEntriesPerGroup) >> 64;
    //       // //CHECK(!std::ranges::contains(v, x));

    //       uint32_t halfBits = (std::bit_width(shuffledEntriesPerGroup - 1) + 1) / 2;
    //       uint32_t halfMask = (1 << halfBits) - 1;

    //       uint32_t x = n;
    //       uint32_t seed = (i + o) * 0x9e3779b9;

    //       do {
    //         uint32_t lo = x & halfMask;
    //         uint32_t hi = (x >> halfBits) & halfMask;
    //         for (uint32_t round : range(4)) {
    //           hi ^= ((lo * 0x45d9f3b + seed + round) >> 4) & halfMask;
    //           std::swap(lo, hi);
    //         }
    //         x = (hi << halfBits) | lo;
    //       } while (x >= shuffledEntriesPerGroup);

    //       CHECK(x < shuffledEntriesPerGroup);
    //       if (std::ranges::contains(v, x)) {
    //         CHECK(false);
    //         failed = true;
    //         break;
    //       }
    //       collisions[v.size() * shuffledEntriesPerGroup + x] += 1;
    //       v.push_back(x);
    //     }
    //     if (failed) {
    //       break;
    //     }
    //   }
    //   if (failed) {
    //     continue;
    //   }
    //   uint32_t nCollisions = *std::ranges::max_element(collisions);
    //   log.info("collisions: %d\n", nCollisions);
    //   if (nCollisions > 32) {
    //     continue;
    //   }

    //   log.info("Success for %#x %#x %#x! indices: %s\n", o, x, c, fmt::to_string(fmt::join(v, ", ")));
    //   // CHECK(false);

    //   for (size_t i : range(numGroupsGlobally)) {
    //     v.clear();
    //     for (size_t n : range(shuffledEntriesPerGroup)) {
    //       uint32_t x = ((uint64_t)(uint32_t)((n + shuffledEntriesPerGroup * i) * c) * shuffledEntriesPerGroup) >> 32;
    //       // uint32_t x = ((uint64_t)(uint32_t)((((n + shuffledEntriesPerGroup * i + o) * 0x9e3779b9) ^ x) * c) *
    //       // shuffledEntriesPerGroup) >> 32;
    //       v.push_back(x);
    //     }
    //     log.info("For %d: %s\n", i, fmt::to_string(fmt::join(v, ", ")));
    //   }
    //   CHECK(false);
    //   randomizedConstant = c;
    //   break;
    // }

    // while (true) {
    //   uint32_t o = random((uint32_t)0, (uint32_t)0xffffffff);
    //   uint32_t x = random((uint32_t)0, (uint32_t)0xffffffff);
    //   uint32_t c = random((uint32_t)0, (uint32_t)0xffffffff);
    //   bool failed = false;
    //   std::vector<uint32_t> v;
    //   std::vector<uint32_t> collisions(shuffledEntriesPerGroup * shuffledEntriesPerGroup);
    //   for (size_t i : range(numGroupsGlobally)) {
    //     v.clear();
    //     for (size_t n : range(shuffledEntriesPerGroup)) {
    //       uint32_t x = ((uint64_t)(uint32_t)((n + o) * c) * shuffledEntriesPerGroup) >> 32;

    //       x = (x + i) % shuffledEntriesPerGroup;

    //       CHECK(x < shuffledEntriesPerGroup);
    //       if (std::ranges::contains(v, x)) {
    //         failed = true;
    //         break;
    //       }
    //       collisions[v.size() * shuffledEntriesPerGroup + x] += 1;
    //       v.push_back(x);
    //     }
    //     if (failed) {
    //       break;
    //     }
    //   }
    //   if (failed) {
    //     continue;
    //   }
    //   uint32_t nCollisions = *std::ranges::max_element(collisions);
    //   log.info("collisions: %d\n", nCollisions);
    //   if (nCollisions > 32) {
    //     continue;
    //   }

    //   log.info("Success for %#x %#x %#x! indices: %s\n", o, x, c, fmt::to_string(fmt::join(v, ", ")));
    //   // CHECK(false);

    //   for (size_t i : range(numGroupsGlobally)) {
    //     v.clear();
    //     for (size_t n : range(shuffledEntriesPerGroup)) {
    //       uint32_t x = ((uint64_t)(uint32_t)((n + o) * c) * shuffledEntriesPerGroup) >> 32;
    //       x = (x + i) % shuffledEntriesPerGroup;
    //       // uint32_t x = ((uint64_t)(uint32_t)((((n + shuffledEntriesPerGroup * i + o) * 0x9e3779b9) ^ x) * c) *
    //       // shuffledEntriesPerGroup) >> 32;
    //       v.push_back(x);
    //     }
    //     log.info("For %d: %s\n", i, fmt::to_string(fmt::join(v, ", ")));
    //   }
    //   //CHECK(false);
    //   randomizedConstant = c;
    //   randomizedOffset = o;
    //   break;
    // }

    // u32 shuffledIndicesAddr = addConst32(shuffledIndices, "shuffledIndices");

    auto setdst = [&](const Edge& e, u32 nodeoffset, Vector<u64>& dstaddrs) {
      auto gindex = [&](size_t i) {
        if (groupSize == 32) {
          return (laneIndex + groupSize * i);
          // return (threadIndex % groupSize + groupSize * i + blockIndex) % shuffledEntriesPerGroup;
        }
        // CHECK(false);
        //  CHECK(groupSize - 1 + groupSize * i +
        //        (shuffledEntriesPerGroup *
        //                (((config.gridSize - 1) * config.blockSize + (config.blockSize - 1)) / groupSize) <
        //            shuffledIndices.size()));

        // u32 index = narrow((widen((threadIndex % groupSize + groupSize * i + randomizedOffset) * randomizedConstant)
        // * shuffledEntriesPerGroup) >> 32); index = (index + ((blockIndex * config.blockSize + threadIndex) /
        // groupSize)) % shuffledEntriesPerGroup; return index;

        // u32 index = (threadIndex % groupSize + groupSize * i) +
        //             (shuffledEntriesPerGroup * ((blockIndex * config.blockSize + threadIndex) / groupSize));
        // // return narrow((widen((threadIndex % groupSize + groupSize * i)) * shuffledEntriesPerGroup) >> 32);
        // // return ld_const_u32(shuffledIndicesAddr + 4 * index);
        // return narrow((widen(index * randomizedConstant) * shuffledEntriesPerGroup) >> 32);
        // return narrow(((widen(index) * 0x9e3779b9) * shuffledEntriesPerGroup) >> 32);

        // return (warpIndex + laneIndex * (config.blockSize / 32)) % groupSize + groupSize * i;
        u32 index = (threadIndex + 32) % groupSize;
        return (index / 32 + index % 32 * (groupSize / 32)) + groupSize * i;
        // return threadIndex % groupSize + groupSize * i;
        //  return index * 1234;
      };
      if (e.destinations.size() == 1 && false) {
        for (size_t i : indices(dstaddrs)) {
          // u32 index = laneIndex + 32 * i;
          u32 index = gindex(i);
          for (size_t z : indices(e.destinations)) {
            IF_D(index == z) {
              dstaddrs[i] = getaddr(e.destinations[z]);
            }
            ELSE {
              dstaddrs[i] = 0;
            }
          }
        }
      } else {
        for (size_t i : indices(dstaddrs)) {
          // u32 index = laneIndex + 32 * i;
          u32 index = gindex(i);
          dstaddrs[i] = nodeaddrpred(index < e.destinations.size(), nodeoffset + index);
          // IF_D(index < e.destinations.size()) {
          //   dstaddrs[i] = nodeaddr(nodeoffset + index);
          // }
          // ELSE {
          //   dstaddrs[i] = 0;
          // }
        }
      }
    };

    if (nowaits) {
      profiling.enter("nowaits");

      u32 nodeoffset = 0;
      for (const Edge& e : localCopies) {
        CHECK(e.sources.size() == 1);
        CHECK(e.destinations.size() >= 1);

        u64 srcaddr = getaddr(e.sources[0]);
        Vector<u64> dstaddrs((e.destinations.size() + emit->groupSize() - 1) / emit->groupSize());
        // Vector<u64> dstaddrs(16);
        setdst(e, nodeoffset, dstaddrs);
        nodeoffset += e.destinations.size();

        if (hasRemoteReads) {
          IF_D(threadIndex == 0) {
            fence_acquire_sys();
          }
        }

        profiling.enter("call emit");

        emit->emit(srcaddr, dstaddrs, e.bytes);
      }

    } else {
      u64 srcaddr = 0;
      Vector<u64> dstaddrs((maxDestinations + emit->groupSize() - 1) / emit->groupSize());
      u32 bytes = 0;
      u32 index = 0;

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
          setdst(e, nodeoffset, dstaddrs);
          bytes = e.bytes;
          if (hasRemoteReads) {
            IF_D(threadIndex == 0) {
              fence_acquire_sys();
            }
          }
          emit->emit(srcaddr, dstaddrs, bytes);

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
        // u32 index = blockSize * i + threadIndex;
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
          u32 blockStride = sizeof(uint32_t) * ranksStride * blockIndex;
          staddr = stbase + widen(itembytes * concurrencyIndex + blockStride);
          ldaddr = concurrency(syncAddrs.local) + widen(ldoffset + blockStride);
        }
      }
      // for (uint32_t i : range((syncsetup.size() + blockSize - 1) / blockSize)) {
      //   profiling.enter("sync setup");
      //   // u32 index = blockSize * i + threadIndex;
      //   u32 index = blockSize * i + (warpIndex + numWarps - 1) % numWarps + laneIndex * (blockSize / 32);
      //   ldaddrs.emplace_back();
      //   staddrs.emplace_back();
      //   u64& ldaddr = ldaddrs.back();
      //   u64& staddr = staddrs.back();
      //   preds.emplace_back();
      //   Value& pred = preds.back();
      //   pred = index < syncsetup.size();
      //   u32 addr = csyncaddr + index * sizeof(SyncSetup);
      //   u64 stbase = predicated(pred, ld_const_u64)(addr);
      //   u32 itembytes = predicated(pred, ld_const_u32)(addr + 8);
      //   u32 ldoffset = predicated(pred, ld_const_u32)(addr + 12);
      //   u32 blockStride = sizeof(uint32_t) * ranksStride * blockIndex;
      //   staddr = stbase + widen(itembytes * concurrencyIndex + blockStride);
      //   ldaddr = concurrency(syncAddrs.local) + widen(ldoffset + blockStride);
      // }
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
        // IF_D(!pred) {
        //   pred = ld_global_relaxed_sys_u32(ldaddr) < targetValue;
        //   IF_D(!pred) {
        //     BREAK;
        //   }
        //   nanosleep(10);
        //   CONTINUE;
        // }
        // pred &= predicated(pred, ld_global_relaxed_sys_u32)(ldaddr) < targetValue;
        // WHILE(vote_sync_any(pred)) {
        //   nanosleep(10);
        //   pred &= predicated(pred, ld_global_relaxed_sys_u32)(ldaddr) < targetValue;
        // }
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
    // profiling.enter("exit sync barrier");
    // barrier_sync();
    // profiling.enter("exit sync");
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
