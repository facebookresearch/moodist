// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx_codegen.h"
#include "common.h"
#include "compile_op_kernel.h"
#include "group.h"
#include "ptx.h"

#include <array>

namespace moodist {

const char* computeTarget(int computeMajor, int computeMinor) {
  if (computeMajor >= 10) {
    return "sm_100a";
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

// Helper: compute volatile address for barrier synchronization.
// Returns base + itembytes * concurrencyIndex + offset, all baked as hex constants.
static ptx::Val barrierAddr(uintptr_t base, uintptr_t itembytes, size_t offset, const ptx::Val& concurrencyIndex) {
  using namespace ptx;
  Val addr(ValType::U64);
  mov_u64(addr, hexImm(base));
  addr += widen(concurrencyIndex) * hexImm(itembytes);
  addr += hexImm(offset);
  return addr;
}

std::string generateCopyKernelPtx(Group* group, const CopyKernelConfig& config, const char* target) {
  using namespace ptx;

  int depth = config.depth;
  const char* loadOp = config.loadOp;

  // Select load function based on loadOp
  auto ld_v4 = [loadOp](std::array<Val, 4>& v, const Val& addr) {
    if (!strcmp(loadOp, "nc")) {
      ldnc_v4(v, addr);
    } else if (!strcmp(loadOp, "cs")) {
      ldcs_v4(v, addr);
    } else {
      ldcv_v4(v, addr);
    }
  };

  int BS = (int)config.blockSize;
  int GS = (int)config.gridSize;
  int loopBytes = BS * 16;

  // Copy engine: emits Phases 1-4 (head alignment, pipelined bulk, remainder, tail).
  // src/dst/bytes are mutable PTX Vals (updated as copy progresses).
  auto emitRegisterCopy = [&](Val& src, Val& dst, Val& bytes, const Val& tid, const Val& blockIdx) {
    // ---------------------------------------------------------------
    // Phase 1: Head alignment
    // ---------------------------------------------------------------
    auto srcMod = narrow(src) & 15;
    auto dstMod = narrow(dst) & 15;
    IF((srcMod == dstMod) & (srcMod != 0) & (bytes >= 16)) {
      auto headBytes = Val(ValType::U32);
      headBytes = 16;
      headBytes -= srcMod;

      // Clamp to remaining bytes
      IF(headBytes > bytes) {
        headBytes = bytes;
      }

      // Byte copy loop: tid stride blockSize
      auto i = Val(ValType::U32);
      i = tid;
      auto srcP = src + widen(tid);
      auto dstP = dst + widen(tid);
      WHILE(i < headBytes) {
        Val v(ValType::U32);
        ld_u8(v, srcP);
        st_u8(dstP, v);
        srcP += BS;
        dstP += BS;
        i += BS;
      }
      barrier_sync();

      // Update pointers and remaining byte count
      auto headWide = widen(headBytes);
      src += headWide;
      dst += headWide;
      bytes -= headBytes;
    }

    // Check alignment after head adjustment
    auto aligned = ((narrow(src) | narrow(dst)) & 15) == 0;

    // ---------------------------------------------------------------
    // Phase 2: Pipelined bulk copy (depth-unrolled)
    // ---------------------------------------------------------------
    IF(aligned & (bytes >= loopBytes)) {
      auto count = bytes / loopBytes;
      auto tid16w = widen(tid * 16);
      int64_t stride = (int64_t)GS * loopBytes;

      auto i = Val(ValType::U32);
      i = blockIdx;

      // Running pointers: avoid recomputing addresses each iteration
      auto srcPtr = src + widen(blockIdx) * loopBytes + tid16w;
      auto dstPtr = dst + widen(blockIdx) * loopBytes + tid16w;

      // Pipeline registers: depth x 4 u32 values
      std::array<std::array<Val, 4>, 56> v; // max depth 56
      for (int k = 0; k < depth; k++) {
        for (int c = 0; c < 4; c++) {
          v[k][c] = Val(ValType::U32);
        }
      }

      // Prime: load depth values
      for (int k = 0; k < depth; k++) {
        IF(i + k * GS < count) {
          ld_v4(v[k], srcPtr);
        }
        srcPtr += stride;
      }

      // Main loop: store-load overlap
      WHILE(i + (2 * depth - 1) * GS < count) {
        // warp_sync();
        for (int j = 0; j < depth; j++) {
          stwt_v4(dstPtr, v[j]);
          dstPtr += stride;
          ld_v4(v[j], srcPtr);
          srcPtr += stride;
        }
        i += depth * GS;
      }

      // Drain: store remaining
      for (int k = 0; k < depth; k++) {
        IF(i + k * GS < count) {
          stwt_v4(dstPtr, v[k]);
        }
        dstPtr += stride;
      }
      i += depth * GS;

      // Tail: simple loop for remaining full blocks
      WHILE(i < count) {
        std::array<Val, 4> t;
        ld_v4(t, srcPtr);
        stwt_v4(dstPtr, t);
        srcPtr += stride;
        dstPtr += stride;
        i += GS;
      }

      // Update pointers
      auto done = count * loopBytes;
      auto doneWide = widen(done);
      src += doneWide;
      dst += doneWide;
      bytes -= done;
    }

    // ---------------------------------------------------------------
    // Phase 3: Remaining aligned uint4 elements
    // ---------------------------------------------------------------
    IF(aligned) {
      auto remaining16 = bytes / 16;
      auto i = Val(ValType::U32);
      i = tid;
      auto srcPtr = src + widen(tid) * 16;
      auto dstPtr = dst + widen(tid) * 16;
      int64_t stride16 = (int64_t)BS * 16;
      WHILE(i < remaining16) {
        std::array<Val, 4> t;
        ld_v4(t, srcPtr);
        stwt_v4(dstPtr, t);
        srcPtr += stride16;
        dstPtr += stride16;
        i += BS;
      }
      auto done16 = remaining16 * 16;
      auto done16w = widen(done16);
      src += done16w;
      dst += done16w;
      bytes -= done16;
    }

    // ---------------------------------------------------------------
    // Phase 4: Tail bytes
    // ---------------------------------------------------------------
    {
      auto i = Val(ValType::U32);
      i = tid;
      auto srcPtr = src + widen(tid);
      auto dstPtr = dst + widen(tid);
      WHILE(i < bytes) {
        Val v(ValType::U32);
        ld_u8(v, srcPtr);
        st_u8(dstPtr, v);
        srcPtr += BS;
        dstPtr += BS;
        i += BS;
      }
    }
  };

  bool useBulk = !strcmp(config.copyEngine, "bulk");

  size_t rank = group->rank;
  const auto& peerIndices = group->peerIndices;

  Module mod;
  mod.target = target;

  auto* fn = mod.newFunction("compile_op_copy");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = BS;
    fn->addParamBytes(8, (int)sizeof(CompileOpCopyParameters));

    // Declare shared memory for bulk copy engine
    // Double-buffered: 2 data buffers + 2 mbarrier arrays (one per warp per phase)
    std::string smemBuf0Sym, smemBuf1Sym, smemMbar0Sym, smemMbar1Sym;
    int bulkChunkSize = (int)config.bulkChunkSize;
    int numWarps = BS / 32;
    constexpr int kMbarrierSize = 8;
    if (useBulk) {
      smemBuf0Sym = fn->addShared(16, bulkChunkSize, "buf0");
      smemBuf1Sym = fn->addShared(16, bulkChunkSize, "buf1");
      smemMbar0Sym = fn->addShared(8, kMbarrierSize * numWarps, "mbar0");
      smemMbar1Sym = fn->addShared(8, kMbarrierSize * numWarps, "mbar1");
    }

    activateNewBlock("entry");

    // --- Load kernel parameters ---
    auto params = paramBase(0);
    auto stepValue = loadParamField(params, 0, ValType::U32);
    auto concurrencyIndex = loadParamField(params, 4, ValType::U32);
    auto numDescriptors = loadParamField(params, 8, ValType::U32);

    auto tid = threadIdx_x();
    auto blockIdx = blockIdx_x();

    // Bulk copy engine: per-warp mbarrier initialization
    Val smemBuf0SharedAddr(ValType::U64), smemBuf1SharedAddr(ValType::U64);
    Val smemBuf0GenAddr(ValType::U64), smemBuf1GenAddr(ValType::U64);
    Val smemMbar0SharedAddr(ValType::U64), smemMbar1SharedAddr(ValType::U64);
    Val warpIdx(ValType::U32);
    Val laneIdx(ValType::U32);
    if (useBulk) {
      smemBuf0SharedAddr = globalAddr(smemBuf0Sym.c_str());
      smemBuf1SharedAddr = globalAddr(smemBuf1Sym.c_str());
      smemBuf0GenAddr = cvta_shared(smemBuf0SharedAddr);
      smemBuf1GenAddr = cvta_shared(smemBuf1SharedAddr);
      smemMbar0SharedAddr = globalAddr(smemMbar0Sym.c_str());
      smemMbar1SharedAddr = globalAddr(smemMbar1Sym.c_str());
      warpIdx = tid / 32;
      laneIdx = tid & 31;
    }

    // =================================================================
    // Entry barrier (thread 0 only)
    // =================================================================
    IF(tid == 0) {
      for (size_t i : peerIndices) {
        auto& arr = group->peerCudaStepValue[i];
        auto addr = barrierAddr(arr.base, arr.itembytes, sizeof(uint32_t) * rank, concurrencyIndex);
        storeGlobalRelaxedSys(addr, stepValue);
      }
      for (size_t i : peerIndices) {
        auto addr = barrierAddr(group->cudaStepValue.buffer.cudaPointer, group->cudaStepValue.itembytes,
            sizeof(uint32_t) * group->ipcRanks[i], concurrencyIndex);
        WHILE(loadGlobalAcquireSys(addr, ValType::U32) < stepValue) {}
      }
    }
    barrier_sync();

    // =================================================================
    // Descriptor loop (staggered for peer diversity)
    // =================================================================
    auto dd = Val(ValType::U32);
    dd = 0;
    WHILE(dd < numDescriptors) {
      // auto d = (blockIdx + dd) % numDescriptors;
      auto d = dd;

      // Load descriptor fields: offset 16 + d * 24
      auto descAddr = params + 16 + widen(d) * 24;
      auto src = loadParamField(descAddr, 0, ValType::U64);
      auto dst = loadParamField(descAddr, 8, ValType::U64);
      auto bytes = loadParamField(descAddr, 16, ValType::U32);

      if (useBulk) {
        // =============================================================
        // Bulk copy engine: per-warp double-buffered cp.async.bulk
        //
        // Each warp independently:
        //   1. DMAs its portion from global → shared (ping-pong buffers)
        //   2. Writes back from shared → global via registers
        //   3. Overlaps DMA of next chunk with write-back of current
        // =============================================================

        // Re-initialize mbarriers for this descriptor (phase must start fresh)
        IF(laneIdx == 0) {
          auto myMbar0 = smemMbar0SharedAddr + widen(warpIdx) * kMbarrierSize;
          auto myMbar1 = smemMbar1SharedAddr + widen(warpIdx) * kMbarrierSize;
          mbarrier_init(myMbar0, 1);
          mbarrier_init(myMbar1, 1);
        }
        barrier_sync();

        // When both DMA and write-back are warp-leader bulk, only lane 0 does
        // any real work. Guard the entire pipeline so the other 31 lanes idle,
        // avoiding divergence overhead and unnecessary warp_sync barriers.
        bool singleLane = config.bulkWarpLeaderDma && config.bulkWriteBack;
        Val guard(ValType::Pred);
        if (singleLane) {
          guard = (laneIdx == 0);
        } else {
          guard = 1;
        }

        IF(guard) {

          // Distribute work across blocks
          auto blockBytes = (bytes / GS / 16) * 16;
          auto blockStart = widen(blockIdx * blockBytes);
          src += blockStart;
          dst += blockStart;
          auto myBytes = Val(ValType::U32);
          myBytes = blockBytes;
          IF(blockIdx == GS - 1) {
            myBytes = bytes - blockIdx * blockBytes;
          }

          // Distribute work across warps within this block
          int perWarpChunk = bulkChunkSize / numWarps;
          auto warpTotal = (myBytes / numWarps / 16) * 16;
          auto warpStartOff = widen(warpIdx * warpTotal);
          auto warpSrc = src + warpStartOff;
          auto warpDst = dst + warpStartOff;
          auto warpRemaining = Val(ValType::U32);
          warpRemaining = warpTotal;
          IF(warpIdx == numWarps - 1) {
            warpRemaining = myBytes - warpIdx * warpTotal;
          }

          // Per-warp shared memory and mbarrier addresses
          auto warpBufOff = widen(warpIdx) * perWarpChunk;
          auto wBuf0S = smemBuf0SharedAddr + warpBufOff;
          auto wBuf1S = smemBuf1SharedAddr + warpBufOff;
          auto wBuf0G = smemBuf0GenAddr + warpBufOff;
          auto wBuf1G = smemBuf1GenAddr + warpBufOff;
          auto wMbar0 = smemMbar0SharedAddr + widen(warpIdx) * kMbarrierSize;
          auto wMbar1 = smemMbar1SharedAddr + widen(warpIdx) * kMbarrierSize;

          // Helpers to emit DMA and write-back (C++ lambdas, called at codegen time)
          auto emitDma = [&](const Val& bufS, const Val& mbar, const Val& srcAddr, const Val& size) {
            if (singleLane) {
              // Single-lane: we ARE lane 0, no branching or sync needed
              mbarrier_expect_tx(mbar, size);
              cp_async_bulk_shared_global(bufS, srcAddr, size, mbar);
              mbarrier_arrive(mbar);
            } else {
              IF(laneIdx == 0) {
                mbarrier_expect_tx(mbar, size);
              }
              if (config.bulkWarpLeaderDma) {
                IF(laneIdx == 0) {
                  cp_async_bulk_shared_global(bufS, srcAddr, size, mbar);
                  mbarrier_arrive(mbar);
                }
              } else {
                auto pt = size / 32;
                auto off = widen(laneIdx * pt);
                cp_async_bulk_shared_global(bufS + off, srcAddr + off, pt, mbar);
                IF(laneIdx == 0) {
                  mbarrier_arrive(mbar);
                }
              }
              warp_sync();
            }
          };

          auto emitWriteBack = [&](const Val& bufS, const Val& bufG, const Val& dstAddr, const Val& size) {
            if (singleLane) {
              // Single-lane: fire-and-forget DMA, no branching or sync
              cp_async_bulk_global_shared(dstAddr, bufS, size);
              cp_async_bulk_commit_group();
            } else if (config.bulkWriteBack) {
              IF(laneIdx == 0) {
                cp_async_bulk_global_shared(dstAddr, bufS, size);
                cp_async_bulk_commit_group();
              }
              warp_sync();
            } else {
              // Register write-back: load from shared, store to global
              // Each thread handles every 32nd 16-byte chunk.
              // stride = 32 * 16 = 512 bytes between consecutive accesses per thread.
              int64_t stride = 32 * 16;
              int itersPerChunk = perWarpChunk / (int)stride;

              if (itersPerChunk > 0) {
                // Unrolled path for full-size chunks (all iterations except possibly the last)
                IF(size == perWarpChunk) {
                  // Pipelined: load depth-2, then store depth-2
                  auto off = widen(laneIdx * 16);
                  int i = 0;
                  while (i + 1 < itersPerChunk) {
                    // Load 2
                    std::array<Val, 4> t0, t1;
                    ld_plain_v4(t0, bufG + off);
                    ld_plain_v4(t1, bufG + off + stride);
                    // Store 2
                    stwt_v4(dstAddr + off, t0);
                    stwt_v4(dstAddr + off + stride, t1);
                    off += stride * 2;
                    i += 2;
                  }
                  // Handle odd remainder
                  if (i < itersPerChunk) {
                    std::array<Val, 4> t;
                    ld_plain_v4(t, bufG + off);
                    stwt_v4(dstAddr + off, t);
                  }
                }
                ELSE {
                  // Dynamic loop for partial last chunk
                  auto i = widen(laneIdx * 16);
                  WHILE(narrow(i) < size) {
                    std::array<Val, 4> t;
                    ld_plain_v4(t, bufG + i);
                    stwt_v4(dstAddr + i, t);
                    i += stride;
                  }
                }
              } else {
                // perWarpChunk < stride: always use dynamic loop
                auto i = widen(laneIdx * 16);
                WHILE(narrow(i) < size) {
                  std::array<Val, 4> t;
                  ld_plain_v4(t, bufG + i);
                  stwt_v4(dstAddr + i, t);
                  i += stride;
                }
              }
              warp_sync();
            }
          };

          // Parity counters for each mbarrier (track phase advancement)
          auto parity0 = Val(ValType::U32);
          parity0 = 0;
          auto parity1 = Val(ValType::U32);
          parity1 = 0;

          // Phase: 0 = current data in buf0, 1 = current data in buf1
          auto phase = Val(ValType::U32);
          phase = 0;

          // Track current chunk size for write-back
          auto curChunk = Val(ValType::U32);
          curChunk = 0;

          // ---- Prolog: DMA first chunk into buf[0] ----
          {
            auto firstChunk = Val(ValType::U32);
            firstChunk = warpRemaining;
            IF(firstChunk > perWarpChunk) {
              firstChunk = perWarpChunk;
            }
            firstChunk = (firstChunk / 16) * 16;

            IF(firstChunk > 0) {
              emitDma(wBuf0S, wMbar0, warpSrc, firstChunk);
              warpSrc += widen(firstChunk);
              warpRemaining -= firstChunk;
            }
            curChunk = firstChunk;
          }

          // ---- Main loop: overlap DMA and write-back ----
          WHILE(warpRemaining >= 16) {
            auto nextChunk = Val(ValType::U32);
            nextChunk = warpRemaining;
            IF(nextChunk > perWarpChunk) {
              nextChunk = perWarpChunk;
            }
            nextChunk = (nextChunk / 16) * 16;

            IF(phase == 0) {
              // Wait for buf[0] DMA, start DMA into buf[1], write back buf[0]
              WHILE(!mbarrier_try_wait_parity(wMbar0, parity0)) {}
              if (config.bulkWriteBack) {
                if (singleLane) {
                  cp_async_bulk_wait_group(1);
                } else {
                  IF(laneIdx == 0) {
                    cp_async_bulk_wait_group(1);
                  }
                  warp_sync();
                }
              }
              emitDma(wBuf1S, wMbar1, warpSrc, nextChunk);
              if (!config.bulkSkipWriteBack) {
                emitWriteBack(wBuf0S, wBuf0G, warpDst, curChunk);
              }
              parity0 = parity0 ^ 1;
            }
            ELSE {
              // Wait for buf[1] DMA, start DMA into buf[0], write back buf[1]
              WHILE(!mbarrier_try_wait_parity(wMbar1, parity1)) {}
              if (config.bulkWriteBack) {
                if (singleLane) {
                  cp_async_bulk_wait_group(1);
                } else {
                  IF(laneIdx == 0) {
                    cp_async_bulk_wait_group(1);
                  }
                  warp_sync();
                }
              }
              emitDma(wBuf0S, wMbar0, warpSrc, nextChunk);
              if (!config.bulkSkipWriteBack) {
                emitWriteBack(wBuf1S, wBuf1G, warpDst, curChunk);
              }
              parity1 = parity1 ^ 1;
            }

            warpDst += widen(curChunk);
            warpSrc += widen(nextChunk);
            warpRemaining -= nextChunk;

            curChunk = nextChunk;
            phase = phase ^ 1;
          }

          // ---- Epilog: write back last chunk ----
          IF(curChunk > 0) {
            IF(phase == 0) {
              WHILE(!mbarrier_try_wait_parity(wMbar0, parity0)) {}
              if (!config.bulkSkipWriteBack) {
                emitWriteBack(wBuf0S, wBuf0G, warpDst, curChunk);
              }
            }
            ELSE {
              WHILE(!mbarrier_try_wait_parity(wMbar1, parity1)) {}
              if (!config.bulkSkipWriteBack) {
                emitWriteBack(wBuf1S, wBuf1G, warpDst, curChunk);
              }
            }
            warpDst += widen(curChunk);
          }

          // ---- Tail: remaining bytes < 16, byte copy ----
          if (!config.bulkSkipWriteBack) {
            IF(warpRemaining > 0) {
              if (singleLane) {
                // Single lane: sequential byte copy
                auto i = Val(ValType::U32);
                i = 0;
                WHILE(i < warpRemaining) {
                  Val v(ValType::U32);
                  ld_u8(v, warpSrc + widen(i));
                  st_u8(warpDst + widen(i), v);
                  i += 1;
                }
              } else {
                // All lanes: strided byte copy
                auto i = Val(ValType::U32);
                i = laneIdx;
                auto tailSrc = warpSrc + widen(laneIdx);
                auto tailDst = warpDst + widen(laneIdx);
                WHILE(i < warpRemaining) {
                  Val v(ValType::U32);
                  ld_u8(v, tailSrc);
                  st_u8(tailDst, v);
                  tailSrc += 32;
                  tailDst += 32;
                  i += 32;
                }
              }
            }
          }
          // ---- Flush: wait for all write-back DMAs before next descriptor ----
          if (config.bulkWriteBack && !config.bulkSkipWriteBack) {
            if (singleLane) {
              cp_async_bulk_wait_group(0);
            } else {
              IF(laneIdx == 0) {
                cp_async_bulk_wait_group(0);
              }
              warp_sync();
            }
          }

        } // end IF(guard)
        if (singleLane) {
          warp_sync(); // reconverge all lanes after single-lane pipeline
        }
      } else {
        emitRegisterCopy(src, dst, bytes, tid, blockIdx);
      }

      dd += 1;
    }

    // =================================================================
    // Exit barrier (per-block, no atomics)
    // =================================================================
    barrier_sync();
    IF(tid == 0) {
      // Write copyDone to each peer at this block's slot.
      // release.sys: ensures all prior data copy stores are visible before
      // the copyDone signal becomes visible to peers.
      for (size_t i : peerIndices) {
        auto& arr = group->peerCudaBlockDone[i];
        auto offset = widen(blockIdx + (uint32_t)(group->peerMyRemoteIndex[i] * maxBlocks)) * 4;
        auto addr = barrierAddr(arr.base, arr.itembytes, 0, concurrencyIndex) + offset;
        storeGlobalReleaseSys(addr, stepValue);
      }
      // Wait for each peer's corresponding block.
      // acquire.sys: ensures the peer's data copies are visible to us.
      for (size_t i : peerIndices) {
        auto offset = widen(blockIdx + (uint32_t)(i * maxBlocks)) * 4;
        auto addr =
            barrierAddr(group->cudaBlockDone.buffer.cudaPointer, group->cudaBlockDone.itembytes, 0, concurrencyIndex) +
            offset;
        WHILE(loadGlobalAcquireSys(addr, ValType::U32) < stepValue) {}
      }
    }

    ret();
  }

  return mod.finalize();
}

} // namespace moodist
