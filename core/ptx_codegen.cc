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

  size_t rank = group->rank;
  const auto& peerIndices = group->peerIndices;
  int BS = (int)config.blockSize;
  int GS = (int)config.gridSize;
  int loopBytes = BS * 16;

  Module mod;
  mod.target = target;

  auto* fn = mod.newFunction("compile_op_copy");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = BS;
    fn->addParamBytes(8, (int)sizeof(CompileOpCopyParameters));

    activateNewBlock("entry");

    // --- Load kernel parameters ---
    auto params = paramBase(0);
    auto stepValue = loadParamField(params, 0, ValType::U32);
    auto concurrencyIndex = loadParamField(params, 4, ValType::U32);
    auto numDescriptors = loadParamField(params, 8, ValType::U32);

    auto tid = threadIdx_x();
    auto blockIdx = blockIdx_x();

    // =================================================================
    // Entry barrier (thread 0 only)
    // =================================================================
    IF(tid == 0) {
      // Every block writes stepValue to each peer (idempotent — same value).
      // relaxed.sys: no data from this kernel needs flushing (prior kernel's
      // data is already visible due to implicit inter-kernel memory barrier).
      for (size_t i : peerIndices) {
        auto& arr = group->peerCudaStepValue[i];
        auto addr = barrierAddr(arr.base, arr.itembytes, sizeof(uint32_t) * rank, concurrencyIndex);
        storeGlobalRelaxedSys(addr, stepValue);
      }
      // Wait for all peers. acquire.sys: ensures subsequent reads (the data
      // copies) see the data written by the prior kernel on the peer GPU.
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
      auto d = (blockIdx + dd) % numDescriptors;

      // Load descriptor fields: offset 16 + d * 24
      auto descAddr = params + 16 + widen(d) * 24;
      auto src = loadParamField(descAddr, 0, ValType::U64);
      auto dst = loadParamField(descAddr, 8, ValType::U64);
      auto bytes = loadParamField(descAddr, 16, ValType::U32);

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
