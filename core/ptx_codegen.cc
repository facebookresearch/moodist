// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx_codegen.h"
#include "common.h"
#include "compile_op_kernel.h"
#include "group.h"
#include "ptx.h"

#include <array>

namespace moodist {

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

std::string generateCopyKernelPtx(Group* group, size_t gridSize, size_t blockSize, int depth, const char* target) {
  using namespace ptx;

  size_t rank = group->rank;
  const auto& peerIndices = group->peerIndices;
  int BS = (int)blockSize;
  int GS = (int)gridSize;
  int loopBytes = BS * 16;

  Module mod;
  mod.target = target;

  // Device globals for inter-block synchronization
  mod.globals.push_back(fmt::sprintf(".global .align 4 .u32 entryCounter[%zu]", Group::maxConcurrency));
  mod.globals.push_back(fmt::sprintf(".global .align 4 .u32 exitCounter[%zu]", Group::maxConcurrency));

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
      auto counterAddr = globalAddr("entryCounter") + widen(concurrencyIndex) * 4;
      auto old = atomicInc(counterAddr, (uint32_t)(GS - 1));
      IF(old == 0) {
        // First block: write stepValue to each peer
        for (size_t i : peerIndices) {
          auto& arr = group->peerCudaStepValue[i];
          auto addr = barrierAddr(arr.base, arr.itembytes, sizeof(uint32_t) * rank, concurrencyIndex);
          storeGlobalVolatile(addr, stepValue);
        }
        membar_sys();
      }
      // Wait for all peers
      for (size_t i : peerIndices) {
        auto addr = barrierAddr(group->cudaStepValue.buffer.cudaPointer, group->cudaStepValue.itembytes,
            sizeof(uint32_t) * group->ipcRanks[i], concurrencyIndex);
        WHILE(loadGlobalVolatile(addr, ValType::U32) < stepValue) {}
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
            ldcv_v4(v[k], srcPtr);
          }
          srcPtr += stride;
        }

        // Main loop: store-load overlap
        WHILE(i + (2 * depth - 1) * GS < count) {
          for (int j = 0; j < depth; j++) {
            stwt_v4(dstPtr, v[j]);
            dstPtr += stride;
            ldcv_v4(v[j], srcPtr);
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
          ldcv_v4(t, srcPtr);
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
          ldcv_v4(t, srcPtr);
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
    // Exit barrier
    // =================================================================
    membar_sys();
    barrier_sync();
    IF(tid == 0) {
      auto counterAddr = globalAddr("exitCounter") + widen(concurrencyIndex) * 4;
      auto old = atomicInc(counterAddr, (uint32_t)(GS - 1));
      IF(old == (GS - 1)) {
        // Last block: write copyDone to peers
        for (size_t i : peerIndices) {
          auto& arr = group->peerCudaCopyDone[i];
          auto addr =
              barrierAddr(arr.base, arr.itembytes, sizeof(uint32_t) * group->peerMyRemoteIndex[i], concurrencyIndex);
          storeGlobalVolatile(addr, stepValue);
        }
        // Wait for peer copyDone
        for (size_t i : peerIndices) {
          auto addr = barrierAddr(group->cudaCopyDone.buffer.cudaPointer, group->cudaCopyDone.itembytes,
              sizeof(uint32_t) * i, concurrencyIndex);
          WHILE(loadGlobalVolatile(addr, ValType::U32) < stepValue) {}
        }
      }
    }

    ret();
  }

  return mod.finalize();
}

} // namespace moodist
