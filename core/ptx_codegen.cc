// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx_codegen.h"
#include "common.h"
#include "compile_op.h"
#include "compile_op_kernel.h"
#include "cputhread.h"
#include "group.h"
#include "ipc_mapper.h"
#include "ptx.h"

#include <array>

namespace moodist {

using namespace ptx;

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

// Select load function based on loadOp config
static void ld_v4(const char* loadOp, std::array<ptx::Val, 4>& v, const ptx::Val& addr) {
  using namespace ptx;
  if (!strcmp(loadOp, "nc")) {
    ldnc_v4(v, addr);
  } else if (!strcmp(loadOp, "cs")) {
    ldcs_v4(v, addr);
  } else {
    ldcv_v4(v, addr);
  }
}

// // Register copy engine: emits Phases 1-4 (head alignment, pipelined bulk, remainder, tail).
// // src/dst/bytes are mutable PTX Vals (updated as copy progresses).
// static void emitRegisterCopy(const CopyKernelConfig& config, u64 src, u64 dst, u32 bytes,
//     u32 tid, u32 blockIdx) {
//   using namespace ptx;

//   int BS = (int)config.blockSize;
//   int GS = (int)config.gridSize;
//   int depth = config.depth.value();
//   int loopBytes = BS * 16;
//   const char* loadOp = config.loadOp.value();

//   // ---------------------------------------------------------------
//   // Phase 1: Head alignment
//   // ---------------------------------------------------------------
//   auto srcMod = narrow(src) & 15;
//   auto dstMod = narrow(dst) & 15;
//   IF((srcMod == dstMod) & (srcMod != 0) & (bytes >= 16)) {
//     auto headBytes = Val(ValType::U32);
//     headBytes = 16;
//     headBytes -= srcMod;

//     IF(headBytes > bytes) {
//       headBytes = bytes;
//     }

//     u32 i = tid;
//     auto srcP = src + widen(tid);
//     auto dstP = dst + widen(tid);
//     WHILE(i < headBytes) {
//       Val v(ValType::U32);
//       ld_u8(v, srcP);
//       st_u8(dstP, v);
//       srcP += BS;
//       dstP += BS;
//       i += BS;
//     }
//     barrier_sync();

//     auto headWide = widen(headBytes);
//     src += headWide;
//     dst += headWide;
//     bytes -= headBytes;
//   }

//   auto aligned = ((narrow(src) | narrow(dst)) & 15) == 0;

//   // ---------------------------------------------------------------
//   // Phase 2: Pipelined bulk copy (depth-unrolled)
//   // ---------------------------------------------------------------
//   IF(aligned & (bytes >= loopBytes)) {
//     auto count = bytes / loopBytes;
//     auto tid16w = widen(tid * 16);
//     int64_t stride = (int64_t)GS * loopBytes;

//     u32 i = blockIdx;

//     auto srcPtr = src + widen(blockIdx) * loopBytes + tid16w;
//     auto dstPtr = dst + widen(blockIdx) * loopBytes + tid16w;

//     std::array<std::array<Val, 4>, 56> v; // max depth 56
//     for (int k = 0; k < depth; k++) {
//       for (int c = 0; c < 4; c++) {
//         v[k][c] = Val(ValType::U32);
//       }
//     }

//     // Prime: load depth values
//     for (int k = 0; k < depth; k++) {
//       IF(i + k * GS < count) {
//         ld_v4(loadOp, v[k], srcPtr);
//       }
//       srcPtr += stride;
//     }

//     // Main loop: store-load overlap
//     WHILE(i + (2 * depth - 1) * GS < count) {
//       for (int j = 0; j < depth; j++) {
//         stwt_v4(dstPtr, v[j]);
//         dstPtr += stride;
//         ld_v4(loadOp, v[j], srcPtr);
//         srcPtr += stride;
//       }
//       i += depth * GS;
//     }

//     // Drain: store remaining
//     for (int k = 0; k < depth; k++) {
//       IF(i + k * GS < count) {
//         stwt_v4(dstPtr, v[k]);
//       }
//       dstPtr += stride;
//     }
//     i += depth * GS;

//     // Tail: simple loop for remaining full blocks
//     WHILE(i < count) {
//       std::array<Val, 4> t;
//       ld_v4(loadOp, t, srcPtr);
//       stwt_v4(dstPtr, t);
//       srcPtr += stride;
//       dstPtr += stride;
//       i += GS;
//     }

//     auto done = count * loopBytes;
//     auto doneWide = widen(done);
//     src += doneWide;
//     dst += doneWide;
//     bytes -= done;
//   }

//   // ---------------------------------------------------------------
//   // Phase 3: Remaining aligned uint4 elements
//   // ---------------------------------------------------------------
//   IF(aligned) {
//     auto remaining16 = bytes / 16;
//     u32 i = tid;
//     auto srcPtr = src + widen(tid) * 16;
//     auto dstPtr = dst + widen(tid) * 16;
//     int64_t stride16 = (int64_t)BS * 16;
//     WHILE(i < remaining16) {
//       std::array<Val, 4> t;
//       ld_v4(loadOp, t, srcPtr);
//       stwt_v4(dstPtr, t);
//       srcPtr += stride16;
//       dstPtr += stride16;
//       i += BS;
//     }
//     auto done16 = remaining16 * 16;
//     auto done16w = widen(done16);
//     src += done16w;
//     dst += done16w;
//     bytes -= done16;
//   }

//   // ---------------------------------------------------------------
//   // Phase 4: Tail bytes
//   // ---------------------------------------------------------------
//   {
//     u32 i = tid;
//     auto srcPtr = src + widen(tid);
//     auto dstPtr = dst + widen(tid);
//     WHILE(i < bytes) {
//       Val v(ValType::U32);
//       ld_u8(v, srcPtr);
//       st_u8(dstPtr, v);
//       srcPtr += BS;
//       dstPtr += BS;
//       i += BS;
//     }
//   }
// }

// // Bulk DMA: global → shared via cp.async.bulk with mbarrier completion
// static void emitBulkDma(const CopyKernelConfig& config, bool singleLane, const ptx::Val& bufS, const ptx::Val& mbar,
//     const ptx::Val& srcAddr, const ptx::Val& size, const ptx::Val& laneIdx) {
//   using namespace ptx;
//   if (singleLane) {
//     mbarrier_expect_tx(mbar, size);
//     cp_async_bulk_shared_global(bufS, srcAddr, size, mbar);
//     mbarrier_arrive(mbar);
//   } else {
//     IF(laneIdx == 0) {
//       mbarrier_expect_tx(mbar, size);
//     }
//     if (config.bulkWarpLeaderDma.value()) {
//       IF(laneIdx == 0) {
//         cp_async_bulk_shared_global(bufS, srcAddr, size, mbar);
//         mbarrier_arrive(mbar);
//       }
//     } else {
//       auto pt = size / 32;
//       auto off = widen(laneIdx * pt);
//       cp_async_bulk_shared_global(bufS + off, srcAddr + off, pt, mbar);
//       IF(laneIdx == 0) {
//         mbarrier_arrive(mbar);
//       }
//     }
//     warp_sync();
//   }
// }

// // Bulk write-back: shared → global via registers or cp.async.bulk DMA
// static void emitBulkWriteBack(const CopyKernelConfig& config, bool singleLane, int perWarpChunk, const ptx::Val&
// bufS,
//     const ptx::Val& bufG, const ptx::Val& dstAddr, const ptx::Val& size, const ptx::Val& laneIdx) {
//   using namespace ptx;
//   if (singleLane) {
//     cp_async_bulk_global_shared(dstAddr, bufS, size);
//     cp_async_bulk_commit_group();
//   } else if (config.bulkWriteBack.value()) {
//     IF(laneIdx == 0) {
//       cp_async_bulk_global_shared(dstAddr, bufS, size);
//       cp_async_bulk_commit_group();
//     }
//     warp_sync();
//   } else {
//     // Register write-back: load from shared, store to global
//     int64_t stride = 32 * 16;
//     int itersPerChunk = perWarpChunk / (int)stride;

//     if (itersPerChunk > 0) {
//       IF(size == perWarpChunk) {
//         // Unrolled path for full-size chunks: pipelined load-2/store-2
//         auto off = widen(laneIdx * 16);
//         int i = 0;
//         while (i + 1 < itersPerChunk) {
//           std::array<Val, 4> t0, t1;
//           ld_plain_v4(t0, bufG + off);
//           ld_plain_v4(t1, bufG + off + stride);
//           stwt_v4(dstAddr + off, t0);
//           stwt_v4(dstAddr + off + stride, t1);
//           off += stride * 2;
//           i += 2;
//         }
//         if (i < itersPerChunk) {
//           std::array<Val, 4> t;
//           ld_plain_v4(t, bufG + off);
//           stwt_v4(dstAddr + off, t);
//         }
//       }
//       ELSE {
//         // Dynamic loop for partial last chunk
//         auto i = widen(laneIdx * 16);
//         WHILE(narrow(i) < size) {
//           std::array<Val, 4> t;
//           ld_plain_v4(t, bufG + i);
//           stwt_v4(dstAddr + i, t);
//           i += stride;
//         }
//       }
//     } else {
//       auto i = widen(laneIdx * 16);
//       WHILE(narrow(i) < size) {
//         std::array<Val, 4> t;
//         ld_plain_v4(t, bufG + i);
//         stwt_v4(dstAddr + i, t);
//         i += stride;
//       }
//     }
//     warp_sync();
//   }
// }

// // N-buffer bulk copy engine: single-lane DMA with manual double-buffering per warp
// static void emitBulkNbuf(const CopyKernelConfig& config, uint32_t numBlockWarps, uint32_t inputSharedSize,
//     uint32_t numBlocks, ptx::Val& src, ptx::Val& dst, const ptx::Val& bytes, const ptx::Val& blockIndex,
//     const ptx::Val& warpIndex, const ptx::Val& laneIdx, const ptx::Val& smemBufS, const ptx::Val& smemMbarS) {
//   using namespace ptx;
//   constexpr int kMbarrierSize = 8;

//   uint32_t numReads = config.nbufReadCount.value();
//   uint32_t numWrites = config.nbufWriteCount.value();
//   uint32_t numBufs = numReads + numWrites;

//   CHECK(numReads >= 1);

//   uint32_t lcm = std::lcm(numBufs, numReads * 2);
//   // while (lcm < 12) {
//   //   lcm *= 2;
//   // }

//   bool shareBarriers = false; // TODO

//   uint32_t blockSharedAlignment = (numBlockWarps * numBufs * 256);
//   uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
//   uint32_t warpSharedSize = blockSharedSize / numBlockWarps;
//   uint32_t bufferSize = warpSharedSize / numBufs;
//   CHECK(bufferSize > 0 && bufferSize % 256 == 0);

//   uint32_t numGlobalWarps = numBlockWarps * numBlocks;
//   auto globalWarpIndex = blockIndex * numBlockWarps + warpIndex;

//   auto bytesPerWarp = (bytes + numGlobalWarps * bufferSize - 1) / (numGlobalWarps * bufferSize) * bufferSize;

//   auto myBytes = bytesPerWarp;

//   auto warpStartGlobal = bytesPerWarp * globalWarpIndex;
//   auto warpStartShared = warpIndex * warpSharedSize;

//   IF(warpStartGlobal + myBytes >= bytes) {
//     IF(warpStartGlobal >= bytes) {
//       myBytes = 0;
//     }
//     ELSE {
//       myBytes = bytes - warpStartGlobal;
//     }
//   }

//   src += widen(warpStartGlobal);
//   dst += widen(warpStartGlobal);

//   auto sharedMemAddr = smemBufS + widen(warpStartShared);
//   Vector<Value> mbars;
//   for (int n : range(numReads)) {
//     mbars.push_back(smemMbarS + widen((numReads * warpIndex + n) * kMbarrierSize));
//     mbarrier_init(mbars.back(), 1);
//   }
//   // auto iteration = u32(0);
//   auto offset = to_u32(0);
//   Label alldone;
//   auto f = [&](bool isFirst) {
//     WHILE(true) {
//       for (uint32_t i : range(lcm)) {
//         uint32_t bufferIndex = i % numBufs;
//         uint32_t prevBufferIndex = (i + numBufs - 1) % numBufs;
//         uint32_t readIndex = i % numReads;
//         uint32_t parity = (i / numReads) & 1;
//         uint32_t prevParity = ((i + lcm - 1) / numReads) & 1;

//         auto& curMbar = mbars[readIndex];
//         auto& prevMbar = mbars[(readIndex + numReads - 1) % numReads];

//         auto step = [&](Value size) {
//           mbarrier_expect_tx(curMbar, size);
//           if (!config.bulkSkipWriteBack.value()) {
//             cp_async_bulk_wait_group(numBufs - 2);
//           }
//           cp_async_bulk_shared_global(sharedMemAddr + bufferSize * bufferIndex, src + widen(offset), size, curMbar);
//           mbarrier_arrive(curMbar);

//           if (!isFirst || i != 0) {
//             WHILE(!mbarrier_try_wait_parity(prevMbar, prevParity)) {}

//             if (!config.bulkSkipWriteBack.value()) {
//               cp_async_bulk_global_shared(
//                   dst + widen(offset - bufferSize), sharedMemAddr + bufferSize * prevBufferIndex, bufferSize);
//               cp_async_bulk_commit_group();
//             }
//           }
//         };

//         auto bytes = min_u32(myBytes - offset, bufferSize);
//         step(bytes);
//         auto nextOffset = offset + bufferSize;
//         IF(nextOffset >= myBytes) {
//           WHILE(!mbarrier_try_wait_parity(curMbar, parity)) {}

//           if (!config.bulkSkipWriteBack.value()) {
//             cp_async_bulk_global_shared(dst + widen(offset), sharedMemAddr + bufferSize * bufferIndex, bytes);
//             cp_async_bulk_commit_group();
//             cp_async_bulk_wait_group(0);
//           }
//           GOTO(alldone);
//         }
//         offset = nextOffset;
//       }
//       if (isFirst) {
//         BREAK;
//       }
//     }
//   };
//   IF(myBytes != 0) {
//     f(true);
//     f(false);
//     LABEL(alldone);
//   }
// }

// // N-buffer bulk copy engine: single-lane DMA with manual double-buffering per warp
// static void emitBulkNbuf2(const CopyKernelConfig& config, uint32_t inputNumBlockWarps, uint32_t inputSharedSize,
//     uint32_t numBlocks, ptx::Val& src, ptx::Val& dst, const ptx::Val& bytes, const ptx::Val& blockIndex,
//     const ptx::Val& inputWarpIndex, const ptx::Val& laneIdx, const ptx::Val& smemBufS, const ptx::Val& smemMbarS,
//     const ptx::Value& smemSpinAddr) {
//   using namespace ptx;
//   constexpr int kMbarrierSize = 8;

//   uint32_t numReads = config.nbufReadCount.value();
//   uint32_t numWrites = config.nbufWriteCount.value();
//   uint32_t numBufs = numReads + numWrites;

//   CHECK(numReads >= 1);

//   uint32_t lcm = std::lcm(numBufs, numReads * 2);
//   // while (lcm < 12) {
//   //   lcm *= 2;
//   // }

//   auto role = inputWarpIndex & 1;
//   auto warpIndex = inputWarpIndex / 2;
//   uint32_t numBlockWarps = inputNumBlockWarps / 2;

//   bool shareBarriers = false; // TODO

//   uint32_t blockSharedAlignment = (numBlockWarps * numBufs * 256);
//   uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
//   uint32_t warpSharedSize = blockSharedSize / numBlockWarps;
//   uint32_t bufferSize = warpSharedSize / numBufs;
//   CHECK(bufferSize > 0 && bufferSize % 256 == 0);

//   uint32_t numGlobalWarps = numBlockWarps * numBlocks;
//   auto globalWarpIndex = blockIndex * numBlockWarps + warpIndex;

//   auto bytesPerWarp = (bytes + numGlobalWarps * bufferSize - 1) / (numGlobalWarps * bufferSize) * bufferSize;

//   auto myBytes = bytesPerWarp;

//   auto warpStartGlobal = bytesPerWarp * globalWarpIndex;
//   auto warpStartShared = warpIndex * warpSharedSize;

//   IF(warpStartGlobal + myBytes >= bytes) {
//     IF(warpStartGlobal >= bytes) {
//       myBytes = 0;
//     }
//     ELSE {
//       myBytes = bytes - warpStartGlobal;
//     }
//   }

//   src += widen(warpStartGlobal);
//   dst += widen(warpStartGlobal);

//   barrier_sync(3);

//   auto sharedMemAddr = smemBufS + widen(warpStartShared);
//   Vector<Value> mbars;
//   for (int n : range(numReads)) {
//     mbars.push_back(smemMbarS + widen((numReads * warpIndex + n) * kMbarrierSize));
//     PRED((laneIdx == 0) & role == 0) mbarrier_init(mbars.back(), 1);
//   }
//   Value spin = smemSpinAddr + widen(warpIndex * 4);
//   PRED((laneIdx == 0) & role == 0) st_shared_release_cta_u32(spin, to_u32(0));
//   barrier_sync(2);
//   // auto iteration = u32(0);
//   auto offset = to_u32(0);
//   Label alldone;
//   auto f = [&](bool isFirst, int role) {
//     WHILE(true) {
//       for (uint32_t i : range(lcm)) {
//         uint32_t bufferIndex = i % numBufs;
//         uint32_t prevBufferIndex = (i + numBufs - 1) % numBufs;
//         uint32_t readIndex = i % numReads;
//         uint32_t parity = (i / numReads) & 1;
//         uint32_t prevParity = ((i + lcm - 1) / numReads) & 1;

//         auto& curMbar = mbars[readIndex];
//         auto& prevMbar = mbars[(readIndex + numReads - 1) % numReads];

//         auto step = [&](Value size) {
//           if (role == 1) {
//             if (!config.bulkSkipWriteBack.value()) {
//               cp_async_bulk_wait_group_read(numBufs - 2);
//             }
//             st_shared_release_cta_u32(spin, offset);
//             mbarrier_expect_tx(curMbar, size);
//           }

//           if (role == 0) {
//             WHILE(ld_shared_acquire_cta_u32(spin) < offset) {}
//             cp_async_bulk_shared_global(sharedMemAddr + bufferSize * bufferIndex, src + widen(offset), size,
//             curMbar); mbarrier_arrive(curMbar);
//           }

//           if (!isFirst || i != 0) {
//             if (role == 1) {
//               WHILE(!mbarrier_try_wait_parity(prevMbar, prevParity)) {}

//               if (!config.bulkSkipWriteBack.value()) {
//                 cp_async_bulk_global_shared(
//                     dst + widen(offset - bufferSize), sharedMemAddr + bufferSize * prevBufferIndex, bufferSize);
//                 cp_async_bulk_commit_group();
//               }
//             }
//           }
//         };

//         auto bytes = min_u32(myBytes - offset, bufferSize);
//         step(bytes);
//         auto nextOffset = offset + bufferSize;
//         IF(nextOffset >= myBytes) {
//           if (role == 1) {
//             WHILE(!mbarrier_try_wait_parity(curMbar, parity)) {}

//             if (!config.bulkSkipWriteBack.value()) {
//               cp_async_bulk_global_shared(dst + widen(offset), sharedMemAddr + bufferSize * bufferIndex, bytes);
//               cp_async_bulk_commit_group();
//               cp_async_bulk_wait_group(0);
//             }
//             st_shared_release_cta_u32(spin, nextOffset);
//           }
//           GOTO(alldone);
//         }
//         offset = nextOffset;
//       }
//       if (isFirst) {
//         BREAK;
//       }
//     }
//   };
//   IF((myBytes != 0) & (laneIdx == 0)) {
//     IF(role == 0) {
//       f(true, 0);
//       f(false, 0);
//     }
//     ELSE {
//       f(true, 1);
//       f(false, 1);
//     }
//     LABEL(alldone);
//   }
//   warp_sync();
// }

// // Warp-pipelined bulk copy engine: warps cooperate as pipeline stages
// static void emitBulkWarpPipe(const CopyKernelConfig& config, int numWarps, int bulkChunkSize, int GS, ptx::Val& src,
//     ptx::Val& dst, const ptx::Val& bytes, const ptx::Val& blockIdx, const ptx::Val& warpIdx, const ptx::Val& laneIdx,
//     const ptx::Val& smemBufS, const ptx::Val& smemBufG, const ptx::Val& smemMbarS) {
//   using namespace ptx;
//   constexpr int kMbarrierSize = 8;

//   int pipeDepth = config.warppipeDepth.value() > 0 ? std::min(config.warppipeDepth.value(), numWarps) : numWarps;
//   pipeDepth = std::min(pipeDepth, 8);

//   int stageChunk = bulkChunkSize / numWarps;
//   CHECK(bulkChunkSize % numWarps == 0);
//   CHECK(numWarps % pipeDepth == 0);

//   int pipeWidth = numWarps / pipeDepth;

//   Value isLeader = laneIdx == 0;

//   // Initialize mbarrier for pipeline warps
//   auto myMbar = smemMbarS + widen(warpIdx) * kMbarrierSize;
//   PRED(isLeader) mbarrier_init(myMbar, 1);

//   // Distribute work across blocks
//   auto blockBytes = (bytes / GS / 256) * 256;
//   auto blockStart = widen(blockIdx * blockBytes);
//   src += blockStart;
//   dst += blockStart;
//   auto myBytes = Val(ValType::U32);
//   myBytes = blockBytes;
//   IF(blockIdx == GS - 1) {
//     myBytes = bytes - blockIdx * blockBytes;
//   }
//   IF(blockIdx * blockBytes >= bytes) {
//     myBytes = 0;
//   }

//   auto myBufS = smemBufS + widen(warpIdx) * stageChunk;
//   auto myBufG = smemBufG + widen(warpIdx) * stageChunk;

//   int roundBytes = numWarps * stageChunk;
//   auto offset = Val(ValType::U64);
//   offset = 0;

//   // bar_arrive(1, 32);
//   // bar_sync(1, 32);

//   auto pipeIndex = warpIdx / pipeWidth;
//   auto isLastPipe = pipeIndex == pipeDepth - 1;

//   int barBase = 2;

//   IF(pipeIndex == 0) {
//     if (pipeDepth > 1) {
//       bar_arrive(barBase + pipeIndex, 32 * pipeWidth * 2);
//     }
//   }

//   WHILE(pipeDepth > 1 ? true : isLeader) {
//     for (int i = 0; i != 2; ++i) {
//       IF(narrow(offset) >= myBytes) {
//         BREAK;
//       }
//       auto stageOff = offset + widen(warpIdx * stageChunk);
//       auto thisChunk = Val(ValType::U32);
//       thisChunk = 0;

//       IF(narrow(stageOff) < myBytes) {
//         thisChunk = myBytes - narrow(stageOff);
//         IF(thisChunk > stageChunk) {
//           thisChunk = stageChunk;
//         }
//         thisChunk = (thisChunk / 16) * 16;
//       }

//       // Issue DMA
//       PRED(isLeader) mbarrier_expect_tx(myMbar, thisChunk);
//       if (pipeDepth > 1) {
//         bar_sync(barBase + pipeIndex, 32 * pipeWidth * 2);
//       }
//       PRED(isLeader) {
//         if (!config.bulkSkipWriteBack.value()) {
//           cp_async_bulk_wait_group(0);
//         }
//         cp_async_bulk_shared_global(myBufS, src + stageOff, thisChunk, myMbar);
//       }
//       if (pipeDepth > 1) {
//         PRED(!isLastPipe) bar_arrive(barBase + pipeIndex + 1, 32 * pipeWidth * 2);
//         PRED(isLastPipe) bar_arrive(barBase, 32 * pipeWidth * 2);
//       }
//       PRED(isLeader) mbarrier_arrive(myMbar);

//       IF(isLeader) {
//         // Wait for DMA completion and write back
//         WHILE(!mbarrier_try_wait_parity(myMbar, i & 1)) {}
//       }
//       // warp_sync();

//       if (!config.bulkSkipWriteBack.value()) {
//         PRED(isLeader) {
//           cp_async_bulk_global_shared(dst + stageOff, myBufS, thisChunk);
//           cp_async_bulk_commit_group();
//         }
//       }

//       offset += roundBytes;
//     }
//   }

//   // Sync all warps before next descriptor
//   barrier_sync();
// }

// // Double-buffered bulk copy engine: per-warp independent ping-pong
// static void emitBulkDoubleBuf(const CopyKernelConfig& config, int numWarps, int bulkChunkSize, int GS, ptx::Val& src,
//     ptx::Val& dst, const ptx::Val& bytes, const ptx::Val& blockIdx, const ptx::Val& warpIdx, const ptx::Val& laneIdx,
//     const ptx::Val& smemBuf0S, const ptx::Val& smemBuf1S, const ptx::Val& smemBuf0G, const ptx::Val& smemBuf1G,
//     const ptx::Val& smemMbar0S, const ptx::Val& smemMbar1S) {
//   using namespace ptx;
//   constexpr int kMbarrierSize = 8;

//   // Re-initialize mbarriers for this descriptor (phase must start fresh)
//   IF(laneIdx == 0) {
//     auto myMbar0 = smemMbar0S + widen(warpIdx) * kMbarrierSize;
//     auto myMbar1 = smemMbar1S + widen(warpIdx) * kMbarrierSize;
//     mbarrier_init(myMbar0, 1);
//     mbarrier_init(myMbar1, 1);
//   }
//   barrier_sync();

//   bool singleLane = config.bulkWarpLeaderDma.value() && config.bulkWriteBack.value();
//   Val guard(ValType::Pred);
//   if (singleLane) {
//     guard = (laneIdx == 0);
//   } else {
//     guard = 1;
//   }

//   IF(guard) {

//     // Distribute work across blocks
//     auto blockBytes = (bytes / GS / 16) * 16;
//     auto blockStart = widen(blockIdx * blockBytes);
//     src += blockStart;
//     dst += blockStart;
//     auto myBytes = Val(ValType::U32);
//     myBytes = blockBytes;
//     IF(blockIdx == GS - 1) {
//       myBytes = bytes - blockIdx * blockBytes;
//     }

//     // Distribute work across warps within this block
//     int perWarpChunk = bulkChunkSize / numWarps;
//     auto warpTotal = (myBytes / numWarps / 16) * 16;
//     auto warpStartOff = widen(warpIdx * warpTotal);
//     auto warpSrc = src + warpStartOff;
//     auto warpDst = dst + warpStartOff;
//     auto warpRemaining = Val(ValType::U32);
//     warpRemaining = warpTotal;
//     IF(warpIdx == numWarps - 1) {
//       warpRemaining = myBytes - warpIdx * warpTotal;
//     }

//     // Per-warp shared memory and mbarrier addresses
//     auto warpBufOff = widen(warpIdx) * perWarpChunk;
//     auto wBuf0S = smemBuf0S + warpBufOff;
//     auto wBuf1S = smemBuf1S + warpBufOff;
//     auto wBuf0G = smemBuf0G + warpBufOff;
//     auto wBuf1G = smemBuf1G + warpBufOff;
//     auto wMbar0 = smemMbar0S + widen(warpIdx) * kMbarrierSize;
//     auto wMbar1 = smemMbar1S + widen(warpIdx) * kMbarrierSize;

//     // Parity counters for each mbarrier
//     auto parity0 = Val(ValType::U32);
//     parity0 = 0;
//     auto parity1 = Val(ValType::U32);
//     parity1 = 0;

//     auto phase = Val(ValType::U32);
//     phase = 0;

//     auto curChunk = Val(ValType::U32);
//     curChunk = 0;

//     // ---- Prolog: DMA first chunk into buf[0] ----
//     {
//       auto firstChunk = Val(ValType::U32);
//       firstChunk = warpRemaining;
//       IF(firstChunk > perWarpChunk) {
//         firstChunk = perWarpChunk;
//       }
//       firstChunk = (firstChunk / 16) * 16;

//       IF(firstChunk > 0) {
//         emitBulkDma(config, singleLane, wBuf0S, wMbar0, warpSrc, firstChunk, laneIdx);
//         warpSrc += widen(firstChunk);
//         warpRemaining -= firstChunk;
//       }
//       curChunk = firstChunk;
//     }

//     // ---- Main loop: overlap DMA and write-back ----
//     WHILE(warpRemaining >= 16) {
//       auto nextChunk = Val(ValType::U32);
//       nextChunk = warpRemaining;
//       IF(nextChunk > perWarpChunk) {
//         nextChunk = perWarpChunk;
//       }
//       nextChunk = (nextChunk / 16) * 16;

//       IF(phase == 0) {
//         WHILE(!mbarrier_try_wait_parity(wMbar0, parity0)) {}
//         if (config.bulkWriteBack.value()) {
//           if (singleLane) {
//             cp_async_bulk_wait_group(1);
//           } else {
//             IF(laneIdx == 0) {
//               cp_async_bulk_wait_group(1);
//             }
//             warp_sync();
//           }
//         }
//         emitBulkDma(config, singleLane, wBuf1S, wMbar1, warpSrc, nextChunk, laneIdx);
//         if (!config.bulkSkipWriteBack.value()) {
//           emitBulkWriteBack(config, singleLane, perWarpChunk, wBuf0S, wBuf0G, warpDst, curChunk, laneIdx);
//         }
//         parity0 = parity0 ^ 1;
//       }
//       ELSE {
//         WHILE(!mbarrier_try_wait_parity(wMbar1, parity1)) {}
//         if (config.bulkWriteBack.value()) {
//           if (singleLane) {
//             cp_async_bulk_wait_group(1);
//           } else {
//             IF(laneIdx == 0) {
//               cp_async_bulk_wait_group(1);
//             }
//             warp_sync();
//           }
//         }
//         emitBulkDma(config, singleLane, wBuf0S, wMbar0, warpSrc, nextChunk, laneIdx);
//         if (!config.bulkSkipWriteBack.value()) {
//           emitBulkWriteBack(config, singleLane, perWarpChunk, wBuf1S, wBuf1G, warpDst, curChunk, laneIdx);
//         }
//         parity1 = parity1 ^ 1;
//       }

//       warpDst += widen(curChunk);
//       warpSrc += widen(nextChunk);
//       warpRemaining -= nextChunk;

//       curChunk = nextChunk;
//       phase = phase ^ 1;
//     }

//     // ---- Epilog: write back last chunk ----
//     IF(curChunk > 0) {
//       IF(phase == 0) {
//         WHILE(!mbarrier_try_wait_parity(wMbar0, parity0)) {}
//         if (!config.bulkSkipWriteBack.value()) {
//           emitBulkWriteBack(config, singleLane, perWarpChunk, wBuf0S, wBuf0G, warpDst, curChunk, laneIdx);
//         }
//       }
//       ELSE {
//         WHILE(!mbarrier_try_wait_parity(wMbar1, parity1)) {}
//         if (!config.bulkSkipWriteBack.value()) {
//           emitBulkWriteBack(config, singleLane, perWarpChunk, wBuf1S, wBuf1G, warpDst, curChunk, laneIdx);
//         }
//       }
//       warpDst += widen(curChunk);
//     }

//     // ---- Tail: remaining bytes < 16, byte copy ----
//     if (!config.bulkSkipWriteBack.value()) {
//       IF(warpRemaining > 0) {
//         if (singleLane) {
//           auto i = Val(ValType::U32);
//           i = 0;
//           WHILE(i < warpRemaining) {
//             Val v(ValType::U32);
//             ld_u8(v, warpSrc + widen(i));
//             st_u8(warpDst + widen(i), v);
//             i += 1;
//           }
//         } else {
//           auto i = Val(ValType::U32);
//           i = laneIdx;
//           auto tailSrc = warpSrc + widen(laneIdx);
//           auto tailDst = warpDst + widen(laneIdx);
//           WHILE(i < warpRemaining) {
//             Val v(ValType::U32);
//             ld_u8(v, tailSrc);
//             st_u8(tailDst, v);
//             tailSrc += 32;
//             tailDst += 32;
//             i += 32;
//           }
//         }
//       }
//     }
//     // ---- Flush: wait for all write-back DMAs before next descriptor ----
//     if (config.bulkWriteBack.value() && !config.bulkSkipWriteBack.value()) {
//       if (singleLane) {
//         cp_async_bulk_wait_group(0);
//       } else {
//         IF(laneIdx == 0) {
//           cp_async_bulk_wait_group(0);
//         }
//         warp_sync();
//       }
//     }

//   } // end IF(guard)
//   if (singleLane) {
//     warp_sync();
//   }
// }

struct EmitSingleCopy {
  virtual ~EmitSingleCopy() = default;
  virtual void init() {}
  virtual void initPostSync() {}
  virtual void emit(u64 src, u64 dst, u32 bytes) = 0;
};

struct EmitMultiCopy {
  virtual ~EmitMultiCopy() = default;
  virtual void init() {}
  virtual void initPostSync() {}
  virtual void emit(u64 src, Vector<u64> dst, u32 ndst, u32 bytes) = 0;
};

// struct EmitFlexBuf : EmitSingleCopy {
//   const CopyKernelConfig& config;
//   const Value& blockIndex;
//   const Value& warpIndex;
//   const Value& laneIndex;
//   const Value& smemBufAddr;
//   const Value& smemMbarAddr;

//   EmitFlexBuf(const CopyKernelConfig& config, const Value& blockIndex, const Value& warpIndex, const Value&
//   laneIndex,
//       const Value& smemBufAddr, const Value& smemMbarAddr)
//       : config(config), blockIndex(blockIndex), warpIndex(warpIndex), laneIndex(laneIndex), smemBufAddr(smemBufAddr),
//         smemMbarAddr(smemMbarAddr) {}

//   // Inputs (set before init())
//   uint32_t numBlocks = config.gridSize;
//   uint32_t inputSharedSize = *config.bulkChunkSize;

//   uint32_t numParallelReads = *config.flexNumParallelReads;
//   uint32_t numParallelWrites = *config.flexNumParallelWrites;

//   // Computed in init()
//   uint32_t numReadWarps;
//   uint32_t numWriteWarps;
//   uint32_t numReadWarpsTotal;
//   uint32_t numWriteWarpsTotal;
//   Value role;
//   Vector<Value> readBarriers;
//   Vector<Value> writeBarriers;

//   uint32_t numBuffers = config.flexNumBuffers.value();
//   uint32_t blockSharedAlignment = numBuffers * 256;
//   uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
//   uint32_t bufferSize = blockSharedSize / numBuffers;

//   Value isReader;
//   Value parallelIndex;
//   Value parallelWarpIndex;
//   Value parallelThreadIndex;

//   void init() override {

//     uint32_t numReadThreads = config.flexReadThreads.value();
//     uint32_t numWriteThreads = config.flexWriteThreads.value();

//     CHECK(numReadThreads % 32 == 0);
//     CHECK(numWriteThreads % 32 == 0);
//     numReadWarps = numReadThreads / 32;
//     numWriteWarps = numWriteThreads / 32;
//     numReadWarpsTotal = numReadWarps * numParallelReads;
//     numWriteWarpsTotal = numWriteWarps * numParallelWrites;

//     for (int n : range(numBuffers)) {
//       readBarriers.push_back(smemMbarAddr + widen(n * 8));
//       writeBarriers.push_back(smemMbarAddr + widen((numBuffers + n) * 8));
//       IF((laneIndex == 0) & (warpIndex == 0)) {
//         mbarrier_init(readBarriers.back(), numReadWarps);
//         mbarrier_init(writeBarriers.back(), numWriteWarps);
//       }
//     }

//     CHECK(bufferSize > 0 && bufferSize % 256 == 0);

//     isReader = warpIndex < numReadWarpsTotal;
//     IF(isReader) {
//       parallelIndex = warpIndex / numReadWarps;
//       parallelWarpIndex = warpIndex % numReadWarps;
//     }
//     ELSE {
//       parallelIndex = (warpIndex - numReadWarpsTotal) / numWriteWarps;
//       parallelWarpIndex = (warpIndex - numReadWarpsTotal) % numWriteWarps;
//     }
//     parallelThreadIndex = parallelWarpIndex * 32 + laneIndex;
//   }

//   enum { roleRead, roleWrite };
//   enum { methodBulk, methodReg };

//   void emit(int role, int method, Value src, Value dst, Value bytes) {
//     Label alldone;
//     auto offset = to_u32(blockIndex * bufferSize);
//     uint32_t numParallel = role == roleRead ? numParallelReads : numParallelWrites;
//     uint32_t numThreads = role == roleRead ? numReadWarps * 32 : numWriteWarps * 32;

//     auto f = [&](uint32_t parallel) {
//       Value isLeader = laneIndex == 0;
//       if (method == methodBulk) {
//         GOTO_IF_NOT(isLeader, alldone);
//       } else {
//         warp_sync();
//       }
//       Label loop;
//       for (int primeIteration : range(2)) {
//         if (primeIteration == 1) {
//           LABEL(loop);
//         }

//         for (uint32_t i : range(numBuffers * 2)) {
//           uint32_t bufferIndex = i % numBuffers;
//           if (bufferIndex % numParallel == parallel) {
//             uint32_t parity = (i / numBuffers) & 1;
//             auto& readBar = readBarriers[bufferIndex];
//             auto& writeBar = writeBarriers[bufferIndex];
//             Value size = min_u32(bytes - offset, bufferSize);

//             if (role == roleRead) {
//               if (method == methodBulk) {
//                 if (primeIteration == 1 || parity == 1) {
//                   mbarrier_wait_parity(writeBar, parity ^ 1);
//                 }
//                 // mbarrier_arrive(readBar);
//                 mbarrier_expect_tx(readBar, size);
//                 mbarrier_arrive_noComplete(readBar);
//                 cp_async_bulk_shared_global(smemBufAddr + bufferSize * bufferIndex, src + widen(offset), size,
//                 readBar); mbarrier_wait_parity(readBar, parity);
//               } else {
//                 if (primeIteration == 1 || parity == 1) {
//                   IF(isLeader) mbarrier_wait_parity(writeBar, parity ^ 1);
//                 }
//                 warp_sync();
//                 auto srcaddr = src + widen(offset);
//                 auto dstaddr = smemBufAddr + bufferSize * bufferIndex;
//                 // FOR(auto i = u32(parallelThreadIndex * 16), i < size, i += numThreads * 16) {
//                 //   std::array<Value, 4> v;
//                 //   ldcv_v4(v, srcaddr + widen(i));
//                 //   st_shared_v4_u32(dstaddr + widen(i), v);
//                 // }
//                 {
//                   auto i = u32(parallelThreadIndex * 16);
//                   WHILE(i < size) {
//                     constexpr int unroll = 4;
//                     std::array<std::array<Value, 4>, unroll> v;
//                     for (int u : range(unroll)) {
//                       Value ui = i + numThreads * 16 * u;
//                       PRED(ui < size) ldcv_v4(v[u], srcaddr + widen(ui));
//                     }
//                     for (int u : range(unroll)) {
//                       Value ui = i + numThreads * 16 * u;
//                       PRED(ui < size) st_shared_v4_u32(dstaddr + widen(ui), v[u]);
//                     }
//                     i += numThreads * 16 * unroll;
//                   }
//                 }
//                 warp_sync();
//                 PRED(isLeader) mbarrier_arrive(readBar);
//               }
//             } else {
//               if (method == methodBulk) {
//                 mbarrier_wait_parity(readBar, parity);
//                 cp_async_bulk_global_shared(dst + widen(offset), smemBufAddr + bufferSize * bufferIndex, size);
//                 cp_async_bulk_commit_group();
//                 cp_async_bulk_wait_group_read(0);
//                 mbarrier_arrive(writeBar);
//               } else {
//                 IF(isLeader) mbarrier_wait_parity(readBar, parity);
//                 warp_sync();
//                 auto srcaddr = smemBufAddr + bufferSize * bufferIndex;
//                 auto dstaddr = dst + widen(offset);
//                 FOR(auto i = u32(parallelThreadIndex * 16), i < size, i += numThreads * 16) {
//                   std::array<Value, 4> v;
//                   ld_shared_v4_u32(v, srcaddr + widen(i));
//                   stwt_v4(dstaddr + widen(i), v);
//                 }
//                 warp_sync();
//                 PRED(isLeader) mbarrier_arrive(writeBar);
//                 warp_sync();
//               }
//             }
//           }

//           offset += numBlocks * bufferSize;
//           GOTO_IF(offset >= bytes, alldone);
//         }

//         if (primeIteration == 1) {
//           GOTO(loop);
//         }
//       }
//       GOTO(alldone);
//     };

//     GOTO_IF(offset >= bytes, alldone);
//     for (uint32_t parallel : range(numParallel)) {
//       IF(parallelIndex == parallel) {
//         f(parallel);
//       }
//     }

//     LABEL(alldone);
//     warp_sync();
//   }

//   void emit(u64 src, u64 dst, u32 bytes) override {
//     IF(isReader) {
//       if (!strcmp(config.flexReadMethod.value(), "bulk")) {
//         emit(roleRead, methodBulk, src, dst, bytes);
//       } else if (!strcmp(config.flexReadMethod.value(), "reg")) {
//         emit(roleRead, methodReg, src, dst, bytes);
//       } else {
//         CHECK(false);
//       }
//     }
//     ELSE {
//       if (!strcmp(config.flexWriteMethod.value(), "bulk")) {
//         emit(roleWrite, methodBulk, src, dst, bytes);
//       } else if (!strcmp(config.flexWriteMethod.value(), "reg")) {
//         emit(roleWrite, methodReg, src, dst, bytes);
//       } else {
//         CHECK(false);
//       }
//     }
//   }
// };

struct EmitLockstep : EmitMultiCopy {
  const CopyKernelConfig& config;
  u32 blockIndex;
  u32 warpIndex;
  u32 laneIndex;

  EmitLockstep(const CopyKernelConfig& config, const u32& blockIndex, const u32& warpIndex, const u32& laneIndex)
      : config(config), blockIndex(blockIndex), warpIndex(warpIndex), laneIndex(laneIndex) {}

  // Inputs (set before init())
  uint32_t numBlocks = config.gridSize;
  uint32_t inputSharedSize = *config.bulkChunkSize;

  // uint32_t numParallel = *config.lockstepNumParallel;

  // Computed in init()
  // uint32_t numWarps;
  // uint32_t numWarpsTotal;

  // uint32_t numBuffers = config.lockstepNumBuffers.value();
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

  void init() override {
    // st_shared_release_cta_u32(sharedOffset, blockIndex * bufferSize);
  }

  void emit(u64 src, Vector<u64> dst, u32 ndst, u32 bytes) override {
    Label alldone;
    // IF(laneIndex != 0) {
    //   WHILE(!bar_red_or(syncIndex, 64, 0)) {}
    //   GOTO(alldone);
    // }

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
      // u64 dstaddr = dst + widen(offset);

      u32 size = min_u32(bytes - offset, bufferSize);

      IF_D(isLeader) {
        cp_async_bulk_wait_group_read(0);
        mbarrier_expect_tx(barrier, size);
        mbarrier_arrive_noComplete(barrier);
        cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
      }

      bar_red_or(syncIndex, 64, 0);

      IF_D(isLeader) {
        mbarrier_wait_parity(barrier, parity);
        for (size_t i : indices(dst)) {
          IF(i < ndst) {
            u64 dstaddr = dst[i] + widen(offset);
            cp_async_bulk_global_shared(dstaddr, buffer, size);
            // multimem_cp_async_bulk_global_shared(dstaddr, buffer, size);
          }
        }
        cp_async_bulk_commit_group();
      }

      GOTO_IF(bar_red_or(syncIndex, 64, 0), alldone);
    }

    GOTO(loop);

    LABEL(alldone);
    IF_D(isLeader) {
      cp_async_bulk_wait_group_read(0);
    }
    barrier_sync();
  }
};

// struct EmitLockstepGlobal {
//   const CopyKernelConfig& config;
//   u32 blockIndex;
//   u32 warpIndex;
//   u32 laneIndex;

//   u64 params;
//   u32 numDescriptors;
//   u32 concurrencyIndex;

//   uint32_t numBlocks = config.gridSize;
//   uint32_t inputSharedSize = *config.bulkChunkSize;

//   uint32_t numBuffers = config.blockSize / 32;
//   uint32_t blockSharedAlignment = numBuffers * 256;
//   uint32_t blockSharedSize = inputSharedSize / blockSharedAlignment * blockSharedAlignment;
//   uint32_t bufferSize = blockSharedSize / numBuffers;

//   uint32_t maxDescriptors = 200;

//   u32 bufferBaseAddr = addShared32(16, numBuffers* bufferSize, "buffer");
//   u32 barrierBaseAddr = addShared32(8, numBuffers * 8, "barrier");
//   u64 sharedOffset = addGlobalVar(4, maxConcurrency* maxDescriptors * 4, "offset");

//   u32 bufferIndex = warpIndex;

//   u32 syncIndex = 1 + warpIndex / 2;
//   u32 pairIndex = warpIndex % 2;

//   u32 buffer = bufferBaseAddr + bufferIndex * bufferSize;
//   u32 barrier = barrierBaseAddr + bufferIndex * 8;

//   void init() {
//     PRED(numDescriptors > maxDescriptors) trap();
//   }

//   void emit() {
//     trap();
//     // Label alldone;
//     // // IF(laneIndex != 0) {
//     // //   WHILE(!bar_red_or(syncIndex, 64, 0)) {}
//     // //   GOTO(alldone);
//     // // }

//     // IF(laneIndex == 0) {
//     //   st_shared_release_cta_u32(sharedOffset, blockIndex * bufferSize);
//     //   mbarrier_init(barrier, 1);
//     // }
//     // barrier_sync();

//     // Value isLeader = laneIndex == 0;

//     // IF(pairIndex != 0) {
//     //   GOTO_IF(bar_red_or(syncIndex, 64, 0), alldone);
//     // }

//     // PRED(bytes % 16 != 0) trap();
//     // PRED(src % 16 != 0) trap();
//     // PRED(dst % 16 != 0) trap();

//     // Label loop;
//     // LABEL(loop);

//     // for (uint32_t parity : range(2)) {
//     //   u32 offset = 0;
//     //   PRED(isLeader) offset = atom_shared_relaxed_cta_add_u32(sharedOffset, numBlocks * bufferSize);
//     //   offset = shfl_sync_idx_b32(offset, 0, 31);
//     //   IF(offset >= bytes) {
//     //     bar_red_or(syncIndex, 64, 1);
//     //     GOTO(alldone);
//     //   }

//     //   u64 srcaddr = src + widen(offset);
//     //   u64 dstaddr = dst + widen(offset);

//     //   u32 size = min_u32(bytes - offset, bufferSize);

//     //   IF(isLeader) {
//     //     cp_async_bulk_wait_group_read(0);
//     //     mbarrier_expect_tx(barrier, size);
//     //     mbarrier_arrive_noComplete(barrier);
//     //     cp_async_bulk_shared_global(buffer, srcaddr, size, barrier);
//     //   }

//     //   bar_red_or(syncIndex, 64, 0);

//     //   IF(isLeader) {
//     //     mbarrier_wait_parity(barrier, parity);
//     //     cp_async_bulk_global_shared(dstaddr, buffer, size);
//     //     // multimem_cp_async_bulk_global_shared(dstaddr, buffer, size);
//     //     cp_async_bulk_commit_group();
//     //   }

//     //   GOTO_IF(bar_red_or(syncIndex, 64, 0), alldone);
//     // }

//     // GOTO(loop);

//     // LABEL(alldone);
//     // barrier_sync();
//   }
// };

struct KernelMappedBuffers {
  AllocatedArray& local;
  Vector<PeerArrayRef> remote;
};

// HashMap<std::string, Vector<std::unique_ptr<KernelMappedBuffers>>> kernelMappedBuffers;
KernelMappedBuffers mapn(
    compile_op::CompileContext& ctx, std::string name, Vector<uint32_t> ranks, size_t bytesPerConcurrencyIndex) {
  name += "______";
  for (uint32_t r : ranks) {
    name += fmt::sprintf(".%d", r);
  }
  auto& x = ctx.cachedBuffers[name];
  AllocatedArray* local = nullptr;
  for (auto& v : x) {
    if (v->itembytes >= bytesPerConcurrencyIndex) {
      local = &*v;
    }
  }
  if (!local) {
    auto arr = std::make_unique<AllocatedArray>(ctx.group->allocateArrayDevice(
        std::max(std::bit_ceil(bytesPerConcurrencyIndex), (size_t)65536), maxConcurrency));
    local = &*arr;
    ctx.group->buffersToReset.push_back(local);
    x.push_back(std::move(arr));
  }

  const uint32_t rank = ctx.group->rank;

  KernelMappedBuffers r(*local);

  Vector<uintptr_t> mapped(ranks.size());
  for (size_t i : indices(ranks)) {
    if (ranks[i] == rank) {
      mapped[i] = local->buffer.cudaPointer;
      continue;
    }
    size_t peerIndex = ctx.group->getPeerIndex(ranks[i]);
    ctx.group->ipcMapper->requestAddress(peerIndex, local->buffer.cudaPointer, local->buffer.bytes, &mapped[i]);
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

std::string generateCopyKernelPtx(const Group* group, const CopyKernelConfig& config, const char* target,
    const CustomOpDescriptor& op, compile_op::CompileContext& ctx) {
  using namespace ptx;

  uint32_t blockSize = config.blockSize;
  uint32_t numBlocks = config.gridSize;

  bool useBulk = !strcmp(config.copyEngine, "bulk");
  bool useFlexbuf = !strcmp(config.copyEngine, "flexbuf");
  bool useLockstep = !strcmp(config.copyEngine, "lockstep");
  bool useLockstepg = !strcmp(config.copyEngine, "lockstepg");
  bool useWarppipe = useBulk && !strcmp(config.bulkMode.value(), "warppipe");
  bool useNbuf = useBulk && !strcmp(config.bulkMode.value(), "nbuf");
  bool useNbuf2 = useBulk && !strcmp(config.bulkMode.value(), "nbuf2");

  size_t rank = group->rank;

  Module mod;
  mod.target = target;
  setModule(&mod);
  if (false) {
    mod.addGlobal(".u8", 4 * 1024 * 1024, "debug_buf");
    mod.addGlobal(".u32", 1, "debug_warp_counter");
    mod.addGlobal(".u64", 1, "debug_clock_ref");
  }

  CHECK(!op.cudaEdges.empty());

  Vector<uint32_t> ranks;
  Vector<uint64_t> localNodes;
  for (auto& v : op.cudaEdges) {
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

  using Edge = CustomOpDescriptor::Edge;
  using Node = CustomOpDescriptor::Node;

  // HashMap<uint32_t, Vector<Node>> sourceNodes;
  // HashMap<uint32_t, Vector<Node>> destinationNodes;

  // for (auto& e : op.cudaEdges) {
  //   CHECK(e.sources.size() >= 1);
  //   CHECK(e.destinations.size() >= 1);
  //   bool outgoing = e.sources[0].rank == rank;
  //   bool filled = e.sources[0].filled;
  //   for (auto& n : e.sources) {
  //     CHECK((n.rank == rank) == outgoing);
  //     CHECK(n.filled == filled);
  //   }
  //   for (auto& n : e.sources) {
  //     sourceNodes[n.rank].push_back(n);
  //   }
  //   for (auto& n : e.destinations) {
  //   }
  // }

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
    return mapn(ctx, name, ranks, bytes);
  };

  auto syncAddrs = map("syncs", sizeof(uint32_t) * ranks.size() * numBlocks);
  auto addrAddrs = map("addresses", sizeof(uint64_t) * op.localCudaTensorMappings.size() * numBlocks);

  auto depFilled = map("depFilled", sizeof(uint32_t) * localNodes.size() * numBlocks);

  auto busyWait = [&](u64 addr, u32 value) {
    IF(ld_global_acquire_sys_u32(addr) < value) {
      WHILE(ld_global_relaxed_sys_u32(addr) < value) {}
      fence_acquire_sys();
    }
  };

  auto* fn = mod.newFunction("compile_op_copy");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = blockSize;
    fn->addParamBytes(8, sizeof(CompileOpCopyParameters));
    fn->addParamBytes(8,
        sizeof(uint64_t) *
            std::max(op.inputs.size() + op.inputCopies.size() + op.outputs.size() + op.outputCopies.size(), (size_t)1));
    fn->addParamBytes(8, sizeof(uint64_t) * std::max(op.remoteCudaTensorMappings.size(), (size_t)1));

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
    u32 laneIndex = threadIndex & 31;

    auto concurrency = [&](const auto& v) {
      if constexpr (std::is_same_v<std::decay_t<decltype(v)>, PeerArrayRef>) {
        return v.base + widen(v.itembytes * concurrencyIndex);
      } else {
        return v.buffer.cudaPointer + widen(v.itembytes * concurrencyIndex);
      }
    };

    auto findLocalMapping = [&](const auto& r) {
      for (size_t i : indices(op.localCudaTensorMappings)) {
        auto& v = op.localCudaTensorMappings[i];
        if (v.rank == r.rank && v.tensorIndex == r.tensorIndex) {
          return i;
        }
      }
      log.error("failed to find mapping for %d %d\n", r.rank, r.tensorIndex);
      for (size_t i : indices(op.localCudaTensorMappings)) {
        auto& v = op.localCudaTensorMappings[i];
        log.error(" index %d: %d %d\n", i, v.rank, v.tensorIndex);
      }
      CHECK(false);
    };

    std::unique_ptr<EmitMultiCopy> emit;
    // if (useFlexbuf) {
    //   emit = std::make_unique<EmitFlexBuf>(config, blockIndex, warpIdx, laneIdx, smemBufSharedAddr,
    //   smemMbarSharedAddr);
    // }
    if (useLockstep) {
      emit = std::make_unique<EmitLockstep>(config, blockIndex, warpIndex, laneIndex);
    }
    CHECK(emit != nullptr);
    emit->init();
    IF_D(threadIndex == 0) {
      for (size_t i : indices(op.remoteCudaTensorMappings)) {
        auto& v = op.remoteCudaTensorMappings[i];
        u64 addr = concurrency(addrAddrs.remote[rankIndex(v.rank)]) +
                   widen(sizeof(uint64_t) * (v.destinationSlot + v.stride * blockIndex));
        st_global_u64(addr, ld_param_u64(mappedAddrs, sizeof(uint64_t) * i));
      }
      fence_release_sys();
      for (size_t i : indices(ranks)) {
        if (i == myIndex) {
          continue;
        }
        u64 addr =
            concurrency(syncAddrs.remote[i]) + widen(sizeof(uint32_t) * (myPeerIndex[i] + ranks.size() * blockIndex));
        st_global_relaxed_sys_u32(addr, stepValue);
      }
      for (size_t i : indices(ranks)) {
        if (i == myIndex) {
          continue;
        }
        u64 addr = concurrency(syncAddrs.local) + widen(sizeof(uint32_t) * (i + ranks.size() * blockIndex));
        WHILE(ld_global_relaxed_sys_u32(addr) < stepValue) {}
      }
      fence_acquire_sys();
    }
    barrier_sync();

    // =================================================================
    // Clock synchronization: normalize clock64() across SMs
    // =================================================================
    if (false) {
      int totalWarps = numBlocks * blockSize / 32;
      auto counterAddr = globalAddr("debug_warp_counter");
      auto refAddr = globalAddr("debug_clock_ref");
      auto lane = threadIndex & 31;

      // Each warp leader atomicInc's the counter
      IF_D(lane == 0) {
        atomicInc(counterAddr, (uint32_t)(totalWarps - 1));
      }
      warp_sync();
      // Spin until all warps have arrived (counter wraps to 0)
      {
        auto v = Val(ValType::U32);
        v = 1;
        WHILE(v != 0) {
          ld_global_u32(v, counterAddr);
        }
      }

      // Grid0/thread0 writes reference clock
      IF(blockIndex == 0) {
        IF_D(threadIndex == 0) {
          st_global_u64(refAddr, clock64());
        }
      }

      // All threads spin until reference is written
      auto ref = Val(ValType::U64);
      ref = 0;
      WHILE(ref == 0) {
        ld_global_u64(ref, refAddr);
      }

      // Compute per-thread clock offset: my_clock - reference
      auto clockOffset = clock64() - ref;
      // clockOffset is now available for adjusting future clock64() reads
      (void)clockOffset; // TODO: use for instrumentation
    }

    Vector<Edge> localCopies;

    for (auto& v : op.cudaEdges) {
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

      if (src.rank == rank) {
        if (*config.copyWrite) {
          localCopies.push_back(v);
        } else if (localdst) {
          Edge copy = v;
          for (auto i = copy.destinations.begin(); i != copy.destinations.end();) {
            if (i->rank == rank) {
              ++i;
            } else {
              i = copy.destinations.erase(i);
            }
          }
          localCopies.push_back(v);
        }
      } else {
        if (!*config.copyWrite) {
          Edge copy = v;
          for (auto i = copy.destinations.begin(); i != copy.destinations.end();) {
            if (i->rank != src.rank) {
              ++i;
            } else {
              i = copy.destinations.erase(i);
            }
          }
          localCopies.push_back(v);
        }
      }
    }

    auto getaddr = [&](const Node& n) {
      if (n.rank == rank) {
        return ld_param_u64(inputAddrs, sizeof(uint64_t) * n.tensorIndex) + n.offset;
      } else {
        return ld_global_u64(
                   concurrency(addrAddrs.local) +
                   widen(sizeof(uint64_t) * (findLocalMapping(n) + op.localCudaTensorMappings.size() * blockIndex))) +
               n.offset;
      }
    };

    size_t maxDestinations = 0;
    for (const Edge& e : localCopies) {
      maxDestinations = std::max(maxDestinations, e.destinations.size());
    }

    if (true) {

      for (const Edge& e : localCopies) {
        CHECK(e.sources.size() == 1);
        CHECK(e.destinations.size() >= 1);

        u64 srcaddr = getaddr(e.sources[0]);
        Vector<u64> dstaddrs(e.destinations.size());
        for (size_t i : indices(dstaddrs)) {
          dstaddrs[i] = getaddr(e.destinations[i]);
        }

        emit->emit(srcaddr, dstaddrs, dstaddrs.size(), e.bytes);
      }

      fence_release_sys();

    } else {

      Label copy;
      u64 srcaddr = 0;
      Vector<u64> dstaddrs(maxDestinations);
      u32 ndst = 0;
      u32 bytes = 0;
      u32 index = 0;

      std::vector<Label> rets;

      WHILE(true) {

        u32 numDone = 0;

        for (const Edge& e : localCopies) {
          CHECK(e.sources.size() == 1);
          CHECK(e.destinations.size() >= 1);
          auto& src = e.sources[0];

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
                fence_acquire_sys();
                pred = 1;
              }
            }
          }
          GOTO_IF_NOT(bar_red_or(15, blockSize, pred != 0), skip);

          srcaddr = getaddr(src);
          for (size_t i : indices(e.destinations)) {
            dstaddrs[i] = getaddr(e.destinations[i]);
          }
          ndst = e.destinations.size();
          // dstaddr = getaddr(dst);
          bytes = e.bytes;
          index = rets.size();
          GOTO(copy);
          Label ret;
          LABEL(ret);
          rets.push_back(std::move(ret));

          barrier_sync();
          IF_D(threadIndex == 0) {
            fence_release_sys();
            HashMap<uint32_t, bool> signalRanks;
            signalRanks[rank] = true;
            // signalRanks[dst.rank] = true;
            for (const Node& n : e.destinations) {
              signalRanks[n.rank] = true;
            }
            for (auto& ne : op.cudaEdges) {
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
              IF_D(threadIndex == 0) {
                u32 ready = 1;
                if (!src.filled) {
                  auto it = std::ranges::find(localNodes, src.id);
                  CHECK(it != localNodes.end());
                  IF(ld_global_relaxed_sys_u32(
                         concurrency(depFilled.local) +
                         widen(sizeof(uint32_t) * ((it - localNodes.begin()) + localNodes.size() * blockIndex))) <
                      stepValue) {
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
                    fence_acquire_sys();
                    pred = 1;
                  }
                }
              }
            }
          }
          numDone += 1;
          LABEL(skip);
          warp_sync();
        }

        IF(bar_red_or(14, blockSize, numDone == localCopies.size())) {
          BREAK;
        }
      }

      CHECK(!rets.empty());

      Label done;
      GOTO(done);

      LABEL(copy);
      emit->emit(srcaddr, dstaddrs, ndst, bytes);

      brx_idx(index, rets);

      LABEL(done);
    }

    // bool copyWriteMode = *config.copyWrite;
    // std::vector<int64_t> signals;
    // std::vector<int8_t> waits;
    // bool anyWaits = false;
    // bool anySignals = false;
    // if (op) {
    //   PRED(numDescriptors != 0) trap();
    //   // CHECK(!op->descriptorTemplates.empty());
    //   // PRED(numDescriptors != op->descriptorTemplates.size()) trap();
    //   // for (size_t i : indices(op->descriptorTemplates)) {
    //   //   const auto& dt = op->descriptorTemplates[i];

    //   //   log.info("%d: descr %d,  %d %d  %d %d\n", rank, i, dt.srcIsOutput, dt.srcRank, dt.dstIsOutput,
    //   dt.dstRank);

    //   //   if (dt.srcIsOutput) {
    //   //     anyWaits = true;
    //   //     waits.push_back(1);
    //   //   } else {
    //   //     waits.push_back(0);
    //   //   }
    //   //   log.info("descr %d, peer rank %d index %d\n", i, dt.peerSignalRank, dt.peerSignalIndex);
    //   //   if (dt.peerSignalIndex != -1) {
    //   //     CHECK(dt.peerSignalRank != rank);
    //   //     anySignals = true;
    //   //     signals.push_back(op->peerSignalBufferAddrs.at(group->getPeerIndex(dt.peerSignalRank)) +
    //   //                       dt.peerSignalIndex * sizeof(uint32_t));
    //   //   } else {
    //   //     signals.push_back(0);
    //   //   }

    //   //   // uintptr_t src, dst;

    //   //   // if (dt.srcRank == rank) {
    //   //   //   src = (dt.srcIsOutput ? outputs : inputs)[dt.srcTensorIndex]->data() + dt.srcOffset;
    //   //   // } else {
    //   //   //   src = remoteAddrs[i];
    //   //   // }

    //   //   // if (dt.dstRank == rank) {
    //   //   //   dst = (dt.dstIsOutput ? outputs : inputs)[dt.dstTensorIndex]->data() + dt.dstOffset;
    //   //   // } else {
    //   //   //   dst = remoteAddrs[i];
    //   //   // }

    //   //   // descrs += fmt::sprintf("%zu bytes from rank %u to rank %u, ", dt.bytes, dt.srcRank, dt.dstRank);
    //   // }
    // }

    // log.info("%d: waits is %s\n", rank, fmt::to_string(fmt::join(waits, ", ")));
    // log.info("%d: signals is %s\n", rank, fmt::to_string(fmt::join(signals, ", ")));

    // u32 signalsAddr;
    // u32 waitsAddr;
    // if (op) {
    //   signalsAddr = addConst32(signals, "signals");
    //   waitsAddr = addConst32(waits, "waits");
    // }

    // // =================================================================
    // // u32 dd = 0;
    // // WHILE(dd < numDescriptors) {
    // FOR(u32 dd = 0, dd != numDescriptors, dd += 1) {
    //   // auto d = (blockIdx + dd) % numDescriptors;
    //   auto d = dd;

    //   if (op && anyWaits) {
    //     IF((blockIndex == 0) & (threadIndex == 0)) {
    //       IF(ld_const_u8(waitsAddr + d) != 0) {
    //         u64 addr = op->signalBufferAddr +
    //                    ((concurrencyIndex * config.gridSize + blockIndex) * op->signalBufferNumDescriptors + d) *
    //                        sizeof(uint32_t);
    //         WHILE(ld_global_acquire_sys_u32(addr) < stepValue) {}
    //       }
    //     }
    //     barrier_sync();
    //   }

    //   // Load descriptor fields: offset 16 + d * 24
    //   auto descAddr = params + 16 + widen(d) * 24;
    //   auto src = loadParamField(descAddr, 0, ValType::U64);
    //   auto dst = loadParamField(descAddr, 8, ValType::U64);
    //   auto bytes = loadParamField(descAddr, 16, ValType::U32);

    //   if (useNbuf) {
    //     IF(laneIdx == 0) {
    //       emitBulkNbuf(config, numWarps, bulkChunkSize, GS, src, dst, bytes, blockIndex, warpIdx, laneIdx,
    //           smemBufSharedAddr, smemMbarSharedAddr);
    //     }
    //     warp_sync();
    //   } else if (useNbuf2) {
    //     emitBulkNbuf2(config, numWarps, bulkChunkSize, GS, src, dst, bytes, blockIndex, warpIdx, laneIdx,
    //         smemBufSharedAddr, smemMbarSharedAddr, smemSpinAddr);
    //   } else if (useFlexbuf) {
    //     emit->emit(src, dst, bytes);
    //   } else if (useLockstep) {
    //     emit->emit(src, dst, bytes);
    //   } else if (useWarppipe) {
    //     emitBulkWarpPipe(config, numWarps, bulkChunkSize, GS, src, dst, bytes, blockIndex, warpIdx, laneIdx,
    //         smemBufSharedAddr, smemBufGenAddr, smemMbarSharedAddr);
    //   } else if (useBulk) {
    //     emitBulkDoubleBuf(config, numWarps, bulkChunkSize, GS, src, dst, bytes, blockIndex, warpIdx, laneIdx,
    //         smemBuf0SharedAddr, smemBuf1SharedAddr, smemBuf0GenAddr, smemBuf1GenAddr, smemMbar0SharedAddr,
    //         smemMbar1SharedAddr);
    //   } else {
    //     emitRegisterCopy(config, src, dst, bytes, threadIndex, blockIndex);
    //   }

    //   if (op && anySignals) {
    //     barrier_sync();
    //     IF((blockIndex == 0) & (threadIndex == 0)) {
    //       u64 w = ld_const_u64(signalsAddr + d * sizeof(uint64_t));
    //       IF(w != 0) {
    //         w += (concurrencyIndex * config.gridSize + blockIndex) * op->signalBufferNumDescriptors *
    //         sizeof(uint32_t); st_global_release_sys_u32(w, stepValue);
    //       }
    //     }
    //   }
    // }

    // =================================================================
    // Exit barrier (per-block, no atomics)
    // =================================================================
    barrier_sync();
    IF_D(threadIndex == 0) {
      // // Write copyDone to each peer at this block's slot.
      // // release.sys: ensures all prior data copy stores are visible before
      // // the copyDone signal becomes visible to peers.
      // for (size_t i : peerIndices) {
      //   auto& arr = group->peerCudaBlockDone[i];
      //   auto offset = widen(blockIndex + (uint32_t)(group->peerMyRemoteIndex[i] * maxBlocks)) * 4;
      //   auto addr = barrierAddr(arr.base, arr.itembytes, 0, concurrencyIndex) + offset;
      //   storeGlobalReleaseSys(addr, stepValue);
      // }
      // // Wait for each peer's corresponding block.
      // // acquire.sys: ensures the peer's data copies are visible to us.
      // for (size_t i : peerIndices) {
      //   auto offset = widen(blockIndex + (uint32_t)(i * maxBlocks)) * 4;
      //   auto addr =
      //       barrierAddr(group->cudaBlockDone.buffer.cudaPointer, group->cudaBlockDone.itembytes, 0, concurrencyIndex)
      //       + offset;
      //   WHILE(loadGlobalAcquireSys(addr, ValType::U32) < stepValue) {}
      // }

      // fence_release_sys();
      for (size_t i : indices(ranks)) {
        if (i == myIndex) {
          continue;
        }
        u64 addr =
            concurrency(syncAddrs.remote[i]) + widen(sizeof(uint32_t) * (myPeerIndex[i] + ranks.size() * blockIndex));
        st_global_relaxed_sys_u32(addr, stepValue + 1);
      }
      for (size_t i : indices(ranks)) {
        if (i == myIndex) {
          continue;
        }
        u64 addr = concurrency(syncAddrs.local) + widen(sizeof(uint32_t) * (i + ranks.size() * blockIndex));
        WHILE(ld_global_relaxed_sys_u32(addr) < stepValue + 1) {}
      }
    }

    ret();
  }

  return mod.finalize();
}

} // namespace moodist
