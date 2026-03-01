// Copyright (c) Meta Platforms, Inc. and affiliates.

// compile_op implementation
// See compile_op.h for design overview
//
// Seven-phase protocol:
// Phase 1: Exchange logical mappings (sender → receiver)
// Phase 2: Overlap resolution via cell decomposition (local)
// Phase 3: Send read requests (receiver → sender)
// Phase 4: Process requests, generate copies (sender)
// Phase 5: Finalize and generate reads (receiver)
// Phase 6: Compute allLocal flag and build reverse mapping (localInputProvides)
// Phase 7: Multicast setup (source-side analysis, create per-region multicast objects)

#include "compile_op.h"
#include "codegen.h"
#include "compile_op_kernel.h"
#include "group.h"
#include "ipc_mapper.h"
#include "ptx_codegen.h"
#include "queue.h"
#include "serialization.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>

namespace moodist {
namespace compile_op {

namespace {

// Helper to serialize data to a CPU TensorPtr for queue operations
template<typename... T>
TensorPtr serializeToTensorPtr(const T&... v) {
  auto buffer = serializeToBuffer(v...);
  size_t size = buffer->size();
  // Create a 1D uint8 CPU tensor and copy data into it
  int64_t shape = static_cast<int64_t>(size);
  CHECK(moodist::wrapperApi.tensorEmpty != nullptr);
  Tensor* t = moodist::wrapperApi.tensorEmpty(&shape, 1, DType::UInt8, -1);
  CHECK(t != nullptr);
  // Copy serialized data into tensor
  void* tensorData = moodist::wrapperApi.tensorDataPtr(t);
  std::memcpy(tensorData, buffer->data(), size);
  return TensorPtr(t);
}

// Helper to deserialize from TensorPtr
template<typename... T>
void deserializeFromTensorPtr(const TensorPtr& tensor, T&... result) {
  void* ptr = tensor.data_ptr();
  size_t size = static_cast<size_t>(tensor.numel());
  deserializeBuffer(ptr, size, result...);
}

// Helper to format Coord for logging
std::string fmtCoord(const Coord& c) {
  return fmt::to_string(fmt::join(c, ","));
}

// Parse device string without validation.
// Returns the parsed DeviceType.
// Throws if device string is invalid.
DeviceType parseDevice(std::string_view device) {
  if (device == "cpu") {
    return DeviceType::CPU;
  }
  if (device == "cuda" || device.starts_with("cuda:")) {
    return DeviceType::CUDA;
  }
  throw std::runtime_error("Invalid device string: " + std::string(device));
}

// Parse device string and validate CUDA index if specified.
// Returns the parsed DeviceType.
// Throws if:
//   - device string is invalid
//   - cuda:N is specified but N != expectedCudaIndex
DeviceType parseAndValidateDevice(std::string_view device, int expectedCudaIndex) {
  if (device == "cpu") {
    return DeviceType::CPU;
  }
  if (device == "cuda") {
    return DeviceType::CUDA;
  }
  if (device.starts_with("cuda:")) {
    // Parse the index
    std::string_view indexStr = device.substr(5);
    int index = 0;
    for (char c : indexStr) {
      if (c < '0' || c > '9') {
        throw std::runtime_error("Invalid device string: " + std::string(device));
      }
      index = index * 10 + (c - '0');
    }
    if (index != expectedCudaIndex) {
      throw std::runtime_error("Device " + std::string(device) + " does not match process group's CUDA device " +
                               std::to_string(expectedCudaIndex));
    }
    return DeviceType::CUDA;
  }
  throw std::runtime_error("Invalid device string: " + std::string(device));
}

// Phase 6: entry sent from reader to source rank describing what will be read
struct ProvideEntry {
  uint32_t readerRank;
  uint32_t sourceInputIndex;
  size_t sourceInputOffset;
  size_t bytes;

  template<typename X>
  void serialize(X& x) {
    x(readerRank, sourceInputIndex, sourceInputOffset, bytes);
  }
};

// Phase 7: multicast assignment sent from source rank to each reader peer.
// Tells the reader: "your copy of (sourceInputIndex, sourceInputOffset, bytes)
// will be delivered via multicast; here is your VA."
struct MulticastAssignment {
  uint32_t sourceRank;
  uint32_t sourceInputIndex;
  size_t sourceInputOffset;
  size_t bytes;
  uintptr_t scratchVA; // Peer's multicast VA (where data arrives)
  size_t allocSize;    // Granularity-aligned buffer size

  template<typename X>
  void serialize(X& x) {
    x(sourceRank, sourceInputIndex, sourceInputOffset, bytes, scratchVA, allocSize);
  }
};

// ============================================================================
// Auto-tuning infrastructure for copy kernel (depth, blockSize)
// ============================================================================

// Size categories for tuning cache keying.
// Copies of similar magnitude likely have similar optimal configs.
enum class SizeCategory : uint8_t {
  Small = 0,  // 0 - 64 KB
  Medium = 1, // 64 KB - 1 MB
  Large = 2,  // 1 MB - 16 MB
  Huge = 3,   // 16 MB+
};

SizeCategory sizeCategoryFor(size_t bytes) {
  if (bytes <= 64 * 1024) {
    return SizeCategory::Small;
  }
  if (bytes <= 1024 * 1024) {
    return SizeCategory::Medium;
  }
  if (bytes <= 16 * 1024 * 1024) {
    return SizeCategory::Large;
  }
  return SizeCategory::Huge;
}

// Representative size for benchmarking each category
size_t representativeSize(SizeCategory cat) {
  switch (cat) {
  case SizeCategory::Small:
    return 32 * 1024; // 32 KB
  case SizeCategory::Medium:
    return 512 * 1024; // 512 KB
  case SizeCategory::Large:
    return 8 * 1024 * 1024; // 8 MB
  case SizeCategory::Huge:
    return 64 * 1024 * 1024; // 64 MB
  }
  return 8 * 1024 * 1024;
}

struct TuningResult {
  CopyKernelConfig config;
  float ms = INFINITY; // elapsed time in milliseconds (lower is better)
};

struct TuningKey {
  int computeArch; // e.g. sm_90 → 90
  SizeCategory sizeCat;

  bool operator==(const TuningKey& o) const {
    return computeArch == o.computeArch && sizeCat == o.sizeCat;
  }
};

// Static tuning cache: persists across compile_op calls within a process.
static std::mutex tuningCacheMutex;
static HashMap<int, HashMap<uint8_t, TuningResult>> tuningCache; // computeArch → (sizeCat → result)

bool tuningCacheLookup(const TuningKey& key, TuningResult& out) {
  auto archIt = tuningCache.find(key.computeArch);
  if (archIt == tuningCache.end()) {
    return false;
  }
  auto catIt = archIt->second.find(static_cast<uint8_t>(key.sizeCat));
  if (catIt == archIt->second.end()) {
    return false;
  }
  out = catIt->second;
  return true;
}

void tuningCacheStore(const TuningKey& key, const TuningResult& result) {
  tuningCache[key.computeArch][static_cast<uint8_t>(key.sizeCat)] = result;
}

// Format a CopyKernelConfig for log output, printing only populated fields.
std::string formatConfig(const CopyKernelConfig& c) {
  std::string s = c.copyEngine;
  if (c.depth.has_value()) {
    s += fmt::sprintf(" depth=%d", c.depth.value());
  }
  s += fmt::sprintf(" blockSize=%zu", c.blockSize);
  if (c.loadOp.has_value()) {
    s += fmt::sprintf(" loadOp=%s", c.loadOp.value());
  }
  if (c.bulkChunkSize.has_value()) {
    s += fmt::sprintf(" chunk=%zuK", c.bulkChunkSize.value() / 1024);
  }
  if (c.bulkMode.has_value()) {
    s += fmt::sprintf(" mode=%s", c.bulkMode.value());
  }
  if (c.bulkWarpLeaderDma.has_value()) {
    s += fmt::sprintf(" dma=%s", c.bulkWarpLeaderDma.value() ? "warp" : "thread");
  }
  if (c.bulkWriteBack.has_value()) {
    s += fmt::sprintf(" wb=%s", c.bulkWriteBack.value() ? "bulk" : "reg");
  }
  if (c.warppipeDepth.has_value()) {
    s += fmt::sprintf(" pipeDepth=%d", c.warppipeDepth.value());
  }
  if (c.nbufReadCount.has_value()) {
    s += fmt::sprintf(" readBufs=%d", c.nbufReadCount.value());
  }
  if (c.nbufWriteCount.has_value()) {
    s += fmt::sprintf(" writeBufs=%d", c.nbufWriteCount.value());
  }
  if (c.copyWrite.has_value()) {
    s += c.copyWrite.value() ? " write" : " read";
  }
  return s;
}

// Run the full tuning sweep for a given size category using the real
// production kernel (with NVLink copies and barriers). All ranks in the
// group must call this collectively with the same candidate sequence.
// Returns the best (depth, blockSize) config.
TuningResult tuneCopyKernel(const CompileContext& ctx, SizeCategory sizeCat) {
  static constexpr int depths[] = {1, 2, 4, 8, 16, 32};
  static constexpr size_t blockSizes[] = {128, 256, 512, 768, 1024};
  size_t gridSize = 8;
  if (auto* env = std::getenv("MOODIST_COPY_GRID_SIZE")) {
    int gs = atoi(env);
    if (gs >= 1 && gs <= 128) {
      gridSize = gs;
    }
  }
  static constexpr int warmupIters = 50;
  static constexpr int measuredIters = 100;

  size_t copyBytes = representativeSize(sizeCat);
  size_t rank = ctx.group->rank;
  size_t size = ctx.group->size;
  CUdevice cuDevice = ctx.group->cuDevice;
  IpcMapper* ipcMapper = &*ctx.group->ipcMapper;

  bool verbose = false;
  if (auto* env = std::getenv("MOODIST_TUNE_VERBOSE")) {
    verbose = !strcmp(env, "1");
  }

  if (verbose) {
    log.info("tuning copy kernel for size category %d (%zu bytes), rank %zu/%zu...\n", static_cast<int>(sizeCat),
        copyBytes, rank, size);
  }

  // Allocate synthetic src and dst buffers
  CUdeviceptr syntheticSrc = 0, syntheticDst = 0;
  CHECK_CU(cuMemAlloc(&syntheticSrc, copyBytes));
  CHECK_CU(cuMemAlloc(&syntheticDst, copyBytes));
  CHECK_CU(cuMemsetD8(syntheticSrc, 0, copyBytes));
  CHECK_CU(cuMemsetD8(syntheticDst, 0, copyBytes));

  // Build the copy descriptor.
  // Multi-GPU: IPC-map our syntheticSrc for the reader peer (ring: rank+1 reads from rank).
  // Single-GPU: local-to-local copy.
  CopyDescriptor desc;
  desc.bytes = static_cast<uint32_t>(copyBytes);

  if (size > 1) {
    // Ring pattern: rank provides syntheticSrc to (rank+1)%size
    size_t readerRank = (rank + 1) % size;
    size_t sourceRank = (rank + size - 1) % size;
    size_t readerPeerIndex = ctx.group->getPeerIndex(readerRank);
    size_t sourcePeerIndex = ctx.group->getPeerIndex(sourceRank);

    // IPC-map our syntheticSrc so the reader peer can access it
    uintptr_t mappedAddr = 0;
    ipcMapper->requestAddress(readerPeerIndex, syntheticSrc, copyBytes, &mappedAddr);
    ipcMapper->wait();

    // Barrier: all ranks have finished IPC mapping
    ctx.barrier();

    // Exchange addresses: push our mapped addr to reader, pop source's addr
    ipcMapper->push(readerPeerIndex, mappedAddr);
    uintptr_t remoteSrcAddr = ipcMapper->pop<uintptr_t>(sourcePeerIndex);

    desc.src = remoteSrcAddr;
    desc.dst = syntheticDst;
  } else {
    // Single-GPU: local copy
    desc.src = syntheticSrc;
    desc.dst = syntheticDst;
  }

  // Create stream and events for timing
  CUstream stream = nullptr;
  CHECK_CU(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, 0));
  Vector<CUevent> events; // pairs: [start0, stop0, start1, stop1, ...]

  auto createEvents = [&](int count) {
    while ((int)events.size() < count * 2) {
      CUevent e = nullptr;
      CHECK_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));
      events.push_back(e);
    }
  };

  // stepValue counter — starts at 1, increments per launch.
  // All ranks iterate the same candidates so barriers stay in sync.
  uint32_t stepValue = 1;

  TuningResult best;
  best.config.gridSize = gridSize;

  auto tuneStart = std::chrono::steady_clock::now();

  // Generate a single source with all kernel variants, compile once.
  {
    codegen::BuilderScope scope;
    emitPreamble();
    for (int depth : depths) {
      for (size_t bs : blockSizes) {
        emitCopyFunction(fmt::sprintf("copy_d%d_b%zu", depth, bs).c_str(), bs, depth);
      }
    }
    codegen::emit("} // namespace");
    codegen::emitBlank();
    for (int depth : depths) {
      for (size_t bs : blockSizes) {
        emitMainKernel(ctx.group, gridSize, bs, fmt::sprintf("compile_op_copy_d%d_b%zu", depth, bs).c_str(),
            fmt::sprintf("copy_d%d_b%zu", depth, bs).c_str());
        codegen::emitBlank();
      }
    }
    std::string source = scope.finalize();

    CompiledModule module = CompiledModule::compile(source, cuDevice);

    for (int depth : depths) {
      for (size_t bs : blockSizes) {
        auto name = fmt::sprintf("compile_op_copy_d%d_b%zu", depth, bs);
        CUfunction fn = module.getFunction(name.c_str());

        // Warmup
        for (int i = 0; i < warmupIters; i++) {
          launchCopyKernel(fn, gridSize, bs, &desc, 1, stepValue++, 0, stream);
        }
        CHECK_CU(cuStreamSynchronize(stream));

        // Measured runs: record all events without synchronizing
        createEvents(measuredIters);
        for (int iter = 0; iter < measuredIters; iter++) {
          CHECK_CU(cuEventRecord(events[iter * 2], stream));
          launchCopyKernel(fn, gridSize, bs, &desc, 1, stepValue++, 0, stream);
          CHECK_CU(cuEventRecord(events[iter * 2 + 1], stream));
        }
        CHECK_CU(cuStreamSynchronize(stream));

        // Query elapsed times
        float bestMs = INFINITY;
        for (int iter = 0; iter < measuredIters; iter++) {
          float ms = 0.0f;
          CHECK_CU(cuEventElapsedTime(&ms, events[iter * 2], events[iter * 2 + 1]));
          if (ms < bestMs) {
            bestMs = ms;
          }
        }

        if (verbose) {
          log.info("  depth=%d blockSize=%zu -> %.3f ms\n", depth, bs, bestMs);
        }

        if (bestMs < best.ms) {
          best.ms = bestMs;
          best.config.depth = depth;
          best.config.blockSize = bs;
        }
      }
    }
  }

  // Cleanup
  for (auto& e : events) {
    cuEventDestroy(e);
  }
  cuStreamDestroy(stream);
  cuMemFree(syntheticDst);
  cuMemFree(syntheticSrc);

  double elapsed = seconds(std::chrono::steady_clock::now() - tuneStart);
  log.info("tuning complete: best depth=%d blockSize=%zu (%.3f ms) in %.1fs\n", best.config.depth.value(),
      best.config.blockSize, best.ms, elapsed);

  return best;
}

// V9 variant: same tuning sweep but using PTX generation + JIT per variant
// instead of codegen DSL + single NVRTC compilation.
TuningResult tuneCopyKernelV9(const CompileContext& ctx, SizeCategory sizeCat) {
  static constexpr int depths[] = {1, 2, 4, 8, 16, 32};
  static constexpr size_t blockSizes[] = {128, 256, 512, 768, 1024};
  size_t gridSize = 8;
  if (auto* env = std::getenv("MOODIST_COPY_GRID_SIZE")) {
    int gs = atoi(env);
    if (gs >= 1 && gs <= 128) {
      gridSize = gs;
    }
  }
  static constexpr int warmupIters = 2;
  static constexpr int measuredIters = 6;

  size_t copyBytes = representativeSize(sizeCat);
  size_t rank = ctx.group->rank;
  size_t size = ctx.group->size;
  CUdevice cuDevice = ctx.group->cuDevice;
  IpcMapper* ipcMapper = &*ctx.group->ipcMapper;

  bool verbose = false;
  if (auto* env = std::getenv("MOODIST_TUNE_VERBOSE")) {
    verbose = !strcmp(env, "1");
  }

  if (verbose) {
    log.info("tuning v9 copy kernel for size category %d (%zu bytes), rank %zu/%zu...\n", static_cast<int>(sizeCat),
        copyBytes, rank, size);
  }

  // Get compute target for PTX generation
  int computeMajor = 0, computeMinor = 0;
  CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  const char* target = computeTarget(computeMajor, computeMinor);

  // Allocate synthetic src and dst buffers
  CUdeviceptr syntheticSrc = 0, syntheticDst = 0;
  CHECK_CU(cuMemAlloc(&syntheticSrc, copyBytes));
  CHECK_CU(cuMemAlloc(&syntheticDst, copyBytes));
  CHECK_CU(cuMemsetD8(syntheticSrc, 0, copyBytes));
  CHECK_CU(cuMemsetD8(syntheticDst, 0, copyBytes));

  // Build the copy descriptor
  CopyDescriptor desc;
  desc.bytes = static_cast<uint32_t>(copyBytes);

  // Set up IPC mappings for both read and write directions
  uintptr_t remoteSrcAddr = 0; // for read mode: remote peer's src buffer
  uintptr_t remoteDstAddr = 0; // for write mode: remote peer's dst buffer

  if (size > 1) {
    size_t readerRank = (rank + 1) % size;
    size_t sourceRank = (rank + size - 1) % size;
    size_t readerPeerIndex = ctx.group->getPeerIndex(readerRank);
    size_t sourcePeerIndex = ctx.group->getPeerIndex(sourceRank);

    // Map src for read mode (reader pulls from our src)
    uintptr_t mappedSrcAddr = 0;
    ipcMapper->requestAddress(readerPeerIndex, syntheticSrc, copyBytes, &mappedSrcAddr);
    ipcMapper->wait();

    // Map dst for write mode (source pushes to our dst)
    uintptr_t mappedDstAddr = 0;
    ipcMapper->requestAddress(sourcePeerIndex, syntheticDst, copyBytes, &mappedDstAddr);
    ipcMapper->wait();

    ctx.barrier();

    // Exchange read-mode addresses: push our src to reader, pop source's src
    ipcMapper->push(readerPeerIndex, mappedSrcAddr);
    remoteSrcAddr = ipcMapper->pop<uintptr_t>(sourcePeerIndex);

    // Exchange write-mode addresses: push our dst to source, pop reader's dst
    ipcMapper->push(sourcePeerIndex, mappedDstAddr);
    remoteDstAddr = ipcMapper->pop<uintptr_t>(readerPeerIndex);

    desc.src = remoteSrcAddr;
    desc.dst = syntheticDst;
  } else {
    desc.src = syntheticSrc;
    desc.dst = syntheticDst;
  }

  // Create stream and events for timing
  CUstream stream = nullptr;
  CHECK_CU(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, 0));
  Vector<CUevent> events; // pairs: [start0, stop0, start1, stop1, ...]

  auto createEvents = [&](int count) {
    while ((int)events.size() < count * 2) {
      CUevent e = nullptr;
      CHECK_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));
      events.push_back(e);
    }
  };

  uint32_t stepValue = 1;

  TuningResult best;
  best.config.gridSize = gridSize;

  auto tuneStart = std::chrono::steady_clock::now();

  // Build candidate configs
  Vector<CopyKernelConfig> candidates;

  // Read env vars for fixed-value settings (applied to all relevant candidates)
  int envWarppipeDepth = 0;
  if (auto* env = std::getenv("MOODIST_WARPPIPE_DEPTH")) {
    int d = atoi(env);
    if (d >= 0 && d <= 32) {
      envWarppipeDepth = d;
    }
  }

  // Register pipeline variants: loadOp × depth × blockSize
  static constexpr const char* loadOps[] = {"cv", "cs", "nc"};
  for (const char* loadOp : loadOps) {
    for (int depth : depths) {
      for (size_t bs : blockSizes) {
        candidates.push_back(CopyKernelConfig::reg(depth, bs, gridSize, loadOp));
      }
    }
  }

  // Bulk copy engine variants: blockSize × chunkSize × dmaMode × bulkMode
  {
    int maxSmem = 0;
    CHECK_CU(cuDeviceGetAttribute(&maxSmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice));
    // static constexpr size_t chunkSizes[] = {8192, 16384, 32768, 65536, 98304, 114688};
    static constexpr size_t chunkSizes[] = {114688};
    static constexpr size_t bulkBlockSizes[] = {32, 64, 128, 256, 512, 1024};
    for (size_t chunk : chunkSizes) {
      for (size_t bs : bulkBlockSizes) {
        size_t numWarps = bs / 32;

        // Double-buffered: total smem = 2 * chunkSize + 2 * numWarps * 8
        {
          size_t totalSmem = 2 * chunk + 2 * numWarps * 8;
          if (totalSmem <= (size_t)maxSmem) {
            // Warp-leader mode: each warp leader DMAs chunk/numWarps bytes; must be multiple of 16
            if (chunk % (numWarps * 16) == 0) {
              candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, true, false));
              candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, true, true));
            }
            // All-threads mode: each thread DMAs chunk/bs bytes; must be multiple of 16
            if (chunk % (bs * 16) == 0) {
              candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, false, false));
              candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, false, true));
            }
          }
        }

        // Warppipe: total smem = chunkSize + numWarps * 8
        // Warppipe uses single-lane DMA (warp leader always), so no dmaMode sweep.
        // stageChunk = chunk/numWarps must divide cleanly for 16-byte aligned DMAs.
        {
          size_t totalSmem = chunk + numWarps * 8;
          if (totalSmem <= (size_t)maxSmem && (chunk * 2) % (numWarps * 1024) == 0) {
            static constexpr int pipeDepths[] = {1, 2, 4, 8, 16};
            // static constexpr int pipeDepths[] = {1};
            for (int depth : pipeDepths) {
              if (envWarppipeDepth == 0 || envWarppipeDepth == depth) {
                candidates.push_back(CopyKernelConfig::warppipe(bs, gridSize, chunk * 2, depth));
              }
            }
          }
        }

        // Nbuf: single buffer + readBufs mbarriers per warp
        {
          if ((chunk * 2) % (numWarps * 1024) == 0) {
            for (int rb = 2; rb <= 4; rb++) {
              for (int wb = 0; wb <= 4; wb++) {
                size_t totalSmem = chunk + numWarps * rb * 8;
                if (totalSmem <= (size_t)maxSmem) {
                  candidates.push_back(CopyKernelConfig::nbuf(bs, gridSize, chunk * 2, rb, wb));

                  if (bs >= 64) {
                    CopyKernelConfig c;
                    c.copyEngine = "bulk";
                    c.blockSize = bs;
                    c.gridSize = gridSize;
                    c.bulkChunkSize = chunk * 2;
                    c.bulkMode = "nbuf2";
                    c.nbufReadCount = rb;
                    c.nbufWriteCount = wb;
                    c.bulkSkipWriteBack = false;
                    candidates.push_back(c);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Expand candidates into read + write variants
  {
    Vector<CopyKernelConfig> expanded;
    expanded.reserve(candidates.size() * 2);
    for (const auto& c : candidates) {
      CopyKernelConfig read = c;
      read.copyWrite = false;
      expanded.push_back(read);
      CopyKernelConfig write = c;
      write.copyWrite = true;
      expanded.push_back(write);
    }
    candidates = std::move(expanded);
  }

  // Filter candidates by env vars
  if (std::getenv("MOODIST_BULK_SKIP_WRITEBACK")) {
    for (auto& c : candidates) {
      c.bulkSkipWriteBack = true;
    } // if (totalSmem > 48 * 1024) {
      //   CHECK_CU(cuFuncSetAttribute(
      //       op->tunedKernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)totalSmem));
      // }
  }
  if (auto* env = std::getenv("MOODIST_COPY_ENGINE")) {
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                         [env](const CopyKernelConfig& c) {
                           return strcmp(c.copyEngine, env) != 0;
                         }),
        candidates.end());
  }
  if (auto* env = std::getenv("MOODIST_COPY_BLOCK_SIZE")) {
    size_t bs = atoi(env);
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                         [bs](const CopyKernelConfig& c) {
                           return c.blockSize != bs;
                         }),
        candidates.end());
  }
  if (auto* env = std::getenv("MOODIST_BULK_MODE")) {
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                         [env](const CopyKernelConfig& c) {
                           return c.bulkMode.has_value() && strcmp(c.bulkMode.value(), env) != 0;
                         }),
        candidates.end());
  }
  if (std::getenv("MOODIST_BULK_WRITEBACK")) {
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                         [](const CopyKernelConfig& c) {
                           return c.bulkWriteBack.has_value() && !c.bulkWriteBack.value();
                         }),
        candidates.end());
  }
  if (auto* env = std::getenv("MOODIST_COPY_WRITE")) {
    bool writeOnly = !strcmp(env, "1");
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                         [writeOnly](const CopyKernelConfig& c) {
                           return c.copyWrite.value_or(false) != writeOnly;
                         }),
        candidates.end());
  }

  REQUIRE(!candidates.empty(), "No tuning candidates remain after env var filtering");

  if (verbose) {
    log.info("tuning v9: %zu candidates after env var filtering\n", candidates.size());
  }

  // Evaluate all candidates
  for (const auto& cfg : candidates) {
    // Set descriptor direction based on copyWrite
    if (size > 1) {
      if (cfg.copyWrite.value_or(false)) {
        desc.src = syntheticSrc;
        desc.dst = remoteDstAddr;
      } else {
        desc.src = remoteSrcAddr;
        desc.dst = syntheticDst;
      }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::string ptx = generateCopyKernelPtx(ctx.group, cfg, target);
    double genSec = seconds(std::chrono::steady_clock::now() - t0);

    if (auto* env = std::getenv("MOODIST_DUMP_TUNE_PTX"); env && !strcmp(env, "1")) {
      fprintf(stderr, "=== PTX for %s ===\n%s\n=== END PTX ===\n", formatConfig(cfg).c_str(), ptx.c_str());
    }

    t0 = std::chrono::steady_clock::now();
    CUmodule variantModule = nullptr;
    char jitErrorLog[4096] = {};
    CUjit_option jitOpts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* jitOptVals[] = {jitErrorLog, (void*)(uintptr_t)sizeof(jitErrorLog)};
    CUresult jitErr = cuModuleLoadDataEx(&variantModule, ptx.c_str(), 2, jitOpts, jitOptVals);
    double jitSec = seconds(std::chrono::steady_clock::now() - t0);
    if (jitErr != CUDA_SUCCESS) {
      if (verbose) {
        log.info("  %s -> JIT failed (error %d), skipping\n", formatConfig(cfg), (int)jitErr);
        if (jitErrorLog[0]) {
          log.info("  ptxas: %s\n", jitErrorLog);
        }
      }
      continue;
    }

    CUfunction fn = nullptr;
    CHECK_CU(cuModuleGetFunction(&fn, variantModule, "compile_op_copy"));

    // Opt in to larger shared memory if needed (bulk copy engine)
    size_t totalSmem = 0;
    if (!strcmp(cfg.copyEngine, "bulk")) {
      size_t chunk = cfg.bulkChunkSize.value();
      size_t numWarps = cfg.blockSize / 32;
      if (!strcmp(cfg.bulkMode.value(), "warppipe")) {
        totalSmem = chunk + numWarps * 2 * 8;
      } else if (!strcmp(cfg.bulkMode.value(), "nbuf")) {
        totalSmem = chunk + numWarps * cfg.nbufReadCount.value() * 8;
      } else {
        totalSmem = 2 * chunk + 2 * numWarps * 8;
      }
    }
    // if (totalSmem > 48 * 1024) {
    //   log.info("totalSmem is %d\n", totalSmem);
    //   CHECK_CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)totalSmem));
    // }

    // Warmup
    for (int i = 0; i < warmupIters; i++) {
      launchCopyKernel(fn, gridSize, cfg.blockSize, &desc, 1, stepValue++, 0, stream);
    }
    CHECK_CU(cuStreamSynchronize(stream));

    // Measured runs: record all events without synchronizing
    t0 = std::chrono::steady_clock::now();
    createEvents(measuredIters);
    for (int iter = 0; iter < measuredIters; iter++) {
      CHECK_CU(cuEventRecord(events[iter * 2], stream));
      launchCopyKernel(fn, gridSize, cfg.blockSize, &desc, 1, stepValue++, 0, stream);
      CHECK_CU(cuEventRecord(events[iter * 2 + 1], stream));
    }
    CHECK_CU(cuStreamSynchronize(stream));
    double measureSec = seconds(std::chrono::steady_clock::now() - t0);

    // Query elapsed times
    float bestMs = INFINITY;
    for (int iter = 0; iter < measuredIters; iter++) {
      float ms = 0.0f;
      CHECK_CU(cuEventElapsedTime(&ms, events[iter * 2], events[iter * 2 + 1]));
      if (ms < bestMs) {
        bestMs = ms;
      }
    }

    if (verbose) {
      log.info("  %s -> %.3f ms  (gen=%.3fs jit=%.3fs measure=%.3fs)\n", formatConfig(cfg), bestMs, genSec, jitSec,
          measureSec);
    }

    if (bestMs < best.ms) {
      best.ms = bestMs;
      best.config = cfg;
    }

    cuModuleUnload(variantModule);
  }

  // Cleanup
  for (auto& e : events) {
    cuEventDestroy(e);
  }
  cuStreamDestroy(stream);
  cuMemFree(syntheticDst);
  cuMemFree(syntheticSrc);

  double elapsed = seconds(std::chrono::steady_clock::now() - tuneStart);
  log.info("v9 tuning complete: best %s (%.3f ms) in %.1fs\n", formatConfig(best.config), best.ms, elapsed);

  return best;
}

} // namespace

std::shared_ptr<CustomOpDescriptor> compile(const CompileContext& ctx, DType dtype,
    std::span<const api::TensorRegion> inputs, std::span<const api::TensorRegion> outputs, ReduceOp reduce,
    bool cpuSync) {

  const size_t rank = ctx.group->rank;
  const size_t size = ctx.group->size;
  size_t itemsize = wrapperApi.dtypeSize(dtype);

  // Validate per-tensorId ndim consistency
  // Different tensorIds can have different ndims (e.g., 2D weight vs 1D bias)
  HashMap<std::string, int> tensorIdNdim;

  auto validateAndRecordNdim = [&](const api::TensorRegion& region, const char* kind, size_t idx) {
    int ndim = static_cast<int>(region.offset.size());
    if (region.offset.size() != region.shape.size()) {
      throw std::runtime_error(fmt::sprintf("moodist.compile_op: %s %zu has mismatched offset/shape sizes (%zu vs %zu)",
          kind, idx, region.offset.size(), region.shape.size()));
    }
    std::string tid(region.tensorId);
    auto it = tensorIdNdim.find(tid);
    if (it != tensorIdNdim.end()) {
      if (it->second != ndim) {
        throw std::runtime_error(
            fmt::sprintf("moodist.compile_op: %s %zu has wrong dimensions for tensorId '%s' (got %d, expected %d)",
                kind, idx, tid.c_str(), ndim, it->second));
      }
    } else {
      tensorIdNdim[tid] = ndim;
    }
  };

  for (size_t i : indices(inputs)) {
    validateAndRecordNdim(inputs[i], "input", i);
  }
  for (size_t i : indices(outputs)) {
    validateAndRecordNdim(outputs[i], "output", i);
  }

  // Validate device strings for local regions (where rank == this rank)
  for (const auto& region : inputs) {
    if (static_cast<size_t>(region.rank) == rank) {
      parseAndValidateDevice(region.device, ctx.group->deviceIndex);
    }
  }
  for (const auto& region : outputs) {
    if (static_cast<size_t>(region.rank) == rank) {
      parseAndValidateDevice(region.device, ctx.group->deviceIndex);
    }
  }

  // Parse inputs and outputs into TensorDescr structs
  Vector<TensorDescr> inputDescrs;
  Vector<TensorDescr> outputDescrs;
  Vector<size_t> inputIndexCounter(size);
  Vector<size_t> outputIndexCounter(size);

  for (const auto& region : inputs) {
    TensorDescr d;
    d.rank = region.rank;
    d.index = inputIndexCounter[region.rank]++;
    d.tensorId = std::string(region.tensorId);
    d.device = parseDevice(region.device);
    int ndim = tensorIdNdim.at(d.tensorId);
    d.offset = Coord(ndim);
    d.shape = Coord(ndim);
    std::ranges::copy(region.offset, d.offset.begin());
    std::ranges::copy(region.shape, d.shape.begin());
    d.numel = numel(d.shape);
    inputDescrs.push_back(std::move(d));
  }

  for (const auto& region : outputs) {
    TensorDescr d;
    d.rank = region.rank;
    d.index = outputIndexCounter[region.rank]++;
    d.tensorId = std::string(region.tensorId);
    d.device = parseDevice(region.device);
    int ndim = tensorIdNdim.at(d.tensorId);
    d.offset = Coord(ndim);
    d.shape = Coord(ndim);
    std::ranges::copy(region.offset, d.offset.begin());
    std::ranges::copy(region.shape, d.shape.begin());
    d.numel = numel(d.shape);
    outputDescrs.push_back(std::move(d));
  }

  // Build lookup tables
  Vector<Vector<size_t>> inputsPerRank(size);
  Vector<Vector<size_t>> outputsPerRank(size);

  for (size_t i : indices(inputDescrs)) {
    inputsPerRank[inputDescrs[i].rank].push_back(i);
  }
  for (size_t i : indices(outputDescrs)) {
    outputsPerRank[outputDescrs[i].rank].push_back(i);
  }

  // Collect my inputs (inputs owned by this rank)
  Vector<size_t> myInputIndices = inputsPerRank[rank];

  // ============================================================================
  // Phase 1: Find intersections and exchange logical mappings
  // ============================================================================

  // For each output (from any rank), find intersecting inputs from myInputs
  // Create LogicalInput entries and group by destination rank
  // Only inputs and outputs with matching tensorId can intersect
  Vector<Vector<LogicalInput>> logicalInputsToSend(size);

  for (size_t outIdx : indices(outputDescrs)) {
    const TensorDescr& out = outputDescrs[outIdx];
    if (out.numel == 0) {
      continue;
    }

    // Check each of my inputs for intersection with this output
    for (size_t myInpIdx : myInputIndices) {
      const TensorDescr& inp = inputDescrs[myInpIdx];

      // Only intersect if tensorIds match
      if (inp.tensorId != out.tensorId) {
        continue;
      }

      // Get ndim for this tensorId (guaranteed to exist after validation)
      int ndim = tensorIdNdim.at(inp.tensorId);

      // Check intersection
      bool intersects = true;
      for (int d = 0; d < ndim; ++d) {
        if (inp.offset[d] >= out.offset[d] + out.shape[d] || out.offset[d] >= inp.offset[d] + inp.shape[d]) {
          intersects = false;
          break;
        }
      }

      if (!intersects) {
        continue;
      }

      // Compute intersection region
      LogicalInput li;
      li.offset = Coord(ndim);
      li.shape = Coord(ndim);
      for (int d = 0; d < ndim; ++d) {
        int64_t lo = std::max(inp.offset[d], out.offset[d]);
        int64_t hi = std::min(inp.offset[d] + inp.shape[d], out.offset[d] + out.shape[d]);
        li.offset[d] = lo;
        li.shape[d] = hi - lo;
      }
      li.inputRank = rank;
      li.inputIndex = inp.index;
      li.outputRank = out.rank;
      li.outputIndex = out.index;

      logicalInputsToSend[out.rank].push_back(std::move(li));
    }
  }

  // Barrier before queue operations
  ctx.barrier();

  // Send logical inputs to each destination rank
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(logicalInputsToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive logical inputs from all source ranks
  Vector<LogicalInput> receivedInputs;
  for (size_t i : range(size)) {
    auto [tensor, qsize] = ctx.queues[rank]->get();
    Vector<LogicalInput> batch;
    deserializeFromTensorPtr(tensor, batch);

    for (LogicalInput& li : batch) {
      CHECK(li.outputRank == rank);
      receivedInputs.push_back(std::move(li));
    }
  }

  // Debug logging
  log.debug("compile_op phase1: rank %zu, sent to %zu ranks, received %zu logical inputs\n", rank, size,
      receivedInputs.size());

  // ============================================================================
  // Phase 2: Overlap resolution via cell decomposition
  // ============================================================================

  // Group received inputs by output index
  size_t numMyOutputs = outputsPerRank[rank].size();
  Vector<Vector<size_t>> inputsPerOutput(numMyOutputs);
  for (size_t i : indices(receivedInputs)) {
    inputsPerOutput[receivedInputs[i].outputIndex].push_back(i);
  }

  // Result: for each output, a list of (cell, source input index) pairs
  // Each cell will become a ReadRequest in Phase 3
  struct CellSource {
    Coord offset;
    Coord shape;
    size_t sourceIdx; // Index into receivedInputs
  };
  Vector<Vector<CellSource>> cellsPerOutput(numMyOutputs);

  for (size_t outIdx : range(numMyOutputs)) {
    size_t globalOutIdx = outputsPerRank[rank][outIdx];
    const TensorDescr& out = outputDescrs[globalOutIdx];
    const auto& inputIdxs = inputsPerOutput[outIdx];

    if (out.numel == 0) {
      continue;
    }

    // Get ndim for this output's tensorId
    int ndim = tensorIdNdim.at(out.tensorId);

    // Collect boundaries in each dimension from output and all covering inputs
    Vector<Vector<int64_t>> boundaries(ndim);
    for (int d = 0; d < ndim; ++d) {
      boundaries[d].push_back(out.offset[d]);
      boundaries[d].push_back(out.offset[d] + out.shape[d]);
      for (size_t i : inputIdxs) {
        const LogicalInput& inp = receivedInputs[i];
        boundaries[d].push_back(inp.offset[d]);
        boundaries[d].push_back(inp.offset[d] + inp.shape[d]);
      }
      std::sort(boundaries[d].begin(), boundaries[d].end());
      boundaries[d].erase(std::unique(boundaries[d].begin(), boundaries[d].end()), boundaries[d].end());
    }

    // Compute total number of cells
    Vector<size_t> numIntervals(ndim);
    size_t totalCells = 1;
    for (int d = 0; d < ndim; ++d) {
      numIntervals[d] = boundaries[d].size() - 1;
      totalCells *= numIntervals[d];
    }

    // Iterate through all cells
    std::string gapDetails;
    std::string overlapDetails;
    size_t gapCount = 0;
    size_t overlapCount = 0;
    constexpr size_t maxReportedRegions = 5;

    for (size_t cellIdx = 0; cellIdx < totalCells; ++cellIdx) {
      // Convert cellIdx to multi-dimensional index and compute cell offset/shape
      Coord cellOffset(ndim);
      Coord cellShape(ndim);
      size_t remaining = cellIdx;
      for (int d = ndim - 1; d >= 0; --d) {
        size_t intervalIdx = remaining % numIntervals[d];
        remaining /= numIntervals[d];
        cellOffset[d] = boundaries[d][intervalIdx];
        cellShape[d] = boundaries[d][intervalIdx + 1] - boundaries[d][intervalIdx];
      }

      // Check if cell is inside output
      bool insideOutput = true;
      for (int d = 0; d < ndim; ++d) {
        if (cellOffset[d] < out.offset[d] || cellOffset[d] + cellShape[d] > out.offset[d] + out.shape[d]) {
          insideOutput = false;
          break;
        }
      }
      if (!insideOutput) {
        continue;
      }

      // Find all inputs that cover this cell
      Vector<size_t> coveringInputs;
      for (size_t i : inputIdxs) {
        const LogicalInput& inp = receivedInputs[i];
        bool covers = true;
        for (int d = 0; d < ndim; ++d) {
          if (cellOffset[d] < inp.offset[d] || cellOffset[d] + cellShape[d] > inp.offset[d] + inp.shape[d]) {
            covers = false;
            break;
          }
        }
        if (covers) {
          coveringInputs.push_back(i);
        }
      }

      if (coveringInputs.empty()) {
        // Gap detected
        gapCount++;
        if (gapCount <= maxReportedRegions) {
          gapDetails += fmt::sprintf("\n  output[%zu]: missing region at [%s] shape [%s]", outIdx,
              fmtCoord(cellOffset).c_str(), fmtCoord(cellShape).c_str());
        }
      } else if (coveringInputs.size() > 1) {
        // Overlap detected
        if (reduce == ReduceOp::None) {
          overlapCount++;
          if (overlapCount <= maxReportedRegions) {
            const LogicalInput& a = receivedInputs[coveringInputs[0]];
            const LogicalInput& b = receivedInputs[coveringInputs[1]];
            overlapDetails += fmt::sprintf("\n  output[%zu]: overlap at [%s] shape [%s] between rank %u and rank %u",
                outIdx, fmtCoord(cellOffset).c_str(), fmtCoord(cellShape).c_str(), a.inputRank, b.inputRank);
          }
        } else {
          // reduce == Any: pick one randomly
          size_t pickIdx = random<size_t>(0, coveringInputs.size() - 1);
          CellSource cs;
          cs.offset = std::move(cellOffset);
          cs.shape = std::move(cellShape);
          cs.sourceIdx = coveringInputs[pickIdx];
          cellsPerOutput[outIdx].push_back(std::move(cs));
        }
      } else {
        // Exactly one covering input
        CellSource cs;
        cs.offset = std::move(cellOffset);
        cs.shape = std::move(cellShape);
        cs.sourceIdx = coveringInputs[0];
        cellsPerOutput[outIdx].push_back(std::move(cs));
      }
    }

    // Report errors
    if (gapCount > 0) {
      std::string msg = "moodist.compile_op: missing input coverage";
      if (gapCount > maxReportedRegions) {
        msg += fmt::sprintf(" (%zu regions, showing first %zu):", gapCount, maxReportedRegions);
      } else {
        msg += ":";
      }
      msg += gapDetails;
      throw std::runtime_error(msg);
    }

    if (overlapCount > 0) {
      std::string msg = "moodist.compile_op: overlapping inputs detected";
      if (overlapCount > maxReportedRegions) {
        msg += fmt::sprintf(" (%zu regions, showing first %zu):", overlapCount, maxReportedRegions);
      } else {
        msg += ":";
      }
      msg += overlapDetails;
      throw std::runtime_error(msg);
    }
  }

  log.debug("compile_op phase2: rank %zu, completed cell decomposition\n", rank);

  // ============================================================================
  // Phase 3: Send read requests (receiver → sender)
  // ============================================================================

  // Convert cells to ReadRequests, grouped by source rank
  Vector<Vector<ReadRequest>> requestsToSend(size);

  for (size_t outIdx : range(numMyOutputs)) {
    for (const auto& cell : cellsPerOutput[outIdx]) {
      const LogicalInput& src = receivedInputs[cell.sourceIdx];

      ReadRequest req;
      req.offset = cell.offset;
      req.shape = cell.shape;
      req.requesterRank = static_cast<uint32_t>(rank);
      req.inputIndex = src.inputIndex;
      req.outputIndex = static_cast<uint32_t>(outIdx);

      requestsToSend[src.inputRank].push_back(std::move(req));
    }
  }

  // Barrier before queue operations
  ctx.barrier();

  // Send read requests to each source rank
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(requestsToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive read requests from all ranks (as a sender)
  Vector<ReadRequest> receivedRequests;
  for (size_t i : range(size)) {
    (void)i;
    auto [tensor, qsize] = ctx.queues[rank]->get();
    Vector<ReadRequest> batch;
    deserializeFromTensorPtr(tensor, batch);
    for (auto& req : batch) {
      receivedRequests.push_back(std::move(req));
    }
  }

  log.debug("compile_op phase3: rank %zu, received %zu read requests\n", rank, receivedRequests.size());

  // ============================================================================
  // Phase 4: Process requests, generate copies, send responses (sender)
  // ============================================================================

  // Track copies needed for non-contiguous reads from my inputs
  struct InputCopy {
    uint32_t inputIndex; // Which of my inputs
    Coord offset;        // Offset within that input
    Coord shape;         // Shape of the region
  };
  Vector<InputCopy> inputCopies;

  // Deduplication map: (inputIndex, offset, shape) -> copyIndex
  // Using a vector of (key, value) pairs with linear search since copy count is typically small
  struct InputCopyKey {
    uint32_t inputIndex;
    Coord offset;
    Coord shape;

    bool operator==(const InputCopyKey& other) const {
      return inputIndex == other.inputIndex && offset == other.offset && shape == other.shape;
    }
  };
  Vector<std::pair<InputCopyKey, uint32_t>> inputCopyDedup;

  // Process each request and build responses
  Vector<Vector<ReadResponse>> responsesToSend(size);

  for (const auto& req : receivedRequests) {
    // Find my input
    CHECK(req.inputIndex < myInputIndices.size());
    size_t globalInputIdx = myInputIndices[req.inputIndex];
    const TensorDescr& inp = inputDescrs[globalInputIdx];

    // Compute offset within input (request offset is in global coords)
    Coord relOffset = req.offset - inp.offset;

    // Check if this region is contiguous within the input tensor
    bool isContiguous = contiguous(req.shape, inp.shape);

    ReadResponse resp;
    resp.requesterRank = req.requesterRank;
    resp.outputIndex = req.outputIndex;
    resp.requestOffset = req.offset;
    resp.requestShape = req.shape;
    resp.senderRank = static_cast<uint32_t>(rank);
    resp.inputDevice = inp.device;

    if (isContiguous) {
      // Can read directly from input
      resp.tensorIndex = req.inputIndex;
      resp.inputOffset = linearOffset(relOffset, inp.shape);
      resp.tensorShape = inp.shape;
      resp.isCopy = false;
    } else {
      // Need to make a contiguous copy - check for deduplication first
      InputCopyKey key{req.inputIndex, relOffset, req.shape};

      // Search for existing copy with same key
      uint32_t copyIndex = 0;
      bool found = false;
      for (const auto& [existingKey, existingIdx] : inputCopyDedup) {
        if (existingKey == key) {
          copyIndex = existingIdx;
          found = true;
          break;
        }
      }

      if (!found) {
        // Create new copy
        copyIndex = static_cast<uint32_t>(myInputIndices.size() + inputCopies.size());

        InputCopy copy;
        copy.inputIndex = req.inputIndex;
        copy.offset = relOffset;
        copy.shape = req.shape;
        inputCopies.push_back(std::move(copy));

        inputCopyDedup.push_back({key, copyIndex});

        // log.debug("compile_op: input copy: inputIndex=%u, offset=[%s], shape=[%s], container=[%s]\n", req.inputIndex,
        //     fmtCoord(relOffset).c_str(), fmtCoord(req.shape).c_str(), fmtCoord(inp.shape).c_str());
      }

      resp.tensorIndex = copyIndex;
      resp.inputOffset = 0;         // Copy is contiguous, starts at 0
      resp.tensorShape = req.shape; // Copy has same shape as request
      resp.isCopy = true;
    }

    responsesToSend[req.requesterRank].push_back(std::move(resp));
  }

  // Barrier before sending responses
  ctx.barrier();

  // Send responses to each requester
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(responsesToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive responses from all ranks
  Vector<ReadResponse> receivedResponses;
  for (size_t i : range(size)) {
    (void)i;
    auto [tensor, qsize] = ctx.queues[rank]->get();
    Vector<ReadResponse> batch;
    deserializeFromTensorPtr(tensor, batch);
    for (auto& resp : batch) {
      receivedResponses.push_back(std::move(resp));
    }
  }

  log.debug("compile_op phase4: rank %zu, %zu input copies, received %zu responses\n", rank, inputCopies.size(),
      receivedResponses.size());

  // ============================================================================
  // Phase 5: Finalize and build CustomOpDescriptor (receiver)
  // ============================================================================

  auto op = std::make_shared<CustomOpDescriptor>();
  op->id = (*ctx.nextOpId)++;
  op->dtype = dtype;
  op->cpuSync = cpuSync;

  // Populate inputs (my inputs) - sizes in bytes
  for (size_t idx : myInputIndices) {
    op->inputs.push_back(itemsize * inputDescrs[idx].numel);
    const Coord& sh = inputDescrs[idx].shape;
    op->inputShapes.emplace_back(sh.begin(), sh.end());
    op->inputDevices.push_back(inputDescrs[idx].device);
  }

  // Populate outputs (my outputs) - sizes in bytes
  for (size_t idx : outputsPerRank[rank]) {
    op->outputs.push_back(itemsize * outputDescrs[idx].numel);
    const Coord& sh = outputDescrs[idx].shape;
    op->outputShapes.emplace_back(sh.begin(), sh.end());
    op->outputDevices.push_back(outputDescrs[idx].device);
  }

  // Add inputCopies from Phase 4
  for (const auto& copy : inputCopies) {
    CustomOpDescriptor::Copy c;
    c.index = copy.inputIndex;
    c.offset = {copy.offset.begin(), copy.offset.end()};
    c.shape = {copy.shape.begin(), copy.shape.end()};
    op->inputCopies.push_back(std::move(c));
  }

  // Track output copies needed for non-contiguous writes
  struct OutputCopy {
    uint32_t outputIndex;
    Coord offset; // Relative to output
    Coord shape;
  };
  Vector<OutputCopy> outputCopyList;

  // Process responses and build reads
  for (const auto& resp : receivedResponses) {
    // Find my output
    CHECK(resp.outputIndex < outputsPerRank[rank].size());
    size_t globalOutputIdx = outputsPerRank[rank][resp.outputIndex];
    const TensorDescr& out = outputDescrs[globalOutputIdx];

    // Compute output offset (request offset is in global coords)
    Coord relOutOffset = resp.requestOffset - out.offset;
    size_t outputOffset = linearOffset(relOutOffset, out.shape);

    // Check if this region is contiguous within the output tensor
    bool outputContiguous = contiguous(resp.requestShape, out.shape);

    // Compute bytes
    size_t bytes = itemsize * numel(resp.requestShape);

    // Check if input side is contiguous (for the Read entry)
    bool inputContiguous = contiguous(resp.requestShape, resp.tensorShape);

    CustomOpDescriptor::Read read;
    read.rank = resp.senderRank;
    read.inputIndex = resp.tensorIndex;
    read.outputIndex = resp.outputIndex;
    read.inputOffset = itemsize * resp.inputOffset; // Convert to bytes
    read.bytes = bytes;

    if (outputContiguous) {
      read.outputOffset = itemsize * outputOffset; // Convert to bytes
    } else {
      // Need to copy to a contiguous region, then write to output
      uint32_t copyIdx = static_cast<uint32_t>(outputsPerRank[rank].size() + outputCopyList.size());

      OutputCopy oc;
      oc.outputIndex = resp.outputIndex;
      oc.offset = relOutOffset;
      oc.shape = resp.requestShape;
      outputCopyList.push_back(std::move(oc));

      // log.debug("compile_op: output copy: outputIndex=%u, offset=[%s], shape=[%s], container=[%s]\n",
      // resp.outputIndex,
      //     fmtCoord(relOutOffset).c_str(), fmtCoord(resp.requestShape).c_str(), fmtCoord(out.shape).c_str());

      // Read into the copy buffer (which will be contiguous)
      read.outputIndex = copyIdx;
      read.outputOffset = 0;
    }

    // For CUDA-to-CUDA transfers: use local copy (NVLink/same-GPU) instead of RDMA read
    // when source is on the same node (IPC-accessible) or is this rank itself.
    // For CPU tensors, IB DMA engines are faster than memcpy, so keep as RDMA reads.
    bool bothCuda = out.device == DeviceType::CUDA && resp.inputDevice == DeviceType::CUDA;
    bool sourceIsLocal =
        read.rank == rank || (read.rank < ctx.group->ipcAccess.size() && ctx.group->ipcAccess[read.rank]);
    if (sourceIsLocal && bothCuda) {
      CustomOpDescriptor::LocalInputCopy lic;
      lic.sourceRank = read.rank;
      lic.sourceInputIndex = read.inputIndex;
      lic.sourceInputOffset = read.inputOffset;
      lic.bytes = read.bytes;
      lic.myOutputIndex = read.outputIndex;
      lic.myOutputOffset = read.outputOffset;
      op->localInputCopies.push_back(lic);
    } else {
      op->reads.push_back(read);
    }

    (void)inputContiguous; // Used by execution path, not stored in descriptor
  }

  // Split large peer copies into chunks and interleave across peers.
  // Without splitting (threshold=0), this is equivalent to staggered rotation.
  // With splitting, reads from different peers are interleaved so that
  // timing drift between ranks doesn't cause NVLink contention.
  // MOODIST_COPY_CHUNK_SIZE sets the max chunk size in bytes (0 = no splitting).
  {
    size_t chunkThreshold = 0;
    if (auto* env = std::getenv("MOODIST_COPY_CHUNK_SIZE")) {
      chunkThreshold = (size_t)atol(env);
    }

    // Group entries by peer, splitting large ones into chunks
    Vector<CustomOpDescriptor::LocalInputCopy> selfCopies;
    Vector<Vector<CustomOpDescriptor::LocalInputCopy>> peerChunks(size);

    for (auto& lic : op->localInputCopies) {
      if (lic.sourceRank == rank) {
        selfCopies.push_back(lic);
        continue;
      }
      if (chunkThreshold > 0 && lic.bytes > chunkThreshold) {
        size_t offset = 0;
        while (offset < lic.bytes) {
          size_t chunkBytes = std::min(chunkThreshold, lic.bytes - offset);
          CustomOpDescriptor::LocalInputCopy chunk = lic;
          chunk.sourceInputOffset += offset;
          chunk.myOutputOffset += offset;
          chunk.bytes = chunkBytes;
          peerChunks[lic.sourceRank].push_back(chunk);
          offset += chunkBytes;
        }
      } else {
        peerChunks[lic.sourceRank].push_back(lic);
      }
    }

    // Build rotated peer order: (rank+1)%N, (rank+2)%N, ...
    Vector<uint32_t> peerOrder;
    for (uint32_t i = 1; i < (uint32_t)size; i++) {
      peerOrder.push_back((rank + i) % size);
    }

    size_t maxRounds = 0;
    for (uint32_t peer : peerOrder) {
      maxRounds = std::max(maxRounds, peerChunks[peer].size());
    }

    // Rebuild: self-copies first, then interleaved peer copies
    op->localInputCopies.clear();
    for (auto& lic : selfCopies) {
      op->localInputCopies.push_back(std::move(lic));
    }
    for (size_t round = 0; round < maxRounds; round++) {
      for (uint32_t peer : peerOrder) {
        if (round < peerChunks[peer].size()) {
          op->localInputCopies.push_back(std::move(peerChunks[peer][round]));
        }
      }
    }

    // Log
    size_t peerCopyCount = op->localInputCopies.size() - selfCopies.size();
    if (chunkThreshold > 0 && maxRounds > 1) {
      log.info("rank %zu: %zu peer copies, %zu rounds (chunk %zuB)\n", rank, peerCopyCount, maxRounds, chunkThreshold);
    } else {
      std::string order;
      for (const auto& lic : op->localInputCopies) {
        if (lic.sourceRank == rank) {
          continue;
        }
        if (!order.empty()) {
          order += ", ";
        }
        order += fmt::sprintf("r%u(%zuB)", lic.sourceRank, lic.bytes);
      }
      log.info("rank %zu: peer read order: [%s]\n", rank, order);
    }
  }

  // Add outputCopies
  for (const auto& oc : outputCopyList) {
    CustomOpDescriptor::Copy c;
    c.index = oc.outputIndex;
    c.offset = {oc.offset.begin(), oc.offset.end()};
    c.shape = {oc.shape.begin(), oc.shape.end()};
    op->outputCopies.push_back(std::move(c));
  }

  // Final barrier
  ctx.barrier();

  log.debug("compile_op phase5: rank %zu, %zu reads, %zu local copies, %zu output copies\n", rank, op->reads.size(),
      op->localInputCopies.size(), op->outputCopies.size());

  // ============================================================================
  // Phase 6: Compute allLocal flag and build reverse mapping (localInputProvides)
  // ============================================================================

  // This rank is all-local if it has no IB reads, no non-contiguous copies, and no CPU sync
  bool myAllLocal = op->reads.empty() && op->inputCopies.empty() && op->outputCopies.empty() && !cpuSync;

  // Build messages to source ranks: tell each source what I will read from their inputs
  Vector<Vector<ProvideEntry>> providesToSend(size);
  for (const auto& lic : op->localInputCopies) {
    if (lic.sourceRank == rank) {
      continue; // Self-copies don't need to be communicated
    }
    ProvideEntry pe;
    pe.readerRank = static_cast<uint32_t>(rank);
    pe.sourceInputIndex = lic.sourceInputIndex;
    pe.sourceInputOffset = lic.sourceInputOffset;
    pe.bytes = lic.bytes;
    providesToSend[lic.sourceRank].push_back(pe);
  }

  // Barrier before queue operations
  ctx.barrier();

  // Send (myAllLocal, provides) to each rank
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(myAllLocal, providesToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive from all ranks
  bool allAllLocal = myAllLocal;
  for (size_t i : range(size)) {
    (void)i;
    auto [tensor, qsize] = ctx.queues[rank]->get();
    bool peerAllLocal;
    Vector<ProvideEntry> entries;
    deserializeFromTensorPtr(tensor, peerAllLocal, entries);
    allAllLocal = allAllLocal && peerAllLocal;

    for (const auto& pe : entries) {
      CustomOpDescriptor::LocalInputProvide lip;
      lip.readerRank = pe.readerRank;
      lip.myInputIndex = pe.sourceInputIndex;
      lip.inputOffset = pe.sourceInputOffset;
      lip.bytes = pe.bytes;
      op->localInputProvides.push_back(lip);
    }
  }

  op->allLocal = allAllLocal;

  // Final barrier after Phase 6
  ctx.barrier();

  log.debug(
      "compile_op phase6: rank %zu, allLocal=%d, %zu provides\n", rank, op->allLocal, op->localInputProvides.size());

  // ============================================================================
  // Phase 7: Multicast setup (if eligible)
  // ============================================================================

  if (op->allLocal && ctx.group->supportsMulticast && ctx.group->compileOpKernels->version == 3) {
    // Analyze localInputProvides (source-side view): group by source region.
    // If a region is read by ≥2 peers, create a multicast object for it.
    // The source rank creates the multicast object and sends the handle to each reader.

    struct RegionKey {
      uint32_t inputIndex;
      size_t inputOffset;
      size_t bytes;

      bool operator==(const RegionKey& other) const {
        return inputIndex == other.inputIndex && inputOffset == other.inputOffset && bytes == other.bytes;
      }
    };

    // Group localInputProvides by source region → list of reader ranks
    Vector<std::pair<RegionKey, Vector<uint32_t>>> regionToReaders;

    for (const auto& lip : op->localInputProvides) {
      RegionKey key{lip.myInputIndex, lip.inputOffset, lip.bytes};

      bool found = false;
      for (auto& [k, readers] : regionToReaders) {
        if (k == key) {
          readers.push_back(lip.readerRank);
          found = true;
          break;
        }
      }
      if (!found) {
        Vector<uint32_t> readers;
        readers.push_back(lip.readerRank);
        regionToReaders.emplace_back(key, std::move(readers));
      }
    }

    // Pending multicast objects: Phase 7a stores these, Phase 7b completes them.
    struct PendingMulticast {
      RegionKey key;
      CUmemGenericAllocationHandle mcHandle;
      size_t mcSize; // mcProp.size (rounded up to multicast granularity)
      size_t mcGranularity;
      Vector<uint32_t> readers;
      HashMap<uint32_t, size_t> readerHandleIndices; // readerRank → handleIndex from Phase 1
    };
    Vector<PendingMulticast> pendingMulticasts;

    bool useFabric = ctx.group->supportsFabric;

    // Phase 7a: Create multicast objects, add source device, send addDevice to peers.
    // After this phase + wait(), ALL devices have been added to each multicast object.
    for (const auto& [key, readers] : regionToReaders) {
      if (readers.size() < 1) {
        continue;
      }

      // numDevices = source + all readers
      CUmulticastObjectProp mcProp;
      std::memset(&mcProp, 0, sizeof(mcProp));
      mcProp.numDevices = static_cast<unsigned int>(readers.size() + 1);
      mcProp.size = key.bytes;
      mcProp.handleTypes =
          useFabric ? (unsigned long long)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC)
                    : (unsigned long long)CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      mcProp.flags = 0;

      size_t mcGranularity = 0;
      CHECK_CU(cuMulticastGetGranularity(&mcGranularity, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
      mcProp.size = (mcProp.size + mcGranularity - 1) / mcGranularity * mcGranularity;

      CUdevice dev;
      CHECK_CU(cuCtxGetDevice(&dev));

      CUmemGenericAllocationHandle mcHandle;
      log.info("compile_op phase7a: cuMulticastCreate numDevices=%u size=%zu handleTypes=%#llx flags=%#llx\n",
          mcProp.numDevices, mcProp.size, mcProp.handleTypes, mcProp.flags);
      CHECK_CU(cuMulticastCreate(&mcHandle, &mcProp));

      // Source adds its own device
      CHECK_CU(cuMulticastAddDevice(mcHandle, dev));

      // Send addDevice requests to all readers (Phase 1 — import + addDevice only)
      PendingMulticast pm;
      pm.key = key;
      pm.mcHandle = mcHandle;
      pm.mcSize = mcProp.size;
      pm.mcGranularity = mcGranularity;
      pm.readers = readers;

      for (uint32_t readerRank : readers) {
        size_t peerIndex = ctx.group->getPeerIndex(readerRank);
        ++ctx.group->ipcMapper->waitCount;
        ctx.group->ipcMapper->sendMulticastHandle(peerIndex, mcHandle, mcProp.size,
            [ipcMapper = ctx.group->ipcMapper.get(), readerRank, &pm](uintptr_t handleIndex) {
              pm.readerHandleIndices[readerRank] = handleIndex;
              --ipcMapper->waitCount;
            });
      }

      // Wait for all addDevice callbacks for this MC object before moving pm
      ctx.group->ipcMapper->wait();

      pendingMulticasts.push_back(std::move(pm));
    }

    log.info("compile_op phase7a: all addDevice complete, %zu multicast objects\n", pendingMulticasts.size());

    // Phase 7b: Now that all devices are added, bind scratch and map on all devices.
    struct CreatedRegion {
      RegionKey key;
      CUdeviceptr sourceVA;
      size_t allocSize;
      HashMap<uint32_t, CUdeviceptr> readerVAs; // readerRank → peer's multicast VA
    };
    Vector<CreatedRegion> createdRegions;

    for (auto& pm : pendingMulticasts) {
      CUdevice dev;
      CHECK_CU(cuCtxGetDevice(&dev));

      // Source's scratch buffer
      CUmemAllocationProp allocProp;
      std::memset(&allocProp, 0, sizeof(allocProp));
      allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      allocProp.location.id = dev;
      allocProp.requestedHandleTypes = useFabric ? CU_MEM_HANDLE_TYPE_FABRIC : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

      size_t allocGranularity = 0;
      CHECK_CU(cuMemGetAllocationGranularity(&allocGranularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
      size_t allocSize = (pm.mcSize + allocGranularity - 1) / allocGranularity * allocGranularity;

      CUmemGenericAllocationHandle scratchHandle;
      CHECK_CU(cuMemCreate(&scratchHandle, allocSize, &allocProp, 0));
      CHECK_CU(cuMulticastBindMem(pm.mcHandle, 0, scratchHandle, 0, allocSize, 0));

      // Map the multicast object on source
      CUdeviceptr mcVA = 0;
      CHECK_CU(cuMemAddressReserve(&mcVA, allocSize, pm.mcGranularity, 0, 0));
      CHECK_CU(cuMemMap(mcVA, allocSize, 0, pm.mcHandle, 0));

      CUmemAccessDesc accessDesc;
      std::memset(&accessDesc, 0, sizeof(accessDesc));
      accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      accessDesc.location.id = dev;
      accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      CHECK_CU(cuMemSetAccess(mcVA, allocSize, &accessDesc, 1));

      // Send bind requests to each reader peer (Phase 2 — scratch + bind + map)
      CreatedRegion cr;
      cr.key = pm.key;
      cr.sourceVA = mcVA;
      cr.allocSize = allocSize;

      for (uint32_t readerRank : pm.readers) {
        size_t peerIndex = ctx.group->getPeerIndex(readerRank);
        size_t handleIndex = pm.readerHandleIndices.at(readerRank);
        ++ctx.group->ipcMapper->waitCount;
        ctx.group->ipcMapper->sendMulticastBind(peerIndex, handleIndex, pm.mcSize,
            [ipcMapper = ctx.group->ipcMapper.get(), readerRank, &cr](uintptr_t peerVA) {
              cr.readerVAs[readerRank] = peerVA;
              --ipcMapper->waitCount;
            });
      }

      ctx.group->ipcMapper->wait();
      createdRegions.push_back(std::move(cr));

      log.info("compile_op phase7b: created multicast for input[%u]+%zu (%zu bytes), "
               "%zu readers, VA=%#llx\n",
          pm.key.inputIndex, pm.key.inputOffset, pm.key.bytes, pm.readers.size(), (unsigned long long)mcVA);
    }

    // Barrier: ensure all ipc_mapper handler work is complete on all ranks
    ctx.barrier();

    // Exchange multicast assignments via queues.
    // Each source sends assignments to each peer; non-sources send empty lists.
    Vector<Vector<MulticastAssignment>> assignmentsToSend(size);

    for (const auto& cr : createdRegions) {
      for (const auto& [readerRank, peerVA] : cr.readerVAs) {
        MulticastAssignment ma;
        ma.sourceRank = static_cast<uint32_t>(rank);
        ma.sourceInputIndex = cr.key.inputIndex;
        ma.sourceInputOffset = cr.key.inputOffset;
        ma.bytes = cr.key.bytes;
        ma.scratchVA = peerVA;
        ma.allocSize = cr.allocSize;
        assignmentsToSend[readerRank].push_back(ma);
      }
    }

    ctx.barrier();

    for (size_t destRank : range(size)) {
      TensorPtr tensor = serializeToTensorPtr(assignmentsToSend[destRank]);
      ctx.queues[destRank]->put(std::move(tensor), 0);
    }

    // Receive assignments from all ranks
    Vector<MulticastAssignment> receivedAssignments;
    for (size_t i : range(size)) {
      (void)i;
      auto [tensor, qsize] = ctx.queues[rank]->get();
      Vector<MulticastAssignment> entries;
      deserializeFromTensorPtr(tensor, entries);
      for (auto& ma : entries) {
        receivedAssignments.push_back(std::move(ma));
      }
    }

    // Build MulticastSource entries (one per created region on this rank).
    // Build MulticastDest entries from received assignments; remove from localInputCopies.
    // Self-copies (sourceRank == rank) stay in localInputCopies — handled by the copy path.

    for (const auto& cr : createdRegions) {
      CustomOpDescriptor::MulticastSource ms;
      ms.sourceInputIndex = cr.key.inputIndex;
      ms.sourceInputOffset = cr.key.inputOffset;
      ms.bytes = cr.key.bytes;
      ms.mcVA = cr.sourceVA;
      op->multicastSources.push_back(ms);
    }

    for (const auto& ma : receivedAssignments) {
      bool matched = false;
      for (auto it = op->localInputCopies.begin(); it != op->localInputCopies.end(); ++it) {
        if (it->sourceRank == ma.sourceRank && it->sourceInputIndex == ma.sourceInputIndex &&
            it->sourceInputOffset == ma.sourceInputOffset && it->bytes == ma.bytes) {
          CustomOpDescriptor::MulticastDest md;
          md.sourceRank = ma.sourceRank;
          md.sourceInputIndex = ma.sourceInputIndex;
          md.sourceInputOffset = ma.sourceInputOffset;
          md.bytes = ma.bytes;
          md.myOutputIndex = it->myOutputIndex;
          md.myOutputOffset = it->myOutputOffset;
          md.scratchVA = ma.scratchVA;
          md.allocSize = ma.allocSize;
          op->multicastDests.push_back(md);
          op->localInputCopies.erase(it);
          matched = true;
          break;
        }
      }
      CHECK(matched);
    }

    log.info("compile_op phase7: rank %zu, %zu multicast sources, %zu multicast dests, "
             "%zu remaining localInputCopies\n",
        rank, op->multicastSources.size(), op->multicastDests.size(), op->localInputCopies.size());

    // Final barrier after Phase 7
    ctx.barrier();
  }

  // Compute byte counts for logging
  size_t inputBytes = 0, outputBytes = 0, readBytes = 0;
  for (const auto& d : inputDescrs) {
    if (d.rank == rank) {
      inputBytes += d.numel * itemsize;
    }
  }
  for (const auto& d : outputDescrs) {
    if (d.rank == rank) {
      outputBytes += d.numel * itemsize;
    }
  }
  size_t localCopyBytes = 0;
  for (const auto& r : op->reads) {
    readBytes += r.bytes;
  }
  for (const auto& lc : op->localInputCopies) {
    localCopyBytes += lc.bytes;
  }

  log.info("compile_op[%u]: rank %zu/%zu, %zu inputs (%zu bytes), %zu outputs (%zu bytes), %zu tensor_ids "
           "-> %zu reads (%zu bytes), %zu local copies (%zu bytes), %zu input copies, %zu output copies\n",
      op->id, rank, size, myInputIndices.size(), inputBytes, outputsPerRank[rank].size(), outputBytes,
      tensorIdNdim.size(), op->reads.size(), readBytes, op->localInputCopies.size(), localCopyBytes,
      op->inputCopies.size(), op->outputCopies.size());

  // ============================================================================
  // Auto-tuning: compile a per-op copy kernel with tuned (depth, blockSize)
  // ============================================================================
  // For v8 (codegen DSL) and v9 (PTX) all-local ops with CUDA copies.
  // Tuning runs microbenchmarks on this GPU and caches results by (arch, size category).

  int kernelVersion = ctx.group->compileOpKernels->version;

  if ((kernelVersion == 8 || kernelVersion == 9) && op->allLocal && !op->localInputCopies.empty()) {
    // Find the dominant size category (largest total bytes per category)
    size_t bytesPerCat[4] = {};
    for (const auto& lic : op->localInputCopies) {
      SizeCategory cat = sizeCategoryFor(lic.bytes);
      bytesPerCat[static_cast<int>(cat)] += lic.bytes;
    }
    SizeCategory dominantCat = SizeCategory::Small;
    size_t maxBytes = 0;
    for (int i = 0; i < 4; i++) {
      if (bytesPerCat[i] > maxBytes) {
        maxBytes = bytesPerCat[i];
        dominantCat = static_cast<SizeCategory>(i);
      }
    }

    // Get compute arch for cache key
    CUdevice cuDevice = ctx.group->cuDevice;
    int computeMajor = 0, computeMinor = 0;
    CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int computeArch = computeMajor * 10 + computeMinor;

    // Use separate cache keys for v8 and v9 so they don't interfere
    TuningKey key{computeArch * 100 + kernelVersion, dominantCat};
    TuningResult result;
    bool needsTuning = false;

    {
      std::lock_guard lock(tuningCacheMutex);
      needsTuning = !tuningCacheLookup(key, result);
    }

    if (needsTuning) {
      // Reset barrier arrays before tuning — tuning kernels use the same
      // barrier memory as production kernels and would leave stale values.
      ctx.resetBarriers();

      if (kernelVersion == 9) {
        result = tuneCopyKernelV9(ctx, dominantCat);
      } else {
        result = tuneCopyKernel(ctx, dominantCat);
      }

      // Reset barrier arrays after tuning — clear residual step values
      // from tuning iterations before production kernels start.
      ctx.resetBarriers();
      std::lock_guard lock(tuningCacheMutex);
      // Double-check: another thread might have tuned while we were running
      TuningResult existing;
      if (!tuningCacheLookup(key, existing)) {
        tuningCacheStore(key, result);
      } else {
        result = existing;
      }
    }

    // Debug override: skip shared→global write-back (applied post-tuning since
    // it doesn't affect tuning performance — the write-back is just skipped).
    if (std::getenv("MOODIST_BULK_SKIP_WRITEBACK")) {
      result.config.bulkSkipWriteBack = true;
    }

    // Generate and compile the final kernel with the tuned config
    log.info("compile_op[%u]: auto-tuned kernel %s (%.3f ms)\n", op->id, formatConfig(result.config), result.ms);

    if (kernelVersion == 9) {
      const char* target = computeTarget(computeMajor, computeMinor);
      std::string ptx = generateCopyKernelPtx(ctx.group, result.config, target);
      if (std::getenv("MOODIST_DUMP_KERNELS")) {
        std::string fn = fmt::sprintf("moodist-compile-op-kernels-rank%zu-op%u.ptx", rank, op->id);
        FILE* f = fopen(fn.c_str(), "wb");
        if (f) {
          fwrite(ptx.data(), ptx.size(), 1, f);
          fclose(f);
          log.info("compile_op PTX dumped to %s\n", fn);
        }
      }
      op->tunedKernel = std::make_unique<CompiledKernel>(compileKernelPtx(ptx, "compile_op_copy"));
      // Opt in to larger shared memory if needed (bulk copy engine)
      if (!strcmp(result.config.copyEngine, "bulk")) {
        size_t chunk = result.config.bulkChunkSize.value();
        size_t numWarps = result.config.blockSize / 32;
        size_t totalSmem;
        if (!strcmp(result.config.bulkMode.value(), "warppipe")) {
          totalSmem = chunk + numWarps * 2 * 8;
        } else if (!strcmp(result.config.bulkMode.value(), "nbuf")) {
          totalSmem = chunk + numWarps * result.config.nbufReadCount.value() * 8;
        } else {
          totalSmem = 2 * chunk + 2 * numWarps * 8;
        }
        // if (totalSmem > 48 * 1024) {
        //   CHECK_CU(cuFuncSetAttribute(
        //       op->tunedKernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)totalSmem));
        // }
      }
    } else {
      std::string finalSource =
          generateCopyKernel(ctx.group, result.config.gridSize, result.config.blockSize, result.config.depth.value());
      op->tunedKernel = std::make_unique<CompiledKernel>(compileKernel(
          finalSource, "compile_op_copy", cuDevice, fmt::sprintf("moodist-tuned-rank%zu-op%u", rank, op->id).c_str()));
    }
    op->tunedConfig = result.config;
  }

  return op;
}

} // namespace compile_op
} // namespace moodist
