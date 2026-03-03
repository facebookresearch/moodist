// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

namespace moodist {

struct Group;

// Max descriptors that fit in 4KB kernel params.
// Each descriptor: 20 bytes (src:8 + dst:8 + bytes:4)
// Overhead: ~16 bytes (stepValue, concurrencyIndex, numDescriptors, padding)
// (4096 - 16) / 20 = ~204
static constexpr uint32_t kMaxCopyDescriptors = 200;

struct CopyDescriptor {
  uintptr_t src;
  uintptr_t dst;
  uint32_t bytes; // Max 4GB per region, sufficient for tensor slices
};

struct CompileOpCopyParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  uint32_t numDescriptors;
  uint32_t _pad;
  CopyDescriptor descriptors[kMaxCopyDescriptors];
};

// Configuration for the PTX copy kernel generator and auto-tuner.
// Engine-specific fields are std::optional — only populated for the relevant engine type.
// Accessing an unpopulated field throws std::bad_optional_access.
struct CopyKernelConfig {
  // Universal fields (all engine types)
  size_t blockSize = 256;
  size_t gridSize = 8;
  const char* copyEngine = "reg"; // "reg" (register pipeline), "bulk" (cp.async.bulk)

  // Reg-only fields
  std::optional<int> depth;          // pipeline depth (1-32)
  std::optional<const char*> loadOp; // "cv" (cache volatile), "cs" (cache streaming), "nc" (non-coherent)

  // Bulk-only fields
  std::optional<size_t> bulkChunkSize;   // staging buffer size (bytes)
  std::optional<const char*> bulkMode;   // "doublebuf", "warppipe", or "nbuf"
  std::optional<bool> bulkSkipWriteBack; // debug: skip shared→global write-back
  std::optional<bool> bulkWriteBack;     // true: use cp.async.bulk for write-back (shared→global DMA)

  // Bulk doublebuf-only fields
  std::optional<bool> bulkWarpLeaderDma; // true: 1 DMA per warp, false: 1 DMA per thread

  // Bulk warppipe-only fields
  std::optional<int> warppipeDepth; // 0 = use numWarps, >0 = limit pipeline depth (max 15)

  // Bulk nbuf-only fields
  std::optional<int> nbufReadCount;  // number of read buffers per warp (1-4)
  std::optional<int> nbufWriteCount; // number of write buffers per warp (1-4)

  // Flexbuf-only fields
  std::optional<const char*> flexReadMethod;  // "bulk" or "reg"
  std::optional<const char*> flexWriteMethod; // "bulk" or "reg"
  std::optional<int> flexReadThreads;         // threads in read group (bulk always 32)
  std::optional<int> flexWriteThreads;        // threads in write group (bulk always 32)
  std::optional<int> flexNumBuffers;          // number of shared memory ring buffers
  std::optional<int> flexNumParallelReads;    // number of parallel read units (read threads split evenly)
  std::optional<int> flexNumParallelWrites;   // number of parallel write units (write threads split evenly)

  // Lockstep-only fields
  std::optional<int> lockstepNumBuffers;    // number of shared memory ring buffers
  std::optional<int> lockstepNumParallel;   // number of parallel read/write units
  std::optional<bool> lockstepMulticast;    // use multimem.cp.async.bulk for writes

  // IPC direction (applies to all engine types)
  std::optional<bool> copyWrite; // true: push (write to remote dst), false: pull (read from remote src)

  // Factory functions for constructing engine-specific configs
  static CopyKernelConfig reg(int depth, size_t blockSize, size_t gridSize, const char* loadOp) {
    CopyKernelConfig c;
    c.copyEngine = "reg";
    c.blockSize = blockSize;
    c.gridSize = gridSize;
    c.depth = depth;
    c.loadOp = loadOp;
    return c;
  }

  static CopyKernelConfig flexbuf(size_t gridSize, size_t chunkSize, const char* readMethod, int readThreads,
      int numParallelReads, const char* writeMethod, int writeThreads, int numParallelWrites, int numBuffers) {
    CopyKernelConfig c;
    c.copyEngine = "flexbuf";
    c.gridSize = gridSize;
    c.blockSize = readThreads * numParallelReads + writeThreads * numParallelWrites;
    c.bulkChunkSize = chunkSize;
    c.bulkSkipWriteBack = false;
    c.flexReadMethod = readMethod;
    c.flexWriteMethod = writeMethod;
    c.flexReadThreads = readThreads;
    c.flexWriteThreads = writeThreads;
    c.flexNumBuffers = numBuffers;
    c.flexNumParallelReads = numParallelReads;
    c.flexNumParallelWrites = numParallelWrites;
    return c;
  }

  static CopyKernelConfig lockstep(
      size_t gridSize, size_t chunkSize, size_t blockSize, int numBuffers, int numParallel,
      bool multicast = false) {
    CopyKernelConfig c;
    c.copyEngine = "lockstep";
    c.gridSize = gridSize;
    c.blockSize = blockSize;
    c.bulkChunkSize = chunkSize;
    c.bulkSkipWriteBack = false;
    c.lockstepNumBuffers = numBuffers;
    c.lockstepNumParallel = numParallel;
    c.lockstepMulticast = multicast;
    return c;
  }

  static CopyKernelConfig bulk(
      size_t blockSize, size_t gridSize, size_t chunkSize, bool warpLeaderDma, bool writeBack) {
    CopyKernelConfig c;
    c.copyEngine = "bulk";
    c.blockSize = blockSize;
    c.gridSize = gridSize;
    c.bulkChunkSize = chunkSize;
    c.bulkMode = "doublebuf";
    c.bulkWarpLeaderDma = warpLeaderDma;
    c.bulkSkipWriteBack = false;
    c.bulkWriteBack = writeBack;
    return c;
  }

  static CopyKernelConfig warppipe(size_t blockSize, size_t gridSize, size_t chunkSize, int pipeDepth) {
    CopyKernelConfig c;
    c.copyEngine = "bulk";
    c.blockSize = blockSize;
    c.gridSize = gridSize;
    c.bulkChunkSize = chunkSize;
    c.bulkMode = "warppipe";
    c.warppipeDepth = pipeDepth;
    c.bulkSkipWriteBack = false;
    return c;
  }

  static CopyKernelConfig nbuf(size_t blockSize, size_t gridSize, size_t chunkSize, int readCount, int writeCount) {
    CopyKernelConfig c;
    c.copyEngine = "bulk";
    c.blockSize = blockSize;
    c.gridSize = gridSize;
    c.bulkChunkSize = chunkSize;
    c.bulkMode = "nbuf";
    c.nbufReadCount = readCount;
    c.nbufWriteCount = writeCount;
    c.bulkSkipWriteBack = false;
    return c;
  }
};

struct CompileOpKernels {
  Group* group;
  CUmodule cuModule = nullptr;
  CUfunction cuCopyKernel = nullptr;

  CUmodule cuMulticastModule = nullptr;
  CUfunction cuMulticastKernel = nullptr;

  size_t gridSize = 8;
  size_t blockSize = 256;
  size_t dynamicSmemBytes = 0; // For v4 cp.async kernel

  int version = 0; // From MOODIST_COPY_KERNEL env var

  CompileOpKernels(Group* group);
  ~CompileOpKernels();
  void compile();
  void compileMulticast();
};

// ---------------------------------------------------------------------------
// Standalone kernel compilation via NVRTC.
//
// CompiledModule owns a CUmodule and can return function handles by name.
// CompiledKernel is a convenience wrapper holding a module + single function.
// ---------------------------------------------------------------------------

struct CompiledModule {
  CUmodule module = nullptr;

  CompiledModule() = default;
  CompiledModule(CompiledModule&& o) noexcept : module(o.module) {
    o.module = nullptr;
  }
  CompiledModule(const CompiledModule&) = delete;
  CompiledModule& operator=(const CompiledModule&) = delete;
  CompiledModule& operator=(CompiledModule&& o) noexcept {
    if (this != &o) {
      if (module) {
        cuModuleUnload(module);
      }
      module = o.module;
      o.module = nullptr;
    }
    return *this;
  }
  ~CompiledModule();

  // Compile CUDA source for the given device. Handles NVRTC loading,
  // arch detection/fallback, error logging, optional source dump.
  static CompiledModule compile(const std::string& source, CUdevice device, const char* dumpPrefix = nullptr);

  // Get a function handle by name from the loaded module.
  CUfunction getFunction(const char* name) const;
};

struct CompiledKernel {
  CompiledModule module;
  CUfunction function = nullptr;

  CompiledKernel() = default;
  CompiledKernel(CompiledKernel&&) = default;
  CompiledKernel& operator=(CompiledKernel&&) = default;
};

CompiledKernel compileKernel(
    const std::string& source, const char* functionName, CUdevice device, const char* dumpPrefix = nullptr);

// Compile a PTX string via cuModuleLoadDataEx JIT and return a CompiledKernel.
CompiledKernel compileKernelPtx(const std::string& ptx, const char* functionName);

// Launch a copy kernel with the standard CompileOpCopyParameters layout.
// Builds the parameter struct from the given arguments and calls cuLaunchKernel.
void launchCopyKernel(CUfunction kernel, size_t gridSize, size_t blockSize, const CopyDescriptor* descriptors,
    uint32_t numDescriptors, uint32_t stepValue, uint32_t concurrencyIndex, CUstream stream,
    size_t dynamicSmemBytes = 0);

} // namespace moodist
