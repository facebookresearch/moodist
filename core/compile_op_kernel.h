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
struct CopyKernelConfig {
  int depth = 4;
  size_t blockSize = 256;
  size_t gridSize = 8;
  const char* loadOp = "cv";      // "cv" (cache volatile), "cs" (cache streaming), "nc" (non-coherent)
  const char* copyEngine = "reg"; // "reg" (register pipeline), "bulk" (cp.async.bulk)
  size_t bulkChunkSize = 16384;   // staging buffer size for bulk engine (bytes)
  bool bulkWarpLeaderDma = true;  // true: 1 DMA per warp, false: 1 DMA per thread
  bool bulkSkipWriteBack = false; // debug: skip shared→global write-back
  bool bulkWriteBack = false;     // true: use cp.async.bulk for write-back (shared→global DMA)
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
