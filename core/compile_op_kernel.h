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
// Takes a CUDA source string, compiles it for the given device, and returns
// a loaded module + function handle. Handles architecture detection,
// error logging, and optional dump to file (MOODIST_DUMP_KERNELS).
// ---------------------------------------------------------------------------

struct CompiledKernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;

  CompiledKernel() = default;
  CompiledKernel(CompiledKernel&& o) noexcept : module(o.module), function(o.function) {
    o.module = nullptr;
    o.function = nullptr;
  }
  CompiledKernel(const CompiledKernel&) = delete;
  CompiledKernel& operator=(const CompiledKernel&) = delete;
  CompiledKernel& operator=(CompiledKernel&& o) noexcept {
    if (this != &o) {
      if (module) {
        cuModuleUnload(module);
      }
      module = o.module;
      function = o.function;
      o.module = nullptr;
      o.function = nullptr;
    }
    return *this;
  }
  ~CompiledKernel();
};

CompiledKernel compileKernel(
    const std::string& source, const char* functionName, CUdevice device, const char* dumpPrefix = nullptr);

// Launch a copy kernel with the standard CompileOpCopyParameters layout.
// Builds the parameter struct from the given arguments and calls cuLaunchKernel.
void launchCopyKernel(CUfunction kernel, size_t gridSize, size_t blockSize, const CopyDescriptor* descriptors,
    uint32_t numDescriptors, uint32_t stepValue, uint32_t concurrencyIndex, CUstream stream,
    size_t dynamicSmemBytes = 0);

} // namespace moodist
