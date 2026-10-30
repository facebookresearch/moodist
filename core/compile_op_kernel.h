// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

namespace moodist {

struct Group;

struct CompileOpCopyParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  // NVLS: the value of the op's step counter (of this concurrency slot) before this launch (see generateKernel)
  uint32_t nvlsCounterBase = 0;
};

struct KernelConfig {
  uint32_t blockSize = 128;
  uint32_t gridSize = 8;

  uint32_t sharedMemory = 0;

  // Bytes per chunk, and pipelines (issuing warps) per block
  uint32_t chunkSize = 32768;
  uint32_t pipelines = 1;
};

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

  // Get a function handle by name from the loaded module.
  CUfunction getFunction(const char* name) const;
};

struct CompiledKernel {
  CompiledModule module;
  CUfunction function = nullptr;
  // The module's compile_op_params kernel, if it has one (see generateKernel)
  CUfunction paramsFunction = nullptr;

  CompiledKernel() = default;
  CompiledKernel(CompiledKernel&&) = default;
  CompiledKernel& operator=(CompiledKernel&&) = default;
};

// Compile a PTX string via cuModuleLoadDataEx JIT and return a CompiledKernel.
CompiledKernel compileKernelPtx(const std::string& ptx, const char* functionName);

} // namespace moodist
