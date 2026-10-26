// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

namespace moodist {

struct Group;

struct CompileOpCopyParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
};

struct KernelConfig {
  std::string copyEngine;

  uint32_t blockSize = 256;
  uint32_t gridSize = 8;

  uint32_t sharedMemory = 0;
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

  CompiledKernel() = default;
  CompiledKernel(CompiledKernel&&) = default;
  CompiledKernel& operator=(CompiledKernel&&) = default;
};

// Compile a PTX string via cuModuleLoadDataEx JIT and return a CompiledKernel.
CompiledKernel compileKernelPtx(const std::string& ptx, const char* functionName);

} // namespace moodist
