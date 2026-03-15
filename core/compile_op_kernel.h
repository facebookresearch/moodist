// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

namespace moodist {

struct Group;

struct CopyDescriptor {
  uintptr_t src;
  uintptr_t dst;
};

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

// struct CompileOpKernels {
//   Group* group;
//   CUmodule cuModule = nullptr;
//   CUfunction cuKernel = nullptr;

//   size_t gridSize = 8;
//   size_t blockSize = 256;

//   int version = 0; // From MOODIST_COPY_KERNEL env var

//   CompileOpKernels(Group* group);
//   ~CompileOpKernels();
//   void compile();
//   void compileMulticast();
// };

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

  // // Compile CUDA source for the given device. Handles NVRTC loading,
  // // arch detection/fallback, error logging, optional source dump.
  // static CompiledModule compile(const std::string& source, CUdevice device, const char* dumpPrefix = nullptr);

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

// CompiledKernel compileKernel(
//     const std::string& source, const char* functionName, CUdevice device, const char* dumpPrefix = nullptr);

// Compile a PTX string via cuModuleLoadDataEx JIT and return a CompiledKernel.

CompiledKernel compileKernelPtx(const std::string& ptx, const char* functionName);
CompiledKernel compileKernelCubin(const std::string& cubin, const char* functionName);

// // Launch a copy kernel with the standard CompileOpCopyParameters layout.
// // Builds the parameter struct from the given arguments and calls cuLaunchKernel.
// void launchCopyKernel(CUfunction kernel, size_t gridSize, size_t blockSize, const CopyDescriptor* descriptors,
//     uint32_t numDescriptors, uint32_t stepValue, uint32_t concurrencyIndex, CUstream stream,
//     size_t dynamicSmemBytes = 0);

std::string compilePtxToCubin(const std::string_view ptx);

} // namespace moodist
