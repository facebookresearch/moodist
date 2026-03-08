// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "compile_op_kernel.h"
#include "common.h"
#include "group.h"
#include "ptx_codegen.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace moodist {

CompileOpKernels::CompileOpKernels(Group* group) : group(group) {
  const char* env = std::getenv("MOODIST_COPY_KERNEL");
  if (env) {
    if (!strcmp(env, "v1")) {
      version = 1;
    } else if (!strcmp(env, "v2")) {
      version = 2;
    } else if (!strcmp(env, "v3")) {
      version = 3;
    } else if (!strcmp(env, "v4")) {
      version = 4;
    } else if (!strcmp(env, "v5")) {
      version = 5;
    } else if (!strcmp(env, "v6")) {
      version = 6;
    } else if (!strcmp(env, "v7")) {
      version = 7;
    } else if (!strcmp(env, "v8")) {
      version = 8;
    } else if (!strcmp(env, "v9")) {
      version = 9;
    }
  }
  if (auto* env = std::getenv("MOODIST_COPY_BLOCK_SIZE")) {
    int bs = atoi(env);
    if (bs >= 32 && bs <= 1024 && (bs % 32) == 0) {
      blockSize = bs;
    }
  }
  if (auto* env = std::getenv("MOODIST_COPY_GRID_SIZE")) {
    int gs = atoi(env);
    if (gs >= 1 && gs <= 128) {
      gridSize = gs;
    }
  }
}

CompileOpKernels::~CompileOpKernels() {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
  if (cuMulticastModule) {
    cuModuleUnload(cuMulticastModule);
  }
}

// ---------------------------------------------------------------------------
// CompiledModule
// ---------------------------------------------------------------------------

CompiledModule::~CompiledModule() {
  if (module) {
    cuModuleUnload(module);
  }
}

CUfunction CompiledModule::getFunction(const char* name) const {
  CUfunction fn = nullptr;
  CHECK_CU(cuModuleGetFunction(&fn, module, name));
  return fn;
}

CompiledKernel compileKernelPtx(const std::string& ptx, const char* functionName) {
  char jitErrorLog[4096] = {};
  char jitInfoLog[4096] = {};
  CUjit_option jitOptions[] = {
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
  };
  void* jitValues[] = {
      jitErrorLog,
      (void*)(uintptr_t)sizeof(jitErrorLog),
      jitInfoLog,
      (void*)(uintptr_t)sizeof(jitInfoLog),
  };

  CompiledKernel result;
  CUresult jitErr = cuModuleLoadDataEx(&result.module.module, ptx.c_str(), 4, jitOptions, jitValues);
  if (jitErr != CUDA_SUCCESS) {
    log.error("PTX JIT compilation failed (error %d)\n", (int)jitErr);
    if (jitErrorLog[0]) {
      log.error("ptxas error log:\n%s\n", jitErrorLog);
    }
    if (jitInfoLog[0]) {
      log.error("ptxas info log:\n%s\n", jitInfoLog);
    }
    CHECK_CU(jitErr);
  }
  CHECK_CU(cuModuleGetFunction(&result.function, result.module.module, functionName));

  int numRegs = 0;
  CHECK_CU(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, result.function));
  int maxThreads = 0;
  CHECK_CU(cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, result.function));
  log.info("compiled PTX '%s': %d regs, max %d threads/block\n", functionName, numRegs, maxThreads);

  return result;
}

// void launchCopyKernel(CUfunction kernel, size_t gridSize, size_t blockSize, const CopyDescriptor* descriptors,
//     uint32_t numDescriptors, uint32_t stepValue, uint32_t concurrencyIndex, CUstream stream, size_t dynamicSmemBytes)
//     {
//   CompileOpCopyParameters params;
//   params.stepValue = stepValue;
//   params.concurrencyIndex = concurrencyIndex;
//   params.numDescriptors = numDescriptors;
//   CHECK(numDescriptors <= kMaxCopyDescriptors);
//   for (uint32_t i = 0; i < numDescriptors; i++) {
//     params.descriptors[i] = descriptors[i];
//   }
//   std::array<void*, 1> kparams = {&params};
//   CHECK_CU(cuLaunchKernel(kernel, gridSize, 1, 1, blockSize, 1, 1, dynamicSmemBytes, stream, kparams.data(),
//   nullptr));
// }

} // namespace moodist
