// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "compile_op_kernel.h"
#include "common.h"
#include "commondefs.h"
#include "cuda_loader.h"
#include "group.h"
#include "ptx_codegen.h"

namespace moodist {

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
    log.error(" === PTX ===\n");
    log.error("%s\n", ptx);
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
  if (ptx.find(".entry compile_op_params(") != std::string::npos) {
    CHECK_CU(cuModuleGetFunction(&result.paramsFunction, result.module.module, "compile_op_params"));
  }

  int numRegs = 0;
  CHECK_CU(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, result.function));
  int maxThreads = 0;
  CHECK_CU(cuFuncGetAttribute(&maxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, result.function));
  int localMemory = 0;
  CHECK_CU(cuFuncGetAttribute(&localMemory, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, result.function));
  int sharedMemory = 0;
  CHECK_CU(cuFuncGetAttribute(&sharedMemory, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, result.function));
  log.info("compiled PTX '%s': %d regs, %d shared bytes, %d local bytes, max %d threads/block\n", functionName, numRegs,
      sharedMemory, localMemory, maxThreads);

  return result;
}

} // namespace moodist
