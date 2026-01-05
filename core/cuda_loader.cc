// CUDA dynamic loader - loads NVML, NVRTC, and CUDA driver API via dlsym

#include "cuda_loader.h"
#include "logging.h"

#include <atomic>
#include <dlfcn.h>

namespace moodist::cuda {

CudaApi cudaApi;
NvmlApi nvmlApi;
NvrtcApi nvrtcApi;

namespace {

// Helper: load a symbol, log error and set ok=false if not found
void* loadSym(void* lib, const char* name, const char* libName, bool& ok) {
  void* sym = dlsym(lib, name);
  if (!sym) {
    log.error("Failed to load %s from %s\n", name, libName);
    ok = false;
  }
  return sym;
}

// Helper: load a symbol, return nullptr if not found (for optional symbols)
void* loadSymOptional(void* lib, const char* name) {
  return dlsym(lib, name);
}

// Helper: find library handle - check RTLD_DEFAULT first, then try dlopen
// Prefers already-loaded libraries over loading new ones
void* findLibrary(const char* testSymbol, const char* const* libNames) {
  if (dlsym(RTLD_DEFAULT, testSymbol)) {
    log.verbose("found %s in RTLD_DEFAULT\n", testSymbol);
    return RTLD_DEFAULT;
  }
  log.debug("%s not in RTLD_DEFAULT\n", testSymbol);
  // First pass: check for already-loaded libraries
  for (const char* const* l = libNames; *l; ++l) {
    void* lib = dlopen(*l, RTLD_NOW | RTLD_NOLOAD);
    if (lib) {
      log.verbose("found %s already loaded\n", *l);
      return lib;
    }
  }
  // Second pass: try to load
  for (const char* const* l = libNames; *l; ++l) {
    void* lib = dlopen(*l, RTLD_NOW);
    if (lib) {
      log.verbose("loaded %s\n", *l);
      return lib;
    }
    log.debug("%s not available: %s\n", *l, dlerror());
  }
  log.debug("no library found for %s\n", testSymbol);
  return nullptr;
}

std::atomic<bool> cudaLoaded{false};
std::atomic<bool> nvmlLoaded{false};
std::atomic<bool> nvrtcLoaded{false};

} // namespace

bool loadCuda() {
  if (cudaLoaded.load(std::memory_order_acquire)) {
    return cudaApi.available();
  }

  const char* libs[] = {"libcuda.so.1", "libcuda.so", nullptr};
  void* lib = findLibrary("cuInit", libs);

  if (lib) {
    bool ok = true;
    const char* name = (lib == RTLD_DEFAULT) ? "RTLD_DEFAULT" : "libcuda.so";

    // Error handling
    cudaApi.getErrorString = (cuGetErrorString_t)loadSym(lib, "cuGetErrorString", name, ok);

    // Initialization
    cudaApi.init = (cuInit_t)loadSym(lib, "cuInit", name, ok);
    cudaApi.driverGetVersion = (cuDriverGetVersion_t)loadSym(lib, "cuDriverGetVersion", name, ok);

    // Device management
    cudaApi.deviceGet = (cuDeviceGet_t)loadSym(lib, "cuDeviceGet", name, ok);
    cudaApi.deviceGetCount = (cuDeviceGetCount_t)loadSym(lib, "cuDeviceGetCount", name, ok);
    cudaApi.deviceGetAttribute = (cuDeviceGetAttribute_t)loadSym(lib, "cuDeviceGetAttribute", name, ok);
    cudaApi.deviceGetPCIBusId = (cuDeviceGetPCIBusId_t)loadSym(lib, "cuDeviceGetPCIBusId", name, ok);

    // Context management
    cudaApi.ctxGetCurrent = (cuCtxGetCurrent_t)loadSym(lib, "cuCtxGetCurrent", name, ok);
    cudaApi.ctxSetCurrent = (cuCtxSetCurrent_t)loadSym(lib, "cuCtxSetCurrent", name, ok);
    cudaApi.ctxSynchronize = (cuCtxSynchronize_t)loadSym(lib, "cuCtxSynchronize", name, ok);
    cudaApi.ctxGetDevice = (cuCtxGetDevice_t)loadSym(lib, "cuCtxGetDevice", name, ok);
    cudaApi.devicePrimaryCtxRetain = (cuDevicePrimaryCtxRetain_t)loadSym(lib, "cuDevicePrimaryCtxRetain", name, ok);

    // Module management
    cudaApi.moduleLoadDataEx = (cuModuleLoadDataEx_t)loadSym(lib, "cuModuleLoadDataEx", name, ok);
    cudaApi.moduleUnload = (cuModuleUnload_t)loadSym(lib, "cuModuleUnload", name, ok);
    cudaApi.moduleGetFunction = (cuModuleGetFunction_t)loadSym(lib, "cuModuleGetFunction", name, ok);

    // Function management
    cudaApi.funcGetAttribute = (cuFuncGetAttribute_t)loadSym(lib, "cuFuncGetAttribute", name, ok);

    // Memory management (note: many have _v2 suffix in actual symbol)
    cudaApi.memAlloc = (cuMemAlloc_t)loadSym(lib, "cuMemAlloc_v2", name, ok);
    cudaApi.memFree = (cuMemFree_t)loadSym(lib, "cuMemFree_v2", name, ok);
    cudaApi.memAllocManaged = (cuMemAllocManaged_t)loadSym(lib, "cuMemAllocManaged", name, ok);
    cudaApi.memGetInfo = (cuMemGetInfo_t)loadSym(lib, "cuMemGetInfo_v2", name, ok);
    cudaApi.memGetAddressRange = (cuMemGetAddressRange_t)loadSym(lib, "cuMemGetAddressRange_v2", name, ok);
    cudaApi.memHostAlloc = (cuMemHostAlloc_t)loadSym(lib, "cuMemHostAlloc", name, ok);
    cudaApi.memFreeHost = (cuMemFreeHost_t)loadSym(lib, "cuMemFreeHost", name, ok);
    cudaApi.memHostRegister = (cuMemHostRegister_t)loadSym(lib, "cuMemHostRegister_v2", name, ok);
    cudaApi.memHostUnregister = (cuMemHostUnregister_t)loadSym(lib, "cuMemHostUnregister", name, ok);
    cudaApi.memHostGetDevicePointer =
        (cuMemHostGetDevicePointer_t)loadSym(lib, "cuMemHostGetDevicePointer_v2", name, ok);
    cudaApi.memsetD8 = (cuMemsetD8_t)loadSym(lib, "cuMemsetD8_v2", name, ok);

    // Memory copy
    cudaApi.memcpyAsync = (cuMemcpyAsync_t)loadSym(lib, "cuMemcpyAsync", name, ok);
    cudaApi.memcpyDtoDAsync = (cuMemcpyDtoDAsync_t)loadSym(lib, "cuMemcpyDtoDAsync_v2", name, ok);
    cudaApi.memcpy2DAsync = (cuMemcpy2DAsync_t)loadSym(lib, "cuMemcpy2DAsync_v2", name, ok);

    // Virtual memory (optional - may not exist on older drivers)
    cudaApi.memUnmap = (cuMemUnmap_t)loadSymOptional(lib, "cuMemUnmap");
    cudaApi.memAddressFree = (cuMemAddressFree_t)loadSymOptional(lib, "cuMemAddressFree");
    cudaApi.memRelease = (cuMemRelease_t)loadSymOptional(lib, "cuMemRelease");

    // Pointer attributes
    cudaApi.pointerGetAttribute = (cuPointerGetAttribute_t)loadSym(lib, "cuPointerGetAttribute", name, ok);
    cudaApi.pointerSetAttribute = (cuPointerSetAttribute_t)loadSym(lib, "cuPointerSetAttribute", name, ok);

    // Stream management
    cudaApi.streamCreateWithPriority =
        (cuStreamCreateWithPriority_t)loadSym(lib, "cuStreamCreateWithPriority", name, ok);
    cudaApi.streamDestroy = (cuStreamDestroy_t)loadSym(lib, "cuStreamDestroy_v2", name, ok);
    cudaApi.streamWaitEvent = (cuStreamWaitEvent_t)loadSym(lib, "cuStreamWaitEvent", name, ok);
    cudaApi.streamWaitValue32 = (cuStreamWaitValue32_t)loadSym(lib, "cuStreamWaitValue32_v2", name, ok);
    cudaApi.streamBatchMemOp = (cuStreamBatchMemOp_t)loadSym(lib, "cuStreamBatchMemOp_v2", name, ok);

    // Event management
    cudaApi.eventCreate = (cuEventCreate_t)loadSym(lib, "cuEventCreate", name, ok);
    cudaApi.eventDestroy = (cuEventDestroy_t)loadSym(lib, "cuEventDestroy_v2", name, ok);
    cudaApi.eventRecord = (cuEventRecord_t)loadSym(lib, "cuEventRecord", name, ok);
    cudaApi.eventQuery = (cuEventQuery_t)loadSym(lib, "cuEventQuery", name, ok);
    cudaApi.eventSynchronize = (cuEventSynchronize_t)loadSym(lib, "cuEventSynchronize", name, ok);

    // IPC
    cudaApi.ipcGetMemHandle = (cuIpcGetMemHandle_t)loadSym(lib, "cuIpcGetMemHandle", name, ok);
    cudaApi.ipcOpenMemHandle = (cuIpcOpenMemHandle_t)loadSym(lib, "cuIpcOpenMemHandle_v2", name, ok);
    cudaApi.ipcCloseMemHandle = (cuIpcCloseMemHandle_t)loadSym(lib, "cuIpcCloseMemHandle", name, ok);
    cudaApi.ipcGetEventHandle = (cuIpcGetEventHandle_t)loadSym(lib, "cuIpcGetEventHandle", name, ok);
    cudaApi.ipcOpenEventHandle = (cuIpcOpenEventHandle_t)loadSym(lib, "cuIpcOpenEventHandle", name, ok);

    // Kernel launch
    cudaApi.launchKernel = (cuLaunchKernel_t)loadSym(lib, "cuLaunchKernel", name, ok);
    cudaApi.launchHostFunc = (cuLaunchHostFunc_t)loadSym(lib, "cuLaunchHostFunc", name, ok);

    if (!ok) {
      cudaApi = {};
    }
  }

  cudaLoaded.store(true, std::memory_order_release);
  return cudaApi.available();
}

bool loadNvml() {
  if (nvmlLoaded.load(std::memory_order_acquire)) {
    return nvmlApi.available();
  }

  const char* libs[] = {"libnvidia-ml.so.1", nullptr};
  void* lib = findLibrary("nvmlInit_v2", libs);

  if (lib) {
    bool ok = true;
    const char* name = (lib == RTLD_DEFAULT) ? "RTLD_DEFAULT" : "libnvidia-ml.so";
    nvmlApi.init = (nvmlInit_v2_t)loadSym(lib, "nvmlInit_v2", name, ok);
    nvmlApi.errorString = (nvmlErrorString_t)loadSym(lib, "nvmlErrorString", name, ok);
    nvmlApi.deviceGetCount = (nvmlDeviceGetCount_t)loadSym(lib, "nvmlDeviceGetCount_v2", name, ok);
    nvmlApi.deviceGetHandleByIndex =
        (nvmlDeviceGetHandleByIndex_v2_t)loadSym(lib, "nvmlDeviceGetHandleByIndex_v2", name, ok);
    nvmlApi.deviceGetHandleByPciBusId =
        (nvmlDeviceGetHandleByPciBusId_v2_t)loadSym(lib, "nvmlDeviceGetHandleByPciBusId_v2", name, ok);
    nvmlApi.deviceGetPciInfo = (nvmlDeviceGetPciInfo_v3_t)loadSym(lib, "nvmlDeviceGetPciInfo_v3", name, ok);
    nvmlApi.deviceGetMemoryAffinity =
        (nvmlDeviceGetMemoryAffinity_t)loadSym(lib, "nvmlDeviceGetMemoryAffinity", name, ok);
    nvmlApi.deviceGetP2PStatus = (nvmlDeviceGetP2PStatus_t)loadSym(lib, "nvmlDeviceGetP2PStatus", name, ok);
    if (!ok) {
      nvmlApi = {};
    }
  }

  nvmlLoaded.store(true, std::memory_order_release);
  return nvmlApi.available();
}

bool loadNvrtc() {
  if (nvrtcLoaded.load(std::memory_order_acquire)) {
    return nvrtcApi.available();
  }

  const char* libs[] = {"libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so", nullptr};
  void* lib = findLibrary("nvrtcVersion", libs);

  if (lib) {
    bool ok = true;
    const char* name = (lib == RTLD_DEFAULT) ? "RTLD_DEFAULT" : "libnvrtc.so";
    nvrtcApi.version = (nvrtcVersion_t)loadSym(lib, "nvrtcVersion", name, ok);
    nvrtcApi.getErrorString = (nvrtcGetErrorString_t)loadSym(lib, "nvrtcGetErrorString", name, ok);
    nvrtcApi.createProgram = (nvrtcCreateProgram_t)loadSym(lib, "nvrtcCreateProgram", name, ok);
    nvrtcApi.destroyProgram = (nvrtcDestroyProgram_t)loadSym(lib, "nvrtcDestroyProgram", name, ok);
    nvrtcApi.compileProgram = (nvrtcCompileProgram_t)loadSym(lib, "nvrtcCompileProgram", name, ok);
    nvrtcApi.getProgramLogSize = (nvrtcGetProgramLogSize_t)loadSym(lib, "nvrtcGetProgramLogSize", name, ok);
    nvrtcApi.getProgramLog = (nvrtcGetProgramLog_t)loadSym(lib, "nvrtcGetProgramLog", name, ok);
    nvrtcApi.getPTXSize = (nvrtcGetPTXSize_t)loadSym(lib, "nvrtcGetPTXSize", name, ok);
    nvrtcApi.getPTX = (nvrtcGetPTX_t)loadSym(lib, "nvrtcGetPTX", name, ok);
    nvrtcApi.getCUBINSize = (nvrtcGetCUBINSize_t)loadSym(lib, "nvrtcGetCUBINSize", name, ok);
    nvrtcApi.getCUBIN = (nvrtcGetCUBIN_t)loadSym(lib, "nvrtcGetCUBIN", name, ok);
    if (!ok) {
      nvrtcApi = {};
    }
  }

  nvrtcLoaded.store(true, std::memory_order_release);
  return nvrtcApi.available();
}

} // namespace moodist::cuda
