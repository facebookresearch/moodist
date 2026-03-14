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
void* loadSym(void* lib, const char* name, const char* libName, bool& ok, std::string& failed) {
  void* sym = dlsym(lib, name);
  if (!sym) {
    if (!failed.empty()) {
      failed += ", ";
    }
    failed += name;
    ok = false;
  }
  return sym;
}

// Helper: load a symbol, return nullptr if not found (for optional symbols)
void* loadSymOptional(void* lib, const char* name) {
  return dlsym(lib, name);
}

// Helper: find library handle - check RTLD_DEFAULT first, then try dlopen
// Prefers already-loaded libraries over loading new ones.
// Returns true if found, with lib set to the handle (which may be RTLD_DEFAULT, i.e. NULL).
bool findLibrary(const char* testSymbol, const char* const* libNames, void*& lib) {
  if (dlsym(RTLD_DEFAULT, testSymbol)) {
    log.verbose("found %s in RTLD_DEFAULT\n", testSymbol);
    lib = RTLD_DEFAULT;
    return true;
  }
  log.debug("%s not in RTLD_DEFAULT\n", testSymbol);
  // First pass: check for already-loaded libraries
  for (const char* const* l = libNames; *l; ++l) {
    lib = dlopen(*l, RTLD_NOW | RTLD_NOLOAD);
    if (lib) {
      log.verbose("found %s already loaded\n", *l);
      return true;
    }
  }
  // Second pass: try to load
  for (const char* const* l = libNames; *l; ++l) {
    lib = dlopen(*l, RTLD_NOW);
    if (lib) {
      log.verbose("loaded %s\n", *l);
      return true;
    }
    log.debug("%s not available: %s\n", *l, dlerror());
  }
  log.debug("no library found for %s\n", testSymbol);
  return false;
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
  void* lib = nullptr;

  if (findLibrary("cuInit", libs, lib)) {
    bool ok = true;
    std::string failed;
    const char* name = (lib == RTLD_DEFAULT) ? "RTLD_DEFAULT" : "libcuda.so";

    // Error handling
    cudaApi.getErrorString = (cuGetErrorString_t)loadSym(lib, "cuGetErrorString", name, ok, failed);

    // Initialization
    cudaApi.init = (cuInit_t)loadSym(lib, "cuInit", name, ok, failed);
    cudaApi.driverGetVersion = (cuDriverGetVersion_t)loadSym(lib, "cuDriverGetVersion", name, ok, failed);

    // Device management
    cudaApi.deviceGet = (cuDeviceGet_t)loadSym(lib, "cuDeviceGet", name, ok, failed);
    cudaApi.deviceGetCount = (cuDeviceGetCount_t)loadSym(lib, "cuDeviceGetCount", name, ok, failed);
    cudaApi.deviceGetAttribute = (cuDeviceGetAttribute_t)loadSym(lib, "cuDeviceGetAttribute", name, ok, failed);
    cudaApi.deviceGetPCIBusId = (cuDeviceGetPCIBusId_t)loadSym(lib, "cuDeviceGetPCIBusId", name, ok, failed);

    // Context management
    cudaApi.ctxGetCurrent = (cuCtxGetCurrent_t)loadSym(lib, "cuCtxGetCurrent", name, ok, failed);
    cudaApi.ctxSetCurrent = (cuCtxSetCurrent_t)loadSym(lib, "cuCtxSetCurrent", name, ok, failed);
    cudaApi.ctxSynchronize = (cuCtxSynchronize_t)loadSym(lib, "cuCtxSynchronize", name, ok, failed);
    cudaApi.ctxGetDevice = (cuCtxGetDevice_t)loadSym(lib, "cuCtxGetDevice", name, ok, failed);
    cudaApi.devicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)loadSym(lib, "cuDevicePrimaryCtxRetain", name, ok, failed);

    // Module management
    cudaApi.moduleLoadDataEx = (cuModuleLoadDataEx_t)loadSym(lib, "cuModuleLoadDataEx", name, ok, failed);
    cudaApi.moduleUnload = (cuModuleUnload_t)loadSym(lib, "cuModuleUnload", name, ok, failed);
    cudaApi.moduleGetFunction = (cuModuleGetFunction_t)loadSym(lib, "cuModuleGetFunction", name, ok, failed);

    // Function management
    cudaApi.funcGetAttribute = (cuFuncGetAttribute_t)loadSym(lib, "cuFuncGetAttribute", name, ok, failed);
    cudaApi.funcSetAttribute = (cuFuncSetAttribute_t)loadSym(lib, "cuFuncSetAttribute", name, ok, failed);

    // Memory management (note: many have _v2 suffix in actual symbol)
    cudaApi.memAlloc = (cuMemAlloc_t)loadSym(lib, "cuMemAlloc_v2", name, ok, failed);
    cudaApi.memFree = (cuMemFree_t)loadSym(lib, "cuMemFree_v2", name, ok, failed);
    cudaApi.memAllocManaged = (cuMemAllocManaged_t)loadSym(lib, "cuMemAllocManaged", name, ok, failed);
    cudaApi.memGetInfo = (cuMemGetInfo_t)loadSym(lib, "cuMemGetInfo_v2", name, ok, failed);
    cudaApi.memGetAddressRange = (cuMemGetAddressRange_t)loadSym(lib, "cuMemGetAddressRange_v2", name, ok, failed);
    cudaApi.memHostAlloc = (cuMemHostAlloc_t)loadSym(lib, "cuMemHostAlloc", name, ok, failed);
    cudaApi.memFreeHost = (cuMemFreeHost_t)loadSym(lib, "cuMemFreeHost", name, ok, failed);
    cudaApi.memHostRegister = (cuMemHostRegister_t)loadSym(lib, "cuMemHostRegister_v2", name, ok, failed);
    cudaApi.memHostUnregister = (cuMemHostUnregister_t)loadSym(lib, "cuMemHostUnregister", name, ok, failed);
    cudaApi.memHostGetDevicePointer =
        (cuMemHostGetDevicePointer_t)loadSym(lib, "cuMemHostGetDevicePointer_v2", name, ok, failed);
    cudaApi.memsetD8 = (cuMemsetD8_t)loadSym(lib, "cuMemsetD8_v2", name, ok, failed);

    // Memory copy
    cudaApi.memcpyAsync = (cuMemcpyAsync_t)loadSym(lib, "cuMemcpyAsync", name, ok, failed);
    cudaApi.memcpyDtoDAsync = (cuMemcpyDtoDAsync_t)loadSym(lib, "cuMemcpyDtoDAsync_v2", name, ok, failed);
    cudaApi.memcpy2DAsync = (cuMemcpy2DAsync_t)loadSym(lib, "cuMemcpy2DAsync_v2", name, ok, failed);

    // Virtual memory management
    cudaApi.memUnmap = (cuMemUnmap_t)loadSym(lib, "cuMemUnmap", name, ok, failed);
    cudaApi.memAddressFree = (cuMemAddressFree_t)loadSym(lib, "cuMemAddressFree", name, ok, failed);
    cudaApi.memRelease = (cuMemRelease_t)loadSym(lib, "cuMemRelease", name, ok, failed);
    cudaApi.memCreate = (cuMemCreate_t)loadSym(lib, "cuMemCreate", name, ok, failed);
    cudaApi.memAddressReserve = (cuMemAddressReserve_t)loadSym(lib, "cuMemAddressReserve", name, ok, failed);
    cudaApi.memMap = (cuMemMap_t)loadSym(lib, "cuMemMap", name, ok, failed);
    cudaApi.memSetAccess = (cuMemSetAccess_t)loadSym(lib, "cuMemSetAccess", name, ok, failed);
    cudaApi.memGetAllocationGranularity =
        (cuMemGetAllocationGranularity_t)loadSym(lib, "cuMemGetAllocationGranularity", name, ok, failed);
    cudaApi.memExportToShareableHandle =
        (cuMemExportToShareableHandle_t)loadSym(lib, "cuMemExportToShareableHandle", name, ok, failed);
    cudaApi.memImportShareableHandle =
        (cuMemImportShareableHandle_t)loadSym(lib, "cuMemImportFromShareableHandle", name, ok, failed);

    // Pointer attributes
    cudaApi.pointerGetAttribute = (cuPointerGetAttribute_t)loadSym(lib, "cuPointerGetAttribute", name, ok, failed);
    cudaApi.pointerSetAttribute = (cuPointerSetAttribute_t)loadSym(lib, "cuPointerSetAttribute", name, ok, failed);

    // Stream management
    cudaApi.streamCreateWithPriority =
        (cuStreamCreateWithPriority_t)loadSym(lib, "cuStreamCreateWithPriority", name, ok, failed);
    cudaApi.streamDestroy = (cuStreamDestroy_t)loadSym(lib, "cuStreamDestroy_v2", name, ok, failed);
    cudaApi.streamSynchronize = (cuStreamSynchronize_t)loadSym(lib, "cuStreamSynchronize", name, ok, failed);
    cudaApi.streamWaitEvent = (cuStreamWaitEvent_t)loadSym(lib, "cuStreamWaitEvent", name, ok, failed);
    cudaApi.streamWaitValue32 = (cuStreamWaitValue32_t)loadSym(lib, "cuStreamWaitValue32_v2", name, ok, failed);
    cudaApi.streamBatchMemOp = (cuStreamBatchMemOp_t)loadSym(lib, "cuStreamBatchMemOp_v2", name, ok, failed);

    // Event management
    cudaApi.eventCreate = (cuEventCreate_t)loadSym(lib, "cuEventCreate", name, ok, failed);
    cudaApi.eventDestroy = (cuEventDestroy_t)loadSym(lib, "cuEventDestroy_v2", name, ok, failed);
    cudaApi.eventRecord = (cuEventRecord_t)loadSym(lib, "cuEventRecord", name, ok, failed);
    cudaApi.eventQuery = (cuEventQuery_t)loadSym(lib, "cuEventQuery", name, ok, failed);
    cudaApi.eventSynchronize = (cuEventSynchronize_t)loadSym(lib, "cuEventSynchronize", name, ok, failed);
    cudaApi.eventElapsedTime = (cuEventElapsedTime_t)loadSym(lib, "cuEventElapsedTime", name, ok, failed);

    // IPC
    cudaApi.ipcGetMemHandle = (cuIpcGetMemHandle_t)loadSym(lib, "cuIpcGetMemHandle", name, ok, failed);
    cudaApi.ipcOpenMemHandle = (cuIpcOpenMemHandle_t)loadSym(lib, "cuIpcOpenMemHandle_v2", name, ok, failed);
    cudaApi.ipcCloseMemHandle = (cuIpcCloseMemHandle_t)loadSym(lib, "cuIpcCloseMemHandle", name, ok, failed);
    cudaApi.ipcGetEventHandle = (cuIpcGetEventHandle_t)loadSym(lib, "cuIpcGetEventHandle", name, ok, failed);
    cudaApi.ipcOpenEventHandle = (cuIpcOpenEventHandle_t)loadSym(lib, "cuIpcOpenEventHandle", name, ok, failed);

    // Kernel launch
    cudaApi.linkCreate = (cuLinkCreate_t)loadSym(lib, "cuLinkCreate_v2", name, ok, failed);
    cudaApi.linkAddData = (cuLinkAddData_t)loadSym(lib, "cuLinkAddData_v2", name, ok, failed);
    cudaApi.linkComplete = (cuLinkComplete_t)loadSym(lib, "cuLinkComplete", name, ok, failed);
    cudaApi.linkDestroy = (cuLinkDestroy_t)loadSym(lib, "cuLinkDestroy", name, ok, failed);
    cudaApi.launchKernel = (cuLaunchKernel_t)loadSym(lib, "cuLaunchKernel", name, ok, failed);
    cudaApi.launchHostFunc = (cuLaunchHostFunc_t)loadSym(lib, "cuLaunchHostFunc", name, ok, failed);

    // Multicast (Hopper+ with NVSwitch)
    cudaApi.multicastCreate = (cuMulticastCreate_t)loadSym(lib, "cuMulticastCreate", name, ok, failed);
    cudaApi.multicastAddDevice = (cuMulticastAddDevice_t)loadSym(lib, "cuMulticastAddDevice", name, ok, failed);
    cudaApi.multicastBindMem = (cuMulticastBindMem_t)loadSym(lib, "cuMulticastBindMem", name, ok, failed);
    cudaApi.multicastGetGranularity =
        (cuMulticastGetGranularity_t)loadSym(lib, "cuMulticastGetGranularity", name, ok, failed);
    cudaApi.multicastUnbind = (cuMulticastUnbind_t)loadSym(lib, "cuMulticastUnbind", name, ok, failed);

    if (!ok) {
      log.error("Failed to load CUDA symbols from %s: %s\n", name, failed.c_str());
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
  void* lib = nullptr;

  if (findLibrary("nvmlInit_v2", libs, lib)) {
    bool ok = true;
    std::string failed;
    const char* name = (lib == RTLD_DEFAULT) ? "RTLD_DEFAULT" : "libnvidia-ml.so";
    nvmlApi.init = (nvmlInit_v2_t)loadSym(lib, "nvmlInit_v2", name, ok, failed);
    nvmlApi.errorString = (nvmlErrorString_t)loadSym(lib, "nvmlErrorString", name, ok, failed);
    nvmlApi.deviceGetCount = (nvmlDeviceGetCount_t)loadSym(lib, "nvmlDeviceGetCount_v2", name, ok, failed);
    nvmlApi.deviceGetHandleByIndex =
        (nvmlDeviceGetHandleByIndex_v2_t)loadSym(lib, "nvmlDeviceGetHandleByIndex_v2", name, ok, failed);
    nvmlApi.deviceGetHandleByPciBusId =
        (nvmlDeviceGetHandleByPciBusId_v2_t)loadSym(lib, "nvmlDeviceGetHandleByPciBusId_v2", name, ok, failed);
    nvmlApi.deviceGetPciInfo = (nvmlDeviceGetPciInfo_v3_t)loadSym(lib, "nvmlDeviceGetPciInfo_v3", name, ok, failed);
    nvmlApi.deviceGetMemoryAffinity =
        (nvmlDeviceGetMemoryAffinity_t)loadSym(lib, "nvmlDeviceGetMemoryAffinity", name, ok, failed);
    nvmlApi.deviceGetP2PStatus = (nvmlDeviceGetP2PStatus_t)loadSym(lib, "nvmlDeviceGetP2PStatus", name, ok, failed);
    if (!ok) {
      log.error("Failed to load NVML symbols from %s: %s\n", name, failed.c_str());
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
  void* lib = nullptr;

  if (findLibrary("nvrtcVersion", libs, lib)) {
    bool ok = true;
    std::string failed;
    const char* name = (lib == RTLD_DEFAULT) ? "RTLD_DEFAULT" : "libnvrtc.so";
    nvrtcApi.version = (nvrtcVersion_t)loadSym(lib, "nvrtcVersion", name, ok, failed);
    nvrtcApi.getErrorString = (nvrtcGetErrorString_t)loadSym(lib, "nvrtcGetErrorString", name, ok, failed);
    nvrtcApi.createProgram = (nvrtcCreateProgram_t)loadSym(lib, "nvrtcCreateProgram", name, ok, failed);
    nvrtcApi.destroyProgram = (nvrtcDestroyProgram_t)loadSym(lib, "nvrtcDestroyProgram", name, ok, failed);
    nvrtcApi.compileProgram = (nvrtcCompileProgram_t)loadSym(lib, "nvrtcCompileProgram", name, ok, failed);
    nvrtcApi.getProgramLogSize = (nvrtcGetProgramLogSize_t)loadSym(lib, "nvrtcGetProgramLogSize", name, ok, failed);
    nvrtcApi.getProgramLog = (nvrtcGetProgramLog_t)loadSym(lib, "nvrtcGetProgramLog", name, ok, failed);
    nvrtcApi.getPTXSize = (nvrtcGetPTXSize_t)loadSym(lib, "nvrtcGetPTXSize", name, ok, failed);
    nvrtcApi.getPTX = (nvrtcGetPTX_t)loadSym(lib, "nvrtcGetPTX", name, ok, failed);
    nvrtcApi.getCUBINSize = (nvrtcGetCUBINSize_t)loadSym(lib, "nvrtcGetCUBINSize", name, ok, failed);
    nvrtcApi.getCUBIN = (nvrtcGetCUBIN_t)loadSym(lib, "nvrtcGetCUBIN", name, ok, failed);
    if (!ok) {
      log.error("Failed to load NVRTC symbols from %s: %s\n", name, failed.c_str());
      nvrtcApi = {};
    }
  }

  nvrtcLoaded.store(true, std::memory_order_release);
  return nvrtcApi.available();
}

} // namespace moodist::cuda
