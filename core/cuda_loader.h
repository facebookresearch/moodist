// CUDA dynamic loader - loads NVML, NVRTC, and CUDA driver API via dlsym

#pragma once

#include <cstddef>
#include <cstdint>

// Forward declare CUstream in global namespace for cuda.h compatibility
struct CUstream_st;
typedef CUstream_st* CUstream;

namespace moodist::cuda {

using CUstream = ::CUstream;

// ============================================================================
// CUDA Driver API types and constants
// ============================================================================

using CUresult = int;
constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
constexpr CUresult CUDA_ERROR_DEINITIALIZED = 4;
constexpr CUresult CUDA_ERROR_NOT_READY = 600;

using CUdeviceptr = unsigned long long;
using CUdevice = int;

struct CUctx_st;
using CUcontext = CUctx_st*;

struct CUmod_st;
using CUmodule = CUmod_st*;

struct CUfunc_st;
using CUfunction = CUfunc_st*;

// CUstream is defined in global namespace and aliased above

struct CUevent_st;
using CUevent = CUevent_st*;

struct CUmemPoolHandle_st;
using CUmemoryPool = CUmemPoolHandle_st*;

struct CUarray_st;
using CUarray = CUarray_st*;

// Integer types
using cuuint32_t = uint32_t;

// Virtual memory management handle
using CUmemGenericAllocationHandle = unsigned long long;

constexpr int CU_IPC_HANDLE_SIZE = 64;

struct CUipcMemHandle {
  char reserved[CU_IPC_HANDLE_SIZE];
};

struct CUipcEventHandle {
  char reserved[CU_IPC_HANDLE_SIZE];
};

// Memory types
enum CUmemorytype {
  CU_MEMORYTYPE_HOST = 0x01,
  CU_MEMORYTYPE_DEVICE = 0x02,
  CU_MEMORYTYPE_ARRAY = 0x03,
  CU_MEMORYTYPE_UNIFIED = 0x04
};

// 2D memory copy parameters
struct CUDA_MEMCPY2D {
  size_t srcXInBytes;
  size_t srcY;
  CUmemorytype srcMemoryType;
  const void* srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  size_t srcPitch;
  size_t dstXInBytes;
  size_t dstY;
  CUmemorytype dstMemoryType;
  void* dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  size_t dstPitch;
  size_t WidthInBytes;
  size_t Height;
};

// Device attributes
using CUdevice_attribute = int;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr CUdevice_attribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

// Pointer attributes
using CUpointer_attribute = int;
constexpr CUpointer_attribute CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6;
constexpr CUpointer_attribute CU_POINTER_ATTRIBUTE_BUFFER_ID = 7;

// Function attributes
using CUfunction_attribute = int;
constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0;
constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3;
constexpr CUfunction_attribute CU_FUNC_ATTRIBUTE_NUM_REGS = 4;

// Event flags
constexpr unsigned int CU_EVENT_DEFAULT = 0x0;
constexpr unsigned int CU_EVENT_BLOCKING_SYNC = 0x1;
constexpr unsigned int CU_EVENT_DISABLE_TIMING = 0x2;
constexpr unsigned int CU_EVENT_INTERPROCESS = 0x4;

// Event wait flags
constexpr unsigned int CU_EVENT_WAIT_DEFAULT = 0x0;

// Stream flags
constexpr unsigned int CU_STREAM_DEFAULT = 0x0;
constexpr unsigned int CU_STREAM_NON_BLOCKING = 0x1;

// Stream wait value flags
constexpr unsigned int CU_STREAM_WAIT_VALUE_GEQ = 0x0;
constexpr unsigned int CU_STREAM_WAIT_VALUE_EQ = 0x1;
constexpr unsigned int CU_STREAM_WAIT_VALUE_AND = 0x2;
constexpr unsigned int CU_STREAM_WAIT_VALUE_NOR = 0x3;
constexpr unsigned int CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30;

// Memory attach flags
constexpr unsigned int CU_MEM_ATTACH_GLOBAL = 0x1;

// IPC memory flags
constexpr unsigned int CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1;

// Memory host register flags
constexpr unsigned int CU_MEMHOSTREGISTER_PORTABLE = 0x01;
constexpr unsigned int CU_MEMHOSTREGISTER_DEVICEMAP = 0x02;

// Memory host alloc flags
constexpr unsigned int CU_MEMHOSTALLOC_DEVICEMAP = 0x02;
constexpr unsigned int CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;

// Stream batch memory op types
using CUstreamBatchMemOpType = int;
constexpr CUstreamBatchMemOpType CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1;
constexpr CUstreamBatchMemOpType CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2;
constexpr CUstreamBatchMemOpType CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4;
constexpr CUstreamBatchMemOpType CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5;

// Stream write value flags
constexpr unsigned int CU_STREAM_WRITE_VALUE_DEFAULT = 0x0;

union CUstreamBatchMemOpParams {
  CUstreamBatchMemOpType operation;
  struct {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union {
      uint32_t value;
      uint64_t value64;
    };
    unsigned int flags;
    CUdeviceptr alias;
  } waitValue;
  struct {
    CUstreamBatchMemOpType operation;
    CUdeviceptr address;
    union {
      uint32_t value;
      uint64_t value64;
    };
    unsigned int flags;
    CUdeviceptr alias;
  } writeValue;
  uint64_t pad[6];
};

// Host function callback type
using CUhostFn = void (*)(void* userData);

// ============================================================================
// CUDA Driver API function pointer types
// ============================================================================

// Error handling
using cuGetErrorString_t = CUresult (*)(CUresult error, const char** pStr);

// Initialization
using cuInit_t = CUresult (*)(unsigned int flags);
using cuDriverGetVersion_t = CUresult (*)(int* driverVersion);

// Device management
using cuDeviceGet_t = CUresult (*)(CUdevice* device, int ordinal);
using cuDeviceGetCount_t = CUresult (*)(int* count);
using cuDeviceGetAttribute_t = CUresult (*)(int* pi, CUdevice_attribute attrib, CUdevice dev);
using cuDeviceGetPCIBusId_t = CUresult (*)(char* pciBusId, int len, CUdevice dev);

// Context management
using cuCtxGetCurrent_t = CUresult (*)(CUcontext* pctx);
using cuCtxSetCurrent_t = CUresult (*)(CUcontext ctx);
using cuCtxSynchronize_t = CUresult (*)();
using cuCtxGetDevice_t = CUresult (*)(CUdevice* device);
using cuDevicePrimaryCtxRetain_t = CUresult (*)(CUcontext* pctx, CUdevice dev);

// Module management
using cuModuleLoadDataEx_t = CUresult (*)(
    CUmodule* module, const void* image, unsigned int numOptions, int* options, void** optionValues);
using cuModuleUnload_t = CUresult (*)(CUmodule hmod);
using cuModuleGetFunction_t = CUresult (*)(CUfunction* hfunc, CUmodule hmod, const char* name);

// Function management
using cuFuncGetAttribute_t = CUresult (*)(int* pi, CUfunction_attribute attrib, CUfunction hfunc);

// Memory management
using cuMemAlloc_t = CUresult (*)(CUdeviceptr* dptr, size_t bytesize);
using cuMemFree_t = CUresult (*)(CUdeviceptr dptr);
using cuMemAllocManaged_t = CUresult (*)(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
using cuMemGetInfo_t = CUresult (*)(size_t* free, size_t* total);
using cuMemGetAddressRange_t = CUresult (*)(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
using cuMemHostAlloc_t = CUresult (*)(void** pp, size_t bytesize, unsigned int flags);
using cuMemFreeHost_t = CUresult (*)(void* p);
using cuMemHostRegister_t = CUresult (*)(void* p, size_t bytesize, unsigned int flags);
using cuMemHostUnregister_t = CUresult (*)(void* p);
using cuMemHostGetDevicePointer_t = CUresult (*)(CUdeviceptr* pdptr, void* p, unsigned int flags);
using cuMemsetD8_t = CUresult (*)(CUdeviceptr dstDevice, unsigned char uc, size_t N);

// Memory copy
using cuMemcpyAsync_t = CUresult (*)(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream);
using cuMemcpyDtoDAsync_t = CUresult (*)(
    CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream);
using cuMemcpy2DAsync_t = CUresult (*)(const CUDA_MEMCPY2D* pCopy, CUstream hStream);

// Virtual memory (optional, may not be present on older drivers)
using cuMemUnmap_t = CUresult (*)(CUdeviceptr ptr, size_t size);
using cuMemAddressFree_t = CUresult (*)(CUdeviceptr ptr, size_t size);
using cuMemRelease_t = CUresult (*)(uint64_t handle);

// Pointer attributes
using cuPointerGetAttribute_t = CUresult (*)(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
using cuPointerSetAttribute_t = CUresult (*)(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr);

// Stream management
using cuStreamCreateWithPriority_t = CUresult (*)(CUstream* phStream, unsigned int flags, int priority);
using cuStreamDestroy_t = CUresult (*)(CUstream hStream);
using cuStreamWaitEvent_t = CUresult (*)(CUstream hStream, CUevent hEvent, unsigned int flags);
using cuStreamWaitValue32_t = CUresult (*)(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
using cuStreamBatchMemOp_t = CUresult (*)(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags);

// Event management
using cuEventCreate_t = CUresult (*)(CUevent* phEvent, unsigned int flags);
using cuEventDestroy_t = CUresult (*)(CUevent hEvent);
using cuEventRecord_t = CUresult (*)(CUevent hEvent, CUstream hStream);
using cuEventQuery_t = CUresult (*)(CUevent hEvent);
using cuEventSynchronize_t = CUresult (*)(CUevent hEvent);

// IPC
using cuIpcGetMemHandle_t = CUresult (*)(CUipcMemHandle* pHandle, CUdeviceptr dptr);
using cuIpcOpenMemHandle_t = CUresult (*)(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags);
using cuIpcCloseMemHandle_t = CUresult (*)(CUdeviceptr dptr);
using cuIpcGetEventHandle_t = CUresult (*)(CUipcEventHandle* pHandle, CUevent event);
using cuIpcOpenEventHandle_t = CUresult (*)(CUevent* phEvent, CUipcEventHandle handle);

// Kernel launch
using cuLaunchKernel_t = CUresult (*)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra);
using cuLaunchHostFunc_t = CUresult (*)(CUstream hStream, CUhostFn fn, void* userData);

// ============================================================================
// CUDA Driver API struct
// ============================================================================

struct CudaApi {
  // Error handling
  cuGetErrorString_t getErrorString = nullptr;

  // Initialization
  cuInit_t init = nullptr;
  cuDriverGetVersion_t driverGetVersion = nullptr;

  // Device management
  cuDeviceGet_t deviceGet = nullptr;
  cuDeviceGetCount_t deviceGetCount = nullptr;
  cuDeviceGetAttribute_t deviceGetAttribute = nullptr;
  cuDeviceGetPCIBusId_t deviceGetPCIBusId = nullptr;

  // Context management
  cuCtxGetCurrent_t ctxGetCurrent = nullptr;
  cuCtxSetCurrent_t ctxSetCurrent = nullptr;
  cuCtxSynchronize_t ctxSynchronize = nullptr;
  cuCtxGetDevice_t ctxGetDevice = nullptr;
  cuDevicePrimaryCtxRetain_t devicePrimaryCtxRetain = nullptr;

  // Module management
  cuModuleLoadDataEx_t moduleLoadDataEx = nullptr;
  cuModuleUnload_t moduleUnload = nullptr;
  cuModuleGetFunction_t moduleGetFunction = nullptr;

  // Function management
  cuFuncGetAttribute_t funcGetAttribute = nullptr;

  // Memory management
  cuMemAlloc_t memAlloc = nullptr;
  cuMemFree_t memFree = nullptr;
  cuMemAllocManaged_t memAllocManaged = nullptr;
  cuMemGetInfo_t memGetInfo = nullptr;
  cuMemGetAddressRange_t memGetAddressRange = nullptr;
  cuMemHostAlloc_t memHostAlloc = nullptr;
  cuMemFreeHost_t memFreeHost = nullptr;
  cuMemHostRegister_t memHostRegister = nullptr;
  cuMemHostUnregister_t memHostUnregister = nullptr;
  cuMemHostGetDevicePointer_t memHostGetDevicePointer = nullptr;
  cuMemsetD8_t memsetD8 = nullptr;

  // Memory copy
  cuMemcpyAsync_t memcpyAsync = nullptr;
  cuMemcpyDtoDAsync_t memcpyDtoDAsync = nullptr;
  cuMemcpy2DAsync_t memcpy2DAsync = nullptr;

  // Virtual memory (optional)
  cuMemUnmap_t memUnmap = nullptr;
  cuMemAddressFree_t memAddressFree = nullptr;
  cuMemRelease_t memRelease = nullptr;

  // Pointer attributes
  cuPointerGetAttribute_t pointerGetAttribute = nullptr;
  cuPointerSetAttribute_t pointerSetAttribute = nullptr;

  // Stream management
  cuStreamCreateWithPriority_t streamCreateWithPriority = nullptr;
  cuStreamDestroy_t streamDestroy = nullptr;
  cuStreamWaitEvent_t streamWaitEvent = nullptr;
  cuStreamWaitValue32_t streamWaitValue32 = nullptr;
  cuStreamBatchMemOp_t streamBatchMemOp = nullptr;

  // Event management
  cuEventCreate_t eventCreate = nullptr;
  cuEventDestroy_t eventDestroy = nullptr;
  cuEventRecord_t eventRecord = nullptr;
  cuEventQuery_t eventQuery = nullptr;
  cuEventSynchronize_t eventSynchronize = nullptr;

  // IPC
  cuIpcGetMemHandle_t ipcGetMemHandle = nullptr;
  cuIpcOpenMemHandle_t ipcOpenMemHandle = nullptr;
  cuIpcCloseMemHandle_t ipcCloseMemHandle = nullptr;
  cuIpcGetEventHandle_t ipcGetEventHandle = nullptr;
  cuIpcOpenEventHandle_t ipcOpenEventHandle = nullptr;

  // Kernel launch
  cuLaunchKernel_t launchKernel = nullptr;
  cuLaunchHostFunc_t launchHostFunc = nullptr;

  bool available() const {
    return init != nullptr;
  }
};

extern CudaApi cudaApi;
bool loadCuda();

// ============================================================================
// CUDA Driver API wrapper functions (match original API names)
// ============================================================================

inline CUresult cuGetErrorString(CUresult error, const char** pStr) {
  return cudaApi.getErrorString(error, pStr);
}
inline CUresult cuInit(unsigned int flags) {
  return cudaApi.init(flags);
}
inline CUresult cuDriverGetVersion(int* driverVersion) {
  return cudaApi.driverGetVersion(driverVersion);
}
inline CUresult cuDeviceGet(CUdevice* device, int ordinal) {
  return cudaApi.deviceGet(device, ordinal);
}
inline CUresult cuDeviceGetCount(int* count) {
  return cudaApi.deviceGetCount(count);
}
inline CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev) {
  return cudaApi.deviceGetAttribute(pi, attrib, dev);
}
inline CUresult cuDeviceGetPCIBusId(char* pciBusId, int len, CUdevice dev) {
  return cudaApi.deviceGetPCIBusId(pciBusId, len, dev);
}
inline CUresult cuCtxGetCurrent(CUcontext* pctx) {
  return cudaApi.ctxGetCurrent(pctx);
}
inline CUresult cuCtxSetCurrent(CUcontext ctx) {
  return cudaApi.ctxSetCurrent(ctx);
}
inline CUresult cuCtxSynchronize() {
  return cudaApi.ctxSynchronize();
}
inline CUresult cuCtxGetDevice(CUdevice* device) {
  return cudaApi.ctxGetDevice(device);
}
inline CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev) {
  return cudaApi.devicePrimaryCtxRetain(pctx, dev);
}
inline CUresult cuModuleLoadDataEx(
    CUmodule* module, const void* image, unsigned int numOptions, int* options, void** optionValues) {
  return cudaApi.moduleLoadDataEx(module, image, numOptions, options, optionValues);
}
inline CUresult cuModuleUnload(CUmodule hmod) {
  return cudaApi.moduleUnload(hmod);
}
inline CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) {
  return cudaApi.moduleGetFunction(hfunc, hmod, name);
}
inline CUresult cuFuncGetAttribute(int* pi, CUfunction_attribute attrib, CUfunction hfunc) {
  return cudaApi.funcGetAttribute(pi, attrib, hfunc);
}
inline CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
  return cudaApi.memAlloc(dptr, bytesize);
}
inline CUresult cuMemFree(CUdeviceptr dptr) {
  return cudaApi.memFree(dptr);
}
inline CUresult cuMemAllocManaged(CUdeviceptr* dptr, size_t bytesize, unsigned int flags) {
  return cudaApi.memAllocManaged(dptr, bytesize, flags);
}
inline CUresult cuMemGetInfo(size_t* free, size_t* total) {
  return cudaApi.memGetInfo(free, total);
}
inline CUresult cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr) {
  return cudaApi.memGetAddressRange(pbase, psize, dptr);
}
inline CUresult cuMemHostAlloc(void** pp, size_t bytesize, unsigned int flags) {
  return cudaApi.memHostAlloc(pp, bytesize, flags);
}
inline CUresult cuMemFreeHost(void* p) {
  return cudaApi.memFreeHost(p);
}
inline CUresult cuMemHostRegister(void* p, size_t bytesize, unsigned int flags) {
  return cudaApi.memHostRegister(p, bytesize, flags);
}
inline CUresult cuMemHostUnregister(void* p) {
  return cudaApi.memHostUnregister(p);
}
inline CUresult cuMemHostGetDevicePointer(CUdeviceptr* pdptr, void* p, unsigned int flags) {
  return cudaApi.memHostGetDevicePointer(pdptr, p, flags);
}
inline CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  return cudaApi.memsetD8(dstDevice, uc, N);
}
inline CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t byteCount, CUstream hStream) {
  return cudaApi.memcpyAsync(dst, src, byteCount, hStream);
}
inline CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t byteCount, CUstream hStream) {
  return cudaApi.memcpyDtoDAsync(dstDevice, srcDevice, byteCount, hStream);
}
inline CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D* pCopy, CUstream hStream) {
  return cudaApi.memcpy2DAsync(pCopy, hStream);
}
inline CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  return cudaApi.memUnmap(ptr, size);
}
inline CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  return cudaApi.memAddressFree(ptr, size);
}
inline CUresult cuMemRelease(uint64_t handle) {
  return cudaApi.memRelease(handle);
}
inline CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attribute, CUdeviceptr ptr) {
  return cudaApi.pointerGetAttribute(data, attribute, ptr);
}
inline CUresult cuPointerSetAttribute(const void* value, CUpointer_attribute attribute, CUdeviceptr ptr) {
  return cudaApi.pointerSetAttribute(value, attribute, ptr);
}
inline CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
  return cudaApi.streamCreateWithPriority(phStream, flags, priority);
}
inline CUresult cuStreamDestroy(CUstream hStream) {
  return cudaApi.streamDestroy(hStream);
}
inline CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
  return cudaApi.streamWaitEvent(hStream, hEvent, flags);
}
inline CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) {
  return cudaApi.streamWaitValue32(stream, addr, value, flags);
}
inline CUresult cuStreamBatchMemOp(
    CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags) {
  return cudaApi.streamBatchMemOp(stream, count, paramArray, flags);
}
inline CUresult cuEventCreate(CUevent* phEvent, unsigned int flags) {
  return cudaApi.eventCreate(phEvent, flags);
}
inline CUresult cuEventDestroy(CUevent hEvent) {
  return cudaApi.eventDestroy(hEvent);
}
inline CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  return cudaApi.eventRecord(hEvent, hStream);
}
inline CUresult cuEventQuery(CUevent hEvent) {
  return cudaApi.eventQuery(hEvent);
}
inline CUresult cuEventSynchronize(CUevent hEvent) {
  return cudaApi.eventSynchronize(hEvent);
}
inline CUresult cuIpcGetMemHandle(CUipcMemHandle* pHandle, CUdeviceptr dptr) {
  return cudaApi.ipcGetMemHandle(pHandle, dptr);
}
inline CUresult cuIpcOpenMemHandle(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int flags) {
  return cudaApi.ipcOpenMemHandle(pdptr, handle, flags);
}
inline CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
  return cudaApi.ipcCloseMemHandle(dptr);
}
inline CUresult cuIpcGetEventHandle(CUipcEventHandle* pHandle, CUevent event) {
  return cudaApi.ipcGetEventHandle(pHandle, event);
}
inline CUresult cuIpcOpenEventHandle(CUevent* phEvent, CUipcEventHandle handle) {
  return cudaApi.ipcOpenEventHandle(phEvent, handle);
}
inline CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void** kernelParams, void** extra) {
  return cudaApi.launchKernel(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}
inline CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void* userData) {
  return cudaApi.launchHostFunc(hStream, fn, userData);
}

// ============================================================================
// NVML types and constants
// ============================================================================

using nvmlReturn_t = int;
constexpr nvmlReturn_t NVML_SUCCESS = 0;

struct nvmlDevice_st;
using nvmlDevice_t = nvmlDevice_st*;

struct nvmlPciInfo_t {
  char busIdLegacy[16];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
  char busId[32];
};

using nvmlGpuP2PStatus_t = int;
constexpr nvmlGpuP2PStatus_t NVML_P2P_STATUS_OK = 0;
constexpr nvmlGpuP2PStatus_t NVML_P2P_STATUS_UNKNOWN = 5;

using nvmlGpuP2PCapsIndex_t = int;
constexpr nvmlGpuP2PCapsIndex_t NVML_P2P_CAPS_INDEX_NVLINK = 2;

using nvmlAffinityScope_t = unsigned int;
constexpr nvmlAffinityScope_t NVML_AFFINITY_SCOPE_NODE = 0;

using nvmlInit_v2_t = nvmlReturn_t (*)();
using nvmlErrorString_t = const char* (*)(nvmlReturn_t);
using nvmlDeviceGetCount_t = nvmlReturn_t (*)(unsigned int*);
using nvmlDeviceGetHandleByIndex_v2_t = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using nvmlDeviceGetHandleByPciBusId_v2_t = nvmlReturn_t (*)(const char*, nvmlDevice_t*);
using nvmlDeviceGetPciInfo_v3_t = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t*);
using nvmlDeviceGetMemoryAffinity_t = nvmlReturn_t (*)(nvmlDevice_t, unsigned int, unsigned long*, nvmlAffinityScope_t);
using nvmlDeviceGetP2PStatus_t = nvmlReturn_t (*)(
    nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t, nvmlGpuP2PStatus_t*);

struct NvmlApi {
  nvmlInit_v2_t init = nullptr;
  nvmlErrorString_t errorString = nullptr;
  nvmlDeviceGetCount_t deviceGetCount = nullptr;
  nvmlDeviceGetHandleByIndex_v2_t deviceGetHandleByIndex = nullptr;
  nvmlDeviceGetHandleByPciBusId_v2_t deviceGetHandleByPciBusId = nullptr;
  nvmlDeviceGetPciInfo_v3_t deviceGetPciInfo = nullptr;
  nvmlDeviceGetMemoryAffinity_t deviceGetMemoryAffinity = nullptr;
  nvmlDeviceGetP2PStatus_t deviceGetP2PStatus = nullptr;

  bool available() const {
    return init != nullptr;
  }
};

extern NvmlApi nvmlApi;
bool loadNvml();

// ============================================================================
// NVRTC types and constants
// ============================================================================

using nvrtcResult = int;
constexpr nvrtcResult NVRTC_SUCCESS = 0;
constexpr nvrtcResult NVRTC_ERROR_INVALID_OPTION = 5;

struct _nvrtcProgram;
using nvrtcProgram = _nvrtcProgram*;

using nvrtcVersion_t = nvrtcResult (*)(int* major, int* minor);
using nvrtcGetErrorString_t = const char* (*)(nvrtcResult result);
using nvrtcCreateProgram_t = nvrtcResult (*)(nvrtcProgram* prog, const char* src, const char* name, int numHeaders,
    const char* const* headers, const char* const* includeNames);
using nvrtcDestroyProgram_t = nvrtcResult (*)(nvrtcProgram* prog);
using nvrtcCompileProgram_t = nvrtcResult (*)(nvrtcProgram prog, int numOptions, const char* const* options);
using nvrtcGetProgramLogSize_t = nvrtcResult (*)(nvrtcProgram prog, size_t* logSizeRet);
using nvrtcGetProgramLog_t = nvrtcResult (*)(nvrtcProgram prog, char* log);
using nvrtcGetPTXSize_t = nvrtcResult (*)(nvrtcProgram prog, size_t* ptxSizeRet);
using nvrtcGetPTX_t = nvrtcResult (*)(nvrtcProgram prog, char* ptx);
using nvrtcGetCUBINSize_t = nvrtcResult (*)(nvrtcProgram prog, size_t* cubinSizeRet);
using nvrtcGetCUBIN_t = nvrtcResult (*)(nvrtcProgram prog, char* cubin);

struct NvrtcApi {
  nvrtcVersion_t version = nullptr;
  nvrtcGetErrorString_t getErrorString = nullptr;
  nvrtcCreateProgram_t createProgram = nullptr;
  nvrtcDestroyProgram_t destroyProgram = nullptr;
  nvrtcCompileProgram_t compileProgram = nullptr;
  nvrtcGetProgramLogSize_t getProgramLogSize = nullptr;
  nvrtcGetProgramLog_t getProgramLog = nullptr;
  nvrtcGetPTXSize_t getPTXSize = nullptr;
  nvrtcGetPTX_t getPTX = nullptr;
  nvrtcGetCUBINSize_t getCUBINSize = nullptr;
  nvrtcGetCUBIN_t getCUBIN = nullptr;

  bool available() const {
    return version != nullptr;
  }
};

extern NvrtcApi nvrtcApi;
bool loadNvrtc();

} // namespace moodist::cuda
