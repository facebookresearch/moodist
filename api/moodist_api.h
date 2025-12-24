// Copyright (c) Meta Platforms, Inc. and affiliates.

// Api structs for moodist - bidirectional interface between libmoodist.so and _C.so
// - WrapperApi: functions in _C.so that libmoodist.so needs to call (tensor/cuda ops)
// - CoreApi: functions in libmoodist.so that _C.so needs to call (store/serialize)

#pragma once

#include "moodist_version.h"

#include <Python.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <span>
#include <string_view>
#include <vector>

namespace moodist {

// Forward declarations (opaque types)
struct StoreHandle;
struct Buffer;
struct Tensor;            // Opaque wrapper for torch::Tensor
class TensorPtr;          // RAII smart pointer for Tensor* (defined in tensor_ptr.h)
struct CudaAllocatorImpl; // CUDA allocator implementation
struct ProcessGroupImpl;  // Process group implementation

// DType enum - matches torch::ScalarType values
enum class DType : int8_t {
  UInt8 = 0,
  Int8 = 1,
  Int16 = 2,
  Int32 = 3,
  Int64 = 4,
  Float16 = 5,
  Float32 = 6,
  Float64 = 7,
  // Complex types omitted for now
  Bool = 11,
  BFloat16 = 15,
};

// ReduceOp enum - matches c10d::ReduceOp::RedOpType values
enum class ReduceOp : uint8_t {
  SUM = 0,
  AVG = 1,
  PRODUCT = 2,
  MIN = 3,
  MAX = 4,
  BAND = 5,
  BOR = 6,
  BXOR = 7,
  PREMUL_SUM = 8,
};

// =============================================================================
// WrapperApi - functions implemented in _C.so, called by libmoodist.so
// These wrap PyTorch APIs that core needs to call
// =============================================================================
struct WrapperApi {
  // CUDA device/stream functions
  int (*cudaCurrentDevice)();
  CUstream (*cudaGetCurrentStream)();
  void* (*cudaStreamGuardEnter)(CUstream stream, int device);
  void (*cudaStreamGuardExit)(void* guard);

  // CUDA memory allocation via PyTorch's caching allocator
  // Returns pointer; sets *cleanupCtx for use with cudaCachingAllocatorFree
  void* (*cudaCachingAllocatorAlloc)(size_t bytes, void** cleanupCtx);
  void (*cudaCachingAllocatorFree)(void* cleanupCtx);

  // Tensor creation - returns opaque Tensor handle
  Tensor* (*tensorFromBlob)(void* data, const int64_t* sizes, int ndim, DType dtype, int device);
  Tensor* (*tensorEmpty)(const int64_t* sizes, int ndim, DType dtype, int device);

  // Tensor ref counting
  void (*tensorAddRef)(Tensor* t);
  void (*tensorDecRef)(Tensor* t);

  // Tensor accessors
  void* (*tensorDataPtr)(Tensor* t);
  int64_t (*tensorNumel)(Tensor* t);
  int64_t (*tensorNdim)(Tensor* t);
  int64_t (*tensorSize)(Tensor* t, int64_t dim);
  int64_t (*tensorStride)(Tensor* t, int64_t dim);
  DType (*tensorDtype)(Tensor* t);
  int (*tensorDevice)(Tensor* t);
  bool (*tensorIsContiguous)(Tensor* t);
  bool (*tensorIsCuda)(Tensor* t);
  bool (*tensorIsCpu)(Tensor* t);
  int64_t (*tensorItemsize)(Tensor* t);

  // Tensor view operations - return new Tensor handle (caller must decref)
  Tensor* (*tensorView)(Tensor* t, const int64_t* sizes, int ndim);
  Tensor* (*tensorNarrow)(Tensor* t, int dim, int64_t start, int64_t length);
  Tensor* (*tensorContiguous)(Tensor* t);

  // Tensor arithmetic - return new Tensor handle (caller must decref)
  Tensor* (*tensorMulScalar)(Tensor* t, double scalar);

  // Tensor in-place operations
  void (*tensorMulScalar_)(Tensor* t, double scalar);
  void (*tensorRecordStream)(Tensor* t, CUstream stream);
  void (*tensorCopy_)(Tensor* dst, Tensor* src);

  // Tensor reduction operations
  void (*tensorSumOut)(Tensor* dst, Tensor* src, int dim);
  void (*tensorAmaxOut)(Tensor* dst, Tensor* src, int dim);
  void (*tensorAminOut)(Tensor* dst, Tensor* src, int dim);

  // Dtype utilities
  size_t (*dtypeSize)(DType dtype);

  // Free memory callback registration
  void (*registerFreeMemoryCallback)();

  // c10d::Store operations - for ProcessGroup to call back into PyTorch's store
  // store is opaque pointer to c10d::Store passed from ProcessGroup constructor
  void (*c10dStoreSet)(void* store, std::string_view key, const std::vector<uint8_t>& value);
  std::vector<uint8_t> (*c10dStoreGet)(void* store, std::string_view key);
  void (*c10dStoreWait)(void* store, std::span<const std::string> keys);
};

// =============================================================================
// CoreApi - functions implemented in libmoodist.so, called by _C.so
// =============================================================================
struct CoreApi {
  // Magic value for runtime verification (changes per build)
  uint64_t magic;

  // Store functions
  StoreHandle* (*createStore)(std::string_view hostname, int port, std::string_view key, int worldSize, int rank);
  void (*storeAddRef)(StoreHandle* handle);
  void (*storeDecRef)(StoreHandle* handle);
  void (*storeSet)(StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::string_view key,
      const std::vector<uint8_t>& value);
  std::vector<uint8_t> (*storeGet)(
      StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::string_view key);
  bool (*storeCheck)(
      StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);
  void (*storeWait)(
      StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

  // Serialize functions
  Buffer* (*serializeObjectImpl)(PyObject* o);
  void* (*serializeBufferPtr)(Buffer* buf);
  size_t (*serializeBufferSize)(Buffer* buf);
  void (*serializeBufferAddRef)(Buffer* buf);
  void (*serializeBufferDecRef)(Buffer* buf);
  PyObject* (*deserializeObjectImpl)(const void* ptr, size_t len);

  // CPU allocator functions
  // cpuAllocatorAlloc: allocates bytes, returns ptr and sets *cleanupCtx for deleter
  // cpuAllocatorFree: deleter function pointer (void(*)(void*)) to pass to DataPtr
  void* (*cpuAllocatorAlloc)(size_t bytes, void** cleanupCtx);
  void (*cpuAllocatorFree)(void* cleanupCtx);

  // CUDA allocator functions
  CudaAllocatorImpl* (*createCudaAllocatorImpl)();
  void (*setCudaAllocatorImpl)(CudaAllocatorImpl* impl);
  void (*cudaAllocatorImplAllocate)(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex,
      uintptr_t* outPtr, void** outCleanupCtx, int* outDeviceIndex);
  void (*cudaAllocatorImplFree)(void* cleanupCtx, CUstream stream);
  void (*cudaAllocatorImplRecordStream)(CudaAllocatorImpl* impl, uintptr_t ptr, CUstream stream);
  int (*cudaAllocatorImplGetDevice)(CudaAllocatorImpl* impl);
  bool (*allocatorOwns)(uintptr_t address);
  void (*allocatorMappedRegion)(uintptr_t address, uintptr_t* outBase, size_t* outSize);
  void* (*allocatorAddFreeCallback)(CudaAllocatorImpl* impl, uintptr_t baseAddress, void* callbackFn);
  void (*allocatorRemoveFreeCallback)(CudaAllocatorImpl* impl, uintptr_t baseAddress, void* handle);

  // ProcessGroupImpl functions
  ProcessGroupImpl* (*createProcessGroupImpl)(void* c10dStore, int rank, int size);
  void (*processGroupImplAddRef)(ProcessGroupImpl* impl);
  void (*processGroupImplDecRef)(ProcessGroupImpl* impl);
  int (*processGroupImplRank)(ProcessGroupImpl* impl);
  int (*processGroupImplSize)(ProcessGroupImpl* impl);

  // Collective operations - core handles all logic
  void (*processGroupImplAllGather)(ProcessGroupImpl* impl, TensorPtr& output, const TensorPtr& input, CUstream stream);
  void (*processGroupImplReduceScatter)(ProcessGroupImpl* impl, TensorPtr& output, const TensorPtr& input,
      ReduceOp reduceOp, CUstream stream, float premulValue);
  void (*processGroupImplAllreduce)(
      ProcessGroupImpl* impl, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue);
  void (*processGroupImplBroadcast)(ProcessGroupImpl* impl, TensorPtr& tensor, int sourceRank, CUstream stream);
  void (*processGroupImplReduce)(
      ProcessGroupImpl* impl, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream);
  void (*processGroupImplBarrier)(ProcessGroupImpl* impl);
  void (*processGroupImplScatter)(
      ProcessGroupImpl* impl, std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream);
  void (*processGroupImplGather)(
      ProcessGroupImpl* impl, std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream);
  void (*processGroupImplAllToAll)(
      ProcessGroupImpl* impl, std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream);
  void (*processGroupImplCudaBarrier)(ProcessGroupImpl* impl, CUstream stream);
  void (*processGroupImplShutdown)(ProcessGroupImpl* impl);
};

} // namespace moodist

// C linkage for dlsym
extern "C" {
// Returns CoreApi struct if version matches, nullptr otherwise
// Caller passes WrapperApi which libmoodist copies to its global
// Caller should verify returned api->magic == kMoodistBuildMagic
__attribute__((visibility("default"))) moodist::CoreApi* moodistGetApi(
    uint32_t expectedVersion, const moodist::WrapperApi* wrapperApi);
}
