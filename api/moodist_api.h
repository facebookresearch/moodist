// Copyright (c) Meta Platforms, Inc. and affiliates.

// Api structs for moodist - bidirectional interface between libmoodist.so and _C.so
// - WrapperApi: functions in _C.so that libmoodist.so needs to call (tensor/cuda ops)
// - CoreApi: functions in libmoodist.so that _C.so needs to call (store/serialize)

#pragma once

#include "moodist_version.h"
#include "types.h" // For ApiHandle, api::Queue, etc.

// #include <Python.h>  // Removed - serialization uses void* now
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <span>
#include <string_view>
#include <vector>

namespace moodist {

// Forward declarations (opaque types)
struct Tensor;            // Opaque wrapper for torch::Tensor
class TensorPtr;          // RAII smart pointer for Tensor* (defined in tensor_ptr.h)
struct CudaAllocatorImpl; // CUDA allocator implementation

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
  // Like tensorFromBlob but with a deleter called when tensor is destroyed
  Tensor* (*tensorFromBlobWithDeleter)(void* data, const int64_t* sizes, int ndim, DType dtype, int device,
      void (*deleter)(void* ctx), void* deleterCtx);

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

  // Signal checking - throws if interrupted (e.g., Ctrl+C)
  void (*checkSignals)();
};

// =============================================================================
// CoreApi - functions implemented in libmoodist.so, called by _C.so
// =============================================================================
struct CoreApi {
  // Magic value for runtime verification (changes per build)
  uint64_t magic;

  // Store functions - refcounted via ApiHandle
  api::StoreHandle (*createStore)(std::string_view hostname, int port, std::string_view key, int worldSize, int rank);
  void (*storeDestroy)(api::Store* store);
  void (*storeClose)(api::Store* store); // Optional: explicitly trigger shutdown
  void (*storeSet)(api::Store* store, std::chrono::steady_clock::duration timeout, std::string_view key,
      const std::vector<uint8_t>& value);
  std::vector<uint8_t> (*storeGet)(
      api::Store* store, std::chrono::steady_clock::duration timeout, std::string_view key);
  bool (*storeCheck)(
      api::Store* store, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);
  void (*storeWait)(
      api::Store* store, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

  // Serialize functions
  // Buffer inherits from ApiRefCounted - wrapper manages refcount via ApiHandle
  // Wrapper's destroy(api::Buffer*) calls bufferDestroy to delete the object
  // Note: PyObject* is void* here to avoid Python dependency in core
  api::BufferHandle (*serializeObjectImpl)(void* pyObject);
  void* (*serializeBufferPtr)(api::Buffer* buf);
  size_t (*serializeBufferSize)(api::Buffer* buf);
  void (*bufferDestroy)(api::Buffer* buf);
  void* (*deserializeObjectImpl)(const void* ptr, size_t len);

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

  // ProcessGroup functions - refcounted via ApiHandle
  api::ProcessGroupHandle (*createProcessGroup)(void* c10dStore, int rank, int size);
  void (*processGroupDestroy)(api::ProcessGroup* pg);
  void (*processGroupShutdown)(api::ProcessGroup* pg); // Optional: explicitly trigger shutdown
  int (*processGroupRank)(api::ProcessGroup* pg);
  int (*processGroupSize)(api::ProcessGroup* pg);
  bool (*processGroupGetPreferKernelLess)(api::ProcessGroup* pg);
  void (*processGroupSetPreferKernelLess)(api::ProcessGroup* pg, bool value);
  int64_t (*processGroupGetOption)(api::ProcessGroup* pg, const char* name);
  void (*processGroupSetOption)(api::ProcessGroup* pg, const char* name, int64_t value);

  // Collective operations - core handles all logic
  void (*processGroupAllGather)(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, CUstream stream);
  void (*processGroupReduceScatter)(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp,
      CUstream stream, float premulValue);
  void (*processGroupAllreduce)(
      api::ProcessGroup* pg, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue);
  void (*processGroupBroadcast)(api::ProcessGroup* pg, TensorPtr& tensor, int sourceRank, CUstream stream);
  void (*processGroupReduce)(
      api::ProcessGroup* pg, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream);
  void (*processGroupBarrier)(api::ProcessGroup* pg);
  void (*processGroupScatter)(
      api::ProcessGroup* pg, std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream);
  void (*processGroupGather)(
      api::ProcessGroup* pg, std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream);
  void (*processGroupAllToAll)(
      api::ProcessGroup* pg, std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream);
  void (*processGroupCudaBarrier)(api::ProcessGroup* pg, CUstream stream);

  // Queue factory functions - return ApiHandle (ownership via RVO)
  api::QueueHandle (*processGroupMakeQueue)(api::ProcessGroup* pg, int location, bool streaming, const char* name);
  api::QueueHandle (*processGroupMakeQueueMulti)(
      api::ProcessGroup* pg, const int* locations, size_t numLocations, bool streaming, const char* name);

  // Queue operations - operate on api::Queue* pointers
  // Queue inherits from ApiRefCounted - wrapper manages refcount via ApiHandle
  // Wrapper's destroy(api::Queue*) calls queueDestroy to delete the object
  void (*queueDestroy)(api::Queue* queue);
  bool (*queueGet)(api::Queue* queue, bool block, const float* timeout, TensorPtr* outTensor, size_t* outSize);
  api::QueueWorkHandle (*queuePut)(
      api::Queue* queue, const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy);
  size_t (*queueQsize)(api::Queue* queue);
  bool (*queueWait)(api::Queue* queue, const float* timeout);
  uint32_t (*queueTransactionBegin)(api::Queue* queue);
  void (*queueTransactionCancel)(api::Queue* queue, uint32_t id);
  void (*queueTransactionCommit)(api::Queue* queue, uint32_t id);
  const char* (*queueName)(api::Queue* queue);

  // QueueWork operations - unique ownership (not refcounted)
  // ApiHandle<api::QueueWork> uses move-only semantics
  void (*queueWorkDestroy)(api::QueueWork* work);
  void (*queueWorkWait)(api::QueueWork* work);

  // Future operations - refcounted via ApiHandle
  void (*futureDestroy)(api::Future* future);
  void (*futureWait)(api::Future* future, CUstream stream);
  bool (*futureGetResult)(api::Future* future, TensorPtr* outTensor);

  // CustomOp operations - refcounted via ApiHandle
  void (*customOpDestroy)(api::CustomOp* op);
  api::FutureHandle (*customOpCall)(
      api::CustomOp* op, TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream);

  // compileOpFull - compiles a distributed tensor operation
  api::CustomOpHandle (*compileOpFull)(api::ProcessGroup* pg, const int* shape, size_t ndim, DType dtype,
      const int* inputRanks, const int* inputOffsets, const int* inputShapes, size_t nInputs, const int* outputRanks,
      const int* outputOffsets, const int* outputShapes, size_t nOutputs);

  // Profiling
  void (*setProfilingEnabled)(bool enabled);
  bool (*getProfilingEnabled)();
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
