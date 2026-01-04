// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation of moodistGetApi() - returns CoreApi, receives WrapperApi

#include "api/moodist_api.h"
#include "api/allocator_api.h"
#include "api/cpu_allocator.h"
#include "api/processgroup_api.h"
#include "api/store_api.h"

namespace moodist {

// Forward declarations for buffer API functions defined in buffer_api.cc
api::Buffer* bufferAllocate(size_t nbytes);
void* serializeBufferPtr(api::Buffer* buf);
size_t serializeBufferSize(api::Buffer* buf);
void bufferSetSize(api::Buffer* buf, size_t size);
void bufferDestroy(api::Buffer* buf);

// Forward declarations for internal allocator functions defined in internal_allocator.cc
void* internalAlloc(size_t bytes);
void internalFree(void* ptr);
size_t internalAllocSize(void* ptr);

// Forward declarations for C-style API wrappers defined in allocator.cc
void cudaAllocatorImplAllocateApi(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex,
    uintptr_t* outPtr, void** outCleanupCtx, int* outDeviceIndex);
void allocatorMappedRegionApi(uintptr_t address, uintptr_t* outBase, size_t* outSize);

// Forward declarations for Queue API functions defined in processgroup.cc
api::QueueHandle processGroupMakeQueue(api::ProcessGroup* pg, int location, bool streaming, const char* name);
api::QueueHandle processGroupMakeQueueMulti(
    api::ProcessGroup* pg, const int* locations, size_t numLocations, bool streaming, const char* name);
void queueDestroy(api::Queue* queue);
bool queueGet(api::Queue* queue, bool block, const float* timeout, TensorPtr* outTensor, size_t* outSize);
api::QueueWorkHandle queuePut(api::Queue* queue, const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy);
size_t queueQsize(api::Queue* queue);
bool queueWait(api::Queue* queue, const float* timeout);
uint32_t queueTransactionBegin(api::Queue* queue);
void queueTransactionCancel(api::Queue* queue, uint32_t id);
void queueTransactionCommit(api::Queue* queue, uint32_t id);
const char* queueName(api::Queue* queue);
void queueWorkDestroy(api::QueueWork* work);
void queueWorkWait(api::QueueWork* work);

// Forward declarations for Future, CustomOp, compileOpFull defined in processgroup.cc
void futureDestroy(api::Future* future);
void futureWait(api::Future* future, CUstream stream);
bool futureGetResult(api::Future* future, TensorPtr* outTensor);
void customOpDestroy(api::CustomOp* op);
api::FutureHandle customOpCall(
    api::CustomOp* op, TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream);
api::CustomOpHandle compileOpFull(api::ProcessGroup* pg, const int* shape, size_t ndim, DType dtype,
    const int* inputRanks, const int* inputOffsets, const int* inputShapes, size_t nInputs, const int* outputRanks,
    const int* outputOffsets, const int* outputShapes, size_t nOutputs);

// Serialization is temporarily disabled - will be moved to separate library
// Forward declaration for bufferDestroy defined in serialize_object.cc
// void bufferDestroy(api::Buffer* buf);

// Global WrapperApi - copied from _C.so during initialization
// libmoodist.so code accesses wrapper functions through this
WrapperApi wrapperApi = {};

// Static CoreApi - populated with libmoodist.so functions
static CoreApi coreApi = {
    // Magic for build verification
    .magic = kMoodistBuildMagic,

    // Store functions - refcounted via ApiHandle
    .createStore = createStore,
    .storeDestroy = storeDestroy,
    .storeClose = storeClose,
    .storeSet = storeSet,
    .storeGet = storeGet,
    .storeCheck = storeCheck,
    .storeWait = storeWait,

    // Internal allocator functions
    .internalAlloc = internalAlloc,
    .internalFree = internalFree,
    .internalAllocSize = internalAllocSize,

    // Buffer functions - implemented in core (buffer_api.cc)
    // Serialize implementation functions are nullptr - will be set by loading serialize library
    .bufferAllocate = bufferAllocate,
    .serializeBufferPtr = serializeBufferPtr,
    .serializeBufferSize = serializeBufferSize,
    .bufferSetSize = bufferSetSize,
    .bufferDestroy = bufferDestroy,
    .serializeObjectImpl = nullptr,   // Set when serialize library is loaded
    .deserializeObjectImpl = nullptr, // Set when serialize library is loaded

    // CPU allocator functions
    .cpuAllocatorAlloc = cpuAllocatorAlloc,
    .cpuAllocatorFree = cpuAllocatorFree,

    // CUDA allocator functions
    .createCudaAllocatorImpl = createCudaAllocatorImpl,
    .setCudaAllocatorImpl = setCudaAllocatorImpl,
    .cudaAllocatorImplAllocate = cudaAllocatorImplAllocateApi,
    .cudaAllocatorImplFree = cudaAllocatorImplFree,
    .cudaAllocatorImplRecordStream = cudaAllocatorImplRecordStream,
    .cudaAllocatorImplGetDevice = cudaAllocatorImplGetDevice,
    .allocatorOwns = allocatorOwns,
    .allocatorMappedRegion = allocatorMappedRegionApi,
    .allocatorAddFreeCallback = allocatorAddFreeCallback,
    .allocatorRemoveFreeCallback = allocatorRemoveFreeCallback,

    // ProcessGroup functions - refcounted via ApiHandle
    .createProcessGroup = createProcessGroup,
    .processGroupDestroy = processGroupDestroy,
    .processGroupShutdown = processGroupShutdown,
    .processGroupRank = processGroupRank,
    .processGroupSize = processGroupSize,
    .processGroupGetPreferKernelLess = processGroupGetPreferKernelLess,
    .processGroupSetPreferKernelLess = processGroupSetPreferKernelLess,
    .processGroupGetOption = processGroupGetOption,
    .processGroupSetOption = processGroupSetOption,
    .processGroupAllGather = processGroupAllGather,
    .processGroupReduceScatter = processGroupReduceScatter,
    .processGroupAllreduce = processGroupAllreduce,
    .processGroupBroadcast = processGroupBroadcast,
    .processGroupReduce = processGroupReduce,
    .processGroupBarrier = processGroupBarrier,
    .processGroupScatter = processGroupScatter,
    .processGroupGather = processGroupGather,
    .processGroupAllToAll = processGroupAllToAll,
    .processGroupCudaBarrier = processGroupCudaBarrier,

    // Queue factory functions
    .processGroupMakeQueue = processGroupMakeQueue,
    .processGroupMakeQueueMulti = processGroupMakeQueueMulti,

    // Queue operations
    .queueDestroy = queueDestroy,
    .queueGet = queueGet,
    .queuePut = queuePut,
    .queueQsize = queueQsize,
    .queueWait = queueWait,
    .queueTransactionBegin = queueTransactionBegin,
    .queueTransactionCancel = queueTransactionCancel,
    .queueTransactionCommit = queueTransactionCommit,
    .queueName = queueName,

    // QueueWork operations
    .queueWorkDestroy = queueWorkDestroy,
    .queueWorkWait = queueWorkWait,

    // Future operations
    .futureDestroy = futureDestroy,
    .futureWait = futureWait,
    .futureGetResult = futureGetResult,

    // CustomOp operations
    .customOpDestroy = customOpDestroy,
    .customOpCall = customOpCall,

    // compileOpFull
    .compileOpFull = compileOpFull,

    // Profiling
    .setProfilingEnabled = setProfilingEnabled,
    .getProfilingEnabled = getProfilingEnabled,
};

} // namespace moodist

extern "C" {
__attribute__((visibility("default"))) moodist::CoreApi* moodistGetApi(
    uint32_t expectedVersion, const moodist::WrapperApi* wrapper) {
  if (expectedVersion != moodist::kMoodistApiVersion) {
    return nullptr;
  }
  // Copy WrapperApi to our global so libmoodist.so code can use it
  moodist::wrapperApi = *wrapper;
  return &moodist::coreApi;
}
}
