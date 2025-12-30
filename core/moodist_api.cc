// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation of moodistGetApi() - returns CoreApi, receives WrapperApi

#include "api/moodist_api.h"
#include "api/allocator_api.h"
#include "api/cpu_allocator.h"
#include "api/processgroup_api.h"
#include "api/serialize_api.h"
#include "api/store_api.h"

namespace moodist {

// Forward declarations for C-style API wrappers defined in allocator.cc
void cudaAllocatorImplAllocateApi(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex,
    uintptr_t* outPtr, void** outCleanupCtx, int* outDeviceIndex);
void allocatorMappedRegionApi(uintptr_t address, uintptr_t* outBase, size_t* outSize);

// Forward declarations for Queue API functions defined in processgroup.cc
api::ApiHandle<api::Queue> processGroupImplMakeQueue(
    ProcessGroupImpl* impl, int location, bool streaming, const char* name);
api::ApiHandle<api::Queue> processGroupImplMakeQueueMulti(
    ProcessGroupImpl* impl, const int* locations, size_t numLocations, bool streaming, const char* name);
void queueDestroy(api::Queue* queue);
bool queueGet(api::Queue* queue, bool block, const float* timeout, TensorPtr* outTensor, size_t* outSize);
void* queuePut(api::Queue* queue, const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy);
size_t queueQsize(api::Queue* queue);
bool queueWait(api::Queue* queue, const float* timeout);
uint32_t queueTransactionBegin(api::Queue* queue);
void queueTransactionCancel(api::Queue* queue, uint32_t id);
void queueTransactionCommit(api::Queue* queue, uint32_t id);
const char* queueName(api::Queue* queue);
void queueWorkDestroy(void* work);
void queueWorkWait(void* work);

// Forward declarations for FutureImpl, CustomOpImpl, compileOpFull defined in processgroup.cc
void futureImplDestroy(void* future);
void futureImplWait(void* future, CUstream stream);
bool futureImplGetResult(void* future, TensorPtr* outTensor);
void customOpImplDestroy(void* op);
void* customOpImplCall(
    void* op, TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream);
void* compileOpFull(ProcessGroupImpl* impl, const int* shape, size_t ndim, DType dtype, const int* inputRanks,
    const int* inputOffsets, const int* inputShapes, size_t nInputs, const int* outputRanks, const int* outputOffsets,
    const int* outputShapes, size_t nOutputs);

// Forward declaration for bufferDestroy defined in serialize_object.cc
void bufferDestroy(api::Buffer* buf);

// Global WrapperApi - copied from _C.so during initialization
// libmoodist.so code accesses wrapper functions through this
WrapperApi wrapperApi = {};

// Static CoreApi - populated with libmoodist.so functions
static CoreApi coreApi = {
    // Magic for build verification
    .magic = kMoodistBuildMagic,

    // Store functions
    .createStore = createStore,
    .storeAddRef = storeAddRef,
    .storeDecRef = storeDecRef,
    .storeSet = storeSet,
    .storeGet = storeGet,
    .storeCheck = storeCheck,
    .storeWait = storeWait,

    // Serialize functions
    .serializeObjectImpl = serializeObjectImpl,
    .serializeBufferPtr = serializeBufferPtr,
    .serializeBufferSize = serializeBufferSize,
    .bufferDestroy = bufferDestroy,
    .deserializeObjectImpl = deserializeObjectImpl,

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

    // ProcessGroupImpl functions
    .createProcessGroupImpl = createProcessGroupImpl,
    .processGroupImplAddRef = processGroupImplAddRef,
    .processGroupImplDecRef = processGroupImplDecRef,
    .processGroupImplRank = processGroupImplRank,
    .processGroupImplSize = processGroupImplSize,
    .processGroupImplAllGather = processGroupImplAllGather,
    .processGroupImplReduceScatter = processGroupImplReduceScatter,
    .processGroupImplAllreduce = processGroupImplAllreduce,
    .processGroupImplBroadcast = processGroupImplBroadcast,
    .processGroupImplReduce = processGroupImplReduce,
    .processGroupImplBarrier = processGroupImplBarrier,
    .processGroupImplScatter = processGroupImplScatter,
    .processGroupImplGather = processGroupImplGather,
    .processGroupImplAllToAll = processGroupImplAllToAll,
    .processGroupImplCudaBarrier = processGroupImplCudaBarrier,
    .processGroupImplShutdown = processGroupImplShutdown,

    // Queue factory functions
    .processGroupImplMakeQueue = processGroupImplMakeQueue,
    .processGroupImplMakeQueueMulti = processGroupImplMakeQueueMulti,

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

    // FutureImpl operations
    .futureImplDestroy = futureImplDestroy,
    .futureImplWait = futureImplWait,
    .futureImplGetResult = futureImplGetResult,

    // CustomOpImpl operations
    .customOpImplDestroy = customOpImplDestroy,
    .customOpImplCall = customOpImplCall,

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
