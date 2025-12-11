// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation of moodistGetApi() - returns CoreApi, receives WrapperApi

#include "moodist_api.h"
#include "allocator_api.h"
#include "cpu_allocator.h"
#include "serialize_api.h"
#include "store_api.h"

namespace moodist {

// Global WrapperApi - copied from _C.so during initialization
// libmoodist.so code accesses wrapper functions through this
WrapperApi wrapperApi = {};

// Static CoreApi - populated with libmoodist.so functions
static CoreApi coreApi = {
    // Magic for build verification
    .magic = kMoodistBuildMagic,

    // Store functions
    .createStoreImpl = createStoreImpl,
    .storeImplAddRef = storeImplAddRef,
    .storeImplDecRef = storeImplDecRef,
    .storeImplSet = storeImplSet,
    .storeImplGet = storeImplGet,
    .storeImplCheck = storeImplCheck,
    .storeImplWait = storeImplWait,

    // Serialize functions
    .serializeObjectImpl = serializeObjectImpl,
    .serializeBufferPtr = serializeBufferPtr,
    .serializeBufferSize = serializeBufferSize,
    .serializeBufferAddRef = serializeBufferAddRef,
    .serializeBufferDecRef = serializeBufferDecRef,
    .deserializeObjectImpl = deserializeObjectImpl,

    // CPU allocator functions
    .cpuAllocatorAlloc = cpuAllocatorAlloc,
    .cpuAllocatorFree = cpuAllocatorFree,

    // CUDA allocator functions
    .createCudaAllocatorImpl = createCudaAllocatorImpl,
    .setCudaAllocatorImpl = setCudaAllocatorImpl,
    .cudaAllocatorImplAllocate = cudaAllocatorImplAllocate,
    .cudaAllocatorImplDeallocate = cudaAllocatorImplDeallocate,
    .cudaAllocatorImplFree = cudaAllocatorImplFree,
    .allocatorOwns = allocatorOwns,
    .allocatorMappedRegion = allocatorMappedRegion,
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
