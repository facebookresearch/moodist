// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation of moodistGetAPI() - returns CoreAPI, receives WrapperAPI

#include "moodist_api.h"
#include "serialize_api.h"
#include "store_api.h"

namespace moodist {

// Global WrapperAPI - copied from _C.so during initialization
// libmoodist.so code accesses wrapper functions through this
WrapperAPI wrapperAPI = {};

// Static CoreAPI - populated with libmoodist.so functions
static CoreAPI coreAPI = {
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
};

} // namespace moodist

extern "C" {
__attribute__((visibility("default"))) moodist::CoreAPI* moodistGetAPI(
    uint32_t expectedVersion, const moodist::WrapperAPI* wrapper) {
  if (expectedVersion != moodist::kMoodistAPIVersion) {
    return nullptr;
  }
  // Copy WrapperAPI to our global so libmoodist.so code can use it
  moodist::wrapperAPI = *wrapper;
  return &moodist::coreAPI;
}
}
