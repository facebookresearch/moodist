// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation of moodistGetAPI() - returns struct of function pointers

#include "moodist_api.h"
#include "serialize_api.h"
#include "store_api.h"

namespace moodist {

// Static API struct - populated once, returned by moodistGetAPI
static MoodistAPI api = {
    // Magic for build verification
    .magic = kMoodistBuildMagic,

    // Store functions
    .createStoreImpl = createStoreImpl,
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
__attribute__((visibility("default")))
moodist::MoodistAPI* moodistGetAPI(uint32_t expectedVersion) {
  if (expectedVersion != moodist::kMoodistAPIVersion) {
    return nullptr;
  }
  return &moodist::api;
}
}
