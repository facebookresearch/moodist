// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation of moodistGetAPI() - returns struct of function pointers

#include "moodist_api.h"
#include "serialize_api.h"
#include "store_api.h"

namespace moodist {

// Static API struct - core functions populated here, wrapper functions set to nullptr
// (wrapper fills them in after loading via getMoodistAPI())
static MoodistAPI api = {
    // Magic for build verification
    .magic = kMoodistBuildMagic,

    // ===== Core functions =====
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

    // ===== Wrapper functions (set to nullptr, filled by _C.so) =====
    .cudaCurrentDevice = nullptr,
    .cudaGetCurrentStream = nullptr,
    .cudaStreamGuardEnter = nullptr,
    .cudaStreamGuardExit = nullptr,
    .tensorFromBlob = nullptr,
    .tensorEmpty = nullptr,
    .tensorAddRef = nullptr,
    .tensorDecRef = nullptr,
    .tensorDataPtr = nullptr,
    .tensorNumel = nullptr,
    .tensorNdim = nullptr,
    .tensorSize = nullptr,
    .tensorStride = nullptr,
    .tensorDtype = nullptr,
    .tensorDevice = nullptr,
    .tensorIsContiguous = nullptr,
    .tensorSumOut = nullptr,
    .tensorAmaxOut = nullptr,
    .tensorAminOut = nullptr,
    .dtypeSize = nullptr,
};

} // namespace moodist

extern "C" {
__attribute__((visibility("default"))) moodist::MoodistAPI* moodistGetAPI(uint32_t expectedVersion) {
  if (expectedVersion != moodist::kMoodistAPIVersion) {
    return nullptr;
  }
  return &moodist::api;
}
}
