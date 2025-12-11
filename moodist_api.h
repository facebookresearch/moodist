// Copyright (c) Meta Platforms, Inc. and affiliates.

// API struct for moodist - passed from libmoodist.so to _C.so
// This avoids dynamic linker RPATH issues and makes the interface explicit.

#pragma once

#include "moodist_version.h"

#include <Python.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace moodist {

// Forward declarations (opaque types)
struct StoreImpl;
struct Buffer;

// API struct containing all function pointers
struct MoodistAPI {
  // Magic value for runtime verification (changes per build)
  uint64_t magic;

  // Store functions
  StoreImpl* (*createStoreImpl)(std::string_view hostname, int port, std::string_view key,
                                 int worldSize, int rank);
  void (*storeImplDecRef)(StoreImpl* impl);
  void (*storeImplSet)(StoreImpl* impl, std::chrono::steady_clock::duration timeout,
                       std::string_view key, const std::vector<uint8_t>& value);
  std::vector<uint8_t> (*storeImplGet)(StoreImpl* impl, std::chrono::steady_clock::duration timeout,
                                        std::string_view key);
  bool (*storeImplCheck)(StoreImpl* impl, std::chrono::steady_clock::duration timeout,
                         std::span<const std::string_view> keys);
  void (*storeImplWait)(StoreImpl* impl, std::chrono::steady_clock::duration timeout,
                        std::span<const std::string_view> keys);

  // Serialize functions
  Buffer* (*serializeObjectImpl)(PyObject* o);
  void* (*serializeBufferPtr)(Buffer* buf);
  size_t (*serializeBufferSize)(Buffer* buf);
  void (*serializeBufferAddRef)(Buffer* buf);
  void (*serializeBufferDecRef)(Buffer* buf);
  PyObject* (*deserializeObjectImpl)(const void* ptr, size_t len);
};

} // namespace moodist

// C linkage for dlsym
extern "C" {
// Returns API struct if version matches, nullptr otherwise
// Caller should also verify api->magic == kMoodistBuildMagic
__attribute__((visibility("default")))
moodist::MoodistAPI* moodistGetAPI(uint32_t expectedVersion);
}
