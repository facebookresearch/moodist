// Copyright (c) Meta Platforms, Inc. and affiliates.

// Store C API for moodist.
// This header can be included by both core library and wrapper.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#ifndef MOODIST_API
#define MOODIST_API __attribute__((visibility("default")))
#endif

namespace moodist {

// Forward declare StoreImpl (opaque type)
struct StoreImpl;

// Factory function to create StoreImpl
MOODIST_API StoreImpl* createStoreImpl(
    std::string hostname, int port, std::string key, int worldSize, int rank);

// StoreImpl public interface
MOODIST_API void storeImplAddRef(StoreImpl* impl);
MOODIST_API void storeImplDecRef(StoreImpl* impl);  // calls shutdown and deletes if refcount reaches 0

MOODIST_API void storeImplSet(
    StoreImpl* impl, std::chrono::steady_clock::duration timeout,
    const std::string& key, const std::vector<uint8_t>& value);

MOODIST_API std::vector<uint8_t> storeImplGet(
    StoreImpl* impl, std::chrono::steady_clock::duration timeout,
    const std::string& key);

MOODIST_API bool storeImplCheck(
    StoreImpl* impl, std::chrono::steady_clock::duration timeout,
    const std::vector<std::string>& keys);

MOODIST_API void storeImplWait(
    StoreImpl* impl, std::chrono::steady_clock::duration timeout,
    const std::vector<std::string>& keys);

} // namespace moodist
