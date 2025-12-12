// Copyright (c) Meta Platforms, Inc. and affiliates.

// Store API for moodist.
// Uses ABI-safe types at the library boundary.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace moodist {

// Forward declare StoreHandle (opaque type)
struct StoreHandle;

// Factory function to create StoreHandle
StoreHandle* createStore(std::string_view hostname, int port, std::string_view key, int worldSize, int rank);

// Ref counting
void storeAddRef(StoreHandle* handle);
void storeDecRef(StoreHandle* handle);

// Set a key-value pair
void storeSet(StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::string_view key,
    const std::vector<uint8_t>& value);

// Get a value by key
std::vector<uint8_t> storeGet(
    StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::string_view key);

// Check if keys exist
bool storeCheck(
    StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

// Wait for keys to exist
void storeWait(
    StoreHandle* handle, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

} // namespace moodist
