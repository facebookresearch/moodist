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

// Forward declare StoreImpl (opaque type)
struct StoreImpl;

// Factory function to create StoreImpl
StoreImpl* createStoreImpl(std::string_view hostname, int port, std::string_view key, int worldSize, int rank);

// Ref counting
void storeImplAddRef(StoreImpl* impl);
void storeImplDecRef(StoreImpl* impl);

// Set a key-value pair
void storeImplSet(StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::string_view key,
    const std::vector<uint8_t>& value);

// Get a value by key
std::vector<uint8_t> storeImplGet(StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::string_view key);

// Check if keys exist
bool storeImplCheck(
    StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

// Wait for keys to exist
void storeImplWait(
    StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

} // namespace moodist
