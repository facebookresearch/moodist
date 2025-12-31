// Copyright (c) Meta Platforms, Inc. and affiliates.

// Store API for moodist.
// Uses ApiHandle pattern for refcounted ownership across the API boundary.

#pragma once

#include "types.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace moodist {

// Factory function - returns ApiHandle (ownership via RVO)
api::StoreHandle createStore(std::string_view hostname, int port, std::string_view key, int worldSize, int rank);

// Destroy function - called by ApiHandle when refcount reaches 0
void storeDestroy(api::Store* store);

// Optional: explicitly trigger shutdown before last reference is dropped
void storeClose(api::Store* store);

// Store operations - operate on api::Store* pointers
void storeSet(api::Store* store, std::chrono::steady_clock::duration timeout, std::string_view key,
    const std::vector<uint8_t>& value);
std::vector<uint8_t> storeGet(api::Store* store, std::chrono::steady_clock::duration timeout, std::string_view key);
bool storeCheck(api::Store* store, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);
void storeWait(api::Store* store, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

} // namespace moodist
