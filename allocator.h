// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#ifdef MOODIST_WRAPPER
#include "function.h"
#endif

namespace moodist {

namespace allocator {

// Core functions - available in both core and wrapper
bool owns(uintptr_t address);
std::pair<uintptr_t, size_t> mappedRegion(uintptr_t address);

#ifdef MOODIST_WRAPPER
// Wrapper-only functions - require CUDAAllocator

void removeFreeCallback(uintptr_t baseAddress, void* handle);

struct FreeCallbackHandle {
  uintptr_t baseAddress = 0;
  void* handle = nullptr;
  FreeCallbackHandle() = default;
  FreeCallbackHandle(FreeCallbackHandle&& n) noexcept {
    *this = std::move(n);
  }
  ~FreeCallbackHandle() {
    if (handle) {
      removeFreeCallback(baseAddress, handle);
    }
  }
  FreeCallbackHandle& operator=(FreeCallbackHandle&& n) noexcept {
    std::swap(baseAddress, n.baseAddress);
    std::swap(handle, n.handle);
    return *this;
  }
  void release() {
    baseAddress = 0;
    handle = nullptr;
  }
};

FreeCallbackHandle addFreeCallback(uintptr_t baseAddress, Function<void()> callback);

#endif // MOODIST_WRAPPER

} // namespace allocator

#ifdef MOODIST_WRAPPER
void enableCudaAllocator();
#endif

} // namespace moodist
