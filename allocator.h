// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "function.h"

#include <cstdint>

namespace moodist {

namespace allocator {

bool owns(uintptr_t address);
std::pair<uintptr_t, size_t> mappedRegion(uintptr_t address);

void removeFreeCallback(uintptr_t baseAddress, void* handle);

struct FreeCallbackHandle {
  uintptr_t baseAddress = 0;
  void* handle = nullptr;
  FreeCallbackHandle() = default;
  FreeCallbackHandle(FreeCallbackHandle&& n) {
    *this = std::move(n);
  }
  ~FreeCallbackHandle() {
    if (handle) {
      removeFreeCallback(baseAddress, handle);
    }
  }
  FreeCallbackHandle& operator=(FreeCallbackHandle&& n) {
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

} // namespace allocator

} // namespace moodist