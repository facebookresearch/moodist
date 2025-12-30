// Copyright (c) Meta Platforms, Inc. and affiliates.

// ApiHandle - smart pointer for refcounted objects across the API boundary.
//
// Core objects inherit from ApiRefCounted to expose their refcount.
// This allows the wrapper to do addRef via direct atomic increment (no API call),
// and only destruction requires crossing the API boundary.

#pragma once

#include "moodist_api.h"

#include <atomic>
#include <cstdint>
#include <utility>

namespace moodist {

// Base class for all refcounted objects passed across the API boundary.
// Must be the first base class (or the struct must start with this layout)
// so that casting void* to ApiRefCounted* works correctly.
struct ApiRefCounted {
  std::atomic_size_t refcount;
};

#ifdef MOODIST_WRAPPER

// The coreApi global - defined in wrapper, declared here for ApiHandle
extern CoreApi coreApi;

// Smart pointer for managing ApiRefCounted objects across the API boundary.
//
// Template parameter is a pointer-to-member for the destroy function in CoreApi.
// The member pointer is a compile-time constant (just an offset), even though
// the actual function pointer is set at runtime via dlsym.
//
// Usage:
//   using QueueHandle = ApiHandle<&CoreApi::queueDestroy>;
//   using FutureHandle = ApiHandle<&CoreApi::futureDestroy>;
//
template<void (*CoreApi::*Destroy)(void*)>
class ApiHandle {
  ApiRefCounted* ptr_ = nullptr;

  // Private constructor - use adopt() or addRef() instead
  struct AdoptTag {};
  ApiHandle(void* p, AdoptTag) : ptr_(static_cast<ApiRefCounted*>(p)) {}

public:
  ApiHandle() = default;

  // Adopt ownership from API call (refcount already set by core's SharedPtr.release())
  static ApiHandle adopt(void* p) {
    return ApiHandle(p, AdoptTag{});
  }

  // Add a reference to an existing object (increments refcount)
  static ApiHandle addRef(void* p) {
    auto handle = ApiHandle(p, AdoptTag{});
    if (handle.ptr_) {
      handle.ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
    }
    return handle;
  }

  ~ApiHandle() {
    reset();
  }

  // Copy: increment refcount
  ApiHandle(const ApiHandle& other) : ptr_(other.ptr_) {
    if (ptr_) {
      ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
    }
  }

  ApiHandle& operator=(const ApiHandle& other) {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      if (ptr_) {
        ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return *this;
  }

  // Move: transfer ownership
  ApiHandle(ApiHandle&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  ApiHandle& operator=(ApiHandle&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  // Decrement refcount, destroy if zero
  void reset() {
    if (ptr_) {
      if (ptr_->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        (coreApi.*Destroy)(ptr_);
      }
      ptr_ = nullptr;
    }
  }

  // Get raw pointer (as void* for API calls)
  void* get() const {
    return ptr_;
  }

  // Get typed pointer (for direct refcount access if needed)
  ApiRefCounted* ptr() const {
    return ptr_;
  }

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  // Release ownership without decrementing refcount
  void* release() {
    void* p = ptr_;
    ptr_ = nullptr;
    return p;
  }
};

#endif // MOODIST_WRAPPER

} // namespace moodist
