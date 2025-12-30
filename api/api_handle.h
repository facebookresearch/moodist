// Copyright (c) Meta Platforms, Inc. and affiliates.

// ApiHandle - smart pointer for refcounted objects across the API boundary.
//
// Core objects inherit from ApiRefCounted to expose their refcount.
// This allows the wrapper to do addRef via direct atomic increment (no API call),
// and only destruction requires crossing the API boundary.
//
// The destroy() function is found via ADL - core and wrapper provide different
// implementations. Since ApiHandle is always returned by value (RVO), the
// destructor only ever runs in the wrapper, so only wrapper's destroy() is called.

#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace moodist {

// Base class for all refcounted objects passed across the API boundary.
// Must be the first base class (or the struct must start with this layout)
// so that casting void* to ApiRefCounted* works correctly.
struct ApiRefCounted {
  std::atomic_size_t refcount;
};

} // namespace moodist

namespace moodist::api {

// Forward declarations of API types (defined in types.h)
struct Store;
struct Queue;
struct Future;
struct CustomOp;
struct ProcessGroup;
struct Buffer;

// destroy() declarations - implementations provided by wrapper only.
// Core never calls these (ApiHandle is always returned to wrapper via RVO).
// If core somehow tried to destroy an ApiHandle, linker error = good.
void destroy(Store*);
void destroy(Queue*);
void destroy(Future*);
void destroy(CustomOp*);
void destroy(ProcessGroup*);
void destroy(Buffer*);

// Smart pointer for managing ApiRefCounted objects across the API boundary.
//
// Template parameter is the API type (e.g., api::Queue).
// The type must inherit from ApiRefCounted.
//
// Usage:
//   // Core returns:
//   ApiHandle<api::Queue> processGroupImplMakeQueue(...);
//
//   // Wrapper receives directly (RVO), manages lifetime automatically.
//
template<typename T>
class ApiHandle {
  static_assert(std::is_base_of_v<ApiRefCounted, T>, "ApiHandle requires type to inherit from ApiRefCounted");

  T* ptr_ = nullptr;

public:
  ApiHandle() = default;

  // Construct with initial refcount (for core creating new objects)
  explicit ApiHandle(T* p) : ptr_(p) {
    if (ptr_) {
      ptr_->refcount.store(1, std::memory_order_relaxed);
    }
  }

  // Adopt ownership from raw pointer (assumes refcount already set)
  struct AdoptTag {};
  ApiHandle(T* p, AdoptTag) : ptr_(p) {}

  static ApiHandle adopt(T* p) {
    return ApiHandle(p, AdoptTag{});
  }

  // Add a reference to an existing object (increments refcount)
  static ApiHandle addRef(T* p) {
    if (p) {
      p->refcount.fetch_add(1, std::memory_order_relaxed);
    }
    return ApiHandle(p, AdoptTag{});
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
        destroy(ptr_); // ADL finds moodist::api::destroy
      }
      ptr_ = nullptr;
    }
  }

  // Get typed pointer
  T* get() const {
    return ptr_;
  }

  T* operator->() const {
    return ptr_;
  }

  T& operator*() const {
    return *ptr_;
  }

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

  // Release ownership without decrementing refcount
  T* release() {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }
};

} // namespace moodist::api
