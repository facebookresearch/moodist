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
#include <cstddef>
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

// ApiProxy - provides member function syntax for API types.
// operator-> returns this proxy, which has methods that call coreApi functions.
// Methods are declared here but defined in wrapper (where coreApi exists).
// Core doesn't call these methods - it has direct access to internals.
template<typename T>
struct ApiProxy {
  T* ptr;
  ApiProxy* operator->() {
    return this;
  }
};

// Specialization for Buffer with data() and size() accessors
template<>
struct ApiProxy<Buffer> {
  Buffer* ptr;
  ApiProxy* operator->() {
    return this;
  }

  void* data() const;  // Defined in wrapper
  size_t size() const; // Defined in wrapper
};

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
//   // Core creates new object:
//   ApiHandle<api::Queue> createQueue() { return ApiHandle<api::Queue>::create(new QueueImpl()); }
//
//   // Wrapper receives directly (RVO), manages lifetime automatically.
//
template<typename T>
class ApiHandle {
  static_assert(std::is_base_of_v<ApiRefCounted, T>, "ApiHandle requires type to inherit from ApiRefCounted");

  T* ptr_ = nullptr;

  // Private: use static factory methods instead
  struct AdoptTag {};
  ApiHandle(T* p, AdoptTag) : ptr_(p) {}

public:
  ApiHandle() = default;

  // Create new handle, setting refcount to 1 (for core creating new objects)
  static ApiHandle create(T* p) {
    if (p) {
      p->refcount.store(1, std::memory_order_relaxed);
    }
    return ApiHandle(p, AdoptTag{});
  }

  // Adopt ownership from raw pointer (assumes refcount already set)
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

  // Returns proxy for member function syntax (e.g., handle->data())
  ApiProxy<T> operator->() const {
    return ApiProxy<T>{ptr_};
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

// Convenience typedefs for common handle types
using BufferHandle = ApiHandle<Buffer>;

} // namespace moodist::api
