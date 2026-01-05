// Copyright (c) Meta Platforms, Inc. and affiliates.

// ApiHandle - smart pointer for objects across the API boundary.
//
// Supports two ownership modes based on whether T inherits from ApiRefCounted:
// - Refcounted (T : ApiRefCounted): shared ownership, copy increments refcount
// - Unique (T not : ApiRefCounted): unique ownership, move-only
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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Forward declare CUstream in global namespace for cuda.h compatibility
struct CUstream_st;
typedef CUstream_st* CUstream;

namespace moodist::cuda {
using CUstream = ::CUstream;
} // namespace moodist::cuda

namespace moodist {

using namespace cuda;

// Base class for refcounted objects passed across the API boundary.
// Objects inheriting from this get shared ownership semantics in ApiHandle.
// Objects NOT inheriting get unique ownership semantics (move-only).
struct ApiRefCounted {
  std::atomic_size_t refcount;
};

// Forward declare TensorPtr for Queue proxy
class TensorPtr;

} // namespace moodist

namespace moodist::api {

using namespace moodist::cuda;

// Forward declarations of API types (defined in types.h)
struct Store;
struct Queue;
struct Future;
struct CustomOp;
struct ProcessGroup;
struct Buffer;
struct QueueWork;

// Forward declare ApiHandle for use in ApiProxy specializations
template<typename T>
class ApiHandle;

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

// Specialization for Store with set/get/check/wait methods
template<>
struct ApiProxy<Store> {
  Store* ptr;
  ApiProxy* operator->() {
    return this;
  }

  // Store operations - defined in wrapper
  void close(); // Optional: explicitly trigger shutdown (called automatically on last ref drop)
  void set(std::chrono::steady_clock::duration timeout, std::string_view key, const std::vector<uint8_t>& value);
  std::vector<uint8_t> get(std::chrono::steady_clock::duration timeout, std::string_view key);
  bool check(std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);
  void wait(std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);
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

// Specialization for Queue with queue operation accessors
template<>
struct ApiProxy<Queue> {
  Queue* ptr;
  ApiProxy* operator->() {
    return this;
  }

  // Simple accessors - defined in wrapper
  size_t qsize() const;
  bool wait(const float* timeout) const;
  uint32_t transactionBegin();
  void transactionCancel(uint32_t id);
  void transactionCommit(uint32_t id);
  const char* name() const;

  // Complex methods with TensorPtr - defined in wrapper
  bool get(bool block, const float* timeout, TensorPtr* outTensor, size_t* outSize);
  ApiHandle<QueueWork> put(const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy);
};

// Specialization for QueueWork
template<>
struct ApiProxy<QueueWork> {
  QueueWork* ptr;
  ApiProxy* operator->() {
    return this;
  }

  void wait(); // Defined in wrapper
};

// Specialization for Future
template<>
struct ApiProxy<Future> {
  Future* ptr;
  ApiProxy* operator->() {
    return this;
  }

  void wait(CUstream stream);           // Defined in wrapper
  bool getResult(TensorPtr* outTensor); // Defined in wrapper
};

// Specialization for CustomOp
template<>
struct ApiProxy<CustomOp> {
  CustomOp* ptr;
  ApiProxy* operator->() {
    return this;
  }

  ApiHandle<Future> call(TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream);
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
void destroy(QueueWork*);

// Smart pointer for managing objects across the API boundary.
//
// Ownership semantics depend on whether T inherits from ApiRefCounted:
// - If T : ApiRefCounted: shared ownership (copy increments refcount)
// - If T not : ApiRefCounted: unique ownership (move-only, no copy)
//
// Usage:
//   // Refcounted (shared):
//   ApiHandle<api::Queue> createQueue() { return ApiHandle<api::Queue>::create(new QueueImpl()); }
//
//   // Unique:
//   ApiHandle<api::QueueWork> createWork() { return ApiHandle<api::QueueWork>::create(new WorkImpl()); }
//
template<typename T>
class ApiHandle {
  static constexpr bool IsRefCounted = std::is_base_of_v<ApiRefCounted, T>;

  T* ptr_ = nullptr;

  // Private: use static factory methods instead
  struct AdoptTag {};
  ApiHandle(T* p, AdoptTag) : ptr_(p) {}

public:
  ApiHandle() = default;

  // Create new handle (for core creating new objects)
  // For refcounted types, sets refcount to 1
  static ApiHandle create(T* p) {
    if constexpr (IsRefCounted) {
      if (p) {
        p->refcount.store(1, std::memory_order_relaxed);
      }
    }
    return ApiHandle(p, AdoptTag{});
  }

  // Adopt ownership from raw pointer
  // For refcounted types, assumes refcount already set
  static ApiHandle adopt(T* p) {
    return ApiHandle(p, AdoptTag{});
  }

  // Add a reference to an existing object (only for refcounted types)
  static ApiHandle addRef(T* p)
    requires IsRefCounted
  {
    if (p) {
      p->refcount.fetch_add(1, std::memory_order_relaxed);
    }
    return ApiHandle(p, AdoptTag{});
  }

  ~ApiHandle() {
    reset();
  }

  // Copy constructor - only for refcounted types
  ApiHandle(const ApiHandle& other)
    requires IsRefCounted
      : ptr_(other.ptr_) {
    if (ptr_) {
      ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Copy constructor - deleted for unique types
  ApiHandle(const ApiHandle&)
    requires(!IsRefCounted)
  = delete;

  // Copy assignment - only for refcounted types
  ApiHandle& operator=(const ApiHandle& other)
    requires IsRefCounted
  {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      if (ptr_) {
        ptr_->refcount.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return *this;
  }

  // Copy assignment - deleted for unique types
  ApiHandle& operator=(const ApiHandle&)
    requires(!IsRefCounted)
  = delete;

  // Move: transfer ownership (always available)
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

  // Destroy the held object
  // For refcounted: decrement refcount, destroy if zero
  // For unique: always destroy
  void reset() {
    if (ptr_) {
      if constexpr (IsRefCounted) {
        if (ptr_->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          destroy(ptr_); // ADL finds moodist::api::destroy
        }
      } else {
        destroy(ptr_);
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

  // Release ownership without destroying
  // For refcounted: does not decrement refcount
  T* release() {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }
};

// Convenience typedefs for common handle types
using StoreHandle = ApiHandle<Store>;
using BufferHandle = ApiHandle<Buffer>;
using QueueHandle = ApiHandle<Queue>;
using QueueWorkHandle = ApiHandle<QueueWork>;
using FutureHandle = ApiHandle<Future>;
using CustomOpHandle = ApiHandle<CustomOp>;
using ProcessGroupHandle = ApiHandle<ProcessGroup>;

} // namespace moodist::api
