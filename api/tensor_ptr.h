// Copyright (c) Meta Platforms, Inc. and affiliates.

// TensorPtr - RAII smart pointer for Tensor* with torch::Tensor-like API
// Used by both _C.so and libmoodist.so to work with tensors.
//
// In _C.so (compiled with -DMOODIST_WRAPPER):
//   - Calls wrapper functions directly via wrappers:: namespace
//
// In libmoodist.so:
//   - Calls wrapper functions through the global wrapperApi struct

#pragma once

#include "moodist_api.h"

#include <span>
#include <utility>
#include <vector>

namespace moodist {

// =============================================================================
// Conditional wrapper function access
// =============================================================================

#ifdef MOODIST_WRAPPER
// In _C.so - call wrapper functions directly (they're in the same library)
namespace wrappers {
int cudaCurrentDevice();
CUstream cudaGetCurrentStream();
void* cudaStreamGuardEnter(CUstream stream, int device);
void cudaStreamGuardExit(void* guard);
void* cudaCachingAllocatorAlloc(size_t bytes, void** cleanupCtx);
void cudaCachingAllocatorFree(void* cleanupCtx);
Tensor* tensorFromBlob(void* data, const int64_t* sizes, int ndim, DType dtype, int device);
Tensor* tensorEmpty(const int64_t* sizes, int ndim, DType dtype, int device);
void tensorAddRef(Tensor* t);
void tensorDecRef(Tensor* t);
void* tensorDataPtr(Tensor* t);
int64_t tensorNumel(Tensor* t);
int64_t tensorNdim(Tensor* t);
int64_t tensorSize(Tensor* t, int64_t dim);
int64_t tensorStride(Tensor* t, int64_t dim);
DType tensorDtype(Tensor* t);
int tensorDevice(Tensor* t);
bool tensorIsContiguous(Tensor* t);
bool tensorIsCuda(Tensor* t);
bool tensorIsCpu(Tensor* t);
int64_t tensorItemsize(Tensor* t);
Tensor* tensorView(Tensor* t, const int64_t* sizes, int ndim);
Tensor* tensorNarrow(Tensor* t, int dim, int64_t start, int64_t length);
Tensor* tensorContiguous(Tensor* t);
Tensor* tensorMulScalar(Tensor* t, double scalar);
void tensorMulScalar_(Tensor* t, double scalar);
void tensorRecordStream(Tensor* t, CUstream stream);
void tensorCopy_(Tensor* dst, Tensor* src);
void tensorSumOut(Tensor* dst, Tensor* src, int dim);
void tensorAmaxOut(Tensor* dst, Tensor* src, int dim);
void tensorAminOut(Tensor* dst, Tensor* src, int dim);
size_t dtypeSize(DType dtype);
} // namespace wrappers
#define W wrappers::
#else
// In libmoodist.so - call through global WrapperApi struct
extern WrapperApi wrapperApi;
#define W wrapperApi.
#endif

// =============================================================================
// TensorPtr class
// =============================================================================

class TensorPtr {
  Tensor* ptr_ = nullptr;

public:
  // =========================================================================
  // Constructors / Destructor
  // =========================================================================

  TensorPtr() = default;

  // Takes ownership of a raw Tensor* (caller should not decref)
  explicit TensorPtr(Tensor* p) : ptr_(p) {}

  ~TensorPtr() {
    if (ptr_) {
      W tensorDecRef(ptr_);
    }
  }

  // =========================================================================
  // Move semantics
  // =========================================================================

  TensorPtr(TensorPtr&& other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }

  TensorPtr& operator=(TensorPtr&& other) noexcept {
    if (this != &other) {
      if (ptr_) {
        W tensorDecRef(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  // =========================================================================
  // Copy semantics (increments refcount)
  // =========================================================================

  TensorPtr(const TensorPtr& other) : ptr_(other.ptr_) {
    if (ptr_) {
      W tensorAddRef(ptr_);
    }
  }

  TensorPtr& operator=(const TensorPtr& other) {
    if (this != &other) {
      if (ptr_) {
        W tensorDecRef(ptr_);
      }
      ptr_ = other.ptr_;
      if (ptr_) {
        W tensorAddRef(ptr_);
      }
    }
    return *this;
  }

  // =========================================================================
  // Validity checks
  // =========================================================================

  explicit operator bool() const {
    return ptr_ != nullptr;
  }
  bool defined() const {
    return ptr_ != nullptr;
  }

  // =========================================================================
  // Raw pointer access
  // =========================================================================

  Tensor* get() const {
    return ptr_;
  }

  // Release ownership (caller must manage refcount)
  Tensor* release() {
    Tensor* p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  // Reset to a new pointer (takes ownership)
  void reset(Tensor* p = nullptr) {
    if (ptr_) {
      W tensorDecRef(ptr_);
    }
    ptr_ = p;
  }

  // =========================================================================
  // Tensor accessors (mirror torch::Tensor API)
  // =========================================================================

  void* data_ptr() const {
    return W tensorDataPtr(ptr_);
  }

  int64_t numel() const {
    return W tensorNumel(ptr_);
  }

  int64_t dim() const {
    return W tensorNdim(ptr_);
  }

  int64_t ndimension() const {
    return dim();
  }

  int64_t size(int64_t d) const {
    return W tensorSize(ptr_, d);
  }

  int64_t stride(int64_t d) const {
    return W tensorStride(ptr_, d);
  }

  std::vector<int64_t> sizes() const {
    int64_t n = dim();
    std::vector<int64_t> result(n);
    for (int64_t i = 0; i < n; ++i) {
      result[i] = size(i);
    }
    return result;
  }

  std::vector<int64_t> strides() const {
    int64_t n = dim();
    std::vector<int64_t> result(n);
    for (int64_t i = 0; i < n; ++i) {
      result[i] = stride(i);
    }
    return result;
  }

  DType scalar_type() const {
    return W tensorDtype(ptr_);
  }

  DType dtype() const {
    return scalar_type();
  }

  int device_index() const {
    return W tensorDevice(ptr_);
  }

  bool is_contiguous() const {
    return W tensorIsContiguous(ptr_);
  }

  bool is_cuda() const {
    return W tensorIsCuda(ptr_);
  }

  bool is_cpu() const {
    return W tensorIsCpu(ptr_);
  }

  int64_t itemsize() const {
    return W tensorItemsize(ptr_);
  }

  int64_t element_size() const {
    return itemsize();
  }

  int64_t nbytes() const {
    return numel() * itemsize();
  }

  // =========================================================================
  // View operations (return new TensorPtr)
  // =========================================================================

  TensorPtr view(std::span<const int64_t> sizes) const {
    return TensorPtr(W tensorView(ptr_, sizes.data(), static_cast<int>(sizes.size())));
  }

  TensorPtr view(std::initializer_list<int64_t> sizes) const {
    return view(std::span<const int64_t>(sizes.begin(), sizes.size()));
  }

  TensorPtr narrow(int dim, int64_t start, int64_t length) const {
    return TensorPtr(W tensorNarrow(ptr_, dim, start, length));
  }

  TensorPtr contiguous() const {
    return TensorPtr(W tensorContiguous(ptr_));
  }

  // =========================================================================
  // Arithmetic operations
  // =========================================================================

  TensorPtr operator*(double scalar) const {
    return TensorPtr(W tensorMulScalar(ptr_, scalar));
  }

  TensorPtr& mul_(double scalar) {
    W tensorMulScalar_(ptr_, scalar);
    return *this;
  }

  // =========================================================================
  // In-place operations
  // =========================================================================

  TensorPtr& copy_(const TensorPtr& src) {
    W tensorCopy_(ptr_, src.ptr_);
    return *this;
  }

  void record_stream(CUstream stream) {
    W tensorRecordStream(ptr_, stream);
  }

  // =========================================================================
  // Static factory functions
  // =========================================================================

  static TensorPtr from_blob(void* data, std::span<const int64_t> sizes, DType dtype, int device) {
    return TensorPtr(W tensorFromBlob(data, sizes.data(), static_cast<int>(sizes.size()), dtype, device));
  }

  static TensorPtr empty(std::span<const int64_t> sizes, DType dtype, int device) {
    return TensorPtr(W tensorEmpty(sizes.data(), static_cast<int>(sizes.size()), dtype, device));
  }
};

// =========================================================================
// Reduction operations (free functions)
// =========================================================================

inline void sum_out(TensorPtr& dst, const TensorPtr& src, int dim) {
  W tensorSumOut(dst.get(), src.get(), dim);
}

inline void amax_out(TensorPtr& dst, const TensorPtr& src, int dim) {
  W tensorAmaxOut(dst.get(), src.get(), dim);
}

inline void amin_out(TensorPtr& dst, const TensorPtr& src, int dim) {
  W tensorAminOut(dst.get(), src.get(), dim);
}

// =========================================================================
// Utility functions
// =========================================================================

inline size_t dtype_size(DType dtype) {
  return W dtypeSize(dtype);
}

// =============================================================================
// StreamGuard RAII class
// =============================================================================

class StreamGuard {
  void* guard_;

public:
  StreamGuard(CUstream stream, int device) : guard_(W cudaStreamGuardEnter(stream, device)) {}

  ~StreamGuard() {
    W cudaStreamGuardExit(guard_);
  }

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&&) = delete;
  StreamGuard& operator=(StreamGuard&&) = delete;
};

#undef W

} // namespace moodist
