// Copyright (c) Meta Platforms, Inc. and affiliates.

// TensorPtr - RAII smart pointer for Tensor* with torch::Tensor-like API
// Used by core library to work with tensors without manual refcounting.

#pragma once

#include "moodist_api.h"

#include <span>
#include <utility>
#include <vector>

namespace moodist {

// Forward declaration - getMoodistAPI() is in moodist_loader.h (wrapper)
// For core, this is resolved at link time
MoodistAPI* getMoodistAPI();

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
      getMoodistAPI()->tensorDecRef(ptr_);
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
        getMoodistAPI()->tensorDecRef(ptr_);
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
      getMoodistAPI()->tensorAddRef(ptr_);
    }
  }

  TensorPtr& operator=(const TensorPtr& other) {
    if (this != &other) {
      if (ptr_) {
        getMoodistAPI()->tensorDecRef(ptr_);
      }
      ptr_ = other.ptr_;
      if (ptr_) {
        getMoodistAPI()->tensorAddRef(ptr_);
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
      getMoodistAPI()->tensorDecRef(ptr_);
    }
    ptr_ = p;
  }

  // =========================================================================
  // Tensor accessors (mirror torch::Tensor API)
  // =========================================================================

  void* data_ptr() const {
    return getMoodistAPI()->tensorDataPtr(ptr_);
  }

  int64_t numel() const {
    return getMoodistAPI()->tensorNumel(ptr_);
  }

  int ndimension() const {
    return getMoodistAPI()->tensorNdim(ptr_);
  }

  int dim() const {
    return ndimension();
  }

  int64_t size(int dim) const {
    return getMoodistAPI()->tensorSize(ptr_, dim);
  }

  int64_t stride(int dim) const {
    return getMoodistAPI()->tensorStride(ptr_, dim);
  }

  std::vector<int64_t> sizes() const {
    int n = ndimension();
    std::vector<int64_t> result(n);
    for (int i = 0; i < n; ++i) {
      result[i] = size(i);
    }
    return result;
  }

  std::vector<int64_t> strides() const {
    int n = ndimension();
    std::vector<int64_t> result(n);
    for (int i = 0; i < n; ++i) {
      result[i] = stride(i);
    }
    return result;
  }

  DType scalar_type() const {
    return getMoodistAPI()->tensorDtype(ptr_);
  }

  DType dtype() const {
    return scalar_type();
  }

  int device_index() const {
    return getMoodistAPI()->tensorDevice(ptr_);
  }

  bool is_contiguous() const {
    return getMoodistAPI()->tensorIsContiguous(ptr_);
  }

  bool is_cuda() const {
    return getMoodistAPI()->tensorIsCuda(ptr_);
  }

  bool is_cpu() const {
    return getMoodistAPI()->tensorIsCpu(ptr_);
  }

  int64_t itemsize() const {
    return getMoodistAPI()->tensorItemsize(ptr_);
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
    return TensorPtr(getMoodistAPI()->tensorView(ptr_, sizes.data(), static_cast<int>(sizes.size())));
  }

  TensorPtr view(std::initializer_list<int64_t> sizes) const {
    return view(std::span<const int64_t>(sizes.begin(), sizes.size()));
  }

  TensorPtr narrow(int dim, int64_t start, int64_t length) const {
    return TensorPtr(getMoodistAPI()->tensorNarrow(ptr_, dim, start, length));
  }

  TensorPtr contiguous() const {
    return TensorPtr(getMoodistAPI()->tensorContiguous(ptr_));
  }

  // =========================================================================
  // Arithmetic operations
  // =========================================================================

  TensorPtr operator*(double scalar) const {
    return TensorPtr(getMoodistAPI()->tensorMulScalar(ptr_, scalar));
  }

  TensorPtr& mul_(double scalar) {
    getMoodistAPI()->tensorMulScalar_(ptr_, scalar);
    return *this;
  }

  // =========================================================================
  // In-place operations
  // =========================================================================

  TensorPtr& copy_(const TensorPtr& src) {
    getMoodistAPI()->tensorCopy_(ptr_, src.ptr_);
    return *this;
  }

  void record_stream(CUstream stream) {
    getMoodistAPI()->tensorRecordStream(ptr_, stream);
  }

  // =========================================================================
  // Static factory functions
  // =========================================================================

  static TensorPtr from_blob(void* data, std::span<const int64_t> sizes, DType dtype, int device) {
    return TensorPtr(
        getMoodistAPI()->tensorFromBlob(data, sizes.data(), static_cast<int>(sizes.size()), dtype, device));
  }

  static TensorPtr empty(std::span<const int64_t> sizes, DType dtype, int device) {
    return TensorPtr(getMoodistAPI()->tensorEmpty(sizes.data(), static_cast<int>(sizes.size()), dtype, device));
  }
};

// =========================================================================
// Reduction operations (free functions)
// =========================================================================

inline void sum_out(TensorPtr& dst, const TensorPtr& src, int dim) {
  getMoodistAPI()->tensorSumOut(dst.get(), src.get(), dim);
}

inline void amax_out(TensorPtr& dst, const TensorPtr& src, int dim) {
  getMoodistAPI()->tensorAmaxOut(dst.get(), src.get(), dim);
}

inline void amin_out(TensorPtr& dst, const TensorPtr& src, int dim) {
  getMoodistAPI()->tensorAminOut(dst.get(), src.get(), dim);
}

// =========================================================================
// Utility functions
// =========================================================================

inline size_t dtype_size(DType dtype) {
  return getMoodistAPI()->dtypeSize(dtype);
}

} // namespace moodist
