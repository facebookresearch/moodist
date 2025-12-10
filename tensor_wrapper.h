// Copyright (c) Meta Platforms, Inc. and affiliates.

// TensorWrapper - opaque wrapper around torch::Tensor for core library.
// Core library includes this header; wrapper library provides implementation.

#pragma once

#include "commondefs.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moodist {

// Internal dtype enum - maps to torch::ScalarType
enum class DType : int {
  Float = 0,
  Double = 1,
  Half = 2,
  BFloat16 = 3,
  Int = 4,
  Long = 5,
  Short = 6,
  Byte = 7,
  Bool = 8,
  UInt8 = 9,
  Int8 = 10,
  // Add more as needed
};

// Internal reduce op enum - maps to c10d::ReduceOp
enum class ReduceOp : int {
  SUM = 0,
  AVG = 1,
  MIN = 2,
  MAX = 3,
  PRODUCT = 4,
  PREMUL_SUM = 5,
};

// Lightweight device info - mirrors torch::Device interface
struct DeviceInfo {
  int index_ = -1;
  bool is_cuda_ = false;

  int index() const {
    return index_;
  }
  bool is_cuda() const {
    return is_cuda_;
  }
};

class TensorWrapper {
  // Storage sized to fit torch::Tensor (typically ~64-88 bytes depending on version)
  // We use 128 bytes with 64-byte alignment to be safe
  alignas(64) char storage_[128];
  bool valid_ = false;

public:
  MOODIST_API TensorWrapper();
  MOODIST_API ~TensorWrapper();
  MOODIST_API TensorWrapper(TensorWrapper&& other) noexcept;
  MOODIST_API TensorWrapper& operator=(TensorWrapper&& other) noexcept;

  // Copy semantics - forwards to torch::Tensor's ref-counted copy
  MOODIST_API TensorWrapper(const TensorWrapper& other);
  MOODIST_API TensorWrapper& operator=(const TensorWrapper& other);

  // Check if wrapper holds a valid tensor
  MOODIST_API explicit operator bool() const;
  MOODIST_API bool defined() const;

  // Accessors - mirror torch::Tensor interface
  MOODIST_API void* data_ptr() const;
  MOODIST_API void* mutable_data_ptr();
  MOODIST_API const void* const_data_ptr() const;
  MOODIST_API size_t numel() const;
  MOODIST_API size_t itemsize() const;
  MOODIST_API DeviceInfo device() const;
  MOODIST_API int device_index() const;
  MOODIST_API DType dtype() const;
  MOODIST_API bool is_contiguous() const;
  MOODIST_API bool is_cuda() const;
  MOODIST_API bool is_cpu() const;
  MOODIST_API const void* storage_data() const; // returns tensor.storage().data()

  // Shape info
  MOODIST_API int64_t ndim() const;
  MOODIST_API int64_t size(int dim) const;
  MOODIST_API int64_t stride(int dim) const;
  MOODIST_API std::vector<int64_t> sizes() const;
  MOODIST_API std::vector<int64_t> strides() const;

  // Access underlying storage (for wrapper implementation only)
  void* storage() {
    return storage_;
  }
  const void* storage() const {
    return storage_;
  }
  void setValid(bool v) {
    valid_ = v;
  }
};

// Factory functions - implemented in wrapper
MOODIST_API TensorWrapper tensorFromBlob(void* data, const std::vector<int64_t>& sizes, DType dtype, int device_index);
MOODIST_API TensorWrapper tensorFromBlob(void* data, const std::vector<int64_t>& sizes, DType dtype, DeviceInfo device);
MOODIST_API TensorWrapper tensorEmpty(const std::vector<int64_t>& sizes, DType dtype, int device_index);
MOODIST_API TensorWrapper tensorEmpty(const std::vector<int64_t>& sizes, DType dtype, DeviceInfo device);

// Tensor operations - implemented in wrapper
MOODIST_API void tensorSumOut(TensorWrapper& dst, const TensorWrapper& src, int dim);
MOODIST_API void tensorAmaxOut(TensorWrapper& dst, const TensorWrapper& src, int dim);
MOODIST_API void tensorAminOut(TensorWrapper& dst, const TensorWrapper& src, int dim);
MOODIST_API void tensorAdd_(TensorWrapper& dst, const TensorWrapper& src);
MOODIST_API void tensorCopy_(TensorWrapper& dst, const TensorWrapper& src);
MOODIST_API void tensorMulScalar_(TensorWrapper& t, float scalar);

// View operations
MOODIST_API TensorWrapper tensorView(TensorWrapper& t, const std::vector<int64_t>& sizes);

// Dtype utilities
MOODIST_API size_t dtypeItemsize(DType dtype);
MOODIST_API const char* dtypeName(DType dtype);

// StorageWrapper - opaque wrapper around torch::Storage for core library.
// Used in QueueWork to hold tensor data safely (Storage doesn't have Python subclass issues)
class StorageWrapper {
  // Storage sized to fit torch::Storage (typically ~16-32 bytes)
  // We use 64 bytes with 64-byte alignment to be safe
  alignas(64) char storage_[64];
  bool valid_ = false;

public:
  MOODIST_API StorageWrapper();
  MOODIST_API ~StorageWrapper();
  MOODIST_API StorageWrapper(StorageWrapper&& other) noexcept;
  MOODIST_API StorageWrapper& operator=(StorageWrapper&& other) noexcept;

  // Copy semantics - forwards to torch::Storage's ref-counted copy
  MOODIST_API StorageWrapper(const StorageWrapper& other);
  MOODIST_API StorageWrapper& operator=(const StorageWrapper& other);

  // Check if wrapper holds a valid storage
  MOODIST_API explicit operator bool() const;

  // Clear the storage
  MOODIST_API void reset();

  // Access underlying storage (for wrapper implementation only)
  void* storage() {
    return storage_;
  }
  const void* storage() const {
    return storage_;
  }
  void setValid(bool v) {
    valid_ = v;
  }
};

// Extract storage from tensor (implemented in wrapper)
MOODIST_API StorageWrapper storageFromTensor(const TensorWrapper& t);

// Forward declaration for TensorDataPtr conversion
struct TensorData;
template<typename T, size_t maxThreadLocalEntries>
struct FLPtr;
using TensorDataPtr = FLPtr<TensorData, 0x40>;

// Create TensorWrapper from TensorDataPtr (implemented in wrapper)
MOODIST_API TensorWrapper tensorFromData(TensorDataPtr ptr);

} // namespace moodist
