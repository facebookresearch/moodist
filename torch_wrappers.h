// Copyright (c) Meta Platforms, Inc. and affiliates.

// This header declares wrapper functions for PyTorch APIs.
// Core library includes this header; wrapper library provides implementations.

#pragma once

#include "commondefs.h"
#include "tensor_wrapper.h"

#include <cuda.h>

namespace moodist {

// Device functions
MOODIST_API int cudaCurrentDevice();
MOODIST_API CUstream cudaGetCurrentStream();

// Stream guard - RAII wrapper for c10::cuda::CUDAStreamGuard
class StreamGuard {
  alignas(64) char storage_[128];
public:
  MOODIST_API StreamGuard(CUstream stream, int device_index);
  MOODIST_API ~StreamGuard();

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard& operator=(const StreamGuard&) = delete;
};

// Memory allocation via PyTorch's caching allocator
// Returns a wrapper that will free memory when destroyed
class DataPtrWrapper {
  alignas(64) char storage_[128];
  bool valid_ = false;
public:
  MOODIST_API DataPtrWrapper();
  MOODIST_API ~DataPtrWrapper();
  MOODIST_API DataPtrWrapper(DataPtrWrapper&& other) noexcept;
  MOODIST_API DataPtrWrapper& operator=(DataPtrWrapper&& other) noexcept;

  DataPtrWrapper(const DataPtrWrapper&) = delete;
  DataPtrWrapper& operator=(const DataPtrWrapper&) = delete;

  MOODIST_API void* get() const;
  MOODIST_API explicit operator bool() const;

  void* storage() { return storage_; }
  const void* storage() const { return storage_; }
  void setValid(bool v) { valid_ = v; }
};

MOODIST_API DataPtrWrapper cudaAllocate(size_t bytes);

} // namespace moodist
