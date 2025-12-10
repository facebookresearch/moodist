// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"
#include "tensor_wrapper.h"

namespace moodist {

struct TensorData {
  AllocatedCpuBufferSharedPtr buffer;
  uintptr_t dataPtr;
  size_t dataBytes;
  int dtype;
  std::vector<int64_t> shape;
  bool isCuda;

  void clear() {
    buffer = {};
    dtype = -1;
    shape.clear();
    dataPtr = 0;
    dataBytes = 0;
    isCuda = false;
  }

  uintptr_t data() {
    return dataPtr;
  }
  size_t bytes() {
    return dataBytes;
  }
  size_t itemsize() {
    return dtypeItemsize(static_cast<DType>(dtype));
  }
  size_t numel() {
    size_t r = 1;
    for (int64_t n : shape) {
      r *= n;
    }
    return r;
  }
};

using TensorDataPtr = FLPtr<TensorData>;
using TensorDataSharedPtr = FLSharedPtr<TensorData>;

// makeTensor() is declared in tensor_factory.h (wrapper library)

struct FutureImpl {
  TensorDataPtr result;
  std::atomic_uint32_t done = 0;
  // WorkCudaDonePtr cudaDone = nullptr;
  void clear() {
    result = nullptr;
    done = 0;
  }
};

using FutureImplSharedPtr = FLSharedPtr<FutureImpl>;

} // namespace moodist
