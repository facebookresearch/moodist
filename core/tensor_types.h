// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

#include <array>

namespace moodist {

struct TensorData {
  AllocatedCpuBufferSharedPtr buffer;
  uintptr_t dataPtr;
  size_t dataBytes;
  int dtype; // Stores torch::ScalarType value (matches moodist::DType in moodist_api.h)
  IVector<int64_t> shape;
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
  // Returns element size in bytes, derived from dtype (torch::ScalarType values).
  size_t itemsize() {
    // DType values: UInt8=0, Int8=1, Int16=2, Int32=3, Int64=4,
    //               Float16=5, Float32=6, Float64=7, Bool=11, BFloat16=15
    static constexpr std::array<size_t, 16> table = {{
        1, // 0: UInt8
        1, // 1: Int8
        2, // 2: Int16
        4, // 3: Int32
        8, // 4: Int64
        2, // 5: Float16
        4, // 6: Float32
        8, // 7: Float64
        0, // 8: unused
        0, // 9: unused
        0, // 10: unused
        1, // 11: Bool
        0, // 12: unused
        0, // 13: unused
        0, // 14: unused
        2, // 15: BFloat16
    }};
    CHECK(dtype >= 0 && static_cast<size_t>(dtype) < table.size() && table[dtype] != 0);
    return table[dtype];
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
