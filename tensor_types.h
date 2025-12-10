// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"
#include "torch_includes.h"

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
    return torch::elementSize(((torch::ScalarType)dtype));
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

inline torch::Tensor makeTensor(TensorDataPtr ptr) {
  torch::Device device(torch::kCPU);

  CHECK(ptr != nullptr);

  CHECK(!ptr->isCuda);

  cpu_allocator::refCpuBuffer(ptr->buffer);

  TensorData* td = &*ptr;

  CHECK(td->data() != 0);

  Function<void()> f = [ptr = std::move(ptr)]() mutable { cpu_allocator::derefCpuBuffer(ptr->data()); };
  auto deleter = [](void* c) { Function<void()>(FunctionPointer(c))(); };
  auto data = torch::DataPtr((void*)td->data(), (void*)f.release(), deleter, device);

  torch::Storage storage(torch::Storage::use_byte_size_t(), td->bytes(), std::move(data), nullptr, false);
  return torch::empty({0}, torch::TensorOptions((torch::ScalarType)td->dtype).device(device))
      .set_(std::move(storage), 0, td->shape);
}

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
