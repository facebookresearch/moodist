// Copyright (c) Meta Platforms, Inc. and affiliates.

// Tensor factory functions that require PyTorch.
// This header is part of the wrapper library.

#pragma once

#include "tensor_types.h"
#include "torch_includes.h"

namespace moodist {

// Create a torch::Tensor from TensorData.
// The tensor takes ownership of the data via the cpu_allocator.
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

} // namespace moodist
