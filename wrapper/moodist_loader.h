// Copyright (c) Meta Platforms, Inc. and affiliates.

// Header for loading the moodist Api in the wrapper library (_C.so)

#pragma once

#include "api/tensor_ptr.h"

// Forward declare torch::Tensor (avoid including torch headers here)
namespace at {
class Tensor;
}
namespace torch {
using Tensor = at::Tensor;
}

namespace moodist {

// Global CoreApi object - initialized by initMoodistApi(), then accessed directly
// Using an object (not pointer) means function pointer reads are single memory ops
extern CoreApi coreApi;

// Initialize the moodist Api by loading libmoodist.so
// Must be called before any Api functions are used (typically in PYBIND11_MODULE)
void initMoodistApi();

// Convert torch::Tensor to TensorPtr
TensorPtr wrapTensor(torch::Tensor t);

// Convert TensorPtr to torch::Tensor
torch::Tensor unwrapTensor(const TensorPtr& t);

} // namespace moodist
