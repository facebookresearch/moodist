// Copyright (c) Meta Platforms, Inc. and affiliates.

// Header for loading the moodist API in the wrapper library

#pragma once

#include "tensor_ptr.h"

// Forward declare torch::Tensor (avoid including torch headers here)
namespace at {
class Tensor;
}
namespace torch {
using Tensor = at::Tensor;
}

namespace moodist {

// Get the MoodistAPI struct, loading libmoodist.so if needed
// This is the wrapper-side function that uses dlopen/dlsym
MoodistAPI* getMoodistAPI();

// Convert torch::Tensor to TensorPtr
TensorPtr wrapTensor(torch::Tensor t);

// Convert TensorPtr to torch::Tensor
torch::Tensor unwrapTensor(const TensorPtr& t);

} // namespace moodist
