// Copyright (c) Meta Platforms, Inc. and affiliates.

// This header provides all PyTorch/c10/ATen includes needed by moodist.
// It handles the CHECK macro conflict between moodist and PyTorch.
//
// When we split moodist into core and PyTorch-dependent libraries,
// this header will become part of the PyTorch-dependent library.

#pragma once

#pragma push_macro("CHECK")

// c10 core
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/util/intrusive_ptr.h>

// c10 CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// ATen ops
#include <ATen/ops/from_blob.h>
#include <ATen/ops/pad.h>

// torch distributed
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

// torch core
#include <torch/torch.h>
#include <torch/types.h>

#undef CHECK
#pragma pop_macro("CHECK")
