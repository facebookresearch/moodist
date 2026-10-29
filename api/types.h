// Copyright (c) Meta Platforms, Inc. and affiliates.

// API types - thin base classes for types that cross the API boundary.
// Core implementation types inherit from these, enabling safe upcasts.
//
// Example:
//   // In core:
//   struct QueueImpl : api::Queue { ... };
//   ApiHandle<api::Queue> createQueue() { return ApiHandle<api::Queue>::create(new QueueImpl()); }
//
//   // In wrapper:
//   auto queue = coreApi.createQueue();  // RVO, wrapper owns directly
//   // queue destructor calls destroy(api::Queue*) which is defined in wrapper

#pragma once

#include "api_handle.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace moodist::api {

// Base types for API boundary.
// These inherit from ApiRefCounted for refcounting.
// They are intentionally minimal - they exist for type safety
// and to provide a common base for safe upcasting.

struct Store : ApiRefCounted {};
struct Queue : ApiRefCounted {};
struct Future : ApiRefCounted {};
struct CustomOp : ApiRefCounted {};
struct ProcessGroup : ApiRefCounted {};
struct Buffer : ApiRefCounted {};

// QueueWork uses unique ownership (no refcounting)
// It doesn't inherit from ApiRefCounted, so ApiHandle uses move-only semantics.
struct QueueWork {};

// RdmaWork uses unique ownership (like QueueWork).
// Waits for RDMA read completion on destruction.
struct RdmaWork {};

// RdmaEndpoint — serializable descriptor for a registered memory region.
// Created by rdmaRegister, sent over a queue, consumed by rdmaRead.
struct RdmaEndpoint {
  uintptr_t address = 0;
  size_t bytes = 0;
  uint32_t rank = 0;
  bool isCuda = false;
  uint32_t rkeys[4] = {}; // one per IB device (maxDevices=4)
};

// Reduce operation for compile_op - how to handle overlapping inputs
enum class ReduceOp : int {
  None = 0, // Error on overlap (default)
  Any = 1,  // Pick any source (all sources have same value)
  // Every output element is the sum (average, minimum, maximum) of the inputs that cover it
  Sum = 2,
  Avg = 3,
  Min = 4,
  Max = 5,
};

// Input/output specification for compile_op
// Describes a region of a tensor that a rank contributes or receives
struct TensorRegion {
  int32_t rank;
  std::vector<int64_t> offset;
  std::vector<int64_t> shape;
  std::string_view tensorId; // Caller keeps underlying string alive
  std::string_view device;   // "cpu", "cuda", or "cuda:N" - caller keeps string alive
};

} // namespace moodist::api
