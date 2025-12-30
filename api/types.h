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

} // namespace moodist::api
