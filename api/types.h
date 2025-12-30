// Copyright (c) Meta Platforms, Inc. and affiliates.

// API types - thin base classes for types that cross the API boundary.
// Core implementation types inherit from these, enabling safe upcasts.
//
// Example:
//   // In core:
//   struct StoreImpl : api::Store { ... };
//   api::Store* createStore() { return new StoreImpl(); }
//
//   // In wrapper:
//   using StoreHandle = ApiHandle<&CoreApi::storeDestroy>;
//   auto store = StoreHandle::adopt(coreApi.createStore());

#pragma once

#include "api_handle.h"

namespace moodist::api {

// Base types for API boundary.
// Most inherit from ApiRefCounted for refcounting.
// These are intentionally minimal - they exist for type safety
// and to provide a common base for safe upcasting.

struct Store : ApiRefCounted {};
struct Queue : ApiRefCounted {};
struct Future : ApiRefCounted {};
struct CustomOp : ApiRefCounted {};
struct ProcessGroup : ApiRefCounted {};
struct Buffer : ApiRefCounted {};

// QueueWork uses unique ownership (no refcounting), but we still
// define it here for type safety and safe casting.
struct QueueWork {};

} // namespace moodist::api
