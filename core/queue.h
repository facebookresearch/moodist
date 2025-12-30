// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "api/tensor_ptr.h"
#include "api/types.h"
#include "commondefs.h"
#include "shared_ptr.h"

#include <memory>
#include <optional>
#include <string_view>

namespace moodist {

struct Group; // forward declaration
struct QueueWorkImpl;

struct QueueWork {
  std::shared_ptr<QueueWorkImpl> impl;
  TensorPtr tensor; // keeps tensor alive during async put (RAII handles refcount)
  bool waitOnDestroy = true;
  QueueWork();
  ~QueueWork();
  QueueWork(const QueueWork&) = delete;
  QueueWork(QueueWork&&) = default;
  QueueWork& operator=(QueueWork&&) = default;
  void wait();
};

// Queue inherits from api::Queue (which inherits from ApiRefCounted).
// This allows safe upcasting to api::Queue* for the API boundary.
// The refcount is inherited from ApiRefCounted.
struct Queue : api::Queue {
  void* impl = nullptr;
  Queue() = delete;
  Queue(void*);
  Queue(Queue&) = delete;
  Queue& operator=(Queue) = delete;
  ~Queue();
  // Returns (tensor, queue_size). Returns empty TensorPtr if no data.
  std::pair<TensorPtr, size_t> get(bool block = true, std::optional<float> timeout = {});
  // Takes a copy of the TensorPtr (refcount handled automatically).
  QueueWork put(TensorPtr value, uint32_t transactionKey, bool waitOnDestroy = true);
  size_t qsize() const;
  bool wait(std::optional<float> timeout) const;

  uint32_t transactionBegin();
  void transactionCancel(uint32_t id);
  void transactionCommit(uint32_t id);

  std::string_view name() const;
};

api::ApiHandle<api::Queue> makeQueue(SharedPtr<Group>, int location, bool streaming, std::string_view name = {});
api::ApiHandle<api::Queue> makeQueue(
    SharedPtr<Group>, std::vector<int> location, bool streaming, std::string_view name = {});

} // namespace moodist
