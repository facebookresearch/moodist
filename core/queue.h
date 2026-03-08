// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "api/tensor_ptr.h"
#include "api/types.h"
#include "buffer.h"
#include "shared_ptr.h"
#include "tensor_types.h"

#include <memory>
#include <optional>
#include <string_view>

namespace moodist {

struct Group; // forward declaration
struct QueueWorkImpl;

// QueueWork inherits from api::QueueWork for type safety at the API boundary.
// Does NOT inherit from ApiRefCounted - uses unique ownership semantics.
struct QueueWork : api::QueueWork {
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
  TensorPtr getTensor(bool block = true, std::optional<float> timeout = {}, size_t* queueSize = nullptr);
  TensorDataPtr get(bool block = true, std::optional<float> timeout = {}, size_t* queueSize = nullptr);
  // Takes a copy of the TensorPtr (refcount handled automatically).
  QueueWork putTensor(TensorPtr value, uint32_t transactionKey = 0, bool waitOnDestroy = true);
  QueueWork putBuffer(BufferHandle value, uint32_t transactionKey = 0, bool waitOnDestroy = false);
  size_t qsize() const;
  bool wait(std::optional<float> timeout) const;

  uint32_t transactionBegin();
  void transactionCancel(uint32_t id);
  void transactionCommit(uint32_t id);

  std::string_view name() const;
};

api::QueueHandle makeQueue(SharedPtr<Group>, int location, bool streaming, std::string_view name = {});
api::QueueHandle makeQueue(SharedPtr<Group>, std::vector<int> location, bool streaming, std::string_view name = {});

} // namespace moodist
