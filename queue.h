// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "commondefs.h"
#include "tensor_wrapper.h"

#include <memory>
#include <optional>
#include <string_view>

namespace moodist {

struct Group;  // forward declaration
struct QueueWorkImpl;

struct QueueWork {
  std::shared_ptr<QueueWorkImpl> impl;
  StorageWrapper storage;  // keeps tensor storage alive during async put (safer than TensorWrapper)
  bool waitOnDestroy = true;
  MOODIST_API QueueWork();
  MOODIST_API ~QueueWork();
  QueueWork(const QueueWork&) = delete;
  QueueWork(QueueWork&&) = default;
  MOODIST_API void wait();
};

struct Queue {
  void* impl = nullptr;
  Queue() = delete;
  MOODIST_API Queue(void*);
  Queue(Queue&) = delete;
  Queue& operator=(Queue) = delete;
  MOODIST_API ~Queue();
  MOODIST_API std::pair<std::optional<TensorWrapper>, size_t> get(bool block = true, std::optional<float> timeout = {});
  MOODIST_API QueueWork put(TensorWrapper value, uint32_t transactionKey, bool waitOnDestroy = true);
  MOODIST_API size_t qsize() const;
  MOODIST_API bool wait(std::optional<float> timeout) const;

  MOODIST_API uint32_t transactionBegin();
  MOODIST_API void transactionCancel(uint32_t id);
  MOODIST_API void transactionCommit(uint32_t id);

  MOODIST_API std::string_view name() const;
};

MOODIST_API std::shared_ptr<Queue> makeQueue(std::shared_ptr<Group>, int location, bool streaming, std::string_view name = {});
MOODIST_API std::shared_ptr<Queue>
makeQueue(std::shared_ptr<Group>, std::vector<int> location, bool streaming, std::string_view name = {});

} // namespace moodist
