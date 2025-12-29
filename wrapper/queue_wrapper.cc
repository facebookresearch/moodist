// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "processgroup_wrapper.h"

#include "../api/tensor_ptr.h"
#include "moodist_loader.h"

namespace moodist {

// QueueWork implementation

QueueWork::~QueueWork() {
  if (impl) {
    coreApi.queueWorkDecRef(impl);
  }
}

QueueWork& QueueWork::operator=(QueueWork&& other) noexcept {
  if (this != &other) {
    if (impl) {
      coreApi.queueWorkDecRef(impl);
    }
    impl = other.impl;
    other.impl = nullptr;
  }
  return *this;
}

void QueueWork::wait() {
  if (impl) {
    coreApi.queueWorkWait(impl);
  }
}

// Queue implementation

Queue::~Queue() {
  if (impl) {
    coreApi.queueDecRef(impl);
  }
}

std::pair<std::optional<torch::Tensor>, size_t> Queue::get(bool block, std::optional<float> timeout) {
  TensorPtr tensor;
  size_t size = 0;
  const float* timeoutPtr = timeout ? &(*timeout) : nullptr;

  bool hasData = coreApi.queueGet(impl, block, timeoutPtr, &tensor, &size);
  if (hasData) {
    return {unwrapTensor(tensor), size};
  }
  return {std::nullopt, size};
}

QueueWork Queue::put(torch::Tensor torchTensor, uint32_t transaction, bool waitOnDestroy) {
  TensorPtr tensor = wrapTensor(std::move(torchTensor));
  void* workPtr = coreApi.queuePut(impl, tensor, transaction, waitOnDestroy);
  return QueueWork(workPtr);
}

size_t Queue::qsize() const {
  return coreApi.queueQsize(impl);
}

bool Queue::wait(std::optional<float> timeout) const {
  const float* timeoutPtr = timeout ? &(*timeout) : nullptr;
  return coreApi.queueWait(impl, timeoutPtr);
}

uint32_t Queue::transactionBegin() {
  return coreApi.queueTransactionBegin(impl);
}

void Queue::transactionCancel(uint32_t id) {
  coreApi.queueTransactionCancel(impl, id);
}

void Queue::transactionCommit(uint32_t id) {
  coreApi.queueTransactionCommit(impl, id);
}

std::string_view Queue::name() const {
  return coreApi.queueName(impl);
}

} // namespace moodist
