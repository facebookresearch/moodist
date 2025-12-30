// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "processgroup_wrapper.h"

#include "api/api_handle.h"
#include "api/tensor_ptr.h"
#include "moodist_loader.h"

namespace moodist {

// destroy() implementation for api::Queue - called by ApiHandle destructor
namespace api {
void destroy(Queue* queue) {
  coreApi.queueDestroy(queue);
}
} // namespace api

// QueueWork implementation

QueueWork::~QueueWork() {
  if (impl) {
    coreApi.queueWorkDestroy(impl);
  }
}

QueueWork& QueueWork::operator=(QueueWork&& other) noexcept {
  if (this != &other) {
    if (impl) {
      coreApi.queueWorkDestroy(impl);
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

// Queue implementation - ApiHandle manages lifetime, destructor is default

std::pair<std::optional<torch::Tensor>, size_t> Queue::get(bool block, std::optional<float> timeout) {
  TensorPtr tensor;
  size_t size = 0;
  const float* timeoutPtr = timeout ? &(*timeout) : nullptr;

  bool hasData = coreApi.queueGet(handle.get(), block, timeoutPtr, &tensor, &size);
  if (hasData) {
    return {unwrapTensor(tensor), size};
  }
  return {std::nullopt, size};
}

QueueWork Queue::put(torch::Tensor torchTensor, uint32_t transaction, bool waitOnDestroy) {
  TensorPtr tensor = wrapTensor(std::move(torchTensor));
  void* workPtr = coreApi.queuePut(handle.get(), tensor, transaction, waitOnDestroy);
  return QueueWork(workPtr);
}

size_t Queue::qsize() const {
  return coreApi.queueQsize(handle.get());
}

bool Queue::wait(std::optional<float> timeout) const {
  const float* timeoutPtr = timeout ? &(*timeout) : nullptr;
  return coreApi.queueWait(handle.get(), timeoutPtr);
}

uint32_t Queue::transactionBegin() {
  return coreApi.queueTransactionBegin(handle.get());
}

void Queue::transactionCancel(uint32_t id) {
  coreApi.queueTransactionCancel(handle.get(), id);
}

void Queue::transactionCommit(uint32_t id) {
  coreApi.queueTransactionCommit(handle.get(), id);
}

std::string_view Queue::name() const {
  return coreApi.queueName(handle.get());
}

} // namespace moodist
