// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "processgroup_wrapper.h"

#include "api/api_handle.h"
#include "api/tensor_ptr.h"
#include "moodist_loader.h"

namespace moodist {

// ApiProxy method implementations for Queue
namespace api {

size_t ApiProxy<Queue>::qsize() const {
  return coreApi.queueQsize(ptr);
}

bool ApiProxy<Queue>::wait(const float* timeout) const {
  return coreApi.queueWait(ptr, timeout);
}

uint32_t ApiProxy<Queue>::transactionBegin() {
  return coreApi.queueTransactionBegin(ptr);
}

void ApiProxy<Queue>::transactionCancel(uint32_t id) {
  coreApi.queueTransactionCancel(ptr, id);
}

void ApiProxy<Queue>::transactionCommit(uint32_t id) {
  coreApi.queueTransactionCommit(ptr, id);
}

const char* ApiProxy<Queue>::name() const {
  return coreApi.queueName(ptr);
}

bool ApiProxy<Queue>::get(bool block, const float* timeout, TensorPtr* outTensor, size_t* outSize) {
  return coreApi.queueGet(ptr, block, timeout, outTensor, outSize);
}

ApiHandle<QueueWork> ApiProxy<Queue>::put(const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy) {
  return coreApi.queuePut(ptr, tensor, transaction, waitOnDestroy);
}

// ApiProxy method implementations for QueueWork
void ApiProxy<QueueWork>::wait() {
  coreApi.queueWorkWait(ptr);
}

// destroy() implementations - called by ApiHandle destructor
void destroy(Queue* queue) {
  coreApi.queueDestroy(queue);
}

void destroy(QueueWork* work) {
  coreApi.queueWorkDestroy(work);
}

} // namespace api

// QueueWork implementation - ApiHandle manages lifetime

void QueueWork::wait() {
  if (handle) {
    handle->wait();
  }
}

// Queue implementation - ApiHandle manages lifetime, destructor is default

std::pair<std::optional<torch::Tensor>, size_t> Queue::get(bool block, std::optional<float> timeout) {
  TensorPtr tensor;
  size_t size = 0;
  const float* timeoutPtr = timeout ? &(*timeout) : nullptr;

  bool hasData = handle->get(block, timeoutPtr, &tensor, &size);
  if (hasData) {
    return {unwrapTensor(tensor), size};
  }
  return {std::nullopt, size};
}

QueueWork Queue::put(torch::Tensor torchTensor, uint32_t transaction, bool waitOnDestroy) {
  TensorPtr tensor = wrapTensor(std::move(torchTensor));
  return QueueWork(handle->put(tensor, transaction, waitOnDestroy));
}

size_t Queue::qsize() const {
  return handle->qsize();
}

bool Queue::wait(std::optional<float> timeout) const {
  const float* timeoutPtr = timeout ? &(*timeout) : nullptr;
  return handle->wait(timeoutPtr);
}

uint32_t Queue::transactionBegin() {
  return handle->transactionBegin();
}

void Queue::transactionCancel(uint32_t id) {
  handle->transactionCancel(id);
}

void Queue::transactionCommit(uint32_t id) {
  handle->transactionCommit(id);
}

std::string_view Queue::name() const {
  return handle->name();
}

} // namespace moodist
