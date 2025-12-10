// Copyright (c) Meta Platforms, Inc. and affiliates.

// Internal queue functions used by core library.
// These functions don't require PyTorch types.

#pragma once

#include "common.h"
#include "group.h"
#include "tensor_types.h"

#include <string_view>

namespace moodist {

constexpr uint8_t transactionOpCommit = 1;
constexpr uint8_t transactionOpCancel = 2;

uint32_t queuePrepare(uintptr_t queueAddress, uint32_t source, uint32_t getKey, uint32_t transactionKey);
void queueFinish(Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key, TensorDataPtr tensor,
    uint32_t getKey, uint32_t transactionKey, size_t queueSize);

void queueRemoteGetStart(
    Group* group, uintptr_t queueAddress, uint32_t source, uintptr_t remoteQueueAddress, uint32_t key);
void queueRemoteGetStop(
    Group* group, uintptr_t queueAddress, uint32_t source, uintptr_t remoteQueueAddress, uint32_t key);
void queueRemoteGetStopAck(
    Group* group, uintptr_t queueAddress, uint32_t source, uintptr_t remoteQueueAddress, uint32_t key);

void queueTransactionCommit(Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key);
void queueTransactionCancel(Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key);

struct QueueInfo {
  uintptr_t address;
  int location;
  bool streaming;
};
QueueInfo queueGetOrCreate(std::string_view name, int location, bool streaming);

} // namespace moodist
