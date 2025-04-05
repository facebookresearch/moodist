#pragma once

#include "common.h"
#include "group.h"

namespace moodist {

constexpr uint8_t transactionOpCommit = 1;
constexpr uint8_t transactionOpCancel = 2;

struct QueueWorkImpl;

struct QueueWork {
  std::shared_ptr<QueueWorkImpl> impl;
  std::optional<torch::Storage> storage;
  QueueWork();
  ~QueueWork();
  QueueWork(const QueueWork&) = delete;
  QueueWork(QueueWork&&) = default;
  void wait();
};

struct Queue {
  void* impl = nullptr;
  Queue() = delete;
  Queue(void*);
  Queue(Queue&) = delete;
  Queue& operator=(Queue) = delete;
  ~Queue();
  std::pair<std::optional<torch::Tensor>, size_t> get(bool block = true, std::optional<float> timeout = {});
  QueueWork put(torch::Tensor value, uint32_t transactionKey);
  size_t qsize() const;
  bool wait(std::optional<float> timeout) const;

  uint32_t transactionBegin();
  void transactionCancel(uint32_t id);
  void transactionCommit(uint32_t id);

  std::string name() const;
};

std::shared_ptr<Queue> makeQueue(std::shared_ptr<Group>, int location, bool streaming, std::optional<std::string> name);
std::shared_ptr<Queue>
makeQueue(std::shared_ptr<Group>, std::vector<int> location, bool streaming, std::optional<std::string> name);

uint32_t queuePrepare(uintptr_t queueAddress, uint32_t source, uint32_t getKey, uint32_t transactionKey);
void queueFinish(
    Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key, TensorDataPtr tensor, uint32_t getKey,
    uint32_t transactionKey, size_t queueSize);

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
QueueInfo queueGetOrCreate(const std::string& name, int location, bool streaming);

} // namespace moodist
