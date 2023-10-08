#pragma once

#include "group.h"
#include "intrusive_list.h"
#include "synchronization.h"
#include "vector.h"

#include <forward_list>
#include <memory>

namespace moodist {

constexpr uint8_t taskTerminate = 0;
constexpr uint8_t taskBarrier = 1;
constexpr uint8_t taskAllreduce = 2;
constexpr uint8_t taskAllgather = 3;

struct QueueEntry {
  IntrusiveListLink<QueueEntry> link;

  uint8_t task;

  uint32_t stepValue;
};

struct QueueEntryAllGather : QueueEntry {
  std::vector<uintptr_t> outputAddresses;
  uintptr_t inputAddress = 0;
  uintptr_t outputAddress = 0;
  size_t bytes = 0;

  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;
  std::vector<uintptr_t> proxyIpcOutputAddresses;

  std::vector<uintptr_t> proxyIpcOutputAddresses2;

  std::atomic_size_t inputBytesReady = 0;

  uintptr_t cpuInput = 0;
  uintptr_t cpuOutput = 0;

  // std::atomic_uint32_t cpuOutStepValue = 0;
  // std::atomic_uint32_t cpuInStepValue = 0;
};

struct QueueEntryAllReduce : QueueEntry {
  uintptr_t inputAddress = 0;
  size_t bytes = 0;

  std::array<uintptr_t, 8> peerInputAddresses;
  std::vector<uintptr_t> proxyIpcOutputAddresses;

  AllocatedBuffer recvBuffers;
};

template<typename T>
struct QueueEntryFreeList {
  SpinMutex mutex;
  std::forward_list<T> all;
  IntrusiveList<QueueEntry, &QueueEntry::link> free;
  size_t nAllocated = 0;
  T* pop() {
    std::lock_guard l(mutex);
    if (!free.empty()) {
      T* r = (T*)&free.back();
      free.pop_back();
      return r;
    }
    ++nAllocated;
    all.emplace_front();
    return &all.front();
  }
  void push(T* v) {
    std::lock_guard l(mutex);
    free.push_back((QueueEntry&)*v);
  }
  ~QueueEntryFreeList() {}
};

struct CpuThread {
  Group* group;

  std::thread thread;

  SpinMutex mutex;
  IntrusiveList<QueueEntry, &QueueEntry::link> queue;
  std::atomic_uint32_t queueSize = 0;
  std::atomic_bool terminate = false;

  QueueEntryFreeList<QueueEntry> freelistTerminate;
  QueueEntryFreeList<QueueEntryAllGather> freelistAllGather;
  QueueEntryFreeList<QueueEntryAllReduce> freelistAllReduce;

  CpuThread(Group*);
  ~CpuThread();

  void start();
  void entry();
  void enqueue(QueueEntry* e) {
    std::unique_lock l(mutex);
    queue.push_back(*e);
    l.unlock();
    ++queueSize;
    futexWakeAll(&queueSize);
  }
};

} // namespace moodist
