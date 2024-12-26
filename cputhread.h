#pragma once

#include "group.h"
#include "intrusive_list.h"
#include "parameters_data.h"
#include "simple_vector.h"
#include "synchronization.h"
#include "vector.h"

#include <forward_list>
#include <memory>

namespace moodist {

constexpr uint8_t taskTerminate = 0;
constexpr uint8_t taskBarrier = 1;
constexpr uint8_t taskAllGather = 2;
constexpr uint8_t taskReduceScatter = 3;
constexpr uint8_t taskBroadcast = 4;
constexpr uint8_t taskAllGatherCpu = 5;
constexpr uint8_t taskReduceScatterCpu = 6;
constexpr uint8_t taskGatherCpu = 7;
constexpr uint8_t taskBroadcastCpu = 8;
constexpr uint8_t taskInternalBarrier = 9;
constexpr uint8_t taskReduce = 10;

inline HashMap<uint32_t, const char*> opTypeToName;
#define OPTYPE(name)                                                                                                   \
  constexpr uint32_t opType##name = __LINE__;                                                                          \
  inline struct ctor##name {                                                                                           \
    ctor##name() {                                                                                                     \
      opTypeToName[opType##name] = #name;                                                                              \
    }                                                                                                                  \
  } instance##name;

inline std::string getOpTypeName(uint32_t value) {
  auto i = opTypeToName.find(value);
  if (i != opTypeToName.end()) {
    return i->second;
  }
  return "unknown value " + std::to_string(value);
}
OPTYPE(Barrier);
OPTYPE(InternalBarrier);
OPTYPE(AllGatherCpu);
OPTYPE(ReduceScatterCpu);
OPTYPE(BroadcastRingCpu);
OPTYPE(GatherCpu);

OPTYPE(AllGatherRingCuda);
OPTYPE(ReduceScatterRingCuda);
OPTYPE(BroadcastCuda);
OPTYPE(ReduceCuda);

OPTYPE(BroadcastRingCuda);
OPTYPE(ReduceTreeCuda);

template<typename DynamicAddresses>
inline void badOp(
    const char* filename, uint32_t line, const DynamicAddresses& dyn, uint32_t opType, uint32_t stepValue,
    size_t bytes) {
  fatal(
      "Mismatched collectives detected at %s:%d.\nLocal parameters: op %s, step %#x, bytes %#x\nRemote parameters: op "
      "%s, step "
      "%#x, bytes %#x\n",
      filename, line, getOpTypeName(opType), stepValue, bytes, getOpTypeName(dyn.opType), dyn.stepValue,
      dyn.gatherBytes);
}
#define CHECK_DYN(dyn, optype, stepvalue, bytes)                                                                       \
  if (dyn.opType != (optype) | dyn.stepValue != (stepvalue) | dyn.gatherBytes != (bytes)) [[unlikely]]                 \
    badOp(__FILE__, __LINE__, dyn, optype, stepvalue, bytes);

struct DataRef {
  uintptr_t address = 0;
  size_t bytes = 0;
};

struct QueueEntry {
  IntrusiveListLink<QueueEntry> link;
  uint8_t task;
  uint32_t stepValue;
  uint32_t concurrencyIndex = -1;
  StreamData* sd = nullptr;
};

struct QueueEntryBarrier : QueueEntry {
  std::atomic_uint32_t* cpuDone = nullptr;
};

struct QueueEntryAllGather : QueueEntry {
  uintptr_t inputAddress = 0;
  uintptr_t outputAddress = 0;
  size_t bytes = 0;
  size_t pitch = 0;

  size_t numDevices = 0;
  size_t numChunks = 0;
  size_t numParallel = 0;
};

struct QueueEntryReduceScatter : QueueEntry {
  uintptr_t inputAddress = 0;
  uintptr_t outputAddress = 0;
  size_t bytes = 0;
  size_t pitch = 0;

  size_t numDevices = 0;
  size_t numChunks = 0;
  size_t numParallel = 0;
};

struct QueueEntryBroadcast : QueueEntry {
  uintptr_t tensorAddress = 0;
  size_t bytes = 0;
  size_t sourceRank = 0;

  size_t numDevices = 0;
  size_t numChunks = 0;
  size_t numParallel = 0;
};

struct QueueEntryReduce : QueueEntry {
  uintptr_t tensorAddress = 0;
  size_t bytes = 0;
  size_t destinationRank = 0;

  uintptr_t recvBuffer = 0;

  size_t numDevices = 0;
  size_t numChunks = 0;
  size_t numParallel = 0;
};

struct QueueEntryAllGatherCpu : QueueEntry {
  uintptr_t inputAddress = 0;
  uintptr_t outputAddress = 0;
  size_t bytes = 0;
  size_t pitch = 0;
  std::atomic_uint32_t* cpuDone = nullptr;
};

struct QueueEntryReduceScatterCpu : QueueEntry {
  uintptr_t inputAddress = 0;
  uintptr_t outputAddress = 0;
  size_t bytes = 0;
  size_t pitch = 0;
  std::atomic_uint32_t* cpuDone = nullptr;
  Dtype dindex;
  Reduction opindex;
};

struct QueueEntryGatherCpu : QueueEntry {
  uintptr_t inputAddress = 0;
  SimpleVector<DataRef> outputList;
  size_t bytes = 0;
  uint32_t destinationRank = -1;
  std::atomic_uint32_t* cpuDone = nullptr;
};

struct QueueEntryBroadcastCpu : QueueEntry {
  uintptr_t tensorAddress = 0;
  size_t bytes = 0;
  size_t sourceRank = 0;
  std::atomic_uint32_t* cpuDone = nullptr;
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
    log.debug("QueueEntryFreeList %s allocated -> %d\n", typeid(T).name(), nAllocated);
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

  std::atomic_bool ready = false;
  std::atomic_bool busy = false;

  QueueEntryFreeList<QueueEntry> freelistTerminate;
  QueueEntryFreeList<QueueEntryBarrier> freelistBarrier;
  QueueEntryFreeList<QueueEntryAllGather> freelistAllGather;
  QueueEntryFreeList<QueueEntryReduceScatter> freelistReduceScatter;
  QueueEntryFreeList<QueueEntryBroadcast> freelistBroadcast;
  QueueEntryFreeList<QueueEntryAllGatherCpu> freelistAllGatherCpu;
  QueueEntryFreeList<QueueEntryReduceScatterCpu> freelistReduceScatterCpu;
  QueueEntryFreeList<QueueEntryGatherCpu> freelistGatherCpu;
  QueueEntryFreeList<QueueEntryBroadcastCpu> freelistBroadcastCpu;
  QueueEntryFreeList<QueueEntryReduce> freelistReduce;

  CpuThread(Group*);
  ~CpuThread();

  void start();
  void enqueue(QueueEntry* e) {
    std::unique_lock l(mutex);
    busy.store(true, std::memory_order_relaxed);
    queue.push_back(*e);
    l.unlock();
    ++queueSize;
    futexWakeAll(&queueSize);
  }
};

} // namespace moodist
