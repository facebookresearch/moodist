// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "group.h"
#include "intrusive_list.h"
#include "moodist_api.h"
#include "simple_vector.h"
#include "synchronization.h"
#include "tensor_types.h"

#include <exception>
#include <forward_list>

namespace moodist {

constexpr uint8_t taskShutdown = 0;
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
constexpr uint8_t taskCreateQueue = 11;
constexpr uint8_t taskQueuePut = 12;
constexpr uint8_t taskQueueRead = 13;
constexpr uint8_t taskQueueReadFinished = 14;
constexpr uint8_t taskQueueGet = 15;
constexpr uint8_t taskQueueTransaction = 16;
constexpr uint8_t taskCat = 17;
constexpr uint8_t taskCopy = 18;
constexpr uint8_t taskCached = 19;
constexpr uint8_t taskReduceScatterDirect = 20;
constexpr uint8_t taskAllGatherBroadcast = 21;
constexpr uint8_t taskCallback = 22;
constexpr uint8_t taskAllGatherDirect = 23;
constexpr uint8_t taskCreateQueueNamed = 24;
constexpr uint8_t taskCustom = 25;

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

OPTYPE(AllGatherKernelLessCuda);
OPTYPE(AllGatherLocalCuda);
OPTYPE(AllGatherRingCuda);
OPTYPE(AllGatherBroadcastCuda);
OPTYPE(AllGatherDirectCuda);
OPTYPE(ReduceScatterKernelLessCuda);
OPTYPE(ReduceScatterRingCuda);
OPTYPE(ReduceScatterDirectCuda);
OPTYPE(BroadcastCuda);
OPTYPE(ReduceCuda);
OPTYPE(AllToAllCuda);
OPTYPE(AllToAllCuda2);
OPTYPE(LocalSmallAllReduceCuda);
OPTYPE(ShareCuda);
OPTYPE(BarrierCuda);

OPTYPE(BroadcastRingCuda);
OPTYPE(ReduceTreeCuda);

OPTYPE(CreateQueue);
OPTYPE(QueuePut);
OPTYPE(QueueReadFinished);
OPTYPE(QueueGet);
OPTYPE(QueueTransaction);
OPTYPE(CreateQueueNamed);
OPTYPE(CreateQueueNamedResult);

OPTYPE(Cat);
OPTYPE(Cached);
OPTYPE(Custom);

OPTYPE(MessageShutdown);

template<typename DynamicAddresses>
inline void badOp(const char* filename, uint32_t line, const DynamicAddresses& dyn, uint32_t opType, uint32_t stepValue,
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
  uint32_t stepValue = 0;
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

  uintptr_t recvAddress = 0;
  uintptr_t sendAddress = 0;

  bool isCopy = false;
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

struct QueueEntryCreateQueue : QueueEntry {
  std::atomic_uintptr_t* outAddress = nullptr;
  std::string* outError = nullptr;
  std::atomic_uint32_t* cpuDone = nullptr;

  size_t location = 0;
  uintptr_t address = 0;
  size_t bytes = 0;
  bool streaming = false;
  std::optional<std::string> name;
};

struct QueueEntryQueuePut : QueueEntry {
  size_t location = 0;
  uintptr_t remoteAddress = 0;
  Function<void(uintptr_t*, uint32_t*)> callback;
  uint32_t putKey = 0;
  uint32_t remoteGetKey = 0;
  uint32_t transactionKey = 0;
  size_t queueSize = -1;
  TensorData* tensor = nullptr;
  bool streaming = false;
};

struct QueueEntryQueueGet : QueueEntry {
  enum { opNone, opStart, opStop, opStopAck };
  size_t location = 0;
  uintptr_t remoteQueueAddress = 0;
  uintptr_t localQueueAddress = 0;
  uint32_t key = 0;
  uint8_t op = 0;
};

struct QueueEntryQueueRead : QueueEntry {
  size_t source = 0;
  uintptr_t queueAddress = 0;
  uintptr_t remoteAddress = 0;
  size_t bytes = 0;
  std::array<uint32_t, maxDevices> rkeys;
  uint32_t remoteOpKey = 0;
  uint32_t localOpKey = 0;
  int tensorDtype = -1;
  uint32_t getKey = 0;
  uint32_t transactionKey = 0;
  size_t queueSize = -1;
  std::vector<int64_t> tensorShape;
};

struct QueueEntryQueueReadFinished : QueueEntry {
  uint32_t key = 0;
  uint32_t cudaDoneValue;
};

struct QueueEntryQueueTransaction : QueueEntry {
  size_t location = 0;
  uintptr_t remoteAddress = 0;
  uint32_t transactionKey = 0;
  uint8_t op;
};

struct QueueEntryCat : QueueEntry {
  std::vector<std::pair<int, TensorDataPtr>> locals;
  FutureImplSharedPtr future;
  TensorDataPtr out;
};

struct QueueEntryCopy : QueueEntry {
  TensorDataPtr destination;
  TensorDataPtr source;
  FutureImplSharedPtr future;
};

struct QueueEntryCached : QueueEntry {
  int op = -1;
  int identifier = -1;
  size_t inputBytes;
  size_t outputBytes;
};

struct QueueEntryCallback : QueueEntry {
  Function<void()> callback;
};

struct CustomOpDescriptor {
  uint32_t id;
  struct Read {
    uint32_t rank;
    uint32_t inputIndex;
    uint32_t outputIndex;
    size_t inputOffset;
    size_t outputOffset;
    size_t bytes;
  };
  struct Copy {
    uint32_t index;
    IVector<int64_t> offset;
    IVector<int64_t> shape;
  };

  IVector<Read> reads;
  IVector<Copy> inputCopies;
  IVector<Copy> outputCopies;

  IVector<size_t> inputs;
  IVector<size_t> outputs;

  IVector<IVector<int64_t>> inputShapes;
  IVector<IVector<int64_t>> outputShapes;
  DType dtype;
};

struct QueueEntryCustom : QueueEntry {
  std::shared_ptr<CustomOpDescriptor> op;
  FutureImplSharedPtr future;
  bool anyCuda;
  bool anyCpu;

  std::vector<TensorDataPtr> inputs;
  std::vector<TensorDataPtr> outputs;
};

template<typename T>
struct QueueEntryFreeList {
  SpinMutex mutex;
  std::forward_list<T, InternalAllocator<T>> all;
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

  QueueEntryFreeList<QueueEntry> freelistShutdown;
  QueueEntryFreeList<QueueEntryBarrier> freelistBarrier;
  QueueEntryFreeList<QueueEntryAllGather> freelistAllGather;
  QueueEntryFreeList<QueueEntryReduceScatter> freelistReduceScatter;
  QueueEntryFreeList<QueueEntryBroadcast> freelistBroadcast;
  QueueEntryFreeList<QueueEntryAllGatherCpu> freelistAllGatherCpu;
  QueueEntryFreeList<QueueEntryReduceScatterCpu> freelistReduceScatterCpu;
  QueueEntryFreeList<QueueEntryGatherCpu> freelistGatherCpu;
  QueueEntryFreeList<QueueEntryBroadcastCpu> freelistBroadcastCpu;
  QueueEntryFreeList<QueueEntryReduce> freelistReduce;
  QueueEntryFreeList<QueueEntryCreateQueue> freelistCreateQueue;
  QueueEntryFreeList<QueueEntryQueuePut> freelistQueuePut;
  QueueEntryFreeList<QueueEntryQueueRead> freelistQueueRead;
  QueueEntryFreeList<QueueEntryQueueReadFinished> freelistQueueReadFinished;
  QueueEntryFreeList<QueueEntryQueueGet> freelistQueueGet;
  QueueEntryFreeList<QueueEntryQueueTransaction> freelistQueueTransaction;
  QueueEntryFreeList<QueueEntryCat> freelistCat;
  QueueEntryFreeList<QueueEntryCopy> freelistCopy;
  QueueEntryFreeList<QueueEntryCached> freelistCached;
  QueueEntryFreeList<QueueEntryCallback> freelistCallback;
  QueueEntryFreeList<QueueEntryCustom> freelistCustom;

  std::exception_ptr initExceptionPtr;
  std::atomic_bool initException = false;

  CpuThread(Group*);
  ~CpuThread();
  void kill(bool wait = true);

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
