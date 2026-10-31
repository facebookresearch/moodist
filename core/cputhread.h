// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "api/moodist_api.h"
#include "compile_op.h"
#include "compile_op_kernel.h"
#include "cuda_loader.h"
#include "group.h"
#include "intrusive_list.h"
#include "ptx_codegen.h"
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
constexpr uint8_t taskFabricMap = 26;
constexpr uint8_t taskRdmaRegister = 27;
constexpr uint8_t taskRdmaRead = 28;

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
OPTYPE(CompileOpLocal);

OPTYPE(MessageShutdown);

OPTYPE(FabricMap);
OPTYPE(FabricMapDone);

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
  IVector<int64_t> tensorShape;
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

struct QueueEntryRdmaRegister : QueueEntry {
  uintptr_t address;
  size_t bytes;
  bool isCuda;
  uint32_t* outRkeys; // points to caller's rkey array [maxDevices]
  std::atomic_uint32_t* cpuDone;
};

struct QueueEntryRdmaRead : QueueEntry {
  uintptr_t dstAddress;
  size_t dstBytes;
  bool dstIsCuda;
  uint32_t srcRank;
  uintptr_t srcAddress;
  std::array<uint32_t, 4> srcRkeys;
  std::atomic_uint32_t* done;
};

struct CustomOpDescriptor {
  uint32_t id;

  struct Copy {
    uint32_t index;
    Vector<int64_t> offset;
    Vector<int64_t> shape;
  };

  struct IbRead {
    uint32_t rank;
    uint32_t inputIndex;
    uint32_t outputIndex;
    size_t inputOffset;
    size_t outputOffset;
    size_t bytes;
    // >= 0: once the read has completed, set signal signalIndex (see writeRdmaSignal in cputhread.cc), which the
    // kernel waits for (the proxy read of a mixed NVLink + RDMA op, a source of a reduction across NVLink domains)
    int32_t signalIndex = -1;
    // >= 0: read only once the source rank has notified this rank that its kernel has written the data (a partial
    // result of a reduction, see Notification), in notification slot waitSlot
    int32_t waitSlot = -1;

    template<typename X>
    void serialize(X& x) {
      x(rank, inputIndex, outputIndex, inputOffset, outputOffset, bytes, signalIndex, waitSlot);
    }
  };

  // A partial result of a reduction that this rank's kernel computes for a rank in another NVLink domain (see
  // crossDomainReductions in compile_op.cc): once the kernel has written the step value into cpuInBuffer slot
  // 16 + flag, and into those of all other notifications for the same rank and slot, the CPU thread notifies that
  // rank (slot `slot` of its notifications from this rank), which then reads the data
  struct Notification {
    uint32_t flag;
    uint32_t rank;
    uint32_t slot;
  };
  // Notification slots per pair of ranks (Group::customNotifications)
  static constexpr size_t numNotificationSlots = Group::numNotificationSlots;

  // With writes instead of reads (MOODIST_COMPILE_OP_TRANSFER=write, see crossDomainReductions in compile_op.cc): a
  // range of a partial result (or of a share of a result) that this rank's kernel computes for a rank in another
  // NVLink domain, which this rank writes into that rank's scratch tensor. Once the kernel has written the step value
  // into cpuInBuffer slot 16 + flag, the CPU thread writes `bytes` bytes from offset sourceOffset of its own scratch
  // tensor to offset destinationOffset of tensor destinationIndex of `rank` (in the list of tensors it publishes, see
  // WorkCustom), then sets signal signalIndex of `rank` (see writeRdmaSignal in cputhread.cc), which its kernel waits
  // for
  struct IbWrite {
    uint32_t flag;
    uint32_t rank;
    uint32_t destinationIndex;
    size_t sourceOffset;
    size_t destinationOffset;
    size_t bytes;
    int32_t signalIndex;
  };

  Vector<Copy> inputCopies;
  Vector<Copy> outputCopies;

  // Vector<size_t> inputs;
  // Vector<size_t> outputs;

  Vector<IbRead> ibReads;
  // How many of its reads without a signal index a rank reads at a time (WorkCustom): 1 if every rank reads the
  // same amount from every rank outside its NVLink domain (each source then has one reader at a time), 8 otherwise
  // (with unequal amounts, reading one source at a time leaves some sources idle; see compile_op.cc)
  uint32_t readSources = 1;
  Vector<Notification> notifications;
  Vector<IbWrite> ibWrites;
  // The ranks this rank exchanges with over RDMA in the op (its peers): those it reads from, writes to or notifies,
  // and those that read from it, write to it or notify it, in the order rank + 1, rank + 2, ... (mod the group size);
  // isPeer per rank. Only peers exchange tensor lists and step values and wait for each other's reads and writes; the
  // other ranks exchange arrival values (Group::cpuArrivalValues, see WorkCustom). dynSelf: the rank reads from its own
  // tensors, and so publishes its list to itself.
  Vector<uint32_t> peers;
  Vector<uint8_t> isPeer;
  bool dynSelf = false;
  // Reductions across NVLink domains or in two phases: the bytes of the scratch tensor (a CUDA tensor of the op's
  // dtype that customOp allocates per launch, after the output copies: the last tensor), which holds the partial
  // results that this rank computes (for other domains, and in an all-reduce for itself) and the ones that it reads
  // from them (or that they write into it), and the pieces that it computes in two phases before it copies them to
  // their destinations (see twoPhaseReductions in compile_op.cc); 0 if none
  size_t scratchBytes = 0;

  Vector<compile_op::Graph::TensorDescr> inputs;
  Vector<compile_op::Graph::TensorDescr> outputs;

  // Vector<Vector<int64_t>> inputShapes;
  // Vector<Vector<int64_t>> outputShapes;
  // Vector<DeviceType> inputDevices;
  // Vector<DeviceType> outputDevices;
  DType dtype;
  bool cpuSync = false;  // Force CPU-side wait in customOp before CUDA operations
  bool allLocal = false; // All ranks local + all CUDA: skip CPU thread entirely
  bool anyRankNeedsRdma = false;
  // CUDA tensors must come from the moodist CUDA allocator (checked in customOp). Off for process group
  // collectives built on compile_op (scatter, gather), which take tensors from any allocator.
  bool requireMoodistMemory = true;

  compile_op::Graph graph;
  std::shared_ptr<KernelHandle> kernelHandle;
  std::unique_ptr<CompiledKernel> kernel;
  KernelConfig config;
};

struct QueueEntryCustom : QueueEntry {
  std::shared_ptr<CustomOpDescriptor> op;
  FutureImplSharedPtr future;
  bool anyCuda;
  bool anyCpu;
  // While profiling: when and on which core the calling thread handed the op to the CPU thread (for the trace)
  std::chrono::system_clock::time_point enqueueTime{};
  int enqueueCpu = -1;

  std::vector<TensorDataPtr> inputs;
  std::vector<TensorDataPtr> outputs;
};

struct QueueEntryFabricMap : QueueEntry {
  uint32_t targetRank;
  size_t chunkIndex;
  CUmemFabricHandle handle;
  size_t offset;
  size_t bytes;
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

  Function<uintptr_t(uint32_t, QueueEntryFabricMap)> onFabricMap;
  Function<void(uint32_t, size_t, uintptr_t)> onFabricMapDone;

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
  QueueEntryFreeList<QueueEntryFabricMap> freelistFabricMap;
  QueueEntryFreeList<QueueEntryRdmaRegister> freelistRdmaRegister;
  QueueEntryFreeList<QueueEntryRdmaRead> freelistRdmaRead;

  std::exception_ptr initExceptionPtr;
  std::atomic_bool initException = false;

  std::atomic_bool killed = false;

  CpuThread(Group*);
  ~CpuThread();
  void kill();

  void start(Function<void()> pghandle);
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
