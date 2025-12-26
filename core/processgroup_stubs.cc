// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup implementation for the API layer.
// This file implements ProcessGroupImpl and the API functions.
// Collective operations are stubs until migration is complete.

#include "allgather.h"
#include "common.h"
#include "cputhread.h"
#include "group.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "moodist_api.h"
#include "processgroup_api.h"
#include "queue.h"
#include "reduce_scatter.h"
#include "serialization.h"
#include "setup_comms.h"
#include "shared_ptr.h"
#include "synchronization.h"
#include "tensor_ptr.h"

#include <atomic>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <mutex>

namespace moodist {

// ============================================================================
// Helper functions for c10d::Store operations via wrapperApi
// ============================================================================

namespace c10dstore {

inline void set(void* store, std::string_view key, std::string_view value) {
  std::vector<uint8_t> valueVec(value.begin(), value.end());
  wrapperApi.c10dStoreSet(store, key, valueVec);
}

inline std::string get(void* store, std::string_view key) {
  std::vector<uint8_t> value = wrapperApi.c10dStoreGet(store, key);
  return std::string(value.begin(), value.end());
}

inline void wait(void* store, const std::vector<std::string>& keys) {
  wrapperApi.c10dStoreWait(store, keys);
}

} // namespace c10dstore

// ============================================================================
// Global state
// ============================================================================

struct ProcessGroupImpl;
std::mutex& activeProcessGroupsMutex = Global();
HashMap<uint32_t, ProcessGroupImpl*>& activeProcessGroups = Global();
uint32_t nextProcessGroupActiveId = 1;

std::atomic_bool globalPreferKernelLess = false;

std::once_flag freeMemoryCallbackOnceFlag;

SharedSpinMutex unmapMemoryMutex;

bool profilingEnabled = false;

// ============================================================================
// Helper types
// ============================================================================

struct EventSerializer {
  Event event;
  CUstream stream;
  EventSerializer(CUevent event, CUstream stream) : event(Event::reference(event)), stream(stream) {
    this->event.wait(stream);
  }
  ~EventSerializer() {
    event.record(stream);
  }
};

// Work objects can outlive ProcessGroupImpl and hold a pointer to WorkStream.
// A proper solution would be to hold a SharedPtr to WorkStreams in Work.
// Instead, we intentionally leak WorkStreams objects through a circular SharedPtr.
struct WorkStreams;
struct WorkStream {
  IntrusiveListLink<WorkStream> link;
  CUstream stream = nullptr;
  CUevent event = nullptr;
  std::atomic<CUstream> joinedStream = nullptr;
  SharedPtr<WorkStreams> owner;
  WorkStream() = default;
  WorkStream(const WorkStream&) = delete;
  WorkStream& operator=(const WorkStream&) = delete;
  WorkStream(WorkStream&& n) noexcept {
    *this = std::move(n);
  }
  WorkStream& operator=(WorkStream&& n) noexcept {
    std::swap(stream, n.stream);
    std::swap(event, n.event);
    std::swap(owner, n.owner);
    // Note: link and joinedStream intentionally not swapped
    return *this;
  }
  ~WorkStream() {
    if (event) {
      cuEventDestroy(event);
    }
    if (stream) {
      cuStreamDestroy(stream);
    }
  }
};

struct WorkStreams {
  SpinMutex mutex;
  Vector<std::unique_ptr<WorkStream>> all;
  IntrusiveList<WorkStream, &WorkStream::link> free;
  std::atomic<int> refcount; // For SharedPtr
};

struct ThreadUnsafe {
  SpinMutex mutex;
  void lock() {
    if (!mutex.try_lock()) {
      throw std::runtime_error("Concurrent use of process group detected - this is not supported. Moodist process "
                               "groups are not thread-safe.");
    }
  }
  void unlock() {
    mutex.unlock();
  }
};

struct IpcEvent : Event {
  IpcEvent() {
    (Event&)* this = Event::createInterprocess();
  }
};

template<typename T>
struct ReusableHandle;
template<typename T>
struct Reusable {
  Vector<T> free;
  ReusableHandle<T> pop() {
    if (free.empty()) [[unlikely]] {
      return NOINLINE_COLD({
        log.debug("Reusable ctor %s\n", typeid(T).name());
        return ReusableHandle<T>(this, T());
      });
    }
    return ReusableHandle<T>(this, free.pop_back_value());
  }
  ReusableHandle<T> pop(Reusable* returnContainer) {
    if (free.empty()) [[unlikely]] {
      return NOINLINE_COLD({
        log.debug("Reusable ctor %s\n", typeid(T).name());
        return ReusableHandle<T>(returnContainer, T());
      });
    }
    return ReusableHandle<T>(returnContainer, free.pop_back_value());
  }
  void push(T&& v) {
    free.push_back(std::move(v));
  }
};
template<typename T>
struct ReusableHandle {
  Reusable<T>* container = nullptr;
  std::optional<T> value;
  ReusableHandle() = default;
  ReusableHandle(Reusable<T>* container, T&& value) : container(container), value(std::move(value)) {}
  ~ReusableHandle() {
    if (container) {
      container->push(std::move(*value));
    }
  }
  ReusableHandle(ReusableHandle&) = delete;
  ReusableHandle(ReusableHandle&& n) noexcept {
    *this = std::move(n);
  }
  ReusableHandle& operator=(ReusableHandle& n) = delete;
  ReusableHandle& operator=(ReusableHandle&& n) noexcept {
    std::swap(container, n.container);
    std::swap(value, n.value);
    return *this;
  }
  T* operator->() {
    return &*value;
  }
  T& operator*() {
    return *value;
  }
  const T* operator->() const {
    return &*value;
  }
  const T& operator*() const {
    return *value;
  }
  operator T&() {
    return *value;
  }
  operator const T&() const {
    return *value;
  }
};

// ============================================================================
// Type converters from API types to internal types
// ============================================================================

// Convert API DType to internal Dtype (for kernel dispatch)
Dtype toInternalDtype(DType dt) {
  switch (dt) {
  case DType::Float32:
    return Dtype::float32;
  case DType::Float64:
    return Dtype::float64;
  case DType::Int32:
    return Dtype::int32;
  case DType::Int64:
    return Dtype::int64;
  case DType::BFloat16:
    return Dtype::bfloat16;
  default:
    throw std::runtime_error("Unsupported dtype for reduce operation");
  }
}

// Convert API ReduceOp to internal Reduction (for kernel dispatch)
Reduction toInternalReduction(ReduceOp op) {
  switch (op) {
  case ReduceOp::SUM:
  case ReduceOp::PREMUL_SUM:
    return Reduction::sum;
  case ReduceOp::AVG:
    return Reduction::avg;
  case ReduceOp::MIN:
    return Reduction::min;
  case ReduceOp::MAX:
    return Reduction::max;
  default:
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// ============================================================================
// ProcessGroupImpl
// ============================================================================

struct ProcessGroupImpl {
  size_t rank = 0;
  size_t size = 0;

  SharedPtr<Group> group;

  ThreadUnsafe threadUnsafe;
  uint32_t nextStepValue = 0x1000;
  uint32_t nextConcurrencyIndex = 0;

  HashMap<CUstream, SharedPtr<WorkStreams>> workStreams;

  std::array<Event, maxConcurrency> concurrencyEvents;

  Event localEvent;
  std::optional<Stream> peerMemcpyStream;
  std::optional<Stream> peerIncomingMemcpyStream;

  HashMap<CUstream, Stream> copyStream;

  std::array<std::optional<Stream>, maxChunks> peerMemcpyStreamPerChunk;

  std::array<std::optional<Stream>, 4> reduceStreamArr;
  size_t reduceCounter = 0;

  std::vector<SharedPtr<Queue>> queues;

  bool preferKernelLess = globalPreferKernelLess;

  Reusable<IpcEvent> ipcEvents;
  Reusable<IpcEvent> pendingIpcEvents;

  int streamPriority = -1;

  // Memory operations batching for CUDA stream memory ops
  std::vector<CUstreamBatchMemOpParams> memops;

  uint32_t activeId = 0;

  void* c10dStore_ = nullptr; // Opaque pointer to c10d::Store for init

  std::atomic<int> refcount; // Reference counting - initialized by SharedPtrDefaultPolicy::create

  ProcessGroupImpl(void* c10dStore, int rank_, int size_) : rank(rank_), size(size_), c10dStore_(c10dStore) {
    CHECK(rank_ >= 0 && size_ > 0 && rank_ < size_);

    log.init();

    group = makeShared<Group>(rank, size);

    init();

    for (auto& v : concurrencyEvents) {
      v = Event::create();
    }

    localEvent = Event::create();

    if (size - 1 == group->peerIndices.size()) {
      streamPriority = -10;
    }

    {
      std::call_once(freeMemoryCallbackOnceFlag, []() {
        wrapperApi.registerFreeMemoryCallback();
      });
      std::lock_guard l(activeProcessGroupsMutex);
      activeId = nextProcessGroupActiveId++;
      CHECK(!activeProcessGroups.contains(activeId));
      activeProcessGroups[activeId] = this;
    }
  }

  ~ProcessGroupImpl() {
    std::lock_guard l(activeProcessGroupsMutex);
    activeProcessGroups.erase(activeId);
  }

  void init();

  std::vector<std::string> decodeAddress(std::string str) {
    std::vector<uint8_t> data;
    size_t i = 0;
    for (char vv : str) {
      size_t index = vv >= '0' && vv <= '9' ? vv - '0' : vv - 'a' + 10;
      if (index >= 16) {
        throw std::invalid_argument("ProcessGroup: invalid address");
      }
      if (i % 2 == 0) {
        data.push_back(index);
      } else {
        data.back() <<= 4;
        data.back() |= index;
      }
      ++i;
    }
    std::vector<std::string> remoteAddresses;
    deserializeBuffer(data.data(), data.size(), remoteAddresses);
    return remoteAddresses;
  }

  std::string getAddress() {
    std::vector<std::string> addresses = group->setupComms->listenerAddresses();
    auto buffer = serializeToBuffer(addresses);

    std::string r;
    for (size_t i = 0; i != buffer->size(); ++i) {
      uint8_t v = (uint8_t)buffer->data()[i];
      r += "0123456789abcdef"[v >> 4];
      r += "0123456789abcdef"[v & 0xf];
    }
    return r;
  }

  Stream getCopyStream(CUstream stream) {
    auto& ref = copyStream[stream];
    if (!ref) {
      ref = Stream::create(streamPriority);
    }
    return Stream::reference(ref);
  }

  WorkStreams* getWorkStreams(CUstream stream) {
    auto& ptr = workStreams[stream];
    if (!ptr) {
      ptr = makeShared<WorkStreams>();
    }
    return ptr.get();
  }

  WorkStream* getWorkStream(CUstream stream) {
    WorkStreams* ws = getWorkStreams(stream);
    std::unique_lock l(ws->mutex);
    WorkStream* w = nullptr;
    int n = 0;
    for (auto& v : ws->free) {
      if (v.joinedStream.load(std::memory_order_relaxed) == stream) {
        w = &v;
        ws->free.erase(v);
        break;
      }
      if (cuEventQuery(v.event) == CUDA_SUCCESS) {
        w = &v;
        ws->free.erase(v);
        break;
      }
      ++n;
    }
    if (!w && n >= static_cast<int>(maxConcurrency)) {
      w = &ws->free.front();
      ws->free.pop_front();
    }
    if (!w) {
      auto ptr = std::make_unique<WorkStream>();
      w = &*ptr;
      ws->all.push_back(std::move(ptr));

      w->owner = workStreams[stream];
      CHECK(&*w->owner == ws);
      CHECK_CU(cuStreamCreateWithPriority(&w->stream, CU_STREAM_NON_BLOCKING, streamPriority));
      CHECK_CU(cuEventCreate(&w->event, CU_EVENT_DISABLE_TIMING));
      log.info("New work stream %#x created for stream %#x\n", (uintptr_t)w->stream, (uintptr_t)stream);
    }
    w->joinedStream.store(w->stream, std::memory_order_relaxed);
    l.unlock();
    return w;
  }

  uint32_t getNextStepValue() {
    uint32_t r = std::exchange(nextStepValue, nextStepValue + 0x1000);
    if (r < 0x80000000) {
      return r;
    }
    // Reset needed - for now just return 1
    nextStepValue = 0x1000;
    return 1;
  }

  AllocatedBuffer alloccuda(size_t size, CUstream stream) {
    if (size == 0) {
      return {};
    }
    if (size < 16) {
      size = 16;
    }
    StreamGuard sg(stream, group->deviceIndex);
    AllocatedBuffer buffer;
    void* cleanupCtx = nullptr;
    void* ptr = wrapperApi.cudaCachingAllocatorAlloc(size, &cleanupCtx);
    buffer.cudaPointer = (uintptr_t)ptr;
    buffer.bytes = size;
    // Capture cleanupCtx in the cleanup function
    buffer.externalAlloc = DeferredCleanup(Function<void()>([cleanupCtx]() {
      wrapperApi.cudaCachingAllocatorFree(cleanupCtx);
    }));
    CHECK(buffer.cudaPointer != 0);
    return buffer;
  }

  void sync(uint32_t stepValue) {
    IpcMapper* ipcMapper = &*group->ipcMapper;
    const auto& peerIndices = group->peerIndices;

    if (!peerIndices.empty()) {
      ipcMapper->wait();
    }
    ipcMapper->setStepValue(stepValue);
  }

  ReusableHandle<IpcEvent> getIpcEvent() {
    return ipcEvents.pop(&pendingIpcEvents);
  }

  void freePendingIpcEvents() {
    for (auto& v : pendingIpcEvents.free) {
      ipcEvents.free.push_back(std::move(v));
    }
    pendingIpcEvents.free.clear();
  }

  void syncPeers(CUstream stream) {
    IpcMapper* ipcMapper = &*group->ipcMapper;
    const auto& peerIndices = group->peerIndices;
    if (peerIndices.empty()) {
      return;
    }
    auto event = getIpcEvent();
    event->record(stream);
    for (size_t peerIndex : peerIndices) {
      ipcMapper->pushEvent(peerIndex, *event);
    }
    for (size_t peerIndex : peerIndices) {
      Event::reference(ipcMapper->popEvent(peerIndex)).wait(stream);
    }
  }

  void memWaitEq(uintptr_t address, uint32_t value) {
    CUstreamBatchMemOpParams op;
    std::memset(&op, 0, sizeof(op));
    op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.address = address;
    op.waitValue.value = value;
    memops.push_back(op);
  }

  void memWaitGeq(uintptr_t address, uint32_t value) {
    CUstreamBatchMemOpParams op;
    std::memset(&op, 0, sizeof(op));
    op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    op.waitValue.address = address;
    op.waitValue.value = value;
    memops.push_back(op);
  }

  void memWrite(uintptr_t address, uint32_t value) {
    CUstreamBatchMemOpParams op;
    std::memset(&op, 0, sizeof(op));
    op.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    op.writeValue.address = address;
    op.writeValue.value = value;
    memops.push_back(op);
  }

  void memFlush(CUstream stream) {
    if (!memops.empty()) {
      CHECK_CU(cuStreamBatchMemOp(stream, memops.size(), memops.data(), 0));
      memops.clear();
    }
  }

  void peerWriteDyn(uint32_t concurrencyIndex, size_t peerIndex, uint32_t opType, uint32_t stepValue,
      uintptr_t gatherAddress, size_t bytes) {
    auto& dyn = group->getPeerVar(peerIndex, group->cpuLocalDyns)[size * concurrencyIndex + rank];
    dyn.opType = opType;
    dyn.gatherAddress = gatherAddress;
    dyn.gatherBytes = bytes;
    dyn.stepValue = stepValue;

    group->getPeerVar(peerIndex, group->atomicStepValue)[size * concurrencyIndex + rank] = stepValue;

    group->ipcMapper->push(peerIndex, std::make_pair(concurrencyIndex, stepValue));
  }

  uintptr_t peerWaitDyn(
      uint32_t concurrencyIndex, size_t peerIndex, uint32_t opType, uint32_t stepValue, size_t bytes) {
    size_t i = group->ipcRanks[peerIndex];
    while (group->atomicStepValue[size * concurrencyIndex + i].load() < stepValue) {
      _mm_pause();
    }
    auto& dyn = group->cpuLocalDyns[size * concurrencyIndex + i];
    CHECK_DYN(dyn, opType, stepValue, bytes);

    CHECK((group->ipcMapper->pop<std::pair<uint32_t, uint32_t>>(peerIndex) ==
           std::make_pair(concurrencyIndex, stepValue)));

    return dyn.gatherAddress;
  }

  void all_gather(TensorPtr& output, const TensorPtr& input, CUstream stream);
  void reduce_scatter(TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp, CUstream stream, float premulValue);
  void allreduce(TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue);
  void broadcast(TensorPtr& tensor, int sourceRank, CUstream stream);
  void reduce(TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream);
  void barrier();
  void scatter(std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream);
  void gather(std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream);
  void alltoall(std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream);
  void cudaBarrier(CUstream stream);
};

// ============================================================================
// ProcessGroupImpl::init() - initialization using c10d::Store
// ============================================================================

void ProcessGroupImpl::init() {
  log.verbose("%d/%d: init\n", rank, size);

  auto start = std::chrono::steady_clock::now();

  std::unique_lock l(threadUnsafe);

  auto f = [&]() {
    std::string key = randomName();

    auto set = [&](std::string_view k, std::string_view value) {
      c10dstore::set(c10dStore_, k, value);
    };
    auto get = [&](std::string_view k) {
      return c10dstore::get(c10dStore_, k);
    };

    // Phase 1: Verify all ranks have started (no dependencies between ranks)
    set(fmt::sprintf("moodist_rank%d_started", rank), "1");
    {
      std::vector<std::string> startedKeys;
      for (size_t i = 0; i != size; ++i) {
        startedKeys.push_back(fmt::sprintf("moodist_rank%d_started", i));
      }
      c10dstore::wait(c10dStore_, startedKeys);
    }

    // Phase 2: Exchange addresses
    if (rank == 0) {
      set("moodist_pg_key", key);
    } else {
      key = get("moodist_pg_key");
    }
    group->setupComms->key = key;
    set(fmt::sprintf("moodist_pg_rank%d_address", rank), getAddress());
    int prevRank = (rank + size - 1) % size;
    int nextRank = (rank + 1) % size;
    std::string prevAddress = get(fmt::sprintf("moodist_pg_rank%d_address", prevRank));
    std::string nextAddress = get(fmt::sprintf("moodist_pg_rank%d_address", nextRank));

    for (auto& address : decodeAddress(prevAddress)) {
      group->setupComms->connect(address);
    }
    for (auto& address : decodeAddress(nextAddress)) {
      group->setupComms->connect(address);
    }

    log.debug("%d: Waiting for setupComms\n", rank);
    group->setupComms->waitForConnections();

    // Phase 3: Signal ready and wait for all ranks
    set(fmt::sprintf("moodist_rank%d_ready", rank), key);
    for (size_t i = 0; i != size; ++i) {
      CHECK(get(fmt::sprintf("moodist_rank%d_ready", i)) == key);
    }

    log.debug("%d: Waiting for connections\n", rank);
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      group->setupComms->sendTo(i, fmt::sprintf("hello %d %s", i, key));
    }
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      log.debug("%d: waiting for greeting from %d\n", rank, i);

      std::string greeting = group->setupComms->recvFrom<std::string>(i);
      CHECK(greeting == fmt::sprintf("hello %d %s", rank, key));
      log.debug("greeting ok!\n");
    }
    log.debug("got all connections\n");

    group->name = key;

    log.info("rank %d created group with name '%s'\n", rank, group->name);

    log.verbose("init took %gs\n", seconds(std::chrono::steady_clock::now() - start));
  };

  group->init(f);
}

// ============================================================================
// ProcessGroupImpl::all_gather() - AllGather collective
// ============================================================================

void ProcessGroupImpl::all_gather(TensorPtr& output, const TensorPtr& input, CUstream stream) {
  std::unique_lock l(threadUnsafe);

  CHECK(input.is_contiguous());
  CHECK(output.is_contiguous());

  bool isCuda = input.is_cuda();

  if (isCuda) {
    CHECK(input.is_cuda());
    CHECK(input.device_index() == group->deviceIndex);
    CHECK(output.is_cuda());
    CHECK(output.device_index() == group->deviceIndex);
  }

  size_t bytes = input.numel() * input.itemsize();
  size_t outputBytes = bytes * size;
  size_t pitch = bytes;

  CHECK(bytes > 0);
  CHECK(static_cast<size_t>(output.numel() * output.itemsize()) == outputBytes);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  uintptr_t inputAddress = (uintptr_t)input.data_ptr();
  uintptr_t outputAddress = (uintptr_t)output.data_ptr();

  if (!isCuda) {
    StreamData& sd = group->getStreamData(nullptr);
    std::atomic_uint32_t cpuDone = 0;

    QueueEntryAllGatherCpu* e = group->cpuThread->freelistAllGatherCpu.pop();
    e->task = taskAllGatherCpu;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->inputAddress = inputAddress;
    e->outputAddress = outputAddress;
    e->bytes = bytes;
    e->pitch = pitch;
    e->cpuDone = &cpuDone;
    group->cpuThread->enqueue(e);
    l.unlock();
    while (cpuDone == 0) {
      futexWait(&cpuDone, 0, std::chrono::seconds(10));
    }
    return;
  }

  StreamData& sd = group->getStreamData(stream);
  EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
  StreamGuard sg(stream, group->deviceIndex);

  Group* group = &*this->group;

  AllGather& allGather = *group->allGather;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerIndices = group->peerIndices;

  bool isLocalOnly = allGather.recvRanks.empty();
  bool isNoLocal = !isLocalOnly && peerIndices.empty();

  AllocatedBuffer alignedBuffer;
  AllocatedBuffer alignedBuffer2;

  bool kernelLess = !isLocalOnly && (isNoLocal || preferKernelLess);

  if (!isNoLocal && !kernelLess) {
    if (bytes % 16 != 0 || outputAddress % 16 != 0) {
      pitch = (bytes + 127u) / 128u * 128u;
      alignedBuffer = alloccuda(pitch * size, stream);
      outputAddress = alignedBuffer.cudaPointer;
      outputBytes = pitch * size;
    }

    if (inputAddress % 16 != 0) {
      alignedBuffer2 = alloccuda(bytes, stream);
      CHECK_CU(cuMemcpyDtoDAsync(alignedBuffer2.cudaPointer, inputAddress, bytes, stream));
      inputAddress = alignedBuffer2.cudaPointer;
    }
  }

  IpcMapper* ipcMapper = &*group->ipcMapper;

  std::shared_lock unmapLock(unmapMemoryMutex);
  sync(stepValue);

  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;

  for (size_t i : peerIndices) {
    ipcMapper->requestAddress(
        i, inputAddress, bytes,
        [ptr = &peerInputAddresses[i]](uintptr_t address) {
          *ptr = address;
        },
        true);

    ipcMapper->requestAddress(
        i, outputAddress, outputBytes,
        [ptr = &peerOutputAddresses[i]](uintptr_t address) {
          *ptr = address;
        },
        true);
  }

  ipcMapper->wait();

  size_t numChunks = 0;
  size_t chunkSize = 0;

  bool kernelLess_ring = kernelLess && false;

  bool direct = !isLocalOnly && kernelLess && !kernelLess_ring;

  if (!isLocalOnly) {
    QueueEntryAllGather* e = group->cpuThread->freelistAllGather.pop();
    e->task = direct ? taskAllGatherDirect : taskAllGather;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->inputAddress = inputAddress;
    e->outputAddress = outputAddress;
    e->bytes = bytes;
    e->pitch = pitch;

    e->numDevices = bytes < 262144 ? 1 : std::max((size_t)2, group->numTrueIbDevs);
    e->numChunks = std::min(bytes / 131072, (size_t)maxChunks);
    if (direct) {
      e->numDevices = 4;
    }
    e->numParallel = 1;
    e->numDevices = std::min(e->numDevices, group->rdmaDevs.size());
    e->numDevices = std::max(e->numDevices, (size_t)1);
    e->numChunks = std::max(e->numChunks, (size_t)1);
    chunkSize = ((bytes + e->numChunks - 1) / e->numChunks + 4095u) / 4096u * 4096u;
    e->numChunks = (bytes + chunkSize - 1) / chunkSize;
    numChunks = e->numChunks;
    group->cpuThread->enqueue(e);
  }

  if (kernelLess_ring) {
    // kernelLess_all_gather_ring_copies not yet ported
    throw std::runtime_error("kernelLess_ring all_gather path not yet implemented");
  }

  if (kernelLess && (!isNoLocal || direct)) {
    CHECK(pitch == bytes);
    // kernelLess_all_gather_impl not yet ported
    throw std::runtime_error("kernelLess all_gather path not yet implemented");
  }
  CHECK(!direct);

  if (!group->kernels->cuAllGather) {
    group->kernels->compile(CompileAllGather);
    CHECK(group->kernels->cuAllGather != nullptr);
    CHECK(group->kernels->cuAllGatherLocal != nullptr);
    CHECK(group->kernels->cuAllGatherNoLocal != nullptr);
  }

  AllGatherParameters parameters;
  parameters.stepValue = stepValue;
  parameters.concurrencyIndex = concurrencyIndex;
  parameters.bytes = bytes;
  parameters.pitch = pitch;
  parameters.chunkSize = chunkSize;
  parameters.inputAddress = inputAddress;
  parameters.outputAddress = outputAddress;
  parameters.peerInputAddresses = peerInputAddresses;
  parameters.peerOutputAddresses = peerOutputAddresses;

  std::array<void*, 1> params = {&parameters};

  size_t gridSize = group->kernels->gridSize;
  size_t blockSize = group->kernels->blockSize;

  if (isLocalOnly) {
    CHECK_CU(cuLaunchKernel(
        group->kernels->cuAllGatherLocal, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));
  } else if (isNoLocal) {

    bool copyOutput = outputAddress + pitch * rank != inputAddress;

    if (copyOutput) {
      localEvent.record(stream);
    }

    if (kernelLess) {
      memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
      memWaitEq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
      memFlush(stream);
    } else {
      CHECK_CU(cuLaunchKernel(group->kernels->cuAllGatherNoLocal, 1, 1, 1, 1, 1, 1, 0, stream, params.data(), nullptr));
    }

    if (copyOutput) {
      auto copyStream = getCopyStream(stream);
      localEvent.wait(copyStream);
      CHECK_CU(cuMemcpyAsync(outputAddress + pitch * rank, inputAddress, bytes, copyStream));
      localEvent.record(copyStream);
      localEvent.wait(stream);
    }

  } else {
    CHECK(chunkSize != 0);
    CHECK_CU(cuLaunchKernel(
        group->kernels->cuAllGather, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));
  }

  if (!isNoLocal && outputAddress != (uintptr_t)output.data_ptr()) {
    CUDA_MEMCPY2D copyArgs = {0};
    copyArgs.srcDevice = alignedBuffer.cudaPointer;
    copyArgs.srcPitch = pitch;
    copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyArgs.dstDevice = (uintptr_t)output.data_ptr();
    copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyArgs.WidthInBytes = bytes;
    copyArgs.Height = size;
    CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));
  }
}

// ============================================================================
// ProcessGroupImpl::reduce_scatter() - ReduceScatter collective
// ============================================================================

void ProcessGroupImpl::reduce_scatter(
    TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp, CUstream stream, float premulValue) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  CHECK(input.is_contiguous());
  CHECK(output.is_contiguous());

  bool isCuda = input.is_cuda();

  if (isCuda) {
    CHECK(input.is_cuda());
    CHECK(input.device_index() == group->deviceIndex);
    CHECK(output.is_cuda());
    CHECK(output.device_index() == group->deviceIndex);
  }

  DType dtype = output.dtype();
  CHECK(input.dtype() == dtype);

  Dtype dindex = toInternalDtype(dtype);
  Reduction opindex = toInternalReduction(reduceOp);
  bool workaroundMean = (reduceOp == ReduceOp::AVG);
  bool premul = (reduceOp == ReduceOp::PREMUL_SUM);

  size_t numel = output.numel();
  size_t itemsize = output.itemsize();
  size_t bytes = numel * itemsize;
  size_t inputBytes = bytes * size;
  size_t pitch = bytes;

  CHECK(bytes > 0);
  CHECK(static_cast<size_t>(input.numel() * input.itemsize()) == inputBytes);

  uintptr_t inputAddress = (uintptr_t)input.data_ptr();
  uintptr_t outputAddress = (uintptr_t)output.data_ptr();

  TensorPtr premulTensor;

  Group* group = &*this->group;

  ReduceScatter& reduceScatter = *group->reduceScatter;

  auto& sendRanks = reduceScatter.sendRanks;
  auto& recvRanks = reduceScatter.recvRanks;

  if (!isCuda) {
    if (premul) {
      premulTensor = input * premulValue;
      inputAddress = (uintptr_t)premulTensor.data_ptr();
      CHECK(static_cast<size_t>(premulTensor.numel() * premulTensor.itemsize()) == inputBytes);
    }
    StreamData& sd = group->getStreamData(nullptr);
    std::atomic_uint32_t cpuDone = 0;
    QueueEntryReduceScatterCpu* e = group->cpuThread->freelistReduceScatterCpu.pop();
    e->task = taskReduceScatterCpu;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->inputAddress = inputAddress;
    e->outputAddress = outputAddress;
    e->bytes = bytes;
    e->pitch = pitch;
    e->cpuDone = &cpuDone;
    e->dindex = dindex;
    e->opindex = opindex;
    group->cpuThread->enqueue(e);
    l.unlock();
    while (cpuDone == 0) {
      futexWait(&cpuDone, 0, std::chrono::seconds(10));
    }
    if (workaroundMean) {
      output.mul_(1.0f / size);
    }
    return;
  }

  StreamData& sd = group->getStreamData(stream);
  EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
  StreamGuard sg(stream, group->deviceIndex);

  if (premul) {
    premulTensor = input * premulValue;
    inputAddress = (uintptr_t)premulTensor.data_ptr();
    CHECK(static_cast<size_t>(premulTensor.numel() * premulTensor.itemsize()) == inputBytes);
  }

  IpcMapper* ipcMapper = &*group->ipcMapper;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerIndices = group->peerIndices;

  AllocatedBuffer alignedBuffer;
  AllocatedBuffer alignedBuffer2;

  bool isLocalOnly = reduceScatter.recvRanks.empty();
  bool isNoLocal = peerIndices.empty();

  // Use kernel method for now (copy/direct paths need more work)
  enum { methodKernel, methodCopy, methodDirect };
  int method = methodKernel;
  if (!isLocalOnly) {
    // For now, only support kernel path - direct/copy require torch tensor ops
    method = methodKernel;
  }

  bool direct = method == methodDirect;
  bool copy = method == methodCopy;

  if (direct || copy) {
    throw std::runtime_error("reduce_scatter copy/direct paths not yet implemented");
  }

  if (bytes % 16 != 0 || inputAddress % 16 != 0) {
    pitch = (bytes + 127u) / 128u * 128u;
    alignedBuffer = alloccuda(pitch * size, stream);
    CUDA_MEMCPY2D copyArgs = {0};
    copyArgs.srcDevice = inputAddress;
    copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyArgs.dstDevice = alignedBuffer.cudaPointer;
    copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyArgs.dstPitch = pitch;
    copyArgs.WidthInBytes = bytes;
    copyArgs.Height = size;
    CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));
    inputAddress = alignedBuffer.cudaPointer;
    inputBytes = pitch * size;
  }

  if (outputAddress % 16 != 0) {
    alignedBuffer2 = alloccuda(pitch, stream);
    CHECK_CU(cuMemcpyDtoDAsync(alignedBuffer2.cudaPointer, outputAddress, bytes, stream));
    outputAddress = alignedBuffer2.cudaPointer;
  }

  AllocatedBuffer sendBuffer;
  if (!isLocalOnly) {
    sendBuffer = alloccuda(pitch * sendRanks.size(), stream);
  }
  AllocatedBuffer recvBuffer;
  if (!isLocalOnly) {
    recvBuffer = alloccuda(pitch * recvRanks.size(), stream);
  }

  std::shared_lock unmapLock(unmapMemoryMutex);
  sync(stepValue);

  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;

  for (size_t i : peerIndices) {
    ipcMapper->requestAddress(
        i, inputAddress, inputBytes,
        [ptr = &peerInputAddresses[i]](uintptr_t address) {
          *ptr = address;
        },
        true);

    ipcMapper->requestAddress(
        i, outputAddress, bytes,
        [ptr = &peerOutputAddresses[i]](uintptr_t address) {
          *ptr = address;
        },
        true);
  }

  ipcMapper->wait();

  size_t chunkSize = 0;

  uintptr_t sendAddress = sendBuffer.cudaPointer;
  uintptr_t recvAddress = recvBuffer.cudaPointer;

  if (!isLocalOnly) {
    QueueEntryReduceScatter* e = group->cpuThread->freelistReduceScatter.pop();
    e->task = taskReduceScatter;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->inputAddress = inputAddress;
    e->outputAddress = outputAddress;
    e->bytes = bytes;
    e->pitch = pitch;

    e->sendAddress = sendAddress;
    e->recvAddress = recvAddress;

    e->isCopy = false;

    e->numDevices = bytes < 262144 ? 1 : std::max((size_t)4, group->numTrueIbDevs);
    e->numChunks = std::min(bytes / 131072, (size_t)4);
    e->numDevices = std::min(e->numDevices, group->rdmaDevs.size());
    e->numDevices = std::max(e->numDevices, (size_t)1);
    e->numChunks = std::max(e->numChunks, (size_t)1);
    e->numParallel = 1;
    chunkSize = ((bytes + e->numChunks - 1) / e->numChunks + 4095u) / 4096u * 4096u;
    e->numChunks = (bytes + chunkSize - 1) / chunkSize;
    group->cpuThread->enqueue(e);
  }

  if (!group->kernels->cuReduceScatter[(size_t)dindex][(size_t)opindex]) {
    group->kernels->compile(CompileReduceScatter, group->kernels->supportedTypes[(size_t)dindex],
        group->kernels->supportedReductions[(size_t)opindex]);
    CHECK(group->kernels->cuReduceScatterLocal[(size_t)dindex][(size_t)opindex] != nullptr);
    CHECK(group->kernels->cuReduceScatter[(size_t)dindex][(size_t)opindex] != nullptr);
  }

  ReduceScatterParameters parameters;
  parameters.stepValue = stepValue;
  parameters.concurrencyIndex = concurrencyIndex;
  parameters.bytes = bytes;
  parameters.pitch = pitch;
  parameters.chunkSize = chunkSize;
  parameters.inputAddress = inputAddress;
  parameters.outputAddress = outputAddress;
  parameters.peerInputAddresses = peerInputAddresses;
  parameters.peerOutputAddresses = peerOutputAddresses;
  parameters.sendAddress = sendAddress;
  parameters.recvAddress = recvAddress;

  std::array<void*, 1> params = {&parameters};

  size_t gridSize = group->kernels->gridSize;
  size_t blockSize = group->kernels->blockSize;

  if (isLocalOnly) {
    CHECK_CU(cuLaunchKernel(group->kernels->cuReduceScatterLocal[(size_t)dindex][(size_t)opindex], gridSize, 1, 1,
        blockSize, 1, 1, 0, stream, params.data(), nullptr));
  } else {
    CHECK_CU(cuLaunchKernel(group->kernels->cuReduceScatter[(size_t)dindex][(size_t)opindex], gridSize, 1, 1, blockSize,
        1, 1, 0, stream, params.data(), nullptr));
  }

  if (outputAddress != (uintptr_t)output.data_ptr()) {
    CHECK_CU(cuMemcpyDtoDAsync((uintptr_t)output.data_ptr(), outputAddress, bytes, stream));
  }

  if (workaroundMean) {
    StreamGuard sg2(stream, group->deviceIndex);
    output.mul_(1.0f / size);
  }
}

// ============================================================================
// ProcessGroupImpl::allreduce() - Allreduce collective
// ============================================================================

void ProcessGroupImpl::allreduce(TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue) {
  // Allreduce is implemented as reduce_scatter + all_gather

  int64_t numel = tensor.numel();
  CHECK(numel > 0);

  bool isCuda = tensor.is_cuda();

  if (isCuda) {
    StreamGuard sg(stream, group->deviceIndex);
  }

  if (numel % static_cast<int64_t>(size) == 0) {
    // Tensor is evenly divisible by size - use in-place view
    std::array<int64_t, 2> viewShape = {static_cast<int64_t>(size), numel / static_cast<int64_t>(size)};
    TensorPtr t = tensor.view(viewShape);
    TensorPtr o = t.narrow(0, static_cast<int64_t>(rank), 1).view({numel / static_cast<int64_t>(size)});

    reduce_scatter(o, t, reduceOp, stream, premulValue);
    all_gather(t, o, stream);
  } else {
    // Tensor not evenly divisible - need temporary
    int64_t newsize =
        (numel + static_cast<int64_t>(size) - 1) / static_cast<int64_t>(size) * static_cast<int64_t>(size);
    size_t bytes = static_cast<size_t>(numel) * static_cast<size_t>(tensor.itemsize());

    int device = isCuda ? tensor.device_index() : -1;
    TensorPtr temporary = TensorPtr::empty({newsize}, tensor.dtype(), device);

    if (isCuda) {
      CHECK_CU(cuMemcpyAsync((uintptr_t)temporary.data_ptr(), (uintptr_t)tensor.data_ptr(), bytes, stream));
    } else {
      std::memcpy(temporary.data_ptr(), tensor.data_ptr(), bytes);
    }

    std::array<int64_t, 2> viewShape = {static_cast<int64_t>(size), newsize / static_cast<int64_t>(size)};
    TensorPtr t = temporary.view(viewShape);
    TensorPtr o = t.narrow(0, static_cast<int64_t>(rank), 1).view({newsize / static_cast<int64_t>(size)});

    reduce_scatter(o, t, reduceOp, stream, premulValue);
    all_gather(t, o, stream);

    if (isCuda) {
      CHECK_CU(cuMemcpyAsync((uintptr_t)tensor.data_ptr(), (uintptr_t)temporary.data_ptr(), bytes, stream));
    } else {
      std::memcpy(tensor.data_ptr(), temporary.data_ptr(), bytes);
    }
  }
}

// ============================================================================
// ProcessGroupImpl::broadcast() - Broadcast collective
// ============================================================================

void ProcessGroupImpl::broadcast(TensorPtr& tensor, int sourceRank, CUstream stream) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  bool isCuda = tensor.is_cuda();

  CHECK(tensor.is_contiguous());
  CHECK(sourceRank >= 0 && static_cast<size_t>(sourceRank) < size);

  if (isCuda) {
    CHECK(tensor.device_index() == group->deviceIndex);
  }

  size_t numel = static_cast<size_t>(tensor.numel());
  size_t bytes = numel * static_cast<size_t>(tensor.itemsize());

  uintptr_t tensorAddress = (uintptr_t)tensor.data_ptr();

  if (!isCuda) {
    StreamData& sd = group->getStreamData(nullptr);
    std::atomic_uint32_t cpuDone = 0;
    QueueEntryBroadcastCpu* e = group->cpuThread->freelistBroadcastCpu.pop();
    e->task = taskBroadcastCpu;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->tensorAddress = tensorAddress;
    e->bytes = bytes;
    e->sourceRank = static_cast<uint32_t>(sourceRank);
    e->cpuDone = &cpuDone;
    group->cpuThread->enqueue(e);
    l.unlock();
    while (cpuDone == 0) {
      futexWait(&cpuDone, 0, std::chrono::seconds(10));
    }
    return;
  }

  StreamData& sd = group->getStreamData(stream);
  EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
  StreamGuard sg(stream, group->deviceIndex);

  Group* group = &*this->group;

  std::shared_lock unmapLock(unmapMemoryMutex);
  sync(stepValue);

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerIndices = group->peerIndices;

  IpcMapper* ipcMapper = &*group->ipcMapper;

  CHECK(memops.empty());

  if (!peerMemcpyStream) {
    peerMemcpyStream = Stream::create(streamPriority);
  }

  size_t sourceLocalRank = group->rankLocalRank.at(static_cast<size_t>(sourceRank));
  size_t localSourceRank = static_cast<size_t>(sourceRank);
  {
    auto& node = group->nodeRanks.at(group->rankToNodeIndex.at(rank));
    localSourceRank = node[sourceLocalRank % node.size()];
  }

  size_t numDevices = std::min(bytes / 262144, std::max((size_t)4, group->numTrueIbDevs));
  size_t numChunks = std::min(bytes / 131072, (size_t)4);
  if (peerIndices.empty()) {
    numDevices = std::min(bytes / 65536, std::max((size_t)4, group->numTrueIbDevs));
    numChunks = std::min(bytes / 65536, (size_t)4);
  }
  numDevices = std::min(numDevices, group->rdmaDevs.size());
  numDevices = std::max(numDevices, (size_t)1);
  numChunks = std::max(numChunks, (size_t)1);
  size_t chunkSize = ((bytes + numChunks - 1) / numChunks + 4095u) / 4096u * 4096u;

  bool networked = group->nodeRanks.size() != 1 && localSourceRank == rank;

  if (networked) {
    QueueEntryBroadcast* e = group->cpuThread->freelistBroadcast.pop();
    e->task = taskBroadcast;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->tensorAddress = tensorAddress;
    e->bytes = bytes;
    e->sourceRank = static_cast<uint32_t>(sourceRank);

    e->numDevices = numDevices;
    e->numChunks = numChunks;
    e->numParallel = 1;

    group->cpuThread->enqueue(e);
  }

  bool anyLocalPeers = !peerIndices.empty();

  if (!anyLocalPeers) {
    CHECK(rank == localSourceRank);
  }

  size_t next = 0;
  size_t prev = peerIndices.size() - 1;

  uintptr_t peerAddress = 0;
  if (anyLocalPeers && ipcRanks[next] != localSourceRank) {
    ipcMapper->requestAddress(
        next, tensorAddress, bytes,
        [ptr = &peerAddress](uintptr_t address) {
          *ptr = address;
        },
        true);

    ipcMapper->wait();
  }

  if (anyLocalPeers) {
    peerWriteDyn(concurrencyIndex, next, opTypeBroadcastCuda, stepValue, peerAddress, bytes);

    peerAddress = peerWaitDyn(concurrencyIndex, prev, opTypeBroadcastCuda, stepValue, bytes);
  }

  if (networked) {
    memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue);
    memFlush(stream);
  }

  size_t offset = 0;
  for (size_t chunkIndex = 0; offset != bytes; ++chunkIndex) {
    size_t currentChunkSize = std::min(chunkSize, bytes - offset);
    if (localSourceRank == rank) {
      CHECK(peerAddress == 0);
      if (static_cast<size_t>(sourceRank) != rank) {
        CHECK(networked);
        memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) +
                       sizeof(uint32_t) * (16 + size * chunkIndex + static_cast<size_t>(sourceRank)),
            stepValue);
        memFlush(stream);
      }
      if (anyLocalPeers) {
        ipcMapper->streamRecord(next, stream);
      }
    } else {
      CHECK(peerAddress != 0);

      ipcMapper->streamWait(prev, stream);

      auto& copyStream = peerMemcpyStreamPerChunk[chunkIndex];
      if (!copyStream) {
        copyStream = Stream::create(streamPriority);
      }

      localEvent.record(stream);
      localEvent.wait(*copyStream);
      CHECK_CU(cuMemcpyDtoDAsync(tensorAddress + offset, peerAddress + offset, currentChunkSize, *copyStream));
      localEvent.record(*copyStream);
      localEvent.wait(stream);

      if (ipcRanks[next] != localSourceRank) {
        ipcMapper->streamRecord(next, stream);
      }
    }
    offset += currentChunkSize;
  }

  if (localSourceRank != rank) {
    ipcMapper->streamRecord(prev, stream);
  }
  if (anyLocalPeers && ipcRanks[next] != localSourceRank) {
    ipcMapper->streamWait(next, stream);
  }

  if (networked) {
    memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
    memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
    memFlush(stream);
  }

  CHECK(memops.empty());
}

// ============================================================================
// ProcessGroupImpl::reduce() - Reduce collective
// ============================================================================

void ProcessGroupImpl::reduce(TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  bool isCuda = tensor.is_cuda();

  CHECK(tensor.is_contiguous());
  CHECK(destRank >= 0 && static_cast<size_t>(destRank) < size);

  if (isCuda) {
    CHECK(tensor.device_index() == group->deviceIndex);
  }

  size_t numel = static_cast<size_t>(tensor.numel());
  size_t bytes = numel * static_cast<size_t>(tensor.itemsize());

  uintptr_t tensorAddress = (uintptr_t)tensor.data_ptr();

  if (!isCuda) {
    throw std::runtime_error("CPU reduce currently not supported :(");
  }

  StreamData& sd = group->getStreamData(stream);
  EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
  StreamGuard sg(stream, group->deviceIndex);

  Group* group = &*this->group;

  std::shared_lock unmapLock(unmapMemoryMutex);
  sync(stepValue);

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerIndices = group->peerIndices;

  IpcMapper* ipcMapper = &*group->ipcMapper;

  // Only SUM supported for now (other ops need sum_out variant)
  CHECK(reduceOp == ReduceOp::SUM);

  CHECK(memops.empty());

  if (!peerMemcpyStream) {
    peerMemcpyStream = Stream::create(streamPriority);
  }
  if (!peerIncomingMemcpyStream) {
    peerIncomingMemcpyStream = Stream::create(streamPriority);
  }

  size_t destinationLocalRank = group->rankLocalRank.at(static_cast<size_t>(destRank));
  size_t localDestinationRank = static_cast<size_t>(destRank);
  {
    auto& node = group->nodeRanks.at(group->rankToNodeIndex.at(rank));
    localDestinationRank = node[destinationLocalRank % node.size()];
  }

  size_t numDevices = std::min(bytes / 262144, std::max((size_t)4, group->numTrueIbDevs));
  size_t numChunks = std::min(bytes / 131072, (size_t)4);
  if (peerIndices.empty()) {
    numDevices = std::min(bytes / 65536, std::max((size_t)4, group->numTrueIbDevs));
    numChunks = std::min(bytes / 65536, (size_t)4);
  }
  numDevices = std::min(numDevices, group->rdmaDevs.size());
  numDevices = std::max(numDevices, (size_t)1);
  numChunks = std::max(numChunks, (size_t)1);
  size_t chunkSize = ((bytes + numChunks - 1) / numChunks + 4095u) / 4096u * 4096u;

  bool networked = group->nodeRanks.size() != 1 && localDestinationRank == rank;

  TreeSendsRecvs* tree = nullptr;
  size_t bufferSize = 0;
  size_t recvOffset = 0;

  AllocatedBuffer reduceBuffer;
  auto& reduceStream = reduceStreamArr[reduceCounter];
  if (localDestinationRank == rank) {
    reduceCounter = (reduceCounter + 1) % reduceStreamArr.size();
  }

  if (!reduceStream) {
    reduceStream = Stream::create(streamPriority);
  }

  if (networked) {
    tree = group->getTree(4, static_cast<size_t>(destRank));

    recvOffset = 1 + peerIndices.size();
    bufferSize = recvOffset + tree->recvs.size();

    reduceBuffer = alloccuda(bytes * bufferSize, stream);

    QueueEntryReduce* e = group->cpuThread->freelistReduce.pop();
    e->task = taskReduce;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->tensorAddress = tensorAddress;
    e->bytes = bytes;
    e->destinationRank = static_cast<uint32_t>(destRank);

    e->recvBuffer = tree->recvs.empty() ? 0 : reduceBuffer.cudaPointer + bytes * recvOffset;

    e->numDevices = numDevices;
    e->numChunks = numChunks;
    e->numParallel = 1;

    group->cpuThread->enqueue(e);
  }

  bool anyLocalPeers = !peerIndices.empty();

  if (!anyLocalPeers) {
    CHECK(rank == localDestinationRank);
  }

  if (localDestinationRank != rank) {
    CHECK(anyLocalPeers);
    CHECK(!networked);
    size_t peerIndex = group->getPeerIndex(localDestinationRank);

    uintptr_t peerAddress = 0;
    ipcMapper->requestAddress(
        peerIndex, tensorAddress, bytes,
        [ptr = &peerAddress](uintptr_t address) {
          *ptr = address;
        },
        true);
    ipcMapper->wait();

    peerWriteDyn(concurrencyIndex, peerIndex, opTypeReduceCuda, stepValue, peerAddress, bytes);

    CHECK(peerWaitDyn(concurrencyIndex, peerIndex, opTypeReduceCuda, stepValue, bytes) == 0);

    localEvent.record(*peerIncomingMemcpyStream);
    localEvent.wait(stream);
    ipcMapper->streamRecord(peerIndex, stream);

    ipcMapper->streamWait(peerIndex, stream);
    localEvent.record(stream);
    localEvent.wait(*peerIncomingMemcpyStream);

  } else {

    std::array<uintptr_t, 8> peerAddresses;
    peerAddresses.fill(0);

    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(concurrencyIndex, peerIndex, opTypeReduceCuda, stepValue, 0, bytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] = peerWaitDyn(concurrencyIndex, peerIndex, opTypeReduceCuda, stepValue, bytes);
    }

    if (networked) {
      memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue);
      memFlush(stream);
    } else {
      CHECK(bufferSize == 0);
      bufferSize = 1 + peerIndices.size();
      reduceBuffer = alloccuda(bytes * bufferSize, stream);
    }

    localEvent.record(*reduceStream);
    localEvent.wait(*peerMemcpyStream);

    localEvent.record(stream);
    localEvent.wait(*reduceStream);

    for (size_t peerIndex : peerIndices) {
      ipcMapper->streamWait(peerIndex, *peerMemcpyStream);
      CHECK(reduceBuffer.cudaPointer != 0);
      CHECK(peerAddresses[peerIndex] != 0);
      CHECK_CU(cuMemcpyDtoDAsync(
          reduceBuffer.cudaPointer + bytes * peerIndex, peerAddresses[peerIndex], bytes, *peerMemcpyStream));

      ipcMapper->streamRecord(peerIndex, *peerMemcpyStream);
    }

    CHECK_CU(
        cuMemcpyDtoDAsync(reduceBuffer.cudaPointer + bytes * peerIndices.size(), tensorAddress, bytes, *reduceStream));

    localEvent.record(*peerMemcpyStream);
    localEvent.wait(*reduceStream);

    // Create TensorPtr for buffer and use sum_out
    std::array<int64_t, 2> bufferShape = {static_cast<int64_t>(bufferSize), static_cast<int64_t>(numel)};
    TensorPtr buffer =
        TensorPtr::from_blob((void*)reduceBuffer.cudaPointer, bufferShape, tensor.dtype(), tensor.device_index());

    CHECK(static_cast<size_t>(buffer.numel() * buffer.itemsize()) == bytes * bufferSize);

    if (networked) {
      size_t offset = 0;
      for (size_t chunkIndex = 0; offset != bytes; ++chunkIndex) {
        size_t currentChunkSize = std::min(chunkSize, bytes - offset);

        for (size_t index = 0; index != tree->recvs.size(); ++index) {
          memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + size * chunkIndex + index),
              stepValue);
        }

        offset += currentChunkSize;
      }
      memFlush(*reduceStream);
    }

    {
      StreamGuard sg2(*reduceStream, group->deviceIndex);
      sum_out(tensor, buffer, 0);
    }

    localEvent.record(*reduceStream);
    localEvent.wait(stream);

    if (networked) {
      size_t offset = 0;
      for (size_t chunkIndex = 0; offset != bytes; ++chunkIndex) {
        size_t currentChunkSize = std::min(chunkSize, bytes - offset);

        CHECK(tree->sends.size() <= 1);
        for (size_t index = 0; index != tree->sends.size(); ++index) {
          memWrite(group->cpuInBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + size * chunkIndex + index),
              stepValue);
        }

        offset += currentChunkSize;
      }
      memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
      memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
      memFlush(stream);
    }
  }

  // sync note at end of broadcast
  for (size_t peerIndex : peerIndices) {
    ipcMapper->push(peerIndex, stepValue + 1);
  }
  for (size_t peerIndex : peerIndices) {
    CHECK(ipcMapper->pop<uint32_t>(peerIndex) == stepValue + 1);
  }

  CHECK(memops.empty());
}

// ============================================================================
// ProcessGroupImpl::barrier() - CPU Barrier collective
// ============================================================================

void ProcessGroupImpl::barrier() {
  std::unique_lock l(threadUnsafe);
  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);
  std::atomic_uint32_t cpuDone = 0;
  QueueEntryBarrier* e = group->cpuThread->freelistBarrier.pop();
  e->task = taskBarrier;
  e->stepValue = stepValue;
  e->sd = &group->getStreamData(nullptr);
  e->concurrencyIndex = concurrencyIndex;
  e->cpuDone = &cpuDone;
  group->cpuThread->enqueue(e);
  l.unlock();
  while (cpuDone == 0) {
    futexWait(&cpuDone, 0, std::chrono::seconds(10));
  }
}

// ============================================================================
// ProcessGroupImpl::scatter() - Scatter collective
// ============================================================================

void ProcessGroupImpl::scatter(std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream) {
  // Scatter is not yet implemented in core
  throw std::runtime_error("scatter is not yet supported :(");
}

// ============================================================================
// ProcessGroupImpl::gather() - Gather collective
// ============================================================================

void ProcessGroupImpl::gather(std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  CHECK(destRank >= 0 && static_cast<size_t>(destRank) < size);
  if (static_cast<size_t>(destRank) == rank) {
    CHECK(outputs.size() == size);
  } else {
    CHECK(outputs.empty());
  }

  CHECK(input.is_contiguous());
  for (auto& output : outputs) {
    CHECK(output.is_contiguous());
  }

  bool isCuda = input.is_cuda();

  if (isCuda) {
    CHECK(input.device_index() == group->deviceIndex);
    for (auto& output : outputs) {
      CHECK(output.is_cuda());
      CHECK(output.device_index() == group->deviceIndex);
    }
  }

  size_t bytes = static_cast<size_t>(input.numel() * input.itemsize());

  CHECK(bytes > 0);

  uintptr_t inputAddress = (uintptr_t)input.data_ptr();

  if (!isCuda) {
    StreamData& sd = group->getStreamData(nullptr);
    std::atomic_uint32_t cpuDone = 0;

    QueueEntryGatherCpu* e = group->cpuThread->freelistGatherCpu.pop();
    e->task = taskGatherCpu;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->inputAddress = inputAddress;

    e->outputList.resize(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto& output = outputs[i];
      e->outputList[i] = {(uintptr_t)output.data_ptr(), static_cast<size_t>(output.itemsize() * output.numel())};
    }

    e->bytes = bytes;
    e->destinationRank = static_cast<uint32_t>(destRank);
    e->cpuDone = &cpuDone;
    group->cpuThread->enqueue(e);
    l.unlock();
    while (cpuDone == 0) {
      futexWait(&cpuDone, 0, std::chrono::seconds(10));
    }
    return;
  }

  throw std::runtime_error("CUDA gather is not yet supported :(");
}

// ============================================================================
// ProcessGroupImpl::alltoall() - AllToAll collective
// ============================================================================

void ProcessGroupImpl::alltoall(std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  CHECK(inputs.size() == size);
  CHECK(outputs.size() == size);

  for (auto& v : inputs) {
    CHECK(v.is_contiguous());
  }
  for (auto& v : outputs) {
    CHECK(v.is_contiguous());
  }

  if (size - 1 != group->peerIndices.size()) {
    throw std::runtime_error("only local alltoall is supported at the moment, sorry :(");
  }

  size_t numel = static_cast<size_t>(inputs[0].numel());
  size_t bytes = numel * static_cast<size_t>(inputs[0].itemsize());

  for (auto& v : inputs) {
    CHECK(v.is_cuda());
    CHECK(static_cast<size_t>(v.numel() * v.itemsize()) == bytes);
  }
  for (auto& v : outputs) {
    CHECK(v.is_cuda());
    CHECK(static_cast<size_t>(v.numel() * v.itemsize()) == bytes);
  }

  StreamData& sd = group->getStreamData(stream);
  EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
  StreamGuard sg(stream, group->deviceIndex);

  Group* group = &*this->group;

  std::shared_lock unmapLock(unmapMemoryMutex);
  sync(stepValue);

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerIndices = group->peerIndices;

  IpcMapper* ipcMapper = &*group->ipcMapper;

  std::vector<AllocatedBuffer> temporaryBuffers;

  std::array<uintptr_t, 8> peerMappedAddresses;
  for (size_t i : peerIndices) {
    uintptr_t ptr = (uintptr_t)inputs[ipcRanks[i]].data_ptr();
    if (ptr % 16 != 0) {
      auto buf = alloccuda(bytes, stream);
      CHECK_CU(cuMemcpyDtoDAsync(buf.cudaPointer, ptr, bytes, stream));
      ptr = buf.cudaPointer;
      temporaryBuffers.push_back(std::move(buf));
    }
    ipcMapper->requestAddress(
        i, ptr, bytes,
        [addr = &peerMappedAddresses[i]](uintptr_t address) {
          *addr = address;
        },
        true);
  }
  ipcMapper->wait();

  std::array<uintptr_t, 8> peerAddresses;
  for (size_t peerIndex : peerIndices) {
    peerWriteDyn(concurrencyIndex, peerIndex, opTypeAllToAllCuda2, stepValue, peerMappedAddresses[peerIndex], bytes);
  }
  for (size_t peerIndex : peerIndices) {
    peerAddresses[peerIndex] = peerWaitDyn(concurrencyIndex, peerIndex, opTypeAllToAllCuda2, stepValue, bytes);
  }

  freePendingIpcEvents();

  localEvent.record(stream);

  syncPeers(stream);

  // AllToAll kernel parameters
  struct AllToAllParameters {
    size_t bytes;
    uintptr_t outputAddress[8];
    uintptr_t peerInputAddresses[8];
  };

  AllToAllParameters parameters;
  parameters.bytes = bytes;
  for (size_t peerIndex : peerIndices) {
    parameters.outputAddress[peerIndex] = (uintptr_t)outputs[ipcRanks[peerIndex]].data_ptr();
    if (parameters.outputAddress[peerIndex] % 16 != 0) {
      auto buf = alloccuda(bytes, stream);
      parameters.outputAddress[peerIndex] = buf.cudaPointer;
      temporaryBuffers.push_back(std::move(buf));
    }
    parameters.peerInputAddresses[peerIndex] = peerAddresses[peerIndex];
  }

  if (!group->kernels->cuAllToAll) {
    group->kernels->compile(CompileAllToAll);
    CHECK(group->kernels->cuAllToAll != nullptr);
  }

  size_t gridSize = group->kernels->gridSize;
  size_t blockSize = group->kernels->blockSize;

  std::array<void*, 1> params = {&parameters};

  CHECK_CU(
      cuLaunchKernel(group->kernels->cuAllToAll, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));

  for (size_t peerIndex : peerIndices) {
    if (parameters.outputAddress[peerIndex] != (uintptr_t)outputs[ipcRanks[peerIndex]].data_ptr()) {
      CHECK_CU(cuMemcpyDtoDAsync(
          (uintptr_t)outputs[ipcRanks[peerIndex]].data_ptr(), parameters.outputAddress[peerIndex], bytes, stream));
    }
  }

  syncPeers(stream);

  if (outputs[rank].data_ptr() != inputs[rank].data_ptr()) {
    auto copyStream = getCopyStream(stream);
    localEvent.wait(copyStream);
    CHECK_CU(
        cuMemcpyDtoDAsync((uintptr_t)outputs[rank].data_ptr(), (uintptr_t)inputs[rank].data_ptr(), bytes, copyStream));
    localEvent.record(copyStream);
    localEvent.wait(stream);
  }
}

// ============================================================================
// ProcessGroupImpl::cudaBarrier() - CUDA Barrier
// ============================================================================

void ProcessGroupImpl::cudaBarrier(CUstream stream) {
  std::unique_lock l(threadUnsafe);
  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);
  const auto& peerIndices = group->peerIndices;
  for (size_t peerIndex : peerIndices) {
    peerWriteDyn(concurrencyIndex, peerIndex, opTypeBarrierCuda, stepValue, 0, 0);
  }
  for (size_t peerIndex : peerIndices) {
    peerWaitDyn(concurrencyIndex, peerIndex, opTypeBarrierCuda, stepValue, 0);
  }
  freePendingIpcEvents();
  syncPeers(stream);
}

// ============================================================================
// API Functions
// ============================================================================

ProcessGroupImpl* createProcessGroupImpl(void* c10dStore, int rank, int size) {
  // Create with refcount=1, return raw pointer
  // Caller is responsible for calling decRef when done
  return SharedPtrDefaultPolicy::create<ProcessGroupImpl>(c10dStore, rank, size);
}

void processGroupImplAddRef(ProcessGroupImpl* impl) {
  SharedPtrDefaultPolicy::inc(impl);
}

void processGroupImplDecRef(ProcessGroupImpl* impl) {
  if (SharedPtrDefaultPolicy::dec(impl)) {
    SharedPtrDefaultPolicy::destroy(impl);
  }
}

int processGroupImplRank(ProcessGroupImpl* impl) {
  return static_cast<int>(impl->rank);
}

int processGroupImplSize(ProcessGroupImpl* impl) {
  return static_cast<int>(impl->size);
}

// ============================================================================
// Collective operations - stubs until migration is complete
// ============================================================================

void processGroupImplAllGather(ProcessGroupImpl* impl, TensorPtr& output, const TensorPtr& input, CUstream stream) {
  impl->all_gather(output, input, stream);
}

void processGroupImplReduceScatter(ProcessGroupImpl* impl, TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp,
    CUstream stream, float premulValue) {
  impl->reduce_scatter(output, input, reduceOp, stream, premulValue);
}

void processGroupImplAllreduce(
    ProcessGroupImpl* impl, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue) {
  impl->allreduce(tensor, reduceOp, stream, premulValue);
}

void processGroupImplBroadcast(ProcessGroupImpl* impl, TensorPtr& tensor, int sourceRank, CUstream stream) {
  impl->broadcast(tensor, sourceRank, stream);
}

void processGroupImplReduce(
    ProcessGroupImpl* impl, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream) {
  impl->reduce(tensor, destRank, reduceOp, stream);
}

void processGroupImplBarrier(ProcessGroupImpl* impl) {
  impl->barrier();
}

void processGroupImplScatter(
    ProcessGroupImpl* impl, std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream) {
  impl->scatter(inputs, output, sourceRank, stream);
}

void processGroupImplGather(
    ProcessGroupImpl* impl, std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream) {
  impl->gather(outputs, input, destRank, stream);
}

void processGroupImplAllToAll(
    ProcessGroupImpl* impl, std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream) {
  impl->alltoall(outputs, inputs, stream);
}

void processGroupImplCudaBarrier(ProcessGroupImpl* impl, CUstream stream) {
  impl->cudaBarrier(stream);
}

void processGroupImplShutdown(ProcessGroupImpl* impl) {
  if (impl && impl->group && impl->group->cpuThread) {
    impl->group->cpuThread->kill(false);
  }
}

void setProfilingEnabled(bool enabled) {
  profilingEnabled = enabled;
}

bool getProfilingEnabled() {
  return profilingEnabled;
}

} // namespace moodist
