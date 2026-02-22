// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup implementation for the API layer.
// This file implements ProcessGroupImpl and the API functions.
// Collective operations are stubs until migration is complete.

#include "allgather.h"
#include "api/moodist_api.h"
#include "api/processgroup_api.h"
#include "api/tensor_ptr.h"
#include "common.h"
#include "compile_op.h"
#include "compile_op_kernel.h"
#include "cputhread.h"
#include "cuda_copy.h"
#include "group.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "queue.h"
#include "reduce_scatter.h"
#include "serialization.h"
#include "setup_comms.h"
#include "shared_ptr.h"
#include "synchronization.h"
#include "tensor_types.h"

#include "arch.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

namespace moodist {

// ============================================================================
// Helper: tensorFromTensorData - converts TensorDataPtr to TensorPtr
// ============================================================================

namespace {

// Deleter function for TensorDataPtr - called when tensor is destroyed
void tensorDataDeleter(void* ctx) {
  // ctx is Storage* (from FLPtr::release()), FLPtr constructor handles the cast
  TensorDataPtr ptr(ctx);
  // ptr destructor releases the buffer
}

} // namespace

// Helper to convert TensorDataPtr to TensorPtr
TensorPtr tensorFromTensorData(TensorDataPtr data) {
  if (!data) {
    return TensorPtr();
  }
  if (!data->isCuda) {
    // For CPU tensors, use tensorFromBlobWithDeleter to manage buffer lifetime
    // Extract all data BEFORE releasing, since release() returns Storage*, not TensorData*
    void* dataPtr = (void*)data->dataPtr;
    DType dtype = static_cast<DType>(data->dtype);
    const int64_t* shapePtr = data->shape.data();
    int ndim = static_cast<int>(data->shape.size());
    void* ctx = data.release(); // This is actually Storage*, but deleter reconstructs FLPtr from it
    return TensorPtr(wrapperApi.tensorFromBlobWithDeleter(dataPtr, shapePtr, ndim, dtype, -1, tensorDataDeleter, ctx));
  }
  // For CUDA tensors, use from_blob (TODO: may need similar treatment)
  int device = 0; // TODO: track actual device
  return TensorPtr::from_blob(
      (void*)data->dataPtr, std::span<const int64_t>(data->shape), static_cast<DType>(data->dtype), device);
}

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

// 2D copy descriptor for kernelless allgather optimization
struct Copy2d {
  size_t offset;
  size_t length;
  size_t pitch;
  size_t num;
};

// Generic hashing utilities for cache keys
constexpr size_t kHashMultiplier = 0x9e3779b97f4a7c15; // 64-bit golden ratio

struct Hasher {
  size_t h = 0;

  template<typename... Args>
  void operator()(const Args&... args) {
    ((h = (h + 1) * kHashMultiplier + static_cast<size_t>(args)), ...);
  }
};

template<typename T>
struct GenericHash {
  size_t operator()(const T& k) const noexcept {
    Hasher hasher;
    k.hash(hasher);
    return hasher.h;
  }
};

// Cache key for scatter collective (used with compile_op)
struct ScatterCacheKey {
  size_t numel;
  DType dtype;
  int sourceRank;
  bool isCuda;

  bool operator==(const ScatterCacheKey&) const = default;

  template<typename X>
  void hash(X& x) const {
    x(numel, dtype, sourceRank, isCuda);
  }
};

// Cache key for gather collective (used with compile_op)
struct GatherCacheKey {
  size_t numel;
  DType dtype;
  int destRank;
  bool isCuda;

  bool operator==(const GatherCacheKey&) const = default;

  template<typename X>
  void hash(X& x) const {
    x(numel, dtype, destRank, isCuda);
  }
};

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

// ApiFuture - API boundary wrapper for async operations
struct ApiFuture : api::Future {
  FutureImplSharedPtr impl;
  std::vector<TensorPtr> holdTensors; // Keep tensors alive until complete
  Function<void()> waitDoneCallback;
};

// CustomOpImpl - API boundary wrapper for compiled custom operations
struct CustomOpImpl : api::CustomOp {
  // The actual custom op function - returns SharedPtr<ApiFuture>
  Function<SharedPtr<ApiFuture>(TensorPtr*, size_t, TensorPtr*, size_t, CUstream)> call;
};

// Forward declaration for futureWait (defined later, used by scatter)
void futureWait(api::Future* future);

struct ProcessGroupImpl : api::ProcessGroup {
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

  // Per-PG options (can be overridden from Python)
  struct Options {
    bool preferKernelLess = false;
    bool forceKernelLess = false; // Force kernelless even on single-node (for testing)
    int64_t numChunks = -1;       // -1 = auto
    int64_t chunkSize = -1;       // -1 = auto
    int64_t method = -1;          // -1 = auto, 0 = kernel, 1 = copy, 2 = direct
  } options;

  // Cached 2D copy patterns for kernelless allgather
  std::optional<std::array<Vector<Copy2d>, 8>> allGather2dCopies;

  Reusable<IpcEvent> ipcEvents;
  Reusable<IpcEvent> pendingIpcEvents;

  int streamPriority = -1;

  // Memory operations batching for CUDA stream memory ops
  std::vector<CUstreamBatchMemOpParams> memops;

  uint32_t activeId = 0;

  uint32_t nextCustomOpId = 1;

  // Cache for scatter collective compiled ops
  HashMap<ScatterCacheKey, SharedPtr<CustomOpImpl>, GenericHash<ScatterCacheKey>> scatterCache;

  // Cache for gather collective compiled ops
  HashMap<GatherCacheKey, SharedPtr<CustomOpImpl>, GenericHash<GatherCacheKey>> gatherCache;

  void* c10dStore_ = nullptr; // Opaque pointer to c10d::Store for init

  ProcessGroupImpl(void* c10dStore, int rank_, int size_) : rank(rank_), size(size_), c10dStore_(c10dStore) {
    CHECK(rank_ >= 0 && size_ > 0 && rank_ < size_);

    // Initialize options from globals
    options.preferKernelLess = globalPreferKernelLess;

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
      cpu_pause();
    }
    auto& dyn = group->cpuLocalDyns[size * concurrencyIndex + i];
    CHECK_DYN(dyn, opType, stepValue, bytes);

    CHECK((group->ipcMapper->pop<std::pair<uint32_t, uint32_t>>(peerIndex) ==
           std::make_pair(concurrencyIndex, stepValue)));

    return dyn.gatherAddress;
  }

  void calculateAllGather2dCopies() {
    const auto& peerIndices = group->peerIndices;
    AllGather& allGather = *group->allGather;

    std::array<Vector<size_t>, 8> recvsPerPeer;

    for (auto& v : allGather.proxyDestinationInfo) {
      recvsPerPeer.at(v.proxyPeerIndex).push_back(v.source);
    }

    for (size_t peerIndex : peerIndices) {
      recvsPerPeer[peerIndex].push_back(group->ipcRanks[peerIndex]);
    }

    for (auto& v : recvsPerPeer) {
      std::sort(v.begin(), v.end());
    }

    allGather2dCopies.emplace();

    for (size_t peerIndex : peerIndices) {
      size_t i = 0;
      auto& recvs = recvsPerPeer[peerIndex];
      auto nextseq = [&]() {
        CHECK(i != recvs.size());
        size_t beginSource = recvs[i];
        size_t len = 0;
        size_t prevSource = -1;
        for (; i != recvs.size(); ++i) {
          size_t source = recvs[i];
          if (prevSource == -1 || source == prevSource + 1) {
            ++len;
            prevSource = source;
          } else {
            break;
          }
        }
        CHECK(len >= 1);
        size_t endSource = beginSource + len;
        CHECK(endSource >= beginSource);
        return std::make_pair(beginSource, endSource);
      };
      while (i != recvs.size()) {
        auto firstSeq = nextseq();
        size_t seqLen = firstSeq.second - firstSeq.first;
        size_t pitch = 0;
        size_t numSeqs = 1;
        auto prevSeq = firstSeq;
        while (i != recvs.size()) {
          size_t oi = i;
          auto seq = nextseq();
          if (seq.second - seq.first != seqLen) {
            i = oi;
            break;
          }
          size_t thisPitch = seq.first - prevSeq.first;
          if (pitch == 0) {
            pitch = thisPitch;
          } else {
            if (thisPitch != pitch) {
              i = oi;
              break;
            }
          }
          ++numSeqs;
          prevSeq = seq;
        }

        if (pitch == 0) {
          pitch = seqLen;
          CHECK(numSeqs == 1);
        }

        Copy2d c;
        c.offset = firstSeq.first;
        c.length = seqLen;
        c.pitch = pitch;
        c.num = numSeqs;

        (*allGather2dCopies)[peerIndex].push_back(c);

        CHECK(numSeqs >= 1 && numSeqs < size);
        CHECK(seqLen >= 1 && seqLen < size);
        CHECK(pitch >= seqLen && pitch < size);
      }
    }
  }

  void kernelLess_all_gather_impl(uint32_t concurrencyIndex, uint32_t stepValue, uintptr_t inputAddress,
      size_t inputBytes, uintptr_t outputAddress, size_t outputBytes, CUstream stream,
      const std::array<uintptr_t, 8>& peerMappedInputAddresses,
      const std::array<uintptr_t, 8>& peerMappedOutputAddresses, bool direct, size_t numChunks, size_t chunkSize,
      bool isLocalOnly) {

    if (!allGather2dCopies) {
      calculateAllGather2dCopies();
    }

    const auto& peerIndices = group->peerIndices;

    // Signal CPU thread (only if not local-only, since CPU thread task isn't launched for local-only)
    if (!isLocalOnly) {
      memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
      memFlush(stream);
    }

    // Copy local rank's data to output
    {
      size_t offset = inputBytes * rank;
      size_t nbytes = inputBytes;
      CHECK(inputBytes == nbytes);

      if (outputAddress + offset != inputAddress) {
        CHECK_CU(cuMemcpyDtoDAsync(outputAddress + offset, inputAddress, nbytes, stream));
      }
    }

    // Wait for CPU thread (only if not local-only and not direct mode)
    if (!isLocalOnly && !direct) {
      memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
      memFlush(stream);
    }

    std::array<uintptr_t, 8> peerAddresses;

    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(concurrencyIndex, peerIndex, opTypeAllGatherKernelLessCuda, stepValue,
          peerMappedOutputAddresses[peerIndex], outputBytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeAllGatherKernelLessCuda, stepValue, outputBytes);
    }

    freePendingIpcEvents();

    if (direct) {
      CUstream curStream = stream;

      size_t flushIndex = 0;
      size_t nQueuedChunks = 0;

      auto flushCopies = [&]() {
        if (nQueuedChunks == 0) {
          return;
        }
        size_t chunkOffset = chunkSize * flushIndex;
        size_t currentChunkSize = std::min(chunkSize * nQueuedChunks, inputBytes - chunkSize * flushIndex);

        auto copy = [&](size_t peerIndex) {
          CHECK(allGather2dCopies.has_value());
          for (auto& c : (*allGather2dCopies)[peerIndex]) {
            CHECK(c.length == 1);
            CUDA_MEMCPY2D copyArgs = {0};
            copyArgs.srcDevice = peerAddresses[peerIndex] + inputBytes * c.offset + chunkOffset;
            copyArgs.srcPitch = inputBytes * c.pitch;
            copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copyArgs.dstDevice = outputAddress + inputBytes * c.offset + chunkOffset;
            copyArgs.dstPitch = inputBytes * c.pitch;
            copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copyArgs.WidthInBytes = currentChunkSize;
            copyArgs.Height = c.num;
            CHECK_CU(cuMemcpy2DAsync(&copyArgs, curStream));
          }
        };

        for (size_t peerIndex : peerIndices) {
          copy(peerIndex);
        }

        flushIndex += nQueuedChunks;
        nQueuedChunks = 0;
      };

      for (size_t chunkIndex = 0; chunkIndex != numChunks; ++chunkIndex) {
        memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + chunkIndex), stepValue);
        memFlush(curStream);

        syncPeers(curStream);
        ++nQueuedChunks;
        if (nQueuedChunks >= 1) {
          flushCopies();
        }
      }
      flushCopies();
      syncPeers(stream);

    } else {
      auto copy = [&](size_t peerIndex) {
        CHECK(allGather2dCopies.has_value());
        for (auto& c : (*allGather2dCopies)[peerIndex]) {
          CUDA_MEMCPY2D copyArgs = {0};
          copyArgs.srcDevice = peerAddresses[peerIndex] + inputBytes * c.offset;
          copyArgs.srcPitch = inputBytes * c.pitch;
          copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
          copyArgs.dstDevice = outputAddress + inputBytes * c.offset;
          copyArgs.dstPitch = inputBytes * c.pitch;
          copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
          copyArgs.WidthInBytes = inputBytes * c.length;
          copyArgs.Height = c.num;
          CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));
        }
      };

      syncPeers(stream);
      for (size_t peerIndex : peerIndices) {
        copy(peerIndex);
      }
      syncPeers(stream);
    }

    if (direct) {
      memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) + 4, stepValue + 1);
      memFlush(stream);
    }
  }

  void reduce_scatter_copy_prepare_sends(uint32_t concurrencyIndex, uint32_t stepValue,
      const std::array<uintptr_t, 8>& peerInputAddresses, uintptr_t inputAddress, uintptr_t bufferAddress,
      uintptr_t sendAddress, size_t bytes, DType dtype, int device, size_t numel, Reduction opindex, CUstream stream,
      size_t copyBatchSize) {
    ReduceScatter& reduceScatter = *group->reduceScatter;
    const auto& peerIndices = group->peerIndices;
    const auto& sendRanks = reduceScatter.sendRanks;
    const size_t numSends = sendRanks.size();

    std::array<uintptr_t, 8> peerAddresses;

    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(concurrencyIndex, peerIndex, opTypeReduceScatterKernelLessCuda, stepValue,
          peerInputAddresses[peerIndex], bytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeReduceScatterKernelLessCuda, stepValue, bytes);
    }
    freePendingIpcEvents();

    syncPeers(stream);

    StreamGuard sg(stream, device);

    TensorPtr bufferTensor = TensorPtr::from_blob(
        (void*)bufferAddress, {(int64_t)((1 + peerIndices.size()) * copyBatchSize), (int64_t)numel}, dtype, device);

    if (sendRanks.size()) {
      CHECK(copyBatchSize >= 1);
      TensorPtr outTensor =
          TensorPtr::from_blob((void*)sendAddress, {(int64_t)sendRanks.size(), (int64_t)numel}, dtype, device);

      TensorPtr buft = bufferTensor.view({(int64_t)copyBatchSize, (int64_t)(1 + peerIndices.size()), (int64_t)numel});

      size_t begin = 0;
      size_t pitch = 0;
      size_t prev = 0;
      size_t n = 1;
      auto flush = [&]() {
        size_t srcOffsetBytes = bytes * sendRanks[begin];
        size_t srcPitchBytes = bytes * pitch;
        size_t dstOffsetBytes = 0;
        size_t dstPitchBytes = bytes * (1 + peerIndices.size());

        CUDA_MEMCPY2D copyArgs = {0};
        copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyArgs.srcPitch = srcPitchBytes;
        copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copyArgs.dstPitch = dstPitchBytes;
        copyArgs.WidthInBytes = bytes;
        copyArgs.Height = n;

        for (size_t peerIndex : peerIndices) {
          copyArgs.srcDevice = peerAddresses[peerIndex] + srcOffsetBytes;
          copyArgs.dstDevice = bufferAddress + dstOffsetBytes;
          CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));

          dstOffsetBytes += bytes;
        }

        copyArgs.srcDevice = inputAddress + srcOffsetBytes;
        copyArgs.dstDevice = bufferAddress + dstOffsetBytes;
        CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));

        TensorPtr src = buft.narrow(0, 0, n);
        TensorPtr dst = outTensor.narrow(0, begin, n);
        if (opindex == Reduction::sum) {
          sum_out(dst, src, 1);
        } else if (opindex == Reduction::max) {
          amax_out(dst, src, 1);
        } else if (opindex == Reduction::min) {
          amin_out(dst, src, 1);
        } else {
          CHECK(false);
        }

        for (size_t z = 0; z < n; ++z) {
          memWrite(group->cpuInBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + begin + z), stepValue);
        }
        memFlush(stream);
      };
      for (size_t i = 1; i < numSends; ++i) {
        size_t thisPitch = sendRanks[i] - sendRanks[prev];
        if ((pitch == 0 || thisPitch == pitch) && n < copyBatchSize) {
          pitch = thisPitch;
          ++n;
          prev = i;
        } else {
          flush();
          begin = i;
          prev = i;
          pitch = 0;
          n = 1;
        }
      }
      flush();
    }

    for (size_t peerIndex : peerIndices) {
      cudaCopy(bufferAddress + bytes * (1 + peerIndex), peerAddresses[peerIndex] + bytes * rank, bytes, stream);
    }
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

  // compileOpFull - compile a distributed tensor operation
  SharedPtr<CustomOpImpl> compileOpFull(DType dtype, std::span<const api::TensorRegion> inputs,
      std::span<const api::TensorRegion> outputs, api::ReduceOp reduce, bool cpuSync = false);

  // Template implementation for compileOpFull - parameterized by ndim
  // OLD: Kept for reference during refactoring. Will be deleted once new impl is working.
  template<int ndim>
  SharedPtr<CustomOpImpl> compileOpFullImpl_OLD(std::span<const int64_t> shape_arg, DType dtype,
      std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> inputs_arg,
      std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> outputs_arg);

  // customOp - execute a compiled custom operation
  SharedPtr<ApiFuture> customOp(std::shared_ptr<CustomOpDescriptor> op, TensorPtr* inputs, size_t nInputs,
      TensorPtr* outputs, size_t nOutputs, CUstream stream);

  // executeLocalOnly - fast path for all-local custom ops (no CPU thread)
  SharedPtr<ApiFuture> executeLocalOnly(std::shared_ptr<CustomOpDescriptor> op, std::vector<TensorDataPtr>& inputs,
      std::vector<TensorDataPtr>& outputs, TensorPtr* inputPtrs, size_t nInputs, TensorPtr* outputPtrs, size_t nOutputs,
      uint32_t concurrencyIndex, uint32_t stepValue, CUstream stream);

  // cat - concatenate tensors from multiple ranks
  api::FutureHandle cat(const int* indices, const TensorPtr* tensors, size_t count, TensorPtr* out);
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

  // forceKernelLess bypasses the isLocalOnly check (for testing)
  bool kernelLess = options.forceKernelLess || (!isLocalOnly && (isNoLocal || options.preferKernelLess));

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
    kernelLess_all_gather_impl(concurrencyIndex, stepValue, inputAddress, bytes, outputAddress, outputBytes, stream,
        peerInputAddresses, peerOutputAddresses, direct, numChunks, chunkSize, isLocalOnly);
    return;
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

  // Method selection (same as old code):
  // - methodKernel for local-only
  // - methodDirect for multi-node
  enum { methodKernel, methodCopy, methodDirect };
  int method = methodKernel;
  if (!isLocalOnly) {
    method = methodDirect;
  }

  bool direct = method == methodDirect;
  bool copy = method == methodCopy;

  size_t copyBatchSize = 0;
  if (copy) {
    copyBatchSize = (reduceScatter.sendRanks.size() + 7) / 8;
    copyBatchSize = std::max(copyBatchSize, (size_t)1);
  }

  if (!copy) {
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

    if (!direct && outputAddress % 16 != 0) {
      alignedBuffer2 = alloccuda(pitch, stream);
      CHECK_CU(cuMemcpyDtoDAsync(alignedBuffer2.cudaPointer, outputAddress, bytes, stream));
      outputAddress = alignedBuffer2.cudaPointer;
    }
  }

  AllocatedBuffer sendBuffer;
  if (!copy || !isNoLocal) {
    if (!isLocalOnly) {
      sendBuffer = alloccuda(pitch * sendRanks.size(), stream);
    }
  }
  AllocatedBuffer recvBuffer;
  if (copy) {
    recvBuffer = alloccuda(pitch * (recvRanks.size() + (1 + peerIndices.size()) * copyBatchSize), stream);
  } else if (direct) {
    recvBuffer = alloccuda(pitch * (1 + recvRanks.size() + peerIndices.size()), stream);
  } else if (!isLocalOnly) {
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
    if (copy || direct) {
      e->task = taskReduceScatterDirect;
    }
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->inputAddress = inputAddress;
    e->outputAddress = outputAddress;
    e->bytes = bytes;
    e->pitch = pitch;

    e->sendAddress = sendAddress;
    e->recvAddress = recvAddress;

    e->isCopy = copy;

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

  if (copy || direct) {

    if (direct) {
      CHECK(!isLocalOnly);

      if (!group->kernels->cuReduceScatterDirect[(size_t)dindex][(size_t)opindex]) {
        group->kernels->compile(CompileReduceScatterDirect, group->kernels->supportedTypes[(size_t)dindex],
            group->kernels->supportedReductions[(size_t)opindex]);
        CHECK(group->kernels->cuReduceScatterDirect[(size_t)dindex][(size_t)opindex] != nullptr);
      }

      std::array<uintptr_t, 8> peerAddresses;

      for (size_t peerIndex : peerIndices) {
        peerWriteDyn(concurrencyIndex, peerIndex, opTypeReduceScatterDirectCuda, stepValue,
            peerInputAddresses[peerIndex], bytes);
      }
      for (size_t peerIndex : peerIndices) {
        peerAddresses[peerIndex] =
            peerWaitDyn(concurrencyIndex, peerIndex, opTypeReduceScatterDirectCuda, stepValue, bytes);
      }

      freePendingIpcEvents();

      memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
      memFlush(stream);

      syncPeers(stream);

      CHECK(sendAddress != 0);

      ReduceScatterParameters parameters;
      parameters.stepValue = stepValue;
      parameters.concurrencyIndex = concurrencyIndex;
      parameters.bytes = bytes;
      parameters.pitch = pitch;
      parameters.chunkSize = chunkSize;
      parameters.inputAddress = inputAddress;
      parameters.outputAddress = 0;
      parameters.peerInputAddresses = peerAddresses;
      parameters.peerOutputAddresses.fill(0);
      parameters.sendAddress = sendAddress;
      parameters.recvAddress = 0;

      std::array<void*, 1> params = {&parameters};

      size_t gridSize = group->kernels->gridSize;
      size_t blockSize = group->kernels->blockSize;

      CHECK_CU(cuLaunchKernel(group->kernels->cuReduceScatterDirect[(size_t)dindex][(size_t)opindex], gridSize * 4, 1,
          1, blockSize, 1, 1, 0, stream, params.data(), nullptr));

      for (size_t peerIndex : peerIndices) {
        cudaCopy(recvAddress + pitch * recvRanks.size() + pitch * (1 + peerIndex),
            peerAddresses[peerIndex] + pitch * rank, bytes, stream);
      }

    } else {
      // copy path
      CHECK(pitch == bytes);

      if (!isLocalOnly) {
        memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
        memFlush(stream);
      }

      if (!isNoLocal) {
        CHECK(sendAddress != 0);
        reduce_scatter_copy_prepare_sends(concurrencyIndex, stepValue, peerInputAddresses, inputAddress,
            recvAddress + bytes * recvRanks.size(), sendAddress, bytes, dtype, group->deviceIndex, numel, opindex,
            stream, copyBatchSize);
      }
    }

    {
      auto ipcEvent = getIpcEvent();
      ipcEvent->record(stream);
      for (size_t peerIndex : peerIndices) {
        ipcMapper->pushEvent(peerIndex, *ipcEvent);
      }
      cudaCopy(recvAddress + pitch * recvRanks.size(), inputAddress + pitch * rank, bytes, stream);

      for (size_t peerIndex : peerIndices) {
        Event::reference(ipcMapper->popEvent(peerIndex)).wait(stream);
      }

      if (!isLocalOnly) {
        memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
        memFlush(stream);
      }
    }

    {
      int device = group->deviceIndex;
      TensorPtr outputTensor = TensorPtr::from_blob((void*)outputAddress, {(int64_t)numel}, dtype, device);
      TensorPtr buffer = TensorPtr::from_blob((void*)recvAddress,
          {(int64_t)(1 + recvRanks.size() + peerIndices.size()), (int64_t)(pitch / itemsize)}, dtype, device);
      if (buffer.size(1) != static_cast<int64_t>(numel)) {
        buffer = buffer.narrow(1, 0, numel);
      }
      if (opindex == Reduction::sum) {
        sum_out(outputTensor, buffer, 0);
      } else if (opindex == Reduction::max) {
        amax_out(outputTensor, buffer, 0);
      } else if (opindex == Reduction::min) {
        amin_out(outputTensor, buffer, 0);
      } else {
        CHECK(false);
      }
    }

    if (!isLocalOnly) {
      memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue + 1);
      memFlush(stream);
    }

  } else {
    // Kernel path

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
      CHECK_CU(cuLaunchKernel(group->kernels->cuReduceScatter[(size_t)dindex][(size_t)opindex], gridSize, 1, 1,
          blockSize, 1, 1, 0, stream, params.data(), nullptr));
    }

    if (outputAddress != (uintptr_t)output.data_ptr()) {
      CHECK_CU(cuMemcpyDtoDAsync((uintptr_t)output.data_ptr(), outputAddress, bytes, stream));
    }
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
  REQUIRE(sourceRank >= 0 && static_cast<size_t>(sourceRank) < size, "scatter: sourceRank %d out of bounds [0, %zu)",
      sourceRank, size);

  // Validate inputs
  if (static_cast<size_t>(sourceRank) == rank) {
    REQUIRE(inputs.size() == size, "scatter: source rank must provide %zu inputs, got %zu", size, inputs.size());
    for (size_t i : indices(inputs)) {
      REQUIRE(inputs[i].is_contiguous(), "scatter: input[%zu] must be contiguous", i);
    }
  } else {
    REQUIRE(inputs.empty(), "scatter: non-source ranks must not provide inputs");
  }

  REQUIRE(output.is_contiguous(), "scatter: output must be contiguous");

  size_t numel = static_cast<size_t>(output.numel());
  DType dtype = output.dtype();
  bool isCuda = output.is_cuda();
  std::string_view deviceStr = isCuda ? "cuda" : "cpu";

  // Validate device index for CUDA tensors
  if (isCuda) {
    REQUIRE(output.device_index() == group->deviceIndex,
        "scatter: output CUDA device %d does not match process group device %d", output.device_index(),
        group->deviceIndex);
  }

  // Validate all inputs have the same numel, dtype, and device as output (on source rank)
  if (static_cast<size_t>(sourceRank) == rank) {
    for (size_t i : indices(inputs)) {
      auto& input = inputs[i];
      REQUIRE(static_cast<size_t>(input.numel()) == numel, "scatter: input[%zu] has %lld elements, expected %zu", i,
          input.numel(), numel);
      REQUIRE(input.dtype() == dtype, "scatter: input[%zu] has different dtype than output", i);
      REQUIRE(input.is_cuda() == isCuda, "scatter: input[%zu] device type differs from output", i);
      if (isCuda) {
        REQUIRE(input.device_index() == group->deviceIndex,
            "scatter: input[%zu] CUDA device %d does not match process group device %d", i, input.device_index(),
            group->deviceIndex);
      }
    }
  }

  ScatterCacheKey key{numel, dtype, sourceRank, isCuda};

  auto it = scatterCache.find(key);
  if (it == scatterCache.end()) {
    // Build the compile_op spec
    // Global shape: 1D with size = world_size * numel
    std::vector<int64_t> shape = {static_cast<int64_t>(size * numel)};

    // Inputs: source rank provides world_size inputs, each at offset [i * numel] with shape [numel]
    std::vector<api::TensorRegion> inputSpecs;
    for (size_t i = 0; i < size; ++i) {
      inputSpecs.push_back(api::TensorRegion{.rank = sourceRank,
          .offset = {static_cast<int64_t>(i * numel)},
          .shape = {static_cast<int64_t>(numel)},
          .tensorId = "0",
          .device = deviceStr});
    }

    // Outputs: each rank receives one output at offset [rank * numel] with shape [numel]
    std::vector<api::TensorRegion> outputSpecs;
    for (size_t i = 0; i < size; ++i) {
      outputSpecs.push_back(api::TensorRegion{.rank = static_cast<int32_t>(i),
          .offset = {static_cast<int64_t>(i * numel)},
          .shape = {static_cast<int64_t>(numel)},
          .tensorId = "0",
          .device = deviceStr});
    }

    SharedPtr<CustomOpImpl> compiled = compileOpFull(dtype, inputSpecs, outputSpecs, api::ReduceOp::None);
    it = scatterCache.insert({key, std::move(compiled)}).first;
  }

  CustomOpImpl* op = it->second.get();

  // Flatten inputs and output for compile_op
  Vector<TensorPtr> flatInputs;
  flatInputs.reserve(inputs.size());
  for (auto& input : inputs) {
    flatInputs.push_back(input.view({static_cast<int64_t>(numel)}));
  }

  TensorPtr flatOutput = output.view({static_cast<int64_t>(numel)});

  // Execute the compiled op
  SharedPtr<ApiFuture> future = op->call(flatInputs.data(), flatInputs.size(), &flatOutput, 1, stream);

  // Wait for completion
  futureWait(future.get());
}

// ============================================================================
// ProcessGroupImpl::gather() - Gather collective
// ============================================================================

void ProcessGroupImpl::gather(std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream) {
  REQUIRE(destRank >= 0 && static_cast<size_t>(destRank) < size, "gather: destRank %d out of bounds [0, %zu)", destRank,
      size);

  // Validate outputs
  if (static_cast<size_t>(destRank) == rank) {
    REQUIRE(outputs.size() == size, "gather: dest rank must provide %zu outputs, got %zu", size, outputs.size());
    for (size_t i : indices(outputs)) {
      REQUIRE(outputs[i].is_contiguous(), "gather: output[%zu] must be contiguous", i);
    }
  } else {
    REQUIRE(outputs.empty(), "gather: non-dest ranks must not provide outputs");
  }

  REQUIRE(input.is_contiguous(), "gather: input must be contiguous");

  size_t numel = static_cast<size_t>(input.numel());
  DType dtype = input.dtype();
  bool isCuda = input.is_cuda();
  std::string_view deviceStr = isCuda ? "cuda" : "cpu";

  // Validate device index for CUDA tensors
  if (isCuda) {
    REQUIRE(input.device_index() == group->deviceIndex,
        "gather: input CUDA device %d does not match process group device %d", input.device_index(),
        group->deviceIndex);
  }

  // Validate all outputs have the same numel, dtype, and device as input (on dest rank)
  if (static_cast<size_t>(destRank) == rank) {
    for (size_t i : indices(outputs)) {
      auto& output = outputs[i];
      REQUIRE(static_cast<size_t>(output.numel()) == numel, "gather: output[%zu] has %lld elements, expected %zu", i,
          output.numel(), numel);
      REQUIRE(output.dtype() == dtype, "gather: output[%zu] has different dtype than input", i);
      REQUIRE(output.is_cuda() == isCuda, "gather: output[%zu] device type differs from input", i);
      if (isCuda) {
        REQUIRE(output.device_index() == group->deviceIndex,
            "gather: output[%zu] CUDA device %d does not match process group device %d", i, output.device_index(),
            group->deviceIndex);
      }
    }
  }

  GatherCacheKey key{numel, dtype, destRank, isCuda};

  auto it = gatherCache.find(key);
  if (it == gatherCache.end()) {
    // Build the compile_op spec
    // Global shape: 1D with size = world_size * numel
    std::vector<int64_t> shape = {static_cast<int64_t>(size * numel)};

    // Inputs: each rank provides one input at offset [rank * numel] with shape [numel]
    std::vector<api::TensorRegion> inputSpecs;
    for (size_t i = 0; i < size; ++i) {
      inputSpecs.push_back(api::TensorRegion{.rank = static_cast<int32_t>(i),
          .offset = {static_cast<int64_t>(i * numel)},
          .shape = {static_cast<int64_t>(numel)},
          .tensorId = "0",
          .device = deviceStr});
    }

    // Outputs: dest rank receives world_size outputs, each at offset [i * numel] with shape [numel]
    std::vector<api::TensorRegion> outputSpecs;
    for (size_t i = 0; i < size; ++i) {
      outputSpecs.push_back(api::TensorRegion{.rank = destRank,
          .offset = {static_cast<int64_t>(i * numel)},
          .shape = {static_cast<int64_t>(numel)},
          .tensorId = "0",
          .device = deviceStr});
    }

    SharedPtr<CustomOpImpl> compiled = compileOpFull(dtype, inputSpecs, outputSpecs, api::ReduceOp::None);
    it = gatherCache.insert({key, std::move(compiled)}).first;
  }

  CustomOpImpl* op = it->second.get();

  // Flatten input and outputs for compile_op
  TensorPtr flatInput = input.view({static_cast<int64_t>(numel)});

  Vector<TensorPtr> flatOutputs;
  flatOutputs.reserve(outputs.size());
  for (auto& output : outputs) {
    flatOutputs.push_back(output.view({static_cast<int64_t>(numel)}));
  }

  // Execute the compiled op
  SharedPtr<ApiFuture> future = op->call(&flatInput, 1, flatOutputs.data(), flatOutputs.size(), stream);

  // Wait for completion
  futureWait(future.get());
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
// Helper to convert TensorPtr to TensorDataPtr for cpuThread operations
// ============================================================================

namespace {

TensorDataPtr getTensorDataFromPtr(const TensorPtr& tensor, Group* group) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Got a non-contiguous tensor. All tensors must be contiguous.");
  }
  if (tensor.is_cuda() && tensor.device_index() != group->deviceIndex) {
    throw std::runtime_error(fmt::sprintf("Got a CUDA tensor on device %d, but process group expects device %d",
        tensor.device_index(), group->deviceIndex));
  }

  TensorDataPtr td = TensorDataPtr::make();
  td->dtype = static_cast<int>(tensor.dtype());
  int ndim = tensor.ndimension();
  td->shape.resize(ndim);
  for (int i = 0; i < ndim; ++i) {
    td->shape[i] = tensor.size(i);
  }
  td->dataPtr = (uintptr_t)tensor.data_ptr();
  td->dataBytes = td->itemsize() * td->numel();
  td->isCuda = tensor.is_cuda();

  return td;
}

} // namespace

// ============================================================================
// ProcessGroupImpl::cat() - Cat (concatenate from multiple ranks)
// ============================================================================

api::FutureHandle ProcessGroupImpl::cat(const int* indices, const TensorPtr* tensors, size_t count, TensorPtr* out) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  auto future = FutureImplSharedPtr::make();

  StreamData& sd = group->getCpuStreamData(concurrencyIndex);
  QueueEntryCat* e = group->cpuThread->freelistCat.pop();
  e->task = taskCat;
  e->stepValue = stepValue;
  e->concurrencyIndex = concurrencyIndex;
  e->sd = &sd;
  e->locals.resize(count);
  for (size_t i : range(count)) {
    e->locals[i] = {indices[i], getTensorDataFromPtr(tensors[i], group.get())};
  }
  e->future = future;
  e->out = nullptr;
  if (out) {
    e->out = getTensorDataFromPtr(*out, group.get());
  }
  group->cpuThread->enqueue(e);

  auto result = makeShared<ApiFuture>();
  result->impl = std::move(future);
  return api::FutureHandle::adopt(result.release());
}

// ============================================================================
// API Functions
// ============================================================================

api::FutureHandle processGroupCat(
    api::ProcessGroup* pg, const int* indices, const TensorPtr* tensors, size_t count, TensorPtr* out) {
  return static_cast<ProcessGroupImpl*>(pg)->cat(indices, tensors, count, out);
}

api::ProcessGroupHandle createProcessGroup(void* c10dStore, int rank, int size) {
  return api::ProcessGroupHandle::create(internalNew<ProcessGroupImpl>(c10dStore, rank, size));
}

void processGroupDestroy(api::ProcessGroup* pg) {
  internalDelete(static_cast<ProcessGroupImpl*>(pg));
}

void processGroupShutdown(api::ProcessGroup* pg) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  if (impl && impl->group && impl->group->cpuThread) {
    impl->group->cpuThread->kill(false);
  }
}

int processGroupRank(api::ProcessGroup* pg) {
  return static_cast<int>(static_cast<ProcessGroupImpl*>(pg)->rank);
}

int processGroupSize(api::ProcessGroup* pg) {
  return static_cast<int>(static_cast<ProcessGroupImpl*>(pg)->size);
}

bool processGroupGetPreferKernelLess(api::ProcessGroup* pg) {
  return static_cast<ProcessGroupImpl*>(pg)->options.preferKernelLess;
}

void processGroupSetPreferKernelLess(api::ProcessGroup* pg, bool value) {
  static_cast<ProcessGroupImpl*>(pg)->options.preferKernelLess = value;
}

int64_t processGroupGetOption(api::ProcessGroup* pg, const char* name) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  std::string_view key(name);
  if (key == "prefer_kernel_less") {
    return impl->options.preferKernelLess ? 1 : 0;
  } else if (key == "force_kernel_less") {
    return impl->options.forceKernelLess ? 1 : 0;
  } else if (key == "num_chunks") {
    return impl->options.numChunks;
  } else if (key == "chunk_size") {
    return impl->options.chunkSize;
  } else if (key == "method") {
    return impl->options.method;
  }
  throw std::runtime_error(std::string("Unknown option: ") + name);
}

void processGroupSetOption(api::ProcessGroup* pg, const char* name, int64_t value) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  std::string_view key(name);
  if (key == "prefer_kernel_less") {
    impl->options.preferKernelLess = (value != 0);
  } else if (key == "force_kernel_less") {
    impl->options.forceKernelLess = (value != 0);
  } else if (key == "num_chunks") {
    impl->options.numChunks = value;
  } else if (key == "chunk_size") {
    impl->options.chunkSize = value;
  } else if (key == "method") {
    impl->options.method = value;
  } else {
    throw std::runtime_error(std::string("Unknown option: ") + name);
  }
}

const char* processGroupGetName(api::ProcessGroup* pg) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  // Return pointer to internal string data - safe as long as ProcessGroup lives
  return impl->group->name.c_str();
}

// ============================================================================
// Collective operations
// ============================================================================

void processGroupAllGather(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->all_gather(output, input, stream);
}

void processGroupReduceScatter(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp,
    CUstream stream, float premulValue) {
  static_cast<ProcessGroupImpl*>(pg)->reduce_scatter(output, input, reduceOp, stream, premulValue);
}

void processGroupAllreduce(
    api::ProcessGroup* pg, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue) {
  static_cast<ProcessGroupImpl*>(pg)->allreduce(tensor, reduceOp, stream, premulValue);
}

void processGroupBroadcast(api::ProcessGroup* pg, TensorPtr& tensor, int sourceRank, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->broadcast(tensor, sourceRank, stream);
}

void processGroupReduce(api::ProcessGroup* pg, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->reduce(tensor, destRank, reduceOp, stream);
}

void processGroupBarrier(api::ProcessGroup* pg) {
  static_cast<ProcessGroupImpl*>(pg)->barrier();
}

void processGroupScatter(
    api::ProcessGroup* pg, std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->scatter(inputs, output, sourceRank, stream);
}

void processGroupGather(
    api::ProcessGroup* pg, std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->gather(outputs, input, destRank, stream);
}

void processGroupAllToAll(
    api::ProcessGroup* pg, std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->alltoall(outputs, inputs, stream);
}

void processGroupCudaBarrier(api::ProcessGroup* pg, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->cudaBarrier(stream);
}

api::QueueHandle processGroupMakeQueue(api::ProcessGroup* pg, int location, bool streaming, const char* name) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  std::string_view nameView = name ? name : "";
  return makeQueue(impl->group, location, streaming, nameView);
}

api::QueueHandle processGroupMakeQueueMulti(
    api::ProcessGroup* pg, const int* locations, size_t numLocations, bool streaming, const char* name) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  std::vector<int> locationVec(locations, locations + numLocations);
  std::string_view nameView = name ? name : "";
  return makeQueue(impl->group, locationVec, streaming, nameView);
}

void queueDestroy(api::Queue* queue) {
  if (queue) {
    internalDelete(static_cast<Queue*>(queue));
  }
}

bool queueGet(api::Queue* queue, bool block, const float* timeout, TensorPtr* outTensor, size_t* outSize) {
  auto* q = static_cast<Queue*>(queue);
  std::optional<float> timeoutOpt;
  if (timeout) {
    timeoutOpt = *timeout;
  }
  auto [tensor, size] = q->get(block, timeoutOpt);
  if (tensor.defined()) {
    *outTensor = std::move(tensor);
    *outSize = size;
    return true;
  }
  return false;
}

api::QueueWorkHandle queuePut(api::Queue* queue, const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy) {
  auto* q = static_cast<Queue*>(queue);
  QueueWork work = q->put(tensor, transaction, waitOnDestroy);
  // Transfer ownership to heap and wrap in ApiHandle
  auto* workPtr = internalNew<QueueWork>(std::move(work));
  return api::QueueWorkHandle::create(workPtr);
}

size_t queueQsize(api::Queue* queue) {
  return static_cast<Queue*>(queue)->qsize();
}

bool queueWait(api::Queue* queue, const float* timeout) {
  auto* q = static_cast<Queue*>(queue);
  std::optional<float> timeoutOpt;
  if (timeout) {
    timeoutOpt = *timeout;
  }
  return q->wait(timeoutOpt);
}

uint32_t queueTransactionBegin(api::Queue* queue) {
  return static_cast<Queue*>(queue)->transactionBegin();
}

void queueTransactionCancel(api::Queue* queue, uint32_t id) {
  static_cast<Queue*>(queue)->transactionCancel(id);
}

void queueTransactionCommit(api::Queue* queue, uint32_t id) {
  static_cast<Queue*>(queue)->transactionCommit(id);
}

const char* queueName(api::Queue* queue) {
  auto name = static_cast<Queue*>(queue)->name();
  // Return pointer to internal string data - safe as long as Queue lives
  return name.data();
}

void queueWorkWait(api::QueueWork* work) {
  static_cast<QueueWork*>(work)->wait();
}

void queueWorkDestroy(api::QueueWork* work) {
  internalDelete(static_cast<QueueWork*>(work));
}

// destroy() for api::QueueWork - needed by ApiHandle destructor in core
namespace api {
void destroy(QueueWork* work) {
  queueWorkDestroy(work);
}
} // namespace api

void setProfilingEnabled(bool enabled) {
  profilingEnabled = enabled;
}

bool getProfilingEnabled() {
  return profilingEnabled;
}

// ============================================================================
// ApiFuture functions
// ============================================================================

void futureWait(api::Future* future);

void futureDestroy(api::Future* future) {
  futureWait(future);
  internalDelete(static_cast<ApiFuture*>(future));
}

void futureWait(api::Future* future) {
  auto* f = static_cast<ApiFuture*>(future);
  if (f->impl) {
    while (f->impl->done == 0) {
      std::this_thread::yield();
    }
  }
  if (f->waitDoneCallback) {
    f->waitDoneCallback();
    f->waitDoneCallback = nullptr;
  }
}

bool futureGetResult(api::Future* future, TensorPtr* outTensor) {
  auto* f = static_cast<ApiFuture*>(future);
  if (f->impl && f->impl->result) {
    *outTensor = tensorFromTensorData(std::move(f->impl->result));
    return true;
  }
  return false;
}

namespace api {
void destroy(Future* future) {
  futureDestroy(future);
}
} // namespace api

// ============================================================================
// CustomOpImpl functions
// ============================================================================

void customOpDestroy(api::CustomOp* op) {
  internalDelete(static_cast<CustomOpImpl*>(op));
}

api::FutureHandle customOpCall(
    api::CustomOp* op, TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream) {
  auto* o = static_cast<CustomOpImpl*>(op);
  SharedPtr<ApiFuture> future = o->call(inputs, nInputs, outputs, nOutputs, stream);
  return api::FutureHandle::adopt(future.release());
}

namespace api {
void destroy(CustomOp* op) {
  customOpDestroy(op);
}
} // namespace api

// ============================================================================
// compileOpFull - implementation
// ============================================================================

namespace {

// Helper to serialize data to a CPU TensorPtr for queue operations
template<typename... T>
TensorPtr serializeToTensorPtr(const T&... v) {
  auto buffer = serializeToBuffer(v...);
  size_t size = buffer->size();
  // Create a 1D uint8 CPU tensor and copy data into it
  int64_t shape = static_cast<int64_t>(size);
  CHECK(moodist::wrapperApi.tensorEmpty != nullptr);
  Tensor* t = moodist::wrapperApi.tensorEmpty(&shape, 1, DType::UInt8, -1);
  CHECK(t != nullptr);
  // Copy serialized data into tensor
  void* tensorData = moodist::wrapperApi.tensorDataPtr(t);
  std::memcpy(tensorData, buffer->data(), size);
  return TensorPtr(t);
}

// Helper to deserialize from TensorPtr
template<typename... T>
void deserializeFromTensorPtr(const TensorPtr& tensor, T&... result) {
  void* ptr = tensor.data_ptr();
  size_t size = static_cast<size_t>(tensor.numel());
  deserializeBuffer(ptr, size, result...);
}

// TInput structure for serializing distributed input information
template<typename T>
struct TInput {
  bool contiguous = false;
  T offset;
  T shape;
  uint32_t inputRank;
  uint32_t outputRank;
  uint32_t inputIndex;
  uint32_t outputIndex;
  size_t inputOffset;
  size_t outputOffset;
  bool outputContiguous;

  template<typename X>
  void serialize(X& x) {
    x(contiguous, offset, shape, inputRank, outputRank, inputIndex, outputIndex, inputOffset, outputOffset,
        outputContiguous);
  }
};

// Packet for exchanging reads among local ranks
struct LocalReadsPacket {
  size_t localRankIndex;
  IVector<CustomOpDescriptor::Read> reads;

  template<typename X>
  void serialize(X& x) {
    x(localRankIndex, reads);
  }
};

// Compute NVLink-optimized read plan by splitting overlapping reads
// into sub-regions for maximum local sharing
void computeNvlinkPlan(CustomOpDescriptor* op, const IVector<IVector<CustomOpDescriptor::Read>>& allLocalReads,
    const Vector<size_t>& nodeRanks, // global ranks on my node
    size_t myLocalRank, size_t numLocalRanks,
    const Vector<size_t>& rankLocalRank, // maps global rank -> local rank index
    size_t myGlobalRank) {
  using Read = CustomOpDescriptor::Read;

  // Helper to check if a rank is local
  auto isLocalRank = [&](size_t r) {
    for (size_t lr : nodeRanks) {
      if (lr == r) {
        return true;
      }
    }
    return false;
  };

  // Track which read each reference came from
  struct ReadRef {
    size_t localRankIdx;
    const Read* read;
  };

  // Group reads by (sourceRank, inputIndex)
  HashMap<std::pair<uint32_t, uint32_t>, IVector<ReadRef>, PairHash> readsBySource;

  for (size_t localIdx : range(numLocalRanks)) {
    for (const Read& r : allLocalReads[localIdx]) {
      auto key = std::make_pair(r.rank, r.inputIndex);
      readsBySource[key].push_back({localIdx, &r});
    }
  }

  // Process each source group
  for (auto& [sourceKey, refs] : readsBySource) {
    auto [sourceRank, inputIndex] = sourceKey;

    // Check if source is on the same node
    bool sourceIsLocal = isLocalRank(sourceRank);

    // Collect all interval boundaries
    IVector<size_t> boundaries;
    for (const auto& ref : refs) {
      boundaries.push_back(ref.read->inputOffset);
      boundaries.push_back(ref.read->inputOffset + ref.read->bytes);
    }
    std::ranges::sort(boundaries);
    auto [newEnd, oldEnd] = std::ranges::unique(boundaries);
    boundaries.erase(newEnd, boundaries.end());

    // For each interval, find which reads contain it
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
      size_t intervalStart = boundaries[i];
      size_t intervalEnd = boundaries[i + 1];
      size_t intervalBytes = intervalEnd - intervalStart;

      if (intervalBytes == 0) {
        continue;
      }

      // Find reads that contain this interval
      IVector<ReadRef> containingRefs;
      for (const auto& ref : refs) {
        size_t readStart = ref.read->inputOffset;
        size_t readEnd = readStart + ref.read->bytes;
        if (readStart <= intervalStart && intervalEnd <= readEnd) {
          containingRefs.push_back(ref);
        }
      }

      if (containingRefs.empty()) {
        continue;
      }

      // Check if I have a read in this interval
      const ReadRef* myRef = nullptr;
      for (const auto& ref : containingRefs) {
        if (ref.localRankIdx == myLocalRank) {
          myRef = &ref;
          break;
        }
      }

      if (!myRef) {
        continue; // I don't need this interval
      }

      // Compute my sub-read's output offset
      size_t myOutputOffset = myRef->read->outputOffset + (intervalStart - myRef->read->inputOffset);

      if (sourceIsLocal) {
        // Source is on same node - copy directly from source's input via NVLink
        // (skip if source is myself - that's handled by inputCopies)
        if (sourceRank != myGlobalRank) {
          CustomOpDescriptor::LocalInputCopy lic;
          lic.sourceRank = sourceRank;
          lic.sourceInputIndex = inputIndex;
          lic.sourceInputOffset = intervalStart;
          lic.bytes = intervalBytes;
          lic.myOutputIndex = myRef->read->outputIndex;
          lic.myOutputOffset = myOutputOffset;
          op->localInputCopies.push_back(lic);
        }
      } else if (containingRefs.size() == 1) {
        // Only I need this interval from remote - direct IB read
        Read r;
        r.rank = sourceRank;
        r.inputIndex = inputIndex;
        r.inputOffset = intervalStart;
        r.bytes = intervalBytes;
        r.outputIndex = myRef->read->outputIndex;
        r.outputOffset = myOutputOffset;
        op->directReads.push_back(r);
      } else {
        // Multiple local ranks need this interval from remote
        // Pick gateway from among those who need it, preferring rail-aligned
        size_t sourceLocalRank = rankLocalRank[sourceRank];
        size_t preferredGatewayLocalIdx = sourceLocalRank % numLocalRanks;

        // Find if preferred gateway is among readers, otherwise use first
        size_t actualGatewayLocalIdx = containingRefs[0].localRankIdx;
        for (const auto& ref : containingRefs) {
          if (ref.localRankIdx == preferredGatewayLocalIdx) {
            actualGatewayLocalIdx = preferredGatewayLocalIdx;
            break;
          }
        }
        size_t gatewayGlobalRank = nodeRanks[actualGatewayLocalIdx];

        if (myLocalRank == actualGatewayLocalIdx) {
          // I'm the gateway - fetch via IB
          CustomOpDescriptor::GatewayRead gr;
          gr.sourceRank = sourceRank;
          gr.inputIndex = inputIndex;
          gr.inputOffset = intervalStart;
          gr.bytes = intervalBytes;
          gr.outputIndex = myRef->read->outputIndex;
          gr.outputOffset = myOutputOffset;
          op->gatewayReads.push_back(gr);
        } else {
          // I receive from gateway's output via NVLink
          const ReadRef* gatewayRef = nullptr;
          for (const auto& ref : containingRefs) {
            if (ref.localRankIdx == actualGatewayLocalIdx) {
              gatewayRef = &ref;
              break;
            }
          }
          CHECK(gatewayRef != nullptr);

          size_t gatewayOutputOffset = gatewayRef->read->outputOffset + (intervalStart - gatewayRef->read->inputOffset);

          CustomOpDescriptor::LocalCopy lc;
          lc.gatewayRank = static_cast<uint32_t>(gatewayGlobalRank);
          lc.gatewayOutputIndex = gatewayRef->read->outputIndex;
          lc.gatewayOutputOffset = gatewayOutputOffset;
          lc.bytes = intervalBytes;
          lc.myOutputIndex = myRef->read->outputIndex;
          lc.myOutputOffset = myOutputOffset;
          op->localCopies.push_back(lc);
        }
      }
    }
  }
}

} // namespace

// ============================================================================
// ProcessGroupImpl::compileOpFullImpl_OLD - template implementation
// OLD: Kept for reference during refactoring. Will be deleted once new impl is working.
// ============================================================================

template<int ndim>
SharedPtr<CustomOpImpl> ProcessGroupImpl::compileOpFullImpl_OLD(std::span<const int64_t> shape_arg, DType dtype,
    std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> inputs_arg,
    std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> outputs_arg) {
  using T = std::array<int64_t, ndim>;
  T shape;
  std::ranges::copy(shape_arg, shape.begin());

  struct TDescr {
    uint32_t rank;
    uint32_t index;
    T shape;
    T offset;
    size_t numel;
  };

  const size_t numInputs = inputs_arg.size();
  const size_t numOutputs = outputs_arg.size();

  constexpr auto numel = [](const T& v) {
    size_t n = 1;
    for (int64_t x : v) {
      n *= x;
    }
    return n;
  };

  constexpr auto linearOffset = [](const T& offset, const T& shape) {
    size_t result = 0;
    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      result += offset[i] * stride;
      stride *= shape[i];
    }
    return result;
  };

  IVector<TDescr> descrs;
  descrs.reserve(numInputs + numOutputs);
  IVector<TDescr*> inputs(numInputs);
  IVector<TDescr*> outputs(numOutputs);
  IVector<size_t> indexCounter(size);

  for (size_t i = 0; i < numInputs; ++i) {
    const auto& [r, off, nshape] = inputs_arg[i];
    descrs.emplace_back();
    TDescr& d = descrs.back();
    d.rank = r;
    std::ranges::copy(off, d.offset.begin());
    std::ranges::copy(nshape, d.shape.begin());
    d.numel = numel(d.shape);
    d.index = indexCounter[r]++;
    inputs[i] = &d;
  }

  std::ranges::fill(indexCounter, 0);
  for (size_t i = 0; i < numOutputs; ++i) {
    const auto& [r, off, nshape] = outputs_arg[i];
    descrs.emplace_back();
    TDescr& d = descrs.back();
    d.rank = r;
    std::ranges::copy(off, d.offset.begin());
    std::ranges::copy(nshape, d.shape.begin());
    d.numel = numel(d.shape);
    d.index = indexCounter[r]++;
    outputs[i] = &d;
  }

  auto ref = [&](auto& v) {
    return std::views::transform(v, [](auto& x) -> decltype(auto) {
      if constexpr (std::is_pointer_v<std::remove_reference_t<decltype(x)>>) {
        return *x;
      } else {
        return x;
      }
    });
  };

  IVector<IVector<const TDescr*>> outputsPerRank(size);
  for (const TDescr& dst : ref(outputs)) {
    CHECK(dst.index == outputsPerRank[dst.rank].size());
    outputsPerRank[dst.rank].push_back(&dst);
  }
  for (size_t i = 0; i < size; ++i) {
    CHECK(outputsPerRank[i].size() == indexCounter[i]);
  }

  constexpr auto intersecting = [](const auto& a, const auto& b) {
    for (int i = 0; i < ndim; ++i) {
      if (a.offset[i] >= b.offset[i] + b.shape[i]) {
        return false;
      }
      if (b.offset[i] >= a.offset[i] + a.shape[i]) {
        return false;
      }
    }
    return true;
  };

  constexpr auto contiguous = [](const T& a, const T& b) {
    for (int i = 0; i < ndim; ++i) {
      if (i && a[i] != b[i]) {
        return false;
      }
    }
    return true;
  };

  struct FindResult {
    T offset;
    T shape;
    size_t index;
  };

  auto find = [&](const auto& dst, const auto& list) -> IVector<FindResult> {
    IVector<FindResult> rv;
    size_t i = 0;
    for (const auto& src : ref(list)) {
      size_t index = i++;
      if (!intersecting(src, dst)) {
        continue;
      }
      T low, high, nshape;
      for (int j = 0; j < ndim; ++j) {
        low[j] = std::max(src.offset[j], dst.offset[j]);
        high[j] = std::min(src.offset[j] + src.shape[j], dst.offset[j] + dst.shape[j]);
        CHECK(low[j] <= high[j]);
        nshape[j] = high[j] - low[j];
      }
      FindResult r;
      r.offset = low;
      r.shape = nshape;
      r.index = index;
      rv.push_back(r);
    }
    return rv;
  };

  using Input = TInput<T>;
  IVector<Input> distInputs;
  IVector<FindResult> copyInputs;

  IVector<TDescr*> myInputs;
  for (auto& v : ref(inputs)) {
    if (v.rank == rank) {
      myInputs.push_back(&v);
    }
  }

  constexpr auto sub = [](const T& a, const T& b) {
    T r;
    for (int i = 0; i < ndim; ++i) {
      r[i] = a[i] - b[i];
    }
    return r;
  };

  for (const TDescr& dst : ref(outputs)) {
    if (dst.numel == 0) {
      continue;
    }
    auto r = find(dst, myInputs);

    for (const FindResult& x : r) {
      auto& src = *myInputs[x.index];
      bool inputContiguous = contiguous(x.shape, src.shape);

      if (!inputContiguous) {
        auto rr = find(dst, copyInputs);
        if (!rr.empty() && rr.size() < 4) {
          size_t n = 0;
          bool cc = true;
          for (auto& x2 : rr) {
            n += numel(x2.shape);
            cc &= contiguous(x2.shape, copyInputs[x2.index].shape);
            if (!cc) {
              break;
            }
          }
          if (cc && n == numel(x.shape)) {
            for (auto& x2 : rr) {
              distInputs.emplace_back();
              Input& inp = distInputs.back();
              inp.offset = x2.offset;
              inp.shape = x2.shape;
              inp.inputRank = rank;
              inp.outputRank = dst.rank;
              inp.inputIndex = myInputs.size() + x2.index;
              inp.outputIndex = dst.index;
              CHECK(inp.inputIndex < myInputs.size() + copyInputs.size());
              CHECK(&dst == outputsPerRank[inp.outputRank][inp.outputIndex]);
              inp.inputOffset = linearOffset(sub(x2.offset, copyInputs[x2.index].offset), copyInputs[x2.index].shape);
              inp.outputOffset = linearOffset(sub(x2.offset, dst.offset), dst.shape);
              inp.outputContiguous = contiguous(x2.shape, dst.shape);
            }
            continue;
          }
        }
      }

      distInputs.emplace_back();
      Input& inp = distInputs.back();
      inp.offset = x.offset;
      inp.shape = x.shape;
      inp.inputRank = rank;
      inp.outputRank = dst.rank;
      inp.inputIndex = x.index;
      inp.outputIndex = dst.index;
      CHECK(inp.inputIndex < myInputs.size());
      CHECK(&dst == outputsPerRank[inp.outputRank][inp.outputIndex]);
      inp.inputOffset = linearOffset(sub(x.offset, src.offset), src.shape);
      inp.outputOffset = linearOffset(sub(x.offset, dst.offset), dst.shape);

      if (!inputContiguous) {
        inp.inputIndex = myInputs.size() + copyInputs.size();
        inp.inputOffset = 0;
        copyInputs.push_back(x);
      }
      inp.outputContiguous = contiguous(x.shape, dst.shape);
    }
  }

  auto op = std::make_shared<CustomOpDescriptor>();
  op->id = nextCustomOpId++;
  op->dtype = dtype;

  size_t itemsize = wrapperApi.dtypeSize(dtype);

  for (auto& x : ref(inputs)) {
    if (x.rank == rank) {
      op->inputs.push_back(itemsize * numel(x.shape));
      op->inputShapes.emplace_back(x.shape.begin(), x.shape.end());
    }
  }
  for (auto& x : ref(outputs)) {
    if (x.rank == rank) {
      op->outputs.push_back(itemsize * numel(x.shape));
      op->outputShapes.emplace_back(x.shape.begin(), x.shape.end());
    }
  }

  // Barrier before queue operations
  barrier();

  // Create internal queues for compileOpFull communication
  if (queues.empty()) {
    for (size_t i = 0; i < size; ++i) {
      auto handle = makeQueue(group, i, false, {});
      CHECK(handle.get() != nullptr);
      // Transfer ownership from ApiHandle to SharedPtr
      Queue* q = static_cast<Queue*>(handle.release());
      CHECK(q->impl != nullptr);
      queues.push_back(SharedPtr<Queue>(q));
    }
  }
  CHECK(queues.size() == size);

  // Exchange distribution info via queues
  IVector<IVector<Input>> s(size);
  for (const Input& x : distInputs) {
    s[x.outputRank].push_back(x);
  }

  for (size_t i = 0; i < size; ++i) {
    CHECK(i < queues.size());
    CHECK(queues[i].get() != nullptr);
    CHECK(queues[i]->impl != nullptr);
    TensorPtr tensor = serializeToTensorPtr(s[i]);
    queues[i]->put(std::move(tensor), 0);
  }

  // Step 1: Gather inputs that matched our outputs (from all source ranks)
  IVector<Input> matchedInputs;
  for (size_t i = 0; i < size; ++i) {
    CHECK(rank < queues.size());
    CHECK(queues[rank].get() != nullptr);
    CHECK(queues[rank]->impl != nullptr);
    auto [tensor, qsize] = queues[rank]->get();
    IVector<Input> n;
    deserializeFromTensorPtr(tensor, n);

    for (const Input& x : n) {
      CHECK(x.outputRank == rank);
      matchedInputs.push_back(x);
    }
  }

  // Debug: print matched inputs and outputs
  log.debug(
      "compile_op_full: rank %zu, %zu matched inputs, %zu outputs\n", rank, matchedInputs.size(), op->outputs.size());
  for (size_t i : indices(matchedInputs)) {
    const auto& x = matchedInputs[i];
    log.debug("  input[%zu]: from rank %u, outputIndex=%u, offset=%s, shape=%s, outputOffset=%zu, contiguous=%d\n", i,
        x.inputRank, x.outputIndex, fmt::to_string(fmt::join(x.offset, ",")).c_str(),
        fmt::to_string(fmt::join(x.shape, ",")).c_str(), x.outputOffset, x.outputContiguous);
  }
  for (size_t i : indices(op->outputs)) {
    log.debug("  output[%zu]: %zu bytes, shape=%s\n", i, op->outputs[i],
        fmt::to_string(fmt::join(op->outputShapes[i], ",")).c_str());
  }

  // Step 2: Validate inputs - check for overlaps using N-dim intersection
  {
    // Group inputs by output index
    IVector<IVector<size_t>> inputsPerOutput(op->outputs.size());
    for (size_t i : indices(matchedInputs)) {
      inputsPerOutput[matchedInputs[i].outputIndex].push_back(i);
    }

    // Check for overlapping inputs within each output
    std::string overlapDetails;
    size_t overlapCount = 0;
    constexpr size_t maxReportedRegions = 5;

    for (size_t outIdx : indices(op->outputs)) {
      const auto& idxs = inputsPerOutput[outIdx];
      for (size_t i = 0; i < idxs.size(); ++i) {
        for (size_t j = i + 1; j < idxs.size(); ++j) {
          const auto& a = matchedInputs[idxs[i]];
          const auto& b = matchedInputs[idxs[j]];
          if (intersecting(a, b)) {
            overlapCount++;
            if (overlapCount <= maxReportedRegions) {
              overlapDetails += fmt::sprintf("\n  output[%zu]: overlap between input from rank %u at [%s] shape [%s] "
                                             "and input from rank %u at [%s] shape [%s]",
                  outIdx, a.inputRank, fmt::to_string(fmt::join(a.offset, ",")).c_str(),
                  fmt::to_string(fmt::join(a.shape, ",")).c_str(), b.inputRank,
                  fmt::to_string(fmt::join(b.offset, ",")).c_str(), fmt::to_string(fmt::join(b.shape, ",")).c_str());
            }
          }
        }
      }
    }

    if (overlapCount > 0) {
      std::string msg = "moodist.compile_op_full: overlapping inputs detected";
      if (overlapCount > maxReportedRegions) {
        msg += fmt::sprintf(" (%zu regions, showing first %zu):", overlapCount, maxReportedRegions);
      } else {
        msg += ":";
      }
      msg += overlapDetails;
      throw std::runtime_error(msg);
    }

    // Gap detection: first do quick numel check, then detailed cell decomposition if needed
    IVector<size_t> outputsWithGaps;
    for (size_t outIdx : indices(op->outputs)) {
      size_t inputNumel = 0;
      for (size_t i : inputsPerOutput[outIdx]) {
        inputNumel += numel(matchedInputs[i].shape);
      }
      size_t outputNumel = op->outputs[outIdx] / itemsize;
      if (inputNumel != outputNumel) {
        outputsWithGaps.push_back(outIdx);
      }
    }

    if (!outputsWithGaps.empty()) {
      // Detailed gap detection using cell decomposition
      std::string gapDetails;
      size_t gapCount = 0;

      for (size_t outIdx : outputsWithGaps) {
        const auto& outDescr = *outputsPerRank[rank][outIdx];
        const auto& idxs = inputsPerOutput[outIdx];

        // Collect boundaries in each dimension
        std::array<IVector<int64_t>, ndim> boundaries;
        for (int d = 0; d < ndim; ++d) {
          boundaries[d].push_back(outDescr.offset[d]);
          boundaries[d].push_back(outDescr.offset[d] + outDescr.shape[d]);
          for (size_t i : idxs) {
            const auto& inp = matchedInputs[i];
            boundaries[d].push_back(inp.offset[d]);
            boundaries[d].push_back(inp.offset[d] + inp.shape[d]);
          }
          std::sort(boundaries[d].begin(), boundaries[d].end());
          boundaries[d].erase(std::unique(boundaries[d].begin(), boundaries[d].end()), boundaries[d].end());
        }

        // Compute total number of cells
        std::array<size_t, ndim> numIntervals;
        size_t totalCells = 1;
        for (int d = 0; d < ndim; ++d) {
          numIntervals[d] = boundaries[d].size() - 1;
          totalCells *= numIntervals[d];
        }

        // Iterate through all cells
        for (size_t cellIdx = 0; cellIdx < totalCells; ++cellIdx) {
          // Convert cellIdx to multi-dimensional index and compute cell offset/shape
          T cellOffset, cellShape;
          size_t remaining = cellIdx;
          for (int d = ndim - 1; d >= 0; --d) {
            size_t intervalIdx = remaining % numIntervals[d];
            remaining /= numIntervals[d];
            cellOffset[d] = boundaries[d][intervalIdx];
            cellShape[d] = boundaries[d][intervalIdx + 1] - boundaries[d][intervalIdx];
          }

          // Check if cell is inside output
          bool insideOutput = true;
          for (int d = 0; d < ndim; ++d) {
            if (cellOffset[d] < outDescr.offset[d] ||
                cellOffset[d] + cellShape[d] > outDescr.offset[d] + outDescr.shape[d]) {
              insideOutput = false;
              break;
            }
          }
          if (!insideOutput) {
            continue;
          }

          // Check if cell is covered by any input
          bool covered = false;
          for (size_t i : idxs) {
            const auto& inp = matchedInputs[i];
            bool inputCovers = true;
            for (int d = 0; d < ndim; ++d) {
              if (cellOffset[d] < inp.offset[d] || cellOffset[d] + cellShape[d] > inp.offset[d] + inp.shape[d]) {
                inputCovers = false;
                break;
              }
            }
            if (inputCovers) {
              covered = true;
              break;
            }
          }

          if (!covered) {
            gapCount++;
            if (gapCount <= maxReportedRegions) {
              gapDetails += fmt::sprintf("\n  output[%zu]: missing region at [%s] shape [%s]", outIdx,
                  fmt::to_string(fmt::join(cellOffset, ",")).c_str(),
                  fmt::to_string(fmt::join(cellShape, ",")).c_str());
            }
          }
        }
      }

      std::string msg = "moodist.compile_op_full: missing input coverage";
      if (gapCount > maxReportedRegions) {
        msg += fmt::sprintf(" (%zu regions, showing first %zu):", gapCount, maxReportedRegions);
      } else {
        msg += ":";
      }
      msg += gapDetails;
      throw std::runtime_error(msg);
    }
  }

  // Step 3: Generate Read/Copy from validated inputs
  for (const Input& x : matchedInputs) {
    CustomOpDescriptor::Read r;
    r.rank = x.inputRank;
    r.inputIndex = x.inputIndex;
    r.outputIndex = x.outputIndex;
    r.inputOffset = itemsize * x.inputOffset;
    r.outputOffset = itemsize * x.outputOffset;
    size_t num = numel(x.shape);
    r.bytes = itemsize * num;

    if (!x.outputContiguous) {
      CustomOpDescriptor::Copy c;
      c.index = r.outputIndex;
      auto off = sub(x.offset, outputsPerRank[rank][x.outputIndex]->offset);
      c.offset = {off.begin(), off.end()};
      c.shape = {x.shape.begin(), x.shape.end()};
      r.outputIndex = op->outputs.size() + op->outputCopies.size();
      op->outputCopies.push_back(c);
      r.outputOffset = 0;
    }

    op->reads.push_back(r);
  }

  // NVLink optimization: exchange reads among local ranks and compute plan
  size_t myNodeIndex = group->rankToNodeIndex[rank];
  auto& myNode = group->nodeRanks[myNodeIndex];
  size_t myLocalRank = group->rankLocalRank[rank];
  size_t numLocalRanks = myNode.size();

  if (false && numLocalRanks > 1) {
    // Exchange reads among local ranks
    LocalReadsPacket myPacket{myLocalRank, op->reads};
    TensorPtr myPacketTensor = serializeToTensorPtr(myPacket);

    // Put to all local ranks except self
    for (size_t lr : myNode) {
      if (lr != rank) {
        queues[lr]->put(TensorPtr(myPacketTensor), 0);
      }
    }

    // Collect from all local ranks
    IVector<IVector<CustomOpDescriptor::Read>> allLocalReads(numLocalRanks);
    allLocalReads[myLocalRank] = op->reads;

    for (size_t i = 1; i < numLocalRanks; ++i) {
      auto [tensor, qsize] = queues[rank]->get();
      LocalReadsPacket packet;
      deserializeFromTensorPtr(tensor, packet);
      allLocalReads[packet.localRankIndex] = std::move(packet.reads);
    }

    // Compute the NVLink-optimized plan (for future use and logging)
    // Note: we keep op->reads intact for now - execution still uses it
    computeNvlinkPlan(op.get(), allLocalReads, myNode, myLocalRank, numLocalRanks, group->rankLocalRank, rank);
  } else {
    // Single rank on node - all reads are direct
    op->directReads = op->reads; // copy, don't move - execution still uses op->reads
  }

  // Log the NVLink plan
  log.debug("compile_op plan: rank %zu, directReads=%zu, gatewayReads=%zu, localCopies=%zu, localInputCopies=%zu\n",
      rank, op->directReads.size(), op->gatewayReads.size(), op->localCopies.size(), op->localInputCopies.size());
  for (const auto& r : op->directReads) {
    log.debug("  directRead: src=%u input=%u offset=%zu bytes=%zu -> output=%u offset=%zu\n", r.rank, r.inputIndex,
        r.inputOffset, r.bytes, r.outputIndex, r.outputOffset);
  }
  for (const auto& r : op->gatewayReads) {
    log.debug("  gatewayRead: src=%u input=%u offset=%zu bytes=%zu -> output=%u offset=%zu\n", r.sourceRank,
        r.inputIndex, r.inputOffset, r.bytes, r.outputIndex, r.outputOffset);
  }
  for (const auto& lc : op->localCopies) {
    log.debug("  localCopy: gateway=%u gatewayOut=%u offset=%zu bytes=%zu -> myOut=%u offset=%zu\n", lc.gatewayRank,
        lc.gatewayOutputIndex, lc.gatewayOutputOffset, lc.bytes, lc.myOutputIndex, lc.myOutputOffset);
  }
  for (const auto& lic : op->localInputCopies) {
    log.debug("  localInputCopy: src=%u srcInput=%u offset=%zu bytes=%zu -> myOut=%u offset=%zu\n", lic.sourceRank,
        lic.sourceInputIndex, lic.sourceInputOffset, lic.bytes, lic.myOutputIndex, lic.myOutputOffset);
  }

  // Barrier after queue operations
  barrier();

  for (auto& x : copyInputs) {
    CustomOpDescriptor::Copy c;
    c.index = x.index;
    auto off = sub(x.offset, myInputs[x.index]->offset);
    c.offset = {off.begin(), off.end()};
    c.shape = {x.shape.begin(), x.shape.end()};
    op->inputCopies.push_back(c);
  }

  // Create the CustomOpImpl with a closure that calls customOp
  auto result = makeShared<CustomOpImpl>();
  auto selfPtr = share(this);
  result->call = [op, selfPtr](TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs,
                     CUstream stream) -> SharedPtr<ApiFuture> {
    return selfPtr->customOp(op, inputs, nInputs, outputs, nOutputs, stream);
  };

  return result;
}

// ============================================================================
// ProcessGroupImpl::compileOpFull - compile a distributed tensor operation
// ============================================================================

SharedPtr<CustomOpImpl> ProcessGroupImpl::compileOpFull(DType dtype, std::span<const api::TensorRegion> inputsVec,
    std::span<const api::TensorRegion> outputsVec, api::ReduceOp reduce, bool cpuSync) {

  // Ensure queues exist for compile_op communication
  if (queues.empty()) {
    for (size_t i = 0; i < size; ++i) {
      auto handle = makeQueue(group, i, false, {});
      CHECK(handle.get() != nullptr);
      Queue* q = static_cast<Queue*>(handle.release());
      CHECK(q->impl != nullptr);
      queues.push_back(SharedPtr<Queue>(q));
    }
  }
  CHECK(queues.size() == size);

  // Build compile context
  compile_op::CompileContext ctx;
  ctx.group = group.get();
  ctx.queues = std::span<SharedPtr<Queue>>(queues.data(), queues.size());
  ctx.barrier = [this]() {
    barrier();
  };
  ctx.nextOpId = &nextCustomOpId;

  // Call compile_op
  auto op = compile_op::compile(ctx, dtype, inputsVec, outputsVec, reduce, cpuSync);

  // Wrap CustomOpDescriptor in CustomOpImpl with execution closure
  auto result = makeShared<CustomOpImpl>();
  auto selfPtr = share(this);
  result->call = [op, selfPtr](TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs,
                     CUstream stream) -> SharedPtr<ApiFuture> {
    return selfPtr->customOp(op, inputs, nInputs, outputs, nOutputs, stream);
  };

  return result;
}

// ============================================================================
// ProcessGroupImpl::customOp - execute a compiled custom operation
// ============================================================================

SharedPtr<ApiFuture> ProcessGroupImpl::customOp(std::shared_ptr<CustomOpDescriptor> op, TensorPtr* inputs,
    size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  if (nInputs != op->inputs.size()) {
    throw std::runtime_error("moodist: custom op expected different number of inputs");
  }
  if (nOutputs != op->outputs.size()) {
    throw std::runtime_error("moodist: custom op expected different number of outputs");
  }

  auto formatShape = [](const auto& shape) {
    return fmt::sprintf("[%s]", fmt::to_string(fmt::join(shape, ", ")));
  };

  auto formatDType = [](int dtype) -> const char* {
    switch (dtype) {
    case 0:
      return "UInt8";
    case 1:
      return "Int8";
    case 2:
      return "Int16";
    case 3:
      return "Int32";
    case 4:
      return "Int64";
    case 5:
      return "Float16";
    case 6:
      return "Float32";
    case 7:
      return "Float64";
    case 11:
      return "Bool";
    case 15:
      return "BFloat16";
    default:
      return "Unknown";
    }
  };

  // Validate inputs and outputs, build TensorDataPtr vectors
  std::vector<TensorDataPtr> inputTDs;
  std::vector<TensorDataPtr> outputTDs;

  for (size_t i = 0; i < nInputs; ++i) {
    auto t = getTensorDataFromPtr(inputs[i], group.get());
    REQUIRE(t->dtype == static_cast<int>(op->dtype),
        "moodist: custom op input[%zu] has wrong dtype: got %s, expected %s", i, formatDType(t->dtype),
        formatDType(static_cast<int>(op->dtype)));
    REQUIRE(t->shape == op->inputShapes[i], "moodist: custom op input[%zu] has wrong shape: got %s, expected %s", i,
        formatShape(t->shape), formatShape(op->inputShapes[i]));
    REQUIRE(t->bytes() == op->inputs[i],
        "moodist: custom op input[%zu] has wrong size: got %zu bytes, expected %zu bytes", i, t->bytes(),
        op->inputs[i]);
    bool expectedCuda = op->inputDevices[i] == DeviceType::CUDA;
    REQUIRE(t->isCuda == expectedCuda, "moodist: custom op input[%zu] has wrong device: got %s, expected %s", i,
        t->isCuda ? "cuda" : "cpu", expectedCuda ? "cuda" : "cpu");
    inputTDs.push_back(std::move(t));
  }
  for (size_t i = 0; i < nOutputs; ++i) {
    auto t = getTensorDataFromPtr(outputs[i], group.get());
    REQUIRE(t->dtype == static_cast<int>(op->dtype),
        "moodist: custom op output[%zu] has wrong dtype: got %s, expected %s", i, formatDType(t->dtype),
        formatDType(static_cast<int>(op->dtype)));
    REQUIRE(t->shape == op->outputShapes[i], "moodist: custom op output[%zu] has wrong shape: got %s, expected %s", i,
        formatShape(t->shape), formatShape(op->outputShapes[i]));
    REQUIRE(t->bytes() == op->outputs[i],
        "moodist: custom op output[%zu] has wrong size: got %zu bytes, expected %zu bytes", i, t->bytes(),
        op->outputs[i]);
    bool expectedCuda = op->outputDevices[i] == DeviceType::CUDA;
    REQUIRE(t->isCuda == expectedCuda, "moodist: custom op output[%zu] has wrong device: got %s, expected %s", i,
        t->isCuda ? "cuda" : "cpu", expectedCuda ? "cuda" : "cpu");
    outputTDs.push_back(std::move(t));
  }

  // Fast path: all-local, all-CUDA, no copies — skip CPU thread entirely
  if (op->allLocal) {
    return executeLocalOnly(
        op, inputTDs, outputTDs, inputs, nInputs, outputs, nOutputs, concurrencyIndex, stepValue, stream);
  }

  // Slow path: dispatch to CPU thread
  auto future = FutureImplSharedPtr::make();

  StreamData& sd = group->getCpuStreamData(concurrencyIndex);
  QueueEntryCustom* e = group->cpuThread->freelistCustom.pop();
  e->task = taskCustom;
  e->stepValue = stepValue;
  e->concurrencyIndex = concurrencyIndex;
  e->sd = &sd;
  e->op = op;
  e->future = future;
  bool anyCuda = false;
  bool anyCpu = false;

  CHECK(e->inputs.empty());
  CHECK(e->outputs.empty());

  for (auto& t : inputTDs) {
    anyCuda |= t->isCuda;
    anyCpu |= !t->isCuda;
    e->inputs.push_back(std::move(t));
  }
  for (auto& t : outputTDs) {
    anyCuda |= t->isCuda;
    anyCpu |= !t->isCuda;
    e->outputs.push_back(std::move(t));
  }

  // Create and return ApiFuture early so we can store tensors in it
  auto result = makeShared<ApiFuture>();

  // Handle inputCopies: narrow input tensors to required slices and make contiguous
  for (const auto& x : op->inputCopies) {
    CHECK(x.index < nInputs);
    TensorPtr t = inputs[x.index];
    CHECK(x.offset.size() == x.shape.size());
    for (size_t i = 0; i < x.offset.size(); ++i) {
      t = t.narrow(i, x.offset[i], x.shape[i]);
    }
    t = t.contiguous();
    auto td = getTensorDataFromPtr(t, group.get());
    anyCuda |= td->isCuda;
    anyCpu |= !td->isCuda;
    e->inputs.push_back(std::move(td));
    // Keep tensor alive
    result->holdTensors.push_back(std::move(t));
  }

  // Handle outputCopies: create temporary tensors that will be copied back after completion
  Vector<std::pair<TensorPtr, size_t>> outputCopyTensors; // (temp tensor, original output index)
  for (const auto& x : op->outputCopies) {
    CHECK(x.index < nOutputs);
    // Create empty tensor with the slice shape on the same device as the output
    std::vector<int64_t> shape64(x.shape.begin(), x.shape.end());
    TensorPtr t = TensorPtr::empty(shape64, op->dtype, outputs[x.index].device_index());
    outputCopyTensors.emplace_back(t, x.index);
    auto td = getTensorDataFromPtr(t, group.get());
    anyCuda |= td->isCuda;
    anyCpu |= !td->isCuda;
    e->outputs.push_back(std::move(td));
    // Keep tensor alive
    result->holdTensors.push_back(std::move(t));
  }

  // Force CPU sync if requested by the compiled op
  if (op->cpuSync) {
    anyCpu = true;
  }

  e->anyCuda = anyCuda;
  e->anyCpu = anyCpu;

  if (anyCuda) {
    memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue);
    memFlush(stream);
  }

  if (!anyCpu) {
    future->done = 1;
  }

  group->cpuThread->enqueue(e);

  result->impl = std::move(future);
  // Hold original tensors alive
  for (size_t i = 0; i < nInputs; ++i) {
    result->holdTensors.push_back(inputs[i]);
  }
  for (size_t i = 0; i < nOutputs; ++i) {
    result->holdTensors.push_back(outputs[i]);
  }

  if (!outputCopyTensors.empty()) {
    // Need to copy temporary outputs back to original output tensors after completion
    // Capture copies of the output TensorPtrs since the outputs pointer may be invalid later
    std::vector<TensorPtr> outputsCopy(outputs, outputs + nOutputs);
    auto selfPtr = share(this);
    result->waitDoneCallback = [selfPtr, concurrencyIndex, stepValue, anyCuda,
                                   copyTensors = std::move(outputCopyTensors), op,
                                   outputsCopy = std::move(outputsCopy)]() {
      if (anyCuda) {
        selfPtr->memWaitGeq(selfPtr->group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
        selfPtr->memFlush(wrapperApi.cudaGetCurrentStream());
      }
      // Copy temporary tensors back to the appropriate slices of original outputs
      CHECK(copyTensors.size() == op->outputCopies.size());
      for (size_t i = 0; i < op->outputCopies.size(); ++i) {
        const auto& [src, outputIdx] = copyTensors[i];
        const auto& x = op->outputCopies[i];
        CHECK(outputIdx < outputsCopy.size());
        TensorPtr dst = outputsCopy[outputIdx];
        for (size_t j = 0; j < x.offset.size(); ++j) {
          dst = dst.narrow(j, x.offset[j], x.shape[j]);
        }
        dst.copy_(src);
      }
    };
  } else if (anyCuda) {
    auto selfPtr = share(this);
    result->waitDoneCallback = [selfPtr, concurrencyIndex, stepValue]() {
      selfPtr->memWaitGeq(selfPtr->group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
      selfPtr->memFlush(wrapperApi.cudaGetCurrentStream());
    };
  }

  return result;
}

// ============================================================================
// ProcessGroupImpl::executeLocalOnly - fast path for all-local custom ops
// ============================================================================

SharedPtr<ApiFuture> ProcessGroupImpl::executeLocalOnly(std::shared_ptr<CustomOpDescriptor> op,
    std::vector<TensorDataPtr>& inputs, std::vector<TensorDataPtr>& outputs, TensorPtr* inputPtrs, size_t nInputs,
    TensorPtr* outputPtrs, size_t nOutputs, uint32_t concurrencyIndex, uint32_t stepValue, CUstream stream) {

  EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
  StreamGuard sg(stream, group->deviceIndex);

  IpcMapper* ipcMapper = &*group->ipcMapper;
  const auto& peerIndices = group->peerIndices;

  std::shared_lock unmapLock(unmapMemoryMutex);
  sync(stepValue);

  int kernelVersion = group->compileOpKernels->version;

  // ==========================================================================
  // Multicast path: O(1) delivery via NVSwitch multicast.
  // Scratch buffers are permanently bound — no per-call rebinding.
  // multicastSources/multicastDests are cleanly separated from localInputCopies.
  // ==========================================================================
  if ((!op->multicastSources.empty() || !op->multicastDests.empty()) && kernelVersion == 3) {
    // Lazy compile of multicast kernel
    if (!group->compileOpKernels->cuMulticastKernel) {
      group->compileOpKernels->compileMulticast();
    }

    // Peer sync for collective mismatch detection
    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(concurrencyIndex, peerIndex, opTypeCompileOpLocal, stepValue, 0, 0);
    }
    for (size_t peerIndex : peerIndices) {
      peerWaitDyn(concurrencyIndex, peerIndex, opTypeCompileOpLocal, stepValue, 0);
    }

    freePendingIpcEvents();

    // Source writes input data to multicast VA.
    CompileOpCopyParameters params;
    params.stepValue = stepValue;
    params.concurrencyIndex = concurrencyIndex;
    params.numDescriptors = 0;
    params._pad = 0;

    for (const auto& ms : op->multicastSources) {
      if (ms.bytes == 0) {
        continue;
      }
      CHECK(params.numDescriptors < kMaxCopyDescriptors);
      auto& d = params.descriptors[params.numDescriptors++];
      d.src = inputs[ms.sourceInputIndex]->data() + ms.sourceInputOffset;
      d.dst = ms.mcVA;
      d.bytes = ms.bytes;
    }

    if (params.numDescriptors > 0) {
      std::array<void*, 1> kparams = {&params};
      CHECK_CU(cuLaunchKernel(group->compileOpKernels->cuMulticastKernel, group->compileOpKernels->gridSize, 1, 1,
          group->compileOpKernels->blockSize, 1, 1, 0, stream, kparams.data(), nullptr));
    }

    // Step M2: Copy from scratch VA to output tensors.
    // Data has arrived in each peer's scratch buffer via the multicast.
    for (const auto& md : op->multicastDests) {
      if (md.bytes == 0) {
        continue;
      }
      CHECK(md.myOutputIndex < outputs.size());
      uintptr_t dst = outputs[md.myOutputIndex]->data() + md.myOutputOffset;
      uintptr_t src = md.scratchVA;
      CHECK_CU(cuMemcpyDtoDAsync(dst, src, md.bytes, stream));
    }

    // Self-copies remaining in localInputCopies (sourceRank == rank).
    for (const auto& lic : op->localInputCopies) {
      if (lic.sourceRank != rank || lic.bytes == 0) {
        continue;
      }
      CHECK(lic.myOutputIndex < outputs.size());
      CHECK(lic.sourceInputIndex < inputs.size());
      uintptr_t dst = outputs[lic.myOutputIndex]->data() + lic.myOutputOffset;
      uintptr_t src = inputs[lic.sourceInputIndex]->data() + lic.sourceInputOffset;
      if (dst != src) {
        CHECK_CU(cuMemcpyDtoDAsync(dst, src, lic.bytes, stream));
      }
    }

    // Step M4: Return result
    auto result = makeShared<ApiFuture>();
    auto future = FutureImplSharedPtr::make();
    future->done = 1;
    result->impl = std::move(future);

    for (size_t i = 0; i < nInputs; ++i) {
      result->holdTensors.push_back(inputPtrs[i]);
    }
    for (size_t i = 0; i < nOutputs; ++i) {
      result->holdTensors.push_back(outputPtrs[i]);
    }

    return result;
  }

  // ==========================================================================
  // Copy path: O(N²) peer copies via kernel or cuMemcpy
  // ==========================================================================

  // Step 1: Self-copies (v0 path only — kernel path includes them in the launch)
  if (kernelVersion == 0) {
    for (const auto& lic : op->localInputCopies) {
      if (lic.sourceRank != rank) {
        continue;
      }
      CHECK(lic.myOutputIndex < outputs.size());
      CHECK(lic.sourceInputIndex < inputs.size());
      uintptr_t dst = outputs[lic.myOutputIndex]->data() + lic.myOutputOffset;
      uintptr_t src = inputs[lic.sourceInputIndex]->data() + lic.sourceInputOffset;
      if (dst != src && lic.bytes > 0) {
        CHECK_CU(cuMemcpyDtoDAsync(dst, src, lic.bytes, stream));
      }
    }
  }

  // Step 2: IPC-map my inputs for readers
  Vector<uintptr_t> mappedAddrs(op->localInputProvides.size());
  for (size_t i : indices(op->localInputProvides)) {
    const auto& lip = op->localInputProvides[i];
    size_t peerIndex = group->getPeerIndex(lip.readerRank);
    CHECK(lip.myInputIndex < inputs.size());
    uintptr_t myAddr = inputs[lip.myInputIndex]->data() + lip.inputOffset;
    ipcMapper->requestAddress(peerIndex, myAddr, lip.bytes, &mappedAddrs[i]);
  }
  ipcMapper->wait();

  // Step 3: Exchange addresses via peerWriteDyn + ipcMapper push/pop
  // Signal all peers (pushes sync pair to FIFO)
  for (size_t peerIndex : peerIndices) {
    peerWriteDyn(concurrencyIndex, peerIndex, opTypeCompileOpLocal, stepValue, 0, 0);
  }
  // Push mapped addresses AFTER signal (matching order per peer)
  for (size_t i : indices(op->localInputProvides)) {
    size_t peerIndex = group->getPeerIndex(op->localInputProvides[i].readerRank);
    ipcMapper->push(peerIndex, mappedAddrs[i]);
  }

  // Wait for all peer signals (pops sync pair from FIFO)
  for (size_t peerIndex : peerIndices) {
    peerWaitDyn(concurrencyIndex, peerIndex, opTypeCompileOpLocal, stepValue, 0);
  }
  // Pop addresses from peers (reader knows count from localInputCopies)
  Vector<uintptr_t> srcAddrs(op->localInputCopies.size());
  for (size_t i : indices(op->localInputCopies)) {
    const auto& lic = op->localInputCopies[i];
    if (lic.sourceRank == rank) {
      srcAddrs[i] = 0; // Already handled in Step 1
      continue;
    }
    size_t peerIndex = group->getPeerIndex(lic.sourceRank);
    srcAddrs[i] = ipcMapper->pop<uintptr_t>(peerIndex);
  }

  freePendingIpcEvents();

  // Step 4: GPU copies (with synchronization)
  if (kernelVersion > 0) {
    // Kernel path: compile kernel if not yet done
    if (!group->compileOpKernels->cuCopyKernel) {
      group->compileOpKernels->compile();
    }

    // Build descriptor table (both self-copies and peer copies)
    CompileOpCopyParameters params;
    params.stepValue = stepValue;
    params.concurrencyIndex = concurrencyIndex;
    params.numDescriptors = 0;
    params._pad = 0;

    // Add self-copies
    for (const auto& lic : op->localInputCopies) {
      if (lic.sourceRank != rank) {
        continue;
      }
      uintptr_t dst = outputs[lic.myOutputIndex]->data() + lic.myOutputOffset;
      uintptr_t src = inputs[lic.sourceInputIndex]->data() + lic.sourceInputOffset;
      if (dst != src && lic.bytes > 0) {
        CHECK(params.numDescriptors < kMaxCopyDescriptors);
        auto& d = params.descriptors[params.numDescriptors++];
        d.src = src;
        d.dst = dst;
        d.bytes = lic.bytes;
      }
    }
    // Add peer copies
    for (size_t i : indices(op->localInputCopies)) {
      const auto& lic = op->localInputCopies[i];
      if (lic.sourceRank == rank) {
        continue;
      }
      if (lic.bytes > 0) {
        CHECK(params.numDescriptors < kMaxCopyDescriptors);
        auto& d = params.descriptors[params.numDescriptors++];
        d.src = srcAddrs[i];
        d.dst = outputs[lic.myOutputIndex]->data() + lic.myOutputOffset;
        d.bytes = lic.bytes;
      }
    }

    // Single kernel launch replaces syncPeers + copies + syncPeers
    std::array<void*, 1> kparams = {&params};
    CHECK_CU(cuLaunchKernel(group->compileOpKernels->cuCopyKernel, group->compileOpKernels->gridSize, 1, 1,
        group->compileOpKernels->blockSize, 1, 1, group->compileOpKernels->dynamicSmemBytes, stream, kparams.data(),
        nullptr));
  } else {
    // v0: existing cuMemcpyDtoDAsync path (self-copies already done in step 1)
    syncPeers(stream);
    for (size_t i : indices(op->localInputCopies)) {
      const auto& lic = op->localInputCopies[i];
      if (lic.sourceRank == rank) {
        continue;
      }
      CHECK(lic.myOutputIndex < outputs.size());
      uintptr_t dst = outputs[lic.myOutputIndex]->data() + lic.myOutputOffset;
      if (lic.bytes > 0) {
        CHECK_CU(cuMemcpyDtoDAsync(dst, srcAddrs[i], lic.bytes, stream));
      }
    }
    syncPeers(stream);
  }

  // Step 5: Return result — CPU-side immediately done, GPU work is on stream
  auto result = makeShared<ApiFuture>();
  auto future = FutureImplSharedPtr::make();
  future->done = 1;
  result->impl = std::move(future);

  // Hold original tensors alive until the future is consumed
  for (size_t i = 0; i < nInputs; ++i) {
    result->holdTensors.push_back(inputPtrs[i]);
  }
  for (size_t i = 0; i < nOutputs; ++i) {
    result->holdTensors.push_back(outputPtrs[i]);
  }

  return result;
}

// API function that calls the member method
api::CustomOpHandle compileOpFull(api::ProcessGroup* pg, DType dtype, std::span<const api::TensorRegion> inputs,
    std::span<const api::TensorRegion> outputs, api::ReduceOp reduce, bool cpuSync) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  SharedPtr<CustomOpImpl> op = impl->compileOpFull(dtype, inputs, outputs, reduce, cpuSync);
  return api::CustomOpHandle::adopt(op.release());
}

} // namespace moodist
