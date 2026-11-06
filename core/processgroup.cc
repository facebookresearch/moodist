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
#include "cuda_loader.h"
#include "group.h"
#include "ipc_mapper.h"
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
#include <sched.h>
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

// Cache key of the scatter of CPU tensors (a collectiveOp compile_op op; CUDA tensors: see runCollective)
struct ScatterCacheKey {
  size_t numel;
  DType dtype;
  int sourceRank;

  bool operator==(const ScatterCacheKey&) const = default;

  template<typename X>
  void hash(X& x) const {
    x(numel, dtype, sourceRank);
  }
};

// Cache key of the gather of CPU tensors (likewise)
struct GatherCacheKey {
  size_t numel;
  DType dtype;
  int destRank;

  bool operator==(const GatherCacheKey&) const = default;

  template<typename X>
  void hash(X& x) const {
    x(numel, dtype, destRank);
  }
};

// The process group collectives that run as compile_op ops (see ProcessGroupImpl::runCollective)
enum class CollectiveKind : uint8_t {
  allGather,
  reduceScatter,
  allReduce,
  broadcast,
  reduce,
  scatter,
  gather,
  allToAll
};

// Cache key of a collective's compiled op: from the call's arguments only, which are the same on every rank, so that
// every rank compiles an op at the same call
struct CollectiveKey {
  CollectiveKind kind;
  DType dtype;          // UInt8 for the collectives that only copy, which move bytes whatever the tensors' dtype
  api::ReduceOp reduce; // None for those
  // The elements of a part (see runCollective), rounded up to 16 bytes
  size_t numel;
  uint32_t root; // the root's rank: of a broadcast, a reduce, a scatter or a gather (else 0)

  bool operator==(const CollectiveKey&) const = default;

  template<typename X>
  void hash(X& x) const {
    x(kind, dtype, reduce, numel, root);
  }
};

struct EventSerializer {
  Event event;
  CUstream stream;
  EventSerializer(CUevent event, CUstream stream, bool wait = true) : event(Event::reference(event)), stream(stream) {
    if (wait) {
      this->event.wait(stream);
    }
  }
  ~EventSerializer() {
    event.record(stream);
  }
};

namespace {
// Whether the stream is capturing a CUDA graph (or is in a capture that has been invalidated)
bool streamCapturing(CUstream stream) {
  // (the legacy default stream is never captured, and the query fails for it while another stream captures)
  if (stream == nullptr) {
    return false;
  }
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  unsigned long long id = 0;
  return cuStreamGetCaptureInfo(stream, &status, &id) == CUDA_SUCCESS && status != CU_STREAM_CAPTURE_STATUS_NONE;
}

// Raises if the stream is capturing a CUDA graph: work that moodist queues on a stream cannot be captured yet (a
// launch takes its step value and concurrency slot from the host, waits for and records the slot's event, and can
// allocate buffers and give the CPU thread work, all per call). Called before anything touches the stream or the
// process group's state, so that the capture stays valid and the process group usable. launchCustomOp counts on it:
// cuStreamGetId fails on a capturing stream, and with some drivers (580, CUDA 13.0) invalidates the capture.
void requireNotCapturing(CUstream stream, const char* what) {
  if (streamCapturing(stream)) {
    throw std::runtime_error(fmt::sprintf("moodist: %s cannot be captured in a CUDA graph yet", what));
  }
}
} // namespace

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

// The compile_op reduce op of a process group reduce op, for the reductions of CUDA tensors (which run on compile_op,
// see ProcessGroupImpl::runCollective)
api::ReduceOp cudaReduce(ReduceOp op) {
  switch (op) {
  case ReduceOp::SUM:
    return api::ReduceOp::Sum;
  case ReduceOp::AVG:
    return api::ReduceOp::Avg;
  case ReduceOp::MIN:
    return api::ReduceOp::Min;
  case ReduceOp::MAX:
    return api::ReduceOp::Max;
  case ReduceOp::PREMUL_SUM:
    return api::ReduceOp::PremulSum;
  default:
    throw std::runtime_error(fmt::sprintf("moodist: reduce op %d is not supported for CUDA tensors (SUM, AVG, MIN, MAX "
                                          "and PREMUL_SUM are)",
        (int)op));
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

// The process group's timeout (see Group::timeout): MOODIST_TIMEOUT (seconds) if set, else the one given (the
// process group's timeout in PyTorch, e.g. init_process_group's), or 30 minutes if none is. At most 1e9 s (about 32
// years, e.g. for a given duration::max()): the waits' deadlines must not overflow.
static std::chrono::steady_clock::duration timeoutFromEnv(std::chrono::steady_clock::duration timeout) {
  if (const char* env = std::getenv("MOODIST_TIMEOUT"); env && *env) {
    char* end = nullptr;
    double s = std::strtod(env, &end);
    REQUIRE(end != env && *end == 0 && s > 0 && s < 1e9, "MOODIST_TIMEOUT must be a number of seconds (got '%s')", env);
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(s));
  }
  if (timeout <= std::chrono::steady_clock::duration::zero()) {
    return std::chrono::minutes(30);
  }
  return std::min<std::chrono::steady_clock::duration>(timeout, std::chrono::seconds(1000000000));
}

// A process group collective's compiled op (see ProcessGroupImpl::runCollective), and for this rank's input and its
// output in it (where it has them) whether other ranks reach them: whether another rank maps it or the op moves data
// over RDMA. Such tensors must be moodist's memory (see runCollective)
struct CollectiveOp {
  std::shared_ptr<CustomOpDescriptor> op;
  std::array<bool, 2> shared{}; // the input, the output
};

struct ProcessGroupImpl : api::ProcessGroup {
  size_t rank = 0;
  size_t size = 0;

  std::atomic_bool cpuThreadRunning = false;
  SharedPtr<Group> group;

  ThreadUnsafe threadUnsafe;
  uint32_t nextStepValue = 0x1000;
  uint32_t nextConcurrencyIndex = 0;

  std::array<Event, maxConcurrency> concurrencyEvents;
  // The stream (its id, see cuStreamGetId) of each concurrency slot's last op with CUDA tensors (see launchCustomOp)
  std::array<std::optional<unsigned long long>, maxConcurrency> concurrencyStreams;

  std::vector<SharedPtr<Queue>> queues;

  // Per-PG options (can be overridden from Python)
  struct Options {
    bool preferKernelLess = false;
    bool forceKernelLess = false; // Force kernelless even on single-node (for testing)
    int64_t numChunks = -1;       // -1 = auto
    int64_t chunkSize = -1;       // -1 = auto
    int64_t method = -1;          // -1 = auto, 0 = kernel, 1 = copy, 2 = direct
  } options;

  Reusable<IpcEvent> ipcEvents;
  Reusable<IpcEvent> pendingIpcEvents;

  // Memory operations batching for CUDA stream memory ops
  std::vector<CUstreamBatchMemOpParams> memops;

  uint32_t activeId = 0;

  uint32_t nextCustomOpId = 1;

  // Cache for scatter collective compiled ops
  HashMap<ScatterCacheKey, SharedPtr<CustomOpImpl>, GenericHash<ScatterCacheKey>> scatterCache;

  // Cache for gather collective compiled ops
  HashMap<GatherCacheKey, SharedPtr<CustomOpImpl>, GenericHash<GatherCacheKey>> gatherCache;

  // The compiled ops of the collectives that run on compile_op, each compiled at its first call (see runCollective)
  HashMap<CollectiveKey, CollectiveOp, GenericHash<CollectiveKey>> collectiveOps;

  void* c10dStore_ = nullptr; // Opaque pointer to c10d::Store for init

  compile_op::CompileContext persistentCompileContext;

  ProcessGroupImpl(void* c10dStore, int rank_, int size_, std::chrono::steady_clock::duration timeout)
      : rank(rank_), size(size_), c10dStore_(c10dStore) {
    CHECK(rank_ >= 0 && size_ > 0 && rank_ < size_);

    // Initialize options from globals
    options.preferKernelLess = globalPreferKernelLess;

    log.init();

    group = makeShared<Group>(rank, size);
    group->setTimeout(timeoutFromEnv(timeout));

    init();

    for (auto& v : concurrencyEvents) {
      v = Event::create();
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
    CHECK(refcount == 0);
    std::lock_guard l(activeProcessGroupsMutex);
    size_t presize = activeProcessGroups.size();
    activeProcessGroups.erase(activeId);
  }

  void shutdown() {
    if (group && group->cpuThread) {
      group->cpuThread->kill();
    }
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

  void internalBarrier() {
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    std::atomic_uint32_t cpuDone = 0;
    QueueEntryBarrier* e = group->cpuThread->freelistBarrier.pop();
    e->task = taskInternalBarrier;
    e->stepValue = 0;
    e->sd = &group->getStreamData(nullptr);
    e->concurrencyIndex = concurrencyIndex;
    e->cpuDone = &cpuDone;
    group->cpuThread->enqueue(e);
    group->waitForCpuThread(cpuDone, "an internal barrier");
  }

  void resetStepValue() {
    // Reset the 32-bit step value to prevent overflow.
    // All ranks must collectively reach this point — we synchronize with
    // internalBarrier before and after zeroing the barrier arrays.
    log.verbose("Reset step value begin\n");
    for (const auto start = std::chrono::steady_clock::now(); group->cpuThread->busy;) {
      if (std::chrono::steady_clock::now() - start >= group->timeout) {
        fatal("rank %d: the CPU thread's work did not complete within %g s (the process group's timeout; "
              "MOODIST_TIMEOUT overrides it); some rank may have failed",
            rank, seconds(group->timeout));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    internalBarrier();

    CHECK_CU(cuCtxSynchronize());
    std::memset(group->mySharedMem, 0, group->mySharedMemSize);
    nextStepValue = 0x1000;
    for (auto& a : group->buffersToReset) {
      auto& buffer = a->buffer;
      if (buffer.hostAllocated || buffer.numaAllocated) {
        std::memset(buffer.cpuPointer, 0, buffer.bytes);
      } else {
        CHECK_CU(cuMemsetD8(buffer.cudaPointer, 0, buffer.bytes));
      }
    }
    CHECK_CU(cuCtxSynchronize());

    log.info("Reset step value done, syncing\n");
    internalBarrier();
    log.verbose("Reset step value end\n");
  }

  uint32_t getNextStepValue() {
    uint32_t r = std::exchange(nextStepValue, nextStepValue + 0x1000);
    if (r < 0x80000000) {
      return r;
    }
    resetStepValue();
    return 1;
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

  void all_gather(TensorPtr& output, const TensorPtr& input, CUstream stream);
  void reduce_scatter(
      TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp, CUstream stream, double premulValue);
  void allreduce(TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, double premulValue);
  void broadcast(TensorPtr& tensor, int sourceRank, CUstream stream);
  void reduce(TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream, double premulValue);
  void barrier();
  void scatter(std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream);
  void gather(std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream);
  void alltoall(std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream);
  void cudaBarrier(CUstream stream);

  // compileOpFull - compile a distributed tensor operation. collectiveOp: an op behind a process group
  // collective (the scatter and gather of CPU tensors), whose tensors may come from any allocator and have any
  // size: no moodist-memory requirement, and edges the copy kernel cannot take go over RDMA (see
  // CompileContext::collectiveOp).
  SharedPtr<CustomOpImpl> compileOpFull(DType dtype, std::span<const api::TensorRegion> inputs,
      std::span<const api::TensorRegion> outputs, api::ReduceOp reduce, bool cpuSync = false,
      bool collectiveOp = false);
  // The op itself (compileOpFull wraps it for the API)
  std::shared_ptr<CustomOpDescriptor> compileOp(DType dtype, std::span<const api::TensorRegion> inputs,
      std::span<const api::TensorRegion> outputs, api::ReduceOp reduce, bool cpuSync, bool collectiveOp);

  // Runs a collective as a compiled op (see CollectiveKey), compiled at its first call. Its tensors are made of parts
  // of numel elements of dtype (bytes, with UInt8, for the collectives that only copy): a rank's share of an
  // all-gather's output, of a reduce-scatter's input, of a scatter's input or of a gather's output (whose root has
  // one part per rank), what a rank sends to or receives from another in an all-to-all, or the whole tensor of an
  // all-reduce, a broadcast or a reduce. inputs and outputs: this rank's input and output (none where it has none:
  // only the root has an input in a broadcast or a scatter, and an output in a reduce or a gather), each one tensor of
  // all of its parts or a list of tensors of one part each (e.g. a scatter's inputs); CUDA tensors on the group's
  // device, contiguous; the same tensor as input and output in place (an all-reduce, a broadcast or a reduce on its
  // root). root: the root's rank of a broadcast, a reduce, a scatter or a gather; scale: PremulSum's factor.
  void runCollective(CollectiveKind kind, DType dtype, api::ReduceOp reduce, size_t root, size_t numel,
      std::span<const TensorPtr> inputs, std::span<const TensorPtr> outputs, CUstream stream, double scale = 1.0);

  // Template implementation for compileOpFull - parameterized by ndim
  // OLD: Kept for reference during refactoring. Will be deleted once new impl is working.
  template<int ndim>
  SharedPtr<CustomOpImpl> compileOpFullImpl_OLD(std::span<const int64_t> shape_arg, DType dtype,
      std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> inputs_arg,
      std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> outputs_arg);

  // customOp - execute a compiled custom operation (scale: the factor of a PremulSum, see compile_op::isScaledSum)
  SharedPtr<ApiFuture> customOp(std::shared_ptr<CustomOpDescriptor> op, TensorPtr* inputs, size_t nInputs,
      TensorPtr* outputs, size_t nOutputs, CUstream stream, double scale = 1.0);

  // launchCustomOp - the launch of a compiled custom operation: customOp's after its checks of the tensors and its
  // copies, and the process group's collectives' (see runCollective). inputTDs: the inputs, then the input copies;
  // outputTDs: the outputs, then the output copies. If the op has a kernel, the caller makes the op's stream current
  // on the group's device (the kernel's launch takes the current context); if it has CUDA tensors, the caller has
  // checked that the stream does not capture (requireNotCapturing). Returns the op's future (done once queued
  // on the stream if no tensor is a CPU tensor), its concurrency slot and whether it has CUDA tensors (a scratch
  // tensor counts: see launchCustomOp).
  struct CustomOpLaunch {
    FutureImplSharedPtr future;
    uint32_t concurrencyIndex = 0;
    bool anyCuda = false;
  };
  CustomOpLaunch launchCustomOp(const std::shared_ptr<CustomOpDescriptor>& op, std::vector<TensorDataPtr> inputTDs,
      std::vector<TensorDataPtr> outputTDs, CUstream stream, double scale);

  // executeLocalOnly - fast path for all-local custom ops (no CPU thread)
  SharedPtr<ApiFuture> executeLocalOnly(const CustomOpDescriptor& op, std::vector<TensorDataPtr>& inputs,
      std::vector<TensorDataPtr>& outputs, TensorPtr* inputPtrs, size_t nInputs, TensorPtr* outputPtrs, size_t nOutputs,
      uint32_t concurrencyIndex, uint32_t stepValue, CUstream stream);

  // cat - concatenate tensors from multiple ranks
  api::FutureHandle cat(const int* indices, const TensorPtr* tensors, size_t count, TensorPtr* out);
};

namespace {

struct GlobalDtor {
  ~GlobalDtor() {
    std::unique_lock l(activeProcessGroupsMutex, std::defer_lock);
    Vector<SharedPtr<ProcessGroupImpl>> list;
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (size_t i = 0;; ++i) {
      l.lock();
      for (auto& v : activeProcessGroups) {
        if (v.second->refcount >= 1 && v.second->cpuThreadRunning) {
          list.push_back(share(v.second));
        }
      }
      l.unlock();
      if (list.empty()) {
        break;
      }
      for (auto& ptr : list) {
        ptr->shutdown();
      }
      list.clear();
      l.lock();
      if (i >= 5 && std::chrono::steady_clock::now() > timeout) {
        log.error("Timed out waiting for process group shutdown (%d process groups still active).",
            activeProcessGroups.size());
        break;
      }
      l.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
} globalDtor;

} // namespace

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

  struct H {
    SharedPtr<ProcessGroupImpl> self;
    H(ProcessGroupImpl* impl) : self(share(impl)) {
      self->cpuThreadRunning = true;
    }
    ~H() {
      if (self) {
        self->cpuThreadRunning = false;
      }
    }
    H(H&&) = default;
    H(const H&) = delete;
    H& operator=(H&&) = default;
    H& operator=(const H&) = default;
  };
  group->init(f, [handle = H(this)] {});
}

// ============================================================================
// ProcessGroupImpl::all_gather() - AllGather collective
// ============================================================================

void ProcessGroupImpl::all_gather(TensorPtr& output, const TensorPtr& input, CUstream stream) {
  CHECK(input.is_contiguous());
  CHECK(output.is_contiguous());

  const size_t bytes = input.numel() * input.itemsize();
  CHECK(static_cast<size_t>(output.numel() * output.itemsize()) == bytes * size);

  // CUDA tensors: a compile_op op (see runCollective)
  if (input.is_cuda()) {
    CHECK(input.device_index() == group->deviceIndex);
    CHECK(output.is_cuda());
    CHECK(output.device_index() == group->deviceIndex);
    runCollective(
        CollectiveKind::allGather, DType::UInt8, api::ReduceOp::None, 0, bytes, {&input, 1}, {&output, 1}, stream);
    return;
  }

  // CPU tensors: the CPU thread
  std::unique_lock l(threadUnsafe);

  CHECK(bytes > 0);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  StreamData& sd = group->getStreamData(nullptr);
  std::atomic_uint32_t cpuDone = 0;

  QueueEntryAllGatherCpu* e = group->cpuThread->freelistAllGatherCpu.pop();
  e->task = taskAllGatherCpu;
  e->stepValue = stepValue;
  e->concurrencyIndex = concurrencyIndex;
  e->sd = &sd;
  e->inputAddress = (uintptr_t)input.data_ptr();
  e->outputAddress = (uintptr_t)output.data_ptr();
  e->bytes = bytes;
  e->pitch = bytes;
  e->cpuDone = &cpuDone;
  group->cpuThread->enqueue(e);
  l.unlock();
  group->waitForCpuThread(cpuDone, "an all-gather of CPU tensors");
}

// ============================================================================
// ProcessGroupImpl::reduce_scatter() - ReduceScatter collective
// ============================================================================

void ProcessGroupImpl::reduce_scatter(
    TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp, CUstream stream, double premulValue) {
  CHECK(input.is_contiguous());
  CHECK(output.is_contiguous());

  const DType dtype = output.dtype();
  CHECK(input.dtype() == dtype);
  const size_t numel = output.numel();
  CHECK(static_cast<size_t>(input.numel()) == numel * size);

  // CUDA tensors: a compile_op op (see runCollective)
  if (input.is_cuda()) {
    CHECK(input.device_index() == group->deviceIndex);
    CHECK(output.is_cuda());
    CHECK(output.device_index() == group->deviceIndex);
    runCollective(CollectiveKind::reduceScatter, dtype, cudaReduce(reduceOp), 0, numel, {&input, 1}, {&output, 1},
        stream, premulValue);
    return;
  }

  // CPU tensors: the CPU thread
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  Dtype dindex = toInternalDtype(dtype);
  // avg: the sum, divided by the group size below (the CPU reduction has no avg)
  bool workaroundMean = (reduceOp == ReduceOp::AVG);
  Reduction opindex = workaroundMean ? Reduction::sum : toInternalReduction(reduceOp);

  const size_t bytes = numel * output.itemsize();
  CHECK(bytes > 0);

  uintptr_t inputAddress = (uintptr_t)input.data_ptr();
  TensorPtr premulTensor;
  if (reduceOp == ReduceOp::PREMUL_SUM) {
    premulTensor = input * premulValue;
    inputAddress = (uintptr_t)premulTensor.data_ptr();
    CHECK(static_cast<size_t>(premulTensor.numel() * premulTensor.itemsize()) == bytes * size);
  }
  StreamData& sd = group->getStreamData(nullptr);
  std::atomic_uint32_t cpuDone = 0;
  QueueEntryReduceScatterCpu* e = group->cpuThread->freelistReduceScatterCpu.pop();
  e->task = taskReduceScatterCpu;
  e->stepValue = stepValue;
  e->concurrencyIndex = concurrencyIndex;
  e->sd = &sd;
  e->inputAddress = inputAddress;
  e->outputAddress = (uintptr_t)output.data_ptr();
  e->bytes = bytes;
  e->pitch = bytes;
  e->cpuDone = &cpuDone;
  e->dindex = dindex;
  e->opindex = opindex;
  group->cpuThread->enqueue(e);
  l.unlock();
  group->waitForCpuThread(cpuDone, "a reduce-scatter of CPU tensors");
  if (workaroundMean) {
    output.mul_(1.0f / size);
  }
}

// ============================================================================
// ProcessGroupImpl::allreduce() - Allreduce collective
// ============================================================================

void ProcessGroupImpl::allreduce(TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, double premulValue) {
  // CUDA tensors: a compile_op op in place (see runCollective)
  if (tensor.is_cuda()) {
    CHECK(tensor.is_contiguous() && tensor.device_index() == group->deviceIndex);
    runCollective(CollectiveKind::allReduce, tensor.dtype(), cudaReduce(reduceOp), 0, tensor.numel(), {&tensor, 1},
        {&tensor, 1}, stream, premulValue);
    return;
  }

  // CPU tensors: a reduce-scatter and an all-gather (in a temporary tensor whose size the group size divides)
  int64_t numel = tensor.numel();
  CHECK(numel > 0);

  if (numel % static_cast<int64_t>(size) == 0) {
    // Tensor is evenly divisible by size - use in-place view
    std::array<int64_t, 2> viewShape = {static_cast<int64_t>(size), numel / static_cast<int64_t>(size)};
    TensorPtr t = tensor.view(viewShape);
    TensorPtr o = t.narrow(0, static_cast<int64_t>(rank), 1).view({numel / static_cast<int64_t>(size)});

    reduce_scatter(o, t, reduceOp, stream, premulValue);
    all_gather(t, o, stream);
  } else {
    int64_t newsize =
        (numel + static_cast<int64_t>(size) - 1) / static_cast<int64_t>(size) * static_cast<int64_t>(size);
    size_t bytes = static_cast<size_t>(numel) * static_cast<size_t>(tensor.itemsize());

    TensorPtr temporary = TensorPtr::empty({newsize}, tensor.dtype(), -1);
    std::memcpy(temporary.data_ptr(), tensor.data_ptr(), bytes);

    std::array<int64_t, 2> viewShape = {static_cast<int64_t>(size), newsize / static_cast<int64_t>(size)};
    TensorPtr t = temporary.view(viewShape);
    TensorPtr o = t.narrow(0, static_cast<int64_t>(rank), 1).view({newsize / static_cast<int64_t>(size)});

    reduce_scatter(o, t, reduceOp, stream, premulValue);
    all_gather(t, o, stream);

    std::memcpy(tensor.data_ptr(), temporary.data_ptr(), bytes);
  }
}

// ============================================================================
// ProcessGroupImpl::broadcast() - Broadcast collective
// ============================================================================

void ProcessGroupImpl::broadcast(TensorPtr& tensor, int sourceRank, CUstream stream) {
  REQUIRE(sourceRank >= 0 && static_cast<size_t>(sourceRank) < size, "broadcast: root rank %d out of range [0, %d)",
      sourceRank, size);
  REQUIRE(tensor.is_contiguous(), "broadcast: the tensor must be contiguous");

  size_t numel = static_cast<size_t>(tensor.numel());
  size_t bytes = numel * static_cast<size_t>(tensor.itemsize());

  // CUDA tensors: a compile_op op (see runCollective), from the root's tensor to every rank's (in place on the root)
  if (tensor.is_cuda()) {
    const std::span<const TensorPtr> root =
        static_cast<size_t>(sourceRank) == rank ? std::span<const TensorPtr>(&tensor, 1) : std::span<const TensorPtr>();
    runCollective(CollectiveKind::broadcast, DType::UInt8, api::ReduceOp::None, static_cast<size_t>(sourceRank), bytes,
        root, {&tensor, 1}, stream);
    return;
  }

  // CPU tensors: the CPU thread
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  uintptr_t tensorAddress = (uintptr_t)tensor.data_ptr();

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
  group->waitForCpuThread(cpuDone, "a broadcast of CPU tensors");
}

// ============================================================================
// ProcessGroupImpl::reduce() - Reduce collective
// ============================================================================

void ProcessGroupImpl::reduce(TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream, double premulValue) {
  REQUIRE(destRank >= 0 && static_cast<size_t>(destRank) < size, "reduce: root rank %d out of range [0, %d)", destRank,
      size);
  REQUIRE(tensor.is_contiguous(), "reduce: the tensor must be contiguous");
  if (!tensor.is_cuda()) {
    throw std::runtime_error("moodist: reduce of CPU tensors is not supported");
  }

  // A compile_op op (see runCollective): every rank's tensor is an input, and the root's the output too (in place)
  const std::span<const TensorPtr> root =
      static_cast<size_t>(destRank) == rank ? std::span<const TensorPtr>(&tensor, 1) : std::span<const TensorPtr>();
  runCollective(CollectiveKind::reduce, tensor.dtype(), cudaReduce(reduceOp), static_cast<size_t>(destRank),
      static_cast<size_t>(tensor.numel()), {&tensor, 1}, root, stream, premulValue);
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
  group->waitForCpuThread(cpuDone, "a barrier");
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

  // CUDA tensors: a compile_op op (see runCollective), from the root's inputs, one per rank
  if (isCuda) {
    runCollective(CollectiveKind::scatter, DType::UInt8, api::ReduceOp::None, static_cast<size_t>(sourceRank),
        numel * static_cast<size_t>(output.itemsize()), std::span<const TensorPtr>(inputs.data(), inputs.size()),
        {&output, 1}, stream);
    return;
  }

  // CPU tensors: a compile_op op over RDMA
  ScatterCacheKey key{numel, dtype, sourceRank};

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

    SharedPtr<CustomOpImpl> compiled =
        compileOpFull(dtype, inputSpecs, outputSpecs, api::ReduceOp::None, /*cpuSync=*/false, /*collectiveOp=*/true);
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

  // CUDA tensors: a compile_op op (see runCollective), into the root's outputs, one per rank
  if (isCuda) {
    runCollective(CollectiveKind::gather, DType::UInt8, api::ReduceOp::None, static_cast<size_t>(destRank),
        numel * static_cast<size_t>(input.itemsize()), {&input, 1},
        std::span<const TensorPtr>(outputs.data(), outputs.size()), stream);
    return;
  }

  // CPU tensors: a compile_op op over RDMA
  GatherCacheKey key{numel, dtype, destRank};

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

    SharedPtr<CustomOpImpl> compiled =
        compileOpFull(dtype, inputSpecs, outputSpecs, api::ReduceOp::None, /*cpuSync=*/false, /*collectiveOp=*/true);
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
  REQUIRE(inputs.size() == size && outputs.size() == size, "all-to-all: expected %d inputs and outputs, got %d and %d",
      size, inputs.size(), outputs.size());
  for (auto* list : {&inputs, &outputs}) {
    for (const TensorPtr& t : *list) {
      if (!t.is_cuda()) {
        throw std::runtime_error("moodist: all-to-all of CPU tensors is not supported");
      }
    }
  }
  const size_t bytes = static_cast<size_t>(inputs[0].numel() * inputs[0].itemsize());

  // A compile_op op (see runCollective): input j goes to rank j, output i comes from rank i (one tensor each if the
  // lists are views of one, as all_to_all_single's)
  runCollective(CollectiveKind::allToAll, DType::UInt8, api::ReduceOp::None, 0, bytes,
      std::span<const TensorPtr>(inputs.data(), inputs.size()),
      std::span<const TensorPtr>(outputs.data(), outputs.size()), stream);
}

// ============================================================================
// ProcessGroupImpl::cudaBarrier() - CUDA Barrier
// ============================================================================

void ProcessGroupImpl::cudaBarrier(CUstream stream) {
  requireNotCapturing(stream, "cuda_barrier");
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

api::ProcessGroupHandle createProcessGroup(
    void* c10dStore, int rank, int size, std::chrono::steady_clock::duration timeout) {
  return api::ProcessGroupHandle::create(internalNew<ProcessGroupImpl>(c10dStore, rank, size, timeout));
}

void processGroupDestroy(api::ProcessGroup* pg) {
  internalDelete(static_cast<ProcessGroupImpl*>(pg));
}

void processGroupShutdown(api::ProcessGroup* pg) {
  ((ProcessGroupImpl*)pg)->shutdown();
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
  } else if (key == "timeout_ms") {
    // Read-only: the process group's timeout (see Group::timeout)
    return std::chrono::duration_cast<std::chrono::milliseconds>(impl->group->timeout).count();
  } else if (key == "nvls_launches" || key == "nvls_loop_launches") {
    // Read-only: how many compile_op launches of this rank in this group used NVLS (copies, reductions; with NVLS
    // runs, a launch every run of which could; see generateKernel), or how many of those ran nvlsReduceLoop (NVLS
    // reductions, or NVLS runs in the warp loop: not with MOODIST_COMPILE_OP_NVLS_LOOP=0), e.g. for tests; as of the
    // launches that have completed
    const Group::Nvls& nvls = impl->group->nvls;
    if (nvls.state != 1) {
      return 0;
    }
    uint32_t value = 0;
    CHECK_CU(cuCtxSetCurrent(impl->group->cuContext));
    CHECK_CU(cuMemcpyAsync(
        (CUdeviceptr)&value, key == "nvls_launches" ? nvls.launches : nvls.loopLaunches, sizeof(value), nullptr));
    CHECK_CU(cuStreamSynchronize(nullptr));
    return value;
  } else if (key == "nvlink_domain") {
    // Read-only: this rank's NVLink domain in the group, as its first rank (the ranks of a multi-node NVLink fabric
    // that are in the group, else of this rank's node; see compile_op's cudaRanks), e.g. for tests
    const Group& g = *impl->group;
    if (!g.fabricDomainRanks.empty()) {
      return (int64_t)g.fabricDomainRanks[0];
    }
    size_t first = g.rank;
    for (size_t r : g.ipcRanks) {
      first = std::min(first, r);
    }
    return (int64_t)first;
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

void processGroupCommPoolAllocate(api::ProcessGroup* pg, size_t bytes, CUstream stream, uintptr_t* outPtr,
    void** outCleanupCtx, int* outDeviceIndex) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  Group* group = &*impl->group;
  CudaAllocation alloc = allocatorCommPoolAllocate(group->getCommPool(), bytes, stream, group->deviceIndex);
  *outPtr = alloc.ptr;
  *outCleanupCtx = alloc.cleanupCtx;
  *outDeviceIndex = alloc.deviceIndex;
}

// ============================================================================
// Collective operations
// ============================================================================

void processGroupAllGather(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->all_gather(output, input, stream);
}

void processGroupReduceScatter(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp,
    CUstream stream, double premulValue) {
  static_cast<ProcessGroupImpl*>(pg)->reduce_scatter(output, input, reduceOp, stream, premulValue);
}

void processGroupAllreduce(
    api::ProcessGroup* pg, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, double premulValue) {
  static_cast<ProcessGroupImpl*>(pg)->allreduce(tensor, reduceOp, stream, premulValue);
}

void processGroupBroadcast(api::ProcessGroup* pg, TensorPtr& tensor, int sourceRank, CUstream stream) {
  static_cast<ProcessGroupImpl*>(pg)->broadcast(tensor, sourceRank, stream);
}

void processGroupReduce(
    api::ProcessGroup* pg, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream, double premulValue) {
  static_cast<ProcessGroupImpl*>(pg)->reduce(tensor, destRank, reduceOp, stream, premulValue);
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
  auto tensor = q->getTensor(block, timeoutOpt, outSize);
  if (tensor.defined()) {
    *outTensor = std::move(tensor);
    return true;
  }
  return false;
}

api::QueueWorkHandle queuePut(api::Queue* queue, const TensorPtr& tensor, uint32_t transaction, bool waitOnDestroy) {
  auto* q = static_cast<Queue*>(queue);
  QueueWork work = q->putTensor(tensor, transaction, waitOnDestroy);
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

// ============================================================================
// RDMA direct read API
// ============================================================================

struct RdmaWorkImpl {
  std::atomic_uint32_t done = 0;
  void wait() {
    while (done.load(std::memory_order_acquire) == 0) {
      futexWait(&done, 0, std::chrono::seconds(1));
    }
  }
};

struct RdmaWork : api::RdmaWork {
  std::shared_ptr<RdmaWorkImpl> impl;
  TensorPtr tensor; // keeps destination tensor alive
  ~RdmaWork() {
    if (impl) {
      impl->wait();
    }
  }
  void wait() {
    if (impl) {
      impl->wait();
    }
  }
};

void rdmaRegister(api::ProcessGroup* pg, uintptr_t address, size_t bytes, bool isCuda, api::RdmaEndpoint* out) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  auto* cpuThread = impl->group->cpuThread;

  std::atomic_uint32_t cpuDone{0};
  uint32_t rkeys[Group::maxDevices] = {};

  auto* e = cpuThread->freelistRdmaRegister.pop();
  e->task = taskRdmaRegister;
  e->address = address;
  e->bytes = bytes;
  e->isCuda = isCuda;
  e->outRkeys = rkeys;
  e->cpuDone = &cpuDone;
  cpuThread->enqueue(e);

  while (cpuDone.load(std::memory_order_acquire) == 0) {
    futexWait(&cpuDone, 0, std::chrono::seconds(1));
  }

  out->address = address;
  out->bytes = bytes;
  out->rank = impl->rank;
  out->isCuda = isCuda;
  for (size_t i = 0; i < 4; ++i) {
    out->rkeys[i] = rkeys[i];
  }
}

api::RdmaWorkHandle rdmaRead(
    api::ProcessGroup* pg, uintptr_t dstAddress, size_t dstBytes, bool dstIsCuda, const api::RdmaEndpoint* src) {
  auto* impl = static_cast<ProcessGroupImpl*>(pg);
  auto* cpuThread = impl->group->cpuThread;

  auto workImpl = std::make_shared<RdmaWorkImpl>();

  auto* e = cpuThread->freelistRdmaRead.pop();
  e->task = taskRdmaRead;
  e->dstAddress = dstAddress;
  e->dstBytes = dstBytes;
  e->dstIsCuda = dstIsCuda;
  e->srcRank = src->rank;
  e->srcAddress = src->address;
  for (size_t i = 0; i < 4; ++i) {
    e->srcRkeys[i] = src->rkeys[i];
  }
  e->done = &workImpl->done;
  cpuThread->enqueue(e);

  auto* work = internalNew<RdmaWork>();
  work->impl = std::move(workImpl);
  return api::RdmaWorkHandle::create(work);
}

void rdmaWorkWait(api::RdmaWork* work) {
  static_cast<RdmaWork*>(work)->wait();
}

void rdmaWorkDestroy(api::RdmaWork* work) {
  internalDelete(static_cast<RdmaWork*>(work));
}

namespace api {
void destroy(RdmaWork* work) {
  rdmaWorkDestroy(work);
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
  // (called from a destructor: an error is logged, as thrown it would end the process; e.g. a compile_op future dropped
  // while the current stream captures a CUDA graph, which neither joins that stream nor copies the outputs back)
  try {
    futureWait(future);
  } catch (const std::exception& e) {
    // (the log adds its own "moodist: ")
    std::string_view what = e.what();
    if (what.starts_with("moodist: ")) {
      what.remove_prefix(std::string_view("moodist: ").size());
    }
    log.error("%s (while dropping the future: the current stream does not wait for the op, and outputs that it copies "
              "back are not written)\n",
        std::string(what));
  }
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

void dumpProfiling(const Group* group, std::shared_ptr<CustomOpDescriptor> op) {
  size_t blockSize = op->config.blockSize;
  size_t gridSize = op->config.gridSize;
  Vector<char> data;
  data.resize(op->kernelHandle->profilingBytesPerThread * blockSize * gridSize);

  cudaCopy((uintptr_t)data.data(), op->kernelHandle->profilingData.cudaPointer, data.size(), 0);
  CHECK_CU(cuCtxSynchronize());

  CUdevice cuDevice;
  CHECK_CU(cuCtxGetDevice(&cuDevice));
  int clockRate;
  CHECK_CU(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice));

  uint64_t clockFreq = (uint64_t)clockRate * 1000;
  uint64_t clockDiv = (clockFreq + 1000000 - 1) / 1000000;

  std::string fn = fmt::sprintf("moodist-trace-cuda-%s-%d-op%u.json", group->name, group->rank, op->id);
  FILE* f = fopen(fn.c_str(), "wb");
  CHECK(f != nullptr);
  fmt::fprintf(f, R"({"traceEvents": [)"
                  "\n");

  bool first = true;
  size_t truncated = 0; // threads that recorded more entries than their area holds (the kernel drops the others)
  for (size_t blockIndex : range(gridSize)) {
    for (size_t threadIndex : range(blockSize)) {
      size_t dataOffset = op->kernelHandle->profilingBytesPerThread * ((blockSize * blockIndex) + threadIndex);

      uint32_t numEntries;
      std::memcpy(&numEntries, data.data() + dataOffset, sizeof(uint32_t));
      uint32_t recorded;
      std::memcpy(&recorded, data.data() + dataOffset + 4, sizeof(uint32_t));
      truncated += recorded > numEntries;
      CHECK(8 + 16 * (size_t)numEntries <= op->kernelHandle->profilingBytesPerThread);

      // log.info("Block %d thread %d has %d entries\n", blockIndex, threadIndex, numEntries);

      for (size_t i : range(numEntries)) {
        uint64_t clock;
        std::memcpy(&clock, data.data() + dataOffset + 8 + 16 * i, sizeof(uint64_t));
        if (i == numEntries - 1) {
          break;
        }
        uint64_t clockMask = (1ull << 48) - 1;
        uint64_t nextClock;
        std::memcpy(&nextClock, data.data() + dataOffset + 8 + 16 * (i + 1), sizeof(uint64_t));
        uint64_t duration = (nextClock & clockMask) - (clock & clockMask);
        uint32_t index;
        // std::memcpy(&index, data.data() + dataOffset + 8 + 16 * i + 8, sizeof(uint32_t));
        index = clock >> 48;
        // (an unnamed marker, profiling.leave(), ends the span before it)
        if (index && !op->kernelHandle->profilingNames.at(index).empty()) {
          std::string name = op->kernelHandle->profilingNames.at(index);
          if (!first) {
            fmt::fprintf(f, ",\n");
          } else {
            first = false;
          }
          fmt::fprintf(f, R"({"ph": "X", "name": "%s", "ts": %d, "dur": %d, "tid": %d, "pid": %d})", name,
              clock & clockMask, duration, blockIndex * blockSize + threadIndex, blockIndex);
        }
      }
    }
  }

  fmt::fprintf(f, "]}\n");
  fclose(f);
  log.info("Chrome trace dumped to %s\n", fn);
  if (truncated) {
    log.error("the trace of compile_op op %u is incomplete: %d threads recorded more entries than fit in %d "
              "bytes each (the later ones are missing)\n",
        op->id, truncated, op->kernelHandle->profilingBytesPerThread);
  }
}

} // namespace

// ============================================================================
// ProcessGroupImpl::compileOpFull - compile a distributed tensor operation
// ============================================================================

SharedPtr<CustomOpImpl> ProcessGroupImpl::compileOpFull(DType dtype, std::span<const api::TensorRegion> inputsVec,
    std::span<const api::TensorRegion> outputsVec, api::ReduceOp reduce, bool cpuSync, bool collectiveOp) {
  std::shared_ptr<CustomOpDescriptor> op = compileOp(dtype, inputsVec, outputsVec, reduce, cpuSync, collectiveOp);

  // Wrap CustomOpDescriptor in CustomOpImpl with execution closure
  auto result = makeShared<CustomOpImpl>();
  auto selfPtr = share(this);
  result->call = [op, selfPtr](TensorPtr* inputs, size_t nInputs, TensorPtr* outputs, size_t nOutputs,
                     CUstream stream) -> SharedPtr<ApiFuture> {
    return selfPtr->customOp(op, inputs, nInputs, outputs, nOutputs, stream);
  };

  return result;
}

std::shared_ptr<CustomOpDescriptor> ProcessGroupImpl::compileOp(DType dtype,
    std::span<const api::TensorRegion> inputsVec, std::span<const api::TensorRegion> outputsVec, api::ReduceOp reduce,
    bool cpuSync, bool collectiveOp) {

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
  compile_op::CompileContext& ctx = persistentCompileContext;
  ctx.group = group.get();
  ctx.queues = std::span<SharedPtr<Queue>>(queues.data(), queues.size());
  ctx.barrier = [this]() {
    barrier();
  };
  ctx.resetBarriers = [this]() {
    resetStepValue();
  };
  ctx.nextOpId = &nextCustomOpId;
  ctx.collectiveOp = collectiveOp;
  ctx.timeout = group->timeout;
  REQUIRE(!ctx.timedOut, "moodist.compile_op: an earlier compile_op of this process group timed out, which left its "
                         "exchanges out of step");

  // Call compile_op
  auto op = compile_op::compile(ctx, dtype, inputsVec, outputsVec, reduce, cpuSync);
  op->requireMoodistMemory = !collectiveOp;
  return op;
}

// ============================================================================
// ProcessGroupImpl::runCollective - process group collectives on compile_op
// ============================================================================

namespace {
// The parts (see runCollective) of rank r's input and output in a collective, 0 where it has none: one each in an
// all-reduce; in an all-gather, the output has one per rank, and in a reduce-scatter the input; in an all-to-all,
// both (what the rank sends to each rank, what it receives from each); every rank has an output in a broadcast, and
// the root an input (in place: then the op's destinations are on every rank, at the same offset of the pool where the
// tensors are there, as multicast stores need, which the kernel does not use for it yet: see docs/internals/
// compile_op.md, Known issues); every rank has an input in a reduce, and the root an output; in a
// scatter every rank has an output, and the root an input of one part per rank; in a gather the other way around.
std::pair<size_t, size_t> collectiveParts(CollectiveKind kind, size_t r, size_t root, size_t size) {
  switch (kind) {
  case CollectiveKind::allGather:
    return {1, size};
  case CollectiveKind::reduceScatter:
    return {size, 1};
  case CollectiveKind::allReduce:
    return {1, 1};
  case CollectiveKind::broadcast:
    return {r == root ? 1 : 0, 1};
  case CollectiveKind::reduce:
    return {1, r == root ? 1 : 0};
  case CollectiveKind::scatter:
    return {r == root ? size : 0, 1};
  case CollectiveKind::gather:
    return {1, r == root ? size : 0};
  case CollectiveKind::allToAll:
    return {size, size};
  }
  CHECK(false);
  return {};
}

// Whether [address, address + bytes) is in one allocation of CUDA memory (one cudaMalloc, one mapping), which copies
// and mappings take as a whole: a list of tensors that are one after the other, but in separate allocations (as
// PyTorch's allocator may hand out), is not one tensor
bool oneAllocation(uintptr_t address, size_t bytes) {
  CUdeviceptr base = 0;
  size_t size = 0;
  return cuMemGetAddressRange(&base, &size, address) == CUDA_SUCCESS && address + bytes <= base + size;
}

// A rank's input or output in a collective, as the caller has it: `rows` parts of `width` bytes (none if rows is 0),
// at addresses[i], one after the other in one allocation if contiguous (one tensor, or a list of views of one, e.g.
// all_to_all_single's; moodist's memory is one allocation)
struct CollectiveSide {
  size_t rows = 0;
  Vector<uintptr_t> addresses;
  bool contiguous = true;

  CollectiveSide(std::span<const TensorPtr> tensors, size_t rows, size_t width, int device, const char* what) {
    REQUIRE(rows == 0 ? tensors.empty() : tensors.size() == 1 || tensors.size() == rows,
        "moodist: got %d %s tensors for %d parts", tensors.size(), what, rows);
    for (const TensorPtr& t : tensors) {
      REQUIRE(t.is_cuda() && t.device_index() == device, "moodist: an %s tensor is not on the process group's device",
          what);
      REQUIRE(t.is_contiguous(), "moodist: an %s tensor is not contiguous", what);
      const size_t parts = tensors.size() == 1 ? rows : 1;
      REQUIRE((size_t)(t.numel() * t.itemsize()) == parts * width,
          "moodist: an %s tensor has %d bytes, expected %d (%d parts of %d)", what, t.numel() * t.itemsize(),
          parts * width, parts, width);
    }
    this->rows = rows;
    addresses.resize(rows);
    for (size_t i : range(rows)) {
      addresses[i] =
          tensors.size() == 1 ? (uintptr_t)tensors[0].data_ptr() + i * width : (uintptr_t)tensors[i].data_ptr();
      contiguous &= addresses[i] == addresses[0] + i * width;
    }
    if (contiguous && tensors.size() > 1 && width != 0 && !allocator::owns(addresses[0])) {
      contiguous = oneAllocation(addresses[0], rows * width);
    }
  }
  uintptr_t address() const {
    return addresses[0];
  }
};
} // namespace

void ProcessGroupImpl::runCollective(CollectiveKind kind, DType dtype, api::ReduceOp reduce, size_t root, size_t numel,
    std::span<const TensorPtr> inputs, std::span<const TensorPtr> outputs, CUstream stream, double scale) {
  const int device = group->deviceIndex;
  StreamGuard sg(stream, device);
  const size_t itemsize = wrapperApi.dtypeSize(dtype);
  CHECK(itemsize != 0 && 16 % itemsize == 0);
  CHECK(root < size);
  const size_t width = numel * itemsize;
  const auto [inputParts, outputParts] = collectiveParts(kind, rank, root, size);
  const CollectiveSide in(inputs, inputParts, width, device, "input");
  const CollectiveSide out(outputs, outputParts, width, device, "output");
  if (std::string error = compile_op::reduceDtypeError(dtype, reduce); !error.empty()) {
    throw std::runtime_error(error);
  }
  if (numel == 0) {
    return;
  }
  if (size == 1) {
    // Nothing to combine or send (an op of one rank would only copy): the input to the output, scaled by PremulSum's
    // factor
    if (in.rows != 0 && out.rows != 0) {
      CHECK(in.rows == 1 && out.rows == 1);
      if (in.address() != out.address()) {
        CHECK_CU(cuMemcpyDtoDAsync(out.address(), in.address(), width, stream));
      }
    }
    if (reduce == api::ReduceOp::PremulSum) {
      CHECK(outputs.size() == 1);
      TensorPtr(outputs[0]).mul_(scale);
    }
    return;
  }
  // (before the compile at a shape's first call, the staging buffers and the launch)
  requireNotCapturing(stream, "process group collectives of CUDA tensors");

  // The op: of parts of numel rounded up to 16 bytes (the copy kernel's unit). A call of another size uses it through
  // buffers whose parts have the padding (below).
  const size_t unit = 16 / itemsize;
  const size_t n = (numel + unit - 1) / unit * unit;
  const size_t pitch = n * itemsize;
  const bool rooted = kind == CollectiveKind::broadcast || kind == CollectiveKind::reduce ||
                      kind == CollectiveKind::scatter || kind == CollectiveKind::gather;
  const CollectiveKey key{kind, dtype, reduce, n, rooted ? (uint32_t)root : 0};
  auto it = collectiveOps.find(key);
  if (it == collectiveOps.end()) {
    // The regions of every rank, in the group's tensor (compile takes them all): a vector of size parts (one for an
    // all-reduce, a broadcast and a reduce), or for an all-to-all a matrix of size x size parts, whose row r is what
    // rank r sends (part j to rank j) and column r what it receives
    std::vector<api::TensorRegion> inputRegions;
    std::vector<api::TensorRegion> outputRegions;
    const int64_t part = (int64_t)n;
    const int64_t all = (int64_t)(size * n);
    for (size_t r : range(size)) {
      auto region = [&](std::vector<int64_t> offset, std::vector<int64_t> shape) {
        return api::TensorRegion{.rank = (int32_t)r,
            .offset = std::move(offset),
            .shape = std::move(shape),
            .tensorId = "0",
            .device = "cuda"};
      };
      const int64_t at = (int64_t)(r * n);
      const auto [rInputs, rOutputs] = collectiveParts(kind, r, root, size);
      if (kind == CollectiveKind::allToAll) {
        inputRegions.push_back(region({(int64_t)r, 0}, {1, all}));
        outputRegions.push_back(region({0, at}, {(int64_t)size, part}));
        continue;
      }
      // Otherwise a rank's tensor of one part is its own part of the vector (the whole vector if it has one part),
      // and one of size parts is the whole vector
      const bool onePart =
          kind == CollectiveKind::allReduce || kind == CollectiveKind::broadcast || kind == CollectiveKind::reduce;
      if (rInputs != 0) {
        inputRegions.push_back(rInputs == size && !onePart ? region({0}, {all}) : region({onePart ? 0 : at}, {part}));
      }
      if (rOutputs != 0) {
        outputRegions.push_back(rOutputs == size && !onePart ? region({0}, {all}) : region({onePart ? 0 : at}, {part}));
      }
    }
    CollectiveOp c;
    c.op = compileOp(dtype, inputRegions, outputRegions, reduce, /*cpuSync=*/false, /*collectiveOp=*/false);
    // The tensors come from any allocator: below, each rank puts those that other ranks reach into moodist's memory.
    // A rank maps every tensor of another that is in one of its edges (whichever rank executes it), so in a group of
    // more than one rank, every input and output counts
    c.op->requireMoodistMemory = false;
    CHECK(c.op->inputs.size() == (inputParts != 0 ? 1 : 0) && c.op->outputs.size() == (outputParts != 0 ? 1 : 0));
    CHECK(c.op->inputCopies.empty() && c.op->outputCopies.empty());
    // (the tensors of a rank: its inputs, its outputs, then a scratch tensor)
    const size_t numInputs = c.op->inputs.size();
    for (const auto& m : c.op->graph.remoteCudaTensorMappings) {
      if (m.tensorIndex < numInputs + c.op->outputs.size()) {
        c.shared[m.tensorIndex < numInputs ? 0 : 1] = true;
      }
    }
    if (c.op->anyRankNeedsRdma) {
      c.shared = {true, true};
    }
    it = collectiveOps.emplace(key, std::move(c)).first;
  }
  const CollectiveOp& c = it->second;
  const std::shared_ptr<CustomOpDescriptor>& op = c.op;
  CHECK(in.rows == 0 || op->inputs.at(0).bytes == in.rows * pitch);
  CHECK(out.rows == 0 || op->outputs.at(0).bytes == out.rows * pitch);

  // This rank's tensors where they are, unless their parts need padding or are not one after the other, other ranks
  // reach them and they are not moodist's memory, or the kernel cannot take their address (16 bytes aligned): then
  // through a buffer of the group's communication pool, which, when every rank does the same, is at the same offset on
  // every rank, so that the op can still use NVLS. Other ranks reach moodist's memory only: its mappings and RDMA
  // registrations are of memory that stays moodist's, while another allocator's (e.g. PyTorch's) could be freed and
  // reused with the mappings of peers still open (undefined behavior for CUDA IPC) and its registrations holding the
  // freed pages. In a group of more than one rank, other ranks reach every input and output (see above): every tensor
  // of another allocator is copied, and the pool keeps the most memory that such buffers took at once (it never frees
  // its slabs). An all-to-all's input that overlaps its output goes through one too (in place, a rank's part for rank
  // j would be where rank j's part for it goes).
  const bool padded = n != numel;
  auto elsewhere = [&](size_t which, const CollectiveSide& s) {
    if (padded || !s.contiguous || (op->kernel && s.address() % 16 != 0)) {
      return true;
    }
    return c.shared[which] && !allocator::owns(s.address());
  };
  // Copies a side's parts into a buffer, or back
  auto copyParts = [&](uintptr_t dst, size_t dstPitch, uintptr_t src, size_t srcPitch, size_t rows) {
    if (rows == 1 || (dstPitch == width && srcPitch == width)) {
      CHECK_CU(cuMemcpyDtoDAsync(dst, src, width * rows, stream));
      return;
    }
    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = src;
    copy.srcPitch = srcPitch;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = dst;
    copy.dstPitch = dstPitch;
    copy.WidthInBytes = width;
    copy.Height = rows;
    CHECK_CU(cuMemcpy2DAsync(&copy, stream));
  };
  auto copyIn = [&](uintptr_t buffer, const CollectiveSide& s) {
    if (s.contiguous) {
      copyParts(buffer, pitch, s.address(), width, s.rows);
    } else {
      for (size_t i : range(s.rows)) {
        CHECK_CU(cuMemcpyDtoDAsync(buffer + i * pitch, s.addresses[i], width, stream));
      }
    }
  };
  auto copyOut = [&](const CollectiveSide& s, uintptr_t buffer) {
    if (s.contiguous) {
      copyParts(s.address(), width, buffer, pitch, s.rows);
    } else {
      for (size_t i : range(s.rows)) {
        CHECK_CU(cuMemcpyDtoDAsync(s.addresses[i], buffer + i * pitch, width, stream));
      }
    }
  };
  uintptr_t inAddress = in.rows != 0 ? in.address() : 0;
  uintptr_t outAddress = out.rows != 0 ? out.address() : 0;
  AllocatedBuffer inputBuffer;
  AllocatedBuffer outputBuffer;
  const bool inPlace =
      in.rows != 0 && out.rows != 0 && in.contiguous && out.contiguous && in.address() == out.address() &&
      in.rows == out.rows &&
      (kind == CollectiveKind::allReduce || kind == CollectiveKind::reduce || kind == CollectiveKind::broadcast);
  if (inPlace) {
    // (an all-reduce, a reduce or a broadcast on its root): one buffer for both
    if (elsewhere(0, in) || elsewhere(1, out)) {
      outputBuffer = allocateCudaBuffer(out.rows * pitch, stream, device, group->getCommPool());
      copyIn(outputBuffer.cudaPointer, in);
      inAddress = outAddress = outputBuffer.cudaPointer;
    }
  } else {
    // (whether some part of an all-to-all's input and some part of its output share memory)
    auto overlaps = [&]() {
      if (kind != CollectiveKind::allToAll) {
        return false;
      }
      if (in.contiguous && out.contiguous) {
        return in.address() < out.address() + out.rows * width && out.address() < in.address() + in.rows * width;
      }
      for (uintptr_t a : in.addresses) {
        for (uintptr_t b : out.addresses) {
          if (a < b + width && b < a + width) {
            return true;
          }
        }
      }
      return false;
    };
    if (in.rows != 0 && (elsewhere(0, in) || overlaps())) {
      inputBuffer = allocateCudaBuffer(in.rows * pitch, stream, device, group->getCommPool());
      copyIn(inputBuffer.cudaPointer, in);
      inAddress = inputBuffer.cudaPointer;
    }
    if (out.rows != 0 && elsewhere(1, out)) {
      outputBuffer = allocateCudaBuffer(out.rows * pitch, stream, device, group->getCommPool());
      outAddress = outputBuffer.cudaPointer;
    }
  }

  // (what other ranks reach is moodist's memory, see elsewhere; customOp does not check, requireMoodistMemory is off)
  CHECK(in.rows == 0 || !c.shared[0] || allocator::owns(inAddress));
  CHECK(out.rows == 0 || !c.shared[1] || allocator::owns(outAddress));

  // The op's tensor data, of its shapes (an all-to-all's are matrices), from the addresses, to launchCustomOp directly:
  // without customOp's checks, which the tensors pass (CUDA tensors on the group's device, see CollectiveSide; the op's
  // dtype, shapes and sizes; moodist's memory where other ranks reach it, aligned for the kernel, see elsewhere), and
  // without a future to wait for (CUDA tensors only: the op is done for the caller once queued on the stream, which sg
  // makes current). Temporary tensors and the future took ~2-3 us per call on GB300.
  auto tensorData = [&](uintptr_t address, const compile_op::Graph::TensorDescr& d) {
    TensorDataPtr t = TensorDataPtr::make();
    t->dtype = static_cast<int>(dtype);
    t->shape.resize(d.shape.size());
    for (size_t i : indices(d.shape)) {
      t->shape[i] = d.shape[i];
    }
    t->dataPtr = address;
    t->dataBytes = d.bytes;
    t->isCuda = true;
    return t;
  };
  std::vector<TensorDataPtr> inputTDs;
  std::vector<TensorDataPtr> outputTDs;
  if (in.rows != 0) {
    inputTDs.push_back(tensorData(inAddress, op->inputs[0]));
  }
  if (out.rows != 0) {
    outputTDs.push_back(tensorData(outAddress, op->outputs[0]));
  }
  launchCustomOp(op, std::move(inputTDs), std::move(outputTDs), stream, scale);

  if (out.rows != 0 && outAddress != out.address()) {
    copyOut(out, outAddress);
  }
  // (the buffers are freed here, for reuse by work after the op and the copies on the stream)
}

namespace {
// The NVLS identity of this rank's CUDA tensor at ptr (see generateKernel): where it is in the group's multicast
// objects, and its multicast address there. In the group's communication pool, at logical offset o: (2 << 60) | o;
// else in moodist's pool, at offset o: (1 << 60) | o; else 0 (and no multicast address). The same identity on two
// ranks means the same offset of the same multicast object. (Bits 62 and 63 stay clear: the kernel tags the words it
// publishes there.)
std::pair<uint64_t, uint64_t> nvlsIdentity(Group* group, uintptr_t ptr, size_t bytes) {
  static_assert(commPoolLogicalSize <= (1ull << 60) && Group::Nvls::poolBytes <= (1ull << 60));
  const Group::Nvls& nvls = group->nvls;
  if (nvls.state != 1) {
    return {0, 0};
  }
  int64_t logical = allocatorCommPoolOffset(group->getCommPool(), ptr, bytes);
  if (logical >= 0) {
    return {(2ull << 60) | (uint64_t)logical, nvls.commMcBase + logical};
  }
  const uintptr_t reserved = allocatorGetReservedBase();
  if (reserved && ptr >= reserved && ptr - reserved + bytes <= Group::Nvls::poolBytes) {
    return {(1ull << 60) | (ptr - reserved), nvls.mcBase + (ptr - reserved)};
  }
  return {0, 0};
}
} // namespace

// ============================================================================
// ProcessGroupImpl::launchCustomOp - launch a compiled custom operation
// ============================================================================

ProcessGroupImpl::CustomOpLaunch ProcessGroupImpl::launchCustomOp(const std::shared_ptr<CustomOpDescriptor>& op,
    std::vector<TensorDataPtr> inputTDs, std::vector<TensorDataPtr> outputTDs, CUstream stream, double scale) {
  std::unique_lock l(threadUnsafe);

  uint32_t stepValue = getNextStepValue();
  uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
  CHECK(stepValue < 0x80000000);

  sync(stepValue);

  // for (size_t peerIndex : group->peerIndices) {
  //   peerWriteDyn(concurrencyIndex, peerIndex, opTypeCompileOpLocal, stepValue, 0, 0);
  // }
  // for (size_t peerIndex : group->peerIndices) {
  //   peerWaitDyn(concurrencyIndex, peerIndex, opTypeCompileOpLocal, stepValue, 0);
  // }
  // freePendingIpcEvents();

  auto future = FutureImplSharedPtr::make();

  bool anyCuda = false;
  bool anyCpu = false;

  for (auto& t : inputTDs) {
    anyCuda |= t->isCuda;
    anyCpu |= !t->isCuda;
  }
  for (auto& t : outputTDs) {
    anyCuda |= t->isCuda;
    anyCpu |= !t->isCuda;
  }

  // Reductions across NVLink domains or in two phases: the scratch tensor (see CustomOpDescriptor::scratchBytes), the
  // last tensor. From moodist's CUDA allocator (the ranks of a multi-node NVLink fabric can map no other memory), on
  // the op's stream, and freed at the end of this call: it is then reused only by work after the op on that stream,
  // which comes after the RDMA transfers into and from it (the stream waits for the CPU thread, see below, which
  // starts once the stream has reached the op). It is a CUDA tensor of the op like the others: a rank can have one
  // without any tensors of its own (one that computes a partial result of its domain's inputs for another domain, see
  // crossDomainReductions), and its stream must still wait for the CPU thread, or the scratch tensor could be reused
  // while other ranks still read or write it.
  AllocatedBuffer scratch;
  if (op->scratchBytes != 0) {
    scratch = allocateCudaBuffer(op->scratchBytes, stream, group->deviceIndex);
    TensorDataPtr td = TensorDataPtr::make();
    td->dtype = static_cast<int>(op->dtype);
    td->shape.resize(1);
    td->shape[0] = (int64_t)(op->scratchBytes / wrapperApi.dtypeSize(op->dtype));
    td->dataPtr = scratch.cudaPointer;
    td->dataBytes = op->scratchBytes;
    td->isCuda = true;
    anyCuda = true;
    outputTDs.push_back(std::move(td));
  }

  bool anyRdma = op->anyRankNeedsRdma;

  if (op->cpuSync) {
    anyCpu = true;
    CHECK(anyRdma);
  }

  // On the stream, the whole op (cpuIn write, kernel, cpuOut wait) comes after the previous op of its concurrency
  // slot, whose cpuIn and cpuOut it reuses (otherwise an op on another stream could satisfy this op's cpuIn wait
  // before this stream gets here), and records the slot's event when done, which wait() uses (see customOp). If the
  // slot's previous op ran on this stream, stream order already puts this op after it: no wait (about 0.3 us of host
  // time and 0.25 us of the GPU's per op on GB300). Streams by id: a destroyed stream's handle can come back, with
  // its last ops still to run; its id does not. (Only these ops wait for and record the slot's event.) The stream
  // does not capture (the callers check, see requireNotCapturing); should the query fail anyway, the op waits.
  std::optional<EventSerializer> slotSerializer;
  if (anyCuda) {
    unsigned long long streamId = 0;
    const bool known = cudaApi.streamGetId(stream, &streamId) == CUDA_SUCCESS;
    const bool wait = !known || concurrencyStreams[concurrencyIndex] != streamId;
    slotSerializer.emplace(concurrencyEvents[concurrencyIndex], stream, wait);
    concurrencyStreams[concurrencyIndex] = known ? std::optional(streamId) : std::nullopt;
  }
  // Whether this launch writes the kernel's profile (MOODIST_COMPILE_OP_PROFILE, see generateKernel)
  bool dumpProfile = false;

  if (anyCuda && anyRdma) {
    memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue);
    memFlush(stream);
  }

  // Without CPU tensors the op is stream-ordered, RDMA part included: the stream waits for cpuOut below, and wait()
  // only joins another current stream to it (see customOp). Blocking the caller until the CPU thread is done would
  // make every op synchronous: the CPU thread then runs dry and sleeps after each op, and every op pays for waking
  // it up again on every rank (occasionally milliseconds).
  if (!anyCpu) {
    future->done = 1;
  }

  if (op->kernel) {
    IpcMapper* ipcMapper = &*group->ipcMapper;
    std::shared_lock unmapLock(unmapMemoryMutex);

    Vector<uintptr_t> tensorAddrs;
    tensorAddrs.reserve(inputTDs.size() + outputTDs.size());
    for (auto& v : inputTDs) {
      tensorAddrs.push_back(v->data());
    }
    for (auto& v : outputTDs) {
      tensorAddrs.push_back(v->data());
    }

    // The addresses at which the peers map this rank's tensors (which the kernel publishes to them, see
    // generateKernel), and in NVLS ops the tensors' NVLS identities and multicast addresses: those of the op's last
    // launch if its tensors are at the same addresses (moodist never unmaps memory it has shared, an address stays in
    // its pool slab, and the op's tensor sizes are fixed), else from the IPC mapper and the allocator (a lookup per
    // peer and tensor: about 1.5 us per launch with 31 peers on GB300)
    CustomOpDescriptor::LaunchCache& cache = op->launchCache;
    const bool cached = cache.valid && cache.tensorAddrs.size() == tensorAddrs.size() &&
                        std::equal(tensorAddrs.begin(), tensorAddrs.end(), cache.tensorAddrs.begin());
    Vector<uintptr_t> mappedAddrs(op->graph.remoteCudaTensorMappings.size());
    if (cached) {
      std::copy(cache.mappedAddrs.begin(), cache.mappedAddrs.end(), mappedAddrs.begin());
    } else {
      cache.valid = false;
      for (size_t i : indices(mappedAddrs)) {
        auto& m = op->graph.remoteCudaTensorMappings[i];
        auto& tensor = m.tensorIndex >= inputTDs.size() ? outputTDs.at(m.tensorIndex - inputTDs.size())
                                                        : inputTDs.at(m.tensorIndex);
        ipcMapper->requestAddressRank(m.rank, tensor->data(), tensor->bytes(), &mappedAddrs[i]);
      }
      ipcMapper->wait();
      // (the kernel publishes these to the peers tagged in bits 62 and 63, see generateKernel)
      for (uintptr_t a : mappedAddrs) {
        CHECK(a >> 62 == 0);
      }
      cache.tensorAddrs.assign(tensorAddrs.begin(), tensorAddrs.end());
      cache.mappedAddrs.assign(mappedAddrs.begin(), mappedAddrs.end());
    }

    if (tensorAddrs.empty()) {
      tensorAddrs.push_back(0);
    }
    if (mappedAddrs.empty()) {
      mappedAddrs.push_back(0);
    }
    // NVLS (see generateKernel): the tensors' NVLS identities and multicast addresses, and the op's step counter
    // before this launch
    uint32_t nvlsCounterBase = 0;
    Vector<uint64_t> nvlsTensors;
    if (op->kernelHandle->nvls) {
      KernelHandle& kh = *op->kernelHandle;
      nvlsTensors.resize(2 * std::max(inputTDs.size() + outputTDs.size(), (size_t)1));
      if (cached) {
        CHECK(cache.nvlsTensors.size() == nvlsTensors.size());
        std::copy(cache.nvlsTensors.begin(), cache.nvlsTensors.end(), nvlsTensors.begin());
      } else {
        size_t n = 0;
        for (auto* list : {&inputTDs, &outputTDs}) {
          for (auto& v : *list) {
            std::tie(nvlsTensors[n], nvlsTensors[n + 1]) = nvlsIdentity(&*group, v->data(), v->bytes());
            n += 2;
          }
        }
        cache.nvlsTensors.assign(nvlsTensors.begin(), nvlsTensors.end());
      }
      nvlsCounterBase = kh.nvlsLaunches.at(concurrencyIndex)++ * kh.nvlsIncrement;
    }
    cache.valid = true;

    CompileOpCopyParameters params{.stepValue = stepValue,
        .concurrencyIndex = concurrencyIndex,
        .nvlsCounterBase = nvlsCounterBase,
        .scale = scale};
    // (with the parameter buffer, the slot's copy is filled first: the slot's previous op, whose kernel read it, is
    // done on the GPU before this one starts, see slotSerializer)
    launchCompileOpKernel(
        *op->kernelHandle, *op->kernel, op->config, stream, params, tensorAddrs, mappedAddrs, nvlsTensors);

    // (the profile is written below, once the CPU thread has the op's RDMA part: the kernel may wait for it)
    dumpProfile = op->kernelHandle->profilingBytesPerThread != 0 && op->kernelHandle->profilingCountdown-- == 0;
  }

  if (anyCuda && anyRdma) {
    memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
    memFlush(stream);
  }
  slotSerializer.reset();

  if (anyRdma) {
    StreamData& sd = group->getCpuStreamData(concurrencyIndex);
    QueueEntryCustom* e = group->cpuThread->freelistCustom.pop();
    e->task = taskCustom;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->op = op;
    e->future = future;
    e->anyCuda = anyCuda;
    e->anyCpu = anyCpu;

    CHECK(e->inputs.empty());
    CHECK(e->outputs.empty());

    e->inputs = std::move(inputTDs);
    e->outputs = std::move(outputTDs);

    // For the trace (entries are reused, so always set)
    e->enqueueTime = getProfilingEnabled() ? std::chrono::system_clock::now() : std::chrono::system_clock::time_point{};
    e->enqueueCpu = e->enqueueTime.time_since_epoch().count() != 0 ? sched_getcpu() : -1;
    group->cpuThread->enqueue(e);
  }
  if (dumpProfile) {
    // (on the group's device, as the launch was)
    StreamGuard sg(stream, group->deviceIndex);
    CHECK_CU(cuCtxSynchronize());
    dumpProfiling(&*group, op);
  }

  return {std::move(future), concurrencyIndex, anyCuda};
}

// ============================================================================
// ProcessGroupImpl::customOp - execute a compiled custom operation
// ============================================================================

SharedPtr<ApiFuture> ProcessGroupImpl::customOp(std::shared_ptr<CustomOpDescriptor> op, TensorPtr* inputs,
    size_t nInputs, TensorPtr* outputs, size_t nOutputs, CUstream stream, double scale) {
  if (nInputs != op->inputs.size()) {
    throw std::runtime_error(
        fmt::sprintf("moodist: custom op expected %d inputs, but got %d", op->inputs.size(), nInputs));
  }
  if (nOutputs != op->outputs.size()) {
    throw std::runtime_error(
        fmt::sprintf("moodist: custom op expected %d outputs, but got %d", op->outputs.size(), nOutputs));
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

  for (size_t i : range(nInputs)) {
    auto t = getTensorDataFromPtr(inputs[i], group.get());
    const auto& ot = op->inputs[i];
    REQUIRE(t->dtype == static_cast<int>(op->dtype),
        "moodist: custom op input[%zu] has wrong dtype: got %s, expected %s", i, formatDType(t->dtype),
        formatDType(static_cast<int>(op->dtype)));
    REQUIRE(t->shape == ot.shape, "moodist: custom op input[%zu] has wrong shape: got %s, expected %s", i,
        formatShape(t->shape), formatShape(ot.shape));
    REQUIRE(t->bytes() == ot.bytes, "moodist: custom op input[%zu] has wrong size: got %zu bytes, expected %zu bytes",
        i, t->bytes(), ot.bytes);
    bool expectedCuda = ot.device == DeviceType::CUDA;
    REQUIRE(t->isCuda == expectedCuda, "moodist: custom op input[%zu] has wrong device: got %s, expected %s", i,
        t->isCuda ? "cuda" : "cpu", expectedCuda ? "cuda" : "cpu");
    inputTDs.push_back(std::move(t));
  }
  for (size_t i = 0; i < nOutputs; ++i) {
    auto t = getTensorDataFromPtr(outputs[i], group.get());
    const auto& ot = op->outputs[i];
    REQUIRE(t->dtype == static_cast<int>(op->dtype),
        "moodist: custom op output[%zu] has wrong dtype: got %s, expected %s", i, formatDType(t->dtype),
        formatDType(static_cast<int>(op->dtype)));
    REQUIRE(t->shape == ot.shape, "moodist: custom op output[%zu] has wrong shape: got %s, expected %s", i,
        formatShape(t->shape), formatShape(ot.shape));
    REQUIRE(t->bytes() == ot.bytes, "moodist: custom op output[%zu] has wrong size: got %zu bytes, expected %zu bytes",
        i, t->bytes(), ot.bytes);
    bool expectedCuda = ot.device == DeviceType::CUDA;
    REQUIRE(t->isCuda == expectedCuda, "moodist: custom op output[%zu] has wrong device: got %s, expected %s", i,
        t->isCuda ? "cuda" : "cpu", expectedCuda ? "cuda" : "cpu");
    outputTDs.push_back(std::move(t));
  }

  // The copy kernel needs 16-byte aligned tensor addresses (TMA bulk copies; edge sizes and offsets are
  // checked at compile time). User ops take CUDA memory from the moodist CUDA allocator only: fabric peers
  // can map nothing else, and moodist never unregisters or unmaps memory it has registered or shared (doing
  // so synchronizes the device and can deadlock), so registered memory must never go back to CUDA.
  auto checkTensors = [&](const char* what, const std::vector<TensorDataPtr>& tds) {
    for (size_t i : indices(tds)) {
      const auto& t = tds[i];
      if (!t->isCuda || t->bytes() == 0) {
        continue;
      }
      REQUIRE(!op->kernel || t->data() % 16 == 0, "moodist: custom op %s[%zu] data pointer %#x is not 16-byte aligned",
          what, i, t->data());
      REQUIRE(!op->requireMoodistMemory || allocator::owns(t->data()),
          "moodist: custom op %s[%zu] (CUDA memory at %#x) was not allocated by the moodist CUDA allocator, which "
          "compile_op requires (call moodist.enable_cuda_allocator() before allocating it)",
          what, i, t->data());
    }
  };
  checkTensors("input", inputTDs);
  checkTensors("output", outputTDs);

  // An op with CUDA tensors (a scratch tensor counts, see launchCustomOp) runs on the stream: not while it captures,
  // checked before the copies
  auto isCuda = [](const TensorDataPtr& t) {
    return t->isCuda;
  };
  if (op->scratchBytes != 0 || std::ranges::any_of(inputTDs, isCuda) || std::ranges::any_of(outputTDs, isCuda)) {
    requireNotCapturing(stream, "compile_op ops of CUDA tensors");
  }

  // Fast path: all-local, all-CUDA, no copies — skip CPU thread entirely
  // if (op->allLocal) {
  // if (true) {
  //   return executeLocalOnly(
  //       *op, inputTDs, outputTDs, inputs, nInputs, outputs, nOutputs, concurrencyIndex, stepValue, stream);
  // }

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
    inputTDs.push_back(std::move(td));
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
    outputTDs.push_back(std::move(td));
    // Keep tensor alive
    result->holdTensors.push_back(std::move(t));
  }

  // The copies come from PyTorch's current CUDA allocator, and other ranks map them (or RDMA registers them) like the
  // tensors themselves, so in user ops they must be moodist's memory too (see checkTensors): they are once
  // moodist.enable_cuda_allocator() has made moodist's allocator PyTorch's, but not when only the tensors are moodist's
  // (e.g. from moodist.comm_empty). Checked before the launch takes the op's step value. TODO: take the copies from
  // moodist's allocator (see docs/internals/compile_op.md, Known issues).
  auto checkCopies = [&](const char* what, std::span<const TensorDataPtr> copies, const auto& of) {
    for (size_t i : indices(copies)) {
      const auto& t = copies[i];
      REQUIRE(!op->requireMoodistMemory || !t->isCuda || t->bytes() == 0 || allocator::owns(t->data()),
          "moodist: custom op %s[%d] has a piece that is not contiguous in it, which the op copies through a "
          "contiguous tensor from PyTorch's CUDA allocator; such ops need that to be the moodist CUDA allocator "
          "(call moodist.enable_cuda_allocator(); tensors from moodist.comm_empty alone are not enough)",
          what, of[i].index);
    }
  };
  checkCopies("input", std::span<const TensorDataPtr>(inputTDs).subspan(nInputs), op->inputCopies);
  checkCopies("output", std::span<const TensorDataPtr>(outputTDs).subspan(nOutputs), op->outputCopies);

  // The op (with the op's stream current on the group's device for the kernel's launch, see launchCustomOp)
  CustomOpLaunch launch;
  {
    std::optional<StreamGuard> sg;
    if (op->kernel) {
      sg.emplace(stream, group->deviceIndex);
    }
    launch = launchCustomOp(op, std::move(inputTDs), std::move(outputTDs), stream, scale);
  }
  const uint32_t concurrencyIndex = launch.concurrencyIndex;
  const bool anyCuda = launch.anyCuda;

  result->impl = std::move(launch.future);
  // Hold original tensors alive
  for (size_t i = 0; i < nInputs; ++i) {
    result->holdTensors.push_back(inputs[i]);
  }
  for (size_t i = 0; i < nOutputs; ++i) {
    result->holdTensors.push_back(outputs[i]);
  }

  // wait() (and dropping the future without it): if the current stream is not the one the op ran on, it waits for
  // the slot's event, which the op recorded when done on its stream (kernel and RDMA part). If the slot has been
  // reused since, the event is a later op's, which only waits longer. On the op's own stream nothing is needed.
  // Not while it captures a CUDA graph: the wait for an event recorded outside the capture would fail and invalidate
  // the capture, and the copies back into the outputs (below) would be captured instead of run. It raises then,
  // before any of it (futureDestroy logs the error of a future dropped meanwhile).
  auto selfPtr = share(this);
  auto joinCurrentStream = [selfPtr, concurrencyIndex, stream]() {
    CUstream current = wrapperApi.cudaGetCurrentStream();
    if (streamCapturing(current)) {
      throw std::runtime_error("moodist: the future of a compile_op op with CUDA tensors cannot be waited for (or "
                               "dropped) while the current stream is capturing a CUDA graph: wait for it before the "
                               "capture begins");
    }
    if (current != stream) {
      selfPtr->concurrencyEvents[concurrencyIndex].wait(current);
    }
  };
  if (!outputCopyTensors.empty()) {
    // Need to copy temporary outputs back to original output tensors after completion
    // Capture copies of the output TensorPtrs since the outputs pointer may be invalid later
    std::vector<TensorPtr> outputsCopy(outputs, outputs + nOutputs);
    result->waitDoneCallback = [joinCurrentStream, anyCuda, copyTensors = std::move(outputCopyTensors), op,
                                   outputsCopy = std::move(outputsCopy)]() {
      if (anyCuda) {
        joinCurrentStream();
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
    result->waitDoneCallback = std::move(joinCurrentStream);
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
