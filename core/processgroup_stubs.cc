// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup implementation for the API layer.
// This file implements ProcessGroupImpl and the API functions.
// Collective operations are stubs until migration is complete.

#include "common.h"
#include "cputhread.h"
#include "group.h"
#include "moodist_api.h"
#include "processgroup_api.h"
#include "queue.h"
#include "serialization.h"
#include "setup_comms.h"
#include "shared_ptr.h"
#include "synchronization.h"

#include <atomic>
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

// ============================================================================
// Helper types
// ============================================================================

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

  f();
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

void processGroupImplAllGather(
    ProcessGroupImpl* /*impl*/, TensorPtr& /*output*/, const TensorPtr& /*input*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplAllGather: not yet implemented");
}

void processGroupImplReduceScatter(ProcessGroupImpl* /*impl*/, TensorPtr& /*output*/, const TensorPtr& /*input*/,
    ReduceOp /*reduceOp*/, CUstream /*stream*/, float /*premulValue*/) {
  throw std::runtime_error("processGroupImplReduceScatter: not yet implemented");
}

void processGroupImplAllreduce(ProcessGroupImpl* /*impl*/, TensorPtr& /*tensor*/, ReduceOp /*reduceOp*/,
    CUstream /*stream*/, float /*premulValue*/) {
  throw std::runtime_error("processGroupImplAllreduce: not yet implemented");
}

void processGroupImplBroadcast(
    ProcessGroupImpl* /*impl*/, TensorPtr& /*tensor*/, int /*sourceRank*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplBroadcast: not yet implemented");
}

void processGroupImplReduce(
    ProcessGroupImpl* /*impl*/, TensorPtr& /*tensor*/, int /*destRank*/, ReduceOp /*reduceOp*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplReduce: not yet implemented");
}

void processGroupImplBarrier(ProcessGroupImpl* /*impl*/) {
  throw std::runtime_error("processGroupImplBarrier: not yet implemented");
}

void processGroupImplScatter(ProcessGroupImpl* /*impl*/, std::span<TensorPtr> /*inputs*/, TensorPtr& /*output*/,
    int /*sourceRank*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplScatter: not yet implemented");
}

void processGroupImplGather(ProcessGroupImpl* /*impl*/, std::span<TensorPtr> /*outputs*/, const TensorPtr& /*input*/,
    int /*destRank*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplGather: not yet implemented");
}

void processGroupImplAllToAll(ProcessGroupImpl* /*impl*/, std::span<TensorPtr> /*outputs*/,
    std::span<TensorPtr> /*inputs*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplAllToAll: not yet implemented");
}

void processGroupImplCudaBarrier(ProcessGroupImpl* /*impl*/, CUstream /*stream*/) {
  throw std::runtime_error("processGroupImplCudaBarrier: not yet implemented");
}

void processGroupImplShutdown(ProcessGroupImpl* impl) {
  if (impl && impl->group && impl->group->cpuThread) {
    impl->group->cpuThread->kill(false);
  }
}

} // namespace moodist
