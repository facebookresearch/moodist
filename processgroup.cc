// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "processgroup.h"
#include "allgather.h"
#include "async.h"
#include "backend.h"
#include "common.h"
#include "cputhread.h"
#include "cuda_copy.h"
#include "group.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "reduce_scatter.h"
#include "serialization.h"
#include "setup_comms.h"
#include "synchronization.h"
#include "vector.h"

#include <ATen/ops/from_blob.h>
#include <ATen/ops/pad.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/types.h>

#include <cublas.h>

namespace moodist {

extern bool profilingEnabled;

async::Scheduler& scheduler = *new async::Scheduler(1);

void throwCuHelper(CUresult error, const char* file, int line) {
  throwCu(error, file, line);
}

void globalsDtor();

struct GlobalsDestroyed {
  bool value = false;
  ~GlobalsDestroyed() {
    value = true;
    globalsDtor();
  }
} globalsDestroyed;

struct ProcessGroupImpl;
std::mutex& activeProcessGroupsMutex = *new std::mutex();
std::vector<ProcessGroupImpl*>& activeProcessGroups = *new std::vector<ProcessGroupImpl*>();

std::atomic_bool globalPreferKernelLess = false;

std::once_flag freeMemoryCallbackOnceFlag;
void registerFreeMemoryCallback();

// Work objects can outlive ProcessGroupImpl and hold a pointer to WorkStream.
// A proper solution would be to hold a shared_ptr to WorkStreams in Work.
// Instead, we intentionally leak WorkStreams objects through a circular std::shared_ptr.
struct WorkStreams;
struct WorkStream {
  IntrusiveListLink<WorkStream> link;
  CUstream stream = nullptr;
  CUevent event = nullptr;
  std::atomic<CUstream> joinedStream = nullptr;
  std::shared_ptr<WorkStreams> owner;
  WorkStream() = default;
  WorkStream(const WorkStream&) = delete;
  WorkStream& operator=(const WorkStream&) = delete;
  WorkStream(WorkStream&& n) {
    *this = std::move(n);
  }
  WorkStream& operator=(WorkStream&& n) {
    std::swap(stream, n.stream);
    std::swap(event, n.event);
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
  ReusableHandle(ReusableHandle&& n) {
    *this = std::move(n);
  }
  ReusableHandle& operator=(ReusableHandle& n) = delete;
  ReusableHandle& operator=(ReusableHandle&& n) {
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

SharedSpinMutex unmapMemoryMutex;

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

struct ProcessGroupImpl {
  size_t rank = 0;
  size_t size = 0;

  std::shared_ptr<Group> group;

  ThreadUnsafe threadUnsafe;
  uint32_t nextStepValue = 0x1000;
  uint32_t nextConcurrencyIndex = 0;

  HashMap<CUstream, std::shared_ptr<WorkStreams>> workStreams;

  std::array<Event, maxConcurrency> concurrencyEvents;

  Event localEvent;
  std::optional<Stream> peerMemcpyStream;
  std::optional<Stream> peerIncomingMemcpyStream;
  // std::optional<Stream> reduceStream;

  HashMap<CUstream, Stream> copyStream;

  std::array<std::optional<Stream>, maxChunks> peerMemcpyStreamPerChunk;

  std::array<std::optional<Stream>, 4> reduceStreamArr;
  size_t reduceCounter = 0;

  std::vector<std::shared_ptr<Queue>> queues;

  bool preferKernelLess = globalPreferKernelLess;

  Reusable<IpcEvent> ipcEvents;
  Reusable<IpcEvent> pendingIpcEvents;

  int streamPriority = -1;

  ProcessGroupImpl(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size) : rank(rank), size(size) {
    CHECK(rank >= 0 && size > 0 && rank < size);

    log.init();

    group = std::make_shared<Group>(rank, size);

    init(store);

    {
      std::call_once(freeMemoryCallbackOnceFlag, &registerFreeMemoryCallback);
      std::lock_guard l(activeProcessGroupsMutex);
      activeProcessGroups.push_back(this);
    }

    for (auto& v : concurrencyEvents) {
      v = Event::create();
    }

    localEvent = Event::create();

    if (size - 1 == group->peerIndices.size()) {
      streamPriority = -10;
    }
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
      ptr = std::make_unique<WorkStreams>();
    }
    return &*ptr;
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
    if (!w && n >= maxConcurrency) {
      w = &ws->free.front();
      ws->free.pop_front();
    }
    if (!w) {
      auto ptr = std::make_unique<WorkStream>();
      w = &*ptr;
      ws->all.push_back(std::move(ptr));

      w->owner = workStreams[stream];
      CHECK(&*w->owner == ws);
      // CHECK_CU(cuStreamCreate(&w->stream, CU_STREAM_NON_BLOCKING));
      CHECK_CU(cuStreamCreateWithPriority(&w->stream, CU_STREAM_NON_BLOCKING, streamPriority));
      CHECK_CU(cuEventCreate(&w->event, CU_EVENT_DISABLE_TIMING));
      log.info("New work stream %#x created for stream %#x\n", (uintptr_t)w->stream, (uintptr_t)stream);
    }
    w->joinedStream.store(w->stream, std::memory_order_relaxed);
    l.unlock();
    return w;
  }

  void freeMemory(bool execute) {
    group->ipcMapper->enqueueUnmapAll();
    // if (execute) {
    //   CHECK_CU(cuCtxSynchronize());
    //   group->ipcMapper->executeQueuedUnmaps();
    // }
  }

  uint32_t getNextStepValue() {
    uint32_t r = std::exchange(nextStepValue, nextStepValue + 0x1000);
    if (r < 0x80000000) {
      return r;
    }

    resetStepValue();
    return 1;
  }
  void resetStepValue() {
    // We need to reset the 32 bit step value to prevent it from overflowing.
    // We could avoid this by changing the way we compare the step values everywhere,
    // but this is simpler.

    log.verbose("Reset step value begin\n");
    while (group->cpuThread->busy) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    internalBarrier();

    CHECK_CU(cuCtxPushCurrent(group->cuContext));
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
    CUcontext poppedContext;
    CHECK_CU(cuCtxPopCurrent(&poppedContext));

    log.info("Reset step value done, syncing\n");
    internalBarrier();
    log.verbose("Reset step value end\n");
  }

  struct TraceEvent {
    std::string name;
    std::chrono::system_clock::time_point begin;
    std::chrono::system_clock::time_point end;
  };
  std::vector<TraceEvent> traceEvents;

  std::chrono::system_clock::time_point beginning = std::chrono::system_clock::now();

  std::string currentTraceName = "";
  std::chrono::system_clock::time_point currentTraceBegin = beginning;
  int threadId = syscall(SYS_gettid);

  void trace(std::string name) {
    // if (opCount < 1000 || opCount >= 1200) {
    if (!profilingEnabled) {
      if (!traceEvents.empty()) {
        std::string fn = fmt::sprintf("moodist-trace-pg-%d.json", rank);
        FILE* f = fopen(fn.c_str(), "wb");
        CHECK(f != nullptr);
        fmt::fprintf(
            f, R"({"traceEvents": [)"
               "\n");
        bool first = true;
        for (auto& e : traceEvents) {
          if (!first) {
            fmt::fprintf(f, ",\n");
          } else {
            first = false;
          }
          fmt::fprintf(
              f, R"({"ph": "X", "name": "%s", "ts": %d, "dur": %d, "tid": %d})", e.name,
              std::chrono::duration_cast<std::chrono::microseconds>(e.begin.time_since_epoch()).count(),
              std::chrono::duration_cast<std::chrono::microseconds>(e.end - e.begin).count(), threadId);
        }
        fmt::fprintf(f, "]}\n");
        fclose(f);
        fmt::printf("Chrome trace dumped to %s\n", fn);
        traceEvents.clear();
      }
      return;
    }
    auto now = std::chrono::system_clock::now();
    if (traceEvents.empty() && currentTraceName.empty()) {
      beginning = now;
    }
    if (!currentTraceName.empty()) {
      traceEvents.emplace_back();
      TraceEvent& e = traceEvents.back();
      e.name = std::move(currentTraceName);
      e.begin = currentTraceBegin;
      e.end = now;
    }
    currentTraceBegin = now;
    currentTraceName = std::move(name);
  };
#define trace(name)

  ~ProcessGroupImpl() {
    trace("");
    std::lock_guard l(activeProcessGroupsMutex);
    auto i = std::find(activeProcessGroups.begin(), activeProcessGroups.end(), this);
    if (i != activeProcessGroups.end()) {
      activeProcessGroups.erase(i);
    }
  }

  void init(const c10::intrusive_ptr<::c10d::Store>& store) {

    log.verbose("%d/%d: init\n", rank, size);

    auto start = std::chrono::steady_clock::now();

    std::unique_lock l(threadUnsafe);

    auto f = [&]() {
      std::string key = randomName();

      auto tovec = [&](std::string str) { return std::vector<uint8_t>(str.begin(), str.end()); };
      auto fromvec = [&](std::vector<uint8_t> vec) { return std::string(vec.begin(), vec.end()); };

      auto set = [&](std::string key, std::string value) { store->set(key, tovec(value)); };
      auto get = [&](std::string key) { return fromvec(store->get(key)); };

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
    while (cpuDone == 0) {
      futexWait(&cpuDone, 0, std::chrono::seconds(10));
    }
  }

  void barrier() {
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

  void sync(uint32_t stepValue) {
    IpcMapper* ipcMapper = &*group->ipcMapper;
    const auto& peerIndices = group->peerIndices;

    bool unmap = ipcMapper->hasQueuedUnmaps.load(std::memory_order_relaxed);
    for (size_t peerIndex : peerIndices) {
      ipcMapper->push(peerIndex, std::make_pair(stepValue, unmap));
    }
    for (size_t peerIndex : peerIndices) {
      auto v = ipcMapper->pop<std::pair<uint32_t, bool>>(peerIndex);
      if (v.first != stepValue) {
        throw std::runtime_error(fmt::sprintf("Moodist step value mismatch: local %#x, peer %#x", stepValue, v.first));
      }
      unmap |= v.second;
    }

    if (unmap) {
      log.info("Unmapping memory at step %#x as requested by cuda allocator\n", stepValue);
      ipcMapper->executeQueuedUnmaps();

      sync(stepValue);
      return;
    }
    ipcMapper->setStepValue(stepValue);
  }

  AllocatedBuffer alloccuda(size_t size, CUstream stream) {
    if (size == 0) {
      return {};
    }
    if (size < 16) {
      size = 16;
    }
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
    AllocatedBuffer buffer;
    buffer.dataPtr = c10::cuda::CUDACachingAllocator::get()->allocate(size);
    buffer.bytes = size;
    buffer.cudaPointer = (uintptr_t)buffer.dataPtr->get();
    CHECK(buffer.cudaPointer != 0);
    return buffer;
  };

  bool localIpcRanksCalculated = false;
  std::vector<size_t> localIpcRanks;
  size_t myLocalRank = -1;
  std::vector<size_t>& getLocalIpcRanks() {
    if (!localIpcRanksCalculated) {
      localIpcRanksCalculated = true;
      std::vector<std::pair<size_t, size_t>> sorted;
      for (size_t peerIndex : group->peerIndices) {
        sorted.emplace_back(group->ipcRanks[peerIndex], peerIndex);
      }
      sorted.emplace_back(rank, -1);
      std::sort(sorted.begin(), sorted.end());
      localIpcRanks.resize(group->peerIndices.size());
      for (auto& v : sorted) {
        CHECK((v.first == rank) == (v.second == -1));
        if (v.first != rank) {
          localIpcRanks[v.second] = &v - sorted.data();
        } else {
          myLocalRank = &v - sorted.data();
        }
      }
      log.info(
          "localIpcRanks is %s, my local rank is %d\n", fmt::to_string(fmt::join(localIpcRanks, ", ")), myLocalRank);
    }
    return localIpcRanks;
  }

  template<typename Extra = void (*)(size_t)>
  void local_all_gather_impl(
      uint32_t concurrencyIndex, uint32_t stepValue, uintptr_t inputAddress, size_t inputBytes, uintptr_t outputAddress,
      size_t outputBytes, CUstream stream, const std::vector<size_t>& ipcRanks, size_t myLocalRank,
      Extra&& extra = [](size_t) {}) {
    // const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;
    CHECK(ipcRanks.size() == peerIndices.size());

    // log.info(
    //     "%d (%d): local all gather, %d %d %#x %d %#x %d\n", rank, myLocalRank, concurrencyIndex, stepValue,
    //     inputAddress, inputBytes, outputAddress, outputBytes);

    IpcMapper* ipcMapper = &*group->ipcMapper;

    std::array<uintptr_t, 8> peerMappedAddresses;
    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(i, inputAddress, inputBytes, &peerMappedAddresses[i], true);
    }
    ipcMapper->wait();

    std::array<uintptr_t, 8> peerAddresses;
    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(
          concurrencyIndex, peerIndex, opTypeAllGatherLocalCuda, stepValue, peerMappedAddresses[peerIndex],
          outputBytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeAllGatherLocalCuda, stepValue, outputBytes);
    }

    localEvent.record(stream);

    size_t n = ipcRanks.size() + 1;
    size_t chunkSize = (outputBytes + n - 1) / n;

    if (true) {
      for (size_t peerIndex : peerIndices) {
        ipcMapper->streamRecord(peerIndex, stream);
      }
      for (size_t peerIndex : peerIndices) {
        ipcMapper->streamWait(peerIndex, stream);
      }
      for (size_t peerIndex : peerIndices) {
        size_t offset = chunkSize * ipcRanks[peerIndex];
        size_t nbytes = ipcRanks[peerIndex] == ipcRanks.size() ? outputBytes - offset : chunkSize;
        CHECK_CU(cuMemcpyDtoDAsync(outputAddress + offset, peerAddresses[peerIndex], nbytes, stream));
        ipcMapper->streamRecord(peerIndex, stream);
      }
      for (size_t peerIndex : peerIndices) {
        ipcMapper->streamWait(peerIndex, stream);
      }
    } else {
      for (size_t peerIndex : peerIndices) {
        size_t incomingPeerIndex = peerIndices.size() - 1 - peerIndex;
        // log.info("rank %d: outgoing %d, incoming %d\n", rank, peerIndex, incomingPeerIndex);

        ipcMapper->streamRecord(incomingPeerIndex, stream);
        ipcMapper->streamWait(peerIndex, stream);

        size_t offset = chunkSize * ipcRanks[peerIndex];
        size_t nbytes = ipcRanks[peerIndex] == ipcRanks.size() ? outputBytes - offset : chunkSize;

        CHECK_CU(cuMemcpyDtoDAsync(outputAddress + offset, peerAddresses[peerIndex], nbytes, stream));

        ipcMapper->streamRecord(peerIndex, stream);
        ipcMapper->streamWait(incomingPeerIndex, stream);
      }
    }

    {
      size_t offset = chunkSize * myLocalRank;
      size_t nbytes = myLocalRank == ipcRanks.size() ? outputBytes - offset : chunkSize;
      CHECK(inputBytes == nbytes);

      if (outputAddress + offset != inputAddress) {
        auto copyStream = getCopyStream(stream);
        localEvent.wait(copyStream);
        CHECK_CU(cuMemcpyDtoDAsync(outputAddress + offset, inputAddress, nbytes, copyStream));
        localEvent.record(copyStream);

        localEvent.wait(stream);
      }
    }
  }

  struct Copy2d {
    size_t offset;
    size_t length;
    size_t pitch;
    size_t num;
  };

  std::optional<std::array<Vector<Copy2d>, 8>> allGather2dCopies;

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
      // log.info("peer %d recvs is %s\n", peerIndex, fmt::to_string(fmt::join(recvs, ", ")));
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
        // log.info("nextseq -> %d %d\n", beginSource, endSource);
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

        // log.info(
        //     "%d: copy from peer %d, %d seqs of length %d with pitch %d and offset %d\n", rank, peerIndex, numSeqs,
        //     seqLen, pitch, c.offset);

        CHECK(numSeqs >= 1 && numSeqs < size);
        CHECK(seqLen >= 1 && seqLen < size);
        CHECK(pitch >= seqLen && pitch < size);
      }
    }
  }

  void kernelLess_all_gather_impl(
      uint32_t concurrencyIndex, uint32_t stepValue, uintptr_t inputAddress, size_t inputBytes, uintptr_t outputAddress,
      size_t outputBytes, CUstream stream, const std::array<uintptr_t, 8>& peerMappedInputAddresses,
      const std::array<uintptr_t, 8>& peerMappedOutputAddresses, bool direct, size_t numChunks, size_t chunkSize) {

    if (!allGather2dCopies) {
      calculateAllGather2dCopies();
    }

    const auto& peerIndices = group->peerIndices;

    IpcMapper* ipcMapper = &*group->ipcMapper;

    memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
    memFlush(stream);

    // if (!group->kernels->cuDummySignal) {
    //   group->kernels->compile(0);
    // }
    // CHECK(group->kernels->cuDummySignal != nullptr);
    // CHECK_CU(cuLaunchKernel(group->kernels->cuDummySignal, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));

    {
      size_t offset = inputBytes * rank;
      size_t nbytes = inputBytes;
      CHECK(inputBytes == nbytes);

      if (outputAddress + offset != inputAddress) {
        CHECK_CU(cuMemcpyDtoDAsync(outputAddress + offset, inputAddress, nbytes, stream));
      }
    }

    if (!direct) {
      memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) + (direct ? 4 : 0), stepValue);
      memFlush(stream);
    }

    // CHECK_CU(cuLaunchKernel(group->kernels->cuDummySignal, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));

    std::array<uintptr_t, 8> peerAddresses;

    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(
          concurrencyIndex, peerIndex, opTypeAllGatherKernelLessCuda, stepValue, peerMappedOutputAddresses[peerIndex],
          outputBytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeAllGatherKernelLessCuda, stepValue, outputBytes);
    }

    freePendingIpcEvents();

    if (direct) {

      // CUstream altStream = getCopyStream(stream);

      // localEvent.record(stream);
      // localEvent.wait(altStream);

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

        // CUstream curStream = chunkIndex % 2 ? altStream : stream;
        //  CUstream curStream = stream;

        memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + chunkIndex), stepValue);
        memFlush(curStream);

        syncPeers(curStream);
        ++nQueuedChunks;
        if (nQueuedChunks >= 1) {
          flushCopies();
        }
      }
      flushCopies();
      // localEvent.record(altStream);
      // localEvent.wait(stream);
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

      if (true) {
        // memSyncPeers(concurrencyIndex, stream);
        // CHECK_CU(cuLaunchKernel(group->kernels->cuDummySignal, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
        syncPeers(stream);
        for (size_t peerIndex : peerIndices) {
          copy(peerIndex);
        }
        // memSyncPeers(concurrencyIndex, stream);
        // CHECK_CU(cuLaunchKernel(group->kernels->cuDummySignal, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));
        syncPeers(stream);
      } else {
        for (size_t peerIndex : peerIndices) {
          size_t incomingPeerIndex = peerIndices.size() - 1 - peerIndex;

          ipcMapper->streamRecord(incomingPeerIndex, stream);
          ipcMapper->streamWait(peerIndex, stream);

          copy(peerIndex);

          ipcMapper->streamRecord(peerIndex, stream);
          ipcMapper->streamWait(incomingPeerIndex, stream);
        }
      }
    }

    if (direct) {
      memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex) + 4, stepValue + 1);
      memFlush(stream);
    }
  }

  void kernelLess_all_gather_ring_copies(
      uint32_t concurrencyIndex, uint32_t stepValue, uintptr_t inputAddress, size_t inputBytes, uintptr_t outputAddress,
      size_t outputBytes, CUstream stream, const std::array<uintptr_t, 8>& peerMappedInputAddresses,
      const std::array<uintptr_t, 8>& peerMappedOutputAddresses, bool direct, size_t numChunks, size_t chunkSize) {

    if (!allGather2dCopies) {
      calculateAllGather2dCopies();
    }

    const auto& peerIndices = group->peerIndices;

    IpcMapper* ipcMapper = &*group->ipcMapper;

    memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
    memFlush(stream);

    {
      size_t offset = inputBytes * rank;
      size_t nbytes = inputBytes;
      CHECK(inputBytes == nbytes);

      if (outputAddress + offset != inputAddress) {
        CHECK_CU(cuMemcpyDtoDAsync(outputAddress + offset, inputAddress, nbytes, stream));
      }
    }

    memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
    memFlush(stream);

    // CHECK_CU(cuLaunchKernel(group->kernels->cuDummySignal, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr));

    std::array<uintptr_t, 8> peerAddresses;

    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(
          concurrencyIndex, peerIndex, opTypeAllGatherKernelLessCuda, stepValue, peerMappedOutputAddresses[peerIndex],
          outputBytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeAllGatherKernelLessCuda, stepValue, outputBytes);
    }

    freePendingIpcEvents();

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

    if (true) {
      syncPeers(stream);
      for (size_t peerIndex : peerIndices) {
        copy(peerIndex);
      }
      syncPeers(stream);
    }
  }

  void all_gather(at::Tensor& output, const at::Tensor& input, CUstream stream) {
    std::unique_lock l(threadUnsafe);
    trace("all_gather");

    size_t size = this->size;

    CHECK(input.is_contiguous());
    CHECK(output.is_contiguous());

    bool isCuda = input.is_cuda();

    if (isCuda) {
      CHECK(input.is_cuda());
      CHECK(input.device().index() == group->deviceIndex);
      CHECK(output.is_cuda());
      CHECK(output.device().index() == group->deviceIndex);
    }

    size_t bytes = input.numel() * input.itemsize();
    size_t outputBytes = bytes * size;
    size_t pitch = bytes;

    // log.info("group size %d doing all-gather of size %d bytes\n", size, bytes);

    CHECK(bytes > 0);
    CHECK(output.numel() * output.itemsize() == outputBytes);

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
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

    Group* group = &*this->group;

    AllGather& allGather = *group->allGather;

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    bool isLocalOnly = allGather.recvRanks.empty();
    bool isNoLocal = !isLocalOnly && peerIndices.empty();

    // if (isLocalOnly) {
    //   size_t localThreshold = 50 * 1024 * 1024;
    //   if (peerIndices.size() <= 4) {
    //     localThreshold /= 2;
    //   }
    //   if (peerIndices.size() <= 2) {
    //     localThreshold /= 2;
    //   }

    //   if (bytes >= localThreshold || bytes % 16 != 0 || outputAddress % 16 != 0 || inputAddress % 16 != 0) {
    //     local_all_gather_impl(
    //         concurrencyIndex, stepValue, inputAddress, bytes, outputAddress, outputBytes, stream, ipcRanks, rank);

    //     return;
    //   }
    // }

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

    trace("ipcMapper");

    std::shared_lock unmapLock(unmapMemoryMutex);
    sync(stepValue);

    std::array<uintptr_t, 8> peerInputAddresses;
    std::array<uintptr_t, 8> peerOutputAddresses;

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, inputAddress, bytes, [ptr = &peerInputAddresses[i]](uintptr_t address) { *ptr = address; }, true);

      ipcMapper->requestAddress(
          i, outputAddress, outputBytes, [ptr = &peerOutputAddresses[i]](uintptr_t address) { *ptr = address; }, true);
    }

    ipcMapper->wait();

    trace("enqueue");

    size_t numChunks = 0;
    size_t chunkSize = 0;

    bool kernelLess_ring = kernelLess && false;

    bool direct = !isLocalOnly && kernelLess && !kernelLess_ring;

    if (!isLocalOnly) {
      QueueEntryAllGather* e = group->cpuThread->freelistAllGather.pop();
      // e->task = kernelLess ? taskAllGatherBroadcast : taskAllGather;
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
      // if (isNoLocal && bytes > 65536) {
      //   e->numDevices = std::max((size_t)4, group->numTrueIbDevs);
      //   e->numChunks = 4;
      // }
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
      kernelLess_all_gather_ring_copies(
          concurrencyIndex, stepValue, inputAddress, bytes, outputAddress, outputBytes, stream, peerInputAddresses,
          peerOutputAddresses, direct, numChunks, chunkSize);
      return;
    }

    if (kernelLess && (!isNoLocal || direct)) {
      // log.info("direct, %d chunks of %d bytes\n", numChunks, chunkSize);
      CHECK(pitch == bytes);
      // kernelLess_all_gather_impl(
      //     concurrencyIndex, stepValue, inputAddress, bytes, outputAddress, outputBytes, stream, peerInputAddresses,
      //     peerOutputAddresses, chunkSize);
      kernelLess_all_gather_impl(
          concurrencyIndex, stepValue, inputAddress, bytes, outputAddress, outputBytes, stream, peerInputAddresses,
          peerOutputAddresses, direct, numChunks, chunkSize);
      return;
    }
    CHECK(!direct);

    trace("launch");

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
        CHECK_CU(
            cuLaunchKernel(group->kernels->cuAllGatherNoLocal, 1, 1, 1, 1, 1, 1, 0, stream, params.data(), nullptr));
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

    trace("post");
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

  uint32_t nextMemSyncPeersValue = 1;

  void memSyncPeers(size_t concurrencyIndex, CUstream stream) {
    const auto& peerIndices = group->peerIndices;
    uint32_t value = nextMemSyncPeersValue;
    ++nextMemSyncPeersValue;
    for (size_t peerIndex : peerIndices) {
      memWrite(
          group->peerCudaMemSync[peerIndex].get(concurrencyIndex) +
              sizeof(uint32_t) * group->peerMyRemoteIndex[peerIndex],
          value);
    }
    for (size_t peerIndex : peerIndices) {
      memWaitGeq(group->cudaMemSync.cuda(concurrencyIndex) + sizeof(uint32_t) * peerIndex, value);
    }
    memFlush(stream);
  }

  void reduce_scatter_copy_prepare_sends(
      uint32_t concurrencyIndex, uint32_t stepValue, const std::array<uintptr_t, 8>& peerInputAddresses,
      uintptr_t inputAddress, uintptr_t bufferAddress, uintptr_t sendAddress, size_t bytes, const auto& dtype,
      const auto& device, size_t numel, Reduction opindex, CUstream stream, size_t copyBatchSize) {
    ReduceScatter& reduceScatter = *group->reduceScatter;
    IpcMapper* ipcMapper = &*group->ipcMapper;
    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;
    const auto& sendRanks = reduceScatter.sendRanks;
    const size_t numSends = sendRanks.size();

    std::array<uintptr_t, 8> peerAddresses;

    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(
          concurrencyIndex, peerIndex, opTypeReduceScatterKernelLessCuda, stepValue, peerInputAddresses[peerIndex],
          bytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeReduceScatterKernelLessCuda, stepValue, bytes);
    }
    freePendingIpcEvents();

    syncPeers(stream);

    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

    torch::Tensor bufferTensor = torch::from_blob(
        (void*)bufferAddress, {(int64_t)((1 + peerIndices.size()) * copyBatchSize), (int64_t)numel},
        at::TensorOptions().dtype(dtype).device(device));

    if (sendRanks.size()) {
      CHECK(copyBatchSize >= 1);
      torch::Tensor outTensor = torch::from_blob(
          (void*)sendAddress, {(int64_t)sendRanks.size(), (int64_t)numel},
          at::TensorOptions().dtype(dtype).device(device));

      auto buft = bufferTensor.view({(int64_t)copyBatchSize, (int64_t)(1 + peerIndices.size()), (int64_t)numel});

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

        // peerSync() ?

        for (size_t peerIndex : peerIndices) {
          copyArgs.srcDevice = peerAddresses[peerIndex] + srcOffsetBytes;
          copyArgs.dstDevice = bufferAddress + dstOffsetBytes;
          CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));

          dstOffsetBytes += bytes;
        }

        copyArgs.srcDevice = inputAddress + srcOffsetBytes;
        copyArgs.dstDevice = bufferAddress + dstOffsetBytes;
        CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));

        auto src = buft.narrow(0, 0, n);
        auto dst = outTensor.narrow(0, begin, n);
        if (opindex == Reduction::sum) {
          torch::sum_out(dst, src, 1);
        } else if (opindex == Reduction::max) {
          torch::amax_out(dst, src, 1);
        } else if (opindex == Reduction::min) {
          torch::amin_out(dst, src, 1);
        } else {
          CHECK(false);
        }

        for (size_t z : range(n)) {
          memWrite(group->cpuInBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + begin + z), stepValue);
        }
        memFlush(stream);
      };
      for (size_t i : range((size_t)1, numSends)) {
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

  void reduce_scatter(at::Tensor& output, const at::Tensor& input, c10d::ReduceOp reduceOp, CUstream stream) {
    std::unique_lock l(threadUnsafe);
    trace("_reduce_scatter_base");

    size_t size = this->size;

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    CHECK(input.is_contiguous());
    CHECK(output.is_contiguous());

    bool isCuda = input.is_cuda();

    if (isCuda) {
      CHECK(input.is_cuda());
      CHECK(input.device().index() == group->deviceIndex);
      CHECK(output.is_cuda());
      CHECK(output.device().index() == group->deviceIndex);
    }

    auto dtype = output.dtype();
    CHECK(input.dtype() == dtype);

    size_t numel = output.numel();
    size_t itemsize = output.itemsize();
    size_t bytes = numel * itemsize;
    size_t inputBytes = bytes * size;
    size_t pitch = bytes;

    // log.info("group size %d doing reduce-scatter of size %d bytes\n", size, bytes);

    CHECK(bytes > 0);
    CHECK(input.numel() * input.itemsize() == inputBytes);

    uintptr_t inputAddress = (uintptr_t)input.data_ptr();
    uintptr_t outputAddress = (uintptr_t)output.data_ptr();

    bool workaroundMean = false;
    bool premul = false;
    float premulValue;

    Dtype dindex;
    switch (dtype.toScalarType()) {
    case torch::Dtype::Float:
      dindex = Dtype::float32;
      break;
    case torch::Dtype::Double:
      dindex = Dtype::float64;
      break;
    case torch::Dtype::Int:
      dindex = Dtype::int32;
      break;
    case torch::Dtype::Long:
      dindex = Dtype::int64;
      break;
    case torch::Dtype::BFloat16:
      dindex = Dtype::bfloat16;
      break;
    default:
      throw std::runtime_error(
          fmt::sprintf("moodist: reduce_scatter: Unsupported dtype %s", std::string(dtype.name())));
    }
    Reduction opindex;
    switch (reduceOp) {
    case c10d::ReduceOp::SUM:
      opindex = Reduction::sum;
      break;
    case c10d::ReduceOp::MIN:
      opindex = Reduction::min;
      break;
    case c10d::ReduceOp::MAX:
      opindex = Reduction::max;
      break;
    case c10d::ReduceOp::AVG:
      opindex = Reduction::sum;
      workaroundMean = true;
      break;
    case c10d::ReduceOp::PREMUL_SUM:
      opindex = Reduction::sum;
      premul = true;
      if (dindex == Dtype::float32) {
        auto* x = reinterpret_cast<c10d::NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
        if (x->tensor_factor.defined()) {
          premulValue = x->tensor_factor.item<float>();
        } else {
          premulValue = x->double_factor;
        }
      } else {
        throw std::runtime_error(
            fmt::sprintf("moodist: reduce_scatter: Pre-mul with dtype %s is not supported", std::string(dtype.name())));
      }
      break;
    default:
      throw std::runtime_error(fmt::sprintf("moodist: reduce_scatter: Unsupported reduceOp %d", reduceOp));
    }

    Group* group = &*this->group;

    ReduceScatter& reduceScatter = *group->reduceScatter;

    auto& sendRanks = reduceScatter.sendRanks;
    auto& recvRanks = reduceScatter.recvRanks;

    torch::Tensor premulTensor;

    if (!isCuda) {
      if (premul) {
        premulTensor = input * premulValue;
        inputAddress = (uintptr_t)premulTensor.data_ptr();
        CHECK(premulTensor.numel() * premulTensor.itemsize() == inputBytes);
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
        output *= 1.0f / size;
      }
      return;
    }

    StreamData& sd = group->getStreamData(stream);
    EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

    if (premul) {
      premulTensor = input * premulValue;
      inputAddress = (uintptr_t)premulTensor.data_ptr();
      CHECK(premulTensor.numel() * premulTensor.itemsize() == inputBytes);
    }

    IpcMapper* ipcMapper = &*group->ipcMapper;

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    AllocatedBuffer alignedBuffer;
    AllocatedBuffer alignedBuffer2;

    bool isLocalOnly = reduceScatter.recvRanks.empty();
    bool isNoLocal = peerIndices.empty();

    enum { methodKernel, methodCopy, methodDirect };
    int method = methodCopy;
    if (isLocalOnly) {
      method = methodKernel;
    } else {
      method = methodDirect;
      // if (bytes >= 262144) {
      //   method = methodDirect;
      // } else {
      //   method = methodKernel;
      // }
    }

    // method = methodDirect;

    bool direct = method == methodDirect;
    bool copy = method == methodCopy;

    size_t copyBatchSize = 0;
    if (copy) {
      copyBatchSize = (reduceScatter.sendRanks.size() + 7) / 8;
      // copyBatchSize = reduceScatter.sendRanks.size();
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

    trace("ipcMapper");

    AllocatedBuffer sendBuffer;
    if (!copy || !isNoLocal) {
      sendBuffer = alloccuda(pitch * sendRanks.size(), stream);
    }
    AllocatedBuffer recvBuffer;
    if (copy) {
      recvBuffer = alloccuda(pitch * (recvRanks.size() + (1 + peerIndices.size()) * copyBatchSize), stream);
    } else if (direct) {
      recvBuffer = alloccuda(pitch * (1 + recvRanks.size() + peerIndices.size()), stream);
    } else {
      recvBuffer = alloccuda(pitch * recvRanks.size(), stream);
    }

    std::shared_lock unmapLock(unmapMemoryMutex);
    sync(stepValue);

    std::array<uintptr_t, 8> peerInputAddresses;
    std::array<uintptr_t, 8> peerOutputAddresses;

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, inputAddress, inputBytes, [ptr = &peerInputAddresses[i]](uintptr_t address) { *ptr = address; }, true);

      ipcMapper->requestAddress(
          i, outputAddress, bytes, [ptr = &peerOutputAddresses[i]](uintptr_t address) { *ptr = address; }, true);
    }

    ipcMapper->wait();

    trace("enqueue");

    size_t chunkSize = 0;

    uintptr_t sendAddress = sendBuffer.cudaPointer;
    uintptr_t recvAddress = recvBuffer.cudaPointer;
    if (!isLocalOnly) {
      if (copy && isNoLocal) {
        CHECK(sendAddress == 0);
      } else {
        CHECK(sendAddress != 0);
      }
      CHECK(recvAddress != 0);
    }

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

      // e->numDevices = std::min(bytes / 262144, std::max(group->ibDevs.size(), group->numTrueIbDevs));
      e->numDevices = bytes < 262144 ? 1 : std::max((size_t)4, group->numTrueIbDevs);
      e->numChunks = std::min(bytes / 131072, (size_t)4);
      // if (peerIndices.empty()) {
      //   e->numDevices = std::min(bytes / 65536, std::max(group->ibDevs.size(), group->numTrueIbDevs));
      //   e->numChunks = std::min(bytes / 65536, (size_t)4);
      // }
      // if (copy) {
      //   e->numDevices = group->ibDevs.size();
      // }
      e->numDevices = std::min(e->numDevices, group->rdmaDevs.size());
      e->numDevices = std::max(e->numDevices, (size_t)1);
      e->numChunks = std::max(e->numChunks, (size_t)1);
      e->numParallel = 1;
      chunkSize = ((bytes + e->numChunks - 1) / e->numChunks + 4095u) / 4096u * 4096u;
      e->numChunks = (bytes + chunkSize - 1) / chunkSize;
      group->cpuThread->enqueue(e);

      // log.info("devices %d, chunks %d\n", e->numDevices, e->numChunks);
    }

    if (copy || direct) {

      if (direct) {
        CHECK(!isLocalOnly);

        if (!group->kernels->cuReduceScatterDirect[(size_t)dindex][(size_t)opindex]) {
          group->kernels->compile(
              CompileReduceScatterDirect, group->kernels->supportedTypes[(size_t)dindex],
              group->kernels->supportedReductions[(size_t)opindex]);
          CHECK(group->kernels->cuReduceScatterDirect[(size_t)dindex][(size_t)opindex] != nullptr);
        }

        std::array<uintptr_t, 8> peerAddresses;

        for (size_t peerIndex : peerIndices) {
          peerWriteDyn(
              concurrencyIndex, peerIndex, opTypeReduceScatterDirectCuda, stepValue, peerInputAddresses[peerIndex],
              bytes);
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

        CHECK_CU(cuLaunchKernel(
            group->kernels->cuReduceScatterDirect[(size_t)dindex][(size_t)opindex], gridSize * 4, 1, 1, blockSize, 1, 1,
            0, stream, params.data(), nullptr));

        for (size_t peerIndex : peerIndices) {
          cudaCopy(
              recvAddress + pitch * recvRanks.size() + pitch * (1 + peerIndex), peerAddresses[peerIndex] + pitch * rank,
              bytes, stream);
        }

      } else {
        CHECK(pitch == bytes);

        if (!isLocalOnly) {
          memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
          memFlush(stream);
        }

        if (!isNoLocal) {
          CHECK(sendAddress != 0);
          reduce_scatter_copy_prepare_sends(
              concurrencyIndex, stepValue, peerInputAddresses, inputAddress, recvAddress + bytes * recvRanks.size(),
              sendAddress, bytes, output.dtype(), output.device(), numel, opindex, stream, copyBatchSize);
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
        c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
        auto tensor = output.flatten();
        CHECK((uintptr_t)tensor.const_data_ptr() == outputAddress);
        torch::Tensor buffer = torch::from_blob(
            (void*)recvAddress, {(int64_t)(1 + recvRanks.size() + peerIndices.size()), (int64_t)(pitch / itemsize)},
            at::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
        if (buffer.size(1) != numel) {
          buffer = buffer.narrow(1, 0, numel);
        }
        if (opindex == Reduction::sum) {
          torch::sum_out(tensor, buffer, 0);
        } else if (opindex == Reduction::max) {
          torch::amax_out(tensor, buffer, 0);
        } else if (opindex == Reduction::min) {
          torch::amin_out(tensor, buffer, 0);
        } else {
          CHECK(false);
        }
      }

      if (!isLocalOnly) {
        memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue + 1);
        memFlush(stream);
      }

    } else {

      trace("launch");

      if (!group->kernels->cuReduceScatter[(size_t)dindex][(size_t)opindex]) {
        group->kernels->compile(
            CompileReduceScatter, group->kernels->supportedTypes[(size_t)dindex],
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
        CHECK_CU(cuLaunchKernel(
            group->kernels->cuReduceScatterLocal[(size_t)dindex][(size_t)opindex], gridSize, 1, 1, blockSize, 1, 1, 0,
            stream, params.data(), nullptr));
      } else {
        CHECK_CU(cuLaunchKernel(
            group->kernels->cuReduceScatter[(size_t)dindex][(size_t)opindex], gridSize, 1, 1, blockSize, 1, 1, 0,
            stream, params.data(), nullptr));
      }
    }

    if (outputAddress != (uintptr_t)output.data_ptr()) {
      CHECK_CU(cuMemcpyDtoDAsync((uintptr_t)output.data_ptr(), outputAddress, bytes, stream));
    }

    if (workaroundMean) {
      c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
      output *= 1.0f / size;
    }

    CHECK(output.numel() == numel);
    CHECK(output.itemsize() == itemsize);

    trace("post");
  }

  void hostFunc(CUstream stream, Function<void()> f) {
    CHECK_CU(cuLaunchHostFunc(stream, [](void* c) { Function<void()>(FunctionPointer(c))(); }, f.release()));
  }

  Vector<CUstreamBatchMemOpParams> memops;
  void memWaitGeq(uintptr_t address, uint32_t value) {
    CUstreamBatchMemOpParams op;
    std::memset(&op, 0, sizeof(op));
    op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    op.waitValue.address = address;
    op.waitValue.value = value;
    memops.push_back(op);
  }
  void memWaitEq(uintptr_t address, uint32_t value) {
    CUstreamBatchMemOpParams op;
    std::memset(&op, 0, sizeof(op));
    op.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    op.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
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
    if (memops.size() == 0) {
      return;
    }
    CHECK_CU(cuStreamBatchMemOp(stream, memops.size(), memops.data(), 0));
    memops.clear();
  }

  void peerWriteDyn(
      uint32_t concurrencyIndex, size_t peerIndex, uint32_t opType, uint32_t stepValue, uintptr_t gatherAddress,
      size_t bytes) {
    auto& dyn = group->getPeerVar(peerIndex, group->cpuLocalDyns)[size * concurrencyIndex + rank];
    dyn.opType = opType;
    dyn.gatherAddress = gatherAddress;
    dyn.gatherBytes = bytes;
    dyn.stepValue = stepValue;

    group->getPeerVar(peerIndex, group->atomicStepValue)[size * concurrencyIndex + rank] = stepValue;

    group->ipcMapper->push(peerIndex, std::make_pair(concurrencyIndex, stepValue));
  }
  uintptr_t
  peerWaitDyn(uint32_t concurrencyIndex, size_t peerIndex, uint32_t opType, uint32_t stepValue, size_t bytes) {
    size_t i = group->ipcRanks[peerIndex];
    while (group->atomicStepValue[size * concurrencyIndex + i].load() < stepValue) {
      _mm_pause();
    }
    auto& dyn = group->cpuLocalDyns[size * concurrencyIndex + i];
    CHECK_DYN(dyn, opType, stepValue, bytes);

    CHECK(
        (group->ipcMapper->pop<std::pair<uint32_t, uint32_t>>(peerIndex) ==
         std::make_pair(concurrencyIndex, stepValue)));

    return dyn.gatherAddress;
  }

  void broadcast(const torch::Tensor& tensor, uint32_t sourceRank, CUstream stream) {
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    bool isCuda = tensor.is_cuda();

    CHECK(tensor.is_contiguous());
    CHECK(sourceRank < size);

    if (isCuda) {
      CHECK(tensor.device().index() == group->deviceIndex);
    }

    size_t numel = tensor.numel();
    size_t bytes = numel * tensor.itemsize();

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
      e->sourceRank = sourceRank;
      e->cpuDone = &cpuDone;
      group->cpuThread->enqueue(e);
      l.unlock();
      while (cpuDone == 0) {
        futexWait(&cpuDone, 0, std::chrono::seconds(10));
      }
      return;
    }

    // CHECK(tensorAddress % 16 == 0);

    StreamData& sd = group->getStreamData(stream);
    EventSerializer es(concurrencyEvents[concurrencyIndex], stream);

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
    if (!localEvent) {
      localEvent = Event::create();
    }

    size_t sourceLocalRank = group->rankLocalRank.at(sourceRank);
    size_t localSourceRank = sourceRank;
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
      e->sourceRank = sourceRank;

      e->numDevices = numDevices;
      e->numChunks = numChunks;
      e->numParallel = 1;

      group->cpuThread->enqueue(e);
    }

    bool anyLocalPeers = peerIndices.size();

    if (!anyLocalPeers) {
      CHECK(rank == localSourceRank);
    }

    size_t next = 0;
    size_t prev = peerIndices.size() - 1;

    uintptr_t peerAddress = 0;
    if (anyLocalPeers && ipcRanks[next] != localSourceRank) {
      ipcMapper->requestAddress(next, tensorAddress, bytes, &peerAddress, true);

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
        if (sourceRank != rank) {
          CHECK(networked);
          memWaitGeq(
              group->cpuOutBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + size * chunkIndex + sourceRank),
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

    // if (ipcRanks[next] != localSourceRank) {
    //   ipcMapper->streamWait(next, stream);
    // }

    // if (localSourceRank != rank) {
    //   ipcMapper->streamWait(prev, stream);
    // }

    // if (ipcRanks[next] != localSourceRank) {
    //   ipcMapper->streamWait(next, stream);
    // }

    CHECK(memops.empty());

    trace("post");
  }

  void reduce(torch::Tensor& tensor, uint32_t destinationRank, c10d::ReduceOp reduceOp, CUstream stream) {
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    bool isCuda = tensor.is_cuda();

    CHECK(tensor.is_contiguous());
    CHECK(destinationRank < size);

    if (isCuda) {
      CHECK(tensor.device().index() == group->deviceIndex);
    }

    size_t numel = tensor.numel();
    size_t bytes = numel * tensor.itemsize();

    uintptr_t tensorAddress = (uintptr_t)tensor.data_ptr();

    if (!isCuda) {
      throw std::runtime_error("CPU reduce currently not supported :(");
    }

    // CHECK(tensorAddress % 16 == 0);

    StreamData& sd = group->getStreamData(stream);
    EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

    Group* group = &*this->group;

    std::shared_lock unmapLock(unmapMemoryMutex);
    sync(stepValue);

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    IpcMapper* ipcMapper = &*group->ipcMapper;

    CHECK(reduceOp == c10d::ReduceOp::SUM);

    CHECK(memops.empty());

    // auto allocbuffer = [&](auto& buffer, size_t size) {
    //   if (size < 4096) {
    //     size = 4096;
    //   }
    //   if (buffer.bytes < size) {
    //     CHECK_CU(cuCtxSynchronize());
    //     buffer = {};
    //     buffer = group->allocateDevice(size);
    //   }
    // };

    if (!peerMemcpyStream) {
      peerMemcpyStream = Stream::create(streamPriority);
    }
    if (!peerIncomingMemcpyStream) {
      peerIncomingMemcpyStream = Stream::create(streamPriority);
    }
    if (!localEvent) {
      localEvent = Event::create();
    }

    size_t destinationLocalRank = group->rankLocalRank.at(destinationRank);
    size_t localDestinationRank = destinationRank;
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

    // AllocatedBuffer& reduceBuffer = reduceBufferArr[reduceCounter];
    AllocatedBuffer reduceBuffer;
    auto& reduceStream = reduceStreamArr[reduceCounter];
    if (localDestinationRank == rank) {
      reduceCounter = (reduceCounter + 1) % reduceStreamArr.size();
    }

    if (!reduceStream) {
      reduceStream = Stream::create(streamPriority);
    }

    if (networked) {
      tree = group->getTree(4, destinationRank);

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
      e->destinationRank = destinationRank;

      e->recvBuffer = tree->recvs.empty() ? 0 : reduceBuffer.cudaPointer + bytes * recvOffset;

      e->numDevices = numDevices;
      e->numChunks = numChunks;
      e->numParallel = 1;

      group->cpuThread->enqueue(e);
    }

    bool anyLocalPeers = peerIndices.size();

    if (!anyLocalPeers) {
      CHECK(rank == localDestinationRank);
    }

    if (!peerMemcpyStream) {
      peerMemcpyStream = Stream::create(streamPriority);
    }
    if (!localEvent) {
      localEvent = Event::create();
    }

    if (localDestinationRank != rank) {
      CHECK(anyLocalPeers);
      CHECK(!networked);
      size_t peerIndex = group->getPeerIndex(localDestinationRank);

      uintptr_t peerAddress;
      ipcMapper->requestAddress(peerIndex, tensorAddress, bytes, &peerAddress, true);
      ipcMapper->wait();

      size_t i = ipcRanks[peerIndex];

      peerWriteDyn(concurrencyIndex, peerIndex, opTypeReduceCuda, stepValue, peerAddress, bytes);

      CHECK(peerWaitDyn(concurrencyIndex, peerIndex, opTypeReduceCuda, stepValue, bytes) == 0);

      if (false) {
        ipcMapper->streamRecord(peerIndex, stream);
        ipcMapper->streamWait(peerIndex, stream);
      } else {
        localEvent.record(*peerIncomingMemcpyStream);
        localEvent.wait(stream);
        ipcMapper->streamRecord(peerIndex, stream);

        ipcMapper->streamWait(peerIndex, stream);
        localEvent.record(stream);
        localEvent.wait(*peerIncomingMemcpyStream);
      }

    } else {

      std::array<uintptr_t, 8> peerAddresses;
      peerAddresses.fill(0);

      for (size_t peerIndex : peerIndices) {
        size_t i = ipcRanks[peerIndex];

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

      if (false) {

        localEvent.record(stream);
        localEvent.wait(*reduceStream);

        for (size_t peerIndex : peerIndices) {
          ipcMapper->streamWait(peerIndex, *reduceStream);
          CHECK(reduceBuffer.cudaPointer != 0);
          CHECK(peerAddresses[peerIndex] != 0);
          CHECK_CU(cuMemcpyDtoDAsync(
              reduceBuffer.cudaPointer + bytes * peerIndex, peerAddresses[peerIndex], bytes, *reduceStream));

          ipcMapper->streamRecord(peerIndex, *reduceStream);
        }

        CHECK_CU(cuMemcpyDtoDAsync(
            reduceBuffer.cudaPointer + bytes * peerIndices.size(), tensorAddress, bytes, *reduceStream));

      } else {

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

        CHECK_CU(cuMemcpyDtoDAsync(
            reduceBuffer.cudaPointer + bytes * peerIndices.size(), tensorAddress, bytes, *reduceStream));

        localEvent.record(*peerMemcpyStream);
        localEvent.wait(*reduceStream);
      }

      torch::Tensor buffer = torch::from_blob(
          (void*)reduceBuffer.cudaPointer, {(int64_t)bufferSize, (int64_t)numel},
          at::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));

      CHECK(buffer.numel() * buffer.itemsize() == bytes * bufferSize);

      CHECK(reduceOp == c10d::ReduceOp::SUM);

      if (networked) {
        size_t offset = 0;
        for (size_t chunkIndex = 0; offset != bytes; ++chunkIndex) {
          size_t currentChunkSize = std::min(chunkSize, bytes - offset);

          for (size_t index = 0; index != tree->recvs.size(); ++index) {
            memWaitGeq(
                group->cpuOutBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + size * chunkIndex + index),
                stepValue);
          }

          offset += currentChunkSize;
        }
        memFlush(*reduceStream);
      }

      {
        c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(*reduceStream, c10::cuda::current_device()));
        torch::sum_out(tensor, buffer, 0);
      }

      localEvent.record(*reduceStream);
      localEvent.wait(stream);

      if (networked) {
        size_t offset = 0;
        for (size_t chunkIndex = 0; offset != bytes; ++chunkIndex) {
          size_t currentChunkSize = std::min(chunkSize, bytes - offset);

          CHECK(tree->sends.size() <= 1);
          for (size_t index = 0; index != tree->sends.size(); ++index) {
            memWrite(
                group->cpuInBuffer.cuda(concurrencyIndex) + sizeof(uint32_t) * (16 + size * chunkIndex + index),
                stepValue);
          }

          offset += currentChunkSize;
        }
        memWrite(group->cpuInBuffer.cuda(concurrencyIndex), stepValue + 1);
        memWaitGeq(group->cpuOutBuffer.cuda(concurrencyIndex), stepValue);
        memFlush(stream);
      }
    }

    // see sync note at end of broadcast
    // ? is a sync necessary here, anyway?
    //    it might be with the current impl of peerWriteDyn, but possibly wouldn't if it used the queue?
    for (size_t peerIndex : peerIndices) {
      ipcMapper->push(peerIndex, stepValue + 1);
    }
    for (size_t peerIndex : peerIndices) {
      CHECK(ipcMapper->pop<uint32_t>(peerIndex) == stepValue + 1);
    }

    CHECK(memops.empty());

    trace("post");
  }

  void gather(const std::vector<at::Tensor>& outputList, const at::Tensor& input, uint32_t destinationRank) {
    trace("gather");
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    CHECK(destinationRank < size);
    if (destinationRank == rank) {
      CHECK(outputList.size() == size);
    } else {
      CHECK(outputList.empty());
    }

    CHECK(input.is_contiguous());
    for (auto& output : outputList) {
      CHECK(output.is_contiguous());
    }

    bool isCuda = input.is_cuda();

    if (isCuda) {
      CHECK(input.is_cuda());
      CHECK(input.device().index() == group->deviceIndex);
      for (auto& output : outputList) {
        CHECK(output.is_cuda());
        CHECK(output.device().index() == group->deviceIndex);
      }
    }

    size_t bytes = input.numel() * input.itemsize();
    size_t outputBytes = bytes * size;

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

      e->outputList.resize(outputList.size());
      for (size_t i = 0; i != outputList.size(); ++i) {
        auto& output = outputList[i];
        e->outputList[i] = {(uintptr_t)output.data_ptr(), (size_t)(output.itemsize() * output.numel())};
      }

      e->bytes = bytes;
      e->destinationRank = destinationRank;
      e->cpuDone = &cpuDone;
      group->cpuThread->enqueue(e);
      l.unlock();
      while (cpuDone == 0) {
        futexWait(&cpuDone, 0, std::chrono::seconds(10));
      }
      return;
    }

    throw std::runtime_error("cuda gather is not yet supported :(");
  }

  TensorDataPtr getTensorData(const torch::Tensor& tensor, bool allowCopy = true) {
    TensorDataPtr td = getTensorReference(tensor);

    if (!td->isCuda) {
      if (cpu_allocator::owns(td->dataPtr)) {
        auto handle = cpu_allocator::getCpuBuffer((uintptr_t)tensor.storage().data());
        CHECK(handle != nullptr);
        td->buffer = std::move(handle);
      } else {
        if (!allowCopy) {
          throw std::runtime_error("Found CPU tensor that we do not own");
        }
        uintptr_t src = td->data();
        td->buffer = AllocatedCpuBufferSharedPtr::make();
        *td->buffer = group->allocateCpu(td->dataBytes);
        td->dataPtr = (uintptr_t)td->buffer->cpuPointer;
        std::memcpy((void*)td->data(), (void*)src, td->bytes());

        CHECK(cpu_allocator::owns(td->data()));
      }
    }

    return td;
  }

  TensorDataPtr getTensorReference(const torch::Tensor& tensor) {
    TensorDataPtr td = TensorDataPtr::make();

    td->dtype = (int)tensor.dtype().toScalarType();
    const auto& shape = tensor.sizes();
    td->shape.resize(shape.size());
    for (size_t i = 0; i != td->shape.size(); ++i) {
      td->shape[i] = shape[i];
    }

    uintptr_t tensorAddress = (uintptr_t)(const void*)tensor.data_ptr();

    td->dataPtr = tensorAddress;
    td->dataBytes = td->itemsize() * td->numel();

    td->isCuda = !tensor.is_cpu();

    if (tensorAddress == 0 && td->dataBytes != 0) {
      throw std::runtime_error(fmt::sprintf("getTensorReference: got a null tensor of %d bytes", td->dataBytes));
    }

    return td;
  }

  Future cat(const std::vector<std::pair<int, torch::Tensor>>& locals, std::optional<torch::Tensor> out) {
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

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
    e->locals.resize(locals.size());
    for (size_t i : indices(locals)) {
      e->locals[i] = {locals[i].first, getTensorData(locals[i].second)};
    }
    e->future = future;
    e->out = nullptr;
    if (out) {
      e->out = getTensorData(*out, false);
    }
    group->cpuThread->enqueue(e);

    Future r;
    for (auto& v : locals) {
      r.tensors.push_back(v.second);
    }
    if (out) {
      r.out = *out;
    }
    r.impl = std::move(future);
    return r;
  }

  Future copy(torch::Tensor& destination, const torch::Tensor& source) {
    std::unique_lock l(threadUnsafe);

    if (!destination.is_contiguous() || !source.is_contiguous()) {
      throw std::runtime_error("copy: both source and destination must be contigiuous");
    }

    auto destinationData = getTensorData(destination, false);
    auto sourceData = getTensorData(source, false);

    if (destinationData->bytes() != sourceData->bytes()) {
      throw std::runtime_error(fmt::sprintf(
          "copy: destination is %d bytes while source is %d bytes\n", destinationData->bytes(), sourceData->bytes()));
    }

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    auto future = FutureImplSharedPtr::make();

    StreamData& sd = group->getCpuStreamData(concurrencyIndex);
    QueueEntryCopy* e = group->cpuThread->freelistCopy.pop();
    e->task = taskCopy;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->destination = std::move(destinationData);
    e->source = std::move(sourceData);
    e->future = future;
    group->cpuThread->enqueue(e);

    Future r;
    r.impl = std::move(future);
    return r;
  }

  void alltoall(const std::vector<at::Tensor>& outputs, const std::vector<at::Tensor>& inputs, CUstream stream) {
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

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

    // bool isCuda = input.is_cuda();

    // if (isCuda) {
    //   CHECK(input.is_cuda());
    //   CHECK(input.device().index() == group->deviceIndex);
    //   CHECK(output.is_cuda());
    //   CHECK(output.device().index() == group->deviceIndex);
    // }

    if (size - 1 != group->peerIndices.size()) {
      throw std::runtime_error("only local alltoall is supported at the moment, sorry :(");
    }

    size_t numel = inputs[0].numel();
    size_t bytes = numel * inputs[0].itemsize();

    for (auto& v : inputs) {
      CHECK(v.is_cuda())
      CHECK(v.numel() * v.itemsize() == bytes);
    }
    for (auto& v : outputs) {
      CHECK(v.is_cuda())
      CHECK(v.numel() * v.itemsize() == bytes);
    }

    // uintptr_t inputAddress = (uintptr_t)input.data_ptr();
    // uintptr_t outputAddress = (uintptr_t)output.data_ptr();

    StreamData& sd = group->getStreamData(stream);
    EventSerializer es(concurrencyEvents[concurrencyIndex], stream);
    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

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
        cudaCopy(buf.cudaPointer, ptr, bytes, stream);
        ptr = buf.cudaPointer;
        temporaryBuffers.push_back(std::move(buf));
      }
      ipcMapper->requestAddress(i, ptr, bytes, &peerMappedAddresses[i], true);
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
    // for (size_t peerIndex : peerIndices) {
    //   cudaCopy((uintptr_t)outputs[ipcRanks[peerIndex]].data_ptr(), peerAddresses[peerIndex], bytes, stream);
    // }
    // cudaCopy((uintptr_t)outputs[rank].data_ptr(), (uintptr_t)inputs[rank].data_ptr(), bytes, stream);

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
        cudaCopy(
            (uintptr_t)outputs[ipcRanks[peerIndex]].data_ptr(), parameters.outputAddress[peerIndex], bytes, stream);
      }
    }

    syncPeers(stream);

    if (outputs[rank].data_ptr() != inputs[rank].data_ptr()) {
      auto copyStream = getCopyStream(stream);
      localEvent.wait(copyStream);
      cudaCopy((uintptr_t)outputs[rank].data_ptr(), (uintptr_t)inputs[rank].data_ptr(), bytes, copyStream);
      localEvent.record(copyStream);
      localEvent.wait(stream);
    }
  }

  void local_small_allreduce(at::Tensor& tensor, c10d::ReduceOp reduceOp, CUstream stream) {
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    CHECK(tensor.is_contiguous());

    bool isCuda = tensor.is_cuda();

    CHECK(isCuda);

    if (size - 1 != group->peerIndices.size()) {
      throw std::runtime_error("only local alltoall is supported at the moment, sorry :(");
    }

    size_t numel = tensor.numel();
    size_t bytes = numel * tensor.itemsize();

    Group* group = &*this->group;

    std::shared_lock unmapLock(unmapMemoryMutex);
    sync(stepValue);

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    IpcMapper* ipcMapper = &*group->ipcMapper;

    c10::cuda::CUDAStreamGuard sg(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));

    auto temporary = torch::empty({(int64_t)numel}, at::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));

    auto local = tensor.flatten();

    cudaCopy((uintptr_t)temporary.data_ptr(), (uintptr_t)local.data_ptr(), bytes, stream);

    uintptr_t address = (uintptr_t)temporary.data_ptr();

    std::array<uintptr_t, 8> peerMappedAddresses;
    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(i, address, bytes, &peerMappedAddresses[i], true);
    }
    ipcMapper->wait();

    std::array<uintptr_t, 8> peerAddresses;
    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(
          concurrencyIndex, peerIndex, opTypeLocalSmallAllReduceCuda, stepValue, peerMappedAddresses[peerIndex], bytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] =
          peerWaitDyn(concurrencyIndex, peerIndex, opTypeLocalSmallAllReduceCuda, stepValue, bytes);
    }

    freePendingIpcEvents();

    syncPeers(stream);
    for (size_t peerIndex : peerIndices) {
      torch::Tensor peerTensor = torch::from_blob(
          (void*)peerAddresses[peerIndex], {(int64_t)numel},
          at::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));

      switch (reduceOp) {
      case c10d::ReduceOp::SUM:
        local.add_(peerTensor);
        break;
      case c10d::ReduceOp::MIN:
        throw std::runtime_error("fixme local small allreduce min");
        break;
      case c10d::ReduceOp::MAX:
        throw std::runtime_error("fixme local small allreduce max");
        break;
      case c10d::ReduceOp::AVG:
        local.add_(peerTensor);
        break;
      default:
        throw std::runtime_error(
            fmt::sprintf("moodist: allreduce: Unsupported reduceOp %s", std::string(tensor.dtype().name())));
      }
    }
    if (reduceOp == c10d::ReduceOp::AVG) {
      local *= 1.0f / size;
    }
    syncPeers(stream);
  }

  std::vector<torch::Tensor> share(const at::Tensor& input, CUstream stream) {
    std::unique_lock l(threadUnsafe);

    size_t size = this->size;

    uint32_t stepValue = getNextStepValue();
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    CHECK(input.is_contiguous());

    bool isCuda = input.is_cuda();

    CHECK(isCuda);

    CHECK(size - 1 == group->peerIndices.size());

    size_t bytes = input.numel() * input.itemsize();

    uintptr_t inputAddress = (uintptr_t)input.data_ptr();

    Group* group = &*this->group;

    std::shared_lock unmapLock(unmapMemoryMutex);
    sync(stepValue);

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    IpcMapper* ipcMapper = &*group->ipcMapper;

    std::array<uintptr_t, 8> peerMappedAddresses;
    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(i, inputAddress, bytes, &peerMappedAddresses[i], true);
    }
    ipcMapper->wait();

    std::array<uintptr_t, 8> peerAddresses;
    for (size_t peerIndex : peerIndices) {
      peerWriteDyn(concurrencyIndex, peerIndex, opTypeShareCuda, stepValue, peerMappedAddresses[peerIndex], bytes);
    }
    for (size_t peerIndex : peerIndices) {
      peerAddresses[peerIndex] = peerWaitDyn(concurrencyIndex, peerIndex, opTypeShareCuda, stepValue, bytes);
    }
    freePendingIpcEvents();
    syncPeers(stream);

    std::vector<torch::Tensor> r;
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        r.push_back(input);
      } else {
        size_t peerIndex = group->getPeerIndex(i);
        r.push_back(torch::from_blob(
            (void*)peerAddresses[peerIndex], input.sizes(),
            at::TensorOptions().dtype(input.dtype()).device(input.device())));
      }
    }

    return r;
  }

  void cudaBarrier(CUstream stream) {
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

  void shutdown() {
    group->cpuThread->kill(false);
  }
};

void globalsDtor() {
  std::vector<ProcessGroupImpl*> pgs;
  {
    std::lock_guard l(activeProcessGroupsMutex);
    pgs = activeProcessGroups;
  }
  for (auto* pg : pgs) {
    if (pg->group && pg->group->cpuThread) {
      pg->group->cpuThread->kill();
    }
  }
}

struct FreeMemoryCallback : at::FreeMemoryCallback {
  virtual bool Execute() {
    std::unique_lock l(activeProcessGroupsMutex);
    std::unique_lock unmapLock(unmapMemoryMutex, std::try_to_lock);
    for (auto* pg : activeProcessGroups) {
      pg->freeMemory(unmapLock.owns_lock());
    }
    return false;
  };
};

void registerFreeMemoryCallback() {
  at::FreeCudaMemoryCallbacksRegistry()->Register("moodist", []() { return std::make_unique<FreeMemoryCallback>(); });
}

ProcessGroup::ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size)
    : c10d::ProcessGroup(rank, size) {
  impl = std::make_unique<ProcessGroupImpl>(store, rank, size);

  setBackend(torch::DeviceType::CPU, c10d::ProcessGroup::CUSTOM, c10::make_intrusive<Backend>(*this));
  setBackend(torch::DeviceType::CUDA, c10d::ProcessGroup::CUSTOM, c10::make_intrusive<Backend>(*this));
}
ProcessGroup::~ProcessGroup() {
  impl.reset();
}

struct WorkImpl : c10d::Work {
  std::optional<std::variant<torch::Tensor, std::vector<torch::Tensor>>> outputs;
  std::optional<std::variant<torch::Tensor, std::vector<torch::Tensor>>> inputs;
  WorkStream* w = nullptr;
  bool waited = false;
  virtual void synchronize() override {}
  virtual bool wait(std::chrono::milliseconds timeout = std::chrono::milliseconds(0)) override {
    waited = true;
    if (w) {
      CHECK(w->stream != nullptr);
      auto currentStream = c10::cuda::getCurrentCUDAStream();
      if (w->joinedStream != currentStream) {
        CHECK_CU(cuStreamWaitEvent(currentStream, w->event, CU_EVENT_WAIT_DEFAULT));
        w->joinedStream.store(currentStream, std::memory_order_relaxed);
        for (auto& optvar : {inputs, outputs}) {
          if (!optvar) {
            continue;
          }
          auto& var = *optvar;
          if (var.index() == 0) {
            std::get<0>(var).record_stream(currentStream);
          } else {
            for (auto& t : std::get<1>(var)) {
              t.record_stream(currentStream);
            }
          }
        }
      }
    }
    return true;
  }

  WorkImpl(
      std::optional<std::variant<torch::Tensor, std::vector<torch::Tensor>>> outputs,
      std::optional<std::variant<torch::Tensor, std::vector<torch::Tensor>>> inputs, WorkStream* w = nullptr)
      : outputs(std::move(outputs)), inputs(std::move(inputs)), w(w) {}

  ~WorkImpl() {
    if (w) {
      if (!waited) {
        wait();
      }
      CHECK(w->stream != nullptr);
      std::lock_guard l(w->owner->mutex);
      w->owner->free.push_back(*w);
    }
  }

  virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    std::vector<torch::Tensor> result;
    if (outputs) {
      if (outputs->index() == 0) {
        result = {std::get<0>(*outputs)};
      } else {
        result = std::get<1>(*outputs);
      }
    }
    auto fut = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), std::vector<c10::Device>{result[0].device()});
    fut->markCompleted(at::IValue(result));
    return fut;
  }
};

c10::intrusive_ptr<Work> ProcessGroup::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  CHECK(outputTensors.size() == 1 && inputTensors.size() == 1);
  const auto& input = inputTensors[0];
  int64_t numel = input.numel();
  size_t bytes = numel * input.itemsize();
  const auto& outputList = outputTensors[0];
  for (auto& v : outputList) {
    CHECK(v.numel() == numel);
  }
  bool isCuda = input.is_cuda();
  CUstream stream = isCuda ? (CUstream)c10::cuda::getCurrentCUDAStream() : nullptr;
  std::optional<c10::cuda::CUDAStreamGuard> sg;
  if (isCuda) {
    sg.emplace(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
  }
  auto output =
      torch::empty({(int64_t)impl->size, numel}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  impl->all_gather(output, input, stream);
  if (isCuda) {
    size_t n = impl->size;
    for (size_t i = 0; i != n; ++i) {
      CHECK_CU(cuMemcpyAsync((uintptr_t)outputList[i].data_ptr(), (uintptr_t)output[i].data_ptr(), bytes, stream));
    }
  } else {
    size_t n = impl->size;
    for (size_t i = 0; i != n; ++i) {
      std::memcpy(outputList[i].data_ptr(), output[i].data_ptr(), bytes);
    }
  }
  return c10::make_intrusive<WorkImpl>(outputList, input);
}

c10::intrusive_ptr<Work>
ProcessGroup::_allgather_base(at::Tensor& outputbuffer, at::Tensor& inputbuffer, const c10d::AllgatherOptions& opts) {
  bool isCuda = outputbuffer.is_cuda();
  CUstream stream = isCuda ? (CUstream)c10::cuda::getCurrentCUDAStream() : nullptr;
  if (isCuda) {
    WorkStream* w = impl->getWorkStream(stream);
    CHECK_CU(cuEventRecord(w->event, stream));
    CHECK_CU(cuStreamWaitEvent(w->stream, w->event, CU_EVENT_WAIT_DEFAULT));
    impl->all_gather(outputbuffer, inputbuffer, w->stream);
    // outputbuffer.record_stream(c10::Stream(c10::Stream::UNSAFE, outputbuffer.device(), (c10::StreamId)w->stream));
    // inputbuffer.record_stream(c10::Stream(c10::Stream::UNSAFE, inputbuffer.device(), (c10::StreamId)w->stream));
    CHECK_CU(cuEventRecord(w->event, w->stream));
    return c10::make_intrusive<WorkImpl>(outputbuffer, inputbuffer, w);
  } else {
    impl->all_gather(outputbuffer, inputbuffer, nullptr);
    return c10::make_intrusive<WorkImpl>(outputbuffer, inputbuffer);
  }
}

c10::intrusive_ptr<Work> ProcessGroup::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts) {
  CHECK(outputTensors.size() == 1);
  CHECK(inputTensors.size() == 1);
  return _allgather_base(outputTensors[0], inputTensors[0], opts);
}

c10::intrusive_ptr<Work> ProcessGroup::_reduce_scatter_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor, const c10d::ReduceScatterOptions& opts) {
  bool isCuda = outputTensor.is_cuda();
  CUstream stream = isCuda ? (CUstream)c10::cuda::getCurrentCUDAStream() : nullptr;
  if (isCuda) {
    WorkStream* w = impl->getWorkStream(stream);
    CHECK_CU(cuEventRecord(w->event, stream));
    CHECK_CU(cuStreamWaitEvent(w->stream, w->event, CU_EVENT_WAIT_DEFAULT));
    impl->reduce_scatter(outputTensor, inputTensor, opts.reduceOp, w->stream);
    // outputTensor.record_stream(c10::Stream(c10::Stream::UNSAFE, outputTensor.device(), (c10::StreamId)w->stream));
    // inputTensor.record_stream(c10::Stream(c10::Stream::UNSAFE, inputTensor.device(), (c10::StreamId)w->stream));
    CHECK_CU(cuEventRecord(w->event, w->stream));
    return c10::make_intrusive<WorkImpl>(outputTensor, inputTensor, w);
  } else {
    impl->reduce_scatter(outputTensor, inputTensor, opts.reduceOp, nullptr);
    return c10::make_intrusive<WorkImpl>(outputTensor, inputTensor);
  }
}

c10::intrusive_ptr<Work> ProcessGroup::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
    const c10d::ReduceScatterOptions& opts) {
  CHECK(outputTensors.size() == 1);
  CHECK(inputTensors.size() == 1);
  return _reduce_scatter_base(outputTensors[0], inputTensors[0], opts);
}

c10::intrusive_ptr<Work> ProcessGroup::barrier(const c10d::BarrierOptions& opts) {
  impl->barrier();
  return c10::make_intrusive<WorkImpl>(torch::Tensor(), torch::Tensor());
}

c10::intrusive_ptr<Work> ProcessGroup::broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  CHECK(tensors.size() == 1);
  auto& tensor = tensors[0];
  bool isCuda = tensor.is_cuda();
  if (isCuda) {
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    WorkStream* w = impl->getWorkStream(stream);
    CHECK_CU(cuEventRecord(w->event, stream));
    CHECK_CU(cuStreamWaitEvent(w->stream, w->event, CU_EVENT_WAIT_DEFAULT));
    impl->broadcast(tensor, opts.rootRank, w->stream);
    CHECK_CU(cuEventRecord(w->event, w->stream));
    return c10::make_intrusive<WorkImpl>(tensor, std::nullopt, w);
  } else {
    impl->broadcast(tensor, opts.rootRank, nullptr);
    return c10::make_intrusive<WorkImpl>(tensors, std::nullopt);
  }
}

c10::intrusive_ptr<Work> ProcessGroup::allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  WorkStream* w = nullptr;
  for (auto& tensor : tensors) {
    int64_t numel = tensor.numel();
    CHECK(numel > 0);
    bool isCuda = tensor.is_cuda();
    CUstream stream = isCuda ? (CUstream)c10::cuda::getCurrentCUDAStream() : nullptr;
    if (isCuda) {
      if (!w) {
        w = impl->getWorkStream(stream);
      }
      CHECK_CU(cuEventRecord(w->event, stream));
      CHECK_CU(cuStreamWaitEvent(w->stream, w->event, CU_EVENT_WAIT_DEFAULT));
      stream = w->stream;
    }
    // if (numel < 1024 * 1024 && impl->size - 1 == impl->group->peerIndices.size() && isCuda &&
    //     (opts.reduceOp == c10d::ReduceOp::SUM || opts.reduceOp == c10d::ReduceOp::AVG)) {
    if (false) {
      impl->local_small_allreduce(tensor, opts.reduceOp, stream);
    } else if (numel % impl->size == 0) {
      auto t = tensor.view({(int64_t)impl->size, -1});
      auto o = t[impl->rank];
      impl->reduce_scatter(o, t, opts.reduceOp, stream);
      impl->all_gather(t, o, stream);
    } else {
      int64_t newsize = (numel + impl->size - 1) / impl->size * impl->size;
      size_t bytes = numel * tensor.itemsize();
      std::optional<c10::cuda::CUDAStreamGuard> sg;
      if (isCuda) {
        sg.emplace(c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
      }
      torch::Tensor temporary =
          torch::empty({newsize}, torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
      if (tensor.device().is_cuda()) {
        CHECK_CU(cuMemcpyAsync((uintptr_t)temporary.data_ptr(), (uintptr_t)tensor.data_ptr(), bytes, stream));
      } else {
        std::memcpy(temporary.data_ptr(), tensor.data_ptr(), bytes);
      }

      auto t = temporary.view({(int64_t)impl->size, -1});
      auto o = t[impl->rank];
      impl->reduce_scatter(o, t, opts.reduceOp, stream);
      impl->all_gather(t, o, stream);

      if (tensor.device().is_cuda()) {
        CHECK_CU(cuMemcpyAsync((uintptr_t)tensor.data_ptr(), (uintptr_t)temporary.data_ptr(), bytes, stream));
      } else {
        std::memcpy(tensor.data_ptr(), temporary.data_ptr(), bytes);
      }
    }
    if (w) {
      CHECK(stream == w->stream);
      // tensor.record_stream(c10::Stream(c10::Stream::UNSAFE, tensor.device(), (c10::StreamId)w->stream));
      CHECK_CU(cuEventRecord(w->event, w->stream));
    }
  }

  for (auto& tensor : tensors) {
    CHECK(tensor.is_cuda() == (w != nullptr));
  }

  return c10::make_intrusive<WorkImpl>(tensors, std::nullopt, w);
}

c10::intrusive_ptr<Work>
ProcessGroup::allreduce_coalesced(std::vector<at::Tensor>& tensors, const c10d::AllreduceCoalescedOptions& opts) {
  return allreduce(tensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroup::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
    const c10d::GatherOptions& opts) {
  CHECK(outputTensors.size() <= 1);
  CHECK(inputTensors.size() == 1);
  impl->gather(outputTensors.empty() ? std::vector<at::Tensor>() : outputTensors[0], inputTensors[0], opts.rootRank);
  return c10::make_intrusive<WorkImpl>(
      outputTensors.empty() ? std::vector<at::Tensor>() : outputTensors[0], inputTensors);
}

c10::intrusive_ptr<Work> ProcessGroup::reduce(std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
  CHECK(tensors.size() == 1);
  auto& tensor = tensors[0];
  if (impl->size == 1) {
    return c10::make_intrusive<WorkImpl>(tensor, std::nullopt);
  }
  bool isCuda = tensor.is_cuda();
  if (isCuda) {
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    WorkStream* w = impl->getWorkStream(stream);
    CHECK_CU(cuEventRecord(w->event, stream));
    CHECK_CU(cuStreamWaitEvent(w->stream, w->event, CU_EVENT_WAIT_DEFAULT));
    impl->reduce(tensor, opts.rootRank, opts.reduceOp, w->stream);
    CHECK_CU(cuEventRecord(w->event, w->stream));
    return c10::make_intrusive<WorkImpl>(tensor, std::nullopt, w);
  } else {
    throw std::runtime_error("cpu reduce is not currently supported :(");
  }
}

c10::intrusive_ptr<Work> ProcessGroup::scatter(
    std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
  CHECK(outputTensors.size() == 1);
  auto& outputs = outputTensors[0];
  if (impl->queues.empty()) {
    for (size_t i = 0; i != getSize(); ++i) {
      impl->queues.push_back(makeQueue(i));
    }
  }
  if (getRank() == opts.rootRank) {
    CHECK(inputTensors.size() == 1);
    auto& inputs = inputTensors[0];
    CHECK(inputs.size() == getSize());
    for (size_t i = 0; i != inputs.size(); ++i) {
      auto t = inputs[i].cpu();
      impl->queues[i]->put(t, 0);
    }
    // outputTensors[0].copy_(inputs[getRank()]);
  }
  auto t = *impl->queues.at(getRank())->get().first;
  outputTensors[0].copy_(t);

  impl->barrier();

  return c10::make_intrusive<WorkImpl>(torch::Tensor(), std::nullopt);
}

c10::intrusive_ptr<Work> ProcessGroup::alltoall(
    std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors, const c10d::AllToAllOptions& opts) {
  CHECK(inputTensors.size() == getSize());
  CHECK(outputTensors.size() == getSize());

  bool isCuda = outputTensors[0].is_cuda();
  if (isCuda) {
    // throw std::runtime_error("Moodist: CUDA alltoall is not currently supported");
    for (auto& v : inputTensors) {
      CHECK(v.is_cuda());
    }
    for (auto& v : outputTensors) {
      CHECK(v.is_cuda());
    }
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    WorkStream* w = impl->getWorkStream(stream);
    Event e = Event::reference(w->event);
    e.record(stream);
    e.wait(w->stream);
    impl->alltoall(outputTensors, inputTensors, w->stream);
    e.record(w->stream);
    return c10::make_intrusive<WorkImpl>(outputTensors, inputTensors, w);
  } else {

    for (size_t i = 0; i != getSize(); ++i) {
      std::vector<at::Tensor> output = {outputTensors[i]};
      std::vector<std::vector<at::Tensor>> input;
      if (i == getRank()) {
        input = {inputTensors};
      }
      c10d::ScatterOptions scatteropts;
      scatteropts.rootRank = i;
      scatter(output, input, scatteropts);
    }

    return c10::make_intrusive<WorkImpl>(torch::Tensor(), std::nullopt);
  }
}

c10::intrusive_ptr<Work> ProcessGroup::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes, const c10d::AllToAllOptions& opts) {

  CHECK(outputSplitSizes.size() == 0);
  CHECK(inputSplitSizes.size() == 0);

  bool isCuda = outputTensor.is_cuda();
  if (isCuda) {
    CHECK(inputTensor.is_cuda());
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    WorkStream* w = impl->getWorkStream(stream);
    Event e = Event::reference(w->event);
    e.record(stream);
    e.wait(w->stream);
    impl->alltoall(outputTensor.tensor_split(getSize()), inputTensor.tensor_split(getSize()), w->stream);
    e.record(w->stream);
    return c10::make_intrusive<WorkImpl>(outputTensor, inputTensor, w);
  } else {
    auto outputs = outputTensor.chunk(getSize());
    auto inputs = inputTensor.chunk(getSize());

    return alltoall(outputs, inputs);
  }
}

std::vector<torch::Tensor> ProcessGroup::share(const torch::Tensor& input) {
  if (!input.is_cuda()) {
    throw std::runtime_error("Moodist: share is only supported with cuda tensors");
  }
  if (impl->size - 1 != impl->group->peerIndices.size()) {
    throw std::runtime_error(
        "Moodist: share is only supported on local groups (all ranks on the same node with nvlink connectivity)");
  }
  return impl->share(input, c10::cuda::getCurrentCUDAStream());
}

void ProcessGroup::cudaBarrier() {
  return impl->cudaBarrier(c10::cuda::getCurrentCUDAStream());
}

std::shared_ptr<Queue> ProcessGroup::makeQueue(int location, bool streaming, std::optional<std::string> name) {
  return moodist::makeQueue(impl->group, location, streaming, name);
}

std::shared_ptr<Queue>
ProcessGroup::makeQueue(std::vector<int> location, bool streaming, std::optional<std::string> name) {
  return moodist::makeQueue(impl->group, location, streaming, name);
}

Future ProcessGroup::cat(const std::vector<std::pair<int, torch::Tensor>>& locals, std::optional<torch::Tensor> out) {
  return impl->cat(locals, out);
}

Future ProcessGroup::copy(torch::Tensor& destination, const torch::Tensor& source) {
  return impl->copy(destination, source);
}

std::string ProcessGroup::moodist_name() const {
  return impl->group->name;
}

void ProcessGroup::shutdown() {
  impl->shutdown();
}

Future::Future() {}
Future::~Future() {
  if (impl) {
    wait();
  }
}
void Future::wait() {
  while (impl->done == 0) {
    futexWait(&impl->done, 0, std::chrono::seconds(1));
  }
  tensors.clear();
}
torch::Tensor Future::result() {
  wait();
  if (out) {
    return *out;
  }
  if (impl->result == nullptr) {
    throw std::runtime_error("result has already been called on this Future - it cannot be called twice");
  }
  return makeTensor(std::move(impl->result));
}

void setPreferKernelLess(bool value) {
  globalPreferKernelLess = value;
}

} // namespace moodist
