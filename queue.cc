// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "queue.h"
#include "common.h"
#include "cputhread.h"
#include "cuda_copy.h"
#include "hash_map.h"
#include "intrusive_list.h"
#include "queue_internal.h"
#include "simple_vector.h"
#include "synchronization.h"
#include "tensor_ptr.h"

#include <mutex>

namespace moodist {

namespace {

// Helper to convert TensorDataPtr to TensorPtr
TensorPtr tensorFromTensorData(TensorDataPtr data) {
  if (!data) {
    return TensorPtr();
  }
  int device = data->isCuda ? 0 : -1; // TODO: track actual device
  return TensorPtr::from_blob(
      (void*)data->dataPtr, std::span<const int64_t>(data->shape), static_cast<DType>(data->dtype), device);
}

} // namespace

struct WorkCudaDone {
  uintptr_t address = 0;
  uint32_t value = 1;
  void clear() {}
};
using WorkCudaDonePtr = FLPtr<WorkCudaDone>;

struct WorkCudaMappedDone {
  uintptr_t address = 0;
  uint32_t value = 1;
  void clear() {}
};
using WorkCudaMappedDonePtr = FLPtr<WorkCudaDone>;

struct QueueWorkImpl {
  std::atomic_uint32_t done = 0;
  WorkCudaDonePtr cudaDone = nullptr;
  WorkCudaDonePtr cudaMappedDone = nullptr;
  bool queuedPutReady;
  uint32_t transactionKey = 0;
  ~QueueWorkImpl() {}
  void wait() {
    if (cudaDone != nullptr) {
      CHECK_CU(cuStreamWaitValue32(
          wrapperApi.cudaGetCurrentStream(), cudaDone->address, cudaDone->value, CU_STREAM_WAIT_VALUE_GEQ));
    } else if (cudaMappedDone != nullptr) {
      CHECK_CU(cuStreamWaitValue32(
          wrapperApi.cudaGetCurrentStream(), cudaMappedDone->address, cudaMappedDone->value, CU_STREAM_WAIT_VALUE_GEQ));
    } else {
      while (done == 0) {
        futexWait(&done, 0, std::chrono::seconds(1));
      }
    }
  }
};

namespace {

struct IncomingSource {
  struct Item {
    TensorDataPtr tensor;
    uint32_t key;
    bool ready = false;
  };
  Vector<Item> items;

  struct Transaction {
    std::vector<Item> items;
    bool committed;

    void clear() {
      items.clear();
      if (items.size() >= 16) {
        items.shrink_to_fit();
      }
      committed = false;
    }
  };

  using TransactionPtr = FLPtr<Transaction>;

  HashMap<uint32_t, TransactionPtr> transactions;
};

struct Waiting {
  Waiting* next = nullptr;
  IntrusiveListLink<Waiting> link;
  uintptr_t remoteQueueAddress;
  uint32_t key;
  uint32_t source;
};

Waiting* allocWaiting(uint32_t source, uint32_t key, uintptr_t remoteQueueAddress) {
  Waiting* w = FreeList<Waiting>::pop();
  if (!w) {
    w = new Waiting();
  }
  w->remoteQueueAddress = remoteQueueAddress;
  w->key = key;
  w->source = source;
  return w;
}

void freeWaiting(Waiting* w) {
  FreeList<Waiting>::push(w, 0x100);
}

uint64_t key64(uint32_t source, uint32_t key) {
  return ((uint64_t)source << 32) | key;
}

struct RemoteGetResult {
  std::atomic_uint32_t done = 0;
  std::atomic_uint32_t safe = 0;
  TensorDataPtr data;
  size_t queueSize = -1;
};

struct QueuedPut {
  Group* group;
  int location;
  uintptr_t remoteAddress;
  TensorDataPtr tensor;
  std::shared_ptr<QueueWorkImpl> work;
  uint32_t transactionKey = 0;
  uint8_t transactionOp = 0;
};

struct QueueStorage {
  std::atomic_uint32_t size = 0;
  std::atomic_uint32_t incomingSize = 0;
  SpinMutex mutex;
  Vector<TensorDataPtr> vector;
  HashMap<uint32_t, std::unique_ptr<IncomingSource>> incoming;

  HashMap<uint32_t, RemoteGetResult*> activeOutgoingRemoteGets;

  HashMap<uint64_t, Waiting*> waitMap;
  IntrusiveList<Waiting, &Waiting::link> waitList;

  bool isMulticast = false;
  bool isMulticastLocal = false;
  SimpleVector<int> multicastLocations;
  SimpleVector<uintptr_t> multicastRemoteAddresses;

  std::atomic_size_t putQueueSize = 0;
  Vector<QueuedPut> putQueue;

  int location = -1;
  bool streaming = false;
};

SpinMutex namedQueuesMutex;
HashMap<std::string, QueueStorage*> namedQueues;

void barrier(Group* group) {
  uint32_t concurrencyIndex = 0;
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

uintptr_t sendCreateQueue(
    Group* group, int location, uintptr_t address, bool streaming, std::optional<std::string> name) {
  uint32_t concurrencyIndex = 0;
  std::atomic_uintptr_t outAddress = 0;
  std::string outError;
  std::atomic_uint32_t cpuDone = 0;
  QueueEntryCreateQueue* e = group->cpuThread->freelistCreateQueue.pop();
  e->task = name ? taskCreateQueueNamed : taskCreateQueue;
  e->stepValue = 0;
  e->sd = &group->getStreamData(nullptr);
  e->concurrencyIndex = concurrencyIndex;
  e->cpuDone = &cpuDone;
  e->outAddress = &outAddress;
  e->outError = &outError;
  e->location = location;
  e->address = address;
  e->streaming = streaming;
  e->name = name;
  group->cpuThread->enqueue(e);
  while (cpuDone == 0) {
    futexWait(&cpuDone, 0, std::chrono::seconds(10));
  }
  if (!outError.empty()) {
    throw std::runtime_error(outError);
  }
  return outAddress;
}

void create(Group* group, int location, QueueStorage*& qs, uintptr_t& remoteAddress, bool streaming,
    std::optional<std::string> name) {
  CHECK(location >= 0 && location < group->size);
  if (streaming) {
    throw std::runtime_error("Queue: streaming is not fully implemented");
  }

  if (name) {
    remoteAddress = sendCreateQueue(group, location, 0, streaming, name);
    CHECK(remoteAddress != 0);
    if (location == group->rank) {
      qs = (QueueStorage*)remoteAddress;
    } else {
      qs = new QueueStorage();
      qs->location = location;
      qs->streaming = streaming;
    }
  } else {
    qs = new QueueStorage();
    qs->location = location;
    qs->streaming = streaming;

    if (location == group->rank) {
      sendCreateQueue(group, location, (uintptr_t)qs, streaming, name);
      remoteAddress = (uintptr_t)qs;
    } else {
      remoteAddress = sendCreateQueue(group, location, 0, streaming, name);
    }

    barrier(group);
  }
}

void create(Group* group, const SimpleVector<int>& locations, QueueStorage*& qs, SimpleVector<uintptr_t>& remoteAddress,
    std::optional<std::string> name) {
  uintptr_t address = 0;

  if (name) {
    remoteAddress.resize(locations.size());
    for (size_t i = 0; i != locations.size(); ++i) {
      int location = locations[i];
      std::string iname = fmt::sprintf("%s[loc-%d]", *name, i);
      uintptr_t a = sendCreateQueue(group, location, 0, false, iname);
      CHECK(a != 0);
      if (location == group->rank) {
        qs = (QueueStorage*)a;
      }
      remoteAddress[i] = a;
    }
  } else {
    qs = new QueueStorage();

    remoteAddress.resize(locations.size());
    for (size_t i = 0; i != locations.size(); ++i) {
      int location = locations[i];
      if (location == group->rank) {
        sendCreateQueue(group, location, (uintptr_t)qs, false, std::nullopt);
        remoteAddress[i] = (uintptr_t)qs;
      } else {
        remoteAddress[i] = sendCreateQueue(group, location, 0, false, std::nullopt);
      }
    }
    barrier(group);
  }
}

void sendTransaction(
    Group* group, int location, uintptr_t remoteAddress, uint32_t transactionKey, uint8_t transactionOp) {
  CHECK(remoteAddress != 0);
  QueueEntryQueueTransaction* e = group->cpuThread->freelistQueueTransaction.pop();
  e->task = taskQueueTransaction;
  e->location = location;
  e->remoteAddress = remoteAddress;
  e->transactionKey = transactionKey;
  e->op = transactionOp;
  group->cpuThread->enqueue(e);
}

std::atomic_uint32_t nextPutKey = getRng()();

template<typename Tensor>
void sendPut(Group* group, int location, bool streaming, uintptr_t remoteAddress, Tensor tensor,
    std::shared_ptr<QueueWorkImpl> work, uint32_t remoteGetKey, uint32_t transactionKey, size_t queueSize = -1) {
  size_t numel = tensor->numel();
  size_t bytes = numel * tensor->itemsize();

  CHECK(tensor->shape.size() < 256);

  uintptr_t tensorAddress = tensor->data();
  if (!tensor->isCuda) {
    CHECK(cpu_allocator::owns(tensorAddress));
  }

  uint32_t key = 0;
  while (key == 0) {
    key = nextPutKey.fetch_add(1);
  }

  QueueEntryQueuePut* e = group->cpuThread->freelistQueuePut.pop();
  e->task = taskQueuePut;
  e->location = location;
  e->remoteAddress = remoteAddress;
  e->tensor = &*tensor;
  e->putKey = key;
  e->remoteGetKey = remoteGetKey;
  e->transactionKey = transactionKey;
  e->queueSize = queueSize;
  e->streaming = streaming;
  if (work) {
    e->callback = [work = std::move(work), tensor = std::move(tensor)](
                      uintptr_t* cudaDoneAddress, uint32_t* cudaDoneValue) {
      if (tensor->isCuda) {
        CHECK(work->cudaDone != nullptr);
        *cudaDoneAddress = work->cudaDone->address;
        *cudaDoneValue = work->cudaDone->value;
      } else {
        if (work->cudaMappedDone != nullptr) {
          *(volatile uint32_t*)work->cudaMappedDone->address = work->cudaMappedDone->value;
        } else {
          work->done = 1;
          futexWakeAll(&work->done);
        }
      }
    };
  } else {
    e->callback = [tensor = std::move(tensor)](uintptr_t*, uint32_t*) {};
  }
  group->cpuThread->enqueue(e);
}

template<typename Tensor>
void sendPutRemote(Group* group, int location, bool streaming, uintptr_t remoteAddress, Tensor tensor,
    uint32_t remoteGetKey, size_t queueSize) {
  CHECK(location != group->rank);
  sendPut(group, location, streaming, remoteAddress, std::move(tensor), nullptr, remoteGetKey, 0, queueSize);
}

std::atomic_uint32_t nextRemoteGetKey = getRng()();

void sendStartGet(Group* group, int location, uintptr_t localQueueAddress, uintptr_t remoteQueueAddress, uint32_t key) {
  QueueEntryQueueGet* e = group->cpuThread->freelistQueueGet.pop();
  e->task = taskQueueGet;
  e->location = location;
  e->localQueueAddress = localQueueAddress;
  e->remoteQueueAddress = remoteQueueAddress;
  e->key = key;
  e->op = QueueEntryQueueGet::opStart;
  group->cpuThread->enqueue(e);
}

void sendStopGet(Group* group, int location, uintptr_t localQueueAddress, uintptr_t remoteQueueAddress, uint32_t key) {
  QueueEntryQueueGet* e = group->cpuThread->freelistQueueGet.pop();
  e->task = taskQueueGet;
  e->location = location;
  e->localQueueAddress = localQueueAddress;
  e->remoteQueueAddress = remoteQueueAddress;
  e->key = key;
  e->op = QueueEntryQueueGet::opStop;
  group->cpuThread->enqueue(e);
}

struct TimeoutHelper {
  bool block;
  bool hasTimeout;
  std::chrono::time_point<std::chrono::steady_clock> now;
  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> end;
  std::atomic_uint32_t* futex;
  TimeoutHelper(std::atomic_uint32_t* futex, bool block, std::optional<float> timeout)
      : futex(futex), block(block), hasTimeout(block && timeout.has_value()) {
    if (block) {
      if (timeout.has_value()) {
        now = std::chrono::steady_clock::now();
        start = now;
        end = now + std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::duration<double, std::ratio<1, 1>>(*timeout));
      }
    }
  }
  bool expired() {
    return !block || (hasTimeout && now >= end);
  }
  void wait() {
    if (block) {
      if (hasTimeout) {
        if (*futex == 0) {
          while (true) {
            if (std::chrono::seconds(1) < end - now) {
              futexWait(futex, 0, std::chrono::seconds(1));
            } else {
              futexWait(futex, 0, end - now);
            }
            if (*futex != 0) {
              break;
            }
            now = std::chrono::steady_clock::now();
            if (now >= end) {
              break;
            }
          }
        }
      } else {
        while (*futex == 0) {
          futexWait(futex, 0, std::chrono::seconds(1));
        }
      }
    }
  }
};

} // namespace

struct QueueImpl {

  std::shared_ptr<Group> group;
  int location = -1;

  QueueStorage* qs = nullptr;
  uintptr_t remoteAddress = 0;

  bool isMulticast = false;
  bool isMulticastLocal = false;
  SimpleVector<int> multicastLocations;
  SimpleVector<uintptr_t> multicastRemoteAddresses;

  std::optional<std::string> name;

  QueueImpl(std::shared_ptr<Group> group, int location, bool streaming, std::optional<std::string> name)
      : group(group), location(location), name(name) {
    create(&*group, location, qs, remoteAddress, streaming, name);
    CHECK(qs != nullptr);
    CHECK(remoteAddress != 0);
    CHECK(qs->streaming == streaming);
  }

  QueueImpl(std::shared_ptr<Group> group, std::vector<int> locations, bool streaming, std::optional<std::string> name)
      : group(group) {
    if (locations.empty()) {
      throw std::runtime_error("Queue cannot be constructed with an empty location list");
    }
    if (streaming) {
      throw std::runtime_error("Streaming broadcast queues are currently not supported!");
    }
    isMulticast = true;
    location = locations[0];
    isMulticastLocal = std::find(locations.begin(), locations.end(), group->rank) != locations.end();
    HashMap<int, bool> duplicate;
    multicastLocations.resize(locations.size());
    for (size_t i = 0; i != locations.size(); ++i) {
      int r = locations[i];
      if (r < 0 || r >= group->size) {
        throw std::runtime_error(fmt::sprintf("Queue: invalid location rank %d", r));
      }
      if (duplicate[r]) {
        throw std::runtime_error(fmt::sprintf("Queue: location rank %d was specified twice in the list", r));
      }
      duplicate[r] = true;

      multicastLocations[i] = r;
    }
    create(&*group, multicastLocations, qs, multicastRemoteAddresses, name);
    CHECK(qs != nullptr);
    CHECK(remoteAddress == 0);
    CHECK(multicastRemoteAddresses.size() == multicastLocations.size());

    qs->isMulticast = isMulticast;
    qs->isMulticastLocal = isMulticastLocal;
    qs->multicastLocations = multicastLocations;
    qs->multicastRemoteAddresses = multicastRemoteAddresses;
  }

  std::pair<TensorPtr, size_t> get(bool block, std::optional<float> timeout) {
    // fixme: this needs to be refactored such that -
    //        local get is fairly queued alongside remote gets
    //        multi-threaded gets (local & remote) on the same object are also fairly queued
    CHECK(qs != nullptr);

    if (location != group->rank && !isMulticast) {
      uint32_t key = 0;
      while (key == 0) {
        key = nextRemoteGetKey.fetch_add(1);
      }
      RemoteGetResult result;
      {
        std::lock_guard l(qs->mutex);
        CHECK(qs->vector.empty());
        CHECK(qs->size == 0);
        CHECK(qs->incomingSize == 0);
        CHECK(qs->activeOutgoingRemoteGets.find(key) == qs->activeOutgoingRemoteGets.end());
        qs->activeOutgoingRemoteGets[key] = &result;
      }
      TimeoutHelper timeouthelper(&result.done, block, timeout);
      sendStartGet(&*group, location, (uintptr_t)qs, remoteAddress, key);
      timeouthelper.wait();
      if (result.done == 0) {
        sendStopGet(&*group, location, (uintptr_t)qs, remoteAddress, key);
      }

      while (result.done == 0) {
        futexWait(&result.done, 0, std::chrono::seconds(1));
      }

      bool hasData = result.data != nullptr;

      while (result.safe == 0) {
        _mm_pause();
      }

      {
        std::unique_lock l(qs->mutex);
        CHECK(qs->vector.empty());
        CHECK(qs->size == 0);
        CHECK(qs->incomingSize == 0);
        CHECK(qs->activeOutgoingRemoteGets.find(key) == qs->activeOutgoingRemoteGets.end());
      }

      CHECK((result.data != nullptr) == hasData);
      CHECK(result.queueSize != -1);

      if (result.data != nullptr) {
        return {tensorFromTensorData(std::move(result.data)), result.queueSize};
      }
      return {TensorPtr(), result.queueSize};
    }

    if (isMulticast && !isMulticastLocal) {
      throw std::runtime_error(
          fmt::sprintf("Rank %d cannot call get on this multicast Queue since it was not specified as a location on "
                       "construction",
              group->rank));
    }

    {
      std::lock_guard l(qs->mutex);
      size_t queueSize = qs->size.load();
      CHECK(qs->vector.size() == queueSize);
      if (queueSize) {
        TensorDataPtr r = std::move(qs->vector.front());
        qs->vector.pop_front();
        --qs->size;
        return {tensorFromTensorData(std::move(r)), queueSize};
      }
    }

    TimeoutHelper timeouthelper(&qs->size, block, timeout);
    while (true) {
      timeouthelper.wait();
      std::lock_guard l(qs->mutex);
      CHECK(qs->size == qs->vector.size());
      if (!qs->vector.empty()) {
        TensorDataPtr r = std::move(qs->vector.front());
        qs->vector.pop_front();
        --qs->size;
        return {tensorFromTensorData(std::move(r)), 0};
      }
      if (timeouthelper.expired()) {
        return {TensorPtr(), 0};
      }
    }
  }

  QueueWork put(TensorPtr value, uint32_t transactionKey, bool waitOnDestroy = true) {
    // CHECK(location != group->rank);
    // CHECK(qs == nullptr);

    CHECK(value.is_contiguous());

    auto work = std::make_shared<QueueWorkImpl>();

    TensorDataPtr td = TensorDataPtr::make();

    td->dtype = static_cast<int>(value.dtype());
    td->itemsize_ = value.itemsize();
    int ndim = value.ndimension();
    td->shape.resize(ndim);
    for (int i = 0; i != ndim; ++i) {
      td->shape[i] = value.size(i);
    }

    int location = this->location;
    uintptr_t remoteAddress = this->remoteAddress;
    if (isMulticast) {
      location = multicastLocations[0];
      remoteAddress = multicastRemoteAddresses[0];
    }
    CHECK(remoteAddress != 0);

    QueueWork r;

    uintptr_t tensorAddress = (uintptr_t)value.data_ptr();

    td->dataPtr = tensorAddress;
    td->dataBytes = td->itemsize() * td->numel();

    if (tensorAddress == 0 && td->dataBytes != 0) {
      throw std::runtime_error(fmt::sprintf("Queue.put value is a %d-byte tensor with null address", td->dataBytes));
    }

    if (value.is_cpu()) {
      if (cpu_allocator::owns(tensorAddress)) {
        // For contiguous tensors, data_ptr() == storage base
        auto handle = cpu_allocator::getCpuBuffer(tensorAddress);
        CHECK(handle != nullptr);
        td->buffer = std::move(handle);
      } else {
        td->buffer = AllocatedCpuBufferSharedPtr::make();
        *td->buffer = group->allocateCpu(td->dataBytes);
        td->dataPtr = (uintptr_t)td->buffer->cpuPointer;
        std::memcpy((void*)td->data(), (void*)tensorAddress, td->bytes());

        CHECK(cpu_allocator::owns(td->data()));
      }

      std::unique_lock l(qs->mutex, std::defer_lock);

      if (qs->putQueueSize && (l.lock(), qs->putQueueSize)) {
        work->queuedPutReady = true;
        ++qs->putQueueSize;
        qs->putQueue.emplace_back();
        auto& qp = qs->putQueue.back();
        qp.group = &*group;
        qp.location = location;
        qp.remoteAddress = remoteAddress;
        qp.work = work;
        qp.tensor = std::move(td);
        qp.transactionKey = transactionKey;

        CHECK(qs->putQueueSize == qs->putQueue.size());
      } else {
        sendPut(&*group, location, qs->streaming, remoteAddress, std::move(td), work, 0, transactionKey);
      }
    } else {
      // CUDA tensors must always wait - we need to keep storage alive until transfer completes
      waitOnDestroy = true;

      if (group->rdmaSupportsCuda) {
        td->isCuda = true;
        work->cudaDone = WorkCudaDonePtr::make();
        if (!work->cudaDone->address) {
          work->cudaDone->address = group->getNextCudaUint32();
        }
        ++work->cudaDone->value;
      } else {
        td->buffer = AllocatedCpuBufferSharedPtr::make();
        *td->buffer = group->allocateCpu(td->dataBytes);
        td->dataPtr = (uintptr_t)td->buffer->cpuPointer;
        cudaCopy(td->data(), tensorAddress, td->bytes(), api->cudaGetCurrentStream());

        work->cudaMappedDone = WorkCudaMappedDonePtr::make();
        if (!work->cudaMappedDone->address) {
          work->cudaMappedDone->address = group->getNextCudaMappedUint32();
        }
        ++work->cudaMappedDone->value;
      }

      work->queuedPutReady = false;

      ++qs->putQueueSize;
      {
        std::lock_guard l(qs->mutex);
        qs->putQueue.emplace_back();
        auto& qp = qs->putQueue.back();
        qp.group = &*group;
        qp.location = location;
        qp.remoteAddress = remoteAddress;
        qp.work = work;
        qp.tensor = std::move(td);
        qp.transactionKey = transactionKey;

        CHECK(qs->putQueueSize == qs->putQueue.size());
      }

      Function<void()> f = [group = this->group, work = &*work, qs = this->qs] {
        work->queuedPutReady = true;
        std::lock_guard l(qs->mutex);
        CHECK(!qs->putQueue.empty());
        while (qs->putQueue.begin()->work->queuedPutReady) {
          auto i = qs->putQueue.begin();
          if (i->transactionOp != 0) {
            sendTransaction(&*group, i->location, i->remoteAddress, i->transactionKey, i->transactionOp);
          } else {
            sendPut(&*group, i->location, qs->streaming, i->remoteAddress, std::move(i->tensor), std::move(i->work), 0,
                i->transactionKey);
          }
          qs->putQueue.pop_front();
          --qs->putQueueSize;
          if (qs->putQueue.empty()) {
            break;
          }
        }
      };

      CHECK_CU(cuLaunchHostFunc(
          wrapperApi.cudaGetCurrentStream(),
          [](void* ptr) {
            Function<void()>((FunctionPointer)ptr)();
          },
          f.release()));
    }

    // Move tensor into QueueWork to keep it alive during async operation
    r.tensor = std::move(value);
    r.waitOnDestroy = waitOnDestroy;
    r.impl = std::move(work);
    return r;
  }

  size_t qsize() const {
    if ((isMulticast && isMulticastLocal) || location == group->rank) {
      return qs->size;
    }
    throw std::runtime_error("Remote qsize is not implemented");
  }

  bool wait(std::optional<float> timeout) const {
    if ((isMulticast && isMulticastLocal) || location == group->rank) {
      TimeoutHelper timeouthelper(&qs->size, true, timeout);
      timeouthelper.wait();
      return qs->size != 0;
    }
    throw std::runtime_error("Remote wait is not implemented");
  }

  uint32_t transactionBegin() {
    uint32_t key = 0;
    while (key == 0) {
      key = nextPutKey.fetch_add(1);
    }
    // log.info("transaction begin returning key %#x\n", key);
    return key;
  }
  void transactionOp(uint32_t transactionKey, uint8_t op) {
    int location = this->location;
    uintptr_t remoteAddress = this->remoteAddress;
    if (isMulticast) {
      location = multicastLocations[0];
      remoteAddress = multicastRemoteAddresses[0];
    }
    CHECK(remoteAddress != 0);
    std::unique_lock l(qs->mutex, std::defer_lock);
    if (qs->putQueueSize && (l.lock(), qs->putQueueSize)) {
      auto work = std::make_shared<QueueWorkImpl>();
      work->queuedPutReady = true;
      ++qs->putQueueSize;
      qs->putQueue.emplace_back();
      auto& qp = qs->putQueue.back();
      qp.group = &*group;
      qp.location = location;
      qp.remoteAddress = remoteAddress;
      qp.work = work;
      qp.tensor = nullptr;
      qp.transactionKey = transactionKey;
      qp.transactionOp = op;

      CHECK(qs->putQueueSize == qs->putQueue.size());
    } else {
      sendTransaction(&*group, location, remoteAddress, transactionKey, op);
    }
  }
  void transactionCancel(uint32_t transactionKey) {
    transactionOp(transactionKey, transactionOpCancel);
  }
  void transactionCommit(uint32_t transactionKey) {
    transactionOp(transactionKey, transactionOpCommit);
  }
};

QueueWork::QueueWork() {}
QueueWork::~QueueWork() {
  if (waitOnDestroy) {
    wait();
  }
}
void QueueWork::wait() {
  if (!impl) {
    return;
  }
  impl->wait();
  // Release the tensor (TensorPtr destructor handles refcount)
  tensor.reset();
}

Queue::~Queue() {
  delete (QueueImpl*)impl;
}

std::pair<TensorPtr, size_t> Queue::get(bool block, std::optional<float> timeout) {
  return ((QueueImpl*)impl)->get(block, timeout);
}
QueueWork Queue::put(TensorPtr value, uint32_t transactionKey, bool waitOnDestroy) {
  return ((QueueImpl*)impl)->put(std::move(value), transactionKey, waitOnDestroy);
}
size_t Queue::qsize() const {
  return ((QueueImpl*)impl)->qsize();
}
bool Queue::wait(std::optional<float> timeout) const {
  return ((QueueImpl*)impl)->wait(timeout);
}

uint32_t Queue::transactionBegin() {
  return ((QueueImpl*)impl)->transactionBegin();
}
void Queue::transactionCancel(uint32_t id) {
  return ((QueueImpl*)impl)->transactionCancel(id);
}
void Queue::transactionCommit(uint32_t id) {
  return ((QueueImpl*)impl)->transactionCommit(id);
}

std::string_view Queue::name() const {
  if (!((QueueImpl*)impl)->name) {
    throw std::runtime_error("This Queue has no name");
  }
  return *((QueueImpl*)impl)->name;
}

static int foo;

Queue::Queue(void* p) {
  CHECK(p == &foo);
}

std::shared_ptr<Queue> makeQueue(std::shared_ptr<Group> group, int location, bool streaming, std::string_view name) {
  auto r = std::make_shared<Queue>(&foo);
  std::optional<std::string> nameOpt = name.empty() ? std::nullopt : std::optional<std::string>(std::string(name));
  r->impl = (void*)new QueueImpl(group, location, streaming, nameOpt);
  return r;
}

std::shared_ptr<Queue> makeQueue(
    std::shared_ptr<Group> group, std::vector<int> location, bool streaming, std::string_view name) {
  auto r = std::make_shared<Queue>(&foo);
  std::optional<std::string> nameOpt = name.empty() ? std::nullopt : std::optional<std::string>(std::string(name));
  r->impl = (void*)new QueueImpl(group, location, streaming, nameOpt);
  return r;
}

static std::atomic_uint32_t nextIncomingKey = getRng()();

uint32_t queuePrepare(uintptr_t queueAddress, uint32_t source, uint32_t getKey, uint32_t transactionKey) {
  auto* qs = (QueueStorage*)queueAddress;
  if (getKey != 0) {
    CHECK(transactionKey == 0);
    std::lock_guard l(qs->mutex);
    auto i = qs->activeOutgoingRemoteGets.find(getKey);
    CHECK(i != qs->activeOutgoingRemoteGets.end());
    return 0;
  }
  uint32_t key = 0;
  while (key == 0) {
    key = nextIncomingKey.fetch_add(1);
  }
  std::lock_guard l(qs->mutex);
  auto& ptr = qs->incoming[source];
  if (!ptr) {
    ptr = std::make_unique<IncomingSource>();
  }
  ++qs->incomingSize;
  if (transactionKey != 0) {
    auto& t = ptr->transactions[transactionKey];
    if (t == nullptr) {
      t = IncomingSource::TransactionPtr::make();
    }
    CHECK(!t->committed);
    t->items.push_back({{}, key, false});
  } else {
    ptr->items.push_back({{}, key, false});
  }
  return key;
}

template<typename Tensor>
void qsAdd(Group* group, QueueStorage* qs, Tensor&& tensor) {
  CHECK(tensor != nullptr);
  if (qs->isMulticast && group->rank == qs->multicastLocations[0]) {
    auto shared = TensorDataSharedPtr::make();
    *shared = *tensor;
    size_t n = qs->multicastLocations.size();
    for (size_t i = 0; i != n; ++i) {
      if (qs->multicastLocations[i] != group->rank) {
        sendPutRemote(group, qs->multicastLocations[i], qs->streaming, qs->multicastRemoteAddresses[i], shared, 0, -1);
      }
    }
  }
  qs->vector.push_back(std::move(tensor));
}

void transferQueuedItems(Group* group, QueueStorage* qs, IncomingSource* ptr, std::unique_lock<SpinMutex>& l) {
  while (!ptr->items.empty() && ptr->items.front().ready) {
    qsAdd(group, qs, std::move(ptr->items.front().tensor));
    ++qs->size;
    ptr->items.pop_front();
    --qs->incomingSize;
  }
  while (!qs->vector.empty() && !qs->waitList.empty()) {
    Waiting* w = &qs->waitList.front();
    qs->waitList.pop_front();
    CHECK(qs->waitMap.find(key64(w->source, w->key)) != qs->waitMap.end());
    qs->waitMap.erase(key64(w->source, w->key));
    TensorDataPtr r = std::move(qs->vector.front());
    qs->vector.pop_front();
    --qs->size;
    sendPutRemote(group, w->source, qs->streaming, w->remoteQueueAddress, std::move(r), w->key, 0);
    freeWaiting(w);
  }
  l.unlock();
  futexWakeAll(&qs->size);
}

void queueFinish(Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key, TensorDataPtr tensor,
    uint32_t getKey, uint32_t transactionKey, size_t queueSize) {
  CHECK(tensor != nullptr);
  auto* qs = (QueueStorage*)queueAddress;
  std::unique_lock l(qs->mutex);
  if (getKey) {
    CHECK(transactionKey == 0);
    auto i = qs->activeOutgoingRemoteGets.find(getKey);
    CHECK(i != qs->activeOutgoingRemoteGets.end());
    RemoteGetResult* r = i->second;
    qs->activeOutgoingRemoteGets.erase(i);
    l.unlock();
    r->data = std::move(tensor);
    r->queueSize = queueSize;
    r->done = 1;
    futexWakeAll(&i->second->done);
    r->safe = 1;
    return;
  }
  IncomingSource* ptr = &*qs->incoming[source];
  CHECK(ptr != nullptr);
  if (transactionKey != 0) {
    auto i = ptr->transactions.find(transactionKey);
    if (i != ptr->transactions.end()) {
      CHECK(i->second != nullptr);
      auto& t = *i->second;
      bool found = false;
      for (auto& v : t.items) {
        if (v.key == key) {
          v.tensor = std::move(tensor);
          v.ready = true;
          found = true;
          break;
        }
      }
      CHECK(found);
      if (t.committed) {
        for (auto& v : t.items) {
          if (!v.ready) {
            // log.info("transaction %#x still has an item which is not ready\n", transactionKey);
            return;
          }
        }
        auto placeholder = ptr->items.end();
        for (auto i = ptr->items.begin(); i != placeholder; ++i) {
          if (i->key == transactionKey) {
            placeholder = i;
            break;
          }
        }
        // log.info("transaction completing %d items\n", t.items.size());
        CHECK(placeholder != ptr->items.end());
        size_t index = placeholder - ptr->items.begin();
        ptr->items.erase(placeholder);
        for (auto& v : t.items) {
          ptr->items.insert(ptr->items.begin() + index, std::move(v));
          ++index;
        }
        ptr->transactions.erase(i);
        transferQueuedItems(group, qs, ptr, l);
      }
    }
    return;
  }
  CHECK(ptr->items.size() != 0);
  if (ptr->items.front().key == key) {
    qsAdd(group, qs, tensor);
    ++qs->size;
    ptr->items.pop_front();
    --qs->incomingSize;
    transferQueuedItems(group, qs, ptr, l);
  } else {
    bool found = false;
    for (auto& v : ptr->items) {
      if (v.key == key) {
        v.tensor = std::move(tensor);
        v.ready = true;
        found = true;
        break;
      }
    }
    CHECK(found);
  }
}

void queueRemoteGetStart(
    Group* group, uintptr_t queueAddress, uint32_t source, uintptr_t remoteQueueAddress, uint32_t key) {
  // log.info("got remote get start %#x %d %#x %#x\n", queueAddress, source, remoteQueueAddress, key);
  auto* qs = (QueueStorage*)queueAddress;
  std::unique_lock l(qs->mutex);
  if (!qs->vector.empty()) {
    // log.info("data is available!\n");
    TensorDataPtr r = std::move(qs->vector.front());
    // log.info("data: %s\n", hexstr(r.data_ptr(), r.itemsize() * r.numel()));
    size_t queueSize = qs->size;
    CHECK(queueSize == qs->vector.size());
    qs->vector.pop_front();
    --qs->size;
    sendPutRemote(group, source, qs->streaming, remoteQueueAddress, std::move(r), key, queueSize);
  } else {
    // log.info("adding to wait list \n");
    Waiting* w = allocWaiting(source, key, remoteQueueAddress);
    qs->waitList.push_back(*w);
    CHECK(qs->waitMap.find(key64(source, key)) == qs->waitMap.end());
    qs->waitMap[key64(source, key)] = w;
  }
}
void queueRemoteGetStop(
    Group* group, uintptr_t queueAddress, uint32_t source, uintptr_t remoteQueueAddress, uint32_t key) {
  // log.info("got remote get stop %#x %d %#x %#x\n", queueAddress, source, remoteQueueAddress, key);

  auto* qs = (QueueStorage*)queueAddress;
  std::lock_guard l(qs->mutex);
  auto i = qs->waitMap.find(key64(source, key));
  if (i != qs->waitMap.end()) {
    Waiting* w = i->second;
    qs->waitList.erase(*i->second);
    qs->waitMap.erase(i);
    freeWaiting(w);
    QueueEntryQueueGet* e = group->cpuThread->freelistQueueGet.pop();
    e->task = taskQueueGet;
    e->location = source;
    e->localQueueAddress = queueAddress;
    e->remoteQueueAddress = remoteQueueAddress;
    e->key = key;
    e->op = QueueEntryQueueGet::opStopAck;
    group->cpuThread->enqueue(e);
  }
}

void queueRemoteGetStopAck(
    Group* group, uintptr_t queueAddress, uint32_t source, uintptr_t remoteQueueAddress, uint32_t key) {
  // log.info("got remote get stop ack %#x %d %#x %#x\n", queueAddress, source, remoteQueueAddress, key);

  auto* qs = (QueueStorage*)queueAddress;
  std::unique_lock l(qs->mutex);
  auto i = qs->activeOutgoingRemoteGets.find(key);
  if (i != qs->activeOutgoingRemoteGets.end()) {
    // log.info("ack remote get found, freeing\n");
    auto* ptr = i->second;
    qs->activeOutgoingRemoteGets.erase(i);
    l.unlock();
    ptr->data = nullptr;
    ptr->queueSize = 0;
    ptr->done = 1;
    futexWakeAll(&ptr->done);
    ptr->safe = 1;
  } else {
    log.error("ack not found!\n");
    CHECK(false);
  }
}

void queueTransactionCommit(Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key) {
  // log.info("transaction commit %#x\n", key);
  auto* qs = (QueueStorage*)queueAddress;
  std::unique_lock l(qs->mutex);
  IncomingSource* ptr = &*qs->incoming[source];
  CHECK(ptr != nullptr);
  auto i = ptr->transactions.find(key);
  if (i == ptr->transactions.end()) {
    return;
  }
  CHECK(i->second != nullptr);
  auto& t = *i->second;
  CHECK(!t.committed);
  t.committed = true;
  for (auto& item : t.items) {
    if (!item.ready) {
      // log.info("transaction not yet ready, inserting placeholder\n");
      ptr->items.push_back({{}, key, false});
      return;
    }
  }
  // log.info("transaction completing %d items (1)\n", t.items.size());
  for (auto& v : t.items) {
    ptr->items.push_back(std::move(v));
  }
  ptr->transactions.erase(i);
  transferQueuedItems(group, qs, ptr, l);
}

void queueTransactionCancel(Group* group, uintptr_t queueAddress, uint32_t source, uint32_t key) {
  // log.info("transaction cancel %#x\n", key);
  auto* qs = (QueueStorage*)queueAddress;
  std::unique_lock l(qs->mutex);
  IncomingSource* ptr = &*qs->incoming[source];
  CHECK(ptr != nullptr);
  ptr->transactions.erase(key);
}

QueueInfo queueGetOrCreate(std::string_view name, int location, bool streaming) {
  std::lock_guard l(namedQueuesMutex);
  std::string nameStr(name); // Convert once for map operations
  auto i = namedQueues.find(nameStr);
  if (i == namedQueues.end()) {
    QueueStorage* qs = new QueueStorage();
    qs->location = location;
    qs->streaming = streaming;
    i = namedQueues.emplace(std::move(nameStr), qs).first;
  }
  QueueInfo r;
  r.address = (uintptr_t)i->second;
  r.location = i->second->location;
  r.streaming = i->second->streaming;
  return r;
}

} // namespace moodist
