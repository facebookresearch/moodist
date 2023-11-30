#include "cputhread.h"
#include "allgather.h"
#include "async.h"
#include "clock.h"
#include "common.h"
#include "freelist.h"
#include "ib_common.h"
#include "intrusive_list.h"
#include "ipc_mapper.h"
#include "reduce_scatter.h"
#include "setup_comms.h"
#include "simple_vector.h"

#include "fmt/printf.h"

#include <atomic>
#include <cstring>
#include <libibverbs/verbs.h>
#include <memory>
#include <numa.h>
#include <random>
#include <type_traits>
#include <utility>

#include <sys/uio.h>

namespace moodist {

struct CpuThreadImpl;

namespace {

struct QuitCpuThread {};

struct MemoryRegistration {
  std::array<ibv_mr*, 32> mrs;
};

struct CudaMapping {
  uint64_t bufferId;
  size_t bytes;
  MemoryRegistration mr;
};

struct Device {
  IntrusiveListLink<Device> link;
  IbCommon* ib;
  ibv_pd* protectionDomain;
  ibv_cq* cq;
  size_t currentCqEntries = 0;
};

struct Callback {
  Function<void()> onComplete;
  size_t refcount = 0;
  Callback* next = nullptr;

  void decref() {
    CHECK(refcount >= 1);
    if (--refcount == 0) {
      std::move(onComplete)();
      FreeList<Callback>::push(this, 0x1000);
    }
  }
};

struct RemoteAddressAndKey {
  uintptr_t address;
  SimpleVector<uint32_t> keys;
};

struct CallbackWrapper {
  Callback* callback;
  CallbackWrapper(Callback* callback) : callback(callback) {
    ++callback->refcount;
  }
  ~CallbackWrapper() {
    callback->decref();
  }
  operator Callback*() {
    return callback;
  }
};

template<typename T, size_t poolSize = 0x1000>
struct PoolAllocator {
  std::deque<SimpleVector<std::aligned_storage_t<sizeof(T), alignof(T)>>> all;
  size_t i0 = 0;
  size_t i1 = 0;
  PoolAllocator() {
    all.emplace_back();
    all.back().resize(poolSize);
  }
  ~PoolAllocator() {
    reset();
  }
  template<typename... Args>
  T* allocate(Args&&... args) {
    CHECK(i0 < all.size());
    CHECK(i1 < poolSize);
    T* r = (T*)&all[i0][i1];
    ++i1;
    if (i1 == poolSize) {
      i1 = 0;
      ++i0;
      if (i0 == all.size()) {
        all.emplace_back();
        all.back().resize(poolSize);
      }
    }
    return new (r) T(std::forward<Args>(args)...);
  }
  void reset() {
    for (size_t i = 0; i != i0; ++i) {
      for (auto& v : all[i]) {
        ((T*)&v)->~T();
      }
    }
    for (size_t i = 0; i != i1; ++i) {
      ((T*)&all[i0][i])->~T();
    }
    i0 = 0;
    i1 = 0;
  }
};

} // namespace

template<Dtype dtype>
struct CpuTypes;

template<>
struct CpuTypes<Dtype::float32> {
  static constexpr size_t bytes = 4;
  using T = float;
};

template<>
struct CpuTypes<Dtype::float64> {
  static constexpr size_t bytes = 8;
  using T = double;
};

template<>
struct CpuTypes<Dtype::int32> {
  static constexpr size_t bytes = 4;
  using T = int32_t;
};

template<>
struct CpuTypes<Dtype::int64> {
  static constexpr size_t bytes = 8;
  using T = int64_t;
};

template<Dtype dtype>
typename CpuTypes<dtype>::T cpuLoad(uintptr_t address) {
  return *(typename CpuTypes<dtype>::T*)address;
}

template<Dtype dtype>
void cpuStore(uintptr_t address, typename CpuTypes<dtype>::T value) {
  *(typename CpuTypes<dtype>::T*)address = value;
}

template<Dtype dtype, Reduction reduction>
struct CpuReductions {
  using T = typename CpuTypes<dtype>::T;
  static T add(T a, T b) {
    return a + b;
  }
  static T min(T a, T b) {
    return std::min(a, b);
  }
  static T max(T a, T b) {
    return a + b;
  }
  static void reduce(uintptr_t dst, uintptr_t src) {
    if (reduction == Reduction::sum) {
      cpuStore<dtype>(dst, add(cpuLoad<dtype>(dst), cpuLoad<dtype>(src)));
    } else if (reduction == Reduction::min) {
      cpuStore<dtype>(dst, min(cpuLoad<dtype>(dst), cpuLoad<dtype>(src)));
    } else if (reduction == Reduction::max) {
      cpuStore<dtype>(dst, max(cpuLoad<dtype>(dst), cpuLoad<dtype>(src)));
    } else {
      fatal("Unsupported cpu reduction");
    }
  }
};

struct TemporaryBufferHandle {
  AllocatedBuffer buffer;
  Vector<AllocatedBuffer>* container = nullptr;
  TemporaryBufferHandle() = default;
  TemporaryBufferHandle(AllocatedBuffer buffer, Vector<AllocatedBuffer>* container)
      : buffer(std::move(buffer)), container(container) {}
  TemporaryBufferHandle(TemporaryBufferHandle&& n) {
    *this = std::move(n);
  }
  TemporaryBufferHandle& operator=(TemporaryBufferHandle&& n) {
    std::swap(buffer, n.buffer);
    std::swap(container, n.container);
    return *this;
  }
  ~TemporaryBufferHandle() {
    if (container) {
      container->push_back(std::move(buffer));
    }
  }
  AllocatedBuffer& operator*() {
    return buffer;
  }
  AllocatedBuffer* operator->() {
    return &buffer;
  }
};

struct CpuThreadImpl {
  CpuThread* cpuThread;
  Group* group;
  bool dead = false;

  CpuThreadImpl(CpuThread* cpuThread) : cpuThread(cpuThread), group(cpuThread->group), devices(initDevices()) {}
  ~CpuThreadImpl() {
    dead = true;
  }

  const size_t rank = group->rank;
  const size_t size = group->size;

  static constexpr size_t maxWr = IbCommon::maxWr;
  static constexpr size_t maxCqEntries = IbCommon::maxCqEntries;

  SetupComms* setupComms = &*group->setupComms;
  SimpleVector<Device> devices;
  IntrusiveList<Device, &Device::link> activeDevices;

  size_t bytesWritten = 0;

  HashMap<uintptr_t, std::unique_ptr<MemoryRegistration>> mrMap;
  HashMap<uintptr_t, std::unique_ptr<CudaMapping>> mrMapCuda;

  std::vector<IbvMr> localMrs;

  MemoryRegistration* commsMr = regMr((uintptr_t)group->mySharedMem, group->mySharedMemSize);

  DynamicAddresses* localDyns = group->localDyns;
  Progress* localProgress = group->localProgress;

  MemoryRegistration* cudaStepValueMr =
      regMrCuda(group->cudaStepValue.buffer.cudaPointer, group->cudaStepValue.buffer.bytes);

  std::vector<RemoteAddressAndKey> remoteComms = distributeAddressAndKeys((uintptr_t)group->mySharedMem, commsMr);
  std::vector<RemoteAddressAndKey> remoteCudaStepValue =
      distributeAddressAndKeys(group->cudaStepValue.buffer.cudaPointer, cudaStepValueMr);

  MemoryRegistration* cudaCommsDeviceDataSentMr =
      regMrCuda(group->cudaCommsDeviceDataSent.buffer.cudaPointer, group->cudaCommsDeviceDataSent.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent =
      distributeAddressAndKeys(group->cudaCommsDeviceDataSent.buffer.cudaPointer, cudaCommsDeviceDataSentMr);

  AllocatedArray sendStepValuesStorage = group->allocateArrayHost(sizeof(uint32_t), Group::maxConcurrency);
  MemoryRegistration* sendStepValuesStorageMr =
      regMr((uintptr_t)sendStepValuesStorage.buffer.cpuPointer, sendStepValuesStorage.buffer.bytes);
  uint32_t* sendStepValues = (uint32_t*)sendStepValuesStorage.buffer.cpuPointer;

  AllocatedArray outDynsBuffer = group->allocateArrayHost(sizeof(DynamicAddresses) * size, Group::maxConcurrency);
  MemoryRegistration* outDynsMr = regMr((uintptr_t)outDynsBuffer.buffer.cpuPointer, outDynsBuffer.buffer.bytes);

  Vector<AllocatedBuffer> temporaryBuffers;

  TemporaryBufferHandle allocateTemporaryBuffer(size_t bytes) {
    if (temporaryBuffers.empty()) {
      return TemporaryBufferHandle(group->allocateHost(bytes), &temporaryBuffers);
    }
    auto buffer = std::move(temporaryBuffers.back());
    temporaryBuffers.pop_back();
    if (buffer.bytes < bytes) {
      buffer = group->allocateHost(bytes);
    }
    return TemporaryBufferHandle(std::move(buffer), &temporaryBuffers);
  }

  SimpleVector<Device> initDevices() {
    SimpleVector<Device> devices;
    for (auto& v : group->ibDevs) {
      devices.resize(devices.size() + 1);
      auto& dev = devices.back();
      dev.ib = &*v;
      dev.protectionDomain = v->protectionDomain;
      dev.cq = v->cq;
    }
    return devices;
  }

  CallbackWrapper makeCallback(Function<void()> f) {
    Callback* callback = FreeList<Callback>::pop();
    if (!callback) {
      callback = new Callback();
    }
    CHECK(callback->refcount == 0);
    callback->onComplete = std::move(f);
    return CallbackWrapper(callback);
  }

  void poll() {
    for (auto i = activeDevices.begin(); i != activeDevices.end();) {
      Device& dev = *i;
      ++i;
      ibv_wc wcs[4];
      int n = ibv_poll_cq(dev.cq, 4, wcs);
      CHECK(n >= 0);
      for (size_t i = 0; i != n; ++i) {
        ibv_wc& wc = wcs[i];
        if (wc.status) {
          fatal("rank %d Work completion with status %d (opcode %d, id %#x)\n", rank, wc.status, wc.opcode, wc.wr_id);
        } else {
          --dev.currentCqEntries;
          if (dev.currentCqEntries == 0) {
            activeDevices.erase(dev);
          }
          if (wc.wr_id) {
            Callback* callback = (Callback*)(void*)wc.wr_id;
            callback->decref();
          }
        }
      }
    }
  }

  void writeData(
      Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      Callback* callback = nullptr) {
    CHECK(i >= 0 && i < size);
    CHECK(i != rank);
    auto post = [&]() {
      ibv_sge sge;
      sge.addr = (uintptr_t)localAddress;
      sge.length = bytes;
      sge.lkey = lkey;

      bytesWritten += bytes;

      if (callback) {
        ++callback->refcount;
      }

      while (dev.currentCqEntries == maxCqEntries) {
        poll();
      }
      ++dev.currentCqEntries;
      if (dev.currentCqEntries == 1) {
        activeDevices.push_back(dev);
      }

      ibv_qp_ex* qp = dev.ib->qpex;

      // fmt::printf(
      //     "%d: rdma write %d bytes (%p -> %p, rkey %#x) (dev %d, i %d)\n", rank, bytes, localAddress,
      //     remoteAddress, rkey, &dev - devices.data(), i);

      bool efa = qp != nullptr;
      if (!efa) {
        qp = dev.ib->qpexs[i];
        CHECK(qp != nullptr);
      }

      ibv_wr_start(qp);
      qp->wr_id = (uint64_t)(void*)callback;
      qp->wr_flags = IBV_SEND_SIGNALED;
      ibv_wr_rdma_write(qp, rkey, (uintptr_t)remoteAddress);
      ibv_wr_set_sge_list(qp, 1, &sge);

      if (efa) {
        ibv_wr_set_ud_addr(qp, dev.ib->ahs[i], dev.ib->remoteAddresses[i].qpNum, 0x4242);
      }
      int error = ibv_wr_complete(qp);
      if (error) {
        fatal("ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
      }
    };

    const size_t threshold = 1024ull * 1024 * 896;
    const size_t chunkSize = 1024ull * 1024 * 768;
    if (bytes >= threshold) {
      size_t remaining = bytes;
      bytes = chunkSize;
      while (remaining > 0) {
        if (remaining < threshold) {
          bytes = remaining;
        }
        post();

        localAddress = (void*)((uintptr_t)localAddress + bytes);
        remoteAddress = (void*)((uintptr_t)remoteAddress + bytes);
        remaining -= bytes;
      }
    } else {
      post();
    }
  }

  MemoryRegistration* regMr(uintptr_t address, size_t bytes) {
    auto i = mrMap.find(address);
    if (i != mrMap.end()) {
      return &*i->second;
    }
    auto ptr = std::make_unique<MemoryRegistration>();
    ptr->mrs.fill(nullptr);
    CHECK(devices.size() <= ptr->mrs.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      IbvMr mr = ibv_reg_mr(
          devices[i].ib->protectionDomain, (void*)address, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (!mr) {
        perror("ibv_reg_mr");
        log.error("rank %d failed to register CPU memory at %#x size %#x\n", group->rank, address, bytes);
        TORCH_CHECK(false);
      }
      ptr->mrs[i] = mr;
      localMrs.push_back(std::move(mr));
    }
    MemoryRegistration* r = &*ptr;
    mrMap[address] = std::move(ptr);
    return r;
  }

  MemoryRegistration* regMrCuda(uintptr_t address, size_t bytes) {
    if (bytes == 0) {
      bytes = 1;
    }
    unsigned long long bufferId = -1;
    CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
    CHECK(bufferId != -1);
    auto i = mrMapCuda.find(address);
    if (i != mrMapCuda.end()) {
      if (i->second->bufferId == bufferId && i->second->bytes >= bytes) {
        return &i->second->mr;
      }
    }
    unsigned long long bufferId2 = -1;
    CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + bytes - 1));
    CHECK(bufferId == bufferId2);
    auto ptr = std::make_unique<CudaMapping>();
    ptr->bufferId = bufferId;
    ptr->bytes = bytes;
    ptr->mr.mrs.fill(nullptr);
    CHECK(devices.size() <= ptr->mr.mrs.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      auto* mr = ibv_reg_mr(
          devices[i].ib->protectionDomain, (void*)address, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (!mr) {
        perror("ibv_reg_mr");
        log.error("rank %d failed to register CUDA memory at %#x size %#x\n", group->rank, address, bytes);
        TORCH_CHECK(false);
      }
      localMrs.push_back(std::move(mr));
      ptr->mr.mrs[i] = mr;
      log.debug(
          "new mapped range of %d bytes at %#x -> fd %d  (mr lkey %#x rkey %#x)\n", bytes, address, bufferId, mr->lkey,
          mr->rkey);
    }
    MemoryRegistration* r = &ptr->mr;
    mrMapCuda[address] = std::move(ptr);
    return r;
  }

  std::vector<RemoteAddressAndKey> distributeAddressAndKeys(uintptr_t address, MemoryRegistration* mr) {
    SimpleVector<uint32_t> localKeys;
    localKeys.resize(devices.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      localKeys[i] = mr->mrs.at(i)->rkey;
    }
    auto keys = setupComms->allgather(localKeys);
    auto addrs = setupComms->allgather(address);
    for (auto& v : keys) {
      CHECK(v.size() == devices.size());
    }
    std::vector<RemoteAddressAndKey> r;
    r.resize(size);
    for (size_t i = 0; i != size; ++i) {
      r[i].address = addrs[i];
      r[i].keys = keys[i];
    }
    return r;
  }

  template<typename Callback>
  void writeDataDistributed2(
      size_t dst, uintptr_t srcAddr, size_t len, MemoryRegistration* mr, uint64_t dstAddr,
      const std::array<uint32_t, 32>& key, Callback&& callback) {
    CHECK(len > 0);
    size_t offset = 0;
    const size_t chunks = devices.size();
    const size_t chunkSize = len / chunks / 128u * 128u;
    for (size_t i = 0; i != chunks; ++i) {
      size_t n = i == chunks - 1 ? len - offset : chunkSize;
      auto& dev = devices[i];
      if (n != 0) {
        bool efa = dev.ib->qpex != nullptr;
        writeData(
            dev, dst, (void*)(srcAddr + offset), mr->mrs[i]->lkey, (void*)(dstAddr + offset), key[i], n,
            makeCallback([callback, i, efa]() { callback(i, efa, true); }));
        if (!efa) {
          callback(i, true, false);
        }
      } else {
        callback(i, true, true);
      }
      offset += n;
    }
  }

  void writeStep(size_t deviceIndex, size_t dst, uint32_t stepValue, size_t concurrencyIndex) {
    Device& dev = devices[deviceIndex];
    uintptr_t dstOffset = group->getSharedOffset(&localProgress[size * concurrencyIndex + rank].stepValue);
    localProgress[size * concurrencyIndex + rank].stepValue = stepValue;
    writeData(
        dev, dst, &localProgress[size * concurrencyIndex + rank].stepValue, commsMr->mrs[deviceIndex]->lkey,
        (void*)(remoteComms[dst].address + dstOffset), remoteComms[dst].keys[deviceIndex], sizeof(stepValue));
  }

  void writeCpuStep(size_t deviceIndex, size_t dst, uint32_t stepValue, size_t concurrencyIndex) {
    Device& dev = devices[deviceIndex];
    uintptr_t dstOffset = group->getSharedOffset(&localProgress[size * concurrencyIndex + rank].cpuStepValue);
    localProgress[size * concurrencyIndex + rank].cpuStepValue = stepValue;
    writeData(
        dev, dst, &localProgress[size * concurrencyIndex + rank].cpuStepValue, commsMr->mrs[deviceIndex]->lkey,
        (void*)(remoteComms[dst].address + dstOffset), remoteComms[dst].keys[deviceIndex], sizeof(stepValue));
  }
  void runTests() {
    AllocatedBuffer testBuffer = group->allocateHost(0x1000 * size);
    for (size_t i = 0; i != 1; ++i) {
      log.debug("rank %d starting CPU test %d!\n", rank, i);

      // AllocatedBuffer testBuffer = group->allocateHost(0x1000 * size);
      auto* testMr = regMr((uintptr_t)testBuffer.cpuPointer, testBuffer.bytes);
      auto testRemote = distributeAddressAndKeys((uintptr_t)testBuffer.cpuPointer, testMr);
      uint64_t* testPtr = (uint64_t*)testBuffer.cpuPointer;
      std::mt19937_64 rng;
      rng.seed(rank);
      for (size_t i = 0; i != 512; ++i) {
        testPtr[512 * rank + i] = rng();
      }
      size_t remaining = 0;
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        size_t offset = 512 * rank;
        for (size_t di = 0; di != devices.size(); ++di) {
          auto& dev = devices[di];
          size_t n = 512 / devices.size();
          ++remaining;
          writeData(
              dev, i, testPtr + offset, testMr->mrs[di]->lkey, (uint64_t*)testRemote[i].address + offset,
              testRemote[i].keys[di], n * 8, makeCallback([&]() { --remaining; }));
          offset += n;
        }
      }
      while (remaining) {
        poll();
      }
      setupComms->allgather(0);
      for (size_t i = 0; i != size; ++i) {
        rng.seed(i);
        for (size_t j = 0; j != 512; ++j) {
          uint64_t v = rng();
          if (testPtr[512 * i + j] != v) {
            log.error("i %d j %d expected %#x, but got %#x\n", i, j, v, testPtr[512 * i + j]);
          }
          CHECK(testPtr[512 * i + j] == v);
        }
      }

      log.debug("rank %d CPU test done!\n", rank);
    }
    for (size_t i = 0; i != 1; ++i) {
      log.debug("rank %d starting CUDA test %d!\n", rank, i);
      AllocatedBuffer testBuffer = group->allocateDevice(0x1000 * size);
      auto* testMr = regMrCuda((uintptr_t)testBuffer.cudaPointer, testBuffer.bytes);
      auto testRemote = distributeAddressAndKeys((uintptr_t)testBuffer.cudaPointer, testMr);
      uint64_t* testPtr = (uint64_t*)testBuffer.cudaPointer;
      size_t remaining = 0;
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        size_t offset = 512 * rank;
        for (size_t di = 0; di != devices.size(); ++di) {
          auto& dev = devices[di];
          size_t n = 512 / devices.size();
          ++remaining;
          writeData(
              dev, i, testPtr + offset, testMr->mrs[di]->lkey, (uint64_t*)testRemote[i].address + offset,
              testRemote[i].keys[di], n * 8, makeCallback([&]() { --remaining; }));
          offset += n;
        }
      }
      while (remaining) {
        poll();
      }
      setupComms->allgather(0);

      log.debug("rank %d CUDA test done!\n", rank);
    }
  }

  struct WorkBase {
    IntrusiveListLink<WorkBase> freeLink;
    IntrusiveListLink<WorkBase> workLink;
    IntrusiveListLink<WorkBase> streamLink;
    IntrusiveListLink<WorkBase> concurrencyLink;
  };
  struct Work : WorkBase {
    CpuThreadImpl& self;
    void* state = nullptr;
    bool done = false;
    void (*stepPtr)(Work*);
    const QueueEntry& params;
    const uint32_t stepValue = params.stepValue;
    const uint32_t concurrencyIndex = params.concurrencyIndex;
    const uint32_t streamIndex = params.sd->cpuThreadIndex;
    const size_t size = self.size;
    const size_t rank = self.rank;
    volatile uint32_t* const cpuIn = &self.group->cpuInBuffer.at<volatile uint32_t>(concurrencyIndex);
    volatile uint32_t* const cpuOut = &self.group->cpuOutBuffer.at<volatile uint32_t>(concurrencyIndex);

    Work(CpuThreadImpl& self, const QueueEntry& params) : self(self), params(params) {}

    template<typename T>
    auto getStepPtr() {
      return [](Work* w) { ((T*)w)->step(); };
    }
  };

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define ENTER                                                                                                          \
  {                                                                                                                    \
    if (this->state) {                                                                                                 \
      goto* this->state;                                                                                               \
    }                                                                                                                  \
  }
#define YIELD                                                                                                          \
  {                                                                                                                    \
    this->state = &&CAT(s, __LINE__);                                                                                  \
    return;                                                                                                            \
    CAT(s, __LINE__) :;                                                                                                \
  }
#define DONE                                                                                                           \
  {                                                                                                                    \
    this->done = true;                                                                                                 \
    return;                                                                                                            \
  }

  struct WorkBarrier : Work {
    QueueEntryBarrier& params;
    size_t i;

    WorkBarrier(CpuThreadImpl& self, QueueEntryBarrier& params) : Work(self, params), params(params) {}

    void step() {
      ENTER
      // log.info("barrier enter stepValue %#x concurrency %d\n", stepValue, concurrencyIndex);
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        self.writeCpuStep((rank + i) % self.devices.size(), i, stepValue, concurrencyIndex);
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (self.localProgress[size * concurrencyIndex + i].cpuStepValue < stepValue) {
          YIELD
        }
      }
      // log.info("barrier leave stepValue %#x concurrency %d\n", stepValue, concurrencyIndex);
      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
      self.cpuThread->freelistBarrier.push(&params);
      self.allocatorBarrier.deallocate(this);
      DONE
    }
  };

  struct WorkAllGather : Work {
    QueueEntryAllGather& params;
    size_t i;
    MemoryRegistration* inputMr;
    size_t liveSends;
    size_t index;
    size_t nDynReady = 0;

    const AllGather& allGather = *self.group->allGather;
    const std::vector<size_t>& sendRanks = allGather.sendRanks;
    const std::vector<size_t>& recvRanks = allGather.recvRanks;

    WorkAllGather(CpuThreadImpl& self, QueueEntryAllGather& params) : Work(self, params), params(params) {}

    void step() {
      ENTER

      {
        inputMr = self.regMrCuda(params.inputAddress, params.bytes);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
        // Thus, we register each ranks output address independently, so they get different MRs.
        for (size_t i : recvRanks) {
          auto* mr = self.regMrCuda(params.outputAddress + params.pitch * i, params.bytes);
          outDyns[i].gatherAddress = params.outputAddress;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
          }
        }

        // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
        // until kernel entry, such that we don't receive data until we're ready.
        nDynReady = 0;
        {
          size_t n = 0;
          for (size_t i : recvRanks) {
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];
            self.writeData(
                dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                self.remoteComms[i].keys[di], sizeof(DynamicAddresses), self.makeCallback([this]() { ++nDynReady; }));
            ++n;
          }
        }
      }

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }

      while (nDynReady < recvRanks.size()) {
        YIELD
      }

      self.sendStepValues[concurrencyIndex] = stepValue;

      CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        {
          size_t i = recvRanks[index];
          size_t di = (rank + index) % self.devices.size();
          auto& dev = self.devices[di];
          self.writeStep(di, i, stepValue + index, concurrencyIndex);
        }
        i = sendRanks[index];
        // wait for dyns
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue + index) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)params.inputAddress;

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, inputMr,
              self.localDyns[size * concurrencyIndex + i].gatherAddress + params.pitch * rank,
              self.localDyns[size * concurrencyIndex + i].gatherKey,
              [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices[di];
                  self.writeData(
                      dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                      (void*)(self.remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                      self.remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                }
              });
        }
        while (liveSends >= self.devices.size() * 2) {
          YIELD
        }
      }

      while (liveSends) {
        YIELD
      }

      cpuOut[0] = stepValue;
      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      self.cpuThread->freelistAllGather.push(&params);
      self.allocatorAllGather.deallocate(this);
      DONE
    }
  };

  struct WorkReduceScatter : Work {
    QueueEntryReduceScatter& params;
    size_t i;
    MemoryRegistration* sendMr;
    size_t liveSends;
    size_t index;
    size_t nDynReady = 0;
    uintptr_t sendAddress;

    const ReduceScatter& reduceScatter = *self.group->reduceScatter;
    const std::vector<size_t>& sendRanks = reduceScatter.sendRanks;
    const std::vector<size_t>& recvRanks = reduceScatter.recvRanks;

    WorkReduceScatter(CpuThreadImpl& self, QueueEntryReduceScatter& params) : Work(self, params), params(params) {}

    void step() {
      ENTER

      {
        uintptr_t recvAddress = params.sd->recvBuffer.cudaPointer;
        CHECK(params.sd->recvBuffer.bytes >= params.pitch * recvRanks.size());

        sendAddress = params.sd->sendBuffer.cudaPointer;
        CHECK(params.sd->sendBuffer.bytes >= params.pitch * sendRanks.size());

        CHECK(sendAddress && recvAddress);

        sendMr = self.regMrCuda(sendAddress, params.pitch * sendRanks.size());
        // auto* recvMr = self.regMrCuda(recvAddress, params.pitch * recvRanks.size());

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
        // Thus, we register each ranks output address independently, so they get different MRs.
        size_t n = 0;
        for (size_t i : recvRanks) {
          auto* mr = self.regMrCuda(recvAddress + params.pitch * n, params.pitch);
          outDyns[i].gatherAddress = recvAddress;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
          }
          ++n;
        }

        // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
        // until kernel entry, such that we don't receive data until we're ready.
        nDynReady = 0;
        {
          size_t n = 0;
          for (size_t i : recvRanks) {
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];
            self.writeData(
                dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                self.remoteComms[i].keys[di], sizeof(DynamicAddresses), self.makeCallback([this]() { ++nDynReady; }));
            ++n;
          }
        }
      }

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }

      while (nDynReady < recvRanks.size()) {
        YIELD
      }

      self.sendStepValues[concurrencyIndex] = stepValue;

      CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        // wait for send ready
        while (cpuIn[1] < stepValue + index + 1) {
          YIELD
        }

        {
          size_t i = recvRanks[index];
          size_t di = (rank + index) % self.devices.size();
          auto& dev = self.devices[di];
          self.writeStep(di, i, stepValue + index, concurrencyIndex);
        }
        i = sendRanks[index];
        // wait for dyns
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue + index) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)sendAddress + params.pitch * index;

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, sendMr,
              self.localDyns[size * concurrencyIndex + i].gatherAddress + params.pitch * index,
              self.localDyns[size * concurrencyIndex + i].gatherKey,
              [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices[di];
                  self.writeData(
                      dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                      (void*)(self.remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                      self.remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                }
              });
        }
        while (liveSends >= self.devices.size() * 2) {
          YIELD
        }
      }

      while (liveSends) {
        YIELD
      }

      cpuOut[0] = stepValue;
      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      self.cpuThread->freelistReduceScatter.push(&params);
      self.allocatorReduceScatter.deallocate(this);
      DONE
    }
  };

  struct WorkReduceScatterCpu : Work {
    QueueEntryReduceScatterCpu& params;
    size_t i;
    MemoryRegistration* sendMr;
    size_t liveSends;
    size_t index;
    size_t nDynReady = 0;
    uintptr_t sendAddress;

    uintptr_t prevReduceOutput = 0;

    TemporaryBufferHandle temporaryBuffer;

    const ReduceScatter& reduceScatter = *self.group->reduceScatter;
    const std::vector<size_t>& sendRanks = reduceScatter.sendRanks;
    const std::vector<size_t>& recvRanks = reduceScatter.recvRanks;
    const std::vector<size_t>& ipcRanks = self.group->ipcRanks;
    const std::vector<size_t>& peerIndices = self.group->peerIndices;

    WorkReduceScatterCpu(CpuThreadImpl& self, QueueEntryReduceScatterCpu& params)
        : Work(self, params), params(params) {}

    template<typename T, size_t dindex = 0, size_t opindex = 0>
    decltype(stepPtr) getStepPtr() {
      if constexpr (dindex < (size_t)Dtype::count && opindex < (size_t)Reduction::count) {
        if (dindex == (size_t)params.dindex && opindex == (size_t)params.opindex) {
          return [](Work* w) { ((T*)w)->template step<(Dtype)dindex, (Reduction)opindex>(); };
        }
      }
      if constexpr (dindex < (size_t)Dtype::count) {
        return getStepPtr<T, dindex + 1, opindex>();
      }
      if constexpr (opindex < (size_t)Reduction::count) {
        return getStepPtr<T, dindex, opindex + 1>();
      }
      CHECK(false);
    }

    template<Dtype dtype, Reduction reduction>
    void reduceAdd(uintptr_t output, uintptr_t input) {
      if (prevReduceOutput != output) {
        std::memcpy((void*)output, (void*)input, params.bytes);
        prevReduceOutput = output;
      } else {
        size_t bytes = params.bytes;
        size_t itembytes = CpuTypes<dtype>::bytes;
        uintptr_t end = output + bytes;
        CHECK(bytes % itembytes == 0);
        while (output != end) {
          CpuReductions<dtype, reduction>::reduce(output, input);
          output += itembytes;
          input += itembytes;
        }
      }
    }

    uintptr_t vmread(int pid, uintptr_t src, size_t bytes) {
      if (temporaryBuffer->bytes < bytes) {
        temporaryBuffer = self.allocateTemporaryBuffer(bytes);
      }
      iovec local;
      local.iov_base = temporaryBuffer->cpuPointer;
      local.iov_len = bytes;
      iovec remote;
      remote.iov_base = (void*)src;
      remote.iov_len = bytes;
      if (process_vm_readv(pid, &local, 1, &remote, 1, 0) != bytes) {
        fatal("process_vm_readv failed with error: %s", std::strerror(errno));
      }
      return (uintptr_t)temporaryBuffer->cpuPointer;
    }

    template<Dtype dtype, Reduction reduction>
    void step() {
      ENTER

      {
        uintptr_t recvAddress = params.sd->recvBuffer.cudaPointer;
        CHECK(params.sd->recvBuffer.bytes >= params.pitch * recvRanks.size());

        sendAddress = params.sd->sendBuffer.cudaPointer;
        CHECK(params.sd->sendBuffer.bytes >= params.pitch * sendRanks.size());

        if (!recvRanks.empty()) {
          CHECK(sendAddress && recvAddress);
          CHECK(false);

          sendMr = self.regMr(sendAddress, params.pitch * sendRanks.size());

          DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

          // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
          // Thus, we register each ranks output address independently, so they get different MRs.
          size_t n = 0;
          for (size_t i : recvRanks) {
            auto* mr = self.regMrCuda(recvAddress + params.pitch * n, params.pitch);
            outDyns[i].gatherAddress = recvAddress;
            for (size_t di = 0; di != self.devices.size(); ++di) {
              outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
            }
            ++n;
          }

          nDynReady = 0;
          {
            size_t n = 0;
            for (size_t i : recvRanks) {
              size_t di = (rank + n) % self.devices.size();
              auto& dev = self.devices[di];
              self.writeData(
                  dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                  (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                  self.remoteComms[i].keys[di], sizeof(DynamicAddresses), self.makeCallback([this]() { ++nDynReady; }));
              ++n;
            }
          }
        } else {
          CHECK(recvRanks.empty() && sendRanks.empty());
        }
      }

      self.group->cpuAddresses[concurrencyIndex].inputAddress = params.inputAddress;
      self.group->cpuAddresses[concurrencyIndex].outputAddress = params.outputAddress;
      self.group->cpuAddresses[concurrencyIndex].pid = self.group->pid;
      self.group->cpuAddresses[concurrencyIndex].stepValue = stepValue;

      for (index = 0; index != ipcRanks.size(); ++index) {
        while (self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex])->stepValue < stepValue) {
          YIELD
        }
      }

      while (nDynReady < recvRanks.size()) {
        YIELD
      }

      self.sendStepValues[concurrencyIndex] = stepValue;

      CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        CHECK(false);
        // wait for send ready
        while (cpuIn[1] < stepValue + index + 1) {
          YIELD
        }

        {
          size_t i = recvRanks[index];
          size_t di = (rank + index) % self.devices.size();
          auto& dev = self.devices[di];
          self.writeStep(di, i, stepValue + index, concurrencyIndex);
        }
        i = sendRanks[index];
        // wait for dyns
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue + index) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)sendAddress + params.pitch * index;

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, sendMr,
              self.localDyns[size * concurrencyIndex + i].gatherAddress + params.pitch * index,
              self.localDyns[size * concurrencyIndex + i].gatherKey,
              [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices[di];
                  self.writeData(
                      dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                      (void*)(self.remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                      self.remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                }
              });
        }
        while (liveSends >= self.devices.size() * 2) {
          YIELD
        }
      }

      reduceAdd<dtype, reduction>(params.outputAddress, params.inputAddress + params.pitch * rank);
      for (size_t peerIndex : peerIndices) {
        auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
        reduceAdd<dtype, reduction>(
            params.outputAddress, vmread(x->pid, x->inputAddress + params.pitch * rank, params.bytes));
      }

      while (liveSends) {
        YIELD
      }

      self.group->cpuAddresses[concurrencyIndex].stepValue = stepValue + 1;

      for (index = 0; index != ipcRanks.size(); ++index) {
        while (self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex])->stepValue < stepValue + 1) {
          YIELD
        }
      }

      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);

      temporaryBuffer = {};

      self.cpuThread->freelistReduceScatterCpu.push(&params);
      self.allocatorReduceScatterCpu.deallocate(this);
      DONE
    }
  };

  struct WorkBroadcast : Work {
    QueueEntryBroadcast& params;
    size_t i;
    MemoryRegistration* tensorMr;
    size_t liveSends;
    size_t nDynReady = 0;

    WorkBroadcast(CpuThreadImpl& self, QueueEntryBroadcast& params) : Work(self, params), params(params) {}

    void step() {
      ENTER

      tensorMr = self.regMrCuda(params.tensorAddress, params.bytes);

      if (params.sourceRank != rank) {
        {
          DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

          for (size_t i = 0; i != size; ++i) {
            if (i == rank) {
              continue;
            }
            outDyns[i].gatherAddress = params.tensorAddress;
            for (size_t di = 0; di != self.devices.size(); ++di) {
              outDyns[i].gatherKey[di] = tensorMr->mrs[di]->rkey;
            }
          }

          // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
          // until kernel entry, such that we don't receive data until we're ready.
          nDynReady = 0;
          {
            size_t n = 0;
            for (size_t i = 0; i != size; ++i) {
              if (i == rank) {
                continue;
              }
              size_t di = (rank + n) % self.devices.size();
              auto& dev = self.devices[di];
              self.writeData(
                  dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                  (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                  self.remoteComms[i].keys[di], sizeof(DynamicAddresses), self.makeCallback([this]() { ++nDynReady; }));
              ++n;
            }
          }
        }

        // wait for kernel
        while (cpuIn[0] < stepValue) {
          YIELD
        }

        while (nDynReady < size - 1) {
          YIELD
        }

        i = params.sourceRank;
        {
          size_t di = (rank + i) % self.devices.size();
          self.writeStep(di, i, stepValue, concurrencyIndex);
        }
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
          YIELD
        }
      } else {

        // wait for kernel
        while (cpuIn[0] < stepValue) {
          YIELD
        }

        liveSends = 0;
        for (i = 0; i != size; ++i) {
          if (i == rank) {
            continue;
          }
          // wait for dyns
          while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
            YIELD
          }

          if (params.bytes > 0) {
            uintptr_t srcAddr = (uintptr_t)params.tensorAddress;

            liveSends += self.devices.size();
            self.writeDataDistributed2(
                i, srcAddr, params.bytes, tensorMr, self.localDyns[size * concurrencyIndex + i].gatherAddress,
                self.localDyns[size * concurrencyIndex + i].gatherKey,
                [this, i = this->i](size_t di, bool ordered, bool done) {
                  if (done) {
                    --liveSends;
                  }
                  if (ordered) {
                  }
                });
          }
          while (liveSends) {
            YIELD
          }
          size_t di = (rank + i) % self.devices.size();
          self.writeStep(di, i, stepValue, concurrencyIndex);
        }

        while (liveSends) {
          YIELD
        }
      }

      cpuOut[0] = stepValue;
      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      self.cpuThread->freelistBroadcast.push(&params);
      self.allocatorBroadcast.deallocate(this);
      DONE
    }
  };

  template<typename T>
  struct WorkAllocator {
    PoolAllocator<T> allocator;
    IntrusiveList<WorkBase, &WorkBase::freeLink> free;
    template<typename... Args>
    T* allocate(Args&&... args) {
      if (!free.empty()) {
        T* r = (T*)&free.front();
        free.pop_front();
        r->~T();
        new (r) T(std::forward<Args>(args)...);
        return r;
      }
      return allocator.allocate(std::forward<Args>(args)...);
    }
    void deallocate(T* ptr) {
      free.push_front(*ptr);
    }
  };
  WorkAllocator<WorkBarrier> allocatorBarrier;
  WorkAllocator<WorkAllGather> allocatorAllGather;
  WorkAllocator<WorkReduceScatter> allocatorReduceScatter;
  WorkAllocator<WorkBroadcast> allocatorBroadcast;
  WorkAllocator<WorkReduceScatterCpu> allocatorReduceScatterCpu;

  void entry() {
    try {

      runTests();

      cpuThread->ready = true;

      SimpleVector<IntrusiveList<WorkBase, &WorkBase::concurrencyLink>> concurrency;
      concurrency.resize(Group::maxConcurrency);
      SimpleVector<IntrusiveList<WorkBase, &WorkBase::streamLink>> streams;
      IntrusiveList<WorkBase, &WorkBase::workLink> activeWorks;

      auto enqueue = [&](auto& allocator, auto& params) {
        CHECK(params.concurrencyIndex < Group::maxConcurrency);
        if (params.sd->cpuThreadIndex == -1) {
          // log.info("new stream index %d\n", streams.size());
          params.sd->cpuThreadIndex = streams.size();
          streams.resize(streams.size() + 1);
        }
        auto* work = allocator.allocate(*this, params);
        // work->stepPtr = (void(Work::*)()) & std::remove_pointer_t<decltype(work)>::step;
        work->stepPtr = work->template getStepPtr<std::remove_pointer_t<decltype(work)>>();
        CHECK(work->streamIndex < streams.size());
        if (concurrency[work->concurrencyIndex].empty() && streams[work->streamIndex].empty()) {
          activeWorks.push_back(*work);
          // log.info("new work %#x (%d) added from queue\n", work->params.stepValue, work->concurrencyIndex);
        }
        // log.info("work enqueued, concurrencyIndex %d, streamIndex %d\n", work->concurrencyIndex, work->streamIndex);
        // log.info(
        //     "pre enqueue concurrency (%d) size: %d  stream (%d) size: %d\n", work->concurrencyIndex,
        //     std::distance(concurrency[work->concurrencyIndex].begin(), concurrency[work->concurrencyIndex].end()),
        //     work->streamIndex, std::distance(streams[work->streamIndex].begin(), streams[work->streamIndex].end()));
        streams[work->streamIndex].push_back(*work);
        concurrency[work->concurrencyIndex].push_back(*work);
      };

      while (!cpuThread->terminate.load(std::memory_order_relaxed)) {
        // log.info("activeWorks.size() is %d\n", std::distance(activeWorks.begin(), activeWorks.end()));
        if (cpuThread->queueSize.load(std::memory_order_relaxed) == 0) {
          if (activeWorks.empty() && activeDevices.empty()) {
            futexWait(&cpuThread->queueSize, 0, std::chrono::seconds(10));
            continue;
          }
        }
        for (int i = 0; i != 256; ++i) {
          if (cpuThread->queueSize.load(std::memory_order_relaxed) != 0) {
            --cpuThread->queueSize;

            std::unique_lock l(cpuThread->mutex);
            CHECK(!cpuThread->queue.empty());
            QueueEntry& queueEntry = cpuThread->queue.front();
            cpuThread->queue.pop_front();
            l.unlock();

            CHECK(queueEntry.stepValue < (1ul << 31));

            if (queueEntry.task == taskBarrier) {
              enqueue(allocatorBarrier, (QueueEntryBarrier&)queueEntry);
            } else if (queueEntry.task == taskAllgather) {
              enqueue(allocatorAllGather, (QueueEntryAllGather&)queueEntry);
            } else if (queueEntry.task == taskReduceScatter) {
              enqueue(allocatorReduceScatter, (QueueEntryReduceScatter&)queueEntry);
            } else if (queueEntry.task == taskTerminate) {
              cpuThread->freelistTerminate.push(&queueEntry);
            } else if (queueEntry.task == taskBroadcast) {
              enqueue(allocatorBroadcast, (QueueEntryBroadcast&)queueEntry);
            } else if (queueEntry.task == taskReduceScatterCpu) {
              enqueue(allocatorReduceScatterCpu, (QueueEntryReduceScatterCpu&)queueEntry);
            } else {
              throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
            }
          }
          for (int i2 = 0; i2 != 4; ++i2) {
            poll();

            for (auto i = activeWorks.begin(); i != activeWorks.end();) {
              Work* w = (Work*)&*i;
              CHECK(!w->done);
              //(w->*w->stepPtr)();
              w->stepPtr(w);
              if (w->done) {
                uint32_t concurrencyIndex = w->concurrencyIndex;
                uint32_t streamIndex = w->streamIndex;
                // log.info("work %#x done, concurrencyIndex %d\n", -1, concurrencyIndex);
                // log.info(
                //     "pre concurrency (%d) size: %d  stream (%d) size: %d\n", concurrencyIndex,
                //     std::distance(concurrency[concurrencyIndex].begin(), concurrency[concurrencyIndex].end()),
                //     streamIndex, std::distance(streams[streamIndex].begin(), streams[streamIndex].end()));
                CHECK(&concurrency[concurrencyIndex].front() == w);
                concurrency[concurrencyIndex].pop_front();
                CHECK(&streams[streamIndex].front() == w);
                streams[streamIndex].pop_front();
                // log.info(
                //     "remaining stream (%d) size: %d\n", streamIndex,
                //     std::distance(streams[streamIndex].begin(), streams[streamIndex].end()));
                // log.info(
                //     "remaining concurrency (%d) size: %d\n", concurrencyIndex,
                //     std::distance(concurrency[concurrencyIndex].begin(), concurrency[concurrencyIndex].end()));
                bool queuedThisConcurrency = false;
                if (!streams[streamIndex].empty()) {
                  Work* nextWork = (Work*)&streams[streamIndex].front();
                  if (&concurrency[nextWork->concurrencyIndex].front() == nextWork) {
                    queuedThisConcurrency = nextWork->concurrencyIndex == concurrencyIndex;
                    activeWorks.push_back(*nextWork);
                    // log.info("new work %#x (%d) added from stream\n", -1, nextWork->concurrencyIndex);
                  }
                }
                if (!queuedThisConcurrency && !concurrency[concurrencyIndex].empty()) {
                  Work* nextWork = (Work*)&concurrency[concurrencyIndex].front();
                  if (&streams[nextWork->streamIndex].front() == nextWork) {
                    activeWorks.push_back(*nextWork);
                    // log.info("new work %#x (%d) added from concurrency\n", -1, nextWork->concurrencyIndex);
                  }
                }
                i = activeWorks.erase(i);

                // log.info("activeWorks.size() -> %d\n", std::distance(activeWorks.begin(), activeWorks.end()));
              } else {
                ++i;
              }
            }

            // for (auto& list : works) {
            //   while (!list.empty()) {
            //     Work* w = (Work*)&list.front();
            //     (w->*w->stepPtr)();
            //     if (!w->done) {
            //       break;
            //     }
            //   }
            // }
          }
        }
      }

      // while (true) {

      //   while (cpuThread->queueSize == 0) {
      //     poll();
      //     int sum = 0;
      //     for (auto& dev : devices) {
      //       sum += dev.currentCqEntries;
      //     }
      //     if (sum == 0) {
      //       futexWait(&cpuThread->queueSize, 0, std::chrono::seconds(10));
      //     }
      //   }
      //   --cpuThread->queueSize;

      //   std::unique_lock l(cpuThread->mutex);
      //   CHECK(!cpuThread->queue.empty());
      //   QueueEntry& queueEntry = cpuThread->queue.front();
      //   cpuThread->queue.pop_front();
      //   l.unlock();

      //   CHECK(queueEntry.stepValue < (1ul << 31));

      //   if (queueEntry.task == taskBarrier) {
      //     QueueEntryBarrier& params = (QueueEntryBarrier&)queueEntry;
      //     barrier(params);
      //     cpuThread->freelistBarrier.push(&params);
      //   } else if (queueEntry.task == taskAllgather) {
      //     QueueEntryAllGather& params = (QueueEntryAllGather&)queueEntry;
      //     all_gather(params);
      //     cpuThread->freelistAllGather.push(&params);
      //   } else if (queueEntry.task == taskReduceScatter) {
      //     QueueEntryReduceScatter& params = (QueueEntryReduceScatter&)queueEntry;
      //     reduce_scatter(params);
      //     cpuThread->freelistReduceScatter.push(&params);
      //   } else if (queueEntry.task == taskTerminate) {
      //     cpuThread->freelistTerminate.push(&queueEntry);
      //     break;
      //   } else {
      //     throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
      //   }
      // }
    } catch (const std::exception& e) {
      fatal("CPU thread error: %s\n", e.what());
    } catch (QuitCpuThread) {
    }
  }
};

CpuThread::CpuThread(Group* group) : group(group) {}
CpuThread::~CpuThread() {
  if (thread.joinable()) {
    QueueEntry* e = freelistTerminate.pop();
    e->task = taskTerminate;
    std::unique_lock l(mutex);
    queue.push_back(*e);
    l.unlock();
    terminate = true;
    ++queueSize;
    futexWakeAll(&queueSize);

    thread.join();
  }
}

void CpuThread::start() {
  thread = std::thread([this] {
    async::setCurrentThreadName("moodist-cputhread");
    CHECK_CU(cuCtxSetCurrent(group->cuContext));
    CpuThreadImpl impl(this);
    impl.entry();
  });
}

} // namespace moodist
