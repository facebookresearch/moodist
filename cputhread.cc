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

#include "fmt/printf.h"

#include <atomic>
#include <libibverbs/verbs.h>
#include <memory>
#include <numa.h>
#include <random>
#include <type_traits>
#include <utility>

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
  std::vector<uint32_t> keys;
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
struct PoolAllocatorReused {
  std::deque<std::vector<std::aligned_storage_t<sizeof(T), alignof(T)>>> all;
  size_t i0 = 0;
  size_t i1 = 0;
  PoolAllocatorReused() {
    all.emplace_back();
    all.back().resize(poolSize);
  }
  ~PoolAllocatorReused() {
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
  std::vector<Device> devices;

  size_t bytesWritten = 0;

  HashMap<uintptr_t, std::unique_ptr<MemoryRegistration>> mrMap;
  HashMap<uintptr_t, std::unique_ptr<CudaMapping>> mrMapCuda;

  std::vector<IbvMr> localMrs;

  MemoryRegistration* commsMr = regMr((uintptr_t)group->mySharedMem, group->mySharedMemSize);

  DynamicAddresses* localDyns = group->localDyns;
  Progress* localProgress = group->localProgress;

  MemoryRegistration* cudaStepValueMr = regMrCuda(group->cudaStepValue.cudaPointer, group->cudaStepValue.bytes);

  std::vector<RemoteAddressAndKey> remoteComms = distributeAddressAndKeys((uintptr_t)group->mySharedMem, commsMr);
  std::vector<RemoteAddressAndKey> remoteCudaStepValue =
      distributeAddressAndKeys(group->cudaStepValue.cudaPointer, cudaStepValueMr);

  MemoryRegistration* cudaCommsDeviceDataSentMr =
      regMrCuda(group->cudaCommsDeviceDataSent.cudaPointer, group->cudaCommsDeviceDataSent.bytes);
  std::vector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent =
      distributeAddressAndKeys(group->cudaCommsDeviceDataSent.cudaPointer, cudaCommsDeviceDataSentMr);

  AllocatedArray sendStepValuesStorage = group->allocateHostArray(sizeof(uint32_t), Group::maxConcurrency);
  MemoryRegistration* sendStepValuesStorageMr =
      regMr((uintptr_t)sendStepValuesStorage.buffer.cpuPointer, sendStepValuesStorage.buffer.bytes);
  uint32_t* sendStepValues = (uint32_t*)sendStepValuesStorage.buffer.cpuPointer;

  volatile uint32_t* const cpuIn = (uint32_t*)group->cpuInBuffer.cpuPointer;
  volatile uint32_t* const cpuOut = (uint32_t*)group->cpuOutBuffer.cpuPointer;

  AllocatedArray outDynsBuffer = group->allocateHostArray(sizeof(DynamicAddresses) * size);
  DynamicAddresses* outDyns = (DynamicAddresses*)outDynsBuffer.cpuPointer;
  MemoryRegistration* outDynsMr = regMr((uintptr_t)outDynsBuffer.cpuPointer, outDynsBuffer.bytes);

  std::vector<uint32_t> dynReady;

  std::vector<Device> initDevices() {
    std::vector<Device> devices;
    for (auto& v : group->ibDevs) {
      devices.emplace_back();
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
    for (auto& dev : devices) {
      ibv_wc wcs[4];
      int n = ibv_poll_cq(dev.cq, 4, wcs);
      CHECK(n >= 0);
      for (size_t i = 0; i != n; ++i) {
        ibv_wc& wc = wcs[i];
        if (wc.status) {
          fatal("rank %d Work completion with status %d (opcode %d, id %#x)\n", rank, wc.status, wc.opcode, wc.wr_id);
        } else {
          --dev.currentCqEntries;
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

      ibv_qp_ex* qp = dev.ib->qpex;

      // fmt::printf(
      //     "%d: rdma write %d bytes (%p -> %p, rkey %#x) (dev %d, i %d)\n", rank, bytes, localAddress,
      //     remoteAddress, rkey, &dev - devices.data(), i);

      bool efa = qp != nullptr;
      if (!efa) {
        qp = dev.ib->qpexs.at(i);
        CHECK(qp != nullptr);
      }

      ibv_wr_start(qp);
      qp->wr_id = (uint64_t)(void*)callback;
      qp->wr_flags = IBV_SEND_SIGNALED;
      ibv_wr_rdma_write(qp, rkey, (uintptr_t)remoteAddress);
      ibv_wr_set_sge_list(qp, 1, &sge);

      if (efa) {
        ibv_wr_set_ud_addr(qp, dev.ib->ahs.at(i), dev.ib->remoteAddresses.at(i).qpNum, 0x4242);
      }
      int error = ibv_wr_complete(qp);
      if (error) {
        log.error("ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
        CHECK(false);
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
    std::vector<uint32_t> localKeys;
    for (size_t i = 0; i != devices.size(); ++i) {
      localKeys.push_back(mr->mrs.at(i)->rkey);
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

  void writeStep(size_t deviceIndex, size_t dst, uint32_t stepValue) {
    Device& dev = devices[deviceIndex];
    uintptr_t dstOffset = group->getSharedOffset(&localProgress[rank].stepValue);
    localProgress[rank].stepValue = stepValue;
    writeData(
        dev, dst, &localProgress[rank].stepValue, commsMr->mrs.at(deviceIndex)->lkey,
        (void*)(remoteComms[dst].address + dstOffset), remoteComms[dst].keys[deviceIndex], sizeof(stepValue));
  }

  void writeCpuStep(size_t deviceIndex, size_t dst, uint32_t stepValue) {
    Device& dev = devices[deviceIndex];
    uintptr_t dstOffset = group->getSharedOffset(&localProgress[rank].cpuStepValue);
    localProgress[rank].cpuStepValue = stepValue;
    writeData(
        dev, dst, &localProgress[rank].cpuStepValue, commsMr->mrs.at(deviceIndex)->lkey,
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
    IntrusiveListLink<WorkBase> link;
  };
  struct Work : WorkBase {
    CpuThreadImpl& self;
    void* state = nullptr;
    bool done = false;
    void (Work::*stepPtr)();

    Work(CpuThreadImpl& self) : self(self) {}
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

    WorkBarrier(CpuThreadImpl& self, QueueEntryBarrier& params) : Work(self), params(params) {}

    void step() {
      const uint32_t stepValue = params.stepValue;
      const size_t size = self.size;
      const size_t rank = self.rank;
      ENTER
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        self.writeCpuStep((rank + i) % self.devices.size(), i, stepValue);
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (self.localProgress[i].cpuStepValue < stepValue) {
          YIELD
        }
      }
      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
      self.cpuThread->freelistBarrier.push(&params);
      IntrusiveList<WorkBase, &WorkBase::link>::erase(*this);
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

    WorkAllGather(CpuThreadImpl& self, QueueEntryAllGather& params) : Work(self), params(params) {}

    void step() {
      const uint32_t stepValue = params.stepValue;
      const size_t size = self.size;
      const size_t rank = self.rank;

      AllGather& allGather = *self.group->allGather;

      auto& sendRanks = allGather.sendRanks;
      auto& recvRanks = allGather.recvRanks;

      ENTER

      inputMr = self.regMrCuda(params.inputAddress, params.bytes);

      // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
      // Thus, we register each ranks output address independently, so they get different MRs.
      for (size_t i : recvRanks) {
        auto* mr = self.regMrCuda(params.outputAddress + params.pitch * i, params.bytes);
        self.outDyns[i].gatherAddress = params.outputAddress;
        for (size_t di = 0; di != self.devices.size(); ++di) {
          self.outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
        }
      }

      // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
      // until kernel entry, such that we don't receive data until we're ready.
      self.dynReady.resize(recvRanks.size());
      {
        size_t n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % self.devices.size();
          auto& dev = self.devices[di];
          self.writeData(
              dev, i, &self.outDyns[i], self.outDynsMr->mrs[di]->lkey,
              (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * rank), self.remoteComms[i].keys[di],
              sizeof(DynamicAddresses),
              self.makeCallback([&, i, di, stepValue, n]() { self.dynReady[n] = stepValue; }));
          ++n;
        }
      }

      // wait for kernel
      while (self.cpuIn[0] < stepValue) {
        YIELD
      }

      for (i = 0; i != recvRanks.size(); ++i) {
        while (self.dynReady[i] < stepValue) {
          YIELD
        }
      }

      self.sendStepValues[0] = stepValue;

      CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        if (true) {
          size_t i = recvRanks[index];
          size_t di = (rank + index) % self.devices.size();
          auto& dev = self.devices[di];
          self.writeStep(di, i, stepValue + index);
        }
        i = sendRanks[index];
        // wait for dyns
        while (self.localProgress[i].stepValue < stepValue + index) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)params.inputAddress;

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, inputMr, self.localDyns[i].gatherAddress + params.pitch * rank,
              self.localDyns[i].gatherKey, [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices.at(di);
                  self.writeData(
                      dev, i, &self.sendStepValues[0], self.sendStepValuesStorageMr->mrs.at(di)->lkey,
                      (void*)(self.remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (self.rank * 32 + di)),
                      self.remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                }
              });
        }
        while (liveSends >= self.devices.size() * 2) {
          YIELD
        }
      }

      while (true) {
        {
          bool stop = true;
          for (auto& dev : self.devices) {
            if (dev.currentCqEntries) {
              stop = false;
              break;
            }
          }
          if (stop) {
            break;
          }
        }
        YIELD
      }
      for (auto& dev : self.devices) {
        CHECK(dev.currentCqEntries == 0);
      }

      self.cpuOut[0] = stepValue;
      while (self.cpuIn[0] < stepValue + 1) {
        YIELD
      }

      self.cpuThread->freelistAllGather.push(&params);
      IntrusiveList<WorkBase, &WorkBase::link>::erase(*this);
      self.allocatorAllGather.deallocate(this);
      DONE
    }
  };

  // void barrier(QueueEntryBarrier& params) {
  //   uint32_t stepValue = params.stepValue;

  //   for (size_t i = 0; i != size; ++i) {
  //     if (i == rank) {
  //       continue;
  //     }
  //     writeCpuStep((rank + i) % devices.size(), i, stepValue);
  //   }
  //   for (size_t i = 0; i != size; ++i) {
  //     if (i == rank) {
  //       continue;
  //     }
  //     while (localProgress[i].cpuStepValue < stepValue) {
  //       poll();
  //     }
  //   }
  //   params.cpuDone->store(1);
  //   futexWakeAll(params.cpuDone);
  // }

  void all_gather(QueueEntryAllGather& params) {
    uint32_t stepValue = params.stepValue;

    AllGather& allGather = *group->allGather;

    auto& sendRanks = allGather.sendRanks;
    auto& recvRanks = allGather.recvRanks;

    auto* inputMr = regMrCuda(params.inputAddress, params.bytes);

    // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
    // Thus, we register each ranks output address independently, so they get different MRs.
    for (size_t i : recvRanks) {
      auto* mr = regMrCuda(params.outputAddress + params.pitch * i, params.bytes);
      outDyns[i].gatherAddress = params.outputAddress;
      for (size_t di = 0; di != devices.size(); ++di) {
        outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
      }
    }

    // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
    // until kernel entry, such that we don't receive data until we're ready.
    dynReady.resize(recvRanks.size());
    size_t n = 0;
    for (size_t i : recvRanks) {
      size_t di = (rank + n) % devices.size();
      auto& dev = devices[di];
      writeData(
          dev, i, &outDyns[i], outDynsMr->mrs[di]->lkey,
          (void*)(remoteComms[i].address + sizeof(DynamicAddresses) * rank), remoteComms[i].keys[di],
          sizeof(DynamicAddresses), makeCallback([&, i, di, stepValue, n]() { dynReady[n] = stepValue; }));
      ++n;
    }

    // wait for kernel
    while (cpuIn[0] < stepValue) {
      poll();
    }

    n = 0;
    for (size_t i : recvRanks) {
      while (dynReady[n] < stepValue) {
        poll();
      }
      ++n;
    }

    auto start = std::chrono::steady_clock::now();

    sendStepValues[0] = stepValue;

    CHECK(recvRanks.size() == sendRanks.size());

    size_t liveSends = 0;
    for (size_t index = 0; index != sendRanks.size(); ++index) {
      if (true) {
        size_t i = recvRanks[index];
        size_t di = (rank + index) % devices.size();
        auto& dev = devices[di];
        writeStep(di, i, stepValue + index);
      }
      size_t i = sendRanks[index];
      // wait for dyns
      while (localProgress[i].stepValue < stepValue + index) {
        poll();
      }
      size_t nbytes = params.bytes;

      if (nbytes > 0) {
        uintptr_t srcAddr = (uintptr_t)params.inputAddress;

        liveSends += devices.size();
        writeDataDistributed2(
            i, srcAddr, nbytes, inputMr, localDyns[i].gatherAddress + params.pitch * rank, localDyns[i].gatherKey,
            [this, &liveSends, i](size_t di, bool ordered, bool done) {
              if (done) {
                --liveSends;
              }
              if (ordered) {
                auto& dev = devices.at(di);
                writeData(
                    dev, i, &sendStepValues[0], sendStepValuesStorageMr->mrs.at(di)->lkey,
                    (void*)(remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (rank * 32 + di)),
                    remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
              }
            });
      }
      while (liveSends >= devices.size() * 2) {
        poll();
      }
    }

    while (true) {
      bool stop = true;
      for (auto& dev : devices) {
        while (dev.currentCqEntries) {
          poll();
          stop = false;
        }
      }
      if (stop) {
        break;
      }
    }
    for (auto& dev : devices) {
      CHECK(dev.currentCqEntries == 0);
    }

    cpuOut[0] = stepValue;
    while (cpuIn[0] < stepValue + 1) {
      poll();
    }
  }

  void reduce_scatter(QueueEntryReduceScatter& params) {
    uint32_t stepValue = params.stepValue;

    ReduceScatter& reduceScatter = *group->reduceScatter;

    auto& sendRanks = reduceScatter.sendRanks;
    auto& recvRanks = reduceScatter.recvRanks;

    uintptr_t recvAddress = params.sd->recvBuffer.cudaPointer;
    CHECK(params.sd->recvBuffer.bytes >= params.pitch * recvRanks.size());

    uintptr_t sendAddress = params.sd->sendBuffer.cudaPointer;
    CHECK(params.sd->sendBuffer.bytes >= params.pitch * sendRanks.size());

    auto* sendMr = regMrCuda(sendAddress, params.pitch * sendRanks.size());
    auto* recvMr = regMrCuda(recvAddress, params.pitch * recvRanks.size());

    for (size_t i : recvRanks) {
      outDyns[i].gatherAddress = recvAddress;
      for (size_t di = 0; di != devices.size(); ++di) {
        outDyns[i].gatherKey[di] = recvMr->mrs[di]->rkey;
      }
    }

    // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
    // until kernel entry, such that we don't receive data until we're ready.
    dynReady.resize(recvRanks.size());
    size_t n = 0;
    for (size_t i : recvRanks) {
      size_t di = (rank + n) % devices.size();
      auto& dev = devices[di];
      writeData(
          dev, i, &outDyns[i], outDynsMr->mrs[di]->lkey,
          (void*)(remoteComms[i].address + sizeof(DynamicAddresses) * rank), remoteComms[i].keys[di],
          sizeof(DynamicAddresses), makeCallback([&, i, di, stepValue, n]() { dynReady[n] = stepValue; }));
      ++n;
    }

    sendStepValues[0] = stepValue;

    n = 0;
    for (size_t i : recvRanks) {
      while (dynReady[n] < stepValue) {
        poll();
      }
      ++n;
    }

    // wait for kernel
    while (cpuIn[0] < stepValue) {
      poll();
    }

    CHECK(recvRanks.size() == sendRanks.size());

    size_t liveSends = 0;
    for (size_t index = 0; index != sendRanks.size(); ++index) {

      // wait for send ready
      while (cpuIn[1] < stepValue + index + 1) {
        poll();
      }

      {
        size_t i = recvRanks[index];
        size_t di = (rank + index) % devices.size();
        auto& dev = devices[di];
        writeStep(di, i, stepValue + index);
      }

      size_t i = sendRanks[index];
      // wait for dyns
      while (localProgress[i].stepValue < stepValue + index) {
        poll();
      }
      size_t nbytes = params.bytes;

      if (nbytes > 0) {
        uintptr_t srcAddr = (uintptr_t)sendAddress + params.pitch * index;

        liveSends += devices.size();
        writeDataDistributed2(
            i, srcAddr, nbytes, sendMr, localDyns[i].gatherAddress + index * params.pitch, localDyns[i].gatherKey,
            [this, &liveSends, i](size_t di, bool ordered, bool done) {
              if (done) {
                --liveSends;
              }
              if (ordered) {
                auto& dev = devices.at(di);
                writeData(
                    dev, i, &sendStepValues[0], sendStepValuesStorageMr->mrs.at(di)->lkey,
                    (void*)(remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (rank * 32 + di)),
                    remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
              }
            });
      }
      while (liveSends >= devices.size() * 2) {
        poll();
      }
    }

    while (true) {
      bool stop = true;
      for (auto& dev : devices) {
        while (dev.currentCqEntries) {
          poll();
          stop = false;
        }
      }
      if (stop) {
        break;
      }
    }
    for (auto& dev : devices) {
      CHECK(dev.currentCqEntries == 0);
    }

    cpuOut[0] = stepValue;
    while (cpuIn[0] < stepValue + 1) {
      poll();
    }
  }

  template<typename T>
  struct WorkAllocator {
    PoolAllocatorReused<T> allocator;
    IntrusiveList<WorkBase, &WorkBase::link> free;
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

  void entry() {
    try {

      runTests();

      cpuThread->ready = true;

      std::vector<IntrusiveList<WorkBase, &WorkBase::link>> works;

      auto enqueue = [&](auto& allocator, auto& params) {
        auto* work = allocator.allocate(*this, params);
        work->stepPtr = (void(Work::*)()) & std::remove_pointer_t<decltype(work)>::step;
        if (params.sd->cpuThreadIndex == -1) {
          log.info("new work index %d\n", works.size());
          params.sd->cpuThreadIndex = works.size();
          works.emplace_back();
        }
        works[params.sd->cpuThreadIndex].push_back(*work);
      };

      while (!cpuThread->terminate.load(std::memory_order_relaxed)) {
        if (cpuThread->queueSize.load(std::memory_order_relaxed) == 0) {
          bool allEmpty = true;
          for (auto& list : works) {
            if (!list.empty()) {
              allEmpty = false;
              break;
            }
          }
          if (allEmpty) {
            int sum = 0;
            for (auto& dev : devices) {
              sum += dev.currentCqEntries;
            }
            if (sum == 0) {
              futexWait(&cpuThread->queueSize, 0, std::chrono::seconds(10));
              continue;
            }
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
              // QueueEntryAllGather& params = (QueueEntryAllGather&)queueEntry;
              // all_gather(params);
              // cpuThread->freelistAllGather.push(&params);
            } else if (queueEntry.task == taskReduceScatter) {
              CHECK(false);
              QueueEntryReduceScatter& params = (QueueEntryReduceScatter&)queueEntry;
              // reduce_scatter(params);
              // cpuThread->freelistReduceScatter.push(&params);
            } else if (queueEntry.task == taskTerminate) {
              cpuThread->freelistTerminate.push(&queueEntry);
            } else {
              throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
            }
          }
          for (int i2 = 0; i2 != 4; ++i2) {
            poll();

            for (auto& list : works) {
              while (!list.empty()) {
                Work* w = (Work*)&list.front();
                (w->*w->stepPtr)();
                if (!w->done) {
                  break;
                }
              }
            }
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
