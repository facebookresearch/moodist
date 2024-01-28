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

extern bool profilingEnabled;

struct CpuThreadImpl;

namespace {

struct QuitCpuThread {};

struct MemoryRegistration {
  std::array<ibv_mr*, 32> mrs{};
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
  bool efa;
  bool ordered;
};

struct Callback {
  IntrusiveListLink<Callback> link;
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
  Callback* callback = nullptr;
  CallbackWrapper() = default;
  CallbackWrapper(std::nullptr_t) {}
  CallbackWrapper(Callback* callback) : callback(callback) {
    if (callback) {
      ++callback->refcount;
    }
  }
  ~CallbackWrapper() {
    if (callback) {
      callback->decref();
    }
  }
  operator Callback*() {
    return callback;
  }
  Callback* operator->() {
    return callback;
  }
  Callback& operator*() {
    return *callback;
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

struct Bfloat16 {
  uint16_t value;
  Bfloat16 operator+(Bfloat16 n) const {
    fatal("cpu bfloat16 is not currently supported");
  }
  bool operator<(Bfloat16 n) const {
    fatal("cpu bfloat16 is not currently supported");
  }
};

template<>
struct CpuTypes<Dtype::bfloat16> {
  static constexpr size_t bytes = 2;
  using T = Bfloat16;
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
    return std::max(a, b);
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
  AllocatedCpuBuffer buffer;
  MemoryRegistration* mr = nullptr;
  Vector<TemporaryBufferHandle>* container = nullptr;
  TemporaryBufferHandle() = default;
  TemporaryBufferHandle(AllocatedCpuBuffer buffer, MemoryRegistration* mr, Vector<TemporaryBufferHandle>* container)
      : buffer(std::move(buffer)), mr(mr), container(container) {}
  TemporaryBufferHandle(TemporaryBufferHandle&& n) {
    *this = std::move(n);
  }
  TemporaryBufferHandle& operator=(TemporaryBufferHandle&& n) {
    std::swap(buffer, n.buffer);
    std::swap(mr, n.mr);
    std::swap(container, n.container);
    return *this;
  }
  ~TemporaryBufferHandle() {
    if (container) {
      auto* container = std::exchange(this->container, nullptr);
      container->push_back(std::move(*this));
    }
  }
  AllocatedCpuBuffer& operator*() {
    return buffer;
  }
  AllocatedCpuBuffer* operator->() {
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
  Progress* localProgress2 = group->localProgress2;

  AllocatedArray internalLocalProgressBuffer = group->allocateArrayHost(sizeof(uint32_t) * size, Group::maxConcurrency);
  MemoryRegistration* internalLocalProgressBufferMr =
      regMr((uintptr_t)internalLocalProgressBuffer.buffer.cpuPointer, internalLocalProgressBuffer.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteInternalLocalProgressBuffer =
      distributeAddressAndKeys((uintptr_t)internalLocalProgressBuffer.buffer.cpuPointer, internalLocalProgressBufferMr);

  MemoryRegistration* cudaStepValueMr =
      regMrCuda(group->cudaStepValue.buffer.cudaPointer, group->cudaStepValue.buffer.bytes);

  std::vector<RemoteAddressAndKey> remoteComms = distributeAddressAndKeys((uintptr_t)group->mySharedMem, commsMr);
  std::vector<RemoteAddressAndKey> remoteCudaStepValue =
      distributeAddressAndKeys(group->cudaStepValue.buffer.cudaPointer, cudaStepValueMr);

  MemoryRegistration* cudaCommsDeviceDataSentMr =
      regMrCuda(group->cudaCommsDeviceDataSent.buffer.cudaPointer, group->cudaCommsDeviceDataSent.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent =
      distributeAddressAndKeys(group->cudaCommsDeviceDataSent.buffer.cudaPointer, cudaCommsDeviceDataSentMr);

  MemoryRegistration* cpuCommsDeviceDataSentMr =
      regMr((uintptr_t)group->cpuCommsDeviceDataSent.buffer.cpuPointer, group->cpuCommsDeviceDataSent.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteCpuCommsDeviceDataSent =
      distributeAddressAndKeys((uintptr_t)group->cpuCommsDeviceDataSent.buffer.cpuPointer, cpuCommsDeviceDataSentMr);

  MemoryRegistration* cudaCommsDeviceDataSent2Mr =
      regMrCuda(group->cudaCommsDeviceDataSent2.buffer.cudaPointer, group->cudaCommsDeviceDataSent2.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent2 =
      distributeAddressAndKeys(group->cudaCommsDeviceDataSent2.buffer.cudaPointer, cudaCommsDeviceDataSent2Mr);

  MemoryRegistration* cpuCommsDeviceDataSent2Mr =
      regMr((uintptr_t)group->cpuCommsDeviceDataSent2.buffer.cpuPointer, group->cpuCommsDeviceDataSent2.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteCpuCommsDeviceDataSent2 =
      distributeAddressAndKeys((uintptr_t)group->cpuCommsDeviceDataSent2.buffer.cpuPointer, cpuCommsDeviceDataSent2Mr);

  AllocatedArray sendStepValuesStorage = group->allocateArrayHost(sizeof(uint32_t), Group::maxConcurrency);
  MemoryRegistration* sendStepValuesStorageMr =
      regMr((uintptr_t)sendStepValuesStorage.buffer.cpuPointer, sendStepValuesStorage.buffer.bytes);
  uint32_t* sendStepValues = (uint32_t*)sendStepValuesStorage.buffer.cpuPointer;

  AllocatedArray sendStepValues2Storage = group->allocateArrayHost(sizeof(uint32_t), 32 * 32 * Group::maxConcurrency);
  MemoryRegistration* sendStepValues2StorageMr =
      regMr((uintptr_t)sendStepValues2Storage.buffer.cpuPointer, sendStepValues2Storage.buffer.bytes);
  uint32_t* sendStepValues2 = (uint32_t*)sendStepValues2Storage.buffer.cpuPointer;
  size_t sendStepValues2Counter = 0;

  AllocatedArray writeStepStorage = group->allocateArrayHost(sizeof(uint32_t) * size, Group::maxConcurrency);
  MemoryRegistration* writeStepStorageMr =
      regMr((uintptr_t)writeStepStorage.buffer.cpuPointer, writeStepStorage.buffer.bytes);
  Vector<uint8_t> writeStepBusy{size * Group::maxConcurrency};
  Vector<Vector<std::pair<size_t, uint32_t>>> writeStepQueue{size * Group::maxConcurrency};

  AllocatedArray outDynsBuffer = group->allocateArrayHost(sizeof(DynamicAddresses) * size, Group::maxConcurrency);
  MemoryRegistration* outDynsMr = regMr((uintptr_t)outDynsBuffer.buffer.cpuPointer, outDynsBuffer.buffer.bytes);

  Vector<uint8_t> dynReadyVector{size * Group::maxConcurrency};
  Vector<MemoryRegistration*> outDynsMrVector{size * Group::maxConcurrency};

  Vector<TemporaryBufferHandle> vmreadBuffers;
  Vector<TemporaryBufferHandle> sendRecvBuffers;
  Vector<TemporaryBufferHandle> allGatherInputBuffers;
  Vector<TemporaryBufferHandle> allGatherOutputBuffers;
  Vector<TemporaryBufferHandle> broadcastBuffers;

  TemporaryBufferHandle allocateTemporaryBuffer(size_t bytes, Vector<TemporaryBufferHandle>& container) {
    for (size_t i = container.size(); i;) {
      --i;
      auto& v = container[i];
      if (v->bytes >= bytes) {
        auto h = std::move(v);
        container.erase(container.begin() + i);
        h.container = &container;
        return h;
      }
    }
    auto buffer = group->allocateCpu(bytes);
    auto* mr = regMr((uintptr_t)buffer.cpuPointer, buffer.bytes);
    return TemporaryBufferHandle(std::move(buffer), mr, &container);
    //  todo: this isn't good enough, for now never free memory.
    //        the problem is we need to de-register when we free memory,
    //        as we don't have a buffer id like with cuda memory,
    //        but it's tricky because of the extra regs we do due to
    //        efa 2gb bug. this code could just iterate over all regs
    //        and free the ones in the region when we reallocate,
    //        but for now let's just do the above (never free).
    // if (container.empty()) {
    //   auto buffer = group->allocateCpu(bytes);
    //   auto* mr = regMr((uintptr_t)buffer.cpuPointer, buffer.bytes);
    //   return TemporaryBufferHandle(std::move(buffer), mr, &container);
    // }
    // auto handle = std::move(container.back());
    // container.pop_back();
    // if (handle->bytes < bytes) {
    //   CHECK(false);
    //   auto i = mrMap.find((uintptr_t)handle->cpuPointer);
    //   if (i != mrMap.end()) {
    //     mrMap.erase(i);
    //   }
    //   for (size_t i = 0; i != devices.size(); ++i) {
    //     if (handle.mr->mrs[i]) {
    //       for (auto& v : localMrs) {
    //         if (&*v == handle.mr->mrs[i]) {
    //           localMrs.erase(localMrs.begin() + (&v - localMrs.data()));
    //           break;
    //         }
    //       }
    //       //ibv_dereg_mr(handle.mr->mrs[i]);
    //     }
    //   }
    //   handle.buffer = group->allocateCpu(bytes);
    //   handle.mr = regMr((uintptr_t)handle->cpuPointer, handle->bytes);
    // }
    // handle.container = &container;
    // return std::move(handle);
  }

  SimpleVector<Device> initDevices() {
    SimpleVector<Device> devices;
    for (auto& v : group->ibDevs) {
      devices.resize(devices.size() + 1);
      auto& dev = devices.back();
      dev.ib = &*v;
      dev.protectionDomain = v->protectionDomain;
      dev.cq = v->cq;
      dev.efa = dev.ib->qpex != nullptr;
      dev.ordered = !dev.efa;
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

  template<typename Callback>
  void writeDataDistributed3(
      size_t dst, uintptr_t srcAddr, size_t len, MemoryRegistration* mr, uint64_t dstAddr,
      const std::array<uint32_t, 32>& key, Callback&& callback) {
    CHECK(len > 0);
    size_t offset = 0;
    const size_t chunks = devices.size();
    const size_t chunkSize = len / chunks / 128u * 128u;
    for (size_t i = 0; i != chunks; ++i) {
      size_t n = i == chunks - 1 ? len - offset : chunkSize;
      auto& dev = devices[i];
      std::chrono::system_clock::time_point sendStart;
      if (profilingEnabled) {
        sendStart = std::chrono::system_clock::now();
      }
      if (n != 0) {
        bool efa = dev.ib->qpex != nullptr;
        writeData(
            dev, dst, (void*)(srcAddr + offset), mr->mrs[i]->lkey, (void*)(dstAddr + offset), key[i], n,
            makeCallback([callback, i, efa, sendStart]() { callback(i, efa, true, sendStart); }));
        if (!efa) {
          callback(i, true, false, sendStart);
        }
      } else {
        callback(i, true, true, sendStart);
      }
      offset += n;
    }
  }

  void writeStep(size_t deviceIndex, size_t dst, uint32_t stepValue, size_t concurrencyIndex) {
    Device& dev = devices[deviceIndex];
    uintptr_t dstOffset = group->getSharedOffset(&localProgress[size * concurrencyIndex + rank].stepValue);
    // localProgress[size * concurrencyIndex + rank].stepValue = stepValue;
    if (writeStepBusy[size * concurrencyIndex + dst]) {
      writeStepQueue[size * concurrencyIndex + dst].emplace_back(deviceIndex, stepValue);
      return;
    }
    writeStepBusy[size * concurrencyIndex + dst] = 1;
    uint32_t& val = ((uint32_t*)writeStepStorage.cpu(concurrencyIndex))[dst];
    val = stepValue;
    writeData(
        dev, dst, &val, writeStepStorageMr->mrs[deviceIndex]->lkey, (void*)(remoteComms[dst].address + dstOffset),
        remoteComms[dst].keys[deviceIndex], sizeof(stepValue), makeCallback([this, concurrencyIndex, dst] {
          writeStepBusy[size * concurrencyIndex + dst] = 0;
          auto& q = writeStepQueue[size * concurrencyIndex + dst];
          if (!q.empty()) {
            auto [deviceIndex, stepValue] = q.front();
            q.pop_front();
            writeStep(deviceIndex, dst, stepValue, concurrencyIndex);
          }
        }));
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

  size_t opCount = 0;

  struct TraceEvent {
    std::string name;
    std::chrono::system_clock::time_point begin;
    std::chrono::system_clock::time_point end;
    int threadId = -1;
  };
  std::vector<TraceEvent> traceEvents;

  std::chrono::system_clock::time_point beginning = std::chrono::system_clock::now();

  std::string currentTraceName = "";
  std::chrono::system_clock::time_point currentTraceBegin = beginning;
  int threadId = gettid();

  template<typename... Args>
  void trace(const char* fmt, Args&&... args) {
    // return;
    //  fmt::printf("trace enter '%s'\n", name);
    //   if (opCount < 1000 || opCount >= 1200) {
    if (!profilingEnabled) {
      if (!traceEvents.empty()) {
        std::string fn = fmt::sprintf("moodist-trace-cpu-%d.json", rank);
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
              std::chrono::duration_cast<std::chrono::microseconds>(e.end - e.begin).count(), e.threadId);
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
      e.threadId = threadId;
    }
    currentTraceBegin = now;
    currentTraceName = fmt::sprintf(fmt, std::forward<Args>(args)...);
  }
  // #define trace(name)

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
  if (self.numActiveWorks != 1 || self.cpuThread->queueSize.load(std::memory_order_relaxed) != 0) {                    \
    this->state = &&CAT(s, __LINE__);                                                                                  \
    return;                                                                                                            \
    CAT(s, __LINE__) :;                                                                                                \
  } else {                                                                                                             \
    self.poll();                                                                                                       \
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
      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
      self.cpuThread->freelistBarrier.push(&params);
      self.allocatorBarrier.deallocate(this);
      DONE
    }
  };

  struct WorkInternalBarrier : Work {
    QueueEntryBarrier& params;
    size_t i;

    WorkInternalBarrier(CpuThreadImpl& self, QueueEntryBarrier& params) : Work(self, params), params(params) {}

    uint32_t* internalLocalProgress() {
      return (uint32_t*)self.internalLocalProgressBuffer.buffer.cpuPointer;
    }

    void step() {
      ENTER
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }

        size_t deviceIndex = (rank + i) % self.devices.size();
        Device& dev = self.devices[deviceIndex];
        self.sendStepValues[concurrencyIndex] = stepValue;
        self.writeData(
            dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[deviceIndex]->lkey,
            (void*)(self.remoteInternalLocalProgressBuffer[i].address + sizeof(uint32_t) * (size * concurrencyIndex + rank)),
            self.remoteInternalLocalProgressBuffer[i].keys[deviceIndex], sizeof(stepValue));
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (internalLocalProgress()[size * concurrencyIndex + i] < stepValue) {
          YIELD
        }
      }
      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
      self.cpuThread->freelistBarrier.push(&params);
      self.allocatorInternalBarrier.deallocate(this);
      DONE
    }
  };

  struct WorkAllGather : Work {
    QueueEntryAllGather& params;
    size_t i;
    MemoryRegistration* inputMr;
    size_t liveSends;
    size_t index;

    const AllGather& allGather = *self.group->allGather;
    const Vector<size_t>& sendRanks = allGather.sendRanks;
    const Vector<size_t>& recvRanks = allGather.recvRanks;

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
        {
          size_t n = 0;
          for (size_t i : recvRanks) {
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];
            self.writeData(
                dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                self.remoteComms[i].keys[di], sizeof(DynamicAddresses),
                self.makeCallback([this, n]() { self.dynReadyVector[size * concurrencyIndex + n] = 1; }));
            ++n;
          }
        }
      }

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }

      self.sendStepValues[concurrencyIndex] = stepValue;

      CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        while (self.dynReadyVector[size * concurrencyIndex + index] == 0) {
          YIELD
        }
        self.dynReadyVector[size * concurrencyIndex + index] = 0;
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

  struct WorkReduceScatterRing : Work {
    QueueEntryReduceScatter& params;
    size_t i;
    // size_t neighbor;
    size_t di;
    MemoryRegistration* sendMr;
    size_t index;
    size_t index2;
    uintptr_t sendAddress;

    const ReduceScatter& reduceScatter = *self.group->reduceScatter;
    const Vector<std::pair<size_t, size_t>>& ringRecvs = reduceScatter.ringRecvs;
    const Vector<std::pair<size_t, size_t>>& ringSends = reduceScatter.ringSends;

    uint32_t* sendStepValue;
    bool kernelHasEntered = false;

    static const size_t numChunks = 8;
    const size_t chunkSize = params.bytes < (size_t)65536 * 96
                                 ? params.bytes
                                 : std::max((params.bytes + numChunks - 1) / numChunks, (size_t)65536 * 72);
    const size_t numParallel = std::max(std::min((params.bytes + chunkSize - 1) / chunkSize / 2, (size_t)2), (size_t)1);

    struct SendState {
      size_t sendIndex;
      size_t chunkIndex;
      size_t liveSends;
    };
    const size_t nDevices = self.devices.size();
    std::array<SendState, 32> sendStates;

    WorkReduceScatterRing(CpuThreadImpl& self, QueueEntryReduceScatter& params) : Work(self, params), params(params) {}

    void callback(
        bool isLastChunk, size_t di, size_t source, size_t chunkIndex, size_t neighbor, bool ordered, bool done,
        std::chrono::system_clock::time_point sendStart) {
      if (done) {
        --sendStates[di].liveSends;

        if (profilingEnabled) {
          TraceEvent e;
          e.name = fmt::sprintf("%d -> %d di %d chunk %d", source, neighbor, di, chunkIndex);
          e.begin = sendStart;
          e.end = std::chrono::system_clock::now();
          e.threadId = di * numChunks + chunkIndex;
          self.traceEvents.push_back(std::move(e));
        }
      }
      if (ordered) {
        auto& dev = self.devices[di];
        if (isLastChunk) {
          self.writeData(
              dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
              (void*)(self.remoteCudaCommsDeviceDataSent[neighbor].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * source + di)),
              self.remoteCudaCommsDeviceDataSent[neighbor].keys[di], sizeof(stepValue));
        } else {
          self.writeData(
              dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
              (void*)(self.remoteCudaCommsDeviceDataSent2[neighbor].address + sizeof(uint32_t) * (32 * 32 * size * concurrencyIndex + 32 * 32 * source + 32 * di + chunkIndex)),
              self.remoteCudaCommsDeviceDataSent2[neighbor].keys[di], sizeof(stepValue));
        }
      }
    }

    void step() {
      ENTER

      self.trace("reduce-scatter ring %#x", stepValue);

      sendStepValue = &self.sendStepValues2[32 * 32 * concurrencyIndex + 32 * ((self.sendStepValues2Counter++) % 32)];
      *sendStepValue = stepValue;
      for (size_t i = 0; i != numChunks; ++i) {
        sendStepValue[i] = stepValue + i;
      }

      {
        uintptr_t recvAddress = params.sd->recvBuffer.cudaPointer;
        CHECK(params.sd->recvBuffer.bytes >= params.pitch * ringRecvs.size());

        sendAddress = params.sd->sendBuffer.cudaPointer;
        CHECK(params.sd->sendBuffer.bytes >= params.pitch * ringSends.size());

        CHECK(sendAddress && recvAddress);

        sendMr = self.regMrCuda(sendAddress, params.pitch * ringSends.size());

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
        // Thus, we register each ranks output address independently, so they get different MRs.
        size_t n = 0;
        for (auto [i, neighbor] : ringRecvs) {
          auto* mr = self.regMrCuda(recvAddress + params.pitch * n, params.bytes);
          outDyns[i].gatherAddress = recvAddress + params.pitch * n;
          outDyns[i].gatherBytes = params.bytes;
          outDyns[i].stepValue = stepValue;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
          }
          ++n;
        }

        // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
        // until kernel entry, such that we don't receive data until we're ready.
        // todo this can all be sent in one write
        {
          size_t n = 0;
          for (auto [i, neighbor] : ringRecvs) {
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];

            auto offset = self.group->getSharedOffset(&self.group->localDyns[size * concurrencyIndex + n]);

            self.writeData(
                dev, neighbor, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[neighbor].address + offset), self.remoteComms[neighbor].keys[di],
                sizeof(DynamicAddresses),
                dev.ordered ? nullptr : self.makeCallback([this, di, n, neighbor = neighbor]() {
                  if (kernelHasEntered) {
                    self.dynReadyVector[size * concurrencyIndex + n] = 1;
                    // log.info("dyn index %d sent to %d\n", n, neighbor);
                    auto& dev = self.devices[di];
                    self.writeData(dev, neighbor,
                            sendStepValue,
                            self.sendStepValues2StorageMr->mrs[di]->lkey,
                            (void*)(self.remoteComms[neighbor].address + self.group->getSharedOffset(&self.localProgress2[size * concurrencyIndex + n].stepValue)),
                            self.remoteComms[neighbor].keys[di],
                            sizeof(stepValue));
                  } else {
                    self.dynReadyVector[size * concurrencyIndex + n] = 2;
                  }
                }));
            ++n;
          }
        }
      }

      // log.info("wait for enter kernel %d\n", stepValue);

      self.trace("kernel-entry");

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }
      kernelHasEntered = true;
      for (index = 0; index != ringSends.size(); ++index) {
        // self.trace("dyn-ready %d", index);
        size_t di = (rank + index) % self.devices.size();
        auto& dev = self.devices[di];
        if (self.dynReadyVector[size * concurrencyIndex + index] == 2 || dev.ordered) {
          auto [i, neighbor] = ringRecvs[index];
          self.writeData(dev, neighbor,
              sendStepValue,
              self.sendStepValues2StorageMr->mrs[di]->lkey,
              (void*)(self.remoteComms[neighbor].address + self.group->getSharedOffset(&self.localProgress2[size * concurrencyIndex + index].stepValue)),
              self.remoteComms[neighbor].keys[di],
              sizeof(stepValue));
          self.dynReadyVector[size * concurrencyIndex + index] = 3;
        }
      }

      // log.info("enter kernel %d ok\n", stepValue);

      CHECK(ringRecvs.size() == ringSends.size());

      for (size_t i = 0; i != nDevices; ++i) {
        sendStates[i].sendIndex = 0;
        sendStates[i].chunkIndex = 0;
        sendStates[i].liveSends = 0;
      }

      self.trace("send loop");

      while (true) {
        {
          size_t nDone = 0;
          for (size_t i = 0; i != nDevices; ++i) {
            auto& state = sendStates[i];
            if (state.liveSends >= numParallel) {
              continue;
            }
            if (state.sendIndex == ringSends.size()) {
              if (state.liveSends) {
                continue;
              }
              ++nDone;
              continue;
            }
            // wait for dyns
            if (self.localProgress2[size * concurrencyIndex + state.sendIndex].stepValue < stepValue) {
              continue;
            }
            // wait for send ready
            if (cpuIn[1] < stepValue + state.sendIndex + 1) {
              continue;
            }
            size_t index = state.sendIndex;
            size_t chunkIndex = state.chunkIndex;
            size_t source = std::get<0>(ringSends[index]);
            size_t neighbor = std::get<1>(ringSends[index]);

            size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

            const size_t devChunkSize = currentChunkSize / nDevices / 1024u * 1024u;
            size_t offset = devChunkSize * i;
            size_t n = i == nDevices - 1 ? currentChunkSize - offset : devChunkSize;
            bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
            if (n == 0) {
              std::chrono::system_clock::time_point sendStart;
              if (profilingEnabled) {
                sendStart = std::chrono::system_clock::now();
              }
              ++state.liveSends;
              callback(isLastChunk, i, source, chunkIndex, neighbor, true, true, sendStart);
            } else {
              uintptr_t srcAddr = sendAddress + params.pitch * index;
              auto* srcMr = sendMr;

              auto& dev = self.devices[i];
              std::chrono::system_clock::time_point sendStart;
              if (profilingEnabled) {
                sendStart = std::chrono::system_clock::now();
              }
              ++state.liveSends;
              uintptr_t dstAddr = self.localDyns[size * concurrencyIndex + index].gatherAddress;
              auto key = self.localDyns[size * concurrencyIndex + index].gatherKey;
              bool efa = dev.ib->qpex != nullptr;
              self.writeData(
                  dev, neighbor, (void*)(srcAddr + chunkSize * chunkIndex + offset), srcMr->mrs[i]->lkey,
                  (void*)(dstAddr + chunkSize * chunkIndex + offset), key[i], n,
                  self.makeCallback([me = this, isLastChunk, i, efa, sendStart, source, chunkIndex, neighbor]() {
                    me->callback(isLastChunk, i, source, chunkIndex, neighbor, efa, true, sendStart);
                  }));
              if (!efa) {
                callback(isLastChunk, i, source, chunkIndex, neighbor, true, false, sendStart);
              }
            }

            if (isLastChunk) {
              state.chunkIndex = 0;
              ++state.sendIndex;
            } else {
              ++state.chunkIndex;
              CHECK(state.chunkIndex < numChunks);
            }
          }
          if (nDone == nDevices) {
            break;
          }
        }
        YIELD
      }

      for (size_t i = 0; i != nDevices; ++i) {
        CHECK(sendStates[i].liveSends == 0);
        CHECK(sendStates[i].sendIndex == ringSends.size());
      }

      // log.info("wait for exit kernel %d\n", stepValue);

      self.trace("kernel-exit");

      cpuOut[0] = stepValue;
      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      // Clean up dyns
      for (index = 0; index != ringSends.size(); ++index) {
        CHECK(self.dynReadyVector[size * concurrencyIndex + index] != 0);
        CHECK(self.dynReadyVector[size * concurrencyIndex + index] != 2);
        self.dynReadyVector[size * concurrencyIndex + index] = 0;
      }

      // log.info("exit kernel %d ok\n", stepValue);

      self.trace("");

      self.cpuThread->freelistReduceScatter.push(&params);
      self.allocatorReduceScatterRing.deallocate(this);
      DONE
    }
  };

  struct WorkAllGatherRing : Work {
    QueueEntryAllGather& params;
    size_t i;
    size_t neighbor;
    size_t di;
    MemoryRegistration* inputMr;
    // size_t liveSends;
    size_t index;
    size_t index2;

    const AllGather& allGather = *self.group->allGather;
    const Vector<std::pair<size_t, size_t>>& ringRecvs = allGather.ringRecvs;
    const Vector<std::pair<size_t, size_t>>& ringSends = allGather.ringSends;

    uint32_t* sendStepValue;
    bool kernelHasEntered = false;

    static const size_t numChunks = 8;
    const size_t chunkSize = params.bytes < (size_t)65536 * 96
                                 ? params.bytes
                                 : std::max((params.bytes + numChunks - 1) / numChunks, (size_t)65536 * 72);
    const size_t numParallel = std::max(std::min((params.bytes + chunkSize - 1) / chunkSize / 2, (size_t)2), (size_t)1);

    struct SendState {
      size_t sendIndex;
      size_t chunkIndex;
      size_t liveSends;
    };
    const size_t nDevices = self.devices.size();
    std::array<SendState, 32> sendStates;

    WorkAllGatherRing(CpuThreadImpl& self, QueueEntryAllGather& params) : Work(self, params), params(params) {}

    void callback(
        bool isLastChunk, size_t di, size_t source, size_t chunkIndex, size_t neighbor, bool ordered, bool done,
        std::chrono::system_clock::time_point sendStart) {
      if (done) {
        --sendStates[di].liveSends;

        if (profilingEnabled) {
          TraceEvent e;
          e.name = fmt::sprintf("%d -> %d di %d chunk %d", source, neighbor, di, chunkIndex);
          e.begin = sendStart;
          e.end = std::chrono::system_clock::now();
          e.threadId = di * numChunks + chunkIndex;
          self.traceEvents.push_back(std::move(e));
        }
      }
      if (ordered) {
        auto& dev = self.devices[di];
        self.writeData(
            dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
            (void*)(self.remoteCpuCommsDeviceDataSent2[neighbor].address + sizeof(uint32_t) * (32 * 32 * size * concurrencyIndex + 32 * 32 * source + 32 * di + chunkIndex)),
            self.remoteCpuCommsDeviceDataSent2[neighbor].keys[di], sizeof(stepValue));
        if (isLastChunk) {
          self.writeData(
              dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
              (void*)(self.remoteCudaCommsDeviceDataSent[neighbor].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * source + di)),
              self.remoteCudaCommsDeviceDataSent[neighbor].keys[di], sizeof(stepValue));
        } else {
          self.writeData(
              dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
              (void*)(self.remoteCudaCommsDeviceDataSent2[neighbor].address + sizeof(uint32_t) * (32 * 32 * size * concurrencyIndex + 32 * 32 * source + 32 * di + chunkIndex)),
              self.remoteCudaCommsDeviceDataSent2[neighbor].keys[di], sizeof(stepValue));
        }
      }
    }

    void step() {
      ENTER

      self.trace("ring %#x", stepValue);

      sendStepValue = &self.sendStepValues2[32 * 32 * concurrencyIndex + 32 * ((self.sendStepValues2Counter++) % 32)];
      *sendStepValue = stepValue;
      for (size_t i = 0; i != numChunks; ++i) {
        sendStepValue[i] = stepValue + i;
      }

      {
        inputMr = self.regMrCuda(params.inputAddress, params.bytes);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
        // Thus, we register each ranks output address independently, so they get different MRs.
        for (auto [i, neighbor] : ringRecvs) {
          auto* mr = self.regMrCuda(params.outputAddress + params.pitch * i, params.bytes);
          self.outDynsMrVector[size * concurrencyIndex + i] = mr;
          outDyns[i].gatherAddress = params.outputAddress + params.pitch * i;
          outDyns[i].gatherBytes = params.bytes;
          outDyns[i].stepValue = stepValue;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
          }

          // log.info("outDyns[%d] address is %#x\n", i, outDyns[i].gatherAddress);
        }

        // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
        // until kernel entry, such that we don't receive data until we're ready.
        {
          size_t n = 0;
          for (auto [i, neighbor] : ringRecvs) {
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];

            auto offset = self.group->getSharedOffset(&self.group->localDyns[size * concurrencyIndex + n]);

            CHECK(self.dynReadyVector[size * concurrencyIndex + n] == 0);

            self.writeData(
                dev, neighbor, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[neighbor].address + offset), self.remoteComms[neighbor].keys[di],
                sizeof(DynamicAddresses),
                dev.ordered ? nullptr : self.makeCallback([this, di, n, neighbor = neighbor]() {
                  if (kernelHasEntered) {
                    self.dynReadyVector[size * concurrencyIndex + n] = 1;
                    // log.info("dyn index %d sent to %d\n", n, neighbor);
                    auto& dev = self.devices[di];
                    self.writeData(dev, neighbor,
                            sendStepValue,
                            self.sendStepValues2StorageMr->mrs[di]->lkey,
                            (void*)(self.remoteComms[neighbor].address + self.group->getSharedOffset(&self.localProgress2[size * concurrencyIndex + n].stepValue)),
                            self.remoteComms[neighbor].keys[di],
                            sizeof(stepValue));
                  } else {
                    self.dynReadyVector[size * concurrencyIndex + n] = 2;
                  }
                }));
            ++n;
          }
        }
      }

      // log.info("wait for enter kernel %d\n", stepValue);

      self.trace("kernel-entry");

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }
      kernelHasEntered = true;
      for (index = 0; index != ringSends.size(); ++index) {
        // self.trace("dyn-ready %d", index);
        size_t di = (rank + index) % self.devices.size();
        auto& dev = self.devices[di];
        if (self.dynReadyVector[size * concurrencyIndex + index] == 2 || dev.ordered) {
          auto [i, neighbor] = ringRecvs[index];
          self.writeData(dev, neighbor,
              sendStepValue,
              self.sendStepValues2StorageMr->mrs[di]->lkey,
              (void*)(self.remoteComms[neighbor].address + self.group->getSharedOffset(&self.localProgress2[size * concurrencyIndex + index].stepValue)),
              self.remoteComms[neighbor].keys[di],
              sizeof(stepValue));
          self.dynReadyVector[size * concurrencyIndex + index] = 3;
        }
      }

      // log.info("enter kernel %d ok\n", stepValue);

      CHECK(ringRecvs.size() == ringSends.size());

      for (size_t i = 0; i != nDevices; ++i) {
        sendStates[i].sendIndex = 0;
        sendStates[i].chunkIndex = 0;
        sendStates[i].liveSends = 0;
      }

      self.trace("send loop");

      while (true) {
        {
          size_t nDone = 0;
          for (size_t i = 0; i != nDevices; ++i) {
            auto& state = sendStates[i];
            if (state.liveSends >= numParallel) {
              continue;
            }
            if (state.sendIndex == ringSends.size()) {
              if (state.liveSends) {
                continue;
              }
              ++nDone;
              continue;
            }
            // wait for dyns
            if (self.localProgress2[size * concurrencyIndex + state.sendIndex].stepValue < stepValue) {
              continue;
            }
            size_t index = state.sendIndex;
            size_t chunkIndex = state.chunkIndex;
            size_t source = std::get<0>(ringSends[index]);
            size_t neighbor = std::get<1>(ringSends[index]);

            size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

            const size_t devChunkSize = currentChunkSize / nDevices / 1024u * 1024u;
            size_t offset = devChunkSize * i;
            size_t n = i == nDevices - 1 ? currentChunkSize - offset : devChunkSize;
            bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
            if (n == 0) {
              std::chrono::system_clock::time_point sendStart;
              if (profilingEnabled) {
                sendStart = std::chrono::system_clock::now();
              }
              ++state.liveSends;
              callback(isLastChunk, i, source, chunkIndex, neighbor, true, true, sendStart);
            } else {
              if (source != rank) {
                if (*(&self.group->cpuCommsDeviceDataSent2.at<uint32_t>(concurrencyIndex) + 32 * 32 * source + 32 * i +
                      chunkIndex) < stepValue) {
                  continue;
                }
              }
              uintptr_t srcAddr =
                  source == rank ? (uintptr_t)params.inputAddress : params.outputAddress + params.pitch * source;
              auto* srcMr = source == rank ? inputMr : self.outDynsMrVector[size * concurrencyIndex + source];

              auto& dev = self.devices[i];
              std::chrono::system_clock::time_point sendStart;
              if (profilingEnabled) {
                sendStart = std::chrono::system_clock::now();
              }
              ++state.liveSends;
              uintptr_t dstAddr = self.localDyns[size * concurrencyIndex + index].gatherAddress;
              auto key = self.localDyns[size * concurrencyIndex + index].gatherKey;
              bool efa = dev.ib->qpex != nullptr;
              self.writeData(
                  dev, neighbor, (void*)(srcAddr + chunkSize * chunkIndex + offset), srcMr->mrs[i]->lkey,
                  (void*)(dstAddr + chunkSize * chunkIndex + offset), key[i], n,
                  self.makeCallback([me = this, isLastChunk, i, efa, sendStart, source, chunkIndex, neighbor]() {
                    me->callback(isLastChunk, i, source, chunkIndex, neighbor, efa, true, sendStart);
                  }));
              if (!efa) {
                callback(isLastChunk, i, source, chunkIndex, neighbor, true, false, sendStart);
              }
            }

            if (isLastChunk) {
              state.chunkIndex = 0;
              ++state.sendIndex;
            } else {
              ++state.chunkIndex;
              CHECK(state.chunkIndex < numChunks);
            }
          }
          if (nDone == nDevices) {
            break;
          }
        }
        YIELD
      }

      for (size_t i = 0; i != nDevices; ++i) {
        CHECK(sendStates[i].liveSends == 0);
        CHECK(sendStates[i].sendIndex == ringSends.size());
      }

      self.trace("wait for recvs");
      for (index = 0; index != ringRecvs.size(); ++index) {
        i = std::get<0>(ringRecvs[index]);
        CHECK(i < size);
        // log.info("exit wait for source %d from %d\n", i, std::get<1>(ringRecvs[index]));
        for (index2 = 0; index2 != (params.bytes + chunkSize - 1) / chunkSize; ++index2) {
          for (di = 0; di != self.devices.size(); ++di) {
            while (*(&self.group->cpuCommsDeviceDataSent2.at<uint32_t>(concurrencyIndex) + 32 * 32 * i + 32 * di +
                     index2) < stepValue) {
              YIELD
            }
          }
        }
      }

      // log.info("wait for exit kernel %d\n", stepValue);

      self.trace("kernel-exit");

      cpuOut[0] = stepValue;
      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      // Clean up dyns
      for (index = 0; index != ringSends.size(); ++index) {
        CHECK(self.dynReadyVector[size * concurrencyIndex + index] != 0);
        CHECK(self.dynReadyVector[size * concurrencyIndex + index] != 2);
        self.dynReadyVector[size * concurrencyIndex + index] = 0;
      }

      // log.info("exit kernel %d ok\n", stepValue);

      self.trace("");

      self.cpuThread->freelistAllGather.push(&params);
      self.allocatorAllGatherRing.deallocate(this);
      DONE
    }
  };

  struct WorkAllGatherCpu : Work {
    QueueEntryAllGatherCpu& params;
    size_t i;
    size_t di;
    TemporaryBufferHandle inputBuffer;
    TemporaryBufferHandle outputBuffer;
    size_t liveSends;
    size_t index;

    const AllGather& allGather = *self.group->allGather;
    const Vector<size_t>& sendRanks = allGather.sendRanks;
    const Vector<size_t>& recvRanks = allGather.recvRanks;
    const std::vector<size_t>& ipcRanks = self.group->ipcRanks;
    const std::vector<size_t>& peerIndices = self.group->peerIndices;

    int prevRecvSource = -1;
    const Vector<ProxyInfo>& proxyInfo = allGather.proxyInfo;
    const Vector<ProxyDestinationInfo>& proxyDestinationInfo = allGather.proxyDestinationInfo;

    WorkAllGatherCpu(CpuThreadImpl& self, QueueEntryAllGatherCpu& params) : Work(self, params), params(params) {}

    void vmcopy(uintptr_t dst, int pid, uintptr_t src, size_t bytes) {
      iovec local;
      local.iov_base = (void*)dst;
      local.iov_len = bytes;
      iovec remote;
      remote.iov_base = (void*)src;
      remote.iov_len = bytes;
      if (process_vm_readv(pid, &local, 1, &remote, 1, 0) != bytes) {
        fatal("process_vm_readv failed with error: %s", std::strerror(errno));
      }
    }

    void step() {
      ENTER

      {
        // Registering the input/output addrs directly here would be fragile, as we cannot
        // check if a previous registration for this address corresponds to the same
        // allocation/physical-address as the current one, like we can for cuda with buffer ids.
        // Thus, we use a staging buffer.
        inputBuffer = self.allocateTemporaryBuffer(params.bytes, self.allGatherInputBuffers);
        outputBuffer = self.allocateTemporaryBuffer(params.pitch * size, self.allGatherOutputBuffers);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
        // Thus, we register each ranks output address independently, so they get different MRs.
        for (size_t i : recvRanks) {
          outDyns[i].gatherAddress = (uintptr_t)outputBuffer->cpuPointer + params.pitch * i;
          outDyns[i].gatherBytes = params.bytes;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = outputBuffer.mr->mrs[di]->rkey;
          }
        }

        {
          size_t n = 0;
          for (size_t i : recvRanks) {
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];
            self.writeData(
                dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                self.remoteComms[i].keys[di], sizeof(DynamicAddresses),
                self.makeCallback([this, di, i]() { self.writeStep(di, i, stepValue, concurrencyIndex); }));
            ++n;
          }
        }
      }

      self.group->cpuAddresses[concurrencyIndex].inputAddress = params.inputAddress;
      self.group->cpuAddresses[concurrencyIndex].outputAddress = (uintptr_t)outputBuffer->cpuPointer;
      self.group->cpuAddresses[concurrencyIndex].bytes = params.bytes;
      self.group->cpuAddresses[concurrencyIndex].pid = self.group->pid;
      self.group->cpuAddresses[concurrencyIndex].stepValue = stepValue;

      std::memcpy(inputBuffer->cpuPointer, (void*)params.inputAddress, params.bytes);

      for (index = 0; index != ipcRanks.size(); ++index) {
        while (self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex])->stepValue < stepValue) {
          YIELD
        }
      }

      self.sendStepValues[concurrencyIndex] = stepValue;

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        i = sendRanks[index];
        // wait for dyns
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)inputBuffer->cpuPointer;

          CHECK(self.localDyns[size * concurrencyIndex + i].gatherBytes == params.bytes);

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, inputBuffer.mr, self.localDyns[size * concurrencyIndex + i].gatherAddress,
              self.localDyns[size * concurrencyIndex + i].gatherKey,
              [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices[di];
                  self.writeData(
                      dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                      (void*)(self.remoteCpuCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                      self.remoteCpuCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                }
              });
        }
        while (liveSends >= self.devices.size() * 2) {
          YIELD
        }
      }

      std::memcpy((void*)(params.outputAddress + params.pitch * rank), (void*)params.inputAddress, params.bytes);
      for (size_t peerIndex : peerIndices) {
        auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
        CHECK(x->bytes == params.bytes);
        size_t i = ipcRanks[peerIndex];
        vmcopy(params.outputAddress + params.pitch * i, x->pid, x->inputAddress, params.bytes);
      }

      {
        for (index = 0; index != std::max(proxyDestinationInfo.size(), proxyInfo.size()); ++index) {
          if (index < proxyInfo.size()) {
            i = proxyInfo[index].source;
            if (prevRecvSource != i) {
              for (di = 0; di != self.devices.size(); ++di) {
                while (*(&self.group->cpuCommsDeviceDataSent.at<uint32_t>(concurrencyIndex) + 32 * i + di) <
                       stepValue) {
                  YIELD
                }
              }
              prevRecvSource = i;
            }

            self.group->getPeerVar(
                proxyInfo[index].destinationPeerIndex, self.group->cpuProxyReady)[size * concurrencyIndex + i] =
                stepValue;

            // std::memcpy(
            //     (void*)(params.outputAddress + params.pitch * i),
            //     (void*)((uintptr_t)outputBuffer->cpuPointer + params.pitch * i), params.bytes);
          }
          if (index < proxyDestinationInfo.size()) {
            i = proxyDestinationInfo[index].source;
            while (self.group->cpuProxyReady[size * concurrencyIndex + i] < stepValue) {
              YIELD
            }

            auto& pdi = proxyDestinationInfo[index];

            auto* x = self.group->getPeerVar(pdi.proxyPeerIndex, &self.group->cpuAddresses[concurrencyIndex]);
            vmcopy(params.outputAddress + params.pitch * i, x->pid, x->outputAddress + params.pitch * i, params.bytes);
          }
        }
      }

      for (index = 0; index != recvRanks.size(); ++index) {
        i = recvRanks[index];
        for (di = 0; di != self.devices.size(); ++di) {
          while (*(&self.group->cpuCommsDeviceDataSent.at<uint32_t>(concurrencyIndex) + 32 * i + di) < stepValue) {
            YIELD
          }
        }
        std::memcpy(
            (void*)(params.outputAddress + params.pitch * i),
            (void*)((uintptr_t)outputBuffer->cpuPointer + params.pitch * i), params.bytes);
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

      inputBuffer = {};
      outputBuffer = {};

      self.cpuThread->freelistAllGatherCpu.push(&params);
      self.allocatorAllGatherCpu.deallocate(this);
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
    const Vector<size_t>& sendRanks = reduceScatter.sendRanks;
    const Vector<size_t>& recvRanks = reduceScatter.recvRanks;

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
    size_t di;
    size_t liveSends;
    size_t index;

    uintptr_t prevReduceOutput = 0;

    TemporaryBufferHandle temporaryBuffer;

    TemporaryBufferHandle sendBuffer;
    TemporaryBufferHandle recvBuffer;

    const ReduceScatter& reduceScatter = *self.group->reduceScatter;
    const Vector<size_t>& sendRanks = reduceScatter.sendRanks;
    const Vector<size_t>& recvRanks = reduceScatter.recvRanks;
    const std::vector<size_t>& ipcRanks = self.group->ipcRanks;
    const std::vector<size_t>& peerIndices = self.group->peerIndices;

    WorkReduceScatterCpu(CpuThreadImpl& self, QueueEntryReduceScatterCpu& params)
        : Work(self, params), params(params) {}

    template<typename T, size_t dindex = 0, size_t opindex = 0>
    decltype(stepPtr) getStepPtr() {
      if (dindex == (size_t)params.dindex && opindex == (size_t)params.opindex) {
        return [](Work* w) { ((T*)w)->template step<(Dtype)dindex, (Reduction)opindex>(); };
      }
      if constexpr (dindex + 1 < (size_t)Dtype::count) {
        return getStepPtr<T, dindex + 1, opindex>();
      }
      if constexpr (opindex + 1 < (size_t)Reduction::count) {
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
        temporaryBuffer = self.allocateTemporaryBuffer(bytes, self.vmreadBuffers);
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
        recvBuffer = self.allocateTemporaryBuffer(params.pitch * recvRanks.size(), self.sendRecvBuffers);
        sendBuffer = self.allocateTemporaryBuffer(params.pitch * sendRanks.size(), self.sendRecvBuffers);
        uintptr_t recvAddress = (uintptr_t)recvBuffer->cpuPointer;

        if (!recvRanks.empty()) {
          CHECK(recvAddress);

          // sendMr = self.regMr(sendAddress, params.pitch * sendRanks.size());

          DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

          // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
          // Thus, we register each ranks output address independently, so they get different MRs.
          size_t n = 0;
          for (size_t i : recvRanks) {
            outDyns[i].gatherAddress = recvAddress + params.pitch * n;
            outDyns[i].gatherBytes = params.bytes;
            for (size_t di = 0; di != self.devices.size(); ++di) {
              outDyns[i].gatherKey[di] = recvBuffer.mr->mrs[di]->rkey;
            }
            ++n;
          }

          {
            size_t n = 0;
            for (size_t i : recvRanks) {
              size_t di = (rank + n) % self.devices.size();
              auto& dev = self.devices[di];
              self.writeData(
                  dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                  (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                  self.remoteComms[i].keys[di], sizeof(DynamicAddresses),
                  self.makeCallback([this, di, i]() { self.writeStep(di, i, stepValue, concurrencyIndex); }));
              ++n;
            }
          }
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

      self.sendStepValues[concurrencyIndex] = stepValue;

      // CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        i = sendRanks[index];
        {
          uintptr_t addr = (uintptr_t)sendBuffer->cpuPointer + params.pitch * index;
          reduceAdd<dtype, reduction>(addr, params.inputAddress + params.pitch * i);
          for (size_t peerIndex : peerIndices) {
            auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
            reduceAdd<dtype, reduction>(addr, vmread(x->pid, x->inputAddress + params.pitch * i, params.bytes));
          }
        }

        // wait for dyns
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)sendBuffer->cpuPointer + params.pitch * index;

          CHECK(self.localDyns[size * concurrencyIndex + i].gatherBytes == params.bytes);

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, sendBuffer.mr, self.localDyns[size * concurrencyIndex + i].gatherAddress,
              self.localDyns[size * concurrencyIndex + i].gatherKey,
              [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices[di];
                  self.writeData(
                      dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                      (void*)(self.remoteCpuCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                      self.remoteCpuCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
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

      for (index = 0; index != recvRanks.size(); ++index) {
        i = recvRanks[index];
        for (di = 0; di != self.devices.size(); ++di) {
          while (*(&self.group->cpuCommsDeviceDataSent.at<uint32_t>(concurrencyIndex) + 32 * i + di) < stepValue) {
            YIELD
          }
        }
        reduceAdd<dtype, reduction>(params.outputAddress, (uintptr_t)recvBuffer->cpuPointer + params.pitch * index);
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
      sendBuffer = {};
      recvBuffer = {};

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

  struct WorkBroadcastCpu : Work {
    QueueEntryBroadcastCpu& params;
    size_t i;
    size_t di;
    size_t offset;
    TemporaryBufferHandle temporaryBuffer;
    size_t liveSends;
    size_t index;

    const std::vector<size_t>& ipcRanks = self.group->ipcRanks;
    const std::vector<size_t>& peerIndices = self.group->peerIndices;

    WorkBroadcastCpu(CpuThreadImpl& self, QueueEntryBroadcastCpu& params) : Work(self, params), params(params) {}

    void vmcopy(uintptr_t dst, int pid, uintptr_t src, size_t bytes) {
      iovec local;
      local.iov_base = (void*)dst;
      local.iov_len = bytes;
      iovec remote;
      remote.iov_base = (void*)src;
      remote.iov_len = bytes;
      if (process_vm_readv(pid, &local, 1, &remote, 1, 0) != bytes) {
        fatal("process_vm_readv failed with error: %s", std::strerror(errno));
      }
    }

    void step() {
      ENTER

      if (rank != params.sourceRank && !self.group->ipcAccess[params.sourceRank]) {
        temporaryBuffer = self.allocateTemporaryBuffer(params.bytes, self.broadcastBuffers);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        size_t i = params.sourceRank;
        outDyns[i].gatherAddress = (uintptr_t)temporaryBuffer->cpuPointer;
        outDyns[i].gatherBytes = params.bytes;
        for (size_t di = 0; di != self.devices.size(); ++di) {
          outDyns[i].gatherKey[di] = temporaryBuffer.mr->mrs[di]->rkey;
        }

        {

          size_t di = (rank + 0) % self.devices.size();
          auto& dev = self.devices[di];
          self.writeData(
              dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
              (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
              self.remoteComms[i].keys[di], sizeof(DynamicAddresses),
              self.makeCallback([this, di, i]() { self.writeStep(di, i, stepValue, concurrencyIndex); }));
        }
      } else if (rank == params.sourceRank) {
        temporaryBuffer = self.allocateTemporaryBuffer(params.bytes, self.broadcastBuffers);
        std::memcpy(temporaryBuffer->cpuPointer, (void*)params.tensorAddress, params.bytes);
      }

      self.group->cpuAddresses[concurrencyIndex].inputAddress = params.tensorAddress;
      self.group->cpuAddresses[concurrencyIndex].outputAddress = params.tensorAddress;
      self.group->cpuAddresses[concurrencyIndex].bytes = params.bytes;
      self.group->cpuAddresses[concurrencyIndex].pid = self.group->pid;
      self.group->cpuAddresses[concurrencyIndex].stepValue = stepValue;

      for (index = 0; index != ipcRanks.size(); ++index) {
        while (self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex])->stepValue < stepValue) {
          YIELD
        }
      }

      liveSends = 0;

      if (rank != params.sourceRank) {
        if (self.group->ipcAccess[params.sourceRank]) {
          auto* x = self.group->getPeerVar(
              self.group->getPeerIndex(params.sourceRank), &self.group->cpuAddresses[concurrencyIndex]);
          CHECK(x->bytes == params.bytes);
          vmcopy(params.tensorAddress, x->pid, x->inputAddress, params.bytes);
        } else {
          i = params.sourceRank;

          for (di = 0; di != self.devices.size(); ++di) {
            while (*(&self.group->cpuCommsDeviceDataSent.at<uint32_t>(concurrencyIndex) + 32 * i + di) < stepValue) {
              YIELD
            }
          }
          std::memcpy((void*)params.tensorAddress, (void*)(uintptr_t)temporaryBuffer->cpuPointer, params.bytes);
        }
      } else if (rank == params.sourceRank) {
        self.sendStepValues[concurrencyIndex] = stepValue;

        for (i = 0; i != size; ++i) {
          if (i == rank || self.group->ipcAccess[i]) {
            continue;
          }
          // wait for dyns
          while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
            YIELD
          }

          if (params.bytes > 0) {
            uintptr_t srcAddr = (uintptr_t)temporaryBuffer->cpuPointer;

            CHECK(self.localDyns[size * concurrencyIndex + i].gatherBytes == params.bytes);

            liveSends += self.devices.size();
            self.writeDataDistributed2(
                i, srcAddr, params.bytes, temporaryBuffer.mr, self.localDyns[size * concurrencyIndex + i].gatherAddress,
                self.localDyns[size * concurrencyIndex + i].gatherKey,
                [this, i = this->i](size_t di, bool ordered, bool done) {
                  if (done) {
                    --liveSends;
                  }
                  if (ordered) {
                    auto& dev = self.devices[di];
                    self.writeData(
                        dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                        (void*)(self.remoteCpuCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                        self.remoteCpuCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                  }
                });
          }
          while (liveSends >= self.devices.size() * 2) {
            YIELD
          }
        }
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

      self.cpuThread->freelistBroadcastCpu.push(&params);
      self.allocatorBroadcastCpu.deallocate(this);
      DONE
    }
  };

  struct WorkGatherCpu : Work {
    QueueEntryGatherCpu& params;
    size_t i;
    size_t di;
    size_t offset;
    TemporaryBufferHandle inputBuffer;
    TemporaryBufferHandle outputBuffer;
    size_t liveSends;
    size_t index;

    const std::vector<size_t>& ipcRanks = self.group->ipcRanks;
    const std::vector<size_t>& peerIndices = self.group->peerIndices;

    WorkGatherCpu(CpuThreadImpl& self, QueueEntryGatherCpu& params) : Work(self, params), params(params) {}

    void vmcopy(uintptr_t dst, int pid, uintptr_t src, size_t bytes) {
      iovec local;
      local.iov_base = (void*)dst;
      local.iov_len = bytes;
      iovec remote;
      remote.iov_base = (void*)src;
      remote.iov_len = bytes;
      if (process_vm_readv(pid, &local, 1, &remote, 1, 0) != bytes) {
        fatal("process_vm_readv failed with error: %s", std::strerror(errno));
      }
    }

    void step() {
      ENTER

      if (rank == params.destinationRank) {
        CHECK(params.outputList.size() == size);
        size_t outputSize = 0;
        for (auto& v : params.outputList) {
          outputSize += (v.bytes + 63) / 64u * 64u;
        }
        outputBuffer = self.allocateTemporaryBuffer(outputSize, self.allGatherOutputBuffers);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        size_t offset = 0;
        for (size_t i = 0; i != size; ++i) {
          if (i == rank || self.group->ipcAccess[i]) {
            offset += (params.outputList[i].bytes + 63) / 64u * 64u;
            continue;
          }
          auto* mr = self.regMr((uintptr_t)outputBuffer->cpuPointer + offset, params.bytes);
          outDyns[i].gatherAddress = (uintptr_t)outputBuffer->cpuPointer + offset;
          outDyns[i].gatherBytes = params.outputList[i].bytes;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
          }
          offset += (params.outputList[i].bytes + 63) / 64u * 64u;
        }

        {
          size_t n = 0;
          for (size_t i = 0; i != size; ++i) {
            if (i == rank || self.group->ipcAccess[i]) {
              continue;
            }
            size_t di = (rank + n) % self.devices.size();
            auto& dev = self.devices[di];
            self.writeData(
                dev, i, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
                (void*)(self.remoteComms[i].address + sizeof(DynamicAddresses) * size * concurrencyIndex + sizeof(DynamicAddresses) * rank),
                self.remoteComms[i].keys[di], sizeof(DynamicAddresses),
                self.makeCallback([this, di, i]() { self.writeStep(di, i, stepValue, concurrencyIndex); }));
            ++n;
          }
        }
      } else {
        CHECK(params.outputList.empty());
        inputBuffer = self.allocateTemporaryBuffer(params.bytes, self.allGatherInputBuffers);
        std::memcpy(inputBuffer->cpuPointer, (void*)params.inputAddress, params.bytes);
      }

      self.group->cpuAddresses[concurrencyIndex].inputAddress = (uintptr_t)inputBuffer->cpuPointer;
      self.group->cpuAddresses[concurrencyIndex].outputAddress = (uintptr_t)outputBuffer->cpuPointer;
      self.group->cpuAddresses[concurrencyIndex].pid = self.group->pid;
      self.group->cpuAddresses[concurrencyIndex].stepValue = stepValue;

      for (index = 0; index != ipcRanks.size(); ++index) {
        while (self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex])->stepValue < stepValue) {
          YIELD
        }
      }

      liveSends = 0;

      if (rank == params.destinationRank) {
      } else if (!self.group->ipcAccess[params.destinationRank]) {
        self.sendStepValues[concurrencyIndex] = stepValue;

        i = params.destinationRank;
        // wait for dyns
        while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
          YIELD
        }

        if (params.bytes > 0) {
          uintptr_t srcAddr = (uintptr_t)inputBuffer->cpuPointer;

          CHECK(self.localDyns[size * concurrencyIndex + i].gatherBytes == params.bytes);

          liveSends += self.devices.size();
          self.writeDataDistributed2(
              i, srcAddr, params.bytes, inputBuffer.mr, self.localDyns[size * concurrencyIndex + i].gatherAddress,
              self.localDyns[size * concurrencyIndex + i].gatherKey,
              [this, i = this->i](size_t di, bool ordered, bool done) {
                if (done) {
                  --liveSends;
                }
                if (ordered) {
                  auto& dev = self.devices[di];
                  self.writeData(
                      dev, i, &self.sendStepValues[concurrencyIndex], self.sendStepValuesStorageMr->mrs[di]->lkey,
                      (void*)(self.remoteCpuCommsDeviceDataSent[i].address + sizeof(uint32_t) * (32 * size * concurrencyIndex + 32 * self.rank + di)),
                      self.remoteCpuCommsDeviceDataSent[i].keys[di], sizeof(stepValue));
                }
              });
        }
        while (liveSends >= self.devices.size() * 2) {
          YIELD
        }
      }

      if (rank == params.destinationRank) {
        CHECK(params.outputList[rank].bytes == params.bytes);
        std::memcpy((void*)params.outputList[rank].address, (void*)params.inputAddress, params.bytes);
        for (size_t peerIndex : peerIndices) {
          auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
          size_t i = ipcRanks[peerIndex];
          vmcopy(params.outputList[i].address, x->pid, x->inputAddress, params.bytes);
        }

        offset = 0;
        for (i = 0; i != size; ++i) {
          if (i == rank || self.group->ipcAccess[i]) {
            offset += (params.outputList[i].bytes + 63) / 64u * 64u;
            continue;
          }
          for (di = 0; di != self.devices.size(); ++di) {
            while (*(&self.group->cpuCommsDeviceDataSent.at<uint32_t>(concurrencyIndex) + 32 * i + di) < stepValue) {
              YIELD
            }
          }
          std::memcpy(
              (void*)params.outputList[i].address, (void*)((uintptr_t)outputBuffer->cpuPointer + offset), params.bytes);
          offset += (params.outputList[i].bytes + 63) / 64u * 64u;
        }
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

      inputBuffer = {};
      outputBuffer = {};

      self.cpuThread->freelistGatherCpu.push(&params);
      self.allocatorGatherCpu.deallocate(this);
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
  WorkAllocator<WorkInternalBarrier> allocatorInternalBarrier;
  WorkAllocator<WorkAllGather> allocatorAllGather;
  WorkAllocator<WorkAllGatherRing> allocatorAllGatherRing;
  WorkAllocator<WorkReduceScatter> allocatorReduceScatter;
  WorkAllocator<WorkReduceScatterRing> allocatorReduceScatterRing;
  WorkAllocator<WorkBroadcast> allocatorBroadcast;
  WorkAllocator<WorkAllGatherCpu> allocatorAllGatherCpu;
  WorkAllocator<WorkReduceScatterCpu> allocatorReduceScatterCpu;
  WorkAllocator<WorkGatherCpu> allocatorGatherCpu;
  WorkAllocator<WorkBroadcastCpu> allocatorBroadcastCpu;

  size_t numActiveWorks = 0;

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
          params.sd->cpuThreadIndex = streams.size();
          streams.resize(streams.size() + 1);
        }
        auto* work = allocator.allocate(*this, params);
        work->stepPtr = work->template getStepPtr<std::remove_pointer_t<decltype(work)>>();
        CHECK(work->streamIndex < streams.size());
        if (concurrency[work->concurrencyIndex].empty() && streams[work->streamIndex].empty()) {
          activeWorks.push_back(*work);
          ++numActiveWorks;
        }
        streams[work->streamIndex].push_back(*work);
        concurrency[work->concurrencyIndex].push_back(*work);
      };

      while (!cpuThread->terminate.load(std::memory_order_relaxed)) {
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
            } else if (queueEntry.task == taskAllGather) {
              enqueue(allocatorAllGatherRing, (QueueEntryAllGather&)queueEntry);
              // enqueue(allocatorAllGather, (QueueEntryAllGather&)queueEntry);
            } else if (queueEntry.task == taskReduceScatter) {
              // enqueue(allocatorReduceScatter, (QueueEntryReduceScatter&)queueEntry);
              enqueue(allocatorReduceScatterRing, (QueueEntryReduceScatter&)queueEntry);
            } else if (queueEntry.task == taskTerminate) {
              cpuThread->freelistTerminate.push(&queueEntry);
            } else if (queueEntry.task == taskBroadcast) {
              enqueue(allocatorBroadcast, (QueueEntryBroadcast&)queueEntry);
            } else if (queueEntry.task == taskAllGatherCpu) {
              enqueue(allocatorAllGatherCpu, (QueueEntryAllGatherCpu&)queueEntry);
            } else if (queueEntry.task == taskReduceScatterCpu) {
              enqueue(allocatorReduceScatterCpu, (QueueEntryReduceScatterCpu&)queueEntry);
            } else if (queueEntry.task == taskGatherCpu) {
              enqueue(allocatorGatherCpu, (QueueEntryGatherCpu&)queueEntry);
            } else if (queueEntry.task == taskBroadcastCpu) {
              enqueue(allocatorBroadcastCpu, (QueueEntryBroadcastCpu&)queueEntry);
            } else if (queueEntry.task == taskInternalBarrier) {
              enqueue(allocatorInternalBarrier, (QueueEntryBarrier&)queueEntry);
            } else {
              throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
            }
          }
          poll();

          for (auto i = activeWorks.begin(); i != activeWorks.end();) {
            Work* w = (Work*)&*i;
            CHECK(!w->done);
            w->stepPtr(w);
            if (w->done) {
              uint32_t concurrencyIndex = w->concurrencyIndex;
              uint32_t streamIndex = w->streamIndex;
              CHECK(&concurrency[concurrencyIndex].front() == w);
              concurrency[concurrencyIndex].pop_front();
              CHECK(&streams[streamIndex].front() == w);
              streams[streamIndex].pop_front();
              bool queuedThisConcurrency = false;
              if (!streams[streamIndex].empty()) {
                Work* nextWork = (Work*)&streams[streamIndex].front();
                if (&concurrency[nextWork->concurrencyIndex].front() == nextWork) {
                  queuedThisConcurrency = nextWork->concurrencyIndex == concurrencyIndex;
                  activeWorks.push_back(*nextWork);
                  ++numActiveWorks;
                }
              }
              if (!queuedThisConcurrency && !concurrency[concurrencyIndex].empty()) {
                Work* nextWork = (Work*)&concurrency[concurrencyIndex].front();
                if (&streams[nextWork->streamIndex].front() == nextWork) {
                  activeWorks.push_back(*nextWork);
                  ++numActiveWorks;
                }
              }
              i = activeWorks.erase(i);
              --numActiveWorks;
            } else {
              ++i;
            }
          }
        }
      }

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
