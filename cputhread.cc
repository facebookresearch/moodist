#include "cputhread.h"
#include "allgather.h"
#include "async.h"
#include "clock.h"
#include "common.h"
#include "freelist.h"
#include "group.h"
#include "hash_map.h"
#include "ib_common.h"
#include "intrusive_list.h"
#include "ipc_mapper.h"
#include "parameters.h"
#include "reduce_scatter.h"
#include "setup_comms.h"
#include "simple_vector.h"

#include "fmt/printf.h"
#include "synchronization.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <libibverbs/verbs.h>
#include <memory>
#include <numa.h>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include <sys/uio.h>

namespace moodist {

extern bool profilingEnabled;

struct CpuThreadImpl;

namespace {

struct MemoryRegistration {
  std::array<ibv_mr*, Group::maxDevices> mrs{};
};

struct CudaMapping {
  uint64_t bufferId;
  size_t bytes;
  MemoryRegistration mr;
};

struct CpuMapping {
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

#define WAIT_DYN(optype, i)                                                                                            \
  while (self.inDyn(concurrencyIndex, i).stepValue != stepValue) {                                                     \
    YIELD                                                                                                              \
  }                                                                                                                    \
  const DynamicAddresses& dyn = self.inDyn(concurrencyIndex, i);                                                       \
  CHECK_DYN(dyn, optype, stepValue, params.bytes);

struct Stats {
  SpinMutex mutex;
  Vector<std::array<float, 7>> values;

  ~Stats() {
    if (!values.empty()) {
      std::string fn = fmt::sprintf("stats-%s.txt", randomName());
      FILE* f = fopen(fn.c_str(), "wb");
      CHECK(f);
      for (auto& v : values) {
        fmt::fprintf(f, "%s\n", fmt::to_string(fmt::join(v, " ")));
      }
      fclose(f);

      log.info("stats dumped to %s (%d entries)\n", fn, values.size());
    }
  }
};
Stats stats;

struct CpuThreadImpl {
  CpuThread* cpuThread;
  Group* group = cpuThread->group;
  bool dead = false;

  CpuThreadImpl(CpuThread* cpuThread) : cpuThread(cpuThread) {
    CHECK(devices.size() <= maxDevices);
  }
  ~CpuThreadImpl() {
    dead = true;
  }

  const size_t rank = group->rank;
  const size_t size = group->size;

  static constexpr size_t maxWr = IbCommon::maxWr;
  static constexpr size_t maxCqEntries = IbCommon::maxCqEntries;

  SetupComms* setupComms = &*group->setupComms;
  SimpleVector<Device> devices = initDevices();
  IntrusiveList<Device, &Device::link> activeDevices;

  size_t bytesWritten = 0;
  size_t bytesRead = 0;

  std::vector<IbvMr> localMrs;

  std::vector<std::unique_ptr<CudaMapping>> cudaMappings;
  std::vector<std::unique_ptr<CpuMapping>> cpuMappings;

  HashMap<uintptr_t, CpuMapping*> mrMap;
  HashMap<uintptr_t, CudaMapping*> mrMapCuda;

  MemoryRegistration* commsMr = regMr((uintptr_t)group->mySharedMem, group->mySharedMemSize);

  DynamicAddresses* inDyns = group->localDyns;
  // Progress* localProgress = group->localProgress;
  // Progress* localProgress2 = group->localProgress2;
  uint32_t* cpuStepValues = group->cpuStepValues;
  uint32_t* cpuStepValuesDeviceChunks = group->cpuStepValuesDeviceChunks;

  // uint32_t* cudaStepValues = group->cudaStepValues;
  // uint32_t* cudaStepValuesDeviceChunks = group->cudaStepValuesDeviceChunks;

  MemoryRegistration* cudaStepValuesDeviceChunksMr = regMrCuda(
      group->cudaStepValuesDeviceChunksBuffer.buffer.cudaPointer, group->cudaStepValuesDeviceChunksBuffer.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteCudaStepValuesDeviceChunks = distributeAddressAndKeys(
      group->cudaStepValuesDeviceChunksBuffer.buffer.cudaPointer, cudaStepValuesDeviceChunksMr);

  AllocatedArray internalInStepValueBuffer = group->allocateArrayHost(sizeof(uint32_t) * size, Group::maxConcurrency);
  MemoryRegistration* internalInStepValueBufferMr =
      regMr((uintptr_t)internalInStepValueBuffer.buffer.cpuPointer, internalInStepValueBuffer.buffer.bytes);
  std::vector<RemoteAddressAndKey> remoteInternalInStepValueBuffer =
      distributeAddressAndKeys((uintptr_t)internalInStepValueBuffer.buffer.cpuPointer, internalInStepValueBufferMr);
  SimpleVector<uint32_t> nextInternalStepValue = SimpleVector<uint32_t>(Group::maxConcurrency);

  // MemoryRegistration* cudaStepValueMr =
  //     regMrCuda(group->cudaStepValue.buffer.cudaPointer, group->cudaStepValue.buffer.bytes);

  std::vector<RemoteAddressAndKey> remoteComms = distributeAddressAndKeys((uintptr_t)group->mySharedMem, commsMr);
  // std::vector<RemoteAddressAndKey> remoteCudaStepValue =
  //     distributeAddressAndKeys(group->cudaStepValue.buffer.cudaPointer, cudaStepValueMr);

  // MemoryRegistration* cudaCommsDeviceDataSentMr =
  //     regMrCuda(group->cudaCommsDeviceDataSent.buffer.cudaPointer, group->cudaCommsDeviceDataSent.buffer.bytes);
  // std::vector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent =
  //     distributeAddressAndKeys(group->cudaCommsDeviceDataSent.buffer.cudaPointer, cudaCommsDeviceDataSentMr);

  // MemoryRegistration* cpuCommsDeviceDataSentMr =
  //     regMr((uintptr_t)group->cpuCommsDeviceDataSent.buffer.cpuPointer, group->cpuCommsDeviceDataSent.buffer.bytes);
  // std::vector<RemoteAddressAndKey> remoteCpuCommsDeviceDataSent =
  //     distributeAddressAndKeys((uintptr_t)group->cpuCommsDeviceDataSent.buffer.cpuPointer, cpuCommsDeviceDataSentMr);

  // MemoryRegistration* cudaCommsDeviceDataSent2Mr =
  //     regMrCuda(group->cudaCommsDeviceDataSent2.buffer.cudaPointer, group->cudaCommsDeviceDataSent2.buffer.bytes);
  // std::vector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent2 =
  //     distributeAddressAndKeys(group->cudaCommsDeviceDataSent2.buffer.cudaPointer, cudaCommsDeviceDataSent2Mr);

  // MemoryRegistration* cpuCommsDeviceDataSent2Mr =
  //     regMr((uintptr_t)group->cpuCommsDeviceDataSent2.buffer.cpuPointer,
  //     group->cpuCommsDeviceDataSent2.buffer.bytes);
  // std::vector<RemoteAddressAndKey> remoteCpuCommsDeviceDataSent2 =
  //     distributeAddressAndKeys((uintptr_t)group->cpuCommsDeviceDataSent2.buffer.cpuPointer,
  //     cpuCommsDeviceDataSent2Mr);

  // AllocatedArray sendStepValues2Storage = group->allocateArrayHost(sizeof(uint32_t), 32 * 1024 *
  // Group::maxConcurrency); MemoryRegistration* sendStepValues2StorageMr =
  //     regMr((uintptr_t)sendStepValues2Storage.buffer.cpuPointer, sendStepValues2Storage.buffer.bytes);
  // uint32_t* sendStepValues2 = (uint32_t*)sendStepValues2Storage.buffer.cpuPointer;
  // size_t sendStepValues2Counter = 0;

  AllocatedArray outStepValuesStorage = group->allocateArrayHost(sizeof(uint32_t) * 32, Group::maxConcurrency);
  MemoryRegistration* outStepValuesStorageMr =
      regMr((uintptr_t)outStepValuesStorage.buffer.cpuPointer, outStepValuesStorage.buffer.bytes);
  uint32_t* outStepValues = (uint32_t*)outStepValuesStorage.buffer.cpuPointer;

  SimpleVector<size_t> concurrencyLiveControlTransfers = SimpleVector<size_t>(Group::maxConcurrency);

  AllocatedArray outDynsBuffer = group->allocateArrayHost(sizeof(DynamicAddresses) * size, Group::maxConcurrency);
  MemoryRegistration* outDynsMr = regMr((uintptr_t)outDynsBuffer.buffer.cpuPointer, outDynsBuffer.buffer.bytes);

  // Vector<uint8_t> dynReadyVector{size * Group::maxConcurrency};
  Vector<MemoryRegistration*> outDynsMrVector{size * Group::maxConcurrency};

  MemoryRegistration* cpuOutMr = regMrCuda(group->cpuOutBuffer.buffer.cudaPointer, group->cpuOutBuffer.buffer.bytes);

  // Vector<uint32_t> debugVector{size};

  // size_t uniqueIndexOffset = 0;
  // const size_t numUniqueIndices = std::max((size_t)1024, size * 2);

  // AllocatedArray uOutDynsBuffer =
  //     group->allocateArrayHost(sizeof(DynamicAddresses) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uOutDynsMr = regMr((uintptr_t)uOutDynsBuffer.buffer.cpuPointer, uOutDynsBuffer.buffer.bytes);
  // AllocatedArray uInDynsBuffer =
  //     group->allocateArrayHost(sizeof(DynamicAddresses) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uInDynsMr = regMr((uintptr_t)uInDynsBuffer.buffer.cpuPointer, uInDynsBuffer.buffer.bytes);
  // std::vector<RemoteAddressAndKey> remoteUInDyns =
  //     distributeAddressAndKeys((uintptr_t)uInDynsBuffer.buffer.cpuPointer, uInDynsMr);

  // AllocatedArray uOutStepValuesBuffer =
  //     group->allocateArrayHost(sizeof(uint32_t) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uStepValueMr =
  //     regMr((uintptr_t)uOutStepValuesBuffer.buffer.cpuPointer, uOutStepValuesBuffer.buffer.bytes);
  // AllocatedArray uInStepValuesBuffer =
  //     group->allocateArrayHost(sizeof(uint32_t) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uInStepValuesMr =
  //     regMr((uintptr_t)uInStepValuesBuffer.buffer.cpuPointer, uInStepValuesBuffer.buffer.bytes);
  // std::vector<RemoteAddressAndKey> remoteUInStepValues =
  //     distributeAddressAndKeys((uintptr_t)uInStepValuesBuffer.buffer.cpuPointer, uInStepValuesMr);

  // DynamicAddresses& uOutDyn(size_t concurrencyIndex, size_t uindex) {
  //   return *((DynamicAddresses*)uOutDynsBuffer.buffer.cpuPointer + numUniqueIndices * concurrencyIndex + uindex);
  // }
  // DynamicAddresses& uInDyn(size_t concurrencyIndex, size_t uindex) {
  //   return *((DynamicAddresses*)uInDynsBuffer.buffer.cpuPointer + numUniqueIndices * concurrencyIndex + uindex);
  // }

  // uint32_t& uOutStepValue(size_t concurrencyIndex, size_t uindex) {
  //   return *((uint32_t*)uOutStepValuesBuffer.buffer.cpuPointer + numUniqueIndices * concurrencyIndex + uindex);
  // }
  // uint32_t& uInStepValue(size_t concurrencyIndex, size_t uindex) {
  //   return *((uint32_t*)uInStepValuesBuffer.buffer.cpuPointer + numUniqueIndices * concurrencyIndex + uindex);
  // }

  DynamicAddresses& outDyn(size_t concurrencyIndex, size_t index) {
    CHECK(index < size);
    return *((DynamicAddresses*)outDynsBuffer.buffer.cpuPointer + size * concurrencyIndex + index);
  }
  DynamicAddresses& inDyn(size_t concurrencyIndex, size_t index) {
    CHECK(index < size);
    return inDyns[size * concurrencyIndex + index];
  }

  uint32_t& outStepValue(size_t concurrencyIndex, size_t index) {
    CHECK(index < 32);
    return outStepValues[32u * concurrencyIndex + index];
  }
  uint32_t& inStepValue(size_t concurrencyIndex, size_t index) {
    CHECK(index < size);
    return cpuStepValues[size * concurrencyIndex + index];
  }

  size_t stepValueDeviceChunkIndex(size_t concurrencyIndex, size_t index, size_t device, size_t chunk) {
    return Group::maxChunks * Group::maxDevices * size * concurrencyIndex + Group::maxChunks * size * device +
           size * chunk + index;
  }

  uint32_t& inStepValueDeviceChunk(size_t concurrencyIndex, size_t index, size_t device, size_t chunk) {
    CHECK(index < size);
    return cpuStepValuesDeviceChunks[stepValueDeviceChunkIndex(concurrencyIndex, index, device, chunk)];
  }

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
            if (wc.wr_id & 1) {
              --*(size_t*)(wc.wr_id & ~(uintptr_t)1);
            } else {
              Callback* callback = (Callback*)(void*)wc.wr_id;
              callback->decref();
            }
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

      // log.info(
      //     "%d: rdma write %d bytes (%p -> %p, rkey %#x) (dev %d, i %d)  (cb %#x)\n", rank, bytes, localAddress,
      //     remoteAddress, rkey, &dev - devices.data(), i, (uint64_t)(void*)callback);

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

  void readData(
      Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      Callback* callback = nullptr) {
    CHECK(i >= 0 && i < size);
    CHECK(i != rank);
    auto post = [&]() {
      ibv_sge sge;
      sge.addr = (uintptr_t)localAddress;
      sge.length = bytes;
      sge.lkey = lkey;

      bytesRead += bytes;

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

      // log.info(
      //     "%d: rdma read %d bytes (%p <- %p, lkey %#x, rkey %#x) (dev %d, i %d)  (cb %#x)\n", rank, bytes,
      //     localAddress, remoteAddress, lkey, rkey, &dev - devices.data(), i, (uint64_t)(void*)callback);

      bool efa = qp != nullptr;
      if (!efa) {
        qp = dev.ib->qpexs[i];
        CHECK(qp != nullptr);
      }

      ibv_wr_start(qp);
      qp->wr_id = (uint64_t)(void*)callback;
      qp->wr_flags = IBV_SEND_SIGNALED;
      ibv_wr_rdma_read(qp, rkey, (uintptr_t)remoteAddress);
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

  void writeControl(
      size_t concurrencyIndex, Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress,
      uint32_t rkey, size_t bytes) {
    CHECK(i >= 0 && i < size);
    // CHECK(i != rank);
    ibv_sge sge;
    sge.addr = (uintptr_t)localAddress;
    sge.length = bytes;
    sge.lkey = lkey;

    bytesWritten += bytes;

    while (dev.currentCqEntries == maxCqEntries) {
      poll();
    }
    ++dev.currentCqEntries;
    if (dev.currentCqEntries == 1) {
      activeDevices.push_back(dev);
    }

    ++concurrencyLiveControlTransfers[concurrencyIndex];

    ibv_qp_ex* qp = dev.ib->qpex;

    // log.info(
    //     "%d: rdma write control %d bytes (%p -> %p, rkey %#x) (dev %d, i %d)\n", rank, bytes, localAddress,
    //     remoteAddress, rkey, &dev - devices.data(), i);

    bool efa = qp != nullptr;
    if (!efa) {
      qp = dev.ib->qpexs[i];
      CHECK(qp != nullptr);
    }

    ibv_wr_start(qp);
    qp->wr_id = (uint64_t)(void*)&concurrencyLiveControlTransfers[concurrencyIndex] | 1;
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
  }

  MemoryRegistration* regMr(uintptr_t address, size_t bytes) {
    auto i = mrMap.find(address);
    if (i != mrMap.end()) {
      CHECK(i->second->bytes >= bytes);
      return &i->second->mr;
    }
    auto ptr = std::make_unique<CpuMapping>();
    ptr->mr.mrs.fill(nullptr);
    CHECK(devices.size() <= ptr->mr.mrs.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (devices[i2].protectionDomain == devices[i].protectionDomain) {
          ptr->mr.mrs[i] = ptr->mr.mrs[i2];
          break;
        }
      }
      if (ptr->mr.mrs[i]) {
        continue;
      }
      ibv_mr* mr = ibv_reg_mr(
          devices[i].ib->protectionDomain, (void*)address, bytes,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
      if (!mr) {
        perror("ibv_reg_mr");
        log.error("rank %d failed to register CPU memory at %#x size %#x\n", group->rank, address, bytes);
        TORCH_CHECK(false);
      }
      ptr->mr.mrs[i] = mr;
      localMrs.push_back(mr);
      log.debug("new cpu mapped range of %d bytes at %#x (mr lkey %#x rkey %#x)\n", bytes, address, mr->lkey, mr->rkey);
    }
    MemoryRegistration* r = &ptr->mr;
    mrMap[address] = &*ptr;
    cpuMappings.push_back(std::move(ptr));
    return r;
  }

  void deregMr(uintptr_t address) {
    auto i = mrMap.find(address);
    if (i == mrMap.end()) {
      return;
    }
    MemoryRegistration* mr = &i->second->mr;
    for (size_t i = 0; i != devices.size(); ++i) {
      if (mr->mrs[i]) {
        for (auto& v : localMrs) {
          if (&*v == mr->mrs[i]) {
            localMrs.erase(localMrs.begin() + (&v - localMrs.data()));
            break;
          }
        }
      }
    }
    mrMap.erase(i);
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
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (devices[i2].protectionDomain == devices[i].protectionDomain) {
          ptr->mr.mrs[i] = ptr->mr.mrs[i2];
          break;
        }
      }
      if (ptr->mr.mrs[i]) {
        continue;
      }
      ibv_mr* mr = ibv_reg_mr(
          devices[i].protectionDomain, (void*)address, bytes,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
      if (!mr) {
        perror("ibv_reg_mr");
        log.error("rank %d failed to register CUDA memory at %#x size %#x\n", group->rank, address, bytes);
        TORCH_CHECK(false);
      }
      ptr->mr.mrs[i] = mr;
      localMrs.push_back(mr);
      log.debug(
          "new cuda mapped range of %d bytes at %#x -> fd %d  (mr lkey %#x rkey %#x)\n", bytes, address, bufferId,
          mr->lkey, mr->rkey);
    }
    // ptr can be destructed here, but r is a plain pointer
    // fixme: return some kind of handle instead?
    MemoryRegistration* r = &ptr->mr;
    mrMapCuda[address] = &*ptr;
    cudaMappings.push_back(std::move(ptr));
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

  void writeDataDistributed(
      size_t dst, uintptr_t srcAddr, size_t len, MemoryRegistration* mr, uint64_t dstAddr,
      const std::array<uint32_t, maxDevices>& key, Callback* callback) {
    CHECK(len > 0);
    size_t offset = 0;
    const size_t chunks = devices.size();
    const size_t chunkSize = len / chunks / 128u * 128u;
    for (size_t i = 0; i != chunks; ++i) {
      size_t n = i == chunks - 1 ? len - offset : chunkSize;
      auto& dev = devices[i];
      if (n != 0) {
        writeData(
            dev, dst, (void*)(srcAddr + offset), mr->mrs[i]->lkey, (void*)(dstAddr + offset), key[i], n, callback);
      }
      offset += n;
    }
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

      deregMr((uintptr_t)testBuffer.cpuPointer);

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
    return;
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
        log.info("Chrome trace dumped to %s\n", fn);
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
    WAIT_CONTROL                                                                                                       \
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

#define WAIT_CONTROL                                                                                                   \
  while (self.concurrencyLiveControlTransfers[concurrencyIndex]) [[unlikely]] {                                        \
    YIELD                                                                                                              \
  }

  // #define SENDSTEPVALUE_SETUP                                                                                            \
//   if (self.sendStepValues2Counter == 0) {                                                                              \
//     while (true) {                                                                                                     \
//       {                                                                                                                \
//         size_t n = 0;                                                                                                  \
//         for (auto& dev : self.devices) {                                                                               \
//           n += dev.currentCqEntries;                                                                                   \
//         }                                                                                                              \
//         if (n == 0) {                                                                                                  \
//           break;                                                                                                       \
//         }                                                                                                              \
//       }                                                                                                                \
//       YIELD                                                                                                            \
//     }                                                                                                                  \
//   }                                                                                                                    \
//   sendStepValue = &self.sendStepValues2[32 * 1024 * concurrencyIndex + 32 * ((self.sendStepValues2Counter++) % 1024)]; \
//   *sendStepValue = stepValue;

  // size_t allocateIndices(size_t n) {
  //   if (numUniqueIndices - uniqueIndexOffset <= n) {
  //     // log.info("unique index overflow, reset!\n");
  //     uniqueIndexOffset = 0;
  //     CHECK(numUniqueIndices >= n);
  //     while (true) {
  //       {
  //         size_t n = 0;
  //         for (auto& dev : devices) {
  //           n += dev.currentCqEntries;
  //         }
  //         if (n == 0) {
  //           break;
  //         }
  //       }
  //       poll();
  //     }
  //   }
  //   return std::exchange(uniqueIndexOffset, uniqueIndexOffset + n);
  // }

  void writeDyn(size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex, size_t numDyns = 1) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inDyn(concurrencyIndex, dstIndex));
    writeControl(
        concurrencyIndex, dev, i, (void*)&outDyn(concurrencyIndex, srcIndex), outDynsMr->mrs[di]->lkey,
        (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di], sizeof(DynamicAddresses) * numDyns);
  }

  void writeStepValue(size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inStepValue(concurrencyIndex, dstIndex));
    writeControl(
        concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di],
        sizeof(uint32_t));
  }

  void writeStepValueDeviceChunk(
      size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex, size_t device, size_t chunk) {
    CHECK(i < size);
    CHECK(srcIndex < size);
    CHECK(dstIndex < size);
    CHECK(device < devices.size());
    CHECK(chunk < maxChunks);
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inStepValueDeviceChunk(concurrencyIndex, dstIndex, device, chunk));
    writeControl(
        concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di],
        sizeof(uint32_t));
  }

  void writeCudaStepValueDeviceChunk(
      size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex, size_t device, size_t chunk) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    size_t index = stepValueDeviceChunkIndex(concurrencyIndex, dstIndex, device, chunk);
    size_t offset = sizeof(uint32_t) * index;
    writeControl(
        concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteCudaStepValuesDeviceChunks[i].address + offset),
        remoteCudaStepValuesDeviceChunks[i].keys[di], sizeof(uint32_t));
  }

  void writeCpuOut(size_t concurrencyIndex, size_t dstIndex, size_t srcIndex) {
    CHECK(srcIndex < 32);
    CHECK(dstIndex < 4096 + maxChunks * size);
    size_t di = 0;
    auto& dev = devices[di];
    auto offset = sizeof(uint32_t) * dstIndex;
    writeControl(
        concurrencyIndex, dev, rank, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(group->cpuOutBuffer.cuda(concurrencyIndex) + offset),
        cpuOutMr->mrs[di]->rkey, sizeof(uint32_t));
  }

  struct WorkBarrier : Work {
    QueueEntryBarrier& params;
    size_t i;

    WorkBarrier(CpuThreadImpl& self, QueueEntryBarrier& params) : Work(self, params), params(params) {}

    void step() {
      ENTER {
        DynamicAddresses& out = self.outDyn(concurrencyIndex, 0);
        out.opType = opTypeBarrier;
        out.gatherAddress = 0;
        out.gatherBytes = 0;
        out.stepValue = stepValue;
      }
      self.outStepValue(concurrencyIndex, 0) = stepValue;
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        self.writeDyn(concurrencyIndex, i, 0, rank);
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (self.inDyn(concurrencyIndex, i).stepValue != stepValue) {
          YIELD
        }
        DynamicAddresses& dyn = self.inDyn(concurrencyIndex, i);
        CHECK_DYN(dyn, opTypeBarrier, stepValue, 0);
        self.writeStepValue(concurrencyIndex, i, 0, rank);
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (self.inStepValue(concurrencyIndex, i) != stepValue) {
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
    uint32_t stepValue;

    WorkInternalBarrier(CpuThreadImpl& self, QueueEntryBarrier& params) : Work(self, params), params(params) {}

    void step() {
      ENTER

      stepValue = (self.nextInternalStepValue[concurrencyIndex] ^= 1);

      {
        DynamicAddresses& out = self.outDyn(concurrencyIndex, 0);
        out.opType = opTypeInternalBarrier;
        out.gatherAddress = 0;
        out.gatherBytes = 0;
        out.stepValue = stepValue;
      }
      self.outStepValue(concurrencyIndex, 0) = stepValue;
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        size_t di = (rank + i) % self.devices.size();
        auto& dev = self.devices[di];
        self.writeControl(
            concurrencyIndex, dev, i, (void*)&self.outStepValue(concurrencyIndex, 0),
            self.outStepValuesStorageMr->mrs[di]->lkey,
            (void*)(self.remoteInternalInStepValueBuffer[i].address +
                    sizeof(uint32_t) * (size * concurrencyIndex + rank)),
            self.remoteInternalInStepValueBuffer[i].keys[di], sizeof(uint32_t));
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (*(&self.internalInStepValueBuffer.at<uint32_t>(concurrencyIndex) + i) != stepValue) {
          YIELD
        }
      }
      WAIT_CONTROL;
      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
      self.cpuThread->freelistBarrier.push(&params);
      self.allocatorInternalBarrier.deallocate(this);
      DONE
    }
  };

  // struct WorkReduceScatterRing : Work {
  //   QueueEntryReduceScatter& params;
  //   size_t i;
  //   // size_t neighbor;
  //   size_t di;
  //   MemoryRegistration* sendMr;
  //   size_t index;
  //   size_t index2;
  //   uintptr_t sendAddress;

  //   const ReduceScatter& reduceScatter = *self.group->reduceScatter;
  //   const Vector<std::pair<size_t, size_t>>& ringRecvs = reduceScatter.ringRecvs;
  //   const Vector<std::pair<size_t, size_t>>& ringSends = reduceScatter.ringSends;

  //   uint32_t* sendStepValue;
  //   bool kernelHasEntered = false;

  //   static const size_t numChunks = 8;
  //   const size_t chunkSize = params.bytes < (size_t)65536 * 96
  //                                ? params.bytes
  //                                : std::max((params.bytes + numChunks - 1) / numChunks, (size_t)65536 * 72);
  //   const size_t numParallel = std::max(std::min((params.bytes + chunkSize - 1) / chunkSize / 2, (size_t)2),
  //   (size_t)1);

  //   struct SendState {
  //     size_t sendIndex;
  //     size_t chunkIndex;
  //     size_t liveSends;
  //   };
  //   const size_t nDevices = self.devices.size();
  //   std::array<SendState, 32> sendStates;

  //   WorkReduceScatterRing(CpuThreadImpl& self, QueueEntryReduceScatter& params) : Work(self, params), params(params)
  //   {}

  //   void callback(
  //       bool isLastChunk, size_t di, size_t source, size_t chunkIndex, size_t neighbor, bool ordered, bool done,
  //       std::chrono::system_clock::time_point sendStart) {
  //     if (done) {
  //       --sendStates[di].liveSends;

  //       if (profilingEnabled) {
  //         TraceEvent e;
  //         e.name = fmt::sprintf("%d -> %d di %d chunk %d", source, neighbor, di, chunkIndex);
  //         e.begin = sendStart;
  //         e.end = std::chrono::system_clock::now();
  //         e.threadId = di * numChunks + chunkIndex;
  //         self.traceEvents.push_back(std::move(e));
  //       }
  //     }
  //     if (ordered) {
  //       auto& dev = self.devices[di];
  //       if (isLastChunk) {
  //         self.writeData(
  //             dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
  //             (void*)(self.remoteCudaCommsDeviceDataSent[neighbor].address + sizeof(uint32_t) * (32 * size *
  //             concurrencyIndex + 32 * source + di)), self.remoteCudaCommsDeviceDataSent[neighbor].keys[di],
  //             sizeof(stepValue));
  //       } else {
  //         self.writeData(
  //             dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
  //             (void*)(self.remoteCudaCommsDeviceDataSent2[neighbor].address + sizeof(uint32_t) * (32 * 32 * size *
  //             concurrencyIndex + 32 * 32 * source + 32 * di + chunkIndex)),
  //             self.remoteCudaCommsDeviceDataSent2[neighbor].keys[di], sizeof(stepValue));
  //       }
  //     }
  //   }

  //   void step() {
  //     ENTER

  //     self.trace("reduce-scatter ring %#x", stepValue);

  //     SENDSTEPVALUE_SETUP;

  //     {
  //       uintptr_t recvAddress = params.sd->recvBuffer.cudaPointer;
  //       CHECK(params.sd->recvBuffer.bytes >= params.pitch * ringRecvs.size());

  //       sendAddress = params.sd->sendBuffer.cudaPointer;
  //       CHECK(params.sd->sendBuffer.bytes >= params.pitch * ringSends.size());

  //       CHECK(sendAddress && recvAddress);

  //       sendMr = self.regMrCuda(sendAddress, params.pitch * ringSends.size());

  //       DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

  //       // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
  //       // Thus, we register each ranks output address independently, so they get different MRs.
  //       size_t n = 0;
  //       for (auto [i, neighbor] : ringRecvs) {
  //         auto* mr = self.regMrCuda(recvAddress + params.pitch * n, params.bytes);
  //         outDyns[i].gatherAddress = recvAddress + params.pitch * n;
  //         outDyns[i].gatherBytes = params.bytes;
  //         outDyns[i].stepValue = stepValue;
  //         for (size_t di = 0; di != self.devices.size(); ++di) {
  //           outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
  //         }
  //         ++n;
  //       }

  //       // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
  //       // until kernel entry, such that we don't receive data until we're ready.
  //       // todo this can all be sent in one write
  //       {
  //         size_t n = 0;
  //         for (auto [i, neighbor] : ringRecvs) {
  //           size_t di = (rank + n) % self.devices.size();
  //           auto& dev = self.devices[di];

  //           auto offset = self.group->getSharedOffset(&self.group->localDyns[size * concurrencyIndex + n]);

  //           self.writeData(
  //               dev, neighbor, &outDyns[i], self.outDynsMr->mrs[di]->lkey,
  //               (void*)(self.remoteComms[neighbor].address + offset), self.remoteComms[neighbor].keys[di],
  //               sizeof(DynamicAddresses),
  //               dev.ordered ? nullptr : self.makeCallback([this, di, n, neighbor = neighbor]() {
  //                 if (kernelHasEntered) {
  //                   self.dynReadyVector[size * concurrencyIndex + n] = 1;
  //                   // log.info("dyn index %d sent to %d\n", n, neighbor);
  //                   auto& dev = self.devices[di];
  //                   self.writeData(dev, neighbor,
  //                           sendStepValue,
  //                           self.sendStepValues2StorageMr->mrs[di]->lkey,
  //                           (void*)(self.remoteComms[neighbor].address +
  //                           self.group->getSharedOffset(&self.localProgress2[size * concurrencyIndex +
  //                           n].stepValue)), self.remoteComms[neighbor].keys[di], sizeof(stepValue));
  //                 } else {
  //                   self.dynReadyVector[size * concurrencyIndex + n] = 2;
  //                 }
  //               }));
  //           ++n;
  //         }
  //       }
  //     }

  //     // log.info("wait for enter kernel %d\n", stepValue);

  //     self.trace("kernel-entry");

  //     // wait for kernel
  //     while (cpuIn[0] < stepValue) {
  //       YIELD
  //     }
  //     kernelHasEntered = true;
  //     for (index = 0; index != ringSends.size(); ++index) {
  //       // self.trace("dyn-ready %d", index);
  //       size_t di = (rank + index) % self.devices.size();
  //       auto& dev = self.devices[di];
  //       if (self.dynReadyVector[size * concurrencyIndex + index] == 2 || dev.ordered) {
  //         auto [i, neighbor] = ringRecvs[index];
  //         self.writeData(dev, neighbor,
  //             sendStepValue,
  //             self.sendStepValues2StorageMr->mrs[di]->lkey,
  //             (void*)(self.remoteComms[neighbor].address + self.group->getSharedOffset(&self.localProgress2[size *
  //             concurrencyIndex + index].stepValue)), self.remoteComms[neighbor].keys[di], sizeof(stepValue));
  //         self.dynReadyVector[size * concurrencyIndex + index] = 3;
  //       }
  //     }

  //     // log.info("enter kernel %d ok\n", stepValue);

  //     CHECK(ringRecvs.size() == ringSends.size());

  //     for (size_t i = 0; i != nDevices; ++i) {
  //       sendStates[i].sendIndex = 0;
  //       sendStates[i].chunkIndex = 0;
  //       sendStates[i].liveSends = 0;
  //     }

  //     self.trace("send loop");

  //     while (true) {
  //       {
  //         size_t nDone = 0;
  //         for (size_t i = 0; i != nDevices; ++i) {
  //           auto& state = sendStates[i];
  //           if (state.liveSends >= numParallel) {
  //             continue;
  //           }
  //           if (state.sendIndex == ringSends.size()) {
  //             if (state.liveSends) {
  //               continue;
  //             }
  //             ++nDone;
  //             continue;
  //           }
  //           // wait for dyns
  //           if (self.localProgress2[size * concurrencyIndex + state.sendIndex].stepValue < stepValue) {
  //             continue;
  //           }
  //           // wait for send ready
  //           if (cpuIn[1] < stepValue + state.sendIndex + 1) {
  //             continue;
  //           }
  //           size_t index = state.sendIndex;
  //           size_t chunkIndex = state.chunkIndex;
  //           size_t source = std::get<0>(ringSends[index]);
  //           size_t neighbor = std::get<1>(ringSends[index]);

  //           size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

  //           const size_t devChunkSize = currentChunkSize / nDevices / 1024u * 1024u;
  //           size_t offset = devChunkSize * i;
  //           size_t n = i == nDevices - 1 ? currentChunkSize - offset : devChunkSize;
  //           bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
  //           if (n == 0) {
  //             std::chrono::system_clock::time_point sendStart;
  //             if (profilingEnabled) {
  //               sendStart = std::chrono::system_clock::now();
  //             }
  //             ++state.liveSends;
  //             callback(isLastChunk, i, source, chunkIndex, neighbor, true, true, sendStart);
  //           } else {
  //             uintptr_t srcAddr = sendAddress + params.pitch * index;
  //             auto* srcMr = sendMr;

  //             auto& dev = self.devices[i];
  //             std::chrono::system_clock::time_point sendStart;
  //             if (profilingEnabled) {
  //               sendStart = std::chrono::system_clock::now();
  //             }
  //             ++state.liveSends;
  //             uintptr_t dstAddr = self.localDyns[size * concurrencyIndex + index].gatherAddress;
  //             auto key = self.localDyns[size * concurrencyIndex + index].gatherKey;
  //             bool efa = dev.ib->qpex != nullptr;
  //             self.writeData(
  //                 dev, neighbor, (void*)(srcAddr + chunkSize * chunkIndex + offset), srcMr->mrs[i]->lkey,
  //                 (void*)(dstAddr + chunkSize * chunkIndex + offset), key[i], n,
  //                 self.makeCallback([me = this, isLastChunk, i, efa, sendStart, source, chunkIndex, neighbor]() {
  //                   me->callback(isLastChunk, i, source, chunkIndex, neighbor, efa, true, sendStart);
  //                 }));
  //             if (!efa) {
  //               callback(isLastChunk, i, source, chunkIndex, neighbor, true, false, sendStart);
  //             }
  //           }

  //           if (isLastChunk) {
  //             state.chunkIndex = 0;
  //             ++state.sendIndex;
  //           } else {
  //             ++state.chunkIndex;
  //             CHECK(state.chunkIndex < numChunks);
  //           }
  //         }
  //         if (nDone == nDevices) {
  //           break;
  //         }
  //       }
  //       YIELD
  //     }

  //     for (size_t i = 0; i != nDevices; ++i) {
  //       CHECK(sendStates[i].liveSends == 0);
  //       CHECK(sendStates[i].sendIndex == ringSends.size());
  //     }

  //     // log.info("wait for exit kernel %d\n", stepValue);

  //     self.trace("kernel-exit");

  //     cpuOut[0] = stepValue;
  //     while (cpuIn[0] < stepValue + 1) {
  //       YIELD
  //     }

  //     // Clean up dyns
  //     for (index = 0; index != ringSends.size(); ++index) {
  //       CHECK(self.dynReadyVector[size * concurrencyIndex + index] != 0);
  //       CHECK(self.dynReadyVector[size * concurrencyIndex + index] != 2);
  //       self.dynReadyVector[size * concurrencyIndex + index] = 0;
  //     }

  //     // log.info("exit kernel %d ok\n", stepValue);

  //     self.trace("");

  //     self.cpuThread->freelistReduceScatter.push(&params);
  //     self.allocatorReduceScatterRing.deallocate(this);
  //     DONE
  //   }
  // };

  struct WorkReduceScatterRingRead : Work {
    QueueEntryReduceScatter& params;
    MemoryRegistration* recvMr;
    uintptr_t recvAddress;

    size_t recvNeighbor;
    size_t sendNeighbor;

    const ReduceScatter& reduceScatter = *self.group->reduceScatter;
    const Vector<std::pair<size_t, size_t>>& ringRecvs = reduceScatter.ringRecvs;
    const Vector<std::pair<size_t, size_t>>& ringSends = reduceScatter.ringSends;

    const size_t numDevices = params.numDevices;
    const size_t numChunks = params.numChunks;
    const size_t chunkSize = ((params.bytes + numChunks - 1) / numChunks + 4095u) / 4096u * 4096u;
    const size_t numParallel = params.numParallel;

    size_t readIndex = 0;
    size_t chunkIndex = 0;
    size_t liveReads = 0;
    std::array<size_t, maxDevices> deviceLiveReads{};
    size_t reduceIndex = 0;
    size_t reduceChunkIndex = 0;

    WorkReduceScatterRingRead(CpuThreadImpl& self, QueueEntryReduceScatter& params)
        : Work(self, params), params(params) {
      CHECK(ringSends.size() != 0);
      CHECK(ringRecvs.size() != 0);
      recvNeighbor = std::get<1>(ringRecvs[0]);
      sendNeighbor = std::get<1>(ringSends[0]);

      CHECK(numDevices <= self.devices.size());
      CHECK(numChunks != 0 && numChunks <= maxChunks);
      CHECK(numParallel != 0);
      CHECK(chunkSize > 0);

      // log.info(
      //     "reduce-scatter ring read bytes %d numDevices %d, numChunks %d, numParallel %d\n", params.bytes,
      //     numDevices, numChunks, numParallel);
    }

    void tryRead(size_t di) {
      size_t index = readIndex;
      size_t chunkIndex = this->chunkIndex;
      size_t source = std::get<0>(ringRecvs[index]);

      size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

      size_t n = currentChunkSize;
      bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
      size_t readOffset = chunkSize * chunkIndex;
      if (n != 0) {
        if (self.inStepValueDeviceChunk(concurrencyIndex, source, 0, chunkIndex) != stepValue) {
          return;
        }

        uintptr_t dstAddr = recvAddress + params.pitch * index;

        ++liveReads;
        ++deviceLiveReads[di];
        const DynamicAddresses& dyn = self.inDyn(concurrencyIndex, 0);
        uintptr_t srcAddr = dyn.gatherAddress + params.pitch * index;
        auto key = dyn.gatherKey;
        auto& dev = self.devices[di];
        self.readData(
            dev, recvNeighbor, (void*)(dstAddr + readOffset), recvMr->mrs[di]->lkey, (void*)(srcAddr + readOffset),
            key[di], n, self.makeCallback([this, di, source, chunkIndex]() {
              --liveReads;
              --deviceLiveReads[di];
              self.writeCpuOut(concurrencyIndex, 16 + size * chunkIndex + source, 0);
            }));
      }

      if (isLastChunk) {
        this->chunkIndex = 0;
        ++readIndex;
      } else {
        ++this->chunkIndex;
        CHECK(this->chunkIndex < numChunks);
      }
    }

    void step() {
      ENTER

      self.trace("ring %#x", stepValue);

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        recvAddress = params.sd->recvBuffer.cudaPointer;
        CHECK(params.sd->recvBuffer.bytes >= params.pitch * ringRecvs.size());

        uintptr_t sendAddress = params.sd->sendBuffer.cudaPointer;
        CHECK(params.sd->sendBuffer.bytes >= params.pitch * ringSends.size());

        CHECK(sendAddress && recvAddress);

        recvMr = self.regMrCuda(recvAddress, params.pitch * ringSends.size());
        auto* sendMr = self.regMrCuda(sendAddress, params.pitch * ringSends.size());

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        outDyns[0].gatherAddress = sendAddress;
        outDyns[0].gatherBytes = params.bytes;
        outDyns[0].stepValue = stepValue;
        outDyns[0].opType = opTypeReduceScatterRingCuda;
        for (size_t di = 0; di != numDevices; ++di) {
          outDyns[0].gatherKey[di] = sendMr->mrs[di]->rkey;
        }
      }

      // log.info("wait for enter kernel %d\n", stepValue);

      self.trace("kernel-entry");

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }

      self.writeDyn(concurrencyIndex, sendNeighbor, 0, 0, 1);
      { WAIT_DYN(opTypeReduceScatterRingCuda, 0); }

      tryRead(0);

      self.trace("read loop");
      // log.info("enter read loop\n");

      // readBegin = self.bytesRead;
      // startTime = std::chrono::steady_clock::now();

      while (true) {
        {
          bool done = false;
          do {
            if (reduceIndex != ringSends.size()) {
              if (cpuIn[16 + size * reduceChunkIndex + reduceIndex] == stepValue) {
                // log.info("send reduce source %d chunk %d\n", ringSends[reduceIndex].first, reduceChunkIndex);
                CHECK(ringSends[reduceIndex].first != rank);
                self.writeStepValueDeviceChunk(
                    concurrencyIndex, sendNeighbor, 0, ringSends[reduceIndex].first, 0, reduceChunkIndex);
                size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * reduceChunkIndex);
                if (chunkSize * reduceChunkIndex + currentChunkSize == params.bytes) {
                  ++reduceIndex;
                  reduceChunkIndex = 0;
                } else {
                  ++reduceChunkIndex;
                }
              }
            }
            if (readIndex == ringRecvs.size()) {
              if (liveReads || reduceIndex != ringSends.size()) {
                continue;
              }
              self.writeStepValueDeviceChunk(concurrencyIndex, recvNeighbor, 0, rank, 0, 0);
              done = true;
              continue;
            }

            size_t bestDevice = 0;
            size_t bestLiveReads = numParallel;
            for (size_t di = 0; di != numDevices; ++di) {
              if (deviceLiveReads[di] < bestLiveReads) {
                bestLiveReads = deviceLiveReads[di];
                bestDevice = di;
              }
            }
            if (bestLiveReads >= numParallel) {
              continue;
            }
            size_t di = bestDevice;

            tryRead(di);
          } while (false);
          if (done) {
            break;
          }
        }
        YIELD
      }

      CHECK(liveReads == 0);
      CHECK(readIndex == ringRecvs.size());

      // {
      //   size_t r = self.bytesRead - readBegin;
      //   float t = seconds(std::chrono::steady_clock::now() - startTime);
      //   log.info(
      //       "read %d bytes in %gs - %gG/s (with %d waits, %d reads)\n", r, t, (float)r / 1024 / 1024 / 1024 / t,
      //       waitCounter, readCounter);
      // }
      // log.info("wait for remote reads\n");
      self.trace("wait for remote reads");
      while (self.inStepValueDeviceChunk(concurrencyIndex, sendNeighbor, 0, 0) != stepValue) {
        YIELD
      }

      // log.info("wait for exit kernel %d\n", stepValue);

      // cpuOut[0] = stepValue;
      self.writeCpuOut(concurrencyIndex, 0, 0);

      self.trace("kernel-exit");

      // {
      //   float t = seconds(std::chrono::steady_clock::now() - startTime);
      //   if (rank == 0) {
      //     if (params.paramsData) {
      //       params.paramsData->addTiming(params.paramsIndex, t);
      //     }
      //     std::lock_guard l(stats.mutex);
      //     stats.values.push_back(
      //         {(float)size, (float)(1 + self.group->ipcRanks.size()), (float)params.bytes, (float)numDevices,
      //          (float)numChunks, (float)numParallel, t});
      //   }
      // }

      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      // log.info("exit kernel %d ok\n", stepValue);

      self.trace("");

      self.cpuThread->freelistReduceScatter.push(&params);
      self.allocatorReduceScatterRingRead.deallocate(this);
      DONE
    }
  };

  struct WorkAllGatherRingRead4 : Work {
    QueueEntryAllGather& params;
    // size_t di;
    MemoryRegistration* inputMr;

    size_t recvNeighbor;
    size_t sendNeighbor;

    // size_t readBegin;
    // std::chrono::steady_clock::time_point startTime;

    const AllGather& allGather = *self.group->allGather;
    const Vector<std::pair<size_t, size_t>>& ringRecvs = allGather.ringRecvs;
    const Vector<std::pair<size_t, size_t>>& ringSends = allGather.ringSends;

    const size_t numDevices = params.numDevices;
    const size_t numChunks = params.numChunks;
    const size_t chunkSize = ((params.bytes + numChunks - 1) / numChunks + 4095u) / 4096u * 4096u;
    const size_t numParallel = params.numParallel;

    // struct DataTime {
    //   bool inProgress = false;
    //   std::chrono::steady_clock::time_point startTime;
    //   std::chrono::steady_clock::duration prevDuration{};
    // };
    // std::array<DataTime, 4> times;
    std::chrono::steady_clock::time_point startTime;

    size_t readIndex = 0;
    size_t chunkIndex = 0;
    size_t liveReads = 0;
    std::array<size_t, maxDevices> deviceLiveReads{};

    WorkAllGatherRingRead4(CpuThreadImpl& self, QueueEntryAllGather& params) : Work(self, params), params(params) {
      CHECK(ringSends.size() != 0);
      CHECK(ringRecvs.size() != 0);
      recvNeighbor = std::get<1>(ringRecvs[0]);
      sendNeighbor = std::get<1>(ringSends[0]);

      CHECK(numDevices <= self.devices.size());
      CHECK(numChunks != 0 && numChunks <= maxChunks);
      CHECK(numParallel != 0);
      CHECK(chunkSize > 0);

      // log.info(
      //     "allgather ring read bytes %d numDevices %d, numChunks %d, numParallel %d\n", params.bytes, numDevices,
      //     numChunks, numParallel);
    }

    void tryRead(size_t di) {
      // CHECK(!times[parallelIndex].inProgress);
      size_t index = readIndex;
      size_t chunkIndex = this->chunkIndex;
      size_t source = std::get<0>(ringRecvs[index]);

      size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

      // const size_t devChunkSize = currentChunkSize / numDevices / 1024u * 1024u;
      // size_t offset = devChunkSize * di;
      // size_t n = di == numDevices - 1 ? currentChunkSize - offset : devChunkSize;
      size_t n = currentChunkSize;
      bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
      size_t readOffset = chunkSize * chunkIndex;
      if (n == 0) {
        if (source != sendNeighbor) {
          self.writeStepValueDeviceChunk(concurrencyIndex, sendNeighbor, 0, source, 0, chunkIndex);
        }
        // self.writeCudaStepValueDeviceChunk(concurrencyIndex, neighbor, 0, source, di, chunkIndex);
      } else {
        if (source != recvNeighbor) {
          if (self.inStepValueDeviceChunk(concurrencyIndex, source, 0, chunkIndex) != stepValue) {
            return;
          }
        }
        // allgather ring read bytes 139985192 numDevices 4, numChunks 32, numParallel 2

        // size_t prevParallelIndex = (parallelIndex + numParallel - 1) % numParallel;
        // auto now = std::chrono::steady_clock::now();
        // if (times[prevParallelIndex].inProgress &&
        //     now - times[prevParallelIndex].startTime < times[prevParallelIndex].prevDuration / 2) {
        //   // log.info("read wait!\n");
        //   ++waitCounter;
        //   continue;
        // }
        // times[parallelIndex].startTime = now;
        // ++parallelCounter;
        uintptr_t dstAddr = params.outputAddress + params.pitch * source;
        auto* srcMr = source == rank ? inputMr : self.outDynsMrVector[size * concurrencyIndex + 0];

        ++liveReads;
        ++deviceLiveReads[di];
        const DynamicAddresses& dyn = self.inDyn(concurrencyIndex, source == recvNeighbor ? 1 : 0);
        uintptr_t srcAddr = source == recvNeighbor ? dyn.gatherAddress : dyn.gatherAddress + params.pitch * source;
        auto key = dyn.gatherKey;
        // times[parallelIndex].inProgress = true;
        //++readCounter;
        auto& dev = self.devices[di];
        self.readData(
            dev, recvNeighbor, (void*)(dstAddr + readOffset), srcMr->mrs[di]->lkey, (void*)(srcAddr + readOffset),
            key[di], n, self.makeCallback([this, di, source, chunkIndex /*, parallelIndex*/]() {
              // auto now = std::chrono::steady_clock::now();
              // times[parallelIndex].inProgress = false;
              // times[parallelIndex].prevDuration = now - times[parallelIndex].startTime;
              // log.info("duration[%d] set to %d\n", parallelIndex, times[parallelIndex].prevDuration.count());
              --liveReads;
              --deviceLiveReads[di];
              if (source != sendNeighbor) {
                self.writeStepValueDeviceChunk(concurrencyIndex, sendNeighbor, 0, source, 0, chunkIndex);
              }
              // self.writeCudaStepValueDeviceChunk(concurrencyIndex, neighbor, 0, source, di, chunkIndex);
              // if (readIndex != ringRecvs.size()) {
              //   tryRead(di);
              // }

              // cpuOut[16 + size * chunkIndex + source] = stepValue;
              self.writeCpuOut(concurrencyIndex, 16 + size * chunkIndex + source, 0);
            }));
      }

      if (isLastChunk) {
        this->chunkIndex = 0;
        ++readIndex;
      } else {
        ++this->chunkIndex;
        CHECK(this->chunkIndex < numChunks);
      }
    }

    void step() {
      ENTER

      self.trace("ring %#x", stepValue);

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        inputMr = self.regMrCuda(params.inputAddress, params.bytes);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        auto* mr = self.regMrCuda(params.outputAddress, params.pitch * size);
        self.outDynsMrVector[size * concurrencyIndex + 0] = mr;
        outDyns[0].gatherAddress = params.outputAddress;
        outDyns[0].gatherBytes = params.bytes;
        outDyns[0].stepValue = stepValue;
        outDyns[0].opType = opTypeAllGatherRingCuda;
        for (size_t di = 0; di != numDevices; ++di) {
          outDyns[0].gatherKey[di] = mr->mrs[di]->rkey;
        }

        outDyns[1].gatherAddress = params.inputAddress;
        outDyns[1].gatherBytes = params.bytes;
        outDyns[1].stepValue = stepValue;
        outDyns[1].opType = opTypeAllGatherRingCuda;
        for (size_t di = 0; di != numDevices; ++di) {
          outDyns[1].gatherKey[di] = inputMr->mrs[di]->rkey;
        }
      }

      // log.info("wait for enter kernel %d\n", stepValue);

      self.trace("kernel-entry");

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }

      self.writeDyn(concurrencyIndex, sendNeighbor, 0, 0, 2);
      { WAIT_DYN(opTypeAllGatherRingCuda, 0); }
      { WAIT_DYN(opTypeAllGatherRingCuda, 1); }

      tryRead(0);

      startTime = std::chrono::steady_clock::now();

      self.trace("read loop");
      // log.info("enter read loop\n");

      // readBegin = self.bytesRead;
      // startTime = std::chrono::steady_clock::now();

      while (true) {
        {
          bool done = false;
          do {
            // if (liveReads >= numParallel) {
            //   continue;
            // }
            if (readIndex == ringRecvs.size()) {
              if (liveReads) {
                continue;
              }
              self.writeStepValueDeviceChunk(concurrencyIndex, recvNeighbor, 0, recvNeighbor, 0, 0);
              done = true;
              continue;
            }
            // size_t parallelIndex = parallelCounter % numParallel;

            size_t bestDevice = 0;
            size_t bestLiveReads = numParallel;
            for (size_t di = 0; di != numDevices; ++di) {
              if (deviceLiveReads[di] < bestLiveReads) {
                bestLiveReads = deviceLiveReads[di];
                bestDevice = di;
              }
            }
            if (bestLiveReads >= numParallel) {
              continue;
            }
            size_t di = bestDevice;

            tryRead(di);
          } while (false);
          if (done) {
            break;
          }
        }
        YIELD
      }

      CHECK(liveReads == 0);
      CHECK(readIndex == ringRecvs.size());

      // {
      //   size_t r = self.bytesRead - readBegin;
      //   float t = seconds(std::chrono::steady_clock::now() - startTime);
      //   log.info(
      //       "read %d bytes in %gs - %gG/s (with %d waits, %d reads)\n", r, t, (float)r / 1024 / 1024 / 1024 / t,
      //       waitCounter, readCounter);
      // }
      // log.info("wait for remote reads\n");
      self.trace("wait for remote reads");
      while (self.inStepValueDeviceChunk(concurrencyIndex, rank, 0, 0) != stepValue) {
        YIELD
      }

      // log.info("wait for exit kernel %d\n", stepValue);

      // cpuOut[0] = stepValue;
      self.writeCpuOut(concurrencyIndex, 0, 0);

      self.trace("kernel-exit");

      // {
      //   float t = seconds(std::chrono::steady_clock::now() - startTime);
      //   if (rank == 0) {
      //     if (params.paramsData) {
      //       params.paramsData->addTiming(params.paramsIndex, t);
      //     }
      //     std::lock_guard l(stats.mutex);
      //     stats.values.push_back(
      //         {(float)size, (float)(1 + self.group->ipcRanks.size()), (float)params.bytes, (float)numDevices,
      //          (float)numChunks, (float)numParallel, t});
      //   }
      // }

      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      // log.info("exit kernel %d ok\n", stepValue);

      self.trace("");

      self.cpuThread->freelistAllGather.push(&params);
      self.allocatorAllGatherRingRead4.deallocate(this);
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
    size_t recvIndex;
    size_t proxyIndex;

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

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      // log.info("Rank %d enter step %#x\n", rank, stepValue);

      {
        // Registering the input/output addrs directly here would be fragile, as we cannot
        // check if a previous registration for this address corresponds to the same
        // allocation/physical-address as the current one, like we can for cuda with buffer ids.
        // Thus, we use a staging buffer.
        inputBuffer = self.allocateTemporaryBuffer(params.bytes, self.allGatherInputBuffers);
        outputBuffer = self.allocateTemporaryBuffer(params.pitch * size, self.allGatherOutputBuffers);

        DynamicAddresses* outDyns = &self.outDyn(concurrencyIndex, 0);

        for (size_t i : recvRanks) {
          outDyns[i].gatherAddress = (uintptr_t)outputBuffer->cpuPointer + params.pitch * i;
          outDyns[i].gatherBytes = params.bytes;
          outDyns[i].stepValue = stepValue;
          outDyns[i].opType = opTypeAllGatherCpu;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = outputBuffer.mr->mrs[di]->rkey;
          }

          self.writeDyn(concurrencyIndex, i, i, rank);
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

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        i = sendRanks[index];

        {
          WAIT_DYN(opTypeAllGatherCpu, i);

          uintptr_t srcAddr = (uintptr_t)inputBuffer->cpuPointer;

          ++liveSends;
          self.writeDataDistributed(
              i, srcAddr, params.bytes, inputBuffer.mr, dyn.gatherAddress, dyn.gatherKey,
              self.makeCallback([this, i = this->i] {
                --liveSends;
                self.writeStepValue(concurrencyIndex, i, 0, rank);
              }));
        }
        while (liveSends >= 2) {
          YIELD
        }
      }

      std::memcpy((void*)(params.outputAddress + params.pitch * rank), (void*)params.inputAddress, params.bytes);
      for (size_t peerIndex : peerIndices) {
        auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
        CHECK(x->bytes == params.bytes);
        CHECK(x->stepValue == stepValue || x->stepValue == stepValue + 1);
        size_t i = ipcRanks[peerIndex];
        vmcopy(params.outputAddress + params.pitch * i, x->pid, x->inputAddress, params.bytes);
      }

      recvIndex = 0;
      proxyIndex = 0;
      while (recvIndex != proxyInfo.size() || proxyIndex != proxyDestinationInfo.size()) {
        if (recvIndex != proxyInfo.size()) {
          size_t i = proxyInfo[recvIndex].source;
          if (prevRecvSource != i) {
            if (self.inStepValue(concurrencyIndex, i) == stepValue) {
              prevRecvSource = i;
              continue;
            }
          } else {
            // log.info("rank %d got recv source %d (forward to %d)\n", rank, i, proxyInfo[recvIndex].destination);
            self.group->getPeerVar(
                proxyInfo[recvIndex].destinationPeerIndex, self.group->cpuProxyReady)[size * concurrencyIndex + i] =
                stepValue;
            ++recvIndex;

            // log.info(
            //     "rank %d next recv source is %d\n", rank,
            //     recvIndex == proxyInfo.size() ? -1 : proxyInfo[recvIndex].source);
            continue;
          }
        }
        if (proxyIndex != proxyDestinationInfo.size()) {
          size_t i = proxyDestinationInfo[proxyIndex].source;
          if (self.group->cpuProxyReady[size * concurrencyIndex + i] >= stepValue) {
            // log.info("rank %d got proxy source %d\n", rank, i);

            auto& pdi = proxyDestinationInfo[proxyIndex];

            auto* x = self.group->getPeerVar(pdi.proxyPeerIndex, &self.group->cpuAddresses[concurrencyIndex]);
            CHECK(x->bytes == params.bytes);
            CHECK(x->stepValue == stepValue || x->stepValue == stepValue + 1);
            vmcopy(params.outputAddress + params.pitch * i, x->pid, x->outputAddress + params.pitch * i, params.bytes);
            ++proxyIndex;
            // log.info(
            //     "rank %d next proxy source is %d\n", rank,
            //     proxyIndex == proxyDestinationInfo.size() ? -1 : proxyDestinationInfo[proxyIndex].source);
            continue;
          }
        }
        YIELD
      }

      for (index = 0; index != recvRanks.size(); ++index) {
        i = recvRanks[index];
        while (self.inStepValue(concurrencyIndex, i) != stepValue) {
          YIELD
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

    uint32_t* sendStepValue;

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

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        recvBuffer = self.allocateTemporaryBuffer(params.pitch * recvRanks.size(), self.sendRecvBuffers);
        sendBuffer = self.allocateTemporaryBuffer(params.pitch * sendRanks.size(), self.sendRecvBuffers);
        uintptr_t recvAddress = (uintptr_t)recvBuffer->cpuPointer;

        if (!recvRanks.empty()) {
          CHECK(recvAddress);

          DynamicAddresses* outDyns = &self.outDyn(concurrencyIndex, 0);

          size_t n = 0;
          for (size_t i : recvRanks) {
            outDyns[i].opType = opTypeReduceScatterCpu;
            outDyns[i].stepValue = stepValue;
            outDyns[i].gatherAddress = recvAddress + params.pitch * n;
            outDyns[i].gatherBytes = params.bytes;
            for (size_t di = 0; di != self.devices.size(); ++di) {
              outDyns[i].gatherKey[di] = recvBuffer.mr->mrs[di]->rkey;
            }
            ++n;

            self.writeDyn(concurrencyIndex, i, i, rank);
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

      // CHECK(recvRanks.size() == sendRanks.size());

      liveSends = 0;
      for (index = 0; index != sendRanks.size(); ++index) {
        i = sendRanks[index];
        {
          uintptr_t addr = (uintptr_t)sendBuffer->cpuPointer + params.pitch * index;
          reduceAdd<dtype, reduction>(addr, params.inputAddress + params.pitch * i);
          for (size_t peerIndex : peerIndices) {
            auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
            CHECK(x->stepValue == stepValue || x->stepValue == stepValue + 1);
            reduceAdd<dtype, reduction>(addr, vmread(x->pid, x->inputAddress + params.pitch * i, params.bytes));
          }
        }

        {
          WAIT_DYN(opTypeReduceScatterCpu, i);

          uintptr_t srcAddr = (uintptr_t)sendBuffer->cpuPointer + params.pitch * index;

          ++liveSends;
          self.writeDataDistributed(
              i, srcAddr, params.bytes, sendBuffer.mr, dyn.gatherAddress, dyn.gatherKey,
              self.makeCallback([this, i = this->i] {
                --liveSends;
                self.writeStepValue(concurrencyIndex, i, 0, rank);
              }));
        }
        while (liveSends >= 2) {
          YIELD
        }
      }

      reduceAdd<dtype, reduction>(params.outputAddress, params.inputAddress + params.pitch * rank);
      for (size_t peerIndex : peerIndices) {
        auto* x = self.group->getPeerVar(peerIndex, &self.group->cpuAddresses[concurrencyIndex]);
        CHECK(x->stepValue == stepValue || x->stepValue == stepValue + 1);
        reduceAdd<dtype, reduction>(
            params.outputAddress, vmread(x->pid, x->inputAddress + params.pitch * rank, params.bytes));
      }

      for (index = 0; index != recvRanks.size(); ++index) {
        while (self.inStepValue(concurrencyIndex, recvRanks[index]) != stepValue) {
          YIELD
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

  template<typename Parent, typename Params, bool hasReduceStep, uint32_t opType>
  struct WorkBroadcastReduceBase : Work {
    Params& params;

    MemoryRegistration* recvMr;
    size_t source;
    size_t i;

    const size_t numDevices = params.numDevices;
    const size_t numChunks = params.numChunks;
    const size_t chunkSize = ((params.bytes + numChunks - 1) / numChunks + 4095u) / 4096u * 4096u;
    const size_t numParallel = params.numParallel;

    size_t readIndex = 0;
    size_t chunkIndex = 0;
    size_t liveReads = 0;
    std::array<size_t, maxDevices> deviceLiveReads{};
    size_t reduceIndex = 0;
    size_t reduceChunkIndex = 0;
    bool hasSentAckReads = false;

    const size_t* sends = nullptr;
    size_t nSends = 0;
    const size_t* recvs = nullptr;
    size_t nRecvs = 0;

    uintptr_t recvAddress;

    WorkBroadcastReduceBase(CpuThreadImpl& self, Params& params) : Work(self, params), params(params) {
      CHECK(numDevices <= self.devices.size());
      CHECK(numChunks != 0 && numChunks <= maxChunks);
      CHECK(numParallel != 0);
      CHECK(chunkSize > 0);

      recvAddress = params.tensorAddress;
    }

    void tryRead(size_t di) {
      size_t index = readIndex;
      size_t chunkIndex = this->chunkIndex;
      size_t source = this->source;

      size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

      size_t sendTo = index >= nSends ? rank : sends[index];
      size_t recvFrom = index >= nRecvs ? rank : recvs[index];

      CHECK(recvFrom != rank);

      size_t n = currentChunkSize;
      bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
      size_t readOffset = chunkSize * chunkIndex;
      if (n == 0) {
        if (sendTo != rank) {
          self.writeStepValueDeviceChunk(concurrencyIndex, sendTo, 0, rank, 0, chunkIndex);
        }
      } else {
        if (recvFrom != source) {
          if (self.inStepValueDeviceChunk(concurrencyIndex, recvFrom, 0, chunkIndex) != stepValue) {
            return;
          }
        }
        uintptr_t dstAddr = recvAddress + params.bytes * index;
        auto* srcMr = recvMr;

        ++liveReads;
        ++deviceLiveReads[di];
        const DynamicAddresses& dyn = self.inDyn(concurrencyIndex, recvFrom);
        uintptr_t srcAddr = dyn.gatherAddress;
        auto key = dyn.gatherKey;
        auto& dev = self.devices[di];
        // log.info(
        //     "%s step %#x tensor %#x recv %#x recvMr %#x (%p, %p) doing read! dstAddr %#x, readOffset %#x, "
        //     "srcMr->mrs[%d]->lkey "
        //     "is %#x\n",
        //     hasReduceStep ? "reduce" : "broadcast", stepValue, params.tensorAddress, recvAddress,
        //     recvMr ? recvMr->mrs[0]->lkey : 0, (void*)recvMr, (void*)recvMr->mrs[0], dstAddr, readOffset, di,
        //     srcMr->mrs[di]->lkey);
        self.readData(
            dev, recvFrom, (void*)(dstAddr + readOffset), srcMr->mrs[di]->lkey, (void*)(srcAddr + readOffset), key[di],
            n, self.makeCallback([this, di, source, chunkIndex, sendTo, index]() {
              --liveReads;
              --deviceLiveReads[di];
              if (!hasReduceStep && sendTo != rank) {
                self.writeStepValueDeviceChunk(concurrencyIndex, sendTo, 0, rank, 0, chunkIndex);
              }
              self.writeCpuOut(concurrencyIndex, 16 + size * chunkIndex + (hasReduceStep ? index : source), 0);
            }));
      }

      if (isLastChunk) {
        this->chunkIndex = 0;
        ++readIndex;
      } else {
        ++this->chunkIndex;
        CHECK(this->chunkIndex < numChunks);
      }
    }

    void step() {
      ENTER

      self.trace("ring %#x", stepValue);

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        MemoryRegistration* tensorMr = self.regMrCuda(params.tensorAddress, params.bytes);
        recvMr = tensorMr;
        if (nRecvs == 0) {
          recvMr = nullptr;
        } else if (recvAddress == params.tensorAddress) {
          CHECK(!hasReduceStep);
          CHECK(nRecvs <= 1);
        } else {
          CHECK(hasReduceStep);
          recvMr = self.regMrCuda(recvAddress, params.bytes * nRecvs);
        }

        // log.info(
        //     "%s step %#x tensor %#x recv %#x tensorMr %#x (%p, %p) recvMr %#x (%p, %p)\n",
        //     hasReduceStep ? "reduce" : "broadcast", stepValue, params.tensorAddress, recvAddress,
        //     tensorMr->mrs[0]->lkey, (void*)tensorMr, tensorMr ? (void*)tensorMr->mrs[0] : 0,
        //     recvMr ? recvMr->mrs[0]->lkey : 0, (void*)recvMr, recvMr ? (void*)recvMr->mrs[0] : 0);

        DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

        outDyns[0].gatherAddress = params.tensorAddress;
        outDyns[0].gatherBytes = params.bytes;
        outDyns[0].stepValue = stepValue;
        outDyns[0].opType = opType;
        for (size_t di = 0; di != numDevices; ++di) {
          outDyns[0].gatherKey[di] = tensorMr->mrs[di]->rkey;
        }
      }

      if (!hasReduceStep) {
        CHECK(nRecvs <= 1);
      } else {
        CHECK(nSends <= 1);
      }

      // log.info("wait for enter kernel %d\n", stepValue);

      self.trace("kernel-entry");

      // wait for kernel
      while (cpuIn[0] < stepValue) {
        YIELD
      }

      // log.info("%s nSends %d nRecvs %d\n", hasReduceStep ? "reduce" : "broadcast", nSends, nRecvs);

      for (size_t i = 0; i != nSends; ++i) {
        // log.info("send %d is %d\n", i, sends[i]);
        self.writeDyn(concurrencyIndex, sends[i], 0, rank);
      }

      for (i = 0; i != nRecvs; ++i) {
        // log.info("recv %d is %d\n", i, recvs[i]);
        WAIT_DYN(opType, recvs[i]);
      }

      if (nRecvs != 0) {
        tryRead(0);
      }

      self.trace("read loop");

      while (true) {
        {
          bool done = false;
          do {
            if (hasReduceStep && reduceIndex != nSends) {
              if (cpuIn[16 + size * reduceChunkIndex + reduceIndex] == stepValue) {
                self.writeStepValueDeviceChunk(concurrencyIndex, sends[reduceIndex], 0, rank, 0, reduceChunkIndex);
                size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * reduceChunkIndex);
                if (chunkSize * reduceChunkIndex + currentChunkSize == params.bytes) {
                  ++reduceIndex;
                  reduceChunkIndex = 0;
                } else {
                  ++reduceChunkIndex;
                }
              }
            }
            if (readIndex == nRecvs) {
              if (liveReads) {
                continue;
              }
              if (!hasSentAckReads) {
                hasSentAckReads = true;
                for (size_t i = 0; i != nRecvs; ++i) {
                  size_t recvFrom = recvs[i];
                  self.writeStepValueDeviceChunk(concurrencyIndex, recvFrom, 0, recvFrom, 0, chunkIndex);
                }
              }
              if (hasReduceStep && reduceIndex != nSends) {
                continue;
              }
              done = true;
              continue;
            }

            size_t bestDevice = 0;
            size_t bestLiveReads = numParallel;
            for (size_t di = 0; di != numDevices; ++di) {
              if (deviceLiveReads[di] < bestLiveReads) {
                bestLiveReads = deviceLiveReads[di];
                bestDevice = di;
              }
            }
            if (bestLiveReads >= numParallel) {
              continue;
            }
            tryRead(bestDevice);
          } while (false);
          if (done) {
            break;
          }
        }
        YIELD
      }

      CHECK(liveReads == 0);
      CHECK(readIndex == nRecvs);
      if (hasReduceStep) {
        CHECK(reduceIndex == nSends);
      }

      CHECK(nSends <= 1)
      for (i = 0; i != nSends; ++i) {
        self.trace("wait for remote reads");
        while (self.inStepValueDeviceChunk(concurrencyIndex, rank, 0, 0) != stepValue) {
          YIELD
        }
      }

      // log.info("wait for exit kernel %d\n", stepValue);

      // cpuOut[0] = stepValue;
      self.writeCpuOut(concurrencyIndex, 0, 0);

      self.trace("kernel-exit");

      // ?? why did we wait for the kernel here ??
      //    this would prevent a new op from running that could write some data to the kernel
      //    but we don't depend on any such data, do we?
      //    just the cpuOut step values
      //    the next op cannot do anything until it gets its step value from the kernel, anyways
      //   could it be legacy leftover from when we sent dyns early?
      while (cpuIn[0] < stepValue + 1) {
        YIELD
      }

      // log.info("exit kernel %d ok\n", stepValue);

      self.trace("");

      static_assert(std::is_base_of_v<WorkBroadcastReduceBase, Parent>);

      ((Parent*)this)->deallocate();
      DONE
    }
  };

  struct WorkBroadcastRing
      : WorkBroadcastReduceBase<WorkBroadcastRing, QueueEntryBroadcast, false, opTypeBroadcastRingCuda> {
    std::array<size_t, 1> sendsArr;
    std::array<size_t, 1> recvsArr;
    WorkBroadcastRing(CpuThreadImpl& self, QueueEntryBroadcast& params) : WorkBroadcastReduceBase(self, params) {
      size_t sourceRank = params.sourceRank;
      source = sourceRank;
      size_t sourceLocalRank = self.group->rankLocalRank.at(sourceRank);
      size_t nNodes = self.group->nodeRanks.size();
      CHECK(nNodes > 1);
      size_t nodeIndex = self.group->rankToNodeIndex.at(rank);
      auto& node = self.group->nodeRanks.at(nodeIndex);
      CHECK(!node.empty());
      CHECK(node[sourceLocalRank % node.size()] == rank);
      auto& nextNode = self.group->nodeRanks.at((nodeIndex + 1) % nNodes);
      auto& prevNode = self.group->nodeRanks.at((nodeIndex + nNodes - 1) % nNodes);
      sendsArr[0] = nextNode.at(sourceLocalRank % nextNode.size());
      recvsArr[0] = prevNode.at(sourceLocalRank % prevNode.size());
      sends = sendsArr.data();
      recvs = recvsArr.data();
      nSends = sends[0] == sourceRank ? 0 : 1;
      nRecvs = rank == sourceRank ? 0 : 1;
    }
    void deallocate() {
      self.cpuThread->freelistBroadcast.push(&params);
      self.allocatorBroadcast.deallocate(this);
    }
  };

  struct WorkReduceTree : WorkBroadcastReduceBase<WorkReduceTree, QueueEntryReduce, true, opTypeReduceTreeCuda> {
    static constexpr size_t nary = 4;
    std::array<size_t, 1> sendsArr;
    std::array<size_t, nary> recvsArr;
    WorkReduceTree(CpuThreadImpl& self, QueueEntryReduce& params) : WorkBroadcastReduceBase(self, params) {
      source = params.destinationRank;
      auto* tree = self.group->getTree(nary, params.destinationRank);

      CHECK(tree->sends.size() <= 1);
      CHECK(tree->recvs.size() <= nary);

      std::copy(tree->sends.begin(), tree->sends.end(), sendsArr.data());
      std::copy(tree->recvs.begin(), tree->recvs.end(), recvsArr.data());

      nSends = tree->sends.size();
      nRecvs = tree->recvs.size();

      sends = sendsArr.data();
      recvs = recvsArr.data();

      recvAddress = params.recvBuffer;
    }
    void deallocate() {
      self.cpuThread->freelistReduce.push(&params);
      self.allocatorReduce.deallocate(this);
    }
  };

  // struct WorkBroadcast : Work {
  //   QueueEntryBroadcast& params;
  //   size_t i;
  //   MemoryRegistration* tensorMr;
  //   size_t liveSends;

  //   WorkBroadcast(CpuThreadImpl& self, QueueEntryBroadcast& params) : Work(self, params), params(params) {}

  //   void step() {
  //     ENTER

  //     tensorMr = self.regMrCuda(params.tensorAddress, params.bytes);

  //     if (params.sourceRank != rank) {
  //       {
  //         DynamicAddresses* outDyns = (DynamicAddresses*)self.outDynsBuffer.cpu(concurrencyIndex);

  //         for (size_t i = 0; i != size; ++i) {
  //           if (i == rank) {
  //             continue;
  //           }
  //           outDyns[i].opType = opTypeReduceScatterCpu;
  //           outDyns[i].stepValue = stepValue;
  //           outDyns[i].gatherBytes = params.bytes;
  //           outDyns[i].gatherAddress = params.tensorAddress;
  //           for (size_t di = 0; di != self.devices.size(); ++di) {
  //             outDyns[i].gatherKey[di] = tensorMr->mrs[di]->rkey;
  //           }

  //           self.writeDyn(concurrencyIndex, i, i, rank);
  //         }
  //       }

  //       // wait for kernel
  //       while (cpuIn[0] < stepValue) {
  //         YIELD
  //       }

  //       i = params.sourceRank;
  //       WAIT_DYN(opTypeBroadcastRingCpu, sendTo);
  //       {
  //         size_t di = (rank + i) % self.devices.size();
  //         self.writeStep(di, i, stepValue, concurrencyIndex);
  //       }
  //       while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
  //         YIELD
  //       }
  //     } else {

  //       // wait for kernel
  //       while (cpuIn[0] < stepValue) {
  //         YIELD
  //       }

  //       liveSends = 0;
  //       for (i = 0; i != size; ++i) {
  //         if (i == rank) {
  //           continue;
  //         }
  //         // wait for dyns
  //         while (self.localProgress[size * concurrencyIndex + i].stepValue < stepValue) {
  //           YIELD
  //         }

  //         if (params.bytes > 0) {
  //           uintptr_t srcAddr = (uintptr_t)params.tensorAddress;

  //           liveSends += self.devices.size();
  //           self.writeDataDistributed2(
  //               i, srcAddr, params.bytes, tensorMr, self.localDyns[size * concurrencyIndex + i].gatherAddress,
  //               self.localDyns[size * concurrencyIndex + i].gatherKey,
  //               [this, i = this->i](size_t di, bool ordered, bool done) {
  //                 if (done) {
  //                   --liveSends;
  //                 }
  //                 if (ordered) {
  //                 }
  //               });
  //         }
  //         while (liveSends) {
  //           YIELD
  //         }
  //         size_t di = (rank + i) % self.devices.size();
  //         self.writeStep(di, i, stepValue, concurrencyIndex);
  //       }

  //       while (liveSends) {
  //         YIELD
  //       }
  //     }

  //     cpuOut[0] = stepValue;
  //     while (cpuIn[0] < stepValue + 1) {
  //       YIELD
  //     }

  //     self.cpuThread->freelistBroadcast.push(&params);
  //     self.allocatorBroadcast.deallocate(this);
  //     DONE
  //   }
  // };

  struct WorkBroadcastRingCpu : Work {
    QueueEntryBroadcastCpu& params;
    // size_t i;
    // size_t di;
    // size_t offset;
    TemporaryBufferHandle temporaryBuffer;
    size_t liveSends;
    size_t index;

    const std::vector<size_t>& ipcRanks = self.group->ipcRanks;
    const std::vector<size_t>& peerIndices = self.group->peerIndices;

    const AllGather& allGather = *self.group->allGather;
    size_t recvFrom = rank;
    size_t sendTo = rank;
    Vector<ProxyInfo*>* proxyInfo = nullptr;
    ProxyDestinationInfo* proxyDestination = nullptr;

    static const size_t numChunks = 8;
    const size_t chunkSize = params.bytes < (size_t)65536 * 96
                                 ? params.bytes
                                 : std::max((params.bytes + numChunks - 1) / numChunks, (size_t)65536 * 72);
    const size_t numParallel = std::max(std::min((params.bytes + chunkSize - 1) / chunkSize / 2, (size_t)2), (size_t)1);

    struct SendState {
      size_t chunkIndex;
      size_t liveSends;
    };
    const size_t nDevices = self.devices.size();
    std::array<SendState, maxDevices> sendStates;

    WorkBroadcastRingCpu(CpuThreadImpl& self, QueueEntryBroadcastCpu& params) : Work(self, params), params(params) {
      auto recvi = allGather.ringRecvsBySource.find(params.sourceRank);
      if (recvi != allGather.ringRecvsBySource.end()) {
        recvFrom = recvi->second;
      }
      auto sendi = allGather.ringSendsBySource.find(params.sourceRank);
      if (sendi != allGather.ringSendsBySource.end()) {
        sendTo = sendi->second;
      }
      auto pi = allGather.proxyInfoBySource.find(params.sourceRank);
      if (pi != allGather.proxyInfoBySource.end()) {
        proxyInfo = &pi->second;
      }
      auto pdi = allGather.proxyDestinationInfoBySource.find(params.sourceRank);
      if (pdi != allGather.proxyDestinationInfoBySource.end()) {
        proxyDestination = pdi->second;
      }
    }

    // void callback(size_t di, size_t source, size_t chunkIndex, size_t neighbor, bool ordered, bool done) {
    //   if (done) {
    //     --sendStates[di].liveSends;
    //   }
    //   if (ordered) {
    //     auto& dev = self.devices[di];
    //     self.writeData(
    //         dev, neighbor, sendStepValue, self.sendStepValues2StorageMr->mrs[di]->lkey,
    //         (void*)(self.remoteCpuCommsDeviceDataSent2[neighbor].address + sizeof(uint32_t) * (32 * 32 * size *
    //         concurrencyIndex + 32 * 32 * source + 32 * di + chunkIndex)),
    //         self.remoteCpuCommsDeviceDataSent2[neighbor].keys[di], sizeof(stepValue));
    //   }
    // }

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

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      if (rank != params.sourceRank && !self.group->ipcAccess[params.sourceRank]) {

        if (recvFrom != rank) {
          // log.info("I will receive the broadcast data directly from neighbor %d!", recvFrom);

          CHECK(proxyDestination == nullptr);

          temporaryBuffer = self.allocateTemporaryBuffer(params.bytes, self.broadcastBuffers);

          DynamicAddresses* outDyns = &self.outDyn(concurrencyIndex, 0);

          outDyns[0].gatherAddress = (uintptr_t)temporaryBuffer->cpuPointer;
          outDyns[0].gatherBytes = params.bytes;
          outDyns[0].stepValue = params.stepValue;
          outDyns[0].opType = opTypeBroadcastRingCpu;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[0].gatherKey[di] = temporaryBuffer.mr->mrs[di]->rkey;
          }

          self.writeDyn(concurrencyIndex, recvFrom, 0, rank);
        } else {
          CHECK(proxyInfo == nullptr);
          CHECK(proxyDestination != nullptr);
          CHECK(sendTo == rank);
          // log.info("I will receive the broadcast data indirectly from peer rank %d!", proxyDestination->proxy);
        }
      } else if (rank == params.sourceRank) {
        temporaryBuffer = self.allocateTemporaryBuffer(params.bytes, self.broadcastBuffers);
        std::memcpy(temporaryBuffer->cpuPointer, (void*)params.tensorAddress, params.bytes);
        CHECK(proxyInfo == nullptr);
        CHECK(proxyDestination == nullptr);
        CHECK(recvFrom == rank);
      } else {
        CHECK(self.group->ipcAccess[params.sourceRank]);
        CHECK(proxyInfo == nullptr);
        CHECK(proxyDestination == nullptr);
        CHECK(recvFrom == rank);
        CHECK(sendTo == rank);
      }

      self.group->cpuAddresses[concurrencyIndex].inputAddress = params.tensorAddress;
      self.group->cpuAddresses[concurrencyIndex].outputAddress = params.tensorAddress;
      self.group->cpuAddresses[concurrencyIndex].bytes = params.bytes;
      self.group->cpuAddresses[concurrencyIndex].pid = self.group->pid;

      liveSends = 0;

      if (sendTo != rank || recvFrom != rank) {

        for (size_t i = 0; i != nDevices; ++i) {
          sendStates[i].chunkIndex = 0;
          sendStates[i].liveSends = 0;
        }

        if (sendTo != rank) {
          WAIT_DYN(opTypeBroadcastRingCpu, sendTo);
        }

        while (true) {
          {
            size_t nDone = 0;
            for (size_t di = 0; di != nDevices; ++di) {
              auto& state = sendStates[di];
              auto& dev = self.devices[di];
              if (state.chunkIndex == numChunks) {
                if (state.liveSends) {
                  continue;
                }
                ++nDone;
                continue;
              }

              size_t chunkIndex = state.chunkIndex;
              size_t source = params.sourceRank;

              size_t currentChunkSize = std::min(chunkSize, params.bytes - chunkSize * chunkIndex);

              const size_t devChunkSize = currentChunkSize / nDevices / 1024u * 1024u;
              size_t offset = devChunkSize * di;
              size_t n = di == nDevices - 1 ? currentChunkSize - offset : devChunkSize;
              bool isLastChunk = chunkSize * chunkIndex + currentChunkSize == params.bytes;
              size_t sendOffset = chunkSize * chunkIndex + offset;
              if (sendTo != rank) {
                if (n == 0) {
                  self.writeStepValueDeviceChunk(concurrencyIndex, sendTo, 0, source, di, chunkIndex);
                } else {
                  if (source != rank) {
                    if (self.inStepValueDeviceChunk(concurrencyIndex, source, di, chunkIndex) != stepValue) {
                      continue;
                    }
                  }
                  uintptr_t srcAddr = (uintptr_t)temporaryBuffer->cpuPointer;
                  auto* srcMr = temporaryBuffer.mr;

                  const DynamicAddresses& dyn = self.inDyn(concurrencyIndex, sendTo);

                  ++state.liveSends;
                  uintptr_t dstAddr = dyn.gatherAddress;
                  auto key = dyn.gatherKey;
                  self.writeData(
                      dev, sendTo, (void*)(srcAddr + sendOffset), srcMr->mrs[di]->lkey, (void*)(dstAddr + sendOffset),
                      key[di], n, self.makeCallback([this, di, source, chunkIndex]() {
                        --sendStates[di].liveSends;
                        self.writeStepValueDeviceChunk(concurrencyIndex, sendTo, 0, source, di, chunkIndex);
                      }));
                }
              } else {
                CHECK(source != rank);
                if (self.inStepValueDeviceChunk(concurrencyIndex, source, di, chunkIndex) != stepValue) {
                  continue;
                }
              }
              std::memcpy(
                  (void*)(params.tensorAddress + sendOffset),
                  (void*)((uintptr_t)temporaryBuffer->cpuPointer + sendOffset), n);

              if (isLastChunk) {
                state.chunkIndex = numChunks;
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
      }
      self.group->cpuAddresses[concurrencyIndex].stepValue = stepValue;

      if (proxyDestination != nullptr || self.group->ipcAccess[params.sourceRank]) {
        index = proxyDestination ? proxyDestination->proxyPeerIndex : self.group->getPeerIndex(params.sourceRank);
        while (self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex])->stepValue < stepValue) {
          YIELD
        }
        auto* x = self.group->getPeerVar(index, &self.group->cpuAddresses[concurrencyIndex]);
        CHECK(x->bytes == params.bytes);
        vmcopy(params.tensorAddress, x->pid, x->inputAddress, params.bytes);
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
      self.allocatorBroadcastRingCpu.deallocate(this);
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
    uint32_t* sendStepValue;

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

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      if (rank == params.destinationRank) {
        CHECK(params.outputList.size() == size);
        size_t outputSize = 0;
        for (auto& v : params.outputList) {
          outputSize += (v.bytes + 63) / 64u * 64u;
        }
        outputBuffer = self.allocateTemporaryBuffer(outputSize, self.allGatherOutputBuffers);

        DynamicAddresses* outDyns = &self.outDyn(concurrencyIndex, 0);

        size_t offset = 0;
        for (size_t i = 0; i != size; ++i) {
          if (i == rank || self.group->ipcAccess[i]) {
            offset += (params.outputList[i].bytes + 63) / 64u * 64u;
            continue;
          }
          outDyns[i].gatherAddress = (uintptr_t)outputBuffer->cpuPointer + offset;
          outDyns[i].gatherBytes = params.outputList[i].bytes;
          for (size_t di = 0; di != self.devices.size(); ++di) {
            outDyns[i].gatherKey[di] = outputBuffer.mr->mrs[di]->rkey;
          }
          outDyns[i].opType = opTypeGatherCpu;
          outDyns[i].stepValue = stepValue;
          offset += (params.outputList[i].bytes + 63) / 64u * 64u;

          self.writeDyn(concurrencyIndex, i, i, rank);
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
          while (self.inStepValue(concurrencyIndex, i) != stepValue) {
            YIELD
          }
          DynamicAddresses* outDyns = &self.outDyn(concurrencyIndex, 0);
          CHECK(outDyns[i].gatherAddress == (uintptr_t)outputBuffer->cpuPointer + offset);
          CHECK(params.outputList[i].bytes == params.bytes);
          std::memcpy(
              (void*)params.outputList[i].address, (void*)((uintptr_t)outputBuffer->cpuPointer + offset), params.bytes);
          offset += (params.outputList[i].bytes + 63) / 64u * 64u;
        }
      } else if (!self.group->ipcAccess[params.destinationRank]) {

        i = params.destinationRank;
        WAIT_DYN(opTypeGatherCpu, i);

        uintptr_t srcAddr = (uintptr_t)inputBuffer->cpuPointer;

        ++liveSends;
        self.writeDataDistributed(
            i, srcAddr, params.bytes, inputBuffer.mr, dyn.gatherAddress, dyn.gatherKey,
            self.makeCallback([this, i = this->i] {
              --liveSends;
              self.writeStepValue(concurrencyIndex, i, 0, rank);
            }));
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
  WorkAllocator<WorkAllGatherRingRead4> allocatorAllGatherRingRead4;
  WorkAllocator<WorkReduceScatterRingRead> allocatorReduceScatterRingRead;
  WorkAllocator<WorkBroadcastRing> allocatorBroadcast;
  WorkAllocator<WorkAllGatherCpu> allocatorAllGatherCpu;
  WorkAllocator<WorkReduceScatterCpu> allocatorReduceScatterCpu;
  WorkAllocator<WorkGatherCpu> allocatorGatherCpu;
  WorkAllocator<WorkBroadcastRingCpu> allocatorBroadcastRingCpu;
  WorkAllocator<WorkReduceTree> allocatorReduce;

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

      bool terminate = false;
      while (!terminate) {
        if (cpuThread->queueSize.load(std::memory_order_relaxed) == 0) {
          if (activeWorks.empty() && activeDevices.empty()) {
            {
              std::unique_lock l(cpuThread->mutex);
              if (cpuThread->queueSize.load(std::memory_order_relaxed) != 0) {
                continue;
              }
              cpuThread->busy.store(false, std::memory_order_relaxed);
            }
            if (terminate) {
              break;
            }
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
              QueueEntryAllGather& e = (QueueEntryAllGather&)queueEntry;
              enqueue(allocatorAllGatherRingRead4, (QueueEntryAllGather&)queueEntry);
            } else if (queueEntry.task == taskReduceScatter) {
              enqueue(allocatorReduceScatterRingRead, (QueueEntryReduceScatter&)queueEntry);
            } else if (queueEntry.task == taskTerminate) {
              terminate = true;
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
              enqueue(allocatorBroadcastRingCpu, (QueueEntryBroadcastCpu&)queueEntry);
            } else if (queueEntry.task == taskInternalBarrier) {
              enqueue(allocatorInternalBarrier, (QueueEntryBarrier&)queueEntry);
            } else if (queueEntry.task == taskReduce) {
              enqueue(allocatorReduce, (QueueEntryReduce&)queueEntry);
            } else {
              throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
            }
          }
          if (terminate) {
            break;
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
    ++queueSize;
    futexWakeAll(&queueSize);

    thread.join();
  }
}

void CpuThread::start() {
  thread = std::thread([this] {
    async::setCurrentThreadName("moodist-cputhread");
    CHECK(numa_run_on_node(group->allocationNode) == 0);
    CHECK_CU(cuCtxSetCurrent(group->cuContext));
    CpuThreadImpl impl(this);
    impl.entry();
  });
}

} // namespace moodist
