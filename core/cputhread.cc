// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cputhread.h"
#include "allgather.h"
#include "api/allocator.h"
#include "api/cpu_allocator.h"
#include "async.h"
#include "common.h"
#include "fmt/printf.h"
#include "freelist.h"
#include "group.h"
#include "hash_map.h"
#include "ib_common.h"
#include "intrusive_list.h"
#include "ipc_mapper.h"
#include "logging.h"
#include "queue_internal.h"
#include "rdma.h"
#include "reduce_scatter.h"
#include "serialization.h"
#include "setup_comms.h"
#include "simple_vector.h"
#include "synchronization.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <thread>
#include <type_traits>
#include <utility>

namespace moodist {

#define Vector IVector

extern bool profilingEnabled;

namespace {

struct Device {
  bool active = false;
  IntrusiveListLink<Device> link;
  Rdma* rdma;

  // ibv_pd* protectionDomain;
  // ibv_cq* cq;
  // size_t currentCqEntries = 0;
  // bool efa;
  // size_t numCqEvents = 0;

  // TemporaryLkeyBuffers temporaryLkeyBuffers;

  // IVector<Function<void()>> queuedWrs;
};

using Callback = RdmaCallback;

struct RemoteAddressAndKey {
  uintptr_t address;
  SimpleVector<uint32_t> keys;
};

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

std::atomic_int nextThreadId;
thread_local int currentThreadId = ++nextThreadId;

struct MemoryRegistration {
  std::array<RdmaMr*, Group::maxDevices> mrs{};
};

struct CudaMapping {
  uint64_t bufferId;
  uintptr_t address;
  size_t bytes;
  MemoryRegistration mr;
};

struct CpuMapping {
  size_t bytes;
  MemoryRegistration mr;
};

struct TemporaryBufferHandle {
  AllocatedCpuBuffer buffer;
  MemoryRegistration* mr = nullptr;
  TemporaryBufferHandle() = default;
  TemporaryBufferHandle(AllocatedCpuBuffer buffer, MemoryRegistration* mr) : buffer(std::move(buffer)), mr(mr) {}
  TemporaryBufferHandle(TemporaryBufferHandle&& n) noexcept {
    *this = std::move(n);
  }
  TemporaryBufferHandle& operator=(TemporaryBufferHandle&& n) noexcept {
    std::swap(buffer, n.buffer);
    std::swap(mr, n.mr);
    return *this;
  }
  ~TemporaryBufferHandle() {}
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

  CpuThreadImpl(CpuThread* cpuThread) : cpuThread(cpuThread) {
    CHECK(devices.size() <= maxDevices);
  }
  ~CpuThreadImpl() {
    destroyReceives();
    for (auto& v : devices) {
      v.rdma->close();
    }
    localMrs.clear();
  }

  const size_t rank = group->rank;
  const size_t size = group->size;

  SetupComms* setupComms = &*group->setupComms;
  SimpleVector<Device> devices = initDevices();
  IntrusiveList<Device, &Device::link> activeDevices;

  size_t bytesWritten = 0;
  size_t bytesRead = 0;

  bool errorState = false;
  bool cqErrored = false;
  std::chrono::steady_clock::time_point errorTime;

  IVector<std::unique_ptr<RdmaMr>> localMrs;

  IVector<std::unique_ptr<CudaMapping>> cudaMappings;
  IVector<std::unique_ptr<CpuMapping>> cpuMappings;

  HashMap<uintptr_t, CpuMapping*> mrMap;
  HashMap<uintptr_t, CudaMapping*> mrMapCuda;

  MemoryRegistration* commsMr = regMr((uintptr_t)group->mySharedMem, group->mySharedMemSize);

  DynamicAddresses* inDyns = group->localDyns;
  DynamicAddresses* inCustomDyns = group->customDyns;
  // Progress* localProgress = group->localProgress;
  // Progress* localProgress2 = group->localProgress2;
  uint32_t* cpuStepValues = group->cpuStepValues;
  uint32_t* cpuStepValuesDeviceChunks = group->cpuStepValuesDeviceChunks;
  ArrivalValue* cpuArrivalValues = group->cpuArrivalValues;
  uint32_t* customNotifications = group->customNotifications;

  // uint32_t* cudaStepValues = group->cudaStepValues;
  // uint32_t* cudaStepValuesDeviceChunks = group->cudaStepValuesDeviceChunks;

  struct RankInfo {
    std::array<char, 64> hostname = {0};
    int pid = 0;
    int device = -1;
    size_t heartbeatValue = 0;
    bool shutdown = false;
    bool exited = false;
    std::chrono::steady_clock::time_point shutdownTime;
    std::chrono::steady_clock::time_point exitTime;
  };

  AllocatedArray rankInfo = group->allocateArrayHost(sizeof(RankInfo), size);
  MemoryRegistration* rankInfoMr = regMr((uintptr_t)rankInfo.buffer.cpuPointer, rankInfo.buffer.bytes);
  IVector<RemoteAddressAndKey> remoteRankInfo =
      distributeAddressAndKeys((uintptr_t)rankInfo.buffer.cpuPointer, rankInfoMr);

  MemoryRegistration* cudaStepValuesDeviceChunksMr = regMrCuda(
      group->cudaStepValuesDeviceChunksBuffer.buffer.cudaPointer, group->cudaStepValuesDeviceChunksBuffer.buffer.bytes);
  IVector<RemoteAddressAndKey> remoteCudaStepValuesDeviceChunks = distributeAddressAndKeys(
      group->cudaStepValuesDeviceChunksBuffer.buffer.cudaPointer, cudaStepValuesDeviceChunksMr);

  AllocatedArray internalInStepValueBuffer =
      group->allocateArrayHost(sizeof(uint32_t) * 2 * size, Group::maxConcurrency);
  MemoryRegistration* internalInStepValueBufferMr =
      regMr((uintptr_t)internalInStepValueBuffer.buffer.cpuPointer, internalInStepValueBuffer.buffer.bytes);
  IVector<RemoteAddressAndKey> remoteInternalInStepValueBuffer =
      distributeAddressAndKeys((uintptr_t)internalInStepValueBuffer.buffer.cpuPointer, internalInStepValueBufferMr);
  SimpleVector<uint32_t> nextInternalStepValue = SimpleVector<uint32_t>(Group::maxConcurrency);

  // MemoryRegistration* cudaStepValueMr =
  //     regMrCuda(group->cudaStepValue.buffer.cudaPointer, group->cudaStepValue.buffer.bytes);

  IVector<RemoteAddressAndKey> remoteComms = distributeAddressAndKeys((uintptr_t)group->mySharedMem, commsMr);
  // IVector<RemoteAddressAndKey> remoteCudaStepValue =
  //     distributeAddressAndKeys(group->cudaStepValue.buffer.cudaPointer, cudaStepValueMr);

  // MemoryRegistration* cudaCommsDeviceDataSentMr =
  //     regMrCuda(group->cudaCommsDeviceDataSent.buffer.cudaPointer, group->cudaCommsDeviceDataSent.buffer.bytes);
  // IVector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent =
  //     distributeAddressAndKeys(group->cudaCommsDeviceDataSent.buffer.cudaPointer, cudaCommsDeviceDataSentMr);

  // MemoryRegistration* cpuCommsDeviceDataSentMr =
  //     regMr((uintptr_t)group->cpuCommsDeviceDataSent.buffer.cpuPointer, group->cpuCommsDeviceDataSent.buffer.bytes);
  // IVector<RemoteAddressAndKey> remoteCpuCommsDeviceDataSent =
  //     distributeAddressAndKeys((uintptr_t)group->cpuCommsDeviceDataSent.buffer.cpuPointer, cpuCommsDeviceDataSentMr);

  // MemoryRegistration* cudaCommsDeviceDataSent2Mr =
  //     regMrCuda(group->cudaCommsDeviceDataSent2.buffer.cudaPointer, group->cudaCommsDeviceDataSent2.buffer.bytes);
  // IVector<RemoteAddressAndKey> remoteCudaCommsDeviceDataSent2 =
  //     distributeAddressAndKeys(group->cudaCommsDeviceDataSent2.buffer.cudaPointer, cudaCommsDeviceDataSent2Mr);

  // MemoryRegistration* cpuCommsDeviceDataSent2Mr =
  //     regMr((uintptr_t)group->cpuCommsDeviceDataSent2.buffer.cpuPointer,
  //     group->cpuCommsDeviceDataSent2.buffer.bytes);
  // IVector<RemoteAddressAndKey> remoteCpuCommsDeviceDataSent2 =
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

  // The values that compile_op's CPU thread work writes into the memory of other ranks and of its own GPU: per
  // concurrency slot, the op's step value + generation for every generation (see compile_op::Graph::Signal), in
  // maxDevices copies (a signal is a word per NIC, see writeRdmaSignal); WorkCustom sets those of its op's
  // generations. The writes from them are control transfers: the slot's next work changes them only once those have
  // completed (see WAIT_CONTROL).
  AllocatedArray generationValuesStorage =
      group->allocateArrayHost(sizeof(uint32_t) * maxDevices * compile_op::maxGenerations, Group::maxConcurrency);
  MemoryRegistration* generationValuesMr =
      regMr((uintptr_t)generationValuesStorage.buffer.cpuPointer, generationValuesStorage.buffer.bytes);

  SimpleVector<size_t> concurrencyLiveControlTransfers = SimpleVector<size_t>(Group::maxConcurrency);

  AllocatedArray outDynsBuffer = group->allocateArrayHost(sizeof(DynamicAddresses) * size, Group::maxConcurrency);
  MemoryRegistration* outDynsMr = regMr((uintptr_t)outDynsBuffer.buffer.cpuPointer, outDynsBuffer.buffer.bytes);

  // Vector<uint8_t> dynReadyVector{size * Group::maxConcurrency};
  Vector<MemoryRegistration*> outDynsMrVector{size * Group::maxConcurrency};

  MemoryRegistration* cpuOutMr = regMrCuda(group->cpuOutBuffer.buffer.cudaPointer, group->cpuOutBuffer.buffer.bytes);

  // compile_op's RDMA signals (Group::rdmaSignalBuffer, see writeRdmaSignal): a registration of their own, without
  // relaxed ordering; and where the other ranks' are (this rank writes them when it writes data into their tensors)
  MemoryRegistration* rdmaSignalMr =
      regMrCudaStrict(group->rdmaSignalBuffer.buffer.cudaPointer, group->rdmaSignalBuffer.buffer.bytes);
  IVector<RemoteAddressAndKey> remoteRdmaSignals =
      distributeAddressAndKeys(group->rdmaSignalBuffer.buffer.cudaPointer, rdmaSignalMr);
  // Whether signals can follow their data without flushes (see writeRdmaSignal)
  const bool orderedSignals = initOrderedSignals();

  // Where flush reads put the byte they read (see flushRead)
  AllocatedBuffer flushBuffer = group->allocateHost(4096);
  MemoryRegistration* flushMr = regMr((uintptr_t)flushBuffer.cpuPointer, flushBuffer.bytes);

  // Vector<uint32_t> debugVector{size};

  // size_t uniqueIndexOffset = 0;
  // const size_t numUniqueIndices = std::max((size_t)1024, size * 2);

  // AllocatedArray uOutDynsBuffer =
  //     group->allocateArrayHost(sizeof(DynamicAddresses) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uOutDynsMr = regMr((uintptr_t)uOutDynsBuffer.buffer.cpuPointer, uOutDynsBuffer.buffer.bytes);
  // AllocatedArray uInDynsBuffer =
  //     group->allocateArrayHost(sizeof(DynamicAddresses) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uInDynsMr = regMr((uintptr_t)uInDynsBuffer.buffer.cpuPointer, uInDynsBuffer.buffer.bytes);
  // IVector<RemoteAddressAndKey> remoteUInDyns =
  //     distributeAddressAndKeys((uintptr_t)uInDynsBuffer.buffer.cpuPointer, uInDynsMr);

  // AllocatedArray uOutStepValuesBuffer =
  //     group->allocateArrayHost(sizeof(uint32_t) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uStepValueMr =
  //     regMr((uintptr_t)uOutStepValuesBuffer.buffer.cpuPointer, uOutStepValuesBuffer.buffer.bytes);
  // AllocatedArray uInStepValuesBuffer =
  //     group->allocateArrayHost(sizeof(uint32_t) * numUniqueIndices, Group::maxConcurrency);
  // MemoryRegistration* uInStepValuesMr =
  //     regMr((uintptr_t)uInStepValuesBuffer.buffer.cpuPointer, uInStepValuesBuffer.buffer.bytes);
  // IVector<RemoteAddressAndKey> remoteUInStepValues =
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
  // compile_op's tensor lists (Group::customDyns): rank `index`'s, in this rank's memory
  DynamicAddresses& inCustomDyn(size_t concurrencyIndex, size_t index) {
    CHECK(index < size);
    return inCustomDyns[size * concurrencyIndex + index];
  }

  uint32_t& outStepValue(size_t concurrencyIndex, size_t index) {
    CHECK(index < 32);
    return outStepValues[32u * concurrencyIndex + index];
  }
  // Copy `word` of the value of generation `generation` (see generationValuesStorage)
  uint32_t& generationValue(size_t concurrencyIndex, size_t generation, size_t word) {
    CHECK(generation < compile_op::maxGenerations && word < maxDevices);
    return ((uint32_t*)generationValuesStorage.buffer
            .cpuPointer)[maxDevices * (compile_op::maxGenerations * concurrencyIndex + generation) + word];
  }
  uint32_t& inStepValue(size_t concurrencyIndex, size_t index) {
    CHECK(index < size);
    return cpuStepValues[size * concurrencyIndex + index];
  }
  // compile_op's arrival values (Group::cpuArrivalValues): rank `index`'s, in this rank's memory
  ArrivalValue& inArrivalValue(size_t concurrencyIndex, size_t index) {
    CHECK(index < size);
    return cpuArrivalValues[size * concurrencyIndex + index];
  }
  // (both words in one load: they come in one write; lock-free, as a lock would not keep the NIC out)
  ArrivalValue loadArrivalValue(size_t concurrencyIndex, size_t index) {
    static_assert(std::atomic_ref<ArrivalValue>::is_always_lock_free &&
                  std::atomic_ref<ArrivalValue>::required_alignment <= alignof(ArrivalValue));
    return std::atomic_ref(inArrivalValue(concurrencyIndex, index)).load(std::memory_order_relaxed);
  }
  // compile_op's notifications (Group::customNotifications): slot `slot` of rank `index`'s, in this rank's memory
  uint32_t& inNotification(size_t concurrencyIndex, size_t index, size_t slot) {
    CHECK(index < size && slot < Group::numNotificationSlots);
    return customNotifications[Group::numNotificationSlots * (size * concurrencyIndex + index) + slot];
  }

  size_t stepValueDeviceChunkIndex(size_t concurrencyIndex, size_t index, size_t device, size_t chunk) {
    return Group::maxChunks * Group::maxDevices * size * concurrencyIndex + Group::maxChunks * size * device +
           size * chunk + index;
  }

  uint32_t& inStepValueDeviceChunk(size_t concurrencyIndex, size_t index, size_t device, size_t chunk) {
    CHECK(index < size);
    return cpuStepValuesDeviceChunks[stepValueDeviceChunkIndex(concurrencyIndex, index, device, chunk)];
  }

  TemporaryBufferHandle allocateTemporaryBuffer(size_t bytes) {
    // CHECK(bytes != 0);
    auto buffer = group->allocateCpu(bytes);
    // auto region = cpu_allocator::regionAt(buffer.cpuPointer);
    // CHECK(region.first != 0);
    // auto* mr = regMr(region.first, region.second);
    auto* mr = regMr((uintptr_t)buffer.cpuPointer, bytes);
    return TemporaryBufferHandle(std::move(buffer), mr);
  }

  SimpleVector<Device> initDevices() {
    SimpleVector<Device> devices;
    devices.resize(group->rdmaDevs.size());
    for (size_t i : indices(group->rdmaDevs)) {
      auto& dev = devices[i];

      RdmaCpuThreadApi cpuThreadApi;
      cpuThreadApi.impl = this;
      cpuThreadApi.context = &dev;

      dev.rdma = &*group->rdmaDevs[i];
      dev.rdma->setCpuThreadApi(cpuThreadApi);
    }
    return devices;
  }

  template<typename F>
  auto makeCallback(F&& f) {
    return makeRdmaCallback(std::forward<F>(f));
  }

  void setErrorState() {
    if (errorState) {
      return;
    }
    errorState = true;
    errorTime = std::chrono::steady_clock::now();
    log.error(
        "The process group is now in an error state, and Moodist will shortly forcefully terminate the process.\n");
  }

  bool pollCalledRecursively = false;

  void poll() {
    CHECK(!pollCalledRecursively);
    pollCalledRecursively = true;
    for (auto i = activeDevices.begin(); i != activeDevices.end();) {
      auto& dev = *i++;
      CHECK(dev.active);
      dev.rdma->poll();
    }
    pollCalledRecursively = false;
  }

  void postWrite(Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
      size_t bytes, Callback* callback, bool allowInline) {
    // log.info(
    //     "%d: rdma write %d bytes (%p -> %p, lkey %#x, rkey %#x) (dev %d, i %d)  (cb %#x)\n", rank, bytes,
    //     localAddress, remoteAddress, lkey, rkey, &dev - devices.data(), i, (uint64_t)(void*)callback);
    dev.rdma->postWrite(i, localAddress, lkey, remoteAddress, rkey, bytes, callback, allowInline);
  }

  void writeData(Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
      size_t bytes, Callback* callback, bool allowInline = true) {
    CHECK(i >= 0 && i < size);

    callback->i = i;

    const size_t threshold = 1024ull * 1024 * 896;
    const size_t chunkSize = 1024ull * 1024 * 768;
    if (bytes >= threshold) {
      size_t remaining = bytes;
      bytes = chunkSize;
      while (remaining > 0) {
        if (remaining < threshold) {
          bytes = remaining;
        }
        postWrite(dev, i, localAddress, lkey, remoteAddress, rkey, bytes, callback, allowInline);

        localAddress = (void*)((uintptr_t)localAddress + bytes);
        remoteAddress = (void*)((uintptr_t)remoteAddress + bytes);
        remaining -= bytes;
      }
    } else {
      postWrite(dev, i, localAddress, lkey, remoteAddress, rkey, bytes, callback, allowInline);
    }
  }

  void writeDataNoInline(Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
      size_t bytes, Callback* callback) {
    return writeData(dev, i, localAddress, lkey, remoteAddress, rkey, bytes, callback, false);
  }

  void postRead(Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
      size_t bytes, Callback* callback) {
    dev.rdma->postRead(i, localAddress, lkey, remoteAddress, rkey, bytes, callback);
  }

  void readData(Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
      size_t bytes, Callback* callback) {
    CHECK(i >= 0 && i < size);

    // Packets are cut from the start of the read, so if the destination does not start at a page boundary, every
    // packet straddles two pages of it, which the NIC handles much more slowly (GB300, 1 GiB read: 11.3 ms with the
    // destination at a page boundary, 14.3 ms with it 1 KiB into a page, 16.5 ms with the source 1 KiB into a page
    // too; a source alone off a boundary costs nothing). Large reads read their start up to the next page boundary
    // separately. The callback runs once all parts have completed (postRead counts them).
    constexpr size_t pageSize = 4096;
    constexpr size_t alignFrom = 64 * 1024;
    if (size_t head = -(uintptr_t)localAddress & (pageSize - 1); head != 0 && bytes >= alignFrom) {
      postRead(dev, i, localAddress, lkey, remoteAddress, rkey, head, callback);
      localAddress = (void*)((uintptr_t)localAddress + head);
      remoteAddress = (void*)((uintptr_t)remoteAddress + head);
      bytes -= head;
    }

    const size_t threshold = 1024ull * 1024 * 896;
    const size_t chunkSize = 1024ull * 1024 * 768;
    if (bytes >= threshold) [[unlikely]] {
      size_t remaining = bytes;
      bytes = chunkSize;
      while (remaining > 0) {
        if (remaining < threshold) {
          bytes = remaining;
        }
        postRead(dev, i, localAddress, lkey, remoteAddress, rkey, bytes, callback);

        localAddress = (void*)((uintptr_t)localAddress + bytes);
        remoteAddress = (void*)((uintptr_t)remoteAddress + bytes);
        remaining -= bytes;
      }
    } else {
      postRead(dev, i, localAddress, lkey, remoteAddress, rkey, bytes, callback);
    }
  }

  // A small write whose completion is counted in the concurrency slot's live control transfers (which the next work on
  // the slot waits for, see ENTER), or in *counter instead. With controlQp, through the peer's control queue pair (see
  // Rdma::postWriteControl): ordered only with the other such writes to rank i through dev.
  void writeControl(size_t concurrencyIndex, Device& dev, size_t i, void* localAddress, uint32_t lkey,
      void* remoteAddress, uint32_t rkey, size_t bytes, size_t* counter = nullptr, bool controlQp = false) {
    CHECK(i >= 0 && i < size);

    auto callbackobj = makeCallback(nullptr);
    Callback* callback = callbackobj;
    callback->counter = counter ? counter : &concurrencyLiveControlTransfers[concurrencyIndex];
    ++*callback->counter;
    if (controlQp) {
      dev.rdma->postWriteControl(i, localAddress, lkey, remoteAddress, rkey, bytes, callback);
    } else {
      postWrite(dev, i, localAddress, lkey, remoteAddress, rkey, bytes, callback, true);
    }
  }

  void postSend(Device& dev, size_t i, void* localAddress, uint32_t lkey, size_t bytes, Callback* callback) {
    dev.rdma->postSend(i, localAddress, lkey, bytes, callback);
  }

  void postRecv(Device& dev, void* localAddress, uint32_t lkey, size_t bytes, Callback* callback) {
    dev.rdma->postRecv(localAddress, lkey, bytes, callback);
  }

  void readDataDistributed(size_t i, void* localAddress, const std::array<uint32_t, maxDevices>& lkeys,
      void* remoteAddress, const std::array<uint32_t, maxDevices>& rkeys, size_t bytes, Callback* callback = nullptr) {
    size_t numDevices = std::min((bytes + 262143) / 262144, std::max((size_t)4, group->numTrueIbDevs));
    numDevices = std::min(numDevices, devices.size());
    size_t numChunks = numDevices;
    size_t chunkSize = ((bytes + numChunks - 1) / numChunks + 4095u) / 4096u * 4096u;
    CHECK(devices.size() >= numDevices);
    for (size_t di = 0; di != devices.size(); ++di) {
      auto& dev = devices[di];
      size_t offset = chunkSize * di;
      size_t n = std::min(bytes - offset, chunkSize);
      readData(dev, i, (void*)((uintptr_t)localAddress + offset), lkeys[di], (void*)((uintptr_t)remoteAddress + offset),
          rkeys[di], n, callback);
      if (offset + n == bytes) {
        break;
      }
    }
  }

  using seq_t = uint32_t;

  bool receivesInitialized = false;
  Vector<seq_t> incomingSeqId;
  Vector<seq_t> outgoingSeqId;
  Vector<Vector<std::pair<seq_t, TemporaryBufferHandle>>> incomingSeqQueued;

  struct Header {
    uint32_t type;
    uint32_t source;
    seq_t seqid;
    template<typename X>
    void serialize(X& x) {
      x(type, source, seqid);
    }
  };

  Header header(uint32_t op, uint32_t destination) {
    return Header{op, (uint32_t)rank, outgoingSeqId[destination]++};
  }

  void onRecv(Device& dev, TemporaryBufferHandle buffer) {
    try {
      std::string_view view((const char*)buffer->cpuPointer, buffer->bytes);
      uint32_t type;
      uint32_t source;
      seq_t seqid;
      view = deserializeBufferPart(view, type, source, seqid);
      CHECK(source < size);
      auto& queue = incomingSeqQueued[source];
      seq_t& expectedSeqId = incomingSeqId[source];
      if (seqid != expectedSeqId) {
        uint32_t expectedValue = expectedSeqId;
        queue.emplace(std::ranges::lower_bound(queue, seqid,
                          [expectedValue](auto& a, auto& b) {
                            auto p = [expectedValue](auto& v) {
                              if constexpr (std::is_same_v<std::decay_t<decltype(v)>, seq_t>) {
                                return seq_t(v - expectedValue);
                              } else {
                                return seq_t(v.first - expectedValue);
                              }
                            };
                            return p(a) < p(b);
                          }),
            seqid, std::move(buffer));
        enqueueCallback([this, &dev]() {
          postRecv(dev, allocateTemporaryBuffer(4096));
        });
        return;
      }
      bool first = true;
      while (true) {
        ++expectedSeqId;
        switch (type) {
        case opTypeQueuePut:
          onRecvQueuePut(view, source);
          break;
        case opTypeQueueReadFinished:
          onRecvQueueReadFinished(view);
          break;
        case opTypeQueueGet:
          onRecvQueueGet(view, source);
          break;
        case opTypeQueueTransaction:
          onRecvQueueTransaction(view, source);
          break;
        case opTypeCreateQueueNamed:
          onRecvCreateQueueNamed(view, source);
          break;
        case opTypeCreateQueueNamedResult:
          onRecvCreateQueueNamedResult(view);
          break;
        case opTypeMessageShutdown:
          onRecvMessageShutdown(view, source);
          break;
        case opTypeFabricMap:
          onRecvFabricMap(view, source);
          break;
        case opTypeFabricMapDone:
          onRecvFabricMapDone(view, source);
          break;
        default:
          fatal("Received unknown message type %#x", type);
        }
        if (first) {
          first = false;
          postRecv(dev, std::move(buffer));
        }
        if (queue.empty()) {
          break;
        }
        auto& x = queue.front();
        if (x.first != expectedSeqId) {
          break;
        }
        buffer = std::move(x.second);
        queue.pop_front();
        uint32_t tmpSource;
        view = std::string_view((const char*)buffer->cpuPointer, buffer->bytes);
        view = deserializeBufferPart(view, type, tmpSource, seqid);
        CHECK(tmpSource == source);
        CHECK(seqid == expectedSeqId);
      }

    } catch (const std::exception& e) {
      fatal("onRecv exception: %s", e.what());
    }
  }

  void postRecv(Device& dev, TemporaryBufferHandle buffer) {
    size_t di = &dev - devices.data();
    void* pointer = buffer->cpuPointer;
    size_t bytes = buffer->bytes;
    uint32_t lkey = buffer.mr->mrs[di]->lkey;
    postRecv(dev, pointer, lkey, bytes, makeCallback([this, &dev, buffer = std::move(buffer)]() mutable {
      onRecv(dev, std::move(buffer));
    }));
    // fixme? on shutdown, work requests are not completed, so their callbacks are leaked.
    //        do we care?
  }

  void initReceives() {
    if (receivesInitialized) {
      return;
    }
    receivesInitialized = true;

    incomingSeqId.resize(size);
    outgoingSeqId.resize(size);
    incomingSeqQueued.resize(size);

    for (size_t di = 0; di != devices.size(); ++di) {
      auto& dev = devices[di];
      for (size_t i = 0; i != IbCommon::maxSrqEntries; ++i) {
        postRecv(dev, allocateTemporaryBuffer(4096));
      }
    }
  }

  void destroyReceives() {
    if (!receivesInitialized) {
      return;
    }
    for (size_t di = 0; di != devices.size(); ++di) {
      auto& dev = devices[di];
      dev.rdma->stopReceives();
    }
  }

  MemoryRegistration nullRegistration;

  MemoryRegistration* regMr(uintptr_t address, size_t bytes) {
    if (bytes == 0) [[unlikely]] {
      return &nullRegistration;
    }
    auto region = cpu_allocator::regionAt(address);
    if (region.first) [[likely]] {
      uintptr_t inputBegin = address;
      uintptr_t inputEnd = address + bytes;
      CHECK(address + bytes <= region.first + region.second);
      size_t regionEnd = region.first + region.second;
      uintptr_t alignment = 1024ull * 1024 * 1024;
      uintptr_t roundedDown = address / alignment * alignment;
      size_t originalEnd = address + bytes;
      address = std::max(roundedDown, region.first);
      bytes = originalEnd - address;
      size_t roundedUp = (bytes + alignment - 1) / alignment * alignment;
      if (address + roundedUp <= regionEnd) {
        bytes = roundedUp;
      } else {
        bytes = regionEnd - address;
      }
      CHECK(address <= inputBegin && address + bytes >= inputEnd);
    }
    auto i = mrMap.find(address);
    if (i != mrMap.end()) [[likely]] {
      if (i->second->bytes >= bytes) [[likely]] {
        return &i->second->mr;
      }
    }
    auto ptr = std::make_unique<CpuMapping>();
    ptr->bytes = bytes;
    ptr->mr.mrs.fill(nullptr);
    CHECK(devices.size() <= ptr->mr.mrs.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (devices[i2].rdma->mrCompatible(devices[i].rdma)) {
          ptr->mr.mrs[i] = ptr->mr.mrs[i2];
          break;
        }
      }
      if (ptr->mr.mrs[i]) {
        continue;
      }
      auto start = std::chrono::steady_clock::now();
      auto mr = devices[i].rdma->regMrCpu((void*)address, bytes);
      ptr->mr.mrs[i] = &*mr;
      log.debug("new cpu mapped range of %d bytes at %#x (mr lkey %#x rkey %#x) (registration took %gs)\n", bytes,
          address, mr->lkey, mr->rkey, seconds(std::chrono::steady_clock::now() - start));
      localMrs.push_back(std::move(mr));
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
    if (bytes == 0) [[unlikely]] {
      return &nullRegistration;
    }
    unsigned long long bufferId = -1;
    CUdeviceptr base = 0;
    size_t size = 0;

    auto myRegion = allocator::mappedRegion(address);

    if (myRegion.first || bytes == 0) {
      bufferId = myRegion.first;
      base = myRegion.first;
      size = myRegion.second;
    } else {
      CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
      CHECK(bufferId != -1);

      unsigned long long bufferId2 = -1;
      CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + bytes - 1));
      CHECK(bufferId == bufferId2);

      CHECK_CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)address));
    }

    auto i = mrMapCuda.find(bufferId);
    if (i != mrMapCuda.end()) {
      if (i->second->address <= address && i->second->address + i->second->bytes >= address + bytes) {
        return &i->second->mr;
      }
    }

    address = base;
    bytes = size;

    auto ptr = std::make_unique<CudaMapping>();
    ptr->bufferId = bufferId;
    ptr->address = address;
    ptr->bytes = bytes;
    ptr->mr.mrs.fill(nullptr);
    CHECK(devices.size() <= ptr->mr.mrs.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (devices[i2].rdma->mrCompatible(devices[i].rdma)) {
          ptr->mr.mrs[i] = ptr->mr.mrs[i2];
          break;
        }
      }
      if (ptr->mr.mrs[i]) {
        continue;
      }
      if (bytes == 0) {
        break;
      }
      auto mr = devices[i].rdma->regMrCuda((void*)address, bytes);
      ptr->mr.mrs[i] = &*mr;
      log.debug("new cuda mapped range of %d bytes at %#x -> fd %d  (mr lkey %#x rkey %#x)\n", bytes, address, bufferId,
          mr->lkey, mr->rkey);
      localMrs.push_back(std::move(mr));
    }
    // ptr can be destructed here, but r is a plain pointer
    // fixme: return some kind of handle instead?
    MemoryRegistration* r = &ptr->mr;
    mrMapCuda[bufferId] = &*ptr;
    cudaMappings.push_back(std::move(ptr));
    return r;
  }

  // A registration of exactly [address, address + bytes) of CUDA memory (whole host pages), without relaxed ordering
  // (see Rdma::regMrCuda), apart from regMrCuda's (whose registrations of the same memory allow relaxed ordering)
  MemoryRegistration* regMrCudaStrict(uintptr_t address, size_t bytes) {
    auto ptr = std::make_unique<CudaMapping>();
    ptr->address = address;
    ptr->bytes = bytes;
    ptr->mr.mrs.fill(nullptr);
    for (size_t i = 0; i != devices.size(); ++i) {
      for (size_t i2 = 0; i2 != i; ++i2) {
        if (devices[i2].rdma->mrCompatible(devices[i].rdma)) {
          ptr->mr.mrs[i] = ptr->mr.mrs[i2];
          break;
        }
      }
      if (ptr->mr.mrs[i]) {
        continue;
      }
      auto mr = devices[i].rdma->regMrCuda((void*)address, bytes, false);
      ptr->mr.mrs[i] = &*mr;
      localMrs.push_back(std::move(mr));
    }
    MemoryRegistration* r = &ptr->mr;
    cudaMappings.push_back(std::move(ptr));
    return r;
  }

  // Signals can follow their data through the NICs that carried it, without flushes (see writeRdmaSignal), if the GPU
  // consumes the writes into its memory in the order they arrive (GPUDirect RDMA writes ordering "owner"), unless
  // MOODIST_RDMA_FLUSH=1
  bool initOrderedSignals() {
    int ordering = CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE;
    CHECK_CU(cuDeviceGetAttribute(&ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, group->cuDevice));
    const char* env = std::getenv("MOODIST_RDMA_FLUSH");
    const bool r = ordering >= CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER && !(env && std::string_view(env) == "1");
    log.debug("rank %zu: GPUDirect RDMA writes ordering %d: RDMA signals %s\n", rank, ordering,
        r ? "follow their data" : "after flushes");
    return r;
  }

  IVector<RemoteAddressAndKey> distributeAddressAndKeys(uintptr_t address, MemoryRegistration* mr) {
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
    IVector<RemoteAddressAndKey> r;
    r.resize(size);
    for (size_t i = 0; i != size; ++i) {
      r[i].address = addrs[i];
      r[i].keys = keys[i];
    }
    return r;
  }

  void writeDataDistributed(size_t dst, uintptr_t srcAddr, size_t len, MemoryRegistration* mr, uint64_t dstAddr,
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

  // The time limit of the setup's connection tests (RDMA writes to every rank): longer than the transport takes to
  // fail a write to a peer that does not answer (ibTransportTimeout), so that such a write fails with the transport's
  // error, which names the peer; at least 60 s, at most the process group's timeout. The tests also end as soon as
  // the setup has failed on another rank (see SetupComms::fail).
  struct TimeoutHelper {
    const Group* group;
    const char* what;
    std::chrono::steady_clock::duration timeout;
    std::chrono::steady_clock::time_point start;
    TimeoutHelper(const Group* group, const char* what) : group(group), what(what) {
      const auto transport = ibTransportTimeout();
      timeout = transport >= group->timeout ? group->timeout
                                            : std::min<std::chrono::steady_clock::duration>(group->timeout,
                                                  std::max<std::chrono::steady_clock::duration>(
                                                      std::chrono::seconds(60), transport + std::chrono::seconds(30)));
      start = std::chrono::steady_clock::now();
    }
    void check() {
      group->setupComms->checkFailed();
      if (std::chrono::steady_clock::now() - start >= timeout) {
        throw std::runtime_error(fmt::sprintf("Moodist timed out after %g s in %s: some RDMA writes to other ranks did "
                                              "not complete (some process may have failed early in its setup)",
            seconds(timeout), what));
      }
    }
  };

  void runTests() {
    TimeoutHelper timeout(group, "its connection test");
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
      // (shared: after an error, the writes may complete after this function has returned)
      auto remaining = std::make_shared<size_t>(0);
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        size_t offset = 512 * rank;
        for (size_t di = 0; di != devices.size(); ++di) {
          auto& dev = devices[di];
          size_t n = 512 / devices.size();
          ++*remaining;
          writeData(dev, i, testPtr + offset, testMr->mrs[di]->lkey, (uint64_t*)testRemote[i].address + offset,
              testRemote[i].keys[di], n * 8, makeCallback([remaining]() {
                --*remaining;
              }));
          offset += n;
        }
      }
      while (*remaining && !errorState) {
        poll();
        timeout.check();
      }
      // Every rank takes part in the allgather, with whether it has failed: a rank that skipped it left the others
      // waiting in it forever (a retry-exceeded error between two ranks of a 64-rank group hung the other 62).
      // Then every rank is in the error state.
      std::vector<uint8_t> failed = setupComms->allgather((uint8_t)errorState);
      std::string failedRanks;
      for (size_t i = 0; i != size; ++i) {
        if (failed.at(i)) {
          failedRanks += fmt::sprintf("%s%zu", failedRanks.empty() ? "" : ", ", i);
        }
      }
      if (!failedRanks.empty()) {
        log.error("%s: the connection test failed on rank(s) %s\n", groupName(), failedRanks);
        setErrorState();
        break;
      }
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
    if (std::ranges::all_of(devices,
            [](auto& v) {
              return v.rdma->supportsCuda();
            }) &&
        false) {
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
            writeDataNoInline(dev, i, testPtr + offset, testMr->mrs[di]->lkey,
                (uint64_t*)testRemote[i].address + offset, testRemote[i].keys[di], n * 8, makeCallback([&]() {
                  --remaining;
                }));
            offset += n;
          }
        }
        while (remaining && !errorState) {
          poll();
          timeout.check();
        }
        if (errorState) {
          break;
        }
        setupComms->allgather(0);

        log.debug("rank %d CUDA test done!\n", rank);
      }
    }
  }

  size_t opCount = 0;

  struct TraceEvent {
    std::string name;
    std::chrono::system_clock::time_point begin;
    std::chrono::system_clock::time_point end;
    int threadId = -1;
  };
  Vector<TraceEvent> traceEvents;
  // Trace thread ids of events outside the per-work spans (those use 10 * concurrency index + sub-thread)
  static constexpr int traceTidPickup = 9998;
  static constexpr int traceTidLoop = 9999;

  std::chrono::system_clock::time_point beginning = std::chrono::system_clock::now();

  HashMap<int, std::string> currentTraceName;
  HashMap<int, std::chrono::system_clock::time_point> currentTraceBegin;
  int pidForTracing = syscall(SYS_gettid);

  template<typename... Args>
  [[gnu::noinline]] [[gnu::cold]] void trace_outlined(int threadId, int subThreadId, const char* fmt, Args&&... args) {
    threadId = 10 * threadId + subThreadId;
    if (!profilingEnabled) {
      if (!traceEvents.empty()) {
        std::string fn = fmt::sprintf("moodist-trace-cpu-%s-%d.json", group->name, rank);
        FILE* f = fopen(fn.c_str(), "wb");
        CHECK(f != nullptr);
        fmt::fprintf(f, R"({"traceEvents": [)"
                        "\n");
        bool first = true;
        for (auto& e : traceEvents) {
          if (!first) {
            fmt::fprintf(f, ",\n");
          } else {
            first = false;
          }
          fmt::fprintf(f, R"({"ph": "X", "name": "%s", "ts": %d, "dur": %d, "tid": %d, "pid": %d})", e.name,
              std::chrono::duration_cast<std::chrono::microseconds>(e.begin.time_since_epoch()).count(),
              std::chrono::duration_cast<std::chrono::microseconds>(e.end - e.begin).count(), e.threadId,
              pidForTracing);
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
    std::string name = fmt::sprintf(fmt, std::forward<Args>(args)...);
    if (currentTraceName[threadId] == name) {
      return;
    }
    if (!currentTraceName[threadId].empty()) {
      traceEvents.emplace_back();
      TraceEvent& e = traceEvents.back();
      e.name = std::move(currentTraceName[threadId]);
      e.begin = currentTraceBegin[threadId];
      e.end = now;
      e.threadId = threadId;
    }
    currentTraceBegin[threadId] = now;
    currentTraceName[threadId] = name;
  }

  template<typename... Args>
  [[gnu::always_inline]] void trace(int threadId, int subThreadId, const char* fmt, Args&&... args) {
    if (profilingEnabled | !traceEvents.empty()) [[unlikely]] {
      trace_outlined(threadId, subThreadId, fmt, std::forward<Args>(args)...);
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
      return [](Work* w) {
        ((T*)w)->step();
      };
    }
  };

#define ENTER                                                                                                          \
  {                                                                                                                    \
    if (this->state) {                                                                                                 \
      goto* this->state;                                                                                               \
    }                                                                                                                  \
    WAIT_CONTROL                                                                                                       \
  }
  // clang-format off
// #define YIELD                                                                                                          \
//   if (self.numActiveWorks != 1 || self.cpuThread->queueSize.load(std::memory_order_relaxed) != 0) {                    \
//     this->state = &&CAT(s, __LINE__);                                                                                  \
//     return;                                                                                                            \
//     CAT(s, __LINE__) :;                                                                                                \
//   } else {                                                                                                             \
//     self.poll();                                                                                                       \
//   }
#define YIELD                                                                                                          \
  this->state = &&CAT(s, __LINE__);                                                                                    \
  return;                                                                                                              \
  CAT(s, __LINE__) :;
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
  // clang-format on

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
    writeControl(concurrencyIndex, dev, i, (void*)&outDyn(concurrencyIndex, srcIndex), outDynsMr->mrs[di]->lkey,
        (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di], sizeof(DynamicAddresses) * numDyns);
  }

  // Writes this rank's compile_op tensor list header (outDyn slot 0) into its slot of rank i's Group::customDyns (see
  // WorkCustom)
  void writeCustomDyn(size_t concurrencyIndex, size_t i) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inCustomDyn(concurrencyIndex, rank));
    writeControl(concurrencyIndex, dev, i, (void*)&outDyn(concurrencyIndex, 0), outDynsMr->mrs[di]->lkey,
        (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di], sizeof(DynamicAddresses));
  }

  void writeStepValue(size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex, size_t* counter = nullptr) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inStepValue(concurrencyIndex, dstIndex));
    writeControl(concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di],
        sizeof(uint32_t), counter);
  }

  // Writes the op's step value and id (outStepValue slots arrivalWords and arrivalWords + 1) as this rank's arrival
  // value into the memory of rank i (see Group::cpuArrivalValues and WorkCustom)
  void writeArrivalValue(size_t concurrencyIndex, size_t i) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inArrivalValue(concurrencyIndex, rank));
    static_assert(sizeof(ArrivalValue) == 2 * sizeof(uint32_t) && arrivalWords % 2 == 0);
    static_assert(arrivalWords + 2 <= 32);
    writeControl(concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, arrivalWords),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di],
        sizeof(ArrivalValue));
  }
  // (outStepValue slots arrivalWords and arrivalWords + 1 hold the step value and the op id, see WorkCustom)
  static constexpr size_t arrivalWords = 8;

  // Writes the op's step value + generation into notification slot `slot` of this rank's in the memory of rank i (see
  // Group::customNotifications and WorkCustom)
  // (through the queue pair of the reads, unlike the credits, see WorkCustom::partDone: through the control queue pair,
  // they made no difference at large sizes, and the times of small ops varied more: reduce-scatter of 1 MiB across 2
  // domains of 32 GPUs, 167 us at the median instead of 159, the fastest 10% 140 us instead of 156)
  void writeNotification(size_t concurrencyIndex, size_t i, size_t slot, size_t generation) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inNotification(concurrencyIndex, rank, slot));
    writeControl(concurrencyIndex, dev, i, (void*)&generationValue(concurrencyIndex, generation, 0),
        generationValuesMr->mrs[di]->lkey, (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di],
        sizeof(uint32_t));
  }

  void writeStepValueDeviceChunk(
      size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex, size_t device, size_t chunk) {
    CHECK(i < size);
    CHECK(srcIndex < size);
    CHECK(dstIndex < size);
    CHECK(device < maxDevices);
    CHECK(chunk < maxChunks);
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    auto offset = group->getSharedOffset(&inStepValueDeviceChunk(concurrencyIndex, dstIndex, device, chunk));
    writeControl(concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteComms[i].address + offset), remoteComms[i].keys[di],
        sizeof(uint32_t));
  }

  void writeCudaStepValueDeviceChunk(
      size_t concurrencyIndex, size_t i, size_t srcIndex, size_t dstIndex, size_t device, size_t chunk) {
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    size_t index = stepValueDeviceChunkIndex(concurrencyIndex, dstIndex, device, chunk);
    size_t offset = sizeof(uint32_t) * index;
    writeControl(concurrencyIndex, dev, i, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(remoteCudaStepValuesDeviceChunks[i].address + offset),
        remoteCudaStepValuesDeviceChunks[i].keys[di], sizeof(uint32_t));
  }

  void writeCpuOut(size_t concurrencyIndex, size_t dstIndex, size_t srcIndex) {
    CHECK(srcIndex < 32);
    CHECK(dstIndex < 4096 + maxChunks * size);
    size_t di = 0;
    auto& dev = devices[di];
    auto offset = sizeof(uint32_t) * dstIndex;
    writeControl(concurrencyIndex, dev, rank, (void*)&outStepValue(concurrencyIndex, srcIndex),
        outStepValuesStorageMr->mrs[di]->lkey, (void*)(group->cpuOutBuffer.cuda(concurrencyIndex) + offset),
        cpuOutMr->mrs[di]->rkey, sizeof(uint32_t));
  }

  // Signals of compile_op's RDMA transfers (Group::rdmaSignalBuffer), which its kernels wait for before they use the
  // data. A NIC's writes into GPU memory reach the GPU in the order the NIC issued them if the later ones are not
  // relaxed-ordered (PCIe lets a write pass earlier ones only with relaxed ordering, which the data's registrations
  // allow for speed), and a GPU whose GPUDirect RDMA writes ordering is "owner" lets its kernels see them in that
  // order. So a signal is a write through each NIC that carried some of the data, after that NIC has placed it (the
  // completion of this rank's reads, or of its writes into the other rank), into a registration without relaxed
  // ordering (rdmaSignalMr) on the same path as the data's (both registered through Data Direct or both not): a word
  // per NIC, and the kernel waits until all words of the signal have the value (see signalWords). Elsewhere (or with
  // MOODIST_RDMA_FLUSH=1) the data is flushed first (see flushRead), then one write sets all words.
  // writeRdmaSignal writes words [first, end) of signal `signal` of rank i (of this rank: a loopback write), through
  // dev (with controlQp, through its control queue pair to rank i, see writeControl): the step value + generation (see
  // compile_op::Graph::Signal). Signals without data to follow (credits, see CustomOpDescriptor::IbRead) take any path.
  void writeRdmaSignal(size_t concurrencyIndex, Device& dev, size_t i, size_t signal, size_t first, size_t end,
      size_t generation, bool controlQp = false) {
    CHECK(i < size && first < end && end <= maxDevices);
    CHECK(signal < group->rdmaSignalBuffer.itembytes / Group::rdmaSignalBytes);
    size_t di = &dev - devices.data();
    auto offset = group->rdmaSignalBuffer.itembytes * concurrencyIndex + Group::rdmaSignalBytes * signal +
                  sizeof(uint32_t) * first;
    writeControl(concurrencyIndex, dev, i, (void*)&generationValue(concurrencyIndex, generation, first),
        generationValuesMr->mrs[di]->lkey, (void*)(remoteRdmaSignals[i].address + offset),
        remoteRdmaSignals[i].keys[di], sizeof(uint32_t) * (end - first), nullptr, controlQp);
  }

  // The words of a signal that NIC di writes when the NICs in `used` (a bit each) carried its data: from the one after
  // the previous used NIC's up to its own, the last used NIC's up to the last word, so that every word is written
  // once, and each used NIC's own by that NIC
  static std::pair<size_t, size_t> signalWords(uint8_t used, size_t di) {
    CHECK(used & (1 << di));
    size_t first = 0;
    for (size_t d = 0; d != di; ++d) {
      if (used & (1 << d)) {
        first = d + 1;
      }
    }
    return {first, (used >> (di + 1)) == 0 ? maxDevices : di + 1};
  }

  // Whether a signal can follow data that the NICs in `used` wrote into memory registered as mr (see
  // writeRdmaSignal): the GPU orders the writes, and the data and the signals take the same path
  bool signalFollows(const MemoryRegistration* mr, uint8_t used) {
    if (!orderedSignals) {
      return false;
    }
    for (size_t di : range(devices.size())) {
      if ((used & (1 << di)) && mr->mrs[di]->dataDirect != rdmaSignalMr->mrs[di]->dataDirect) {
        return false;
      }
    }
    return true;
  }

  // A flush: a read of one byte at `address`, GPU memory of rank i (of this rank: a loopback read), through dev.
  // Writes into GPU memory through a NIC are visible to the GPU's kernels once a read through the same NIC that
  // follows them has completed: PCIe does not let a read pass the writes before it. Used before the signals of data
  // where they cannot follow it (see writeRdmaSignal), and at the end of an op for the reads without a signal, before
  // the stream goes on.
  void flushRead(Device& dev, size_t i, uintptr_t address, uint32_t rkey, Callback* callback) {
    size_t di = &dev - devices.data();
    postRead(dev, i, flushBuffer.cpuPointer, flushMr->mrs[di]->lkey, (void*)address, rkey, 1, callback);
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

      stepValue = self.nextInternalStepValue[concurrencyIndex] + 1;
      self.nextInternalStepValue[concurrencyIndex] += 4;

      {
        DynamicAddresses& out = self.outDyn(concurrencyIndex, 0);
        out.opType = opTypeInternalBarrier;
        out.gatherAddress = 0;
        out.gatherBytes = 0;
        out.stepValue = stepValue;
      }
      self.outStepValue(concurrencyIndex, 0) = stepValue;
      self.outStepValue(concurrencyIndex, 1) = stepValue + 1;
      self.outStepValue(concurrencyIndex, 2) = stepValue + 2;
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        size_t di = (rank + i) % self.devices.size();
        auto& dev = self.devices[di];
        self.writeControl(concurrencyIndex, dev, i, (void*)&self.outStepValue(concurrencyIndex, 1),
            self.outStepValuesStorageMr->mrs[di]->lkey,
            (void*)(self.remoteInternalInStepValueBuffer[i].address +
                    sizeof(uint32_t) * (2 * size * concurrencyIndex + 2 * rank)),
            self.remoteInternalInStepValueBuffer[i].keys[di], sizeof(uint32_t));
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (*(&self.internalInStepValueBuffer.at<uint32_t>(concurrencyIndex) + 2 * i) != stepValue + 1) {
          YIELD
        }
      }
      WAIT_CONTROL;
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        size_t di = (rank + i) % self.devices.size();
        auto& dev = self.devices[di];
        self.writeControl(concurrencyIndex, dev, i, (void*)&self.outStepValue(concurrencyIndex, 2),
            self.outStepValuesStorageMr->mrs[di]->lkey,
            (void*)(self.remoteInternalInStepValueBuffer[i].address +
                    sizeof(uint32_t) * (2 * size * concurrencyIndex + 2 * rank + 1)),
            self.remoteInternalInStepValueBuffer[i].keys[di], sizeof(uint32_t));
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (*(&self.internalInStepValueBuffer.at<uint32_t>(concurrencyIndex) + 2 * i + 1) != stepValue + 2) {
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
    const IVector<size_t>& ipcRanks = self.group->ipcRanks;
    const IVector<size_t>& peerIndices = self.group->peerIndices;

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
        inputBuffer = self.allocateTemporaryBuffer(params.bytes);
        outputBuffer = self.allocateTemporaryBuffer(params.pitch * size);

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
          self.writeDataDistributed(i, srcAddr, params.bytes, inputBuffer.mr, dyn.gatherAddress, dyn.gatherKey,
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
            self.group->getPeerVar(proxyInfo[recvIndex].destinationPeerIndex,
                self.group->cpuProxyReady)[size * concurrencyIndex + i] = stepValue;
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
        std::memcpy((void*)(params.outputAddress + params.pitch * i),
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
    const IVector<size_t>& ipcRanks = self.group->ipcRanks;
    const IVector<size_t>& peerIndices = self.group->peerIndices;

    WorkReduceScatterCpu(CpuThreadImpl& self, QueueEntryReduceScatterCpu& params)
        : Work(self, params), params(params) {}

    template<typename T, size_t dindex = 0, size_t opindex = 0>
    decltype(stepPtr) getStepPtr() {
      if (dindex == (size_t)params.dindex && opindex == (size_t)params.opindex) {
        return [](Work* w) {
          ((T*)w)->template step<(Dtype)dindex, (Reduction)opindex>();
        };
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

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        recvBuffer = self.allocateTemporaryBuffer(params.pitch * recvRanks.size());
        sendBuffer = self.allocateTemporaryBuffer(params.pitch * sendRanks.size());
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
          self.writeDataDistributed(i, srcAddr, params.bytes, sendBuffer.mr, dyn.gatherAddress, dyn.gatherKey,
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

  struct WorkBroadcastRingCpu : Work {
    QueueEntryBroadcastCpu& params;
    // size_t i;
    // size_t di;
    // size_t offset;
    TemporaryBufferHandle temporaryBuffer;
    size_t liveSends;
    size_t index;

    const IVector<size_t>& ipcRanks = self.group->ipcRanks;
    const IVector<size_t>& peerIndices = self.group->peerIndices;

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

          temporaryBuffer = self.allocateTemporaryBuffer(params.bytes);

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
        temporaryBuffer = self.allocateTemporaryBuffer(params.bytes);
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
                  self.writeData(dev, sendTo, (void*)(srcAddr + sendOffset), srcMr->mrs[di]->lkey,
                      (void*)(dstAddr + sendOffset), key[di], n, self.makeCallback([this, di, source, chunkIndex]() {
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
              std::memcpy((void*)(params.tensorAddress + sendOffset),
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

  struct WorkCreateQueue : Work {
    QueueEntryCreateQueue& params;
    size_t i;
    uint32_t stepValue;

    WorkCreateQueue(CpuThreadImpl& self, QueueEntryCreateQueue& params) : Work(self, params), params(params) {}

    void step() {
      ENTER

      stepValue = self.nextInternalStepValue[concurrencyIndex] + 1;
      self.nextInternalStepValue[concurrencyIndex] += 4;

      self.outStepValue(concurrencyIndex, 0) = stepValue;
      self.outStepValue(concurrencyIndex, 1) = stepValue + 1;
      self.outStepValue(concurrencyIndex, 2) = stepValue + 2;

      if (rank == params.location) {
        DynamicAddresses& out = self.outDyn(concurrencyIndex, 0);
        out.opType = opTypeCreateQueue;
        out.gatherAddress = params.address;
        out.gatherBytes = params.bytes;
        out.stepValue = stepValue;
        static_assert(maxDevices >= 2);
        out.gatherKey[0] = params.location;
        out.gatherKey[1] = params.streaming;
        for (i = 0; i != size; ++i) {
          if (i == self.rank) {
            continue;
          }
          self.writeDyn(concurrencyIndex, i, 0, 0);
        }
      } else {
        WAIT_DYN(opTypeCreateQueue, 0);
        if (params.location != dyn.gatherKey[0]) {
          fatal("Queue location mismatch. Queue was created on rank %d with location %d, but on rank %d with location "
                "%d",
              rank, params.location, params.location, dyn.gatherKey[0]);
        }
        if (params.streaming != dyn.gatherKey[1]) {
          fatal("Queue streaming mismatch. Queue was created on rank %d with streaming %d, but on rank %d with "
                "streaming "
                "%d",
              rank, params.streaming, params.location, dyn.gatherKey[1]);
        }
        *params.outAddress = dyn.gatherAddress;
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        size_t di = (rank + i) % self.devices.size();
        auto& dev = self.devices[di];
        self.writeControl(concurrencyIndex, dev, i, (void*)&self.outStepValue(concurrencyIndex, 1),
            self.outStepValuesStorageMr->mrs[di]->lkey,
            (void*)(self.remoteInternalInStepValueBuffer[i].address +
                    sizeof(uint32_t) * (2 * size * concurrencyIndex + 2 * rank)),
            self.remoteInternalInStepValueBuffer[i].keys[di], sizeof(uint32_t));
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (*(&self.internalInStepValueBuffer.at<uint32_t>(concurrencyIndex) + 2 * i) != stepValue + 1) {
          YIELD
        }
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        size_t di = (rank + i) % self.devices.size();
        auto& dev = self.devices[di];
        self.writeControl(concurrencyIndex, dev, i, (void*)&self.outStepValue(concurrencyIndex, 2),
            self.outStepValuesStorageMr->mrs[di]->lkey,
            (void*)(self.remoteInternalInStepValueBuffer[i].address +
                    sizeof(uint32_t) * (2 * size * concurrencyIndex + 2 * rank + 1)),
            self.remoteInternalInStepValueBuffer[i].keys[di], sizeof(uint32_t));
      }
      for (i = 0; i != size; ++i) {
        if (i == self.rank) {
          continue;
        }
        while (*(&self.internalInStepValueBuffer.at<uint32_t>(concurrencyIndex) + 2 * i + 1) != stepValue + 2) {
          YIELD
        }
      }
      WAIT_CONTROL
      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
      self.cpuThread->freelistCreateQueue.push(&params);
      self.allocatorCreateQueue.deallocate(this);
      DONE
    }
  };

  void onRecvQueuePut(std::string_view view, uint32_t source) {
    uintptr_t queueAddress;
    uintptr_t tensorAddress;
    size_t bytes;
    uint32_t remoteOpKey;
    std::array<uint32_t, maxDevices> rkeys;
    int dtype;
    uint8_t shapeSize;
    std::array<int64_t, 256> shapeArr;
    uint32_t getKey;
    uint32_t transactionKey;
    size_t queueSize;
    bool inlined;
    // log.info("view is %#x %d\n", (uintptr_t)view.data(), view.size());
    const char* inlineData = nullptr;
    {
      Deserialize x(view);
      x(queueAddress, tensorAddress, bytes);
      x(remoteOpKey, getKey, transactionKey, queueSize);
      for (size_t i = 0; i != maxDevices; ++i) {
        x(rkeys[i]);
      }
      x(dtype);
      x(shapeSize);
      for (size_t i = 0; i != shapeSize; ++i) {
        x(shapeArr[i]);
      }
      x(inlined);
      if (inlined) {
        inlineData = x.consume(bytes);
      }
    }

    if (inlineData) {
      auto buffer = group->allocateCpu(bytes);

      uintptr_t localAddress = (uintptr_t)buffer.cpuPointer;

      std::memcpy((void*)localAddress, inlineData, bytes);

      auto tensor = TensorDataPtr::make();
      tensor->buffer = AllocatedCpuBufferSharedPtr::make();
      *tensor->buffer = std::move(buffer);
      tensor->dataPtr = (uintptr_t)localAddress;
      tensor->dataBytes = bytes;
      tensor->dtype = dtype;
      tensor->shape.resize(shapeSize);
      for (size_t i = 0; i != shapeSize; ++i) {
        tensor->shape[i] = shapeArr[i];
      }

      queueFinish(group, queueAddress, source, queuePrepare(queueAddress, source, getKey, transactionKey),
          std::move(tensor), getKey, transactionKey, queueSize);
      enqueueCallback([this, source, remoteOpKey] {
        sendQueueReadFinished(source, remoteOpKey);
      });
      return;
    }

    // log.info("recv queue put %#x\n", remoteOpKey);

    CHECK(remoteOpKey != 0);

    QueueEntryQueueRead* e = cpuThread->freelistQueueRead.pop();
    e->task = taskQueueRead;
    e->source = source;
    e->remoteAddress = tensorAddress;
    e->remoteOpKey = remoteOpKey;
    e->rkeys = rkeys;
    e->bytes = bytes;
    e->queueAddress = queueAddress;
    e->localOpKey = queuePrepare(queueAddress, source, getKey, transactionKey);
    e->tensorShape.resize(shapeSize);
    for (size_t i = 0; i != shapeSize; ++i) {
      e->tensorShape[i] = shapeArr[i];
    }
    e->tensorDtype = dtype;
    e->getKey = getKey;
    e->transactionKey = transactionKey;
    e->queueSize = queueSize;
    group->cpuThread->enqueue(e);
  }

  struct QueuePutInfo {
    uintptr_t tensorAddress;
    size_t bytes;
    std::array<uint32_t, maxDevices> gatherKey;
    int tensorDtype;
  };

  HashMap<uint32_t, Function<void(uintptr_t*, uint32_t*)>> queuePutCallbacks;

  void executeQueuePut(QueueEntryQueuePut& params) {

    uintptr_t tensorAddress = (uintptr_t)params.tensor->data();
    size_t bytes = params.tensor->bytes();
    MemoryRegistration* tensorMr = nullptr;

    CHECK(params.putKey != 0);

    bool isCuda = params.tensor->isCuda;

    if (bytes != 0) {
      if (isCuda) {
        tensorMr = regMrCuda(tensorAddress, bytes);
      } else {
        auto region = cpu_allocator::regionAt(tensorAddress);

        if (region.first) {
          CHECK(region.first <= tensorAddress && region.first + region.second >= tensorAddress + bytes);
          tensorMr = regMr(tensorAddress, bytes);
        } else {
          fatal("unreachable; queue put called with an unknown cpu address %#x", tensorAddress);
          CHECK(false);
        }
      }
    }
    queuePutCallbacks[params.putKey] = std::move(params.callback);
    CHECK(params.tensor->shape.size() < 256);
    auto buffer = allocateTemporaryBuffer(4096);
    size_t i = params.location;
    void* cpuPointer = buffer->cpuPointer;
    size_t serializedBytes = serializeToUnchecked(cpuPointer, SerializeFunction([&](auto& x) {
      x(header(opTypeQueuePut, i), params.remoteAddress, tensorAddress, bytes);
      x(params.putKey, params.remoteGetKey, params.transactionKey, params.queueSize);
      for (size_t i = 0; i != maxDevices; ++i) {
        x(tensorMr && tensorMr->mrs[i] ? tensorMr->mrs[i]->rkey : 0);
      }
      x(params.tensor->dtype);
      x((uint8_t)params.tensor->shape.size());
      for (int64_t v : params.tensor->shape) {
        x(v);
      }
      x(false);
    }));
    if ((!isCuda && !params.streaming) || bytes == 0) {
      if (serializedBytes + bytes <= buffer->bytes) {
        bool inlineFlag = true;
        std::memcpy((char*)cpuPointer + serializedBytes - 1, &inlineFlag, sizeof(bool));
        std::memcpy((char*)cpuPointer + serializedBytes, (void*)tensorAddress, bytes);
        serializedBytes += bytes;
      } else {
        CHECK(bytes != 0);
      }
    }
    sendBuffer(i, std::move(buffer), serializedBytes);
    cpuThread->freelistQueuePut.push(&params);
  }

  void onRecvQueueReadFinished(std::string_view view) {
    QueueEntryQueueReadFinished* e = cpuThread->freelistQueueReadFinished.pop();
    e->task = taskQueueReadFinished;
    deserializeBufferPart(view, e->key);
    group->cpuThread->enqueue(e);
  }

  void executeQueueReadFinished(QueueEntryQueueReadFinished& params) {
    auto i = queuePutCallbacks.find(params.key);
    if (i == queuePutCallbacks.end()) {
      fatal("Queue put callback key %#x not found", params.key);
    }
    uintptr_t cudaDoneAddress = 0;
    std::move(i->second)(&cudaDoneAddress, &params.cudaDoneValue);
    if (cudaDoneAddress) {
      size_t di = 0;
      auto& dev = devices[di];
      auto* mr = regMrCuda(cudaDoneAddress, 4);
      uintptr_t sourceAddress = (uintptr_t)&params.cudaDoneValue;
      auto* sourceMr = regMr(sourceAddress, 4);
      writeData(dev, rank, (void*)sourceAddress, sourceMr->mrs[di]->lkey, (void*)cudaDoneAddress, mr->mrs[di]->rkey,
          sizeof(uint32_t), makeCallback([this, &params] {
            cpuThread->freelistQueueReadFinished.push(&params);
          }));
    } else {
      cpuThread->freelistQueueReadFinished.push(&params);
    }
    queuePutCallbacks.erase(i);
  }

  template<typename F = std::nullptr_t>
  void sendBuffer(size_t i, TemporaryBufferHandle buffer, size_t bytes, F&& callback = nullptr) {
    CHECK(bytes <= buffer->bytes);
    void* cpuPointer = buffer->cpuPointer;
    size_t di = (rank + i) % devices.size();
    auto& dev = devices[di];
    uint32_t lkey = buffer.mr->mrs[di]->lkey;
    if constexpr (std::is_null_pointer_v<F>) {
      postSend(dev, i, cpuPointer, lkey, bytes, makeCallback([this, buffer = std::move(buffer)] {}));
    } else {
      postSend(dev, i, cpuPointer, lkey, bytes,
          makeCallback([this, buffer = std::move(buffer), callback = std::move(callback)] {
            callback();
          }));
    }
  }

  void sendQueueReadFinished(uint32_t source, uint32_t remoteOpKey) {
    CHECK(remoteOpKey != 0);
    auto buffer = allocateTemporaryBuffer(4096);
    size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeQueueReadFinished, source), remoteOpKey);
    sendBuffer(source, std::move(buffer), bytes);
  }

  void executeQueueRead(QueueEntryQueueRead& params) {
    size_t bytes = params.bytes;

    // log.info(
    //     "execute queue read! %d bytes! %#x %#x  queue size %d\n", bytes, params.localOpKey, params.remoteOpKey,
    //     params.queueSize);

    CHECK(params.remoteOpKey != 0);

    auto buffer = allocateTemporaryBuffer(bytes);

    void* cpuPointer = buffer->cpuPointer;
    std::array<uint32_t, maxDevices> lkeys;
    for (size_t i = 0; i != maxDevices; ++i) {
      lkeys[i] = buffer.mr->mrs[i] ? buffer.mr->mrs[i]->lkey : 0;
    }

    uintptr_t localAddress = (uintptr_t)buffer->cpuPointer;

    auto tensor = TensorDataPtr::make();
    tensor->buffer = AllocatedCpuBufferSharedPtr::make();
    *tensor->buffer = std::move(buffer.buffer);
    tensor->dataPtr = (uintptr_t)tensor->buffer->cpuPointer;
    tensor->dataBytes = tensor->buffer->bytes;
    tensor->dtype = params.tensorDtype;
    tensor->shape = params.tensorShape;

    auto callback = makeCallback([this, &params, tensor = std::move(tensor)]() mutable {
      // log.info("queue read finished! %#x\n", params.localOpKey);
      queueFinish(group, params.queueAddress, params.source, params.localOpKey, std::move(tensor), params.getKey,
          params.transactionKey, params.queueSize);
      sendQueueReadFinished(params.source, params.remoteOpKey);

      cpuThread->freelistQueueRead.push(&params);
    });

    readDataDistributed(
        params.source, (void*)localAddress, lkeys, (void*)params.remoteAddress, params.rkeys, bytes, callback);
  }

  void onRecvQueueGet(std::string_view view, uint32_t source) {
    uint8_t op;
    uintptr_t queueAddress;
    uintptr_t remoteQueueAddress;
    uint32_t key;
    deserializeBufferPart(view, op, queueAddress, remoteQueueAddress, key);
    switch (op) {
    case QueueEntryQueueGet::opStart:
      queueRemoteGetStart(group, queueAddress, source, remoteQueueAddress, key);
      break;
    case QueueEntryQueueGet::opStop:
      queueRemoteGetStop(group, queueAddress, source, remoteQueueAddress, key);
      break;
    case QueueEntryQueueGet::opStopAck:
      queueRemoteGetStopAck(group, queueAddress, source, remoteQueueAddress, key);
      break;
    default:
      CHECK(false);
    }
  }

  void executeQueueGet(QueueEntryQueueGet& params) {
    auto buffer = allocateTemporaryBuffer(4096);
    size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeQueueGet, params.location), params.op,
        params.remoteQueueAddress, params.localQueueAddress, params.key);
    sendBuffer(params.location, std::move(buffer), bytes);
    cpuThread->freelistQueueGet.push(&params);
  }

  void onRecvQueueTransaction(std::string_view view, uint32_t source) {
    uintptr_t queueAddress;
    uint8_t op;
    uint32_t key;
    deserializeBufferPart(view, queueAddress, op, key);
    // log.info("%#x %#x %#x %#x\n", source, queueAddress, op, key);
    CHECK(queueAddress != 0);
    switch (op) {
    case transactionOpCommit:
      queueTransactionCommit(group, queueAddress, source, key);
      break;
    case transactionOpCancel:
      queueTransactionCancel(group, queueAddress, source, key);
      break;
    default:
      CHECK(false);
    }
  }

  void executeQueueTransaction(QueueEntryQueueTransaction& params) {
    CHECK(params.remoteAddress != 0);
    auto buffer = allocateTemporaryBuffer(4096);
    size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeQueueTransaction, params.location),
        params.remoteAddress, params.op, params.transactionKey);
    sendBuffer(params.location, std::move(buffer), bytes);
    cpuThread->freelistQueueTransaction.push(&params);
  }

  struct WorkCat : Work {
    QueueEntryCat& params;
    size_t i;
    size_t o;

    TemporaryBufferHandle listBuffer;

    struct CatTensor {
      uint32_t index;
      uint8_t dtype;
      uintptr_t address;
      size_t bytes;
      std::array<uint32_t, maxDevices> rkey;
      union {
        uint32_t ndim;
        uint32_t nelem;
      };
    };
    static_assert(std::is_trivial_v<CatTensor>);

    Vector<TemporaryBufferHandle> remoteLists;
    Vector<uint8_t> remoteListReady;
    Vector<std::pair<size_t, const CatTensor*>> sorted;
    size_t numReceivedParts = 0;
    size_t nextPartIndex = 0;
    size_t nLiveReads = 0;
    size_t nReadsDone = 0;
    Vector<std::pair<size_t, size_t>> readOrder;

    TemporaryBufferHandle resultBuffer;
    MemoryRegistration* outMr = nullptr;

    TensorDataPtr resultTensor;

    Vector<TemporaryBufferHandle> temporaryBuffers;

    WorkCat(CpuThreadImpl& self, QueueEntryCat& params) : Work(self, params), params(params) {}

    void allocateResult() {
      size_t total = 0;

      size_t n = sorted.size();
      readOrder.resize(n);
      for (size_t i = 0; i != n; ++i) {
        readOrder[i] = {i, total};
        total += sorted[i].second->bytes;
      }

      std::shuffle(readOrder.begin(), readOrder.end(), getRng());

      if (params.out) {
        outMr = params.out->isCuda ? self.regMrCuda(params.out->data(), params.out->bytes())
                                   : self.regMr(params.out->data(), params.out->bytes());
        if (params.out->bytes() != total) {
          fatal("cat got an out tensor of %d bytes, but the result is %d bytes\n", params.out->bytes(), total);
        }
      } else {
        resultBuffer = self.allocateTemporaryBuffer(total);
        outMr = resultBuffer.mr;
      }

      // log.info("CAT %#x/%d allocated %d bytes\n", stepValue, concurrencyIndex, resultBuffer->bytes);
    }

    bool progress() {
      if (nextPartIndex == sorted.size()) {
        return true;
      }
      if (nLiveReads >= 8) {
        return false;
      }
      auto [index, offset] = readOrder[nextPartIndex];
      CHECK(sorted[index].second != nullptr);
      ++nextPartIndex;
      size_t i = sorted[index].first;
      const CatTensor* t = sorted[index].second;
      CHECK(t->index == index);

      if (t->bytes) {
        void* dst = (void*)((params.out ? params.out->data() : (uintptr_t)resultBuffer->cpuPointer) + offset);
        void* src = (void*)t->address;

        std::array<uint32_t, maxDevices> lkeys;
        for (size_t i = 0; i != maxDevices; ++i) {
          lkeys[i] = outMr->mrs[i] ? outMr->mrs[i]->lkey : 0;
        }
        ++nLiveReads;
        self.readDataDistributed(i, dst, lkeys, src, t->rkey, t->bytes, self.makeCallback([this, index] {
          --nLiveReads;
          ++nReadsDone;
        }));
      } else {
        ++nReadsDone;
      }
      return false;
    }

    void step() {
      ENTER

      // log.info("CAT %#x/%d enter\n", stepValue, concurrencyIndex);

      // self.initReceives();

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        size_t n = params.locals.size();
        size_t nbytes = sizeof(CatTensor) * n;
        for (auto& v : params.locals) {
          nbytes += 8 * v.second->shape.size();
        }
        listBuffer = self.allocateTemporaryBuffer(nbytes);
        CatTensor* list = (CatTensor*)listBuffer->cpuPointer;
        for (size_t i = 0; i != n; ++i) {
          auto& [index, tensor] = params.locals[i];
          auto& v = list[i];
          v.index = index;
          v.address = tensor->data();
          v.bytes = tensor->bytes();
          v.dtype = tensor->dtype;
          v.ndim = tensor->shape.size();
          auto* mr = tensor->isCuda ? self.regMrCuda(v.address, v.bytes) : self.regMr(v.address, v.bytes);
          for (size_t i = 0; i != maxDevices; ++i) {
            v.rkey[i] = mr->mrs[i] ? mr->mrs[i]->rkey : 0;
          }
        }
        char* shapeptr = (char*)(list + n);
        for (auto& v : params.locals) {
          auto& tensor = v.second;
          CHECK(tensor->shape.size() != 0);
          static_assert(sizeof(tensor->shape[0]) == 8);
          size_t n = 8 * tensor->shape.size();
          std::memcpy(shapeptr, tensor->shape.data(), n);
          shapeptr += n;
        }
        CHECK(shapeptr == (const char*)listBuffer->cpuPointer + nbytes);

        resultTensor = TensorDataPtr::make();
      }

      for (size_t o = 0; o != size; ++o) {
        size_t i = self.rank + o;
        if (i >= size) {
          i -= size;
        }
        CHECK(i >= 0 && i < size);
        DynamicAddresses& out = self.outDyn(concurrencyIndex, i);
        out.opType = opTypeCat;
        out.gatherAddress = (uintptr_t)listBuffer->cpuPointer;
        out.gatherBytes = listBuffer->bytes;
        out.stepValue = stepValue;
        for (size_t i = 0; i != maxDevices; ++i) {
          out.gatherKey[i] = listBuffer.mr->mrs[i] ? listBuffer.mr->mrs[i]->rkey : 0;
        }
        self.writeDyn(concurrencyIndex, i, i, rank);
      }

      CHECK(remoteLists.empty());
      remoteLists.resize(size);
      remoteListReady.resize(size);
      sorted.clear();

      for (o = 0; o != size; ++o) {
        i = self.rank - o;
        if (i >= size) {
          i += size;
        }
        CHECK(i >= 0 && i < size);

        while (self.inDyn(concurrencyIndex, i).stepValue != stepValue) {
          YIELD
        }
        DynamicAddresses& dyn = self.inDyn(concurrencyIndex, i);
        size_t bytes = std::exchange(dyn.gatherBytes, 0);
        CHECK_DYN(dyn, opTypeCat, stepValue, 0);

        if (bytes == 0) {
          remoteListReady[i] = 1;
        } else {
          auto& targetBuffer = remoteLists[i];
          targetBuffer = self.allocateTemporaryBuffer(bytes);

          size_t di = o % self.devices.size();
          auto& dev = self.devices[di];
          self.readData(dev, i, targetBuffer->cpuPointer, targetBuffer.mr->mrs[di]->lkey, (void*)dyn.gatherAddress,
              dyn.gatherKey[di], bytes, self.makeCallback([this, i = this->i] {
                remoteListReady[i] = 1;
              }));
        }
      }

      for (o = 0; o != size; ++o) {
        i = self.rank - o;
        if (i >= size) {
          i += size;
        }
        CHECK(i >= 0 && i < size);

        while (remoteListReady[i] == 0) {
          YIELD
        }
        const CatTensor* rl = (CatTensor*)remoteLists[i]->cpuPointer;
        size_t remainingBytes = remoteLists[i]->bytes;
        while (remainingBytes) {
          if (sorted.size() <= rl->index) {
            sorted.resize(rl->index + 1);
          }
          if (sorted[rl->index].second) {
            fatal("cat: got a duplicate tensor for index %d", rl->index);
          }
          sorted[rl->index] = {i, rl};

          if (resultTensor->dtype == -1) {
            resultTensor->dtype = rl->dtype;
          } else {
            if (rl->dtype != resultTensor->dtype) {
              fatal("cat: got mismatching dtypes");
            }
          }

          remainingBytes -= sizeof(CatTensor);
          remainingBytes -= 8 * rl->ndim;

          CHECK((ssize_t)remainingBytes >= 0);

          ++numReceivedParts;
          ++rl;
        }
        const int64_t* shapeptr = (const int64_t*)rl;
        const int64_t* endptr = (const int64_t*)((const char*)remoteLists[i]->cpuPointer + remoteLists[i]->bytes);
        for (const CatTensor* t = (CatTensor*)remoteLists[i]->cpuPointer; t != rl; ++t) {
          size_t ndim = t->ndim;
          CHECK(shapeptr + ndim <= endptr);
          if (resultTensor->shape.empty()) {
            if (ndim == 0) {
              fatal("cat: got ndim == 0 tensor");
            }
            resultTensor->shape.resize(ndim);
            for (size_t i = 0; i != ndim; ++i) {
              resultTensor->shape[i] = shapeptr[i];
            }
          } else {
            if (ndim != resultTensor->shape.size()) {
              fatal("cat: expected all tensors to have the same number of dimensions, but got %d and %d",
                  resultTensor->shape.size(), ndim);
            }
            resultTensor->shape[0] += shapeptr[0];
            for (size_t i = 0; i != ndim; ++i) {
              if (i == 0) {
                continue;
              }
              if (resultTensor->shape[i] != shapeptr[i]) {
                fatal("cat: expected all tensors to have same shape except in cat dimension (0), but got %d and %d in "
                      "dimension %d",
                    resultTensor->shape[i], shapeptr[i], i);
              }
            }
          }
          shapeptr += ndim;
        }
      }

      if (numReceivedParts != sorted.size()) {
        std::string s;
        for (size_t i = 0; i != sorted.size(); ++i) {
          if (sorted[i].second == nullptr) {
            if (!s.empty()) {
              s += ", ";
            }
            s += std::to_string(i);
          }
        }
        fatal("cat operation missing parts: [%s]", s);
      }

      allocateResult();

      while (!progress()) {
        YIELD
      }

      while (nReadsDone != sorted.size()) {
        YIELD
      }

      for (i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        self.writeStepValue(concurrencyIndex, i, 0, rank);
      }

      for (i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        while (self.inStepValue(concurrencyIndex, i) < stepValue) {
          YIELD
        }
      }

      if (!params.out) {
        resultTensor->buffer = AllocatedCpuBufferSharedPtr::make();
        *resultTensor->buffer = std::move(resultBuffer.buffer);
        resultTensor->dataPtr = (uintptr_t)resultTensor->buffer->cpuPointer;
        resultTensor->dataBytes = resultTensor->buffer->bytes;

        CHECK(resultTensor->itemsize() * resultTensor->numel() == resultTensor->dataBytes);

        params.future->result = std::move(resultTensor);
      }

      params.future->done = 1;
      futexWakeAll(&params.future->done);

      remoteLists.clear();
      listBuffer = {};

      params.locals.clear();

      temporaryBuffers.clear();

      // log.info("CAT %#x/%d done\n", stepValue, concurrencyIndex);

      self.cpuThread->freelistCat.push(&params);
      self.allocatorCat.deallocate(this);
      DONE
    }
  };

  void executeCopy(QueueEntryCopy& params) {

    auto* destinationMr = params.destination->isCuda
                              ? regMrCuda(params.destination->data(), params.destination->bytes())
                              : regMr(params.destination->data(), params.destination->bytes());
    auto* sourceMr = params.source->isCuda ? regMrCuda(params.source->data(), params.source->bytes())
                                           : regMr(params.source->data(), params.source->bytes());

    std::array<uint32_t, maxDevices> lkeys;
    for (size_t i = 0; i != maxDevices; ++i) {
      lkeys[i] = destinationMr->mrs[i] ? destinationMr->mrs[i]->lkey : 0;
    }

    std::array<uint32_t, maxDevices> rkeys;
    for (size_t i = 0; i != maxDevices; ++i) {
      rkeys[i] = sourceMr->mrs[i] ? sourceMr->mrs[i]->lkey : 0;
    }

    readDataDistributed(rank, (void*)params.destination->data(), lkeys, (void*)params.source->data(), rkeys,
        params.destination->bytes(), makeCallback([&params] {
          params.future->done = 1;
          futexWakeAll(&params.future->done);
        }));
  }

  struct WorkCached : Work {
    QueueEntryCached& params;
    size_t i;

    WorkCached(CpuThreadImpl& self, QueueEntryCached& params) : Work(self, params), params(params) {}

    struct Info {
      int op = -1;
      int identifier = -1;
      size_t inputBytes;
      size_t outputBytes;
    };

    TemporaryBufferHandle infoBuffer;
    TemporaryBufferHandle remoteInfoBuffer;
    bool readDone = false;

    void step() {
      ENTER

      self.outStepValue(concurrencyIndex, 0) = stepValue;

      {
        infoBuffer = self.allocateTemporaryBuffer(sizeof(Info));
        Info i;
        i.op = params.op;
        i.identifier = params.identifier;
        i.inputBytes = params.inputBytes;
        i.outputBytes = params.outputBytes;
        std::memcpy(infoBuffer->cpuPointer, &i, sizeof(Info));
      }

      if (size > 1) {
        size_t i = self.rank + 1;
        if (i >= size) {
          i -= size;
        }
        CHECK(i >= 0 && i < size);
        DynamicAddresses& out = self.outDyn(concurrencyIndex, i);
        out.opType = opTypeCached;
        out.gatherAddress = (uintptr_t)infoBuffer->cpuPointer;
        out.gatherBytes = infoBuffer->bytes;
        out.stepValue = stepValue;
        for (size_t i = 0; i != maxDevices; ++i) {
          out.gatherKey[i] = infoBuffer.mr->mrs[i] ? infoBuffer.mr->mrs[i]->rkey : 0;
        }
        self.writeDyn(concurrencyIndex, i, i, rank);
      }

      if (size > 1) {
        i = self.rank - 1;
        if (i >= size) {
          i += size;
        }
        CHECK(i >= 0 && i < size);

        while (self.inDyn(concurrencyIndex, i).stepValue != stepValue) {
          YIELD
        }
        DynamicAddresses& dyn = self.inDyn(concurrencyIndex, i);
        CHECK_DYN(dyn, opTypeCached, stepValue, sizeof(Info));

        remoteInfoBuffer = self.allocateTemporaryBuffer(sizeof(Info));

        size_t di = 1 % self.devices.size();
        auto& dev = self.devices[di];
        self.readData(dev, i, remoteInfoBuffer->cpuPointer, remoteInfoBuffer.mr->mrs[di]->lkey,
            (void*)dyn.gatherAddress, dyn.gatherKey[di], sizeof(Info), self.makeCallback([this, i = this->i] {
              readDone = true;
            }));
      }

      if (size > 1) {
        i = self.rank - 1;
        if (i >= size) {
          i += size;
        }
        CHECK(i >= 0 && i < size);

        while (!readDone) {
          YIELD
        }
        Info ri;
        std::memcpy(&ri, remoteInfoBuffer->cpuPointer, sizeof(Info));

        if (ri.op != params.op || ri.identifier != params.identifier || ri.inputBytes != params.inputBytes ||
            ri.outputBytes != params.outputBytes) {
          // fixme: this op could be synchronous, and errors raised as exceptions
          fatal("Mismatched cached operation detected.\nLocal s: op %d, identifier %d, input bytes %d, output "
                "bytes %d\nRemote parameters: op %d, identifier %d, input bytes %d, output bytes %d\n",
              params.op, params.identifier, params.inputBytes, params.outputBytes, ri.op, ri.identifier, ri.inputBytes,
              ri.outputBytes);
        }
      }

      for (i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        self.writeStepValue(concurrencyIndex, i, 0, rank);
      }

      for (i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        while (self.inStepValue(concurrencyIndex, i) < stepValue) {
          YIELD
        }
      }

      // log.info("Cached op  ok!\n");

      infoBuffer = {};
      remoteInfoBuffer = {};

      self.cpuThread->freelistCached.push(&params);
      self.allocatorCached.deallocate(this);
      DONE
    }
  };

  void enqueueCallback(Function<void()> callback) {
    QueueEntryCallback* e = cpuThread->freelistCallback.pop();
    e->task = taskCallback;
    e->callback = std::move(callback);
    cpuThread->enqueue(e);
  }

  void executeCallback(QueueEntryCallback& params) {
    std::move(params.callback)();
    cpuThread->freelistCallback.push(&params);
  }

  void executeRdmaRegister(QueueEntryRdmaRegister& params) {
    MemoryRegistration* mr;
    if (params.isCuda) {
      mr = regMrCuda(params.address, params.bytes);
    } else {
      mr = regMr(params.address, params.bytes);
    }
    for (size_t i = 0; i < Group::maxDevices; ++i) {
      params.outRkeys[i] = mr->mrs[i] ? mr->mrs[i]->rkey : 0;
    }
    params.cpuDone->store(1, std::memory_order_release);
    futexWakeAll(params.cpuDone);
    cpuThread->freelistRdmaRegister.push(&params);
  }

  void executeRdmaRead(QueueEntryRdmaRead& params) {
    MemoryRegistration* dstMr;
    if (params.dstIsCuda) {
      dstMr = regMrCuda(params.dstAddress, params.dstBytes);
    } else {
      dstMr = regMr(params.dstAddress, params.dstBytes);
    }
    std::array<uint32_t, maxDevices> lkeys;
    for (size_t i = 0; i < maxDevices; ++i) {
      lkeys[i] = dstMr->mrs[i] ? dstMr->mrs[i]->lkey : 0;
    }
    readDataDistributed(params.srcRank, (void*)params.dstAddress, lkeys, (void*)params.srcAddress, params.srcRkeys,
        params.dstBytes, makeCallback([this, &params] {
          params.done->store(1, std::memory_order_release);
          futexWakeAll(params.done);
          cpuThread->freelistRdmaRead.push(&params);
        }));
  }

  uint32_t nextCreateQueueKey = random<uint32_t>();
  HashMap<uint32_t, QueueEntryCreateQueue*> queueCreateOutgoing;

  void executeCreateQueueNamed(QueueEntryCreateQueue& params) {
    uint32_t key = nextCreateQueueKey++;
    CHECK(params.name != std::nullopt);
    queueCreateOutgoing[key] = &params;
    auto buffer = allocateTemporaryBuffer(4096);
    size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeCreateQueueNamed, params.location), key,
        *params.name, (uint32_t)params.location, params.streaming);
    sendBuffer(params.location, std::move(buffer), bytes);
  }

  void onRecvCreateQueueNamed(std::string_view view, uint32_t sourceRank) {
    uint32_t key;
    std::string_view name;
    uint32_t location;
    bool streaming;
    deserializeBufferPart(view, key, name, location, streaming);

    std::string fullName = fmt::sprintf("%s.%s", group->name, name);

    QueueInfo qi = queueGetOrCreate(fullName, location, streaming);

    enqueueCallback([this, qi, key, sourceRank] {
      auto buffer = allocateTemporaryBuffer(4096);
      size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeCreateQueueNamedResult, sourceRank), key,
          qi.address, (uint32_t)qi.location, qi.streaming);
      sendBuffer(sourceRank, std::move(buffer), bytes);
    });
  }

  void onRecvCreateQueueNamedResult(std::string_view view) {
    uint32_t key;
    uintptr_t address;
    uint32_t location;
    bool streaming;
    deserializeBufferPart(view, key, address, location, streaming);
    enqueueCallback([this, key, address, location, streaming] {
      auto it = queueCreateOutgoing.find(key);
      CHECK(it != queueCreateOutgoing.end());
      auto& params = *it->second;
      queueCreateOutgoing.erase(it);

      *params.outAddress = address;
      if (params.location != location || params.streaming != streaming) {
        std::string err = fmt::sprintf(
            "Queue creation found mismatched parameters for named queue '%s'. Local: location %d, streaming %d. "
            "Remote: "
            "location %d, streaming %d\n",
            *params.name, params.location, params.streaming, location, streaming);
        if (params.outError) {
          *params.outError = err;
        } else {
          fatal("%s", err);
        }
      }

      cpuThread->freelistCreateQueue.push(&params);

      params.cpuDone->store(1);
      futexWakeAll(params.cpuDone);
    });
  }

  void onRecvMessageShutdown(std::string_view view, uint32_t sourceRank) {
    bool shutdown;
    deserializeBufferPart(view, shutdown);
    if (shutdown) {
      enqueueCallback([this, sourceRank] {
        auto& ri = rankInfo.at<RankInfo>(sourceRank);
        ri.shutdownTime = std::chrono::steady_clock::now();
        ri.shutdown = true;
      });
    } else {
      enqueueCallback([this, sourceRank] {
        auto& ri = rankInfo.at<RankInfo>(sourceRank);
        ri.exitTime = std::chrono::steady_clock::now();
        ri.exited = true;

        if (!shutdownInProgress) {
          log.error("%s: Received unexpected exit from %s\n", groupName(), rankName(sourceRank));
          setErrorState();
        }
      });
    }
  }

  struct WorkShutdown : Work {
    QueueEntry& params;
    size_t i;

    std::chrono::steady_clock::time_point start;
    std::shared_ptr<size_t> counter = std::make_shared<size_t>(0);

    WorkShutdown(CpuThreadImpl& self, QueueEntry& params) : Work(self, params), params(params) {}

    void step() {
      ENTER

      while (self.numActiveWorks != 1 || !self.activeDevices.empty()) {
        YIELD
      }

      if (!self.shutdownInProgress) {
        self.shutdownInProgress = true;

        start = std::chrono::steady_clock::now();

        for (size_t i = 0; i != size; ++i) {
          size_t di = 0;
          auto& dev = self.devices[di];
          ++*counter;
          auto buffer = self.allocateTemporaryBuffer(64);
          size_t bytes = serializeToUnchecked(buffer->cpuPointer, self.header(opTypeMessageShutdown, i), true);
          self.sendBuffer(i, std::move(buffer), bytes, [counter = counter] {
            --*counter;
          });
        }
        while (*counter && std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
          YIELD
        }
        for (i = 0; i != size; ++i) {
          while (!self.rankInfo.at<RankInfo>(i).shutdown &&
                 std::chrono::steady_clock::now() - start < std::chrono::seconds(self.cqErrored ? 4 : 8)) {
            YIELD
          }
        }
        {
          Vector<size_t> list;
          for (size_t i : range(size)) {
            if (!self.rankInfo.at<RankInfo>(i).shutdown) {
              list.push_back(i);
            }
          }
          if (!list.empty()) {
            std::vector<std::string> strs;
            for (size_t i : list) {
              strs.push_back(self.rankName(i));
            }
            log.error("%s: Failed to synchronize shutdown with the following ranks: %s\n", self.groupName(),
                fmt::to_string(fmt::join(strs, ", ")));
          }
        }
        self.shutdownSuppressErrors = true;
        start = std::chrono::steady_clock::now();
        for (size_t i = 0; i != size; ++i) {
          size_t di = 0;
          auto& dev = self.devices[di];
          ++*counter;
          auto buffer = self.allocateTemporaryBuffer(64);
          size_t bytes = serializeToUnchecked(buffer->cpuPointer, self.header(opTypeMessageShutdown, i), false);
          self.sendBuffer(i, std::move(buffer), bytes, [counter = counter] {
            --*counter;
          });
        }
        while (*counter && std::chrono::steady_clock::now() - start < std::chrono::seconds(5)) {
          YIELD
        }
        for (i = 0; i != size; ++i) {
          while (!self.rankInfo.at<RankInfo>(i).exited &&
                 std::chrono::steady_clock::now() - start < std::chrono::seconds(8)) {
            YIELD
          }
        }
      }

      self.cpuThread->freelistShutdown.push(&params);
      self.allocatorShutdown.deallocate(this);
      DONE
    }
  };

  struct WorkCustom : Work {
    QueueEntryCustom& params;
    size_t i;
    size_t o;

    CustomOpDescriptor* op = &*params.op;
    TemporaryBufferHandle listBuffer;

    struct InputTensor {
      uintptr_t address;
      size_t bytes;
      std::array<uint32_t, maxDevices> rkey;
    };

    static constexpr size_t maxInlinedInputs = 2;

    // (stepValue last: other ranks read the fields once they see the new step value, so none may follow it in the
    // write)
    struct Header {
      uint32_t opid;
      uint32_t numInputs;
      std::array<InputTensor, maxInlinedInputs> inputs;
      uint32_t flags;
      uint32_t stepValue;
    };
    static_assert(offsetof(Header, stepValue) + sizeof(uint32_t) == sizeof(Header));
    // Header::flags: the signals of writes into this rank's tensors cannot follow the data (see writeRdmaSignal): the
    // writers flush first
    static constexpr uint32_t flagSignalsNeedFlush = 1;

    struct alignas(128) Dyn : DynamicAddressesBase {
      Header custom;
    };
    static_assert(sizeof(Dyn) <= 128);

    TemporaryBufferHandle dynbuf;

    Vector<TemporaryBufferHandle> remoteInputList;
    Vector<bool> remoteInputListReady;
    // Reads of input lists in flight (lists of more than maxInlinedInputs tensors)
    size_t liveListReads = 0;
    // Per rank: its Header::flags
    Vector<uint32_t> remoteFlags;

    // The op's reads (op->ibReads), those with a signal index first. Those go in order, a part at a time, a read that
    // cannot be issued yet letting the ones after it go first (see tryRead): the reads with parts not issued yet form
    // a list in that order (firstPending, nextPending), and readIssued has the bytes of each read issued so far. The
    // others go as the parts that prepareReads orders (unsignaledParts: the read, and the part's offset in it and
    // size).
    static constexpr uint32_t noRead = ~(uint32_t)0;
    uint32_t firstPending = noRead;
    Vector<uint32_t> nextPending;
    Vector<size_t> readIssued;
    struct ReadPart {
      uint32_t read;
      size_t offset;
      size_t bytes;
    };
    Vector<ReadPart> unsignaledParts;
    size_t unsignaledIndex = 0;
    // Per read: the size of its parts (the last may be smaller), and its parts not completed yet
    Vector<size_t> readPartBytes;
    Vector<uint32_t> readPartsLeft;
    // The most waiting entries (reads, notifications) that a pass looks past for one that is ready: with rings (see
    // crossDomainReductions in compile_op.cc), most reads and notifications of an op wait for most of the op (for the
    // slots of the parts before them), and passes over all of them took longer than the transfers (reduce-scatter of
    // 256 MiB across 16 domains of 4 GPUs: 1920 reads and as many notifications per rank, 67 ms instead of 60). They
    // wait only for entries before them (and for other ranks), so the ones further on can wait.
    static constexpr size_t scanWindow = 64;
    bool readsPending() const {
      return firstPending != noRead || unsignaledIndex != unsignaledParts.size();
    }

    // Notifications (see CustomOpDescriptor::Notification): whether the kernel has set each one's flag
    Vector<uint8_t> flagSeen;
    size_t notificationsLeft = 0;
    size_t firstUnseen = 0;

    size_t liveReads = 0;
    // Per NIC: the parts of reads and writes in flight
    std::array<size_t, maxDevices> deviceLive{};

    // Reads and writes are split into parts of about partBytes, and each NIC has up to numParallel of them in flight
    static constexpr size_t partBytes = 128 << 10;
    static constexpr size_t numParallel = 8;
    // When an op has fewer reads (or writes) than NICs, smaller ones are split too, into parts of at least this size
    static constexpr size_t minPartBytes = 64 << 10;
    const size_t numDevices = self.devices.size();

    // The signals of reads into CUDA tensors (see writeRdmaSignal): per read with a signal index, the NICs that
    // carried parts of it (a bit each). The paths that carried parts of reads without a signal index into CUDA
    // tensors, and a destination of such a read for each, flushed at the end (see flushRead): a bit per NIC and kind
    // of registration of the destination on it (index 2 * NIC + Data Direct), since a flush read only orders the
    // writes that took its own path (a NIC that failed to register memory through Data Direct registers the later
    // tensors without it, see RdmaMr::dataDirect).
    Vector<uint8_t> readDevices;
    uint8_t unsignaledPaths = 0;
    std::array<std::pair<uintptr_t, uint32_t>, 2 * maxDevices> unsignaledDestination{};
    static_assert(2 * maxDevices <= 8);
    size_t liveFlushes = 0;

    // Writes (see CustomOpDescriptor::IbWrite). Once the kernel has written a write's flag, its parts go into the
    // queues of the NICs that carry them (queueWrites); each NIC writes its words of the write's signal once its parts
    // have completed (see writeRdmaSignal), or where the signal cannot follow the data, its queue has a flush read
    // after the parts, and the signal is written once the flushes have completed. Per write: the NICs that carry it
    // (writeDevices), per NIC the parts not completed (writePartsLeft), and the NICs whose words or flushes are still
    // to come (writePending).
    struct WritePart {
      uint32_t write;
      bool flush;
      size_t offset;
      size_t bytes;
    };
    std::array<Vector<WritePart>, maxDevices> writeQueues;
    std::array<size_t, maxDevices> writeQueueHeads{};
    Vector<uint8_t> writeQueued;
    Vector<uint8_t> writeDevices;
    Vector<uint8_t> writeFlushes;
    Vector<std::array<uint32_t, maxDevices>> writePartsLeft;
    Vector<uint32_t> writePending;
    size_t firstUnqueuedWrite = 0;
    size_t writesLeft = 0; // writes whose signal has not been sent
    size_t liveWrites = 0; // parts of writes and their flushes in flight

    // The writes of the op's step value to the peers in flight (see step()), and the next rank (as an offset from this
    // one) that postArrivals considers
    size_t livePeerStepValues = 0;
    size_t nextArrivalRank = 1;
    static constexpr size_t arrivalsPerRound = 8;

    Vector<MemoryRegistration*> outputMrs;

    WorkCustom(CpuThreadImpl& self, QueueEntryCustom& params) : Work(self, params), params(params) {}

    // Whether a read can be issued: its source's input list has arrived, the source has notified that the data is
    // there (if the read waits for that), and the destination is free (if it is a slot of a ring of the scratch
    // tensor that the kernel has used before: the kernel has set the done flag of that use)
    bool readReady(const CustomOpDescriptor::IbRead& read) {
      if (!remoteInputListReady[read.rank]) {
        return false;
      }
      if (read.waitSlot >= 0 &&
          self.inNotification(concurrencyIndex, read.rank, read.waitSlot) < stepValue + read.waitGeneration) {
        return false;
      }
      return read.freeFlag < 0 || cpuIn[16 + read.freeFlag] >= stepValue + read.freeGeneration;
    }

    // Write the notifications whose flags the kernel has set (they are in the order in which the kernel computes
    // their items; looking past at most scanWindow whose flags are not set)
    void forwardNotifications() {
      const auto& notifications = op->notifications;
      size_t waiting = 0;
      for (size_t i = firstUnseen; i != notifications.size() && waiting != scanWindow; ++i) {
        const auto& n = notifications[i];
        if (flagSeen[i]) {
          continue;
        }
        if (cpuIn[16 + n.flag] < stepValue + n.generation) {
          ++waiting;
          continue;
        }
        flagSeen[i] = 1;
        --notificationsLeft;
        self.trace(concurrencyIndex, 0, "%#x notify %d", stepValue, n.rank);
        self.writeNotification(concurrencyIndex, n.rank, n.slot, n.generation);
      }
      while (firstUnseen != notifications.size() && flagSeen[firstUnseen]) {
        ++firstUnseen;
      }
    }

    // Issues the next part of a read that is ready, on NIC di. Of the reads with a signal index, the first one that
    // is ready: a read that waits for a notification or for its destination to be free lets the reads after it go
    // first, until it can be issued, which keeps the order (moving all parts of the ready read ahead instead made
    // all-reduce of 16 MiB across 16 domains 10% slower: the order of the reads is what keeps one reader on each
    // partial rank at a time), looking past at most scanWindow reads that wait. Else the next part of the others, if
    // it is ready.
    void tryRead(size_t di) {
      const auto& ibReads = op->ibReads;
      size_t waiting = 0;
      for (uint32_t r = firstPending, previous = noRead; r != noRead && waiting != scanWindow;
          previous = r, r = nextPending[r]) {
        const auto& read = ibReads[r];
        if (!readReady(read)) {
          // (a read that waits only for its source's tensor list holds the ones after it: those of a proxied transfer
          // go in signal order)
          if (read.waitSlot < 0 && read.freeFlag < 0) {
            return;
          }
          ++waiting;
          continue;
        }
        const size_t offset = readIssued[r];
        const size_t bytes = std::min(readPartBytes[r], read.bytes - offset);
        readIssued[r] += bytes;
        if (readIssued[r] == read.bytes) {
          (previous == noRead ? firstPending : nextPending[previous]) = nextPending[r];
        }
        issueRead(di, r, offset, bytes);
        return;
      }
      if (unsignaledIndex != unsignaledParts.size() && readReady(ibReads[unsignaledParts[unsignaledIndex].read])) {
        const ReadPart part = unsignaledParts[unsignaledIndex++];
        issueRead(di, part.read, part.offset, part.bytes);
      }
    }

    // Issues the part of read r at `offset`, of `bytes` bytes, on NIC di
    void issueRead(size_t di, uint32_t r, size_t offset, size_t bytes) {
      const auto& read = op->ibReads[r];
      uint32_t source = read.rank;

      CHECK(source < size);
      InputTensor* arr = (InputTensor*)remoteInputList[source]->cpuPointer;
      CHECK(arr != nullptr);
      CHECK(remoteInputList[source]->bytes >= sizeof(InputTensor) * (read.inputIndex + 1));
      auto& it = arr[read.inputIndex];
      CHECK(read.inputOffset + read.bytes <= it.bytes);
      uintptr_t sourceAddress = it.address + read.inputOffset + offset;

      CHECK(params.outputs.size() > read.outputIndex);
      auto& ot = params.outputs[read.outputIndex];
      CHECK(ot->bytes() >= read.outputOffset + read.bytes);

      uintptr_t localAddress = ot->data() + read.outputOffset + offset;

      if (bytes == 0) {
        partDone(r);
        return;
      }

      // (data in CPU tensors needs no flush: the completion, written to host memory too, does not pass it)
      if (ot->isCuda) {
        if (read.signalIndex >= 0) {
          readDevices[r] |= 1 << di;
        } else {
          const size_t path = 2 * di + outputMrs[read.outputIndex]->mrs[di]->dataDirect;
          unsignaledPaths |= 1 << path;
          unsignaledDestination[path] = {localAddress, read.outputIndex};
        }
      }

      ++liveReads;
      ++deviceLive[di];
      self.trace(concurrencyIndex, 1 + di, "%#x read %d", stepValue, source);

      auto key = it.rkey;
      auto& dev = self.devices[di];
      self.readData(dev, source, (void*)localAddress, outputMrs[read.outputIndex]->mrs[di]->lkey, (void*)sourceAddress,
          key[di], bytes, self.makeCallback([this, di, r]() {
            self.trace(concurrencyIndex, 1 + di, "");
            --liveReads;
            --deviceLive[di];
            partDone(r);
          }));
    }

    // A part of a read is done: once all parts of a read with a signal index are, signal the kernel, which waits for
    // the data (see writeRdmaSignal): through every NIC that carried a part (loopback writes), or after flushes through
    // them; and if the read's source is a slot of a ring of the source rank's scratch tensor, tell that rank that the
    // slot is free (a credit, after the signal: any path will do, the data has left). Credits go through the control
    // queue pair, not the one of the reads from that rank (see IbCommon::controlQps; reduce-scatter of 256 MiB across
    // 16 domains of 4 GPUs along rings of the domains, groups of 4 parts, a credit per 2 MiB read: 52.1 ms instead of
    // 45.6); all credits to a signal go that way, so they arrive in order.
    void partDone(uint32_t r) {
      if (--readPartsLeft.at(r) != 0) {
        return;
      }
      const auto& read = op->ibReads[r];
      signalRead(read, readDevices[r]);
      if (read.creditSignal >= 0) {
        self.writeRdmaSignal(concurrencyIndex, self.devices[(rank + read.rank) % numDevices], read.rank,
            read.creditSignal, 0, maxDevices, read.creditGeneration, true);
      }
    }

    // Sets the signal of a read whose parts have all completed (if it has one); `used`: the NICs that carried parts
    void signalRead(const CustomOpDescriptor::IbRead& read, uint8_t used) {
      const int32_t signal = read.signalIndex;
      if (signal < 0) {
        return;
      }
      const size_t generation = read.signalGeneration;
      if (used == 0) {
        self.writeRdmaSignal(concurrencyIndex, self.devices[0], rank, signal, 0, maxDevices, generation);
        return;
      }
      const uint32_t outputIndex = read.outputIndex;
      if (self.signalFollows(outputMrs[outputIndex], used)) {
        for (size_t di : range(numDevices)) {
          if (used & (1 << di)) {
            auto [first, end] = signalWords(used, di);
            self.writeRdmaSignal(concurrencyIndex, self.devices[di], rank, signal, first, end, generation);
          }
        }
        return;
      }
      ++liveFlushes;
      auto callback = self.makeCallback([this, signal, generation]() {
        --liveFlushes;
        self.writeRdmaSignal(concurrencyIndex, self.devices[0], rank, signal, 0, maxDevices, generation);
      });
      const uintptr_t address = params.outputs[outputIndex]->data() + read.outputOffset;
      for (size_t di : range(numDevices)) {
        if (used & (1 << di)) {
          self.flushRead(self.devices[di], rank, address, outputMrs[outputIndex]->mrs[di]->rkey, callback);
        }
      }
    }

    // The flushes of the reads without a signal index (see flushRead), once they have completed: the stream goes on
    // once the CPU thread writes cpuOutBuffer slot 0, after them
    void flushUnsignaledReads() {
      CHECK(liveReads == 0);
      if (unsignaledPaths == 0) {
        return;
      }
      ++liveFlushes;
      auto callback = self.makeCallback([this]() {
        --liveFlushes;
      });
      for (size_t path : range(2 * numDevices)) {
        if (unsignaledPaths & (1 << path)) {
          auto [address, outputIndex] = unsignaledDestination[path];
          const size_t di = path / 2;
          self.flushRead(self.devices[di], rank, address, outputMrs[outputIndex]->mrs[di]->rkey, callback);
        }
      }
    }

    // The writes whose flags the kernel has written (and whose destination's tensor list has arrived) go into the
    // queues of their NICs: parts of about partBytes like reads, round robin over the NICs from one that rotates per
    // write; if the destination needs flushes (Header::flags), after its last part on each NIC a flush read through
    // that NIC (see issueWrites)
    void queueWrites() {
      const auto& writes = op->ibWrites;
      const bool fewWrites = writes.size() < numDevices;
      for (size_t w = firstUnqueuedWrite; w != writes.size(); ++w) {
        const auto& write = writes[w];
        if (writeQueued[w] || cpuIn[16 + write.flag] < stepValue || !remoteInputListReady[write.rank]) {
          continue;
        }
        writeQueued[w] = 1;
        checkWrite(write);
        size_t parts = std::max<size_t>((write.bytes + partBytes / 2) / partBytes, 1);
        if (fewWrites) {
          parts = std::max(parts, std::clamp(write.bytes / minPartBytes, (size_t)1, numDevices));
        }
        const size_t bytesPerPart = std::max(((write.bytes + parts - 1) / parts + 4095) & ~(size_t)4095, (size_t)4096);
        uint8_t used = 0;
        size_t di = w % numDevices;
        for (size_t offset = 0; offset < write.bytes; offset += bytesPerPart) {
          writeQueues[di].push_back({(uint32_t)w, false, offset, std::min(bytesPerPart, write.bytes - offset)});
          ++writePartsLeft[w][di];
          used |= 1 << di;
          di = di + 1 == numDevices ? 0 : di + 1;
        }
        writeDevices[w] = used;
        writeFlushes[w] = (remoteFlags[write.rank] & flagSignalsNeedFlush) != 0;
        if (used == 0) {
          self.writeRdmaSignal(concurrencyIndex, self.devices[0], write.rank, write.signalIndex, 0, maxDevices, 0);
          --writesLeft;
          continue;
        }
        for (size_t d : range(numDevices)) {
          if (used & (1 << d)) {
            ++writePending[w];
            if (writeFlushes[w]) {
              writeQueues[d].push_back({(uint32_t)w, true, 0, 0});
            }
          }
        }
      }
      while (firstUnqueuedWrite != writes.size() && writeQueued[firstUnqueuedWrite]) {
        ++firstUnqueuedWrite;
      }
    }

    // Posts the queued parts of writes, up to numParallel parts (of reads and writes) in flight per NIC. A NIC writes
    // its words of a write's signal once its parts of the write have completed (the destination's NIC has then
    // received them and issued their writes, which the signal's cannot pass), or with flushes, posts its flush read
    // right after its last part (the peer executes it after the writes before it, see Rdma::inOrder; otherwise once
    // they have completed), and the signal goes once all the flushes of the write have completed.
    void issueWrites() {
      for (size_t di : range(numDevices)) {
        auto& queue = writeQueues[di];
        size_t& head = writeQueueHeads[di];
        auto& dev = self.devices[di];
        while (head != queue.size()) {
          const WritePart part = queue[head];
          const auto& write = op->ibWrites[part.write];
          const InputTensor& dst = ((const InputTensor*)remoteInputList[write.rank]->cpuPointer)[write.destinationIndex];
          if (part.flush) {
            if (!dev.rdma->inOrder && writePartsLeft[part.write][di] != 0) {
              break;
            }
            ++liveWrites;
            self.flushRead(dev, write.rank, dst.address + write.destinationOffset, dst.rkey[di],
                self.makeCallback([this, w = part.write]() {
                  --liveWrites;
                  if (--writePending[w] == 0) {
                    const auto& write = op->ibWrites[w];
                    self.writeRdmaSignal(
                        concurrencyIndex, self.devices[0], write.rank, write.signalIndex, 0, maxDevices, 0);
                    --writesLeft;
                  }
                }));
          } else {
            if (deviceLive[di] >= numParallel) {
              break;
            }
            ++liveWrites;
            ++deviceLive[di];
            self.trace(concurrencyIndex, 1 + di, "%#x write %d", stepValue, write.rank);
            auto& src = params.outputs.back();
            // (no inline data: the source is GPU memory)
            self.writeDataNoInline(dev, write.rank, (void*)(src->data() + write.sourceOffset + part.offset),
                outputMrs.back()->mrs[di]->lkey, (void*)(dst.address + write.destinationOffset + part.offset),
                dst.rkey[di], part.bytes, self.makeCallback([this, di, w = part.write]() {
                  self.trace(concurrencyIndex, 1 + di, "");
                  --liveWrites;
                  --deviceLive[di];
                  if (--writePartsLeft[w][di] == 0 && !writeFlushes[w]) {
                    const auto& write = op->ibWrites[w];
                    auto [first, end] = signalWords(writeDevices[w], di);
                    self.writeRdmaSignal(
                        concurrencyIndex, self.devices[di], write.rank, write.signalIndex, first, end, 0);
                    if (--writePending[w] == 0) {
                      --writesLeft;
                    }
                  }
                }));
          }
          ++head;
        }
      }
    }

    // Writes the rank's arrival value (Group::cpuArrivalValues) into the memory of up to n more of the ranks that are
    // not peers, in the order rank + 1, rank + 2, ...
    void postArrivals(size_t n) {
      while (n != 0 && nextArrivalRank != size) {
        size_t r = rank + nextArrivalRank;
        if (r >= size) {
          r -= size;
        }
        ++nextArrivalRank;
        if (!op->isPeer[r]) {
          self.writeArrivalValue(concurrencyIndex, r);
          --n;
        }
      }
    }

    // Whether rank i, not a peer, has reached the op: its arrival value is at least the op's step value. With the
    // op's step value, the op ids must match (the peers compare theirs in their tensor lists). A larger value is from
    // a later op of the slot, so rank i has finished this one, and has seen this rank's arrival value for it (this
    // rank's later ones come after this wait), so it has checked the pair.
    bool arrived(size_t i) {
      const ArrivalValue v = self.loadArrivalValue(concurrencyIndex, i);
      if (v.stepValue == stepValue && v.opid != op->id) [[unlikely]] {
        fatal("Mismatched custom operation: Local id is %d, remote id is %d (rank %d, in its arrival value)", op->id,
            v.opid, i);
      }
      return v.stepValue >= stepValue;
    }

    // Checks the writes against the tensors of their destinations, once their lists have arrived
    void checkWrite(const CustomOpDescriptor::IbWrite& write) {
      CHECK(write.rank < size && write.rank != rank);
      CHECK(remoteInputList[write.rank]->bytes >= sizeof(InputTensor) * (write.destinationIndex + 1));
      const InputTensor& dst = ((const InputTensor*)remoteInputList[write.rank]->cpuPointer)[write.destinationIndex];
      CHECK(write.destinationOffset + write.bytes <= dst.bytes);
      CHECK(op->scratchBytes != 0 && write.sourceOffset + write.bytes <= params.outputs.back()->bytes());
    }

    // Splits the op's reads into parts of about partBytes (reads), in the order in which they are issued, each on the
    // NIC with the fewest in flight (up to numParallel). Reads that signal the kernel come first, in the order of the
    // descriptor (proxy reads in signal order, the chunks of a transfer in order: copies on this and other ranks wait
    // for them; reads of reductions across NVLink domains in the order of their edges, see crossDomainReductions). The
    // others follow in the order compile_op gave them (by source, rotated per rank so that every source has one reader
    // per step), op->readSources at a time. With 1, a rank reads from one or two sources at a time, which keeps the
    // ranks in step when they all read equal amounts. Larger parts let the ranks drift apart, until some sources have
    // several readers and others none: with one part per NIC and two in flight, a source port of an all-to-all
    // (1 GiB per rank, 32 GB300 over RoCE) had no reader 13% of the time and three or more 30%, and the op took
    // 14.8 ms instead of 10.2 ms. One part in flight per NIC keeps the ranks in step, but then each part costs a
    // round trip (about 13 us from post to completion besides the transfer). With more than 1 (unequal amounts),
    // the parts of that many reads go round robin, and a read whose parts have all been issued makes room for the
    // next one. With fewer reads than NICs, smaller reads are split as well, into parts of at least minPartBytes,
    // so that every NIC gets some. The parts of a read share its signal: it is written when the last part has
    // completed.
    void prepareReads() {
      const auto& ibReads = op->ibReads;
      const bool fewReads = ibReads.size() < numDevices;
      nextPending.resize(ibReads.size());
      std::ranges::fill(nextPending, noRead);
      readIssued.resize(ibReads.size());
      readPartBytes.resize(ibReads.size());
      readPartsLeft.resize(ibReads.size());
      readDevices.resize(ibReads.size());
      // The number of parts of each read without a signal index, in order
      Vector<size_t> runs;
      uint32_t lastPending = noRead;
      for (size_t i : indices(ibReads)) {
        const auto& r = ibReads[i];
        // (the values of its signal and credit are in the op's generations, see step())
        CHECK(
            r.signalGeneration < op->numGenerations && (r.creditSignal < 0 || r.creditGeneration < op->numGenerations));
        size_t parts = std::max<size_t>((r.bytes + partBytes / 2) / partBytes, 1);
        if (fewReads) {
          parts = std::max(parts, std::clamp(r.bytes / minPartBytes, (size_t)1, numDevices));
        }
        const size_t bytesPerPart = std::max(((r.bytes + parts - 1) / parts + 4095) & ~(size_t)4095, (size_t)4096);
        // (as many parts as tryRead issues: of bytesPerPart, the last one with the rest, at least one)
        parts = std::max<size_t>((r.bytes + bytesPerPart - 1) / bytesPerPart, 1);
        readPartBytes[i] = bytesPerPart;
        readPartsLeft[i] = parts;
        if (r.signalIndex >= 0) {
          // (the reads with a signal index come first)
          CHECK(unsignaledParts.empty());
          (lastPending == noRead ? firstPending : nextPending[lastPending]) = i;
          lastPending = i;
          continue;
        }
        for (size_t p : range(parts)) {
          const size_t offset = bytesPerPart * p;
          unsignaledParts.push_back({(uint32_t)i, offset, std::min(bytesPerPart, r.bytes - offset)});
        }
        runs.push_back(parts);
      }
      if (op->readSources <= 1) {
        return;
      }
      // One run of parts per read, in order
      Vector<ReadPart> order;
      order.reserve(unsignaledParts.size());
      // [next part, end) of the reads being read
      Vector<std::pair<size_t, size_t>> active;
      size_t nextRun = 0;
      size_t nextPart = 0;
      auto take = [&]() {
        std::pair<size_t, size_t> run(nextPart, nextPart + runs[nextRun]);
        nextPart = run.second;
        ++nextRun;
        return run;
      };
      while (active.size() < op->readSources && nextRun != runs.size()) {
        active.push_back(take());
      }
      size_t i = 0;
      while (!active.empty()) {
        auto& run = active[i];
        order.push_back(unsignaledParts[run.first++]);
        if (run.first != run.second) {
          ++i;
        } else if (nextRun != runs.size()) {
          run = take();
          ++i;
        } else {
          active.erase(&run);
        }
        if (i >= active.size()) {
          i = 0;
        }
      }
      CHECK(order.size() == unsignaledParts.size());
      unsignaledParts = std::move(order);
    }

    void step() {
      ENTER

      // Trace spans (moodist.enable_profiling): one per phase, and one per peer the CPU thread actually waits for
      self.trace(concurrencyIndex, 0, "%#x start", stepValue);
      if (profilingEnabled && params.enqueueTime.time_since_epoch().count() != 0) [[unlikely]] {
        // From the call to the CPU thread picking the op up (tid traceTidPickup)
        self.traceEvents.emplace_back();
        auto& e = self.traceEvents.back();
        e.name =
            fmt::sprintf("%#x enqueued on cpu %d, picked up on cpu %d", stepValue, params.enqueueCpu, sched_getcpu());
        e.begin = params.enqueueTime;
        e.end = std::chrono::system_clock::now();
        e.threadId = traceTidPickup;
      }

      self.outStepValue(concurrencyIndex, 0) = stepValue;
      self.outStepValue(concurrencyIndex, 1) = stepValue + 1;
      self.outStepValue(concurrencyIndex, arrivalWords) = stepValue;
      self.outStepValue(concurrencyIndex, arrivalWords + 1) = op->id;
      // The values of the signals and notifications that the CPU thread writes (see CustomOpDescriptor::IbRead)
      CHECK(op->numGenerations >= 1 && op->numGenerations <= compile_op::maxGenerations);
      for (size_t g : range(op->numGenerations)) {
        for (size_t d : range(maxDevices)) {
          self.generationValue(concurrencyIndex, g, d) = stepValue + g;
        }
      }

      // log.info("CUSTOM %#x/%d enter\n", stepValue, concurrencyIndex);
      {
        CHECK(params.inputs.size() == op->inputs.size() + op->inputCopies.size());
        CHECK(params.outputs.size() == op->outputs.size() + op->outputCopies.size() + (op->scratchBytes != 0));

        outputMrs.resize(params.outputs.size());
        for (size_t i : indices(outputMrs)) {
          auto& t = params.outputs[i];
          outputMrs[i] = t->isCuda ? self.regMrCuda(t->data(), t->bytes()) : self.regMr(t->data(), t->bytes());
        }

        Dyn& out = (Dyn&)self.outDyn(concurrencyIndex, 0);
        out.opType = opTypeCustom;
        out.gatherBytes = 0;
        out.stepValue = stepValue;

        // The tensors that other ranks read from: the inputs (and input copies), and the scratch tensor, if any (the
        // last tensor, with partial results of reductions, see CustomOpDescriptor::scratchBytes), which other ranks
        // may also write into (whose signals then need flushes if they cannot follow the data on any NIC)
        const size_t numTensors = params.inputs.size() + (op->scratchBytes != 0);
        out.custom.opid = op->id;
        out.custom.stepValue = stepValue;
        out.custom.numInputs = numTensors;
        out.custom.flags = op->scratchBytes != 0 && !self.signalFollows(outputMrs.back(), (1 << numDevices) - 1)
                               ? flagSignalsNeedFlush
                               : 0;

        auto describe = [&](size_t index, InputTensor& v) {
          const bool input = index < params.inputs.size();
          auto& tensor = input ? params.inputs[index] : params.outputs.back();
          v.address = tensor->data();
          v.bytes = tensor->bytes();
          auto* mr = !input           ? outputMrs.back()
                     : tensor->isCuda ? self.regMrCuda(v.address, v.bytes)
                                      : self.regMr(v.address, v.bytes);
          for (size_t d = 0; d != maxDevices; ++d) {
            v.rkey[d] = mr->mrs[d] ? mr->mrs[d]->rkey : 0;
          }
        };

        if (numTensors <= maxInlinedInputs) {

          for (size_t i : range(numTensors)) {
            describe(i, out.custom.inputs[i]);
          }

          out.gatherAddress = 0;

        } else {

          dynbuf = self.allocateTemporaryBuffer(sizeof(InputTensor) * numTensors);

          InputTensor* l = (InputTensor*)dynbuf->cpuPointer;

          for (size_t i : range(numTensors)) {
            describe(i, l[i]);
          }

          out.gatherAddress = (uintptr_t)dynbuf->cpuPointer;
          for (size_t i = 0; i != maxDevices; ++i) {
            out.gatherKey[i] = dynbuf.mr->mrs[i] ? dynbuf.mr->mrs[i]->rkey : 0;
          }
        }

        // (to the peers, see CustomOpDescriptor::peers, and to itself if it reads from its own tensors; into words of
        // their own, Group::customDyns: the process group's all-gather ring keeps its neighbor's lists in list slots 0
        // and 1, which the lists of ranks 0 and 1 overwrote when a peer was still in the ring op 16 ops earlier)
        if (op->dynSelf) {
          self.writeCustomDyn(concurrencyIndex, rank);
        }
        for (uint32_t peer : op->peers) {
          self.writeCustomDyn(concurrencyIndex, peer);
        }
      }

      CHECK(remoteInputList.empty());
      CHECK(remoteInputListReady.empty());
      remoteInputList.resize(size);
      remoteInputListReady.resize(size);
      remoteFlags.resize(size);

      // (its own list first if it reads from itself, then the peers' in the order rank - 1, rank - 2, ...)
      for (o = op->dynSelf ? 0 : 1; o <= op->peers.size(); ++o) {
        i = o == 0 ? rank : op->peers[op->peers.size() - o];
        CHECK(i >= 0 && i < size);

        if (self.inCustomDyn(concurrencyIndex, i).stepValue != stepValue) {
          self.trace(concurrencyIndex, 0, "%#x wait dyn %d", stepValue, i);
        }
        while (self.inCustomDyn(concurrencyIndex, i).stepValue != stepValue) {
          YIELD
        }
        {
          Dyn& dyn = (Dyn&)self.inCustomDyn(concurrencyIndex, i);
          CHECK_DYN(dyn, opTypeCustom, stepValue, 0);
        }

        while (((Dyn&)self.inCustomDyn(concurrencyIndex, i)).custom.stepValue != stepValue) {
          YIELD
        }
        Dyn& dyn = (Dyn&)self.inCustomDyn(concurrencyIndex, i);

        if (dyn.custom.opid != op->id) {
          fatal("Mismatched custom operation: Local id is %d, remote id is %d (rank %d)", op->id, dyn.custom.opid, i);
        }
        remoteFlags[i] = dyn.custom.flags;

        auto& targetBuffer = remoteInputList[i];
        size_t bytes = sizeof(InputTensor) * dyn.custom.numInputs;
        targetBuffer = self.allocateTemporaryBuffer(bytes);

        if (dyn.custom.numInputs <= maxInlinedInputs) {
          std::memcpy(targetBuffer->cpuPointer, dyn.custom.inputs.data(), sizeof(InputTensor) * dyn.custom.numInputs);
          remoteInputListReady[i] = true;
        } else {
          size_t di = o % self.devices.size();
          auto& dev = self.devices[di];
          ++liveListReads;
          self.readData(dev, i, targetBuffer->cpuPointer, targetBuffer.mr->mrs[di]->lkey, (void*)dyn.gatherAddress,
              dyn.gatherKey[di], bytes, self.makeCallback([this, i = this->i] {
                remoteInputListReady[i] = true;
                --liveListReads;
              }));
        }
      }

      prepareReads();

      flagSeen.resize(op->notifications.size());
      notificationsLeft = op->notifications.size();
      for (const auto& n : op->notifications) {
        CHECK(n.generation < op->numGenerations);
      }

      writeQueued.resize(op->ibWrites.size());
      writeDevices.resize(op->ibWrites.size());
      writeFlushes.resize(op->ibWrites.size());
      writePartsLeft.resize(op->ibWrites.size());
      writePending.resize(op->ibWrites.size());
      writesLeft = op->ibWrites.size();

      if (params.anyCuda) {
        // wait for kernel
        if (cpuIn[0] < stepValue) {
          self.trace(concurrencyIndex, 0, "%#x wait kernel", stepValue);
        }
        while (cpuIn[0] < stepValue) {
          YIELD
        }
      }
      // Step values with the peers (the rank's stream has reached the op, so its tensors are ready), whose tensors the
      // reads and writes below use; the other ranks get arrival values, a few at a time while the reads and writes go
      // on (postArrivals), which only the end of the op waits for
      self.trace(concurrencyIndex, 0, "%#x step exchange", stepValue);
      for (uint32_t peer : op->peers) {
        self.writeStepValue(concurrencyIndex, peer, 0, rank, &livePeerStepValues);
      }
      for (o = 1; o <= op->peers.size(); ++o) {
        i = op->peers[op->peers.size() - o];
        if (self.inStepValue(concurrencyIndex, i) < stepValue) {
          self.trace(concurrencyIndex, 0, "%#x wait step %d", stepValue, i);
        }
        while (self.inStepValue(concurrencyIndex, i) < stepValue) {
          YIELD
        }
      }

      self.trace(concurrencyIndex, 0, "%#x reads", stepValue);

      // Reads, the notifications of partial results this rank's kernel computes for other ranks, and the writes of
      // those that it writes to them (the writes go first: the kernels of other ranks wait for them)
      while (readsPending() || notificationsLeft != 0 || writesLeft != 0) {
        if (notificationsLeft != 0) {
          forwardNotifications();
        }
        if (firstUnqueuedWrite != op->ibWrites.size()) {
          queueWrites();
        }
        issueWrites();
        if (readsPending()) {
          size_t bestDevice = 0;
          size_t bestLive = numParallel;
          for (size_t di = 0; di != numDevices; ++di) {
            if (deviceLive[di] < bestLive) {
              bestLive = deviceLive[di];
              bestDevice = di;
            }
          }
          if (bestLive < numParallel) {
            tryRead(bestDevice);
          }
        }
        postArrivals(arrivalsPerRound);
        YIELD
      }
      postArrivals(size);

      // (and of the reads of input lists: nothing else waits for those of the peers that this rank neither reads from
      // nor writes to, and their completions use this Work)
      if (liveReads || liveWrites || liveListReads) {
        self.trace(concurrencyIndex, 0, "%#x wait read and write completions", stepValue);
      }
      while (liveReads || liveWrites || liveListReads) {
        YIELD
      }
      flushUnsignaledReads();

      // The step values to the peers have arrived: stepValue + 1 goes to the same address, and RDMA writes may land
      // out of order, so it could otherwise be overwritten by stepValue
      while (livePeerStepValues != 0) {
        YIELD
      }

      for (uint32_t peer : op->peers) {
        self.writeStepValue(concurrencyIndex, peer, 1, rank);
      }
      self.trace(concurrencyIndex, 0, "%#x wait for remote reads", stepValue);
      for (o = 1; o <= op->peers.size(); ++o) {
        i = op->peers[op->peers.size() - o];
        if (self.inStepValue(concurrencyIndex, i) < stepValue + 1) {
          self.trace(concurrencyIndex, 0, "%#x wait reads of %d", stepValue, i);
        }
        while (self.inStepValue(concurrencyIndex, i) < stepValue + 1) {
          YIELD
        }
      }

      // All other ranks have reached the op too (their arrival values) before its stream goes on, so that no rank gets
      // more than an op ahead of another, as when all ranks waited for each other's reads. The op itself does not need
      // this. It writes into another rank's memory only once that rank's tensor list for the same op has arrived (that
      // rank is then done with the slot's previous op, which may be a process group collective waiting for exact
      // values in the slot's words), except into words of their own: its tensor list, which its peers read before the
      // op can end on this rank, and its arrival values. And the words that it waits on (step values, notifications,
      // arrival values) only the rank that they come from writes, with ever larger values (the notifications have
      // words of their own, see Group::customNotifications). But the process group collectives need it: they write
      // their tensor lists into the memory of other ranks without waiting for them to reach the op (the all-gather
      // ring also writes step values under the indices of other ranks, and keeps its neighbor's lists in list slots 0
      // and 1, which the other collectives use for ranks 0 and 1), which is safe only if those ranks have finished the
      // slot's previous op (16 ops earlier) by then. On one stream, the collectives that wait for all ranks (barriers,
      // all-gathers, reduce-scatters, all-reduces) keep every rank within an op of every other too; the peers alone
      // would let a rank get any number of ops ahead of the ranks that it does not exchange with.
      for (o = 1; o < size; ++o) {
        i = self.rank - o;
        if (i >= size) {
          i += size;
        }
        if (op->isPeer[i]) {
          continue;
        }
        if (!arrived(i)) {
          self.trace(concurrencyIndex, 0, "%#x wait arrival of %d", stepValue, i);
        }
        while (!arrived(i)) {
          YIELD
        }
      }

      // (the flushes of reads, before the stream goes on to use their data)
      if (liveFlushes) {
        self.trace(concurrencyIndex, 0, "%#x wait flushes", stepValue);
      }
      while (liveFlushes) {
        YIELD
      }

      if (params.anyCuda) {
        self.writeCpuOut(concurrencyIndex, 0, 0);
      }
      self.trace(concurrencyIndex, 0, "");

      params.future->done = 1;
      futexWakeAll(&params.future->done);
      params.future.reset();

      params.op.reset();

      dynbuf = {};
      remoteInputList.clear();

      params.inputs.clear();
      params.outputs.clear();

      self.cpuThread->freelistCustom.push(&params);
      self.allocatorCustom.deallocate(this);
      DONE
    }
  };

  void executeFabricMap(QueueEntryFabricMap& params) {
    auto buffer = allocateTemporaryBuffer(4096);
    size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeFabricMap, params.targetRank),
        params.chunkIndex, params.handle, params.offset, params.bytes);
    sendBuffer(params.targetRank, std::move(buffer), bytes);
  }

  void onRecvFabricMap(std::string_view view, uint32_t source) {
    QueueEntryFabricMap p;
    deserializeBufferPart(view, p.chunkIndex, p.handle, p.offset, p.bytes);
    CHECK(cpuThread->onFabricMap != nullptr);
    uintptr_t base = cpuThread->onFabricMap(source, p);
    enqueueCallback([this, base, source, chunkIndex = p.chunkIndex] {
      auto buffer = allocateTemporaryBuffer(4096);
      size_t bytes = serializeToUnchecked(buffer->cpuPointer, header(opTypeFabricMapDone, source), chunkIndex, base);
      sendBuffer(source, std::move(buffer), bytes);
    });
  }

  void onRecvFabricMapDone(std::string_view view, uint32_t source) {
    size_t chunkIndex;
    uintptr_t base;
    deserializeBufferPart(view, chunkIndex, base);
    CHECK(cpuThread->onFabricMapDone != nullptr);
    cpuThread->onFabricMapDone(source, chunkIndex, base);
  }

  template<typename T>
  struct WorkAllocator {
    PoolAllocator<T> allocator;
    IntrusiveList<WorkBase, &WorkBase::freeLink> free;
    size_t nAllocated = 0;
    template<typename... Args>
    T* allocate(Args&&... args) {
      if (!free.empty()) [[likely]] {
        T* r = (T*)&free.front();
        free.pop_front();
        r->~T();
        new (r) T(std::forward<Args>(args)...);
        return r;
      }
      ++nAllocated;
      log.debug("Work %s -> %d\n", typeid(T).name(), nAllocated);
      return allocator.allocate(std::forward<Args>(args)...);
    }
    void deallocate(T* ptr) {
      free.push_front(*ptr);
    }
  };
  WorkAllocator<WorkBarrier> allocatorBarrier;
  WorkAllocator<WorkInternalBarrier> allocatorInternalBarrier;
  WorkAllocator<WorkAllGatherCpu> allocatorAllGatherCpu;
  WorkAllocator<WorkReduceScatterCpu> allocatorReduceScatterCpu;
  WorkAllocator<WorkBroadcastRingCpu> allocatorBroadcastRingCpu;
  WorkAllocator<WorkCreateQueue> allocatorCreateQueue;
  // WorkAllocator<WorkQueuePut> allocatorQueuePut;
  WorkAllocator<WorkCat> allocatorCat;
  WorkAllocator<WorkCached> allocatorCached;
  WorkAllocator<WorkShutdown> allocatorShutdown;
  WorkAllocator<WorkCustom> allocatorCustom;

  size_t numActiveWorks = 0;

  size_t localHeartbeatValue = 1;
  struct HeartbeatState {
    size_t value = 0;
    std::optional<std::chrono::steady_clock::time_point> activeWrite;
    std::optional<std::chrono::steady_clock::time_point> valueChangeTime;

    bool hasReportedExit = false;
  };
  IVector<HeartbeatState> heartbeatStates{size};

  std::chrono::steady_clock::duration heartbeatInterval = std::chrono::seconds(30);

  size_t heartbeatIndex = 0;

  bool shutdownInProgress = false;
  bool shutdownScheduled = false;
  std::atomic_bool shutdownSuppressErrors = false;

  void heartbeat(std::chrono::steady_clock::time_point now) {

    if (shutdownInProgress) {
      return;
    }

    ++localHeartbeatValue;

    CHECK(heartbeatStates.size() == size);

    size_t nRange = std::min(size - 1, (size_t)16);

    // (registered at the start, see entry)
    auto* mr = regMr((uintptr_t)&localHeartbeatValue, sizeof(localHeartbeatValue));

    for (size_t i : range(size)) {
      if (i == rank) {
        continue;
      }
      auto& state = heartbeatStates[i];
      if (rankInfo.at<RankInfo>(i).exited) {
        if (!state.hasReportedExit) {
          state.hasReportedExit = true;
          log.error("%s: %s has exited\n", groupName(), rankName(i));
        }
      }
      if (cqErrored) {
        continue;
      }
      if (size + i < size + rank - nRange || i > rank + nRange) {
        continue;
      }
      size_t currentValue = rankInfo.at<RankInfo>(i).heartbeatValue;
      if (!state.valueChangeTime || currentValue != state.value) {
        state.valueChangeTime = now;
        state.value = currentValue;
      } else {
        if (now - *state.valueChangeTime >= heartbeatInterval * 3) {
          float s = seconds(now - *state.valueChangeTime);
          log.error("%s: Heartbeat timeout: %s has not responded in %g seconds\n", groupName(), rankName(i), s);

          if (now - *state.valueChangeTime >= heartbeatInterval * 6) {
            setErrorState();
          }
        }
      }
      if (state.activeWrite) {
        log.error("%s: Heartbeat to %s has not completed after %g seconds\n", groupName(), rankName(i),
            seconds(now - *state.activeWrite));

        if (now - *state.activeWrite >= heartbeatInterval * 6) {
          setErrorState();
        }
        continue;
      }
      state.activeWrite = now;
      size_t di = (localHeartbeatValue + rank + i) % devices.size();
      auto& dev = devices[di];
      writeData(dev, i, &localHeartbeatValue, mr->mrs[di]->lkey,
          (void*)(remoteRankInfo[i].address + sizeof(RankInfo) * rank + offsetof(RankInfo, heartbeatValue)),
          remoteRankInfo[i].keys[di], sizeof(uint32_t), makeCallback([&state] {
            state.activeWrite.reset();
          }));
    }
  }

  void sendRankInfo() {
    TimeoutHelper timeout(group, "sending its rank's information");

    auto buffer = group->allocateCpu(sizeof(RankInfo));
    RankInfo* local = new (buffer.cpuPointer) RankInfo();
    ::gethostname(local->hostname.data(), local->hostname.size());
    local->pid = ::getpid();
    local->device = group->deviceIndex;
    local->heartbeatValue = 1;

    auto* mr = regMr((uintptr_t)local, sizeof(RankInfo));

    auto activeWrites = std::make_shared<size_t>(0);

    for (size_t i = 0; i != size; ++i) {
      size_t di = 0;
      auto& dev = devices[di];
      ++*activeWrites;
      writeData(dev, i, local, mr->mrs[di]->lkey, (void*)(remoteRankInfo[i].address + sizeof(RankInfo) * rank),
          remoteRankInfo[i].keys[di], sizeof(RankInfo), makeCallback([activeWrites] {
            --*activeWrites;
          }));
    }

    while (*activeWrites && !errorState) {
      poll();
      timeout.check();
    }
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i != size; ++i) {
      while (rankInfo.at<RankInfo>(i).heartbeatValue == 0 &&
             std::chrono::steady_clock::now() - start < std::chrono::seconds(30) && !errorState) {
        poll();
      }
    }
  }

  std::string groupName() {
    return fmt::sprintf("(group '%s' of size %d, rank %d)", group->name, size, rank);
  }

  std::string rankName(size_t i) {
    const RankInfo& ri = rankInfo.at<RankInfo>(i);
    std::string hostname(ri.hostname.data(), strnlen(ri.hostname.data(), ri.hostname.size()));
    if (hostname.empty()) {
      hostname = "<unknown>";
    }
    return fmt::sprintf("(rank %d, hostname %s, pid %d, device %d)", i, hostname, ri.pid, ri.device);
  }

  int mainThreadId = currentThreadId;

  long prevMinorFaults = 0;
  long prevMajorFaults = 0;
  long prevContextSwitchesVoluntary = 0;
  long prevContextSwitchesPreempted = 0;

  long waitCount = 0;
  long prevWaitCount = 0;

  void checkPreemptions(bool report) {
    struct rusage usage;
    if (getrusage(RUSAGE_THREAD, &usage)) {
      log.error("getrusage failed: %s\n", std::strerror(errno));
      return;
    }
    int minorFaults = usage.ru_minflt - std::exchange(prevMinorFaults, usage.ru_minflt);
    int majorFaults = usage.ru_majflt - std::exchange(prevMajorFaults, usage.ru_majflt);
    int voluntary = usage.ru_nvcsw - std::exchange(prevContextSwitchesVoluntary, usage.ru_nvcsw);
    int preempted = usage.ru_nivcsw - std::exchange(prevContextSwitchesPreempted, usage.ru_nivcsw);
    int waits = waitCount - std::exchange(prevWaitCount, waitCount);
    if (report) {
      if (minorFaults | majorFaults) {
        log.info("cpu thread faults: %d minor, %d major\n", minorFaults, majorFaults);
      }
      if (voluntary | preempted) {
        log.info("cpu thread context switches: %d voluntary, %d preempted\n", voluntary, preempted);
      }

      log.info("%d waits\n", waits);
    }
  }

  void entry() {

    sendRankInfo();

    // The heartbeats' source word, registered now: registered at the first heartbeat, 30 s in, it held this thread up
    // for 1.3-1.6 ms (4 ConnectX-8 NICs), and an iteration of a reduce-scatter of 8 MiB per rank across 16 nodes of 4
    // GPUs took 4-5.5 ms instead of 1.5
    regMr((uintptr_t)&localHeartbeatValue, sizeof(localHeartbeatValue));

    runTests();

    initReceives();

    log.info("%s started", groupName());

    cpuThread->ready = true;

    Vector<IntrusiveList<WorkBase, &WorkBase::concurrencyLink>> concurrency;
    concurrency.resize(Group::maxConcurrency);
    Vector<IntrusiveList<WorkBase, &WorkBase::streamLink>> streams;
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

    std::chrono::steady_clock::time_point prevHeartbeatTime = std::chrono::steady_clock::now();

    auto prevRunTime = std::chrono::steady_clock::now();

    bool debugCpuTime = false;
    bool heartbeatEnabled = true;
    {
      const char* c = std::getenv("MOODIST_DEBUG_CPU");
      if (c && !strcmp(c, "1")) {
        debugCpuTime = true;
      }
      c = std::getenv("MOODIST_DISABLE_HEARTBEAT");
      if (c && !strcmp(c, "1")) {
        heartbeatEnabled = false;
      }
    }

    // While profiling, the loop's sleeps, stalls of 20 us or more (e.g. the thread was descheduled) and moves to
    // another core become trace events (tid traceTidLoop), with the core numbers
    std::chrono::system_clock::time_point traceLoopTime{};
    int traceLoopCpu = -1;
    bool traceLoopSlept = false;

    auto traceLoop = [&]() {
      if (profilingEnabled) [[unlikely]] {
        auto t = std::chrono::system_clock::now();
        int cpu = sched_getcpu();
        if (traceLoopTime.time_since_epoch().count() != 0 &&
            (t - traceLoopTime >= std::chrono::microseconds(20) || cpu != traceLoopCpu)) {
          traceEvents.emplace_back();
          TraceEvent& e = traceEvents.back();
          e.name = fmt::sprintf("%s cpu %d -> %d", traceLoopSlept ? "sleep" : "stall", traceLoopCpu, cpu);
          e.begin = traceLoopTime;
          e.end = t;
          e.threadId = traceTidLoop;
        }
        traceLoopTime = t;
        traceLoopCpu = cpu;
        traceLoopSlept = false;
      } else if (traceLoopTime.time_since_epoch().count() != 0) [[unlikely]] {
        traceLoopTime = {};
      }
    };

    while (true) {
      traceLoop();
      auto now = std::chrono::steady_clock::now();
      if (now - prevHeartbeatTime >= heartbeatInterval) {
        prevHeartbeatTime = now;
        if (heartbeatEnabled) {
          heartbeat(now);
        }
        if (debugCpuTime) {
          checkPreemptions(true);
        }
        if (errorState && now - errorTime >= std::chrono::seconds(60)) {
          fatal("Moodist is terminating the process due to the aforementioned errors\n");
        }
      }
      if (debugCpuTime) {
        auto now = std::chrono::steady_clock::now();
        if (now - prevRunTime >= std::chrono::milliseconds(1)) {
          log.info("lost %gms\n", seconds(now - prevRunTime) * 1000);
        }
        prevRunTime = now;
      }
      if (cpuThread->queueSize.load(std::memory_order_relaxed) == 0) {
        if (activeWorks.empty()) {
          if (activeDevices.empty()) {
            {
              std::unique_lock l(cpuThread->mutex);
              if (cpuThread->queueSize.load(std::memory_order_relaxed) != 0) {
                continue;
              }
              cpuThread->busy.store(false, std::memory_order_relaxed);
            }
            if (shutdownInProgress) {
              break;
            }
            ++waitCount;
            traceLoopSlept = true;
            futexWait(&cpuThread->queueSize, 0, std::chrono::seconds(10));
            if (debugCpuTime) {
              prevRunTime = std::chrono::steady_clock::now();
            }
            continue;
          } else {
            if (shutdownInProgress) {
              break;
            }
          }
        }
      }
      for (int i = 0; i != 256; ++i) {
        traceLoop();
        if (cpuThread->queueSize.load(std::memory_order_relaxed) != 0) {
          --cpuThread->queueSize;

          std::unique_lock l(cpuThread->mutex);
          CHECK(!cpuThread->queue.empty());
          QueueEntry& queueEntry = cpuThread->queue.front();
          cpuThread->queue.pop_front();
          l.unlock();

          CHECK(queueEntry.stepValue < (1ul << 31));

          if (shutdownScheduled && queueEntry.task != taskCallback) [[unlikely]] {
            if (queueEntry.task != taskShutdown) {
              log.error("Work (%d) has been scheduled in a process group after shutdown. This work cannot complete.",
                  queueEntry.task);
              setErrorState();
            }
          } else {
            switch (queueEntry.task) {
            case taskBarrier:
              enqueue(allocatorBarrier, (QueueEntryBarrier&)queueEntry);
              break;
            case taskShutdown:
              shutdownScheduled = true;
              enqueue(allocatorShutdown, queueEntry);
              break;
            case taskAllGatherCpu:
              enqueue(allocatorAllGatherCpu, (QueueEntryAllGatherCpu&)queueEntry);
              break;
            case taskReduceScatterCpu:
              enqueue(allocatorReduceScatterCpu, (QueueEntryReduceScatterCpu&)queueEntry);
              break;
            case taskBroadcastCpu:
              enqueue(allocatorBroadcastRingCpu, (QueueEntryBroadcastCpu&)queueEntry);
              break;
            case taskInternalBarrier:
              enqueue(allocatorInternalBarrier, (QueueEntryBarrier&)queueEntry);
              break;
            case taskCreateQueue:
              enqueue(allocatorCreateQueue, (QueueEntryCreateQueue&)queueEntry);
              break;
            case taskCreateQueueNamed:
              executeCreateQueueNamed((QueueEntryCreateQueue&)queueEntry);
              break;
            case taskQueuePut:
              executeQueuePut((QueueEntryQueuePut&)queueEntry);
              break;
            case taskQueueRead:
              executeQueueRead((QueueEntryQueueRead&)queueEntry);
              break;
            case taskQueueReadFinished:
              executeQueueReadFinished((QueueEntryQueueReadFinished&)queueEntry);
              break;
            case taskQueueGet:
              executeQueueGet((QueueEntryQueueGet&)queueEntry);
              break;
            case taskQueueTransaction:
              executeQueueTransaction((QueueEntryQueueTransaction&)queueEntry);
              break;
            case taskCat:
              enqueue(allocatorCat, (QueueEntryCat&)queueEntry);
              break;
            case taskCopy:
              executeCopy((QueueEntryCopy&)queueEntry);
              break;
            case taskCached:
              enqueue(allocatorCached, (QueueEntryCached&)queueEntry);
              break;
            case taskCallback:
              executeCallback((QueueEntryCallback&)queueEntry);
              break;
            case taskCustom:
              enqueue(allocatorCustom, (QueueEntryCustom&)queueEntry);
              break;
            case taskFabricMap:
              executeFabricMap((QueueEntryFabricMap&)queueEntry);
              break;
            case taskRdmaRegister:
              executeRdmaRegister((QueueEntryRdmaRegister&)queueEntry);
              break;
            case taskRdmaRead:
              executeRdmaRead((QueueEntryRdmaRead&)queueEntry);
              break;
            default:
              throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
            }
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
  }
};

} // namespace

void RdmaCpuThreadApi::setActive() {
  CpuThreadImpl* self = (CpuThreadImpl*)impl;
  Device& dev = *(Device*)context;
  CHECK(!dev.active);
  self->activeDevices.push_back(dev);
  dev.active = true;
}

void RdmaCpuThreadApi::setInactive() {
  CpuThreadImpl* self = (CpuThreadImpl*)impl;
  Device& dev = *(Device*)context;
  CHECK(dev.active);
  self->activeDevices.erase(dev);
  dev.active = false;
}

std::string RdmaCpuThreadApi::groupName() {
  CpuThreadImpl* self = (CpuThreadImpl*)impl;
  return self->groupName();
}

std::string RdmaCpuThreadApi::rankName(size_t i) {
  CpuThreadImpl* self = (CpuThreadImpl*)impl;
  return self->rankName(i);
}

[[gnu::noinline]] [[gnu::cold]] void RdmaCpuThreadApi::setError() {
  CpuThreadImpl* self = (CpuThreadImpl*)impl;
  self->cqErrored = true;
  self->setErrorState();
}

bool RdmaCpuThreadApi::suppressErrors() {
  CpuThreadImpl* self = (CpuThreadImpl*)impl;
  return self->shutdownSuppressErrors;
}

CpuThread::CpuThread(Group* group) : group(group) {}
CpuThread::~CpuThread() {
  kill();
  if (thread.joinable()) {
    if (thread.get_id() == std::this_thread::get_id()) {
      thread.detach();
    } else {
      thread.join();
    }
  }
}

void CpuThread::kill() {
  if (thread.joinable() && !killed.exchange(true)) {
    QueueEntry* e = freelistShutdown.pop();
    e->task = taskShutdown;
    e->concurrencyIndex = 0;
    e->sd = &group->getStreamData(nullptr);
    std::unique_lock l(mutex);
    queue.push_back(*e);
    l.unlock();
    ++queueSize;
    futexWakeAll(&queueSize);
  }
}

void CpuThread::start(Function<void()> pghandle) {
  thread = std::thread(
      [this](Function<void()> pghandle) {
        // An error of this rank in the setup, which the other ranks' setup must not wait for (see SetupComms::fail).
        // If the setup had already failed (e.g. on another rank) and the error does not come from a wait of the setup's
        // ring, which reports that failure (e.g. a wait of the store, which does not check for one, ended with an error
        // of its own), the failure is logged.
        auto setupError = [this](const std::exception& e) {
          if (!dynamic_cast<const SetupCommsError*>(&e)) {
            if (const std::string reason = group->setupComms->failureReason(); !reason.empty()) {
              log.error("rank %d: the setup of the process group had failed (%s) before this error: %s\n", group->rank,
                  reason, e.what());
            }
          }
          group->setupComms->fail(fmt::sprintf("rank %d failed: %s", group->rank, e.what()));
        };
        try {
          async::setCurrentThreadName("moodist-cputhread");
          numa_run_on_node(group->allocationNode);
          numa_membind(group->allocationNode);
          CHECK_CU(cuCtxSetCurrent(group->cuContext));

          group->init2();
        } catch (const std::exception& e) {
          setupError(e);
          initExceptionPtr = std::current_exception();
          initException = true;
          return;
        }

        try {

          CpuThreadImpl impl(this);
          impl.entry();
          pghandle();

        } catch (const std::exception& e) {
          const CudaError* ce = dynamic_cast<const CudaError*>(&e);
          if (ce && ce->error == CUDA_ERROR_DEINITIALIZED) {
            log.verbose("CPU thread got a CUDA deinitialized error.\n");
            // (in the setup, the main thread waits for it)
            if (!ready) {
              initExceptionPtr = std::current_exception();
              initException = true;
            }
          } else {
            if (!ready) {
              setupError(e);
            }
            fatal("CPU thread error: %s\n", e.what());
          }
        }
      },
      std::move(pghandle));
}

} // namespace moodist
