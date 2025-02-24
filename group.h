#pragma once

#include "common.h"
#include "hash_map.h"
#include "logging.h"
#include "simple_vector.h"
#include "vector.h"
#include <ATen/core/ATen_fwd.h>
#include <torch/types.h>

namespace moodist {

struct SetupComms;
struct IpcMapper;
struct CpuThread;
struct IbCommon;
struct Kernels;
struct AllGather;
struct ReduceScatter;

struct DynamicAddresses;

enum class Dtype { float32, float64, int32, int64, bfloat16, count };
enum class Reduction { sum, min, max, avg, count };

static constexpr size_t maxConcurrency = 16;
static constexpr size_t maxDevices = 4;
static constexpr size_t maxChunks = 8;

struct alignas(64) Progress {
  uint32_t stepValue;
  uint32_t cpuStepValue;
};

struct alignas(64) CpuAddresses {
  uint32_t stepValue;
  int pid;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  size_t bytes;
};

struct DynamicAddressesNoKeys {
  uint32_t stepValue;
  uint32_t opType;
  uintptr_t gatherAddress;
  size_t gatherBytes;
};

struct DynamicAddressesBase : DynamicAddressesNoKeys {
  std::array<uint32_t, maxDevices> gatherKey;
};

struct alignas(128) DynamicAddresses : DynamicAddressesBase {};
static_assert(sizeof(DynamicAddresses) <= 128);

struct alignas(64) CpuDynamicAddresses {
  uint32_t stepValue;
  uint32_t opType;
  uintptr_t gatherAddress;
  size_t gatherBytes;
};

struct AddressPair {
  uintptr_t inputAddress;
  size_t inputBytes;
  uintptr_t outputAddress;
  size_t outputBytes;
};

struct StreamData {
  // AllocatedBuffer sendBuffer;
  // AllocatedBuffer recvBuffer;
  // AllocatedBuffer alignedBuffer;
  // AllocatedBuffer alignedBuffer2;
  size_t cpuThreadIndex = -1;
  StreamData();
  ~StreamData();
};

struct SyncData {
  std::atomic_bool shouldSync = false;
  std::atomic_uint32_t syncStepValue = 0;
  std::atomic_uint32_t myStepValue = 0;
  std::atomic_uint32_t syncStepValue2 = 0;
};

struct TreeSendsRecvs {
  Vector<size_t> sends;
  Vector<size_t> recvs;
};

struct Group {
  size_t rank;
  size_t size;

  int deviceIndex = 0;
  int pid = 0;

  CUcontext cuContext = nullptr;
  CUdevice cuDevice;

  static constexpr size_t maxConcurrency = moodist::maxConcurrency;
  static constexpr size_t maxDevices = moodist::maxDevices;
  static constexpr size_t maxChunks = moodist::maxChunks;

  std::unique_ptr<SetupComms> setupComms;
  std::unique_ptr<IpcMapper> ipcMapper;
  std::vector<std::unique_ptr<IbCommon>> ibDevs;
  size_t numTrueIbDevs = 0;
  std::unique_ptr<Kernels> kernels;
  std::unique_ptr<AllGather> allGather;
  std::unique_ptr<ReduceScatter> reduceScatter;
  CpuThread* cpuThread = nullptr;

  AllocatedArray cpuOutBuffer;
  AllocatedArray cpuInBuffer;

  AllocatedArray cudaStepValue;
  std::array<PeerArrayRef, 8> peerCudaStepValue;

  AllocatedArray cudaPeerAddresses;
  std::array<PeerArrayRef, 8> peerCudaPeerAddresses;

  AllocatedArray cudaCommsDeviceDataSent;
  AllocatedArray cpuCommsDeviceDataSent;

  AllocatedArray cudaCommsDeviceDataSent2;
  AllocatedArray cpuCommsDeviceDataSent2;

  AllocatedArray cudaProxyReady;
  std::array<PeerArrayRef, 8> peerCudaProxyReady;

  AllocatedArray cudaCopyDone;
  std::array<PeerArrayRef, 8> peerCudaCopyDone;

  std::vector<AllocatedArray*> buffersToReset;

  std::vector<size_t> ipcRanks;
  std::vector<size_t> peerIndices;
  std::vector<uint8_t> ipcAccess;
  std::vector<std::vector<uint8_t>> peerIpcAccess;
  std::array<size_t, 8> peerMyRemoteIndex;

  Vector<Vector<size_t>> nodeRanks;
  Vector<size_t> rankToNodeIndex;
  Vector<size_t> rankLocalRank;

  int allocationNode = -1;

  void* mySharedMem = nullptr;
  size_t mySharedMemSize = 0;
  std::array<void*, 8> peerSharedMem;
  DynamicAddresses* localDyns = nullptr;
  CpuDynamicAddresses* cpuLocalDyns = nullptr;
  Progress* localProgress = nullptr;
  CpuAddresses* cpuAddresses = nullptr;
  SyncData* syncData = nullptr;
  uint32_t* cpuProxyReady = nullptr;

  Progress* localProgress2 = nullptr;

  uint32_t* cpuStepValues = nullptr;
  uint32_t* cpuStepValuesDeviceChunks = nullptr;

  std::atomic_uint32_t* atomicStepValue = nullptr;

  AllocatedArray cudaStepValuesBuffer;
  AllocatedArray cudaStepValuesDeviceChunksBuffer;

  HashMap<CUstream, std::unique_ptr<StreamData>> streamData;
  std::array<std::unique_ptr<StreamData>, maxConcurrency> cpuStreamData;

  template<typename T>
  size_t getSharedOffset(T* myVar) const {
    size_t offset = (uintptr_t)(void*)myVar - (uintptr_t)mySharedMem;
    CHECK(offset + sizeof(T) <= mySharedMemSize);
    return offset;
  }

  template<typename T>
  T* getPeerVar(size_t peerIndex, T* myVar) const {
    return (T*)((uintptr_t)peerSharedMem[peerIndex] + getSharedOffset(myVar));
  }

  Group(size_t rank, size_t size);
  ~Group();

  void init();

  size_t getPeerIndex(size_t rank) {
    for (size_t i = 0; i != ipcRanks.size(); ++i) {
      if (ipcRanks[i] == rank) {
        return i;
      }
    }
    TORCH_CHECK(false, "getPeerIndex failed");
    return (size_t)-1;
  }

  AllocatedBuffer allocateManaged(size_t bytes);
  AllocatedBuffer allocateDevice(size_t bytes);
  AllocatedBuffer allocateHostMapped(size_t bytes);
  AllocatedBuffer allocateHost(size_t bytes);
  AllocatedBuffer allocateWriteCombined(size_t bytes);
  AllocatedBuffer allocateDeviceMapped(size_t bytes);

  AllocatedCpuBuffer allocateCpu(size_t bytes);

  AllocatedArray allocateArrayHost(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayHostMapped(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayDevice(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayManaged(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayWriteCombined(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayDeviceMapped(size_t itembytes, size_t numel);

  void createStreamData(std::unique_ptr<StreamData>& ptr);
  StreamData& getStreamData(CUstream stream) {
    auto& ptr = streamData[stream];
    if (!ptr) {
      createStreamData(ptr);
    }
    return *ptr;
  }

  StreamData& getCpuStreamData(uint32_t concurrencyIndex) {
    auto& ptr = cpuStreamData[concurrencyIndex];
    if (!ptr) {
      createStreamData(ptr);
    }
    return *ptr;
  }

  std::atomic_size_t bytesManaged = 0;
  std::atomic_size_t bytesDevice = 0;
  std::atomic_size_t bytesHost = 0;

  void reportBytes() {
    if (currentLogLevel < LOG_VERBOSE) {
      return;
    }
    size_t m = bytesManaged;
    size_t d = bytesDevice;
    size_t h = bytesHost;
    double div = 1024 * 1042 * 1024;
    log.verbose(
        "Bytes managed: %d (%gG)\nBytes device: %d (%gG)\nBytes host: %d (%gG)\n", m, m / div, d, d / div, h, h / div);
  }

  TreeSendsRecvs generateTree(size_t nary, size_t rootRank);
  HashMap<size_t, std::unique_ptr<TreeSendsRecvs>> cachedTrees;
  TreeSendsRecvs* getTree(size_t nary, size_t rootRank) {
    size_t index = size * nary + rootRank;
    auto it = cachedTrees.find(index);
    if (it != cachedTrees.end()) {
      return &*it->second;
    }
    auto p = std::make_unique<TreeSendsRecvs>(generateTree(nary, rootRank));
    auto* r = &*p;
    cachedTrees[index] = std::move(p);
    return r;
  }

  std::mutex cudaUint32Mutex;
  std::vector<AllocatedBuffer> cudaUint32List;
  size_t cudaUint32Offset = 0;
  uintptr_t getNextCudaUint32();
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

} // namespace moodist
