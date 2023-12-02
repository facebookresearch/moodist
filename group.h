#pragma once

#include "common.h"
#include "hash_map.h"

namespace moodist {

struct SetupComms;
struct IpcMapper;
struct CpuThread;
struct IbCommon;
struct Kernels;
struct AllGather;
struct ReduceScatter;

struct DynamicAddresses;

enum class Dtype { float32, float64, int32, int64, count };
enum class Reduction { sum, min, max, avg, count };

static constexpr size_t maxConcurrency = 16;

struct alignas(64) Progress {
  uint32_t stepValue;
  uint32_t cpuStepValue;
};

struct alignas(64) CpuAddresses {
  uint32_t stepValue;
  int pid;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
};

struct alignas(64) DynamicAddresses {
  std::array<uint32_t, 32> gatherKey;
  uintptr_t gatherAddress;
};

struct AddressPair {
  uintptr_t inputAddress;
  size_t inputBytes;
  uintptr_t outputAddress;
  size_t outputBytes;
};

struct StreamData {
  AllocatedBuffer sendBuffer;
  AllocatedBuffer recvBuffer;
  AllocatedBuffer alignedBuffer;
  AllocatedBuffer alignedBuffer2;
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

struct Group {
  size_t rank;
  size_t size;

  int deviceIndex = 0;
  int pid = 0;

  CUcontext cuContext = nullptr;
  CUdevice cuDevice;

  HashMap<CUstream, std::unique_ptr<StreamData>> streamData;
  static constexpr size_t maxConcurrency = 16;

  std::unique_ptr<SetupComms> setupComms;
  std::unique_ptr<IpcMapper> ipcMapper;
  std::vector<std::unique_ptr<IbCommon>> ibDevs;
  std::unique_ptr<Kernels> kernels;
  std::unique_ptr<AllGather> allGather;
  std::unique_ptr<ReduceScatter> reduceScatter;
  std::unique_ptr<CpuThread> cpuThread;

  AllocatedArray cpuOutBuffer;
  AllocatedArray cpuInBuffer;

  AllocatedArray cudaStepValue;
  std::array<PeerArrayRef, 8> peerCudaStepValue;

  AllocatedArray cudaPeerAddresses;
  std::array<PeerArrayRef, 8> peerCudaPeerAddresses;

  AllocatedArray cudaCommsDeviceDataSent;
  AllocatedArray cpuCommsDeviceDataSent;

  AllocatedArray cudaProxyReady;
  std::array<PeerArrayRef, 8> peerCudaProxyReady;

  AllocatedArray cudaCopyDone;
  std::array<PeerArrayRef, 8> peerCudaCopyDone;

  std::vector<size_t> ipcRanks;
  std::vector<size_t> peerIndices;
  std::vector<uint8_t> ipcAccess;
  std::vector<std::vector<uint8_t>> peerIpcAccess;
  std::array<size_t, 8> peerMyRemoteIndex;

  std::vector<std::vector<size_t>> nodeRanks;

  int allocationNode = -1;

  void* mySharedMem = nullptr;
  size_t mySharedMemSize = 0;
  std::array<void*, 8> peerSharedMem;
  DynamicAddresses* localDyns = nullptr;
  Progress* localProgress = nullptr;
  CpuAddresses* cpuAddresses = nullptr;
  SyncData* syncData = nullptr;
  uint32_t* cpuProxyReady = nullptr;

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

  AllocatedCpuBuffer allocateCpu(size_t bytes);

  AllocatedArray allocateArrayHost(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayHostMapped(size_t itembytes, size_t numel);
  AllocatedArray allocateArrayDevice(size_t itembytes, size_t numel);

  void createStreamData(std::unique_ptr<StreamData>& ptr);
  StreamData& getStreamData(CUstream stream) {
    auto& ptr = streamData[stream];
    if (!ptr) {
      createStreamData(ptr);
    }
    return *ptr;
  }
};

} // namespace moodist
