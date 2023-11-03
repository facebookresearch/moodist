#pragma once

#include "common.h"

namespace moodist {

struct SetupComms;
struct IpcMapper;
struct CpuThread;
struct IbCommon;
struct Kernels;
struct AllGather;
struct ReduceScatter;

struct DynamicAddresses;

struct alignas(64) Progress {
  uint32_t stepValue;
  uint32_t cpuStepValue;
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

struct Group {
  size_t rank;
  size_t size;

  int deviceIndex = 0;

  CUcontext cuContext = nullptr;
  CUdevice cuDevice;

  CUstream stream = nullptr;

  std::array<CUstream, 9> extraStreams{};
  std::array<CUevent, 9> extraEvents{};

  std::unique_ptr<SetupComms> setupComms;
  std::unique_ptr<IpcMapper> ipcMapper;
  std::vector<std::unique_ptr<IbCommon>> ibDevs;
  std::unique_ptr<Kernels> kernels;
  std::unique_ptr<AllGather> allGather;
  std::unique_ptr<ReduceScatter> reduceScatter;
  std::unique_ptr<CpuThread> cpuThread;

  // AllocatedBuffer temporaryBuffer;
  // AllocatedBuffer cudaStepDoneBuffer;
  // AllocatedBuffer cpuStepDoneBuffer;

  AllocatedBuffer cpuOutBuffer;
  AllocatedBuffer cpuInBuffer;

  AllocatedBuffer cudaStepValue;
  std::array<uintptr_t, 8> peerCudaStepValue;

  AllocatedBuffer cudaPeerAddresses;
  std::array<uintptr_t, 8> peerCudaPeerAddresses;

  AllocatedBuffer cudaCommsDeviceDataSent;

  AllocatedBuffer cudaProxyReady;
  std::array<uintptr_t, 8> peerCudaProxyReady;

  AllocatedBuffer cudaCopyDone;
  std::array<uintptr_t, 8> peerCudaCopyDone;

  // AllocatedBuffer peerInBuffer;
  // std::array<uintptr_t, 8> peerOut;

  std::vector<size_t> ipcRanks;
  std::vector<size_t> peerIndices;
  std::vector<uint8_t> ipcAccess;
  std::vector<std::vector<uint8_t>> peerIpcAccess;
  std::array<size_t, 8> peerMyRemoteIndex;

  std::vector<std::vector<size_t>> nodeRanks;

  AllocatedBuffer sendBuffer;
  AllocatedBuffer recvBuffer;
  AllocatedBuffer alignedBuffer;

  int allocationNode = -1;

  // AllocatedBuffer commsBuffer;
  void* mySharedMem = nullptr;
  size_t mySharedMemSize = 0;
  std::array<void*, 8> peerSharedMem;
  DynamicAddresses* localDyns = nullptr;
  Progress* localProgress = nullptr;

  std::atomic_uint32_t* myStepCounter = nullptr;

  using AddressPairs = std::array<AddressPair, 8>;
  using PeerStepValues = std::array<std::atomic_uint32_t, 8>;
  AddressPairs* peerAddrs = nullptr;
  PeerStepValues* peerCopyDone = nullptr;

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

  static constexpr size_t dataChunks = 1;

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
};

} // namespace moodist
