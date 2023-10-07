#pragma once

#include "common.h"

namespace moodist {

struct SetupComms;
struct IpcMapper;
struct CpuThread;

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
  std::unique_ptr<CpuThread> cputhread;

  // AllocatedBuffer temporaryBuffer;
  // AllocatedBuffer cudaStepDoneBuffer;
  // AllocatedBuffer cpuStepDoneBuffer;

  // AllocatedBuffer cpuOutBuffer;
  // AllocatedBuffer cpuInBuffer;

  // AllocatedBuffer peerInBuffer;
  // std::array<uintptr_t, 8> peerOut;

  std::vector<size_t> ipcRanks;
  std::vector<size_t> peerIndices;
  // std::vector<uint8_t> ipcAccess;
  // std::vector<std::vector<uint8_t>> peerIpcAccess;

  // std::vector<uintptr_t> rankCudaStepDoneBase;
  // std::array<size_t, 8> peerMyRemoteIndex;

  int allocationNode = -1;

  static constexpr size_t dataChunks = 4;

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
