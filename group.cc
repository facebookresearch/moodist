#include "group.h"
#include "clock.h"
#include "common.h"
#include "cputhread.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include <algorithm>
#include <cstring>

namespace moodist {

static std::string readBootId() {
  char buf[64];
  std::memset(buf, 0, sizeof(buf));
  FILE* f = ::fopen("/proc/sys/kernel/random/boot_id", "rb");
  if (f) {
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    while (n && buf[n - 1] == '\n') {
      --n;
    }
    buf[n] = 0;
    fclose(f);
  }
  return buf;
}

Group::Group(size_t rank, size_t size) : rank(rank), size(size) {
  setupComms = createSetupComms(this);
  ipcMapper = createIpcMapper(this);
  cputhread = std::make_unique<CpuThread>(this);
}

Group::~Group() {}

void Group::init() {

  auto start = Clock::now();

  deviceIndex = c10::cuda::current_device();

  CHECK_CU(cuInit(0));

  CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
  CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
  CHECK_CU(cuCtxSetCurrent(cuContext));

  int clockRate;
  CHECK_CU(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice));
  fmt::printf("device clock rate: %dkhz\n", clockRate);

  int asyncEngines;
  CHECK_CU(cuDeviceGetAttribute(&asyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cuDevice));
  fmt::printf("device async engines: %d\n", asyncEngines);

  CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // // temporaryBuffer = allocateDevice(temporaryBytes);
  // cudaStepDoneBuffer = allocateDevice(size * 8);
  // cpuStepDoneBuffer = allocateHost(size * 8);

  // // cpuOutBuffer = allocateHostMapped(4096 + 16 * size);
  // cpuOutBuffer = allocateManaged(4096 + 16 * size);
  // CHECK_CU(cuMemAdvise(cpuOutBuffer.cudaPointer, cpuOutBuffer.bytes, CU_MEM_ADVISE_SET_READ_MOSTLY, cuDevice));
  // cpuInBuffer = allocateHost(4096 + 16 * size);

  CHECK_NVML(nvmlInit_v2());

  std::array<char, 0x100> cudaPciBus;
  CHECK_CU(cuDeviceGetPCIBusId(cudaPciBus.data(), cudaPciBus.size(), cuDevice));
  cudaPciBus[0xff] = 0;
  std::string cudaPath = fmt::sprintf("/sys/bus/pci/devices/%s", cudaPciBus.data());
  for (auto& v : cudaPath) {
    if (v >= 'A' && v <= 'Z') {
      v += 0x20;
    }
  }
  char* path = realpath(cudaPath.c_str(), nullptr);
  if (path) {
    cudaPath = path;
    free(path);
  }
  cudaPath = removePciPathPrefix(cudaPath);
  // fmt::printf("cuda path is %s\n", cudaPath);

  nvmlDevice_t nvmlDevice;
  CHECK_NVML(nvmlDeviceGetHandleByPciBusId_v2(cudaPciBus.data(), &nvmlDevice));

  std::string bootId = readBootId();

  std::vector<std::string> allRanksBootIds = setupComms->allgather(bootId);

  std::vector<size_t> localRanksByBootId;
  for (size_t i = 0; i != size; ++i) {
    if (allRanksBootIds[i] == bootId) {
      localRanksByBootId.push_back(i);
    }
  }
  TORCH_CHECK(std::find(localRanksByBootId.begin(), localRanksByBootId.end(), rank) != localRanksByBootId.end());

  fmt::printf("%d: local ranks: [%s]\n", rank, fmt::to_string(fmt::join(localRanksByBootId, ", ")));

  size_t localRankIndex = ~(size_t)0;
  for (size_t i = 0; i != localRanksByBootId.size(); ++i) {
    if (localRanksByBootId[i] == rank) {
      localRankIndex = i;
      break;
    }
  }
  TORCH_CHECK(localRankIndex >= 0 && localRankIndex < localRanksByBootId.size());

  std::unordered_map<size_t, int> localNvlink;

  std::vector<std::string> allRanksCudaPciBus = setupComms->allgather(std::string(cudaPciBus.data()));
  for (size_t i : localRanksByBootId) {
    if (i == rank) {
      continue;
    }
    nvmlDevice_t peerDevice;
    if (nvmlDeviceGetHandleByPciBusId_v2(allRanksCudaPciBus.at(i).c_str(), &peerDevice) == NVML_SUCCESS) {
      nvmlGpuP2PStatus_t status = NVML_P2P_STATUS_UNKNOWN;
      CHECK_NVML(nvmlDeviceGetP2PStatus(nvmlDevice, peerDevice, NVML_P2P_CAPS_INDEX_NVLINK, &status));
      if (status == NVML_P2P_STATUS_OK) {
        fmt::printf("rank %d: connected to local rank %d through nvlink\n", rank, i);
        localNvlink[i] = 1;
      } else {
        fmt::printf("rank %d: connected to local rank %d through NOT nvlink\n", rank, i);
      }
    } else {
      fmt::printf("rank %d: nvml failed to access cuda device for local rank %d\n", rank, i);
    }
  }

  std::vector<std::unordered_map<size_t, int>> allRanksNvlink = setupComms->allgather(localNvlink);

  ipcRanks.clear();
  peerIndices.clear();

  for (size_t i : localRanksByBootId) {
    if (i != rank) {
      ipcRanks.push_back(i);
      peerIndices.push_back(peerIndices.size());
    }
  }

  ipcMapper->init();

  cputhread->start();

  fmt::printf("%d: init ok!\n", rank);

  // synchronize to prevent ipc init race conditions
  setupComms->allgather(0);

  fmt::printf("%d: init synchronized!\n", rank);
}

AllocatedBuffer Group::allocateManaged(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  CUdeviceptr ptr;
  CHECK_CU(cuMemAllocManaged(&ptr, bytes, CU_MEM_ATTACH_GLOBAL));
  CHECK_CU(cuMemsetD8(ptr, 0, bytes));
  r.hostAllocated = false;
  r.cudaPointer = ptr;
  r.cpuPointer = (void*)ptr;
  fmt::printf("allocated managed memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}
AllocatedBuffer Group::allocateDevice(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  CUdeviceptr ptr;
  CHECK_CU(cuMemAlloc(&ptr, bytes));
  CHECK_CU(cuMemsetD8(ptr, 0, bytes));
  r.hostAllocated = false;
  r.cudaPointer = ptr;
  r.cpuPointer = nullptr;
  fmt::printf("allocated device memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}
AllocatedBuffer Group::allocateHostMapped(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  CHECK_CU(cuMemHostAlloc(&r.cpuPointer, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
  std::memset(r.cpuPointer, 0, bytes);
  r.hostAllocated = true;
  CUdeviceptr ptr;
  CHECK_CU(cuMemHostGetDevicePointer(&ptr, r.cpuPointer, 0));
  r.cudaPointer = ptr;
  fmt::printf("allocated device mapped host memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}
AllocatedBuffer Group::allocateHost(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  CHECK_CU(cuMemHostAlloc(&r.cpuPointer, r.bytes, 0));
  std::memset(r.cpuPointer, 0, bytes);
  r.hostAllocated = true;
  r.cudaPointer = (uintptr_t)r.cpuPointer;
  fmt::printf("allocated host memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}
AllocatedBuffer Group::allocateWriteCombined(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  CHECK_CU(cuMemHostAlloc(&r.cpuPointer, bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED));
  std::memset(r.cpuPointer, 0, bytes);
  r.hostAllocated = true;
  CUdeviceptr ptr;
  CHECK_CU(cuMemHostGetDevicePointer(&ptr, r.cpuPointer, 0));
  r.cudaPointer = ptr;
  fmt::printf("allocated write-combined memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}

} // namespace moodist
