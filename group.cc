#include "group.h"
#include "allgather.h"
#include "clock.h"
#include "common.h"
#include "cputhread.h"
#include "ib_common.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "reduce_scatter.h"
#include "setup_comms.h"

#include <algorithm>
#include <cstring>

#include <numa.h>
#include <type_traits>

namespace moodist {

struct LocalDevice {
  IbvPtr<ibv_context, ibv_close_device> ctx;
  int portNum = 0;
  ibv_port_attr portAttributes;
  std::string ibPath;
};

struct LocalDeviceNode {
  std::string ibPath;
  std::tuple<int, int, int> score;
  template<typename X>
  void serialize(X& x) {
    x(ibPath, std::get<0>(score), std::get<1>(score), std::get<2>(score));
  }
};

struct RankForIbSetup {
  std::string bootId;
  std::vector<LocalDeviceNode> ibDevices;
  template<typename X>
  void serialize(X& x) {
    x(bootId, ibDevices);
  }
};

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
  setupComms = createSetupComms(rank, size);
  ipcMapper = createIpcMapper(this);
  cpuThread = std::make_unique<CpuThread>(this);
  allGather = std::make_unique<AllGather>(this);
  reduceScatter = std::make_unique<ReduceScatter>(this);
}

Group::~Group() {}

void Group::init() {

  auto start = Clock::now();

  deviceIndex = c10::cuda::current_device();

  cuContext = nullptr;
  cuCtxGetCurrent(&cuContext);
  if (!cuContext) {
    CHECK_CU(cuInit(0));

    CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
    CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
    CHECK_CU(cuCtxSetCurrent(cuContext));

    cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, 0x10000);

  } else {
    CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
  }

  int clockRate;
  CHECK_CU(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice));
  log.verbose("device clock rate: %dkhz\n", clockRate);

  int asyncEngines;
  CHECK_CU(cuDeviceGetAttribute(&asyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cuDevice));
  log.verbose("device async engines: %d\n", asyncEngines);

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

  CHECK(numa_available() >= 0);

  unsigned long nodeSet = 0;
  CHECK_NVML(nvmlDeviceGetMemoryAffinity(nvmlDevice, 1, &nodeSet, NVML_AFFINITY_SCOPE_NODE));

  for (unsigned int i = 0; i != 64; ++i) {
    if (nodeSet & (1ull << i)) {
      allocationNode = i;
      break;
    }
  }

  log.debug("%d: allocationNode is %d\n", rank, allocationNode);

  std::vector<LocalDevice> localDevices;

  std::vector<LocalDeviceNode> localDeviceNodes;

  std::string bootId = readBootId();

  ibv_fork_init();

  ibv_device** list = ibv_get_device_list(nullptr);
  for (ibv_device** i = list; *i; ++i) {
    ibv_device* di = *i;

    // fmt::printf("device %s\n", di->name);
    IbvPtr<ibv_context, ibv_close_device> ctx = ibv_open_device(di);
    if (ctx) {
      ibv_device_attr attributes;
      std::memset(&attributes, 0, sizeof(attributes));
      if (ibv_query_device(ctx, &attributes) != 0) {
        continue;
      }

      log.debug("max_mr_size is %d\n", attributes.max_mr_size);
      log.debug("max_mr is %d\n", attributes.max_mr);

      int portNum = -1;
      ibv_port_attr portAttributes;
      int bestSpeed = -1;
      for (size_t i = 0; i != attributes.phys_port_cnt; ++i) {
        ibv_port_attr attributes;
        std::memset(&attributes, 0, sizeof(attributes));
        if (ibv_query_port(ctx, 1 + i, &attributes) == 0) {
          if (attributes.state != IBV_PORT_ACTIVE) {
            continue;
          }
          log.debug("got a port with link layer %d\n", attributes.link_layer);
          log.debug("port mtu %d, speed %d\n", attributes.active_mtu, attributes.active_speed);
          int speed = attributes.active_speed;
          if (attributes.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            speed += 0x100000;
          }
          if (speed > bestSpeed) {
            bestSpeed = speed;
            portNum = 1 + i;
            portAttributes = attributes;
          }
        }
        // fmt::printf("queried port %d\n", 1 + i);
      }
      if (portNum == -1) {
        continue;
      }

      std::string ibPath = fmt::sprintf("%s/device", di->ibdev_path);
      char* path = realpath(ibPath.c_str(), nullptr);
      if (path) {
        ibPath = path;
        free(path);
      }
      ibPath = removePciPathPrefix(ibPath);
      // fmt::printf("ib path is %s\n", ibPath);

      int nmatch = 0;
      for (size_t i = 0; i != std::min(cudaPath.size(), ibPath.size()); ++i) {
        if (cudaPath[i] == ibPath[i]) {
          if (cudaPath[i] == '/') {
            ++nmatch;
          }
        } else {
          break;
        }
      }

      log.debug("rank %d: %s -> %s has score %d %d\n", rank, cudaPath, ibPath, nmatch, bestSpeed);

      for (auto& v : localDevices) {
        CHECK(v.ibPath != ibPath);
      }

      LocalDevice lc;
      lc.ctx = std::move(ctx);
      lc.portNum = portNum;
      lc.ibPath = ibPath;
      lc.portAttributes = portAttributes;
      localDevices.push_back(std::move(lc));

      for (auto& v : localDeviceNodes) {
        CHECK(v.ibPath != ibPath);
      }

      std::tuple<int, int, int> score = {nmatch, 0, bestSpeed};
      LocalDeviceNode n;
      n.ibPath = ibPath;
      n.score = score;
      localDeviceNodes.push_back(n);
    }
  }
  ibv_free_device_list(list);

  RankForIbSetup localRankForIbSetup;
  localRankForIbSetup.bootId = bootId;
  localRankForIbSetup.ibDevices = localDeviceNodes;

  std::vector<RankForIbSetup> allRanksDeviceNodes = setupComms->allgather(localRankForIbSetup);

  for (size_t i = 0; i != size; ++i) {
    if (allRanksDeviceNodes[i].ibDevices.empty()) {
      throw std::runtime_error(fmt::sprintf("No usable InfiniBand device found for rank %d", i));
    }
  }

  std::vector<size_t> localRanksByBootId;
  for (size_t i = 0; i != size; ++i) {
    if (allRanksDeviceNodes[i].bootId == bootId) {
      localRanksByBootId.push_back(i);
    }
  }
  TORCH_CHECK(std::find(localRanksByBootId.begin(), localRanksByBootId.end(), rank) != localRanksByBootId.end());

  log.debug("%d: local ranks: [%s]\n", rank, fmt::to_string(fmt::join(localRanksByBootId, ", ")));

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
    if (allRanksDeviceNodes[i].bootId == bootId && allRanksCudaPciBus.at(i) == std::string(cudaPciBus.data())) {
      throw std::runtime_error(
          fmt::sprintf("Rank %d and %d share the same CUDA device! (%s)", rank, i, std::string(cudaPciBus.data())));
    }
    nvmlDevice_t peerDevice;
    if (nvmlDeviceGetHandleByPciBusId_v2(allRanksCudaPciBus.at(i).c_str(), &peerDevice) == NVML_SUCCESS) {
      nvmlGpuP2PStatus_t status = NVML_P2P_STATUS_UNKNOWN;
      CHECK_NVML(nvmlDeviceGetP2PStatus(nvmlDevice, peerDevice, NVML_P2P_CAPS_INDEX_NVLINK, &status));
      if (status == NVML_P2P_STATUS_OK) {
        log.debug("rank %d: connected to local rank %d through nvlink\n", rank, i);
        localNvlink[i] = 1;
      } else {
        log.debug("rank %d: connected to local rank %d through NOT nvlink\n", rank, i);
      }
    } else {
      log.error("rank %d: nvml failed to access cuda device for local rank %d\n", rank, i);
    }
  }

  std::vector<std::unordered_map<size_t, int>> allRanksNvlink = setupComms->allgather(localNvlink);

  std::vector<std::pair<LocalDeviceNode*, LocalDevice*>> useDevices;

  {
    HashMap<std::string, bool> taken;
    while (true) {
      std::vector<LocalDeviceNode*> current;
      HashMap<std::string_view, int> useCount;
      current.resize(localRanksByBootId.size());
      std::vector<LocalDeviceNode*> best;
      std::tuple<int, int, int> bestScore = {0, 0, 0};
      int solutions = 0;
      std::function<void(size_t)> visit = [&](size_t index) {
        if (index == localRanksByBootId.size()) {
          std::tuple<int, int, int> score = {0, 0, 0};
          for (LocalDeviceNode* n : current) {
            auto s = n->score;
            if (useCount[n->ibPath] > 1) {
              std::get<2>(s) -= useCount[n->ibPath];
              if (std::get<2>(s) > 0) {
                std::get<2>(s) /= useCount[n->ibPath];
              }
            }
            std::get<0>(score) += std::get<0>(s);
            std::get<1>(score) += std::get<1>(s);
            std::get<2>(score) += std::get<2>(s);
            if (best.empty() || score > bestScore) {
              bestScore = score;
              best = current;
            }
          }
          ++solutions;
          return;
        }
        size_t rank = localRanksByBootId.at(index);
        std::vector<std::pair<std::tuple<int, int, int>, LocalDeviceNode*>> sorted;
        for (LocalDeviceNode& v : allRanksDeviceNodes.at(rank).ibDevices) {
          if (std::get<0>(v.score) && !taken[v.ibPath]) {
            sorted.emplace_back(v.score, &v);
          }
        }
        if (sorted.empty()) {
          return;
        }
        std::sort(sorted.begin(), sorted.end());
        std::reverse(sorted.begin(), sorted.end());
        if (sorted.size() > 4) {
          sorted.resize(4);
        }
        for (auto& v : sorted) {
          ++useCount[v.second->ibPath];
          current[index] = v.second;
          visit(index + 1);
          --useCount[v.second->ibPath];
        }
      };
      visit(0);

      if (best.empty()) {
        if (!useDevices.empty()) {
          break;
        }
        throw std::runtime_error(fmt::sprintf(
            "No usable InfiniBand device found for rank %d (refusing to use devices routed through CPU)", rank));
      }

      for (auto* v : best) {
        taken[v->ibPath] = true;
      }

      log.debug("evaluated %d solutions\n", solutions);
      log.debug("best score %d %d %d\n", std::get<0>(bestScore), std::get<1>(bestScore), std::get<2>(bestScore));

      for (auto& v : localDeviceNodes) {
        if (v.ibPath == best[localRankIndex]->ibPath) {
          for (auto& v2 : localDevices) {
            if (v2.ibPath == v.ibPath) {
              useDevices.emplace_back(&v, &v2);
            }
          }
        }
      }
    }
  }

  auto peersDeviceCounts = setupComms->allgather(useDevices.size());
  for (size_t i = 0; i != size; ++i) {
    if (peersDeviceCounts[i] != useDevices.size()) {
      throw std::runtime_error(fmt::sprintf(
          "Topology mismatch: rank %d has %d ib devices, but rank %d has %d ib devices", rank, useDevices.size(), i,
          peersDeviceCounts[i]));
    }
  }

  for (auto& [dn, d] : useDevices) {

    auto ibCommon = std::make_unique<IbCommon>(this);

    IbvPtr<ibv_context, ibv_close_device>& bestCtx = ibCommon->context;
    int bestPortNum = 0;
    ibv_port_attr bestPortAttributes;

    bestCtx = std::move(d->ctx);
    bestPortNum = d->portNum;
    bestPortAttributes = d->portAttributes;
    TORCH_CHECK(bestCtx != nullptr);

    log.verbose("rank %d using device %s (%s)\n", rank, bestCtx->device->name, bestCtx->device->ibdev_path);
    std::string devId = bestCtx->device->ibdev_path;

    ibCommon->init(bestPortNum, bestPortAttributes);

    ibDevs.push_back(std::move(ibCommon));
  }

  CHECK(!ibDevs.empty());

  std::vector<std::string> allRanksBootIds = setupComms->allgather(bootId);

  HashMap<std::string, size_t> nodeMap;
  for (size_t i = 0; i != size; ++i) {
    std::string& id = allRanksBootIds[i];
    auto it = nodeMap.find(id);
    if (it == nodeMap.end()) {
      nodeMap[id] = nodeRanks.size();
      nodeRanks.emplace_back();
    }
    nodeRanks.at(nodeMap[id]).push_back(i);
  }

  AllocatedBuffer testMemory = allocateDevice(0x1000);

  CUipcMemHandle testIpcHandle;
  CHECK_CU(cuIpcGetMemHandle(&testIpcHandle, testMemory.cudaPointer));

  auto allTestIpcHandles = setupComms->allgather(testIpcHandle);

  ipcAccess.resize(size);

  for (size_t i = 0; i != size; ++i) {
    if (allRanksBootIds[i] == bootId && i != rank) {
      CUdeviceptr mapPtr = 0;
      bool success =
          cuIpcOpenMemHandle(&mapPtr, allTestIpcHandles[i], CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS) == CUDA_SUCCESS;
      log.debug("ipc success? %d address %#x\n", success, mapPtr);
      if (success) {
        cuIpcCloseMemHandle(mapPtr);
        ipcAccess[i] = true;
        ipcRanks.push_back(i);
        peerIndices.push_back(peerIndices.size());
      }
    }
  }

  CHECK(peerIndices.size() <= 8);

  if (rank == 0) {
    log.debug("initial connection setup took %g ms\n", seconds(Clock::now() - start) * 1000);
    start = Clock::now();
  }

  std::sort(ipcRanks.begin(), ipcRanks.end());
  std::stable_partition(ipcRanks.begin(), ipcRanks.end(), [&](size_t i) { return i > rank; });

  std::vector<std::vector<size_t>> peerIpcRanks = setupComms->allgather(ipcRanks);

  {

    std::vector<size_t> newIpcRanks;
    for (size_t i : ipcRanks) {
      bool keep = true;
      if (localNvlink[i] == 0) {
        for (size_t n : peerIpcRanks[i]) {
          if (allRanksNvlink[n][i]) {
            log.debug(
                "rank %d: dropping connection to %d as there is no nvlink, but there is an nvlink through rank %d\n",
                rank, i, n);
            keep = false;
            break;
          }
        }
      }
      if (keep) {
        newIpcRanks.push_back(i);
      }
    }
    ipcRanks = newIpcRanks;
    peerIpcRanks = setupComms->allgather(ipcRanks);

    log.debug("rank %d ipc ranks is %s\n", rank, fmt::to_string(fmt::join(ipcRanks, ", ")));
  }

  // rankIbDevNames = setupComms->allgather(std::string(bestCtx->device->name));
  // TORCH_CHECK(rankIbDevNames.size() == size);
  // TORCH_CHECK(rankIbDevNames.at(rank) == bestCtx->device->name);
  // if (rank == 0) {
  //   for (size_t i = 0; i != size; ++i) {
  //     fmt::printf("rank %d is using device %s\n", i, rankIbDevNames[i]);
  //   }
  // }

  if (rank == 0) {
    log.debug("ipc ranks sync took %g ms\n", seconds(Clock::now() - start) * 1000);
    start = Clock::now();
  }

  peerIpcAccess.resize(size);

  for (size_t i = 0; i != size; ++i) {
    peerIpcAccess[i].resize(size);
    for (size_t n : peerIpcRanks[i]) {
      CHECK(std::find(peerIpcRanks[n].begin(), peerIpcRanks[n].end(), i) != peerIpcRanks[n].end());
      peerIpcAccess[i][n] = true;
    }
  }

  if (rank == 0) {
    log.debug("final connection setup and sync took %g ms\n", seconds(Clock::now() - start) * 1000);
    start = Clock::now();
  }

  for (auto& v : peerMyRemoteIndex) {
    v = (uint32_t)-1;
  }

  for (size_t i : ipcRanks) {
    size_t peerIndex = getPeerIndex(i);
    setupComms->sendTo(i, peerIndex);
  }
  for (size_t i : ipcRanks) {
    auto remoteIndex = setupComms->recvFrom<size_t>(i);
    peerMyRemoteIndex[getPeerIndex(i)] = remoteIndex;
  }

  {
    // we want shared memory (allocated with memfd_create) to be allocated on allocationNode,
    // but we don't want to change the current numa allocation policy.
    // thus we create a temporary thread to do it.
    std::thread tmp([&] {
      if (allocationNode != -1) {
        auto* bitmask = numa_bitmask_alloc(allocationNode + 1);
        if (bitmask) {
          numa_bitmask_clearall(bitmask);
          numa_bitmask_setbit(bitmask, allocationNode);
          numa_bind(bitmask);
          numa_bitmask_free(bitmask);
        }
      }
      ipcMapper->init();
    });
    tmp.join();
  }

  mySharedMem = ipcMapper->getMySharedMem(0, 0);
  peerSharedMem.fill(nullptr);
  for (size_t i : peerIndices) {
    peerSharedMem[i] = ipcMapper->getPeerSharedMem(i, 0, 0);
  }

  size_t offset = 0;
  uintptr_t addr = (uintptr_t)mySharedMem;
  CHECK(addr % 64 == 0);
  auto get = [&](auto*& ptr, size_t n) {
    using T = std::remove_pointer_t<std::remove_reference_t<decltype(ptr)>>;
    if (offset % alignof(T) != 0) {
      offset += alignof(T) - offset % alignof(T);
    }
    CHECK((addr + offset) % alignof(T) == 0);
    ptr = (T*)(addr + offset);
    offset += sizeof(T) * n;
  };
  get(localDyns, size);
  get(localProgress, size);
  get(myStepCounter, 1);
  get(peerAddrs, 1);
  get(peerCopyDone, 1);

  mySharedMemSize = offset;

  ipcMapper->getMySharedMem(0, offset);

  cpuOutBuffer = allocateHostMapped(4096 + 16 * size);
  cpuInBuffer = allocateHostMapped(4096 + 16 * size);

  auto mapPeerAddrs = [&](AllocatedBuffer& localBuffer, std::array<uintptr_t, 8>& peerPtrs) {
    peerPtrs.fill(0);

    std::array<uintptr_t, 8> peerMapping;

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, localBuffer.cudaPointer, localBuffer.bytes, [&, i](uintptr_t address) { peerMapping[i] = address; });
    }
    ipcMapper->wait();
    for (size_t i : peerIndices) {
      setupComms->sendTo(ipcRanks[i], std::tuple(rank, i, peerMapping[i]));
    }
    for (size_t i : peerIndices) {
      auto [sourceRank, sourcePeerIndex, address] =
          setupComms->recvFrom<std::tuple<size_t, size_t, uintptr_t>>(ipcRanks[i]);
      CHECK(sourceRank == ipcRanks[i]);
      CHECK(sourcePeerIndex == peerMyRemoteIndex[i]);
      peerPtrs[i] = address;
    }
    setupComms->allgather(0);
  };

  cudaStepValue = allocateDevice(sizeof(uint64_t) * size);
  mapPeerAddrs(cudaStepValue, peerCudaStepValue);

  cudaPeerAddresses = allocateDevice(sizeof(uintptr_t) * 2 * size);
  mapPeerAddrs(cudaPeerAddresses, peerCudaPeerAddresses);

  cudaCommsDeviceDataSent = allocateDevice(sizeof(uint32_t) * size * 32);

  cudaProxyReady = allocateDevice(sizeof(uint32_t) * size);
  mapPeerAddrs(cudaProxyReady, peerCudaProxyReady);

  cudaCopyDone = allocateDevice(sizeof(uint64_t) * 8);
  mapPeerAddrs(cudaCopyDone, peerCudaCopyDone);

  allGather->init();
  reduceScatter->init();

  log.debug("%d: init ok!\n", rank);

  // synchronize to prevent ipc init race conditions
  setupComms->allgather(0);

  log.debug("%d: init synchronized!\n", rank);

  // cpuThread takes over setupComms from here
  cpuThread->start();
  while (!cpuThread->ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
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
  log.verbose("allocated managed memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
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
  log.verbose("allocated device memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}

void* numaAllocate(size_t bytes, int node) {
  void* r;
  if (node != -1) {
    r = numa_alloc_onnode(bytes, node);
  } else {
    r = numa_alloc_local(bytes);
  }
  if (!r) {
    log.error("ERROR: Failed to allocate %d bytes of host memory on NUMA node %d\n", bytes, node);
    throw std::bad_alloc();
  }
  return r;
}

AllocatedBuffer Group::allocateHostMapped(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  r.cpuPointer = numaAllocate(bytes, allocationNode);
  CHECK_CU(cuMemHostRegister(r.cpuPointer, bytes, CU_MEMHOSTREGISTER_DEVICEMAP));
  std::memset(r.cpuPointer, 0, bytes);
  r.numaAllocated = true;
  CUdeviceptr ptr;
  CHECK_CU(cuMemHostGetDevicePointer(&ptr, r.cpuPointer, 0));
  r.cudaPointer = ptr;
  log.verbose(
      "allocated device mapped host memory (%p %#x) of %#x bytes (numa allocated on %d)\n", r.cpuPointer, r.cudaPointer,
      r.bytes, allocationNode);
  return r;
}
AllocatedBuffer Group::allocateHost(size_t bytes) {
  if (bytes < 4096) {
    bytes = 4096;
  }
  AllocatedBuffer r;
  r.bytes = bytes;
  r.cpuPointer = numaAllocate(bytes, allocationNode);
  CHECK_CU(cuMemHostRegister(r.cpuPointer, bytes, 0));
  std::memset(r.cpuPointer, 0, bytes);
  r.numaAllocated = true;
  r.cudaPointer = (uintptr_t)r.cpuPointer;
  log.verbose(
      "allocated host memory (%p %#x) of %#x bytes (numa allocated on %d)\n", r.cpuPointer, r.cudaPointer, r.bytes,
      allocationNode);
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
  log.verbose("allocated write-combined memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  return r;
}

StreamData::StreamData() = default;
StreamData::~StreamData() = default;

void Group::createStreamData(std::unique_ptr<StreamData>& ptr) {
  ptr = std::make_unique<StreamData>();
  ptr->kernels = std::make_unique<Kernels>(this);
}

} // namespace moodist
