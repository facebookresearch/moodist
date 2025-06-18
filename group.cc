// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "group.h"
#include "allgather.h"
#include "common.h"
#include "cputhread.h"
#include "ib_common.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "reduce_scatter.h"
#include "setup_comms.h"

#include <algorithm>
#include <cstring>
#include <exception>
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
  kernels = std::make_unique<Kernels>(this);
  allGather = std::make_unique<AllGather>(this);
  reduceScatter = std::make_unique<ReduceScatter>(this);
}

Group::~Group() {
  if (cpuThread) {
    cpuThread->~CpuThread();
    internalFree(cpuThread);
    cpuThread = nullptr;
  }
}

void Group::init(Function<void()> f) {

  auto start = std::chrono::steady_clock::now();

  deviceIndex = c10::cuda::current_device();
  pid = ::getpid();

  cuContext = nullptr;
  cuCtxGetCurrent(&cuContext);
  if (!cuContext) {
    CHECK_CU(cuInit(0));

    CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
    CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
    CHECK_CU(cuCtxSetCurrent(cuContext));
  } else {
    CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
  }

  // cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, 0x10000);

  int driverVersion = 5000;
  CHECK_CU(cuDriverGetVersion(&driverVersion));

  log.verbose("cuda driver version is %d\n", driverVersion);

  int clockRate;
  CHECK_CU(cuDeviceGetAttribute(&clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cuDevice));
  log.verbose("device clock rate: %dkhz\n", clockRate);

  int asyncEngines;
  CHECK_CU(cuDeviceGetAttribute(&asyncEngines, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cuDevice));
  log.verbose("device async engines: %d\n", asyncEngines);

  CHECK_NVML(nvmlInit_v2());

  std::array<char, 0x400> cudaPciBus;
  CHECK_CU(cuDeviceGetPCIBusId(cudaPciBus.data(), cudaPciBus.size(), cuDevice));
  cudaPciBus[0x3ff] = 0;
  auto getDevicePath = [](std::string pciBus) {
    std::string s = fmt::sprintf("/sys/bus/pci/devices/%s", pciBus);
    for (auto& v : s) {
      if (v >= 'A' && v <= 'Z') {
        v += 0x20;
      }
    }
    char* path = realpath(s.c_str(), nullptr);
    if (path) {
      s = path;
      free(path);
    }
    return removePciPathPrefix(s);
  };
  std::string cudaPath = getDevicePath(cudaPciBus.data());
  log.debug("cuda path is %s\n", cudaPath);

  nvmlDevice_t nvmlDevice;
  CHECK_NVML(nvmlDeviceGetHandleByPciBusId_v2(cudaPciBus.data(), &nvmlDevice));

  std::vector<std::string> allCudaPaths;

  bool localCudaDeviceFound = false;
  size_t localAllCudaPathsIndex = 0;
  unsigned int deviceCount = 0;
  CHECK_NVML(nvmlDeviceGetCount(&deviceCount));
  for (unsigned int i = 0; i != deviceCount; ++i) {
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex_v2(i, &device));
    nvmlPciInfo_t pciInfo;
    CHECK_NVML(nvmlDeviceGetPciInfo_v3(device, &pciInfo));
    std::string path = getDevicePath(pciInfo.busIdLegacy);
    allCudaPaths.push_back(path);
    if (path == cudaPath) {
      localCudaDeviceFound = true;
      localAllCudaPathsIndex = i;
    }
    log.debug("there is a cuda device at %s\n", path);
  }
  if (!localCudaDeviceFound) {
    throw std::runtime_error("The current CUDA device could not be found using NVML!");
  }

  unsigned long nodeSet = 0;
  CHECK_NVML(nvmlDeviceGetMemoryAffinity(nvmlDevice, 1, &nodeSet, NVML_AFFINITY_SCOPE_NODE));

  for (unsigned int i = 0; i != 64; ++i) {
    if (nodeSet & (1ull << i)) {
      allocationNode = i;
      break;
    }
  }

  log.debug("%d: allocationNode is %d\n", rank, allocationNode);

  internalAllocatorNode = allocationNode;
  cpuThread = new (internalAlloc(sizeof(CpuThread))) CpuThread(this);

  init2 = [&]() {
    f();

    std::vector<LocalDevice> localDevices;

    std::vector<LocalDeviceNode> localDeviceNodes;

    std::vector<std::vector<LocalDeviceNode>> allDeviceNodes;
    allDeviceNodes.resize(allCudaPaths.size());

    std::string bootId = readBootId();

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

        auto getScore = [&](std::string cudaPath) {
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
          return nmatch;
        };

        log.debug("rank %d: %s -> %s has score %d %d\n", rank, cudaPath, ibPath, getScore(cudaPath), bestSpeed);

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

        std::tuple<int, int, int> score = {getScore(cudaPath), 0, bestSpeed};
        LocalDeviceNode n;
        n.ibPath = ibPath;
        n.score = score;
        localDeviceNodes.push_back(n);

        for (size_t i = 0; i != allCudaPaths.size(); ++i) {
          std::tuple<int, int, int> score = {getScore(allCudaPaths[i]), 0, bestSpeed};
          LocalDeviceNode n;
          n.ibPath = ibPath;
          n.score = score;
          allDeviceNodes.at(i).push_back(n);
        }
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
    CHECK(std::find(localRanksByBootId.begin(), localRanksByBootId.end(), rank) != localRanksByBootId.end());

    log.debug("%d: local ranks: [%s]\n", rank, fmt::to_string(fmt::join(localRanksByBootId, ", ")));

    size_t localRankIndex = ~(size_t)0;
    for (size_t i = 0; i != localRanksByBootId.size(); ++i) {
      if (localRanksByBootId[i] == rank) {
        localRankIndex = i;
        break;
      }
    }
    CHECK(localRankIndex >= 0 && localRankIndex < localRanksByBootId.size());

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
        current.resize(allDeviceNodes.size());
        std::vector<LocalDeviceNode*> best;
        std::tuple<int, int, int> bestScore = {0, 0, 0};
        int solutions = 0;
        std::function<void(size_t)> visit = [&](size_t index) {
          if (index == allDeviceNodes.size()) {
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
          std::vector<std::pair<std::tuple<int, int, int>, LocalDeviceNode*>> sorted;
          for (LocalDeviceNode& v : allDeviceNodes.at(index)) {
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

        for (size_t i = 0; i != best.size(); ++i) {
          log.debug("best device for %d is %s with score %d\n", i, best[i]->ibPath, std::get<0>(best[i]->score));
        }

        for (auto& v : localDeviceNodes) {
          if (v.ibPath == best.at(localAllCudaPathsIndex)->ibPath) {
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

      IbvSharedPtr<ibv_context, ibv_close_device>& bestCtx = ibCommon->context;
      int bestPortNum = 0;
      ibv_port_attr bestPortAttributes;

      bestCtx = std::move(d->ctx);
      bestPortNum = d->portNum;
      bestPortAttributes = d->portAttributes;
      CHECK(bestCtx != nullptr);

      log.verbose("rank %d using device %s (%s)\n", rank, bestCtx->device->name, bestCtx->device->ibdev_path);

      ibCommon->init(bestPortNum, bestPortAttributes);

      ibDevs.push_back(std::move(ibCommon));

      if (ibDevs.size() == maxDevices) {
        break;
      }
    }
    numTrueIbDevs = ibDevs.size();
    while (ibDevs.size() >= 1 && ibDevs.size() + useDevices.size() <= maxDevices) {
      // while (ibDevs.size() >= 1 && ibDevs.size() + useDevices.size() <= 2) {
      for (size_t i = 0; i != useDevices.size(); ++i) {
        auto ibCommon = std::make_unique<IbCommon>(this);
        ibCommon->protectionDomain = ibDevs.at(i)->protectionDomain;

        IbvSharedPtr<ibv_context, ibv_close_device>& bestCtx = ibCommon->context;
        int bestPortNum = 0;
        ibv_port_attr bestPortAttributes;

        auto* d = useDevices.at(i).second;

        bestCtx = ibDevs.at(i)->context;
        bestPortNum = d->portNum;
        bestPortAttributes = d->portAttributes;
        CHECK(bestCtx != nullptr);

        log.verbose(
            "rank %d using duplicate of device %s (%s)\n", rank, bestCtx->device->name, bestCtx->device->ibdev_path);

        ibCommon->init(bestPortNum, bestPortAttributes);

        ibDevs.push_back(std::move(ibCommon));
      }
    }

    CHECK(!ibDevs.empty());
    CHECK(ibDevs.size() <= maxDevices);

    std::vector<std::string> allRanksBootIds = setupComms->allgather(bootId);

    HashMap<std::string, size_t> nodeMap;
    for (size_t i = 0; i != size; ++i) {
      std::string& id = allRanksBootIds[i];
      auto it = nodeMap.find(id);
      if (it == nodeMap.end()) {
        nodeMap[id] = nodeRanks.size();
        nodeRanks.emplace_back();
      }
      size_t index = nodeMap[id];
      size_t localRank = nodeRanks.at(index).size();
      nodeRanks.at(index).push_back(i);
      rankToNodeIndex.push_back(index);
      rankLocalRank.push_back(localRank);
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
      log.debug("initial connection setup took %g ms\n", seconds(std::chrono::steady_clock::now() - start) * 1000);
      start = std::chrono::steady_clock::now();
    }

    std::sort(ipcRanks.begin(), ipcRanks.end());
    std::stable_partition(ipcRanks.begin(), ipcRanks.end(), [&](size_t i) { return i > rank; });

    std::vector<IVector<size_t>> peerIpcRanks = setupComms->allgather(ipcRanks);

    {

      IVector<size_t> newIpcRanks;
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
    // CHECK(rankIbDevNames.size() == size);
    // CHECK(rankIbDevNames.at(rank) == bestCtx->device->name);
    // if (rank == 0) {
    //   for (size_t i = 0; i != size; ++i) {
    //     fmt::printf("rank %d is using device %s\n", i, rankIbDevNames[i]);
    //   }
    // }

    if (rank == 0) {
      log.debug("ipc ranks sync took %g ms\n", seconds(std::chrono::steady_clock::now() - start) * 1000);
      start = std::chrono::steady_clock::now();
    }

    peerIpcAccess.resize(size);

    for (size_t i = 0; i != size; ++i) {
      peerIpcAccess[i].resize(size);
      for (size_t n : peerIpcRanks[i]) {
        CHECK(std::find(peerIpcRanks[n].begin(), peerIpcRanks[n].end(), i) != peerIpcRanks[n].end());
        peerIpcAccess[i][n] = true;
      }
    }

    // int localSupportsMulticast = 0;
    // CHECK_CU(cuDeviceGetAttribute(&localSupportsMulticast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cuDevice));
    // supportsMulticast = localSupportsMulticast;
    // for (size_t i : peerIndices) {
    //   setupComms->sendTo(ipcRanks[i], supportsMulticast);
    // }
    // for (size_t i : peerIndices) {
    //   supportsMulticast &= setupComms->recvFrom<bool>(ipcRanks[i]);
    // }

    // if (supportsMulticast) {
    //   log.info("Multicast supported by all local devices\n");
    // } else {
    //   log.info("Multicast not supported by all local devices\n");
    // }

    if (rank == 0) {
      log.debug(
          "final connection setup and sync took %g ms\n", seconds(std::chrono::steady_clock::now() - start) * 1000);
      start = std::chrono::steady_clock::now();
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

    ipcMapper->init(allocationNode);

    mySharedMem = ipcMapper->getMySharedMem(0, 0);
    peerSharedMem.fill(nullptr);
    for (size_t i : peerIndices) {
      peerSharedMem[i] = ipcMapper->getPeerSharedMem(i, 0, 0);
    }

    size_t offset = 0;
    uintptr_t addr = (uintptr_t)mySharedMem;
    CHECK(addr % 128 == 0);
    auto get = [&](auto*& ptr, size_t n) {
      using T = std::remove_pointer_t<std::remove_reference_t<decltype(ptr)>>;
      if (offset % alignof(T) != 0) {
        offset += alignof(T) - offset % alignof(T);
      }
      CHECK((addr + offset) % alignof(T) == 0);
      ptr = (T*)(addr + offset);
      offset += sizeof(T) * n;
    };
    get(localDyns, size * Group::maxConcurrency);
    get(cpuLocalDyns, size * Group::maxConcurrency);
    get(localProgress, size * Group::maxConcurrency);
    get(cpuAddresses, Group::maxConcurrency);
    get(syncData, 1);
    get(cpuProxyReady, size * Group::maxConcurrency);
    get(localProgress2, size * Group::maxConcurrency);
    get(cpuStepValues, size * Group::maxConcurrency);
    get(cpuStepValuesDeviceChunks, Group::maxChunks * Group::maxDevices * size * Group::maxConcurrency);
    get(atomicStepValue, size * Group::maxConcurrency);

    mySharedMemSize = offset;

    ipcMapper->getMySharedMem(0, offset);

    cpuOutBuffer = allocateArrayDevice(4096 + sizeof(uint32_t) * maxChunks * size, Group::maxConcurrency);
    cpuInBuffer = allocateArrayHostMapped(4096 + sizeof(uint32_t) * maxChunks * size, Group::maxConcurrency);

    {
      unsigned int value = 1;
      CHECK_CU(cuPointerSetAttribute(&value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, cpuOutBuffer.buffer.cudaPointer));
    }

    auto mapPeerAddrs = [&](AllocatedArray& localArray, std::array<PeerArrayRef, 8>& peerPtrs) {
      peerPtrs.fill({});
      auto& localBuffer = localArray.buffer;

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
        peerPtrs[i] = {address, localArray.itembytes};
      }
      setupComms->allgather(0);
    };

    cudaStepValue = allocateArrayDevice(sizeof(uint64_t) * size, Group::maxConcurrency);
    mapPeerAddrs(cudaStepValue, peerCudaStepValue);

    cudaPeerAddresses = allocateArrayDevice(sizeof(uintptr_t) * 2 * size, Group::maxConcurrency);
    mapPeerAddrs(cudaPeerAddresses, peerCudaPeerAddresses);

    cudaCommsDeviceDataSent = allocateArrayDevice(sizeof(uint32_t) * size * 32, Group::maxConcurrency);
    cpuCommsDeviceDataSent = allocateArrayHost(sizeof(uint32_t) * size * 32, Group::maxConcurrency);

    cudaCommsDeviceDataSent2 = allocateArrayDevice(sizeof(uint32_t) * size * 32 * 32, Group::maxConcurrency);
    cpuCommsDeviceDataSent2 = allocateArrayHost(sizeof(uint32_t) * size * 32 * 32, Group::maxConcurrency);

    cudaProxyReady = allocateArrayDevice(
        sizeof(uint32_t) * std::max(size * maxChunks, size * peerIndices.size()), Group::maxConcurrency);
    mapPeerAddrs(cudaProxyReady, peerCudaProxyReady);

    cudaCopyDone = allocateArrayDevice(sizeof(uint64_t) * 8, Group::maxConcurrency);
    mapPeerAddrs(cudaCopyDone, peerCudaCopyDone);

    cudaStepValuesBuffer = allocateArrayDevice(sizeof(uint32_t) * size, Group::maxConcurrency);
    cudaStepValuesDeviceChunksBuffer =
        allocateArrayDevice(sizeof(uint32_t) * Group::maxChunks * Group::maxDevices * size, Group::maxConcurrency);

    cudaMemSync = allocateArrayDevice(sizeof(uint32_t) * size, Group::maxConcurrency);
    mapPeerAddrs(cudaMemSync, peerCudaMemSync);

    // This list MUST contain all buffers that should be reset on step value reset.
    // Any buffers which receive step values from peers should be reset.
    // All such buffers should be allocated in this function.
    buffersToReset = {
        &cpuOutBuffer,
        &cpuInBuffer,
        &cudaStepValue,
        &cudaPeerAddresses,
        &cudaCommsDeviceDataSent,
        &cpuCommsDeviceDataSent,
        &cudaCommsDeviceDataSent2,
        &cpuCommsDeviceDataSent2,
        &cudaProxyReady,
        &cudaCopyDone,
        &cudaStepValuesBuffer,
        &cudaStepValuesDeviceChunksBuffer,
        &cudaMemSync,
    };

    allGather->init();
    reduceScatter->init();

    log.debug("%d: init ok!\n", rank);

    // synchronize to prevent ipc init race conditions
    setupComms->allgather(0);

    log.debug("%d: init synchronized!\n", rank);
  };

  cpuThread->start();
  while (!cpuThread->ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (cpuThread->initException) {
      std::rethrow_exception(cpuThread->initExceptionPtr);
    }
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

  bytesManaged += bytes;
  reportBytes();
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

  bytesDevice += bytes;
  reportBytes();
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
  CHECK(((uintptr_t)r & 63) == 0);
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
  bytesHost += bytes;
  reportBytes();
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
  bytesHost += bytes;
  reportBytes();
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
AllocatedBuffer Group::allocateDeviceMapped(size_t bytes) {
  CHECK(false);
  return {};
  // if (bytes < 4096) {
  //   bytes = 4096;
  // }
  // AllocatedBuffer r;
  // r.bytes = bytes;

  // size_t allocationGranularity = 1;
  // CUmemAllocationProp prop;
  // std::memset(&prop, 0, sizeof(prop));
  // prop.location.id = deviceIndex;
  // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  // CHECK_CU(cuMemGetAllocationGranularity(&allocationGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  // log.info("allocation granularity is %#x\n", allocationGranularity);

  // size_t alignment = std::max((size_t)4096, allocationGranularity);
  // r.bytes = (bytes + alignment - 1) / alignment * alignment;

  // CHECK_CU(cuMemCreate(&r.handle, r.bytes, &prop, 0));

  // log.info("handle created\n");

  // CUdeviceptr ptr = 0;
  // CHECK_CU(cuMemAddressReserve(&ptr, r.bytes, alignment, 0, 0));
  // log.info("reserved address %#x\n", ptr);
  // CHECK_CU(cuMemMap(ptr, r.bytes, 0, r.handle, 0));
  // log.info("mapped moo!\n");

  // // std::array<CUmemAccessDesc, 2> desc;
  // // std::memset(desc.data(), 0, sizeof(CUmemAccessDesc) * desc.size());
  // // desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // // desc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  // // desc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // // desc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // // desc[1].location.id = deviceIndex;
  // // CHECK_CU(cuMemSetAccess(ptr, r.bytes, desc.data(), 2));

  // std::array<CUmemAccessDesc, 2> desc;
  // std::memset(desc.data(), 0, sizeof(CUmemAccessDesc) * desc.size());
  // desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // desc[0].location.id = deviceIndex;
  // CHECK_CU(cuMemSetAccess(ptr, r.bytes, desc.data(), 1));

  // log.info("access 1 ok\n");

  // std::memset(desc.data(), 0, sizeof(CUmemAccessDesc) * desc.size());
  // desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // desc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  // desc[0].location.id = allocationNode;
  // CHECK_CU(cuMemSetAccess(ptr, r.bytes, desc.data(), 1));

  // log.info("access 2 ok\n");

  // r.cudaPointer = ptr;
  // r.cpuPointer = (void*)ptr;

  // std::memset(r.cpuPointer, 0, bytes);
  // r.handleAllocated = true;
  // log.verbose("allocated host mapped device memory (%p %#x) of %#x bytes\n", r.cpuPointer, r.cudaPointer, r.bytes);
  // return r;
}

AllocatedCpuBuffer Group::allocateCpu(size_t bytes) {
  AllocatedCpuBuffer r;
  r.bytes = bytes;
  r.cpuPointer = cpu_allocator::moo_alloc(bytes);
  return r;
}

AllocatedArray Group::allocateArrayHost(size_t itembytes, size_t numel) {
  AllocatedArray r;
  r.buffer = allocateHost(itembytes * numel);
  r.itembytes = itembytes;
  r.numel = numel;
  return r;
}
AllocatedArray Group::allocateArrayHostMapped(size_t itembytes, size_t numel) {
  AllocatedArray r;
  r.buffer = allocateHostMapped(itembytes * numel);
  r.itembytes = itembytes;
  r.numel = numel;
  return r;
}
AllocatedArray Group::allocateArrayDevice(size_t itembytes, size_t numel) {
  AllocatedArray r;
  r.buffer = allocateDevice(itembytes * numel);
  r.itembytes = itembytes;
  r.numel = numel;
  return r;
}
AllocatedArray Group::allocateArrayWriteCombined(size_t itembytes, size_t numel) {
  AllocatedArray r;
  r.buffer = allocateWriteCombined(itembytes * numel);
  r.itembytes = itembytes;
  r.numel = numel;
  return r;
}
AllocatedArray Group::allocateArrayManaged(size_t itembytes, size_t numel) {
  AllocatedArray r;
  r.buffer = allocateManaged(itembytes * numel);
  r.itembytes = itembytes;
  r.numel = numel;
  return r;
}
AllocatedArray Group::allocateArrayDeviceMapped(size_t itembytes, size_t numel) {
  AllocatedArray r;
  r.buffer = allocateDeviceMapped(itembytes * numel);
  r.itembytes = itembytes;
  r.numel = numel;
  return r;
}

TreeSendsRecvs Group::generateTree(size_t nary, size_t rootRank) {
  log.info("Generating new tree, %d-ary, root %d\n", nary, rootRank);
  size_t rootLocalRank = rankLocalRank.at(rootRank);
  size_t nNodes = nodeRanks.size();
  CHECK(nNodes > 1);
  size_t rootNode = rankToNodeIndex.at(rootRank);
  size_t nodeIndex = rankToNodeIndex.at(rank);
  size_t rotatedNodeIndex = (nodeIndex + nNodes - rootNode) % nNodes;
  auto& node = nodeRanks.at(nodeIndex);
  CHECK(node.at(rootLocalRank % node.size()) == rank);

  size_t parent = (rotatedNodeIndex - 1) / nary;
  size_t childBegin = rotatedNodeIndex * nary + 1;

  TreeSendsRecvs r;

  if (rotatedNodeIndex != 0) {
    CHECK(parent < nNodes);
    auto& parentNode = nodeRanks.at((parent + rootNode) % nNodes);
    r.sends.push_back(parentNode.at(rootLocalRank % parentNode.size()));
  }
  for (size_t i = childBegin; i != childBegin + nary; ++i) {
    if (i >= nNodes) {
      break;
    }
    auto& childNode = nodeRanks.at((i + rootNode) % nNodes);
    r.recvs.push_back(childNode.at(rootLocalRank % childNode.size()));
  }

  return r;
}

StreamData::StreamData() = default;
StreamData::~StreamData() = default;

void Group::createStreamData(std::unique_ptr<StreamData>& ptr) {
  ptr = std::make_unique<StreamData>();
}

uintptr_t Group::getNextCudaUint32() {
  std::lock_guard l(cudaUint32Mutex);
  if (cudaUint32List.empty() || cudaUint32Offset == cudaUint32List.back().bytes) {
    cudaUint32List.push_back(allocateDevice(0x1000));
    cudaUint32Offset = 0;
  }
  size_t o = cudaUint32Offset;
  cudaUint32Offset += 4;
  return cudaUint32List.back().cudaPointer + o;
}

} // namespace moodist
