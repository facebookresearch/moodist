#include "allgather.h"
#include "common.h"
#include "group.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include <vector>

namespace moodist {

AllGather::AllGather(Group* group) : CollectiveBase(group) {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
}

AllGather::~AllGather() {}

void AllGather::init() {

  const size_t rank = group->rank;
  const size_t size = group->size;

  SetupComms* setupComms = &*group->setupComms;

  auto& ipcRanks = group->ipcRanks;
  auto& ipcAccess = group->ipcAccess;
  auto& peerIpcAccess = group->peerIpcAccess;

  // Quick and dirty algorithm which requires that all nodes have the same number of local ranks,
  // and that all local ranks have ipc access between each other

  size_t myNode = -1;
  size_t myLocalRank = -1;
  for (size_t i = 0; i != group->nodeRanks.size(); ++i) {
    auto& list = group->nodeRanks[i];
    auto it = std::find(list.begin(), list.end(), rank);
    if (it != list.end()) {
      myNode = i;
      myLocalRank = it - list.begin();
    }
  }
  CHECK(myNode != -1 && myLocalRank != -1);
  size_t ranksPerNode = group->nodeRanks[myNode].size();
  for (auto& list : group->nodeRanks) {
    CHECK(list.size() == ranksPerNode);
    // fmt::printf("node is %s\n", fmt::to_string(fmt::join(list, ", ")));
    for (size_t dst = 0; dst != list.size(); ++dst) {
      for (size_t src = 0; src != list.size(); ++src) {
        if (src == dst) {
          continue;
        }
        if (!peerIpcAccess.at(list[src]).at(list[dst])) {
          fmt::printf("no ipc %d <-> %d\n", list[src], list[dst]);
        }
        CHECK(peerIpcAccess.at(list[src]).at(list[dst]));
      }
    }
  }

  auto findLocalRank = [&](size_t rank) {
    for (auto& list : group->nodeRanks) {
      for (size_t n = 0; n != list.size(); ++n) {
        if (list[n] == rank) {
          return n;
        }
      }
    }
    CHECK(false);
    return (size_t)-1;
  };

  std::vector<std::vector<size_t>> paths;
  for (size_t src = 0; src != size; ++src) {
    size_t l = findLocalRank(src);
    for (auto& list : group->nodeRanks) {
      if (std::find(list.begin(), list.end(), src) != list.end()) {
        for (size_t dst : list) {
          if (src == dst) {
            continue;
          }
          // paths.push_back({src, dst});
        }
      } else {
        for (size_t n = 0; n != list.size(); ++n) {
          if (n == l) {
            paths.push_back({src, list[n]});
          } else {
            paths.push_back({src, list[l], list[n]});
          }
        }
      }
    }
  }

  fmt::printf("paths.size() is %d\n", paths.size());
  CHECK(paths.size() == size * (size - ranksPerNode));
  // for (auto& path : paths) {
  //   fmt::printf(" - path: %s\n", fmt::to_string(fmt::join(path, " -> ")));
  // }

  for (auto& path : paths) {
    CHECK(path.size() <= 3);
    CHECK(path.size() >= 2);
    size_t source = path[0];
    size_t proxy = path[1];
    size_t destination = path.back();
    if (path.size() == 3) {
      CHECK(source != proxy && source != destination && proxy != destination);
    } else if (path.size() == 2) {
      CHECK(source != proxy && source != destination && proxy == destination);
    }
    CHECK((proxy == destination) == (path.size() == 2));
    if (source == rank) {
      if (std::find(sendRanks.begin(), sendRanks.end(), proxy) == sendRanks.end()) {
        sendRanks.push_back(proxy);
      }
    } else if (proxy == rank) {
      if (std::find(recvRanks.begin(), recvRanks.end(), source) == recvRanks.end()) {
        recvRanks.push_back(source);
      }
      if (destination != rank) {
        CHECK(path.size() >= 3);
        ProxyInfo pi;
        pi.source = source;
        pi.destination = destination;
        pi.destinationPeerIndex = group->getPeerIndex(destination);
        proxyInfo.push_back(pi);
      }
    } else if (destination == rank) {
      CHECK(path.size() >= 3);
      ProxyDestinationInfo pid;
      pid.source = source;
      pid.proxy = proxy;
      pid.proxyPeerIndex = group->getPeerIndex(proxy);
      proxyDestinationInfo.push_back(pid);
    }
  }

  // auto& localRanks = group->nodeRanks[myNode];
  // for (size_t n = 0; n != localRanks.size(); ++n) {
  //   size_t i = localRanks[n];
  //   if (i == rank) {
  //     continue;
  //   }
  //   for (auto& list : group->nodeRanks) {
  //     if (&localRanks == &list) {
  //       continue;
  //     }
  //     ProxyInfo pi;
  //     pi.source = list.at(n);
  //     pi.destination = i;
  //     pi.destinationPeerIndex = group->getPeerIndex(i);
  //     proxyInfo.push_back(pi);

  //     ProxyDestinationInfo pdi;
  //     pdi.source = list.at(n);
  //     pdi.proxy = i;
  //     pdi.proxyPeerIndex = group->getPeerIndex(i);
  //     proxyDestinationInfo.push_back(pdi);
  //   }
  // }
  fmt::printf("rank %d: ipc ranks are [%s]\n", rank, fmt::to_string(fmt::join(ipcRanks, ", ")));
  fmt::printf("rank %d: send ranks are [%s]\n", rank, fmt::to_string(fmt::join(sendRanks, ", ")));
  fmt::printf("rank %d: recv ranks are [%s]\n", rank, fmt::to_string(fmt::join(recvRanks, ", ")));

  for (auto& v : proxyInfo) {
    fmt::printf(
        "%d: proxy info source %d, destination %d, peer %d\n", rank, v.source, v.destination, v.destinationPeerIndex);
  }
  for (auto& v : proxyDestinationInfo) {
    fmt::printf("%d: proxy destination info source %d, proxy %d, peer %d\n", rank, v.source, v.proxy, v.proxyPeerIndex);
  }

  // commBuffer = group->allocateHost(sizeof(AllGatherComms) * size);

  auto mapPeerAddrs = [&](AllocatedBuffer& localBuffer, std::array<uintptr_t, 8>& peerPtrs) {
    peerPtrs.fill(0);

    std::array<uintptr_t, 8> peerMapping;

    for (size_t i : group->peerIndices) {
      group->ipcMapper->requestAddress(
          i, localBuffer.cudaPointer, localBuffer.bytes, [&, i](uintptr_t address) { peerMapping[i] = address; });
    }
    group->ipcMapper->wait();
    for (size_t i : group->peerIndices) {
      setupComms->sendTo(group->ipcRanks[i], std::tuple(rank, i, peerMapping[i]));
    }
    for (size_t i : group->peerIndices) {
      auto [sourceRank, sourcePeerIndex, address] =
          setupComms->recvFrom<std::tuple<size_t, size_t, uintptr_t>>(group->ipcRanks[i]);
      CHECK(sourceRank == group->ipcRanks[i]);
      CHECK(sourcePeerIndex == group->peerMyRemoteIndex[i]);
      peerPtrs[i] = address;
    }
    setupComms->allgather(0);
  };

  cudaStepValue = group->allocateDevice(sizeof(uint64_t) * size);
  mapPeerAddrs(cudaStepValue, peerCudaStepValue);

  cudaCopyDone = group->allocateDevice(sizeof(uint64_t) * 8);
  mapPeerAddrs(cudaCopyDone, peerCudaCopyDone);

  // const size_t rank = group->rank;
  // const size_t size = group->size;

  // SetupComms* setupComms = &*group->setupComms;

  // auto& ipcRanks = group->ipcRanks;
  // auto& ipcAccess = group->ipcAccess;
  // auto& peerIpcAccess = group->peerIpcAccess;
  // auto& rankIbDevIndex = group->rankIbDevIndex;
  // auto& rankIbDevNames = group->rankIbDevNames;
  // auto& rankCudaStepDoneBase = group->rankCudaStepDoneBase;

  // peerProxies.resize(size);

  // std::unordered_map<size_t, std::unordered_map<size_t, int>> dataLocationMap;
  // std::unordered_map<size_t, std::unordered_map<size_t, int>> dataNetMap;
  // std::unordered_map<size_t, int> incomingDataCount;
  // std::unordered_map<size_t, int> incomingIbDevDataCount;
  // for (size_t source = 0; source != size; ++source) {
  //   for (size_t destination = 0; destination != size; ++destination) {
  //     if (source == destination || peerIpcAccess[source][destination]) {
  //       dataLocationMap[source][destination] = 1;
  //       continue;
  //     }
  //   }
  // }
  // auto findProxies = [&](int specificProxy = -1) {
  //   for (size_t source = 0; source != size; ++source) {
  //     for (size_t destination = 0; destination != size; ++destination) {
  //       if (dataNetMap[source][destination] || dataLocationMap[source][destination] == 1) {
  //         continue;
  //       }
  //       size_t bestProxy = size;
  //       size_t bestProxyScore = std::numeric_limits<size_t>::max();
  //       for (size_t proxy = 0; proxy != size; ++proxy) {
  //         if (proxy == source || proxy == destination) {
  //           continue;
  //         }
  //         if (specificProxy != -1 && proxy != specificProxy) {
  //           continue;
  //         }
  //         if (dataLocationMap[source][proxy] && peerIpcAccess[proxy][destination]) {
  //           size_t score = 5 + dataLocationMap[source][proxy] + peerProxies[proxy].size() + incomingDataCount[proxy];
  //           int neg = 0;
  //           for (auto& v : peerProxies[proxy]) {
  //             if (v.first == source) {
  //               ++neg;
  //               --score;
  //             }
  //           }
  //           if (score < bestProxyScore) {
  //             bestProxyScore = score;
  //             bestProxy = proxy;
  //           }
  //         }
  //       }
  //       if (bestProxy != size) {
  //         size_t prevScore = dataLocationMap[source][destination];
  //         if (prevScore && prevScore <= bestProxyScore) {
  //           continue;
  //         }
  //         if (prevScore) {
  //           if (rank == 0) {
  //             fmt::printf("%d: replacing an existing proxy path!\n", rank);
  //           }
  //           int found = 0;
  //           for (auto& v : peerProxies) {
  //             for (auto i = v.begin(); i != v.end();) {
  //               auto [s, d] = *i;
  //               if (s == source && d == destination) {
  //                 i = v.erase(i);
  //                 ++found;
  //               } else {
  //                 ++i;
  //               }
  //             }
  //           }
  //           TORCH_CHECK(found == 1);
  //         }
  //         dataLocationMap[source][destination] = bestProxyScore;
  //         peerProxies[bestProxy].emplace_back(source, destination);
  //         if (rank == 0) {
  //           fmt::printf(
  //               "%d: found proxy path with score %d: %d -> %d -> %d\n", rank, bestProxyScore, source, bestProxy,
  //               destination);
  //         }
  //         destination = 0;
  //         source = 0;
  //       }
  //     }
  //   }
  // };
  // findProxies();
  // findProxies();
  // auto findNetworkPath = [&]() {
  //   for (size_t source = 0; source != size; ++source) {
  //     size_t bestDestination = size;
  //     size_t bestScore = std::numeric_limits<size_t>::max();
  //     for (size_t destination = 0; destination != size; ++destination) {
  //       if (dataLocationMap[source][destination]) {
  //         continue;
  //       }
  //       size_t score = peerProxies[destination].size() + incomingDataCount[destination];
  //       score += incomingIbDevDataCount[rankIbDevIndex[destination]];
  //       score *= 2;
  //       if (rankIbDevNames[source] != rankIbDevNames[destination]) {
  //         score += 1;
  //       }
  //       if (score < bestScore) {
  //         bestScore = score;
  //         bestDestination = destination;
  //       }
  //     }
  //     if (bestDestination != size) {
  //       if (rank == 0) {
  //         fmt::printf(
  //             "%d: network path %d -> %d (ib dev index %d) (%s -> %s)\n", rank, source, bestDestination,
  //             rankIbDevIndex[bestDestination], rankIbDevNames[source], rankIbDevNames[bestDestination]);
  //       }
  //       ++incomingIbDevDataCount[rankIbDevIndex[bestDestination]];
  //       ++incomingDataCount[bestDestination];
  //       dataNetMap[source][bestDestination] = 1;
  //       dataLocationMap[source][bestDestination] = 1000;
  //       findProxies(bestDestination);
  //       findProxies();
  //       return true;
  //     }
  //   }
  //   return false;
  // };
  // while (findNetworkPath()) {
  //   // findProxies();
  // }
  // // for (size_t source = 0; source != size; ++source) {
  // //   for (size_t destination = 0; destination != size; ++destination) {
  // //     if (dataLocationMap[source][destination]) {
  // //       continue;
  // //     }
  // //     if (rank == 0) {
  // //       fmt::printf("%d: network path %d -> %d\n", rank, source, destination);
  // //     }
  // //     dataLocationMap[source][destination] = 10;
  // //     findProxies();
  // //     findProxies();
  // //   }
  // // }

  // ipcProxies = peerProxies[rank];

  // if (rank == 0 || true) {
  //   std::string str;
  //   for (size_t source = 0; source != size; ++source) {
  //     for (size_t destination = 0; destination != size; ++destination) {
  //       if (source == destination || peerIpcAccess[source][destination]) {
  //         continue;
  //       }
  //       std::vector<size_t> path;
  //       path.push_back(destination);
  //       while (path.back() != source) {
  //         bool direct = false;
  //         if (dataNetMap[source][path.back()] || peerIpcAccess[source][path.back()]) {
  //           direct = true;
  //         }
  //         bool success = false;
  //         for (size_t proxy = 0; proxy != size && !success; ++proxy) {
  //           for (auto& [s, d] : peerProxies[proxy]) {
  //             if (s == source && d == path.back()) {
  //               TORCH_CHECK(!direct);
  //               TORCH_CHECK(!success);
  //               path.push_back(proxy);
  //               success = true;
  //             }
  //           }
  //         }
  //         if (direct) {
  //           path.push_back(source);
  //           continue;
  //         }
  //         // fmt::printf("partial path for %d -> %d : %s\n", fmt::to_string(fmt::join(path, " <- ")));
  //         if (!success || path.size() >= 8) {
  //           fmt::printf("path for %d -> %d : %s\n", source, destination, fmt::to_string(fmt::join(path, " -> ")));
  //           throw std::runtime_error("deadend path?\n");
  //         }
  //       }
  //       // TORCH_CHECK(peerIpcAccess[source][path.back()]);
  //       std::reverse(path.begin(), path.end());
  //       // str += fmt::sprintf("path for %d -> %d : %s\n", source, destination, fmt::to_string(fmt::join(path, " ->
  //       // ")));

  //       group->networkPaths.emplace_back();
  //       Path& v = group->networkPaths.back();
  //       v.source = source;
  //       v.destination = destination;
  //       v.path = std::move(path);
  //     }
  //   }
  //   // fmt::printf("paths ---\n%s\n", str);
  // }
  // // std::quick_exit(0);

  // // fmt::printf("rank %d: proxying for %d ranks\n", rank, op.ipcProxies.size());

  // std::vector<uint8_t> skipSends;
  // skipSends.resize(size);

  // for (size_t proxy = 0; proxy != size; ++proxy) {
  //   for (auto& [source, destination] : peerProxies[proxy]) {
  //     if (source == rank) {
  //       TORCH_CHECK(!skipSends[destination]);
  //       skipSends[destination] = true;

  //       // fmt::printf(
  //       //     "rank %d skipping send to %d due to a proxy (%d -> %d -> %d)\n", rank, destination, source, proxy,
  //       //     destination);
  //     }
  //   }
  // }

  // std::vector<uint8_t> skipRecv;
  // skipRecv.resize(size);

  // for (size_t proxy = 0; proxy != size; ++proxy) {
  //   for (auto& [source, destination] : peerProxies[proxy]) {
  //     if (destination == rank) {
  //       TORCH_CHECK(!skipRecv[source]);
  //       skipRecv[source] = true;

  //       // fmt::printf(
  //       //     "rank %d skipping recv from %d due to a proxy (%d -> %d -> %d)\n", rank, source, source, proxy,
  //       //     destination);
  //     }
  //   }
  // }

  // rankForwardOrder.reserve(size - 1);
  // for (size_t i = rank == size - 1 ? 0 : rank + 1; i != rank; i = (i == size - 1 ? 0 : i + 1)) {
  //   if (ipcAccess[i] || skipSends[i]) {
  //     continue;
  //   }
  //   rankForwardOrder.push_back(i);
  // }
  // std::vector<uint32_t> rankBackwardOrder = rankForwardOrder;
  // std::reverse(rankBackwardOrder.begin(), rankBackwardOrder.end());

  // for (size_t i = rank == size - 1 ? 0 : rank + 1; i != rank; i = (i == size - 1 ? 0 : i + 1)) {
  //   if (ipcAccess[i] || skipRecv[i]) {
  //     continue;
  //   }
  //   recvRanks.push_back(i);
  // }

  // fmt::printf("rank %d: ipc ranks are [%s]\n", rank, fmt::to_string(fmt::join(ipcRanks, ", ")));
  // fmt::printf("rank %d: forward ranks are [%s]\n", rank, fmt::to_string(fmt::join(rankForwardOrder, ", ")));
  // fmt::printf("rank %d: recv ranks are [%s]\n", rank, fmt::to_string(fmt::join(recvRanks, ", ")));

  // peerInBuffer = group->allocateDevice(peerInBufferSize);
  // peerIpcInputAddress = group->allocateDevice(8 * 16 * 8);

  // CUipcMemHandle peerInIpcMemHandle;
  // CHECK_CU(cuIpcGetMemHandle(&peerInIpcMemHandle, peerInBuffer.cudaPointer));

  // std::vector<CUipcMemHandle> remotePeerInHandles = setupComms->allgather(peerInIpcMemHandle);
  // for (size_t i : ipcRanks) {
  //   CUipcMemHandle remotePeerInHandle = remotePeerInHandles.at(i);
  //   CUdeviceptr peerOut;
  //   CHECK_CU(cuIpcOpenMemHandle(&peerOut, remotePeerInHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
  //   peerOutBase[getPeerIndex(i)] = peerOut;
  // }

  // std::vector<std::vector<size_t>> proxyDestinationsFromSource;
  // proxyDestinationsFromSource.resize(size);
  // for (auto& [source, destination] : ipcProxies) {
  //   proxyDestinationsFromSource[source].push_back(destination);
  // }

  // std::vector<std::vector<std::vector<size_t>>> peerProxyDestinationsFromSource;
  // peerProxyDestinationsFromSource.resize(size);
  // for (size_t proxy = 0; proxy != size; ++proxy) {
  //   auto& list = peerProxyDestinationsFromSource[proxy];
  //   list.resize(size);
  //   for (auto& [source, destination] : peerProxies[proxy]) {
  //     list[source].push_back(destination);
  //   }
  // }

  // auto getPeerProxyBufferOffsetFor = [&](size_t proxy, size_t source) {
  //   size_t offset = sizeof(uintptr_t) * size;
  //   for (size_t s = 0; s != source; ++s) {
  //     if (!peerProxyDestinationsFromSource[proxy][s].empty()) {
  //       offset += sizeof(uintptr_t) * 2 * (peerProxyDestinationsFromSource[proxy][s].size() + 1);
  //     }
  //   }
  //   return offset;
  // };

  // std::vector<size_t> incomingProxiesArrayIndex;
  // std::vector<size_t> incomingProxies;
  // std::vector<size_t> incomingProxiesSource;
  // for (size_t proxy = 0; proxy != size; ++proxy) {
  //   for (auto [source, destination] : peerProxies[proxy]) {
  //     if (destination == rank) {
  //       auto& list = peerProxyDestinationsFromSource[proxy][source];
  //       auto i = std::find(list.begin(), list.end(), rank);
  //       TORCH_CHECK(i != list.end());
  //       incomingProxiesArrayIndex.push_back(i - list.begin());
  //       incomingProxies.push_back(proxy);
  //       incomingProxiesSource.push_back(source);

  //       size_t proxyPeerIndex = getPeerIndex(proxy);

  //       ProxyDestinationInfo info;
  //       info.source = source;
  //       info.proxy = proxy;
  //       info.proxyPeerIndex = proxyPeerIndex;
  //       proxyDestinationInfo.push_back(info);
  //     }
  //   }
  // }
  // std::sort(proxyDestinationInfo.begin(), proxyDestinationInfo.end(), [&](auto& a, auto& b) {
  //   return a.proxyPeerIndex < b.proxyPeerIndex;
  // });
  // proxyDestinationCount = incomingProxies.size();

  // // initialize remainingCopies to the correct value
  // uint32_t remainingCopies = 1 + proxyDestinationCount + ipcRanks.size();
  // CHECK_CU(cuMemcpyHtoD(peerInBuffer.cudaPointer + 5120, &remainingCopies, sizeof(remainingCopies)));

  // recvProxiesPeerIndices.resize(size);

  // for (size_t i : recvRanks) {
  //   for (auto& [source, destination] : ipcProxies) {
  //     if (source == i) {
  //       recvProxiesPeerIndices.at(i).push_back(getPeerIndex(destination));
  //     }
  //   }
  //   std::sort(recvProxiesPeerIndices[i].begin(), recvProxiesPeerIndices[i].end());
  // }
}

void AllGather::compile() {
  CUdevice& cuDevice = group->cuDevice;
  CUcontext& cuContext = group->cuContext;

  int major = 6;
  int minor = 0;
  CHECK_NVRTC(nvrtcVersion(&major, &minor));
  fmt::printf("nvrtc version is %d %d\n", major, minor);

  int driverVersion = 5000;
  CHECK_CU(cuDriverGetVersion(&driverVersion));

  fmt::printf("cuda driver version is %d\n", driverVersion);

  CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  fmt::printf("cuda device compute capability is %d.%d\n", computeMajor, computeMinor);

  int cudaWarpSize = 0;
  int cudaMaxSharedMemory = 0;
  int maxThreadsPerSm = 0;
  int smCount = 0;

  CHECK_CU(cuDeviceGetAttribute(&cudaWarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&cudaMaxSharedMemory, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice));

  CHECK_CU(cuDeviceGetAttribute(&maxThreadsPerSm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice));

  fmt::printf("warp size: %d\nmax shared memory: %d\n", cudaWarpSize, cudaMaxSharedMemory);
  fmt::printf("multiprocessor count: %d\nmax threads per multiprocessor: %d\n", smCount, maxThreadsPerSm);

  // allgatherBlocksize = group->ipcRanks.size() == group->size - 1 ? 256 : 64;
  // allgatherGridsize = group->ipcRanks.size() == group->size - 1 ? smCount / 4 : smCount / 4;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;

  auto waitFor32 = [&](uintptr_t address, std::string value) {
    return replace(R"(while (*(volatile uint32_t*)$ptr < $value);)", "$ptr", address, "$value", value);
  };

  std::string writes;
  std::string waits;
  for (size_t i : group->peerIndices) {
    waits += waitFor32(cudaStepValue.cudaPointer + sizeof(uint64_t) * group->ipcRanks[i], "stepValue") + "\n";
    writes += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", peerCudaStepValue[i] + sizeof(uint64_t) * rank);
  }

  std::string copyDoneFunctions;
  std::string copyDoneAllCode;
  std::string waitForCopyDones;
  for (size_t i : group->peerIndices) {
    copyDoneFunctions += replace(
        R"(
extern "C" __global__ void allgather_copy_done_$i(uint32_t stepValue) {
  *(volatile uint32_t*)$ptr = stepValue;
}
      )",
        "$i", i, "$ptr", peerCudaCopyDone[i] + sizeof(uint32_t) * peerMyRemoteIndex[i]);
    copyDoneAllCode += replace(
        R"(
  *(volatile uint32_t*)$ptr = stepValue;
      )",
        "$ptr", peerCudaCopyDone[i] + sizeof(uint32_t) * peerMyRemoteIndex[i]);
    waitForCopyDones += replace(
        R"(
  while (*(volatile uint32_t*)$ptr < stepValue);
      )",
        "$i", i, "$ptr", cudaCopyDone.cudaPointer + sizeof(uint32_t) * i);
  }

  std::string waitForRecvs;
  for (size_t i : recvRanks) {
    waitForRecvs += waitFor32(cudaStepValue.cudaPointer + sizeof(uint32_t) * i, "stepValue");
  }

  std::string source = replace(
      R"zz(

typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;


extern "C" __global__ void allgather_entry(uint32_t stepValue) {
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  cpuIn[0] = stepValue;
  $writes
  $waits
}

extern "C" __global__ void allgather_exit(uint32_t stepValue) {
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  while (cpuOut[0] < stepValue);
  $waitForCopyDones
  $waitForRecvs
}

$copyDoneFunctions

extern "C" __global__ void allgather_copy_all_done(uint32_t stepValue) {
  $copyDoneAllCode
}

  )zz",
      "$cpuOut", cast("volatile uint32_t*", cpuOutBuffer.cudaPointer), "$cpuIn",
      cast("volatile uint32_t*", cpuInBuffer.cudaPointer), "$writes", writes, "$waits", waits, "$copyDoneFunctions",
      copyDoneFunctions, "$waitForCopyDones", waitForCopyDones, "$waitForRecvs", waitForRecvs, "$copyDoneAllCode",
      copyDoneAllCode);

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  // if (rank == 0) {
  //   fmt::printf("code\n----\n%s\n", source);
  // }

  if (rank == 0 || true) {
    std::string fn = fmt::sprintf("allgather_kernel-rank%d.cu", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(source.data(), source.size(), 1, f);
      fclose(f);
      fmt::printf("source dumped to %s\n", fn);
    }
  }

  nvrtcProgram program;
  CHECK_NVRTC(nvrtcCreateProgram(&program, source.c_str(), nullptr, 0, nullptr, nullptr));

  std::vector<std::pair<int, std::string>> archOptions;
  archOptions.emplace_back(9000, "--gpu-architecture=sm_90");
  archOptions.emplace_back(8090, "--gpu-architecture=sm_89");
  archOptions.emplace_back(8070, "--gpu-architecture=sm_87");
  archOptions.emplace_back(8000, "--gpu-architecture=sm_80");
  archOptions.emplace_back(7050, "--gpu-architecture=sm_75");
  archOptions.emplace_back(7020, "--gpu-architecture=sm_72");
  archOptions.emplace_back(7000, "--gpu-architecture=sm_70");
  archOptions.emplace_back(6020, "--gpu-architecture=sm_62");
  archOptions.emplace_back(6010, "--gpu-architecture=sm_61");
  archOptions.emplace_back(6000, "--gpu-architecture=sm_60");
  archOptions.emplace_back(5030, "--gpu-architecture=sm_53");
  archOptions.emplace_back(5020, "--gpu-architecture=sm_52");
  archOptions.emplace_back(0, "--gpu-architecture=sm_50");

  std::vector<std::string> options;
  options.push_back("<gpu-architecture>");
  options.push_back("--use_fast_math");
  options.push_back("--std=c++17");
  options.push_back("-lineinfo");
  // options.push_back("--maxrregcount=32");
  //    options.push_back("-G");
  //     options.push_back("--dopt=on");
  nvrtcResult error = NVRTC_ERROR_INVALID_OPTION;
  for (size_t i = 0; i != archOptions.size() && error == NVRTC_ERROR_INVALID_OPTION; ++i) {
    if (computeMajor * 1000 + computeMinor * 10 < archOptions[i].first) {
      continue;
    }
    options[0] = archOptions[i].second;
    std::vector<const char*> options2;
    for (auto& v : options) {
      options2.push_back(v.c_str());
    }
    error = nvrtcCompileProgram(program, options2.size(), options2.data());
    if (error == NVRTC_SUCCESS) {
      fmt::printf("success with %s\n", archOptions[i].second);
    }
  }
  if (error != NVRTC_SUCCESS) {
    fmt::printf("Failed to compile--\n%s\n", source.c_str());
    size_t logSize = 0;
    std::string log;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &logSize));
    log.resize(logSize);
    CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));
    fmt::printf("%s\n", log);

    CHECK_NVRTC(error);
  } else {
    size_t logSize = 0;
    std::string log;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &logSize));
    log.resize(logSize);
    CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));
    fmt::printf("compile log\n---\n%s\n", log);
  }

  // size_t ptxSize = 0;
  // CHECK_NVRTC(nvrtcGetPTXSize(program, &ptxSize));
  // std::vector<char> ptx;
  // ptx.resize(ptxSize);
  // CHECK_NVRTC(nvrtcGetPTX(program, ptx.data()));

  // CHECK_NVRTC(nvrtcDestroyProgram(&program));

  // CHECK_CU(cuModuleLoadDataEx(&op.cuModule, ptx.data(), 0, nullptr, nullptr));
  // CHECK_CU(cuModuleGetFunction(&op.cuAllgather, op.cuModule, "allgather"));

  size_t cubinSize = 0;
  CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubinSize));
  std::vector<char> cubin;
  cubin.resize(cubinSize);
  fmt::printf("cubin size is %d\n", cubin.size());
  CHECK_NVRTC(nvrtcGetCUBIN(program, cubin.data()));

  if (rank == 0 || true) {
    FILE* f = fopen(fmt::sprintf("allgather_kernel-rank%d.o", rank).c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      fmt::printf("cubin dumped to allgather_kernel.o\n");
    }
  }

  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));
  CHECK_CU(cuModuleGetFunction(&cuAllgatherEntry, cuModule, "allgather_entry"));
  CHECK_CU(cuModuleGetFunction(&cuAllgatherExit, cuModule, "allgather_exit"));
  for (size_t i : group->peerIndices) {
    CHECK_CU(
        cuModuleGetFunction(&cuAllgatherCopyDone[i], cuModule, replace("allgather_copy_done_$i", "$i", i).c_str()));
  }
  CHECK_CU(cuModuleGetFunction(&cuAllgatherCopyAllDone, cuModule, "allgather_copy_all_done"));

  // for (size_t i = 0; i != ipcRanks.size(); ++i) {
  //   CHECK_CU(cuModuleGetFunction(
  //       &cuAllgatherWaitForPeer[i], cuModule, fmt::sprintf("allgather_wait_for_peer_%d", i).c_str()));
  // }
  // cuAllgatherWaitForPeerData.resize(size);
  // for (size_t i : recvRanks) {
  //   CHECK_CU(cuModuleGetFunction(
  //       &cuAllgatherWaitForPeerData[i], cuModule, fmt::sprintf("allgather_wait_for_data_%d", i).c_str()));
  // }

  // size_t globalStepValueSize = 0;
  // CHECK_CU(cuModuleGetGlobal(&cuGlobalStepValue, &globalStepValueSize, cuModule, "globalStepValue"));
  // TORCH_CHECK(globalStepValueSize == 4);
}

} // namespace moodist
