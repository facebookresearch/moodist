#include "allgather.h"
#include "group.h"
#include "setup_comms.h"

#include <vector>

namespace moodist {

AllGather::AllGather(Group* group) : group(group) {}

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
    //fmt::printf("node is %s\n", fmt::to_string(fmt::join(list, ", ")));
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

  commsBuffer = group->allocateHost(sizeof(AllGatherComms) * size);

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

} // namespace moodist
