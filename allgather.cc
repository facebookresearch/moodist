#include "allgather.h"
#include "common.h"
#include "group.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "setup_comms.h"

#include <vector>

namespace moodist {

AllGather::AllGather(Group* group) : CollectiveBase(group) {}

AllGather::~AllGather() {}

void AllGather::init() {

  const size_t rank = group->rank;
  const size_t size = group->size;

  SetupComms* setupComms = &*group->setupComms;

  auto& ipcRanks = group->ipcRanks;
  auto& ipcAccess = group->ipcAccess;
  auto& peerIpcAccess = group->peerIpcAccess;
  auto& nodeRanks = group->nodeRanks;

  // Quick and dirty algorithm which requires that all nodes have the same number of local ranks,
  // and that all local ranks have ipc access between each other
  // This also assumes that ranks are sequential on nodes.

  size_t myNode = -1;
  size_t myLocalRank = -1;
  for (size_t i = 0; i != nodeRanks.size(); ++i) {
    auto& list = nodeRanks[i];
    auto it = std::find(list.begin(), list.end(), rank);
    if (it != list.end()) {
      myNode = i;
      myLocalRank = it - list.begin();
    }
  }
  CHECK(myNode != -1 && myLocalRank != -1);
  size_t ranksPerNode = nodeRanks[myNode].size();
  for (auto& list : nodeRanks) {
    // CHECK(list.size() == ranksPerNode);
    //  fmt::printf("node is %s\n", fmt::to_string(fmt::join(list, ", ")));
    for (size_t dst = 0; dst != list.size(); ++dst) {
      for (size_t src = 0; src != list.size(); ++src) {
        if (src == dst) {
          continue;
        }
        if (!peerIpcAccess.at(list[src]).at(list[dst])) {
          log.error("no ipc %d <-> %d\n", list[src], list[dst]);
        }
        CHECK(peerIpcAccess.at(list[src]).at(list[dst]));
      }
    }
  }

  auto findLocalRank = [&](size_t rank) {
    for (auto& list : nodeRanks) {
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
    size_t ol = findLocalRank(src);
    for (auto& list : nodeRanks) {
      if (std::find(list.begin(), list.end(), src) != list.end()) {
        continue;
      }
      size_t ll = ol % list.size();
      if (src == rank) {
        paths.push_back({src, list[ll]});
      } else if (list[ll] == rank) {
        for (size_t n = 0; n != list.size(); ++n) {
          if (n == ll) {
            paths.push_back({src, list[n]});
          } else {
            paths.push_back({src, list[ll], list[n]});
          }
        }
      } else if (std::find(list.begin(), list.end(), rank) != list.end()) {
        paths.push_back({src, list[ll], rank});
      }
    }
  }

  auto abs = [&](size_t v, size_t n) {
    v = (v + n) % n;
    return v <= (n - v) ? v : n - v;
  };

  auto sortFn = [&](auto& a, auto& b) {
    if (a.size() != b.size()) {
      return a.size() < b.size();
    }
    int am;
    int bm;
    // am = abs(a[1] - a[0], size);
    // bm = abs(b[1] - b[0], size);
    am = (size + a[1] - a[0]) % size;
    bm = (size + b[1] - b[0]) % size;
    if (am != bm) {
      return am < bm;
    }
    if (a.size() == 3) {
      am = (ranksPerNode + a[2] - a[1]) % ranksPerNode;
      bm = (ranksPerNode + b[2] - b[1]) % ranksPerNode;
      if (am != bm) {
        return am > bm;
      }
    }
    // if ((a[1] - a[0] + size) % size != (b[1] - b[0] + size) % size) {
    //   return (a[1] - a[0] + size) % size > (b[1] - b[0] + size) % size;
    // }
    return a < b;
  };
  std::sort(paths.begin(), paths.end(), sortFn);

  std::stable_partition(paths.begin(), paths.end(), [&](auto& a) { return a[0] == rank; });
  std::stable_partition(paths.begin(), paths.end(), [&](auto& a) {
    if (a.size() < 3) {
      return true;
    }
    return a[1] != rank;
  });

  for (auto& v : paths) {
    if (v.size() == 2) {
      CHECK(v[0] == rank || v[1] == rank);
    } else if (v.size() == 3) {
      CHECK(v[0] != rank);
      CHECK(v[1] == rank || v[2] == rank);
    } else {
      CHECK(false);
    }
  }

  if (false) {
    fmt::printf("paths.size() is %d\n", paths.size());
    if (rank == 0 || true) {
      FILE* f = fopen(fmt::sprintf("paths-%d.txt", rank).c_str(), "wb");
      CHECK(f != nullptr);
      for (auto& path : paths) {
        fmt::fprintf(f, "[%s]\n", fmt::to_string(fmt::join(path, ", ")));
      }
      fclose(f);
    }
  }

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
      CHECK(path.size() == 2);
      CHECK(std::find(sendRanks.begin(), sendRanks.end(), destination) == sendRanks.end());
      sendRanks.push_back(destination);
    } else if (path.size() == 2 && destination == rank) {
      CHECK(std::find(recvRanks.begin(), recvRanks.end(), source) == recvRanks.end());
      recvRanks.push_back(source);
    } else if (proxy == rank) {
      CHECK(std::find(recvRanks.begin(), recvRanks.end(), source) != recvRanks.end());
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

  std::vector<size_t> rankNodes;
  rankNodes.resize(size);
  for (size_t i = 0; i != nodeRanks.size(); ++i) {
    for (size_t r : nodeRanks[i]) {
      rankNodes[r] = i;
    }
  }

  {
    std::vector<size_t> neighbors;
    for (auto& path : paths) {
      if (path.size() != 2) {
        continue;
      }
      if (path[0] == rank) {
        size_t sourceNode = rankNodes[path[0]];
        size_t destinationNode = rankNodes[path[1]];
        // if (abs(destinationNode - sourceNode, nodeRanks.size()) == 1) {
        if (destinationNode == (sourceNode + nodeRanks.size() - 1) % nodeRanks.size()) {
          neighbors.push_back(path[1]);
          log.info("%d is a neighbor\n", path[1]);
        }
      } else if (path[1] == rank) {
        size_t source = path[0];
        size_t bestDistance = size;
        size_t proxy = -1;
        for (auto& v : neighbors) {
          size_t d = abs(v - source, size);
          if (d < bestDistance) {
            bestDistance = d;
            proxy = v;
          }
        }
        ringRecvs.push_back({source, proxy});
      }
    }
    auto allRingRecvs = setupComms->allgather(ringRecvs);
    for (size_t i = 0;; ++i) {
      bool any = false;
      for (size_t r = 0; r != size; ++r) {
        auto& v = allRingRecvs.at(r);
        if (v.size() <= i) {
          continue;
        }
        any = true;
        if (std::get<1>(v[i]) == rank) {
          ringSends.push_back({std::get<0>(v[i]), r});
        }
      }
      if (!any) {
        break;
      }
    }
    std::string s;
    for (auto& v : ringSends) {
      if (!s.empty()) {
        s += ", ";
      }
      s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
    }
    log.info("rank %d: ring sends are [%s]\n", rank, s);
    s.clear();
    for (auto& v : ringRecvs) {
      if (!s.empty()) {
        s += ", ";
      }
      s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
    }
    log.info("rank %d: ring recvs are [%s]\n", rank, s);
    // for (auto& x : ringRecvs) {
    //   CHECK(std::get<2>(x) == (size_t)-1);
    // }
    // auto allRingSends = setupComms->allgather(ringSends);
    // for (size_t n : neighbors) {
    //   size_t index = 0;
    //   for (auto& t : allRingSends[n]) {
    //     if (std::get<1>(t) == rank) {
    //       for (auto& x : ringRecvs) {
    //         if (std::get<0>(t) == n == std::get<0>(x) && std::get<1>(x) == n) {
    //           CHECK(std::get<2>(x) == (size_t)-1);
    //           std::get<2>(x) = index;
    //         }
    //       }
    //     }
    //     ++index;
    //   }
    // }
    // for (auto& x : ringRecvs) {
    //   CHECK(std::get<2>(x) != (size_t)-1 && std::get<2>(x) < size);
    // }
  }

  // CHECK(proxyInfo.size() == proxyDestinationInfo.size());

  // auto proxyDestinationCounts = setupComms->allgather(proxyDestinationInfo.size());
  // for (size_t n : proxyDestinationCounts) {
  //   CHECK(n == proxyDestinationInfo.size());
  // }

  log.verbose("rank %d: ipc ranks are [%s]\n", rank, fmt::to_string(fmt::join(ipcRanks, ", ")));
  log.verbose("rank %d: send ranks are [%s]\n", rank, fmt::to_string(fmt::join(sendRanks, ", ")));
  log.verbose("rank %d: recv ranks are [%s]\n", rank, fmt::to_string(fmt::join(recvRanks, ", ")));

  std::string proxyForwardOrder;
  std::string proxyCopyOrder;

  for (auto& v : proxyInfo) {
    proxyForwardOrder += fmt::sprintf(" (%d %d)", v.source, v.destination);
  }
  for (auto& v : proxyDestinationInfo) {
    proxyCopyOrder += fmt::sprintf(" (%d %d)", v.source, v.proxy);
  }

  log.verbose("%d: proxy forward order %s\n", rank, proxyForwardOrder);
  log.verbose("%d: proxy copy order %s\n", rank, proxyCopyOrder);
}

std::string AllGather::generate() {

  // allgatherBlocksize = group->ipcRanks.size() == group->size - 1 ? 256 : 64;
  // allgatherGridsize = group->ipcRanks.size() == group->size - 1 ? smCount / 4 : smCount / 4;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;
  const auto& peerIndices = group->peerIndices;

  const auto& cudaStepValue = group->cudaStepValue;
  const auto& peerCudaStepValue = group->peerCudaStepValue;
  const auto& cudaCopyDone = group->cudaCopyDone;
  const auto& peerCudaCopyDone = group->peerCudaCopyDone;
  const auto& cudaProxyReady = group->cudaProxyReady;
  const auto& peerCudaProxyReady = group->peerCudaProxyReady;

  auto waitFor32 = [&](std::string address, std::string value) {
    return replace(
        R"(while (*(volatile uint32_t*)($ptr) < $value);
    )",
        "$ptr", address, "$value", value);
  };

  static const size_t numChunks = 8;

  auto waitForRecv = [&](size_t i, size_t chunkIndex, bool isLastChunk) {
    std::string s;
    s = replace("// wait for recv from $i, chunk $chunkIndex\n", "$i", i, "$chunkIndex", chunkIndex);
    for (size_t di = 0; di != group->ibDevs.size(); ++di) {
      if (isLastChunk) {
        s += waitFor32(concurrencyIndex(group->cudaCommsDeviceDataSent, sizeof(uint32_t) * (i * 32 + di)), "stepValue");
      } else {
        s += waitFor32(
            concurrencyIndex(group->cudaCommsDeviceDataSent2, sizeof(uint32_t) * (i * 32 * 32 + 32 * di + chunkIndex)),
            "stepValue");
      }
    }
    return s;
  };

  bool isInGrid = true;

  auto addCopy = [&](std::string dst, std::string src, std::string bytes) {
    if (isInGrid) {
      return replace("copy_impl($dst, $src, $bytes);\n", "$dst", dst, "$src", src, "$bytes", bytes);
    } else {
      CHECK(false);
      return replace("allgather_copy_add(copies, $dst, $src);\n", "$dst", dst, "$src", src);
    }
  };

  std::string localCopies;
  localCopies += addCopy(
      "(void*)(params.outputAddress + params.pitch * $rank)", "(const void*)params.inputAddress", "params.bytes");
  for (size_t peerIndex : peerIndices) {
    size_t i = ipcRanks[peerIndex];
    localCopies += addCopy(
        replace("(void*)(params.outputAddress + params.pitch * $i)", "$i", i),
        replace(
            "*(const void**)$src", "$src",
            concurrencyIndex(group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * peerIndex))),
        "params.bytes");
  }
  // isInGrid = false;

  std::string globaldefs;

  std::string recvCopies;
  HashMap<size_t, bool> hasWaitedForRecv;
  bool hasEstablishedFirstThread = false;
  CHECK(proxyInfo.size() == proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    auto& pi = proxyInfo[i];
    auto& pdi = proxyDestinationInfo[i];

    for (size_t chunkIndex = 0; chunkIndex != numChunks; ++chunkIndex) {
      // recvCopies += replace("{\n// chunk $chunkIndex\n", "$chunkIndex", chunkIndex);

      recvCopies += replace(
          replace(
              R"(
          {
            // chunk $chunkIndex
            const size_t currentChunkSize = $currentChunkSize;
            const size_t chunkOffset = chunkSize * $chunkIndex;
      )",
              "$currentChunkSize", "min(chunkSize, params.bytes - chunkSize * $chunkIndex)"),
          "$chunkIndex", chunkIndex);

      if (!hasWaitedForRecv[pi.source * numChunks + chunkIndex]) {
        // recvCopies += "allgather_copy_flush(copies);\n";
        hasWaitedForRecv[pi.source * numChunks + chunkIndex] = true;
        if (isInGrid) {
          recvCopies += "if (threadIdx.x == 0) {\n";
        }
        recvCopies += replace(
            R"(
          if (chunkOffset + currentChunkSize == params.bytes) {
            $waitlast
          } else {
            $waitnotlast
          }
          )",
            "$waitlast", waitForRecv(pi.source, chunkIndex, true), "$waitnotlast",
            waitForRecv(pi.source, chunkIndex, false));
        if (isInGrid) {
          recvCopies += "}\n";
        }

        if (!isInGrid) {
          CHECK(false);
          //   size_t ss = pi.source;
          //   for (auto& pi : proxyInfo) {
          //     if (pi.source != ss) {
          //       continue;
          //     }
          //     recvCopies += replace(
          //         R"(
          //   // forward recv $forwardSource to $forwardRank
          //   *(volatile uint32_t*)$forwardPtr = stepValue + $chunkIndex;
          // )",
          //         "$forwardPtr",
          //         concurrencyIndex(peerCudaProxyReady[pi.destinationPeerIndex], sizeof(uint32_t) * pi.source),
          //         "$forwardSource", pi.source, "$forwardRank", ipcRanks[pi.destinationPeerIndex], "$chunkIndex",
          //         chunkIndex);
          //   }
        } else {
          if (!hasEstablishedFirstThread) {
            hasEstablishedFirstThread = true;
            globaldefs += replace("__device__ uint32_t firstThreadCounter[$maxConcurrency];\n");
            recvCopies += replace(
                R"(
        if (threadIdx.x == 0) {
          if (atomicInc(&firstThreadCounter[concurrencyIndex], $gridSize - 1) == 0) {
            isFirstThread = true;
          }
        }
        syncthreads();
      )");
          }
          // recvCopies += "if (isFirstThread) {\n";
          // size_t ss = pi.source;
          // for (auto& pi : proxyInfo) {
          //   if (pi.source != ss) {
          //     continue;
          //   }
          //   recvCopies += replace(
          //       R"(
          //   // forward recv $forwardSource to $forwardRank
          //   //*(volatile uint32_t*)$forwardPtr = stepValue;
          // )",
          //       "$forwardPtr",
          //       concurrencyIndex(peerCudaProxyReady[pi.destinationPeerIndex], sizeof(uint32_t) * pi.source),
          //       "$forwardSource", pi.source, "$forwardRank", ipcRanks[pi.destinationPeerIndex]);
          // }
          // recvCopies += "}\n";
        }
      }

      if (isInGrid) {
        // globaldefs += replace("__device__ uint32_t dataReadyCounter_$n[$maxConcurrency];\n", "$n", i);
        recvCopies += replace(
            R"(
          __syncwarp();
          if (threadIdx.x == 0) {
            if (isFirstThread) {
              *(volatile uint32_t*)$forwardPtr = stepValue + $chunkIndex;
            }
            while (*(volatile uint32_t*)$readyPtr < stepValue + $chunkIndex);
          }
          syncthreads();
        )",
            "$n", i, "$forwardPtr",
            concurrencyIndex(peerCudaProxyReady[pi.destinationPeerIndex], sizeof(uint32_t) * pi.source), "$readyPtr",
            concurrencyIndex(cudaProxyReady, +sizeof(uint32_t) * pdi.source), "$chunkIndex", chunkIndex);
      } else {
        CHECK(false);
        // recvCopies += replace(
        //     R"(
        //   // forward recv $forwardSource to $forwardRank
        //   // *(volatile uint32_t*)$forwardPtr = stepValue;
        //   // wait for proxy recv $proxySource at $proxyRank
        //   while (*(volatile uint32_t*)$readyPtr < stepValue);
        // )",
        //     "$forwardPtr", concurrencyIndex(peerCudaProxyReady[pi.destinationPeerIndex], sizeof(uint32_t) *
        //     pi.source),
        //     "$readyPtr", concurrencyIndex(cudaProxyReady, sizeof(uint32_t) * pdi.source), "$forwardSource",
        //     pi.source,
        //     "$forwardRank", ipcRanks[pi.destinationPeerIndex], "$proxySource", pdi.source, "$proxyRank",
        //     ipcRanks[pdi.proxyPeerIndex]);
      }

      // recvCopies += replace(
      //     replace(
      //         R"(
      //     const size_t currentChunkSize = $currentChunkSize;
      //     const size_t chunkOffset = chunkSize * $chunkIndex;
      // )",
      //         "$currentChunkSize",
      //         chunkIndex == numChunks - 1 ? "params.bytes - chunkSize * $chunkIndex" : "chunkSize"),
      //     "$chunkIndex", chunkIndex);

      recvCopies += addCopy(
          replace("(void*)(params.outputAddress + params.pitch * $source + chunkOffset)", "$source", pdi.source),
          replace(
              "(void*)(*(uintptr_t*)$src + params.pitch * $source + chunkOffset)", "$source", pdi.source, "$src",
              concurrencyIndex(
                  group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * pdi.proxyPeerIndex + sizeof(uintptr_t)))),
          "currentChunkSize");

      if (chunkIndex != numChunks - 1) {
        recvCopies += "if (chunkSize * $chunkIndex + $currentChunkSize != params.bytes) {\n";
      } else {
        recvCopies += "assert(chunkSize * $chunkIndex + $currentChunkSize == params.bytes);\n";
      }

      recvCopies = replace(recvCopies, "$currentChunkSize", "min(chunkSize, params.bytes - chunkSize * $chunkIndex)");
      recvCopies = replace(recvCopies, "$chunkIndex", chunkIndex);
    }
    for (size_t chunkIndex = 0; chunkIndex != numChunks; ++chunkIndex) {
      recvCopies += "}\n";
      if (chunkIndex != numChunks - 1) {
        recvCopies += "}\n";
      }
    }
  }

  std::string source;

  source += R"(
namespace {

struct AllGatherParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  size_t bytes;
  size_t pitch;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  uintptr_t peerInputAddresses[8];
  uintptr_t peerOutputAddresses[8];
};

constexpr size_t copyQueueSize = 16;
struct AllGatherCopyParameters {
  size_t bytes;
  const void* src[copyQueueSize];
  void* dst[copyQueueSize];
  uint32_t n;
};

}

extern "C" __global__ void $launchBounds allgather_copy_kernel(AllGatherCopyParameters params) {
  assert(params.n > 0);
  assert(params.n <= copyQueueSize);
#pragma unroll 1
  for (uint32_t i = 0; i != params.n; ++i) {
    syncthreads();
    copy_impl(params.dst[i], params.src[i], params.bytes);
  }
}

namespace {

__device__ void allgather_copy_flush(AllGatherCopyParameters& params) {
  if (params.n) {
    // work around compiler bug
    AllGatherCopyParameters params2;
    memcpy(&params2, &params, sizeof(params2));
    allgather_copy_kernel<<<$gridSize, $blockSize>>>(params2);
    params.n = 0;
  }
}

__device__ void allgather_copy_add(AllGatherCopyParameters& params, void* dst, const void* src) {
  params.dst[params.n] = dst;
  params.src[params.n] = src;
  ++params.n;
  if (params.n == copyQueueSize) {
    allgather_copy_flush(params);
  }
}

}

  )";

  source += replace(
      R"zz(

$globaldefs

extern "C" __global__ void $launchBounds allgather_local(AllGatherParameters params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  grid_generic_entry_local(params);

  $localCopies

  grid_generic_exit_local(stepValue, concurrencyIndex);
}

extern "C" __global__ void allgather_exit(uint32_t stepValue, uint32_t concurrencyIndex) {
  generic_exit(stepValue, concurrencyIndex);
}

extern "C" __global__ void $launchBounds allgather(AllGatherParameters params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  grid_generic_entry(params);

  $localCopies

  // __threadfence_system();
  // syncthreads();
  // if (threadIdx.x != 0 || atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) != $gridSize - 1) {
  //   return;
  // }

  // AllGatherCopyParameters copies;
  // copies.bytes = params.bytes;
  // copies.n = 0;

  const size_t chunkSize = params.bytes < (size_t)65536 * 48 ? params.bytes : max(params.bytes / $numChunks, (size_t)65536 * 36);
  bool isFirstThread = false;

  $recvCopies

  grid_generic_exit(stepValue, concurrencyIndex);

  // allgather_copy_flush(copies);

  // allgather_exit<<<1, 1>>>(stepValue, concurrencyIndex);
}

  )zz",
      "$localCopies", localCopies, "$recvCopies", recvCopies, "$globaldefs", globaldefs);

  source = replace(source, "$numChunks", numChunks);

  return source;
}

} // namespace moodist
