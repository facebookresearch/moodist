// Copyright (c) Meta Platforms, Inc. and affiliates.

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

  for (auto& pi : proxyInfo) {
    proxyInfoBySource[pi.source].push_back(&pi);
  }
  for (auto& pdi : proxyDestinationInfo) {
    CHECK(proxyDestinationInfoBySource.find(pdi.source) == proxyDestinationInfoBySource.end());
    proxyDestinationInfoBySource[pdi.source] = &pdi;
  }

  std::vector<size_t> rankNodes;
  rankNodes.resize(size);
  for (size_t i = 0; i != nodeRanks.size(); ++i) {
    for (size_t r : nodeRanks[i]) {
      rankNodes[r] = i;
    }
  }

  {
    auto allSendRanks = setupComms->allgather(sendRanks);
    for (size_t i : recvRanks) {
      auto& s = allSendRanks[i];
      auto it = std::find(s.begin(), s.end(), rank);
      CHECK(it != s.end());
      if (std::next(it) == s.end()) {
        recvRanksNext.push_back(-1);
      } else {
        recvRanksNext.push_back(*std::next(it));
      }
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
          log.debug("%d is a neighbor\n", path[1]);
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
    log.debug("rank %d: all gather ring sends are [%s]\n", rank, s);
    s.clear();
    for (auto& v : ringRecvs) {
      if (!s.empty()) {
        s += ", ";
      }
      s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
    }
    log.debug("rank %d: all gather ring recvs are [%s]\n", rank, s);

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

  for (auto [i, neighbor] : ringSends) {
    CHECK(ringSendsBySource.find(i) == ringSendsBySource.end());
    ringSendsBySource[i] = neighbor;
  }
  for (auto [i, neighbor] : ringRecvs) {
    CHECK(ringRecvsBySource.find(i) == ringRecvsBySource.end());
    ringRecvsBySource[i] = neighbor;
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
        R"(
          while (*(volatile uint32_t*)($ptr) < $value);
    )",
        "$ptr", address, "$value", value);
  };

  auto waitForRecv = [&](std::string i, size_t chunkIndex) {
    std::string s;
    s = replace("// wait for recv from $i, chunk $chunkIndex\n", "$i", i, "$chunkIndex", chunkIndex);
    s += waitFor32(replace("&cpuOut[$offset + $i]", "$i", i, "$offset", 16 + size * chunkIndex), "stepValue");
    return s;
  };

  auto addCopy = [&](std::string dst, std::string src, std::string bytes) {
    return replace(
        "dynamicBlockIndex = copy_impl(dynamicBlockIndex, $dst, $src, $bytes);\n", "$dst", dst, "$src", src, "$bytes",
        bytes);
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

  std::string globaldefs;

  std::string recvCopies;
  HashMap<size_t, bool> hasWaitedForRecv;
  CHECK(proxyInfo.size() == proxyDestinationInfo.size());

  Vector<Vector<std::pair<ProxyDestinationInfo*, ProxyInfo*>>> groupedProxyMap;

  size_t prevRecvSource = -1;
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    auto& pi = proxyInfo[i];
    auto& pdi = proxyDestinationInfo[i];
    if (pi.source != prevRecvSource) {
      prevRecvSource = pi.source;
      groupedProxyMap.emplace_back();
    }
    groupedProxyMap.back().emplace_back(&pdi, &pi);
  }

  std::vector<uint32_t> instructions;

  size_t numProxies = 0;
  if (!groupedProxyMap.empty()) {
    numProxies = groupedProxyMap[0].size();
  }

  for (auto& gg : groupedProxyMap) {
    CHECK(!gg.empty());
    size_t source = gg[0].second->source;
    CHECK(!hasWaitedForRecv[source]);
    hasWaitedForRecv[source] = true;
    instructions.push_back(source);
    CHECK(gg.size() == numProxies);
    for (auto& vv : gg) {
      auto& pi = *vv.second;
      CHECK(pi.source == source);
      auto& pdi = *vv.first;
      CHECK(pdi.proxyPeerIndex < 256);
      CHECK(pi.destinationPeerIndex < 256);
      CHECK(pdi.source < 65536);
      instructions.push_back((pdi.proxyPeerIndex << 24) | (pi.destinationPeerIndex << 16) | pdi.source);
    }
  }
  instructions.push_back(-1);

  std::vector<std::string> instructionsHex;
  for (auto& v : instructions) {
    instructionsHex.push_back(fmt::sprintf("%#x", v));
  }
  std::string instructionsArray = fmt::sprintf(
      "__constant__ uint32_t allGatherInstructions[%d] = {%s};", instructions.size(),
      fmt::to_string(fmt::join(instructionsHex, ", ")));

  for (size_t chunkIndex = 0; chunkIndex != maxChunks; ++chunkIndex) {
    recvCopies += R"(
        {
          // chunk $chunkIndex
          const size_t currentChunkSize = $currentChunkSize;
          const size_t chunkOffset = chunkSize * $chunkIndex;
      )";
    if (chunkIndex == 0) {
      recvCopies += "uint32_t* entryPtr = ptr;\n";
    } else {
      recvCopies += "ptr = entryPtr;\n";
    }
    for (size_t proxyIndex = 0; proxyIndex != numProxies; ++proxyIndex) {

      recvCopies += R"(
        {
        uint32_t value = *ptr++;
        uint32_t pdi_source = value & 0xffff;
        uint32_t destinationPeerIndex = (value >> 16) & 0xff;
        uint32_t proxyPeerIndex = (value >> 24) & 0xff;
      )";

      recvCopies += "if (threadIdx.x % 32 == 0) {\n";

      if (proxyIndex == 0) {
        recvCopies += waitForRecv("source", chunkIndex);
      }

      recvCopies += replace(
          R"(
        *(volatile uint32_t*)$forwardPtr = stepValue;
        while (*(volatile uint32_t*)$readyPtr != stepValue);
      )",
          "$forwardPtr",
          "(peerCudaProxyReady[destinationPeerIndex] + sizeof(uint32_t) * ($size * $maxChunks * concurrencyIndex + "
          "$size * $chunkIndex + source))",
          "$readyPtr",
          replace(
              "($base + sizeof(uint32_t) * ($size * $maxChunks * concurrencyIndex + $size * $chunkIndex + pdi_source))",
              "$base", cudaProxyReady.buffer.cudaPointer),
          "$chunkIndex", chunkIndex);

      recvCopies += "}\n__syncwarp();\n";

      recvCopies += addCopy(
          "(void*)(params.outputAddress + params.pitch * pdi_source + chunkOffset)",
          replace(
              "(void*)(*(uintptr_t*)$src + params.pitch * pdi_source + chunkOffset)", "$src",
              concurrencyIndex(
                  group->cudaPeerAddresses, "(sizeof(uintptr_t) * 2 * proxyPeerIndex + sizeof(uintptr_t))")),
          "currentChunkSize");

      recvCopies += "}\n";
    }
    if (chunkIndex != maxChunks - 1) {
      recvCopies += "if (chunkSize * $chunkIndex + $currentChunkSize != params.bytes) {\n";
      recvCopies += "assert(chunkSize * $chunkIndex + $currentChunkSize < params.bytes);\n";
    } else {
      recvCopies += "assert(chunkSize * $chunkIndex + $currentChunkSize == params.bytes);\n";
    }

    recvCopies = replace(recvCopies, "$currentChunkSize", "min(chunkSize, params.bytes - chunkSize * $chunkIndex)");
    recvCopies = replace(recvCopies, "$chunkIndex", chunkIndex);
  }
  for (size_t chunkIndex = 0; chunkIndex != maxChunks; ++chunkIndex) {
    recvCopies += "}\n";
    if (chunkIndex != maxChunks - 1) {
      recvCopies += "}\n";
    }
  }

  recvCopies = replace(
      R"(
    for (auto* ptr = &allGatherInstructions[0];;) {
      uint32_t source = *ptr++;
      if (source == -1) {
        break;
      }
      $recvCopies
    }
  )",
      "$recvCopies", recvCopies);

  std::string source;

  source += R"(
namespace {

struct AllGatherParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  size_t bytes;
  size_t pitch;
  size_t chunkSize;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  uintptr_t peerInputAddresses[8];
  uintptr_t peerOutputAddresses[8];
};

}
  )";

  source += replace(
      R"zz(

$globaldefs

extern "C" __global__ void allgather_no_local(AllGatherParameters params) {
  generic_entry(params);
  generic_exit(params.stepValue, params.concurrencyIndex);
}


extern "C" __global__ void $launchBounds allgather_local(AllGatherParameters params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  grid_generic_entry_local(params);

  uint32_t dynamicBlockIndex = $gridSize * (threadIdx.x / 32u) + blockIdx.x;

  $localCopies

  grid_generic_exit_local(stepValue, concurrencyIndex);
}

extern "C" __global__ void allgather_exit(uint32_t stepValue, uint32_t concurrencyIndex) {
  generic_exit(stepValue, concurrencyIndex);
}

$instructionsArray

extern "C" __global__ void $launchBounds allgather(AllGatherParameters params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  [[maybe_unused]] volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  grid_generic_entry(params);

  uint32_t dynamicBlockIndex = $gridSize * (threadIdx.x / 32u) + blockIdx.x;

  $localCopies

  const size_t chunkSize = params.chunkSize;

  $recvCopies

  grid_generic_exit(stepValue, concurrencyIndex);
}

  )zz",
      "$localCopies", localCopies, "$recvCopies", recvCopies, "$globaldefs", globaldefs, "$instructionsArray",
      instructionsArray);

  return source;
}

const IVector<std::pair<size_t, size_t>>& AllGather::shuffledReads(size_t numRecvs, size_t numChunks) const {
  std::lock_guard l(shuffledReadsCacheMutex);
  auto i = shuffledReadsCache.find(std::make_pair(numRecvs, numChunks));
  if (i != shuffledReadsCache.end()) [[likely]] {
    return i->second;
  }
  return NOINLINE_COLD({
    CHECK(numRecvs == recvRanks.size());
    Vector<std::pair<size_t, size_t>> rr;
    rr.reserve(numRecvs * numChunks);
    auto sortedRanks = recvRanks;
    sortedRanks.push_back(rank);
    std::sort(sortedRanks.begin(), sortedRanks.end());
    size_t myIndex = std::ranges::find(sortedRanks, rank) - sortedRanks.begin();
    for (size_t i : sortedRanks) {
      for (size_t chunkIndex : range(numChunks)) {
        rr.emplace_back(i, chunkIndex);
      }
    }
    std::minstd_rand rng(131071 + size * 8191 + numRecvs * 49 + numChunks);
    std::ranges::shuffle(rr, rng);
    std::ranges::stable_sort(rr, [](auto& a, auto& b) { return a.second < b.second; });
    IVector<std::pair<size_t, size_t>> r;
    for (auto& v : rr) {
      auto i = std::ranges::find(sortedRanks, v.first);
      CHECK(i != sortedRanks.end());
      size_t n = (i - sortedRanks.begin() + myIndex) % sortedRanks.size();
      if (sortedRanks[n] != rank) {
        auto i = std::ranges::find(recvRanks, sortedRanks[n]);
        CHECK(i != recvRanks.end());
        r.emplace_back(i - recvRanks.begin(), v.second);
      }
    }
    std::string s;
    for (auto& v : r) {
      if (!s.empty()) {
        s += ", ";
      }
      s += fmt::sprintf("(%d, %d)", v.first, v.second);
    }
    log.info("shuffled reads indices %d, %d -> [%s]\n", numRecvs, numChunks, s);
    rr.clear();
    for (auto& v : r) {
      rr.emplace_back(recvRanks.at(v.first), v.second);
    }
    s.clear();
    for (auto& v : rr) {
      if (!s.empty()) {
        s += ", ";
      }
      s += fmt::sprintf("(%d, %d)", v.first, v.second);
    }
    log.info("shuffled reads ranks %d, %d -> [%s]\n", numRecvs, numChunks, s);
    shuffledReadsCache[std::make_pair(numRecvs, numChunks)] = r;
    return shuffledReadsCache[std::make_pair(numRecvs, numChunks)];
  });
}

} // namespace moodist
