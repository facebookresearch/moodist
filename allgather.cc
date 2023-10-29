#include "allgather.h"
#include "common.h"
#include "group.h"
#include "ipc_mapper.h"
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
    size_t l = findLocalRank(src);
    for (auto& list : nodeRanks) {
      if (std::find(list.begin(), list.end(), src) != list.end()) {
        continue;
      }
      if (src == rank) {
        paths.push_back({src, list[l]});
      } else if (list[l] == rank) {
        for (size_t n = 0; n != list.size(); ++n) {
          if (n == l) {
            paths.push_back({src, list[n]});
          } else {
            paths.push_back({src, list[l], list[n]});
          }
        }
      } else if (std::find(list.begin(), list.end(), rank) != list.end()) {
        paths.push_back({src, list[l], rank});
      }
    }
  }

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

  fmt::printf("paths.size() is %d\n", paths.size());
  if (rank == 0 || true) {
    FILE* f = fopen(fmt::sprintf("paths-%d.txt", rank).c_str(), "wb");
    CHECK(f != nullptr);
    for (auto& path : paths) {
      fmt::fprintf(f, "[%s]\n", fmt::to_string(fmt::join(path, ", ")));
    }
    fclose(f);
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

  CHECK(proxyInfo.size() == proxyDestinationInfo.size());

  auto proxyDestinationCounts = setupComms->allgather(proxyDestinationInfo.size());
  for (size_t n : proxyDestinationCounts) {
    CHECK(n == proxyDestinationInfo.size());
  }

  fmt::printf("rank %d: ipc ranks are [%s]\n", rank, fmt::to_string(fmt::join(ipcRanks, ", ")));
  fmt::printf("rank %d: send ranks are [%s]\n", rank, fmt::to_string(fmt::join(sendRanks, ", ")));
  fmt::printf("rank %d: recv ranks are [%s]\n", rank, fmt::to_string(fmt::join(recvRanks, ", ")));

  std::string proxyForwardOrder;
  std::string proxyCopyOrder;

  for (auto& v : proxyInfo) {
    proxyForwardOrder += fmt::sprintf(" (%d %d)", v.source, v.destination);
  }
  for (auto& v : proxyDestinationInfo) {
    proxyCopyOrder += fmt::sprintf(" (%d %d)", v.source, v.proxy);
  }

  fmt::printf("%d: proxy forward order %s\n", rank, proxyForwardOrder);
  fmt::printf("%d: proxy copy order %s\n", rank, proxyCopyOrder);
}

std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> AllGather::generate() {

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

  auto waitFor32 = [&](uintptr_t address, std::string value) {
    return replace(
        R"(while (*(volatile uint32_t*)$ptr < $value);
    )",
        "$ptr", address, "$value", value);
  };

  auto waitForRecv = [&](size_t i, size_t chunkIndex) {
    std::string s;
    s = replace("// wait for recv from $i, chunk $chunkIndex\n", "$i", i, "$chunkIndex", chunkIndex);
    for (size_t di = 0; di != group->ibDevs.size(); ++di) {
      s += waitFor32(
          group->cudaCommsDeviceDataSent.cudaPointer + sizeof(uint32_t) * (i * 32 + di),
          replace("stepValue + $chunkIndex", "$chunkIndex", chunkIndex));
    }
    return s;
  };

  std::string localCopies;
  localCopies += R"(
    allgather_copy_add(copies, (void*)(params.outputAddress + params.bytes * $rank), (const void*)params.inputAddress);
  )";
  for (size_t peerIndex : peerIndices) {
    size_t i = ipcRanks[peerIndex];
    localCopies += replace(
        R"(
      allgather_copy_add(copies, (void*)(params.outputAddress + params.bytes * $i), *(const void**)$src);
    )",
        "$i", i, "$src", group->cudaPeerAddresses.cudaPointer + (sizeof(uintptr_t) * 2 * peerIndex));
  }

  std::string recvCopies;
  int prevRecvSource = -1;
  CHECK(proxyInfo.size() == proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    auto& pi = proxyInfo[i];
    auto& pdi = proxyDestinationInfo[i];

    if (prevRecvSource != pi.source) {
      recvCopies += waitForRecv(pi.source, Group::dataChunks - 1);
      prevRecvSource = pi.source;
    }

    recvCopies += replace(
        R"(
      *(volatile uint32_t*)$forwardPtr = stepValue + $c;
      while (*(volatile uint32_t*)$readyPtr < stepValue + $c);
      allgather_copy_add(copies, (void*)(params.outputAddress + params.bytes * $source), (void*)(*(uintptr_t*)$src + params.bytes * $source));
    )",
        "$forwardPtr", peerCudaProxyReady[pi.destinationPeerIndex] + sizeof(uint32_t) * pi.source, "$c",
        Group::dataChunks - 1, "$readyPtr", cudaProxyReady.cudaPointer + sizeof(uint32_t) * pdi.source, "$source",
        pdi.source, "$src",
        group->cudaPeerAddresses.cudaPointer + (sizeof(uintptr_t) * 2 * pdi.proxyPeerIndex + sizeof(uintptr_t)));
  }

  std::string source = replace(
      R"zz(

struct AllGatherParameters {
  size_t bytes;
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

__global__ void allgather_copy_kernel(AllGatherCopyParameters params) {
  assert(params.n > 0);
  assert(params.n <= copyQueueSize);
#pragma unroll 1
  for (uint32_t i = 0; i != params.n; ++i) {
    syncthreads();
    copy_impl(params.dst[i], params.src[i], params.bytes);
  }
}

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

extern "C" __global__ void allgather_local_exit() {
  generic_exit_local();
}

extern "C" __global__ void allgather_exit() {
  generic_exit();
}


extern "C" __global__ void allgather_local(AllGatherParameters params) {
  [[maybe_unused]] const uint32_t stepValue = globalStepValue;
  generic_entry_local(params);

  AllGatherCopyParameters copies;
  copies.bytes = params.bytes;
  copies.n = 0;

  $localCopies

  allgather_copy_flush(copies);

  allgather_local_exit<<<1, 1>>>();
}

extern "C" __global__ void allgather(AllGatherParameters params) {
  [[maybe_unused]] const uint32_t stepValue = globalStepValue;
  generic_entry(params);

  AllGatherCopyParameters copies;
  copies.bytes = params.bytes;
  copies.n = 0;

  $localCopies

  $recvCopies

  allgather_copy_flush(copies);

  allgather_exit<<<1, 1>>>();
}


  )zz",
      "$localCopies", localCopies, "$recvCopies", recvCopies);

  // if (rank == 0) {
  //   fmt::printf("code\n----\n%s\n", source);
  // }

  std::vector<std::pair<CUfunction*, std::string>> functions;

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  // fn(cuAllgatherEntry, "allgather_entry");
  // fn(cuAllgatherExit, "allgather_exit");

  // for (size_t i : group->peerIndices) {
  //   fn(cuAllgatherCopyDone[i], replace("allgather_copy_done_$i", "$i", i));
  // }
  // fn(cuAllgatherCopyAllDone, "allgather_copy_all_done");

  // cuAllgatherWaitForProxy.resize(proxyDestinationInfo.size());
  // for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
  //   cuAllgatherWaitForProxy[i].resize(Group::dataChunks);
  //   for (size_t c = 0; c != Group::dataChunks; ++c) {
  //     fn(cuAllgatherWaitForProxy[i][c], replace("allgather_wait_for_proxy_$i_$c", "$i", i, "$c", c));
  //   }
  // }

  // cuAllgatherWaitForRecvForward.resize(proxyDestinationInfo.size());
  // cuAllgatherWaitForReady.resize(proxyDestinationInfo.size());
  // for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
  //   cuAllgatherWaitForRecvForward[i].resize(Group::dataChunks);
  //   cuAllgatherWaitForReady[i].resize(Group::dataChunks);
  //   size_t recvSource = proxyInfo[i].source;
  //   for (size_t c = 0; c != Group::dataChunks; ++c) {
  //     fn(cuAllgatherWaitForRecvForward[i][c],
  //        replace("allgather_wait_for_recv_forward_$source_$c_$i", "$i", i, "$c", c, "$source", proxyInfo[i].source));
  //     fn(cuAllgatherWaitForReady[i][c],
  //        replace("allgather_wait_for_ready_$i_$c", "$i", proxyDestinationInfo[i].source, "$c", c));
  //   }
  // }

  fn(cuAllGatherLocal, "allgather_local");
  fn(cuAllGather, "allgather");

  return std::make_pair(source, functions);
}

} // namespace moodist
