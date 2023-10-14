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

  std::string writes;
  std::string waits;
  for (size_t i : group->peerIndices) {
    waits += waitFor32(cudaStepValue.cudaPointer + sizeof(uint32_t) * group->ipcRanks[i], "stepValue");
    writes += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", peerCudaStepValue[i] + sizeof(uint32_t) * rank);
  }

  std::string copyDoneFunctions;
  std::string copyDoneAllCode;
  std::string waitForCopyDones;
  for (size_t i : group->peerIndices) {
    copyDoneFunctions += replace(
        R"(
extern "C" __global__ void allgather_copy_done_$i() {
  uint32_t stepValue = globalStepValue;
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

  std::string waitForRecvs;
  for (size_t i : recvRanks) {
    waitForRecvs += waitForRecv(i, Group::dataChunks - 1);
  }

  std::string waitForProxyFunctions;
  size_t prevRecvSource = -1;
  CHECK(proxyInfo.size() == proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    auto& pi = proxyInfo[i];
    auto& pdi = proxyDestinationInfo[i];
    for (size_t c = 0; c != Group::dataChunks; ++c) {
      waitForProxyFunctions += replace(
          R"(
  extern "C" __global__ void allgather_wait_for_proxy_$i_$c() {
    uint32_t stepValue = globalStepValue;
    $wait
    // forward source $forwardSource destination $forwardDestination (peer index $forwardDestinationPeerIndex)
    *(volatile uint32_t*)$forwardPtr = stepValue + $c;
    // wait for ready source $proxySource proxy $proxyProxy (peer index $proxyProxyPeerIndex)
    while (*(volatile uint32_t*)$readyPtr < stepValue + $c);
  }

  extern "C" __global__ void allgather_wait_for_recv_forward_$forwardSource_$c_$i() {
    uint32_t stepValue = globalStepValue;
    $wait
    // forward source $forwardSource destination $forwardDestination (peer index $forwardDestinationPeerIndex)
    *(volatile uint32_t*)$forwardPtr = stepValue + $c;
  }
  extern "C" __global__ void allgather_wait_for_ready_$proxySource_$c() {
    uint32_t stepValue = globalStepValue;
    // wait for ready source $proxySource proxy $proxyProxy (peer index $proxyProxyPeerIndex)
    while (*(volatile uint32_t*)$readyPtr < stepValue + $c);
  }
      )",
          "$i", i, "$c", c, "$readyPtr", cudaProxyReady.cudaPointer + sizeof(uint32_t) * pdi.source, "$wait",
          pi.source == prevRecvSource ? "// same recv source as previous" : waitForRecv(pi.source, c), "$forwardPtr",
          peerCudaProxyReady[pi.destinationPeerIndex] + sizeof(uint32_t) * pi.source, "$proxySource", pdi.source,
          "$proxyProxy", pdi.proxy, "$proxyProxyPeerIndex", pdi.proxyPeerIndex, "$forwardSource", pi.source,
          "$forwardDestination", pi.destination, "$forwardDestinationPeerIndex", pi.destinationPeerIndex);
    }
    prevRecvSource = pi.source;
  }

  std::string source = replace(
      R"zz(

extern "C" __global__ void allgather_entry() {
  uint32_t stepValue = globalStepValue;
  // printf("enter %#x\n", stepValue);
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  cpuIn[0] = stepValue;
  $writes
  $waits
}

extern "C" __global__ void allgather_exit() {
  uint32_t stepValue = globalStepValue;
  $copyDoneAllCode
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  while (cpuOut[0] < stepValue);
  $waitForCopyDones
  $waitForRecvs

  globalStepValue += 256;
}

$copyDoneFunctions

extern "C" __global__ void allgather_copy_all_done() {
  uint32_t stepValue = globalStepValue;
  $copyDoneAllCode
}

$waitForProxyFunctions

  )zz",
      "$writes", writes, "$waits", waits, "$copyDoneFunctions", copyDoneFunctions, "$waitForCopyDones",
      waitForCopyDones, "$waitForRecvs", waitForRecvs, "$copyDoneAllCode", copyDoneAllCode, "$waitForProxyFunctions",
      waitForProxyFunctions);

  // if (rank == 0) {
  //   fmt::printf("code\n----\n%s\n", source);
  // }

  std::vector<std::pair<CUfunction*, std::string>> functions;

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  fn(cuAllgatherEntry, "allgather_entry");
  fn(cuAllgatherExit, "allgather_exit");

  for (size_t i : group->peerIndices) {
    fn(cuAllgatherCopyDone[i], replace("allgather_copy_done_$i", "$i", i));
  }
  fn(cuAllgatherCopyAllDone, "allgather_copy_all_done");

  cuAllgatherWaitForProxy.resize(proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    cuAllgatherWaitForProxy[i].resize(Group::dataChunks);
    for (size_t c = 0; c != Group::dataChunks; ++c) {
      fn(cuAllgatherWaitForProxy[i][c], replace("allgather_wait_for_proxy_$i_$c", "$i", i, "$c", c));
    }
  }

  cuAllgatherWaitForRecvForward.resize(proxyDestinationInfo.size());
  cuAllgatherWaitForReady.resize(proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    cuAllgatherWaitForRecvForward[i].resize(Group::dataChunks);
    cuAllgatherWaitForReady[i].resize(Group::dataChunks);
    size_t recvSource = proxyInfo[i].source;
    for (size_t c = 0; c != Group::dataChunks; ++c) {
      fn(cuAllgatherWaitForRecvForward[i][c],
         replace("allgather_wait_for_recv_forward_$source_$c_$i", "$i", i, "$c", c, "$source", proxyInfo[i].source));
      fn(cuAllgatherWaitForReady[i][c],
         replace("allgather_wait_for_ready_$i_$c", "$i", proxyDestinationInfo[i].source, "$c", c));
    }
  }

  return std::make_pair(source, functions);
}

} // namespace moodist
