#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct AllGatherParameters {
  size_t bytes;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;
};

struct AllGather : CollectiveBase {

  std::vector<size_t> sendRanks;
  std::vector<size_t> recvRanks;

  std::vector<ProxyInfo> proxyInfo;
  std::vector<ProxyDestinationInfo> proxyDestinationInfo;

  CUfunction cuAllgatherEntry = nullptr;
  CUfunction cuAllgatherExit = nullptr;
  std::array<CUfunction, 8> cuAllgatherCopyDone{};
  CUfunction cuAllgatherCopyAllDone = nullptr;
  std::vector<std::vector<CUfunction>> cuAllgatherWaitForProxy{};

  std::vector<std::vector<CUfunction>> cuAllgatherWaitForRecvForward{};
  std::vector<std::vector<CUfunction>> cuAllgatherWaitForReady{};

  CUfunction cuAllGatherLocal = nullptr;
  CUfunction cuAllGather = nullptr;

  AllGather(Group* group);
  ~AllGather();
  void init();
  std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> generate();
};

} // namespace moodist
