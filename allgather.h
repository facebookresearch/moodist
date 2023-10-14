#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {


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

  AllGather(Group* group);
  ~AllGather();
  void init();
  std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> generate();
};

} // namespace moodist
