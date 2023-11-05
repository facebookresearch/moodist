#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct AllGatherParameters {
  size_t bytes;
  size_t pitch;
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

  size_t blockSizeLocal = 0;
  size_t gridSizeLocal = 0;
  size_t blockSize = 0;
  size_t gridSize = 0;

  CUfunction cuAllGatherLocal = nullptr;
  CUfunction cuAllGather = nullptr;

  CUfunction cuAllGatherCopyKernel = nullptr;

  AllGather(Group* group);
  ~AllGather();
  void init();
  std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> generate();
};

} // namespace moodist
