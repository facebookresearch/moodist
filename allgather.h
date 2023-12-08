#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct AllGatherParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  size_t bytes;
  size_t pitch;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;
};

struct AllGather : CollectiveBase {

  Vector<size_t> sendRanks;
  Vector<size_t> recvRanks;

  Vector<ProxyInfo> proxyInfo;
  Vector<ProxyDestinationInfo> proxyDestinationInfo;

  Vector<std::pair<size_t, size_t>> ringSends;
  Vector<std::pair<size_t, size_t>> ringRecvs;

  AllGather(Group* group);
  ~AllGather();
  void init();
  std::string generate();
};

} // namespace moodist
