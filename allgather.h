#pragma once

#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct ProxyInfo {
  size_t source;
  size_t destination;
  size_t destinationPeerIndex;
};

struct ProxyDestinationInfo {
  size_t source;
  size_t proxy;
  size_t proxyPeerIndex;
};

struct AllGatherComms {
  uint32_t stepValue;
  size_t bytesReady;
};

struct Group;

struct AllGather {

  Group* group = nullptr;

  std::vector<size_t> sendRanks;
  std::vector<size_t> recvRanks;

  std::vector<ProxyInfo> proxyInfo;
  std::vector<ProxyDestinationInfo> proxyDestinationInfo;

  AllocatedBuffer commsBuffer;

  // std::vector<std::vector<std::pair<size_t, size_t>>> peerProxies;
  // std::vector<std::pair<size_t, size_t>> ipcProxies;

  // std::vector<uint32_t> rankForwardOrder;
  // std::vector<uint32_t> recvRanks;

  // uint32_t proxyDestinationCount = 0;
  // std::vector<ProxyDestinationInfo> proxyDestinationInfo;

  // std::vector<std::vector<size_t>> recvProxiesPeerIndices;

  AllGather(Group* group);
  ~AllGather();
  void init();
};

} // namespace moodist
