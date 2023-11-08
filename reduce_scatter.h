#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct ReduceScatterParameters {
  uint32_t stepValue;
  size_t bytes;
  size_t pitch;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;
  uintptr_t sendAddress;
  uintptr_t recvAddress;
};

struct ReduceScatter : CollectiveBase {

  std::vector<size_t> sendRanks;
  std::vector<size_t> recvRanks;

  ReduceScatter(Group* group);
  ~ReduceScatter();
  void init();
  std::string generate();
};

} // namespace moodist
