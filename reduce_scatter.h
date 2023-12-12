#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct ReduceScatterParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
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

  Vector<size_t> sendRanks;
  Vector<size_t> recvRanks;

  Vector<std::pair<size_t, size_t>> ringSends;
  Vector<std::pair<size_t, size_t>> ringRecvs;

  ReduceScatter(Group* group);
  ~ReduceScatter();
  void init();
  std::string generate(std::vector<std::string> generateTypes, std::vector<std::string> generateReductions);
};

} // namespace moodist
