#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct ReduceScatterParameters {
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

  CUfunction cuReduceScatterEntry = nullptr;
  CUfunction cuReduceScatterExit = nullptr;
  CUfunction cuLocalReduceScatterEntry = nullptr;
  CUfunction cuLocalReduceScatterExit = nullptr;

  std::vector<CUfunction> cuSendReady;
  std::vector<CUfunction> cuWaitForRecv;

  CUfunction cuReduceScatterLocal = nullptr;
  CUfunction cuReduceScatter = nullptr;

  // std::vector<std::vector<CUfunction>> cuAllgatherWaitForRecv{};
  // std::vector<std::vector<CUfunction>> cuAllgatherWaitForReady{};

  ReduceScatter(Group* group);
  ~ReduceScatter();
  void init();
  std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> generate();
};

} // namespace moodist
