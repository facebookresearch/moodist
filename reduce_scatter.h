#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct ReduceScatter : CollectiveBase {

  std::vector<size_t> sendRanks;
  std::vector<size_t> recvRanks;

  CUfunction cuReduceScatterEntry = nullptr;
  CUfunction cuReduceScatterExit = nullptr;
  CUfunction cuLocalReduceScatterEntry = nullptr;
  CUfunction cuLocalReduceScatterExit = nullptr;

  std::vector<CUfunction> cuSendReady;
  std::vector<CUfunction> cuWaitForRecv;

  // std::vector<std::vector<CUfunction>> cuAllgatherWaitForRecv{};
  // std::vector<std::vector<CUfunction>> cuAllgatherWaitForReady{};

  ReduceScatter(Group* group);
  ~ReduceScatter();
  void init();
  std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> generate();
};

} // namespace moodist
