// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

struct ReduceScatter : CollectiveBase {

  IVector<size_t> sendRanks;
  IVector<size_t> recvRanks;
  IVector<size_t> sendRemoteRecvIndex;

  IVector<std::pair<size_t, size_t>> ringSends;
  IVector<std::pair<size_t, size_t>> ringRecvs;

  ReduceScatter(Group* group);
  ~ReduceScatter();
  void init();
};

} // namespace moodist
