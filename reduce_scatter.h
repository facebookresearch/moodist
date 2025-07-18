// Copyright (c) Meta Platforms, Inc. and affiliates.

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
  size_t chunkSize;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;
  uintptr_t sendAddress;
  uintptr_t recvAddress;
};

struct ReduceScatter : CollectiveBase {

  IVector<size_t> sendRanks;
  IVector<size_t> recvRanks;
  IVector<size_t> sendRemoteRecvIndex;

  IVector<std::pair<size_t, size_t>> ringSends;
  IVector<std::pair<size_t, size_t>> ringRecvs;

  ReduceScatter(Group* group);
  ~ReduceScatter();
  void init();
  std::string generate(std::vector<std::string> generateTypes, std::vector<std::string> generateReductions);
};

} // namespace moodist
