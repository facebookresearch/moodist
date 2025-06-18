// Copyright (c) Meta Platforms, Inc. and affiliates.

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
  size_t chunkSize;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  std::array<uintptr_t, 8> peerInputAddresses;
  std::array<uintptr_t, 8> peerOutputAddresses;
};

struct AllGather : CollectiveBase {

  IVector<size_t> sendRanks;
  IVector<size_t> recvRanks;

  IVector<ProxyInfo> proxyInfo;
  IVector<ProxyDestinationInfo> proxyDestinationInfo;

  HashMap<size_t, IVector<ProxyInfo*>> proxyInfoBySource;
  HashMap<size_t, ProxyDestinationInfo*> proxyDestinationInfoBySource;

  IVector<std::pair<size_t, size_t>> ringSends;
  IVector<std::pair<size_t, size_t>> ringRecvs;

  HashMap<size_t, size_t> ringSendsBySource;
  HashMap<size_t, size_t> ringRecvsBySource;

  IVector<size_t> recvRanksNext;

  AllGather(Group* group);
  ~AllGather();
  void init();
  std::string generate();

  mutable SpinMutex shuffledReadsCacheMutex;
  mutable HashMap<std::pair<size_t, size_t>, IVector<std::pair<size_t, size_t>>, PairHash> shuffledReadsCache;
  const IVector<std::pair<size_t, size_t>>& shuffledReads(size_t numRecvs, size_t numChunks) const;
};

} // namespace moodist
