// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "collective_base.h"
#include "common.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace moodist {

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
};

} // namespace moodist
