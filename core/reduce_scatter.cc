// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "reduce_scatter.h"
#include "allgather.h"
#include "common.h"
#include "group.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include <vector>

namespace moodist {

ReduceScatter::ReduceScatter(Group* group) : CollectiveBase(group) {}

ReduceScatter::~ReduceScatter() {}

void ReduceScatter::init() {
  sendRanks = group->allGather->recvRanks;
  recvRanks = group->allGather->sendRanks;

  ringSends = group->allGather->ringSends;
  ringRecvs = group->allGather->ringRecvs;
  if (!ringSends.empty() && !ringRecvs.empty()) {
    std::rotate(ringSends.begin(), ringSends.begin() + 1, ringSends.end());
    std::rotate(ringRecvs.begin(), ringRecvs.begin() + 1, ringRecvs.end());
    ringSends.back().first = ringSends.back().second;
    ringRecvs.back().first = rank;
  }

  sendRemoteRecvIndex.resize(recvRanks.size());
  auto allRecvRanks = group->setupComms->allgather(recvRanks);
  for (size_t index : indices(sendRanks)) {
    auto& c = allRecvRanks.at(sendRanks[index]);
    auto it = std::ranges::find(c, rank);
    CHECK(it != c.end());
    sendRemoteRecvIndex[index] = it - c.begin();
  }

  std::string s;
  for (auto& v : ringSends) {
    if (!s.empty()) {
      s += ", ";
    }
    s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
  }
  log.debug("rank %d: reduce scatter ring sends are [%s]\n", rank, s);
  s.clear();
  for (auto& v : ringRecvs) {
    if (!s.empty()) {
      s += ", ";
    }
    s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
  }
  log.debug("rank %d: reduce scatter ring recvs are [%s]\n", rank, s);

  // CHECK(recvRanks.size() == sendRanks.size());
  // auto allSendRanks = group->setupComms->allgather(sendRanks);
  // for (auto& v : allSendRanks) {
  //   if (&v - allSendRanks.data() == rank) {
  //     continue;
  //   }
  //   CHECK(v.size() == sendRanks.size());
  //   for (size_t i = 0; i != sendRanks.size(); ++i) {
  //     CHECK(v[i] != sendRanks[i]);
  //   }
  // }
}

} // namespace moodist
