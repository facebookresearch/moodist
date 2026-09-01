// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ib_common.h"
#include "group.h"
#include "setup_comms.h"

#include "providers/efa/efadv.h"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/fcntl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

extern "C" bool is_efa_dev(ibv_device* device);

namespace moodist {

IbCommon::IbCommon(Group* group) : group(group) {}

IbCommon::~IbCommon() {}

// Parse an integer environment variable, validating that it lies within [minValue, maxValue].
// Returns fallback if the variable is unset, empty, or invalid.
static int intEnv(const char* name, int fallback, int minValue, int maxValue) {
  const char* c = std::getenv(name);
  if (!c || !*c) {
    return fallback;
  }
  char* end = nullptr;
  long value = std::strtol(c, &end, 10);
  if (end == c || *end != '\0' || value < minValue || value > maxValue) {
    log.error("Ignoring invalid value '%s' for %s (expected an integer in [%d, %d])\n", c, name, minValue, maxValue);
    return fallback;
  }
  return (int)value;
}

// Select the best GID index for the given port.
// Preference order: IB > RoCE v2 > RoCE v1. Among GIDs of the best type, the last one is
// preferred (tuned for IPv6-only RoCE fabrics). Can be overridden with MOODIST_IB_GID_INDEX.
// Returns the GID index and the resolved GID value.
std::pair<uint32_t, ibv_gid> selectGid(ibv_context* context, int portNum, bool quiet) {
  auto gidTypeName = [](uint32_t gid_type) -> const char* {
    switch (gid_type) {
    case IBV_GID_TYPE_IB:
      return "IB";
    case IBV_GID_TYPE_ROCE_V1:
      return "RoCE v1";
    case IBV_GID_TYPE_ROCE_V2:
      return "RoCE v2";
    default:
      return "unknown";
    }
  };

  // Explicit override via environment variable.
  int gidOverride = intEnv("MOODIST_IB_GID_INDEX", -1, 0, 255);
  if (gidOverride >= 0) {
    ibv_gid_entry entry;
    int error = ibv_query_gid_ex(context, portNum, (uint32_t)gidOverride, &entry, 0);
    ibv_gid zero{};
    if (error || std::memcmp(&entry.gid, &zero, sizeof(zero)) == 0) {
      log.error("MOODIST_IB_GID_INDEX=%d is not a valid GID on port %d; falling back to automatic selection\n",
          gidOverride, portNum);
    } else {
      if (!quiet) {
        log.info("selectGid: using GID index %d (%s) [MOODIST_IB_GID_INDEX override]\n", gidOverride,
            gidTypeName(entry.gid_type));
      }
      return {(uint32_t)gidOverride, entry.gid};
    }
  }

  auto gidTypePriority = [](uint32_t gid_type) -> int {
    switch (gid_type) {
    case IBV_GID_TYPE_IB:
      return 0;
    case IBV_GID_TYPE_ROCE_V2:
      return 1;
    case IBV_GID_TYPE_ROCE_V1:
      return 2;
    default:
      return 3;
    }
  };

  uint32_t bestIndex = 0;
  ibv_gid bestGid{};
  uint32_t bestGidType = 0;
  int bestPriority = 4; // worse than any valid type
  bool found = false;
  for (uint32_t i = 0; i < 256; ++i) {
    ibv_gid_entry entry;
    int error = ibv_query_gid_ex(context, portNum, i, &entry, 0);
    if (error) {
      continue;
    }
    // Skip empty GIDs
    ibv_gid zero{};
    if (std::memcmp(&entry.gid, &zero, sizeof(zero)) == 0) {
      continue;
    }
    log.debug("gid %d is %s of type %d    -- %d %d %d\n", i, hexstr(&entry.gid, 16), entry.gid_type, entry.gid_index,
        entry.port_num, entry.ndev_ifindex);
    int priority = gidTypePriority(entry.gid_type);
    // Prefer the last GID of the best priority (tuned for IPv6-only RoCE fabrics).
    if (priority <= bestPriority) {
      bestPriority = priority;
      bestIndex = i;
      bestGid = entry.gid;
      bestGidType = entry.gid_type;
      found = true;
    }
  }

  if (!found) {
    // Fallback: use index 0 with ibv_query_gid (original behavior)
    log.error("selectGid: no valid GID found via ibv_query_gid_ex, falling back to index 0\n");
    int error = ibv_query_gid(context, portNum, 0, &bestGid);
    if (error) {
      throwErrno(errno, "ibv_query_gid");
    }
  } else if (!quiet) {
    log.info("selectGid: using GID index %d (%s)\n", bestIndex, gidTypeName(bestGidType));
  }

  return {bestIndex, bestGid};
}

namespace {
// Wire format for the rail probe address exchange. Flattened (no nested serialize) and made
// of trivially-copyable fields only. `valid` is 0 when the sending side has no usable QP.
struct ProbeAddress {
  uint16_t lid = 0;
  uint32_t qpNum = 0;
  int mtuIndex = 0;
  ibv_gid gid{};
  uint32_t rkey = 0;
  uint64_t addr = 0;
  uint8_t valid = 0;
  template<typename X>
  void serialize(X& x) {
    x(lid, qpNum, mtuIndex, gid, rkey, addr, valid);
  }
};
} // namespace

std::vector<uint8_t> probeIbRailBatch(
    Group* group, const std::vector<ProbeEndpoint>& local, size_t peerRank, bool initiator) {
  SetupComms* sc = group->setupComms.get();
  const size_t n = local.size();

  const int trafficClass = intEnv("MOODIST_IB_TC", 0, 0, 255);
  const int serviceLevel = intEnv("MOODIST_IB_SL", 0, 0, 15);
  // Upper bound on how long we wait for a probe write to complete. A same-rail write completes
  // in microseconds and returns immediately; only cross-rail (black-holed) writes ever approach
  // this bound, and only if the HCA never reports RETRY_EXC. Kept well above any realistic
  // same-rail latency so it never false-negatives a real link.
  const int timeoutMs = intEnv("MOODIST_IB_RAIL_PROBE_TIMEOUT_MS", 500, 1, 60000);

  // Per-slot temporary RDMA resources. Constructed once (never reallocated), so the MR always
  // points at a stable buffer address; RAII tears everything down on return.
  struct Slot {
    IbvPtr<ibv_pd, ibv_dealloc_pd> pd;
    IbvCq cq;
    IbvMr mr;
    IbvQp qp;
    uint64_t buffer = 0x6d6f6f6469737400ull; // "moodist\0" — arbitrary probe payload
    bool localValid = false;
  };
  std::vector<Slot> slots(n);
  std::vector<ProbeAddress> localAddrs(n);

  // Bring each local slot up to INIT (RESET -> INIT). Any verbs failure just leaves the slot
  // invalid (valid == 0 in the exchanged address); it never aborts the batch.
  for (size_t j = 0; j != n; ++j) {
    const ProbeEndpoint& ep = local[j];
    if (ep.context == nullptr) {
      continue;
    }
    Slot& s = slots[j];
    ibv_pd* p = ibv_alloc_pd(ep.context);
    if (!p) {
      continue;
    }
    s.pd = IbvPtr<ibv_pd, ibv_dealloc_pd>(p);
    ibv_cq* c = ibv_create_cq(ep.context, 4, nullptr, nullptr, 0);
    if (!c) {
      continue;
    }
    s.cq = IbvCq(c);
    ibv_mr* m = ibv_reg_mr(
        s.pd, &s.buffer, sizeof(s.buffer), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!m) {
      continue;
    }
    s.mr = IbvMr(m);
    ibv_qp_init_attr_ex ia;
    std::memset(&ia, 0, sizeof(ia));
    ia.qp_type = IBV_QPT_RC;
    ia.send_cq = s.cq;
    ia.recv_cq = s.cq;
    ia.cap.max_send_wr = 4;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_wr = 4;
    ia.cap.max_recv_sge = 1;
    ia.cap.max_inline_data = 0;
    ia.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    ia.pd = s.pd;
    ia.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
    ibv_qp* q = ibv_create_qp_ex(ep.context, &ia);
    if (!q) {
      continue;
    }
    s.qp = IbvQp(q);
    ibv_qp_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = ep.portNum;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(s.qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
      continue;
    }
    s.localValid = true;
    localAddrs[j].lid = ep.portAttributes.lid;
    localAddrs[j].qpNum = s.qp->qp_num;
    localAddrs[j].mtuIndex = ep.portAttributes.active_mtu;
    localAddrs[j].gid = ep.gid;
    localAddrs[j].rkey = s.mr->rkey;
    localAddrs[j].addr = (uint64_t)(uintptr_t)&s.buffer;
    localAddrs[j].valid = 1;
  }

  // Fixed exchange #1: swap the whole address vector. Always exactly one send + one recv,
  // whatever the slot count (including zero) or per-slot validity, so both sides stay in
  // lockstep.
  sc->sendTo(peerRank, localAddrs);
  std::vector<ProbeAddress> remoteAddrs = sc->recvFrom<std::vector<ProbeAddress>>(peerRank);
  remoteAddrs.resize(n); // defensive: both sides use the same n

  // Connect each slot whose endpoints are valid on both sides (INIT -> RTR -> RTS).
  std::vector<uint8_t> bothValid(n, 0);
  for (size_t j = 0; j != n; ++j) {
    Slot& s = slots[j];
    if (!s.localValid || remoteAddrs[j].valid == 0) {
      continue;
    }
    const ProbeEndpoint& ep = local[j];
    ibv_qp_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.ah_attr.grh.dgid = remoteAddrs[j].gid;
    attr.ah_attr.grh.sgid_index = ep.gidIndex;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = trafficClass;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.sl = serviceLevel;
    attr.ah_attr.port_num = ep.portNum;
    attr.ah_attr.dlid = remoteAddrs[j].lid;
    attr.path_mtu = (ibv_mtu)std::min<int>(ep.portAttributes.active_mtu, remoteAddrs[j].mtuIndex);
    attr.dest_qp_num = remoteAddrs[j].qpNum;
    attr.rq_psn = 4979;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    if (ibv_modify_qp(s.qp, &attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER) != 0) {
      continue;
    }
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 4979;
    // Low retry so a cross-rail (black-holed) write reports RETRY_EXC quickly instead of
    // stalling the wave; ~4 tries * (4.096us * 2^12 ~= 17ms) => ~67ms to RETRY_EXC.
    attr.timeout = 12;
    attr.retry_cnt = 3;
    attr.rnr_retry = 3;
    attr.max_rd_atomic = 1;
    if (ibv_modify_qp(s.qp, &attr,
            IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
      continue;
    }
    bothValid[j] = 1;
  }

  // Fixed exchange #2: swap per-slot readiness. The initiator must not write to a slot until the
  // responder's QP for it is connected and ready.
  sc->sendTo(peerRank, bothValid);
  std::vector<uint8_t> remoteReady = sc->recvFrom<std::vector<uint8_t>>(peerRank);
  remoteReady.resize(n);
  std::vector<uint8_t> ready(n, 0);
  for (size_t j = 0; j != n; ++j) {
    ready[j] = (bothValid[j] && remoteReady[j]) ? 1 : 0;
  }

  std::vector<uint8_t> result(n, 0);
  if (initiator) {
    // Post every ready slot's write, then poll all of them together, early-exiting as soon as
    // each posted write has a completion (success or error). Only slots waiting on a black-holed
    // write ever run to the deadline, and they all wait in parallel.
    std::vector<uint8_t> done(n, 0);
    size_t remaining = 0;
    for (size_t j = 0; j != n; ++j) {
      if (!ready[j]) {
        continue;
      }
      Slot& s = slots[j];
      ibv_qp_ex* qpx = ibv_qp_to_qp_ex(s.qp);
      ibv_wr_start(qpx);
      qpx->wr_id = j;
      qpx->wr_flags = IBV_SEND_SIGNALED;
      ibv_wr_rdma_write(qpx, remoteAddrs[j].rkey, remoteAddrs[j].addr);
      ibv_wr_set_sge(qpx, s.mr->lkey, (uintptr_t)&s.buffer, sizeof(s.buffer));
      if (ibv_wr_complete(qpx) == 0) {
        ++remaining;
      } else {
        done[j] = 1; // could not post; leaves result 0
      }
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (remaining != 0 && std::chrono::steady_clock::now() < deadline) {
      bool progressed = false;
      for (size_t j = 0; j != n; ++j) {
        if (!ready[j] || done[j]) {
          continue;
        }
        ibv_wc wc;
        std::memset(&wc, 0, sizeof(wc));
        int m = ibv_poll_cq(slots[j].cq, 1, &wc);
        if (m < 0) {
          done[j] = 1;
          --remaining;
          progressed = true;
        } else if (m > 0) {
          result[j] = (wc.status == IBV_WC_SUCCESS) ? 1 : 0;
          done[j] = 1;
          --remaining;
          progressed = true;
        }
      }
      if (remaining != 0 && !progressed) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
    // Fixed exchange #3: tell the responder the verdicts so both sides agree, and return them.
    sc->sendTo(peerRank, result);
  } else {
    result = sc->recvFrom<std::vector<uint8_t>>(peerRank);
    result.resize(n);
  }
  return result;
}

void IbCommon::init2Ib(int portNum, ibv_port_attr portAttributes) {
  const size_t rank = group->rank;
  const size_t size = group->size;

  log.debug("Init IB port %d\n", portNum);

  auto [gidIndex, gid] = selectGid(context, portNum);

  int trafficClass = intEnv("MOODIST_IB_TC", 0, 0, 255);
  int serviceLevel = intEnv("MOODIST_IB_SL", 0, 0, 15);
  if (trafficClass != 0 || serviceLevel != 0) {
    log.info("Using RoCE traffic_class %d, service level %d\n", trafficClass, serviceLevel);
  }

  IbAddress loopbackAddress;

  recvChannel = ibv_create_comp_channel(context);
  if (!recvChannel) {
    throwErrno(errno, "ibv_create_comp_channel");
  }

  recvCq = ibv_create_cq(context, maxSrqEntries, nullptr, recvChannel, 0);
  if (!recvCq) {
    throwErrno(errno, "ibv_create_cq");
  }

  ibv_srq_init_attr srqAttrs;
  std::memset(&srqAttrs, 0, sizeof(srqAttrs));
  srqAttrs.attr.max_sge = 1;
  srqAttrs.attr.max_wr = maxSrqEntries;
  sharedReceiveQueue = ibv_create_srq(protectionDomain, &srqAttrs);
  if (!sharedReceiveQueue) {
    throwErrno(errno, "ibv_create_srq");
  }

  inlineBytes = 64;

  for (size_t i = 0; i != size; ++i) {
    if (i == rank && false) {
      qps.emplace_back();
      qpexs.emplace_back();
    } else {
      ibv_qp_init_attr_ex initAttributes;
      std::memset(&initAttributes, 0, sizeof(initAttributes));
      initAttributes.qp_type = IBV_QPT_RC;
      initAttributes.send_cq = cq;
      initAttributes.recv_cq = recvCq;
      initAttributes.cap.max_send_wr = maxWr;
      initAttributes.cap.max_send_sge = 1;
      initAttributes.cap.max_recv_wr = 1;
      initAttributes.cap.max_recv_sge = 1;
      initAttributes.srq = sharedReceiveQueue;
      initAttributes.sq_sig_all = 0;
      initAttributes.cap.max_inline_data = inlineBytes;
      initAttributes.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
      initAttributes.pd = protectionDomain;
      initAttributes.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_READ | IBV_QP_EX_WITH_SEND;

      IbvQp qp = ibv_create_qp_ex(context, &initAttributes);
      if (!qp) {
        throwErrno(errno, "ibv_create_qp_ex");
      }
      ibv_qp_attr attr;
      memset(&attr, 0, sizeof(attr));
      attr.qp_state = IBV_QPS_RESET;
      int error = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
      if (error) {
        throwErrno(error, fmt::sprintf("ibv_modify_qp ->RESET (peer rank %zu, port %d)", i, portNum).c_str());
      }

      memset(&attr, 0, sizeof(attr));
      attr.qp_state = IBV_QPS_INIT;
      attr.pkey_index = 0;
      attr.port_num = portNum;
      attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
      error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
      if (error) {
        throwErrno(error, fmt::sprintf("ibv_modify_qp ->INIT (peer rank %zu, port %d)", i, portNum).c_str());
      }

      IbAddress address;
      address.lid = portAttributes.lid;
      address.qpNum = qp->qp_num;
      address.mtuIndex = portAttributes.active_mtu;
      address.gid = gid;

      if (i != rank) {
        group->setupComms->sendTo(i, address);
      } else {
        loopbackAddress = address;
      }

      qpexs.push_back(ibv_qp_to_qp_ex(qp));
      qps.push_back(std::move(qp));
    }
  }

  // RTR triggers RoCE neighbor-discovery (GID -> L2 address) for each peer. Under a synchronized
  // connection storm at scale this can transiently time out, so RTR is retried below with
  // full-jitter backoff (a failed RTR leaves the QP in INIT, so re-issuing it is valid). Only
  // ETIMEDOUT is retried; bounded by MOODIST_IB_RTR_MAX_RETRIES.
  const int maxRtrRetries = intEnv("MOODIST_IB_RTR_MAX_RETRIES", 12, 0, 1000);
  const int rtrMask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  // Stagger the connection order per rank: at step j, rank r connects to peer (r+1+j), so every
  // peer is targeted by exactly one rank at a time instead of all ranks resolving peer 0 (then
  // 1, ...) simultaneously. This spreads the ND load and avoids a thundering herd on the
  // low-numbered ranks during QP bring-up at scale.
  for (size_t j = 0; j != size; ++j) {
    const size_t i = (rank + 1 + j) % size;
    auto& qp = qps.at(i);

    auto remoteAddress = i == rank ? loopbackAddress : group->setupComms->recvFrom<IbAddress>(i);

    ibv_qp_attr attr;
    ibv_qp_init_attr initAttr;
    int error = ibv_query_qp(qp, &attr, IBV_QP_STATE, &initAttr);
    CHECK(attr.qp_state == IBV_QPS_INIT);
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    std::memset(&attr.ah_attr, 0, sizeof(attr.ah_attr));
    log.debug("REMOTE gid for index %d is %s\n", gidIndex, hexstr(&remoteAddress.gid, 16));
    attr.ah_attr.grh.dgid = remoteAddress.gid;
    attr.ah_attr.grh.sgid_index = gidIndex;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = trafficClass;
    attr.ah_attr.is_global = true;
    attr.ah_attr.sl = serviceLevel;
    attr.ah_attr.port_num = portNum;
    attr.ah_attr.dlid = remoteAddress.lid;
    attr.path_mtu = portAttributes.active_mtu;
    attr.dest_qp_num = remoteAddress.qpNum;
    attr.rq_psn = 4979;
    attr.max_dest_rd_atomic = 8;
    attr.min_rnr_timer = 5; // 0.06ms
    for (int attempt = 0;; ++attempt) {
      error = ibv_modify_qp(qp, &attr, rtrMask);
      if (error != ETIMEDOUT || attempt >= maxRtrRetries) {
        break;
      }
      // Full-jitter exponential backoff: sleep in [0, min(1000ms, 20ms << attempt)).
      const int cap = (int)std::min<long>(1000, 20L << std::min(attempt, 6));
      const int backoffMs = random<int>(0, cap);
      log.verbose("RTR to peer rank %zu timed out (attempt %d/%d); retrying in %dms\n", i, attempt + 1, maxRtrRetries,
          backoffMs);
      std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));
    }
    if (error) {
      // RTR resolves the peer's GID -> L2 address on RoCE, so ETIMEDOUT here (after retries)
      // typically means the destination is unreachable (cross-rail) or neighbor/ND resolution
      // keeps failing (e.g. IPv6 neighbor table exhausted at scale).
      throwErrno(error, fmt::sprintf("ibv_modify_qp ->RTR (peer rank %zu, port %d, remote gid %s)", i, portNum,
                            hexstr(&remoteAddress.gid, 16))
                            .c_str());
    }

    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 4979;
    attr.timeout = 5; // ?ms
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 8;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    error = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC |
            IBV_QP_ACCESS_FLAGS);
    if (error) {
      throwErrno(error, fmt::sprintf("ibv_modify_qp ->RTS (peer rank %zu, port %d)", i, portNum).c_str());
    }
  }
}

void IbCommon::init2Efa(int portNum, ibv_port_attr portAttributes) {
  const size_t rank = group->rank;
  const size_t size = group->size;

  log.debug("Init EFA\n");

  auto [gidIndex, gid] = selectGid(context, portNum);

  recvChannel = ibv_create_comp_channel(context);
  if (!recvChannel) {
    throwErrno(errno, "ibv_create_comp_channel");
  }

  recvCq = ibv_create_cq(context, maxSrqEntries, nullptr, recvChannel, 0);
  if (!recvCq) {
    throwErrno(errno, "ibv_create_cq");
  }

  inlineBytes = 0;

  ibv_qp_init_attr_ex initAttributes;
  std::memset(&initAttributes, 0, sizeof(initAttributes));
  initAttributes.qp_type = IBV_QPT_DRIVER;
  initAttributes.send_cq = cq;
  initAttributes.recv_cq = recvCq;
  initAttributes.cap.max_send_wr = maxWr;
  initAttributes.cap.max_send_sge = 1;
  initAttributes.cap.max_recv_wr = maxSrqEntries;
  initAttributes.cap.max_recv_sge = 1;
  initAttributes.srq = nullptr;
  initAttributes.sq_sig_all = 0;
  initAttributes.cap.max_inline_data = inlineBytes;
  initAttributes.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  initAttributes.pd = protectionDomain;
  initAttributes.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_READ | IBV_QP_EX_WITH_SEND;
  efadv_qp_init_attr efaAttr;
  std::memset(&efaAttr, 0, sizeof(efaAttr));
  efaAttr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  qp = efadv_create_qp_ex(context, &initAttributes, &efaAttr, sizeof(efaAttr));
  if (!qp) {
    throwErrno(errno, "efadv_create_qp_ex");
  }
  log.debug("Init EFA\n");
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  int error = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (error) {
    throwErrno(errno, "ibv_modify_qp");
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = portNum;
  attr.qkey = 0x4242;
  error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
  if (error) {
    throwErrno(errno, "ibv_modify_qp");
  }

  IbAddress address;
  address.lid = portAttributes.lid;
  address.qpNum = qp->qp_num;
  address.mtuIndex = portAttributes.active_mtu;
  address.gid = gid;

  qpex = ibv_qp_to_qp_ex(qp);

  for (size_t i = 0; i != size; ++i) {
    if (i != rank) {
      group->setupComms->sendTo(i, address);
    }
  }

  for (size_t i = 0; i != size; ++i) {
    // if (i == rank) {
    //   ahs.emplace_back();
    //   remoteAddresses.emplace_back();
    //   continue;
    // }
    auto remoteAddress = i == rank ? address : group->setupComms->recvFrom<IbAddress>(i);

    ibv_ah_attr ah_attr;
    std::memset(&ah_attr, 0, sizeof(ah_attr));
    ah_attr.port_num = portNum;
    ah_attr.dlid = remoteAddress.lid;
    ah_attr.is_global = true;
    ah_attr.grh.sgid_index = gidIndex;
    ah_attr.grh.hop_limit = 64;
    std::memcpy(&ah_attr.grh.dgid.raw, &remoteAddress.gid, sizeof(remoteAddress.gid));
    ahs.push_back(ibv_create_ah(protectionDomain, &ah_attr));
    if (!ahs.back()) {
      throwErrno(errno, "ibv_create_ah");
    }

    remoteAddresses.push_back(remoteAddress);
  }

  {
    ibv_qp_attr attr;
    ibv_qp_init_attr initAttr;
    int error = ibv_query_qp(qp, &attr, IBV_QP_STATE, &initAttr);
    CHECK(attr.qp_state == IBV_QPS_INIT);
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    error = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    if (error) {
      throwErrno(errno, "ibv_modify_qp");
    }

    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 4979;
    attr.rnr_retry = 7;
    error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY);
    if (error) {
      throwErrno(errno, "ibv_modify_qp");
    }
  }
  log.debug("connected, yey!\n");
}

void IbCommon::init(int portNum, ibv_port_attr portAttributes) {
  const size_t rank = group->rank;
  const size_t size = group->size;

  log.debug("IbCommon %p init (group size %d)\n", (void*)this, group->size);

  if (!protectionDomain) {
    protectionDomain = ibv_alloc_pd(context);
    if (!protectionDomain) {
      throwErrno(errno, "ibv_alloc_pd");
    }
  }

  cq = ibv_create_cq(context, maxCqEntries, nullptr, nullptr, 0);
  if (!cq) {
    throwErrno(errno, "ibv_create_cq");
  }

  if (is_efa_dev(context->device)) {
    return init2Efa(portNum, portAttributes);
  }

  return init2Ib(portNum, portAttributes);
}

namespace ib_poll {

struct PollThread {
  std::once_flag flag;
  std::thread thread;
  int epollFd = -1;
  int eventFd = -1;
  std::atomic_uint32_t removeCount = 0;

  SpinMutex mutex;
  HashMap<int, Function<void()>> callbacksContainer;

  void entry() {
    std::array<epoll_event, 1024> events;

    eventFd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (eventFd > 0) {
      add(eventFd, [this]() {
        char buf[8];
        read(eventFd, buf, 8);
      });
    }

    while (true) {
      uint32_t rc = removeCount.load(std::memory_order_relaxed);
      int n = epoll_wait(epollFd, events.data(), events.size(), 250);
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        throwErrno(errno, "epoll_wait");
      }
      for (int i = 0; i != n; ++i) {
        Function<void()> f((FunctionPointer)events[i].data.ptr);
        f();
        f.release();
      }

      if (n < events.size() && rc) {
        removeCount -= rc;
        futexWakeAll(&removeCount);
      }
    }

    remove(eventFd);
    eventFd = -1;

    close(epollFd);
  }

  void add(int fd, Function<void()> callback) {
    std::call_once(flag, [&] {
      epollFd = epoll_create1(EPOLL_CLOEXEC);
      if (epollFd == -1) {
        throwErrno(errno, "epoll_create1");
      }
      thread = std::thread([&] {
        async::setCurrentThreadName("moo/ib epoll");
        entry();
      });
    });

    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

    epoll_event e;
    e.data.ptr = callback.getPointer();
    e.events = EPOLLIN | EPOLLOUT | EPOLLET;
    if (epoll_ctl(epollFd, EPOLL_CTL_ADD, fd, &e)) {
      throwErrno(errno, "epoll_ctl");
    }

    std::lock_guard l(mutex);
    callbacksContainer[fd] = std::move(callback);
  }
  void remove(int fd) {
    epoll_event e;
    epoll_ctl(epollFd, EPOLL_CTL_DEL, fd, &e);

    ++removeCount;
    uint32_t rc = removeCount;
    while (rc != 0) {
      uint64_t buf = 1;
      write(eventFd, &buf, 8);
      futexWait(&removeCount, rc, std::chrono::seconds(1));
      rc = removeCount;
    }

    std::lock_guard l(mutex);
    auto i = callbacksContainer.find(fd);
    CHECK(i != callbacksContainer.end());
    callbacksContainer.erase(i);
  }
};
PollThread* pollThread = internalNew<PollThread>();

void add(int fd, Function<void()> callback) {
  pollThread->add(fd, std::move(callback));
}

void remove(int fd) {
  pollThread->remove(fd);
}

} // namespace ib_poll

} // namespace moodist
