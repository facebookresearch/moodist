// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ib_common.h"
#include "group.h"
#include "setup_comms.h"

#include "providers/efa/efadv.h"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/fcntl.h>

extern "C" bool is_efa_dev(ibv_device* device);

namespace moodist {

IbCommon::IbCommon(Group* group) : group(group) {}

IbCommon::~IbCommon() {}

// Select the best GID index for the given port.
// Preference order: IB > RoCE v2 > RoCE v1.
// Returns the GID index and the resolved GID value.
static std::pair<uint32_t, ibv_gid> selectGid(ibv_context* context, int portNum) {
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
    log.info("gid %d is %s of type %d    -- %d %d %d\n", i, hexstr(&entry.gid, 16), entry.gid_type, entry.gid_index,
        entry.port_num, entry.ndev_ifindex);
    int priority = gidTypePriority(entry.gid_type);
    // if (i == 3) {
    //   priority -= 100;
    // }
    if (priority <= bestPriority) {
      bestPriority = priority;
      bestIndex = i;
      bestGid = entry.gid;
      bestGidType = entry.gid_type;
      found = true;
    }
    // break;
  }

  if (!found) {
    // Fallback: use index 0 with ibv_query_gid (original behavior)
    log.error("selectGid: no valid GID found via ibv_query_gid_ex, falling back to index 0\n");
    int error = ibv_query_gid(context, portNum, 0, &bestGid);
    if (error) {
      throwErrno(errno, "ibv_query_gid");
    }
  } else {
    log.debug("selectGid: using GID index %d (%s)\n", bestIndex, gidTypeName(bestGidType));
  }

  return {bestIndex, bestGid};
}

void IbCommon::init2Ib(int portNum, ibv_port_attr portAttributes) {
  const size_t rank = group->rank;
  const size_t size = group->size;

  log.debug("Init IB port %d\n", portNum);

  auto [gidIndex, gid] = selectGid(context, portNum);

  log.info("gid for index %d is %s\n", gidIndex, hexstr(&gid, 16));

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
        throwErrno(errno, "ibv_modify_qp");
      }

      memset(&attr, 0, sizeof(attr));
      attr.qp_state = IBV_QPS_INIT;
      attr.pkey_index = 0;
      attr.port_num = portNum;
      attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
      error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
      if (error) {
        throwErrno(errno, "ibv_modify_qp");
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

  for (size_t i = 0; i != size; ++i) {
    // if (i == rank) {
    //   continue;
    // }

    auto& qp = qps.at(i);

    auto remoteAddress = i == rank ? loopbackAddress : group->setupComms->recvFrom<IbAddress>(i);

    ibv_qp_attr attr;
    ibv_qp_init_attr initAttr;
    int error = ibv_query_qp(qp, &attr, IBV_QP_STATE, &initAttr);
    CHECK(attr.qp_state == IBV_QPS_INIT);
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    std::memset(&attr.ah_attr, 0, sizeof(attr.ah_attr));
    log.info("REMOTE gid for index %d is %s\n", gidIndex, hexstr(&remoteAddress.gid, 16));
    attr.ah_attr.grh.dgid = remoteAddress.gid;
    attr.ah_attr.grh.sgid_index = gidIndex;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.is_global = true;
    attr.ah_attr.port_num = portNum;
    attr.ah_attr.dlid = remoteAddress.lid;
    attr.path_mtu = portAttributes.active_mtu;
    attr.dest_qp_num = remoteAddress.qpNum;
    attr.rq_psn = 4979;
    attr.max_dest_rd_atomic = 8;
    attr.min_rnr_timer = 5; // 0.06ms
    // attr.min_rnr_timer = 22;
    error = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
            IBV_QP_MIN_RNR_TIMER);
    if (error) {
      throwErrno(errno, "ibv_modify_qp");
    }

    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 4979;
    attr.timeout = 5; // ?ms
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    // attr.retry_cnt = 2;
    // attr.rnr_retry = 2;
    // attr.rnr_retry = 7; // infinite
    // attr.timeout = 17;
    // attr.retry_cnt = 7;
    // attr.rnr_retry = 7; // infinite
    attr.max_rd_atomic = 8;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    error = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC |
            IBV_QP_ACCESS_FLAGS);
    if (error) {
      throwErrno(errno, "ibv_modify_qp");
    }

    // fmt::printf("connected, yey!\n");
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

  // std::thread at([this]() {
  //   log.info("started async event loop\n");
  //   while (true) {
  //     ibv_async_event event;
  //     int err = ibv_get_async_event(context, &event);
  //     log.info("async err %d\n", err);
  //     if (err) {
  //       break;
  //     }
  //     log.info("async event %d\n", event.event_type);
  //   }
  // });
  // at.detach();

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
