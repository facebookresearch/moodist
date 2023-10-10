#include "ib_common.h"
#include "group.h"
#include "setup_comms.h"

#include "providers/efa/efadv.h"

namespace moodist {

IbCommon::IbCommon(Group* group) : group(group) {}

IbCommon::~IbCommon() {
  fmt::printf("~IbCommon %p (group size %d)\n", (void*)this, group->size);
}

void IbCommon::init(int portNum, ibv_port_attr portAttributes) {
  CHECK(false);
  // const size_t rank = group->rank;
  // const size_t size = group->size;

  // fmt::printf("IbCommon %p init (group size %d)\n", (void*)this, group->size);

  // protectionDomain = ibv_alloc_pd(context);

  // cq = ibv_create_cq(context, maxCqEntries, nullptr, nullptr, 0);

  // for (size_t i = 0; i != size; ++i) {
  //   if (i == rank) {
  //     qps.emplace_back();
  //   } else {
  //     ibv_qp_init_attr initAttributes;
  //     std::memset(&initAttributes, 0, sizeof(initAttributes));
  //     initAttributes.qp_type = IBV_QPT_RC;
  //     initAttributes.send_cq = cq;
  //     initAttributes.recv_cq = cq;
  //     initAttributes.cap.max_send_wr = maxWr;
  //     initAttributes.cap.max_send_sge = 1;
  //     initAttributes.cap.max_recv_wr = 1;
  //     initAttributes.cap.max_recv_sge = 1;
  //     initAttributes.srq = nullptr;
  //     initAttributes.sq_sig_all = 0;
  //     initAttributes.cap.max_inline_data = sizeof(DynamicAddresses);
  //     IbvQp qp = ibv_create_qp(protectionDomain, &initAttributes);
  //     if (!qp) {
  //       perror("ibv_create_qp");
  //       TORCH_CHECK(false);
  //     }
  //     ibv_qp_attr attr;
  //     memset(&attr, 0, sizeof(attr));
  //     attr.qp_state = IBV_QPS_RESET;
  //     int error = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  //     if (error) {
  //       fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
  //       std::fflush(stderr);
  //       TORCH_CHECK(false);
  //     }

  //     memset(&attr, 0, sizeof(attr));
  //     attr.qp_state = IBV_QPS_INIT;
  //     attr.pkey_index = 0;
  //     attr.port_num = portNum;
  //     attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  //     error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  //     if (error) {
  //       fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
  //       std::fflush(stderr);
  //       TORCH_CHECK(false);
  //     }

  //     IbAddress address;
  //     address.lid = portAttributes.lid;
  //     address.qpNum = qp->qp_num;
  //     address.mtuIndex = portAttributes.active_mtu;
  //     error = ibv_query_gid(context, portNum, 0, &address.gid);
  //     if (error) {
  //       perror("ibv_query_gid");
  //       CHECK(false);
  //     }

  //     group->setupComms->sendTo(i, address);

  //     qps.push_back(std::move(qp));
  //   }
  // }

  // for (size_t i = 0; i != size; ++i) {
  //   if (i == rank) {
  //     continue;
  //   }

  //   auto& qp = qps.at(i);

  //   auto remoteAddress = group->setupComms->recvFrom<IbAddress>(i);

  //   ibv_qp_attr attr;
  //   ibv_qp_init_attr initAttr;
  //   int error = ibv_query_qp(qp, &attr, IBV_QP_STATE, &initAttr);
  //   TORCH_CHECK(attr.qp_state == IBV_QPS_INIT);
  //   std::memset(&attr, 0, sizeof(attr));
  //   attr.qp_state = IBV_QPS_RTR;
  //   std::memset(&attr.ah_attr, 0, sizeof(attr.ah_attr));
  //   attr.ah_attr.port_num = portNum;
  //   attr.ah_attr.dlid = remoteAddress.lid;
  //   attr.path_mtu = portAttributes.active_mtu;
  //   attr.dest_qp_num = remoteAddress.qpNum;
  //   attr.rq_psn = 4979;
  //   attr.max_dest_rd_atomic = 16;
  //   attr.min_rnr_timer = 5; // 0.06ms
  //   // attr.min_rnr_timer = 22;
  //   error = ibv_modify_qp(
  //       qp, &attr,
  //       IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
  //           IBV_QP_MIN_RNR_TIMER);
  //   if (error) {
  //     fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
  //     std::fflush(stderr);
  //     TORCH_CHECK(false);
  //   }

  //   std::memset(&attr, 0, sizeof(attr));
  //   attr.qp_state = IBV_QPS_RTS;
  //   attr.sq_psn = 4979;
  //   attr.timeout = 8; // 1ms
  //   attr.retry_cnt = 7;
  //   attr.rnr_retry = 7;
  //   // attr.retry_cnt = 2;
  //   // attr.rnr_retry = 2;
  //   // attr.rnr_retry = 7; // infinite
  //   // attr.timeout = 17;
  //   // attr.retry_cnt = 7;
  //   // attr.rnr_retry = 7; // infinite
  //   attr.max_rd_atomic = 16;
  //   attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  //   error = ibv_modify_qp(
  //       qp, &attr,
  //       IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC
  //       |
  //           IBV_QP_ACCESS_FLAGS);
  //   if (error) {
  //     fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
  //     std::fflush(stderr);
  //     TORCH_CHECK(false);
  //   }

  //   // fmt::printf("connected, yey!\n");
  // }
}

void IbCommon::initEfa(int portNum, ibv_port_attr portAttributes) {
  const size_t rank = group->rank;
  const size_t size = group->size;

  fmt::printf("IbCommon %p init (EFA) (group size %d)\n", (void*)this, group->size);

  protectionDomain = ibv_alloc_pd(context);

  cq = ibv_create_cq(context, maxCqEntries, nullptr, nullptr, 0);

  ibv_qp_init_attr_ex initAttributes;
  std::memset(&initAttributes, 0, sizeof(initAttributes));
  initAttributes.qp_type = IBV_QPT_DRIVER;
  initAttributes.send_cq = cq;
  initAttributes.recv_cq = cq;
  initAttributes.cap.max_send_wr = maxWr;
  initAttributes.cap.max_send_sge = 1;
  initAttributes.cap.max_recv_wr = 1;
  initAttributes.cap.max_recv_sge = 1;
  initAttributes.srq = nullptr;
  initAttributes.sq_sig_all = 0;
  initAttributes.cap.max_inline_data = 32;
  initAttributes.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  initAttributes.pd = protectionDomain;
  initAttributes.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE;
  efadv_qp_init_attr efaAttr;
  std::memset(&efaAttr, 0, sizeof(efaAttr));
  efaAttr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  qp = efadv_create_qp_ex(context, &initAttributes, &efaAttr, sizeof(efaAttr));
  if (!qp) {
    perror("efadv_create_qp_ex");
    CHECK(false);
  }
  ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  int error = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (error) {
    fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
    std::fflush(stderr);
    CHECK(false);
  }

  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = portNum;
  attr.qkey = 0x4242;
  error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
  if (error) {
    fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
    std::fflush(stderr);
    CHECK(false);
  }

  IbAddress address;
  address.lid = portAttributes.lid;
  address.qpNum = qp->qp_num;
  address.mtuIndex = portAttributes.active_mtu;
  error = ibv_query_gid(context, portNum, 0, &address.gid);
  if (error) {
    perror("ibv_query_gid");
    CHECK(false);
  }

  qpex = ibv_qp_to_qp_ex(qp);

  for (size_t i = 0; i != size; ++i) {
    if (i != rank) {
      group->setupComms->sendTo(i, address);
    }
  }

  for (size_t i = 0; i != size; ++i) {
    if (i == rank) {
      ahs.emplace_back();
      remoteAddresses.emplace_back();
      continue;
    }
    auto remoteAddress = group->setupComms->recvFrom<IbAddress>(i);

    ibv_ah_attr ah_attr;
    std::memset(&ah_attr, 0, sizeof(ah_attr));
    ah_attr.port_num = portNum;
    ah_attr.dlid = remoteAddress.lid;
    ah_attr.is_global = true;
    std::memcpy(&ah_attr.grh.dgid.raw, &remoteAddress.gid, sizeof(remoteAddress.gid));
    ahs.push_back(ibv_create_ah(protectionDomain, &ah_attr));
    if (!ahs.back()) {
      perror("ibv_create_ah");
      CHECK(false);
    }

    remoteAddresses.push_back(remoteAddress);
  }

  {
    ibv_qp_attr attr;
    ibv_qp_init_attr initAttr;
    int error = ibv_query_qp(qp, &attr, IBV_QP_STATE, &initAttr);
    TORCH_CHECK(attr.qp_state == IBV_QPS_INIT);
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    error = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    if (error) {
      fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
      std::fflush(stderr);
      TORCH_CHECK(false);
    }

    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 4979;
    attr.rnr_retry = 7;
    error = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RNR_RETRY);
    if (error) {
      fmt::fprintf(stderr, "ibv_modify_qp failed with error %d: %s\n", error, std::strerror(error));
      std::fflush(stderr);
      TORCH_CHECK(false);
    }
  }
  fmt::printf("connected, yey!\n");
}

ibv_mr* IbCommon::getMr(uintptr_t address, size_t length) {
  unsigned long long bufferId = -1;
  CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
  TORCH_CHECK(bufferId != -1);
  // unsigned long long bufferId2 = -1;
  // CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + length));
  // TORCH_CHECK(bufferId == bufferId2);
  auto it = mrs.find(bufferId);
  uint32_t lkey;
  uint32_t rkey;
  if (it != mrs.end() && (uintptr_t)it->second.mr->addr + it->second.mr->length >= address + length &&
      it->second.valid()) {
    return (ibv_mr*)it->second.mr;
  } else {
    // int deviceOrdinal = 0;
    // CHECK_CU(cuPointerGetAttribute(&deviceOrdinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (uintptr_t)address));
    // cudaSetDevice(deviceOrdinal);
    CUdeviceptr base = 0;
    size_t size = 0;
    CHECK_CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)address));
    MrInfo mri;
    if (length > size) {
      while (size < length) {
        unsigned long long bufferId2 = -1;
        CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + size));
        fmt::printf("buffer2 %d\n", bufferId2);
        mri.buffers.emplace_back(address + size, bufferId2);
        CUdeviceptr base2 = 0;
        size_t size2 = 0;
        CHECK_CU(cuMemGetAddressRange(&base2, &size2, (CUdeviceptr)address + size));
        TORCH_CHECK(base2 == base + size);
        size += size2;
      }
      TORCH_CHECK(false);
    }

    // int fd = -1;
    // CHECK_CU(cuMemGetHandleForAddressRange(&fd, base, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

    // mri.mr = ibv_reg_dmabuf_mr(protectionDomain, 0, size, base, fd, IBV_ACCESS_LOCAL_WRITE |
    // IBV_ACCESS_REMOTE_WRITE);
    mri.mr = ibv_reg_mr(protectionDomain, (void*)base, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mri.mr) {
      fmt::printf(
          "rank %d failed to register %#x size %#x,  original address %#x size %#x\n", group->rank, (uintptr_t)base,
          size, address, length);
      perror("ibv_reg_mr");
      TORCH_CHECK(false);
    }

    // fmt::printf(
    //     "new mapped range of %d bytes at %#x -> fd %d  (mr lkey %#x rkey %#x)\n", size, base, bufferId, mri.mr->lkey,
    //     mri.mr->rkey);
    // for (auto [address, bufferId] : mri.buffers) {
    //   fmt::printf("  also spans buffer %d at %#x\n", bufferId, address);
    // }

    ibv_mr* r = mri.mr;

    mrs.emplace(bufferId, std::move(mri));

    return r;
  }
}
} // namespace moodist
