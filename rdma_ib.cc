#include "group.h"
#include "ib_common.h"
#include "rdma.h"
#include <ranges>

namespace moodist {

namespace {

constexpr size_t maxWr = IbCommon::maxWr;
constexpr size_t maxCqEntries = IbCommon::maxCqEntries;
constexpr size_t maxSrqEntries = IbCommon::maxSrqEntries;

struct QueuedWr {
  size_t i;
  void* localAddress;
  uint32_t lkey;
  size_t bytes;
  RdmaCallback* callback;
};

struct Mr : RdmaMr {
  IbvMr mr;

  Mr(IbvMr mr) : mr(std::move(mr)) {
    lkey = this->mr->lkey;
    rkey = this->mr->rkey;
  }
};

constexpr const char* typestr = "ib";

struct RdmaIb : Rdma {
  Group* group = nullptr;
  std::unique_ptr<IbCommon> ib = nullptr;
  RdmaCpuThreadApi cpuThreadApi;

  size_t currentCqEntries = 0;

  IVector<Function<void()>> queuedWrs;

  size_t numCqEvents = 0;
  bool receivesRunning = false;

  RdmaIb(Group* group, std::unique_ptr<IbCommon> ib) : group(group), ib(std::move(ib)) {
    this->type = typestr;
  }

  void setCpuThreadApi(RdmaCpuThreadApi cpuThreadApi) {
    this->cpuThreadApi = cpuThreadApi;
  }

  void postWriteImpl(
      size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback, bool allowInline) {
    CHECK(i >= 0 && i < group->size);

    // bytesWritten += bytes;

    ibv_qp_ex* qp = ib->qpex;

    // log.info(
    //     "%d: rdma write %d bytes (%p -> %p, lkey %#x, rkey %#x) (dev %d, i %d)  (cb %#x)\n", rank, bytes,
    //     localAddress, remoteAddress, lkey, rkey, &dev - devices.data(), i, (uint64_t)(void*)callback);

    bool efa = qp != nullptr;
    if (!efa) {
      qp = ib->qpexs[i];
      CHECK(qp != nullptr);
    }

    ibv_wr_start(qp);
    qp->wr_id = (uint64_t)(void*)callback;
    qp->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write(qp, rkey, (uintptr_t)remoteAddress);
    if (allowInline && bytes <= ib->inlineBytes) {
      ibv_wr_set_inline_data(qp, localAddress, bytes);
    } else {
      ibv_wr_set_sge(qp, lkey, (uintptr_t)localAddress, bytes);
    }

    if (efa) {
      ibv_wr_set_ud_addr(qp, ib->ahs[i], ib->remoteAddresses[i].qpNum, 0x4242);
    }
    int error = ibv_wr_complete(qp);
    if (error) {
      throwErrno(error, "ibv_wr_complete");
    }
  }

  void postWrite(
      size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback, bool allowInline) {
    CHECK(i >= 0 && i < group->size);

    callback->i = i;
    ++callback->refcount;

    if (currentCqEntries == maxCqEntries) [[unlikely]] {
      queuedWrs.push_back([this, i, localAddress, lkey, remoteAddress, rkey, bytes, callback, allowInline] {
        postWriteImpl(i, localAddress, lkey, remoteAddress, rkey, bytes, callback, allowInline);
      });
      return;
    }
    ++currentCqEntries;
    if (currentCqEntries == 1) {
      // group->cpuThread->activeDevices.push_back(dev);
      cpuThreadApi.setActive();
    }
    postWriteImpl(i, localAddress, lkey, remoteAddress, rkey, bytes, callback, allowInline);
  }

  void postReadImpl(
      size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback) {
    CHECK(i >= 0 && i < group->size);

    // bytesRead += bytes;

    ibv_qp_ex* qp = ib->qpex;

    // log.info(
    //     "%d: rdma read %d bytes (%p <- %p, lkey %#x, rkey %#x) (dev %d, i %d)  (cb %#x)\n", rank, bytes,
    //     localAddress, remoteAddress, lkey, rkey, &dev - devices.data(), i, (uint64_t)(void*)callback);

    bool efa = qp != nullptr;
    if (!efa) {
      qp = ib->qpexs[i];
      CHECK(qp != nullptr);
    }

    ibv_wr_start(qp);
    qp->wr_id = (uint64_t)(void*)callback;
    qp->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_read(qp, rkey, (uintptr_t)remoteAddress);
    ibv_wr_set_sge(qp, lkey, (uintptr_t)localAddress, bytes);

    if (efa) {
      ibv_wr_set_ud_addr(qp, ib->ahs[i], ib->remoteAddresses[i].qpNum, 0x4242);
    }
    int error = ibv_wr_complete(qp);
    if (error) {
      fatal("ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
    }
  }

  void postRead(
      size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback) {
    callback->i = i;
    ++callback->refcount;

    if (currentCqEntries == maxCqEntries) [[unlikely]] {
      queuedWrs.push_back([this, i, localAddress, lkey, remoteAddress, rkey, bytes, callback] {
        postReadImpl(i, localAddress, lkey, remoteAddress, rkey, bytes, callback);
      });
      return;
    }
    ++currentCqEntries;
    if (currentCqEntries == 1) {
      // activeDevices.push_back(dev);
      cpuThreadApi.setActive();
    }

    return postReadImpl(i, localAddress, lkey, remoteAddress, rkey, bytes, callback);
  }

  void postSendImpl(size_t i, void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) {
    CHECK(i >= 0 && i < group->size);

    // bytesWritten += bytes;

    // log.info(
    //     "%d: post send %d bytes (%p lkey %#x) (dev %d, i %d, dev.currentCqEntries %d)\n", rank, bytes,
    //     localAddress, lkey, &dev - devices.data(), i, dev.currentCqEntries);

    ibv_qp_ex* qp = ib->qpex;
    bool efa = qp != nullptr;
    if (!efa) {
      qp = ib->qpexs[i];
      CHECK(qp != nullptr);
    }

    ibv_wr_start(qp);
    qp->wr_id = (uint64_t)(void*)callback;
    qp->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_send(qp);
    ibv_wr_set_sge(qp, lkey, (uintptr_t)localAddress, bytes);

    if (efa) {
      ibv_wr_set_ud_addr(qp, ib->ahs[i], ib->remoteAddresses[i].qpNum, 0x4242);
    }
    int error = ibv_wr_complete(qp);
    if (error) {
      fatal("ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
    }
  }

  void postSend(size_t i, void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) {
    callback->i = i;
    ++callback->refcount;

    if (currentCqEntries == maxCqEntries) [[unlikely]] {
      queuedWrs.push_back(
          [this, i, localAddress, lkey, bytes, callback] { postSendImpl(i, localAddress, lkey, bytes, callback); });
      return;
    }

    ++currentCqEntries;
    if (currentCqEntries == 1) {
      // activeDevices.push_back(dev);
      cpuThreadApi.setActive();
    }

    postSendImpl(i, localAddress, lkey, bytes, callback);
  }

  void postRecv(void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) {
    ibv_sge sge;
    sge.addr = (uintptr_t)localAddress;
    sge.length = bytes;
    sge.lkey = lkey;

    // bytesWritten += bytes;

    ++callback->refcount;

    // log.info("%d: post recv %d bytes (%p lkey %#x)\n", group->rank, bytes, localAddress, lkey);

    ibv_recv_wr wr;
    wr.wr_id = (uint64_t)(void*)callback;
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    ibv_recv_wr* badWr = nullptr;
    if (ib->qpex) {
      int error = ibv_post_recv(ib->qp, &wr, &badWr);
      if (error) {
        fatal("ibv_post_recv failed with error %d: %s\n", error, std::strerror(error));
      }
    } else {
      CHECK(ib->sharedReceiveQueue != nullptr);
      int error = ibv_post_srq_recv(ib->sharedReceiveQueue, &wr, &badWr);
      if (error) {
        fatal("ibv_post_srq_recv failed with error %d: %s\n", error, std::strerror(error));
      }
    }

    if (!receivesRunning) {
      startReceives();
    }
  }

  void startReceives() {
    if (receivesRunning) {
      return;
    }
    receivesRunning = true;
    CHECK(ib->recvChannel != nullptr);
    ib_poll::add(ib->recvChannel->fd, [this, channel = (ibv_comp_channel*)ib->recvChannel] {
      void* ctx;
      ibv_cq* cq = nullptr;
      if (ibv_get_cq_event(channel, &cq, &ctx)) {
        perror("ibv_get_cq_event");
        CHECK(false);
      }
      CHECK(cq == ib->recvCq);
      ++numCqEvents;

      CHECK(ibv_req_notify_cq(cq, 0) == 0);

      while (true) {
        std::array<ibv_wc, 4> wcs;
        int n = ibv_poll_cq(cq, wcs.size(), wcs.data());
        CHECK(n >= 0);
        for (size_t i = 0; i != n; ++i) {
          ibv_wc& wc = wcs[i];
          if (wc.status) {
            fatal(
                "rank %d Work completion with status %d (opcode %d, id %#x) (recv)\n", group->rank, wc.status,
                wc.opcode, wc.wr_id);
          } else {
            RdmaCallback* callback = (RdmaCallback*)(void*)wc.wr_id;
            callback->decref();
          }
        }
        if (n < wcs.size()) {
          break;
        }
      }
    });

    CHECK(ibv_req_notify_cq(ib->recvCq, 0) == 0);
  }

  void stopReceives() {
    if (!receivesRunning) {
      return;
    }
    receivesRunning = false;
    CHECK(ib->recvChannel != nullptr);
    ib_poll::remove(ib->recvChannel->fd);

    ibv_ack_cq_events(ib->recvCq, numCqEvents);
  }

  void poll() {
    std::array<ibv_wc, 4> wcs;
    int n = ibv_poll_cq(ib->cq, wcs.size(), wcs.data());
    for (auto& wc : std::views::take(wcs, n)) {
      if (wc.status) [[unlikely]] {
        NOINLINE_COLD({
          RdmaCallback* callback = (RdmaCallback*)(void*)wc.wr_id;
          if (cpuThreadApi.suppressErrors()) {
            log.error(
                "%s: Error communicating with %s: Work completion with status %d (%s).\n", cpuThreadApi.groupName(),
                cpuThreadApi.rankName(callback->i), wc.status, ibv_wc_status_str(wc.status));
            cpuThreadApi.setError();
          }
        });
      } else {
        CHECK(currentCqEntries != 0);
        if (!queuedWrs.empty()) [[unlikely]] {
          CHECK(currentCqEntries == maxCqEntries);
          queuedWrs.pop_front_value()();
        } else {
          --currentCqEntries;
          if (currentCqEntries == 0) {
            // activeDevices.erase(dev);
            cpuThreadApi.setInactive();
          }
        }
        CHECK(wc.wr_id != 0);
        RdmaCallback* callback = (RdmaCallback*)(void*)wc.wr_id;
        callback->decref();
      }
    }
  }

  bool mrCompatible(Rdma* other) {
    if (other->type != type) {
      return false;
    }
    return ((RdmaIb*)other)->ib->protectionDomain == ib->protectionDomain;
  }

  std::unique_ptr<RdmaMr> regMrCpu(void* address, size_t bytes) {
    ibv_mr* mr = ibv_reg_mr(
        ib->protectionDomain, address, bytes,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
    if (!mr) {
      perror("ibv_reg_mr");
      fatal("rank %d failed to register CPU memory at %#x size %#x\n", group->rank, address, bytes);
    }
    return std::make_unique<Mr>(mr);
  }
  std::unique_ptr<RdmaMr> regMrCuda(void* address, size_t bytes) {
    ibv_mr* mr = ibv_reg_mr(
        ib->protectionDomain, address, bytes,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING);
    if (!mr) {
      perror("ibv_reg_mr");
      fatal("rank %d failed to register CUDA memory at %#x size %#x\n", group->rank, address, bytes);
    }
    return std::make_unique<Mr>(mr);
  }

  constexpr bool supportsCuda() const {
    return true;
  }

  void close() {
    cpuThreadApi = {};
  }
};

} // namespace

std::unique_ptr<Rdma> makeRdmaIb(Group* group, std::unique_ptr<IbCommon> ib) {
  return std::make_unique<RdmaIb>(group, std::move(ib));
}

} // namespace moodist
