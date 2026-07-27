// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"
#include "hash_map.h"

#define RDMA_STATIC_PROVIDERS mlx5
#include "libibverbs/verbs.h"

namespace moodist {

struct IbAddress {
  uint16_t lid;
  uint32_t qpNum;
  int mtuIndex;
  ibv_gid gid;

  bool operator==(const IbAddress& n) const = delete;
  // bool operator==(const IbAddress& n) const {
  //   return lid == n.lid && qpNum == n.qpNum;
  // }
  template<typename T>
  void serialize(T& x) {
    x(lid, qpNum, mtuIndex, gid);
  }
};

template<typename T, int (*destroy)(T*)>
struct IbvPtr {
  T* value = nullptr;
  IbvPtr() = default;
  IbvPtr(const IbvPtr&) = delete;
  IbvPtr& operator=(const IbvPtr&) = delete;
  IbvPtr(std::nullptr_t) {}
  IbvPtr(T* value) : value(value) {}
  IbvPtr(IbvPtr&& n) noexcept {
    std::swap(value, n.value);
  }
  IbvPtr& operator=(IbvPtr&& n) noexcept {
    std::swap(value, n.value);
    return *this;
  }
  ~IbvPtr() {
    if (value) [[unlikely]] {
      int error = destroy(value);
      if (error) [[unlikely]] {
        NOINLINE_COLD(log.error("Failed to destroy ibv instance of type %s: error %d: %s\n", typeid(T).name(), error,
            std::strerror(error)););
      }
    }
  }
  operator T*() {
    return value;
  }
  T& operator*() {
    return *value;
  }
  T* operator->() {
    return value;
  }
  operator const T*() const {
    return value;
  }
  const T& operator*() const {
    return *value;
  }
  const T* operator->() const {
    return value;
  }
  T* release() {
    return std::exchange(value, nullptr);
  }
};

template<typename T, int (*destroy)(T*)>
struct IbvSharedPtr {
  std::shared_ptr<IbvPtr<T, destroy>> ptr;
  IbvSharedPtr() = default;
  IbvSharedPtr(T* value) : ptr(std::make_shared<IbvPtr<T, destroy>>(value)) {}
  IbvSharedPtr(IbvPtr<T, destroy>&& n) {
    ptr = std::make_shared<IbvPtr<T, destroy>>(std::move(n));
  }
  operator T*() {
    if (!ptr) {
      return nullptr;
    }
    return &**ptr;
  }
  T& operator*() {
    return **ptr;
  }
  T* operator->() {
    return &**ptr;
  }
  operator const T*() const {
    if (!ptr) {
      return nullptr;
    }
    return &**ptr;
  }
  const T& operator*() const {
    return **ptr;
  }
  const T* operator->() const {
    return &**ptr;
  }
};

using IbvCq = IbvPtr<ibv_cq, ibv_destroy_cq>;
using IbvQp = IbvPtr<ibv_qp, ibv_destroy_qp>;
using IbvMr = IbvPtr<ibv_mr, ibv_dereg_mr>;
using IbvAh = IbvPtr<ibv_ah, ibv_destroy_ah>;

struct Group;

struct IbCommon {
  Group* group;

  static constexpr size_t maxWr = 2048;
  static constexpr size_t maxCqEntries = 2048;
  static constexpr size_t maxSrqEntries = 256;

  IbvSharedPtr<ibv_context, ibv_close_device> context;
  IbvSharedPtr<ibv_pd, ibv_dealloc_pd> protectionDomain;
  IbvCq cq;
  IbvQp qp;
  ibv_qp_ex* qpex = nullptr;

  IbvPtr<ibv_srq, ibv_destroy_srq> sharedReceiveQueue;
  IbvPtr<ibv_comp_channel, ibv_destroy_comp_channel> recvChannel;
  IbvCq recvCq;

  std::vector<IbvAh> ahs;
  std::vector<IbAddress> remoteAddresses;

  std::vector<IbvQp> qps;
  std::vector<ibv_qp_ex*> qpexs;

  size_t inlineBytes = 0;

  void init(int portNum, ibv_port_attr portAttributes);

  void init2Ib(int portNum, ibv_port_attr portAttributes);
  void init2Efa(int portNum, ibv_port_attr portAttributes);

  IbCommon(Group* group);
  ~IbCommon();
};

// Select the best GID index for RoCE/IB on `portNum` (preference IB > RoCE v2 > RoCE v1; among
// GIDs of the best type the last one is chosen — tuned for IPv6-only RoCE fabrics; overridable
// via MOODIST_IB_GID_INDEX). Exposed so callers can hoist the 256-entry GID scan out of repeated
// setup loops. Throws only if no GID can be resolved at all.
std::pair<uint32_t, ibv_gid> selectGid(ibv_context* context, int portNum, bool quiet = false);

// One local endpoint participating in a batched rail probe (see probeIbRailBatch). A null
// `context` marks a slot this side cannot supply a NIC for (e.g. the responder did not find the
// requested ibPath); the slot still participates in the exchange to keep both sides in lockstep
// and resolves to "not connected".
struct ProbeEndpoint {
  ibv_context* context = nullptr;
  int portNum = 0;
  ibv_port_attr portAttributes{};
  uint32_t gidIndex = 0;
  ibv_gid gid{};
};

// Batched rail connectivity probe.
//
// Brings up one temporary RC QP per entry in `local`, each paired with the peer rank's
// corresponding entry (slot j here <-> slot j on the peer), and — on the initiator only —
// performs one test RDMA write per slot, polling all of them together. Both endpoints must call
// this with the same `local.size()`, matching `peerRank`, and opposite `initiator`. Returns a
// vector of the same length where entry j is 1 iff slot j's write completed successfully — i.e.
// the two NICs are on the same routable rail. This is the empirical, topology-agnostic
// definition of "same rail" used by rail-aware device selection (no assumptions about device
// names or GID subnets). The result is authoritative on the initiator; the responder receives
// and returns the same verdicts.
//
// Batching collapses what would be many serial probes into a single set of parallel QP
// bring-ups and one shared poll, so a whole wave of candidates costs ~one round-trip plus one
// (early-exiting) completion wait instead of one per candidate.
//
// Never throws: on any verbs failure (including a cross-rail QP that cannot route) the slot
// resolves to 0, and the same fixed setupComms message exchange always runs, so both sides stay
// in lockstep regardless of how many slots are valid (including an empty `local`). The GID /
// traffic-class / service level match the real QP setup, so the probe faithfully reflects what
// production QPs will do. The completion wait is bounded by MOODIST_IB_RAIL_PROBE_TIMEOUT_MS.
std::vector<uint8_t> probeIbRailBatch(Group* group, const std::vector<ProbeEndpoint>& local, size_t peerRank,
    bool initiator);

namespace ib_poll {
void add(int fd, Function<void()> callback);
void remove(int fd);
} // namespace ib_poll

} // namespace moodist
