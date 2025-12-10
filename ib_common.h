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

namespace ib_poll {
void add(int fd, Function<void()> callback);
void remove(int fd);
} // namespace ib_poll

} // namespace moodist
