#pragma once

#include "common.h"
#include "hash_map.h"

#define RDMA_STATIC_PROVIDERS all
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
  IbvPtr(IbvPtr&& n) {
    std::swap(value, n.value);
  }
  IbvPtr& operator=(IbvPtr&& n) {
    std::swap(value, n.value);
    return *this;
  }
  ~IbvPtr() {
    if (value) {
      int error = destroy(value);
      if (error) {
        fmt::fprintf(
            stderr, "Failed to destroy ibv instance of type %s: error %d: %s\n", typeid(T).name(), error,
            std::strerror(error));
        fflush(stderr);
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

using IbvCq = IbvPtr<ibv_cq, ibv_destroy_cq>;
using IbvQp = IbvPtr<ibv_qp, ibv_destroy_qp>;
using IbvMr = IbvPtr<ibv_mr, ibv_dereg_mr>;
using IbvAh = IbvPtr<ibv_ah, ibv_destroy_ah>;

struct Group;

struct IbCommon {
  Group* group;

  static constexpr size_t maxWr = 1024;
  static constexpr size_t maxCqEntries = 1024;

  IbvPtr<ibv_context, ibv_close_device> context;
  IbvPtr<ibv_pd, ibv_dealloc_pd> protectionDomain;
  IbvCq cq;
  IbvQp qp;
  ibv_qp_ex* qpex = nullptr;
  std::vector<IbvAh> ahs;
  std::vector<IbAddress> remoteAddresses;

  std::vector<IbvQp> qps;
  std::vector<ibv_qp_ex*> qpexs;

  void init(int portNum, ibv_port_attr portAttributes);

  void init2Ib(int portNum, ibv_port_attr portAttributes);

  IbCommon(Group* group);
  ~IbCommon();
};

} // namespace moodist
