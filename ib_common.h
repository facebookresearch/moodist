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

struct FixedAddresses {
  uintptr_t dataAddress = 0;
  uintptr_t cpuStepDoneAddress = 0;
  uintptr_t cudaStepDoneAddress = 0;
  uintptr_t dynamicAddressesAddress = 0;
  uint32_t dataRkey = 0;
  uint32_t cpuStepDoneRkey = 0;
  uint32_t cudaStepDoneRkey = 0;
  uint32_t dynamicAddressesRkey = 0;

  std::string bootId;
  std::string devId;
  CUipcMemHandle dataIpcHandle;
  CUipcMemHandle cudaStepDoneIpcHandle;

  template<typename X>
  void serialize(X& x) {
    x(dataAddress, cpuStepDoneAddress, cudaStepDoneAddress, dynamicAddressesAddress, dataRkey, cpuStepDoneRkey,
      cudaStepDoneRkey, dynamicAddressesRkey, bootId, devId, dataIpcHandle, cudaStepDoneIpcHandle);
  }
};

struct alignas(64) DynamicAddresses {
  std::array<uint32_t, 32> gatherKey;
  uintptr_t gatherAddress;
};

// struct alignas(64) DynamicAddresses {
//   uintptr_t gatherDataAddress;
//   uintptr_t reduceDataAddress;
//   uint32_t gatherDataRkey;
//   uint32_t reduceDataRkey;
//   uint32_t stepValue;

//   // template<typename X>
//   // void serialize(X& x) {
//   //   x(gatherDataAddress, gatherDataRkey, reduceDataAddress, reduceDataRkey, stepValue);
//   // }
// };

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

  std::vector<FixedAddresses> fixedAddresses;
  std::vector<DynamicAddresses> dynamicAddresses;

  IbvMr dynamicAddressesMr;

  void init(int portNum, ibv_port_attr portAttributes);
  void initEfa(int portNum, ibv_port_attr portAttributes);

  struct MrInfo {
    IbvMr mr;
    std::vector<std::pair<uintptr_t, unsigned long long>> buffers;

    bool valid() const {
      for (auto [address, bufferId] : buffers) {
        unsigned long long bufferId2 = -1;
        cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address);
        if (bufferId2 != bufferId) {
          fmt::printf(
              "mr %#x %#x invalidated due to buffer id change %d -> %d\n", mr->lkey, mr->rkey, bufferId, bufferId2);
          return false;
        }
      }
      return true;
    }
  };
  HashMap<unsigned long long, MrInfo> mrs;

  ibv_mr* getMr(uintptr_t address, size_t length);

  IbCommon(Group* group);
  ~IbCommon();
};

} // namespace moodist
