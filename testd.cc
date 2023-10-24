
#include "common.h"
#include "serialization.h"
#include "setup_comms.h"
#include "socket.h"

#define RDMA_STATIC_PROVIDERS all
#include "libibverbs/verbs.h"
#include "providers/efa/efadv.h"

namespace moodist {

void connect(SetupComms* setup, size_t rank, size_t size, std::string rank0Address) {
  std::mutex mutex;
  auto getAddress = [&]() {
    std::vector<std::string> addresses = setup->listenerAddresses();
    auto buffer = serializeToBuffer(addresses);

    std::string r;
    for (size_t i = 0; i != buffer->size(); ++i) {
      uint8_t v = (uint8_t)buffer->data()[i];
      r += "0123456789abcdef"[v >> 4];
      r += "0123456789abcdef"[v & 0xf];
    }
    return r;
  };

  std::string addr = getAddress();

  std::vector<Socket> conns;

  Socket sock;
  CachedReader reader(&sock);
  std::atomic_int state = 0;
  size_t len;

  auto buffer = serializeToBuffer(addr);

  if (rank == 0) {
    sock = Socket::Tcp();

    sock.listen("0.0.0.0:51022");
    sock.accept([&addr, &conns, &buffer, &mutex](Error* e, Socket conn) {
      if (!e) {
        std::lock_guard l(mutex);
        iovec vec;
        vec.iov_base = buffer->data();
        vec.iov_len = buffer->size();
        conn.writev(&vec, 1, [](Error*) {});
        conns.push_back(std::move(conn));
      }
    });
  } else {
    std::atomic_bool error = true;
    while (error) {
      error = false;
      sock = Socket::Tcp();
      sock.connect(rank0Address + ":51022", [&error](Error* e) {
        if (e) {
          error = true;
        }
      });
      sock.setOnRead([&](Error* e) {
        if (e) {
          error = true;
          return;
        }
        if (state == 0) {
          if (!reader.readCopy(&len, sizeof(len))) {
            return;
          }
          state = 1;
        }
        if (state == 1) {
          rank0Address.resize(len);
          if (!reader.readCopy(rank0Address.data(), len)) {
            return;
          }
          state = 2;
        }
      });

      // fmt::printf("waiting for address\n");
      while (state != 2) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (error) {
          fmt::printf("connect error, retrying!\n");
          break;
        }
      }
      if (!error) {
        // fmt::printf("got address! %s\n", rank0Address);
      }
    }
    std::vector<uint8_t> data;
    size_t i = 0;
    for (char vv : rank0Address) {
      size_t index = vv >= '0' && vv <= '9' ? vv - '0' : vv - 'a' + 10;
      if (index >= 16) {
        throw std::invalid_argument("ProcessGroup: invalid address");
      }
      if (i % 2 == 0) {
        data.push_back(index);
      } else {
        data.back() <<= 4;
        data.back() |= index;
      }
      ++i;
    }
    std::vector<std::string> remoteAddresses;
    deserializeBuffer(data.data(), data.size(), remoteAddresses);

    for (auto& address : remoteAddresses) {
      setup->connect(address);
    }
  }

  setup->waitForConnections();
  for (size_t i = 0; i != size; ++i) {
    if (i == rank) {
      continue;
    }
    setup->sendTo(i, fmt::sprintf("hello %d from %d", i, rank));
  }
  for (size_t i = 0; i != size; ++i) {
    if (i == rank) {
      continue;
    }
    std::string greeting = setup->recvFrom<std::string>(i);
    CHECK(greeting == fmt::sprintf("hello %d from %d", rank, i));
  }

  // fmt::printf("connect done!\n");
}

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

struct IbAddress {
  uint16_t lid;
  uint32_t qpNum;
  int mtuIndex;
  ibv_gid gid;

  bool operator==(const IbAddress& n) const = delete;
  template<typename T>
  void serialize(T& x) {
    x(lid, qpNum, mtuIndex, gid);
  }
};

struct Device {
  static constexpr size_t maxWr = 1024;
  static constexpr size_t maxCqEntries = 1024;

  IbvPtr<ibv_context, ibv_close_device> context;
  IbvPtr<ibv_pd, ibv_dealloc_pd> protectionDomain;
  IbvCq cq;
  IbvQp qp;
  ibv_qp_ex* qpex = nullptr;
  std::vector<IbvAh> ahs;
  std::vector<IbAddress> remoteAddresses;

  void initEfa(SetupComms* setup, int portNum, ibv_port_attr portAttributes, size_t rank, size_t size) {

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
        setup->sendTo(i, address);
      }
    }

    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        ahs.emplace_back();
        remoteAddresses.emplace_back();
        continue;
      }
      auto remoteAddress = setup->recvFrom<IbAddress>(i);

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
    // fmt::printf("connected, yey!\n");
  }
};

struct Callback {
  std::function<void()> onComplete;
};

struct DeviceInfo {
  IbvPtr<ibv_context, ibv_close_device> context;
  int portNum;
  ibv_port_attr portAttributes;
};

void test(int rank, int size, std::string rank0Address) {

  auto setup = createSetupComms(rank, size);

  connect(&*setup, rank, size, rank0Address);

  std::vector<DeviceInfo> usableDevices;

  ibv_device** list = ibv_get_device_list(nullptr);
  for (ibv_device** i = list; *i; ++i) {
    ibv_device* di = *i;

    // fmt::printf("device %s\n", di->name);
    IbvPtr<ibv_context, ibv_close_device> ctx = ibv_open_device(di);
    if (ctx) {
      ibv_device_attr attributes;
      std::memset(&attributes, 0, sizeof(attributes));
      if (ibv_query_device(ctx, &attributes) != 0) {
        continue;
      }

      int portNum = -1;
      ibv_port_attr portAttributes;
      int bestSpeed = -1;
      for (size_t i = 0; i != attributes.phys_port_cnt; ++i) {
        ibv_port_attr attributes;
        std::memset(&attributes, 0, sizeof(attributes));
        if (ibv_query_port(ctx, 1 + i, &attributes) == 0) {
          if (attributes.state != IBV_PORT_ACTIVE) {
            continue;
          }
          int speed = attributes.active_speed;
          if (attributes.link_layer == IBV_LINK_LAYER_INFINIBAND) {
            speed += 0x100000;
          }
          if (speed > bestSpeed) {
            bestSpeed = speed;
            portNum = 1 + i;
            portAttributes = attributes;
          }
        }
        // fmt::printf("queried port %d\n", 1 + i);
      }
      if (portNum == -1) {
        continue;
      }

      usableDevices.emplace_back();
      auto& info = usableDevices.back();
      info.context = std::move(ctx);
      info.portNum = portNum;
      info.portAttributes = portAttributes;

      // break;
    }
  }
  ibv_free_device_list(list);

  auto allDeviceSizes = setup->allgather(usableDevices.size());
  for (size_t remoteDevicesSize : allDeviceSizes) {
    CHECK(remoteDevicesSize == usableDevices.size());
  }

  std::vector<Device> devices;
  for (auto& info : usableDevices) {
    devices.emplace_back();
    auto& dev = devices.back();
    dev.context = std::move(info.context);
    dev.initEfa(&*setup, info.portNum, info.portAttributes, rank, size);
  }

  if (rank == 0) {
    fmt::printf("devices.size() is %d\n", devices.size());
  }

  struct MemoryRegistration {
    std::array<IbvMr, 32> mrs;
  };

  auto regMr = [&](uintptr_t address, size_t bytes) {
    auto ptr = std::make_unique<MemoryRegistration>();
    CHECK(devices.size() <= ptr->mrs.size());
    for (size_t i = 0; i != devices.size(); ++i) {
      IbvMr mr = ibv_reg_mr(
          devices[i].protectionDomain, (void*)address, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
      if (!mr) {
        fmt::printf("rank %d failed to register CPU memory at %#x size %#x\n", rank, address, bytes);
        perror("ibv_reg_mr");
        TORCH_CHECK(false);
      }
      // fmt::printf("new mapped range of %d bytes at %#x -> mr lkey %#x rkey %#x\n", size, address, mr->lkey,
      // mr->rkey);
      ptr->mrs[i] = std::move(mr);
    }
    return ptr;
  };

  struct RemoteAddressAndKey {
    uintptr_t address;
    std::vector<uint32_t> keys;
  };

  auto distributeAddressAndKeys = [&](uintptr_t address, std::unique_ptr<MemoryRegistration>& mr) {
    std::vector<uint32_t> localKeys;
    for (size_t i = 0; i != devices.size(); ++i) {
      localKeys.push_back(mr->mrs.at(i)->rkey);
    }
    auto keys = setup->allgather(localKeys);
    auto addrs = setup->allgather(address);
    for (auto& v : keys) {
      CHECK(v.size() == devices.size());
    }
    std::vector<RemoteAddressAndKey> r;
    r.resize(size);
    for (size_t i = 0; i != size; ++i) {
      r[i].address = addrs[i];
      r[i].keys = keys[i];
    }
    return r;
  };

  auto poll = [&]() {
    for (auto& dev : devices) {
      ibv_wc wcs[4];
      int n = ibv_poll_cq(dev.cq, 4, wcs);
      CHECK(n >= 0);
      for (size_t i = 0; i != n; ++i) {
        ibv_wc& wc = wcs[i];
        if (wc.status) {
          fmt::fprintf(
              stderr, "rank %d Work completion with status %d (opcode %d, id %d)\n", rank, wc.status, wc.opcode,
              wc.wr_id);
          std::fflush(stderr);
          CHECK(false);
        } else {

          if (wc.wr_id) {
            Callback* callback = (Callback*)(void*)wc.wr_id;
            callback->onComplete();
            delete callback;
          }
        }
      }
    }
  };

  auto writeData = [&](Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
                       size_t bytes, Callback* callback = nullptr) {
    CHECK(i >= 0 && i < size);
    CHECK(i != rank);
    ibv_send_wr wr;
    ibv_sge sge;
    sge.addr = (uintptr_t)localAddress;
    sge.length = bytes;
    sge.lkey = lkey;

    ibv_qp_ex* qp = dev.qpex;

    // fmt::printf(
    //     "%d: rdma write %d bytes (%p -> %p, lkey %#x, rkey %#x) (dev %d, i %d)\n", rank, bytes, localAddress,
    //     remoteAddress, lkey, rkey, &dev - devices.data(), i);

    ibv_wr_start(qp);
    qp->wr_id = (uint64_t)(void*)callback;
    qp->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_rdma_write(qp, rkey, (uintptr_t)remoteAddress);
    ibv_wr_set_sge_list(qp, 1, &sge);
    ibv_wr_set_ud_addr(qp, dev.ahs.at(i), dev.remoteAddresses.at(i).qpNum, 0x4242);
    int error = ibv_wr_complete(qp);
    if (error) {
      fmt::fprintf(stderr, "ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
      std::fflush(stderr);
      CHECK(false);
    }
  };

  void* testBufferAddress = std::malloc(0x1000 * size);
  auto testMr = regMr((uintptr_t)testBufferAddress, 0x1000 * size);
  auto testRemote = distributeAddressAndKeys((uintptr_t)testBufferAddress, testMr);
  for (size_t i = 0; i != 100; ++i) {
    fmt::printf("rank %d starting CPU test %d!\n", rank, i);
    uint64_t* testPtr = (uint64_t*)testBufferAddress;
    std::mt19937_64 rng;
    rng.seed(rank);
    for (size_t i = 0; i != 512; ++i) {
      testPtr[512 * rank + i] = rng();
    }
    size_t remaining = 0;
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      size_t offset = 512 * rank;
      for (size_t di = 0; di != devices.size(); ++di) {
        auto& dev = devices[di];
        size_t n = 512 / devices.size();
        ++remaining;
        writeData(
            dev, i, testPtr + offset, testMr->mrs[di]->lkey, (uint64_t*)testRemote[i].address + offset,
            testRemote[i].keys[di], n * 8, new Callback{[&]() { --remaining; }});
        offset += n;
      }
    }
    while (remaining) {
      poll();
    }
    // synchronize just to ensure all ranks are finished
    setup->allgather(0);
    for (size_t i = 0; i != size; ++i) {
      rng.seed(i);
      for (size_t j = 0; j != 512; ++j) {
        uint64_t v = rng();
        if (testPtr[512 * i + j] != v) {
          fmt::printf("i %d j %d expected %#x, but got %#x\n", i, j, v, testPtr[512 * i + j]);
        }
        CHECK(testPtr[512 * i + j] == v);
      }
    }

    fmt::printf("rank %d CPU test done!\n", rank);
  }
}

} // namespace moodist

int main(int argc, const char** argv) {

  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  if (argc >= 3) {
    moodist::test(std::atoi(argv[1]), std::atoi(argv[2]), argc >= 4 ? argv[3] : "");
  } else {
    const char* rank = std::getenv("RANK");
    const char* size = std::getenv("WORLD_SIZE");
    const char* masterAddr = std::getenv("MASTER_ADDR");
    CHECK(rank != nullptr);
    CHECK(size != nullptr);
    CHECK(masterAddr != nullptr);
    moodist::test(std::atoi(rank), std::atoi(size), masterAddr);
  }

  return 0;
}