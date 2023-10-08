#include "cputhread.h"
#include "clock.h"
#include "common.h"
#include "freelist.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include "fmt/printf.h"

#include <atomic>
#include <memory>
#include <numa.h>
#include <rdma/fi_errno.h>
#include <utility>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

namespace moodist {

inline void throwFi(int error, const char* file, int line) {
  const char* str = "unknown libfabric error";
  str = fi_strerror(error);
  throw std::runtime_error(fmt::sprintf("%s:%d: libfabric error %d: %s", file, line, error, str));
}

#define CHECK_FI(x)                                                                                                    \
  {                                                                                                                    \
    int error__ = (x);                                                                                                 \
    if (error__ != FI_SUCCESS) throwFi(error__, __FILE__, __LINE__);                                                   \
  }

CpuThread::CpuThread(Group* group) {
  this->group = group;
}
CpuThread::~CpuThread() {
  if (thread.joinable()) {
    QueueEntry* e = freelistTerminate.pop();
    e->task = taskTerminate;
    std::unique_lock l(mutex);
    queue.push_back(*e);
    l.unlock();
    ++queueSize;
    futexWakeAll(&queueSize);
    terminate = true;

    thread.join();
  }
}

void CpuThread::start() {
  thread = std::thread([this] { entry(); });
}

inline void volatileMemcpy(void* dst, volatile void* src, size_t size) {
  TORCH_CHECK(size % sizeof(unsigned long) == 0);
  volatile unsigned long* s = (volatile unsigned long*)src;
  char* d = (char*)dst;
  while (size) {
    unsigned long v = *s++;
    std::memcpy(d, &v, sizeof(v));
    d += sizeof(unsigned long);
    size -= sizeof(unsigned long);
  }
}

struct QuitCpuThread {};

template<typename Buffer>
std::string tostr(const Buffer& buffer) {
  std::string r;
  for (size_t i = 0; i != buffer.size(); ++i) {
    uint8_t v = (uint8_t)buffer.data()[i];
    r += "0123456789abcdef"[v >> 4];
    r += "0123456789abcdef"[v & 0xf];
  }
  return r;
}
std::string tostr(const void* ptr, size_t len) {
  std::string r;
  for (size_t i = 0; i != len; ++i) {
    uint8_t v = ((uint8_t*)ptr)[i];
    r += "0123456789abcdef"[v >> 4];
    r += "0123456789abcdef"[v & 0xf];
  }
  return r;
}

struct Device {
  fid_fabric* fabric = nullptr;
  fid_domain* domain = nullptr;
  fid_ep* ep = nullptr;

  fi_av_attr av_attr;
  fid_av* av = nullptr;

  fi_cq_attr cq_attr;
  fid_cq* cq = nullptr;

  std::vector<char> address;

  std::vector<fi_addr_t> peerAddresses;
  uint32_t addrformat;
};

struct MemoryRegion {
  std::array<fid_mr*, 32> mrs;
};

struct alignas(64) DynamicAddresses {
  std::array<uint64_t, 32> gatherKey;
  uintptr_t gatherAddress;
};

struct alignas(64) Progress {
  uint32_t stepValue;
  std::array<size_t, 32> bytesReady;
};

void CpuThread::entry() {
  try {
    const size_t rank = group->rank;
    const size_t size = group->size;

    // CHECK_CU(cuCtxSetCurrent(group->cuContext));

    // CpuProcess& process = *group->cpuprocess;

    if (group->allocationNode != -1) {
      numa_run_on_node(group->allocationNode);
    }

    CUcontext cuCtx;
    // cuCtxCreate(&cuCtx, 0, group->cuDevice);
    // CHECK_CU(cuCtxSetCurrent(cuCtx));
    CHECK_CU(cuCtxSetCurrent(group->cuContext));

    CUstream stream;
    // CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CHECK_CU(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, -10000));

    SetupComms* setupComms = &*group->setupComms;

    std::vector<Device> devices;
    devices.reserve(0x100);

    fi_info* info = nullptr;
    fi_info* hints = fi_allocinfo();
    std::memset(hints, 0, sizeof(*hints));
    hints->caps = FI_RMA | FI_HMEM | FI_FENCE;
    CHECK_FI(fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, nullptr, &info));
    fi_freeinfo(hints);

    for (fi_info* i = info; i; i = i->next) {
      fmt::printf(" interface caps %#x\n", i->caps);
      if (i->domain_attr->mr_mode & FI_MR_ENDPOINT) {
        continue;
      }

      //if ((i->caps & FI_RMA) && (i->caps & FI_HMEM)) {
      if (i->caps & FI_RMA) {
        fmt::printf("   we can use this one!\n");

        //CHECK((i->caps & FI_HMEM) != 0);
        //CHECK((i->caps & FI_FENCE) != 0);

        devices.emplace_back();
        Device& dev = devices.back();
        CHECK_FI(fi_fabric(i->fabric_attr, &dev.fabric, nullptr));
        CHECK_FI(fi_domain(dev.fabric, i, &dev.domain, nullptr));

        // i->domain_attr->mr_mode

        fmt::printf("type is %d\n", (int)i->ep_attr->type);

        fmt::printf("inject_size is %d\n", i->tx_attr->inject_size);

        fmt::printf("i->ep_attr->tx_ctx_cnt is %d\n", i->ep_attr->tx_ctx_cnt);

        CHECK_FI(fi_endpoint(dev.domain, i, &dev.ep, nullptr));

        std::memset(&dev.av_attr, 0, sizeof(dev.av_attr));
        dev.av_attr.count = 32 * size;
        // dev.av_attr.type = FI_AV_MAP;
        dev.av_attr.type = FI_AV_TABLE;

        CHECK_FI(fi_av_open(dev.domain, &dev.av_attr, &dev.av, nullptr));
        CHECK_FI(fi_ep_bind(dev.ep, &dev.av->fid, 0));

        std::memset(&dev.cq_attr, 0, sizeof(dev.cq_attr));
        dev.cq_attr.size = 0x100;
        dev.cq_attr.format = FI_CQ_FORMAT_MSG;

        CHECK_FI(fi_cq_open(dev.domain, &dev.cq_attr, &dev.cq, nullptr));
        CHECK_FI(fi_ep_bind(dev.ep, &dev.cq->fid, FI_TRANSMIT | FI_RECV));

        CHECK_FI(fi_enable(dev.ep));

        size_t addrlen = 0x1000;
        std::array<char, 0x1000> buf;
        CHECK_FI(fi_getname(&dev.ep->fid, &buf, &addrlen));

        dev.address.resize(addrlen);
        std::memcpy(dev.address.data(), buf.data(), addrlen);

        fmt::printf("local address: address %s\n", tostr(dev.address));

        dev.addrformat = i->addr_format;
        fmt::printf("addrformat is %#x\n", i->addr_format);

        fmt::printf("i->domain_attr->mr_mode is %#x\n", i->domain_attr->mr_mode);
      }
    }

    {
      std::vector<std::vector<char>> localAddresses;
      std::vector<uint32_t> localAddrformats;
      for (auto& dev : devices) {
        localAddresses.push_back(dev.address);
        localAddrformats.push_back(dev.addrformat);
      }
      auto peerAddresses = setupComms->allgather(localAddresses);
      auto peerAddrformats = setupComms->allgather(localAddrformats);
      for (size_t di = 0; di != devices.size(); ++di) {
        auto& dev = devices[di];
        dev.peerAddresses.resize(size, -1);
        for (size_t i = 0; i != size; ++i) {
          if (i == rank) {
            continue;
          }
          auto& addrs = peerAddresses.at(i);
          CHECK(addrs.size() > di);
          if (dev.addrformat != peerAddrformats.at(i).at(di)) {
            fmt::printf("mismatching address formats :(\n");
            continue;
          }
          CHECK(fi_av_insert(dev.av, addrs.at(di).data(), 1, &dev.peerAddresses[i], 0, nullptr) == 1);
        }
      }
      // for (auto& dev : devices) {
      //   dev.peerAddresses.resize(size, -1);
      //   for (size_t i = 0; i != size; ++i) {
      //     if (i == rank) {
      //       continue;
      //     }
      //     size_t n = 0;
      //     for (auto& address : peerAddresses.at(i)) {
      //       if (dev.addrformat != peerAddrformats.at(i).at(n)) {
      //         fmt::printf("mismatching address formats :(\n");
      //         continue;
      //       }
      //       fmt::printf("inserting address %s\n", tostr(address));
      //       CHECK(fi_av_insert(dev.av, address.data(), 1, &dev.peerAddresses[n], 0, nullptr) == 1);
      //       fmt::printf("insert %d ..ok? -> %#x\n", n, dev.peerAddresses[n]);
      //       ++n;
      //     }
      //   }
      // }

      for (auto& dev : devices) {
        std::string s;
        for (uint64_t v : dev.peerAddresses) {
          s += fmt::sprintf(" %#x", v);
        }
        fmt::printf("device has peer addresses: %s\n", s);
      }
    }

    // fi_freeinfo(info);

    fmt::printf("Opened %d devices\n", devices.size());

    CHECK(!devices.empty());

    std::atomic_uint32_t* myStepCounter = (std::atomic_uint32_t*)group->ipcMapper->getMySharedMem(0x100, 4);

    HashMap<uintptr_t, std::unique_ptr<MemoryRegion>> mrMap;
    HashMap<uintptr_t, std::unique_ptr<MemoryRegion>> mrMapCuda;

    auto regMr = [&](uintptr_t address, size_t bytes) {
      auto i = mrMap.find(address);
      if (i != mrMap.end()) {
        return &*i->second;
      }
      auto ptr = std::make_unique<MemoryRegion>();
      ptr->mrs.fill(nullptr);
      CHECK(devices.size() <= ptr->mrs.size());
      for (size_t i = 0; i != devices.size(); ++i) {
        CHECK_FI(fi_mr_reg(
            devices[i].domain, (void*)address, bytes, FI_WRITE | FI_REMOTE_WRITE, 0, 0, 0, &ptr->mrs[i], nullptr));
      }
      MemoryRegion* r = &*ptr;
      mrMap[address] = std::move(ptr);
      return r;
    };

    auto regMrCuda = [&](uintptr_t address, size_t bytes) {
      auto i = mrMapCuda.find(address);
      if (i != mrMapCuda.end()) {
        return &*i->second;
      }
      auto ptr = std::make_unique<MemoryRegion>();
      ptr->mrs.fill(nullptr);
      CHECK(devices.size() <= ptr->mrs.size());
      for (size_t i = 0; i != devices.size(); ++i) {
        iovec iov;
        fi_mr_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        iov.iov_base = (void*)address;
        iov.iov_len = bytes;
        attr.access = FI_WRITE | FI_REMOTE_WRITE;
        attr.iface = FI_HMEM_CUDA;
        attr.device.cuda = group->cuDevice;
        fmt::printf("doing cuda register\n");

        // int fd;
        // CHECK_CU(cuMemGetHandleForAddressRange(&fd, address, bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

        CHECK_FI(fi_mr_regattr(devices[i].domain, &attr, FI_HMEM | FI_HMEM_DEVICE_ONLY, &ptr->mrs[i]));
        fmt::printf("cuda register ok!\n");
      }
      MemoryRegion* r = &*ptr;
      mrMapCuda[address] = std::move(ptr);
      return r;
    };

    size_t activeWrs = 0;

    auto poll = [&]() {
      std::array<fi_cq_err_entry, 32> arr;
      for (auto& dev : devices) {
        int r = fi_cq_read(dev.cq, arr.data(), 32);
        if (r > 0) {
          CHECK(activeWrs >= r);
          activeWrs -= r;
          // fmt::printf("got completion x%d active wrs -> %d\n", r, activeWrs);
        }
        if (r >= 0 || r == -FI_EAGAIN) {
          continue;
        }
        if (r == -FI_EAVAIL) {
          r = fi_cq_readerr(dev.cq, arr.data(), 0);
          fmt::printf("got completion with error: %#x\n", arr[0].tag);
          throw std::runtime_error("completion error");
        }
        CHECK_FI(-r);
        CHECK(false);
      }
    };

    AllocatedBuffer commsBuffer = group->allocateHost(0x100 * size);
    auto* commsMr = regMr((uintptr_t)commsBuffer.cpuPointer, commsBuffer.bytes);

    DynamicAddresses* localDyns = (DynamicAddresses*)commsBuffer.cpuPointer;

    size_t progressOffset = sizeof(DynamicAddresses) * size;
    Progress* localProgress = (Progress*)(void*)((uintptr_t)commsBuffer.cpuPointer + progressOffset);

    std::vector<std::vector<uint64_t>> peerCommsKeys;
    std::vector<uintptr_t> peerCommsAddrs;

    {
      std::vector<uint64_t> localCommsKeys;
      for (size_t i = 0; i != devices.size(); ++i) {
        localCommsKeys.push_back(commsMr->mrs.at(i)->key);
      }
      peerCommsKeys = setupComms->allgather(localCommsKeys);
      peerCommsAddrs = setupComms->allgather((uintptr_t)commsBuffer.cpuPointer);

      for (auto& v : peerCommsKeys) {
        CHECK(v.size() == devices.size());
      }
    }

    std::vector<size_t> recvRanks;
    std::vector<size_t> sendRanks;
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      recvRanks.push_back(i);
      sendRanks.push_back(i);
    }

    auto write = [&](fid_ep* ep, const void* buf, size_t len, void* desc, fi_addr_t dest_addr, uint64_t addr,
                     uint64_t key) {
      ++activeWrs;
      // fmt::printf("write, active wrs -> %d\n", activeWrs);
      while (true) {
        int err = fi_write(ep, buf, len, desc, dest_addr, addr, key, nullptr);
        if (err == -FI_EAGAIN) {
          poll();
          continue;
        }
        CHECK_FI(err);
        break;
      }
    };
    auto injectWrite = [&](fid_ep* ep, uintptr_t srcAddr, size_t len, fi_addr_t peerAddr, uint64_t dstAddr,
                           uint64_t dstKey) {
      ++activeWrs;
      // fmt::printf("inject write, active wrs -> %d\n", activeWrs);
      while (true) {
        fi_rma_iov rma_iov;
        iovec iov;
        fi_msg_rma msg;
        void* desc = nullptr;
        iov.iov_base = (void*)srcAddr;
        iov.iov_len = len;
        rma_iov.addr = dstAddr;
        rma_iov.len = len;
        rma_iov.key = dstKey;
        msg.msg_iov = &iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.addr = peerAddr;
        msg.rma_iov = &rma_iov;
        msg.rma_iov_count = 1;
        msg.context = nullptr;
        msg.data = 0;
        int err = fi_writemsg(ep, &msg, FI_INJECT);
        if (err == -FI_EAGAIN) {
          poll();
          continue;
        }
        CHECK_FI(err);
        break;
      }
    };
    auto injectWriteFenced = [&](fid_ep* ep, uintptr_t srcAddr, size_t len, fi_addr_t dest_addr, uint64_t dstAddr,
                                 uint64_t key) {
      CHECK(dest_addr != -1);
      ++activeWrs;
      // fmt::printf("inject write fenced, active wrs -> %d\n", activeWrs);
      while (true) {
        fi_rma_iov rma_iov;
        iovec iov;
        fi_msg_rma msg;
        void* desc = nullptr;
        iov.iov_base = (void*)srcAddr;
        iov.iov_len = len;
        rma_iov.addr = dstAddr;
        rma_iov.len = len;
        rma_iov.key = key;
        msg.msg_iov = &iov;
        msg.desc = &desc;
        msg.iov_count = 1;
        msg.addr = dest_addr;
        msg.rma_iov = &rma_iov;
        msg.rma_iov_count = 1;
        msg.context = nullptr;
        msg.data = 0;
        // int err = fi_writemsg(ep, &msg, FI_INJECT | FI_FENCE);
        int err = fi_writemsg(ep, &msg, FI_INJECT);
        if (err == -FI_EAGAIN) {
          poll();
          continue;
        }
        CHECK_FI(err);
        break;
      }
    };
    auto writeData = [&](size_t dst, uintptr_t srcAddr, size_t len, MemoryRegion* mr, uint64_t dstAddr,
                         const std::array<uint64_t, 32>& key, size_t bytesReady) {
      size_t offset = 0;
      const size_t chunks = devices.size();
      const size_t chunkSize = len / chunks;
      for (size_t i = 0; i != chunks; ++i) {
        size_t n = i == chunks - 1 ? len - offset : chunkSize;
        auto& dev = devices[i];
        if (n != 0) {
          write(
              dev.ep, (void*)(srcAddr + offset), n, mr->mrs[i]->mem_desc, dev.peerAddresses[dst], dstAddr + offset,
              key[i]);
          offset += n;
        }
        uintptr_t dstOffset = (uintptr_t)(void*)&localProgress[rank].bytesReady[i] - (uintptr_t)commsBuffer.cpuPointer;
        injectWriteFenced(
            dev.ep, (uintptr_t)&bytesReady, sizeof(bytesReady), dev.peerAddresses[dst], peerCommsAddrs[dst] + dstOffset,
            peerCommsKeys[dst][i]);
      }
    };
    auto writeStep = [&](size_t deviceIndex, size_t dst, uint32_t stepValue) {
      Device& dev = devices[deviceIndex];
      uintptr_t dstOffset = (uintptr_t)(void*)&localProgress[rank].stepValue - (uintptr_t)commsBuffer.cpuPointer;
      injectWriteFenced(
          dev.ep, (uintptr_t)&stepValue, sizeof(stepValue), dev.peerAddresses[dst], peerCommsAddrs[dst] + dstOffset,
          peerCommsKeys[dst][deviceIndex]);
    };

    std::vector<size_t> bytesReadyOffset;
    bytesReadyOffset.resize(size);
    std::vector<size_t> bytesSentOffset;
    bytesSentOffset.resize(size);

    while (true) {

      // if (op.queueSize != 0) {
      //   fmt::printf("another op queued immediately!\n");
      // }
      while (queueSize == 0) {
        poll();
        if (activeWrs == 0) {
          futexWait(&queueSize, 0, std::chrono::seconds(1));
        }
      }
      --queueSize;

      std::unique_lock l(mutex);
      TORCH_CHECK(!queue.empty());
      QueueEntry& queueEntry = queue.front();
      queue.pop_front();
      l.unlock();

      TORCH_CHECK(queueEntry.stepValue < (1ul << 31));

      if (queueEntry.task == taskAllgather) {
        QueueEntryAllGather& params = (QueueEntryAllGather&)queueEntry;

        fmt::printf("cpu thread got all gather (step %d)\n", params.stepValue);

        auto* inputMr = regMrCuda(params.inputAddress, params.bytes);
        auto* outputMr = regMrCuda(params.outputAddress, params.bytes * size);

        DynamicAddresses* mydyn = &localDyns[rank];
        for (size_t i = 0; i != devices.size(); ++i) {
          mydyn->gatherKey[i] = outputMr->mrs[i]->key;
        }
        mydyn->gatherAddress = (uintptr_t)params.outputAddress;

        // fmt::printf("mydyn->gatherAddress is %#x\n", mydyn->gatherAddress);
        // fmt::printf("mydyn->gatherKey is %#s\n", tostr(&mydyn->gatherKey, sizeof(mydyn->gatherKey)));

        size_t n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          injectWrite(
              dev.ep, (uintptr_t)(void*)mydyn, sizeof(*mydyn), dev.peerAddresses[i],
              peerCommsAddrs[i] + sizeof(DynamicAddresses) * rank, peerCommsKeys[i][di]);
          // writeStep(di, i, params.stepValue);
          ++n;
        }
        while (activeWrs) {
          poll();
        }
        n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          writeStep(di, i, params.stepValue);
          ++n;
        }

        fmt::printf("sent dyns\n");

        for (size_t i : sendRanks) {
          while (localProgress[i].stepValue < params.stepValue) {
            poll();
          }
          // fmt::printf("localDyns[i].gatherAddress is %#x\n", localDyns[i].gatherAddress);
          // fmt::printf(
          //     "localDyns[i].gatherKey is %#s\n", tostr(&localDyns[i].gatherKey, sizeof(localDyns[i].gatherKey)));
        }

        fmt::printf("got all dyns!\n");

        auto start = std::chrono::steady_clock::now();

        size_t bytesSent = 0;
        size_t maxSize = std::max(params.bytes / 128, (size_t)(1024 * 1024));
        while (bytesSent < params.bytes) {
          size_t bytesReady = params.inputBytesReady.load(std::memory_order_relaxed);
          while (bytesReady == bytesSent) {
            bytesReady = params.inputBytesReady.load(std::memory_order_relaxed);
          }
          size_t offset = bytesSent;
          size_t n = std::min(bytesReady - bytesSent, maxSize);

          uintptr_t srcAddr = (uintptr_t)params.inputAddress + offset;

          bytesSent += n;

          for (size_t i : sendRanks) {
            writeData(
                i, srcAddr, n, inputMr, localDyns[i].gatherAddress + rank * params.bytes + offset,
                localDyns[i].gatherKey, bytesSentOffset[i] + bytesSent);
          }
          poll();
          // fmt::printf("sent %d  -> %d/%d\n", n, bytesSent, params.bytes);
        }
        CHECK(bytesSent == params.bytes);

        while (activeWrs) {
          poll();
        }
        for (size_t i : sendRanks) {
          writeData(i, 0, 0, inputMr, 0, localDyns[i].gatherKey, bytesSentOffset[i] + bytesSent);
          bytesSentOffset[i] += bytesSent;
        }

        fmt::printf(
            "%fG/s\n", bytesSent / seconds(std::chrono::steady_clock::now() - start) / 1024.0f / 1024.0f / 1024.0f);

        fmt::printf("sent all data!\n");

        while (true) {
          poll();
          bool done = true;
          for (size_t i : recvRanks) {
            for (size_t di = 0; di != devices.size(); ++di) {
              size_t bytesReceived = localProgress[i].bytesReady[di] - bytesReadyOffset[i];
              if (bytesReceived > params.bytes) {
                fmt::printf("localProgress[%d].bytesReady[%d] is %d\n", i, di, localProgress[i].bytesReady[di]);
                fmt::printf("bytesReadyOffset[%d] is %d\n", i, bytesReadyOffset[i]);
              }
              CHECK(bytesReceived <= params.bytes);
              // fmt::printf("[%d, %d] received %d/%d\n", i, di, bytesReceived, params.bytes);
              if (bytesReceived < params.bytes) {
                done = false;
              }
            }
          }
          if (done) {
            break;
          }
        }

        // futexWaitWhileLess(myStepCounter, queueEntry.stepValue + size);

        fmt::printf("cpu thread all gather step 1 done!\n");

        myStepCounter->store(params.stepValue + 1, std::memory_order_relaxed);
        futexWakeAll(myStepCounter);

        for (size_t i : recvRanks) {
          bytesReadyOffset[i] += params.bytes;
        }

        // futexWaitWhileLess(myStepCounter, queueEntry.stepValue + size * 2);

        fmt::printf("cpu thread all gather all done!\n");

        freelistAllGather.push(&params);
      } else if (queueEntry.task == taskTerminate) {
        freelistTerminate.push(&queueEntry);
        break;
      } else {
        throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
      }
    }

  } catch (const std::exception& e) {
    fmt::fprintf(stderr, "Error: %s\n", e.what());
    fflush(stderr);
    TORCH_CHECK(false, "Moodist cpu thread got an exception");
  } catch (QuitCpuThread) {
  }
}

} // namespace moodist
