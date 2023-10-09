#include "cputhread.h"
#include "allgather.h"
#include "clock.h"
#include "common.h"
#include "freelist.h"
#include "ib_common.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include "fmt/printf.h"

#include <atomic>
#include <infiniband/verbs.h>
#include <libibverbs/verbs.h>
#include <memory>
#include <numa.h>
#include <utility>

namespace moodist {

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
  CHECK(size % sizeof(unsigned long) == 0);
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

// struct Device {
//   fid_fabric* fabric = nullptr;
//   fid_domain* domain = nullptr;
//   fid_ep* ep = nullptr;

//   fi_av_attr av_attr;
//   fid_av* av = nullptr;

//   fi_cq_attr cq_attr;
//   fid_cq* cq = nullptr;

//   std::vector<char> address;

//   std::vector<fi_addr_t> peerAddresses;
//   uint32_t addrformat;
// };

struct MemoryRegion {
  std::array<ibv_mr*, 32> mrs;
};

// struct alignas(64) DynamicAddresses {
//   std::array<uint64_t, 32> gatherKey;
//   uintptr_t gatherAddress;
// };

struct Device {
  IbCommon* ib;
  ibv_pd* protectionDomain;
  ibv_cq* cq;
  std::vector<ibv_qp*> qps;
  size_t currentCqEntries = 0;
  std::vector<size_t> currentWr;
  std::vector<size_t> signaledWr;
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
    for (auto& v : group->ibDevs) {
      devices.emplace_back();
      auto& dev = devices.back();
      dev.ib = &*v;
      dev.protectionDomain = v->protectionDomain;
      dev.cq = v->cq;
      for (auto& qp : v->qps) {
        dev.qps.push_back(&*qp);
      }
      dev.currentWr.resize(size);
      dev.signaledWr.resize(size);
    }

    // ibv_pd* protectionDomain = ib->protectionDomain;
    // ibv_cq* cq = ib->cq;
    // ibv_qp* qp = ib->qp;

    // fmt::printf("rank %d got all syncs\n", rank);

    const size_t maxWr = IbCommon::maxWr;
    const size_t maxCqEntries = IbCommon::maxCqEntries;

    auto poll = [&]() {
      if (terminate.load(std::memory_order_relaxed)) {
        throw QuitCpuThread();
      }
      for (auto& dev : devices) {
        ibv_wc wcs[4];
        int n = ibv_poll_cq(dev.cq, 4, wcs);
        CHECK(n >= 0);
        for (size_t i = 0; i != n; ++i) {
          ibv_wc& wc = wcs[i];
          // fmt::fprintf(
          //     stderr, "rank %d Work completion with status %d (opcode %d, id %d)\n", rank, wc.status, wc.opcode,
          //     wc.wr_id);
          // std::fflush(stderr);
          if (wc.status) {
            fmt::fprintf(
                stderr, "rank %d Work completion with status %d (opcode %d, id %d)\n", rank, wc.status, wc.opcode,
                wc.wr_id);
            std::fflush(stderr);
            CHECK(false);
          } else {
            size_t index = wc.wr_id / maxWr;
            size_t wrCount = wc.wr_id % maxWr;
            fflush(stdout);
            CHECK(dev.currentWr[index] >= wrCount);
            CHECK(dev.signaledWr[index] >= wrCount);
            dev.currentWr[index] -= wrCount;
            dev.signaledWr[index] -= wrCount;
            CHECK(dev.currentCqEntries > 0);
            --dev.currentCqEntries;

            // fmt::printf(
            //     "%p (size %d) :: poll wr_id %#x index %d wrCount %d currentWr[%d] %d, signaledWr[%d] %d, "
            //     "currentCqEntries %d\n",
            //     (void*)this, group->size, wc.wr_id, index, wrCount, index, currentWr[index], index,
            //     signaledWr[index], currentCqEntries);
          }
        }
      }
    };

    int prevPollLine = -1;
    auto pollStart = Clock::now();
    auto debugPoll = [&](int line) {
      auto now = Clock::now();
      if (prevPollLine != line) {
        prevPollLine = line;
        pollStart = now;
      } else {
        if (now - pollStart >= std::chrono::seconds(30)) {
          fmt::printf("rank %d of group with size %d stuck at line %d\n", rank, size, line);
          pollStart = now;
        }
      }
      poll();
    };

    // #define poll() debugPoll(__LINE__)

    auto preSend = [&](Device& dev, size_t index, auto& wr) {
      while (dev.currentWr[index] == maxWr || dev.currentCqEntries == maxCqEntries) {
        poll();
      }
      size_t n = ++dev.currentWr[index];
      wr.wr_id = index * maxWr;
      if (n == (maxWr + 1) / 2 - 1) {
        wr.send_flags |= IBV_SEND_SIGNALED;
      }
      if (wr.send_flags & IBV_SEND_SIGNALED) {
        n -= dev.signaledWr[index];
        wr.wr_id += n;
        dev.signaledWr[index] += n;
        ++dev.currentCqEntries;

        // fmt::printf(
        //     "%p (size %d) :: preSend n %d currentWr[%d] %d, signaledWr[%d] %d, currentCqEntries %d\n", (void*)this,
        //     group->size, n, index, currentWr[index], index, signaledWr[index], currentCqEntries);
      }
    };

    size_t bytesWritten = 0;

    bool forceSignalNext = false;
    auto writeData = [&](Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
                         size_t bytes) {
      CHECK(i >= 0 && i < size);
      CHECK(i != rank);
      auto post = [&]() {
        ibv_send_wr wr;
        ibv_sge sge;
        sge.addr = (uintptr_t)localAddress;
        sge.length = bytes;
        sge.lkey = lkey;

        bytesWritten += bytes;

        std::memset(&wr, 0, sizeof(wr));
        wr.next = nullptr;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        // wr.send_flags = lkey == 0 && bytes <= 32 ? IBV_SEND_INLINE : 0;
        wr.send_flags = 0;
        if (forceSignalNext) {
          wr.send_flags |= IBV_SEND_SIGNALED;
          forceSignalNext = false;
        }
        wr.wr.rdma.remote_addr = (uintptr_t)remoteAddress;
        wr.wr.rdma.rkey = rkey;

        wr.send_flags |= IBV_SEND_SIGNALED;
        preSend(dev, i, wr);

        ibv_qp_ex* qp = dev.ib->qpexs.at(i);

        // fmt::printf(
        //     "rdma write %d bytes (%p -> %p) (dev %d, i %d)\n", bytes, localAddress, remoteAddress,
        //     &dev - devices.data(), i);

        ibv_wr_start(qp);
        qp->wr_id = wr.wr_id;
        qp->wr_flags = wr.send_flags;
        ibv_wr_rdma_write(qp, rkey, (uintptr_t)remoteAddress);
        // if (wr.send_flags & IBV_SEND_INLINE) {
        //   ibv_wr_set_inline_data(qp, localAddress, bytes);
        // }
        ibv_wr_set_sge_list(qp, 1, &sge);
        ibv_wr_set_ud_addr(qp, dev.ib->ahs.at(i), dev.ib->remoteAddresses.at(i).qpNum, 0x4242);
        int error = ibv_wr_complete(qp);
        if (error) {
          fmt::fprintf(stderr, "ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
          std::fflush(stderr);
          CHECK(false);
        }

        // fmt::printf(
        //     "%p (size %d) :: rank %d -> %d (signal %d) write %sdata of %#x bytes from local %#x (lkey %#x) to remote
        //     "
        //     "%#x (rkey %#x)\n",
        //     (void*)this, group->size, rank, i, (wr.send_flags & IBV_SEND_SIGNALED) ? 1 : 0,
        //     wr.send_flags & IBV_SEND_INLINE ? "INLINE " : "", sge.length, sge.addr, sge.lkey, wr.wr.rdma.remote_addr,
        //     wr.wr.rdma.rkey);

        // preSend(dev, i, wr);
        // ibv_send_wr* bad = nullptr;
        // int error = ibv_post_send(dev.qps[i], &wr, &bad);
        // if (error) {
        //   fmt::fprintf(stderr, "ibv_post_send failed with error %d: %s\n", error, std::strerror(error));
        //   std::fflush(stderr);
        //   CHECK(false);
        // }
      };

      if (bytes >= 1024ull * 1024 * 896) {
        size_t remaining = bytes;
        bytes = 1024ull * 1024 * 768;
        bool signal = forceSignalNext;
        while (remaining > 0) {
          if (remaining < 1024ull * 1024 * 896) {
            bytes = remaining;
          }
          forceSignalNext = signal;
          post();

          localAddress = (void*)((uintptr_t)localAddress + bytes);
          remoteAddress = (void*)((uintptr_t)remoteAddress + bytes);
          remaining -= bytes;
        }
      } else {
        post();
      }
    };

    // std::atomic_uint32_t* myStepCounter = (std::atomic_uint32_t*)group->ipcMapper->getMySharedMem(0x100, 4);

    HashMap<uintptr_t, std::unique_ptr<MemoryRegion>> mrMap;
    HashMap<uintptr_t, std::unique_ptr<MemoryRegion>> mrMapCuda;

    std::vector<IbvMr> localMrs;

    auto regMr = [&](uintptr_t address, size_t bytes) {
      auto i = mrMap.find(address);
      if (i != mrMap.end()) {
        return &*i->second;
      }
      auto ptr = std::make_unique<MemoryRegion>();
      ptr->mrs.fill(nullptr);
      CHECK(devices.size() <= ptr->mrs.size());
      for (size_t i = 0; i != devices.size(); ++i) {
        IbvMr mr = ibv_reg_mr(
            devices[i].ib->protectionDomain, (void*)address, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr) {
          fmt::printf("rank %d failed to CPU register %#x size %#x\n", group->rank, address, bytes);
          perror("ibv_reg_mr");
          TORCH_CHECK(false);
        }
        ptr->mrs[i] = mr;
        localMrs.push_back(std::move(mr));
      }
      MemoryRegion* r = &*ptr;
      mrMap[address] = std::move(ptr);
      return r;
    };

    auto regMrCuda = [&](uintptr_t address, size_t bytes) {
      unsigned long long bufferId = -1;
      CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
      CHECK(bufferId != -1);
      auto i = mrMapCuda.find(bufferId);
      if (i != mrMapCuda.end()) {
        return &*i->second;
      }
      auto ptr = std::make_unique<MemoryRegion>();
      ptr->mrs.fill(nullptr);
      CHECK(devices.size() <= ptr->mrs.size());
      for (size_t i = 0; i != devices.size(); ++i) {
        ptr->mrs[i] = devices[i].ib->getMr(address, bytes);
      }
      MemoryRegion* r = &*ptr;
      mrMapCuda[bufferId] = std::move(ptr);
      return r;
    };

    auto* commsMr = regMr((uintptr_t)group->mySharedMem, group->mySharedMemSize);

    DynamicAddresses* localDyns = group->localDyns;
    Progress* localProgress = group->localProgress;

    std::vector<std::vector<uint32_t>> peerCommsKeys;
    std::vector<uintptr_t> peerCommsAddrs;

    {
      std::vector<uint32_t> localCommsKeys;
      for (size_t i = 0; i != devices.size(); ++i) {
        localCommsKeys.push_back(commsMr->mrs.at(i)->rkey);
      }
      peerCommsKeys = setupComms->allgather(localCommsKeys);
      peerCommsAddrs = setupComms->allgather((uintptr_t)group->mySharedMem);

      for (auto& v : peerCommsKeys) {
        CHECK(v.size() == devices.size());
      }
    }

    auto writeDataDistributed = [&](size_t dst, uintptr_t srcAddr, size_t len, MemoryRegion* mr, uint64_t dstAddr,
                                    const std::array<uint32_t, 32>& key, size_t bytesReady) {
      size_t offset = 0;
      const size_t chunks = devices.size();
      const size_t chunkSize = len / chunks;
      for (size_t i = 0; i != chunks; ++i) {
        size_t n = i == chunks - 1 ? len - offset : chunkSize;
        auto& dev = devices[i];
        if (n != 0) {
          writeData(dev, dst, (void*)(srcAddr + offset), mr->mrs[i]->lkey, (void*)(dstAddr + offset), key[i], n);
        }
        offset += n;
        // uintptr_t dstOffset = (uintptr_t)(void*)&localProgress[rank].bytesReady[i] -
        // (uintptr_t)commsBuffer.cpuPointer; injectWriteFenced(
        //     dev.ep, (uintptr_t)&bytesReady, sizeof(bytesReady), dev.peerAddresses[dst], peerCommsAddrs[dst] +
        //     dstOffset, peerCommsKeys[dst][i]);
      }
    };

    auto writeStep = [&](size_t deviceIndex, size_t dst, uint32_t stepValue) {
      Device& dev = devices[deviceIndex];
      // uintptr_t dstOffset = (uintptr_t)(void*)&localProgress[rank].stepValue - (uintptr_t)commsBuffer.cpuPointer;
      uintptr_t dstOffset = group->getSharedOffset(&localProgress[rank].stepValue);
      localProgress[rank].stepValue = stepValue;
      forceSignalNext = true;
      writeData(
          dev, dst, &localProgress[rank].stepValue, commsMr->mrs.at(deviceIndex)->lkey,
          (void*)(peerCommsAddrs[dst] + dstOffset), peerCommsKeys[dst][deviceIndex], sizeof(stepValue));
      // writeData(
      //     dev, dst, &stepValue, 0, (void*)(peerCommsAddrs[dst] + dstOffset), peerCommsKeys[dst][deviceIndex],
      //     sizeof(stepValue));
    };

    volatile uint32_t* cpuIn = (uint32_t*)group->cpuInBuffer.cpuPointer;
    volatile uint32_t* cpuOut = (uint32_t*)group->cpuOutBuffer.cpuPointer;

    // std::vector<size_t> bytesReadyOffset;
    // bytesReadyOffset.resize(size);
    // std::vector<size_t> bytesSentOffset;
    // bytesSentOffset.resize(size);

    while (true) {

      // if (op.queueSize != 0) {
      //   fmt::printf("another op queued immediately!\n");
      // }
      while (queueSize == 0) {
        poll();
        int sum = 0;
        for (auto& dev : devices) {
          sum += dev.currentCqEntries;
        }
        if (sum == 0) {
          futexWait(&queueSize, 0, std::chrono::seconds(1));
        }
      }
      --queueSize;

      std::unique_lock l(mutex);
      CHECK(!queue.empty());
      QueueEntry& queueEntry = queue.front();
      queue.pop_front();
      l.unlock();

      CHECK(queueEntry.stepValue < (1ul << 31));

      if (queueEntry.task == taskAllgather) {
        QueueEntryAllGather& params = (QueueEntryAllGather&)queueEntry;

        AllGather& allGather = *group->allGather;

        auto& sendRanks = allGather.sendRanks;
        auto& recvRanks = allGather.recvRanks;

        // fmt::printf("cpu thread got all gather (step %d)\n", params.stepValue);

        auto* inputMr = regMrCuda(params.inputAddress, params.bytes);
        auto* outputMr = regMrCuda(params.outputAddress, params.bytes * size);

        DynamicAddresses* mydyn = &localDyns[rank];
        for (size_t i = 0; i != devices.size(); ++i) {
          mydyn->gatherKey[i] = outputMr->mrs[i]->rkey;
        }
        mydyn->gatherAddress = (uintptr_t)params.outputAddress;

        // fmt::printf("mydyn->gatherAddress is %#x\n", mydyn->gatherAddress);
        // fmt::printf("mydyn->gatherKey is %#s\n", tostr(&mydyn->gatherKey, sizeof(mydyn->gatherKey)));

        size_t n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          writeData(
              dev, i, mydyn, commsMr->mrs[di]->lkey, (void*)(peerCommsAddrs[i] + sizeof(DynamicAddresses) * rank),
              peerCommsKeys[i][di], sizeof(*mydyn));
          // writeStep(di, i, params.stepValue);
          //  injectWrite(
          //      dev.ep, (uintptr_t)(void*)mydyn, sizeof(*mydyn), dev.peerAddresses[i],
          //      peerCommsAddrs[i] + sizeof(DynamicAddresses) * rank, peerCommsKeys[i][di]);
          //  writeStep(di, i, params.stepValue);
          ++n;
        }
        n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          while (dev.currentCqEntries) {
            poll();
          }
          writeStep(di, i, params.stepValue);
          // injectWrite(
          //     dev.ep, (uintptr_t)(void*)mydyn, sizeof(*mydyn), dev.peerAddresses[i],
          //     peerCommsAddrs[i] + sizeof(DynamicAddresses) * rank, peerCommsKeys[i][di]);
          // writeStep(di, i, params.stepValue);
          ++n;
        }
        // while (activeWrs) {
        //   poll();
        // }
        // n = 0;
        // for (size_t i : recvRanks) {
        //   size_t di = (rank + n) % devices.size();
        //   auto& dev = devices[di];
        //   writeStep(di, i, params.stepValue);
        //   ++n;
        // }

        // fmt::printf("sent dyns\n");

        for (size_t i : sendRanks) {
          while (localProgress[i].stepValue < params.stepValue) {
            poll();
          }
          // fmt::printf("localDyns[i].gatherAddress is %#x\n", localDyns[i].gatherAddress);
          // fmt::printf(
          //     "localDyns[i].gatherKey is %#s\n", tostr(&localDyns[i].gatherKey, sizeof(localDyns[i].gatherKey)));
        }

        // fmt::printf("got all dyns!\n");

        auto start = std::chrono::steady_clock::now();

        size_t bytesSent = 0;
        // size_t maxSize = std::max(params.bytes / 4, (size_t)(1024 * 1024));
        size_t maxSize = params.bytes;
        while (bytesSent < params.bytes) {
          // size_t bytesReady = params.inputBytesReady.load(std::memory_order_relaxed);
          // while (bytesReady == bytesSent) {
          //   bytesReady = params.inputBytesReady.load(std::memory_order_relaxed);
          // }
          size_t bytesReady = params.bytes;
          size_t offset = bytesSent;
          size_t n = std::min(bytesReady - bytesSent, maxSize);

          uintptr_t srcAddr = (uintptr_t)params.inputAddress + offset;

          bytesSent += n;

          for (size_t i : sendRanks) {
            writeDataDistributed(
                i, srcAddr, n, inputMr, localDyns[i].gatherAddress + rank * params.bytes + offset,
                localDyns[i].gatherKey, bytesSent);
            // writeDataDistributed(
            //     i, srcAddr, n, inputMr, localDyns[i].gatherAddress + rank * params.bytes + offset,
            //     localDyns[i].gatherKey, bytesSentOffset[i] + bytesSent);
          }
          poll();
          // fmt::printf("sent %d  -> %d/%d\n", n, bytesSent, params.bytes);
        }
        CHECK(bytesSent == params.bytes);

        for (auto& dev : devices) {
          while (dev.currentCqEntries) {
            poll();
          }
        }
        for (size_t i : sendRanks) {
          // writeData(i, 0, 0, inputMr, 0, localDyns[i].gatherKey, bytesSentOffset[i] + bytesSent);
          // bytesSentOffset[i] += bytesSent;
        }

        // fmt::printf(
        //     "%fG/s\n", bytesWritten / seconds(std::chrono::steady_clock::now() - start) / 1024.0f / 1024.0f /
        //     1024.0f);
        // bytesWritten = 0;

        // fmt::printf("sent all data!\n");

        for (size_t i : sendRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          writeStep(di, i, params.stepValue + 1);
        }
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          while (localProgress[i].stepValue < params.stepValue + 1) {
            poll();
          }
        }

        // while (true) {
        //   poll();
        //   bool done = true;
        //   for (size_t i : recvRanks) {
        //     for (size_t di = 0; di != devices.size(); ++di) {
        //       // size_t bytesReceived = localProgress[i].bytesReady[di] - bytesReadyOffset[i];
        //       // if (bytesReceived > params.bytes) {
        //       //   fmt::printf("localProgress[%d].bytesReady[%d] is %d\n", i, di, localProgress[i].bytesReady[di]);
        //       //   fmt::printf("bytesReadyOffset[%d] is %d\n", i, bytesReadyOffset[i]);
        //       // }
        //       // CHECK(bytesReceived <= params.bytes);
        //       // // fmt::printf("[%d, %d] received %d/%d\n", i, di, bytesReceived, params.bytes);
        //       // if (bytesReceived < params.bytes) {
        //       //   done = false;
        //       // }
        //     }
        //   }
        //   if (done) {
        //     break;
        //   }
        // }

        // futexWaitWhileLess(myStepCounter, queueEntry.stepValue + size);

        // fmt::printf("cpu thread all gather step 1 done!\n");

        // params.threadStepValue.store(params.stepValue + 1, std::memory_order_relaxed);
        // futexWakeAll(&params.threadStepValue);
        // myStepCounter->store(params.stepValue + 1, std::memory_order_relaxed);
        // futexWakeAll(myStepCounter);

        // for (size_t i : recvRanks) {
        //   bytesReadyOffset[i] += params.bytes;
        // }

        // futexWaitWhileLess(myStepCounter, queueEntry.stepValue + size * 2);

        // fmt::printf("cpu thread all gather all done!\n");

        cpuOut[0] = queueEntry.stepValue;
        while (cpuIn[0] < queueEntry.stepValue + 1) {
          poll();
        }

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
    throw std::runtime_error("Moodist cpu thread got an exception");
  } catch (QuitCpuThread) {
  }
}

} // namespace moodist
