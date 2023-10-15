#include "cputhread.h"
#include "allgather.h"
#include "async.h"
#include "clock.h"
#include "common.h"
#include "freelist.h"
#include "ib_common.h"
#include "ipc_mapper.h"
#include "reduce_scatter.h"
#include "setup_comms.h"

#include "fmt/printf.h"

#include <atomic>
#include <libibverbs/verbs.h>
#include <memory>
#include <numa.h>
#include <random>
#include <utility>

namespace moodist {

extern bool profilingEnabled;

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
  ibv_qp* qp;
  size_t currentCqEntries = 0;
  // std::vector<size_t> currentWr;
  // std::vector<size_t> signaledWr;
};

struct Callback {
  int partsLeft = 0;
  Function<void()> onComplete;
};

template<typename F>
struct CallbackWrapper {
  Callback* callback;
  F decref;
  CallbackWrapper(Callback* callback, F decref) : callback(callback), decref(decref) {
    ++callback->partsLeft;
  }
  ~CallbackWrapper() {
    decref(callback);
  }
  operator Callback*() {
    return callback;
  }
};

template<typename T, size_t poolSize = 0x1000>
struct PoolAllocatorReused {
  std::deque<std::vector<T>> all;
  size_t i0 = 0;
  size_t i1 = 0;
  PoolAllocatorReused() {
    all.emplace_back();
    all.back().resize(poolSize);
  }
  T* allocate() {
    CHECK(i0 < all.size());
    CHECK(i1 < poolSize);
    T* r = &all[i0][i1];
    ++i1;
    if (i1 == poolSize) {
      i1 = 0;
      ++i0;
      if (i0 == all.size()) {
        all.emplace_back();
        all.back().resize(poolSize);
      }
    }
    return r;
  }
  void reset() {
    i0 = 0;
    i1 = 0;
  }
};

void CpuThread::entry() {
  try {
    const size_t rank = group->rank;
    const size_t size = group->size;

    async::setCurrentThreadName("moodist-cputhread");

    // if (group->allocationNode != -1) {
    //   numa_run_on_node(group->allocationNode);
    // }

    CUcontext cuCtx;
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
      dev.qp = v->qp;
    }

    const size_t maxWr = IbCommon::maxWr;
    const size_t maxCqEntries = IbCommon::maxCqEntries;

    PoolAllocatorReused<Callback, 0x100> callbackAllocator;
    std::vector<Callback*> callbackFreeList;

    auto callbackDecref = [&](Callback* callback) {
      CHECK(callback->partsLeft >= 1);
      if (--callback->partsLeft == 0) {
        std::move(callback->onComplete)();
        callbackFreeList.push_back(callback);
      }
    };

    auto makeCallback = [&](Function<void()> f) {
      Callback* callback;
      if (callbackFreeList.empty()) {
        callback = callbackAllocator.allocate();
      } else {
        callback = callbackFreeList.back();
        callbackFreeList.pop_back();
      }
      CHECK(callback->partsLeft == 0);
      callback->onComplete = std::move(f);
      // return callback;
      return CallbackWrapper(callback, callbackDecref);
    };

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
            --dev.currentCqEntries;

            if (wc.wr_id) {
              Callback* callback = (Callback*)(void*)wc.wr_id;
              callbackDecref(callback);
            }

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
      while (dev.currentCqEntries == maxCqEntries) {
        poll();
      }
      ++dev.currentCqEntries;
    };

    size_t bytesWritten = 0;

    bool forceSignalNext = false;
    auto writeData = [&](Device& dev, size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey,
                         size_t bytes, Callback* callback = nullptr) {
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

        if (callback) {
          ++callback->partsLeft;
        }

        preSend(dev, i, wr);

        ibv_qp_ex* qp = dev.ib->qpex;

        // fmt::printf(
        //     "%d: rdma write %d bytes (%p -> %p, rkey %#x) (dev %d, i %d)\n", rank, bytes, localAddress,
        //     remoteAddress, rkey, &dev - devices.data(), i);

        wr.wr_id = (uint64_t)(void*)callback;

        ibv_wr_start(qp);
        qp->wr_id = wr.wr_id;
        qp->wr_flags = wr.send_flags;
        ibv_wr_rdma_write(qp, rkey, (uintptr_t)remoteAddress);
        ibv_wr_set_sge_list(qp, 1, &sge);
        ibv_wr_set_ud_addr(qp, dev.ib->ahs.at(i), dev.ib->remoteAddresses.at(i).qpNum, 0x4242);
        int error = ibv_wr_complete(qp);
        if (error) {
          fmt::fprintf(stderr, "ibv_wr_complete failed with error %d: %s\n", error, std::strerror(error));
          std::fflush(stderr);
          CHECK(false);
        }

        // while (true) {
        //   bool stop = true;
        //   for (auto& dev : devices) {
        //     while (dev.currentCqEntries) {
        //       poll();
        //       stop = false;
        //     }
        //   }
        //   if (stop) {
        //     break;
        //   }
        // }
      };

      const size_t threshold = 1024ull * 1024 * 896;
      const size_t chunkSize = 1024ull * 1024 * 768;
      if (bytes >= threshold) {
        size_t remaining = bytes;
        bytes = chunkSize;
        bool signal = forceSignalNext;
        while (remaining > 0) {
          if (remaining < threshold) {
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

    struct CudaMapping {
      uint64_t bufferId;
      size_t bytes;
      MemoryRegion mr;
    };

    HashMap<uintptr_t, std::unique_ptr<MemoryRegion>> mrMap;
    HashMap<uintptr_t, std::unique_ptr<CudaMapping>> mrMapCuda;

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
      unsigned long long bufferId2 = -1;
      CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + bytes - 1));
      CHECK(bufferId == bufferId2);
      auto i = mrMapCuda.find(address);
      if (i != mrMapCuda.end()) {
        if (i->second->bufferId == bufferId && i->second->bytes >= bytes) {
          return &i->second->mr;
        }
      }
      auto ptr = std::make_unique<CudaMapping>();
      ptr->bufferId = bufferId;
      ptr->bytes = bytes;
      ptr->mr.mrs.fill(nullptr);
      CHECK(devices.size() <= ptr->mr.mrs.size());
      for (size_t i = 0; i != devices.size(); ++i) {
        // ptr->mr.mrs[i] = devices[i].ib->getMr(address, bytes);
        auto* mr = ibv_reg_mr(
            devices[i].ib->protectionDomain, (void*)address, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        localMrs.push_back(std::move(mr));
        ptr->mr.mrs[i] = mr;
        fmt::printf(
            "new mapped range of %d bytes at %#x -> fd %d  (mr lkey %#x rkey %#x)\n", bytes, address, bufferId,
            mr->lkey, mr->rkey);
      }
      MemoryRegion* r = &ptr->mr;
      mrMapCuda[address] = std::move(ptr);
      return r;
    };

    // auto regMrCuda = [&](uintptr_t address, size_t bytes) {
    //   unsigned long long bufferId = -1;
    //   CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
    //   CHECK(bufferId != -1);
    //   unsigned long long bufferId2 = -1;
    //   CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + bytes - 1));
    //   CHECK(bufferId == bufferId2);
    //   auto i = mrMapCuda.find(bufferId);
    //   if (i != mrMapCuda.end()) {
    //     CHECK(
    //         (uintptr_t)(*i->second).mrs[0]->addr <= address &&
    //         (uintptr_t)(*i->second).mrs[0]->addr + (*i->second).mrs[0]->length >= address + bytes);
    //     return &*i->second;
    //   }
    //   auto ptr = std::make_unique<MemoryRegion>();
    //   ptr->mrs.fill(nullptr);
    //   CHECK(devices.size() <= ptr->mrs.size());
    //   for (size_t i = 0; i != devices.size(); ++i) {
    //     ptr->mrs[i] = devices[i].ib->getMr(address, bytes);
    //   }
    //   MemoryRegion* r = &*ptr;
    //   mrMapCuda[bufferId] = std::move(ptr);
    //   return r;
    // };

    auto* commsMr = regMr((uintptr_t)group->mySharedMem, group->mySharedMemSize);

    DynamicAddresses* localDyns = group->localDyns;
    Progress* localProgress = group->localProgress;

    struct RemoteAddressAndKey {
      uintptr_t address;
      std::vector<uint32_t> keys;
    };

    auto distributeAddressAndKeys = [&](uintptr_t address, MemoryRegion* mr) {
      std::vector<uint32_t> localKeys;
      for (size_t i = 0; i != devices.size(); ++i) {
        localKeys.push_back(mr->mrs.at(i)->rkey);
      }
      auto keys = setupComms->allgather(localKeys);
      auto addrs = setupComms->allgather(address);
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

    auto* cudaStepValueMr = regMrCuda(group->cudaStepValue.cudaPointer, group->cudaStepValue.bytes);

    std::vector<RemoteAddressAndKey> remoteComms = distributeAddressAndKeys((uintptr_t)group->mySharedMem, commsMr);
    std::vector<RemoteAddressAndKey> remoteCudaStepValue =
        distributeAddressAndKeys(group->cudaStepValue.cudaPointer, cudaStepValueMr);

    auto* cudaCommsDeviceDataSentMr =
        regMrCuda(group->cudaCommsDeviceDataSent.cudaPointer, group->cudaCommsDeviceDataSent.bytes);
    auto remoteCudaCommsDeviceDataSent =
        distributeAddressAndKeys(group->cudaCommsDeviceDataSent.cudaPointer, cudaCommsDeviceDataSentMr);

    AllocatedBuffer sendStepValuesStorage = group->allocateHost(sizeof(uint32_t) * Group::dataChunks);
    auto* sendStepValuesStorageMr = regMr((uintptr_t)sendStepValuesStorage.cpuPointer, sendStepValuesStorage.bytes);
    uint32_t* sendStepValues = (uint32_t*)sendStepValuesStorage.cpuPointer;

    auto writeDataDistributed = [&](size_t dst, uintptr_t srcAddr, size_t len, MemoryRegion* mr, uint64_t dstAddr,
                                    const std::array<uint32_t, 32>& key, Callback* callback) {
      CHECK(len > 0);
      size_t offset = 0;
      const size_t chunks = devices.size();
      const size_t chunkSize = len / chunks;
      for (size_t i = 0; i != chunks; ++i) {
        size_t n = i == chunks - 1 ? len - offset : chunkSize;
        auto& dev = devices[i];
        if (n != 0) {
          writeData(
              dev, dst, (void*)(srcAddr + offset), mr->mrs[i]->lkey, (void*)(dstAddr + offset), key[i], n, callback);
        }
        offset += n;
      }
    };

    auto writeStep = [&](size_t deviceIndex, size_t dst, uint32_t stepValue) {
      Device& dev = devices[deviceIndex];
      uintptr_t dstOffset = group->getSharedOffset(&localProgress[rank].stepValue);
      localProgress[rank].stepValue = stepValue;
      forceSignalNext = true;
      writeData(
          dev, dst, &localProgress[rank].stepValue, commsMr->mrs.at(deviceIndex)->lkey,
          (void*)(remoteComms[dst].address + dstOffset), remoteComms[dst].keys[deviceIndex], sizeof(stepValue));
    };

    volatile uint32_t* cpuIn = (uint32_t*)group->cpuInBuffer.cpuPointer;
    volatile uint32_t* cpuOut = (uint32_t*)group->cpuOutBuffer.cpuPointer;

    fmt::printf("rank %d starting test!\n", rank);

    AllocatedBuffer testBuffer = group->allocateHost(0x1000 * size);
    auto* testMr = regMr((uintptr_t)testBuffer.cpuPointer, testBuffer.bytes);
    auto testRemote = distributeAddressAndKeys((uintptr_t)testBuffer.cpuPointer, testMr);
    uint64_t* testPtr = (uint64_t*)testBuffer.cpuPointer;
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
            testRemote[i].keys[di], n * 8, makeCallback([&]() { --remaining; }));
        offset += n;
      }
    }
    while (remaining) {
      poll();
    }
    setupComms->allgather(0);
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

    fmt::printf("rank %d test done!\n", rank);

    AllocatedBuffer outDynsBuffer = group->allocateHost(sizeof(DynamicAddresses) * size);
    DynamicAddresses* outDyns = (DynamicAddresses*)outDynsBuffer.cpuPointer;
    MemoryRegion* outDynsMr = regMr((uintptr_t)outDynsBuffer.cpuPointer, outDynsBuffer.bytes);

    std::vector<uint32_t> dynReady;

    size_t opCount = 0;

    struct TraceEvent {
      std::string name;
      std::chrono::system_clock::time_point begin;
      std::chrono::system_clock::time_point end;
    };
    std::vector<TraceEvent> traceEvents;

    std::chrono::system_clock::time_point beginning = std::chrono::system_clock::now();

    std::string currentTraceName = "";
    std::chrono::system_clock::time_point currentTraceBegin = beginning;
    int threadId = gettid();

    auto trace = [&](std::string name) {
      // if (opCount < 1000 || opCount >= 1200) {
      if (!profilingEnabled) {
        if (!traceEvents.empty()) {
          std::string fn = fmt::sprintf("moodist-trace-cpu-%d.json", rank);
          FILE* f = fopen(fn.c_str(), "wb");
          CHECK(f != nullptr);
          fmt::fprintf(
              f, R"({"traceEvents": [)"
                 "\n");
          bool first = true;
          for (auto& e : traceEvents) {
            if (!first) {
              fmt::fprintf(f, ",\n");
            } else {
              first = false;
            }
            fmt::fprintf(
                f, R"({"ph": "X", "name": "%s", "ts": %d, "dur": %d, "tid": %d})", e.name,
                std::chrono::duration_cast<std::chrono::microseconds>(e.begin.time_since_epoch()).count(),
                std::chrono::duration_cast<std::chrono::microseconds>(e.end - e.begin).count(), threadId);
          }
          fmt::fprintf(f, "]}\n");
          fclose(f);
          fmt::printf("Chrome trace dumped to %s\n", fn);
          traceEvents.clear();
        }
        return;
      }
      auto now = std::chrono::system_clock::now();
      if (traceEvents.empty() && currentTraceName.empty()) {
        beginning = now;
      }
      if (!currentTraceName.empty()) {
        traceEvents.emplace_back();
        TraceEvent& e = traceEvents.back();
        e.name = std::move(currentTraceName);
        e.begin = currentTraceBegin;
        e.end = now;
      }
      currentTraceBegin = now;
      currentTraceName = std::move(name);
    };
#define trace(name)

    while (true) {

      while (queueSize == 0) {
        poll();
        int sum = 0;
        for (auto& dev : devices) {
          sum += dev.currentCqEntries;
        }
        if (sum == 0) {
          futexWait(&queueSize, 0, std::chrono::seconds(100));
        }
      }
      --queueSize;

      ++opCount;

      std::unique_lock l(mutex);
      CHECK(!queue.empty());
      QueueEntry& queueEntry = queue.front();
      queue.pop_front();
      l.unlock();

      CHECK(queueEntry.stepValue < (1ul << 31));

      if (queueEntry.task == taskAllgather) {
        QueueEntryAllGather& params = (QueueEntryAllGather&)queueEntry;

        uint32_t stepValue = params.stepValue;

        AllGather& allGather = *group->allGather;

        auto& sendRanks = allGather.sendRanks;
        auto& recvRanks = allGather.recvRanks;

        trace(fmt::sprintf("op_%d_setup_%d", opCount - 1, params.bytes));

        // fmt::printf("rank %d cpu thread got all gather (step %#x)\n", rank, stepValue);

        // fmt::printf(
        //     "rank %d inputAddress %#x outputAddress %#x bytes %d\n", rank, params.inputAddress, params.outputAddress,
        //     params.bytes);

        auto* inputMr = regMrCuda(params.inputAddress, params.bytes);
        // auto* outputMr = regMrCuda(params.outputAddress, params.bytes * size);

        // DynamicAddresses* mydyn = &localDyns[rank];
        // for (size_t i = 0; i != devices.size(); ++i) {
        //   mydyn->gatherKey[i] = outputMr->mrs[i]->rkey;
        // }
        // mydyn->gatherAddress = (uintptr_t)params.outputAddress;

        // EFA throws a IBV_WC_BAD_RESP_ERR if we try to write into a MR at an offset > 2GB.
        // Thus, we register each ranks output address independently, so they get different MRs.
        for (size_t i : recvRanks) {
          auto* mr = regMrCuda(params.outputAddress + params.bytes * i, params.bytes);
          outDyns[i].gatherAddress = params.outputAddress;
          for (size_t di = 0; di != devices.size(); ++di) {
            outDyns[i].gatherKey[di] = mr->mrs[di]->rkey;
          }
        }

        // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
        // until kernel entry, such that we don't receive data until we're ready.
        dynReady.resize(recvRanks.size());
        size_t n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          writeData(
              dev, i, &outDyns[i], outDynsMr->mrs[di]->lkey,
              (void*)(remoteComms[i].address + sizeof(DynamicAddresses) * rank), remoteComms[i].keys[di],
              sizeof(DynamicAddresses), makeCallback([&, i, di, stepValue, n]() { dynReady[n] = stepValue; }));
          ++n;
        }

        // fmt::printf("sent dyns\n");

        // fmt::printf("got all dyns!\n");

        trace("enter_wait_for_kernel");

        // wait for kernel
        while (cpuIn[0] < stepValue) {
          poll();
        }

        trace("dyn_ready");

        // Now we can signal remote peers
        n = 0;
        for (size_t i : recvRanks) {
          while (dynReady[n] < stepValue) {
            poll();
          }
          // size_t di = (rank + n) % devices.size();
          // auto& dev = devices[di];
          // writeStep(di, i, stepValue);
          ++n;
        }

        auto start = std::chrono::steady_clock::now();

        for (size_t i = 0; i != Group::dataChunks; ++i) {
          sendStepValues[i] = stepValue + i;
        }

        CHECK(recvRanks.size() == sendRanks.size());

        // for (size_t i : sendRanks) {
        for (size_t index = 0; index != sendRanks.size(); ++index) {
          if (true) {
            size_t i = recvRanks[index];
            trace(fmt::sprintf("%d_write_step_%d", i, stepValue + index));
            size_t di = (rank + index) % devices.size();
            auto& dev = devices[di];
            writeStep(di, i, stepValue + index);
          }
          size_t i = sendRanks[index];
          // wait for dyns
          trace(fmt::sprintf("%d_wait_for_step_%d", i, stepValue + index));
          while (localProgress[i].stepValue < stepValue + index) {
            poll();
          }
          size_t offset = 0;
          size_t liveSends = 0;
          size_t liveStepValues = 0;
          size_t chunkSize = std::max(params.bytes / Group::dataChunks, (size_t)(1024 * 1024));
          for (size_t chunkIndex = 0; chunkIndex != Group::dataChunks; ++chunkIndex) {
            size_t nbytes = std::min(params.bytes - offset, chunkSize);

            if (nbytes > 0) {
              uintptr_t srcAddr = (uintptr_t)params.inputAddress + offset;

              // fmt::printf(
              //     "%d: write %#x bytes of remote data to %d at %#x + %#x + %#x\n", rank, nbytes, i,
              //     localDyns[i].gatherAddress, rank * params.bytes, offset);

              trace(fmt::sprintf("%d_send_%d_bytes", i, nbytes));
              ++liveSends;
              writeDataDistributed(
                  i, srcAddr, nbytes, inputMr, localDyns[i].gatherAddress + rank * params.bytes + offset,
                  localDyns[i].gatherKey, makeCallback([&, i, stepValue, chunkIndex]() { --liveSends; }));
              offset += nbytes;
            }
            trace(fmt::sprintf("%d_wait_for_liveSends", i));
            while (liveSends) {
              poll();
            }
            trace(fmt::sprintf("%d_wait_for_liveStepValues", i));
            while (liveStepValues) {
              poll();
            }
            trace(fmt::sprintf("%d_send_stepValues", i));
            // fmt::printf("%d: stepValue %#x + %d sent to %d\n", rank, stepValue, chunkIndex, i);
            for (size_t di = 0; di != devices.size(); ++di) {
              Device& dev = devices[di];
              ++liveStepValues;
              writeData(
                  dev, i, &sendStepValues[chunkIndex], sendStepValuesStorageMr->mrs.at(di)->lkey,
                  (void*)(remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (rank * 32 + di)),
                  remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue),
                  makeCallback([&]() { --liveStepValues; }));
            }
          }
          trace(fmt::sprintf("%d_wait_for_liveStepValues_final", i));
          while (liveStepValues) {
            poll();
          }
          // fmt::printf("sent %d  -> %d/%d\n", n, bytesSent, params.bytes);
          CHECK(offset == params.bytes);
          CHECK(liveSends == 0);
          CHECK(liveStepValues == 0);
        }

        trace("wait_for_cq_entries");

        while (true) {
          bool stop = true;
          for (auto& dev : devices) {
            while (dev.currentCqEntries) {
              poll();
              stop = false;
            }
          }
          if (stop) {
            break;
          }
        }
        for (auto& dev : devices) {
          CHECK(dev.currentCqEntries == 0);
        }

        // fmt::printf(
        //     "%fG/s\n", bytesWritten / seconds(std::chrono::steady_clock::now() - start) / 1024.0f / 1024.0f /
        //     1024.0f);
        // bytesWritten = 0;

        // fmt::printf("rank %d cpu thread all gather all done! (step %#x)\n", rank, stepValue);

        trace("exit_wait_for_kernel");

        cpuOut[0] = stepValue;
        while (cpuIn[0] < stepValue + 1) {
          poll();
        }

        trace("");

        freelistAllGather.push(&params);
      } else if (queueEntry.task == taskReduceScatter) {
        QueueEntryReduceScatter& params = (QueueEntryReduceScatter&)queueEntry;

        uint32_t stepValue = params.stepValue;

        ReduceScatter& reduceScatter = *group->reduceScatter;

        auto& sendRanks = reduceScatter.sendRanks;
        auto& recvRanks = reduceScatter.recvRanks;

        uintptr_t recvAddress = group->recvBuffer.cudaPointer;
        CHECK(group->recvBuffer.bytes >= params.bytes * 2);

        uintptr_t sendAddress = group->sendBuffer.cudaPointer;
        CHECK(group->sendBuffer.bytes >= params.bytes * sendRanks.size());

        auto* sendMr = regMrCuda(sendAddress, params.bytes * sendRanks.size());
        auto* recvMr = regMrCuda(recvAddress, params.bytes * 2);

        for (size_t i : recvRanks) {
          outDyns[i].gatherAddress = recvAddress;
          for (size_t di = 0; di != devices.size(); ++di) {
            outDyns[i].gatherKey[di] = recvMr->mrs[di]->rkey;
          }
        }

        // We can send the dyn up here, before kernel entry, but we must not signal to remote peers
        // until kernel entry, such that we don't receive data until we're ready.
        dynReady.resize(recvRanks.size());
        size_t n = 0;
        for (size_t i : recvRanks) {
          size_t di = (rank + n) % devices.size();
          auto& dev = devices[di];
          writeData(
              dev, i, &outDyns[i], outDynsMr->mrs[di]->lkey,
              (void*)(remoteComms[i].address + sizeof(DynamicAddresses) * rank), remoteComms[i].keys[di],
              sizeof(DynamicAddresses), makeCallback([&, i, di, stepValue, n]() { dynReady[n] = stepValue; }));
          ++n;
        }

        // fmt::printf("sent dyns\n");

        // fmt::printf("got all dyns!\n");

        trace("enter_wait_for_kernel");

        for (size_t i = 0; i != Group::dataChunks; ++i) {
          sendStepValues[i] = stepValue + i;
        }

        n = 0;
        for (size_t i : recvRanks) {
          while (dynReady[n] < stepValue) {
            poll();
          }
          ++n;
        }

        // wait for kernel
        while (cpuIn[0] < stepValue) {
          poll();
        }

        trace("dyn_ready");

        CHECK(recvRanks.size() == sendRanks.size());

        // for (size_t i : sendRanks) {
        for (size_t index = 0; index != sendRanks.size(); ++index) {

          // wait for send ready
          // this also functions as wait for recv ready
          while (cpuIn[1] < index + 1) {
            poll();
          }

          {
            size_t i = recvRanks[index];
            trace(fmt::sprintf("%d_write_step_%d", i, stepValue + index));
            size_t di = (rank + index) % devices.size();
            auto& dev = devices[di];
            writeStep(di, i, stepValue + index);
          }

          size_t i = sendRanks[index];
          // wait for dyns
          trace(fmt::sprintf("%d_wait_for_step_%d", i, stepValue + index));
          while (localProgress[i].stepValue < stepValue + index) {
            poll();
          }
          size_t offset = 0;
          size_t liveSends = 0;
          size_t liveStepValues = 0;
          size_t chunkSize = std::max(params.bytes / Group::dataChunks, (size_t)(1024 * 1024));
          for (size_t chunkIndex = 0; chunkIndex != Group::dataChunks; ++chunkIndex) {
            size_t nbytes = std::min(params.bytes - offset, chunkSize);

            if (nbytes > 0) {
              uintptr_t srcAddr = (uintptr_t)sendAddress + params.bytes * index + offset;

              // fmt::printf(
              //     "%d: write %#x bytes of remote data to %d at %#x + %#x + %#x\n", rank, nbytes, i,
              //     localDyns[i].gatherAddress, rank * params.bytes, offset);

              trace(fmt::sprintf("%d_send_%d_bytes", i, nbytes));
              ++liveSends;
              writeDataDistributed(
                  i, srcAddr, nbytes, sendMr, localDyns[i].gatherAddress + (params.bytes * (index % 2)) + offset,
                  localDyns[i].gatherKey, makeCallback([&, i, stepValue, chunkIndex]() { --liveSends; }));
              offset += nbytes;
            }
            trace(fmt::sprintf("%d_wait_for_liveSends", i));
            while (liveSends) {
              poll();
            }
            trace(fmt::sprintf("%d_wait_for_liveStepValues", i));
            while (liveStepValues) {
              poll();
            }
            trace(fmt::sprintf("%d_send_stepValues", i));
            // fmt::printf("%d: stepValue %#x + %d sent to %d\n", rank, stepValue, chunkIndex, i);
            for (size_t di = 0; di != devices.size(); ++di) {
              Device& dev = devices[di];
              ++liveStepValues;
              writeData(
                  dev, i, &sendStepValues[chunkIndex], sendStepValuesStorageMr->mrs.at(di)->lkey,
                  (void*)(remoteCudaCommsDeviceDataSent[i].address + sizeof(uint32_t) * (rank * 32 + di)),
                  remoteCudaCommsDeviceDataSent[i].keys[di], sizeof(stepValue),
                  makeCallback([&]() { --liveStepValues; }));
            }
          }
          trace(fmt::sprintf("%d_wait_for_liveStepValues_final", i));
          while (liveStepValues) {
            poll();
          }
          // fmt::printf("sent %d  -> %d/%d\n", n, bytesSent, params.bytes);
          CHECK(offset == params.bytes);
          CHECK(liveSends == 0);
          CHECK(liveStepValues == 0);
        }

        trace("wait_for_cq_entries");

        while (true) {
          bool stop = true;
          for (auto& dev : devices) {
            while (dev.currentCqEntries) {
              poll();
              stop = false;
            }
          }
          if (stop) {
            break;
          }
        }
        for (auto& dev : devices) {
          CHECK(dev.currentCqEntries == 0);
        }

        cpuOut[0] = stepValue;
        while (cpuIn[0] < stepValue + 1) {
          poll();
        }

        freelistReduceScatter.push(&params);
      } else if (queueEntry.task == taskTerminate) {
        freelistTerminate.push(&queueEntry);
        break;
      } else {
        throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
      }
    }

    trace("");

  } catch (const std::exception& e) {
    fmt::fprintf(stderr, "Error: %s\n", e.what());
    fflush(stderr);
    throw std::runtime_error("Moodist cpu thread got an exception");
  } catch (QuitCpuThread) {
  }
}

} // namespace moodist
