
#include "common.h"
#include "group.h"
#include "intrusive_list.h"
#include "vector.h"

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

namespace moodist {

inline LogLevel memLogLevel = []() {
  const char* c = std::getenv("MOODIST_MEM_LOG_LEVEL");
  LogLevel r = LOG_INFO;
  if (c) {
    std::string s = c;
    for (auto& c : s) {
      c = std::toupper(c);
    }
    if (s == "NONE") {
      r = LOG_NONE;
    } else if (s == "ERROR") {
      r = LOG_ERROR;
    } else if (s == "INFO") {
      r = LOG_INFO;
    } else if (s == "VERBOSE") {
      r = LOG_VERBOSE;
    } else if (s == "DEBUG") {
      r = LOG_DEBUG;
    } else {
      r = (LogLevel)std::atoi(c);
    }
    if (currentLogLevel < LOG_ERROR) {
      r = LOG_ERROR;
    }
  }
  return r;
}();

inline struct MemLog {
  template<typename... Args>
  void error(const char* fmt, Args&&... args) {
    logat(LOG_ERROR, fmt, std::forward<Args>(args)...);
  }
  template<typename... Args>
  void info(const char* fmt, Args&&... args) {
    if (memLogLevel >= LOG_INFO) {
      [[unlikely]];
      logat(LOG_INFO, fmt, std::forward<Args>(args)...);
    }
  }
  template<typename... Args>
  void verbose(const char* fmt, Args&&... args) {
    if (memLogLevel >= LOG_VERBOSE) {
      [[unlikely]];
      logat(LOG_VERBOSE, fmt, std::forward<Args>(args)...);
    }
  }
  template<typename... Args>
  void debug(const char* fmt, Args&&... args) {
    if (memLogLevel >= LOG_DEBUG) {
      [[unlikely]];
      logat(LOG_DEBUG, fmt, std::forward<Args>(args)...);
    }
  }
} memlog;

namespace {

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

void add_span(std::vector<Span>& spans, uintptr_t begin, uintptr_t end) {
  CHECK(end > begin);
  // printf("add span %#lx %#lx\n", begin, end);
  // printf("PRE %d spans:\n", spans.size());
  // for (auto& v : spans) {
  //   printf("[%#lx, %#lx)\n", v.begin, v.end);
  // }
  // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
  Span* s = spans.data();
  size_t a = 0;
  size_t b = spans.size();
  size_t e = spans.size();
  if (b) {
    while (a < b) {
      size_t mid = (a + b) / 2;
      if (s[mid].begin <= begin) {
        a = mid + 1;
      } else {
        b = mid;
      }
    }
    CHECK(a <= e);
    if (a != 0) {
      if (s[a - 1].end == begin) {
        if (a != e && s[a].begin == end) {
          s[a - 1].end = s[a].end;
          spans.erase(spans.begin() + a);
          // printf("%d spans:\n", spans.size());
          // for (auto& v : spans) {
          //   printf("[%#lx, %#lx)\n", v.begin, v.end);
          // }
          // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
          return;
        }
        s[a - 1].end = end;
        // printf("%d spans:\n", spans.size());
        // for (auto& v : spans) {
        //   printf("[%#lx, %#lx)\n", v.begin, v.end);
        // }
        // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
        return;
      }
    }
    if (a != e && s[a].begin == end) {
      s[a].begin = begin;
      // printf("%d spans:\n", spans.size());
      // for (auto& v : spans) {
      //   printf("[%#lx, %#lx)\n", v.begin, v.end);
      // }
      // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
      return;
    }
  }
  CHECK(a <= spans.size());
  spans.insert(spans.begin() + a, Span{begin, end});

  // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));

  // printf("INSERT %d spans:\n", spans.size());
  // for (auto& v : spans) {
  //   printf("[%#lx, %#lx)\n", v.begin, v.end);
  // }
}

struct MemoryEvent {
  IntrusiveListLink<MemoryEvent> link;
  size_t refcount = 0;
  CUevent event = nullptr;
};

struct MemoryEventHandle {
  MemoryEvent* e = nullptr;
  Vector<MemoryEvent*>* container = nullptr;
  MemoryEventHandle() = default;
  MemoryEventHandle(MemoryEvent* e, Vector<MemoryEvent*>* container) : e(e), container(container) {
    ++e->refcount;
  }
  ~MemoryEventHandle() {
    decref();
  }
  MemoryEventHandle(MemoryEventHandle&& n) {
    *this = std::move(n);
  }
  MemoryEventHandle(const MemoryEventHandle& n) {
    *this = n;
  }
  MemoryEventHandle& operator=(const MemoryEventHandle& n) {
    if (e) {
      decref();
    }
    e = n.e;
    container = n.container;
    if (e) {
      ++e->refcount;
    }
    return *this;
  }
  MemoryEventHandle& operator=(MemoryEventHandle&& n) {
    std::swap(e, n.e);
    std::swap(container, n.container);
    return *this;
  }
  void decref() {
    if (e && --e->refcount == 0) {
      container->push_back(e);
      e = nullptr;
      container = nullptr;
    }
  }
  MemoryEvent* operator->() {
    return e;
  }
  MemoryEvent& operator*() {
    return *e;
  }
};

struct CudaAllocatorImpl {
  CudaAllocatorImpl() {}

  PoolAllocator<MemoryEvent> memoryEventAllocator;
  Vector<MemoryEvent*> memoryEventFreelist;

  std::vector<Span> mappedRegions;

  std::vector<Span> freeMemory;
  std::vector<std::pair<Span, MemoryEventHandle>> freeMemoryEvents;

  std::vector<Span> pendingFreeMemory;
  Vector<std::pair<Span, MemoryEventHandle>> pendingFreeMemoryEvents;

  size_t nBytesMapped = 0;
  size_t reservedSize = 0;
  uintptr_t reservedBase = 0;
  size_t nextMapBase = 0;

  int deviceIndex = -1;

  Vector<CUmemGenericAllocationHandle> cuMemHandles;

  Vector<std::tuple<uintptr_t, size_t, CUstream>> pendingDeallocations;

  bool mapMoreMemory(size_t minbytes) {
    if (reservedBase == 0) {
      size_t free = 0;
      size_t total = 0;
      CHECK_CU(cuMemGetInfo(&free, &total));
      memlog.info("Moodist CUDA Allocator initializing. Device has %d free, %d total bytes of memory.\n", free, total);

      // constexpr size_t alignment = (size_t)1024 * 1024 * 1024;
      constexpr size_t alignment = (size_t)1024 * 1024 * 1024 * 1024;
      size_t reserveSize = (total + alignment - 1) / alignment * alignment;

      // reserveSize /= 4;

      memlog.info("Moodist CUDA Allocator reserving %d bytes\n", reserveSize);

      CUdeviceptr base = 0;
      CHECK_CU(cuMemAddressReserve(&base, reserveSize, alignment, 0, 0));
      reservedBase = base;
      reservedSize = reserveSize;
      nextMapBase = base;

      deviceIndex = c10::cuda::current_device();
    }

    size_t bytes = reservedSize - nextMapBase;
    size_t safebytes = reservedSize - 1024 * 1024 * 1024 - nextMapBase;
    if (safebytes >= minbytes) {
      bytes = safebytes;
    }
    if (bytes < minbytes) {
      return false;
    }
    // bytes = std::min(bytes, std::max(minbytes, (size_t)1024 * 1024 * 1024 * 4));
    bytes = std::min(bytes, std::max(minbytes, (size_t)1024 * 1024 * 64));
    constexpr size_t alignment = (size_t)1024 * 1024 * 64;
    bytes = (bytes + alignment - 1) / alignment * alignment;

    CUmemGenericAllocationHandle handle;
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(prop));
    prop.location.id = deviceIndex;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    CHECK_CU(cuMemCreate(&handle, bytes, &prop, 0));

    cuMemHandles.push_back(handle);

    uintptr_t address = nextMapBase + bytes;
    nextMapBase += bytes;

    CHECK_CU(cuMemMap(address, bytes, 0, handle, 0));

    std::array<CUmemAccessDesc, 1> desc;
    std::memset(desc.data(), 0, sizeof(CUmemAccessDesc) * desc.size());
    desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc[0].location.id = deviceIndex;
    CHECK_CU(cuMemSetAccess(address, bytes, desc.data(), 1));

    memlog.info("Moodist successfully mapped %d bytes at %#x\n", bytes, address);

    add_span(mappedRegions, address, address + bytes);
    add_span(freeMemory, address, address + bytes);

    return true;
  }

  uintptr_t allocate(size_t bytes, CUstream stream) {
    memlog.debug("allocate %d\n", bytes);
    if (!pendingDeallocations.empty()) {
      for (auto& v : pendingDeallocations) {
        deallocateImpl(std::get<0>(v), std::get<1>(v), std::get<2>(v));
      }
      pendingDeallocations.clear();
    }
    CHECK(bytes > 0);
    for (auto& v : freeMemory) {
      uintptr_t ptr = v.begin;
      if (v.end - v.begin >= bytes) {
        if (v.end - v.begin <= bytes) {
          freeMemory.erase(freeMemory.begin() + (&v - freeMemory.data()));
        } else {
          // printf("grab alignedminbytes %ld\n", alignedminbytes);
          v.begin += bytes;
          CHECK(v.begin < v.end);
        }
        for (auto i = freeMemoryEvents.begin(); i != freeMemoryEvents.end();) {
          if (i->first.begin >= ptr + bytes) {
            break;
          }
          if (i->first.end > ptr && i->first.begin < ptr + bytes) {
            // memlog.info("allocate, wait for event %p", (void*)v.second->event);
            CHECK_CU(cuStreamWaitEvent(stream, i->second->event, CU_EVENT_WAIT_DEFAULT));
            if (i->first.begin < ptr) {
              auto nv = *i;
              nv.first.end = ptr;
              CHECK(i >= freeMemoryEvents.begin() && i < freeMemoryEvents.end());
              i = freeMemoryEvents.insert(i, nv);
              CHECK(i >= freeMemoryEvents.begin() && i < freeMemoryEvents.end());
              ++i;
              CHECK(i >= freeMemoryEvents.begin() && i < freeMemoryEvents.end());
              i->first.begin = ptr;
              CHECK(i >= freeMemoryEvents.begin() && i < freeMemoryEvents.end());
            }
            if (i->first.end <= ptr + bytes) {
              CHECK(i >= freeMemoryEvents.begin() && i < freeMemoryEvents.end());
              i = freeMemoryEvents.erase(i);
              CHECK(i >= freeMemoryEvents.begin() && i <= freeMemoryEvents.end());
            } else {
              CHECK(i >= freeMemoryEvents.begin() && i < freeMemoryEvents.end());
              i->first.begin = ptr + bytes;
              CHECK(i->first.end > i->first.begin);
              ++i;
              CHECK(i >= freeMemoryEvents.begin() && i <= freeMemoryEvents.end());
            }
            // CHECK(v.first.end > v.first.begin);
          } else {
            ++i;
          }
        }
        return ptr;
      }
    }
    // memlog.info("Prefetcher memory exhausted, moving pending\n");
    if (!pendingFreeMemory.empty()) {
      freePending();
      return allocate(bytes, stream);
    }
    if (memLogLevel >= LOG_VERBOSE) {
      memlog.verbose(
          "Memory exhausted during allocation of %s. Free memory:\n%s\n", this->bytes(bytes), debugFreeMemory());
    }
    if (mapMoreMemory(bytes)) {
      return allocate(bytes, stream);
    }
    // memlog.verbose("Prefetcher free memory:\n%s\n", debugFreeMemory());
    throw std::runtime_error(fmt::sprintf("Prefetcher failed to allocate %d bytes", bytes));
  }

  void deallocateImpl(uintptr_t cudaPtr, size_t alignedbytes, CUstream stream) {
    // memlog.info("deallocate %d\n", alignedbytes);
    MemoryEventHandle me = getMemoryEvent();
    CHECK_CU(cuEventRecord(me->event, stream));
    add_span(pendingFreeMemory, cudaPtr, cudaPtr + alignedbytes);
    pendingFreeMemoryEvents.emplace_back(Span{cudaPtr, cudaPtr + alignedbytes}, std::move(me));
  }

  void deallocate(uintptr_t cudaPtr, size_t alignedbytes, CUstream stream) {
    if (!memoryEventFreelist.empty()) {
      MemoryEventHandle me = getMemoryEvent();
      if (cuEventRecord(me->event, stream) == CUDA_SUCCESS) {
        add_span(pendingFreeMemory, cudaPtr, cudaPtr + alignedbytes);
        pendingFreeMemoryEvents.emplace_back(Span{cudaPtr, cudaPtr + alignedbytes}, std::move(me));
        return;
      }
    }
    pendingDeallocations.emplace_back(cudaPtr, alignedbytes, stream);
  }

  void freePending() {
    // memlog.info("free pending\n");
    CHECK(pendingFreeMemoryEvents.size() >= pendingFreeMemory.size());
    for (auto& v : pendingFreeMemory) {
      add_span(freeMemory, v.begin, v.end);
    }
    for (auto& v : pendingFreeMemoryEvents) {
      freeMemoryEvents.push_back(v);
    }
    std::sort(freeMemoryEvents.begin(), freeMemoryEvents.end(), [&](auto& a, auto& b) {
      return a.first.begin < b.first.begin;
    });
    pendingFreeMemory.clear();
    pendingFreeMemoryEvents.clear();
  }

  std::string bytes(size_t n) {
    return fmt::sprintf("%d bytes (%gG)", n, n / 1024.0 / 1024.0 / 1024.0);
  }

  std::string debugFreeMemory() {
    std::string s;
    size_t total = 0;
    size_t totalMappedBytes = 0;
    size_t largestChunk = 0;
    for (auto& v : mappedRegions) {
      totalMappedBytes += v.end - v.begin;
      size_t prevEnd = v.begin;
      for (auto& v : freeMemory) {
        // if (v.begin != prevEnd) {
        //   s += fmt::sprintf("ALLOCATED %s\n", bytes(v.begin - prevEnd));
        // }
        // s += fmt::sprintf("[%#x, %#x)  %s\n", v.begin, v.end, bytes(v.end - v.begin));
        largestChunk = std::max(largestChunk, v.end - v.begin);
        total += v.end - v.begin;
        prevEnd = v.end;
      }
      // if (prevEnd != v.end) {
      //   s += fmt::sprintf("ALLOCATED %s\n", bytes(v.end - prevEnd));
      // }
      CHECK(prevEnd <= v.end);
    }
    s += fmt::sprintf(
        "%s total, %s used, %s free. Largest free chunk %s\n", bytes(totalMappedBytes), bytes(totalMappedBytes - total),
        bytes(total), bytes(largestChunk));

    // s += fmt::sprintf("%d live cuda tensors:\n", cudaCpuMappings.size());
    // for (auto& v : cudaCpuMappings) {
    //   s += fmt::sprintf("%#x: %s\n", v.first, bytes(v.second->bytes));
    // }
    // s += fmt::sprintf("%d live cuda tensors:\n", cudaCpuMaps.size());
    // total = 0;
    // for (auto& v : cudaCpuMaps) {
    //   total += v.second.itemsize() * v.second.numel();
    //   s += fmt::sprintf("%#x: %s\n", v.first, bytes(v.second.itemsize() * v.second.numel()));
    // }
    // s += fmt::sprintf("total %s\n", bytes(total));
    return s;
  }

  MemoryEventHandle getMemoryEvent() {
    MemoryEvent* me;
    if (!memoryEventFreelist.empty()) {
      me = memoryEventFreelist.back();
      memoryEventFreelist.pop_back();
    } else {
      me = memoryEventAllocator.allocate();
      CHECK_CU(cuEventCreate(&me->event, CU_EVENT_DISABLE_TIMING));
      memlog.debug("allocate new memory event!\n");
    }
    CHECK(me->event != nullptr);
    CHECK(me->refcount == 0);
    return MemoryEventHandle(me, &memoryEventFreelist);
  }
};

CudaAllocatorImpl& cudaAllocImpl = *new CudaAllocatorImpl();

// void deleter_func(void* ptr) {
//   cudaAllocImpl.deallocate((uintptr_t)ptr);
// }

} // namespace

struct CUDAAllocator : c10::cuda::CUDACachingAllocator::CUDAAllocator {

  SpinMutex mutex;

  virtual torch::DataPtr allocate(size_t bytes) override {
    std::lock_guard l(mutex);
    constexpr size_t alignment = 0x1000;
    size_t alignedbytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    uintptr_t cudaPtr = cudaAllocImpl.allocate(alignedbytes, stream);
    Function<void()> f = [this, cudaPtr, alignedbytes, stream] {
      std::lock_guard l(mutex);
      cudaAllocImpl.deallocate(cudaPtr, alignedbytes, stream);
    };
    auto deleter = [](void* c) { Function<void()>(FunctionPointer(c))(); };
    torch::Device device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    memlog.debug("allocate(%d) returning cudaPtr %#x\n", bytes, cudaPtr);
    return torch::DataPtr((void*)cudaPtr, (void*)f.release(), deleter, device);
  }
  virtual torch::DeleterFnPtr raw_deleter() const override {
    // throw std::runtime_error("moodist CUDAAllocator::raw_deleter: not implemented");
    // return deleter_func;
    return nullptr;
  }
  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    fatal("waa");
    throw std::runtime_error("moodist CUDAAllocator::copy_data: not implemented");
  }
  virtual void* raw_alloc(size_t nbytes) override {
    return raw_alloc_with_stream(nbytes, c10::cuda::getCurrentCUDAStream());
  }
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    throw std::runtime_error("moodist CUDAAllocator::raw_alloc_with_stream: not implemented");
  }
  virtual void raw_delete(void* ptr) override {
    throw std::runtime_error("moodist CUDAAllocator::raw_delete: not implemented");
  }
  virtual void init(int device_count) override {
    // throw std::runtime_error("moodist CUDAAllocator::init: not implemented");
  }
  virtual bool initialized() override {
    throw std::runtime_error("moodist CUDAAllocator::initialized: not implemented");
  }
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    throw std::runtime_error("moodist CUDAAllocator::setMemoryFraction: not implemented");
  }
  virtual void emptyCache() override {
    throw std::runtime_error("moodist CUDAAllocator::emptyCache: not implemented");
  }
  virtual void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) override {
    throw std::runtime_error("moodist CUDAAllocator::cacheInfo: not implemented");
  }
  virtual void* getBaseAllocation(void* ptr, size_t* size) override {
    throw std::runtime_error("moodist CUDAAllocator::getBaseAllocation: not implemented");
  }
  virtual void recordStream(const torch::DataPtr&, c10::cuda::CUDAStream stream) override {
    throw std::runtime_error("moodist CUDAAllocator::recordStream: not implemented");
  }
  virtual c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    throw std::runtime_error("moodist CUDAAllocator::getDeviceStats: not implemented");
  }
  virtual void resetAccumulatedStats(c10::DeviceIndex device) override {
    throw std::runtime_error("moodist CUDAAllocator::resetAccumulatedStats: not implemented");
  }
  virtual void resetPeakStats(c10::DeviceIndex device) override {
    throw std::runtime_error("moodist CUDAAllocator::resetPeakStats: not implemented");
  }
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override {
    throw std::runtime_error("moodist CUDAAllocator::snapshot: not implemented");
  }
  virtual void beginAllocateToPool(
      c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id, std::function<bool(cudaStream_t)> filter) override {
    throw std::runtime_error("moodist CUDAAllocator::beginAllocateToPool: not implemented");
  }
  virtual void endAllocateToPool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    throw std::runtime_error("moodist CUDAAllocator::endAllocateToPool: not implemented");
  }
  virtual void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    throw std::runtime_error("moodist CUDAAllocator::releasePool: not implemented");
  }
  virtual c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(void* ptr) override {
    throw std::runtime_error("moodist CUDAAllocator::shareIpcHandle: not implemented");
  }
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    throw std::runtime_error("moodist CUDAAllocator::getIpcDevPtr: not implemented");
  }
  virtual void recordHistory(
      bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder, size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) override {
    throw std::runtime_error("moodist CUDAAllocator::recordHistory: not implemented");
  }
  virtual void attachOutOfMemoryObserver(c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override {
    throw std::runtime_error("moodist CUDAAllocator::attachOutOfMemoryObserver: not implemented");
  }

  virtual void attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override {
    throw std::runtime_error("moodist CUDAAllocator::attachOutOfMemoryObserver: not implemented");
  }

  virtual void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {
    throw std::runtime_error("moodist CUDAAllocator::enablePeerAccess: not implemented");
  }

  virtual cudaError_t memcpyAsync(
      void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream,
      bool p2p_enabled) override {
    throw std::runtime_error("moodist CUDAAllocator::memcpyAsync: not implemented");
  }
  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, c10::cuda::MempoolId_t id) override {
    throw std::runtime_error("moodist CUDAAllocator::getCheckpointState: not implemented");
  }
  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device, std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) override {
    throw std::runtime_error("moodist CUDAAllocator::setCheckpointPoolState: not implemented");
  }
  virtual std::string name() override {
    throw std::runtime_error("moodist CUDAAllocator::name: not implemented");
  }
};

void enableCudaAllocator() {
  c10::cuda::CUDACachingAllocator::allocator = new CUDAAllocator();
}

} // namespace moodist
