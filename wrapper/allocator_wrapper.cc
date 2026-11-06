// Copyright (c) Meta Platforms, Inc. and affiliates.

// CUDA Allocator wrapper - thin shim around CoreApi.
// This file is compiled into _C.so (wrapper library).

#include "api/allocator.h"
#include "moodist_loader.h"
#include "torch_includes.h"

#include <torch/version.h>

#include <mutex>

namespace moodist {

namespace {

// A PyTorch pool filter (see beginAllocateToPool), as the core calls it
struct PoolFilter {
  std::function<bool(cudaStream_t)> fn;
  static bool call(void* ctx, CUstream stream) {
    return static_cast<PoolFilter*>(ctx)->fn(stream);
  }
  static void destroy(void* ctx) {
    delete static_cast<PoolFilter*>(ctx);
  }
};

void setStat(c10::CachingAllocator::Stat& out, const CudaAllocatorStat& s) {
  out.current = s.current;
  out.peak = s.peak;
  out.allocated = s.allocated;
  out.freed = s.freed;
}

} // namespace

struct CUDAAllocator : c10::cuda::CUDACachingAllocator::CUDAAllocator {
  // Allocates bytes for use on stream; throws PyTorch's OutOfMemoryError if out of memory
  torch::DataPtr allocate(size_t bytes, CUstream stream, int deviceIndex) {
    uintptr_t ptr;
    void* cleanupCtx;
    int actualDevice;
    try {
      coreApi.cudaAllocatorImplAllocate(nullptr, bytes, stream, deviceIndex, &ptr, &cleanupCtx, &actualDevice);
    } catch (const CudaOutOfMemoryError& e) {
      std::vector<c10::cuda::CUDACachingAllocator::OutOfMemoryObserver> observers;
      {
        std::lock_guard l(observersMutex);
        observers = oomObservers;
      }
      if (!observers.empty()) {
        size_t free = 0;
        size_t total = 0;
        cudaMemGetInfo(&free, &total);
        for (auto& observer : observers) {
          observer(deviceIndex, bytes, total, free);
        }
      }
      TORCH_CHECK_WITH(OutOfMemoryError, false, e.what());
    }
    auto deleter = [](void* ctx) {
      coreApi.cudaAllocatorImplFree(ctx, nullptr);
    };
    return torch::DataPtr((void*)ptr, cleanupCtx, deleter, torch::Device(torch::kCUDA, actualDevice));
  }

  virtual torch::DataPtr allocate(size_t bytes) override {
    int deviceIndex = c10::cuda::current_device();
    return allocate(bytes, c10::cuda::getCurrentCUDAStream(deviceIndex), deviceIndex);
  }

  virtual torch::DeleterFnPtr raw_deleter() const override {
    return nullptr;
  }

  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

  void* rawAllocate(torch::DataPtr ptr) {
    void* r = ptr.get();
    std::lock_guard l(rawAllocMutex);
    rawAllocations[r] = std::move(ptr);
    return r;
  }

  virtual void* raw_alloc(size_t nbytes) override {
    return rawAllocate(allocate(nbytes));
  }

  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    return rawAllocate(allocate(nbytes, stream, c10::cuda::current_device()));
  }

  virtual void raw_delete(void* ptr) override {
    std::unique_lock l(rawAllocMutex);
    auto i = rawAllocations.find(ptr);
    if (i == rawAllocations.end()) {
      throw std::runtime_error("moodist CUDAAllocator::raw_delete: allocation not found");
    }
    auto h = std::move(i->second);
    rawAllocations.erase(i);
    l.unlock();
    // h destructor calls deleter
  }

  virtual void init(int device_count) override {}
  virtual bool initialized() override {
    return true;
  }
  // torch.cuda.set_per_process_memory_fraction: a limit on the memory the allocator maps (see
  // cudaAllocatorSetMemoryLimit)
  virtual double getMemoryFraction(c10::DeviceIndex) {
    return memoryFraction;
  }
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    TORCH_CHECK(fraction >= 0 && fraction <= 1, "invalid fraction: ", fraction, ". Please set within [0, 1].");
    size_t free = 0;
    size_t total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    memoryFraction = fraction;
    coreApi.cudaAllocatorSetMemoryLimit(fraction >= 1 ? 0 : std::max((size_t)1, (size_t)(fraction * total)));
  }
  virtual std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize> getExpandableSegmentSizes(c10::DeviceIndex) {
    return {};
  }
  virtual void enable(bool) {}
  virtual bool isEnabled() const {
    return true;
  }
  // Memory of released pools (CUDA graphs and MemPools destroyed) goes back to moodist's main pool, after a device sync
  // (see cudaAllocatorEmptyCache); moodist never unmaps memory
  virtual void emptyCache() {
    coreApi.cudaAllocatorEmptyCache();
  }
  virtual void emptyCache(at::cuda::MempoolId_t) {
    coreApi.cudaAllocatorEmptyCache();
  }

  // The largest allocation that would succeed without mapping more memory, or else the device's free memory (an
  // estimate, as PyTorch's)
  virtual void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) override {
    if (*largestBlock == 0) {
      size_t total = 0;
      cudaMemGetInfo(largestBlock, &total);
    }
    CudaAllocatorStats stats;
    coreApi.cudaAllocatorGetStats(&stats);
    *largestBlock = std::max(*largestBlock, stats.largestFreeBlock);
  }

  virtual void* getBaseAllocation(void* ptr, size_t* size) override {
    uintptr_t base;
    size_t sz;
    coreApi.allocatorMappedRegion((uintptr_t)ptr, &base, &sz);
    *size = sz;
    return (void*)base;
  }

  virtual void recordStream(const torch::DataPtr& ptr, c10::cuda::CUDAStream stream) override {
    coreApi.cudaAllocatorImplRecordStream(nullptr, (uintptr_t)ptr.get(), stream.stream());
  }

  virtual c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    CudaAllocatorStats s;
    coreApi.cudaAllocatorGetStats(&s);
    c10::CachingDeviceAllocator::DeviceStats r;
    for (size_t i = 0; i != 3; ++i) {
      setStat(r.allocation[i], s.allocations[i]);
      setStat(r.active[i], s.allocations[i]);
      setStat(r.allocated_bytes[i], s.allocatedBytes[i]);
      setStat(r.active_bytes[i], s.allocatedBytes[i]);
      setStat(r.requested_bytes[i], s.requestedBytes[i]);
    }
    // (all memory is in chunks, PyTorch's large segments)
    for (size_t i : {0, 2}) {
      setStat(r.reserved_bytes[i], s.reservedBytes);
      setStat(r.segment[i], s.segments);
    }
    r.num_ooms = s.numOoms;
    r.num_device_alloc = s.numMaps;
    return r;
  }

  virtual void resetAccumulatedStats(c10::DeviceIndex device) override {
    coreApi.cudaAllocatorResetStats(false, true);
  }

  virtual void resetPeakStats(c10::DeviceIndex device) override {
    coreApi.cudaAllocatorResetStats(true, false);
  }

  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() {
    return {};
  }
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(at::cuda::MempoolId_t) {
    return {};
  }
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(at::cuda::MempoolId_t, bool) {
    return {};
  }

  // Memory pools (CUDA graphs' private pools, torch.cuda.MemPool): see cudaAllocatorBeginAllocateToPool
  virtual void beginAllocateToPool(
      c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id, std::function<bool(cudaStream_t)> filter) override {
    auto* f = new PoolFilter{std::move(filter)};
    coreApi.cudaAllocatorBeginAllocateToPool(
        mempool_id.first, mempool_id.second, &PoolFilter::call, f, &PoolFilter::destroy);
  }
  virtual void endAllocateToPool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    coreApi.cudaAllocatorEndAllocateToPool(mempool_id.first, mempool_id.second);
  }
  virtual void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    coreApi.cudaAllocatorReleasePool(mempool_id.first, mempool_id.second);
  }
  virtual int getPoolUseCount(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    return coreApi.cudaAllocatorGetPoolUseCount(mempool_id.first, mempool_id.second);
  }
  virtual void createOrIncrefPool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator) override {
    TORCH_CHECK(!allocator,
        "moodist's CUDA allocator does not support memory pools with an allocator of their own (torch.cuda.MemPool "
        "with an allocator)");
    coreApi.cudaAllocatorCreateOrIncrefPool(mempool_id.first, mempool_id.second);
  }
  // (no `override` on the following: not virtuals of every supported PyTorch version)
  virtual void setUseOnOOM(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id, bool use_on_oom) {}
  virtual void setNoSplit(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) {}
  // (PyTorch 2.13+: right after a capture begins and ends; see cudaAllocatorSetCaptureMarks)
  virtual void markCaptureBegin(c10::DeviceIndex device) {
    coreApi.cudaAllocatorMarkCaptureBegin();
  }
  virtual void markCaptureEnd(c10::DeviceIndex device) {
    coreApi.cudaAllocatorMarkCaptureEnd();
  }

  virtual c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(void* ptr) override {
    throw std::runtime_error("moodist CUDAAllocator::shareIpcHandle: not implemented");
  }

  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    throw std::runtime_error("moodist CUDAAllocator::getIpcDevPtr: not implemented");
  }

  virtual void recordHistory(bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries, c10::cuda::CUDACachingAllocator::RecordContext when) {}
  virtual void recordHistory(bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries, c10::cuda::CUDACachingAllocator::RecordContext when, bool) {}
  virtual void recordHistory(bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries, c10::cuda::CUDACachingAllocator::RecordContext when, bool,
      const std::vector<std::string>&) {}

  virtual void attachOutOfMemoryObserver(c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override {
    std::lock_guard l(observersMutex);
    oomObservers.push_back(std::move(observer));
  }
  // Pure virtual since torch 2.12 (c10::cuda::CUDACachingAllocator::OomRejectionObserver). The type is spelled
  // out and there is no `override` so this still compiles against older torch, where it's just an unused virtual.
  // (moodist rejects no allocation ahead of trying it)
  virtual void attachOomRejectionObserver(std::function<void(int64_t, size_t, size_t, size_t)> observer) {}
  virtual void attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override {}
  virtual void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {}

  virtual cudaError_t memcpyAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
      cudaStream_t stream, bool p2p_enabled) override {
    cudaError_t err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
    return err;
  }

  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> getCheckpointState(
      c10::DeviceIndex device, c10::cuda::MempoolId_t id) override {
    return nullptr;
  }

  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device, std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) override {
    return {};
  }

  virtual std::string name() override {
    return "moodist";
  }

private:
  std::mutex rawAllocMutex;
  std::unordered_map<void*, torch::DataPtr> rawAllocations;
  std::mutex observersMutex;
  std::vector<c10::cuda::CUDACachingAllocator::OutOfMemoryObserver> oomObservers;
  double memoryFraction = 1.0;
};

namespace {
std::mutex assignmentMutex;
CUDAAllocator* cudaAllocator = nullptr;
} // namespace

namespace allocator {

bool owns(uintptr_t address) {
  return coreApi.allocatorOwns(address);
}

std::pair<uintptr_t, size_t> mappedRegion(uintptr_t address) {
  uintptr_t base;
  size_t size;
  coreApi.allocatorMappedRegion(address, &base, &size);
  return {base, size};
}

} // namespace allocator

void enableCudaAllocator() {
  std::lock_guard l(assignmentMutex);
  if (!cudaAllocator) {
    cudaAllocator = new CUDAAllocator();
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 13)
    coreApi.cudaAllocatorSetCaptureMarks(true);
#endif
  }
  c10::cuda::CUDACachingAllocator::allocator = cudaAllocator;
}

} // namespace moodist
