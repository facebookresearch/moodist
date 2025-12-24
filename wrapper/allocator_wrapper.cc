// Copyright (c) Meta Platforms, Inc. and affiliates.

// CUDA Allocator wrapper - thin shim around CoreApi.
// This file is compiled into _C.so (wrapper library).

#include "allocator.h"
#include "moodist_loader.h"
#include "torch_includes.h"

#include <mutex>

namespace moodist {

struct CUDAAllocator : c10::cuda::CUDACachingAllocator::CUDAAllocator {
  CudaAllocatorImpl* impl = nullptr;

  CUDAAllocator() {
    impl = coreApi.createCudaAllocatorImpl();
    coreApi.setCudaAllocatorImpl(impl);
  }

  virtual torch::DataPtr allocate(size_t bytes) override {
    CUstream stream = c10::cuda::getCurrentCUDAStream();
    int deviceIndex = c10::cuda::current_device();

    uintptr_t ptr;
    void* cleanupCtx;
    int actualDevice;
    coreApi.cudaAllocatorImplAllocate(impl, bytes, stream, deviceIndex, &ptr, &cleanupCtx, &actualDevice);

    // Create deleter that captures the stream at free time
    auto deleter = [](void* ctx) {
      CUstream freeStream = c10::cuda::getCurrentCUDAStream();
      coreApi.cudaAllocatorImplFree(ctx, freeStream);
    };

    torch::Device device = torch::Device(torch::kCUDA, actualDevice);
    return torch::DataPtr((void*)ptr, cleanupCtx, deleter, device);
  }

  virtual torch::DeleterFnPtr raw_deleter() const override {
    return nullptr;
  }

  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }

  virtual void* raw_alloc(size_t nbytes) override {
    auto ptr = allocate(nbytes);
    void* r = ptr.get();
    std::lock_guard l(rawAllocMutex);
    rawAllocations[r] = std::move(ptr);
    return r;
  }

  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    throw std::runtime_error("moodist CUDAAllocator::raw_alloc_with_stream: not implemented");
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
  virtual double getMemoryFraction(c10::DeviceIndex) {
    return 1.0;
  }
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) override {}
  virtual void enable(bool) {}
  virtual bool isEnabled() const {
    return true;
  }
  virtual void emptyCache() {}
  virtual void emptyCache(at::cuda::MempoolId_t) {}

  virtual void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) override {
    *largestBlock = 0;
  }

  virtual void* getBaseAllocation(void* ptr, size_t* size) override {
    uintptr_t base;
    size_t sz;
    coreApi.allocatorMappedRegion((uintptr_t)ptr, &base, &sz);
    *size = sz;
    return (void*)base;
  }

  virtual void recordStream(const torch::DataPtr& ptr, c10::cuda::CUDAStream stream) override {
    coreApi.cudaAllocatorImplRecordStream(impl, (uintptr_t)ptr.get(), stream.stream());
  }

  virtual c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    return deviceStats;
  }

  virtual void resetAccumulatedStats(c10::DeviceIndex device) override {
    deviceStats.allocated_bytes[0].reset_accumulated();
    deviceStats.active_bytes[0].reset_accumulated();
  }

  virtual void resetPeakStats(c10::DeviceIndex device) override {
    deviceStats.allocated_bytes[0].reset_peak();
    deviceStats.active_bytes[0].reset_peak();
  }

  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() {
    return {};
  }
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(at::cuda::MempoolId_t) {
    return {};
  }

  virtual void beginAllocateToPool(
      c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id, std::function<bool(cudaStream_t)> filter) override {}
  virtual void endAllocateToPool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {}
  virtual void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {}

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

  virtual void attachOutOfMemoryObserver(c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override {}
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
  c10::CachingDeviceAllocator::DeviceStats deviceStats;
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
  }
  c10::cuda::CUDACachingAllocator::allocator = cudaAllocator;
}

} // namespace moodist
