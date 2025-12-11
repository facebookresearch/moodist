// Copyright (c) Meta Platforms, Inc. and affiliates.

// CUDA Allocator wrapper - contains CUDAAllocator class and torch-dependent code.
// This file is compiled into _C.so (wrapper library).

#include "allocator.h"
#include "allocator_api.h"
#include "common.h"
#include "function.h"
#include "hash_map.h"
#include "moodist_loader.h"
#include "vector.h"

#include "torch_includes.h"

namespace moodist {

namespace {

struct PointerHash {
  template<typename T>
  size_t operator()(T p) const {
    return (uintptr_t)p * 6364136223846793005ull % 2147483647ull;
  }
};

} // namespace

struct CUDAAllocator : c10::cuda::CUDACachingAllocator::CUDAAllocator {

  SpinMutex mutex;

  CudaAllocatorImpl* impl = nullptr;

  CUcontext cuContext = nullptr;
  int deviceIndex = -1;

  Vector<std::unique_ptr<Vector<CUstream>>> streamListFree;
  HashMap<uintptr_t, std::unique_ptr<Vector<CUstream>>, PointerHash> streamUses;

  c10::CachingDeviceAllocator::DeviceStats deviceStats;

  HashMap<uintptr_t, FunctionPointer, PointerHash> freeCallbacks;

  CUDAAllocator() {
    impl = coreApi.createCudaAllocatorImpl();
    coreApi.setCudaAllocatorImpl(impl);
  }

  virtual torch::DataPtr allocate(size_t bytes) override {
    std::lock_guard l(mutex);

    if (deviceIndex == -1) {
      deviceIndex = c10::cuda::current_device();
      CHECK(deviceIndex != -1);

      cuCtxGetCurrent(&cuContext);
      if (!cuContext) {
        CUdevice cuDevice;
        CHECK_CU(cuInit(0));
        CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
        CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
        CHECK_CU(cuCtxSetCurrent(cuContext));
      }
    } else {
      int currentDevice = c10::cuda::current_device();
      if (currentDevice != deviceIndex) {
        throw std::runtime_error(fmt::sprintf(
            "Moodist CUDA Allocator can only be used on one device. It was initialized on device %d, but the "
            "requested "
            "allocation is on device %d. This is not supported.",
            deviceIndex, currentDevice));
      }
      CUcontext currentContext = nullptr;
      cuCtxGetCurrent(&currentContext);
      if (!currentContext) {
        CHECK_CU(cuCtxSetCurrent(cuContext));
      } else {
        if (currentContext != cuContext) {
          throw std::runtime_error(fmt::sprintf("Moodist CUDA Allocator CUDA context changed. The Moodist CUDA "
                                                "Allocator can only be used on one CUDA context."));
        }
      }
    }

    constexpr size_t alignment = 0x100;
    size_t alignedbytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);

    CUstream stream = c10::cuda::getCurrentCUDAStream();
    void* cleanupCtx = nullptr;
    uintptr_t cudaPtr = coreApi.cudaAllocatorImplAllocate(impl, alignedbytes, stream, deviceIndex, &cleanupCtx);
    deviceStats.allocated_bytes[0].increase(alignedbytes);
    deviceStats.active_bytes[0].increase(alignedbytes);
    Function<void()> f = [this, cudaPtr, alignedbytes, stream] {
      std::lock_guard l(mutex);
      {
        auto it = freeCallbacks.find(cudaPtr);
        if (it != freeCallbacks.end()) {
          FunctionPointer head = it->second;
          while (head) {
            FunctionPointer next = head->next;
            Function<void()>{head}();
            head = next;
          }
          freeCallbacks.erase(it);
        }
      }
      auto it = streamUses.find(cudaPtr);
      if (it != streamUses.end()) {
        auto& list = it->second;
        list->insert(list->begin(), stream);
        coreApi.cudaAllocatorImplDeallocate(impl, cudaPtr, alignedbytes, list->data(), list->size());
        list->clear();
        streamListFree.push_back(std::move(list));
        streamUses.erase(it);
      } else {
        CUstream streams[1] = {stream};
        coreApi.cudaAllocatorImplDeallocate(impl, cudaPtr, alignedbytes, streams, 1);
      }
      deviceStats.allocated_bytes[0].decrease(alignedbytes);
      deviceStats.active_bytes[0].decrease(alignedbytes);
    };
    auto deleter = [](void* c) {
      Function<void()>(FunctionPointer(c))();
    };
    torch::Device device = torch::Device(torch::kCUDA, deviceIndex);
    return torch::DataPtr((void*)cudaPtr, (void*)f.release(), deleter, device);
  }
  virtual torch::DeleterFnPtr raw_deleter() const override {
    return nullptr;
  }
  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    throw std::runtime_error("moodist CUDAAllocator::copy_data: not implemented");
  }
  HashMap<void*, torch::DataPtr, PointerHash> rawAllocations;
  virtual void* raw_alloc(size_t nbytes) override {
    auto ptr = allocate(nbytes);
    std::lock_guard l(mutex);
    void* r = ptr.get();
    rawAllocations[r] = std::move(ptr);
    return r;
  }
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    throw std::runtime_error("moodist CUDAAllocator::raw_alloc_with_stream: not implemented");
  }
  virtual void raw_delete(void* ptr) override {
    std::unique_lock l(mutex);
    auto i = rawAllocations.find(ptr);
    if (i == rawAllocations.end()) {
      throw std::runtime_error("moodist CUDAAllocator::raw_delete: allocation not found");
    }
    auto h = std::move(i->second);
    rawAllocations.erase(i);
    l.unlock();
  }
  virtual void init(int device_count) override {}
  virtual bool initialized() override {
    throw std::runtime_error("moodist CUDAAllocator::initialized: not implemented");
  }
  virtual double getMemoryFraction(c10::DeviceIndex) {
    return 1.0;
  }
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    throw std::runtime_error("moodist CUDAAllocator::setMemoryFraction: not implemented");
  }
  virtual void enable(bool) {}
  virtual bool isEnabled() const {
    return true;
  }
  virtual void emptyCache() {}
  virtual void emptyCache(at::cuda::MempoolId_t) {}
  virtual void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) override {
    throw std::runtime_error("moodist CUDAAllocator::cacheInfo: not implemented");
  }
  virtual void* getBaseAllocation(void* ptr, size_t* size) override {
    throw std::runtime_error("moodist CUDAAllocator::getBaseAllocation: not implemented");
  }
  virtual void recordStream(const torch::DataPtr& ptr, c10::cuda::CUDAStream stream) override {
    auto& list = streamUses[(uintptr_t)ptr.get()];
    if (!list) {
      if (!streamListFree.empty()) {
        list = std::move(streamListFree.back());
        streamListFree.pop_back();
      } else {
        list = std::make_unique<Vector<CUstream>>();
      }
    }
    list->push_back(stream);
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
  virtual void recordHistory(bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries, c10::cuda::CUDACachingAllocator::RecordContext when) {}
  virtual void recordHistory(bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries, c10::cuda::CUDACachingAllocator::RecordContext when, bool) {}
  virtual void attachOutOfMemoryObserver(c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override {
    throw std::runtime_error("moodist CUDAAllocator::attachOutOfMemoryObserver: not implemented");
  }
  virtual void attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override {
    throw std::runtime_error("moodist CUDAAllocator::attachAllocatorTraceTracker: not implemented");
  }
  virtual void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {}
  virtual cudaError_t memcpyAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count,
      cudaStream_t stream, bool p2p_enabled) override {
    CHECK_CU(cuMemcpyAsync((uintptr_t)dst, (uintptr_t)src, count, stream));
    return cudaSuccess;
  }
  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> getCheckpointState(
      c10::DeviceIndex device, c10::cuda::MempoolId_t id) override {
    throw std::runtime_error("moodist CUDAAllocator::getCheckpointState: not implemented");
  }
  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device, std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) override {
    throw std::runtime_error("moodist CUDAAllocator::setCheckpointPoolState: not implemented");
  }
  virtual std::string name() override {
    return "moodist";
  }
};

namespace {
std::mutex assignmentMutex;
CUDAAllocator* cudaAllocator = nullptr;
} // namespace

namespace allocator {

FreeCallbackHandle addFreeCallback(uintptr_t baseAddress, Function<void()> callback) {
  CHECK(cudaAllocator != nullptr);
  std::lock_guard l(cudaAllocator->mutex);
  FunctionPointer fp = callback.release();
  auto& head = cudaAllocator->freeCallbacks[baseAddress];
  fp->next = head;
  head = fp;
  FreeCallbackHandle r;
  r.baseAddress = baseAddress;
  r.handle = fp;
  return r;
}

void removeFreeCallback(uintptr_t baseAddress, void* handle) {
  CHECK(cudaAllocator != nullptr);
  std::lock_guard l(cudaAllocator->mutex);
  auto** head = &cudaAllocator->freeCallbacks[baseAddress];
  while (head) {
    if (*head == handle) {
      FunctionPointer f = *head;
      *head = (*head)->next;
      Function<void()>{f};
      return;
    }
    head = &(*head)->next;
  }
  throw std::runtime_error("removeFreeCallback: no such handle");
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
