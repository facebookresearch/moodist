// Copyright (c) Meta Platforms, Inc. and affiliates.

// CPU Allocator wrapper - contains CpuAllocator class and torch-dependent code.
// This file is compiled into _C.so (wrapper library).

#include "api/cpu_allocator.h"
#include "moodist_loader.h"
#include "torch_includes.h"

#include <mutex>

namespace moodist {

namespace {

struct CpuAllocator : torch::Allocator {
  virtual torch::DataPtr allocate(size_t bytes) override {
    void* cleanupCtx = nullptr;
    void* ptr = coreApi.cpuAllocatorAlloc(bytes, &cleanupCtx);

    auto deleter = [](void* ctx) {
      coreApi.cpuAllocatorFree(ctx);
    };

    torch::Device device(torch::kCPU);
    return torch::DataPtr(ptr, cleanupCtx, deleter, device);
  }

  virtual torch::DeleterFnPtr raw_deleter() const override {
    return nullptr;
  }

  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }
};

std::mutex assignmentMutex;
CpuAllocator* cpuAllocator = nullptr;

} // namespace

void enableCpuAllocator() {
  std::lock_guard l(assignmentMutex);
  if (!cpuAllocator) {
    cpuAllocator = new CpuAllocator();
  }
  torch::SetAllocator(torch::kCPU, cpuAllocator);
}

} // namespace moodist
