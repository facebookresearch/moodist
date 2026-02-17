// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "api/cpu_allocator.h"
#include "common.h"
#include "hash_map.h"

namespace moodist {

namespace {

template<typename T>
struct Indestructible {
  std::aligned_storage_t<sizeof(T), alignof(T)> storage;
  Indestructible() {
    new (&storage) T();
  }
  T& operator*() {
    return (T&)storage;
  }
  T* operator->() {
    return &**this;
  }
};

struct Globals {
  SpinMutex mutex;
  HashMap<uintptr_t, AllocatedCpuBufferSharedPtr> sharedHandles;
};

Indestructible<Globals> globals;

// CoreApi function: allocate and register handle, return ptr and cleanup context
void* cpuAllocatorAllocImpl(size_t bytes, void** cleanupCtx) {
  if (bytes == 0) {
    *cleanupCtx = nullptr;
    return nullptr;
  }
  AllocatedCpuBufferSharedPtr handle = AllocatedCpuBufferSharedPtr::make();

  void* ptr = internalAlloc(bytes);

  handle->cpuPointer = ptr;
  handle->bytes = bytes;

  CHECK(handle.ptr->refcount == 1);

  {
    std::lock_guard l(globals->mutex);
    globals->sharedHandles[(uintptr_t)ptr] = std::move(handle);
  }

  // Create cleanup function that removes from sharedHandles
  Function<void()> f = [ptr] {
    std::unique_lock l(globals->mutex);
    auto i = globals->sharedHandles.find((uintptr_t)ptr);
    CHECK(i != globals->sharedHandles.end());
    auto handle = std::move(i->second);
    globals->sharedHandles.erase(i);
    l.unlock();
    // handle destructor calls cpu_allocator::moo_free → internalFree
  };

  *cleanupCtx = (void*)f.release();
  return ptr;
}

// CoreApi function: deleter for DataPtr - calls cleanup function
void cpuAllocatorFreeImpl(void* cleanupCtx) {
  if (cleanupCtx) {
    Function<void()>(FunctionPointer(cleanupCtx))();
  }
}

} // namespace

void cpuAllocatorDebug() {
  std::lock_guard l(globals->mutex);
  log.error("cpu_allocator sharedHandles: %ld entries\n", globals->sharedHandles.size());
}

namespace cpu_allocator {

std::pair<uintptr_t, size_t> regionAt(uintptr_t address) {
  return internalRegionAt(address);
}
bool owns(uintptr_t address) {
  return internalOwns(address);
}
bool owns(const void* ptr) {
  return owns((uintptr_t)ptr);
}
std::pair<uintptr_t, size_t> regionAt(const void* ptr) {
  return regionAt((uintptr_t)ptr);
}

void* moo_alloc(size_t bytes) {
  return internalAlloc(bytes);
}
void moo_free(void* ptr) {
  internalFree(ptr);
}

AllocatedCpuBufferSharedPtr getCpuBuffer(uintptr_t address) {
  std::lock_guard l(globals->mutex);
  auto i = globals->sharedHandles.find(address);
  if (i != globals->sharedHandles.end()) {
    return i->second;
  }
  return nullptr;
}

void refCpuBuffer(AllocatedCpuBufferSharedPtr ptr) {
  std::lock_guard l(globals->mutex);
  bool success = globals->sharedHandles.try_emplace((uintptr_t)ptr->cpuPointer, std::move(ptr)).second;
  CHECK(success);
}
void derefCpuBuffer(uintptr_t address) {
  std::lock_guard l(globals->mutex);
  auto i = globals->sharedHandles.find(address);
  CHECK(i != globals->sharedHandles.end());
  auto handle = std::move(i->second);
  globals->sharedHandles.erase(i);
}

} // namespace cpu_allocator

// Public CoreApi functions (called via function pointers from _C.so)
void* cpuAllocatorAlloc(size_t bytes, void** cleanupCtx) {
  return cpuAllocatorAllocImpl(bytes, cleanupCtx);
}

void cpuAllocatorFree(void* cleanupCtx) {
  cpuAllocatorFreeImpl(cleanupCtx);
}

bool cpuAllocatorOwns(uintptr_t address) {
  return cpu_allocator::owns(address);
}

} // namespace moodist
