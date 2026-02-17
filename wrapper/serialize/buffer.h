// Copyright (c) Meta Platforms, Inc. and affiliates.

// Buffer handle for serialize library.
// Allocates via cpuAllocatorAlloc so serialized data is directly usable by queues.
// Data pointer = allocation pointer (no header), registered in cpu_allocator sharedHandles.

#pragma once

#include "api/moodist_api.h"

#include <cstddef>
#include <new>
#include <utility>

namespace moodist {

extern CoreApi serializeCoreApi;

struct BufferHandle {
  std::byte* ptr = nullptr;
  size_t msize = 0;
  void* cleanupCtx = nullptr;

  BufferHandle() = default;
  BufferHandle(std::nullptr_t) noexcept {}
  BufferHandle(const BufferHandle&) = delete;
  BufferHandle& operator=(const BufferHandle&) = delete;
  BufferHandle(BufferHandle&& n) noexcept : ptr(n.ptr), msize(n.msize), cleanupCtx(n.cleanupCtx) {
    n.ptr = nullptr;
    n.cleanupCtx = nullptr;
  }
  BufferHandle& operator=(BufferHandle&& n) noexcept {
    std::swap(ptr, n.ptr);
    std::swap(msize, n.msize);
    std::swap(cleanupCtx, n.cleanupCtx);
    return *this;
  }
  ~BufferHandle() {
    if (cleanupCtx) {
      serializeCoreApi.cpuAllocatorFree(cleanupCtx);
      ptr = nullptr;
      cleanupCtx = nullptr;
    }
  }

  explicit operator bool() const noexcept {
    return ptr;
  }

  // operator-> returns this so buffer->msize and buffer->data() keep working
  BufferHandle* operator->() noexcept {
    return this;
  }
  const BufferHandle* operator->() const noexcept {
    return this;
  }

  std::byte* data() const noexcept {
    return ptr;
  }
  size_t size() const noexcept {
    return msize;
  }
};

inline BufferHandle makeBuffer(size_t nbytes) {
  BufferHandle h;
  h.ptr = (std::byte*)serializeCoreApi.cpuAllocatorAlloc(nbytes, &h.cleanupCtx);
  if (!h.ptr) {
    throw std::bad_alloc();
  }
  h.msize = nbytes;
  return h;
}

inline size_t internalAllocSize(const BufferHandle& h) {
  return serializeCoreApi.internalAllocSize(h.ptr);
}

} // namespace moodist
