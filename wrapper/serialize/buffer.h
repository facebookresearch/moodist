// Copyright (c) Meta Platforms, Inc. and affiliates.

// Buffer type for serialize library - uses coreApi for allocation
// This is a copy of core/buffer.h but calls through coreApi instead of directly

#pragma once

#include "api/types.h"

#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace moodist {

// Forward declaration - serializeCoreApi is defined in serialize_object.cc
struct CoreApi;
extern CoreApi serializeCoreApi;

struct alignas(std::max_align_t) Buffer : api::Buffer {
  union {
    Buffer* next;
    size_t msize;
  };
  std::byte* data() {
    return (std::byte*)(this + 1);
  }
  const std::byte* data() const {
    return (const std::byte*)(this + 1);
  }
  size_t size() const {
    return msize;
  }
  Buffer() = delete;
  ~Buffer() = delete;

  static Buffer* allocate(size_t nbytes) {
    Buffer* r = (Buffer*)serializeCoreApi.internalAlloc(sizeof(Buffer) + nbytes);
    if (!r) {
      throw std::bad_alloc();
    }
    r->msize = nbytes;
    r->refcount.store(0, std::memory_order_relaxed);
    return r;
  }
  static void deallocate(Buffer* buffer) {
    serializeCoreApi.internalFree((std::byte*)buffer);
  }
  static size_t allocSize(Buffer* buffer) {
    return serializeCoreApi.internalAllocSize((std::byte*)buffer);
  }
};
static_assert(std::is_trivially_copyable_v<Buffer>);

struct BufferHandle {
  Buffer* buffer = nullptr;
  BufferHandle() = default;
  BufferHandle(std::nullptr_t) noexcept {}
  explicit BufferHandle(Buffer* buffer) noexcept : buffer(buffer) {}
  BufferHandle(const BufferHandle&) = delete;
  BufferHandle& operator=(const BufferHandle&) = delete;
  BufferHandle(BufferHandle&& n) noexcept {
    buffer = n.buffer;
    n.buffer = nullptr;
  }
  BufferHandle& operator=(BufferHandle&& n) noexcept {
    std::swap(buffer, n.buffer);
    return *this;
  }
  ~BufferHandle() {
    if (buffer) {
      Buffer::deallocate(buffer);
      buffer = nullptr;
    }
  }
  explicit operator bool() const noexcept {
    return buffer;
  }
  Buffer* operator->() const noexcept {
    return buffer;
  }
  operator Buffer*() const noexcept {
    return buffer;
  }
  Buffer* release() noexcept {
    Buffer* r = buffer;
    buffer = nullptr;
    return r;
  }
};

inline BufferHandle makeBuffer(size_t nbytes) {
  return BufferHandle(Buffer::allocate(nbytes));
}

inline size_t internalAllocSize(Buffer* buffer) {
  return Buffer::allocSize(buffer);
}

} // namespace moodist
