// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"
#include "commondefs.h"

#include <atomic>
#include <type_traits>

namespace moodist {

struct alignas(std::max_align_t) Buffer {
  union {
    Buffer* next;
    size_t msize;
  };
  std::atomic_uint32_t refcount;
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
    Buffer* r = (Buffer*)internalAlloc(sizeof(Buffer) + nbytes);
    if (!r) {
      throw std::bad_alloc();
    }
    r->msize = nbytes;
    r->refcount.store(0, std::memory_order_relaxed);
    return r;
  }
  static void deallocate(Buffer* buffer) {
    internalFree((std::byte*)buffer);
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
struct SharedBufferHandle {
  Buffer* buffer = nullptr;
  SharedBufferHandle() = default;
  SharedBufferHandle(std::nullptr_t) noexcept {}
  explicit SharedBufferHandle(Buffer* buffer) noexcept : buffer(buffer) {
    if (buffer) {
      CHECK(buffer->refcount.load(std::memory_order_relaxed) == 0);
      addref();
    }
  }
  SharedBufferHandle(const SharedBufferHandle& n) noexcept {
    buffer = n.buffer;
    if (buffer) {
      addref();
    }
  }
  SharedBufferHandle& operator=(const SharedBufferHandle& n) noexcept {
    buffer = n.buffer;
    if (buffer) {
      addref();
    }
    return *this;
  }
  SharedBufferHandle(SharedBufferHandle&& n) noexcept {
    buffer = n.buffer;
    n.buffer = nullptr;
  }
  SharedBufferHandle& operator=(SharedBufferHandle&& n) noexcept {
    std::swap(buffer, n.buffer);
    return *this;
  }
  ~SharedBufferHandle() {
    if (buffer && decref() == 0) {
      Buffer::deallocate(buffer);
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
  int addref() noexcept {
    return buffer->refcount.fetch_add(1, std::memory_order_acquire) + 1;
  }
  int decref() noexcept {
    return buffer->refcount.fetch_sub(1) - 1;
  }
  Buffer* release() noexcept {
    Buffer* r = buffer;
    buffer = nullptr;
    return r;
  }
  void acquire(Buffer* buffer) noexcept {
    buffer = buffer;
  }
};

inline BufferHandle makeBuffer(size_t nbytes) {
  return BufferHandle(Buffer::allocate(nbytes));
}

} // namespace moodist
