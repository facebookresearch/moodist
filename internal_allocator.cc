// Copyright (c) Meta Platforms, Inc. and affiliates.

// Internal allocator implementation
// Region-based allocator with size classes, optimized for low-latency allocation/free

#include "common.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include <pthread.h>
#include <sys/mman.h>
#include <time.h>
#include <x86intrin.h>

namespace moodist {

namespace internal_allocator {

#include "moo_allocator_impl.h"

}

// ============================================================================
// Public API
// ============================================================================

#ifdef MOODIST_USE_STDLIB_ALLOCATOR
// Use stdlib malloc for ASan compatibility
#include <malloc.h>
[[gnu::malloc]]
void* internalAlloc(size_t bytes) {
  return std::malloc(bytes);
}
void internalFree(void* ptr) {
  std::free(ptr);
}
size_t internalAllocSize(void* ptr) {
  return malloc_usable_size(ptr);
}
void internalAllocatorSetNode(int) {
  // No-op with stdlib malloc
}
#elif defined(MOODIST_ALLOCATOR_GUARDS)
// Add guard bytes to detect buffer overflows and track allocations for double-free detection
#include <bit>
#include <mutex>
#include <unordered_set>

static constexpr uint64_t GUARD_MAGIC = 0xDEADBEEFCAFEBABEULL;
static constexpr uint64_t FREED_MAGIC = 0xFEEDFACEFEEDFACEULL;
static constexpr size_t GUARD_SIZE = 64; // bytes of guard region

struct GuardHeader {
  void* rawPtr; // Original allocation pointer (for freeing)
  size_t requestedSize;
  uint64_t magic;
};

// Use heap-allocated statics that are never destroyed to avoid crashes
// during program exit when other threads are still using the allocator
static std::mutex& getGuardsMutex() {
  static std::mutex* m = new std::mutex();
  return *m;
}

static std::unordered_set<void*>& getLiveAllocations() {
  static std::unordered_set<void*>* s = new std::unordered_set<void*>();
  return *s;
}

[[gnu::malloc]]
void* internalAlloc(size_t bytes) {
  // Preserve alignment property: align to next power of 2 up to 4096
  size_t alignment = std::min((size_t)4096, std::bit_ceil(std::max(bytes, (size_t)8)));

  // Need space for: alignment padding + header + user data + guard
  size_t totalSize = alignment + sizeof(GuardHeader) + bytes + GUARD_SIZE;
  void* raw = internal_allocator::moo_alloc(totalSize);
  if (!raw) {
    return nullptr;
  }

  // Calculate aligned user pointer position
  // User pointer must be aligned, and header must fit before it
  uintptr_t rawAddr = (uintptr_t)raw;
  uintptr_t userAddr = (rawAddr + sizeof(GuardHeader) + alignment - 1) & ~(alignment - 1);
  char* userPtr = (char*)userAddr;

  // Header is immediately before userPtr
  GuardHeader* hdr = (GuardHeader*)userPtr - 1;
  hdr->rawPtr = raw;
  hdr->requestedSize = bytes;
  hdr->magic = GUARD_MAGIC;

  // Fill trailing guard with magic pattern
  uint64_t* guard = (uint64_t*)(userPtr + bytes);
  for (size_t i = 0; i < GUARD_SIZE / sizeof(uint64_t); ++i) {
    guard[i] = GUARD_MAGIC;
  }

  // Track this allocation
  {
    std::lock_guard<std::mutex> lock(getGuardsMutex());
    getLiveAllocations().insert(userPtr);
  }

  return userPtr;
}

void internalFree(void* ptr) {
  if (!ptr) {
    return;
  }

  // Check if this pointer was allocated by us
  {
    std::lock_guard<std::mutex> lock(getGuardsMutex());
    auto& liveAllocations = getLiveAllocations();
    auto it = liveAllocations.find(ptr);
    if (it == liveAllocations.end()) {
      // Print error first before accessing potentially invalid memory
      log.error("INVALID FREE: ptr=%p was never allocated or was already freed!\n", ptr);
      // Try to check if it might be a double-free (may segfault if ptr is garbage)
      GuardHeader* hdr = (GuardHeader*)ptr - 1;
      if (hdr->magic == FREED_MAGIC) {
        log.error("  (header indicates double-free)\n");
      }
      std::abort();
    }
    liveAllocations.erase(it);
  }

  GuardHeader* hdr = (GuardHeader*)ptr - 1;

  // Check header magic
  if (hdr->magic != GUARD_MAGIC) {
    log.error("HEAP CORRUPTION: header magic corrupted! ptr=%p expected=%llx got=%llx\n", ptr,
        (unsigned long long)GUARD_MAGIC, (unsigned long long)hdr->magic);
    std::abort();
  }

  // Check trailing guard
  char* userPtr = (char*)ptr;
  uint64_t* guard = (uint64_t*)(userPtr + hdr->requestedSize);
  for (size_t i = 0; i < GUARD_SIZE / sizeof(uint64_t); ++i) {
    if (guard[i] != GUARD_MAGIC) {
      log.error("HEAP CORRUPTION: buffer overflow detected! ptr=%p size=%zu guard[%zu]=%llx\n", ptr, hdr->requestedSize,
          i, (unsigned long long)guard[i]);
      std::abort();
    }
  }

  // Mark as freed (helps detect double-free even after tracking set is cleared)
  hdr->magic = FREED_MAGIC;

  internal_allocator::moo_free(hdr->rawPtr);
}

size_t internalAllocSize(void* ptr) {
  if (!ptr) {
    return 0;
  }
  GuardHeader* hdr = (GuardHeader*)ptr - 1;
  return hdr->requestedSize;
}

void internalAllocatorSetNode(int node) {
  if (internal_allocator::allocatorNode == node) {
    return;
  }
  log.debug("allocator node changed from %d to %d\n", internal_allocator::allocatorNode, node);
  internal_allocator::allocatorNode = node;
  internal_allocator::move_pages();
}
#else
// Default: use the new allocator directly (thread-safe, no locking needed)
[[gnu::malloc]]
void* internalAlloc(size_t bytes) {
  return internal_allocator::moo_alloc(bytes);
}

void internalFree(void* ptr) {
  internal_allocator::moo_free(ptr);
}

size_t internalAllocSize(void* ptr) {
  return internal_allocator::moo_alloc_size(ptr);
}

void internalAllocatorSetNode(int node) {
  if (internal_allocator::allocatorNode == node) {
    return;
  }
  log.debug("allocator node changed from %d to %d\n", internal_allocator::allocatorNode, node);
  internal_allocator::allocatorNode = node;
  internal_allocator::move_pages();
}
#endif

} // namespace moodist
