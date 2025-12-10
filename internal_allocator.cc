// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "common.h"
#include "cpu_allocator.h"
#include "hash_map.h"
#include "vector.h"

#include <sys/mman.h>
#include <type_traits>

namespace moodist {

int internalAllocatorNode = -1;

namespace {

void* nalloc(size_t bytes) {
  void* r = numa_alloc_onnode(bytes, internalAllocatorNode);
  if (!r) {
    log.error("ERROR: Failed to allocate %d bytes of memory\n", bytes);
    throw std::bad_alloc();
  }
  CHECK(((uintptr_t)r & 63) == 0);
  return r;
}

void nfree(void* ptr, size_t bytes) {
  return numa_free(ptr, bytes);
}

template<typename T>
struct IntrusiveList {
  T* head = nullptr;

  static T*& next(T* at) noexcept {
    return at->link.second;
  }
  static T*& prev(T* at) noexcept {
    return at->link.first;
  }

  void push_front(T& v) {
    for (auto& x : *this) {
      CHECK(&x != &v);
    }
    prev(&v) = nullptr;
    next(&v) = head;
    if (head) {
      prev(head) = &v;
    }
    head = &v;
  }
  T& front() {
    return *head;
  }
  bool empty() {
    return head == nullptr;
  }
  void clear() {
    head = nullptr;
  }
  struct iterator {
  private:
    T* ptr = nullptr;

  public:
    iterator() = default;
    explicit iterator(T* ptr) : ptr(ptr) {}

    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using iterator_category = std::bidirectional_iterator_tag;

    T& operator*() const noexcept {
      return *ptr;
    }
    T* operator->() const noexcept {
      return ptr;
    }
    iterator& operator++() noexcept {
      ptr = next(ptr);
      return *this;
    }
    iterator operator++(int) noexcept {
      iterator r = (*this);
      ptr = next(ptr);
      return r;
    }
    iterator& operator--() noexcept {
      ptr = prev(ptr);
      return *this;
    }
    iterator operator--(int) noexcept {
      iterator r = (*this);
      ptr = prev(ptr);
      return r;
    }
    bool operator==(iterator n) const noexcept {
      return ptr == n.ptr;
    }
    bool operator!=(iterator n) const noexcept {
      return ptr != n.ptr;
    }
  };
  iterator begin() {
    return iterator(head);
  }
  iterator end() {
    return iterator(nullptr);
  }

  iterator erase(iterator at) noexcept {
    T* nextItem = next(&*at);
    T* prevItem = prev(&*at);
    if (nextItem) {
      prev(nextItem) = prevItem;
    }
    if (prevItem) {
      next(prevItem) = nextItem;
    } else {
      head = nextItem;
    }
    prev(&*at) = nullptr;
    next(&*at) = nullptr;
    return iterator(nextItem);
  }
  void erase(T& item) noexcept {
    erase(iterator(&item));
  }
};

namespace basic {

void* allocate(size_t size) {
  return nalloc(size);
}

void deallocate(void* ptr, size_t size) {
  nfree(ptr, size);
}

template<class T>
struct Allocator {
  typedef T value_type;
  Allocator() = default;
  template<class U>
  constexpr Allocator(const Allocator<U>&) noexcept {}
  T* allocate(size_t n) {
    void* r = basic::allocate(sizeof(T) * n);
    if (!r) {
      throw std::bad_alloc();
    }
    return (T*)r;
  }
  void deallocate(T* p, std::size_t n) noexcept {
    basic::deallocate(p, sizeof(T) * n);
  }
};

} // namespace basic

void* moo_alloc(size_t);
void moo_free(void*);

constexpr size_t alignment = 1024 * 1024 * 2;

struct Thread;
struct Region {
  std::pair<Region*, Region*> link;
  size_t index;
  size_t allocated;
  void* freelist;
  uintptr_t freearea;
  uintptr_t end;
};
struct Thread {
  size_t numFullyAllocatedRegions = 0;
  IntrusiveList<Region> activeRegions[64];
};

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

// template<typename T>
//  struct Indestructible {
//    std::aligned_storage_t<sizeof(T), alignof(T)> storage;
//    Indestructible() {
//      new (&storage) T();
//    }
//    T& operator*() {
//      return (T&)storage;
//    }
//    T* operator->() {
//      return &**this;
//    }
//  };

struct Globals {
  SpinMutex mutex;

  Thread singleThread;

  IntrusiveList<Region> freeRegions;

  Vector<Span, basic::Allocator<Span>> spans;
  Vector<Span, basic::Allocator<Span>> allMappedRegions;
};

// Indestructible<Globals> globals;
AlignedStorage<sizeof(Globals), alignof(Globals)> globalsStorage;
#define globals ((Globals&)globalsStorage)
#define spans (globals.spans)
#define allMappedRegions (globals.allMappedRegions)
#define singleThread (globals.singleThread)
#define freeRegions (globals.freeRegions)
// Globals& globals = (Globals&)globalsStorage;

// Vector<Span, basic::Allocator<Span>>& spans = globals.spans;
// Vector<Span, basic::Allocator<Span>>& allMappedRegions = globals.allMappedRegions;

// Thread& singleThread = globals.singleThread;
// IntrusiveList<Region>& freeRegions = globals.freeRegions;

constexpr size_t regionSize =
    (sizeof(Region) + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) * alignof(std::max_align_t);

void add_span(Vector<Span, basic::Allocator<Span>>& rspans, uintptr_t begin, uintptr_t end) {
  CHECK(end > begin);
  Span* s = rspans.data();
  size_t a = 0;
  size_t b = rspans.size();
  size_t e = rspans.size();
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
          rspans.erase(rspans.begin() + a);
          return;
        }
        s[a - 1].end = end;
        return;
      }
    }
    if (a != e && s[a].begin == end) {
      s[a].begin = begin;
      return;
    }
  }
  CHECK(a <= rspans.size());
  rspans.insert(rspans.begin() + a, Span{begin, end});
}

uintptr_t nextAddr;

void initRegion(Region* r, size_t index, size_t nbytes) {
  r->allocated = 0;
  r->freearea = (uintptr_t)r + std::max(regionSize, std::min((size_t)4096, nbytes));
  r->freelist = nullptr;
  r->index = index;
  singleThread.activeRegions[index].push_front(*r);

  size_t n = 0;
  while (r->freearea + nbytes <= r->end) {
    void* rv = (void*)r->freearea;
    CHECK((uintptr_t)rv / alignment * alignment == (uintptr_t)r);
    r->freearea += nbytes;
    CHECK(r->freearea <= r->end);
    std::memcpy(rv, &r->freelist, sizeof(void*));
    r->freelist = rv;
    ++n;
    if (n >= 1) {
      __builtin_prefetch(rv);
      break;
    }
  }
  CHECK(r->freelist != nullptr);
}

uint64_t rngSeed() {
  uint64_t r = 0x42;
  r += (__rdtsc() * 954311185259313919ul) % 0xffffffffffffl;
  timespec ts{0, 0};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  r += ts.tv_sec * 1000000000l + ts.tv_nsec;
  r += __rdtsc();
  return r;
}

static_assert(sizeof(long) == 8);

uint64_t rngState;
uint64_t rng() {
  if (rngState == 0) {
    rngState = rngSeed();
  }
  rngState = (rngState * 2862933555777941757l) + 1;
  return ((rngState >> 16) & 0xfff) | (rngState << 16);
}

size_t mmappedBytes;

void move_pages() {
  for (auto& v : allMappedRegions) {
    log.debug("moving %#x of %d bytes to numa node %d\n", v.begin, v.end - v.begin, internalAllocatorNode);
    numa_move((void*)v.begin, v.end - v.begin, internalAllocatorNode);
  }
}

void allocate_memory(size_t index) {
  CHECK(index != 0);

  size_t nbytes = std::max(1ul << (64 - index), alignof(std::max_align_t));

  size_t minbytes = regionSize + nbytes;
  size_t alignedminbytes = std::max(alignment, (minbytes + alignment - 1) / alignment * alignment);

  for (auto& v : spans) {
    if (v.end - v.begin >= minbytes) {
      CHECK(v.begin % alignment == 0);
      Region* r = (Region*)v.begin;
      if (v.end - v.begin <= alignedminbytes) {
        r->end = v.end;
        spans.erase(spans.begin() + (&v - spans.data()));
      } else {
        v.begin += alignedminbytes;
        CHECK(v.begin < v.end);
        r->end = v.begin;
      }

      initRegion(r, index, nbytes);
      return;
    }
  }

  size_t bytes = std::max(alignment, (size_t)(alignment + regionSize + nbytes * (nbytes < alignment ? 8 : 2)));
  bytes = std::max(bytes, (mmappedBytes / 4 + alignment - 1) / alignment * alignment);
  bytes = (bytes + alignment - 1) / alignment * alignment;
  void* rv = nullptr;

  uintptr_t addr = (uintptr_t)nextAddr;

  for (size_t i = 0; i != 0x1000; ++i) {
    if (addr) {
      rv = mmap((void*)addr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
      if (rv && rv != MAP_FAILED) {
        nextAddr = addr + bytes;
        if (rv != (void*)addr) {
          nextAddr = (rng() % 0x800000000000l) / alignment * alignment;
        }
        break;
      }
    }
    uintptr_t prevAddr = addr;
    addr = (rng() % 0x800000000000l) / alignment * alignment;
  }

  if (!rv || rv == MAP_FAILED) {
    log.error("fallback allocation of %ld bytes\n", bytes);
    rv = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (!rv || rv == MAP_FAILED) {
      throw std::system_error(errno, std::generic_category(), "mmap");
    }
  }

  if (internalAllocatorNode != -1) {
    numa_move(rv, bytes, internalAllocatorNode);
  }

  mmappedBytes += bytes;
  uintptr_t e = ((uintptr_t)rv + bytes) / alignment * alignment;
  rv = (void*)(((uintptr_t)rv + alignment - 1) / alignment * alignment);
  bytes = e - (uintptr_t)rv;

  log.verbose(
      "(internal) mapped %d bytes at %#x (total %d bytes, %dG) on node %d\n", bytes, (uintptr_t)rv, mmappedBytes,
      mmappedBytes / 1024 / 1024 / 1024, internalAllocatorNode);

  // auto start = std::chrono::steady_clock::now();
  // mlock(rv, bytes);

  // log.error("locked in %gs\n", seconds(std::chrono::steady_clock::now() - start));

  CHECK(bytes > 0);

  add_span(spans, (uintptr_t)rv, (uintptr_t)rv + bytes);
  add_span(allMappedRegions, (uintptr_t)rv, (uintptr_t)rv + bytes);

  allocate_memory(index);
}

[[gnu::noinline]] void* moo_alloc_slow(size_t bytes, size_t index) {
  bytes = (bytes + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) * alignof(std::max_align_t);
  size_t nbytes = std::max(1ul << (64 - index), (size_t)alignof(std::max_align_t));
  CHECK(singleThread.activeRegions[index].empty());
  if (!freeRegions.empty()) {
    size_t abytes = std::max(alignment, (size_t)(alignment + regionSize + nbytes * (nbytes < alignment ? 8 : 2)));
    for (auto& v : freeRegions) {
      Region* r = &v;
      size_t size = r->end - ((uintptr_t)r + regionSize);
      if (size >= nbytes && size <= abytes * 2) {
        freeRegions.erase(v);
        if (r->index != index) {
          add_span(spans, (uintptr_t)&v, v.end);
          allocate_memory(index);
        } else {
          singleThread.activeRegions[index].push_front(*r);
        }
        return moo_alloc(bytes);
      }
    }
    for (auto& v : freeRegions) {
      add_span(spans, (uintptr_t)&v, v.end);
    }
    freeRegions.clear();
  }
  allocate_memory(index);
  return moo_alloc(bytes);
}

size_t currentLiveAllocations = 0;

[[gnu::noinline]] void* moo_alloc(size_t bytes) {
  size_t index = __builtin_ia32_lzcnt_u64(
      (std::max(bytes, (size_t)1) + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) *
          alignof(std::max_align_t) -
      1);
  CHECK(index && index < 64);
  Thread* thread = &singleThread;
  if (!thread->activeRegions[index].empty()) [[likely]] {
    Region* r = &thread->activeRegions[index].front();
    CHECK(r->index == index);
    ++r->allocated;
    void* rv = r->freelist;
    CHECK(rv != nullptr);
    std::memcpy(&r->freelist, rv, sizeof(void*));
    if (!r->freelist) {
      size_t nbytes = std::max(1ul << (64 - index), alignof(std::max_align_t));
      if (r->freearea + nbytes <= r->end) {
        void* rv = (void*)r->freearea;
        CHECK((uintptr_t)rv / alignment * alignment == (uintptr_t)r);
        r->freearea += nbytes;
        CHECK(r->freearea <= r->end);
        std::memcpy(rv, &r->freelist, sizeof(void*));
        r->freelist = rv;
        __builtin_prefetch(rv);
      } else {
        ++thread->numFullyAllocatedRegions;
        thread->activeRegions[index].erase(*r);
      }
    } else {
      __builtin_prefetch(r->freelist);
    }
    CHECK((uintptr_t)rv / alignment * alignment == (uintptr_t)r);
    CHECK(index);
    CHECK(r->index == index);
    ++currentLiveAllocations;
    return rv;
  }
  return moo_alloc_slow(bytes, index);
}

[[gnu::noinline]] void free_internal_free_region(Thread* thread, Region* r) {
  thread->activeRegions[r->index].erase(*r);
  freeRegions.push_front(*r);
}

[[gnu::always_inline]] inline void free_internal(Thread* thread, Region* r, void* p) {
  CHECK(r->allocated >= 1);
  --r->allocated;
  if (!r->freelist) {
    --thread->numFullyAllocatedRegions;
    thread->activeRegions[r->index].push_front(*r);
  }
  std::memcpy(p, &r->freelist, sizeof(void*));
  r->freelist = p;

  if (r->allocated == 0) [[unlikely]] {
    free_internal_free_region(thread, r);
  }
}

[[gnu::noinline]] void moo_free(void* p) {
  Thread* thread = &singleThread;
  uintptr_t a = (uintptr_t)p / alignment * alignment;
  Region* r = (Region*)a;
  --currentLiveAllocations;
  free_internal(thread, r, p);
}

size_t moo_alloc_size(void* p) {
  uintptr_t a = (uintptr_t)p / alignment * alignment;
  Region* r = (Region*)a;
  return std::max(1ul << (64 - r->index), alignof(std::max_align_t));
}

} // namespace

[[gnu::malloc]]
void* internalAlloc(size_t bytes) {
  std::lock_guard l(globals.mutex);
  return moo_alloc(bytes);
}
void internalFree(void* ptr) {
  std::lock_guard l(globals.mutex);
  return moo_free(ptr);
}
size_t internalAllocSize(void* ptr) {
  return moo_alloc_size(ptr);
}

void internalAllocatorSetNode(int node) {
  std::lock_guard l(globals.mutex);
  if (internalAllocatorNode == node) {
    return;
  }
  log.debug("allocator node changed from %d to %d\n", internalAllocatorNode, node);
  internalAllocatorNode = node;
  move_pages();
}

} // namespace moodist
