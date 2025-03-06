
#include "cpu_allocator.h"
#include "common.h"
#include "hash_map.h"
#include "vector.h"

#include <c10/cuda/CUDAFunctions.h>
#include <sys/mman.h>
#include <type_traits>

namespace moodist {

namespace {

void* nalloc(size_t bytes) {
  void* r = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (!r) {
    log.error("ERROR: Failed to allocate %d bytes of memory\n", bytes);
    throw std::bad_alloc();
  }
  CHECK(((uintptr_t)r & 63) == 0);
  return r;
}

void nfree(void* ptr, size_t bytes) {
  munmap(ptr, bytes);
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
    bool found = false;
    for (auto& v : *this) {
      if (&v == &*at) {
        found = true;
      }
    }
    CHECK(found);
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
  Thread* thread;
  std::pair<Region*, Region*> link;
  size_t index;
  size_t allocated;
  void* freelist;
  uintptr_t freearea;
  uintptr_t end;
};
struct Thread {
  std::atomic_size_t nFrees = 0;
  SpinMutex mutex;
  void* queuedFrees = nullptr;
  bool running = true;
  size_t numFullyAllocatedRegions = 0;
  IntrusiveList<Region> activeRegions[64];
};

Thread singleThread;

IntrusiveList<Region> freeRegions;

Thread* currentThread() {
  return &singleThread;
}

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

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

  Vector<Span, basic::Allocator<Span>> spans;
  Vector<Span, basic::Allocator<Span>> allMappedRegions;

  HashMap<uintptr_t, AllocatedCpuBufferSharedPtr> sharedHandles;
};

Indestructible<Globals> globals;

Vector<Span, basic::Allocator<Span>>& spans = globals->spans;
Vector<Span, basic::Allocator<Span>>& allMappedRegions = globals->allMappedRegions;

constexpr size_t regionSize =
    (sizeof(Region) + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) * alignof(std::max_align_t);

void add_span(Vector<Span, basic::Allocator<Span>>& spans, uintptr_t begin, uintptr_t end) {
  CHECK(end > begin);
  Span* s = spans.data();
  size_t a = 0;
  size_t b = spans.size();
  size_t e = spans.size();
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
          spans.erase(spans.begin() + a);
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
  CHECK(a <= spans.size());
  spans.insert(spans.begin() + a, Span{begin, end});
}

uintptr_t nextAddr;

void initRegion(Region* r, size_t index, size_t nbytes) {
  r->allocated = 0;
  r->freearea = (uintptr_t)r + std::max(regionSize, std::min((size_t)4096, nbytes));
  r->freelist = nullptr;
  r->index = index;
  r->thread = currentThread();
  r->thread->activeRegions[index].push_front(*r);

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
  r += __rdtsc();
  timespec ts{0, 0};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  r += ts.tv_sec + 1000000000l + ts.tv_nsec;
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

extern "C" void allocate_memory(size_t index) {
  // printf("allocate memory %d (%#llx)! -> %p\n", (int)n, 1ull << (64 - n), r);
  // printf("base is %p\n", base);

  CHECK(index != 0);

  size_t nbytes = std::max(1ul << (64 - index), alignof(std::max_align_t));

  // log.debug("allocate_memory(%ld) nbytes %d\n", index, nbytes);

  size_t minbytes = regionSize + nbytes;
  size_t alignedminbytes = std::max(alignment, (minbytes + alignment - 1) / alignment * alignment);

  for (auto& v : spans) {
    if (v.end - v.begin >= minbytes) {
      CHECK(v.begin % alignment == 0);
      Region* r = (Region*)v.begin;
      if (v.end - v.begin <= alignedminbytes) {
        // printf("grab entire span! %ld\n", v.end - v.begin);
        r->end = v.end;
        spans.erase(spans.begin() + (&v - spans.data()));
        // printf("%d spans:\n", spans.size());
        // for (auto& v : spans) {
        //   printf("[%#lx, %#lx)\n", v.begin, v.end);
        // }
      } else {
        // printf("grab alignedminbytes %ld\n", alignedminbytes);
        v.begin += alignedminbytes;
        CHECK(v.begin < v.end);
        r->end = v.begin;
      }

      // log.debug("new region %d at %p\n", index, (void*)r);

      initRegion(r, index, nbytes);
      return;
    }
  }

  size_t bytes = std::max(alignment, (size_t)(alignment + regionSize + nbytes * (nbytes < alignment ? 8 : 2)));
  bytes = std::max(bytes, (mmappedBytes / 4 + alignment - 1) / alignment * alignment);
  // if (nbytes >= 1024 * 1024 * 256) {
  //   bytes = nbytes + alignof(std::max_align_t);
  // }
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
    printf("fallback allocation of %ld bytes\n", bytes);
    rv = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (!rv || rv == MAP_FAILED) {
      throw std::system_error(errno, std::generic_category(), "mmap");
    }
  }

  mmappedBytes += bytes;
  uintptr_t e = ((uintptr_t)rv + bytes) / alignment * alignment;
  rv = (void*)(((uintptr_t)rv + alignment - 1) / alignment * alignment);
  bytes = e - (uintptr_t)rv;

  log.error(
      "mapped %d bytes at %#x (total %d bytes, %dG)\n", bytes, (uintptr_t)rv, mmappedBytes,
      mmappedBytes / 1024 / 1024 / 1024);

  // madvise(rv, bytes, MADV_HUGEPAGE);

  auto start = std::chrono::steady_clock::now();
  mlock(rv, bytes);

  log.error("locked in %gs\n", seconds(std::chrono::steady_clock::now() - start));

  // numa_tonode_memory(rv, bytes, allocationNode);
  // start = std::chrono::steady_clock::now();

  // {
  //   int deviceIndex = c10::cuda::current_device();
  //   CUcontext cuContext;
  //   CUdevice cuDevice;
  //   cuCtxGetCurrent(&cuContext);
  //   if (!cuContext) {
  //     CHECK_CU(cuInit(0));

  //     CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
  //     CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
  //     CHECK_CU(cuCtxSetCurrent(cuContext));
  //   } else {
  //     CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
  //   }

  //   CHECK_CU(cuMemHostRegister(rv, bytes, CU_MEMHOSTREGISTER_DEVICEMAP));
  // }

  // log.error("registered in %gs\n", seconds(std::chrono::steady_clock::now() - start));

  CHECK(bytes > 0);

  add_span(spans, (uintptr_t)rv, (uintptr_t)rv + bytes);
  add_span(allMappedRegions, (uintptr_t)rv, (uintptr_t)rv + bytes);

  allocate_memory(index);
}

[[gnu::noinline]] void* moo_alloc_slow(size_t bytes, size_t index) {
  bytes = (bytes + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) * alignof(std::max_align_t);
  size_t nbytes = std::max(1ul << (64 - index), (size_t)alignof(std::max_align_t));
  CHECK(currentThread()->activeRegions[index].empty());
  if (!freeRegions.empty()) {
    size_t abytes = std::max(alignment, (size_t)(alignment + regionSize + nbytes * (nbytes < alignment ? 8 : 2)));
    for (auto& v : freeRegions) {
      Region* r = &v;
      size_t size = r->end - ((uintptr_t)r + regionSize);
      if (size >= nbytes && size <= abytes * 2) {
        freeRegions.erase(v);
        CHECK(v.thread == nullptr);
        if (r->index != index) {
          add_span(spans, (uintptr_t)&v, v.end);
          allocate_memory(index);
        } else {
          r->thread = currentThread();
          r->thread->activeRegions[index].push_front(*r);
        }
        return moo_alloc(bytes);
      }
    }
    // printf("reset spans!\n");
    for (auto& v : freeRegions) {
      add_span(spans, (uintptr_t)&v, v.end);
    }
    freeRegions.clear();
  }
  allocate_memory(index);
  return moo_alloc(bytes);
}

[[gnu::noinline]] void* moo_alloc_deferred_frees(Thread* thread, size_t bytes) {
  std::unique_lock l(thread->mutex);
  void* q = std::exchange(thread->queuedFrees, nullptr);
  thread->nFrees = 0;
  l.unlock();
  while (q) {
    void* n;
    std::memcpy(&n, q, sizeof(void*));
    moo_free(q);
    q = n;
  }
  return moo_alloc(bytes);
}

size_t currentLiveAllocations = 0;

[[gnu::noinline]] void* moo_alloc(size_t bytes) {
  Thread* thread = currentThread();
  if (thread->nFrees.load(std::memory_order_relaxed)) [[unlikely]] {
    CHECK(false);
    return moo_alloc_deferred_frees(thread, bytes);
  }
  size_t index = __builtin_ia32_lzcnt_u64(
      (std::max(bytes, (size_t)1) + alignof(std::max_align_t) - 1) / alignof(std::max_align_t) *
          alignof(std::max_align_t) -
      1);
  CHECK(index && index < 64);
  if (!thread->activeRegions[index].empty()) [[likely]] {
    Region* r = &thread->activeRegions[index].front();
    CHECK(r->thread == thread);
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
    // printf("fast-alloc %p from region %d at %p\n", rv, r->index, r);
    CHECK((uintptr_t)rv / alignment * alignment == (uintptr_t)r);
    CHECK(index);
    CHECK(r->index == index);
    ++currentLiveAllocations;
    return rv;
  }
  // printf("moo_alloc(%ld) -> index %ld\n", bytes, index);
  return moo_alloc_slow(bytes, index);
}

[[gnu::noinline]] void free_internal_free_region(Thread* thread, Region* r) {
  CHECK(r->freelist != nullptr);
  // log.debug("thread %p give up region %p index %ld\n", (void*)thread, (void*)r, r->index);
  bool found = false;
  for (auto& v : thread->activeRegions[r->index]) {
    if (&v == r) {
      found = true;
    }
  }
  CHECK(found);
  thread->activeRegions[r->index].erase(*r);
  r->thread = nullptr;
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

[[gnu::noinline]] void moo_free_other_thread(Region* r, void* p) {
  Thread* thread = r->thread;
  std::unique_lock l(thread->mutex);
  if (thread->running) [[likely]] {
    std::memcpy(p, &r->thread->queuedFrees, sizeof(void*));
    thread->queuedFrees = p;
    ++thread->nFrees;
  } else {
    free_internal(thread, r, p);
    if (thread->numFullyAllocatedRegions == 0) {
      bool empty = true;
      for (auto& v : thread->activeRegions) {
        if (!v.empty()) {
          empty = false;
          break;
        }
      }
      if (empty) {
        l.unlock();
        munmap(thread, sizeof(Thread));
      }
    }
  }
}

[[gnu::noinline]] void moo_free(void* p) {
  Thread* thread = currentThread();
  uintptr_t a = (uintptr_t)p / alignment * alignment;
  Region* r = (Region*)a;
  if (r->thread == thread) [[likely]] {
    --currentLiveAllocations;
    free_internal(thread, r, p);
  } else {
    CHECK(false);
    return moo_free_other_thread(r, p);
  }
}

size_t moo_alloc_size(void* p) {
  uintptr_t a = (uintptr_t)p / alignment * alignment;
  Region* r = (Region*)a;
  return std::max(1ul << (64 - r->index), alignof(std::max_align_t));
}

struct CpuAllocator : torch::Allocator {

  int deviceIndex = -1;

  virtual torch::DataPtr allocate(size_t bytes) override {
    if (bytes == 0) {
      return torch::DataPtr(nullptr, nullptr, nullptr, torch::Device(torch::kCPU));
    }
    AllocatedCpuBufferSharedPtr handle = AllocatedCpuBufferSharedPtr::make();

    std::lock_guard l(globals->mutex);

    // if (deviceIndex == -1) {
    //   log.init();
    //   deviceIndex = c10::cuda::current_device();
    //   CHECK(deviceIndex != -1);

    //   if (numa_available() == -1) {
    //     throw std::runtime_error("Moodist CPU Allocator cannot be used since NUMA is not available.");
    //   }

    //   allocationNode = getNode(c10::cuda::current_device());
    //   if (allocationNode == -1) {
    //     throw std::runtime_error(fmt::sprintf(
    //         "Moodist CPU Allocator failed to find a NUMA node associated with device index %d.", deviceIndex));
    //   }
    //   long long free = 0;
    //   long long total = numa_node_size(allocationNode, &free);
    //   log.verbose(
    //       "CPU Allocator: CUDA device is %d, allocating CPU memory on NUMA node %d. "
    //       "Node has %d total and %d free bytes of memory.",
    //       deviceIndex, allocationNode, total, free);
    // }
    // int currentDeviceIndex = c10::cuda::current_device();
    // if (deviceIndex != currentDeviceIndex) {
    //   throw std::runtime_error(fmt::sprintf(
    //       "Moodist CPU Allocator was initialized with CUDA device %d, but the current CUDA device is %d. "
    //       "Please set the CUDA device before allocating any tensors, and do not change it.",
    //       deviceIndex, currentDeviceIndex));
    // }

    void* ptr = moo_alloc(bytes);

    handle->cpuPointer = ptr;
    handle->bytes = bytes;

    CHECK(handle.ptr->refcount == 1);

    globals->sharedHandles[(uintptr_t)ptr] = std::move(handle);

    Function<void()> f = [this, ptr] {
      std::unique_lock l(globals->mutex);
      auto i = globals->sharedHandles.find((uintptr_t)ptr);
      CHECK(i != globals->sharedHandles.end());
      auto handle = std::move(i->second);
      globals->sharedHandles.erase(i);
      l.unlock();
    };

    // void* ptr = moo_alloc(bytes);
    // Function<void()> f = [this, ptr] { moo_free(ptr); };

    auto deleter = [](void* c) { Function<void()>(FunctionPointer(c))(); };
    torch::Device device(torch::kCPU);
    return torch::DataPtr((void*)ptr, (void*)f.release(), deleter, device);
  }
  virtual torch::DeleterFnPtr raw_deleter() const override {
    return nullptr;
  }
  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    throw std::runtime_error("moodist CpuAllocator::copy_data: not implemented");
  }
};

std::mutex assignmentMutex;
CpuAllocator* cpuAllocator = nullptr;

void* moo_alloc2(size_t bytes) {
  std::lock_guard l(globals->mutex);
  return moo_alloc(bytes);
}
void moo_free2(void* ptr) {
  std::lock_guard l(globals->mutex);
  return moo_free(ptr);
}

} // namespace

void enableCpuAllocator() {
  std::lock_guard l(assignmentMutex);
  if (!cpuAllocator) {
    cpuAllocator = new CpuAllocator();
  }
  torch::SetAllocator(torch::kCPU, cpuAllocator);
}

void cpuAllocatorDebug() {
  auto* thread = currentThread();
  for (size_t i = 0; i != 64; ++i) {
    size_t n = std::distance(thread->activeRegions[i].begin(), thread->activeRegions[i].end());
    if (n == 0) {
      continue;
    }
    log.error("index %ld, %ld active regions:\n", i, n);
  }
  log.error("%d free regions\n", std::distance(freeRegions.begin(), freeRegions.end()));

  log.error("%ld spans:\n", spans.size());
  size_t bytes = 0;
  for (auto& v : spans) {
    log.error("  [%#lx, %#lx)\n", v.begin, v.end);
    bytes += v.end - v.begin;
  }
  log.error("%ld bytes\n", bytes);

  log.error("%ld mapped regions:\n", allMappedRegions.size());
  bytes = 0;
  for (auto& v : allMappedRegions) {
    log.error("  [%#lx, %#lx)\n", v.begin, v.end);
    bytes += v.end - v.begin;
  }
  log.error("%ld bytes\n", bytes);

  log.error("current live allocations: %d\n", currentLiveAllocations);
}

namespace cpu_allocator {
std::pair<uintptr_t, size_t> regionAtIterate(uintptr_t address) {
  for (auto& v : allMappedRegions) {
    if (address >= v.begin && address < v.end) {
      return {v.begin, v.end - v.begin};
    }
  }
  return {0, 0};
}
std::pair<uintptr_t, size_t> regionAt(uintptr_t address) {
  if (allMappedRegions.size() <= 8) {
    return regionAtIterate(address);
  }
  auto i = std::lower_bound(allMappedRegions.begin(), allMappedRegions.end(), address, [](const Span& a, uintptr_t b) {
    return a.begin > b;
  });
  std::pair<uintptr_t, size_t> r = {0, 0};
  if (i != allMappedRegions.end()) {
    if (i->end > address) {
      r = {i->begin, i->end - i->begin};
    }
  }
  // CHECK(r == regionAtIterate(address));
  return r;
}
bool owns(uintptr_t address) {
  return regionAt(address).second != 0;
}

bool owns(const void* ptr) {
  return owns((uintptr_t)ptr);
}
std::pair<uintptr_t, size_t> regionAt(const void* ptr) {
  return regionAt((uintptr_t)ptr);
}

void* moo_alloc(size_t bytes) {
  void* r = moo_alloc2(bytes);
  if (!owns(r)) {
    log.error("%p not owned !?\n", r);

    cpuAllocatorDebug();

    CHECK(false);
  }
  return r;
}
void moo_free(void* ptr) {
  return moo_free2(ptr);
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

} // namespace moodist
