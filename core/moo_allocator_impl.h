// Moo allocator implementation
// Region-based allocator with size classes

// Simple max/min to avoid libstdc++ dependency
template<typename T>
constexpr T max(T a, T b) {
  return a > b ? a : b;
}
template<typename T>
constexpr T min(T a, T b) {
  return a < b ? a : b;
}

static inline void* bootstrap_alloc(size_t size) {
  void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return nullptr;
  }
  return p;
}

static inline void bootstrap_free(void* ptr, size_t size) {
  if (ptr) {
    munmap(ptr, size);
  }
}

template<typename T>
struct Vector {
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;

  Vector() = default;
  ~Vector() {
    clear();
    if (data_) {
      bootstrap_free(data_, capacity_ * sizeof(T));
    }
  }

  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;
  Vector(Vector&& o) noexcept : data_(o.data_), size_(o.size_), capacity_(o.capacity_) {
    o.data_ = nullptr;
    o.size_ = 0;
    o.capacity_ = 0;
  }

  size_t size() const {
    return size_;
  }
  bool empty() const {
    return size_ == 0;
  }
  T* data() {
    return data_;
  }
  T* begin() {
    return data_;
  }
  T* end() {
    return data_ + size_;
  }

  void clear() {
    for (size_t i = 0; i < size_; ++i) {
      data_[i].~T();
    }
    size_ = 0;
  }

  void reserve(size_t n) {
    if (n <= capacity_) {
      return;
    }
    T* newData = static_cast<T*>(bootstrap_alloc(sizeof(T) * n));
    if (!newData) {
      fprintf(stderr, "moo: bootstrap_alloc failed for %zu bytes\n", sizeof(T) * n);
      abort();
    }
    if (data_) {
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(newData, data_, sizeof(T) * size_);
      } else {
        for (size_t i = 0; i < size_; ++i) {
          new (&newData[i]) T(std::move(data_[i]));
          data_[i].~T();
        }
      }
      bootstrap_free(data_, capacity_ * sizeof(T));
    }
    data_ = newData;
    capacity_ = n;
  }

  void push_back(const T& v) {
    if (size_ == capacity_) {
      size_t newCap = capacity_ * 2;
      if (newCap < 16) {
        newCap = 16;
      }
      reserve(newCap);
    }
    new (&data_[size_]) T(v);
    ++size_;
  }

  T* insert(T* pos, const T& v) {
    size_t idx = pos - data_;
    if (size_ == capacity_) {
      reserve(max(capacity_ * 2, size_t(16)));
    }
    for (size_t i = size_; i > idx; --i) {
      if (i == size_) {
        new (&data_[i]) T(std::move(data_[i - 1]));
      } else {
        data_[i] = std::move(data_[i - 1]);
      }
    }
    if (idx < size_) {
      data_[idx] = v;
    } else {
      new (&data_[idx]) T(v);
    }
    ++size_;
    return data_ + idx;
  }

  T* erase(T* pos) {
    size_t idx = pos - data_;
    data_[idx].~T();
    for (size_t i = idx; i + 1 < size_; ++i) {
      new (&data_[i]) T(std::move(data_[i + 1]));
      data_[i + 1].~T();
    }
    --size_;
    return data_ + idx;
  }
};

// Simple RAII wrapper for pthread_mutex_t
struct PthreadLock {
  pthread_mutex_t* m;
  PthreadLock(pthread_mutex_t& mutex) : m(&mutex) {
    pthread_mutex_lock(m);
  }
  ~PthreadLock() {
    pthread_mutex_unlock(m);
  }
  PthreadLock(const PthreadLock&) = delete;
  PthreadLock& operator=(const PthreadLock&) = delete;
};

static int allocatorNode = -1;

struct Link {
  uintptr_t prev;
  uintptr_t next;
};

template<typename T, Link T::* link>
struct IntrusiveList2 {
private:
  Link head;
  static size_t offset() {
    return reinterpret_cast<uintptr_t>(&(static_cast<T*>(nullptr)->*link));
  }

  static uintptr_t& next(uintptr_t at) noexcept {
    return ((Link*)(at + offset()))->next;
  }
  static uintptr_t& prev(uintptr_t at) noexcept {
    return ((Link*)(at + offset()))->prev;
  }

  uintptr_t headvalue() const noexcept {
    return (uintptr_t)&head - offset();
  }

public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  struct Iterator {
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;

    uintptr_t ptr;

    explicit Iterator(uintptr_t p) : ptr(p) {}

    T& operator*() const noexcept {
      return *(T*)ptr;
    }
    T* operator->() const noexcept {
      return (T*)ptr;
    }

    Iterator& operator++() noexcept {
      ptr = next(ptr);
      return *this;
    }

    Iterator operator++(int) noexcept {
      Iterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool operator==(const Iterator& other) const noexcept {
      return ptr == other.ptr;
    }
    bool operator!=(const Iterator& other) const noexcept {
      return ptr != other.ptr;
    }
  };

  Iterator begin() noexcept {
    return Iterator(next(headvalue()));
  }
  Iterator end() noexcept {
    return Iterator(headvalue());
  }

  bool empty() const noexcept {
    return head.next == headvalue();
  }

  void clear() noexcept {
    head.next = headvalue();
    head.prev = headvalue();
  }
  static uintptr_t insert(uintptr_t at, T& item) noexcept {
    uintptr_t nextItem = at;
    uintptr_t prevItem = prev(at);
    prev(nextItem) = (uintptr_t)&item;
    next(prevItem) = (uintptr_t)&item;
    next((uintptr_t)&item) = nextItem;
    prev((uintptr_t)&item) = prevItem;
    return at;
  }
  static uintptr_t erase(uintptr_t at) noexcept {
    uintptr_t nextItem = next(at);
    uintptr_t prevItem = prev(at);
    prev(nextItem) = prevItem;
    next(prevItem) = nextItem;
    prev(at) = 0;
    next(at) = 0;
    return nextItem;
  }
  static void erase(T& item) noexcept {
    erase((uintptr_t)&item);
  }
  uintptr_t push_front(T& item) noexcept {
    return insert(head.next, item);
  }
  uintptr_t push_back(T& item) noexcept {
    return insert(headvalue(), item);
  }
  void pop_front() noexcept {
    erase(head.next);
  }
  void pop_back() noexcept {
    erase(prev(headvalue()));
  }
  T& front() noexcept {
    return *(T*)head.next;
  }
  T& back() noexcept {
    return *(T*)prev(headvalue());
  }
};

static constexpr size_t alignment = 1024 * 1024 * 2;
static constexpr size_t standardBinSize = 1024 * 16; // 16KB

struct Thread;
struct Region;

struct alignas(64) Bin {
  Link link;
  void* freelist;
  uint32_t capacity;
  uint32_t remainingFrees; // Decremented on free; when 0, bin is fully freed
  uintptr_t freearea;
  uintptr_t end;
  bool full;
  bool active;
  bool idle;
  bool orphaned;
  uint16_t orphan_outstanding;
  uint8_t index;
};

struct FastThread;

struct alignas(64) Region {
  // Hot fields first (used in moo_free fast path)
  uintptr_t databegin;
  uint8_t binshift;
  uint8_t index;
  bool full;
  bool empty;       // True when all allocations have been freed
  FastThread* fast; // For fast cross-thread detection
  // Cold fields
  Thread* thread;
  Link link;      // for activeRegions list
  Link allLink;   // for allRegions list (never removed until reclaimed)
  Link emptyLink; // for emptyRegions list
  IntrusiveList2<Bin, &Bin::link> activebins;
  IntrusiveList2<Bin, &Bin::link> idlebins;
  uint16_t numIdleBins;
  uint16_t minIdleBins;
  uint16_t numCreatedBins; // Set at thread exit for orphan tracking
  uint16_t numEmptyBins;   // Count of bins with all allocations freed
  uintptr_t freebinarea;
  uintptr_t freearea;
  uintptr_t end;
};

struct alignas(64) Thread {
  IntrusiveList2<Region, &Region::link> activeRegions[64];
  IntrusiveList2<Region, &Region::allLink> allRegions;
  IntrusiveList2<Region, &Region::emptyLink> emptyRegions[64]; // Per-index empty regions
  uint16_t numEmptyRegions[64];                                // Per-index count
  std::atomic<void*> remoteFreeList;                           // for cross-thread frees
  std::atomic<bool> exited;                                    // true when thread has exited
  pthread_mutex_t orphanMutex;                                 // protects orphan state transitions
};

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

struct Globals {
  Vector<Span> spans;
  Vector<Span> allMappedRegions;
  Vector<Thread*> freeThreads;   // Reusable Thread objects from exited threads
  pthread_mutex_t slowPathMutex; // Protects spans, allMappedRegions, nextAddr, freeThreads
};

template<size_t Size, size_t Alignment>
struct AlignedStorage {
  alignas(Alignment) unsigned char storage[Size];
};

static AlignedStorage<sizeof(Globals), alignof(Globals)> globalsStorage;
#define globals (reinterpret_cast<Globals&>(globalsStorage))
#define spans (globals.spans)
#define allMappedRegions (globals.allMappedRegions)

static thread_local Thread* currentThread = nullptr;

struct FastThread {
  // Hot fields split into separate arrays for better cache locality
  // When count is 0 (common), checking multiple size classes hits same cache line

  // Stack counts - 64 bytes total, fits in 1 cache line
  // Checked on every alloc (pop) and free (push)
  alignas(64) uint8_t stackCounts[64];

  // Bin pointers - 512 bytes, 8 cache lines
  // Current bin for each size class; checked on every free, used on alloc for remainingFrees
  alignas(64) Bin* bins[64];

  // Stack entries - only accessed when count > 0
  static constexpr size_t STACK_CAPACITY = 8;
  void* stackEntries[64][STACK_CAPACITY];

  // Slot data - bump/end/slotSize for bump allocation, freelist for overflow
  struct Slot {
    uintptr_t bump;     // current bump pointer (0 = uninitialized)
    uintptr_t end;      // bin->end + 1 (for < comparison)
    void* freelist;     // intrusive freelist (overflow or when switching bins)
    uintptr_t slotSize; // size of each slot
  };
  Slot slots[64];
};

static thread_local FastThread fastThread;

void createThread();

[[gnu::always_inline]]
static inline Thread* getCurrentThread() {
  return currentThread;
}

[[gnu::always_inline]]
static inline FastThread* fastthread() {
  return &fastThread;
}

static void drainRemoteFrees(Thread* thread);
static void add_span(Vector<Span>& rspans, uintptr_t begin, uintptr_t end);

// Count entries in a freelist
static size_t countFreelist(void* head) {
  size_t count = 0;
  while (head) {
    void* next;
    std::memcpy(&next, head, sizeof(void*));
    head = next;
    ++count;
  }
  return count;
}

// Compute outstanding allocations for a bin
static size_t computeBinOutstanding(Bin* bin, uint8_t regionIndex) {
  size_t slot_size = 1ul << (64 - regionIndex);
  uintptr_t data_start = bin->end - ((size_t)bin->capacity * slot_size);
  size_t allocated = (bin->freearea - data_start) / slot_size;
  size_t freed = countFreelist(bin->freelist);
  return allocated - freed;
}

// Push pointer to bin's freelist and handle full->idle transition
static Region* pushToFreelist(void* p) {
  Region* r = (Region*)((uintptr_t)p & ~(alignment - 1));
  size_t binIndex = ((uintptr_t)p - r->databegin) >> r->binshift;
  Bin* bin = (Bin*)(r + 1) + binIndex;

  std::memcpy(p, &bin->freelist, sizeof(void*));
  bin->freelist = p;

  if (bin->full) {
    bin->full = false;
    bin->idle = true;
    r->idlebins.push_back(*bin);
    ++r->numIdleBins;
  }

  return r;
}

// Process a single orphan free - called with orphanMutex held
// Returns true if this free caused the region to become fully empty
static bool processOrphanFree(void* p) {
  Region* r = (Region*)((uintptr_t)p & ~(alignment - 1));
  size_t binIndex = ((uintptr_t)p - r->databegin) >> r->binshift;
  Bin* bin = (Bin*)(r + 1) + binIndex;

  std::memcpy(p, &bin->freelist, sizeof(void*));
  bin->freelist = p;

  if (bin->full) {
    bin->full = false;
    bin->idle = true;
    r->idlebins.push_back(*bin);
    ++r->numIdleBins;
  }

  if (bin->orphaned) {
    if (--bin->orphan_outstanding == 0) {
      bin->orphaned = false;
      if (++r->numEmptyBins == r->numCreatedBins) {
        r->thread = nullptr; // Mark for reclaim
        r->fast = nullptr;
        return true;
      }
    }
  }
  return false;
}

// Reclaim regions and optionally add thread to freeThreads
// Called with orphanMutex held, will take slowPathMutex for reclaim
// Removes reclaimed regions from allRegions while holding orphanMutex
static void reclaimEmptyRegions(Thread* thread) {
  Vector<Region*> toReclaim;

  // Collect regions marked for reclaim (thread == nullptr)
  // and remove from allRegions while we hold orphanMutex
  for (auto it = thread->allRegions.begin(); it != thread->allRegions.end();) {
    Region& r = *it;
    ++it; // Advance before potential erase
    if (r.thread == nullptr) {
      IntrusiveList2<Region, &Region::allLink>::erase(r);
      toReclaim.push_back(&r);
    }
  }

  if (toReclaim.empty()) {
    return;
  }

  // Add spans under slowPathMutex
  // Also check if thread should be added to freeThreads
  bool threadEmpty = thread->allRegions.empty();

  {
    PthreadLock lock(globals.slowPathMutex);
    for (size_t i = 0; i < toReclaim.size(); ++i) {
      Region* r = toReclaim.data()[i];
      add_span(spans, (uintptr_t)r, r->end);
    }

    if (threadEmpty) {
      globals.freeThreads.push_back(thread);
    }
  }
}

// ============================================================================
// Live region reclamation helpers
// ============================================================================

// Get the threshold of empty regions before we start reclaiming
// Larger regions have lower thresholds (we want to reclaim them sooner)
[[gnu::always_inline]]
static inline uint16_t getEmptyRegionThreshold(size_t regionSize) {
  // 2MB regions: threshold 4
  // 4MB regions: threshold 2
  // 8MB+ regions: threshold 1
  if (regionSize <= 2 * 1024 * 1024) {
    return 4;
  }
  if (regionSize <= 4 * 1024 * 1024) {
    return 2;
  }
  return 1;
}

// Mark a region as empty and add to emptyRegions list
// Called when numEmptyBins == numCreatedBins (all bins fully freed)
// Region stays in activeRegions - it can still be allocated from
[[gnu::noinline]]
static void markRegionEmpty(Thread* thread, Region* r) {
  if (r->empty) {
    return; // Already marked
  }
  r->empty = true;

  thread->emptyRegions[r->index].push_back(*r);
  ++thread->numEmptyRegions[r->index];
}

// Mark a region as non-empty and remove from emptyRegions list
// Called when a bin goes from 0 to 1 allocations
[[gnu::noinline]]
static void markRegionNonEmpty(Thread* thread, Region* r) {
  if (!r->empty) {
    return; // Not marked empty
  }
  r->empty = false;

  // Remove from emptyRegions list for this index
  // Invariant: r->empty == true means r is in emptyRegions list
  thread->emptyRegions[r->index].erase(*r);
  --thread->numEmptyRegions[r->index];
}

// Reclaim a single region - return its memory to spans
// Caller must have removed region from all lists except allRegions
static void reclaimSingleRegion(Region* r) {
  PthreadLock lock(globals.slowPathMutex);
  add_span(spans, (uintptr_t)r, r->end);
}

// Called when a bin transitions from fully-freed (remainingFrees==0) to non-empty
// This happens on the first alloc from a fully-freed bin
[[gnu::always_inline]]
static inline void handleBinBecameNonEmpty(Bin* bin) {
  Region* r = (Region*)((uintptr_t)bin & ~(alignment - 1));
  --r->numEmptyBins;
  if (r->empty) {
    Thread* thread = getCurrentThread();
    markRegionNonEmpty(thread, r);
  }
}

// Same as above but returns the pointer - allows tail call optimization
[[gnu::noinline]]
static void* handleBinBecameNonEmpty_ret(Bin* bin, void* p) {
  handleBinBecameNonEmpty(bin);
  return p;
}

static void maybeReclaimEmptyRegions(Thread* thread, size_t index);

// Called when a bin becomes fully freed - check if region should be marked empty
// Allows tail call optimization on free path
[[gnu::noinline]]
static void handleBinBecameEmpty(Region* r) {
  if (++r->numEmptyBins == r->numCreatedBins) {
    Thread* thread = getCurrentThread();
    markRegionEmpty(thread, r);
    maybeReclaimEmptyRegions(thread, r->index);
  }
}

// Check if we should reclaim empty regions for a specific index
// Called periodically from free path when a region becomes empty
[[gnu::noinline]]
static void maybeReclaimEmptyRegions(Thread* thread, size_t index) {
  auto& count = thread->numEmptyRegions[index];
  if (count == 0) {
    return;
  }
  auto& list = thread->emptyRegions[index];
  Region& oldest = list.front();
  size_t regionSize = oldest.end - (uintptr_t)&oldest;
  uint16_t threshold = getEmptyRegionThreshold(regionSize);
  if (count < threshold) {
    return;
  }

  // Reclaim regions until we're under threshold
  while (count >= threshold) {
    Region& r = list.front();

    // Remove from emptyRegions
    list.pop_front();
    --count;
    r.empty = false; // Maintain invariant: empty == in list

    // Remove from activeRegions
    if (!r.full) {
      thread->activeRegions[index].erase(r);
    }

    // Clear FastThread state if it points to a bin in this region
    // This prevents stale pointers from being reused after reclamation
    FastThread* fast = &fastThread;
    Bin* oldBin = fast->bins[index];
    if (oldBin) {
      Region* fastBinRegion = (Region*)((uintptr_t)oldBin & ~(alignment - 1));
      if (fastBinRegion == &r) {
        fast->stackCounts[index] = 0;
        auto& slot = fast->slots[index];
        slot.freelist = nullptr;
        slot.bump = 0;
        slot.end = 0;
        fast->bins[index] = nullptr;
      }
    }

    IntrusiveList2<Region, &Region::allLink>::erase(r);
    reclaimSingleRegion(&r);
  }
}

// Drain remote frees for an orphaned thread
static void drainOrphanFrees(Thread* owner) {
  PthreadLock lock(owner->orphanMutex);

  // Re-check: Thread might have been reused while we waited
  if (!owner->exited.load(std::memory_order_relaxed)) {
    return;
  }

  void* head = owner->remoteFreeList.exchange(nullptr, std::memory_order_acquire);
  while (head != nullptr) {
    void* next;
    std::memcpy(&next, head, sizeof(void*));
    processOrphanFree(head);
    head = next;
  }

  // Reclaim any newly-empty regions
  reclaimEmptyRegions(owner);
}

// Cleanup thread resources on thread exit
static void destroyThread(void* p) {
  Thread* thread = (Thread*)p;

  // CRITICAL: Drain stack and slot.freelist back to bin->freelist BEFORE computing orphan state
  // Otherwise entries won't be counted and bins will have wrong outstanding count
  FastThread* fast = &fastThread;
  for (size_t i = 0; i < 64; ++i) {
    auto& slot = fast->slots[i];
    Bin* bin = fast->bins[i];

    // Drain stack first - all entries belong to the current bin
    if (bin) {
      while (fast->stackCounts[i] > 0) {
        void* p = fast->stackEntries[i][--fast->stackCounts[i]];
        std::memcpy(p, &bin->freelist, sizeof(void*));
        bin->freelist = p;
      }
      // Sync freearea from slot.bump
      bin->freearea = slot.bump;
    }

    void* fl = slot.freelist;
    while (fl) {
      void* next;
      std::memcpy(&next, fl, sizeof(void*));
      uintptr_t a = (uintptr_t)fl & ~(alignment - 1);
      Region* r = (Region*)a;
      size_t binIndex = ((uintptr_t)fl - r->databegin) >> r->binshift;
      Bin* flBin = (Bin*)(r + 1) + binIndex;
      std::memcpy(fl, &flBin->freelist, sizeof(void*));
      flBin->freelist = fl;
      fl = next;
    }
    slot.freelist = nullptr;
    fast->bins[i] = nullptr;
  }

  drainRemoteFrees(thread);

  {
    PthreadLock lock(thread->orphanMutex);

    // Mark as exited - all subsequent frees will trigger drainOrphanFrees
    thread->exited.store(true, std::memory_order_release);

    // Drain anything that arrived after first drain but before exited=true
    void* head = thread->remoteFreeList.exchange(nullptr, std::memory_order_acquire);
    while (head != nullptr) {
      void* next;
      std::memcpy(&next, head, sizeof(void*));
      pushToFreelist(head);
      head = next;
    }

    // Compute orphan state for all bins in all regions
    for (Region& r : thread->allRegions) {
      r.numCreatedBins = (r.freebinarea - (uintptr_t)(&r + 1)) / sizeof(Bin);
      r.numEmptyBins = 0;

      Bin* binsStart = (Bin*)(&r + 1);
      for (size_t i = 0; i < r.numCreatedBins; ++i) {
        Bin* bin = binsStart + i;
        size_t outstanding = computeBinOutstanding(bin, r.index);

        if (outstanding == 0) {
          r.numEmptyBins++;
          bin->orphaned = false;
        } else {
          bin->orphaned = true;
          bin->orphan_outstanding = outstanding;
        }
      }

      if (r.numEmptyBins == r.numCreatedBins && r.numCreatedBins > 0) {
        r.thread = nullptr;
      }
    }

    reclaimEmptyRegions(thread);
  }
}

bool pthreadKeyInited;
pthread_key_t pthreadKey;

// Create a new Thread struct for the current thread
[[gnu::noinline]] [[gnu::cold]]
void createThread() {
  Thread* thread = nullptr;

  // First, try to reuse a Thread from the freelist
  {
    PthreadLock lock(globals.slowPathMutex);
    if (!pthreadKeyInited) {
      pthreadKeyInited = true;
      pthread_key_create(&pthreadKey, &destroyThread);
    }
    if (!globals.freeThreads.empty()) {
      thread = globals.freeThreads.data()[globals.freeThreads.size() - 1];
      globals.freeThreads.erase(globals.freeThreads.end() - 1);
    }
  }

  if (thread) {
    // Reuse Thread from freeThreads
    // Thread is only added to freeThreads when all its regions are reclaimed,
    // so allRegions is guaranteed to be empty here - just reset the exited flag
    thread->exited.store(false, std::memory_order_relaxed);
    pthread_setspecific(pthreadKey, thread);
    currentThread = thread;
  } else {
    void* mem = mmap(nullptr, sizeof(Thread), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (!mem || mem == MAP_FAILED) {
      abort();
    }
    thread = new (mem) Thread();
  }
  for (auto& v : thread->activeRegions) {
    v.clear();
  }
  thread->allRegions.clear();
  for (auto& v : thread->emptyRegions) {
    v.clear();
  }
  memset(thread->numEmptyRegions, 0, sizeof(thread->numEmptyRegions));
  thread->remoteFreeList = nullptr;

  pthread_setspecific(pthreadKey, thread);
  currentThread = thread;
}

static constexpr size_t regionSize =
    (sizeof(Region) + alignof(max_align_t) - 1) / alignof(max_align_t) * alignof(max_align_t);

static void add_span(Vector<Span>& rspans, uintptr_t begin, uintptr_t end) {
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

static uintptr_t nextAddr;

static void initRegion(Region* r, size_t index, size_t nbytes) {
  r->thread = currentThread;
  r->fast = &fastThread;
  r->full = false;
  r->empty = false;
  r->link = {};
  r->allLink = {};
  r->emptyLink = {};
  r->freearea = (uintptr_t)r + sizeof(Region);
  r->activebins.clear();
  r->idlebins.clear();
  r->numIdleBins = 0;
  r->numCreatedBins = 0;
  r->numEmptyBins = 0;
  r->index = index;
  currentThread->activeRegions[index].push_front(*r);
  currentThread->allRegions.push_back(*r);

  size_t binsize = max(standardBinSize, nbytes);
  r->binshift = __builtin_ctzll(binsize);
  size_t numBins = (r->end - r->freearea) >> r->binshift;

  r->minIdleBins = max(numBins / 8, (size_t)1);

  r->freebinarea = r->freearea;
  r->freearea += sizeof(Bin) * numBins;

  size_t align = min((size_t)4096, nbytes);
  r->freearea = (r->freearea + align - 1) & ~(align - 1);
  r->databegin = r->freearea;

  CHECK(r->end - r->freearea >= binsize);

  size_t n = 0;
  while (r->freearea + binsize <= r->end) {
    void* rv = (void*)r->freearea;
    CHECK((uintptr_t)rv / alignment * alignment == (uintptr_t)r);
    CHECK((uintptr_t)r->freebinarea % alignof(Bin) == 0);
    Bin* bin = (Bin*)r->freebinarea;
    r->freebinarea += sizeof(Bin);
    CHECK(r->freebinarea <= r->end);
    bin->freelist = nullptr;
    bin->remainingFrees = 0;
    bin->index = index;
    bin->full = false;
    bin->freearea = (uintptr_t)rv;
    bin->end = bin->freearea + binsize;
    bin->capacity = binsize >> __builtin_ctzll(nbytes);
    CHECK(nbytes * bin->capacity == binsize);
    bin->active = true;
    bin->idle = false;
    bin->orphaned = false;
    bin->orphan_outstanding = 0;
    ++r->numCreatedBins;
    ++r->numEmptyBins;
    r->activebins.push_back(*bin);

    r->freearea += binsize;
    CHECK(r->freearea <= r->end);
    ++n;
    if (n >= 1) {
      __builtin_prefetch(rv);
      break;
    }
  }
  CHECK(!r->activebins.empty());
}

static uint64_t rngSeed() {
  uint64_t r = 0x42;
  r += (__rdtsc() * 954311185259313919ul) % 0xffffffffffffl;
  timespec ts{0, 0};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  r += ts.tv_sec * 1000000000l + ts.tv_nsec;
  r += __rdtsc();
  return r;
}

static uint64_t rngState;
static uint64_t rng() {
  if (rngState == 0) {
    rngState = rngSeed();
  }
  rngState = (rngState * 2862933555777941757l) + 1;
  return ((rngState >> 16) & 0xfff) | (rngState << 16);
}

static size_t mmappedBytes;

static void move_pages() {
  PthreadLock lock(globals.slowPathMutex);
  for (auto& v : allMappedRegions) {
    log.debug("moving %#lx of %lu bytes to numa node %d\n", v.begin, v.end - v.begin, allocatorNode);
    numa_move((void*)v.begin, v.end - v.begin, allocatorNode);
  }
}

static void* moo_alloc(size_t);

static void allocate_memory(size_t index) {
  CHECK(index != 0);

  size_t nbytes = max(1ul << (64 - index), (size_t)8);

  size_t minbytes = regionSize + nbytes;
  size_t alignedminbytes = max(alignment, (minbytes + alignment - 1) / alignment * alignment);

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

  size_t bytes = max(alignment, (size_t)(alignment + regionSize + nbytes * (nbytes < alignment ? 8 : 2)));
  bytes = max(bytes, (mmappedBytes / 4 + alignment - 1) / alignment * alignment);
  bytes = (bytes + alignment - 1) / alignment * alignment;
  void* rv = nullptr;

  uintptr_t addr = nextAddr;

  for (size_t i = 0; i != 0x1000; ++i) {
    if (addr) {
      rv = mmap((void*)addr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
      if (rv && rv != MAP_FAILED) {
        nextAddr = addr + bytes;
        if (rv != (void*)addr) {
          nextAddr = (0x100000000000ul + (rng() % 0x600000000000ul)) / alignment * alignment;
        }
        break;
      }
    }
    addr = (0x100000000000ul + (rng() % 0x600000000000ul)) / alignment * alignment;
  }

  if (!rv || rv == MAP_FAILED) {
    log.error("fallback allocation of %zu bytes\n", bytes);
    rv = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (!rv || rv == MAP_FAILED) {
      fprintf(stderr, "moo: mmap failed: %s\n", strerror(errno));
      abort();
    }
  }

  madvise(rv, bytes, MADV_HUGEPAGE);

  if (allocatorNode != -1) {
    numa_move(rv, bytes, allocatorNode);
  }

  mmappedBytes += bytes;
  uintptr_t e = ((uintptr_t)rv + bytes) / alignment * alignment;
  rv = (void*)(((uintptr_t)rv + alignment - 1) / alignment * alignment);
  bytes = e - (uintptr_t)rv;

  log.verbose("(internal) mapped %zu bytes at %#lx (total %zu bytes, %zuG) on "
              "node %d\n",
      bytes, (uintptr_t)rv, mmappedBytes, mmappedBytes / 1024 / 1024 / 1024, allocatorNode);

  CHECK(bytes > 0);

  add_span(spans, (uintptr_t)rv, (uintptr_t)rv + bytes);
  add_span(allMappedRegions, (uintptr_t)rv, (uintptr_t)rv + bytes);

  allocate_memory(index);
}

static void* moo_alloc(size_t bytes);
static void drainRemoteFrees(Thread* thread);

[[gnu::noinline]] static void* moo_alloc_slow(size_t bytes, size_t index) {
  // Use the exact size for this size class to avoid changing size classes
  bytes = 1ull << (64 - index);
  CHECK(currentThread->activeRegions[index].empty());

  Thread* thread = getCurrentThread();
  if (thread->remoteFreeList.load(std::memory_order_relaxed) != nullptr) [[unlikely]] {
    drainRemoteFrees(thread);
    return moo_alloc(bytes);
  }

  // Lock for slow path - protects spans, allMappedRegions, mmap, etc.
  PthreadLock lock(globals.slowPathMutex);

  allocate_memory(index);
  return moo_alloc(bytes);
}

static size_t currentLiveAllocations = 0;

[[gnu::noinline]] [[gnu::cold]]
static void* moo_alloc_new_thread(size_t bytes) {
  createThread();
  return moo_alloc(bytes);
}

size_t mooFastPathHits = 0;
size_t mooSlowPathHits = 0;

size_t mooCounts[64];

// Medium/slow path: get a new bin with bump space or freelist
[[gnu::noinline]] static void* moo_alloc_medium(Thread* thread, FastThread* fast, size_t bytes, size_t index) {
  size_t nbytes = 1ul << (64 - index);

  if (!thread->activeRegions[index].empty()) [[likely]] {
    Region* r = &thread->activeRegions[index].front();

    CHECK(r);

    while (true) {
      if (!r->activebins.empty()) [[likely]] {
        Bin* bin = &r->activebins.front();
        CHECK(bin->active);

        // Check if fully-freed bin - reset for bump allocation
        // Fully-freed bins have remainingFrees==0 and non-empty freelist
        // (New bins have remainingFrees==0 but empty freelist)
        if (bin->remainingFrees == 0 && bin->freelist != nullptr) [[unlikely]] {
          // Reset bin for bump allocation
          // Don't call handleBinBecameNonEmpty here - the bump alloc will do it
          // when remainingFrees goes 0→1
          bin->freearea = bin->end - nbytes * bin->capacity;
          bin->freelist = nullptr;
          // remainingFrees stays 0 so the bump alloc knows to call handleBinBecameNonEmpty
        }

        if (bin->freearea + nbytes <= bin->end) {
          void* rv = (void*)bin->freearea;
          bin->freearea += nbytes;
          CHECK(bin->freearea <= bin->end);
          fast->bins[index] = bin;
          if (bin->remainingFrees++ == 0) [[unlikely]] {
            return handleBinBecameNonEmpty_ret(bin, rv);
          }
          return rv;
        }
        if (bin->freelist != nullptr) {
          void* rv = bin->freelist;
          void* next;
          std::memcpy(&next, rv, sizeof(void*));
          bin->freelist = next;
          fast->bins[index] = bin;
          if (bin->remainingFrees++ == 0) [[unlikely]] {
            return handleBinBecameNonEmpty_ret(bin, rv);
          }
          return rv;
        }
        // Bin exhausted (no bump, no freelist) - mark and remove from activebins
        bin->full = true;
        bin->active = false;
        bin->idle = true;
        r->activebins.erase(*bin);
        r->idlebins.push_back(*bin);
        ++r->numIdleBins;
        if (fast->bins[index] == bin) {
          fast->bins[index] = nullptr;
        }
        continue;
      } else {
        // No activebins - create a new bin
        size_t binsize = 1ull << r->binshift;
        if (r->freearea + binsize <= r->end) {
          void* rv = (void*)r->freearea;
          CHECK((uintptr_t)rv / alignment * alignment == (uintptr_t)r);
          CHECK((uintptr_t)r->freebinarea % alignof(Bin) == 0);
          Bin* bin = (Bin*)r->freebinarea;
          r->freebinarea += sizeof(Bin);
          CHECK(r->freebinarea <= r->end);
          bin->freelist = nullptr;
          bin->remainingFrees = 0;
          bin->index = index;
          bin->full = false;
          bin->freearea = (uintptr_t)rv;
          bin->end = bin->freearea + binsize;
          bin->capacity = binsize >> __builtin_ctzll(nbytes);
          bin->active = true;
          bin->idle = false;
          bin->orphaned = false;
          bin->orphan_outstanding = 0;
          ++r->numCreatedBins;
          ++r->numEmptyBins; // New bin counts as empty until first alloc
          r->activebins.push_back(*bin);
          fast->bins[index] = bin;

          r->freearea += binsize;
          CHECK(r->freearea <= r->end);
          continue;
        } else {
          r->full = true;
          thread->activeRegions[index].erase(*r);
          if (thread->activeRegions[index].empty()) {
            break;
          } else {
            r = &thread->activeRegions[index].front();
          }
        }
      }
    }
  }

  return moo_alloc_slow(bytes, index);
}

[[gnu::always_inline]] static inline size_t getSizeClass(size_t bytes) {
  bytes += !bytes;
  // Use | 7 for 8-byte minimum (needed for freelist pointer in hybrid mode)
  return __builtin_ia32_lzcnt_u64((bytes - 1) | 7);
}

[[gnu::noinline]] static inline void* moo_alloc2(size_t index) {
  Thread* thread = getCurrentThread();
  if (!thread) [[unlikely]] {
    size_t bytes = 1ull << (64 - index);
    return moo_alloc_new_thread(bytes);
  }
  CHECK(index && index < 64);

  FastThread* fast = fastthread();
  auto& slot = fast->slots[index];

  Bin* oldBin = fast->bins[index];

  // Check if current bin has freelist entries (from frees)
  // This is the hybrid fallback - reuse freed slots before getting new bin
  if (oldBin && oldBin->freelist != nullptr) {
    void* rv = oldBin->freelist;
    void* next;
    std::memcpy(&next, rv, sizeof(void*));
    oldBin->freelist = next;
    if (oldBin->remainingFrees++ == 0) [[unlikely]] {
      return handleBinBecameNonEmpty_ret(oldBin, rv);
    }
    // Note: don't touch slot.bump - it's already exhausted (that's why we're here)
    // bin->freearea is stale (not synced with fast path bumps) so we can't use it
    return rv;
  }

  // Current bin is exhausted (no bump space, no freelist) - move to idlebins
  if (oldBin) {
    // CRITICAL: Drain stack and slot.freelist back to bin before moving to idle
    while (fast->stackCounts[index] > 0) {
      void* p = fast->stackEntries[index][--fast->stackCounts[index]];
      std::memcpy(p, &oldBin->freelist, sizeof(void*));
      oldBin->freelist = p;
    }

    void* fl = slot.freelist;
    while (fl) {
      void* next;
      std::memcpy(&next, fl, sizeof(void*));
      std::memcpy(fl, &oldBin->freelist, sizeof(void*));
      oldBin->freelist = fl;
      fl = next;
    }
    slot.freelist = nullptr;
    fast->bins[index] = nullptr;

    // CRITICAL: Sync bin->freearea from slot.bump before moving to idle
    // The fast path bumps slot.bump without updating bin->freearea.
    // If we don't sync, moo_alloc_medium will think there's bump space when
    // this bin is later promoted from idle (causing double-allocation bugs).
    oldBin->freearea = slot.bump;

    oldBin->full = true;
    oldBin->active = false;
    oldBin->idle = true;
    Region* oldRegion = (Region*)((uintptr_t)oldBin & ~(alignment - 1));
    decltype(Region::activebins)::erase(*oldBin);
    oldRegion->idlebins.push_back(*oldBin);
    ++oldRegion->numIdleBins;
  }

  // Note: idle bins with freelists are promoted to activebins by moo_free2

  size_t bytes = 1ull << (64 - index);
  void* rv = moo_alloc_medium(thread, fast, bytes, index);

  // moo_alloc_medium sets fast->bins[index] - use it to set up FastSlot
  Bin* bin = fast->bins[index];
  if (bin) {
    slot.bump = bin->freearea;
    slot.end = bin->end + 1;
    slot.slotSize = bytes;
    // Note: moo_alloc_medium already incremented remainingFrees and called
    // handleBinBecameNonEmpty if needed (for new bins or reset bins)
  }

  return rv;
}

[[gnu::always_inline]] static inline void* moo_alloc(size_t bytes) {
  size_t index = getSizeClass(bytes);
  FastThread* fast = fastthread();
  auto& slot = fast->slots[index];

  // Fast path 1: Pop from non-intrusive stack (no memory touch needed!)
  if (fast->stackCounts[index] > 0) [[likely]] {
    void* p = fast->stackEntries[index][--fast->stackCounts[index]];
    // Update remainingFrees for the current bin
    Bin* bin = fast->bins[index];
    if (bin->remainingFrees++ == 0) [[unlikely]] {
      return handleBinBecameNonEmpty_ret(bin, p);
    }
    return p;
  }

  // Fast path 2: Check slot.freelist (intrusive, for overflow or drained entries)
  void* fl = slot.freelist;
  if (fl) [[unlikely]] {
    void* next;
    std::memcpy(&next, fl, sizeof(void*));
    slot.freelist = next;
    // Update remainingFrees for the bin this pointer came from
    uintptr_t a = (uintptr_t)fl & ~(alignment - 1);
    Region* r = (Region*)a;
    size_t binIndex = ((uintptr_t)fl - r->databegin) >> r->binshift;
    Bin* bin = (Bin*)(r + 1) + binIndex;
    if (bin->remainingFrees++ == 0) [[unlikely]] {
      return handleBinBecameNonEmpty_ret(bin, fl);
    }
    return fl;
  }

  // Fast path 3: Bump allocation
  uintptr_t ptr = slot.bump;
  uintptr_t slotSize = slot.slotSize;
  uintptr_t newBump = ptr + slotSize;

  // For uninitialized slots: bump=0, slotSize=0, end=0 -> newBump=0, 0 < 0 is false, goes to slow path
  if (newBump < slot.end) [[likely]] {
    slot.bump = newBump;
    Bin* bin = fast->bins[index];
    if (bin->remainingFrees++ == 0) [[unlikely]] {
      return handleBinBecameNonEmpty_ret(bin, (void*)ptr);
    }
    return (void*)ptr;
  }

  return moo_alloc2(index);
}

static void moo_free(void* p);

// Drain remote frees from other threads
static void drainRemoteFrees(Thread* thread) {
  void* head = thread->remoteFreeList.exchange(nullptr, std::memory_order_acquire);
  while (head != nullptr) {
    void* next;
    std::memcpy(&next, head, sizeof(void*));
    moo_free(head);
    head = next;
  }
}

// Cross-thread free - atomic push to owner's remoteFreeList
[[gnu::noinline]] [[gnu::cold]]
static void moo_free_remote(Thread* owner, void* p) {
  void* head = owner->remoteFreeList.load(std::memory_order_relaxed);
  do {
    std::memcpy(p, &head, sizeof(void*));
  } while (!owner->remoteFreeList.compare_exchange_weak(head, p, std::memory_order_release, std::memory_order_relaxed));

  // If owner has exited, drain and process as orphan
  if (owner->exited.load(std::memory_order_acquire)) [[unlikely]] {
    drainOrphanFrees(owner);
  }
}

[[gnu::noinline]] [[gnu::cold]]
static void moo_free_new_thread(void* p) {
  createThread();
  moo_free(p);
}

[[gnu::always_inline]] static inline void moo_free2(void* p, Region* r, Bin* bin, FastThread* fast) {
  if (r->fast != fast) [[unlikely]] {
    // Need Thread* for remote free - get it lazily
    Thread* thread = getCurrentThread();
    if (!thread) [[unlikely]] {
      return moo_free_new_thread(p);
    }
    return moo_free_remote(r->thread, p);
  }

  bool wasEmpty = (bin->freelist == nullptr);

  std::memcpy(p, &bin->freelist, sizeof(void*));
  bin->freelist = p;

  // If idle bin got its first freelist entry, promote to activebins
  if (bin->idle && wasEmpty) [[unlikely]] {
    bin->idle = false;
    bin->full = false; // Clear full flag - bin can now service allocations
    r->idlebins.erase(*bin);
    --r->numIdleBins;
    bin->active = true;
    r->activebins.push_back(*bin);

    // If region was full, add back to activeRegions
    if (r->full) {
      r->full = false;
      Thread* thread = getCurrentThread();
      thread->activeRegions[bin->index].push_back(*r);
    }
  }

  if (--bin->remainingFrees == 0) [[unlikely]] {
    return handleBinBecameEmpty(r);
  }
}

[[gnu::always_inline]] static inline void moo_free(void* p) {
  FastThread* fast = fastthread();
  uintptr_t a = (uintptr_t)p & ~(alignment - 1);
  if (a == 0) [[unlikely]] {
    return;
  }
  Region* r = (Region*)a;

  // Compute bin from pointer
  size_t binIndex = ((uintptr_t)p - r->databegin) >> r->binshift;
  Bin* bin = (Bin*)(r + 1) + binIndex;

  size_t index = r->index; // Use region's index (same cache line as databegin/binshift)

  // Fast path: freeing to current bin - use non-intrusive stack
  if (bin == fast->bins[index]) [[unlikely]] {
    if (fast->stackCounts[index] < FastThread::STACK_CAPACITY) [[likely]] {
      // Push to stack - no write to freed memory!
      fast->stackEntries[index][fast->stackCounts[index]++] = p;
      if (--bin->remainingFrees == 0) [[unlikely]] {
        return handleBinBecameEmpty(r);
      }
      return;
    }
    // Stack full - fall through to intrusive freelist via slow path
  }

  // Slow path: handle cross-thread/non-current-bin cases or stack overflow
  return moo_free2(p, r, bin, fast);
}

static size_t moo_alloc_size(void* p) {
  uintptr_t a = (uintptr_t)p & ~(alignment - 1);
  if (!a) {
    return 0;
  }
  Region* r = (Region*)a;
  return max(1ul << (64 - r->index), (size_t)8);
}

// ============================================================================
// Public API
// ============================================================================

void* alloc(size_t bytes) {
  return moo_alloc(bytes);
}

void free(void* ptr) {
  moo_free(ptr);
}

size_t allocSize(void* ptr) {
  return moo_alloc_size(ptr);
}

void setNode(int node) {
  if (allocatorNode == node) {
    return;
  }
  log.debug("allocator node changed from %d to %d\n", allocatorNode, node);
  allocatorNode = node;
  move_pages();
}

size_t getMmappedBytes() {
  return mmappedBytes;
}
size_t getLiveAllocations() {
  return currentLiveAllocations;
}
