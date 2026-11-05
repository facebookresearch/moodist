// Copyright (c) Meta Platforms, Inc. and affiliates.

// moodist's CUDA allocator.
//
// Memory. The allocator reserves allocatorReserveSize bytes of virtual addresses and maps cuMemCreate chunks into them,
// one after another, as it needs more memory (mapMoreMemory). It never unmaps them. Chunk listeners bind the chunks
// into process groups' multicast objects and export them to peers.
//
// Pools. Free memory is kept in pools (Pool): sets of free lists of address-ordered regions that merge with their
// neighbors (Regions), from which an allocation takes the best fit (the smallest region that fits, the lowest address
// among equal sizes; the allocation is its first bytes).
//  - The main pool: allocations of the CUDA allocator (cudaAllocatorImplAllocate), and communication pools' slabs.
//  - Communication pools (CommPool, see allocator_api.h): in logical coordinates of their own.
//  - Private pools (PrivatePool): PyTorch's memory pools, e.g. a CUDA graph's (see cudaAllocatorBeginAllocateToPool).
//
// Streams. An allocation is for use on the stream it is allocated on, and on any streams recorded for it
// (recordStream), as in PyTorch. When it is freed, an event is recorded on each of those streams, and the memory
// carries the events (EventRegion): an allocation on another stream that takes any of that memory makes its stream
// wait for them on the GPU (cuStreamWaitEvent). An allocation on the same stream needs no wait (stream order).
//
// Normal mode. A pool keeps memory freed in a list per stream (the stream it was allocated on), and new memory in its
// common list, so that memory is reused on its own stream, without waits. An allocation takes, in this order:
//  1. the best fit in its stream's list
//  2. the best fit in the common list
//  3. new memory: the main pool maps a chunk; a communication pool takes a slab from the main pool; a private pool
//     takes the allocation from the main pool's free memory (no waits first), else maps a chunk.
// Nothing here depends on the GPU's progress, so where allocations go depends only on the sequence of allocations and
// frees (sizes, streams and order): the same on every rank that does the same (communication pools rely on this).
// Memory in the list of a stream that never allocates again would be lost to the others: when moodist destroys a
// stream of its own (compile_op's tuning stream), its lists go to the common lists (see streamDestroyed). Memory freed
// by other streams that are destroyed later (PyTorch never destroys its streams) stays in their lists until pressure
// mode.
//
// Streams are identified by their ids, not their handles (see StreamRef): the driver hands a destroyed stream's handle
// to the next stream created, even while the destroyed stream's work still runs. Memory must be freed before the
// streams that used it are destroyed.
//
// Pressure mode. Once a pool cannot get new memory (the device is out of it), it merges all its lists into its common
// list, for good, and frees go there too: per-stream lists would keep the free memory split between streams. An
// allocation then takes the best fit among the regions that need no wait (the events of other streams on them have
// completed: the host queries events, each stream's in the order recorded, and drops the completed ones; see
// processEvents), else the best fit (with waits), else new memory, else it fails (CudaOutOfMemoryError).
//
// CUDA graphs. During a capture, PyTorch routes the capture's allocations to the graph's private pool (see
// cudaAllocatorBeginAllocateToPool).
//  - An event recorded during a capture is a node of the capture's graph: work captured in the same capture can wait
//    for it (an edge of the graph), nothing else can, and the host cannot query it. Such events
//    (EventRegion::captureId) are dropped when their capture has ended (cudaAllocatorEndAllocateToPool): later captures
//    into the same pool, and frees after replays, rely on PyTorch's rules (graphs that share a pool are replayed in
//    the order captured, one at a time).
//  - A capture cannot wait for an event recorded outside it, so an allocation during a capture takes only memory
//    without such events (completed events are dropped first: queries of events recorded outside captures are allowed
//    during captures in relaxed capture mode, see RelaxedCaptureMode).
//  - Memory of any other pool freed during a capture (a stream that used it is capturing) is freed when the capture
//    has ended (deferredFrees), when its events can be recorded outside any capture.
//  - A private pool's memory goes back to the main pool once the pool is released and the graphs that may use it are
//    gone: each capture that allocates or frees memory of the pool has its graph retain a CUDA user object, which the
//    driver destroys only after the graph and its executable graphs are destroyed and their last launch has completed
//    (see attachGraph).
//
// Debugging: MOODIST_DEBUG_ALLOCATOR=1 checks the allocator's data structures after every call (slow), =2 logs where
// the free memory is whenever a chunk is mapped, =3 both. MOODIST_DEBUG_ALLOCATOR_TRACE=<path> writes every allocation
// and free to <path>.<pid> (see trace).

#include "api/allocator.h"
#include "api/allocator_api.h"
#include "common.h"
#include "function.h"
#include "group.h"
#include "hash_map.h"
#include "intrusive_list.h"
#include "vector.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <unistd.h>

namespace moodist {

namespace {

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

// Adds [begin, end) to spans (sorted by address, adjacent spans merged)
void add_span(Vector<Span>& spans, uintptr_t begin, uintptr_t end) {
  CHECK(end > begin);
  Span* s = spans.data();
  size_t a = 0;
  size_t b = spans.size();
  const size_t e = spans.size();
  while (a < b) {
    size_t mid = (a + b) / 2;
    if (s[mid].begin <= begin) {
      a = mid + 1;
    } else {
      b = mid;
    }
  }
  if (a != 0 && s[a - 1].end == begin) {
    if (a != e && s[a].begin == end) {
      s[a - 1].end = s[a].end;
      spans.erase(spans.begin() + a);
    } else {
      s[a - 1].end = end;
    }
    return;
  }
  if (a != e && s[a].begin == end) {
    s[a].begin = begin;
    return;
  }
  spans.insert(spans.begin() + a, Span{begin, end});
}

// A stream: its handle, and its id (cuStreamGetId), which identifies it. Handles do not: the driver hands a destroyed
// stream's handle to the next stream created, even while the destroyed stream's work still runs. Ids are taken while
// the stream is in use (allocation, recordStream); the driver may not be called with a destroyed stream's handle, so
// memory must be freed before the streams that used it are destroyed (the free records an event on each).
struct StreamRef {
  CUstream stream = nullptr;
  uint64_t id = 0;
};

uint64_t streamId(CUstream stream) {
  unsigned long long id = 0;
  if (cudaApi.streamGetId && cudaApi.streamGetId(stream, &id) == CUDA_SUCCESS) {
    return id;
  }
  // (drivers before CUDA 12.0 have no stream ids)
  return (uintptr_t)stream;
}

// Free memory [span) was in use on eventStream until event (recorded when it was freed): an allocation on another
// stream that takes any of it waits for the event first
struct EventRegion {
  IntrusiveListLink<EventRegion> link;     // in its free region's list, by address
  IntrusiveListLink<EventRegion> fifoLink; // if captureId is 0: in its stream's list, in the order recorded
  Span span;
  Event event;
  CUstream eventStream = nullptr;
  uint64_t eventStreamId = 0;
  // Nonzero if recorded during a capture: the capture's id (see the top of the file)
  unsigned long long captureId = 0;
};

using EventList = IntrusiveList<EventRegion, &EventRegion::link>;
using EventFifo = IntrusiveList<EventRegion, &EventRegion::fifoLink>;

template<typename RegionContainer>
struct RegionSize {
  size_t bytes = 0;
  mutable typename RegionContainer::iterator regionIterator;

  RegionSize() = default;
  RegionSize(size_t bytes, typename RegionContainer::iterator regionIterator)
      : bytes(bytes), regionIterator(regionIterator) {}
};

struct RegionSizeCompare {
  using is_transparent = std::true_type;
  template<typename RegionSize>
  bool operator()(const RegionSize& a, const RegionSize& b) const {
    if (a.bytes != b.bytes) {
      return a.bytes < b.bytes;
    }
    return a.regionIterator->span.begin < b.regionIterator->span.begin;
  }
  template<typename RegionSize>
  bool operator()(const RegionSize& a, size_t b) const {
    return a.bytes < b;
  }
  template<typename RegionSize>
  bool operator()(size_t a, const RegionSize& b) const {
    return a < b.bytes;
  }
};

struct RegionCompare {
  using is_transparent = std::true_type;
  template<typename Region>
  bool operator()(const Region& a, const Region& b) const {
    return a.span.begin < b.span.begin;
  }
  template<typename Region>
  bool operator()(const Region& a, Span b) const {
    return a.span.begin < b.begin;
  }
  template<typename Region>
  bool operator()(Span a, const Region& b) const {
    return a.begin < b.span.begin;
  }
};

// A free region, with the events of the streams that used its memory (in address order; each within the region)
struct Region {
  mutable Span span;
  mutable EventList events;
  mutable std::multiset<RegionSize<std::set<Region, RegionCompare>>, RegionSizeCompare>::iterator sizeIterator;

  Region() = default;
  Region(Span span) : span(span) {}
};

using RegionMap = std::set<Region, RegionCompare>;
using RegionSizeMap = std::multiset<RegionSize<std::set<Region, RegionCompare>>, RegionSizeCompare>;

// A free list: regions by address (never adjacent: adjacent regions merge) and by size
struct Regions {
  RegionMap map;
  RegionSizeMap sizes;
};

template<typename T>
struct LocalFreeList {
  PoolAllocator<T> allocator;
  Vector<T*> list;

  struct Handle {
    LocalFreeList* owner = nullptr;
    T* ptr;
    Handle() = default;
    Handle(LocalFreeList* owner, T* ptr) : owner(owner), ptr(ptr) {}
    ~Handle() {
      if (owner && ptr) {
        owner->list.push_back(ptr);
      }
    }
    Handle(Handle&& n) noexcept {
      std::swap(owner, n.owner);
      std::swap(ptr, n.ptr);
    }
    Handle& operator=(Handle&& n) noexcept {
      std::swap(owner, n.owner);
      std::swap(ptr, n.ptr);
      return *this;
    }
    operator T&() const {
      return *ptr;
    }
    T& operator*() const {
      return *ptr;
    }
    T* operator->() const {
      return &*ptr;
    }
    T* release() {
      return std::exchange(ptr, nullptr);
    }
  };

  void push(T* ptr) {
    list.push_back(ptr);
  }

  Handle pop() {
    if (!list.empty()) {
      T* ptr = list.back();
      list.pop_back();
      return Handle(this, ptr);
    }
    return Handle(this, allocator.allocate());
  }
};

struct PointerHash {
  template<typename T>
  size_t operator()(T p) const {
    return (uintptr_t)p * 6364136223846793005ull % 2147483647ull;
  }
};

// The free memory of a pool (see the top of the file)
struct Pool {
  struct StreamFree {
    StreamRef stream;
    Regions free;
  };
  // Normal mode: memory freed, in a list per stream it was allocated on (in the order of the streams' first frees), by
  // stream id
  Vector<std::unique_ptr<StreamFree>> streams;
  HashMap<uint64_t, StreamFree*, PointerHash> streamIndex;
  // Normal mode: memory no stream has used yet (and memory returned by private pools). Pressure mode: all free memory.
  Regions common;
  bool pressure = false;

  Regions* findStreamFree(uint64_t streamId) {
    auto i = streamIndex.find(streamId);
    return i != streamIndex.end() ? &i->second->free : nullptr;
  }
  Regions& streamFree(StreamRef stream) {
    if (Regions* r = findStreamFree(stream.id)) {
      return *r;
    }
    auto sf = std::make_unique<StreamFree>();
    sf->stream = stream;
    StreamFree* p = sf.get();
    streams.push_back(std::move(sf));
    streamIndex[stream.id] = p;
    return p->free;
  }
  template<typename F>
  void forEachList(F&& f) {
    f(common);
    for (auto& sf : streams) {
      f(sf->free);
    }
  }
};

enum class PoolKind : uint8_t { main, comm, private_ };

struct MempoolId {
  uint64_t first = 0;
  uint64_t second = 0;
  bool operator==(const MempoolId& n) const {
    return first == n.first && second == n.second;
  }
};

// Within its scope, this thread may make calls that a capture's mode forbids (cuEventQuery), whatever the mode of any
// capture: queries of events recorded outside captures are allowed during captures in relaxed mode, and invalidate
// none (unlike in the global and thread-local modes, where the query fails and invalidates the capture)
struct RelaxedCaptureMode {
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
  RelaxedCaptureMode() {
    CHECK_CU(cuThreadExchangeStreamCaptureMode(&mode));
  }
  ~RelaxedCaptureMode() {
    cuThreadExchangeStreamCaptureMode(&mode);
  }
  RelaxedCaptureMode(const RelaxedCaptureMode&) = delete;
  RelaxedCaptureMode& operator=(const RelaxedCaptureMode&) = delete;
};

// Set by graphReleased (on a thread of the CUDA driver), cleared by the allocator when it handles it
std::atomic_bool graphsReleased = false;

} // namespace

// See allocatorCommPoolCreate. Guarded by the allocator's mutex.
struct CommPool {
  // Slabs are 2 MiB aligned (multicast binding granularity), at least minSlabBytes, with slabGap unused logical bytes
  // before each (so that free memory of two slabs never merges into one region)
  static constexpr size_t slabAlignment = (size_t)2 << 20;
  static constexpr size_t minSlabBytes = (size_t)256 << 20;
  static constexpr size_t slabGap = slabAlignment;
  struct Slab {
    uintptr_t local; // the slab's memory in this process
    size_t logical;  // its logical offset
    size_t bytes;
    bool bound = true; // every listener bound every piece
  };
  Pool pool; // in logical coordinates
  Vector<Slab> slabs;
  size_t nextLogical = slabGap;
  Vector<std::pair<bool (*)(void*, size_t, unsigned long long, size_t, size_t), void*>> listeners;
};

// A PyTorch memory pool (see cudaAllocatorBeginAllocateToPool). Guarded by the allocator's mutex, except liveGraphs.
struct PrivatePool {
  MempoolId id;
  Pool pool;
  // PyTorch's use count: cudaAllocatorCreateOrIncrefPool and cudaAllocatorBeginAllocateToPool add one,
  // cudaAllocatorReleasePool removes one
  int useCount = 0;
  // Allocations from the pool not freed yet
  size_t liveAllocations = 0;
  // Graphs that may use the pool's memory: those captured with allocations or frees in the pool, until the driver
  // destroys their user object (see attachGraph); their capture ids
  std::atomic_int liveGraphs = 0;
  Vector<unsigned long long> captureIds;
  bool released = false; // useCount reached 0
  bool retired = false;  // its free memory went back to the main pool, as does memory freed since
};

namespace {
// A user object's destructor (see attachGraph), called by the CUDA driver on a thread of its own once the graph that
// retained the object, and the executable graphs instantiated from it, are destroyed and their launches have
// completed. No CUDA calls and no locks here: the allocator retires the pool on its next call (see maintenance).
void graphReleased(void* ptr) {
  static_cast<PrivatePool*>(ptr)->liveGraphs.fetch_sub(1);
  graphsReleased.store(true);
}
} // namespace

// CudaAllocatorImpl is defined in moodist namespace so it can be forward-declared
struct CudaAllocatorImpl {
  CudaAllocatorImpl() {
    const char* debug = std::getenv("MOODIST_DEBUG_ALLOCATOR");
    const int level = debug ? std::atoi(debug) : 0;
    validateEveryCall = level == 1 || level >= 3;
    reportOnMap = level >= 2;
    if (const char* path = std::getenv("MOODIST_DEBUG_ALLOCATOR_TRACE")) {
      traceFile = std::fopen(fmt::sprintf("%s.%d", path, (int)getpid()).c_str(), "w");
      if (traceFile) {
        std::setvbuf(traceFile, nullptr, _IOFBF, 1 << 20);
      }
    }
  }

  // Mutex for all allocator operations
  SpinMutex mutex;

  // CUDA context - initialized on first allocation
  CUcontext cuContext = nullptr;
  CUdevice cuDevice = 0;
  int deviceIndex = -1;

  // Stream tracking for recordStream (ptr -> list of streams that have used it)
  HashMap<uintptr_t, std::unique_ptr<Vector<StreamRef>>, PointerHash> streamUses;
  Vector<std::unique_ptr<Vector<StreamRef>>> streamListFree;

  // Free callbacks (ptr -> linked list of callbacks)
  HashMap<uintptr_t, FunctionPointer, PointerHash> freeCallbacks;

  // The chunks mapped (see mapMoreMemory)
  SpinMutex mappedRegionsMutex;
  Vector<Span> mappedRegions;
  std::atomic_size_t reservedSize = 0;
  std::atomic_uintptr_t reservedBase = 0;
  size_t nextMapBase = 0;
  Vector<std::pair<size_t, CUmemGenericAllocationHandle>> cuMemHandles;
  // See allocatorAddChunkListener
  Vector<std::pair<void (*)(void*, unsigned long long, size_t, size_t), void*>> chunkListeners;
  size_t allocationGranularity = 0;
  bool fabricSupported = false;
  bool gpuDirectRdmaSupported = false;

  Pool mainPool;
  Vector<CommPool*> commPools; // all of them (never freed), for allocatorAnyCommPoolOffset and validate

  // Recycled region nodes and event regions
  Vector<RegionMap::node_type> freeRegionNodes;
  Vector<RegionSizeMap::node_type> freeRegionSizeNodes;
  LocalFreeList<EventRegion> freeListEventRegions;

  // Event regions with eager events (captureId 0), by stream, each stream's in the order recorded (see processEvents)
  struct StreamEvents {
    StreamRef stream;
    EventFifo events;
    bool detached = false; // the stream was destroyed (see streamDestroyed): removed once empty
  };
  Vector<std::unique_ptr<StreamEvents>> eagerEvents;
  HashMap<uint64_t, StreamEvents*, PointerHash> eagerEventsIndex;
  // Event regions with captured events
  size_t numCapturedEvents = 0;

  // Private pools (see cudaAllocatorBeginAllocateToPool): the routing scopes, the pools by id (until released), the
  // released pools that wait for their graphs
  struct Scope {
    PrivatePool* pool = nullptr;
    CudaAllocatorPoolFilter filter = nullptr;
    void* filterCtx = nullptr;
    void (*destroyFilterCtx)(void*) = nullptr;
  };
  Vector<Scope> scopes;
  Vector<PrivatePool*> privatePools;
  Vector<PrivatePool*> releasedPools;

  // Frees that wait for the end of a capture (see freeIn)
  struct DeferredFree {
    PoolKind kind;
    void* owner;
    Span span;
    Vector<StreamRef> streams;
  };
  Vector<DeferredFree> deferredFrees;

  CudaAllocatorStats stats;
  // The most memory to map (0: no limit; see cudaAllocatorSetMemoryLimit)
  size_t mapLimit = 0;
  bool validateEveryCall = false;
  bool reportOnMap = false;
  // MOODIST_DEBUG_ALLOCATOR_TRACE=<path>: every allocation and free goes to <path>.<pid> (see trace)
  FILE* traceFile = nullptr;
  std::chrono::steady_clock::time_point traceStart = std::chrono::steady_clock::now();

  // Temporaries
  Vector<StreamRef> tmpStreams;
  Vector<unsigned long long> tmpCaptureIds;
  Vector<LocalFreeList<EventRegion>::Handle> tmpEventRegionHandles;
  Vector<EventRegion*> tmpEventRegions;
  Vector<std::pair<uint64_t, unsigned long long>> tmpStreamCaptures;

  // ---------------------------------------------------------------------------------------------------------------
  // Free lists

  void updateSize(Regions& regions, RegionMap::iterator i) {
    auto sizeNode = regions.sizes.extract(i->sizeIterator);
    CHECK(sizeNode.value().regionIterator == i);
    sizeNode.value().bytes = i->span.end - i->span.begin;
    i->sizeIterator = regions.sizes.insert(std::move(sizeNode));
  }

  // Adds span (with the event regions of eventRegionList, in address order, within span) to regions, merging it with
  // adjacent regions
  template<typename EventRegionList>
  RegionMap::iterator insertRegion(Regions& regions, Span span, const EventRegionList& eventRegionList) {
    auto& map = regions.map;
    auto& sizeMap = regions.sizes;
    auto i = map.upper_bound(span);

    if (i != map.begin()) {
      auto pi = std::prev(i);
      CHECK(pi->span.end <= span.begin);
      if (pi->span.end == span.begin) {
        if (i != map.end() && i->span.begin == span.end) {
          pi->span.end = i->span.end;
          updateSize(regions, pi);
          for (auto& v : eventRegionList) {
            pi->events.push_back(*v);
          }
          for (auto it = i->events.begin(); it != i->events.end();) {
            EventRegion* r = &*it;
            it = i->events.erase(it);
            pi->events.push_back(*r);
          }
          CHECK(i->events.empty());
          freeRegionSizeNodes.push_back(sizeMap.extract(i->sizeIterator));
          freeRegionNodes.push_back(map.extract(i));
          return pi;
        }
        pi->span.end = span.end;
        updateSize(regions, pi);
        for (auto& v : eventRegionList) {
          pi->events.push_back(*v);
        }
        return pi;
      }
    }
    if (i != map.end()) {
      CHECK(i->span.begin >= span.end);
    }
    if (i != map.end() && i->span.begin == span.end) {
      i->span.begin = span.begin;
      updateSize(regions, i);
      auto at = i->events.begin();
      for (auto& v : eventRegionList) {
        i->events.insert(at, *v);
      }
      return i;
    }
    if (freeRegionNodes.empty()) {
      i = map.emplace_hint(i, span);
    } else {
      auto node = std::move(freeRegionNodes.back());
      freeRegionNodes.pop_back();
      node.value().span = span;
      CHECK(node.value().events.empty());
      i = map.insert(i, std::move(node));
    }
    CHECK(i->events.empty());
    for (auto& v : eventRegionList) {
      i->events.push_back(*v);
    }
    if (freeRegionSizeNodes.empty()) {
      i->sizeIterator = sizeMap.emplace(span.end - span.begin, i);
    } else {
      auto node = std::move(freeRegionSizeNodes.back());
      freeRegionSizeNodes.pop_back();
      node.value().bytes = span.end - span.begin;
      node.value().regionIterator = i;
      i->sizeIterator = sizeMap.insert(std::move(node));
    }
    return i;
  }

  // Removes region i of regions (its event regions stay linked to nothing: the caller takes them)
  void eraseRegion(Regions& regions, RegionMap::iterator i) {
    freeRegionSizeNodes.push_back(regions.sizes.extract(i->sizeIterator));
    freeRegionNodes.push_back(regions.map.extract(i));
  }

  // Moves every region of from (with its events) into to
  void moveRegions(Regions& from, Regions& to) {
    for (auto i = from.map.begin(); i != from.map.end();) {
      Span span = i->span;
      tmpEventRegions.clear();
      for (auto& e : i->events) {
        tmpEventRegions.push_back(&e);
      }
      i->events.clear();
      auto next = std::next(i);
      eraseRegion(from, i);
      i = next;
      insertRegion(to, span, tmpEventRegions);
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Events

  EventFifo& eagerFifo(StreamRef stream) {
    auto i = eagerEventsIndex.find(stream.id);
    if (i != eagerEventsIndex.end()) {
      return i->second->events;
    }
    auto se = std::make_unique<StreamEvents>();
    se->stream = stream;
    StreamEvents* p = se.get();
    eagerEvents.push_back(std::move(se));
    eagerEventsIndex[stream.id] = p;
    return p->events;
  }

  // Recycles e, which its region no longer lists
  void releaseEventRegion(EventRegion& e) {
    if (e.captureId != 0) {
      CHECK(numCapturedEvents > 0);
      --numCapturedEvents;
      e.captureId = 0;
    } else if (e.fifoLink.prev) {
      EventFifo::erase(e);
    }
    freeListEventRegions.push(&e);
  }

  // Drops the event regions whose (eager) events have completed: each stream's events complete in the order recorded,
  // so this queries each stream's oldest events until one has not completed
  void processEvents() {
    bool any = false;
    for (auto& se : eagerEvents) {
      if (!se->events.empty()) {
        any = true;
        break;
      }
    }
    if (!any) {
      return;
    }
    RelaxedCaptureMode relaxed;
    for (auto& se : eagerEvents) {
      while (!se->events.empty()) {
        EventRegion& e = se->events.front();
        CUresult r = cuEventQuery(e.event);
        if (r == CUDA_ERROR_NOT_READY) {
          break;
        }
        CHECK_CU(r);
        EventList::erase(e);
        releaseEventRegion(e);
      }
    }
    for (size_t i = 0; i != eagerEvents.size();) {
      if (eagerEvents[i]->detached && eagerEvents[i]->events.empty()) {
        eagerEvents.erase(eagerEvents.begin() + i);
      } else {
        ++i;
      }
    }
  }

  // stream is about to be destroyed: no allocation will come from its free lists again, so their memory (with its
  // events) goes to the pools' common lists. (moodist's own temporary streams, e.g. compile_op's tuning stream: their
  // memory would be lost to the other streams otherwise.)
  void streamDestroyed(CUstream stream) {
    const uint64_t id = streamId(stream);
    auto release = [&](Pool& pool) {
      auto i = pool.streamIndex.find(id);
      if (i == pool.streamIndex.end()) {
        return;
      }
      Pool::StreamFree* sf = i->second;
      pool.streamIndex.erase(i);
      moveRegions(sf->free, pool.common);
      for (auto j = pool.streams.begin(); j != pool.streams.end(); ++j) {
        if (j->get() == sf) {
          pool.streams.erase(j);
          break;
        }
      }
    };
    release(mainPool);
    for (CommPool* cp : commPools) {
      release(cp->pool);
    }
    for (PrivatePool* pp : privatePools) {
      release(pp->pool);
    }
    for (PrivatePool* pp : releasedPools) {
      release(pp->pool);
    }
    // (its events stay valid)
    auto i = eagerEventsIndex.find(id);
    if (i != eagerEventsIndex.end()) {
      i->second->detached = true;
      eagerEventsIndex.erase(i);
    }
  }

  struct CaptureInfo {
    unsigned long long id = 0; // 0: not capturing
    CUgraph graph = nullptr;   // null if the capture has been invalidated
  };
  static constexpr unsigned long long invalidatedCapture = ~0ull;
  CaptureInfo captureInfo(CUstream stream) {
    CaptureInfo r;
    if (stream == nullptr) {
      // The legacy default stream is never captured (and the query can fail while another stream captures)
      return r;
    }
    CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
    unsigned long long id = 0;
    CUgraph graph = nullptr;
    CUresult err = cuStreamGetCaptureInfo(stream, &status, &id, &graph);
    if (err != CUDA_SUCCESS) {
      // (e.g. a stream destroyed since: not capturing)
      log.debug("Moodist CUDA allocator: cuStreamGetCaptureInfo failed with error %d\n", err);
      return r;
    }
    if (status == CU_STREAM_CAPTURE_STATUS_ACTIVE) {
      CHECK(id != 0);
      r.id = id;
      r.graph = graph;
    } else if (status == CU_STREAM_CAPTURE_STATUS_INVALIDATED) {
      // (until the capture ends: an id that matches no capture's events, and no graph)
      r.id = invalidatedCapture;
    }
    return r;
  }

  // Drops the captured events of pp whose capture has ended (all: all of them)
  void dropCapturedEvents(PrivatePool& pp, bool all) {
    if (numCapturedEvents == 0) {
      return;
    }
    tmpStreamCaptures.clear();
    auto ended = [&](const EventRegion& e) {
      if (all) {
        return true;
      }
      for (auto& [stream, id] : tmpStreamCaptures) {
        if (stream == e.eventStreamId) {
          return id != e.captureId;
        }
      }
      unsigned long long id = captureInfo(e.eventStream).id;
      tmpStreamCaptures.emplace_back(e.eventStreamId, id);
      return id != e.captureId;
    };
    pp.pool.forEachList([&](Regions& regions) {
      for (auto& region : regions.map) {
        for (auto i = region.events.begin(); i != region.events.end();) {
          if (i->captureId != 0 && ended(*i)) {
            EventRegion& e = *i;
            i = region.events.erase(i);
            releaseEventRegion(e);
          } else {
            ++i;
          }
        }
      }
    });
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Allocation

  // The stream an allocation is for, and the capture it is in (if any)
  struct AllocContext {
    CUstream stream = nullptr;
    uint64_t streamId = 0;
    unsigned long long captureId = 0;
    CUgraph graph = nullptr;
  };
  AllocContext context(CUstream stream) {
    AllocContext ctx;
    ctx.stream = stream;
    ctx.streamId = moodist::streamId(stream);
    // (PyTorch routes a capture's allocations to a private pool before the capture begins: no scopes, no captures)
    if (!scopes.empty()) {
      CaptureInfo ci = captureInfo(stream);
      ctx.captureId = ci.id;
      ctx.graph = ci.graph;
    }
    return ctx;
  }

  // The private pool allocations on stream are routed to (see cudaAllocatorBeginAllocateToPool), if any
  PrivatePool* route(CUstream stream) {
    for (auto& s : scopes) {
      if (s.filter(s.filterCtx, stream)) {
        return s.pool;
      }
    }
    return nullptr;
  }

  enum class FitMode { any, noWait };

  // Whether an allocation in ctx can take the first bytes of r: the events on them are of the capture the allocation
  // is in (none: of no capture), and, for noWait, of the allocation's stream
  static bool usable(const Region& r, size_t bytes, const AllocContext& ctx, FitMode mode) {
    const uintptr_t end = r.span.begin + bytes;
    for (auto& e : r.events) {
      if (e.span.begin >= end) {
        break;
      }
      if (e.captureId != ctx.captureId) {
        return false;
      }
      if (mode == FitMode::noWait && e.eventStreamId != ctx.streamId) {
        return false;
      }
    }
    return true;
  }

  // The best fit for bytes in regions (the smallest usable region, the lowest address among equal sizes), or end
  RegionMap::iterator findFit(Regions& regions, size_t bytes, const AllocContext& ctx, FitMode mode) {
    auto it = regions.sizes.lower_bound(bytes);
    if (it == regions.sizes.end()) {
      return regions.map.end();
    }
    if (mode == FitMode::any && ctx.captureId == 0 && numCapturedEvents == 0) {
      // (every region is usable)
      return it->regionIterator;
    }
    for (; it != regions.sizes.end(); ++it) {
      if (usable(*it->regionIterator, bytes, ctx, mode)) {
        return it->regionIterator;
      }
    }
    return regions.map.end();
  }

  // Allocates the first bytes of region r of regions (usable in ctx, see findFit): ctx's stream waits for the events
  // of other streams on them
  uintptr_t carve(Regions& regions, RegionMap::iterator r, size_t bytes, const AllocContext& ctx) {
    const uintptr_t ptr = r->span.begin;
    const uintptr_t ptrEnd = ptr + bytes;
    CHECK(ptrEnd <= r->span.end);
    for (auto i = r->events.begin(); i != r->events.end();) {
      if (i->span.begin >= ptrEnd) {
        break;
      }
      CHECK(i->captureId == ctx.captureId);
      if (i->eventStreamId != ctx.streamId) {
        CHECK_CU(cuStreamWaitEvent(ctx.stream, i->event, CU_EVENT_WAIT_DEFAULT));
      }
      if (i->span.end > ptrEnd) {
        i->span.begin = ptrEnd;
        ++i;
      } else {
        EventRegion& e = *i;
        i = r->events.erase(i);
        releaseEventRegion(e);
      }
    }
    if (r->span.end == ptrEnd) {
      CHECK(r->events.empty());
      eraseRegion(regions, r);
    } else {
      r->span.begin = ptrEnd;
      updateSize(regions, r);
    }
    return ptr;
  }

  uintptr_t allocateFrom(Regions& regions, size_t bytes, const AllocContext& ctx, FitMode mode) {
    auto r = findFit(regions, bytes, ctx, mode);
    return r != regions.map.end() ? carve(regions, r, bytes, ctx) : 0;
  }

  Pool& poolOf(PoolKind kind, void* owner) {
    switch (kind) {
    case PoolKind::main:
      return mainPool;
    case PoolKind::comm:
      return static_cast<CommPool*>(owner)->pool;
    case PoolKind::private_:
      return static_cast<PrivatePool*>(owner)->pool;
    }
    CHECK(false);
  }

  // An allocation of bytes from the pool of kind (owner: its CommPool or PrivatePool) in ctx, or 0 if there is no
  // memory for it (see the top of the file)
  uintptr_t allocateIn(PoolKind kind, void* owner, size_t bytes, const AllocContext& ctx) {
    Pool& pool = poolOf(kind, owner);
    if (ctx.captureId != 0) {
      // (completed events are no obstacle)
      processEvents();
    }
    if (!pool.pressure) {
      if (Regions* own = pool.findStreamFree(ctx.streamId)) {
        if (uintptr_t r = allocateFrom(*own, bytes, ctx, FitMode::any)) {
          return r;
        }
      }
      if (uintptr_t r = allocateFrom(pool.common, bytes, ctx, FitMode::any)) {
        return r;
      }
      if (uintptr_t r = grow(kind, owner, bytes, ctx)) {
        return r;
      }
      if (freeBytes(pool) < bytes) {
        // (no placement could fit it: not a reason for pressure mode)
        return 0;
      }
      enterPressureMode(pool);
    }
    processEvents();
    if (uintptr_t r = allocateFrom(pool.common, bytes, ctx, FitMode::noWait)) {
      return r;
    }
    if (uintptr_t r = allocateFrom(pool.common, bytes, ctx, FitMode::any)) {
      return r;
    }
    return grow(kind, owner, bytes, ctx);
  }

  static size_t freeBytes(Pool& pool) {
    size_t n = 0;
    pool.forEachList([&](Regions& regions) {
      for (auto& r : regions.map) {
        n += r.span.end - r.span.begin;
      }
    });
    return n;
  }

  void enterPressureMode(Pool& pool) {
    if (pool.pressure) {
      return;
    }
    pool.pressure = true;
    for (auto& sf : pool.streams) {
      moveRegions(sf->free, pool.common);
    }
    log.verbose("Moodist CUDA allocator: a pool is out of new memory, and from now on shares its free memory between "
                "streams\n");
  }

  uintptr_t grow(PoolKind kind, void* owner, size_t bytes, const AllocContext& ctx) {
    switch (kind) {
    case PoolKind::main:
      return growMain(bytes, ctx);
    case PoolKind::comm:
      return growComm(*static_cast<CommPool*>(owner), bytes, ctx);
    case PoolKind::private_:
      return takeFromMain(bytes, ctx);
    }
    CHECK(false);
  }

  // A new chunk, into the main pool's common list (where it merges with free memory at the end of the previous
  // chunk, if any), and the allocation from there. During a capture, where free memory before the chunk may carry
  // events the allocation cannot use, the allocation is the chunk's first bytes.
  uintptr_t growMain(size_t bytes, const AllocContext& ctx) {
    Span span = mapMoreMemory(bytes);
    if (span.begin == 0) {
      return 0;
    }
    if (ctx.captureId == 0) {
      insertRegion(mainPool.common, span, std::array<EventRegion*, 0>{});
      uintptr_t r = allocateFrom(mainPool.common, bytes, ctx, FitMode::any);
      CHECK(r != 0);
      return r;
    }
    if (span.end - span.begin > bytes) {
      insertRegion(mainPool.common, Span{span.begin + bytes, span.end}, std::array<EventRegion*, 0>{});
    }
    return span.begin;
  }

  // A private pool's new memory: the best fit among all free memory of the main pool that needs no wait (that does
  // not exist during a capture: no other events are usable then), then the best fit with waits, then a new chunk
  uintptr_t takeFromMain(size_t bytes, const AllocContext& ctx) {
    processEvents();
    for (int pass = 0; pass != 2; ++pass) {
      for (FitMode mode : {FitMode::noWait, FitMode::any}) {
        Regions* best = nullptr;
        RegionMap::iterator bestIt;
        mainPool.forEachList([&](Regions& regions) {
          auto i = findFit(regions, bytes, ctx, mode);
          if (i != regions.map.end() && (!best || i->sizeIterator->bytes < bestIt->sizeIterator->bytes)) {
            best = &regions;
            bestIt = i;
          }
        });
        if (best) {
          return carve(*best, bestIt, bytes, ctx);
        }
      }
      if (pass == 0) {
        if (uintptr_t r = growMain(bytes, ctx)) {
          return r;
        }
        if (mainPool.pressure || freeBytes(mainPool) < bytes) {
          break;
        }
        enterPressureMode(mainPool);
      }
    }
    return 0;
  }

  // A new slab for pool (from the main pool, allocated in ctx): the allocation is its first bytes, the rest goes to
  // the pool's common list
  uintptr_t growComm(CommPool& pool, size_t bytes, const AllocContext& ctx) {
    constexpr size_t align = CommPool::slabAlignment;
    const size_t slabBytes = std::max((bytes + align - 1) / align * align, CommPool::minSlabBytes);
    if (pool.nextLogical + slabBytes > commPoolLogicalSize) {
      throw std::runtime_error(
          fmt::sprintf("Moodist communication pool: no logical address space left for %d more bytes (%d in use)", bytes,
              pool.nextLogical));
    }
    // From the main pool, at a boundary of the binding granularity (the chunks start at one); never freed
    uintptr_t raw = allocateIn(PoolKind::main, nullptr, slabBytes + align, ctx);
    if (raw == 0) {
      return 0;
    }
    noteAllocation(slabBytes + align, slabBytes + align);
    trace('a', 's', raw, slabBytes + align, ctx.stream);
    CommPool::Slab slab;
    slab.local = (raw + align - 1) / align * align;
    slab.logical = pool.nextLogical;
    slab.bytes = slabBytes;
    pool.nextLogical += slabBytes + CommPool::slabGap;
    for (auto& [fn, fnCtx] : pool.listeners) {
      forEachChunkPiece(slab.local, slab.bytes,
          [&](CUmemGenericAllocationHandle handle, size_t memOffset, size_t size, size_t offset) {
            slab.bound &= fn(fnCtx, slab.logical + offset, handle, memOffset, size);
          });
    }
    pool.slabs.push_back(slab);
    log.verbose("Moodist communication pool: slab of %d bytes at %#x, logical offset %#x\n", slab.bytes, slab.local,
        slab.logical);

    if (slabBytes > bytes) {
      // The main pool may have reused memory on ctx.stream without waits: an allocation of the rest on another stream
      // waits for ctx.stream's work so far
      auto h = freeListEventRegions.pop();
      if (!h->event) {
        h->event = Event::create();
      }
      CHECK_CU(cuEventRecord(h->event, ctx.stream));
      h->span = Span{slab.logical + bytes, slab.logical + slabBytes};
      h->eventStream = ctx.stream;
      h->eventStreamId = ctx.streamId;
      h->captureId = 0;
      insertRegion(pool.pool.common, h->span, std::array<EventRegion*, 1>{&*h});
      eagerFifo(StreamRef{ctx.stream, ctx.streamId}).push_back(*h);
      h.release();
    }
    return slab.logical;
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Free

  // Frees [span) of the pool of kind (owner: its CommPool or PrivatePool), memory used on streamList (the allocation's
  // stream first): records an event on each of these streams, and puts the memory into the pool's free lists (memory
  // of a retired private pool into the main pool's). Returns false, and does nothing, if the memory cannot be freed
  // yet: a stream that used it is capturing (unless the memory is of the capture's private pool), or an event could
  // not be recorded. See deferredFrees.
  template<typename StreamList>
  bool freeIn(PoolKind kind, void* owner, Span span, const StreamList& streamList) {
    tmpStreams.clear();
    for (const StreamRef& s : streamList) {
      bool seen = false;
      for (const StreamRef& t : tmpStreams) {
        seen |= t.id == s.id;
      }
      if (!seen) {
        tmpStreams.push_back(s);
      }
    }
    CHECK(!tmpStreams.empty());
    PrivatePool* pp = kind == PoolKind::private_ ? static_cast<PrivatePool*>(owner) : nullptr;
    if (pp && pp->retired) {
      kind = PoolKind::main;
      owner = nullptr;
    }
    tmpCaptureIds.clear();
    for (size_t i = 0; i != tmpStreams.size(); ++i) {
      tmpCaptureIds.push_back(0);
    }
    if (!scopes.empty()) {
      for (size_t i = 0; i != tmpStreams.size(); ++i) {
        CaptureInfo ci = captureInfo(tmpStreams[i].stream);
        if (ci.id == 0) {
          continue;
        }
        // Only memory of the capture's own private pool is freed during a capture (with events of the capture)
        if (kind != PoolKind::private_ || !ci.graph || route(tmpStreams[i].stream) != pp || !attachGraph(*pp, ci)) {
          return false;
        }
        tmpCaptureIds[i] = ci.id;
      }
    }
    tmpEventRegionHandles.clear();
    for (size_t i = 0; i != tmpStreams.size(); ++i) {
      auto h = freeListEventRegions.pop();
      if (!h->event) {
        h->event = Event::tryCreate();
        if (!h->event) {
          return false;
        }
      }
      if (cuEventRecord(h->event, tmpStreams[i].stream) != CUDA_SUCCESS) {
        return false;
      }
      h->span = span;
      h->eventStream = tmpStreams[i].stream;
      h->eventStreamId = tmpStreams[i].id;
      h->captureId = tmpCaptureIds[i];
      tmpEventRegionHandles.push_back(std::move(h));
    }
    Pool& pool = poolOf(kind, owner);
    insertRegion(pool.pressure ? pool.common : pool.streamFree(tmpStreams[0]), span, tmpEventRegionHandles);
    for (auto& h : tmpEventRegionHandles) {
      EventRegion& e = *h;
      if (e.captureId != 0) {
        ++numCapturedEvents;
      } else {
        eagerFifo(StreamRef{e.eventStream, e.eventStreamId}).push_back(e);
      }
      h.release();
    }
    tmpEventRegionHandles.clear();
    if (pp) {
      CHECK(pp->liveAllocations > 0);
      --pp->liveAllocations;
      if (pp->retired && pp->liveAllocations == 0) {
        internalDelete(pp);
      }
    }
    return true;
  }

  // freeIn, or later (see processDeferredFrees)
  template<typename StreamList>
  void freeOrDefer(PoolKind kind, void* owner, Span span, const StreamList& streamList) {
    if (!freeIn(kind, owner, span, streamList)) {
      DeferredFree d;
      d.kind = kind;
      d.owner = owner;
      d.span = span;
      for (const StreamRef& s : streamList) {
        d.streams.push_back(s);
      }
      deferredFrees.push_back(std::move(d));
    }
  }

  void processDeferredFrees() {
    if (deferredFrees.empty()) {
      return;
    }
    Vector<DeferredFree> list = std::move(deferredFrees);
    deferredFrees.clear();
    for (auto& d : list) {
      if (!freeIn(d.kind, d.owner, d.span, d.streams)) {
        deferredFrees.push_back(std::move(d));
      }
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Private pools

  PrivatePool* findPool(MempoolId id) {
    for (PrivatePool* pp : privatePools) {
      if (pp->id == id) {
        return pp;
      }
    }
    return nullptr;
  }

  PrivatePool* createOrIncrefPool(MempoolId id) {
    PrivatePool* pp = findPool(id);
    if (!pp) {
      pp = internalNew<PrivatePool>();
      pp->id = id;
      privatePools.push_back(pp);
    }
    ++pp->useCount;
    return pp;
  }

  // The graph being captured (ci) may use memory of pp: pp's memory goes back to the main pool only after the driver
  // has destroyed a user object that the graph retains. Returns false if the graph could not retain it.
  bool attachGraph(PrivatePool& pp, const CaptureInfo& ci) {
    if (std::find(pp.captureIds.begin(), pp.captureIds.end(), ci.id) != pp.captureIds.end()) {
      return true;
    }
    if (!ci.graph) {
      return false;
    }
    RelaxedCaptureMode relaxed;
    CUuserObject object = nullptr;
    CHECK_CU(cuUserObjectCreate(&object, &pp, graphReleased, 1, CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
    pp.liveGraphs.fetch_add(1);
    CUresult r = cuGraphRetainUserObject(ci.graph, object, 1, CU_GRAPH_USER_OBJECT_MOVE);
    if (r != CUDA_SUCCESS) {
      // (destroys the object: graphReleased undoes the count)
      cuUserObjectRelease(object, 1);
      log.error("Moodist CUDA allocator: cuGraphRetainUserObject failed with error %d\n", r);
      return false;
    }
    pp.captureIds.push_back(ci.id);
    return true;
  }

  // Once pp is released and no graph can use its memory anymore, its free memory goes back to the main pool (into the
  // list of the stream it was freed by, in normal mode), as does memory freed from it later (see freeIn), and the pool
  // is deleted when its last allocation is freed. Returns whether it did (pp may be deleted then).
  bool tryRetire(PrivatePool* pp) {
    CHECK(pp->released && !pp->retired);
    if (pp->liveGraphs.load() != 0) {
      return false;
    }
    dropCapturedEvents(*pp, true);
    Pool& p = pp->pool;
    if (mainPool.pressure) {
      p.forEachList([&](Regions& regions) {
        moveRegions(regions, mainPool.common);
      });
    } else {
      moveRegions(p.common, mainPool.common);
      for (auto& sf : p.streams) {
        moveRegions(sf->free, mainPool.streamFree(sf->stream));
      }
    }
    pp->retired = true;
    if (pp->liveAllocations == 0) {
      internalDelete(pp);
    }
    return true;
  }

  // Work left from earlier calls: released pools whose graphs are gone, frees that waited for captures to end
  void maintenance() {
    if (graphsReleased.load(std::memory_order_relaxed) && graphsReleased.exchange(false)) {
      for (size_t i = 0; i != releasedPools.size();) {
        if (tryRetire(releasedPools[i])) {
          releasedPools.erase(releasedPools.begin() + i);
        } else {
          ++i;
        }
      }
    }
    if (!deferredFrees.empty() && scopes.empty()) {
      processDeferredFrees();
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Mapping

  CUmemAllocationProp allocationProp() {
    CUmemAllocationProp prop;
    std::memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes =
        fabricSupported
            ? (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC)
            : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceIndex;
    prop.allocFlags.gpuDirectRDMACapable = gpuDirectRdmaSupported ? 1 : 0;
    return prop;
  }

  // Reserves the virtual addresses (on the first map)
  void reserve() {
    // Query fabric handle support
    int fabricAttr = 0;
    cuDeviceGetAttribute(&fabricAttr, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cuDevice);
    fabricSupported = fabricAttr != 0;

    // cuMemCreate fails with CUDA_ERROR_INVALID_DEVICE if gpuDirectRDMACapable is requested on a device
    // without GPUDirect RDMA (e.g. GeForce), so only request it where it is supported.
    int gdrAttr = 0;
    cuDeviceGetAttribute(&gdrAttr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, cuDevice);
    gpuDirectRdmaSupported = gdrAttr != 0;
    if (!gpuDirectRdmaSupported) {
      log.info("GPUDirect RDMA with CUDA VMM is not supported on this device; allocating without it\n");
    }

    CUmemAllocationProp prop = allocationProp();
    size_t granularity = 0;
    CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    allocationGranularity = granularity;

    // Probe: verify fabric handles actually work (device may report support
    // but IMEX daemon may not be running)
    if (fabricSupported) {
      CUmemGenericAllocationHandle probeHandle;
      auto probeErr = cuMemCreate(&probeHandle, granularity, &prop, 0);
      if (probeErr != CUDA_SUCCESS) {
        fabricSupported = false;
        log.info("Fabric handle probe failed, falling back to POSIX fd for VMM IPC\n");
      } else {
        cuMemRelease(probeHandle);
      }
    }

    constexpr size_t reserveSize = allocatorReserveSize; // 1 TB
    CUdeviceptr base = 0;
    CHECK_CU(cuMemAddressReserve(&base, reserveSize, granularity, 0, 0));
    reservedBase = base;
    reservedSize = reserveSize;
    nextMapBase = base;

    log.info(
        "Moodist CUDA Allocator reserved %d bytes of VA at %#x (granularity %d)\n", reserveSize, base, granularity);

    if (reservedBase & (1ull << 63)) {
      throw std::runtime_error(
          "Oh no, CUDA returned memory with bit 63 set! This should not happen. Moodist needs this bit for tagging.");
    }
  }

  // Maps a new chunk of at least minbytes after the chunks mapped so far: a multiple of the chunk size (1/64 of the
  // device's memory), or less if the device does not have that much free memory (leaving 512 MiB to the rest of the
  // process) or the limit (mapLimit) does not allow it. Returns its span, or an empty span if there is not enough
  // memory.
  Span mapMoreMemory(size_t minbytes) {
    std::lock_guard mrl(mappedRegionsMutex);
    CHECK(deviceIndex != -1);

    size_t free = 0;
    size_t total = 0;
    CHECK_CU(cuMemGetInfo(&free, &total));

    if (reservedBase == 0) {
      reserve();
    }

    constexpr size_t buffer = (size_t)512 << 20;
    const size_t granularity = allocationGranularity;
    auto roundUp = [&](size_t n, size_t to) {
      return (n + to - 1) / to * to;
    };
    const size_t chunkBytes = roundUp(std::max((size_t)4 << 20, total / 64), granularity);
    const size_t minimum = roundUp(minbytes, granularity);
    size_t available = free > buffer ? (free - buffer) / granularity * granularity : 0;
    if (mapLimit != 0) {
      const size_t mapped = nextMapBase - reservedBase;
      available = std::min(available, mapLimit > mapped ? (mapLimit - mapped) / granularity * granularity : 0);
    }
    if (available < minimum) {
      log.verbose(
          "Moodist CUDA Allocator: cannot map %d bytes: the device has %d of %d bytes free\n", minimum, free, total);
      return {};
    }
    size_t bytes = std::min(roundUp(minbytes, chunkBytes), available);

    CUmemAllocationProp prop = allocationProp();
    CUmemGenericAllocationHandle handle;
    CUresult err = cuMemCreate(&handle, bytes, &prop, 0);
    if (err != CUDA_SUCCESS && bytes > minimum) {
      bytes = minimum;
      err = cuMemCreate(&handle, bytes, &prop, 0);
    }
    if (err != CUDA_SUCCESS) {
      const char* str = "unknown cuda error";
      cuGetErrorString(err, &str);
      if (err == CUDA_ERROR_OUT_OF_MEMORY) {
        log.verbose("Moodist CUDA Allocator: failed to create %d bytes: %s\n", bytes, str);
      } else {
        log.error("Moodist CUDA Allocator: failed to create %d bytes: %s\n", bytes, str);
      }
      return {};
    }

    uintptr_t address = nextMapBase;
    if (address + bytes > reservedBase + reservedSize) {
      log.error("CUDA Allocator: mapping %d bytes at %#x would exceed 1TB VA reservation\n", bytes, address);
      cuMemRelease(handle);
      return {};
    }
    CHECK_CU(cuMemMap(address, bytes, 0, handle, 0));

    CUmemAccessDesc accessDesc;
    std::memset(&accessDesc, 0, sizeof(accessDesc));
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = deviceIndex;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CHECK_CU(cuMemSetAccess(address, bytes, &accessDesc, 1));

    cuMemHandles.emplace_back(bytes, handle);
    nextMapBase += bytes;
    for (auto& [fn, ctx] : chunkListeners) {
      fn(ctx, handle, address - reservedBase, bytes);
    }

    add_span(mappedRegions, address, address + bytes);
    addStat(stats.reservedBytes, (int64_t)bytes);
    addStat(stats.segments, 1);
    ++stats.numMaps;

    log.info("Moodist CUDA Allocator mapped %d bytes at %#x (%d mapped in all; the device had %d of %d bytes free)\n",
        bytes, address, nextMapBase - reservedBase, free, total);
    if (reportOnMap) {
      log.info("%s", freeMemoryReport(minbytes));
    }

    return Span{address, address + bytes};
  }

  // The parts of [address, address + bytes) in the cuMemCreate chunks: fn(handle, offset in the chunk, size, offset in
  // the range)
  template<typename F>
  void forEachChunkPiece(uintptr_t address, size_t bytes, F&& fn) {
    uintptr_t chunkBase = reservedBase;
    const uintptr_t end = address + bytes;
    for (auto& [size, handle] : cuMemHandles) {
      uintptr_t chunkEnd = chunkBase + size;
      uintptr_t begin = std::max(address, chunkBase);
      uintptr_t stop = std::min(end, chunkEnd);
      if (begin < stop) {
        fn(handle, begin - chunkBase, stop - begin, begin - address);
      }
      chunkBase = chunkEnd;
    }
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Statistics

  static void addStat(CudaAllocatorStat& s, int64_t n) {
    s.current += n;
    if (n > 0) {
      s.allocated += n;
      s.peak = std::max(s.peak, s.current);
    } else {
      s.freed -= n;
    }
  }

  void noteAllocation(size_t bytes, size_t requested, int sign = 1) {
    const size_t pool = bytes <= ((size_t)1 << 20) ? 1 : 2;
    for (size_t i : {(size_t)0, pool}) {
      addStat(stats.allocations[i], sign);
      addStat(stats.allocatedBytes[i], sign * (int64_t)bytes);
      addStat(stats.requestedBytes[i], sign * (int64_t)requested);
    }
  }

  void noteFree(size_t bytes, size_t requested) {
    noteAllocation(bytes, requested, -1);
  }

  // The free bytes and the largest free region of the main pool and the private pools
  std::pair<size_t, size_t> freeMemory() {
    size_t total = 0;
    size_t largest = 0;
    auto count = [&](Regions& regions) {
      for (auto& r : regions.map) {
        total += r.span.end - r.span.begin;
      }
      if (!regions.sizes.empty()) {
        largest = std::max(largest, std::prev(regions.sizes.end())->bytes);
      }
    };
    mainPool.forEachList(count);
    for (PrivatePool* pp : privatePools) {
      pp->pool.forEachList(count);
    }
    for (PrivatePool* pp : releasedPools) {
      pp->pool.forEachList(count);
    }
    return {total, largest};
  }

  // A line of the allocation trace: op (a: allocation, f: free), what (t: PyTorch's, i: moodist's own, c: of a
  // communication pool, s: a communication pool's slab), address, bytes (as allocated), stream, microseconds
  void trace(char op, char what, uintptr_t address, size_t bytes, CUstream stream) {
    if (traceFile) {
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - traceStart);
      std::fprintf(traceFile, "%c %c %#lx %lu %#lx %ld\n", op, what, (unsigned long)address, (unsigned long)bytes,
          (unsigned long)(uintptr_t)stream, (long)us.count());
    }
  }

  // Where the free memory is (MOODIST_DEBUG_ALLOCATOR=2: when a chunk is mapped, for a request of bytes)
  std::string freeMemoryReport(size_t bytes) {
    std::string s = fmt::sprintf("Moodist CUDA allocator free memory (map for a request of %d bytes; %d allocated):\n",
        bytes, stats.allocatedBytes[0].current);
    auto list = [&](const char* what, CUstream stream, Regions& regions) {
      if (regions.map.empty()) {
        return;
      }
      size_t total = 0;
      for (auto& r : regions.map) {
        total += r.span.end - r.span.begin;
      }
      s += fmt::sprintf("  %s stream %#x: %d bytes in %d regions, the largest %d\n", what, (uintptr_t)stream, total,
          regions.map.size(), std::prev(regions.sizes.end())->bytes);
    };
    auto pool = [&](const char* what, Pool& p) {
      list(fmt::sprintf("%s%s common list,", what, p.pressure ? " (pressure mode)" : "").c_str(), nullptr, p.common);
      for (auto& sf : p.streams) {
        list(fmt::sprintf("%s list of", what).c_str(), sf->stream.stream, sf->free);
      }
    };
    pool("main pool", mainPool);
    for (PrivatePool* pp : privatePools) {
      pool("private pool", pp->pool);
    }
    return s;
  }

  std::string bytesString(size_t n) {
    return fmt::sprintf("%.2f GiB", n / 1024.0 / 1024.0 / 1024.0);
  }

  std::string outOfMemoryMessage(size_t bytes) {
    size_t mapped = 0;
    {
      std::lock_guard mrl(mappedRegionsMutex);
      for (auto& v : mappedRegions) {
        mapped += v.end - v.begin;
      }
    }
    auto [freeBytes, largest] = freeMemory();
    size_t deviceFree = 0;
    size_t deviceTotal = 0;
    cuMemGetInfo(&deviceFree, &deviceTotal);
    return fmt::sprintf(
        "CUDA out of memory (moodist CUDA allocator): tried to allocate %s (%d bytes). The device has %s free of %s. "
        "The allocator has mapped %s, of which %s are allocated (in the main pool and memory pools; %d live "
        "allocations) and %s are free (the largest free region is %s).",
        bytesString(bytes), bytes, bytesString(deviceFree), bytesString(deviceTotal), bytesString(mapped),
        bytesString(stats.allocatedBytes[0].current), stats.allocations[0].current, bytesString(freeBytes),
        bytesString(largest));
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Entry points (with the mutex held)

  // An allocation of bytes (requested: the size asked for) for use on stream: from the private pool that allocations
  // on stream are routed to, if any, else from the main pool. Throws CudaOutOfMemoryError if out of memory.
  struct Allocation {
    uintptr_t ptr;
    PrivatePool* privatePool;
    uint64_t streamId;
  };
  Allocation allocate(size_t bytes, size_t requested, CUstream stream) {
    CHECK(bytes > 0);
    maintenance();
    AllocContext ctx = context(stream);
    PrivatePool* pp = scopes.empty() ? nullptr : route(stream);
    uintptr_t r;
    if (pp) {
      if (ctx.captureId != 0 && ctx.graph && !attachGraph(*pp, CaptureInfo{ctx.captureId, ctx.graph})) {
        throw std::runtime_error("Moodist CUDA allocator: could not attach a private pool to a CUDA graph capture");
      }
      r = allocateIn(PoolKind::private_, pp, bytes, ctx);
    } else {
      r = allocateIn(PoolKind::main, nullptr, bytes, ctx);
    }
    if (r == 0) {
      ++stats.numOoms;
      throw CudaOutOfMemoryError(outOfMemoryMessage(bytes));
    }
    if (pp) {
      ++pp->liveAllocations;
    }
    noteAllocation(bytes, requested);
    if (validateEveryCall) {
      validate();
    }
    return {r, pp, ctx.streamId};
  }

  // ---------------------------------------------------------------------------------------------------------------
  // Debugging (MOODIST_DEBUG_ALLOCATOR=1: after every allocation and free)

  void validateRegions(Regions& regions, Vector<Span>& all, size_t& eagerCount, size_t& capturedCount) {
    CHECK(regions.map.size() == regions.sizes.size());
    uintptr_t prevEnd = 0;
    for (auto i = regions.map.begin(); i != regions.map.end(); ++i) {
      auto& r = *i;
      CHECK(r.span.end > r.span.begin);
      CHECK(r.span.begin > prevEnd || prevEnd == 0);
      prevEnd = r.span.end;
      CHECK(r.sizeIterator->regionIterator == i);
      CHECK(r.sizeIterator->bytes == r.span.end - r.span.begin);
      uintptr_t prevBegin = r.span.begin;
      for (auto& e : r.events) {
        CHECK(e.span.end > e.span.begin);
        CHECK(e.span.begin >= prevBegin);
        CHECK(e.span.begin >= r.span.begin && e.span.end <= r.span.end);
        prevBegin = e.span.begin;
        if (e.captureId != 0) {
          CHECK(e.fifoLink.prev == nullptr);
          ++capturedCount;
        } else {
          CHECK(e.fifoLink.prev != nullptr);
          ++eagerCount;
        }
      }
      all.push_back(r.span);
    }
  }

  void checkDisjoint(Vector<Span>& all) {
    std::sort(all.begin(), all.end(), [](const Span& a, const Span& b) {
      return a.begin < b.begin;
    });
    for (size_t i = 1; i < all.size(); ++i) {
      CHECK(all[i].begin >= all[i - 1].end);
    }
  }

  void validate() {
    size_t eagerCount = 0;
    size_t capturedCount = 0;
    Vector<Span> all;
    mainPool.forEachList([&](Regions& regions) {
      validateRegions(regions, all, eagerCount, capturedCount);
    });
    for (PrivatePool* pp : privatePools) {
      pp->pool.forEachList([&](Regions& regions) {
        validateRegions(regions, all, eagerCount, capturedCount);
      });
    }
    for (PrivatePool* pp : releasedPools) {
      pp->pool.forEachList([&](Regions& regions) {
        validateRegions(regions, all, eagerCount, capturedCount);
      });
    }
    checkDisjoint(all);
    for (CommPool* cp : commPools) {
      all.clear();
      cp->pool.forEachList([&](Regions& regions) {
        validateRegions(regions, all, eagerCount, capturedCount);
      });
      checkDisjoint(all);
    }
    size_t fifoCount = 0;
    for (auto& se : eagerEvents) {
      for (auto& e : se->events) {
        CHECK(e.eventStreamId == se->stream.id && e.captureId == 0);
        ++fifoCount;
      }
    }
    CHECK(fifoCount == eagerCount);
    CHECK(capturedCount == numCapturedEvents);
    for (auto& sf : mainPool.streams) {
      CHECK(!mainPool.pressure || sf->free.map.empty());
    }
  }
};

// Cleanup context structure - stored per allocation
struct CudaAllocCleanupCtx {
  CudaAllocatorImpl* impl;
  uintptr_t ptr; // As allocated (with bytes, what is deallocated; in a communication pool, the logical offset)
  size_t bytes;
  CUstream allocStream;         // Stream used for allocation
  uintptr_t userPtr;            // As returned (see cudaAllocatorImplAllocate): the key of streamUses and freeCallbacks
  CommPool* commPool = nullptr; // The communication pool the memory is from, if any
  PrivatePool* privatePool = nullptr; // The private pool the memory is from, if any
  size_t requested = 0;               // The size asked for (for the statistics)
  uint64_t allocStreamId = 0;         // allocStream's id (see StreamRef)
};

// Global allocator impl pointer for allocator::owns/mappedRegion
CudaAllocatorImpl* globalCudaAllocatorImpl = nullptr;

namespace {
std::once_flag initFlag;
void init() {
  globalCudaAllocatorImpl = internalNew<CudaAllocatorImpl>();
}
CudaAllocatorImpl* getImpl() {
  std::call_once(initFlag, init);
  return globalCudaAllocatorImpl;
}
} // namespace

// CoreApi function implementations (the allocator is global; see allocator_api.h)
CudaAllocatorImpl* createCudaAllocatorImpl() {
  return nullptr;
}

void setCudaAllocatorImpl(CudaAllocatorImpl* impl) {}

namespace {
// The allocator, its device and context set up (on the first allocation) and checked. Called with its mutex held.
void prepareAllocation(CudaAllocatorImpl* impl, int deviceIndex) {
  if (!loadCuda()) {
    throw std::runtime_error("CUDA driver not available");
  }

  // Initialize CUDA context on first allocation
  if (impl->deviceIndex == -1) {
    if (deviceIndex == -1) {
      // Auto-detect from current context
      CUdevice dev;
      CHECK_CU(cuCtxGetDevice(&dev));
      int devIndex;
      CHECK_CU(cuDeviceGet(&devIndex, dev));
      deviceIndex = devIndex;
    }
    impl->deviceIndex = deviceIndex;

    cuCtxGetCurrent(&impl->cuContext);
    if (!impl->cuContext) {
      CHECK_CU(cuInit(0));
      CHECK_CU(cuDeviceGet(&impl->cuDevice, deviceIndex));
      CHECK_CU(cuDevicePrimaryCtxRetain(&impl->cuContext, impl->cuDevice));
      CHECK_CU(cuCtxSetCurrent(impl->cuContext));
    } else {
      CHECK_CU(cuDeviceGet(&impl->cuDevice, deviceIndex));
    }
  } else {
    // Verify we're still on the same device
    if (deviceIndex != -1 && deviceIndex != impl->deviceIndex) {
      throw std::runtime_error("Moodist CUDA Allocator can only be used on one device. It was initialized on device " +
                               std::to_string(impl->deviceIndex) + ", but the requested allocation is on device " +
                               std::to_string(deviceIndex) + ". This is not supported.");
    }
    // Ensure context is set
    CUcontext currentContext = nullptr;
    cuCtxGetCurrent(&currentContext);
    if (!currentContext) {
      CHECK_CU(cuCtxSetCurrent(impl->cuContext));
    } else if (currentContext != impl->cuContext) {
      throw std::runtime_error("Moodist CUDA Allocator CUDA context changed. The Moodist CUDA "
                               "Allocator can only be used on one CUDA context.");
    }
  }
}
} // namespace

namespace {
CudaAllocation allocateImpl(size_t bytes, CUstream stream, int deviceIndex, bool fromWrapper);
} // namespace

CudaAllocation cudaAllocatorImplAllocate(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex) {
  return allocateImpl(bytes, stream, deviceIndex, false);
}

namespace {
// (fromWrapper: PyTorch's allocation, for the trace)
CudaAllocation allocateImpl(size_t bytes, CUstream stream, int deviceIndex, bool fromWrapper) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  prepareAllocation(impl, deviceIndex);

  // Align allocation. Allocations are 256-byte aligned; those of at least 64 KiB start at a 4 KiB boundary (a little
  // more is allocated, and the returned pointer is the first boundary in it). RDMA into memory that starts in the
  // middle of a page is much slower (GB300, all-gather of 256 MiB per rank between two nodes: 2.9 ms at a page
  // boundary, 4.4-4.8 ms at a 256-byte or 1 KiB offset), and without this, any small allocation shifts the large ones
  // after it.
  constexpr size_t alignment = 0x100;
  constexpr size_t pageAlignment = 0x1000;
  constexpr size_t pageAlignedFrom = 0x10000;
  size_t alignedBytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);
  const size_t padding = bytes >= pageAlignedFrom ? pageAlignment - alignment : 0;

  auto [ptr, privatePool, allocStreamId] = impl->allocate(alignedBytes + padding, bytes, stream);
  uintptr_t userPtr = padding ? (ptr + pageAlignment - 1) & ~(uintptr_t)(pageAlignment - 1) : ptr;
  CHECK(userPtr + alignedBytes <= ptr + alignedBytes + padding);

  auto* ctx = internalNew<CudaAllocCleanupCtx>(
      impl, ptr, alignedBytes + padding, stream, userPtr, nullptr, privatePool, bytes, allocStreamId);
  impl->trace('a', fromWrapper ? 't' : 'i', ptr, alignedBytes + padding, stream);
  return CudaAllocation{userPtr, ctx, impl->deviceIndex};
}
} // namespace

void cudaAllocatorImplFree(void* cleanupCtx, CUstream) {
  if (!cleanupCtx) {
    return;
  }

  auto* ctx = static_cast<CudaAllocCleanupCtx*>(cleanupCtx);
  CudaAllocatorImpl* impl = ctx->impl;

  std::lock_guard l(impl->mutex);

  // The streams that used the memory: the allocation's, then any recorded
  struct StreamArrayView {
    StreamRef* data;
    size_t count;
    StreamRef* begin() const {
      return data;
    }
    StreamRef* end() const {
      return data + count;
    }
  };
  impl->trace('f', ctx->commPool ? 'c' : '-', ctx->ptr, ctx->bytes, ctx->allocStream);
  auto deallocate = [&](const auto& streams) {
    if (ctx->commPool) {
      impl->freeOrDefer(PoolKind::comm, ctx->commPool, Span{ctx->ptr, ctx->ptr + ctx->bytes}, streams);
    } else {
      impl->noteFree(ctx->bytes, ctx->requested);
      if (ctx->privatePool) {
        impl->freeOrDefer(PoolKind::private_, ctx->privatePool, Span{ctx->ptr, ctx->ptr + ctx->bytes}, streams);
      } else {
        impl->freeOrDefer(PoolKind::main, nullptr, Span{ctx->ptr, ctx->ptr + ctx->bytes}, streams);
      }
    }
  };

  // Invoke any registered free callbacks for this address
  auto cbIt = impl->freeCallbacks.find(ctx->userPtr);
  if (cbIt != impl->freeCallbacks.end()) {
    FunctionPointer head = cbIt->second;
    while (head) {
      FunctionPointer next = head->next;
      Function<void()>{head}();
      head = next;
    }
    impl->freeCallbacks.erase(cbIt);
  }

  auto it = impl->streamUses.find(ctx->userPtr);
  if (it != impl->streamUses.end()) {
    auto& list = it->second;
    list->insert(list->begin(), StreamRef{ctx->allocStream, ctx->allocStreamId});
    deallocate(*list);
    list->clear();
    impl->streamListFree.push_back(std::move(list));
    impl->streamUses.erase(it);
  } else {
    StreamRef streams[1] = {StreamRef{ctx->allocStream, ctx->allocStreamId}};
    deallocate(StreamArrayView{streams, 1});
  }

  internalDelete(ctx);
  if (impl->validateEveryCall) {
    impl->validate();
  }
}

CommPool* allocatorCommPoolCreate() {
  CudaAllocatorImpl* impl = getImpl();
  CommPool* pool = internalNew<CommPool>();
  std::lock_guard l(impl->mutex);
  impl->commPools.push_back(pool);
  return pool;
}

CudaAllocation allocatorCommPoolAllocate(CommPool* pool, size_t bytes, CUstream stream, int deviceIndex) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  prepareAllocation(impl, deviceIndex);
  impl->maintenance();
  auto ctx = impl->context(stream);
  if (ctx.captureId != 0) {
    throw std::runtime_error("Moodist communication pools do not support CUDA graph capture yet: allocation on a "
                             "capturing stream");
  }
  constexpr size_t alignment = 0x1000;
  const size_t alignedBytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);
  uintptr_t logical = impl->allocateIn(PoolKind::comm, pool, alignedBytes, ctx);
  if (logical == 0) {
    ++impl->stats.numOoms;
    throw CudaOutOfMemoryError(impl->outOfMemoryMessage(alignedBytes));
  }
  uintptr_t userPtr = 0;
  for (auto& s : pool->slabs) {
    if (logical >= s.logical && logical + alignedBytes <= s.logical + s.bytes) {
      userPtr = s.local + (logical - s.logical);
      break;
    }
  }
  CHECK(userPtr != 0);
  auto* ctx2 =
      internalNew<CudaAllocCleanupCtx>(impl, logical, alignedBytes, stream, userPtr, pool, nullptr, 0, ctx.streamId);
  impl->trace('a', 'c', logical, alignedBytes, stream);
  if (impl->validateEveryCall) {
    impl->validate();
  }
  return CudaAllocation{userPtr, ctx2, impl->deviceIndex};
}

void allocatorCommPoolAddSlabListener(CommPool* pool,
    bool (*fn)(void* ctx, size_t logicalOffset, unsigned long long handle, size_t memOffset, size_t size), void* ctx) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  for (auto& slab : pool->slabs) {
    impl->forEachChunkPiece(
        slab.local, slab.bytes, [&](CUmemGenericAllocationHandle handle, size_t memOffset, size_t size, size_t offset) {
          slab.bound &= fn(ctx, slab.logical + offset, handle, memOffset, size);
        });
  }
  pool->listeners.emplace_back(fn, ctx);
}

namespace {
int64_t commPoolOffset(CommPool* pool, uintptr_t address, size_t bytes) {
  for (auto& s : pool->slabs) {
    if (address >= s.local && address + bytes <= s.local + s.bytes) {
      return s.bound ? (int64_t)(s.logical + (address - s.local)) : -1;
    }
  }
  return -1;
}
} // namespace

int64_t allocatorCommPoolOffset(CommPool* pool, uintptr_t address, size_t bytes) {
  if (!globalCudaAllocatorImpl) {
    return -1;
  }
  std::lock_guard l(globalCudaAllocatorImpl->mutex);
  return commPoolOffset(pool, address, bytes);
}

int64_t allocatorAnyCommPoolOffset(uintptr_t address, size_t bytes) {
  if (!globalCudaAllocatorImpl) {
    return -1;
  }
  std::lock_guard l(globalCudaAllocatorImpl->mutex);
  for (CommPool* pool : globalCudaAllocatorImpl->commPools) {
    int64_t r = commPoolOffset(pool, address, bytes);
    if (r >= 0) {
      return r;
    }
  }
  return -1;
}

void cudaAllocatorImplRecordStream(CudaAllocatorImpl* impl, uintptr_t ptr, CUstream stream) {
  impl = globalCudaAllocatorImpl;
  if (!impl) {
    return;
  }
  std::lock_guard l(impl->mutex);

  auto& list = impl->streamUses[ptr];
  if (!list) {
    if (!impl->streamListFree.empty()) {
      list = std::move(impl->streamListFree.back());
      impl->streamListFree.pop_back();
    } else {
      list = std::make_unique<Vector<StreamRef>>();
    }
  }
  list->push_back(StreamRef{stream, streamId(stream)});
}

int cudaAllocatorImplGetDevice(CudaAllocatorImpl* impl) {
  impl = globalCudaAllocatorImpl;
  return impl ? impl->deviceIndex : -1;
}

void* allocatorAddFreeCallback(CudaAllocatorImpl* impl, uintptr_t baseAddress, void* callbackFn) {
  impl = getImpl();
  std::lock_guard l(impl->mutex);
  FunctionPointer fp = static_cast<FunctionPointer>(callbackFn);
  auto& head = impl->freeCallbacks[baseAddress];
  fp->next = head;
  head = fp;
  return fp;
}

void allocatorRemoveFreeCallback(CudaAllocatorImpl* impl, uintptr_t baseAddress, void* handle) {
  impl = getImpl();
  std::lock_guard l(impl->mutex);
  auto** head = &impl->freeCallbacks[baseAddress];
  while (*head) {
    if (*head == handle) {
      FunctionPointer f = *head;
      *head = (*head)->next;
      Function<void()>{f}; // Destroy without calling
      return;
    }
    head = &(*head)->next;
  }
  throw std::runtime_error("allocatorRemoveFreeCallback: no such handle");
}

bool allocatorOwns(uintptr_t address) {
  if (!globalCudaAllocatorImpl) {
    return false;
  }
  std::lock_guard mrl(globalCudaAllocatorImpl->mappedRegionsMutex);
  for (auto& v : globalCudaAllocatorImpl->mappedRegions) {
    if (address >= v.begin && address < v.end) {
      return true;
    }
  }
  return false;
}

std::pair<uintptr_t, size_t> allocatorMappedRegion(uintptr_t address) {
  if (!globalCudaAllocatorImpl) {
    return {0, 0};
  }
  std::lock_guard mrl(globalCudaAllocatorImpl->mappedRegionsMutex);
  for (auto& v : globalCudaAllocatorImpl->mappedRegions) {
    if (address >= v.begin && address < v.end) {
      return {v.begin, v.end - v.begin};
    }
  }
  return {0, 0};
}

void cudaAllocatorImplAllocateApi(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex,
    uintptr_t* outPtr, void** outCleanupCtx, int* outDeviceIndex) {
  CudaAllocation alloc = allocateImpl(bytes, stream, deviceIndex, true);
  *outPtr = alloc.ptr;
  *outCleanupCtx = alloc.cleanupCtx;
  *outDeviceIndex = alloc.deviceIndex;
}

void allocatorMappedRegionApi(uintptr_t address, uintptr_t* outBase, size_t* outSize) {
  auto result = allocatorMappedRegion(address);
  *outBase = result.first;
  *outSize = result.second;
}

uintptr_t allocatorGetReservedBase() {
  if (!globalCudaAllocatorImpl) {
    return 0;
  }
  return globalCudaAllocatorImpl->reservedBase.load(std::memory_order_relaxed);
}

size_t allocatorGetChunkCount() {
  if (!globalCudaAllocatorImpl) {
    return 0;
  }
  std::lock_guard l(globalCudaAllocatorImpl->mutex);
  return globalCudaAllocatorImpl->cuMemHandles.size();
}

bool allocatorGetChunk(size_t index, unsigned long long* outHandle, size_t* outOffset, size_t* outSize) {
  if (!globalCudaAllocatorImpl) {
    return false;
  }
  std::lock_guard l(globalCudaAllocatorImpl->mutex);
  auto& handles = globalCudaAllocatorImpl->cuMemHandles;
  if (index >= handles.size()) {
    return false;
  }
  size_t offset = 0;
  for (size_t i = 0; i < index; ++i) {
    offset += handles[i].first;
  }
  *outHandle = handles[index].second;
  *outOffset = offset;
  *outSize = handles[index].first;
  return true;
}

void allocatorAddChunkListener(
    void (*fn)(void* ctx, unsigned long long handle, size_t offset, size_t size), void* ctx) {
  CHECK(globalCudaAllocatorImpl);
  std::lock_guard l(globalCudaAllocatorImpl->mutex);
  size_t offset = 0;
  for (auto& [size, handle] : globalCudaAllocatorImpl->cuMemHandles) {
    fn(ctx, handle, offset, size);
    offset += size;
  }
  globalCudaAllocatorImpl->chunkListeners.emplace_back(fn, ctx);
}

bool allocatorSupportsFabric() {
  if (!globalCudaAllocatorImpl) {
    return false;
  }
  return globalCudaAllocatorImpl->fabricSupported;
}

void allocatorStreamDestroyed(CUstream stream) {
  CudaAllocatorImpl* impl = globalCudaAllocatorImpl;
  if (!impl) {
    return;
  }
  std::lock_guard l(impl->mutex);
  impl->streamDestroyed(stream);
  if (impl->validateEveryCall) {
    impl->validate();
  }
}

void cudaAllocatorGetStats(CudaAllocatorStats* out) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  *out = impl->stats;
  out->largestFreeBlock = impl->freeMemory().second;
}

void cudaAllocatorSetMemoryLimit(size_t bytes) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  impl->mapLimit = bytes;
}

void cudaAllocatorResetStats(bool peaks, bool accumulated) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  auto& s = impl->stats;
  auto reset = [&](CudaAllocatorStat& stat) {
    if (peaks) {
      stat.peak = stat.current;
    }
    if (accumulated) {
      stat.allocated = 0;
      stat.freed = 0;
    }
  };
  for (size_t i = 0; i != 3; ++i) {
    reset(s.allocations[i]);
    reset(s.allocatedBytes[i]);
    reset(s.requestedBytes[i]);
  }
  reset(s.reservedBytes);
  reset(s.segments);
  if (accumulated) {
    s.numOoms = 0;
    s.numMaps = 0;
  }
}

// Private pools (see the top of the file and allocator_api.h)

void cudaAllocatorBeginAllocateToPool(
    uint64_t id0, uint64_t id1, CudaAllocatorPoolFilter filter, void* filterCtx, void (*destroyFilterCtx)(void*)) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  for (auto& s : impl->scopes) {
    if (s.pool->id == MempoolId{id0, id1}) {
      if (destroyFilterCtx) {
        destroyFilterCtx(filterCtx);
      }
      throw std::runtime_error("Moodist CUDA allocator: beginAllocateToPool: already allocating to this pool");
    }
  }
  impl->maintenance();
  CudaAllocatorImpl::Scope scope;
  scope.pool = impl->createOrIncrefPool(MempoolId{id0, id1});
  scope.filter = filter;
  scope.filterCtx = filterCtx;
  scope.destroyFilterCtx = destroyFilterCtx;
  impl->scopes.push_back(scope);
}

void cudaAllocatorEndAllocateToPool(uint64_t id0, uint64_t id1) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  for (auto i = impl->scopes.begin(); i != impl->scopes.end(); ++i) {
    if (i->pool->id == MempoolId{id0, id1}) {
      CudaAllocatorImpl::Scope scope = *i;
      impl->scopes.erase(i);
      if (scope.destroyFilterCtx) {
        scope.destroyFilterCtx(scope.filterCtx);
      }
      // The events recorded in the pool's capture, if it has ended
      impl->dropCapturedEvents(*scope.pool, false);
      impl->processDeferredFrees();
      impl->maintenance();
      if (impl->validateEveryCall) {
        impl->validate();
      }
      return;
    }
  }
  throw std::runtime_error("Moodist CUDA allocator: endAllocateToPool: not allocating to this pool");
}

void cudaAllocatorReleasePool(uint64_t id0, uint64_t id1) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  PrivatePool* pp = impl->findPool(MempoolId{id0, id1});
  if (!pp) {
    throw std::runtime_error("Moodist CUDA allocator: releasePool: no such pool");
  }
  CHECK(pp->useCount > 0);
  if (--pp->useCount == 0) {
    for (auto& s : impl->scopes) {
      CHECK(s.pool != pp);
    }
    impl->privatePools.erase(std::find(impl->privatePools.begin(), impl->privatePools.end(), pp));
    pp->released = true;
    if (!impl->tryRetire(pp)) {
      impl->releasedPools.push_back(pp);
    }
  }
  if (impl->validateEveryCall) {
    impl->validate();
  }
}

void cudaAllocatorCreateOrIncrefPool(uint64_t id0, uint64_t id1) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  impl->createOrIncrefPool(MempoolId{id0, id1});
}

int cudaAllocatorGetPoolUseCount(uint64_t id0, uint64_t id1) {
  CudaAllocatorImpl* impl = getImpl();
  std::lock_guard l(impl->mutex);
  PrivatePool* pp = impl->findPool(MempoolId{id0, id1});
  return pp ? pp->useCount : 0;
}

// allocator namespace functions that are in core
namespace allocator {

bool owns(uintptr_t address) {
  return allocatorOwns(address);
}

std::pair<uintptr_t, size_t> mappedRegion(uintptr_t address) {
  return allocatorMappedRegion(address);
}

// addFreeCallback, removeFreeCallback, and enableCudaAllocator are in allocator_wrapper.cc

} // namespace allocator

} // namespace moodist
