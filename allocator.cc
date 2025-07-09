// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "allocator.h"
#include "common.h"
#include "function.h"
#include "group.h"
#include "hash_map.h"
#include "intrusive_list.h"
#include "vector.h"

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <type_traits>

namespace moodist {

inline LogLevel memLogLevel = []() {
  const char* c = std::getenv("MOODIST_MEM_LOG_LEVEL");
  LogLevel r = LOG_INFO;
  if (c) {
    std::string s = c;
    for (auto& c : s) {
      c = std::toupper(c);
    }
    if (s == "NONE") {
      r = LOG_NONE;
    } else if (s == "ERROR") {
      r = LOG_ERROR;
    } else if (s == "INFO") {
      r = LOG_INFO;
    } else if (s == "VERBOSE") {
      r = LOG_VERBOSE;
    } else if (s == "DEBUG") {
      r = LOG_DEBUG;
    } else {
      r = (LogLevel)std::atoi(c);
    }
    if (currentLogLevel < LOG_ERROR) {
      r = LOG_ERROR;
    }
  }
  return r;
}();

inline struct MemLog {
  template<typename... Args>
  void error(const char* fmt, Args&&... args) {
    logat(LOG_ERROR, fmt, std::forward<Args>(args)...);
  }
  template<typename... Args>
  void info(const char* fmt, Args&&... args) {
    if (memLogLevel >= LOG_INFO) {
      [[unlikely]];
      logat(LOG_INFO, fmt, std::forward<Args>(args)...);
    }
  }
  template<typename... Args>
  void verbose(const char* fmt, Args&&... args) {
    if (memLogLevel >= LOG_VERBOSE) {
      [[unlikely]];
      logat(LOG_VERBOSE, fmt, std::forward<Args>(args)...);
    }
  }
  template<typename... Args>
  void debug(const char* fmt, Args&&... args) {
    if (memLogLevel >= LOG_DEBUG) {
      [[unlikely]];
      logat(LOG_DEBUG, fmt, std::forward<Args>(args)...);
    }
  }
} memlog;

namespace {

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

void add_span(Vector<Span>& spans, uintptr_t begin, uintptr_t end) {
  CHECK(end > begin);
  // printf("add span %#lx %#lx\n", begin, end);
  // printf("PRE %d spans:\n", spans.size());
  // for (auto& v : spans) {
  //   printf("[%#lx, %#lx)\n", v.begin, v.end);
  // }
  // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
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
          // printf("%d spans:\n", spans.size());
          // for (auto& v : spans) {
          //   printf("[%#lx, %#lx)\n", v.begin, v.end);
          // }
          // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
          return;
        }
        s[a - 1].end = end;
        // printf("%d spans:\n", spans.size());
        // for (auto& v : spans) {
        //   printf("[%#lx, %#lx)\n", v.begin, v.end);
        // }
        // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
        return;
      }
    }
    if (a != e && s[a].begin == end) {
      s[a].begin = begin;
      // printf("%d spans:\n", spans.size());
      // for (auto& v : spans) {
      //   printf("[%#lx, %#lx)\n", v.begin, v.end);
      // }
      // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));
      return;
    }
  }
  CHECK(a <= spans.size());
  spans.insert(spans.begin() + a, Span{begin, end});

  // CHECK(std::is_sorted(spans.begin(), spans.end(), [&](auto& a, auto& b) { return a.begin < b.begin; }));

  // printf("INSERT %d spans:\n", spans.size());
  // for (auto& v : spans) {
  //   printf("[%#lx, %#lx)\n", v.begin, v.end);
  // }
}

// struct MemoryEvent {
//   IntrusiveListLink<MemoryEvent> link;
//   size_t refcount = 0;
//   CUevent event = nullptr;
// };

// struct MemoryEventHandle {
//   MemoryEvent* e = nullptr;
//   Vector<MemoryEvent*>* container = nullptr;
//   MemoryEventHandle() = default;
//   MemoryEventHandle(MemoryEvent* e, Vector<MemoryEvent*>* container) : e(e), container(container) {
//     ++e->refcount;
//   }
//   ~MemoryEventHandle() {
//     decref();
//   }
//   MemoryEventHandle(MemoryEventHandle&& n) {
//     *this = std::move(n);
//   }
//   MemoryEventHandle(const MemoryEventHandle& n) {
//     *this = n;
//   }
//   MemoryEventHandle& operator=(const MemoryEventHandle& n) {
//     if (e) {
//       decref();
//     }
//     e = n.e;
//     container = n.container;
//     if (e) {
//       ++e->refcount;
//     }
//     return *this;
//   }
//   MemoryEventHandle& operator=(MemoryEventHandle&& n) {
//     std::swap(e, n.e);
//     std::swap(container, n.container);
//     return *this;
//   }
//   void decref() {
//     if (e && --e->refcount == 0) {
//       container->push_back(e);
//       e = nullptr;
//       container = nullptr;
//     }
//   }
//   MemoryEvent* operator->() {
//     return e;
//   }
//   MemoryEvent& operator*() {
//     return *e;
//   }
// };

struct EventRegion {
  IntrusiveListLink<EventRegion> link;
  Span span;
  Event event;
  CUstream eventStream = nullptr;
};

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

struct Region {
  mutable Span span;
  mutable IntrusiveList<EventRegion, &EventRegion::link> events;
  mutable std::multiset<RegionSize<std::set<Region, RegionCompare>>, RegionSizeCompare>::iterator sizeIterator;

  Region() = default;
  Region(Span span) : span(span) {}
};

using RegionMap = std::set<Region, RegionCompare>;
using RegionSizeMap = std::multiset<RegionSize<std::set<Region, RegionCompare>>, RegionSizeCompare>;

void checkMap(RegionMap& map, RegionSizeMap& sizeMap) {
  return;
  uintptr_t e = 0;
  // memlog.info("check map -- \n");
  Vector<Span> tmp;
  HashMap<EventRegion*, bool> exists;
  for (auto& v : map) {
    for (auto& v2 : v.events) {
      CHECK(!exists[&v2]);
      exists[&v2] = true;
    }
  }
  for (auto& v : map) {
    // memlog.info("span %#x %#x\n", v.span.begin, v.span.end);
    CHECK(v.span.begin > e);
    CHECK(v.span.end > v.span.begin);
    e = v.span.end;

    CHECK(v.sizeIterator->bytes == v.span.end - v.span.begin);

    tmp.clear();

    CHECK(
        std::is_sorted(v.events.begin(), v.events.end(), [](auto& a, auto& b) { return a.span.begin < b.span.begin; }));
    uintptr_t ee = v.span.begin;
    for (auto& v2 : v.events) {
      // memlog.info(" event %#x %#x\n", v2.span.begin, v2.span.end);
      CHECK(v2.span.end > v2.span.begin);
      CHECK(v2.span.begin >= v.span.begin);
      CHECK(v2.span.end <= v.span.end);
      CHECK(v2.span.begin >= ee);
      ee = v2.span.begin;

      // add_span(tmp, v2.span.begin, v2.span.end);
    }

    // for (auto& v : tmp) {
    //   memlog.info(" tmp %#x %#x\n", v.begin, v.end);
    // }

    // CHECK(tmp.size() == 1);
    // CHECK(tmp[0].begin == v.span.begin && tmp[0].end == v.span.end);
  }
}

struct Regions {
  RegionMap map;
  RegionSizeMap sizes;
};

template<typename T>
struct FreeList {
  PoolAllocator<T> allocator;
  Vector<T*> list;

  struct Handle {
    FreeList* owner = nullptr;
    T* ptr;
    Handle() = default;
    Handle(FreeList* owner, T* ptr) : owner(owner), ptr(ptr) {}
    ~Handle() {
      if (owner && ptr) {
        owner->list.push_back(ptr);
      }
    }
    Handle(Handle&& n) {
      std::swap(owner, n.owner);
      std::swap(ptr, n.ptr);
    }
    Handle& operator=(Handle&& n) {
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

struct CudaAllocatorImpl {
  CudaAllocatorImpl() {}

  // PoolAllocator<MemoryEvent> memoryEventAllocator;
  // Vector<MemoryEvent*> memoryEventFreelist;

  SpinMutex mappedRegionsMutex;

  Vector<Span> mappedRegions;

  Regions freeMemory;
  Regions pendingFreeMemory;

  Vector<RegionMap::node_type> freeRegionNodes;
  Vector<RegionSizeMap::node_type> freeRegionSizeNodes;

  // std::set<Region, RegionCompare> pendingFreeMemory;

  // size_t nBytesMapped = 0;
  std::atomic_size_t reservedSize = 0;
  std::atomic_uintptr_t reservedBase = 0;
  size_t nextMapBase = 0;

  int deviceIndex = -1;

  Vector<std::pair<size_t, CUmemGenericAllocationHandle>> cuMemHandles;

  FreeList<Vector<CUstream>> freeListStreams;
  FreeList<EventRegion> freeListEventRegions;
  Vector<std::tuple<uintptr_t, size_t, FreeList<Vector<CUstream>>::Handle>> pendingDeallocations;

  HashMap<CUstream, std::unique_ptr<Regions>, PointerHash> streamFreeMemory;
  HashMap<CUstream, std::unique_ptr<Regions>, PointerHash>::iterator nextStreamFreeMemory = streamFreeMemory.end();

  HashMap<CUstream, std::unique_ptr<Regions>, PointerHash> streamPendingFreeMemory;

  HashMap<CUstream, bool, PointerHash> tmpDeallocateStreamMap;
  Vector<FreeList<EventRegion>::Handle> tmpDeallocateEventRegions;
  Vector<EventRegion*> tmpEventRegions;

  Regions& getStreamFree(CUstream stream) {
    auto i = streamFreeMemory.find(stream);
    if (i != streamFreeMemory.end()) {
      return *i->second;
    }
    auto& ptr = streamFreeMemory[stream];
    ptr = std::make_unique<Regions>();
    nextStreamFreeMemory = streamFreeMemory.begin();
    return *ptr;
  }

  Regions& getStreamPendingFree(CUstream stream) {
    auto i = streamPendingFreeMemory.find(stream);
    if (i != streamPendingFreeMemory.end()) {
      return *i->second;
    }
    auto& ptr = streamPendingFreeMemory[stream];
    ptr = std::make_unique<Regions>();
    return *ptr;
  }

  void updateSize(Regions& regions, RegionMap::iterator i) {
    auto sizeNode = regions.sizes.extract(i->sizeIterator);
    CHECK(sizeNode.value().regionIterator == i);
    sizeNode.value().bytes = i->span.end - i->span.begin;
    i->sizeIterator = regions.sizes.insert(std::move(sizeNode));
  }

  // template<typename EventRegionList = std::array<EventRegion*, 0>>
  template<typename EventRegionList>
  std::set<Region, RegionCompare>::iterator
  insertRegion(Regions& regions, Span span, const EventRegionList& eventRegionList) {
    auto& map = regions.map;
    auto& sizeMap = regions.sizes;
    auto i = map.upper_bound(span);

    checkMap(map, sizeMap);

    if (i != map.begin()) {
      auto pi = std::prev(i);
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

          // CHECK(std::is_sorted(
          //     pi->events.begin(), pi->events.end(), [](auto& a, auto& b) { return a.span.begin < b.span.begin; }));

          CHECK(i->events.empty());
          freeRegionSizeNodes.push_back(sizeMap.extract(i->sizeIterator));
          freeRegionNodes.push_back(map.extract(i));
          checkMap(map, sizeMap);
          return pi;
        }
        pi->span.end = span.end;
        updateSize(regions, pi);

        for (auto& v : eventRegionList) {
          pi->events.push_back(*v);
        }

        checkMap(map, sizeMap);
        return pi;
      }
    }
    if (i != map.end() && i->span.begin == span.end) {
      i->span.begin = span.begin;
      updateSize(regions, i);
      auto at = i->events.begin();
      for (auto& v : eventRegionList) {
        i->events.insert(at, *v);
      }
      // CHECK(std::is_sorted(
      //     i->events.begin(), i->events.end(), [](auto& a, auto& b) { return a.span.begin < b.span.begin; }));
      checkMap(map, sizeMap);
      return i;
    }
    checkMap(map, sizeMap);
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
      // memlog.info("v event span %#x %#x\n", v->span.begin, v->span.end);
      i->events.push_back(*v);
    }
    // CHECK(std::is_sorted(
    //     i->events.begin(), i->events.end(), [](auto& a, auto& b) { return a.span.begin < b.span.begin; }));
    if (freeRegionSizeNodes.empty()) {
      i->sizeIterator = sizeMap.emplace(span.end - span.begin, i);
    } else {
      auto node = std::move(freeRegionSizeNodes.back());
      freeRegionSizeNodes.pop_back();
      node.value().bytes = span.end - span.begin;
      node.value().regionIterator = i;
      i->sizeIterator = sizeMap.insert(std::move(node));
    }
    // memlog.info(
    //     "added span %#x %#x, size %d, i->sizeIterator bytes is %d\n", i->span.begin, i->span.end,
    //     i->span.end - i->span.begin, i->sizeIterator->bytes);
    // memlog.info(
    //     "added span %#x %#x, size %d, i->sizeIterator bytes is %d\n", span.begin, span.end, span.end - span.begin,
    //     i->sizeIterator->bytes);
    checkMap(map, sizeMap);
    return i;
  }

  bool mapMoreMemory(size_t minbytes) {
    // if (reservedBase == 0) {
    //   size_t free = 0;
    //   size_t total = 0;
    //   CHECK_CU(cuMemGetInfo(&free, &total));
    //   memlog.info("Moodist CUDA Allocator initializing. Device has %d free, %d total bytes of memory.\n", free,
    //   total);

    //   constexpr size_t alignment = (size_t)1024 * 1024 * 1024 * 1024;
    //   size_t reserveSize = (total + alignment - 1) / alignment * alignment;

    //   memlog.info("Moodist CUDA Allocator reserving %d bytes\n", reserveSize);

    //   CUdeviceptr base = 0;
    //   CHECK_CU(cuMemAddressReserve(&base, reserveSize, alignment, 0, 0));
    //   reservedBase = base;
    //   reservedSize = reserveSize;
    //   nextMapBase = base;

    //   deviceIndex = c10::cuda::current_device();
    // }

    std::lock_guard mrl(mappedRegionsMutex);

    size_t free = 0;
    size_t total = 0;
    CHECK_CU(cuMemGetInfo(&free, &total));
    if (deviceIndex == -1) {
      deviceIndex = c10::cuda::current_device();
      CHECK(deviceIndex != -1);
    }
    CHECK(c10::cuda::current_device() == deviceIndex);

    size_t bytes = free;
    constexpr size_t buffer = (size_t)1024 * 1024 * 512;
    size_t safebytes = free > buffer ? free - buffer : 0;
    if (safebytes >= minbytes) {
      bytes = safebytes;
    }
    if (mappedRegions.empty() && free / 2 > minbytes) {
      bytes = free / 2;
    }
    memlog.info(
        "Moodist CUDA Allocator attempting to map %d bytes of memory. Device has %d free, %d total bytes of memory.\n",
        bytes, free, total);
    if (bytes < minbytes) {
      return false;
    }
    // // bytes = std::min(bytes, std::max(minbytes, (size_t)1024 * 1024 * 1024 * 4));
    // bytes = std::min(bytes, std::max(minbytes, (size_t)1024 * 1024 * 512));

    // CUmemGenericAllocationHandle handle;
    // CUmemAllocationProp prop;
    // std::memset(&prop, 0, sizeof(prop));
    // prop.location.id = deviceIndex;
    // prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    // prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    // prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    // prop.allocFlags.gpuDirectRDMACapable = 1;

    // constexpr size_t alignment = (size_t)1024 * 1024 * 128;
    // bytes = (bytes + alignment - 1) / alignment * alignment;

    // CHECK_CU(cuMemCreate(&handle, bytes, &prop, 0));

    // cuMemHandles.emplace_back(bytes, handle);

    // uintptr_t address = nextMapBase;
    // nextMapBase += bytes;

    // memlog.info("mem map %#x %#x\n", address, bytes);

    // CHECK_CU(cuMemMap(address, bytes, 0, handle, 0));

    // int ndevices = 0;
    // CHECK_CU(cuDeviceGetCount(&ndevices));

    // log.info("device count is %d\n", ndevices);

    // //CHECK(false);

    // for (size_t i = 0; i != ndevices; ++i) {
    //   std::array<CUmemAccessDesc, 1> desc;
    //   std::memset(desc.data(), 0, sizeof(CUmemAccessDesc) * desc.size());
    //   desc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    //   desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    //   desc[0].location.id = i;
    //   CHECK_CU(cuMemSetAccess(address, bytes, desc.data(), 1));
    //   log.info("access ok for device %d\n", i);
    // }

    CUdeviceptr ptr;
    auto err = cuMemAlloc(&ptr, bytes);
    while (err != CUDA_SUCCESS) {
      if (bytes > minbytes + 1024 * 1024) {
        bytes -= 1024 * 1024;
      } else {
        bytes = minbytes;
      }
      err = cuMemAlloc(&ptr, bytes);
      if (bytes <= minbytes) {
        break;
      }
    }
    if (err != CUDA_SUCCESS) {
      const char* str = "unknown cuda error";
      cuGetErrorString(err, &str);
      memlog.error("CUDA Allocator failed to map %d bytes; %s\n", bytes, str);
      return false;
    }
    uintptr_t address = ptr;

    memlog.info("Moodist successfully mapped %d bytes at %#x\n", bytes, address);

    add_span(mappedRegions, address, address + bytes);

    auto h = freeListEventRegions.pop();
    h->event = Event::create();
    h->span = Span{address, address + bytes};
    h->eventStream = c10::cuda::getCurrentCUDAStream();
    CHECK_CU(cuEventRecord(h->event, c10::cuda::getCurrentCUDAStream()));
    insertRegion(freeMemory, Span{address, address + bytes}, std::array<EventRegion*, 1>{&*h});
    h.release();

    return true;
  }

  uintptr_t allocateFrom(Regions& regions, size_t bytes, CUstream stream) {
    checkMap(regions.map, regions.sizes);
    auto it = regions.sizes.lower_bound(bytes);
    // auto it = regions.sizes.empty() ? regions.sizes.end() : std::prev(regions.sizes.end());
    // if (it != regions.sizes.end()) {
    //   if (it->bytes < bytes) {
    //     it = regions.sizes.end();
    //   }
    // }
    // if (bytes < 1024 * 1024 * 1024) {
    //   it = regions.sizes.lower_bound(bytes);
    // }
    // if (it != regions.sizes.end()) {
    //   if (it->bytes > bytes + bytes / 2 && it->bytes < bytes * 2) {
    //     auto it2 = regions.sizes.lower_bound(bytes * 2);
    //     if (it2 != regions.sizes.end()) {
    //       it = it2;
    //     }
    //   }
    // }
    if (it != regions.sizes.end()) {
      CHECK(it->bytes >= bytes);
      auto r = it->regionIterator;
      CHECK(r->span.end - r->span.begin == it->bytes);
      uintptr_t ptr = r->span.begin;
      uintptr_t ptrEnd = ptr + bytes;
      for (auto i = r->events.begin(); i != r->events.end();) {
        if (i->span.begin >= ptrEnd) {
          break;
        }
        //        memlog.debug("wait for event %#x  %#x %#x!\n", (uintptr_t)(CUevent)i->event, i->span.begin,
        //        i->span.end);

        if (i->eventStream != stream) {
          CHECK_CU(cuStreamWaitEvent(stream, i->event, CU_EVENT_WAIT_DEFAULT));
        }

        if (i->span.end > ptrEnd) {
          i->span.begin = ptrEnd;
          ++i;
        } else {
          freeListEventRegions.push(&*i);
          i = r->events.erase(i);
        }
      }
      if (it->bytes == bytes) {
        CHECK(r->events.empty());
        freeRegionNodes.push_back(regions.map.extract(r));
        freeRegionSizeNodes.push_back(regions.sizes.extract(it));
      } else {
        r->span.begin += bytes;
        updateSize(regions, r);
      }

      // memlog.debug("span %#x %#x\n", r->span.begin, r->span.end);
      // for (auto& v : r->events) {
      //   memlog.debug("event %#x %#x\n", v.span.begin, v.span.end);
      // }

      checkMap(regions.map, regions.sizes);
      // memlog.debug("ok, returning %#x\n", ptr);
      return ptr;
    }
    return 0;
  }

  uintptr_t allocate(size_t bytes, CUstream stream) {
    memlog.debug("allocate %d\n", bytes);
    if (!pendingDeallocations.empty()) {
      // memlog.info("waa %d pending deallocations\n", pendingDeallocations.size());
      auto tmp = std::move(pendingDeallocations);
      for (auto& v : tmp) {
        deallocate<true>(std::get<0>(v), std::get<1>(v), *std::get<2>(v));
      }
    }
    CHECK(bytes > 0);

    {
      uintptr_t r = allocateFrom(getStreamFree(stream), bytes, stream);
      if (r) {
        return r;
      }
    }

    if (freeSomePending(getStreamPendingFree(stream))) {
      return allocate(bytes, stream);
    }

    uintptr_t r = allocateFrom(freeMemory, bytes, stream);
    if (r) {
      return r;
    }

    if (freeSomePending(getStreamPendingFree(stream)) || freePending(getStreamPendingFree(stream))) {
      return allocate(bytes, stream);
    }

    // memlog.info("Prefetcher memory exhausted, moving pending\n");
    if (!pendingFreeMemory.map.empty()) {
      freePending(pendingFreeMemory);
      return allocate(bytes, stream);
    }
    if (memLogLevel >= LOG_VERBOSE) {
      memlog.verbose(
          "Memory exhausted during allocation of %s. Free memory:\n%s\n", this->bytes(bytes), debugFreeMemory());
    }
    for (size_t i = 0; i != streamFreeMemory.size(); ++i) {
      CHECK(nextStreamFreeMemory != streamFreeMemory.end());
      uintptr_t r = allocateFrom(*nextStreamFreeMemory->second, bytes, stream);
      ++nextStreamFreeMemory;
      if (nextStreamFreeMemory == streamFreeMemory.end()) {
        nextStreamFreeMemory = streamFreeMemory.begin();
      }
      if (r) {
        return r;
      }
    }
    // log.error("WAAA emergency free from stream memory!\n");
    if (freeStreamMemory(bytes)) {
      return allocate(bytes, stream);
    }
    bool retry = false;
    for (auto& v : streamPendingFreeMemory) {
      Regions& region = *v.second;
      retry |= freePending(region);
    }
    if (retry) {
      return allocate(bytes, stream);
    }
    if (mapMoreMemory(bytes)) {
      return allocate(bytes, stream);
    }
    memlog.verbose("Free memory:\n%s\n", debugFreeMemory());

    std::lock_guard mrl(mappedRegionsMutex);

    size_t nMapped = 0;
    size_t nFree = 0;
    size_t largestChunk = 0;
    size_t nMappedRegions = 0;
    for (auto& v : mappedRegions) {
      nMapped += v.end - v.begin;
      ++nMappedRegions;
    }
    auto count = [&](Regions& regions) {
      for (auto& v : regions.sizes) {
        nFree += v.bytes;
        largestChunk = std::max(largestChunk, v.bytes);
      }
    };
    count(freeMemory);
    for (auto& v : streamFreeMemory) {
      count(*v.second);
    }
    size_t nAllocated = nMapped - nFree;

    throw std::runtime_error(fmt::sprintf(
        "Moodist CUDA Allocator failed to allocate %s. We have mapped %s (in %d regions), of which %s are "
        "currently allocated, and %s are free. The largest free chunk is %s",
        this->bytes(bytes), this->bytes(nMapped), nMappedRegions, this->bytes(nAllocated), this->bytes(nFree),
        this->bytes(largestChunk)));
  }

  // void deallocateImpl(uintptr_t cudaPtr, size_t alignedbytes, Vector<CUstream>& streams) {
  //   // memlog.info("deallocate %d\n", alignedbytes);
  //   for (CUstream stream : streams) {
  //     MemoryEventHandle me = getMemoryEvent();
  //     CHECK_CU(cuEventRecord(me->event, stream));
  //   }
  //   insertRegion(pendingFreeMemory, Span{cudaPtr, cudaPtr + alignedbytes});
  //   // pendingFreeMemoryEvents.emplace_back(Span{cudaPtr, cudaPtr + alignedbytes}, std::move(me));
  // }

  template<bool allowErrors, typename StreamList>
  void deallocate(uintptr_t cudaPtr, size_t alignedbytes, const StreamList& streamList) {
    checkMap(freeMemory.map, freeMemory.sizes);
    tmpDeallocateStreamMap.clear();
    CUstream owningStream = nullptr;
    bool first = true;
    for (auto& v : streamList) {
      if (first) {
        owningStream = v;
        first = false;
      }
      tmpDeallocateStreamMap.emplace(v);
    }
    CHECK(tmpDeallocateStreamMap.size() > 0);

    Span span{cudaPtr, cudaPtr + alignedbytes};

    bool failed = false;
    tmpDeallocateEventRegions.clear();
    for (auto& v : tmpDeallocateStreamMap) {
      CUstream stream = v.first;
      auto h = freeListEventRegions.pop();
      if (allowErrors) {
        if (!h->event) {
          h->event = Event::create();
        }
        h->eventStream = stream;
        CHECK_CU(cuEventRecord(h->event, stream));
        // memlog.debug(
        //     "recorded event (0) for ptr %#x stream %#x event %#x\n", cudaPtr, (uintptr_t)stream,
        //     (uintptr_t)(CUevent)h->event);
      } else {
        if (!h->event) {
          h->event = Event::tryCreate();
          if (!h->event) {
            failed = true;
            break;
          }
        }
        h->eventStream = stream;
        if (cuEventRecord(h->event, stream) != CUDA_SUCCESS) {
          memlog.info("failed to record event!\n");
          failed = true;
          break;
        }
        // memlog.debug(
        //     "recorded event for ptr %#x stream %#x event %#x\n", cudaPtr, (uintptr_t)stream,
        //     (uintptr_t)(CUevent)h->event);
      }
      h->span = span;
      tmpDeallocateEventRegions.push_back(std::move(h));
    }
    if (!failed) {
      insertRegion(getStreamFree(owningStream), span, tmpDeallocateEventRegions);
      // if (tmpDeallocateEventRegions.size() == 1 || true) {
      //   CUstream stream = tmpDeallocateStreamMap.begin()->first;
      //   insertRegion(getStreamFree(stream), span, tmpDeallocateEventRegions);
      // } else {
      //   CUstream stream = tmpDeallocateStreamMap.begin()->first;
      //   insertRegion(getStreamPendingFree(stream), span, tmpDeallocateEventRegions);
      //   // // insertRegion(freeMemory, Span{cudaPtr, cudaPtr + alignedbytes}, tmpDeallocateEventRegions);
      //   // insertRegion(pendingFreeMemory, Span{cudaPtr, cudaPtr + alignedbytes}, tmpDeallocateEventRegions);
      // }
      for (auto& v : tmpDeallocateEventRegions) {
        v.release();
      }
      checkMap(freeMemory.map, freeMemory.sizes);
    } else {
      auto streams = freeListStreams.pop();
      streams->clear();
      for (auto& v : tmpDeallocateStreamMap) {
        streams->push_back(v.first);
      }
      pendingDeallocations.emplace_back(cudaPtr, alignedbytes, std::move(streams));
      checkMap(freeMemory.map, freeMemory.sizes);
    }
  }

  bool freeStreamMemory(size_t reqbytes) {
    checkMap(freeMemory.map, freeMemory.sizes);
    for (auto& v : streamFreeMemory) {
      Regions& region = *v.second;
      for (auto i = region.map.begin(); i != region.map.end();) {
        Span span = i->span;
        auto events = std::move(i->events);
        freeRegionSizeNodes.push_back(region.sizes.extract(i->sizeIterator));
        i->sizeIterator = {};
        auto ni = std::next(i);
        freeRegionNodes.push_back(region.map.extract(i));
        i = ni;
        tmpEventRegions.clear();
        for (auto& v : events) {
          tmpEventRegions.push_back(&v);
        }
        insertRegion(freeMemory, span, tmpEventRegions);

        size_t n = span.end - span.begin;
        if (n >= reqbytes) {
          checkMap(freeMemory.map, freeMemory.sizes);
          return true;
        }
        reqbytes -= n;
      }
    }
    checkMap(freeMemory.map, freeMemory.sizes);
    return false;
  }

  bool freeSomePending(Regions& pendingFreeMemory) {
    // CHECK(pendingFreeMemoryEvents.size() >= pendingFreeMemory.size());
    checkMap(pendingFreeMemory.map, pendingFreeMemory.sizes);
    checkMap(freeMemory.map, freeMemory.sizes);
    // for (auto i = pendingFreeMemory.sizes.begin(); i != pendingFreeMemory.sizes.end();) {
    //   auto ni = std::next(i);
    //   freeRegionSizeNodes.push_back(pendingFreeMemory.sizes.extract(i));
    //   i = ni;
    // }
    bool retval = false;
    for (auto i = pendingFreeMemory.map.begin(); i != pendingFreeMemory.map.end();) {
      if (!i->events.empty()) {
        auto& e = i->events.front();
        if (cuEventQuery(e.event) != CUDA_SUCCESS) {
          continue;
        }
      }
      Span span = i->span;
      auto events = std::move(i->events);
      freeRegionSizeNodes.push_back(pendingFreeMemory.sizes.extract(i->sizeIterator));
      i->sizeIterator = {};
      auto ni = std::next(i);
      freeRegionNodes.push_back(pendingFreeMemory.map.extract(i));
      i = ni;
      tmpEventRegions.clear();
      for (auto& v : events) {
        tmpEventRegions.push_back(&v);
      }
      insertRegion(freeMemory, span, tmpEventRegions);

      // memlog.info("free %d bytes of pending\n", span.end - span.begin);

      retval = true;
    }
    // if (!retval) {
    //   memlog.info("failed to free any pending\n");
    // }
    checkMap(freeMemory.map, freeMemory.sizes);
    return retval;
  }

  bool freePending(Regions& pendingFreeMemory) {
    // memlog.info("free pending\n");
    //  CHECK(pendingFreeMemoryEvents.size() >= pendingFreeMemory.size());
    checkMap(pendingFreeMemory.map, pendingFreeMemory.sizes);
    checkMap(freeMemory.map, freeMemory.sizes);
    for (auto i = pendingFreeMemory.sizes.begin(); i != pendingFreeMemory.sizes.end();) {
      auto ni = std::next(i);
      freeRegionSizeNodes.push_back(pendingFreeMemory.sizes.extract(i));
      i = ni;
    }
    bool retval = false;
    for (auto i = pendingFreeMemory.map.begin(); i != pendingFreeMemory.map.end();) {
      Span span = i->span;
      auto events = std::move(i->events);
      i->sizeIterator = {};
      auto ni = std::next(i);
      freeRegionNodes.push_back(pendingFreeMemory.map.extract(i));
      i = ni;
      tmpEventRegions.clear();
      for (auto& v : events) {
        tmpEventRegions.push_back(&v);
      }
      insertRegion(freeMemory, span, tmpEventRegions);
      retval = true;
    }
    checkMap(freeMemory.map, freeMemory.sizes);
    return retval;
  }

  std::string bytes(size_t n) {
    return fmt::sprintf("%d bytes (%gG)", n, n / 1024.0 / 1024.0 / 1024.0);
  }

  std::string debugFreeMemory() {
    std::string s;
    size_t total = 0;
    size_t totalMappedBytes = 0;
    size_t largestChunk = 0;
    for (auto& v : mappedRegions) {
      totalMappedBytes += v.end - v.begin;
      size_t prevEnd = v.begin;
      for (auto& x : freeMemory.map) {
        if (x.span.begin < v.begin || x.span.begin >= v.end) {
          continue;
        }
        auto& v = x.span;
        if (v.begin != prevEnd) {
          s += fmt::sprintf("ALLOCATED %s\n", bytes(v.begin - prevEnd));
        }
        s += fmt::sprintf("[%#x, %#x)  %s\n", v.begin, v.end, bytes(v.end - v.begin));
        largestChunk = std::max(largestChunk, v.end - v.begin);
        total += v.end - v.begin;
        prevEnd = v.end;
      }
      if (prevEnd != v.end) {
        s += fmt::sprintf("ALLOCATED %s\n", bytes(v.end - prevEnd));
      }
      CHECK(prevEnd <= v.end);
    }
    s += fmt::sprintf(
        "%s total, %s used, %s free. Largest free chunk %s\n", bytes(totalMappedBytes), bytes(totalMappedBytes - total),
        bytes(total), bytes(largestChunk));

    // s += fmt::sprintf("%d live cuda tensors:\n", cudaCpuMappings.size());
    // for (auto& v : cudaCpuMappings) {
    //   s += fmt::sprintf("%#x: %s\n", v.first, bytes(v.second->bytes));
    // }
    // s += fmt::sprintf("%d live cuda tensors:\n", cudaCpuMaps.size());
    // total = 0;
    // for (auto& v : cudaCpuMaps) {
    //   total += v.second.itemsize() * v.second.numel();
    //   s += fmt::sprintf("%#x: %s\n", v.first, bytes(v.second.itemsize() * v.second.numel()));
    // }
    // s += fmt::sprintf("total %s\n", bytes(total));
    return s;
  }

  // MemoryEventHandle getMemoryEvent() {
  //   if (memoryEventFreelist.empty()) {
  //     memlog.debug("allocate new memory events!\n");
  //     for (size_t i = 0; i != 0x1000; ++i) {
  //       MemoryEvent* me = memoryEventAllocator.allocate();
  //       CHECK_CU(cuEventCreate(&me->event, CU_EVENT_DISABLE_TIMING));
  //       memoryEventFreelist.push_back(me);
  //     }
  //   }
  //   MemoryEvent* me = memoryEventFreelist.back();
  //   memoryEventFreelist.pop_back();
  //   CHECK(me->event != nullptr);
  //   CHECK(me->refcount == 0);
  //   return MemoryEventHandle(me, &memoryEventFreelist);
  // }
};

// CudaAllocatorImpl& cudaAllocImpl = *new CudaAllocatorImpl();

// void deleter_func(void* ptr) {
//   cudaAllocImpl.deallocate((uintptr_t)ptr);
// }

} // namespace

struct CUDAAllocator : c10::cuda::CUDACachingAllocator::CUDAAllocator {

  SpinMutex mutex;

  CudaAllocatorImpl impl;

  CUcontext cuContext = nullptr;
  int deviceIndex = -1;

  Vector<std::unique_ptr<Vector<CUstream>>> streamListFree;
  HashMap<uintptr_t, std::unique_ptr<Vector<CUstream>>, PointerHash> streamUses;

  c10::CachingDeviceAllocator::DeviceStats deviceStats;

  HashMap<uintptr_t, FunctionPointer, PointerHash> freeCallbacks;

  virtual torch::DataPtr allocate(size_t bytes) override {
    std::lock_guard l(mutex);

    if (deviceIndex == -1) {
      deviceIndex = c10::cuda::current_device();
      CHECK(deviceIndex != -1);

      cuCtxGetCurrent(&cuContext);
      if (!cuContext) {
        CUdevice cuDevice;
        CHECK_CU(cuInit(0));
        CHECK_CU(cuDeviceGet(&cuDevice, deviceIndex));
        CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
        CHECK_CU(cuCtxSetCurrent(cuContext));
      }
    } else {
      int currentDevice = c10::cuda::current_device();
      if (currentDevice != deviceIndex) {
        throw std::runtime_error(fmt::sprintf(
            "Moodist CUDA Allocator can only be used on one device. It was initialized on device %d, but the "
            "requested "
            "allocation is on device %d. This is not supported.",
            deviceIndex, currentDevice));
      }
      CUcontext currentContext = nullptr;
      cuCtxGetCurrent(&currentContext);
      if (!currentContext) {
        CHECK_CU(cuCtxSetCurrent(cuContext));
      } else {
        if (currentContext != cuContext) {
          throw std::runtime_error(fmt::sprintf("Moodist CUDA Allocator CUDA context changed. The Moodist CUDA "
                                                "Allocator can only be used on one CUDA context."));
        }
      }
    }

    constexpr size_t alignment = 0x100;
    size_t alignedbytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);

    // size_t index = __builtin_ia32_lzcnt_u64(alignedbytes);
    // alignedbytes = ((alignedbytes >> (61 - index)) + 1) << (61 - index);

    // alignedbytes = std::max(alignment, (bytes + alignment - 1) / alignment * alignment);

    CUstream stream = c10::cuda::getCurrentCUDAStream();
    memlog.debug("trying to allocate %d bytes on stream %#x\n", alignedbytes, (uintptr_t)stream);
    uintptr_t cudaPtr = impl.allocate(alignedbytes, stream);
    deviceStats.allocated_bytes[0].increase(alignedbytes);
    deviceStats.active_bytes[0].increase(alignedbytes);
    Function<void()> f = [this, cudaPtr, alignedbytes, stream] {
      std::lock_guard l(mutex);
      memlog.debug("deallocate %#x  %d bytes on stream %#x\n", cudaPtr, alignedbytes, (uintptr_t)stream);
      {
        auto it = freeCallbacks.find(cudaPtr);
        if (it != freeCallbacks.end()) {
          FunctionPointer head = it->second;
          while (head) {
            FunctionPointer next = head->next;
            Function<void()>{head}();
            head = next;
          }
          freeCallbacks.erase(it);
        }
      }
      auto it = streamUses.find(cudaPtr);
      if (it != streamUses.end()) {
        auto& list = it->second;
        // it is unclear from the pytorch docs if, when record_stream is called,
        // the allocator should also sync the original stream the tensor was
        // allocated on. we do it just in case
        // list->push_back(stream);
        list->insert(list->begin(), stream);
        impl.deallocate<false>(cudaPtr, alignedbytes, *list);
        list->clear();
        streamListFree.push_back(std::move(list));
        streamUses.erase(it);
      } else {
        impl.deallocate<false>(cudaPtr, alignedbytes, std::array<CUstream, 1>{stream});
      }
      deviceStats.allocated_bytes[0].decrease(alignedbytes);
      deviceStats.active_bytes[0].decrease(alignedbytes);
    };
    auto deleter = [](void* c) { Function<void()>(FunctionPointer(c))(); };
    torch::Device device = torch::Device(torch::kCUDA, deviceIndex);
    memlog.debug("allocate(%d) returning cudaPtr %#x\n", bytes, cudaPtr);
    return torch::DataPtr((void*)cudaPtr, (void*)f.release(), deleter, device);
  }
  virtual torch::DeleterFnPtr raw_deleter() const override {
    // throw std::runtime_error("moodist CUDAAllocator::raw_deleter: not implemented");
    // return deleter_func;
    return nullptr;
  }
  virtual void copy_data(void* dest, const void* src, std::size_t count) const override {
    throw std::runtime_error("moodist CUDAAllocator::copy_data: not implemented");
  }
  HashMap<void*, torch::DataPtr, PointerHash> rawAllocations;
  virtual void* raw_alloc(size_t nbytes) override {
    // return raw_alloc_with_stream(nbytes, c10::cuda::getCurrentCUDAStream());
    auto ptr = allocate(nbytes);
    std::lock_guard l(mutex);
    void* r = ptr.get();
    rawAllocations[r] = std::move(ptr);
    return r;
  }
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    throw std::runtime_error("moodist CUDAAllocator::raw_alloc_with_stream: not implemented");
  }
  virtual void raw_delete(void* ptr) override {
    std::unique_lock l(mutex);
    auto i = rawAllocations.find(ptr);
    if (i == rawAllocations.end()) {
      throw std::runtime_error("moodist CUDAAllocator::raw_delete: allocation not found");
    }
    auto h = std::move(i->second);
    rawAllocations.erase(i);
    l.unlock();
  }
  virtual void init(int device_count) override {
    // throw std::runtime_error("moodist CUDAAllocator::init: not implemented");
  }
  virtual bool initialized() override {
    throw std::runtime_error("moodist CUDAAllocator::initialized: not implemented");
  }
  virtual double getMemoryFraction(c10::DeviceIndex) {
    return 1.0;
  }
  virtual void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    throw std::runtime_error("moodist CUDAAllocator::setMemoryFraction: not implemented");
  }
  virtual void enable(bool) {}
  virtual bool isEnabled() const {
    return true;
  }
  virtual void emptyCache() {}
  virtual void emptyCache(at::cuda::MempoolId_t) {}
  virtual void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) override {
    throw std::runtime_error("moodist CUDAAllocator::cacheInfo: not implemented");
  }
  virtual void* getBaseAllocation(void* ptr, size_t* size) override {
    throw std::runtime_error("moodist CUDAAllocator::getBaseAllocation: not implemented");
  }
  virtual void recordStream(const torch::DataPtr& ptr, c10::Stream stream) override {
    auto& list = streamUses[(uintptr_t)ptr.get()];
    if (!list) {
      if (!streamListFree.empty()) {
        list = std::move(streamListFree.back());
        streamListFree.pop_back();
      } else {
        list = std::make_unique<Vector<CUstream>>();
      }
    }
    list->push_back(c10::cuda::CUDAStream{stream});
  }
  virtual c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    // throw std::runtime_error("moodist CUDAAllocator::getDeviceStats: not implemented");
    return deviceStats;
  }
  virtual void resetAccumulatedStats(c10::DeviceIndex device) override {
    // throw std::runtime_error("moodist CUDAAllocator::resetAccumulatedStats: not implemented");
    deviceStats.allocated_bytes[0].reset_accumulated();
    deviceStats.active_bytes[0].reset_accumulated();
  }
  virtual void resetPeakStats(c10::DeviceIndex device) override {
    // throw std::runtime_error("moodist CUDAAllocator::resetPeakStats: not implemented");
    deviceStats.allocated_bytes[0].reset_peak();
    deviceStats.active_bytes[0].reset_peak();
  }
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() {
    return {};
  }
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(at::cuda::MempoolId_t) {
    return {};
  }
  virtual void beginAllocateToPool(
      c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id, std::function<bool(cudaStream_t)> filter) override {
    throw std::runtime_error("moodist CUDAAllocator::beginAllocateToPool: not implemented");
  }
  virtual void endAllocateToPool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    throw std::runtime_error("moodist CUDAAllocator::endAllocateToPool: not implemented");
  }
  virtual void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override {
    throw std::runtime_error("moodist CUDAAllocator::releasePool: not implemented");
  }
  virtual c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(void* ptr) override {
    throw std::runtime_error("moodist CUDAAllocator::shareIpcHandle: not implemented");
  }
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    throw std::runtime_error("moodist CUDAAllocator::getIpcDevPtr: not implemented");
  }
  virtual void recordHistory(
      bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder, size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) {
    // throw std::runtime_error("moodist CUDAAllocator::recordHistory: not implemented");
  }
  virtual void recordHistory(
      bool enabled, c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder, size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when, bool) {
    // throw std::runtime_error("moodist CUDAAllocator::recordHistory: not implemented");
  }
  virtual void attachOutOfMemoryObserver(c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override {
    throw std::runtime_error("moodist CUDAAllocator::attachOutOfMemoryObserver: not implemented");
  }

  virtual void attachAllocatorTraceTracker(c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override {
    throw std::runtime_error("moodist CUDAAllocator::attachOutOfMemoryObserver: not implemented");
  }

  virtual void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {
    // throw std::runtime_error("moodist CUDAAllocator::enablePeerAccess: not implemented");
  }

  virtual cudaError_t memcpyAsync(
      void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream,
      bool p2p_enabled) override {
    CHECK_CU(cuMemcpyAsync((uintptr_t)dst, (uintptr_t)src, count, stream));
    // throw std::runtime_error("moodist CUDAAllocator::memcpyAsync: not implemented");
    return cudaSuccess;
  }
  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, c10::cuda::MempoolId_t id) override {
    throw std::runtime_error("moodist CUDAAllocator::getCheckpointState: not implemented");
  }
  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device, std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) override {
    throw std::runtime_error("moodist CUDAAllocator::setCheckpointPoolState: not implemented");
  }
  virtual std::string name() override {
    throw std::runtime_error("moodist CUDAAllocator::name: not implemented");
  }
};

namespace {
std::mutex assignmentMutex;
CUDAAllocator* cudaAllocator = nullptr;
} // namespace

namespace allocator {

bool owns(uintptr_t address) {
  if (!cudaAllocator) {
    return false;
  }
  std::lock_guard mrl(cudaAllocator->impl.mappedRegionsMutex);
  for (auto& v : cudaAllocator->impl.mappedRegions) {
    if (address >= v.begin && address < v.end) {
      return true;
    }
  }
  return false;
}

std::pair<uintptr_t, size_t> mappedRegion(uintptr_t address) {
  if (!cudaAllocator) {
    return {0, 0};
  }
  std::lock_guard mrl(cudaAllocator->impl.mappedRegionsMutex);
  for (auto& v : cudaAllocator->impl.mappedRegions) {
    if (address >= v.begin && address < v.end) {
      return {v.begin, v.end - v.begin};
    }
  }
  return {0, 0};
}

FreeCallbackHandle addFreeCallback(uintptr_t baseAddress, Function<void()> callback) {
  CHECK(cudaAllocator != nullptr);
  std::lock_guard l(cudaAllocator->mutex);
  FunctionPointer fp = callback.release();
  auto& head = cudaAllocator->freeCallbacks[baseAddress];
  fp->next = head;
  head = fp;
  FreeCallbackHandle r;
  r.baseAddress = baseAddress;
  r.handle = fp;
  return r;
}
void removeFreeCallback(uintptr_t baseAddress, void* handle) {
  CHECK(cudaAllocator != nullptr);
  std::lock_guard l(cudaAllocator->mutex);
  auto** head = &cudaAllocator->freeCallbacks[baseAddress];
  while (head) {
    if (*head == handle) {
      FunctionPointer f = *head;
      *head = (*head)->next;
      Function<void()>{f};
      return;
    }
    head = &(*head)->next;
  }
  throw std::runtime_error("removeFreeCallback: no such handle");
}

// bool owns(uintptr_t address) {
//   if (!cudaAllocator) {
//     return false;
//   }
//   uintptr_t base = cudaAllocator->impl.reservedBase.load(std::memory_order_relaxed);
//   size_t size = cudaAllocator->impl.reservedSize.load(std::memory_order_relaxed);
//   return address >= base && address < base + size;
// }
// uintptr_t offset(uintptr_t address) {
//   CHECK(cudaAllocator);
//   return address - cudaAllocator->impl.reservedSize.load(std::memory_order_relaxed);
// }

// Vector<std::pair<size_t, CUmemGenericAllocationHandle>> cuMemHandles() {
//   CHECK(cudaAllocator);
//   std::lock_guard l(cudaAllocator->mutex);
//   return cudaAllocator->impl.cuMemHandles;
// }

// std::pair<uintptr_t, size_t> reserved() {
//   CHECK(cudaAllocator);
//   std::lock_guard l(cudaAllocator->mutex);
//   return {cudaAllocator->impl.reservedBase, cudaAllocator->impl.reservedSize};
// }

} // namespace allocator

void enableCudaAllocator() {
  std::lock_guard l(assignmentMutex);
  if (!cudaAllocator) {
    cudaAllocator = new CUDAAllocator();
  }
  c10::cuda::CUDACachingAllocator::allocator = cudaAllocator;
}

} // namespace moodist
