// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "commondefs.h"

#pragma push_macro("CHECK")

#include "async.h"
#include "cpu_allocator.h"
#include "logging.h"
#include "vector.h"

#include "fmt/printf.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <nvml.h>
#include <nvrtc.h>
#include <torch/cuda.h>
#include <torch/types.h>

#include <optional>
#include <random>
#include <ranges>
#include <stdexcept>
#include <utility>

#undef CHECK
#pragma pop_macro("CHECK")

namespace moodist {

extern async::Scheduler& scheduler;

inline std::string removePciPathPrefix(std::string path) {
  std::string_view prefix = "/sys/devices/";
  if (path.find(prefix) == 0) {
    return path.substr(prefix.size());
  }
  return path;
}

struct AllocatedBuffer {
  void* cpuPointer = nullptr;
  uintptr_t cudaPointer = 0;
  size_t bytes = 0;
  CUmemGenericAllocationHandle handle;
  bool hostAllocated = false;
  bool numaAllocated = false;
  bool handleAllocated = false;
  std::optional<torch::DataPtr> dataPtr;

  AllocatedBuffer() = default;
  AllocatedBuffer(AllocatedBuffer&& n) noexcept {
    *this = std::move(n);
  }

  AllocatedBuffer& operator=(AllocatedBuffer&& n) noexcept {
    std::swap(cpuPointer, n.cpuPointer);
    std::swap(cudaPointer, n.cudaPointer);
    std::swap(bytes, n.bytes);
    std::swap(hostAllocated, n.hostAllocated);
    std::swap(numaAllocated, n.numaAllocated);
    std::swap(dataPtr, n.dataPtr);
    return *this;
  }

  void dtor() {
    if (numaAllocated) {
      if (cpuPointer) {
        log.verbose("free %d bytes of cpu memory %p\n", bytes, cpuPointer);
        cuMemHostUnregister(cpuPointer);
        numa_free(cpuPointer, bytes);
      }
    } else if (hostAllocated) {
      if (cpuPointer) {
        log.verbose("free %d bytes of host memory %p\n", bytes, cpuPointer);
        cuMemFreeHost(cpuPointer);
      }
    } else if (handleAllocated) {
      if (cudaPointer) {
        log.verbose("free %d bytes of mapped memory at %#x\n", bytes, cudaPointer);
        cuMemUnmap(cudaPointer, bytes);
        cuMemAddressFree(cudaPointer, bytes);
        cuMemRelease(handle);
      }
    } else {
      if (dataPtr) {
        dataPtr.reset();
      } else if (cudaPointer) {
        log.verbose("free %d bytes of cuda memory at %#x\n", bytes, cudaPointer);
        cuMemFree(cudaPointer);
      }
    }
  }

  ~AllocatedBuffer() {
    if ((uintptr_t)cpuPointer | cudaPointer) {
      dtor();
    }
  }
};

struct AllocatedArray {
  AllocatedBuffer buffer;
  size_t itembytes;
  size_t numel;
  void* cpu(size_t index) {
    CHECK(index < numel);
    return (void*)((uintptr_t)buffer.cpuPointer + itembytes * index);
  }
  uintptr_t cuda(size_t index) {
    CHECK(index < numel);
    return (uintptr_t)buffer.cudaPointer + itembytes * index;
  }

  template<typename T>
  T& at(size_t index) {
    return *(T*)cpu(index);
  }
};

struct AllocatedCpuBuffer {
  void* cpuPointer = nullptr;
  size_t bytes = 0;

  AllocatedCpuBuffer() = default;
  AllocatedCpuBuffer(const AllocatedBuffer&) = delete;
  AllocatedCpuBuffer(AllocatedCpuBuffer&& n) noexcept {
    *this = std::move(n);
  }
  AllocatedCpuBuffer& operator=(const AllocatedCpuBuffer&) = delete;
  AllocatedCpuBuffer& operator=(AllocatedCpuBuffer&& n) noexcept {
    std::swap(cpuPointer, n.cpuPointer);
    std::swap(bytes, n.bytes);
    return *this;
  }

  void clear() {
    if (cpuPointer) {
      cpu_allocator::moo_free(cpuPointer);
      cpuPointer = nullptr;
    }
  }

  ~AllocatedCpuBuffer() {
    clear();
  }
};

struct PeerArrayRef {
  uintptr_t base = 0;
  uintptr_t itembytes = 0;

  uintptr_t get(size_t index) {
    return (uintptr_t)base + itembytes * index;
  }
};

inline size_t bufferHash(const void* data, size_t len) {
  auto rotl = [&](auto v, int n) {
    static_assert(std::is_unsigned_v<decltype(v)>);
    return (v << n) | (v >> (8 * sizeof(v) - n));
  };

  const char* ptr = (const char*)data;
  const char* end = ptr + len / sizeof(size_t) * sizeof(size_t);
  size_t r = 0;
  while (ptr != end) {
    size_t v;
    std::memcpy(&v, ptr, sizeof(size_t));
    r = rotl(r + v * 31, 15);
    ptr += sizeof(size_t);
  }
  len -= len / sizeof(size_t) * sizeof(size_t);
  while (len) {
    uint8_t v;
    std::memcpy(&v, ptr, sizeof(uint8_t));
    r = rotl(r + v * 31, 15);
    ptr += sizeof(uint8_t);
    --len;
  }
  return r * 11400714819323198485llu;
}

struct IpcMemHash {
  template<typename T>
  size_t operator()(const T& v) const {
    static_assert(sizeof(v) % sizeof(size_t) == 0);
    return bufferHash(&v, sizeof(v));
  }
};
struct IpcMemEqual {
  template<typename T>
  bool operator()(const T& a, const T& b) const {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
  }
};

struct Error : std::exception {
  std::string str;
  Error() {}
  Error(std::string&& str) : str(std::move(str)) {}
  virtual const char* what() const noexcept override {
    return str.c_str();
  }
};

inline auto seedRng() {
  std::random_device dev;
  auto start = std::chrono::high_resolution_clock::now();
  std::seed_seq ss(
      {(uint32_t)dev(), (uint32_t)dev(), (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(),
       (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count(), (uint32_t)dev(),
       (uint32_t)std::chrono::system_clock::now().time_since_epoch().count(),
       (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
       (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(), (uint32_t)dev(),
       (uint32_t)(std::chrono::high_resolution_clock::now() - start).count(), (uint32_t)dev(),
       (uint32_t)std::hash<std::thread::id>()(std::this_thread::get_id())});
  return std::mt19937_64(ss);
};

inline std::mt19937_64& getRng() {
  thread_local std::mt19937_64 rng{seedRng()};
  return rng;
}

template<typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
T random(T min = std::numeric_limits<T>::min(), T max = std::numeric_limits<T>::max()) {
  return std::uniform_int_distribution<T>(min, max)(getRng());
}

inline std::string randomName() {
  std::string r;
  for (int i = 0; i != 16; ++i) {
    r += "0123456789abcdef"[std::uniform_int_distribution<int>(0, 15)(getRng())];
  }
  return r;
}

template<typename Duration>
float seconds(Duration duration) {
  return std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1>>>(duration).count();
}

inline std::string hexstr(const void* data, size_t len) {
  std::string r;
  const uint8_t* p = (uint8_t*)data;
  const uint8_t* e = p + len;
  while (p != e) {
    r += "0123456789abcdef"[*p >> 4];
    r += "0123456789abcdef"[*p & 0x0f];
    ++p;
  }
  return r;
}

struct Event {
  CUevent event = nullptr;
  bool owning = false;
  Event() = default;
  Event(const Event&) = delete;
  Event(Event&& n) {
    *this = std::move(n);
  }
  ~Event() {
    if (owning && event) {
      cuEventDestroy(event);
    }
  }

  static Event reference(CUevent event) {
    Event r;
    r.event = event;
    r.owning = false;
    return r;
  }
  static Event create() {
    // log.info("Creating a new event\n");
    Event r;
    CHECK_CU(cuEventCreate(&r.event, CU_EVENT_DISABLE_TIMING));
    r.owning = true;
    return r;
  }
  static Event tryCreate() noexcept {
    // log.info("Creating a new event\n");
    Event r;
    if (cuEventCreate(&r.event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
      return r;
    }
    r.owning = true;
    return r;
  }
  static Event createInterprocess() {
    Event r;
    CHECK_CU(cuEventCreate(&r.event, CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS));
    r.owning = true;
    return r;
  }

  Event& operator=(const Event&) = delete;
  Event& operator=(Event&& n) {
    std::swap(event, n.event);
    std::swap(owning, n.owning);
    return *this;
  }

  operator CUevent() {
    return event;
  }

  void record(CUstream stream) {
    CHECK_CU(cuEventRecord(event, stream));
  }
  void wait(CUstream stream) {
    CHECK_CU(cuStreamWaitEvent(stream, event, CU_EVENT_WAIT_DEFAULT));
  }

  void synchronize() {
    CHECK_CU(cuEventSynchronize(event));
  }
};

struct Stream {
  CUstream stream = nullptr;
  bool owning = false;
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream(Stream&& n) {
    *this = std::move(n);
  }
  ~Stream() {
    if (owning && stream) {
      cuStreamDestroy(stream);
    }
  }
  Stream& operator=(const Stream&) = delete;
  Stream& operator=(Stream&& n) {
    std::swap(stream, n.stream);
    std::swap(owning, n.owning);
    return *this;
  }
  operator CUstream() {
    return stream;
  }

  static Stream reference(CUstream stream) {
    Stream r;
    r.stream = stream;
    r.owning = false;
    return r;
  }
  static Stream create(int priority = 0) {
    Stream r;
    // CHECK_CU(cuStreamCreate(&r.stream, CU_STREAM_NON_BLOCKING));
    CHECK_CU(cuStreamCreateWithPriority(&r.stream, CU_STREAM_NON_BLOCKING, priority));
    r.owning = true;
    return r;
  }
};

template<typename T, size_t maxThreadLocalEntries = 0x40>
struct FLPtr {
  struct Storage {
    Storage* next;
    T object;

    template<typename... Args>
    Storage(Args&&... args) : object(std::forward<Args>(args)...) {}

    void clear() {
      object.clear();
    }
  };
  Storage* ptr = nullptr;
  FLPtr(std::nullptr_t) {}
  FLPtr() = default;
  FLPtr(void* ptr) : ptr((Storage*)ptr) {}
  ~FLPtr() {
    if (ptr) {
      ptr->clear();
      FreeList<Storage>::push(ptr, maxThreadLocalEntries);
      ptr = nullptr;
    }
  }
  FLPtr(const FLPtr&) = delete;
  FLPtr& operator=(const FLPtr&) = delete;
  FLPtr(FLPtr&& n) {
    ptr = std::exchange(n.ptr, nullptr);
  }
  static FLPtr make() {
    Storage* ptr = FreeList<Storage>::pop();
    if (!ptr) [[unlikely]] {
      ptr = new (cpu_allocator::moo_alloc(sizeof(Storage))) Storage();
      ptr->clear();
    }
    FLPtr r;
    r.ptr = ptr;
    return r;
  }
  void* release() {
    return std::exchange(ptr, nullptr);
  }
  FLPtr& operator=(FLPtr&& n) {
    std::swap(ptr, n.ptr);
    return *this;
  }
  operator T&() const {
    return ptr->object;
  }
  T& operator*() const {
    return ptr->object;
  }
  T* operator->() const {
    return &ptr->object;
  }

  bool operator!=(std::nullptr_t) const {
    return ptr != nullptr;
  }
  bool operator==(std::nullptr_t) const {
    return ptr == nullptr;
  }
  explicit operator bool() const {
    return ptr != nullptr;
  }
};

template<typename T, size_t maxThreadLocalEntries = 0x100>
struct FLSharedPtr {
  struct Storage {
    Storage* next;
    std::atomic_size_t refcount = 0;
    T object;

    template<typename... Args>
    Storage(Args&&... args) : object(std::forward<Args>(args)...) {}

    void clear() {
      object.clear();
    }
  };
  Storage* ptr = nullptr;
  FLSharedPtr(std::nullptr_t) {}
  FLSharedPtr() = default;
  FLSharedPtr(Storage* ptr) : ptr(ptr) {
    if (ptr) {
      CHECK(ptr->refcount != 0);
    }
  }
  ~FLSharedPtr() {
    if (ptr && --ptr->refcount == 0) {
      ptr->clear();
      FreeList<Storage>::push(ptr, maxThreadLocalEntries);
      ptr = nullptr;
    }
  }
  FLSharedPtr(const FLSharedPtr& n) {
    ptr = n.ptr;
    if (ptr) {
      ++ptr->refcount;
    }
  }
  FLSharedPtr& operator=(const FLSharedPtr& n) {
    ptr = n.ptr;
    if (ptr) {
      ++ptr->refcount;
    }
    return *this;
  }
  FLSharedPtr(FLSharedPtr&& n) {
    ptr = std::exchange(n.ptr, nullptr);
  }
  static FLSharedPtr make() {
    Storage* ptr = FreeList<Storage>::pop();
    if (!ptr) [[unlikely]] {
      ptr = new (cpu_allocator::moo_alloc(sizeof(Storage))) Storage();
      ptr->clear();
    }
    CHECK(ptr->refcount == 0);
    ptr->refcount = 1;
    FLSharedPtr r;
    r.ptr = ptr;
    return r;
  }
  FLSharedPtr& operator=(FLSharedPtr&& n) {
    std::swap(ptr, n.ptr);
    return *this;
  }
  operator T&() const {
    return ptr->object;
  }
  T& operator*() const {
    return ptr->object;
  }
  T* operator->() const {
    return &ptr->object;
  }

  bool operator!=(std::nullptr_t) const {
    return ptr != nullptr;
  }
  bool operator==(std::nullptr_t) const {
    return ptr == nullptr;
  }
  explicit operator bool() const {
    return ptr != nullptr;
  }

  Storage* release() {
    return std::exchange(ptr, nullptr);
  }
};

using AllocatedCpuBufferSharedPtr = FLSharedPtr<AllocatedCpuBuffer>;

struct TensorData {
  AllocatedCpuBufferSharedPtr buffer;
  uintptr_t dataPtr;
  size_t dataBytes;
  int dtype;
  std::vector<int64_t> shape;
  bool isCuda;

  void clear() {
    buffer = {};
    dtype = -1;
    shape.clear();
    dataPtr = 0;
    dataBytes = 0;
    isCuda = false;
  }

  uintptr_t data() {
    return dataPtr;
  }
  size_t bytes() {
    return dataBytes;
  }
  size_t itemsize() {
    return torch::elementSize(((torch::ScalarType)dtype));
  }
  size_t numel() {
    size_t r = 1;
    for (int64_t n : shape) {
      r *= n;
    }
    return r;
  }
};

using TensorDataPtr = FLPtr<TensorData>;
using TensorDataSharedPtr = FLSharedPtr<TensorData>;

namespace cpu_allocator {
AllocatedCpuBufferSharedPtr getCpuBuffer(uintptr_t address);
void refCpuBuffer(AllocatedCpuBufferSharedPtr ptr);
void derefCpuBuffer(uintptr_t address);
} // namespace cpu_allocator

inline torch::Tensor makeTensor(TensorDataPtr ptr) {
  torch::Device device(torch::kCPU);

  CHECK(ptr != nullptr);

  CHECK(!ptr->isCuda);

  cpu_allocator::refCpuBuffer(ptr->buffer);

  TensorData* td = &*ptr;

  CHECK(td->data() != 0);

  Function<void()> f = [ptr = std::move(ptr)]() mutable { cpu_allocator::derefCpuBuffer(ptr->data()); };
  auto deleter = [](void* c) { Function<void()>(FunctionPointer(c))(); };
  auto data = torch::DataPtr((void*)td->data(), (void*)f.release(), deleter, device);

  torch::Storage storage(torch::Storage::use_byte_size_t(), td->bytes(), std::move(data), nullptr, false);
  return torch::empty({0}, torch::TensorOptions((torch::ScalarType)td->dtype).device(device))
      .set_(std::move(storage), 0, td->shape);
}

struct FutureImpl {
  TensorDataPtr result;
  std::atomic_uint32_t done = 0;
  // WorkCudaDonePtr cudaDone = nullptr;
  void clear() {
    result = nullptr;
    done = 0;
  }
};

using FutureImplSharedPtr = FLSharedPtr<FutureImpl>;

template<typename T>
auto range(T begin, T end) {
  return std::views::iota(begin, end);
}
auto range(auto n) {
  return std::views::iota((decltype(n))0, n);
}
auto indices(auto&& c) {
  return range(c.size());
}

struct PairHash {
  template<typename A, typename B>
  size_t operator()(const std::pair<A, B>& v) const {
    return std::hash<A>()(v.first) ^ std::hash<B>()(v.second);
  }
};

template<class T>
struct InternalAllocator {
  typedef T value_type;
  InternalAllocator() = default;
  template<class U>
  constexpr InternalAllocator(const InternalAllocator<U>&) noexcept {}
  T* allocate(size_t n) {
    void* r = internalAlloc(sizeof(T) * n);
    if (!r) {
      throw std::bad_alloc();
    }
    return (T*)r;
  }
  void deallocate(T* p, std::size_t n) noexcept {
    internalFree(p);
  }
};

template<typename T>
using IVector = Vector<T, InternalAllocator<T>>;

template<size_t size, size_t alignment>
struct AlignedStorage {
  alignas(alignment) unsigned char storage[size];
};

template<typename U>
struct UniqueImpl {
  U u;
  UniqueImpl() = default;
  ~UniqueImpl() {
    if (u.impl) [[unlikely]] {
      u.destroy();
    }
  }
  UniqueImpl(std::nullptr_t) {}
  UniqueImpl(UniqueImpl&& n) {
    u.impl = std::exchange(n.u.impl, nullptr);
  }
  UniqueImpl(const UniqueImpl&) = delete;
  UniqueImpl& operator=(UniqueImpl&& n) {
    std::swap(u.impl, n.u.impl);
    return *this;
  }
  UniqueImpl& operator=(const UniqueImpl&) = delete;

  UniqueImpl& operator=(std::nullptr_t) {
    if (u.impl) [[unlikely]] {
      u.destroy();
      u.impl = nullptr;
    }
    return *this;
  }

  operator U*() {
    return &u;
  }
  U& operator*() {
    return u;
  }
  U* operator->() {
    return &u;
  }

  template<typename T>
  T* as() {
    return (T*)u.impl;
  }
};

template<typename... Args>
struct Global {
  std::optional<std::tuple<Args...>> args;
  Global(Args&&... args) {
    this->args.emplace(std::forward<Args>(args)...);
  }
  template<typename T>
  operator T&() {
    static_assert(alignof(T) <= alignof(std::max_align_t));
    CHECK(args);
    return std::apply(
        []<typename... X>(X&&... args) -> T& { return *new (internalAlloc(sizeof(T))) T(std::forward<X>(args)...); },
        std::move(*args));
  }
};

} // namespace moodist
