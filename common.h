#pragma once

#include "async.h"
#include "cpu_allocator.h"
#include "logging.h"

#include "fmt/printf.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <numa.h>
#include <nvml.h>
#include <nvrtc.h>
#include <torch/cuda.h>
#include <torch/types.h>

#include <random>
#include <stdexcept>
#include <utility>

namespace moodist {

extern async::Scheduler& scheduler;

struct CudaError : std::runtime_error {
  CUresult error;
  CudaError(CUresult error, const std::string& message) : error(error), std::runtime_error(message) {}
};

inline void throwNvrtc(nvrtcResult error, const char* file, int line) {
  throw std::runtime_error(fmt::sprintf("%s:%d: nvrtc error %d %s", file, line, error, nvrtcGetErrorString(error)));
}
inline void throwCu(CUresult error, const char* file, int line) {
  const char* str = "unknown cuda error";
  cuGetErrorString(error, &str);
  throw CudaError(error, fmt::sprintf("%s:%d: cuda error %d: %s", file, line, error, str));
}
inline void throwNvml(nvmlReturn_t error, const char* file, int line) {
  const char* str = "unknown nvml error";
  str = nvmlErrorString(error);
  throw std::runtime_error(fmt::sprintf("%s:%d: nvml error %d: %s", file, line, error, str));
}

#define CHECK_NVRTC(x)                                                                                                 \
  {                                                                                                                    \
    nvrtcResult error__ = (x);                                                                                         \
    if (error__ != NVRTC_SUCCESS) [[unlikely]]                                                                         \
      throwNvrtc(error__, __FILE__, __LINE__);                                                                         \
  }
#define CHECK_CU(x)                                                                                                    \
  {                                                                                                                    \
    CUresult error__ = (x);                                                                                            \
    if (error__ != CUDA_SUCCESS) [[unlikely]]                                                                          \
      throwCu(error__, __FILE__, __LINE__);                                                                            \
  }

#define CHECK_NVML(x)                                                                                                  \
  {                                                                                                                    \
    nvmlReturn_t error__ = (x);                                                                                        \
    if (error__ != NVML_SUCCESS) [[unlikely]]                                                                          \
      throwNvml(error__, __FILE__, __LINE__);                                                                          \
  }

#undef CHECK
#define CHECK(x)                                                                                                       \
  {                                                                                                                    \
    if (!(x)) [[unlikely]]                                                                                             \
      moodist::fatal("[CHECK FAILED %s:%d] %s\n", __FILE__, __LINE__, #x);                                             \
  }

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

struct IpcMemHash {
  template<typename T>
  size_t operator()(const T& v) const {
    static_assert(sizeof(v) % sizeof(size_t) == 0);
    auto rotl = [&](auto v, int n) {
      static_assert(std::is_unsigned_v<decltype(v)>);
      return (v << n) | (v >> (8 * sizeof(v) - n));
    };

    const char* ptr = (const char*)&v;
    const char* end = ptr + sizeof(v);
    size_t r = 0;
    while (ptr != end) {
      size_t v;
      std::memcpy(&v, ptr, sizeof(size_t));
      r = rotl(r + v * 31, 15);
      ptr += sizeof(size_t);
    }
    return r * 11400714819323198485llu;
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
  static Stream create() {
    Stream r;
    CHECK_CU(cuStreamCreate(&r.stream, CU_STREAM_NON_BLOCKING));
    // CHECK_CU(cuStreamCreateWithPriority(&r.stream, CU_STREAM_NON_BLOCKING, 100));
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

} // namespace moodist
