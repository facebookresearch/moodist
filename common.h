#pragma once

#include "async.h"
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

extern async::Scheduler scheduler;

inline void throwNvrtc(nvrtcResult error, const char* file, int line) {
  throw std::runtime_error(fmt::sprintf("%s:%d: nvrtc error %d %s", file, line, error, nvrtcGetErrorString(error)));
}
inline void throwCu(CUresult error, const char* file, int line) {
  const char* str = "unknown cuda error";
  cuGetErrorString(error, &str);
  throw std::runtime_error(fmt::sprintf("%s:%d: cuda error %d: %s", file, line, error, str));
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
      fatal("[CHECK FAILED %s:%d] %s\n", __FILE__, __LINE__, #x);                                                      \
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
  bool hostAllocated = false;
  bool numaAllocated = false;

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
    } else {
      if (cudaPointer) {
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
  AllocatedCpuBuffer(AllocatedCpuBuffer&& n) noexcept {
    *this = std::move(n);
  }

  AllocatedCpuBuffer& operator=(AllocatedCpuBuffer&& n) noexcept {
    std::swap(cpuPointer, n.cpuPointer);
    std::swap(bytes, n.bytes);
    return *this;
  }

  ~AllocatedCpuBuffer() {
    if (cpuPointer) {
      numa_free(cpuPointer, bytes);
    }
  }
};

struct PeerArrayRef {
  uintptr_t base = 0;
  uintptr_t itembytes = 0;
};

struct IpcMemHash {
  size_t operator()(const CUipcMemHandle& v) const {
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
  bool operator()(const CUipcMemHandle& a, const CUipcMemHandle& b) const {
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

} // namespace moodist
