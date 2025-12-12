// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "logging.h"

#include "fmt/printf.h"

#include <cuda.h>
#include <nvml.h>
#include <nvrtc.h>

#include <cstddef>
#include <stdexcept>

// Marks functions that are exported across the core/wrapper library boundary.
// Both libmoodist.so (core) and _C.so (wrapper) need to call each other,
// so these symbols have default visibility while everything else is hidden.
#define MOODIST_API __attribute__((visibility("default")))

namespace moodist {

void* numa_alloc_onnode(size_t bytes, int node);
void* numa_alloc_local(size_t bytes);
void numa_free(void* ptr, size_t bytes);
void numa_run_on_node(int node);
void numa_move(void* ptr, size_t bytes, int node);
void numa_membind(int node);

void* internalAlloc(size_t bytes);
void internalFree(void* ptr);
void internalAllocatorSetNode(int node);
size_t internalAllocSize(void* ptr);

template<typename T, typename... Args>
T* internalNew(Args&&... args) {
  void* mem = internalAlloc(sizeof(T));
  if (!mem) {
    throw std::bad_alloc();
  }
  return new (mem) T(std::forward<Args>(args)...);
}

template<typename T>
void internalDelete(T* ptr) {
  if (ptr) {
    ptr->~T();
    internalFree(ptr);
  }
}

template<typename T>
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
  void deallocate(T* p, std::size_t) noexcept {
    internalFree(p);
  }
};

struct CudaError : std::runtime_error {
  CUresult error;
  CudaError(CUresult error, const std::string& message) : error(error), std::runtime_error(message) {}
};

[[noreturn]] [[gnu::cold]] inline void throwNvrtc(nvrtcResult error, const char* file, int line) {
  throw std::runtime_error(fmt::sprintf("%s:%d: nvrtc error %d %s", file, line, error, nvrtcGetErrorString(error)));
}
[[noreturn]] [[gnu::cold]] inline void throwCu(CUresult error, const char* file, int line) {
  const char* str = "unknown cuda error";
  cuGetErrorString(error, &str);
  throw CudaError(error, fmt::sprintf("%s:%d: cuda error %d: %s", file, line, error, str));
}
[[noreturn]] [[gnu::cold]] inline void throwNvml(nvmlReturn_t error, const char* file, int line) {
  const char* str = "unknown nvml error";
  str = nvmlErrorString(error);
  throw std::runtime_error(fmt::sprintf("%s:%d: nvml error %d: %s", file, line, error, str));
}

[[noreturn]] [[gnu::cold]] inline void throwCheckFail(const char* file, int line, const char* text) {
  std::string str = fmt::sprintf("[CHECK FAILED %s:%d] %s\n", file, line, text);
  log.error("%s", str);
  throw std::runtime_error(str);
}

[[noreturn]] [[gnu::cold]] inline void throwErrno(int e, const char* text) {
  log.error("%s: error %d: %s", text, e, std::strerror(e));
  throw std::system_error(e, std::generic_category(), text);
}

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)

#define NORETURN(...)                                                                                                  \
  [&] [[noreturn]] [[gnu::cold]] [[gnu::noinline]] () -> decltype(auto) {                                              \
    __VA_ARGS__                                                                                                        \
  }();
#define NOINLINE(...)                                                                                                  \
  [&] [[gnu::noinline]] () -> decltype(auto) {                                                                         \
    __VA_ARGS__                                                                                                        \
  }();
#define NOINLINE_COLD(...)                                                                                             \
  [&] [[gnu::noinline]] [[gnu::cold]] () -> decltype(auto) {                                                           \
    __VA_ARGS__                                                                                                        \
  }();

template<typename F>
struct CtorCall {
  CtorCall(F&& f) {
    std::forward<F>(f)();
  }
};

#define OUTLINE CtorCall CAT(ctorcall_, __LINE__) = [&] [[gnu::noinline]] () -> decltype(auto)

#define CHECK_NVRTC(x)                                                                                                 \
  {                                                                                                                    \
    nvrtcResult error__ = (x);                                                                                         \
    if (error__ != NVRTC_SUCCESS) [[unlikely]]                                                                         \
      NORETURN(throwNvrtc(error__, __FILE__, __LINE__);)                                                               \
  }
#define CHECK_CU(x)                                                                                                    \
  {                                                                                                                    \
    CUresult error__ = (x);                                                                                            \
    if (error__ != CUDA_SUCCESS) [[unlikely]]                                                                          \
      NORETURN(throwCu(error__, __FILE__, __LINE__);)                                                                  \
  }

#define CHECK_NVML(x)                                                                                                  \
  {                                                                                                                    \
    nvmlReturn_t error__ = (x);                                                                                        \
    if (error__ != NVML_SUCCESS) [[unlikely]]                                                                          \
      NORETURN(throwNvml(error__, __FILE__, __LINE__);)                                                                \
  }

#undef CHECK
#define CHECK(x)                                                                                                       \
  {                                                                                                                    \
    if (!(x)) [[unlikely]]                                                                                             \
      NORETURN(throwCheckFail(__FILE__, __LINE__, #x);)                                                                \
  }

} // namespace moodist
