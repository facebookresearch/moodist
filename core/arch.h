// Copyright (c) Meta Platforms, Inc. and affiliates.

// Architecture-specific intrinsics abstraction.
// Provides portable wrappers for x86/aarch64 intrinsics.

#pragma once

#include <cstdint>

#if defined(__x86_64__)
#include <x86intrin.h>
#elif defined(__aarch64__)
// No special header needed - we use inline asm / builtins
#else
#error "Unsupported architecture"
#endif

namespace moodist {

inline void cpu_pause() {
#if defined(__x86_64__)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield");
#endif
}

inline uint64_t rdtsc() {
#if defined(__x86_64__)
  return __rdtsc();
#elif defined(__aarch64__)
  uint64_t val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
#endif
}

inline uint64_t lzcnt64(uint64_t x) {
#if defined(__x86_64__)
  return __builtin_ia32_lzcnt_u64(x);
#elif defined(__aarch64__)
  return __builtin_clzll(x);
#endif
}

} // namespace moodist
