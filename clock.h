/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include <x86intrin.h>

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define IS_USING_TSAN
#endif
#endif

namespace moodist {

inline struct Timekeeper {
  std::atomic_int64_t tscThreshold = 0;
  std::atomic_int64_t tscDivisor = 0;
  unsigned __int128 prevVals = 0;
  int64_t lastBenchmarkTime = 0;
  int64_t lastBenchmarkTsc = 0;
  std::atomic_int64_t count = 0;
  std::atomic_bool mutex = false;

  [[gnu::noinline]] int64_t longPath(int64_t tsc) {
    int64_t now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (mutex.exchange(true)) {
      return now;
    }
    unsigned __int128 newVal = ((unsigned __int128)now << 64) | ((unsigned __int128)tsc);
#ifdef IS_USING_TSAN
    unsigned __int128 prev = __sync_val_compare_and_swap_16(&prevVals, 0, 0);
#else
    unsigned __int128 prev = prevVals;
#endif
    __sync_val_compare_and_swap_16(&prevVals, prev, newVal);
    const int64_t benchmarkInterval = 1000000000;
    const int64_t resetTimeInterval = 100000000;
    if (now - lastBenchmarkTime >= (tscThreshold ? benchmarkInterval : benchmarkInterval / 10)) {
      int64_t pt = lastBenchmarkTime;
      int64_t pttsc = lastBenchmarkTsc;
      lastBenchmarkTime = now;
      lastBenchmarkTsc = tsc;
      if (pttsc) {
        int64_t tscDiff = tsc - pttsc;
        int64_t timeDiff = now - pt;
        if (tscDiff > 0 && timeDiff > 0) {
          uint64_t a = tscDiff;
          uint64_t b = timeDiff;
          uint64_t x = (a << 16) / b;
          if (a << 16 >> 16 == a) {
            tscDivisor = x;
            tscThreshold =
                std::min(tscDivisor * resetTimeInterval / 65536, std::numeric_limits<int64_t>::max() / 65536);
          } else {
            tscThreshold = 0;
          }
        }
      }
    }
    mutex.store(false, std::memory_order_release);
    return now;
  }

  int64_t now() {
#ifdef IS_USING_TSAN
    unsigned __int128 vals = __sync_val_compare_and_swap_16(&prevVals, 0, 0);
    int64_t prevTime;
    int64_t prevTimeTsc;
    prevTimeTsc = vals;
    prevTime = vals >> 64;
#else
    int64_t prevTime;
    int64_t prevTimeTsc;
    do {
      prevTimeTsc = prevVals;
      prevTime = prevVals >> 64;
      __sync_synchronize();
    } while ((int64_t)prevVals != prevTimeTsc);
#endif
    int64_t tsc = __rdtsc();
    int64_t tscPassed = tsc - prevTimeTsc;
    auto threshold = tscThreshold.load(std::memory_order_relaxed);
    if (threshold > 0 & tscPassed < threshold) {
      [[likely]];
      return prevTime + ((uint64_t)std::max(tscPassed, (int64_t)1) * (uint64_t)65536) /
                            (uint64_t)tscDivisor.load(std::memory_order_relaxed);
    }
    [[unlikely]];
    return longPath(tsc);
  }
} timekeeper;

struct FastClock {
  using duration = std::chrono::nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<FastClock, duration>;
  static constexpr bool is_steady = true;

  static time_point now() noexcept {
    return time_point(std::chrono::nanoseconds(timekeeper.now()));
  }
};

using Clock = FastClock;
// using Clock = std::chrono::steady_clock;

} // namespace rpc
