/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "clock.h"

#include <atomic>
#include <climits>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include <linux/futex.h>
#include <semaphore.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <x86intrin.h>

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define IS_USING_TSAN
#endif
#endif

#ifdef IS_USING_TSAN
#include <sanitizer/tsan_interface.h>
#else

#define __tsan_mutex_pre_lock(...)
#define __tsan_mutex_post_lock(...)
#define __tsan_mutex_pre_unlock(...)
#define __tsan_mutex_post_unlock(...)

#endif

namespace moodist {

inline void futexWakeAll(std::atomic_uint32_t* futex) {
  syscall(SYS_futex, (void*)futex, FUTEX_WAKE, INT_MAX, nullptr);
}

inline void futexWait(std::atomic_uint32_t* futex, uint32_t expected, std::chrono::nanoseconds timeout) {
  auto nanoseconds = timeout;
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(nanoseconds);
  timespec ts;
  ts.tv_sec = seconds.count();
  ts.tv_nsec = (nanoseconds - seconds).count();
  syscall(SYS_futex, (void*)futex, FUTEX_WAIT, expected, (void*)&ts);
}

inline void futexWaitWhileLess(std::atomic_uint32_t* futex, uint32_t target) {
  uint32_t value = futex->load();
  while (value < target) {
    futexWait(futex, value, std::chrono::seconds(1));
    value = futex->load();
  }
}

inline void futexWaitWhileLessDebug(const char* fn, int line, std::atomic_uint32_t* futex, uint32_t target) {
  auto start = std::chrono::steady_clock::now();
  uint32_t value = futex->load();
  while (value < target) {
    futexWait(futex, value, std::chrono::milliseconds(1));
    value = futex->load();
    if (std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(500)) {
      printf("deadlocked on futex wait at %s:%d\n", fn, line);
    }
  }
}

// #define futexWaitWhileLess(a, b) futexWaitWhileLessDebug(__FILE__, __LINE__, a, b)

#if 0
using SpinMutex = std::mutex;
#elif 0
inline std::atomic_int mutexThreadIdCounter = 0;
inline thread_local int mutexThreadId = mutexThreadIdCounter++;
class SpinMutex {
  int magic = 0x42;
  std::atomic<bool> locked_{false};
  std::atomic<int*> owner = nullptr;
  std::atomic<void*> ownerAddress = nullptr;

public:
  void lock() {
    if (owner == &mutexThreadId) {
      printf("recursive lock\n");
      std::abort();
    }
    if (magic != 0x42) {
      printf("BAD MUTEX MAGIC\n");
      std::abort();
    }
    auto start = Clock::now();
    do {
      while (locked_.load(std::memory_order_acquire)) {
        _mm_pause();
        if (magic != 0x42) {
          printf("BAD MUTEX MAGIC\n");
          std::abort();
        }
        if (Clock::now() - start >= std::chrono::milliseconds(100)) {
          int* p = owner.load();
          printf(
              "deadlock detected in thread %d! held by thread %d (my return address %p, owner return address %p, "
              "&mutexThreadIdCounter is %p)\n",
              mutexThreadId, p ? *p : -1, __builtin_return_address(0), ownerAddress.load(),
              (void*)&mutexThreadIdCounter);
          start = Clock::now();
        }
      }
    } while (locked_.exchange(true, std::memory_order_acq_rel));
    owner = &mutexThreadId;
    ownerAddress = __builtin_return_address(0);
  }
  void unlock() {
    if (magic != 0x42) {
      printf("BAD MUTEX MAGIC\n");
      std::abort();
    }
    owner = nullptr;
    locked_.store(false);
  }
  bool try_lock() {
    if (owner == &mutexThreadId) {
      printf("recursive try_lock\n");
      std::abort();
    }
    if (locked_.load(std::memory_order_acquire)) {
      return false;
    }
    bool r = !locked_.exchange(true, std::memory_order_acq_rel);
    if (r) {
      owner = &mutexThreadId;
    }
    return r;
  }
};
#else
class SpinMutex {
  std::atomic<bool> locked = false;

public:
  void lock() {
    __tsan_mutex_pre_lock(this, 0);
    do {
      while (locked.load(std::memory_order_relaxed)) {
        _mm_pause();
      }
    } while (locked.exchange(true, std::memory_order_acq_rel));
    __tsan_mutex_post_lock(this, 0, 0);
  }
  void unlock() {
    __tsan_mutex_pre_unlock(this, 0);
    locked.store(false, std::memory_order_release);
    __tsan_mutex_post_unlock(this, 0);
  }
  bool try_lock() {
    __tsan_mutex_pre_lock(this, __tsan_mutex_try_lock);
    if (locked.load(std::memory_order_relaxed)) {
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
      return false;
    }
    if (!locked.exchange(true, std::memory_order_acq_rel)) {
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock, 0);
      return true;
    } else {
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
      return false;
    }
  }
};
#endif

#if 0
using SharedSpinMutex = std::shared_mutex;
#else
class SharedSpinMutex {
  std::atomic_bool locked = false;
  std::atomic_int shareCount = 0;

public:
  void lock() {
    __tsan_mutex_pre_lock(this, 0);
    do {
      while (locked.load(std::memory_order_relaxed)) {
        _mm_pause();
      }
    } while (locked.exchange(true));
    while (shareCount.load()) {
      _mm_pause();
    }
    __tsan_mutex_post_lock(this, 0, 0);
  }
  void unlock() {
    __tsan_mutex_pre_unlock(this, 0);
    locked.store(false, std::memory_order_release);
    __tsan_mutex_post_unlock(this, 0);
  }
  bool try_lock() {
    __tsan_mutex_pre_lock(this, __tsan_mutex_try_lock);
    if (locked.load(std::memory_order_relaxed)) {
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
      return false;
    }
    if (shareCount.load(std::memory_order_relaxed) == 0 && !locked.exchange(true, std::memory_order_acq_rel)) {
      if (shareCount.load(std::memory_order_relaxed) == 0) {
        __tsan_mutex_post_lock(this, __tsan_mutex_try_lock, 0);
        return true;
      } else {
        locked.store(false, std::memory_order_release);
      }
    }
    __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
    return false;
  }
  void lock_shared() {
    __tsan_mutex_pre_lock(this, __tsan_mutex_read_lock);
    while (true) {
      while (locked.load(std::memory_order_relaxed)) {
        _mm_pause();
      }
      shareCount.fetch_add(1);
      if (locked.load()) {
        shareCount.fetch_sub(1, std::memory_order_acq_rel);
      } else {
        break;
      }
    }
    __tsan_mutex_post_lock(this, __tsan_mutex_read_lock, 0);
  }
  void unlock_shared() {
    __tsan_mutex_pre_unlock(this, __tsan_mutex_read_lock);
    shareCount.fetch_sub(1, std::memory_order_release);
    __tsan_mutex_post_unlock(this, __tsan_mutex_read_lock);
  }
  bool try_lock_shared() {
    __tsan_mutex_pre_lock(this, __tsan_mutex_try_lock | __tsan_mutex_read_lock);
    if (locked.load(std::memory_order_relaxed)) {
      return false;
    }
    shareCount.fetch_add(1);
    if (locked.load()) {
      shareCount.fetch_sub(1, std::memory_order_acq_rel);
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed | __tsan_mutex_read_lock, 0);
      return false;
    }
    __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_read_lock, 0);
    return true;
  }
};
#endif

class Semaphore {
  sem_t sem;

public:
  Semaphore() noexcept {
    sem_init(&sem, 0, 0);
  }
  ~Semaphore() {
    sem_destroy(&sem);
  }
  void post() noexcept {
    sem_post(&sem);
  }
  void wait() noexcept {
    while (sem_wait(&sem)) {
      if (errno != EINTR) {
        printf("sem_wait returned errno %d", (int)errno);
        std::abort();
      }
    }
  }
  template<typename Rep, typename Period>
  void wait_for(const std::chrono::duration<Rep, Period>& duration) noexcept {
    struct timespec ts;
    auto absduration = std::chrono::system_clock::now().time_since_epoch() + duration;
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(absduration);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(nanoseconds);
    ts.tv_sec = seconds.count();
    ts.tv_nsec = (nanoseconds - seconds).count();
    while (sem_timedwait(&sem, &ts)) {
      if (errno == ETIMEDOUT) {
        break;
      }
      if (errno != EINTR) {
        printf("sem_timedwait returned errno %d", (int)errno);
        std::abort();
      }
    }
  }
  template<typename Clock, typename Duration>
  void wait_until(const std::chrono::time_point<Clock, Duration>& timePoint) noexcept {
    wait_for(timePoint - Clock::now());
  }

  Semaphore(const Semaphore&) = delete;
  Semaphore(const Semaphore&&) = delete;
  Semaphore& operator=(const Semaphore&) = delete;
  Semaphore& operator=(const Semaphore&&) = delete;
};

} // namespace moodist
