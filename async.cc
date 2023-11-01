/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "async.h"
#include "clock.h"
#include "synchronization.h"
#include "vector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <list>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <sched.h>

namespace moodist {
namespace async {

void setCurrentThreadName(std::string name) {
  if (name.size() > 15) {
    name[15] = 0;
  }
  pthread_setname_np(pthread_self(), name.c_str());
}

thread_local bool isThisAThread = false;

struct Thread {
  FunctionPointer f = nullptr;
  std::thread thread;
  int n = 0;
  std::atomic_bool terminate = false;
  template<typename WaitFunction>
  void entry(WaitFunction&& waitFunction) noexcept {
    isThisAThread = true;
    waitFunction(this);
    while (f) {
      try {
        Function<void()>{f}();
      } catch (const std::exception& e) {
        fflush(stdout);
        fprintf(stderr, "Unhandled exception in async function: %s\n", e.what());
        fflush(stderr);
        std::abort();
      }
      f = nullptr;
      waitFunction(this);
    };
  }
};

struct ThreadPool {
  alignas(64) mutable SpinMutex mutex;
  std::list<Thread> threads;
  std::atomic_size_t numThreads = 0;
  size_t maxThreads = 0;
  std::string name;
  ThreadPool(std::string name = "async") noexcept : name(std::move(name)) {
    maxThreads = std::thread::hardware_concurrency();
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
      int n = CPU_COUNT(&set);
      if (n > 0) {
        maxThreads = std::max(n - 1, 1);
      }
    }
    maxThreads = std::min(std::max(maxThreads, (size_t)1), (size_t)64);
  }
  ThreadPool(size_t maxThreads) : maxThreads(std::min(maxThreads, (size_t)64)) {}
  ~ThreadPool() {
    for (auto& v : threads) {
      v.thread.join();
    }
  }
  template<typename WaitFunction>
  Thread* addThread(Function<void()>& f, WaitFunction&& waitFunction) noexcept {
    std::unique_lock l(mutex);
    if (numThreads >= maxThreads) {
      return nullptr;
    }
    int n = ++numThreads;
    threads.emplace_back();
    auto* t = &threads.back();
    t->n = threads.size() - 1;
    t->f = f.release();
    t->thread = std::thread([this, n, t, waitFunction = std::forward<WaitFunction>(waitFunction)]() mutable {
      setCurrentThreadName(name + "-" + std::to_string(n));
      t->entry(std::move(waitFunction));
    });
    return t;
  }
  void setMaxThreads(size_t n) {
    std::unique_lock l(mutex);
    maxThreads = n;
  }
  size_t getMaxThreads() const {
    std::lock_guard l(mutex);
    return maxThreads;
  }
};

struct SchedulerImpl {
  // std::array<std::atomic<Thread*>, 64> threads{};

  struct alignas(64) Incoming {
    std::atomic<FunctionPointer> f;
    std::atomic_uint32_t req = 0;
  };
  struct alignas(64) Outgoing {
    std::atomic_uint32_t ack = 0;
    std::atomic_bool sleeping = false;
  };
  std::array<Incoming, 64> incoming;
  std::array<Outgoing, 64> outgoing;

  struct ThreadLocalCache {
    std::array<uint8_t, 65> prev;
    std::array<uint8_t, 65> next;
    ThreadLocalCache(size_t maxThreads) {
      for (auto& v : prev) {
        v = 0xff;
      }
      for (auto& v : next) {
        v = 0xff;
      }
      std::minstd_rand rng(std::hash<std::thread::id>()(std::this_thread::get_id()));
      std::array<uint8_t, 64> order;
      for (size_t i = 0; i != 64; ++i) {
        order[i] = i;
      }
      std::shuffle(order.begin(), order.begin() + maxThreads, rng);
      next[64] = order[0];
      prev[order[0]] = 64;
      auto cur = order[0];
      // printf(" headf is %d\n", cur);
      for (size_t i = 1; i != maxThreads; ++i) {
        auto n = order[i];
        // printf(" %d - %d\n", (int)i, n);
        prev[n] = cur;
        next[cur] = n;
        cur = n;
      }
      prev[64] = cur;
      next[cur] = 64;
      std::string s;
      size_t nn = 0;
      for (size_t i = next[64]; i != 64; i = next[i]) {
        if (!s.empty()) {
          s += ", ";
        }
        s += std::to_string(i);
        ++nn;
      }
      // printf("maxThreads %d, order -> %s\n", (int)maxThreads, s.c_str());
      if (nn != maxThreads) {
        std::abort();
      }
    }
  };

  std::atomic_size_t globalNextReadIndex = 0;
  std::atomic_size_t globalNextWriteIndex = 0;
  struct GlobalQueue {
    SpinMutex mutex;
    std::atomic_size_t size = 0;
    Vector<Function<void()>> queue;
  };
  std::array<GlobalQueue, 32> globalQueues;

  ThreadPool pool;

  SchedulerImpl() = default;
  SchedulerImpl(size_t nThreads) : pool(nThreads) {}

  ~SchedulerImpl() {
    for (auto& t : pool.threads) {
      t.terminate = true;
      if (outgoing[t.n].sleeping) {
        // fixme: since we don't actually modify req here, the futexWait below may not return.
        //        workaround is to let the wait below be short (1 second), such that on exit
        //        there is an upper bound to how long it might wait.
        futexWakeAll(&incoming[t.n].req);
      }
    }
  }

  void wait(Thread* t) noexcept {
    size_t index = t->n;
    auto& i = incoming[index];
    auto& o = outgoing[index];
    uint32_t ack = o.ack.load(std::memory_order_relaxed);
    uint32_t req = i.req.load(std::memory_order_acquire);
    size_t spinCount = 0;
    while (req == ack) {
      _mm_pause();
      if (++spinCount >= 1 << 12) {
        if (t->terminate.load(std::memory_order_relaxed)) {
          return;
        }
        o.sleeping = true;
        futexWait(&i.req, req, std::chrono::seconds(1));
        o.sleeping = false;
      }

      size_t readIndex = globalNextReadIndex.load(std::memory_order_relaxed);
      while (readIndex != globalNextWriteIndex.load(std::memory_order_relaxed)) {
        if (globalNextReadIndex.compare_exchange_weak(readIndex, readIndex + 1)) {
          auto& q = globalQueues[readIndex % globalQueues.size()];
          while (q.size.load(std::memory_order_relaxed) == 0) {
            _mm_pause();
            if (++spinCount > 1 << 12) {
              std::this_thread::yield();
            }
          }
          std::lock_guard l(q.mutex);
          t->f = q.queue.front().release();
          q.queue.pop_front();
          --q.size;
          return;
        }
      }
      req = i.req.load(std::memory_order_acquire);
    }
    do {
      t->f = i.f.exchange(nullptr, std::memory_order_relaxed);
    } while (!t->f);
    ++o.ack;
  }

  void scheduleGlobally(Function<void()> f) {
    size_t index = globalNextWriteIndex++;
    auto& q = globalQueues[index % globalQueues.size()];
    std::lock_guard l(q.mutex);
    q.queue.push_back(std::move(f));
    ++q.size;
  }

  void run(Function<void()> f) noexcept {
    thread_local ThreadLocalCache tlsCache(pool.maxThreads);
    auto& cache = tlsCache;

    for (auto index = cache.next[64]; index != 64; index = cache.next[index]) {
      auto& i = incoming[index];
      auto& o = outgoing[index];
      uint32_t req = i.req.load();
      uint32_t ack = o.ack.load();
      if (req == ack) {
        if (i.req.compare_exchange_strong(req, req + 1)) {
          i.f.store(f.release(), std::memory_order_relaxed);
          if (o.sleeping) {
            futexWakeAll(&i.req);
          }
          if (index != cache.next[64]) {
            auto prev = cache.prev[index];
            auto next = cache.next[index];
            cache.prev[next] = prev;
            cache.next[prev] = next;
            next = cache.next[64];
            cache.next[64] = index;
            cache.prev[index] = 64;
            cache.next[index] = next;
            cache.prev[next] = index;
          }

          while (pool.numThreads.load(std::memory_order_relaxed) <= index) {
            pool.addThread(f, [this](Thread* t) { wait(t); });
          }
          return;
        }
      }
    }

    scheduleGlobally(std::move(f));
  }
};

Scheduler::Scheduler() {
  impl_ = std::make_unique<SchedulerImpl>();
}
Scheduler::Scheduler(size_t nThreads) {
  impl_ = std::make_unique<SchedulerImpl>(nThreads);
}
Scheduler::~Scheduler() {}
void Scheduler::run(Function<void()> f) noexcept {
  impl_->run(std::move(f));
}
void Scheduler::setMaxThreads(size_t nThreads) {
  impl_->pool.setMaxThreads(nThreads);
}
size_t Scheduler::getMaxThreads() const {
  return impl_->pool.getMaxThreads();
}
bool Scheduler::isInThread() const noexcept {
  return isThisAThread;
}
void Scheduler::setName(std::string name) {
  impl_->pool.name = name;
}

} // namespace async
} // namespace moodist
