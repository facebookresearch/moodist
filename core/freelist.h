// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "commondefs.h"
#include "synchronization.h"

#include <utility>
#include <vector>

namespace moodist {

template<typename T>
struct FreeListTlsStorage {
  T* head = nullptr;
  size_t size = 0;
  static FreeListTlsStorage& get() {
    thread_local FreeListTlsStorage storage;
    return storage;
  }
};

template<typename T>
struct FreeListGlobalStorage {
  SpinMutex mutex;
  std::vector<std::pair<T*, size_t>, InternalAllocator<std::pair<T*, size_t>>> lists;
  static FreeListGlobalStorage& get() {
    static FreeListGlobalStorage* storage = internalNew<FreeListGlobalStorage>();
    return *storage;
  }
};

template<typename T>
struct FreeList {
  template<typename S>
  static auto load(S& s) {
    if constexpr (std::is_scalar_v<S>) {
      return s;
    } else {
      return s.load(std::memory_order_relaxed);
    }
  }
  template<typename D, typename S>
  static void store(D& d, S v) {
    if constexpr (std::is_scalar_v<D>) {
      d = v;
    } else {
      d.store(v, std::memory_order_relaxed);
    }
  }
  [[gnu::always_inline]] static void push(T* value, size_t maxThreadLocalEntries) {
    auto& tls = FreeListTlsStorage<T>::get();
    if (tls.size == maxThreadLocalEntries) {
      [[unlikely]];
      moveToGlobal(tls, maxThreadLocalEntries / 4u);
    }
    [[likely]];
    ++tls.size;
    T* next = std::exchange(tls.head, value);
    store(value->next, next);
  }
  [[gnu::always_inline]] static T* pop() {
    auto& tls = FreeListTlsStorage<T>::get();
    T* r = tls.head;
    if (r) {
      [[likely]];
      --tls.size;
      tls.head = (T*)load(r->next);
      return r;
    }
    [[unlikely]];
    return popSlow(tls);
  }

  [[gnu::noinline]] static T* popSlow(FreeListTlsStorage<T>& tls) {
    auto& global = FreeListGlobalStorage<T>::get();
    std::unique_lock l(global.mutex);
    if (global.lists.empty()) {
      [[unlikely]];
      return nullptr;
    }
    T* r;
    size_t newSize;
    std::tie(r, newSize) = global.lists.back();
    global.lists.pop_back();
    l.unlock();
    tls.head = (T*)load(r->next);
    tls.size = newSize - 1;
    return r;
  }

  [[gnu::noinline]] static void moveToGlobal(FreeListTlsStorage<T>& tls, size_t nToKeep) {
    T* head = tls.head;
    size_t newSize = 1;
    while (newSize < nToKeep) {
      T* next = (T*)load(head->next);
      head = next;
      ++newSize;
    }
    size_t oldSize = tls.size;
    tls.size = newSize;
    T* next = (T*)load(head->next);
    store(head->next, nullptr);
    auto& global = FreeListGlobalStorage<T>::get();
    std::lock_guard l(global.mutex);
    global.lists.emplace_back(next, oldSize - newSize);
  }
};

} // namespace moodist
