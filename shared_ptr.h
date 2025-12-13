// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "commondefs.h"

namespace moodist {

namespace detail {

template<typename T, typename = void>
struct HasFetchAdd : std::false_type {};

template<typename T>
struct HasFetchAdd<T, std::void_t<decltype(std::declval<T>().fetch_add(1))>> : std::true_type {};

} // namespace detail

template<auto Member>
struct SharedPtrMemberPolicy {
  template<typename T>
  static void inc(T* ptr) noexcept {
    if constexpr (detail::HasFetchAdd<std::remove_reference_t<decltype(ptr->*Member)>>::value) {
      (ptr->*Member).fetch_add(1, std::memory_order_relaxed);
    } else {
      ++(ptr->*Member);
    }
  }

  template<typename T>
  static bool dec(T* ptr) noexcept {
    if constexpr (detail::HasFetchAdd<std::remove_reference_t<decltype(ptr->*Member)>>::value) {
      return (ptr->*Member).fetch_sub(1, std::memory_order_acq_rel) == 1;
    } else {
      return --(ptr->*Member) == 0;
    }
  }

  template<typename T, typename... Args>
  static T* create(Args&&... args) {
    T* ptr = internalNew<T>(std::forward<Args>(args)...);
    ptr->*Member = 1;
    return ptr;
  }

  template<typename T>
  static void destroy(T* ptr) noexcept {
    internalDelete(ptr);
  }
};

struct SharedPtrDefaultPolicy {
  template<typename T>
  static void inc(T* ptr) noexcept {
    if constexpr (detail::HasFetchAdd<decltype(ptr->refcount)>::value) {
      ptr->refcount.fetch_add(1, std::memory_order_relaxed);
    } else {
      ++ptr->refcount;
    }
  }

  template<typename T>
  static bool dec(T* ptr) noexcept {
    if constexpr (detail::HasFetchAdd<decltype(ptr->refcount)>::value) {
      return ptr->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1;
    } else {
      return --ptr->refcount == 0;
    }
  }

  template<typename T, typename... Args>
  static T* create(Args&&... args) {
    T* ptr = internalNew<T>(std::forward<Args>(args)...);
    ptr->refcount = 1;
    return ptr;
  }

  template<typename T>
  static void destroy(T* ptr) noexcept {
    internalDelete(ptr);
  }
};

template<typename T, typename Policy = SharedPtrDefaultPolicy>
struct SharedPtr {
private:
  T* ptr = nullptr;

public:
  SharedPtr() noexcept = default;
  SharedPtr(std::nullptr_t) noexcept {}

  explicit SharedPtr(T* p) noexcept : ptr(p) {}

  SharedPtr(const SharedPtr& o) noexcept : ptr(o.ptr) {
    if (ptr) {
      Policy::inc(ptr);
    }
  }

  SharedPtr(SharedPtr&& o) noexcept : ptr(o.ptr) {
    o.ptr = nullptr;
  }

  template<typename U>
  SharedPtr(const SharedPtr<U, Policy>& o) noexcept : ptr(o.get()) {
    if (ptr) {
      Policy::inc(ptr);
    }
  }

  template<typename U>
  SharedPtr(SharedPtr<U, Policy>&& o) noexcept : ptr(o.get()) {
    o.release();
  }

  ~SharedPtr() {
    if (ptr && Policy::dec(ptr)) {
      Policy::destroy(ptr);
    }
  }

  SharedPtr& operator=(const SharedPtr& o) noexcept {
    if (ptr != o.ptr) {
      if (ptr && Policy::dec(ptr)) {
        Policy::destroy(ptr);
      }
      ptr = o.ptr;
      if (ptr) {
        Policy::inc(ptr);
      }
    }
    return *this;
  }

  SharedPtr& operator=(SharedPtr&& o) noexcept {
    if (this != &o) {
      if (ptr && Policy::dec(ptr)) {
        Policy::destroy(ptr);
      }
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }

  SharedPtr& operator=(std::nullptr_t) noexcept {
    if (ptr && Policy::dec(ptr)) {
      Policy::destroy(ptr);
    }
    ptr = nullptr;
    return *this;
  }

  T* get() const noexcept {
    return ptr;
  }
  T* operator->() const noexcept {
    return ptr;
  }
  T& operator*() const noexcept {
    return *ptr;
  }

  explicit operator bool() const noexcept {
    return ptr != nullptr;
  }

  void reset() noexcept {
    if (ptr && Policy::dec(ptr)) {
      Policy::destroy(ptr);
    }
    ptr = nullptr;
  }

  void reset(T* p) noexcept {
    if (ptr && Policy::dec(ptr)) {
      Policy::destroy(ptr);
    }
    ptr = p;
  }

  T* release() noexcept {
    T* p = ptr;
    ptr = nullptr;
    return p;
  }

  void swap(SharedPtr& o) noexcept {
    std::swap(ptr, o.ptr);
  }

  bool operator==(const SharedPtr& o) const noexcept {
    return ptr == o.ptr;
  }
  bool operator!=(const SharedPtr& o) const noexcept {
    return ptr != o.ptr;
  }
  bool operator==(std::nullptr_t) const noexcept {
    return ptr == nullptr;
  }
  bool operator!=(std::nullptr_t) const noexcept {
    return ptr != nullptr;
  }
};

template<typename T, typename Policy = SharedPtrDefaultPolicy, typename... Args>
SharedPtr<T, Policy> makeShared(Args&&... args) {
  return SharedPtr<T, Policy>(Policy::template create<T>(std::forward<Args>(args)...));
}

template<typename T, typename Policy = SharedPtrDefaultPolicy>
SharedPtr<T, Policy> share(T* p) noexcept {
  Policy::inc(p);
  return SharedPtr<T, Policy>(p);
}

} // namespace moodist
