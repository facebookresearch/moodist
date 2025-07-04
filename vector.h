// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#pragma push_macro("assert")
#undef assert
// #define assert(x) (bool(x) ? 0 : (printf("assert failure %s:%d\n", __FILE__, __LINE__), std::abort(), 0))
#define assert(x)

namespace moodist {

template<typename T, typename Allocator = std::allocator<T>>
struct Vector {
  T* storagebegin = nullptr;
  T* storageend = nullptr;
  T* beginptr = nullptr;
  T* endptr = nullptr;
  size_t msize = 0;
  using value_type = T;
  Vector() = default;
  Vector(const Vector& n) {
    *this = n;
  }
  Vector(Vector&& n) {
    *this = std::move(n);
  }
  Vector(size_t count) {
    resize(count);
  }
  ~Vector() {
    if (beginptr != endptr) {
      clear();
    }
    if (storagebegin) {
      // std::free(storagebegin);
      Allocator().deallocate(storagebegin, storageend - storagebegin);
    }
  }
  Vector& operator=(const Vector& n) {
    clear();
    reserve(n.size());
    size_t k = n.size();
    for (size_t i = 0; i != k; ++i) {
      new (endptr) T(n[i]);
      ++endptr;
      ++msize;
    }
    return *this;
  }
  Vector& operator=(Vector&& n) {
    std::swap(storagebegin, n.storagebegin);
    std::swap(storageend, n.storageend);
    std::swap(beginptr, n.beginptr);
    std::swap(endptr, n.endptr);
    std::swap(msize, n.msize);
    return *this;
  }
  T& at(size_t index) {
    if (index >= msize) {
      throw std::out_of_range("Vector::at out of range");
    }
    return beginptr[index];
  }
  const T& at(size_t index) const {
    if (index >= msize) {
      throw std::out_of_range("Vector::at out of range");
    }
    return beginptr[index];
  }
  size_t size() const {
    return msize;
  }
  T* data() {
    return beginptr;
  }
  const T* data() const {
    return beginptr;
  }
  T* begin() {
    return beginptr;
  }
  T* end() {
    return endptr;
  }
  const T* begin() const {
    return beginptr;
  }
  const T* end() const {
    return endptr;
  }
  T& operator[](size_t index) {
    return beginptr[index];
  }
  const T& operator[](size_t index) const {
    return beginptr[index];
  }
  void clear() {
    for (auto* i = beginptr; i != endptr; ++i) {
      i->~T();
    }
    beginptr = storagebegin;
    endptr = beginptr;
    msize = 0;
  }
  void move(T* dst, T* begin, T* end) {
    if constexpr (std::is_trivially_copyable_v<T>) {
      std::memmove((void*)dst, (void*)begin, (end - begin) * sizeof(T));
    } else {
      if (dst <= begin) {
        for (auto* i = begin; i != end;) {
          *dst = std::move(*i);
          ++dst;
          ++i;
        }
      } else {
        auto* dsti = dst + (end - begin);
        for (auto* i = end; i != begin;) {
          --dsti;
          --i;
          *dsti = std::move(*i);
        }
      }
    }
  }
  void erase(T* begin, T* end) {
    size_t n = end - begin;
    msize -= n;
    if (begin == beginptr) {
      for (auto* i = begin; i != end; ++i) {
        i->~T();
      }
      beginptr = end;
      if (beginptr != endptr) {
        size_t unused = beginptr - storagebegin;
        if (unused > msize && unused >= 1024 * 512 / sizeof(T)) {
          if constexpr (std::is_trivially_copyable_v<T>) {
            move(storagebegin, beginptr, endptr);
          } else {
            auto* sbi = storagebegin;
            auto* bi = beginptr;
            while (sbi != beginptr && bi != endptr) {
              new (sbi) T(std::move(*bi));
              ++sbi;
              ++bi;
            }
            move(sbi, bi, endptr);
            for (auto* i = bi; i != endptr; ++i) {
              i->~T();
            }
          }
          beginptr = storagebegin;
          endptr = beginptr + msize;
        }
      }
    } else {
      move(begin, end, endptr);
      for (auto* i = endptr - n; i != endptr; ++i) {
        i->~T();
      }
      endptr -= n;
    }
    if (beginptr == endptr) {
      beginptr = storagebegin;
      endptr = beginptr;
    }
  }
  T* erase(T* at) {
    size_t index = at - beginptr;
    assert(index < msize);
    erase(at, at + 1);
    return beginptr + index;
  }
  void resize(size_t n) {
    if (msize > n) {
      T* i = endptr;
      T* e = beginptr + n;
      while (i != e) {
        --i;
        i->~T();
      }
    } else if (n > msize) {
      reserve(n);
      T* i = endptr;
      T* e = beginptr + n;
      while (i != e) {
        new (i) T();
        ++i;
      }
    }
    endptr = beginptr + n;
    msize = n;
  }
  bool empty() const {
    return msize == 0;
  }
  size_t capacity() {
    return storageend - beginptr;
  }
  void reserveImpl(size_t n) {
    auto* lbegin = beginptr;
    auto* lend = endptr;
    auto* prevstorage = storagebegin;
    size_t msize = this->msize;
    T* newptr = Allocator().allocate(n);
    if (prevstorage) {
      if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(newptr, lbegin, sizeof(T) * msize);
      } else {
        T* dst = newptr;
        for (T* i = lbegin; i != lend; ++i) {
          new (dst) T(std::move(*i));
          i->~T();
          ++dst;
        }
      }
      Allocator().deallocate(prevstorage, storageend - prevstorage);
    }
    storagebegin = newptr;
    storageend = newptr + n;
    beginptr = newptr;
    endptr = newptr + msize;
  }
  void reserve(size_t n) {
    if (n <= capacity()) {
      return;
    }
    reserveImpl(n);
  }
  void expand() {
    reserveImpl(std::max(capacity() * 2, (size_t)16));
  }
  void push_back(const T& v) {
    emplace_back(v);
  }
  void push_back(T&& v) {
    emplace_back(std::move(v));
  }
  template<typename... Args>
  void emplace_back(Args&&... args) {
    if (endptr == storageend) {
      if (capacity() != size()) {
        __builtin_unreachable();
      }
      [[unlikely]];
      expand();
    }
    new (endptr) T(std::forward<Args>(args)...);
    ++endptr;
    ++msize;
  }
  T& front() {
    assert(msize > 0);
    return *beginptr;
  }
  T& back() {
    assert(msize > 0);
    return endptr[-1];
  }
  const T& front() const {
    assert(msize > 0);
    return *beginptr;
  }
  const T& back() const {
    assert(msize > 0);
    return endptr[-1];
  }
  void pop_back() {
    assert(msize > 0);
    --endptr;
    --msize;
    endptr->~T();
  }
  void pop_front() {
    assert(msize > 0);
    erase(beginptr, beginptr + 1);
  }
  T pop_back_value() {
    T r = std::move(back());
    pop_back();
    return r;
  }
  T pop_front_value() {
    T r = std::move(front());
    pop_front();
    return r;
  }
  template<typename V>
  T* insert(T* at, V&& value) {
    if (at == endptr) {
      push_back(std::forward<V>(value));
      return &back();
    }
    if (endptr == storageend) {
      if (capacity() != size()) {
        __builtin_unreachable();
      }
      [[unlikely]];
      size_t index = at - beginptr;
      expand();
      at = beginptr + index;
    }
    new (endptr) T();
    move(at + 1, at, endptr);
    *at = std::forward<V>(value);
    ++endptr;
    ++msize;
    return at;
  }
  template<typename... Args>
  T* emplace(T* at, Args&&... args) {
    if (at == endptr) {
      emplace_back(std::forward<Args>(args)...);
      return &back();
    }
    if (endptr == storageend) {
      if (capacity() != size()) {
        __builtin_unreachable();
      }
      [[unlikely]];
      size_t index = at - beginptr;
      expand();
      at = beginptr + index;
    }
    T tmp(std::forward<Args>(args)...);
    new (endptr) T();
    move(at + 1, at, endptr);
    *at = std::move(tmp);
    ++endptr;
    ++msize;
    return at;
  }
};

} // namespace moodist

#pragma pop_macro("assert")
