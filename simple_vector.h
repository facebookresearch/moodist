/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace moodist {

template<typename T, typename Allocator = std::allocator<T>>
struct SimpleVector {
  T* beginptr = nullptr;
  size_t msize = 0;
  using value_type = T;
  SimpleVector() = default;
  SimpleVector(const SimpleVector& n) {
    *this = n;
  }
  SimpleVector(SimpleVector&& n) {
    *this = std::move(n);
  }
  ~SimpleVector() {
    clear();
  }
  SimpleVector(std::initializer_list<T> list) {
    resize(list.size());
    size_t i = 0;
    for (auto& v : list) {
      (*this)[i] = std::move(v);
      ++i;
    }
  }
  SimpleVector& operator=(const SimpleVector& n) {
    resize(n.msize);
    for (size_t i = msize; i;) {
      --i;
      beginptr[i] = n.beginptr[i];
    }
    return *this;
  }
  SimpleVector& operator=(SimpleVector&& n) noexcept {
    std::swap(beginptr, n.beginptr);
    std::swap(msize, n.msize);
    return *this;
  }
  T& at(size_t index) {
    if (index >= msize) {
      throw std::out_of_range("SimpleVector::at out of range");
    }
    return beginptr[index];
  }
  const T& at(size_t index) const {
    if (index >= msize) {
      throw std::out_of_range("SimpleVector::at out of range");
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
    return beginptr + msize;
  }
  const T* begin() const {
    return beginptr;
  }
  const T* end() const {
    return beginptr + msize;
  }
  T& operator[](size_t index) {
    return beginptr[index];
  }
  const T& operator[](size_t index) const {
    return beginptr[index];
  }
  void clear() {
    if (msize) {
      for (size_t i = msize; i;) {
        --i;
        beginptr[i].~T();
      }
      Allocator().deallocate(beginptr, msize);
      msize = 0;
    }
  }
  void move(T* dst, T* begin, T* end) noexcept {
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
  void resize(size_t n) {
    if (msize > n) {
      T* i = beginptr + msize;
      T* e = beginptr + n;
      while (i != e) {
        --i;
        i->~T();
      }
    } else if (n > msize) {
      T* newbeginptr = Allocator().allocate(n);
      move(newbeginptr, beginptr, beginptr + msize);
      Allocator().deallocate(beginptr, msize);
      beginptr = newbeginptr;
      T* i = beginptr + msize;
      T* e = beginptr + n;
      while (i != e) {
        try {
          new (i) T();
        } catch (...) {
          msize = i - beginptr;
          throw;
        }
        ++i;
      }
    }
    msize = n;
  }
  bool empty() const {
    return msize == 0;
  }
  T& front() {
    return *beginptr;
  }
  T& back() {
    return beginptr[msize - 1];
  }
};

} // namespace moodist
