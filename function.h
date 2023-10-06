/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "freelist.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace moodist {

namespace function {

namespace impl {

struct OpsBase {
  const size_t size = 0;
};

template<typename F, typename R, typename... Args>
struct TemplatedStorage;

struct Storage {
  union {
    Storage* next = nullptr;
    std::atomic<Storage*> nextAtomic;
    static_assert(std::is_standard_layout_v<decltype(nextAtomic)>);
    static_assert(std::is_trivially_destructible_v<decltype(nextAtomic)>);
  };
  void* callPtr = nullptr;
  const OpsBase* ops = nullptr;
  template<typename F, typename R, typename... Args>
  F& as() {
    return (F&)((TemplatedStorage<F, R, Args...>*)this)->f;
  }
};

template<typename F, typename R, typename... Args>
struct OpsConstructor;

template<typename F, typename R, typename... Args>
struct TemplatedStorage : Storage {
  std::aligned_storage_t<sizeof(F), alignof(F)> f;

  static TemplatedStorage* pop() {
    TemplatedStorage* r = FreeList<TemplatedStorage>::pop();
    if (r) {
      [[likely]];
      return r;
    } else {
      [[unlikely]];
      r = new TemplatedStorage();
      auto* ops = &OpsConstructor<F, R, Args...>::value;
      r->ops = ops;
      r->callPtr = (void*)ops->call;
      return r;
    }
  }
  static void push(TemplatedStorage* storage) {
    FreeList<TemplatedStorage>::push(storage, std::max((size_t)65536 / sizeof(TemplatedStorage), (size_t)64));
  }
};

template<typename T>
struct ArgType {
  using type = typename std::conditional_t<std::is_reference_v<T>, T, T&&>;
};

template<typename R, typename... Args>
struct Ops : OpsBase {
  R (*call)(Storage&, Args&&...) = nullptr;
  R (*callAndDtor)(Storage&, Args&&...) = nullptr;
  Storage* (*copyCtor)(Storage&) = nullptr;
  void (*copy)(Storage&, Storage&) = nullptr;
  void (*dtor)(Storage&) = nullptr;
};

template<typename F, typename R, typename... Args>
struct OpsConstructor {
  static constexpr Ops<R, Args...> make() {
    Ops<R, Args...> r{{sizeof(F)}};
    r.call = [](Storage& s, Args&&... args) {
      // return std::invoke(s.as<F>(), std::forward<Args>(args)...);
      return s.template as<F, R, Args...>()(std::forward<Args>(args)...);
    };
    r.callAndDtor = [](Storage& s, Args&&... args) {
      F& f = s.template as<F, R, Args...>();
      if constexpr (std::is_same_v<R, void>) {
        f(std::forward<Args>(args)...);
        f.~F();
        TemplatedStorage<F, R, Args...>::push((TemplatedStorage<F, R, Args...>*)&s);
      } else {
        auto r = f(std::forward<Args>(args)...);
        f.~F();
        TemplatedStorage<F, R, Args...>::push((TemplatedStorage<F, R, Args...>*)&s);
        return r;
      }
    };
    r.dtor = [](Storage& s) {
      s.template as<F, R, Args...>().~F();
      TemplatedStorage<F, R, Args...>::push((TemplatedStorage<F, R, Args...>*)&s);
    };
    r.copy = [](Storage& to, Storage& from) {
      if constexpr (std::is_copy_assignable_v<F>) {
        to.template as<F, R, Args...>() = from.template as<F, R, Args...>();
      } else {
        throw std::runtime_error("function is not copy assignable");
      }
    };
    r.copyCtor = [](Storage& from) -> Storage* {
      if constexpr (std::is_copy_constructible_v<F>) {
        Storage* r = TemplatedStorage<F, R, Args...>::pop();
        new (&r->template as<F, R, Args...>()) F(from.template as<F, R, Args...>());
        return r;
      } else {
        throw std::runtime_error("function is not copy constructible");
      }
    };
    return r;
  }
  static constexpr Ops<R, Args...> value = make();
};
template<typename R, typename... Args>
struct NullOps {
  static constexpr Ops<R, Args...> value{};
};

using FunctionPointer = Storage*;

template<typename T>
class Function;
template<typename R, typename... Args>
class Function<R(Args...)> {
  Storage* storage = nullptr;

public:
  Function() = default;
  Function(const Function& n) {
    *this = n;
  }
  Function(Function&& n) noexcept {
    std::swap(storage, n.storage);
  }
  Function(std::nullptr_t) noexcept {}
  Function(FunctionPointer ptr) noexcept {
    if (ptr) {
      storage = ptr;
    }
  }

  template<typename F>
  Function(F&& f) {
    *this = std::forward<F>(f);
  }

  ~Function() {
    *this = nullptr;
  }

  using CallType = R (*)(Storage&, Args&&...);
  CallType getCall() const {
    return (CallType)storage->callPtr;
  }
  const Ops<R, Args...>* getOps() const {
    return (const Ops<R, Args...>*)storage->ops;
  }

  FunctionPointer release() noexcept {
    return std::exchange(storage, nullptr);
  }

  Function& operator=(FunctionPointer ptr) noexcept {
    *this = nullptr;
    if (ptr) {
      storage = ptr;
    }
    return *this;
  }

  R operator()(Args... args) const& {
    return getCall()(*storage, std::forward<Args>(args)...);
  }

  R operator()(Args... args) && {
    auto* storage = this->storage;
    if constexpr (std::is_same_v<std::decay_t<R>, void>) {
      getOps()->callAndDtor(*storage, std::forward<Args>(args)...);
      this->storage = nullptr;
    } else {
      auto r = getOps()->callAndDtor(*storage, std::forward<Args>(args)...);
      this->storage = nullptr;
      return r;
    }
  }

  Function& operator=(const Function& n) {
    if (!n.storage) {
      *this = nullptr;
    } else if (!storage) {
      storage = n.getOps()->copyCtor(*n.storage);
    } else if (storage->ops == n.storage->ops) {
      getOps()->copy(*storage, *n.storage);
    } else {
      *this = nullptr;
      storage = n.getOps()->copyCtor(*n.storage);
    }
    return *this;
  }
  Function& operator=(Function&& n) noexcept {
    std::swap(storage, n.storage);
    return *this;
  }

  Function& operator=(std::nullptr_t) noexcept {
    if (storage) {
      auto* ops = getOps();
      auto* storage = std::exchange(this->storage, nullptr);
      ops->dtor(*storage);
    }
    return *this;
  }

  template<typename F, std::enable_if_t<!std::is_same_v<std::remove_reference_t<F>, Function>>* = nullptr>
  Function& operator=(F&& f) {
    if (storage) {
      auto* ops = getOps();
      if (ops->dtor) {
        ops->dtor(*storage);
      }
    }
    using FT = std::remove_reference_t<F>;
    auto* newStorage = TemplatedStorage<FT, R, Args...>::pop();
    try {
      new (&newStorage->template as<FT, R, Args...>()) FT(std::forward<F>(f));
    } catch (...) {
      TemplatedStorage<FT, R, Args...>::push(newStorage);
      throw;
    }
    this->storage = newStorage;
    return *this;
  }

  explicit operator bool() const noexcept {
    return storage != nullptr;
  }

  template<typename T>
  T& as() const noexcept {
    return storage->as<T>();
  }

  bool operator==(FunctionPointer f) const noexcept {
    return storage == f;
  }
  bool operator!=(FunctionPointer f) const noexcept {
    return storage != f;
  }
};

} // namespace impl

using FunctionPointer = impl::FunctionPointer;
template<typename T>
using Function = impl::Function<T>;

} // namespace function

using FunctionPointer = function::FunctionPointer;
template<typename T>
using Function = function::Function<T>;

} // namespace moodist
