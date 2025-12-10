#pragma once

#include "ib_common.h"
#include "intrusive_list.h"
#include "tcpdev.h"

#include <memory>

namespace moodist {

namespace smallfunction {

struct SmallFunction;

inline constexpr struct Ops {
  void (*call)(SmallFunction*) = nullptr;
  void (*dtor)(SmallFunction*) = nullptr;
  void (*callAndDtor)(SmallFunction*) = nullptr;
} nullops;
template<typename SF, typename F>
struct TemplatedOps {
  void (*call)(SmallFunction*) = [](SF* self) {
    ((F&)self->storage)();
  };
  void (*dtor)(SmallFunction*) = std::is_trivially_destructible_v<F> ? (void (*)(SmallFunction*)) nullptr
                                                                     : (void (*)(SmallFunction*))[](SF * self) {
    ((F&)self->storage).~F();
  };
  void (*callAndDtor)(SmallFunction*) = [](SF* self) {
    ((F&)self->storage)();
    ((F&)self->storage).~F();
  };
};

template<typename T>
inline constexpr TemplatedOps<SmallFunction, T> TOps;

struct alignas(std::max_align_t) SmallFunction {
  AlignedStorage<0x38, alignof(void*)> storage;
  const Ops* ops = &nullops;

  SmallFunction& operator=(std::nullptr_t) {
    if (ops->dtor) {
      ops->dtor(this);
    }
    ops = &nullops;
    return *this;
  }

  ~SmallFunction() {
    if (ops->dtor) {
      ops->dtor(this);
    }
  }
  SmallFunction() = default;
  SmallFunction(const SmallFunction&) = delete;
  SmallFunction& operator=(const SmallFunction&) = delete;

  template<typename F>
  SmallFunction& operator=(F&& f) {
    static_assert(sizeof(F) <= sizeof(storage));
    if (ops->dtor) {
      ops->dtor(this);
    }
    new (&storage) F(std::forward<F>(f));
    ops = (Ops*)&TOps<F>;
    return *this;
  }

  void operator()() & {
    return ops->call(this);
  }

  void operator()() && {
    ops->callAndDtor(this);
    ops = &nullops;
  }

  operator bool() const {
    return ops != &nullops;
  }

  bool operator==(std::nullptr_t) const {
    return ops == &nullops;
  }
};

} // namespace smallfunction

using smallfunction::SmallFunction;

struct RdmaCallback {
  IntrusiveListLink<RdmaCallback> link;
  SmallFunction onComplete;
  size_t refcount = 0;
  RdmaCallback* next = nullptr;

  size_t* counter = nullptr;
  size_t i = -1;

  void decref() {
    CHECK(refcount >= 1);
    if (--refcount == 0) {
      if (onComplete) {
        std::move(onComplete)();
      } else if (counter) {
        --*counter;
      }
      FreeList<RdmaCallback>::push(this, 0x1000);
    }
  }
};

struct RdmaCallbackWrapper {
  RdmaCallback* callback = nullptr;
  RdmaCallbackWrapper() = default;
  RdmaCallbackWrapper(std::nullptr_t) {}
  RdmaCallbackWrapper(RdmaCallback* callback) : callback(callback) {
    if (callback) {
      ++callback->refcount;
    }
  }
  ~RdmaCallbackWrapper() {
    if (callback) {
      if (callback->refcount > 1) [[likely]] {
        --callback->refcount;
      } else {
        callback->decref();
      }
    }
  }
  operator RdmaCallback*() {
    return callback;
  }
  RdmaCallback* operator->() {
    return callback;
  }
  RdmaCallback& operator*() {
    return *callback;
  }
};

template<typename F>
RdmaCallbackWrapper makeRdmaCallback(F&& f) {
  RdmaCallback* callback = FreeList<RdmaCallback>::pop();
  if (!callback) {
    // callback = new Callback();
    callback = new (internalAlloc(sizeof(RdmaCallback))) RdmaCallback();
  }
  CHECK(callback->refcount == 0);
  CHECK(callback->onComplete == nullptr);
  callback->onComplete = std::move(f);
  callback->counter = nullptr;
  callback->i = -1;
  return RdmaCallbackWrapper(callback);
}

struct RdmaMr {
  uint32_t lkey;
  uint32_t rkey;

  virtual ~RdmaMr() {}
};

struct Rdma;

struct RdmaCpuThreadApi {
  void* impl = nullptr;
  void* context = nullptr;
  void setActive();
  void setInactive();
  std::string groupName();
  std::string rankName(size_t i);
  void setError();
  bool suppressErrors();
};

struct Rdma {
  const char* type = nullptr;

  virtual ~Rdma() {}
  virtual void setCpuThreadApi(RdmaCpuThreadApi cpuThreadApi) = 0;
  virtual void poll() = 0;
  virtual void postWrite(size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback, bool allowInline) = 0;
  virtual void postRead(size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback) = 0;
  virtual void postSend(size_t i, void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) = 0;
  virtual void postRecv(void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) = 0;
  virtual void stopReceives() = 0;
  virtual bool mrCompatible(Rdma* other) = 0;
  virtual std::unique_ptr<RdmaMr> regMrCpu(void* address, size_t bytes) = 0;
  virtual std::unique_ptr<RdmaMr> regMrCuda(void* address, size_t bytes) = 0;
  constexpr virtual bool supportsCuda() const = 0;
  virtual void close() = 0;
};

std::unique_ptr<Rdma> makeRdmaIb(Group* group, std::unique_ptr<IbCommon> ib);
std::unique_ptr<Rdma> makeRdmaTcp(Group* group);

} // namespace moodist
