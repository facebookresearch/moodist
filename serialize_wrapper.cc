// Copyright (c) Meta Platforms, Inc. and affiliates.

// Wrapper for serialize/deserialize that uses torch::Tensor.
// Uses function pointers from coreAPI to call into libmoodist.so.

#include "moodist_loader.h"
#include "torch_includes.h"

#include <pybind11/pybind11.h>

namespace moodist {

namespace py = pybind11;

// RAII wrapper for Buffer* - copyable via refcount
class BufferGuard {
  Buffer* buf_;

public:
  explicit BufferGuard(Buffer* buf) : buf_(buf) {}
  ~BufferGuard() {
    if (buf_) {
      coreAPI.serializeBufferDecRef(buf_);
    }
  }

  BufferGuard(const BufferGuard& other) : buf_(other.buf_) {
    if (buf_) {
      coreAPI.serializeBufferAddRef(buf_);
    }
  }
  BufferGuard& operator=(const BufferGuard& other) {
    if (this != &other) {
      if (buf_) {
        coreAPI.serializeBufferDecRef(buf_);
      }
      buf_ = other.buf_;
      if (buf_) {
        coreAPI.serializeBufferAddRef(buf_);
      }
    }
    return *this;
  }

  BufferGuard(BufferGuard&& other) noexcept : buf_(other.buf_) {
    other.buf_ = nullptr;
  }
  BufferGuard& operator=(BufferGuard&& other) noexcept {
    if (this != &other) {
      if (buf_) {
        coreAPI.serializeBufferDecRef(buf_);
      }
      buf_ = other.buf_;
      other.buf_ = nullptr;
    }
    return *this;
  }

  Buffer* get() const {
    return buf_;
  }

  void reset() {
    if (buf_) {
      coreAPI.serializeBufferDecRef(buf_);
      buf_ = nullptr;
    }
  }
};

torch::Tensor serializeObject(py::object o) {
  BufferGuard guard(coreAPI.serializeObjectImpl(o.ptr()));
  void* ptr = coreAPI.serializeBufferPtr(guard.get());
  size_t size = coreAPI.serializeBufferSize(guard.get());

  // Capture guard by value - copy increments refcount
  // Explicitly reset in callback to free exactly when deleter is called
  return torch::from_blob(
      ptr, {static_cast<int64_t>(size)},
      [guard](void*) mutable {
        guard.reset();
      },
      torch::TensorOptions().dtype(torch::kUInt8));
}

py::object deserializeObject(torch::Tensor t) {
  PyObject* result = coreAPI.deserializeObjectImpl(t.data_ptr(), t.itemsize() * t.numel());
  // reinterpret_steal takes ownership of the new reference
  return py::reinterpret_steal<py::object>(result);
}

} // namespace moodist
