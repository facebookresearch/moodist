// Copyright (c) Meta Platforms, Inc. and affiliates.

// Wrapper for serialize/deserialize that uses torch::Tensor.

#include "serialize_api.h"
#include "torch_includes.h"

#include <pybind11/pybind11.h>

namespace moodist {

namespace py = pybind11;

// RAII wrapper for Buffer* - movable so it can be captured by lambda
class BufferGuard {
  Buffer* buf_;
public:
  explicit BufferGuard(Buffer* buf) : buf_(buf) {}
  ~BufferGuard() { if (buf_) serializeBufferDecRef(buf_); }

  BufferGuard(BufferGuard&& other) noexcept : buf_(other.buf_) { other.buf_ = nullptr; }
  BufferGuard& operator=(BufferGuard&& other) noexcept {
    if (this != &other) {
      if (buf_) serializeBufferDecRef(buf_);
      buf_ = other.buf_;
      other.buf_ = nullptr;
    }
    return *this;
  }

  BufferGuard(const BufferGuard&) = delete;
  BufferGuard& operator=(const BufferGuard&) = delete;

  Buffer* get() const { return buf_; }
};

torch::Tensor serializeObject(py::object o) {
  BufferGuard guard(serializeObjectImpl(o.ptr()));
  void* ptr = serializeBufferPtr(guard.get());
  size_t size = serializeBufferSize(guard.get());

  // Move guard into lambda - destructor handles cleanup when tensor is freed
  return torch::from_blob(
      ptr, {static_cast<int64_t>(size)},
      [guard = std::move(guard)](void*) {},
      torch::TensorOptions().dtype(torch::kUInt8));
}

py::object deserializeObject(torch::Tensor t) {
  PyObject* result = deserializeObjectImpl(t.data_ptr(), t.itemsize() * t.numel());
  // reinterpret_steal takes ownership of the new reference
  return py::reinterpret_steal<py::object>(result);
}

} // namespace moodist
