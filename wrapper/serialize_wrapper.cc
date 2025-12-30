// Copyright (c) Meta Platforms, Inc. and affiliates.

// Wrapper for serialize/deserialize that uses torch::Tensor.
// Uses function pointers from coreApi to call into libmoodist.so.

#include "api/types.h"
#include "moodist_loader.h"
#include "torch_includes.h"

#include <pybind11/pybind11.h>

namespace moodist {

namespace py = pybind11;

// ApiProxy method implementations for Buffer
namespace api {

void* ApiProxy<Buffer>::data() const {
  return coreApi.serializeBufferPtr(ptr);
}

size_t ApiProxy<Buffer>::size() const {
  return coreApi.serializeBufferSize(ptr);
}

// destroy() implementation for api::Buffer - called by ApiHandle destructor
void destroy(Buffer* buffer) {
  coreApi.bufferDestroy(buffer);
}

} // namespace api

torch::Tensor serializeObject(py::object o) {
  auto handle = coreApi.serializeObjectImpl(o.ptr());
  void* ptr = handle->data();
  size_t size = handle->size();

  // Capture handle by value - copy increments refcount
  // Handle destructor will decref when the tensor's deleter is called
  return torch::from_blob(
      ptr, {static_cast<int64_t>(size)},
      [handle](void*) mutable {
        handle.reset();
      },
      torch::TensorOptions().dtype(torch::kUInt8));
}

py::object deserializeObject(torch::Tensor t) {
  PyObject* result = coreApi.deserializeObjectImpl(t.data_ptr(), t.itemsize() * t.numel());
  // reinterpret_steal takes ownership of the new reference
  return py::reinterpret_steal<py::object>(result);
}

} // namespace moodist
