// Copyright (c) Meta Platforms, Inc. and affiliates.

// Wrapper for serialize/deserialize that uses torch::Tensor.
// Uses function pointers from coreApi to call into libmoodist.so.

#include "api/api_handle.h"
#include "moodist_loader.h"
#include "torch_includes.h"

#include <pybind11/pybind11.h>

namespace moodist {

namespace py = pybind11;

// Buffer handle using ApiHandle for refcount management
using BufferHandle = ApiHandle<&CoreApi::bufferDestroy>;

torch::Tensor serializeObject(py::object o) {
  BufferHandle handle = BufferHandle::adopt(coreApi.serializeObjectImpl(o.ptr()));
  void* ptr = coreApi.serializeBufferPtr(static_cast<Buffer*>(handle.get()));
  size_t size = coreApi.serializeBufferSize(static_cast<Buffer*>(handle.get()));

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
