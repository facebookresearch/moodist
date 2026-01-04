// Copyright (c) Meta Platforms, Inc. and affiliates.

// Serialize/deserialize wrapper and Buffer API implementations.
// Uses function pointers from coreApi to call into libmoodist.so.

#include "serialize_wrapper.h"

#include "api/types.h"
#include "moodist_loader.h"
#include "torch_includes.h"

#include <stdexcept>

// THPVariable_Wrap from libtorch_python.so
PyObject* THPVariable_Wrap(const at::TensorBase& tensor);

// THPVariable helpers defined in module.cc
bool THPVariable_Check(PyObject* obj);
const at::Tensor& THPVariable_Unpack(PyObject* obj);

namespace moodist {

// =============================================================================
// Buffer API proxy implementations
// =============================================================================

namespace api {

void* ApiProxy<Buffer>::data() const {
  if (!coreApi.serializeBufferPtr) {
    throw std::runtime_error("Serialization not available");
  }
  return coreApi.serializeBufferPtr(ptr);
}

size_t ApiProxy<Buffer>::size() const {
  if (!coreApi.serializeBufferSize) {
    throw std::runtime_error("Serialization not available");
  }
  return coreApi.serializeBufferSize(ptr);
}

// destroy() implementation for api::Buffer - called by ApiHandle destructor
void destroy(Buffer* buffer) {
  if (coreApi.bufferDestroy) {
    coreApi.bufferDestroy(buffer);
  }
}

} // namespace api

// =============================================================================
// Serialize/Deserialize Python functions
// =============================================================================

namespace wrapper {

PyObject* serialize(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* obj;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }

  if (!coreApi.serializeObjectImpl) {
    PyErr_SetString(PyExc_RuntimeError, "Serialization not available (moodist serialize component not loaded)");
    return nullptr;
  }

  try {
    auto handle = coreApi.serializeObjectImpl(obj);
    void* ptr = coreApi.serializeBufferPtr(handle.get());
    size_t size = coreApi.serializeBufferSize(handle.get());

    // Create tensor from buffer data
    // Use at::from_blob with a custom deleter that releases the handle
    auto* handlePtr = new api::BufferHandle(std::move(handle));
    auto tensor = at::from_blob(
        ptr, {static_cast<int64_t>(size)},
        [handlePtr](void*) {
          delete handlePtr;
        },
        at::TensorOptions().dtype(at::kByte));

    return THPVariable_Wrap(tensor);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

PyObject* deserialize(PyObject* self, PyObject* args) {
  (void)self;
  PyObject* tensor_obj;
  if (!PyArg_ParseTuple(args, "O", &tensor_obj)) {
    return nullptr;
  }

  if (!THPVariable_Check(tensor_obj)) {
    PyErr_SetString(PyExc_TypeError, "Expected a torch.Tensor");
    return nullptr;
  }

  if (!coreApi.deserializeObjectImpl) {
    PyErr_SetString(PyExc_RuntimeError, "Deserialization not available (moodist serialize component not loaded)");
    return nullptr;
  }

  try {
    const at::Tensor& tensor = THPVariable_Unpack(tensor_obj);
    void* result = coreApi.deserializeObjectImpl(tensor.data_ptr(), tensor.itemsize() * tensor.numel());
    // deserializeObjectImpl returns a new reference
    return reinterpret_cast<PyObject*>(result);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

} // namespace wrapper
} // namespace moodist
