// Copyright (c) Meta Platforms, Inc. and affiliates.

// Serialize/deserialize wrapper functions for Python stable API.

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

namespace moodist {
namespace wrapper {

// Serialize a Python object to a uint8 tensor
PyObject* serialize(PyObject* self, PyObject* args);

// Deserialize a uint8 tensor to a Python object
PyObject* deserialize(PyObject* self, PyObject* args);

} // namespace wrapper
} // namespace moodist
