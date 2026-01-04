// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Python bindings using stable API (Py_LIMITED_API defined by CMake via USE_SABI)
// This file replaces pybind.cc for Python-version-independent builds.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// PyTorch headers (C++ - not affected by Py_LIMITED_API)
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

// Moodist headers
#include "moodist_loader.h"
#include "store.h"

#include <string>
#include <vector>

// =============================================================================
// pybind11-compatible instance layout
// This struct must match pybind11::detail::instance exactly
// =============================================================================

struct pybind11_instance {
  PyObject_HEAD
  void* simple_value_holder[3];  // [0]=ptr, [1..2]=holder (e.g. intrusive_ptr)
  PyObject* weakrefs;
  bool owned : 1;
  bool simple_layout : 1;
  bool simple_holder_constructed : 1;
  bool simple_instance_registered : 1;
  bool has_patients : 1;
  bool is_alias : 1;
};

// =============================================================================
// Helper: Get base type from torch.distributed
// =============================================================================

static PyTypeObject* get_torch_store_type() {
  static PyTypeObject* store_type = nullptr;
  if (!store_type) {
    PyObject* torch_dist = PyImport_ImportModule("torch.distributed");
    if (!torch_dist) return nullptr;
    store_type = (PyTypeObject*)PyObject_GetAttrString(torch_dist, "Store");
    Py_DECREF(torch_dist);
  }
  return store_type;
}

// =============================================================================
// Framework: Create a type that inherits from a pybind11 base
// =============================================================================

struct TypeBuilder {
  const char* name;
  const char* doc;
  PyTypeObject* base;
  initproc tp_init;
  destructor tp_dealloc;
  PyMethodDef* methods;

  PyObject* build() {
    PyType_Slot slots[] = {
        {Py_tp_init, (void*)tp_init},
        {Py_tp_dealloc, (void*)tp_dealloc},
        {Py_tp_methods, methods},
        {Py_tp_doc, (void*)doc},
        {0, nullptr}
    };

    PyType_Spec spec = {
        .name = name,
        .basicsize = 0,  // Inherit from base
        .itemsize = 0,
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .slots = slots
    };

    PyObject* bases = PyTuple_Pack(1, base);
    if (!bases) return nullptr;

    PyObject* type = PyType_FromSpecWithBases(&spec, bases);
    Py_DECREF(bases);

    return type;
  }
};

// =============================================================================
// Helper: Standard dealloc for pybind11-compatible types with intrusive_ptr
// =============================================================================

template<typename T>
static void intrusive_ptr_dealloc(PyObject* self) {
  auto* inst = reinterpret_cast<pybind11_instance*>(self);

  if (inst->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }

  if (inst->simple_holder_constructed) {
    auto* holder = reinterpret_cast<c10::intrusive_ptr<T>*>(&inst->simple_value_holder[1]);
    holder->~intrusive_ptr();
    inst->simple_holder_constructed = false;
  }

  // Use stable API to get tp_free
  PyTypeObject* tp = Py_TYPE(self);
  freefunc free_func = (freefunc)PyType_GetSlot(tp, Py_tp_free);
  if (free_func) {
    free_func(self);
  }
  Py_DECREF(tp);
}

// =============================================================================
// Helper: Initialize pybind11_instance with an intrusive_ptr holder
// =============================================================================

template<typename T>
static int init_holder(PyObject* self, c10::intrusive_ptr<T> holder) {
  auto* inst = reinterpret_cast<pybind11_instance*>(self);
  inst->simple_value_holder[0] = holder.get();
  inst->owned = true;
  inst->simple_layout = true;
  new (&inst->simple_value_holder[1]) c10::intrusive_ptr<T>(std::move(holder));
  inst->simple_holder_constructed = true;
  return 0;
}

// Helper macro to wrap init in try-catch
#define INIT_TRY(expr) \
  try { \
    return expr; \
  } catch (const moodist::PythonErrorAlreadySet&) { \
    return -1; /* Python error already set */ \
  } catch (const std::exception& e) { \
    PyErr_SetString(PyExc_RuntimeError, e.what()); \
    return -1; \
  }

// =============================================================================
// TcpStore - real store using moodist core
// =============================================================================

// Helper to convert Python object to seconds (handles float, int, timedelta)
static bool parse_timeout(PyObject* obj, double* out_seconds) {
  if (obj == nullptr || obj == Py_None) {
    *out_seconds = 300.0;  // Default 5 minutes
    return true;
  }

  // Check for numeric types first
  if (PyFloat_Check(obj)) {
    *out_seconds = PyFloat_AsDouble(obj);
    return true;
  }
  if (PyLong_Check(obj)) {
    *out_seconds = PyLong_AsDouble(obj);
    return true;
  }

  // Check for timedelta
  PyObject* total_seconds = PyObject_CallMethod(obj, "total_seconds", nullptr);
  if (total_seconds) {
    *out_seconds = PyFloat_AsDouble(total_seconds);
    Py_DECREF(total_seconds);
    return true;
  }
  PyErr_Clear();

  PyErr_SetString(PyExc_TypeError, "timeout must be a number or timedelta");
  return false;
}

// TcpStore tp_init
static int tcpstore_init(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"hostname", "port", "key", "world_size", "rank", "timeout", nullptr};
  const char* hostname = nullptr;
  int port = 0;
  const char* key = nullptr;
  int world_size = 0;
  int rank = 0;
  PyObject* timeout_obj = nullptr;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "sisii|O", const_cast<char**>(kwlist),
          &hostname, &port, &key, &world_size, &rank, &timeout_obj)) {
    return -1;
  }

  double timeout_seconds;
  if (!parse_timeout(timeout_obj, &timeout_seconds)) {
    return -1;
  }

  auto timeout = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(timeout_seconds));

  INIT_TRY(init_holder(self, c10::make_intrusive<moodist::wrapper::TcpStore>(
      hostname, port, key, world_size, rank, timeout)));
}

static PyMethodDef tcpstore_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

// Create TcpStore type
static PyObject* create_tcpstore_type() {
  PyTypeObject* base = get_torch_store_type();
  if (!base) return nullptr;

  TypeBuilder builder = {
      .name = "moodist._C.TcpStore",
      .doc = "A moodist TCP store",
      .base = base,
      .tp_init = tcpstore_init,
      .tp_dealloc = intrusive_ptr_dealloc<moodist::wrapper::TcpStore>,
      .methods = tcpstore_methods
  };

  return builder.build();
}

// =============================================================================
// Module functions
// =============================================================================

static PyObject* hello_world(PyObject* self, PyObject* args) {
  (void)self;
  (void)args;
  return PyUnicode_FromString("Hello from stable API!");
}

static PyMethodDef module_methods[] = {
    {"hello", hello_world, METH_NOARGS, "Returns a greeting"},
    {nullptr, nullptr, 0, nullptr}
};

// =============================================================================
// Module definition
// =============================================================================

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_C",
    "Moodist Python bindings (stable API)",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC PyInit__C(void) {
  PyObject* m = PyModule_Create(&module_def);
  if (!m) {
    return nullptr;
  }

  // Initialize moodist API (loads libmoodist.so)
  try {
    moodist::initMoodistApi();
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    Py_DECREF(m);
    return nullptr;
  }

  // Register TcpStore
  PyObject* tcpstore_type = create_tcpstore_type();
  if (tcpstore_type) {
    PyModule_AddObject(m, "TcpStore", tcpstore_type);
  }

  return m;
}
