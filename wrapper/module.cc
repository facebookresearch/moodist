// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Python bindings using stable API (Py_LIMITED_API defined by CMake via USE_SABI)
// This file replaces pybind.cc for Python-version-independent builds.
//
// Known limitations vs pybind.cc:
// - serialize/deserialize functions require pybind11 for Python object handling
//   and will be in a separate library (libserialize.so).
// - options() context manager is handled in Python (moodist/options.py) using
//   the _get_option/_set_option methods exposed here.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// PyTorch headers (C++ - not affected by Py_LIMITED_API)
#include <c10/util/intrusive_ptr.h>
#include <c10/util/MaybeOwned.h>
#include <torch/csrc/distributed/c10d/Store.hpp>

// THPVariable_Wrap from libtorch_python.so
// We can't include torch/python.h because it pulls in pybind11
PyObject* THPVariable_Wrap(const at::TensorBase& tensor);

// PyTorch tensor object layout (from torch/csrc/autograd/python_variable.h)
// We define this ourselves to avoid pulling in pybind11 headers
struct THPVariable {
  PyObject_HEAD
  c10::MaybeOwned<at::Tensor> cdata;
  PyObject* backward_hooks;
  PyObject* post_accumulate_grad_hooks;
};

// Cache the torch.Tensor type for fast isinstance checks
static PyTypeObject* get_tensor_type() {
  static PyTypeObject* tensor_type = nullptr;
  if (!tensor_type) {
    PyObject* torch_mod = PyImport_ImportModule("torch");
    if (torch_mod) {
      tensor_type = (PyTypeObject*)PyObject_GetAttrString(torch_mod, "Tensor");
      Py_DECREF(torch_mod);
    }
  }
  return tensor_type;
}

// Check if object is a torch.Tensor
static inline bool THPVariable_Check(PyObject* obj) {
  PyTypeObject* tensor_type = get_tensor_type();
  return tensor_type && PyObject_IsInstance(obj, (PyObject*)tensor_type);
}

// Get type name (stable API compatible) - returns new reference, caller must decref
static PyObject* get_type_name(PyObject* obj) {
  PyObject* type = (PyObject*)Py_TYPE(obj);
  return PyObject_GetAttrString(type, "__name__");
}

static inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return *reinterpret_cast<THPVariable*>(obj)->cdata;
}

// Moodist headers
#include "moodist_loader.h"
#include "store.h"
#include "processgroup_wrapper.h"
#include "backend.h"
// Note: allocator headers not included - allocator_wrapper.cc/cpu_allocator_wrapper.cc
// are commented out in CMakeLists.txt and need to be added back for allocator support

#include <memory>
#include <string>
#include <vector>

// =============================================================================
// RAII wrapper for Python objects (like std::unique_ptr for PyObject*)
// =============================================================================

class PyRef {
  PyObject* obj_;
public:
  explicit PyRef(PyObject* obj = nullptr) noexcept : obj_(obj) {}
  ~PyRef() { Py_XDECREF(obj_); }

  // No copying
  PyRef(const PyRef&) = delete;
  PyRef& operator=(const PyRef&) = delete;

  // Move support
  PyRef(PyRef&& other) noexcept : obj_(other.obj_) { other.obj_ = nullptr; }
  PyRef& operator=(PyRef&& other) noexcept {
    if (this != &other) {
      Py_XDECREF(obj_);
      obj_ = other.obj_;
      other.obj_ = nullptr;
    }
    return *this;
  }

  PyObject* get() const noexcept { return obj_; }
  PyObject* release() noexcept { PyObject* tmp = obj_; obj_ = nullptr; return tmp; }
  explicit operator bool() const noexcept { return obj_ != nullptr; }
  PyObject* operator->() const noexcept { return obj_; }
};

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

static PyTypeObject* get_torch_processgroup_type() {
  static PyTypeObject* pg_type = nullptr;
  if (!pg_type) {
    PyObject* torch_dist = PyImport_ImportModule("torch.distributed");
    if (!torch_dist) return nullptr;
    pg_type = (PyTypeObject*)PyObject_GetAttrString(torch_dist, "ProcessGroup");
    Py_DECREF(torch_dist);
  }
  return pg_type;
}

static PyTypeObject* get_torch_backend_type() {
  static PyTypeObject* backend_type = nullptr;
  if (!backend_type) {
    PyObject* torch_dist = PyImport_ImportModule("torch.distributed");
    if (!torch_dist) return nullptr;
    backend_type = (PyTypeObject*)PyObject_GetAttrString(torch_dist, "Backend");
    Py_DECREF(torch_dist);
  }
  return backend_type;
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
// Helper: GIL release (RAII, exception-safe)
// =============================================================================

class GilRelease {
  PyThreadState* state_;
public:
  GilRelease() : state_(PyEval_SaveThread()) {}
  ~GilRelease() { PyEval_RestoreThread(state_); }
  GilRelease(const GilRelease&) = delete;
  GilRelease& operator=(const GilRelease&) = delete;
};

// =============================================================================
// Helper: Get typed pointer from pybind11_instance
// =============================================================================

template<typename T>
static T* get_ptr(PyObject* self) {
  auto* inst = reinterpret_cast<pybind11_instance*>(self);
  return static_cast<T*>(inst->simple_value_holder[0]);
}

// =============================================================================
// Helper: Method wrapper with exception handling
// =============================================================================

// For methods returning PyObject* (most methods)
template<typename T, typename Func>
static PyObject* method_impl(PyObject* self, Func&& func) {
  try {
    return func(get_ptr<T>(self));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

// For methods that release GIL
template<typename T, typename Func>
static PyObject* method_impl_nogil(PyObject* self, Func&& func) {
  try {
    T* ptr = get_ptr<T>(self);
    GilRelease gil;
    return func(ptr);
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

// =============================================================================
// PyWrapper: Template for standalone Python types (not inheriting from pybind11)
// =============================================================================

// T can be the object directly, std::unique_ptr<X>, std::shared_ptr<X>, etc.
// Note: weak references not supported due to limited API restrictions
template<typename T>
struct PyWrapper {
  PyObject_HEAD
  T holder;
};

// Get pointer to the wrapped object (works with raw, unique_ptr, shared_ptr)
template<typename T>
static auto* wrapper_get(PyObject* self) {
  auto* obj = reinterpret_cast<PyWrapper<T>*>(self);
  if constexpr (std::is_pointer_v<T>) {
    return obj->holder;
  } else if constexpr (requires { obj->holder.get(); }) {
    return obj->holder.get();  // shared_ptr, unique_ptr
  } else {
    return &obj->holder;  // direct object
  }
}

// Standard dealloc for PyWrapper types
template<typename T>
static void wrapper_dealloc(PyObject* self) {
  auto* obj = reinterpret_cast<PyWrapper<T>*>(self);
  obj->holder.~T();
  PyTypeObject* tp = Py_TYPE(self);
  freefunc free_func = (freefunc)PyType_GetSlot(tp, Py_tp_free);
  if (free_func) free_func(self);
  Py_DECREF(tp);
}

// Create a standalone type (no weakref support due to limited API)
template<typename T>
static PyObject* create_wrapper_type(const char* name, const char* doc,
                                      PyMethodDef* methods, ternaryfunc tp_call = nullptr) {
  std::vector<PyType_Slot> slots = {
      {Py_tp_dealloc, (void*)wrapper_dealloc<T>},
      {Py_tp_methods, methods},
      {Py_tp_doc, (void*)doc},
  };
  if (tp_call) {
    slots.push_back({Py_tp_call, (void*)tp_call});
  }
  slots.push_back({0, nullptr});

  PyType_Spec spec = {
      .name = name,
      .basicsize = (int)sizeof(PyWrapper<T>),
      .itemsize = 0,
      .flags = Py_TPFLAGS_DEFAULT,
      .slots = slots.data()
  };

  return PyType_FromSpec(&spec);
}

// Allocate a PyWrapper instance using tp_alloc (properly zeros memory)
template<typename T>
static PyWrapper<T>* alloc_wrapper(PyTypeObject* type) {
  allocfunc alloc = (allocfunc)PyType_GetSlot(type, Py_tp_alloc);
  if (!alloc) {
    PyErr_SetString(PyExc_RuntimeError, "Type has no tp_alloc");
    return nullptr;
  }
  PyObject* obj = alloc(type, 0);
  if (!obj) return nullptr;
  return reinterpret_cast<PyWrapper<T>*>(obj);
}

// =============================================================================
// Conversion helpers: Python -> C++
// =============================================================================

// Convert Python list of tensors to std::vector<torch::Tensor>
// Returns false on error (Python exception set)
static bool parse_tensor_list(PyObject* list, std::vector<torch::Tensor>& out) {
  if (!PyList_Check(list)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list of tensors");
    return false;
  }
  Py_ssize_t len = PyList_Size(list);
  out.reserve(len);
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PyList_GetItem(list, i);
    if (!THPVariable_Check(item)) {
      PyRef type_name(get_type_name(item));
      if (type_name) {
        PyErr_Format(PyExc_TypeError, "Expected tensor at index %zd, got %S", i, type_name.get());
      } else {
        PyErr_Format(PyExc_TypeError, "Expected tensor at index %zd", i);
      }
      return false;
    }
    out.push_back(THPVariable_Unpack(item));
  }
  return true;
}

// Convert Python list of ints to std::vector<int>
static bool parse_int_list(PyObject* list, std::vector<int>& out) {
  if (!PyList_Check(list)) {
    PyErr_SetString(PyExc_TypeError, "Expected a list of integers");
    return false;
  }
  Py_ssize_t len = PyList_Size(list);
  out.reserve(len);
  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PyList_GetItem(list, i);
    if (!PyLong_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "Expected integer in list");
      return false;
    }
    out.push_back(PyLong_AsLong(item));
  }
  return true;
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
// Helper: Extract c10d::Store from Python Store object
// =============================================================================

static c10::intrusive_ptr<c10d::Store> extract_store(PyObject* store_obj) {
  if (!store_obj || store_obj == Py_None) {
    throw std::runtime_error("store cannot be None");
  }

  // The store object is a pybind11 instance wrapping c10::intrusive_ptr<Store>
  auto* inst = reinterpret_cast<pybind11_instance*>(store_obj);
  if (!inst->simple_holder_constructed) {
    throw std::runtime_error("store object has no holder");
  }

  // The holder is an intrusive_ptr<Store> at simple_value_holder[1]
  auto* holder = reinterpret_cast<c10::intrusive_ptr<c10d::Store>*>(&inst->simple_value_holder[1]);
  return *holder;
}

// =============================================================================
// MoodistProcessGroup - ProcessGroup implementation using moodist core
// =============================================================================

using MoodistProcessGroup = moodist::wrapper::ProcessGroup;

// ProcessGroup tp_init
static int processgroup_init(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"store", "rank", "size", nullptr};
  PyObject* store_obj = nullptr;
  int rank = 0;
  int size = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii", const_cast<char**>(kwlist),
          &store_obj, &rank, &size)) {
    return -1;
  }

  try {
    auto store = extract_store(store_obj);
    return init_holder(self, c10::make_intrusive<MoodistProcessGroup>(store, rank, size));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return -1;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
}

// ProcessGroup methods (using helpers to reduce boilerplate)
static PyObject* processgroup_moodist_name(PyObject* self, PyObject*) {
  return method_impl<MoodistProcessGroup>(self, [](auto* pg) {
    return PyUnicode_FromString(pg->moodist_name().c_str());
  });
}

static PyObject* processgroup_get_prefer_kernel_less(PyObject* self, PyObject*) {
  return method_impl<MoodistProcessGroup>(self, [](auto* pg) {
    return PyBool_FromLong(pg->getPreferKernelLess());
  });
}

static PyObject* processgroup_set_prefer_kernel_less(PyObject* self, PyObject* args) {
  int value = 0;
  if (!PyArg_ParseTuple(args, "p", &value)) {
    return nullptr;
  }
  return method_impl<MoodistProcessGroup>(self, [value](auto* pg) {
    pg->setPreferKernelLess(value != 0);
    Py_RETURN_NONE;
  });
}

static PyObject* processgroup_get_option(PyObject* self, PyObject* args) {
  const char* name = nullptr;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return nullptr;
  }
  return method_impl<MoodistProcessGroup>(self, [name](auto* pg) {
    return PyLong_FromLongLong(pg->getOption(name));
  });
}

static PyObject* processgroup_set_option(PyObject* self, PyObject* args) {
  const char* name = nullptr;
  long long value = 0;
  if (!PyArg_ParseTuple(args, "sL", &name, &value)) {
    return nullptr;
  }
  return method_impl<MoodistProcessGroup>(self, [name, value](auto* pg) {
    pg->setOption(name, value);
    Py_RETURN_NONE;
  });
}

static PyObject* processgroup_cuda_barrier(PyObject* self, PyObject*) {
  return method_impl_nogil<MoodistProcessGroup>(self, [](auto* pg) {
    pg->cudaBarrier();
    Py_RETURN_NONE;
  });
}

// Forward declarations for wrap functions (defined later)
static PyObject* wrap_queue(std::shared_ptr<moodist::wrapper::Queue> queue);
static PyObject* wrap_customop(moodist::wrapper::CustomOp&& op);

static PyObject* processgroup_make_queue(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"location", "streaming", "name", nullptr};
  PyObject* location_obj = nullptr;
  int streaming = 0;
  const char* name = nullptr;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pz", const_cast<char**>(kwlist),
          &location_obj, &streaming, &name)) {
    return nullptr;
  }

  auto* pg = get_ptr<MoodistProcessGroup>(self);
  try {
    std::optional<std::string> opt_name = name ? std::make_optional(std::string(name)) : std::nullopt;
    std::shared_ptr<moodist::wrapper::Queue> queue;

    if (PyLong_Check(location_obj)) {
      // Single int location
      int location = PyLong_AsLong(location_obj);
      if (PyErr_Occurred()) return nullptr;
      {
        GilRelease gil;
        queue = pg->makeQueue(location, streaming != 0, opt_name);
      }
    } else {
      // Try to convert to a sequence (list, tuple, range, etc.)
      PyRef seq(PySequence_Fast(location_obj, "location must be an int or sequence of ints"));
      if (!seq) return nullptr;

      std::vector<int> locations;
      Py_ssize_t len = PySequence_Size(seq.get());
      if (len < 0) return nullptr;
      locations.reserve(len);
      for (Py_ssize_t i = 0; i < len; ++i) {
        PyRef item(PySequence_GetItem(seq.get(), i));
        if (!item) return nullptr;
        if (!PyLong_Check(item.get())) {
          PyErr_SetString(PyExc_TypeError, "location must be an int or sequence of ints");
          return nullptr;
        }
        locations.push_back(PyLong_AsLong(item.get()));
      }
      {
        GilRelease gil;
        queue = pg->makeQueue(locations, streaming != 0, opt_name);
      }
    }

    return wrap_queue(std::move(queue));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* processgroup_compile_op_full(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"shape", "dtype", "inputs", "outputs", nullptr};
  PyObject* shape_obj = nullptr;
  PyObject* dtype_obj = nullptr;
  PyObject* inputs_obj = nullptr;
  PyObject* outputs_obj = nullptr;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", const_cast<char**>(kwlist),
          &shape_obj, &dtype_obj, &inputs_obj, &outputs_obj)) {
    return nullptr;
  }

  auto* pg = get_ptr<MoodistProcessGroup>(self);
  try {
    // Parse shape (list of ints)
    std::vector<int> shape;
    if (!PyList_Check(shape_obj)) {
      PyErr_SetString(PyExc_TypeError, "shape must be a list of ints");
      return nullptr;
    }
    Py_ssize_t shape_len = PyList_Size(shape_obj);
    shape.reserve(shape_len);
    for (Py_ssize_t i = 0; i < shape_len; ++i) {
      shape.push_back(PyLong_AsLong(PyList_GetItem(shape_obj, i)));
    }

    // Parse dtype - get torch.dtype from the Python object
    // The dtype is a torch.dtype enum value
    // We need to extract the underlying ScalarType
    PyObject* torch_mod = PyImport_ImportModule("torch");
    if (!torch_mod) return nullptr;

    // Get the dtype's value - torch.dtype objects have a private _dtype attribute
    // Actually, we can compare with known dtypes or use at::python::detail::py_scalar_type
    // For now, let's get the string name and parse it
    PyObject* dtype_name = PyObject_Str(dtype_obj);
    if (!dtype_name) {
      Py_DECREF(torch_mod);
      return nullptr;
    }
    // PyUnicode_AsUTF8 is not in limited API, use PyUnicode_AsEncodedString
    PyObject* dtype_bytes = PyUnicode_AsEncodedString(dtype_name, "utf-8", "strict");
    Py_DECREF(dtype_name);
    if (!dtype_bytes) {
      Py_DECREF(torch_mod);
      return nullptr;
    }
    const char* dtype_str = PyBytes_AsString(dtype_bytes);
    torch::Dtype dtype;
    if (strcmp(dtype_str, "torch.float32") == 0 || strcmp(dtype_str, "torch.float") == 0) {
      dtype = torch::kFloat32;
    } else if (strcmp(dtype_str, "torch.float16") == 0 || strcmp(dtype_str, "torch.half") == 0) {
      dtype = torch::kFloat16;
    } else if (strcmp(dtype_str, "torch.bfloat16") == 0) {
      dtype = torch::kBFloat16;
    } else if (strcmp(dtype_str, "torch.float64") == 0 || strcmp(dtype_str, "torch.double") == 0) {
      dtype = torch::kFloat64;
    } else if (strcmp(dtype_str, "torch.int32") == 0 || strcmp(dtype_str, "torch.int") == 0) {
      dtype = torch::kInt32;
    } else if (strcmp(dtype_str, "torch.int64") == 0 || strcmp(dtype_str, "torch.long") == 0) {
      dtype = torch::kInt64;
    } else {
      PyErr_Format(PyExc_ValueError, "Unsupported dtype: %s", dtype_str);
      Py_DECREF(dtype_bytes);
      Py_DECREF(torch_mod);
      return nullptr;
    }
    Py_DECREF(dtype_bytes);
    Py_DECREF(torch_mod);

    // Parse inputs and outputs: list of tuples (rank, src_offsets, dst_offsets)
    auto parse_transfer_list = [](PyObject* list) -> std::vector<std::tuple<int, std::vector<int>, std::vector<int>>> {
      std::vector<std::tuple<int, std::vector<int>, std::vector<int>>> result;
      if (!PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of tuples");
        return {};
      }
      Py_ssize_t len = PyList_Size(list);
      result.reserve(len);
      for (Py_ssize_t i = 0; i < len; ++i) {
        PyObject* item = PyList_GetItem(list, i);
        if (!PyTuple_Check(item) || PyTuple_Size(item) != 3) {
          PyErr_SetString(PyExc_TypeError, "Each item must be a tuple of (rank, src_offsets, dst_offsets)");
          return {};
        }
        int rank = PyLong_AsLong(PyTuple_GetItem(item, 0));

        auto parse_int_list = [](PyObject* obj) -> std::vector<int> {
          std::vector<int> vec;
          if (!PyList_Check(obj)) return vec;
          Py_ssize_t len = PyList_Size(obj);
          vec.reserve(len);
          for (Py_ssize_t i = 0; i < len; ++i) {
            vec.push_back(PyLong_AsLong(PyList_GetItem(obj, i)));
          }
          return vec;
        };

        auto src_offsets = parse_int_list(PyTuple_GetItem(item, 1));
        auto dst_offsets = parse_int_list(PyTuple_GetItem(item, 2));
        result.emplace_back(rank, std::move(src_offsets), std::move(dst_offsets));
      }
      return result;
    };

    auto inputs = parse_transfer_list(inputs_obj);
    if (PyErr_Occurred()) return nullptr;
    auto outputs = parse_transfer_list(outputs_obj);
    if (PyErr_Occurred()) return nullptr;

    moodist::wrapper::CustomOp op;
    {
      GilRelease gil;
      op = pg->compileOpFull(shape, dtype, inputs, outputs);
    }
    return wrap_customop(std::move(op));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* processgroup_options(PyObject* self, PyObject* args, PyObject* kwds) {
  // Import the Python options module
  PyObject* options_mod = PyImport_ImportModule("moodist.options");
  if (!options_mod) return nullptr;

  PyObject* MoodistOptions = PyObject_GetAttrString(options_mod, "MoodistOptions");
  Py_DECREF(options_mod);
  if (!MoodistOptions) return nullptr;

  // Create options object: MoodistOptions(self)
  // Note: We don't cache because our type doesn't support dynamic attrs.
  // This is fine - MoodistOptions is just a lightweight wrapper.
  PyObject* opts = PyObject_CallFunctionObjArgs(MoodistOptions, self, nullptr);
  Py_DECREF(MoodistOptions);
  if (!opts) return nullptr;

  // If kwargs provided, call opts(**kwargs) to get context manager
  if (kwds && PyDict_Size(kwds) > 0) {
    PyObject* empty_tuple = PyTuple_New(0);
    PyObject* result = PyObject_Call(opts, empty_tuple, kwds);
    Py_DECREF(empty_tuple);
    Py_DECREF(opts);
    return result;
  }

  // Otherwise return the options object
  return opts;
}

static PyMethodDef processgroup_methods[] = {
    {"moodist_name", processgroup_moodist_name, METH_NOARGS, "Get moodist process group name"},
    {"get_prefer_kernel_less", processgroup_get_prefer_kernel_less, METH_NOARGS, "Get prefer_kernel_less setting"},
    {"set_prefer_kernel_less", processgroup_set_prefer_kernel_less, METH_VARARGS, "Set prefer_kernel_less setting"},
    {"_get_option", processgroup_get_option, METH_VARARGS, "Get option by name"},
    {"_set_option", processgroup_set_option, METH_VARARGS, "Set option by name"},
    {"cuda_barrier", processgroup_cuda_barrier, METH_NOARGS, "CUDA barrier synchronization"},
    {"Queue", (PyCFunction)processgroup_make_queue, METH_VARARGS | METH_KEYWORDS, "Create a message queue"},
    {"compile_op_full", (PyCFunction)processgroup_compile_op_full, METH_VARARGS | METH_KEYWORDS, "Compile a custom collective operation"},
    {"options", (PyCFunction)processgroup_options, METH_VARARGS | METH_KEYWORDS, "Get options object or create context manager for temporary option overrides"},
    {nullptr, nullptr, 0, nullptr}
};

// Create ProcessGroup type
static PyObject* create_processgroup_type() {
  PyTypeObject* base = get_torch_processgroup_type();
  if (!base) return nullptr;

  TypeBuilder builder = {
      .name = "moodist._C.MoodistProcessGroup",
      .doc = "A moodist process group",
      .base = base,
      .tp_init = processgroup_init,
      .tp_dealloc = intrusive_ptr_dealloc<MoodistProcessGroup>,
      .methods = processgroup_methods
  };

  return builder.build();
}

// =============================================================================
// MoodistBackend - Backend implementation using moodist core
// =============================================================================

using MoodistBackend = moodist::wrapper::Backend;

// Backend tp_init
static int backend_init(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"store", "rank", "size", nullptr};
  PyObject* store_obj = nullptr;
  int rank = 0;
  int size = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Oii", const_cast<char**>(kwlist),
          &store_obj, &rank, &size)) {
    return -1;
  }

  try {
    auto store = extract_store(store_obj);
    return init_holder(self, c10::make_intrusive<MoodistBackend>(store, rank, size));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return -1;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
}

static PyMethodDef backend_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

// Create Backend type
static PyObject* create_backend_type() {
  PyTypeObject* base = get_torch_backend_type();
  if (!base) return nullptr;

  TypeBuilder builder = {
      .name = "moodist._C.MoodistBackend",
      .doc = "A moodist backend",
      .base = base,
      .tp_init = backend_init,
      .tp_dealloc = intrusive_ptr_dealloc<MoodistBackend>,
      .methods = backend_methods
  };

  return builder.build();
}

// =============================================================================
// QueueWork - work handle for async queue operations
// =============================================================================

using MoodistQueueWork = moodist::wrapper::QueueWork;
using QueueWorkHolder = std::unique_ptr<MoodistQueueWork>;

static PyTypeObject* QueueWorkType = nullptr;

static PyObject* queuework_wait(PyObject* self, PyObject*) {
  auto* work = wrapper_get<QueueWorkHolder>(self);
  try {
    GilRelease gil;
    work->wait();
    Py_RETURN_NONE;
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyMethodDef queuework_methods[] = {
    {"wait", queuework_wait, METH_NOARGS, "Wait for the queue operation to complete"},
    {nullptr, nullptr, 0, nullptr}
};

static PyObject* create_queuework_type() {
  return create_wrapper_type<QueueWorkHolder>(
      "moodist._C.QueueWork", "Queue work handle for async operations", queuework_methods);
}

// Helper to create QueueWork Python object from C++ object
static PyObject* wrap_queuework(MoodistQueueWork&& work) {
  if (!QueueWorkType) {
    PyErr_SetString(PyExc_RuntimeError, "QueueWork type not initialized");
    return nullptr;
  }
  auto* wrapper = alloc_wrapper<QueueWorkHolder>(QueueWorkType);
  if (!wrapper) return nullptr;
  new (&wrapper->holder) QueueWorkHolder(std::make_unique<MoodistQueueWork>(std::move(work)));
  return reinterpret_cast<PyObject*>(wrapper);
}

// =============================================================================
// Queue - message queue for inter-rank communication
// =============================================================================

using MoodistQueue = moodist::wrapper::Queue;
using QueueHolder = std::shared_ptr<MoodistQueue>;

static PyTypeObject* QueueType = nullptr;

static PyObject* queue_put(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"tensor", "transaction", "wait_on_destroy", nullptr};
  PyObject* tensor_obj = nullptr;
  unsigned int transaction = 0;
  int wait_on_destroy = 1;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI|p", const_cast<char**>(kwlist),
          &tensor_obj, &transaction, &wait_on_destroy)) {
    return nullptr;
  }

  auto* queue = wrapper_get<QueueHolder>(self);
  if (!queue) {
    PyErr_SetString(PyExc_RuntimeError, "Queue object not initialized");
    return nullptr;
  }
  if (!THPVariable_Check(tensor_obj)) {
    PyRef type_name(get_type_name(tensor_obj));
    if (type_name) {
      PyErr_Format(PyExc_TypeError, "Expected tensor, got %S", type_name.get());
    } else {
      PyErr_SetString(PyExc_TypeError, "Expected tensor");
    }
    return nullptr;
  }
  try {
    const auto& tensor = THPVariable_Unpack(tensor_obj);
    MoodistQueueWork work;
    {
      GilRelease gil;
      work = queue->put(tensor, transaction, wait_on_destroy != 0);
    }
    return wrap_queuework(std::move(work));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_get(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"block", "timeout", nullptr};
  int block = 1;
  PyObject* timeout_obj = nullptr;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pO", const_cast<char**>(kwlist),
          &block, &timeout_obj)) {
    return nullptr;
  }

  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    std::optional<float> timeout;
    if (timeout_obj && timeout_obj != Py_None) {
      timeout = static_cast<float>(PyFloat_AsDouble(timeout_obj));
      if (PyErr_Occurred()) return nullptr;
    }

    std::pair<std::optional<torch::Tensor>, size_t> result;
    {
      GilRelease gil;
      result = queue->get(block != 0, timeout);
    }

    // Return (tensor_or_None, size) tuple
    PyObject* tensor_py;
    if (result.first) {
      tensor_py = THPVariable_Wrap(*result.first);
      if (!tensor_py) return nullptr;
    } else {
      Py_INCREF(Py_None);
      tensor_py = Py_None;
    }

    PyObject* size_py = PyLong_FromSize_t(result.second);
    if (!size_py) {
      Py_DECREF(tensor_py);
      return nullptr;
    }

    PyObject* tuple = PyTuple_Pack(2, tensor_py, size_py);
    Py_DECREF(tensor_py);
    Py_DECREF(size_py);
    return tuple;
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_qsize(PyObject* self, PyObject*) {
  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    size_t size;
    {
      GilRelease gil;
      size = queue->qsize();
    }
    return PyLong_FromSize_t(size);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_wait(PyObject* self, PyObject* args) {
  PyObject* timeout_obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &timeout_obj)) {
    return nullptr;
  }

  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    std::optional<float> timeout;
    if (timeout_obj && timeout_obj != Py_None) {
      timeout = static_cast<float>(PyFloat_AsDouble(timeout_obj));
      if (PyErr_Occurred()) return nullptr;
    }

    bool result;
    {
      GilRelease gil;
      result = queue->wait(timeout);
    }
    return PyBool_FromLong(result);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_transaction_begin(PyObject* self, PyObject*) {
  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    uint32_t id;
    {
      GilRelease gil;
      id = queue->transactionBegin();
    }
    return PyLong_FromUnsignedLong(id);
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_transaction_cancel(PyObject* self, PyObject* args) {
  unsigned int id = 0;
  if (!PyArg_ParseTuple(args, "I", &id)) {
    return nullptr;
  }

  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    GilRelease gil;
    queue->transactionCancel(id);
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_transaction_commit(PyObject* self, PyObject* args) {
  unsigned int id = 0;
  if (!PyArg_ParseTuple(args, "I", &id)) {
    return nullptr;
  }

  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    GilRelease gil;
    queue->transactionCommit(id);
    Py_RETURN_NONE;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* queue_name(PyObject* self, PyObject*) {
  auto* queue = wrapper_get<QueueHolder>(self);
  try {
    std::string_view name = queue->name();
    return PyUnicode_FromStringAndSize(name.data(), name.size());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyMethodDef queue_methods[] = {
    {"put", (PyCFunction)queue_put, METH_VARARGS | METH_KEYWORDS, "Put a tensor into the queue"},
    {"get", (PyCFunction)queue_get, METH_VARARGS | METH_KEYWORDS, "Get a tensor from the queue"},
    {"qsize", queue_qsize, METH_NOARGS, "Get the queue size"},
    {"wait", queue_wait, METH_VARARGS, "Wait for queue to be empty"},
    {"transaction_begin", queue_transaction_begin, METH_NOARGS, "Begin a transaction"},
    {"transaction_cancel", queue_transaction_cancel, METH_VARARGS, "Cancel a transaction"},
    {"transaction_commit", queue_transaction_commit, METH_VARARGS, "Commit a transaction"},
    {"name", queue_name, METH_NOARGS, "Get the queue name"},
    {nullptr, nullptr, 0, nullptr}
};

static PyObject* create_queue_type() {
  return create_wrapper_type<QueueHolder>(
      "moodist._C.Queue", "Message queue for inter-rank communication", queue_methods);
}

// Helper to create Queue Python object from C++ shared_ptr
static PyObject* wrap_queue(std::shared_ptr<MoodistQueue> queue) {
  if (!queue) {
    PyErr_SetString(PyExc_RuntimeError, "wrap_queue: queue is null");
    return nullptr;
  }
  if (!QueueType) {
    PyErr_SetString(PyExc_RuntimeError, "Queue type not initialized");
    return nullptr;
  }
  auto* wrapper = alloc_wrapper<QueueHolder>(QueueType);
  if (!wrapper) return nullptr;
  new (&wrapper->holder) QueueHolder(std::move(queue));
  if (!wrapper->holder) {
    PyErr_SetString(PyExc_RuntimeError, "wrap_queue: holder is null after construction");
    return nullptr;
  }
  return reinterpret_cast<PyObject*>(wrapper);
}

// =============================================================================
// Future - async operation result
// =============================================================================

using MoodistFuture = moodist::wrapper::Future;
using FutureHolder = std::unique_ptr<MoodistFuture>;

static PyTypeObject* FutureType = nullptr;

static PyObject* future_wait(PyObject* self, PyObject*) {
  auto* future = wrapper_get<FutureHolder>(self);
  try {
    GilRelease gil;
    future->wait();
    Py_RETURN_NONE;
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject* future_result(PyObject* self, PyObject*) {
  auto* future = wrapper_get<FutureHolder>(self);
  try {
    torch::Tensor result;
    {
      GilRelease gil;
      result = future->result();
    }
    return THPVariable_Wrap(result);
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyMethodDef future_methods[] = {
    {"wait", future_wait, METH_NOARGS, "Wait for the future to complete"},
    {"result", future_result, METH_NOARGS, "Get the result tensor"},
    {nullptr, nullptr, 0, nullptr}
};

static PyObject* create_future_type() {
  return create_wrapper_type<FutureHolder>(
      "moodist._C.Future", "Async operation result", future_methods);
}

// Helper to create Future Python object from C++ object
static PyObject* wrap_future(MoodistFuture&& future) {
  if (!FutureType) {
    PyErr_SetString(PyExc_RuntimeError, "Future type not initialized");
    return nullptr;
  }
  auto* wrapper = alloc_wrapper<FutureHolder>(FutureType);
  if (!wrapper) return nullptr;
  new (&wrapper->holder) FutureHolder(std::make_unique<MoodistFuture>(std::move(future)));
  return reinterpret_cast<PyObject*>(wrapper);
}

// =============================================================================
// CustomOp - compiled collective operation
// =============================================================================

using MoodistCustomOp = moodist::wrapper::CustomOp;
using CustomOpHolder = std::unique_ptr<MoodistCustomOp>;

static PyTypeObject* CustomOpType = nullptr;

static PyObject* customop_call(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"inputs", "outputs", nullptr};
  PyObject* inputs_obj = nullptr;
  PyObject* outputs_obj = nullptr;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist),
          &inputs_obj, &outputs_obj)) {
    return nullptr;
  }

  auto* op = wrapper_get<CustomOpHolder>(self);
  try {
    std::vector<torch::Tensor> inputs, outputs;
    if (!parse_tensor_list(inputs_obj, inputs) || !parse_tensor_list(outputs_obj, outputs)) {
      return nullptr;
    }

    MoodistFuture future;
    {
      GilRelease gil;
      future = (*op)(inputs, outputs);
    }
    return wrap_future(std::move(future));
  } catch (const moodist::PythonErrorAlreadySet&) {
    return nullptr;
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyMethodDef customop_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static PyObject* create_customop_type() {
  return create_wrapper_type<CustomOpHolder>(
      "moodist._C.CustomOp", "Compiled collective operation", customop_methods, customop_call);
}

// Helper to create CustomOp Python object from C++ object
static PyObject* wrap_customop(MoodistCustomOp&& op) {
  if (!CustomOpType) {
    PyErr_SetString(PyExc_RuntimeError, "CustomOp type not initialized");
    return nullptr;
  }
  auto* wrapper = alloc_wrapper<CustomOpHolder>(CustomOpType);
  if (!wrapper) return nullptr;
  new (&wrapper->holder) CustomOpHolder(std::make_unique<MoodistCustomOp>(std::move(op)));
  return reinterpret_cast<PyObject*>(wrapper);
}

// =============================================================================
// Module functions
// =============================================================================

static PyObject* hello_world(PyObject* self, PyObject* args) {
  (void)self;
  (void)args;
  return PyUnicode_FromString("Hello from stable API!");
}

// TODO: Add enable_cuda_allocator and enable_cpu_allocator once
// allocator_wrapper.cc and cpu_allocator_wrapper.cc are added back to CMakeLists.txt

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

  // Register MoodistProcessGroup
  PyObject* pg_type = create_processgroup_type();
  if (pg_type) {
    PyModule_AddObject(m, "MoodistProcessGroup", pg_type);
  }

  // Register MoodistBackend
  PyObject* backend_type = create_backend_type();
  if (backend_type) {
    PyModule_AddObject(m, "MoodistBackend", backend_type);
  }

  // Register Queue types (standalone, not inheriting from pybind11 base)
  PyObject* queuework_type = create_queuework_type();
  if (queuework_type) {
    QueueWorkType = (PyTypeObject*)queuework_type;
    PyModule_AddObject(m, "QueueWork", queuework_type);
  }

  PyObject* queue_type = create_queue_type();
  if (queue_type) {
    QueueType = (PyTypeObject*)queue_type;
    PyModule_AddObject(m, "Queue", queue_type);
  }

  PyObject* future_type = create_future_type();
  if (future_type) {
    FutureType = (PyTypeObject*)future_type;
    PyModule_AddObject(m, "Future", future_type);
  }

  PyObject* customop_type = create_customop_type();
  if (customop_type) {
    CustomOpType = (PyTypeObject*)customop_type;
    PyModule_AddObject(m, "CustomOp", customop_type);
  }

  return m;
}
