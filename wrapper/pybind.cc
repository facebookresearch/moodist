// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "torch_includes.h"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>

#include "api/allocator.h"
#include "api/cpu_allocator.h"
#include "backend.h"
#include "moodist_loader.h"
#include "processgroup_wrapper.h"
#include "store.h"

// For accessing pybind11's instance layout
#include <pybind11/detail/common.h>
#include <pybind11/detail/internals.h>

namespace moodist::wrapper {

// Serialize/deserialize - defined in serialize_wrapper.cc
torch::Tensor serializeObject(py::object o);
py::object deserializeObject(torch::Tensor t);

} // namespace moodist::wrapper

namespace py = pybind11;

using MoodistProcessGroup = moodist::wrapper::ProcessGroup;
using MoodistBackend = moodist::wrapper::Backend;

// =============================================================================
// EXPERIMENTAL: Test creating a Store type without py::class_
// This tests whether we can bypass pybind11's registration while still
// being compatible with PyTorch's py::cast<c10d::Store&>()
// =============================================================================

namespace {

// Minimal struct matching pybind11::detail::instance layout
// This allows us to avoid including pybind11 headers in stable API code
struct pybind11_instance {
  PyObject_HEAD
  // Storage: [0] = C++ pointer, [1..2] = holder (up to 2 pointers for shared_ptr)
  void* simple_value_holder[3];
  PyObject* weakrefs;
  bool owned : 1;
  bool simple_layout : 1;
  bool simple_holder_constructed : 1;
  bool simple_instance_registered : 1;
  bool has_patients : 1;
  bool is_alias : 1;
};

// Verify layout matches pybind11's at compile time
static_assert(offsetof(pybind11_instance, simple_value_holder) ==
              offsetof(pybind11::detail::instance, simple_value_holder),
              "simple_value_holder offset mismatch");
static_assert(offsetof(pybind11_instance, weakrefs) ==
              offsetof(pybind11::detail::instance, weakrefs),
              "weakrefs offset mismatch");

// Simple test Store that just stores a string
class TestStore : public c10d::Store {
 public:
  explicit TestStore(std::string name) : name_(std::move(name)) {}
  ~TestStore() override = default;

  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    data_[key] = value;
  }

  std::vector<uint8_t> get(const std::string& key) override {
    auto it = data_.find(key);
    if (it == data_.end()) {
      throw std::runtime_error("Key not found: " + key);
    }
    return it->second;
  }

  int64_t add(const std::string&, int64_t) override {
    throw std::runtime_error("add not implemented");
  }

  bool deleteKey(const std::string&) override {
    throw std::runtime_error("deleteKey not implemented");
  }

  int64_t getNumKeys() override {
    return data_.size();
  }

  bool check(const std::vector<std::string>& keys) override {
    for (const auto& key : keys) {
      if (data_.find(key) == data_.end()) return false;
    }
    return true;
  }

  void wait(const std::vector<std::string>&) override {}

  void wait(const std::vector<std::string>&, const std::chrono::milliseconds&) override {}

  c10::intrusive_ptr<c10d::Store> clone() override {
    // Return a new TestStore with the same name (shallow clone for testing)
    return c10::make_intrusive<TestStore>(name_);
  }

  const std::string& getName() const { return name_; }

 private:
  std::string name_;
  std::unordered_map<std::string, std::vector<uint8_t>> data_;
};

// Our tp_init - sets up the C++ object in pybind11's instance layout
static int teststore_init(PyObject* self, PyObject* args, PyObject* kwds) {
  static const char* kwlist[] = {"name", nullptr};
  const char* name = nullptr;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", const_cast<char**>(kwlist), &name)) {
    return -1;
  }

  // self was allocated by inherited tp_new (pybind11_object_new)
  // Cast to our matching instance layout
  auto* inst = reinterpret_cast<pybind11_instance*>(self);

  // Create our C++ object via make_intrusive (handles refcount correctly)
  auto holder = c10::make_intrusive<TestStore>(name);
  auto* store = holder.get();

  // Put pointer where pybind11 expects it
  // For simple layout: simple_value_holder[0] = C++ pointer
  inst->simple_value_holder[0] = store;
  inst->owned = true;
  inst->simple_layout = true;

  // Move holder into simple_value_holder[1]
  new (&inst->simple_value_holder[1]) c10::intrusive_ptr<TestStore>(std::move(holder));
  inst->simple_holder_constructed = true;

  return 0;
}

// Our tp_dealloc - clean up the C++ object
static void teststore_dealloc(PyObject* self) {
  auto* inst = reinterpret_cast<pybind11_instance*>(self);

  // Clear weak references (if any)
  if (inst->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }

  if (inst->simple_holder_constructed) {
    // Destroy the intrusive_ptr holder
    auto* holder = reinterpret_cast<c10::intrusive_ptr<TestStore>*>(&inst->simple_value_holder[1]);
    holder->~intrusive_ptr();
    inst->simple_holder_constructed = false;
  }

  // Free the Python object memory (don't call base dealloc - it would double-destroy)
  PyTypeObject* tp = Py_TYPE(self);
  tp->tp_free(self);
  Py_DECREF(tp);  // Heap types need their type decref'd (Python 3.8+)
}

// Method to get the name (to verify our C++ object is accessible)
static PyObject* teststore_get_name(PyObject* self, PyObject*) {
  auto* inst = reinterpret_cast<pybind11_instance*>(self);
  auto* store = static_cast<TestStore*>(inst->simple_value_holder[0]);
  return PyUnicode_FromString(store->getName().c_str());
}

static PyMethodDef teststore_methods[] = {
    {"get_name", teststore_get_name, METH_NOARGS, "Get the store name"},
    {nullptr, nullptr, 0, nullptr}
};

// Create the TestStore type at module init
static PyObject* create_teststore_type() {
  // Get torch.distributed.Store type
  PyObject* torch_dist = PyImport_ImportModule("torch.distributed");
  if (!torch_dist) return nullptr;

  PyObject* store_type = PyObject_GetAttrString(torch_dist, "Store");
  Py_DECREF(torch_dist);
  if (!store_type) return nullptr;

  // Create our type spec - note: we DON'T set Py_tp_new, so we inherit it!
  static PyType_Slot slots[] = {
      {Py_tp_init, (void*)teststore_init},
      {Py_tp_dealloc, (void*)teststore_dealloc},
      {Py_tp_methods, teststore_methods},
      {0, nullptr}
  };

  static PyType_Spec spec = {
      .name = "moodist._C.TestStore",
      .basicsize = 0,  // Inherit from base
      .itemsize = 0,
      .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
      .slots = slots
  };

  PyObject* bases = PyTuple_Pack(1, store_type);
  Py_DECREF(store_type);
  if (!bases) return nullptr;

  PyObject* type = PyType_FromSpecWithBases(&spec, bases);
  Py_DECREF(bases);

  return type;
}

} // anonymous namespace

// =============================================================================
// END EXPERIMENTAL
// =============================================================================

PYBIND11_MODULE(_C, m) {
  // Initialize the moodist Api (loads libmoodist.so)
  moodist::initMoodistApi();

  // Register experimental TestStore type
  PyObject* teststore_type = create_teststore_type();
  if (teststore_type) {
    m.add_object("TestStore", py::reinterpret_steal<py::object>(teststore_type));
  }

  // MINIMAL BUILD: TcpStore + serialize/deserialize
  py::class_<moodist::wrapper::TcpStore, c10::intrusive_ptr<moodist::wrapper::TcpStore>, c10d::Store>(
      m, "TcpStore", R"d(
    A moodist tcp store.
  )d")
      .def(py::init<std::string, int, std::string, int, int, std::chrono::steady_clock::duration>(),
          py::arg("hostname"), py::arg("port"), py::arg("key"), py::arg("world_size"), py::arg("rank"),
          py::arg("timeout"), py::call_guard<py::gil_scoped_release>());

  m.def("serialize", &moodist::wrapper::serializeObject);
  m.def("deserialize", &moodist::wrapper::deserializeObject);

  m.def("enable_cuda_allocator", &moodist::enableCudaAllocator);
  m.def("enable_cpu_allocator", &moodist::enableCpuAllocator);

  py::class_<MoodistProcessGroup, c10::intrusive_ptr<MoodistProcessGroup>, c10d::ProcessGroup>(
      m, "MoodistProcessGroup", py::dynamic_attr(), R"d(
    A moodist process group :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>(), py::call_guard<py::gil_scoped_release>())
      .def("moodist_name", &MoodistProcessGroup::moodist_name)
      .def("get_prefer_kernel_less", &MoodistProcessGroup::getPreferKernelLess)
      .def("set_prefer_kernel_less", &MoodistProcessGroup::setPreferKernelLess, py::arg("value"))
      .def("_get_option", &MoodistProcessGroup::getOption, py::arg("name"))
      .def("_set_option", &MoodistProcessGroup::setOption, py::arg("name"), py::arg("value"))
      .def(
          "options",
          [](py::object self, py::kwargs kwargs) {
            // Import the Python options module (separate file to avoid circular imports)
            py::module_ options_mod = py::module_::import("moodist.options");
            py::object MoodistOptions = options_mod.attr("MoodistOptions");

            // Get or create the cached options object
            py::object opts;
            if (py::hasattr(self, "_moodist_options_cache")) {
              opts = self.attr("_moodist_options_cache");
            } else {
              opts = MoodistOptions(self);
              self.attr("_moodist_options_cache") = opts;
            }

            // If kwargs provided, return context manager; otherwise return options object
            if (kwargs.size() > 0) {
              return opts.attr("__call__")(**kwargs);
            }
            return opts;
          },
          R"d(
    Get options object or create context manager for temporary option overrides.

    Usage:
        # Get options object for direct access:
        pg.options().prefer_kernel_less = True

        # Create context manager with kwargs:
        with pg.options(prefer_kernel_less=True) as pg:
            pg.allgather(...)
  )d")
      .def("cuda_barrier", &MoodistProcessGroup::cudaBarrier, py::call_guard<py::gil_scoped_release>())
      .def("Queue", py::overload_cast<int, bool, std::optional<std::string>>(&MoodistProcessGroup::makeQueue),
          py::arg("location"), py::arg("streaming") = false, py::arg("name") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def("Queue",
          py::overload_cast<std::vector<int>, bool, std::optional<std::string>>(&MoodistProcessGroup::makeQueue),
          py::arg("location"), py::arg("streaming") = false, py::arg("name") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def("compile_op_full", &MoodistProcessGroup::compileOpFull, py::arg("shape"), py::arg("dtype"),
          py::arg("inputs"), py::arg("outputs"), py::call_guard<py::gil_scoped_release>());

  py::class_<MoodistBackend, c10::intrusive_ptr<MoodistBackend>, c10d::Backend>(m, "MoodistBackend", R"d(
    A moodist backend :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>(), py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::wrapper::Future>(m, "Future")
      .def("wait", &moodist::wrapper::Future::wait, py::call_guard<py::gil_scoped_release>())
      .def("result", &moodist::wrapper::Future::result, py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::wrapper::CustomOp>(m, "CustomOp")
      .def("__call__", &moodist::wrapper::CustomOp::operator(), py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::wrapper::Queue, std::shared_ptr<moodist::wrapper::Queue>>(m, "Queue")
      .def("put", &moodist::wrapper::Queue::put, py::arg("tensor"), py::arg("transaction"),
          py::arg("wait_on_destroy") = true, py::call_guard<py::gil_scoped_release>())
      .def("get", &moodist::wrapper::Queue::get, py::arg("block") = true, py::arg("timeout") = std::nullopt,
          py::call_guard<py::gil_scoped_release>())
      .def("qsize", &moodist::wrapper::Queue::qsize, py::call_guard<py::gil_scoped_release>())
      .def("wait", &moodist::wrapper::Queue::wait, py::arg("timeout"), py::call_guard<py::gil_scoped_release>())
      .def("transaction_begin", &moodist::wrapper::Queue::transactionBegin, py::call_guard<py::gil_scoped_release>())
      .def("transaction_cancel", &moodist::wrapper::Queue::transactionCancel, py::call_guard<py::gil_scoped_release>())
      .def("transaction_commit", &moodist::wrapper::Queue::transactionCommit, py::call_guard<py::gil_scoped_release>())
      .def("name", &moodist::wrapper::Queue::name);

  py::class_<moodist::wrapper::QueueWork>(m, "QueueWork")
      .def("wait", &moodist::wrapper::QueueWork::wait, py::call_guard<py::gil_scoped_release>());
}
