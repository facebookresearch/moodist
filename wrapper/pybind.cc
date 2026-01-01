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

namespace moodist::wrapper {

// Serialize/deserialize - defined in serialize_wrapper.cc
torch::Tensor serializeObject(py::object o);
py::object deserializeObject(torch::Tensor t);

} // namespace moodist::wrapper

namespace py = pybind11;

using MoodistProcessGroup = moodist::wrapper::ProcessGroup;
using MoodistBackend = moodist::wrapper::Backend;

PYBIND11_MODULE(_C, m) {
  // Initialize the moodist Api (loads libmoodist.so)
  moodist::initMoodistApi();

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
