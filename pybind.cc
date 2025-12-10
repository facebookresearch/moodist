// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "torch_includes.h"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/utils/pybind.h>

// MINIMAL BUILD: Only store + serialize for now
#include "store.h"

// Commented out until we add more components:
// #include "backend.h"
// #include "common.h"
// #include "cuda_copy.h"
// #include "processgroup.h"

namespace moodist {
bool profilingEnabled = false;

// Serialize/deserialize - defined in serialize_wrapper.cc
torch::Tensor serializeObject(py::object o);
py::object deserializeObject(torch::Tensor t);

// Commented out - defined in other files:
// void enableCudaAllocator();
// void enableCpuAllocator();
// void cpuAllocatorDebug();
// void setPreferKernelLess(bool);
// TensorWrapper wrapTensor(torch::Tensor t);
// torch::Tensor unwrapTensor(TensorWrapper& w);

} // namespace moodist

namespace py = pybind11;

// MINIMAL BUILD: Commented out until we add more components
// using MoodistProcessGroup = moodist::ProcessGroup;
// using MoodistBackend = moodist::Backend;

PYBIND11_MODULE(_C, m) {
  // MINIMAL BUILD: TcpStore + serialize/deserialize
  py::class_<moodist::TcpStore, c10::intrusive_ptr<moodist::TcpStore>, c10d::Store>(m, "TcpStore", R"d(
    A moodist tcp store.
  )d")
      .def(
          py::init<std::string, int, std::string, int, int, std::chrono::steady_clock::duration>(), py::arg("hostname"),
          py::arg("port"), py::arg("key"), py::arg("world_size"), py::arg("rank"), py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>());

  m.def("serialize", &moodist::serializeObject);
  m.def("deserialize", &moodist::deserializeObject);

  // Commented out until we add more components:
  /*
  py::class_<MoodistProcessGroup, c10::intrusive_ptr<MoodistProcessGroup>, c10d::ProcessGroup>(
      m, "MoodistProcessGroup", R"d(
    A moodist process group :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>(), py::call_guard<py::gil_scoped_release>())
      .def(
          "Queue", py::overload_cast<int, bool, std::optional<std::string>>(&MoodistProcessGroup::makeQueue),
          py::arg("location"), py::arg("streaming") = false, py::arg("name") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "Queue",
          py::overload_cast<std::vector<int>, bool, std::optional<std::string>>(&MoodistProcessGroup::makeQueue),
          py::arg("location"), py::arg("streaming") = false, py::arg("name") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "cat", &MoodistProcessGroup::cat, py::arg("locals"), py::arg("out") = py::none(),
          py::call_guard<py::gil_scoped_release>())
      .def("copy", &MoodistProcessGroup::copy, py::call_guard<py::gil_scoped_release>())
      .def("moodist_name", &MoodistProcessGroup::moodist_name)
      .def("share", &MoodistProcessGroup::share, py::call_guard<py::gil_scoped_release>())
      .def("cuda_barrier", &MoodistProcessGroup::cudaBarrier, py::call_guard<py::gil_scoped_release>())
      .def("compile_op_full", &MoodistProcessGroup::compileOpFull, py::call_guard<py::gil_scoped_release>());

  py::class_<MoodistBackend, c10::intrusive_ptr<MoodistBackend>, c10d::Backend>(m, "MoodistBackend", R"d(
    A moodist process group :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>(), py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::Future>(m, "Future")
      .def("wait", &moodist::Future::wait, py::call_guard<py::gil_scoped_release>())
      .def("result", &moodist::Future::result, py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::CustomOp>(m, "CustomOp")
      .def("__call__", &moodist::CustomOp::operator(), py::call_guard<py::gil_scoped_release>());

  m.def("enable_profiling", [](bool b) {
    printf("enable profiling -> %d\n", b);
    moodist::profilingEnabled = b;
  });
  m.def("enable_cuda_allocator", &moodist::enableCudaAllocator);
  m.def("enable_cpu_allocator", &moodist::enableCpuAllocator);
  m.def("cpu_allocator_debug", &moodist::cpuAllocatorDebug);

  m.def("set_prefer_kernel_less", &moodist::setPreferKernelLess);

  m.def("cuda_copy", &moodist::cudaCopyTensor, py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::Queue, std::shared_ptr<moodist::Queue>>(m, "Queue")
      .def(
          "put",
          [](moodist::Queue& q, torch::Tensor tensor, uint32_t transaction, bool wait_on_destroy) {
            return q.put(moodist::wrapTensor(std::move(tensor)), transaction, wait_on_destroy);
          },
          py::arg("tensor"), py::arg("transaction"), py::arg("wait_on_destroy") = true,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get",
          [](moodist::Queue& q, bool block, std::optional<float> timeout) {
            auto [tw, size] = q.get(block, timeout);
            std::optional<torch::Tensor> result;
            if (tw) {
              result = moodist::unwrapTensor(*tw);
            }
            return std::make_pair(result, size);
          },
          py::arg("block"), py::arg("timeout"), py::call_guard<py::gil_scoped_release>())
      .def("qsize", &moodist::Queue::qsize, py::call_guard<py::gil_scoped_release>())
      .def("wait", &moodist::Queue::wait, py::arg("timeout"), py::call_guard<py::gil_scoped_release>())
      .def("transaction_begin", &moodist::Queue::transactionBegin, py::call_guard<py::gil_scoped_release>())
      .def("transaction_cancel", &moodist::Queue::transactionCancel, py::call_guard<py::gil_scoped_release>())
      .def("transaction_commit", &moodist::Queue::transactionCommit, py::call_guard<py::gil_scoped_release>())
      .def("name", &moodist::Queue::name);
  py::class_<moodist::QueueWork>(m, "QueueWork")
      .def("wait", &moodist::QueueWork::wait, py::call_guard<py::gil_scoped_release>());
  */
}
