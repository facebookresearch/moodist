/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/cuda/CUDAStream.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>

#include "backend.h"
#include "common.h"
#include "cuda_copy.h"
#include "processgroup.h"

namespace moodist {
bool profilingEnabled = false;
void enableCudaAllocator();
void enableCpuAllocator();
void cpuAllocatorDebug();

namespace {

std::shared_ptr<Cache> cache(c10d::ProcessGroup& pg, std::string opname) {
  auto r = std::make_shared<Cache>();
  auto* p = dynamic_cast<moodist::ProcessGroup*>(&pg);
  if (p) {
    r->add(&*p->impl, opname);
  } else {
    const auto& backend = pg.getDefaultBackend();
    auto* p = dynamic_cast<moodist::Backend*>(&*backend);
    if (p) {
      r->add(&*p->pg.impl, opname);
    } else {
      throw std::runtime_error("cache: the specified group is not a moodist process group");
    }
  }
  return r;
}

} // namespace

} // namespace moodist

namespace {
void cudaCopy(torch::Tensor& dst, const torch::Tensor& src) {
  CHECK(dst.is_contiguous());
  CHECK(src.is_contiguous());
  size_t srcbytes = src.itemsize() * src.numel();
  size_t dstbytes = dst.itemsize() * src.numel();
  if (srcbytes != dstbytes) {
    throw std::runtime_error(fmt::sprintf("cuda_copy: dst is %d bytes, but src is %d bytes", dstbytes, srcbytes));
  }
  uintptr_t dstAddress = (uintptr_t)(void*)dst.mutable_data_ptr();
  uintptr_t srcAddress = (uintptr_t)(const void*)src.const_data_ptr();
  moodist::cudaCopy(dstAddress, srcAddress, srcbytes, c10::cuda::getCurrentCUDAStream());
}
} // namespace

namespace py = pybind11;

using MoodistProcessGroup = moodist::ProcessGroup;
using MoodistBackend = moodist::Backend;

PYBIND11_MODULE(_C, m) {
  py::class_<MoodistProcessGroup, c10::intrusive_ptr<MoodistProcessGroup>, c10d::ProcessGroup>(
      m, "MoodistProcessGroup", R"d(
    A moodist process group :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>(), py::call_guard<py::gil_scoped_release>())
      .def(
          "Queue", py::overload_cast<int>(&MoodistProcessGroup::makeQueue), py::arg("location"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "Queue", py::overload_cast<std::vector<int>>(&MoodistProcessGroup::makeQueue), py::arg("location"),
          py::call_guard<py::gil_scoped_release>())
      .def("cat", &MoodistProcessGroup::cat, py::call_guard<py::gil_scoped_release>())
      .def("copy", &MoodistProcessGroup::copy, py::call_guard<py::gil_scoped_release>())
      .def("_set_group_name", &MoodistProcessGroup::setGroupName)
      .def_property_readonly("group_name", &MoodistProcessGroup::getGroupName);

  py::class_<MoodistBackend, c10::intrusive_ptr<MoodistBackend>, c10d::Backend>(m, "MoodistBackend", R"d(
    A moodist process group :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>(), py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::Future>(m, "Future")
      .def("wait", &moodist::Future::wait, py::call_guard<py::gil_scoped_release>())
      .def("result", &moodist::Future::result, py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::Cache, std::shared_ptr<moodist::Cache>>(m, "Cache").def("discard", &moodist::Cache::discard);

  m.def("enable_profiling", [](bool b) {
    printf("enable profiling -> %d\n", b);
    moodist::profilingEnabled = b;
  });
  m.def("enable_cuda_allocator", &moodist::enableCudaAllocator);
  m.def("enable_cpu_allocator", &moodist::enableCpuAllocator);
  m.def("cpu_allocator_debug", &moodist::cpuAllocatorDebug);

  m.def("cuda_copy", &cudaCopy, py::call_guard<py::gil_scoped_release>());

  m.def("cache", &moodist::cache, py::call_guard<py::gil_scoped_release>());

  py::class_<moodist::Queue, std::shared_ptr<moodist::Queue>>(m, "Queue")
      .def(
          "put", &moodist::Queue::put, py::arg("tensor"), py::arg("transaction"),
          py::call_guard<py::gil_scoped_release>())
      .def("get", &moodist::Queue::get, py::arg("block"), py::arg("timeout"), py::call_guard<py::gil_scoped_release>())
      .def("qsize", &moodist::Queue::qsize, py::call_guard<py::gil_scoped_release>())
      .def("wait", &moodist::Queue::wait, py::arg("timeout"), py::call_guard<py::gil_scoped_release>())
      .def("transaction_begin", &moodist::Queue::transactionBegin, py::call_guard<py::gil_scoped_release>())
      .def("transaction_cancel", &moodist::Queue::transactionCancel, py::call_guard<py::gil_scoped_release>())
      .def("transaction_commit", &moodist::Queue::transactionCommit, py::call_guard<py::gil_scoped_release>());
  py::class_<moodist::QueueWork>(m, "QueueWork")
      .def("wait", &moodist::QueueWork::wait, py::call_guard<py::gil_scoped_release>());
}
