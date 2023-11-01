/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pytypes.h>
#undef __GXX_ABI_VERSION
#define __GXX_ABI_VERSION 1011

#include <torch/csrc/utils/pybind.h>

#include "processgroup.h"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

// __pybind11_internals_v4_gcc_libstdcpp_cxxabi1013
// __pybind11_internals_v4_gcc_libstdcpp_cxxabi1011

namespace moodist {
bool profilingEnabled = false;
}

namespace py = pybind11;

using MoodistProcessGroup = moodist::ProcessGroup;

PYBIND11_MODULE(_C, m) {
  py::class_<MoodistProcessGroup, c10::intrusive_ptr<MoodistProcessGroup>, c10d::ProcessGroup>(
      m, "MoodistProcessGroup", R"d(
    A moodist process group :D
  )d")
      .def(py::init<const c10::intrusive_ptr<::c10d::Store>&, int, int>());
  m.def("enable_profiling", [](bool b) {
    printf("enable profiling -> %d\n", b);
    moodist::profilingEnabled = b;
  });
}
