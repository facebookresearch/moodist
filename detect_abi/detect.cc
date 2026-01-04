// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Simple ABI detection module using Python stable API (no pybind11)
// Only needs to successfully link against PyTorch to detect correct ABI.

#define PY_SSIZE_T_CLEAN
#define Py_LIMITED_API 0x03090000
#include <Python.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace moodist {

// Inherit from ProcessGroup to test ABI compatibility
class ProcessGroup final : public c10d::ProcessGroup {
 public:
  ProcessGroup() : c10d::ProcessGroup(0, 1) {}
};

} // namespace moodist

static PyMethodDef module_methods[] = {
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "moodist_detect_abi",
    "ABI detection module",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit_moodist_detect_abi(void) {
  // Force linker to include ProcessGroup symbol
  (void)sizeof(moodist::ProcessGroup);
  return PyModule_Create(&module_def);
}
