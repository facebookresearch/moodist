
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/utils/pybind.h>
#include <torch/torch.h>

namespace py = pybind11;

namespace moodist {

class TORCH_API ProcessGroup final : public c10d::ProcessGroup {
  ProcessGroup() : c10d::ProcessGroup(0, 1) {}
};

torch::Tensor foo(torch::Tensor x) {
  return x + 1;
}

std::string bar(std::string x) {
  return x + "1";
}

} // namespace moodist

PYBIND11_MODULE(moodist_detect_abi, m) {
  py::class_<moodist::ProcessGroup, c10::intrusive_ptr<moodist::ProcessGroup>, c10d::ProcessGroup>(m, "ProcessGroup");

  m.def("foo", &moodist::foo);
  m.def("bar", &moodist::bar);
}
