
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <memory>

namespace moodist {

struct ProcessGroupImpl;

class TORCH_API ProcessGroup {
public:
  std::unique_ptr<ProcessGroupImpl> impl;

  ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size);
  ~ProcessGroup();

  void all_gather_list(std::vector<at::Tensor>& output, const at::Tensor& input);
  void all_gather(at::Tensor& output, const at::Tensor& input);

  void reduce_scatter_list(at::Tensor& output, const std::vector<at::Tensor>& input, c10d::ReduceOp op);
  void reduce_scatter(at::Tensor& output, const at::Tensor& input, c10d::ReduceOp op);

  void barrier();
};

} // namespace moodist
