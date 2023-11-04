
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <memory>

namespace moodist {

struct ProcessGroupImpl;

class TORCH_API ProcessGroup : public torch::CustomClassHolder {
public:
  std::unique_ptr<ProcessGroupImpl> impl;

  ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size);
  ~ProcessGroup();

  void broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions());

  void allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  void allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) {
    TORCH_CHECK(false, "allreduce_coalesced called");
  }

  void reduce(std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts = c10d::ReduceOptions()) {
    TORCH_CHECK(false, "reduce called");
  }

  void allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

  void _allgather_base(
      at::Tensor& outputbuffer, at::Tensor& inputbuffer, const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

  void allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) {
    TORCH_CHECK(false, "allgather_coalesced called");
  }

  void reduce_scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) {
    TORCH_CHECK(false, "reduce_scatter called");
  }

  void _reduce_scatter_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions());

  void barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions());

  void alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes, const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) {
    TORCH_CHECK(false, "alltoall_base called");
  }

  void alltoall(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) {
    TORCH_CHECK(false, "alltoall called");
  }

  void send(std::vector<at::Tensor>& tensors, int dstRank, int tag) {
    TORCH_CHECK(false, "send called");
  }

  void recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
    TORCH_CHECK(false, "recv called");
  }

  void gather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) {
    TORCH_CHECK(false, "gather called");
  }

  void scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) {
    TORCH_CHECK(false, "scatter called");
  }

  void recvAnysource(std::vector<at::Tensor>& tensors, int tag) {
    TORCH_CHECK(false, "recvAnysource called");
  }
};

} // namespace moodist
