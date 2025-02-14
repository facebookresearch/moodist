
#include "common.h"
#include "queue.h"

#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <memory>

namespace moodist {

struct ProcessGroupImpl;

struct FutureImpl;
struct Future {
  FutureImplSharedPtr impl;
  Future();
  ~Future();
  Future(const Future&) = delete;
  Future(Future&&) = default;
  void wait();
  torch::Tensor result();
};

using Work = c10d::Work;

class TORCH_API ProcessGroup final : public c10d::ProcessGroup {
public:
  std::unique_ptr<ProcessGroupImpl> impl;

  ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size);
  ~ProcessGroup();

  const std::string getBackendName() const override {
    return "moodist";
  }

  c10::intrusive_ptr<Work>
  broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<Work>
  allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override {
    TORCH_CHECK(false, "allreduce_coalesced not supported");
  }

  c10::intrusive_ptr<Work>
  reduce(std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer, at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    TORCH_CHECK(false, "allgather_coalesced not supported");
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override {
    TORCH_CHECK(false, "reduce_scatter not supported");
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes, const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    TORCH_CHECK(false, "alltoall_base not supported");
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    TORCH_CHECK(false, "alltoall not supported");
  }

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override {
    TORCH_CHECK(false, "send not supported");
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override {
    TORCH_CHECK(false, "recv not supported");
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override {
    TORCH_CHECK(false, "scatter not supported");
  }

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor>& tensors, int tag) override {
    TORCH_CHECK(false, "recvAnysource not supported");
  }

  std::shared_ptr<Queue> makeQueue(int location);
  std::shared_ptr<Queue> makeQueue(std::vector<int> location);

  Future cat(const std::vector<std::pair<int, torch::Tensor>>& locals);
  Future copy(torch::Tensor& destination, const torch::Tensor& source);
};

} // namespace moodist
