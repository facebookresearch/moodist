
#include <torch/torch.h>

#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
namespace tccl_work = c10d;
#else
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
using tccl_work = c10d::ProcessGroup;
#endif

#include <memory>

namespace moodist {

struct ProcessGroupImpl;

class TORCH_API ProcessGroup : public torch::CustomClassHolder {
public:
  std::unique_ptr<ProcessGroupImpl> impl;

  ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size);
  ~ProcessGroup();

  c10::intrusive_ptr<tccl_work::Work>
  broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions());

  c10::intrusive_ptr<tccl_work::Work>
  allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts = c10d::AllreduceOptions());

  c10::intrusive_ptr<tccl_work::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) {
    TORCH_CHECK(false, "allreduce_coalesced called");
  }

  c10::intrusive_ptr<tccl_work::Work>
  reduce(std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts = c10d::ReduceOptions()) {
    TORCH_CHECK(false, "reduce called");
  }

  c10::intrusive_ptr<tccl_work::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

  c10::intrusive_ptr<tccl_work::Work> _allgather_base(
      at::Tensor& outputbuffer, at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions());

  c10::intrusive_ptr<tccl_work::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) {
    TORCH_CHECK(false, "allgather_coalesced called");
  }

  c10::intrusive_ptr<tccl_work::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) {
    TORCH_CHECK(false, "reduce_scatter called");
  }

  c10::intrusive_ptr<tccl_work::Work> _reduce_scatter_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions());

  c10::intrusive_ptr<tccl_work::Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions());

  c10::intrusive_ptr<tccl_work::Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes, const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) {
    TORCH_CHECK(false, "alltoall_base called");
  }

  c10::intrusive_ptr<tccl_work::Work> alltoall(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) {
    TORCH_CHECK(false, "alltoall called");
  }

  c10::intrusive_ptr<tccl_work::Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) {
    TORCH_CHECK(false, "send called");
  }

  c10::intrusive_ptr<tccl_work::Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
    TORCH_CHECK(false, "recv called");
  }

  c10::intrusive_ptr<tccl_work::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) {
    TORCH_CHECK(false, "gather called");
  }

  c10::intrusive_ptr<tccl_work::Work> scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) {
    TORCH_CHECK(false, "scatter called");
  }

  c10::intrusive_ptr<tccl_work::Work> recvAnysource(std::vector<at::Tensor>& tensors, int tag) {
    TORCH_CHECK(false, "recvAnysource called");
  }
};

} // namespace moodist
