
#include <torch/torch.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <memory>

#include "processgroup.h"

namespace moodist {

struct ProcessGroupImpl;

using Work = c10d::Work;

class TORCH_API Backend final : public c10d::Backend {
public:
  std::unique_ptr<ProcessGroup> pgptr;
  ProcessGroup& pg;

  Backend(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size)
      : c10d::Backend(rank, size), pgptr(std::make_unique<ProcessGroup>(store, rank, size)), pg(*pgptr) {}
  Backend(ProcessGroup& pg) : c10d::Backend(pg.getRank(), pg.getSize()), pg(pg) {}
  ~Backend() {}

  const std::string getBackendName() const override {
    return "moodist";
  }

  c10::intrusive_ptr<Work>
  broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    return pg.broadcast(tensors, opts);
  }

  c10::intrusive_ptr<Work>
  allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    return pg.allreduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override {
    TORCH_CHECK(false, "allreduce_coalesced not supported");
  }

  virtual c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    return pg.allgather_into_tensor_coalesced(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work>
  reduce(std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override {
    return pg.reduce(tensors, opts);
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    return pg.allgather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer, at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    return pg._allgather_base(outputbuffer, inputbuffer, opts);
  }

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
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override {
    return pg._reduce_scatter_base(outputTensor, inputTensor, opts);
  }

  virtual c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::ReduceScatterOptions& opts) override {
    return pg.reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override {
    return pg.barrier(opts);
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes, const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    TORCH_CHECK(false, "alltoall_base not supported");
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    return pg.alltoall(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override {
    TORCH_CHECK(false, "send not supported");
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override {
    TORCH_CHECK(false, "recv not supported");
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override {
    return pg.gather(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override {
    return pg.scatter(outputTensors, inputTensors, opts);
  }

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor>& tensors, int tag) override {
    TORCH_CHECK(false, "recvAnysource not supported");
  }
};

} // namespace moodist
