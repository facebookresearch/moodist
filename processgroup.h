#pragma once

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
  std::vector<torch::Tensor> tensors;
  std::optional<torch::Tensor> out;
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

  std::string moodist_name() const;

  c10::intrusive_ptr<Work>
  broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<Work>
  allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override;

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
    throw std::runtime_error("allgather_coalesced not supported");
  }
  virtual c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override {
    throw std::runtime_error("reduce_scatter not supported");
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  virtual c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::ReduceScatterOptions& opts) override;

  c10::intrusive_ptr<Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor, std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes, const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override {
    throw std::runtime_error("send not supported");
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override {
    throw std::runtime_error("recv not supported");
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors, std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor>& tensors, int tag) override {
    throw std::runtime_error("recvAnysource not supported");
  }

  std::shared_ptr<Queue>
  makeQueue(int location, bool streaming = false, std::optional<std::string> name = std::nullopt);
  std::shared_ptr<Queue>
  makeQueue(std::vector<int> location, bool streaming = false, std::optional<std::string> name = std::nullopt);

  Future cat(const std::vector<std::pair<int, torch::Tensor>>& locals, std::optional<torch::Tensor> out = std::nullopt);
  Future copy(torch::Tensor& destination, const torch::Tensor& source);

  std::vector<torch::Tensor> share(const torch::Tensor& input);
  void cudaBarrier();
};

} // namespace moodist
