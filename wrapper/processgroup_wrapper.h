// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup wrapper header - minimal header for _C.so
// Only includes what's needed for the c10d::ProcessGroup wrapper.
// Does NOT include core headers (common.h, etc.)

#pragma once

#include "api/types.h"
#include "torch_includes.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace moodist {

// Forward declarations
class Queue;

// QueueWork - wraps api::QueueWork via ApiHandle (unique ownership)
struct QueueWork {
  api::QueueWorkHandle handle;

  QueueWork() = default;
  QueueWork(api::QueueWorkHandle h) : handle(std::move(h)) {}
  ~QueueWork() = default; // ApiHandle destructor handles cleanup

  QueueWork(const QueueWork&) = delete;
  QueueWork(QueueWork&&) noexcept = default;
  QueueWork& operator=(const QueueWork&) = delete;
  QueueWork& operator=(QueueWork&&) noexcept = default;

  void wait();
};

// Queue - wrapper around api::Queue managed via ApiHandle
class Queue {
public:
  api::QueueHandle handle; // Manages refcount automatically

  Queue(api::QueueHandle h) : handle(std::move(h)) {}
  ~Queue() = default; // ApiHandle destructor handles cleanup

  // Wrapper methods that call CoreApi functions
  std::pair<std::optional<torch::Tensor>, size_t> get(bool block = true, std::optional<float> timeout = {});
  QueueWork put(torch::Tensor tensor, uint32_t transaction, bool waitOnDestroy = true);
  size_t qsize() const;
  bool wait(std::optional<float> timeout) const;
  uint32_t transactionBegin();
  void transactionCancel(uint32_t id);
  void transactionCommit(uint32_t id);
  std::string_view name() const;
};

// Stub types for unimplemented methods - minimal definitions
struct Future {
  api::FutureHandle handle;
  std::vector<torch::Tensor> holdTensors; // Keep tensors alive in wrapper

  Future() = default;
  Future(api::FutureHandle h) : handle(std::move(h)) {}
  ~Future() = default; // ApiHandle destructor handles cleanup

  Future(const Future&) = delete;
  Future(Future&&) noexcept = default;
  Future& operator=(const Future&) = delete;
  Future& operator=(Future&&) noexcept = default;

  void wait();
  torch::Tensor result();
};

struct CustomOp {
  api::CustomOpHandle handle;

  CustomOp() = default;
  CustomOp(api::CustomOpHandle h) : handle(std::move(h)) {}
  ~CustomOp() = default; // ApiHandle destructor handles cleanup

  CustomOp(const CustomOp&) = delete;
  CustomOp(CustomOp&&) noexcept = default;
  CustomOp& operator=(const CustomOp&) = delete;
  CustomOp& operator=(CustomOp&&) noexcept = default;

  Future operator()(const std::vector<torch::Tensor>& inputs, const std::vector<torch::Tensor>& outputs);
};

using Work = c10d::Work;

class TORCH_API ProcessGroup final : public c10d::ProcessGroup {
public:
  api::ProcessGroupHandle handle; // Manages refcount automatically

  ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size);
  ~ProcessGroup() = default; // ApiHandle destructor handles cleanup

  const std::string getBackendName() const override {
    return "moodist";
  }

  std::string moodist_name() const;

  // Standard c10d collectives
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts = c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(at::Tensor& outputbuffer, at::Tensor& inputbuffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    throw std::runtime_error("allgather_coalesced not supported");
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override {
    throw std::runtime_error("reduce_scatter not supported");
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(at::Tensor& outputTensor, at::Tensor& inputTensor,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors, const c10d::ReduceScatterOptions& opts) override;

  c10::intrusive_ptr<Work> barrier(const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(at::Tensor& outputTensor, at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes, std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(std::vector<at::Tensor>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override {
    throw std::runtime_error("send not supported");
  }

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override {
    throw std::runtime_error("recv not supported");
  }

  c10::intrusive_ptr<Work> gather(std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors, const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor>& tensors, int tag) override {
    throw std::runtime_error("recvAnysource not supported");
  }

  void cudaBarrier();
  void shutdown();

  // Stubs for unimplemented features - these use core types that don't cross the API boundary
  std::shared_ptr<Queue> makeQueue(
      int location, bool streaming = false, std::optional<std::string> name = std::nullopt);
  std::shared_ptr<Queue> makeQueue(
      std::vector<int> location, bool streaming = false, std::optional<std::string> name = std::nullopt);
  Future cat(const std::vector<std::pair<int, torch::Tensor>>& locals, std::optional<torch::Tensor> out = std::nullopt);
  Future copy(torch::Tensor& destination, const torch::Tensor& source);
  std::vector<torch::Tensor> share(const torch::Tensor& input);
  CustomOp compileOpFull(const std::vector<int>& shape, torch::Dtype dtype,
      const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>& inputs,
      const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>& outputs);
};

void registerFreeMemoryCallback();

} // namespace moodist
