// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup wrapper - thin shell that inherits from c10d::ProcessGroup
// and calls into libmoodist.so via coreApi.
//
// This file should be minimal - just convert types and call core.
// All collective operation logic belongs in ProcessGroupImpl (core).

#include "backend.h"
#include "moodist_loader.h"
#include "processgroup_wrapper.h"
#include "tensor_ptr.h"
#include "torch_includes.h"

namespace moodist {

// ============================================================================
// Helpers
// ============================================================================

static ReduceOp toReduceOp(c10d::ReduceOp op) {
  switch (static_cast<int>(op)) {
  case static_cast<int>(c10d::ReduceOp::SUM):
    return ReduceOp::SUM;
  case static_cast<int>(c10d::ReduceOp::AVG):
    return ReduceOp::AVG;
  case static_cast<int>(c10d::ReduceOp::PRODUCT):
    return ReduceOp::PRODUCT;
  case static_cast<int>(c10d::ReduceOp::MIN):
    return ReduceOp::MIN;
  case static_cast<int>(c10d::ReduceOp::MAX):
    return ReduceOp::MAX;
  case static_cast<int>(c10d::ReduceOp::PREMUL_SUM):
    return ReduceOp::PREMUL_SUM;
  default:
    throw std::runtime_error("Unsupported reduce op");
  }
}

static float extractPremulValue(c10d::ReduceOp reduceOp, at::ScalarType dtype) {
  if (static_cast<int>(reduceOp) != static_cast<int>(c10d::ReduceOp::PREMUL_SUM)) {
    return 0.0f;
  }
  if (dtype != at::ScalarType::Float) {
    throw std::runtime_error("PREMUL_SUM only supported for float32");
  }
  auto* supplement = reinterpret_cast<c10d::NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
  if (supplement->tensor_factor.defined()) {
    return supplement->tensor_factor.item<float>();
  }
  return static_cast<float>(supplement->double_factor);
}

static CUstream getStream(const at::Tensor& t) {
  return t.is_cuda() ? c10::cuda::getCurrentCUDAStream(t.device().index()).stream() : nullptr;
}

// ============================================================================
// WorkImpl
// ============================================================================

struct WorkImpl : c10d::Work {
  std::vector<torch::Tensor> tensors_;

  WorkImpl() = default;
  explicit WorkImpl(torch::Tensor t) {
    tensors_.push_back(std::move(t));
  }
  explicit WorkImpl(std::vector<torch::Tensor> ts) : tensors_(std::move(ts)) {}

  void synchronize() override {}
  bool wait(std::chrono::milliseconds = std::chrono::milliseconds(0)) override {
    return true;
  }
  bool isCompleted() override {
    return true;
  }
  bool isSuccess() const override {
    return true;
  }
  std::vector<at::Tensor> result() override {
    return tensors_;
  }
};

// ============================================================================
// Construction / Destruction
// ============================================================================

ProcessGroup::ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size)
    : c10d::ProcessGroup(rank, size) {
  // Pass c10d::Store as opaque pointer - core will call back via WrapperApi for store operations
  impl = coreApi.createProcessGroupImpl(store.get(), rank, size);

  setBackend(torch::DeviceType::CPU, c10d::ProcessGroup::CUSTOM, c10::make_intrusive<Backend>(*this));
  setBackend(torch::DeviceType::CUDA, c10d::ProcessGroup::CUSTOM, c10::make_intrusive<Backend>(*this));
}

ProcessGroup::~ProcessGroup() {
  if (impl) {
    coreApi.processGroupImplDecRef(impl);
    impl = nullptr;
  }
}

// ============================================================================
// Collectives - thin wrappers that convert types and call core
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroup::allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions&) {
  TORCH_CHECK(outputTensors.size() == 1 && inputTensors.size() == 1);
  // Flatten output list to contiguous buffer for core
  auto& input = inputTensors[0];
  auto output = torch::empty({static_cast<int64_t>(getSize()), input.numel()}, input.options());

  TensorPtr inPtr = wrapTensor(input);
  TensorPtr outPtr = wrapTensor(output);
  coreApi.processGroupImplAllGather(impl, outPtr, inPtr, getStream(input));

  // Copy back to output list
  for (int i = 0; i < getSize(); ++i) {
    outputTensors[0][i].copy_(output[i]);
  }

  return c10::make_intrusive<WorkImpl>(outputTensors[0]);
}

c10::intrusive_ptr<Work> ProcessGroup::_allgather_base(
    at::Tensor& output, at::Tensor& input, const c10d::AllgatherOptions&) {
  TensorPtr inPtr = wrapTensor(input);
  TensorPtr outPtr = wrapTensor(output);
  coreApi.processGroupImplAllGather(impl, outPtr, inPtr, getStream(input));
  return c10::make_intrusive<WorkImpl>(output);
}

c10::intrusive_ptr<Work> ProcessGroup::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const c10d::AllgatherOptions&) {
  TORCH_CHECK(outputs.size() == inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorPtr inPtr = wrapTensor(inputs[i]);
    TensorPtr outPtr = wrapTensor(outputs[i]);
    coreApi.processGroupImplAllGather(impl, outPtr, inPtr, getStream(inputs[i]));
  }
  return c10::make_intrusive<WorkImpl>(outputs);
}

c10::intrusive_ptr<Work> ProcessGroup::_reduce_scatter_base(
    at::Tensor& output, at::Tensor& input, const c10d::ReduceScatterOptions& opts) {
  TensorPtr inPtr = wrapTensor(input);
  TensorPtr outPtr = wrapTensor(output);
  coreApi.processGroupImplReduceScatter(impl, outPtr, inPtr, toReduceOp(opts.reduceOp), getStream(input),
      extractPremulValue(opts.reduceOp, input.scalar_type()));
  return c10::make_intrusive<WorkImpl>(output);
}

c10::intrusive_ptr<Work> ProcessGroup::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(outputs.size() == inputs.size());
  ReduceOp op = toReduceOp(opts.reduceOp);
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorPtr inPtr = wrapTensor(inputs[i]);
    TensorPtr outPtr = wrapTensor(outputs[i]);
    coreApi.processGroupImplReduceScatter(
        impl, outPtr, inPtr, op, getStream(inputs[i]), extractPremulValue(opts.reduceOp, inputs[i].scalar_type()));
  }
  return c10::make_intrusive<WorkImpl>(outputs);
}

c10::intrusive_ptr<Work> ProcessGroup::allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  TORCH_CHECK(!tensors.empty());
  auto& t = tensors[0];
  TensorPtr tPtr = wrapTensor(t);
  coreApi.processGroupImplAllreduce(
      impl, tPtr, toReduceOp(opts.reduceOp), getStream(t), extractPremulValue(opts.reduceOp, t.scalar_type()));
  return c10::make_intrusive<WorkImpl>(t);
}

c10::intrusive_ptr<Work> ProcessGroup::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceCoalescedOptions& opts) {
  ReduceOp op = toReduceOp(opts.reduceOp);
  for (auto& t : tensors) {
    TensorPtr tPtr = wrapTensor(t);
    coreApi.processGroupImplAllreduce(impl, tPtr, op, getStream(t), extractPremulValue(opts.reduceOp, t.scalar_type()));
  }
  return c10::make_intrusive<WorkImpl>(tensors);
}

c10::intrusive_ptr<Work> ProcessGroup::broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(!tensors.empty());
  auto& t = tensors[0];
  TensorPtr tPtr = wrapTensor(t);
  coreApi.processGroupImplBroadcast(impl, tPtr, opts.rootRank, getStream(t));
  return c10::make_intrusive<WorkImpl>(t);
}

c10::intrusive_ptr<Work> ProcessGroup::reduce(std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
  TORCH_CHECK(!tensors.empty());
  auto& t = tensors[0];
  TensorPtr tPtr = wrapTensor(t);
  coreApi.processGroupImplReduce(impl, tPtr, opts.rootRank, toReduceOp(opts.reduceOp), getStream(t));
  return c10::make_intrusive<WorkImpl>(t);
}

c10::intrusive_ptr<Work> ProcessGroup::barrier(const c10d::BarrierOptions&) {
  coreApi.processGroupImplBarrier(impl);
  return c10::make_intrusive<WorkImpl>();
}

c10::intrusive_ptr<Work> ProcessGroup::scatter(
    std::vector<at::Tensor>& outputs, std::vector<std::vector<at::Tensor>>& inputs, const c10d::ScatterOptions& opts) {
  TORCH_CHECK(!outputs.empty());
  auto& output = outputs[0];

  std::vector<TensorPtr> inPtrs;
  if (!inputs.empty() && !inputs[0].empty()) {
    for (auto& t : inputs[0]) {
      inPtrs.push_back(wrapTensor(t));
    }
  }

  TensorPtr outPtr = wrapTensor(output);
  coreApi.processGroupImplScatter(impl, inPtrs, outPtr, opts.rootRank, getStream(output));

  return c10::make_intrusive<WorkImpl>(output);
}

c10::intrusive_ptr<Work> ProcessGroup::gather(
    std::vector<std::vector<at::Tensor>>& outputs, std::vector<at::Tensor>& inputs, const c10d::GatherOptions& opts) {
  TORCH_CHECK(!inputs.empty());
  auto& input = inputs[0];

  std::vector<TensorPtr> outPtrs;
  if (!outputs.empty() && !outputs[0].empty()) {
    for (auto& t : outputs[0]) {
      outPtrs.push_back(wrapTensor(t));
    }
  }

  TensorPtr inPtr = wrapTensor(input);
  coreApi.processGroupImplGather(impl, outPtrs, inPtr, opts.rootRank, getStream(input));

  return c10::make_intrusive<WorkImpl>(outputs.empty() ? std::vector<torch::Tensor>{} : outputs[0]);
}

c10::intrusive_ptr<Work> ProcessGroup::alltoall(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const c10d::AllToAllOptions&) {
  std::vector<TensorPtr> outPtrs;
  std::vector<TensorPtr> inPtrs;
  for (auto& t : outputs) {
    outPtrs.push_back(wrapTensor(t));
  }
  for (auto& t : inputs) {
    inPtrs.push_back(wrapTensor(t));
  }

  coreApi.processGroupImplAllToAll(impl, outPtrs, inPtrs, getStream(inputs[0]));

  return c10::make_intrusive<WorkImpl>(outputs);
}

c10::intrusive_ptr<Work> ProcessGroup::alltoall_base(
    at::Tensor& output, at::Tensor& input, std::vector<int64_t>&, std::vector<int64_t>&, const c10d::AllToAllOptions&) {
  // Split into per-rank tensors
  auto outs = output.tensor_split(getSize());
  auto ins = input.tensor_split(getSize());

  std::vector<TensorPtr> outPtrs;
  std::vector<TensorPtr> inPtrs;
  for (auto& t : outs) {
    outPtrs.push_back(wrapTensor(t));
  }
  for (auto& t : ins) {
    inPtrs.push_back(wrapTensor(t));
  }

  coreApi.processGroupImplAllToAll(impl, outPtrs, inPtrs, getStream(input));

  return c10::make_intrusive<WorkImpl>(output);
}

// ============================================================================
// Other methods
// ============================================================================

void ProcessGroup::cudaBarrier() {
  coreApi.processGroupImplCudaBarrier(impl, c10::cuda::getCurrentCUDAStream().stream());
}

void ProcessGroup::shutdown() {
  if (impl) {
    coreApi.processGroupImplShutdown(impl);
  }
}

std::string ProcessGroup::moodist_name() const {
  return "moodist"; // TODO: get from impl if needed
}

// ============================================================================
// Stubs - need additional CoreApi functions
// ============================================================================

std::vector<torch::Tensor> ProcessGroup::share(const torch::Tensor&) {
  throw std::runtime_error("share: not yet implemented");
}

std::shared_ptr<Queue> ProcessGroup::makeQueue(int, bool, std::optional<std::string>) {
  throw std::runtime_error("makeQueue: not yet implemented");
}

std::shared_ptr<Queue> ProcessGroup::makeQueue(std::vector<int>, bool, std::optional<std::string>) {
  throw std::runtime_error("makeQueue: not yet implemented");
}

Future ProcessGroup::cat(const std::vector<std::pair<int, torch::Tensor>>&, std::optional<torch::Tensor>) {
  throw std::runtime_error("cat: not yet implemented");
}

Future ProcessGroup::copy(torch::Tensor&, const torch::Tensor&) {
  throw std::runtime_error("copy: not yet implemented");
}

CustomOp ProcessGroup::compileOpFull(const std::vector<int>&, torch::Dtype,
    const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>&,
    const std::vector<std::tuple<int, std::vector<int>, std::vector<int>>>&) {
  throw std::runtime_error("compileOpFull: not yet implemented");
}

// ============================================================================
// Free memory callback
// ============================================================================

struct FreeMemoryCallbackImpl : at::FreeMemoryCallback {
  bool Execute() override {
    // TODO: Call into core to free memory
    return false;
  }
};

void registerFreeMemoryCallback() {
  at::FreeCudaMemoryCallbacksRegistry()->Register("moodist", []() {
    return std::make_unique<FreeMemoryCallbackImpl>();
  });
}

} // namespace moodist
