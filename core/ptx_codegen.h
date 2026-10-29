// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"
#include <cstddef>
#include <memory>
#include <string>

namespace moodist {

struct KernelConfig;
struct CustomOpDescriptor;
struct Group;
namespace compile_op {
struct CompileContext;
struct Graph;
} // namespace compile_op

void returnKernelMemory(size_t key);

struct KernelHandle {
  std::string ptx;
  size_t memoryKey = 0;
  KernelHandle() = default;
  ~KernelHandle() {
    if (memoryKey) {
      returnKernelMemory(memoryKey);
    }
  }
  KernelHandle(const KernelHandle&) = delete;
  KernelHandle& operator=(const KernelHandle&) = delete;

  // Device memory the kernel uses (its address tables, when they do not fit in shared memory)
  AllocatedBuffer scratch;

  AllocatedBuffer profilingData;
  size_t profilingBytesPerThread = 0;
  Vector<std::string> profilingNames;
  size_t profilingCountdown = 0;

  // NVLS (see generateKernel): the op uses the group's multicast objects (on every rank of the group). customOp
  // passes the NVLS identities and multicast addresses of the tensors (see nvlsIdentity in processgroup.cc) in a
  // fourth kernel parameter, and the op's step counter (of the launch's concurrency slot) before the launch: every
  // launch adds nvlsIncrement (the chunks all ranks store with NVLS).
  bool nvls = false;
  uint32_t nvlsIncrement = 0;
  std::array<uint32_t, 16> nvlsLaunches = {}; // per concurrency slot
};

// The copy kernel of an op on this rank. With tuning, the kernel does not wait for the CPU thread's signals that
// nodes have been filled over RDMA (tuning launches run without the CPU thread).
std::shared_ptr<KernelHandle> generateKernel(const Group* group, const KernelConfig& config, std::string target,
    const compile_op::Graph& graph, compile_op::CompileContext& ctx, bool tuning);

// Why the kernels of ranks that execute reductions cannot run with config, or empty if they can
std::string reduceConfigError(const KernelConfig& config);

// Return the PTX target string (e.g. "sm_90a") for the given compute capability.
const char* computeTarget(int computeMajor, int computeMinor);

} // namespace moodist
