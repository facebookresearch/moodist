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

  AllocatedBuffer profilingData;
  size_t profilingBytesPerThread = 0;
  Vector<std::string> profilingNames;
  size_t profilingCountdown = 0;
};

std::shared_ptr<KernelHandle> generateKernel(const Group* group, const KernelConfig& config, std::string target,
    const compile_op::Graph& graph, compile_op::CompileContext& ctx);

// Return the PTX target string (e.g. "sm_90a") for the given compute capability.
const char* computeTarget(int computeMajor, int computeMinor);

} // namespace moodist
