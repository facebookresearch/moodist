// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <string>

namespace moodist {

struct KernelConfig;
struct CustomOpDescriptor;
struct Group;
namespace compile_op {
struct CompileContext;
struct Graph;
} // namespace compile_op
// Generate a complete copy kernel as PTX source, bypassing NVRTC.
// Uses the PTX DSL (ptx.h) to emit the same algorithm as v8.
// When op is non-null, the codegen can access the full compile op descriptor
// (e.g. for ring synchronization). Null during tuning micro-benchmarks.
std::string generateKernel(const Group* group, const KernelConfig& config, std::string target,
    const compile_op::Graph& graph, compile_op::CompileContext& ctx);

// Return the PTX target string (e.g. "sm_90a") for the given compute capability.
const char* computeTarget(int computeMajor, int computeMinor);

namespace ptx {
// Test function: generates simple kernels and returns the PTX string.
std::string ptxTest(const char* target);
} // namespace ptx

} // namespace moodist
