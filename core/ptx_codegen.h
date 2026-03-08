// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <string>

namespace moodist {

struct CopyKernelConfig;
struct CustomOpDescriptor;
struct Group;
namespace compile_op {
struct CompileContext;
}
// Generate a complete copy kernel as PTX source, bypassing NVRTC.
// Uses the PTX DSL (ptx.h) to emit the same algorithm as v8.
// When op is non-null, the codegen can access the full compile op descriptor
// (e.g. for ring synchronization). Null during tuning micro-benchmarks.
std::string generateCopyKernelPtx(const Group* group, const CopyKernelConfig& config, const char* target,
    const CustomOpDescriptor& op, compile_op::CompileContext& ctx);

// Return the PTX target string (e.g. "sm_90a") for the given compute capability.
const char* computeTarget(int computeMajor, int computeMinor);

namespace ptx {
// Test function: generates simple kernels and returns the PTX string.
std::string ptxTest(const char* target);
} // namespace ptx

} // namespace moodist
