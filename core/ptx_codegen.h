// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <string>

namespace moodist {

struct Group;

// Generate a complete copy kernel as PTX source, bypassing NVRTC.
// Uses the PTX DSL (ptx.h) to emit the same algorithm as v8.
std::string generateCopyKernelPtx(Group* group, size_t gridSize, size_t blockSize, int depth, const char* target);

namespace ptx {
// Test function: generates simple kernels and returns the PTX string.
std::string ptxTest(const char* target);
} // namespace ptx

} // namespace moodist
