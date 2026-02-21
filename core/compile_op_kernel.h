// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

namespace moodist {

struct Group;

// Max descriptors that fit in 4KB kernel params.
// Each descriptor: 20 bytes (src:8 + dst:8 + bytes:4)
// Overhead: ~16 bytes (stepValue, concurrencyIndex, numDescriptors, padding)
// (4096 - 16) / 20 = ~204
static constexpr uint32_t kMaxCopyDescriptors = 200;

struct CopyDescriptor {
  uintptr_t src;
  uintptr_t dst;
  uint32_t bytes; // Max 4GB per region, sufficient for tensor slices
};

struct CompileOpCopyParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  uint32_t numDescriptors;
  uint32_t _pad;
  CopyDescriptor descriptors[kMaxCopyDescriptors];
};

struct CompileOpKernels {
  Group* group;
  CUmodule cuModule = nullptr;
  CUfunction cuCopyKernel = nullptr;

  CUmodule cuMulticastModule = nullptr;
  CUfunction cuMulticastKernel = nullptr;

  size_t gridSize = 64;
  size_t blockSize = 256;

  int version = 0; // From MOODIST_COPY_KERNEL env var

  CompileOpKernels(Group* group);
  ~CompileOpKernels();
  void compile();
  void compileMulticast();
};

} // namespace moodist
