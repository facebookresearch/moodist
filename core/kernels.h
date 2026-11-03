// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "collective_base.h"
#include "common.h"

namespace moodist {

enum CompileFlags {
  CompileAllToAll = 8,
};

struct Group;
struct Kernels : CollectiveBase {
  std::vector<CUmodule> cuModules;
  Kernels(Group* group) : CollectiveBase(group) {}
  Kernels(const Kernels&) = delete;
  ~Kernels();
  void compile(int flags);

  size_t gridSize;
  size_t blockSize;

  CUfunction cuBroadcast = nullptr;

  CUfunction cuDummySignal = nullptr;

  CUfunction cuAllToAll = nullptr;

  std::string emitCopySeq(std::vector<std::string> sources, std::vector<std::string> destinations, std::string bytes,
      size_t gridSize, size_t blockSize, std::string threadIndex, std::string blockIndex);
};

} // namespace moodist
