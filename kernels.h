#pragma once

#include "collective_base.h"
#include "common.h"

namespace moodist {

enum CompileFlags {
  CompileAllGather = 1,
  CompileReduceScatter = 2,
};

struct Group;
struct Kernels : CollectiveBase {
  std::vector<CUmodule> cuModules;
  Kernels(Group* group) : CollectiveBase(group) {}
  Kernels(const Kernels&) = delete;
  ~Kernels();
  void compile(int flags, std::string type = "", std::string reduction = "");

  size_t gridSize;
  size_t blockSize;

  CUfunction cuAllGatherLocal = nullptr;
  CUfunction cuAllGather = nullptr;
  CUfunction cuAllGatherNoLocal = nullptr;
  CUfunction cuAllGatherCopyKernel = nullptr;

  std::array<std::array<CUfunction, (size_t)Reduction::count>, (size_t)Dtype::count> cuReduceScatterLocal{};
  std::array<std::array<CUfunction, (size_t)Reduction::count>, (size_t)Dtype::count> cuReduceScatter{};

  CUfunction cuBroadcast = nullptr;

  std::string emitCopySeq(
      std::vector<std::string> sources, std::vector<std::string> destinations, std::string bytes, size_t gridSize,
      size_t blockSize, std::string threadIndex, std::string blockIndex);

  std::string emitReduceFunctionSeq(
      std::string type, size_t typesize, size_t nsources, std::string op, size_t gridSize, size_t blockSize,
      std::string threadIndex, std::string blockIndex);
};

} // namespace moodist
