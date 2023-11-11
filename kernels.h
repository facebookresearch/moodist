#pragma once

#include "collective_base.h"
#include "common.h"

namespace moodist {

enum class ReduceOp {
  sum,
};
enum Dtype {
  float32,
};

struct Group;
struct Kernels : CollectiveBase {
  CUmodule cuModule = nullptr;
  Kernels(Group* group) : CollectiveBase(group) {}
  Kernels(const Kernels&) = delete;
  ~Kernels();
  void compile();

  size_t gridSize;
  size_t blockSize;

  CUfunction cuAllGatherLocal = nullptr;
  CUfunction cuAllGather = nullptr;
  CUfunction cuAllGatherCopyKernel = nullptr;

  CUfunction cuReduceScatterLocal = nullptr;
  CUfunction cuReduceScatter = nullptr;

  CUfunction cuBroadcast = nullptr;

  std::string emitCopySeq(
      std::vector<std::string> sources, std::vector<std::string> destinations, std::string bytes, size_t gridSize,
      size_t blockSize, std::string threadIndex, std::string blockIndex);

  std::string emitReduceFunctionSeq(
      std::string type, size_t typesize, size_t nsources, std::string op, size_t gridSize, size_t blockSize);
};

} // namespace moodist
