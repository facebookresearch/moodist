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

  CUfunction cuReduce = nullptr;
  size_t reduceGridSize = 0;
  size_t reduceBlockSize = 0;
};

} // namespace moodist
