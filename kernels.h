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

  std::string emitReduceFunction(std::string type, size_t typesize, size_t nsources, std::string op, size_t blockSize);
};

} // namespace moodist
