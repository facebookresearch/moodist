#pragma once

#include "common.h"

namespace moodist {

struct Group;
struct Kernels {
  Group* group;
  CUmodule cuModule = nullptr;
  Kernels(Group* group) : group(group) {}
  Kernels(const Kernels&) = delete;
  ~Kernels();
  void compile();
};

} // namespace moodist
