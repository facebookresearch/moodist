#pragma once

#include "common.h"
#include "vector.h"

namespace moodist {

struct ParametersData {
  SpinMutex mutex;
  Vector<std::tuple<size_t, float>> timings;

  void addTiming(size_t index, float v) {
    std::lock_guard l(mutex);
    timings.emplace_back(index, v);
  }
};

} // namespace moodist
