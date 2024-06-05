#pragma once

#include "common.h"

namespace moodist {

struct ModelOutput {
  size_t numDevices = 0;
  size_t numChunks = 0;
  size_t numParallel = 0;
};

ModelOutput evalModel(size_t worldSize, size_t localRanks, size_t bytes);

} // namespace moodist
