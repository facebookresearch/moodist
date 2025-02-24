#pragma once

#include "common.h"
#include "hash_map.h"

namespace moodist {

namespace cuda_copy {
inline SpinMutex registerMutex;
inline HashMap<uintptr_t, size_t> registered;

inline void ensureRegistered(uintptr_t address) {
  auto region = cpu_allocator::regionAt(address);
  if (region.first) {
    std::lock_guard l(registerMutex);
    auto& s = registered[region.first];
    if (s != region.second) {
      CHECK(region.second > s);
      if (s != 0) {
        CHECK_CU(cuCtxSynchronize());
        CHECK_CU(cuMemHostUnregister((void*)region.first));
      }
      CHECK_CU(cuMemHostRegister(
          (void*)region.first, region.second, CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_PORTABLE));
      s = region.second;
    }
  }
}

inline void copy(uintptr_t dstAddress, uintptr_t srcAddress, size_t bytes, CUstream stream) {
  ensureRegistered(dstAddress);
  ensureRegistered(srcAddress);
  try {
    CHECK_CU(cuMemcpyAsync(dstAddress, srcAddress, bytes, stream));
  } catch (...) {
    log.error("failed to copy %#x %#x %d\n", dstAddress, srcAddress, bytes);
    throw;
  }
}

} // namespace cuda_copy

inline void cudaCopy(uintptr_t dstAddress, uintptr_t srcAddress, size_t bytes, CUstream stream) {
  cuda_copy::copy(dstAddress, srcAddress, bytes, stream);
}

} // namespace moodist
