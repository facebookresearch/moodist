
#include <cstdint>
#include <string>

#include "common.h"
#include "vector.h"

namespace moodist {

namespace allocator {

bool owns(uintptr_t address);
uintptr_t offset(uintptr_t address);

std::pair<uintptr_t, size_t> reserved();

struct PeerMemory {
  void* impl = nullptr;
  uintptr_t remoteBaseAddress();
  void remoteExtend(uintptr_t length);

  operator bool() const {
    return impl != nullptr;
  }
};

PeerMemory getPeerMemory(std::string id);

std::string id();

Vector<std::pair<size_t, CUmemGenericAllocationHandle>> cuMemHandles();

} // namespace allocator

} // namespace moodist
