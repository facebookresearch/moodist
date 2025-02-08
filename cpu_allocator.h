
#include <cstddef>
#include <cstdint>
#include <utility>

namespace moodist {

namespace cpu_allocator {
bool owns(uintptr_t);
std::pair<uintptr_t, size_t> regionAt(uintptr_t);
bool owns(const void*);
std::pair<uintptr_t, size_t> regionAt(const void*);

void* moo_alloc(size_t);
void moo_free(void*);

} // namespace cpu_allocator

} // namespace moodist
