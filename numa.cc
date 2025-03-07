
#include "common.h"

#include <sys/mman.h>
#include <sys/syscall.h>

namespace moodist {

void* numa_alloc_onnode(size_t bytes, int node) {
  void* r = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (node >= 0) {
    constexpr int mode_bind = 2;
    unsigned long mask = (unsigned long)1 << node;
    if (syscall(SYS_mbind, (uintptr_t)r, bytes, mode_bind, (uintptr_t)&mask, 64, 0)) {
      log.debug("mbind error: %d\n", errno);
    }
  }
  return r;
}
void* numa_alloc_local(size_t bytes) {
  return mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
}
void numa_free(void* ptr, size_t bytes) {
  munmap(ptr, bytes);
}

int numa_run_on_node(int node) {
  std::string fn = fmt::sprintf("/sys/devices/system/node/node%d/cpulist", node);
  FILE* f = fopen(fn.c_str(), "rb");
  if (!f) {
    errno = ENOENT;
    return -1;
  }
  std::string s;
  while (true) {
    size_t o = s.size();
    s.resize(o + 0x10);
    size_t n = fread(s.data() + o, 1, 0x10, f);
    if (n < 0) {
      errno = EIO;
      return -1;
    }
    if (n < 0x10) {
      s.resize(o + n);
      break;
    }
  }
  fclose(f);

  static_assert(sizeof(unsigned long) == 8);

  std::vector<unsigned long> cpumask;
  cpumask.resize(0x100);
  for (auto& v : cpumask) {
    v = 0;
  }

  const char* p = s.c_str();
  const char* e = p + s.size();
  while (p != e) {
    const char* n = p;
    auto num = [&]() {
      int r = 0;
      while (p != e && *p >= '0' && *p <= '9') {
        r = r * 10 + *p - '0';
        ++p;
      }
      return r;
    };
    int a = num();
    int b = a;
    if (p != e && *p == '-') {
      ++p;
      b = num();
    }
    while (a <= b) {
      if (a / 64 < cpumask.size()) {
        cpumask[a / 64] |= 1ul << (a % 64);
      }
      ++a;
    }
    if (p != e && *p == ',') {
      ++p;
    } else {
      break;
    }
  }

  return syscall(SYS_sched_setaffinity, 0, 8 * cpumask.size(), cpumask.data());
}

} // namespace moodist
