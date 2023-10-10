
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>
#include <vector>

#include "fmt/printf.h"

#undef CHECK
#define CHECK(x)                                                                                                       \
  (bool(x) ? 0                                                                                                         \
           : (printf("[CHECK FAILED %s:%d] %s\n", __FILE__, __LINE__, #x), fflush(stderr), fflush(stdout),             \
              std::abort(), 0))

template<typename T, size_t maxSize>
struct SmallVector {
  static_assert(std::is_trivial_v<T>);
  std::array<T, maxSize> storage;
  size_t msize = 0;
  SmallVector() = default;
  SmallVector(std::initializer_list<T> list) {
    for (auto& v : list) {
      push_back(v);
    }
  }
  void push_back(T v) {
    storage[msize++] = v;
  }
  void pop_back() {
    --msize;
  }
  T* begin() {
    return storage.data();
  }
  T* end() {
    return storage.data() + msize;
  }
  const T* begin() const {
    return storage.data();
  }
  const T* end() const {
    return storage.data() + msize;
  }
  size_t size() const {
    return msize;
  }
  T& operator[](size_t index) {
    return storage[index];
  }
  const T& operator[](size_t index) const {
    return storage[index];
  }
  T* data() {
    return storage.data();
  }
  const T* data() const {
    return storage.data();
  }
  T& back() {
    return storage[msize - 1];
  }
  const T& back() const {
    return storage[msize - 1];
  }
  bool operator<(const SmallVector& v) {
    return std::basic_string_view<T>(data(), size()) < std::basic_string_view<T>(v.data(), v.size());
  }
};

int main() {

  size_t size = 2048;

  size_t ranksPerNode = 8;

  std::vector<std::vector<size_t>> nodeRanks;
  nodeRanks.resize(size / ranksPerNode);
  for (size_t i = 0; i != size; ++i) {
    nodeRanks[i / ranksPerNode].push_back(i);
  }

  size_t rank = 0;

  auto findLocalRank = [&](size_t rank) {
    for (auto& list : nodeRanks) {
      for (size_t n = 0; n != list.size(); ++n) {
        if (list[n] == rank) {
          return n;
        }
      }
    }
    CHECK(false);
    return (size_t)-1;
  };

  std::vector<std::vector<size_t>> paths;
  for (size_t src = 0; src != size; ++src) {
    size_t l = findLocalRank(src);
    for (auto& list : nodeRanks) {
      if (std::find(list.begin(), list.end(), src) != list.end()) {
        continue;
      }
      if (src == rank) {
        paths.push_back({src, list[l]});
      } else if (list[l] == rank) {
        for (size_t n = 0; n != list.size(); ++n) {
          if (n == l) {
            paths.push_back({src, list[n]});
          } else {
            paths.push_back({src, list[l], list[n]});
          }
        }
      } else if (std::find(list.begin(), list.end(), rank) != list.end()) {
        paths.push_back({src, list[l], rank});
      }
    }
  }

  auto sortFn = [&](auto& a, auto& b) {
    if (a.size() != b.size()) {
      return a.size() < b.size();
    }
    int am;
    int bm;
    am = (size + a[1] - a[0]) % size;
    bm = (size + b[1] - b[0]) % size;
    if (am != bm) {
      return am < bm;
    }
    if (a.size() == 3) {
      am = (ranksPerNode + a[2] - a[1]) % ranksPerNode;
      bm = (ranksPerNode + b[2] - b[1]) % ranksPerNode;
      if (am != bm) {
        return am > bm;
      }
    }
    return a < b;
  };
  std::sort(paths.begin(), paths.end(), sortFn);

  std::stable_partition(paths.begin(), paths.end(), [&](auto& a) { return a[0] == rank; });
  std::stable_partition(paths.begin(), paths.end(), [&](auto& a) {
    if (a.size() < 3) {
      return true;
    }
    return a[1] != rank;
  });

  for (auto& v : paths) {
    if (v.size() == 2) {
      CHECK(v[0] == rank || v[1] == rank);
    } else if (v.size() == 3) {
      CHECK(v[0] != rank);
      CHECK(v[1] == rank || v[2] == rank);
    } else {
      CHECK(false);
    }
  }

  fmt::printf("%d paths\n", paths.size());
  size_t n = 0;
  for (auto& v : paths) {
    if (std::find(v.begin(), v.end(), rank) != v.end()) {
      ++n;
    }
  }
  fmt::printf("%d paths involving me\n", n);

  // for (auto& path : paths) {
  //   fmt::printf("[%s]\n", fmt::to_string(fmt::join(path, ", ")));
  // }

  std::vector<size_t> sendRanks;
  std::vector<size_t> recvRanks;
  size_t proxyCount = 0;
  size_t proxyDestinationCount = 0;

  for (auto& path : paths) {
    CHECK(path.size() <= 3);
    CHECK(path.size() >= 2);
    size_t source = path[0];
    size_t proxy = path[1];
    size_t destination = path.back();
    if (path.size() == 3) {
      CHECK(source != proxy && source != destination && proxy != destination);
    } else if (path.size() == 2) {
      CHECK(source != proxy && source != destination && proxy == destination);
    }
    CHECK((proxy == destination) == (path.size() == 2));
    if (source == rank) {
      CHECK(path.size() == 2);
      CHECK(std::find(sendRanks.begin(), sendRanks.end(), destination) == sendRanks.end());
      sendRanks.push_back(destination);
    } else if (path.size() == 2 && destination == rank) {
      CHECK(std::find(recvRanks.begin(), recvRanks.end(), source) == recvRanks.end());
      recvRanks.push_back(source);
    } else if (proxy == rank) {
      CHECK(std::find(recvRanks.begin(), recvRanks.end(), source) != recvRanks.end());
      if (destination != rank) {
        CHECK(path.size() >= 3);
        ++proxyCount;
      }
    } else if (destination == rank) {
      CHECK(path.size() >= 3);
      ++proxyDestinationCount;
    }
  }

  fmt::printf(
      "num send: %d recv: %d proxy: %d proxy destination: %d\n", sendRanks.size(), recvRanks.size(), proxyCount,
      proxyDestinationCount);

  return 0;
}