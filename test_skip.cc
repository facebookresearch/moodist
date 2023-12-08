
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

  size_t size = 8;

  size_t ranksPerNode = 1;

  std::vector<std::vector<size_t>> nodeRanks;
  nodeRanks.resize(size / ranksPerNode);
  for (size_t i = 0; i != size; ++i) {
    nodeRanks[i / ranksPerNode].push_back(i);
  }

  std::vector<size_t> rankNodes;
  rankNodes.resize(size);
  for (size_t i = 0; i != nodeRanks.size(); ++i) {
    for (size_t r : nodeRanks[i]) {
      rankNodes[r] = i;
    }
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
    size_t ol = findLocalRank(src);
    for (auto& list : nodeRanks) {
      if (std::find(list.begin(), list.end(), src) != list.end()) {
        continue;
      }
      size_t ll = ol % list.size();
      if (src == rank) {
        paths.push_back({src, list[ll]});
      } else if (list[ll] == rank) {
        for (size_t n = 0; n != list.size(); ++n) {
          if (n == ll) {
            paths.push_back({src, list[n]});
          } else {
            paths.push_back({src, list[ll], list[n]});
          }
        }
      } else if (std::find(list.begin(), list.end(), rank) != list.end()) {
        paths.push_back({src, list[ll], rank});
      }
    }
  }

  // std::vector<size_t> tmp;
  // std::vector<std::vector<size_t>> paths;
  // for (size_t src = 0; src != size; ++src) {
  //   size_t ol = findLocalRank(src);
  //   for (auto& list : nodeRanks) {
  //     if (std::find(list.begin(), list.end(), src) != list.end()) {
  //       continue;
  //     }
  //     size_t ll = ol % list.size();
  //     if (src == rank) {
  //       paths.push_back({src, list[ll]});
  //     } else if (list[ll] == rank) {
  //       for (size_t n = 0; n != list.size(); ++n) {
  //         if (n == ll) {
  //           paths.push_back({src, list[n]});
  //         } else {
  //           paths.push_back({src, list[ll], list[n]});
  //         }
  //       }
  //     } else if (std::find(list.begin(), list.end(), rank) != list.end()) {
  //       paths.push_back({src, list[ll], rank});
  //     } else {
  //       size_t source = src;
  //       size_t destination = list[ll];
  //       size_t sourceNode = rankNodes[source];
  //       size_t destinationNode = rankNodes[destination];
  //       size_t d = (destinationNode - sourceNode + size) % size;
  //       if (d != 1 && d != size - 1) {
  //         size_t add = d <= (size - d) ? 1 : size - 1;
  //         size_t l = findLocalRank(source);
  //         tmp.clear();
  //         for (size_t i = sourceNode; i != destinationNode; i = (i + add) % size) {
  //           tmp.push_back(nodeRanks[i][l % nodeRanks[i].size()]);
  //         }
  //         tmp.push_back(destination);
  //         CHECK(tmp.size() >= 3);
  //         tmp.erase(tmp.begin(), tmp.begin() + tmp.size() - 3);
  //         tmp[0] = source;

  //         if (tmp[1] == rank) {
  //           paths.push_back(tmp);
  //         }
  //       }
  //     }
  //   }
  // }

  // for (auto i = paths.begin(); i != paths.end();) {
  //   auto& v = *i;
  //   if (v.size() != 2) {
  //     ++i;
  //     continue;
  //   }
  //   size_t source = v[0];
  //   size_t destination = v[1];
  //   size_t sourceNode = rankNodes[source];
  //   size_t destinationNode = rankNodes[destination];
  //   size_t d = (destinationNode - sourceNode + size) % size;
  //   if (d != 1 && d != size - 1) {
  //     size_t add = d <= (size - d) ? 1 : size - 1;
  //     size_t l = findLocalRank(source);
  //     v.clear();
  //     for (size_t i = sourceNode; i != destinationNode; i = (i + add) % size) {
  //       v.push_back(nodeRanks[i][l % nodeRanks[i].size()]);
  //     }
  //     v.push_back(destination);
  //     CHECK(v.size() >= 3);
  //     while (v.size() > 3) {
  //       v.erase(v.begin());
  //     }
  //     v[0] = source;

  //     if (v[1] != rank && v[2] != rank) {
  //       i = paths.erase(i);
  //     } else {
  //       ++i;
  //     }
  //   } else {
  //     ++i;
  //   }
  // }

  // std::vector<std::vector<size_t>> paths;

  // size_t myLocalRank = findLocalRank(rank);

  // size_t skip = ranksPerNode * 2;

  // for (size_t src = 0; src != size; ++src) {
  //   for (size_t offset = 0; offset < size; offset += skip) {
  //     size_t source = (src + size - offset) % size;
  //     for (size_t s = 0; s < skip; s += ranksPerNode) {
  //       size_t dst = (src + ranksPerNode + s) % size;
  //       if (source == dst) {
  //         continue;
  //       }
  //       if (src == rank || dst == rank) {
  //         paths.push_back({source, src, dst});
  //       }
  //     }
  //   }
  // }

  auto abs = [&](size_t v, size_t n) {
    v = (v + n) % n;
    return v <= (n - v) ? v : n - v;
  };

  auto sortFn = [&](auto& a, auto& b) {
    if (a.size() != b.size()) {
      return a.size() < b.size();
    }
    int am;
    int bm;
    am = abs(a[1] - a[0], size);
    bm = abs(b[1] - b[0], size);
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
    if ((a[1] - a[0] + size) % size != (b[1] - b[0] + size) % size) {
      return (a[1] - a[0] + size) % size > (b[1] - b[0] + size) % size;
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

  // for (auto& v : paths) {
  //   if (v.size() == 2) {
  //     CHECK(v[0] == rank || v[1] == rank);
  //   } else if (v.size() == 3) {
  //     CHECK(v[0] != rank);
  //     CHECK(v[1] == rank || v[2] == rank);
  //   } else {
  //     CHECK(false);
  //   }
  // }

  fmt::printf("%d paths\n", paths.size());
  size_t n = 0;
  for (auto& v : paths) {
    if (std::find(v.begin(), v.end(), rank) != v.end()) {
      ++n;
    }
  }
  fmt::printf("%d paths involving me\n", n);

  for (auto& path : paths) {
    fmt::printf("[%s]\n", fmt::to_string(fmt::join(path, ", ")));
  }

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

  std::vector<std::vector<size_t>> sends;
  std::vector<std::vector<size_t>> recvs;
  std::vector<size_t> neighbors;
  for (auto& path : paths) {
    if (path.size() != 2) {
      continue;
    }
    if (path[0] == rank) {
      size_t sourceNode = rankNodes[path[0]];
      size_t destinationNode = rankNodes[path[1]];
      if (abs(destinationNode - sourceNode, nodeRanks.size()) == 1) {
        sends.push_back({rank, path[1]});
        neighbors.push_back(path[1]);
        fmt::printf("%d is a neighbor\n", path[1]);
      }
    } else if (path[1] == rank) {
      size_t source = path[0];
      size_t bestDistance = size;
      size_t proxy = -1;
      for (auto& v : neighbors) {
        size_t d = abs(v - source, size);
        if (d < bestDistance) {
          bestDistance = d;
          proxy = v;
        }
      }
      recvs.push_back({source, proxy});
    }
  }
  // for (auto& v : recvs) {
  //   size_t source = v[0];
  //   size_t bestDistance = size;
  //   size_t proxy = -1;
  //   for (auto& v : neighbors) {
  //     size_t d = abs(v - source, size);
  //     if (d < bestDistance) {
  //       bestDistance = d;
  //       proxy = v;
  //     }
  //   }
  //   if (bestDistance < abs(source - proxy, size)) {
  //     sends.push_back({source, proxy});
  //   }
  // }
  // for (auto& path : paths) {
  //   if (path.size() != 2) {
  //     continue;
  //   }
  //   if (path[0] != rank) {
  //     continue;
  //   }
  //   size_t source = path[0];
  //   for (auto& v : neighbors) {
  //       size_t d = abs(v - source, size);
  //       if (d < bestDistance) {
  //         bestDistance = d;
  //         proxy = v;
  //       }
  //     }
  //   size_t sourceNode = rankNodes[path[0]];
  //   size_t destinationNode = rankNodes[path[1]];
  //   if (abs(destinationNode - sourceNode, nodeRanks.size()) <= 1) {
  //     continue;
  //   }
  // }

  fmt::printf("sends:\n");
  for (auto& v : sends) {
    fmt::printf("[%d, %d]\n", v[0], v[1]);
  }
  fmt::printf("recvs:\n");
  for (auto& v : recvs) {
    fmt::printf("[%d, %d]\n", v[0], v[1]);
  }

  fmt::printf(
      "num send: %d recv: %d proxy: %d proxy destination: %d\n", sendRanks.size(), recvRanks.size(), proxyCount,
      proxyDestinationCount);

  return 0;
}