#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

int main() {

  std::atomic_size_t nbytes = 0;

  std::vector<std::thread> threads;

  auto entry = [&]() {
    std::vector<uint8_t> dst;
    std::vector<uint8_t> src;
    dst.resize(1024 * 1024 * 1024);
    src.resize(1024 * 1024 * 1024);

    printf("thread entry, %ld bytes\n", dst.size());

    while (true) {
      // std::memcpy(dst.data(), src.data(), dst.size());
      std::memset(dst.data(), 0, dst.size());
      nbytes += dst.size();
    }
  };

  for (int i = 0; i != 32; ++i) {
    threads.emplace_back(entry);
  }

  for (int i = 0; i != 8; ++i) {
    nbytes = 0;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    size_t n = nbytes;
    float t = 10.0f;
    printf("%ld bytes in %gs, %gG/s\n", n, t, n / t / 1024 / 1024 / 1024);
  }

  std::quick_exit(0);

  return 0;
}