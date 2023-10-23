
#include "common.h"
#include "processgroup.h"
#include "serialization.h"
#include "socket.h"

namespace moodist {

struct Foo {
  BufferHandle buf;
  Foo(int val) : buf(serializeToBuffer(val)) {}
  bool operator==(const Foo& n) {
    return buf->size() == n.buf->size() && !memcmp(buf->data(), n.buf->data(), buf->size());
  }
};

#define tx                                                                                                             \
  {                                                                                                                    \
    CHECK(a.size() == b.size());                                                                                       \
    for (size_t i = 0; i != a.size(); ++i)                                                                             \
      CHECK(a[i] == b[i]);                                                                                             \
  }

void testv() {
  std::vector<Foo> a;
  Vector<Foo> b;
  for (int i = 0; i != 100000; ++i) {
    a.push_back(i);
    b.push_back(i);
  }
  tx;
  for (int i = 0; i != 100000; ++i) {
    size_t index = rand() % a.size();
    if (rand() % 2 == 0) {
      index = 0;
    }
    auto ai = a.erase(a.begin() + index);
    auto bi = b.erase(b.begin() + index);
    tx;
    CHECK(ai - a.begin() == bi - b.begin());
    a.push_back(i);
    b.push_back(i);
  }

  fmt::printf("all ok\n");
  std::quick_exit(0);
}

void test(int rank, int size, std::string rank0Address) {
  ProcessGroup pg(rank, size);
  std::mutex mutex;

  std::string addr = pg.getAddress();

  std::vector<Socket> conns;

  Socket sock;
  CachedReader reader(&sock);
  std::atomic_int state = 0;
  size_t len;

  auto buffer = serializeToBuffer(addr);

  if (rank == 0) {
    fmt::printf("Address: %s\n", pg.getAddress());

    sock = Socket::Tcp();

    sock.listen("0.0.0.0:51022");
    sock.accept([&addr, &conns, &buffer, &mutex](Error* e, Socket conn) {
      if (!e) {
        std::lock_guard l(mutex);
        iovec vec;
        vec.iov_base = buffer->data();
        vec.iov_len = buffer->size();
        conn.writev(&vec, 1, [](Error*) {});
        conns.push_back(std::move(conn));
      }
    });
  } else {
    std::atomic_bool error = true;
    while (error) {
      error = false;
      sock = Socket::Tcp();
      sock.connect(rank0Address + ":51022", [&error](Error* e) {
        if (e) {
          error = true;
        }
      });
      sock.setOnRead([&](Error* e) {
        if (e) {
          error = true;
          return;
        }
        if (state == 0) {
          if (!reader.readCopy(&len, sizeof(len))) {
            return;
          }
          state = 1;
        }
        if (state == 1) {
          rank0Address.resize(len);
          if (!reader.readCopy(rank0Address.data(), len)) {
            return;
          }
          state = 2;
        }
      });

      fmt::printf("waiting for address\n");
      while (state != 2) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (error) {
          fmt::printf("connect error, retrying!\n");
          break;
        }
      }
      if (!error) {
        fmt::printf("got address! %s\n", rank0Address);
      }
    }
  }

  std::fflush(stdout);
  pg.init(rank0Address);

  fmt::printf("init done!\n");
}

} // namespace moodist

int main(int argc, const char** argv) {

  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  // moodist::testv();

  CHECK(argc >= 3);

  moodist::test(std::atoi(argv[1]), std::atoi(argv[2]), argc >= 4 ? argv[3] : "");

  return 0;
}