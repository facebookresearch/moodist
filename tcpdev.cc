
#include "tcpdev.h"
#include "connection.h"
#include "group.h"
#include "serialization.h"
#include "setup_comms.h"
#include <type_traits>

namespace moodist {

static TcpContext context;

static constexpr uint32_t unconnected = -1;

static constexpr uint64_t signatureConnect = 0x5a59123966da2598;

struct TcpDevImpl {
  Group* group = nullptr;

  std::vector<std::shared_ptr<Listener>> listeners;

  SpinMutex mutex;
  std::atomic_bool dead = false;

  struct QueuedSend {
    BufferHandle buffer;
    void* payload = nullptr;
    size_t payloadBytes = 0;
    Function<void(Error*)> callback;
  };

  HashMap<uint32_t, Vector<QueuedSend>> queuedSends;

  struct ConnectionInfo {
    uint32_t rank = unconnected;
    std::shared_ptr<Connection> connection;

    void close() {
      if (connection) {
        connection->close();
      }
    }
  };

  std::vector<std::shared_ptr<ConnectionInfo>> connections;
  std::vector<std::shared_ptr<ConnectionInfo>> floatingConnections;

  void* onReadHandle = nullptr;
  std::atomic_size_t refcount = 0;

  const uint32_t rank = group->rank;
  const uint32_t size = group->size;

  TcpDevImpl(Group* group, void* onReadHandle) : group(group), onReadHandle(onReadHandle) {

    connections.resize(size);

    std::unique_lock l(mutex);

    for (auto& addr : {"0.0.0.0:0", "[::]:0"}) {

      try {
        auto listener = context.listen(addr);

        listener->accept([this](Error* error, std::shared_ptr<Connection> connection) {
          if (error) {
            return;
          }
          std::lock_guard l(mutex);
          if (dead) {
            return;
          }
          addConnection(std::move(connection));
        });

        listeners.push_back(listener);

        // log.info("tcp listening on %s\n", fmt::to_string(fmt::join(listener->localAddresses(), ", ")));

      } catch (const std::exception& e) {
        log.error("Error while listening on '%s': %s\n", addr, e.what());
      }
    }

    std::vector<std::string> addresses;
    for (auto& v : listeners) {
      for (auto& a : v->localAddresses()) {
        addresses.push_back(a);
      }
    }
    auto allRanksAddresses = group->setupComms->allgather(addresses);
    for (size_t i : range(size)) {
      if (allRanksAddresses[i].empty()) {
        throw std::runtime_error(fmt::sprintf("Moodist: rank %d failed to listen on any tcp addresses", i));
      }
      if (i > rank) {
        for (auto& a : allRanksAddresses[i]) {
          // log.info("connecting to rank %d at %s\n", i, a);
          auto connection = context.connect(a);
          connection->write(serializeToBuffer(signatureConnect, group->name, rank), nullptr);

          addConnection(std::move(connection));
        }
      }
    }
  }

  template<typename Container>
  void closeAll(Container& container) {
    while (true) {
      std::unique_lock l(mutex);
      auto tmp = std::move(container);
      container.clear();
      l.unlock();
      if (tmp.empty()) {
        break;
      }
      for (auto& v : tmp) {
        if (v) {
          v->close();
        }
      }
    }
  }

  ~TcpDevImpl() {
    {
      std::lock_guard l(mutex);
      dead = true;
    }
    closeAll(listeners);
    closeAll(floatingConnections);
    closeAll(connections);
    while (refcount) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  void close() {
    for (auto& v : listeners) {
      v->close();
    }
    for (auto& v : floatingConnections) {
      v->close();
    }
    for (auto& v : connections) {
      if (v) {
        v->close();
      }
    }
  }

  void addConnection(std::shared_ptr<Connection> connection) {
    CHECK(!dead);
    auto ci = std::make_shared<ConnectionInfo>();
    ci->connection = connection;
    floatingConnections.push_back(ci);
    // log.info("add new connection - %d floating connections\n", floatingConnections.size());
    connection->read([this, ci](Error* e, BufferHandle data) {
      if (!e) {
        onRead(std::move(data), ci);
      } else {
        // if (ci->rank != unconnected) {
        //   log.error("Moodist tcp connection error (rank %d): %s\n", ci->rank, e->what());
        // }
        ci->connection->close();

        if (ci->rank != unconnected) {
          tcpOnErrorCallback(onReadHandle, ci->rank, e);
        }
      }
    });
  }

  void onRead(BufferHandle data, const std::shared_ptr<ConnectionInfo>& ci) {
    std::unique_lock l(mutex);
    if (dead) {
      return;
    }
    try {
      if (ci->rank == unconnected) {
        size_t n = data->size();
        const char* buf = (const char*)data->data();
        if (n < 8) {
          ci->connection->close();
          return;
        }
        uint64_t signature;
        std::memcpy(&signature, buf, 8);
        if (signature != signatureConnect) {
          ci->connection->close();
          return;
        }
        std::string_view view(&buf[8], &buf[n]);
        std::string groupName;
        uint32_t nRank;
        deserializeBuffer(view, groupName, nRank);
        if (groupName != group->name) {
          ci->connection->close();
          return;
        }
        CHECK(nRank < size);
        if (connections[nRank]) {
          ci->connection->close();
          return;
        }
        CHECK(ci->rank == unconnected);
        ci->rank = nRank;
        connections[nRank] = ci;
        // log.info("Got a connection to rank %d\n", nRank);
        if (nRank < rank) {
          ci->connection->write(serializeToBuffer(signatureConnect, group->name, rank), nullptr);
        }
        auto it = queuedSends.find(nRank);
        if (it != queuedSends.end()) {
          for (auto& v : it->second) {
            sendImpl(nRank, std::move(v.buffer), v.payload, v.payloadBytes, std::move(v.callback));
          }
          queuedSends.erase(it);
        }
      } else {
        l.unlock();
        onMessage(ci->rank, std::move(data));
      }

    } catch (const SerializationError& e) {
      log.error("tcp recv error: %s\n", e.what());
    }
  }

  struct ReadHelper {
    size_t i;
    void* data;
    size_t bytes;
    void clear() {
      data = nullptr;
    }
  };
  using ReadHelperPtr = FLPtr<ReadHelper>;

  void onMessage(size_t i, BufferHandle buffer) {
    ReadHelper rh;
    rh.i = i;
    rh.data = nullptr;

    tcpOnReadCallback(i, std::move(buffer), onReadHandle, &rh);
  }

  void onMessage(size_t i, BufferHandle buffer, void* payload, size_t payloadBytes) {
    ReadHelper rh;
    rh.i = i;
    rh.data = payload;
    rh.bytes = payloadBytes;

    tcpOnReadCallback(i, std::move(buffer), onReadHandle, &rh);
  }

  void sendImpl(size_t i, BufferHandle buffer, void* payload, size_t payloadBytes, Function<void(Error*)> callback) {
    CHECK(connections[i]->connection);
    if (payload) {
      std::array<iovec, 1> iovec;
      iovec[0].iov_base = payload;
      iovec[0].iov_len = payloadBytes;
      connections[i]->connection->write(std::move(buffer), iovec, std::move(callback));
    } else {
      connections[i]->connection->write(std::move(buffer), std::move(callback));
    }
  }

  void send(size_t i, BufferHandle buffer, void* payload, size_t payloadBytes, Function<void(Error*)> callback) {
    std::lock_guard l(mutex);
    if (dead) {
      return;
    }
    CHECK(i < connections.size());
    if (i == rank) {
      ++refcount;
      scheduler.run(
          [this, i, buffer = std::move(buffer), payload, payloadBytes, callback = std::move(callback)] mutable {
            onMessage(i, std::move(buffer), payload, payloadBytes);
            if (callback) {
              std::move(callback)(nullptr);
            }
            --refcount;
          });
      return;
    }
    if (!connections[i] || connections[i]->rank == unconnected) {
      // log.info("rank is not yet connected, queueing send\n");
      QueuedSend q;
      q.buffer = std::move(buffer);
      q.payload = payload;
      q.payloadBytes = payloadBytes;
      q.callback = std::move(callback);
      queuedSends[i].push_back(std::move(q));
    } else {
      sendImpl(i, std::move(buffer), payload, payloadBytes, std::move(callback));
    }
  }

  void read(void* handle, void* address, size_t bytes, Function<void()> callback) {
    ReadHelper* ptr = (ReadHelper*)handle;
    if (ptr->data) {
      CHECK(bytes == ptr->bytes);
      std::memcpy(address, ptr->data, bytes);
      std::move(callback)();
    } else {
      CHECK(connections[ptr->i] && connections[ptr->i]->rank == ptr->i);

      connections[ptr->i]->connection->inread_iovec(address, bytes, std::move(callback));
    }
  }
};

static_assert(std::is_standard_layout_v<TcpDevImpl>);

void TcpDev::destroy() {
  delete (TcpDevImpl*)impl;
}

void TcpDev::send(
    size_t i, BufferHandle&& buffer, void* payload, size_t payloadBytes, Function<void(Error*)> callback) {
  return ((TcpDevImpl*)impl)->send(i, std::move(buffer), payload, payloadBytes, std::move(callback));
}

void TcpDev::send(size_t i, BufferHandle buffer, Function<void(Error*)> callback) {
  return ((TcpDevImpl*)impl)->send(i, std::move(buffer), nullptr, 0, std::move(callback));
}

void TcpDev::read(void* handle, void* address, size_t bytes, Function<void()> callback) {
  return ((TcpDevImpl*)impl)->read(handle, address, bytes, std::move(callback));
}

void TcpDev::close() {
  return ((TcpDevImpl*)impl)->close();
}

UniqueImpl<TcpDev> makeTcpDev(Group* group, void* onReadHandle) {
  UniqueImpl<TcpDev> r;
  r.u.impl = new TcpDevImpl(group, onReadHandle);
  return r;
}

} // namespace moodist
