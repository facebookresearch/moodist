
#include "setup_comms.h"
#include "buffer.h"
#include "connection.h"
#include "serialization.h"

namespace moodist {
struct SetupCommsImpl : SetupComms {

  static constexpr uint64_t signature = 0x116dc74103a4dc0f;

  TcpContext context;
  Vector<std::shared_ptr<Listener>> listeners;
  Vector<std::shared_ptr<Connection>> connections;
  Vector<std::shared_ptr<Connection>> floatingConnections;

  uint64_t myId;
  Vector<uint64_t> connectionIds;

  SpinMutex mutex;
  Vector<BufferHandle> incomingData;
  std::atomic_uint32_t incomingDataCount = 0;

  SetupCommsImpl(size_t rank, size_t size) : SetupComms(rank, size) {

    connections.resize(size);

    myId = random<uint64_t>();
    connectionIds.resize(size);

    if (rank == 0) {
      for (auto& addr : {"0.0.0.0:0", "[::]:0"}) {

        try {
          auto listener = context.listen(addr);

          listener->accept([this](Error* error, std::shared_ptr<Connection> connection) {
            if (error) {
              return;
            }
            // fmt::fprintf(
            //     stdout, "Got new connection, local address: %s remote address: %s\n", connection->localAddress(),
            //     connection->remoteAddress());
            // std::fflush(stdout);
            std::lock_guard l(mutex);
            floatingConnections.push_back(connection);
            connection->read([this, connection](Error* e, BufferHandle data) {
              if (!e) {
                onRead(std::move(data), connection);
              } else {
                connection->close();
              }
            });
          });

          listeners.push_back(listener);

        } catch (const std::exception& e) {
          fmt::fprintf(stderr, "Error while listening on '%s': %s\n", addr, e.what());
          fflush(stderr);
        }
      }
    }
  }

  std::vector<std::string> listenerAddresses() {
    std::vector<std::string> r;
    for (auto& v : listeners) {
      for (auto& addr : v->localAddresses()) {
        r.push_back(addr);
      }
    }
    return r;
  }

  void waitForConnections() {
    std::unique_lock l(mutex);
    if (rank == 0) {
      for (size_t i = 0; i != size; ++i) {
        if (i == rank) {
          continue;
        }
        while (!connections[i]) {
          l.unlock();
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          l.lock();
        }
      }
    } else {
      while (!connections[0]) {
        l.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        l.lock();
      }
    }
  }

  void connect(std::string address) {
    if (!context.isReachable("no-loopback", address)) {
      return;
    }
    // fmt::printf("Connecting to: %s\n", address);
    // std::fflush(stdout);
    auto connection = context.connect(address);
    auto buffer = serializeToBuffer(signature, myId, (uint32_t)rank, ~(uint32_t)0, std::string_view("connect"));
    connection->write(std::move(buffer), nullptr);

    floatingConnections.push_back(connection);
    connection->read([this, connection](Error* e, BufferHandle data) {
      if (!e) {
        onRead(std::move(data), connection);
      } else {
        // fmt::fprintf(stderr, "Socket read error: %s\n", e->what());
        // std::fflush(stderr);
        connection->close();
      }
    });
  }

  void onRead(BufferHandle data, const std::shared_ptr<Connection>& connection) {
    if (data->size() < 24) {
      return;
    }
    uint64_t sig;
    uint64_t rankId;
    uint32_t sourceRank;
    uint32_t destinationRank;
    auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank);
    if (sig != signature || sourceRank >= size) {
      return;
    }
    if (destinationRank != rank) {
      if (rank == 0 && destinationRank < size) {
        if (connections[destinationRank]) {
          connections[destinationRank]->write(std::move(data), nullptr);
        }
      } else {
        CHECK(destinationRank == ~(uint32_t)0);
      }
      std::unique_lock l(mutex);
      if (!connections[sourceRank]) {
        // fmt::printf("Got new connection to rank %d (id %#x) (destination %d)\n", sourceRank, rankId, destinationRank);
        // std::fflush(stdout);
        connections[sourceRank] = connection;
        connectionIds[sourceRank] = rankId;
      }
      if (rankId != connectionIds[sourceRank]) {
        throw std::runtime_error(fmt::sprintf(
            "Connection id mismatch for source rank %d. Got %#x, expected %#x", sourceRank, rankId,
            connectionIds[sourceRank]));
      }
      return;
    }
    std::unique_lock l(mutex);
    if (!connections[sourceRank]) {
      // fmt::printf("Got new connection to rank %d (id %#x)\n", sourceRank, rankId);
      // std::fflush(stdout);
      connections[sourceRank] = connection;
      connectionIds[sourceRank] = rankId;
    }
    if (rankId != connectionIds[sourceRank]) {
      throw std::runtime_error(fmt::sprintf(
          "Connection id mismatch for source rank %d. Got %#x, expected %#x", sourceRank, rankId,
          connectionIds[sourceRank]));
    }
    incomingData.push_back(std::move(data));
    l.unlock();
    ++incomingDataCount;
    futexWakeAll(&incomingDataCount);
  }

  template<typename F>
  bool getIncomingData(F&& f) {
    std::lock_guard l(mutex);
    for (auto i = incomingData.begin(); i != incomingData.end();) {
      if (f(*i)) {
        i = incomingData.erase(i);
        return true;
      } else {
        ++i;
      }
    }
    return false;
  }
  template<typename F>
  void waitForIncomingData(F&& f) {
    size_t prev = 0;
    while (true) {
      size_t cur = incomingDataCount;
      while (cur == prev) {
        futexWait(&incomingDataCount, prev, std::chrono::seconds(1));
        cur = incomingDataCount;
      }
      prev = cur;
      if (getIncomingData(f)) {
        return;
      }
    }
  }

  std::vector<BufferHandle> allgatherBuffers;
  std::vector<std::string_view> allgatherResult;
  std::vector<std::string_view>& allgather(BufferHandle data) {
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      BufferHandle buffer2 = makeBuffer(data->size());
      std::memcpy(buffer2->data(), data->data(), data->size());
      sendBufferTo(i, std::move(buffer2));
    }
    uint64_t sig;
    uint64_t rankId;
    uint32_t sourceRank;
    uint32_t destinationRank;
    size_t remaining = size - 1;
    allgatherResult.resize(size);
    allgatherBuffers.resize(size);
    for (auto& v : allgatherBuffers) {
      v = nullptr;
    }
    allgatherResult[rank] = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank);
    allgatherBuffers[rank] = std::move(data);
    while (remaining) {
      waitForIncomingData([&](BufferHandle& data) {
        auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank);
        if (allgatherBuffers[sourceRank]) {
          return false;
        }
        allgatherBuffers[sourceRank] = std::move(data);
        allgatherResult[sourceRank] = view;
        --remaining;
        return true;
      });
    }

    return allgatherResult;
  }

  void sendBufferTo(size_t destinationRank, BufferHandle data) {
    CHECK(connections.size() > destinationRank);
    TORCH_CHECK(data->size() > 16);
    auto* ptr = data->data();
    std::memcpy(ptr, &signature, sizeof(uint64_t));
    std::memcpy(ptr + 8, &myId, sizeof(uint64_t));
    uint32_t sourceRank = rank;
    std::memcpy(ptr + 16, &sourceRank, sizeof(uint32_t));
    std::memcpy(ptr + 20, &destinationRank, sizeof(uint32_t));
    if (connections[destinationRank]) {
      connections[destinationRank]->write(std::move(data), nullptr);
    } else {
      TORCH_CHECK(connections[0] != nullptr);
      connections[0]->write(std::move(data), nullptr);
    }
  }

  BufferHandle recvBuffer;
  std::string_view recvBufferFrom(size_t rank) {
    std::string_view r;
    waitForIncomingData([&](BufferHandle& data) {
      uint64_t sig;
      uint64_t rankId;
      uint32_t sourceRank;
      uint32_t destinationRank;
      auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank);
      if (sourceRank == rank) {
        recvBuffer = std::move(data);
        r = view;
        return true;
      }
      return false;
    });
    return r;
  }
};

void SetupComms::connect(std::string address) {
  ((SetupCommsImpl*)this)->connect(address);
}
std::vector<std::string> SetupComms::listenerAddresses() {
  return ((SetupCommsImpl*)this)->listenerAddresses();
}
void SetupComms::waitForConnections() {
  return ((SetupCommsImpl*)this)->waitForConnections();
}

std::vector<std::string_view>& SetupComms::allgather(BufferHandle buffer) {
  return ((SetupCommsImpl*)this)->allgather(std::move(buffer));
}
void SetupComms::sendBufferTo(size_t rank, BufferHandle buffer) {
  return ((SetupCommsImpl*)this)->sendBufferTo(rank, std::move(buffer));
}
std::string_view SetupComms::recvBufferFrom(size_t rank) {
  return ((SetupCommsImpl*)this)->recvBufferFrom(rank);
}

std::unique_ptr<SetupComms> createSetupComms(size_t rank, size_t size) {
  return std::make_unique<SetupCommsImpl>(rank, size);
}

} // namespace moodist
