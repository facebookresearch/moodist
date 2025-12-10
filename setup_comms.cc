// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "setup_comms.h"
#include "buffer.h"
#include "connection.h"
#include "serialization.h"
#include "synchronization.h"

namespace moodist {
struct SetupCommsImpl : SetupComms {

  static constexpr uint64_t signature = 0x116dc74103a4dc0f;
  static constexpr uint64_t signatureAllgather = 0x116dc74102a4dc0f;

  std::atomic_uint32_t connectionReady = 0;

  uint64_t myId;
  Vector<uint64_t> connectionIds;

  SpinMutex mutex;
  Vector<BufferHandle> incomingData;
  std::atomic_uint32_t incomingDataCount = 0;

  TcpContext context;
  Vector<std::shared_ptr<Listener>> listeners;
  std::shared_ptr<Connection> prev;
  std::shared_ptr<Connection> next;
  Vector<std::shared_ptr<Connection>> floatingConnections;

  std::atomic_bool dying = false;
  std::atomic_bool dead = false;

  SetupCommsImpl(size_t rank, size_t size) : SetupComms(rank, size) {

    myId = random<uint64_t>();
    connectionIds.resize(size);

    std::lock_guard l(mutex);

    for (auto& addr : {"0.0.0.0:0", "[::]:0"}) {

      try {
        auto listener = context.listen(addr);

        listener->accept([this](Error* error, std::shared_ptr<Connection> connection) {
          if (error) {
            return;
          }
          std::lock_guard l(mutex);
          if (dying) {
            return;
          }
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
        log.error("Error while listening on '%s': %s\n", addr, e.what());
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
        v->close();
      }
    }
  }

  ~SetupCommsImpl() {
    dying = true;
    closeAll(listeners);
    closeAll(floatingConnections);
    std::vector<std::shared_ptr<Connection>> tmp;
    {
      std::lock_guard l(mutex);
      if (prev) {
        tmp.push_back(prev);
      }
      if (next) {
        tmp.push_back(next);
      }
    }
    closeAll(tmp);
    dead = true;
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
    if (size == 1) {
      return;
    }
    std::unique_lock l(mutex);
    while (!prev || !next) {
      l.unlock();
      futexWait(&connectionReady, 0, std::chrono::seconds(2));
      connectionReady = 0;
      l.lock();
    }
  }

  void connect(std::string address) {
    if (!context.isReachable("no-loopback", address)) {
      return;
    }
    std::lock_guard l(mutex);
    auto connection = context.connect(address);
    auto buffer =
        serializeToBuffer(signature, myId, (uint32_t)rank, ~(uint32_t)0, (uint32_t)0, std::string_view("connect"), key);
    connection->write(std::move(buffer), nullptr);

    floatingConnections.push_back(connection);
    connection->read([this, connection](Error* e, BufferHandle data) {
      if (!e) {
        onRead(std::move(data), connection);
      } else {
        connection->close();
      }
    });
  }

  void onRead(BufferHandle data, const std::shared_ptr<Connection>& connection) {
    std::unique_lock l(mutex);
    CHECK(!dead);
    if (data->size() < 28 || dying) {
      return;
    }
    uint64_t sig;
    uint64_t rankId;
    uint32_t sourceRank;
    uint32_t destinationRank;
    uint32_t ttl;
    auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank, ttl);
    if ((sig != signature && sig != signatureAllgather) || sourceRank >= size || sourceRank == rank) {
      return;
    }
    CHECK(connectionIds.size() > sourceRank);
    if (destinationRank != rank) {
      if (destinationRank < size) {
        CHECK(ttl);
        if (next && prev && ttl) {
          auto* ptr = data->data();
          --ttl;
          std::memcpy(ptr + 24, &ttl, sizeof(uint32_t));
          ((destinationRank + size - rank) % size <= size / 2 ? next : prev)->write(std::move(data), nullptr);
        }
      } else {
        CHECK(destinationRank == ~(uint32_t)0);

        try {
          std::string_view connectstr;
          std::string_view keystr;
          deserializeBuffer(view, connectstr, keystr);
          if (connectstr != "connect" || keystr != key) {
            return;
          }
        } catch (const std::exception&) {
          return;
        }

        bool isNew = false;
        bool isPrev = !prev && sourceRank == (rank + size - 1) % size;
        bool isNext = !next && sourceRank == (rank + 1) % size;
        if (isPrev || isNext) {
          log.debug("%d: Got %s connection to rank %d (id %#x) (destination %d)\n", rank, isNext ? "next" : "prev",
              sourceRank, rankId, destinationRank);
          (isNext ? next : prev) = connection;
          connectionIds[sourceRank] = rankId;
          isNew = true;
        }
        if (isNew) {
          auto buffer = serializeToBuffer(
              signature, myId, (uint32_t)rank, ~(uint32_t)0, (uint32_t)0, std::string_view("connect"), key);
          connection->write(std::move(buffer), nullptr);

          connectionReady = 1;
          futexWakeAll(&connectionReady);
        }
      }
      if ((sourceRank == (rank + size - 1) % size || sourceRank == (rank + 1) % size) &&
          rankId != connectionIds[sourceRank]) {
        throw std::runtime_error(fmt::sprintf("Connection id mismatch for source rank %d. Got %#x, expected %#x",
            sourceRank, rankId, connectionIds[sourceRank]));
      }
      return;
    }
    // if (rankId != connectionIds[sourceRank]) {
    //   throw std::runtime_error(fmt::sprintf(
    //       "Connection id mismatch for source rank %d. Got %#x, expected %#x", sourceRank, rankId,
    //       connectionIds[sourceRank]));
    // }
    // CHECK(rankId != myId);
    // fmt::printf("%d: recv %d bytes (ttl %d)\n", rank, data->size(), ttl);
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
    if (size > 1) {
      BufferHandle buffer2 = makeBuffer(data->size());
      std::memcpy(buffer2->data(), data->data(), data->size());
      sendBufferTo((rank + 1) % size, std::move(buffer2), true);
    }
    uint64_t sig;
    uint64_t rankId;
    uint32_t sourceRank;
    uint32_t destinationRank;
    uint32_t ttl;
    size_t remaining = size - 1;
    allgatherResult.resize(size);
    allgatherBuffers.resize(size);
    for (auto& v : allgatherBuffers) {
      v = nullptr;
    }
    allgatherResult[rank] = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank, ttl);
    allgatherBuffers[rank] = std::move(data);
    while (remaining) {
      waitForIncomingData([&](BufferHandle& data) {
        auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank, ttl);
        if (allgatherBuffers[sourceRank] || sig != signatureAllgather) {
          return false;
        }
        BufferHandle buffer2 = makeBuffer(data->size());
        std::memcpy(buffer2->data(), data->data(), data->size());
        allgatherBuffers[sourceRank] = std::move(data);
        allgatherResult[sourceRank] = view;
        --remaining;
        size_t nextRank = (rank + 1) % size;
        if (nextRank != sourceRank) {
          CHECK(ttl > 0);
          auto* ptr = buffer2->data();
          std::memcpy(ptr + 20, &nextRank, sizeof(uint32_t));
          --ttl;
          std::memcpy(ptr + 24, &ttl, sizeof(uint32_t));
          next->write(std::move(buffer2), nullptr);
        }
        return true;
      });
    }

    return allgatherResult;
  }

  void sendBufferTo(size_t destinationRank, BufferHandle data, bool isAllgather = false) {
    CHECK(data->size() >= 28);
    auto* ptr = data->data();
    std::memcpy(ptr, isAllgather ? &signatureAllgather : &signature, sizeof(uint64_t));
    std::memcpy(ptr + 8, &myId, sizeof(uint64_t));
    uint32_t sourceRank = rank;
    std::memcpy(ptr + 16, &sourceRank, sizeof(uint32_t));
    std::memcpy(ptr + 20, &destinationRank, sizeof(uint32_t));
    uint32_t ttl = size;
    std::memcpy(ptr + 24, &ttl, sizeof(uint32_t));
    ((destinationRank + size - rank) % size <= size / 2 ? next : prev)->write(std::move(data), nullptr);
  }

  BufferHandle recvBuffer;
  std::string_view recvBufferFrom(size_t rank) {
    std::string_view r;
    waitForIncomingData([&](BufferHandle& data) {
      uint64_t sig;
      uint64_t rankId;
      uint32_t sourceRank;
      uint32_t destinationRank;
      uint32_t ttl;
      auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank, ttl);
      if (sourceRank == rank && sig == signature) {
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
