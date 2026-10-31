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
  // A rank's failure (see fail), passed on around the ring: the rank that failed and why
  static constexpr uint64_t signatureFail = 0x116dc74104a4dc0f;

  std::atomic_uint32_t connectionReady = 0;

  uint64_t myId;
  Vector<uint64_t> connectionIds;

  SpinMutex mutex;
  Vector<BufferHandle> incomingData;
  std::atomic_uint32_t incomingDataCount = 0;

  TcpContext context;
  Vector<SharedPtr<Listener>> listeners;
  SharedPtr<Connection> prev;
  SharedPtr<Connection> next;
  Vector<SharedPtr<Connection>> floatingConnections;

  std::atomic_bool dying = false;
  std::atomic_bool dead = false;

  // Why the waits fail (see fail), set once (under mutex)
  std::string failure;
  std::atomic_bool failed = false;

  SetupCommsImpl(size_t rank, size_t size) : SetupComms(rank, size) {

    myId = random<uint64_t>();
    connectionIds.resize(size);

    std::lock_guard l(mutex);

    for (auto& addr : {"0.0.0.0:0", "[::]:0"}) {

      try {
        auto listener = context.listen(addr);

        listener->accept([this](Error* error, SharedPtr<Connection> connection) {
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
              onClosed(connection);
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
    Vector<SharedPtr<Connection>> tmp;
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

  size_t prevRank() const {
    return (rank + size - 1) % size;
  }
  size_t nextRank() const {
    return (rank + 1) % size;
  }

  void waitForConnections() {
    if (size == 1) {
      return;
    }
    const auto start = std::chrono::steady_clock::now();
    std::unique_lock l(mutex);
    while (!prev || !next) {
      const bool missingPrev = !prev;
      const bool missingNext = !next;
      l.unlock();
      checkFailed();
      checkTimeout(start, [&] {
        return missingPrev && missingNext && prevRank() != nextRank()
                   ? fmt::sprintf("its setup connections with ranks %d and %d", prevRank(), nextRank())
                   : fmt::sprintf("its setup connection with rank %d", missingPrev ? prevRank() : nextRank());
      });
      futexWait(&connectionReady, 0, std::chrono::seconds(1));
      connectionReady = 0;
      l.lock();
    }
  }

  // Wakes the waits (to see a failure)
  void wake() {
    connectionReady = 1;
    futexWakeAll(&connectionReady);
    ++incomingDataCount;
    futexWakeAll(&incomingDataCount);
  }

  // Sends a failure (of rank origin) to the neighbors, except to the one at rank `except` (-1: none): they pass it on
  // in the same direction (see onRead). Under mutex.
  void sendFail(uint32_t origin, std::string_view reason, uint32_t ttl, size_t except) {
    for (auto* c : {&prev, &next}) {
      const size_t peer = c == &prev ? prevRank() : nextRank();
      if (!*c || peer == except) {
        continue;
      }
      auto buffer = serializeToBuffer(signatureFail, myId, (uint32_t)rank, ~(uint32_t)0, ttl, origin, reason);
      (*c)->write(std::move(buffer), nullptr);
    }
  }

  void fail(std::string reason) {
    std::unique_lock l(mutex);
    if (failed) {
      return;
    }
    failure = std::move(reason);
    failed = true;
    sendFail(rank, failure, size, -1);
    l.unlock();
    wake();
  }

  void checkFailed() {
    if (failed.load(std::memory_order_acquire)) {
      std::lock_guard l(mutex);
      throw SetupCommsError(fmt::sprintf("moodist: rank %d cannot set up its process group: %s", rank, failure));
    }
  }

  std::string failureReason() {
    if (!failed.load(std::memory_order_acquire)) {
      return {};
    }
    std::lock_guard l(mutex);
    return failure;
  }

  [[noreturn]] void timedOut(std::chrono::steady_clock::time_point start, std::string what) {
    const float s = seconds(std::chrono::steady_clock::now() - start);
    fail(fmt::sprintf("rank %d timed out after %g s waiting for %s", rank, s, what));
    throw SetupCommsError(fmt::sprintf("moodist: rank %d timed out after %g s waiting for %s (the process group's "
                                       "timeout; MOODIST_TIMEOUT overrides it)",
        rank, s, what));
  }

  // A connection closed: if it is one of the ring's, the rank at its other end failed or exited, and this rank's
  // setup cannot complete
  void onClosed(const SharedPtr<Connection>& connection) {
    std::unique_lock l(mutex);
    if (dying || (connection != prev && connection != next)) {
      return;
    }
    const size_t peer = connection == prev ? prevRank() : nextRank();
    l.unlock();
    fail(fmt::sprintf("rank %d lost its setup connection with rank %d, which failed or exited", rank, peer));
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
        onClosed(connection);
        connection->close();
      }
    });
  }

  void onRead(BufferHandle data, const SharedPtr<Connection>& connection) {
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
    if (sig == signatureFail) {
      uint32_t origin;
      std::string_view reason;
      try {
        deserializeBuffer(view, origin, reason);
      } catch (const std::exception&) {
        return;
      }
      if (failed || origin >= size || sourceRank >= size) {
        return;
      }
      failure = std::string(reason);
      failed = true;
      // (on around the ring, away from the neighbor it came from)
      if (ttl > 1) {
        sendFail(origin, reason, ttl - 1, sourceRank);
      }
      l.unlock();
      wake();
      return;
    }
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
  // Waits until f takes one of the incoming messages; describe() says what for (see checkTimeout)
  template<typename F, typename D>
  void waitForIncomingData(F&& f, D&& describe) {
    const auto start = std::chrono::steady_clock::now();
    size_t prev = 0;
    while (true) {
      size_t cur = incomingDataCount;
      while (cur == prev) {
        checkFailed();
        checkTimeout(start, describe);
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
    // (for the timeout's message: each rank's data comes around the ring through the ranks between it and this one,
    // so the nearest rank before this one whose data has not arrived holds up that of the others missing)
    auto missing = [&] {
      size_t n = 0;
      size_t nearest = rank;
      for (size_t d = 1; d != size; ++d) {
        const size_t i = (rank + size - d) % size;
        if (!allgatherBuffers[i]) {
          nearest = n == 0 ? i : nearest;
          ++n;
        }
      }
      return n > 1 ? fmt::sprintf("a setup allgather (no data from rank %d, nor from %d other%s whose data comes "
                                  "through it)",
                         nearest, n - 1, n > 2 ? "s" : "")
                   : fmt::sprintf("a setup allgather (no data from rank %d)", nearest);
    };
    while (remaining) {
      waitForIncomingData(
          [&](BufferHandle& data) {
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
          },
          missing);
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
  std::string_view recvBufferFrom(size_t source) {
    std::string_view r;
    waitForIncomingData(
        [&](BufferHandle& data) {
          uint64_t sig;
          uint64_t rankId;
          uint32_t sourceRank;
          uint32_t destinationRank;
          uint32_t ttl;
          auto view = deserializeBufferPart(data, sig, rankId, sourceRank, destinationRank, ttl);
          if (sourceRank == source && sig == signature) {
            recvBuffer = std::move(data);
            r = view;
            return true;
          }
          return false;
        },
        [&] {
          return fmt::sprintf("a setup message from rank %d", source);
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
void SetupComms::fail(std::string reason) {
  ((SetupCommsImpl*)this)->fail(std::move(reason));
}
void SetupComms::checkFailed() {
  ((SetupCommsImpl*)this)->checkFailed();
}
std::string SetupComms::failureReason() {
  return ((SetupCommsImpl*)this)->failureReason();
}
void SetupComms::timedOut(std::chrono::steady_clock::time_point start, std::string what) {
  ((SetupCommsImpl*)this)->timedOut(start, std::move(what));
}

std::unique_ptr<SetupComms> createSetupComms(size_t rank, size_t size) {
  return std::make_unique<SetupCommsImpl>(rank, size);
}

} // namespace moodist
