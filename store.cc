// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "store.h"
#include "buffer.h"
#include "common.h"
#include "connection.h"
#include "serialization.h"
#include "socket.h"
#include "synchronization.h"

#include <pybind11/gil.h>
#include <netdb.h>
#include <sys/socket.h>

namespace moodist {

static constexpr uint64_t signatureConnect = 0x120a63c0941561a4;
static constexpr uint64_t signatureConnectAck = 0x120a63c0941561a5;
static constexpr uint64_t signatureAddresses = 0x10f963c0941561a6;
static constexpr uint64_t signatureAddressesAck = 0x10f963c0941561a7;
static constexpr uint64_t signatureMessage = 0x220a74d1a52672b8;
static constexpr uint64_t signatureMessageAck = 0x220a74d1a52672b9;
static constexpr uint64_t signatureKeepalive = 0x220a74d1a53772ba;
static constexpr uint64_t signatureKeepaliveAck = 0x220a74d1a53772bb;
static constexpr uint64_t signatureDiagnosticHello = 0x330b85e2b63783cc;
static constexpr uint64_t signatureDiagnosticHelloAck = 0x330b85e2b63783cd;
static constexpr uint64_t signatureDiagnosticInfo = 0x330b85e2b63783ce;

static constexpr uint8_t messageDone = 0x3;
static constexpr uint8_t messageSet = 0x4;
static constexpr uint8_t messageGet = 0x5;
static constexpr uint8_t messageCheck = 0x6;
static constexpr uint8_t messageWait = 0x7;
static constexpr uint8_t messageExit = 0x8;
static constexpr uint8_t messageDiagnosticSource = 0x9;

namespace {
TcpContext context;

// Helper function to resolve an IP address string to sockaddr(s) synchronously
Vector<Vector<char>> resolveAddressSync(std::string_view addrStr) {
  Vector<Vector<char>> result;
  auto [host, port] = decodeIpAddress(addrStr);

  struct addrinfo hints = {};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;

  std::string hostStr(host);
  std::string portStr = std::to_string(port);

  struct addrinfo* res = nullptr;
  int err = getaddrinfo(hostStr.c_str(), portStr.c_str(), &hints, &res);
  if (err == 0 && res) {
    for (auto* p = res; p; p = p->ai_next) {
      if (p->ai_family == AF_INET || p->ai_family == AF_INET6) {
        Vector<char> addr;
        addr.resize(p->ai_addrlen);
        std::memcpy(addr.data(), p->ai_addr, p->ai_addrlen);
        result.push_back(std::move(addr));
      }
    }
    freeaddrinfo(res);
  }
  return result;
}

template<typename T>
struct Indestructible {
  std::aligned_storage_t<sizeof(T), alignof(T)> storage;
  Indestructible() {
    new (&storage) T();
  }
  T& operator*() {
    return (T&)storage;
  }
  T* operator->() {
    return &**this;
  }
};

Indestructible<SpinMutex> activeStoresMutex;
Indestructible<Vector<StoreImpl*>> activeStores;
} // namespace

struct StoreImpl {
  Vector<std::shared_ptr<Socket>> udps;

  std::atomic_size_t refcount = 0;

  std::string hostname;
  int port;
  uint32_t worldSize;
  uint32_t rank;
  std::string storekey;
  std::string myId;

  std::atomic_bool dead = false;
  std::atomic_uint32_t anyQueued = 0;
  std::thread thread;

  struct Callback {
    std::chrono::steady_clock::time_point time;
    Function<void()> callback;
  };

  Vector<Callback> callbacks;
  SpinMutex mutex;
  SpinMutex queuedCallbacksMutex;
  Vector<Callback> queuedCallbacks;

  struct ConnectionInfo {
    std::shared_ptr<Connection> connection;
    std::chrono::steady_clock::time_point lastReceive;
    uint32_t sourceRank = -1;

    void close() {
      connection->close();
    }
  };

  Vector<std::shared_ptr<Listener>> listeners;
  Vector<std::shared_ptr<ConnectionInfo>> floatingConnections;

  std::atomic_bool connectAcked = false;
  int connectCounter = 0;
  HashMap<uint32_t, std::string> unackedAddresses;

  bool destroyed = false;
  std::chrono::steady_clock::time_point destroyTime;
  std::optional<uint32_t> destroySourceRank;

  bool deletionInitiated = false;
  bool deleted = false;

  struct PeerInfo {
    std::string uid;
    std::string networkKey;
    HashMap<uint32_t, std::string> addresses;
    Vector<char> udpaddr;
    HashMap<uint32_t, bool> connected;

    std::chrono::steady_clock::time_point lastConnectEdge;

    Vector<std::shared_ptr<ConnectionInfo>> tcpconnections;
    uint32_t incomingSeq = 0;
    uint32_t outgoingSeq = 0;

    struct OutgoingMessage {
      uint32_t seq;
      SharedBufferHandle buffer;
      std::chrono::steady_clock::time_point queueTime;
      uint32_t diagnosticSource = -1;
    };
    Vector<OutgoingMessage> outgoingQueue;

    struct IncomingMessage {
      uint32_t seq;
      BufferHandle buffer;
    };

    Vector<IncomingMessage> incomingQueue;

    bool destroyed = false;

    // Tracks which diagnostic sources we've sent messageDiagnosticSource for to this edge
    HashSet<uint32_t> sentDiagnosticSourceFor;
  };
  Vector<std::optional<PeerInfo>> peerInfos;

  HashMap<uint32_t, std::unique_ptr<PeerInfo>> edgePeers;

  // Info about diagnostic sources (originators of requests) for sending diagnostics via UDP
  struct DiagnosticSourceInfo {
    Vector<std::string> addrStrings;  // original address strings
    Vector<Vector<char>> resolvedAddrs;  // resolved sockaddrs
    bool verified = false;  // have we received a hello ack?
    Vector<char> verifiedAddr;  // the actual UDP address that worked (from recvFromAddr)
    std::chrono::steady_clock::time_point lastAttempt;
    int attemptCount = 0;
  };
  HashMap<uint32_t, DiagnosticSourceInfo> diagnosticSources;  // diagnosticSource rank -> info

  // Our own UDP addresses for diagnostic purposes
  Vector<std::string> myUdpAddresses;

  // Received diagnostic info from intermediate ranks (for timeout error messages)
  struct ReceivedDiagnostic {
    uint32_t reportingRank;  // rank that sent the diagnostic
    uint32_t stuckEdge;      // edge that the reporting rank is stuck on
    std::string reason;
    std::chrono::steady_clock::time_point receiveTime;
  };
  Vector<ReceivedDiagnostic> receivedDiagnostics;

  Vector<Vector<uint32_t>> edges;
  Vector<uint32_t> firsthop;

  std::chrono::steady_clock::time_point prevConnectTime = {};

  void onreadudp(Socket* socket, Error* e) {
    if (e) {
      log.error("Moodist store recv error: %s\n", e->what());
      return;
    }

    char buf[0x1000];
    std::array<iovec, 1> vec;
    vec[0].iov_base = buf;
    vec[0].iov_len = 0x1000;
    size_t n = socket->readv(vec.data(), vec.size());

    if (n < 8) {
      return;
    }
    try {
      uint64_t signature;
      std::memcpy(&signature, buf, 8);
      std::string_view view(&buf[8], &buf[n]);
      if (signature == signatureConnect) {
        std::string key;
        uint32_t sourceRank;
        std::string uid;
        std::string networkKey;
        std::vector<std::pair<uint32_t, std::string>> addresses;
        deserializeBufferPart(view, key, sourceRank, uid, networkKey, addresses);
        if (key == storekey && sourceRank < worldSize && rank == 0) {
          auto udpaddr = socket->recvFromAddr();

          {
            std::lock_guard l(mutex);
            auto& opt = peerInfos.at(sourceRank);
            if (opt) {
              if (opt->uid != uid) {
                log.error(
                    "Moodist store: two different processes connected, both claiming to be rank %d (uid %s vs %s)",
                    sourceRank, opt->uid, uid);
                return;
              }
            } else {
              opt.emplace();
            }
            PeerInfo& pi = *opt;
            pi.uid = uid;
            pi.networkKey = networkKey;
            for (auto& v : addresses) {
              pi.addresses.insert(v);
            }
            pi.udpaddr.resize(udpaddr.second);
            std::memcpy(pi.udpaddr.data(), udpaddr.first, udpaddr.second);
          }

          std::vector<uint32_t> ra;
          for (auto& v : addresses) {
            ra.push_back(v.first);
          }

          auto buf = serializeToBuffer(signatureConnectAck, key, sourceRank, uid, ra);

          ::sendto(
              socket->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL, (sockaddr*)udpaddr.first, udpaddr.second);

          for (uint32_t e : edges[sourceRank]) {
            sendAddresses(sourceRank, e);
          }
        }
      } else if (signature == signatureConnectAck) {
        std::string key;
        uint32_t sourceRank;
        std::string uid;
        std::vector<uint32_t> addresses;
        deserializeBufferPart(view, key, sourceRank, uid, addresses);
        if (key == storekey && sourceRank == rank && uid == myId) {
          // log.info("Moodist store connected\n");
          std::lock_guard l(mutex);
          size_t n = 0;
          for (uint32_t i : addresses) {
            auto it = unackedAddresses.find(i);
            if (it != unackedAddresses.end()) {
              unackedAddresses.erase(it);
              ++n;
            }
          }
          // log.info("acked %d addresses, %d left\n", n, unackedAddresses.size());
          if (unackedAddresses.empty()) {
            connectAcked = true;
          }
        }
      } else if (signature == signatureAddresses) {
        std::string key;
        uint32_t sourceRank;
        std::string sourceId;
        std::string networkKey;
        std::vector<std::pair<uint32_t, std::string>> addresses;
        deserializeBufferPart(view, key, sourceRank, sourceId, networkKey, addresses);
        if (key == storekey && sourceRank < worldSize) {
          {
            std::lock_guard l(mutex);
            CHECK(sourceRank != -1);
            PeerInfo& pi = getEdge(sourceRank);
            pi.uid = sourceId;
            pi.networkKey = networkKey;
            for (auto& v : addresses) {
              pi.addresses.insert(v);
            }
            auto udpaddr = socket->recvFromAddr();
            pi.udpaddr.resize(udpaddr.second);
            std::memcpy(pi.udpaddr.data(), udpaddr.first, udpaddr.second);

            if (!pi.tcpconnections.empty()) {
              auto buf = serializeToBuffer(signatureAddressesAck, key, rank, sourceRank);
              ::sendto(
                  socket->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL, (sockaddr*)udpaddr.first, udpaddr.second);
            }
          }
          connectEdge(sourceRank);
        }
      } else if (signature == signatureAddressesAck) {
        std::string key;
        uint32_t sourceRank;
        uint32_t edge;
        deserializeBufferPart(view, key, sourceRank, edge);
        if (key == storekey && sourceRank < worldSize) {
          std::lock_guard l(mutex);
          auto& pi = peerInfos[sourceRank];
          if (pi) {
            peerInfos[sourceRank]->connected[edge] = true;
          }
        }
      } else if (signature == signatureDiagnosticHello) {
        // Another rank is trying to establish a diagnostic UDP path with us
        std::string key;
        uint32_t sourceRank;
        deserializeBufferPart(view, key, sourceRank);
        if (key == storekey && sourceRank < worldSize) {
          // Send ack back
          auto udpaddr = socket->recvFromAddr();
          auto buf = serializeToBuffer(signatureDiagnosticHelloAck, key, rank);
          ::sendto(
              socket->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL,
              (sockaddr*)udpaddr.first, udpaddr.second);
        }
      } else if (signature == signatureDiagnosticHelloAck) {
        // Our diagnostic hello was acknowledged - UDP path is verified
        std::string key;
        uint32_t sourceRank;
        deserializeBufferPart(view, key, sourceRank);
        if (key == storekey && sourceRank < worldSize) {
          auto udpaddr = socket->recvFromAddr();
          std::lock_guard l(mutex);
          auto it = diagnosticSources.find(sourceRank);
          if (it != diagnosticSources.end() && !it->second.verified) {
            it->second.verified = true;
            // Store the actual UDP address that worked
            it->second.verifiedAddr.resize(udpaddr.second);
            std::memcpy(it->second.verifiedAddr.data(), udpaddr.first, udpaddr.second);
          }
        }
      } else if (signature == signatureDiagnosticInfo) {
        // An intermediate rank is telling us about a problem it encountered
        std::string key;
        uint32_t reportingRank;
        uint32_t stuckEdge;
        std::string reason;
        deserializeBufferPart(view, key, reportingRank, stuckEdge, reason);
        if (key == storekey && reportingRank < worldSize) {
          std::lock_guard l(mutex);
          // Store the diagnostic info for later inclusion in timeout messages
          receivedDiagnostics.push_back(ReceivedDiagnostic{
              .reportingRank = reportingRank,
              .stuckEdge = stuckEdge,
              .reason = std::move(reason),
              .receiveTime = std::chrono::steady_clock::now()
          });
          // Keep only recent diagnostics (last 60 seconds)
          auto cutoff = std::chrono::steady_clock::now() - std::chrono::seconds(60);
          while (!receivedDiagnostics.empty() && receivedDiagnostics.front().receiveTime < cutoff) {
            receivedDiagnostics.pop_front();
          }
        }
      }

    } catch (const SerializationError& e) {
      log.error("store udp recv error: %s\n", e.what());
    }
  }

  void reaptcpconnections(PeerInfo& pi) {
    for (auto i = pi.tcpconnections.begin(); i != pi.tcpconnections.end();) {
      auto& v = *i;
      if (v->connection->closed()) {
        i = pi.tcpconnections.erase(i);
      } else {
        ++i;
      }
    }
  }

  void connectEdge(uint32_t edge) {
    std::unique_lock l(mutex);
    if (dead) {
      return;
    }
    CHECK(edgePeers.contains(edge));
    CHECK(edge != -1);
    auto& pi = getEdge(edge);
    reaptcpconnections(pi);
    if (!pi.tcpconnections.empty()) {
      return;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - pi.lastConnectEdge <= std::chrono::seconds(1)) {
      return;
    }
    pi.lastConnectEdge = now;
    std::string uid = pi.uid;
    size_t n = 0;
    for (auto& a : pi.addresses) {
      if (!context.isReachable(pi.networkKey, a.second)) {
        continue;
      }
      addCallback(
          std::chrono::steady_clock::now() + std::chrono::milliseconds(250 * n), [this, a = a.second, edge, uid] {
            std::lock_guard l(mutex);
            CHECK(edge != -1);
            auto& pi = getEdge(edge);
            if (dead || !pi.tcpconnections.empty() || pi.destroyed || destroyed) {
              return;
            }
            // log.info("connecting to edge %d at %s\n", edge, a);
            auto connection = context.connect(a);
            auto buffer = serializeToBuffer(signatureConnect, storekey, rank, myId, edge, uid);
            connection->write(std::move(buffer), nullptr);

            addConnection(std::move(connection));
          });
      ++n;
    }
  }

  void sendAddresses(uint32_t sourceRank, uint32_t edge) {
    std::unique_lock l(mutex);
    auto& pi = peerInfos[sourceRank];
    CHECK(pi);
    auto& pi2 = peerInfos[edge];
    if (!pi2 || pi2->connected.contains(sourceRank)) {
      return;
    }
    Vector<std::pair<uint32_t, std::string>> addresses;
    for (auto& v : pi->addresses) {
      addresses.push_back(v);
    }
    if (addresses.size() > 8) {
      std::ranges::shuffle(addresses, getRng());
      addresses.resize(8);
    }
    auto buf = serializeToBuffer(signatureAddresses, storekey, sourceRank, pi->uid, pi->networkKey, addresses);
    int r = ::sendto(
        udps.at(random<size_t>(0, udps.size() - 1))->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL,
        (sockaddr*)pi2->udpaddr.data(), pi2->udpaddr.size());
    auto t = std::chrono::milliseconds(2000);
    if (r < 0) {
      t = std::chrono::milliseconds(250);
    }
    l.unlock();

    addCallback(std::chrono::steady_clock::now() + t, [this, sourceRank, edge] { sendAddresses(sourceRank, edge); });
  }

  // Try to establish a verified UDP connection to a diagnostic source
  void tryDiagnosticHello(uint32_t diagnosticSource) {
    std::unique_lock l(mutex);
    auto it = diagnosticSources.find(diagnosticSource);
    if (it == diagnosticSources.end() || it->second.verified || it->second.resolvedAddrs.empty()) {
      return;
    }

    auto& info = it->second;
    auto now = std::chrono::steady_clock::now();

    // Don't retry too frequently
    if (info.attemptCount > 0 && now - info.lastAttempt < std::chrono::milliseconds(500)) {
      return;
    }

    info.lastAttempt = now;
    info.attemptCount++;

    // Try the next address in round-robin fashion
    size_t addrIndex = (info.attemptCount - 1) % info.resolvedAddrs.size();
    auto& addr = info.resolvedAddrs[addrIndex];

    auto buf = serializeToBuffer(signatureDiagnosticHello, storekey, rank);
    ::sendto(
        udps.at(random<size_t>(0, udps.size() - 1))->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL,
        (sockaddr*)addr.data(), addr.size());

    // Schedule retry if not verified after some attempts
    if (info.attemptCount < 10) {
      l.unlock();
      addCallback(now + std::chrono::milliseconds(500 * info.attemptCount), [this, diagnosticSource] {
        tryDiagnosticHello(diagnosticSource);
      });
    }
  }

  // Send diagnostic info via UDP to a diagnostic source about a stuck edge
  void sendDiagnosticInfo(uint32_t diagnosticSource, uint32_t stuckEdge, const std::string& reason) {
    auto it = diagnosticSources.find(diagnosticSource);
    if (it == diagnosticSources.end()) {
      return;  // No info about this diagnostic source
    }

    auto& info = it->second;

    // Prefer the verified address, but fall back to resolved addresses
    const char* addr = nullptr;
    size_t addrLen = 0;

    if (!info.verifiedAddr.empty()) {
      addr = info.verifiedAddr.data();
      addrLen = info.verifiedAddr.size();
    } else if (!info.resolvedAddrs.empty()) {
      // Try the first resolved address
      addr = info.resolvedAddrs[0].data();
      addrLen = info.resolvedAddrs[0].size();
    } else {
      return;  // No addresses to try
    }

    auto buf = serializeToBuffer(signatureDiagnosticInfo, storekey, rank, stuckEdge, reason);
    ::sendto(
        udps.at(random<size_t>(0, udps.size() - 1))->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL,
        (sockaddr*)addr, addrLen);
  }

  // Check for stuck messages and send diagnostic info
  void checkStuckMessages() {
    std::lock_guard l(mutex);
    auto now = std::chrono::steady_clock::now();

    for (auto& [edge, piPtr] : edgePeers) {
      auto& pi = *piPtr;

      for (auto& msg : pi.outgoingQueue) {
        if (msg.diagnosticSource == (uint32_t)-1 || msg.diagnosticSource == rank) {
          continue;  // Skip messages without diagnostic source or from ourselves
        }

        auto age = now - msg.queueTime;
        if (age < std::chrono::seconds(5)) {
          continue;  // Message hasn't been stuck long enough
        }

        // Determine the reason for being stuck
        std::string reason;
        if (pi.tcpconnections.empty()) {
          if (pi.addresses.empty()) {
            reason = fmt::sprintf("no TCP addresses for rank %d", edge);
          } else {
            reason = fmt::sprintf("no TCP connection to rank %d (addresses known)", edge);
          }
        } else {
          reason = fmt::sprintf("message pending to rank %d for %.1fs (TCP connected)", edge, seconds(age));
        }

        sendDiagnosticInfo(msg.diagnosticSource, edge, reason);
      }
    }
  }

  bool listen(int port, bool ipv6) {
    auto socket = std::make_shared<Socket>(Socket::Udp(ipv6));
    if (!socket->bind(port)) {
      return false;
    }
    socket->setOnRead([this, socket = &*socket](Error* e) { this->onreadudp(socket, e); });
    udps.push_back(std::move(socket));
    return true;
  }

  void threadEntry() {
    while (!dead) {
      if (callbacks.empty()) {
        futexWait(&anyQueued, 0, std::chrono::seconds(100));
      } else {
        futexWait(&anyQueued, 0, callbacks.front().time - std::chrono::steady_clock::now());
      }
      if (anyQueued) {
        std::lock_guard l(queuedCallbacksMutex);
        anyQueued = 0;
        bool sort = false;
        for (auto& v : queuedCallbacks) {
          if (!callbacks.empty() && v.time < callbacks.back().time) {
            sort = true;
          }
          callbacks.push_back(std::move(v));
        }
        queuedCallbacks.clear();
        if (sort) {
          std::sort(
              callbacks.begin(), callbacks.end(), [](const Callback& a, const Callback& b) { return a.time < b.time; });
        }
      }
      if (!callbacks.empty()) {
        if (std::chrono::steady_clock::now() >= callbacks.front().time) {
          callbacks.pop_front_value().callback();
        }
      }
    }
  }

  void addCallback(std::chrono::steady_clock::time_point time, Function<void()> callback) {
    {
      std::lock_guard l(queuedCallbacksMutex);
      queuedCallbacks.emplace_back();
      queuedCallbacks.back().time = time;
      queuedCallbacks.back().callback = std::move(callback);
      anyQueued = 1;
    }
    futexWakeAll(&anyQueued);
  }

  void onreadtcp(BufferHandle data, const std::shared_ptr<ConnectionInfo>& ci) {
    size_t n = data->size();
    const char* buf = (const char*)data->data();

    if (n < 8) {
      return;
    }

    try {
      uint64_t signature;
      std::memcpy(&signature, buf, 8);
      std::string_view view(&buf[8], &buf[n]);
      if (signature == signatureConnect || signature == signatureConnectAck) {
        std::string key;
        uint32_t sourceRank;
        std::string sourceId;
        uint32_t destinationRank;
        std::string destinationId;
        deserializeBufferPart(view, key, sourceRank, sourceId, destinationRank, destinationId);
        if (key == storekey && destinationRank == rank && destinationId == myId) {
          if (std::ranges::find(edges[rank], sourceRank) != edges[rank].end()) {

            auto now = std::chrono::steady_clock::now();

            {
              std::lock_guard l(mutex);
              CHECK(sourceRank != -1);
              auto& pi = getEdge(sourceRank);
              if (dead || !pi.tcpconnections.empty()) {
                ci->connection->close();
              } else {
                ci->sourceRank = sourceRank;
                ci->lastReceive = std::chrono::steady_clock::now();
                pi.tcpconnections.push_back(ci);

                // log.info("connected to rank %d\n", sourceRank);

                auto buf = serializeToBuffer(signatureAddressesAck, key, rank, sourceRank);
                ::sendto(
                    udps.at(random<size_t>(0, udps.size() - 1))->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL,
                    (sockaddr*)pi.udpaddr.data(), pi.udpaddr.size());

                if (signature == signatureConnect) {
                  auto buffer = serializeToBuffer(signatureConnectAck, key, rank, myId, sourceRank, sourceId);
                  ci->connection->write(std::move(buffer), nullptr);
                }

                for (auto& v : pi.outgoingQueue) {
                  ci->connection->write(v.buffer, nullptr);
                }
              }
            }
          }
        }
      } else if (signature == signatureMessage && ci->sourceRank < worldSize) {
        uint32_t ttl;
        uint32_t seq;
        deserializeBufferPart(view, ttl, seq);
        CHECK(ttl != 0);
        std::lock_guard l(mutex);
        CHECK(edgePeers.contains(ci->sourceRank));
        CHECK(ci->sourceRank != -1);
        auto& pi = getEdge(ci->sourceRank);

        ci->lastReceive = std::chrono::steady_clock::now();

        auto buffer = serializeToBuffer(signatureMessageAck, seq);
        ci->connection->write(std::move(buffer), nullptr);

        if ((int32_t)(seq - pi.incomingSeq) >= 0) {
          if (seq == pi.incomingSeq) {
            processMessage(std::move(data), ci->sourceRank);
            ++pi.incomingSeq;
            std::ranges::sort(pi.incomingQueue, [&](auto& a, auto& b) {
              return (int32_t)(a.seq - pi.incomingSeq) < (int32_t)(b.seq - pi.incomingSeq);
            });
            while (!pi.incomingQueue.empty()) {
              auto& v = pi.incomingQueue.front();
              if ((int32_t)(seq - pi.incomingSeq) < 0) {
                pi.incomingQueue.pop_front();
                continue;
              }
              if (v.seq != pi.incomingSeq) {
                break;
              }
              auto x = std::move(v);
              pi.incomingQueue.pop_front();
              processMessage(std::move(x.buffer), ci->sourceRank);
              ++pi.incomingSeq;
            }
          } else {
            pi.incomingQueue.emplace_back();
            pi.incomingQueue.back().seq = seq;
            pi.incomingQueue.back().buffer = std::move(data);
          }
        }
      } else if (signature == signatureMessageAck) {
        uint32_t seq;
        deserializeBufferPart(view, seq);
        std::lock_guard l(mutex);
        ci->lastReceive = std::chrono::steady_clock::now();
        CHECK(edgePeers.contains(ci->sourceRank));
        CHECK(ci->sourceRank != -1);
        auto& pi = getEdge(ci->sourceRank);
        for (auto i = pi.outgoingQueue.begin(); i != pi.outgoingQueue.end(); ++i) {
          if (i->seq == seq) {
            i = pi.outgoingQueue.erase(i);
            break;
          }
        }
      } else if (signature == signatureKeepalive) {
        std::lock_guard l(mutex);
        ci->lastReceive = std::chrono::steady_clock::now();
        // log.info("keepalive ack to %d\n", ci->sourceRank);
        ci->connection->write(serializeToBuffer(signatureKeepaliveAck), nullptr);
      } else if (signature == signatureKeepaliveAck) {
        std::lock_guard l(mutex);
        ci->lastReceive = std::chrono::steady_clock::now();
      }

    } catch (const SerializationError& e) {
      log.error("store tcp recv error: %s\n", e.what());
    }
  }

  PeerInfo& getEdge(uint32_t rank) {
    auto& ptr = edgePeers[rank];
    if (!ptr) {
      ptr = std::make_unique<PeerInfo>();
    }
    return *ptr;
  }

  void processMessage(BufferHandle data, uint32_t edgeRank) {
    size_t n = data->size();
    const char* buf = (const char*)data->data();

    std::string_view view(&buf[8], &buf[n]);

    uint32_t ttl;
    uint32_t seq;
    uint32_t destinationRank;
    uint32_t id;
    uint32_t sourceRank;
    uint32_t diagnosticSource;
    uint8_t t;
    view = deserializeBufferPart(view, ttl, seq, destinationRank, id, sourceRank, diagnosticSource, t);

    // log.info(
    //     "message ttl %d, seq %d, destination %d, id %#x, source %d, diagSrc %d, t %d\n", ttl, seq, destinationRank, id,
    //     sourceRank, diagnosticSource, t);

    if (destinationRank != rank) {
      CHECK(ttl != 0);
      --ttl;
      std::memcpy(&data->data()[8], &ttl, sizeof(ttl));

      uint32_t edge = firsthop[destinationRank];
      // log.info("forwarding message to %d through edge %d\n", destinationRank, edge);
      CHECK(edge != -1 && edge != worldSize);
      auto& pi = getEdge(edge);
      uint32_t seq = pi.outgoingSeq++;

      std::memcpy(&data->data()[12], &seq, sizeof(seq));

      sendMessage(edge, seq, std::move(data), pi, diagnosticSource);
      return;
    }

    if (t == messageSet) {
      std::string key;
      std::vector<uint8_t> value;
      deserializeBufferPart(view, key, value);
      internalStoreSetValue(key, std::move(value));

      CHECK(edgeRank != -1);
      auto& pi = getEdge(edgeRank);
      uint32_t seq = pi.outgoingSeq++;

      sendMessage(
          edgeRank, seq, serializeToBuffer(signatureMessage, worldSize, seq, sourceRank, id, rank, diagnosticSource, messageDone), pi, diagnosticSource);
    } else if (t == messageDone) {
      bool b = false;
      std::vector<uint8_t> value;
      if (view.size()) {
        if (view.size() == 1) {
          deserializeBufferPart(view, b);
        } else {
          deserializeBufferPart(view, value);
        }
      }
      auto i = waits.find(id);
      if (i != waits.end()) {
        i->second->b = b;
        i->second->data = std::move(value);
        i->second->futex = 1;
        futexWakeAll(&i->second->futex);
        waits.erase(i);
      }
    } else if (t == messageGet || t == messageWait) {
      std::string key;
      deserializeBufferPart(view, key);
      internalStoreGetValue(key, [this, edgeRank, seq, sourceRank, diagnosticSource, id, t, key](const std::vector<uint8_t>& data) {
        CHECK(edgeRank != -1);
        auto& pi = getEdge(edgeRank);
        uint32_t seq = pi.outgoingSeq++;
        if (t == messageGet) {
          sendMessage(
              edgeRank, seq,
              serializeToBuffer(signatureMessage, worldSize, seq, sourceRank, id, rank, diagnosticSource, messageDone, data), pi, diagnosticSource);
        } else {
          sendMessage(
              edgeRank, seq, serializeToBuffer(signatureMessage, worldSize, seq, sourceRank, id, rank, diagnosticSource, messageDone),
              pi, diagnosticSource);
        }
      });
    } else if (t == messageCheck) {
      std::string key;
      deserializeBufferPart(view, key);
      bool r = store.find(key) != store.end();
      CHECK(edgeRank != -1);
      auto& pi = getEdge(edgeRank);
      uint32_t seq = pi.outgoingSeq++;
      sendMessage(
          edgeRank, seq, serializeToBuffer(signatureMessage, worldSize, seq, sourceRank, id, rank, diagnosticSource, messageDone, r), pi, diagnosticSource);
    } else if (t == messageExit) {
      CHECK(edgeRank != -1);
      auto& pi = getEdge(edgeRank);
      pi.destroyed = true;
      if (!destroySourceRank) {
        destroySourceRank = sourceRank;
      }
      addCallback(std::chrono::steady_clock::now(), [this] { destroy(); });
    } else if (t == messageDiagnosticSource) {
      // Receive diagnostic source UDP addresses from another rank
      std::vector<std::string> addrs;
      deserializeBufferPart(view, addrs);

      auto& info = diagnosticSources[diagnosticSource];
      info.addrStrings = Vector<std::string>(addrs.begin(), addrs.end());
      // Resolve addresses synchronously
      for (const auto& addrStr : addrs) {
        auto resolved = resolveAddressSync(addrStr);
        for (auto& addr : resolved) {
          info.resolvedAddrs.push_back(std::move(addr));
        }
      }
      // Shuffle to avoid bias (e.g., all IPv6 addresses first)
      std::ranges::shuffle(info.resolvedAddrs, getRng());
      // Start diagnostic hello handshake to verify UDP path
      if (!info.resolvedAddrs.empty() && !info.verified) {
        tryDiagnosticHello(diagnosticSource);
      }
    }
  }

  void findPaths() {
    edges.resize(worldSize);
    auto addedge = [&](uint32_t i, uint32_t e) {
      if (i == e) {
        return;
      }
      auto& v = edges.at(i);
      if (std::ranges::find(v, e) == v.end()) {
        v.push_back(e);
        auto& v2 = edges.at(e);
        CHECK(std::ranges::find(v2, i) == v2.end());
        v2.push_back(i);
      }
    };

    std::mt19937_64 rng(64 + stringHash(storekey));

    Vector<uint32_t> connections;
    connections.resize(worldSize);
    for (uint32_t i : range(worldSize)) {
      connections[i] = i;
    };
    std::ranges::shuffle(connections, rng);

    if (worldSize > 4) {
      auto neighbors = [&](uint32_t a, uint32_t b) {
        uint32_t x = a > b ? a - b : b - a;
        return x == 1 || x == worldSize - 1;
      };
      for (uint32_t i : range(worldSize)) {
        while (connections[i] == i || neighbors(i, connections[i]) || connections[connections[i]] == i) {
          uint32_t x = std::uniform_int_distribution<size_t>(0, worldSize - 1)(rng);
          if (!neighbors(x, connections[i])) {
            std::swap(connections[i], connections[x]);
          }
        }
      }
    }

    for (uint32_t i : range(worldSize)) {
      addedge(i, (i + 1) % worldSize);
      addedge(i, connections[i]);
    }
    for (auto& v : edges) {
      std::ranges::shuffle(v, getRng());
    }
    struct Node {
      uint32_t i;
      uint32_t first;
      uint32_t distance = 0;
    };
    firsthop.resize(worldSize);
    std::ranges::fill(firsthop, worldSize);
    firsthop[rank] = rank;
    Vector<bool> visited;
    visited.resize(worldSize);
    Vector<Node> open;
    visited[rank] = true;
    for (auto& v : edges[rank]) {
      CHECK(!visited[v]);
      visited[v] = true;
      open.emplace_back();
      open.back().i = v;
      open.back().first = v;
      open.back().distance = 1;
      firsthop[v] = v;
    }
    while (!open.empty()) {
      auto x = open.pop_front_value();
      for (auto& v : edges[x.i]) {
        if (visited[v]) {
          continue;
        }
        visited[v] = true;
        open.emplace_back();
        open.back().i = v;
        open.back().first = x.first;
        open.back().distance = x.distance + 1;
        firsthop[v] = x.first;

        // log.info("%d found %d in distance %d\n", rank, v, x.distance + 1);
      }
    }
    for (auto& v : visited) {
      CHECK(v);
    }

    HashMap<uint32_t, int> hopCounts;

    for (uint32_t i : range(worldSize)) {
      // log.error("first hop to go %d -> %d is %d\n", rank, i, first[i]);
      hopCounts[firsthop[i]] += 1;
    }

    // for (auto& v : hopCounts) {
    //   log.info("hop count for %d is %d\n", v.first, v.second);
    // }
  }

  StoreImpl(std::string hostname, int port, std::string key, int worldSize, int rank)
      : hostname(hostname), port(port), worldSize(worldSize), rank(rank) {
    log.init();
    if (key.size() > 80) {
      key.resize(80);
    }
    storekey = fmt::sprintf("%s-%d-%d-%s", hostname, port, worldSize, key);
    CHECK(worldSize > 0);
    CHECK(rank >= 0 && rank < worldSize);

    std::lock_guard l(mutex);

    myId = randomName();

    findPaths();

    peerInfos.resize(worldSize);

    if (rank == 0) {
      bool success = listen(port, true);
      success |= listen(port, false);
      for (int i : range(success ? 4 : 16)) {
        if (listen(57360 + i, true) | listen(57360 + i, false)) {
          break;
        }
      }
    } else {
      listen(0, true);
      listen(0, false);
    }

    if (udps.empty()) {
      throw std::runtime_error("Moodist store failed to listen on any ports");
    }

    // Collect our own UDP addresses for diagnostic purposes
    for (auto& udp : udps) {
      for (auto& addr : udp->localAddresses()) {
        myUdpAddresses.push_back(addr);
      }
    }

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

    CHECK(unackedAddresses.empty());
    for (auto& v : listeners) {
      for (auto& x : v->localAddresses()) {
        unackedAddresses[unackedAddresses.size()] = x;
      }
    }

    thread = std::thread([this]() { threadEntry(); });

    addCallback(std::chrono::steady_clock::now(), [this]() { connect(); });
    addCallback(std::chrono::steady_clock::now() + std::chrono::seconds(8), [this]() { keepalive(); });

    {
      std::lock_guard l(*activeStoresMutex);
      activeStores->push_back(this);
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

  ~StoreImpl() {
    {
      std::unique_lock l(mutex);
      dead = true;
      anyQueued = 1;
      futexWakeAll(&anyQueued);
    }
    closeAll(listeners);
    closeAll(floatingConnections);
    while (true) {
      Vector<std::shared_ptr<ConnectionInfo>> allconnections;
      std::unique_lock l(mutex);
      for (auto& v : edgePeers) {
        for (auto& x : v.second->tcpconnections) {
          allconnections.push_back(x);
        }
        v.second->tcpconnections.clear();
      }
      l.unlock();
      if (allconnections.empty()) {
        break;
      }
      closeAll(allconnections);
    }
    thread.join();
    for (auto& v : udps) {
      v->close();
    }

    {
      std::lock_guard l(*activeStoresMutex);
      auto i = std::ranges::find(*activeStores, this);
      CHECK(i != activeStores->end());
      activeStores->erase(i);
    }
  }

  void destroy() {
    std::unique_lock l(mutex);
    if (!destroyed) {
      destroyed = true;
      destroyTime = std::chrono::steady_clock::now();
      if (!destroySourceRank) {
        destroySourceRank = rank;
      }
      for (auto& v : waits) {
        destroyWait(v.second);
      }
      addCallback(std::chrono::steady_clock::now(), [this] {
        std::unique_lock l(mutex);
        for (auto& v : edgePeers) {
          auto& pi = *v.second;
          if (!pi.tcpconnections.empty()) {
            uint32_t edgeRank = v.first;
            uint32_t seq = pi.outgoingSeq++;
            // diagnosticSource = rank (not important for destroy, but needed for message format)
            sendMessage(
                edgeRank, seq,
                serializeToBuffer(
                    signatureMessage, worldSize, seq, edgeRank, (uint32_t)0, *destroySourceRank, rank, messageExit),
                pi, rank);
          }
        }
      });
    }

    if (refcount == 0 && !deletionInitiated) {
      deletionInitiated = true;
      auto t = std::chrono::milliseconds(50);
      while (t < std::chrono::seconds(2)) {
        addCallback(std::chrono::steady_clock::now() + t, [this] {
          std::lock_guard l(mutex);
          for (auto& v : edgePeers) {
            auto& pi = *v.second;
            if (!pi.tcpconnections.empty() && !pi.destroyed) {
              return;
            }
          }
          if (!deleted) {
            deleted = true;
            scheduler.run([this] { delete this; });
          }
        });
        t += std::chrono::milliseconds(50);
      }

      addCallback(std::chrono::steady_clock::now() + std::chrono::seconds(2), [this] {
        std::lock_guard l(mutex);
        if (!deleted) {
          deleted = true;
          scheduler.run([this] { delete this; });
        }
      });
    }
  }

  void keepalive() {
    std::unique_lock l(mutex);

    auto now = std::chrono::steady_clock::now();

    Vector<uint32_t> reconnect;

    for (auto& v : edgePeers) {
      auto& pi = *v.second;
      if (pi.tcpconnections.empty() && !pi.addresses.empty()) {
        reconnect.push_back(v.first);
      }
      for (auto& ci : pi.tcpconnections) {
        if (now - ci->lastReceive >= std::chrono::seconds(40) || ci->connection->closed()) {
          // log.error("Moodist Store connection to %d timed out\n", ci->sourceRank);
          ci->connection->close();
          if (!pi.addresses.empty()) {
            reconnect.push_back(ci->sourceRank);
          }
        } else if (now - ci->lastReceive >= std::chrono::seconds(20)) {
          // log.info("send keepalive to %d\n", ci->sourceRank);
          ci->connection->write(serializeToBuffer(signatureKeepalive), nullptr);
        }
      }
      reaptcpconnections(pi);
    }

    // log.info("floatingConnections.size() is %d\n", floatingConnections.size());

    for (auto i = floatingConnections.begin(); i != floatingConnections.end();) {
      auto& ci = *i;
      if (ci->sourceRank != -1) {
        i = floatingConnections.erase(i);
      } else if (now - ci->lastReceive >= std::chrono::seconds(20)) {
        ci->connection->close();
        i = floatingConnections.erase(i);
      } else {
        ++i;
      }
    }

    l.unlock();
    for (uint32_t r : reconnect) {
      connectEdge(r);
    }

    // Check for stuck messages and send diagnostics
    checkStuckMessages();

    addCallback(std::chrono::steady_clock::now() + std::chrono::seconds(random(2, 8)), [this]() { keepalive(); });
  }

  void addConnection(std::shared_ptr<Connection> connection) {
    CHECK(!dead);
    auto ci = std::make_shared<ConnectionInfo>();
    ci->connection = connection;
    ci->lastReceive = std::chrono::steady_clock::now();
    floatingConnections.push_back(ci);
    // log.info("add new connection - %d floating connections\n", floatingConnections.size());
    connection->read([this, ci](Error* e, BufferHandle data) {
      if (!e) {
        addCallback(std::chrono::steady_clock::now(), [this, data = std::move(data), ci]() mutable {
          onreadtcp(std::move(data), ci);
        });
      } else {
        if (strcmp(e->what(), "Connection closed")) {
          log.error("Moodist Store tcp connection error: %s\n", e->what());
        }
        ci->connection->close();

        addCallback(std::chrono::steady_clock::now(), [this, ci]() mutable {
          std::lock_guard l(mutex);
          if (ci->sourceRank != -1) {
            auto& pi = getEdge(ci->sourceRank);
            reaptcpconnections(pi);
            if (!pi.addresses.empty()) {
              addCallback(
                  std::chrono::steady_clock::now() + std::chrono::seconds(1),
                  [this, sourceRank = ci->sourceRank] { connectEdge(sourceRank); });
            }
          }
        });
      }
    });
  }

  void connect() {
    if (connectAcked) {
      return;
    }

    std::lock_guard l(mutex);

    auto t = [this](int port) {
      if (connectAcked) {
        return;
      }
      udps.at(random<size_t>(0, udps.size() - 1))->resolve(hostname, port, [this](void* addr, size_t addrlen) {
        Vector<char> addrbuf;
        addrbuf.resize(addrlen);
        std::memcpy(addrbuf.data(), addr, addrlen);

        addCallback(std::chrono::steady_clock::now(), [this, addrbuf] {
          std::lock_guard l(mutex);

          for (size_t i = 0; i != udps.size(); ++i) {
            auto now = std::chrono::steady_clock::now();
            auto t = std::min(
                std::max(prevConnectTime + std::chrono::milliseconds(250), now), now + std::chrono::seconds(2));
            prevConnectTime = t;
            addCallback(t, [this, i, addrbuf]() {
              if (connectAcked) {
                return;
              }

              std::lock_guard l(mutex);

              Vector<std::pair<uint32_t, std::string>> tcpaddresses;
              for (auto& v : unackedAddresses) {
                tcpaddresses.push_back(v);
              }
              if (tcpaddresses.size() > 8) {
                std::ranges::shuffle(tcpaddresses, getRng());
                tcpaddresses.resize(8);
              }

              auto buf =
                  serializeToBuffer(signatureConnect, storekey, rank, myId, context.getNetworkKey(), tcpaddresses);

              // log.info("Sending udp connect to %s\n", Socket::ipAndPort(addrbuf.data(), addrbuf.size()));

              ::sendto(
                  udps[random<size_t>(0, udps.size() - 1)]->nativeFd(), buf->data(), buf->size(), MSG_NOSIGNAL,
                  (sockaddr*)addrbuf.data(), addrbuf.size());
            });
          }
        });
      });
    };

    t(port);
    for (int i : range(16)) {
      addCallback(std::chrono::steady_clock::now() + std::chrono::seconds(i), [this, i = 57360 + i, t]() { t(i); });
    }

    auto retryDelay = std::chrono::seconds(std::min(1 + connectCounter, 30));
    ++connectCounter;

    addCallback(std::chrono::steady_clock::now() + retryDelay, [this]() { connect(); });
  }

  uint32_t stringHash(std::string_view s) {
    uint32_t r = 524287u;
    for (uint8_t x : s) {
      r = r * 42643801u + x;
    }
    return (r % 2147483647u);
  }

  uint32_t storeRank(std::string_view key) {
    return stringHash(key) % worldSize;
  }

  std::atomic_uint32_t nextId = 1;

  struct Wait {
    bool b = false;
    std::vector<uint8_t> data;
    std::atomic_uint32_t futex = 0;
    bool error = false;
    std::string errorMessage;

    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point end;

    // Diagnostic info for timeout messages
    StoreImpl* store = nullptr;
    uint32_t targetRank = -1;
    uint32_t firstHopEdge = -1;

    SpinMutex mutex;

    std::optional<std::string> wait() {
      auto now = std::chrono::steady_clock::now();

      while (true) {
        futexWait(&futex, 0, std::min((std::chrono::steady_clock::duration)std::chrono::seconds(1), end - now));
        if (futex) {
          break;
        }
        now = std::chrono::steady_clock::now();
        if (now >= end) {
          std::lock_guard l(mutex);
          if (!error) {
            std::string msg = fmt::sprintf("timed out after %g seconds", seconds(end - start));
            if (store) {
              msg += store->getConnectionDiagnostics(targetRank, firstHopEdge);
            }
            return msg;
          }
          break;
        }
      }
      std::lock_guard l(mutex);
      if (error) {
        return errorMessage;
      }
      return {};
    }
  };

  HashMap<uint32_t, std::shared_ptr<Wait>> waits;

  HashMap<std::string, std::vector<uint8_t>> store;
  HashMap<std::string, Vector<Function<void(const std::vector<uint8_t>&)>>> storeWaiters;

  std::string getConnectionDiagnostics(uint32_t targetRank, uint32_t firstHopEdge) {
    std::lock_guard l(mutex);
    std::string result;

    // Check if store was destroyed
    if (destroyed) {
      if (destroySourceRank) {
        result += fmt::sprintf("\n  Store was destroyed by rank %d", *destroySourceRank);
      } else {
        result += "\n  Store was destroyed";
      }
      return result;
    }

    // Check if we've connected to rank 0 (for non-rank-0 processes)
    if (rank != 0 && !connectAcked) {
      result += "\n  Never received acknowledgment from rank 0 - rank 0 may not have started";
    }

    // On rank 0, report which ranks haven't connected via UDP
    if (rank == 0) {
      Vector<uint32_t> missingRanks;
      for (uint32_t r = 0; r < worldSize; ++r) {
        if (r != rank && !peerInfos[r].has_value()) {
          missingRanks.push_back(r);
        }
      }
      if (!missingRanks.empty()) {
        result += fmt::sprintf("\n  Rank(s) never connected: %s", fmt::to_string(fmt::join(missingRanks, ", ")));
        if (missingRanks.size() == worldSize - 1) {
          result += " (no other ranks have connected - did they start?)";
        }
      }
    }

    // Check connection to first hop edge
    if (firstHopEdge != (uint32_t)-1 && firstHopEdge != rank) {
      auto edgeIt = edgePeers.find(firstHopEdge);
      if (edgeIt != edgePeers.end()) {
        auto& pi = *edgeIt->second;
        if (pi.destroyed) {
          result += fmt::sprintf("\n  Rank %d (first hop to target rank %d) has exited", firstHopEdge, targetRank);
        } else if (pi.tcpconnections.empty()) {
          result += fmt::sprintf("\n  No TCP connection to rank %d (first hop to target rank %d)", firstHopEdge, targetRank);
          if (pi.addresses.empty()) {
            result += " - no TCP addresses received";
          }
        } else {
          // Connection exists but no response
          result += fmt::sprintf("\n  TCP connected to rank %d but no response for target rank %d",
                                 firstHopEdge, targetRank);
          size_t pendingMsgs = pi.outgoingQueue.size();
          if (pendingMsgs > 0) {
            result += fmt::sprintf(" (%zu messages pending)", pendingMsgs);
          }
        }
      } else {
        result += fmt::sprintf("\n  No edge info for rank %d - address not received from rank 0", firstHopEdge);
      }
    }

    // On rank 0, if target rank is different from first hop, check if we know about the target
    if (rank == 0 && targetRank != firstHopEdge && targetRank != rank && targetRank != (uint32_t)-1) {
      if (!peerInfos[targetRank].has_value()) {
        result += fmt::sprintf("\n  Target rank %d never connected", targetRank);
      } else if (peerInfos[targetRank]->destroyed) {
        result += fmt::sprintf("\n  Target rank %d has exited", targetRank);
      }
    }

    // Include any received diagnostics from intermediate ranks
    if (!receivedDiagnostics.empty()) {
      for (const auto& diag : receivedDiagnostics) {
        result += fmt::sprintf("\n  Rank %d reports: %s", diag.reportingRank, diag.reason);
      }
    }

    return result;
  }

  void sendMessage(uint32_t edge, uint32_t seq, BufferHandle buffer, PeerInfo& pi, uint32_t diagnosticSource) {
    if (edge == rank) {
      processMessage(std::move(buffer), edge);
      return;
    }

    // Ensure diagnostic source info is sent before the actual message
    ensureDiagnosticSourceSent(edge, diagnosticSource, pi);

    pi.outgoingQueue.emplace_back();
    auto& v = pi.outgoingQueue.back();
    v.seq = seq;
    v.buffer = SharedBufferHandle(buffer.release());
    v.queueTime = std::chrono::steady_clock::now();
    v.diagnosticSource = diagnosticSource;

    if (!pi.tcpconnections.empty()) {
      pi.tcpconnections.at(0)->connection->write(v.buffer, nullptr);
    }
  }

  // Send diagnostic source info to an edge if not already sent
  // Must be called before sending a message with the given diagnosticSource to this edge
  void ensureDiagnosticSourceSent(uint32_t edge, uint32_t diagnosticSource, PeerInfo& pi) {
    if (edge == rank) {
      return;  // No need to inform ourselves
    }
    if (pi.sentDiagnosticSourceFor.contains(diagnosticSource)) {
      return;  // Already sent
    }
    pi.sentDiagnosticSourceFor.insert(diagnosticSource);

    // Get the UDP addresses for this diagnostic source
    std::vector<std::string> addrs;
    if (diagnosticSource == rank) {
      // We are the diagnostic source - use our own addresses
      addrs = std::vector<std::string>(myUdpAddresses.begin(), myUdpAddresses.end());
    } else {
      // Forward the addresses we received
      auto it = diagnosticSources.find(diagnosticSource);
      if (it != diagnosticSources.end()) {
        addrs = std::vector<std::string>(it->second.addrStrings.begin(), it->second.addrStrings.end());
      }
    }

    if (addrs.empty()) {
      return;  // No addresses to send
    }

    // Send messageDiagnosticSource
    uint32_t seq = pi.outgoingSeq++;
    // diagnosticSource field in the message indicates whose UDP addresses these are
    // This will recurse into sendMessage, but sentDiagnosticSourceFor guard prevents infinite recursion
    sendMessage(
        edge, seq,
        serializeToBuffer(signatureMessage, worldSize, seq, edge, 0, rank, diagnosticSource, messageDiagnosticSource, addrs),
        pi, diagnosticSource);
  }

  void internalStoreSetValue(std::string key, std::vector<uint8_t> value) {
    auto& ref = store[key] = std::move(value);
    auto i = storeWaiters.find(key);
    if (i != storeWaiters.end()) {
      auto vec = std::move(i->second);
      storeWaiters.erase(i);
      for (auto& x : vec) {
        std::move(x)(ref);
      }
      CHECK(&store[key] == &ref);
    }
  }

  template<typename Callback>
  void internalStoreGetValue(std::string key, Callback&& callback) {
    auto i = store.find(key);
    if (i != store.end()) {
      callback(i->second);
    } else {
      storeWaiters[key].push_back(
          [callback = std::forward<Callback>(callback)](const std::vector<uint8_t>& data) { callback(data); });
    }
  }

  void destroyWait(std::shared_ptr<Wait>& w) {
    std::lock_guard l(w->mutex);
    w->error = true;
    if (destroySourceRank && destroySourceRank != rank) {
      w->errorMessage = fmt::sprintf("store was destroyed on rank %d", *destroySourceRank);
    } else {
      w->errorMessage = fmt::sprintf("store was destroyed");
    }
    w->futex = 1;
    futexWakeAll(&w->futex);
  }

  template<typename... Args>
  std::shared_ptr<Wait> doWaitOp(
      std::chrono::steady_clock::duration timeout, uint8_t messageType, std::string_view key, const Args&... args) {
    uint32_t id = nextId.fetch_add(1);
    uint32_t r = storeRank(key);
    std::unique_lock l(mutex);
    uint32_t edge = firsthop[r];
    CHECK(edge != worldSize);
    CHECK(edge != -1);
    auto& pi = getEdge(edge);
    uint32_t seq = pi.outgoingSeq++;

    auto w = std::make_shared<Wait>();
    // Set diagnostic info for better timeout error messages
    w->store = this;
    w->targetRank = r;
    w->firstHopEdge = edge;

    if (destroyed) {
      destroyWait(w);
    } else {
      waits[id] = w;

      // diagnosticSource = rank (we are the originator)
      sendMessage(
          edge, seq, serializeToBuffer(signatureMessage, worldSize, seq, r, id, rank, rank, messageType, key, args...), pi, rank);
    }
    l.unlock();

    w->start = std::chrono::steady_clock::now();
    w->end = w->start + timeout;
    return w;
  }

  void set(std::chrono::steady_clock::duration timeout, std::string_view key, const std::vector<uint8_t>& value) {
    auto w = doWaitOp(timeout, messageSet, key, value);
    auto error = w->wait();
    if (error) {
      throw std::runtime_error(fmt::sprintf("Moodist Store set(%s): %s", key, *error));
    }
  }

  std::vector<uint8_t> get(std::chrono::steady_clock::duration timeout, std::string_view key) {
    auto w = doWaitOp(timeout, messageGet, key);
    auto error = w->wait();
    if (error) {
      throw std::runtime_error(fmt::sprintf("Moodist Store get(%s): %s", key, *error));
    }
    return std::move(w->data);
  }
  bool check(std::chrono::steady_clock::duration timeout, std::string_view key) {
    auto w = doWaitOp(timeout, messageCheck, key);
    auto error = w->wait();
    if (error) {
      throw std::runtime_error(fmt::sprintf("Moodist Store get(%s): %s", key, *error));
    }
    return w->b;
  }
  bool check(std::chrono::steady_clock::duration timeout, const std::vector<std::string>& keys) {
    Vector<std::shared_ptr<Wait>> v;
    for (auto& k : keys) {
      v.push_back(doWaitOp(timeout, messageCheck, k));
    }
    bool r = true;
    for (auto& w : v) {
      auto error = w->wait();
      if (error) {
        throw std::runtime_error(
            fmt::sprintf("Moodist Store check(%s): %s", fmt::to_string(fmt::join(keys, ", ")), *error));
      }
      r &= w->b;
    }
    return r;
  }

  void wait(std::chrono::steady_clock::duration timeout, const std::vector<std::string>& keys) {
    Vector<std::shared_ptr<Wait>> v;
    for (auto& k : keys) {
      v.push_back(doWaitOp(timeout, messageWait, k));
    }
    bool r = true;
    for (auto& w : v) {
      auto error = w->wait();
      if (error) {
        throw std::runtime_error(
            fmt::sprintf("Moodist Store wait(%s): %s", fmt::to_string(fmt::join(keys, ", ")), *error));
      }
    }
  }
};

namespace {

struct Dtor {
  ~Dtor() {
    auto now = std::chrono::steady_clock::now();
    auto end = now;
    std::unique_lock l(*activeStoresMutex);
    for (auto* x : *activeStores) {
      x->destroy();
      end = std::max(end, x->destroyTime + std::chrono::seconds(2));
    }
    l.unlock();
    while (now < end) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      l.lock();
      if (activeStores->empty()) {
        l.unlock();
        break;
      }
      l.unlock();
      now = std::chrono::steady_clock::now();
    }
  }
} dtor;

} // namespace

TcpStore::TcpStore(StoreImpl* impl) : impl(impl) {
  ++impl->refcount;
}

TcpStore::TcpStore(
    std::string hostname, int port, std::string key, int worldSize, int rank,
    std::chrono::steady_clock::duration timeout) {
  impl = new StoreImpl(hostname, port, key, worldSize, rank);
  ++impl->refcount;
  timeout_ = std::chrono::ceil<std::chrono::milliseconds>(timeout);
}

TcpStore::~TcpStore() {
  if (--impl->refcount == 0) {
    impl->destroy();
  }
}

void TcpStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  impl->set(timeout_, key, value);
}

std::vector<uint8_t> TcpStore::get(const std::string& key) {
  return impl->get(timeout_, key);
}

int64_t TcpStore::add(const std::string& key, int64_t value) {
  throw std::runtime_error("Moodist Store add method is not implemented");
  return 0;
}

bool TcpStore::deleteKey(const std::string& key) {
  throw std::runtime_error("Moodist Store deleteKey method is not implemented");
}

bool TcpStore::check(const std::vector<std::string>& keys) {
  return impl->check(timeout_, keys);
}

int64_t TcpStore::getNumKeys() {
  throw std::runtime_error("Moodist Store getNumKeys method is not implemented");
}

void TcpStore::wait(const std::vector<std::string>& keys) {
  impl->wait(timeout_, keys);
}

void TcpStore::wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout) {
  impl->wait(timeout, keys);
}

} // namespace moodist
