// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "socket.h"

#include "async.h"
#include "function.h"
#include "hash_map.h"
#include "logging.h"
#include "shared_ptr.h"
#include "vector.h"

#include "fmt/printf.h"

#include <chrono>
#include <condition_variable>
#include <ifaddrs.h>
#include <limits.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace moodist {

// Global scheduler instance - used by socket.cc and tcpdev.cc
async::Scheduler& scheduler = Global(1);

namespace poll {
void add(SharedPtr<SocketImpl> impl);
}

struct ResolveHandle {
  std::string address;
  int port;
  addrinfo* result = nullptr;
  Function<void(Error*, addrinfo*)> callback;
  std::chrono::steady_clock::time_point startTime;

  ResolveHandle() = default;
  ResolveHandle(const ResolveHandle&) = delete;
  ResolveHandle& operator=(const ResolveHandle&) = delete;
  ~ResolveHandle() {
    if (result) {
      freeaddrinfo(result);
    }
  }
};

// Cached DNS entry - stores copies of sockaddr data
// We cache both successful resolutions and errors to avoid spamming DNS
struct DnsCacheEntry {
  struct Address {
    int family;
    int socktype;
    int protocol;
    socklen_t addrlen;
    sockaddr_storage addr;
  };
  Vector<Address> addresses;
  std::chrono::steady_clock::time_point expiry;
  int errorCode = 0; // 0 = success, non-zero = getaddrinfo error
};

// Helper to set port in a sockaddr_storage
inline void setPort(sockaddr_storage& addr, uint16_t port) {
  if (addr.ss_family == AF_INET) {
    reinterpret_cast<sockaddr_in&>(addr).sin_port = htons(port);
  } else if (addr.ss_family == AF_INET6) {
    reinterpret_cast<sockaddr_in6&>(addr).sin6_port = htons(port);
  }
}

// Builds addrinfo chain from cached addresses and invokes callback
static void invokeResolveCallback(
    const std::shared_ptr<ResolveHandle>& h, const Vector<DnsCacheEntry::Address>& addresses, int errorCode) {
  if (errorCode != 0) {
    Error e(errorCode == EAI_SYSTEM ? "system error" : gai_strerror(errorCode));
    h->callback(&e, nullptr);
    return;
  }
  Vector<addrinfo> infos(addresses.size());
  for (size_t i : indices(addresses)) {
    auto& addr = addresses[i];
    auto& info = infos[i];
    info.ai_flags = 0;
    info.ai_family = addr.family;
    info.ai_socktype = addr.socktype;
    info.ai_protocol = addr.protocol;
    info.ai_addrlen = addr.addrlen;
    info.ai_addr = reinterpret_cast<sockaddr*>(const_cast<sockaddr_storage*>(&addr.addr));
    info.ai_canonname = nullptr;
    info.ai_next = (i + 1 < addresses.size()) ? &infos[i + 1] : nullptr;
  }
  h->callback(nullptr, infos.empty() ? nullptr : &infos[0]);
}

// DNS resolver with lazy thread pool, caching, and queue limits
struct DnsResolver {
  static constexpr size_t kMaxWorkers = 4;
  static constexpr size_t kMaxQueueSize = 24;
  static constexpr auto kIdleTimeout = std::chrono::seconds(5);
  static constexpr auto kCacheTtl = std::chrono::seconds(10);

  std::mutex mutex;
  std::condition_variable cv;
  Vector<std::string> queue;                                           // Keys waiting to be resolved
  HashMap<std::string, Vector<std::weak_ptr<ResolveHandle>>> inFlight; // Key -> waiting handles
  HashMap<std::string, DnsCacheEntry> cache;
  std::atomic<size_t> numWorkers{0};
  size_t idleWorkers = 0; // protected by mutex
  bool shutdown = false;  // protected by mutex
  std::atomic<size_t> nextWorkerId{0};

  DnsResolver() = default;

  ~DnsResolver() {
    {
      std::lock_guard lock(mutex);
      shutdown = true;
    }
    cv.notify_all();
    // Wait for all workers to exit
    while (numWorkers.load() > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Returns true if found in cache and callback was invoked/scheduled
  // For async=true, schedules callback via scheduler. For async=false, invokes directly.
  bool tryCache(const std::string& key, const std::shared_ptr<ResolveHandle>& h, bool async) {
    Vector<DnsCacheEntry::Address> addresses;
    int errorCode;
    int port = h->port;
    {
      std::lock_guard lock(mutex);
      auto it = cache.find(key);
      if (it == cache.end()) {
        return false;
      }
      auto& entry = it->second;
      if (std::chrono::steady_clock::now() >= entry.expiry) {
        cache.erase(it);
        return false;
      }
      // Copy the cached data so we can release the lock before invoking callback
      addresses = entry.addresses;
      errorCode = entry.errorCode;
    }

    // Set the port in all cached addresses
    for (auto& addr : addresses) {
      setPort(addr.addr, port);
    }

    if (async) {
      // Schedule callback asynchronously to avoid deadlock (caller may hold mutexes)
      std::weak_ptr<ResolveHandle> wh = h;
      scheduler.run([wh, addresses = std::move(addresses), errorCode]() {
        if (auto h = wh.lock()) {
          invokeResolveCallback(h, addresses, errorCode);
        }
      });
    } else {
      invokeResolveCallback(h, addresses, errorCode);
    }
    return true;
  }

  void addToCache(const std::string& key, addrinfo* result, int errorCode) {
    // Cache both successful resolutions and errors to avoid spamming DNS
    std::lock_guard lock(mutex);
    DnsCacheEntry entry;
    entry.expiry = std::chrono::steady_clock::now() + kCacheTtl;
    entry.errorCode = errorCode;
    if (errorCode == 0 && result) {
      for (auto* i = result; i; i = i->ai_next) {
        if (i->ai_addrlen <= sizeof(sockaddr_storage)) {
          DnsCacheEntry::Address addr;
          addr.family = i->ai_family;
          addr.socktype = i->ai_socktype;
          addr.protocol = i->ai_protocol;
          addr.addrlen = i->ai_addrlen;
          std::memcpy(&addr.addr, i->ai_addr, i->ai_addrlen);
          entry.addresses.push_back(std::move(addr));
        }
      }
    }
    cache[key] = std::move(entry);
  }

  // Enqueues a request. Coalesces with in-flight requests for the same key.
  // If queue is full, randomly evicts an existing element.
  void enqueue(const std::string& key, std::weak_ptr<ResolveHandle> wh) {
    bool needWorker = false;
    {
      std::lock_guard lock(mutex);
      if (shutdown) {
        return;
      }
      // Check if this key is already being resolved
      auto it = inFlight.find(key);
      if (it != inFlight.end()) {
        // Coalesce: add to existing waiters
        it->second.push_back(std::move(wh));
        return;
      }
      // New key - add to inFlight and queue
      if (queue.size() >= kMaxQueueSize) {
        // Random eviction: drop a random key and its waiters
        size_t idx = random<size_t>(0, queue.size() - 1);
        inFlight.erase(queue[idx]);
        queue.erase(queue.begin() + idx);
      }
      inFlight[key].push_back(std::move(wh));
      queue.push_back(key);
      if (idleWorkers == 0 && numWorkers.load() < kMaxWorkers) {
        needWorker = true;
        ++numWorkers;
      }
    }
    if (needWorker) {
      size_t id = nextWorkerId++;
      std::thread([this, id]() {
        workerLoop(id);
      }).detach();
    }
    cv.notify_one();
  }

  void workerLoop(size_t id) {
    pthread_setname_np(pthread_self(), fmt::sprintf("moo/dns-%zu", id).c_str());

    while (true) {
      std::string key;
      std::string address;
      int port = 0;
      {
        std::unique_lock lock(mutex);
        ++idleWorkers;
        bool gotWork = cv.wait_for(lock, kIdleTimeout, [this] {
          return shutdown || !queue.empty();
        });
        --idleWorkers;

        if (shutdown) {
          --numWorkers;
          return;
        }
        if (!gotWork) {
          // Timed out with no work - exit this thread
          log.debug("[DNS] worker %zu exiting after idle timeout", id);
          --numWorkers;
          return;
        }
        key = std::move(queue.front());
        queue.pop_front();

        // Get address/port from one of the waiting handles
        auto it = inFlight.find(key);
        if (it == inFlight.end() || it->second.empty()) {
          // All waiters cancelled - skip this key
          continue;
        }
        // Find a valid handle to get address/port
        bool foundHandle = false;
        for (auto& wh : it->second) {
          if (auto h = wh.lock()) {
            address = h->address;
            port = h->port;
            foundHandle = true;
            break;
          }
        }
        if (!foundHandle) {
          // All handles cancelled
          inFlight.erase(it);
          continue;
        }
      }

      auto workerStartTime = std::chrono::steady_clock::now();
      log.debug("[DNS] worker %zu started for %s:%d", id, address.c_str(), port);

      addrinfo* result = nullptr;
      int r = getaddrinfo(address.c_str(), std::to_string(port).c_str(), nullptr, &result);
      int savedErrno = errno;

      auto resolveMs =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - workerStartTime)
              .count();
      log.debug("[DNS] getaddrinfo(%s:%d) returned %d in %ldms", address.c_str(), port, r, resolveMs);
      if (r != 0) {
        log.info("[DNS] resolution failed for %s: %s", address.c_str(),
            r == EAI_SYSTEM ? std::strerror(savedErrno) : gai_strerror(r));
      }

      // Add to cache
      addToCache(key, result, r);

      // Free result now - callbacks will use cached data
      if (result) {
        freeaddrinfo(result);
      }

      // Get all waiters and notify them
      Vector<std::weak_ptr<ResolveHandle>> waiters;
      Vector<DnsCacheEntry::Address> cachedAddresses;
      {
        std::lock_guard lock(mutex);
        auto it = inFlight.find(key);
        if (it != inFlight.end()) {
          waiters = std::move(it->second);
          inFlight.erase(it);
        }
        // Get cached addresses for successful resolutions
        if (r == 0) {
          auto cacheIt = cache.find(key);
          if (cacheIt != cache.end()) {
            cachedAddresses = cacheIt->second.addresses;
          }
        }
      }

      // Schedule callbacks for all waiters
      for (auto& wh : waiters) {
        // Each waiter may have a different port - copy addresses and set port
        auto h = wh.lock();
        if (!h) {
          continue;
        }
        int port = h->port;
        Vector<DnsCacheEntry::Address> addrCopy = cachedAddresses;
        for (auto& addr : addrCopy) {
          setPort(addr.addr, port);
        }
        scheduler.run([wh, r, addrCopy = std::move(addrCopy)]() {
          auto h = wh.lock();
          if (!h) {
            return;
          }
          auto totalMs =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - h->startTime)
                  .count();
          log.debug("[DNS] invoking callback for %s:%d (total %ldms)", h->address.c_str(), h->port, totalMs);
          invokeResolveCallback(h, addrCopy, r);
        });
      }
    }
  }
};

DnsResolver& getDnsResolver() {
  // Intentionally leaked to avoid static destruction order issues
  static DnsResolver* resolver = new DnsResolver();
  return *resolver;
}

template<typename F>
std::shared_ptr<ResolveHandle> resolveIpAddress(std::string_view address, int port, bool asynchronous, F&& callback) {
  auto h = std::make_shared<ResolveHandle>();
  h->address = address;
  h->port = port;
  h->callback = std::move(callback);
  h->startTime = std::chrono::steady_clock::now();

  std::string key = h->address;
  auto& resolver = getDnsResolver();

  if (asynchronous) {
    // Check cache first
    if (resolver.tryCache(key, h, true)) {
      log.debug("[DNS] cache hit for %s:%d", h->address.c_str(), port);
      return h;
    }
    log.debug("[DNS] resolveIpAddress(%s:%d) async started", h->address.c_str(), port);
    resolver.enqueue(key, h);
  } else {
    // Check cache first for sync too
    if (resolver.tryCache(key, h, false)) {
      log.debug("[DNS] cache hit for %s:%d (sync)", h->address.c_str(), port);
      return h;
    }
    log.debug("[DNS] resolveIpAddress(%s:%d) sync started", h->address.c_str(), port);
    int r = getaddrinfo(h->address.c_str(), std::to_string(port).c_str(), nullptr, &h->result);
    int savedErrno = errno;
    auto resolveMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - h->startTime).count();
    log.debug("[DNS] getaddrinfo(%s:%d) returned %d in %ldms", h->address.c_str(), port, r, resolveMs);
    if (r != 0) {
      log.info("[DNS] resolution failed for %s: %s", h->address.c_str(),
          r == EAI_SYSTEM ? std::strerror(savedErrno) : gai_strerror(r));
    }
    // Cache sync results too
    resolver.addToCache(key, h->result, r);
    if (r == 0) {
      h->callback(nullptr, h->result);
    } else {
      std::string str = r == EAI_SYSTEM ? std::strerror(savedErrno) : gai_strerror(r);
      Error e(std::move(str));
      h->callback(&e, nullptr);
    }
  }
  return h;
}

std::pair<std::string_view, int> decodeIpAddress(std::string_view address) {
  std::string_view hostname = address;
  int port = 0;
  auto bpos = address.find('[');
  if (bpos != std::string_view::npos) {
    auto bepos = address.find(']', bpos);
    if (bepos != std::string_view::npos) {
      hostname = address.substr(bpos + 1, bepos - (bpos + 1));
      address = address.substr(bepos + 1);
    }
  }
  auto cpos = address.find(':');
  if (cpos != std::string_view::npos) {
    if (hostname == address) {
      hostname = address.substr(0, cpos);
    }
    ++cpos;
    while (cpos != address.size()) {
      char c = address[cpos];
      if (c < '0' || c > '9') {
        break;
      }
      port *= 10;
      port += c - '0';
      ++cpos;
    }
  }
  return {hostname, port};
}

uint32_t writeFdFlag = 0x413ffc3f;

struct SocketImpl {
  std::atomic_size_t refcount = 0;
  int af = -1;
  int fd = -1;
  std::atomic_bool closed = false;
  uint32_t resolveCounter = 0;
  HashMap<uint32_t, std::shared_ptr<ResolveHandle>> resolveHandles;
  bool addedInPoll = false;
  bool isUdp = false;

  sockaddr_storage recvFromAddr;
  size_t recvFromAddrLen = 0;

  alignas(64) std::atomic_int writeTriggerCount = 0;
  SpinMutex writeQueueMutex;
  Vector<iovec> queuedWrites;
  Vector<std::pair<size_t, Function<void(Error*)>>> queuedWriteCallbacks;
  SpinMutex writeMutex;
  Vector<iovec> newWrites;
  Vector<std::pair<size_t, Function<void(Error*)>>> newWriteCallbacks;
  Vector<iovec> activeWrites;
  Vector<std::pair<size_t, Function<void(Error*)>>> activeWriteCallbacks;

  alignas(64) std::atomic_int readTriggerCount = 0;
  bool wantsRead = false;
  SpinMutex readMutex;
  Function<void(Error*)> onRead;
  std::vector<int> receivedFds;

  void close() {
    if (closed.exchange(true)) {
      return;
    }

    // Clean up resources under write lock
    {
      std::lock_guard l2(writeMutex);
      std::unique_lock l3(writeQueueMutex);
      // Close fd under writeMutex to synchronize with connect/listen callbacks
      if (fd != -1) {
        ::close(fd);
        fd = -1;
      }
      queuedWrites.clear();
      queuedWriteCallbacks.clear();
      resolveHandles.clear();
      CHECK(newWriteCallbacks.empty());
      CHECK(activeWriteCallbacks.empty());
    }

    // Clean up onRead and receivedFds under readMutex.
    // If we can't get the lock, schedule async cleanup.
    std::unique_lock l(readMutex, std::try_to_lock);
    if (l.owns_lock()) {
      onRead = nullptr;
      for (auto v : receivedFds) {
        ::close(v);
      }
      receivedFds.clear();
    } else {
      scheduler.run([me = share(this)] {
        std::unique_lock l(me->readMutex);
        me->onRead = nullptr;
        for (auto v : me->receivedFds) {
          ::close(v);
        }
        me->receivedFds.clear();
      });
    }
  }

  ~SocketImpl() {
    close();
  }

  void listen(std::string_view address) {
    std::unique_lock wl(writeMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    if (af == AF_UNIX) {
      sockaddr_un sa;
      memset(&sa, 0, sizeof(sa));
      sa.sun_family = AF_UNIX;
      sa.sun_path[0] = 0;
      std::string path = "moolib-" + std::string(address);
      size_t len = std::min(path.size(), sizeof(sa.sun_path) - 2);
      std::memcpy(&sa.sun_path[1], path.data(), len);
      if (::bind(fd, (const sockaddr*)&sa, sizeof(sa)) == -1) {
        throw std::system_error(errno, std::generic_category(), "bind");
      }
      if (::listen(fd, 50) == -1) {
        throw std::system_error(errno, std::generic_category(), "listen:");
      }
    } else if (af == AF_INET || af == AF_INET6) {
      wl.unlock();
      int port = 0;
      std::tie(address, port) = decodeIpAddress(address);
      auto h =
          resolveIpAddress(address, port, false, [this, address = std::string(address), port](Error* e, addrinfo* aix) {
            if (e) {
              throw *e;
            } else {
              std::unique_lock rl(readMutex);
              std::unique_lock wl(writeMutex);
              std::string errors;
              int tries = 0;
              for (auto* i = aix; i; i = i->ai_next) {
                if ((i->ai_family == AF_INET || i->ai_family == AF_INET6) && i->ai_socktype == SOCK_STREAM) {
                  ++tries;

                  if (fd == -1) {
                    fd = ::socket(i->ai_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
                    if (fd == -1) {
                      errors += fmt::sprintf("socket: %s", std::strerror(errno));
                      continue;
                    }
                  }

                  int reuseaddr = 1;
                  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuseaddr, sizeof(reuseaddr));
                  int reuseport = 1;
                  ::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &reuseport, sizeof(reuseport));

                  if (::bind(fd, i->ai_addr, i->ai_addrlen) == 0 && ::listen(fd, 50) == 0) {
                    poll::add(share(this));
                    return;
                  } else {
                    const char* se = std::strerror(errno);
                    ::close(fd);
                    fd = -1;

                    if (!errors.empty()) {
                      errors += ", ";
                    }
                    char buf[128];
                    memset(buf, 0, sizeof(buf));
                    int r = getnameinfo(i->ai_addr, i->ai_addrlen, buf, sizeof(buf) - 1, nullptr, 0, NI_NUMERICHOST);
                    const char* s = r ? gai_strerror(r) : buf;
                    errors += fmt::sprintf("%s (port %d): %s", s, port, se);
                  }
                }
              }
              if (tries == 0) {
                errors += "Name did not resolve to any usable addresses";
              }
              throw Error(std::move(errors));
            }
          });
      wl.lock();
    } else {
      throw Error("listen: unkown address family\n");
    }
  }

  bool bind(int port) {
    bool r = false;
    if (af == AF_INET) {
      sockaddr_in sa;
      std::memset(&sa, 0, sizeof(sa));
      sa.sin_family = af;
      sa.sin_port = htons(port);
      r = ::bind(fd, (sockaddr*)&sa, sizeof(sa)) == 0;
    } else if (af == AF_INET6) {
      sockaddr_in6 sa;
      std::memset(&sa, 0, sizeof(sa));
      sa.sin6_family = af;
      sa.sin6_port = htons(port);
      r = ::bind(fd, (sockaddr*)&sa, sizeof(sa)) == 0;
    }
    if (r) {
      poll::add(share(this));
    }
    return r;
  }

  void setTcpSockOpts() {
    int nodelay = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  }

  void accept(Function<void(Error*, Socket)> callback) {
    std::lock_guard l(readMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    auto sharedCallback = std::make_shared<Function<void(Error*, Socket)>>(std::move(callback));
    onRead = [this, sharedCallback](Error* error) mutable {
      while (true) {
        if (closed.load(std::memory_order_relaxed)) {
          wantsRead = false;
          return;
        }
        readTriggerCount.store(-0xffff, std::memory_order_relaxed);
        int r = ::accept4(fd, nullptr, 0, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (r == -1) {
          wantsRead = false;
          if (errno == EAGAIN) {
            return;
          }
          return;
        } else {
          Socket s;
          s.impl = makeShared<SocketImpl>();
          s.impl->af = af;
          s.impl->fd = r;
          s.impl->setTcpSockOpts();
          poll::add(s.impl);
          scheduler.run([s = std::move(s), sharedCallback]() mutable {
            (*sharedCallback)(nullptr, std::move(s));
          });
        }
      }
    };
  }

  void connect(std::string_view address, Function<void(Error*)> callback) {
    std::unique_lock wl(writeMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    if (af == AF_UNIX) {
      sockaddr_un sa;
      memset(&sa, 0, sizeof(sa));
      sa.sun_family = AF_UNIX;
      sa.sun_path[0] = 0;
      std::string path = "moolib-" + std::string(address);
      size_t len = std::min(path.size(), sizeof(sa.sun_path) - 2);
      std::memcpy(&sa.sun_path[1], path.data(), len);
      if (::connect(fd, (const sockaddr*)&sa, sizeof(sa)) && errno != EAGAIN) {
        Error e(std::strerror(errno));
        std::move(callback)(&e);
      } else {
        std::move(callback)(nullptr);
        std::unique_lock ql(writeQueueMutex);
        writeLoop(wl, ql);
      }
    } else {
      uint32_t resolveKey = resolveCounter++;
      resolveHandles[resolveKey] = nullptr;
      wl.unlock();
      int port = 0;
      std::tie(address, port) = decodeIpAddress(address);
      auto h = resolveIpAddress(address, port, true,
          [this, me = share(this), address = std::string(address), resolveKey](Error* e, addrinfo* aix) {
            std::unique_lock rl(readMutex);
            std::unique_lock wl(writeMutex);
            resolveHandles.erase(resolveKey);
            if (closed.load(std::memory_order_relaxed)) {
              return;
            }
            if (!e) {
              for (auto* i = aix; i; i = i->ai_next) {
                if ((i->ai_family == AF_INET || i->ai_family == AF_INET6) && i->ai_socktype == SOCK_STREAM) {
                  if (fd == -1) {
                    fd = ::socket(i->ai_family, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
                    if (fd == -1) {
                      continue;
                    }
                  }

                  if (::connect(fd, i->ai_addr, i->ai_addrlen) == 0 || errno == EAGAIN || errno == EINPROGRESS) {
                    rl.unlock();
                    setTcpSockOpts();
                    poll::add(share(this));
                    std::unique_lock ql(writeQueueMutex);
                    writeLoop(wl, ql);
                    return;
                  } else {
                    ::close(fd);
                    fd = -1;
                  }
                }
              }
            }
          });
      wl.lock();
      if (resolveHandles.contains(resolveKey)) {
        resolveHandles[resolveKey] = std::move(h);
      }
    }
  }

  void triggerRead() {
    scheduler.run([me = share(this), this] {
      std::unique_lock l(readMutex);
      if (closed.load(std::memory_order_relaxed)) {
        return;
      }
      while (true) {
        readTriggerCount.store(-0xffff, std::memory_order_relaxed);
        wantsRead = true;
        while (onRead && wantsRead) {
          wantsRead = false;
          onRead(nullptr);
          if (closed.load(std::memory_order_relaxed)) {
            return;
          }
        }
        int v = -0xffff;
        if (readTriggerCount.compare_exchange_strong(v, 0)) {
          break;
        }
      }
    });
  }

  void triggerWrite() {
    scheduler.run([me = share(this), this] {
      std::unique_lock wl(writeMutex, std::try_to_lock);
      if (wl.owns_lock()) {
        std::unique_lock ql(writeQueueMutex);
        writeLoop(wl, ql);
      }
    });
  }

  void writeLoop(std::unique_lock<SpinMutex>& wl, std::unique_lock<SpinMutex>& ql) {
    while (true) {
      writeTriggerCount.store(-0xffff, std::memory_order_relaxed);
      if (queuedWrites.empty()) {
        wl.unlock();
        ql.unlock();
        return;
      }
      activeWrites.clear();
      activeWriteCallbacks.clear();
      std::swap(activeWrites, queuedWrites);
      std::swap(activeWriteCallbacks, queuedWriteCallbacks);
      ql.unlock();
      bool canWrite = writevImpl(
          activeWrites.data(), activeWrites.size(), activeWriteCallbacks.data(), activeWriteCallbacks.size(), ql);
      if (!ql.owns_lock()) {
        ql.lock();
      }
      if (queuedWrites.empty()) {
        wl.unlock();
        ql.unlock();
        return;
      }
      if (!canWrite) {
        wl.unlock();
        int v = -0xffff;
        if (writeTriggerCount.compare_exchange_strong(v, 0)) {
          ql.unlock();
          return;
        }
        if (!wl.try_lock()) {
          ql.unlock();
          return;
        }
      }
    }
  }

  void setOnRead(Function<void(Error*)> callback) {
    std::unique_lock l(readMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    if (onRead) {
      throw Error("onRead callback is already set");
    }
    onRead = std::move(callback);
    l.unlock();
    triggerRead();
  }

  size_t readv(const iovec* vec, size_t veclen) {
    if (closed.load(std::memory_order_relaxed) || fd == -1) {
      wantsRead = false;
      return 0;
    }
    msghdr msg = {0};
    union {
      char buf[CMSG_SPACE(sizeof(int))];
      cmsghdr align;
    } u;
    msg.msg_control = u.buf;
    msg.msg_controllen = sizeof(u.buf);
    msg.msg_iov = (::iovec*)vec;
    msg.msg_iovlen = std::min(veclen, (size_t)IOV_MAX);
    readTriggerCount.store(-0xffff, std::memory_order_relaxed);
    if (isUdp) {
      msg.msg_name = &recvFromAddr;
      msg.msg_namelen = sizeof(recvFromAddr);
    }
    ssize_t r = ::recvmsg(fd, &msg, 0);
    wantsRead = true;
    if (r == -1) {
      int error = errno;
      if (error == EINTR) {
        return 0;
      }
      wantsRead = false;
      if (error == EAGAIN || error == EWOULDBLOCK || error == ENOTCONN) {
        return 0;
      }
      Error e(std::strerror(error));
      if (onRead) {
        onRead(&e);
      }
      return 0;
    } else {
      if (r == 0 && !isUdp) {
        wantsRead = false;
        Error e("Connection closed");
        if (onRead) {
          onRead(&e);
        }
        return 0;
      }
      if (msg.msg_controllen != 0) {
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
          int fd;
          std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
          receivedFds.push_back(fd);
        }
      }
      if (isUdp) {
        recvFromAddrLen = msg.msg_namelen;
      }
      return r;
    }
  }

  bool writevImpl(const iovec* vec, size_t veclen, std::pair<size_t, Function<void(Error*)>>* callbacks,
      size_t callbacksLen, std::unique_lock<SpinMutex>& ql) {
    if (closed.load(std::memory_order_relaxed)) {
      activeWrites.clear();
      activeWriteCallbacks.clear();
      return false;
    }

    msghdr msg;
    union {
      char buf[CMSG_SPACE(sizeof(int))];
      cmsghdr align;
    } u;

    msg.msg_name = nullptr;
    msg.msg_namelen = 0;
    msg.msg_control = nullptr;
    msg.msg_controllen = 0;
    msg.msg_flags = 0;

    static_assert(sizeof(iovec) == sizeof(::iovec));
    msg.msg_iov = (::iovec*)vec;
    msg.msg_iovlen = std::min(veclen, (size_t)IOV_MAX);
    if (af == AF_UNIX) {
      for (size_t i = 0; i != msg.msg_iovlen; ++i) {
        if (vec[i].iov_base == (void*)&writeFdFlag) {
          if (i == 0) {
            int fd = (int)(uintptr_t)vec[i + 1].iov_base;
            msg.msg_iovlen = 1;
            msg.msg_control = u.buf;
            msg.msg_controllen = sizeof(u.buf);
            cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
            cmsg->cmsg_level = SOL_SOCKET;
            cmsg->cmsg_type = SCM_RIGHTS;
            cmsg->cmsg_len = CMSG_LEN(sizeof(fd));
            std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
            break;
          } else {
            msg.msg_iovlen = i;
            break;
          }
        }
      }
    }
    size_t nbytes = 0;
    for (size_t i : range(veclen)) {
      nbytes += vec[i].iov_len;
    }
    writeTriggerCount.store(-0xffff, std::memory_order_relaxed);
    ssize_t r = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
    bool canWrite = true;
    if (r == -1) {
      int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK) {
        canWrite = false;
        r = 0;
      } else if (e == EINTR) {
        r = 0;
      }
    }
    if (r == -1) {
      canWrite = false;
      int e = errno;
      Error ee(std::strerror(e));
      std::move(callbacks[0].second)(&ee);
      activeWrites.clear();
      activeWriteCallbacks.clear();
      ql.lock();
      queuedWrites.clear();
      queuedWriteCallbacks.clear();
    } else {
      size_t writtenForCallback = r;
      while (writtenForCallback) {
        if (callbacksLen == 0) {
          throw Error("writev empty callback list");
        }
        if (writtenForCallback >= callbacks[0].first) {
          writtenForCallback -= callbacks[0].first;
          std::move(callbacks[0].second)(nullptr);
          ++callbacks;
          --callbacksLen;
        } else {
          break;
        }
      }
      size_t offset = 0;
      for (; offset != veclen; ++offset) {
        if (r < vec[offset].iov_len) {
          break;
        }
        r -= vec[offset].iov_len;
      }
      if (offset == veclen) {
        activeWrites.clear();
        activeWriteCallbacks.clear();
      } else {
        newWrites.clear();
        newWriteCallbacks.clear();
        if (vec == activeWrites.data() && veclen == activeWrites.size()) {
          activeWrites.erase(activeWrites.begin(), activeWrites.begin() + offset);
          if (r) {
            iovec& v = activeWrites[0];
            v.iov_base = (char*)v.iov_base + r;
            v.iov_len = v.iov_len - r;
          }
          std::swap(newWrites, activeWrites);
        } else {
          for (size_t i = offset; i != veclen; ++i) {
            if (i == offset) {
              iovec v;
              v.iov_base = (char*)vec[i].iov_base + r;
              v.iov_len = vec[i].iov_len - r;
              newWrites.push_back(v);
            } else {
              newWrites.push_back(vec[i]);
            }
          }
        }
        if (callbacks + callbacksLen == activeWriteCallbacks.data() + activeWriteCallbacks.size()) {
          activeWriteCallbacks.erase(
              activeWriteCallbacks.begin(), activeWriteCallbacks.begin() + (callbacks - activeWriteCallbacks.data()));
          if (writtenForCallback) {
            activeWriteCallbacks[0].first -= writtenForCallback;
          }
          std::swap(newWriteCallbacks, activeWriteCallbacks);
        } else {
          for (size_t i = 0; i != callbacksLen; ++i) {
            if (i == 0) {
              newWriteCallbacks.emplace_back(callbacks[i].first - writtenForCallback, std::move(callbacks[i].second));
            } else {
              newWriteCallbacks.push_back(std::move(callbacks[i]));
            }
          }
        }
        activeWrites.clear();
        activeWriteCallbacks.clear();
        ql.lock();
        if (!queuedWrites.empty()) {
          for (auto& v : queuedWrites) {
            newWrites.push_back(v);
          }
          for (auto& v : queuedWriteCallbacks) {
            newWriteCallbacks.push_back(std::move(v));
          }
          queuedWrites.clear();
          queuedWriteCallbacks.clear();
        }
        std::swap(queuedWrites, newWrites);
        std::swap(queuedWriteCallbacks, newWriteCallbacks);
      }
    }

    return canWrite;
  }

  void writev(const iovec* vec, size_t veclen, Function<void(Error*)> callback) {
    std::unique_lock ql(writeQueueMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    std::unique_lock wl(writeMutex, std::defer_lock);
    if (queuedWrites.empty()) {
      if (wl.try_lock() && fd != -1) {
        ql.unlock();
        size_t bytes = 0;
        for (size_t i = 0; i != veclen; ++i) {
          bytes += vec[i].iov_len;
        }
        std::pair<size_t, Function<void(Error*)>> p;
        p.first = bytes;
        p.second = std::move(callback);
        bool canWrite = writevImpl(vec, veclen, &p, 1, ql);
        if (!ql.owns_lock()) {
          ql.lock();
        }
        if (queuedWrites.empty()) {
          wl.unlock();
          ql.unlock();
          return;
        }
        if (!canWrite) {
          wl.unlock();
          int v = -0xffff;
          if (writeTriggerCount.compare_exchange_strong(v, 0)) {
            ql.unlock();
            return;
          }
          if (!wl.try_lock()) {
            ql.unlock();
            return;
          }
        }
        writeLoop(wl, ql);
        return;
      } else if (wl.owns_lock()) {
        wl.unlock();
      }
    }
    size_t bytes = 0;
    for (size_t i = 0; i != veclen; ++i) {
      queuedWrites.push_back(vec[i]);
      bytes += vec[i].iov_len;
    }
    queuedWriteCallbacks.emplace_back(bytes, std::move(callback));
  }

  void sendFd(int fd, Function<void(Error*)> callback) {
    std::array<iovec, 2> iovecs;
    iovecs[0].iov_base = &writeFdFlag;
    iovecs[0].iov_len = 4;
    iovecs[1].iov_base = (void*)(uintptr_t)fd;
    iovecs[1].iov_len = 0;
    writev(iovecs.data(), 2, std::move(callback));
  }

  int recvFd(CachedReader& reader) {
    uint32_t flag = 0;
    if (!reader.readCopy(&flag, sizeof(flag))) {
      return -1;
    }
    if (flag != writeFdFlag) {
      Error e("recvFd flag mismatch");
      if (onRead) {
        onRead(&e);
      }
      return -1;
    }
    if (receivedFds.empty()) {
      Error e("receivedFds is empty!");
      if (onRead) {
        onRead(&e);
      }
      return -1;
    }
    int fd = receivedFds.front();
    receivedFds.erase(receivedFds.begin());
    return fd;
  }

  static std::string ipAndPort(const sockaddr* addr, socklen_t addrlen) {
    char host[128];
    memset(host, 0, sizeof(host));
    char port[16];
    memset(port, 0, sizeof(port));
    int r = getnameinfo(addr, addrlen, host, sizeof(host) - 1, port, sizeof(port) - 1, NI_NUMERICHOST | NI_NUMERICSERV);
    if (r) {
      throw std::runtime_error(gai_strerror(r));
    }
    if (addr->sa_family == AF_INET) {
      return fmt::sprintf("%s:%s", host, port);
    } else if (addr->sa_family == AF_INET6) {
      return fmt::sprintf("[%s]:%s", host, port);
    } else {
      return "";
    }
  }

  std::vector<std::string> localAddresses() const {
    if (af == AF_INET || af == AF_INET6) {
      std::vector<std::string> r;
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getsockname(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category(), "getsockname");
      }
      bool isAnyAddr = false;
      int port = 0;
      if (addr.ss_family == AF_INET) {
        const sockaddr_in* sa = (const sockaddr_in*)&addr;
        port = sa->sin_port;
        if (sa->sin_addr.s_addr == INADDR_ANY) {
          isAnyAddr = true;
        }
      } else if (addr.ss_family == AF_INET6) {
        const sockaddr_in6* sa = (const sockaddr_in6*)&addr;
        port = sa->sin6_port;
        if (!memcmp(&sa->sin6_addr, &in6addr_any, sizeof(in6addr_any))) {
          isAnyAddr = true;
        }
      }
      if (isAnyAddr) {
        struct ifaddrs* list;
        if (::getifaddrs(&list) == 0) {
          for (; list; list = list->ifa_next) {
            try {
              if (list->ifa_addr && (list->ifa_flags & IFF_RUNNING) == IFF_RUNNING &&
                  list->ifa_addr->sa_family == addr.ss_family) {
                if (addr.ss_family == AF_INET) {
                  sockaddr_in sa;
                  std::memcpy(&sa, list->ifa_addr, sizeof(sa));
                  sa.sin_port = port;
                  r.push_back(ipAndPort((const sockaddr*)&sa, sizeof(sa)));
                } else if (addr.ss_family == AF_INET6) {
                  sockaddr_in6 sa;
                  std::memcpy(&sa, list->ifa_addr, sizeof(sa));
                  sa.sin6_port = port;
                  r.push_back(ipAndPort((const sockaddr*)&sa, sizeof(sa)));
                }
              }
            } catch (const std::exception& e) {
            }
          }
          ::freeifaddrs(list);
        }
      } else {
        r.push_back(ipAndPort((const sockaddr*)&addr, addrlen));
      }
      return r;
    } else {
      return {};
    }
  }

  std::string localAddress() const {
    if (af == AF_INET || af == AF_INET6) {
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getsockname(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category(), "getsockname");
      }
      return ipAndPort((const sockaddr*)&addr, addrlen);
    } else {
      return "";
    }
  }

  std::string remoteAddress() const {
    if (af == AF_INET || af == AF_INET6) {
      sockaddr_storage addr;
      socklen_t addrlen = sizeof(addr);
      if (::getpeername(fd, (sockaddr*)&addr, &addrlen)) {
        throw std::system_error(errno, std::generic_category(), "getpeername");
      }
      return ipAndPort((const sockaddr*)&addr, addrlen);
    } else {
      return "";
    }
  }

  void resolve(std::string_view address, int port, Function<void(void*, size_t)> callback) {
    std::unique_lock wl(writeMutex);
    if (closed.load(std::memory_order_relaxed)) {
      return;
    }
    uint32_t resolveKey = resolveCounter++;
    resolveHandles[resolveKey] = nullptr;
    wl.unlock();
    int socktype = isUdp ? SOCK_DGRAM : SOCK_STREAM;
    auto h = resolveIpAddress(address, port, true,
        [this, me = share(this), address = std::string(address), socktype, callback, resolveKey](
            Error* e, addrinfo* aix) {
          std::unique_lock rl(readMutex);
          std::unique_lock wl(writeMutex);
          resolveHandles.erase(resolveKey);
          if (closed.load(std::memory_order_relaxed)) {
            return;
          }
          if (!e) {
            for (auto* i = aix; i; i = i->ai_next) {
              if ((i->ai_family == AF_INET || i->ai_family == AF_INET6) && i->ai_socktype == socktype) {
                callback(i->ai_addr, i->ai_addrlen);
              }
            }
          }
        });
    wl.lock();
    if (resolveHandles.contains(resolveKey)) {
      resolveHandles[resolveKey] = std::move(h);
    }
  }
  void resolve(std::string_view address, Function<void(void*, size_t)> callback) {
    int port = 0;
    std::tie(address, port) = decodeIpAddress(address);
    resolve(address, port, std::move(callback));
  }
};

namespace poll {

struct PollThread {
  ~PollThread() {
    terminate = true;
    wake();
    if (thread.joinable()) {
      thread.join();
    }
  }
  std::atomic_bool terminate = false;
  std::atomic_bool terminateWhenEmpty = false;
  std::once_flag flag;
  std::thread thread;
  int fd = -1;
  int wakefd = -1;
  std::atomic_bool anyDead = false;
  std::mutex mutex;
  std::vector<SharedPtr<SocketImpl>> activeList;
  std::vector<SharedPtr<SocketImpl>> deadList;

  void wake() {
    if (wakefd != -1) {
      uint64_t val = 1;
      ::write(wakefd, &val, sizeof(val));
    }
  }

  void tryTerminate() {
    // Called under mutex - check if we should terminate
    if (terminateWhenEmpty && activeList.empty()) {
      terminate = true;
      wake();
    }
  }

  void entry() {
    std::array<epoll_event, 1024> events;
    int timeout = 250;

    while (true) {
      int n = epoll_wait(fd, events.data(), events.size(), timeout);
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        throw std::system_error(errno, std::generic_category(), "epoll_wait");
      }

      // Use zero timeout after full batch to quickly drain pending events.
      // This matters because we only clear deadList when n < events.size()
      // (to guarantee no more events for removed sockets), so if an eventfd
      // wake coincides with exactly 1024 events, we'd otherwise wait 250ms.
      timeout = ((size_t)n == events.size()) ? 0 : 250;

      for (int i = 0; i != n; ++i) {
        if (events[i].data.ptr == nullptr) {
          uint64_t val;
          ::read(wakefd, &val, sizeof(val));
          continue;
        }
        if (events[i].events & (EPOLLIN | EPOLLERR)) {
          SocketImpl* impl = (SocketImpl*)events[i].data.ptr;
          if (++impl->readTriggerCount == 1) {
            impl->triggerRead();
          }
        }
        if (events[i].events & EPOLLOUT) {
          SocketImpl* impl = (SocketImpl*)events[i].data.ptr;
          if (++impl->writeTriggerCount == 1) {
            impl->triggerWrite();
          }
        }
      }

      if ((size_t)n < events.size()) {
        if (anyDead.load(std::memory_order_relaxed)) {
          anyDead = false;
          std::lock_guard l(mutex);
          if (!deadList.empty()) {
          }
          deadList.clear();
        }
        if (terminate.load(std::memory_order_relaxed)) {
          break;
        }
      }
    }

    ::close(wakefd);
    ::close(fd);
  }

  void add(SharedPtr<SocketImpl> impl) {
    std::lock_guard l(mutex);
    if (terminate) {
      return;
    }
    std::call_once(flag, [&] {
      fd = epoll_create1(EPOLL_CLOEXEC);
      if (fd == -1) {
        throw std::system_error(errno, std::generic_category(), "epoll_create1");
      }
      wakefd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
      if (wakefd == -1) {
        throw std::system_error(errno, std::generic_category(), "eventfd");
      }
      epoll_event we;
      we.data.ptr = nullptr;
      we.events = EPOLLIN;
      if (epoll_ctl(fd, EPOLL_CTL_ADD, wakefd, &we)) {
        throw std::system_error(errno, std::generic_category(), "epoll_ctl wakefd");
      }
      thread = std::thread([&] {
        async::setCurrentThreadName("moo/socket epoll");
        entry();
      });
    });
    epoll_event e;
    e.data.ptr = &*impl;
    e.events = EPOLLIN | EPOLLOUT | EPOLLET;
    if (epoll_ctl(fd, EPOLL_CTL_ADD, impl->fd, &e)) {
      throw std::system_error(errno, std::generic_category(), "epoll_ctl");
    }
    impl->addedInPoll = true;
    activeList.push_back(std::move(impl));
  }
  void remove(SocketImpl* impl) {
    std::lock_guard l(mutex);
    impl->addedInPoll = false;
    epoll_event e;
    epoll_ctl(fd, EPOLL_CTL_DEL, impl->fd, &e);

    for (auto i = activeList.begin(); i != activeList.end(); ++i) {
      if (impl == &**i) {
        anyDead = true;
        deadList.push_back(std::move(*i));
        activeList.erase(i);
        wake();
        tryTerminate();
        return;
      }
    }
  }

  bool isAdded(SocketImpl* impl) {
    std::lock_guard l(mutex);
    return impl->addedInPoll;
  }
};
PollThread* pollThread = internalNew<PollThread>();

struct Dtor {
  ~Dtor() {
    std::lock_guard l(pollThread->mutex);
    pollThread->terminateWhenEmpty = true;
    pollThread->tryTerminate();
  }
} dtor;

void add(SharedPtr<SocketImpl> impl) {
  pollThread->add(std::move(impl));
}

void remove(SocketImpl* impl) {
  pollThread->remove(impl);
}

bool isAdded(SocketImpl* impl) {
  return pollThread->isAdded(impl);
}

} // namespace poll

Socket Socket::Unix() {
  Socket r;
  r.impl = makeShared<SocketImpl>();
  r.impl->af = AF_UNIX;
  r.impl->fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (r.impl->fd == -1) {
    throw std::system_error(errno, std::generic_category(), "socket");
  }
  poll::add(r.impl);
  return r;
}

Socket Socket::Tcp() {
  Socket r;
  r.impl = makeShared<SocketImpl>();
  r.impl->af = AF_INET;
  r.impl->fd = -1;
  return r;
}

Socket Socket::Udp(bool ipv6) {
  Socket r;
  r.impl = makeShared<SocketImpl>();
  r.impl->af = ipv6 ? AF_INET6 : AF_INET;
  r.impl->fd = ::socket(r.impl->af, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, IPPROTO_UDP);
  r.impl->isUdp = true;
  if (r.impl->fd == -1) {
    throw std::system_error(errno, std::generic_category(), "socket");
  }
  return r;
}

Socket::Socket() {}
Socket::Socket(Socket&& n) noexcept {
  std::swap(impl, n.impl);
}
Socket& Socket::operator=(Socket&& n) noexcept {
  std::swap(impl, n.impl);
  return *this;
}

Socket::~Socket() {
  close();
}

void Socket::close() {
  if (impl) {
    if (poll::isAdded(&*impl)) {
      poll::remove(&*impl);
    } else {
    }
    impl->close();
  }
}

bool Socket::closed() const {
  return impl ? (bool)impl->closed : true;
}

void Socket::writev(const iovec* vec, size_t veclen, Function<void(Error*)> callback) {
  impl->writev(vec, veclen, std::move(callback));
}

void Socket::listen(std::string_view address) {
  impl->listen(address);
}

bool Socket::bind(int port) {
  return impl->bind(port);
}

void Socket::accept(Function<void(Error*, Socket)> callback) {
  impl->accept(std::move(callback));
}

void Socket::connect(std::string_view address, Function<void(Error*)> callback) {
  impl->connect(address, std::move(callback));
}

void Socket::setOnRead(Function<void(Error*)> callback) {
  impl->setOnRead(std::move(callback));
}

FunctionPointer Socket::onReadFunction() const {
  return impl->onRead.getPointer();
}

size_t Socket::readv(const iovec* vec, size_t veclen) {
  return impl->readv(vec, veclen);
}

void Socket::sendFd(int fd, Function<void(Error*)> callback) {
  impl->sendFd(fd, std::move(callback));
}

int Socket::recvFd(CachedReader& reader) {
  return impl->recvFd(reader);
}

std::vector<std::string> Socket::localAddresses() const {
  return impl->localAddresses();
}

std::string Socket::localAddress() const {
  return impl->localAddress();
}

std::string Socket::remoteAddress() const {
  return impl->remoteAddress();
}

int Socket::nativeFd() const {
  return impl->fd;
}

void Socket::resolve(std::string_view address, Function<void(void*, size_t)> callback) {
  impl->resolve(address, std::move(callback));
}

void Socket::resolve(std::string_view address, int port, Function<void(void*, size_t)> callback) {
  impl->resolve(address, port, std::move(callback));
}

std::string Socket::ipAndPort(const void* addr, size_t addrlen) {
  return SocketImpl::ipAndPort((const sockaddr*)addr, (socklen_t)addrlen);
}

std::pair<const void*, size_t> Socket::recvFromAddr() {
  return {&impl->recvFromAddr, impl->recvFromAddrLen};
}

} // namespace moodist
