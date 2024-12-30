
#include "allocator.h"
#include "common.h"
#include "connection.h"
#include "hash_map.h"
#include "serialization.h"
#include "socket.h"
#include "synchronization.h"
#include <memory>

namespace moodist {

namespace allocator {

namespace {

constexpr uint64_t signature = 0xa2dda177c8e9a8b7;

constexpr uint32_t idRequestMap = 0x88b0e8fb;
constexpr uint32_t idResponseMap = 0xfbd193fc;
constexpr uint32_t idResponseError = 0xea39c870;

struct PeerMemoryImpl;

CUcontext cuContext = nullptr;

Socket listener;
std::mutex globalMutex;
HashMap<std::string, PeerMemoryImpl*> peerMemories;
std::atomic_uint32_t newConnectionCounter = 0;

struct PeerMemoryImpl : std::enable_shared_from_this<PeerMemoryImpl> {
  std::shared_ptr<Connection> connection;
  bool connected = false;

  std::shared_ptr<PeerMemoryImpl> selfRef;

  std::mutex mutex;
  std::mutex extendMutex;
  std::atomic_uint32_t recvFutex = 0;
  Vector<std::pair<uintptr_t, size_t>> mapResponses;
  Vector<std::string> errorResponses;

  uintptr_t localBaseAddress = 0;

  uintptr_t remoteBaseAddress = 0;
  uintptr_t remoteMappedSize = 0;

  void onRead(BufferHandle buffer, int fd) {
    std::lock_guard l(mutex);
    log.info("yey onread buffer of %d bytes\n", buffer->size());
    try {
      if (!connected) {
        uint64_t sig;
        std::string id;
        deserializeBuffer(buffer, sig, id);
        if (sig == signature) {
          std::lock_guard l(globalMutex);
          auto i = peerMemories.find(id);
          if (i != peerMemories.end()) {
            log.error("Allocator IPC got a duplicate connection to a peer (id %s)\n", id);
            connection->close();
            return;
          }
          log.info("(%s) Got connection to id %s\n", allocator::id(), id);
          connected = true;
          // keep this object alive forever
          selfRef = shared_from_this();
          peerMemories[id] = this;
          ++newConnectionCounter;
          futexWakeAll(&newConnectionCounter);
        } else {
          connection->close();
        }
      } else {
        uint32_t id;
        auto view = deserializeBufferPart(buffer, id);

        switch (id) {
        case idRequestMap: {
          uintptr_t offset;
          size_t length;
          size_t reservedSize;
          deserializeBuffer(view, offset, length, reservedSize);
          if (fd == -1) {
            log.error("Allocator IPC connection did not receive a file descriptor where one was expected\n");
            connection->close();
          } else {
            log.info("Allocator IPC got fd %d\n", fd);
            try {
              CHECK(cuContext);
              CHECK_CU(cuCtxSetCurrent(cuContext));
              if (!localBaseAddress) {
                constexpr size_t alignment = (size_t)1024 * 1024 * 1024 * 1024;
                CUdeviceptr base = 0;
                CHECK_CU(cuMemAddressReserve(&base, reservedSize, alignment, 0, 0));
                localBaseAddress = base;
                log.info("Reserved remote address range at %#x of length %#x\n", localBaseAddress, reservedSize);
              }
              log.info("Allocator IPC mapping remote offset %#x length %#x\n", offset, length);
              CUmemGenericAllocationHandle handle;
              CHECK_CU(cuMemImportFromShareableHandle(&handle, &fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
              CHECK_CU(cuMemMap(localBaseAddress + offset, length, 0, handle, 0));
              log.info("Map success!\n");
              connection->write(serializeToBuffer(idResponseMap, offset, length), nullptr);
            } catch (const std::exception& e) {
              connection->write(serializeToBuffer(idResponseError, std::string(e.what())), nullptr);
            }
          }
          break;
        }
        case idResponseMap: {
          uintptr_t offset;
          size_t length;
          deserializeBuffer(view, offset, length);
          log.info("Got map response for offset %#x length %#x\n", offset, length);
          mapResponses.emplace_back(offset, length);
          ++recvFutex;
          futexWakeAll(&recvFutex);
          break;
        }
        case idResponseError: {
          std::string message;
          deserializeBuffer(view, message);
          log.info("Got remote error: %s\n", message);
          errorResponses.push_back(message);
          ++recvFutex;
          futexWakeAll(&recvFutex);
          break;
        }
        default:
          log.error("Allocator IPC protocol error\n");
          connection->close();
        }
      }
    } catch (const SerializationError& e) {
      log.error("Allocator IPC connection error: %s\n", e.what());
      connection->close();
    }

    if (fd != -1) {
      ::close(fd);
    }
  }

  void extend(uintptr_t length, std::unique_lock<std::mutex>& l) {
    CHECK(l.owns_lock());
    if (length <= remoteMappedSize) {
      return;
    }
    std::lock_guard l2(extendMutex);
    log.info("extend to %d\n", length);
    auto handles = allocator::cuMemHandles();
    uintptr_t o = 0;
    size_t reservedSize = allocator::reserved().second;
    Vector<std::pair<uintptr_t, size_t>> requests;
    for (auto& [size, handle] : handles) {
      if (o >= remoteMappedSize) {
        log.info("Sending a map request for [%#x, %#x]\n", o, o + size);
        int fd = -1;
        CHECK_CU(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
        log.info("Exported mapping to fd %d\n", fd);
        CHECK(fd != -1);
        connection->writefd(serializeToBuffer(idRequestMap, o, size, reservedSize), fd);
        ::close(fd);

        requests.emplace_back(o, size);
      }
      o += size;
    }

    CHECK(o >= length);

    auto start = std::chrono::steady_clock::now();

    for (int s = 0; (s < 30 || std::chrono::steady_clock::now() - start < std::chrono::minutes(2)) &&
                    !connection->closed() && !requests.empty();
         ++s) {
      l.unlock();
      futexWait(&recvFutex, 0, std::chrono::seconds(1));
      l.lock();
      recvFutex = 0;
      for (auto& v : mapResponses) {
        auto i = std::find(requests.begin(), requests.end(), v);
        if (i != requests.end()) {
          requests.erase(i);
        }
      }
      for (auto& v : errorResponses) {
        throw std::runtime_error(fmt::sprintf("Moodist Allocator IPC remote error: %s\n", v));
      }
    }
    if (requests.empty()) {
      remoteMappedSize = o;
      log.info("All requests successful! Updating remoteMappedSize to %#x\n", remoteMappedSize);
      return;
    }
    if (connection->closed()) {
      throw std::runtime_error("Moodist Allocator IPC remote process unexpectedly exited or closed connection.");
    } else {
      throw std::runtime_error("Moodist Allocator IPC timed out waiting for response from another process.");
    }
  }
};

std::shared_ptr<PeerMemoryImpl> connect(Socket socket) {
  auto peer = std::make_shared<PeerMemoryImpl>();
  peer->connection = std::make_shared<Connection>(std::move(socket));
  peer->connection->readfd([peer](Error* e, BufferHandle buffer, int fd) {
    if (e) {
      peer->connection->close();
    } else {
      peer->onRead(std::move(buffer), fd);
    }
  });
  auto buffer = serializeToBuffer(signature, id());
  peer->connection->write(std::move(buffer), nullptr);
  return peer;
}

void onAccept(Error* error, Socket socket) {
  if (error) {
    log.error("accept error %s\n", error->what());
    return;
  }
  log.debug("got a new connection!\n");

  connect(std::move(socket));
}

} // namespace

std::string initIpc() {
  std::string id = fmt::sprintf("moodist-allocator-ipc-%s-%d", randomName(), ::getpid());

  listener = Socket::Unix();
  listener.listen(id);

  listener.accept(&onAccept);

  cuCtxGetCurrent(&cuContext);
  if (!cuContext) {
    CUdevice cuDevice;
    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&cuDevice, c10::cuda::current_device()));
    CHECK_CU(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
    CHECK_CU(cuCtxSetCurrent(cuContext));
  }

  return id;
}

std::string id() {
  static std::string s = initIpc();
  return s;
}
uintptr_t PeerMemory::remoteBaseAddress() {
  PeerMemoryImpl* impl = (PeerMemoryImpl*)this->impl;
  std::unique_lock l(impl->mutex);

  if (impl->remoteBaseAddress) {
    return impl->remoteBaseAddress;
  }

  impl->extend(0x1000, l);
  CHECK(impl->remoteBaseAddress);
  return impl->remoteBaseAddress;
}
void PeerMemory::remoteExtend(uintptr_t length) {
  PeerMemoryImpl* impl = (PeerMemoryImpl*)this->impl;
  std::unique_lock l(impl->mutex);
  impl->extend(length, l);
}

PeerMemory getPeerMemory(std::string id) {
  std::unique_lock l(globalMutex);
  auto i = peerMemories.find(id);
  if (i != peerMemories.end()) {
    return PeerMemory{i->second};
  }
  auto start = std::chrono::steady_clock::now();

  log.info("(%s) connecting to (%s)\n", allocator::id(), id);

  auto socket = Socket::Unix();
  socket.connect(id, [](Error* e) {});
  auto peer = connect(std::move(socket));

  for (int s = 0;
       (s < 30 || std::chrono::steady_clock::now() - start < std::chrono::minutes(2)) && !peer->connection->closed();
       ++s) {
    l.unlock();
    futexWait(&newConnectionCounter, 0, std::chrono::seconds(1));
    l.lock();
    log.info("closed? %d\n", peer->connection->closed());
    newConnectionCounter = 0;
    i = peerMemories.find(id);
    if (i != peerMemories.end()) {
      log.error("Moodist Allocator IPC successfully connected!\n");
      return PeerMemory{i->second};
    }
  }
  throw std::runtime_error("Moodist Allocator IPC failed to connect to a process");
}

} // namespace allocator
} // namespace moodist