#include "ipc_mapper.h"
#include "async.h"
#include "common.h"
#include "group.h"
#include "setup_comms.h"
#include "socket.h"
#include "synchronization.h"

#include <sys/mman.h>
#include <unistd.h>

namespace moodist {

struct Memfd {
  int fd = -1;
  void* base = nullptr;
  size_t size = 0;

  Memfd() = default;
  ~Memfd() {
    if (base != nullptr) {
      munmap(base, size);
      base = nullptr;
    }
    if (fd != -1) {
      close(fd);
      fd = -1;
    }
  }
  Memfd(const Memfd&) = delete;
  Memfd(Memfd&& n) {
    std::swap(fd, n.fd);
    std::swap(base, n.base);
    std::swap(size, n.size);
  }
  Memfd& operator=(const Memfd&) = delete;
  Memfd& operator=(Memfd&& n) {
    std::swap(fd, n.fd);
    std::swap(base, n.base);
    std::swap(size, n.size);
    return *this;
  }

  static Memfd create(size_t size) {
    int fd;
    fd = memfd_create("Memfd", MFD_CLOEXEC);
    if (fd == -1) {
      throw std::system_error(errno, std::generic_category(), "memfd_create");
    }
    if (ftruncate(fd, size)) {
      close(fd);
      throw std::system_error(errno, std::generic_category(), "ftruncate");
    }

    log.debug("memfd create %d\n", fd);

    return map(fd, size);
  }
  static Memfd map(int fd, size_t size) {
    void* base;
    base = mmap(nullptr, size, PROT_READ | PROT_WRITE | MAP_POPULATE, MAP_SHARED, fd, 0);
    if (!base || base == MAP_FAILED) {
      close(fd);
      throw std::system_error(errno, std::generic_category(), "mmap");
    }

    Memfd r;
    r.fd = fd;
    r.size = size;
    r.base = base;
    return r;
  }
};

struct IpcMapperImpl : IpcMapper {

  struct alignas(256) SharedStruct {
    std::atomic_uint32_t count = 0;

    struct Slot {
      std::atomic_size_t sourceRank = -1;
      std::atomic_int stage = 0;
      uint32_t requestStepValue;
      CUipcMemHandle request;
      size_t requestBytes;
      uintptr_t requestUnmapAddress;
      CUdeviceptr response;
    };

    std::array<Slot, 32> slots;
  };

  IpcMapperImpl(Group* group) {
    this->group = group;

    memsize = 4096 * 32 + 1024 * group->size * Group::maxConcurrency;
  }
  virtual ~IpcMapperImpl() override {
    if (thread.joinable()) {
      terminate = true;
      shared->count = -1;
      futexWakeAll(&shared->count);
      thread.join();
    }
  }

  size_t memsize;
  static constexpr uint64_t signature = 0x4a23d70736500758;

  Memfd mymem;
  std::array<Memfd, 8> peermem;

  std::atomic_bool terminate = false;
  std::thread thread;

  SharedStruct* shared = nullptr;
  std::array<SharedStruct*, 8> peershared;

  struct OutgoingRequest {
    size_t peerIndex;
    size_t slotIndex;
    Function<void(uintptr_t)> callback;
  };
  std::vector<OutgoingRequest> outgoing;

  void entry() {

    try {

      async::setCurrentThreadName("ipc mapper");

      // async::Scheduler unmapScheduler;
      // unmapScheduler.setMaxThreads(1);
      // unmapScheduler.setName("ipc unmap");

      CHECK_CU(cuCtxSetCurrent(group->cuContext));

      // unmapScheduler.run([this]() { CHECK_CU(cuCtxSetCurrent(group->cuContext)); });

      HashMap<uintptr_t, uint32_t> activeMaps;

      while (true) {
        if (terminate.load(std::memory_order_relaxed)) {
          break;
        }
        int32_t n = shared->count;
        if (n <= 0) {
          futexWait(&shared->count, 0, std::chrono::seconds(1));
          continue;
        }
        {
          std::lock_guard l(mutex);
          if (!outgoing.empty()) {
            for (auto i = outgoing.begin(); i != outgoing.end();) {
              SharedStruct* nshared = peershared[i->peerIndex];
              auto& slot = nshared->slots[i->slotIndex];
              if (slot.stage == 3) {
                log.debug("%d: got request response -> %#x\n", group->rank, slot.response);
                std::move(i->callback)(slot.response);
                slot.response = 0;
                slot.stage = 0;
                i = outgoing.erase(i);
              } else {
                ++i;
              }
            }
          }
        }

        for (auto& v : shared->slots) {
          int stage = v.stage;
          if (stage == 2) {
            size_t sourceRank = v.sourceRank;
            CHECK(sourceRank < group->size);
            if (v.requestUnmapAddress) {
              log.debug(
                  "%d: got ipc unmap request (address %#x size %#x) from rank %d!\n", group->rank,
                  v.requestUnmapAddress, v.requestBytes, sourceRank);
              std::lock_guard l(mutex);
              if (this->stepValue > v.requestStepValue) {
                v.response = 0;
                log.error(
                    "Cannot unmap due to local stepvalue %#x vs request stepvalue %#x\n", this->stepValue,
                    v.requestStepValue);
              } else {
                auto i = activeMaps.find(v.requestUnmapAddress);
                CHECK(i != activeMaps.end());
                CHECK(i->second == 1);
                activeMaps.erase(i);

                CHECK_CU(cuIpcCloseMemHandle(v.requestUnmapAddress));
                v.response = 1;

                // auto ptr = std::make_shared<std::atomic_bool>(false);
                // unmapScheduler.run([address = v.requestUnmapAddress, ptr]() {
                //   CHECK_CU(cuIpcCloseMemHandle(address));
                //   *ptr = true;
                // });
                // v.response = 0;
                // auto start = Clock::now();
                // while (Clock::now() - start < std::chrono::milliseconds(10)) {
                //   if (*ptr) {
                //     v.response = 1;
                //     break;
                //   }
                // }
                // if (v.response == 0) {
                //   log.error("Timed out waiting for unmap\n");
                // }
              }
            } else {
              log.debug("%d: got ipc map request (size %#x) from rank %d!\n", group->rank, v.requestBytes, sourceRank);
              CHECK_CU(cuIpcOpenMemHandle(&v.response, v.request, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
              log.debug("%d: mapped %#x bytes to %#x\n", group->rank, v.requestBytes, v.response);

              ++activeMaps[v.response];
              CHECK(activeMaps[v.response] == 1);
            }
            v.stage = 3;

            SharedStruct* nshared = peershared.at(group->getPeerIndex(sourceRank));
            ++nshared->count;
            futexWakeAll(&nshared->count);
          }
        }

        shared->count -= n;
      }
      for (auto& v : activeMaps) {
        CHECK_CU(cuIpcCloseMemHandle(v.first));
      }
    } catch (const std::exception& e) {
      log.error("ipc mapper got exception %s\n", e.what());
      while (true)
        ;
      std::lock_guard l(mutex);
      if (!hasException) {
        hasException = true;
        exception = std::current_exception();
      }
    }
  }

  void
  sendRequestAddress(size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback) {
    size_t slotIndex = 0;
    SharedStruct* nshared = peershared.at(peerIndex);
    TORCH_CHECK(nshared != nullptr);
    while (true) {
      if (hasException) {
        std::lock_guard l(mutex);
        std::rethrow_exception(*exception);
      }
      int zero = 0;
      if (nshared->slots[slotIndex].stage.compare_exchange_strong(zero, 1)) {
        break;
      }
      if (slotIndex == nshared->slots.size() - 1) {
        slotIndex = 0;
      } else {
        ++slotIndex;
      }
    }
    log.debug(
        "sending ipc map request to rank %d using slot %d (size %#x)\n", group->ipcRanks.at(peerIndex), slotIndex,
        size);
    std::unique_lock l(mutex);
    outgoing.emplace_back();
    OutgoingRequest& req = outgoing.back();
    req.peerIndex = peerIndex;
    req.slotIndex = slotIndex;
    req.callback = std::move(callback);
    l.unlock();

    auto& slot = nshared->slots[slotIndex];
    slot.sourceRank = group->rank;
    slot.requestStepValue = this->stepValue;
    slot.request = handle;
    slot.requestBytes = size;
    slot.requestUnmapAddress = 0;
    slot.stage = 2;
    ++nshared->count;
    futexWakeAll(&nshared->count);
  }

  void sendRequestUnmap(size_t peerIndex, uintptr_t base, size_t size, Function<void(uintptr_t)> callback) {
    size_t slotIndex = 0;
    SharedStruct* nshared = peershared.at(peerIndex);
    TORCH_CHECK(nshared != nullptr);
    while (true) {
      if (hasException) {
        std::lock_guard l(mutex);
        std::rethrow_exception(*exception);
      }
      int zero = 0;
      if (nshared->slots[slotIndex].stage.compare_exchange_strong(zero, 1)) {
        break;
      }
      if (slotIndex == nshared->slots.size() - 1) {
        slotIndex = 0;
      } else {
        ++slotIndex;
      }
    }
    log.debug(
        "sending ipc unmap request to rank %d using slot %d (base %#x size %#x)\n", group->ipcRanks.at(peerIndex),
        slotIndex, base, size);
    std::unique_lock l(mutex);
    outgoing.emplace_back();
    OutgoingRequest& req = outgoing.back();
    req.peerIndex = peerIndex;
    req.slotIndex = slotIndex;
    req.callback = std::move(callback);
    l.unlock();

    auto& slot = nshared->slots[slotIndex];
    slot.sourceRank = group->rank;
    slot.requestStepValue = this->stepValue;
    slot.request = {};
    slot.requestBytes = size;
    slot.requestUnmapAddress = base;
    CHECK(base != 0);
    slot.stage = 2;
    ++nshared->count;
    futexWakeAll(&nshared->count);
  }

  void init() {

    mymem = Memfd::create(memsize);

    shared = new (mymem.base) SharedStruct();

    std::string myAddress = "ipc-mapper-" + randomName();

    struct SocketHelper {
      Socket socket;
      CachedReader reader;
      int32_t recvdRank = -1;

      SocketHelper(Socket socket) : socket(std::move(socket)), reader(&this->socket) {}
    };

    std::vector<std::shared_ptr<SocketHelper>> sockets;

    auto onRead = [&](std::shared_ptr<SocketHelper> socket, Error* error) {
      if (error) {
        log.error("read error %s\n", error->what());
        socket->socket.close();
        return;
      }
      if (socket->recvdRank == -1) {
        char* ptr = (char*)socket->reader.readBufferPointer(12);
        if (!ptr) {
          return;
        }
        uint64_t sig;
        uint32_t sourceRank;
        std::memcpy(&sig, ptr, 8);
        std::memcpy(&sourceRank, ptr + 8, 4);
        if (sig != signature) {
          socket->socket.close();
          return;
        }
        socket->recvdRank = sourceRank;
        if (socket->recvdRank == -1) {
          socket->socket.close();
        }
      }
      if (socket->recvdRank != -1) {
        int fd = socket->socket.recvFd(socket->reader);
        if (fd != -1) {
          size_t peerIndex = group->getPeerIndex(socket->recvdRank);
          {
            std::lock_guard l(mutex);
            peermem[peerIndex] = Memfd::map(fd, memsize);
          }
          socket->recvdRank = -1;
          socket->socket.close();
        }
      }
    };

    auto listener = Socket::Unix();
    listener.listen(myAddress);
    listener.accept([&](Error* error, Socket socket) {
      if (error) {
        log.error("accept error %s\n", error->what());
        return;
      }
      log.debug("got a new connection!\n");
      auto s = std::make_shared<SocketHelper>(std::move(socket));
      s->socket.setOnRead([s, &onRead](Error* error) { onRead(s, error); });
    });

    for (size_t i : group->ipcRanks) {
      group->setupComms->sendTo(i, myAddress);
    }
    auto data = serializeToBuffer(signature, (uint32_t)group->rank);
    for (size_t i : group->ipcRanks) {
      auto s = std::make_shared<SocketHelper>(Socket::Unix());
      s->socket.connect(group->setupComms->recvFrom<std::string>(i), [](Error* error) {
        if (error) {
          log.error("connect error %s\n", error->what());
        }
      });

      iovec iov;
      iov.iov_base = data->data();
      iov.iov_len = data->size();

      s->socket.writev(&iov, 1, [](Error* error) {
        if (error) {
          log.error("write error %s\n", error->what());
        }
      });
      s->socket.sendFd(mymem.fd, [](Error* error) {
        if (error) {
          log.error("sendfd error %s\n", error->what());
        }
      });

      sockets.push_back(s);
    }

    for (size_t i : group->ipcRanks) {
      size_t index = group->getPeerIndex(i);
      std::unique_lock l(mutex);
      while (!peermem[index].base) {
        l.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        l.lock();
      }
    }

    sockets.clear();
    listener.close();

    for (size_t i = 0; i != peermem.size(); ++i) {
      peershared[i] = (SharedStruct*)peermem[i].base;
    }

    thread = std::thread([this] { entry(); });
  }

  void* getPeerSharedMem(size_t peerIndex, size_t offset, size_t size) {
    TORCH_CHECK(offset + size <= memsize);
    return (void*)((uintptr_t)(void*)(peershared[peerIndex] + 1) + offset);
  }

  void* getMySharedMem(size_t offset, size_t size) {
    TORCH_CHECK(offset + size <= memsize);
    return (void*)((uintptr_t)(void*)(shared + 1) + offset);
  }
};

void IpcMapper::init() {
  ((IpcMapperImpl*)this)->init();
}

void IpcMapper::sendRequestAddress(
    size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendRequestAddress(peerIndex, handle, size, std::move(callback));
}

void IpcMapper::sendRequestUnmap(size_t peerIndex, uintptr_t base, size_t size, Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendRequestUnmap(peerIndex, base, size, std::move(callback));
}

void* IpcMapper::getMySharedMem(size_t offset, size_t size) {
  return ((IpcMapperImpl*)this)->getMySharedMem(offset, size);
}

void* IpcMapper::getPeerSharedMem(size_t peerIndex, size_t offset, size_t size) {
  return ((IpcMapperImpl*)this)->getPeerSharedMem(peerIndex, offset, size);
}

std::unique_ptr<IpcMapper> createIpcMapper(Group* group) {
  return std::make_unique<IpcMapperImpl>(group);
}

} // namespace moodist
