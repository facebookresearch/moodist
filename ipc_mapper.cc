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

  static Memfd create(size_t size, int node) {
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

    auto r = map(fd, size);
    if (node >= 0) {
      numa_move(r.base, r.size, node);
    }
    return r;
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

enum { requestBad, requestMapAddress, requestMapEvent, requestUnmap };

struct IpcMapperImpl : IpcMapper {

  struct alignas(256) SharedStruct {
    std::atomic_uint32_t count = 0;

    struct Slot {
      std::atomic_size_t sourceRank = -1;
      std::atomic_int stage = 0;
      uint32_t requestStepValue;

      int kind = requestBad;

      union {
        CUipcMemHandle requestMapMemory;
        CUipcEventHandle requestMapEvent;
        uintptr_t requestUnmapAddress;
      };
      size_t requestBytes;
      CUdeviceptr response;
    };

    std::array<Slot, 32> slots;

    struct Queue {
      std::atomic_size_t front = 0;
      std::atomic_size_t back = 0;
      std::array<uint8_t, 256> data;
    };

    std::array<Queue, 8> queue;

    struct EventQueue {
      std::atomic_size_t front = 0;
      std::atomic_size_t back = 0;
      std::array<uintptr_t, 8> data;
    };
    std::array<EventQueue, 8> eventQueue;
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
            if (v.kind == requestUnmap) {
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
              }
            } else if (v.kind == requestMapAddress) {
              log.debug(
                  "%d: got ipc map memory request (size %#x) from rank %d!\n", group->rank, v.requestBytes, sourceRank);
              CHECK_CU(cuIpcOpenMemHandle(&v.response, v.requestMapMemory, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
              log.debug("%d: mapped %#x bytes to %#x\n", group->rank, v.requestBytes, v.response);

              ++activeMaps[v.response];
              CHECK(activeMaps[v.response] == 1);
            } else if (v.kind == requestMapEvent) {
              log.debug(
                  "%d: got ipc map event request (size %#x) from rank %d!\n", group->rank, v.requestBytes, sourceRank);
              CUevent event = nullptr;
              CHECK_CU(cuIpcOpenEventHandle(&event, v.requestMapEvent));
              v.response = (uintptr_t)event;
              log.debug("%d: event mapped to %#x\n", group->rank, v.response);
            } else {
              CHECK(false);
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
        auto err = cuIpcCloseMemHandle(v.first);
        if (err == CUDA_ERROR_DEINITIALIZED) {
          break;
        }
        CHECK_CU(err);
      }
    } catch (const std::exception& e) {
      log.error("ipc mapper got exception %s\n", e.what());
      std::lock_guard l(mutex);
      if (!hasException) {
        hasException = true;
        exception = std::current_exception();
      }
    }
  }

  template<typename F>
  void enqueue(size_t peerIndex, Function<void(uintptr_t)> callback, F&& f) {
    size_t slotIndex = 0;
    SharedStruct* nshared = peershared.at(peerIndex);
    CHECK(nshared != nullptr);
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

    std::unique_lock l(mutex);
    outgoing.emplace_back();
    OutgoingRequest& req = outgoing.back();
    req.peerIndex = peerIndex;
    req.slotIndex = slotIndex;
    req.callback = std::move(callback);
    l.unlock();

    auto& slot = nshared->slots[slotIndex];
    f(slot);
    log.debug(
        "sending ipc request (kind %d) to rank %d using slot %d\n", slot.kind, group->ipcRanks.at(peerIndex),
        slotIndex);
    slot.stage = 2;
    ++nshared->count;
    futexWakeAll(&nshared->count);
  }

  void
  sendRequestAddress(size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback) {
    enqueue(peerIndex, std::move(callback), [&](auto& slot) {
      slot.kind = requestMapAddress;
      slot.sourceRank = group->rank;
      slot.requestStepValue = this->stepValue;
      slot.requestMapMemory = handle;
      slot.requestBytes = size;
    });
  }

  void sendRequestEvent(size_t peerIndex, const CUipcEventHandle& handle, Function<void(uintptr_t)> callback) {
    enqueue(peerIndex, std::move(callback), [&](auto& slot) {
      slot.kind = requestMapEvent;
      slot.sourceRank = group->rank;
      slot.requestStepValue = this->stepValue;
      slot.requestMapEvent = handle;
    });
  }

  void sendRequestUnmap(size_t peerIndex, uintptr_t base, size_t size, Function<void(uintptr_t)> callback) {
    enqueue(peerIndex, std::move(callback), [&](auto& slot) {
      slot.kind = requestUnmap;
      slot.sourceRank = group->rank;
      slot.requestStepValue = this->stepValue;
      slot.requestBytes = size;
      slot.requestUnmapAddress = base;
    });
  }

  void init(int node) {

    mymem = Memfd::create(memsize, node);

    shared = new (mymem.base) SharedStruct();

    // for (size_t i : group->ipcRanks) {
    //   group->setupComms->sendTo(i, allocator::id());
    // }
    // for (size_t i : group->ipcRanks) {
    //   peerMemoryId.at(group->getPeerIndex(i)) = group->setupComms->recvFrom<std::string>(i);
    // }

    std::string myAddress = fmt::sprintf("ipc-mapper-%d-%s", ::getpid(), randomName());

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
    CHECK(offset + size <= memsize);
    return (void*)((uintptr_t)(void*)(peershared[peerIndex] + 1) + offset);
  }

  void* getMySharedMem(size_t offset, size_t size) {
    CHECK(offset + size <= memsize);
    return (void*)((uintptr_t)(void*)(shared + 1) + offset);
  }

  void pushImpl(size_t peerIndex, const void* ptr, size_t n) {
    SharedStruct* nshared = peershared.at(peerIndex);
    auto& queue = nshared->queue[group->peerMyRemoteIndex[peerIndex]];
    while (queue.data.size() - (queue.front - queue.back) < n) {
      _mm_pause();
    }
    size_t offset = queue.front % queue.data.size();
    size_t c = std::min(queue.data.size() - offset, n);
    std::memcpy(queue.data.data() + offset, ptr, c);
    std::memcpy(queue.data.data(), (uint8_t*)ptr + c, n - c);
    queue.front += n;
  }
  void popImpl(size_t peerIndex, void* ptr, size_t n) {
    auto& queue = shared->queue[peerIndex];
    while (queue.front - queue.back < n) {
      _mm_pause();
    }
    size_t offset = queue.back % queue.data.size();
    size_t c = std::min(queue.data.size() - offset, n);
    std::memcpy(ptr, queue.data.data() + offset, c);
    std::memcpy((uint8_t*)ptr + c, queue.data.data(), n - c);
    queue.back += n;
  }

  void push(size_t peerIndex, const void* ptr, size_t n) {
    pushImpl(peerIndex, &n, sizeof(n));
    pushImpl(peerIndex, ptr, n);
  }
  void pop(size_t peerIndex, void* ptr, size_t n) {
    size_t nn;
    popImpl(peerIndex, &nn, sizeof(nn));
    CHECK(n == nn);
    popImpl(peerIndex, ptr, n);
  }

  void pushEventQueue(size_t peerIndex, uintptr_t value) {
    SharedStruct* nshared = peershared.at(peerIndex);
    auto& queue = nshared->eventQueue.at(group->peerMyRemoteIndex.at(peerIndex));
    while (queue.front - queue.back == queue.data.size()) {
      _mm_pause();
    }
    queue.data[queue.front % queue.data.size()] = value;
    ++queue.front;
  }
  uintptr_t popEventQueue(size_t peerIndex) {
    auto& queue = shared->eventQueue[peerIndex];
    while (queue.front == queue.back) {
      _mm_pause();
    }
    uintptr_t r = queue.data[queue.back % queue.data.size()];
    ++queue.back;
    return r;
  }

  std::array<size_t, 8> numIpcEvents{};

  void streamRecord(size_t peerIndex, CUstream stream) {
    CUevent event = nullptr;
    if (numIpcEvents[peerIndex] >= 4) {
      event = (CUevent)popEventQueue(peerIndex);
    } else {
      ++numIpcEvents[peerIndex];
      log.debug("%d: Create new ipc event for peer %d!\n", group->rank, peerIndex);
      CHECK_CU(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS));
    }
    CHECK_CU(cuEventRecord(event, stream));
    IpcMapper::push<uint32_t>(peerIndex, 0xf1020304);
    pushEvent(peerIndex, event);
    IpcMapper::push<uint32_t>(peerIndex, 0x01020304);
    IpcMapper::push<uintptr_t>(peerIndex, (uintptr_t)event);
  }
  void streamWait(size_t peerIndex, CUstream stream) {
    CHECK(IpcMapper::pop<uint32_t>(peerIndex) == 0xf1020304);
    CHECK_CU(cuStreamWaitEvent(stream, popEvent(peerIndex), CU_EVENT_WAIT_DEFAULT));
    CHECK(IpcMapper::pop<uint32_t>(peerIndex) == 0x01020304);
    pushEventQueue(peerIndex, IpcMapper::pop<uintptr_t>(peerIndex));

    // enqueue(
    //     peerIndex, [](uintptr_t) {},
    //     [this, ev](auto& slot) {
    //       slot.kind = requestReturnEvent;
    //       slot.sourceRank = group->rank;
    //       slot.returnEventValue = (uintptr_t)ev;
    //     });
  }
};

void IpcMapper::init(int node) {
  ((IpcMapperImpl*)this)->init(node);
}

void IpcMapper::sendRequestAddress(
    size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendRequestAddress(peerIndex, handle, size, std::move(callback));
}

void IpcMapper::sendRequestEvent(size_t peerIndex, const CUipcEventHandle& handle, Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendRequestEvent(peerIndex, handle, std::move(callback));
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

void IpcMapper::push(size_t peerIndex, const void* ptr, size_t n) {
  return ((IpcMapperImpl*)this)->push(peerIndex, ptr, n);
}
void IpcMapper::pop(size_t peerIndex, void* ptr, size_t n) {
  return ((IpcMapperImpl*)this)->pop(peerIndex, ptr, n);
}

void IpcMapper::streamRecord(size_t peerIndex, CUstream stream) {
  return ((IpcMapperImpl*)this)->streamRecord(peerIndex, stream);
}

void IpcMapper::streamWait(size_t peerIndex, CUstream stream) {
  return ((IpcMapperImpl*)this)->streamWait(peerIndex, stream);
}

std::unique_ptr<IpcMapper> createIpcMapper(Group* group) {
  return std::make_unique<IpcMapperImpl>(group);
}

} // namespace moodist
