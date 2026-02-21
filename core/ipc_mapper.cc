// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ipc_mapper.h"
#include "api/allocator_api.h"
#include "async.h"
#include "common.h"
#include "group.h"
#include "setup_comms.h"
#include "socket.h"
#include "synchronization.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
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
  Memfd(Memfd&& n) noexcept {
    std::swap(fd, n.fd);
    std::swap(base, n.base);
    std::swap(size, n.size);
  }
  Memfd& operator=(const Memfd&) = delete;
  Memfd& operator=(Memfd&& n) noexcept {
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

enum {
  requestBad,
  requestMapAddress,
  requestMapEvent,
  requestUnmap,
  requestMapVMM,
  requestMapVMMFabric,
  requestMulticast,
  requestMulticastFabric,
  requestMulticastBind
};

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
        CUmemFabricHandle requestMapFabric;
      };
      size_t requestBytes;
      size_t requestVMMOffset; // For VMM: offset within the handle's allocation
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
    for (int& fd : vmmSendFds) {
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
    }
    for (int& fd : vmmRecvFds) {
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
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

  // Persistent Unix socket connections for VMM fd exchange.
  // vmmSendFds[peerIndex] is used by the sender to send fds to peer.
  // vmmRecvFds[peerIndex] is used by the handler thread to receive fds from peer.
  std::array<int, 8> vmmSendFds{};
  std::array<int, 8> vmmRecvFds{};

  // When true, use fabric handles instead of POSIX fd passing for VMM IPC
  bool useFabric = false;

  // Multicast objects imported by the handler thread, tracked for cleanup
  struct MulticastPeerState {
    CUmemGenericAllocationHandle mcHandle;
    CUmemGenericAllocationHandle scratchHandle;
    CUdeviceptr mcVA;
    size_t allocSize;
  };
  std::vector<MulticastPeerState> multicastPeerStates;

  // Multicast handles imported during addDevice phase, waiting for bind phase.
  // Indexed by the handle index returned to the sender.
  struct StoredMcHandle {
    CUmemGenericAllocationHandle handle;
    size_t size;
  };
  std::vector<StoredMcHandle> storedMcHandles;

  // Send a file descriptor over a Unix socket using SCM_RIGHTS
  static void sendFdOverSocket(int sockFd, int fdToSend) {
    char buf[1] = {0};
    ::iovec iov;
    iov.iov_base = buf;
    iov.iov_len = 1;

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct msghdr msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    struct cmsghdr* cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    std::memcpy(CMSG_DATA(cmptr), &fdToSend, sizeof(int));

    ssize_t n;
    do {
      n = sendmsg(sockFd, &msg, 0);
    } while (n == -1 && errno == EINTR);
    CHECK(n == 1);
  }

  // Receive a file descriptor over a Unix socket using SCM_RIGHTS
  static int recvFdFromSocket(int sockFd) {
    char buf[1];
    ::iovec iov;
    iov.iov_base = buf;
    iov.iov_len = 1;

    union {
      struct cmsghdr cm;
      char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct msghdr msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    ssize_t n;
    do {
      n = recvmsg(sockFd, &msg, 0);
    } while (n == -1 && errno == EINTR);
    CHECK(n == 1);

    struct cmsghdr* cmptr = CMSG_FIRSTHDR(&msg);
    CHECK(cmptr != nullptr);
    CHECK(cmptr->cmsg_len == CMSG_LEN(sizeof(int)));
    CHECK(cmptr->cmsg_level == SOL_SOCKET);
    CHECK(cmptr->cmsg_type == SCM_RIGHTS);

    int fd;
    std::memcpy(&fd, CMSG_DATA(cmptr), sizeof(int));
    return fd;
  }

  void entry() {

    try {

      async::setCurrentThreadName("ipc mapper");

      // async::Scheduler unmapScheduler;
      // unmapScheduler.setMaxThreads(1);
      // unmapScheduler.setName("ipc unmap");

      CHECK_CU(cuCtxSetCurrent(group->cuContext));

      // unmapScheduler.run([this]() { CHECK_CU(cuCtxSetCurrent(group->cuContext)); });

      HashMap<uintptr_t, uint32_t> activeMaps;

      struct HandlerVMMPeer {
        uintptr_t reservedBase = 0;
        size_t reservedSize = 0;
        struct ImportedChunk {
          CUmemGenericAllocationHandle handle;
          size_t offset;
          size_t size;
        };
        std::vector<ImportedChunk> importedChunks;
      };
      std::array<HandlerVMMPeer, 8> handlerVMMPeers;

      // Common VMM mapping logic: lazy-init VA reservation, map, set access, track
      auto mapVMMHandle = [&](HandlerVMMPeer& hpeer, CUmemGenericAllocationHandle importedHandle, size_t mapSize,
                              size_t chunkOffset, size_t peerIndex, size_t sourceRank) -> CUdeviceptr {
        // Lazy-init: reserve 1TB VA range for this peer on first chunk
        if (hpeer.reservedBase == 0) {
          CUmemAllocationProp prop;
          std::memset(&prop, 0, sizeof(prop));
          prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
          prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          CUdevice dev;
          CHECK_CU(cuCtxGetDevice(&dev));
          prop.location.id = dev;

          size_t granularity = 0;
          CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

          constexpr size_t reserveSize = (size_t)1024 * 1024 * 1024 * 1024; // 1 TB
          CUdeviceptr base = 0;
          CHECK_CU(cuMemAddressReserve(&base, reserveSize, granularity, 0, 0));
          hpeer.reservedBase = base;
          hpeer.reservedSize = reserveSize;

          log.debug("%d: reserved 1TB VA at %#x for peer %d (rank %d)\n", group->rank, base, peerIndex, sourceRank);
        }

        CHECK(chunkOffset + mapSize <= hpeer.reservedSize);
        CUdeviceptr mapAddr = hpeer.reservedBase + chunkOffset;

        CHECK_CU(cuMemMap(mapAddr, mapSize, 0, importedHandle, 0));

        CUmemAccessDesc accessDesc;
        std::memset(&accessDesc, 0, sizeof(accessDesc));
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        CUdevice dev;
        CHECK_CU(cuCtxGetDevice(&dev));
        accessDesc.location.id = dev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CHECK_CU(cuMemSetAccess(mapAddr, mapSize, &accessDesc, 1));

        hpeer.importedChunks.push_back({importedHandle, chunkOffset, mapSize});

        log.debug("%d: VMM mapped %#x bytes at %#x (peer %d, offset %#x)\n", group->rank, mapSize, mapAddr, peerIndex,
            chunkOffset);

        return mapAddr;
      };

      // Phase 1: Import a multicast handle and add this device. Store handle for later bind.
      auto addDeviceToMulticast = [&](CUmemGenericAllocationHandle mcHandle, size_t size) -> size_t {
        CUdevice dev;
        CHECK_CU(cuCtxGetDevice(&dev));
        CHECK_CU(cuMulticastAddDevice(mcHandle, dev));

        size_t idx = storedMcHandles.size();
        storedMcHandles.push_back({mcHandle, size});

        log.debug("addDeviceToMulticast: stored handle at index %zu, size=%zu\n", idx, size);
        return idx;
      };

      // Phase 2: Allocate scratch, bind to multicast object, map VA.
      // Called after ALL devices have been added via addDeviceToMulticast.
      auto bindAndMapMulticast = [&](size_t handleIndex) -> CUdeviceptr {
        CHECK(handleIndex < storedMcHandles.size());
        auto mcHandle = storedMcHandles[handleIndex].handle;
        auto size = storedMcHandles[handleIndex].size;

        CUdevice dev;
        CHECK_CU(cuCtxGetDevice(&dev));

        // Allocate scratch buffer (physical memory for this peer)
        CUmemAllocationProp prop;
        std::memset(&prop, 0, sizeof(prop));
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        if (useFabric) {
          prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        } else {
          prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        }

        size_t granularity = 0;
        CHECK_CU(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

        // Round up size to granularity
        size_t allocSize = (size + granularity - 1) / granularity * granularity;

        CUmemGenericAllocationHandle scratchHandle;
        CHECK_CU(cuMemCreate(&scratchHandle, allocSize, &prop, 0));

        // Bind scratch to multicast object at offset 0
        CHECK_CU(cuMulticastBindMem(mcHandle, 0, scratchHandle, 0, allocSize, 0));

        // Reserve VA and map the multicast object
        CUdeviceptr mcVA = 0;
        CHECK_CU(cuMemAddressReserve(&mcVA, allocSize, granularity, 0, 0));
        CHECK_CU(cuMemMap(mcVA, allocSize, 0, mcHandle, 0));

        CUmemAccessDesc accessDesc;
        std::memset(&accessDesc, 0, sizeof(accessDesc));
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = dev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CHECK_CU(cuMemSetAccess(mcVA, allocSize, &accessDesc, 1));

        // Track for cleanup
        multicastPeerStates.push_back({mcHandle, scratchHandle, mcVA, allocSize});

        log.debug("bindAndMapMulticast: idx=%zu, VA=%#llx, size=%zu, allocSize=%zu\n", handleIndex,
            (unsigned long long)mcVA, size, allocSize);
        return mcVA;
      };

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

        // Process any VMM chain continuations (outside mutex)
        processVMMChains();

        for (auto& v : shared->slots) {
          int stage = v.stage;
          if (stage == 2) {
            size_t sourceRank = v.sourceRank;
            CHECK(sourceRank < group->size);
            if (v.kind == requestUnmap) {
              log.debug("%d: got ipc unmap request (address %#x size %#x) from rank %d!\n", group->rank,
                  v.requestUnmapAddress, v.requestBytes, sourceRank);
              std::lock_guard l(mutex);
              if (this->stepValue > v.requestStepValue) {
                v.response = 0;
                log.error("Cannot unmap due to local stepvalue %#x vs request stepvalue %#x\n", this->stepValue,
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
            } else if (v.kind == requestMapVMM) {
              size_t peerIndex = group->getPeerIndex(sourceRank);
              log.debug("%d: got VMM map request (size %#x, offset %#x) from rank %d!\n", group->rank, v.requestBytes,
                  v.requestVMMOffset, sourceRank);

              int fd = recvFdFromSocket(vmmRecvFds[peerIndex]);

              CUmemGenericAllocationHandle importedHandle;
              CHECK_CU(cuMemImportShareableHandle(
                  &importedHandle, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
              ::close(fd);

              v.response = mapVMMHandle(handlerVMMPeers[peerIndex], importedHandle, v.requestBytes, v.requestVMMOffset,
                  peerIndex, sourceRank);
            } else if (v.kind == requestMapVMMFabric) {
              size_t peerIndex = group->getPeerIndex(sourceRank);
              log.debug("%d: got VMM fabric map request (size %#x, offset %#x) from rank %d!\n", group->rank,
                  v.requestBytes, v.requestVMMOffset, sourceRank);

              CUmemGenericAllocationHandle importedHandle;
              CHECK_CU(cuMemImportShareableHandle(&importedHandle, &v.requestMapFabric, CU_MEM_HANDLE_TYPE_FABRIC));

              v.response = mapVMMHandle(handlerVMMPeers[peerIndex], importedHandle, v.requestBytes, v.requestVMMOffset,
                  peerIndex, sourceRank);
            } else if (v.kind == requestMulticast) {
              size_t peerIndex = group->getPeerIndex(sourceRank);
              log.debug("%d: got multicast addDevice request (size %#x) from rank %d!\n", group->rank, v.requestBytes,
                  sourceRank);

              int fd = recvFdFromSocket(vmmRecvFds[peerIndex]);
              CUmemGenericAllocationHandle mcHandle;
              CHECK_CU(
                  cuMemImportShareableHandle(&mcHandle, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
              ::close(fd);

              v.response = addDeviceToMulticast(mcHandle, v.requestBytes);
            } else if (v.kind == requestMulticastFabric) {
              log.debug("%d: got multicast fabric addDevice request (size %#x) from rank %d!\n", group->rank,
                  v.requestBytes, sourceRank);

              CUmemGenericAllocationHandle mcHandle;
              CHECK_CU(cuMemImportShareableHandle(&mcHandle, &v.requestMapFabric, CU_MEM_HANDLE_TYPE_FABRIC));

              v.response = addDeviceToMulticast(mcHandle, v.requestBytes);
            } else if (v.kind == requestMulticastBind) {
              log.debug("%d: got multicast bind request (handleIndex %#x, size %#x) from rank %d!\n", group->rank,
                  v.requestVMMOffset, v.requestBytes, sourceRank);

              v.response = bindAndMapMulticast(v.requestVMMOffset);
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
      for (auto& peer : handlerVMMPeers) {
        for (auto& chunk : peer.importedChunks) {
          auto err = cuMemUnmap(peer.reservedBase + chunk.offset, chunk.size);
          if (err == CUDA_ERROR_DEINITIALIZED) {
            break;
          }
          cuMemRelease(chunk.handle);
        }
        if (peer.reservedBase) {
          cuMemAddressFree(peer.reservedBase, peer.reservedSize);
        }
      }
      // Clean up multicast peer states
      for (auto& mc : multicastPeerStates) {
        auto err = cuMemUnmap(mc.mcVA, mc.allocSize);
        if (err == CUDA_ERROR_DEINITIALIZED) {
          break;
        }
        cuMemAddressFree(mc.mcVA, mc.allocSize);
        cuMemRelease(mc.scratchHandle);
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
    log.debug("sending ipc request (kind %d) to rank %d using slot %d\n", slot.kind, group->ipcRanks.at(peerIndex),
        slotIndex);
    slot.stage = 2;
    ++nshared->count;
    futexWakeAll(&nshared->count);
  }

  void sendRequestAddress(
      size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback) {
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

  void sendRequestVMM(size_t peerIndex, CUmemGenericAllocationHandle handle, size_t handleSize, size_t offset,
      Function<void(uintptr_t)> callback) {
    // Export the handle as a POSIX fd
    int fd = -1;
    CHECK_CU(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    CHECK(fd >= 0);

    enqueue(peerIndex, std::move(callback), [&](auto& slot) {
      slot.kind = requestMapVMM;
      slot.sourceRank = group->rank;
      slot.requestStepValue = this->stepValue;
      slot.requestBytes = handleSize;
      slot.requestVMMOffset = offset;
    });

    // Send the fd over the persistent Unix socket connection
    sendFdOverSocket(vmmSendFds[peerIndex], fd);
    ::close(fd);
  }

  void sendRequestVMMFabric(size_t peerIndex, CUmemGenericAllocationHandle handle, size_t handleSize, size_t offset,
      Function<void(uintptr_t)> callback) {
    CUmemFabricHandle fabricHandle;
    CHECK_CU(cuMemExportToShareableHandle(&fabricHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

    enqueue(peerIndex, std::move(callback), [&](auto& slot) {
      slot.kind = requestMapVMMFabric;
      slot.sourceRank = group->rank;
      slot.requestStepValue = this->stepValue;
      slot.requestBytes = handleSize;
      slot.requestVMMOffset = offset;
      slot.requestMapFabric = fabricHandle;
    });
  }

  void sendMulticastHandle(
      size_t peerIndex, CUmemGenericAllocationHandle mcHandle, size_t size, Function<void(uintptr_t)> callback) {
    if (useFabric) {
      // Export as fabric handle, send via slot
      CUmemFabricHandle fabricHandle;
      CHECK_CU(cuMemExportToShareableHandle(&fabricHandle, mcHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
      enqueue(peerIndex, std::move(callback), [&](auto& slot) {
        slot.kind = requestMulticastFabric;
        slot.sourceRank = group->rank;
        slot.requestStepValue = this->stepValue;
        slot.requestBytes = size;
        slot.requestMapFabric = fabricHandle;
      });
    } else {
      // Export as POSIX fd, send via VMM socket
      int fd = -1;
      CHECK_CU(cuMemExportToShareableHandle(&fd, mcHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      enqueue(peerIndex, std::move(callback), [&](auto& slot) {
        slot.kind = requestMulticast;
        slot.sourceRank = group->rank;
        slot.requestStepValue = this->stepValue;
        slot.requestBytes = size;
      });
      sendFdOverSocket(vmmSendFds[peerIndex], fd);
      ::close(fd);
    }
  }

  void sendMulticastBind(size_t peerIndex, size_t handleIndex, size_t size, Function<void(uintptr_t)> callback) {
    enqueue(peerIndex, std::move(callback), [&](auto& slot) {
      slot.kind = requestMulticastBind;
      slot.sourceRank = group->rank;
      slot.requestStepValue = this->stepValue;
      slot.requestBytes = size;
      slot.requestVMMOffset = handleIndex;
    });
  }

  void sendNextVMMChunk(size_t peerIndex) {
    std::unique_lock l(mutex);
    auto& peer = vmmPeers[peerIndex];

    // Check if we've mapped enough
    if (peer.mappedWatermark >= peer.pendingNeeded) {
      peer.mappingInProgress = false;
      auto callbacks = std::move(peer.pendingCallbacks);
      uintptr_t peerBase = peer.peerBase;
      l.unlock();
      for (auto& cb : callbacks) {
        log.debug("sendNextVMMChunk: invoking callback offset %#x -> %#x\n", cb.offset, peerBase + cb.offset);
        std::move(cb.callback)(peerBase + cb.offset);
      }
      --waitCount;
      return;
    }

    // Get the next chunk to map
    size_t chunkIndex = peer.mappedChunkCount;
    unsigned long long handle;
    size_t chunkOffset, chunkSize;
    l.unlock();

    bool ok = allocatorGetChunk(chunkIndex, &handle, &chunkOffset, &chunkSize);
    CHECK(ok);

    constexpr size_t reserveSize = (size_t)1024 * 1024 * 1024 * 1024;
    CHECK(chunkOffset + chunkSize <= reserveSize);

    log.debug("sendNextVMMChunk: mapping chunk %d for peer %d (offset %#x, size %#x)\n", chunkIndex, peerIndex,
        chunkOffset, chunkSize);

    auto callback = [this, peerIndex, chunkOffset, chunkSize](uintptr_t mappedAddress) {
      // NOTE: mutex IS held here (callback dispatched from entry() under lock_guard)
      auto& peer = vmmPeers[peerIndex];

      // On first chunk, record the peer's base address
      if (peer.mappedChunkCount == 0) {
        peer.peerBase = mappedAddress - chunkOffset;
        log.debug("sendNextVMMChunk: peer %d base = %#x\n", peerIndex, peer.peerBase);
      }

      peer.mappedWatermark = chunkOffset + chunkSize;
      peer.mappedChunkCount++;
      // Don't decrement waitCount or call sendNextVMMChunk here —
      // we're under mutex and can't call enqueue(). The entry() thread
      // will pick this up on the next iteration via vmmChainNeeded.
      peer.vmmChainPending = true;
    };

    if (useFabric) {
      sendRequestVMMFabric(peerIndex, handle, chunkSize, chunkOffset, std::move(callback));
    } else {
      sendRequestVMM(peerIndex, handle, chunkSize, chunkOffset, std::move(callback));
    }
  }

  // Called from the entry() thread after releasing the mutex to continue
  // VMM chunk chaining for any peer that needs it.
  void processVMMChains() {
    std::unique_lock l(mutex);
    for (size_t i = 0; i < vmmPeers.size(); ++i) {
      if (vmmPeers[i].vmmChainPending) {
        vmmPeers[i].vmmChainPending = false;
        l.unlock();
        sendNextVMMChunk(i);
        l.lock();
      }
    }
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
      s->socket.setOnRead([s, &onRead](Error* error) {
        onRead(s, error);
      });
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

    // Establish persistent blocking Unix socket connections for VMM fd exchange.
    // Each process creates a listener and each peer connects to it.
    // The connection from peer becomes the recv fd for that peer.
    // Our outgoing connection to peer becomes the send fd for that peer.
    for (size_t i = 0; i != vmmSendFds.size(); ++i) {
      vmmSendFds[i] = -1;
      vmmRecvFds[i] = -1;
    }

    // Check if fabric handles are available — probe with an actual cuMemCreate
    // since the device attribute alone doesn't guarantee IMEX is running.
    useFabric = group->supportsFabric;

    if (!useFabric) {
      std::string vmmAddress = fmt::sprintf("ipc-vmm-%d-%s", ::getpid(), randomName());

      // Create listener
      int listenFd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
      CHECK(listenFd >= 0);

      {
        sockaddr_un sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sun_family = AF_UNIX;
        sa.sun_path[0] = '\0'; // Abstract namespace
        std::string path = "moolib-" + vmmAddress;
        size_t len = std::min(path.size(), sizeof(sa.sun_path) - 2);
        std::memcpy(&sa.sun_path[1], path.data(), len);
        int rc = ::bind(listenFd, (const sockaddr*)&sa, sizeof(sa));
        CHECK(rc == 0);
        rc = ::listen(listenFd, 16);
        CHECK(rc == 0);
      }

      // Exchange VMM socket addresses
      for (size_t i : group->ipcRanks) {
        group->setupComms->sendTo(i, vmmAddress);
      }

      // Connect to each peer's listener (creates our send fd for that peer)
      auto vmmData = serializeToBuffer(signature, (uint32_t)group->rank);
      for (size_t i : group->ipcRanks) {
        std::string peerAddr = group->setupComms->recvFrom<std::string>(i);
        size_t peerIndex = group->getPeerIndex(i);

        int sockFd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        CHECK(sockFd >= 0);

        sockaddr_un sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sun_family = AF_UNIX;
        sa.sun_path[0] = '\0';
        std::string path = "moolib-" + peerAddr;
        size_t len = std::min(path.size(), sizeof(sa.sun_path) - 2);
        std::memcpy(&sa.sun_path[1], path.data(), len);

        int rc;
        do {
          rc = ::connect(sockFd, (const sockaddr*)&sa, sizeof(sa));
        } while (rc == -1 && errno == EINTR);
        CHECK(rc == 0);

        // Send our rank for identification
        ssize_t n;
        do {
          n = ::send(sockFd, vmmData->data(), vmmData->size(), 0);
        } while (n == -1 && errno == EINTR);
        CHECK(n == (ssize_t)vmmData->size());

        vmmSendFds[peerIndex] = sockFd;
      }

      // Accept connections from all peers (creates recv fd for each peer)
      for (size_t c = 0; c < group->ipcRanks.size(); ++c) {
        int connFd;
        do {
          connFd = ::accept(listenFd, nullptr, nullptr);
        } while (connFd == -1 && errno == EINTR);
        CHECK(connFd >= 0);

        // Read the rank identification
        char buf[12];
        size_t totalRead = 0;
        while (totalRead < sizeof(buf)) {
          ssize_t n;
          do {
            n = ::recv(connFd, buf + totalRead, sizeof(buf) - totalRead, 0);
          } while (n == -1 && errno == EINTR);
          CHECK(n > 0);
          totalRead += n;
        }

        uint64_t sig;
        uint32_t sourceRank;
        std::memcpy(&sig, buf, 8);
        std::memcpy(&sourceRank, buf + 8, 4);
        CHECK(sig == signature);

        size_t peerIndex = group->getPeerIndex(sourceRank);
        vmmRecvFds[peerIndex] = connFd;
      }

      ::close(listenFd);
    } // !useFabric

    thread = std::thread([this] {
      entry();
    });
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
      cpu_pause();
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
      cpu_pause();
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
      cpu_pause();
    }
    queue.data[queue.front % queue.data.size()] = value;
    ++queue.front;
  }
  uintptr_t popEventQueue(size_t peerIndex) {
    auto& queue = shared->eventQueue[peerIndex];
    while (queue.front == queue.back) {
      cpu_pause();
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

void IpcMapper::sendRequestVMM(size_t peerIndex, CUmemGenericAllocationHandle handle, size_t handleSize, size_t offset,
    Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendRequestVMM(peerIndex, handle, handleSize, offset, std::move(callback));
}

void IpcMapper::sendNextVMMChunk(size_t peerIndex) {
  ((IpcMapperImpl*)this)->sendNextVMMChunk(peerIndex);
}

void IpcMapper::sendMulticastHandle(
    size_t peerIndex, CUmemGenericAllocationHandle mcHandle, size_t size, Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendMulticastHandle(peerIndex, mcHandle, size, std::move(callback));
}

void IpcMapper::sendMulticastBind(
    size_t peerIndex, size_t handleIndex, size_t size, Function<void(uintptr_t)> callback) {
  ((IpcMapperImpl*)this)->sendMulticastBind(peerIndex, handleIndex, size, std::move(callback));
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
