// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "api/allocator.h"
#include "api/allocator_api.h"
#include "common.h"
#include "cputhread.h"
#include "function.h"
#include "hash_map.h"
#include "synchronization.h"

#include <optional>
#include <type_traits>

#include "group.h"

namespace moodist {

struct Group;
struct IpcMapper {
  Group* group;

  SpinMutex mutex;
  int requestNum = 0;
  SpinMutex unmapMutex;

  std::atomic_int waitCount = 0;

  uint32_t stepValue = 0;

  struct Mapped {
    uintptr_t peerAddress;
    uintptr_t localAddress;
    uintptr_t size;
    bool unmappable;
    unsigned long bufferId;
  };

  std::array<HashMap<CUipcMemHandle, Mapped, IpcMemHash, IpcMemEqual>, 8> peerIpcMap;
  std::array<HashMap<uintptr_t, std::pair<uintptr_t, unsigned long long>>, 8> peerIpcAddressMap;
  std::array<HashMap<CUevent, uintptr_t>, 8> peerIpcEventMap;

  std::array<HashMap<CUipcMemHandle, bool, IpcMemHash, IpcMemEqual>, 8> peerQueuedUnmaps;

  std::atomic_bool hasException = false;
  std::optional<std::exception_ptr> exception;

  std::atomic_bool hasQueuedUnmaps = false;

  struct MapCallbacks {
    struct ListEntry {
      Function<void(uintptr_t)> callback;
      size_t offset;
      uintptr_t address;
    };
    std::vector<ListEntry> list;
    uintptr_t base;
    size_t size;
    unsigned long bufferId;
    CUevent event;
  };

  std::array<HashMap<CUipcMemHandle, MapCallbacks, IpcMemHash, IpcMemEqual>, 8> peerMapCallbacks;

  std::array<HashMap<CUipcEventHandle, MapCallbacks, IpcMemHash, IpcMemEqual>, 8> peerEventMapCallbacks;

  struct VMMPeerState {
    uintptr_t peerBase = 0;      // VA base the peer reserved for our allocator memory
    size_t mappedWatermark = 0;  // Bytes the peer has mapped so far
    size_t mappedChunkCount = 0; // Number of our chunks the peer has mapped
    bool mappingInProgress = false;
    bool vmmChainPending = false; // Set by callback to trigger next chunk outside mutex
    size_t pendingNeeded = 0;
    struct PendingCallback {
      Function<void(uintptr_t)> callback;
      size_t offset;
    };
    std::vector<PendingCallback> pendingCallbacks;
  };
  std::array<VMMPeerState, 8> vmmPeers;

  struct FabricMapState {
    uintptr_t base = 0;
    uintptr_t size = 0;
    size_t nextChunkIndex = 0;
    struct Waiter {
      Function<void()> callback;
    };
    Vector<size_t> inFlightChunks;
    Vector<Waiter> waiters;
  };
  Vector<FabricMapState> fabricMap;

  virtual ~IpcMapper() {}

  void init(int node);

  void sendRequestAddress(
      size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback);
  void sendRequestEvent(size_t peerIndex, const CUipcEventHandle& handle, Function<void(uintptr_t)> callback);

  void sendRequestUnmap(size_t peerIndex, uintptr_t base, size_t size, Function<void(uintptr_t)> callback);

  void sendRequestVMM(size_t peerIndex, CUmemGenericAllocationHandle handle, size_t handleSize, size_t offset,
      Function<void(uintptr_t)> callback);

  void sendNextVMMChunk(size_t peerIndex);

  // Phase 1: Share a multicast handle with a peer. The peer's handler thread
  // will import the handle and call cuMulticastAddDevice. The callback receives
  // a handle index that must be passed to sendMulticastBind in Phase 2.
  // ALL devices must complete addDevice before ANY device can bind.
  void sendMulticastHandle(
      size_t peerIndex, CUmemGenericAllocationHandle mcHandle, size_t size, Function<void(uintptr_t)> callback);

  // Phase 2: Tell a peer to allocate scratch, bind, and map a multicast object
  // that was previously registered via sendMulticastHandle. handleIndex is the
  // value returned by the Phase 1 callback. Callback receives the peer's multicast VA.
  void sendMulticastBind(size_t peerIndex, size_t handleIndex, size_t size, Function<void(uintptr_t)> callback);

  void* getMySharedMem(size_t offset, size_t size);
  void* getPeerSharedMem(size_t peerIndex, size_t offset, size_t size);

  void setStepValue(uint32_t stepValue) {
    std::lock_guard l(mutex);
    this->stepValue = stepValue;
  }

  void executeQueuedUnmaps() {
    std::unique_lock l(mutex);
    hasQueuedUnmaps = false;
    for (size_t peerIndex = 0; peerIndex != peerIpcMap.size(); ++peerIndex) {
      std::vector<CUipcMemHandle> unmapList;
      for (auto& v : peerQueuedUnmaps[peerIndex]) {
        unmapList.push_back(v.first);
      }
      peerQueuedUnmaps[peerIndex].clear();
      if (!unmapList.empty()) {
        tryToUnmapList(peerIndex, unmapList, l);
      }
    }
  }

  void enqueueUnmapAll() {
    std::unique_lock l(mutex);
    for (size_t peerIndex = 0; peerIndex != peerIpcMap.size(); ++peerIndex) {
      auto& ipcMap = peerIpcMap[peerIndex];
      for (auto i = ipcMap.begin(); i != ipcMap.end(); ++i) {
        if (!i->second.unmappable) {
          continue;
        }
        peerQueuedUnmaps[peerIndex].emplace(i->first, true);
        hasQueuedUnmaps = true;
      }
    }
  }

  void tryToUnmapList(size_t peerIndex, const std::vector<CUipcMemHandle>& unmapList, std::unique_lock<SpinMutex>& l) {
    CHECK(l.owns_lock());
    if (unmapList.empty()) {
      return;
    }
    auto& ipcMap = peerIpcMap[peerIndex];
    for (auto& v : unmapList) {
      auto i = ipcMap.find(v);
      if (i == ipcMap.end()) {
        continue;
      }
      log.debug("tryToUnmapList: requesting unmap of %#x bytes at %#x (mapped at %#x)!\n", i->second.size,
          i->second.localAddress, i->second.peerAddress);
      ++waitCount;
      uintptr_t peerAddress = i->second.peerAddress;
      size_t size = i->second.size;
      l.unlock();
      sendRequestUnmap(peerIndex, peerAddress, size, [this, v, &ipcMap, peerIndex](uintptr_t response) {
        // mutex is held
        auto i = ipcMap.find(v);
        if (i != ipcMap.end()) {
          if (response) {
            peerIpcAddressMap[peerIndex].clear();
            ipcMap.erase(i);
          } else {
            log.error("unmap failed!\n");
          }
        }
        --waitCount;
      });
      l.lock();
    }
    l.unlock();
    auto start = std::chrono::steady_clock::now();
    while (waitCount.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      auto now = std::chrono::steady_clock::now();
      if (now - start >= std::chrono::seconds(60)) {
        log.error("Timeout waiting for ipc unmap!\n");
        break;
      }
    }
    l.lock();
  }

  template<typename Callback>
  void requestAddress(
      size_t peerIndex, uintptr_t address, size_t length, Callback&& callback, bool unmappable = false) {
    CHECK(length > 0);
    CHECK(address != 0);

    unsigned long long bufferId = -1;
    CUdeviceptr base = 0;
    size_t size = 0;

    uintptr_t srcBase = allocatorGetReservedBase();

    auto myRegion = allocator::mappedRegion(address);

    if (myRegion.first && srcBase != 0) {
      // VMM path: watermark-based mapping
      size_t offset = address - srcBase;
      size_t needed = offset + length;

      std::unique_lock l(mutex);
      auto& peer = vmmPeers[peerIndex];

      // Fast path: already mapped enough
      if (needed <= peer.mappedWatermark && peer.peerBase != 0) {
        l.unlock();
        callback(peer.peerBase + offset);
        return;
      }

      // Queue this callback
      peer.pendingCallbacks.push_back({std::move(callback), offset});
      if (needed > peer.pendingNeeded) {
        peer.pendingNeeded = needed;
      }

      if (!peer.mappingInProgress) {
        peer.mappingInProgress = true;
        ++waitCount;
        l.unlock();
        sendNextVMMChunk(peerIndex);
      }
      return;
    } else if (myRegion.first) {
      // Allocator memory but VMM not initialized — shouldn't happen
      bufferId = myRegion.first;
      base = myRegion.first;
      size = myRegion.second;
    } else {
      CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
      CHECK(bufferId != -1);
      unsigned long long bufferId2 = -1;
      CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + length - 1));
      CHECK(bufferId == bufferId2);
      try {
        CHECK_CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)address));
      } catch (const std::exception& e) {
        log.error("requestAddress: %#x bytes at %#x (buffer id %d), error %s\n", length, address, bufferId, e.what());
        throw;
      }
    }
    std::unique_lock l(mutex);
    auto& addressMap = peerIpcAddressMap[peerIndex];
    auto i = addressMap.find(base);
    if (i != addressMap.end()) {
      if (i->second.second == bufferId) {
        size_t offset = address - base;
        callback(i->second.first + offset);
        return;
      }
      addressMap.erase(i);
      log.debug("requestAddress: bufferId changed for %#x bytes at %#x\n", length, address);
    }

    CHECK(size >= length);
    size_t offset = address - base;

    {
      // Legacy IPC path: cuIpcGetMemHandle
      log.debug("requestAddress: %#x bytes at %#x is part of allocation of %#x bytes at %#x (buffer id %d)\n", length,
          address, size, base, bufferId);
      CUipcMemHandle handle;
      CHECK_CU(cuIpcGetMemHandle(&handle, base));
      Mapped& mapped = peerIpcMap[peerIndex][handle];
      uintptr_t baseAddress = mapped.peerAddress;
      if (baseAddress) {
        log.debug("requestAddress: (allocation mapped) %#x bytes at %#x is already mapped at %#x (offset %#x)\n",
            length, address, baseAddress + offset, offset);
        CHECK(mapped.localAddress == base);
        CHECK(mapped.size == size);
        CHECK(mapped.bufferId == bufferId);
        CHECK(mapped.unmappable == unmappable);
        addressMap[base] = {baseAddress, bufferId};
        l.unlock();
        callback(baseAddress + offset);
      } else {
        auto it = peerMapCallbacks[peerIndex].find(handle);
        if (it != peerMapCallbacks[peerIndex].end()) {
          CHECK(waitCount > 0);
          auto& q = it->second;
          CHECK(q.bufferId == bufferId);
          CHECK(q.base == base);
          CHECK(q.size == size);
          q.list.emplace_back();
          auto& e = q.list.back();
          e.callback = std::move(callback);
          e.offset = offset;
          e.address = address;
          log.debug("requestAddress: already being mapped, adding callback\n");
          return;
        }
        std::vector<CUipcMemHandle> unmapList;
        auto& ipcMap = peerIpcMap[peerIndex];
        for (auto i = ipcMap.begin(); i != ipcMap.end(); ++i) {
          if (i->second.localAddress + i->second.size > base && i->second.localAddress < base + size) {
            log.debug(
                "requestAddress: enqueueing unmap of %#x bytes at %#x (mapped at %#x) due to allocations changing!\n",
                i->second.size, i->second.localAddress, i->second.peerAddress);
            peerQueuedUnmaps[peerIndex].emplace(i->first, true);
            hasQueuedUnmaps = true;
          }
        }
        auto& q = peerMapCallbacks[peerIndex][handle];
        q.bufferId = bufferId;
        q.base = base;
        q.size = size;
        q.list.emplace_back();
        auto& e = q.list.back();
        e.callback = std::move(callback);
        e.offset = offset;
        e.address = address;
        l.unlock();
        ++waitCount;
        sendRequestAddress(peerIndex, handle, size,
            [this, peerIndex, handle, bufferId, length, base, size, unmappable](uintptr_t mappedAddress) {
              auto& v = peerIpcMap[peerIndex][handle];
              v.localAddress = base;
              v.peerAddress = mappedAddress;
              v.size = size;
              v.unmappable = unmappable;
              v.bufferId = bufferId;
              auto it = peerMapCallbacks[peerIndex].find(handle);
              CHECK(it != peerMapCallbacks[peerIndex].end());
              auto& q = it->second;
              CHECK(q.bufferId == bufferId);
              CHECK(!q.list.empty());
              for (auto& e : q.list) {
                log.debug("requestAddress: new mapping -> %#x bytes at %#x mapped at %#x (offset %#x)\n", length,
                    e.address, mappedAddress + e.offset, e.offset);
                peerIpcAddressMap[peerIndex][e.address] = {mappedAddress, bufferId};
                std::move(e.callback)(mappedAddress + e.offset);
              }
              peerMapCallbacks[peerIndex].erase(it);
              --waitCount;
            });
      }
    }
  }

  void requestAddress(size_t peerIndex, uintptr_t address, size_t length, uintptr_t* ptr, bool unmappable = false) {
    return requestAddress(
        peerIndex, address, length,
        [ptr](uintptr_t value) {
          *ptr = value;
        },
        unmappable);
  }

  void requestAddressRank(size_t rank, uintptr_t address, size_t length, uintptr_t* ptr) {
    if (group->ipcAccess.at(rank)) {
      return requestAddress(group->getPeerIndex(rank), address, length, ptr);
    }
    uintptr_t base = allocatorGetReservedBase();
    auto region = allocator::mappedRegion(address);
    CHECK(base != 0);
    if (region.first == 0) {
      throw std::runtime_error(
          fmt::sprintf("Attempting to fabric map memory (%#x) which is not owned by the moodist cuda allocator. This "
                       "is not supported - enable the moodist cuda allocator through moodist.enable_cuda_allocator().",
              address));
    }
    std::lock_guard l(mutex);
    if (fabricMap.size() <= rank) {
      fabricMap.resize(group->size);
    }
    uintptr_t addressOffset = address - base;
    uintptr_t regionOffset = region.first - base;
    uintptr_t regionEnd = regionOffset + region.second;
    auto& im = fabricMap.at(rank);
    if (im.inFlightChunks.empty() && im.base != 0) {
      if (im.size >= regionEnd) {
        *ptr = im.base + addressOffset;
        return;
      }
    }
    while (im.size < regionEnd) {
      size_t i = im.nextChunkIndex;
      unsigned long long chunkHandle;
      size_t chunkOffset;
      size_t chunkSize;
      CHECK(allocatorGetChunk(i, &chunkHandle, &chunkOffset, &chunkSize));
      CHECK(chunkOffset == im.size);

      CUmemFabricHandle fabricHandle;
      CHECK_CU(cuMemExportToShareableHandle(&fabricHandle, chunkHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));

      log.info("sending export handle yey, chunk %d at %d+%d to %d\n", i, chunkOffset, chunkSize, rank);

      CHECK(!std::ranges::contains(im.inFlightChunks, i));
      im.inFlightChunks.push_back(i);
      im.nextChunkIndex = i + 1;
      im.size += chunkSize;

      QueueEntryFabricMap* e = group->cpuThread->freelistFabricMap.pop();
      e->task = taskFabricMap;
      e->targetRank = rank;
      e->chunkIndex = i;
      e->handle = fabricHandle;
      e->offset = chunkOffset;
      e->bytes = chunkSize;
      group->cpuThread->enqueue(e);
    }
    ++waitCount;
    FabricMapState::Waiter w;
    w.callback = [this, ptr, rank, addressOffset, length]() {
      auto& im = fabricMap[rank];
      CHECK(im.base != 0);
      CHECK(im.size >= addressOffset + length);
      *ptr = im.base + addressOffset;
      --waitCount;
    };
    im.waiters.push_back(w);
  }

  template<typename Callback>
  void requestEvent(size_t peerIndex, CUevent event, Callback&& callback) {
    CHECK(sizeof(CUevent) == sizeof(uintptr_t));
    std::unique_lock l(mutex);
    auto& eventMap = peerIpcEventMap[peerIndex];
    auto i = eventMap.find(event);
    if (i != eventMap.end()) {
      callback(i->second);
      return;
    }
    CUipcEventHandle handle;
    CHECK_CU(cuIpcGetEventHandle(&handle, event));
    auto it = peerEventMapCallbacks[peerIndex].find(handle);
    if (it != peerEventMapCallbacks[peerIndex].end()) {
      CHECK(waitCount > 0);
      auto& q = it->second;
      CHECK(q.event == event);
      q.list.emplace_back();
      auto& e = q.list.back();
      e.callback = std::move(callback);
      log.debug("requestEvent: already being mapped, adding callback\n");
      return;
    }
    auto& q = peerEventMapCallbacks[peerIndex][handle];
    q.event = event;
    q.list.emplace_back();
    auto& e = q.list.back();
    e.callback = std::move(callback);
    l.unlock();
    ++waitCount;
    sendRequestEvent(peerIndex, handle, [this, peerIndex, handle, event](uintptr_t mappedAddress) {
      auto it = peerEventMapCallbacks[peerIndex].find(handle);
      CHECK(it != peerEventMapCallbacks[peerIndex].end());
      auto& q = it->second;
      CHECK(q.event == event);
      CHECK(!q.list.empty());
      for (auto& e : q.list) {
        log.debug(
            "requestEvent: new mapping -> event at %#x mapped at %#x\n", (uintptr_t)event, mappedAddress, e.offset);
        peerIpcEventMap[peerIndex][event] = mappedAddress;
        std::move(e.callback)(mappedAddress);
      }
      peerEventMapCallbacks[peerIndex].erase(it);
      --waitCount;
    });
  }

  void requestEvent(size_t peerIndex, CUevent event, uintptr_t* ptr) {
    return requestEvent(peerIndex, event, [ptr](uintptr_t value) {
      *ptr = value;
    });
  }

  void wait() {
    while (waitCount.load(std::memory_order_relaxed)) {
      if (hasException.load(std::memory_order_relaxed)) {
        std::lock_guard l(mutex);
        std::rethrow_exception(*exception);
      }
    }
    for (auto& v : peerMapCallbacks) {
      CHECK(v.empty());
    }
  }

  void push(size_t peerIndex, const void* ptr, size_t n);
  void pop(size_t peerIndex, void* ptr, size_t n);

  template<typename T>
  void push(size_t peerIndex, const T& value) {
    static_assert(std::is_trivially_copy_constructible_v<T>);
    push(peerIndex, &value, sizeof(value));
  }
  template<typename T>
  T pop(size_t peerIndex) {
    static_assert(std::is_trivially_copy_constructible_v<T>);
    T r;
    pop(peerIndex, &r, sizeof(r));
    return r;
  }

  void pushEvent(size_t peerIndex, CUevent event) {
    static_assert(sizeof(CUevent) == sizeof(uintptr_t));
    requestEvent(peerIndex, event, [this, peerIndex](uintptr_t value) {
      push(peerIndex, value);
    });
    wait();
  }
  CUevent popEvent(size_t peerIndex) {
    return pop<CUevent>(peerIndex);
  }

  void streamRecord(size_t peerIndex, CUstream stream);
  void streamWait(size_t peerIndex, CUstream stream);
};

std::unique_ptr<IpcMapper> createIpcMapper(Group* group);

} // namespace moodist
