#pragma once

#include "common.h"
#include "function.h"
#include "hash_map.h"
#include "synchronization.h"
#include <optional>
#include <type_traits>

// #include "allocator.h"
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

  // std::array<std::string, 8> peerMemoryId{};
  // std::array<allocator::PeerMemory, 8> peerMemory{};
  // std::array<uintptr_t, 8> peerMemoryBase{};
  // std::array<uintptr_t, 8> peerMemorySize{};

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

  virtual ~IpcMapper() {}

  void init();

  void
  sendRequestAddress(size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback);
  void sendRequestEvent(size_t peerIndex, const CUipcEventHandle& handle, Function<void(uintptr_t)> callback);

  void sendRequestUnmap(size_t peerIndex, uintptr_t base, size_t size, Function<void(uintptr_t)> callback);

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
      log.debug(
          "tryToUnmapList: requesting unmap of %#x bytes at %#x (mapped at %#x)!\n", i->second.size,
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
    auto start = Clock::now();
    while (waitCount.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (Clock::now() - start >= std::chrono::seconds(60)) {
        log.error("Timeout waiting for ipc unmap!\n");
        start = Clock::now();
      }
    }
    l.lock();
  }

  template<typename Callback>
  void
  requestAddress(size_t peerIndex, uintptr_t address, size_t length, Callback&& callback, bool unmappable = false) {
    CHECK(length > 0);
    CHECK(address != 0);

    // if (allocator::owns(address)) {
    //   std::unique_lock l(mutex);
    //   uintptr_t offset = allocator::offset(address);
    //   if (offset + length <= peerMemorySize[peerIndex]) {
    //     callback(peerMemoryBase[peerIndex] + offset);
    //     return;
    //   }
    //   if (!peerMemory[peerIndex]) {
    //     peerMemory[peerIndex] = allocator::getPeerMemory(peerMemoryId[peerIndex]);
    //     peerMemoryBase[peerIndex] = peerMemory[peerIndex].remoteBaseAddress();
    //   }
    //   peerMemory[peerIndex].remoteExtend(offset + length);
    //   callback(peerMemoryBase[peerIndex] + offset);
    //   return;
    // }

    unsigned long long bufferId = -1;
    CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
    CHECK(bufferId != -1);
    unsigned long long bufferId2 = -1;
    CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + length - 1));
    CHECK(bufferId == bufferId2);
    CUdeviceptr base = 0;
    size_t size = 0;
    try {
      CHECK_CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)address));
    } catch (const std::exception& e) {
      log.error("requestAddress: %#x bytes at %#x (buffer id %d), error %s\n", length, address, bufferId, e.what());
      throw;
    }
    std::unique_lock l(mutex);
    auto& addressMap = peerIpcAddressMap[peerIndex];
    auto i = addressMap.find(base);
    if (i != addressMap.end()) {
      if (i->second.second == bufferId) {
        // fmt::printf("requestAddress: %#x bytes at %#x is already mapped at %#x\n", length, address, i->second.first);
        size_t offset = address - base;
        callback(i->second.first + offset);
        return;
      }
      addressMap.erase(i);
      log.debug("requestAddress: bufferId changed for %#x bytes at %#x\n", length, address);
    }

    CHECK(size >= length);
    log.debug(
        "requestAddress: %#x bytes at %#x is part of allocation of %#x bytes at %#x (buffer id %d)\n", length, address,
        size, base, bufferId);
    CUipcMemHandle handle;
    CHECK_CU(cuIpcGetMemHandle(&handle, base));
    size_t offset = address - base;
    Mapped& mapped = peerIpcMap[peerIndex][handle];
    uintptr_t baseAddress = mapped.peerAddress;
    if (baseAddress) {
      log.debug(
          "requestAddress: (allocation mapped) %#x bytes at %#x is already mapped at %#x (offset %#x)\n", length,
          address, baseAddress + offset, offset);
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
      sendRequestAddress(
          peerIndex, handle, size,
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
              log.debug(
                  "requestAddress: new mapping -> %#x bytes at %#x mapped at %#x (offset %#x)\n", length, e.address,
                  mappedAddress + e.offset, e.offset);
              peerIpcAddressMap[peerIndex][e.address] = {mappedAddress, bufferId};
              std::move(e.callback)(mappedAddress + e.offset);
            }
            peerMapCallbacks[peerIndex].erase(it);
            --waitCount;
          });
    }
  }

  void requestAddress(size_t peerIndex, uintptr_t address, size_t length, uintptr_t* ptr, bool unmappable = false) {
    return requestAddress(peerIndex, address, length, [ptr](uintptr_t value) { *ptr = value; }, unmappable);
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
    return requestEvent(peerIndex, event, [ptr](uintptr_t value) { *ptr = value; });
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
    requestEvent(peerIndex, event, [this, peerIndex](uintptr_t value) { push(peerIndex, value); });
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
