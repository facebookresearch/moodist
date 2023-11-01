#pragma once

#include "common.h"
#include "function.h"
#include "hash_map.h"
#include "synchronization.h"

namespace moodist {

struct Group;
struct IpcMapper {
  Group* group;

  SpinMutex mutex;
  int requestNum = 0;

  std::atomic_int waitCount = 0;

  struct Mapped {
    uintptr_t peerAddress;
    uintptr_t localAddress;
    uintptr_t size;
    bool unmappable;
  };

  std::array<HashMap<CUipcMemHandle, Mapped, IpcMemHash, IpcMemEqual>, 8> peerIpcMap;
  std::array<HashMap<uintptr_t, std::pair<uintptr_t, unsigned long long>>, 8> peerIpcAddressMap;

  std::atomic_bool hasException = false;
  std::optional<std::exception_ptr> exception;

  virtual ~IpcMapper() {}

  void init();

  void
  sendRequestAddress(size_t peerIndex, const CUipcMemHandle& handle, size_t size, Function<void(uintptr_t)> callback);

  void sendRequestUnmap(size_t peerIndex, uintptr_t base, size_t size, Function<void(uintptr_t)> callback);

  void* getMySharedMem(size_t offset, size_t size);
  void* getPeerSharedMem(size_t peerIndex, size_t offset, size_t size);

  void unmapAll() {
    std::unique_lock l(mutex);
    fmt::printf("ipc mapper unmapping all memory!\n");
    while (true) {
      bool stop = true;
      for (size_t peerIndex = 0; peerIndex != peerIpcMap.size(); ++peerIndex) {
        peerIpcAddressMap[peerIndex].clear();
        auto& ipcMap = peerIpcMap[peerIndex];
        for (auto i = ipcMap.begin(); i != ipcMap.end(); ++i) {
          if (!i->second.unmappable) {
            continue;
          }
          fmt::printf(
              "requestAddress: requesting unmap of %#x bytes at %#x (mapped at %#x)!\n", i->second.size,
              i->second.localAddress, i->second.peerAddress);
          ++waitCount;
          uintptr_t peerAddress = i->second.peerAddress;
          size_t size = i->second.size;
          ipcMap.erase(i);
          l.unlock();
          stop = false;
          sendRequestUnmap(peerIndex, peerAddress, size, [this](uintptr_t) { --waitCount; });
          l.lock();
          break;
        }
      }
      if (stop) {
        break;
      }
    }
    l.unlock();
    auto start = Clock::now();
    while (waitCount.load(std::memory_order_relaxed) && Clock::now() - start < std::chrono::milliseconds(5));
  }

  template<typename Callback>
  void
  requestAddress(size_t peerIndex, uintptr_t address, size_t length, Callback&& callback, bool unmappable = false) {
    unsigned long long bufferId = -1;
    CHECK_CU(cuPointerGetAttribute(&bufferId, CU_POINTER_ATTRIBUTE_BUFFER_ID, address));
    CHECK(bufferId != -1);
    unsigned long long bufferId2 = -1;
    CHECK_CU(cuPointerGetAttribute(&bufferId2, CU_POINTER_ATTRIBUTE_BUFFER_ID, address + length - 1));
    CHECK(bufferId == bufferId2);
    std::unique_lock l(mutex);
    auto& addressMap = peerIpcAddressMap[peerIndex];
    auto i = addressMap.find(address);
    if (i != addressMap.end()) {
      if (i->second.second == bufferId) {
        // fmt::printf("requestAddress: %#x bytes at %#x is already mapped at %#x\n", length, address, i->second.first);
        callback(i->second.first);
        return;
      }
      addressMap.erase(i);
      fmt::printf("requestAddress: bufferId changed for %#x bytes at %#x\n", length, address);
    }

    CUdeviceptr base = 0;
    size_t size = 0;
    CHECK_CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)address));
    CHECK(size >= length);
    fmt::printf(
        "requestAddress: %#x bytes at %#x is part of allocation of %#x bytes at %#x\n", length, address, size, base);
    CUipcMemHandle handle;
    CHECK_CU(cuIpcGetMemHandle(&handle, base));
    size_t offset = address - base;
    uintptr_t baseAddress = peerIpcMap[peerIndex][handle].peerAddress;
    if (baseAddress) {
      fmt::printf(
          "requestAddress: (allocation mapped) %#x bytes at %#x is already mapped at %#x (offset %#x)\n", length,
          address, baseAddress + offset, offset);
      addressMap[address] = {baseAddress + offset, bufferId};
      l.unlock();
      callback(baseAddress + offset);
    } else {
      while (true) {
        bool anyUnmaps = false;
        auto& ipcMap = peerIpcMap[peerIndex];
        for (auto i = ipcMap.begin(); i != ipcMap.end(); ++i) {
          if (i->second.localAddress + i->second.size > base && i->second.localAddress < base + size) {
            fmt::printf(
                "requestAddress: requesting unmap of %#x bytes at %#x (mapped at %#x) due to allocations changing!\n",
                i->second.size, i->second.localAddress, i->second.peerAddress);
            anyUnmaps = true;
            ++waitCount;
            uintptr_t peerAddress = i->second.peerAddress;
            size_t size = i->second.size;
            ipcMap.erase(i);
            l.unlock();
            sendRequestUnmap(peerIndex, peerAddress, size, [this](uintptr_t) { --waitCount; });
            break;
          }
        }
        if (!anyUnmaps) {
          break;
        }
        wait();
        l.lock();
      }
      l.unlock();
      ++waitCount;
      sendRequestAddress(
          peerIndex, handle, size,
          [this, peerIndex, address, handle, offset, bufferId, length, base, size,
           callback = std::forward<Callback>(callback), unmappable](uintptr_t mappedAddress) {
            fmt::printf(
                "requestAddress: new mapping -> %#x bytes at %#x mapped at %#x (offset %#x)\n", length, address,
                mappedAddress + offset, offset);
            peerIpcAddressMap[peerIndex][address] = {mappedAddress + offset, bufferId};
            auto& v = peerIpcMap[peerIndex][handle];
            v.localAddress = base;
            v.peerAddress = mappedAddress;
            v.size = size;
            v.unmappable = unmappable;
            callback(mappedAddress + offset);
            --waitCount;
          });
    }
  }

  void wait() {
    while (waitCount.load(std::memory_order_relaxed)) {
      if (hasException.load(std::memory_order_relaxed)) {
        std::lock_guard l(mutex);
        std::rethrow_exception(*exception);
      }
    }
  }
};

std::unique_ptr<IpcMapper> createIpcMapper(Group* group);

} // namespace moodist
