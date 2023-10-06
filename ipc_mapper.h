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

  std::array<HashMap<CUipcMemHandle, uintptr_t, IpcMemHash, IpcMemEqual>, 8> peerIpcMap;
  std::array<HashMap<uintptr_t, uintptr_t>, 8> peerIpcAddressMap;

  std::atomic_bool hasException = false;
  std::optional<std::exception_ptr> exception;

  virtual ~IpcMapper() {}

  void init();

  void sendRequestAddress(size_t peerIndex, const CUipcMemHandle& handle, Function<void(uintptr_t)> callback);

  void* getMySharedMem(size_t offset, size_t size);
  void* getPeerSharedMem(size_t peerIndex, size_t offset, size_t size);

  template<typename Callback>
  void requestAddress(size_t peerIndex, uintptr_t address, size_t length, Callback&& callback) {
    std::unique_lock l(mutex);
    uintptr_t retAddress = peerIpcAddressMap[peerIndex][address];
    if (retAddress) {
      callback(retAddress);
      return;
    }

    CUdeviceptr base = 0;
    size_t size = 0;
    CHECK_CU(cuMemGetAddressRange(&base, &size, (CUdeviceptr)address));
    TORCH_CHECK(size >= length);
    CUipcMemHandle handle;
    CHECK_CU(cuIpcGetMemHandle(&handle, base));
    size_t offset = address - base;
    uintptr_t baseAddress = peerIpcMap[peerIndex][handle];
    l.unlock();
    if (baseAddress) {
      callback(baseAddress + offset);
    } else {
      ++waitCount;
      sendRequestAddress(
          peerIndex, handle,
          [this, peerIndex, address, handle, offset,
           callback = std::forward<Callback>(callback)](uintptr_t mappedAddress) {
            peerIpcMap[peerIndex][handle] = mappedAddress;
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
