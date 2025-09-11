
#include "rdma.h"
#include "serialization.h"
#include "tcpdev.h"

namespace moodist {

namespace {

struct MrInfo {
  bool active = false;
  bool cpu;
  void* address;
  size_t bytes;
  std::atomic_size_t refcount = 0;
};

struct MrManager {
  SpinMutex mutex;
  IVector<MrInfo> mrs;
  IVector<uint32_t> freeIndices;

  uint32_t reg(bool cpu, void* address, size_t bytes) {
    std::lock_guard l(mutex);
    uint32_t r = 0;
    if (freeIndices.empty()) {
      while (r == 0) {
        r = mrs.size();
        mrs.emplace_back();
      }
    } else {
      r = freeIndices.pop_front_value();
    }
    CHECK(r != 0);
    auto& i = mrs[r];
    CHECK(!i.active);
    CHECK(i.refcount == 0);
    i.active = true;
    i.cpu = cpu;
    i.address = address;
    i.bytes = bytes;
    return r;
  }

  void dereg(uint32_t lkey) {
    std::lock_guard l(mutex);
    CHECK(lkey < mrs.size());
    CHECK(mrs[lkey].active);
    while (mrs[lkey].refcount) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    mrs[lkey].active = false;
    freeIndices.push_back(lkey);
  }
} mrManager;

struct MrHandle {
  uint32_t key = 0;
  MrHandle() = default;
  MrHandle(uint32_t key) : key(key) {
    ++mrManager.mrs[key].refcount;
  }
  MrHandle(MrHandle&& n) {
    key = std::exchange(n.key, 0);
  }
  MrHandle& operator=(MrHandle&& n) {
    std::swap(key, n.key);
    return *this;
  }
  ~MrHandle() {
    if (key) {
      CHECK((int32_t)mrManager.mrs[key].refcount > 0);
      --mrManager.mrs[key].refcount;
    }
  }
  operator bool() const {
    return key != 0;
  }
};

struct Mr : RdmaMr {
  ~Mr() {
    mrManager.dereg(lkey);
  }
};

constexpr const char* typestr = "tcp";

constexpr uint8_t messageWrite = 1;
constexpr uint8_t messageCompletionSuccess = 2;
constexpr uint8_t messageCompletionError = 3;
constexpr uint8_t messageSend = 4;
constexpr uint8_t messageRead = 5;
constexpr uint8_t messageReadSuccess = 6;

struct RdmaTcp : Rdma {
  Group* group = nullptr;
  UniqueImpl<TcpDev> tcp = nullptr;
  RdmaCpuThreadApi cpuThreadApi;

  struct Recv {
    void* address;
    size_t bytes;
    RdmaCallback* callback;
    MrHandle mr;
  };
  struct QueuedRecvData {
    Vector<uint8_t> data;
    size_t i;
    size_t index;
  };

  SpinMutex mutex;
  Vector<RdmaCallback*> finished;
  Vector<RdmaCallback*> tmpFinished;
  std::atomic_size_t nFinished = 0;
  size_t nActive = 0;
  uint64_t activeIndex = 0;
  HashMap<uint64_t, RdmaCallback*> activeCallbacks;
  HashMap<uint64_t, Recv> activeReads;
  Vector<Recv> activeRecvs;
  Vector<QueuedRecvData> queuedRecvData;
  std::atomic_uint32_t completionError = (uint32_t)-1;
  std::atomic_size_t errorStateSourceRank = -1;
  std::atomic_bool errorState = false;

  SpinMutex recvCallbackMutex;

  RdmaTcp(Group* group) : group(group) {
    this->type = typestr;

    tcp = makeTcpDev(group, this);
  }

  ~RdmaTcp() {
    tcp = nullptr;
  }

  void setCpuThreadApi(RdmaCpuThreadApi cpuThreadApi) {
    this->cpuThreadApi = cpuThreadApi;
  }

  [[gnu::noinline]] [[gnu::cold]] void setError(size_t sourceRank = -1) {
    if (errorState) {
      return;
    }
    errorStateSourceRank = sourceRank;
    errorState = true;
    tcp->close();
  }

  void onRead(size_t i, BufferHandle buffer, void* handle) {
    CHECK(buffer->size() > 1);
    try {
      uint8_t type;
      auto view = deserializeBufferPart(buffer, type);
      switch (type) {
      case messageWrite: {
        uint64_t index;
        uint32_t rkey;
        void* address;
        size_t bytes;
        deserializeBuffer(view, index, rkey, address, bytes);
        if (auto mr = getMr(rkey, address, bytes)) {
          tcp->read(handle, address, bytes, [this, i, index, mr = std::move(mr)] {
            tcp->send(i, serializeToBuffer(messageCompletionSuccess, index), nullptr);
          });
        } else {
          tcp->send(i, serializeToBuffer(messageCompletionError, index), nullptr);
        }
        break;
      }
      case messageCompletionSuccess: {
        uint64_t index;
        deserializeBuffer(view, index);
        std::unique_lock l(mutex);
        auto i = activeCallbacks.find(index);
        CHECK(i != activeCallbacks.end());
        if (i != activeCallbacks.end()) {
          auto* callback = i->second;
          activeCallbacks.erase(i);
          finished.push_back(callback);
          l.unlock();
          ++nFinished;
        }
        break;
      }
      case messageCompletionError: {
        uint64_t index;
        deserializeBuffer(view, index);
        std::unique_lock l(mutex);
        auto it = activeCallbacks.find(index);
        if (it != activeCallbacks.end()) {
          completionError = i;
        } else {
          auto rit = activeReads.find(index);
          if (rit != activeReads.end()) {
            completionError = i;
          } else {
            log.error("TCP got completion error for unknown op\n");
            completionError = i;
          }
        }
        break;
      }
      case messageSend: {
        uint64_t index;
        size_t bytes;
        deserializeBuffer(view, index, bytes);
        std::unique_lock l(mutex);
        if (!activeRecvs.empty()) {
          auto x = activeRecvs.pop_front_value();
          std::unique_lock cl(recvCallbackMutex);
          l.unlock();
          if (x.bytes >= bytes) {
            tcp->read(handle, x.address, bytes, [this, i, index, callback = x.callback] {
              tcp->send(i, serializeToBuffer(messageCompletionSuccess, index), nullptr);
              callback->decref();
            });
          } else {
            log.error("TCP got RDMA send of %d bytes, but posted recv only has %d bytes\n", bytes, x.bytes);
            tcp->send(i, serializeToBuffer(messageCompletionError, index), nullptr);
            setError();
          }
        } else {
          Vector<uint8_t> data;
          data.resize(bytes);
          void* ptr = data.data();
          tcp->read(handle, ptr, bytes, [this, i, index, data = std::move(data)] {
            size_t bytes = data.size();
            std::unique_lock l(mutex);
            if (!activeRecvs.empty()) {
              auto x = activeRecvs.pop_front_value();
              std::unique_lock cl(recvCallbackMutex);
              l.unlock();
              if (x.bytes >= bytes) {
                std::memcpy(x.address, data.data(), bytes);
                tcp->send(i, serializeToBuffer(messageCompletionSuccess, index), nullptr);
                x.callback->decref();
              } else {
                log.error("TCP got RDMA send of %d bytes, but posted recv only has %d bytes\n", bytes, x.bytes);
                tcp->send(i, serializeToBuffer(messageCompletionError, index), nullptr);
              }
            } else {
              queuedRecvData.push_back({std::move(data), i, index});
            }
          });
        }
        break;
      }
      case messageRead: {
        uint64_t index;
        uint32_t rkey;
        void* address;
        size_t bytes;
        deserializeBuffer(view, index, rkey, address, bytes);
        if (auto mr = getMr(rkey, address, bytes)) {
          tcp->send(
              i, serializeToBuffer(messageReadSuccess, index, bytes), address, bytes, [mr = std::move(mr)](Error*) {});
        } else {
          tcp->send(i, serializeToBuffer(messageCompletionError, index), nullptr);
        }
        break;
      }
      case messageReadSuccess: {
        uint64_t index;
        size_t bytes;
        deserializeBuffer(view, index, bytes);
        std::unique_lock l(mutex);
        auto it = activeReads.find(index);
        if (it != activeReads.end()) {
          auto& xv = it->second;
          void* xaddress = xv.address;
          size_t xbytes = xv.bytes;
          RdmaCallback* xcallback = xv.callback;
          MrHandle mr(xv.mr.key);
          activeReads.erase(it);
          l.unlock();
          if (xbytes == bytes) {
            tcp->read(handle, xaddress, bytes, [this, callback = xcallback, mr = std::move(mr)] {
              std::unique_lock l(mutex);
              finished.push_back(callback);
              l.unlock();
              ++nFinished;
            });
            break;
          } else {
            log.error("TCP read size mismatch - got %d bytes, expected %d\n", bytes, xbytes);
          }
        } else {
          log.error("TCP read index %#x not found\n", index);
        }
        setError();
        break;
      }
      default:
        log.error("TCP got an unknown message type %d\n", type);
        setError();
      }
    } catch (const SerializationError& e) {
      fatal("tcp rdma serialization error: %s", e.what());
    }
  }

  void onError(size_t i, Error* e) {
    if (!cpuThreadApi.suppressErrors()) {
      log.error("Tcp connection error (remote rank %d): %s.\n", i, e->what());
      setError(i);
    }
  }

  MrHandle getMr(uint32_t key, void* address, size_t bytes) {
    CHECK(key != 0);
    std::lock_guard l(mrManager.mutex);
    auto& i = mrManager.mrs.at(key);
    if (!i.active) {
      log.error("TCP: Bad memory registration key %#x\n", key);
      setError();
      return false;
    }
    if (!i.cpu) {
      log.error("TCP: CUDA memory ops are not supported!\n");
      setError();
      return false;
    }
    if (bytes > 1ull << 40) {
      log.error("TCP: memory op too large (%d bytes)\n", bytes);
      setError();
      return false;
    }
    if ((uintptr_t)address < (uintptr_t)i.address || (uintptr_t)address + bytes > (uintptr_t)i.address + i.bytes ||
        (uintptr_t)address + bytes < (uintptr_t)address) {
      log.error(
          "TCP: memory op out of range. (requested [%#x, %#x], registration [%#x, %#x])\n", (uintptr_t)address, bytes,
          (uintptr_t)i.address, i.bytes);
      setError();
      return false;
    }
    return MrHandle(key);
  }

  void poll() {
    uint32_t errorRank = completionError.load(std::memory_order_relaxed);
    if (errorRank != (uint32_t)-1) {
      log.error("TCP: Work completion error from rank %d.\n", errorRank);
      completionError = (uint32_t)-1;
      cpuThreadApi.setError();
      errorState = true;
      return;
    }
    if (errorState) {
      size_t sourceRank = errorStateSourceRank.exchange((size_t)-1);
      if (sourceRank != (size_t)-1) {
        log.error("%s: Error communicating with %s.\n", cpuThreadApi.groupName(), cpuThreadApi.rankName(sourceRank));
      }
      cpuThreadApi.setError();
      return;
    }
    if ((ssize_t)nFinished <= 0) {
      return;
    }
    std::unique_lock l(mutex);
    std::swap(finished, tmpFinished);
    l.unlock();
    CHECK(!tmpFinished.empty());
    nFinished -= tmpFinished.size();
    CHECK(nActive >= tmpFinished.size());
    nActive -= tmpFinished.size();
    if (nActive == 0) {
      cpuThreadApi.setInactive();
    }
    for (auto& v : tmpFinished) {
      v->decref();
    }
    tmpFinished.clear();
  }

  auto writeCallback(MrHandle mr) {
    return [mr = std::move(mr)](Error* e) {
      if (e) {
        log.error("TCP write error: %s\n", e->what());
      }
    };
  }

  void postWrite(
      size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback, bool allowInline) {
    callback->i = i;
    ++callback->refcount;
    ++nActive;
    if (nActive == 1) {
      cpuThreadApi.setActive();
    }
    auto mr = getMr(lkey, localAddress, bytes);
    if (!mr) {
      return;
    }
    uint64_t index = activeIndex++;
    std::unique_lock l(mutex);
    activeCallbacks[index] = callback;
    l.unlock();
    tcp->send(
        i, serializeToBuffer(messageWrite, index, rkey, remoteAddress, bytes), localAddress, bytes,
        writeCallback(std::move(mr)));
  }
  void postRead(
      size_t i, void* localAddress, uint32_t lkey, void* remoteAddress, uint32_t rkey, size_t bytes,
      RdmaCallback* callback) {
    callback->i = i;
    ++callback->refcount;
    ++nActive;
    if (nActive == 1) {
      cpuThreadApi.setActive();
    }
    auto mr = getMr(lkey, localAddress, bytes);
    if (!mr) {
      return;
    }
    uint64_t index = activeIndex++;
    std::unique_lock l(mutex);
    activeReads[index] = {localAddress, bytes, callback};
    l.unlock();
    tcp->send(i, serializeToBuffer(messageRead, index, rkey, remoteAddress, bytes), writeCallback(std::move(mr)));
  }
  void postSend(size_t i, void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) {
    callback->i = i;
    ++callback->refcount;
    ++nActive;
    if (nActive == 1) {
      cpuThreadApi.setActive();
    }
    auto mr = getMr(lkey, localAddress, bytes);
    if (!mr) {
      return;
    }
    uint64_t index = activeIndex++;
    std::unique_lock l(mutex);
    activeCallbacks[index] = callback;
    l.unlock();
    tcp->send(i, serializeToBuffer(messageSend, index, bytes), localAddress, bytes, writeCallback(std::move(mr)));
  }
  void postRecv(void* localAddress, uint32_t lkey, size_t bytes, RdmaCallback* callback) {
    ++callback->refcount;
    auto mr = getMr(lkey, localAddress, bytes);
    if (!mr) {
      return;
    }
    std::unique_lock l(mutex);
    if (!queuedRecvData.empty()) {
      CHECK(activeRecvs.empty());
      auto d = queuedRecvData.pop_front_value();
      std::unique_lock cl(recvCallbackMutex);
      l.unlock();
      if (bytes >= d.data.size()) {
        std::memcpy(localAddress, d.data.data(), d.data.size());
        tcp->send(d.i, serializeToBuffer(messageCompletionSuccess, d.index), nullptr);
        callback->decref();
      } else {
        log.error("TCP got RDMA send of %d bytes, but posted recv only has %d bytes\n", d.data.size(), bytes);
        tcp->send(d.i, serializeToBuffer(messageCompletionError, d.index), nullptr);
        setError();
      }
      return;
    }
    activeRecvs.push_back({localAddress, bytes, callback, std::move(mr)});
    l.unlock();
  }
  void stopReceives() {
    std::unique_lock l(mutex);
    std::unique_lock cl(recvCallbackMutex);
    activeRecvs.clear();
  }
  bool mrCompatible(Rdma* other) {
    return other->type == type;
  }

  std::unique_ptr<RdmaMr> regMrCpu(void* address, size_t bytes) {
    auto r = std::make_unique<Mr>();
    r->rkey = r->lkey = mrManager.reg(true, address, bytes);
    return r;
  }
  std::unique_ptr<RdmaMr> regMrCuda(void* address, size_t bytes) {
    auto r = std::make_unique<Mr>();
    r->rkey = r->lkey = mrManager.reg(false, address, bytes);
    return r;
  }

  constexpr bool supportsCuda() const {
    return false;
  }

  void close() {
    tcp->close();
    cpuThreadApi = {};
  }
};

} // namespace

void tcpOnReadCallback(size_t i, BufferHandle buffer, void* destination, void* source) {
  ((RdmaTcp*)destination)->onRead(i, std::move(buffer), source);
}

void tcpOnErrorCallback(void* handle, size_t i, Error* error) {
  ((RdmaTcp*)handle)->onError(i, error);
}

std::unique_ptr<Rdma> makeRdmaTcp(Group* group) {
  return std::make_unique<RdmaTcp>(group);
}

} // namespace moodist
