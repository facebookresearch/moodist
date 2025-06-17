// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "connection.h"
#include "serialization.h"

#include <cstdio>
#include <limits.h>
#include <sys/uio.h>
#include <unistd.h>

namespace moodist {

void Listener::accept(Function<void(Error*, std::shared_ptr<Connection>)> callback) {
  socket.accept([this, callback = std::move(callback)](Error* error, Socket socket) {
    if (error) {
      callback(error, nullptr);
    } else {
      auto connection = std::make_shared<Connection>(std::move(socket));
      callback(nullptr, std::move(connection));
    }
  });
}

std::vector<std::string> Listener::localAddresses() const {
  return socket.localAddresses();
}

const uint32_t sigSocketData = 0x40ea69f4;
const uint32_t sigSocketDataFd = 0xf555d4e4;

SharedBufferHandle toShared(SharedBufferHandle h) {
  return h;
}
SharedBufferHandle toShared(BufferHandle h) {
  return SharedBufferHandle(h.release());
}

template<typename Buffer>
void Connection::write(Buffer buffer, Function<void(Error*)> callback) {
  uint32_t size = buffer->size();
  if ((size_t)size != buffer->size()) {
    throw Error("write: buffer is too large (size does not fit in 32 bits)");
  }

  auto buffer0 = serializeToBuffer(sigSocketData, size);
  std::array<iovec, 2> iovec;
  iovec[0].iov_base = buffer0->data();
  iovec[0].iov_len = buffer0->size();
  iovec[1].iov_base = buffer->data();
  iovec[1].iov_len = buffer->size();
  socket.writev(
      iovec.data(), iovec.size(),
      [buffer0 = std::move(buffer0), buffer = std::move(buffer), callback = std::move(callback)](Error* error) {
        if (callback) {
          callback(error);
        }
      });
}

template void Connection::write(BufferHandle, Function<void(Error*)>);
template void Connection::write(SharedBufferHandle, Function<void(Error*)>);

template<typename Buffer>
void Connection::writefd(Buffer buffer, int fd) {
  uint32_t size = buffer->size();
  if ((size_t)size != buffer->size()) {
    throw Error("write: buffer is too large (size does not fit in 32 bits)");
  }

  int dupfd = ::dup(fd);

  log.info("writefd dupfd is %d\n", dupfd);

  auto buffer0 = serializeToBuffer(sigSocketDataFd, size);
  std::array<iovec, 2> iovec;
  iovec[0].iov_base = buffer0->data();
  iovec[0].iov_len = buffer0->size();
  iovec[1].iov_base = buffer->data();
  iovec[1].iov_len = buffer->size();
  socket.writev(
      iovec.data(), iovec.size(), [buffer0 = std::move(buffer0), buffer = std::move(buffer)](Error* error) {});

  socket.sendFd(dupfd, [dupfd](Error* error) { ::close(dupfd); });
}

template void Connection::writefd(BufferHandle, int);
template void Connection::writefd(SharedBufferHandle, int);

void Connection::close() {
  socket.close();
}
bool Connection::closed() const {
  return socket.closed();
}

struct ReadState {
  Connection* connection;
  int state = 0;
  bool recvFd = false;
  int fd = -1;
  Function<void(Error*, BufferHandle)> callback;
  Function<void(Error*, BufferHandle, int)> fdcallback;
  BufferHandle buffer;
  CachedReader reader;
  ReadState(Connection* connection, Function<void(Error*, BufferHandle)> callback)
      : connection(connection), reader(&connection->socket), callback(std::move(callback)) {}
  ReadState(Connection* connection, Function<void(Error*, BufferHandle, int)> fdcallback)
      : connection(connection), reader(&connection->socket), fdcallback(std::move(fdcallback)) {}
  void callError(Error* error) {
    if (fdcallback) {
      fdcallback(error, nullptr, -1);
    } else {
      callback(error, nullptr);
    }
  }
  void operator()(Error* error) {
    if (error) {
      callError(error);
      return;
    }

    static constexpr int stateZero = 0;
    static constexpr int stateSocketReadIovecs = 1;
    static constexpr int stateRecvFd = 2;
    static constexpr int stateAllDone = 3;

    while (true) {
      switch (state) {
      default:
        connection->close();
        return;
      case stateZero: {
        void* ptr = reader.readBufferPointer(8);
        if (!ptr) {
          return;
        }
        uint32_t bufferSize;
        uint32_t recvSignature;
        deserializeBuffer(ptr, 8, recvSignature, bufferSize);
        switch (recvSignature) {
        case sigSocketData:
          break;
        case sigSocketDataFd:
          recvFd = true;
          break;
        default:
          state = -1;
          Error e("bad signature");
          callError(&e);
          return;
        }
        buffer = makeBuffer(bufferSize);
        state = stateSocketReadIovecs;
        reader.newRead();
        reader.addIovec(buffer->data(), buffer->size());
        reader.startRead();
        [[fallthrough]];
      }
      case stateSocketReadIovecs:
        if (!reader.done()) {
          return;
        } else {
          state = recvFd ? stateRecvFd : stateAllDone;
          recvFd = false;
          break;
        }
      case stateRecvFd: {
        fd = connection->socket.recvFd(reader);
        if (fd == -1) {
          return;
        }
        state = stateAllDone;
        [[fallthrough]];
      }
      case stateAllDone: {
        state = stateZero;
        if (fdcallback) {
          fdcallback(nullptr, std::move(buffer), fd);
          fd = -1;
        } else {
          callback(nullptr, std::move(buffer));
        }
        break;
      }
      }
    }
  }
};

void Connection::read(Function<void(Error*, BufferHandle)> callback) {
  socket.setOnRead(ReadState(this, std::move(callback)));
}

void Connection::readfd(Function<void(Error*, BufferHandle, int)> callback) {
  socket.setOnRead(ReadState(this, std::move(callback)));
}

std::string Connection::localAddress() const {
  return socket.localAddress();
}

std::string Connection::remoteAddress() const {
  return socket.remoteAddress();
}

Connection::~Connection() {}

std::shared_ptr<Listener> UnixContext::listen(std::string_view addr) {
  auto listener = std::make_shared<Listener>(Socket::Unix());
  listener->socket.listen(addr);
  return listener;
}

std::shared_ptr<Connection> UnixContext::connect(std::string_view addr) {
  auto connection = std::make_shared<Connection>(Socket::Unix());
  connection->socket.connect(addr, [](Error* e) {});
  return connection;
}

std::shared_ptr<Listener> TcpContext::listen(std::string_view addr) {
  auto listener = std::make_shared<Listener>(Socket::Tcp());
  listener->socket.listen(addr);
  return listener;
}

std::shared_ptr<Connection> TcpContext::connect(std::string_view addr) {
  auto connection = std::make_shared<Connection>(Socket::Tcp());
  connection->socket.connect(addr, [connection](Error* e) {});
  return connection;
}

std::string readBootId() {
  char buf[64];
  std::memset(buf, 0, sizeof(buf));
  FILE* f = ::fopen("/proc/sys/kernel/random/boot_id", "rb");
  if (f) {
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    while (n && buf[n - 1] == '\n') {
      --n;
    }
    buf[n] = 0;
    fclose(f);
  }
  return buf;
}

std::string bootId = readBootId();

bool UnixContext::isReachable(std::string_view networkKey, std::string_view address) {
  return networkKey == bootId;
}
std::string UnixContext::getNetworkKey() {
  return bootId;
}

bool TcpContext::isReachable(std::string_view networkKey, std::string_view address) {
  if (isAnyAddress(address)) {
    return false;
  }
  if (isLoopbackAddress(address)) {
    return networkKey == bootId;
  }
  return true;
}
std::string TcpContext::getNetworkKey() {
  return bootId;
}

} // namespace moodist
