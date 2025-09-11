// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "buffer.h"
#include "socket.h"
#include "synchronization.h"

#include "fmt/printf.h"

#include <memory>
#include <string_view>

namespace moodist {

struct Connection;
struct Listener;

struct UnixContext {
  bool valid() {
    return true;
  }
  std::shared_ptr<Listener> listen(std::string_view addr);
  std::shared_ptr<Connection> connect(std::string_view addr);
  static bool isReachable(std::string_view networkKey, std::string_view address);
  static std::string getNetworkKey();
};

struct TcpContext {
  bool valid() {
    return true;
  }
  std::shared_ptr<Listener> listen(std::string_view addr);
  std::shared_ptr<Connection> connect(std::string_view addr);
  static bool isReachable(std::string_view networkKey, std::string_view address);
  static std::string getNetworkKey();
};

struct Listener {
  Socket socket;
  Listener(Socket socket) : socket(std::move(socket)) {}

  void close() {
    socket.close();
  }

  void accept(Function<void(Error*, std::shared_ptr<Connection>)> callback);

  std::vector<std::string> localAddresses() const;
};

struct Connection : std::enable_shared_from_this<Connection> {
  Socket socket;

  Connection(Socket socket) : socket(std::move(socket)) {}
  ~Connection();

  void close();
  bool closed() const;
  void read(Function<void(Error*, BufferHandle)>);
  void readfd(Function<void(Error*, BufferHandle, int)>);
  template<typename Buffer>
  void write(Buffer buffer, Function<void(Error*)>);
  template<typename Buffer>
  void write(Buffer buffer, std::span<iovec> extra, Function<void(Error*)>);
  template<typename Buffer>
  void writefd(Buffer buffer, int fd);

  void inread_iovec(void* ptr, size_t bytes, Function<void()> callback);

  std::string localAddress() const;
  std::string remoteAddress() const;
};

} // namespace moodist
