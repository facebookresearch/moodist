#pragma once

#include "buffer.h"
#include "common.h"

namespace moodist {

struct Group;

struct TcpDev {
  void* impl = nullptr;
  void destroy();

  void setOnRead(void*);

  void send(size_t i, BufferHandle&& buffer, void* payload, size_t payloadBytes, Function<void(Error*)> callback = nullptr);
  void send(size_t i, BufferHandle buffer, Function<void(Error*)> callback = nullptr);
  void read(void* handle, void* address, size_t bytes, Function<void()> callback);

  void close();
};

void tcpOnReadCallback(size_t i, BufferHandle buffer, void*, void*);
void tcpOnErrorCallback(void*, size_t i, Error*);

UniqueImpl<TcpDev> makeTcpDev(Group* group, void* onReadHandle);

} // namespace moodist
