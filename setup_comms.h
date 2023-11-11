#pragma once

#include "common.h"
#include "group.h"
#include "serialization.h"

namespace moodist {

struct SetupComms {
  const size_t rank;
  const size_t size;

  std::string key;

  SetupComms(size_t rank, size_t size) : rank(rank), size(size) {}
  virtual ~SetupComms() {}

  void connect(std::string address);
  std::vector<std::string> listenerAddresses();
  void waitForConnections();

  std::vector<std::string_view>& allgather(BufferHandle);

  template<typename T>
  std::vector<T> allgather(const T& v) {
    try {
      auto buffer = serializeToBuffer((uint64_t)0, (uint64_t)0, (uint32_t)0, (uint32_t)0, (uint32_t)0, v);
      auto& outputList = allgather(std::move(buffer));
      std::vector<T> r;
      r.resize(size);
      for (size_t i = 0; i != size; ++i) {
        deserializeBuffer(outputList.at(i), r[i]);
      }
      return r;
    } catch (const std::exception& e) {
      throw std::runtime_error(fmt::sprintf("exception in allgather <%s>: %s", typeid(T).name(), e.what()));
    }
  }

  void sendBufferTo(size_t rank, BufferHandle);
  std::string_view recvBufferFrom(size_t rank);

  template<typename T>
  void sendTo(size_t rank, const T& v) {
    auto buffer =
        serializeToBuffer((uint64_t)0, (uint64_t)0, (uint32_t)0, (uint32_t)0, (uint32_t)0, std::string_view(typeid(T).name()), v);
    sendBufferTo(rank, std::move(buffer));
  }
  template<typename T>
  T recvFrom(size_t rank) {
    try {
      auto view = recvBufferFrom(rank);
      std::string_view tname;
      view = deserializeBufferPart(view, tname);
      if (tname != typeid(T).name()) {
        throw std::runtime_error(
            fmt::sprintf("recvFrom got unexpected type '%s'. Expected '%s'", tname, typeid(T).name()));
      }
      T r;
      deserializeBuffer(view, r);
      return r;
    } catch (const std::exception& e) {
      throw std::runtime_error(fmt::sprintf("recvFrom<%s>(%d): %s", typeid(T).name(), rank, e.what()));
    }
  }
};

std::unique_ptr<SetupComms> createSetupComms(size_t rank, size_t size);

} // namespace moodist
