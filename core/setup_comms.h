// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"
#include "group.h"
#include "serialization.h"

namespace moodist {

// The error of a wait of the setup's ring (see SetupComms): the setup has failed, or the wait timed out
struct SetupCommsError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The ranks' TCP ring for the exchanges of the process group's setup (and of later setup steps, e.g. NVLS). Its waits
// fail instead of waiting forever: once the ring's connection to a neighbor closes (that rank failed or exited), once
// another rank reports a failure (see fail), or after the timeout.
struct SetupComms {
  const size_t rank;
  const size_t size;

  std::string key;

  // How long a wait for other ranks may take (set from the process group's timeout, see Group::setTimeout)
  std::chrono::steady_clock::duration timeout = std::chrono::minutes(30);

  SetupComms(size_t rank, size_t size) : rank(rank), size(size) {}
  virtual ~SetupComms() {}

  void connect(std::string address);
  std::vector<std::string> listenerAddresses();
  void waitForConnections();

  // Makes this rank's waits fail from now on, and tells the other ranks (around the ring, best effort), whose waits
  // then fail too: for errors in the setup that only this rank has found
  void fail(std::string reason);
  // Throws if the waits have failed (see fail): for other waits of the setup
  void checkFailed();
  // Why the waits fail (see fail), or empty if they do not
  std::string failureReason();
  // Throws (after fail, which tells the other ranks) if a wait that began at `start` has taken longer than the
  // timeout; what() describes what it waits for
  template<typename F>
  void checkTimeout(std::chrono::steady_clock::time_point start, F&& what) {
    if (std::chrono::steady_clock::now() - start >= timeout) {
      timedOut(start, what());
    }
  }
  [[noreturn]] void timedOut(std::chrono::steady_clock::time_point start, std::string what);

  std::vector<std::string_view>& allgather(BufferHandle);

  template<typename T>
  std::vector<T> allgather(const T& v) {
    auto buffer = serializeToBuffer((uint64_t)0, (uint64_t)0, (uint32_t)0, (uint32_t)0, (uint32_t)0, v);
    auto& outputList = allgather(std::move(buffer));
    try {
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
    auto buffer = serializeToBuffer(
        (uint64_t)0, (uint64_t)0, (uint32_t)0, (uint32_t)0, (uint32_t)0, std::string_view(typeid(T).name()), v);
    sendBufferTo(rank, std::move(buffer));
  }
  template<typename T>
  T recvFrom(size_t rank) {
    auto view = recvBufferFrom(rank);
    try {
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
