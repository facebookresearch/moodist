// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "torch_includes.h"

namespace moodist {

struct StoreImpl;

class TORCH_API TcpStore final : public c10d::Store {
public:
  TcpStore(StoreImpl* impl);
  TcpStore(
      std::string hostname, int port, std::string key, int worldSize, int rank,
      std::chrono::steady_clock::duration timeout);
  ~TcpStore();
  TcpStore(const TcpStore&) = delete;
  TcpStore& operator=(const TcpStore&) = delete;

  StoreImpl* impl = nullptr;

  virtual c10::intrusive_ptr<Store> clone() {
    return c10::make_intrusive<TcpStore>(impl);
  }

  virtual void set(const std::string& key, const std::vector<uint8_t>& value) override;
  virtual std::vector<uint8_t> get(const std::string& key) override;
  virtual int64_t add(const std::string& key, int64_t value) override;
  virtual bool deleteKey(const std::string& key) override;
  virtual bool check(const std::vector<std::string>& keys) override;
  virtual int64_t getNumKeys() override;
  virtual void wait(const std::vector<std::string>& keys) override;
  virtual void wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout) override;
};

} // namespace moodist