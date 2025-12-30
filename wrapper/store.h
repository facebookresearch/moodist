// Copyright (c) Meta Platforms, Inc. and affiliates.

// TcpStore - PyTorch-compatible wrapper around StoreImpl.
// This class inherits from c10d::Store and lives in the _C wrapper library.

#pragma once

#include "api/moodist_api.h"
#include "torch_includes.h"

namespace moodist {

// Global CoreApi object - defined in moodist_loader.cc
extern CoreApi coreApi;

class TORCH_API TcpStore final : public c10d::Store {
public:
  TcpStore(StoreHandle* handle);
  TcpStore(std::string hostname, int port, std::string key, int worldSize, int rank,
      std::chrono::steady_clock::duration timeout);
  ~TcpStore();
  TcpStore(const TcpStore&) = delete;
  TcpStore& operator=(const TcpStore&) = delete;

  StoreHandle* handle = nullptr;

  virtual c10::intrusive_ptr<Store> clone() override;

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
