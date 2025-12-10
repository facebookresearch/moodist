// Copyright (c) Meta Platforms, Inc. and affiliates.

// TcpStore implementation - wrapper around StoreImpl for PyTorch compatibility.

#include "store.h"

namespace moodist {

TcpStore::TcpStore(StoreImpl* impl) : impl(impl) {
  storeImplAddRef(impl);
}

TcpStore::TcpStore(std::string hostname, int port, std::string key, int worldSize, int rank,
    std::chrono::steady_clock::duration timeout) {
  impl = createStoreImpl(std::move(hostname), port, std::move(key), worldSize, rank);
  storeImplAddRef(impl);
  timeout_ = std::chrono::ceil<std::chrono::milliseconds>(timeout);
}

TcpStore::~TcpStore() {
  storeImplDecRef(impl);
}

void TcpStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  storeImplSet(impl, timeout_, key, value);
}

std::vector<uint8_t> TcpStore::get(const std::string& key) {
  return storeImplGet(impl, timeout_, key);
}

int64_t TcpStore::add(const std::string& key, int64_t value) {
  throw std::runtime_error("Moodist Store add method is not implemented");
  return 0;
}

bool TcpStore::deleteKey(const std::string& key) {
  throw std::runtime_error("Moodist Store deleteKey method is not implemented");
}

bool TcpStore::check(const std::vector<std::string>& keys) {
  return storeImplCheck(impl, timeout_, keys);
}

int64_t TcpStore::getNumKeys() {
  throw std::runtime_error("Moodist Store getNumKeys method is not implemented");
}

void TcpStore::wait(const std::vector<std::string>& keys) {
  storeImplWait(impl, timeout_, keys);
}

void TcpStore::wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout) {
  storeImplWait(impl, timeout, keys);
}

} // namespace moodist
