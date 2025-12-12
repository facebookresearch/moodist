// Copyright (c) Meta Platforms, Inc. and affiliates.

// TcpStore implementation - wrapper around StoreImpl for PyTorch compatibility.
// Uses function pointers from coreApi to call into libmoodist.so.

#include "moodist_loader.h"
#include "store.h"

namespace moodist {

TcpStore::TcpStore(StoreHandle* handle) : handle(handle) {
  // Note: caller is responsible for ensuring handle has correct refcount
}

TcpStore::TcpStore(std::string hostname, int port, std::string key, int worldSize, int rank,
    std::chrono::steady_clock::duration timeout) {
  handle = coreApi.createStoreImpl(hostname, port, key, worldSize, rank);
  timeout_ = std::chrono::ceil<std::chrono::milliseconds>(timeout);
}

TcpStore::~TcpStore() {
  coreApi.storeImplDecRef(handle);
}

c10::intrusive_ptr<c10d::Store> TcpStore::clone() {
  coreApi.storeImplAddRef(handle);
  auto cloned = c10::make_intrusive<TcpStore>(handle);
  cloned->timeout_ = timeout_;
  return cloned;
}

void TcpStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  coreApi.storeImplSet(handle, timeout_, key, value);
}

std::vector<uint8_t> TcpStore::get(const std::string& key) {
  return coreApi.storeImplGet(handle, timeout_, key);
}

int64_t TcpStore::add(const std::string& key, int64_t value) {
  throw std::runtime_error("Moodist Store add method is not implemented");
  return 0;
}

bool TcpStore::deleteKey(const std::string& key) {
  throw std::runtime_error("Moodist Store deleteKey method is not implemented");
}

bool TcpStore::check(const std::vector<std::string>& keys) {
  std::vector<std::string_view> keyViews;
  keyViews.reserve(keys.size());
  for (const auto& k : keys) {
    keyViews.push_back(k);
  }
  return coreApi.storeImplCheck(handle, timeout_, keyViews);
}

int64_t TcpStore::getNumKeys() {
  throw std::runtime_error("Moodist Store getNumKeys method is not implemented");
}

void TcpStore::wait(const std::vector<std::string>& keys) {
  std::vector<std::string_view> keyViews;
  keyViews.reserve(keys.size());
  for (const auto& k : keys) {
    keyViews.push_back(k);
  }
  coreApi.storeImplWait(handle, timeout_, keyViews);
}

void TcpStore::wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout) {
  std::vector<std::string_view> keyViews;
  keyViews.reserve(keys.size());
  for (const auto& k : keys) {
    keyViews.push_back(k);
  }
  coreApi.storeImplWait(handle, timeout, keyViews);
}

} // namespace moodist
