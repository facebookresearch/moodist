// Copyright (c) Meta Platforms, Inc. and affiliates.

// TcpStore implementation - wrapper around StoreImpl for PyTorch compatibility.
// Uses ApiHandle for refcounted ownership and ApiProxy for method access.

#include "moodist_loader.h"
#include "store.h"

namespace moodist {

// ApiProxy<Store> method implementations - delegate to coreApi
namespace api {

void ApiProxy<Store>::close() {
  coreApi.storeClose(ptr);
}

void ApiProxy<Store>::set(
    std::chrono::steady_clock::duration timeout, std::string_view key, const std::vector<uint8_t>& value) {
  coreApi.storeSet(ptr, timeout, key, value);
}

std::vector<uint8_t> ApiProxy<Store>::get(std::chrono::steady_clock::duration timeout, std::string_view key) {
  return coreApi.storeGet(ptr, timeout, key);
}

bool ApiProxy<Store>::check(std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys) {
  return coreApi.storeCheck(ptr, timeout, keys);
}

void ApiProxy<Store>::wait(std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys) {
  coreApi.storeWait(ptr, timeout, keys);
}

void destroy(Store* store) {
  coreApi.storeDestroy(store);
}

} // namespace api

namespace wrapper {

TcpStore::TcpStore(api::StoreHandle h) : handle(std::move(h)) {}

TcpStore::TcpStore(std::string hostname, int port, std::string key, int worldSize, int rank,
    std::chrono::steady_clock::duration timeout)
    : handle(coreApi.createStore(hostname, port, key, worldSize, rank)) {
  timeout_ = std::chrono::ceil<std::chrono::milliseconds>(timeout);
}

c10::intrusive_ptr<c10d::Store> TcpStore::clone() {
  // Copy the handle (increments refcount via ApiHandle copy constructor)
  auto cloned = c10::make_intrusive<TcpStore>(api::StoreHandle(handle));
  cloned->timeout_ = timeout_;
  return cloned;
}

void TcpStore::set(const std::string& key, const std::vector<uint8_t>& value) {
  handle->set(timeout_, key, value);
}

std::vector<uint8_t> TcpStore::get(const std::string& key) {
  return handle->get(timeout_, key);
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
  return handle->check(timeout_, keyViews);
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
  handle->wait(timeout_, keyViews);
}

void TcpStore::wait(const std::vector<std::string>& keys, const std::chrono::milliseconds& timeout) {
  std::vector<std::string_view> keyViews;
  keyViews.reserve(keys.size());
  for (const auto& k : keys) {
    keyViews.push_back(k);
  }
  handle->wait(timeout, keyViews);
}

} // namespace wrapper
} // namespace moodist
