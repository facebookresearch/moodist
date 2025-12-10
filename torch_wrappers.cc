// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "torch_wrappers.h"
#include "tensor_factory.h"
#include "tensor_wrapper.h"
#include "torch_includes.h"

#include <stdexcept>

namespace moodist {

namespace {

[[noreturn]] void notImplemented(const char* func) {
  throw std::runtime_error(std::string("Not implemented: ") + func);
}

// Helper to access the torch::Tensor inside TensorWrapper
torch::Tensor& getTensor(TensorWrapper& w) {
  return *reinterpret_cast<torch::Tensor*>(w.storage());
}
const torch::Tensor& getTensor(const TensorWrapper& w) {
  return *reinterpret_cast<const torch::Tensor*>(w.storage());
}

// Helper to access torch::DataPtr inside DataPtrWrapper
torch::DataPtr& getDataPtr(DataPtrWrapper& w) {
  return *reinterpret_cast<torch::DataPtr*>(w.storage());
}
const torch::DataPtr& getDataPtr(const DataPtrWrapper& w) {
  return *reinterpret_cast<const torch::DataPtr*>(w.storage());
}

// Helper to access torch::Storage inside StorageWrapper
torch::Storage& getStorage(StorageWrapper& w) {
  return *reinterpret_cast<torch::Storage*>(w.storage());
}
const torch::Storage& getStorage(const StorageWrapper& w) {
  return *reinterpret_cast<const torch::Storage*>(w.storage());
}

} // namespace

// ============================================================================
// Device/Stream functions
// ============================================================================

int cudaCurrentDevice() {
  return c10::cuda::current_device();
}

CUstream cudaGetCurrentStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

// ============================================================================
// StreamGuard
// ============================================================================

StreamGuard::StreamGuard(CUstream stream, int device_index) {
  notImplemented("StreamGuard::StreamGuard");
}

StreamGuard::~StreamGuard() {
  // If not implemented, destructor is no-op
}

// ============================================================================
// DataPtrWrapper
// ============================================================================

DataPtrWrapper::DataPtrWrapper() : valid_(false) {}

DataPtrWrapper::~DataPtrWrapper() {
  if (valid_) {
    getDataPtr(*this).~DataPtr();
  }
}

DataPtrWrapper::DataPtrWrapper(DataPtrWrapper&& other) noexcept : valid_(other.valid_) {
  if (valid_) {
    new (storage_) torch::DataPtr(std::move(getDataPtr(other)));
    other.valid_ = false;
  }
}

DataPtrWrapper& DataPtrWrapper::operator=(DataPtrWrapper&& other) noexcept {
  if (this != &other) {
    if (valid_) {
      getDataPtr(*this).~DataPtr();
    }
    valid_ = other.valid_;
    if (valid_) {
      new (storage_) torch::DataPtr(std::move(getDataPtr(other)));
      other.valid_ = false;
    }
  }
  return *this;
}

void* DataPtrWrapper::get() const {
  if (!valid_) {
    return nullptr;
  }
  return getDataPtr(*this).get();
}

DataPtrWrapper::operator bool() const {
  return valid_ && getDataPtr(*this).get() != nullptr;
}

DataPtrWrapper cudaAllocate(size_t bytes) {
  notImplemented("cudaAllocate");
}

// ============================================================================
// StorageWrapper
// ============================================================================

StorageWrapper::StorageWrapper() : valid_(false) {}

StorageWrapper::~StorageWrapper() {
  if (valid_) {
    getStorage(*this).~Storage();
  }
}

StorageWrapper::StorageWrapper(StorageWrapper&& other) noexcept : valid_(other.valid_) {
  if (valid_) {
    new (storage_) torch::Storage(std::move(getStorage(other)));
    other.valid_ = false;
  }
}

StorageWrapper& StorageWrapper::operator=(StorageWrapper&& other) noexcept {
  if (this != &other) {
    if (valid_) {
      getStorage(*this).~Storage();
    }
    valid_ = other.valid_;
    if (valid_) {
      new (storage_) torch::Storage(std::move(getStorage(other)));
      other.valid_ = false;
    }
  }
  return *this;
}

StorageWrapper::StorageWrapper(const StorageWrapper& other) : valid_(other.valid_) {
  if (valid_) {
    new (storage_) torch::Storage(getStorage(other));
  }
}

StorageWrapper& StorageWrapper::operator=(const StorageWrapper& other) {
  if (this != &other) {
    if (valid_) {
      getStorage(*this).~Storage();
    }
    valid_ = other.valid_;
    if (valid_) {
      new (storage_) torch::Storage(getStorage(other));
    }
  }
  return *this;
}

StorageWrapper::operator bool() const {
  return valid_;
}

void StorageWrapper::reset() {
  if (valid_) {
    getStorage(*this).~Storage();
    valid_ = false;
  }
}

StorageWrapper storageFromTensor(const TensorWrapper& t) {
  if (!t.defined()) {
    return StorageWrapper();
  }
  StorageWrapper w;
  new (w.storage()) torch::Storage(getTensor(t).storage());
  w.setValid(true);
  return w;
}

// ============================================================================
// TensorWrapper
// ============================================================================

TensorWrapper::TensorWrapper() : valid_(false) {}

TensorWrapper::~TensorWrapper() {
  if (valid_) {
    getTensor(*this).~Tensor();
  }
}

TensorWrapper::TensorWrapper(TensorWrapper&& other) noexcept : valid_(other.valid_) {
  if (valid_) {
    new (storage_) torch::Tensor(std::move(getTensor(other)));
    other.valid_ = false;
  }
}

TensorWrapper& TensorWrapper::operator=(TensorWrapper&& other) noexcept {
  if (this != &other) {
    if (valid_) {
      getTensor(*this).~Tensor();
    }
    valid_ = other.valid_;
    if (valid_) {
      new (storage_) torch::Tensor(std::move(getTensor(other)));
      other.valid_ = false;
    }
  }
  return *this;
}

TensorWrapper::TensorWrapper(const TensorWrapper& other) : valid_(other.valid_) {
  if (valid_) {
    new (storage_) torch::Tensor(getTensor(other));
  }
}

TensorWrapper& TensorWrapper::operator=(const TensorWrapper& other) {
  if (this != &other) {
    if (valid_) {
      getTensor(*this).~Tensor();
    }
    valid_ = other.valid_;
    if (valid_) {
      new (storage_) torch::Tensor(getTensor(other));
    }
  }
  return *this;
}

TensorWrapper::operator bool() const {
  return valid_ && getTensor(*this).defined();
}

bool TensorWrapper::defined() const {
  return valid_ && getTensor(*this).defined();
}

void* TensorWrapper::data_ptr() const {
  if (!valid_) {
    return nullptr;
  }
  return getTensor(*this).data_ptr();
}

void* TensorWrapper::mutable_data_ptr() {
  if (!valid_) {
    return nullptr;
  }
  return getTensor(*this).mutable_data_ptr();
}

const void* TensorWrapper::const_data_ptr() const {
  if (!valid_) {
    return nullptr;
  }
  return getTensor(*this).const_data_ptr();
}

size_t TensorWrapper::numel() const {
  if (!valid_) {
    return 0;
  }
  return getTensor(*this).numel();
}

size_t TensorWrapper::itemsize() const {
  if (!valid_) {
    return 0;
  }
  return getTensor(*this).itemsize();
}

DeviceInfo TensorWrapper::device() const {
  if (!valid_) {
    return DeviceInfo{};
  }
  auto d = getTensor(*this).device();
  return DeviceInfo{d.index(), d.is_cuda()};
}

int TensorWrapper::device_index() const {
  if (!valid_) {
    return -1;
  }
  return getTensor(*this).device().index();
}

DType TensorWrapper::dtype() const {
  if (!valid_) {
    return DType::Float;
  }
  return static_cast<DType>(static_cast<int>(getTensor(*this).scalar_type()));
}

bool TensorWrapper::is_contiguous() const {
  if (!valid_) {
    return false;
  }
  return getTensor(*this).is_contiguous();
}

bool TensorWrapper::is_cuda() const {
  if (!valid_) {
    return false;
  }
  return getTensor(*this).is_cuda();
}

bool TensorWrapper::is_cpu() const {
  if (!valid_) {
    return false;
  }
  return getTensor(*this).is_cpu();
}

const void* TensorWrapper::storage_data() const {
  if (!valid_) {
    return nullptr;
  }
  return getTensor(*this).storage().data();
}

int64_t TensorWrapper::ndim() const {
  if (!valid_) {
    return 0;
  }
  return getTensor(*this).ndimension();
}

int64_t TensorWrapper::size(int dim) const {
  if (!valid_) {
    return 0;
  }
  return getTensor(*this).size(dim);
}

int64_t TensorWrapper::stride(int dim) const {
  if (!valid_) {
    return 0;
  }
  return getTensor(*this).stride(dim);
}

std::vector<int64_t> TensorWrapper::sizes() const {
  if (!valid_) {
    return {};
  }
  auto s = getTensor(*this).sizes();
  return std::vector<int64_t>(s.begin(), s.end());
}

std::vector<int64_t> TensorWrapper::strides() const {
  if (!valid_) {
    return {};
  }
  auto s = getTensor(*this).strides();
  return std::vector<int64_t>(s.begin(), s.end());
}

// Factory functions
TensorWrapper tensorFromBlob(void* data, const std::vector<int64_t>& sizes, DType dtype, int device_index) {
  notImplemented("tensorFromBlob");
}

TensorWrapper tensorEmpty(const std::vector<int64_t>& sizes, DType dtype, int device_index) {
  notImplemented("tensorEmpty");
}

// Tensor operations
void tensorSumOut(TensorWrapper& dst, const TensorWrapper& src, int dim) {
  notImplemented("tensorSumOut");
}

void tensorAmaxOut(TensorWrapper& dst, const TensorWrapper& src, int dim) {
  notImplemented("tensorAmaxOut");
}

void tensorAminOut(TensorWrapper& dst, const TensorWrapper& src, int dim) {
  notImplemented("tensorAminOut");
}

void tensorAdd_(TensorWrapper& dst, const TensorWrapper& src) {
  notImplemented("tensorAdd_");
}

void tensorCopy_(TensorWrapper& dst, const TensorWrapper& src) {
  notImplemented("tensorCopy_");
}

void tensorMulScalar_(TensorWrapper& t, float scalar) {
  notImplemented("tensorMulScalar_");
}

TensorWrapper tensorView(TensorWrapper& t, const std::vector<int64_t>& sizes) {
  notImplemented("tensorView");
}

// Dtype utilities
size_t dtypeItemsize(DType dtype) {
  return torch::elementSize(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
}

const char* dtypeName(DType dtype) {
  return toString(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
}

// TensorDataPtr conversion
TensorWrapper tensorFromData(TensorDataPtr ptr) {
  torch::Tensor t = makeTensor(std::move(ptr));
  TensorWrapper w;
  new (w.storage()) torch::Tensor(std::move(t));
  w.setValid(true);
  return w;
}

// torch::Tensor conversion (wrapper-only, not declared in header)
TensorWrapper wrapTensor(torch::Tensor t) {
  TensorWrapper w;
  new (w.storage()) torch::Tensor(std::move(t));
  w.setValid(true);
  return w;
}

// Extract torch::Tensor from TensorWrapper (wrapper-only)
torch::Tensor unwrapTensor(TensorWrapper& w) {
  if (!w.defined()) {
    return torch::Tensor();
  }
  return getTensor(w);
}

} // namespace moodist
