// Copyright (c) Meta Platforms, Inc. and affiliates.

// Loads libmoodist.so via dlopen and provides wrapper functions for PyTorch APIs

#include "moodist_loader.h"
#include "moodist_api.h"
#include "processgroup_wrapper.h"
#include "torch_includes.h"

#include <atomic>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace moodist {

// ============================================================================
// Tensor - opaque type that holds a torch::Tensor
// ============================================================================

struct Tensor {
  torch::Tensor tensor;
  std::atomic<int> refcount{1};
};

// ============================================================================
// Wrapper functions - these wrap PyTorch APIs and are called by libmoodist.so
// In _C.so, TensorPtr calls these directly via wrappers::
// ============================================================================

namespace wrappers {

// CUDA device/stream functions
int cudaCurrentDevice() {
  return c10::cuda::current_device();
}

CUstream cudaGetCurrentStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

void* cudaStreamGuardEnter(CUstream stream, int device) {
  auto* guard = new c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, device));
  return guard;
}

void cudaStreamGuardExit(void* guard) {
  delete static_cast<c10::cuda::CUDAStreamGuard*>(guard);
}

void* cudaCachingAllocatorAlloc(size_t bytes, void** cleanupCtx) {
  auto dataPtr = c10::cuda::CUDACachingAllocator::get()->allocate(bytes);
  void* ptr = dataPtr.get();
  // Store the DataPtr on heap so it lives until free is called
  *cleanupCtx = new c10::DataPtr(std::move(dataPtr));
  return ptr;
}

void cudaCachingAllocatorFree(void* cleanupCtx) {
  delete static_cast<c10::DataPtr*>(cleanupCtx);
}

// Tensor creation
Tensor* tensorFromBlob(void* data, const int64_t* sizes, int ndim, DType dtype, int device) {
  auto options = at::TensorOptions().dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
  if (device < 0) {
    options = options.device(torch::kCPU);
  } else {
    options = options.device(torch::kCUDA, device);
  }
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  t->tensor = torch::from_blob(data, sizeVec, options);
  return t;
}

Tensor* tensorEmpty(const int64_t* sizes, int ndim, DType dtype, int device) {
  auto options = at::TensorOptions().dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
  if (device < 0) {
    options = options.device(torch::kCPU);
  } else {
    options = options.device(torch::kCUDA, device);
  }
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  t->tensor = torch::empty(sizeVec, options);
  return t;
}

Tensor* tensorFromBlobWithDeleter(
    void* data, const int64_t* sizes, int ndim, DType dtype, int device, void (*deleter)(void* ctx), void* deleterCtx) {
  auto options = at::TensorOptions().dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
  if (device < 0) {
    options = options.device(torch::kCPU);
  } else {
    options = options.device(torch::kCUDA, device);
  }
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  // Capture deleter and context in a lambda - PyTorch's from_blob passes data ptr to deleter,
  // but we ignore it and use our context instead
  t->tensor = torch::from_blob(
      data, sizeVec,
      [deleter, deleterCtx](void*) {
        deleter(deleterCtx);
      },
      options);
  return t;
}

// Tensor ref counting
void tensorAddRef(Tensor* t) {
  t->refcount.fetch_add(1);
}

void tensorDecRef(Tensor* t) {
  if (t->refcount.fetch_sub(1) == 1) {
    delete t;
  }
}

// Tensor accessors
void* tensorDataPtr(Tensor* t) {
  return t->tensor.data_ptr();
}

int64_t tensorNumel(Tensor* t) {
  return t->tensor.numel();
}

int64_t tensorNdim(Tensor* t) {
  return t->tensor.dim();
}

int64_t tensorSize(Tensor* t, int64_t dim) {
  return t->tensor.size(dim);
}

int64_t tensorStride(Tensor* t, int64_t dim) {
  return t->tensor.stride(dim);
}

DType tensorDtype(Tensor* t) {
  return static_cast<DType>(static_cast<int>(t->tensor.scalar_type()));
}

int tensorDevice(Tensor* t) {
  return t->tensor.device().index();
}

bool tensorIsContiguous(Tensor* t) {
  return t->tensor.is_contiguous();
}

bool tensorIsCuda(Tensor* t) {
  return t->tensor.is_cuda();
}

bool tensorIsCpu(Tensor* t) {
  return t->tensor.is_cpu();
}

int64_t tensorItemsize(Tensor* t) {
  return t->tensor.element_size();
}

// Tensor view operations
Tensor* tensorView(Tensor* t, const int64_t* sizes, int ndim) {
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* result = new Tensor();
  result->tensor = t->tensor.view(sizeVec);
  return result;
}

Tensor* tensorNarrow(Tensor* t, int dim, int64_t start, int64_t length) {
  auto* result = new Tensor();
  result->tensor = t->tensor.narrow(dim, start, length);
  return result;
}

Tensor* tensorContiguous(Tensor* t) {
  auto* result = new Tensor();
  result->tensor = t->tensor.contiguous();
  return result;
}

// Tensor arithmetic
Tensor* tensorMulScalar(Tensor* t, double scalar) {
  auto* result = new Tensor();
  result->tensor = t->tensor * scalar;
  return result;
}

void tensorMulScalar_(Tensor* t, double scalar) {
  t->tensor.mul_(scalar);
}

void tensorRecordStream(Tensor* t, CUstream stream) {
  c10::cuda::CUDACachingAllocator::recordStream(
      t->tensor.storage().data_ptr(), c10::cuda::getStreamFromExternal(stream, t->tensor.device().index()));
}

void tensorCopy_(Tensor* dst, Tensor* src) {
  dst->tensor.copy_(src->tensor);
}

// Tensor reduction operations
void tensorSumOut(Tensor* dst, Tensor* src, int dim) {
  torch::sum_out(dst->tensor, src->tensor, dim);
}

void tensorAmaxOut(Tensor* dst, Tensor* src, int dim) {
  torch::amax_out(dst->tensor, src->tensor, dim);
}

void tensorAminOut(Tensor* dst, Tensor* src, int dim) {
  torch::amin_out(dst->tensor, src->tensor, dim);
}

// Dtype utilities
size_t dtypeSize(DType dtype) {
  return torch::elementSize(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
}

// Free memory callback - delegates to moodist::registerFreeMemoryCallback from processgroup_wrapper.cc
void registerFreeMemoryCallback() {
  ::moodist::registerFreeMemoryCallback();
}

// c10d::Store wrapper functions
void c10dStoreSet(void* store, std::string_view key, const std::vector<uint8_t>& value) {
  auto* c10dStore = static_cast<c10d::Store*>(store);
  c10dStore->set(std::string(key), value);
}

std::vector<uint8_t> c10dStoreGet(void* store, std::string_view key) {
  auto* c10dStore = static_cast<c10d::Store*>(store);
  return c10dStore->get(std::string(key));
}

void c10dStoreWait(void* store, std::span<const std::string> keys) {
  auto* c10dStore = static_cast<c10d::Store*>(store);
  std::vector<std::string> keyVec(keys.begin(), keys.end());
  c10dStore->wait(keyVec);
}

} // namespace wrappers

namespace {

void* libHandle = nullptr;

std::string getLibraryPath() {
  // Get path to this module (_C.so)
  Dl_info info;
  if (dladdr((void*)&getLibraryPath, &info) == 0) {
    throw std::runtime_error("Failed to get _C.so path via dladdr");
  }

  std::string path(info.dli_fname);
  size_t lastSlash = path.rfind('/');
  if (lastSlash == std::string::npos) {
    return "libmoodist.so";
  }

  // Try same directory first
  std::string sameDirPath = path.substr(0, lastSlash + 1) + "libmoodist.so";
  if (access(sameDirPath.c_str(), F_OK) == 0) {
    return sameDirPath;
  }

  // Try parent directory (for multi-version wheels)
  std::string parentPath = path.substr(0, lastSlash);
  lastSlash = parentPath.rfind('/');
  if (lastSlash != std::string::npos) {
    std::string parentDirPath = parentPath.substr(0, lastSlash + 1) + "libmoodist.so";
    if (access(parentDirPath.c_str(), F_OK) == 0) {
      return parentDirPath;
    }
  }

  // Fall back to just the name (let dlopen search)
  return "libmoodist.so";
}

} // namespace

// Global CoreApi object - function pointers copied here after loading
// Using an object (not pointer) means each function pointer access is a single memory read
CoreApi coreApi = {};

void initMoodistApi() {
  if (coreApi.magic != 0) {
    return; // Already initialized
  }

  // Build WrapperApi with pointers to our wrapper functions
  WrapperApi wrapper = {
      .cudaCurrentDevice = wrappers::cudaCurrentDevice,
      .cudaGetCurrentStream = wrappers::cudaGetCurrentStream,
      .cudaStreamGuardEnter = wrappers::cudaStreamGuardEnter,
      .cudaStreamGuardExit = wrappers::cudaStreamGuardExit,
      .cudaCachingAllocatorAlloc = wrappers::cudaCachingAllocatorAlloc,
      .cudaCachingAllocatorFree = wrappers::cudaCachingAllocatorFree,
      .tensorFromBlob = wrappers::tensorFromBlob,
      .tensorEmpty = wrappers::tensorEmpty,
      .tensorFromBlobWithDeleter = wrappers::tensorFromBlobWithDeleter,
      .tensorAddRef = wrappers::tensorAddRef,
      .tensorDecRef = wrappers::tensorDecRef,
      .tensorDataPtr = wrappers::tensorDataPtr,
      .tensorNumel = wrappers::tensorNumel,
      .tensorNdim = wrappers::tensorNdim,
      .tensorSize = wrappers::tensorSize,
      .tensorStride = wrappers::tensorStride,
      .tensorDtype = wrappers::tensorDtype,
      .tensorDevice = wrappers::tensorDevice,
      .tensorIsContiguous = wrappers::tensorIsContiguous,
      .tensorIsCuda = wrappers::tensorIsCuda,
      .tensorIsCpu = wrappers::tensorIsCpu,
      .tensorItemsize = wrappers::tensorItemsize,
      .tensorView = wrappers::tensorView,
      .tensorNarrow = wrappers::tensorNarrow,
      .tensorContiguous = wrappers::tensorContiguous,
      .tensorMulScalar = wrappers::tensorMulScalar,
      .tensorMulScalar_ = wrappers::tensorMulScalar_,
      .tensorRecordStream = wrappers::tensorRecordStream,
      .tensorCopy_ = wrappers::tensorCopy_,
      .tensorSumOut = wrappers::tensorSumOut,
      .tensorAmaxOut = wrappers::tensorAmaxOut,
      .tensorAminOut = wrappers::tensorAminOut,
      .dtypeSize = wrappers::dtypeSize,
      .registerFreeMemoryCallback = wrappers::registerFreeMemoryCallback,
      .c10dStoreSet = wrappers::c10dStoreSet,
      .c10dStoreGet = wrappers::c10dStoreGet,
      .c10dStoreWait = wrappers::c10dStoreWait,
  };

  std::string libPath = getLibraryPath();

  libHandle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    throw std::runtime_error(std::string("Failed to load libmoodist.so: ") + dlerror());
  }

  auto getApi = reinterpret_cast<CoreApi* (*)(uint32_t, const WrapperApi*)>(dlsym(libHandle, "moodistGetApi"));
  if (getApi == nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error(std::string("Failed to find moodistGetApi: ") + dlerror());
  }

  CoreApi* api = getApi(kMoodistApiVersion, &wrapper);
  if (api == nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error("moodist Api version mismatch: _C.so and libmoodist.so are incompatible");
  }

  if (api->magic != kMoodistBuildMagic) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error("moodist build magic mismatch: _C.so and libmoodist.so are from different builds");
  }

  // Copy the CoreApi to our global
  coreApi = *api;
}

TensorPtr wrapTensor(torch::Tensor t) {
  auto* result = new Tensor();
  result->tensor = std::move(t);
  return TensorPtr(result);
}

torch::Tensor unwrapTensor(const TensorPtr& t) {
  if (!t) {
    return torch::Tensor();
  }
  return static_cast<Tensor*>(t.get())->tensor;
}

} // namespace moodist
