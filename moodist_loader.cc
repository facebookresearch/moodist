// Copyright (c) Meta Platforms, Inc. and affiliates.

// Loads libmoodist.so via dlopen and registers API function implementations

#include "moodist_loader.h"
#include "moodist_api.h"
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
// API function implementations
// ============================================================================

namespace {

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

// Tensor creation
Tensor* tensorFromBlob(void* data, const int64_t* sizes, int ndim, DType dtype, int device) {
  auto options =
      at::TensorOptions().dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype))).device(torch::kCUDA, device);
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  t->tensor = torch::from_blob(data, sizeVec, options);
  return t;
}

Tensor* tensorEmpty(const int64_t* sizes, int ndim, DType dtype, int device) {
  auto options =
      at::TensorOptions().dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype))).device(torch::kCUDA, device);
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  t->tensor = torch::empty(sizeVec, options);
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

int tensorNdim(Tensor* t) {
  return t->tensor.ndimension();
}

int64_t tensorSize(Tensor* t, int dim) {
  return t->tensor.size(dim);
}

int64_t tensorStride(Tensor* t, int dim) {
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

// Register all API function implementations
void registerAPIFunctions(MoodistAPI* api) {
  api->cudaCurrentDevice = cudaCurrentDevice;
  api->cudaGetCurrentStream = cudaGetCurrentStream;
  api->cudaStreamGuardEnter = cudaStreamGuardEnter;
  api->cudaStreamGuardExit = cudaStreamGuardExit;
  api->tensorFromBlob = tensorFromBlob;
  api->tensorEmpty = tensorEmpty;
  api->tensorAddRef = tensorAddRef;
  api->tensorDecRef = tensorDecRef;
  api->tensorDataPtr = tensorDataPtr;
  api->tensorNumel = tensorNumel;
  api->tensorNdim = tensorNdim;
  api->tensorSize = tensorSize;
  api->tensorStride = tensorStride;
  api->tensorDtype = tensorDtype;
  api->tensorDevice = tensorDevice;
  api->tensorIsContiguous = tensorIsContiguous;
  api->tensorIsCuda = tensorIsCuda;
  api->tensorIsCpu = tensorIsCpu;
  api->tensorItemsize = tensorItemsize;
  api->tensorView = tensorView;
  api->tensorNarrow = tensorNarrow;
  api->tensorContiguous = tensorContiguous;
  api->tensorMulScalar = tensorMulScalar;
  api->tensorMulScalar_ = tensorMulScalar_;
  api->tensorRecordStream = tensorRecordStream;
  api->tensorCopy_ = tensorCopy_;
  api->tensorSumOut = tensorSumOut;
  api->tensorAmaxOut = tensorAmaxOut;
  api->tensorAminOut = tensorAminOut;
  api->dtypeSize = dtypeSize;
}

void* libHandle = nullptr;
MoodistAPI* api = nullptr;

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

MoodistAPI* getMoodistAPI() {
  if (api != nullptr) {
    return api;
  }

  std::string libPath = getLibraryPath();

  libHandle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    throw std::runtime_error(std::string("Failed to load libmoodist.so: ") + dlerror());
  }

  auto getAPI = reinterpret_cast<MoodistAPI* (*)(uint32_t)>(dlsym(libHandle, "moodistGetAPI"));
  if (getAPI == nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error(std::string("Failed to find moodistGetAPI: ") + dlerror());
  }

  api = getAPI(kMoodistAPIVersion);
  if (api == nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error("moodist API version mismatch: _C.so and libmoodist.so are incompatible");
  }

  if (api->magic != kMoodistBuildMagic) {
    dlclose(libHandle);
    libHandle = nullptr;
    api = nullptr;
    throw std::runtime_error("moodist build magic mismatch: _C.so and libmoodist.so are from different builds");
  }

  // Register API function implementations so core can call back into _C
  registerAPIFunctions(api);

  return api;
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
