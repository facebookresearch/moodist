// Copyright (c) Meta Platforms, Inc. and affiliates.

// Loads libmoodist.so via dlopen and registers wrapper functions

#include "moodist_api.h"
#include "torch_includes.h"

#include <atomic>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace moodist {

// ============================================================================
// Tensor wrapper - opaque type that holds a torch::Tensor
// ============================================================================

struct Tensor {
  torch::Tensor tensor;
  std::atomic<int> refcount{1};
};

// ============================================================================
// Wrapper function implementations
// ============================================================================

namespace {

// CUDA device/stream functions
int wrapCudaCurrentDevice() {
  return c10::cuda::current_device();
}

CUstream wrapCudaGetCurrentStream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

void* wrapCudaStreamGuardEnter(CUstream stream, int device) {
  auto* guard = new c10::cuda::CUDAStreamGuard(
      c10::cuda::getStreamFromExternal(stream, device));
  return guard;
}

void wrapCudaStreamGuardExit(void* guard) {
  delete static_cast<c10::cuda::CUDAStreamGuard*>(guard);
}

// Tensor creation
Tensor* wrapTensorFromBlob(void* data, const int64_t* sizes, int ndim, DType dtype, int device) {
  auto options = at::TensorOptions()
                     .dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype)))
                     .device(torch::kCUDA, device);
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  t->tensor = torch::from_blob(data, sizeVec, options);
  return t;
}

Tensor* wrapTensorEmpty(const int64_t* sizes, int ndim, DType dtype, int device) {
  auto options = at::TensorOptions()
                     .dtype(static_cast<torch::ScalarType>(static_cast<int>(dtype)))
                     .device(torch::kCUDA, device);
  std::vector<int64_t> sizeVec(sizes, sizes + ndim);
  auto* t = new Tensor();
  t->tensor = torch::empty(sizeVec, options);
  return t;
}

// Tensor ref counting
void wrapTensorAddRef(Tensor* t) {
  t->refcount.fetch_add(1);
}

void wrapTensorDecRef(Tensor* t) {
  if (t->refcount.fetch_sub(1) == 1) {
    delete t;
  }
}

// Tensor accessors
void* wrapTensorDataPtr(Tensor* t) {
  return t->tensor.data_ptr();
}

int64_t wrapTensorNumel(Tensor* t) {
  return t->tensor.numel();
}

int wrapTensorNdim(Tensor* t) {
  return t->tensor.ndimension();
}

int64_t wrapTensorSize(Tensor* t, int dim) {
  return t->tensor.size(dim);
}

int64_t wrapTensorStride(Tensor* t, int dim) {
  return t->tensor.stride(dim);
}

DType wrapTensorDtype(Tensor* t) {
  return static_cast<DType>(static_cast<int>(t->tensor.scalar_type()));
}

int wrapTensorDevice(Tensor* t) {
  return t->tensor.device().index();
}

bool wrapTensorIsContiguous(Tensor* t) {
  return t->tensor.is_contiguous();
}

// Tensor operations
void wrapTensorSumOut(Tensor* dst, Tensor* src, int dim) {
  torch::sum_out(dst->tensor, src->tensor, dim);
}

void wrapTensorAmaxOut(Tensor* dst, Tensor* src, int dim) {
  torch::amax_out(dst->tensor, src->tensor, dim);
}

void wrapTensorAminOut(Tensor* dst, Tensor* src, int dim) {
  torch::amin_out(dst->tensor, src->tensor, dim);
}

// Dtype utilities
size_t wrapDtypeSize(DType dtype) {
  return torch::elementSize(static_cast<torch::ScalarType>(static_cast<int>(dtype)));
}

// Register all wrapper functions in the API struct
void registerWrapperFunctions(MoodistAPI* api) {
  api->cudaCurrentDevice = wrapCudaCurrentDevice;
  api->cudaGetCurrentStream = wrapCudaGetCurrentStream;
  api->cudaStreamGuardEnter = wrapCudaStreamGuardEnter;
  api->cudaStreamGuardExit = wrapCudaStreamGuardExit;
  api->tensorFromBlob = wrapTensorFromBlob;
  api->tensorEmpty = wrapTensorEmpty;
  api->tensorAddRef = wrapTensorAddRef;
  api->tensorDecRef = wrapTensorDecRef;
  api->tensorDataPtr = wrapTensorDataPtr;
  api->tensorNumel = wrapTensorNumel;
  api->tensorNdim = wrapTensorNdim;
  api->tensorSize = wrapTensorSize;
  api->tensorStride = wrapTensorStride;
  api->tensorDtype = wrapTensorDtype;
  api->tensorDevice = wrapTensorDevice;
  api->tensorIsContiguous = wrapTensorIsContiguous;
  api->tensorSumOut = wrapTensorSumOut;
  api->tensorAmaxOut = wrapTensorAmaxOut;
  api->tensorAminOut = wrapTensorAminOut;
  api->dtypeSize = wrapDtypeSize;
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

  // Register wrapper functions so core can call back into _C
  registerWrapperFunctions(api);

  return api;
}

} // namespace moodist
