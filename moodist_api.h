// Copyright (c) Meta Platforms, Inc. and affiliates.

// API struct for moodist - bidirectional interface between libmoodist.so and _C.so
// Core functions are populated by moodistGetAPI(), wrapper functions by _C after loading.

#pragma once

#include "moodist_version.h"

#include <Python.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <span>
#include <string_view>
#include <vector>

namespace moodist {

// Forward declarations (opaque types)
struct StoreImpl;
struct Buffer;
struct Tensor; // Opaque wrapper for torch::Tensor

// DType enum - matches torch::ScalarType values
enum class DType : int8_t {
  UInt8 = 0,
  Int8 = 1,
  Int16 = 2,
  Int32 = 3,
  Int64 = 4,
  Float16 = 5,
  Float32 = 6,
  Float64 = 7,
  // Complex types omitted for now
  Bool = 11,
  BFloat16 = 15,
};

// ReduceOp enum - matches c10d::ReduceOp::RedOpType values
enum class ReduceOp : uint8_t {
  SUM = 0,
  AVG = 1,
  PRODUCT = 2,
  MIN = 3,
  MAX = 4,
  BAND = 5,
  BOR = 6,
  BXOR = 7,
  PREMUL_SUM = 8,
};

// API struct containing all function pointers
struct MoodistAPI {
  // Magic value for runtime verification (changes per build)
  uint64_t magic;

  // =========================================================================
  // Core functions - populated by libmoodist.so (moodistGetAPI)
  // =========================================================================

  // Store functions
  StoreImpl* (*createStoreImpl)(std::string_view hostname, int port, std::string_view key, int worldSize, int rank);
  void (*storeImplAddRef)(StoreImpl* impl);
  void (*storeImplDecRef)(StoreImpl* impl);
  void (*storeImplSet)(StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::string_view key,
      const std::vector<uint8_t>& value);
  std::vector<uint8_t> (*storeImplGet)(
      StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::string_view key);
  bool (*storeImplCheck)(
      StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);
  void (*storeImplWait)(
      StoreImpl* impl, std::chrono::steady_clock::duration timeout, std::span<const std::string_view> keys);

  // Serialize functions
  Buffer* (*serializeObjectImpl)(PyObject* o);
  void* (*serializeBufferPtr)(Buffer* buf);
  size_t (*serializeBufferSize)(Buffer* buf);
  void (*serializeBufferAddRef)(Buffer* buf);
  void (*serializeBufferDecRef)(Buffer* buf);
  PyObject* (*deserializeObjectImpl)(const void* ptr, size_t len);

  // =========================================================================
  // Wrapper functions - populated by _C.so after loading
  // These wrap PyTorch APIs that core needs to call
  // =========================================================================

  // CUDA device/stream functions
  int (*cudaCurrentDevice)();
  CUstream (*cudaGetCurrentStream)();
  // Note: CUDAStreamGuard handled via enter/exit pattern
  void* (*cudaStreamGuardEnter)(CUstream stream, int device); // Returns opaque guard handle
  void (*cudaStreamGuardExit)(void* guard);

  // Tensor creation - returns opaque Tensor handle
  Tensor* (*tensorFromBlob)(void* data, const int64_t* sizes, int ndim, DType dtype, int device);
  Tensor* (*tensorEmpty)(const int64_t* sizes, int ndim, DType dtype, int device);

  // Tensor ref counting
  void (*tensorAddRef)(Tensor* t);
  void (*tensorDecRef)(Tensor* t);

  // Tensor accessors
  void* (*tensorDataPtr)(Tensor* t);
  int64_t (*tensorNumel)(Tensor* t);
  int (*tensorNdim)(Tensor* t);
  int64_t (*tensorSize)(Tensor* t, int dim);
  int64_t (*tensorStride)(Tensor* t, int dim);
  DType (*tensorDtype)(Tensor* t);
  int (*tensorDevice)(Tensor* t);
  bool (*tensorIsContiguous)(Tensor* t);
  bool (*tensorIsCuda)(Tensor* t);
  bool (*tensorIsCpu)(Tensor* t);
  int64_t (*tensorItemsize)(Tensor* t);

  // Tensor view operations - return new Tensor handle (caller must decref)
  Tensor* (*tensorView)(Tensor* t, const int64_t* sizes, int ndim);
  Tensor* (*tensorNarrow)(Tensor* t, int dim, int64_t start, int64_t length);
  Tensor* (*tensorContiguous)(Tensor* t);

  // Tensor arithmetic - return new Tensor handle (caller must decref)
  Tensor* (*tensorMulScalar)(Tensor* t, double scalar);

  // Tensor in-place operations
  void (*tensorMulScalar_)(Tensor* t, double scalar);
  void (*tensorRecordStream)(Tensor* t, CUstream stream);
  void (*tensorCopy_)(Tensor* dst, Tensor* src);

  // Tensor reduction operations
  void (*tensorSumOut)(Tensor* dst, Tensor* src, int dim);
  void (*tensorAmaxOut)(Tensor* dst, Tensor* src, int dim);
  void (*tensorAminOut)(Tensor* dst, Tensor* src, int dim);

  // Dtype utilities
  size_t (*dtypeSize)(DType dtype);
};

} // namespace moodist

// C linkage for dlsym
extern "C" {
// Returns API struct if version matches, nullptr otherwise
// Caller should also verify api->magic == kMoodistBuildMagic
__attribute__((visibility("default"))) moodist::MoodistAPI* moodistGetAPI(uint32_t expectedVersion);
}
