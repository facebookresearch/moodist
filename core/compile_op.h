// Copyright (c) Meta Platforms, Inc. and affiliates.

// compile_op implementation
// Compiles a description of input/output tensor mappings into an efficient
// data transfer operation.

#pragma once

#include "api/types.h"
#include "common.h"
#include "cputhread.h"
#include "queue.h"
#include "serialization.h"
#include "shared_ptr.h"

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace moodist {

// Forward declarations
class Queue;
struct Group;

namespace compile_op {

// Use the ReduceOp enum from the API
using ReduceOp = api::ReduceOp;

// N-dimensional coordinate with runtime dimension count
// Used for offsets and shapes in compile_op
struct Coord {
  int64_t* data_ = nullptr;
  int ndim_ = 0;

  Coord() = default;

  explicit Coord(int n) : data_(n > 0 ? static_cast<int64_t*>(internalAlloc(n * sizeof(int64_t))) : nullptr), ndim_(n) {
    if (data_) {
      std::memset(data_, 0, n * sizeof(int64_t));
    }
  }

  ~Coord() {
    if (data_) {
      internalFree(data_);
    }
  }

  // Copy
  Coord(const Coord& o) : Coord(o.ndim_) {
    if (data_) {
      std::memcpy(data_, o.data_, ndim_ * sizeof(int64_t));
    }
  }

  Coord& operator=(const Coord& o) {
    if (this != &o) {
      if (ndim_ != o.ndim_) {
        if (data_) {
          internalFree(data_);
        }
        ndim_ = o.ndim_;
        data_ = ndim_ > 0 ? static_cast<int64_t*>(internalAlloc(ndim_ * sizeof(int64_t))) : nullptr;
      }
      if (data_) {
        std::memcpy(data_, o.data_, ndim_ * sizeof(int64_t));
      }
    }
    return *this;
  }

  // Move
  Coord(Coord&& o) noexcept : data_(o.data_), ndim_(o.ndim_) {
    o.data_ = nullptr;
    o.ndim_ = 0;
  }

  Coord& operator=(Coord&& o) noexcept {
    std::swap(data_, o.data_);
    std::swap(ndim_, o.ndim_);
    return *this;
  }

  int64_t& operator[](size_t i) {
    return data_[i];
  }
  int64_t operator[](size_t i) const {
    return data_[i];
  }

  int64_t* begin() {
    return data_;
  }
  int64_t* end() {
    return data_ + ndim_;
  }
  const int64_t* begin() const {
    return data_;
  }
  const int64_t* end() const {
    return data_ + ndim_;
  }
  size_t size() const {
    return static_cast<size_t>(ndim_);
  }

  // Serialize (const = writing)
  template<typename X>
  void serialize(X& x) const {
    x(ndim_);
    for (int i = 0; i < ndim_; ++i) {
      x(data_[i]);
    }
  }

  // Deserialize (non-const = reading)
  template<typename X>
  void serialize(X& x) {
    int n = x.template read<int>();
    Coord tmp(n);
    for (int i = 0; i < n; ++i) {
      x(tmp.data_[i]);
    }
    *this = std::move(tmp);
  }
};

// Coord operations
inline Coord operator-(const Coord& a, const Coord& b) {
  Coord r(a.ndim_);
  for (size_t i : range(a.size())) {
    r[i] = a[i] - b[i];
  }
  return r;
}

inline Coord operator+(const Coord& a, const Coord& b) {
  Coord r(a.ndim_);
  for (size_t i : range(a.size())) {
    r[i] = a[i] + b[i];
  }
  return r;
}

inline bool operator==(const Coord& a, const Coord& b) {
  if (a.ndim_ != b.ndim_) {
    return false;
  }
  for (size_t i : range(a.size())) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

inline size_t numel(const Coord& c) {
  size_t n = 1;
  for (int64_t x : c) {
    n *= x;
  }
  return n;
}

// Compute linear offset given N-dim offset within a shape (for strides)
inline size_t linearOffset(const Coord& offset, const Coord& shape) {
  size_t result = 0;
  size_t stride = 1;
  for (int i = offset.ndim_ - 1; i >= 0; --i) {
    result += offset[i] * stride;
    stride *= shape[i];
  }
  return result;
}

// Check if inner shape is contiguous within outer shape.
// A region is contiguous if all dimensions after the last mismatch have inner == outer,
// and all dimensions before the last mismatch have inner == 1.
inline bool contiguous(const Coord& inner, const Coord& outer) {
  // Find the last (innermost) dimension where inner != outer
  int lastMismatch = -1;
  for (size_t i : range(inner.size())) {
    if (inner[i] != outer[i]) {
      lastMismatch = static_cast<int>(i);
    }
  }

  // If no mismatch, entire tensor is selected - contiguous
  if (lastMismatch < 0) {
    return true;
  }

  // All dimensions before lastMismatch must have size 1
  for (int i = 0; i < lastMismatch; ++i) {
    if (inner[i] != 1) {
      return false;
    }
  }

  return true;
}

// Context passed from ProcessGroupImpl to compile_op
struct CompileContext {
  Group* group;
  std::span<SharedPtr<Queue>> queues;
  Function<void()> barrier;
  Function<void()> resetBarriers; // Reset step value / barrier arrays (before/after tuning)

  // For generating unique op IDs
  uint32_t* nextOpId;

  Vector<std::pair<uint32_t, TensorDataPtr>> queuedReceives;

  HashMap<std::string, Vector<std::unique_ptr<AllocatedArray>>> cachedBuffers;

  template<typename... T>
  void send(uint32_t torank, const T&... v) {
    CHECK(torank < queues.size());
    auto& q = queues[torank];
    uint32_t t = q->transactionBegin();
    q->putBuffer(serializeToBuffer((uint32_t)group->rank), t);
    q->putBuffer(serializeToBuffer(v...), t);
    q->transactionCommit(t);
  }
  template<typename... T>
  void receive(uint32_t fromrank, T&... v) {
    auto deserialize = [&](auto&& t, auto&... v) {
      deserializeBuffer((void*)t->data(), t->bytes(), v...);
    };
    for (auto i = queuedReceives.begin(); i != queuedReceives.end(); ++i) {
      if (i->first == fromrank) {
        deserialize(i->second, v...);
        queuedReceives.erase(i);
        return;
      }
    }
    auto& q = queues[group->rank];
    while (true) {
      uint32_t sourceRank;
      deserialize(q->get(), sourceRank);
      if (sourceRank == fromrank) {
        deserialize(q->get(), v...);
        return;
      }
      queuedReceives.emplace_back(sourceRank, q->get());
    }
  }
};

// Tensor descriptor (for inputs and outputs)
struct TensorDescr {
  uint32_t rank;
  uint32_t index;
  Coord offset;
  Coord shape;
  size_t numel;
  std::string tensorId; // Groups related tensor regions together
  DeviceType device;    // Device type (CPU or CUDA)
};

// Phase 1: Logical mapping exchanged between ranks (sender → receiver)
// Contains only global coordinates, no precomputed offsets
struct LogicalInput {
  Coord offset; // Global coordinates of intersection
  Coord shape;
  uint32_t inputRank;
  uint32_t inputIndex;
  uint32_t outputRank;
  uint32_t outputIndex;

  template<typename X>
  void serialize(X& x) {
    x(offset, shape, inputRank, inputIndex, outputRank, outputIndex);
  }
};

// Phase 3: Read request (receiver → sender)
// Tells sender exactly which region we need
struct ReadRequest {
  Coord offset; // Global coordinates
  Coord shape;
  uint32_t cellIndex;
  uint32_t requesterRank; // Who is requesting (for response routing)
  uint32_t inputIndex;    // Sender's input index
  uint32_t outputIndex;   // For correlation in response

  template<typename X>
  void serialize(X& x) const {
    x(offset, shape, cellIndex, requesterRank, inputIndex, outputIndex);
  }

  template<typename X>
  void serialize(X& x) {
    x(offset, shape, cellIndex, requesterRank, inputIndex, outputIndex);
  }
};

// Phase 4: Read response (sender → receiver)
// Contains resolved offset and tensor info
struct ReadResponse {
  uint32_t requesterRank; // Who requested (for response routing)
  uint32_t outputIndex;   // Correlation with request
  Coord requestOffset;    // Original request offset (for output offset computation)
  Coord requestShape;     // Original request shape (for output contiguity check)
  uint32_t senderRank;    // Who is sending (for CustomOpDescriptor)
  uint32_t tensorIndex;   // Input index or copy index
  size_t inputOffset;     // Linear offset within tensor
  Coord tensorShape;      // For stride computation at receiver
  uint32_t cellIndex;     // Global cell index for dependency signaling
  bool isCopy;            // True if tensorIndex refers to a copy
  DeviceType inputDevice; // Device type of input tensor

  template<typename X>
  void serialize(X& x) const {
    x(requesterRank, outputIndex, requestOffset, requestShape, senderRank, tensorIndex, inputOffset, tensorShape,
        cellIndex, isCopy, inputDevice);
  }

  template<typename X>
  void serialize(X& x) {
    x(requesterRank, outputIndex, requestOffset, requestShape, senderRank, tensorIndex, inputOffset, tensorShape,
        cellIndex, isCopy, inputDevice);
  }
};

// Main compile function
// Returns a compiled CustomOpDescriptor ready for execution
// ndim is validated per-tensorId (different tensorIds can have different ndims)
std::shared_ptr<CustomOpDescriptor> compile(CompileContext& ctx, DType dtype, std::span<const api::TensorRegion> inputs,
    std::span<const api::TensorRegion> outputs, ReduceOp reduce = ReduceOp::None, bool cpuSync = false);

} // namespace compile_op
} // namespace moodist
