// Copyright (c) Meta Platforms, Inc. and affiliates.

// compile_op implementation
// Compiles a description of input/output tensor mappings into an efficient
// data transfer operation.

#pragma once

#include "api/types.h"
#include "common.h"
#include "group.h"
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
struct CustomOpDescriptor;

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
};

namespace compile_op {

// Use the ReduceOp enum from the API
using ReduceOp = api::ReduceOp;

struct Graph {

  struct Node {
    uint64_t id;
    uint32_t rank;
    uint32_t tensorIndex;
    size_t offset;
    bool filled;
    DeviceType device;
  };

  struct Edge {
    uint64_t id;
    Vector<Node> sources;
    Vector<Node> destinations;
    size_t bytes;
    uint32_t cellIndex;
    uint32_t executorRank;

    template<typename X>
    void serialize(X& x) {
      x(id, sources, destinations, bytes, cellIndex, executorRank);
    }
  };

  Vector<Edge> edges;
  Vector<Edge> rdmaEdges;
  Vector<Edge> cudaEdges;

  struct CudaTensorMapping {
    uint32_t rank;
    uint32_t tensorIndex;
    size_t destinationSlot;
    size_t stride;
  };

  Vector<CudaTensorMapping> remoteCudaTensorMappings;
  Vector<CudaTensorMapping> localCudaTensorMappings;

  struct TensorDescr {
    size_t bytes;
    DType dtype;
    Vector<int64_t> shape;
    DeviceType device;
  };
  size_t numInputs;
  size_t numInputCopies;
  size_t numOutputs;
  size_t numOutputCopies;
  Vector<TensorDescr> tensors;
};

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

// Main compile function
// Returns a compiled CustomOpDescriptor ready for execution
// ndim is validated per-tensorId (different tensorIds can have different ndims)
std::shared_ptr<CustomOpDescriptor> compile(CompileContext& ctx, DType dtype, std::span<const api::TensorRegion> inputs,
    std::span<const api::TensorRegion> outputs, ReduceOp reduce = ReduceOp::None, bool cpuSync = false);

} // namespace compile_op
} // namespace moodist
