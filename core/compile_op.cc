// Copyright (c) Meta Platforms, Inc. and affiliates.

// compile_op implementation; overview in docs/internals/compile_op.md
//
// Phase 1: Exchange logical mappings (sender → receiver)
// Phase 2: Overlap resolution via cell decomposition (local)
// Phase 3: Send read requests (receiver → sender)
// Phase 4: Process requests, generate copies (sender)
// Phase 5: Build the edges (receiver), exchange node ids
// Then: kernel / RDMA split, graph variants (GraphBuilder), kernel tuning (tuneKernels)

#include "compile_op.h"
#include "api/tensor_ptr.h"
#include "compile_op_kernel.h"
#include "cputhread.h"
#include "cuda_loader.h"
#include "group.h"
#include "ipc_mapper.h"
#include "ptx_codegen.h"
#include "queue.h"
#include "serialization.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>

namespace moodist {
namespace compile_op {

namespace {

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

// Helper to format Coord for logging
std::string fmtCoord(const Coord& c) {
  return fmt::to_string(fmt::join(c, ","));
}

// The name of a dtype, as torch has it (for error messages)
const char* dtypeName(DType dtype) {
  switch (dtype) {
  case DType::UInt8:
    return "uint8";
  case DType::Int8:
    return "int8";
  case DType::Int16:
    return "int16";
  case DType::Int32:
    return "int32";
  case DType::Int64:
    return "int64";
  case DType::Float16:
    return "float16";
  case DType::Float32:
    return "float32";
  case DType::Float64:
    return "float64";
  case DType::Bool:
    return "bool";
  case DType::BFloat16:
    return "bfloat16";
  }
  return "unknown";
}

// Parse device string without validation.
// Returns the parsed DeviceType.
// Throws if device string is invalid.
DeviceType parseDevice(std::string_view device) {
  if (device == "cpu") {
    return DeviceType::CPU;
  }
  if (device == "cuda" || device.starts_with("cuda:")) {
    return DeviceType::CUDA;
  }
  throw std::runtime_error("Invalid device string: " + std::string(device));
}

// Parse device string and validate CUDA index if specified.
// Returns the parsed DeviceType.
// Throws if:
//   - device string is invalid
//   - cuda:N is specified but N != expectedCudaIndex
DeviceType parseAndValidateDevice(std::string_view device, int expectedCudaIndex) {
  if (device == "cpu") {
    return DeviceType::CPU;
  }
  if (device == "cuda") {
    return DeviceType::CUDA;
  }
  if (device.starts_with("cuda:")) {
    // Parse the index
    std::string_view indexStr = device.substr(5);
    int index = 0;
    for (char c : indexStr) {
      if (c < '0' || c > '9') {
        throw std::runtime_error("Invalid device string: " + std::string(device));
      }
      index = index * 10 + (c - '0');
    }
    if (index != expectedCudaIndex) {
      throw std::runtime_error("Device " + std::string(device) + " does not match process group's CUDA device " +
                               std::to_string(expectedCudaIndex));
    }
    return DeviceType::CUDA;
  }
  throw std::runtime_error("Invalid device string: " + std::string(device));
}

using Edge = Graph::Edge;
using Node = Graph::Node;

// crossDomainReductions: a partial result that an executor asks a rank in another NVLink domain to compute, and the
// answer: where it will be (the partial rank's scratch tensor) and the notification slot it will be announced in
struct PartialRequest {
  Vector<Node> sources; // in (rank, tensor, offset) order
  size_t bytes;
  uint32_t divisor;
  uint32_t cellIndex;
  uint32_t chunkIndex;
  template<typename X>
  void serialize(X& x) {
    x(sources, bytes, divisor, cellIndex, chunkIndex);
  }
};
struct PartialResponse {
  uint32_t tensorIndex; // in the partial rank's published list (see WorkCustom)
  size_t offset;
  uint32_t slot;
  template<typename X>
  void serialize(X& x) {
    x(tensorIndex, offset, slot);
  }
};

struct InternalContext {
  CompileContext& ctx;

  uint64_t edgeIdCounter = 0;
  uint64_t nextEdgeId() {
    return (1ull << 63) + ((uint64_t)(ctx.group->rank + 1) << 32) + edgeIdCounter++;
  }
};

static Vector<size_t> computeCudaRanks(const Group* group, bool allowFabric) {
  Vector<size_t> cudaRanks = group->ipcRanks;
  if (std::ranges::find(cudaRanks, group->rank) == cudaRanks.end()) {
    cudaRanks.push_back(group->rank);
  }
  std::ranges::sort(cudaRanks);
  if (allowFabric && !group->fabricDomainRanks.empty()) {
    cudaRanks = group->fabricDomainRanks;
  }
  return cudaRanks;
}

struct GraphBuilder {
  InternalContext& ictx;
  CompileContext& ctx = ictx.ctx;
  Group* group = ctx.group;

  const size_t rank = group->rank;
  const size_t size = group->size;

  Graph graph;

  Vector<Edge>& edges = graph.edges;

  Vector<size_t> cudaRanks;

  GraphBuilder(InternalContext& ictx, const Graph& baseGraph) : ictx(ictx) {
    graph = baseGraph;
    cudaRanks = computeCudaRanks(group, !ctx.collectiveOp);
  }

  void multicast() {
    HashMap<uint32_t, Vector<Edge>> singleSourceNodes;
    for (auto i = edges.begin(); i != edges.end();) {
      auto& e = *i;
      if (e.sources.size() != 1 || !e.sourceSignals.empty() || e.doneSignal >= 0) {
        // a relay edge (no sources) or a reduction (several sources, see mergeReductions, or part of one across
        // NVLink domains, see crossDomainReductions), which stays with its executor
        ++i;
        continue;
      }
      CHECK(e.destinations.size() == 1);
      singleSourceNodes[e.sources[0].rank].push_back(e);
      i = edges.erase(i);
    }
    for (size_t i : range(size)) {
      ctx.send(i, singleSourceNodes[i]);
    }
    HashMap<uint64_t, size_t> nodeEdgeIndex;
    Vector<Edge> el;
    for (size_t i : range(size)) {
      ctx.receive(i, el);
      for (const Edge& e : el) {
        CHECK(e.sources.size() == 1 && e.sources[0].rank == rank);
        CHECK(e.destinations.size() == 1);
        uint64_t sourceNode = e.sources[0].id;
        auto it = nodeEdgeIndex.find(sourceNode);
        if (it == nodeEdgeIndex.end()) {
          nodeEdgeIndex[sourceNode] = edges.size();
          edges.push_back(e);
          edges.back().executorRank = rank;
          edges.back().id = ictx.nextEdgeId();
        } else {
          auto& targetEdge = edges[it->second];
          CHECK(targetEdge.cellIndex == e.cellIndex && targetEdge.bytes == e.bytes);
          CHECK(targetEdge.sources[0].id == e.sources[0].id);
          targetEdge.destinations.push_back(e.destinations[0]);
        }
      }
    }
    for (auto& e : edges) {
      auto proj = [](auto& n) {
        return n.rank;
      };
      std::ranges::sort(e.destinations, {}, proj);
      std::ranges::rotate(e.destinations, std::ranges::lower_bound(e.destinations, rank, {}, proj));
    }
  }

  // Reductions: the edges of a piece of a reduced cell (see reductionPieces), one built by each destination rank of
  // the cell, have the same sources and the same executor. Merge them into one edge with every destination, executed
  // by that rank. Its sources start at the executor's own, if any, and continue in rank order from there, and so do
  // its destinations: the ranks then read from and write to different ranks at the same time (e.g. in a
  // reduce-scatter, rank r reads from ranks r, r + 1, r + 2, ...), and the kernel combines the sources in this order.
  // (compile() merges the base graph's edges, before crossDomainReductions)
  void mergeReductions() {
    CHECK(isReduction(graph.reduce));
    HashMap<uint32_t, Vector<Edge>> byExecutor;
    for (auto i = edges.begin(); i != edges.end();) {
      if (i->sources.size() < 2) {
        ++i;
        continue;
      }
      CHECK(i->destinations.size() == 1);
      byExecutor[i->executorRank].push_back(std::move(*i));
      i = edges.erase(i);
    }
    for (size_t i : range(size)) {
      ctx.send(i, byExecutor[i]);
    }
    HashMap<std::string, size_t> merged; // the ids of an edge's sources -> its index in edges
    Vector<Edge> el;
    for (size_t i : range(size)) {
      ctx.receive(i, el);
      for (Edge& e : el) {
        CHECK(e.executorRank == rank);
        std::string key;
        for (const Node& n : e.sources) {
          key.append((const char*)&n.id, sizeof(n.id));
        }
        auto it = merged.find(key);
        if (it == merged.end()) {
          merged[key] = edges.size();
          e.id = ictx.nextEdgeId();
          edges.push_back(std::move(e));
        } else {
          Edge& target = edges[it->second];
          CHECK(target.bytes == e.bytes && target.cellIndex == e.cellIndex && target.chunkIndex == e.chunkIndex);
          target.destinations.push_back(e.destinations.at(0));
        }
      }
    }
    auto proj = [](const Node& n) {
      return n.rank;
    };
    for (Edge& e : edges) {
      if (e.sources.size() < 2) {
        continue;
      }
      // (the sources are in rank order, see Phase 5)
      std::ranges::rotate(e.sources, std::ranges::lower_bound(e.sources, rank, {}, proj));
      std::ranges::sort(e.destinations, {}, proj);
      std::ranges::rotate(e.destinations, std::ranges::lower_bound(e.destinations, rank, {}, proj));
    }
  }

  void constructCudaTensorMappings() {
    Vector<uint32_t> nranks;
    for (auto& v : graph.cudaEdges) {
      Vector<uint32_t> tensors;
      Vector<uint32_t> ranks;
      for (auto& l : {&v.sources, &v.destinations}) {
        for (auto& x : *l) {
          if (x.rank == rank) {
            tensors.push_back(x.tensorIndex);
          } else {
            ranks.push_back(x.rank);
            if (std::ranges::find(nranks, x.rank) == nranks.end()) {
              nranks.push_back(x.rank);
            }
          }
        }
      }
      for (uint32_t t : tensors) {
        for (uint32_t r : ranks) {
          bool found = false;
          for (auto& v : graph.remoteCudaTensorMappings) {
            if (v.rank == r && v.tensorIndex == t) {
              found = true;
              break;
            }
          }
          if (!found) {
            graph.remoteCudaTensorMappings.emplace_back();
            auto& v = graph.remoteCudaTensorMappings.back();
            v.rank = r;
            v.tensorIndex = t;
            v.destinationSlot = -1;
            v.stride = 0;
          }
        }
      }
    }
    for (uint32_t rank : nranks) {
      Vector<uint32_t> c;
      for (auto& x : graph.remoteCudaTensorMappings) {
        if (x.rank == rank) {
          c.push_back(x.tensorIndex);
        }
      }
      ctx.send(rank, c);
    }
    Vector<Graph::CudaTensorMapping> localSlots;
    for (uint32_t rank : nranks) {
      Vector<uint32_t> c;
      ctx.receive(rank, c);
      size_t offset = localSlots.size();
      for (uint32_t id : c) {
        Graph::CudaTensorMapping x;
        x.rank = rank;
        x.tensorIndex = id;
        x.destinationSlot = localSlots.size();
        x.stride = 0;
        localSlots.push_back(x);
      }
      ctx.send(rank, offset, c.size());
    }
    for (uint32_t rank : nranks) {
      size_t offset;
      size_t n;
      ctx.receive(rank, offset, n);
      for (auto& x : graph.remoteCudaTensorMappings) {
        if (x.rank == rank) {
          x.destinationSlot = offset++;
          --n;
        }
      }
      CHECK(n == 0);
    }
    graph.localCudaTensorMappings = localSlots;
    for (uint32_t rank : nranks) {
      ctx.send(rank, localSlots.size());
    }
    for (uint32_t rank : nranks) {
      size_t n;
      ctx.receive(rank, n);
      for (auto& x : graph.remoteCudaTensorMappings) {
        if (x.rank == rank) {
          x.stride = n;
        }
      }
    }
    for (auto& x : graph.remoteCudaTensorMappings) {
      CHECK(x.stride != 0);
    }
  }

  void finalize() {
    HashMap<uint32_t, Vector<Edge>> edgePerRank;
    HashMap<uint64_t, size_t> localEdges;
    for (const Edge& e : edges) {
      CHECK(!localEdges.contains(e.id));
      localEdges[e.id] = &e - edges.data();
      // (once per rank: a rank can have several nodes in an edge, e.g. a source and a destination in a reduction)
      Vector<uint32_t> sentTo;
      for (auto* l : {&e.sources, &e.destinations}) {
        for (const Node& n : *l) {
          if (n.rank != rank && std::ranges::find(sentTo, n.rank) == sentTo.end()) {
            sentTo.push_back(n.rank);
            edgePerRank[n.rank].push_back(e);
          }
        }
      }
    }

    for (size_t i : range(size)) {
      ctx.send(i, edgePerRank[i]);
    }
    Vector<Edge> el;
    for (size_t i : range(size)) {
      ctx.receive(i, el);
      for (const Edge& e : el) {
        CHECK(!localEdges.contains(e.id));
        localEdges[e.id] = edges.size();
        edges.push_back(e);
      }
    }

    for (auto& edge : edges) {
      bool cuda = true;
      bool local = true;
      for (auto& l : {&edge.sources, &edge.destinations}) {
        for (auto& n : *l) {
          if (n.device != DeviceType::CUDA) {
            cuda = false;
          }
          if (n.rank != rank && std::ranges::find(cudaRanks, n.rank) == cudaRanks.end()) {
            local = false;
          }
        }
      }
      if (local & cuda) {
        graph.cudaEdges.push_back(std::move(edge));
      } else {
        graph.rdmaEdges.push_back(std::move(edge));
      }
    }

    constructCudaTensorMappings();
  }

  void ring(bool writeMode) {
    // Chains are per cell, and per chunk for cells whose proxied transfer is split into chunks
    auto key = [](const Edge& e) {
      return ((uint64_t)e.cellIndex << 32) | e.chunkIndex;
    };
    Vector<std::tuple<uint64_t, size_t, Node>> myReceives;
    HashMap<uint64_t, Edge> edgeMap;

    for (auto& edge : edges) {
      // we could, technically, have multiple overlapping outputs.
      // these should not all go into the ring, though - instead all but one
      // should be leaf nodes going off of the one that is part of the ring
      // fixme: these should probably not be separate edges at this point.
      //        we need some passes that merge sources and destinations on the same edge
      //        and also consider splitting cuda and cpu destinations on the same edge
      //        mixed cpu & cuda sources is a problem for later (reductions)

      // fixme: why is this done by cell indices at all?
      //        this should be done by node ids. we know exactly the node id of the source(s),
      //        and this is what we should group on, not cell index.
      CHECK(!edgeMap.contains(key(edge)));
      edgeMap[key(edge)] = edge;
      for (auto& d : edge.destinations) {
        CHECK(d.rank == rank);
        if (d.device == DeviceType::CUDA) {
          myReceives.emplace_back(key(edge), edge.bytes, d);
          break;
        }
      }
    }

    for (size_t i : cudaRanks) {
      ctx.send(i, myReceives);
    }

    HashMap<uint64_t, Vector<Node>> groups;

    for (size_t i : cudaRanks) {
      Vector<std::tuple<uint64_t, size_t, Node>> nReceives;
      ctx.receive(i, nReceives);

      for (auto [cell, bytes, node] : nReceives) {
        if (!edgeMap.contains(cell)) {
          continue;
        }
        CHECK(edgeMap[cell].bytes == bytes);
        CHECK(node.rank == i);
        groups[cell].push_back(node);
      }
    }

    for (auto& [cell, ranks] : groups) {
      CHECK(ranks.size() != 0);
      if (ranks.size() == 1) {
        continue;
      }
      auto proj = [](Node& x) {
        return x.rank;
      };
      std::ranges::sort(ranks, {}, proj);

      // TEMPORARY rotate around source rank
      // works okay for local-only,
      // TERRIBLE for remote sources
      // FIXME proper load distribution
      CHECK(edgeMap.contains(cell));
      auto& edge = edgeMap.at(cell);
      // A relay edge (no sources; the proxy rank of a transfer over RDMA, whose node the other ranks' edges
      // read) heads the chain.
      uint32_t sourceRank = edge.sources.empty() ? rank : edge.sources[0].rank;
      std::ranges::rotate(ranks, std::ranges::lower_bound(ranks, sourceRank, {}, proj));

      Vector<uint32_t> rankns;
      for (auto& v : ranks) {
        rankns.push_back(v.rank);
      }
      log.debug("%d: ranks for cell %d chunk %d is [%s]\n", rank, cell >> 32, cell & 0xffffffff,
          fmt::to_string(fmt::join(rankns, ", ")));

      auto it = std::ranges::find(ranks, rank, proj);
      CHECK(it != ranks.end());
      if (it != ranks.begin()) {
        auto prev = *std::prev(it);
        CHECK(edge.sources.size() == 1);
        edge.sources.clear();
        edge.sources.push_back(prev);
        edge.id = ictx.nextEdgeId();
        CHECK(!prev.filled);
      }
    }

    edges.clear();
    for (auto& v : edgeMap) {
      edges.push_back(v.second);
    }

    for (auto& e : edges) {
      if (e.sources.empty()) {
        CHECK(e.rdmaSignal >= 0 && e.executorRank == rank);
        continue;
      }
      CHECK(e.sources.size() == 1);
      CHECK(e.destinations.size() == 1);
      if (writeMode) {
        e.executorRank = e.sources[0].rank;
      } else {
        e.executorRank = e.destinations[0].rank;
      }
    }
  }
};

Graph buildNothing(InternalContext& ictx, const Graph& baseGraph) {
  GraphBuilder builder(ictx, baseGraph);
  builder.finalize();
  return builder.graph;
}

Graph buildMulticast(InternalContext& ictx, const Graph& baseGraph) {
  GraphBuilder builder(ictx, baseGraph);
  builder.multicast();
  builder.finalize();
  return builder.graph;
}

Graph buildRing(InternalContext& ictx, const Graph& baseGraph, bool writeMode) {
  GraphBuilder builder(ictx, baseGraph);
  CHECK(!isReduction(baseGraph.reduce));
  builder.ring(writeMode);
  builder.finalize();
  return builder.graph;
}

void logGraph(const Graph& graph) {
  auto f = [&](auto& l) {
    std::string s;
    for (const auto& n : l) {
      if (!s.empty()) {
        s += ", ";
      }
      s += fmt::sprintf("%#x r%u:t%u+%zu%s", n.id, n.rank, n.tensorIndex, n.offset, n.filled ? "f" : "w");
    }
    return s;
  };
  auto signal = [](const Edge& e) {
    std::string s = e.rdmaSignal >= 0 ? fmt::sprintf(" rdmaSignal=%d", e.rdmaSignal) : std::string();
    if (!e.sourceSignals.empty()) {
      s += fmt::sprintf(" sourceSignals=[%s] partialSources=%u divisor=%u",
          fmt::to_string(fmt::join(e.sourceSignals, ",")), e.partialSources, e.divisor);
    }
    if (e.doneSignal >= 0) {
      s += fmt::sprintf(" doneSignal=%d divisor=%u", e.doneSignal, e.divisor);
    }
    return s;
  };
  for (size_t i : indices(graph.rdmaEdges)) {
    const auto& e = graph.rdmaEdges[i];
    log.info("  rdmaEdge[%zu]: %#x %zu bytes executor=%d cell=%u%s [%s] -> [%s]\n", i, e.id, e.bytes, e.executorRank,
        e.cellIndex, signal(e), f(e.sources), f(e.destinations));
  }
  for (size_t i : indices(graph.cudaEdges)) {
    const auto& e = graph.cudaEdges[i];
    log.info("  cudaEdge[%zu]: %#x %zu bytes executor=%d cell=%u%s [%s] -> [%s]\n", i, e.id, e.bytes, e.executorRank,
        e.cellIndex, signal(e), f(e.sources), f(e.destinations));
  }
}

struct CompileOpConstructor {

  CompileContext& ctx;

  const Group* group = ctx.group;

  const size_t rank = group->rank;
  const size_t size = group->size;

  uint64_t nodeId(uint32_t rank, uint32_t localIndex) {
    return ((uint64_t)(rank + 1) << 32) + localIndex;
  }

  struct Setting {
    Graph graph;
    KernelConfig config;
  };

  // The value of the environment variable `name` (or `defaultValue` if unset), which must be the same on every
  // rank (it selects collective steps).
  std::string agreedEnv(const char* name, const char* defaultValue) {
    std::string value = defaultValue;
    if (const char* env = std::getenv(name)) {
      value = env;
    }
    for (size_t i : range(size)) {
      ctx.send(i, value);
    }
    bool mismatch = false;
    for (size_t i : range(size)) {
      std::string s;
      ctx.receive(i, s);
      mismatch |= s != value;
    }
    REQUIRE(!mismatch, "compile_op: %s differs between ranks (this rank: '%s')", name, value);
    return value;
  }

  // A comma separated list from agreedEnv.
  Vector<std::string> agreedEnvList(const char* name, const char* defaultValue) {
    std::string value = agreedEnv(name, defaultValue);
    Vector<std::string> r;
    for (size_t pos = 0; pos <= value.size();) {
      size_t end = std::min(value.find(',', pos), value.size());
      r.push_back(value.substr(pos, end - pos));
      pos = end + 1;
    }
    return r;
  }

  // Mixed NVLink + RDMA transfers: data that several ranks of an NVLink domain (the ranks this rank's kernel can
  // reach, computeCudaRanks) need from a rank outside the domain should cross the network once. The ranks of a
  // domain exchange their edges from outside the domain and group them by source node. In a group with two or
  // more destinations, one destination rank, the proxy, reads the data over RDMA into its own destination node,
  // and the other destinations copy it from there with the kernel: their edges' source becomes the proxy's
  // node. The proxy's RDMA edge gets a signal index (rdmaSignal): its CPU thread signals the kernel once the
  // read has completed. A relay edge on the proxy (no sources, the proxy's node as destination, same signal
  // index) makes the kernel wait for that signal and then mark the node as filled, after which the copies from
  // it run as dependent edges in every graph variant. Large transfers are split into chunks, each with its own
  // read, signal, relay edge and copies (chunkIndex), so that the copies overlap the reads.
  // Proxied are only groups whose copies the kernel can take (CUDA destinations, sizes and destination offsets
  // multiples of 16 bytes) and that have one destination per rank. The proxy is a destination with the same local
  // rank as the source if there is one (NICs on the same rail), and among the candidates the one that proxies the
  // fewest bytes so far. Every member of a domain computes the same assignment from the same exchanged lists.
  void proxyRemoteEdges(InternalContext& ictx, Vector<Edge>& edges) {
    Vector<size_t> domain = computeCudaRanks(group, !ctx.collectiveOp);
    auto inDomain = [&](uint32_t r) {
      return std::ranges::find(domain, r) != domain.end();
    };

    Vector<Edge> candidates;
    for (const Edge& e : edges) {
      if (e.sources.size() != 1 || e.destinations.size() != 1) {
        continue;
      }
      const Node& s = e.sources[0];
      const Node& d = e.destinations[0];
      CHECK(d.rank == rank);
      if (inDomain(s.rank) || d.device != DeviceType::CUDA || e.bytes % 16 != 0 || d.offset % 16 != 0) {
        continue;
      }
      candidates.push_back(e);
    }
    for (size_t r : domain) {
      ctx.send(r, domain, candidates);
    }
    HashMap<uint64_t, Vector<Edge>> groups;
    for (size_t r : domain) {
      Vector<size_t> peerDomain;
      Vector<Edge> l;
      ctx.receive(r, peerDomain, l);
      CHECK(peerDomain == domain);
      for (Edge& e : l) {
        CHECK(e.destinations.at(0).rank == r);
        groups[e.sources.at(0).id].push_back(std::move(e));
      }
    }
    // Each domain takes the sources in rank order starting after its own first rank. The CPU thread issues a
    // proxy's reads in signal order, i.e. in this order, a few at a time: with the same order in every domain,
    // all domains would read from the same sources first (e.g. with four domains, three readers on one source and
    // none on others), with rotated orders they read from different ones.
    Vector<uint64_t> sourceIds;
    for (auto& v : groups) {
      sourceIds.push_back(v.first);
    }
    std::ranges::sort(sourceIds, {}, [&](uint64_t id) {
      size_t sourceRank = groups[id].at(0).sources.at(0).rank;
      return std::make_pair((sourceRank + size - domain.at(0)) % size, id);
    });

    // Signal flags live in cpuOutBuffer after its first 16 slots
    const size_t maxSignals = group->cpuOutBuffer.itembytes / sizeof(uint32_t) - 16;
    static constexpr size_t minChunkBytes = 4 << 20;
    static constexpr size_t maxTransferChunks = 32;
    HashMap<uint32_t, size_t> proxyBytes;
    HashMap<uint32_t, size_t> proxySignals;
    HashMap<uint32_t, uint32_t> chunkNodeCounter;
    size_t numGroups = 0;
    size_t numChunksTotal = 0;
    size_t numMine = 0;
    for (uint64_t sourceId : sourceIds) {
      Vector<Edge>& g = groups[sourceId];
      if (g.size() < 2) {
        continue;
      }
      std::ranges::sort(g, {}, [](const Edge& e) {
        return e.destinations[0].rank;
      });
      bool eligible = true;
      for (size_t i : indices(g)) {
        eligible &= g[i].bytes == g[0].bytes && g[i].cellIndex == g[0].cellIndex;
        eligible &= i == 0 || g[i].destinations[0].rank != g[i - 1].destinations[0].rank;
      }
      if (!eligible) {
        continue;
      }
      // Split transfers of at least 2 * minChunkBytes into chunks: each chunk is a read with its own signal, a relay
      // edge and copies, so that the copies of a chunk can start as soon as it has arrived
      const size_t bytes = g[0].bytes;
      size_t numChunks = 1;
      size_t chunkBytes = bytes;
      if (bytes >= 2 * minChunkBytes) {
        numChunks = std::min(maxTransferChunks, bytes / minChunkBytes);
        chunkBytes = ((bytes + numChunks - 1) / numChunks + 4095) / 4096 * 4096;
        numChunks = (bytes + chunkBytes - 1) / chunkBytes;
      }
      // The least loaded destination with the source's local rank (a domain that spans several nodes has one per
      // node), else the least loaded one
      uint32_t sourceRank = g[0].sources[0].rank;
      auto pick = [&](bool sameLocalRank) {
        size_t best = g.size();
        for (size_t m : indices(g)) {
          uint32_t r = g[m].destinations[0].rank;
          if (proxySignals[r] + numChunks > maxSignals ||
              (sameLocalRank && group->rankLocalRank.at(r) != group->rankLocalRank.at(sourceRank))) {
            continue;
          }
          if (best == g.size() || proxyBytes[r] < proxyBytes[g[best].destinations[0].rank]) {
            best = m;
          }
        }
        return best;
      };
      size_t proxyIndex = pick(true);
      if (proxyIndex == g.size()) {
        proxyIndex = pick(false);
      }
      if (proxyIndex == g.size()) {
        continue;
      }
      const uint32_t proxyRank = g[proxyIndex].destinations[0].rank;
      const int32_t firstSignal = proxySignals[proxyRank];
      proxySignals[proxyRank] += numChunks;
      proxyBytes[proxyRank] += bytes;
      ++numGroups;
      numChunksTotal += numChunks;

      // chunkNodes[m][c]: chunk c of member m's destination node. A single chunk keeps the node itself; split
      // transfers get new nodes, with ids from a range their owner never assigns (bit 31 of the local index),
      // numbered in the order of this pass, which every member of the domain follows.
      Vector<Vector<Node>> chunkNodes(g.size());
      for (size_t c : range(numChunks)) {
        for (size_t m : indices(g)) {
          Node node = g[m].destinations[0];
          if (numChunks > 1) {
            node.id = ((uint64_t)(node.rank + 1) << 32) | (0x80000000u + chunkNodeCounter[node.rank]++);
            node.offset += chunkBytes * c;
          }
          chunkNodes[m].push_back(node);
        }
      }

      for (size_t m : indices(g)) {
        if (g[m].destinations[0].rank != rank) {
          continue;
        }
        ++numMine;
        auto it = std::ranges::find(edges, g[m].id, &Edge::id);
        CHECK(it != edges.end());
        Edge original = std::move(*it);
        edges.erase(it);
        for (size_t c : range(numChunks)) {
          size_t offset = chunkBytes * c;
          size_t n = std::min(chunkBytes, bytes - offset);
          if (rank == proxyRank) {
            // The RDMA read of the chunk (an ibRead after the split), signaling once done
            Edge read = original;
            read.id = ictx.nextEdgeId();
            read.sources[0].offset += offset;
            read.destinations[0] = chunkNodes[m][c];
            read.bytes = n;
            read.rdmaSignal = firstSignal + c;
            read.chunkIndex = c;
            edges.push_back(std::move(read));
            Edge relay;
            relay.id = ictx.nextEdgeId();
            relay.destinations.push_back(chunkNodes[m][c]);
            relay.bytes = n;
            relay.cellIndex = original.cellIndex;
            relay.executorRank = rank;
            relay.rdmaSignal = firstSignal + c;
            relay.chunkIndex = c;
            edges.push_back(std::move(relay));
          } else {
            Edge copy = original;
            copy.id = ictx.nextEdgeId();
            copy.sources.clear();
            copy.sources.push_back(chunkNodes[proxyIndex][c]);
            copy.destinations[0] = chunkNodes[m][c];
            copy.bytes = n;
            copy.chunkIndex = c;
            edges.push_back(std::move(copy));
          }
        }
      }
    }
    log.debug("compile_op: rank %zu: %zu proxied groups (%zu chunks) in a domain of %zu ranks (%zu of my edges, %zu "
              "signals as proxy)\n",
        rank, numGroups, numChunksTotal, domain.size(), numMine, proxySignals[rank]);
  }

  // Reductions across NVLink domains. The executor of a reduced piece (see reductionPieces and mergeReductions)
  // reads the sources in its own domain over NVLink, and those of other domains over RDMA, combined within each
  // domain first: for every other domain with several sources, a rank there (the partial rank, see pickPartialRank)
  // computes their partial result into its scratch tensor (an edge with a doneSignal), its CPU thread notifies the
  // executor's once the kernel has stored it (CustomOpDescriptor::Notification), and the executor's CPU thread reads
  // it into the executor's scratch tensor and signals the kernel (an ibRead with a waitSlot and a signalIndex). The
  // input of a domain with one source is read as it is. The executor's edge then combines the sources of its own
  // domain (in the order of mergeReductions), the inputs read from other domains, and the partial results (in the
  // order of their domains' first ranks), each once its read has arrived (Edge::sourceSignals). The kernels compute
  // the partial results before anything else (see generateKernel). Partial results are of the op's dtype (16-bit
  // floats are rounded once per domain); for an average, they are already divided by the number of all inputs
  // (Edge::partialSources, divisor), which keeps the sums of many 16-bit floats in range.
  // So far for pieces whose destinations are all in the executor's domain (e.g. reduce-scatter, reduce to one rank):
  // the others remain as they are, an error below. Every rank takes part (the requests are exchanged); returns an
  // error message (a full signal table), or an empty string.
  std::string crossDomainReductions(
      InternalContext& ictx, Graph& graph, CustomOpDescriptor& op, const Vector<uint32_t>& domainIds) {
    const uint32_t myDomain = domainIds.at(rank);
    auto inMyDomain = [&](const Node& n) {
      return domainIds.at(n.rank) == myDomain;
    };
    // The domains in the order of their first ranks, and every rank's position in its domain
    Vector<uint32_t> firstRanks = domainIds;
    std::ranges::sort(firstRanks);
    firstRanks.erase(std::unique(firstRanks.begin(), firstRanks.end()), firstRanks.end());
    const size_t numDomains = firstRanks.size();
    Vector<size_t> domainIndex(size);
    Vector<size_t> position(size);
    Vector<size_t> domainSize(numDomains);
    for (size_t r : range(size)) {
      domainIndex[r] = std::ranges::lower_bound(firstRanks, domainIds[r]) - firstRanks.begin();
      position[r] = domainSize[domainIndex[r]]++;
    }
    auto domainRanks = [&](uint32_t domain) {
      Vector<uint32_t> r;
      for (size_t i : range(size)) {
        if (domainIds[i] == domain) {
          r.push_back(i);
        }
      }
      return r;
    };
    // The rank of `domain` that computes the k-th partial result of that domain for this rank: one with this rank's
    // local rank (NICs on the same rail), the one at this rank's position among the ranks of its domain with that
    // local rank, plus k (so that with domains of the same shape, every rank computes as many partial results)
    auto pickPartialRank = [&](uint32_t domain, size_t k) {
      const size_t localRank = group->rankLocalRank.at(rank);
      Vector<uint32_t> all = domainRanks(domain);
      Vector<uint32_t> candidates;
      for (uint32_t r : all) {
        if (group->rankLocalRank.at(r) == localRank) {
          candidates.push_back(r);
        }
      }
      if (candidates.empty()) {
        candidates = all;
      }
      size_t railPosition = 0;
      for (uint32_t r : domainRanks(myDomain)) {
        if (r == rank) {
          break;
        }
        railPosition += group->rankLocalRank.at(r) == localRank;
      }
      return candidates.at((railPosition + k) % candidates.size());
    };

    // The scratch tensor (the last tensor, see CustomOpDescriptor::scratchBytes): its regions start at page aligned
    // offsets (RDMA reads into page aligned destinations are faster)
    const uint32_t scratchIndex = graph.tensors.size();
    const uint32_t scratchOutputIndex = scratchIndex - graph.numInputs - graph.numInputCopies;
    size_t scratchBytes = 0;
    auto allocate = [&](size_t bytes) {
      size_t offset = scratchBytes;
      scratchBytes += (bytes + 4095) / 4096 * 4096;
      return offset;
    };
    uint32_t nextNodeIndex = 0;
    auto scratchNode = [&](size_t offset, bool filled) {
      Node n;
      // (ids from a range of this rank's that Phase 5 and proxyRemoteEdges do not use)
      n.id = ((uint64_t)(rank + 1) << 32) | (0xc0000000u + nextNodeIndex++);
      n.rank = rank;
      n.tensorIndex = scratchIndex;
      n.offset = offset;
      n.filled = filled;
      n.device = DeviceType::CUDA;
      return n;
    };
    int32_t nextSignal = 0;
    auto read = [&](uint32_t sourceRank, uint32_t inputIndex, size_t inputOffset, size_t bytes, int32_t waitSlot) {
      CustomOpDescriptor::IbRead r;
      r.rank = sourceRank;
      r.inputIndex = inputIndex;
      r.outputIndex = scratchOutputIndex;
      r.inputOffset = inputOffset;
      r.outputOffset = allocate(bytes);
      r.bytes = bytes;
      r.signalIndex = nextSignal++;
      r.waitSlot = waitSlot;
      op.ibReads.push_back(r);
      log.info("  ibRead: rank %u input[%u]+%zu -> output[%u]+%zu, %zu bytes, signal %d, wait slot %d\n", r.rank,
          r.inputIndex, r.inputOffset, r.outputIndex, r.outputOffset, r.bytes, r.signalIndex, r.waitSlot);
      return r;
    };

    // The new sources of an edge of this rank's; the partial results' are filled in from the responses
    struct Plan {
      size_t edgeIndex;
      Vector<Node> sources;
      Vector<int32_t> signals;
      uint32_t partialSources = 0;
      uint32_t divisor;
      Vector<std::tuple<size_t, uint32_t, size_t>> partials; // (source position, partial rank, request index)
    };
    Vector<Plan> plans;
    Vector<Vector<PartialRequest>> requests(size);
    HashMap<uint32_t, size_t> requestedFrom; // per domain, partial results requested so far
    for (size_t ei : indices(graph.edges)) {
      const Edge& e = graph.edges[ei];
      if (e.sources.size() < 2 || std::ranges::all_of(e.sources, inMyDomain)) {
        continue;
      }
      bool supported = std::ranges::all_of(e.destinations, inMyDomain);
      for (const auto* l : {&e.sources, &e.destinations}) {
        for (const Node& n : *l) {
          supported &= n.device == DeviceType::CUDA;
        }
      }
      if (!supported) {
        continue;
      }
      CHECK(e.executorRank == rank);
      Plan plan;
      plan.edgeIndex = ei;
      plan.divisor = e.sources.size();
      Vector<uint32_t> domains;
      for (const Node& n : e.sources) {
        if (inMyDomain(n)) {
          plan.sources.push_back(n);
          plan.signals.push_back(-1);
        } else if (std::ranges::find(domains, domainIds[n.rank]) == domains.end()) {
          domains.push_back(domainIds[n.rank]);
        }
      }
      std::ranges::sort(domains);
      auto domainSources = [&](uint32_t domain) {
        Vector<Node> r;
        for (const Node& n : e.sources) {
          if (domainIds[n.rank] == domain) {
            r.push_back(n);
          }
        }
        std::ranges::sort(r, {}, [](const Node& n) {
          return std::make_tuple(n.rank, n.tensorIndex, n.offset);
        });
        return r;
      };
      // The inputs of domains with one source, as they are
      for (uint32_t domain : domains) {
        Vector<Node> s = domainSources(domain);
        if (s.size() == 1) {
          auto r = read(s[0].rank, s[0].tensorIndex, s[0].offset, e.bytes, -1);
          plan.sources.push_back(scratchNode(r.outputOffset, true));
          plan.signals.push_back(r.signalIndex);
        }
      }
      // The partial results of the others
      for (uint32_t domain : domains) {
        Vector<Node> s = domainSources(domain);
        if (s.size() == 1) {
          continue;
        }
        uint32_t partialRank = pickPartialRank(domain, requestedFrom[domain]++);
        plan.partials.emplace_back(plan.sources.size(), partialRank, requests[partialRank].size());
        PartialRequest request;
        request.sources = std::move(s);
        request.bytes = e.bytes;
        request.divisor = plan.divisor;
        request.cellIndex = e.cellIndex;
        request.chunkIndex = e.chunkIndex;
        requests[partialRank].push_back(std::move(request));
        plan.sources.emplace_back();
        plan.signals.push_back(-1);
        ++plan.partialSources;
      }
      plans.push_back(std::move(plan));
    }

    // As a partial rank: an edge per request, from its sources (in rank order from this rank's own, like
    // mergeReductions) to a region of the scratch tensor, and a notification for the requesting rank. A rank has
    // numNotificationSlots slots for the notifications from each other rank: with more requests, consecutive ones
    // share a slot (notified once all of them are done).
    // The kernel computes the partial results in the order of the edges (see generateKernel): part by part (chunkIndex,
    // see reductionPieces), so that every executor gets the first parts from all domains early and can combine them
    // while the next ones are on their way; and within a part, starting with the requests of the domain after this
    // rank's, at the position of this rank, so that at any time the partial ranks send to different executors, and
    // the executors receive from different partial ranks (as in the read order below; with all partial ranks
    // starting at the same executor, its NIC took all of their data at first while the others had nothing to read).
    for (size_t i : range(size)) {
      ctx.send(i, requests[i]);
    }
    Vector<Vector<PartialRequest>> incoming(size);
    Vector<std::pair<uint32_t, uint32_t>> order; // (requesting rank, request index)
    for (size_t i : range(size)) {
      ctx.receive(i, incoming[i]);
      for (size_t j : indices(incoming[i])) {
        order.emplace_back(i, j);
      }
    }
    std::ranges::stable_sort(order, {}, [&](const std::pair<uint32_t, uint32_t>& x) {
      const uint32_t e = x.first;
      const size_t n = domainSize[domainIndex[e]];
      return std::make_tuple(incoming[e][x.second].chunkIndex,
          (domainIndex[e] + numDomains - 1 - domainIndex[rank]) % numDomains,
          (position[e] + n - position[rank] % n) % n);
    });
    Vector<Vector<PartialResponse>> responses(size);
    for (size_t i : range(size)) {
      responses[i].resize(incoming[i].size());
    }
    uint32_t numFlags = 0;
    for (auto [i, j] : order) {
      const size_t numRequests = incoming[i].size();
      PartialRequest& request = incoming[i][j];
      Edge p;
      p.id = ictx.nextEdgeId();
      p.sources = std::move(request.sources);
      std::ranges::rotate(p.sources, std::ranges::lower_bound(p.sources, rank, {}, [](const Node& n) {
        return n.rank;
      }));
      size_t offset = allocate(request.bytes);
      p.destinations.push_back(scratchNode(offset, false));
      p.bytes = request.bytes;
      p.cellIndex = request.cellIndex;
      p.chunkIndex = request.chunkIndex;
      p.executorRank = rank;
      p.divisor = request.divisor;
      p.doneSignal = numFlags++;
      const uint32_t slot = numRequests <= CustomOpDescriptor::numNotificationSlots
                                ? j
                                : j * CustomOpDescriptor::numNotificationSlots / numRequests;
      op.notifications.push_back({(uint32_t)p.doneSignal, i, slot});
      // (the index of the scratch tensor in the list of tensors that WorkCustom publishes: after the inputs and input
      // copies)
      responses[i][j] = {(uint32_t)(graph.numInputs + graph.numInputCopies), offset, slot};
      graph.edges.push_back(std::move(p));
    }
    for (size_t i : range(size)) {
      ctx.send(i, responses[i]);
    }
    Vector<Vector<PartialResponse>> received(size);
    for (size_t i : range(size)) {
      ctx.receive(i, received[i]);
      CHECK(received[i].size() == requests[i].size());
    }

    // The executor's edges, with the partial results read from the partial ranks' scratch tensors once notified. The
    // CPU thread issues the reads in signal order: per part, from the domain before this rank's first, then the one
    // before that, ..., which is the order in which the partial ranks compute them (see above): at any time, every
    // partial rank has one reader. (The kernel combines them in the order of the domains, the same for every part.)
    for (Plan& plan : plans) {
      Edge& e = graph.edges[plan.edgeIndex];
      auto readOrder = plan.partials;
      std::ranges::stable_sort(readOrder, {}, [&](const auto& x) {
        return (domainIndex[rank] + numDomains - 1 - domainIndex[std::get<1>(x)]) % numDomains;
      });
      for (auto [sourcePosition, partialRank, requestIndex] : readOrder) {
        const PartialResponse& response = received[partialRank].at(requestIndex);
        auto r = read(partialRank, response.tensorIndex, response.offset, e.bytes, response.slot);
        plan.sources[sourcePosition] = scratchNode(r.outputOffset, true);
        plan.signals[sourcePosition] = r.signalIndex;
      }
      e.sources = std::move(plan.sources);
      e.sourceSignals = std::move(plan.signals);
      e.partialSources = plan.partialSources;
      e.divisor = plan.divisor;
    }

    if (scratchBytes != 0) {
      Graph::TensorDescr t;
      t.bytes = scratchBytes;
      t.dtype = op.dtype;
      t.shape.push_back(scratchBytes / wrapperApi.dtypeSize(t.dtype));
      t.device = DeviceType::CUDA;
      graph.tensors.push_back(t);
      op.scratchBytes = scratchBytes;
    }
    log.debug("compile_op: rank %zu: %zu reductions across NVLink domains, %u partial results for other ranks, %d "
              "signals, %zu bytes of scratch\n",
        rank, plans.size(), numFlags, nextSignal, scratchBytes);

    const size_t maxSignals = group->cpuOutBuffer.itembytes / sizeof(uint32_t) - 16;
    const size_t maxFlags = group->cpuInBuffer.itembytes / sizeof(uint32_t) - 16;
    if ((size_t)nextSignal > maxSignals || numFlags > maxFlags) {
      return fmt::sprintf("rank %zu would read %d regions from other NVLink domains and compute %u partial results for "
                          "them, more than its signals allow (%zu and %zu)",
          rank, nextSignal, numFlags, maxSignals, maxFlags);
    }
    return "";
  }

  Setting tuneKernels(Vector<Graph> graphs, std::string target) {
    auto start = std::chrono::steady_clock::now();
    // One pipeline of 32K chunks per block reaches an SM's NVLink bandwidth; the grid size is what is tuned.
    // Reductions: the blocks also combine the data (with more threads), and they read from other GPUs, which takes
    // more SMs than writing (all-gather with the read graph on 32 GB300: 304, 631 and 659 GB/s per GPU with 8, 16 and
    // 32 blocks; writes: 650-672).
    const bool reduction = isReduction(graphs.at(0).reduce);
    Vector<KernelConfig> configs;
    static constexpr std::array<uint32_t, 4> copyGrids = {4, 8, 12, 16};
    static constexpr std::array<uint32_t, 4> reductionGrids = {8, 16, 24, 32};
    for (uint32_t gridSize : reduction ? reductionGrids : copyGrids) {
      KernelConfig config;
      config.blockSize = reduction ? 256 : 128;
      config.sharedMemory = 1024 * 226;
      config.gridSize = gridSize;
      config.chunkSize = 32768;
      config.pipelines = 1;
      configs.push_back(config);
    }
    // MOODIST_COMPILE_OP_CONFIGS="blockSize:sharedMemory:gridSize[:chunkSize[:pipelines]],..." replaces the
    // candidates (for benchmarks and experiments; the same on every rank)
    if (Vector<std::string> items = agreedEnvList("MOODIST_COMPILE_OP_CONFIGS", "");
        items.size() != 1 || !items[0].empty()) {
      configs.clear();
      for (const std::string& item : items) {
        Vector<uint32_t> fields;
        bool ok = true;
        for (size_t p = 0; p <= item.size();) {
          size_t e = std::min(item.find(':', p), item.size());
          std::string field = item.substr(p, e - p);
          ok &= !field.empty() && field.size() <= 9 && std::ranges::all_of(field, [](char c) {
            return c >= '0' && c <= '9';
          });
          fields.push_back(ok ? std::stoul(field) : 0);
          p = e + 1;
        }
        KernelConfig config;
        if (ok && fields.size() >= 3 && fields.size() <= 5) {
          config.blockSize = fields[0];
          config.sharedMemory = fields[1];
          config.gridSize = fields[2];
          config.chunkSize = fields.size() > 3 ? fields[3] : config.chunkSize;
          config.pipelines = fields.size() > 4 ? fields[4] : config.pipelines;
        }
        ok &= config.blockSize % 32 == 0 && config.blockSize >= 32 && config.blockSize <= 1024 &&
              config.gridSize >= 1 && config.chunkSize >= 16 && config.chunkSize % 16 == 0 && config.pipelines >= 1 &&
              config.pipelines * 32 <= config.blockSize;
        REQUIRE(ok,
            "compile_op: MOODIST_COMPILE_OP_CONFIGS: bad config '%s' (expected blockSize:sharedMemory:gridSize[:"
            "chunkSize[:pipelines]], with blockSize a multiple of 32 up to 1024, chunkSize a multiple of 16 and at "
            "most blockSize / 32 pipelines)",
            item);
        configs.push_back(config);
      }
    }
    // Only the ranks that execute reductions generate kernels with the reduce loop, which takes only some
    // configurations: checked here, where every rank takes part
    if (reduction) {
      for (const KernelConfig& config : configs) {
        std::string error = reduceConfigError(config);
        REQUIRE(error.empty(), "compile_op: MOODIST_COMPILE_OP_CONFIGS: %s", error);
      }
    }

    Vector<Setting> settings;
    for (const Graph& graph : graphs) {
      Setting s;
      s.graph = graph;
      for (const KernelConfig& config : configs) {
        s.config = config;
        settings.push_back(s);
      }
    }

    // Timings are combined per setting index below, so every rank must have the same settings.
    for (size_t i : range(size)) {
      ctx.send(i, settings.size());
    }
    for (size_t i : range(size)) {
      size_t n;
      ctx.receive(i, n);
      CHECK(settings.size() == n);
    }

    Vector<std::shared_ptr<KernelHandle>> sources;
    Vector<CompiledKernel> kernels;

    for (const auto& s : settings) {
      // Tuning launches run without the CPU thread: nodes filled over RDMA count as filled. A rank that no kernel edge
      // touches has no kernel; it takes part in the barriers and the exchange of timings (with none of its own).
      sources.push_back(
          s.graph.cudaEdges.empty() ? nullptr : generateKernel(group, s.config, target, s.graph, ctx, true));
    }

    log.info("Generated %d kernels in %gs\n", sources.size(), seconds(std::chrono::steady_clock::now() - start));
    start = std::chrono::steady_clock::now();

    for (const auto& h : sources) {
      kernels.push_back(h ? compileKernelPtx(h->ptx, "compile_op_copy") : CompiledKernel());
    }

    log.info("Compiled %d kernels in %gs\n", kernels.size(), seconds(std::chrono::steady_clock::now() - start));
    start = std::chrono::steady_clock::now();

    uint32_t stepValue = 1;
    uint32_t concurrencyIndex = 0;

    static constexpr size_t nWarmup = 12;
    static constexpr size_t nRuns = 2;
    static constexpr size_t nBatch = 12;

    Vector<std::pair<Event, Event>> events;
    for (size_t i : range(nRuns)) {
      events.emplace_back(Event::createTiming(), Event::createTiming());
    }

    Stream stream = Stream::create();

    ctx.barrier();
    ctx.resetBarriers();
    ctx.barrier();

    Vector<std::chrono::nanoseconds> timings;

    for (size_t i : indices(settings)) {
      const Setting& setting = settings[i];
      const Graph& graph = setting.graph;
      const KernelConfig& config = setting.config;

      log.info("run %d\n", i);

      if (!sources.at(i)) {
        ctx.barrier();
        timings.push_back(std::chrono::nanoseconds::max());
        continue;
      }

      Vector<TensorPtr> tensors;
      Vector<uintptr_t> tensorAddrs;

      for (size_t ti = 0; ti < graph.tensors.size(); ti++) {
        auto& t = graph.tensors[ti];
        if (t.device != DeviceType::CUDA) {
          // (kernel edges have CUDA nodes only)
          tensorAddrs.push_back(0);
          continue;
        }
        auto x = TensorPtr::empty(t.shape, t.dtype, group->deviceIndex);
        log.debug("  tensor %d: shape=[%s] dtype=%d deviceIndex=%d data_ptr=%p\n", ti,
            fmt::to_string(fmt::join(t.shape, ",")), (int)t.dtype, group->deviceIndex,
            x.defined() ? x.data_ptr() : nullptr);
        CHECK(x.defined());
        x.zero_();
        tensors.push_back(x);
        tensorAddrs.push_back((uintptr_t)x.data_ptr());
      }

      IpcMapper* ipcMapper = &*group->ipcMapper;

      Vector<uintptr_t> mappedAddrs(graph.remoteCudaTensorMappings.size());
      for (size_t i : indices(mappedAddrs)) {
        auto& m = graph.remoteCudaTensorMappings[i];
        ipcMapper->requestAddressRank(
            m.rank, tensorAddrs.at(m.tensorIndex), graph.tensors.at(m.tensorIndex).bytes, &mappedAddrs[i]);
      }
      ipcMapper->wait();

      if (tensorAddrs.empty()) {
        tensorAddrs.push_back(0);
      }
      if (mappedAddrs.empty()) {
        mappedAddrs.push_back(0);
      }

      CompileOpCopyParameters params{.stepValue = stepValue, .concurrencyIndex = concurrencyIndex};
      std::array<void*, 3> kparams = {&params, tensorAddrs.data(), mappedAddrs.data()};
      CUlaunchConfig launchConfig;
      CUlaunchAttribute attr;
      launchConfig.grid = {config.gridSize, 1, 1};
      launchConfig.block = {config.blockSize, 1, 1};
      launchConfig.sharedMemBytes = 0;
      launchConfig.hStream = stream;
      launchConfig.attrs = &attr;
      launchConfig.numAttrs = 1;
      attr.id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
      attr.value.value = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE;

      auto run = [&]() {
        CHECK_CU(cuLaunchKernelEx(&launchConfig, kernels.at(i).function, kparams.data(), nullptr));
        stepValue += 0x1000;
        concurrencyIndex = (concurrencyIndex + 1) % maxConcurrency;
        params.stepValue = stepValue;
        params.concurrencyIndex = concurrencyIndex;
      };

      ctx.barrier();

      for (size_t i : range(nWarmup)) {
        run();
      }

      auto start = std::chrono::steady_clock::now();
      for (size_t i : range(nRuns)) {
        events[i].first.record(stream);
        for (size_t i : range(nBatch)) {
          run();
        }
        events[i].second.record(stream);
      }
      stream.synchronize();
      std::chrono::nanoseconds timing;
      for (size_t i : range(nRuns)) {
        float t = 0.0f;
        CHECK_CU(cuEventElapsedTime(&t, events[i].first, events[i].second));
        auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float, std::ratio<1, 1000>>(t));
        if (i == 0 || ns < timing) {
          timing = ns;
        }
      }
      timings.push_back(timing);
    }

    ctx.barrier();
    ctx.resetBarriers();
    ctx.barrier();

    log.info("Benchmarked %d kernels in %gs\n", timings.size(), seconds(std::chrono::steady_clock::now() - start));
    start = std::chrono::steady_clock::now();

    for (size_t i : indices(settings)) {
      log.info("setting %d: %fms\n", i, seconds(timings.at(i)) * 1000);
    }

    // The minimum over ranks: ranks without a kernel time nanoseconds::max() (a maximum would need to skip those)
    for (size_t i : range(size)) {
      ctx.send(i, timings);
    }
    Vector<std::chrono::nanoseconds> ntimings;
    for (size_t i : range(size)) {
      ctx.receive(i, ntimings);
      CHECK(ntimings.size() == timings.size());
      for (size_t i : indices(timings)) {
        timings[i] = std::min(timings[i], ntimings[i]);
      }
    }

    // The fastest setting, with a small penalty per block of the grid: the copy kernel occupies its SMs, which the
    // computation it overlaps with lacks meanwhile. A bigger grid has to be clearly faster to be chosen, as it is
    // for copies limited by the SMs' bandwidth; for latency bound ones the smaller grid wins.
    static constexpr double gridPenalty = 0.005;
    auto score = [&](size_t i) {
      return timings[i].count() * (1 + gridPenalty * settings[i].config.gridSize);
    };
    size_t index = 0;
    for (size_t i : indices(timings)) {
      if (score(i) < score(index)) {
        index = i;
      }
    }

    log.info("Best setting: index %d, %fms\n", index, seconds(timings[index]) * 1000);

    return settings.at(index);
  }

  std::shared_ptr<CustomOpDescriptor> compile(DType dtype, std::span<const api::TensorRegion> inputs,
      std::span<const api::TensorRegion> outputs, ReduceOp reduce, bool cpuSync) {
    const Group* group = ctx.group;

    size_t itemsize = wrapperApi.dtypeSize(dtype);

    // Validate per-tensorId ndim consistency
    // Different tensorIds can have different ndims (e.g., 2D weight vs 1D bias)
    HashMap<std::string, int> tensorIdNdim;

    auto validateAndRecordNdim = [&](const api::TensorRegion& region, const char* kind, size_t idx) {
      int ndim = static_cast<int>(region.offset.size());
      if (region.offset.size() != region.shape.size()) {
        throw std::runtime_error(
            fmt::sprintf("moodist.compile_op: %s %zu has mismatched offset/shape sizes (%zu vs %zu)", kind, idx,
                region.offset.size(), region.shape.size()));
      }
      std::string tid(region.tensorId);
      auto it = tensorIdNdim.find(tid);
      if (it != tensorIdNdim.end()) {
        if (it->second != ndim) {
          throw std::runtime_error(
              fmt::sprintf("moodist.compile_op: %s %zu has wrong dimensions for tensorId '%s' (got %d, expected %d)",
                  kind, idx, tid.c_str(), ndim, it->second));
        }
      } else {
        tensorIdNdim[tid] = ndim;
      }
    };

    for (size_t i : indices(inputs)) {
      validateAndRecordNdim(inputs[i], "input", i);
    }
    for (size_t i : indices(outputs)) {
      validateAndRecordNdim(outputs[i], "output", i);
    }

    // Validate device strings for local regions (where rank == this rank)
    for (const auto& region : inputs) {
      if (static_cast<size_t>(region.rank) == rank) {
        parseAndValidateDevice(region.device, group->deviceIndex);
      }
    }
    for (const auto& region : outputs) {
      if (static_cast<size_t>(region.rank) == rank) {
        parseAndValidateDevice(region.device, group->deviceIndex);
      }
    }

    // Parse inputs and outputs into TensorDescr structs
    Vector<TensorDescr> inputDescrs;
    Vector<TensorDescr> outputDescrs;
    Vector<size_t> inputIndexCounter(size);
    Vector<size_t> outputIndexCounter(size);

    for (const auto& region : inputs) {
      TensorDescr d;
      d.rank = region.rank;
      d.index = inputIndexCounter[region.rank]++;
      d.tensorId = std::string(region.tensorId);
      d.device = parseDevice(region.device);
      int ndim = tensorIdNdim.at(d.tensorId);
      d.offset = Coord(ndim);
      d.shape = Coord(ndim);
      std::ranges::copy(region.offset, d.offset.begin());
      std::ranges::copy(region.shape, d.shape.begin());
      d.numel = numel(d.shape);
      inputDescrs.push_back(std::move(d));
    }

    for (const auto& region : outputs) {
      TensorDescr d;
      d.rank = region.rank;
      d.index = outputIndexCounter[region.rank]++;
      d.tensorId = std::string(region.tensorId);
      d.device = parseDevice(region.device);
      int ndim = tensorIdNdim.at(d.tensorId);
      d.offset = Coord(ndim);
      d.shape = Coord(ndim);
      std::ranges::copy(region.offset, d.offset.begin());
      std::ranges::copy(region.shape, d.shape.begin());
      d.numel = numel(d.shape);
      outputDescrs.push_back(std::move(d));
    }

    // Build lookup tables
    Vector<Vector<size_t>> inputsPerRank(size);
    Vector<Vector<size_t>> outputsPerRank(size);

    for (size_t i : indices(inputDescrs)) {
      inputsPerRank[inputDescrs[i].rank].push_back(i);
    }
    for (size_t i : indices(outputDescrs)) {
      outputsPerRank[outputDescrs[i].rank].push_back(i);
    }

    // Collect my inputs (inputs owned by this rank)
    Vector<size_t> myInputIndices = inputsPerRank[rank];

    // ============================================================================
    // Phase 1: Find intersections and exchange logical mappings
    // ============================================================================

    // For each output (from any rank), find intersecting inputs from myInputs
    // Create LogicalInput entries and group by destination rank
    // Only inputs and outputs with matching tensorId can intersect
    Vector<Vector<LogicalInput>> logicalInputsToSend(size);

    for (size_t outIdx : indices(outputDescrs)) {
      const TensorDescr& out = outputDescrs[outIdx];
      if (out.numel == 0) {
        continue;
      }

      // Check each of my inputs for intersection with this output
      for (size_t myInpIdx : myInputIndices) {
        const TensorDescr& inp = inputDescrs[myInpIdx];

        // Only intersect if tensorIds match
        if (inp.tensorId != out.tensorId) {
          continue;
        }

        // Get ndim for this tensorId (guaranteed to exist after validation)
        int ndim = tensorIdNdim.at(inp.tensorId);

        // Check intersection
        bool intersects = true;
        for (int d = 0; d < ndim; ++d) {
          if (inp.offset[d] >= out.offset[d] + out.shape[d] || out.offset[d] >= inp.offset[d] + inp.shape[d]) {
            intersects = false;
            break;
          }
        }

        if (!intersects) {
          continue;
        }

        // Compute intersection region
        LogicalInput li;
        li.offset = Coord(ndim);
        li.shape = Coord(ndim);
        for (int d = 0; d < ndim; ++d) {
          int64_t lo = std::max(inp.offset[d], out.offset[d]);
          int64_t hi = std::min(inp.offset[d] + inp.shape[d], out.offset[d] + out.shape[d]);
          li.offset[d] = lo;
          li.shape[d] = hi - lo;
        }
        li.inputRank = rank;
        li.inputIndex = inp.index;
        li.outputRank = out.rank;
        li.outputIndex = out.index;

        logicalInputsToSend[out.rank].push_back(std::move(li));
      }
    }

    // Barrier before queue operations
    ctx.barrier();

    // Every rank must compile the same op: the steps below depend on the dtype and the reduce op (e.g. only
    // reductions merge their edges at the executors). compile.py checks this too, but the C++ API comes straight
    // here. Every rank receives the values of all ranks before it raises, so that the queues stay in step.
    // The same exchange tells every rank the NVLink domain of each rank (its first rank).
    Vector<uint32_t> domainIds(size);
    for (size_t i : range(size)) {
      ctx.send(i, (int)dtype, (int)reduce, (uint32_t)computeCudaRanks(group, !ctx.collectiveOp).at(0));
    }
    std::string mismatch;
    for (size_t i : range(size)) {
      int peerDtype;
      int peerReduce;
      ctx.receive(i, peerDtype, peerReduce, domainIds[i]);
      if (mismatch.empty() && (peerDtype != (int)dtype || peerReduce != (int)reduce)) {
        mismatch = fmt::sprintf("rank %zu: dtype %s, reduce='%s'; rank %zu: dtype %s, reduce='%s'", rank,
            dtypeName(dtype), reduceName(reduce), i, dtypeName((DType)peerDtype), reduceName((ReduceOp)peerReduce));
      }
    }
    REQUIRE(mismatch.empty(), "moodist.compile_op: ranks specified different dtypes or reduce ops (%s)", mismatch);

    // Reductions combine the values in the kernel (see generateKernel): floating point types (accumulated in at
    // least 32 bits) and 32- and 64-bit integers
    if (isReduction(reduce)) {
      const bool floating =
          dtype == DType::Float32 || dtype == DType::Float64 || dtype == DType::Float16 || dtype == DType::BFloat16;
      const bool integer = dtype == DType::Int32 || dtype == DType::Int64;
      REQUIRE(floating || (integer && reduce != ReduceOp::Avg),
          "moodist.compile_op: reduce='%s' does not support dtype %s (supported: float32, float64, float16, "
          "bfloat16%s)",
          reduceName(reduce), dtypeName(dtype), reduce == ReduceOp::Avg ? "" : ", int32, int64");
    }

    // Send logical inputs to each destination rank
    for (size_t destRank : range(size)) {
      ctx.send(destRank, logicalInputsToSend[destRank]);
    }

    // Receive logical inputs from all source ranks
    Vector<LogicalInput> receivedInputs;
    for (size_t i : range(size)) {
      Vector<LogicalInput> batch;
      ctx.receive(i, batch);

      for (LogicalInput& li : batch) {
        CHECK(li.outputRank == rank);
        receivedInputs.push_back(std::move(li));
      }
    }

    // Debug logging
    log.debug("compile_op phase1: rank %zu, sent to %zu ranks, received %zu logical inputs\n", rank, size,
        receivedInputs.size());

    // ============================================================================
    // Phase 2: Overlap resolution via cell decomposition
    // ============================================================================

    // Group received inputs by output index
    size_t numMyOutputs = outputsPerRank[rank].size();
    Vector<Vector<size_t>> inputsPerOutput(numMyOutputs);
    for (size_t i : indices(receivedInputs)) {
      inputsPerOutput[receivedInputs[i].outputIndex].push_back(i);
    }

    // Result: for each output, its cells and their sources. Each cell becomes a ReadRequest per source in Phase 3,
    // and an edge in Phase 5, or with several sources (a reduction) an edge per piece (see reductionPieces)
    struct CellSource {
      Coord offset;
      Coord shape;
      Vector<size_t> sources; // Indices into receivedInputs: one, or with a reduction, every covering input
      uint32_t cellIndex;
      // A reduced cell's pieces: (byte offset in the cell, bytes, executor rank)
      Vector<std::tuple<size_t, size_t, uint32_t>> pieces;
    };
    Vector<Vector<CellSource>> cellsPerOutput(numMyOutputs);

    // A reduced cell is computed once per element, wherever its outputs are. With several destination ranks (the
    // ranks whose outputs cover the cell, e.g. every rank in an all-reduce), it is split into pieces of whole 16-byte
    // units, one per destination rank, and destination rank (p + cellIndex) mod m computes piece p: it reads the piece
    // from every source, combines them, and stores the result to every destination (a reduce-scatter and an
    // all-gather in one pass). A cell with one destination rank (e.g. in a reduce-scatter) is computed by that rank.
    // The destination ranks of a cell all compute the same pieces, from the same regions. Pieces are byte ranges of
    // the cell in row-major order, which is how both its inputs and its outputs hold it (a cell that is not
    // contiguous in a tensor goes through a contiguous copy, see Phases 4 and 5).
    // In a cell whose sources and destinations are not all in one NVLink domain, a piece of 8 MiB or more is split
    // further, into parts of at least 4 MiB (up to 32, with the same executor), so that the partial results that the
    // executor reads from other domains arrive in parts, which it can combine while the next ones are on their way
    // (see crossDomainReductions).
    auto reductionPieces = [&](const std::string& tensorId, const Coord& cellOffset, const Coord& cellShape,
                               uint32_t cellIndex, const Vector<size_t>& sourceInputs) {
      Vector<uint32_t> destinationRanks;
      for (const TensorDescr& out : outputDescrs) {
        bool contains = out.tensorId == tensorId;
        for (size_t d : range(cellShape.size())) {
          contains &= cellOffset[d] >= out.offset[d] && cellOffset[d] + cellShape[d] <= out.offset[d] + out.shape[d];
        }
        if (contains && std::ranges::find(destinationRanks, out.rank) == destinationRanks.end()) {
          destinationRanks.push_back(out.rank);
        }
      }
      std::ranges::sort(destinationRanks);
      CHECK(!destinationRanks.empty());
      bool crossDomain = false;
      for (size_t i : sourceInputs) {
        crossDomain |= domainIds[receivedInputs[i].inputRank] != domainIds[destinationRanks[0]];
      }
      for (uint32_t r : destinationRanks) {
        crossDomain |= domainIds[r] != domainIds[destinationRanks[0]];
      }
      const size_t bytes = itemsize * numel(cellShape);
      const size_t units = bytes / 16;
      const size_t m = destinationRanks.size();
      // (a size that is not a multiple of 16 bytes is an error for kernel edges, raised below)
      const size_t numPieces = bytes % 16 == 0 ? std::clamp<size_t>(units, 1, m) : 1;
      static constexpr size_t minPartBytes = 4 << 20;
      static constexpr size_t maxParts = 32;
      Vector<std::tuple<size_t, size_t, uint32_t>> pieces;
      for (size_t p : range(numPieces)) {
        size_t begin = numPieces == 1 ? 0 : units * p / numPieces * 16;
        size_t end = numPieces == 1 ? bytes : units * (p + 1) / numPieces * 16;
        const uint32_t executor = destinationRanks[(p + cellIndex) % m];
        size_t partBytes = end - begin;
        if (crossDomain && end - begin >= 2 * minPartBytes && bytes % 16 == 0) {
          size_t numParts = std::min(maxParts, (end - begin) / minPartBytes);
          partBytes = ((end - begin + numParts - 1) / numParts + 4095) / 4096 * 4096;
        }
        for (size_t offset = begin; offset < end; offset += partBytes) {
          pieces.emplace_back(offset, std::min(partBytes, end - offset), executor);
        }
      }
      return pieces;
    };

    struct TensorCells {
      Vector<Vector<int64_t>> boundaries;
      Vector<size_t> numIntervals;
      size_t cellsOffset;
      size_t totalCells;
    };

    HashMap<std::string, TensorCells> tensorCells;

    Vector<std::string> tensorIds;
    for (auto& v : tensorIdNdim) {
      tensorIds.push_back(v.first);
    }
    std::sort(tensorIds.begin(), tensorIds.end());

    size_t numTotalCells = 0;

    for (const auto& tensorId : tensorIds) {
      const auto& ndim = tensorIdNdim.at(tensorId);
      auto& tc = tensorCells[tensorId];
      auto& boundaries = tc.boundaries;
      boundaries.resize(ndim);
      for (const auto& descrs : {&inputDescrs, &outputDescrs}) {
        for (const auto& v : *descrs) {
          if (v.tensorId == tensorId) {
            for (int d : range(ndim)) {
              boundaries[d].push_back(v.offset[d]);
              boundaries[d].push_back(v.offset[d] + v.shape[d]);
            }
          }
        }
      }
      for (int d : range(ndim)) {
        std::sort(boundaries[d].begin(), boundaries[d].end());
        boundaries[d].erase(std::unique(boundaries[d].begin(), boundaries[d].end()), boundaries[d].end());
      }

      // Compute total number of cells
      auto& numIntervals = tc.numIntervals;
      numIntervals.resize(ndim);
      tc.totalCells = 1;
      for (int d = 0; d < ndim; ++d) {
        numIntervals[d] = boundaries[d].size() - 1;
        tc.totalCells *= numIntervals[d];
      }

      tc.cellsOffset = numTotalCells;
      numTotalCells += tc.totalCells;
    }

    for (size_t outIdx : range(numMyOutputs)) {
      size_t globalOutIdx = outputsPerRank[rank][outIdx];
      const TensorDescr& out = outputDescrs[globalOutIdx];
      const auto& inputIdxs = inputsPerOutput[outIdx];

      if (out.numel == 0) {
        continue;
      }

      // Get ndim for this output's tensorId
      int ndim = tensorIdNdim.at(out.tensorId);

      auto& tc = tensorCells[out.tensorId];
      auto& boundaries = tc.boundaries;
      CHECK(!boundaries.empty());
      auto& numIntervals = tc.numIntervals;

      // Iterate through all cells
      std::string gapDetails;
      std::string overlapDetails;
      size_t gapCount = 0;
      size_t overlapCount = 0;
      constexpr size_t maxReportedRegions = 5;

      for (size_t cellIdx : range(tc.totalCells)) {
        // Convert cellIdx to multi-dimensional index and compute cell offset/shape
        Coord cellOffset(ndim);
        Coord cellShape(ndim);
        size_t remaining = cellIdx;
        for (int d = ndim - 1; d >= 0; --d) {
          size_t intervalIdx = remaining % numIntervals[d];
          remaining /= numIntervals[d];
          cellOffset[d] = boundaries[d][intervalIdx];
          cellShape[d] = boundaries[d][intervalIdx + 1] - boundaries[d][intervalIdx];
        }

        // Check if cell is inside output
        bool insideOutput = true;
        for (int d = 0; d < ndim; ++d) {
          if (cellOffset[d] < out.offset[d] || cellOffset[d] + cellShape[d] > out.offset[d] + out.shape[d]) {
            insideOutput = false;
            break;
          }
        }
        if (!insideOutput) {
          continue;
        }

        // Find all inputs that cover this cell
        Vector<size_t> coveringInputs;
        for (size_t i : inputIdxs) {
          const LogicalInput& inp = receivedInputs[i];
          bool covers = true;
          for (int d = 0; d < ndim; ++d) {
            if (cellOffset[d] < inp.offset[d] || cellOffset[d] + cellShape[d] > inp.offset[d] + inp.shape[d]) {
              covers = false;
              break;
            }
          }
          if (covers) {
            coveringInputs.push_back(i);
          }
        }

        if (coveringInputs.empty()) {
          // Gap detected
          gapCount++;
          if (gapCount <= maxReportedRegions) {
            gapDetails += fmt::sprintf("\n  output[%zu]: missing region at [%s] shape [%s]", outIdx,
                fmtCoord(cellOffset).c_str(), fmtCoord(cellShape).c_str());
          }
        } else if (coveringInputs.size() > 1) {
          // Overlap detected
          if (reduce == ReduceOp::None) {
            overlapCount++;
            if (overlapCount <= maxReportedRegions) {
              const LogicalInput& a = receivedInputs[coveringInputs[0]];
              const LogicalInput& b = receivedInputs[coveringInputs[1]];
              overlapDetails += fmt::sprintf("\n  output[%zu]: overlap at [%s] shape [%s] between rank %u and rank %u",
                  outIdx, fmtCoord(cellOffset).c_str(), fmtCoord(cellShape).c_str(), a.inputRank, b.inputRank);
            }
          } else if (reduce == ReduceOp::Any) {
            // reduce == Any: pick one randomly
            size_t pickIdx = random<size_t>(0, coveringInputs.size() - 1);
            CellSource cs;
            cs.offset = std::move(cellOffset);
            cs.shape = std::move(cellShape);
            cs.sources.push_back(coveringInputs[pickIdx]);
            cs.cellIndex = tc.cellsOffset + cellIdx;
            cellsPerOutput[outIdx].push_back(std::move(cs));
          } else {
            // A reduction: all of them, in rank order
            CHECK(isReduction(reduce));
            std::ranges::sort(coveringInputs, {}, [&](size_t i) {
              return std::make_pair(receivedInputs[i].inputRank, receivedInputs[i].inputIndex);
            });
            CellSource cs;
            cs.cellIndex = tc.cellsOffset + cellIdx;
            cs.pieces = reductionPieces(out.tensorId, cellOffset, cellShape, cs.cellIndex, coveringInputs);
            cs.offset = std::move(cellOffset);
            cs.shape = std::move(cellShape);
            cs.sources = std::move(coveringInputs);
            cellsPerOutput[outIdx].push_back(std::move(cs));
          }
        } else {
          // Exactly one covering input
          CellSource cs;
          cs.offset = std::move(cellOffset);
          cs.shape = std::move(cellShape);
          cs.sources.push_back(coveringInputs[0]);
          cs.cellIndex = tc.cellsOffset + cellIdx;
          cellsPerOutput[outIdx].push_back(std::move(cs));
        }
      }

      // Report errors
      if (gapCount > 0) {
        std::string msg = "moodist.compile_op: missing input coverage";
        if (gapCount > maxReportedRegions) {
          msg += fmt::sprintf(" (%zu regions, showing first %zu):", gapCount, maxReportedRegions);
        } else {
          msg += ":";
        }
        msg += gapDetails;
        throw std::runtime_error(msg);
      }

      if (overlapCount > 0) {
        std::string msg = "moodist.compile_op: overlapping inputs detected";
        if (overlapCount > maxReportedRegions) {
          msg += fmt::sprintf(" (%zu regions, showing first %zu):", overlapCount, maxReportedRegions);
        } else {
          msg += ":";
        }
        msg += overlapDetails;
        throw std::runtime_error(msg);
      }
    }

    log.debug("compile_op phase2: rank %zu, completed cell decomposition, %d total cells\n", rank, numTotalCells);

    // ============================================================================
    // Phase 3: Send read requests (receiver → sender)
    // ============================================================================

    // Convert cells to ReadRequests, grouped by source rank
    Vector<Vector<ReadRequest>> requestsToSend(size);

    for (size_t outIdx : range(numMyOutputs)) {
      for (const auto& cell : cellsPerOutput[outIdx]) {
        for (size_t sourceIdx : cell.sources) {
          const LogicalInput& src = receivedInputs[sourceIdx];

          ReadRequest req;
          req.offset = cell.offset;
          req.shape = cell.shape;
          req.cellIndex = cell.cellIndex;
          req.requesterRank = static_cast<uint32_t>(rank);
          req.inputIndex = src.inputIndex;
          req.outputIndex = static_cast<uint32_t>(outIdx);

          requestsToSend[src.inputRank].push_back(std::move(req));
        }
      }
    }

    // Barrier before queue operations
    ctx.barrier();

    // Send read requests to each source rank
    for (size_t destRank : range(size)) {
      ctx.send(destRank, requestsToSend[destRank]);
    }

    // Receive read requests from all ranks (as a sender)
    Vector<ReadRequest> receivedRequests;
    for (size_t i : range(size)) {
      Vector<ReadRequest> batch;
      ctx.receive(i, batch);
      for (auto& req : batch) {
        receivedRequests.push_back(std::move(req));
      }
    }

    log.debug("compile_op phase3: rank %zu, received %zu read requests\n", rank, receivedRequests.size());

    // ============================================================================
    // Phase 4: Process requests, generate copies, send responses (sender)
    // ============================================================================

    // Track copies needed for non-contiguous reads from my inputs
    struct InputCopy {
      uint32_t inputIndex; // Which of my inputs
      Coord offset;        // Offset within that input
      Coord shape;         // Shape of the region
    };
    Vector<InputCopy> inputCopies;

    // Deduplication map: (inputIndex, offset, shape) -> copyIndex
    // Using a vector of (key, value) pairs with linear search since copy count is typically small
    struct InputCopyKey {
      uint32_t inputIndex;
      Coord offset;
      Coord shape;

      bool operator==(const InputCopyKey& other) const {
        return inputIndex == other.inputIndex && offset == other.offset && shape == other.shape;
      }
    };
    Vector<std::pair<InputCopyKey, uint32_t>> inputCopyDedup;

    // Process each request and build responses
    Vector<Vector<ReadResponse>> responsesToSend(size);

    for (const auto& req : receivedRequests) {
      // Find my input
      CHECK(req.inputIndex < myInputIndices.size());
      size_t globalInputIdx = myInputIndices[req.inputIndex];
      const TensorDescr& inp = inputDescrs[globalInputIdx];

      // Compute offset within input (request offset is in global coords)
      Coord relOffset = req.offset - inp.offset;

      // Check if this region is contiguous within the input tensor
      bool isContiguous = contiguous(req.shape, inp.shape);

      ReadResponse resp;
      resp.requesterRank = req.requesterRank;
      resp.outputIndex = req.outputIndex;
      resp.cellIndex = req.cellIndex;
      resp.requestOffset = req.offset;
      resp.requestShape = req.shape;
      resp.senderRank = static_cast<uint32_t>(rank);
      resp.inputDevice = inp.device;

      if (isContiguous) {
        // Can read directly from input
        resp.tensorIndex = req.inputIndex;
        resp.inputOffset = linearOffset(relOffset, inp.shape);
        resp.tensorShape = inp.shape;
        resp.isCopy = false;
      } else {
        // Need to make a contiguous copy - check for deduplication first
        InputCopyKey key{req.inputIndex, relOffset, req.shape};

        // Search for existing copy with same key
        uint32_t copyIndex = 0;
        bool found = false;
        for (const auto& [existingKey, existingIdx] : inputCopyDedup) {
          if (existingKey == key) {
            copyIndex = existingIdx;
            found = true;
            break;
          }
        }

        if (!found) {
          // Create new copy
          copyIndex = static_cast<uint32_t>(myInputIndices.size() + inputCopies.size());

          InputCopy copy;
          copy.inputIndex = req.inputIndex;
          copy.offset = relOffset;
          copy.shape = req.shape;
          inputCopies.push_back(std::move(copy));

          inputCopyDedup.push_back({key, copyIndex});
        }

        resp.tensorIndex = copyIndex;
        resp.inputOffset = 0;         // Copy is contiguous, starts at 0
        resp.tensorShape = req.shape; // Copy has same shape as request
        resp.isCopy = true;
      }

      responsesToSend[req.requesterRank].push_back(std::move(resp));
    }

    // Barrier before sending responses
    ctx.barrier();

    // Send responses to each requester
    for (size_t destRank : range(size)) {
      ctx.send(destRank, responsesToSend[destRank]);
    }

    // Receive responses from all ranks
    Vector<ReadResponse> receivedResponses;
    for (size_t i : range(size)) {
      Vector<ReadResponse> batch;
      ctx.receive(i, batch);
      for (auto& resp : batch) {
        receivedResponses.push_back(std::move(resp));
      }
    }

    log.debug("compile_op phase4: rank %zu, %zu input copies, received %zu responses\n", rank, inputCopies.size(),
        receivedResponses.size());

    // ============================================================================
    // Phase 5: Finalize and build CustomOpDescriptor (receiver)
    // ============================================================================

    auto op = std::make_shared<CustomOpDescriptor>();
    op->id = (*ctx.nextOpId)++;
    op->dtype = dtype;
    op->cpuSync = cpuSync;

    auto gt = [&](auto& d) {
      Graph::TensorDescr t;
      t.bytes = itemsize * d.numel;
      t.shape = Vector<int64_t>(d.shape.begin(), d.shape.end());
      t.device = d.device;
      t.dtype = dtype;
      return t;
    };

    for (size_t i : inputsPerRank[rank]) {
      op->inputs.push_back(gt(inputDescrs.at(i)));
    }
    for (size_t i : outputsPerRank[rank]) {
      op->outputs.push_back(gt(outputDescrs.at(i)));
    }

    log.info("op has %d inputs, %d outputs\n", op->inputs.size(), op->outputs.size());

    // Add inputCopies from Phase 4
    for (const auto& copy : inputCopies) {
      CustomOpDescriptor::Copy c;
      c.index = copy.inputIndex;
      c.offset = {copy.offset.begin(), copy.offset.end()};
      c.shape = {copy.shape.begin(), copy.shape.end()};
      op->inputCopies.push_back(std::move(c));
    }

    // Track output copies needed for non-contiguous writes
    struct OutputCopy {
      uint32_t outputIndex;
      Coord offset; // Relative to output
      Coord shape;
    };
    Vector<OutputCopy> outputCopyList;

    Graph baseGraph;
    InternalContext ictx(ctx);

    Vector<Edge>& edges = baseGraph.edges;

    // Reduced cells (several sources): their responses, one per source, by output and cell
    auto cellKey = [](uint32_t outputIndex, uint32_t cellIndex) {
      return ((uint64_t)outputIndex << 32) | cellIndex;
    };
    HashMap<uint64_t, Vector<const ReadResponse*>> reducedResponses;
    for (size_t outIdx : range(numMyOutputs)) {
      for (const auto& cell : cellsPerOutput[outIdx]) {
        if (cell.sources.size() > 1) {
          reducedResponses[cellKey(outIdx, cell.cellIndex)];
        }
      }
    }

    // Process responses and build reads
    for (const auto& resp : receivedResponses) {
      // Find my output
      CHECK(resp.outputIndex < outputsPerRank[rank].size());
      if (auto it = reducedResponses.find(cellKey(resp.outputIndex, resp.cellIndex)); it != reducedResponses.end()) {
        it->second.push_back(&resp);
        continue;
      }
      size_t globalOutputIdx = outputsPerRank[rank][resp.outputIndex];
      const TensorDescr& out = outputDescrs[globalOutputIdx];

      // Compute output offset (request offset is in global coords)
      Coord relOutOffset = resp.requestOffset - out.offset;
      size_t outputOffset = linearOffset(relOutOffset, out.shape);

      // Check if this region is contiguous within the output tensor
      bool outputContiguous = contiguous(resp.requestShape, out.shape);

      // Compute bytes
      size_t bytes = itemsize * numel(resp.requestShape);

      Edge edge;
      Node source;
      Node destination;

      source.id = 0;
      source.rank = resp.senderRank;
      source.tensorIndex = resp.tensorIndex;
      source.offset = itemsize * resp.inputOffset;
      source.filled = true;
      source.device = resp.inputDevice;

      destination.id = 0;
      destination.rank = rank;
      destination.tensorIndex = myInputIndices.size() + inputCopies.size() + resp.outputIndex;
      destination.offset = itemsize * outputOffset;
      destination.filled = false;
      destination.device = out.device;

      edge.bytes = bytes;
      edge.cellIndex = resp.cellIndex;
      edge.id = ictx.nextEdgeId();
      edge.executorRank = rank;

      if (!outputContiguous) {
        destination.tensorIndex =
            myInputIndices.size() + inputCopies.size() + outputsPerRank[rank].size() + outputCopyList.size();
        destination.offset = 0;

        OutputCopy oc;
        oc.outputIndex = resp.outputIndex;
        oc.offset = relOutOffset;
        oc.shape = resp.requestShape;
        outputCopyList.push_back(std::move(oc));
      }

      edge.sources.push_back(source);
      edge.destinations.push_back(destination);

      edges.push_back(std::move(edge));
    }

    // Reduced cells: an edge per piece (see reductionPieces), with every source, executed by the piece's executor.
    // The sources are in (rank, tensor, offset) order, the same on every destination rank of the cell (see
    // mergeReductions).
    for (size_t outIdx : range(numMyOutputs)) {
      for (const auto& cell : cellsPerOutput[outIdx]) {
        if (cell.sources.size() <= 1) {
          continue;
        }
        const Vector<const ReadResponse*>& responses = reducedResponses.at(cellKey(outIdx, cell.cellIndex));
        CHECK(responses.size() == cell.sources.size());
        const TensorDescr& out = outputDescrs[outputsPerRank[rank][outIdx]];
        Coord relOutOffset = cell.offset - out.offset;
        Node destination;
        destination.id = 0;
        destination.rank = rank;
        destination.tensorIndex = myInputIndices.size() + inputCopies.size() + outIdx;
        destination.offset = itemsize * linearOffset(relOutOffset, out.shape);
        destination.filled = false;
        destination.device = out.device;
        if (!contiguous(cell.shape, out.shape)) {
          destination.tensorIndex =
              myInputIndices.size() + inputCopies.size() + outputsPerRank[rank].size() + outputCopyList.size();
          destination.offset = 0;
          OutputCopy oc;
          oc.outputIndex = outIdx;
          oc.offset = relOutOffset;
          oc.shape = cell.shape;
          outputCopyList.push_back(std::move(oc));
        }
        Vector<Node> sources;
        for (const ReadResponse* resp : responses) {
          Node source;
          source.id = 0;
          source.rank = resp->senderRank;
          source.tensorIndex = resp->tensorIndex;
          source.offset = itemsize * resp->inputOffset;
          source.filled = true;
          source.device = resp->inputDevice;
          sources.push_back(source);
        }
        std::ranges::sort(sources, {}, [](const Node& n) {
          return std::make_tuple(n.rank, n.tensorIndex, n.offset);
        });
        for (size_t p : indices(cell.pieces)) {
          auto [pieceOffset, pieceBytes, executor] = cell.pieces[p];
          Edge edge;
          for (Node source : sources) {
            source.offset += pieceOffset;
            edge.sources.push_back(source);
          }
          edge.destinations.push_back(destination);
          edge.destinations.back().offset += pieceOffset;
          edge.bytes = pieceBytes;
          edge.cellIndex = cell.cellIndex;
          edge.chunkIndex = p;
          edge.executorRank = executor;
          edge.id = ictx.nextEdgeId();
          edges.push_back(std::move(edge));
        }
      }
    }

    // Add outputCopies
    for (const auto& oc : outputCopyList) {
      CustomOpDescriptor::Copy c;
      c.index = oc.outputIndex;
      c.offset = {oc.offset.begin(), oc.offset.end()};
      c.shape = {oc.shape.begin(), oc.shape.end()};
      op->outputCopies.push_back(std::move(c));
    }

    {
      ctx.barrier();

      Vector<Node> localNodes;
      HashMap<uint32_t, Vector<Node*>> remoteNodes;
      auto equal = [](const Node& a, const Node& b) {
        if (a.rank == b.rank && a.tensorIndex == b.tensorIndex && a.offset == b.offset) {
          CHECK(a.device == b.device && a.filled == b.filled);
          return true;
        }
        return false;
      };
      for (auto& e : edges) {
        for (auto& l : {&e.sources, &e.destinations}) {
          for (auto& n : *l) {
            remoteNodes[n.rank].push_back(&n);
          }
        }
      }

      Vector<Node> l;
      Vector<uint64_t> ids;
      for (size_t i : range(size)) {
        auto it = remoteNodes.find(i);
        if (it != remoteNodes.end()) {
          for (auto* w : it->second) {
            l.push_back(*w);
          }
          CHECK(!l.empty());
        }
        ctx.send(i, l);
        l.clear();
      }
      for (size_t i : range(size)) {
        ctx.receive(i, l);
        if (l.empty()) {
          continue;
        }
        ids.clear();
        for (auto& n : l) {
          bool found = false;
          for (auto& w : localNodes) {
            if (equal(n, w)) {
              CHECK(!found);
              found = true;
              ids.push_back(w.id);
            }
          }
          if (!found) {
            n.id = (((uint64_t)rank + 1) << 32) + localNodes.size();
            localNodes.push_back(n);
            ids.push_back(n.id);
          }
        }
        ctx.send(i, ids);
      }

      for (auto& v : remoteNodes) {
        ctx.receive(v.first, ids);
        CHECK(ids.size() == v.second.size());
        for (size_t i : indices(ids)) {
          v.second[i]->id = ids[i];
        }
      }

      ctx.barrier();
    }

    baseGraph.reduce = reduce;
    baseGraph.numInputs = op->inputs.size();
    baseGraph.numInputCopies = op->inputCopies.size();
    baseGraph.numOutputs = op->outputs.size();
    baseGraph.numOutputCopies = op->outputCopies.size();
    for (auto t : op->inputs) {
      baseGraph.tensors.push_back(t);
    }
    for (auto& c : op->inputCopies) {
      const auto& ot = op->inputs.at(c.index);
      Graph::TensorDescr t;
      t.device = ot.device;
      t.dtype = ot.dtype;
      t.shape = c.shape;
      CHECK(ot.dtype == dtype);
      size_t n = 1;
      for (auto m : t.shape) {
        n *= m;
      }
      t.bytes = itemsize * n;
      baseGraph.tensors.push_back(t);
    }
    for (auto t : op->outputs) {
      baseGraph.tensors.push_back(t);
    }
    for (auto& c : op->outputCopies) {
      const auto& ot = op->outputs.at(c.index);
      Graph::TensorDescr t;
      t.device = ot.device;
      t.dtype = ot.dtype;
      t.shape = c.shape;
      size_t n = 1;
      for (auto m : t.shape) {
        n *= m;
      }
      t.bytes = itemsize * n;
      baseGraph.tensors.push_back(t);
    }

    Vector<Graph> graphs;

    // Reductions: merge the edges of each piece at its executor, then split those that reduce sources from other
    // NVLink domains
    std::string crossDomainError;
    if (isReduction(reduce)) {
      GraphBuilder merger(ictx, baseGraph);
      merger.mergeReductions();
      baseGraph.edges = std::move(merger.graph.edges);
      crossDomainError = crossDomainReductions(ictx, baseGraph, *op, domainIds);
    }

    // Proxy transfers from outside the NVLink domain (MOODIST_COMPILE_OP_PROXY=0 disables it), except in reductions
    // (the reduce loop takes no relay edges; their transfers across domains are crossDomainReductions'). Undone below
    // if the kernel path is not used.
    Vector<Edge> unproxiedEdges = baseGraph.edges;
    if (agreedEnv("MOODIST_COMPILE_OP_PROXY", "1") != "0" && !isReduction(reduce)) {
      proxyRemoteEdges(ictx, baseGraph.edges);
    }

    // Generate ibReads for edges that cannot be handled by the CUDA kernel
    // (remote ranks not reachable via NVLink, or CPU tensors).
    // Remove those edges from baseGraph so only kernel-eligible edges remain.
    bool kernelPath = false;
    {
      Vector<size_t> cudaRanks = computeCudaRanks(group, !ctx.collectiveOp);
      size_t numInputsAndCopies = op->inputs.size() + op->inputCopies.size();

      // Whether the copy kernel takes an edge: CUDA tensors on NVLink peers only. The kernel also needs sizes
      // and offsets that are multiples of 16 bytes (TMA bulk copies): user ops must provide them (checked
      // below), collective ops send other edges over RDMA.
      auto kernelEdge = [&](const Edge& edge) {
        bool cuda = true;
        bool local = true;
        bool aligned = edge.bytes % 16 == 0;
        for (auto& l : {&edge.sources, &edge.destinations}) {
          for (auto& n : *l) {
            if (n.device != DeviceType::CUDA) {
              cuda = false;
            }
            if (n.rank != rank && std::ranges::find(cudaRanks, n.rank) == cudaRanks.end()) {
              local = false;
            }
            if (n.offset % 16 != 0) {
              aligned = false;
            }
          }
        }
        return cuda && local && (aligned || !ctx.collectiveOp);
      };

      // Kernel tuning and launch below are collective (sends/barriers across all ranks), so every rank must
      // agree on whether the kernel path is used at all: it is if any rank has an edge the kernel can take. A
      // rank without one (e.g. one that only sends, or only receives over RDMA) still takes part in the tuning, and
      // runs a kernel if the kernel edges of other ranks read or write its tensors (tuneKernels). Every edge the
      // kernel cannot take becomes an ibRead. Reductions run in the kernel only: every edge with several sources
      // must be a kernel edge (crossDomainReductions has made those with sources in other domains kernel edges, where
      // it could).
      bool anyKernelEdge = std::ranges::any_of(baseGraph.edges, kernelEdge);
      std::string reductionError = crossDomainError;
      for (const Edge& e : baseGraph.edges) {
        if (reductionError.empty() && e.sources.size() > 1 && !kernelEdge(e)) {
          bool cpu = false;
          for (const auto* l : {&e.sources, &e.destinations}) {
            for (const Node& n : *l) {
              cpu |= n.device != DeviceType::CUDA;
            }
          }
          reductionError =
              cpu ? fmt::sprintf("rank %zu reduces CPU tensors (reductions need CUDA tensors)", rank)
                  : fmt::sprintf("rank %zu reduces a region whose outputs are in several NVLink domains, which is not "
                                 "supported yet across domains (e.g. all-reduce across domains)",
                        rank);
        }
      }
      for (size_t i : range(size)) {
        ctx.send(i, anyKernelEdge, reductionError);
      }
      std::string firstReductionError;
      for (size_t i : range(size)) {
        bool v;
        std::string error;
        ctx.receive(i, v, error);
        kernelPath |= v;
        if (firstReductionError.empty()) {
          firstReductionError = std::move(error);
        }
      }
      REQUIRE(
          firstReductionError.empty(), "moodist.compile_op: reduce='%s': %s", reduceName(reduce), firstReductionError);
      if (!kernelPath) {
        // Without the kernel, the copies from a proxy's node would be RDMA reads racing with the proxy's own read
        baseGraph.edges = std::move(unproxiedEdges);
      }

      for (auto it = baseGraph.edges.begin(); it != baseGraph.edges.end();) {
        auto& edge = *it;
        if (kernelPath && kernelEdge(edge)) {
          ++it;
          continue;
        }
        for (auto& dest : edge.destinations) {
          if (dest.rank != rank) {
            continue;
          }
          for (auto& src : edge.sources) {
            CustomOpDescriptor::IbRead read;
            read.rank = src.rank;
            read.inputIndex = src.tensorIndex;
            read.outputIndex = dest.tensorIndex - numInputsAndCopies;
            read.inputOffset = src.offset;
            read.outputOffset = dest.offset;
            read.bytes = edge.bytes;
            read.signalIndex = edge.rdmaSignal;
            op->ibReads.push_back(read);
            log.info("  ibRead: rank %u input[%u]+%zu -> output[%u]+%zu, %zu bytes, signal %d\n", read.rank,
                read.inputIndex, read.inputOffset, read.outputIndex, read.outputOffset, read.bytes, read.signalIndex);
          }
        }
        it = baseGraph.edges.erase(it);
      }

      // The order of the reads. The CPU thread issues the reads that signal the kernel first (in signal order, see
      // proxyRemoteEdges), then the others in this order, in small parts (see WorkCustom). Every rank takes the
      // domains after its own in turn, and in a domain of n ranks, the ranks from its own position mod n on:
      // (domain i, position p) reads (domain i + 1 + a, position p + b) in step (a, b), so that in each step every
      // source has one reader (with domains of equal size). In random order, some sources had several readers and
      // others none (all-to-all of 1 GiB per rank on 32 GB300 over RoCE: 12.2 ms instead of 10.2). A rotation by
      // rank ((source - rank) mod size) does not work: the ranks of a domain skip its own ranks, so they all start
      // on the same source. Sources in the rank's own domain (edges the kernel cannot take) come last.
      {
        Vector<uint32_t> firstRanks = domainIds;
        std::ranges::sort(firstRanks);
        firstRanks.erase(std::unique(firstRanks.begin(), firstRanks.end()), firstRanks.end());
        const size_t numDomains = firstRanks.size();
        Vector<size_t> domainIndex(size);
        Vector<size_t> position(size);
        Vector<size_t> domainSize(numDomains);
        for (size_t r : range(size)) {
          domainIndex[r] = std::ranges::lower_bound(firstRanks, domainIds[r]) - firstRanks.begin();
          position[r] = domainSize[domainIndex[r]]++;
        }
        std::ranges::stable_sort(op->ibReads, {}, [&](const CustomOpDescriptor::IbRead& read) {
          size_t j = domainIndex[read.rank];
          size_t n = domainSize[j];
          size_t a = (j + numDomains - 1 - domainIndex[rank]) % numDomains;
          size_t b = (position[read.rank] + n - position[rank] % n) % n;
          return std::make_pair(a, b);
        });
      }

      // How many sources a rank reads from at a time (readSources). In the order above, every source has one reader
      // per step, so reading one source at a time is best if all steps take equally long: every rank that reads
      // (without a signal) reads from every rank outside its domain (or from every other rank), the same amount
      // from each (within 1/16). With unequal amounts, readers pile up on the larger transfers while other sources
      // idle, and reading several sources at a time is faster. On 32 GB300 over RoCE (8 domains of 4 GPUs), 256 MiB
      // per rank on average, one at a time vs 8: all-to-all 2.59 vs 3.01 ms; with lognormal sizes per pair (sigma 1)
      // 6.58 vs 5.05 ms, one destination receiving 4 times as much 13.1 vs 10.7 ms. 4 at a time was up to 4% slower
      // than 8, all at once no faster and less steady. Each rank describes its own reads in the exchange below.
      Vector<size_t> bytesFrom(size);
      for (const auto& read : op->ibReads) {
        if (read.signalIndex < 0) {
          bytesFrom[read.rank] += read.bytes;
        }
      }
      bool hasReads = false;
      size_t minBytes = std::numeric_limits<size_t>::max();
      size_t maxBytes = 0;
      size_t numRemote = 0;
      size_t numRemoteRead = 0;
      size_t numLocal = 0;
      size_t numLocalRead = 0;
      for (size_t s : range(size)) {
        if (s == rank) {
          continue;
        }
        bool remote = domainIds[s] != domainIds[rank];
        ++(remote ? numRemote : numLocal);
        if (bytesFrom[s]) {
          ++(remote ? numRemoteRead : numLocalRead);
          hasReads = true;
          minBytes = std::min(minBytes, bytesFrom[s]);
          maxBytes = std::max(maxBytes, bytesFrom[s]);
        }
      }
      const bool dense = numRemoteRead == numRemote && (numLocalRead == 0 || numLocalRead == numLocal);

      bool localNeedsRdma = !op->ibReads.empty();
      for (size_t i : range(size)) {
        ctx.send(i, localNeedsRdma, hasReads, dense, minBytes, maxBytes);
      }
      bool anyRankNeedsRdma = false;
      bool uniform = true;
      size_t globalMinBytes = std::numeric_limits<size_t>::max();
      size_t globalMaxBytes = 0;
      for (size_t i : range(size)) {
        bool needsRdma;
        bool peerHasReads;
        bool peerDense;
        size_t peerMin;
        size_t peerMax;
        ctx.receive(i, needsRdma, peerHasReads, peerDense, peerMin, peerMax);
        anyRankNeedsRdma |= needsRdma;
        if (peerHasReads) {
          uniform &= peerDense;
          globalMinBytes = std::min(globalMinBytes, peerMin);
          globalMaxBytes = std::max(globalMaxBytes, peerMax);
        }
      }
      uniform &= globalMaxBytes == 0 || globalMaxBytes - globalMinBytes <= globalMaxBytes / 16;
      op->anyRankNeedsRdma = anyRankNeedsRdma;
      op->readSources = uniform ? 1 : 8;
      log.debug("compile_op: rank %zu: reads %s (%zu of %zu remote sources, %zu of %zu local, %zu-%zu bytes each); "
                "readSources %u\n",
          rank, uniform ? "uniform" : "not uniform", numRemoteRead, numRemote, numLocalRead, numLocal,
          hasReads ? minBytes : 0, maxBytes, op->readSources);
    }

    // The copy kernel moves data with TMA bulk copies (cp.async.bulk), which need 16-byte aligned addresses
    // and sizes: every kernel edge must be a multiple of 16 bytes at 16-byte aligned offsets (tensor
    // addresses are checked when the op runs). User ops must provide that (unaligned edges are not supported
    // yet); collective ops sent unaligned edges over RDMA above. Each rank checks its own edges, so agree
    // before raising: an error on some ranks alone would leave the others waiting in the collective steps
    // below.
    {
      std::string alignmentError;
      for (auto& edge : baseGraph.edges) {
        if (edge.bytes % 16 != 0) {
          alignmentError = fmt::sprintf("edge size %zu is not a multiple of 16 bytes", edge.bytes);
        }
        for (auto& l : {&edge.sources, &edge.destinations}) {
          for (auto& n : *l) {
            if (alignmentError.empty() && n.offset % 16 != 0) {
              alignmentError = fmt::sprintf(
                  "edge offset %zu (rank %u tensor %u) is not a multiple of 16 bytes", n.offset, n.rank, n.tensorIndex);
            }
          }
        }
        if (!alignmentError.empty()) {
          break;
        }
      }
      for (size_t i : range(size)) {
        ctx.send(i, alignmentError);
      }
      std::string firstError;
      size_t firstErrorRank = 0;
      for (size_t i : range(size)) {
        std::string e;
        ctx.receive(i, e);
        if (firstError.empty() && !e.empty()) {
          firstError = std::move(e);
          firstErrorRank = i;
        }
      }
      REQUIRE(firstError.empty(),
          "compile_op: %s (found on rank %zu). CUDA transfers between NVLink peers require sizes and offsets that "
          "are multiples of 16 bytes",
          firstError, firstErrorRank);
    }

    // Graph variants for tuneKernels to choose from, selected by MOODIST_COMPILE_OP_GRAPHS (comma separated,
    // default "multicast"; every rank must use the same list):
    //   multicast   the source rank reads each chunk once and writes it to all its destinations
    //   read        every destination rank reads its chunks itself
    //   ring_read   the destinations of a chunk form a chain, each reading it from the previous one
    //   ring_write  the same chain, each rank writing the chunk to the next one
    // Reductions use the multicast graph only, where every reduced element is computed once (see mergeReductions).
    Vector<std::string> graphNames = agreedEnvList("MOODIST_COMPILE_OP_GRAPHS", "multicast");
    if (isReduction(reduce)) {
      graphNames.clear();
      graphNames.push_back("multicast");
    }
    for (const std::string& name : graphNames) {
      if (name == "multicast") {
        graphs.push_back(buildMulticast(ictx, baseGraph));
      } else if (name == "read") {
        graphs.push_back(buildNothing(ictx, baseGraph));
      } else if (name == "ring_read") {
        graphs.push_back(buildRing(ictx, baseGraph, false));
      } else if (name == "ring_write") {
        graphs.push_back(buildRing(ictx, baseGraph, true));
      } else {
        REQUIRE(false,
            "compile_op: unknown graph '%s' in MOODIST_COMPILE_OP_GRAPHS (expected multicast, read, ring_read or "
            "ring_write)",
            name);
      }
    }

    // Final barrier
    ctx.barrier();

    if (kernelPath) {
      CUdevice cuDevice = group->cuDevice;
      int computeMajor = 0, computeMinor = 0;
      CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
      CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

      log.info("cuda compute capacity is %d.%d\n", computeMajor, computeMinor);

      const char* target = computeTarget(computeMajor, computeMinor);

      Setting setting = tuneKernels(graphs, target);

      op->graph = setting.graph;
      op->config = setting.config;

      for (size_t i : range(size)) {
        ctx.barrier();
        if (i == rank) {
          log.info("Rank %d graph:\n", rank);
          logGraph(setting.graph);
        }
        ctx.barrier();
      }

      // A rank that no kernel edge touches runs no kernel
      if (!setting.graph.cudaEdges.empty()) {
        std::shared_ptr<KernelHandle> kernelHandle =
            generateKernel(group, setting.config, target, setting.graph, ctx, false);
        const std::string& ptx = kernelHandle->ptx;
        if (std::getenv("MOODIST_DUMP_KERNELS")) {
          std::string fn = fmt::sprintf("moodist-compile-op-kernels-rank%zu-op%u.ptx", rank, op->id);
          FILE* f = fopen(fn.c_str(), "wb");
          if (f) {
            fwrite(ptx.data(), ptx.size(), 1, f);
            fclose(f);
            log.info("compile_op PTX dumped to %s\n", fn);
          }
        }
        op->kernel = std::make_unique<CompiledKernel>(compileKernelPtx(ptx, "compile_op_copy"));
        op->kernelHandle = std::move(kernelHandle);
      }
    } else {
      op->graph = graphs[0];
    }

    return op;
  }
};

} // namespace

std::shared_ptr<CustomOpDescriptor> compile(CompileContext& ctx, DType dtype, std::span<const api::TensorRegion> inputs,
    std::span<const api::TensorRegion> outputs, ReduceOp reduce, bool cpuSync) {

  return CompileOpConstructor(ctx).compile(dtype, inputs, outputs, reduce, cpuSync);
}

} // namespace compile_op
} // namespace moodist
