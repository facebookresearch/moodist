// Copyright (c) Meta Platforms, Inc. and affiliates.

// compile_op implementation
// See compile_op.h for design overview
//
// Seven-phase protocol:
// Phase 1: Exchange logical mappings (sender → receiver)
// Phase 2: Overlap resolution via cell decomposition (local)
// Phase 3: Send read requests (receiver → sender)
// Phase 4: Process requests, generate copies (sender)
// Phase 5: Finalize and generate reads (receiver)
// Phase 6: Compute allLocal flag and build reverse mapping (localInputProvides)
// Phase 7: Multicast setup (source-side analysis, create per-region multicast objects)

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

struct InternalContext {
  CompileContext& ctx;

  uint64_t edgeIdCounter = 0;
  uint64_t nextEdgeId() {
    return (1ull << 63) + ((uint64_t)(ctx.group->rank + 1) << 32) + edgeIdCounter++;
  }
};

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

    // fixme
    cudaRanks = group->ipcRanks;
    if (std::ranges::find(cudaRanks, rank) == cudaRanks.end()) {
      cudaRanks.push_back(rank);
    }
    std::ranges::sort(cudaRanks);

    if (!group->fabricDomainRanks.empty()) {
      cudaRanks = group->fabricDomainRanks;
    }
  }

  void multicast() {
    HashMap<uint32_t, Vector<Edge>> singleSourceNodes;
    for (auto i = edges.begin(); i != edges.end();) {
      auto& e = *i;
      if (e.sources.size() != 1) {
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

    // for (size_t i : indices(op.localCudaTensorMappings)) {
    //   auto& v = op.localCudaTensorMappings[i];
    //   log.info(" local slot %d: %d %d %d %d\n", i, v.rank, v.tensorIndex, v.destinationSlot, v.stride);
    // }
    // for (size_t i : indices(op.remoteCudaTensorMappings)) {
    //   auto& v = op.remoteCudaTensorMappings[i];
    //   log.info(" remote slot %d: %d %d %d %d\n", i, v.rank, v.tensorIndex, v.destinationSlot, v.stride);
    // }
  }

  void finalize() {
    HashMap<uint32_t, Vector<Edge>> edgePerRank;
    HashMap<uint64_t, size_t> localEdges;
    for (const Edge& e : edges) {
      CHECK(!localEdges.contains(e.id));
      localEdges[e.id] = &e - edges.data();
      for (auto* l : {&e.sources, &e.destinations}) {
        for (const Node& n : *l) {
          if (n.rank != rank) {
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
    Vector<std::tuple<uint32_t, uint32_t, Node>> myReceives;
    HashMap<uint32_t, Edge> edgeMap;

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
      CHECK(!edgeMap.contains(edge.cellIndex));
      edgeMap[edge.cellIndex] = edge;
      for (auto& d : edge.destinations) {
        CHECK(d.rank == rank);
        if (d.device == DeviceType::CUDA) {
          myReceives.emplace_back(edge.cellIndex, edge.bytes, d);
          break;
        }
      }
    }

    for (size_t i : cudaRanks) {
      ctx.send(i, myReceives);
    }

    HashMap<uint32_t, Vector<Node>> groups;

    for (size_t i : cudaRanks) {
      Vector<std::tuple<uint32_t, uint32_t, Node>> nReceives;
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
      CHECK(!edge.sources.empty());
      uint32_t sourceRank = edge.sources[0].rank;
      std::ranges::rotate(ranks, std::ranges::lower_bound(ranks, sourceRank, {}, proj));

      Vector<uint32_t> rankns;
      for (auto& v : ranks) {
        rankns.push_back(v.rank);
      }
      log.info("%d: ranks for cell %d is [%s]\n", rank, cell, fmt::to_string(fmt::join(rankns, ", ")));

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
  for (size_t i : indices(graph.rdmaEdges)) {
    const auto& e = graph.rdmaEdges[i];
    log.info("  rdmaEdge[%zu]: %#x %zu bytes executor=%d cell=%u [%s] -> [%s]\n", i, e.id, e.bytes, e.executorRank,
        e.cellIndex, f(e.sources), f(e.destinations));
  }
  for (size_t i : indices(graph.cudaEdges)) {
    const auto& e = graph.cudaEdges[i];
    log.info("  cudaEdge[%zu]: %#x %zu bytes executor=%d cell=%u [%s] -> [%s]\n", i, e.id, e.bytes, e.executorRank,
        e.cellIndex, f(e.sources), f(e.destinations));
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

  Setting tuneKernels(Vector<Graph> graphs, std::string target) {
    auto start = std::chrono::steady_clock::now();
    Vector<KernelConfig> configs;
    // for (uint32_t blockSize : {64, 128, 256, 448, 896}) {
    //   for (uint32_t sharedSize : {1024 * 16, 1024 * 32, 1024 * 64, 1024 * 224}) {
    //     KernelConfig config;
    //     config.copyEngine = "lockstep";
    //     config.blockSize = blockSize;
    //     config.sharedMemory = sharedSize;
    //     configs.push_back(config);
    //   }
    // }

    // for (uint32_t blockSize : {32, 64, 128, 256, 512, 1024}) {
    //   for (uint32_t sharedSize : {1024 * 16, 1024 * 32, 1024 * 64, 1024 * 226}) {
    //     KernelConfig config;
    //     config.copyEngine = "buffered";
    //     config.blockSize = blockSize;
    //     config.sharedMemory = sharedSize;
    //     configs.push_back(config);
    //   }
    // }

    for (uint32_t blockSize : {32, 64, 128, 256, 512, 1024}) {
      for (uint32_t sharedSize : {1024 * 16, 1024 * 32, 1024 * 64, 1024 * 226}) {
        // for (uint32_t blockSize : {32}) {
        //   for (uint32_t sharedSize : {1024 * 16}) {
        KernelConfig config;
        config.copyEngine = "simple";
        config.blockSize = blockSize;
        config.sharedMemory = sharedSize;
        configs.push_back(config);
      }
    }

    {
      auto tmp = configs;
      configs.clear();
      for (KernelConfig c : tmp) {
        c.gridSize = 8;
        configs.push_back(c);
        c.gridSize = 12;
        configs.push_back(c);
        c.gridSize = 16;
        configs.push_back(c);
        c.gridSize = 32;
        configs.push_back(c);
        c.gridSize = 64;
        configs.push_back(c);
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

    HashMap<std::string, Vector<int>> ranksPerTarget;

    for (size_t i : range(size)) {
      ctx.send(i, settings.size(), target);
    }
    for (size_t i : range(size)) {
      size_t n;
      std::string ntarget;
      ctx.receive(i, n, ntarget);
      CHECK(settings.size() == n);
      ranksPerTarget[ntarget].push_back(i);
    }
    for (auto& v : ranksPerTarget) {
      std::ranges::sort(v.second);
    }

    Vector<std::shared_ptr<KernelHandle>> sources;
    Vector<CompiledKernel> kernels;

    for (const auto& s : settings) {
      sources.push_back(generateKernel(group, s.config, target, s.graph, ctx));
    }

    log.info("Generated %d kernels in %gs\n", sources.size(), seconds(std::chrono::steady_clock::now() - start));
    start = std::chrono::steady_clock::now();

    for (const auto& h : sources) {
      kernels.push_back(compileKernelPtx(h->ptx, "compile_op_copy"));
    }

    // {
    //   Vector<std::thread> threads;
    //   kernels.resize(sources.size());
    //   for (size_t n : indices(sources)) {
    //     size_t i = (rank + n) % sources.size();
    //     threads.emplace_back([&, i] {
    //       CHECK_CU(cuCtxSetCurrent(group->cuContext));
    //       kernels[i] = compileKernelPtx(sources[i], "compile_op_copy");
    //     });
    //   }
    //   for (auto& v : threads) {
    //     v.join();
    //   }
    // }

    // {
    //   auto& ranks = ranksPerTarget[target];
    //   auto myIt = std::ranges::find(ranks, rank);
    //   CHECK(myIt != ranks.end());
    //   for (size_t i = myIt - ranks.begin(); i < sources.size(); i += ranks.size()) {
    //     std::string obj = compilePtxToCubin(sources[i]);
    //     for (size_t r : ranks) {
    //       ctx.send(r, i, obj);
    //     }
    //   }
    //   for (size_t i : indices(sources)) {
    //     uint32_t sourceRank = ranks[i % ranks.size()];
    //     size_t ni;
    //     std::string obj;
    //     ctx.receive(sourceRank, ni, obj);
    //     CHECK(ni == i);
    //     kernels.push_back(compileKernelCubin(obj, "compile_op_copy"));
    //   }
    // }

    log.info("Compiled %d kernels in %gs\n", kernels.size(), seconds(std::chrono::steady_clock::now() - start));
    start = std::chrono::steady_clock::now();

    uint32_t stepValue = 1;
    uint32_t concurrencyIndex = 0;

    static constexpr size_t nWarmup = 4;
    static constexpr size_t nRuns = 6;
    static constexpr size_t nBatch = 4;

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

      Vector<TensorPtr> tensors;
      Vector<uintptr_t> tensorAddrs;

      for (auto& t : graph.tensors) {
        auto x = TensorPtr::empty(t.shape, t.dtype, t.device == DeviceType::CUDA ? group->deviceIndex : -1);
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

    auto index = std::ranges::min_element(timings) - timings.begin();

    log.info("Best setting: index %d, %fms\n", index, seconds(timings[index]) * 1000);

    return settings.at(index);
  }

  // TuningResult tuneCopyKernel(CompileContext& ctx, SizeCategory sizeCat) {
  //   static constexpr int depths[] = {1, 2, 4, 8, 16, 32};
  //   static constexpr size_t blockSizes[] = {128, 256, 512, 768, 1024};
  //   size_t gridSize = 8;
  //   if (auto* env = std::getenv("MOODIST_COPY_GRID_SIZE")) {
  //     int gs = atoi(env);
  //     if (gs >= 1 && gs <= 128) {
  //       gridSize = gs;
  //     }
  //   }
  //   static constexpr int warmupIters = 2;
  //   static constexpr int measuredIters = 6;

  //   const Group* group = ctx.group;

  //   const size_t copyBytes = representativeSize(sizeCat);
  //   const size_t rank = group->rank;
  //   const size_t size = group->size;
  //   CUdevice cuDevice = group->cuDevice;
  //   IpcMapper* ipcMapper = &*group->ipcMapper;

  //   bool verbose = false;
  //   if (auto* env = std::getenv("MOODIST_TUNE_VERBOSE")) {
  //     verbose = !strcmp(env, "1");
  //   }

  //   if (verbose) {
  //     log.info("tuning copy kernel for size category %d (%zu bytes), rank %zu/%zu...\n", static_cast<int>(sizeCat),
  //         copyBytes, rank, size);
  //   }

  //   // Get compute target for PTX generation
  //   int computeMajor = 0, computeMinor = 0;
  //   CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  //   CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
  //   const char* target = computeTarget(computeMajor, computeMinor);

  //   // Allocate synthetic src and dst buffers
  //   CUdeviceptr syntheticSrc = 0, syntheticDst = 0;
  //   CHECK_CU(cuMemAlloc(&syntheticSrc, copyBytes));
  //   CHECK_CU(cuMemAlloc(&syntheticDst, copyBytes));
  //   CHECK_CU(cuMemsetD8(syntheticSrc, 0, copyBytes));
  //   CHECK_CU(cuMemsetD8(syntheticDst, 0, copyBytes));

  //   CustomOpDescriptor op;
  //   CustomOpDescriptor::Edge edge1;
  //   edge1.sources.emplace_back(nodeId(rank, 0), rank, 0, 0, true);
  //   edge1.destinations.emplace_back(nodeId((rank + 1) % size, 1), (rank + 1) % size, 1, 0, false);
  //   edge1.bytes = copyBytes;
  //   edge1.cellIndex = 0;
  //   edge1.id = (1ull << 63) + rank;
  //   CustomOpDescriptor::Edge edge2;
  //   edge2.sources.emplace_back(nodeId((rank + size - 1) % size, 0), (rank + size - 1) % size, 0, 0, true);
  //   edge2.destinations.emplace_back(nodeId(rank, 1), rank, 1, 0, false);
  //   edge2.bytes = copyBytes;
  //   edge2.cellIndex = 0;
  //   edge2.id = (1ull << 63) + (rank + size - 1) % size;

  //   op.cudaEdges.push_back(edge1);
  //   if (size != 1) {
  //     op.cudaEdges.push_back(edge2);
  //   }

  //   for (size_t i : indices(op.cudaEdges)) {
  //     const auto& e = op.cudaEdges[i];
  //     std::string srcStr, dstStr;
  //     for (const auto& s : e.sources) {
  //       if (!srcStr.empty()) {
  //         srcStr += ", ";
  //       }
  //       srcStr += fmt::sprintf("r%u:t%u+%zu", s.rank, s.tensorIndex, s.offset);
  //     }
  //     for (const auto& d : e.destinations) {
  //       if (!dstStr.empty()) {
  //         dstStr += ", ";
  //       }
  //       dstStr += fmt::sprintf("r%u:t%u+%zu", d.rank, d.tensorIndex, d.offset);
  //     }
  //     log.info(" tuning cudaEdge[%zu]: %zuB cell=%u [%s] -> [%s]\n", i, e.bytes, e.cellIndex, srcStr, dstStr);
  //   }

  //   op.inputs.push_back(copyBytes);
  //   op.outputs.push_back(copyBytes);

  //   constructCudaTensorMappings(op);

  //   Vector<uintptr_t> tensors;
  //   tensors.push_back(syntheticSrc);
  //   tensors.push_back(syntheticDst);

  //   Vector<uintptr_t> mappedAddrs(op.remoteCudaTensorMappings.size());
  //   for (size_t i : indices(mappedAddrs)) {
  //     auto& m = op.remoteCudaTensorMappings[i];
  //     size_t peerIndex = group->getPeerIndex(m.rank);
  //     ipcMapper->requestAddress(peerIndex, tensors.at(m.tensorIndex), copyBytes, &mappedAddrs[i]);
  //   }
  //   ipcMapper->wait();

  //   // Create stream and events for timing
  //   CUstream stream = nullptr;
  //   CHECK_CU(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, 0));
  //   Vector<CUevent> events; // pairs: [start0, stop0, start1, stop1, ...]

  //   auto createEvents = [&](int count) {
  //     while ((int)events.size() < count * 2) {
  //       CUevent e = nullptr;
  //       CHECK_CU(cuEventCreate(&e, CU_EVENT_DEFAULT));
  //       events.push_back(e);
  //     }
  //   };

  //   uint32_t stepValue = 1;

  //   TuningResult best;
  //   best.config.gridSize = gridSize;

  //   auto tuneStart = std::chrono::steady_clock::now();

  //   // Build candidate configs
  //   Vector<CopyKernelConfig> candidates;

  //   // Read env vars for fixed-value settings (applied to all relevant candidates)
  //   int envWarppipeDepth = 0;
  //   if (auto* env = std::getenv("MOODIST_WARPPIPE_DEPTH")) {
  //     int d = atoi(env);
  //     if (d >= 0 && d <= 32) {
  //       envWarppipeDepth = d;
  //     }
  //   }

  //   // Register pipeline variants: loadOp × depth × blockSize
  //   static constexpr const char* loadOps[] = {"cv", "cs", "nc"};
  //   for (const char* loadOp : loadOps) {
  //     for (int depth : depths) {
  //       for (size_t bs : blockSizes) {
  //         candidates.push_back(CopyKernelConfig::reg(depth, bs, gridSize, loadOp));
  //       }
  //     }
  //   }

  //   // Bulk copy engine variants: blockSize × chunkSize × dmaMode × bulkMode
  //   {
  //     int maxSmem = 0;
  //     CHECK_CU(cuDeviceGetAttribute(&maxSmem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice));
  //     // static constexpr size_t chunkSizes[] = {8192, 16384, 32768, 65536, 98304, 114688};
  //     static constexpr size_t chunkSizes[] = {114688};
  //     static constexpr size_t bulkBlockSizes[] = {32, 64, 128, 256, 512, 1024};
  //     for (size_t chunk : chunkSizes) {
  //       for (size_t bs : bulkBlockSizes) {
  //         size_t numWarps = bs / 32;

  //         // Double-buffered: total smem = 2 * chunkSize + 2 * numWarps * 8
  //         {
  //           size_t totalSmem = 2 * chunk + 2 * numWarps * 8;
  //           if (totalSmem <= (size_t)maxSmem) {
  //             // Warp-leader mode: each warp leader DMAs chunk/numWarps bytes; must be multiple of 16
  //             if (chunk % (numWarps * 16) == 0) {
  //               candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, true, false));
  //               candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, true, true));
  //             }
  //             // All-threads mode: each thread DMAs chunk/bs bytes; must be multiple of 16
  //             if (chunk % (bs * 16) == 0) {
  //               candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, false, false));
  //               candidates.push_back(CopyKernelConfig::bulk(bs, gridSize, chunk, false, true));
  //             }
  //           }
  //         }

  //         // Warppipe: total smem = chunkSize + numWarps * 8
  //         // Warppipe uses single-lane DMA (warp leader always), so no dmaMode sweep.
  //         // stageChunk = chunk/numWarps must divide cleanly for 16-byte aligned DMAs.
  //         {
  //           size_t totalSmem = chunk + numWarps * 8;
  //           if (totalSmem <= (size_t)maxSmem && (chunk * 2) % (numWarps * 1024) == 0) {
  //             static constexpr int pipeDepths[] = {1, 2, 4, 8, 16};
  //             // static constexpr int pipeDepths[] = {1};
  //             for (int depth : pipeDepths) {
  //               if (envWarppipeDepth == 0 || envWarppipeDepth == depth) {
  //                 candidates.push_back(CopyKernelConfig::warppipe(bs, gridSize, chunk * 2, depth));
  //               }
  //             }
  //           }
  //         }

  //         // Nbuf: single buffer + readBufs mbarriers per warp
  //         {
  //           if ((chunk * 2) % (numWarps * 1024) == 0) {
  //             for (int rb = 2; rb <= 4; rb++) {
  //               for (int wb = 0; wb <= 4; wb++) {
  //                 size_t totalSmem = chunk + numWarps * rb * 8;
  //                 if (totalSmem <= (size_t)maxSmem) {
  //                   candidates.push_back(CopyKernelConfig::nbuf(bs, gridSize, chunk * 2, rb, wb));

  //                   if (bs >= 64) {
  //                     CopyKernelConfig c;
  //                     c.copyEngine = "bulk";
  //                     c.blockSize = bs;
  //                     c.gridSize = gridSize;
  //                     c.bulkChunkSize = chunk * 2;
  //                     c.bulkMode = "nbuf2";
  //                     c.nbufReadCount = rb;
  //                     c.nbufWriteCount = wb;
  //                     c.bulkSkipWriteBack = false;
  //                     candidates.push_back(c);
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }

  //     // Flexbuf: split block into read group + write group, each bulk or reg
  //     {
  //       static constexpr int flexNumBuffers[] = {2, 4, 8, 16, 32};
  //       static constexpr int regThreadCounts[] = {32, 64, 128, 256, 512};
  //       for (size_t chunk : chunkSizes) {
  //         for (int numBuffers : flexNumBuffers) {
  //           size_t totalSmem = chunk + numBuffers * 8;
  //           if (totalSmem > (size_t)maxSmem) {
  //             continue;
  //           }
  //           // bulk-read + bulk-write
  //           // for (int readParallel : {1, 2, 3, 4, 5, 6, 7, 8}) {
  //           //   for (int writeParallel : {1, 2, 3, 4, 5, 6, 7, 8}) {
  //           //     candidates.push_back(CopyKernelConfig::flexbuf(
  //           //         gridSize, chunk * 2, "bulk", 32, readParallel, "bulk", 32, writeParallel, numBuffers));
  //           //   }
  //           // }
  //           for (int readParallel : {1, 2, 4, 8, 16}) {
  //             for (int writeParallel : {1, 2, 4, 8, 16}) {
  //               candidates.push_back(CopyKernelConfig::flexbuf(
  //                   gridSize, chunk * 2, "bulk", 32, readParallel, "bulk", 32, writeParallel, numBuffers));
  //               if (candidates.back().blockSize > 1024) {
  //                 candidates.pop_back();
  //               }
  //               // for (int p : range(6)) {
  //               //   candidates.push_back(CopyKernelConfig::flexbuf(
  //               //       gridSize, chunk * 2, "reg", 32 * (1 << p), readParallel, "bulk", 32, writeParallel,
  //               //       numBuffers));
  //               //   if (candidates.back().blockSize > 1024) {
  //               //     candidates.pop_back();
  //               //   }
  //               // }
  //             }
  //           }
  //           // // bulk-read + reg-write
  //           // for (int writeThreads : regThreadCounts) {
  //           //   if (32 + writeThreads <= 1024) {
  //           //     candidates.push_back(
  //           //         CopyKernelConfig::flexbuf(gridSize, chunk, "bulk", 32, 1, "reg", writeThreads, 1,
  //           numBuffers));
  //           //   }
  //           // }
  //           // // reg-read + bulk-write
  //           // for (int readThreads : regThreadCounts) {
  //           //   if (readThreads + 32 <= 1024) {
  //           //     candidates.push_back(
  //           //         CopyKernelConfig::flexbuf(gridSize, chunk, "reg", readThreads, 1, "bulk", 32, 1, numBuffers));
  //           //   }
  //           // }
  //         }
  //       }
  //     }
  //   }

  //   // Lockstep candidates
  //   {
  //     static constexpr size_t lockstepBlockSizes[] = {64, 128, 192, 256, 512, 768};
  //     bool lockstepMulticast = std::getenv("MOODIST_LOCKSTEP_MULTICAST") != nullptr;
  //     for (size_t bs : lockstepBlockSizes) {
  //       // candidates.push_back(CopyKernelConfig::lockstep(gridSize, 229376, bs, 1, 1, lockstepMulticast));
  //       // static constexpr size_t chunkSizes[] = {8192, 16384, 32768, 65536, 98304, 114688, 229376};
  //       static constexpr size_t chunkSizes[] = {1024 * 16};
  //       for (size_t cs : chunkSizes) {
  //         candidates.push_back(CopyKernelConfig::lockstep(gridSize, cs, bs, 1, 1, lockstepMulticast));
  //       }
  //     }
  //   }
  //   // {
  //   //   static constexpr size_t lockstepBlockSizes[] = {64, 128, 192, 256, 512, 768};
  //   //   for (size_t bs : lockstepBlockSizes) {
  //   //     candidates.push_back(CopyKernelConfig::lockstep(gridSize, 229376, bs, 1, 1, false));
  //   //     candidates.back().copyEngine = "lockstepg";
  //   //   }
  //   // }

  //   // // Expand candidates into read + write variants, and optionally ring variants
  //   // {
  //   //   Vector<CopyKernelConfig> expanded;
  //   //   expanded.reserve(candidates.size() * 4);
  //   //   for (const auto& c : candidates) {
  //   //     CopyKernelConfig read = c;
  //   //     read.copyWrite = false;
  //   //     expanded.push_back(read);
  //   //     CopyKernelConfig write = c;
  //   //     write.copyWrite = true;
  //   //     expanded.push_back(write);
  //   //     // CopyKernelConfig readRing = c;
  //   //     // readRing.copyWrite = false;
  //   //     // readRing.copyRing = true;
  //   //     // expanded.push_back(readRing);
  //   //     // CopyKernelConfig writeRing = c;
  //   //     // writeRing.copyWrite = true;
  //   //     // writeRing.copyRing = true;
  //   //     // expanded.push_back(writeRing);
  //   //   }
  //   //   candidates = std::move(expanded);
  //   // }

  //   // Filter candidates by env vars
  //   if (std::getenv("MOODIST_BULK_SKIP_WRITEBACK")) {
  //     for (auto& c : candidates) {
  //       c.bulkSkipWriteBack = true;
  //     } // if (totalSmem > 48 * 1024) {
  //       //   CHECK_CU(cuFuncSetAttribute(
  //       //       op->tunedKernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)totalSmem));
  //       // }
  //   }
  //   if (auto* env = std::getenv("MOODIST_COPY_ENGINE")) {
  //     candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
  //                          [env](const CopyKernelConfig& c) {
  //                            return strcmp(c.copyEngine, env) != 0;
  //                          }),
  //         candidates.end());
  //   }
  //   if (auto* env = std::getenv("MOODIST_COPY_BLOCK_SIZE")) {
  //     size_t bs = atoi(env);
  //     candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
  //                          [bs](const CopyKernelConfig& c) {
  //                            return c.blockSize != bs;
  //                          }),
  //         candidates.end());
  //   }
  //   if (auto* env = std::getenv("MOODIST_BULK_MODE")) {
  //     candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
  //                          [env](const CopyKernelConfig& c) {
  //                            return c.bulkMode.has_value() && strcmp(c.bulkMode.value(), env) != 0;
  //                          }),
  //         candidates.end());
  //   }
  //   if (std::getenv("MOODIST_BULK_WRITEBACK")) {
  //     candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
  //                          [](const CopyKernelConfig& c) {
  //                            return c.bulkWriteBack.has_value() && !c.bulkWriteBack.value();
  //                          }),
  //         candidates.end());
  //   }
  //   // if (auto* env = std::getenv("MOODIST_COPY_WRITE")) {
  //   //   bool writeOnly = !strcmp(env, "1");
  //   //   candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
  //   //                        [writeOnly](const CopyKernelConfig& c) {
  //   //                          return c.copyWrite.value_or(false) != writeOnly;
  //   //                        }),
  //   //       candidates.end());
  //   // }
  //   if (auto* env = std::getenv("MOODIST_COPY_RING")) {
  //     bool ringOnly = !strcmp(env, "1");
  //     candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
  //                          [ringOnly](const CopyKernelConfig& c) {
  //                            return c.copyRing.value_or(false) != ringOnly;
  //                          }),
  //         candidates.end());
  //   }

  //   REQUIRE(!candidates.empty(), "No tuning candidates remain after env var filtering");

  //   if (verbose) {
  //     log.info("tuning: %zu candidates after env var filtering\n", candidates.size());
  //   }

  //   // Evaluate all candidates
  //   Vector<float> candidateTimings(candidates.size());
  //   for (auto& t : candidateTimings) {
  //     t = INFINITY;
  //   }
  //   for (size_t ci : indices(candidates)) {
  //     const auto& cfg = candidates[ci];

  //     auto t0 = std::chrono::steady_clock::now();
  //     std::string ptx = generateCopyKernelPtx(group, cfg, target, op, ctx);
  //     double genSec = seconds(std::chrono::steady_clock::now() - t0);

  //     if (auto* env = std::getenv("MOODIST_DUMP_TUNE_PTX"); env && !strcmp(env, "1")) {
  //       fprintf(stderr, "=== PTX for %s ===\n%s\n=== END PTX ===\n", formatConfig(cfg).c_str(), ptx.c_str());
  //     }

  //     t0 = std::chrono::steady_clock::now();
  //     CUmodule variantModule = nullptr;
  //     char jitErrorLog[4096] = {};
  //     CUjit_option jitOpts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  //     void* jitOptVals[] = {jitErrorLog, (void*)(uintptr_t)sizeof(jitErrorLog)};
  //     CUresult jitErr = cuModuleLoadDataEx(&variantModule, ptx.c_str(), 2, jitOpts, jitOptVals);
  //     double jitSec = seconds(std::chrono::steady_clock::now() - t0);
  //     if (jitErr != CUDA_SUCCESS) {
  //       if (verbose) {
  //         log.info("  %s -> JIT failed (error %d), skipping\n", formatConfig(cfg), (int)jitErr);
  //         if (jitErrorLog[0]) {
  //           log.info("  ptxas: %s\n", jitErrorLog);
  //         }
  //       }
  //       continue;
  //     }

  //     CUfunction fn = nullptr;
  //     CHECK_CU(cuModuleGetFunction(&fn, variantModule, "compile_op_copy"));

  //     auto run = [&]() {
  //       stepValue += 0x1000;
  //       CompileOpCopyParameters params;
  //       params.stepValue = stepValue;
  //       params.concurrencyIndex = 0;
  //       std::array<void*, 3> kparams = {
  //           &params, tensors.data(), mappedAddrs.data() ? mappedAddrs.data() : (void*)&params};
  //       CHECK_CU(cuLaunchKernel(fn, cfg.gridSize, 1, 1, cfg.blockSize, 1, 1, 0, stream, kparams.data(), nullptr));
  //     };

  //     // Warmup
  //     for (int i = 0; i < warmupIters; i++) {
  //       run();
  //     }
  //     CHECK_CU(cuStreamSynchronize(stream));

  //     // Measured runs: record all events without synchronizing
  //     t0 = std::chrono::steady_clock::now();
  //     createEvents(measuredIters);
  //     for (int iter : range(measuredIters)) {
  //       CHECK_CU(cuEventRecord(events[iter * 2], stream));
  //       for (int n : range(4)) {
  //         run();
  //       }
  //       CHECK_CU(cuEventRecord(events[iter * 2 + 1], stream));
  //     }
  //     CHECK_CU(cuStreamSynchronize(stream));
  //     double measureSec = seconds(std::chrono::steady_clock::now() - t0);

  //     // Query elapsed times
  //     float bestMs = INFINITY;
  //     for (int iter = 0; iter < measuredIters; iter++) {
  //       float ms = 0.0f;
  //       CHECK_CU(cuEventElapsedTime(&ms, events[iter * 2], events[iter * 2 + 1]));
  //       if (ms < bestMs) {
  //         bestMs = ms;
  //       }
  //     }

  //     if (verbose) {
  //       log.info("  %s -> %.3f ms  (gen=%.3fs jit=%.3fs measure=%.3fs)\n", formatConfig(cfg), bestMs, genSec, jitSec,
  //           measureSec);
  //     }

  //     candidateTimings[ci] = bestMs;

  //     cuModuleUnload(variantModule);
  //   }

  //   // Cleanup
  //   for (auto& e : events) {
  //     cuEventDestroy(e);
  //   }
  //   cuStreamDestroy(stream);
  //   // cuMemFree(syntheticDst);
  //   // cuMemFree(syntheticSrc);

  //   // Collective decision: exchange timings across all ranks, sum them,
  //   // and pick the candidate with the lowest total time.
  //   {
  //     ctx.barrier();

  //     // Send our timings to all ranks
  //     for (size_t destRank : range(size)) {
  //       ctx.send(destRank, candidateTimings);
  //     }

  //     // Receive timings from all ranks and sum
  //     Vector<float> summedTimings(candidates.size());
  //     for (auto& t : summedTimings) {
  //       t = 0.0f;
  //     }
  //     for (size_t i : range(size)) {
  //       Vector<float> peerTimings;
  //       ctx.receive(i, peerTimings);
  //       CHECK(peerTimings.size() == candidates.size());
  //       for (size_t ci : indices(candidates)) {
  //         summedTimings[ci] += peerTimings[ci];
  //       }
  //     }

  //     // Pick the best from summed timings
  //     for (size_t ci : indices(candidates)) {
  //       if (summedTimings[ci] < best.ms) {
  //         best.ms = summedTimings[ci];
  //         best.config = candidates[ci];
  //       }
  //     }
  //     // Convert summed timing to per-rank average for reporting
  //     best.ms /= size;

  //     ctx.barrier();
  //   }

  //   if (best.config.copyEngine == nullptr) {
  //     throw std::runtime_error("Tuning failed to find any working copy kernel");
  //   }

  //   double elapsed = seconds(std::chrono::steady_clock::now() - tuneStart);
  //   log.info("tuning complete: best %s (%.3f ms) in %.1fs\n", formatConfig(best.config), best.ms, elapsed);

  //   return best;
  // }
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

    // Result: for each output, a list of (cell, source input index) pairs
    // Each cell will become a ReadRequest in Phase 3
    struct CellSource {
      Coord offset;
      Coord shape;
      size_t sourceIdx; // Index into receivedInputs
      uint32_t cellIndex;
    };
    Vector<Vector<CellSource>> cellsPerOutput(numMyOutputs);

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
          } else {
            // reduce == Any: pick one randomly
            size_t pickIdx = random<size_t>(0, coveringInputs.size() - 1);
            CellSource cs;
            cs.offset = std::move(cellOffset);
            cs.shape = std::move(cellShape);
            cs.sourceIdx = coveringInputs[pickIdx];
            cs.cellIndex = tc.cellsOffset + cellIdx;
            cellsPerOutput[outIdx].push_back(std::move(cs));
          }
        } else {
          // Exactly one covering input
          CellSource cs;
          cs.offset = std::move(cellOffset);
          cs.shape = std::move(cellShape);
          cs.sourceIdx = coveringInputs[0];
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
        const LogicalInput& src = receivedInputs[cell.sourceIdx];

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

          // log.debug("compile_op: input copy: inputIndex=%u, offset=[%s], shape=[%s], container=[%s]\n",
          // req.inputIndex,
          //     fmtCoord(relOffset).c_str(), fmtCoord(req.shape).c_str(), fmtCoord(inp.shape).c_str());
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

    // Process responses and build reads
    for (const auto& resp : receivedResponses) {
      // Find my output
      CHECK(resp.outputIndex < outputsPerRank[rank].size());
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

    // graphs.push_back(buildNothing(ictx, baseGraph));
    graphs.push_back(buildMulticast(ictx, baseGraph));
    // graphs.push_back(buildRing(ictx, baseGraph, true));
    // graphs.push_back(buildRing(ictx, baseGraph, false));

    // Final barrier
    ctx.barrier();

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

    if (!setting.graph.cudaEdges.empty()) {
      std::shared_ptr<KernelHandle> kernelHandle = generateKernel(group, setting.config, target, setting.graph, ctx);
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
