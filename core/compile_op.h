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

// A CUDA buffer of moodist's CUDA allocator (whose memory the ranks of a multi-node NVLink fabric can map, unlike other
// allocators'), or of the communication pool `pool` if given (see allocatorCommPoolAllocate), for use on stream. It is
// freed when the AllocatedBuffer is destroyed (cudaAllocatorImplFree): its memory is then reused only by work that
// comes after the work queued on stream by then.
AllocatedBuffer allocateCudaBuffer(size_t bytes, CUstream stream, int deviceIndex, CommPool* pool = nullptr);

namespace compile_op {

// Use the ReduceOp enum from the API
using ReduceOp = api::ReduceOp;

// The generations of the values of an op (see Graph::Signal) are less than this. Step values advance by 0x1000 per op
// (getNextStepValue in processgroup.cc; after a reset the next op gets 1, the one after it 0x1000), so the values of
// an op stay below those of the next op on its concurrency slot.
static constexpr uint32_t maxGenerations = 0xf00;

struct Graph {

  // A word that the kernel waits for, or writes, in reductions across NVLink domains or in two phases: an RDMA signal
  // (Group::rdmaSignalBuffer, which the CPU threads set, see writeRdmaSignal in cputhread.cc) or a done flag
  // (cpuInBuffer slot 16 + index, which the kernel sets once an edge is done), and its generation: the word is set to
  // the op's step value + generation, and a wait waits until it is at least that. A slot of a ring of scratch regions
  // is filled and freed once per use, with the use's generation (see crossDomainReductions in compile_op.cc); other
  // words have generation 0.
  struct Signal {
    int32_t index = -1; // < 0: none
    uint16_t generation = 0;
    bool done = false; // a done flag, else an RDMA signal

    template<typename X>
    void serialize(X& x) {
      x(index, generation, done);
    }
  };

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
    // One source (a copy), none (a relay edge, see below), or for a reduction several, combined in this order (the
    // executor's own source first, then the others in rank order from there; see mergeReductions in compile_op.cc,
    // and crossDomainReductions for reductions across NVLink domains)
    Vector<Node> sources;
    Vector<Node> destinations;
    size_t bytes;
    uint32_t cellIndex;
    uint32_t executorRank;
    // >= 0 for the two edges of a proxied transfer on the proxy rank (see proxyRemoteEdges in compile_op.cc):
    // the RDMA edge, whose read sets signal rdmaSignal once done (see writeRdmaSignal in cputhread.cc), and the relay
    // edge (no sources, the same destination), with which the kernel waits for that signal and then marks the
    // destination as filled.
    int32_t rdmaSignal = -1;
    // Which part of its cell the edge moves (0 if the cell is one edge): its part of at most maxEdgeBytes (Phase 5 in
    // compile_op.cc), its piece or part of a piece in a reduction (reductionPieces), or for a chunk of a proxied
    // transfer, part * 32 + chunk (proxyRemoteEdges)
    uint32_t chunkIndex = 0;

    // Reductions across NVLink domains (see crossDomainReductions in compile_op.cc) or in two phases (see
    // twoPhaseReductions):
    // per source, what the kernel waits for before it loads the source (none: index -1): the RDMA signal of the
    // transfer that fills it, or the done flag of the edge of this rank's that computes it (in an all-reduce across
    // domains, the partial result of the executor's own domain, and in two phases, the piece that the edge copies);
    // empty if no source waits
    Vector<Signal> sourceWaits;
    // What the kernel waits for before the edge stores to its destinations: that the regions of rings of the scratch
    // tensor that it stores to are free again (another rank has read their previous contents, or an edge of this
    // rank's has used them)
    Vector<Signal> destinationWaits;
    // The done flags that the kernel sets once all of the edge has been stored: for its CPU thread (a partial result
    // that ranks in other domains read over RDMA, the regions of a ring that the edge has used), or for the edges of
    // its own that wait for it (sourceWaits, destinationWaits)
    Vector<Signal> doneFlags;
    // The done flags only say that the edge has loaded its sources (it has used regions of rings, which are then free
    // again): the kernel sets them once all of the edge has been loaded, without waiting for its stores to complete
    bool doneWhenLoaded = false;
    // The number of sources, at the end, that are partial results of NVLink domains: an average (PremulSum) has
    // divided those by the divisor (multiplied them by the factor) already, the others it scales after combining them
    uint32_t partialSources = 0;
    // Avg: the number of inputs the result is the average of (0: the number of sources)
    uint32_t divisor = 0;

    // Whether the edge waits for anything (a signal of the CPU thread, or another edge) or sets done flags
    bool hasSignals() const {
      return !sourceWaits.empty() || !destinationWaits.empty() || !doneFlags.empty();
    }

    template<typename X>
    void serialize(X& x) {
      x(id, sources, destinations, bytes, cellIndex, executorRank, rdmaSignal, chunkIndex, sourceWaits,
          destinationWaits, doneFlags, doneWhenLoaded, partialSources, divisor);
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

  // Sum, Avg, Min, Max or PremulSum: edges with several sources reduce them (see compile_op.cc)
  ReduceOp reduce = ReduceOp::None;
};

// Whether reduce is an operation on the values (Sum, Avg, Min, Max, PremulSum), as opposed to None and Any
inline bool isReduction(ReduceOp reduce) {
  return reduce == ReduceOp::Sum || reduce == ReduceOp::Avg || reduce == ReduceOp::Min || reduce == ReduceOp::Max ||
         reduce == ReduceOp::PremulSum;
}

// Whether reduce scales the sum of an edge's inputs before it is rounded to the dtype: Avg divides it by the number of
// the inputs that cover each element, PremulSum multiplies it by the launch's factor (CompileOpCopyParameters::scale),
// as if each input had been multiplied by it first (the process group's PREMUL_SUM; in float32, so that no value is
// rounded to a 16-bit type before: multiplying the inputs first keeps float16 sums in range). Both apply to the inputs
// that an edge combines, before the partial results of other NVLink domains, which carry it already (see
// crossDomainReductions in compile_op.cc); floating point types only.
inline bool isScaledSum(ReduceOp reduce) {
  return reduce == ReduceOp::Avg || reduce == ReduceOp::PremulSum;
}

// NVLS reductions (multimem.ld_reduce, see generateKernel in ptx_codegen.cc), within the NVLink domain of the edge's
// executor (domain: the group's ranks in it, sorted; the multicast objects are per domain, see Group::Nvls). Whether
// a reduction edge has their shape: a source on every rank of the domain, at the same tensor and offset on each, and a
// destination on every rank of the domain (likewise; stored with multicast stores) or, in domains of at least
// nvlsReduceLocalMinRanks (see ptx_codegen.cc), one, on the edge's executor (nvlsReduceLoop takes such edges). Whether
// an edge of the reduce loop has NVLS runs (it loads its first domain.size() sources with multimem.ld_reduce, as one,
// if they are such sources, and stores to its first domain.size() destinations with one multicast store if they are
// such destinations; e.g. the edges of reductions across domains). And whether an op's reductions could use them on
// this rank: its domain (enough ranks, multicast support), the dtype and the reduce op (what multimem.ld_reduce
// combines), not the edges.
bool nvlsReductionShape(const Graph::Edge& e, std::span<const size_t> domain);
struct NvlsRuns {
  bool sources = false;
  bool destinations = false;
};
NvlsRuns nvlsReductionRuns(const Graph::Edge& e, std::span<const size_t> domain);
bool nvlsReductionsPossible(const Group* group, DType dtype, ReduceOp reduce, bool collectiveOp);
// Whether this rank's domain can also have NVLS runs of destinations alone (multicast stores of results computed from
// other sources, e.g. an all-reduce's across domains), see nvlsReductionRuns
bool nvlsMulticastRuns(const Group* group, DType dtype, ReduceOp reduce, bool collectiveOp);
// The most bytes that the switches may combine for an op within an NVLink domain (with multimem.ld_reduce: the bytes of
// its ranks' NVLS reductions and of their edges with NVLS runs of sources, together) for it to use NVLS there (more
// use the reduce loop alone)
uint64_t nvlsReductionMaxBytes(DType dtype);

// The name of a reduce op, as compile_op takes it
inline const char* reduceName(ReduceOp reduce) {
  switch (reduce) {
  case ReduceOp::None:
    return "none";
  case ReduceOp::Any:
    return "any";
  case ReduceOp::Sum:
    return "sum";
  case ReduceOp::Avg:
    return "avg";
  case ReduceOp::Min:
    return "min";
  case ReduceOp::Max:
    return "max";
  case ReduceOp::PremulSum:
    return "premul_sum";
  }
  return "unknown";
}

// Context passed from ProcessGroupImpl to compile_op
struct CompileContext {
  Group* group;
  std::span<SharedPtr<Queue>> queues;
  Function<void()> barrier;
  Function<void()> resetBarriers; // Reset step value / barrier arrays (before/after tuning)

  // For generating unique op IDs
  uint32_t* nextOpId;

  // An op behind a process group collective (the scatter and gather of CPU tensors; the collectives of CUDA tensors
  // are user ops, see runCollective in processgroup.cc), whose tensors may come from any allocator and have any size
  // (set per compile). Such ops use no multi-node NVLink fabric peers (e.g. NVL72; they
  // can only map memory from the moodist CUDA allocator), so edges to other nodes go over RDMA, and edges
  // the copy kernel cannot move (sizes or offsets not multiples of 16 bytes) also go over RDMA instead of
  // raising.
  bool collectiveOp = false;

  Vector<std::pair<uint32_t, TensorDataPtr>> queuedReceives;

  // How long a receive waits for the other rank (the process group's timeout). After a receive has timed out, the
  // exchanges are out of step (a late message would be taken for a later one): compile fails from then on.
  std::chrono::steady_clock::duration timeout = std::chrono::minutes(30);
  bool timedOut = false;

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
    const auto start = std::chrono::steady_clock::now();
    auto get = [&] {
      const float left = std::chrono::duration<float>(timeout - (std::chrono::steady_clock::now() - start)).count();
      TensorDataPtr t = left > 0 ? q->get(true, left) : nullptr;
      if (!t) {
        timedOut = true;
        throw std::runtime_error(
            fmt::sprintf("moodist.compile_op: rank %d timed out after %g s waiting for rank %d (the "
                         "process group's timeout; MOODIST_TIMEOUT overrides it)",
                group->rank, std::chrono::duration<float>(timeout).count(), fromrank));
      }
      return t;
    };
    while (true) {
      uint32_t sourceRank;
      deserialize(get(), sourceRank);
      if (sourceRank == fromrank) {
        deserialize(get(), v...);
        return;
      }
      queuedReceives.emplace_back(sourceRank, get());
    }
  }
};

// Why reduce cannot combine values of dtype, or empty if it can (reductions: floating point types and 32- and 64-bit
// integers; avg and PremulSum floating point types only)
std::string reduceDtypeError(DType dtype, ReduceOp reduce);

// Main compile function
// Returns a compiled CustomOpDescriptor ready for execution
// ndim is validated per-tensorId (different tensorIds can have different ndims)
std::shared_ptr<CustomOpDescriptor> compile(CompileContext& ctx, DType dtype, std::span<const api::TensorRegion> inputs,
    std::span<const api::TensorRegion> outputs, ReduceOp reduce = ReduceOp::None, bool cpuSync = false);

} // namespace compile_op
} // namespace moodist
