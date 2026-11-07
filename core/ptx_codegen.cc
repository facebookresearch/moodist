// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx_codegen.h"
#include "common.h"
#include "compile_op.h"
#include "compile_op_kernel.h"
#include "cputhread.h"
#include "group.h"
#include "ipc_mapper.h"
#include "ptx.h"

namespace moodist {

using namespace ptx;

const char* computeTarget(int computeMajor, int computeMinor) {
  if (computeMajor >= 10) {
    if (computeMinor >= 3) {
      return "sm_103a";
    } else {
      return "sm_100a";
    }
  } else if (computeMajor == 9) {
    return "sm_90a";
  } else if (computeMajor == 8 && computeMinor >= 9) {
    return "sm_89";
  } else if (computeMajor == 8) {
    return "sm_80";
  } else if (computeMajor == 7) {
    return "sm_70";
  }
  return "sm_60";
}

// compile_op_params (see paramBuffer in generateKernel): the words one launch copies (its second parameter, 31 KiB),
// and its block size
static constexpr uint32_t paramFillWords = 3968;
static constexpr uint32_t paramFillBlockSize = 256;

// The reduce loop's shared memory (see generateKernel): 2 to reduceMaxStages stages of chunkSize bytes with an mbarrier
// each, reduceResults result buffers, a ring of up to reduceMaxStages + 2 claimed items (8 bytes each), 16 bytes for
// the flag of the NVLS check (kernels with NVLS reductions or NVLS runs), and 256 bytes of alignment padding
static constexpr uint32_t reduceResults = 2;
static constexpr uint32_t reduceMaxStages = 16;
static size_t reduceStageBytes(const KernelConfig& config) {
  return config.chunkSize + 8;
}
static size_t reduceFixedBytes(const KernelConfig& config) {
  return 256 + reduceResults * config.chunkSize + 8 * (reduceMaxStages + 2) + 16;
}

std::string reduceConfigError(const KernelConfig& config) {
  if (config.pipelines != 1 || config.chunkSize % (16 * config.blockSize) != 0) {
    return fmt::sprintf("kernels with reductions need one pipeline per block and chunks of a multiple of 16 bytes per "
                        "thread (block size %d, chunk size %d, %d pipelines)",
        config.blockSize, config.chunkSize, config.pipelines);
  }
  if (config.sharedMemory < reduceFixedBytes(config) + 2 * reduceStageBytes(config)) {
    return fmt::sprintf("%d bytes of shared memory are not enough for the reduce loop with chunks of %d bytes",
        config.sharedMemory, config.chunkSize);
  }
  return "";
}

namespace {
// NVLS (see generateKernel): MOODIST_COMPILE_OP_NVLS=0 turns it off (tests)
bool nvlsEnabled() {
  const char* env = std::getenv("MOODIST_COMPILE_OP_NVLS");
  return !env || std::string_view(env) != "0";
}
// What NVLS needs of the group and the op, besides its edges and tensors: an op not behind a process group collective,
// this rank in an NVLink domain of at least minRanks ranks of the group (its multicast objects are the domain's, see
// Group::Nvls) that supports multicast (and has not failed to set up its multicast objects), moodist's allocator pool
bool nvlsDomainCapable(const Group* g, bool collectiveOp, size_t minRanks) {
  return nvlsEnabled() && !collectiveOp && g->fabricDomainRanks.size() >= minRanks && g->supportsMulticast &&
         g->nvls.state != -1 && allocatorGetReservedBase() != 0;
}
// Whether nodes are one per rank of an NVLink domain (its ranks sorted), at the same tensor and offset on each, of CUDA
// tensors
bool nvlsDomainNodes(std::span<const compile_op::Graph::Node> nodes, std::span<const size_t> domain) {
  if (nodes.size() != domain.size() || nodes.empty()) {
    return false;
  }
  Vector<uint8_t> seen(domain.size());
  for (const compile_op::Graph::Node& n : nodes) {
    auto it = std::ranges::lower_bound(domain, (size_t)n.rank);
    if (it == domain.end() || *it != n.rank || seen[it - domain.begin()] || n.tensorIndex != nodes[0].tensorIndex ||
        n.offset != nodes[0].offset || n.device != DeviceType::CUDA) {
      return false;
    }
    seen[it - domain.begin()] = 1;
  }
  return true;
}
// NVLS reductions (see generateKernel) from this many ranks of a domain, those with a destination on every rank of it
// (multicast stores, an all-reduce's) from nvlsReduceMinRanks, those with one destination (a reduce-scatter's, a
// reduce to one rank's) from nvlsReduceLocalMinRanks: with one destination, each GPU sends its whole input into the
// switches, N / (N - 1) times what the other ranks read with the reduce loop (bfloat16 on GB300, NVLS / reduce loop:
// all-reduce of 256 MiB on 8 GPUs 568 / 714 us, on 12 606 / 801 us, on 32 551 / 842 us; reduce-scatter of 64 MiB per
// rank on 8 GPUs 753 / 723 us, on 12 1113 / 1131 us, on 32 2917 / 3270 us; within one node (4 GPUs), an all-reduce
// kernel of the probe's was as fast as the reduce loop, 601 / 599 us at 256 MiB)
static constexpr size_t nvlsReduceMinRanks = 8;
static constexpr size_t nvlsReduceLocalMinRanks = 12;
// NVLS runs of destinations alone (see nvlsReductionRuns: the multicast stores of an edge whose sources are not a run,
// e.g. an all-reduce's results across NVLink domains) from this many ranks, as copies with multicast stores (see
// nvlsMinRanks in generateKernel): across 2 domains of 8 GB300, an all-reduce of 256 MiB took 874 us with them, 772
// us without (the switches also send each chunk back to its sender, 1/8 of a GPU's ingress)
static constexpr size_t nvlsStoreMinRanks = 16;
// (smaller domains get no multicast objects, see Group::setupNvls)
static_assert(nvlsReduceMinRanks >= Group::Nvls::minDomainRanks && nvlsReduceLocalMinRanks >= nvlsReduceMinRanks &&
              nvlsStoreMinRanks >= nvlsReduceMinRanks);
// The multimem.ld_reduce of a 16-byte vector of a dtype for a reduce op: perVector instructions of op (the forms of the
// integers and of float64 have no vectors: 4 or 8 bytes each, into 64-bit registers if wide), or none. No min / max of
// floating point types: multimem.ld_reduce has none for float32 and float64, and those of float16 and bfloat16 return
// the other value where one is NaN (GB300), where the reduce loop propagates NaN. No average (nor PremulSum) of
// float16: the switches round the sum to float16, which overflows where the average does not (from 65504 / N), where
// the reduce loop scales the float32 sum. (Sums of float types in the switches start from +0: a sum of -0 only is +0,
// where the reduce loop's is -0. Their sums of 16-bit floats are within 1 ulp of the exact sum, but not always the
// nearest value: on 8 GB300, 24-26% of the elements of sums of random values were 1 ulp off, with no bias measurable,
// where the reduce loop rounds to nearest. Integer sums wrap, as the reduce loop's.)
struct NvlsReduceInst {
  const char* op;
  uint32_t perVector;
  bool wide;
};
std::optional<NvlsReduceInst> nvlsReduceInst(DType dtype, compile_op::ReduceOp reduce) {
  using compile_op::ReduceOp;
  const bool sum = reduce == ReduceOp::Sum || compile_op::isScaledSum(reduce);
  const bool isMin = reduce == ReduceOp::Min;
  if (!compile_op::isReduction(reduce)) {
    return {};
  }
  switch (dtype) {
  case DType::Float32:
    return sum ? std::optional<NvlsReduceInst>({"add.v4.f32", 1, false}) : std::nullopt;
  case DType::BFloat16:
    return sum ? std::optional<NvlsReduceInst>({"add.acc::f32.v4.bf16x2", 1, false}) : std::nullopt;
  case DType::Float16:
    return reduce == ReduceOp::Sum ? std::optional<NvlsReduceInst>({"add.acc::f32.v4.f16x2", 1, false}) : std::nullopt;
  case DType::Float64:
    return sum ? std::optional<NvlsReduceInst>({"add.f64", 2, true}) : std::nullopt;
  case DType::Int32:
    return NvlsReduceInst{sum ? "add.s32" : isMin ? "min.s32" : "max.s32", 4, false};
  case DType::Int64:
    // (the sum of two's complement integers: the add of u64, there is none of s64)
    return NvlsReduceInst{sum ? "add.u64" : isMin ? "min.s64" : "max.s64", 2, true};
  default:
    return {};
  }
}

// The arithmetic of the reduce loop (see generateKernel), which the NVLS warp loop shares for the edges of NVLS runs:
// the accumulators of a dtype for a reduce op (16-bit floats in f32, float64 and the 64-bit integers in 64-bit
// registers), the identity they start from, combining a value into one, the accumulator values of a loaded 16-byte
// vector, and the vector to store from them
struct ReduceArith {
  DType dtype;
  compile_op::ReduceOp op;
  bool acc64;
  bool accFloat;
  bool half;
  uint32_t accPerVector;
  bool sum;
  bool isMin;
  // The identity: -0 for sums (x + -0 = x for every x, -0 included), +-inf or the integer extremes for min / max
  int64_t identity;

  ReduceArith(DType dtype, compile_op::ReduceOp op) : dtype(dtype), op(op) {
    using compile_op::ReduceOp;
    acc64 = dtype == DType::Float64 || dtype == DType::Int64;
    accFloat = dtype != DType::Int32 && dtype != DType::Int64;
    half = dtype == DType::Float16 || dtype == DType::BFloat16;
    accPerVector = half ? 8 : acc64 ? 2 : 4;
    sum = op == ReduceOp::Sum || compile_op::isScaledSum(op);
    isMin = op == ReduceOp::Min;
    CHECK(compile_op::isReduction(op) && (accFloat || !compile_op::isScaledSum(op)));
    if (sum) {
      identity = !accFloat ? 0 : acc64 ? (int64_t)0x8000000000000000ull : 0x80000000ll;
    } else if (isMin) {
      identity = !accFloat ? (acc64 ? INT64_MAX : 0x7fffffffll) : acc64 ? 0x7ff0000000000000ll : 0x7f800000ll;
    } else {
      identity = !accFloat ? (acc64 ? INT64_MIN : 0x80000000ll) : acc64 ? (int64_t)0xfff0000000000000ull : 0xff800000ll;
    }
  }

  // acc = op(acc, x); floating point min / max propagate NaN (min.NaN; there is no min.NaN.f64)
  void combine(const Value& acc, const Value& x) const {
    if (sum) {
      arith(accFloat ? (acc64 ? "add.rn.f64" : "add.rn.f32") : (acc64 ? "add.s64" : "add.s32"), acc, acc, x);
    } else if (!accFloat) {
      arith(isMin ? (acc64 ? "min.s64" : "min.s32") : (acc64 ? "max.s64" : "max.s32"), acc, acc, x);
    } else if (!acc64) {
      arith(isMin ? "min.NaN.f32" : "max.NaN.f32", acc, acc, x);
    } else {
      Value nan(ValType::Pred);
      setp_nan("f64", nan, acc, x);
      Value m(ValType::U64);
      arith(isMin ? "min.f64" : "max.f64", m, acc, x);
      Value nanSum(ValType::U64);
      arith("add.rn.f64", nanSum, acc, x);
      emit("selp.b64 " + acc.str() + ", " + nanSum.str() + ", " + m.str() + ", " + nan.str());
    }
  }

  // An average divides by a divisor, PremulSum multiplies by a factor (of the accumulators' type)
  void scale(const Value& acc, const Value& by) const {
    if (op == compile_op::ReduceOp::Avg) {
      arith(acc64 ? "div.rn.f64" : "div.rn.f32", acc, acc, by);
    } else {
      arith(acc64 ? "mul.rn.f64" : "mul.rn.f32", acc, acc, by);
    }
  }

  // The accumulator values of a loaded vector
  Vector<Value> unpack(const std::array<u32, 4>& x) const {
    Vector<Value> r;
    for (size_t j : range(4)) {
      if (dtype == DType::BFloat16) {
        r.push_back(x[j] << 16);
        r.push_back(x[j] & 0xffff0000ll);
      } else if (dtype == DType::Float16) {
        Value lo(ValType::U16);
        Value hi(ValType::U16);
        unpack_b32(lo, hi, x[j]);
        r.emplace_back(ValType::U32);
        arith("cvt.f32.f16", r.back(), lo);
        r.emplace_back(ValType::U32);
        arith("cvt.f32.f16", r.back(), hi);
      } else if (acc64) {
        if (j % 2 == 0) {
          r.emplace_back(ValType::U64);
          pack_b64(r.back(), x[j], x[j + 1]);
        }
      } else {
        r.push_back(x[j]);
      }
    }
    CHECK(r.size() == accPerVector);
    return r;
  }

  // The vector to store from accumulator values (16-bit floats rounded to nearest even)
  std::array<u32, 4> pack(const Vector<Value>& acc) const {
    std::array<u32, 4> x;
    for (size_t j : range(4)) {
      if (half) {
        // (cvt.rn.bf16x2.f32 d, a, b puts a in the upper half)
        arith(dtype == DType::BFloat16 ? "cvt.rn.bf16x2.f32" : "cvt.rn.f16x2.f32", x[j], acc[2 * j + 1], acc[2 * j]);
      } else if (acc64) {
        if (j % 2 == 0) {
          unpack_b64(x[j], x[j + 1], acc[j / 2]);
        }
      } else {
        x[j] = acc[j];
      }
    }
    return x;
  }
};
} // namespace

namespace {
// The first n nodes
std::span<const compile_op::Graph::Node> firstNodes(const Vector<compile_op::Graph::Node>& nodes, size_t n) {
  return std::span<const compile_op::Graph::Node>(nodes.data(), std::min(n, nodes.size()));
}
} // namespace

bool compile_op::nvlsReductionShape(const Graph::Edge& e, std::span<const size_t> domain) {
  if (domain.size() < 2 || e.hasSignals() || e.partialSources != 0 ||
      !nvlsDomainNodes(firstNodes(e.sources, e.sources.size()), domain) ||
      !std::ranges::all_of(e.sources, &Graph::Node::filled)) {
    return false;
  }
  if (e.destinations.size() == 1) {
    return domain.size() >= nvlsReduceLocalMinRanks && e.destinations[0].rank == e.executorRank &&
           e.destinations[0].device == DeviceType::CUDA;
  }
  return nvlsDomainNodes(firstNodes(e.destinations, e.destinations.size()), domain);
}

compile_op::NvlsRuns compile_op::nvlsReductionRuns(const Graph::Edge& e, std::span<const size_t> domain) {
  NvlsRuns r;
  const size_t k = domain.size();
  if (k < nvlsReduceMinRanks) {
    return r;
  }
  // The sources (those of the executor's domain come first, then those of other domains and the partial results, see
  // crossDomainReductions in compile_op.cc): inputs, there at the start (no waits)
  bool sources = e.sources.size() >= k + e.partialSources && nvlsDomainNodes(firstNodes(e.sources, k), domain);
  for (size_t i : range(sources ? k : 0)) {
    sources &= e.sources[i].filled && (i >= e.sourceWaits.size() || e.sourceWaits[i].index < 0);
  }
  const bool destinations = e.destinations.size() >= k && nvlsDomainNodes(firstNodes(e.destinations, k), domain);
  // Both, from nvlsReduceMinRanks (as an all-reduce of nvlsReduceLoop); sources alone (one destination) from
  // nvlsReduceLocalMinRanks; destinations alone from nvlsStoreMinRanks
  r.sources = sources && (destinations || k >= nvlsReduceLocalMinRanks);
  r.destinations = destinations && (sources || k >= nvlsStoreMinRanks);
  return r;
}

namespace {
// NVLS runs in the NVLS warp loop (see nvlsReduceLoop in generateKernel): MOODIST_COMPILE_OP_NVLS_LOOP=0 leaves them in
// the reduce loop (tests: the same bits either way; experiments)
bool nvlsRunLoopEnabled() {
  const char* env = std::getenv("MOODIST_COMPILE_OP_NVLS_LOOP");
  return !env || std::string_view(env) != "0";
}
// Whether the NVLS warp loop can take an edge of a kernel with NVLS runs: every source and destination besides the
// edge's runs is a CUDA node of this rank (which the warps load and store with plain loads and stores)
bool nvlsRunLoopEdge(const compile_op::Graph::Edge& e, std::span<const size_t> domain, size_t rank) {
  const compile_op::NvlsRuns runs = compile_op::nvlsReductionRuns(e, domain);
  auto local = [&](const Vector<compile_op::Graph::Node>& nodes, size_t first) {
    return std::all_of(
        nodes.begin() + std::min(first, nodes.size()), nodes.end(), [&](const compile_op::Graph::Node& n) {
          return n.rank == rank && n.device == DeviceType::CUDA;
        });
  };
  return !e.sources.empty() && local(e.sources, runs.sources ? domain.size() : 0) &&
         local(e.destinations, runs.destinations ? domain.size() : 0);
}
} // namespace

bool compile_op::nvlsMulticastRuns(const Group* group, DType dtype, ReduceOp reduce, bool collectiveOp) {
  return nvlsReductionsPossible(group, dtype, reduce, collectiveOp) &&
         group->fabricDomainRanks.size() >= nvlsStoreMinRanks;
}

bool compile_op::nvlsReductionsPossible(const Group* group, DType dtype, ReduceOp reduce, bool collectiveOp) {
  return nvlsDomainCapable(group, collectiveOp, nvlsReduceMinRanks) && nvlsReduceInst(dtype, reduce).has_value();
}

uint64_t compile_op::nvlsReductionMaxBytes(DType dtype) {
  // int32 takes four 4-byte multimem.ld_reduce per vector, where the vector forms take one: all-reduce on 32 GB300,
  // NVLS against the reduce loop: 1 MiB 29 / 46 us, 4 MiB 45 / 50 us, 8 MiB 69 / 53 us, 256 MiB 1554 / 894 us (bfloat16
  // with NVLS: 551 us; float64 and int64, two 8-byte ones, were faster at every size: 256 MiB 818 / 859-894 us)
  return dtype == DType::Int32 ? (4 << 20) : std::numeric_limits<uint64_t>::max();
}

struct Profiling {
  bool enabled = false;
  HashMap<std::string, size_t> nameToIndex;
  size_t nextIndex = 0;
  std::optional<u64> clockOffset;

  std::optional<u64> addrData;

  std::optional<u32> nextEntryIndex;
  std::optional<u32> dataOffset;

  static constexpr size_t maxBytesPerThread = 1024 * 128;
  // A thread's area: the number of its entries that it kept, the number it recorded, then the entries (16 bytes
  // each, of which the first 8 are written); entries that do not fit are dropped (dumpProfiling says so)
  static constexpr uint32_t maxEntries = (maxBytesPerThread - 8) / 16;

  size_t dataSize(const KernelConfig& config) {
    return maxBytesPerThread * config.blockSize * config.gridSize;
  }

  void start(const KernelConfig& config, u64 addrData) {
    u64 myclock = clock64();
    clockOffset = -myclock;
    enabled = true;

    this->addrData = addrData;
    nextEntryIndex = 0;
    dataOffset = maxBytesPerThread * (config.blockSize * blockIdx_x() + threadIdx_x());

    nameToIndex["<invalid>"] = nextIndex++;
  }
  void finish() {
    if (!enabled) {
      return;
    }
    st_global_u32(*addrData + widen(*dataOffset), min_u32(*nextEntryIndex, maxEntries));
    st_global_u32(*addrData + widen(*dataOffset + 4), *nextEntryIndex);
  }
  void enter(std::string name, std::optional<u32> arg = {}) {
    if (!enabled) {
      return;
    }
    if (!nameToIndex.contains(name)) {
      nameToIndex[name] = nextIndex++;
    }
    u32 o = 8 + 16 * *nextEntryIndex;
    u32 a = *dataOffset + o;
    PRED(*nextEntryIndex < maxEntries) {
      st_global_u64(*addrData + widen(a), clock64() + *clockOffset + (nameToIndex[name] << 48));
    }
    *nextEntryIndex = *nextEntryIndex + 1;
  }
  void leave() {
    enter("");
  }
};

static thread_local Profiling profiling;

// The copy engine's stage ring, run by one lane per pipeline (a block runs config.pipelines pipelines, on the first
// lane of each of its first warps). A pipeline copies chunks through a ring of numStages stages of chunkSize bytes in
// shared memory: a bulk load (TMA) of a chunk into a stage, then bulk stores of the stage to each of the chunk's
// destinations. A TMA copy is latency bound: an SM reaches its NVLink bandwidth (~52 GB/s on GB300) only with
// ~100 KB of loads in flight. So the pipeline keeps loads in flight ahead of its stores: after issuing the stores of
// a chunk, it waits until the stores of an earlier chunk have read their stage (cp.async.bulk.wait_group.read) and
// loads the next chunk into that stage (see storeLag and the loops that use the ring). A stage's record holds what
// its stores need: the chunk's offset and size, the generic address of the array of destination addresses (u64),
// and a tag for the caller. The ring's position persists across copies (the mbarrier phases of the stages
// continue).
struct PipeRing {
  static constexpr uint32_t recordSize = 32;
  uint32_t chunkSize;
  uint32_t numStages;
  u32 buffers;  // stage s at buffers + chunkSize * s
  u32 barriers; // the mbarrier of stage s at barriers + 8 * s
  u32 records;  // the record of stage s at records + recordSize * s

  u32 loadStage = 0;   // the stage of the next load
  u32 storeStage = 0;  // the stage of the next store
  u32 storeParity = 0; // the phase parity of the mbarrier of that stage

  PipeRing(uint32_t chunkSize, uint32_t numStages, const u32& buffers, const u32& barriers, const u32& records)
      : chunkSize(chunkSize), numStages(numStages), buffers(buffers), barriers(barriers), records(records) {}

  void init() {
    for (uint32_t s : range(numStages)) {
      mbarrier_init(barriers + 8 * s, 1);
    }
    fence_mbarrier_init();
  }

  // Load bytes at src + offset into the next stage, to be stored to the numDestinations addresses (plus offset) in
  // the array at dsts; tag is returned by its store(), aux kept for nextStoreAux()
  void load(const u64& src, const u64& offset, const u32& bytes, const u64& dsts, const u32& numDestinations,
      const u32& tag, const u32& aux) {
    u32 record = records + loadStage * recordSize;
    st_shared_u64(record, offset);
    st_shared_u64(record + 8, dsts);
    st_shared_u32(record + 16, bytes);
    st_shared_u32(record + 20, numDestinations);
    st_shared_u32(record + 24, tag);
    st_shared_u32(record + 28, aux);
    u32 barrier = barriers + loadStage * 8;
    mbarrier_arrive_expect_tx(barrier, bytes);
    cp_async_bulk_shared_global(buffers + loadStage * chunkSize, src + offset, bytes, barrier);
    u32 next = loadStage + 1;
    loadStage = selp_u32(0, next, next == numStages);
  }

  // The aux value of the chunk that store() stores next (its record is written when it is loaded)
  u32 nextStoreAux() {
    return ld_shared_u32(records + storeStage * recordSize + 28);
  }

  // Wait for the chunk of the next stage to arrive, and issue its stores (one bulk group). Returns its tag.
  u32 store() {
    u32 record = records + storeStage * recordSize;
    mbarrier_wait_parity_spin(barriers + storeStage * 8, storeParity);
    u64 offset = ld_shared_u64(record);
    u64 dsts = ld_shared_u64(record + 8);
    u32 bytes = ld_shared_u32(record + 16);
    u32 numDestinations = ld_shared_u32(record + 20);
    u32 tag = ld_shared_u32(record + 24);
    u32 buffer = buffers + storeStage * chunkSize;
    FOR(u32 i = 0, i < numDestinations, i += 1) {
      cp_async_bulk_global_shared(ld_u64(dsts + widen(i * 8)) + offset, buffer, bytes);
    }
    cp_async_bulk_commit_group();
    u32 next = storeStage + 1;
    Value wrap = next == numStages;
    storeStage = selp_u32(0, next, wrap);
    storeParity = storeParity ^ selp_u32(1, 0, wrap);
    return tag;
  }
};

struct KernelMappedBuffers {
  AllocatedArray& local;
  Vector<PeerArrayRef> remote;
};

struct MemoryKeyCounter {
  SpinMutex mutex;
  size_t nextKey = 1;
  Vector<size_t> freelist;

  HashMap<std::string, bool> freeMap;
  HashMap<size_t, Vector<std::string>> live;

  void put(size_t key) {
    std::lock_guard l(mutex);
    freelist.push_back(key);
    CHECK(live.contains(key));
    for (auto& s : live[key]) {
      freeMap[s] = true;
    }
    live.erase(key);
  }
  size_t get() {
    std::lock_guard l(mutex);
    size_t key;
    if (freelist.empty()) {
      key = nextKey++;
    } else {
      key = freelist.pop_back_value();
    }
    CHECK(!live.contains(key));
    live.emplace(key);
    return key;
  }

  std::string skey(const Group* group, const AllocatedArray* v) {
    return fmt::sprintf("%s-%#x\n", group->name, v->buffer.cudaPointer);
  }

  bool isFree(const Group* group, const AllocatedArray* v) {
    std::lock_guard l(mutex);
    return freeMap.contains(skey(group, v));
  }
  void add(size_t key, const Group* group, const AllocatedArray* v) {
    std::lock_guard l(mutex);
    CHECK(live.contains(key));
    live[key].push_back(skey(group, v));
    auto it = freeMap.find(skey(group, v));
    if (it != freeMap.end()) {
      freeMap.erase(it);
    }
  }
}& memoryKeyCounter = Global();

void returnKernelMemory(size_t key) {
  return memoryKeyCounter.put(key);
}
size_t acquireMemoryKey() {
  auto key = memoryKeyCounter.get();
  return key;
}

KernelMappedBuffers mapn(size_t memoryKey, compile_op::CompileContext& ctx, std::string name, Vector<uint32_t> ranks,
    size_t bytesPerConcurrencyIndex) {
  bytesPerConcurrencyIndex = std::max(std::bit_ceil(bytesPerConcurrencyIndex), (size_t)16);
  auto& x = ctx.cachedBuffers[""];
  AllocatedArray* local = nullptr;
  for (auto& v : x) {
    if (v->itembytes == bytesPerConcurrencyIndex && memoryKeyCounter.isFree(ctx.group, &*v)) {
      local = &*v;
      break;
    }
  }
  if (!local) {
    auto arr =
        std::make_unique<AllocatedArray>(ctx.group->allocateArrayDevice(bytesPerConcurrencyIndex, maxConcurrency));
    local = &*arr;
    ctx.group->buffersToReset.push_back(local);
    x.push_back(std::move(arr));
  }

  memoryKeyCounter.add(memoryKey, ctx.group, local);

  const uint32_t rank = ctx.group->rank;

  KernelMappedBuffers r(*local);

  Vector<uintptr_t> mapped(ranks.size());
  for (size_t i : indices(ranks)) {
    if (ranks[i] == rank) {
      mapped[i] = local->buffer.cudaPointer;
      continue;
    }
    ctx.group->ipcMapper->requestAddressRank(ranks[i], local->buffer.cudaPointer, local->buffer.bytes, &mapped[i]);
  }
  ctx.group->ipcMapper->wait();

  for (size_t i : indices(ranks)) {
    ctx.send(ranks[i], mapped[i], local->itembytes);
  }
  r.remote.resize(ranks.size());
  for (size_t i : indices(ranks)) {
    size_t itembytes;
    ctx.receive(ranks[i], r.remote[i].base, itembytes);
    r.remote[i].itembytes = itembytes;
  }

  return r;
}

// Quick note that PTX docs do not even mention inter-process memory semantics.
// CUDA Programming Guide talks about it a bit, but it's quite vague.
// In practice, we assume that sys scope includes the entire universe.
// We also assume that cuda has sys scope acquire fence before a kernel and
// release after a kernel.
// In practice, nvlink data is not cached, but we do not make this assumption
// (ie we still use proper fences).
// This kernel runs in the remote memory synchronization domain.

std::shared_ptr<KernelHandle> generateKernel(const Group* group, const KernelConfig& config, std::string target,
    const compile_op::Graph& graph, compile_op::CompileContext& ctx, bool tuning) {
  using namespace ptx;

  uint32_t blockSize = config.blockSize;
  uint32_t numBlocks = config.gridSize;

  size_t rank = group->rank;

  auto rv = std::make_shared<KernelHandle>();
  rv->memoryKey = acquireMemoryKey();

  Module mod;
  mod.target = target;
  mod.topLevelComments.push_back(fmt::sprintf("block size: %d  grid size: %d", config.blockSize, config.gridSize));
  mod.topLevelComments.push_back(fmt::sprintf("shared memory: %d", config.sharedMemory));
  setModule(&mod);

  CHECK(!graph.cudaEdges.empty());

  Vector<uint32_t> ranks;
  Vector<uint64_t> localNodes;
  HashMap<uint64_t, uint32_t> localNodeIndices; // (the index of each in localNodes)
  for (auto& v : graph.cudaEdges) {
    for (auto& l : {&v.sources, &v.destinations}) {
      for (auto& x : *l) {
        if (std::ranges::find(ranks, x.rank) == ranks.end()) {
          ranks.push_back(x.rank);
        }
        if (localNodeIndices.try_emplace(x.id, (uint32_t)localNodes.size()).second) {
          localNodes.push_back(x.id);
        }
      }
    }
  }
  std::ranges::sort(ranks);

  using Edge = compile_op::Graph::Edge;
  using Node = compile_op::Graph::Node;

  auto rankIndex = [&](uint32_t rank) {
    auto it = std::ranges::find(ranks, rank);
    CHECK(it != ranks.end());
    return it - ranks.begin();
  };

  size_t myIndex = rankIndex(rank);

  Vector<size_t> myPeerIndex(ranks.size());

  for (size_t i : indices(ranks)) {
    ctx.send(ranks[i], i);
  }
  for (size_t i : indices(ranks)) {
    ctx.receive(ranks[i], myPeerIndex[i]);
  }

  // The exit sync buffer holds one row of ranks.size() slots per block. Peers can have different numbers of
  // ranks (irregular graphs), so writers use the row length of the peer they write to.
  Vector<size_t> peerRanksSize(ranks.size());
  for (uint32_t r : ranks) {
    ctx.send(r, ranks.size());
  }
  for (size_t i : indices(ranks)) {
    ctx.receive(ranks[i], peerRanksSize[i]);
  }

  struct SyncIndex {
    size_t index;
    size_t stride;
  };

  HashMap<uint32_t, HashMap<uint64_t, SyncIndex>> dependencyIndices;

  for (size_t r : ranks) {
    ctx.send(r, localNodes);
  }
  for (size_t r : ranks) {
    Vector<uint64_t> nn;
    ctx.receive(r, nn);
    for (size_t i : indices(nn)) {
      uint64_t id = nn[i];
      SyncIndex si;
      si.index = i;
      si.stride = nn.size();
      CHECK(!dependencyIndices[r].contains(id));
      dependencyIndices[r][id] = si;
    }
  }

  // The nodes that edges read, as seen by this rank and its peers. An edge must signal its destinations if some
  // edge reads one of them, and that edge may be one this rank does not know (finalize gives a rank the edges
  // that touch its nodes): e.g. in a ring_write chain, rank 0 writes rank 1's node, which the edge from rank 1 to
  // rank 2 reads. The reader's executor is a peer of the writer (the node's owner, or the edge's destination,
  // which knows the edge writing the node), so the union over the peers is complete.
  HashMap<uint64_t, bool> readNodes;
  {
    Vector<uint64_t> myReadNodes;
    for (auto& v : graph.cudaEdges) {
      for (auto& n : v.sources) {
        myReadNodes.push_back(n.id);
      }
    }
    for (size_t r : ranks) {
      ctx.send(r, myReadNodes);
    }
    for (size_t r : ranks) {
      Vector<uint64_t> nn;
      ctx.receive(r, nn);
      for (uint64_t id : nn) {
        readNodes[id] = true;
      }
    }
  }

  auto map = [&](std::string name, size_t bytes) {
    return mapn(rv->memoryKey, ctx, name, ranks, bytes);
  };

  size_t addrAddrsBlockStride = graph.localCudaTensorMappings.size();
  auto addrAddrs = map("addresses", sizeof(uint64_t) * addrAddrsBlockStride * numBlocks);
  auto syncAddrs = map("syncs", sizeof(uint32_t) * ranks.size() * numBlocks);
  auto depFilled = map("depFilled", sizeof(uint32_t) * localNodes.size() * numBlocks);

  // The address arrays: the tensor addresses (8 bytes per tensor), the addresses at which the peers map this rank's
  // tensors (8 per mapping) and, with NVLS, the tensors' identities and multicast addresses (16 per tensor). Kernel
  // parameters are limited to 32764 bytes (an op with about 4000 tensor-peer mappings did not JIT), so an op whose
  // arrays could exceed paramBufferMinBytes (counting the NVLS words, decided further below) has them in a buffer
  // instead: a global array of the module with a copy per concurrency slot, which the module's second kernel,
  // compile_op_params, fills from its own parameters before each launch (launchCompileOpKernel). A slot's ops are
  // serialized on the GPU (launchCustomOp), so a launch's copy stays until its kernel has read it.
  // MOODIST_COMPILE_OP_PARAM_BUFFER=1 uses the buffer for every op (tests).
  static constexpr size_t paramBufferMinBytes = 32000;
  const size_t numTensorWords = std::max(graph.tensors.size(), (size_t)1);
  const size_t numMappingWords = std::max(graph.remoteCudaTensorMappings.size(), (size_t)1);
  const char* forceParamBuffer = std::getenv("MOODIST_COMPILE_OP_PARAM_BUFFER");
  const bool paramBuffer =
      sizeof(CompileOpCopyParameters) + 8 * (numTensorWords + numMappingWords) + 16 * numTensorWords >
          paramBufferMinBytes ||
      (forceParamBuffer && std::string_view(forceParamBuffer) == "1");

  auto* fn = mod.newFunction("compile_op_copy");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = blockSize;
    fn->addParamBytes(8, sizeof(CompileOpCopyParameters));
    if (!paramBuffer) {
      fn->addParamBytes(8, sizeof(uint64_t) * numTensorWords);
      fn->addParamBytes(8, sizeof(uint64_t) * numMappingWords);
    }

    activateNewBlock("entry");

    // --- Load kernel parameters ---
    u32 params = paramBase(0);
    u32 stepValue = loadParamField(params, 0, ValType::U32);
    u32 concurrencyIndex = loadParamField(params, 4, ValType::U32);
    static_assert(offsetof(CompileOpCopyParameters, nvlsCounterBase) == 8);
    // PremulSum's factor (CompileOpCopyParameters::scale), as a float64, or converted to float32
    static_assert(offsetof(CompileOpCopyParameters, scale) == 16);
    auto launchScale = [&](bool f64) {
      Value bits = loadParamField(params, 16, ValType::U64);
      if (f64) {
        return bits;
      }
      Value r(ValType::U32);
      arith("cvt.rn.f32.f64", r, bits);
      return r;
    };

    u32 threadIndex = threadIdx_x();
    u32 blockIndex = blockIdx_x();

    u32 warpIndex = threadIndex / 32;
    u32 laneIndex = laneid();

    // Kernel profiling (debugging aid; MOODIST_COMPILE_OP_PROFILE=N): in the kernel of each op (not in the tuning
    // candidates), every thread records a timestamp at each profiling.enter(), and the op's N-th launch writes them as
    // a Chrome trace (dumpProfiling in processgroup.cc). Needs Profiling::maxBytesPerThread of device memory per
    // thread.
    profiling = Profiling();
    if (const char* env = std::getenv("MOODIST_COMPILE_OP_PROFILE"); env && !tuning) {
      int launch = std::atoi(env);
      REQUIRE(launch >= 1, "MOODIST_COMPILE_OP_PROFILE must be the number of the launch to trace (1 or more)");
      auto profilingBuffer = ctx.group->allocateDevice(profiling.dataSize(config));
      profiling = Profiling();
      profiling.start(config, profilingBuffer.cudaPointer);
      rv->profilingData = std::move(profilingBuffer);
      rv->profilingBytesPerThread = Profiling::maxBytesPerThread;
      rv->profilingCountdown = launch - 1;
    }

    auto predicated = [&](auto pred, auto f) {
      return [&, pred = std::move(pred), f = std::move(f)](auto&&... args) {
        if constexpr (std::is_same_v<decltype(f(args...)), void>) {
          PRED(pred) f(args...);
        } else {
          decltype(f(args...)) tmp;
          PRED(pred) tmp = f(args...);
          return tmp;
        }
      };
    };

    auto isThreadZero = threadIndex == 0;
    // (the kernel's start, the launch parity and the copy of the entry's tables into shared memory, is generated after
    // the tables below, which the copy needs: see "The kernel's start")

    auto concurrency = [&](const auto& v) {
      if constexpr (std::is_same_v<std::decay_t<decltype(v)>, PeerArrayRef>) {
        return v.base + widen(v.itembytes * concurrencyIndex);
      } else {
        return v.buffer.cudaPointer + widen(v.itembytes * concurrencyIndex);
      }
    };

    auto findLocalMapping = [&](const auto& r) {
      for (size_t i : indices(graph.localCudaTensorMappings)) {
        auto& v = graph.localCudaTensorMappings[i];
        if (v.rank == r.rank && v.tensorIndex == r.tensorIndex) {
          return i;
        }
      }
      log.error("failed to find mapping for %d %d\n", r.rank, r.tensorIndex);
      for (size_t i : indices(graph.localCudaTensorMappings)) {
        auto& v = graph.localCudaTensorMappings[i];
        log.error(" index %d: %d %d\n", i, v.rank, v.tensorIndex);
      }
      CHECK(false);
    };

    // ---- The edges this rank executes ----
    Vector<Edge> localCopies;

    for (auto& v : graph.cudaEdges) {
      if (v.sources.empty()) {
        // Relay edge: this rank's CPU thread fills the destination over RDMA
        CHECK(v.rdmaSignal >= 0 && v.destinations.size() == 1);
        CHECK(v.executorRank == rank && v.destinations[0].rank == rank);
        localCopies.push_back(v);
        continue;
      }
      // One source, or for a reduction several (see Graph::Edge)
      CHECK(v.sources.size() == 1 || compile_op::isReduction(graph.reduce));
      CHECK(v.destinations.size() >= 1);
      bool localSource = false;
      for (const Node& n : v.sources) {
        CHECK(std::ranges::find(ranks, n.rank) != ranks.end());
        localSource |= n.rank == rank;
      }
      const Node* localdst = nullptr;
      for (const Node& n : v.destinations) {
        CHECK(!n.filled);
        // gotta decide some things about the cudaEdges, rdmaEdges split
        // are these expected to be completely sepatate, such that
        // the following CHECK always passes?
        // this separation is perhaps broken by design
        //    what happens to reductions with mixed cuda and rdma sources?
        //    well, we could set up a temporary buffer with a dependency chain
        // feasible for destination, though...
        CHECK(std::ranges::find(ranks, n.rank) != ranks.end());
        if (n.rank == rank) {
          // fixme: we could have multiple outputs that map to the same cell.
          //        not really sure if that should be handled here, though.
          //        might be better to have a pass in compile_op.cc which sets
          //        up a dependency chain.
          // (the reduce loop stores to any destinations: in an all-reduce in two rounds, an executor stores its share
          // to its output and to its scratch tensor, see crossDomainReductions in compile_op.cc)
          CHECK(localdst == nullptr || compile_op::isReduction(graph.reduce));
          localdst = &n;
        }
      }
      CHECK(localSource || localdst);

      if (v.executorRank == rank) {
        localCopies.push_back(v);
      }
    }
    // reduceKernel: this rank executes reductions (edges with several sources, or parts of a reduction across NVLink
    // domains or in two phases), and runs the reduce loop below
    const bool reduceKernel = std::ranges::any_of(localCopies, [](const Edge& e) {
      return e.sources.size() > 1 || e.hasSignals();
    });
    if (reduceKernel) {
      // The reduce loop's blocks take items in order, and a block waits (for a source from another NVLink domain, or
      // for a region of the scratch tensor to be free) only once it has finished its items before (see reduceLoop).
      // The edges stay in the order of the graph: that of crossDomainReductions in compile_op.cc, in which every wait
      // is for an edge that comes earlier in that order on its rank (or for a transfer of such data), so that no rank
      // waits for a rank that waits for it, and which puts the partial results that other ranks wait for ahead of the
      // edges that use them; in two phases (twoPhaseReductions), the pieces before their copies. (The order of peers
      // below is for copies: a reduction's item reads all its sources and stores to all its destinations.)
    } else {
      // Relay edges first: the scan for ready edges below stops at the first ready copy, and always reaches them
      auto firstCopy = std::stable_partition(localCopies.begin(), localCopies.end(), [](const Edge& e) {
        return e.sources.empty();
      });
      // Then the copies in the order of the peer they move data to, or from, rotated to start at this rank: in rank
      // order, every rank would write to the same peer at once (in an all-to-all, all ranks to rank 0 first, then to
      // rank 1, ...), whose NVLink ingress then limits all of them while the other peers idle. An edge counts by its
      // first destination (multicast edges list their destinations rotated the same way), or by its source if it
      // reads another rank's node into this rank (the read graphs), where the source's egress limits the readers in
      // the same way (all-gather on 32 GB300 with the read graph and 16 blocks: 74 GB/s per GPU, 631 rotated by
      // source).
      std::stable_sort(firstCopy, localCopies.end(), [&](const Edge& a, const Edge& b) {
        auto key = [&](const Edge& e) {
          size_t r = e.destinations.at(0).rank;
          if (r == rank && !e.sources.empty() && e.sources[0].rank != rank) {
            r = e.sources[0].rank;
          }
          return (r + group->size - rank) % group->size;
        };
        return key(a) < key(b);
      });
    }

    // nowaits: no edge waits for another edge or for the CPU thread, so the edges run in order (the reduce loop's
    // waits aside, see Edge::sourceWaits and Edge::destinationWaits)
    bool nowaits = true;
    bool hasRemoteWrites = false;
    size_t maxDestinations = 0;
    for (const Edge& e : localCopies) {
      if (e.sources.empty() || !std::ranges::all_of(e.sources, &Node::filled)) {
        nowaits = false;
      }
      for (const Node& n : e.destinations) {
        if (n.rank != rank) {
          hasRemoteWrites = true;
        }
      }
      maxDestinations = std::max(maxDestinations, e.destinations.size());
    }

    // Whether another edge reads one of e's destination nodes (e.g. the next rank of a ring), and so waits for
    // e to be signaled as done.
    auto hasDependents = [&](const Edge& e) {
      for (auto& n : e.destinations) {
        if (readNodes.contains(n.id)) {
          return true;
        }
      }
      return false;
    };

    // relayItems: the edges wait only for relay edges of this rank (the copies of a proxy's chunks in the multicast
    // graph, see proxyRemoteEdges in compile_op.cc), and no other edge waits for one of them. They then go through the
    // item sequence like edges without waits (see PipeEdge), where a pipeline loads an item of an edge whose source a
    // relay edge fills once the relay's signal has arrived (relaySignal), instead of the rounds below, in which each
    // block copies its own chunks of an edge (chunk c, block c % grid size): the copy rates of the SMs vary under load,
    // and with that static split, an all-gather of 256 MiB per rank across two NVLink domains of 32 GB300 took 30.2 ms
    // (560 GB/s per GPU), in the item sequence 25.3 ms (668 GB/s). The copies wait for their relays' signals in the
    // item sequence, in signal order: the order in which the CPU thread issues the reads. No relay node may be read by
    // an edge of another rank (which would wait for its depFilled slot).
    HashMap<uint64_t, int32_t> relaySignal; // relay node id -> its signal
    bool relayItems = !nowaits && !reduceKernel;
    for (const Edge& e : localCopies) {
      if (e.sources.empty()) {
        relaySignal[e.destinations.at(0).id] = e.rdmaSignal;
      }
    }
    for (const Edge& e : localCopies) {
      if (!e.sources.empty()) {
        const Node& s = e.sources[0];
        relayItems &= e.sources.size() == 1 && (s.filled || relaySignal.contains(s.id)) && !hasDependents(e);
      }
    }
    for (const Edge& e : graph.cudaEdges) {
      if (e.executorRank != rank) {
        for (const Node& n : e.sources) {
          relayItems &= !relaySignal.contains(n.id);
        }
      }
    }
    if (relayItems) {
      // The relay edges (nothing to copy), then the edges without waits, then those with, in signal order
      auto key = [&](const Edge& e) {
        if (e.sources.empty()) {
          return std::make_pair(0, (int32_t)0);
        }
        auto it = relaySignal.find(e.sources[0].id);
        return it == relaySignal.end() ? std::make_pair(1, (int32_t)0) : std::make_pair(2, it->second);
      };
      std::ranges::stable_sort(localCopies, {}, key);
    }

    // ---- Tables ----
    // The kernel reads its edges from tables at runtime, so its code size does not depend on the number of
    // edges. All tables are in one initialized global array.
    //
    // A node address is the node's offset plus, for slot = 8 * tensorIndex | 1, this rank's tensor (a kernel
    // parameter), else the peer's tensor at byte offset slot in the exchanged address table (bit 1 marks the NVLS
    // slot of an edge; see the address table below).
    struct NodeSetup {
      uint32_t slot;
      uint32_t offsetLow;
      uint32_t offsetHigh;
    };
    // An edge record: EdgeSetup, then the node setups of the source and of the destinations, with a fixed stride
    // so that the loads of an edge do not depend on each other
    struct EdgeSetup {
      uint32_t flags;
      uint32_t signal;    // a relay edge's signal index
      uint32_t sourceDep; // depFilled index of an unfilled source node
      uint32_t bytes;
      uint32_t numDestinations;
      uint32_t signals; // signal setup index of the first depFilled slot to write when the edge is done
      uint32_t numSignals;
      uint32_t pad;
    };
    static constexpr size_t sourceSetupOffset = sizeof(EdgeSetup);
    static constexpr size_t destinationSetupsOffset = sizeof(EdgeSetup) + sizeof(NodeSetup);
    const size_t edgeStride = (destinationSetupsOffset + sizeof(NodeSetup) * maxDestinations + 15) / 16 * 16;
    // A depFilled slot on a peer (or this rank): base + itembytes * concurrencyIndex + 4 * (index + stride *
    // blockIndex)
    struct SignalSetup {
      uint64_t base;
      uint32_t itembytes;
      uint32_t index;
      uint32_t stride;
      uint32_t pad;
    };
    // Where to write the address at which a peer maps one of this rank's tensors (in the peer's address table), and
    // in NVLS ops the tensor's identity (in the peer's identity table, at the same slot). (The address is the word of
    // the mapping in the kernel parameters, of its index in graph.remoteCudaTensorMappings: see the address exchange.)
    struct AddrSetup {
      uint64_t base;
      uint32_t itembytes;
      uint32_t stride;
      uint32_t slot;
      uint32_t idParamOffset;
      uint64_t idBase;
      uint32_t idItembytes;
      uint32_t pad;
    };
    static_assert(sizeof(AddrSetup) == 40 && offsetof(AddrSetup, idBase) == 24);
    static constexpr uint32_t edgeRelay = 1;   // relay edge: waits for the CPU thread's signal, copies nothing
    static constexpr uint32_t edgeWait = 2;    // waits for its source node (relay edges: for the signal)
    static constexpr uint32_t edgeAcquire = 4; // fence.acquire.sys before the copy (source written by another agent)
    static constexpr uint32_t edgeSignal = 8;  // other edges wait for its destinations: write their depFilled slots
    static constexpr uint32_t edgeNvls = 16;   // stored with NVLS when the launch can: the NVLS slot follows the dsts
    static constexpr uint32_t edgeSourceWaits = 32; // reduce loop: sources wait for signals (PipeEdge::waits)
    static constexpr uint32_t edgeDoneFlags = 64;   // reduce loop: sets done flags once stored (see Edge::doneFlags)
    static constexpr uint32_t edgeDestinationWaits = 128; // reduce loop: waits before it stores (PipeEdge::destWaits)
    static constexpr uint32_t edgeDoneWhenLoaded = 256;   // reduce loop: sets its done flags once loaded (not stored)
    static constexpr uint32_t edgeRelayWait = 512; // relayItems: loaded once relay signal PipeEdge::waits has arrived
    // NVLS reductions (with edgeNvls: the NVLS slot of the local source follows the destinations): the result goes to
    // every rank with multicast stores (the NVLS slot of the local destination follows the source's)
    static constexpr uint32_t edgeNvlsStore = 1024;
    // NVLS runs in the reduce loop (see nvlsReductionRuns): the first domain.size() sources are one, loaded with
    // multimem.ld_reduce (the NVLS slot of the local source follows the destinations), or the first domain.size()
    // destinations, stored with one multicast store (the NVLS slot of the local destination follows, after the
    // source's if any)
    static constexpr uint32_t edgeNvlsSources = 2048;
    static constexpr uint32_t edgeNvlsDestinations = 4096;
    // A word of the wait table (a Signal that the kernel waits for, see Edge::sourceWaits): the generation in the low
    // waitGenerationBits bits, the index + 1 in the bits above them (0 if the source does not wait), and waitDoneFlag
    // for a done flag of this kernel's (in cpuInBuffer), else the signal of an RDMA transfer (in rdmaSignalBuffer). A
    // word of the done flag table (see Edge::doneFlags): the generation in the low bits, the index above them.
    static constexpr uint32_t waitGenerationBits = 12;
    static constexpr uint32_t waitDoneFlag = 0x80000000;
    static_assert(compile_op::maxGenerations <= (1u << waitGenerationBits));
    auto waitWord = [&](const compile_op::Graph::Signal& s) -> uint32_t {
      if (s.index < 0 || (tuning && !s.done)) {
        // (the tuning launches run without the CPU thread, see PipeEdge below)
        return 0;
      }
      CHECK(s.generation < compile_op::maxGenerations && (uint64_t)s.index + 1 < (waitDoneFlag >> waitGenerationBits));
      return (s.done ? waitDoneFlag : 0) | (((uint32_t)s.index + 1) << waitGenerationBits) | s.generation;
    };

    // ---- NVLS (NVLink SHARP multicast) ----
    // An edge to every rank of the group, at the same tensor and offset on each (an all-gather's), can store each
    // chunk once, to the group's multicast mapping of that offset (see Group::Nvls), which the NVLink switches copy
    // to every GPU, instead of once per destination. One SM then reaches the GPUs' NVLink ingress, where unicast
    // stores need 12-16 (all-gather on 32 GB300: 1M 62 vs 73 us, 4M 208 vs 239, 256M 12.4 vs 12.7 ms). It needs the
    // destination tensors at the same place of one of the group's multicast objects on every rank (at the same
    // offset of every rank's pool, or of the group's communication pool), which every launch checks with the NVLS
    // identities the peers publish (see the address table below); otherwise the launch stores to each destination as
    // usual. A rank uses NVLS if every edge it executes is such an edge, every rank
    // that does stores the same number of chunks (see the step counter below), and the group has at least
    // nvlsMinRanks ranks, all in one NVLink domain: within one node (4 GPUs), multicast stores were slower from 4M
    // on (the switch also sends each chunk back to the sender, 1/4 of a GPU's ingress; 1/32 on 32 GPUs).
    //
    // Step counter: multicast stores from all ranks at full speed overload the switches' links to the GPUs (more
    // than their ingress arrives), which then deliver about 2/3 of their bandwidth (all-gather of 256M on 32 GPUs:
    // 17-19 ms instead of 12.3). So the ranks advance in step: once the stores of a chunk have completed, a rank adds
    // 1 to the op's step counter through its multicast address (one multimem.red reaches every GPU), and it stores
    // its chunk c only once its local copy of the counter shows that every rank has completed its chunks up to
    // c - nvlsWindow. This bounds the data in flight to what the links deliver, without a clock (pacing the stores
    // by time worked too, but needs the rate). The counter of each concurrency slot only grows: launchCustomOp passes
    // its value before the launch (launches times nvlsIncrement).
    static constexpr size_t nvlsMinRanks = 16;
    static_assert(nvlsMinRanks >= Group::Nvls::minDomainRanks);
    static constexpr uint32_t nvlsSignalLag = 1; // a chunk is signaled once the stores of the one after it are issued
    static constexpr size_t nvlsCountersStride = 64; // bytes per counter (one per concurrency slot)
    // With NVLS, a pipeline stores chunks of nvlsChunkSize through its own ring over the same stage memory (up to
    // nvlsMaxStages stages; a chunk's single store reads its stage quickly, so it reloads the stage of the chunk
    // before): per chunk, the window and the counter cost time, and 64 KiB chunks reached 674 GB/s of a GPU's
    // ingress (32 GPUs, all-gather) where 32 KiB chunks reached 654. At least 2 stages: of the 4 the unicast ring has.
    const uint32_t nvlsChunkSize = std::max(config.chunkSize, std::min(65536u, 2 * config.chunkSize));
    static constexpr uint32_t nvlsMaxStages = 4;
    static constexpr uint32_t nvlsStoreLag = 1;
    static_assert(std::tuple_size_v<decltype(KernelHandle::nvlsLaunches)> == maxConcurrency);

    // The local destination of an edge that can store with NVLS (if every rank's pool holds its destination at the
    // same offset), else nullptr
    auto nvlsLocalDestination = [&](const Edge& e) -> const Node* {
      if (e.sources.size() != 1 || !e.sources[0].filled || e.destinations.size() != group->size || hasDependents(e)) {
        return nullptr;
      }
      Vector<uint8_t> seen(group->size);
      const Node* local = nullptr;
      for (const Node& n : e.destinations) {
        const Node& first = e.destinations[0];
        if (n.rank >= group->size || seen[n.rank] || n.tensorIndex != first.tensorIndex || n.offset != first.offset ||
            n.device != DeviceType::CUDA) {
          return nullptr;
        }
        seen[n.rank] = 1;
        if (n.rank == rank) {
          local = &n;
        }
      }
      return local;
    };

    // NVLS reductions: a reduction with a source on every rank of the executor's NVLink domain (the group's ranks in
    // it; the multicast objects are per domain, see Group::Nvls), at the same tensor and offset on each (see
    // nvlsReductionShape), can load each 16-byte vector of its result with multimem.ld_reduce (one, or two or four for
    // float64 and the integers, see nvlsReduceInst) from the multicast address of its local source (the switches
    // combine the values of every member of the multicast object), and store it with one multimem.st to the multicast
    // address of its local destination if it has a destination on every rank of the domain too (an all-reduce's: each
    // GPU then sends its input once and receives the result once, (1 + 1/N) times the data instead of 2 (N - 1) / N
    // with unicast pieces), else to its one destination, on this rank (a reduce-scatter's, a reduce to one rank's). A
    // rank does so (nvlsReduceLoop) if every edge it executes is such an edge, in domains of at least
    // nvlsReduceMinRanks ranks (nvlsReduceLocalMinRanks with one destination), for the dtypes and reduce ops that
    // multimem.ld_reduce has (see nvlsReduceInst) and up to nvlsReductionMaxBytes. Every launch checks the sources too
    // (a multimem.ld_reduce at an offset where a member has no memory bound makes the context fail with
    // CUDA_ERROR_CONTAINED); a launch that cannot use NVLS runs the reduce loop. On 32 GB300 (bfloat16): all-reduce of
    // 64 KiB in 18 us, 256 MiB in 551 us, against 31 and 842 us with the reduce loop. NVLS runs: other ranks' edges,
    // such as those of reductions across NVLink domains (see crossDomainReductions in compile_op.cc), load the first
    // sources of an edge with multimem.ld_reduce, as one source, if they are a source on every rank of the domain at
    // the same place (the inputs of the domain; a partial result that it computes for other domains), and store to the
    // first destinations with one multicast store if they are a destination on every rank of the domain at the same
    // place (an all-reduce's result in the domain), see nvlsReductionRuns; the same dtypes, reduce ops and checks, the
    // same numbers of ranks but nvlsStoreMinRanks for destinations alone. A rank whose edges have all their other
    // sources and destinations on itself (nvlsRunLoopEdge) runs them in nvlsReduceLoop (nvlsRunLoop) in a launch whose
    // every entry passes, else in the reduce loop, where each run falls back on its own (an entry per run); other
    // ranks' edges with runs are always in the reduce loop. The bytes of the vectors that a thread has in flight
    // (nvlsReduceVectors of 16 bytes, an item of a warp), and the bytes that the GPUs of an op have in flight together:
    // in the probe, the fewest blocks that were fastest held 128 KiB per GPU with 32 GPUs, 256 KiB with 16 and 1 MiB
    // with 4 (there limited by the SMs: each GPU combines a quarter of the tensor), about 4 MiB in all; more were
    // slower with 32 (16 blocks of 512 threads with 8 vectors of each: 573 us at 256 MiB). A rank's blocks take its
    // share of that (its bytes of the op's NVLS reductions: the root of a reduce to one rank takes them all, but at
    // most the grid, tuned for the reduce loop, which can limit it: on 16 GB300, 256 MiB took 1.82 ms with the tuned
    // grid, 3.46 ms with 8 blocks, 1.30 with 32, 1.29 with 64).
    static constexpr uint32_t nvlsReduceVectors = 8;
    static constexpr uint32_t nvlsReduceItemBytes = 32 * 16 * nvlsReduceVectors;
    static constexpr uint64_t nvlsReduceInFlightBytes = 4 << 20;

    // Whether the op uses NVLS (decided by all ranks of this rank's NVLink domain together, the same on each), and if
    // so, the chunks this rank stores with it per launch (0 if none), the ranks that store any (all the same number),
    // the window and the offset of the op's counters in the group's counter area; with reductions, whether this rank's
    // kernel reduces with NVLS in nvlsReduceLoop (when the launch can), or has NVLS runs, and whether it runs them in
    // nvlsReduceLoop (when the launch can) instead of the reduce loop, and the blocks of nvlsReduceLoop
    bool nvlsOp = false;
    uint32_t nvlsChunks = 0;  // stored by this rank
    uint32_t nvlsSenders = 0; // the ranks that store any
    uint32_t nvlsWindow = 0;
    size_t nvlsCountersOffset = 0;
    bool nvlsReduceLocal = false;
    uint32_t nvlsReduceBlocks = 0;
    bool nvlsRunsLocal = false;
    bool nvlsRunLoop = false;
    const std::span<const size_t> domain(ctx.group->fabricDomainRanks.data(), ctx.group->fabricDomainRanks.size());
    if (!tuning) {
      Group* g = ctx.group;
      const bool reduction = compile_op::isReduction(graph.reduce);
      // The ranks of an op's kernels that exchange with each other decide together (see the exchange below): those of
      // one domain, if each has exactly the domain's ranks as its peers (ranks), as it does in reductions where a run
      // (or a reduction of nvlsReduceLoop) touches every rank of the domain. Copies use NVLS only in a group that is
      // one domain.
      const bool peersAreDomain = std::ranges::equal(ranks, domain);
      const bool capable =
          peersAreDomain &&
          (reduction ? !graph.tensors.empty() &&
                           compile_op::nvlsReductionsPossible(g, graph.tensors[0].dtype, graph.reduce, ctx.collectiveOp)
                     : nvlsDomainCapable(g, ctx.collectiveOp, nvlsMinRanks) && domain.size() == g->size);
      uint64_t chunks = 0;
      bool all = capable && !reduction && nowaits && !localCopies.empty();
      for (const Edge& e : localCopies) {
        all &= nvlsLocalDestination(e) != nullptr;
        chunks += (e.bytes + nvlsChunkSize - 1) / nvlsChunkSize;
      }
      uint32_t myChunks = all && chunks < (1u << 30) ? (uint32_t)chunks : 0;
      // Reductions: the bytes that this rank combines with NVLS in nvlsReduceLoop (0 if it does not), else the bytes
      // of its edges with NVLS runs, and of those among them with a run of sources (which the switches combine), and
      // whether any has a run of destinations
      uint64_t reduceBytes = 0;
      uint64_t runBytes = 0;
      uint64_t runLoadBytes = 0;
      bool destinationRuns = false;
      bool reduceAll = capable && reduction && reduceKernel;
      for (const Edge& e : localCopies) {
        reduceAll &= compile_op::nvlsReductionShape(e, domain);
        reduceBytes += e.bytes;
        if (capable && reduction) {
          const compile_op::NvlsRuns runs = compile_op::nvlsReductionRuns(e, domain);
          runBytes += runs.sources || runs.destinations ? e.bytes : 0;
          runLoadBytes += runs.sources ? e.bytes : 0;
          destinationRuns |= runs.destinations;
        }
      }
      const bool runsPossible = !reduceAll && capable && reduction && reduceKernel;
      const uint64_t myReduceBytes = reduceAll ? reduceBytes : 0;
      const uint64_t myRunBytes = runsPossible ? runBytes : 0;
      const uint64_t myRunLoadBytes = runsPossible ? runLoadBytes : 0;
      for (uint32_t r : ranks) {
        ctx.send(r, capable, myChunks, myReduceBytes, myRunBytes, myRunLoadBytes);
      }
      bool allCapable = true;
      bool uniform = true;
      uint32_t sendersChunks = 0;
      uint64_t reduceBytesTotal = 0;
      uint64_t runBytesTotal = 0;
      uint64_t runLoadBytesTotal = 0;
      for (uint32_t r : ranks) {
        bool c;
        uint32_t n;
        uint64_t b;
        uint64_t rb;
        uint64_t rlb;
        ctx.receive(r, c, n, b, rb, rlb);
        allCapable &= c;
        if (n) {
          uniform &= nvlsSenders == 0 || n == sendersChunks;
          sendersChunks = n;
          ++nvlsSenders;
        }
        reduceBytesTotal += b;
        runBytesTotal += rb;
        runLoadBytesTotal += rlb;
      }
      // (copies and reductions do not mix: an op with a reduce op has no NVLS copies)
      const bool copies = uniform && nvlsSenders > 0;
      // (every rank of the domain has the same totals; a rank without tensors is not capable; nvlsReductionMaxBytes
      // bounds what the switches combine, not the multicast stores of runs of destinations alone)
      if (reduction && (graph.tensors.empty() || reduceBytesTotal + runLoadBytesTotal >
                                                     compile_op::nvlsReductionMaxBytes(graph.tensors[0].dtype))) {
        reduceBytesTotal = 0;
        runBytesTotal = 0;
      }
      // A rank whose peers are not its domain's ranks says so (capable is false): every rank that has it among its
      // peers hears it, and so does every other rank of the domain (whose peers are the domain's ranks, if capable),
      // so they all agree. setupNvls is collective over the group: copies call it here (the group is one domain, all
      // of ranks), reductions in compile (compile_op.cc, where every rank takes part), and use its result here.
      nvlsOp = allCapable && (copies || reduceBytesTotal > 0 || runBytesTotal > 0) &&
               (reduction ? g->nvls.state == 1 : g->setupNvls());
      if (nvlsOp && copies && g->nvls.nextCounters + nvlsCountersStride * maxConcurrency > Group::Nvls::countersBytes) {
        if (rank == 0) {
          log.error("moodist: compile_op: no NVLS step counters left for more ops; this op does not use NVLS\n");
        }
        nvlsOp = false;
      }
      rv->nvls = nvlsOp;
      // The blocks of nvlsReduceLoop: this rank's share of the bytes in flight (see nvlsReduceVectors), by its bytes
      // of NVLS reductions, or of edges with NVLS runs, among the domain's
      auto nvlsBlocks = [&](uint64_t bytes, uint64_t total) {
        const uint64_t perBlock = (uint64_t)blockSize * nvlsReduceVectors * 16;
        const uint64_t blocks = (nvlsReduceInFlightBytes * bytes / total + perBlock - 1) / perBlock;
        uint32_t r = std::clamp<uint64_t>(blocks, 1, numBlocks);
        // MOODIST_COMPILE_OP_NVLS_BLOCKS: the blocks of every rank that reduces with NVLS (experiments)
        if (const char* env = std::getenv("MOODIST_COMPILE_OP_NVLS_BLOCKS"); env && std::atoi(env) > 0) {
          r = std::min<uint32_t>(std::atoi(env), numBlocks);
        }
        return r;
      };
      if (nvlsOp && runBytesTotal > 0) {
        nvlsRunsLocal = myRunBytes != 0;
        nvlsRunLoop = nvlsRunsLocal && nvlsRunLoopEnabled() && std::ranges::all_of(localCopies, [&](const Edge& e) {
          return nvlsRunLoopEdge(e, domain, rank);
        });
        if (nvlsRunLoop) {
          // With runs of destinations (an all-reduce's: the items of partial results with multimem.ld_reduce, then the
          // plans with multicast stores, each phase with every warp), three times the share: across 2 domains of 32
          // GB300 (bfloat16, 64 MiB / 256 MiB / 1 GiB), 4 blocks took 215 / 779 / 3055 us, 8 222 / 713 / 2595, 12 237
          // / 687 / 2357, 24 262 / 666 / 2430, the reduce loop 263 / 719 / 2353 (across 4 domains of 16, two rounds,
          // the same with any)
          nvlsReduceBlocks = nvlsBlocks(destinationRuns ? 3 * myRunBytes : myRunBytes, runBytesTotal);
        }
        if (rank == domain[0]) {
          log.info("compile_op: NVLS: the reduce loop's edges of %d bytes in the NVLink domain of rank %d with "
                   "multimem.ld_reduce or multicast stores%s\n",
              runBytesTotal, rank, nvlsRunLoop ? fmt::sprintf(" (in the NVLS loop, %d blocks)", nvlsReduceBlocks) : "");
        }
      }
      if (nvlsOp && reduceBytesTotal > 0) {
        nvlsReduceLocal = myReduceBytes != 0;
        // (the ranks that reduce with NVLS, while others of the domain may have NVLS runs, see above)
        if (nvlsReduceLocal) {
          nvlsReduceBlocks = nvlsBlocks(myReduceBytes, reduceBytesTotal);
        }
        if (rank == domain[0]) {
          log.info("compile_op: NVLS: reductions of %d bytes with multimem.ld_reduce in the NVLink domain of rank %d\n",
              reduceBytesTotal, rank);
        }
      }
      if (nvlsOp && copies) {
        nvlsCountersOffset = g->nvls.nextCounters;
        g->nvls.nextCounters += nvlsCountersStride * maxConcurrency;
        nvlsChunks = myChunks;
        // A pipeline holds up to nvlsSignalLag stored but unsignaled chunks while it waits for the counter, so the
        // window must be larger than all of them (8: see the results above)
        nvlsWindow = std::max<uint32_t>(8, config.pipelines * nvlsSignalLag + 1);
        rv->nvlsIncrement = nvlsSenders * sendersChunks;
        if (rank == 0) {
          log.info("compile_op: NVLS: %d ranks store %d chunks of %d bytes each with multicast stores\n", nvlsSenders,
              sendersChunks, nvlsChunkSize);
        }
      }
    }
    // Whether this rank's kernel stores with NVLS (copies), and whether its launches check NVLS identities (copies or
    // reductions)
    const bool nvlsLocal = nvlsChunks != 0;
    const bool nvlsChecks = nvlsLocal || nvlsReduceLocal || nvlsRunsLocal;
    // In NVLS ops, every rank publishes, besides the address at which a peer maps one of its tensors (the address
    // exchange below), where the tensor is in the group's multicast objects (its "NVLS identity", see nvlsIdentity
    // in processgroup.cc), into a table of the peer's with the layout of the address table. launchCustomOp passes the
    // identities of the rank's tensors and their multicast addresses in a fourth kernel parameter (or in the parameter
    // buffer): per tensor, 8 bytes of identity (0 if none), then 8 bytes of multicast address (0 if none).
    std::optional<KernelMappedBuffers> nvlsIds;
    if (nvlsOp) {
      nvlsIds.emplace(map("nvlsids", sizeof(uint64_t) * addrAddrsBlockStride * numBlocks));
      if (!paramBuffer) {
        fn->addParamBytes(8, 16 * numTensorWords);
      }
    }

    // A word of the address arrays (see paramBuffer above): array 1 (tensor addresses), 2 (the peers' mappings) or 3
    // (NVLS words), at a byte offset. The parameter buffer holds the copies of all concurrency slots, each with the
    // three arrays in this order.
    u64 paramBufferAddr;
    if (paramBuffer) {
      rv->paramBuffer = true;
      rv->paramMappingsOffset = 8 * numTensorWords;
      rv->paramNvlsOffset = rv->paramMappingsOffset + 8 * numMappingWords;
      rv->paramBufferStride = rv->paramNvlsOffset + (nvlsOp ? 16 * numTensorWords : 0);
      CHECK((size_t)rv->paramBufferStride * maxConcurrency < (1ull << 31));
      paramBufferAddr = addGlobalVar(16, rv->paramBufferStride * maxConcurrency, "params") +
                        widen(concurrencyIndex * rv->paramBufferStride);
    }
    auto ldArrayWord = [&](int array, const u32& offset) -> u64 {
      CHECK(array >= 1 && array <= 3);
      if (paramBuffer) {
        const uint32_t base = array == 1 ? 0 : array == 2 ? rv->paramMappingsOffset : rv->paramNvlsOffset;
        return ld_global_nc_u64(paramBufferAddr + widen(offset + base));
      }
      return ld_param_u64(paramBase(array) + offset, 0);
    };

    auto nodeSetup = [&](const Node& n) {
      NodeSetup s;
      s.offsetLow = n.offset;
      s.offsetHigh = n.offset >> 32;
      if (n.rank == rank) {
        s.slot = sizeof(uint64_t) * n.tensorIndex | 1;
      } else {
        s.slot = sizeof(uint64_t) * findLocalMapping(n);
      }
      return s;
    };
    auto localNodeIndex = [&](uint64_t id) {
      auto it = localNodeIndices.find(id);
      CHECK(it != localNodeIndices.end());
      return it->second;
    };
    // The depFilled slots to write once e's destination nodes are filled (per block): on this rank, on the
    // destination ranks, and on the destination ranks of edges that read e's destinations.
    auto appendSignals = [&](const Edge& e, std::vector<SignalSetup>& out) {
      HashMap<uint32_t, bool> signalRanks;
      signalRanks[rank] = true;
      for (const Node& n : e.destinations) {
        signalRanks[n.rank] = true;
      }
      for (auto& ne : graph.cudaEdges) {
        bool concerned = false;
        for (auto& n : ne.sources) {
          for (auto& n2 : e.destinations) {
            if (n.id == n2.id) {
              concerned = true;
            }
          }
        }
        if (concerned) {
          for (auto& n : ne.destinations) {
            signalRanks[n.rank] = true;
          }
        }
      }
      for (auto& v : signalRanks) {
        uint32_t r = v.first;
        for (const Node& n : e.destinations) {
          auto it = dependencyIndices[r].find(n.id);
          if (it != dependencyIndices[r].end()) {
            const auto& peer = depFilled.remote.at(rankIndex(r));
            SignalSetup s = {};
            s.base = peer.base;
            s.itembytes = peer.itembytes;
            s.index = it->second.index;
            s.stride = it->second.stride;
            out.push_back(s);
          }
        }
      }
    };

    std::vector<uint8_t> edgeTable(edgeStride * localCopies.size());
    std::vector<SignalSetup> signalTable;
    for (size_t i : indices(localCopies)) {
      const Edge& e = localCopies[i];
      uint8_t* record = edgeTable.data() + edgeStride * i;
      EdgeSetup s = {};
      CHECK(e.bytes < (1ull << 32));
      s.bytes = e.bytes;
      if (e.sources.empty()) {
        s.flags = edgeRelay | edgeWait;
        s.signal = e.rdmaSignal;
      } else {
        // (the record holds one source, for the path with waits; reductions only run without waits, from the
        // PipeEdge table below, which holds all their sources)
        const Node& src = e.sources[0];
        NodeSetup source = nodeSetup(src);
        std::memcpy(record + sourceSetupOffset, &source, sizeof(source));
        if (!src.filled) {
          CHECK(e.sources.size() == 1);
          s.flags |= edgeWait | edgeAcquire;
          s.sourceDep = localNodeIndex(src.id);
        }
        for (const Node& n : e.sources) {
          if (n.rank != rank) {
            s.flags |= edgeAcquire;
          }
        }
      }
      s.numDestinations = e.destinations.size();
      for (size_t d : indices(e.destinations)) {
        NodeSetup n = nodeSetup(e.destinations[d]);
        std::memcpy(record + destinationSetupsOffset + sizeof(NodeSetup) * d, &n, sizeof(n));
      }
      if (hasDependents(e)) {
        s.flags |= edgeSignal;
        s.signals = signalTable.size();
        appendSignals(e, signalTable);
        s.numSignals = signalTable.size() - s.signals;
      }
      std::memcpy(record, &s, sizeof(s));
    }

    std::vector<AddrSetup> addrTable;
    for (size_t i : indices(graph.remoteCudaTensorMappings)) {
      auto& v = graph.remoteCudaTensorMappings[i];
      auto& r = addrAddrs.remote[rankIndex(v.rank)];

      AddrSetup s = {};
      s.base = r.base;
      s.itembytes = r.itembytes;
      s.stride = sizeof(uint64_t) * v.stride;
      s.slot = sizeof(uint64_t) * v.destinationSlot;
      if (nvlsIds) {
        auto& ir = nvlsIds->remote[rankIndex(v.rank)];
        s.idBase = ir.base;
        s.idItembytes = ir.itembytes;
        s.idParamOffset = 16 * v.tensorIndex;
      }
      addrTable.push_back(s);
    }

    // Without waits (and with relayItems, but for the relay edges), the items of all edges form one sequence, which the
    // pipelines of all blocks take items from (see the copy loop below). An item is a chunk of an edge and a group of
    // its destinations: an edge's destinations are split into groups when the op has few chunks, so that the stores of
    // an all-gather of a few chunks to many ranks spread over the SMs too (a load per group: more local reads, which
    // are cheap). Per edge: the index of its first item in the sequence, the destination groups (item i of the edge
    // covers chunk i / groups and destination group i % groups), and the slots of its addresses in the address table
    // that every block resolves at the start (its sources (one, except in reductions), then its destinations, then with
    // NVLS the NVLS slot: the multicast address of the local destination, or 0 if the launch stores to each
    // destination; with NVLS reductions, the NVLS slots of the local source and, with multicast stores, of the local
    // destination). A last record (firstItem = the number of items) ends the sequence. With NVLS, the chunks of all
    // edges also form one sequence (nvlsFirst: the index of an edge's first chunk), which a launch that uses NVLS
    // stores instead; with NVLS reductions, and edges with NVLS runs in nvlsReduceLoop (nvlsRunLoop), so do their
    // items of nvlsReduceItemBytes. The reduce loop takes whole chunks (one destination group).
    const uint32_t chunkSize = config.chunkSize;
    const uint32_t numPipelines = config.pipelines;
    const uint32_t numClaimers = numBlocks * numPipelines;
    struct PipeEdge {
      uint32_t firstItem;
      uint32_t numItems;
      uint32_t bytes;
      uint32_t addrSlot;
      uint32_t numDestinations;
      uint32_t groups;
      uint32_t groupSize; // destinations per group (the last group may have fewer)
      uint32_t flags;
      uint32_t signals;
      uint32_t numSignals;
      uint32_t nvlsFirst;
      uint32_t numSources;
      // Reductions: the sources that an average divides after combining them (the first ones; the others are
      // partial results, see Edge::partialSources) and the divisor; with edgeSourceWaits, the index of the first
      // source's word in the wait table (a word per source, see waitWord); with edgeDestinationWaits, the index of the
      // first word in the wait table that the edge waits for before it stores (Edge::destinationWaits) and their
      // number; with edgeDoneFlags, the index of the first word in the done flag table and their number
      // (Edge::doneFlags). With edgeRelayWait, waits is the signal of the relay edge that fills the source.
      uint32_t inputSources;
      uint32_t divisor;
      uint32_t waits;
      uint32_t destWaits;
      uint32_t numDestWaits;
      uint32_t doneFlags;
      uint32_t numDoneFlags;
      uint32_t pad;
    };
    static_assert(sizeof(PipeEdge) == 80 && offsetof(PipeEdge, nvlsFirst) == 40 &&
                  offsetof(PipeEdge, numSources) == 44 && offsetof(PipeEdge, numDoneFlags) == 72);
    std::vector<PipeEdge> pipeEdges;
    std::vector<NodeSetup> pipeAddrSetups;
    std::vector<uint32_t> pipeWaits;
    std::vector<uint32_t> pipeDoneFlags;
    uint32_t pipeNumItems = 0;
    uint32_t pipeNoWaitItems = 0; // (with relayItems: the items of the edges that wait for no relay, which come first)
    // The nodes that the check of every launch compares with an NVLS slot (see below): the first node's address slot,
    // the number of nodes and the NVLS slot (a destination's for the destinations of copies, the source's for the
    // sources of reductions, the destination's for their destinations with multicast stores)
    std::vector<std::array<uint32_t, 4>> nvlsEdgeTable;
    uint32_t nvlsChunksTotal = 0;
    uint32_t nvlsReduceItems = 0;
    CHECK(chunkSize >= 16 && chunkSize % 16 == 0);
    CHECK(numPipelines >= 1 && numPipelines * 32 <= blockSize);
    // The item sequence (see the copy loop): without waits, or with relayItems
    const bool itemPath = nowaits || relayItems;
    if (itemPath) {
      uint64_t numChunksTotal = 0;
      for (const Edge& e : localCopies) {
        if (!e.sources.empty()) {
          numChunksTotal += (e.bytes + chunkSize - 1) / chunkSize;
        }
      }
      // Destination groups for about 4 items per pipeline: items are taken whole (in batches), so with few items,
      // some pipelines get one more than others, e.g. with 32 chunks to 31 destinations each (an all-gather of
      // 1 MB to 32 GPUs) and 12 pipelines. More groups cost more loads (and were slower at 8 items per pipeline).
      const uint64_t groupsWanted =
          std::max<uint64_t>((4 * numClaimers + numChunksTotal - 1) / std::max<uint64_t>(numChunksTotal, 1), 1);
      for (size_t i : indices(localCopies)) {
        const Edge& e = localCopies[i];
        if (e.sources.empty()) {
          // A relay edge (relayItems): the items of the copies from its node wait for its signal
          continue;
        }
        EdgeSetup s;
        std::memcpy(&s, edgeTable.data() + edgeStride * i, sizeof(s));
        CHECK(e.bytes > 0 && !e.destinations.empty());
        const uint32_t numChunks = (e.bytes + chunkSize - 1) / chunkSize;
        const uint32_t numDestinations = e.destinations.size();
        PipeEdge p = {};
        const uint64_t groups = reduceKernel ? 1 : std::min<uint64_t>(groupsWanted, numDestinations);
        p.groupSize = (numDestinations + groups - 1) / groups;
        p.groups = (numDestinations + p.groupSize - 1) / p.groupSize;
        const Node* nvlsDestination = nvlsLocal ? nvlsLocalDestination(e) : nullptr;
        CHECK(nvlsDestination || !nvlsLocal);
        p.firstItem = pipeNumItems;
        p.numItems = numChunks * p.groups;
        p.bytes = e.bytes;
        p.addrSlot = pipeAddrSetups.size();
        p.numDestinations = numDestinations;
        p.flags = s.flags | (nvlsDestination ? edgeNvls : 0);
        p.signals = s.signals;
        p.numSignals = s.numSignals;
        p.nvlsFirst = nvlsChunksTotal;
        p.numSources = e.sources.size();
        CHECK(e.partialSources <= e.sources.size());
        p.inputSources = e.sources.size() - e.partialSources;
        p.divisor = e.divisor != 0 ? e.divisor : e.sources.size();
        if (auto it = relaySignal.find(e.sources[0].id); relayItems && it != relaySignal.end()) {
          CHECK(it->second >= 0);
          p.flags |= edgeRelayWait;
          p.waits = it->second;
        }
        // (the tuning launches run without the CPU thread: they do not wait for its RDMA signals, but for the done
        // flags of their own edges, which they set as the op's launches do; each launch has a step value of its own)
        CHECK(e.sourceWaits.empty() || e.sourceWaits.size() == e.sources.size());
        if (std::ranges::any_of(e.sourceWaits, waitWord)) {
          p.flags |= edgeSourceWaits;
          p.waits = pipeWaits.size();
          for (const auto& s : e.sourceWaits) {
            pipeWaits.push_back(waitWord(s));
          }
        }
        if (std::ranges::any_of(e.destinationWaits, waitWord)) {
          p.flags |= edgeDestinationWaits;
          p.destWaits = pipeWaits.size();
          for (const auto& s : e.destinationWaits) {
            if (uint32_t w = waitWord(s)) {
              pipeWaits.push_back(w);
            }
          }
          p.numDestWaits = pipeWaits.size() - p.destWaits;
        }
        if (!e.doneFlags.empty()) {
          p.flags |= edgeDoneFlags | (e.doneWhenLoaded ? edgeDoneWhenLoaded : 0);
          p.doneFlags = pipeDoneFlags.size();
          p.numDoneFlags = e.doneFlags.size();
          for (const auto& s : e.doneFlags) {
            CHECK(s.done && s.index >= 0 && s.generation < compile_op::maxGenerations);
            pipeDoneFlags.push_back(((uint32_t)s.index << waitGenerationBits) | s.generation);
          }
        }
        for (const Node& n : e.sources) {
          pipeAddrSetups.push_back(nodeSetup(n));
        }
        for (const Node& n : e.destinations) {
          pipeAddrSetups.push_back(nodeSetup(n));
        }
        if (nvlsDestination) {
          // The NVLS slot: the local destination's setup, marked by bit 1 (see the address table below)
          NodeSetup m = nodeSetup(*nvlsDestination);
          m.slot |= 2;
          pipeAddrSetups.push_back(m);
          nvlsEdgeTable.push_back({p.addrSlot + 1, numDestinations, p.addrSlot + 1 + numDestinations, 0});
          nvlsChunksTotal += (e.bytes + nvlsChunkSize - 1) / nvlsChunkSize;
        }
        if (nvlsRunsLocal) {
          // The NVLS slots of the runs (the local node of each)
          const compile_op::NvlsRuns runs = compile_op::nvlsReductionRuns(e, domain);
          const uint32_t k = domain.size();
          auto local = [&](const Vector<Node>& nodes) {
            auto it = std::find_if(nodes.begin(), nodes.begin() + k, [&](const Node& n) {
              return n.rank == rank;
            });
            CHECK(it != nodes.begin() + k);
            NodeSetup m = nodeSetup(*it);
            m.slot |= 2;
            pipeAddrSetups.push_back(m);
            return (uint32_t)pipeAddrSetups.size() - 1;
          };
          if (runs.sources) {
            nvlsEdgeTable.push_back({p.addrSlot, k, local(e.sources), 0});
            p.flags |= edgeNvlsSources;
          }
          if (runs.destinations) {
            nvlsEdgeTable.push_back({p.addrSlot + (uint32_t)e.sources.size(), k, local(e.destinations), 0});
            p.flags |= edgeNvlsDestinations;
          }
        }
        if (nvlsRunLoop) {
          // The edge's items in nvlsReduceLoop (every edge, with runs or not)
          p.nvlsFirst = nvlsReduceItems;
          CHECK((uint64_t)nvlsReduceItems + (e.bytes + nvlsReduceItemBytes - 1) / nvlsReduceItemBytes < (1ull << 30));
          nvlsReduceItems += (e.bytes + nvlsReduceItemBytes - 1) / nvlsReduceItemBytes;
        }
        if (nvlsReduceLocal) {
          // The NVLS slots of the local source and, with a destination on every rank, of the local destination
          CHECK(compile_op::nvlsReductionShape(e, domain));
          const uint32_t numSources = e.sources.size();
          const uint32_t sourceSlot = pipeAddrSetups.size();
          auto local = [&](const Vector<Node>& nodes) {
            auto it = std::ranges::find(nodes, (uint32_t)rank, &Node::rank);
            CHECK(it != nodes.end());
            NodeSetup m = nodeSetup(*it);
            m.slot |= 2;
            pipeAddrSetups.push_back(m);
          };
          local(e.sources);
          nvlsEdgeTable.push_back({p.addrSlot, numSources, sourceSlot, 0});
          if (numDestinations > 1) {
            p.flags |= edgeNvlsStore;
            local(e.destinations);
            nvlsEdgeTable.push_back({p.addrSlot + numSources, numDestinations, sourceSlot + 1, 0});
          }
          p.flags |= edgeNvls;
          p.nvlsFirst = nvlsReduceItems;
          CHECK((uint64_t)nvlsReduceItems + (e.bytes + nvlsReduceItemBytes - 1) / nvlsReduceItemBytes < (1ull << 30));
          nvlsReduceItems += (e.bytes + nvlsReduceItemBytes - 1) / nvlsReduceItemBytes;
        }
        pipeEdges.push_back(p);
        CHECK((uint64_t)pipeNumItems + p.numItems < (1ull << 30));
        pipeNumItems += p.numItems;
        if (!(p.flags & edgeRelayWait)) {
          pipeNoWaitItems += p.numItems;
        }
      }
      PipeEdge end = {};
      end.firstItem = pipeNumItems;
      end.nvlsFirst = nvlsLocal ? nvlsChunksTotal : nvlsReduceItems;
      pipeEdges.push_back(end);
      CHECK(nvlsChunksTotal == nvlsChunks);
    }

    std::vector<uint8_t> tableData;
    auto appendTable = [&](const auto& table, size_t align) {
      size_t offset = (tableData.size() + align - 1) / align * align;
      size_t bytes = sizeof(table[0]) * table.size();
      tableData.resize(offset + bytes);
      if (bytes) {
        std::memcpy(tableData.data() + offset, table.data(), bytes);
      }
      return offset;
    };
    size_t edgeTableOffset = appendTable(edgeTable, 8);
    size_t signalTableOffset = appendTable(signalTable, 8);
    size_t pipeEdgeTableOffset = appendTable(pipeEdges, 8);
    size_t pipeWaitsOffset = appendTable(pipeWaits, 8);
    size_t pipeDoneFlagsOffset = appendTable(pipeDoneFlags, 8);
    size_t addrTableOffset = appendTable(addrTable, 8);
    // The tables that the address table and the NVLS check below read (the node setups and the NVLS edge table), the
    // entry's tables, come last, together: a block copies them into shared memory at its start (see entryTablesStaged)
    const size_t entryTablesOffset = (tableData.size() + 15) / 16 * 16;
    size_t pipeAddrSetupOffset = appendTable(pipeAddrSetups, 16);
    size_t nvlsEdgeTableOffset = appendTable(nvlsEdgeTable, 8);
    CHECK(pipeAddrSetupOffset == entryTablesOffset);
    // (a whole number of 16-byte units, for the bulk copy)
    const size_t entryTablesBytes = (tableData.size() - entryTablesOffset + 15) / 16 * 16;
    tableData.resize(entryTablesOffset + entryTablesBytes);
    u64 tables = addGlobalData(16, tableData.data(), tableData.size(), "tables");

    // Shared memory: with waits, the scan for ready edges keeps a done bit per edge and the selected edge, and the
    // pipelines need the destination addresses of the edge being copied; without waits, the address table (in
    // global memory if it does not fit in pipeTableSharedLimit). The stage rings take the rest.
    const uint32_t numEdges = localCopies.size();
    const uint32_t numDoneWords = (numEdges + 31) / 32;
    const uint32_t scanShared = itemPath ? 0 : (4 * numDoneWords + 8 + 15) / 16 * 16;
    static constexpr size_t pipeTableSharedLimit = 32 * 1024;
    const size_t pipeTableBytes = 8 * pipeAddrSetups.size();
    bool pipeTableInShared = false;
    uint32_t numStages = 0;
    size_t spareShared = 0;      // what the stages leave of config.sharedMemory
    size_t stageMemoryBytes = 0; // the stage memory, of all stages (and pipelines)
    // The reduce loop (see below): stages of chunkSize with an mbarrier each (no records), reduceResults result
    // buffers, and a ring of reduceItemSlots claimed items (8 bytes each)
    uint32_t reduceItemSlots = 0;
    if (reduceKernel) {
      CHECK(nowaits);
      // (tuneKernels checks the configurations of reductions on every rank)
      std::string error = reduceConfigError(config);
      REQUIRE(error.empty(), "compile_op: %s", error);
      const size_t perStage = reduceStageBytes(config);
      // (including, with NVLS reductions, the check's flag)
      size_t fixed = reduceFixedBytes(config);
      if (pipeTableBytes <= pipeTableSharedLimit && config.sharedMemory >= fixed + pipeTableBytes + 2 * perStage) {
        pipeTableInShared = true;
        fixed += pipeTableBytes;
      }
      numStages = std::min<size_t>((config.sharedMemory - fixed) / perStage, reduceMaxStages);
      reduceItemSlots = numStages + 2;
      spareShared = config.sharedMemory - fixed - numStages * perStage;
      stageMemoryBytes = numStages * chunkSize;
      mod.topLevelComments.push_back(fmt::sprintf("reduce loop: chunk size %d, %d stages%s", chunkSize, numStages,
          pipeTableInShared ? "" : ", address table in global memory"));
    } else {
      const size_t perStage = chunkSize + 8 + PipeRing::recordSize;
      // 256: alignment padding; with NVLS, the check's flag and the NVLS rings' barriers and records
      size_t fixed = 256 + scanShared + (itemPath ? 0 : 8 * maxDestinations) +
                     (nvlsLocal ? 32 + numPipelines * nvlsMaxStages * (8 + PipeRing::recordSize) : 0);
      if (itemPath && pipeTableBytes <= pipeTableSharedLimit &&
          config.sharedMemory >= fixed + pipeTableBytes + 4 * numPipelines * perStage) {
        pipeTableInShared = true;
        fixed += pipeTableBytes;
      }
      REQUIRE(config.sharedMemory >= fixed + 4 * numPipelines * perStage,
          "compile_op: %d bytes of shared memory are not enough for 4 stages of %d bytes per pipeline",
          config.sharedMemory, chunkSize);
      numStages = std::min<size_t>((config.sharedMemory - fixed) / (numPipelines * perStage), 16);
      spareShared = config.sharedMemory - fixed - numStages * numPipelines * perStage;
      stageMemoryBytes = numPipelines * numStages * chunkSize;
      mod.topLevelComments.push_back(fmt::sprintf("chunk size %d, %d pipelines of %d stages per block%s", chunkSize,
          numPipelines, numStages, itemPath && !pipeTableInShared ? ", address table in global memory" : ""));
    }
    // The entry's tables in shared memory (see entryTablesOffset): the address table and the NVLS check below load
    // their words in loops after barriers, where the loads cannot start early, and from global memory, each took a
    // round trip to L2 more (the NVLS check two: its entry, then the node setups that the entry names), on the way of
    // every rank to its copies once the last peer has published its addresses. A block copies them at its start with
    // one bulk copy (cp.async.bulk) into the start of its stage memory, which the copy loops first load into after the
    // entry (after the barrier that ends it), so they take no shared memory of their own: the copy's mbarrier only (8
    // bytes, from what the stages leave; a buffer of their own would take a stage of 32 KiB from the default
    // configurations, which leave about 1 KiB). If they fit in entryTablesStageLimit and in the stage memory, and the
    // stages leave room for the mbarrier (else the threads load them from global memory). On 32 GB300 (one NVLink
    // domain, the kernel of the rank that waits least, bfloat16): all-reduce of 4 KiB 11.38 -> 11.17 us, 1 MiB 14.16
    // -> 13.84, reduce-scatter of 4 KiB per rank 9.36 -> 8.79, 256 KiB 20.76 -> 20.21, all-gather of 4 KiB per rank
    // 9.81 -> 9.47 (with the address exchange's parameter load below); on 4 GPUs (no NVLS check), within 0.1 us. (The
    // address exchange loads its own table from global memory: the compiler issues those loads at the kernel's start,
    // with the launch parity's, and the bulk copy takes longer than both, about 0.6 us: staged, it delayed the
    // exchange, and the kernel was up to 0.25 us slower on 4 GPUs.)
    static constexpr size_t entryTablesStageLimit = 16 * 1024;
    const bool entryTablesStaged = entryTablesBytes > 0 && entryTablesBytes <= entryTablesStageLimit &&
                                   entryTablesBytes <= stageMemoryBytes && spareShared >= 16;

    // Each pipeline's stage ring (the warps beyond the pipelines alias the first ring, and never use it); with the
    // reduce loop, its stages, barriers, result buffers and item ring
    std::optional<PipeRing> ring;
    Value isPipelineLane = (laneIndex == 0) & (warpIndex < numPipelines);
    u32 stageMemory; // the stage memory (of all pipelines)
    u32 reduceStages;
    u32 reduceBarriers;
    u32 reduceResultBuffers;
    u32 reduceItemRing;
    if (reduceKernel) {
      reduceStages = addShared32(128, numStages * chunkSize, "reducestages");
      reduceResultBuffers = addShared32(128, reduceResults * chunkSize, "reduceresults");
      reduceBarriers = addShared32(8, numStages * 8, "reducebarriers");
      reduceItemRing = addShared32(8, reduceItemSlots * 8, "reduceitems");
      stageMemory = reduceStages;
    } else {
      u32 buffers = addShared32(128, numPipelines * numStages * chunkSize, "stages");
      u32 barriers = addShared32(8, numPipelines * numStages * 8, "stagebarriers");
      u32 records = addShared32(16, numPipelines * numStages * PipeRing::recordSize, "stagerecords");
      u32 w = selp_u32(warpIndex, 0, warpIndex < numPipelines);
      ring.emplace(chunkSize, numStages, buffers + w * (numStages * chunkSize), barriers + w * (numStages * 8),
          records + w * (numStages * PipeRing::recordSize));
      stageMemory = buffers;
    }
    // With NVLS, each pipeline's ring of nvlsChunkSize stages over the same stage memory (a launch uses one of them)
    std::optional<PipeRing> nvlsRing;
    uint32_t nvlsStages = 0;
    if (nvlsLocal) {
      nvlsStages = std::min<uint32_t>(numStages * chunkSize / nvlsChunkSize, nvlsMaxStages);
      CHECK(nvlsStages >= 2);
      u32 barriers = addShared32(8, numPipelines * nvlsStages * 8, "nvlsbarriers");
      u32 records = addShared32(16, numPipelines * nvlsStages * PipeRing::recordSize, "nvlsrecords");
      u32 w = selp_u32(warpIndex, 0, warpIndex < numPipelines);
      nvlsRing.emplace(nvlsChunkSize, nvlsStages, ring->buffers, barriers + w * (nvlsStages * 8),
          records + w * (nvlsStages * PipeRing::recordSize));
    }

    // ---- The kernel's start ----
    // The first thread of the last warp starts the bulk copy of the entry's tables (see entryTablesStaged) first, so
    // that it overlaps the launch parity below (not thread 0, whose load of the parity it would delay, and with it the
    // address exchange); the threads wait for it before the address table.
    u32 entryTables;  // the staged tables (at the start of the stage memory)
    u32 entryBarrier; // the copy's mbarrier
    if (entryTablesStaged) {
      entryTables = stageMemory;
      entryBarrier = addShared32(8, 8, "entrybarrier");
      IF_D(threadIndex == blockSize - 32) {
        mbarrier_init(entryBarrier, 1);
        fence_mbarrier_init();
        u32 bytes = (int64_t)entryTablesBytes;
        mbarrier_arrive_expect_tx(entryBarrier, bytes);
        cp_async_bulk_shared_global(entryTables, tables + entryTablesOffset, bytes, entryBarrier);
      }
      // The last warp reconverges before the launch parity's block barrier (bar.red, aligned): the ptxas of CUDA 13.0
      // can fail to reconverge a warp after a branch of one thread (see the reduce loop)
      warp_sync();
    }
    // A word of the entry's tables, at a byte offset into them: from shared memory if staged, else from global memory.
    // Only in the entry (the address table and the NVLS check): the copy loops load into the stage memory after the
    // barrier that ends it, over the staged copy, so code after that reads these tables from global memory (tables +
    // pipeAddrSetupOffset, tables + nvlsEdgeTableOffset).
    auto ldEntry32 = [&](const u32& offset) -> u32 {
      if (entryTablesStaged) {
        return ld_shared_u32(entryTables + offset);
      }
      return ld_global_nc_u32(tables + entryTablesOffset + widen(offset));
    };
    // (the offsets of the entry's tables in them)
    const uint32_t pipeAddrSetupEntry = pipeAddrSetupOffset - entryTablesOffset;
    const uint32_t nvlsEdgeTableEntry = nvlsEdgeTableOffset - entryTablesOffset;

    profiling.enter("parity setup");

    u64 parityAddr = addGlobalVar(4, maxConcurrency * numBlocks * 4, "parity") +
                     widen(4 * numBlocks * concurrencyIndex + 4 * blockIndex);
    u32 parityBit = 0;
    parityBit = predicated(isThreadZero, ld_global_u32)(parityAddr) ^ 1;
    profiling.enter("parity setup bar");
    // (also makes the mbarrier_init above visible to the other threads, which wait on the mbarrier below)
    auto parityPred = bar_red_or(0, blockSize, (parityBit != 0) & isThreadZero);
    profiling.enter("parity setup post-bar");
    predicated(isThreadZero, st_global_u32)(parityAddr, parityBit);
    parityBit = selp_u32(1, 0, parityPred);

    // The tag of the words that this launch publishes (the addresses of the address exchange below, and in NVLS ops the
    // identities): bit 63 with parity 1, bit 62 with parity 0, so that a word of 0 never passes for a published one.
    // The step value reset (resetStepValue: at every compile, and when the step values wrap) clears these tables, also
    // those of ops that are still in use, whose parity goes on: with a tag of 0 for parity 0, a launch would take a
    // cleared word for its peer's (an address of 0).
    static constexpr uint64_t parityTagMask = 3ull << 62;
    u64 parityTag = (widen(parityBit) << 63) | (widen(parityBit ^ 1) << 62);

    profiling.leave();

    // The stage rings' mbarriers
    if (reduceKernel) {
      IF_D(isThreadZero) {
        for (uint32_t s : range(numStages)) {
          mbarrier_init(reduceBarriers + 8 * s, 1);
        }
        fence_mbarrier_init();
      }
    } else {
      IF_D(isPipelineLane) {
        ring->init();
      }
    }
    if (nvlsLocal) {
      IF_D(isPipelineLane) {
        nvlsRing->init();
      }
    }

    struct SyncSetup {
      uint64_t stbase;
      uint32_t itembytes;
      uint32_t ldoffset;
      uint32_t stBlockStride; // bytes per block row in the peer's sync buffer
    };

    std::vector<SyncSetup> syncsetup;
    for (size_t i : indices(ranks)) {
      if (ranks[i] == rank) {
        continue;
      }
      auto& r = syncAddrs.remote[i];
      SyncSetup s;
      s.stbase = r.base + sizeof(uint32_t) * myPeerIndex[i];
      s.itembytes = r.itembytes;
      s.ldoffset = sizeof(uint32_t) * i;
      s.stBlockStride = sizeof(uint32_t) * peerRanksSize[i];
      syncsetup.push_back(s);
    }

    u32 csyncaddr = addConst32(syncsetup, "syncsetup");

    profiling.enter("syncsetup");

    // Address exchange: write the addresses at which the peers map this rank's tensors into their address tables (in
    // NVLS ops, and the tensors' identities into their identity tables)
    for (uint32_t i : range((graph.remoteCudaTensorMappings.size() + blockSize - 1) / blockSize)) {
      u32 index = blockSize * i + threadIndex;
      auto pred = index < graph.remoteCudaTensorMappings.size();
      u64 caddr = tables + addrTableOffset + widen(index * sizeof(AddrSetup));
      u64 base = predicated(pred, ld_global_nc_u64)(caddr + offsetof(AddrSetup, base));
      u32 itembytes = predicated(pred, ld_global_nc_u32)(caddr + offsetof(AddrSetup, itembytes));
      u32 stride = predicated(pred, ld_global_nc_u32)(caddr + offsetof(AddrSetup, stride));
      u32 slot = predicated(pred, ld_global_nc_u32)(caddr + offsetof(AddrSetup, slot));
      u64 addr = base + widen(itembytes * concurrencyIndex + slot + stride * blockIndex);
      // (the word of the mapping: from its index, not from the table, so that the load of the kernel parameter, whose
      // latency is on the way to the store as the table's is, does not wait for the table's)
      predicated(pred, st_global_relaxed_sys_u64)(addr, predicated(pred, [&](const u32& o) {
        return ldArrayWord(2, o);
      })(index * sizeof(uint64_t)) | parityTag);
      if (nvlsIds) {
        u64 idBase = predicated(pred, ld_global_nc_u64)(caddr + offsetof(AddrSetup, idBase));
        u32 idItembytes = predicated(pred, ld_global_nc_u32)(caddr + offsetof(AddrSetup, idItembytes));
        u32 idParamOffset = predicated(pred, ld_global_nc_u32)(caddr + offsetof(AddrSetup, idParamOffset));
        u64 idAddr = idBase + widen(idItembytes * concurrencyIndex + slot + stride * blockIndex);
        predicated(pred, st_global_relaxed_sys_u64)(idAddr, predicated(pred, [&](const u32& o) {
          return ldArrayWord(3, o);
        })(idParamOffset) | parityTag);
      }
    }

    profiling.leave();

    profiling.enter("syncsetup warp_sync");
    warp_sync();
    profiling.leave();

    u64 addrAddrsLocalBase = concurrency(addrAddrs.local) + widen(sizeof(uint64_t) * addrAddrsBlockStride * blockIndex);
    u64 edgeTableAddr = tables + edgeTableOffset;
    u64 signalTableAddr = tables + signalTableOffset;

    // The address of a node from its setup words (the same for all threads)
    auto nodeaddr = [&](const u32& slot, const u32& offsetlo, const u32& offsethi) {
      u64 r;
      IF_D((slot & 1) != 0) {
        r = ldArrayWord(1, slot & ~3u);
      }
      ELSE {
        u64 gaddr = addrAddrsLocalBase + widen(slot);
        r = ld_global_relaxed_sys_u64(gaddr);
        WHILE((r & parityTagMask) != parityTag) {
          nanosleep(10);
          r = ld_global_relaxed_sys_u64(gaddr);
        }
        r &= ~parityTagMask;
      }
      r += widen(offsetlo) | (widen(offsethi) << 32);
      return r;
    };

    // The address of a node from its setup words, for the threads where pred is true (0 for the others)
    auto nodeaddrpred = [&](Value pred, const u32& slot, const u32& offsetlo, const u32& offsethi) {
      profiling.enter("nodeaddr");
      u64 r;
      Value done = !pred;
      Value isParam = (slot & 1) != 0;
      r = predicated(!done & isParam, [&](const u32& o) {
        return ldArrayWord(1, o);
      })(slot & ~3u);
      done |= isParam;
      u64 gaddr = addrAddrsLocalBase + widen(slot);
      r = selp(r, predicated(!done, ld_global_relaxed_sys_u64)(gaddr), done);
      done |= (r & parityTagMask) == parityTag;
      profiling.enter("nodeaddr loop");
      WHILE(vote_sync_any(!done)) {
        nanosleep(10);
        fence_acquire_sys();
        r = selp(r, predicated(!done, ld_global_relaxed_sys_u64)(gaddr), done);
        done |= (r & parityTagMask) == parityTag;
      }
      profiling.enter("nodeaddr post");
      r &= ~parityTagMask;

      r += widen(offsetlo) | (widen(offsethi) << 32);
      profiling.leave();
      return selp_u64(r, 0, pred);
    };

    // With waits, the threads resolve the destination addresses of an edge: slot i of a lane takes destination
    // gindex(i) (the same in all warps)
    const size_t numDstSlots = (maxDestinations + 31) / 32;
    auto gindex = [&](size_t i) {
      return laneIndex + 32 * i;
    };

    // The words of an edge record this thread needs: the header fields, the source's setup, and the setups of the
    // destinations of its slots
    struct EdgeWords {
      u32 flags;
      u32 bytes;
      u32 numDestinations;
      std::array<u32, 3> source;
      Vector<std::array<u32, 3>> destinations;
    };
    // Load the words of the edge at desc into w's registers (if pred, when given). Nothing uses them yet, so the
    // sequential path can have the loads of the next edge in flight while it copies the current one.
    auto loadEdge = [&](EdgeWords& w, const u64& desc, const Value* pred) {
      w.destinations.resize(numDstSlots);
      auto load = [&](const Value& d, const u64& addr) {
        if (pred) {
          PRED(*pred) ld_global_nc_u32_into(d, addr);
        } else {
          ld_global_nc_u32_into(d, addr);
        }
      };
      load(w.flags, desc);
      load(w.bytes, desc + 12);
      load(w.numDestinations, desc + 16);
      for (size_t j : range(3)) {
        load(w.source[j], desc + sourceSetupOffset + 4 * j);
      }
      for (size_t i : range(numDstSlots)) {
        u32 index = gindex(i);
        u64 caddr = desc + destinationSetupsOffset + widen(index * sizeof(NodeSetup));
        // Slots beyond the record's destinations stay unloaded
        Value valid = index < maxDestinations;
        Value p = pred ? valid & *pred : valid;
        for (size_t j : range(3)) {
          PRED(p) ld_global_nc_u32_into(w.destinations[i][j], caddr + 4 * j);
        }
      }
    };

    // Signal (this block's slots) that the destination nodes of the edge at `desc` are filled: a release fence,
    // then the depFilled stores. For the thread that has seen the edge done (thread 0 after a copy by the whole
    // block, or the lane that has seen a relay edge's signal); several lanes may run it at once, hence the
    // divergent loop.
    auto signalEdge = [&](const u64& desc) {
      u32 i = ld_global_nc_u32(desc + 20);
      u32 end = i + ld_global_nc_u32(desc + 24);
      fence_release_sys();
      Label loop;
      Label done;
      LABEL(loop);
      GOTO_IF_NOT_D(i < end, done);
      u64 s = signalTableAddr + widen(i * sizeof(SignalSetup));
      u64 base = ld_global_nc_u64(s);
      u32 itembytes = ld_global_nc_u32(s + 8);
      u32 index = ld_global_nc_u32(s + 12);
      u32 stride = ld_global_nc_u32(s + 16);
      st_global_relaxed_sys_u32(
          base + widen(itembytes * concurrencyIndex + 4 * (index + stride * blockIndex)), stepValue);
      i += 1;
      GOTO(loop);
      LABEL(done);
    };

    Vector<u64> dstaddrs(numDstSlots);
    // Resolve the addresses of an edge from its loaded words: returns the source address, sets dstaddrs
    auto resolveEdge = [&](const EdgeWords& w) {
      u64 srcaddr = nodeaddr(w.source[0], w.source[1], w.source[2]);
      for (size_t i : range(numDstSlots)) {
        auto& d = w.destinations[i];
        dstaddrs[i] = nodeaddrpred(gindex(i) < w.numDestinations, d[0], d[1], d[2]);
      }
      return srcaddr;
    };
    // After storing chunk k, a pipeline loads the next chunk into the stage of chunk k - storeLag, once the stores
    // of that chunk have read it: up to storeLag + 1 chunks' stores in flight (the stores of an all-gather chunk to
    // many GPUs take long), and numStages - storeLag loads
    static constexpr int storeLag = 2;
    CHECK(reduceKernel || storeLag < (int)numStages); // (the reduce loop does not lag, and may have 2 stages)

    // With waits: the destination addresses of the edge being copied, for the pipelines
    std::optional<u32> pipeDestinations;
    if (!itemPath) {
      pipeDestinations = addShared32(8, 8 * std::max(maxDestinations, (size_t)1), "destinations");
    }
    // With waits: copy this block's chunks of an edge, pipelined within the edge. Chunk c of an edge belongs to
    // block c % numBlocks, the same split on every rank, which the per-block depFilled slots rely on; the pipelines
    // of a block take its chunks in turn. All threads call this; it returns when this block's stores are done.
    auto pipeCopyEdge = [&](const u64& srcaddr, const u32& numDestinations, const u32& bytes) {
      IF(warpIndex == 0) {
        for (size_t i : range(numDstSlots)) {
          u32 index = gindex(i);
          PRED(index < numDestinations) st_shared_u64(*pipeDestinations + index * 8, dstaddrs[i]);
        }
      }
      barrier_sync();
      IF_D(isPipelineLane) {
        // The loads below may read data whose readiness generic loads have observed
        fence_proxy_async_global();
        u32 numChunks = (bytes - 1) / chunkSize + 1;
        u32 chunk = blockIndex + warpIndex * numBlocks;
        u64 destinations = cvta_shared(widen(*pipeDestinations));
        u32 loaded = 0;
        u32 stored = 0;
        auto loadChunk = [&]() {
          u64 offset = widen(chunk) * chunkSize;
          u32 n = min_u32(bytes - narrow(offset), chunkSize);
          ring->load(srcaddr, offset, n, destinations, numDestinations, 0, 0);
          chunk += numBlocks * numPipelines;
          loaded += 1;
        };
        WHILE((chunk < numChunks) & (loaded < numStages)) {
          loadChunk();
        }
        WHILE(stored < loaded) {
          ring->store();
          stored += 1;
          // The stage of the chunk before is free once its stores have read it
          IF((chunk < numChunks) & (stored >= storeLag + 1)) {
            cp_async_bulk_wait_group_read(storeLag);
            loadChunk();
          }
        }
        cp_async_bulk_wait_group(0);
      }
      barrier_sync();
    };

    // Copy the edge at `desc` (the same for all threads) with the whole block: flags, bytes and srcaddr taken
    // from its words (copied into registers of their own, so the words can be reloaded), dstaddrs resolved
    auto copyEdge = [&](const u64& desc, const u32& flags, const u32& bytes, const u64& srcaddr,
                        const u32& numDestinations) {
      // Every thread has read the source's address itself (remote sources), and any may issue loads
      IF((flags & edgeAcquire) != 0) {
        fence_acquire_sys();
      }
      pipeCopyEdge(srcaddr, numDestinations, bytes);
      IF_D(isThreadZero & ((flags & edgeSignal) != 0)) {
        signalEdge(desc);
      }
    };

    if (numEdges == 0) {
    } else if (itemPath) {
      // The pipelines of all blocks take the items of the sequence (see PipeEdge) from a counter: the first claim
      // of a pipeline takes enough to fill its ring (if there are enough items for all pipelines), the later ones
      // one item (two were slower: the last pipelines to finish had more left). A pipeline claims its next item when
      // it starts on one, so that the claim has returned when it needs it. Taking items dynamically keeps all SMs
      // busy until the end: the copy rate of an SM varies (with a static split, the slowest SMs finished up to 47%
      // after the others).
      // Each concurrency slot has two counters, used by alternate launches (by the launch parity): block 0 resets
      // the other one, which the previous launch used and the next one will use.
      // (the reduce loop claims one item at a time, see below; with relayItems, the first claims share the items
      // without waits, which come first: with the first claims for all items, in an all-gather of 1 MiB per rank
      // across two domains of 32 GPUs, half the pipelines copied the rank's own input, and the other half waited for
      // the chunk from the other domain)
      const uint32_t firstBatch =
          reduceKernel
              ? 1
              : std::clamp<uint32_t>((relayItems ? pipeNoWaitItems : pipeNumItems) / numClaimers, 1, numStages);
      static constexpr uint32_t pipeBatch = 1;
      u64 claimCounters = addGlobalVar(8, 8 * maxConcurrency, "claims");
      IF_D(isThreadZero & (blockIndex == 0)) {
        st_global_u32(claimCounters + widen(concurrencyIndex * 8 + (parityBit ^ 1) * 4), 0);
      }
      u64 claimCounter = claimCounters + widen(concurrencyIndex * 8 + parityBit * 4);
      u32 nextBase; // the first item of the pending claim
      IF_D(isPipelineLane) {
        atom_global_relaxed_gpu_add_u32_into(nextBase, claimCounter, firstBatch);
      }
      // NVLS reductions and NVLS runs in nvlsReduceLoop: the claim counters of their items, the same way
      std::optional<u64> nvlsClaimCounter;
      if (nvlsReduceLocal || nvlsRunLoop) {
        u64 counters = addGlobalVar(8, 8 * maxConcurrency, "nvlsclaims");
        IF_D(isThreadZero & (blockIndex == 0)) {
          st_global_u32(counters + widen(concurrencyIndex * 8 + (parityBit ^ 1) * 4), 0);
        }
        nvlsClaimCounter = counters + widen(concurrencyIndex * 8 + parityBit * 4);
      }

      // The address table: every block resolves the addresses of all edges at the start (waiting for the peers to
      // publish theirs for this launch), so that edges cost nothing to switch between
      const uint32_t numSlots = pipeAddrSetups.size();
      u64 table;
      if (pipeTableInShared) {
        table = cvta_shared(widen(addShared32(16, 8 * numSlots, "addresses")));
      } else {
        const size_t blockBytes = 8 * numSlots;
        rv->scratch = ctx.group->allocateDevice(blockBytes * numBlocks * maxConcurrency);
        u64 base = (int64_t)rv->scratch.cudaPointer;
        table = base + widen(concurrencyIndex) * (blockBytes * numBlocks) + widen(blockIndex) * blockBytes;
      }
      // NVLS: the flag the check below sets if an entry fails (thread 0 clears it before the resolve barrier)
      const Group::Nvls& groupNvls = ctx.group->nvls;
      u32 nvlsFailedAddr;
      if (nvlsChecks) {
        nvlsFailedAddr = addShared32(4, 4, "nvlsfailed");
        IF_D(isThreadZero) {
          st_shared_u32(nvlsFailedAddr, 0);
        }
      }
      if (entryTablesStaged) {
        profiling.enter("entry tables wait");
        u32 phase = 0;
        mbarrier_wait_parity_spin(entryBarrier, phase);
      }
      profiling.enter("resolve addresses");
      FOR(u32 first = 0, first < numSlots, first += blockSize) {
        u32 i = first + threadIndex;
        Value valid = i < numSlots;
        u32 setup = i * sizeof(NodeSetup) + pipeAddrSetupEntry;
        u32 slot = predicated(valid, ldEntry32)(setup);
        u32 offsetLow = predicated(valid, ldEntry32)(setup + 4);
        u32 offsetHigh = predicated(valid, ldEntry32)(setup + 8);
        u64 addr = nodeaddrpred(valid, slot, offsetLow, offsetHigh);
        if (nvlsChecks) {
          // An NVLS slot (a local node's setup, slot = 8 * tensorIndex | 3): the node through the multicast address of
          // its tensor, if it has one, else 0
          Value isNvls = valid & ((slot & 2) != 0);
          u64 mcTensor = predicated(isNvls, [&](const u32& o) {
            return ldArrayWord(3, o);
          })((slot & ~3u) * 2 + 8);
          u64 mcAddr = selp_u64(mcTensor + (widen(offsetLow) | (widen(offsetHigh) << 32)), 0, mcTensor != 0);
          addr = selp_u64(mcAddr, addr, isNvls);
        }
        PRED(valid) st_u64(table + widen(i * 8), addr);
      }
      bool anyRemoteSources = false;
      bool anySignals = false;
      for (const PipeEdge& p : pipeEdges) {
        anyRemoteSources |= (p.flags & edgeAcquire) != 0;
        anySignals |= (p.flags & edgeSignal) != 0;
      }
      // Copies of sources on other ranks: after the reads of their addresses (see copyEdge)
      if (anyRemoteSources) {
        fence_acquire_sys();
      }
      profiling.enter("resolve addresses barrier");
      // (unaligned: after thread 0's branch above, see the reduce loop)
      barrier_sync_unaligned();
      profiling.leave();

      // NVLS check: an entry of the NVLS edge table (the destinations of copies; the sources of reductions, and their
      // destinations with multicast stores; a run of the reduce loop) passes if every node of it has the identity of
      // the local node of the entry's NVLS slot (the same tensor at the same place of the same multicast object; the
      // node offset is the same on every rank): from the identities the peers publish (with the launch's parity, like
      // the addresses). All of them before any multimem.ld_reduce, which fails (the context, with
      // CUDA_ERROR_CONTAINED) at an offset where a member has no memory bound. Warp w checks the entries w, w +
      // numWarps, ...; an entry that fails gets 0 in its NVLS slot and sets the flag. Copies and nvlsReduceLoop use
      // NVLS only if no entry fails (nvlsMode), a run of the reduce loop if its own entry passes (see runSlotAddress).
      // Every block decides the same: from the same identities.
      Value nvlsMode(ValType::Pred);
      if (nvlsChecks) {
        profiling.enter("nvls check");
        const uint32_t numWarps = blockSize / 32;
        u64 idLocalBase = concurrency(nvlsIds->local) + widen(sizeof(uint64_t) * addrAddrsBlockStride * blockIndex);
        static_assert(sizeof(nvlsEdgeTable[0]) == 16);
        FOR(u32 e = warpIndex, e < (uint32_t)nvlsEdgeTable.size(), e += numWarps) {
          u32 record = e * 16 + nvlsEdgeTableEntry;
          u32 firstSlot = ldEntry32(record);
          u32 numNodes = ldEntry32(record + 4);
          u32 nvlsSlot = ldEntry32(record + 8);
          u64 nvlsAddr = ld_u64(table + widen(nvlsSlot * 8));
          u32 localSlot = ldEntry32(nvlsSlot * sizeof(NodeSetup) + pipeAddrSetupEntry);
          u64 identity = ldArrayWord(3, (localSlot & ~3u) * 2);
          Value ok = (nvlsAddr != 0) & (identity != 0);
          FOR(u32 d0 = 0, d0 < numNodes, d0 += 32) {
            u32 d = d0 + laneIndex;
            Value valid = d < numNodes;
            u32 dslot = predicated(valid, ldEntry32)((firstSlot + d) * sizeof(NodeSetup) + pipeAddrSetupEntry);
            Value remote = valid & ((dslot & 1) == 0);
            u64 idAddr = idLocalBase + widen(dslot);
            u64 published = predicated(remote, ld_global_relaxed_sys_u64)(idAddr);
            Value done = !remote | ((published & parityTagMask) == parityTag);
            WHILE(vote_sync_any(!done)) {
              nanosleep(10);
              published = selp(published, predicated(!done, ld_global_relaxed_sys_u64)(idAddr), done);
              done |= (published & parityTagMask) == parityTag;
            }
            Value bad = remote & ((published & ~parityTagMask) != identity);
            ok = ok & !vote_sync_any(bad);
          }
          IF_D(!ok & (laneIndex == 0)) {
            st_u64(table + widen(nvlsSlot * 8), 0);
            st_shared_u32(nvlsFailedAddr, 1);
          }
        }
        // (unaligned: after lane 0's branch above, see the reduce loop)
        barrier_sync_unaligned();
        nvlsMode = ld_shared_u32(nvlsFailedAddr) == 0;
        // Count the launch, if every entry passed (see Group::Nvls::launches), and if so, whether it runs
        // nvlsReduceLoop (Group::Nvls::loopLaunches)
        IF_D(nvlsMode & isThreadZero & (blockIndex == 0)) {
          u64 launches = (int64_t)groupNvls.launches;
          u32 old;
          atom_global_relaxed_gpu_add_u32_into(old, launches, 1);
          if (nvlsReduceLocal || nvlsRunLoop) {
            u64 loopLaunches = (int64_t)groupNvls.loopLaunches;
            atom_global_relaxed_gpu_add_u32_into(old, loopLaunches, 1);
          }
        }
        profiling.leave();
      }

      std::optional<u64> edgeDone;
      if (anySignals) {
        edgeDone = addGlobalVar(4, 4 * maxConcurrency * numEdges, "edgedone");
      }

      // The op's step counter of this concurrency slot, through the multicast and the local mapping
      u64 nvlsCounterMc;
      u64 nvlsCounterLocal;
      if (nvlsLocal) {
        u64 slotOffset = widen(concurrencyIndex * (uint32_t)nvlsCountersStride);
        nvlsCounterMc = slotOffset + (int64_t)(groupNvls.countersMc + nvlsCountersOffset);
        nvlsCounterLocal = slotOffset + (int64_t)(groupNvls.countersLocal + nvlsCountersOffset);
      }

      // Without NVLS: the pipelines of all blocks take items (see above). With NVLS edges, a launch that does not
      // use NVLS adds the chunks to the op's step counter at the end, so that its value stays the same for every
      // launch (ranks whose launch uses NVLS may wait for them).
      auto unicastCopy = [&]() {
        IF_D(isPipelineLane) {
          profiling.enter("copy");
          // The loads below may read data whose readiness generic loads have observed
          fence_proxy_async_global();
          u64 pipeEdgeTable = tables + pipeEdgeTableOffset;
          u32 item = 0;     // the next item of the current batch
          u32 batchEnd = 0; // the end of the current batch
          u32 claimSize = firstBatch;
          Value exhausted(ValType::Pred);
          exhausted = 0;
          // The edge of the last item loaded, and the first item of the edge after it
          u32 edge = 0;
          u32 nextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge));
          u32 loaded = 0;
          u32 stored = 0;
          // With NVLS edges: the NVLS chunks (of nvlsChunkSize, see PipeEdge::nvlsFirst) of the edges whose first
          // item this pipeline loaded
          u32 nvlsLoaded;
          if (nvlsLocal) {
            nvlsLoaded = 0;
          }
          // With relayItems: whether the edge of the item taken last waits for a relay signal that this pipeline has
          // not seen arrive yet (its items, and edges, only increase)
          const bool relayWaits = relayItems && !tuning;
          Value waitPending(ValType::Pred);
          auto edgeWaits = [&]() {
            u32 flags = ld_global_nc_u32(pipeEdgeTable + widen(edge * sizeof(PipeEdge)) + offsetof(PipeEdge, flags));
            return (flags & edgeRelayWait) != 0;
          };
          if (relayWaits) {
            waitPending = edgeWaits();
          }

          // Take the next item; false if there are none left
          auto take = [&]() {
            IF(item >= batchEnd) {
              u32 base = nextBase;
              item = base;
              batchEnd = min_u32(base + claimSize, pipeNumItems);
              IF(base < pipeNumItems) {
                claimSize = pipeBatch;
                atom_global_relaxed_gpu_add_u32_into(nextBase, claimCounter, pipeBatch);
              }
            }
            return item < batchEnd;
          };
          // Items only increase: advance to the edge of `item`
          auto advance = [&]() {
            WHILE(item >= nextFirst) {
              edge += 1;
              nextFirst = ld_global_nc_u32(pipeEdgeTable + widen((edge + 1) * sizeof(PipeEdge)));
              if (relayWaits) {
                waitPending = edgeWaits();
              }
            }
          };
          // With relayItems: whether `item` (taken, not loaded) can be loaded: unless its edge's source is filled by a
          // relay edge's RDMA read, whose signal has not arrived yet (a word per NIC, all of which get the step value,
          // see writeRdmaSignal in cputhread.cc). Blocking: wait for it. (The tuning launches run without the CPU
          // thread, and do not wait.)
          auto ready = [&](bool blocking) {
            Value r(ValType::Pred);
            r = 1;
            if (relayWaits) {
              advance();
              IF(waitPending) {
                u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
                u32 signal = ld_global_nc_u32(record + offsetof(PipeEdge, waits));
                u64 flag = concurrency(group->rdmaSignalBuffer) + widen(signal * Group::rdmaSignalBytes);
                static_assert(Group::rdmaSignalBytes == 16 && maxDevices == 4);
                auto load = [&]() {
                  std::array<Value, 4> words;
                  ld_global_relaxed_sys_v4_u32(words, flag);
                  return min_u32(min_u32(words[0], words[1]), min_u32(words[2], words[3]));
                };
                u32 value = load();
                if (blocking) {
                  IF(value < stepValue) {
                    profiling.enter("relay wait");
                    WHILE(value < stepValue) {
                      value = load();
                    }
                    profiling.leave();
                  }
                }
                r = value >= stepValue;
                IF(r) {
                  // The data arrived before the signal; the loads are bulk copies
                  fence_acquire_sys();
                  fence_proxy_async_global();
                  waitPending = 0;
                }
              }
            }
            return r;
          };
          auto loadItem = [&]() {
            advance();
            u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
            u32 edgeFirst = ld_global_nc_u32(record);
            u32 edgeBytes = ld_global_nc_u32(record + 8);
            u32 addrSlot = ld_global_nc_u32(record + 12);
            u32 numDestinations = ld_global_nc_u32(record + 16);
            u32 groups = ld_global_nc_u32(record + 20);
            u32 groupSize = ld_global_nc_u32(record + 24);
            u32 flags = ld_global_nc_u32(record + 28);
            u64 src = ld_u64(table + widen(addrSlot * 8));
            u32 index = item - edgeFirst;
            u32 chunk = index / groups;
            u32 firstDestination = (index - chunk * groups) * groupSize;
            u64 offset = widen(chunk) * chunkSize;
            u32 n = min_u32(edgeBytes - narrow(offset), chunkSize);
            u32 numGroupDestinations = min_u32(numDestinations - firstDestination, groupSize);
            // The tag: the edge, and whether it signals (bit 31)
            u32 tag = edge | ((flags & edgeSignal) << 28);
            ring->load(
                src, offset, n, table + widen((addrSlot + 1 + firstDestination) * 8), numGroupDestinations, tag, 0);
            item += 1;
            loaded += 1;
            if (nvlsLocal) {
              IF(index == 0) {
                nvlsLoaded += ld_global_nc_u32(record + sizeof(PipeEdge) + offsetof(PipeEdge, nvlsFirst)) -
                              ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
              }
            }
          };

          if (relayItems) {
            // As below, but an item whose relay signal has not arrived is not loaded yet: the pipeline stores the
            // items it has loaded meanwhile, and waits for the signal only once it has none left. It loads an item
            // into a stage that no item has used, or, once the stores of all but the last storeLag items stored have
            // read their stages, while fewer than numStages - storeLag loads are ahead of the stores: one per store
            // as below, more after a wait.
            CHECK(!anySignals && !nvlsLocal);
            WHILE(!exhausted & (loaded < numStages)) {
              IF(take()) {
                IF(!ready(false)) {
                  BREAK;
                }
                loadItem();
              }
              ELSE {
                exhausted = 1;
              }
            }
            WHILE(true) {
              IF(stored < loaded) {
                ring->store();
                stored += 1;
                IF(!exhausted & (stored >= storeLag + 1)) {
                  cp_async_bulk_wait_group_read(storeLag);
                  WHILE(!exhausted & (loaded - stored < numStages - storeLag)) {
                    IF(take()) {
                      IF(!ready(false)) {
                        BREAK;
                      }
                      loadItem();
                    }
                    ELSE {
                      exhausted = 1;
                    }
                  }
                }
              }
              ELSE {
                IF(exhausted) {
                  BREAK;
                }
                IF(take()) {
                  cp_async_bulk_wait_group_read(storeLag);
                  ready(true);
                  loadItem();
                }
                ELSE {
                  exhausted = 1;
                }
              }
            }
          } else {
            // Fill the ring, then store items in order, loading the next one into the stage of the item before (free
            // once its stores have read it)
            WHILE(!exhausted & (loaded < numStages)) {
              IF(take()) {
                loadItem();
              }
              ELSE {
                exhausted = 1;
              }
            }
            WHILE(stored < loaded) {
              u32 tag = ring->store();
              stored += 1;
              if (anySignals) {
                // An item of an edge that other edges (on other ranks) wait for. Its items went to any block, while the
                // waiting kernels wait for per-block slots: the last item to finish signals the slots of all blocks.
                // Per launch, every item counts once, so the counter wraps back to 0.
                IF((tag >> 31) != 0) {
                  u32 e = tag & 0x7fffffff;
                  u64 record = pipeEdgeTable + widen(e * sizeof(PipeEdge));
                  u32 numItems = ld_global_nc_u32(record + 4);
                  cp_async_bulk_wait_group(0);
                  fence_release_gpu();
                  u32 old = atom_global_relaxed_inc_u32(
                      *edgeDone + widen((concurrencyIndex * numEdges + e) * 4), numItems - 1);
                  IF(old == numItems - 1) {
                    fence_acquire_gpu();
                    fence_release_sys();
                    u32 i = ld_global_nc_u32(record + 32);
                    u32 end = i + ld_global_nc_u32(record + 36);
                    WHILE(i < end) {
                      u64 s = signalTableAddr + widen(i * sizeof(SignalSetup));
                      u64 base = ld_global_nc_u64(s);
                      u32 itembytes = ld_global_nc_u32(s + 8);
                      u32 index = ld_global_nc_u32(s + 12);
                      u32 stride = ld_global_nc_u32(s + 16);
                      u64 slot = base + widen(itembytes * concurrencyIndex + 4 * index);
                      FOR(u32 b = 0, b < numBlocks, b += 1) {
                        st_global_relaxed_sys_u32(slot + widen(4 * stride * b), stepValue);
                      }
                      i += 1;
                    }
                  }
                }
              }
              IF(!exhausted & (stored >= storeLag + 1)) {
                cp_async_bulk_wait_group_read(storeLag);
                IF(take()) {
                  loadItem();
                }
                ELSE {
                  exhausted = 1;
                }
              }
            }
          }
          profiling.enter("drain");
          cp_async_bulk_wait_group(0);
          profiling.leave();
          if (nvlsLocal) {
            IF(nvlsLoaded != 0) {
              multimem_red_relaxed_sys_add_u32(nvlsCounterMc, nvlsLoaded);
            }
          }
        }
      };

      // With NVLS: the pipelines of block 0 store the chunks of the NVLS sequence (see PipeEdge), pipeline p chunks
      // p, p + numPipelines, ..., one store each, to the NVLS slot of the chunk's edge, in step with the other ranks
      // (see the step counter above: before storing chunk c, wait until the counter is at least its value before the
      // launch plus nvlsSenders * (c + 1 - nvlsWindow); once a chunk's stores have completed, add 1 to it).
      auto nvlsCopy = [&]() {
        profiling.enter("nvls copy");
        // The loads below may read data whose readiness generic loads have observed
        fence_proxy_async_global();
        u64 pipeEdgeTable = tables + pipeEdgeTableOffset;
        u32 counterBase = loadParamField(params, 8, ValType::U32);
        u32 chunk = warpIndex; // the next chunk this pipeline loads
        // The edge of the last chunk loaded, and the first chunk of the edge after it
        u32 edge = 0;
        u32 nextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge) + offsetof(PipeEdge, nvlsFirst));
        u32 loaded = 0;
        u32 stored = 0;
        u32 signaled = 0;
        auto loadChunk = [&]() {
          // Chunks only increase: advance to the chunk's edge
          WHILE(chunk >= nextFirst) {
            edge += 1;
            nextFirst =
                ld_global_nc_u32(pipeEdgeTable + widen((edge + 1) * sizeof(PipeEdge)) + offsetof(PipeEdge, nvlsFirst));
          }
          u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
          u32 edgeBytes = ld_global_nc_u32(record + 8);
          u32 addrSlot = ld_global_nc_u32(record + 12);
          u32 numDestinations = ld_global_nc_u32(record + 16);
          u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
          u64 src = ld_u64(table + widen(addrSlot * 8));
          u64 offset = widen(chunk - first) * nvlsChunkSize;
          u32 n = min_u32(edgeBytes - narrow(offset), nvlsChunkSize);
          nvlsRing->load(src, offset, n, table + widen((addrSlot + 1 + numDestinations) * 8), 1, edge, chunk);
          chunk += numPipelines;
          loaded += 1;
        };
        WHILE((chunk < nvlsChunks) & (loaded < nvlsStages)) {
          loadChunk();
        }
        WHILE(stored < loaded) {
          // The window. A wait gives up after a second: the counter only paces the stores, and a launch must not
          // hang on it (the kernels of a launch start together: each waits for all peers' addresses)
          u32 c = nvlsRing->nextStoreAux();
          IF(c >= nvlsWindow) {
            u32 target = counterBase + (c + 1 - nvlsWindow) * nvlsSenders;
            u32 value = ld_global_relaxed_sys_u32(nvlsCounterLocal);
            IF((value - target) >= 0x80000000u) {
              profiling.enter("nvls window");
              u64 t0 = globaltimer();
              u64 waited = globaltimer() - t0;
              WHILE(((value - target) >= 0x80000000u) & (waited < 1000000000ull)) {
                value = ld_global_relaxed_sys_u32(nvlsCounterLocal);
                waited = globaltimer() - t0;
              }
              profiling.leave();
            }
          }
          nvlsRing->store();
          stored += 1;
          // Signal the chunks whose stores have completed (all but the last nvlsSignalLag). Relaxed: a release
          // would wait for all of this thread's outstanding stores (1 SM then reached 1/3 of the bandwidth).
          cp_async_bulk_wait_group(nvlsSignalLag);
          IF(stored > signaled + nvlsSignalLag) {
            multimem_red_relaxed_sys_add_u32(nvlsCounterMc, stored - nvlsSignalLag - signaled);
            signaled = stored - nvlsSignalLag;
          }
          IF((chunk < nvlsChunks) & (stored >= nvlsStoreLag + 1)) {
            cp_async_bulk_wait_group_read(nvlsStoreLag);
            loadChunk();
          }
        }
        profiling.enter("drain");
        cp_async_bulk_wait_group(0);
        profiling.leave();
        IF(stored != signaled) {
          multimem_red_relaxed_sys_add_u32(nvlsCounterMc, stored - signaled);
        }
        profiling.leave();
      };

      // The reduce loop (kernels with reductions; see reductionPieces and mergeReductions in compile_op.cc). The blocks
      // take items one at a time (an item is a chunk of an edge, with all its destinations). Thread 0 loads the chunk
      // of each source of an item in turn with bulk loads, into a ring of stages, ahead across items as far as the
      // ring allows, and all threads combine each source's chunk as it arrives into their accumulators, in the order
      // of the edge's sources (the executor's own first, then the others in rank order from there, so that the ranks
      // read from different ranks at the same time). The threads then write the result into one of two result
      // buffers, and thread 0 stores it to every destination with bulk stores (the stores of the item before may still
      // be reading the other buffer). Thread t holds the bytes at 16 * (t + v * blockSize) of a chunk, v < numVectors.
      // The 16-bit floating point types are combined in f32 and rounded (to nearest even) once, when stored. An item
      // with one source (a cell that one input covers) is stored as loaded.
      auto reduceLoop = [&]() {
        profiling.enter("reduce");
        using compile_op::ReduceOp;
        const DType dtype = graph.tensors.at(0).dtype;
        const ReduceOp op = graph.reduce;
        const uint32_t numVectors = chunkSize / (16 * blockSize);
        const ReduceArith ra(dtype, op);
        const bool acc64 = ra.acc64;

        u64 pipeEdgeTable = tables + pipeEdgeTableOffset;

        // Reductions across NVLink domains: sources that the CPU thread reads from other domains (not in the tuning
        // launches, which run without the CPU thread), partial results that it sends to them, or that this kernel
        // combines itself, into regions of rings of the scratch tensor that must be free first (see
        // crossDomainReductions in compile_op.cc); in one domain in two phases, the pieces that this kernel computes
        // into its scratch tensor and then copies to their destinations (see twoPhaseReductions in compile_op.cc)
        bool anySourceWaits = false;
        bool anyDestWaits = false;
        bool anyDoneFlags = false;
        for (const PipeEdge& p : pipeEdges) {
          anySourceWaits |= (p.flags & edgeSourceWaits) != 0;
          anyDestWaits |= (p.flags & edgeDestinationWaits) != 0;
          anyDoneFlags |= (p.flags & edgeDoneFlags) != 0;
        }
        const bool anyWaits = anySourceWaits || anyDestWaits;
        // Per concurrency slot and edge: the items of an edge with done flags stored so far
        std::optional<u64> partialDone;
        if (anyDoneFlags) {
          partialDone = addGlobalVar(4, 4 * maxConcurrency * numEdges, "partialdone");
        }

        // NVLS runs (see nvlsReductionRuns): an edge whose first nvlsRunNodes sources are a run loads them, if the
        // launch can (the run's NVLS slot holds the multicast address of the local source, see the NVLS check), with
        // multimem.ld_reduce, as one source: thread 0 loads nothing for it, its stage only completes a phase, and the
        // threads load their vectors of it from the multicast address when they get to it. An edge whose first
        // nvlsRunNodes destinations are a run stores to them with one bulk store to the multicast address of the local
        // destination (its NVLS slot, after the source's if the edge has both runs), if the launch can.
        bool anyNvlsSources = false;
        bool anyNvlsDestinations = false;
        for (const PipeEdge& p : pipeEdges) {
          anyNvlsSources |= (p.flags & edgeNvlsSources) != 0;
          anyNvlsDestinations |= (p.flags & edgeNvlsDestinations) != 0;
        }
        const uint32_t nvlsRunNodes = domain.size();
        std::optional<NvlsReduceInst> runInst;
        if (anyNvlsSources) {
          runInst = nvlsReduceInst(dtype, op);
          CHECK(runInst.has_value());
        }
        // The multicast address of the run of the edge at record (whose address slot and numbers of nodes are given) in
        // the address table, and the nodes that the run replaces (0 if none, or the launch cannot use NVLS for it)
        auto runSlotAddress = [&](const u64& record, uint32_t flag, const u32& addrSlot, const u32& numSources,
                                  const u32& numDestinations) {
          u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
          Value has = (flags & flag) != 0;
          u32 slot = addrSlot + numSources + numDestinations;
          if (flag == edgeNvlsDestinations) {
            slot = slot + selp_u32(1, 0, (flags & edgeNvlsSources) != 0);
          }
          u64 mc = predicated(has, [&](const u64& a) {
            return ld_u64(a);
          })(table + widen(slot * 8));
          Value use = has & (mc != 0);
          return std::make_pair(selp_u64(mc, 0, use), selp_u32(nvlsRunNodes, 0, use));
        };

        // Thread 0's loads: the stage of the next load, the item being loaded (its edge, the first item of the edge
        // after it, its next source, its number of sources, its address slot, its offset in the edge and its size,
        // and with waits, its first source wait word or ~0 if none, and its destination wait words not passed yet
        // [ldDestWait, ldDestWaitEnd)), its slot in the item ring, and with NVLS runs, the sources that its run of
        // sources replaces (0 if none); ldDone once no items are left; the loads issued so far
        u32 ldStage = 0;
        u32 ldEdge = 0;
        u32 ldNextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge));
        u32 ldSource = 0;
        u32 ldNumSources = 0;
        u32 ldAddrSlot = 0;
        u64 ldOffset = 0;
        u32 ldBytes = 0;
        u32 ldWaits = 0;
        u32 ldDestWait = 0;
        u32 ldDestWaitEnd = 0;
        u32 ldSlot = 0;
        u32 ldRun = 0;
        Value ldDone(ValType::Pred);
        ldDone = 0;
        u32 ldIssued = 0;
        // The stages that the threads have combined so far (the same in all threads)
        u32 consumed = 0;
        // Take the item of the claim in flight, claim the next one, and write the item (its edge and chunk) to the
        // item ring, or past the last item, the end mark (claims of one block only increase)
        auto startItem = [&]() {
          u32 item = nextBase;
          IF(item < pipeNumItems) {
            atom_global_relaxed_gpu_add_u32_into(nextBase, claimCounter, 1);
            WHILE(item >= ldNextFirst) {
              ldEdge += 1;
              ldNextFirst = ld_global_nc_u32(pipeEdgeTable + widen((ldEdge + 1) * sizeof(PipeEdge)));
            }
            u64 record = pipeEdgeTable + widen(ldEdge * sizeof(PipeEdge));
            u32 chunk = item - ld_global_nc_u32(record + offsetof(PipeEdge, firstItem));
            ldNumSources = ld_global_nc_u32(record + offsetof(PipeEdge, numSources));
            ldAddrSlot = ld_global_nc_u32(record + offsetof(PipeEdge, addrSlot));
            ldOffset = widen(chunk) * chunkSize;
            ldBytes = min_u32(ld_global_nc_u32(record + offsetof(PipeEdge, bytes)) - narrow(ldOffset), chunkSize);
            ldSource = 0;
            if (anyNvlsSources) {
              ldRun = runSlotAddress(record, edgeNvlsSources, ldAddrSlot, ldNumSources,
                  ld_global_nc_u32(record + offsetof(PipeEdge, numDestinations)))
                          .second;
            }
            if (anyWaits) {
              u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
              ldWaits = selp_u32(
                  ld_global_nc_u32(record + offsetof(PipeEdge, waits)), 0xffffffffll, (flags & edgeSourceWaits) != 0);
              if (anyDestWaits) {
                Value has = (flags & edgeDestinationWaits) != 0;
                u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, destWaits));
                ldDestWait = selp_u32(first, 0, has);
                ldDestWaitEnd = selp_u32(first + ld_global_nc_u32(record + offsetof(PipeEdge, numDestWaits)), 0, has);
              }
            }
            st_shared_u32(reduceItemRing + ldSlot * 8, ldEdge);
            st_shared_u32(reduceItemRing + ldSlot * 8 + 4, chunk);
          }
          ELSE {
            ldDone = 1;
            u32 endMark = 0xffffffffll;
            st_shared_u32(reduceItemRing + ldSlot * 8, endMark);
          }
        };
        // Load the next source chunk of the item being loaded into the next stage (which the threads have combined
        // into their accumulators), and after the item's last source, take the next item
        auto issueLoad = [&]() {
          u32 barrier = reduceBarriers + ldStage * 8;
          auto bulkLoad = [&]() {
            u64 src = ld_u64(table + widen((ldAddrSlot + ldSource) * 8)) + ldOffset;
            mbarrier_arrive_expect_tx(barrier, ldBytes);
            cp_async_bulk_shared_global(reduceStages + ldStage * chunkSize, src, ldBytes, barrier);
          };
          if (anyNvlsSources) {
            // A run of sources: no data, the phase only (see runSlotAddress above)
            Value run = (ldSource == 0) & (ldRun != 0);
            IF(run) {
              mbarrier_arrive(barrier);
            }
            ELSE {
              bulkLoad();
            }
            ldSource = selp_u32(ldRun, ldSource + 1, run);
          } else {
            bulkLoad();
            ldSource += 1;
          }
          u32 next = ldStage + 1;
          ldStage = selp_u32(0, next, next == numStages);
          ldIssued += 1;
          IF(ldSource == ldNumSources) {
            u32 nextSlot = ldSlot + 1;
            ldSlot = selp_u32(0, nextSlot, nextSlot == reduceItemSlots);
            startItem();
          }
        };
        // Issue loads while the ring has room (thread 0). An item that stores to regions of rings of the scratch tensor
        // is loaded once they are free (its destination waits, before its first source), a source that the CPU thread
        // reads from another NVLink domain once its read has arrived (its signal in rdmaSignalBuffer), and a partial
        // result that this kernel computes for itself (in an all-reduce) once all of it has been stored (its done flag
        // in cpuInBuffer): until then, the loads stop there, and the block goes on with the stages it has. It waits
        // only when it has combined all stages loaded before (blocking, at the start of a stage): its earlier items
        // are then complete, and every wait is for an edge that comes earlier on its rank (see localCopies), so all
        // blocks of all ranks make progress.
        u64 signalFlags = concurrency(group->rdmaSignalBuffer);
        u64 doneFlags = concurrency(group->cpuInBuffer) + 4 * 16;
        auto fill = [&](bool blocking) {
          WHILE(!ldDone & (ldIssued - consumed < numStages)) {
            if (anyWaits) {
              // The next word to wait for: a destination wait of the item not passed yet, else the source's
              u32 wait = 0;
              Value destWait(ValType::Pred);
              destWait = 0;
              if (anyDestWaits) {
                destWait = ldDestWait != ldDestWaitEnd;
                IF(destWait) {
                  wait = ld_global_nc_u32(tables + pipeWaitsOffset + widen(ldDestWait * 4));
                }
              }
              if (anySourceWaits) {
                IF(!destWait & (ldWaits != 0xffffffffll)) {
                  wait = ld_global_nc_u32(tables + pipeWaitsOffset + widen((ldWaits + ldSource) * 4));
                }
              }
              IF(wait != 0) {
                // An RDMA signal is a word per NIC, all of which get the same value (see writeRdmaSignal in
                // cputhread.cc), a done flag one word; the value to wait for is the step value + the generation
                Value done = wait >= (int64_t)waitDoneFlag;
                u32 index = ((wait >> waitGenerationBits) & (int64_t)((waitDoneFlag >> waitGenerationBits) - 1)) - 1;
                u32 target = stepValue + (wait & (int64_t)((1u << waitGenerationBits) - 1));
                u64 flag =
                    selp_u64(doneFlags + widen(index * 4), signalFlags + widen(index * Group::rdmaSignalBytes), done);
                static_assert(Group::rdmaSignalBytes == 16 && maxDevices == 4);
                auto load = [&]() {
                  u32 v = predicated(done, ld_global_relaxed_sys_u32)(flag);
                  std::array<Value, 4> words;
                  PRED(!done) ld_global_relaxed_sys_v4_u32(words, flag);
                  return selp_u32(v, min_u32(min_u32(words[0], words[1]), min_u32(words[2], words[3])), done);
                };
                u32 value = load();
                if (blocking) {
                  IF(ldIssued == consumed) {
                    profiling.enter("reduce wait");
                    WHILE(value < target) {
                      value = load();
                    }
                    profiling.leave();
                  }
                }
                IF(value < target) {
                  BREAK;
                }
                // The data arrived (or was stored) before the signal, or the region that the item stores to has been
                // read (or loaded) before it; the loads and stores are bulk copies
                fence_acquire_sys();
                fence_proxy_async_global();
                if (anyDestWaits) {
                  IF(destWait) {
                    ldDestWait += 1;
                    CONTINUE;
                  }
                }
              }
            }
            issueLoad();
          }
        };

        // Fill the ring. (The block barriers of the loop are unaligned: they follow branches of thread 0, after which
        // the ptxas of CUDA 13.0 can fail to reconverge warp 0, see the warp sync at the end of an item.)
        IF_D(isThreadZero) {
          // The loads below may read data whose readiness generic loads have observed
          fence_proxy_async_global();
          startItem();
          fill(false);
        }
        barrier_sync_unaligned();

        u32 consStage = 0;
        u32 consParity = 0;
        u32 consSlot = 0;
        u32 result = 0; // the result buffer of the next item
        Vector<Vector<Value>> acc(numVectors);
        for (auto& a : acc) {
          for (size_t j : range(ra.accPerVector)) {
            a.emplace_back(acc64 ? ValType::U64 : ValType::U32);
          }
        }
        Vector<std::array<u32, 4>> raw(numVectors); // the last loaded vectors
        WHILE(true) {
          u32 entry = reduceItemRing + consSlot * 8;
          u32 edge = ld_shared_u32(entry);
          IF(edge == 0xffffffffll) {
            BREAK;
          }
          u32 chunk = ld_shared_u32(entry + 4);
          profiling.enter("item");
          u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
          u32 numSources = ld_global_nc_u32(record + offsetof(PipeEdge, numSources));
          u32 numDestinations = ld_global_nc_u32(record + offsetof(PipeEdge, numDestinations));
          u32 addrSlot = ld_global_nc_u32(record + offsetof(PipeEdge, addrSlot));
          u64 offset = widen(chunk) * chunkSize;
          u32 n = min_u32(ld_global_nc_u32(record + offsetof(PipeEdge, bytes)) - narrow(offset), chunkSize);
          // An average divides, PremulSum multiplies by the launch's factor, once it has combined the inputs among
          // the sources (the partial results of other domains after them are scaled already)
          u32 inputSources;
          Value divisor(acc64 ? ValType::U64 : ValType::U32);
          Value factor(acc64 ? ValType::U64 : ValType::U32);
          if (op == ReduceOp::Avg) {
            inputSources = ld_global_nc_u32(record + offsetof(PipeEdge, inputSources));
            arith(acc64 ? "cvt.rn.f64.u32" : "cvt.rn.f32.u32", divisor,
                ld_global_nc_u32(record + offsetof(PipeEdge, divisor)));
          } else if (op == ReduceOp::PremulSum) {
            inputSources = ld_global_nc_u32(record + offsetof(PipeEdge, inputSources));
            factor = launchScale(acc64);
          }
          for (auto& a : acc) {
            for (Value& v : a) {
              v = ra.identity;
            }
          }
          // With an NVLS run of sources: the multicast address of the chunk of the local source, and the sources that
          // the run replaces (0 if none, see runSlotAddress)
          u64 runAddress;
          u32 run;
          if (anyNvlsSources) {
            auto [mc, nodes] = runSlotAddress(record, edgeNvlsSources, addrSlot, numSources, numDestinations);
            runAddress = mc + offset;
            run = nodes;
          }
          u32 s = 0;
          WHILE(s < numSources) {
            if (anyWaits) {
              // The load of this stage has not been issued: it waits for a source from another domain
              IF_D(isThreadZero & (ldIssued == consumed)) {
                fill(true);
              }
            }
            mbarrier_wait_parity_spin(reduceBarriers + consStage * 8, consParity);
            u32 stage = reduceStages + consStage * chunkSize;
            auto loadStage = [&]() {
              for (uint32_t v : range(numVectors)) {
                u32 off = threadIndex * 16 + 16 * blockSize * v;
                PRED(off < n) ld_shared_v4_u32_into(raw[v], stage + off);
              }
            };
            // The sources that this stage covers
            u32 step = 1;
            if (anyNvlsSources) {
              Value isRun = (s == 0) & (run != 0);
              IF(isRun) {
                profiling.enter("nvls ld");
                // The run's vectors, with multimem.ld_reduce from the multicast address (never under a PTX predicate,
                // see ptx.h: a vector past the chunk's end loads the chunk's last vector again, which is not stored).
                // The switches combine the domain's values (see nvlsReduceInst), which the threads then combine as one
                // source.
                for (uint32_t v : range(numVectors)) {
                  u32 off = threadIndex * 16 + 16 * blockSize * v;
                  u64 address = runAddress + widen(selp_u32(off, n - 16, off < n));
                  if (runInst->perVector == 1) {
                    std::array<Value, 4> x;
                    for (Value& w : x) {
                      w = Value(ValType::U32);
                    }
                    multimem_ld_reduce_relaxed_sys_v4(runInst->op, x, address);
                    for (size_t j : range(4)) {
                      raw[v][j] = x[j];
                    }
                  } else {
                    const uint32_t width = 16 / runInst->perVector;
                    for (uint32_t j : range(runInst->perVector)) {
                      u64 a = address + j * width;
                      if (runInst->wide) {
                        Value w(ValType::U64);
                        multimem_ld_reduce_relaxed_sys(runInst->op, w, a);
                        unpack_b64(raw[v][2 * j], raw[v][2 * j + 1], w);
                      } else {
                        multimem_ld_reduce_relaxed_sys(runInst->op, raw[v][j], a);
                      }
                    }
                  }
                }
              }
              ELSE {
                profiling.enter("stage");
                loadStage();
              }
              step = selp_u32(run, 1, isRun);
            } else {
              loadStage();
            }
            for (uint32_t v : range(numVectors)) {
              Vector<Value> x = ra.unpack(raw[v]);
              for (size_t j : range(ra.accPerVector)) {
                ra.combine(acc[v][j], x[j]);
              }
            }
            if (compile_op::isScaledSum(op)) {
              IF(s + step == inputSources) {
                for (auto& a : acc) {
                  for (Value& v : a) {
                    ra.scale(v, op == ReduceOp::Avg ? divisor : factor);
                  }
                }
              }
            }
            s += step;
            u32 next = consStage + 1;
            Value wrap = next == numStages;
            consStage = selp_u32(0, next, wrap);
            consParity = consParity ^ selp_u32(1, 0, wrap);
            consumed += 1;
            // Every thread has combined the stage: load the next chunk into it
            barrier_sync_unaligned();
            IF_D(isThreadZero) {
              fill(false);
            }
          }
          // The result buffer is free once the stores of the item before the previous one have read it
          profiling.enter("result");
          IF_D(isThreadZero) {
            cp_async_bulk_wait_group_read(reduceResults - 1);
          }
          barrier_sync_unaligned();
          u32 buffer = reduceResultBuffers + result * chunkSize;
          Value copy = numSources == 1;
          for (uint32_t v : range(numVectors)) {
            std::array<u32, 4> x = ra.pack(acc[v]);
            for (size_t j : range(4)) {
              x[j] = selp_u32(raw[v][j], x[j], copy);
            }
            u32 off = threadIndex * 16 + 16 * blockSize * v;
            PRED(off < n) st_shared_v4_u32(buffer + off, x);
          }
          // The bulk stores below read what the threads have written
          fence_proxy_async_shared_cta();
          barrier_sync_unaligned();
          IF_D(isThreadZero) {
            // An NVLS run of destinations: one bulk store to the multicast address, then the destinations after it
            u32 firstDestination = 0;
            if (anyNvlsDestinations) {
              auto [mc, nodes] = runSlotAddress(record, edgeNvlsDestinations, addrSlot, numSources, numDestinations);
              IF(nodes != 0) {
                profiling.enter("mc store");
                cp_async_bulk_global_shared(mc + offset, buffer, n);
              }
              firstDestination = nodes;
            }
            FOR(u32 d = firstDestination, d < numDestinations, d += 1) {
              u64 dst = ld_u64(table + widen((addrSlot + numSources + d) * 8)) + offset;
              cp_async_bulk_global_shared(dst, buffer, n);
            }
            cp_async_bulk_commit_group();
            if (anyDoneFlags) {
              // An edge with done flags (a partial result that another domain reads, or a result that this kernel
              // waits for: its own domain's partial result, a piece of two phases; an edge that has used regions of
              // rings of the scratch tensor): once all its items (of any block) are stored, the last one sets the flags
              // (per launch, every item counts once, so the counter wraps back to 0). Flags that only say that the
              // edge's sources have been loaded (edgeDoneWhenLoaded) do not wait for the stores: the item's loads have
              // completed (its stages have been combined).
              u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
              IF((flags & edgeDoneFlags) != 0) {
                IF((flags & edgeDoneWhenLoaded) == 0) {
                  cp_async_bulk_wait_group(0);
                  fence_proxy_async_global();
                }
                fence_release_gpu();
                u32 numItems = ld_global_nc_u32(record + offsetof(PipeEdge, numItems));
                u32 old = atom_global_relaxed_inc_u32(
                    *partialDone + widen((concurrencyIndex * numEdges + edge) * 4), numItems - 1);
                IF(old == numItems - 1) {
                  fence_acquire_gpu();
                  fence_release_sys();
                  u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, doneFlags));
                  u32 end = first + ld_global_nc_u32(record + offsetof(PipeEdge, numDoneFlags));
                  FOR(u32 i = first, i < end, i += 1) {
                    u32 word = ld_global_nc_u32(tables + pipeDoneFlagsOffset + widen(i * 4));
                    st_global_relaxed_sys_u32(doneFlags + widen((word >> waitGenerationBits) * 4),
                        stepValue + (word & (int64_t)((1u << waitGenerationBits) - 1)));
                  }
                }
              }
            }
          }
          // Warp 0 together again before the next item: the ptxas of CUDA 13.0 (the JIT of driver 580) can fail to
          // reconverge a warp after a branch of one thread (fixed in CUDA 13.2.2), and an aligned block barrier
          // (bar.sync) of a diverged warp is undefined (a variant of this loop whose thread 0 alone counted the items
          // after the last barrier, with aligned barriers, hung in every all-reduce of 64 MiB across 2 domains of 32
          // GB300; with this, never). The loop's barriers are unaligned now too, which a diverged warp may reach.
          warp_sync();
          result = result ^ 1;
          u32 nextSlot = consSlot + 1;
          consSlot = selp_u32(0, nextSlot, nextSlot == reduceItemSlots);
        }
        IF_D(isThreadZero) {
          cp_async_bulk_wait_group(0);
        }
        profiling.leave();
      };

      // NVLS reductions (see nvlsReduceLocal above): the warps of the first nvlsReduceBlocks blocks take the items of
      // the NVLS sequence (nvlsReduceItemBytes of an edge each, see PipeEdge::nvlsFirst) from a counter of their own,
      // one at a time (lane 0 claims the next one while the warp works on the current one; each counter of a
      // concurrency slot serves alternate launches, like the claim counters above). Lane l loads the 16-byte vectors
      // l, l + 32, ... of the item with multimem.ld_reduce from the multicast address of the edge's local source (in
      // four or two instructions for the integers and float64, whose forms have no vectors), divides them for an
      // average, and stores them with multimem.st to the multicast address of the local destination (an all-reduce's:
      // every rank receives the result), or to the local destination. The switches combine 16-bit floats in f32
      // (.acc::f32) and return their sums within 1 ulp (see nvlsReduceInst); an average (of bfloat16: not of float16,
      // see there) divides that sum (exactly with a power of 2 ranks).
      // NVLS runs (nvlsRunLoop: the reduce loop's edges with runs, e.g. those of reductions across NVLink domains) go
      // through the same loop, the items of every edge in the order of the edges (that of the reduce loop): an item
      // loads the edge's run of sources with multimem.ld_reduce, if it has one (a run waits for nothing), then passes
      // the edge's waits, if the warp has not yet (the reduce loop's: the destination waits, then the sources'; every
      // lane polls them, then fences), loads the edge's other sources with plain loads (they are this rank's: partial
      // results that the CPU thread has read from other domains, items that this kernel has computed), the first of
      // them before it combines the run, and combines them as the reduce loop does (in the edge's order from the
      // identity, the run as one source, an average divided and PremulSum multiplied after the inputs, 16-bit floats in
      // f32 rounded once: the same bits), then stores the result with a multicast store to the run of destinations, if
      // any, and to the other destinations (this rank's). Edges with done flags: a warp counts its items of an edge,
      // and adds its count to the edge's counter (lane 0, after a release fence, which the warp's barrier orders its
      // lanes' loads and stores before) when it goes on to an item of another edge, before that edge's waits, and at
      // the end; the warp whose add completes the count sets the flags, as in the reduce loop. So a warp never waits
      // holding a count, every wait is for an edge earlier in the order (or for a transfer of one, see the reduce
      // loop), and a warp has one item at a time: all warps make progress. The warps of a block go on independently,
      // where in the reduce loop a block waits for the slowest vector of its 32 KiB chunk, and thread 0 for its bulk
      // stores to complete before it counts an item with done flags, which the block's next chunk then waits for
      // (reduce-scatter of 13 MiB per rank across 2 domains of 32 GB300: 8-15 us per item of the partial results for
      // the other domain, a quarter of the kernel; 1207 us with the 24 blocks of the tuned grid, 1526 us with 8).
      auto nvlsReduceLoop = [&]() {
        profiling.enter("nvls reduce");
        using compile_op::ReduceOp;
        const DType dtype = graph.tensors.at(0).dtype;
        const NvlsReduceInst inst = nvlsReduceInst(dtype, graph.reduce).value();
        // Avg divides the switches' sums, PremulSum multiplies them by the launch's factor
        const bool avg = graph.reduce == ReduceOp::Avg;
        const bool scaled = compile_op::isScaledSum(graph.reduce);
        CHECK(!scaled || dtype == DType::Float32 || dtype == DType::Float64 || dtype == DType::BFloat16);
        // The stores the edges need: multicast stores (edgeNvlsStore), stores to the local destination, or both
        bool anyMulticast = false;
        bool anyLocalStore = false;
        for (const PipeEdge& p : pipeEdges) {
          if (p.flags & edgeNvls) {
            ((p.flags & edgeNvlsStore) ? anyMulticast : anyLocalStore) = true;
          }
        }
        u64 pipeEdgeTable = tables + pipeEdgeTableOffset;

        // An item of an NVLS reduction (nvlsReduceLocal)
        auto reductionItem = [&](const u32& item, const u64& record) {
          u32 bytes = ld_global_nc_u32(record + offsetof(PipeEdge, bytes));
          u32 addrSlot = ld_global_nc_u32(record + offsetof(PipeEdge, addrSlot));
          u32 numSources = ld_global_nc_u32(record + offsetof(PipeEdge, numSources));
          u32 numDestinations = ld_global_nc_u32(record + offsetof(PipeEdge, numDestinations));
          u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
          u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
          u32 nvlsSlot = addrSlot + numSources + numDestinations;
          u64 source = ld_u64(table + widen(nvlsSlot * 8));
          Value multicast = (flags & edgeNvlsStore) != 0;
          u64 destination = ld_u64(table + widen(selp_u32(nvlsSlot + 1, addrSlot + numSources, multicast) * 8));
          u32 base = (item - first) * nvlsReduceItemBytes + laneIndex * 16;
          Vector<Value> valid;
          Vector<u64> offsets;
          Vector<std::array<Value, 4>> x(nvlsReduceVectors);
          Vector<std::array<Value, 2>> wide(nvlsReduceVectors); // (64-bit types)
          for (uint32_t v : range(nvlsReduceVectors)) {
            u32 offset = base + 512 * v;
            valid.push_back(offset < bytes);
            offsets.push_back(widen(offset));
            // A vector past the edge's end loads the edge's last vector again (and stores nothing): no multimem
            // instruction under a PTX predicate (see the stores below)
            u32 loadOffset = selp_u32(offset, bytes - 16, valid.back());
            u64 address = source + widen(loadOffset);
            for (Value& w : x[v]) {
              w = Value(ValType::U32);
            }
            if (inst.perVector == 1) {
              multimem_ld_reduce_relaxed_sys_v4(inst.op, x[v], address);
            } else {
              const uint32_t width = 16 / inst.perVector;
              for (uint32_t j : range(inst.perVector)) {
                u64 a = address + j * width;
                if (inst.wide) {
                  wide[v][j] = Value(ValType::U64);
                  multimem_ld_reduce_relaxed_sys(inst.op, wide[v][j], a);
                } else {
                  multimem_ld_reduce_relaxed_sys(inst.op, x[v][j], a);
                }
              }
            }
          }
          Value divisor;
          if (avg) {
            divisor = Value(dtype == DType::Float64 ? ValType::U64 : ValType::U32);
            arith(dtype == DType::Float64 ? "cvt.rn.f64.u32" : "cvt.rn.f32.u32", divisor,
                ld_global_nc_u32(record + offsetof(PipeEdge, divisor)));
          } else if (scaled) {
            divisor = launchScale(dtype == DType::Float64);
          }
          // (divisor: the factor for PremulSum)
          const char* const scale32 = avg ? "div.rn.f32" : "mul.rn.f32";
          for (uint32_t v : range(nvlsReduceVectors)) {
            if (scaled && dtype == DType::Float64) {
              for (Value& w : wide[v]) {
                arith(avg ? "div.rn.f64" : "mul.rn.f64", w, w, divisor);
              }
            } else if (scaled) {
              for (Value& w : x[v]) {
                if (dtype == DType::Float32) {
                  arith(scale32, w, w, divisor);
                  continue;
                }
                // A pair of bfloat16 (cvt.rn.bf16x2.f32 d, a, b puts a in the upper half)
                Value lo(ValType::U32);
                Value hi(ValType::U32);
                lo = w << 16;
                hi = w & 0xffff0000ll;
                arith(scale32, lo, lo, divisor);
                arith(scale32, hi, hi, divisor);
                arith("cvt.rn.bf16x2.f32", w, hi, lo);
              }
            }
            if (inst.wide) {
              unpack_b64(x[v][0], x[v][1], wide[v][0]);
              unpack_b64(x[v][2], x[v][3], wide[v][1]);
            }
          }
          // The multicast store under a branch: the ptxas of CUDA 13.0 (the JIT of driver 580) drops the predicate
          // of a multimem.st (and multimem.red) under a PTX predicate, storing every vector (past the end of an
          // edge too: an all-reduce of 64 KiB on 32 GB300 overwrote other pieces on every rank, a reduce-scatter
          // wrote past its outputs); a branch, which it turns into a predicated store itself, works (ptxas of
          // CUDA 13.2 keeps both). (Only the stores that some edge needs: no multimem.st without multicast stores.)
          for (uint32_t v : range(nvlsReduceVectors)) {
            u64 address = destination + offsets[v];
            if (anyMulticast) {
              IF_D(valid[v] & multicast) {
                multimem_st_relaxed_sys_v4(address, x[v]);
              }
            }
            if (anyLocalStore) {
              Value toLocal = valid[v] & !multicast;
              PRED(toLocal) st_global_v4_b32(address, x[v]);
            }
          }
        };

        // NVLS runs (nvlsRunLoop): what the edges have (the parts that none has are left out), the nodes of a run, the
        // accumulators' arithmetic, the signals and done flags of the waits (see the reduce loop), and per concurrency
        // slot and edge, the items of the edge with done flags counted so far
        bool allSourceRuns = true;
        bool anyOtherSources = false;
        bool anyDestinationRuns = false;
        bool anyWaits = false;
        bool anyDoneFlags = false;
        if (nvlsRunLoop) {
          for (size_t i : indices(localCopies)) {
            const PipeEdge& p = pipeEdges.at(i);
            const bool sourceRun = (p.flags & edgeNvlsSources) != 0;
            allSourceRuns &= sourceRun;
            anyOtherSources |= !sourceRun || p.numSources > domain.size();
            anyDestinationRuns |= (p.flags & edgeNvlsDestinations) != 0;
            anyWaits |= (p.flags & (edgeSourceWaits | edgeDestinationWaits)) != 0;
            anyDoneFlags |= (p.flags & edgeDoneFlags) != 0;
          }
        }
        const uint32_t runNodes = domain.size();
        std::optional<ReduceArith> ra;
        std::optional<u64> signalFlags;
        std::optional<u64> doneFlags;
        std::optional<u64> runDone;
        if (nvlsRunLoop) {
          ra.emplace(dtype, graph.reduce);
          if (anyWaits || anyDoneFlags) {
            signalFlags = concurrency(group->rdmaSignalBuffer);
            doneFlags = concurrency(group->cpuInBuffer) + 4 * 16;
          }
          if (anyDoneFlags) {
            runDone = addGlobalVar(4, 4 * maxConcurrency * numEdges, "nvlsrundone");
          }
        }
        // This warp's state (NVLS runs): the last edge whose waits it has passed, and the edge whose items it has
        // counted, with the count it has not added to the edge's counter yet
        std::optional<u32> passedEdge;
        std::optional<u32> countedEdge;
        std::optional<u32> counted;

        // Wait until the word of the wait table (see waitWord) has passed: every lane loads the signal or done flag
        // (the same word), so that every lane has seen it before its acquire fence
        auto waitFor = [&](const u32& word) {
          Value done = word >= (int64_t)waitDoneFlag;
          u32 index = ((word >> waitGenerationBits) & (int64_t)((waitDoneFlag >> waitGenerationBits) - 1)) - 1;
          u32 target = stepValue + (word & (int64_t)((1u << waitGenerationBits) - 1));
          u64 flag =
              selp_u64(*doneFlags + widen(index * 4), *signalFlags + widen(index * Group::rdmaSignalBytes), done);
          static_assert(Group::rdmaSignalBytes == 16 && maxDevices == 4);
          auto load = [&]() {
            u32 v = predicated(done, ld_global_relaxed_sys_u32)(flag);
            std::array<Value, 4> words;
            PRED(!done) ld_global_relaxed_sys_v4_u32(words, flag);
            return selp_u32(v, min_u32(min_u32(words[0], words[1]), min_u32(words[2], words[3])), done);
          };
          u32 value = load();
          WHILE(vote_sync_any(value < target)) {
            value = load();
          }
        };
        // Pass the waits of the edge at record: its destination waits (the regions of rings of the scratch tensor
        // that it stores to are free), then its sources' (they have arrived, or been stored)
        auto passWaits = [&](const u64& record, const u32& flags, const u32& numSources) {
          profiling.enter("nvls wait");
          IF((flags & edgeDestinationWaits) != 0) {
            u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, destWaits));
            u32 end = first + ld_global_nc_u32(record + offsetof(PipeEdge, numDestWaits));
            FOR(u32 i = first, i < end, i += 1) {
              waitFor(ld_global_nc_u32(tables + pipeWaitsOffset + widen(i * 4)));
            }
          }
          IF((flags & edgeSourceWaits) != 0) {
            u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, waits));
            FOR(u32 s = 0, s < numSources, s += 1) {
              u32 word = ld_global_nc_u32(tables + pipeWaitsOffset + widen((first + s) * 4));
              IF(word != 0) {
                waitFor(word);
              }
            }
          }
          // The data arrived (or was stored) before the signal, or the region that the edge stores to has been read
          // (or loaded) before it
          IF((flags & (edgeDestinationWaits | edgeSourceWaits)) != 0) {
            fence_acquire_sys();
          }
          profiling.enter("nvls reduce");
        };
        // Add this warp's count of the items of countedEdge to the edge's counter, and if that completes the count, set
        // the edge's done flags (see the reduce loop). The warp barrier orders the lanes' loads and stores of the items
        // before lane 0's release fence; the counter starts from 0 again for the next launch of the concurrency slot
        // (which runs after this one)
        auto flushCount = [&]() {
          IF(*counted != 0) {
            profiling.enter("nvls count");
            warp_sync();
            IF_D(laneIndex == 0) {
              u64 record = pipeEdgeTable + widen(*countedEdge * sizeof(PipeEdge));
              u32 items = ld_global_nc_u32(record + sizeof(PipeEdge) + offsetof(PipeEdge, nvlsFirst)) -
                          ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
              u64 counter = *runDone + widen((concurrencyIndex * numEdges + *countedEdge) * 4);
              fence_release_gpu();
              Value old = atom_global_relaxed_gpu_add_u32(counter, *counted);
              IF(old + *counted == items) {
                st_global_u32(counter, Value(0));
                fence_acquire_gpu();
                fence_release_sys();
                u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, doneFlags));
                u32 end = first + ld_global_nc_u32(record + offsetof(PipeEdge, numDoneFlags));
                FOR(u32 i = first, i < end, i += 1) {
                  u32 word = ld_global_nc_u32(tables + pipeDoneFlagsOffset + widen(i * 4));
                  st_global_relaxed_sys_u32(*doneFlags + widen((word >> waitGenerationBits) * 4),
                      stepValue + (word & (int64_t)((1u << waitGenerationBits) - 1)));
                }
              }
            }
            warp_sync();
            *counted = 0;
            profiling.enter("nvls reduce");
          }
        };
        // An item of an edge with NVLS runs (nvlsRunLoop)
        auto runItem = [&](const u32& item, const u32& edge, const u64& record) {
          const bool acc64 = ra->acc64;
          u32 bytes = ld_global_nc_u32(record + offsetof(PipeEdge, bytes));
          u32 addrSlot = ld_global_nc_u32(record + offsetof(PipeEdge, addrSlot));
          u32 numSources = ld_global_nc_u32(record + offsetof(PipeEdge, numSources));
          u32 numDestinations = ld_global_nc_u32(record + offsetof(PipeEdge, numDestinations));
          u32 flags = ld_global_nc_u32(record + offsetof(PipeEdge, flags));
          u32 first = ld_global_nc_u32(record + offsetof(PipeEdge, nvlsFirst));
          // The count of the previous edge, before this one's waits
          if (anyDoneFlags) {
            IF(edge != *countedEdge) {
              flushCount();
            }
          }
          u32 base = (item - first) * nvlsReduceItemBytes + laneIndex * 16;
          Vector<Value> valid;
          Vector<u64> offsets;
          Vector<u32> loadOffsets;
          for (uint32_t v : range(nvlsReduceVectors)) {
            u32 offset = base + 512 * v;
            valid.push_back(offset < bytes);
            offsets.push_back(widen(offset));
            // A vector past the edge's end loads the edge's last vector again (and stores nothing), as above
            loadOffsets.push_back(selp_u32(offset, bytes - 16, valid.back()));
          }
          // The runs (their NVLS slots follow the destinations, the sources' first, see runSlotAddress in the reduce
          // loop), and the sources and destinations after them
          Value sourceRun = (flags & edgeNvlsSources) != 0;
          Value destinationRun = (flags & edgeNvlsDestinations) != 0;
          u32 nvlsSlot = addrSlot + numSources + numDestinations;
          u32 firstSource = selp_u32(runNodes, 0, sourceRun);
          u32 firstDestination = selp_u32(runNodes, 0, destinationRun);
          // The run of sources, with multimem.ld_reduce (as above)
          Vector<std::array<Value, 4>> x(nvlsReduceVectors);
          Vector<std::array<Value, 2>> wide(nvlsReduceVectors); // (64-bit types)
          for (uint32_t v : range(nvlsReduceVectors)) {
            for (Value& w : x[v]) {
              w = Value(ValType::U32);
            }
            if (inst.wide) {
              for (Value& w : wide[v]) {
                w = Value(ValType::U64);
              }
            }
          }
          auto loadRun = [&]() {
            u64 source = ld_u64(table + widen(nvlsSlot * 8));
            for (uint32_t v : range(nvlsReduceVectors)) {
              u64 address = source + widen(loadOffsets[v]);
              if (inst.perVector == 1) {
                multimem_ld_reduce_relaxed_sys_v4(inst.op, x[v], address);
              } else {
                const uint32_t width = 16 / inst.perVector;
                for (uint32_t j : range(inst.perVector)) {
                  u64 a = address + j * width;
                  if (inst.wide) {
                    multimem_ld_reduce_relaxed_sys(inst.op, wide[v][j], a);
                  } else {
                    multimem_ld_reduce_relaxed_sys(inst.op, x[v][j], a);
                  }
                }
              }
            }
          };
          if (allSourceRuns) {
            loadRun();
          } else {
            IF(sourceRun) {
              loadRun();
            }
          }
          if (anyWaits) {
            IF(edge != *passedEdge) {
              passWaits(record, flags, numSources);
              *passedEdge = edge;
            }
          }
          // The accumulators, from the identity; an average divides by the edge's divisor, PremulSum multiplies by the
          // launch's factor, once the inputs among the sources are combined (PipeEdge::inputSources)
          Vector<Vector<Value>> acc(nvlsReduceVectors);
          for (auto& a : acc) {
            for (size_t j : range(ra->accPerVector)) {
              a.emplace_back(acc64 ? ValType::U64 : ValType::U32);
              a.back() = ra->identity;
            }
          }
          u32 inputSources;
          Value by(acc64 ? ValType::U64 : ValType::U32);
          if (avg) {
            inputSources = ld_global_nc_u32(record + offsetof(PipeEdge, inputSources));
            arith(acc64 ? "cvt.rn.f64.u32" : "cvt.rn.f32.u32", by,
                ld_global_nc_u32(record + offsetof(PipeEdge, divisor)));
          } else if (scaled) {
            inputSources = ld_global_nc_u32(record + offsetof(PipeEdge, inputSources));
            by = launchScale(acc64);
          }
          auto scaleAfter = [&](const Value& sources) {
            if (scaled) {
              IF(sources == inputSources) {
                for (auto& a : acc) {
                  for (Value& v : a) {
                    ra->scale(v, by);
                  }
                }
              }
            }
          };
          auto combineVectors = [&](const Vector<std::array<u32, 4>>& words) {
            for (uint32_t v : range(nvlsReduceVectors)) {
              Vector<Value> values = ra->unpack(words[v]);
              for (size_t j : range(ra->accPerVector)) {
                ra->combine(acc[v][j], values[j]);
              }
            }
          };
          // The other sources' vectors (the first one's loaded before the run is combined)
          Vector<std::array<u32, 4>> y(nvlsReduceVectors);
          auto loadSource = [&](const u32& s) {
            u64 address = ld_u64(table + widen((addrSlot + s) * 8));
            for (uint32_t v : range(nvlsReduceVectors)) {
              std::array<Value, 4> w;
              for (Value& r : w) {
                r = Value(ValType::U32);
              }
              ld_v4_u32(w, address + widen(loadOffsets[v]));
              for (size_t j : range(4)) {
                y[v][j] = w[j];
              }
            }
          };
          if (anyOtherSources) {
            IF(firstSource < numSources) {
              loadSource(firstSource);
            }
          }
          auto combineRun = [&]() {
            Vector<std::array<u32, 4>> words(nvlsReduceVectors);
            for (uint32_t v : range(nvlsReduceVectors)) {
              if (inst.wide) {
                unpack_b64(words[v][0], words[v][1], wide[v][0]);
                unpack_b64(words[v][2], words[v][3], wide[v][1]);
              } else {
                for (size_t j : range(4)) {
                  words[v][j] = x[v][j];
                }
              }
            }
            combineVectors(words);
            scaleAfter(firstSource);
          };
          if (allSourceRuns) {
            combineRun();
          } else {
            IF(sourceRun) {
              combineRun();
            }
          }
          if (anyOtherSources) {
            FOR(u32 s = firstSource, s < numSources, s += 1) {
              IF(s != firstSource) {
                loadSource(s);
              }
              combineVectors(y);
              scaleAfter(s + 1);
            }
          }
          // The result (an edge with one source, a copy, as loaded, see the reduce loop): one multicast store to the
          // run of destinations (under a branch, see the NVLS reductions above), then a store to each other destination
          Vector<std::array<Value, 4>> result(nvlsReduceVectors);
          Value copy = numSources == 1;
          for (uint32_t v : range(nvlsReduceVectors)) {
            std::array<u32, 4> r = ra->pack(acc[v]);
            for (size_t j : range(4)) {
              result[v][j] = anyOtherSources ? selp_u32(y[v][j], r[j], copy) : r[j];
            }
          }
          if (anyDestinationRuns) {
            u64 multicast = ld_u64(table + widen((nvlsSlot + selp_u32(1, 0, sourceRun)) * 8));
            for (uint32_t v : range(nvlsReduceVectors)) {
              IF_D(valid[v] & destinationRun) {
                multimem_st_relaxed_sys_v4(multicast + offsets[v], result[v]);
              }
            }
          }
          FOR(u32 d = firstDestination, d < numDestinations, d += 1) {
            u64 address = ld_u64(table + widen((addrSlot + numSources + d) * 8));
            for (uint32_t v : range(nvlsReduceVectors)) {
              PRED(valid[v]) st_global_v4_b32(address + offsets[v], result[v]);
            }
          }
          if (anyDoneFlags) {
            IF((flags & edgeDoneFlags) != 0) {
              *countedEdge = edge;
              *counted += 1;
            }
          }
        };

        IF(blockIndex < nvlsReduceBlocks) {
          Value isLaneZero = laneIndex == 0;
          u32 claim;
          PRED(isLaneZero) atom_global_relaxed_gpu_add_u32_into(claim, *nvlsClaimCounter, 1);
          u32 item = shfl_sync_idx_b32(claim, 0, 0x1f);
          // The edge of the item, and the first item of the edge after it (items only increase)
          u32 edge = 0;
          u32 nextFirst = ld_global_nc_u32(pipeEdgeTable + sizeof(PipeEdge) + offsetof(PipeEdge, nvlsFirst));
          if (nvlsRunLoop) {
            passedEdge.emplace(0xffffffffll);
            countedEdge.emplace(0xffffffffll);
            counted.emplace(0);
          }
          WHILE(item < nvlsReduceItems) {
            PRED(isLaneZero) atom_global_relaxed_gpu_add_u32_into(claim, *nvlsClaimCounter, 1);
            WHILE(item >= nextFirst) {
              edge += 1;
              nextFirst = ld_global_nc_u32(
                  pipeEdgeTable + widen((edge + 1) * sizeof(PipeEdge)) + offsetof(PipeEdge, nvlsFirst));
            }
            u64 record = pipeEdgeTable + widen(edge * sizeof(PipeEdge));
            if (nvlsRunLoop) {
              runItem(item, edge, record);
            } else {
              reductionItem(item, record);
            }
            item = shfl_sync_idx_b32(claim, 0, 0x1f);
          }
          if (anyDoneFlags) {
            flushCount();
          }
        }
        profiling.leave();
      };

      if (reduceKernel) {
        CHECK(!nvlsLocal && !anySignals);
        if (nvlsReduceLocal || nvlsRunLoop) {
          IF(nvlsMode) {
            nvlsReduceLoop();
          }
          ELSE {
            reduceLoop();
          }
        } else {
          reduceLoop();
        }
      } else if (nvlsLocal) {
        IF(nvlsMode) {
          IF_D(isPipelineLane & (blockIndex == 0)) {
            nvlsCopy();
          }
        }
        ELSE {
          unicastCopy();
        }
      } else {
        unicastCopy();
      }
      // All copies done before the exit's release fence (unaligned: after the branches of thread 0 or of the pipelines'
      // lanes, see the reduce loop)
      profiling.enter("copy end barrier");
      barrier_sync_unaligned();
      profiling.leave();
    } else {
      // Rounds: warp 0 scans the edges that are not done, 32 at a time (one per lane). Relay edges whose signal
      // has arrived are finished on the spot. The first ready copy edge is selected; after a block barrier, the
      // whole block copies it and thread 0 marks it done. The selection slot alternates between two words, so
      // that warp 0 can write the next round's while other warps still read this round's.
      static constexpr uint32_t selWait = 0xffffffff; // nothing ready: scan again
      static constexpr uint32_t selDone = 0xfffffffe; // all edges done
      u32 doneAddr = addShared32(4, 4 * numDoneWords, "done");
      u32 selAddr = addShared32(4, 8, "sel");
      for (uint32_t i : range((numDoneWords + blockSize - 1) / blockSize)) {
        u32 w = blockSize * i + threadIndex;
        PRED(w < numDoneWords) st_shared_u32(doneAddr + 4 * w, 0);
      }
      barrier_sync(14);

      u32 round = 0;
      WHILE(true) {
        u32 selSlot = selAddr + 4 * (round & 1);
        IF(warpIndex == 0) {
          warp_sync();
          u32 found = selWait;
          u32 pending = 0;
          FOR(u32 w = 0, w < numDoneWords, w += 1) {
            u32 doneWord = ld_shared_u32(doneAddr + 4 * w);
            u32 e = 32 * w + laneIndex;
            Value notDone = (e < numEdges) & (((doneWord >> laneIndex) & 1) == 0);
            u64 desc = edgeTableAddr + widen(e * edgeStride);
            u32 flags = predicated(notDone, ld_global_nc_u32)(desc);
            u32 signal = predicated(notDone, ld_global_nc_u32)(desc + 4);
            u32 sourceDep = predicated(notDone, ld_global_nc_u32)(desc + 8);
            Value relay = notDone & ((flags & edgeRelay) != 0);
            Value depWait = notDone & ((flags & edgeWait) != 0) & !relay;
            u32 value = stepValue;
            u64 depAddr = concurrency(depFilled.local) + widen(4 * (sourceDep + localNodes.size() * blockIndex));
            value = selp(predicated(depWait, ld_global_relaxed_sys_u32)(depAddr), value, depWait);
            if (!tuning) {
              // (the signal's words, one per NIC, see writeRdmaSignal in cputhread.cc)
              u64 flagAddr = concurrency(group->rdmaSignalBuffer) + widen(signal * Group::rdmaSignalBytes);
              std::array<Value, 4> words;
              PRED(relay) ld_global_relaxed_sys_v4_u32(words, flagAddr);
              u32 signalValue = min_u32(min_u32(words[0], words[1]), min_u32(words[2], words[3]));
              value = selp(signalValue, value, relay);
            }
            Value ready = notDone & (value >= stepValue);
            Value relayReady = ready & relay;
            u32 relayBits = vote_sync_ballot(relayReady);
            u32 copyBits = vote_sync_ballot(ready & !relay);
            pending |= vote_sync_ballot(notDone) & ~relayBits;
            // A relay edge whose signal has arrived: its node is filled; the lane that has seen the signal
            // passes it on
            IF_D(relayReady) {
              signalEdge(desc);
            }
            IF_D((laneIndex == 0) & (relayBits != 0)) {
              st_shared_u32(doneAddr + 4 * w, doneWord | relayBits);
            }
            IF(copyBits != 0) {
              found = 32 * w + lowest_bit_b32(copyBits);
              BREAK;
            }
          }
          // The lane that has seen the selected edge's source filled orders its load before the copy, which
          // the block barrier below extends to all threads
          IF(found != selWait) {
            fence_acquire_sys();
          }
          IF_D(laneIndex == 0) {
            u32 waitValue = selWait;
            u32 doneValue = selDone;
            st_shared_u32(selSlot, selp(found, selp(waitValue, doneValue, pending != 0), found != selWait));
          }
        }
        barrier_sync(14);
        u32 sel = ld_shared_u32(selSlot);
        round += 1;
        IF(sel == selDone) {
          BREAK;
        }
        IF(sel != selWait) {
          u64 desc = edgeTableAddr + widen(sel * edgeStride);
          EdgeWords w;
          loadEdge(w, desc, nullptr);
          u64 srcaddr = resolveEdge(w);
          copyEdge(desc, w.flags, w.bytes, srcaddr, w.numDestinations);
          IF_D(isThreadZero) {
            u32 wordAddr = doneAddr + 4 * (sel / 32);
            u32 one = 1;
            st_shared_u32(wordAddr, ld_shared_u32(wordAddr) | (one << (sel % 32)));
          }
        }
      }
    }

    auto execSync = [&](uint32_t stepOffset) {
      Vector<u64> ldaddrs;
      Vector<u64> staddrs;
      Vector<Value> preds;
      uint32_t numWarps = blockSize / 32;
      for (uint32_t i : range((syncsetup.size() + blockSize - 1) / blockSize)) {
        profiling.enter("sync setup");
        ldaddrs.emplace_back();
        staddrs.emplace_back();
        preds.emplace_back();
        u32 index = blockSize * i + (warpIndex + numWarps - 1) % numWarps + laneIndex * (blockSize / 32);
        u64& ldaddr = ldaddrs.back();
        u64& staddr = staddrs.back();
        Value& pred = preds.back();
        pred = index < syncsetup.size();
        PRED(pred) {
          u32 addr = csyncaddr + index * sizeof(SyncSetup);
          u64 stbase = ld_const_u64(addr);
          u32 itembytes = ld_const_u32(addr + 8);
          u32 ldoffset = ld_const_u32(addr + 12);
          u32 stBlockStride = ld_const_u32(addr + 16);
          staddr = stbase + widen(itembytes * concurrencyIndex + stBlockStride * blockIndex);
          ldaddr = concurrency(syncAddrs.local) + widen(ldoffset + sizeof(uint32_t) * ranks.size() * blockIndex);
        }
      }
      u32 targetValue = stepValue + stepOffset;
      profiling.enter("exit sync barrier");
      // (unaligned: after thread 0's release fence, see the reduce loop)
      barrier_sync_unaligned();
      profiling.enter("exit sync barrier wait");
      for (size_t i : indices(staddrs)) {
        profiling.enter("sync st");
        PRED(preds[i]) st_global_relaxed_sys_u32(staddrs[i], targetValue);
      }
      for (size_t i : indices(ldaddrs)) {
        profiling.enter("sync ld");
        u64 ldaddr = ldaddrs[i];
        Value pred = preds[i];
        Label loop;
        Label done;
        GOTO_IF_NOT(pred, done);
        LABEL(loop);
        pred = ld_global_relaxed_sys_u32(ldaddr) < targetValue;
        GOTO_IF_NOT(pred, done);
        nanosleep(10);
        GOTO(loop);
        LABEL(done);
      }
    };

    profiling.enter("exit sync entry");
    if (hasRemoteWrites) {
      IF_D(threadIndex == 0) {
        profiling.enter("fence.release.sys");
        fence_release_sys();
      }
    }
    execSync(1);

    profiling.leave();
    profiling.finish();
    rv->profilingNames.resize(profiling.nextIndex);
    for (auto& v : profiling.nameToIndex) {
      rv->profilingNames.at(v.second) = v.first;
    }
    ret();
  }

  // compile_op_params (see paramBuffer above): copies the first `words` words of its second parameter to the copy of
  // concurrency slot `slot` in the parameter buffer, from word `first` of the copy on
  if (paramBuffer) {
    auto* pf = mod.newFunction("compile_op_params");
    FunctionScope pfScope(pf);
    pf->maxThreads = paramFillBlockSize;
    pf->addParamBytes(8, 16); // slot, first, words (uint32_t each)
    pf->addParamBytes(8, 8 * paramFillWords);
    activateNewBlock("entry");
    u32 header = paramBase(0);
    u32 slot = loadParamField(header, 0, ValType::U32);
    u32 first = loadParamField(header, 4, ValType::U32);
    u32 words = loadParamField(header, 8, ValType::U32);
    u32 i = blockIdx_x() * paramFillBlockSize + threadIdx_x();
    IF_D(i < words) {
      u64 v = ld_param_u64(paramBase(1) + i * 8, 0);
      st_global_u64(globalAddr("global_params") + widen(slot * rv->paramBufferStride + (first + i) * 8), v);
    }
    ret();
  }

  std::string ptx = mod.finalize();
  rv->ptx = ptx;
  return rv;
}

void fillCompileOpParamBuffer(const KernelHandle& kh, const CompiledKernel& kernel, CUstream stream, uint32_t slot,
    std::span<const uintptr_t> tensorAddrs, std::span<const uintptr_t> mappedAddrs,
    std::span<const uint64_t> nvlsTensors) {
  CHECK(kh.paramBuffer && kernel.paramsFunction != nullptr && slot < maxConcurrency);
  CHECK(8 * tensorAddrs.size() <= kh.paramMappingsOffset);
  CHECK(kh.paramMappingsOffset + 8 * mappedAddrs.size() <= kh.paramNvlsOffset);
  CHECK(kh.paramNvlsOffset + 8 * nvlsTensors.size() <= kh.paramBufferStride);
  std::vector<uint64_t> words(kh.paramBufferStride / 8);
  std::ranges::copy(tensorAddrs, words.begin());
  std::ranges::copy(mappedAddrs, words.begin() + kh.paramMappingsOffset / 8);
  std::ranges::copy(nvlsTensors, words.begin() + kh.paramNvlsOffset / 8);
  // (the launch copies its parameters, so the arrays can be reused at once)
  struct Header {
    uint32_t slot;
    uint32_t first;
    uint32_t words;
    uint32_t pad;
  };
  static_assert(sizeof(Header) == 16);
  std::vector<uint64_t> data(paramFillWords);
  for (size_t first = 0; first < words.size(); first += paramFillWords) {
    Header header = {slot, (uint32_t)first, (uint32_t)std::min<size_t>(paramFillWords, words.size() - first), 0};
    std::copy_n(words.begin() + first, header.words, data.begin());
    std::array<void*, 2> args = {&header, data.data()};
    CHECK_CU(cuLaunchKernel(kernel.paramsFunction, (header.words + paramFillBlockSize - 1) / paramFillBlockSize, 1, 1,
        paramFillBlockSize, 1, 1, 0, stream, args.data(), nullptr));
  }
}

void launchCompileOpKernel(const KernelHandle& kh, const CompiledKernel& kernel, const KernelConfig& config,
    CUstream stream, const CompileOpCopyParameters& params, std::span<const uintptr_t> tensorAddrs,
    std::span<const uintptr_t> mappedAddrs, std::span<const uint64_t> nvlsTensors, bool fill) {
  if (kh.paramBuffer && fill) {
    fillCompileOpParamBuffer(kh, kernel, stream, params.concurrencyIndex, tensorAddrs, mappedAddrs, nvlsTensors);
  }
  // (with the parameter buffer, the kernel takes params only; without NVLS, it takes no NVLS words)
  CompileOpCopyParameters p = params;
  std::array<void*, 4> kparams = {&p, (void*)tensorAddrs.data(), (void*)mappedAddrs.data(), (void*)nvlsTensors.data()};
  CUlaunchConfig launchConfig;
  CUlaunchAttribute attr;
  std::memset(&launchConfig, 0, sizeof(launchConfig));
  launchConfig.grid = {config.gridSize, 1, 1};
  launchConfig.block = {config.blockSize, 1, 1};
  launchConfig.sharedMemBytes = 0;
  launchConfig.hStream = stream;
  launchConfig.attrs = &attr;
  launchConfig.numAttrs = 1;
  attr.id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
  attr.value.value = CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE;
  CHECK_CU(cuLaunchKernelEx(&launchConfig, kernel.function, kparams.data(), nullptr));
}

} // namespace moodist
