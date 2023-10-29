#include "reduce_scatter.h"
#include "allgather.h"
#include "common.h"
#include "group.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include <vector>

namespace moodist {

ReduceScatter::ReduceScatter(Group* group) : CollectiveBase(group) {}

ReduceScatter::~ReduceScatter() {}

void ReduceScatter::init() {
  sendRanks = group->allGather->sendRanks;
  recvRanks = group->allGather->recvRanks;

  CHECK(recvRanks.size() == sendRanks.size());
  auto allSendRanks = group->setupComms->allgather(sendRanks);
  for (auto& v : allSendRanks) {
    if (&v - allSendRanks.data() == rank) {
      continue;
    }
    CHECK(v.size() == sendRanks.size());
    for (size_t i = 0; i != sendRanks.size(); ++i) {
      CHECK(v[i] != sendRanks[i]);
    }
  }
}

std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> ReduceScatter::generate() {

  // allgatherBlocksize = group->ipcRanks.size() == group->size - 1 ? 256 : 64;
  // allgatherGridsize = group->ipcRanks.size() == group->size - 1 ? smCount / 4 : smCount / 4;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;
  const auto& peerIndices = group->peerIndices;

  const auto& cudaStepValue = group->cudaStepValue;
  const auto& peerCudaStepValue = group->peerCudaStepValue;
  const auto& cudaCopyDone = group->cudaCopyDone;
  const auto& peerCudaCopyDone = group->peerCudaCopyDone;
  const auto& cudaProxyReady = group->cudaProxyReady;
  const auto& peerCudaProxyReady = group->peerCudaProxyReady;

  auto waitFor32 = [&](uintptr_t address, std::string value) {
    return replace(
        R"(while (*(volatile uint32_t*)$ptr < $value);
    )",
        "$ptr", address, "$value", value);
  };

  auto waitForRecv = [&](size_t i, size_t chunkIndex) {
    std::string s;
    s = replace("// wait for recv from $i, chunk $chunkIndex\n", "$i", i, "$chunkIndex", chunkIndex);
    for (size_t di = 0; di != group->ibDevs.size(); ++di) {
      s += waitFor32(
          group->cudaCommsDeviceDataSent.cudaPointer + sizeof(uint32_t) * (i * 32 + di),
          replace("stepValue + $chunkIndex", "$chunkIndex", chunkIndex));
    }
    return s;
  };

  std::string reduceCode;

  std::unordered_map<std::string, bool> dstVisited;
  std::string prevReduceDst;
  std::string prevReduceSrc;
  auto reduceFlush = [&]() {
    if (prevReduceDst != "" && prevReduceSrc != "") {
      reduceCode +=
          replace("reduce_add<T>(reduces, $dst, $src, nullptr);\n", "$dst", prevReduceDst, "$src", prevReduceSrc);
    }
    prevReduceSrc = "";
    prevReduceDst = "";
  };
  auto reduceAdd = [&](std::string dst, std::string src) {
    std::string r;
    if (dst != prevReduceDst) {
      reduceFlush();
      if (dstVisited[dst]) {
        reduceCode += replace("reduce_add<T>(reduces, $dst, $dst, $src);\n", "$dst", dst, "$src", src);
      } else {
        prevReduceDst = dst;
        prevReduceSrc = src;
        dstVisited[dst] = true;
      }
    } else {
      if (prevReduceSrc == "") {
        reduceCode += replace("reduce_add<T>(reduces, $dst, $dst, $src);\n", "$dst", dst, "$src", src);
      } else {
        reduceCode +=
            replace("reduce_add<T>(reduces, $dst, $src1, $src2);\n", "$dst", dst, "$src1", prevReduceSrc, "$src2", src);
        prevReduceSrc = "";
      }
    }
  };

  auto addLocals = [&]() {
    reduceAdd("(T*)params.outputAddress", "(const T*)(params.inputAddress + params.bytes * $rank)");
    for (size_t peerIndex : peerIndices) {
      size_t i = ipcRanks[peerIndex];
      reduceAdd(
          "(T*)params.outputAddress", replace(
                                          "(const T*)(*(uintptr_t*)$src + params.bytes * $rank)", "$src",
                                          group->cudaPeerAddresses.cudaPointer + (sizeof(uintptr_t) * 2 * peerIndex)));
    }
  };

  if (recvRanks.empty()) {
    addLocals();
  } else {
    auto addRecv = [&](size_t n) {
      reduceCode += replace(
          R"(
        reduce_flush<T>(reduces);
        reduce_scatter_wait_for_recv_$n<<<1, 1>>>();
        )",
          "$n", n);
      reduceAdd("(T*)params.outputAddress", replace("(const T*)(params.recvAddress + params.bytes * $n)", "$n", n));
    };

    size_t n = 0;
    for (size_t i : sendRanks) {
      std::string addr = replace("(T*)(params.sendAddress + params.bytes * $n)", "$n", n);
      reduceAdd(addr, replace("(const T*)(params.inputAddress + params.bytes * $i)", "$i", i));
      for (size_t peerIndex : peerIndices) {
        reduceAdd(
            addr, replace(
                      "(const T*)(*(uintptr_t*)$src + params.bytes * $i)", "$i", i, "$src",
                      group->cudaPeerAddresses.cudaPointer + (sizeof(uintptr_t) * 2 * peerIndex)));
      }

      reduceFlush();
      reduceCode += replace(
          R"(
        reduce_flush<T>(reduces);
        reduce_scatter_send_ready<<<1, 1>>>(stepValue + $n + 1);
      )",
          "$n", n);

      if (n != 0) {
        addRecv(n - 1);
      }

      ++n;
    }
    addLocals();
    addRecv(n - 1);
  }
  reduceFlush();

  std::string waitFunctions;
  for (size_t n = 0; n != recvRanks.size(); ++n) {
    waitFunctions += replace(
        R"(
      __global__ void reduce_scatter_wait_for_recv_$n() {
        uint32_t stepValue = globalStepValue;
        $code
      }
    )",
        "$code", waitForRecv(recvRanks.at(n), Group::dataChunks - 1), "$n", n);
  }

  std::string source = replace(
      R"zz(

constexpr size_t reduceQueueSize = 16;
struct ReduceParameters {
  size_t bytes;
  const void* src1[reduceQueueSize];
  const void* src2[reduceQueueSize];
  void* dst[reduceQueueSize];
  uint32_t n;
};

template<typename T>
__global__ void reduce_kernel(ReduceParameters params) {
  assert(params.n > 0);
  assert(params.n <= reduceQueueSize);
#pragma unroll 1
  for (uint32_t i = 0; i != params.n; ++i) {
    syncthreads();
    if (params.src1[i] == params.dst[i]) {
      reduce_sum((T*)params.dst[i], (const T*)params.src2[i], params.bytes / sizeof(T));
    } else if (params.src2[i] != nullptr) {
      reduce2_sum((T*)params.dst[i], (const T*)params.src1[i], (const T*)params.src2[i], params.bytes / sizeof(T));
    } else {
      copy_impl(params.dst[i], params.src1[i], params.bytes);
    }
  }
}

__global__ void reduce_scatter_send_ready(uint32_t value) {
  $cpuIn[1] = value;
}

$waitFunctions

template<typename T>
__device__ void reduce_flush(ReduceParameters& params) {
  if (params.n) {
    // work around compiler bug
    ReduceParameters params2;
    memcpy(&params2, &params, sizeof(params2));
    reduce_kernel<T><<<$gridSize, $blockSize>>>(params2);
    params.n = 0;
  }
}

template<typename T>
__device__ void reduce_add(ReduceParameters& params, void* dst, const void* src1, const void* src2) {
  params.dst[params.n] = dst;
  params.src1[params.n] = src1;
  params.src2[params.n] = src2;
  ++params.n;
  if (params.n == reduceQueueSize) {
    reduce_flush<T>(params);
  }
}

struct ReduceScatterParameters {
  size_t bytes;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  uintptr_t peerInputAddresses[8];
  uintptr_t peerOutputAddresses[8];
  uintptr_t sendAddress;
  uintptr_t recvAddress;
};

extern "C" __global__ void reduce_scatter_local_exit() {
  generic_exit_local();
}

__device__ void reduce_scatter_impl(ReduceScatterParameters& params) {
  [[maybe_unused]] const uint32_t stepValue = globalStepValue;
  ReduceParameters reduces;
  reduces.bytes = params.bytes;
  reduces.n = 0;

  using T = float;

  $reduceCode

  reduce_flush<T>(reduces);

  reduce_scatter_local_exit<<<1, 1>>>();
}

extern "C" __global__ void reduce_scatter_local(ReduceScatterParameters params) {
  generic_entry_local(params);
  reduce_scatter_impl(params);
}

extern "C" __global__ void reduce_scatter_exit() {
  generic_exit();
}

extern "C" __global__ void reduce_scatter(ReduceScatterParameters params) {
  generic_entry(params);
  reduce_scatter_impl(params);
}

  )zz",
      "$reduceCode", reduceCode, "$waitFunctions", waitFunctions);

  std::vector<std::pair<CUfunction*, std::string>> functions;

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  // fn(cuLocalReduceScatterEntry, "local_reduce_scatter_entry");
  // fn(cuLocalReduceScatterExit, "local_reduce_scatter_exit");
  // fn(cuReduceScatterEntry, "reduce_scatter_entry");
  // fn(cuReduceScatterExit, "reduce_scatter_exit");

  // cuSendReady.resize(sendRanks.size());
  // for (size_t n = 0; n != sendRanks.size(); ++n) {
  //   fn(cuSendReady[n], replace("reduce_scatter_send_ready_$n", "$n", n));
  // }
  // cuWaitForRecv.resize(recvRanks.size());
  // for (size_t n = 0; n != recvRanks.size(); ++n) {
  //   size_t i = recvRanks[n];
  //   fn(cuWaitForRecv[n], replace("reduce_scatter_wait_for_recv_$i_$n", "$i", i, "$n", n));
  // }

  fn(cuReduceScatterLocal, "reduce_scatter_local");
  fn(cuReduceScatter, "reduce_scatter");

  return std::make_pair(source, functions);
}

} // namespace moodist
