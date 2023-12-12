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
  sendRanks = group->allGather->recvRanks;
  recvRanks = group->allGather->sendRanks;

  ringSends = group->allGather->ringSends;
  ringRecvs = group->allGather->ringRecvs;
  if (!ringSends.empty()) {
    CHECK(ringSends.size() == ringRecvs.size());
    std::rotate(ringSends.begin(), ringSends.begin() + 1, ringSends.end());
    std::rotate(ringRecvs.begin(), ringRecvs.begin() + 1, ringRecvs.end());
    ringSends.back().first = ringSends.back().second;
    ringRecvs.back().first = rank;
  }

  std::string s;
  for (auto& v : ringSends) {
    if (!s.empty()) {
      s += ", ";
    }
    s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
  }
  log.info("rank %d: reduce scatter ring sends are [%s]\n", rank, s);
  s.clear();
  for (auto& v : ringRecvs) {
    if (!s.empty()) {
      s += ", ";
    }
    s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
  }
  log.info("rank %d: reduce scatter ring recvs are [%s]\n", rank, s);

  // CHECK(recvRanks.size() == sendRanks.size());
  // auto allSendRanks = group->setupComms->allgather(sendRanks);
  // for (auto& v : allSendRanks) {
  //   if (&v - allSendRanks.data() == rank) {
  //     continue;
  //   }
  //   CHECK(v.size() == sendRanks.size());
  //   for (size_t i = 0; i != sendRanks.size(); ++i) {
  //     CHECK(v[i] != sendRanks[i]);
  //   }
  // }
}

std::string
ReduceScatter::generate(std::vector<std::string> generateTypes, std::vector<std::string> generateReductions) {

  // allgatherBlocksize = group->ipcRanks.size() == group->size - 1 ? 256 : 64;
  // allgatherGridsize = group->ipcRanks.size() == group->size - 1 ? smCount / 4 : smCount / 4;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;
  const auto& peerIndices = group->peerIndices;
  const auto& cudaProxyReady = group->cudaProxyReady;
  const auto& peerCudaProxyReady = group->peerCudaProxyReady;

  auto waitFor32 = [&](std::string address, std::string value) {
    return replace(
        R"(while (*(volatile uint32_t*)($ptr) < $value);
    )",
        "$ptr", address, "$value", value);
  };

  auto waitForRecv = [&](size_t i, size_t chunkIndex, bool isLastChunk) {
    std::string s;
    s = replace("// wait for recv from $i, chunk $chunkIndex\n", "$i", i, "$chunkIndex", chunkIndex);
    for (size_t di = 0; di != group->ibDevs.size(); ++di) {
      if (isLastChunk) {
        s += waitFor32(concurrencyIndex(group->cudaCommsDeviceDataSent, sizeof(uint32_t) * (i * 32 + di)), "stepValue");
      } else {
        s += waitFor32(
            concurrencyIndex(group->cudaCommsDeviceDataSent2, sizeof(uint32_t) * (i * 32 * 32 + 32 * di + chunkIndex)),
            "stepValue");
      }
    }
    return s;
  };

  std::string reduceCode;

  std::unordered_map<std::string, bool> dstVisited;
  std::string prevReduceDst;
  std::string prevReduceSrc;
  bool isInGrid = true;
  auto callMemcpy = [&](std::string dst, std::string src) {
    if (isInGrid) {
      reduceCode += replace("copy_impl($dst, $src, params.bytes);\n", "$dst", dst, "$src", src);
    } else {
      reduceCode += replace("reduce_add<T, R>(reduces, $dst, $src, nullptr);\n", "$dst", dst, "$src", src);
    }
  };
  auto callReduceAdd = [&](std::string dst, std::string src1, std::string src2) {
    if (isInGrid) {
      reduceCode += replace(
          "reduce2<T, R>((T*)$dst, (const T*)$src1, (const T*)$src2, params.bytes / sizeof(T));\n", "$dst", dst,
          "$src1", src1, "$src2", src2);
    } else {
      reduceCode +=
          replace("reduce_add<T, R>(reduces, $dst, $src1, $src2);\n", "$dst", dst, "$src1", src1, "$src2", src2);
    }
  };
  auto reduceFlush = [&]() {
    if (prevReduceDst != "" && prevReduceSrc != "") {
      callMemcpy(prevReduceDst, prevReduceSrc);
    }
    prevReduceSrc = "";
    prevReduceDst = "";
  };
  auto reduceAdd = [&](std::string dst, std::string src) {
    std::string r;
    if (dst != prevReduceDst) {
      reduceFlush();
      if (dstVisited[dst]) {
        callReduceAdd(dst, dst, src);
      } else {
        prevReduceDst = dst;
        prevReduceSrc = src;
        dstVisited[dst] = true;
      }
    } else {
      if (prevReduceSrc == "") {
        callReduceAdd(dst, dst, src);
      } else {
        callReduceAdd(dst, prevReduceSrc, src);
        prevReduceSrc = "";
      }
    }
  };

  std::string globaldefs;

  // auto peerSync = [&](size_t peerIndex, size_t syncIndex) {
  //   if (peerIndices.empty()) {
  //     return;
  //   }
  //   CHECK(syncIndex < size);
  //   CHECK(peerIndex < peerIndices.size());
  //   reduceCode += replace(
  //       R"(
  //     if (threadIdx.x == 0) {
  //       if (isFirstThreadBlock) {
  //         *(volatile uint32_t*)$forwardPtr = stepValue + $peerIndex;
  //       }
  //       while (*(volatile uint32_t*)$readyPtr < stepValue + $peerIndex);
  //     }
  //     syncthreads();
  //   )",
  //       "$forwardPtr", concurrencyIndex(peerCudaProxyReady[0], sizeof(uint32_t) * syncIndex), "$readyPtr",
  //       concurrencyIndex(cudaProxyReady, sizeof(uint32_t) * syncIndex), "$peerIndex", peerIndex);
  // };

  auto peerSync = [&](size_t syncIndex) {
    if (peerIndices.empty()) {
      return;
    }
    if (ringSends.empty()) {
      return;
    }
    CHECK(syncIndex < size);
    std::string forwards;
    std::string waits;
    for (size_t peerIndex : peerIndices) {
      forwards += replace(
          "if (threadIdx.x == $peerIndex) *(volatile uint32_t*)$forwardPtr = stepValue + $syncIndex;\n", "$forwardPtr",
          concurrencyIndex(peerCudaProxyReady[peerIndex], sizeof(uint32_t) * peerMyRemoteIndex[peerIndex]),
          "$peerIndex", peerIndex);
      waits += replace(
          "if (threadIdx.x == $peerIndex) while (*(volatile uint32_t*)$readyPtr < stepValue + $syncIndex);\n",
          "$readyPtr", concurrencyIndex(cudaProxyReady, sizeof(uint32_t) * peerIndex), "$peerIndex", peerIndex);
    }
    std::string code = replace(
        R"(
      static_assert($blockSize >= $nPeers);
      if (isFirstThreadBlock) {
        $forwards
      }
      $waits
      syncthreads();
    )",
        "$forwards", forwards, "$waits", waits, "$nPeers", peerIndices.size());
    code = replace(code, "$syncIndex", syncIndex);
    reduceCode += code;
  };

  auto addLocals = [&]() {
    reduceAdd("(T*)params.outputAddress", "(const T*)(params.inputAddress + params.pitch * $rank)");
    peerSync(ringSends.size());
    for (size_t peerIndex : peerIndices) {
      // peerSync(peerIndex, ringSends.size());
      size_t i = ipcRanks[peerIndex];
      reduceAdd(
          "(T*)params.outputAddress",
          replace(
              "(const T*)(*(uintptr_t*)$src + params.pitch * $rank)", "$src",
              concurrencyIndex(group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * peerIndex))));
    }
  };

  if (recvRanks.empty()) {
    addLocals();
  } else {

    reduceCode += "static_assert($blockSize >= $gridSize);\n";
    size_t n = 0;
    for (auto [i, neighbor] : ringSends) {
      CHECK(i != rank);
      std::string addr = replace("(T*)(params.sendAddress + params.pitch * $n)", "$n", n);
      reduceAdd(addr, replace("(const T*)(params.inputAddress + params.pitch * $i)", "$i", i));
      peerSync(n);
      for (size_t peerIndex : peerIndices) {
        // peerSync(peerIndex, n);
        reduceAdd(
            addr, replace(
                      "(const T*)(*(uintptr_t*)$src + params.pitch * $i)", "$i", i, "$src",
                      concurrencyIndex(group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * peerIndex))));
      }

      reduceFlush();
      if (n != 0) {
        reduceCode += replace(
            R"(
          if (threadIdx.x == 0) {
            $wait
          }
          syncthreads();
        )",
            "$wait", waitForRecv(i, 0, true));
        reduceAdd(addr, replace("(const T*)(params.recvAddress + params.pitch * $n)", "$n", n - 1));
        reduceFlush();
      }
      globaldefs +=
          replace("__device__ volatile uint32_t sendReadyCounter_$n[$gridSize * $maxConcurrency];\n", "$n", n);
      reduceCode += replace(
          R"(
        __threadfence_system();
        syncthreads();
        if (threadIdx.x == 0) {
          sendReadyCounter_$n[$gridSize * concurrencyIndex + blockIdx.x] = stepValue;
        }
        if (isFirstThreadBlock) {
          if (threadIdx.x < $gridSize) {
            while (sendReadyCounter_$n[$gridSize * concurrencyIndex + threadIdx.x] < stepValue);
          }
          syncthreads();
          if (threadIdx.x == 0) {
            $cpuIn[1] = stepValue + $n + 1;
          }
        }
      )",
          "$n", n);
      ++n;
    }
    CHECK(prevReduceDst == "");
    addLocals();
    reduceCode += replace(
        R"(
          if (threadIdx.x == 0) {
            $wait
          }
          syncthreads();
        )",
        "$wait", waitForRecv(rank, 0, true));
    reduceAdd(
        "(T*)params.outputAddress",
        replace("(const T*)(params.recvAddress + params.pitch * $n)", "$n", ringSends.size() - 1));
  }
  reduceFlush();

  std::string kernels;
  for (auto& type : generateTypes) {
    for (auto& red : generateReductions) {
      kernels += replace(
          R"(
extern "C" __global__ void $launchBounds reduce_scatter_local_$type_$red(ReduceScatterParameters params) {
  reduce_scatter_local<$type, $red>(params);
}
extern "C" __global__ void $launchBounds reduce_scatter_$type_$red(ReduceScatterParameters params) {
  reduce_scatter<$type, $red>(params);
}
)",
          "$type", type, "$red", red);
    }
  }

  std::string source = replace(
      R"zz(

namespace {

constexpr size_t reduceQueueSize = 16;
struct ReduceParameters {
  size_t bytes;
  const void* src1[reduceQueueSize];
  const void* src2[reduceQueueSize];
  void* dst[reduceQueueSize];
  uint32_t n;
};

template<typename T, typename R>
__global__ void $launchBounds reduce_kernel(ReduceParameters params) {
  assert(params.n > 0);
  assert(params.n <= reduceQueueSize);
#pragma unroll 1
  for (uint32_t i = 0; i != params.n; ++i) {
    syncthreads();
    if (params.src2[i] != nullptr) {
      reduce2<T, R>((T*)params.dst[i], (const T*)params.src1[i], (const T*)params.src2[i], params.bytes / sizeof(T));
    } else {
      copy_impl(params.dst[i], params.src1[i], params.bytes);
    }
  }
}

$globaldefs

template<typename T, typename R>
__device__ void reduce_flush(ReduceParameters& params) {
  if (params.n) {
    // work around compiler bug
    ReduceParameters params2;
    memcpy(&params2, &params, sizeof(params2));
    reduce_kernel<T, R><<<$gridSize, $blockSize>>>(params2);
    params.n = 0;
  }
}

template<typename T, typename R>
__device__ void reduce_add(ReduceParameters& params, void* dst, const void* src1, const void* src2) {
  params.dst[params.n] = dst;
  params.src1[params.n] = src1;
  params.src2[params.n] = src2;
  ++params.n;
  if (params.n == reduceQueueSize) {
    reduce_flush<T, R>(params);
  }
}

struct ReduceScatterParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  size_t bytes;
  size_t pitch;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  uintptr_t peerInputAddresses[8];
  uintptr_t peerOutputAddresses[8];
  uintptr_t sendAddress;
  uintptr_t recvAddress;
};

template<typename T, typename R>
__device__ bool reduce_scatter_impl(ReduceScatterParameters& params, bool isFirstThreadBlock) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  // ReduceParameters reduces;
  // reduces.bytes = params.bytes;
  // reduces.n = 0;

  $reduceCode

  //reduce_flush<T, R>(reduces);
  return true;
}

}

extern "C" __global__ void reduce_scatter_exit(uint32_t stepValue, uint32_t concurrencyIndex) {
  generic_exit(stepValue, concurrencyIndex);
}

__device__ uint32_t firstThreadCounter[$maxConcurrency];

template<typename T, typename R>
__device__ void reduce_scatter_local(ReduceScatterParameters params) {
  grid_generic_entry_local(params);
  reduce_scatter_impl<T, R>(params, false);
  grid_generic_exit_local(params.stepValue, params.concurrencyIndex);
}

template<typename T, typename R>
__device__ void reduce_scatter(ReduceScatterParameters params) {
  bool isFirstThreadBlock = false;
  if (threadIdx.x == 0) {
    if (atomicInc(&firstThreadCounter[params.concurrencyIndex], $gridSize - 1) == 0) {
      isFirstThreadBlock = true;
    }
  }
  isFirstThreadBlock = syncthreads_or(isFirstThreadBlock);
  grid_generic_entry(params);
  reduce_scatter_impl<T, R>(params, isFirstThreadBlock);
  grid_generic_exit_no_recv(params.stepValue, params.concurrencyIndex);
  // if (reduce_scatter_impl<T, R>(params, isFirstThreadBlock)) {
  //   reduce_scatter_exit<<<1, 1>>>(params.stepValue, params.concurrencyIndex);
  // }
}

$kernels

  )zz",
      "$reduceCode", reduceCode, "$globaldefs", globaldefs, "$kernels", kernels);

  return source;
}

} // namespace moodist
