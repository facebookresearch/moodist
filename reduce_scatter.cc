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
  if (!ringSends.empty() && !ringRecvs.empty()) {
    std::rotate(ringSends.begin(), ringSends.begin() + 1, ringSends.end());
    std::rotate(ringRecvs.begin(), ringRecvs.begin() + 1, ringRecvs.end());
    ringSends.back().first = ringSends.back().second;
    ringRecvs.back().first = rank;
  }

  sendRemoteRecvIndex.resize(recvRanks.size());
  auto allRecvRanks = group->setupComms->allgather(recvRanks);
  for (size_t index : indices(sendRanks)) {
    auto& c = allRecvRanks.at(sendRanks[index]);
    auto it = std::ranges::find(c, rank);
    CHECK(it != c.end());
    sendRemoteRecvIndex[index] = it - c.begin();
  }

  std::string s;
  for (auto& v : ringSends) {
    if (!s.empty()) {
      s += ", ";
    }
    s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
  }
  log.debug("rank %d: reduce scatter ring sends are [%s]\n", rank, s);
  s.clear();
  for (auto& v : ringRecvs) {
    if (!s.empty()) {
      s += ", ";
    }
    s += fmt::sprintf("(%d %d)", std::get<0>(v), std::get<1>(v));
  }
  log.debug("rank %d: reduce scatter ring recvs are [%s]\n", rank, s);

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

  auto waitForRecv = [&](std::string i, size_t chunkIndex) {
    std::string s;
    s = replace("// wait for recv from $i, chunk $chunkIndex\n", "$i", i, "$chunkIndex", chunkIndex);
    s += waitFor32(replace("&cpuOut[$offset + $i]", "$i", i, "$offset", 16 + size * chunkIndex), "stepValue");
    return s;
  };

  std::string reduceCode;

  std::string globaldefs;

  std::unordered_map<std::string, bool> dstVisited;
  std::string prevReduceDst;
  std::string prevReduceSrc;
  std::string prevReduceBytes;
  bool isInGrid = true;
  auto callMemcpy = [&](std::string dst, std::string src, std::string bytes) {
    reduceCode += replace(
        "newDynamicBlockIndex = copy_impl(dynamicBlockIndex, $dst, $src, $bytes);\n", "$dst", dst, "$src", src,
        "$bytes", bytes);
  };
  auto callReduceAdd = [&](std::string dst, std::string src1, std::string src2, std::string bytes) {
    reduceCode += replace(
        "newDynamicBlockIndex = reduce2<T, R>(dynamicBlockIndex, (T*)$dst, (const T*)$src1, (const T*)$src2, "
        "($bytes) / sizeof(T));\n",
        "$dst", dst, "$src1", src1, "$src2", src2, "$bytes", bytes);
  };
  auto reduceFlush = [&]() {
    if (prevReduceDst != "" && prevReduceSrc != "") {
      callMemcpy(prevReduceDst, prevReduceSrc, prevReduceBytes);
    }
    prevReduceSrc = "";
    prevReduceDst = "";
  };
  auto reduceAdd = [&](std::string dst, std::string src, std::string bytes) {
    std::string r;
    if (dst != prevReduceDst) {
      reduceFlush();
      if (dstVisited[dst]) {
        callReduceAdd(dst, dst, src, bytes);
      } else {
        prevReduceDst = dst;
        prevReduceSrc = src;
        prevReduceBytes = bytes;
        dstVisited[dst] = true;
      }
    } else {
      CHECK(bytes == prevReduceBytes);
      if (prevReduceSrc == "") {
        callReduceAdd(dst, dst, src, bytes);
      } else {
        callReduceAdd(dst, prevReduceSrc, src, bytes);
        prevReduceSrc = "";
      }
    }
  };

  // auto peerSync = [&]() {
  //   if (peerIndices.empty()) {
  //     return;
  //   }
  //   if (ringSends.empty()) {
  //     return;
  //   }
  //   CHECK(peerIndices.size() < 32);
  //   std::string forwards;
  //   std::string waits;
  //   size_t stride = size * maxChunks / peerIndices.size();
  //   for (size_t peerIndex : peerIndices) {
  //     forwards += replace(
  //         "if (threadIdx.x % 32 == $peerIndex) *(volatile uint32_t*)($forwardPtr + 4 * syncIndex) = stepValue;\n",
  //         "$forwardPtr",
  //         concurrencyIndex(peerCudaProxyReady[peerIndex], sizeof(uint32_t) * stride * peerMyRemoteIndex[peerIndex]),
  //         "$peerIndex", peerIndex);
  //     waits += replace(
  //         "if (threadIdx.x % 32 == $peerIndex) while (*(volatile uint32_t*)($readyPtr + 4 * syncIndex) <
  //         stepValue);\n",
  //         "$readyPtr", concurrencyIndex(cudaProxyReady, sizeof(uint32_t) * stride * peerIndex), "$peerIndex",
  //         peerIndex);
  //   }
  //   std::string code = replace(
  //       R"(
  //     static_assert($blockSize >= $nPeers);
  //     assert(syncIndex < $size * $maxChunks / $nPeers);
  //     $forwards
  //     $waits
  //     __syncwarp();
  //     ++syncIndex;
  //   )",
  //       "$forwards", forwards, "$waits", waits, "$nPeers", peerIndices.size());
  //   reduceCode += code;
  // };

  auto addLocals = [&]() {
    reduceAdd("(T*)params.outputAddress", "(const T*)(params.inputAddress + params.pitch * $rank)", "params.bytes");
    // peerSync(ringSends.size());
    for (size_t peerIndex : peerIndices) {
      // peerSync(peerIndex, ringSends.size());
      size_t i = ipcRanks[peerIndex];
      reduceAdd(
          "(T*)params.outputAddress",
          replace(
              "(const T*)(*(uintptr_t*)$src + params.pitch * $rank)", "$src",
              concurrencyIndex(group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * peerIndex))),
          "params.bytes");
    }
  };

  std::vector<uint32_t> instructions;

  for (auto [i, neighbor] : ringSends) {
    instructions.push_back(i);
  }

  std::vector<std::string> instructionsHex;
  for (auto& v : instructions) {
    instructionsHex.push_back(fmt::sprintf("%#x", v));
  }
  std::string instructionsArray = fmt::sprintf(
      "__constant__ uint32_t reduceScatterInstructions[%d] = {%s};", instructions.size(),
      fmt::to_string(fmt::join(instructionsHex, ", ")));

  if (instructions.empty()) {
    instructionsArray = "";
  }

  if (recvRanks.empty()) {
    addLocals();
    reduceFlush();
  } else {

    reduceCode += "static_assert($blockSize >= $gridSize);\n";

    reduceCode += replace(
        R"(
      for (uint32_t n = 0; n != $n; ++n) {
        uint32_t source = reduceScatterInstructions[n];
    )",
        "$n", instructions.size());

    auto forEachChunk = [&](auto&& f) {
      for (size_t chunkIndex = 0; chunkIndex != maxChunks; ++chunkIndex) {
        reduceCode += replace(
            R"(
          {
            // chunk $chunkIndex
            const size_t currentChunkSize = $currentChunkSize;
            const size_t chunkOffset = chunkSize * $chunkIndex;
            const uint32_t chunkIndex = $chunkIndex;
        )",
            "$currentChunkSize", "min(chunkSize, params.bytes - chunkSize * $chunkIndex)", "$chunkIndex", chunkIndex);

        f(chunkIndex);

        if (chunkIndex != maxChunks - 1) {
          reduceCode += "if (chunkSize * $chunkIndex + currentChunkSize != params.bytes) {\n";
          reduceCode += "assert(chunkSize * $chunkIndex + currentChunkSize < params.bytes);\n";
        } else {
          reduceCode += "assert(chunkSize * $chunkIndex + currentChunkSize == params.bytes);\n";
        }

        reduceCode = replace(reduceCode, "$chunkIndex", chunkIndex);
      }

      for (size_t chunkIndex = 0; chunkIndex != maxChunks; ++chunkIndex) {
        if (chunkIndex != maxChunks - 1) {
          reduceCode += "}\n";
        }
        reduceCode += "}\n";
      }
    };

    globaldefs +=
        replace("__device__ uint32_t sendReadyCounter[$n][$maxChunks][$maxConcurrency];\n", "$n", instructions.size());

    forEachChunk([&](size_t chunkIndex) {
      CHECK(prevReduceDst == "" && prevReduceSrc == "");
      dstVisited.clear();
      std::string addr = "(T*)(params.sendAddress + params.pitch * n + chunkOffset)";
      if (peerIndices.empty()) {
        reduceCode += "if (n == 0) {\n";
      }
      reduceAdd(addr, "(const T*)(params.inputAddress + params.pitch * source + chunkOffset)", "currentChunkSize");
      // peerSync();
      for (size_t peerIndex : peerIndices) {
        reduceAdd(
            addr,
            replace(
                "(const T*)(*(uintptr_t*)$src + params.pitch * source + chunkOffset)", "$src",
                concurrencyIndex(group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * peerIndex))),
            "currentChunkSize");
      }
      reduceFlush();
      if (peerIndices.empty()) {
        dstVisited.clear();
        reduceCode += "} else {\n";
      } else {
        reduceCode += "if (n != 0) {\n";
      }
      reduceCode += replace(
          R"(
          if (threadIdx.x % 32 == 0) {
            $wait
          }
          __syncwarp();
        )",
          "$wait", waitForRecv("source", chunkIndex));
      if (peerIndices.empty()) {
        reduceAdd(addr, "(const T*)(params.inputAddress + params.pitch * source + chunkOffset)", "currentChunkSize");
      }
      reduceAdd(addr, "(const T*)(params.recvAddress + params.pitch * (n - 1) + chunkOffset)", "currentChunkSize");
      reduceFlush();
      reduceCode += "}\n";
      reduceCode += R"(
        __syncwarp();
        if (threadIdx.x % 32 == 0) {
          if (atomicInc(&sendReadyCounter[n][$chunkIndex][concurrencyIndex], $gridSize * $blockSize / 32 - 1) == $gridSize * $blockSize / 32 - 1) {
            $cpuIn[16 + $size * $chunkIndex + n] = stepValue;
          }
        }
        __syncwarp();
      )";
      reduceCode += "dynamicBlockIndex = newDynamicBlockIndex;\n";
    });

    reduceCode += "}\n";

    // addLocals();
    // reduceFlush();
    forEachChunk([&](size_t chunkIndex) {
      CHECK(prevReduceDst == "" && prevReduceSrc == "");
      dstVisited.clear();
      std::string addr = "(T*)(params.outputAddress + chunkOffset)";
      reduceAdd(addr, "(const T*)(params.inputAddress + params.pitch * $rank + chunkOffset)", "currentChunkSize");
      // peerSync();
      for (size_t peerIndex : peerIndices) {
        size_t i = ipcRanks[peerIndex];
        reduceAdd(
            addr,
            replace(
                "(const T*)(*(uintptr_t*)$src + params.pitch * $rank + chunkOffset)", "$src",
                concurrencyIndex(group->cudaPeerAddresses, (sizeof(uintptr_t) * 2 * peerIndex))),
            "currentChunkSize");
      }
      reduceCode += replace(
          R"(
            if (threadIdx.x % 32 == 0) {
              $wait
            }
            __syncwarp();
          )",
          "$wait", waitForRecv("$rank", chunkIndex));
      reduceAdd(
          addr, replace("(const T*)(params.recvAddress + params.pitch * $n + chunkOffset)", "$n", ringSends.size() - 1),
          "currentChunkSize");
      reduceFlush();
      reduceCode += "dynamicBlockIndex = newDynamicBlockIndex;\n";
    });
  }

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

$globaldefs

struct ReduceScatterParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  size_t bytes;
  size_t pitch;
  size_t chunkSize;
  uintptr_t inputAddress;
  uintptr_t outputAddress;
  uintptr_t peerInputAddresses[8];
  uintptr_t peerOutputAddresses[8];
  uintptr_t sendAddress;
  uintptr_t recvAddress;
};

$instructionsArray

template<typename T, typename R>
__device__ bool reduce_scatter_impl(ReduceScatterParameters& params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  [[maybe_unused]] volatile uint32_t* __restrict__ cpuOut = $cpuOut;

  uint32_t dynamicBlockIndex = $gridSize * (threadIdx.x / 32u) + blockIdx.x;
  uint32_t newDynamicBlockIndex = dynamicBlockIndex;
  uint32_t dynamicBlockCounter = 0;
  uint32_t syncIndex = 0;

  const size_t chunkSize = params.chunkSize;

  $reduceCode

  return true;
}

}

extern "C" __global__ void reduce_scatter_exit(uint32_t stepValue, uint32_t concurrencyIndex) {
  generic_exit(stepValue, concurrencyIndex);
}

template<typename T, typename R>
__device__ void reduce_scatter_local(ReduceScatterParameters params) {
  grid_generic_entry_local(params);
  reduce_scatter_impl<T, R>(params);
  grid_generic_exit_local(params.stepValue, params.concurrencyIndex);
}

template<typename T, typename R>
__device__ void reduce_scatter(ReduceScatterParameters params) {
  grid_generic_entry(params);
  reduce_scatter_impl<T, R>(params);
  grid_generic_exit_no_recv(params.stepValue, params.concurrencyIndex);
}

$kernels

  )zz",
      "$reduceCode", reduceCode, "$globaldefs", globaldefs, "$kernels", kernels, "$instructionsArray",
      instructionsArray);

  return source;
}

} // namespace moodist
