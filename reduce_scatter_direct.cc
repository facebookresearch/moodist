
#include "reduce_scatter.h"

namespace moodist {

std::string generate_reduce_scatter_direct(
    const Group* group, std::vector<std::string> generateTypes, std::vector<std::string> generateReductions) {
  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;
  const auto& peerIndices = group->peerIndices;
  const auto& cudaProxyReady = group->cudaProxyReady;
  const auto& peerCudaProxyReady = group->peerCudaProxyReady;

  const auto& reduceScatter = *group->reduceScatter;

  const auto& replace = [&]<typename... T>(T&&... v) { return reduceScatter.replace(std::forward<T>(v)...); };
  const auto& concurrencyIndex = [&]<typename... T>(T&&... v) {
    return reduceScatter.concurrencyIndex(std::forward<T>(v)...);
  };

  const size_t rank = group->rank;
  const size_t size = group->size;

  std::string globaldefs;
  std::string reduceCode;

  std::vector<uint32_t> instructions;

  for (auto [i, neighbor] : reduceScatter.ringSends) {
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

  std::unordered_map<std::string, bool> dstVisited;
  std::string prevReduceDst;
  std::string prevReduceSrc;
  std::string prevReduceBytes;
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
    // reduceCode += replace(
    //     R"(if (threadIdx.x == 0) printf("reduce add dst %#lx, src %#lx, bytes %d\n", (unsigned long)($dst), (unsigned
    //     long)($src), (int)($bytes)); syncthreads();
    // )",
    //     "$dst", dst, "$src", src, "$bytes", bytes);
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

  reduceCode += "static_assert($blockSize >= $gridSize);\n";

  reduceCode += replace(
      R"(
      for (uint32_t n = 0; n != $n; ++n) {
        uint32_t source = reduceScatterInstructions[n];
    )",
      "$n", instructions.size());

  std::string syncForward;
  std::string syncWait;
  if (false) {
    for (size_t peerIndex : peerIndices) {
      syncForward += replace(
          "*(volatile uint32_t*)($forwardPtr + 4 * n) = stepValue;\n", "$forwardPtr",
          concurrencyIndex(peerCudaProxyReady[peerIndex], sizeof(uint32_t) * size * peerMyRemoteIndex[peerIndex]),
          "$peerIndex", peerIndex);
      syncWait += replace(
          "if (threadIdx.x % 32 == $peerIndex) while (*(volatile uint32_t*)($readyPtr + 4 * (n - 1)) < stepValue);\n",
          "$readyPtr", concurrencyIndex(cudaProxyReady, sizeof(uint32_t) * size * peerIndex), "$peerIndex", peerIndex);
    }
    syncWait += "__syncwarp();\n";
  }

  reduceCode += replace(
      R"(
      if (n) {
        $syncWait
      }
    )",
      "$syncWait", syncWait);

  auto forEachChunk = [&](auto&& f) {
    for (size_t chunkIndex = 0; chunkIndex != maxChunks; ++chunkIndex) {
      reduceCode += replace(
          R"(
          {
            // chunk $chunkIndex
            const size_t currentChunkSize = $currentChunkSize;
            const size_t chunkOffset = chunkSize * $chunkIndex;
            const uint32_t chunkIndex = $chunkIndex;
            const bool isLastChunk = chunkSize * $chunkIndex + currentChunkSize == params.bytes;
        )",
          "$currentChunkSize", "min(chunkSize, params.bytes - chunkSize * $chunkIndex)", "$chunkIndex", chunkIndex);

      f(chunkIndex);

      if (chunkIndex != maxChunks - 1) {
        reduceCode += "if (!isLastChunk) {\n";
        // reduceCode += "assert(chunkSize * $chunkIndex + currentChunkSize < params.bytes);\n";
      } else {
        // reduceCode += "assert(chunkSize * $chunkIndex + currentChunkSize == params.bytes);\n";
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
    reduceAdd(addr, "(const T*)(params.inputAddress + params.pitch * source + chunkOffset)", "currentChunkSize");
    for (size_t peerIndex : peerIndices) {
      reduceAdd(
          addr,
          replace(
              "(const T*)(params.peerInputAddresses[$peerIndex] + params.pitch * source + chunkOffset)", "$peerIndex",
              peerIndex),
          "currentChunkSize");
    }
    reduceFlush();
    reduceCode += replace(
        R"(
        __syncwarp();
        if (threadIdx.x % 32 == 0) {
          if (atomicInc(&sendReadyCounter[n][$chunkIndex][concurrencyIndex], $gridSize * $blockSize / 32 - 1) == $gridSize * $blockSize / 32 - 1) {
            $cpuIn[16 + $size * $chunkIndex + n] = stepValue;
            if (isLastChunk) {
              $syncForward
            }
          }
        }
        __syncwarp();
      )",
        "$syncForward", syncForward);
    // reduceCode += "dynamicBlockIndex = newDynamicBlockIndex;\n";
  });

  reduceCode += "}\n";

  std::string kernels;
  for (auto& type : generateTypes) {
    for (auto& red : generateReductions) {
      kernels += replace(
          R"(
extern "C" __global__ void $launchBounds reduce_scatter_direct_$type_$red(ReduceScatterParameters params) {
  reduce_scatter_direct<$type, $red>(params);
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
__device__ void reduce_scatter_direct(ReduceScatterParameters params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  [[maybe_unused]] volatile uint32_t* __restrict__ cpuOut = $cpuOut;

  uint32_t dynamicBlockIndex = $gridSize * (threadIdx.x / 32u) + blockIdx.x;
  uint32_t newDynamicBlockIndex = dynamicBlockIndex;
  uint32_t dynamicBlockCounter = 0;
  uint32_t syncIndex = 0;

  const size_t chunkSize = params.chunkSize;

  $reduceCode
}

}

$kernels

  )zz",
      "$reduceCode", reduceCode, "$globaldefs", globaldefs, "$kernels", kernels, "$instructionsArray",
      instructionsArray);
  return source;
}

} // namespace moodist
