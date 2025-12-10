// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "reduce_scatter.h"

namespace moodist {

std::string generate_alltoall(const Group* group) {
  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;
  const auto& peerIndices = group->peerIndices;
  const auto& cudaProxyReady = group->cudaProxyReady;
  const auto& peerCudaProxyReady = group->peerCudaProxyReady;

  const auto& reduceScatter = *group->reduceScatter;

  const auto& replace = [&]<typename... T>(T&&... v) {
    return reduceScatter.replace(std::forward<T>(v)...);
  };
  const auto& concurrencyIndex = [&]<typename... T>(T&&... v) {
    return reduceScatter.concurrencyIndex(std::forward<T>(v)...);
  };

  const size_t rank = group->rank;
  const size_t size = group->size;

  std::string globaldefs;

  auto addCopy = [&](std::string dst, std::string src, std::string bytes) {
    return replace("dynamicBlockIndex = copy_impl(dynamicBlockIndex, $dst, $src, $bytes);\n", "$dst", dst, "$src", src,
        "$bytes", bytes);
  };

  std::string code;
  for (size_t i : peerIndices) {
    code += addCopy(replace("(void*)params.outputAddress[$i]", "$i", i),
        replace("(void*)params.peerInputAddresses[$i]", "$i", i), "params.bytes");
  }

  std::string source = replace(
      R"zz(

namespace {

$globaldefs

struct AllToAllParameters {
  size_t bytes;
  uintptr_t outputAddress[8];
  uintptr_t peerInputAddresses[8];
};

}

extern "C" __global__ void $launchBounds alltoall(AllToAllParameters params) {
  uint32_t dynamicBlockIndex = $gridSize * (threadIdx.x / 32u) + blockIdx.x;
  uint32_t newDynamicBlockIndex = dynamicBlockIndex;
  uint32_t dynamicBlockCounter = 0;
  uint32_t syncIndex = 0;

  $code
}

  )zz",
      "$code", code, "$globaldefs", globaldefs);
  return source;
}

} // namespace moodist
