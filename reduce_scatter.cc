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

void ReduceScatter::init() {}

std::pair<std::string, std::vector<std::pair<CUfunction*, std::string>>> ReduceScatter::generate() {

  // allgatherBlocksize = group->ipcRanks.size() == group->size - 1 ? 256 : 64;
  // allgatherGridsize = group->ipcRanks.size() == group->size - 1 ? smCount / 4 : smCount / 4;

  const auto& ipcRanks = group->ipcRanks;
  const auto& peerMyRemoteIndex = group->peerMyRemoteIndex;
  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;

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

  std::string writes;
  std::string waits;
  for (size_t i : group->peerIndices) {
    waits += waitFor32(cudaStepValue.cudaPointer + sizeof(uint32_t) * group->ipcRanks[i], "stepValue");
    writes += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", peerCudaStepValue[i] + sizeof(uint32_t) * rank);
  }

  std::string copyDoneAllCode;
  std::string waitForCopyDones;
  for (size_t i : group->peerIndices) {
    copyDoneAllCode += replace(
        R"(
  *(volatile uint32_t*)$ptr = stepValue;
      )",
        "$ptr", peerCudaCopyDone[i] + sizeof(uint32_t) * peerMyRemoteIndex[i]);
    waitForCopyDones += replace(
        R"(
  while (*(volatile uint32_t*)$ptr < stepValue);
      )",
        "$i", i, "$ptr", cudaCopyDone.cudaPointer + sizeof(uint32_t) * i);
  }

  std::string source = replace(
      R"zz(

extern "C" __global__ void reduce_scatter_entry() {
  uint32_t stepValue = globalStepValue;
  // printf("enter %#x\n", stepValue);
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  cpuIn[0] = stepValue;
  $writes
  $waits
}

extern "C" __global__ void reduce_scatter_exit() {
  uint32_t stepValue = globalStepValue;
  $copyDoneAllCode
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  while (cpuOut[0] < stepValue);
  $waitForCopyDones
  //$waitForRecvs

  globalStepValue += 256;
}

  )zz",
      "$cpuOut", cast("volatile uint32_t*", cpuOutBuffer.cudaPointer), "$cpuIn",
      cast("volatile uint32_t*", cpuInBuffer.cudaPointer), "$writes", writes, "$waits", waits, "$waitForCopyDones",
      waitForCopyDones, "$copyDoneAllCode", copyDoneAllCode);

  std::vector<std::pair<CUfunction*, std::string>> functions;

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  fn(cuReduceScatterEntry, "reduce_scatter_entry");
  fn(cuReduceScatterExit, "reduce_scatter_exit");

  return std::make_pair(source, functions);
}

} // namespace moodist
