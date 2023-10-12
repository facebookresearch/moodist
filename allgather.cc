#include "allgather.h"
#include "common.h"
#include "group.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include <vector>

namespace moodist {

AllGather::AllGather(Group* group) : CollectiveBase(group) {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
}

AllGather::~AllGather() {}

void AllGather::init() {

  const size_t rank = group->rank;
  const size_t size = group->size;

  SetupComms* setupComms = &*group->setupComms;

  auto& ipcRanks = group->ipcRanks;
  auto& ipcAccess = group->ipcAccess;
  auto& peerIpcAccess = group->peerIpcAccess;
  auto& nodeRanks = group->nodeRanks;

  // Quick and dirty algorithm which requires that all nodes have the same number of local ranks,
  // and that all local ranks have ipc access between each other
  // This also assumes that ranks are sequential on nodes.

  size_t myNode = -1;
  size_t myLocalRank = -1;
  for (size_t i = 0; i != nodeRanks.size(); ++i) {
    auto& list = nodeRanks[i];
    auto it = std::find(list.begin(), list.end(), rank);
    if (it != list.end()) {
      myNode = i;
      myLocalRank = it - list.begin();
    }
  }
  CHECK(myNode != -1 && myLocalRank != -1);
  size_t ranksPerNode = nodeRanks[myNode].size();
  for (auto& list : nodeRanks) {
    CHECK(list.size() == ranksPerNode);
    // fmt::printf("node is %s\n", fmt::to_string(fmt::join(list, ", ")));
    for (size_t dst = 0; dst != list.size(); ++dst) {
      for (size_t src = 0; src != list.size(); ++src) {
        if (src == dst) {
          continue;
        }
        if (!peerIpcAccess.at(list[src]).at(list[dst])) {
          fmt::printf("no ipc %d <-> %d\n", list[src], list[dst]);
        }
        CHECK(peerIpcAccess.at(list[src]).at(list[dst]));
      }
    }
  }

  auto findLocalRank = [&](size_t rank) {
    for (auto& list : nodeRanks) {
      for (size_t n = 0; n != list.size(); ++n) {
        if (list[n] == rank) {
          return n;
        }
      }
    }
    CHECK(false);
    return (size_t)-1;
  };

  std::vector<std::vector<size_t>> paths;
  for (size_t src = 0; src != size; ++src) {
    size_t l = findLocalRank(src);
    for (auto& list : nodeRanks) {
      if (std::find(list.begin(), list.end(), src) != list.end()) {
        continue;
      }
      if (src == rank) {
        paths.push_back({src, list[l]});
      } else if (list[l] == rank) {
        for (size_t n = 0; n != list.size(); ++n) {
          if (n == l) {
            paths.push_back({src, list[n]});
          } else {
            paths.push_back({src, list[l], list[n]});
          }
        }
      } else if (std::find(list.begin(), list.end(), rank) != list.end()) {
        paths.push_back({src, list[l], rank});
      }
    }
  }

  auto sortFn = [&](auto& a, auto& b) {
    if (a.size() != b.size()) {
      return a.size() < b.size();
    }
    int am;
    int bm;
    am = (size + a[1] - a[0]) % size;
    bm = (size + b[1] - b[0]) % size;
    if (am != bm) {
      return am < bm;
    }
    if (a.size() == 3) {
      am = (ranksPerNode + a[2] - a[1]) % ranksPerNode;
      bm = (ranksPerNode + b[2] - b[1]) % ranksPerNode;
      if (am != bm) {
        return am > bm;
      }
    }
    return a < b;
  };
  std::sort(paths.begin(), paths.end(), sortFn);

  std::stable_partition(paths.begin(), paths.end(), [&](auto& a) { return a[0] == rank; });
  std::stable_partition(paths.begin(), paths.end(), [&](auto& a) {
    if (a.size() < 3) {
      return true;
    }
    return a[1] != rank;
  });

  for (auto& v : paths) {
    if (v.size() == 2) {
      CHECK(v[0] == rank || v[1] == rank);
    } else if (v.size() == 3) {
      CHECK(v[0] != rank);
      CHECK(v[1] == rank || v[2] == rank);
    } else {
      CHECK(false);
    }
  }

  fmt::printf("paths.size() is %d\n", paths.size());
  if (rank == 0 || true) {
    FILE* f = fopen(fmt::sprintf("paths-%d.txt", rank).c_str(), "wb");
    CHECK(f != nullptr);
    for (auto& path : paths) {
      fmt::fprintf(f, "[%s]\n", fmt::to_string(fmt::join(path, ", ")));
    }
    fclose(f);
  }

  for (auto& path : paths) {
    CHECK(path.size() <= 3);
    CHECK(path.size() >= 2);
    size_t source = path[0];
    size_t proxy = path[1];
    size_t destination = path.back();
    if (path.size() == 3) {
      CHECK(source != proxy && source != destination && proxy != destination);
    } else if (path.size() == 2) {
      CHECK(source != proxy && source != destination && proxy == destination);
    }
    CHECK((proxy == destination) == (path.size() == 2));
    if (source == rank) {
      CHECK(path.size() == 2);
      CHECK(std::find(sendRanks.begin(), sendRanks.end(), destination) == sendRanks.end());
      sendRanks.push_back(destination);
    } else if (path.size() == 2 && destination == rank) {
      CHECK(std::find(recvRanks.begin(), recvRanks.end(), source) == recvRanks.end());
      recvRanks.push_back(source);
    } else if (proxy == rank) {
      CHECK(std::find(recvRanks.begin(), recvRanks.end(), source) != recvRanks.end());
      if (destination != rank) {
        CHECK(path.size() >= 3);
        ProxyInfo pi;
        pi.source = source;
        pi.destination = destination;
        pi.destinationPeerIndex = group->getPeerIndex(destination);
        proxyInfo.push_back(pi);
      }
    } else if (destination == rank) {
      CHECK(path.size() >= 3);
      ProxyDestinationInfo pid;
      pid.source = source;
      pid.proxy = proxy;
      pid.proxyPeerIndex = group->getPeerIndex(proxy);
      proxyDestinationInfo.push_back(pid);
    }
  }

  CHECK(proxyInfo.size() == proxyDestinationInfo.size());

  auto proxyDestinationCounts = setupComms->allgather(proxyDestinationInfo.size());
  for (size_t n : proxyDestinationCounts) {
    CHECK(n == proxyDestinationInfo.size());
  }

  fmt::printf("rank %d: ipc ranks are [%s]\n", rank, fmt::to_string(fmt::join(ipcRanks, ", ")));
  fmt::printf("rank %d: send ranks are [%s]\n", rank, fmt::to_string(fmt::join(sendRanks, ", ")));
  fmt::printf("rank %d: recv ranks are [%s]\n", rank, fmt::to_string(fmt::join(recvRanks, ", ")));

  std::string proxyForwardOrder;
  std::string proxyCopyOrder;

  for (auto& v : proxyInfo) {
    proxyForwardOrder += fmt::sprintf(" (%d %d)", v.source, v.destination);
  }
  for (auto& v : proxyDestinationInfo) {
    proxyCopyOrder += fmt::sprintf(" (%d %d)", v.source, v.proxy);
  }

  fmt::printf("%d: proxy forward order %s\n", rank, proxyForwardOrder);
  fmt::printf("%d: proxy copy order %s\n", rank, proxyCopyOrder);
}

void AllGather::compile() {
  CUdevice& cuDevice = group->cuDevice;
  CUcontext& cuContext = group->cuContext;

  int major = 6;
  int minor = 0;
  CHECK_NVRTC(nvrtcVersion(&major, &minor));
  fmt::printf("nvrtc version is %d %d\n", major, minor);

  int driverVersion = 5000;
  CHECK_CU(cuDriverGetVersion(&driverVersion));

  fmt::printf("cuda driver version is %d\n", driverVersion);

  CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  fmt::printf("cuda device compute capability is %d.%d\n", computeMajor, computeMinor);

  int cudaWarpSize = 0;
  int cudaMaxSharedMemory = 0;
  int maxThreadsPerSm = 0;
  int smCount = 0;

  CHECK_CU(cuDeviceGetAttribute(&cudaWarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&cudaMaxSharedMemory, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice));

  CHECK_CU(cuDeviceGetAttribute(&maxThreadsPerSm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice));

  fmt::printf("warp size: %d\nmax shared memory: %d\n", cudaWarpSize, cudaMaxSharedMemory);
  fmt::printf("multiprocessor count: %d\nmax threads per multiprocessor: %d\n", smCount, maxThreadsPerSm);

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
        R"(while (*(volatile uint32_t*)$ptr < $value) __threadfence_system();
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

  std::string copyDoneFunctions;
  std::string copyDoneAllCode;
  std::string waitForCopyDones;
  for (size_t i : group->peerIndices) {
    copyDoneFunctions += replace(
        R"(
extern "C" __global__ void allgather_copy_done_$i(uint32_t stepValue) {
  *(volatile uint32_t*)$ptr = stepValue;
}
      )",
        "$i", i, "$ptr", peerCudaCopyDone[i] + sizeof(uint32_t) * peerMyRemoteIndex[i]);
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

  std::string waitForRecvs;
  for (size_t i : recvRanks) {
    waitForRecvs += waitForRecv(i, Group::dataChunks - 1);
  }

  std::string waitForProxyFunctions;
  CHECK(proxyInfo.size() == proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    auto& pi = proxyInfo[i];
    auto& pdi = proxyDestinationInfo[i];
    for (size_t c = 0; c != Group::dataChunks; ++c) {
      waitForProxyFunctions += replace(
          R"(
  extern "C" __global__ void allgather_wait_for_proxy_$i_$c(uint32_t stepValue) {
    __threadfence_system();
    $wait
    __threadfence_system();
    // forward source $forwardSource destination $forwardDestination (peer index $forwardDestinationPeerIndex)
    *(volatile uint32_t*)$forwardPtr = stepValue + $c;
    __threadfence_system();
    // wait for ready source $proxySource proxy $proxyProxy (peer index $proxyProxyPeerIndex)
    while (*(volatile uint32_t*)$readyPtr < stepValue + $c) __threadfence_system();
  }

  extern "C" __global__ void allgather_wait_for_recv_$forwardSource_$c_$i(uint32_t stepValue) {
    __threadfence_system();
    $wait
    __threadfence_system();
    // forward source $forwardSource destination $forwardDestination (peer index $forwardDestinationPeerIndex)
    *(volatile uint32_t*)$forwardPtr = stepValue + $c;
  }
  extern "C" __global__ void allgather_wait_for_ready_$proxySource_$c(uint32_t stepValue) {
    // wait for ready source $proxySource proxy $proxyProxy (peer index $proxyProxyPeerIndex)
    while (*(volatile uint32_t*)$readyPtr < stepValue + $c) __threadfence_system();
  }
      )",
          "$i", i, "$c", c, "$readyPtr", cudaProxyReady.cudaPointer + sizeof(uint32_t) * pdi.source, "$wait",
          waitForRecv(pi.source, c), "$forwardPtr",
          peerCudaProxyReady[pi.destinationPeerIndex] + sizeof(uint32_t) * pi.source, "$proxySource", pdi.source,
          "$proxyProxy", pdi.proxy, "$proxyProxyPeerIndex", pdi.proxyPeerIndex, "$forwardSource", pi.source,
          "$forwardDestination", pi.destination, "$forwardDestinationPeerIndex", pi.destinationPeerIndex);
    }
  }

  std::string source = replace(
      R"zz(

typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;


extern "C" __global__ void allgather_entry(uint32_t stepValue) {
  // printf("enter %#x\n", stepValue);
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  cpuIn[0] = stepValue;
  $writes
  __threadfence_system();
  $waits
}

extern "C" __global__ void allgather_exit(uint32_t stepValue) {
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  __threadfence_system();
  while (cpuOut[0] < stepValue);
  $waitForCopyDones
  $waitForRecvs
  __threadfence_system();
}

$copyDoneFunctions

extern "C" __global__ void allgather_copy_all_done(uint32_t stepValue) {
  $copyDoneAllCode
}

$waitForProxyFunctions

  )zz",
      "$cpuOut", cast("volatile uint32_t*", cpuOutBuffer.cudaPointer), "$cpuIn",
      cast("volatile uint32_t*", cpuInBuffer.cudaPointer), "$writes", writes, "$waits", waits, "$copyDoneFunctions",
      copyDoneFunctions, "$waitForCopyDones", waitForCopyDones, "$waitForRecvs", waitForRecvs, "$copyDoneAllCode",
      copyDoneAllCode, "$waitForProxyFunctions", waitForProxyFunctions);

  source = replace(source, "$rank", rank);

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  // if (rank == 0) {
  //   fmt::printf("code\n----\n%s\n", source);
  // }

  if (rank == 0 || true) {
    std::string fn = fmt::sprintf("allgather_kernel-rank%d.cu", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(source.data(), source.size(), 1, f);
      fclose(f);
      fmt::printf("source dumped to %s\n", fn);
    }
  }

  nvrtcProgram program;
  CHECK_NVRTC(nvrtcCreateProgram(&program, source.c_str(), nullptr, 0, nullptr, nullptr));

  std::vector<std::pair<int, std::string>> archOptions;
  archOptions.emplace_back(9000, "--gpu-architecture=sm_90");
  archOptions.emplace_back(8090, "--gpu-architecture=sm_89");
  archOptions.emplace_back(8070, "--gpu-architecture=sm_87");
  archOptions.emplace_back(8000, "--gpu-architecture=sm_80");
  archOptions.emplace_back(7050, "--gpu-architecture=sm_75");
  archOptions.emplace_back(7020, "--gpu-architecture=sm_72");
  archOptions.emplace_back(7000, "--gpu-architecture=sm_70");
  archOptions.emplace_back(6020, "--gpu-architecture=sm_62");
  archOptions.emplace_back(6010, "--gpu-architecture=sm_61");
  archOptions.emplace_back(6000, "--gpu-architecture=sm_60");
  archOptions.emplace_back(5030, "--gpu-architecture=sm_53");
  archOptions.emplace_back(5020, "--gpu-architecture=sm_52");
  archOptions.emplace_back(0, "--gpu-architecture=sm_50");

  std::vector<std::string> options;
  options.push_back("<gpu-architecture>");
  options.push_back("--use_fast_math");
  options.push_back("--std=c++17");
  options.push_back("-lineinfo");
  // options.push_back("--maxrregcount=32");
  //    options.push_back("-G");
  //     options.push_back("--dopt=on");
  nvrtcResult error = NVRTC_ERROR_INVALID_OPTION;
  for (size_t i = 0; i != archOptions.size() && error == NVRTC_ERROR_INVALID_OPTION; ++i) {
    if (computeMajor * 1000 + computeMinor * 10 < archOptions[i].first) {
      continue;
    }
    options[0] = archOptions[i].second;
    std::vector<const char*> options2;
    for (auto& v : options) {
      options2.push_back(v.c_str());
    }
    error = nvrtcCompileProgram(program, options2.size(), options2.data());
    if (error == NVRTC_SUCCESS) {
      fmt::printf("success with %s\n", archOptions[i].second);
    }
  }
  if (error != NVRTC_SUCCESS) {
    fmt::printf("Failed to compile--\n%s\n", source.c_str());
    size_t logSize = 0;
    std::string log;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &logSize));
    log.resize(logSize);
    CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));
    fmt::printf("%s\n", log);

    CHECK_NVRTC(error);
  } else {
    size_t logSize = 0;
    std::string log;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &logSize));
    log.resize(logSize);
    CHECK_NVRTC(nvrtcGetProgramLog(program, log.data()));
    for (char& c : log) {
      if (c < 32 || c >= 127) {
        if (c != 10 && c != 13) {
          c = 32;
        }
      }
    }
    fmt::printf("compile log\n---\n%s\n", log);
  }

  // size_t ptxSize = 0;
  // CHECK_NVRTC(nvrtcGetPTXSize(program, &ptxSize));
  // std::vector<char> ptx;
  // ptx.resize(ptxSize);
  // CHECK_NVRTC(nvrtcGetPTX(program, ptx.data()));

  // CHECK_NVRTC(nvrtcDestroyProgram(&program));

  // CHECK_CU(cuModuleLoadDataEx(&op.cuModule, ptx.data(), 0, nullptr, nullptr));
  // CHECK_CU(cuModuleGetFunction(&op.cuAllgather, op.cuModule, "allgather"));

  size_t cubinSize = 0;
  CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubinSize));
  std::vector<char> cubin;
  cubin.resize(cubinSize);
  fmt::printf("cubin size is %d\n", cubin.size());
  CHECK_NVRTC(nvrtcGetCUBIN(program, cubin.data()));

  if (rank == 0 || true) {
    FILE* f = fopen(fmt::sprintf("allgather_kernel-rank%d.o", rank).c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      fmt::printf("cubin dumped to allgather_kernel.o\n");
    }
  }

  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));
  CHECK_CU(cuModuleGetFunction(&cuAllgatherEntry, cuModule, "allgather_entry"));
  CHECK_CU(cuModuleGetFunction(&cuAllgatherExit, cuModule, "allgather_exit"));
  for (size_t i : group->peerIndices) {
    CHECK_CU(
        cuModuleGetFunction(&cuAllgatherCopyDone[i], cuModule, replace("allgather_copy_done_$i", "$i", i).c_str()));
  }
  CHECK_CU(cuModuleGetFunction(&cuAllgatherCopyAllDone, cuModule, "allgather_copy_all_done"));
  cuAllgatherWaitForProxy.resize(proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    cuAllgatherWaitForProxy[i].resize(Group::dataChunks);
    for (size_t c = 0; c != Group::dataChunks; ++c) {
      CHECK_CU(cuModuleGetFunction(
          &cuAllgatherWaitForProxy[i][c], cuModule,
          replace("allgather_wait_for_proxy_$i_$c", "$i", i, "$c", c).c_str()));
    }
  }

  cuAllgatherWaitForRecv.resize(proxyDestinationInfo.size());
  cuAllgatherWaitForReady.resize(proxyDestinationInfo.size());
  for (size_t i = 0; i != proxyDestinationInfo.size(); ++i) {
    cuAllgatherWaitForRecv[i].resize(Group::dataChunks);
    cuAllgatherWaitForReady[i].resize(Group::dataChunks);
    for (size_t c = 0; c != Group::dataChunks; ++c) {
      CHECK_CU(cuModuleGetFunction(
          &cuAllgatherWaitForRecv[i][c], cuModule,
          replace("allgather_wait_for_recv_$source_$c_$i", "$i", i, "$c", c, "$source", proxyInfo[i].source).c_str()));
      CHECK_CU(cuModuleGetFunction(
          &cuAllgatherWaitForReady[i][c], cuModule,
          replace("allgather_wait_for_ready_$i_$c", "$i", proxyDestinationInfo[i].source, "$c", c).c_str()));
    }
  }

  // for (size_t i = 0; i != ipcRanks.size(); ++i) {
  //   CHECK_CU(cuModuleGetFunction(
  //       &cuAllgatherWaitForPeer[i], cuModule, fmt::sprintf("allgather_wait_for_peer_%d", i).c_str()));
  // }
  // cuAllgatherWaitForPeerData.resize(size);
  // for (size_t i : recvRanks) {
  //   CHECK_CU(cuModuleGetFunction(
  //       &cuAllgatherWaitForPeerData[i], cuModule, fmt::sprintf("allgather_wait_for_data_%d", i).c_str()));
  // }

  // size_t globalStepValueSize = 0;
  // CHECK_CU(cuModuleGetGlobal(&cuGlobalStepValue, &globalStepValueSize, cuModule, "globalStepValue"));
  // TORCH_CHECK(globalStepValueSize == 4);
}

} // namespace moodist
