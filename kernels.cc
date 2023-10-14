
#include "kernels.h"
#include "allgather.h"
#include "common.h"
#include "group.h"

namespace moodist {

Kernels::~Kernels() {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
}

void Kernels::compile() {
  CUdevice& cuDevice = group->cuDevice;
  CUcontext& cuContext = group->cuContext;

  int computeMajor = 0;
  int computeMinor = 0;

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

  std::string source;
  std::vector<std::pair<CUfunction*, std::string>> functions;

  auto add = [&](auto&& v) {
    source += v.first;
    functions.insert(functions.end(), v.second.begin(), v.second.end());
  };

  add(group->allGather->generate());

  if (group->rank == 0 || true) {
    std::string fn = fmt::sprintf("moodist-kernels-rank%d.cu", group->rank);
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

  size_t cubinSize = 0;
  CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubinSize));
  std::vector<char> cubin;
  cubin.resize(cubinSize);
  fmt::printf("cubin size is %d\n", cubin.size());
  CHECK_NVRTC(nvrtcGetCUBIN(program, cubin.data()));

  if (group->rank == 0 || true) {
    FILE* f = fopen(fmt::sprintf("moodist-kernels-rank%d.o", group->rank).c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      fmt::printf("cubin dumped to allgather_kernel.o\n");
    }
  }

  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));

  for (auto& v : functions) {
    CHECK_CU(cuModuleGetFunction(v.first, cuModule, v.second.c_str()));
  }
}

} // namespace moodist
