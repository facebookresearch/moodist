
#include "kernels.h"
#include "allgather.h"
#include "common.h"
#include "group.h"
#include "reduce_scatter.h"

namespace moodist {

Kernels::~Kernels() {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
}

std::string
Kernels::emitReduceFunction(std::string type, size_t typesize, size_t nsources, std::string op, size_t blockSize) {
  TORCH_CHECK(type != "");
  std::string sourcesargs;
  for (size_t i = 0; i != nsources; ++i) {
    if (!sourcesargs.empty()) {
      sourcesargs += ", ";
    }
    sourcesargs += replace("$type* source_$i", "$type", type, "$i", i);
  }
  std::string addressesdefs;
  std::string dst = "dst";
  addressesdefs += "uintptr_t dst = (uintptr_t)destination + sizeof($type) * offset;\n";
  // addressesdefs += "printf(\"dst alignment is %d\\n\", dst & 127);\n";
  std::vector<std::string> addresses;
  for (size_t i = 0; i != nsources; ++i) {
    addresses.push_back(fmt::sprintf("src%d", i));
    addressesdefs += replace("uintptr_t src$i = (uintptr_t)source_$i + sizeof($type) * offset;\n", "$i", i);
    // addressesdefs += replace("printf(\"src$i alignment is %d\\n\", src$i & 127);\n", "$i", i);
  }
  addressesdefs = replace(addressesdefs, "$type", type);
  size_t stride = blockSize;
  auto addOffset = [&](int typesize) {
    std::string s;
    s += fmt::sprintf("%s += %d * (int)%%offset;\n", dst, typesize);
    for (auto& v : addresses) {
      s += fmt::sprintf("%s += %d * (int)%%offset;\n", v, typesize);
    }
    return s;
  };
  auto reduceopcode = [&](auto& names) {
    if (op == "sum") {
      return fmt::to_string(fmt::join(names, " + "));
    } else {
      TORCH_CHECK(false, "unknown reduce op");
    }
  };
  std::function<std::string(std::string, std::string, std::vector<std::string>)> getreducecode =
      [&](std::string datatype, std::string result, std::vector<std::string> inputs) {
        if (datatype == type) {
          return replace("$result = $code;", "$result", result, "$code", reduceopcode(inputs));
        } else if (datatype == "uint4") {
          std::string s;
          for (auto& field : {"x", "y", "z", "w"}) {
            auto inputs2 = inputs;
            for (std::string& v : inputs2) {
              v += ".";
              v += field;
            }
            s += getreducecode("uint32_t", result + "." + field, inputs2);
            s += "\n";
          }
          return s;
        } else if (datatype == "uint32_t") {
          if (type == "float") {
            for (auto& v : inputs) {
              v = replace("__uint_as_float($v)", "$v", v);
            }
            return replace("$result = __float_as_uint($op);", "$result", result, "$op", reduceopcode(inputs));
          } else {
            TORCH_CHECK(false, "unhandled reduce type for datatype uint32_t");
          }
        } else {
          TORCH_CHECK(false, "don't know how to reduce these types");
        }
      };
  auto generate = [&](std::string datatype, int datatypesize, int unroll) {
    TORCH_CHECK(datatypesize % typesize == 0);
    std::string s = R"(
$pre
  while (%remaining >= $loopsize) {
    size_t %loopcount = %remaining / $loopsize;
    $preloop
#pragma unroll 1
    for (size_t %i = 0; %i != %loopcount - 1; ++%i) {
      //__syncwarp();
      syncthreads();
      $loop
    }
    $postloop
    %remaining -= $loopsize * %loopcount;
  }
$post
)";
    std::string pre = addOffset(datatypesize);
    std::string post = addOffset(-(int)datatypesize);
    // pre += replace(R"(printf("pre destination -> %#lx\n", $dst);)", "$dst", addresses[0]);
    // pre += replace(R"(printf("pre source -> %#lx\n", $dst);)", "$dst", addresses[1]);
    std::string preloop;
    std::string loop;
    std::string postloop;
    // if (datatypesize == 16 && cg->computeMajor >= 8 && false) {
    if (false) {
      // std::string sharedinit;
      // size_t sharedOffset = 0;
      // for (size_t n = 0; n != addresses.size(); ++n) {
      //   for (int i = 0; i != unroll; ++i) {
      //     // sharedinit += replace("alignas(128) __shared__ $datatype data_$i_$n[$stride];\n", "$i", i, "$n", n);
      //     sharedinit += replace(
      //         R"(
      //       $datatype* data_$i_$n = ($datatype*)((uintptr_t)sharedptr + $offset);
      //     )",
      //         "$i", i, "$n", n, "$offset", sharedOffset);
      //     sharedOffset += datatypesize * stride;
      //   }
      // }
      // currentFunction->sharedMemSize = std::max(currentFunction->sharedMemSize, sharedOffset);
      // preloop = replace(
      //     R"(
      //   $sharedinit
      // )",
      //     "$sharedinit", sharedinit, "$unroll", unroll);

      // for (int i = 0; i != unroll; ++i) {
      //   std::string reads;
      //   for (size_t n = 0; n != addresses.size(); ++n) {
      //     auto& src = addresses[n];
      //     std::string read = replace(
      //         R"(
      //         asm volatile ("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::
      //         "r"((uint32_t)(__cvta_generic_to_shared(&data_$i_$n[%offset]))), "l"((void*)($src)) : "memory"); $src
      //         += $datatypesize * $stride;
      //     )",
      //         "$i", i, "$n", n, "$src", src);
      //     preloop += read;
      //     reads += read;
      //   }
      //   preloop += R"(asm volatile ("cp.async.commit_group; "::);
      //   )";
      //   reads += R"(asm volatile ("cp.async.commit_group; "::);
      //   )";

      //   std::string settemps;
      //   std::vector<std::string> tempnames;
      //   for (size_t n = 0; n != addresses.size(); ++n) {
      //     std::string temp = getVar("temporary");
      //     preloop += replace("$datatype $temp;\n", "$temp", temp);
      //     settemps += replace("$temp = data_$i_$n[%offset];\n", "$temp", temp, "$i", i, "$n", n);
      //     tempnames.push_back(temp);
      //   }
      //   std::string result = getVar("result");
      //   std::string reducecode = getreducecode(datatype, result, tempnames);
      //   preloop += replace("$datatype $result;\n", "$result", result);
      //   std::string write = replace(
      //       R"($settemps
      //       $reducecode
      //       __stwt(($datatype*)(void*)$dst, $result); $dst += $datatypesize * $stride;
      //               )",
      //       "$result", result, "$dst", dst, "$reducecode", reducecode, "$settemps", settemps);
      //   postloop += replace(
      //       R"(asm volatile ("cp.async.wait_group $N; ":::"memory");
      //   )",
      //       "$N", unroll - i - 1);
      //   loop += replace(
      //       R"(asm volatile ("cp.async.wait_group $N; ":::"memory");
      //   )",
      //       "$N", unroll - 1);
      //   postloop += write;
      //   loop += write;
      //   loop += reads;
      // }

    } else {
      for (int i = 0; i != unroll; ++i) {
        std::vector<std::string> tempnames;
        std::string reads;
        for (size_t n = 0; n != addresses.size(); ++n) {
          auto& src = addresses[n];
          std::string temp = getVar("temporary");
          tempnames.push_back(temp);
          std::string read = replace(
              R"($temp = __ldcv(($datatype*)(void*)$src); $src += $datatypesize * $stride;
                  )",
              "$temp", temp, "$src", src);
          preloop += datatype + " " + read;
          reads += read;
        }
        std::string result = getVar("result");
        std::string reducecode = getreducecode(datatype, result, tempnames);
        preloop += replace("$datatype $result;\n", "$result", result);
        std::string write = replace(
            R"($reducecode
            __stwt(($datatype*)(void*)$dst, $result); $dst += $datatypesize * $stride;
                    )",
            "$result", result, "$dst", dst, "$reducecode", reducecode);
        postloop += write;
        loop += write;
        loop += reads;
      }
    }
    size_t loopsize = (datatypesize / typesize) * unroll * stride;
    s = replace(
        s, "$loopsize", loopsize, "$preloop", preloop, "$postloop", postloop, "$loop", loop, "$pre", pre, "$post",
        post);
    s = replace(s, "$datatype", datatype, "$datatypesize", datatypesize, "$stride", stride, "$type", type);
    return s;
  };
  auto generatelast = [&](std::string datatype, int datatypesize) {
    TORCH_CHECK(datatypesize % typesize == 0);
    std::string s = R"(
$pre
  $preloop
  $loop
)";
    std::string pre = addOffset(datatypesize);
    std::string preloop;
    std::string loop;
    std::vector<std::string> tempnames;
    std::string reads;
    for (size_t n = 0; n != addresses.size(); ++n) {
      auto& src = addresses[n];
      std::string temp = getVar("temporary");
      tempnames.push_back(temp);
      std::string read = replace(
          R"($temp = __ldcv(($datatype*)(void*)$src); $src += $datatypesize * $stride;
                )",
          "$temp", temp, "$src", src);
      preloop += datatype + " " + read;
      reads += read;
    }
    std::string result = getVar("result");
    std::string reducecode = getreducecode(datatype, result, tempnames);
    preloop += replace("$datatype $result;\n", "$result", result);
    std::string write = replace(
        R"($reducecode
          __stwt(($datatype*)(void*)$dst, $result); $dst += $datatypesize * $stride;
                  )",
        "$result", result, "$dst", dst, "$reducecode", reducecode);
    loop += write;
    loop += reads;
    s = replace(s, "$preloop", preloop, "$loop", loop, "$pre", pre);
    s = replace(s, "$datatype", datatype, "$datatypesize", datatypesize, "$stride", stride, "$type", type);
    return s;
  };
  std::string reducelastitem = generatelast(type, typesize);
  std::string reduce1 = generate("uint4", 16, 16);
  // std::string reduce1 = generate("uint4", 16, 5);
  std::string reduce2 = generate(type, typesize, 1);
  std::string s = replace(
      R"(
__device__ __noinline__ void reduce_$typename_$op_n$nsources(void* sharedptr, size_t offset, size_t size, $typename* destination, $sourcesargs) {
  size_t %remaining = size;
  uint32_t %offset = threadIdx.x;
  if ((uintptr_t)destination < 0x100000 || (uintptr_t)source_0 < 0x100000) {
    printf("bad reduce with destination %p, source %p, offset %#x\n", destination, source_0, %offset);
    while(1);
  }
  $addressesdefs
  // if (threadIdx.x == 0)
  // printf("$rank: reduce with destination %#lx, source_0 %#lx, source_1 %#lx, offset %#x\n", dst, src0, src1, %offset);
  $reduce1
  $reduce2
  if (%offset < (uint32_t)%remaining) {
    $reducelastitem
  }
}
  )",
      "$sourcesargs", sourcesargs, "$op", op, "$nsources", nsources, "$typename", type, "$reduce1", reduce1, "$reduce2",
      reduce2, "$addressesdefs", addressesdefs, "$blockSize", blockSize, "$reducelastitem", reducelastitem);
  s = makeVars(s);
  return s;
}

void Kernels::compile() {
  CUdevice& cuDevice = group->cuDevice;
  CUcontext& cuContext = group->cuContext;

  if (cuModule) {
    cuModuleUnload(cuModule);
    cuModule = nullptr;
  }

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

  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;

  std::string source;
  std::vector<std::pair<CUfunction*, std::string>> functions;

  reduceGridSize = 128;
  reduceBlockSize = 128;

  source = R"(
typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;

__device__ uint32_t globalStepValue = 1;

__device__ void syncthreads() {
  asm volatile ("barrier.sync 0;" :: );
}

)";

  source += emitReduceFunction("float", 4, 2, "sum", reduceBlockSize);

  source += replace(
      R"(
extern "C" __global__ void reduce(float* dst, float* src, size_t numel) {
  size_t blockIndex = blockIdx.x;
  size_t threadIndex = threadIdx.x;
  for (size_t i = blockIndex * $reduceBlockSize + threadIndex; i < numel; i += $reduceGridSize * $reduceBlockSize) {
    dst[i] += src[i];
  }
  // size_t chunkSize = numel / $reduceGridSize / 1024u * 1024u;
  // size_t offset = blockIdx.x * chunkSize;
  // if (blockIdx.x == $reduceGridSize - 1) {
  //   chunkSize = numel - offset;
  // }
  // if (chunkSize > 0) {
  //   reduce_float_sum_n2(nullptr, offset, chunkSize, dst, dst, src);
  // }
}
  )",
      "$reduceGridSize", reduceGridSize, "$reduceBlockSize", reduceBlockSize);

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  fn(cuReduce, "reduce");

  auto add = [&](auto&& v) {
    source += v.first;
    functions.insert(functions.end(), v.second.begin(), v.second.end());
  };

  add(group->allGather->generate());
  add(group->reduceScatter->generate());

  source = replace(
      source, "$rank", rank, "$cpuOut", cast("volatile uint32_t*", cpuOutBuffer.cudaPointer), "$cpuIn",
      cast("volatile uint32_t*", cpuInBuffer.cudaPointer));

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  if (rank == 0 || true) {
    std::string fn = fmt::sprintf("moodist-kernels-rank%d.cu", rank);
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

  if (rank == 0 || true) {
    std::string fn = fmt::sprintf("moodist-kernels-rank%d.o", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      fmt::printf("cubin dumped to %s\n", fn);
    }
  }

  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));

  for (auto& v : functions) {
    CHECK_CU(cuModuleGetFunction(v.first, cuModule, v.second.c_str()));
  }
}

} // namespace moodist
