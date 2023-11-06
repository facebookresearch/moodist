
#include "kernels.h"
#include "allgather.h"
#include "common.h"
#include "group.h"
#include "reduce_scatter.h"

#include <algorithm>
#include <link.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace moodist {

Kernels::~Kernels() {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
}

std::string Kernels::emitCopySeq(
    std::vector<std::string> sources, std::vector<std::string> destinations, std::string bytes, size_t gridSize,
    size_t blockSize, std::string threadIndex, std::string blockIndex) {
  TORCH_CHECK(sources.size() == destinations.size());
  auto genCopy = [&](std::string type, int typesize, int unroll) {
    std::string nbytes;
    std::string condi;
    std::string s;
    if (unroll == 1) {
      nbytes = "$typesize";
      condi = "$blockSize * %i + %threadIndex";
    } else {
      nbytes = "$loopbytes";
      condi = "%i";
    }
    s = replace(
        R"(
    %count = (%bytes - %offset) / $nbytes;
    %i = %blockIndex;
    if ($cond < %count) {
      $preloop
      while ($cond < %count) {
        $loop
      }
      $postloop
      if (%count < %numBlocks) {
        return;
      }
    } else {
      if (%count < %numBlocks) {
        %numBlocks -= %count;
        %blockIndex -= %count;
      }
    }
    %offset += $loopbytes * %count;
    if (%offset == %bytes) {
      return;
    }
  )",
        "$nbytes", nbytes, "$cond", condi);
    std::string preloop;
    std::string loop;
    std::string postloop;
    std::string loopreads;
    std::string loopwrites;
    if (unroll != 1) {
      preloop += "__syncthreads();\n";
      loop += "__syncthreads();\n";
    }
    for (int i = 0; i != unroll; ++i) {
      for (size_t n = 0; n != destinations.size(); ++n) {
        auto& src = sources[n];
        std::string temp = getVar("temporary");
        std::string read = replace(
            R"($temp = __ldcv(($type*)(void*)($src + %offset + $loopbytes * %i + $typesize * $blockSize * $i + $typesize * %threadIndex));
                )",
            "$type", type, "$temp", temp, "$src", src, "$i", i, "$typesize", typesize);
        if (i == unroll - 1 && n == destinations.size() - 1) {
          read += "%i += %numBlocks;\n";
        }
        preloop += type + " " + read;
        auto& dst = destinations[n];
        std::string write = replace(
            R"(__stwt(($type*)(void*)($dst + %offset + $loopbytes * (%i - %numBlocks) + $typesize * $blockSize * $i + $typesize * %threadIndex), $temp);
                )",
            "$type", type, "$temp", temp, "$dst", dst, "$i", i, "$typesize", typesize);
        postloop += write;
        loop += write;
        loop += read;
      }
    }
    size_t loopbytes = typesize * unroll * blockSize;
    s = replace(s, "$loop", loop, "$preloop", preloop, "$postloop", postloop, "$loopbytes", loopbytes);
    s = replace(s, "$datatype", type, "$typesize", typesize, "$type", type, "$loopbytes", loopbytes, "$unroll", unroll);
    return s;
  };
  std::string s;
  s += fmt::sprintf(
      "/* copy  bytes: %s  gridSize: %d  blockSize: %d  threadIndex: %s  blockIndex: %s */\n", bytes, gridSize,
      blockSize, threadIndex, blockIndex);
  s += R"(
  const size_t %bytes = $bytes;
  const uint32_t %threadIndex = $threadIndex;
  static_assert($gridSize % $blocksPerSm == 0);
  uint32_t %blockIndex = $blockIndex / $blocksPerSm + ($gridSize / $blocksPerSm * ($blockIndex % $blocksPerSm));
  uint32_t %numBlocks = $gridSize;
  size_t %offset = 0;
  size_t %count = 0;
  size_t %i;
  $defs
  $copy1
)";

  std::string defs;
  for (auto& v : sources) {
    auto var = getVar(v);
    defs += fmt::sprintf("uintptr_t %s = (uintptr_t)(void*)(%s);\n", var, v);
    v = var;
  }
  for (auto& v : destinations) {
    auto var = getVar(v);
    defs += fmt::sprintf("uintptr_t %s = (uintptr_t)(void*)(%s);\n", var, v);
    v = var;
  }
  std::string copy1;
  // 16 uint4 uses 64 registers. we could go higher to hide more latency,
  // but it seems to not improve performance
  // copy1 = genCopy("uint4", 16, 32);
  copy1 += genCopy("uint4", 16, 16);
  copy1 += genCopy("uint4", 16, 8);
  copy1 += genCopy("uint4", 16, 4);
  copy1 += genCopy("uint4", 16, 2);
  copy1 += genCopy("uint4", 16, 1);
  s = replace(s, "$bytes", bytes, "$defs", defs, "$copy1", copy1);
  s = replace(
      s, "$threadIndex", threadIndex, "$blockIndex", blockIndex, "$gridSize", gridSize, "$blockSize", blockSize);
  s = makeVars(s);
  return s;
}

std::string Kernels::emitCopy(
    std::vector<std::string> sources, std::vector<std::string> destinations, std::string bytes, size_t gridSize,
    size_t blockSize, std::string threadIndex, std::string blockIndex) {
  TORCH_CHECK(sources.size() == destinations.size());
  size_t stride = gridSize * blockSize;
  auto genwrites = [&](std::string& writes, std::string type, int typesize,
                       const std::vector<std::string>& temporaries) {
    for (size_t i = 0; i != destinations.size(); ++i) {
      auto& v = destinations[i];
      writes += fmt::sprintf("__stwt((%s*)(void*)%s, %s);\n", type, v, temporaries[i]);
    }
  };
  auto readwrite = [&](std::string& reads, std::string& writes, std::string type, int typesize) {
    std::vector<std::string> temporaries;
    for (auto& v : sources) {
      std::string temp = getVar("temporary");
      temporaries.push_back(temp);
      reads += fmt::sprintf("%s %s = __ldcv((%s*)(void*)%s);\n", type, temp, type, v);
    }
    genwrites(writes, type, typesize, temporaries);
  };
  auto addOffset = [&](int typesize) {
    std::string s;
    for (auto& v : sources) {
      s += fmt::sprintf("%s += %d * %%offset;\n", v, typesize);
    }
    for (auto& v : destinations) {
      s += fmt::sprintf("%s += %d * %%offset;\n", v, typesize);
    }
    return s;
  };
  auto genCopy = [&](std::string type, int typesize, int unroll) {
    std::string s = R"(
$pre
  while (%bytes >= $loopbytes) {
    size_t %loopcount = %bytes / $loopbytes;
    $preloop
    syncthreads();
#pragma unroll 1
    for (size_t %i = 0; %i != %loopcount - 1; ++%i) {
      //__syncwarp();
      //syncthreads();
      $loop
    }
    $postloop
    %bytes -= $loopbytes * %loopcount;
  }
$post
)";
    std::string pre = addOffset(typesize);
    std::string preloop;
    std::string loop;
    std::string postloop;
    for (int i = 0; i != unroll; ++i) {
      for (size_t n = 0; n != destinations.size(); ++n) {
        auto& src = sources[n];
        std::string temp = getVar("temporary");
        std::string read = replace(
            R"($temp = __ldcv(($type*)(void*)$src); $src += $typesize * $stride;
                )",
            "$type", type, "$temp", temp, "$src", src, "$typesize", typesize, "$stride", stride);
        preloop += type + " " + read;
        auto& dst = destinations[n];
        std::string write = replace(
            R"(__stwt(($type*)(void*)$dst, $temp); $dst += $typesize * $stride;
                )",
            "$type", type, "$temp", temp, "$dst", dst, "$typesize", typesize, "$stride", stride);
        postloop += write;
        loop += write;
        loop += read;
      }
    }
    std::string post;
    for (auto& v : sources) {
      post += fmt::sprintf("%s -= %d * %%offset;\n", v, typesize);
    }
    for (auto& v : destinations) {
      post += fmt::sprintf("%s -= %d * %%offset;\n", v, typesize);
    }
    size_t loopbytes = typesize * unroll * stride;
    s = replace(
        s, "$loop", loop, "$pre", pre, "$post", post, "$loopbytes", loopbytes, "$preloop", preloop, "$postloop",
        postloop);
    s = replace(s, "$datatype", type, "$datatypesize", typesize, "$stride", stride, "$type", type);
    return s;
  };
  std::string s;
  s += fmt::sprintf(
      "/* copy  bytes: %s  gridSize: %d  blockSize: %d  threadIndex: %s  blockIndex: %s */\n", bytes, gridSize,
      blockSize, threadIndex, blockIndex);
  s += R"(
  size_t %bytes = $bytes;
  uint32_t %threadIndex = $threadIndex;
  uint32_t %blockIndex = $blockIndex;
  uint32_t %offset = $blockSize * %blockIndex + %threadIndex;
  $defs
  $copy1
  $copy2
  $copy3
  if (%offset < (uint32_t)%bytes) {
    $copylastbyte
  }
)";

  std::string defs;
  for (auto& v : sources) {
    auto var = getVar(v);
    defs += fmt::sprintf("uintptr_t %s = (uintptr_t)(void*)(%s);\n", var, v);
    v = var;
  }
  for (auto& v : destinations) {
    auto var = getVar(v);
    defs += fmt::sprintf("uintptr_t %s = (uintptr_t)(void*)(%s);\n", var, v);
    // defs += replace(R"(if (threadIdx.x == 0) printf("rank $rank got dst %%#llx from $v\n", (uintptr_t)$var);
    // )", "$var", var, "$v", v);
    v = var;
  }
  std::string copy1;
  // 16 uint4 uses 64 registers. we could go higher to hide more latency,
  // but it seems to not improve performance
  // copy1 = genCopy("uint4", 16, 32);
  copy1 += genCopy("uint4", 16, 16);
  copy1 += genCopy("uint4", 16, 8);
  copy1 += genCopy("uint4", 16, 4);
  copy1 += genCopy("uint4", 16, 1);
  std::string copy2 = genCopy("unsigned int", 4, 2) + genCopy("unsigned int", 4, 1);
  std::string copy3 = genCopy("unsigned char", 1, 1);
  std::string copylastbyte = addOffset(1);
  readwrite(copylastbyte, copylastbyte, "unsigned char", 1);
  s = replace(
      s, "$bytes", bytes, "$defs", defs, "$copy1", copy1, "$copy2", copy2, "$copy3", copy3, "$copylastbyte",
      copylastbyte);
  s = replace(
      s, "$threadIndex", threadIndex, "$blockIndex", blockIndex, "$copyGridSize", gridSize, "$blockSize", blockSize);
  s = makeVars(s);
  return s;
}

std::string Kernels::emitReduceFunction(
    std::string type, size_t typesize, size_t nsources, std::string op, size_t gridSize, size_t blockSize) {
  TORCH_CHECK(type != "");
  std::string sourcesargs;
  for (size_t i = 0; i != nsources; ++i) {
    if (!sourcesargs.empty()) {
      sourcesargs += ", ";
    }
    sourcesargs += replace("const $type* source_$i", "$type", type, "$i", i);
  }
  std::string addressesdefs;
  std::string dst = "dst";
  addressesdefs += "uintptr_t dst = (uintptr_t)destination;\n";
  // addressesdefs += "printf(\"dst alignment is %d\\n\", dst & 127);\n";
  std::vector<std::string> addresses;
  for (size_t i = 0; i != nsources; ++i) {
    addresses.push_back(fmt::sprintf("src%d", i));
    addressesdefs += replace("uintptr_t src$i = (uintptr_t)source_$i;\n", "$i", i);
    // addressesdefs += replace("printf(\"src$i alignment is %d\\n\", src$i & 127);\n", "$i", i);
  }
  addressesdefs = replace(addressesdefs, "$type", type);
  size_t stride = gridSize * blockSize;
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
    syncthreads();
#pragma unroll 1
    for (size_t %i = 0; %i != %loopcount - 1; ++%i) {
      //__syncwarp();
      //syncthreads();
      $loop
    }
    $postloop
    %remaining -= $loopsize * %loopcount;
  }
$post
)";
    std::string pre = addOffset(datatypesize);
    std::string post = addOffset(-(int)datatypesize);
    std::string preloop;
    std::string loop;
    std::string postloop;
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
  std::string reduce1; // = generate("uint4", 16, 32);
  reduce1 += generate("uint4", 16, 8);
  reduce1 += generate("uint4", 16, 4);
  reduce1 += generate("uint4", 16, 2);
  reduce1 += generate("uint4", 16, 1);
  std::string reduce2;
  if (typesize <= 4) {
    reduce2 += generate(type, typesize, 2);
  }
  reduce2 += generate(type, typesize, 1);
  std::string s = replace(
      R"(
__device__ void reduce_$typename_$op_n$nsources(size_t size, $typename* destination, $sourcesargs) {
  size_t %remaining = size;
  uint32_t %offset = $blockSize * blockIdx.x + threadIdx.x;
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

  source = R"(
typedef unsigned long uintptr_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef unsigned char uint8_t;

namespace {

__device__ uint32_t globalStepValue = 1;

__device__ void syncthreads() {
  asm volatile ("barrier.sync 0;" :: );
}

__device__ void copy_impl(void* __restrict__ dst, const void* __restrict__ src, size_t bytes) {
  $copyCode
}

)";

  gridSize = 36;
  blockSize = 256;
  size_t blocksPerSm = 3;

  source = replace(
      source, "$copyCode", emitCopySeq({"src"}, {"dst"}, "bytes", gridSize, blockSize, "threadIdx.x", "blockIdx.x"));

  source += emitReduceFunction("float", 4, 2, "sum", gridSize, blockSize);

  source += R"(
template<typename T>
__device__ void reduce2_sum(T* __restrict__ dst, const T* __restrict__ src1, const T* __restrict__ src2, size_t numel) {
  reduce_float_sum_n2(numel, dst, src1, src2);
}
)";

  auto waitFor32 = [&](uintptr_t address, std::string value) {
    return replace(
        R"(while (*(volatile uint32_t*)$ptr < $value);
    )",
        "$ptr", address, "$value", value);
  };

  std::string writes1;
  std::string writes2;
  std::string waits;
  for (size_t i : group->peerIndices) {
    waits += waitFor32(group->cudaStepValue.cudaPointer + sizeof(uint32_t) * group->ipcRanks[i], "stepValue");
    writes1 += replace(
        R"(
      ((volatile uintptr_t*)$addrPtr)[0] = params.peerInputAddresses[$i];
      ((volatile uintptr_t*)$addrPtr)[1] = params.peerOutputAddresses[$i];
    )",
        "$addrPtr", group->peerCudaPeerAddresses[i] + (sizeof(uintptr_t) * 2 * group->peerMyRemoteIndex[i]), "$i", i);
    writes2 += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", group->peerCudaStepValue[i] + sizeof(uint32_t) * rank);
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
  for (size_t i : group->allGather->recvRanks) {
    waitForRecvs += waitForRecv(i, Group::dataChunks - 1);
  }

  std::string copyDoneAllCode;
  std::string waitForCopyDones;
  for (size_t i : group->peerIndices) {
    copyDoneAllCode += replace(
        R"(
  *(volatile uint32_t*)$ptr = stepValue;
      )",
        "$ptr", group->peerCudaCopyDone[i] + sizeof(uint32_t) * group->peerMyRemoteIndex[i]);
    waitForCopyDones += replace(
        R"(
  while (*(volatile uint32_t*)$ptr < stepValue);
      )",
        "$i", i, "$ptr", group->cudaCopyDone.cudaPointer + sizeof(uint32_t) * i);
  }

  source += replace(
      R"(
template<typename Params>
__device__ void generic_entry_local(const Params& params) {
  [[maybe_unused]] const uint32_t stepValue = globalStepValue;
  $writes1
  __threadfence_system();
  $writes2
  $waits
}
__device__ void generic_exit_local() {
  [[maybe_unused]] const uint32_t stepValue = globalStepValue;
  $copyDoneAllCode
  $waitForCopyDones

  globalStepValue += 4096;
}
template<typename Params>
__device__ void generic_entry(const Params& params) {
  [[maybe_unused]] const uint32_t stepValue = globalStepValue;
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  cpuIn[0] = stepValue;
  $writes1
  __threadfence_system();
  $writes2
  $waits
}
__device__ void generic_exit() {
  uint32_t stepValue = globalStepValue;
  $copyDoneAllCode
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  while (cpuOut[0] < stepValue);
  $waitForCopyDones
  $waitForRecvs

  globalStepValue += 4096;
}

__device__ uint32_t entryCounter = 0;
__device__ uint32_t exitCounter = 0;

template<typename Params>
__device__ void grid_generic_entry_local(const Params& params) {
  if (threadIdx.x == 0) {
    [[maybe_unused]] const uint32_t stepValue = globalStepValue;
    if (atomicInc(&entryCounter, $gridSize - 1) == 0) {
      $writes1
      __threadfence_system();
      $writes2
    }
    $waits
  }
  syncthreads();
}
__device__ void grid_generic_exit_local() {
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter, $gridSize - 1) == $gridSize - 1) {
    [[maybe_unused]] const uint32_t stepValue = globalStepValue;
    $copyDoneAllCode
    $waitForCopyDones

    globalStepValue += 4096;
  }
}
template<typename Params>
__device__ void grid_generic_entry(const Params& params) {
  if (threadIdx.x == 0) {
    [[maybe_unused]] const uint32_t stepValue = globalStepValue;
    volatile uint32_t* __restrict__ cpuIn = $cpuIn;
    if (atomicInc(&entryCounter, $gridSize - 1) == 0) {
      cpuIn[0] = stepValue;
      $writes1
      __threadfence_system();
      $writes2
    }
    $waits
  }
  syncthreads();
}
__device__ void grid_generic_exit() {
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter, $gridSize - 1) == $gridSize - 1) {
    uint32_t stepValue = globalStepValue;
    $copyDoneAllCode
    volatile uint32_t* __restrict__ cpuIn = $cpuIn;
    volatile uint32_t* __restrict__ cpuOut = $cpuOut;
    cpuIn[0] = stepValue + 1;
    while (cpuOut[0] < stepValue);
    $waitForCopyDones
    $waitForRecvs

    globalStepValue += 4096;
  }
}

}
  )",
      "$writes1", writes1, "$writes2", writes2, "$waits", waits, "$waitForCopyDones", waitForCopyDones, "$waitForRecvs",
      waitForRecvs, "$copyDoneAllCode", copyDoneAllCode);

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  auto add = [&](auto&& v) {
    source += v.first;
    functions.insert(functions.end(), v.second.begin(), v.second.end());
  };

  add(group->allGather->generate());
  add(group->reduceScatter->generate());

  source = replace(source, "$launchBounds", "__launch_bounds__($blockSize, $blocksPerSm)");
  source = replace(source, "$gridSize", gridSize, "$blockSize", blockSize, "$blocksPerSm", blocksPerSm);

  source = replace(source, "$sharedMemSize", 40960);

  source = replace(
      source, "$rank", rank, "$cpuOut", cast("volatile uint32_t*", cpuOutBuffer.cudaPointer), "$cpuIn",
      cast("volatile uint32_t*", cpuInBuffer.cudaPointer));

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  std::string cuFilename = fmt::sprintf("moodist-kernels-rank%d.cu", rank);

  if (rank == 0 || true) {
    FILE* f = fopen(cuFilename.c_str(), "wb");
    if (f) {
      fwrite(source.data(), source.size(), 1, f);
      fclose(f);
      fmt::printf("source dumped to %s\n", cuFilename);
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
  options.push_back("--relocatable-device-code=true");
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

  size_t ptxSize = 0;
  CHECK_NVRTC(nvrtcGetPTXSize(program, &ptxSize));
  std::vector<char> ptx;
  ptx.resize(ptxSize);
  CHECK_NVRTC(nvrtcGetPTX(program, ptx.data()));

  if (rank == 0 || true) {
    std::string fn = fmt::sprintf("moodist-kernels-rank%d-ptx.s", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(ptx.data(), ptx.size(), 1, f);
      fclose(f);
      fmt::printf("ptx dumped to %s\n", fn);
    }
  }

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

  // CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));

  // for (auto& v : functions) {
  //   CHECK_CU(cuModuleGetFunction(v.first, cuModule, v.second.c_str()));
  // }

  auto exists = [&](std::string fn) {
    struct stat buf;
    return ::stat(fn.c_str(), &buf) == 0;
  };

  std::string devrtPath = CUDADEVRT_PATH;
  if (!exists(devrtPath)) {
    std::string fn = "/usr/local/cuda/lib64/libcudadevrt.a";
    if (exists(fn)) {
      devrtPath = fn;
    }
  }

  // dl_iterate_phdr(
  //     [](struct dl_phdr_info* info, size_t size, void* data) {
  //       Function<void(struct dl_phdr_info*)>((FunctionPointer)data)(info);
  //       return 0;
  //     },
  //     Function<void(struct dl_phdr_info*)>([&](struct dl_phdr_info* info) {
  //       fmt::printf("%s\n", info->dlpi_name);
  //     }).release());

  if (!exists(devrtPath)) {
    std::vector<std::string> modules;
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t size, void* data) {
          Function<void(struct dl_phdr_info*)>((FunctionPointer)data)(info);
          return 0;
        },
        Function<void(struct dl_phdr_info*)>([&](struct dl_phdr_info* info) {
          modules.push_back(info->dlpi_name);
        }).release());
    std::reverse(modules.begin(), modules.end());
    for (auto path : modules) {
      auto slash = path.rfind('/');
      if (slash != std::string::npos) {
        path.resize(slash);
      }
      path += "libcudadevrt.a";
      if (exists(path)) {
        devrtPath = path;
        break;
      }
    }
  }

  if (!exists(devrtPath)) {
    fmt::fprintf(stderr, "Cannot find libcudadevrt.a (at compile time, it was at %s)\n", devrtPath);
    fflush(stderr);
  }

  fmt::printf("devrtPath is %s\n", devrtPath);

  CUlinkState linkState;
  CHECK_CU(cuLinkCreate(0, nullptr, nullptr, &linkState));
  CHECK_CU(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, devrtPath.c_str(), 0, nullptr, nullptr));
  // CHECK_CU(cuLinkAddData(
  //     linkState, CU_JIT_INPUT_PTX, ptx.data(), ptx.size(), cuFilename.c_str(), 0, nullptr, nullptr));
  CHECK_CU(cuLinkAddData(
      linkState, CU_JIT_INPUT_CUBIN, cubin.data(), cubin.size(), cuFilename.c_str(), 0, nullptr, nullptr));
  void* linkedCubin = nullptr;
  size_t linkedCubinSize = 0;
  CHECK_CU(cuLinkComplete(linkState, &linkedCubin, &linkedCubinSize));

  CHECK_CU(cuModuleLoadDataEx(&cuModule, linkedCubin, 0, nullptr, nullptr));

  for (auto& v : functions) {
    CHECK_CU(cuModuleGetFunction(v.first, cuModule, v.second.c_str()));

    int numRegs = 0;
    CHECK_CU(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, *v.first));
    int maxThreadsPerBlock = 0;
    CHECK_CU(cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, *v.first));
    int localBytes = 0;
    CHECK_CU(cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, *v.first));
    fmt::printf("kernel %s uses %d registers\n", v.second, numRegs);
    fmt::printf("kernel %s max threads per block: %d\n", v.second, maxThreadsPerBlock);
    fmt::printf("kernel %s local bytes: %d\n", v.second, localBytes);
  }

  CHECK_CU(cuLinkDestroy(linkState));
}

} // namespace moodist
