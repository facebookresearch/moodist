
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
  for (auto cuModule : cuModules) {
    cuModuleUnload(cuModule);
  }
}

std::string Kernels::emitCopySeq(
    std::vector<std::string> sources, std::vector<std::string> destinations, std::string bytes, size_t gridSize,
    size_t blockSize, std::string threadIndex, std::string blockIndex) {
  CHECK(sources.size() == destinations.size());
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
    if ($unroll != 1) {
      dynamicBlockIndex = (dynamicBlockIndex + %count) % $gridSize;
    }
    %i = %blockIndex;
    assert(%offset + $nbytes * %count <= %bytes);
    if ($cond < %count) {
      $preloop
      while ($cond < %count) {
        $loop
      }
      $postloop
      if ($unroll != 1 && %count < %numBlocks) {
        %numBlocks -= %count;
        %blockIndex -= %count;
      }
    } else {
      if ($unroll != 1 && %count < %numBlocks) {
        %numBlocks -= %count;
        %blockIndex -= %count;
      }
    }
    %offset += $nbytes * %count;
    assert(%offset <= %bytes);
    if (%offset == %bytes) {
      return dynamicBlockIndex;
    }
  )",
        "$nbytes", nbytes, "$cond", condi);
    std::string preloop;
    std::string loop;
    std::string postloop;
    if (unroll != 1) {
      preloop += "__syncwarp();\n";
      loop += "__syncwarp();\n";
    }
    std::vector<std::string> srcPtr;
    std::vector<std::string> dstPtr;
    for (size_t n = 0; n != destinations.size(); ++n) {
      srcPtr.push_back(getVar("src_ptr"));
      dstPtr.push_back(getVar("dst_ptr"));
    }
    for (int i = 0; i != unroll; ++i) {
      for (size_t n = 0; n != destinations.size(); ++n) {
        auto& src = sources[n];
        auto& dst = destinations[n];
        std::string temp = getVar("temporary");
        if (i == 0) {
          std::string srcptr = replace(
              R"($ptr = $src + %offset + $loopbytes * %i + $typesize * %threadIndex;
                )",
              "$ptr", srcPtr[n], "$src", src);
          preloop += "uintptr_t " + srcptr;
          loop += srcptr;

          std::string dstptr = replace(
              R"($ptr = $dst + %offset + $loopbytes * (%i - %numBlocks) + $typesize * %threadIndex;
                )",
              "$ptr", dstPtr[n], "$dst", dst);
          preloop += replace("uintptr_t $ptr;\n", "$ptr", dstPtr[n]);
          loop += dstptr;
          postloop += dstptr;
        }
        std::string read = replace(
            R"($temp = __ldcv(($type*)(void*)($ptr + $typesize * $blockSize * $i));
                )",
            "$temp", temp, "$ptr", srcPtr[n], "$i", i);
        if (i == unroll - 1 && n == destinations.size() - 1) {
          read += "%i += %numBlocks;\n";
        }
        preloop += type + " " + read;
        std::string write = replace(
            R"(__stwt(($type*)(void*)($ptr + $typesize * $blockSize * $i), $temp);
                )",
            "$ptr", dstPtr[n], "$i", i, "$temp", temp);
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
  uint32_t %blockIndex = $blockIndex;
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
  copy1 += genCopy("uint4", 16, 16);
  copy1 += genCopy("uint4", 16, 8);
  copy1 += genCopy("uint4", 16, 4);
  copy1 += genCopy("uint4", 16, 2);
  copy1 += genCopy("uint4", 16, 1);
  copy1 += genCopy("unsigned char", 1, 1);
  s = replace(s, "$bytes", bytes, "$defs", defs, "$copy1", copy1);
  s = replace(
      s, "$threadIndex", threadIndex, "$blockIndex", blockIndex, "$gridSize", gridSize, "$blockSize", blockSize);
  s = makeVars(s);
  return s;
}

std::string Kernels::emitReduceFunctionSeq(
    std::string sourcetype, size_t sourcetypesize, size_t nsources, std::string op, size_t gridSize, size_t blockSize,
    std::string threadIndex, std::string blockIndex) {
  CHECK(sourcetype != "");
  std::string sourcesargs;
  for (size_t i = 0; i != nsources; ++i) {
    if (!sourcesargs.empty()) {
      sourcesargs += ", ";
    }
    sourcesargs += replace("const $type* source_$i", "$type", sourcetype, "$i", i);
  }
  std::string addressesdefs;
  std::string dst = "dst";
  addressesdefs += "uintptr_t dst = (uintptr_t)destination;\n";
  std::vector<std::string> addresses;
  for (size_t i = 0; i != nsources; ++i) {
    addresses.push_back(fmt::sprintf("src%d", i));
    addressesdefs += replace("uintptr_t src$i = (uintptr_t)source_$i;\n", "$i", i);
  }
  addressesdefs = replace(addressesdefs, "$type", sourcetype);
  auto reduceopcode = [&](auto& names) {
    if (op == "rsum") {
      return fmt::to_string(fmt::join(names, " + "));
    } else if (op == "rmin") {
      return replace("min($args)", "$args", fmt::to_string(fmt::join(names, ", ")));
    } else if (op == "rmax") {
      return replace("max($args)", "$args", fmt::to_string(fmt::join(names, ", ")));
    } else if (op == "ravg") {
      fatal("fixme: avg is broken");
      std::string s;
      for (auto& v : names) {
        if (!s.empty()) {
          s += " + ";
        }
        if (sourcetypesize > 4) {
          s += replace("($type)($v * (1.0 / $size))", "$v", v, "$type", sourcetype);
        } else {
          s += replace("($type)($v * (1.0f / $size))", "$v", v, "$type", sourcetype);
        }
      }
      return s;
    }
    fatal("unknown reduce op %s", op);
  };
  std::function<std::string(std::string, std::string, std::vector<std::string>)> getreducecode =
      [&](std::string datatype, std::string result, std::vector<std::string> inputs) {
        if (datatype == sourcetype) {
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
        } else if (datatype == "ulonglong2") {
          std::string s;
          for (auto& field : {"x", "y"}) {
            auto inputs2 = inputs;
            for (std::string& v : inputs2) {
              v += ".";
              v += field;
            }
            s += getreducecode("uint64_t", result + "." + field, inputs2);
            s += "\n";
          }
          return s;
        } else if (datatype == "uint32_t") {
          if (sourcetype == "float") {
            for (auto& v : inputs) {
              v = replace("__uint_as_float($v)", "$v", v);
            }
            return replace("$result = __float_as_uint($op);", "$result", result, "$op", reduceopcode(inputs));
          }
          if (sourcetype == "int32_t") {
            return replace("$result = $op;", "$result", result, "$op", reduceopcode(inputs));
          }
          if (sourcetype == "bf16") {
            std::vector<std::string> inputslo;
            std::vector<std::string> inputshi;
            for (auto& v : inputs) {
              inputslo.push_back(replace("convert<bf16>((uint16_t)(($v) & 0xffff))", "$v", v));
              inputshi.push_back(replace("convert<bf16>((uint16_t)(($v) >> 16))", "$v", v));
            }
            return replace(
                R"(
              $result = (uint32_t)convert<uint16_t>($lo) | ((uint32_t)convert<uint16_t>($hi) << 16);)",
                "$result", result, "$lo", reduceopcode(inputslo), "$hi", reduceopcode(inputshi));
          }
        } else if (datatype == "uint64_t") {
          if (sourcetype == "int64_t") {
            return replace("$result = $op;", "$result", result, "$op", reduceopcode(inputs));
          }
          if (sourcetype == "double") {
            for (auto& v : inputs) {
              v = replace("__longlong_as_double($v)", "$v", v);
            }
            return replace("$result = __double_as_longlong($op);", "$result", result, "$op", reduceopcode(inputs));
          }
        } else if (datatype == "uint16_t") {
          if (sourcetype == "bf16") {
            for (auto& v : inputs) {
              v = replace("convert<bf16>($v)", "$v", v);
            }
            return replace("$result = convert<uint16_t>($op);", "$result", result, "$op", reduceopcode(inputs));
          }
        }
        fatal("unhandled reduce type %s for datatype %s", sourcetype, datatype);
      };
  auto generate = [&](std::string datatype, int datatypesize, int unroll) {
    CHECK(datatypesize % sourcetypesize == 0);
    std::string nbytes;
    std::string condi;
    std::string s;
    if (unroll == 1) {
      nbytes = "$datatypesize";
      condi = "$blockSize * %i + %threadIndex";
    } else {
      nbytes = "$loopbytes";
      condi = "%i";
    }
    s = replace(
        R"(
    %count = (%bytes - %offset) / $nbytes;
    if ($unroll != 1) {
      dynamicBlockIndex = (dynamicBlockIndex + %count) % $gridSize;
    }
    %i = %blockIndex;
    assert(%offset + $nbytes * %count <= %bytes);
    if ($cond < %count) {
      $preloop
      while ($cond < %count) {
        $loop
      }
      $postloop
      if ($unroll != 1 && %count < %numBlocks) {
        %numBlocks -= %count;
        %blockIndex -= %count;
      }
    } else {
      if ($unroll != 1 && %count < %numBlocks) {
        %numBlocks -= %count;
        %blockIndex -= %count;
      }
    }
    %offset += $nbytes * %count;
    assert(%offset <= %bytes);
    if (%offset == %bytes) {
      return dynamicBlockIndex;
    }
  )",
        "$nbytes", nbytes, "$cond", condi);
    std::string preloop;
    std::string loop;
    std::string postloop;
    if (unroll != 1) {
      preloop += "__syncwarp();\n";
      loop += "__syncwarp();\n";
    }
    for (int i = 0; i != unroll; ++i) {
      std::vector<std::string> tempnames;
      std::string reads;
      for (size_t n = 0; n != addresses.size(); ++n) {
        auto& src = addresses[n];
        std::string temp = getVar("temporary");
        tempnames.push_back(temp);
        std::string read = replace(
            R"($temp = __ldcv(($datatype*)(void*)($src + %offset + $loopbytes * %i + $datatypesize * $blockSize * $i + $datatypesize * %threadIndex));
                )",
            "$temp", temp, "$src", src, "$i", i);
        if (i == unroll - 1 && n == addresses.size() - 1) {
          read += "%i += %numBlocks;\n";
        }
        preloop += datatype + " " + read;
        reads += read;
      }
      std::string result = getVar("result");
      std::string reducecode = getreducecode(datatype, result, tempnames);
      preloop += replace("$datatype $result;\n", "$result", result);
      std::string write = replace(
          R"($reducecode
              __stwt(($datatype*)(void*)($dst + %offset + $loopbytes * (%i - %numBlocks) + $datatypesize * $blockSize * $i + $datatypesize * %threadIndex), $result);
                )",
          "$result", result, "$dst", dst, "$i", i, "$reducecode", reducecode);
      postloop += write;
      loop += write;
      loop += reads;
    }
    size_t loopbytes = datatypesize * unroll * blockSize;
    s = replace(s, "$preloop", preloop, "$postloop", postloop, "$loop", loop, "$loopbytes", loopbytes);
    s = replace(
        s, "$datatype", datatype, "$datatypesize", datatypesize, "$sourcetype", sourcetype, "$sourcetypesize",
        sourcetypesize, "$loopbytes", loopbytes, "$unroll", unroll);
    return s;
  };
  std::string reduce1;
  if (sourcetypesize == 8) {
    reduce1 += generate("ulonglong2", 16, 16);
    reduce1 += generate("ulonglong2", 16, 8);
    reduce1 += generate("ulonglong2", 16, 4);
    reduce1 += generate("ulonglong2", 16, 2);
    reduce1 += generate("ulonglong2", 16, 1);
  } else {
    reduce1 += generate("uint4", 16, 16);
    reduce1 += generate("uint4", 16, 8);
    reduce1 += generate("uint4", 16, 4);
    reduce1 += generate("uint4", 16, 2);
    reduce1 += generate("uint4", 16, 1);
  }
  std::string singletype = sourcetype;
  if (sourcetype == "bf16") {
    singletype = "uint16_t";
  }
  reduce1 += generate(singletype, sourcetypesize, 1);
  std::string s = replace(
      R"(
template<>
__device__ uint32_t reduce_n$nsources<$typename, $op>(uint32_t dynamicBlockIndex, size_t size, $typename* destination, $sourcesargs) {
  $addressesdefs
  const size_t %bytes = $sourcetypesize * size;
  const uint32_t %threadIndex = $threadIndex;
  static_assert($gridSize % $blocksPerSm == 0);
  uint32_t %blockIndex = $blockIndex;
  uint32_t %numBlocks = $gridSize;
  assert(%blockIndex < %numBlocks);
  size_t %offset = 0;
  size_t %count = 0;
  size_t %i;
  $reduce1
  assert(false);
  return dynamicBlockIndex;
}
  )",
      "$sourcesargs", sourcesargs, "$op", op, "$nsources", nsources, "$typename", sourcetype, "$sourcetypesize",
      sourcetypesize, "$reduce1", reduce1, "$addressesdefs", addressesdefs);
  s = replace(
      s, "$gridSize", gridSize, "$blockSize", blockSize, "$blockIndex", blockIndex, "$threadIndex", threadIndex);
  s = makeVars(s);
  return s;
}

void Kernels::compile(int flags, std::string compileType, std::string compileReduction) {
  auto start = std::chrono::steady_clock::now();
  CUdevice& cuDevice = group->cuDevice;
  CUcontext& cuContext = group->cuContext;

  int computeMajor = 0;
  int computeMinor = 0;

  int major = 6;
  int minor = 0;
  CHECK_NVRTC(nvrtcVersion(&major, &minor));
  log.verbose("nvrtc version is %d %d\n", major, minor);

  int driverVersion = 5000;
  CHECK_CU(cuDriverGetVersion(&driverVersion));

  log.verbose("cuda driver version is %d\n", driverVersion);

  CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  log.verbose("cuda device compute capability is %d.%d\n", computeMajor, computeMinor);

  int cudaWarpSize = 0;
  int cudaMaxSharedMemory = 0;
  int maxThreadsPerSm = 0;
  int smCount = 0;

  CHECK_CU(cuDeviceGetAttribute(&cudaWarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&cudaMaxSharedMemory, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice));

  CHECK_CU(cuDeviceGetAttribute(&maxThreadsPerSm, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice));

  log.verbose("warp size: %d\nmax shared memory: %d\n", cudaWarpSize, cudaMaxSharedMemory);
  log.verbose("multiprocessor count: %d\nmax threads per multiprocessor: %d\n", smCount, maxThreadsPerSm);

  const auto& cpuInBuffer = group->cpuInBuffer;
  const auto& cpuOutBuffer = group->cpuOutBuffer;

  std::string source;
  std::vector<std::pair<CUfunction*, std::string>> functions;

  source = R"z(
using uintptr_t =  unsigned long;
using uint64_t = unsigned long;
using uint32_t = unsigned int;
using uint16_t = unsigned short;
using uint8_t = unsigned char;
using int32_t = int;
using int64_t = long;
using int16_t = short;
using int8_t = char;

struct bf16 {
  unsigned short x;
  __device__ bf16 operator+(bf16 n) const {
    bf16 r;
    //asm("add.bf16 %0, %1, %2;" : "=h"(r.x) : "h"(x), "h"(n.x));
    uint16_t one = 0x3f80;
    asm("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(r.x) : "h"(x), "h"(one), "h"(n.x));
    return r;
  }
};
static_assert(alignof(bf16) == 2);
static_assert(sizeof(bf16) == 2);

template<typename Out, typename In>
__device__ Out convert(In value);

template<>
__device__ bf16 convert<bf16, uint16_t>(uint16_t value) {
  return bf16{value};
}

template<>
__device__ uint16_t convert<uint16_t, bf16>(bf16 value) {
  return value.x;
}

struct rsum {};
struct rmin {};
struct rmax {};
struct ravg {};

namespace {

__device__ void syncthreads() {
  asm volatile ("barrier.sync 0;" :: );
}

__device__ uint32_t smid() {
  uint32_t r;
  asm("mov.u32 %%0, %%smid;" : "=r"(r));
  return r;
}

$constantDefs

__device__ size_t stepValueDeviceChunkIndex(size_t concurrencyIndex, size_t index, size_t device, size_t chunk) {
  return $maxChunks * $maxDevices * $size * concurrencyIndex + $maxChunks * $size * device + $size * chunk + index;
}

__device__ size_t stepValueChunkIndex(size_t concurrencyIndex, size_t index, size_t chunk) {
  return $maxChunks * $size * concurrencyIndex + $size * chunk + index;
}

__device__ uint32_t copy_impl(uint32_t dynamicBlockIndex, void* __restrict__ dst, const void* __restrict__ src, size_t bytes) {
  $copyCode
  return dynamicBlockIndex;
}

template<typename T, typename R>
__device__ uint32_t reduce_n2(uint32_t dynamicBlockIndex, size_t numel, T* __restrict__ dst, const T* __restrict__ src1, const T* __restrict__ src2);

)z";

  std::string constantDefs;

  const auto& peerCudaProxyReady = group->peerCudaProxyReady;
  std::vector<std::string> peerCudaProxyReadyArray;
  for (auto& v : peerCudaProxyReady) {
    peerCudaProxyReadyArray.push_back(replace("$base", "$base", v.base));
  }
  constantDefs += fmt::sprintf(
      "__constant__ uintptr_t peerCudaProxyReady[%d] = {%s};", peerCudaProxyReady.size(),
      fmt::to_string(fmt::join(peerCudaProxyReadyArray, ", ")));

  source = replace(source, "$constantDefs", constantDefs);

  gridSize = 16;
  blockSize = 256;
  size_t blocksPerSm = 1;

  if (group->ipcRanks.size() == group->size - 1) {
    gridSize = 32;
  }
  if (group->ipcRanks.size() == 0) {
    gridSize = 4;
  }

  source = replace(
      source, "$copyCode",
      emitCopySeq({"src"}, {"dst"}, "bytes", gridSize * blockSize / 32, 32, "threadIdx.x % 32u", "dynamicBlockIndex"));

  if (flags & CompileReduceScatter) {
    auto i = std::find(supportedTypes.begin(), supportedTypes.end(), compileType);
    CHECK(i != supportedTypes.end());
    source += emitReduceFunctionSeq(
        compileType, supportedTypeSizes[i - supportedTypes.begin()], 2, compileReduction, gridSize * blockSize / 32, 32,
        "threadIdx.x % 32u", "dynamicBlockIndex");
  }

  source += R"(

template<typename T, typename R>
__device__ uint32_t reduce2(uint32_t dynamicBlockIndex, T* __restrict__ dst, const T* __restrict__ src1, const T* __restrict__ src2, size_t numel) {
  return reduce_n2<T, R>(dynamicBlockIndex, numel, dst, src1, src2);
}
)";

  auto waitFor32 = [&](std::string address, std::string value) {
    return replace(
        R"(while (*(volatile uint32_t*)($ptr) < $value);
    )",
        "$ptr", address, "$value", value);
  };

  std::string writes1;
  std::string writes2;
  std::string waits;
  for (size_t i : group->peerIndices) {
    waits += waitFor32(concurrencyIndex(group->cudaStepValue, sizeof(uint32_t) * group->ipcRanks[i]), "stepValue");
    writes1 += replace(
        R"(
      ((volatile uintptr_t*)$addrPtr)[0] = params.peerInputAddresses[$i];
      ((volatile uintptr_t*)$addrPtr)[1] = params.peerOutputAddresses[$i];
    )",
        "$addrPtr",
        concurrencyIndex(group->peerCudaPeerAddresses[i], (sizeof(uintptr_t) * 2 * group->peerMyRemoteIndex[i])), "$i",
        i);
    writes2 += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", concurrencyIndex(group->peerCudaStepValue[i], sizeof(uint32_t) * rank));
  }

  std::string copyDoneAllCode;
  std::string waitForCopyDones;
  for (size_t i : group->peerIndices) {
    copyDoneAllCode += replace(
        R"(
  *(volatile uint32_t*)$ptr = stepValue;
      )",
        "$ptr", concurrencyIndex(group->peerCudaCopyDone[i], sizeof(uint32_t) * group->peerMyRemoteIndex[i]));
    waitForCopyDones += replace(
        R"(
  while (*(volatile uint32_t*)$ptr < stepValue);
      )",
        "$i", i, "$ptr", concurrencyIndex(group->cudaCopyDone, sizeof(uint32_t) * i));
  }

  source += replace(
      R"(
template<typename Params>
__device__ void generic_entry_local(const Params& params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  $writes1
  __threadfence_system();
  $writes2
  $waits
}
__device__ void generic_exit_local(uint32_t stepValue, uint32_t concurrencyIndex) {
  $copyDoneAllCode
  $waitForCopyDones
}
template<typename Params>
__device__ void generic_entry(const Params& params) {
  [[maybe_unused]] const uint32_t stepValue = params.stepValue;
  [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  cpuIn[0] = stepValue;
  $writes1
  __threadfence_system();
  $writes2
  $waits
}
__device__ void generic_exit(uint32_t stepValue, uint32_t concurrencyIndex) {
  $copyDoneAllCode
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  $waitForCopyDones
  while (cpuOut[0] < stepValue);
}

__device__ uint32_t entryCounter[$maxConcurrency];
__device__ uint32_t exitCounter[$maxConcurrency];

template<typename Params>
__device__ void grid_generic_entry_local(const Params& params) {
  if (threadIdx.x == 0) {
    [[maybe_unused]] const uint32_t stepValue = params.stepValue;
    [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
    if (atomicInc(&entryCounter[concurrencyIndex], $gridSize - 1) == 0) {
      $writes1
      __threadfence_system();
      $writes2
    }
    $waits
  }
  syncthreads();
}
__device__ void grid_generic_exit_local(uint32_t stepValue, uint32_t concurrencyIndex) {
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) == $gridSize - 1) {
    $copyDoneAllCode
    $waitForCopyDones
  }
}
template<typename Params>
__device__ void grid_generic_entry(const Params& params) {
  if (threadIdx.x == 0) {
    [[maybe_unused]] const uint32_t stepValue = params.stepValue;
    [[maybe_unused]] const uint32_t concurrencyIndex = params.concurrencyIndex;
    volatile uint32_t* __restrict__ cpuIn = $cpuIn;
    if (atomicInc(&entryCounter[concurrencyIndex], $gridSize - 1) == 0) {
      cpuIn[0] = stepValue;
      $writes1
      __threadfence_system();
      $writes2
    }
    $waits
  }
  syncthreads();
}
__device__ void grid_generic_exit(uint32_t stepValue, uint32_t concurrencyIndex) {
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) == $gridSize - 1) {
    $copyDoneAllCode
    volatile uint32_t* __restrict__ cpuIn = $cpuIn;
    volatile uint32_t* __restrict__ cpuOut = $cpuOut;
    cpuIn[0] = stepValue + 1;
    while (cpuOut[0] < stepValue);
    $waitForCopyDones
  }
}

__device__ void grid_generic_exit_no_recv(uint32_t stepValue, uint32_t concurrencyIndex) {
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) == $gridSize - 1) {
    $copyDoneAllCode
    volatile uint32_t* __restrict__ cpuIn = $cpuIn;
    volatile uint32_t* __restrict__ cpuOut = $cpuOut;
    cpuIn[0] = stepValue + 1;
    while (cpuOut[0] < stepValue);
    $waitForCopyDones
  }
}

}

extern "C" __global__ void broadcast(uint32_t stepValue, uint32_t concurrencyIndex) {
  volatile uint32_t* __restrict__ cpuIn = $cpuIn;
  volatile uint32_t* __restrict__ cpuOut = $cpuOut;
  cpuIn[0] = stepValue + 1;
  while (cpuOut[0] < stepValue);
}
  )",
      "$writes1", writes1, "$writes2", writes2, "$waits", waits, "$waitForCopyDones", waitForCopyDones,
      "$copyDoneAllCode", copyDoneAllCode);

  auto fn = [&](CUfunction& ref, std::string name) { functions.emplace_back(&ref, name); };

  if (flags & CompileAllGather) {
    source += group->allGather->generate();
    fn(cuAllGatherLocal, "allgather_local");
    fn(cuAllGather, "allgather");
    fn(cuAllGatherNoLocal, "allgather_no_local");
  }
  if (flags & CompileReduceScatter) {
    source += group->reduceScatter->generate({compileType}, {compileReduction});

    auto i = std::find(supportedTypes.begin(), supportedTypes.end(), compileType);
    auto i2 = std::find(supportedReductions.begin(), supportedReductions.end(), compileReduction);
    CHECK(i != supportedTypes.end());
    CHECK(i2 != supportedReductions.end());
    size_t t = i - supportedTypes.begin();
    size_t r = i2 - supportedReductions.begin();
    fn(cuReduceScatterLocal[t][r],
       replace("reduce_scatter_local_$type_$red", "$type", compileType, "$red", compileReduction));
    fn(cuReduceScatter[t][r], replace("reduce_scatter_$type_$red", "$type", compileType, "$red", compileReduction));
  }

  fn(cuBroadcast, "broadcast");

  source = replace(source, "$launchBounds", "__launch_bounds__($blockSize, $blocksPerSm)");
  source = replace(source, "$gridSize", gridSize, "$blockSize", blockSize, "$blocksPerSm", blocksPerSm);

  source = replace(
      source, "$cpuOut", cast("volatile uint32_t*", concurrencyIndex(cpuOutBuffer)), "$cpuIn",
      cast("volatile uint32_t*", concurrencyIndex(cpuInBuffer)));
  source = replace(
      source, "$rank", rank, "$size", size, "$maxConcurrency", Group::maxConcurrency, "$maxChunks", Group::maxChunks,
      "$maxDevices", Group::maxDevices);

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  std::string cuFilename = fmt::sprintf("moodist-kernels-rank%d.cu", rank);

  auto boolenv = [&](const char* name) {
    const char* c = std::getenv(name);
    if (!c) {
      return false;
    }
    return !strcmp(c, "1");
  };

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    FILE* f = fopen(cuFilename.c_str(), "wb");
    if (f) {
      fwrite(source.data(), source.size(), 1, f);
      fclose(f);
      log.info("source dumped to %s\n", cuFilename);
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
  // options.push_back("--relocatable-device-code=true");
  // options.push_back("--maxrregcount=32");
  // options.push_back("-G");
  // options.push_back("--dopt=on");
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
      log.verbose("success with %s\n", archOptions[i].second);
    }
  }
  if (error != NVRTC_SUCCESS) {
    log.error("Failed to compile--\n%s\n", source.c_str());
    size_t logSize = 0;
    std::string logstr;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &logSize));
    logstr.resize(logSize);
    CHECK_NVRTC(nvrtcGetProgramLog(program, logstr.data()));
    log.error("%s\n", logstr);

    CHECK_NVRTC(error);
  } else {
    size_t logSize = 0;
    std::string logstr;
    CHECK_NVRTC(nvrtcGetProgramLogSize(program, &logSize));
    logstr.resize(logSize);
    CHECK_NVRTC(nvrtcGetProgramLog(program, logstr.data()));
    for (char& c : logstr) {
      if (c < 32 || c >= 127) {
        if (c != 10 && c != 13) {
          c = 32;
        }
      }
    }
    log.debug("compile log\n---\n%s\n", logstr);
  }

  size_t ptxSize = 0;
  CHECK_NVRTC(nvrtcGetPTXSize(program, &ptxSize));
  std::vector<char> ptx;
  ptx.resize(ptxSize);
  CHECK_NVRTC(nvrtcGetPTX(program, ptx.data()));

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    std::string fn = fmt::sprintf("moodist-kernels-rank%d-ptx.s", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(ptx.data(), ptx.size(), 1, f);
      fclose(f);
      log.info("ptx dumped to %s\n", fn);
    }
  }

  // CHECK_NVRTC(nvrtcDestroyProgram(&program));

  // CHECK_CU(cuModuleLoadDataEx(&op.cuModule, ptx.data(), 0, nullptr, nullptr));

  size_t cubinSize = 0;
  CHECK_NVRTC(nvrtcGetCUBINSize(program, &cubinSize));
  std::vector<char> cubin;
  cubin.resize(cubinSize);
  log.debug("cubin size is %d\n", cubin.size());
  CHECK_NVRTC(nvrtcGetCUBIN(program, cubin.data()));

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    std::string fn = fmt::sprintf("moodist-kernels-rank%d.o", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      log.info("cubin dumped to %s\n", fn);
    }
  }

  CHECK_NVRTC(nvrtcDestroyProgram(&program));

  if (false) {
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
      log.error("Cannot find libcudadevrt.a (at compile time, it was at %s)\n", devrtPath);
    }

    log.verbose("devrtPath is %s\n", devrtPath);
  }

  // CUlinkState linkState;
  // CHECK_CU(cuLinkCreate(0, nullptr, nullptr, &linkState));
  // CHECK_CU(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, devrtPath.c_str(), 0, nullptr, nullptr));
  // // CHECK_CU(cuLinkAddData(
  // //     linkState, CU_JIT_INPUT_PTX, ptx.data(), ptx.size(), cuFilename.c_str(), 0, nullptr, nullptr));
  // CHECK_CU(cuLinkAddData(
  //     linkState, CU_JIT_INPUT_CUBIN, cubin.data(), cubin.size(), cuFilename.c_str(), 0, nullptr, nullptr));
  // void* linkedCubin = nullptr;
  // size_t linkedCubinSize = 0;
  // CHECK_CU(cuLinkComplete(linkState, &linkedCubin, &linkedCubinSize));

  // CUmodule cuModule = nullptr;
  // CHECK_CU(cuModuleLoadDataEx(&cuModule, linkedCubin, 0, nullptr, nullptr));
  // cuModules.push_back(cuModule);

  CUmodule cuModule = nullptr;
  CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));
  cuModules.push_back(cuModule);

  for (auto& v : functions) {
    log.info("%s\n", v.second);
    CHECK_CU(cuModuleGetFunction(v.first, cuModule, v.second.c_str()));

    int numRegs = 0;
    CHECK_CU(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, *v.first));
    int maxThreadsPerBlock = 0;
    CHECK_CU(cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, *v.first));
    int localBytes = 0;
    CHECK_CU(cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, *v.first));
    log.verbose("kernel %s uses %d registers\n", v.second, numRegs);
    log.verbose("kernel %s max threads per block: %d\n", v.second, maxThreadsPerBlock);
    log.verbose("kernel %s local bytes: %d\n", v.second, localBytes);
  }

  // CHECK_CU(cuLinkDestroy(linkState));

  log.info("compile took %gs", seconds(std::chrono::steady_clock::now() - start));
}

} // namespace moodist
