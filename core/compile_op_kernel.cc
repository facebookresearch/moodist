// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "compile_op_kernel.h"
#include "common.h"
#include "group.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace moodist {

CompileOpKernels::CompileOpKernels(Group* group) : group(group) {
  const char* env = std::getenv("MOODIST_COPY_KERNEL");
  if (env) {
    if (!strcmp(env, "v1")) {
      version = 1;
    } else if (!strcmp(env, "v2")) {
      version = 2;
    }
  }
}

CompileOpKernels::~CompileOpKernels() {
  if (cuModule) {
    cuModuleUnload(cuModule);
  }
  if (cuMulticastModule) {
    cuModuleUnload(cuMulticastModule);
  }
}

// Helper for string template substitution, matching CollectiveBase::replace pattern
static std::string replace(std::string str, std::vector<std::pair<std::string, std::string>> vars) {
  std::sort(vars.begin(), vars.end(), [](auto& a, auto& b) {
    return a.first.size() > b.first.size();
  });
  size_t pos = 0;
  while (pos != str.size()) {
    std::string key;
    std::string value;
    size_t index = std::string::npos;
    for (auto& v : vars) {
      auto i = str.find(v.first, pos);
      if (i == std::string::npos) {
        continue;
      }
      if (i < index) {
        key = v.first;
        value = v.second;
        index = i;
      }
    }
    if (index == std::string::npos) {
      break;
    }
    str.replace(index, key.size(), value);
    pos = index + value.size();
  }
  return str;
}

template<typename Vars>
static void extractvars(Vars&) {}

template<typename Vars, typename Key, typename Value, typename... Args>
static void extractvars(Vars& vars, Key key, Value value, Args... args) {
  if constexpr (std::is_integral_v<Value>) {
    if (value > 65536) {
      vars.emplace_back(key, fmt::sprintf("%#x", value));
    } else {
      vars.emplace_back(key, std::to_string(value));
    }
  } else {
    vars.emplace_back(key, value);
  }
  extractvars(vars, args...);
}

template<typename... Args>
static std::string replace(std::string str, Args... args) {
  static_assert(sizeof...(Args) % 2 == 0);
  std::vector<std::pair<std::string, std::string>> vars;
  extractvars(vars, args...);
  return replace(str, vars);
}

static std::string autoindent(const std::string& s) {
  std::string r;
  const char* c = s.c_str();
  int level = 0;
  for (int line = 1; *c; ++line) {
    while (*c == ' ') {
      ++c;
    }
    const char* p = c;
    if (*c == '}') {
      --level;
      ++c;
    }
    for (int i = 0; i < level; ++i) {
      r += "  ";
    }
    if (*c == '/' && c[1] == '/') {
      while (*c && *c != '\n') {
        ++c;
      }
    } else {
      while (*c && *c != '\n') {
        if (*c == '{') {
          ++level;
        } else if (*c == '}') {
          --level;
        }
        ++c;
      }
    }
    if (*c) {
      ++c;
    }
    r.append(p, c - p);
  }
  return r;
}

static std::string addLineCountComments(const std::string& s) {
  std::string r;
  const char* c = s.c_str();
  bool isInComment = false;
  for (int line = 1; *c; ++line) {
    if (isInComment) {
      r += fmt::sprintf(" * % 5d *  ", line);
    } else {
      r += fmt::sprintf("/* % 5d */ ", line);
    }
    const char* p = c;
    while (*c && *c != '\n') {
      if (c[0] == '/' && c[1] == '*') {
        isInComment = true;
      } else if (c[0] == '*' && c[1] == '/') {
        isInComment = false;
      }
      ++c;
    }
    if (*c) {
      ++c;
    }
    r.append(p, c - p);
  }
  return r;
}

static std::string concurrencyIndexExpr(uintptr_t base, size_t itembytes, size_t offset = 0) {
  return replace(
      "($base + $itembytes * concurrencyIndex + $offset)", "$base", base, "$itembytes", itembytes, "$offset", offset);
}

static std::string concurrencyIndexExpr(const AllocatedArray& arr, size_t offset = 0) {
  return concurrencyIndexExpr(arr.buffer.cudaPointer, arr.itembytes, offset);
}

static std::string concurrencyIndexExpr(const PeerArrayRef& arr, size_t offset = 0) {
  return concurrencyIndexExpr(arr.base, arr.itembytes, offset);
}

// Generate cascading unrolled uint4 copy loops as CUDA source.
// The generated code references local variables: i (size_t, chunk index),
// count (size_t, total chunks), numBlocks (uint32_t), loopBytes (size_t),
// tid16 (size_t), src and dst (uintptr_t).
static std::string emitCascadingCopy(const std::vector<int>& unrollFactors) {
  std::string code;
  for (int uf : unrollFactors) {
    if (uf == 1) {
      code += "while (i < count) {\n";
    } else {
      code += fmt::sprintf("while (i + %d * (size_t)numBlocks < count) {\n", uf - 1);
    }
    for (int j = 0; j < uf; j++) {
      code += fmt::sprintf(
          "uint4 v%d = __ldcv((const uint4*)(src + (i + %d * (size_t)numBlocks) * loopBytes + tid16));\n", j, j);
    }
    for (int j = 0; j < uf; j++) {
      code += fmt::sprintf("__stwt((uint4*)(dst + (i + %d * (size_t)numBlocks) * loopBytes + tid16), v%d);\n", j, j);
    }
    code += fmt::sprintf("i += %d * (size_t)numBlocks;\n", uf);
    code += "}\n";
  }
  return code;
}

void CompileOpKernels::compile() {
  auto start = std::chrono::steady_clock::now();
  CUdevice& cuDevice = group->cuDevice;

  int computeMajor = 0;
  int computeMinor = 0;

  int major = 6;
  int minor = 0;
  if (!loadNvrtc()) {
    throw std::runtime_error("NVRTC not available");
  }
  CHECK_NVRTC(nvrtcApi.version(&major, &minor));

  CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  size_t rank = group->rank;
  size_t size = group->size;
  const auto& peerIndices = group->peerIndices;

  // Build sync address code fragments (same pattern as kernels.cc)
  std::string stepValueWrites;
  std::string stepValueWaits;
  std::string copyDoneWrites;
  std::string copyDoneWaits;

  for (size_t i : peerIndices) {
    // Entry: write our stepValue into peer's stepValue array at our rank's slot
    stepValueWrites += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", concurrencyIndexExpr(group->peerCudaStepValue[i], sizeof(uint32_t) * rank));

    // Entry: wait for peer's stepValue at their rank's slot in our array
    stepValueWaits += replace(
        R"(while (*(volatile uint32_t*)($ptr) < stepValue);
    )",
        "$ptr", concurrencyIndexExpr(group->cudaStepValue, sizeof(uint32_t) * group->ipcRanks[i]));

    // Exit: write copyDone into peer's copyDone array at our slot
    copyDoneWrites += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", concurrencyIndexExpr(group->peerCudaCopyDone[i], sizeof(uint32_t) * group->peerMyRemoteIndex[i]));

    // Exit: wait for peer's copyDone at their slot in our array
    copyDoneWaits += replace(
        R"(
      while (*(volatile uint32_t*)$ptr < stepValue);
    )",
        "$ptr", concurrencyIndexExpr(group->cudaCopyDone, sizeof(uint32_t) * i));
  }

  // Generate source
  std::string source;

  source += R"z(
using uintptr_t = unsigned long;
using uint64_t = unsigned long;
using uint32_t = unsigned int;
using uint16_t = unsigned short;
using uint8_t = unsigned char;
using int32_t = int;
using int64_t = long;
using size_t = unsigned long;

namespace {

__device__ void syncthreads() {
  asm volatile ("barrier.sync 0;" :: );
}

struct CopyDescriptor {
  uintptr_t src;
  uintptr_t dst;
  uint32_t bytes;
};

struct CompileOpCopyParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  uint32_t numDescriptors;
  uint32_t _pad;
  CopyDescriptor descriptors[$kMaxCopyDescriptors];
};

__device__ uint32_t entryCounter[$maxConcurrency];
__device__ uint32_t exitCounter[$maxConcurrency];

)z";

  source =
      replace(source, "$kMaxCopyDescriptors", kMaxCopyDescriptors, "$maxConcurrency", (size_t)Group::maxConcurrency);

  // v1 kernel: dynamic block index distribution with cascading unrolled copy
  if (version == 1) {
    std::string cascadingCopy = emitCascadingCopy({32, 16, 8, 4, 2, 1});

    source += replace(
        R"z(
// Copy a region using all blocks cooperatively.
// dynamicBlockIndex rotates across descriptors so idle blocks from small
// descriptors become first-in-line for the next descriptor.
__device__ uint32_t copy_descriptor_block(
    uint32_t dynamicBlockIndex, uintptr_t src, uintptr_t dst, size_t bytes) {

  const uint32_t tid = threadIdx.x;
  const size_t tid16 = (size_t)tid * 16;
  const size_t loopBytes = (size_t)$blockSize * 16;
  const uint32_t numBlocks = $gridSize;
  const uint32_t blockIndex = dynamicBlockIndex;

  // Head: align src and dst to 16-byte boundary
  uint32_t srcMod = (uint32_t)(src & 15);
  uint32_t dstMod = (uint32_t)(dst & 15);
  if (srcMod == dstMod && srcMod != 0 && bytes >= 16) {
    uint32_t headBytes = 16 - srcMod;
    if (headBytes > bytes) headBytes = (uint32_t)bytes;
    for (uint32_t i = tid; i < headBytes; i += $blockSize) {
      *(uint8_t*)(dst + i) = *(const uint8_t*)(src + i);
    }
    src += headBytes;
    dst += headBytes;
    bytes -= headBytes;
  }

  bool aligned = ((src | dst) & 15) == 0;

  if (aligned && bytes >= loopBytes) {
    size_t count = bytes / loopBytes;
    dynamicBlockIndex = (dynamicBlockIndex + (uint32_t)count) % numBlocks;
    size_t i = blockIndex;

    $cascadingCopy

    size_t done = count * loopBytes;
    src += done;
    dst += done;
    bytes -= done;
  }

  // Remaining aligned uint4 elements (less than one block stride)
  if (aligned) {
    uint32_t remaining16 = (uint32_t)(bytes / 16);
    for (uint32_t i = tid; i < remaining16; i += $blockSize) {
      uint4 val = __ldcv((const uint4*)(src + (uintptr_t)i * 16));
      __stwt((uint4*)(dst + (uintptr_t)i * 16), val);
    }
    size_t done16 = (size_t)remaining16 * 16;
    src += done16;
    dst += done16;
    bytes -= done16;
  }

  // Tail or fully unaligned: byte copy
  for (uint32_t i = tid; i < (uint32_t)bytes; i += $blockSize) {
    *(uint8_t*)(dst + i) = *(const uint8_t*)(src + i);
  }

  return dynamicBlockIndex;
}

} // namespace

extern "C" __global__ void __launch_bounds__($blockSize, 1)
compile_op_copy(CompileOpCopyParameters params) {
  const uint32_t stepValue = params.stepValue;
  const uint32_t concurrencyIndex = params.concurrencyIndex;

  // Entry barrier: first block signals peers, all blocks wait for peers
  if (threadIdx.x == 0) {
    if (atomicInc(&entryCounter[concurrencyIndex], $gridSize - 1) == 0) {
      $stepValueWrites
      __threadfence_system();
    }
    $stepValueWaits
  }
  syncthreads();

  // Copy work: dynamic block index rotates across descriptors
  uint32_t dynamicBlockIndex = blockIdx.x;
  for (uint32_t d = 0; d < params.numDescriptors; d++) {
    dynamicBlockIndex = copy_descriptor_block(
        dynamicBlockIndex,
        params.descriptors[d].src,
        params.descriptors[d].dst,
        params.descriptors[d].bytes);
  }

  // Exit barrier: last block signals copyDone, waits for peers
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) == $gridSize - 1) {
    $copyDoneWrites
    $copyDoneWaits
  }
}
)z",
        "$cascadingCopy", cascadingCopy, "$stepValueWrites", stepValueWrites, "$stepValueWaits", stepValueWaits,
        "$copyDoneWrites", copyDoneWrites, "$copyDoneWaits", copyDoneWaits, "$gridSize", gridSize, "$blockSize",
        blockSize);
  }

  // v2 kernel: dynamic work distribution with global atomic counter
  if (version == 2) {
    source += replace(
        R"z(
__device__ uint32_t workCounter[$maxConcurrency];

// Prefix-sum array for mapping byte offset -> descriptor index
// Built in shared memory by first warp
__device__ void copy_region_v2(uintptr_t src, uintptr_t dst, uint32_t bytes, uint32_t offset, uint32_t len) {
  uint32_t tid = threadIdx.x % 32u;
  src += offset;
  dst += offset;

  uint32_t srcMod = (uint32_t)(src & 15);
  uint32_t dstMod = (uint32_t)(dst & 15);

  if (srcMod == dstMod) {
    // Same alignment: copy head bytes to reach 16-byte alignment
    uint32_t headBytes = 0;
    if (srcMod != 0 && len >= 16) {
      headBytes = 16 - srcMod;
      if (headBytes > len) headBytes = len;
    } else if (len < 16) {
      headBytes = len;
    }
    for (uint32_t i = tid; i < headBytes; i += 32) {
      *(uint8_t*)(dst + i) = *(const uint8_t*)(src + i);
    }
    src += headBytes;
    dst += headBytes;
    len -= headBytes;

    uint32_t bodyElements = len / 16;
    for (uint32_t i = tid; i < bodyElements; i += 32) {
      uint4 val = __ldcv((const uint4*)(src + (uintptr_t)i * 16));
      __stwt((uint4*)(dst + (uintptr_t)i * 16), val);
    }

    uint32_t tailOffset = bodyElements * 16;
    uint32_t tailBytes = len - tailOffset;
    for (uint32_t i = tid; i < tailBytes; i += 32) {
      *(uint8_t*)(dst + tailOffset + i) = *(const uint8_t*)(src + tailOffset + i);
    }
  } else {
    // Different alignment: byte-level copy
    for (uint32_t i = tid; i < len; i += 32) {
      *(uint8_t*)(dst + i) = *(const uint8_t*)(src + i);
    }
  }
}

} // namespace

extern "C" __global__ void __launch_bounds__($blockSize, 1)
compile_op_copy(CompileOpCopyParameters params) {
  const uint32_t stepValue = params.stepValue;
  const uint32_t concurrencyIndex = params.concurrencyIndex;

  // Build prefix sums in shared memory (each block builds independently)
  __shared__ uint64_t prefixSum[$kMaxCopyDescriptors + 1];

  if (threadIdx.x == 0) {
    prefixSum[0] = 0;
    for (uint32_t d = 0; d < params.numDescriptors; d++) {
      prefixSum[d + 1] = prefixSum[d] + params.descriptors[d].bytes;
    }
  }

  // Entry barrier
  if (threadIdx.x == 0) {
    if (atomicInc(&entryCounter[concurrencyIndex], $gridSize - 1) == 0) {
      workCounter[concurrencyIndex] = 0;
      $stepValueWrites
      __threadfence_system();
    }
    $stepValueWaits
  }
  syncthreads();

  uint64_t totalBytes = prefixSum[params.numDescriptors];
  const uint32_t chunkSize = 512;
  uint32_t totalChunks = (uint32_t)((totalBytes + chunkSize - 1) / chunkSize);

  // Each warp grabs work from global counter
  while (true) {
    uint32_t chunkIdx;
    if (threadIdx.x % 32 == 0) {
      chunkIdx = atomicAdd(&workCounter[concurrencyIndex], 1);
    }
    chunkIdx = __shfl_sync(0xffffffff, chunkIdx, 0);
    if (chunkIdx >= totalChunks) break;

    uint64_t byteStart = (uint64_t)chunkIdx * chunkSize;
    uint64_t byteEnd = byteStart + chunkSize;
    if (byteEnd > totalBytes) byteEnd = totalBytes;

    // Binary search for first descriptor containing byteStart
    uint32_t lo = 0, hi = params.numDescriptors;
    while (lo < hi) {
      uint32_t mid = (lo + hi) / 2;
      if (prefixSum[mid + 1] <= byteStart) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    // Process this chunk across possibly multiple descriptors
    uint64_t pos = byteStart;
    for (uint32_t d = lo; d < params.numDescriptors && pos < byteEnd; d++) {
      uint64_t descStart = prefixSum[d];
      uint64_t descEnd = prefixSum[d + 1];
      uint64_t regionStart = pos > descStart ? pos - descStart : 0;
      uint64_t regionEnd = (byteEnd < descEnd ? byteEnd : descEnd) - descStart;
      if (regionEnd > regionStart) {
        copy_region_v2(
            params.descriptors[d].src,
            params.descriptors[d].dst,
            params.descriptors[d].bytes,
            (uint32_t)regionStart,
            (uint32_t)(regionEnd - regionStart));
      }
      pos = descEnd;
    }
  }

  // Exit barrier
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) == $gridSize - 1) {
    $copyDoneWrites
    $copyDoneWaits
  }
}
)z",
        "$stepValueWrites", stepValueWrites, "$stepValueWaits", stepValueWaits, "$copyDoneWrites", copyDoneWrites,
        "$copyDoneWaits", copyDoneWaits, "$gridSize", gridSize, "$blockSize", blockSize, "$kMaxCopyDescriptors",
        kMaxCopyDescriptors, "$maxConcurrency", (size_t)Group::maxConcurrency);
  }

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  auto boolenv = [&](const char* name) {
    const char* c = std::getenv(name);
    if (!c) {
      return false;
    }
    return !strcmp(c, "1");
  };

  std::string cuFilename = fmt::sprintf("moodist-compile-op-kernels-rank%d.cu", rank);

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    FILE* f = fopen(cuFilename.c_str(), "wb");
    if (f) {
      fwrite(source.data(), source.size(), 1, f);
      fclose(f);
      log.info("compile_op kernel source dumped to %s\n", cuFilename);
    }
  }

  nvrtcProgram program;
  CHECK_NVRTC(nvrtcApi.createProgram(&program, source.c_str(), nullptr, 0, nullptr, nullptr));

  std::vector<std::pair<int, std::string>> archOptions;
  archOptions.emplace_back(10000, "--gpu-architecture=sm_100");
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
    error = nvrtcApi.compileProgram(program, options2.size(), options2.data());
    if (error == NVRTC_SUCCESS) {
      log.verbose("compile_op kernel: success with %s\n", archOptions[i].second);
    }
  }
  if (error != NVRTC_SUCCESS) {
    log.error("Failed to compile compile_op kernel--\n%s\n", source.c_str());
    size_t logSize = 0;
    std::string logstr;
    CHECK_NVRTC(nvrtcApi.getProgramLogSize(program, &logSize));
    logstr.resize(logSize);
    CHECK_NVRTC(nvrtcApi.getProgramLog(program, logstr.data()));
    log.error("%s\n", logstr);
    CHECK_NVRTC(error);
  }

  size_t cubinSize = 0;
  CHECK_NVRTC(nvrtcApi.getCUBINSize(program, &cubinSize));
  std::vector<char> cubin;
  cubin.resize(cubinSize);
  CHECK_NVRTC(nvrtcApi.getCUBIN(program, cubin.data()));

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    std::string fn = fmt::sprintf("moodist-compile-op-kernels-rank%d.o", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      log.info("compile_op cubin dumped to %s\n", fn);
    }
  }

  CHECK_NVRTC(nvrtcApi.destroyProgram(&program));

  CHECK_CU(cuModuleLoadDataEx(&cuModule, cubin.data(), 0, nullptr, nullptr));

  CHECK_CU(cuModuleGetFunction(&cuCopyKernel, cuModule, "compile_op_copy"));

  int numRegs = 0;
  CHECK_CU(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, cuCopyKernel));
  int maxThreadsPerBlock = 0;
  CHECK_CU(cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuCopyKernel));
  int localBytes = 0;
  CHECK_CU(cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, cuCopyKernel));
  log.info(
      "compile_op_copy: %d registers, %d local bytes, max %d threads/block\n", numRegs, localBytes, maxThreadsPerBlock);

  log.info("compile_op kernel compile took %gs\n", seconds(std::chrono::steady_clock::now() - start));
}

void CompileOpKernels::compileMulticast() {
  auto start = std::chrono::steady_clock::now();
  CUdevice& cuDevice = group->cuDevice;

  int computeMajor = 0;
  int computeMinor = 0;

  int major = 6;
  int minor = 0;
  if (!loadNvrtc()) {
    throw std::runtime_error("NVRTC not available");
  }
  CHECK_NVRTC(nvrtcApi.version(&major, &minor));

  CHECK_CU(cuDeviceGetAttribute(&computeMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
  CHECK_CU(cuDeviceGetAttribute(&computeMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));

  // Multicast with multimem.cp.async.bulk requires sm_90+
  if (computeMajor * 10 + computeMinor < 90) {
    log.error("compile_op multicast kernel requires sm_90+, have sm_%d%d\n", computeMajor, computeMinor);
    return;
  }

  size_t rank = group->rank;
  const auto& peerIndices = group->peerIndices;

  // Build sync address code fragments (same pattern as copy kernel)
  std::string stepValueWrites;
  std::string stepValueWaits;
  std::string copyDoneWrites;
  std::string copyDoneWaits;

  for (size_t i : peerIndices) {
    stepValueWrites += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", concurrencyIndexExpr(group->peerCudaStepValue[i], sizeof(uint32_t) * rank));

    stepValueWaits += replace(
        R"(while (*(volatile uint32_t*)($ptr) < stepValue);
    )",
        "$ptr", concurrencyIndexExpr(group->cudaStepValue, sizeof(uint32_t) * group->ipcRanks[i]));

    copyDoneWrites += replace(
        R"(
      *(volatile uint32_t*)$ptr = stepValue;
    )",
        "$ptr", concurrencyIndexExpr(group->peerCudaCopyDone[i], sizeof(uint32_t) * group->peerMyRemoteIndex[i]));

    copyDoneWaits += replace(
        R"(
      while (*(volatile uint32_t*)$ptr < stepValue);
    )",
        "$ptr", concurrencyIndexExpr(group->cudaCopyDone, sizeof(uint32_t) * i));
  }

  // Multicast kernel using multimem.st (PTX ISA 8.1, sm_90+).
  // Each thread loads from global memory and stores to the multicast VA using
  // multimem.st, which replicates the write to all bound GPUs via NVSwitch.
  // No shared memory staging needed — direct global → multicast.

  std::string source;

  source += R"z(
using uintptr_t = unsigned long;
using uint64_t = unsigned long;
using uint32_t = unsigned int;
using uint16_t = unsigned short;
using uint8_t = unsigned char;
using int32_t = int;
using int64_t = long;
using size_t = unsigned long;

namespace {

__device__ void syncthreads() {
  asm volatile ("barrier.sync 0;" :: );
}

struct CopyDescriptor {
  uintptr_t src;
  uintptr_t dst;
  uint32_t bytes;
};

struct CompileOpCopyParameters {
  uint32_t stepValue;
  uint32_t concurrencyIndex;
  uint32_t numDescriptors;
  uint32_t _pad;
  CopyDescriptor descriptors[$kMaxCopyDescriptors];
};

__device__ uint32_t entryCounter[$maxConcurrency];
__device__ uint32_t exitCounter[$maxConcurrency];

// Store 16 bytes to multicast VA using multimem.st (replicates to all bound GPUs)
// Vector qualifier requires floating-point type, so use .v4.f32 (same bits as uint4)
__device__ void multimem_st_v4(uintptr_t dst, uint4 val) {
  asm volatile(
    "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1, %2, %3, %4};"
    :: "l"(dst), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
    : "memory"
  );
}

} // namespace

extern "C" __global__ void __launch_bounds__($blockSize, 1)
compile_op_multicast(CompileOpCopyParameters params) {
  const uint32_t stepValue = params.stepValue;
  const uint32_t concurrencyIndex = params.concurrencyIndex;

  // Entry barrier: first block signals peers, all blocks wait for peers
  if (threadIdx.x == 0) {
    if (atomicInc(&entryCounter[concurrencyIndex], $gridSize - 1) == 0) {
      $stepValueWrites
      __threadfence_system();
    }
    $stepValueWaits
  }
  syncthreads();

  // Process descriptors cooperatively across all blocks
  const uint32_t numBlocks = $gridSize;
  const uint32_t globalTid = blockIdx.x * $blockSize + threadIdx.x;
  const uint32_t totalThreads = numBlocks * $blockSize;

  for (uint32_t d = 0; d < params.numDescriptors; d++) {
    uintptr_t src = params.descriptors[d].src;
    uintptr_t dst = params.descriptors[d].dst;
    uint32_t totalBytes = params.descriptors[d].bytes;

    // Aligned uint4 elements (16 bytes each) — use multimem.st
    uint32_t elements = totalBytes / 16;
    for (uint32_t i = globalTid; i < elements; i += totalThreads) {
      uint4 val = __ldcv((const uint4*)(src + (uintptr_t)i * 16));
      multimem_st_v4(dst + (uintptr_t)i * 16, val);
    }

    // Tail bytes (< 16) — use regular byte stores
    uint32_t tailStart = elements * 16;
    uint32_t tailBytes = totalBytes - tailStart;
    for (uint32_t i = globalTid; i < tailBytes; i += totalThreads) {
      *(volatile uint8_t*)(dst + tailStart + i) = *(const uint8_t*)(src + tailStart + i);
    }
  }

  // Fence to ensure multicast writes are visible across devices
  asm volatile("fence.sc.sys;" ::: "memory");

  // Exit barrier: last block signals copyDone, waits for peers
  __threadfence_system();
  syncthreads();
  if (threadIdx.x == 0 && atomicInc(&exitCounter[concurrencyIndex], $gridSize - 1) == $gridSize - 1) {
    $copyDoneWrites
    $copyDoneWaits
  }
}
)z";

  source =
      replace(source, "$kMaxCopyDescriptors", kMaxCopyDescriptors, "$maxConcurrency", (size_t)Group::maxConcurrency,
          "$stepValueWrites", stepValueWrites, "$stepValueWaits", stepValueWaits, "$copyDoneWrites", copyDoneWrites,
          "$copyDoneWaits", copyDoneWaits, "$gridSize", gridSize, "$blockSize", blockSize);

  source = replace(source, "%%", "%");
  source = autoindent(source);
  source = addLineCountComments(source);

  auto boolenv = [&](const char* name) {
    const char* c = std::getenv(name);
    if (!c) {
      return false;
    }
    return !strcmp(c, "1");
  };

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    std::string fn = fmt::sprintf("moodist-multicast-kernel-rank%d.cu", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(source.data(), source.size(), 1, f);
      fclose(f);
      log.info("multicast kernel source dumped to %s\n", fn);
    }
  }

  nvrtcProgram program;
  CHECK_NVRTC(nvrtcApi.createProgram(&program, source.c_str(), nullptr, 0, nullptr, nullptr));

  // Multicast requires sm_90+
  std::vector<std::string> options;
  options.push_back("--gpu-architecture=sm_90");
  options.push_back("--use_fast_math");
  options.push_back("--std=c++17");
  options.push_back("-lineinfo");
  std::vector<const char*> options2;
  for (auto& v : options) {
    options2.push_back(v.c_str());
  }
  nvrtcResult error = nvrtcApi.compileProgram(program, options2.size(), options2.data());

  if (error != NVRTC_SUCCESS) {
    log.error("Failed to compile multicast kernel--\n%s\n", source.c_str());
    size_t logSize = 0;
    std::string logstr;
    CHECK_NVRTC(nvrtcApi.getProgramLogSize(program, &logSize));
    logstr.resize(logSize);
    CHECK_NVRTC(nvrtcApi.getProgramLog(program, logstr.data()));
    log.error("%s\n", logstr);
    CHECK_NVRTC(error);
  }

  size_t cubinSize = 0;
  CHECK_NVRTC(nvrtcApi.getCUBINSize(program, &cubinSize));
  std::vector<char> cubin;
  cubin.resize(cubinSize);
  CHECK_NVRTC(nvrtcApi.getCUBIN(program, cubin.data()));

  if (boolenv("MOODIST_DUMP_KERNELS")) {
    std::string fn = fmt::sprintf("moodist-multicast-kernel-rank%d.o", rank);
    FILE* f = fopen(fn.c_str(), "wb");
    if (f) {
      fwrite(cubin.data(), cubin.size(), 1, f);
      fclose(f);
      log.info("multicast cubin dumped to %s\n", fn);
    }
  }

  CHECK_NVRTC(nvrtcApi.destroyProgram(&program));

  CHECK_CU(cuModuleLoadDataEx(&cuMulticastModule, cubin.data(), 0, nullptr, nullptr));
  CHECK_CU(cuModuleGetFunction(&cuMulticastKernel, cuMulticastModule, "compile_op_multicast"));

  int numRegs = 0;
  CHECK_CU(cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, cuMulticastKernel));
  int maxThreadsPerBlock = 0;
  CHECK_CU(cuFuncGetAttribute(&maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuMulticastKernel));
  int localBytes = 0;
  CHECK_CU(cuFuncGetAttribute(&localBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, cuMulticastKernel));
  log.info("compile_op_multicast: %d registers, %d local bytes, max %d threads/block\n", numRegs, localBytes,
      maxThreadsPerBlock);

  log.info("compile_op multicast kernel compile took %gs\n", seconds(std::chrono::steady_clock::now() - start));
}

} // namespace moodist
