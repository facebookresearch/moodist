
#include "processgroup.h"
#include "allgather.h"
#include "clock.h"
#include "common.h"
#include "cputhread.h"
#include "group.h"
#include "ipc_mapper.h"
#include "kernels.h"
#include "reduce_scatter.h"
#include "serialization.h"
#include "setup_comms.h"
#include "synchronization.h"
#include "vector.h"

#include <ATen/ops/pad.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cstring>
#include <random>
#include <torch/types.h>
#include <variant>

#include <cublas.h>

namespace moodist {

extern bool profilingEnabled;

async::Scheduler scheduler;

void throwCuHelper(CUresult error, const char* file, int line) {
  throwCu(error, file, line);
}

struct ProcessGroupImpl;
std::mutex activeProcessGroupsMutex;
std::vector<ProcessGroupImpl*> activeProcessGroups;

std::once_flag freeMemoryCallbackOnceFlag;
void registerFreeMemoryCallback();

std::once_flag globalInitFlag;
void globalInit() {

  const char* logLevel = std::getenv("MOODIST_LOG_LEVEL");
  if (logLevel) {
    std::string s = logLevel;
    for (auto& c : s) {
      c = std::toupper(c);
    }
    if (s == "NONE") {
      currentLogLevel = LOG_NONE;
    } else if (s == "ERROR") {
      currentLogLevel = LOG_ERROR;
    } else if (s == "INFO") {
      currentLogLevel = LOG_INFO;
    } else if (s == "VERBOSE") {
      currentLogLevel = LOG_VERBOSE;
    } else if (s == "DEBUG") {
      currentLogLevel = LOG_DEBUG;
    } else {
      currentLogLevel = (LogLevel)std::atoi(logLevel);
    }
    if (currentLogLevel < LOG_ERROR) {
      currentLogLevel = LOG_ERROR;
    }

    fmt::printf("currentLogLevel set to %d\n", (int)currentLogLevel);
  }
}

struct ProcessGroupImpl {
  size_t rank = 0;
  size_t size = 0;

  std::shared_ptr<Group> group;

  SpinMutex mutex;
  uint32_t nextStepValue = 1;
  uint32_t nextConcurrencyIndex = 0;

  ProcessGroupImpl(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size) : rank(rank), size(size) {
    TORCH_CHECK(rank >= 0 && size > 0 && rank < size);

    scheduler.setMaxThreads(1);

    std::call_once(globalInitFlag, &globalInit);

    group = std::make_shared<Group>(rank, size);

    init(store);

    {
      std::call_once(freeMemoryCallbackOnceFlag, &registerFreeMemoryCallback);
      std::lock_guard l(activeProcessGroupsMutex);
      activeProcessGroups.push_back(this);
    }
  }

  void freeMemory() {
    std::lock_guard l(mutex);
    CHECK_CU(cuCtxSynchronize());
    group->ipcMapper->unmapAll();
  }

  struct TraceEvent {
    std::string name;
    std::chrono::system_clock::time_point begin;
    std::chrono::system_clock::time_point end;
  };
  std::vector<TraceEvent> traceEvents;

  std::chrono::system_clock::time_point beginning = std::chrono::system_clock::now();

  std::string currentTraceName = "";
  std::chrono::system_clock::time_point currentTraceBegin = beginning;
  int threadId = gettid();

  void trace(std::string name) {
    // if (opCount < 1000 || opCount >= 1200) {
    if (!profilingEnabled) {
      if (!traceEvents.empty()) {
        std::string fn = fmt::sprintf("moodist-trace-pg-%d.json", rank);
        FILE* f = fopen(fn.c_str(), "wb");
        CHECK(f != nullptr);
        fmt::fprintf(
            f, R"({"traceEvents": [)"
               "\n");
        bool first = true;
        for (auto& e : traceEvents) {
          if (!first) {
            fmt::fprintf(f, ",\n");
          } else {
            first = false;
          }
          fmt::fprintf(
              f, R"({"ph": "X", "name": "%s", "ts": %d, "dur": %d, "tid": %d})", e.name,
              std::chrono::duration_cast<std::chrono::microseconds>(e.begin.time_since_epoch()).count(),
              std::chrono::duration_cast<std::chrono::microseconds>(e.end - e.begin).count(), threadId);
        }
        fmt::fprintf(f, "]}\n");
        fclose(f);
        fmt::printf("Chrome trace dumped to %s\n", fn);
        traceEvents.clear();
      }
      return;
    }
    auto now = std::chrono::system_clock::now();
    if (traceEvents.empty() && currentTraceName.empty()) {
      beginning = now;
    }
    if (!currentTraceName.empty()) {
      traceEvents.emplace_back();
      TraceEvent& e = traceEvents.back();
      e.name = std::move(currentTraceName);
      e.begin = currentTraceBegin;
      e.end = now;
    }
    currentTraceBegin = now;
    currentTraceName = std::move(name);
  };
#define trace(name)

  ~ProcessGroupImpl() {
    trace("");

    std::lock_guard l(activeProcessGroupsMutex);
    auto i = std::find(activeProcessGroups.begin(), activeProcessGroups.end(), this);
    if (i != activeProcessGroups.end()) {
      activeProcessGroups.erase(i);
    }
  }

  void init(const c10::intrusive_ptr<::c10d::Store>& store) {

    log.verbose("%d/%d: init\n", rank, size);

    auto start = Clock::now();

    std::unique_lock l(mutex);

    std::string key = randomName();

    auto tovec = [&](std::string str) { return std::vector<uint8_t>(str.begin(), str.end()); };
    auto fromvec = [&](std::vector<uint8_t> vec) { return std::string(vec.begin(), vec.end()); };

    auto set = [&](std::string key, std::string value) { store->set(key, tovec(value)); };
    auto get = [&](std::string key) { return fromvec(store->get(key)); };

    if (rank == 0) {
      set("moodist_pg_key", key);
    } else {
      key = get("moodist_pg_key");
    }
    group->setupComms->key = key;
    set(fmt::sprintf("moodist_pg_rank%d_address", rank), getAddress());
    int prevRank = (rank + size - 1) % size;
    int nextRank = (rank + 1) % size;
    std::string prevAddress = get(fmt::sprintf("moodist_pg_rank%d_address", prevRank));
    std::string nextAddress = get(fmt::sprintf("moodist_pg_rank%d_address", nextRank));

    for (auto& address : decodeAddress(prevAddress)) {
      group->setupComms->connect(address);
    }
    for (auto& address : decodeAddress(nextAddress)) {
      group->setupComms->connect(address);
    }

    log.debug("%d: Waiting for setupComms\n", rank);
    std::fflush(stdout);
    group->setupComms->waitForConnections();

    set(fmt::sprintf("moodist_rank%d_ready", rank), key);
    for (size_t i = 0; i != size; ++i) {
      CHECK(get(fmt::sprintf("moodist_rank%d_ready", i)) == key);
    }

    log.debug("%d: Waiting for connections\n", rank);
    std::fflush(stdout);
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      group->setupComms->sendTo(i, fmt::sprintf("hello %d %s", i, key));
    }
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      log.debug("%d: waiting for greeting from %d\n", rank, i);
      std::fflush(stdout);
      std::string greeting = group->setupComms->recvFrom<std::string>(i);
      TORCH_CHECK(greeting == fmt::sprintf("hello %d %s", rank, key));
      log.debug("greeting ok!\n");
      std::fflush(stdout);
    }
    log.debug("got all connections\n");
    std::fflush(stdout);

    group->init();

    log.verbose("init took %gs\n", seconds(Clock::now() - start));
  }

  std::vector<std::string> decodeAddress(std::string str) {
    std::vector<uint8_t> data;
    size_t i = 0;
    for (char vv : str) {
      size_t index = vv >= '0' && vv <= '9' ? vv - '0' : vv - 'a' + 10;
      if (index >= 16) {
        throw std::invalid_argument("ProcessGroup: invalid address");
      }
      if (i % 2 == 0) {
        data.push_back(index);
      } else {
        data.back() <<= 4;
        data.back() |= index;
      }
      ++i;
    }
    std::vector<std::string> remoteAddresses;
    deserializeBuffer(data.data(), data.size(), remoteAddresses);
    return remoteAddresses;
  }

  std::string getAddress() {
    std::vector<std::string> addresses = group->setupComms->listenerAddresses();
    auto buffer = serializeToBuffer(addresses);

    std::string r;
    for (size_t i = 0; i != buffer->size(); ++i) {
      uint8_t v = (uint8_t)buffer->data()[i];
      r += "0123456789abcdef"[v >> 4];
      r += "0123456789abcdef"[v & 0xf];
    }
    return r;
  }

  void barrier() {
    std::lock_guard l(mutex);
    uint32_t stepValue = std::exchange(nextStepValue, nextStepValue + 0x1000);
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);
    std::atomic_uint32_t cpuDone = 0;
    QueueEntryBarrier* e = group->cpuThread->freelistBarrier.pop();
    e->task = taskBarrier;
    e->stepValue = stepValue;
    e->sd = &group->getStreamData(nullptr);
    e->concurrencyIndex = concurrencyIndex;
    e->cpuDone = &cpuDone;
    group->cpuThread->enqueue(e);
    while (cpuDone == 0) {
      futexWait(&cpuDone, 0, std::chrono::seconds(10));
    }
  }

  void all_gather(at::Tensor& output, const at::Tensor& input) {
    trace("all_gather");
    std::lock_guard l(mutex);

    size_t size = this->size;

    uint32_t stepValue = std::exchange(nextStepValue, nextStepValue + 0x1000);
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(input.device().index() == group->deviceIndex);
    TORCH_CHECK(output.is_contiguous());
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(output.device().index() == group->deviceIndex);

    size_t bytes = input.numel() * input.itemsize();
    size_t outputBytes = bytes * size;
    size_t pitch = bytes;

    TORCH_CHECK(output.numel() * output.itemsize() == outputBytes);

    uintptr_t inputAddress = (uintptr_t)input.data_ptr();
    uintptr_t outputAddress = (uintptr_t)output.data_ptr();

    CHECK(inputAddress % 16 == 0);
    CHECK(outputAddress % 16 == 0);

    CUstream stream = c10::cuda::getCurrentCUDAStream();

    StreamData& sd = group->getStreamData(stream);

    auto allocbuffer = [&](auto& buffer, size_t size) {
      if (buffer.bytes < size) {
        CHECK_CU(cuCtxSynchronize());
        buffer = {};
        buffer = group->allocateDevice(size);
      }
    };

    if (bytes % 16 != 0) {
      pitch = (bytes + 127u) / 128u * 128u;
      allocbuffer(sd.alignedBuffer, pitch * size);
      outputAddress = sd.alignedBuffer.cudaPointer;
      outputBytes = pitch * size;
    }

    IpcMapper* ipcMapper = &*group->ipcMapper;

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    trace("ipcMapper");

    std::array<uintptr_t, 8> peerInputAddresses;
    std::array<uintptr_t, 8> peerOutputAddresses;

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, inputAddress, pitch, [ptr = &peerInputAddresses[i]](uintptr_t address) { *ptr = address; }, true);

      ipcMapper->requestAddress(
          i, outputAddress, outputBytes, [ptr = &peerOutputAddresses[i]](uintptr_t address) { *ptr = address; }, true);
    }

    ipcMapper->wait();

    trace("enqueue");

    Group* group = &*this->group;

    AllGather& allGather = *group->allGather;

    if (!group->kernels->cuModule) {
      group->kernels->compile();
    }

    bool isLocalOnly = allGather.recvRanks.empty();

    if (!isLocalOnly) {
      QueueEntryAllGather* e = group->cpuThread->freelistAllGather.pop();
      e->task = taskAllgather;
      e->stepValue = stepValue;
      e->concurrencyIndex = concurrencyIndex;
      e->sd = &group->getStreamData(stream);
      e->inputAddress = inputAddress;
      e->outputAddress = outputAddress;
      e->bytes = pitch;
      e->pitch = pitch;
      group->cpuThread->enqueue(e);
    }

    trace("launch");

    AllGatherParameters parameters;
    parameters.stepValue = stepValue;
    parameters.concurrencyIndex = concurrencyIndex;
    parameters.bytes = pitch;
    parameters.pitch = pitch;
    parameters.inputAddress = inputAddress;
    parameters.outputAddress = outputAddress;
    parameters.peerInputAddresses = peerInputAddresses;
    parameters.peerOutputAddresses = peerOutputAddresses;

    std::array<void*, 1> params = {&parameters};

    size_t gridSize = group->kernels->gridSize;
    size_t blockSize = group->kernels->blockSize;

    if (isLocalOnly) {
      CHECK_CU(cuLaunchKernel(
          group->kernels->cuAllGatherLocal, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));
    } else {
      CHECK_CU(cuLaunchKernel(
          group->kernels->cuAllGather, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));
    }

    if (outputAddress != (uintptr_t)output.data_ptr()) {
      CUDA_MEMCPY2D copyArgs = {0};
      copyArgs.srcDevice = sd.alignedBuffer.cudaPointer;
      copyArgs.srcPitch = pitch;
      copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copyArgs.dstDevice = (uintptr_t)output.data_ptr();
      copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copyArgs.WidthInBytes = bytes;
      copyArgs.Height = size;
      CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));
    }

    trace("post");
  }

  void reduce_scatter(at::Tensor& output, const at::Tensor& input, c10d::ReduceOp reduceOp) {
    trace("_reduce_scatter_base");
    std::lock_guard l(mutex);

    CHECK(reduceOp == c10d::ReduceOp::SUM);

    size_t size = this->size;

    uint32_t stepValue = std::exchange(nextStepValue, nextStepValue + 0x1000);
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(input.device().index() == group->deviceIndex);
    TORCH_CHECK(output.is_contiguous());
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(output.device().index() == group->deviceIndex);

    TORCH_CHECK(input.dtype() == torch::Dtype::Float);
    TORCH_CHECK(output.dtype() == torch::Dtype::Float);

    size_t numel = output.numel();
    size_t bytes = numel * output.itemsize();
    size_t inputBytes = bytes * size;
    size_t pitch = bytes;

    TORCH_CHECK(input.numel() * input.itemsize() == inputBytes);

    uintptr_t inputAddress = (uintptr_t)input.data_ptr();
    uintptr_t outputAddress = (uintptr_t)output.data_ptr();

    CHECK(inputAddress % 16 == 0);
    CHECK(outputAddress % 16 == 0);

    CUstream stream = c10::cuda::getCurrentCUDAStream();

    StreamData& sd = group->getStreamData(stream);

    auto allocbuffer = [&](auto& buffer, size_t size) {
      if (buffer.bytes < size) {
        CHECK_CU(cuCtxSynchronize());
        buffer = {};
        buffer = group->allocateDevice(size);
      }
    };

    if (bytes % 16 != 0) {
      pitch = (bytes + 127u) / 128u * 128u;
      allocbuffer(sd.alignedBuffer, pitch * size);
      CUDA_MEMCPY2D copyArgs = {0};
      copyArgs.srcDevice = inputAddress;
      copyArgs.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copyArgs.dstDevice = sd.alignedBuffer.cudaPointer;
      copyArgs.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copyArgs.dstPitch = pitch;
      copyArgs.WidthInBytes = bytes;
      copyArgs.Height = size;
      CHECK_CU(cuMemcpy2DAsync(&copyArgs, stream));
      inputAddress = sd.alignedBuffer.cudaPointer;
      inputBytes = pitch * size;
    }

    IpcMapper* ipcMapper = &*group->ipcMapper;

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    trace("ipcMapper");

    std::array<uintptr_t, 8> peerInputAddresses;
    std::array<uintptr_t, 8> peerOutputAddresses;

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, inputAddress, inputBytes, [ptr = &peerInputAddresses[i]](uintptr_t address) { *ptr = address; }, true);

      ipcMapper->requestAddress(
          i, outputAddress, bytes, [ptr = &peerOutputAddresses[i]](uintptr_t address) { *ptr = address; }, true);
    }

    ipcMapper->wait();

    trace("enqueue");

    Group* group = &*this->group;

    ReduceScatter& reduceScatter = *group->reduceScatter;

    auto& sendRanks = reduceScatter.sendRanks;
    auto& recvRanks = reduceScatter.recvRanks;

    allocbuffer(sd.sendBuffer, pitch * sendRanks.size());
    allocbuffer(sd.recvBuffer, pitch * recvRanks.size());

    if (!group->kernels->cuModule) {
      group->kernels->compile();
    }

    bool isLocalOnly = reduceScatter.recvRanks.empty();

    if (!isLocalOnly) {
      QueueEntryReduceScatter* e = group->cpuThread->freelistReduceScatter.pop();
      e->task = taskReduceScatter;
      e->stepValue = stepValue;
      e->concurrencyIndex = concurrencyIndex;
      e->sd = &sd;
      e->inputAddress = inputAddress;
      e->outputAddress = outputAddress;
      e->bytes = bytes;
      e->pitch = pitch;
      group->cpuThread->enqueue(e);
    }

    trace("launch");

    uintptr_t sendAddress = sd.sendBuffer.cudaPointer;
    uintptr_t recvAddress = sd.recvBuffer.cudaPointer;
    if (!isLocalOnly) {
      CHECK(sendAddress != 0);
      CHECK(recvAddress != 0);
    }

    ReduceScatterParameters parameters;
    parameters.stepValue = stepValue;
    parameters.concurrencyIndex = concurrencyIndex;
    parameters.bytes = bytes;
    parameters.pitch = pitch;
    parameters.inputAddress = inputAddress;
    parameters.outputAddress = outputAddress;
    parameters.peerInputAddresses = peerInputAddresses;
    parameters.peerOutputAddresses = peerOutputAddresses;
    parameters.sendAddress = sendAddress;
    parameters.recvAddress = recvAddress;

    std::array<void*, 1> params = {&parameters};

    size_t gridSize = group->kernels->gridSize;
    size_t blockSize = group->kernels->blockSize;

    if (isLocalOnly) {
      CHECK_CU(cuLaunchKernel(
          group->kernels->cuReduceScatterLocal, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));
    } else {
      CHECK_CU(cuLaunchKernel(
          group->kernels->cuReduceScatter, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));
    }

    trace("post");
  }

  void broadcast(const torch::Tensor& tensor, uint32_t sourceRank) {
    std::lock_guard l(mutex);

    size_t size = this->size;

    uint32_t stepValue = std::exchange(nextStepValue, nextStepValue + 0x1000);
    uint32_t concurrencyIndex = std::exchange(nextConcurrencyIndex, (nextConcurrencyIndex + 1) % Group::maxConcurrency);
    CHECK(stepValue < 0x80000000);

    TORCH_CHECK(tensor.is_contiguous());
    TORCH_CHECK(tensor.is_cuda());
    TORCH_CHECK(tensor.device().index() == group->deviceIndex);
    TORCH_CHECK(sourceRank < size);

    size_t numel = tensor.numel();
    size_t bytes = numel * tensor.itemsize();

    uintptr_t tensorAddress = (uintptr_t)tensor.data_ptr();

    CHECK(tensorAddress % 16 == 0);

    CUstream stream = c10::cuda::getCurrentCUDAStream();

    StreamData& sd = group->getStreamData(stream);

    Group* group = &*this->group;

    if (!group->kernels->cuModule) {
      group->kernels->compile();
    }

    QueueEntryBroadcast* e = group->cpuThread->freelistBroadcast.pop();
    e->task = taskBroadcast;
    e->stepValue = stepValue;
    e->concurrencyIndex = concurrencyIndex;
    e->sd = &sd;
    e->tensorAddress = tensorAddress;
    e->bytes = bytes;
    e->sourceRank = sourceRank;
    group->cpuThread->enqueue(e);

    std::array<void*, 2> params = {&stepValue, &concurrencyIndex};

    size_t gridSize = 1;
    size_t blockSize = 1;

    CHECK_CU(cuLaunchKernel(
        group->kernels->cuBroadcast, gridSize, 1, 1, blockSize, 1, 1, 0, stream, params.data(), nullptr));

    trace("post");
  }
};

struct FreeMemoryCallback : at::FreeMemoryCallback {
  virtual bool Execute() {
    std::unique_lock l(activeProcessGroupsMutex);
    for (auto* pg : activeProcessGroups) {
      pg->freeMemory();
    }
    return false;
  };
};

void registerFreeMemoryCallback() {
  at::FreeCudaMemoryCallbacksRegistry()->Register("moodist", []() { return std::make_unique<FreeMemoryCallback>(); });
}

ProcessGroup::ProcessGroup(const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size)
    : c10d::ProcessGroup(rank, size) {
  impl = std::make_unique<ProcessGroupImpl>(store, rank, size);
}
ProcessGroup::~ProcessGroup() {
  impl.reset();
}

struct WorkImpl : c10d::Work {
  std::variant<torch::Tensor, std::vector<torch::Tensor>> var;
  virtual void synchronize() override {}
  virtual bool wait(std::chrono::milliseconds timeout) override {
    return true;
  }

  WorkImpl(torch::Tensor result) : var(result) {}
  WorkImpl(std::vector<torch::Tensor> result) : var(result) {}

  virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    std::vector<torch::Tensor> result;
    if (var.index() == 0) {
      result = {std::get<0>(var)};
    } else {
      result = std::get<1>(var);
    }
    auto fut = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), std::vector<c10::Device>{result[0].device()});
    fut->markCompleted(at::IValue(result));
    return fut;
  }
};

c10::intrusive_ptr<Work> ProcessGroup::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1 && inputTensors.size() == 1);
  const auto& input = inputTensors[0];
  int64_t numel = input.numel();
  const auto& outputList = outputTensors[0];
  for (auto& v : outputList) {
    TORCH_CHECK(v.numel() == numel);
  }
  auto output =
      torch::empty({(int64_t)impl->size, numel}, torch::TensorOptions().dtype(input.dtype()).device(input.device()));
  impl->all_gather(output, input);
  for (size_t i = impl->size; i;) {
    --i;
    outputList[i].copy_(output[i]);
  }
  return c10::make_intrusive<WorkImpl>(outputList);
}

c10::intrusive_ptr<Work>
ProcessGroup::_allgather_base(at::Tensor& outputbuffer, at::Tensor& inputbuffer, const c10d::AllgatherOptions& opts) {
  impl->all_gather(outputbuffer, inputbuffer);
  return c10::make_intrusive<WorkImpl>(outputbuffer);
}

c10::intrusive_ptr<Work> ProcessGroup::_reduce_scatter_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor, const c10d::ReduceScatterOptions& opts) {
  impl->reduce_scatter(outputTensor, inputTensor, opts.reduceOp);
  return c10::make_intrusive<WorkImpl>(outputTensor);
}

c10::intrusive_ptr<Work> ProcessGroup::barrier(const c10d::BarrierOptions& opts) {
  impl->barrier();
  return c10::make_intrusive<WorkImpl>(torch::Tensor());
}

c10::intrusive_ptr<Work> ProcessGroup::broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1);
  impl->broadcast(tensors[0], opts.rootRank);
  return c10::make_intrusive<WorkImpl>(tensors);
}

c10::intrusive_ptr<Work> ProcessGroup::allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1);
  auto& tensor = tensors[0];
  if (tensor.numel() % impl->size == 0) {
    auto t = tensor.view({(int64_t)impl->size, -1});
    auto o = t[impl->rank];
    impl->reduce_scatter(o, t, opts.reduceOp);
    impl->all_gather(t, o);
    return c10::make_intrusive<WorkImpl>(tensors);
  } else {
    int64_t numel = tensor.numel();
    int64_t newsize = (numel + impl->size - 1) / impl->size * impl->size;
    newsize = (newsize + 127) / 128u * 128u;
    size_t bytes = numel * tensor.itemsize();
    torch::Tensor temporary =
        torch::empty({newsize}, torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
    CHECK_CU(cuMemcpyAsync(
        (uintptr_t)temporary.data_ptr(), (uintptr_t)tensor.data_ptr(), bytes, c10::cuda::getCurrentCUDAStream()));

    auto t = temporary.view({(int64_t)impl->size, -1});
    auto o = t[impl->rank];
    impl->reduce_scatter(o, t, opts.reduceOp);
    impl->all_gather(t, o);

    CHECK_CU(cuMemcpyAsync(
        (uintptr_t)tensor.data_ptr(), (uintptr_t)temporary.data_ptr(), bytes, c10::cuda::getCurrentCUDAStream()));

    return c10::make_intrusive<WorkImpl>(tensors);
  }
}

// void ProcessGroup::all_gather(at::Tensor& output, const at::Tensor& input) {
//   return impl->all_gather(output, input);
// }

// void ProcessGroup::all_gather_list(std::vector<at::Tensor>& output, const at::Tensor& input) {
//   return impl->all_gather_list(output, input);
// }

// void ProcessGroup::reduce_scatter(at::Tensor& output, const at::Tensor& input, c10d::ReduceOp op) {
//   return impl->reduce_scatter(output, input, op);
// }

// void ProcessGroup::reduce_scatter_list(at::Tensor& output, const std::vector<at::Tensor>& input, c10d::ReduceOp op) {
//   return impl->reduce_scatter_list(output, input, op);
// }

// void ProcessGroup::barrier() {
//   return impl->barrier();
// }

} // namespace moodist
