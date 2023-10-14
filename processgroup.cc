
#include "processgroup.h"
#include "allgather.h"
#include "clock.h"
#include "common.h"
#include "cputhread.h"
#include "group.h"
#include "ipc_mapper.h"
#include "serialization.h"
#include "setup_comms.h"
#include "synchronization.h"
#include "vector.h"

#include <c10/core/Device.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cstring>
#include <random>

namespace moodist {

extern bool profilingEnabled;

void throwCuHelper(CUresult error, const char* file, int line) {
  throwCu(error, file, line);
}

struct Op {
  std::atomic_bool busy = false;
  CUevent inputEvent;
  CUevent outputEvent;
};

class OpFutureWrappingWork : public tccl_work::Work {
public:
  std::shared_ptr<Op> op;
  std::vector<at::Tensor> outputs;
  bool done = false;
  OpFutureWrappingWork(std::shared_ptr<Op> op, std::vector<at::Tensor> outputs)
      : Work(), op(op), outputs(std::move(outputs)) {}

  ~OpFutureWrappingWork() {
    if (!done) {
      op->busy = false;
    }
  }

  bool checkDone() const {
    if (done) {
      return true;
    }
    CUresult r = cuEventQuery(op->outputEvent);
    if (r == CUDA_SUCCESS) {
      return true;
    }
    if (r == CUDA_ERROR_NOT_READY) {
      return false;
    }
    CHECK_CU(r);
    return false;
  }

  bool isCompleted() override {
    return checkDone();
  }

  bool isSuccess() const override {
    return checkDone();
  }

  std::exception_ptr exception() const override {
    return nullptr;
    // return fut_->exception_ptr();
  }

  int sourceRank() const override {
    TORCH_CHECK(false, "FutureWrappingWork::sourceRank() not implemented");
  }

  std::vector<at::Tensor> result() override {
    return outputs;
    // return fut_->value().toPyObjectHolder()->extractTensors();
  }

  virtual void synchronize() override final {
    if (done) {
      return;
    }
    // c10::cuda::CUDACachingAllocator::recordStream(const DataPtr &, CUDAStream stream)
    // fmt::printf("op future synchronize()!\n");
    auto stream = c10::cuda::getCurrentCUDAStream();
    // fmt::printf("synchronize, wait for event %p\n", (void*)op->event);
    // fmt::printf("synchronize, SKIP wait for event %p\n", (void*)op->outputEvent);
    CHECK_CU(cuStreamWaitEvent(stream, op->outputEvent, CU_EVENT_WAIT_DEFAULT));
    op->busy = false;
    done = true;
  }

  bool wait(std::chrono::milliseconds timeout) override {
    if (done) {
      return true;
    }
    synchronize();
    return true;
  }

  void abort() override {
    TORCH_CHECK(false, "FutureWrappingWork::abort() not implemented");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    TORCH_CHECK(false, "getFuture");
    return {};
    // return fut_;
  }

  // private:
  //   c10::intrusive_ptr<c10::ivalue::Future> fut_;
};

c10::intrusive_ptr<tccl_work::Work> makeFuture(std::shared_ptr<Op> op, std::vector<at::Tensor> outputs) {
  return c10::make_intrusive<OpFutureWrappingWork>(op, std::move(outputs));
}

struct MemoryManager {
  Group* group;
  std::vector<AllocatedBuffer> free;
  AllocatedBuffer allocate(size_t bytes) {
    for (auto i = free.end(); i != free.begin();) {
      --i;
      if (i->bytes >= bytes) {
        AllocatedBuffer r = std::move(*i);
        free.erase(i);
        return r;
      }
    }
    return group->allocateHost(bytes);
  }
  void deallocate(AllocatedBuffer buffer) {
    free.push_back(std::move(buffer));
  }
};

struct ProcessGroupImpl {
  size_t rank = 0;
  size_t size = 0;

  std::shared_ptr<Group> group;

  MemoryManager inputMemory;
  MemoryManager outputMemory;

  SpinMutex mutex;
  std::vector<std::shared_ptr<Op>> allOps;
  std::atomic_uint32_t nextStepValue = 1;

  ProcessGroupImpl(int rank, int size) : rank(rank), size(size) {
    TORCH_CHECK(rank >= 0 && size > 0 && rank < size);

    scheduler.setMaxThreads(1);

    group = std::make_shared<Group>(rank, size);

    for (auto& v : {&inputMemory, &outputMemory}) {
      v->group = &*group;
    }
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
        std::string fn = fmt::sprintf("moodist-trace-%d.json", rank);
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
//#define trace(name)

  ~ProcessGroupImpl() {
    trace("");
  }

  void init(std::string rank0Address) {

    fmt::printf("%d: init\n", rank);

    std::unique_lock l(mutex);

    if (rank != 0) {
      // fmt::printf("v is %d: %s\n", v.size(), v);
      std::vector<uint8_t> data;
      size_t i = 0;
      for (char vv : rank0Address) {
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
      // std::string str;
      // for (auto& v : data) {
      //   str += fmt::sprintf("%02x", (unsigned char)v);
      // }
      // fmt::printf("data.size() is %d: %s\n", data.size(), str);
      std::vector<std::string> remoteAddresses;
      deserializeBuffer(data.data(), data.size(), remoteAddresses);

      for (auto& address : remoteAddresses) {
        group->setupComms->connect(address);
      }
    }

    fmt::printf("%d: Waiting for setupComms\n", rank);
    std::fflush(stdout);
    group->setupComms->waitForConnections();

    fmt::printf("%d: Waiting for connections\n", rank);
    std::fflush(stdout);
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      group->setupComms->sendTo(i, fmt::sprintf("hello %d", i));
    }
    for (size_t i = 0; i != size; ++i) {
      if (i == rank) {
        continue;
      }
      fmt::printf("%d: waiting for greeting from %d\n", rank, i);
      std::fflush(stdout);
      std::string greeting = group->setupComms->recvFrom<std::string>(i);
      TORCH_CHECK(greeting == fmt::sprintf("hello %d", rank));
      fmt::printf("greeting ok!\n");
      std::fflush(stdout);
    }
    fmt::printf("got all connections\n");
    std::fflush(stdout);

    group->init();
  }

  std::string getAddress() {
    std::vector<std::string> addresses = group->setupComms->listenerAddresses();
    auto buffer = serializeToBuffer(addresses);

    // fmt::printf("serialized to %d bytes\n", buffer->size);

    std::string r;
    for (size_t i = 0; i != buffer->size(); ++i) {
      uint8_t v = (uint8_t)buffer->data()[i];
      r += "0123456789abcdef"[v >> 4];
      r += "0123456789abcdef"[v & 0xf];
    }
    return r;
  }

  std::shared_ptr<Op> getOp() {
    for (auto& v : allOps) {
      if (v->busy.load(std::memory_order_relaxed)) {
        continue;
      }
      v->busy = true;
      return v;
    }
    std::shared_ptr<Op> op = std::make_shared<Op>();
    op->busy = true;
    CHECK_CU(cuEventCreate(&op->inputEvent, CU_EVENT_DISABLE_TIMING));
    CHECK_CU(cuEventCreate(&op->outputEvent, CU_EVENT_DISABLE_TIMING));
    allOps.push_back(op);
    return op;
  }

  c10::intrusive_ptr<tccl_work::Work> allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    TORCH_CHECK(false, "allreduce");
  }

  c10::intrusive_ptr<tccl_work::Work> barrier(const c10d::BarrierOptions& opts) {
    TORCH_CHECK(false, "barrier");
  }

  struct CopyNodeInfo {
    CUgraphNode node;
    uintptr_t dst;
    uintptr_t src;
    size_t bytes;
  };

  struct AllGatherParameters {};

  struct AllGatherGraph {
    CUgraph graph = nullptr;
    CUgraphExec graphExec = nullptr;

    std::vector<CUgraphNode> nodes;
    std::vector<CopyNodeInfo> copyNodes;
  };

  template<typename Graph>
  struct GraphBuilder {
    Graph& graph;
    Group* group;

    GraphBuilder(Graph& graph, Group* group) : graph(graph), group(group) {
      CHECK_CU(cuGraphCreate(&graph.graph, 0));
    }

    CUgraphNode addKernel(CUgraphNode dependency, CUfunction function) {
      CUDA_KERNEL_NODE_PARAMS kernelParams;
      std::memset(&kernelParams, 0, sizeof(kernelParams));
      kernelParams.func = function;
      kernelParams.gridDimX = 1;
      kernelParams.gridDimY = 1;
      kernelParams.gridDimZ = 1;
      kernelParams.blockDimX = 1;
      kernelParams.blockDimY = 1;
      kernelParams.blockDimZ = 1;
      kernelParams.kernelParams = nullptr;
      CUgraphNode node;
      CHECK_CU(cuGraphAddKernelNode(&node, graph.graph, &dependency, dependency ? 1 : 0, &kernelParams));
      graph.nodes.push_back(node);
      return node;
    }

    CUgraphNode addCopy(CUgraphNode dependency, uintptr_t dst, uintptr_t src, size_t bytes) {
      CUDA_MEMCPY3D copyParams;
      std::memset(&copyParams, 0, sizeof(copyParams));

      TORCH_CHECK(dst != 0);
      TORCH_CHECK(src != 0);
      TORCH_CHECK(bytes != 0);

      copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copyParams.dstDevice = dst;
      copyParams.srcDevice = src;
      copyParams.WidthInBytes = bytes;
      copyParams.Height = 1;
      copyParams.Depth = 1;
      fmt::printf("add copy node dst %#x src %#x bytes %d\n", dst, src, bytes);
      CUgraphNode node;
      CHECK_CU(
          cuGraphAddMemcpyNode(&node, graph.graph, &dependency, dependency ? 1 : 0, &copyParams, group->cuContext));
      graph.nodes.push_back(node);
      graph.copyNodes.emplace_back();
      auto& n = graph.copyNodes.back();
      n.node = node;
      n.dst = dst;
      n.src = src;
      n.bytes = bytes;
      return node;
    };

    void launch() {
      CHECK_CU(cuGraphInstantiateWithFlags(&graph.graphExec, graph.graph, 0));
      CHECK_CU(cuGraphDebugDotPrint(graph.graph, fmt::sprintf("graph-%d.dot", group->rank).c_str(), 0));

      CHECK_CU(cuGraphLaunch(graph.graphExec, group->stream));
    }
  };

  template<typename Graph>
  struct GraphUpdater {
    Graph& graph;
    Group* group;
    ProcessGroupImpl& impl;
    size_t copyIndex = 0;

    GraphUpdater(Graph& graph, Group* group, ProcessGroupImpl& impl) : graph(graph), group(group), impl(impl) {}

    void* addKernel(void*, CUfunction function) {
      return nullptr;
    }

    void* addCopy(void*, uintptr_t dst, uintptr_t src, size_t bytes) {
      auto& n = graph.copyNodes.at(copyIndex);
      ++copyIndex;
      if (n.dst != dst || n.src != src || n.bytes != bytes) {
        n.dst = dst;
        n.src = src;
        n.bytes = bytes;
        CUDA_MEMCPY3D copyParams;
        std::memset(&copyParams, 0, sizeof(copyParams));
        copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copyParams.dstDevice = 0;
        copyParams.srcDevice = 0;
        copyParams.WidthInBytes = 0;
        copyParams.Height = 1;
        copyParams.Depth = 1;
        copyParams.dstDevice = n.dst;
        copyParams.srcDevice = n.src;
        copyParams.WidthInBytes = n.bytes;
        // fmt::printf("change copy node dst %#x src %#x bytes %d\n", n.dst, n.src, n.bytes);
        CHECK_CU(cuGraphExecMemcpyNodeSetParams(graph.graphExec, n.node, &copyParams, group->cuContext));
      }
      return nullptr;
    }

    void launch() {
      impl.trace("cuGraphLaunch");
      CHECK_CU(cuGraphLaunch(graph.graphExec, group->stream));
      impl.trace("post-cuGraphLaunch");
    }
  };

  AllGatherGraph allGatherGraph;

  c10::intrusive_ptr<tccl_work::Work>
  _allgather_base(at::Tensor& output, at::Tensor& input, const c10d::AllgatherOptions& opts) {
    trace("_allgather_base");
    std::lock_guard l(mutex);

    size_t size = this->size;

    uint32_t stepValue = nextStepValue.fetch_add(256);
    CHECK(stepValue < 0x10000000);

    QueueEntryAllGather* e = group->cpuThread->freelistAllGather.pop();
    e->task = taskAllgather;
    e->stepValue = stepValue;

    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(input.is_cuda());
    TORCH_CHECK(input.device().index() == group->deviceIndex);
    TORCH_CHECK(output.is_contiguous());
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(output.device().index() == group->deviceIndex);

    size_t bytes = input.numel() * input.itemsize();
    size_t outputBytes = bytes * size;

    TORCH_CHECK(output.numel() * output.itemsize() == outputBytes);

    e->outputAddresses.clear();
    e->inputAddress = (uintptr_t)input.data_ptr();
    e->outputAddress = (uintptr_t)output.data_ptr();
    e->bytes = bytes;
    e->inputBytesReady = 0;

    TORCH_CHECK((e->inputAddress & 0xf) == 0);
    TORCH_CHECK((e->outputAddress & 0xf) == 0);
    TORCH_CHECK((bytes & 0xf) == 0);

    std::shared_ptr<Op> op = getOp();
    CHECK_CU(cuEventRecord(op->inputEvent, c10::cuda::getCurrentCUDAStream()));
    CHECK_CU(cuStreamWaitEvent(group->stream, op->inputEvent, CU_EVENT_WAIT_DEFAULT));

    IpcMapper* ipcMapper = &*group->ipcMapper;

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    trace("ipcMapper");

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, e->inputAddress, bytes, [ptr = &e->peerInputAddresses[i]](uintptr_t address) { *ptr = address; });

      ipcMapper->requestAddress(
          i, e->outputAddress, outputBytes, [ptr = &e->peerOutputAddresses[i]](uintptr_t address) { *ptr = address; });
    }

    ipcMapper->wait();

    trace("enqueue");

    Group* group = &*this->group;

    uintptr_t inputAddress = e->inputAddress;
    uintptr_t outputAddress = e->outputAddress;

    if (group->extraStreams[0] == nullptr) {
      for (auto& v : group->extraStreams) {
        CHECK_CU(cuStreamCreate(&v, CU_STREAM_NON_BLOCKING));
        // CHECK_CU(cuStreamCreateWithPriority(&v, CU_STREAM_NON_BLOCKING, -100));
      }
      for (auto& v : group->extraEvents) {
        CHECK_CU(cuEventCreate(&v, CU_EVENT_DISABLE_TIMING));
      }
    }

    AllGather& allGather = *group->allGather;

    if (!allGather.cuModule) {
      allGather.compile();
    }

    size_t inputChunks = 1;

    group->cpuThread->enqueue(e);

    trace("sync");

    for (size_t i : peerIndices) {
      AddressPair ad;
      ad.inputAddress = e->peerInputAddresses[i];
      ad.inputBytes = bytes;
      ad.outputAddress = e->peerOutputAddresses[i];
      ad.outputBytes = outputBytes;
      (*group->getPeerVar(i, group->peerAddrs))[group->peerMyRemoteIndex[i]] = ad;
    }
    group->myStepCounter->store(stepValue);
    futexWakeAll(group->myStepCounter);

    for (size_t i : peerIndices) {
      futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue);
      auto& peerAddrs = *group->peerAddrs;
      CHECK(peerAddrs[i].inputAddress && peerAddrs[i].outputAddress);
      CHECK(peerAddrs[i].inputBytes == bytes);
      CHECK(peerAddrs[i].outputBytes == outputBytes);
    }

    trace("launch");

    // if (!allGatherGraph.graph) {
    //   auto& graph = allGatherGraph;

    //   std::array<void*, 1> params = {nullptr};

    //   CHECK_CU(cuGraphCreate(&graph.graph, 0));
    //   CUDA_KERNEL_NODE_PARAMS kernelParams;
    //   std::memset(&kernelParams, 0, sizeof(kernelParams));
    //   kernelParams.func = nullptr;
    //   kernelParams.gridDimX = 1;
    //   kernelParams.gridDimY = 1;
    //   kernelParams.gridDimZ = 1;
    //   kernelParams.blockDimX = 1;
    //   kernelParams.blockDimY = 1;
    //   kernelParams.blockDimZ = 1;
    //   kernelParams.kernelParams = params.data();

    //   auto addKernel = [&](std::vector<CUgraphNode> dependencies, CUfunction function) {
    //     kernelParams.func = function;
    //     CUgraphNode node;
    //     CHECK_CU(cuGraphAddKernelNode(&node, graph.graph, dependencies.data(), dependencies.size(), &kernelParams));
    //     graph.nodes.push_back(node);
    //     return node;
    //   };

    //   auto addCopy = [&](std::vector<CUgraphNode> dependencies, uintptr_t dst, uintptr_t src, size_t bytes) {
    //     CUDA_MEMCPY3D copyParams;
    //     std::memset(&copyParams, 0, sizeof(copyParams));

    //     TORCH_CHECK(dst != 0);
    //     TORCH_CHECK(src != 0);
    //     TORCH_CHECK(bytes != 0);

    //     copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    //     copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    //     copyParams.dstDevice = dst;
    //     copyParams.srcDevice = src;
    //     copyParams.WidthInBytes = bytes;
    //     copyParams.Height = 1;
    //     copyParams.Depth = 1;
    //     fmt::printf("add copy node dst %#x src %#x bytes %d\n", dst, src, bytes);
    //     CUgraphNode node;
    //     CHECK_CU(cuGraphAddMemcpyNode(
    //         &node, graph.graph, dependencies.data(), dependencies.size(), &copyParams, group->cuContext));
    //     graph.nodes.push_back(node);
    //     graph.copyNodes.emplace_back();
    //     auto& n = graph.copyNodes.back();
    //     n.node = node;
    //     n.dst = dst;
    //     n.src = src;
    //     n.bytes = bytes;
    //     return node;
    //   };

    //   addCopy({}, e->outputAddress + bytes * rank, inputAddress, bytes);

    //   auto* prev = addKernel({}, allGather.cuAllgatherEntry);
    //   auto* entry = prev;

    //   for (size_t i : peerIndices) {
    //     auto& peerAddrs = *group->peerAddrs;
    //     prev = addCopy({prev}, outputAddress + bytes * ipcRanks[i], peerAddrs[i].inputAddress, bytes);
    //   }

    //   for (auto& v : allGather.proxyDestinationInfo) {
    //     size_t n = &v - allGather.proxyDestinationInfo.data();
    //     size_t i = v.proxyPeerIndex;
    //     size_t source = v.source;
    //     size_t recvSource = allGather.proxyInfo.at(n).source;
    //     auto stream = group->stream;
    //     auto& peerAddrs = *group->peerAddrs;
    //     size_t chunkSize = std::max(bytes / Group::dataChunks, (size_t)(1024 * 1024));
    //     size_t offset = 0;
    //     for (size_t c = 0; c != Group::dataChunks; ++c) {
    //       size_t nbytes = std::min(bytes - offset, chunkSize);
    //       if (nbytes > 0) {
    //         // CHECK_CU(cuLaunchKernel(
    //         //     allGather.cuAllgatherWaitForProxy.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
    //         //     nullptr));
    //         CHECK_CU(cuLaunchKernel(
    //             allGather.cuAllgatherWaitForRecv.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
    //             nullptr));
    //         CHECK_CU(cuLaunchKernel(
    //             allGather.cuAllgatherWaitForReady.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
    //             nullptr));
    //         CHECK(bytes * source + offset + nbytes <= outputBytes);
    //         auto e = cuMemcpyDtoDAsync(
    //             outputAddress + bytes * source + offset, peerAddrs[i].outputAddress + bytes * source + offset,
    //             nbytes, stream);
    //         if (e != CUDA_SUCCESS) {
    //           fmt::printf(
    //               "memcpy failed. i %d source %d c %d offset %#x nbytes %#x outputAddress %#x
    //               peerAddrs[i]
    //                   .outputAddress "
    //                                  "%#x\n",
    //               i, source, c, offset, nbytes, outputAddress, peerAddrs[i].outputAddress);
    //           CHECK_CU(e);
    //         }
    //       }
    //       offset += nbytes;
    //     }
    //   }

    //   prev = addKernel({prev}, allGather.cuAllgatherExit);

    //   CHECK_CU(cuGraphInstantiateWithFlags(&graph.graphExec, graph.graph, 0));

    //   CHECK_CU(cuGraphDebugDotPrint(graph.graph, fmt::sprintf("graph-%d.dot", rank).c_str(), 0));
    // }

    // CUDA_MEMCPY3D copyParams;
    // std::memset(&copyParams, 0, sizeof(copyParams));
    // copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    // copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    // copyParams.dstDevice = 0;
    // copyParams.srcDevice = 0;
    // copyParams.WidthInBytes = 0;
    // copyParams.Height = 1;
    // copyParams.Depth = 1;

    // size_t copyIndex = 0;
    // auto checkCopy = [&](uintptr_t dst, uintptr_t src, size_t bytes) {
    //   auto& n = allGatherGraph.copyNodes.at(copyIndex);
    //   ++copyIndex;
    //   if (n.dst != dst || n.src != src || n.bytes != bytes) {
    //     n.dst = dst;
    //     n.src = src;
    //     n.bytes = bytes;
    //     copyParams.dstDevice = n.dst;
    //     copyParams.srcDevice = n.src;
    //     copyParams.WidthInBytes = n.bytes;
    //     fmt::printf("change copy node dst %#x src %#x bytes %d\n", n.dst, n.src, n.bytes);
    //     CHECK_CU(cuGraphExecMemcpyNodeSetParams(allGatherGraph.graphExec, n.node, &copyParams, group->cuContext));
    //   }
    // };

    // checkCopy(e->outputAddress + bytes * rank, inputAddress, bytes);
    // for (size_t i : peerIndices) {
    //   auto& peerAddrs = *group->peerAddrs;
    //   checkCopy(outputAddress + bytes * ipcRanks[i], peerAddrs[i].inputAddress, bytes);
    // }

    // CHECK_CU(cuGraphLaunch(allGatherGraph.graphExec, group->stream));

    auto graph = [&](auto&& builder) {
      builder.addCopy({}, e->outputAddress + bytes * rank, inputAddress, bytes);

      auto* prev = builder.addKernel({}, allGather.cuAllgatherEntry);
      auto* entry = prev;

      for (size_t i : peerIndices) {
        auto& peerAddrs = *group->peerAddrs;
        prev = builder.addCopy(prev, outputAddress + bytes * ipcRanks[i], peerAddrs[i].inputAddress, bytes);
      }

      for (auto& v : allGather.proxyDestinationInfo) {
        size_t n = &v - allGather.proxyDestinationInfo.data();
        size_t i = v.proxyPeerIndex;
        size_t source = v.source;
        size_t recvSource = allGather.proxyInfo.at(n).source;
        auto stream = group->stream;
        auto& peerAddrs = *group->peerAddrs;
        size_t chunkSize = std::max(bytes / Group::dataChunks, (size_t)(1024 * 1024));
        size_t offset = 0;
        for (size_t c = 0; c != Group::dataChunks; ++c) {
          size_t nbytes = std::min(bytes - offset, chunkSize);
          if (nbytes > 0) {
            prev = builder.addKernel(prev, allGather.cuAllgatherWaitForRecv.at(n).at(c));
            prev = builder.addKernel(prev, allGather.cuAllgatherWaitForReady.at(n).at(c));
            prev = builder.addCopy(
                prev, outputAddress + bytes * source + offset, peerAddrs[i].outputAddress + bytes * source + offset,
                nbytes);
          }
          offset += nbytes;
        }
      }

      prev = builder.addKernel(prev, allGather.cuAllgatherExit);

      builder.launch();
    };

    if (!allGatherGraph.graph) {
      graph(GraphBuilder(allGatherGraph, group));
    } else {
      graph(GraphUpdater(allGatherGraph, group, *this));
    }

    // std::array<void*, 1> params = {&stepValue};
    // CHECK_CU(cuLaunchKernel(allGather.cuAllgatherEntry, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));
    // CHECK_CU(cuEventRecord(op->inputEvent, group->stream));

    // CHECK_CU(cuStreamWaitEvent(group->extraStreams[0], op->inputEvent, CU_EVENT_WAIT_DEFAULT));
    // CHECK_CU(cuMemcpyDtoDAsync(e->outputAddress + bytes * rank, inputAddress, bytes, group->extraStreams[0]));
    // CHECK_CU(cuEventRecord(group->extraEvents[0], group->extraStreams[0]));

    // for (size_t i : peerIndices) {
    //   auto stream = group->stream;
    //   auto& peerAddrs = *group->peerAddrs;
    //   CHECK_CU(cuStreamWaitEvent(stream, op->inputEvent, CU_EVENT_WAIT_DEFAULT));
    //   CHECK_CU(cuMemcpyDtoDAsync(outputAddress + bytes * ipcRanks[i], peerAddrs[i].inputAddress, bytes, stream));
    // }

    // for (auto& v : allGather.proxyDestinationInfo) {
    //   size_t n = &v - allGather.proxyDestinationInfo.data();
    //   size_t i = v.proxyPeerIndex;
    //   size_t source = v.source;
    //   size_t recvSource = allGather.proxyInfo.at(n).source;
    //   auto stream = group->stream;
    //   auto& peerAddrs = *group->peerAddrs;
    //   size_t chunkSize = std::max(bytes / Group::dataChunks, (size_t)(1024 * 1024));
    //   size_t offset = 0;
    //   for (size_t c = 0; c != Group::dataChunks; ++c) {
    //     size_t nbytes = std::min(bytes - offset, chunkSize);
    //     if (nbytes > 0) {
    //       // CHECK_CU(cuLaunchKernel(
    //       //     allGather.cuAllgatherWaitForProxy.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
    //       //     nullptr));
    //       CHECK_CU(cuLaunchKernel(
    //           allGather.cuAllgatherWaitForRecv.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
    //           nullptr));
    //       CHECK_CU(cuLaunchKernel(
    //           allGather.cuAllgatherWaitForReady.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
    //           nullptr));
    //       CHECK(bytes * source + offset + nbytes <= outputBytes);
    //       auto e = cuMemcpyDtoDAsync(
    //           outputAddress + bytes * source + offset, peerAddrs[i].outputAddress + bytes * source + offset, nbytes,
    //           stream);
    //       if (e != CUDA_SUCCESS) {
    //         fmt::printf(
    //             "memcpy failed. i %d source %d c %d offset %#x nbytes %#x outputAddress %#x
    //             peerAddrs[i].outputAddress "
    //             "%#x\n",
    //             i, source, c, offset, nbytes, outputAddress, peerAddrs[i].outputAddress);
    //         CHECK_CU(e);
    //       }
    //     }
    //     offset += nbytes;
    //   }
    // }
    // CHECK_CU(cuLaunchKernel(allGather.cuAllgatherExit, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));

    // // wait for the local input -> output copy
    // CHECK_CU(cuStreamWaitEvent(group->stream, group->extraEvents[0], CU_EVENT_WAIT_DEFAULT));

    trace("post");

    CHECK_CU(cuEventRecord(op->outputEvent, group->stream));

    group->myStepCounter->store(stepValue + 1);
    futexWakeAll(group->myStepCounter);
    for (size_t i : peerIndices) {
      futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue + 1);
    }

    c10::cuda::CUDAStreamGuard sg(c10::cuda::CUDAStream(
        c10::cuda::CUDAStream::UNCHECKED, c10::Stream(c10::Stream::UNSAFE, input.device(), (long)group->stream)));
    c10::cuda::CUDACachingAllocator::recordStream(input.data().storage().data_ptr(), sg.current_stream());
    c10::cuda::CUDACachingAllocator::recordStream(output.data().storage().data_ptr(), sg.current_stream());
    trace("");
    return makeFuture(op, {output});
  }

  c10::intrusive_ptr<tccl_work::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts) {
    TORCH_CHECK(false, "allgather impl");
  }
};

ProcessGroup::ProcessGroup(int rank, int size) : c10d::ProcessGroup(rank, size) {
  impl = std::make_unique<ProcessGroupImpl>(rank, size);
}
ProcessGroup::~ProcessGroup() {
  impl.reset();
}

void ProcessGroup::init(std::string rank0Address) {
  impl->init(rank0Address);
}
std::string ProcessGroup::getAddress() {
  return impl->getAddress();
}

c10::intrusive_ptr<tccl_work::Work>
ProcessGroup::broadcast(std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(false, "broadcast");
}

c10::intrusive_ptr<tccl_work::Work>
ProcessGroup::allreduce(std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  return impl->allreduce(tensors, opts);
}

c10::intrusive_ptr<tccl_work::Work>
ProcessGroup::_allgather_base(at::Tensor& outputbuffer, at::Tensor& inputbuffer, const c10d::AllgatherOptions& opts) {
  return impl->_allgather_base(outputbuffer, inputbuffer, opts);
}

c10::intrusive_ptr<tccl_work::Work> ProcessGroup::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors, std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  return impl->allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<tccl_work::Work> ProcessGroup::barrier(const c10d::BarrierOptions& opts) {
  return impl->barrier(opts);
}

} // namespace moodist
