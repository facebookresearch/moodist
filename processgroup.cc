
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

  // struct CopyNodeInfo {
  //   CUgraphNode node;
  //   uintptr_t dst;
  //   uintptr_t src;
  //   size_t bytes;
  // };

  // struct ReduceNodeInfo {
  //   CUgraphNode node;
  //   uintptr_t dst;
  //   uintptr_t src;
  //   size_t numel;
  // };

  // struct Graph {
  //   CUgraph graph = nullptr;
  //   CUgraphExec graphExec = nullptr;

  //   std::vector<CUgraphNode> nodes;
  //   std::vector<CopyNodeInfo> copyNodes;
  //   std::vector<ReduceNodeInfo> reduceNodes;
  // };

  // struct Dependency {
  //   std::vector<CUgraphNode> vec;
  //   template<typename... T>
  //   Dependency(T... args) : vec({args...}) {
  //     for (auto i = vec.begin(); i != vec.end();) {
  //       if (!*i) {
  //         i = vec.erase(i);
  //       } else {
  //         ++i;
  //       }
  //     }
  //   }
  //   CUgraphNode* data() {
  //     return vec.data();
  //   }
  //   size_t size() {
  //     return vec.size();
  //   }
  //   template<typename... T>
  //   void add(T... v) {
  //     ((v && (vec.push_back(v), 0)), ...);
  //   }
  // };
  // struct NopDependency {
  //   template<typename... T>
  //   NopDependency(T...) {}
  //   template<typename... T>
  //   void add(T...) {}
  // };

  // template<typename Graph>
  // struct GraphBuilder {
  //   Graph& graph;
  //   Group* group;

  //   GraphBuilder(Graph& graph, Group* group) : graph(graph), group(group) {
  //     CHECK_CU(cuGraphCreate(&graph.graph, 0));
  //   }

  //   Dependency dep() {
  //     return Dependency();
  //   }

  //   CUgraphNode addKernel(Dependency dependency, CUfunction function) {
  //     CUDA_KERNEL_NODE_PARAMS kernelParams;
  //     std::memset(&kernelParams, 0, sizeof(kernelParams));
  //     kernelParams.func = function;
  //     kernelParams.gridDimX = 1;
  //     kernelParams.gridDimY = 1;
  //     kernelParams.gridDimZ = 1;
  //     kernelParams.blockDimX = 1;
  //     kernelParams.blockDimY = 1;
  //     kernelParams.blockDimZ = 1;
  //     kernelParams.kernelParams = nullptr;
  //     CUgraphNode node;
  //     CHECK_CU(cuGraphAddKernelNode(&node, graph.graph, dependency.data(), dependency.size(), &kernelParams));
  //     graph.nodes.push_back(node);
  //     return node;
  //   }

  //   CUgraphNode addReduce(Dependency dependency, uintptr_t dst, uintptr_t src, size_t numel) {
  //     CUDA_KERNEL_NODE_PARAMS kernelParams;
  //     std::memset(&kernelParams, 0, sizeof(kernelParams));
  //     kernelParams.func = group->kernels->cuReduce;
  //     kernelParams.gridDimX = group->kernels->reduceGridSize;
  //     kernelParams.gridDimY = 1;
  //     kernelParams.gridDimZ = 1;
  //     kernelParams.blockDimX = group->kernels->reduceBlockSize;
  //     kernelParams.blockDimY = 1;
  //     kernelParams.blockDimZ = 1;
  //     std::array<void*, 3> params = {&dst, &src, &numel};
  //     kernelParams.kernelParams = params.data();
  //     CUgraphNode node;
  //     CHECK_CU(cuGraphAddKernelNode(&node, graph.graph, dependency.data(), dependency.size(), &kernelParams));
  //     graph.nodes.push_back(node);
  //     graph.reduceNodes.emplace_back();
  //     auto& n = graph.reduceNodes.back();
  //     n.node = node;
  //     n.dst = dst;
  //     n.src = src;
  //     n.numel = numel;
  //     return node;
  //   }

  //   CUgraphNode addCopy(Dependency dependency, uintptr_t dst, uintptr_t src, size_t bytes) {
  //     CUDA_MEMCPY3D copyParams;
  //     std::memset(&copyParams, 0, sizeof(copyParams));

  //     CHECK(dst != 0);
  //     CHECK(src != 0);
  //     CHECK(bytes != 0);

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
  //         &node, graph.graph, dependency.data(), dependency.size(), &copyParams, group->cuContext));
  //     graph.nodes.push_back(node);
  //     graph.copyNodes.emplace_back();
  //     auto& n = graph.copyNodes.back();
  //     n.node = node;
  //     n.dst = dst;
  //     n.src = src;
  //     n.bytes = bytes;
  //     return node;
  //   };

  //   void launch() {
  //     CHECK_CU(cuGraphInstantiateWithFlags(&graph.graphExec, graph.graph, 0));
  //     CHECK_CU(cuGraphDebugDotPrint(graph.graph, fmt::sprintf("graph-%d.dot", group->rank).c_str(), 0));

  //     CHECK_CU(cuGraphLaunch(graph.graphExec, group->stream));
  //   }
  // };

  // template<typename Graph>
  // struct GraphUpdater {
  //   Graph& graph;
  //   Group* group;
  //   ProcessGroupImpl& impl;
  //   size_t copyIndex = 0;
  //   size_t reduceIndex = 0;

  //   GraphUpdater(Graph& graph, Group* group, ProcessGroupImpl& impl) : graph(graph), group(group), impl(impl) {}

  //   NopDependency dep() {
  //     return NopDependency();
  //   }

  //   void* addKernel(NopDependency, CUfunction function) {
  //     return (void*)1;
  //   }

  //   void* addReduce(NopDependency, uintptr_t dst, uintptr_t src, size_t numel) {
  //     auto& n = graph.reduceNodes.at(reduceIndex);
  //     ++reduceIndex;
  //     if (n.dst != dst || n.src != src || n.numel != numel) {
  //       n.dst = dst;
  //       n.src = src;
  //       n.numel = numel;
  //       CUDA_KERNEL_NODE_PARAMS kernelParams;
  //       std::memset(&kernelParams, 0, sizeof(kernelParams));
  //       kernelParams.func = group->kernels->cuReduce;
  //       kernelParams.gridDimX = group->kernels->reduceGridSize;
  //       kernelParams.gridDimY = 1;
  //       kernelParams.gridDimZ = 1;
  //       kernelParams.blockDimX = group->kernels->reduceBlockSize;
  //       kernelParams.blockDimY = 1;
  //       kernelParams.blockDimZ = 1;
  //       std::array<void*, 3> params = {&dst, &src, &numel};
  //       kernelParams.kernelParams = params.data();
  //       CHECK_CU(cuGraphExecKernelNodeSetParams(graph.graphExec, n.node, &kernelParams));
  //     }
  //     return (void*)1;
  //   }

  //   void* addCopy(NopDependency, uintptr_t dst, uintptr_t src, size_t bytes) {
  //     auto& n = graph.copyNodes.at(copyIndex);
  //     ++copyIndex;
  //     if (n.dst != dst || n.src != src || n.bytes != bytes) {
  //       n.dst = dst;
  //       n.src = src;
  //       n.bytes = bytes;
  //       CUDA_MEMCPY3D copyParams;
  //       std::memset(&copyParams, 0, sizeof(copyParams));
  //       copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  //       copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  //       copyParams.Height = 1;
  //       copyParams.Depth = 1;
  //       copyParams.dstDevice = n.dst;
  //       copyParams.srcDevice = n.src;
  //       copyParams.WidthInBytes = n.bytes;
  //       // fmt::printf("change copy node dst %#x src %#x bytes %d\n", n.dst, n.src, n.bytes);
  //       CHECK_CU(cuGraphExecMemcpyNodeSetParams(graph.graphExec, n.node, &copyParams, group->cuContext));
  //     }
  //     return (void*)1;
  //   }

  //   void launch() {
  //     // impl.trace("cuGraphLaunch");
  //     CHECK_CU(cuGraphLaunch(graph.graphExec, group->stream));
  //     // impl.trace("post-cuGraphLaunch");
  //   }
  // };

  // Graph allGatherGraph;
  // Graph reduceScatterGraph;

  // c10::intrusive_ptr<tccl_work::Work>
  // _allgather_base_old(at::Tensor& output, at::Tensor& input, const c10d::AllgatherOptions& opts) {
  //   trace("_allgather_base");
  //   std::lock_guard l(mutex);

  //   // fmt::printf("pg %p doing all gather\n", (void*)this);

  //   size_t size = this->size;

  //   uint32_t stepValue = nextStepValue.fetch_add(4096);
  //   CHECK(stepValue < 0x10000000);

  //   QueueEntryAllGather* e = group->cpuThread->freelistAllGather.pop();
  //   e->task = taskAllgather;
  //   e->stepValue = stepValue;

  //   TORCH_CHECK(input.is_contiguous());
  //   TORCH_CHECK(input.is_cuda());
  //   TORCH_CHECK(input.device().index() == group->deviceIndex);
  //   TORCH_CHECK(output.is_contiguous());
  //   TORCH_CHECK(output.is_cuda());
  //   TORCH_CHECK(output.device().index() == group->deviceIndex);

  //   size_t bytes = input.numel() * input.itemsize();
  //   size_t outputBytes = bytes * size;

  //   TORCH_CHECK(output.numel() * output.itemsize() == outputBytes);

  //   e->outputAddresses.clear();
  //   e->inputAddress = (uintptr_t)input.data_ptr();
  //   e->outputAddress = (uintptr_t)output.data_ptr();
  //   e->bytes = bytes;

  //   std::shared_ptr<Op> op = getOp();
  //   CHECK_CU(cuEventRecord(op->inputEvent, c10::cuda::getCurrentCUDAStream()));
  //   CHECK_CU(cuStreamWaitEvent(group->stream, op->inputEvent, CU_EVENT_WAIT_DEFAULT));

  //   IpcMapper* ipcMapper = &*group->ipcMapper;

  //   const auto& ipcRanks = group->ipcRanks;
  //   const auto& peerIndices = group->peerIndices;

  //   trace("ipcMapper");

  //   for (size_t i : peerIndices) {
  //     ipcMapper->requestAddress(
  //         i, e->inputAddress, bytes, [ptr = &e->peerInputAddresses[i]](uintptr_t address) { *ptr = address; });

  //     ipcMapper->requestAddress(
  //         i, e->outputAddress, outputBytes, [ptr = &e->peerOutputAddresses[i]](uintptr_t address) { *ptr = address;
  //         });
  //   }

  //   ipcMapper->wait();

  //   trace("enqueue");

  //   Group* group = &*this->group;

  //   uintptr_t inputAddress = e->inputAddress;
  //   uintptr_t outputAddress = e->outputAddress;

  //   AllGather& allGather = *group->allGather;

  //   if (!allGather.cuAllgatherEntry) {
  //     group->kernels->compile();
  //   }

  //   group->cpuThread->enqueue(e);

  //   trace("sync");

  //   for (size_t i : peerIndices) {
  //     AddressPair ad;
  //     ad.inputAddress = e->peerInputAddresses[i];
  //     ad.inputBytes = bytes;
  //     ad.outputAddress = e->peerOutputAddresses[i];
  //     ad.outputBytes = outputBytes;
  //     (*group->getPeerVar(i, group->peerAddrs))[group->peerMyRemoteIndex[i]] = ad;
  //   }
  //   group->myStepCounter->store(stepValue);
  //   futexWakeAll(group->myStepCounter);

  //   for (size_t i : peerIndices) {
  //     futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue);
  //     auto& peerAddrs = *group->peerAddrs;
  //     CHECK(peerAddrs[i].inputAddress && peerAddrs[i].outputAddress);
  //     CHECK(peerAddrs[i].inputBytes == bytes);
  //     CHECK(peerAddrs[i].outputBytes == outputBytes);
  //   }

  //   trace("launch");

  //   auto graph = [&](auto&& builder) {
  //     builder.addCopy({}, e->outputAddress + bytes * rank, inputAddress, bytes);

  //     auto* prev = builder.addKernel({}, allGather.cuAllgatherEntry);
  //     auto* entry = prev;

  //     for (size_t i : peerIndices) {
  //       auto& peerAddrs = *group->peerAddrs;
  //       prev = builder.addCopy(prev, outputAddress + bytes * ipcRanks[i], peerAddrs[i].inputAddress, bytes);
  //     }

  //     for (auto& v : allGather.proxyDestinationInfo) {
  //       size_t n = &v - allGather.proxyDestinationInfo.data();
  //       size_t i = v.proxyPeerIndex;
  //       size_t source = v.source;
  //       auto stream = group->stream;
  //       auto& peerAddrs = *group->peerAddrs;
  //       size_t chunkSize = std::max(bytes / Group::dataChunks, (size_t)(1024 * 1024));
  //       size_t offset = 0;
  //       for (size_t c = 0; c != Group::dataChunks; ++c) {
  //         size_t nbytes = std::min(bytes - offset, chunkSize);
  //         if (nbytes > 0) {
  //           // if (true) {
  //           if (false) {
  //             prev = builder.addKernel(prev, allGather.cuAllgatherWaitForProxy.at(n).at(c));
  //           } else {
  //             // for debugging, it can be useful to split the wait for proxy kernel into two parts
  //             prev = builder.addKernel(prev, allGather.cuAllgatherWaitForRecvForward.at(n).at(c));
  //             prev = builder.addKernel(prev, allGather.cuAllgatherWaitForReady.at(n).at(c));
  //           }
  //           prev = builder.addCopy(
  //               prev, outputAddress + bytes * source + offset, peerAddrs[i].outputAddress + bytes * source + offset,
  //               nbytes);
  //         }
  //         offset += nbytes;
  //       }
  //     }

  //     prev = builder.addKernel(prev, allGather.cuAllgatherExit);

  //     builder.launch();
  //   };

  //   if (!allGatherGraph.graph) {
  //     graph(GraphBuilder(allGatherGraph, group));
  //   } else {
  //     graph(GraphUpdater(allGatherGraph, group, *this));
  //   }

  //   trace("post");

  //   CHECK_CU(cuEventRecord(op->outputEvent, group->stream));

  //   group->myStepCounter->store(stepValue + 1);
  //   futexWakeAll(group->myStepCounter);
  //   for (size_t i : peerIndices) {
  //     futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue + 1);
  //   }

  //   c10::cuda::CUDAStreamGuard sg(c10::cuda::CUDAStream(
  //       c10::cuda::CUDAStream::UNCHECKED, c10::Stream(c10::Stream::UNSAFE, input.device(), (long)group->stream)));
  //   c10::cuda::CUDACachingAllocator::recordStream(input.data().storage().data_ptr(), sg.current_stream());
  //   c10::cuda::CUDACachingAllocator::recordStream(output.data().storage().data_ptr(), sg.current_stream());
  //   trace("");
  //   return makeFuture(op, {output});
  // }

  c10::intrusive_ptr<tccl_work::Work>
  _allgather_base(at::Tensor& output, at::Tensor& input, const c10d::AllgatherOptions& opts) {
    trace("_allgather_base");
    std::lock_guard l(mutex);

    // fmt::printf("pg %p doing all gather\n", (void*)this);

    size_t size = this->size;

    uint32_t stepValue = nextStepValue.fetch_add(4096);
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

    AllGather& allGather = *group->allGather;

    if (!group->kernels->cuModule) {
      group->kernels->compile();
    }

    bool isLocalOnly = allGather.recvRanks.empty();

    if (!isLocalOnly) {
      group->cpuThread->enqueue(e);
    }

    trace("launch");

    AllGatherParameters parameters;
    parameters.bytes = bytes;
    parameters.inputAddress = inputAddress;
    parameters.outputAddress = outputAddress;
    parameters.peerInputAddresses = e->peerInputAddresses;
    parameters.peerOutputAddresses = e->peerOutputAddresses;

    std::array<void*, 1> params = {&parameters};

    if (isLocalOnly) {
      CHECK_CU(cuLaunchKernel(allGather.cuAllGatherLocal, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));
    } else {
      CHECK_CU(cuLaunchKernel(allGather.cuAllGather, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));
    }

    trace("post");

    CHECK_CU(cuEventRecord(op->outputEvent, group->stream));

    if (isLocalOnly) {
      group->cpuThread->freelistAllGather.push(e);
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

  // c10::intrusive_ptr<tccl_work::Work>
  // _reduce_scatter_base_old(at::Tensor& output, at::Tensor& input, const c10d::ReduceScatterOptions& opts) {
  //   trace("_reduce_scatter_base");
  //   std::lock_guard l(mutex);

  //   // fmt::printf("pg %p doing reduce scatter\n", (void*)this);

  //   CHECK(opts.reduceOp == c10d::ReduceOp::SUM);

  //   size_t size = this->size;

  //   uint32_t stepValue = nextStepValue.fetch_add(4096);
  //   CHECK(stepValue < 0x10000000);

  //   QueueEntryReduceScatter* e = group->cpuThread->freelistReduceScatter.pop();
  //   e->task = taskReduceScatter;
  //   e->stepValue = stepValue;

  //   TORCH_CHECK(input.is_contiguous());
  //   TORCH_CHECK(input.is_cuda());
  //   TORCH_CHECK(input.device().index() == group->deviceIndex);
  //   TORCH_CHECK(output.is_contiguous());
  //   TORCH_CHECK(output.is_cuda());
  //   TORCH_CHECK(output.device().index() == group->deviceIndex);

  //   TORCH_CHECK(input.dtype() == torch::Dtype::Float);
  //   TORCH_CHECK(output.dtype() == torch::Dtype::Float);

  //   size_t numel = output.numel();
  //   size_t bytes = numel * output.itemsize();
  //   size_t inputBytes = bytes * size;

  //   TORCH_CHECK(input.numel() * input.itemsize() == inputBytes);

  //   e->outputAddresses.clear();
  //   e->inputAddress = (uintptr_t)input.data_ptr();
  //   e->outputAddress = (uintptr_t)output.data_ptr();
  //   e->bytes = bytes;

  //   TORCH_CHECK((e->inputAddress & 0xf) == 0);
  //   TORCH_CHECK((e->outputAddress & 0xf) == 0);

  //   std::shared_ptr<Op> op = getOp();
  //   CHECK_CU(cuEventRecord(op->inputEvent, c10::cuda::getCurrentCUDAStream()));
  //   CHECK_CU(cuStreamWaitEvent(group->stream, op->inputEvent, CU_EVENT_WAIT_DEFAULT));

  //   IpcMapper* ipcMapper = &*group->ipcMapper;

  //   const auto& ipcRanks = group->ipcRanks;
  //   const auto& peerIndices = group->peerIndices;

  //   trace("ipcMapper");

  //   for (size_t i : peerIndices) {
  //     ipcMapper->requestAddress(
  //         i, e->inputAddress, inputBytes, [ptr = &e->peerInputAddresses[i]](uintptr_t address) { *ptr = address; });

  //     ipcMapper->requestAddress(
  //         i, e->outputAddress, bytes, [ptr = &e->peerOutputAddresses[i]](uintptr_t address) { *ptr = address; });
  //   }

  //   ipcMapper->wait();

  //   trace("enqueue");

  //   Group* group = &*this->group;

  //   uintptr_t inputAddress = e->inputAddress;
  //   uintptr_t outputAddress = e->outputAddress;

  //   ReduceScatter& reduceScatter = *group->reduceScatter;

  //   auto& sendRanks = reduceScatter.sendRanks;
  //   auto& recvRanks = reduceScatter.recvRanks;

  //   bool recompile = false;
  //   auto allocbuffer = [&](auto& buffer, size_t size) {
  //     if (buffer.bytes < size) {
  //       CHECK_CU(cuStreamSynchronize(group->stream));
  //       buffer = {};
  //       buffer = group->allocateDevice(size);
  //       // recompile = true;
  //     }
  //   };
  //   allocbuffer(group->sendBuffer, bytes * sendRanks.size());
  //   allocbuffer(group->recvBuffer, bytes * 2);
  //   allocbuffer(group->temporaryBuffer, bytes * peerIndices.size());

  //   if (!reduceScatter.cuReduceScatterEntry || recompile) {
  //     group->kernels->compile();
  //   }

  //   bool isLocalOnly = reduceScatter.recvRanks.empty();

  //   if (!isLocalOnly) {
  //     group->cpuThread->enqueue(e);
  //   }

  //   trace("sync");

  //   for (size_t i : peerIndices) {
  //     AddressPair ad;
  //     ad.inputAddress = e->peerInputAddresses[i];
  //     ad.inputBytes = inputBytes;
  //     ad.outputAddress = e->peerOutputAddresses[i];
  //     ad.outputBytes = bytes;
  //     (*group->getPeerVar(i, group->peerAddrs))[group->peerMyRemoteIndex[i]] = ad;
  //   }
  //   group->myStepCounter->store(stepValue);
  //   futexWakeAll(group->myStepCounter);

  //   for (size_t i : peerIndices) {
  //     futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue);
  //     auto& peerAddrs = *group->peerAddrs;
  //     CHECK(peerAddrs[i].inputAddress && peerAddrs[i].outputAddress);
  //     CHECK(peerAddrs[i].inputBytes == inputBytes);
  //     CHECK(peerAddrs[i].outputBytes == bytes);
  //   }

  //   trace("launch");

  //   uintptr_t sendAddress = group->sendBuffer.cudaPointer;
  //   uintptr_t recvAddress = group->recvBuffer.cudaPointer;
  //   if (!isLocalOnly) {
  //     CHECK(sendAddress != 0);
  //     CHECK(recvAddress != 0);
  //   }

  //   auto graph = [&](auto&& builder) {
  //     CHECK(outputAddress != 0);
  //     auto* outputReduce = builder.addCopy({}, outputAddress, inputAddress + bytes * rank, bytes);

  //     auto* entry = builder.addKernel(
  //         {}, isLocalOnly ? reduceScatter.cuLocalReduceScatterEntry : reduceScatter.cuReduceScatterEntry);
  //     auto* prev = entry;

  //     if (!isLocalOnly) {
  //       auto addRecv = [&](size_t n) {
  //         prev = builder.addKernel(prev, reduceScatter.cuWaitForRecv.at(n));
  //         outputReduce = builder.addReduce({outputReduce, prev}, outputAddress, recvAddress + bytes * (n % 2),
  //         numel);
  //       };

  //       size_t n = 0;
  //       for (size_t i : reduceScatter.sendRanks) {
  //         uintptr_t addr = sendAddress + bytes * n;
  //         CHECK(
  //             addr >= group->sendBuffer.cudaPointer &&
  //             addr + bytes <= group->sendBuffer.cudaPointer + group->sendBuffer.bytes);
  //         auto* sendReduce = builder.addCopy({}, addr, inputAddress + bytes * i, bytes);
  //         for (size_t peerIndex : peerIndices) {
  //           auto& peerAddrs = *group->peerAddrs;
  //           uintptr_t tempaddr = group->temporaryBuffer.cudaPointer + bytes * peerIndex;
  //           prev = builder.addCopy(prev, tempaddr, peerAddrs[peerIndex].inputAddress + bytes * i, bytes);
  //           sendReduce = builder.addReduce({prev, sendReduce}, addr, tempaddr, numel);
  //           // prev = builder.addReduce({prev, l}, addr, peerAddrs[peerIndex].inputAddress + bytes * i, numel);
  //         }
  //         prev = builder.addKernel(sendReduce, reduceScatter.cuSendReady.at(n));

  //         if (n != 0) {
  //           addRecv(n - 1);
  //         }

  //         ++n;
  //       }
  //       addRecv(n - 1);
  //     }

  //     for (size_t peerIndex : peerIndices) {
  //       auto& peerAddrs = *group->peerAddrs;
  //       uintptr_t tempaddr = group->temporaryBuffer.cudaPointer + bytes * peerIndex;
  //       prev = builder.addCopy(prev, tempaddr, peerAddrs[peerIndex].inputAddress + bytes * rank, bytes);
  //       outputReduce = builder.addReduce({outputReduce, prev}, outputAddress, tempaddr, numel);
  //       // prev = builder.addReduce({prev, input}, outputAddress, peerAddrs[i].inputAddress + bytes * rank, numel);
  //     }
  //     prev = builder.addKernel(
  //         {prev, outputReduce},
  //         isLocalOnly ? reduceScatter.cuLocalReduceScatterExit : reduceScatter.cuReduceScatterExit);

  //     builder.launch();
  //   };

  //   if (!reduceScatterGraph.graph) {
  //     fmt::printf("build graph!\n");
  //     graph(GraphBuilder(reduceScatterGraph, group));
  //   } else {
  //     graph(GraphUpdater(reduceScatterGraph, group, *this));
  //   }

  //   trace("post");

  //   CHECK_CU(cuEventRecord(op->outputEvent, group->stream));

  //   group->myStepCounter->store(stepValue + 1);
  //   futexWakeAll(group->myStepCounter);
  //   for (size_t i : peerIndices) {
  //     futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue + 1);
  //   }

  //   if (isLocalOnly) {
  //     group->cpuThread->freelistReduceScatter.push(e);
  //   }

  //   c10::cuda::CUDAStreamGuard sg(c10::cuda::CUDAStream(
  //       c10::cuda::CUDAStream::UNCHECKED, c10::Stream(c10::Stream::UNSAFE, input.device(), (long)group->stream)));
  //   c10::cuda::CUDACachingAllocator::recordStream(input.data().storage().data_ptr(), sg.current_stream());
  //   c10::cuda::CUDACachingAllocator::recordStream(output.data().storage().data_ptr(), sg.current_stream());
  //   trace("");
  //   return makeFuture(op, {output});
  // }

  c10::intrusive_ptr<tccl_work::Work>
  _reduce_scatter_base(at::Tensor& output, at::Tensor& input, const c10d::ReduceScatterOptions& opts) {
    trace("_reduce_scatter_base");
    std::lock_guard l(mutex);

    // fmt::printf("pg %p doing reduce scatter\n", (void*)this);

    CHECK(opts.reduceOp == c10d::ReduceOp::SUM);

    size_t size = this->size;

    uint32_t stepValue = nextStepValue.fetch_add(4096);
    CHECK(stepValue < 0x10000000);

    QueueEntryReduceScatter* e = group->cpuThread->freelistReduceScatter.pop();
    e->task = taskReduceScatter;
    e->stepValue = stepValue;

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

    TORCH_CHECK(input.numel() * input.itemsize() == inputBytes);

    e->outputAddresses.clear();
    e->inputAddress = (uintptr_t)input.data_ptr();
    e->outputAddress = (uintptr_t)output.data_ptr();
    e->bytes = bytes;

    std::shared_ptr<Op> op = getOp();
    CHECK_CU(cuEventRecord(op->inputEvent, c10::cuda::getCurrentCUDAStream()));
    CHECK_CU(cuStreamWaitEvent(group->stream, op->inputEvent, CU_EVENT_WAIT_DEFAULT));

    IpcMapper* ipcMapper = &*group->ipcMapper;

    const auto& ipcRanks = group->ipcRanks;
    const auto& peerIndices = group->peerIndices;

    trace("ipcMapper");

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, e->inputAddress, inputBytes, [ptr = &e->peerInputAddresses[i]](uintptr_t address) { *ptr = address; });

      ipcMapper->requestAddress(
          i, e->outputAddress, bytes, [ptr = &e->peerOutputAddresses[i]](uintptr_t address) { *ptr = address; });
    }

    ipcMapper->wait();

    trace("enqueue");

    Group* group = &*this->group;

    uintptr_t inputAddress = e->inputAddress;
    uintptr_t outputAddress = e->outputAddress;

    ReduceScatter& reduceScatter = *group->reduceScatter;

    auto& sendRanks = reduceScatter.sendRanks;
    auto& recvRanks = reduceScatter.recvRanks;

    auto allocbuffer = [&](auto& buffer, size_t size) {
      if (buffer.bytes < size) {
        CHECK_CU(cuStreamSynchronize(group->stream));
        buffer = {};
        buffer = group->allocateDevice(size);
      }
    };
    allocbuffer(group->sendBuffer, bytes * sendRanks.size());
    allocbuffer(group->recvBuffer, bytes * recvRanks.size());
    allocbuffer(group->temporaryBuffer, bytes * peerIndices.size());

    if (!group->kernels->cuModule) {
      group->kernels->compile();
    }

    bool isLocalOnly = reduceScatter.recvRanks.empty();

    if (!isLocalOnly) {
      group->cpuThread->enqueue(e);
    }

    trace("launch");

    uintptr_t sendAddress = group->sendBuffer.cudaPointer;
    uintptr_t recvAddress = group->recvBuffer.cudaPointer;
    if (!isLocalOnly) {
      CHECK(sendAddress != 0);
      CHECK(recvAddress != 0);
    }

    ReduceScatterParameters parameters;
    parameters.bytes = bytes;
    parameters.inputAddress = inputAddress;
    parameters.outputAddress = outputAddress;
    parameters.peerInputAddresses = e->peerInputAddresses;
    parameters.peerOutputAddresses = e->peerOutputAddresses;
    parameters.sendAddress = sendAddress;
    parameters.recvAddress = recvAddress;

    std::array<void*, 1> params = {&parameters};

    if (isLocalOnly) {
      CHECK_CU(cuLaunchKernel(
          reduceScatter.cuReduceScatterLocal, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));
    } else {
      CHECK_CU(
          cuLaunchKernel(reduceScatter.cuReduceScatter, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));
    }

    trace("post");

    CHECK_CU(cuEventRecord(op->outputEvent, group->stream));

    if (isLocalOnly) {
      group->cpuThread->freelistReduceScatter.push(e);
    }

    c10::cuda::CUDAStreamGuard sg(c10::cuda::CUDAStream(
        c10::cuda::CUDAStream::UNCHECKED, c10::Stream(c10::Stream::UNSAFE, input.device(), (long)group->stream)));
    c10::cuda::CUDACachingAllocator::recordStream(input.data().storage().data_ptr(), sg.current_stream());
    c10::cuda::CUDACachingAllocator::recordStream(output.data().storage().data_ptr(), sg.current_stream());
    trace("");
    return makeFuture(op, {output});
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

c10::intrusive_ptr<tccl_work::Work> ProcessGroup::_reduce_scatter_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor, const c10d::ReduceScatterOptions& opts) {
  return impl->_reduce_scatter_base(outputTensor, inputTensor, opts);
}

c10::intrusive_ptr<tccl_work::Work> ProcessGroup::barrier(const c10d::BarrierOptions& opts) {
  return impl->barrier(opts);
}

} // namespace moodist
