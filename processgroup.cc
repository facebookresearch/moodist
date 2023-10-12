
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

  ~ProcessGroupImpl() {}

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

  c10::intrusive_ptr<tccl_work::Work>
  _allgather_base(at::Tensor& output, at::Tensor& input, const c10d::AllgatherOptions& opts) {
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

    for (size_t i : peerIndices) {
      ipcMapper->requestAddress(
          i, e->inputAddress, bytes, [ptr = &e->peerInputAddresses[i]](uintptr_t address) { *ptr = address; });

      ipcMapper->requestAddress(
          i, e->outputAddress, outputBytes, [ptr = &e->peerOutputAddresses[i]](uintptr_t address) { *ptr = address; });
    }

    ipcMapper->wait();

    Group* group = &*this->group;

    uintptr_t inputAddress = e->inputAddress;
    uintptr_t outputAddress = e->outputAddress;

    if (group->extraStreams[0] == nullptr) {
      for (auto& v : group->extraStreams) {
        CHECK_CU(cuStreamCreate(&v, CU_STREAM_NON_BLOCKING));
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

    std::array<void*, 1> params = {&stepValue};
    CHECK_CU(cuLaunchKernel(allGather.cuAllgatherEntry, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));
    CHECK_CU(cuEventRecord(op->inputEvent, group->stream));

    CHECK_CU(cuStreamWaitEvent(group->extraStreams[0], op->inputEvent, CU_EVENT_WAIT_DEFAULT));
    CHECK_CU(cuMemcpyDtoDAsync(e->outputAddress + bytes * rank, inputAddress, bytes, group->extraStreams[0]));
    CHECK_CU(cuEventRecord(group->extraEvents[0], group->extraStreams[0]));

    for (size_t i : peerIndices) {
      auto stream = group->stream;

      futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue);
      auto& peerAddrs = *group->peerAddrs;
      CHECK(peerAddrs[i].inputAddress && peerAddrs[i].outputAddress);
      CHECK(peerAddrs[i].inputBytes == bytes);
      CHECK(peerAddrs[i].outputBytes == outputBytes);

      CHECK_CU(cuStreamWaitEvent(stream, op->inputEvent, CU_EVENT_WAIT_DEFAULT));
      CHECK_CU(cuMemcpyDtoDAsync(outputAddress + bytes * ipcRanks[i], peerAddrs[i].inputAddress, bytes, stream));
    }
    // if (!allGather.proxyInfo.empty()) {
    //   // We run this on another stream in the hope that it will run side-by-side with the proxy wait below.
    //   // In case they are in fact scheduled sequentially, the proxy wait kernel also does the forward work,
    //   // such that we avoid deadlocks.
    //   CHECK_CU(cuStreamWaitEvent(group->extraStreams[1], op->inputEvent, CU_EVENT_WAIT_DEFAULT));
    //   CHECK_CU(cuLaunchKernel(
    //       allGather.cuAllgatherForwardProxies, 1, 1, 1, 1, 1, 1, 0, group->extraStreams[1], params.data(), nullptr));
    //   CHECK_CU(cuEventRecord(group->extraEvents[1], group->extraStreams[1]));
    // }
    for (auto& v : allGather.proxyDestinationInfo) {
      size_t n = &v - allGather.proxyDestinationInfo.data();
      size_t i = v.proxyPeerIndex;
      size_t source = v.source;
      auto stream = group->stream;
      auto& peerAddrs = *group->peerAddrs;
      size_t chunkSize = std::max(bytes / Group::dataChunks, (size_t)(1024 * 1024));
      size_t offset = 0;
      for (size_t c = 0; c != Group::dataChunks; ++c) {
        size_t nbytes = std::min(bytes - offset, chunkSize);
        if (nbytes > 0) {
          // CHECK_CU(cuLaunchKernel(
          //     allGather.cuAllgatherWaitForProxy.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
          //     nullptr));
          CHECK_CU(cuLaunchKernel(
              allGather.cuAllgatherWaitForRecv.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
              nullptr));
          CHECK_CU(cuLaunchKernel(
              allGather.cuAllgatherWaitForReady.at(n).at(c), 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(),
              nullptr));
          CHECK(bytes * source + offset + nbytes <= outputBytes);
          auto e = cuMemcpyDtoDAsync(
              outputAddress + bytes * source + offset, peerAddrs[i].outputAddress + bytes * source + offset, nbytes,
              stream);
          if (e != CUDA_SUCCESS) {
            fmt::printf(
                "memcpy failed. i %d source %d c %d offset %#x nbytes %#x outputAddress %#x peerAddrs[i].outputAddress "
                "%#x\n",
                i, source, c, offset, nbytes, outputAddress, peerAddrs[i].outputAddress);
            CHECK_CU(e);
          }
          // CHECK_CU(cuMemcpyDtoDAsync(
          //     outputAddress + bytes * source + offset, peerAddrs[i].second + bytes * source + offset, nbytes,
          //     stream));
        }
        offset += nbytes;
      }
    }

    group->myStepCounter->store(stepValue + 1);
    futexWakeAll(group->myStepCounter);
    for (size_t i : peerIndices) {
      futexWaitWhileLess(group->getPeerVar(i, group->myStepCounter), stepValue + 1);
    }

    // if (!allGather.proxyInfo.empty()) {
    //   // wait for forward proxies
    //   CHECK_CU(cuStreamWaitEvent(group->stream, group->extraEvents[1], CU_EVENT_WAIT_DEFAULT));
    // }

    // todo: inline this into exit
    CHECK_CU(
        cuLaunchKernel(allGather.cuAllgatherCopyAllDone, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));

    CHECK_CU(cuLaunchKernel(allGather.cuAllgatherExit, 1, 1, 1, 1, 1, 1, 0, group->stream, params.data(), nullptr));

    // wait for the local input -> output copy
    CHECK_CU(cuStreamWaitEvent(group->stream, group->extraEvents[0], CU_EVENT_WAIT_DEFAULT));

    CHECK_CU(cuEventRecord(op->outputEvent, group->stream));

    c10::cuda::CUDAStreamGuard sg(c10::cuda::CUDAStream(
        c10::cuda::CUDAStream::UNCHECKED, c10::Stream(c10::Stream::UNSAFE, input.device(), (long)group->stream)));
    c10::cuda::CUDACachingAllocator::recordStream(input.data().storage().data_ptr(), sg.current_stream());
    c10::cuda::CUDACachingAllocator::recordStream(output.data().storage().data_ptr(), sg.current_stream());
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
