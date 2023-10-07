#include "cputhread.h"
#include "clock.h"
#include "common.h"
#include "freelist.h"
#include "ipc_mapper.h"
#include "setup_comms.h"

#include "fmt/printf.h"

#include <memory>
#include <utility>

namespace moodist {

CpuThread::CpuThread(Group* group) {
  this->group = group;
}
CpuThread::~CpuThread() {
  if (thread.joinable()) {
    QueueEntry* e = freelistTerminate.pop();
    e->task = taskTerminate;
    std::unique_lock l(mutex);
    queue.push_back(*e);
    l.unlock();
    ++queueSize;
    futexWakeAll(&queueSize);
    terminate = true;

    thread.join();
  }
}

void CpuThread::start() {
  thread = std::thread([this] { entry(); });
}

inline void volatileMemcpy(void* dst, volatile void* src, size_t size) {
  TORCH_CHECK(size % sizeof(unsigned long) == 0);
  volatile unsigned long* s = (volatile unsigned long*)src;
  char* d = (char*)dst;
  while (size) {
    unsigned long v = *s++;
    std::memcpy(d, &v, sizeof(v));
    d += sizeof(unsigned long);
    size -= sizeof(unsigned long);
  }
}

struct QuitCpuThread {};

void CpuThread::entry() {
  try {
    const size_t rank = group->rank;
    const size_t size = group->size;

    // CHECK_CU(cuCtxSetCurrent(group->cuContext));

    // CpuProcess& process = *group->cpuprocess;

    CUcontext cuCtx;
    // cuCtxCreate(&cuCtx, 0, group->cuDevice);
    // CHECK_CU(cuCtxSetCurrent(cuCtx));
    CHECK_CU(cuCtxSetCurrent(group->cuContext));

    CUstream stream;
    // CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CHECK_CU(cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, -10000));

    std::atomic_uint32_t* myStepCounter = (std::atomic_uint32_t*)group->ipcMapper->getMySharedMem(0x100, 4);

    while (true) {

      // if (op.queueSize != 0) {
      //   fmt::printf("another op queued immediately!\n");
      // }
      while (queueSize == 0) {
        futexWait(&queueSize, 0, std::chrono::seconds(1));
      }
      --queueSize;

      std::unique_lock l(mutex);
      TORCH_CHECK(!queue.empty());
      QueueEntry& queueEntry = queue.front();
      queue.pop_front();
      l.unlock();

      TORCH_CHECK(queueEntry.stepValue < (1ul << 31));

      if (queueEntry.task == taskAllgather) {
        QueueEntryAllGather& params = (QueueEntryAllGather&)queueEntry;

        //fmt::printf("cpu thread got all gather\n");

        futexWaitWhileLess(myStepCounter, queueEntry.stepValue + size);

        //fmt::printf("cpu thread all gather step 1 done!\n");

        futexWaitWhileLess(myStepCounter, queueEntry.stepValue + size * 2);

        //fmt::printf("cpu thread all gather all done!\n");

        freelistAllGather.push(&params);
      } else if (queueEntry.task == taskTerminate) {
        freelistTerminate.push(&queueEntry);
        break;
      } else {
        throw std::runtime_error(fmt::sprintf("internal error: unknown task %d", queueEntry.task));
      }
    }

  } catch (const std::exception& e) {
    fmt::fprintf(stderr, "Error: %s\n", e.what());
    fflush(stderr);
    TORCH_CHECK(false, "Moodist cpu thread got an exception");
  } catch (QuitCpuThread) {
  }
}

} // namespace moodist
