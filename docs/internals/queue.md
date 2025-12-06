# Queue Internals

This document describes the internal architecture of the moodist Queue implementation.

## Overview

The Queue provides inter-rank communication with put/get semantics. Each queue has a "location" (the rank that owns the storage). Any rank can put to or get from a queue, regardless of where the storage lives.

**Key files:**
- `queue.cc` - Main implementation (~1100 lines)
- `queue.h` - Public header and internal declarations
- `cputhread.cc` - Background thread that processes queue operations (lines 4650-4890)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Code                                    │
│   queue.put_tensor(tensor)  /  queue.get_tensor()                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      QueueImpl (queue.cc)                            │
│   - put() method (lines 560-697)                                    │
│   - get() method (lines 400-540)                                    │
│   - Manages QueueStorage pointer (local or proxy)                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           ┌───────────────┐       ┌───────────────┐
           │  Local Path   │       │  Remote Path  │
           │ (same rank)   │       │ (diff rank)   │
           └───────────────┘       └───────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CpuThread (cputhread.cc)                        │
│   - Background thread processing queue operations                   │
│   - Handles network send/receive                                    │
│   - Invokes callbacks when operations complete                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Data Structures

### QueueStorage (queue.cc:125-150)
The actual queue storage, located on one rank.

```cpp
struct QueueStorage {
  SpinMutex mutex;
  std::deque<TensorDataPtr> vector;      // The actual queue data
  size_t size = 0;                        // Current queue size
  int location;                           // Which rank owns this
  bool streaming;                         // Streaming mode (not fully implemented)

  // For ordering puts
  size_t putQueueSize = 0;
  std::deque<QueuedPut> putQueue;

  // For handling incoming remote puts
  HashMap<uint32_t, std::unique_ptr<IncomingSource>> incoming;

  // For remote gets
  std::deque<PendingRemoteGet> pendingRemoteGets;
  HashMap<uint32_t, RemoteGetResult*> activeOutgoingRemoteGets;
};
```

### QueueWork / QueueWorkImpl (queue.cc:40-80, 762-775)
Handle returned by `put_tensor()` for tracking async operation completion.

```cpp
struct QueueWorkImpl {
  std::atomic_uint32_t done = 0;          // Set to 1 when operation completes
  bool queuedPutReady = false;            // For ordering queued puts
  WorkCudaDonePtr cudaDone;               // For CUDA completion signaling
  WorkCudaMappedDonePtr cudaMappedDone;   // For CUDA-mapped completion

  void wait();  // Blocks until done == 1 using futex
};

struct QueueWork {
  std::shared_ptr<QueueWorkImpl> impl;
  std::optional<c10::Storage> storage;    // Holds tensor storage alive
  bool waitOnDestroy = true;              // Whether destructor calls wait()

  ~QueueWork() {
    if (waitOnDestroy) {
      wait();
    }
  }
};
```

## Put Operation Flow

### CPU Tensor Put (Local Queue)

```
1. User: queue.put_tensor(cpu_tensor)
   └─> QueueImpl::put() [queue.cc:562]

2. put() allocates TensorDataPtr, copies tensor data if needed
   └─> Checks if value.is_cpu() [line 598]

3. Calls sendPut() [line 628]
   └─> Creates QueueEntryQueuePut with callback
   └─> Enqueues to group->cpuThread [line 327]
   └─> Returns immediately

4. Returns QueueWork with:
   └─> impl set to track completion
   └─> storage set to keep tensor alive
   └─> waitOnDestroy controls destructor behavior

5. [ASYNC] CpuThread processes taskQueuePut
   └─> executeQueuePut() [cputhread.cc:4741]
   └─> Serializes tensor data inline (for small CPU tensors)
   └─> Stores callback in queuePutCallbacks[putKey] [line 4766]
   └─> Sends buffer to self via sendBuffer() [line 4795]

6. [ASYNC] CpuThread receives the message (self-send)
   └─> onRecvQueuePut() [cputhread.cc:4650]
   └─> For inlined data: immediately calls queueFinish() [line 4701-4703]
       └─> queueFinish() adds tensor to qs->vector, increments qs->size
   └─> Enqueues sendQueueReadFinished() as callback [line 4704]

7. [ASYNC] CpuThread sends QueueReadFinished message
   └─> sendQueueReadFinished() [cputhread.cc:4844]

8. [ASYNC] CpuThread receives QueueReadFinished
   └─> onRecvQueueReadFinished() [cputhread.cc:4799]
   └─> executeQueueReadFinished() [cputhread.cc:4806]
   └─> Looks up and invokes callback [line 4812]
   └─> Callback sets work->done = 1, wakes futex [queue.cc:319-320]

9. When QueueWork is destroyed (if waitOnDestroy=true):
   └─> Destructor calls wait()
   └─> Blocks until done == 1
```

### CUDA Tensor Put (Local Queue)

```
1. User: queue.put_tensor(cuda_tensor)
   └─> QueueImpl::put() [queue.cc:562]

2. put() allocates TensorDataPtr
   └─> Checks !value.is_cpu() [line 630]
   └─> Forces waitOnDestroy = true (CUDA always waits)

3. Sets up CUDA completion tracking
   └─> If rdmaSupportsCuda: uses cudaDone pointer [lines 634-640]
   └─> Else: copies to CPU buffer, uses cudaMappedDone [lines 641-651]

4. Queues the put operation [lines 655-668]
   └─> Will be processed when CUDA stream reaches the host callback

5. Launches host callback on CUDA stream [lines 670-693]
   └─> cuLaunchHostFunc() schedules callback after GPU work completes
   └─> Callback calls sendPut() to actually send the data

6. Returns QueueWork with:
   └─> impl set to track completion
   └─> storage set to keep tensor memory alive for RDMA
   └─> waitOnDestroy = true (always, for CUDA safety)

7-11. Same as CPU steps 5-8, but triggered by CUDA callback

12. When QueueWork is destroyed:
   └─> Destructor always calls wait() (waitOnDestroy forced true)
   └─> Ensures RDMA completes before storage is freed
```

### Remote Put (Different Rank)

Similar flow, but data goes over the network instead of self-send.
The `QueueReadFinished` message comes back from the remote rank.

## Get Operation Flow

### Local Get

```
1. User: queue.get_tensor()
   └─> QueueImpl::get() [queue.cc:400]

2. If queue is not empty:
   └─> Directly pops from qs->vector [lines 473-490]
   └─> Returns immediately with tensor

3. If queue is empty and blocking:
   └─> Waits on futex until data available
```

### Remote Get

```
1. User: queue.get_tensor() on rank A, queue located on rank B
   └─> QueueImpl::get() [queue.cc:400]

2. Sends get request to rank B
   └─> sendStartGet() [line 340]

3. Rank B processes request
   └─> If data available: sends it back via sendPutRemote()
   └─> If empty: queues the get request

4. Rank A receives data
   └─> queueFinish() with getKey set [queue.cc:905-917]
   └─> Wakes waiting thread
```

## Design Notes

### waitOnDestroy Parameter

The `put()` function accepts a `waitOnDestroy` parameter in C++ that controls whether the
`QueueWork` destructor calls `wait()`.

- `waitOnDestroy=true`: Destructor blocks until transfer completes
- `waitOnDestroy=false`: Destructor does not wait (fire-and-forget)

The Python layer does not expose this parameter directly. Instead, the behavior differs by method:
- `put_tensor()`: Always passes `waitOnDestroy=true` (safe for tensor memory)
- `put_object()`: Always passes `waitOnDestroy=false` (fire-and-forget, since data is serialized to internal buffer)

For CUDA tensors, the C++ layer forces `waitOnDestroy=true` regardless of the parameter value,
since we need to keep storage alive until the transfer completes (RDMA requires valid source memory).

## Known Issues

### FIXME: Local vs remote get ordering (queue.cc:473-475)

```cpp
// fixme: this needs to be refactored such that -
//        local get is fairly queued alongside remote gets
//        multi-threaded gets (local & remote) on the same object are also fairly queued
```

Local gets directly access the queue, while remote gets go through the message queue.
This can lead to unfair scheduling where local gets always win.

### Streaming not implemented (queue.cc:201-203)

```cpp
if (streaming) {
  throw std::runtime_error("Queue: streaming is not fully implemented");
}
```

Streaming would allow the caller to provide a pre-allocated output tensor.

## Synchronization Model

### What `work.wait()` guarantees

For CPU tensors:
- The tensor data has been copied to the queue storage
- `qs->size` has been incremented
- `qsize()` will reflect the put

For CUDA tensors:
- The GPU stream has completed the copy
- BUT: the cpuThread may not have finished processing
- `torch.cuda.synchronize()` is needed before `qsize()` will be accurate

### Thread safety

- `QueueStorage::mutex` (SpinMutex) protects the queue data structures
- Operations are serialized through cpuThread for network operations
- Local gets bypass cpuThread (source of the ordering issue above)

## Message Types (cputhread.cc)

| Op Type | Direction | Purpose |
|---------|-----------|---------|
| `opTypeQueuePut` | sender → queue owner | Send tensor to queue |
| `opTypeQueueReadFinished` | queue owner → sender | Acknowledge put complete |
| `opTypeQueueGet` | getter → queue owner | Request tensor from queue |

## Performance Considerations

- Small CPU tensors are inlined in the message (no separate RDMA read)
- Large tensors use RDMA for data transfer
- CUDA tensors wait for GPU stream before sending
- Local operations still go through cpuThread for consistency (could be optimized)
