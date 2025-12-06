# moodist.Queue

A distributed queue for inter-rank communication of tensors and Python objects.

## Class Signature

```python
class Queue:
    def __init__(
        self,
        process_group: MoodistProcessGroup | str,
        location: int | Iterable[int],
        streaming: bool = False,
        name: str | None = None
    )
```

## Overview

`Queue` is a flexible inter-process communication primitive that allows ranks to send and receive tensors and Python objects. Unlike collective operations that have fixed communication patterns, queues provide point-to-point and multicast semantics with FIFO (first-in, first-out) ordering.

**Key Features:**
- Send tensors directly between ranks with minimal overhead
- Send arbitrary Python objects (automatically serialized)
- Blocking and non-blocking receive operations
- Timeout support for receives
- Multicast mode to broadcast to multiple receivers
- Transactional puts for atomic batching
- Named queues that persist across serialization

**When to use `Queue` vs `compile_op`:**
- Use `compile_op` for regular patterns with known tensor shapes at compile time
- Use `Queue` when you need:
  - Dynamic, irregular communication patterns
  - Variable-sized messages
  - Producer-consumer patterns
  - Sending Python objects (not just tensors)
  - Control messages between ranks

## Constructor Parameters

### `process_group`
**Type:** `MoodistProcessGroup` or `str`

The distributed process group for this queue. If a string is provided, it looks up the group by name using `moodist.find_process_group()`.

### `location`
**Type:** `int` or `Iterable[int]`

The rank(s) where the queue is hosted:
- **Single `int`**: Creates a queue hosted on that rank. Any rank can put to the queue, and any rank can get from it (remote gets are supported).
- **`list[int]`** (multicast): Creates a queue that broadcasts puts to all specified ranks. Each put is delivered to all ranks in the list. Only ranks in the list can get from their local copy.

**Example:**
```python
# Queue hosted on rank 0
queue = moodist.Queue(group, location=0)

# Multicast queue - puts go to ranks 0, 1, and 2
queue = moodist.Queue(group, location=[0, 1, 2])
```

### `streaming`
**Type:** `bool` (default: `False`)

> **Note:** Streaming mode is not fully implemented. Leave as `False`.

### `name`
**Type:** `str` or `None` (default: `None`)

Optional name for the queue. Named queues have two special properties:

1. **Non-blocking construction**: Named queues can be constructed at any time, on any rank, independently. Ranks that don't need the queue don't need to construct it at all. In contrast, unnamed queues require all ranks to construct them together (barrier operation).

2. **Serialization support**: Named queues can be serialized and sent through other queues. This allows a reference to a queue to be transmitted to another process, which can then use it to communicate.

Names must be unique within a process group. If a queue with the same name already exists:
- With the same parameters (`location`, `streaming`): returns the existing queue
- With different parameters: raises an error

**Example:**
```python
# Unnamed queue - all ranks must construct together (barrier)
queue = moodist.Queue(group, location=0)

# Named queue - can be constructed independently on each rank
if rank == 0:
    queue = moodist.Queue(group, location=0, name="data_queue")
# Other ranks can construct later, or not at all if they don't need it

# Named queues can be sent through other queues
control_queue.put_object(queue)  # Send queue reference
received_queue = control_queue.get_object()  # Receive and use it
```

## Methods

### `put_tensor(tensor)`

Send a tensor to the queue.

**Parameters:**
- `tensor`: A contiguous PyTorch tensor (CPU or CUDA)

**Returns:** `QueueWork` object for synchronization

**Synchronization behavior:**
- The `QueueWork` destructor automatically calls `wait()`, blocking until the transfer completes
- For **CUDA tensors**: `work.wait()` inserts a GPU stream wait, allowing the CPU to continue
- For **CPU tensors**: `work.wait()` blocks the CPU until the transfer completes

**Usage:**
```python
# Default: waits for completion when work goes out of scope
queue.put_tensor(tensor)

# Capture work object to wait later (allows overlapping work)
work = queue.put_tensor(tensor)
# ... do other work ...
work.wait()
```

### `get_tensor(block=True, timeout=None, return_size=False)`

Receive a tensor from the queue.

**Parameters:**
- `block`: If `True` (default), blocks until a tensor is available. If `False`, raises `queue.Empty` immediately if queue is empty.
- `timeout`: Maximum seconds to wait (only applies when `block=True`). `None` means wait indefinitely.
- `return_size`: If `True`, returns a tuple `(tensor, queue_size)` where `queue_size` is the queue size at the time of the get.

**Returns:**
- If `return_size=False`: The received tensor
- If `return_size=True`: Tuple of `(tensor, queue_size)`

**Raises:** `queue.Empty` if `block=False` and queue is empty, or if timeout expires

**Usage:**
```python
from queue import Empty

# Blocking get
tensor = queue.get_tensor()

# Non-blocking get
try:
    tensor = queue.get_tensor(block=False)
except Empty:
    print("Queue is empty")

# Get with timeout
try:
    tensor = queue.get_tensor(timeout=5.0)  # Wait up to 5 seconds
except Empty:
    print("Timeout waiting for data")
```

### `put_object(object)`

Send an arbitrary Python object to the queue.

The object is serialized using `moodist.serialize()` before being sent. The receiving rank will get a deserialized copy.

**Parameters:**
- `object`: Any picklable Python object

**Returns:** `QueueWork` object for synchronization

**Synchronization behavior:**
- The `QueueWork` destructor does **not** wait (fire-and-forget behavior)
- This is safe because serialized objects are copied to internal buffers, so the original object can be safely modified immediately
- If you need to ensure the object has been delivered, call `.wait()` on the returned work object

**Usage:**
```python
# Default (fire-and-forget): no automatic wait
queue.put_object({'key': 'value', 'data': [1, 2, 3]})
queue.put_object((42, 'hello', None))
queue.put_object(MyCustomClass())

# If you need to wait for delivery:
work = queue.put_object(data)
work.wait()
```

### `get_object(block=True, timeout=None, return_size=False)`

Receive a Python object from the queue.

**Parameters:** Same as `get_tensor()`

**Returns:**
- If `return_size=False`: The received object
- If `return_size=True`: Tuple of `(object, queue_size)`

**Raises:** `queue.Empty` if `block=False` and queue is empty, or if timeout expires

### `qsize()`

Return the current number of items in the queue.

**Returns:** `int`

> **Note:** Only works when the local rank hosts the queue (i.e., for single-location queues where `location == rank`, or for multicast queues where the local rank is in the location list).

### `empty()`

Check if the queue is empty.

**Returns:** `bool` - `True` if queue has no items

> **Note:** Only works when the local rank hosts the queue.

### `wait(timeout=None)`

Wait for items to appear in the queue.

**Parameters:**
- `timeout`: Maximum seconds to wait. `None` means wait indefinitely.

**Returns:** `bool` - `True` if queue has items, `False` if timeout expired

> **Note:** Only works when the local rank hosts the queue.

### `transaction()`

Create a transaction context for atomic batching of puts.

**Returns:** `TransactionContextManager`

Within a transaction, all puts are held until the transaction commits. If the context exits normally, the transaction commits and all puts become visible atomically. If an exception occurs, the transaction is cancelled and all puts are discarded.

**Usage:**
```python
# All three tensors become visible atomically
with queue.transaction() as txn:
    txn.put_tensor(tensor1)
    txn.put_tensor(tensor2)
    txn.put_object({'metadata': 'info'})
# Transaction commits here

# If an error occurs, nothing is sent
with queue.transaction() as txn:
    txn.put_tensor(tensor1)
    raise ValueError("Something went wrong")
    txn.put_tensor(tensor2)  # Never executed
# Transaction cancelled - tensor1 is NOT sent
```

## Examples

### Example 1: Basic Producer-Consumer

One rank produces data, another consumes it.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()

# Queue hosted on rank 1 (consumer)
queue = moodist.Queue(group, location=1)

NUM_TENSORS = 10

if rank == 0:
    # Producer: send tensors to the queue
    for i in range(NUM_TENSORS):
        tensor = torch.randn(100, 100)
        queue.put_tensor(tensor)

elif rank == 1:
    # Consumer: receive and process tensors
    for i in range(NUM_TENSORS):
        tensor = queue.get_tensor()
        # Process tensor...
```

### Example 2: Multicast Broadcasting

Send data to multiple ranks simultaneously.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()

# Multicast queue - rank 0 sends, ranks 1, 2, 3 receive
queue = moodist.Queue(group, location=[1, 2, 3])

if rank == 0:
    # Broadcast data to all receivers
    tensor = torch.randn(50, 50)
    queue.put_tensor(tensor)

elif rank in [1, 2, 3]:
    # Each receiver gets a copy
    tensor = queue.get_tensor()
    print(f"Rank {rank} received tensor of shape {tensor.shape}")
```

### Example 3: Remote Queue Access

Any rank can put to a queue hosted on another rank.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()

# Queue hosted on rank 0
queue = moodist.Queue(group, location=0)

if rank == 0:
    # Receive from all other ranks
    for _ in range(group.size() - 1):
        tensor = queue.get_tensor()
        print(f"Received tensor with sum={tensor.sum().item()}")
else:
    # All other ranks send to rank 0
    tensor = torch.full((10,), float(rank))
    queue.put_tensor(tensor)
```

### Example 4: Transactional Puts

Ensure multiple items are delivered atomically.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()

queue = moodist.Queue(group, location=1)

if rank == 0:
    # Send header + data + footer atomically
    with queue.transaction() as txn:
        txn.put_object({'type': 'batch', 'count': 3})
        txn.put_tensor(torch.randn(10))
        txn.put_tensor(torch.randn(20))
        txn.put_tensor(torch.randn(30))
    # All four items become visible at once

elif rank == 1:
    # Receiver sees either all items or none
    header = queue.get_object()
    assert header['type'] == 'batch'
    tensors = [queue.get_tensor() for _ in range(header['count'])]
```

> **Important:** Tensors and objects share the same queue. Objects are serialized into tensors before being sent. Each `put_tensor` must be matched with a `get_tensor`, and each `put_object` must be matched with a `get_object`. Mixing them (e.g., `put_tensor` followed by `get_object`) will cause deserialization errors.

### Example 5: Named Queues

Named queues can be sent through other queues to establish communication channels dynamically. This example also demonstrates remote gets - rank 1 gets from a queue hosted on rank 0.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()

# Control queue for coordination (all ranks construct together)
control_queue = moodist.Queue(group, location=0)

if rank == 0:
    # Create a named work queue
    work_queue = moodist.Queue(group, location=0, name="work_queue")

    # Send the queue reference to rank 1
    control_queue.put_object(work_queue)

    # Now receive work through the work queue
    task = work_queue.get_object()

elif rank == 1:
    # Receive the queue reference
    work_queue = control_queue.get_object()

    # Use it to send work to rank 0
    work_queue.put_object({'task': 'process_data', 'id': 42})
```

### Example 6: Non-Blocking Polling

Check for data without blocking.

```python
import torch
import moodist
from queue import Empty

group = moodist.find_process_group("my_group")
rank = group.rank()

# Separate queues for different message types
command_queue = moodist.Queue(group, location=rank, name=f"commands_{rank}")
data_queue = moodist.Queue(group, location=rank, name=f"data_{rank}")

# Poll for data while doing other work
for iteration in range(1000):
    # Do computation...

    # Check for incoming commands (objects)
    try:
        cmd = command_queue.get_object(block=False)
        handle_command(cmd)
    except Empty:
        pass

    # Check for incoming data (tensors)
    try:
        tensor = data_queue.get_tensor(block=False)
        process_tensor(tensor)
    except Empty:
        pass
```

## Implementation Notes

### Memory Management

- **CPU tensors**: If the tensor is not from Moodist's CPU allocator, it is copied to registered memory before sending. For best performance with CPU tensors, use Moodist's CPU allocator.
- **CUDA tensors**: Transfers use RDMA when supported, enabling zero-copy GPU-to-GPU communication.

After `put_tensor()` returns, you must not modify the tensor's contents until after `QueueWork.wait()` completes (either explicitly or via the destructor). However, you can safely reassign the variable or let it go out of scope - the library keeps the underlying storage alive as needed.

### Ordering Guarantees

- Items from the same sender are delivered in FIFO order
- Items from different senders may be interleaved
- Transactions preserve ordering of all items within the transaction

### Thread Safety

- Multiple threads can safely put to the same queue
- Multiple threads can safely get from the same queue
- Individual puts and gets are atomic

## See Also

- [`moodist.compile_op`](compile_op.md) - Compile custom collective operations
- [`moodist.serialize`](serialize.md) - Object serialization
- [`moodist.MoodistProcessGroup`](process_group.md) - Process group management
