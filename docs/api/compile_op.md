# moodist.compile_op

Compile a custom collective operation for distributed tensor communication.

## Function Signature

```python
def compile_op(
    group: MoodistProcessGroup,
    dtype: torch.dtype | None = None,
    inputs: list[TensorRegion | DTensor] | None = None,
    outputs: list[TensorRegion | DTensor] | None = None,
    reduce: str | None = None,
    cpu_sync: bool = False
) -> CustomOp
```

```python
@dataclass
class TensorRegion:
    offset: list[int]
    shape: list[int]
    tensor_id: str = "0"
```

## Overview

`compile_op` is a powerful primitive that creates optimized collective operations for arbitrary data movement patterns between processes in a distributed group. It generalizes standard collective operations (like `all_gather`, `reduce_scatter`, `scatter`, `gather`) by allowing you to specify exactly which tensor slices each rank contributes (inputs) and receives (outputs).

**Key Features:**
- Define custom communication patterns beyond standard collectives
- Specify arbitrary tensor slice distributions across ranks
- Automatic optimization of data transfers
- Support for both contiguous and non-contiguous memory patterns
- Works with both CPU and CUDA tensors
- Multi-tensor batching via `tensor_id` (batch tensors with different dimensionalities)
- Overlap handling with `reduce="any"` (additional reduction ops planned)

**When to use `compile_op` vs standard collectives:**
- Use standard collectives (`all_gather`, `reduce_scatter`, etc.) when your pattern matches exactly
- Use `compile_op` when you need:
  - Custom slice distributions that don't match standard patterns
  - Multiple different slices per rank
  - Overlapping inputs with `reduce="any"` (picks one source)
  - Batching multiple tensors with different dimensionalities (`tensor_id`)
  - Complex multi-rank communication patterns

**DTensor Support:**

`compile_op` can work directly with PyTorch DTensors (`torch.distributed.tensor.DTensor`). Instead of manually specifying offsets and shapes, you can pass DTensors and the sharding information is extracted automatically:

```python
# Redistribute a DTensor from one sharding to another
op = moodist.compile_op(
    group,
    inputs=[input_dtensor],    # Sharded one way
    outputs=[output_dtensor]   # Sharded differently
)

# Execute: takes local tensors, not DTensors
op([input_dtensor.to_local()], [output_dtensor.to_local()])
```

This is useful for tensor redistribution operations where you want to change the sharding of a DTensor. The DTensors can be meta tensors (no actual data) if you only need to specify the sharding pattern.

## Parameters

### `group`
**Type:** `MoodistProcessGroup`

The distributed process group that will participate in this collective operation. All ranks in this group must call `compile_op` collectively.

### `dtype`
**Type:** `torch.dtype` or `None`

The PyTorch data type for the operation (e.g., `torch.float32`, `torch.int64`). All ranks must specify the same dtype.

Can be omitted if using DTensors for inputs/outputs, in which case the dtype is derived automatically.

### `inputs`
**Type:** `list[TensorRegion | DTensor]` or `None` (default: `None`)

Optional list of input tensor specifications that this rank will contribute. Each element can be either:

**TensorRegion format:**
- `offset`: `list[int]` - Starting position in the global tensor
- `shape`: `list[int]` - Size of this input slice
- `tensor_id`: `str` (default: `"0"`) - Identifier for multi-tensor batching

**DTensor format:**
- A `torch.distributed.tensor.DTensor` instance. The offset and shape are derived automatically from the DTensor's placements and device mesh.

All regions with the same `tensor_id` must have the same number of dimensions. Different `tensor_id` values can have different dimensionalities, enabling batching of tensors like 2D weights and 1D biases in a single call.

If `None`, this rank contributes no inputs to the operation.

**Example (TensorRegion):**
```python
from moodist import TensorRegion

inputs = [
    TensorRegion(offset=[0, 0], shape=[2, 4]),  # First slice at position [0,0]
    TensorRegion(offset=[2, 0], shape=[2, 4]),  # Second slice at position [2,0]
]

# Multi-tensor batching with different dimensionalities:
inputs = [
    TensorRegion(offset=[0, 0], shape=[4, 4], tensor_id="weight"),  # 2D
    TensorRegion(offset=[0], shape=[4], tensor_id="bias"),          # 1D
]
```

**Example (DTensor):**
```python
inputs = [input_dtensor]  # Offset and shape derived from sharding
```

### `outputs`
**Type:** `list[TensorRegion | DTensor]` or `None` (default: `None`)

Optional list of output tensor specifications that this rank will receive. Format is identical to `inputs` (supports both TensorRegion and DTensor). If `None`, this rank receives no outputs from the operation.

### `reduce`
**Type:** `str` or `None` (default: `None`)

Specifies how to handle overlapping input regions. When multiple inputs (from the same or different ranks) cover the same output cell, this parameter determines the behavior.

**Currently supported:**
- `None` (default) - Error if any output cell is covered by multiple inputs
- `"any"` - Arbitrarily pick one input source for overlapping cells (no actual reduction)

**Planned (not yet implemented):**
- `"sum"` - Add overlapping values
- `"mean"` - Average overlapping values
- `"max"` - Take maximum of overlapping values
- `"min"` - Take minimum of overlapping values

**Behavior with overlapping inputs:**

When multiple ranks contribute data to the same region of the global tensor:

```python
from moodist import TensorRegion

# Example: Two ranks both write to offset [0, 0]
# Rank 0:
inputs = [TensorRegion(offset=[0, 0], shape=[2, 4])]  # Contains values [1, 2, 3, ...]

# Rank 1:
inputs = [TensorRegion(offset=[0, 0], shape=[2, 4])]  # Contains values [10, 20, 30, ...]

# With reduce=None (default): raises an error due to overlap
# With reduce="any": output contains values from either rank (arbitrary choice)
# With reduce="sum" (planned): output would be [11, 22, 33, ...]
```

The `reduce="any"` option is useful for replicated data patterns (e.g., `Replicate` placement in DTensor) where all sources have identical data and any one can be used.

### `cpu_sync`
**Type:** `bool` (default: `False`)

If `True`, forces CPU-side synchronization before CUDA operations in the compiled operation. This ensures the CPU thread waits for the data transfer to complete before scheduling any CUDA operations.

**When to use:**
- When running on a CUDA stream that may have device-wide synchronization pending (e.g., from the CUDA memory allocator)
- When you encounter deadlocks during `Future.wait()` with CUDA tensors

**Performance note:** Enabling `cpu_sync` may reduce concurrency between CPU and GPU work. Only enable it if you experience deadlock issues.

## Return Value

**Type:** `CustomOp`

A compiled custom operation object that can be called to execute the specified collective communication pattern. The compiled operation can be reused multiple times with different tensor data.

**Calling the compiled operation:**

```python
op(input_tensors, output_tensors)
```

- `input_tensors`: List of PyTorch tensors matching the input specifications from `compile_op`
- `output_tensors`: List of PyTorch tensors matching the output specifications from `compile_op`
- Returns a `Future` object

**Synchronization behavior:**

The returned `Future` ensures the operation completes before proceeding. You have two options:

```python
# Option 1: Implicit synchronization (blocks at end of statement)
op(inputs, outputs)  # Synchronizes immediately when Future is destroyed

# Option 2: Explicit synchronization (allows overlapping work)
future = op(inputs, outputs)
# ... do other work while transfer happens ...
future.wait()  # Explicitly wait for completion
```

For **CUDA tensors**, synchronization is non-blocking on the CPU - it inserts a wait into the CUDA stream, allowing the CPU to continue while the GPU waits for the transfer to complete. For **CPU tensors**, synchronization blocks the CPU until the transfer completes. You can mix CPU and CUDA tensors in the same operation, in which case both synchronization methods apply.

Different ranks can independently use any combination of CPU and CUDA tensors - for example, rank 0 could send from a CUDA tensor while rank 1 receives into a CPU tensor.

The operation executes **asynchronously** using RDMA for zero-copy data transfers directly between ranks.

## Examples

### Example 1: Point-to-Point Transfer

Simple transfer from rank 0 to rank 1.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")

if group.rank() == 0:
    # Rank 0 sends a 2×4 tensor
    inputs = [TensorRegion(offset=[0, 0], shape=[2, 4])]
    outputs = None
else:
    # Rank 1 receives a 2×4 tensor
    inputs = None
    outputs = [TensorRegion(offset=[0, 0], shape=[2, 4])]

# Compile the operation (collective call - all ranks must participate)
op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs
)

# Create tensors and execute the operation
if group.rank() == 0:
    input_tensor = torch.randn(2, 4)
    op([input_tensor], [])
else:
    output_tensor = torch.empty(2, 4)
    op([], [output_tensor])
    # output_tensor now contains the data from rank 0
```

**Communication Pattern:**
```
       sends [2×4 tensor]
Rank 0 ──────────────────→ Rank 1
```

### Example 2: Scatter Pattern

Rank 0 distributes different slices to multiple ranks.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")
rank = group.rank()

if rank == 0:
    # Rank 0 sends from a single contiguous tensor
    inputs = [TensorRegion(offset=[0, 0], shape=[6, 4])]
    outputs = None
else:
    # Ranks 1, 2, 3 each receive their slice
    inputs = None
    outputs = [TensorRegion(offset=[(rank-1)*2, 0], shape=[2, 4])]

op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs
)
```

**Communication Pattern:**
```
Rank 0 sends 3 slices:
  - slice_1 [offset: 0,0] → Rank 1
  - slice_2 [offset: 2,0] → Rank 2
  - slice_3 [offset: 4,0] → Rank 3

         Rank 0
         /  |  \
        /   |   \
       ↓    ↓    ↓
   Rank 1  Rank 2  Rank 3
   [s1]    [s2]    [s3]
```

### Example 3: Gather Pattern

Multiple ranks send slices to rank 0.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")
rank = group.rank()

if rank == 0:
    # Rank 0 receives all slices into a single contiguous tensor
    inputs = None
    outputs = [TensorRegion(offset=[0, 0], shape=[6, 4])]
else:
    # Ranks 1, 2, 3 each send their slice
    inputs = [TensorRegion(offset=[(rank-1)*2, 0], shape=[2, 4])]
    outputs = None

op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs
)
```

**Communication Pattern:**
```
Rank 0 receives 3 slices:
  - slice_1 [offset: 0,0] ← Rank 1
  - slice_2 [offset: 2,0] ← Rank 2
  - slice_3 [offset: 4,0] ← Rank 3

   Rank 1  Rank 2  Rank 3
   [s1]    [s2]    [s3]
     ↓      ↓      ↓
      \     |     /
       \    |    /
         Rank 0
   [s1][s2][s3]
```

### Example 4: All-Gather Pattern

Every rank receives slices from all ranks.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")
rank = group.rank()
size = group.size()

# Each rank contributes one slice and receives all slices
inputs = [TensorRegion(offset=[rank*2, 0], shape=[2, 4])]
outputs = [TensorRegion(offset=[0, 0], shape=[size*2, 4])]

op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs
)
```

**Communication Pattern (4 ranks):**
```
Each rank contributes 1 slice and receives all 4 slices:

Before:                  After:
Rank 0: [data_0]   →   Rank 0: [data_0][data_1][data_2][data_3]
Rank 1: [data_1]   →   Rank 1: [data_0][data_1][data_2][data_3]
Rank 2: [data_2]   →   Rank 2: [data_0][data_1][data_2][data_3]
Rank 3: [data_3]   →   Rank 3: [data_0][data_1][data_2][data_3]

All ranks exchange their slices with all other ranks.
```

### Example 5: Reduce-Scatter Pattern (Planned)

Each rank sends the full tensor and receives a different slice. With reduction, overlapping inputs are combined.

> **Note:** This example requires `reduce="sum"` which is not yet implemented. Once available, the pattern will work as shown.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")
rank = group.rank()
size = group.size()

# Each rank contributes the full tensor
inputs = [TensorRegion(offset=[0, 0], shape=[size*2, 4])]

# Each rank receives a different slice
outputs = [TensorRegion(offset=[rank*2, 0], shape=[2, 4])]

op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs,
    reduce='sum'  # Combine overlapping inputs (planned, not yet implemented)
)
```

**Communication Pattern (4 ranks):**
```
Each rank sends the full tensor, receives its designated slice:

All ranks send:           Each rank receives (after reduction):
Rank 0: [s0][s1][s2][s3]  →  Rank 0: [s0] (sum of all rank's s0 slices)
Rank 1: [s0][s1][s2][s3]  →  Rank 1: [s1] (sum of all rank's s1 slices)
Rank 2: [s0][s1][s2][s3]  →  Rank 2: [s2] (sum of all rank's s2 slices)
Rank 3: [s0][s1][s2][s3]  →  Rank 3: [s3] (sum of all rank's s3 slices)
```

### Example 6: Custom Pattern - Ring Communication

Each rank sends to the next rank in a ring topology.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")
rank = group.rank()
size = group.size()

# Send to next rank (rank+1 % size), receive from previous (rank-1 % size)
inputs = [TensorRegion(offset=[(rank+1) % size * 2, 0], shape=[2, 4])]
outputs = [TensorRegion(offset=[rank * 2, 0], shape=[2, 4])]

op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs
)
```

**Communication Pattern (4 ranks):**
```
Ring topology - each rank sends to next, receives from previous:

    Rank 0 ──→ Rank 1
      ↑           ↓
      │           │
    Rank 3 ←── Rank 2

(Rank 0→1, Rank 1→2, Rank 2→3, Rank 3→0)
```

### Example 7: All-Reduce Pattern (Planned)

All ranks contribute overlapping data that gets combined and distributed to all ranks.

> **Note:** This example requires `reduce="sum"` which is not yet implemented. Once available, the pattern will work as shown.

```python
import torch
import moodist
from moodist import TensorRegion

group = moodist.find_process_group("my_group")
rank = group.rank()

# All ranks contribute data to the same global location
inputs = [TensorRegion(offset=[0, 0], shape=[4, 8])]

# All ranks receive the reduced result
outputs = [TensorRegion(offset=[0, 0], shape=[4, 8])]

op = moodist.compile_op(
    group,
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs,
    reduce='sum'  # Sum all inputs across ranks (planned, not yet implemented)
)
```

**Communication Pattern (4 ranks):**
```
All ranks contribute overlapping data, which gets summed and returned to all:

Before:                          After (with reduce='sum'):
Rank 0: tensor_0 ──┐           Rank 0: sum(tensor_0 + tensor_1 +
Rank 1: tensor_1 ──┼─→ SUM  →  Rank 1:     tensor_2 + tensor_3)
Rank 2: tensor_2 ──┤           Rank 2:
Rank 3: tensor_3 ──┘           Rank 3:

This is equivalent to PyTorch's all_reduce collective operation.
```

### Example 8: DTensor Redistribution

Change the sharding of a DTensor from row-sharded to column-sharded.

```python
import torch
import moodist
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, distribute_tensor

group = moodist.find_process_group("my_group")

# Create a device mesh (4 ranks in a row)
mesh = DeviceMesh("cuda", torch.arange(4))

# Create input DTensor sharded along dim 0 (rows)
global_tensor = torch.randn(8, 16, device="cuda")
input_dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

# Create output DTensor sharded along dim 1 (columns)
output_dtensor = distribute_tensor(torch.empty(8, 16, device="cuda"), mesh, [Shard(1)])

# Compile redistribution operation - shape/dtype derived from DTensors
op = moodist.compile_op(
    group,
    inputs=[input_dtensor],
    outputs=[output_dtensor]
)

# Execute with local tensors
op([input_dtensor.to_local()], [output_dtensor.to_local()])
# output_dtensor now has the same data but sharded along columns
```

**Communication Pattern:**
```
Before (Shard(0) - row sharding):     After (Shard(1) - column sharding):
Rank 0: rows [0:2]                    Rank 0: cols [0:4]
Rank 1: rows [2:4]          →         Rank 1: cols [4:8]
Rank 2: rows [4:6]                    Rank 2: cols [8:12]
Rank 3: rows [6:8]                    Rank 3: cols [12:16]
```

Note that the compiled operation only transfers the data each rank actually needs - the full tensor is never materialized on any single rank. Each rank sends its relevant slices directly to the ranks that need them, making this efficient even for large tensors.

## Implementation Notes

### Collective Synchronization

`compile_op` is a **synchronous collective operation** that must be called by all ranks in the group. The function performs collective coordination to exchange and validate each rank's input/output specifications before returning.

**Important:** If any rank fails to call `compile_op`, the operation will hang indefinitely waiting for all ranks to participate.

### Performance Considerations

- **Compilation overhead**: There is overhead when calling `compile_op` as it involves collective coordination across all ranks. The compiled operation object returned can be reused multiple times to amortize this cost.
- **Asynchronous execution**: Calling the compiled operation returns immediately with a `Future` object. The actual data transfer happens asynchronously on a background thread.
- **RDMA-based transfers**: Data is transferred directly between ranks using RDMA (Remote Direct Memory Access), enabling zero-copy transfers without CPU involvement in the data movement.
- **Memory patterns**: The implementation optimizes for contiguous memory access when possible and handles non-contiguous patterns automatically (may involve intermediate copies).

## See Also

- [`moodist.Queue`](queue.md) - Inter-rank communication queue
- [`moodist.MoodistProcessGroup`](process_group.md) - Process group management
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) - Standard collective operations
