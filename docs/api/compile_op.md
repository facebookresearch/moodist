# moodist.compile_op

Compile a custom collective operation for distributed tensor communication.

## Function Signature

```python
def compile_op(
    group: MoodistProcessGroup,
    shape: tuple | list | None = None,
    dtype: torch.dtype | None = None,
    inputs: list[dict | DTensor] | None = None,
    outputs: list[dict | DTensor] | None = None,
    reduction: str | None = None
) -> CustomOp
```

> **Note:** The `reduction` parameter is not yet implemented but is documented here for completeness.

## Overview

`compile_op` is a powerful primitive that creates optimized collective operations for arbitrary data movement patterns between processes in a distributed group. It generalizes standard collective operations (like `all_gather`, `reduce_scatter`, `scatter`, `gather`) by allowing you to specify exactly which tensor slices each rank contributes (inputs) and receives (outputs).

**Key Features:**
- Define custom communication patterns beyond standard collectives
- Specify arbitrary tensor slice distributions across ranks
- Automatic optimization of data transfers
- Support for both contiguous and non-contiguous memory patterns
- Works with both CPU and CUDA tensors
- Reduction operations for overlapping inputs (sum, max, min, etc.)

**When to use `compile_op` vs standard collectives:**
- Use standard collectives (`all_gather`, `reduce_scatter`, etc.) when your pattern matches exactly
- Use `compile_op` when you need:
  - Custom slice distributions that don't match standard patterns
  - Multiple different slices per rank
  - Overlapping send/receive regions with reduction
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

### `shape`
**Type:** `tuple` or `list` of `int`, or `None`

The global tensor shape that defines the logical address space for the operation. All input and output slices are defined relative to this global shape. All ranks must specify the same shape.

Can be omitted if using DTensors for inputs/outputs, in which case the shape is derived automatically from the DTensor's global shape.

**Example:** `[8, 16]` defines a 2D global tensor of size 8×16

### `dtype`
**Type:** `torch.dtype` or `None`

The PyTorch data type for the operation (e.g., `torch.float32`, `torch.int64`). All ranks must specify the same dtype.

Can be omitted if using DTensors for inputs/outputs, in which case the dtype is derived automatically.

### `inputs`
**Type:** `list[dict | DTensor]` or `None` (default: `None`)

Optional list of input tensor specifications that this rank will contribute. Each element can be either:

**Dict format:**
- `'offset'`: `tuple` or `list` of `int` - Starting position in the global tensor
- `'shape'`: `tuple` or `list` of `int` - Size of this input slice

**DTensor format:**
- A `torch.distributed.tensor.DTensor` instance. The offset and shape are derived automatically from the DTensor's placements and device mesh.

The `offset` and `shape` must have the same number of dimensions as the global `shape`. If `None`, this rank contributes no inputs to the operation.

**Example (dict):**
```python
inputs = [
    {'offset': [0, 0], 'shape': [2, 4]},  # First slice at position [0,0]
    {'offset': [2, 0], 'shape': [2, 4]},  # Second slice at position [2,0]
]
```

**Example (DTensor):**
```python
inputs = [input_dtensor]  # Offset and shape derived from sharding
```

### `outputs`
**Type:** `list[dict | DTensor]` or `None` (default: `None`)

Optional list of output tensor specifications that this rank will receive. Format is identical to `inputs` (supports both dict and DTensor). If `None`, this rank receives no outputs from the operation.

### `reduction`
**Type:** `str` or `None` (default: `None`)

Specifies the reduction operation to apply when multiple input slices overlap in the global tensor space. When inputs from different ranks (or the same rank) specify overlapping regions, the reduction operator determines how values are combined.

**Supported reduction operations:**
- `'sum'` - Add overlapping values
- `'prod'` - Multiply overlapping values
- `'min'` - Take minimum of overlapping values
- `'max'` - Take maximum of overlapping values
- `'avg'` or `'mean'` - Average overlapping values
- `None` (default) - No reduction; overlapping inputs will use the last written value

**Behavior with overlapping inputs:**

When multiple ranks contribute data to the same region of the global tensor, the reduction operation is applied:

```python
# Example: Two ranks both write to offset [0, 0]
# Rank 0:
inputs = [{'offset': [0, 0], 'shape': [2, 4]}]  # Contains values [1, 2, 3, ...]

# Rank 1:
inputs = [{'offset': [0, 0], 'shape': [2, 4]}]  # Contains values [10, 20, 30, ...]

# With reduction='sum', output at [0, 0] would be [11, 22, 33, ...]
# With reduction='max', output at [0, 0] would be [10, 20, 30, ...]
```

This enables true reduce-scatter and all-reduce patterns where data is combined across ranks.

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

group = moodist.find_process_group("my_group")

if group.rank() == 0:
    # Rank 0 sends a 2×4 tensor
    inputs = [{'offset': [0, 0], 'shape': [2, 4]}]
    outputs = None
else:
    # Rank 1 receives a 2×4 tensor
    inputs = None
    outputs = [{'offset': [0, 0], 'shape': [2, 4]}]

# Compile the operation (collective call - all ranks must participate)
op = moodist.compile_op(
    group,
    shape=[2, 4],
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

group = moodist.find_process_group("my_group")
rank = group.rank()

if rank == 0:
    # Rank 0 sends from a single contiguous tensor
    inputs = [{'offset': [0, 0], 'shape': [6, 4]}]
    outputs = None
else:
    # Ranks 1, 2, 3 each receive their slice
    inputs = None
    outputs = [{'offset': [(rank-1)*2, 0], 'shape': [2, 4]}]

op = moodist.compile_op(
    group,
    shape=[6, 4],
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

group = moodist.find_process_group("my_group")
rank = group.rank()

if rank == 0:
    # Rank 0 receives all slices into a single contiguous tensor
    inputs = None
    outputs = [{'offset': [0, 0], 'shape': [6, 4]}]
else:
    # Ranks 1, 2, 3 each send their slice
    inputs = [{'offset': [(rank-1)*2, 0], 'shape': [2, 4]}]
    outputs = None

op = moodist.compile_op(
    group,
    shape=[6, 4],
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

group = moodist.find_process_group("my_group")
rank = group.rank()
size = group.size()

# Each rank contributes one slice and receives all slices
inputs = [{'offset': [rank*2, 0], 'shape': [2, 4]}]
outputs = [{'offset': [0, 0], 'shape': [size*2, 4]}]

op = moodist.compile_op(
    group,
    shape=[size*2, 4],
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

### Example 5: Reduce-Scatter Pattern

Each rank sends the full tensor and receives a different slice. With reduction, overlapping inputs are combined.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()
size = group.size()

# Each rank contributes the full tensor
inputs = [{'offset': [0, 0], 'shape': [size*2, 4]}]

# Each rank receives a different slice
outputs = [{'offset': [rank*2, 0], 'shape': [2, 4]}]

op = moodist.compile_op(
    group,
    shape=[size*2, 4],
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs,
    reduction='sum'  # Combine overlapping inputs
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

group = moodist.find_process_group("my_group")
rank = group.rank()
size = group.size()

# Send to next rank (rank+1 % size), receive from previous (rank-1 % size)
inputs = [{'offset': [(rank+1) % size * 2, 0], 'shape': [2, 4]}]
outputs = [{'offset': [rank * 2, 0], 'shape': [2, 4]}]

op = moodist.compile_op(
    group,
    shape=[size*2, 4],
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

### Example 7: All-Reduce Pattern

All ranks contribute overlapping data that gets combined and distributed to all ranks.

```python
import torch
import moodist

group = moodist.find_process_group("my_group")
rank = group.rank()

# All ranks contribute data to the same global location
inputs = [{'offset': [0, 0], 'shape': [4, 8]}]

# All ranks receive the reduced result
outputs = [{'offset': [0, 0], 'shape': [4, 8]}]

op = moodist.compile_op(
    group,
    shape=[4, 8],
    dtype=torch.float32,
    inputs=inputs,
    outputs=outputs,
    reduction='sum'  # Sum all inputs across ranks
)
```

**Communication Pattern (4 ranks):**
```
All ranks contribute overlapping data, which gets summed and returned to all:

Before:                          After (with reduction='sum'):
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
