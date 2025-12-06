"""
Tests for moodist.compile_op

compile_op creates custom collective operations for arbitrary input/output patterns.
Each rank specifies which slices of a global tensor it contributes (inputs) and receives (outputs).

These tests require distributed execution (multiple ranks).

NOTE: compile_op requires CUDA tensors (or CPU tensors allocated through moodist's allocator).
All tests use device="cuda".
"""

import torch
import moodist
from framework import TestContext, test, create_process_group

# All compile_op tests use CUDA - CPU requires moodist's allocator
DEVICE = "cuda"


@test
def test_compile_op_point_to_point(ctx: TestContext):
    """Test simple point-to-point: rank 0 sends to rank 1."""
    if ctx.world_size < 2:
        return  # Need at least 2 ranks

    pg = create_process_group(ctx)

    shape = [4]
    dtype = torch.float32

    if ctx.rank == 0:
        inputs = [{'offset': [0], 'shape': [4]}]
        outputs = None
    elif ctx.rank == 1:
        inputs = None
        outputs = [{'offset': [0], 'shape': [4]}]
    else:
        inputs = None
        outputs = None

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    # Execute the op
    if ctx.rank == 0:
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)
        future = op([input_tensor], [])
    elif ctx.rank == 1:
        output_tensor = torch.zeros(4, dtype=dtype, device=DEVICE)
        future = op([], [output_tensor])
    else:
        future = op([], [])

    future.wait()

    if ctx.rank == 1:
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"got {output_tensor}, expected {expected}"
        )


@test
def test_compile_op_broadcast(ctx: TestContext):
    """Test broadcast: rank 0 sends to all ranks."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    shape = [8]
    dtype = torch.float32

    # Rank 0 is the source, all ranks receive
    if ctx.rank == 0:
        inputs = [{'offset': [0], 'shape': [8]}]
    else:
        inputs = None

    # All ranks receive
    outputs = [{'offset': [0], 'shape': [8]}]

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    # Execute
    input_tensors = []
    if ctx.rank == 0:
        input_tensors = [torch.arange(8, dtype=dtype, device=DEVICE)]

    output_tensor = torch.zeros(8, dtype=dtype, device=DEVICE)
    future = op(input_tensors, [output_tensor])
    future.wait()

    expected = torch.arange(8, dtype=dtype, device=DEVICE)
    ctx.assert_true(
        torch.equal(output_tensor, expected),
        f"rank {ctx.rank}: got {output_tensor}, expected {expected}"
    )


@test
def test_compile_op_gather(ctx: TestContext):
    """Test gather: all ranks send to rank 0."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 4
    shape = [chunk_size * ctx.world_size]
    dtype = torch.float32

    # Each rank contributes its chunk
    inputs = [{'offset': [ctx.rank * chunk_size], 'shape': [chunk_size]}]

    # Only rank 0 receives
    if ctx.rank == 0:
        outputs = [{'offset': [0], 'shape': [chunk_size * ctx.world_size]}]
    else:
        outputs = None

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    # Each rank contributes its rank value
    input_tensor = torch.full((chunk_size,), float(ctx.rank), dtype=dtype, device=DEVICE)

    output_tensors = []
    if ctx.rank == 0:
        output_tensor = torch.zeros(chunk_size * ctx.world_size, dtype=dtype, device=DEVICE)
        output_tensors = [output_tensor]

    future = op([input_tensor], output_tensors)
    future.wait()

    if ctx.rank == 0:
        # Verify each chunk
        for r in range(ctx.world_size):
            chunk = output_tensor[r * chunk_size : (r + 1) * chunk_size]
            expected = torch.full((chunk_size,), float(r), dtype=dtype, device=DEVICE)
            ctx.assert_true(
                torch.equal(chunk, expected),
                f"chunk {r}: got {chunk}, expected {expected}"
            )


@test
def test_compile_op_scatter(ctx: TestContext):
    """Test scatter: rank 0 sends different chunks to each rank."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 4
    shape = [chunk_size * ctx.world_size]
    dtype = torch.float32

    # Only rank 0 provides input (full tensor)
    if ctx.rank == 0:
        inputs = [{'offset': [0], 'shape': [chunk_size * ctx.world_size]}]
    else:
        inputs = None

    # Each rank receives its chunk
    outputs = [{'offset': [ctx.rank * chunk_size], 'shape': [chunk_size]}]

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensors = []
    if ctx.rank == 0:
        # Create input where each chunk has rank-specific data
        input_tensor = torch.cat([
            torch.full((chunk_size,), float(r * 10), dtype=dtype, device=DEVICE)
            for r in range(ctx.world_size)
        ])
        input_tensors = [input_tensor]

    output_tensor = torch.zeros(chunk_size, dtype=dtype, device=DEVICE)
    future = op(input_tensors, [output_tensor])
    future.wait()

    expected = torch.full((chunk_size,), float(ctx.rank * 10), dtype=dtype, device=DEVICE)
    ctx.assert_true(
        torch.equal(output_tensor, expected),
        f"rank {ctx.rank}: got {output_tensor}, expected {expected}"
    )


@test
def test_compile_op_allgather(ctx: TestContext):
    """Test all-gather pattern: all ranks contribute and receive full tensor."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 4
    shape = [chunk_size * ctx.world_size]
    dtype = torch.float32

    # Each rank contributes its chunk
    inputs = [{'offset': [ctx.rank * chunk_size], 'shape': [chunk_size]}]
    # Each rank receives the full tensor
    outputs = [{'offset': [0], 'shape': [chunk_size * ctx.world_size]}]

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensor = torch.full((chunk_size,), float(ctx.rank), dtype=dtype, device=DEVICE)
    output_tensor = torch.zeros(chunk_size * ctx.world_size, dtype=dtype, device=DEVICE)

    future = op([input_tensor], [output_tensor])
    future.wait()

    # Verify all chunks
    for r in range(ctx.world_size):
        chunk = output_tensor[r * chunk_size : (r + 1) * chunk_size]
        expected = torch.full((chunk_size,), float(r), dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(chunk, expected),
            f"chunk {r}: got {chunk}, expected {expected}"
        )


@test
def test_compile_op_2d_tensor(ctx: TestContext):
    """Test with 2D tensor shape."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    # Global shape: [world_size, 4] - each rank contributes one row
    shape = [ctx.world_size, 4]
    dtype = torch.float32

    # Each rank contributes its row
    inputs = [{'offset': [ctx.rank, 0], 'shape': [1, 4]}]
    # All ranks receive full tensor
    outputs = [{'offset': [0, 0], 'shape': [ctx.world_size, 4]}]

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    # Input: row with rank-specific values
    input_tensor = torch.full((1, 4), float(ctx.rank * 10), dtype=dtype, device=DEVICE)
    output_tensor = torch.zeros(ctx.world_size, 4, dtype=dtype, device=DEVICE)

    future = op([input_tensor], [output_tensor])
    future.wait()

    # Verify each row
    for r in range(ctx.world_size):
        row = output_tensor[r]
        expected = torch.full((4,), float(r * 10), dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(row, expected),
            f"row {r}: got {row}, expected {expected}"
        )


@test
def test_compile_op_multiple_inputs(ctx: TestContext):
    """Test with multiple input tensors from same rank."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    shape = [4]
    dtype = torch.float32

    # Rank 0 provides two separate input tensors that together cover the full output
    if ctx.rank == 0:
        inputs = [
            {'offset': [0], 'shape': [2]},
            {'offset': [2], 'shape': [2]},
        ]
    else:
        inputs = None

    # Rank 1 receives the full tensor
    if ctx.rank == 1:
        outputs = [{'offset': [0], 'shape': [4]}]
    else:
        outputs = None

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensors = []
    if ctx.rank == 0:
        input_tensors = [
            torch.tensor([1.0, 2.0], dtype=dtype, device=DEVICE),
            torch.tensor([3.0, 4.0], dtype=dtype, device=DEVICE),
        ]

    output_tensors = []
    if ctx.rank == 1:
        output_tensor = torch.zeros(4, dtype=dtype, device=DEVICE)
        output_tensors = [output_tensor]

    future = op(input_tensors, output_tensors)
    future.wait()

    if ctx.rank == 1:
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"got {output_tensor}, expected {expected}"
        )


@test
def test_compile_op_different_dtypes(ctx: TestContext):
    """Test compile_op with different dtypes."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        shape = [4]

        if ctx.rank == 0:
            inputs = [{'offset': [0], 'shape': [4]}]
        else:
            inputs = None
        outputs = [{'offset': [0], 'shape': [4]}]

        op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

        input_tensors = []
        if ctx.rank == 0:
            input_tensors = [torch.tensor([1, 2, 3, 4], dtype=dtype, device=DEVICE)]

        output_tensor = torch.zeros(4, dtype=dtype, device=DEVICE)
        future = op(input_tensors, [output_tensor])
        future.wait()

        expected = torch.tensor([1, 2, 3, 4], dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"dtype {dtype}: got {output_tensor}, expected {expected}"
        )


@test
def test_compile_op_reuse(ctx: TestContext):
    """Test that compiled op can be reused multiple times."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    shape = [4]
    dtype = torch.float32

    if ctx.rank == 0:
        inputs = [{'offset': [0], 'shape': [4]}]
    else:
        inputs = None
    outputs = [{'offset': [0], 'shape': [4]}]

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    # Execute multiple times with different data
    for i in range(3):
        input_tensors = []
        if ctx.rank == 0:
            input_tensors = [torch.full((4,), float(i * 10), dtype=dtype, device=DEVICE)]

        output_tensor = torch.zeros(4, dtype=dtype, device=DEVICE)
        future = op(input_tensors, [output_tensor])
        future.wait()

        expected = torch.full((4,), float(i * 10), dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"iteration {i}: got {output_tensor}, expected {expected}"
        )


@test
def test_compile_op_no_inputs_no_outputs(ctx: TestContext):
    """Test that ranks can have neither inputs nor outputs."""
    if ctx.world_size < 3:
        return  # Need 3 ranks: sender, receiver, bystander

    pg = create_process_group(ctx)

    shape = [4]
    dtype = torch.float32

    if ctx.rank == 0:
        inputs = [{'offset': [0], 'shape': [4]}]
        outputs = None
    elif ctx.rank == 1:
        inputs = None
        outputs = [{'offset': [0], 'shape': [4]}]
    else:
        # Rank 2+ are bystanders
        inputs = None
        outputs = None

    op = moodist.compile_op(pg, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensors = []
    output_tensors = []

    if ctx.rank == 0:
        input_tensors = [torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)]
    elif ctx.rank == 1:
        output_tensor = torch.zeros(4, dtype=dtype, device=DEVICE)
        output_tensors = [output_tensor]

    future = op(input_tensors, output_tensors)
    future.wait()

    if ctx.rank == 1:
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=DEVICE)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"got {output_tensor}, expected {expected}"
        )
