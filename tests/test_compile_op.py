"""
Tests for moodist.compile_op

compile_op creates custom collective operations for arbitrary input/output patterns.
Each rank specifies which slices of a global tensor it contributes (inputs) and receives (outputs).

These tests require distributed execution (multiple ranks).

NOTE: compile_op requires CUDA tensors from moodist's CUDA allocator (or CPU tensors allocated through
moodist's CPU allocator), and CUDA transfers between NVLink peers use sizes and offsets that are multiples of
16 bytes. Tests run on both CPU and CUDA via @test_cpu_cuda decorator.
"""

import torch
import moodist
from moodist import TensorRegion
from framework import TestContext, test_cpu_cuda, test_devices, create_process_group

# compile_op takes CPU tensors from moodist's CPU allocator and CUDA tensors from its CUDA allocator
moodist.enable_cpu_allocator()
moodist.enable_cuda_allocator()


@test_cpu_cuda
def test_compile_op_point_to_point(ctx: TestContext, device: str):
    """Test simple point-to-point: rank 0 sends to rank 1."""
    if ctx.world_size < 2:
        return  # Need at least 2 ranks

    pg = create_process_group(ctx)

    # Global shape: [4]
    dtype = torch.float32

    if ctx.rank == 0:
        inputs = [TensorRegion(offset=[0], shape=[4], device=device)]
        outputs = None
    elif ctx.rank == 1:
        inputs = None
        outputs = [TensorRegion(offset=[0], shape=[4], device=device)]
    else:
        inputs = None
        outputs = None

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Execute the op
    if ctx.rank == 0:
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)
        future = op([input_tensor], [])
    elif ctx.rank == 1:
        output_tensor = torch.zeros(4, dtype=dtype, device=device)
        future = op([], [output_tensor])
    else:
        future = op([], [])

    future.wait()

    if ctx.rank == 1:
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"got {output_tensor}, expected {expected}"
        )


@test_cpu_cuda
def test_compile_op_broadcast(ctx: TestContext, device: str):
    """Test broadcast: rank 0 sends to all ranks."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    # Global shape: [8]
    dtype = torch.float32

    # Rank 0 is the source, all ranks receive
    if ctx.rank == 0:
        inputs = [TensorRegion(offset=[0], shape=[8], device=device)]
    else:
        inputs = None

    # All ranks receive
    outputs = [TensorRegion(offset=[0], shape=[8], device=device)]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Execute
    input_tensors = []
    if ctx.rank == 0:
        input_tensors = [torch.arange(8, dtype=dtype, device=device)]

    output_tensor = torch.zeros(8, dtype=dtype, device=device)
    future = op(input_tensors, [output_tensor])
    future.wait()

    expected = torch.arange(8, dtype=dtype, device=device)
    ctx.assert_true(
        torch.equal(output_tensor, expected),
        f"rank {ctx.rank}: got {output_tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_compile_op_gather(ctx: TestContext, device: str):
    """Test gather: all ranks send to rank 0."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 4
    # Global shape: [chunk_size * world_size]
    dtype = torch.float32

    # Each rank contributes its chunk
    inputs = [TensorRegion(offset=[ctx.rank * chunk_size], shape=[chunk_size], device=device)]

    # Only rank 0 receives
    if ctx.rank == 0:
        outputs = [TensorRegion(offset=[0], shape=[chunk_size * ctx.world_size], device=device)]
    else:
        outputs = None

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Each rank contributes its rank value
    input_tensor = torch.full((chunk_size,), float(ctx.rank), dtype=dtype, device=device)

    output_tensors = []
    if ctx.rank == 0:
        output_tensor = torch.zeros(chunk_size * ctx.world_size, dtype=dtype, device=device)
        output_tensors = [output_tensor]

    future = op([input_tensor], output_tensors)
    future.wait()

    if ctx.rank == 0:
        # Verify each chunk
        for r in range(ctx.world_size):
            chunk = output_tensor[r * chunk_size : (r + 1) * chunk_size]
            expected = torch.full((chunk_size,), float(r), dtype=dtype, device=device)
            ctx.assert_true(
                torch.equal(chunk, expected),
                f"chunk {r}: got {chunk}, expected {expected}"
            )


@test_cpu_cuda
def test_compile_op_scatter(ctx: TestContext, device: str):
    """Test scatter: rank 0 sends different chunks to each rank."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 4
    # Global shape: [chunk_size * world_size]
    dtype = torch.float32

    # Only rank 0 provides input (full tensor)
    if ctx.rank == 0:
        inputs = [TensorRegion(offset=[0], shape=[chunk_size * ctx.world_size], device=device)]
    else:
        inputs = None

    # Each rank receives its chunk
    outputs = [TensorRegion(offset=[ctx.rank * chunk_size], shape=[chunk_size], device=device)]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensors = []
    if ctx.rank == 0:
        # Create input where each chunk has rank-specific data
        input_tensor = torch.cat([
            torch.full((chunk_size,), float(r * 10), dtype=dtype, device=device)
            for r in range(ctx.world_size)
        ])
        input_tensors = [input_tensor]

    output_tensor = torch.zeros(chunk_size, dtype=dtype, device=device)
    future = op(input_tensors, [output_tensor])
    future.wait()

    expected = torch.full((chunk_size,), float(ctx.rank * 10), dtype=dtype, device=device)
    ctx.assert_true(
        torch.equal(output_tensor, expected),
        f"rank {ctx.rank}: got {output_tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_compile_op_allgather(ctx: TestContext, device: str):
    """Test all-gather pattern: all ranks contribute and receive full tensor."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 4
    # Global shape: [chunk_size * world_size]
    dtype = torch.float32

    # Each rank contributes its chunk
    inputs = [TensorRegion(offset=[ctx.rank * chunk_size], shape=[chunk_size], device=device)]
    # Each rank receives the full tensor
    outputs = [TensorRegion(offset=[0], shape=[chunk_size * ctx.world_size], device=device)]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensor = torch.full((chunk_size,), float(ctx.rank), dtype=dtype, device=device)
    output_tensor = torch.zeros(chunk_size * ctx.world_size, dtype=dtype, device=device)

    future = op([input_tensor], [output_tensor])
    future.wait()

    # Verify all chunks
    for r in range(ctx.world_size):
        chunk = output_tensor[r * chunk_size : (r + 1) * chunk_size]
        expected = torch.full((chunk_size,), float(r), dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(chunk, expected),
            f"chunk {r}: got {chunk}, expected {expected}"
        )


@test_cpu_cuda
def test_compile_op_2d_tensor(ctx: TestContext, device: str):
    """Test with 2D tensor shape."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    # Global shape: [world_size, 4] - each rank contributes one row
    dtype = torch.float32

    # Each rank contributes its row
    inputs = [TensorRegion(offset=[ctx.rank, 0], shape=[1, 4], device=device)]
    # All ranks receive full tensor
    outputs = [TensorRegion(offset=[0, 0], shape=[ctx.world_size, 4], device=device)]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Input: row with rank-specific values
    input_tensor = torch.full((1, 4), float(ctx.rank * 10), dtype=dtype, device=device)
    output_tensor = torch.zeros(ctx.world_size, 4, dtype=dtype, device=device)

    future = op([input_tensor], [output_tensor])
    future.wait()

    # Verify each row
    for r in range(ctx.world_size):
        row = output_tensor[r]
        expected = torch.full((4,), float(r * 10), dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(row, expected),
            f"row {r}: got {row}, expected {expected}"
        )


@test_cpu_cuda
def test_compile_op_multiple_inputs(ctx: TestContext, device: str):
    """Test with multiple input tensors from same rank."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    # Global shape: [8] (pieces of 16 bytes: CUDA transfers between NVLink peers need multiples of 16 bytes)
    dtype = torch.float32

    # Rank 0 provides two separate input tensors that together cover the full output
    if ctx.rank == 0:
        inputs = [
            TensorRegion(offset=[0], shape=[4], device=device),
            TensorRegion(offset=[4], shape=[4], device=device),
        ]
    else:
        inputs = None

    # Rank 1 receives the full tensor
    if ctx.rank == 1:
        outputs = [TensorRegion(offset=[0], shape=[8], device=device)]
    else:
        outputs = None

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensors = []
    if ctx.rank == 0:
        input_tensors = [
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device),
            torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=dtype, device=device),
        ]

    output_tensors = []
    if ctx.rank == 1:
        output_tensor = torch.zeros(8, dtype=dtype, device=device)
        output_tensors = [output_tensor]

    future = op(input_tensors, output_tensors)
    future.wait()

    if ctx.rank == 1:
        expected = torch.arange(1, 9, dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"got {output_tensor}, expected {expected}"
        )


@test_cpu_cuda
def test_compile_op_different_dtypes(ctx: TestContext, device: str):
    """Test compile_op with different dtypes."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    # Global shape: [4]
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:

        if ctx.rank == 0:
            inputs = [TensorRegion(offset=[0], shape=[4], device=device)]
        else:
            inputs = None
        outputs = [TensorRegion(offset=[0], shape=[4], device=device)]

        op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

        input_tensors = []
        if ctx.rank == 0:
            input_tensors = [torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)]

        output_tensor = torch.zeros(4, dtype=dtype, device=device)
        future = op(input_tensors, [output_tensor])
        future.wait()

        expected = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"dtype {dtype}: got {output_tensor}, expected {expected}"
        )


@test_devices("cuda")
def test_compile_op_wait_other_stream(ctx: TestContext, device: str):
    """The op runs on the stream that is current when it is called; wait() joins the stream that is current then.

    Run an all-gather on a side stream and wait for it on the main stream, which then reads the output. On several
    nodes the data of other nodes arrives over RDMA and is then copied on by the kernel, so the main stream must
    wait for both.
    """
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    chunk_size = 1 << 20  # 4 MiB per rank
    dtype = torch.float32
    inputs = [TensorRegion(offset=[ctx.rank * chunk_size], shape=[chunk_size], device=device)]
    outputs = [TensorRegion(offset=[0], shape=[chunk_size * ctx.world_size], device=device)]
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensor = torch.empty(chunk_size, dtype=dtype, device=device)
    output_tensor = torch.zeros(chunk_size * ctx.world_size, dtype=dtype, device=device)
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    for i in range(20):
        # The previous iteration's read of the output (on main) must be done before the op overwrites it
        side.wait_stream(main)
        with torch.cuda.stream(side):
            input_tensor.fill_(float(i * 100 + ctx.rank))
            future = op([input_tensor], [output_tensor])
        future.wait()
        got = output_tensor.view(ctx.world_size, chunk_size).clone()
        expected = (torch.arange(ctx.world_size, dtype=dtype, device=device) + i * 100).unsqueeze(1)
        ctx.assert_true(
            torch.equal(got, expected.expand_as(got)),
            f"iteration {i}: {(got != expected).sum().item()} wrong elements"
        )


@test_cpu_cuda
def test_compile_op_reuse(ctx: TestContext, device: str):
    """Test that compiled op can be reused multiple times."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    # Global shape: [4]
    dtype = torch.float32

    if ctx.rank == 0:
        inputs = [TensorRegion(offset=[0], shape=[4], device=device)]
    else:
        inputs = None
    outputs = [TensorRegion(offset=[0], shape=[4], device=device)]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Execute multiple times with different data
    for i in range(3):
        input_tensors = []
        if ctx.rank == 0:
            input_tensors = [torch.full((4,), float(i * 10), dtype=dtype, device=device)]

        output_tensor = torch.zeros(4, dtype=dtype, device=device)
        future = op(input_tensors, [output_tensor])
        future.wait()

        expected = torch.full((4,), float(i * 10), dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"iteration {i}: got {output_tensor}, expected {expected}"
        )


@test_cpu_cuda
def test_compile_op_no_inputs_no_outputs(ctx: TestContext, device: str):
    """Test that ranks can have neither inputs nor outputs."""
    if ctx.world_size < 3:
        return  # Need 3 ranks: sender, receiver, bystander

    pg = create_process_group(ctx)

    # Global shape: [4]
    dtype = torch.float32

    if ctx.rank == 0:
        inputs = [TensorRegion(offset=[0], shape=[4], device=device)]
        outputs = None
    elif ctx.rank == 1:
        inputs = None
        outputs = [TensorRegion(offset=[0], shape=[4], device=device)]
    else:
        # Rank 2+ are bystanders
        inputs = None
        outputs = None

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    input_tensors = []
    output_tensors = []

    if ctx.rank == 0:
        input_tensors = [torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)]
    elif ctx.rank == 1:
        output_tensor = torch.zeros(4, dtype=dtype, device=device)
        output_tensors = [output_tensor]

    future = op(input_tensors, output_tensors)
    future.wait()

    if ctx.rank == 1:
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=device)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"got {output_tensor}, expected {expected}"
        )


@test_cpu_cuda
def test_compile_op_tensor_id_multi_tensor(ctx: TestContext, device: str):
    """Test tensor_id feature: batch multiple tensors with different ndims in one compile_op.

    This test transfers multiple tensors in a single compiled operation:
    - weight: 2D [world_size, 4], sharded on dim 0 (each rank provides 1 row)
    - weight2: 2D [world_size, 8], sharded on dim 0 (same ndim as weight, different tensor_id)
    - bias: 1D [4], replicated from rank 0
    - scale: 1D [8], replicated from rank 1 (same ndim as bias, different tensor_id)
    (All rows and vectors are multiples of 16 bytes, as CUDA transfers between NVLink peers require.)
    """
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    dtype = torch.float32

    # Tensor dimensions
    weight_rows = ctx.world_size  # One row per rank
    weight_cols = 4
    weight2_cols = 8
    bias_size = 4
    scale_size = 8

    # Inputs: each rank provides its row of weight and weight2
    # Rank 0 provides bias, rank 1 provides scale
    inputs = [
        TensorRegion(offset=[ctx.rank, 0], shape=[1, weight_cols], device=device, tensor_id="weight"),
        TensorRegion(offset=[ctx.rank, 0], shape=[1, weight2_cols], device=device, tensor_id="weight2"),
    ]
    if ctx.rank == 0:
        inputs.append(TensorRegion(offset=[0], shape=[bias_size], device=device, tensor_id="bias"))
    if ctx.rank == 1:
        inputs.append(TensorRegion(offset=[0], shape=[scale_size], device=device, tensor_id="scale"))

    # Outputs: all ranks receive all tensors
    outputs = [
        TensorRegion(offset=[0, 0], shape=[weight_rows, weight_cols], device=device, tensor_id="weight"),
        TensorRegion(offset=[0, 0], shape=[weight_rows, weight2_cols], device=device, tensor_id="weight2"),
        TensorRegion(offset=[0], shape=[bias_size], device=device, tensor_id="bias"),
        TensorRegion(offset=[0], shape=[scale_size], device=device, tensor_id="scale"),
    ]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs, reduce="any")

    # Create input tensors
    # Weight: each rank has 1 row, filled with rank value
    weight_input = torch.full((1, weight_cols), float(ctx.rank), dtype=dtype, device=device)
    # Weight2: each rank has 1 row, filled with rank + 100
    weight2_input = torch.full((1, weight2_cols), float(ctx.rank + 100), dtype=dtype, device=device)
    input_tensors = [weight_input, weight2_input]

    if ctx.rank == 0:
        # Bias: filled with -1.0
        bias_input = torch.full((bias_size,), -1.0, dtype=dtype, device=device)
        input_tensors.append(bias_input)
    if ctx.rank == 1:
        # Scale: filled with -2.0
        scale_input = torch.full((scale_size,), -2.0, dtype=dtype, device=device)
        input_tensors.append(scale_input)

    # Output tensors
    weight_output = torch.zeros(weight_rows, weight_cols, dtype=dtype, device=device)
    weight2_output = torch.zeros(weight_rows, weight2_cols, dtype=dtype, device=device)
    bias_output = torch.zeros(bias_size, dtype=dtype, device=device)
    scale_output = torch.zeros(scale_size, dtype=dtype, device=device)

    future = op(input_tensors, [weight_output, weight2_output, bias_output, scale_output])
    future.wait()

    # Verify weight: each row should contain the rank that provided it
    for r in range(ctx.world_size):
        expected_row = torch.full((1, weight_cols), float(r), dtype=dtype, device=device)
        actual_row = weight_output[r:r+1]
        ctx.assert_true(
            torch.equal(actual_row, expected_row),
            f"weight row {r}: got {actual_row}, expected {expected_row}"
        )

    # Verify weight2: each row should contain rank + 100
    for r in range(ctx.world_size):
        expected_row = torch.full((1, weight2_cols), float(r + 100), dtype=dtype, device=device)
        actual_row = weight2_output[r:r+1]
        ctx.assert_true(
            torch.equal(actual_row, expected_row),
            f"weight2 row {r}: got {actual_row}, expected {expected_row}"
        )

    # Verify bias: should be -1.0 (from rank 0)
    expected_bias = torch.full((bias_size,), -1.0, dtype=dtype, device=device)
    ctx.assert_true(
        torch.equal(bias_output, expected_bias),
        f"bias: got {bias_output}, expected {expected_bias}"
    )

    # Verify scale: should be -2.0 (from rank 1)
    expected_scale = torch.full((scale_size,), -2.0, dtype=dtype, device=device)
    ctx.assert_true(
        torch.equal(scale_output, expected_scale),
        f"scale: got {scale_output}, expected {expected_scale}"
    )


@test_cpu_cuda
def test_compile_op_device_mismatch_error(ctx: TestContext, device: str):
    """Test that passing a tensor with wrong device raises an error."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    dtype = torch.float32

    # Compile with the test device
    if ctx.rank == 0:
        inputs = [TensorRegion(offset=[0], shape=[4], device=device)]
        outputs = None
    else:
        inputs = None
        outputs = [TensorRegion(offset=[0], shape=[4], device=device)]

    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Try to execute with wrong device tensor
    wrong_device = "cpu" if device == "cuda" else "cuda"

    # Skip if wrong_device is cuda but CUDA not available
    if wrong_device == "cuda" and not torch.cuda.is_available():
        return

    try:
        if ctx.rank == 0:
            # Create tensor on wrong device
            input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype, device=wrong_device)
            future = op([input_tensor], [])
        else:
            output_tensor = torch.zeros(4, dtype=dtype, device=wrong_device)
            future = op([], [output_tensor])

        future.wait()
        ctx.log("FAIL: Expected error for device mismatch, but op succeeded")
        ctx.assert_true(False, "Expected RuntimeError for device mismatch")
    except RuntimeError as e:
        error_msg = str(e)
        ctx.assert_true(
            "wrong device" in error_msg,
            f"Expected 'wrong device' in error message, got: {error_msg}"
        )


@test_cpu_cuda
def test_compile_op_errors_on_all_ranks(ctx: TestContext, device: str):
    """An error that one rank finds (in its own outputs or arguments) is raised on every rank, and the group's
    compile_op still works afterwards (its queues stayed in step): before, the other ranks waited forever."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)
    n = ctx.world_size
    last = n - 1
    dtype = torch.float32
    k = 4  # 16 bytes per rank

    def expect_error(cls, text, inputs, outputs):
        try:
            moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)
        except cls as e:
            ctx.assert_true(text in str(e), f"rank {ctx.rank}: expected '{text}' in the error, got: {e}")
            return
        ctx.assert_true(False, f"rank {ctx.rank}: compile_op did not raise {cls.__name__} ('{text}')")

    def region(offset, shape, device=device):
        return TensorRegion(offset=[offset], shape=[shape], device=device)

    inputs = [region(ctx.rank * k, k)]
    everything = [region(0, n * k)]

    # A gap in the last rank's output only (it wants k elements more than the inputs cover)
    outputs = [region(0, (n + 1) * k)] if ctx.rank == last else everything
    expect_error(RuntimeError, f"missing input coverage in the outputs of rank {last}", inputs, outputs)

    # An overlap in rank 0's output only: the last rank has a second input over rank 0's
    overlapping = inputs + [region(0, k)] if ctx.rank == last else inputs
    outputs = [region(0, k)] if ctx.rank == 0 else [region(k, (n - 1) * k)]
    expect_error(RuntimeError, "overlapping inputs detected in the outputs of rank 0", overlapping, outputs)

    # A CUDA device index that is not rank 1's own (only rank 1 can check it)
    if device == "cuda":
        wrong = f"cuda:{torch.cuda.current_device() + 1}"
        mine = [region(ctx.rank * k, k, wrong if ctx.rank == 1 else device)]
        expect_error(RuntimeError, "(found on rank 1)", mine, everything)

    # A malformed spec on rank 1 (compile.py checks it before the spec exchange)
    bad = [TensorRegion(offset=[0.5], shape=[k], device=device)] if ctx.rank == 1 else inputs
    expect_error(TypeError, "offset[0] must be an int", bad, everything)

    # The group still compiles and runs ops
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=everything)
    x = torch.arange(k, dtype=dtype, device=device) + 100 * ctx.rank
    out = torch.zeros(n * k, dtype=dtype, device=device)
    op([x], [out]).wait()
    expected = torch.cat([torch.arange(k, dtype=dtype, device=device) + 100 * r for r in range(n)])
    ctx.assert_true(torch.equal(out, expected), f"rank {ctx.rank}: got {out.tolist()}")


@test_devices("cuda")
def test_compile_op_mixed_devices(ctx: TestContext, device: str):
    """One op with a CUDA and a CPU tensor on every rank, both all-gathered: the kernel copies the CUDA one between
    NVLink peers (and is tuned with scratch tensors for the CUDA tensors only), RDMA reads move the CPU one."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)
    n = ctx.world_size
    dtype = torch.float32
    g = 16
    c = 8
    inputs = [
        TensorRegion(offset=[ctx.rank * g], shape=[g], device="cuda", tensor_id="g"),
        TensorRegion(offset=[ctx.rank * c], shape=[c], device="cpu", tensor_id="c"),
    ]
    outputs = [
        TensorRegion(offset=[0], shape=[n * g], device="cuda", tensor_id="g"),
        TensorRegion(offset=[0], shape=[n * c], device="cpu", tensor_id="c"),
    ]
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    g_input = torch.arange(g, dtype=dtype, device="cuda") + 100 * ctx.rank
    c_input = torch.arange(c, dtype=dtype) + 1000 * ctx.rank
    g_output = torch.zeros(n * g, dtype=dtype, device="cuda")
    c_output = torch.zeros(n * c, dtype=dtype)
    op([g_input, c_input], [g_output, c_output]).wait()

    g_expected = torch.cat([torch.arange(g, dtype=dtype, device="cuda") + 100 * r for r in range(n)])
    c_expected = torch.cat([torch.arange(c, dtype=dtype) + 1000 * r for r in range(n)])
    ctx.assert_true(torch.equal(g_output, g_expected), f"CUDA tensor: got {g_output.tolist()}")
    ctx.assert_true(torch.equal(c_output, c_expected), f"CPU tensor: got {c_output.tolist()}")
