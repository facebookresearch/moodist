#!/usr/bin/env python
"""
Test NVLink optimization planning in compile_op.

Run with torchrun (single node):
    torchrun --nproc_per_node=4 tests/test_nvlink_plan.py

Run with torchrun (multi-node):
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29500 tests/test_nvlink_plan.py
"""

import torch
import moodist
from framework import TestContext, test, create_process_group, create_context_from_env, TestRunner


@test
def test_nvlink_plan_allgather(ctx: TestContext):
    """All-gather pattern: each rank has 1 row, all ranks want all rows.

    This tests local sharing - all local ranks want data from all other ranks,
    so the NVLink planner should designate gateways to fetch via IB and share locally.
    """
    pg = create_process_group(ctx)

    shape = [ctx.world_size, 256]
    dtype = torch.float32

    inputs = [{'offset': [ctx.rank, 0], 'shape': [1, 256], 'device': 'cuda'}]
    outputs = [{'offset': [0, 0], 'shape': [ctx.world_size, 256], 'device': 'cuda'}]

    ctx.log(f"Compiling all-gather op (shape={shape})")
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Run it
    input_t = torch.full((1, 256), float(ctx.rank), device="cuda", dtype=dtype)
    output_t = torch.zeros((ctx.world_size, 256), device="cuda", dtype=dtype)

    work = op([input_t], [output_t])
    work.wait()
    torch.cuda.synchronize()

    # Verify
    expected = torch.arange(ctx.world_size, device="cuda", dtype=dtype).view(-1, 1).expand(ctx.world_size, 256)
    if torch.allclose(output_t, expected):
        ctx.log("PASSED")
    else:
        ctx.log("FAILED")
        for i in range(ctx.world_size):
            ctx.log(f"  row {i}: expected {expected[i, 0].item()}, got {output_t[i, 0].item()}")
        raise AssertionError("All-gather result mismatch")


@test
def test_nvlink_plan_broadcast(ctx: TestContext):
    """Broadcast pattern: rank 0 has data, all ranks want it.

    For single-node: all ranks copy from rank 0's input via NVLink (localInputCopy).
    For multi-node: local gateway fetches via IB, others copy via NVLink (localCopy).
    """
    pg = create_process_group(ctx)

    shape = [1, 1024]
    dtype = torch.float32

    # Only rank 0 provides input
    if ctx.rank == 0:
        inputs = [{'offset': [0, 0], 'shape': [1, 1024], 'device': 'cuda'}]
    else:
        inputs = None

    # All ranks want the full data
    outputs = [{'offset': [0, 0], 'shape': [1, 1024], 'device': 'cuda'}]

    ctx.log(f"Compiling broadcast op (shape={shape})")
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Run it
    input_tensors = []
    if ctx.rank == 0:
        input_tensors = [torch.arange(1024, device="cuda", dtype=dtype).view(1, 1024)]

    output_t = torch.zeros((1, 1024), device="cuda", dtype=dtype)
    work = op(input_tensors, [output_t])
    work.wait()
    torch.cuda.synchronize()

    # Verify
    expected = torch.arange(1024, device="cuda", dtype=dtype).view(1, -1)
    if torch.allclose(output_t, expected):
        ctx.log("PASSED")
    else:
        ctx.log(f"FAILED: expected {expected[0, :5]}..., got {output_t[0, :5]}...")
        raise AssertionError("Broadcast result mismatch")


@test
def test_nvlink_plan_partial_overlap(ctx: TestContext):
    """Partial overlap pattern: local ranks read overlapping regions from a remote rank.

    This tests the interval splitting logic - reads from local ranks overlap,
    so the planner should split into intervals and assign gateways for each.
    Also tests non-contiguous input handling (inputCopies).
    """
    if ctx.world_size < 2:
        ctx.log("Skipping (need >= 2 ranks)")
        return

    pg = create_process_group(ctx)

    # Global shape: [1, 4096]
    dtype = torch.float32

    # Only rank 0 provides input
    if ctx.rank == 0:
        inputs = [{'offset': [0, 0], 'shape': [1, 4096], 'device': 'cuda'}]
    else:
        inputs = None

    # Each local rank wants a different but overlapping slice
    # Local rank i wants bytes [i*256, i*256 + 1024]
    start = ctx.local_rank * 256
    end = start + 1024
    if end > 4096:
        end = 4096
        start = end - 1024

    outputs = [{'offset': [0, start], 'shape': [1, end - start], 'device': 'cuda'}]

    ctx.log(f"Compiling partial overlap op (slice [{start}, {end}))")
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Run it
    input_tensors = []
    if ctx.rank == 0:
        input_tensors = [torch.arange(4096, device="cuda", dtype=dtype).view(1, 4096)]

    output_t = torch.zeros((1, end - start), device="cuda", dtype=dtype)
    work = op(input_tensors, [output_t])
    work.wait()
    torch.cuda.synchronize()

    # Verify
    expected = torch.arange(start, end, device="cuda", dtype=dtype).view(1, -1)
    if torch.allclose(output_t, expected):
        ctx.log("PASSED")
    else:
        ctx.log(f"FAILED: expected {expected[0, :5]}..., got {output_t[0, :5]}...")
        raise AssertionError("Partial overlap result mismatch")


@test
def test_nvlink_plan_distributed_inputs(ctx: TestContext):
    """Distributed inputs: each rank provides part of the data, all ranks want overlapping slices.

    This tests:
    - Reads split across multiple source ranks
    - Interval splitting when reads span multiple sources
    - localInputCopy for local sources, reads for remote sources
    """
    if ctx.world_size < 2:
        ctx.log("Skipping (need >= 2 ranks)")
        return

    pg = create_process_group(ctx)

    # Global shape: [1, 2048] - split evenly across ranks
    chunk_size = 1024
    total_size = chunk_size * ctx.world_size
    dtype = torch.float32

    # Each rank provides its chunk
    my_start = ctx.rank * chunk_size
    inputs = [{'offset': [0, my_start], 'shape': [1, chunk_size], 'device': 'cuda'}]

    # Each rank wants a slice that spans multiple chunks
    # Rank i wants [i*512, i*512 + chunk_size] which overlaps with neighbors
    output_start = (ctx.rank * chunk_size // 2) % total_size
    output_end = output_start + chunk_size
    if output_end > total_size:
        output_end = total_size
        output_start = output_end - chunk_size

    outputs = [{'offset': [0, output_start], 'shape': [1, output_end - output_start], 'device': 'cuda'}]

    ctx.log(f"Compiling distributed inputs op: input=[{my_start}, {my_start + chunk_size}), output=[{output_start}, {output_end})")
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Run it - each rank provides its chunk filled with rank value
    input_t = torch.full((1, chunk_size), float(ctx.rank), device="cuda", dtype=dtype)
    output_t = torch.zeros((1, output_end - output_start), device="cuda", dtype=dtype)

    work = op([input_t], [output_t])
    work.wait()
    torch.cuda.synchronize()

    # Verify - output should contain data from appropriate source ranks
    # Each position in output comes from rank = position // chunk_size
    passed = True
    for i in range(output_end - output_start):
        global_pos = output_start + i
        expected_rank = global_pos // chunk_size
        expected_val = float(expected_rank)
        actual_val = output_t[0, i].item()
        if actual_val != expected_val:
            ctx.log(f"FAILED at pos {i} (global {global_pos}): expected {expected_val}, got {actual_val}")
            passed = False
            break

    if passed:
        ctx.log("PASSED")
    else:
        raise AssertionError("Distributed inputs result mismatch")


@test
def test_nvlink_plan_subset_reads(ctx: TestContext):
    """Subset reads: rank 0 provides full data, others read non-overlapping subsets.

    This tests non-zero sourceInputOffset - readers want different slices
    of a single contiguous input from rank 0.
    """
    if ctx.world_size < 2:
        ctx.log("Skipping (need >= 2 ranks)")
        return

    pg = create_process_group(ctx)

    # Global shape: large enough for each rank to get a unique slice
    slice_size = 512
    total_size = slice_size * ctx.world_size
    dtype = torch.float32

    # Only rank 0 provides the full input (contiguous, no inputCopy needed)
    if ctx.rank == 0:
        inputs = [{'offset': [0, 0], 'shape': [1, total_size], 'device': 'cuda'}]
    else:
        inputs = None

    # Each rank wants its own non-overlapping slice
    # Rank i wants [i*slice_size, (i+1)*slice_size)
    my_start = ctx.rank * slice_size
    my_end = my_start + slice_size

    outputs = [{'offset': [0, my_start], 'shape': [1, slice_size], 'device': 'cuda'}]

    ctx.log(f"Compiling subset reads op: output=[{my_start}, {my_end})")
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs)

    # Run it - rank 0 provides data where each slice_size chunk has value = chunk_index
    input_tensors = []
    if ctx.rank == 0:
        input_data = torch.zeros(total_size, device="cuda", dtype=dtype)
        for i in range(ctx.world_size):
            input_data[i * slice_size:(i + 1) * slice_size] = float(i)
        input_tensors = [input_data.view(1, total_size)]

    output_t = torch.zeros((1, slice_size), device="cuda", dtype=dtype)
    work = op(input_tensors, [output_t])
    work.wait()
    torch.cuda.synchronize()

    # Verify - my slice should contain my rank value
    expected = torch.full((1, slice_size), float(ctx.rank), device="cuda", dtype=dtype)
    if torch.allclose(output_t, expected):
        ctx.log("PASSED")
    else:
        ctx.log(f"FAILED: expected {expected[0, 0].item()}, got {output_t[0, 0].item()}")
        raise AssertionError("Subset reads result mismatch")


if __name__ == "__main__":
    import sys
    torch.cuda.set_device(int(__import__('os').environ.get("LOCAL_RANK", 0)))

    ctx = create_context_from_env()
    runner = TestRunner(ctx)

    if ctx.rank == 0:
        print(f"\nRunning NVLink plan tests with {ctx.world_size} ranks\n")

    runner.run_all()
    success = runner.summarize()

    sys.exit(0 if success else 1)
