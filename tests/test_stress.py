"""
Stress tests for MoodistProcessGroup collectives.

These tests run many operations in tight loops to catch:
- State corruption from mixed operations
- Edge cases with varying tensor sizes
- Concurrency issues

Pattern: pre-allocate inputs/outputs, run ops in tight loop, verify after.
"""

import torch
from framework import TestContext, test, test_cpu_cuda, create_process_group


@test_cpu_cuda
def test_stress_allgather(ctx: TestContext, device: str):
    """Many allgathers in rapid succession."""
    pg = create_process_group(ctx)
    iterations = 100
    chunk_size = 64

    # Pre-allocate all inputs and outputs
    inputs = [
        torch.full((chunk_size,), float(ctx.rank + i), device=device, dtype=torch.float32)
        for i in range(iterations)
    ]
    outputs = [
        torch.zeros(chunk_size * ctx.world_size, device=device, dtype=torch.float32)
        for i in range(iterations)
    ]

    # Tight loop - just run the ops
    for i in range(iterations):
        pg._allgather_base(outputs[i], inputs[i]).wait()

    # Verify correctness after
    for i in range(iterations):
        for r in range(ctx.world_size):
            expected = float(r + i)
            chunk = outputs[i][r * chunk_size : (r + 1) * chunk_size]
            ctx.assert_true(
                torch.all(chunk == expected),
                f"allgather iteration {i}, rank {r} chunk mismatch"
            )


@test_cpu_cuda
def test_stress_reduce_scatter(ctx: TestContext, device: str):
    """Many reduce_scatters in rapid succession."""
    pg = create_process_group(ctx)
    iterations = 100
    chunk_size = 64

    # Pre-allocate all inputs and outputs
    # Each rank contributes (rank + 1) * (i + 1), sum = world_size*(world_size+1)/2 * (i+1)
    inputs = [
        torch.full(
            (chunk_size * ctx.world_size,),
            float((ctx.rank + 1) * (i + 1)),
            device=device,
            dtype=torch.float32
        )
        for i in range(iterations)
    ]
    outputs = [
        torch.zeros(chunk_size, device=device, dtype=torch.float32)
        for i in range(iterations)
    ]

    # Tight loop
    for i in range(iterations):
        pg._reduce_scatter_base(outputs[i], inputs[i]).wait()

    # Verify correctness after
    expected_base = ctx.world_size * (ctx.world_size + 1) / 2
    for i in range(iterations):
        expected = float(expected_base * (i + 1))
        ctx.assert_true(
            torch.allclose(outputs[i], torch.full_like(outputs[i], expected)),
            f"reduce_scatter iteration {i} mismatch: got {outputs[i][0]}, expected {expected}"
        )


@test_cpu_cuda
def test_stress_broadcast(ctx: TestContext, device: str):
    """Many broadcasts in rapid succession."""
    pg = create_process_group(ctx)
    iterations = 100
    tensor_size = 64

    # Pre-allocate tensors - root has data, others have zeros
    root_rank = 0
    tensors = []
    for i in range(iterations):
        if ctx.rank == root_rank:
            tensors.append(torch.full((tensor_size,), float(i), device=device, dtype=torch.float32))
        else:
            tensors.append(torch.zeros(tensor_size, device=device, dtype=torch.float32))

    # Tight loop
    for i in range(iterations):
        pg.broadcast(tensors[i], root_rank).wait()

    # Verify correctness after
    for i in range(iterations):
        expected = float(i)
        ctx.assert_true(
            torch.all(tensors[i] == expected),
            f"broadcast iteration {i} mismatch"
        )


@test_cpu_cuda
def test_stress_allreduce(ctx: TestContext, device: str):
    """Many allreduces in rapid succession."""
    pg = create_process_group(ctx)
    iterations = 100
    tensor_size = 64

    # Pre-allocate tensors - each rank has (rank + 1) * (i + 1)
    tensors = [
        torch.full(
            (tensor_size,),
            float((ctx.rank + 1) * (i + 1)),
            device=device,
            dtype=torch.float32
        )
        for i in range(iterations)
    ]

    # Tight loop
    for i in range(iterations):
        pg.allreduce([tensors[i]]).wait()

    # Verify correctness after - sum of (rank+1)*(i+1) for all ranks
    expected_base = ctx.world_size * (ctx.world_size + 1) / 2
    for i in range(iterations):
        expected = float(expected_base * (i + 1))
        ctx.assert_true(
            torch.allclose(tensors[i], torch.full_like(tensors[i], expected)),
            f"allreduce iteration {i} mismatch: got {tensors[i][0]}, expected {expected}"
        )


@test_cpu_cuda
def test_stress_mixed_collectives(ctx: TestContext, device: str):
    """Run different collectives interleaved."""
    pg = create_process_group(ctx)
    iterations = 50
    chunk_size = 32

    # Pre-allocate for allgather
    ag_inputs = [
        torch.full((chunk_size,), float(ctx.rank), device=device, dtype=torch.float32)
        for _ in range(iterations)
    ]
    ag_outputs = [
        torch.zeros(chunk_size * ctx.world_size, device=device, dtype=torch.float32)
        for _ in range(iterations)
    ]

    # Pre-allocate for reduce_scatter
    rs_inputs = [
        torch.full(
            (chunk_size * ctx.world_size,),
            float(ctx.rank + 1),
            device=device,
            dtype=torch.float32
        )
        for _ in range(iterations)
    ]
    rs_outputs = [
        torch.zeros(chunk_size, device=device, dtype=torch.float32)
        for _ in range(iterations)
    ]

    # Pre-allocate for broadcast
    root_rank = 0
    bc_tensors = []
    for i in range(iterations):
        if ctx.rank == root_rank:
            bc_tensors.append(torch.full((chunk_size,), float(i * 10), device=device, dtype=torch.float32))
        else:
            bc_tensors.append(torch.zeros(chunk_size, device=device, dtype=torch.float32))

    # Pre-allocate for allreduce
    ar_tensors = [
        torch.full((chunk_size,), float(ctx.rank + 1), device=device, dtype=torch.float32)
        for _ in range(iterations)
    ]

    # Tight loop - interleave all operations
    for i in range(iterations):
        pg._allgather_base(ag_outputs[i], ag_inputs[i]).wait()
        pg._reduce_scatter_base(rs_outputs[i], rs_inputs[i]).wait()
        pg.broadcast(bc_tensors[i], root_rank).wait()
        pg.allreduce([ar_tensors[i]]).wait()
        pg.barrier().wait()

    # Verify allgather
    for i in range(iterations):
        for r in range(ctx.world_size):
            chunk = ag_outputs[i][r * chunk_size : (r + 1) * chunk_size]
            ctx.assert_true(
                torch.all(chunk == float(r)),
                f"mixed: allgather iteration {i}, rank {r} mismatch"
            )

    # Verify reduce_scatter
    expected_rs = float(ctx.world_size * (ctx.world_size + 1) / 2)
    for i in range(iterations):
        ctx.assert_true(
            torch.allclose(rs_outputs[i], torch.full_like(rs_outputs[i], expected_rs)),
            f"mixed: reduce_scatter iteration {i} mismatch"
        )

    # Verify broadcast
    for i in range(iterations):
        expected_bc = float(i * 10)
        ctx.assert_true(
            torch.all(bc_tensors[i] == expected_bc),
            f"mixed: broadcast iteration {i} mismatch"
        )

    # Verify allreduce
    expected_ar = float(ctx.world_size * (ctx.world_size + 1) / 2)
    for i in range(iterations):
        ctx.assert_true(
            torch.allclose(ar_tensors[i], torch.full_like(ar_tensors[i], expected_ar)),
            f"mixed: allreduce iteration {i} mismatch"
        )


@test_cpu_cuda
def test_stress_varying_sizes(ctx: TestContext, device: str):
    """Test collectives with varying tensor sizes."""
    pg = create_process_group(ctx)

    # Various sizes including edge cases
    sizes = [1, 7, 64, 255, 1024, 4096, 65536]

    # Pre-allocate for each size
    ag_data = []
    for size in sizes:
        ag_data.append({
            'input': torch.full((size,), float(ctx.rank), device=device, dtype=torch.float32),
            'output': torch.zeros(size * ctx.world_size, device=device, dtype=torch.float32),
            'size': size
        })

    rs_data = []
    for size in sizes:
        rs_data.append({
            'input': torch.full((size * ctx.world_size,), float(ctx.rank + 1), device=device, dtype=torch.float32),
            'output': torch.zeros(size, device=device, dtype=torch.float32),
            'size': size
        })

    # Tight loop
    for i, size in enumerate(sizes):
        pg._allgather_base(ag_data[i]['output'], ag_data[i]['input']).wait()
        pg._reduce_scatter_base(rs_data[i]['output'], rs_data[i]['input']).wait()

    # Verify allgather
    for i, size in enumerate(sizes):
        for r in range(ctx.world_size):
            chunk = ag_data[i]['output'][r * size : (r + 1) * size]
            ctx.assert_true(
                torch.all(chunk == float(r)),
                f"varying sizes: allgather size {size}, rank {r} mismatch"
            )

    # Verify reduce_scatter
    expected_rs = float(ctx.world_size * (ctx.world_size + 1) / 2)
    for i, size in enumerate(sizes):
        ctx.assert_true(
            torch.allclose(rs_data[i]['output'], torch.full_like(rs_data[i]['output'], expected_rs)),
            f"varying sizes: reduce_scatter size {size} mismatch"
        )


@test_cpu_cuda
def test_stress_large_tensors(ctx: TestContext, device: str):
    """Test with larger tensors to stress memory handling."""
    pg = create_process_group(ctx)
    iterations = 10
    # 4MB per rank
    chunk_size = 1024 * 1024

    # Pre-allocate
    inputs = [
        torch.full((chunk_size,), float(ctx.rank), device=device, dtype=torch.float32)
        for _ in range(iterations)
    ]
    outputs = [
        torch.zeros(chunk_size * ctx.world_size, device=device, dtype=torch.float32)
        for _ in range(iterations)
    ]

    # Tight loop
    for i in range(iterations):
        pg._allgather_base(outputs[i], inputs[i]).wait()

    # Verify
    for i in range(iterations):
        for r in range(ctx.world_size):
            # Just check first and last elements to avoid slow full comparison
            chunk = outputs[i][r * chunk_size : (r + 1) * chunk_size]
            ctx.assert_true(
                chunk[0] == float(r) and chunk[-1] == float(r),
                f"large tensors: iteration {i}, rank {r} mismatch"
            )
