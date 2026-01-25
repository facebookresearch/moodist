"""
Tests for MoodistProcessGroup

These tests verify the distributed collectives work correctly.
ProcessGroup is created directly (not via torch.distributed.init_process_group)
to avoid side effects and allow per-test isolation.
"""

import torch
from framework import TestContext, test, test_cpu_cuda, test_cpu_cuda_kernel_modes, create_process_group


@test
def test_pg_creation(ctx: TestContext):
    """Test that we can create a ProcessGroup."""
    pg = create_process_group(ctx)
    ctx.assert_equal(pg.rank(), ctx.rank)
    ctx.assert_equal(pg.size(), ctx.world_size)


@test_cpu_cuda_kernel_modes
def test_pg_allgather_base(ctx: TestContext, device: str, kernel_less: bool):
    """Test _allgather_base: each rank contributes a chunk, gather all chunks."""
    with create_process_group(ctx, force_kernel_less=kernel_less) as pg:
        # Each rank has a small tensor with its rank value
        chunk_size = 4
        input_tensor = torch.full(
            (chunk_size,), float(ctx.rank), device=device, dtype=torch.float32
        )

        # Output tensor to hold all chunks
        output_tensor = torch.zeros(
            (chunk_size * ctx.world_size,), device=device, dtype=torch.float32
        )

        # Run allgather
        work = pg._allgather_base(output_tensor, input_tensor)
        work.wait()

        # Verify: output should be [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
        expected = torch.cat([
            torch.full((chunk_size,), float(r), device=device, dtype=torch.float32)
            for r in range(ctx.world_size)
        ])
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"allgather result mismatch: got {output_tensor}, expected {expected}"
        )


@test_cpu_cuda
def test_pg_reduce_scatter_base(ctx: TestContext, device: str):
    """Test _reduce_scatter_base: sum across ranks, each gets a chunk."""
    pg = create_process_group(ctx)

    chunk_size = 4
    # Each rank has full-size input with rank-specific values
    # After reduce-scatter with sum, each rank gets sum of corresponding chunks
    input_tensor = torch.full(
        (chunk_size * ctx.world_size,), float(ctx.rank + 1), device=device, dtype=torch.float32
    )

    output_tensor = torch.zeros(
        (chunk_size,), device=device, dtype=torch.float32
    )

    # Run reduce-scatter
    work = pg._reduce_scatter_base(output_tensor, input_tensor)
    work.wait()

    # Each rank gets its chunk, which is the sum of all ranks' contributions
    # All ranks contribute (rank+1), so sum = 1+2+3+...+world_size = world_size*(world_size+1)/2
    expected_value = ctx.world_size * (ctx.world_size + 1) / 2
    expected = torch.full((chunk_size,), expected_value, device=device, dtype=torch.float32)

    ctx.assert_true(
        torch.allclose(output_tensor, expected),
        f"reduce_scatter result mismatch: got {output_tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_pg_allgather_varying_data(ctx: TestContext, device: str):
    """Test allgather with different data per rank."""
    pg = create_process_group(ctx)

    chunk_size = 8
    # Each rank contributes unique data: [rank*100, rank*100+1, ...]
    input_tensor = torch.arange(
        ctx.rank * 100, ctx.rank * 100 + chunk_size, device=device, dtype=torch.float32
    )

    output_tensor = torch.zeros(
        (chunk_size * ctx.world_size,), device=device, dtype=torch.float32
    )

    work = pg._allgather_base(output_tensor, input_tensor)
    work.wait()

    # Verify each chunk
    for r in range(ctx.world_size):
        expected_chunk = torch.arange(
            r * 100, r * 100 + chunk_size, device=device, dtype=torch.float32
        )
        actual_chunk = output_tensor[r * chunk_size : (r + 1) * chunk_size]
        ctx.assert_true(
            torch.equal(actual_chunk, expected_chunk),
            f"chunk {r} mismatch: got {actual_chunk}, expected {expected_chunk}"
        )


@test_cpu_cuda
def test_pg_large_allgather(ctx: TestContext, device: str):
    """Test allgather with larger tensors."""
    pg = create_process_group(ctx)

    # 1MB per rank
    chunk_size = 256 * 1024

    # Use a deterministic pattern based on rank
    torch.manual_seed(ctx.rank + 42)
    input_tensor = torch.randn(chunk_size, device=device, dtype=torch.float32)

    output_tensor = torch.zeros(
        chunk_size * ctx.world_size, device=device, dtype=torch.float32
    )

    work = pg._allgather_base(output_tensor, input_tensor)
    work.wait()

    # Verify our own chunk is in the right place
    our_chunk = output_tensor[ctx.rank * chunk_size : (ctx.rank + 1) * chunk_size]
    ctx.assert_true(
        torch.equal(our_chunk, input_tensor),
        "our chunk doesn't match input"
    )

    # Verify other ranks' chunks with same seeds
    for r in range(ctx.world_size):
        torch.manual_seed(r + 42)
        expected = torch.randn(chunk_size, device=device, dtype=torch.float32)
        actual = output_tensor[r * chunk_size : (r + 1) * chunk_size]
        ctx.assert_true(
            torch.equal(actual, expected),
            f"rank {r} chunk mismatch"
        )


@test_cpu_cuda
def test_pg_broadcast(ctx: TestContext, device: str):
    """Test broadcast: root rank sends data to all other ranks."""
    pg = create_process_group(ctx)

    root_rank = 0
    tensor_size = 16

    if ctx.rank == root_rank:
        # Root has the data to broadcast
        tensor = torch.arange(tensor_size, device=device, dtype=torch.float32)
    else:
        # Other ranks have zeros, will receive the broadcast
        tensor = torch.zeros(tensor_size, device=device, dtype=torch.float32)

    work = pg.broadcast(tensor, root_rank)
    work.wait()

    # All ranks should have the same data as root
    expected = torch.arange(tensor_size, device=device, dtype=torch.float32)
    ctx.assert_true(
        torch.equal(tensor, expected),
        f"broadcast result mismatch: got {tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_pg_broadcast_nonzero_root(ctx: TestContext, device: str):
    """Test broadcast with a non-zero root rank."""
    if ctx.world_size < 2:
        return

    pg = create_process_group(ctx)

    root_rank = ctx.world_size - 1  # Last rank is root
    tensor_size = 8

    if ctx.rank == root_rank:
        tensor = torch.full((tensor_size,), 42.0, device=device, dtype=torch.float32)
    else:
        tensor = torch.zeros(tensor_size, device=device, dtype=torch.float32)

    work = pg.broadcast(tensor, root_rank)
    work.wait()

    expected = torch.full((tensor_size,), 42.0, device=device, dtype=torch.float32)
    ctx.assert_true(
        torch.equal(tensor, expected),
        f"broadcast from rank {root_rank} failed: got {tensor}"
    )


@test_cpu_cuda
def test_pg_allreduce_sum(ctx: TestContext, device: str):
    """Test allreduce with sum: each rank contributes, result is sum on all ranks."""
    pg = create_process_group(ctx)

    tensor_size = 8
    # Each rank has tensor filled with (rank + 1)
    tensor = torch.full((tensor_size,), float(ctx.rank + 1), device=device, dtype=torch.float32)

    work = pg.allreduce([tensor])
    work.wait()

    # Sum of all ranks' contributions: 1 + 2 + ... + world_size
    expected_value = ctx.world_size * (ctx.world_size + 1) / 2
    expected = torch.full((tensor_size,), expected_value, device=device, dtype=torch.float32)

    ctx.assert_true(
        torch.allclose(tensor, expected),
        f"allreduce sum mismatch: got {tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_pg_allreduce_varying_data(ctx: TestContext, device: str):
    """Test allreduce with element-wise varying data."""
    pg = create_process_group(ctx)

    tensor_size = 4
    # Each rank has [rank, rank+1, rank+2, rank+3]
    tensor = torch.arange(ctx.rank, ctx.rank + tensor_size, device=device, dtype=torch.float32)

    work = pg.allreduce([tensor])
    work.wait()

    # Element i should be sum of (rank + i) for all ranks
    # = sum(rank for all ranks) + i * world_size
    # = (0 + 1 + ... + world_size-1) + i * world_size
    # = world_size*(world_size-1)/2 + i * world_size
    rank_sum = ctx.world_size * (ctx.world_size - 1) / 2
    expected = torch.tensor(
        [rank_sum + i * ctx.world_size for i in range(tensor_size)],
        device=device, dtype=torch.float32
    )

    ctx.assert_true(
        torch.allclose(tensor, expected),
        f"allreduce varying data mismatch: got {tensor}, expected {expected}"
    )


@test
def test_pg_barrier(ctx: TestContext):
    """Test barrier: all ranks synchronize."""
    pg = create_process_group(ctx)

    # Simple barrier - if it returns, it worked
    work = pg.barrier()
    work.wait()

    # Do it a few times to ensure it's reliable
    for _ in range(3):
        work = pg.barrier()
        work.wait()


@test_cpu_cuda
def test_pg_scatter(ctx: TestContext, device: str):
    """Test scatter: root rank distributes different chunks to each rank."""
    pg = create_process_group(ctx)

    chunk_size = 4
    root_rank = 0

    # Root rank prepares scatter_list with different data for each rank
    if ctx.rank == root_rank:
        scatter_list = [
            torch.full((chunk_size,), float(r * 10), device=device, dtype=torch.float32)
            for r in range(ctx.world_size)
        ]
    else:
        scatter_list = None

    # Output tensor for this rank
    output_tensor = torch.zeros((chunk_size,), device=device, dtype=torch.float32)

    # Run scatter
    from torch.distributed import ScatterOptions
    opts = ScatterOptions()
    opts.rootRank = root_rank
    work = pg.scatter([output_tensor], [scatter_list] if scatter_list else [], opts)
    work.wait()

    # Each rank should receive its designated chunk
    expected = torch.full((chunk_size,), float(ctx.rank * 10), device=device, dtype=torch.float32)

    ctx.assert_true(
        torch.equal(output_tensor, expected),
        f"scatter result mismatch: got {output_tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_pg_scatter_non_zero_root(ctx: TestContext, device: str):
    """Test scatter with a non-zero root rank."""
    if ctx.world_size < 2:
        return  # Need at least 2 ranks

    pg = create_process_group(ctx)

    chunk_size = 4
    root_rank = ctx.world_size - 1  # Use last rank as root

    # Root rank prepares scatter_list with different data for each rank
    if ctx.rank == root_rank:
        scatter_list = [
            torch.full((chunk_size,), float(r * 100 + 7), device=device, dtype=torch.float32)
            for r in range(ctx.world_size)
        ]
    else:
        scatter_list = None

    # Output tensor for this rank
    output_tensor = torch.zeros((chunk_size,), device=device, dtype=torch.float32)

    # Run scatter
    from torch.distributed import ScatterOptions
    opts = ScatterOptions()
    opts.rootRank = root_rank
    work = pg.scatter([output_tensor], [scatter_list] if scatter_list else [], opts)
    work.wait()

    # Each rank should receive its designated chunk
    expected = torch.full((chunk_size,), float(ctx.rank * 100 + 7), device=device, dtype=torch.float32)

    ctx.assert_true(
        torch.equal(output_tensor, expected),
        f"scatter with root={root_rank} mismatch: got {output_tensor}, expected {expected}"
    )


@test_cpu_cuda
def test_pg_gather(ctx: TestContext, device: str):
    """Test gather: all ranks send to root rank which collects all chunks."""
    pg = create_process_group(ctx)

    chunk_size = 4
    root_rank = 0

    # Each rank prepares its input tensor with rank-specific data
    input_tensor = torch.full((chunk_size,), float(ctx.rank * 10), device=device, dtype=torch.float32)

    # Root rank prepares output list
    if ctx.rank == root_rank:
        gather_list = [
            torch.zeros((chunk_size,), device=device, dtype=torch.float32)
            for _ in range(ctx.world_size)
        ]
    else:
        gather_list = None

    # Run gather
    from torch.distributed import GatherOptions
    opts = GatherOptions()
    opts.rootRank = root_rank
    work = pg.gather([gather_list] if gather_list else [], [input_tensor], opts)
    work.wait()

    # Root rank should have all chunks
    if ctx.rank == root_rank:
        for r in range(ctx.world_size):
            expected = torch.full((chunk_size,), float(r * 10), device=device, dtype=torch.float32)
            ctx.assert_true(
                torch.equal(gather_list[r], expected),
                f"gather result mismatch at rank {r}: got {gather_list[r]}, expected {expected}"
            )


@test_cpu_cuda
def test_pg_gather_non_zero_root(ctx: TestContext, device: str):
    """Test gather with a non-zero root rank."""
    if ctx.world_size < 2:
        return  # Need at least 2 ranks

    pg = create_process_group(ctx)

    chunk_size = 4
    root_rank = ctx.world_size - 1  # Use last rank as root

    # Each rank prepares its input tensor with rank-specific data
    input_tensor = torch.full((chunk_size,), float(ctx.rank * 100 + 7), device=device, dtype=torch.float32)

    # Root rank prepares output list
    if ctx.rank == root_rank:
        gather_list = [
            torch.zeros((chunk_size,), device=device, dtype=torch.float32)
            for _ in range(ctx.world_size)
        ]
    else:
        gather_list = None

    # Run gather
    from torch.distributed import GatherOptions
    opts = GatherOptions()
    opts.rootRank = root_rank
    work = pg.gather([gather_list] if gather_list else [], [input_tensor], opts)
    work.wait()

    # Root rank should have all chunks
    if ctx.rank == root_rank:
        for r in range(ctx.world_size):
            expected = torch.full((chunk_size,), float(r * 100 + 7), device=device, dtype=torch.float32)
            ctx.assert_true(
                torch.equal(gather_list[r], expected),
                f"gather with root={root_rank} mismatch at rank {r}: got {gather_list[r]}, expected {expected}"
            )

