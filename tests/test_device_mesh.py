"""
Tests for moodist compatibility with PyTorch's DeviceMesh.

These tests verify that moodist works correctly when used through
torch.distributed.init_process_group and with DeviceMesh APIs.

This exercises a different code path than test_processgroup.py, which creates
ProcessGroups directly. When using init_process_group, PyTorch stores the
ProcessGroup in internal registries and later resolves it by name via
_resolve_process_group. This can cause issues if the resolved object is
different from the original (which happens with non-pybind11 backends).
"""

import torch
import torch.distributed as dist
from framework import TestContext, test


def _cleanup_distributed():
    """Clean up PyTorch distributed state."""
    if dist.is_initialized():
        dist.destroy_process_group()


@test
def test_device_mesh_basic(ctx: TestContext):
    """Test that DeviceMesh works with moodist backend.

    This test catches the bug where _resolve_process_group returns a different
    Python object than what's stored in pg_map, causing "Group is not registered"
    errors.
    """
    try:
        torch.cuda.set_device(ctx.local_rank)

        # Initialize via PyTorch's API (not direct MoodistProcessGroup creation)
        store = ctx.create_store(key="device_mesh_basic")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        # Create a 1D device mesh
        device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )

        # Get the group - this internally calls _resolve_process_group
        dim_group = device_mesh.get_group(0)

        # This was failing with "Group is not registered" before the fix
        group_rank = dist.get_rank(dim_group)
        ctx.assert_equal(group_rank, ctx.rank, "get_rank returned wrong value")

        group_size = dist.get_world_size(dim_group)
        ctx.assert_equal(group_size, ctx.world_size, "get_world_size returned wrong value")

    finally:
        _cleanup_distributed()


@test
def test_device_mesh_wrapper_registered(ctx: TestContext):
    """Test that the pybind11 wrapper is properly registered in _world.

    With moodist, default_pg is a MoodistProcessGroup but _resolve_process_group
    returns a pybind11 wrapper. Both should be usable for lookups.
    """
    try:
        torch.cuda.set_device(ctx.local_rank)

        store = ctx.create_store(key="device_mesh_identity")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        # Get the default process group (MoodistProcessGroup)
        default_pg = dist.distributed_c10d._world.default_pg

        # Create mesh and get its group (pybind11 wrapper via _resolve_process_group)
        device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )
        dim_group = device_mesh.get_group(0)

        # They may be different Python objects (MoodistProcessGroup vs pybind11 wrapper)
        # but both should be usable for distributed operations
        ctx.assert_true(
            dim_group in dist.distributed_c10d._world.pg_group_ranks,
            "dim_group (pybind11 wrapper) not found in pg_group_ranks"
        )
        ctx.assert_true(
            default_pg in dist.distributed_c10d._world.pg_group_ranks,
            "default_pg (MoodistProcessGroup) not found in pg_group_ranks"
        )

    finally:
        _cleanup_distributed()


@test
def test_device_mesh_custom_methods(ctx: TestContext):
    """Test that custom moodist methods work on the default ProcessGroup.

    Note: DeviceMesh.get_group() returns a pybind11 wrapper which doesn't have
    custom moodist methods. Use the default_pg directly for moodist-specific APIs.
    """
    try:
        torch.cuda.set_device(ctx.local_rank)

        store = ctx.create_store(key="device_mesh_custom")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        # The default_pg is the MoodistProcessGroup with custom methods
        default_pg = dist.distributed_c10d._world.default_pg

        # Custom moodist methods should be available on default_pg
        ctx.assert_true(
            hasattr(default_pg, "moodist_name"),
            "moodist_name method not available on default_pg"
        )

        name = default_pg.moodist_name()
        ctx.assert_true(
            isinstance(name, str) and len(name) > 0,
            f"moodist_name() returned invalid value: {name!r}"
        )

        # Test prefer_kernel_less methods
        ctx.assert_true(
            hasattr(default_pg, "get_prefer_kernel_less"),
            "get_prefer_kernel_less not available"
        )
        ctx.assert_true(
            hasattr(default_pg, "set_prefer_kernel_less"),
            "set_prefer_kernel_less not available"
        )

        # Should be able to get/set without error
        original = default_pg.get_prefer_kernel_less()
        default_pg.set_prefer_kernel_less(not original)
        ctx.assert_equal(
            default_pg.get_prefer_kernel_less(),
            not original,
            "set_prefer_kernel_less didn't take effect"
        )
        default_pg.set_prefer_kernel_less(original)  # restore

    finally:
        _cleanup_distributed()


@test
def test_device_mesh_collective(ctx: TestContext):
    """Test that collectives work through DeviceMesh groups."""
    try:
        torch.cuda.set_device(ctx.local_rank)

        store = ctx.create_store(key="device_mesh_collective")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )
        dim_group = device_mesh.get_group(0)

        # Run an allreduce through the mesh group
        tensor = torch.full((4,), float(ctx.rank + 1), device="cuda")
        dist.all_reduce(tensor, group=dim_group)

        expected_sum = ctx.world_size * (ctx.world_size + 1) / 2
        expected = torch.full((4,), expected_sum, device="cuda")
        ctx.assert_true(
            torch.allclose(tensor, expected),
            f"allreduce mismatch: got {tensor}, expected {expected}"
        )

    finally:
        _cleanup_distributed()


@test
def test_wrapper_cleanup_on_destroy(ctx: TestContext):
    """Test that the pybind11 wrapper is cleaned up when the ProcessGroup is destroyed.

    This tests that we don't leak the wrapper in _world.pg_group_ranks after
    destroy_process_group is called.
    """
    import gc

    torch.cuda.set_device(ctx.local_rank)

    store = ctx.create_store(key="wrapper_cleanup")
    dist.init_process_group(
        backend="moodist",
        store=store,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )

    # Get references before destruction

    # Find the wrapper (it's a different type than MoodistProcessGroup)
    wrappers_before = [
        pg for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"  # pybind11 wrapper type
    ]
    ctx.assert_true(
        len(wrappers_before) >= 1,
        f"Expected at least 1 wrapper in pg_group_ranks, found {len(wrappers_before)}"
    )

    # Count total entries before
    count_before = len(dist.distributed_c10d._world.pg_group_ranks)

    # Destroy the process group
    dist.destroy_process_group()

    # Force garbage collection to trigger weakref callbacks
    gc.collect()

    # Check that pg_group_ranks is cleaned up
    count_after = len(dist.distributed_c10d._world.pg_group_ranks)

    # After destroy, the dict should be empty (or at least smaller)
    ctx.assert_true(
        count_after < count_before,
        f"pg_group_ranks not cleaned up: before={count_before}, after={count_after}"
    )

    # Specifically check that our wrapper is gone
    wrappers_after = [
        pg for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"
    ]
    ctx.assert_equal(
        len(wrappers_after), 0,
        f"Wrapper still in pg_group_ranks after destroy: {wrappers_after}"
    )


@test
def test_single_group_destroy_cleanup(ctx: TestContext):
    """Test that destroying a single ProcessGroup cleans up its wrapper.

    When destroying just one group (not the entire world), PyTorch only removes
    that group from _world dicts. Without proper cleanup, the pybind11 wrapper
    we inserted would leak.
    """
    import gc

    torch.cuda.set_device(ctx.local_rank)

    store = ctx.create_store(key="single_destroy")
    dist.init_process_group(
        backend="moodist",
        store=store,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )

    # Create an additional group
    new_pg = dist.new_group(list(range(ctx.world_size)))

    # Count wrappers before
    wrappers_before = [
        pg for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"
    ]
    ctx.assert_equal(
        len(wrappers_before), 2,
        f"Expected 2 wrappers before destroy, got {len(wrappers_before)}"
    )

    # Destroy only the new group (not the world)
    dist.destroy_process_group(new_pg)

    # Must delete local reference for finalizer to fire
    del new_pg
    gc.collect()

    # Check that the wrapper for new_pg is cleaned up
    wrappers_after = [
        pg for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"
    ]

    # Should have 1 wrapper (for default_pg), not 2
    ctx.assert_equal(
        len(wrappers_after), 1,
        f"Wrapper leak! Expected 1 wrapper after destroying new_pg, got {len(wrappers_after)}"
    )

    # Clean up the rest
    dist.destroy_process_group()


@test
def test_dist_scatter_via_init(ctx: TestContext):
    """Test scatter using torch.distributed.scatter API with proper init."""
    try:
        torch.cuda.set_device(ctx.local_rank)
        device = "cuda"

        store = ctx.create_store(key="dist_scatter")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        chunk_size = 4
        src_rank = 0

        # Output tensor for this rank
        output_tensor = torch.zeros((chunk_size,), device=device, dtype=torch.float32)

        # Source rank prepares scatter_list
        if ctx.rank == src_rank:
            scatter_list = [
                torch.full((chunk_size,), float(r * 10), device=device, dtype=torch.float32)
                for r in range(ctx.world_size)
            ]
        else:
            scatter_list = None

        # Use torch.distributed.scatter
        dist.scatter(output_tensor, scatter_list, src=src_rank)

        # Verify result
        expected = torch.full((chunk_size,), float(ctx.rank * 10), device=device, dtype=torch.float32)
        ctx.assert_true(
            torch.equal(output_tensor, expected),
            f"dist.scatter mismatch: got {output_tensor}, expected {expected}"
        )

    finally:
        _cleanup_distributed()


@test
def test_dist_gather_via_init(ctx: TestContext):
    """Test gather using torch.distributed.gather API with proper init."""
    try:
        torch.cuda.set_device(ctx.local_rank)
        device = "cuda"

        store = ctx.create_store(key="dist_gather")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        chunk_size = 4
        dst_rank = 0

        # Input tensor for this rank
        input_tensor = torch.full((chunk_size,), float(ctx.rank * 10), device=device, dtype=torch.float32)

        # Dest rank prepares gather_list
        if ctx.rank == dst_rank:
            gather_list = [
                torch.zeros((chunk_size,), device=device, dtype=torch.float32)
                for _ in range(ctx.world_size)
            ]
        else:
            gather_list = None

        # Use torch.distributed.gather
        dist.gather(input_tensor, gather_list, dst=dst_rank)

        # Verify result on dest rank
        if ctx.rank == dst_rank:
            for r in range(ctx.world_size):
                expected = torch.full((chunk_size,), float(r * 10), device=device, dtype=torch.float32)
                ctx.assert_true(
                    torch.equal(gather_list[r], expected),
                    f"dist.gather mismatch at rank {r}: got {gather_list[r]}, expected {expected}"
                )

    finally:
        _cleanup_distributed()


@test
def test_dtensor_distribute_shard(ctx: TestContext):
    """Test DTensor distribute_tensor with Shard placement."""
    from torch.distributed.tensor import Shard, distribute_tensor

    try:
        torch.cuda.set_device(ctx.local_rank)
        device = "cuda"

        store = ctx.create_store(key="dtensor_shard")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        # Create a 1D device mesh
        mesh = dist.device_mesh.init_device_mesh(
            device,
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )

        # Create a global tensor - rank 0 has the data, others have zeros
        # (distribute_tensor will scatter from rank 0)
        global_size = ctx.world_size * 4
        if ctx.rank == 0:
            global_tensor = torch.arange(global_size, device=device, dtype=torch.float32)
        else:
            global_tensor = torch.zeros(global_size, device=device, dtype=torch.float32)

        # Distribute with Shard(0) placement - this uses scatter internally
        dtensor = distribute_tensor(global_tensor, mesh, [Shard(0)])

        # Verify local tensor
        local_tensor = dtensor.to_local()
        chunk_size = global_size // ctx.world_size
        expected_start = ctx.rank * chunk_size
        expected = torch.arange(expected_start, expected_start + chunk_size, device=device, dtype=torch.float32)

        ctx.assert_true(
            torch.equal(local_tensor, expected),
            f"DTensor shard mismatch: got {local_tensor}, expected {expected}"
        )

    finally:
        _cleanup_distributed()


@test
def test_dtensor_redistribute(ctx: TestContext):
    """Test DTensor redistribute from Replicate to Shard."""
    from torch.distributed.tensor import Shard, Replicate, distribute_tensor

    try:
        torch.cuda.set_device(ctx.local_rank)
        device = "cuda"

        store = ctx.create_store(key="dtensor_redistribute")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        mesh = dist.device_mesh.init_device_mesh(
            device,
            (ctx.world_size,),
            mesh_dim_names=("dp",),
        )

        # Create the same tensor on all ranks
        tensor_size = ctx.world_size * 4
        global_tensor = torch.arange(tensor_size, device=device, dtype=torch.float32)

        # First distribute as replicated (no communication needed)
        dtensor_replicated = distribute_tensor(global_tensor, mesh, [Replicate()])

        # Verify all ranks have the full tensor
        local_replicated = dtensor_replicated.to_local()
        ctx.assert_true(
            torch.equal(local_replicated, global_tensor),
            f"Replicated DTensor mismatch: got {local_replicated}"
        )

        # Now redistribute to sharded - this tests the redistribution path
        dtensor_sharded = dtensor_replicated.redistribute(mesh, [Shard(0)])

        # Verify local shard
        local_sharded = dtensor_sharded.to_local()
        chunk_size = tensor_size // ctx.world_size
        expected_start = ctx.rank * chunk_size
        expected = torch.arange(expected_start, expected_start + chunk_size, device=device, dtype=torch.float32)

        ctx.assert_true(
            torch.equal(local_sharded, expected),
            f"Sharded DTensor mismatch: got {local_sharded}, expected {expected}"
        )

    finally:
        _cleanup_distributed()


@test
def test_dtensor_2d_mesh_shard(ctx: TestContext):
    """Test DTensor distribute_tensor with a 2D mesh and Shard placements.

    This tests the case where the device mesh creates subgroups that are
    strict subsets of the world. The 1D mesh tests don't catch this because
    the mesh subgroup equals the default process group.

    With a 2D mesh (e.g., 2x2), init_device_mesh creates row and column
    subgroups. When distribute_tensor uses Shard placement, it calls
    mesh_scatter which internally uses get_rank(dim_group). This can fail
    if the subgroup isn't properly registered with the correct global ranks.
    """
    from torch.distributed.tensor import Shard, distribute_tensor

    # This test requires exactly 4 ranks for a 2x2 mesh
    if ctx.world_size != 4:
        ctx.log(f"Skipping: test requires exactly 4 ranks, got {ctx.world_size}")
        return

    try:
        torch.cuda.set_device(ctx.local_rank)
        device = "cuda"

        store = ctx.create_store(key="dtensor_2d_mesh")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        # Create a 2D device mesh (2x2)
        # This creates subgroups:
        # - "a" dimension: 2 groups of 2 (e.g., [0,1] and [2,3] or [0,2] and [1,3])
        # - "b" dimension: 2 groups of 2 (the other split)
        mesh = dist.device_mesh.init_device_mesh(
            device,
            (2, 2),
            mesh_dim_names=("a", "b"),
        )

        # Create a tensor that will be sharded along both dimensions
        # Shape (4, 8) with Shard(0), Shard(1) on a 2x2 mesh means:
        # - Each rank gets a (2, 4) local tensor
        if ctx.rank == 0:
            global_tensor = torch.arange(4 * 8, device=device, dtype=torch.float32).view(4, 8)
        else:
            global_tensor = torch.zeros(4, 8, device=device, dtype=torch.float32)

        # Distribute with sharding on both dimensions
        # This triggers mesh_scatter on both mesh dimensions
        placements = [Shard(0), Shard(1)]
        dtensor = distribute_tensor(global_tensor, mesh, placements)

        # Verify local tensor shape
        local_tensor = dtensor.to_local()
        ctx.assert_equal(
            list(local_tensor.shape), [2, 4],
            f"Expected local shape [2, 4], got {list(local_tensor.shape)}"
        )

        # Verify content based on mesh coordinates
        # The mesh coordinates determine which shard this rank owns
        mesh_coord = mesh.get_coordinate()
        ctx.assert_true(
            mesh_coord is not None,
            "mesh.get_coordinate() returned None"
        )

        row, col = mesh_coord
        # Expected values: row determines which rows (0-1 or 2-3), col determines which cols (0-3 or 4-7)
        expected_row_start = row * 2
        expected_col_start = col * 4
        expected = torch.arange(4 * 8, device=device, dtype=torch.float32).view(4, 8)
        expected = expected[expected_row_start:expected_row_start + 2, expected_col_start:expected_col_start + 4]

        ctx.assert_true(
            torch.equal(local_tensor, expected),
            f"Rank {ctx.rank} (coord {mesh_coord}): got {local_tensor}, expected {expected}"
        )

    finally:
        _cleanup_distributed()

