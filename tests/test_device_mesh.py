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
        store = ctx.create_store(key="device_mesh_collective")
        dist.init_process_group(
            backend="moodist",
            store=store,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )

        torch.cuda.set_device(ctx.local_rank)

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

    store = ctx.create_store(key="wrapper_cleanup")
    dist.init_process_group(
        backend="moodist",
        store=store,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )

    # Get references before destruction
    default_pg = dist.distributed_c10d._world.default_pg

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
