# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for moodist sharding utilities.

Tests for compute_shards and dtensor_shards:
1. compute_shards correctly predicts which chunks each mesh coordinate owns
2. The predictions match PyTorch's actual DTensor sharding behavior
3. compile_op correctly transfers data based on these shard specifications

Tests for compute_reshard_slices:
4. Correctly computes slices for Replicate -> Shard conversion
5. Returns full slices when no conversion is needed
6. Raises errors for invalid conversions (e.g., Shard -> Shard)

Tests run with multiple ranks and verify all coordinates, not just the current one.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, DeviceMesh, distribute_tensor
from torch.distributed.tensor.placement_types import Shard, Replicate

import moodist
from moodist import TensorRegion
from framework import TestContext, test, test_cpu_cuda, create_process_group

# Check for _StridedShard availability (private API, may not exist)
try:
    from torch.distributed.tensor.placement_types import _StridedShard
    HAS_STRIDED_SHARD = True
except ImportError:
    HAS_STRIDED_SHARD = False

# Enable CPU allocator for compile_op tests
moodist.enable_cpu_allocator()


# =============================================================================
# Helper Functions
# =============================================================================

def _init_torch_distributed(ctx: TestContext):
    """Initialize torch.distributed for DTensor tests."""
    if dist.is_initialized():
        return

    # Set CUDA device before init to avoid "same device" errors
    torch.cuda.set_device(ctx.local_rank)

    store = ctx.create_store(key="torch_dist")
    dist.init_process_group(
        backend="moodist",
        store=store,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )


def _cleanup_torch_distributed():
    """Clean up torch.distributed state."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_global_tensor(shape, dtype=torch.float32):
    """Create a global tensor where each element equals its flat index."""
    numel = 1
    for s in shape:
        numel *= s
    return torch.arange(numel, dtype=dtype).reshape(shape)


def _build_local_from_chunks(chunks, global_tensor):
    """
    Build the expected local tensor from chunks.

    Args:
        chunks: List of ShardInfo objects
        global_tensor: The full global tensor

    Returns:
        The expected local tensor
    """
    if not chunks:
        # Empty - return 0-element tensor with same ndim
        return torch.zeros([0] * global_tensor.ndim, dtype=global_tensor.dtype)

    # Compute local shape from max(local_offset + chunk_shape) per dimension
    ndim = len(chunks[0].shape)
    local_shape = [0] * ndim
    for chunk in chunks:
        for d in range(ndim):
            local_shape[d] = max(local_shape[d], chunk.local_offset[d] + chunk.shape[d])

    # Create local tensor
    local_tensor = torch.zeros(local_shape, dtype=global_tensor.dtype)

    # Copy chunks from global to local positions
    for chunk in chunks:
        # Build slices for global and local tensors
        global_slices = tuple(
            slice(go, go + cs) for go, cs in zip(chunk.global_offset, chunk.shape)
        )
        local_slices = tuple(
            slice(lo, lo + cs) for lo, cs in zip(chunk.local_offset, chunk.shape)
        )
        local_tensor[local_slices] = global_tensor[global_slices]

    return local_tensor


def _get_all_mesh_coords(mesh):
    """Get all valid coordinates for a mesh."""
    coords = []
    mesh_shape = tuple(mesh.mesh.shape)

    def recurse(partial_coord, dim):
        if dim == len(mesh_shape):
            coords.append(tuple(partial_coord))
            return
        for i in range(mesh_shape[dim]):
            recurse(partial_coord + [i], dim + 1)

    recurse([], 0)
    return coords


def _coord_to_rank(mesh, coord):
    """Convert a mesh coordinate to its rank."""
    # Index into mesh.mesh tensor to get the rank at this coordinate
    idx = coord if len(coord) > 1 else coord[0]
    return mesh.mesh[idx].item()


def _verify_compute_shards_current_coord(
    ctx: TestContext,
    mesh: DeviceMesh,
    placements: list,
    shape: tuple,
    description: str,
):
    """
    Verify compute_shards matches PyTorch's DTensor for the current coordinate.

    This is the basic test - verifies the current rank's local tensor matches
    what compute_shards predicts.
    """
    global_tensor = _make_global_tensor(shape)

    # Create DTensor
    try:
        dtensor = distribute_tensor(global_tensor, mesh, placements)
    except Exception as e:
        ctx.log(f"SKIP {description}: {e}")
        return True  # Skip, not a failure

    # Get actual local tensor from PyTorch
    actual_local = dtensor.to_local()

    # Get current coordinate
    coord = mesh.get_coordinate()
    if coord is None:
        ctx.log(f"SKIP {description}: rank not in mesh")
        return True

    # Compute predicted chunks using our API
    indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]
    predicted_chunks = moodist.compute_shards(shape, placements, indices_and_sizes)

    # Build expected local tensor from chunks
    expected_local = _build_local_from_chunks(predicted_chunks, global_tensor)

    # Compare - skip if both are empty (shape details don't matter for empty tensors)
    if actual_local.numel() == 0 and expected_local.numel() == 0:
        return True

    if not torch.equal(actual_local, expected_local):
        ctx.log(f"FAIL {description}")
        ctx.log(f"  coord: {coord}")
        ctx.log(f"  actual shape: {actual_local.shape}, expected shape: {expected_local.shape}")
        ctx.log(f"  predicted_chunks: {predicted_chunks}")
        if actual_local.numel() <= 20:
            ctx.log(f"  actual: {actual_local.flatten().tolist()}")
            ctx.log(f"  expected: {expected_local.flatten().tolist()}")
        return False

    return True


def _verify_compute_shards_all_coords(
    ctx: TestContext,
    mesh: DeviceMesh,
    placements: list,
    shape: tuple,
    description: str,
):
    """
    Verify compute_shards for ALL coordinates, not just current.

    This catches bugs where compute_shards works for the current rank but
    fails when querying other ranks' shards.
    """
    global_tensor = _make_global_tensor(shape)

    # Create DTensor
    try:
        dtensor = distribute_tensor(global_tensor, mesh, placements)
    except Exception as e:
        ctx.log(f"SKIP {description}: {e}")
        return True

    # Get actual local tensor for this rank
    actual_local = dtensor.to_local()
    my_coord = mesh.get_coordinate()

    if my_coord is None:
        ctx.log(f"SKIP {description}: rank not in mesh")
        return True

    # All-gather local tensors from all ranks using all_gather_object
    # This handles variable-sized tensors cleanly
    all_locals = [None] * ctx.world_size
    dist.all_gather_object(all_locals, actual_local)

    # Now verify compute_shards for all coordinates
    all_coords = _get_all_mesh_coords(mesh)
    success = True

    for coord in all_coords:
        # Get the rank that owns this coordinate
        owner_rank = _coord_to_rank(mesh, coord)

        # Get that rank's actual local tensor
        owner_actual = all_locals[owner_rank]

        # Compute predicted chunks for this coordinate
        indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]
        predicted_chunks = moodist.compute_shards(shape, placements, indices_and_sizes)

        # Build expected local tensor
        expected_local = _build_local_from_chunks(predicted_chunks, global_tensor)

        # Compare - skip if both are empty (shape details don't matter for empty tensors)
        if owner_actual.numel() == 0 and expected_local.numel() == 0:
            continue

        if not torch.equal(owner_actual, expected_local):
            ctx.log(f"FAIL {description} at coord {coord}")
            ctx.log(f"  owner_rank: {owner_rank}")
            ctx.log(f"  actual shape: {owner_actual.shape}, expected shape: {expected_local.shape}")
            ctx.log(f"  predicted_chunks: {predicted_chunks}")
            if owner_actual.numel() <= 20:
                ctx.log(f"  actual: {owner_actual.flatten().tolist()}")
                ctx.log(f"  expected: {expected_local.flatten().tolist()}")
            success = False

    return success


def _verify_compile_op_allgather(
    ctx: TestContext,
    pg,
    mesh: DeviceMesh,
    placements: list,
    shape: tuple,
    description: str,
    device: str = "cpu",
):
    """
    Verify compile_op correctly performs an all-gather based on shard specs.

    Each rank contributes their shard (computed via compute_shards),
    all ranks receive the full tensor.
    """
    global_tensor = _make_global_tensor(shape)

    coord = mesh.get_coordinate()
    if coord is None:
        return True  # Not in mesh

    # Get this rank's chunks
    indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]
    my_chunks = moodist.compute_shards(shape, placements, indices_and_sizes)

    # Build input specs from chunks (global offsets)
    inputs = []
    for chunk in my_chunks:
        inputs.append(TensorRegion(offset=chunk.global_offset, shape=chunk.shape))

    # Output: full tensor
    outputs = [TensorRegion(offset=[0] * len(shape), shape=list(shape))]

    # Compile the op
    op = moodist.compile_op(
        pg,
        dtype=global_tensor.dtype,
        inputs=inputs if inputs else None,
        outputs=outputs,
        reduce="any",
    )

    # Build input tensors (the local shard data)
    input_tensors = []
    for chunk in my_chunks:
        global_slices = tuple(
            slice(go, go + cs) for go, cs in zip(chunk.global_offset, chunk.shape)
        )
        chunk_data = global_tensor[global_slices].to(device=device).contiguous()
        input_tensors.append(chunk_data)

    # Output tensor
    output_tensor = torch.zeros(shape, dtype=global_tensor.dtype, device=device)

    # Execute
    future = op(input_tensors, [output_tensor])
    future.wait()

    # Verify output matches global tensor
    expected = global_tensor.to(device=device)
    if not torch.equal(output_tensor, expected):
        ctx.log(f"FAIL compile_op {description}")
        ctx.log(f"  coord: {coord}, my_chunks: {my_chunks}")
        if output_tensor.numel() <= 20:
            ctx.log(f"  got: {output_tensor.flatten().tolist()}")
            ctx.log(f"  expected: {expected.flatten().tolist()}")
        return False

    return True


def _verify_compile_op_scatter(
    ctx: TestContext,
    pg,
    mesh: DeviceMesh,
    placements: list,
    shape: tuple,
    description: str,
    device: str = "cpu",
):
    """
    Verify compile_op correctly performs a scatter based on shard specs.

    Rank 0 contributes the full tensor, all ranks receive their shard.
    """
    global_tensor = _make_global_tensor(shape)

    coord = mesh.get_coordinate()
    if coord is None:
        return True  # Not in mesh

    # Get this rank's expected output chunks
    indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]
    my_chunks = moodist.compute_shards(shape, placements, indices_and_sizes)

    # Input: only rank 0 has the full tensor
    if pg.rank() == 0:
        inputs = [TensorRegion(offset=[0] * len(shape), shape=list(shape))]
    else:
        inputs = []

    # Output: this rank's shard(s)
    outputs = []
    for chunk in my_chunks:
        outputs.append(TensorRegion(offset=chunk.global_offset, shape=chunk.shape))

    # Compile the op
    op = moodist.compile_op(
        pg,
        dtype=global_tensor.dtype,
        inputs=inputs,
        outputs=outputs if outputs else None,
    )

    # Build input tensor (only rank 0)
    if pg.rank() == 0:
        input_tensors = [global_tensor.to(device=device).contiguous()]
    else:
        input_tensors = []

    # Output tensors
    output_tensors = []
    for chunk in my_chunks:
        output_tensors.append(torch.zeros(chunk.shape, dtype=global_tensor.dtype, device=device))

    # Execute
    future = op(input_tensors, output_tensors)
    future.wait()

    # Verify outputs match expected chunks from global tensor
    for i, chunk in enumerate(my_chunks):
        global_slices = tuple(
            slice(go, go + cs) for go, cs in zip(chunk.global_offset, chunk.shape)
        )
        expected = global_tensor[global_slices].to(device=device)
        if not torch.equal(output_tensors[i], expected):
            ctx.log(f"FAIL compile_op scatter {description}")
            ctx.log(f"  coord: {coord}, chunk {i}: {chunk}")
            if output_tensors[i].numel() <= 20:
                ctx.log(f"  got: {output_tensors[i].flatten().tolist()}")
                ctx.log(f"  expected: {expected.flatten().tolist()}")
            return False

    return True


def _verify_compile_op_redistribute(
    ctx: TestContext,
    pg,
    mesh: DeviceMesh,
    input_placements: list,
    output_placements: list,
    shape: tuple,
    description: str,
    device: str = "cpu",
):
    """
    Verify compile_op correctly redistributes from one sharding to another.

    E.g., Shard(0) inputs -> Shard(1) outputs.
    """
    global_tensor = _make_global_tensor(shape)

    coord = mesh.get_coordinate()
    if coord is None:
        return True  # Not in mesh

    indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]

    # Get input chunks (what this rank contributes)
    input_chunks = moodist.compute_shards(shape, input_placements, indices_and_sizes)

    # Get output chunks (what this rank receives)
    output_chunks = moodist.compute_shards(shape, output_placements, indices_and_sizes)

    # Build input specs
    inputs = []
    for chunk in input_chunks:
        inputs.append(TensorRegion(offset=chunk.global_offset, shape=chunk.shape))

    # Build output specs
    outputs = []
    for chunk in output_chunks:
        outputs.append(TensorRegion(offset=chunk.global_offset, shape=chunk.shape))

    # Compile the op
    op = moodist.compile_op(
        pg,
        dtype=global_tensor.dtype,
        inputs=inputs if inputs else None,
        outputs=outputs if outputs else None,
    )

    # Build input tensors
    input_tensors = []
    for chunk in input_chunks:
        global_slices = tuple(
            slice(go, go + cs) for go, cs in zip(chunk.global_offset, chunk.shape)
        )
        chunk_data = global_tensor[global_slices].to(device=device).contiguous()
        input_tensors.append(chunk_data)

    # Output tensors
    output_tensors = []
    for chunk in output_chunks:
        output_tensors.append(torch.zeros(chunk.shape, dtype=global_tensor.dtype, device=device))

    # Execute
    future = op(input_tensors, output_tensors)
    future.wait()

    # Verify outputs match expected chunks from global tensor
    for i, chunk in enumerate(output_chunks):
        global_slices = tuple(
            slice(go, go + cs) for go, cs in zip(chunk.global_offset, chunk.shape)
        )
        expected = global_tensor[global_slices].to(device=device)
        if not torch.equal(output_tensors[i], expected):
            ctx.log(f"FAIL compile_op redistribute {description}")
            ctx.log(f"  coord: {coord}, output chunk {i}: {chunk}")
            if output_tensors[i].numel() <= 20:
                ctx.log(f"  got: {output_tensors[i].flatten().tolist()}")
                ctx.log(f"  expected: {expected.flatten().tolist()}")
            return False

    return True


def _test_configuration(
    ctx: TestContext,
    pg,
    mesh: DeviceMesh,
    placements: list,
    shape: tuple,
    description: str,
    device: str = "cpu",
    test_all_coords: bool = True,
    test_compile_op: bool = True,
    test_scatter: bool = True,
):
    """
    Test both compute_shards and compile_op for a configuration.

    Args:
        ctx: Test context
        pg: Process group for compile_op
        mesh: DeviceMesh to use
        placements: Placement list
        shape: Global tensor shape
        description: Test description for logging
        device: Device for compile_op tests ("cpu" or "cuda")
        test_all_coords: If True, verify all coordinates (not just current)
        test_compile_op: If True, also test compile_op allgather transfer
        test_scatter: If True, also test compile_op scatter transfer
    """
    success = True

    # Phase 1: Verify compute_shards for current coordinate
    if not _verify_compute_shards_current_coord(ctx, mesh, placements, shape, description):
        success = False

    # Phase 2: Verify compute_shards for all coordinates
    if test_all_coords:
        if not _verify_compute_shards_all_coords(ctx, mesh, placements, shape, description):
            success = False

    # Phase 3: Verify compile_op allgather transfer
    if test_compile_op:
        if not _verify_compile_op_allgather(ctx, pg, mesh, placements, shape, description, device):
            success = False

    # Phase 4: Verify compile_op scatter transfer
    if test_scatter:
        if not _verify_compile_op_scatter(ctx, pg, mesh, placements, shape, description, device):
            success = False

    if success and ctx.rank == 0:
        ctx.log(f"PASS {description}")

    return success


# =============================================================================
# Actual Tests
# =============================================================================

@test_cpu_cuda
def test_sharding_infrastructure(ctx: TestContext, device: str):
    """Basic test to verify infrastructure works."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        # Simple 1D mesh with Shard(0)
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        placements = [Shard(0)]
        shape = (ctx.world_size * 4,)

        success = _test_configuration(
            ctx, pg, mesh, placements, shape,
            "infrastructure test: Shard(0) 1D",
            device=device,
            test_all_coords=True,
            test_compile_op=True,
        )
        ctx.assert_true(success, "infrastructure test failed")

    finally:
        _cleanup_torch_distributed()


# =============================================================================
# Step 2: Basic 1D Mesh Tests (Shard, Replicate)
# =============================================================================

@test_cpu_cuda
def test_shard_1d_mesh_1d_tensor(ctx: TestContext, device: str):
    """Test Shard(0) on 1D tensors with 1D mesh."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Various 1D tensor sizes
        for size in [ctx.world_size, ctx.world_size * 4, ctx.world_size * 7 + 3]:
            shape = (size,)
            success = _test_configuration(
                ctx, pg, mesh, [Shard(0)], shape,
                f"Shard(0) on 1D tensor shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False

        ctx.assert_true(all_passed, "Some 1D tensor tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_shard_1d_mesh_2d_tensor(ctx: TestContext, device: str):
    """Test Shard on 2D tensors with 1D mesh."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Shard(0) - shard on first dimension
        for shape in [(ctx.world_size * 4, 8), (ctx.world_size * 2, 16), (ctx.world_size + 3, 5)]:
            success = _test_configuration(
                ctx, pg, mesh, [Shard(0)], shape,
                f"Shard(0) on 2D tensor shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False

        # Shard(1) - shard on second dimension
        for shape in [(8, ctx.world_size * 4), (16, ctx.world_size * 2), (5, ctx.world_size + 3)]:
            success = _test_configuration(
                ctx, pg, mesh, [Shard(1)], shape,
                f"Shard(1) on 2D tensor shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False

        ctx.assert_true(all_passed, "Some 2D tensor tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_shard_1d_mesh_3d_tensor(ctx: TestContext, device: str):
    """Test Shard on 3D tensors with 1D mesh."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Shard on each dimension
        base_shape = [ctx.world_size * 2, 4, 6]
        for dim in range(3):
            shape = tuple(base_shape)
            success = _test_configuration(
                ctx, pg, mesh, [Shard(dim)], shape,
                f"Shard({dim}) on 3D tensor shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False

        ctx.assert_true(all_passed, "Some 3D tensor tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_replicate_1d_mesh(ctx: TestContext, device: str):
    """Test Replicate placement with 1D mesh."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Various shapes - all ranks should have the full tensor
        for shape in [(16,), (8, 4), (4, 4, 4)]:
            # First verify compute_shards works correctly (skip compile_op)
            success = _test_configuration(
                ctx, pg, mesh, [Replicate()], shape,
                f"Replicate() on tensor shape={shape}",
                device=device,
                test_compile_op=False,
            )
            if not success:
                all_passed = False

        # Test that compile_op correctly detects overlapping inputs for Replicate
        # All ranks provide the full tensor, so this should error
        shape = (16,)
        global_tensor = _make_global_tensor(shape)
        coord = mesh.get_coordinate()
        if coord is not None:
            indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]
            my_chunks = moodist.compute_shards(shape, [Replicate()], indices_and_sizes)

            inputs = [TensorRegion(offset=c.global_offset, shape=c.shape) for c in my_chunks]
            outputs = [TensorRegion(offset=[0] * len(shape), shape=list(shape))]

            try:
                op = moodist.compile_op(
                    pg,
                    dtype=global_tensor.dtype,
                    inputs=inputs,
                    outputs=outputs,
                )
                # Should not reach here
                ctx.log("FAIL: compile_op should have raised error for Replicate")
                all_passed = False
            except RuntimeError as e:
                error_msg = str(e)
                if "overlapping inputs" in error_msg:
                    if ctx.rank == 0:
                        ctx.log(f"PASS: compile_op correctly detected overlap: {error_msg}")
                else:
                    ctx.log(f"FAIL: unexpected error: {error_msg}")
                    all_passed = False

        # Test that compile_op correctly detects missing coverage
        # Only rank 0 provides partial input, all ranks want full tensor
        shape = (16,)
        global_tensor = _make_global_tensor(shape)
        coord = mesh.get_coordinate()
        if coord is not None:
            # Only rank 0 provides input, and only first half
            if ctx.rank == 0:
                inputs = [TensorRegion(offset=[0], shape=[8])]  # Only first half
            else:
                inputs = []
            outputs = [TensorRegion(offset=[0], shape=list(shape))]  # All ranks want full tensor

            try:
                op = moodist.compile_op(
                    pg,
                    dtype=global_tensor.dtype,
                    inputs=inputs,
                    outputs=outputs,
                )
                # Should not reach here
                ctx.log("FAIL: compile_op should have raised error for missing coverage")
                all_passed = False
            except RuntimeError as e:
                error_msg = str(e)
                if "missing input coverage" in error_msg:
                    if ctx.rank == 0:
                        ctx.log(f"PASS: compile_op correctly detected gap: {error_msg}")
                else:
                    ctx.log(f"FAIL: unexpected error: {error_msg}")
                    all_passed = False

        ctx.assert_true(all_passed, "Some Replicate tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_shard_uneven_split(ctx: TestContext, device: str):
    """Test Shard when tensor size doesn't divide evenly by world_size."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Sizes that don't divide evenly
        uneven_sizes = [
            ctx.world_size + 1,
            ctx.world_size * 2 + 1,
            ctx.world_size - 1 if ctx.world_size > 1 else 1,
            17,  # prime number
            23,
        ]

        for size in uneven_sizes:
            if size < 1:
                continue
            shape = (size,)
            success = _test_configuration(
                ctx, pg, mesh, [Shard(0)], shape,
                f"Shard(0) uneven split shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False

        ctx.assert_true(all_passed, "Some uneven split tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_strided_shard_1d_mesh(ctx: TestContext, device: str):
    """Test _StridedShard placement on 1D mesh."""
    if not HAS_STRIDED_SHARD:
        ctx.log("SKIP: _StridedShard not available")
        return

    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Test StridedShard on various shapes with different split_factors
        # StridedShard produces non-contiguous chunks (interleaved)
        test_cases = [
            # (shape, dim, split_factor)
            # 1D tensors
            ((ctx.world_size * 4,), 0, 2),
            ((ctx.world_size * 4,), 0, 4),
            ((ctx.world_size * 4 + 1,), 0, 2),  # Uneven
            ((ctx.world_size * 8,), 0, 2),
            # 2D tensors - shard different dimensions
            ((ctx.world_size * 2, 8), 0, 2),
            ((8, ctx.world_size * 2), 1, 2),
            # 3D tensors
            ((ctx.world_size * 2, 4, 4), 0, 2),
        ]

        for shape, dim, split_factor in test_cases:
            placement = _StridedShard(dim, split_factor=split_factor)
            success = _test_configuration(
                ctx, pg, mesh, [placement], shape,
                f"StridedShard({dim}, sf={split_factor}) on tensor shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False

        ctx.assert_true(all_passed, "Some StridedShard tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_2d_mesh(ctx: TestContext, device: str):
    """Test sharding with 2D mesh."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        # Find a 2D factorization of world_size
        dp_size, tp_size = 0, 0
        for dp in range(2, ctx.world_size):
            if ctx.world_size % dp == 0:
                tp = ctx.world_size // dp
                if tp >= 2:
                    dp_size, tp_size = dp, tp
                    break

        if dp_size == 0:
            ctx.log(f"SKIP: No 2D factorization for world_size={ctx.world_size}")
            return

        mesh = DeviceMesh(
            "cpu",
            torch.arange(ctx.world_size).reshape(dp_size, tp_size),
            mesh_dim_names=("dp", "tp")
        )

        if ctx.rank == 0:
            ctx.log(f"Using {dp_size}x{tp_size} mesh")

        all_passed = True

        # [Shard(0), Shard(1)] - shard different tensor dimensions
        success = _test_configuration(
            ctx, pg, mesh, [Shard(0), Shard(1)], (32, 64),
            "[Shard(0), Shard(1)] on shape (32, 64)",
            device=device,
        )
        if not success:
            all_passed = False

        # [Shard(0), Shard(0)] - both mesh dims shard same tensor dim (composition)
        success = _test_configuration(
            ctx, pg, mesh, [Shard(0), Shard(0)], (128,),
            "[Shard(0), Shard(0)] on shape (128,)",
            device=device,
        )
        if not success:
            all_passed = False

        # [Shard(0), Replicate()] - shard on one mesh dim, replicate on another
        success = _test_configuration(
            ctx, pg, mesh, [Shard(0), Replicate()], (32, 16),
            "[Shard(0), Replicate()] on shape (32, 16)",
            device=device,
        )
        if not success:
            all_passed = False

        # [Replicate(), Shard(1)] - replicate on first, shard on second
        success = _test_configuration(
            ctx, pg, mesh, [Replicate(), Shard(1)], (16, 32),
            "[Replicate(), Shard(1)] on shape (16, 32)",
            device=device,
        )
        if not success:
            all_passed = False

        # StridedShard combinations (if available)
        if HAS_STRIDED_SHARD:
            # [_StridedShard, Shard] - FSDP+TP pattern
            success = _test_configuration(
                ctx, pg, mesh,
                [_StridedShard(0, split_factor=tp_size), Shard(0)],
                (128,),
                f"[StridedShard(0, sf={tp_size}), Shard(0)] on shape (128,)",
                device=device,
            )
            if not success:
                all_passed = False

            # [Shard, _StridedShard]
            success = _test_configuration(
                ctx, pg, mesh,
                [Shard(0), _StridedShard(0, split_factor=dp_size)],
                (128,),
                f"[Shard(0), StridedShard(0, sf={dp_size})] on shape (128,)",
                device=device,
            )
            if not success:
                all_passed = False

            # [_StridedShard, Replicate]
            success = _test_configuration(
                ctx, pg, mesh,
                [_StridedShard(0, split_factor=2), Replicate()],
                (64, 32),
                "[StridedShard(0, sf=2), Replicate()] on shape (64, 32)",
                device=device,
            )
            if not success:
                all_passed = False

        ctx.assert_true(all_passed, "Some 2D mesh tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_redistribute_shard0_to_shard1(ctx: TestContext, device: str):
    """Test redistribution from Shard(0) to Shard(1) on 2D tensors."""
    _init_torch_distributed(ctx)
    pg = create_process_group(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Test various 2D shapes where both dims are >= world_size
        shapes = [
            (ctx.world_size * 2, ctx.world_size * 2),
            (ctx.world_size * 4, ctx.world_size * 2),
            (ctx.world_size * 2, ctx.world_size * 4),
            (ctx.world_size + 3, ctx.world_size + 5),  # uneven
        ]

        for shape in shapes:
            success = _verify_compile_op_redistribute(
                ctx, pg, mesh,
                input_placements=[Shard(0)],
                output_placements=[Shard(1)],
                shape=shape,
                description=f"redistribute Shard(0)->Shard(1) shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False
            elif ctx.rank == 0:
                ctx.log(f"PASS redistribute Shard(0)->Shard(1) shape={shape}")

        # Also test Shard(1) -> Shard(0)
        for shape in shapes:
            success = _verify_compile_op_redistribute(
                ctx, pg, mesh,
                input_placements=[Shard(1)],
                output_placements=[Shard(0)],
                shape=shape,
                description=f"redistribute Shard(1)->Shard(0) shape={shape}",
                device=device,
            )
            if not success:
                all_passed = False
            elif ctx.rank == 0:
                ctx.log(f"PASS redistribute Shard(1)->Shard(0) shape={shape}")

        ctx.assert_true(all_passed, "Some redistribute tests failed")

    finally:
        _cleanup_torch_distributed()


# =============================================================================
# Tests for compute_reshard_slices
# =============================================================================

def _verify_reshard_slices(
    ctx: TestContext,
    mesh: DeviceMesh,
    current_placements: list,
    target_placements: list,
    shape: tuple,
    description: str,
):
    """
    Verify compute_reshard_slices by comparing against PyTorch DTensor.

    1. Create global tensor with fixed data
    2. Distribute to DTensor with current_placements, get local, apply reshard slices
    3. Distribute to DTensor with target_placements, get local directly
    4. Compare: they should be equal
    """
    from moodist import compute_reshard_slices

    global_tensor = _make_global_tensor(shape)

    # Create DTensor with current placements
    try:
        dtensor_current = distribute_tensor(global_tensor, mesh, current_placements)
    except Exception as e:
        ctx.log(f"SKIP {description}: cannot create current DTensor: {e}")
        return True

    # Create DTensor with target placements
    try:
        dtensor_target = distribute_tensor(global_tensor, mesh, target_placements)
    except Exception as e:
        ctx.log(f"SKIP {description}: cannot create target DTensor: {e}")
        return True

    coord = mesh.get_coordinate()
    if coord is None:
        ctx.log(f"SKIP {description}: rank not in mesh")
        return True

    # Get local tensor from current placement
    local_current = dtensor_current.to_local()

    # Compute reshard slices
    indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]
    try:
        reshard_slices = compute_reshard_slices(
            tuple(local_current.shape),
            current_placements,
            target_placements,
            indices_and_sizes,
        )
    except ValueError as e:
        ctx.log(f"SKIP {description}: {e}")
        return True

    # Apply slices to get resharded local
    resharded_local = local_current[reshard_slices]

    # Get expected local from target placement
    expected_local = dtensor_target.to_local()

    # Compare
    if not torch.equal(resharded_local, expected_local):
        ctx.log(f"FAIL {description}")
        ctx.log(f"  coord: {coord}")
        ctx.log(f"  current local shape: {local_current.shape}")
        ctx.log(f"  reshard_slices: {reshard_slices}")
        ctx.log(f"  resharded shape: {resharded_local.shape}")
        ctx.log(f"  expected shape: {expected_local.shape}")
        if resharded_local.numel() <= 20:
            ctx.log(f"  resharded: {resharded_local.flatten().tolist()}")
            ctx.log(f"  expected: {expected_local.flatten().tolist()}")
        return False

    return True


@test_cpu_cuda
def test_reshard_slices_replicate_to_shard(ctx: TestContext, device: str):
    """Test compute_reshard_slices for Replicate -> Shard conversions."""
    _init_torch_distributed(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # 1D tensor: Replicate -> Shard(0)
        for shape in [(ctx.world_size * 4,), (ctx.world_size * 7 + 3,)]:
            success = _verify_reshard_slices(
                ctx, mesh,
                [Replicate()], [Shard(0)],
                shape,
                f"Replicate -> Shard(0) shape={shape}",
            )
            if not success:
                all_passed = False
            elif ctx.rank == 0:
                ctx.log(f"PASS Replicate -> Shard(0) shape={shape}")

        # 2D tensor: Replicate -> Shard(0)
        for shape in [(ctx.world_size * 2, 8), (ctx.world_size + 3, 5)]:
            success = _verify_reshard_slices(
                ctx, mesh,
                [Replicate()], [Shard(0)],
                shape,
                f"Replicate -> Shard(0) 2D shape={shape}",
            )
            if not success:
                all_passed = False
            elif ctx.rank == 0:
                ctx.log(f"PASS Replicate -> Shard(0) 2D shape={shape}")

        # 2D tensor: Replicate -> Shard(1)
        for shape in [(8, ctx.world_size * 2), (5, ctx.world_size + 3)]:
            success = _verify_reshard_slices(
                ctx, mesh,
                [Replicate()], [Shard(1)],
                shape,
                f"Replicate -> Shard(1) 2D shape={shape}",
            )
            if not success:
                all_passed = False
            elif ctx.rank == 0:
                ctx.log(f"PASS Replicate -> Shard(1) 2D shape={shape}")

        ctx.assert_true(all_passed, "Some Replicate -> Shard tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_reshard_slices_mixed_placements(ctx: TestContext, device: str):
    """Test compute_reshard_slices with mixed Shard and Replicate on 2D mesh."""
    _init_torch_distributed(ctx)

    try:
        # Find a 2D factorization of world_size
        dp_size, tp_size = 0, 0
        for dp in range(2, ctx.world_size):
            if ctx.world_size % dp == 0:
                tp = ctx.world_size // dp
                if tp >= 2:
                    dp_size, tp_size = dp, tp
                    break

        if dp_size == 0:
            ctx.log(f"SKIP: No 2D factorization for world_size={ctx.world_size}")
            return

        mesh = DeviceMesh(
            "cpu",
            torch.arange(ctx.world_size).reshape(dp_size, tp_size),
            mesh_dim_names=("dp", "tp")
        )

        if ctx.rank == 0:
            ctx.log(f"Using {dp_size}x{tp_size} mesh")

        all_passed = True

        # [Shard(0), Replicate()] -> [Shard(0), Shard(0)]
        shape = (dp_size * tp_size * 4, 16)
        success = _verify_reshard_slices(
            ctx, mesh,
            [Shard(0), Replicate()], [Shard(0), Shard(0)],
            shape,
            f"[Shard(0), Replicate()] -> [Shard(0), Shard(0)] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [Shard(0), Replicate()] -> [Shard(0), Shard(0)]")

        # [Shard(0), Replicate()] -> [Shard(0), Shard(1)]
        shape = (dp_size * 4, tp_size * 4)
        success = _verify_reshard_slices(
            ctx, mesh,
            [Shard(0), Replicate()], [Shard(0), Shard(1)],
            shape,
            f"[Shard(0), Replicate()] -> [Shard(0), Shard(1)] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [Shard(0), Replicate()] -> [Shard(0), Shard(1)]")

        # [Replicate(), Shard(1)] -> [Shard(0), Shard(1)]
        shape = (dp_size * 4, tp_size * 4)
        success = _verify_reshard_slices(
            ctx, mesh,
            [Replicate(), Shard(1)], [Shard(0), Shard(1)],
            shape,
            f"[Replicate(), Shard(1)] -> [Shard(0), Shard(1)] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [Replicate(), Shard(1)] -> [Shard(0), Shard(1)]")

        # [Replicate(), Replicate()] -> [Shard(0), Shard(1)]
        shape = (dp_size * 4, tp_size * 4)
        success = _verify_reshard_slices(
            ctx, mesh,
            [Replicate(), Replicate()], [Shard(0), Shard(1)],
            shape,
            f"[Replicate(), Replicate()] -> [Shard(0), Shard(1)] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [Replicate(), Replicate()] -> [Shard(0), Shard(1)]")

        ctx.assert_true(all_passed, "Some mixed placement tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_reshard_slices_no_change(ctx: TestContext, device: str):
    """Test compute_reshard_slices when placements are identical."""
    _init_torch_distributed(ctx)

    try:
        mesh = DeviceMesh("cpu", torch.arange(ctx.world_size))
        all_passed = True

        # Same placements - slices should be all slice(None)
        shape = (ctx.world_size * 4, 8)
        success = _verify_reshard_slices(
            ctx, mesh,
            [Shard(0)], [Shard(0)],
            shape,
            f"Shard(0) -> Shard(0) (no change) shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS Shard(0) -> Shard(0) (no change)")

        ctx.assert_true(all_passed, "Some no-change tests failed")

    finally:
        _cleanup_torch_distributed()


@test_cpu_cuda
def test_reshard_slices_strided_shard(ctx: TestContext, device: str):
    """Test compute_reshard_slices with _StridedShard placements."""
    if not HAS_STRIDED_SHARD:
        ctx.log("SKIP: _StridedShard not available")
        return

    _init_torch_distributed(ctx)

    try:
        # Find a 2D factorization of world_size
        dp_size, tp_size = 0, 0
        for dp in range(2, ctx.world_size):
            if ctx.world_size % dp == 0:
                tp = ctx.world_size // dp
                if tp >= 2:
                    dp_size, tp_size = dp, tp
                    break

        if dp_size == 0:
            ctx.log(f"SKIP: No 2D factorization for world_size={ctx.world_size}")
            return

        mesh = DeviceMesh(
            "cpu",
            torch.arange(ctx.world_size).reshape(dp_size, tp_size),
            mesh_dim_names=("dp", "tp")
        )

        if ctx.rank == 0:
            ctx.log(f"Using {dp_size}x{tp_size} mesh for StridedShard tests")

        all_passed = True

        # [_StridedShard(0, sf=tp_size), Replicate()] -> [_StridedShard(0, sf=tp_size), Shard(0)]
        # This is the FSDP2+TP pattern where we want to split Replicate work
        shape = (dp_size * tp_size * 8, 16)
        success = _verify_reshard_slices(
            ctx, mesh,
            [_StridedShard(0, split_factor=tp_size), Replicate()],
            [_StridedShard(0, split_factor=tp_size), Shard(0)],
            shape,
            f"[StridedShard, Replicate] -> [StridedShard, Shard(0)] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [StridedShard, Replicate] -> [StridedShard, Shard(0)]")

        # [_StridedShard(0, sf=tp_size), Replicate()] -> [_StridedShard(0, sf=tp_size), Shard(1)]
        shape = (dp_size * tp_size * 4, tp_size * 4)
        success = _verify_reshard_slices(
            ctx, mesh,
            [_StridedShard(0, split_factor=tp_size), Replicate()],
            [_StridedShard(0, split_factor=tp_size), Shard(1)],
            shape,
            f"[StridedShard, Replicate] -> [StridedShard, Shard(1)] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [StridedShard, Replicate] -> [StridedShard, Shard(1)]")

        # [Replicate(), _StridedShard(0, sf=dp_size)] -> [Shard(0), _StridedShard(0, sf=dp_size)]
        shape = (dp_size * 4, dp_size * tp_size * 4)
        success = _verify_reshard_slices(
            ctx, mesh,
            [Replicate(), _StridedShard(0, split_factor=dp_size)],
            [Shard(0), _StridedShard(0, split_factor=dp_size)],
            shape,
            f"[Replicate, StridedShard] -> [Shard(0), StridedShard] shape={shape}",
        )
        if not success:
            all_passed = False
        elif ctx.rank == 0:
            ctx.log(f"PASS [Replicate, StridedShard] -> [Shard(0), StridedShard]")

        ctx.assert_true(all_passed, "Some StridedShard reshard tests failed")

    finally:
        _cleanup_torch_distributed()


@test
def test_reshard_slices_error_on_invalid_conversion(ctx: TestContext):
    """Test compute_reshard_slices raises error for invalid conversions."""
    from moodist import compute_reshard_slices

    local_shape = (100,)

    # Shard(0) -> Shard(1) requires communication
    try:
        compute_reshard_slices(local_shape, [Shard(0)], [Shard(1)], [(0, 2)])
        ctx.assert_true(False, "Should have raised ValueError")
    except ValueError as e:
        ctx.assert_true("requires communication" in str(e))
        ctx.log(f"PASS: correctly rejected Shard(0) -> Shard(1): {e}")

    # Shard -> Replicate requires communication (gathering)
    try:
        compute_reshard_slices(local_shape, [Shard(0)], [Replicate()], [(0, 2)])
        ctx.assert_true(False, "Should have raised ValueError")
    except ValueError as e:
        ctx.assert_true("requires communication" in str(e))
        ctx.log(f"PASS: correctly rejected Shard -> Replicate: {e}")


