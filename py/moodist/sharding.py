# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Sharding utilities for computing DTensor chunk ownership.

This module provides functions to compute which chunks of a tensor
a given owner (identified by mesh coordinates) owns, based on the
tensor's placement strategy.
"""

from dataclasses import dataclass
from itertools import product
from typing import List, Tuple

from torch.distributed.tensor.placement_types import Shard, Replicate


@dataclass
class ShardInfo:
    """Information about a single shard/chunk of a tensor.

    Attributes:
        global_offset: Position in the global tensor (list of ints, one per dimension).
        local_offset: Position in the local tensor (list of ints, one per dimension).
        shape: Shape of this chunk (list of ints, one per dimension).
    """
    global_offset: List[int]
    local_offset: List[int]
    shape: List[int]


def _compose_ranges(existing_ranges, local_start, local_size):
    """
    Given existing ranges that map local positions to global positions,
    compute the global ranges for local[local_start : local_start + local_size].

    Args:
        existing_ranges: list of (global_offset, size) tuples
        local_start: start position in the local (concatenated) view
        local_size: size of the slice in the local view

    Returns:
        list of (global_offset, size) tuples
    """
    result = []
    local_pos = 0
    local_end = local_start + local_size

    for global_offset, range_size in existing_ranges:
        range_local_start = local_pos
        range_local_end = local_pos + range_size

        # Check overlap with [local_start, local_end)
        overlap_start = max(range_local_start, local_start)
        overlap_end = min(range_local_end, local_end)

        if overlap_start < overlap_end:
            offset_within_range = overlap_start - range_local_start
            overlap_size = overlap_end - overlap_start
            result.append((global_offset + offset_within_range, overlap_size))

        local_pos += range_size
        if local_pos >= local_end:
            break

    return result


def _strided_shard_ranges(dim_size, split_factor, group_size, rank_index):
    """Compute the ranges for _StridedShard (before composition)."""
    virtual_size = split_factor * group_size
    chunk_size = (dim_size + virtual_size - 1) // virtual_size
    num_chunks = (dim_size + chunk_size - 1) // chunk_size

    ranges = []
    chunk_idx = rank_index
    while chunk_idx < num_chunks:
        offset = chunk_idx * chunk_size
        size = min(chunk_size, dim_size - offset)
        if size > 0:
            ranges.append((offset, size))
        chunk_idx += group_size

    return ranges


def _regular_shard_ranges(dim_size, group_size, rank_index):
    """Compute the range for regular Shard."""
    chunk_size = (dim_size + group_size - 1) // group_size
    offset = min(chunk_size * rank_index, dim_size)
    size = min(chunk_size, dim_size - offset)
    if size > 0:
        return [(offset, size)]
    return []


def compute_shards(shape, placements, indices_and_sizes) -> List[ShardInfo]:
    """
    Compute the chunks an owner has given the tensor shape and placements.

    This function handles regular Shard, Replicate, and _StridedShard placements,
    including composition when multiple placements shard the same dimension.

    Args:
        shape: Global tensor shape as a tuple/list of integers.
        placements: List of PyTorch placement objects (Shard, Replicate, _StridedShard).
        indices_and_sizes: List of (index, group_size) tuples, one per placement.
            - index: The owner's position within this mesh dimension (0 to group_size-1)
            - group_size: The size of this mesh dimension

    Returns:
        List of ShardInfo objects, each containing:
            - global_offset: Position in global tensor
            - local_offset: Position in local tensor
            - shape: Shape of this chunk

    Example:
        >>> from torch.distributed.tensor.placement_types import Shard
        >>> # 2D mesh (4x2), rank at position (1, 0)
        >>> shape = (128, 64)
        >>> placements = [Shard(0), Shard(1)]
        >>> indices_and_sizes = [(1, 4), (0, 2)]
        >>> chunks = compute_shards(shape, placements, indices_and_sizes)
    """
    ndim = len(shape)

    if len(placements) != len(indices_and_sizes):
        raise ValueError(
            f"placements and indices_and_sizes must have same length, "
            f"got {len(placements)} and {len(indices_and_sizes)}"
        )

    # For each dimension, track current ranges: list of (global_offset, size)
    # Initially each dimension has one range covering full extent
    dim_ranges = {d: [(0, shape[d])] for d in range(ndim)}

    for p, (rank_index, group_size) in zip(placements, indices_and_sizes):
        if isinstance(p, Replicate):
            continue

        dim = p.dim
        current_ranges = dim_ranges[dim]

        # Total local size is sum of current range sizes
        total_local_size = sum(size for _, size in current_ranges)

        if total_local_size == 0:
            continue

        if hasattr(p, 'split_factor'):
            # _StridedShard
            sf = p.split_factor

            # Compute strided ranges in the "local" view
            strided = _strided_shard_ranges(total_local_size, sf, group_size, rank_index)

            # Map through existing ranges to get global ranges
            new_ranges = []
            for local_offset, local_size in strided:
                mapped = _compose_ranges(current_ranges, local_offset, local_size)
                new_ranges.extend(mapped)

            dim_ranges[dim] = new_ranges

        elif isinstance(p, Shard):
            # Regular Shard
            shard = _regular_shard_ranges(total_local_size, group_size, rank_index)

            if shard:
                local_offset, local_size = shard[0]
                dim_ranges[dim] = _compose_ranges(current_ranges, local_offset, local_size)
            else:
                dim_ranges[dim] = []

    # Cartesian product of ranges across dimensions = rectangular chunks
    all_dim_ranges = [dim_ranges.get(d, [(0, shape[d])]) for d in range(ndim)]

    # Handle empty dimensions
    for ranges in all_dim_ranges:
        if not ranges:
            return []

    # Precompute local offsets for each range in each dimension
    # local_offset[d][i] = sum of sizes of ranges 0..i-1 on dimension d
    dim_local_offsets = []
    for d in range(ndim):
        offsets = []
        cumsum = 0
        for _, size in all_dim_ranges[d]:
            offsets.append(cumsum)
            cumsum += size
        dim_local_offsets.append(offsets)

    chunks = []
    for combo in product(*[enumerate(ranges) for ranges in all_dim_ranges]):
        # combo is list of (range_idx, (global_offset, size))
        global_offset = [r[1][0] for r in combo]
        local_offset = [dim_local_offsets[d][r[0]] for d, r in enumerate(combo)]
        chunk_shape = [r[1][1] for r in combo]
        if all(s > 0 for s in chunk_shape):
            chunks.append(ShardInfo(global_offset, local_offset, chunk_shape))

    return chunks


def dtensor_shards(dtensor, coord=None) -> List[ShardInfo]:
    """
    Compute chunks for a coordinate in a DTensor's mesh.

    This is a convenience wrapper around compute_shards for DTensors.

    Args:
        dtensor: A PyTorch DTensor.
        coord: Mesh coordinate tuple, e.g., (2, 1) for a 2D mesh.
            If None, uses the current process's coordinate in this mesh.

    Returns:
        List of ShardInfo objects. Returns empty list if coord is None
        and current process is not in mesh.

    Example:
        >>> # Get chunks for current process
        >>> chunks = dtensor_shards(my_dtensor)
        >>> # Get chunks for a specific mesh coordinate
        >>> chunks = dtensor_shards(my_dtensor, coord=(0, 1))
    """
    mesh = dtensor.device_mesh

    if coord is None:
        coord = mesh.get_coordinate()
        if coord is None:
            return []  # This rank not in mesh

    if len(coord) != mesh.ndim:
        raise ValueError(
            f"coord must have {mesh.ndim} dimensions for this mesh, got {len(coord)}"
        )

    indices_and_sizes = [(coord[i], mesh.size(i)) for i in range(mesh.ndim)]

    return compute_shards(
        tuple(dtensor.shape),
        list(dtensor.placements),
        indices_and_sizes
    )
