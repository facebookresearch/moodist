# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Communication pools: CUDA buffers at the same place of a process group's multicast objects (one per NVLink domain)
on every rank."""

import math

import torch
import torch.distributed

from . import _core


def _group(group):
    if group is None:
        group = torch.distributed.group.WORLD
    if group is None:
        raise RuntimeError("moodist: no process group (call torch.distributed.init_process_group first)")
    return group


def comm_empty(*size, dtype=None, group=None):
    """An uninitialized CUDA tensor from the communication pool of a moodist process group.

    Each process group has its own pool. When every rank of the group allocates (and frees) its communication
    buffers in the same order, with the same sizes, each buffer gets the same offset in the pool on every rank,
    whatever else the ranks allocate and whenever their GPUs finish with freed memory (out of device memory, an
    allocation raises rather than take memory freed on other streams). compile_op ops whose outputs are such buffers
    store to all ranks at once with NVLink multicast (NVLS) where the group supports it: an all-gather, for example,
    then needs one SM instead of a dozen. Reductions whose inputs (and for an all-reduce, outputs) are such buffers
    have the NVLink switches combine them (an all-reduce then moves about half the data), within each NVLink domain of
    the group. Launches whose buffers are not in the pool store to each rank (and read each input) as usual. The
    ranks must agree: a launch whose buffers are in the pool on some ranks only, or at different offsets on different
    ranks, fails (its kernel reports the error and stops, which fails the CUDA context).

    The memory comes from moodist's CUDA allocator (in large slabs the pool keeps), and is usable on the current
    stream, like torch.empty's. The process group's allocate_tensor (which FSDP2 uses after
    set_allocate_memory_from_process_group_for_comm(True)) allocates from the same pool. The process group's own
    collectives of CUDA tensors run as compile_op ops, so they use NVLS on such buffers too, and need no copies of them
    (other ranks reach moodist's memory only: the collectives copy every tensor of another allocator, e.g. PyTorch's,
    through such buffers, which the pool keeps the memory of).

    Args:
        size: The shape (ints, or one tuple / list / torch.Size).
        dtype: The element type (default: torch.get_default_dtype()).
        group: The moodist process group (default: the default group).
    """
    if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
        size = tuple(size[0])
    if dtype is None:
        dtype = torch.get_default_dtype()
    device = torch.device("cuda", torch.cuda.current_device())
    backend = _group(group)._get_backend(device)
    numel = math.prod(int(s) for s in size)
    return backend.allocate_tensor(numel, dtype=dtype, device=device).view(size)


def comm_pool_offset(tensor):
    """The offset of a tensor's memory in the communication pool it is in (see comm_empty; a buffer is in at most one
    group's pool), or -1 if it is in none. Buffers at the same offset of their group's pool on every rank of the group
    can use NVLink multicast."""
    return _core.comm_pool_offset(tensor)
