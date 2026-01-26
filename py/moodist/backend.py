# Copyright (c) Meta Platforms, Inc. and affiliates.

"""PyTorch distributed backend registration and process group management."""

import weakref
from datetime import timedelta

import torch
import torch.distributed

from ._core import MoodistProcessGroup, TcpStore
from .options import MoodistOptions, MoodistOptionsContext


_name_to_group = weakref.WeakValueDictionary()


class PreferKernelLessContext:
    """Context manager for temporarily setting prefer_kernel_less on a ProcessGroup.

    Usage:
        pg = moodist.MoodistProcessGroup(store, rank, size)

        # As context manager (auto-restores after):
        with pg.prefer_kernel_less(True):
            pg.allgather(...)

        # Or just call to set directly (returns context manager you can ignore):
        pg.prefer_kernel_less(True)
    """

    def __init__(self, pg, value: bool):
        self.pg = pg
        self.new_value = value
        self.old_value = pg.get_prefer_kernel_less()
        pg.set_prefer_kernel_less(value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pg.set_prefer_kernel_less(self.old_value)
        return False


def prefer_kernel_less(pg, value: bool):
    """Set prefer_kernel_less on a ProcessGroup, returning a context manager.

    Can be used either as a direct setter or as a context manager:

        # Direct set (ignoring the return value):
        prefer_kernel_less(pg, True)

        # As context manager (restores after):
        with prefer_kernel_less(pg, True):
            ...
    """
    return PreferKernelLessContext(pg, value)


# Monkey-patch prefer_kernel_less onto MoodistProcessGroup if available
# Note: options() method is defined in pybind.cc
if MoodistProcessGroup is not None:
    MoodistProcessGroup.prefer_kernel_less = lambda self, value: PreferKernelLessContext(self, value)


def find_process_group(name: str):
    """Find a MoodistProcessGroup by its name."""
    return _name_to_group.get(name, None)


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    """Create a MoodistProcessGroup directly.

    This is the public API for creating standalone MoodistProcessGroups that
    are not managed by PyTorch's distributed module. These groups do not need
    to be registered in torch.distributed's _world state.

    Args:
        store: A torch.distributed.Store for coordination
        rank: This process's rank within the group
        size: Total number of processes in the group
        timeout: Timeout for operations

    Returns:
        A MoodistProcessGroup instance with custom moodist methods available.
    """
    if MoodistProcessGroup is None:
        raise RuntimeError("MoodistProcessGroup not available in this build")

    moodist_obj = MoodistProcessGroup(store, rank, size)
    _name_to_group[moodist_obj.moodist_name()] = moodist_obj
    return moodist_obj


def _create_pytorch_backend(dist_backend_opts, backend_options):
    """Create a MoodistProcessGroup for PyTorch's distributed module.

    This is an internal function used by torch.distributed.Backend.register_backend.
    It uses the extended API to receive global_ranks_in_group, which is needed to
    correctly register subgroups created by new_group() or DeviceMesh.

    This function also pre-registers the pybind11 wrapper in PyTorch's _world
    so that DeviceMesh lookups via _resolve_process_group succeed.
    """
    if MoodistProcessGroup is None:
        raise RuntimeError("MoodistProcessGroup not available in this build")

    from torch._C._distributed_c10d import (
        _register_process_group,
        _resolve_process_group,
        _unregister_process_group,
    )
    from torch.distributed import distributed_c10d as c10d

    # Extract fields from the extended API options
    store = dist_backend_opts.store
    rank = dist_backend_opts.group_rank
    size = dist_backend_opts.group_size
    global_ranks = list(dist_backend_opts.global_ranks_in_group)

    # Create the moodist ProcessGroup
    moodist_obj = MoodistProcessGroup(store, rank, size)
    temp_name = f"__moodist_temp_{id(moodist_obj)}"

    # Register temporarily to cache pybind11 wrapper in PyTorch's instance registry.
    # This ensures future _resolve_process_group calls return a consistent wrapper.
    _register_process_group(temp_name, moodist_obj)
    wrapper = _resolve_process_group(temp_name)
    _unregister_process_group(temp_name)

    # Pre-register the wrapper in key _world dicts.
    # This ensures that get_backend(), get_rank(), functional collectives, etc. work.
    # Note: We use a temp group_name here - PyTorch will overwrite it later with the real one.
    #
    # For subgroups (new_group, DeviceMesh), global_ranks contains the global ranks
    # participating in this group. We need to map global_rank -> group_rank.
    # For the default group (init_process_group), global_ranks is empty and we use
    # identity mapping.
    if global_ranks:
        rank_mapping = {global_rank: group_rank for group_rank, global_rank in enumerate(global_ranks)}
    else:
        rank_mapping = {i: i for i in range(size)}
    temp_group_name = f"__temp_moodist_{id(wrapper)}"

    c10d._world.pg_map[wrapper] = ("moodist", store)
    c10d._world.pg_names[wrapper] = temp_group_name
    c10d._world.pg_group_ranks[wrapper] = rank_mapping
    c10d._world.pg_backend_config[wrapper] = "moodist"

    # Register cleanup callback to remove wrapper when moodist_obj is destroyed.
    def cleanup_wrapper():
        c10d._world.pg_map.pop(wrapper, None)
        c10d._world.pg_names.pop(wrapper, None)
        c10d._world.pg_group_ranks.pop(wrapper, None)
        c10d._world.pg_backend_config.pop(wrapper, None)
        # Tags are managed by PyTorch, skip for wrapper

    weakref.finalize(moodist_obj, cleanup_wrapper)

    _name_to_group[moodist_obj.moodist_name()] = moodist_obj
    return moodist_obj


def rendezvous_handler(
    url, timeout: timedelta = torch.distributed.distributed_c10d.default_pg_timeout
):
    """Handle moodist:// rendezvous URLs for torch.distributed.init_process_group."""
    import urllib.parse

    if TcpStore is None:
        raise RuntimeError("TcpStore not available in this build")

    result = urllib.parse.urlparse(url)
    if result.hostname is None:
        raise ValueError(f"Moodist rendezvous URL missing hostname: {url}")
    if result.port is None:
        raise ValueError(f"Moodist rendezvous URL missing port: {url}")
    query = urllib.parse.parse_qs(result.query)
    if "rank" not in query:
        raise ValueError(f"Moodist rendezvous URL missing 'rank' query parameter: {url}")
    if "world_size" not in query:
        raise ValueError(f"Moodist rendezvous URL missing 'world_size' query parameter: {url}")

    world_size = int(query["world_size"][0])
    rank = int(query["rank"][0])

    yield (
        TcpStore(result.hostname, result.port, "foo", world_size, rank, timeout),
        rank,
        world_size,
    )


# Register backend with PyTorch distributed (only if MoodistProcessGroup is available)
if MoodistProcessGroup is not None:
    torch.distributed.Backend.register_backend(
        "moodist", _create_pytorch_backend, extended_api=True, devices=("cpu", "cuda")
    )

if TcpStore is not None:
    torch.distributed.distributed_c10d.register_rendezvous_handler(
        "moodist", rendezvous_handler
    )
