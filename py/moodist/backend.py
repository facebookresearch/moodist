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
    """Create a MoodistProcessGroup and register it by name."""
    if MoodistProcessGroup is None:
        raise RuntimeError("MoodistProcessGroup not available in this build")
    obj = MoodistProcessGroup(store, rank, size)
    _name_to_group[obj.moodist_name()] = obj
    return obj


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
        "moodist", create_moodist_backend, devices=("cpu", "cuda")
    )

if TcpStore is not None:
    torch.distributed.distributed_c10d.register_rendezvous_handler(
        "moodist", rendezvous_handler
    )
