# Copyright (c) Meta Platforms, Inc. and affiliates.

"""PyTorch distributed backend registration and process group management."""

import weakref
from datetime import timedelta

import torch
import torch.distributed

from ._core import MoodistProcessGroup, TcpStore


_name_to_group = weakref.WeakValueDictionary()


def find_process_group(name: str):
    """Find a MoodistProcessGroup by its name."""
    return _name_to_group.get(name, None)


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    """Create a MoodistProcessGroup and register it by name."""
    obj = MoodistProcessGroup(store, rank, size)
    _name_to_group[obj.moodist_name()] = obj
    return obj


def rendezvous_handler(
    url, timeout: timedelta = torch.distributed.distributed_c10d.default_pg_timeout
):
    """Handle moodist:// rendezvous URLs for torch.distributed.init_process_group."""
    import urllib.parse

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


# Register backend with PyTorch distributed
torch.distributed.Backend.register_backend(
    "moodist", create_moodist_backend, devices=("cpu", "cuda")
)

torch.distributed.distributed_c10d.register_rendezvous_handler(
    "moodist", rendezvous_handler
)
