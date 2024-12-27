# Copyright (c) Facebook, Inc. and its affiliates.
import torch.distributed
from ._C import (
    MoodistProcessGroup,
    enable_profiling,
    enable_cuda_allocator,
)

from datetime import timedelta

__version__ = "0.1.2"


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    return MoodistProcessGroup(store, rank, size)


torch.distributed.Backend.register_backend(
    "moodist", create_moodist_backend, devices=("cpu", "cuda")
)

__all__ = [
    "MoodistProcessGroup",
    "enable_profiling",
    "enable_cuda_allocator",
    "create_moodist_backend",
]
