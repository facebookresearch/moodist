# Copyright (c) Facebook, Inc. and its affiliates.
import torch.distributed
from ._C import MoodistProcessGroup

from datetime import timedelta


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    return MoodistProcessGroup(store, rank, size)


torch.distributed.Backend.register_backend("moodist", create_moodist_backend, devices=("cpu", "cuda"))


def enable_profiling(b):
    _C.enable_profiling(b)


__all__ = []
