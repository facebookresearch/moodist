# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Moodist: High-performance distributed communication library."""

from datetime import timedelta
from queue import Empty
from typing import TYPE_CHECKING, Any

import torch

from .version import __version__

# C extension
from ._core import (
    MoodistProcessGroup,
    MoodistBackend,
    enable_profiling,
    enable_cuda_allocator,
    enable_cpu_allocator,
    cpu_allocator_debug,
    cuda_copy,
    set_prefer_kernel_less,
    TcpStore,
    serialize,
    deserialize,
)

# Backend (importing triggers registration with torch.distributed)
from .backend import find_process_group, create_moodist_backend, PreferKernelLessContext

# Options (separate module to avoid circular imports)
from .options import MoodistOptions, MoodistOptionsContext

# Queue
from .queue import Queue, TransactionContextManager

# Compile op
from .compile import compile_op


if TYPE_CHECKING:

    class MoodistProcessGroup(torch.distributed.ProcessGroup): ...

    class TcpStore(torch.distributed.Store):
        def __init__(
            self,
            hostname: str,
            port: int,
            key: str,
            world_size: int,
            rank: int,
            timeout: timedelta,
        ): ...

    def serialize(x: object) -> torch.Tensor: ...
    def deserialize(x: torch.Tensor) -> Any: ...

    def cuda_copy(dst: torch.Tensor, src: torch.Tensor) -> None: ...


__all__ = [
    # Version
    "__version__",
    # C extension
    "MoodistProcessGroup",
    "MoodistBackend",
    "enable_profiling",
    "enable_cuda_allocator",
    "enable_cpu_allocator",
    "cpu_allocator_debug",
    "cuda_copy",
    "set_prefer_kernel_less",
    "TcpStore",
    "serialize",
    "deserialize",
    # Queue
    "Queue",
    "TransactionContextManager",
    "Empty",
    # Backend
    "find_process_group",
    "create_moodist_backend",
    "PreferKernelLessContext",
    "MoodistOptions",
    "MoodistOptionsContext",
    # Compile
    "compile_op",
]
