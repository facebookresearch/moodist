# Copyright (c) Facebook, Inc. and its affiliates.
import torch.distributed
from ._C import (
    MoodistProcessGroup,
    MoodistBackend,
    enable_profiling,
    enable_cuda_allocator,
    enable_cpu_allocator,
    cpu_allocator_debug,
    cuda_copy,
    set_prefer_kernel_less,
)
from .version import __version__

from datetime import timedelta
import pickle
from queue import Empty

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    class MoodistProcessGroup(torch.distributed.ProcessGroup): ...

class TransactionContextManager:
    def __init__(self, queue):
        self.queue = queue

    def __enter__(self):
        self.id = self.queue.impl.transaction_begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.queue.impl.transaction_cancel(self.id)
        else:
            self.queue.impl.transaction_commit(self.id)

    def put_tensor(self, tensor):
        return self.queue.put_tensor(tensor, self.id)

    def put_object(self, object):
        return self.queue.put_object(object, self.id)


class Queue:
    def __init__(self, process_group, location):
        if not hasattr(process_group, "Queue"):
            raise RuntimeError(
                "moodist.Queue process_group parameter must be a MoodistProcessGroup, but got %s"
                % str(type(process_group)),
            )
        self.impl = process_group.Queue(location=location)

    def put_tensor(self, tensor, transaction=0):
        return self.impl.put(tensor, transaction)

    def get_tensor(self, block=True, timeout=None):
        r = self.impl.get(block=block, timeout=timeout)
        if r is None:
            raise Empty
        return r

    def put_object(self, object, transaction=0):
        return self.impl.put(
            torch.frombuffer(pickle.dumps(object), dtype=torch.uint8), transaction
        )

    def get_object(self, block=True, timeout=None):
        return pickle.loads(
            self.get_tensor(block=block, timeout=timeout).numpy().tobytes()
        )

    def qsize(self):
        return self.impl.qsize()

    def empty(self):
        return self.impl.qsize() == 0

    def wait(self, timeout=None):
        return self.impl.wait(timeout=timeout)

    def transaction(self):
        return TransactionContextManager(self)


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    return MoodistProcessGroup(store, rank, size)


torch.distributed.Backend.register_backend(
    "moodist", create_moodist_backend, devices=("cpu", "cuda")
)

__all__ = [
    "MoodistProcessGroup",
    "MoodistBackend",
    "enable_profiling",
    "enable_cuda_allocator",
    "enable_cpu_allocator",
    "cpu_allocator_debug",
    "create_moodist_backend",
    "Empty",
    "cuda_copy",
    "set_prefer_kernel_less",
]
