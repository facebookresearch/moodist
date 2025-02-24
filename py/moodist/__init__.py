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
    cache,
)

from datetime import timedelta
import pickle
from queue import Empty

__version__ = "0.1.4"


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


class LocalQueue:
    def __init__(self, process_group):
        self.queues = [
            Queue(process_group, location=i) for i in range(process_group.size())
        ]
        self.rank = process_group.rank()

    def put_tensor(self, tensor, location=None):
        return self.queues[self.rank if location is None else location].put_tensor(
            tensor
        )

    def get_tensor(self, location=None, block=True, timeout=None):
        return self.queues[self.rank if location is None else location].get_tensor(
            block, timeout
        )

    def put_object(self, object, location=None):
        return self.queues[self.rank if location is None else location].put_object(
            object
        )

    def get_object(self, location=None, block=True, timeout=None):
        return self.queues[self.rank if location is None else location].get_object(
            block, timeout
        )

    def qsize(self, location=None):
        return self.queues[self.rank if location is None else location].qsize()

    def empty(self, location=None):
        return self.queues[self.rank if location is None else location].empty()

    def wait(self, location=None, timeout=None):
        return self.queues[self.rank if location is None else location].wait(timeout)


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    return MoodistBackend(store, rank, size)


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
    "cache",
]
