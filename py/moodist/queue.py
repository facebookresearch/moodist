# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Queue and transaction management."""

from queue import Empty

from ._core import serialize, deserialize
from .backend import find_process_group


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
        return self.queue.put_tensor(tensor, transaction=self.id)

    def put_object(self, object):
        return self.queue.put_object(object, transaction=self.id)


class Queue:
    def __init__(
        self,
        process_group,
        location,
        streaming=False,
        name=None,
    ):
        if isinstance(process_group, str):
            pg_name = process_group
            process_group = find_process_group(pg_name)
            if process_group is None:
                raise ValueError(
                    "The Moodist process group by name '%s' could not be found" % pg_name
                )
        if not hasattr(process_group, "Queue"):
            raise RuntimeError(
                "moodist.Queue process_group parameter must be a MoodistProcessGroup, but got %s"
                % str(type(process_group)),
            )
        self.impl = process_group.Queue(
            location=location, streaming=streaming, name=name
        )
        self.process_group_name = process_group.moodist_name()
        self.location = location
        self.streaming = streaming

    def __reduce__(self):
        return type(self), (
            self.process_group_name,
            self.location,
            self.streaming,
            self.impl.name(),
        )

    def put_tensor(self, tensor, *, transaction=0):
        return self.impl.put(tensor, transaction, True)  # waitOnDestroy=True

    def get_tensor(self, block=True, timeout=None, return_size=False):
        r, size = self.impl.get(block=block, timeout=timeout)
        if r is None:
            raise Empty
        if return_size:
            return r, size
        else:
            return r

    def put_object(self, object, *, transaction=0):
        return self.impl.put(serialize(object), transaction, False)  # waitOnDestroy=False

    def get_object(self, block=True, timeout=None, return_size=False):
        if return_size:
            tensor, size = self.get_tensor(
                block=block, timeout=timeout, return_size=True
            )
            return deserialize(tensor), size
        return deserialize(self.get_tensor(block=block, timeout=timeout))

    def qsize(self):
        return self.impl.qsize()

    def empty(self):
        return self.impl.qsize() == 0

    def wait(self, timeout=None):
        return self.impl.wait(timeout=timeout)

    def transaction(self):
        return TransactionContextManager(self)
