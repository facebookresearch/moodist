# Copyright (c) Meta Platforms, Inc. and affiliates.

import weakref
from datetime import timedelta
from queue import Empty
from typing import TYPE_CHECKING, Any
import importlib

import torch
import torch.distributed

from .version import __version__, cversions

clist = [
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
]

torchversion = torch.__version__

found = False
for k, v in cversions.items():
    if torchversion.startswith(k):
        assert not found, "Moodist matched multiple pytorch versions? %s %s" % (
            torchversion,
            list(cversions.keys()),
        )
        _C = importlib.import_module(v, "moodist")
        found = True

if not found:
    raise RuntimeError(
        "Moodist was not built for the currently installed pytorch version."
        " Found pytorch %s. Moodist was built for: %s"
        % (torchversion, list(cversions.keys()))
    )

for n in clist:
    globals()[n] = getattr(_C, n)

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
        process_group: MoodistProcessGroup | str,
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


_name_to_group = weakref.WeakValueDictionary()


def find_process_group(name: str):
    return _name_to_group.get(name, None)


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    obj = MoodistProcessGroup(store, rank, size)
    _name_to_group[obj.moodist_name()] = obj
    return obj


def rendezvous_handler(
    url, timeout: timedelta = torch.distributed.distributed_c10d.default_pg_timeout
):
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


torch.distributed.Backend.register_backend(
    "moodist", create_moodist_backend, devices=("cpu", "cuda")
)

torch.distributed.distributed_c10d.register_rendezvous_handler(
    "moodist", rendezvous_handler
)


def compile_op(group, shape=None, dtype=None, inputs=None, outputs=None):
    from .compile import compile_op

    return compile_op(group, shape=shape, dtype=dtype, inputs=inputs, outputs=outputs)


__all__ = [*clist, "create_moodist_backend", "Empty", "compile_op"]
