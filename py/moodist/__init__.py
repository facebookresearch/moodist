# Copyright (c) Facebook, Inc. and its affiliates.
from ._C import (
    MoodistProcessGroup
)

import torch.distributed



class MyProcessGroup(MoodistProcessGroup):
    def __init__(self, store, rank, size, timeout):
        super().__init__(rank, size)
        if rank == 0:
            store.set("moodist_distributed_rank0_address", self.get_address())
        self.init(store.get("moodist_distributed_rank0_address"))



class ProcessGroup(torch.distributed.ProcessGroup):
    def __init__(self, store, rank, size, timeout):
        super().__init__(rank, size)
        self.nccl = torch.distributed.ProcessGroupNCCL(store, rank, size, timeout)
        self.moodist = MyProcessGroup(store, rank, size, timeout)

    def _get_backend_name(self):
        return "moodist-nccl"

    def broadcast(self, *args, **kwargs):
        return self.nccl.broadcast(*args, **kwargs)

    def allreduce(self, *args, **kwargs):
        return self.nccl.allreduce(*args, **kwargs)

    def allreduce_coalesced(self, *args, **kwargs):
        return self.nccl.allreduce_coalesced(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        return self.nccl.reduce(*args, **kwargs)

    def allgather(self, *args, **kwargs):
        return self.moodist.allgather(*args, **kwargs)

    def _allgather_base(self, *args, **kwargs):
        return self.moodist._allgather_base(*args, **kwargs)

    def allgather_coalesced(self, *args, **kwargs):
        assert False
        return self.nccl.allgather_coalesced(*args, **kwargs)

    def gather(self, *args, **kwargs):
        return self.nccl.gather(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        return self.nccl.scatter(*args, **kwargs)

    def reduce_scatter(self, *args, **kwargs):
        return self.nccl.reduce_scatter(*args, **kwargs)

    def _reduce_scatter_base(self, *args, **kwargs):
        return self.nccl._reduce_scatter_base(*args, **kwargs)

    def alltoall_base(self, *args, **kwargs):
        return self.nccl.alltoall_base(*args, **kwargs)

    def alltoall(self, *args, **kwargs):
        return self.nccl.alltoall(*args, **kwargs)

    def send(self, *args, **kwargs):
        return self.nccl.send(*args, **kwargs)

    def recv(self, *args, **kwargs):
        return self.nccl.recv(*args, **kwargs)

    def recv_anysource(self, *args, **kwargs):
        return self.nccl.recv_anysource(*args, **kwargs)

    def barrier(self, *args, **kwargs):
        return self.nccl.barrier(*args, **kwargs)



from datetime import timedelta


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    return ProcessGroup(store, rank, size, timeout)


torch.distributed.Backend.register_backend("moodist", create_moodist_backend)


__all__ = []
