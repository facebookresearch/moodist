# Copyright (c) Facebook, Inc. and its affiliates.
from ._C import MoodistProcessGroup

import torch.distributed


class MyProcessGroup(MoodistProcessGroup):
    def __init__(self, store, rank, size, timeout):
        super().__init__(rank, size)
        if rank == 0:
            store.set("moodist_rank0_address", self.get_address())
        self.init(store.get("moodist_rank0_address"))


def find_tensors(tup):
    s = ""
    for v in tup:
        if s != "":
            s += " "
        if isinstance(v, list) or isinstance(v, tuple):
            s += find_tensors(v)
        elif isinstance(v, torch.Tensor):
            s += "(tensor numel %d itemsize %d ptr %#x)" % (v.numel(), v.element_size(), v.data_ptr())
        else:
            s += "?"
    return s


class ProcessGroup(torch.distributed.ProcessGroup):
    def __init__(self, store, rank, size, timeout):
        super().__init__(rank, size)
        self.moodist = MyProcessGroup(store, rank, size, timeout)
        self.nccl = torch.distributed.ProcessGroupNCCL(store, rank, size, timeout)

    def _get_backend_name(self):
        return "moodist-nccl"

    def broadcast(self, *args, **kwargs):
        print("size %d broadcast %s" % (self.size(), find_tensors(args)))
        return self.nccl.broadcast(*args, **kwargs)

    def allreduce(self, *args, **kwargs):
        #print("size %d allreduce %s" % (self.size(), find_tensors(args)))
        return self.nccl.allreduce(*args, **kwargs)

    def allreduce_coalesced(self, *args, **kwargs):
        print("size %d allreduce_coalesced %s" % (self.size(), find_tensors(args)))
        return self.nccl.allreduce_coalesced(*args, **kwargs)

    def reduce(self, *args, **kwargs):
        print("size %d reduce %s" % (self.size(), find_tensors(args)))
        return self.nccl.reduce(*args, **kwargs)

    def allgather(self, *args, **kwargs):
        #print("size %d allgather %s" % (self.size(), find_tensors(args)))
        return self.nccl.allgather(*args, **kwargs)
        # return self.moodist.allgather(*args, **kwargs)

    def _allgather_base(self, *args, **kwargs):
        # print("size %d _allgather_base %s" % (self.size(), find_tensors(args)))
        # return self.nccl._allgather_base(*args, **kwargs)
        return self.moodist._allgather_base(*args, **kwargs)

    def allgather_coalesced(self, *args, **kwargs):
        print("size %d allgather_coalesced %s" % (self.size(), find_tensors(args)))
        assert False
        return self.nccl.allgather_coalesced(*args, **kwargs)

    def gather(self, *args, **kwargs):
        print("size %d gather %s" % (self.size(), find_tensors(args)))
        return self.nccl.gather(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        print("size %d scatter %s" % (self.size(), find_tensors(args)))
        return self.nccl.scatter(*args, **kwargs)

    def reduce_scatter(self, *args, **kwargs):
        print("size %d size %d reduce_scatter %s" % (self.size(), find_tensors(args)))
        return self.nccl.reduce_scatter(*args, **kwargs)

    def _reduce_scatter_base(self, *args, **kwargs):
        # print("size %d _reduce_scatter_base %s" % (self.size(), find_tensors(args)))
        #return self.nccl._reduce_scatter_base(*args, **kwargs)
        return self.moodist._reduce_scatter_base(*args, **kwargs)

    def alltoall_base(self, *args, **kwargs):
        print("size %d alltoall_base %s" % (self.size(), find_tensors(args)))
        return self.nccl.alltoall_base(*args, **kwargs)

    def alltoall(self, *args, **kwargs):
        print("size %d alltoall %s" % (self.size(), find_tensors(args)))
        return self.nccl.alltoall(*args, **kwargs)

    def send(self, *args, **kwargs):
        print("size %d send %s" % (self.size(), find_tensors(args)))
        return self.nccl.send(*args, **kwargs)

    def recv(self, *args, **kwargs):
        print("size %d recv %s" % (self.size(), find_tensors(args)))
        return self.nccl.recv(*args, **kwargs)

    def recv_anysource(self, *args, **kwargs):
        print("size %d recv_anysource %s" % (self.size(), find_tensors(args)))
        return self.nccl.recv_anysource(*args, **kwargs)

    def barrier(self, *args, **kwargs):
        print("size %d barrier %s" % (self.size(), find_tensors(args)))
        return self.nccl.barrier(*args, **kwargs)


from datetime import timedelta


def create_moodist_backend(
    store: torch.distributed.Store, rank: int, size: int, timeout: timedelta
):
    return MoodistProcessGroup(store, rank, size)


torch.distributed.Backend.register_backend("moodist", create_moodist_backend)


def enable_profiling(b):
    _C.enable_profiling(b)

__all__ = []
