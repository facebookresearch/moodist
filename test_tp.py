import os
import subprocess
import sys
import socket
import random
from typing import List

import torch
import torch.distributed as dist


import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

import torch.distributed

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import torch.distributed._functional_collectives

from torch.distributed._tensor import Replicate, Shard

import moodist


def foo():
    assert False


original_all_gather_tensor = torch.distributed._functional_collectives.all_gather_tensor


def resolve_group(group):
    if isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            return dmesh.get_group(dim)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    return None


@torch.library.custom_op("moodist::share", mutates_args=[])
def share(group_name: str, local_tensor: torch.Tensor) -> List[torch.Tensor]:
    group = moodist.find_process_group(group_name)
    r = group.share(local_tensor)
    r[group.rank()] = local_tensor.clone()
    return r


@share.register_fake
def _(group_name: str, local_tensor: torch.Tensor):
    group = moodist.find_process_group(group_name)
    return [torch.empty_like(local_tensor) for _ in range(group.size())]


def all_gather_tensor(self: torch.Tensor, gather_dim: int, group, tag: str = ""):
    assert self.is_contiguous()
    group = resolve_group(group)
    tensors: list[torch.Tensor] = share(group.moodist_name(), self)
    # print("tensors is ", tensors)
    return torch.cat(tensors, dim=gather_dim)


torch.distributed._functional_collectives.all_gather_tensor = all_gather_tensor
torch.distributed._functional_collectives.all_gather_tensor_autograd = foo

torch.distributed._functional_collectives.all_reduce = foo

if "LOCAL_RANK" not in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    master_addr = hostnames.split()[0].decode("utf-8")
    master_port = 8195

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]


def entry(backend):

    print("hostname: %s\n" % (socket.gethostname()))

    if backend == "moodist":
        moodist.enable_cpu_allocator()
        moodist.enable_cuda_allocator()
        moodist.set_prefer_kernel_less(True)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    dist.init_process_group(
        backend=backend,
    )

    group = dist.new_group()

    rank = group.rank()

    rng = random.Random(42)

    torch.cuda.manual_seed(42 + rank)
    torch.manual_seed(42 + rank)

    lin = nn.Linear(1024, 64).cuda()
    lin2 = nn.Linear(64, 4).cuda()

    tp_mesh = init_device_mesh("cuda", [group.size()])

    parallelize_module(
        module=lin,
        device_mesh=tp_mesh,
        parallelize_plan=ColwiseParallel(output_layouts=Replicate()),
    )

    parallelize_module(
        module=lin2,
        device_mesh=tp_mesh,
        parallelize_plan=RowwiseParallel(),
    )

    x = lin(torch.randn(2, 1024, device="cuda"))

    print(x)

    # print(x.shape)

    x.sum().backward()

    print(lin.weight.grad)

    # x = lin2(x)

    # print(x.shape)


entry(sys.argv[1])
