import os
import subprocess
import sys
import socket
import random

import torch
import torch.distributed as dist

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


class Op:
    def run(group, data):
        raise NotImplementedError


class AllGather:
    def run(
        self,
        group: dist.ProcessGroup,
        data: torch.Tensor,
        rng: random.Random,
        sync=False,
    ):
        output = torch.randn_like(data)
        input = data.chunk(group.size())[group.rank()].clone()
        handle = group._allgather_base(output, input)
        if sync:
            handle.wait()
        if rng.randrange(0, 8) == 0:
            handle.wait()
            input.zero_()
        return (handle, output, data)

    def wait(self, x):
        handle, output, data = x
        handle.wait()
        assert torch.equal(output, data)


class ReduceScatter:
    def run(
        self,
        group: dist.ProcessGroup,
        data: torch.Tensor,
        rng: random.Random,
        sync=False,
    ):
        result = torch.zeros_like(data).chunk(group.size())[group.rank()].clone()
        for i in range(group.size()):
            x = torch.randn_like(data)
            result.add_(x.chunk(group.size())[group.rank()])
            if i == group.rank():
                input = x
        output = torch.randn_like(result)
        if rng.randrange(0, 8) == 0:
            tmp = input
            input = tmp.clone()
            tmp.zero_()
        if rng.randrange(0, 8) == 0:
            tmp = torch.zeros_like(input)
            tmp.copy_(input)
        handle = group._reduce_scatter_base(output, input)
        if sync:
            handle.wait()
        if rng.randrange(0, 8) == 0:
            handle.wait()
            input.zero_()
        return (handle, output, result)

    def wait(self, x):
        handle, output, data = x
        handle.wait()
        assert torch.allclose(output, data, 1e-3, 1e-2)

class AllToAll:
    def run(
        self,
        group: dist.ProcessGroup,
        data: torch.Tensor,
        rng: random.Random,
        sync=False,
    ):
        output = torch.randn_like(data)
        output.zero_()
        result = []
        for i in range(group.size()):
            x = torch.randn_like(data)
            result.append(x.chunk(group.size())[group.rank()])
            if i == group.rank():
                input = x
        result = torch.cat(result)
        handle = group.alltoall_base(output, input, [], [])
        if sync:
            handle.wait()
        if rng.randrange(0, 8) == 0:
            handle.wait()
            input.zero_()
        return (handle, output, result)

    def wait(self, x):
        handle, output, data = x
        handle.wait()
        assert torch.equal(output, data)

class AllToAll2:
    def run(
        self,
        group: dist.ProcessGroup,
        data: torch.Tensor,
        rng: random.Random,
        sync=False,
    ):
        output = torch.randn_like(data)
        output.zero_()
        result = []
        for i in range(group.size()):
            x = torch.randn_like(data)
            result.append(x.chunk(group.size())[group.rank()])
            if i == group.rank():
                input = x
        result = torch.cat(result)
        handle = group.alltoall(output.chunk(group.size()), input.chunk(group.size()))
        if sync:
            handle.wait()
        if rng.randrange(0, 8) == 0:
            handle.wait()
            input.zero_()
        return (handle, output, result)

    def wait(self, x):
        handle, output, data = x
        handle.wait()
        assert torch.equal(output, data)


class Mixed:
    def run(
        self,
        group: dist.ProcessGroup,
        data: torch.Tensor,
        rng: random.Random,
        sync=False,
    ):
        if rng.randrange(2) == 0:
            op = AllGather()
        else:
            op = ReduceScatter()
        return (op, op.run(group, data, rng, sync=sync))

    def wait(self, x):
        op, x = x
        op.wait(x)


def random_data(rng: random.Random, step):
    while True:
        n = rng.randrange(step, 1024 * 1024 * 2, step)
        n2 = rng.randrange(1, 1024 * 1024 * 2)
        if n * n2 > 1024 * 1024 * 64:
            continue
        if rng.randrange(2) == 0:
            t = torch.rand((n,), device="cuda")
        else:
            t = torch.rand((n * n2), device="cuda")
        return t


streams = {}


def step_streams(rng: random.Random, group: dist.ProcessGroup, op):
    global streams
    handles = []
    for s in range(rng.randrange(4)):
        if s not in streams:
            streams[s] = torch.cuda.Stream()
        stream = streams[s]
        with torch.cuda.stream(stream):
            for i in range(rng.randrange(1, 4)):
                data = random_data(rng, group.size())
                handles.append(op.run(group, data, rng))
    for h in handles:
        op.wait(h)


def step(rng: random.Random, group: dist.ProcessGroup, op):
    handles = []
    for s in range(rng.randrange(4)):
        for i in range(rng.randrange(1, 4)):
            data = random_data(rng, group.size())
            handles.append(op.run(group, data, rng))
    for h in handles:
        op.wait(h)
        random_data(rng, group.size())


def step_sequential(rng: random.Random, group: dist.ProcessGroup, op):
    for i in range(rng.randrange(1, 4)):
        data = random_data(rng, group.size())
        op.wait(op.run(group, data, rng))

    data = [random_data(rng, group.size()) for _ in range(rng.randrange(1, 4))]
    for d in data:
        op.wait(op.run(group, d, rng))


def step_parallel(rng: random.Random, group: dist.ProcessGroup, op):
    handles = []
    for i in range(rng.randrange(1, 4)):
        data = random_data(rng, group.size())
        h = op.run(group, data, rng, sync=True)
        handles.append(h)
    for h in handles:
        op.wait(h)


def entry(backend):

    print("hostname: %s\n" % (socket.gethostname()))

    if backend == "moodist":
        import moodist

        moodist.enable_cpu_allocator()
        moodist.enable_cuda_allocator()
        moodist.set_prefer_kernel_less(True)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    dist.init_process_group(
        backend=backend,
    )

    group = dist.new_group()

    rng = random.Random(42)

    torch.cuda.manual_seed(42)

    ops = [AllGather(), ReduceScatter(), Mixed()]
    #ops = [AllToAll(), AllToAll2()]
    funcs = [step, step_streams, step_sequential, step_parallel]

    for i in range(1000):
        for op in ops:
            for f in funcs:
                if rng.randrange(3) == 0:
                    f(rng, group, op)

        print("%d ok" % i)


entry(sys.argv[1])
