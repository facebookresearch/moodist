import os
import subprocess
import time
import sys

import random

if "LOCAL_RANK" not in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # define master address and master port
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )
    master_addr = hostnames.split()[0].decode("utf-8")
    master_port = 8195
    # print(PREFIX + f"Master address: {params.master_addr}")
    # print(PREFIX + f"Master port   : {params.master_port}")

    # set environment variables for 'env://'
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]

else:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

os.environ["NCCL_PROTO"] = "LL128"


def f(n):
    import torch
    import torch.distributed as dist

    if n == "moodist":
        import sys

        sys.path.append("/home/vegardmella/moodist/py")
        import moodist

    # print(torch.randn(1).cuda())

    import socket

    print("hostname: %s\n" % (socket.gethostname()))

    dist.init_process_group(
        backend=n,
    )

    # group1 = dist.new_group()
    # group2 = dist.new_group()

    rank = dist.get_rank()
    size = dist.get_world_size()

    print("%d: world size is %d\n" % (rank, size))

    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("init ok")

    torch.manual_seed(42 + rank)

    seed = time.monotonic_ns()
    if rank == 0:
        dist.broadcast_object_list([seed])
    else:
        l = [None]
        dist.broadcast_object_list(l)
        seed = l[0]
    random.seed(seed)

    # assert size % 8 == 0
    # nodes = size // 8

    if random.random() < 0.1:
        nbytes = int(2 ** (7 + random.random() * 16)) // 256 * 256
    else:
        nbytes = int(2 ** (12 + random.random() * 16)) // 1024 * 1024
    nelem = max(nbytes // 4, 1)

    input = torch.randn(nelem, device="cuda")
    output = torch.empty(size * nelem, device="cuda")
    for z in range(1000):
        dist.all_gather_into_tensor(output, input)
        torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    print("finish?")
    time.sleep(5)
    print("finish!")

    # for i in range(100):
    #     local_size = random.randint(1, 8)
    #     nnodes = random.randint(1, nodes - 1)
    #     ranks = []

    #     nbytes = int(2 ** (random.random() * 28))
    #     nelem = max(nbytes // 4, 1)

    #     for n in range(nnodes):
    #         ranks += list(range(8 * n, 8 * n + local_size))

    #     print("local_size %d, nodes %d, ranks %s" % (local_size, nnodes, ranks))

    #     group = dist.new_group(ranks=ranks)

    #     if rank in ranks:
    #         input = torch.randn(nelem, device="cuda")
    #         output = torch.empty(len(ranks) * nelem, device="cuda")
    #         for z in range(1000):
    #             group._allgather_base(output, input).wait()
    #             torch.cuda.synchronize()

    #     dist.barrier()
    #     torch.cuda.synchronize()
    #     dist.barrier()
    #     torch.cuda.synchronize()


if len(sys.argv) < 3:
    f(sys.argv[1])
    sys.exit(0)

ngpus = int(sys.argv[1])

fds = []
for i in range(ngpus):
    fds.append(
        os.open(
            "out-%s.txt" % str(int(os.environ["SLURM_PROCID"]) * ngpus + i),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        )
    )

# for n in ("moolib", "nccl", "moolib", "nccl", "moolib", "nccl", "moolib", "nccl"):
# for n in ("moodist", "nccl"):
# for n in ("nccl", "moodist"):
for n in ("moodist",):
    # for n in ("moodist",):
    os.environ["MASTER_PORT"] = str(master_port)
    master_port += 1
    pids = []
    for i in range(ngpus):
        os.environ["RANK"] = str(int(os.environ["SLURM_PROCID"]) * ngpus + i)
        os.environ["WORLD_SIZE"] = str(int(os.environ["SLURM_NTASKS"]) * ngpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)

        if os.environ["RANK"] == "0":
            print(n)
        pid = os.fork()
        if pid == 0:
            fd = fds[i]
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            f(n)
            sys.exit(0)
            # os._exit(0)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)
print("bye")
