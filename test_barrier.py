import os
import subprocess
import time
import sys

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

        # moodist.enable_profiling(True)
    if n == "tccl":
        import tccl

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

    for i in range(1000):
        start = time.monotonic()
        for _ in range(10000):
            dist.barrier()
        print("barrier %d done (%f/s)" % (i, 10000 / (time.monotonic() - start)))

    dist.barrier()
    torch.cuda.synchronize()


if len(sys.argv) < 3:
    f(sys.argv[1])
    sys.exit(0)

ngpus = 8

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
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)
print("bye")
