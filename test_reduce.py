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

os.environ["NCCL_PROTO"] = "^LL,^LL128"


def f(n):
    import torch
    import torch.distributed as dist

    if n == "moodist":
        import sys

        sys.path.insert(0, "/home/vegardmella/moodist/py")
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

    torch.manual_seed(42 + rank)

    device = torch.device("cuda:0")
    world_size = size

    #data = torch.randn(1024 * 1024 * 4).cuda() + 1
    data = torch.randn(1024 * 412 + 13).cuda() + 1

    tmp = data.clone()

    print("%d: input is (sum %f) " % (rank, data.sum()), data)

    correct_result = []
    for r in range(size):
        torch.manual_seed(42 + r)
        rdata = torch.randn(data.numel()).cuda() + 1
        correct_result.append(rdata)

    torch.manual_seed(420 + rank)

    for _ in range(1000):
        # print("rank %d warmup %d" % (rank, _))
        # dist.all_gather(result, tmp)
        result = [torch.zeros_like(data)]
        result[0] -= 1
        if _ % 3 == 0:
            tmp = torch.zeros_like(tmp)
        tmp.copy_(data)
        dist.reduce(tmp, 0)
        result[0].copy_(tmp)
        tmp.zero_()
        if rank == 0:
            v = torch.stack(correct_result).sum(0)
            i = 0
            if not torch.allclose(result[i], v, 1e-3, 1e-2):
                print(
                    "%d: result[%d].data_ptr is %#x" % (rank, i, result[i].data_ptr())
                )
                print("%d: data.data_ptr() is %#x" % (rank, data.data_ptr()))
                print("%d: result %d sum %f" % (rank, i, result[i].sum()))
                print("%d: should be %f" % (rank, v.sum()))
                indices = ((result[i] - v).abs() >= 1e-3).nonzero(as_tuple=True)[0]
                print(
                    "%d: indices " % rank,
                    indices,
                )
                print(result[i][indices])
                print(v[indices])
                print(
                    "allclose 1 ",
                    torch.allclose(result[i], v, 1e-3, 1e-2),
                    torch.allclose(result[i], v, 1e-3, 1e-2),
                    torch.allclose(result[i], v, 1e-3, 1e-2),
                )
                time.sleep(1)
                print(
                    "allclose 2 ",
                    torch.allclose(result[i], v, 1e-3, 1e-2),
                    torch.allclose(result[i], v, 1e-3, 1e-2),
                    torch.allclose(result[i], v, 1e-3, 1e-2),
                )
                print("%d: result %d sum %f" % (rank, i, result[i].sum()))
                raise RuntimeError("%d: wrong result for index %d" % (rank, i))
        torch.cuda.synchronize()
        # print("rank %d warmup %d done" % (rank, _))
    tmp.copy_(data)

    # print("rank %d warmup done" % (rank))

    warmup_result = [t.clone() for t in result]
    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()

    if 1 == 2:
        from torch.profiler import profile, record_function, ProfilerActivity

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            loopcount = 10000
            handles = []
            for _ in range(loopcount):
                handles.append(dist.reduce(tmp, 0, async_op=True))
                if len(handles) >= 8:
                    handles.pop(0).wait()
            for v in handles:
                v.wait()
        torch.cuda.synchronize()
        prof.export_chrome_trace(f"trace-{rank}.json")
    else:
        loopcount = 10000
        handles = []
        for i in range(loopcount):
            handles.append(dist.reduce(tmp, i % size, async_op=True))
            if len(handles) >= 16:
                handles.pop(0).wait()
            # torch.cuda.synchronize()
        for v in handles:
            v.wait()
    # moodist.enable_profiling(False)

    dist.reduce(tmp, 0)

    torch.cuda.synchronize()
    t = time.time() - start
    print("rank %d all done!" % rank)
    dist.barrier()
    if rank == 0:
        print(
            "time: %g, %g/s  %gG/s"
            % (
                t,
                loopcount / t,
                data.numel() * data.element_size() / 1024 / 1024 / 1024 * loopcount / t,
            )
        )

    if rank == 0:
        s = ""
        for t in result:
            # s = "%s %f" % (s, t.square().sum().sqrt())
            s = "%s %f" % (s, t.sum())
        print("rank %d result %d: %s" % (rank, _, s))

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
# for n in ("nccl", "moodist"):
for n in ("moodist", "nccl"):
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
