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

        sys.path.append("/home/vegardmella/moodist/py")
        import moodist

        # moodist.enable_profiling(True)
    if n == "tccl":
        import tccl

    # print(torch.randn(1).cuda())

    dist.init_process_group(
        backend=n,
    )

    gloo = dist.new_group(backend="gloo")

    rank = dist.get_rank()

    op = "all_gather"

    def t(i):
        if rank == 0:
            print("Running benchmark with world size %d" % i)
        ranks = list(range(i))
        group = dist.new_group(ranks)
        if not rank in range(i):
            gloo.barrier()
            return

        max_size = 1024 * 1024 * 40

        f = None
        if rank == 0:
            f = open("speed-%d-%s.txt" % (i, dist.get_backend(group)), "w")

        s = 73728 // 4
        #s = 9437184
        xi = 0
        while True:
            xi += 1
            # s = s + (s + 1) // 2
            s = s * 2
            s = min(s, max_size)

            if s * i > 33554432 * 128:
                break

            # if xi == 8:
            #     group = group1
            #     print("time group1")
            # if xi == 16:
            #     group = group2
            #     print("time group2")
            # if xi == 24:
            #     # group = dist
            #     print("time dist")

            #     break

            print("test world %d  size %d" % (i, s))

            iterations = min((max_size * 40) // max(s, 4096), 2000)
            print("%d iterations" % iterations)

            if op == "all_gather":
                output = torch.randn(s * i).cuda()
                input = torch.randn(s).cuda()
            elif op == "reduce_scatter":
                output = torch.randn(s).cuda()
                input = torch.randn(s * i).cuda()
            else:
                assert False

            # print("backend is ", dist.get_backend(group))

            group.barrier()
            torch.cuda.synchronize()
            group.barrier()
            torch.cuda.synchronize()

            # warmup
            if op == "all_gather":
                events = []
                for _ in range(iterations):
                    group._allgather_base(output, input).wait()
                    torch.cuda.synchronize()
            elif op == "reduce_scatter":
                events = []
                for _ in range(iterations):
                    group._reduce_scatter_base(output, input).wait()
                    torch.cuda.synchronize()
            else:
                assert False
                # torch.cuda.synchronize()
            # for _ in range(iterations):
            #     if len(events) >= 2:
            #         events.pop(0).synchronize()
            #     group._reduce_scatter_base(output, input)
            #     e = torch.cuda.Event()
            #     e.record()
            #     events.append(e)

            group.barrier()
            torch.cuda.synchronize()
            group.barrier()
            torch.cuda.synchronize()

            for x in range(3):
                freeevents = [
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                    torch.cuda.Event(),
                ]

                events = []
                start = time.monotonic()
                # for _ in range(iterations):
                #     group._reduce_scatter_base(output, input)
                #     #torch.cuda.synchronize()
                if op == "all_gather":
                    if False:
                        for _ in range(iterations):
                            dist.all_gather_into_tensor(output, input)
                            torch.cuda.synchronize()
                    else:
                        # if x == 1:
                        #    moodist.enable_profiling(True)
                        for _ in range(iterations):
                            if len(events) >= 2:
                                e = events.pop(0)
                                e.synchronize()
                                freeevents.append(e)
                            group._allgather_base(output, input).wait()
                            e = freeevents.pop(0)
                            e.record()
                            events.append(e)
                        # if x == 1:
                        #    moodist.enable_profiling(False)
                elif op == "reduce_scatter":
                    for _ in range(iterations):
                        if len(events) >= 2:
                            e = events.pop(0)
                            e.synchronize()
                            freeevents.append(e)
                        group._reduce_scatter_base(output, input).wait()
                        e = freeevents.pop(0)
                        e.record()
                        events.append(e)
                else:
                    assert False
                torch.cuda.synchronize()
                t = time.monotonic() - start
                torch.cuda.synchronize()

                if rank == 0 or True:
                    rate = iterations / t
                    bw = s * output.element_size() / 1024 / 1024 / 1024 * iterations / t
                    print(
                        "time: %g, %g/s  %gG/s (iterations %d)"
                        % (t, rate, bw, iterations)
                    )
                    if f is not None:
                        f.write("%d %d %d %g %g %g\n" % (i, s, iterations, t, rate, bw))
                        f.flush()

            if s == max_size:
                break

        dist.destroy_process_group(group)

        torch.cuda.synchronize()
        gloo.barrier()

    world_size = dist.get_world_size()
    print("%d: world size is %d\n" % (rank, world_size))

    # dist.barrier()
    torch.cuda.synchronize()
    # dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("init ok")

    torch.manual_seed(42 + rank)

    i = min(800, world_size)
    while True:
        t(i)

        if i >= world_size:
            break
        # i = i + (i + 1) // 2
        i = max(i * 2, 1)
        if i >= 8:
            i = (i + 7) // 8 * 8
        i = min(i, world_size)


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
#for n in ("moodist", "nccl"):
for n in ("moodist", "nccl"):
    # for n in ("nccl",):
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
