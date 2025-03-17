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

    # os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]

    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]

# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

# os.environ["NCCL_PROTO"] = "LL128"


def f(n):
    import torch
    import torch.distributed as dist

    if n == "moodist":
        import sys

        # sys.path.append("/home/vegardmella/moolib/py")
        # sys.path.append("/private/home/vegardmella/moolib/py")
        import moodist

        moodist.enable_cuda_allocator()
        moodist.enable_cpu_allocator()
        
        moodist.set_prefer_kernel_less(True)

        # moodist.enable_profiling(True)
    if n == "tccl":
        import tccl

    # print(torch.randn(1).cuda())

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    dist.init_process_group(
        backend=n,
    )

    rank = dist.get_rank()
    size = dist.get_world_size()

    print("%d: world size is %d\n" % (rank, size))

    # dist.barrier()
    torch.cuda.synchronize()
    # dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("init ok")

    torch.manual_seed(42 + rank)

    device = torch.device("cuda:0")
    world_size = size

    test_gather = True

    # data = torch.randn(1024 * 1024 * 100 // size).cuda()
    # data = torch.randn(1024 * 1024 * 100 // size).cuda()
    # data = torch.randn(1024 * 1024 * 40).cuda() + 1
    # data = torch.randn(589824 * 8 * size).cuda() + 1
    # data = torch.randn((1024 * 512 - 1024 * 8) * size).cuda() + 1
    # data = torch.randn(32768 * size).cuda() + 1
    # data = torch.randn(1024 * 1024 * 64).cuda() + 1
    #data = torch.randn(2 * 8192 * 7168).cuda() + 1
    # data = torch.randn(1024 * 1024 * 64 * size).cuda() + 1
    # data = torch.randn((442416 - 4) * size).cuda() + 1
    # data = torch.randn(527040 * size).cuda() + 1
    # data = torch.randn(589824 * size).cuda() + 1
    #data = torch.randn(294912 * size).cuda() + 1
    data = torch.randn(3784800 // 4 * size).cuda() + 1
    # data = torch.randn(524288 * size).cuda() + 1
    # data = torch.randn(1024 * 1024 * 2 * size).cuda() + 1
    # data = torch.randn(1024 * 1024 * 256 * size).cuda() + 1
    # data *= 0
    # data += 2 ** rank
    # data = torch.randn(1024 * 1024 * 800).cuda() + 1
    # data = torch.randn(1536024 // 2, device="cuda") + 1
    # data = torch.randn(1024 * 1024 + 123 * 14 + 91).cuda() + 1
    # data = torch.randn(128 * 4).cuda() + 1
    if rank == 0:
        print("reduce-scatter")
    tmp = data.clone()

    result0 = torch.zeros(data.numel() // size).cuda()
    # result0 = torch.zeros(1024 * 1024 * 800 * size, device="cuda")

    #dtype = torch.bfloat16
    dtype = torch.float

    data = data.to(dtype)
    result0 = result0.to(dtype)
    tmp = tmp.to(dtype)

    print("%d: input is (sum %f) " % (rank, data.sum()), data.view(size, -1))

    check = True

    if check:
        all_inputs = []
        for r in range(size):
            torch.manual_seed(42 + r)
            rdata = torch.randn(data.numel()).cuda() + 1
            # rdata *= 0
            # rdata += 2 ** r
            all_inputs.append(rdata.to(dtype))

            # print("%d: input sum %d is %f" % (rank, r, rdata.chunk(size)[rank].sum()))

        correct_result = sum(all_inputs).chunk(size)[rank]

    torch.manual_seed(420 + rank)

    # print("result0 is at %#x" % result0.data_ptr())

    # x = torch.randn(1024 * 1024).cuda()
    # y = torch.zeros(x.numel() * size).cuda()

    for _ in range(1000):
        # print("rank %d warmup %d" % (rank, _))
        # dist.all_gather(result, tmp)
        result0 -= 1
        if _ % 3 == 0:
            tmp = torch.zeros_like(tmp)
        # if _ % 9 <= 4:
        #     dist.all_gather_into_tensor(y, x)
        tmp.copy_(data)
        # dist.all_gather(result, tmp)
        # print("result0 numel is ", result0.numel())
        # print("tmp numel is ", tmp.numel())
        dist.reduce_scatter_tensor(result0, tmp)
        tmp.zero_()
        result = [result0]
        # dist._all_gather_base(result, tmp)
        if check:
            i = 0
            v = correct_result
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
                print("%d: got %s" % (rank, result[i][indices]))
                print("%d: should be %s" % (rank, v[indices]))
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
        # torch.cuda.synchronize()
        # print("rank %d warmup %d done" % (rank, _))
    tmp.copy_(data)

    print("rank %d warmup done" % (rank))

    if False:
        tmpz = []
        for x in range(10):
            print("rank %d enter test2 %d\n" % (rank, x))
            test2_data = torch.randn(3072048, device="cuda")
            test2_result = torch.zeros(3072048 * size, device="cuda")

            for i in range(10):
                dist._all_gather_base(test2_result, test2_data)
                dist.reduce_scatter_tensor(test2_data, test2_result)
            torch.cuda.synchronize()
            print("rank %d test2 done" % rank)

            tmpz.append(test2_data)
            tmpz.append(test2_result)

            print("rank %d exit test2 %d\n" % (rank, x))

        tmpz = None

        import random

        random.seed(42)
        for x in range(100):
            print("rank %d enter test3 %d\n" % (rank, x))
            s = random.randint(1024, 1024 * 10) * 4
            test3_data = torch.randn(s, device="cuda")
            test3_result = torch.zeros(s * size, device="cuda")

            for i in range(10):
                dist._all_gather_base(test3_result, test3_data)
                dist.reduce_scatter_tensor(test3_data, test3_result)
            torch.cuda.synchronize()
            print("rank %d test3 done" % rank)

            print("rank %d exit test3 %d\n" % (rank, x))

    warmup_result = [t.clone() for t in result]
    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()

    if 1 == 11:
        # result = [torch.zeros_like(data) for _ in range(size)]
        from torch.profiler import profile, record_function, ProfilerActivity

        moodist.enable_profiling(True)

        test = result0.clone()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            loopcount = 1000
            events = []
            for i in range(loopcount):
                if len(events) >= 2:
                    events.pop(0).synchronize()
                dist.reduce_scatter_tensor(result0, tmp)
                test.add_(result0)
                e = torch.cuda.Event()
                e.record()
                events.append(e)
        torch.cuda.synchronize()
        prof.export_chrome_trace(f"trace-{rank}.json")
        moodist.enable_profiling(False)

        dist.reduce_scatter_tensor(result0, tmp)
    elif 1 == 13:
        x1 = torch.nn.Linear(1024, 1024)
        x2 = torch.nn.Linear(1024, 1024)
        y = torch.randn(1024)
        loopcount = 1000
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        for i in range(loopcount):
            with torch.cuda.stream(stream1):
                dist.reduce_scatter_tensor(result0, tmp)
            # with torch.cuda.stream(stream2):
            #    x1(y)
            #    x2(y)

            torch.cuda.synchronize()
    elif 1 == 12:
        loopcount = 10000
        events = []
        for i in range(loopcount):
            dist.reduce_scatter_tensor(result0, tmp)
            torch.cuda.synchronize()
    elif 1 == 1:
        loopcount = 10000
        freeevents = [
            torch.cuda.Event(),
            torch.cuda.Event(),
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]
        events = []
        # if n == "moodist":
        #     moodist.enable_profiling(True)
        for _ in range(loopcount):
            if len(events) >= 2:
                e = events.pop(0)
                e.synchronize()
                freeevents.append(e)
            dist.reduce_scatter_tensor(result0, tmp)
            e = freeevents.pop(0)
            e.record()
            events.append(e)
        # if n == "moodist":
        #     moodist.enable_profiling(False)

        dist.reduce_scatter_tensor(result0, tmp)
    elif True:
        loopcount = 10000
        events = []
        for i in range(loopcount):
            if len(events) >= 2:
                events.pop(0).synchronize()
            dist.reduce_scatter_tensor(result0, tmp)
            e = torch.cuda.Event()
            e.record()
            events.append(e)

    print("rank %d all done!" % rank)
    dist.barrier()
    torch.cuda.synchronize()
    t = time.time() - start
    if rank == 0:
        print(
            "time: %g, %g/s  %gG/s"
            % (
                t,
                loopcount / t,
                data.numel()
                // size
                * data.element_size()
                / 1024
                / 1024
                / 1024
                * loopcount
                / t,
            )
        )

    if rank == 0:
        s = ""
        for t in result:
            # s = "%s %f" % (s, t.square().sum().sqrt())
            s = "%s %f" % (s, t.sum())
        print("rank %d result %d: %s" % (rank, _, s))

    # dist.barrier()
    torch.cuda.synchronize()

    # moodist.enable_profiling(False)


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
for n in ("moodist", "nccl"):
    # for n in ("nccl", "moodist"):
    # for n in ("moodist", ):
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
