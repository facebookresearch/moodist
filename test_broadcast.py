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

        # sys.path.append("/home/vegardmella/moolib/py")
        # sys.path.append("/private/home/vegardmella/moolib/py")
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

    group1 = dist.new_group()
    group2 = dist.new_group()

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

    # process_group = torch.distributed.new_group("foo")

    # if "cuda" in str(device):

    #     input_tensor = torch.ones(1).to(device)
    #     output = list(torch.zeros(world_size).to(device).chunk(world_size))
    #     dist.all_gather(output, input_tensor, group=process_group)
    #     assert torch.cat(output).sum() == float(world_size), (
    #         f"found {torch.cat(output).sum()} devices in process group but "
    #         f"world_size={world_size}. Check torch.cuda.set_device is called properly"
    #     )

    #     assert False

    # all_data = []
    # for i in range(dist.get_world_size()):
    #    all_data.append(torch.randn(1024 * 1024 * 100).cuda())
    # data: torch.Tensor = all_data[rank]
    # print("data: ", data)
    # data = torch.randn(1024 * 1024 * 10).cuda()

    # data = torch.randn(1024 * 1024 * 100 // size).cuda()
    # data = torch.randn(1024 * 1024 * 100 // size).cuda()
    # data = torch.randn(4).cuda() + 1
    data = torch.randn(1024 * 1024 * 4).cuda() + 1
    # data = torch.randn(1024 * 1024 * 256).cuda() + 1
    # data = torch.randn(263520).cuda() + 1
    #data = torch.randn(442416).cuda() + 1
    # data = torch.randn(262144 - 1024).cuda() + 1
    # data = torch.randn(262144 - 64).cuda() + 1
    # data = torch.randn(682678 // 2).cuda() + 1
    # data = torch.randn(1024 * 1024).cuda() + 1
    # data = torch.randn(1024).cuda() + 1
    # data = torch.randn(1024 * 1024 * 800).cuda() + 1
    # data = torch.randn(1536024 // 2, device="cuda") + 1
    # data = torch.randn(1024 * 1024 + 123 * 14 + 91).cuda() + 1
    # data = torch.randn(1024 * 1024 * 4).cuda() + 1
    if rank == 0:
        print("all-gather")
    result = [torch.zeros_like(data) for _ in range(size)]
    # data2 = data.clone() + 1
    # result2 = [torch.empty_like(data2) for _ in range(size)]
    tmp = data.clone()
    # tmp2 = data2.clone()

    result0 = torch.cat(result)
    # result0 = torch.zeros(1024 * 1024 * 800 * size, device="cuda")

    print("%d: input is (sum %f) " % (rank, data.sum()), data)

    correct_result = []
    for r in range(1):
        torch.manual_seed(42 + r)
        rdata = torch.randn(data.numel()).cuda() + 1
        correct_result.append(rdata)

    torch.manual_seed(420 + rank)

    datax = torch.randn(20, 1024 * 1024 * 10).cuda()

    # for i in range(20):
    #     print(rank, i)
    #     dist.all_gather(result, datax[i])
    #     # datax[i].fill_(float("nan"))

    #     # for x in result:
    #     #     assert not x.sum().isnan()

    # print(rank, "datax done")

    # result = torch.stack(result)

    print("result0 is at %#x" % result0.data_ptr())

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    tmp2 = tmp.clone()
    result02 = result0.clone()

    for _ in range(1000):
        # print("rank %d warmup %d" % (rank, _))
        # dist.all_gather(result, tmp)
        result = [torch.zeros_like(data) for _ in range(size)]
        result0 -= 1
        if _ % 3 == 0:
            tmp = torch.zeros_like(tmp)
        torch.randn(1024 * 1024)
        tmp.copy_(data)
        ostream = torch.cuda.current_stream()
        dist.broadcast(tmp, 0)
        result = [tmp.clone()]
        tmp.zero_()
        # result = result0.chunk(size)
        # dist._all_gather_base(result, tmp)
        if True:
            for i, v in zip(range(size), correct_result):
                if not torch.allclose(result[i], v, 1e-3, 1e-2):
                    print(
                        "%d: result[%d].data_ptr is %#x"
                        % (rank, i, result[i].data_ptr())
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
        #torch.cuda.synchronize()
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
            torch.cuda.synchronize()
            print("rank %d test3 done" % rank)

            print("rank %d exit test3 %d\n" % (rank, x))

    warmup_result = [t.clone() for t in result]
    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()
    if 1 == 13:
        loopcount = 8000
        for i in range(loopcount):
            dist.broadcast(tmp, 0)
            torch.cuda.synchronize()
    if 1 == 1:
        loopcount = 8000
        events = []
        for i in range(loopcount):
            if len(events) >= 2:
                events.pop(0).synchronize()
            dist.broadcast(tmp, 0)
            e = torch.cuda.Event()
            e.record()
            events.append(e)

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

    if rank == 0 or True:
        s = ""
        for t in result:
            # s = "%s %f" % (s, t.square().sum().sqrt())
            s = "%s %f" % (s, t.sum())
        print("rank %d result %d: %s" % (rank, _, s))

    # dist.barrier()
    torch.cuda.synchronize()

    # moodist.enable_profiling(False)

    # correct = torch.zeros_like(data)

    # torch.manual_seed(42)

    # for i in range(dist.get_world_size()):
    #     correct += torch.randn(1024 * 1024 * 100).cuda()

    # #print("correct: ", correct)
    # if not data.equal(correct):
    #     print("bad!!")
    # diff = data - correct
    # print(diff.abs().max())


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
for n in ("nccl", "moodist"):
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
