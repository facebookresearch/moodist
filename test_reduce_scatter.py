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


def f(n):
    import torch
    import torch.distributed as dist

    if n == "moodist":
        import sys

        # sys.path.append("/home/vegardmella/moolib/py")
        # sys.path.append("/private/home/vegardmella/moolib/py")
        import moodist
        #moodist.enable_profiling(True)
    if n == "tccl":
        import tccl

    # print(torch.randn(1).cuda())

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
    # data = torch.randn(1024 * 1024 * 64).cuda() + 1
    data = torch.randn(1024 * 1024 * 1).cuda() + 1
    #data = torch.randn(1024 * 1024 * 800).cuda() + 1
    #data = torch.randn(1536024 // 2, device="cuda") + 1
    # data = torch.randn(1024 * 1024 + 123 * 14 + 91).cuda() + 1
    # data = torch.randn(128 * 4).cuda() + 1
    if rank == 0:
        print("reduce-scatter")
    tmp = data.clone()

    result0 = torch.zeros(data.numel() // size).cuda()
    #result0 = torch.zeros(1024 * 1024 * 800 * size, device="cuda")

    print("%d: input is (sum %f) " % (rank, data.sum()), data)

    all_inputs = []
    for r in range(size):
        torch.manual_seed(42 + r)
        rdata = torch.randn(data.numel()).cuda() + 1
        all_inputs.append(rdata)
    
    correct_result = sum(all_inputs).chunk(size)[rank]

    print("sum(all_inputs) shape ", sum(all_inputs).shape)
    print("correct_result shape ", correct_result.shape)

    torch.manual_seed(420 + rank)

    print("result0 is at %#x" % result0.data_ptr())

    for _ in range(10):
        print("rank %d warmup %d" % (rank, _))
        # dist.all_gather(result, tmp)
        result0 -= 1
        if _ % 3 == 0:
            tmp = torch.zeros_like(tmp)
        tmp.copy_(data)
        # dist.all_gather(result, tmp)
        print("result0 numel is ", result0.numel())
        print("tmp numel is ", tmp.numel())
        dist.reduce_scatter_tensor(result0, tmp)
        tmp.zero_()
        result = [result0]
        # dist._all_gather_base(result, tmp)
        if True:
            i = 0
            v = correct_result
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
        torch.cuda.synchronize()
        print("rank %d warmup %d done" % (rank, _))
    tmp.copy_(data)

    print("rank %d warmup done" % (rank))


    warmup_result = [t.clone() for t in result]
    dist.barrier()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    start = time.time()

    loopcount = 0

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

    #moodist.enable_profiling(False)


if len(sys.argv) < 2:
    f("moodist")
    sys.exit(0)

# for n in ("moolib", "nccl", "moolib", "nccl", "moolib", "nccl", "moolib", "nccl"):
for n in ("moodist",):
    os.environ["MASTER_PORT"] = str(master_port)
    master_port += 1
    pids = []
    ngpus = 8
    for i in range(ngpus):
        os.environ["RANK"] = str(int(os.environ["SLURM_PROCID"]) * ngpus + i)
        os.environ["WORLD_SIZE"] = str(int(os.environ["SLURM_NTASKS"]) * ngpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i)

        if os.environ["RANK"] == "0":
            print(n)
        pid = os.fork()
        if pid == 0:
            f(n)
            os._exit(0)
        pids.append(pid)
    for pid in pids:
        os.waitpid(pid, 0)
