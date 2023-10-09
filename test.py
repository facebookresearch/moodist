
import os
import subprocess
import time
import sys

if "LOCAL_RANK" not in os.environ:
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # define master address and master port
    hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
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
    if n == "tccl":
        import tccl

    #print(torch.randn(1).cuda())
    
    dist.init_process_group(
        backend=n,
    )

    rank = dist.get_rank()
    size = dist.get_world_size()

    #dist.barrier()
    torch.cuda.synchronize()
    #dist.barrier()
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

    #all_data = []
    #for i in range(dist.get_world_size()):
    #    all_data.append(torch.randn(1024 * 1024 * 100).cuda())
    #data: torch.Tensor = all_data[rank]
    #print("data: ", data)
    #data = torch.randn(1024 * 1024 * 10).cuda()

    test_gather = True

    if test_gather:
        #data = torch.randn(1024 * 1024 * 100 // size).cuda()
        #data = torch.randn(1024 * 1024 * 100 // size).cuda()
        #data = torch.randn(1024 * 1024 * 40).cuda() + 1
        #data = torch.randn(1024 * 1024 * 64).cuda() + 1
        data = torch.randn(1024 * 1024 * 20).cuda() + 1
        #data = torch.randn(1024 * 1024 + 123 * 14 + 91).cuda() + 1
        #data = torch.randn(128 * 4).cuda() + 1
        if rank == 0:
            print("all-gather")
        result = [torch.zeros_like(data) for _ in range(size)]
        #data2 = data.clone() + 1
        #result2 = [torch.empty_like(data2) for _ in range(size)]
        tmp = data.clone()
        #tmp2 = data2.clone()

        result0 = torch.cat(result)

        print("%d: input is (sum %f) " % (rank, data.sum()), data)

        correct_result = []
        for r in range(size):
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

        #result = torch.stack(result)

        for _ in range(100):
            #print("rank %d warmup %d" % (rank, _))
            #dist.all_gather(result, tmp)
            result = [torch.zeros_like(data) for _ in range(size)]
            result0 -= 1
            tmp.copy_(data)
            #dist.all_gather(result, tmp)
            dist._all_gather_base(result0, tmp)
            tmp.zero_()
            result = result0.chunk(size)
            #dist._all_gather_base(result, tmp)
            if True and False:
                for i, v in zip(range(size), correct_result):
                    if not torch.allclose(result[i], v, 1e-3, 1e-2):
                        print("%d: result[%d].data_ptr is %#x" % (rank, i, result[i].data_ptr()))
                        print("%d: data.data_ptr() is %#x" % (rank, data.data_ptr()))
                        print("%d: result %d sum %f" % (rank, i, result[i].sum()))
                        print("%d: should be %f" % (rank, v.sum()))
                        print("%d: indices " % rank, ((result[i] - v).abs() >= 1e-3).nonzero(as_tuple=True)[0])
                        print(result[i])
                        print(v)
                        print("allclose 1 ", torch.allclose(result[i], v, 1e-3, 1e-2), torch.allclose(result[i], v, 1e-3, 1e-2), torch.allclose(result[i], v, 1e-3, 1e-2))
                        #time.sleep(4)
                        print("allclose 2 ", torch.allclose(result[i], v, 1e-3, 1e-2), torch.allclose(result[i], v, 1e-3, 1e-2), torch.allclose(result[i], v, 1e-3, 1e-2))
                        print("%d: result %d sum %f" % (rank, i, result[i].sum()))
                        raise RuntimeError("%d: wrong result for index %d" % (rank, i))
            torch.cuda.synchronize()
            #print("rank %d warmup %d done" % (rank, _))
        tmp.copy_(data)
        
        warmup_result = [t.clone() for t in result]
        dist.barrier()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        start = time.time()
        if 1 == 13:
            lin = torch.nn.Linear(4096, 4096).cuda()
            lin2 = torch.nn.Linear(4096, 4096).cuda()
            lin3 = torch.nn.Linear(4096, 4096).cuda()
            linin = torch.randn(4096).cuda()
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                loopcount = 10
                for _ in range(loopcount):
                    print("do a")
                    now = time.monotonic()
                    a = dist.all_gather(result, tmp)
                    torch.cuda.synchronize()
                    print("a took %f" % ((time.monotonic() - now) * 1000))
                    #now = time.monotonic()
                    #b = dist.all_gather(result, tmp, async_op=True)
                    #print("b took %f" % ((time.monotonic() - now) * 1000))
                    x = lin3(lin2(lin(linin)))
                    torch.cuda.synchronize()
                    #a.wait()
                    print("x done")
                    #b.wait()
                    #tmp[0:4096] += x
                    #dist.all_gather(result, tmp)
                    #dist._all_gather_base(result, tmp)
                    ##tmp.copy_(data)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(f"trace-{rank}.json")
        elif 1 == 12:
            lin = torch.nn.Linear(1024, 1024).cuda()
            lin2 = torch.nn.Linear(1024, 1024).cuda()
            lin3 = torch.nn.Linear(1024, 1024).cuda()
            linin = torch.randn(1024).cuda()
            loopcount = 1000
            for _ in range(loopcount):
                now = time.monotonic()
                a = dist.all_gather(result, tmp, async_op=True)
                print("a took %f" % ((time.monotonic() - now) * 1000))
                now = time.monotonic()
                b = dist.all_gather(result, tmp, async_op=True)
                print("b took %f" % ((time.monotonic() - now) * 1000))
                x = lin3(lin2(lin(linin)))
                a.wait()
                b.wait()
                tmp[0:1024] += x
                #dist.all_gather(result, tmp)
                #dist._all_gather_base(result, tmp)
                ##tmp.copy_(data)
                torch.cuda.synchronize()
        elif 1 == 11:
            #result = [torch.zeros_like(data) for _ in range(size)]
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                loopcount = 1000
                for _ in range(loopcount):
                    dist._all_gather_base(result0, tmp)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(f"trace-{rank}.json")
        else:
            #result = [torch.zeros_like(data) for _ in range(size)]
            loopcount = 1000
            for _ in range(loopcount):
                now = time.monotonic()
                #a = dist.all_gather(result, tmp, async_op=True)
                a = dist._all_gather_base(result0, tmp, async_op=True)
                #print("a took %f" % ((time.monotonic() - now) * 1000))
                #now = time.monotonic()
                #b = dist.all_gather(result, tmp, async_op=True)
                #print("b took %f" % ((time.monotonic() - now) * 1000))
                now = time.monotonic()
                a.wait()
                #b.wait()
                #dist.all_gather(result, tmp)
                #dist._all_gather_base(result, tmp)
                ##tmp.copy_(data)
                torch.cuda.synchronize()
                #print("synchronize took %f" % ((time.monotonic() - now) * 1000))
        # for _ in range(100):
        #     tmp.copy_(data)
        #     for v in result:
        #         v.zero_()
        #     # result = [torch.empty_like(data) for _ in range(size)]
        #     torch.cuda.synchronize()
        #     t = time.time()
        #     dist.all_gather(result, tmp)
        #     tmp.zero_()
        #     for tx, tx2 in zip(result, correct_result):
        #         if not tx.equal(tx2):
        #             s = ""
        #             for t in result:
        #                 s = "%s %f" % (s, t.sum())
        #             s2 = ""
        #             for t in correct_result:
        #                 s2 = "%s %f" % (s2, t.sum())
        #             print("rank %d result %d: %s  (should be %s)" % (rank, _, s, s2))
        #             raise RuntimeError("rank %d mismatch" % rank)
        #     #result[0].copy_(tmp)
        #     # #data = torch.randn(1024 * 1024 * 100).cuda()
        #     # tmp.copy_(data)
        #     # #tmp2.copy_(data2)
        #     # #torch.cuda.synchronize()
        #     # #t = time.time()
        #     # dist.all_gather(result, tmp)
        #     # torch.cuda.synchronize()
        #     # print("rank %d took %fms" % (rank, (time.time() - t) * 1000))
        #     # # #dist.all_gather(result2, tmp2)
        #     # tmp.zero_()
        #     # # #dist.barrier()
        #     # # #tmp2.zero_()
        #     # torch.cuda.synchronize()
        #     # s = ""
        #     # for t in result:
        #     #     #s = "%s %f" % (s, t.square().sum().sqrt())
        #     #     s = "%s %f" % (s, t.sum())
        #     # # for t in result2:
        #     # #    s = "%s %f" % (s, t.square().sum().sqrt())
        #     # #if rank == 0:
        #     # #print("rank %d result %d: %s" % (rank, _, s))
        #     # print("rank %d result %d: %s (should be %f)" % (rank, _, s, data.sum()))

        print("rank %d all done!" % rank)
        dist.barrier()
        torch.cuda.synchronize()
        t = time.time() - start
        if rank == 0:
            print("time: %g, %g/s  %gG/s" % (t, loopcount / t, data.numel() * data.element_size() / 1024 / 1024 / 1024 * loopcount / t))

        if rank == 0:
            s = ""
            for t in result:
                #s = "%s %f" % (s, t.square().sum().sqrt())
                s = "%s %f" % (s, t.sum())
            print("rank %d result %d: %s" % (rank, _, s))

    else:
        #items = 1024 * 1024 * 64 * size
        #items = 1024 * 1024 * 20 * size
        #items = 1024 * 1024 * 50
        items = 1024 * 1024 * 40
        #items = 128
        sum = 0
        sumdata = torch.zeros(items).cuda()
        for i in range(size):
            torch.manual_seed(42 + i)
            data = torch.randn(items).cuda()
            sumdata += data
            sum += data.sum().item()
        if rank == 0:
            print("%d: sum should be %f" % (rank, sum * 1))
            # for i, v in zip(range(4), sumdata.split(items // size)):
            #     print("%d: chunk %d should be " % (rank, i), v)
        torch.manual_seed(42 + rank)
        data = torch.randn(items).cuda()
        data2 = data.clone()
        tmp = data.clone()
        tmp2 = tmp.clone()
        if rank == 0:
            print("all-reduce")

        # for i, v in zip(range(4), data.split(items // size)):
        #     print("%d: input chunk %d is " % (rank, i), v)

        print(">=10 ? ", (data >= 10).sum())

        for _ in range(50):
            # if rank == _ % size:
            #     time.sleep(0.15)
            print("%d: start allreduce %d, data is " % (rank, _), data)
            datax = data.clone()
            tmp.copy_(datax)
            datax.zero_()
            dist.all_reduce(tmp)
            torch.cuda.synchronize()
            print("%d: finished allreduce %d!" % (rank, _))

            #print("%d:" % rank, tmp)
            tmpsum = tmp.sum()
            if False and not torch.allclose(tmp, sumdata, 1e-3, 1e-2):
                print("%d: sum %f" % (rank, tmpsum))
                for i, v in zip(range(4), tmp.split(items // size)):
                    print("%d: chunk %d is " % (rank, i), v)
                print("%d: indices " % rank, ((sumdata - tmp).abs() >= 1e-3).nonzero(as_tuple=True)[0])
                print("%d: values " % rank, tmp[((sumdata - tmp).abs() >= 1e-3).nonzero(as_tuple=True)[0]])
                print("%d: my inputs " % rank, data[((sumdata - tmp).abs() >= 1e-3).nonzero(as_tuple=True)[0]])
                raise RuntimeError("wrong sum")
            tmp.zero_()
            tmp = data.clone()
            #os._exit(1)

        start = time.time()
        sum = 0
        loopcount = 1000
        for _ in range(loopcount):
            #tmp.copy_(data)
            #tmp2.copy_(data2)
            dist.all_reduce(tmp)
            #dist.all_reduce(tmp2)
            #si = tmp.sum().item() + tmp2.sum().item()
            ##si = tmp.sum().item()
            #print("-> %f" % si)
            #sum += si
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        if rank == 0:
            print("sum: %f" % sum)
        t = time.time() - start
        if rank == 0:
            #print("time: %g, %g/s" % (t, loopcount / t))
            print("time: %g, %g/s  %gG/s" % (t, loopcount / t, data.numel() * data.element_size() / size / 1024 / 1024 / 1024 * loopcount / t))

        data = tmp
        #print("result: ", data)

    #dist.barrier()
    torch.cuda.synchronize()

    # correct = torch.zeros_like(data)

    # torch.manual_seed(42)

    # for i in range(dist.get_world_size()):
    #     correct += torch.randn(1024 * 1024 * 100).cuda()
    
    # #print("correct: ", correct)
    # if not data.equal(correct):
    #     print("bad!!")
    # diff = data - correct
    # print(diff.abs().max())


if 1 == 1:
    f("moodist")
    sys.exit(0)

#for n in ("moolib", "nccl", "moolib", "nccl", "moolib", "nccl", "moolib", "nccl"):
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


