import torch
import torch.distributed
import moodist
import time
import os

moodist.enable_cpu_allocator()
moodist.enable_cuda_allocator()

lin1 = torch.nn.Linear(4096, 8192).cuda()
lin1a = torch.nn.Linear(4096, 8192).cuda()
lin1b = torch.nn.Linear(4096, 8192).cuda()
lin2 = torch.nn.Linear(8192, 16).cuda()
lin3 = torch.nn.Linear(16, 4096).cuda()
lin3a = torch.nn.Linear(16, 4096).cuda()
lin3b = torch.nn.Linear(16, 4096).cuda()
lin3c = torch.nn.Linear(16, 4096).cuda()

x = torch.randn(4096).cuda()

s0 = torch.cuda.Stream()
e0 = torch.cuda.Event()


def f(x):
    sums = []
    zz = []
    for _ in range(32):
        # print(_)
        # torch.cuda.synchronize()
        skip = x
        x = lin1(x).relu() + lin1a(x) + lin1b(x.square())
        x = x.clone()
        e0.record()
        with torch.cuda.stream(s0):
            # if True:
            # print("x numel is %d" % x.numel())
            x2 = torch.zeros_like(x)
            x3 = x2.sum()
            # torch.cuda.synchronize()
            x4 = x2.sum()
            #x5 = [x4.clone() for _ in range(16)]
            # for _ in range(128):
            #     x5 = x4.clone()
            for i in range(16):
                x5 = x4 * x4 * x4 * x4 * x4 * x4
            del x4
            del x5
            # print(x3)
            e0.record()
        y = x.clone()
        # del x
        # [torch.randn_like(y) for _ in range(64)]
        x = y
        x = lin2(x)
        y = x
        x = lin3(x).relu() + lin3a(x) + lin3b(-x) + lin3c(x)
        # x += lin3(x2)
        x = x * x.softmax(-1) + skip

        e0.wait()

        x2.record_stream(torch.cuda.current_stream())
        x3.record_stream(torch.cuda.current_stream())

        # print("x2 is %#x" % x2.data_ptr())
        # print("x3 is %#x" % x3.data_ptr())

        sums.append((x2.sum(), x3))

        # zz.append(x2)
        # zz.append(x3)

        del x2
        del x3

        # print(sums[-1])
        # print("x4 ", x4)

        # assert torch.allclose(sums[-1][0], sums[-1][1])

        # assert torch.allclose(x4, y, rtol=1e-3, atol=1e-4)
        # assert torch.allclose(x3, y, rtol=1e-3, atol=1e-4)
        # assert torch.allclose(x2, y, rtol=1e-3, atol=1e-4)
    for t in sums:
        a, b = t
        if not torch.allclose(a, b):
            print(a, b)
        assert torch.allclose(a, b)
    return x


# for _ in range(100):
#     f(torch.randn((16, 4096), device="cuda"))

torch.distributed.init_process_group(backend="moodist")

for i in range(40):
    # if i == 10:
    #     moodist.enable_cuda_allocator()
    start = time.time()

    grad_sum = 0

    for _ in range(2):
        x = 0
        for bs in range(1, 512, 32):
            inp = torch.randn((bs, 4096), device="cuda", requires_grad=True)
            x = x + f(inp).sum()
        x.sum().backward()
        lsum = inp.grad.sum()
        torch.distributed.all_reduce(lsum)
        grad_sum += lsum

    torch.cuda.synchronize()
    print(time.time() - start)

    print("grad_sum: %g" % grad_sum)

    print("%gG" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    # os.system("nvidia-smi")

# x = torch.zeros(8, device="cuda")

# print(x)

# # x += torch.randn(8).cuda()

# # print(x)

torch.cuda.synchronize()


os.system("nvidia-smi")
