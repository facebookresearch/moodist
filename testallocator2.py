import torch
import moodist
import time
import os

import random

moodist.enable_cuda_allocator()

random.seed(42)

def f():
    l = []
    for i in range(600):
        t = torch.empty(random.randrange(1024 * 1024 * 58), device="cuda")
        l.append(t)
    for _ in range(64):
        for i in range(256):
            l.pop(random.randrange(len(l)))
        for i in range(256):
            t = torch.empty(random.randrange(1024 * 1024 * 58), device="cuda")
            l.append(t)
    # n = sum(t.numel() for t in l)
    # print(n)
    # print("%gG" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    #print(l)

# for _ in range(100):
#     f(torch.randn((16, 4096), device="cuda"))

s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

for i in range(4):
    # if i == 10:
    #     moodist.enable_cuda_allocator()
    start = time.time()

    grad_sum = 0

    with torch.cuda.stream(s0):
        for _ in range(32):
            f()
    # with torch.cuda.stream(s1):
    #     for _ in range(32):
    #         f()
    # with torch.cuda.stream(s2):
    #     for _ in range(32):
    #         f()

    torch.cuda.synchronize()
    print(time.time() - start)

    print("%gG" % (torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
    # os.system("nvidia-smi")

# x = torch.zeros(8, device="cuda")

# print(x)

# # x += torch.randn(8).cuda()

# # print(x)

torch.cuda.synchronize()


os.system("nvidia-smi")
