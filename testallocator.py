
import torch
import moodist
import time
import os

moodist.enable_cuda_allocator()

lin = torch.nn.Linear(4096, 4096).cuda()

x = torch.randn(4096).cuda()

for _ in range(1000):
    x = lin(x).relu()

for i in range(20):
    # if i == 10:
    #     moodist.enable_cuda_allocator()
    start = time.time()
    
    x = torch.randn((16, 4096), device="cuda")

    for _ in range(1000):
        x = lin(x).relu()

    #print(x)
    x.sum().backward()

    torch.cuda.synchronize()
    print(time.time() - start)
    
    os.system("nvidia-smi")

# x = torch.zeros(8, device="cuda")

# print(x)

# # x += torch.randn(8).cuda()

# # print(x)

torch.cuda.synchronize()

