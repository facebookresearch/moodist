
import os
import subprocess
import time
import random

import torch

import moodist

moodist.enable_cpu_allocator()

data = torch.randn(1024 * 1024 * 3).to("cuda:0")
datax = data.clone()
#tmp = data.clone().cpu().pin_memory(device="cuda:0")
tmp = data.clone().cpu()

torch.distributed.init_process_group(backend="moodist")

group = torch.distributed.new_group()

#data2 = data.clone().to("cuda:1")
#tmp2 = data2.clone().cpu().pin_memory(device="cuda:1")

#print(data)
#print(data2)

for i in range(100):
    #tmp.copy_(data, non_blocking=True)
    moodist.cuda_copy(tmp, data)
    #group.copy(tmp, data).wait()

torch.cuda.synchronize("cuda:0")

start = time.time()
for i in range(100):
    # source = torch.randn(random.randrange(1024 * 1024 * 4))
    # destination = torch.empty_like(source, device="cuda")
    # moodist.cuda_copy(destination, source)
    #datax.copy_(data, non_blocking=True)
    #data.copy_(data2, non_blocking=True)
    #data2.copy_(data, non_blocking=True)
    #tmp.copy_(data, non_blocking=True)
    moodist.cuda_copy(tmp, data)
    #group.copy(tmp, data).wait()
    # torch.cuda.synchronize(device="cuda:1")
    #tmp2.copy_(data2, non_blocking=True)
    # torch.cuda.synchronize(device="cuda:0")
    torch.cuda.synchronize(device="cuda:0")
torch.cuda.synchronize("cuda:0")
#torch.cuda.synchronize("cuda:1")

i += 1

bytes = data.numel() * data.element_size() * i * 1

print("bytes is %d" % bytes)

t = time.time() - start
print("%d in %g, %g/s  %gM/s" % (i, t, i/t, bytes/t/1024/1024))


