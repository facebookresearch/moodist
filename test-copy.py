
import os
import subprocess
import time

import torch


data = torch.randn(1024 * 1024 * 3).to("cuda:0")
datax = data.clone()
tmp = data.clone().cpu().pin_memory(device="cuda:0")

data2 = data.clone().to("cuda:1")
tmp2 = data2.clone().cpu().pin_memory(device="cuda:1")

print(data)
print(data2)


start = time.time()
for i in range(100):
    #datax.copy_(data, non_blocking=True)
    #data.copy_(data2, non_blocking=True)
    #data2.copy_(data, non_blocking=True)
    tmp.copy_(data, non_blocking=True)
    # torch.cuda.synchronize(device="cuda:1")
    #tmp2.copy_(data2, non_blocking=True)
    # torch.cuda.synchronize(device="cuda:0")
    torch.cuda.synchronize(device="cuda:0")
torch.cuda.synchronize("cuda:0")
torch.cuda.synchronize("cuda:1")

i += 1

bytes = data.numel() * data.element_size() * i * 1

t = time.time() - start
print("%d in %g, %g/s  %gM/s" % (i, t, i/t, bytes/t/1024/1024))


