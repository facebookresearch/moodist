import torch
import torch.distributed
import moodist

import os
import time
import random
import weakref
from queue import Empty

moodist.enable_cpu_allocator()

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")

group = torch.distributed.new_group()

rank = group.rank()
size = group.size()

if False:

    if rank == 0:
        a = torch.randn(4, 2, 1)

        print("a is: %s" % (a,))

        locals = [(0, a), (2, a)]
    elif rank == 1:
        a = torch.randn(16, 2, 1)

        print("a is: %s" % (a,))

        locals = [(1, a)]
    else:
        locals = []

    result = group.cat(locals).result()

    print("result is: %s" % (result,))
else:

    torch.manual_seed(42 + rank)
    #locals = [(rank, torch.randn(1024 * 1024 * 8))]
    locals = [(rank, torch.randn(random.randrange(1024 * 1024))) for _ in range(100)]

    # c = []
    # for i in range(size):
    #     torch.manual_seed(42 + i)
    #     t = torch.randn(1024 * 1024 * 8)
    #     c.append(t)
    # c = torch.cat(c)

    # for i in range(100):
    #     result = group.cat(locals[i]).result()
    #     #assert torch.equal(result, c)
    
    print("rank %d warmup done" % rank)

    for _ in range(8):
        start = time.monotonic()
        ops = []
        for i in range(100):
            #result = group.cat(locals).result()
            ops.append(group.cat([locals[i]]))
            #assert torch.equal(result, c)
        
        total_bytes = 0
        for op in ops:
            op.wait()
            tensor = op.result()
            total_bytes += tensor.itemsize * tensor.numel()
        
        t = time.monotonic() - start
        g = total_bytes * 100 / 1024 / 1024 / 1024

        print("rank %d took %gs, %gG/s" % (rank, t, g / t))

torch.distributed.barrier()
