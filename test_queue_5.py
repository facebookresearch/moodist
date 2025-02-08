import torch
import torch.distributed
import moodist

import os
import time
import random
import weakref
from queue import Empty

moodist.enable_cpu_allocator()

# print(torch.randn(4))

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")

group = torch.distributed.new_group()

#queue = moodist.Queue(group, location=0)
queue = moodist.Queue(group, location=range(group.size()))
ready_queue = moodist.Queue(group, 0)

rank = group.rank()
size = group.size()

iterations = 16

torch.distributed.barrier()

for i in range(iterations):
    with queue.transaction() as t:
        t.put_object((rank, 1))
        t.put_object((rank, 2))
        torch.distributed.barrier()
        t.put_object((rank, 3))
        t.put_tensor(torch.randn(4, device="cuda"))
    # queue.put_object((rank, 1))
    # torch.distributed.barrier()
    # queue.put_object((rank, 2))
    # queue.put_object((rank, 3))

for i in range(3 * iterations):
    for r in range(size):
        torch.distributed.barrier()
        if r == rank:
            a = queue.get_object(timeout=1)
            b = queue.get_object()
            c = queue.get_object()
            t = queue.get_tensor()
            assert a[0] == b[0] == c[0]
            assert b[1] == a[1] + 1
            assert c[1] == b[1] + 1
            print("rank %d got %s %s %s %s OK" % (rank, a, b, c, t))
            #print("rank %d got: %s" % (rank, queue.get_object()))
        torch.distributed.barrier()


torch.distributed.barrier()
