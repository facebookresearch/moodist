import torch
import torch.distributed
import moodist

import os
import time
import random
import weakref

moodist.enable_cpu_allocator()

#print(torch.randn(4))

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")

group = torch.distributed.new_group()

queue = moodist.Queue(group, location=0)
ready_queue = moodist.Queue(group, 0)


rank = group.rank()
size = group.size()

torch.manual_seed(42 + rank)

random.seed(42 + rank)

for i in range(size):
    torch.distributed.barrier()
    if i == rank:
        print("rank %d hello" % rank)

start = time.time()

#queue.put_object(rank)

for iteration in range(8):

    for i in range(size):
        torch.distributed.barrier()
        if i == rank:
            n = random.randrange(1024 * 1024 * 200)
            #n = 1024 * 1024 * 20
            t = torch.randn(n)
            bytes = t.element_size() * t.numel() * 2
            start = time.time()
            for i in range(2):
                t.clone()
            t = time.time() - start
            print("%d: clone took %g, %gG/s" % (rank, t, bytes / t / 1024 / 1024 / 1024))

            t = torch.randn(n)
            bytes = t.element_size() * t.numel() * 2
            start = time.time()
            for i in range(2):
                queue.put_tensor(t)
                queue.get_tensor()
            t = time.time() - start
            print("%d: put-get took %g, %gG/s" % (rank, t, bytes / t / 1024 / 1024 / 1024))
        torch.distributed.barrier()

    if rank == 0:
        for i in range(size):
            queue.put_tensor(torch.randn(1024 * 1024 * 20))
        
        for i in range(size):
            ready_queue.put_object(True)

    #time.sleep(2)

    ready_queue.get_object()

    start = time.time()

    foo: torch.Tensor = queue.get_tensor()

    t = time.time() - start
    bytes = foo.element_size() * foo.numel()

    for i in range(size):
        torch.distributed.barrier()
        if i == rank:
            print("%d: get took %g, %gG/s" % (rank, t, bytes / t / 1024 / 1024 / 1024))
            print("rank %d got: %s" % (rank, foo.sum()))

    torch.distributed.barrier()

for i in range(size):
    torch.distributed.barrier()
    if i == rank:
        print("rank %d bye" % rank)

        print("done in %g" % (time.time() - start))
