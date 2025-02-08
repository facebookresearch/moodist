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

queue = moodist.Queue(group, location=range(group.size()))
ready_queue = moodist.Queue(group, 0)

rank = group.rank()
size = group.size()

torch.manual_seed(42 + rank)

random.seed(42 + rank)

for i in range(size):
    torch.distributed.barrier()
    if i == rank:
        print("rank %d hello" % rank)

torch.distributed.barrier()

start = time.time()

# if rank == 0:
#     for i in range(160000):
#         queue.put_object(i)

torch.distributed.barrier()

counter = 0
# def getint():
#     global counter
#     r = counter
#     counter += 1
#     return r

start = time.time()
nitems = 0
for iteration in range(16000):
    try:
        if rank == 0:
            queue.put_object(counter)
            print("put %d ok" % counter)
            counter += 1
        item = queue.get_object(block=True)
        # if nitems >= 160000:
        #     raise Empty
        # item = getint()
        nitems += 1
    except Empty:
        print("rank %d: queue is empty!" % rank)
        break
    # if rank == 1:
    #     time.sleep(0.5)

    for r in range(size):
        #torch.distributed.barrier()
        if rank == r:
            print("rank %d got: %s" % (r, item))
        #torch.distributed.barrier()

torch.distributed.barrier()

for r in range(size):
    torch.distributed.barrier()
    if rank == r:
        t = time.time() - start
        print("%d items in %g, %g/s" % (nitems, t, nitems / t))

torch.distributed.barrier()
