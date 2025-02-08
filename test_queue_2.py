import torch
import torch.distributed
import moodist

import os
import time

import weakref

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")


group = torch.distributed.new_group()

queue = moodist.Queue(group, location=0)
#queue2 = group.Queue(location=1)


rank = group.rank()
size = group.size()

torch.manual_seed(42 + rank)


for i in range(size):
    torch.distributed.barrier()
    if i == rank:
        print("rank %d hello" % rank)

start = time.time()

#queue.put_object(rank)

if rank == 0:
    for i in range(size):
        queue.put_object(i)

time.sleep(2)

foo = None if rank == 0 else queue.get_object()

for i in range(size):
    torch.distributed.barrier()
    if i == rank:
        print("rank %d got: %s" % (rank, foo))

torch.distributed.barrier()

for i in range(size):
    torch.distributed.barrier()
    if i == rank:
        print("rank %d bye" % rank)

        print("done in %g" % (time.time() - start))
