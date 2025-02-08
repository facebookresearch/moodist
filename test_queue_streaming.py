import torch
import torch.distributed
import moodist

import os
import time

import weakref

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")


group = torch.distributed.new_group()

queue = group.Queue(location=0)
#queue2 = group.Queue(location=1)

for i in range(100):
    group.Queue(location=0)

rank = group.rank()

torch.manual_seed(42 + rank)

torch.randn((1, 4))

torch.distributed.barrier()
print("rank %d hello" % rank)

start = time.time()

# if rank == 1:
#     # for i in range(1600):
#     #     item = queue.get()
#     queue2.put(torch.randn(1))

print("woo")
if True:
    #time.sleep(1)
    handles = []
    for i in range(1600):
        #time.sleep(1)
        t = torch.randn((1 + i % 16, 2 + i % 16))
        #print("rank %d put: %s" % (rank, t))
        handles.append(queue.put(t))
    for v in handles:
        v.wait()
# if rank == 1:
#     queue2.put(torch.randn(1))
print("waa")
# if rank == 0:
if True:
    #queue2.get()
    # if rank == 0:
    #     time.sleep(2)
    for i in range(1600):
        item = queue.get()
        #print("rank %d got: %s" % (rank, item))

# if rank == 1:
#     #time.sleep(1)
#     handles = []
#     for i in range(2000):
#         #time.sleep(1)
#         t = torch.randn((1 + i % 16, 2 + i % 16))
#         #print("rank %d put: %s" % (rank, t))
#         handles.append(queue.put(t))
#     for v in handles:
#         v.wait()

# if rank == 0:
#     # if rank == 0:
#     #     time.sleep(2)
#     for i in range(2000):
#         item = queue.get()
#         #print("rank %d got: %s" % (rank, item))

torch.distributed.barrier()
print("rank %d bye" % rank)

print("done in %g" % (time.time() - start))
