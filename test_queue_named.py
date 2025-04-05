import torch
import torch.distributed
import moodist

import os
import time

import weakref

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")


group = torch.distributed.new_group()

group2 = torch.distributed.new_group()

print("group name is ", group.moodist_name())

queue = moodist.Queue(group, location=0, name="foobar")
queue2 = moodist.Queue(group, location=0, name="foobar 2")

print("queue name is ", queue.impl.name())

rank = group.rank()

torch.manual_seed(42 + rank)

torch.randn((1, 4))

torch.distributed.barrier()
print("rank %d hello" % rank)

start = time.time()


def mem():
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


begin_mem = mem()
for i in range(10000):
    if rank == 0:
        queue.put_object("hello world from 0")
        queue.put_object(queue2)
        queue2.get_object()
        # q.put_object("barbar data yeah")
        # queue.put_object(q)
        # q = moodist.Queue(group, location=1, name="barbar reponse queue %d" % i)
        # queue.put_object(q)
        # print("response? ", q.get_object())
    if rank == 1:
        queue.get_object()
        queue.get_object().put_object("hello world from 1")
        q = moodist.Queue(group2, location=1, name="barbar %d" % 2)
        # torch.distributed.destroy_process_group(group2)
        # del group2
        # print("get_object: ", queue.get_object())
        # print("bar? ", queue.get_object().get_object())
        # queue.get_object().put_object(" this is teh response!")
end_mem = mem()

for i in range(group.size()):
    torch.distributed.barrier()
    if i == rank:
        print("mem: %g -> %g, growth: %g" % (begin_mem, end_mem, end_mem - begin_mem))

        print("rank %d bye" % rank)

        print("done in %g" % (time.time() - start))
