import torch
import torch.distributed
import moodist

import os
import time

import weakref

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")

moodist.enable_cpu_allocator()

group = torch.distributed.new_group()

queue = moodist.Queue(group, location=0, streaming=False)

rank = group.rank()

for iteration in range(4):
    if rank == 0:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        handles = []
        for i in range(160):
            t = torch.randn((128, 1 + i, 2 + i), device="cpu")
            # def ff():
            #     print("waa finalizer called")
            # weakref.finalize(t, ff)
            print(" put ", i, t.shape)
            handles.append(queue.put_tensor(t))
            # t.zero_()
            # print("post-put qsize %d" % queue.qsize())
            # print("put %d - %d bytes" % (i, t.element_size() * t.numel()))
        start = time.time()
        for h in handles:
            h.wait()
        print("wait took %gs" % (time.time() - start))

    if rank == 1:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # queue.put(torch.randn((1, 2)))

        for i in range(160):
            t = torch.randn((128, 1 + i, 2 + i), device="cpu").cpu()
            # print("queue size is %d" % queue.qsize())
            item = queue.get_tensor()

            print(" got ", i, item.shape)

            # print(t - item)
            assert torch.allclose(t, item)
            # print("got: %s" % str(item))

            # time.sleep(1)
            # print("%d ok" % i)
            del item

    torch.distributed.barrier()

    moodist.cpu_allocator_debug()

    torch.distributed.barrier()

    # if iteration == 1:
    #     break

torch.distributed.barrier()
print("rank %d bye" % rank)
