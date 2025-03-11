import torch
import torch.distributed
import moodist

import os
import time
import random
import weakref
import gc

gc.disable()

torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))

torch.distributed.init_process_group("moodist")

moodist.enable_cpu_allocator()

group = torch.distributed.new_group()

rank = group.rank()
size = group.size()

location = size - 1
#location = 0

n_consumers = size // 2

consumer = rank < n_consumers
n_producers = size - n_consumers

assert n_producers > 0

n_tickets = 3

queue = [moodist.Queue(group, location=i) for i in range(n_consumers)]
wait_queue = moodist.Queue(group, location=location)

finished = moodist.Queue(group, location=range(size))

ticket_queue = [moodist.Queue(group, location=location) for _ in range(n_tickets)]

n_iters = 3200

n_iters = (n_iters + n_producers - 1) // n_producers * n_producers

assert n_iters >= n_tickets

for iteration in range(400):
    if consumer:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        post_counter = 0
        if consumer:
            for i in range(n_tickets):
                ticket_queue[i].put_object((i, rank)).wait()
                wait_queue.put_object((i, rank))
                post_counter += 1

        visited = set()
        prevn = {}
        for i in range(n_iters):
            ticket, source_rank, item = queue[rank].get_object()

            print(" got %s %s %s\n" % (ticket, source_rank, item), end=None)

            #time.sleep(random.random())

            if source_rank not in prevn:
                prevn[source_rank] = -1

            assert isinstance(item, int)
            assert item > prevn[source_rank]
            prevn[source_rank] = item

            key = (source_rank, item)
            assert key not in visited
            visited.add(key)

            if post_counter < n_iters:
                ticket_queue[ticket[0]].put_object(ticket).wait()
                wait_queue.put_object(ticket)
                post_counter += 1
                
            
            if rank == location:
                print("wait queue size is %d" % (wait_queue.qsize()))

    else:
        for i in range(n_iters // n_producers * n_consumers):
            while not finished.empty():
                finished_rank = finished.get_object()
                print(
                    "rank %d finished, %d is on iteration %d" % (finished_rank, rank, i)
                )
            #time.sleep(random.random())
            # ticket = ticket_queue.get_object()
            try:
                wait_ticket = wait_queue.get_object(block=False)
                #print("no block yey")
            except moodist.Empty:
                start = time.monotonic()
                while True:
                    try:
                        wait_ticket = wait_queue.get_object(timeout=0.01)
                        break
                    except moodist.Empty:
                        pass
                t = time.monotonic() - start
                if t >= 0.1:
                    print("blocked for %gs" % t)

            ticket = None
            for ti in range(n_tickets):
                try:
                    ticket = ticket_queue[ti].get_object(block=False)
                    break
                except moodist.Empty:
                    pass
            assert ticket is not None
            print("%d: ticket %s, i %d\n" % (rank, ticket, i), end=None)

            queue[ticket[1]].put_object((ticket, rank, i))

        finished.put_object(rank).wait()

    # moodist.cpu_allocator_debug()

    if consumer:
        assert queue[rank].empty()
        # time.sleep(1)
        # assert queue[rank].empty()

    group.barrier()
    # time.sleep(0.25)

    gc.collect()

    while not finished.empty():
        finished.get_object()

    if rank == location:
        print("ticket queue size is %d/%d" % (wait_queue.qsize(), 0))
        assert wait_queue.qsize() == 0
        for tq in ticket_queue:
            assert tq.qsize() == 0
    if consumer:
        assert queue[rank].empty()

    group.barrier()

    assert finished.empty()

    print("%d: iteration %d done" % (rank, iteration))

    # if iteration == 1:
    #     break

torch.distributed.barrier()
print("rank %d bye" % rank)
