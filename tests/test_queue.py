"""
Tests for moodist Queue

Queue provides inter-rank communication with put/get semantics.
Each queue has a "location" (the rank that owns the storage).
"""

from queue import Empty

import torch
import moodist
from framework import TestContext, test, create_process_group


@test
def test_queue_basic_put_get(ctx: TestContext):
    """Test basic put/get on a queue owned by rank 0."""
    pg = create_process_group(ctx)

    # Create queue on rank 0
    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 0:
        # Rank 0 puts a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        queue.put_tensor(tensor)

    ctx.barrier()

    if ctx.rank == 0:
        # Rank 0 gets it back (result may be on CPU)
        result = queue.get_tensor()
        expected = torch.tensor([1.0, 2.0, 3.0])
        ctx.assert_true(torch.equal(result.cpu(), expected), "tensor mismatch")


@test
def test_queue_cross_rank_put_get(ctx: TestContext):
    """Test put from one rank, get from another."""
    if ctx.world_size < 2:
        return  # Need at least 2 ranks

    pg = create_process_group(ctx)

    # Queue located on rank 0
    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 1:
        # Rank 1 puts a tensor
        tensor = torch.tensor([42.0, 43.0, 44.0], device="cuda")
        queue.put_tensor(tensor)

    ctx.barrier()

    if ctx.rank == 0:
        # Rank 0 gets it (result may be on CPU)
        result = queue.get_tensor()
        expected = torch.tensor([42.0, 43.0, 44.0])
        ctx.assert_true(torch.equal(result.cpu(), expected), "tensor mismatch")


@test
def test_queue_multiple_puts(ctx: TestContext):
    """Test multiple puts and gets maintain order."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 0:
        # Put 5 tensors
        for i in range(5):
            tensor = torch.tensor([float(i)], device="cuda")
            queue.put_tensor(tensor)

    ctx.barrier()

    if ctx.rank == 0:
        # Get them back in order (results may be on CPU)
        for i in range(5):
            result = queue.get_tensor()
            expected = torch.tensor([float(i)])
            ctx.assert_true(torch.equal(result.cpu(), expected), f"tensor {i} mismatch")


@test
def test_queue_qsize(ctx: TestContext):
    """Test qsize returns correct count."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 0:
        ctx.assert_equal(queue.qsize(), 0, "initial size should be 0")

        for i in range(3):
            tensor = torch.tensor([float(i)], device="cuda")
            queue.put_tensor(tensor)

        # Synchronize CUDA stream to ensure puts complete before checking qsize
        torch.cuda.synchronize()
        ctx.assert_equal(queue.qsize(), 3, "size should be 3 after 3 puts")

        queue.get_tensor()
        ctx.assert_equal(queue.qsize(), 2, "size should be 2 after 1 get")


@test
def test_queue_empty_check(ctx: TestContext):
    """Test empty() method."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 0:
        ctx.assert_true(queue.empty(), "queue should be empty initially")

        tensor = torch.tensor([1.0], device="cuda")
        queue.put_tensor(tensor)

        # Synchronize CUDA stream to ensure put completes before checking empty
        torch.cuda.synchronize()
        ctx.assert_false(queue.empty(), "queue should not be empty after put")

        queue.get_tensor()
        ctx.assert_true(queue.empty(), "queue should be empty after get")


@test
def test_queue_nonblocking_get(ctx: TestContext):
    """Test non-blocking get raises Empty when queue is empty."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 0:
        # Non-blocking get on empty queue should raise Empty
        ctx.assert_raises(Empty, queue.get_tensor, block=False)


@test
def test_queue_put_object_get_object(ctx: TestContext):
    """Test put_object/get_object with Python objects."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    if ctx.rank == 0:
        # Put a Python dict
        obj = {"key": "value", "number": 42, "list": [1, 2, 3]}
        queue.put_object(obj)

    ctx.barrier()

    if ctx.rank == 0:
        result = queue.get_object()
        expected = {"key": "value", "number": 42, "list": [1, 2, 3]}
        ctx.assert_equal(result, expected, "object mismatch")


@test
def test_queue_large_tensor(ctx: TestContext):
    """Test queue with larger tensors."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    original = None
    if ctx.rank == 0:
        # 1MB tensor
        tensor = torch.randn(256 * 1024, device="cuda", dtype=torch.float32)
        original = tensor.cpu().clone()  # Keep CPU copy for comparison
        queue.put_tensor(tensor)

    ctx.barrier()

    if ctx.rank == 0:
        result = queue.get_tensor()
        ctx.assert_true(torch.equal(result.cpu(), original), "large tensor mismatch")


@test
def test_queue_from_all_ranks(ctx: TestContext):
    """Test all ranks putting to the same queue."""
    pg = create_process_group(ctx)

    queue = moodist.Queue(pg, location=0)

    # Each rank puts its rank number
    tensor = torch.tensor([float(ctx.rank)], device="cuda")
    queue.put_tensor(tensor)

    ctx.barrier()

    if ctx.rank == 0:
        # Rank 0 collects all
        received = []
        for _ in range(ctx.world_size):
            t = queue.get_tensor()
            received.append(int(t.item()))

        # All ranks should be present (order may vary)
        ctx.assert_equal(sorted(received), list(range(ctx.world_size)), "missing ranks")
