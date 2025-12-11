"""
Tests for moodist allocators (CUDA and CPU)

These tests verify that the moodist allocators work correctly:
- Basic allocation and deallocation
- Tensor operations on allocated memory
- Integration with PyTorch operations
"""

import gc

import torch

import moodist
from framework import TestContext, test


@test
def test_cpu_allocator_basic(ctx: TestContext):
    """Test basic CPU allocator functionality."""
    moodist.enable_cpu_allocator()

    # Allocate a tensor - should use moodist allocator
    t = torch.zeros(1000, dtype=torch.float32)

    ctx.assert_equal(t.device.type, "cpu")
    ctx.assert_equal(t.numel(), 1000)
    ctx.assert_equal(t.sum().item(), 0.0)

    # Modify and verify
    t.fill_(42.0)
    ctx.assert_equal(t.sum().item(), 42.0 * 1000)

    # Allocate more tensors
    t2 = torch.ones(500, dtype=torch.float32)
    ctx.assert_equal(t2.sum().item(), 500.0)

    # Clean up
    del t, t2
    gc.collect()


@test
def test_cpu_allocator_various_sizes(ctx: TestContext):
    """Test CPU allocator with various tensor sizes."""
    moodist.enable_cpu_allocator()

    sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]

    for size in sizes:
        t = torch.zeros(size, dtype=torch.float32)
        ctx.assert_equal(t.numel(), size)
        t.fill_(1.0)
        ctx.assert_equal(t.sum().item(), float(size))
        del t

    gc.collect()


@test
def test_cpu_allocator_various_dtypes(ctx: TestContext):
    """Test CPU allocator with various data types."""
    moodist.enable_cpu_allocator()

    dtypes = [
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ]

    for dtype in dtypes:
        t = torch.zeros(100, dtype=dtype)
        ctx.assert_equal(t.numel(), 100)
        ctx.assert_equal(t.dtype, dtype)
        del t

    gc.collect()


@test
def test_cpu_allocator_multidimensional(ctx: TestContext):
    """Test CPU allocator with multidimensional tensors."""
    moodist.enable_cpu_allocator()

    shapes = [
        (10, 10),
        (100, 100),
        (10, 20, 30),
        (2, 3, 4, 5),
    ]

    for shape in shapes:
        t = torch.zeros(shape, dtype=torch.float32)
        ctx.assert_equal(list(t.shape), list(shape))
        del t

    gc.collect()


@test
def test_cpu_allocator_operations(ctx: TestContext):
    """Test that PyTorch operations work with moodist-allocated tensors."""
    moodist.enable_cpu_allocator()

    a = torch.randn(100, 100)
    b = torch.randn(100, 100)

    # Matrix multiply
    c = torch.mm(a, b)
    ctx.assert_equal(c.shape, (100, 100))

    # Element-wise operations
    d = a + b
    e = a * b
    f = torch.relu(a)

    # Reductions
    s = a.sum()
    m = a.mean()

    del a, b, c, d, e, f
    gc.collect()


@test
def test_cuda_allocator_basic(ctx: TestContext):
    """Test basic CUDA allocator functionality."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cuda_allocator()

    # Allocate a tensor - should use moodist allocator
    t = torch.zeros(1000, dtype=torch.float32, device='cuda')

    ctx.assert_equal(t.device.type, "cuda")
    ctx.assert_equal(t.numel(), 1000)
    ctx.assert_equal(t.sum().item(), 0.0)

    # Modify and verify
    t.fill_(42.0)
    ctx.assert_equal(t.sum().item(), 42.0 * 1000)

    # Allocate more tensors
    t2 = torch.ones(500, dtype=torch.float32, device='cuda')
    ctx.assert_equal(t2.sum().item(), 500.0)

    # Clean up
    del t, t2
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_allocator_various_sizes(ctx: TestContext):
    """Test CUDA allocator with various tensor sizes."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cuda_allocator()

    sizes = [1, 10, 100, 1000, 10000, 100000, 1000000]

    for size in sizes:
        t = torch.zeros(size, dtype=torch.float32, device='cuda')
        ctx.assert_equal(t.numel(), size)
        t.fill_(1.0)
        ctx.assert_equal(t.sum().item(), float(size))
        del t

    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_allocator_various_dtypes(ctx: TestContext):
    """Test CUDA allocator with various data types."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cuda_allocator()

    dtypes = [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ]

    for dtype in dtypes:
        t = torch.zeros(100, dtype=dtype, device='cuda')
        ctx.assert_equal(t.numel(), 100)
        ctx.assert_equal(t.dtype, dtype)
        del t

    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_allocator_operations(ctx: TestContext):
    """Test that PyTorch operations work with moodist-allocated CUDA tensors."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cuda_allocator()

    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')

    # Matrix multiply
    c = torch.mm(a, b)
    ctx.assert_equal(c.shape, (100, 100))

    # Element-wise operations
    d = a + b
    e = a * b
    f = torch.relu(a)

    # Reductions
    s = a.sum()
    m = a.mean()

    del a, b, c, d, e, f
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_allocator_cpu_cuda_transfer(ctx: TestContext):
    """Test transfers between CPU and CUDA with moodist allocators."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cpu_allocator()
    moodist.enable_cuda_allocator()

    # Create on CPU
    cpu_t = torch.randn(1000)

    # Transfer to CUDA
    cuda_t = cpu_t.cuda()
    ctx.assert_equal(cuda_t.device.type, "cuda")

    # Transfer back to CPU
    cpu_t2 = cuda_t.cpu()
    ctx.assert_equal(cpu_t2.device.type, "cpu")

    # Verify data integrity
    ctx.assert_true(torch.allclose(cpu_t, cpu_t2))

    del cpu_t, cuda_t, cpu_t2
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_allocator_reallocation(ctx: TestContext):
    """Test that memory is properly reused after deallocation."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cuda_allocator()

    # Allocate and deallocate multiple times
    for _ in range(10):
        t = torch.zeros(1000000, dtype=torch.float32, device='cuda')
        t.fill_(1.0)
        del t

    torch.cuda.synchronize()
    gc.collect()

    # Should still be able to allocate
    t = torch.zeros(1000000, dtype=torch.float32, device='cuda')
    ctx.assert_equal(t.numel(), 1000000)

    del t
    torch.cuda.synchronize()
    gc.collect()
