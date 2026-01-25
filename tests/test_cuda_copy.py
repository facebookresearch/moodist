"""
Tests for moodist.cuda_copy

cuda_copy copies tensor data using cuMemcpyAsync with host memory registration.
When the moodist CPU allocator is enabled, host memory is registered with CUDA
for efficient DMA transfers between CPU and GPU.
"""

import gc

import torch

import moodist
from framework import TestContext, test


@test
def test_cuda_copy_cuda_to_cuda(ctx: TestContext):
    """Test CUDA to CUDA copy."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)

    src = torch.arange(100, device="cuda", dtype=torch.float32)
    dst = torch.zeros(100, device="cuda", dtype=torch.float32)

    moodist.cuda_copy(dst, src)
    torch.cuda.synchronize()

    ctx.assert_true(torch.equal(dst, src))

    del src, dst
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_cpu_to_cuda(ctx: TestContext):
    """Test CPU to CUDA copy with moodist CPU allocator."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cpu_allocator()

    src = torch.arange(100, dtype=torch.float32)  # CPU tensor
    dst = torch.zeros(100, device="cuda", dtype=torch.float32)

    moodist.cuda_copy(dst, src)
    torch.cuda.synchronize()

    ctx.assert_true(torch.equal(dst.cpu(), src))

    del src, dst
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_cuda_to_cpu(ctx: TestContext):
    """Test CUDA to CPU copy with moodist CPU allocator."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cpu_allocator()

    src = torch.arange(100, device="cuda", dtype=torch.float32)
    dst = torch.zeros(100, dtype=torch.float32)  # CPU tensor

    moodist.cuda_copy(dst, src)
    torch.cuda.synchronize()

    ctx.assert_true(torch.equal(dst, src.cpu()))

    del src, dst
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_various_dtypes(ctx: TestContext):
    """Test cuda_copy with various data types."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cpu_allocator()

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
    ]

    for dtype in dtypes:
        # CPU to CUDA
        src = torch.ones(100, dtype=dtype)
        dst = torch.zeros(100, device="cuda", dtype=dtype)
        moodist.cuda_copy(dst, src)
        torch.cuda.synchronize()
        ctx.assert_true(torch.equal(dst.cpu(), src), f"CPU to CUDA failed for {dtype}")

        # CUDA to CPU
        src = torch.ones(100, device="cuda", dtype=dtype)
        dst = torch.zeros(100, dtype=dtype)
        moodist.cuda_copy(dst, src)
        torch.cuda.synchronize()
        ctx.assert_true(torch.equal(dst, src.cpu()), f"CUDA to CPU failed for {dtype}")

        del src, dst

    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_various_shapes(ctx: TestContext):
    """Test cuda_copy with different shapes having same total bytes."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cpu_allocator()

    # Source is 2D, destination is 1D - same total elements
    src = torch.arange(100, dtype=torch.float32).view(10, 10)
    dst = torch.zeros(100, device="cuda", dtype=torch.float32)

    moodist.cuda_copy(dst, src)
    torch.cuda.synchronize()

    ctx.assert_true(torch.equal(dst.cpu(), src.view(-1)))

    # Source is 3D, destination is 2D
    src = torch.arange(120, device="cuda", dtype=torch.float32).view(2, 3, 20)
    dst = torch.zeros(6, 20, dtype=torch.float32)

    moodist.cuda_copy(dst, src)
    torch.cuda.synchronize()

    ctx.assert_true(torch.equal(dst, src.cpu().view(6, 20)))

    del src, dst
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_large(ctx: TestContext):
    """Test cuda_copy with larger tensors."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cpu_allocator()

    # 10 MB tensor
    size = 10 * 1024 * 1024 // 4  # 10MB of float32
    src = torch.randn(size, dtype=torch.float32)
    dst = torch.zeros(size, device="cuda", dtype=torch.float32)

    moodist.cuda_copy(dst, src)
    torch.cuda.synchronize()

    ctx.assert_true(torch.equal(dst.cpu(), src))

    del src, dst
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_error_non_contiguous(ctx: TestContext):
    """Test that cuda_copy raises error for non-contiguous tensors."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)

    # Create non-contiguous tensor via transpose
    src = torch.arange(100, device="cuda", dtype=torch.float32).view(10, 10).t()
    dst = torch.zeros(100, device="cuda", dtype=torch.float32)

    ctx.assert_false(src.is_contiguous())

    try:
        moodist.cuda_copy(dst, src)
        ctx.fail("Expected RuntimeError for non-contiguous src")
    except RuntimeError as e:
        ctx.assert_true("contiguous" in str(e).lower())

    # Also test non-contiguous dst
    src = torch.arange(100, device="cuda", dtype=torch.float32)
    dst = torch.zeros(10, 10, device="cuda", dtype=torch.float32).t()

    ctx.assert_false(dst.is_contiguous())

    try:
        moodist.cuda_copy(dst, src)
        ctx.fail("Expected RuntimeError for non-contiguous dst")
    except RuntimeError as e:
        ctx.assert_true("contiguous" in str(e).lower())

    del src, dst
    torch.cuda.synchronize()
    gc.collect()


@test
def test_cuda_copy_error_size_mismatch(ctx: TestContext):
    """Test that cuda_copy raises error when sizes don't match."""
    if not torch.cuda.is_available():
        ctx.log("CUDA not available, skipping")
        return

    torch.cuda.set_device(ctx.local_rank)

    src = torch.arange(100, device="cuda", dtype=torch.float32)
    dst = torch.zeros(50, device="cuda", dtype=torch.float32)

    try:
        moodist.cuda_copy(dst, src)
        ctx.fail("Expected RuntimeError for size mismatch")
    except RuntimeError as e:
        ctx.assert_true("bytes" in str(e).lower())

    del src, dst
    torch.cuda.synchronize()
    gc.collect()
