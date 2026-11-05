"""
Tests for compile_op with PyTorch's own CUDA allocator (moodist.enable_cuda_allocator() not called): the tensors of an
op must still be moodist's memory, e.g. buffers from moodist.comm_empty. An op whose regions have pieces that are not
contiguous in a tensor moves them through contiguous copies, which come from PyTorch's current CUDA allocator, and
other ranks map those like the tensors (they can map moodist's memory only): such an op raises when it runs, on every
rank that has copies, before its launch.

This file must not enable moodist's CUDA allocator. tests/run.py runs each file in a process of its own unless
--no-fork; then another file may have enabled it, and the test does nothing.
"""

import torch
import moodist
from moodist import TensorRegion
from framework import TestContext, test, create_process_group


def all_ranks_ok(ctx, pg, ok, message):
    """Asserts on every rank that every rank is ok (CPU tensors: the CPU thread's all-reduce, not a compile_op op)."""
    flags = torch.zeros(ctx.world_size, dtype=torch.int32)
    flags[ctx.rank] = 0 if ok else 1
    pg.allreduce([flags]).wait()
    bad = [r for r, f in enumerate(flags.tolist()) if f]
    ctx.assert_true(not bad, f"{message} (ranks {bad[:16]})")


@test
def test_compile_op_copies_need_moodist_allocator(ctx: TestContext):
    """With buffers from comm_empty and PyTorch's allocator: an all-gather (no copies) works; an op whose pieces are not
    contiguous in the input (rows to columns: input copies) and one whose pieces are not contiguous in the output (an
    all-gather of columns: output copies) raise on every rank; an all-gather after them still works (the failed calls
    took no step values)."""
    if ctx.world_size < 2 or torch.cuda.get_allocator_backend() == "moodist":
        return
    pg = create_process_group(ctx)
    n = ctx.world_size
    rows, c = 4, 8  # (32 bytes per row of a piece: the copy kernel takes multiples of 16)
    dtype = torch.float32

    def all_gather(salt):
        region = [TensorRegion(offset=[ctx.rank * c], shape=[c], device="cuda")]
        whole = [TensorRegion(offset=[0], shape=[n * c], device="cuda")]
        op = moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=whole)
        x = moodist.comm_empty(c, dtype=dtype, group=pg)
        x.copy_(torch.arange(c, dtype=dtype, device="cuda") + ctx.rank * 100 + salt)
        out = moodist.comm_empty(n * c, dtype=dtype, group=pg)
        op([x], [out]).wait()
        expected = torch.cat([torch.arange(c, dtype=dtype, device="cuda") + r * 100 + salt for r in range(n)])
        return torch.equal(out, expected)

    all_ranks_ok(ctx, pg, all_gather(0), "all-gather of comm_empty buffers: wrong output")

    # (input offset, input shape, output offset, output shape) of a matrix
    cases = {
        # rank r has rows r as input and receives columns r: its piece for another rank is a block of its columns
        "input": ([ctx.rank * rows, 0], [rows, n * c], [0, ctx.rank * c], [n * rows, c]),
        # an all-gather of columns: the piece from another rank is a block of the output's columns
        "output": ([0, ctx.rank * c], [rows, c], [0, 0], [rows, n * c]),
    }
    for what, (in_offset, in_shape, out_offset, out_shape) in cases.items():
        op = moodist.compile_op(
            pg,
            dtype=dtype,
            inputs=[TensorRegion(offset=in_offset, shape=in_shape, device="cuda")],
            outputs=[TensorRegion(offset=out_offset, shape=out_shape, device="cuda")],
        )
        x = moodist.comm_empty(*in_shape, dtype=dtype, group=pg)
        out = moodist.comm_empty(*out_shape, dtype=dtype, group=pg)
        error = ""
        try:
            op([x], [out]).wait()
        except RuntimeError as e:
            error = str(e)
        ok = f"custom op {what}[0] has a piece that is not contiguous" in error and "enable_cuda_allocator" in error
        all_ranks_ok(ctx, pg, ok, f"op with {what} copies: expected the error about its copies, got {error!r}")

    all_ranks_ok(ctx, pg, all_gather(7), "all-gather after the failed calls: wrong output")
