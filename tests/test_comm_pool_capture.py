"""
Tests for moodist's communication pools during CUDA graph captures (moodist.comm_empty, the process group's
allocate_tensor): a capture's buffers come from memory of the capture's own (what its buffers freed during the
capture, or new slabs), which nothing else takes later, so that their offsets are the same on every rank that captures
the same, and so are the eager offsets after the capture.

With PyTorch's own CUDA allocator (as here), which routes nothing to moodist's allocator during a capture, and with
moodist's (enabled up front, e.g. by moodist_dev's run_tests.sh with CUDA_ALLOC=1), which PyTorch routes the capture's
tensors to.

Set MOODIST_DEBUG_ALLOCATOR=1 to also check the allocator's data structures after every call.
"""

import torch
import moodist
from framework import TestContext, test, create_process_group

# torch.cuda.graph's default ("global") and torchtitan's
MODES = ["global", "thread_local"]


def gather_ints(pg, ctx, values):
    """Every rank's values (a list of ints of the same length on every rank), as a list per rank."""
    t = torch.tensor(values, dtype=torch.int64, device="cuda")
    out = torch.empty(ctx.world_size * len(values), dtype=torch.int64, device="cuda")
    pg._allgather_base(out, t).wait()
    return out.view(ctx.world_size, len(values)).tolist()


def assert_same_on_all_ranks(pg, ctx, values, what):
    rows = gather_ints(pg, ctx, values)
    for r, row in enumerate(rows):
        ctx.assert_true(row == rows[0], f"{what}: rank {r} has {row}, rank 0 has {rows[0]}")


def setup(ctx):
    """The process group, after an eager all-gather: in a group of one NVLink domain that supports NVLS, its compile
    sets up the group's multicast objects, whose listener then binds the new slabs of captures during the captures
    (a buffer of a slab that is not bound has no offset)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    gather_ints(pg, ctx, [0])
    return pg


def comm(pg, n, dtype=torch.uint8):
    return moodist.comm_empty(n, dtype=dtype, group=pg)


def span(t):
    """The logical range of a buffer of the pool (its allocation is rounded up to 4 KiB)"""
    begin = moodist.comm_pool_offset(t)
    assert begin >= 0
    return begin, begin + (t.numel() * t.element_size() + 4095) // 4096 * 4096


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


@test
def test_comm_pool_capture_offsets(ctx: TestContext):
    """Buffers allocated during a capture (in both capture modes), some freed and their memory reused by the capture
    (a buffer of the same size gets the same offset), some kept past it: the same offsets on every rank; they overlap no
    eager buffer allocated before or after the capture, whose offsets are the same on every rank too; the graph's
    replays write the kept buffers."""
    pg = setup(ctx)
    for mode in MODES:
        before = [comm(pg, n) for n in [1 << 20, 3 << 20]]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode=mode):
            a = comm(pg, 1 << 20)
            b = comm(pg, 5 << 20)
            a.fill_(1)
            b.fill_(2)
            span_a = span(a)
            del a
            c = comm(pg, 1 << 20)
            c.fill_(3)
            d = comm(pg, 300 << 20)  # (larger than a slab: one of its own)
            d.fill_(4)
        inside = [b, c, d]
        ctx.assert_true(span(c) == span_a, f"{mode}: the capture did not reuse its freed memory: {span_a}, {span(c)}")
        after = [comm(pg, n) for n in [1 << 20, 5 << 20, 64 << 20]]
        for t in after:
            t.fill_(9)
        for t in inside:
            t.fill_(0)
        g.replay()
        torch.cuda.synchronize()
        for t, v in zip(inside, [2, 3, 4]):
            ctx.assert_true(bool((t == v).all()), f"{mode}: a replay did not write the capture's buffer of {v}")
        for t in after:
            ctx.assert_true(bool((t == 9).all()), f"{mode}: a replay wrote into an eager buffer")
        for x in inside + [span_a]:
            sx = x if isinstance(x, tuple) else span(x)
            for t in before + after:
                ctx.assert_true(not overlap(sx, span(t)), f"{mode}: capture {sx} and eager {span(t)} overlap")
        offsets = [span(t)[0] for t in before + inside + after] + [span_a[0]]
        assert_same_on_all_ranks(pg, ctx, offsets, f"{mode}: offsets")
        del g, inside, before, after, b, c, d
        torch.cuda.synchronize()


@test
def test_comm_pool_capture_replay(ctx: TestContext):
    """A graph whose buffers come from the pool, on two streams of the capture (one freed during it and its memory
    reused on its stream), replayed with new inputs: every replay's result is right, with eager buffers of the pool
    allocated, written and freed in between."""
    pg = setup(ctx)
    n = 1 << 20
    x = torch.zeros(n, device="cuda")
    side = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        a = comm(pg, n, torch.float32)
        a.copy_(x)
        b = comm(pg, n, torch.float32)
        torch.mul(a, 2, out=b)
        offsets = [span(a)[0], span(b)[0]]
        del a
        c = comm(pg, n, torch.float32)
        torch.add(b, 1, out=c)
        offsets.append(span(c)[0])
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            d = comm(pg, n, torch.float32)
            torch.mul(c, 3, out=d)
            e = comm(pg, n, torch.float32)
            torch.add(d, 1, out=e)
            offsets += [span(d)[0], span(e)[0]]
            del d
            f = comm(pg, n, torch.float32)
            torch.sub(e, 1, out=f)
            offsets.append(span(f)[0])
        torch.cuda.current_stream().wait_stream(side)
        out = f.clone()
    ctx.assert_true(offsets[2] == offsets[0], f"the main stream did not reuse its freed memory: {offsets}")
    ctx.assert_true(offsets[5] == offsets[3], f"the side stream did not reuse its freed memory: {offsets}")
    for it in range(5):
        x.copy_(torch.randn(n, device="cuda"))
        g.replay()
        noise = [comm(pg, s) for s in [n, 4 * n, 7 << 20]]
        for t in noise:
            t.fill_(0xAB)
        del noise
        torch.cuda.synchronize()
        expected = ((x * 2 + 1) * 3 + 1) - 1
        ctx.assert_true(torch.allclose(out, expected), f"replay {it}: wrong result")
    assert_same_on_all_ranks(pg, ctx, offsets, "offsets")
    del g, b, c, e, f, out
    torch.cuda.synchronize()


@test
def test_comm_pool_capture_eager_free(ctx: TestContext):
    """Eager buffers freed during a capture, one used only by the legacy default stream, one by the capturing stream
    (whose free waits for the end of the capture): the capture's buffers take neither's memory, and once the capture
    has ended, the next eager buffers of their sizes on their streams get it again (on every rank alike)."""
    pg = setup(ctx)
    s = torch.cuda.Stream()
    t = comm(pg, 3 << 20)
    t.fill_(7)
    with torch.cuda.stream(s):
        u = comm(pg, 2 << 20)
        u.fill_(8)
    torch.cuda.synchronize()
    span_t, span_u = span(t), span(u)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        del t, u
        v = comm(pg, 3 << 20)
        w = comm(pg, 2 << 20)
        v.fill_(1)
        w.fill_(2)
    for x in [v, w]:
        for y in [span_t, span_u]:
            ctx.assert_true(not overlap(span(x), y), f"the capture took freed eager memory: {span(x)}, {y}")
    g.replay()
    torch.cuda.synchronize()
    t2 = comm(pg, 3 << 20)
    with torch.cuda.stream(s):
        u2 = comm(pg, 2 << 20)
    ctx.assert_true(span(t2) == span_t, f"the default stream's freed memory: {span_t}, then {span(t2)}")
    ctx.assert_true(span(u2) == span_u, f"the capturing stream's freed memory: {span_u}, then {span(u2)}")
    offsets = [span(x)[0] for x in [v, w, t2, u2]]
    assert_same_on_all_ranks(pg, ctx, offsets, "offsets")
    del g, v, w, t2, u2
    torch.cuda.synchronize()


@test
def test_comm_pool_capture_two_graphs(ctx: TestContext):
    """Two captures: the second takes none of the first's memory (both graphs may be replayed, also at the same
    time), even what the first freed during its capture or its buffers freed later; both replay correctly."""
    pg = setup(ctx)
    n = 1 << 20
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1):
        a = comm(pg, n)
        a.fill_(1)
        spans1 = [span(a)]
        del a
        b = comm(pg, n)
        b.fill_(2)
        spans1.append(span(b))
    span_b = span(b)
    del b  # (freed eagerly: the capture's memory still)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        c = comm(pg, n)
        c.fill_(3)
        spans2 = [span(c)]
    for x in spans2:
        for y in spans1 + [span_b]:
            ctx.assert_true(not overlap(x, y), f"the second capture took the first's memory: {x}, {y}")
    for _ in range(3):
        c.fill_(0)
        g1.replay()
        g2.replay()
    torch.cuda.synchronize()
    ctx.assert_true(bool((c == 3).all()), "the second graph's buffer")
    assert_same_on_all_ranks(pg, ctx, [x[0] for x in spans1 + spans2], "offsets")
    del g1, g2, c
    torch.cuda.synchronize()
