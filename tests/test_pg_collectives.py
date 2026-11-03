"""
Tests for the process group's all-gather, reduce-scatter and all-reduce of CUDA tensors, which run as compile_op ops
(one per collective and shape, compiled at its first call): every dtype and reduce op, sizes from 16 bytes to several
MiB per rank, in place and not, tensors from PyTorch's allocator, from the communication pool (comm_empty) and at
addresses the kernel cannot take and sizes that are not a multiple of 16 bytes (which go through buffers of the pool),
on several streams, PREMUL_SUM; and avg of CPU tensors.
"""

import torch
import torch.distributed as dist

import moodist
from framework import TestContext, test, create_process_group

FLOATS = [torch.float32, torch.bfloat16, torch.float16, torch.float64]
INTS = [torch.int32, torch.int64]
OPS = {"sum": dist.ReduceOp.SUM, "avg": dist.ReduceOp.AVG, "min": dist.ReduceOp.MIN, "max": dist.ReduceOp.MAX}


def combinations():
    """Every (dtype, reduce op) of the reductions: avg of floating point types only."""
    r = [(dtype, op) for dtype in FLOATS for op in OPS]
    r += [(dtype, op) for dtype in INTS for op in OPS if op != "avg"]
    return r


def rank_values(rank, start, numel, dtype, salt, device="cuda"):
    """Rank's values of the elements start..start + numel of a tensor: small integers in [-4, 4] (sums of up to 64
    ranks are exact in every dtype)."""
    i = torch.arange(start, start + numel, device=device, dtype=torch.int64)
    return ((i * 3 + rank * 7 + salt) % 9 - 4).to(dtype)


def premul_sum(factor):
    """ReduceOp.PREMUL_SUM with a factor (torch 2.14: callable; before: _make_nccl_premul_sum)."""
    try:
        return dist.ReduceOp.PREMUL_SUM(factor)
    except TypeError:
        return dist._make_nccl_premul_sum(factor)


def reference(op, world_size, start, numel, salt, device="cuda"):
    """op over every rank's values of the elements start..start + numel, in float64 (op "premul_<factor>": the sum
    of the values times the factor)."""
    acc = None
    for r in range(world_size):
        v = rank_values(r, start, numel, torch.float64, salt, device)
        if acc is None:
            acc = v
        elif op == "min":
            acc = torch.minimum(acc, v)
        elif op == "max":
            acc = torch.maximum(acc, v)
        else:
            acc = acc + v
    if op.startswith("premul_"):
        return acc * float(op[len("premul_") :])
    return acc / world_size if op == "avg" else acc


def reduce_op(op):
    """The ReduceOp of a reduce op name ("premul_<factor>": PREMUL_SUM with that factor)."""
    return premul_sum(float(op[len("premul_") :])) if op.startswith("premul_") else OPS[op]


def exact(op):
    """Whether op's results are exact for the test's values (sums of small integers, a factor of 0.25 times one)."""
    return op in ("sum", "min", "max") or op == "premul_0.25"


def tolerance(dtype):
    """(rtol, atol) of the results that are not exact (averages, PREMUL_SUM with 1/3): the ranks combine the sum in
    another order, and across NVLink domains each domain's share is scaled and rounded before the shares are added."""
    return {
        torch.bfloat16: (2**-6, 2**-5),
        torch.float16: (2**-9, 2**-8),
        torch.float32: (1e-6, 1e-5),
        torch.float64: (1e-14, 1e-12),
    }[dtype]


def check(ctx, pg, what, out, expected, op, dtype):
    """Checks a result against its float64 reference on every rank, and raises on every rank if any rank failed (so
    that all ranks leave the test together)."""
    got = out.to(torch.float64)
    if exact(op):
        ok = bool(torch.equal(got, expected))
    else:
        rtol, atol = tolerance(dtype)
        ok = bool(torch.allclose(got, expected, rtol=rtol, atol=atol))
    error = ""
    if not ok:
        bad = (got != expected).nonzero()
        first = bad[0].item() if len(bad) else 0
        error = f"{what}: {len(bad)} of {got.numel()} wrong, first at {first}: {got[first].item()} expected " \
                f"{expected[first].item()}"
    raise_on_all_ranks(ctx, pg, error)


def raise_on_all_ranks(ctx, pg, error):
    flags = torch.zeros(ctx.world_size, dtype=torch.int32)
    flags[ctx.rank] = 1 if error else 0
    # (CPU tensors: the CPU thread's all-reduce, not the ops under test)
    pg.allreduce([flags]).wait()
    failed = [r for r, f in enumerate(flags.tolist()) if f]
    if failed:
        ctx.assert_true(False, error or f"wrong results on ranks {failed[:16]}")


def all_gather(ctx, pg, numel, dtype, salt, alloc=None, in_place=False):
    """An all-gather of numel elements per rank; returns the output."""
    alloc = alloc or cuda_empty
    n = ctx.world_size
    out = alloc(n * numel, dtype=dtype)
    out.fill_(99)
    inp = out[ctx.rank * numel:(ctx.rank + 1) * numel] if in_place else alloc(numel, dtype=dtype)
    inp.copy_(rank_values(ctx.rank, ctx.rank * numel, numel, dtype, salt))
    pg._allgather_base(out, inp).wait()
    expected = torch.cat([rank_values(r, r * numel, numel, dtype, salt) for r in range(n)]).to(torch.float64)
    check(ctx, pg, f"all-gather {dtype} {numel} per rank{' in place' if in_place else ''}", out, expected, "sum", dtype)
    return out


def reduce_scatter(ctx, pg, numel, dtype, op, salt, alloc=None):
    """A reduce-scatter of numel elements per rank."""
    alloc = alloc or cuda_empty
    n = ctx.world_size
    inp = alloc(n * numel, dtype=dtype)
    inp.copy_(rank_values(ctx.rank, 0, n * numel, dtype, salt))
    out = alloc(numel, dtype=dtype)
    out.fill_(99)
    opts = dist.ReduceScatterOptions()
    opts.reduceOp = reduce_op(op)
    pg._reduce_scatter_base(out, inp, opts).wait()
    expected = reference(op, n, ctx.rank * numel, numel, salt)
    check(ctx, pg, f"reduce-scatter {op} {dtype} {numel} per rank", out, expected, op, dtype)


def all_reduce(ctx, pg, numel, dtype, op, salt, alloc=None):
    """An all-reduce (in place) of numel elements."""
    alloc = alloc or cuda_empty
    t = alloc(numel, dtype=dtype)
    t.copy_(rank_values(ctx.rank, 0, numel, dtype, salt))
    opts = dist.AllreduceOptions()
    opts.reduceOp = reduce_op(op)
    pg.allreduce([t], opts).wait()
    expected = reference(op, ctx.world_size, 0, numel, salt)
    check(ctx, pg, f"all-reduce {op} {dtype} {numel}", t, expected, op, dtype)


def cuda_empty(numel, dtype):
    return torch.empty(numel, dtype=dtype, device="cuda")


@test
def test_pg_collectives_dtypes(ctx: TestContext):
    """Every dtype and reduce op, 64 elements per rank (a multiple of 16 bytes in every dtype)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for i, (dtype, op) in enumerate(combinations()):
        reduce_scatter(ctx, pg, 64, dtype, op, salt=i)
        all_reduce(ctx, pg, 64, dtype, op, salt=i)
    for i, dtype in enumerate(FLOATS + INTS + [torch.uint8]):
        all_gather(ctx, pg, 64, dtype, salt=i)


@test
def test_pg_collectives_sizes(ctx: TestContext):
    """Sizes from 16 bytes to 4 MiB per rank (several chunks, parts and pieces), each twice (the second call runs the
    compiled op again)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for numel in [4, 1000, (1 << 18) + 12, 1 << 20]:
        for salt in range(2):
            all_gather(ctx, pg, numel, torch.float32, salt)
            reduce_scatter(ctx, pg, numel, torch.float32, "sum", salt)
            all_reduce(ctx, pg, numel, torch.float32, "sum", salt)
            reduce_scatter(ctx, pg, 2 * numel, torch.bfloat16, "avg", salt)
            all_reduce(ctx, pg, 2 * numel, torch.bfloat16, "avg", salt)


@test
def test_pg_all_gather_in_place(ctx: TestContext):
    """An all-gather whose input is this rank's part of the output (as FSDP2 does)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for numel in [64, 1 << 18]:
        for salt in range(2):
            all_gather(ctx, pg, numel, torch.bfloat16, salt, in_place=True)


@test
def test_pg_collectives_comm_pool(ctx: TestContext):
    """Tensors from the group's communication pool (NVLS where the group supports it), several times in a row."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)

    def alloc(numel, dtype):
        return moodist.comm_empty(numel, dtype=dtype, group=pg)

    for numel in [256, 1 << 18]:
        for salt in range(3):
            all_gather(ctx, pg, numel, torch.bfloat16, salt, alloc=alloc)
            all_gather(ctx, pg, numel, torch.bfloat16, salt, alloc=alloc, in_place=True)
            reduce_scatter(ctx, pg, numel, torch.bfloat16, "sum", salt, alloc=alloc)
            reduce_scatter(ctx, pg, numel, torch.float32, "avg", salt, alloc=alloc)
            all_reduce(ctx, pg, numel, torch.bfloat16, "sum", salt, alloc=alloc)
            all_reduce(ctx, pg, numel, torch.float32, "avg", salt, alloc=alloc)


@test
def test_pg_collectives_misaligned(ctx: TestContext):
    """Tensors at addresses that are not 16-byte aligned (views at an offset of 4 bytes), and in place, which go
    through buffers of the communication pool."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)

    def alloc(numel, dtype):
        return torch.empty(numel + 4, dtype=dtype, device="cuda")[1:numel + 1]

    for numel in [64, 1 << 16]:
        all_gather(ctx, pg, numel, torch.float32, 0, alloc=alloc)
        all_gather(ctx, pg, numel, torch.float32, 1, alloc=alloc, in_place=True)
        reduce_scatter(ctx, pg, numel, torch.float32, "sum", 2, alloc=alloc)
        all_reduce(ctx, pg, numel, torch.float32, "max", 3, alloc=alloc)


@test
def test_pg_collectives_streams(ctx: TestContext):
    """Collectives on two streams, interleaved, with the tensors used on the stream that produced them."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    numel = 1 << 16
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    results = []
    for it in range(8):
        s = streams[it % 2]
        with torch.cuda.stream(s):
            t = rank_values(ctx.rank, 0, numel, torch.float32, it)
            pg.allreduce([t]).wait()
            out = torch.empty(ctx.world_size * numel, dtype=torch.float32, device="cuda")
            inp = rank_values(ctx.rank, ctx.rank * numel, numel, torch.float32, it)
            pg._allgather_base(out, inp).wait()
            results.append((t, out))
    for s in streams:
        torch.cuda.current_stream().wait_stream(s)
    for it, (t, out) in enumerate(results):
        check(ctx, pg, f"all-reduce on stream {it % 2}, iteration {it}", t,
              reference("sum", ctx.world_size, 0, numel, it), "sum", torch.float32)
        expected = torch.cat([rank_values(r, r * numel, numel, torch.float64, it) for r in range(ctx.world_size)])
        check(ctx, pg, f"all-gather on stream {it % 2}, iteration {it}", out, expected, "sum", torch.float32)


@test
def test_pg_collectives_unaligned_sizes(ctx: TestContext):
    """Sizes that are not a multiple of 16 bytes (the ops of the next multiple, through buffers whose rows are padded),
    in place too, and at addresses that are not aligned either."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)

    def misaligned(numel, dtype):
        return torch.empty(numel + 4, dtype=dtype, device="cuda")[1 : numel + 1]

    for numel in [1, 3, 1001, (1 << 17) + 5]:
        all_gather(ctx, pg, numel, torch.float32, numel)
        all_gather(ctx, pg, numel, torch.bfloat16, numel, in_place=True)
        all_gather(ctx, pg, numel, torch.uint8, numel, alloc=misaligned)
        reduce_scatter(ctx, pg, numel, torch.float32, "sum", numel)
        reduce_scatter(ctx, pg, numel, torch.bfloat16, "avg", numel, alloc=misaligned)
        all_reduce(ctx, pg, numel, torch.float32, "sum", numel)
        all_reduce(ctx, pg, numel, torch.float16, "max", numel, alloc=misaligned)


@test
def test_pg_premul_sum(ctx: TestContext):
    """PREMUL_SUM of every floating point type: the sum of the inputs, each multiplied by the factor (in float32 in the
    kernel, before the result is rounded to the dtype): exact with a factor of 0.25, within rounding with 1/3."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for i, dtype in enumerate(FLOATS):
        for op in ["premul_0.25", f"premul_{1 / 3}"]:
            reduce_scatter(ctx, pg, 64, dtype, op, salt=i)
            all_reduce(ctx, pg, 64, dtype, op, salt=i)
            all_reduce(ctx, pg, 1001, dtype, op, salt=i)


@test
def test_pg_premul_sum_float16_range(ctx: TestContext):
    """PREMUL_SUM of float16 values whose sum overflows float16 while the result does not: the factor applies before
    anything is rounded to float16 (the inputs are multiplied first, as documented)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 1 << 12
    t = torch.full((numel,), 30000.0, dtype=torch.float16, device="cuda")
    opts = dist.AllreduceOptions()
    opts.reduceOp = premul_sum(1.0 / n)
    pg.allreduce([t], opts).wait()
    inp = torch.full((n * numel,), 30000.0, dtype=torch.float16, device="cuda")
    out = torch.empty(numel, dtype=torch.float16, device="cuda")
    rs_opts = dist.ReduceScatterOptions()
    rs_opts.reduceOp = premul_sum(1.0 / n)
    pg._reduce_scatter_base(out, inp, rs_opts).wait()
    error = ""
    for what, x in [("all-reduce", t), ("reduce-scatter", out)]:
        if not bool((x == 30000.0).all()):
            error = f"{what}: {x[:4].tolist()} expected 30000"
    raise_on_all_ranks(ctx, pg, error)


@test
def test_pg_avg_int_error(ctx: TestContext):
    """avg of integers: an error on every rank (compile_op averages floating point types only)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    t = torch.ones(64, dtype=torch.int32, device="cuda")
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.AVG
    ctx.assert_raises(RuntimeError, lambda: pg.allreduce([t], opts).wait())


@test
def test_pg_alltoall_local(ctx: TestContext):
    """The process group's all-to-all of CUDA tensors (within a node only; its kernel comes from the same NVRTC module
    as the all-gather's and reduce-scatter's did)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 1000
    inputs = [torch.full((numel,), float(ctx.rank * 100 + j), device="cuda") for j in range(n)]
    outputs = [torch.empty(numel, device="cuda") for _ in range(n)]
    try:
        pg.alltoall(outputs, inputs).wait()
    except RuntimeError as e:
        if "only local alltoall" in str(e):
            return
        raise
    bad = [j for j, out in enumerate(outputs) if not bool((out == float(j * 100 + ctx.rank)).all())]
    raise_on_all_ranks(ctx, pg, f"all-to-all: wrong parts from ranks {bad[:8]}" if bad else "")


@test
def test_pg_avg_cpu(ctx: TestContext):
    """avg of CPU tensors (the CPU thread's reduction has none: the sum, divided)."""
    pg = create_process_group(ctx)
    numel = 100
    n = ctx.world_size
    inp = rank_values(ctx.rank, 0, n * numel, torch.float32, 5, device="cpu")
    out = torch.empty(numel, dtype=torch.float32)
    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.AVG
    pg._reduce_scatter_base(out, inp, opts).wait()
    expected = reference("avg", n, ctx.rank * numel, numel, 5, device="cpu")
    ok = bool(torch.allclose(out.to(torch.float64), expected, rtol=0, atol=1e-5))
    raise_on_all_ranks(ctx, pg, "" if ok else f"reduce-scatter avg of CPU tensors: {out[:4]} expected {expected[:4]}")
