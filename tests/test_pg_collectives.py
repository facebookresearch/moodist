"""
Tests for the process group's collectives of CUDA tensors, which run as compile_op ops (one per collective and shape,
compiled at its first call): all-gather, reduce-scatter and all-reduce with every dtype and reduce op, sizes from 16
bytes to several MiB per rank, in place and not, tensors from PyTorch's allocator, from the communication pool
(comm_empty) and at addresses the kernel cannot take and sizes that are not a multiple of 16 bytes (which go through
buffers of the pool), PyTorch's tensors freed after their calls (their memory comes back), on several streams,
PREMUL_SUM; broadcast, reduce, scatter, gather and all-to-all (from and to several roots, lists of tensors); and avg
of CPU tensors.
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


def broadcast(ctx, pg, numel, dtype, root, salt, alloc=None):
    """A broadcast of numel elements from root."""
    alloc = alloc or cuda_empty
    t = alloc(numel, dtype=dtype)
    if ctx.rank == root:
        t.copy_(rank_values(root, 0, numel, dtype, salt))
    else:
        t.fill_(99)
    pg.broadcast(t, root).wait()
    expected = rank_values(root, 0, numel, torch.float64, salt)
    check(ctx, pg, f"broadcast {dtype} {numel} from {root}", t, expected, "sum", dtype)


def reduce(ctx, pg, numel, dtype, op, root, salt, alloc=None):
    """A reduce of numel elements to root (in place there); the other ranks' tensors stay as they were."""
    alloc = alloc or cuda_empty
    t = alloc(numel, dtype=dtype)
    t.copy_(rank_values(ctx.rank, 0, numel, dtype, salt))
    opts = dist.ReduceOptions()
    opts.reduceOp = reduce_op(op)
    opts.rootRank = root
    pg.reduce([t], opts).wait()
    if ctx.rank == root:
        check(ctx, pg, f"reduce {op} {dtype} {numel} to {root}", t, reference(op, ctx.world_size, 0, numel, salt), op,
              dtype)
    else:
        check(ctx, pg, f"reduce {op} {dtype} {numel} to {root}: rank {ctx.rank}'s input", t,
              rank_values(ctx.rank, 0, numel, torch.float64, salt), "sum", dtype)


def part_inputs(ctx, numel, dtype, rank, salt, alloc, contiguous):
    """rank's parts for every rank (part j: the elements j * numel..), as a list of tensors: views of one tensor if
    contiguous, else separate tensors."""
    n = ctx.world_size
    if contiguous:
        whole = alloc(n * numel, dtype=dtype)
        whole.copy_(rank_values(rank, 0, n * numel, dtype, salt))
        return list(whole.split(numel)) if numel else [whole[:0] for _ in range(n)]
    parts = [alloc(numel, dtype=dtype) for _ in range(n)]
    for j, p in enumerate(parts):
        p.copy_(rank_values(rank, j * numel, numel, dtype, salt))
    return parts


def empty_parts(ctx, numel, dtype, alloc, contiguous):
    n = ctx.world_size
    if contiguous:
        whole = alloc(n * numel, dtype=dtype)
        whole.fill_(99)
        return list(whole.split(numel)) if numel else [whole[:0] for _ in range(n)]
    parts = [alloc(numel, dtype=dtype) for _ in range(n)]
    for p in parts:
        p.fill_(99)
    return parts


def scatter(ctx, pg, numel, dtype, root, salt, alloc=None, contiguous=False):
    """A scatter of numel elements per rank from root's list."""
    alloc = alloc or cuda_empty
    out = alloc(numel, dtype=dtype)
    out.fill_(99)
    inputs = [part_inputs(ctx, numel, dtype, root, salt, alloc, contiguous)] if ctx.rank == root else []
    opts = dist.ScatterOptions()
    opts.rootRank = root
    pg.scatter([out], inputs, opts).wait()
    expected = rank_values(root, ctx.rank * numel, numel, torch.float64, salt)
    check(ctx, pg, f"scatter {dtype} {numel} per rank from {root}", out, expected, "sum", dtype)


def gather(ctx, pg, numel, dtype, root, salt, alloc=None, contiguous=False):
    """A gather of numel elements per rank into root's list."""
    alloc = alloc or cuda_empty
    inp = alloc(numel, dtype=dtype)
    inp.copy_(rank_values(ctx.rank, ctx.rank * numel, numel, dtype, salt))
    outputs = [empty_parts(ctx, numel, dtype, alloc, contiguous)] if ctx.rank == root else []
    opts = dist.GatherOptions()
    opts.rootRank = root
    pg.gather(outputs, [inp], opts).wait()
    if ctx.rank == root:
        got = torch.cat(outputs[0])
        expected = torch.cat([rank_values(r, r * numel, numel, torch.float64, salt) for r in range(ctx.world_size)])
    else:
        got = inp
        expected = rank_values(ctx.rank, ctx.rank * numel, numel, torch.float64, salt)
    check(ctx, pg, f"gather {dtype} {numel} per rank to {root}", got, expected, "sum", dtype)


def all_to_all(ctx, pg, numel, dtype, salt, alloc=None, lists=False):
    """An all-to-all of numel elements per pair of ranks: all_to_all_single's (one tensor each), or all_to_all's
    (lists of separate tensors)."""
    alloc = alloc or cuda_empty
    n = ctx.world_size
    if lists:
        inputs = part_inputs(ctx, numel, dtype, ctx.rank, salt, alloc, contiguous=False)
        outputs = empty_parts(ctx, numel, dtype, alloc, contiguous=False)
        pg.alltoall(outputs, inputs).wait()
        out = torch.cat(outputs)
    else:
        inp = alloc(n * numel, dtype=dtype)
        inp.copy_(rank_values(ctx.rank, 0, n * numel, dtype, salt))
        out = alloc(n * numel, dtype=dtype)
        out.fill_(99)
        pg.alltoall_base(out, inp, [], []).wait()
    # (rank i's part for this rank: its elements rank * numel..)
    expected = torch.cat([rank_values(i, ctx.rank * numel, numel, torch.float64, salt) for i in range(n)])
    check(ctx, pg, f"all-to-all{' (lists)' if lists else ''} {dtype} {numel} per pair", out, expected, "sum", dtype)


def roots(ctx):
    """Root ranks to test: the first, the last, and one in the middle."""
    return sorted({0, ctx.world_size - 1, ctx.world_size // 2})


def cuda_empty(numel, dtype):
    return torch.empty(numel, dtype=dtype, device="cuda")


def misaligned_empty(numel, dtype):
    """A tensor at an address that is not 16-byte aligned (a view at an offset of one element)."""
    return torch.empty(numel + 4, dtype=dtype, device="cuda")[1 : numel + 1]


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
def test_pg_freed_tensors_memory(ctx: TestContext):
    """Collectives of tensors from PyTorch's allocator, which are freed after their calls and their memory returned to
    the driver (torch.cuda.empty_cache: cudaFree), round after round: the GPU memory comes back. Other ranks reach
    moodist's memory only (such tensors go through buffers of the communication pool), so no peer's mapping and no
    RDMA registration holds memory that PyTorch freed."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    # (with moodist's allocator enabled in this process, e.g. by another test module loaded into it, every tensor would
    # be moodist's, and the test would pass whatever the process group did)
    ctx.assert_true(
        torch.cuda.get_allocator_backend() != "moodist", "needs PyTorch's allocator, but moodist's is enabled"
    )
    numel = 16 << 20  # 64 MiB of float32
    part = numel // ctx.world_size

    def collectives(salt):
        all_reduce(ctx, pg, numel, torch.float32, "sum", salt)
        reduce_scatter(ctx, pg, part, torch.float32, "sum", salt)
        all_gather(ctx, pg, part, torch.float32, salt)
        broadcast(ctx, pg, numel, torch.float32, 0, salt)
        reduce(ctx, pg, numel, torch.float32, "sum", ctx.world_size - 1, salt)
        all_to_all(ctx, pg, part, torch.float32, salt)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    collectives(0)  # (compiles the ops: their tuning, scratch and communication buffers are moodist's, which it keeps)
    free = torch.cuda.mem_get_info()[0]
    for salt in range(1, 9):
        collectives(salt)
    lost = free - torch.cuda.mem_get_info()[0]
    # (what peers' mappings or the NICs' registrations of the freed tensors held would be gigabytes)
    raise_on_all_ranks(ctx, pg, f"{lost >> 20} MiB of GPU memory did not come back" if lost > (256 << 20) else "")


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
def test_pg_synchronous_calls(ctx: TestContext):
    """Synchronous calls (asyncOp false: torch.distributed's async_op=False) return no Work, as ProcessGroupNCCL's do,
    asynchronous ones (the default) a Work; the results are on the stream either way."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 1000
    c10d = torch._C._distributed_c10d
    for async_op in (False, True):
        salt = 20 + int(async_op)

        def call(what, fn, options, **fields):
            opts = options()
            for k, v in fields.items():
                setattr(opts, k, v)
            opts.asyncOp = async_op
            work = fn(opts)
            ok = (work is not None) == async_op
            raise_on_all_ranks(ctx, pg, "" if ok else f"{what} with asyncOp {async_op} returned {work!r}")
            if work is not None:
                work.wait()

        t = rank_values(ctx.rank, 0, numel, torch.float32, salt)
        call("allreduce", lambda o: pg.allreduce([t], o), c10d.AllreduceOptions, reduceOp=dist.ReduceOp.SUM)
        check(ctx, pg, f"all-reduce, asyncOp {async_op}", t, reference("sum", n, 0, numel, salt), "sum", torch.float32)
        out = torch.empty(n * numel, device="cuda")
        inp = rank_values(ctx.rank, ctx.rank * numel, numel, torch.float32, salt)
        call("_allgather_base", lambda o: pg._allgather_base(out, inp, o), c10d.AllgatherOptions)
        expected = torch.cat([rank_values(r, r * numel, numel, torch.float64, salt) for r in range(n)])
        check(ctx, pg, f"all-gather, asyncOp {async_op}", out, expected, "sum", torch.float32)
        inp = rank_values(ctx.rank, 0, n * numel, torch.float32, salt)
        out = torch.empty(numel, device="cuda")
        call("_reduce_scatter_base", lambda o: pg._reduce_scatter_base(out, inp, o), c10d.ReduceScatterOptions,
             reduceOp=dist.ReduceOp.SUM)
        check(ctx, pg, f"reduce-scatter, asyncOp {async_op}", out, reference("sum", n, ctx.rank * numel, numel, salt),
              "sum", torch.float32)
        t = rank_values(ctx.rank, 0, numel, torch.float32, salt)
        call("broadcast", lambda o: pg.broadcast([t], o), c10d.BroadcastOptions, rootRank=0)
        check(ctx, pg, f"broadcast, asyncOp {async_op}", t, rank_values(0, 0, numel, torch.float64, salt), "sum",
              torch.float32)


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
def test_pg_broadcast_reduce(ctx: TestContext):
    """Broadcast and reduce from and to several roots: every reduce op, several dtypes and sizes (unaligned too), from
    PyTorch's allocator, misaligned and from the communication pool."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)

    def pool(numel, dtype):
        return moodist.comm_empty(numel, dtype=dtype, group=pg)

    for i, root in enumerate(roots(ctx)):
        for numel in [64, 1001, 1 << 18]:
            broadcast(ctx, pg, numel, torch.float32, root, salt=i)
            broadcast(ctx, pg, numel, torch.bfloat16, root, salt=i + 1, alloc=misaligned_empty)
            reduce(ctx, pg, numel, torch.float32, "sum", root, salt=i)
            reduce(ctx, pg, numel, torch.bfloat16, "avg", root, salt=i)
        broadcast(ctx, pg, 1 << 16, torch.bfloat16, root, salt=i, alloc=pool)
        reduce(ctx, pg, 1 << 16, torch.bfloat16, "sum", root, salt=i, alloc=pool)
        reduce(ctx, pg, 1 << 16, torch.float32, "max", root, salt=i, alloc=pool)
        reduce(ctx, pg, 4096, torch.int32, "min", root, salt=i, alloc=misaligned_empty)
        reduce(ctx, pg, 4096, torch.float16, "premul_0.25", root, salt=i)


@test
def test_pg_reduce_dtypes(ctx: TestContext):
    """Reduce of every dtype and reduce op, to the last rank."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for i, (dtype, op) in enumerate(combinations()):
        reduce(ctx, pg, 256, dtype, op, ctx.world_size - 1, salt=i)
    reduce(ctx, pg, 256, torch.bfloat16, f"premul_{1 / 3}", 0, salt=1)


@test
def test_pg_scatter_gather(ctx: TestContext):
    """Scatter and gather from and to several roots, with lists of separate tensors and of views of one, several sizes
    (unaligned too)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for i, root in enumerate(roots(ctx)):
        for numel in [1, 64, 1001, 1 << 17]:
            for contiguous in [False, True]:
                scatter(ctx, pg, numel, torch.float32, root, salt=i, contiguous=contiguous)
                gather(ctx, pg, numel, torch.float32, root, salt=i, contiguous=contiguous)
        scatter(ctx, pg, 1000, torch.bfloat16, root, salt=i, alloc=misaligned_empty)
        gather(ctx, pg, 1000, torch.bfloat16, root, salt=i, alloc=misaligned_empty)


@test
def test_pg_all_to_all(ctx: TestContext):
    """All-to-all: all_to_all_single's (one tensor each) and all_to_all's (lists of separate tensors), several sizes
    (unaligned too), from the communication pool and misaligned."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)

    def pool(numel, dtype):
        return moodist.comm_empty(numel, dtype=dtype, group=pg)

    # (1 MiB parts: PyTorch's allocator puts separate tensors one after the other in separate allocations)
    for numel in [1, 64, 1001, 1 << 17, 1 << 18]:
        for salt in range(2):
            all_to_all(ctx, pg, numel, torch.float32, salt)
            all_to_all(ctx, pg, numel, torch.float32, salt, lists=True)
    all_to_all(ctx, pg, 1 << 16, torch.bfloat16, 3, alloc=pool)
    all_to_all(ctx, pg, 1000, torch.int64, 4, alloc=misaligned_empty)


@test
def test_pg_all_to_all_in_place(ctx: TestContext):
    """all_to_all_single with the same tensor as input and output (the input goes through a buffer)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 1000
    t = rank_values(ctx.rank, 0, n * numel, torch.float32, 7)
    pg.alltoall_base(t, t, [], []).wait()
    expected = torch.cat([rank_values(i, ctx.rank * numel, numel, torch.float64, 7) for i in range(n)])
    check(ctx, pg, "all-to-all in place", t, expected, "sum", torch.float32)


@test
def test_pg_all_to_all_uneven_error(ctx: TestContext):
    """all_to_all_single with uneven splits: an error (on every rank, as every rank's splits are uneven here)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    n = ctx.world_size
    if n == 1:
        return
    splits = [2] * n
    splits[0], splits[1] = 1, 3
    inp = torch.zeros(2 * n, device="cuda")
    out = torch.zeros(2 * n, device="cuda")
    ctx.assert_raises(RuntimeError, lambda: pg.alltoall_base(out, inp, splits, splits).wait())


@test
def test_pg_list_collectives(ctx: TestContext):
    """The forms with lists of tensors: reduce_scatter's inputs, all_gather_coalesced."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 1000
    inputs = part_inputs(ctx, numel, torch.float32, ctx.rank, 2, cuda_empty, contiguous=False)
    out = torch.empty(numel, device="cuda")
    opts = dist.ReduceScatterOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    pg.reduce_scatter([out], [inputs], opts).wait()
    check(ctx, pg, "reduce_scatter (list)", out, reference("sum", n, ctx.rank * numel, numel, 2), "sum", torch.float32)
    if n > 1:
        # (inputs of other sizes than the output's, even if they add up: an error, on every rank here)
        uneven = [torch.zeros(numel + (1 if j == 0 else -1 if j == 1 else 0), device="cuda") for j in range(n)]
        ctx.assert_raises(RuntimeError, lambda: pg.reduce_scatter([out], [uneven], opts).wait())
    a = rank_values(ctx.rank, 0, 100, torch.float32, 3)
    b = rank_values(ctx.rank, 0, 37, torch.int64, 4)
    outputs = [[torch.empty_like(a), torch.empty_like(b)] for _ in range(n)]
    pg.allgather_coalesced(outputs, [a, b]).wait()
    got = torch.cat([torch.cat([o[0].to(torch.float64), o[1].to(torch.float64)]) for o in outputs])
    expected = torch.cat([torch.cat([rank_values(r, 0, 100, torch.float64, 3), rank_values(r, 0, 37, torch.float64, 4)])
                          for r in range(n)])
    check(ctx, pg, "allgather_coalesced", got, expected, "sum", torch.float32)


@test
def test_pg_cpu_errors(ctx: TestContext):
    """Reduce and all-to-all of CPU tensors: errors (on every rank)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    t = torch.ones(16)
    opts = dist.ReduceOptions()
    opts.rootRank = 0
    ctx.assert_raises(RuntimeError, lambda: pg.reduce([t], opts).wait())
    inputs = [torch.ones(4) for _ in range(n)]
    outputs = [torch.zeros(4) for _ in range(n)]
    ctx.assert_raises(RuntimeError, lambda: pg.alltoall(outputs, inputs).wait())


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
