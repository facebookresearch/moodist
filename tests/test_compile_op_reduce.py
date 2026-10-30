"""
Tests for compile_op reductions: reduce="sum", "avg", "min" and "max", where every output element is the op over all
the input regions (of all ranks) that cover it, e.g. reduce-scatter and all-reduce.

Reductions need CUDA tensors. They work across NVLink domains (nodes, or e.g. NVL72 racks) too, where the tests run
unchanged.

Inputs are small integers that depend on the rank, the region and the global index, so that every rank can compute
every rank's inputs, and sums are exact in every dtype: the expected results are exact too (avg: the exact sum divided
by the number of inputs, rounded once in float32, then to the dtype, as the kernel does; across domains, the partial
results of the domains are divided and rounded to the dtype first, which is exact with a power of 2 ranks).
"""

import contextlib
import math
import os
import socket
import zlib

import torch
import moodist
from moodist import TensorRegion
from framework import TestContext, test, create_process_group

moodist.enable_cuda_allocator()

FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64]
INT_DTYPES = [torch.int32, torch.int64]


def ops_for(dtype):
    return ["sum", "avg", "min", "max"] if dtype.is_floating_point else ["sum", "min", "max"]


# Every dtype and op at one size, these at the others (each compile_op takes a second or more on many ranks)
SOME = [(torch.float32, "sum"), (torch.bfloat16, "avg"), (torch.int64, "max"), (torch.float16, "min")]


def all_combinations():
    return [(dtype, op_name) for dtype in FLOAT_DTYPES + INT_DTYPES for op_name in ops_for(dtype)]


def global_index(gshape, offset, shape):
    """The linear indices (in a tensor of shape gshape) of the elements of the region (offset, shape)."""
    index = torch.zeros(shape, dtype=torch.int64, device="cuda")
    stride = 1
    for d in reversed(range(len(gshape))):
        r = torch.arange(offset[d], offset[d] + shape[d], device="cuda", dtype=torch.int64)
        view = [1] * len(gshape)
        view[d] = shape[d]
        index = index + r.view(view) * stride
        stride *= gshape[d]
    return index


def make_input(rank, region, gshape, offset, shape, dtype, salt):
    """Small integers in [-4, 4] for rank's input region number `region`."""
    v = (global_index(gshape, offset, shape) * 3 + rank * 7 + region * 5 + salt) % 9 - 4
    return v.to(dtype)


def run_reduce(ctx, pg, op_name, dtype, spec, salt=0, in_place=False, repeat=1):
    """spec: tensor_id -> (global shape, [per rank: list of input (offset, shape)], [per rank: list of output (offset,
    shape)]). Compiles and runs the op on this rank's regions, and checks its outputs against the expected values.
    in_place: an output with the region of one of the rank's inputs (same tensor id, offset and shape) is that input
    tensor (the op writes over it). repeat: runs the op that many times, with the inputs filled in anew each time.
    Returns this rank's outputs."""
    inputs = []
    outputs = []
    input_values = []
    input_tensors = []
    output_tensors = []
    aliased = []  # per output: whether it is an input tensor
    expected = []
    for tid, (gshape, ins, outs) in spec.items():
        # The expected values of the whole global tensor, from every rank's inputs
        acc_dtype = torch.float64
        if op_name == "min":
            acc = torch.full(gshape, math.inf, dtype=acc_dtype, device="cuda")
        elif op_name == "max":
            acc = torch.full(gshape, -math.inf, dtype=acc_dtype, device="cuda")
        else:
            acc = torch.zeros(gshape, dtype=acc_dtype, device="cuda")
        count = torch.zeros(gshape, dtype=torch.int64, device="cuda")
        for r in range(ctx.world_size):
            for i, (offset, shape) in enumerate(ins[r]):
                v = make_input(r, i, gshape, offset, shape, torch.float64, salt)
                sl = tuple(slice(o, o + s) for o, s in zip(offset, shape))
                if op_name == "min":
                    acc[sl] = torch.minimum(acc[sl], v)
                elif op_name == "max":
                    acc[sl] = torch.maximum(acc[sl], v)
                else:
                    acc[sl] += v
                count[sl] += 1
        if op_name == "avg":
            if dtype == torch.float64:
                acc = acc / count
            else:
                acc = acc.to(torch.float32) / count.to(torch.float32)
        result = acc.to(dtype)
        my_inputs = {}
        for i, (offset, shape) in enumerate(ins[ctx.rank]):
            inputs.append(TensorRegion(offset=list(offset), shape=list(shape), device="cuda", tensor_id=tid))
            input_values.append(make_input(ctx.rank, i, gshape, offset, shape, dtype, salt))
            input_tensors.append(input_values[-1].clone())
            my_inputs[(tuple(offset), tuple(shape))] = input_tensors[-1]
        for offset, shape in outs[ctx.rank]:
            outputs.append(TensorRegion(offset=list(offset), shape=list(shape), device="cuda", tensor_id=tid))
            sl = tuple(slice(o, o + s) for o, s in zip(offset, shape))
            expected.append(result[sl])
            same = my_inputs.get((tuple(offset), tuple(shape))) if in_place else None
            aliased.append(same is not None)
            output_tensors.append(same if same is not None else torch.empty(shape, dtype=dtype, device="cuda"))
    op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs, reduce=op_name)
    errors = []
    for run in range(repeat):
        for t, v in zip(input_tensors, input_values):
            t.copy_(v)
        for t, a in zip(output_tensors, aliased):
            if not a:
                t.fill_(99)
        op(input_tensors, output_tensors).wait()
        for i, (out, exp) in enumerate(zip(output_tensors, expected)):
            if not torch.equal(out, exp):
                bad = (out != exp).nonzero()
                first = tuple(bad[0].tolist())
                errors.append(
                    f"reduce={op_name} {dtype} output {i}{' (in place)' if aliased[i] else ''}, run {run}: {len(bad)} "
                    f"of {out.numel()} wrong, first at {first}: {out[first].item()} expected {exp[first].item()}"
                )
    raise_on_all_ranks(ctx, pg, errors)
    return output_tensors


def raise_on_all_ranks(ctx, pg, errors):
    """Raises on every rank if any rank has errors, so that all ranks leave the test together (a rank that raised
    alone would leave the others waiting in the next collective)."""
    count = torch.tensor([len(errors)], dtype=torch.int64, device="cuda")
    counts = torch.empty(ctx.world_size, dtype=torch.int64, device="cuda")
    pg._allgather_base(counts, count).wait()
    failed = [r for r, c in enumerate(counts.tolist()) if c]
    if failed:
        ctx.assert_true(False, "; ".join(errors[:4]) if errors else f"wrong results on ranks {failed[:16]}")


@test
def test_reduce_scatter(ctx: TestContext):
    """Every rank's input covers the whole tensor, every rank's output its shard."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    for numel, combinations in [(64, all_combinations()), (100000, SOME)]:
        spec = {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}
        for dtype, op_name in combinations:
            run_reduce(ctx, pg, op_name, dtype, spec)


@test
def test_all_reduce(ctx: TestContext):
    """Every rank's input and output cover the whole tensor (sizes from 16 bytes to several chunks per rank)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    for numel, combinations in [(4096, all_combinations()), (8, SOME), ((1 << 20) + 64, SOME)]:
        spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
        for dtype, op_name in combinations:
            run_reduce(ctx, pg, op_name, dtype, spec, salt=numel)


@test
def test_all_reduce_in_place(ctx: TestContext):
    """All-reduce with the output tensor the input tensor."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 300000
    spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
    for dtype in [torch.float32, torch.bfloat16]:
        for op_name in ["sum", "max"]:
            run_reduce(ctx, pg, op_name, dtype, spec, in_place=True)


@test
def test_reduce_to_one(ctx: TestContext):
    """Every rank's input covers the tensor, one rank (the last) has the output; the others only send."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 50000
    outs = [[] for _ in range(n)]
    outs[n - 1] = [((0,), (numel,))]
    spec = {"0": ([numel], [[((0,), (numel,))]] * n, outs)}
    for dtype in [torch.float32, torch.int64]:
        run_reduce(ctx, pg, "sum", dtype, spec)


@test
def test_reduce_scatter_large(ctx: TestContext):
    """Reduce-scatter of 10 MiB per rank, and reduce to one rank of 10 MiB (across NVLink domains, the partial results
    of other domains arrive in parts)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 5 << 19
    spec = {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}
    run_reduce(ctx, pg, "sum", torch.float32, spec)
    outs = [[] for _ in range(n)]
    outs[0] = [((0,), (numel,))]
    spec = {"0": ([numel], [[((0,), (numel,))]] * n, outs)}
    run_reduce(ctx, pg, "avg", torch.bfloat16, spec)


@test
def test_reduce_scatter_tensor_ids(ctx: TestContext):
    """Reduce-scatter of 40 tensors in one op (across NVLink domains, more partial results per pair of ranks than they
    have notification slots, with small domains)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    spec = {}
    for t in range(40):
        numel = 64 * (t % 3 + 1)
        spec[f"t{t}"] = ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])
    for op_name in ["sum", "avg"]:
        run_reduce(ctx, pg, op_name, torch.float32, spec)


@test
def test_reduce_some_ranks(ctx: TestContext):
    """Inputs or outputs on some ranks only: inputs on ranks 0 and 1 (across NVLink domains, one domain has all
    sources), on the first and the last rank (one source per domain, read as it is), and outputs on the first 8 ranks
    of inputs on all (with more domains than those, domains that only have sources)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 40000
    whole = [((0,), (numel,))]
    first_two = [whole if r < 2 else [] for r in range(n)]
    first_last = [whole if r in (0, n - 1) else [] for r in range(n)]
    first_eight = [whole if r < 8 else [] for r in range(n)]
    for ins, outs in [(first_two, [whole] * n), (first_last, [whole] * n), ([whole] * n, first_eight)]:
        spec = {"0": ([numel], ins, outs)}
        for dtype, op_name in [(torch.float32, "sum"), (torch.bfloat16, "avg"), (torch.int32, "max")]:
            run_reduce(ctx, pg, op_name, dtype, spec)


@test
def test_all_reduce_large(ctx: TestContext):
    """All-reduce of 64 MiB (across NVLink domains, the pieces are split into parts, which the executors of every
    domain read and combine one by one); the inputs differ at every position, so that a part from the wrong place
    shows."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    for dtype, op_name in [(torch.float32, "sum"), (torch.bfloat16, "avg")]:
        numel = (64 << 20) // torch.tensor([], dtype=dtype).element_size()
        spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
        run_reduce(ctx, pg, op_name, dtype, spec)


@test
def test_all_reduce_in_place_one_source(ctx: TestContext):
    """In-place all-reduce with inputs on the first and the last rank only, whose outputs are their inputs, and outputs
    on all ranks. Across NVLink domains, a domain whose one source is the input of one of its destinations contributes
    a copy of it: the input is also the output that the domain's executor stores the sum to, while the executors of
    the other domains need it (they used to read it over RDMA as it was being overwritten)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    for numel in [1 << 18, 1 << 24]:
        whole = [((0,), (numel,))]
        spec = {"0": ([numel], [whole if r in (0, n - 1) else [] for r in range(n)], [whole] * n)}
        for dtype, op_name in [(torch.float32, "sum"), (torch.bfloat16, "avg")]:
            run_reduce(ctx, pg, op_name, dtype, spec, in_place=True, repeat=3)


@contextlib.contextmanager
def environment(name, value):
    """The environment variable `name` set to `value` (on every rank: compile_op checks that they agree)."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@test
def test_all_reduce_two_rounds(ctx: TestContext):
    """All-reduces across NVLink domains in two rounds wherever they can be (MOODIST_COMPILE_OP_ROUNDS=2; by default
    only with 3 or more domains, and large enough): each executor computes a share of its piece from the shares of
    the contributions of all domains, then copies the other executors' shares to its domain. All sizes (a few
    elements: one round), in place, inputs or outputs on some ranks only, 64 MiB (parts), and in place with inputs on
    two ranks (copies of the inputs)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    with environment("MOODIST_COMPILE_OP_ROUNDS", "2"):
        for numel, combinations in [(4096, SOME), (8, SOME), ((1 << 20) + 64, SOME)]:
            spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
            for dtype, op_name in combinations:
                run_reduce(ctx, pg, op_name, dtype, spec, salt=numel)
        numel = 300000
        whole = [((0,), (numel,))]
        spec = {"0": ([numel], [whole] * n, [whole] * n)}
        run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=2)
        run_reduce(ctx, pg, "max", torch.bfloat16, spec, in_place=True)
        first_two = [whole if r < 2 else [] for r in range(n)]
        first_last = [whole if r in (0, n - 1) else [] for r in range(n)]
        first_eight = [whole if r < 8 else [] for r in range(n)]
        for ins, outs in [(first_two, [whole] * n), (first_last, [whole] * n), ([whole] * n, first_eight)]:
            for dtype, op_name in [(torch.float32, "sum"), (torch.bfloat16, "avg")]:
                run_reduce(ctx, pg, op_name, dtype, {"0": ([numel], ins, outs)})
        for dtype, op_name in [(torch.float32, "sum"), (torch.bfloat16, "avg")]:
            numel = (64 << 20) // torch.tensor([], dtype=dtype).element_size()
            spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
            run_reduce(ctx, pg, op_name, dtype, spec)
        for numel in [1 << 18, 1 << 24]:
            whole = [((0,), (numel,))]
            spec = {"0": ([numel], [whole if r in (0, n - 1) else [] for r in range(n)], [whole] * n)}
            run_reduce(ctx, pg, "avg", torch.bfloat16, spec, in_place=True, repeat=3)


@test
def test_all_reduce_rounds_same_bits(ctx: TestContext):
    """Random floats: an all-reduce gives the same bits in one round and in two (MOODIST_COMPILE_OP_ROUNDS), and in
    one pass and in two phases (MOODIST_COMPILE_OP_PHASES), on every rank: the executors combine the same values in
    the same order either way."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    variants = [
        ("MOODIST_COMPILE_OP_ROUNDS", "1"),
        ("MOODIST_COMPILE_OP_ROUNDS", "2"),
        ("MOODIST_COMPILE_OP_PHASES", "1"),
        ("MOODIST_COMPILE_OP_PHASES", "2"),
    ]
    for dtype, op_name, numel in [(torch.bfloat16, "sum", 1 << 20), (torch.float32, "avg", 3 << 20)]:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(4321 + ctx.rank)
        x = torch.randn(numel, generator=gen, device="cuda").to(dtype)
        region = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
        results = []
        for name, value in variants:
            with environment(name, value):
                op = moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce=op_name)
            out = torch.full((numel,), 99, dtype=dtype, device="cuda")
            op([x], [out]).wait()
            results.append(out)
        errors = []
        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
        for (name, value), out in zip(variants[1:], results[1:]):
            if not torch.equal(results[0].view(bits), out.view(bits)):
                errors.append(f"reduce={op_name} {dtype}: {name}={value} gave other bits than one round")
        everyone = torch.empty(n * numel, dtype=dtype, device="cuda")
        pg._allgather_base(everyone, results[0]).wait()
        for r in range(n):
            if not torch.equal(everyone[r * numel : (r + 1) * numel], results[0]):
                errors.append(f"reduce={op_name} {dtype}: rank {r} has different bits")
                break
        raise_on_all_ranks(ctx, pg, errors)


@test
def test_all_reduce_two_phases(ctx: TestContext):
    """All-reduces in two phases within each NVLink domain (MOODIST_COMPILE_OP_PHASES=2): each executor computes its
    piece into its scratch tensor, then copies it to the destinations. All sizes, in place, inputs or outputs on some
    ranks only, and a reduce-scatter (one destination per piece: as it is)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    with environment("MOODIST_COMPILE_OP_PHASES", "2"):
        for numel, combinations in [(4096, all_combinations()), (8, SOME), ((1 << 20) + 64, SOME)]:
            spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
            for dtype, op_name in combinations:
                run_reduce(ctx, pg, op_name, dtype, spec, salt=numel)
        numel = 300000
        whole = [((0,), (numel,))]
        spec = {"0": ([numel], [whole] * n, [whole] * n)}
        run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=3)
        run_reduce(ctx, pg, "max", torch.bfloat16, spec, in_place=True)
        first_two = [whole if r < 2 else [] for r in range(n)]
        first_eight = [whole if r < 8 else [] for r in range(n)]
        for ins, outs in [(first_two, [whole] * n), ([whole] * n, first_eight)]:
            run_reduce(ctx, pg, "avg", torch.bfloat16, {"0": ([numel], ins, outs)})
        numel = (64 << 20) // 4
        spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
        run_reduce(ctx, pg, "sum", torch.float32, spec, repeat=2)
        numel = 50000
        spec = {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}
        run_reduce(ctx, pg, "sum", torch.float32, spec)


@test
def test_reduce_small_edges(ctx: TestContext):
    """Regions split into many edges (MOODIST_COMPILE_OP_MAX_EDGE_BYTES=4096; by default 1 GiB, as the kernel's tables
    hold 32-bit sizes): the pieces of reductions become several parts. Reduce-scatter, all-reduce, in place, and in two
    rounds."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    with environment("MOODIST_COMPILE_OP_MAX_EDGE_BYTES", "4096"):
        for numel in [8, 20000]:
            spec = {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}
            run_reduce(ctx, pg, "sum", torch.float32, spec)
        numel = 100000
        whole = [((0,), (numel,))]
        spec = {"0": ([numel], [whole] * n, [whole] * n)}
        run_reduce(ctx, pg, "avg", torch.bfloat16, spec)
        run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=2)
        with environment("MOODIST_COMPILE_OP_ROUNDS", "2"):
            run_reduce(ctx, pg, "sum", torch.float32, spec)


def node_local_ranks(ctx, pg):
    """Every rank's index among the ranks on its node, in rank order (as moodist numbers them)."""
    host = torch.tensor([zlib.crc32(socket.gethostname().encode())], dtype=torch.int64, device="cuda")
    hosts = torch.empty(ctx.world_size, dtype=torch.int64, device="cuda")
    pg._allgather_base(hosts, host).wait()
    hosts = hosts.tolist()
    return [hosts[:r].count(hosts[r]) for r in range(ctx.world_size)]


@test
def test_reduce_partial_rank_without_tensors(ctx: TestContext):
    """Reduce to rank 0 from rank 0 and from the ranks that are not the first of their node. Across NVLink domains,
    the partial result of another domain comes from a rank with rank 0's local rank there, which has no tensors in
    the op: its scratch tensor must stay untouched until rank 0 has read it, though the caller drops the op's future
    at once and the stream allocates and fills new tensors right after the op."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    local = node_local_ranks(ctx, pg)
    for numel in [1 << 20, 1 << 22]:
        whole = [((0,), (numel,))]
        ins = [whole if r == 0 or local[r] != 0 else [] for r in range(n)]
        outs = [whole if r == 0 else [] for r in range(n)]
        # (the expected values, from the op run as the other tests run it)
        expected = run_reduce(ctx, pg, "sum", torch.float32, {"0": ([numel], ins, outs)})
        region = TensorRegion(offset=[0], shape=[numel], device="cuda")
        op = moodist.compile_op(
            pg, dtype=torch.float32, inputs=[region] * len(ins[ctx.rank]), outputs=[region] * len(outs[ctx.rank]),
            reduce="sum",
        )
        x = [make_input(ctx.rank, 0, [numel], (0,), (numel,), torch.float32, 0) for _ in ins[ctx.rank]]
        errors = []
        for run in range(4):
            y = [torch.full((numel,), 99.0, device="cuda") for _ in outs[ctx.rank]]
            op(x, y)  # the future is dropped right away
            # New tensors on the stream, of the sizes of the scratch tensor's regions (the whole cell, or its parts)
            garbage = [torch.full((numel >> s,), 1e9, device="cuda") for s in range(4) for _ in range(2)]
            del garbage
            torch.cuda.synchronize()
            for out, exp in zip(y, expected):
                if not torch.equal(out, exp):
                    errors.append(f"numel {numel}, run {run}: {(out != exp).sum().item()} of {numel} wrong")
        raise_on_all_ranks(ctx, pg, errors)


@test
def test_reduce_overlaps(ctx: TestContext):
    """2-D regions that overlap in different numbers: rank r's input and output cover the column bands r and r + 1 of
    every row. A band is covered by two inputs and two outputs (cells with two destination ranks, which are not
    contiguous in the inputs or outputs: copies), the first and the last by one (copied as they are). avg divides by
    each element's number of inputs."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    rows = 4
    band = 16
    gshape = [rows, (n + 1) * band]
    regions = [[((0, r * band), (rows, 2 * band))] for r in range(n)]
    spec = {"0": (gshape, regions, regions)}
    for dtype in [torch.float32, torch.bfloat16, torch.int32]:
        for op_name in ops_for(dtype):
            run_reduce(ctx, pg, op_name, dtype, spec)


@test
def test_reduce_tensor_ids(ctx: TestContext):
    """Two tensors in one op: an all-reduced 2-D weight and a reduce-scattered 1-D bias."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    w = ([8, 256], [[((0, 0), (8, 256))]] * n, [[((0, 0), (8, 256))]] * n)
    b = ([n * 32], [[((0,), (n * 32,))]] * n, [[((r * 32,), (32,))] for r in range(n)])
    for op_name in ["sum", "avg"]:
        run_reduce(ctx, pg, op_name, torch.float32, {"w": w, "b": b})


@test
def test_reduce_deterministic(ctx: TestContext):
    """Random floats: the same inputs give the same bits on every run, and an all-reduce the same bits on every rank;
    the sum is close to the float64 sum."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 200000
    op = moodist.compile_op(
        pg,
        dtype=torch.float32,
        inputs=[TensorRegion(offset=[0], shape=[numel], device="cuda")],
        outputs=[TensorRegion(offset=[0], shape=[numel], device="cuda")],
        reduce="sum",
    )
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234 + ctx.rank)
    x = torch.randn(numel, generator=gen, device="cuda")
    a = torch.empty(numel, device="cuda")
    b = torch.empty(numel, device="cuda")
    op([x], [a]).wait()
    op([x], [b]).wait()
    ctx.assert_true(torch.equal(a, b), "two runs gave different results")
    everyone = torch.empty(n * numel, device="cuda")
    pg._allgather_base(everyone, a).wait()
    for r in range(n):
        ctx.assert_true(torch.equal(everyone[r * numel : (r + 1) * numel], a), f"rank {r} has different bits")
    reference = torch.zeros(numel, dtype=torch.float64, device="cuda")
    for r in range(n):
        g = torch.Generator(device="cuda")
        g.manual_seed(1234 + r)
        reference += torch.randn(numel, generator=g, device="cuda").double()
    err = (a.double() - reference).abs().max().item()
    ctx.assert_true(err < 1e-4 * n, f"max error {err} against the float64 sum")


@test
def test_reduce_errors(ctx: TestContext):
    """avg of integers and unknown reduce ops raise on every rank (compile_op takes float32, float64, float16,
    bfloat16, int32 and int64 at all)."""
    pg = create_process_group(ctx)
    region = [TensorRegion(offset=[0], shape=[64], device="cuda")]
    for dtype in [torch.int32, torch.int64]:
        ctx.assert_raises(
            RuntimeError,
            lambda: moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce="avg"),
        )
    ctx.assert_raises(
        ValueError, lambda: moodist.compile_op(pg, dtype=torch.float32, inputs=region, outputs=region, reduce="mean")
    )


@test
def test_reduce_special_values(ctx: TestContext):
    """NaN, infinities and zeros in an all-reduce: min and max propagate NaN, sums follow IEEE (inf + -inf = NaN),
    and a sum of -0 only is -0 (sums start from -0)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 64
    # vals[r]: rank r's input, small integers but for these elements
    vals = (torch.arange(n * numel, dtype=torch.float64, device="cuda").view(n, numel) * 5) % 7 - 3
    vals[0, 0] = math.nan  # NaN from the first rank
    vals[n - 1, 1] = math.nan  # from the last
    vals[:, 2] = -0.0  # -0 everywhere
    vals[0, 3] = math.inf
    vals[n - 1, 4] = -math.inf
    vals[0, 5] = math.inf
    vals[n - 1, 5] = -math.inf  # inf and -inf (with one rank: -inf)
    vals[0, 6] = math.nan
    vals[n - 1, 6] = math.inf
    vals[:, 7] = math.inf
    region = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
    bits = {torch.float16: torch.int16, torch.bfloat16: torch.int16, torch.float32: torch.int32,
            torch.float64: torch.int64}
    for dtype in FLOAT_DTYPES:
        x = vals.to(dtype)
        x64 = x.double()
        for op_name in ["sum", "avg", "min", "max"]:
            if op_name in ("sum", "avg"):
                e = x64.sum(0)
                e[(torch.signbit(x64) & (x64 == 0)).all(0)] = -0.0
                if op_name == "avg":
                    e = e / n if dtype == torch.float64 else e.float() / n
            elif op_name == "min":
                e = x64.amin(0)
            else:
                e = x64.amax(0)
            expected = e.to(dtype)
            out = torch.full((numel,), 99, dtype=dtype, device="cuda")
            op = moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce=op_name)
            op([x[ctx.rank].clone()], [out]).wait()
            # NaN where expected, and elsewhere the same bits (-0 is not 0)
            same_nan = torch.equal(out.isnan(), expected.isnan())
            a = torch.where(out.isnan(), torch.zeros_like(out), out).view(bits[dtype])
            b = torch.where(expected.isnan(), torch.zeros_like(expected), expected).view(bits[dtype])
            ctx.assert_true(
                same_nan and torch.equal(a, b),
                f"reduce={op_name} {dtype}: got {out[:8].tolist()}, expected {expected[:8].tolist()}",
            )


@test
def test_reduce_cpu_error(ctx: TestContext):
    """A reduction of CPU tensors raises on every rank (for now), and the group compiles ops after it."""
    if ctx.world_size < 2:
        return
    pg = create_process_group(ctx)
    n = ctx.world_size
    region = [TensorRegion(offset=[0], shape=[64], device="cpu")]
    ctx.assert_raises(
        RuntimeError,
        lambda: moodist.compile_op(pg, dtype=torch.float32, inputs=region, outputs=region, reduce="sum"),
    )
    spec = {"0": ([64], [[((0,), (64,))]] * n, [[((0,), (64,))]] * n)}
    run_reduce(ctx, pg, "sum", torch.float32, spec)


@test
def test_reduce_mismatch(ctx: TestContext):
    """Ranks that pass different reduce ops or dtypes get an error on every rank, from compile_op (checked in
    Python) and from compile_op_full (checked in C++, for the C++ API), and the group compiles ops after it."""
    if ctx.world_size < 2:
        return
    pg = create_process_group(ctx)
    n = ctx.world_size
    region = [TensorRegion(offset=[0], shape=[64], device="cuda")]
    op_name = "max" if ctx.rank == n - 1 else "sum"
    dtype = torch.float64 if ctx.rank == 0 else torch.float32
    ctx.assert_raises(
        ValueError, lambda: moodist.compile_op(pg, dtype=torch.float32, inputs=region, outputs=region, reduce=op_name)
    )
    ctx.assert_raises(
        ValueError, lambda: moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce="sum")
    )
    # compile_op_full takes the regions of every rank: (rank, offset, shape, tensor_id, device)
    everyone = [(r, (0,), (64,), "0", "cuda") for r in range(n)]
    ctx.assert_raises(
        RuntimeError, lambda: pg.compile_op_full(torch.float32, everyone, everyone, reduce=op_name, cpu_sync=False)
    )
    ctx.assert_raises(
        RuntimeError, lambda: pg.compile_op_full(dtype, everyone, everyone, reduce="sum", cpu_sync=False)
    )
    spec = {"0": ([64], [[((0,), (64,))]] * n, [[((0,), (64,))]] * n)}
    run_reduce(ctx, pg, "sum", torch.float32, spec)


@test
def test_reduce_configs(ctx: TestContext):
    """MOODIST_COMPILE_OP_CONFIGS with reductions: configurations that the reduce loop cannot use raise on every rank,
    also in a reduce-to-one, where only one rank executes reductions; 48 KiB chunks (2 stages) work."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 100000
    to_one = [[] for _ in range(n)]
    to_one[n - 1] = [((0,), (numel,))]
    reduce_to_one = {"0": ([numel], [[((0,), (numel,))]] * n, to_one)}
    all_reduce = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
    name = "MOODIST_COMPILE_OP_CONFIGS"
    previous = os.environ.get(name)
    try:
        # two pipelines per block; 64 KiB of shared memory for 16 KiB chunks
        for config in ["256:231424:8:32768:2", "256:65536:8:16384"]:
            os.environ[name] = config
            ctx.assert_raises(RuntimeError, lambda: run_reduce(ctx, pg, "sum", torch.float32, reduce_to_one))
        os.environ[name] = "256:231424:8:49152"
        run_reduce(ctx, pg, "sum", torch.float32, all_reduce)
        run_reduce(ctx, pg, "max", torch.bfloat16, reduce_to_one)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    run_reduce(ctx, pg, "sum", torch.float32, all_reduce)
