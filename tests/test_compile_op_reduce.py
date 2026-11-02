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
import random
import socket
import time
import zlib

import torch
import moodist
from moodist import TensorRegion
from framework import TestContext, test, create_process_group

moodist.enable_cuda_allocator()

# Set MOODIST_TEST_EXPECT_NVLS=1 when the group is in one NVLink domain that supports NVLS, to also check that the
# launches of the NVLS tests (test_all_reduce_nvls, test_all_reduce_nvls_edges, test_reduce_nvls_mismatch) use it
# where it applies: all-reduces with 8 ranks or more, reductions with one destination (reduce-scatters, reductions to
# one rank) with 12 or more
EXPECT_NVLS = os.environ.get("MOODIST_TEST_EXPECT_NVLS") == "1"

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
    """Reduce-scatter of 40 tensors in one op (across NVLink domains, more partial results per pair of ranks than their
    rings have slots, with small domains)."""
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
    """Random floats: an all-reduce gives the same bits in one round and in two (MOODIST_COMPILE_OP_ROUNDS), in one
    pass and in two phases (MOODIST_COMPILE_OP_PHASES), with reads and with writes (MOODIST_COMPILE_OP_TRANSFER), and
    with other slots for the rings of the scratch tensors (MOODIST_COMPILE_OP_SLOTS, MOODIST_COMPILE_OP_SLOT_BYTES: other
    parts), on every rank: the executors combine the same values in the same order either way. Without NVLS
    (MOODIST_COMPILE_OP_NVLS=0), whose switches combine them in an order of their own (see test_all_reduce_nvls)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    variants = [
        ("MOODIST_COMPILE_OP_ROUNDS", "1"),
        ("MOODIST_COMPILE_OP_ROUNDS", "2"),
        ("MOODIST_COMPILE_OP_PHASES", "1"),
        ("MOODIST_COMPILE_OP_PHASES", "2"),
        ("MOODIST_COMPILE_OP_TRANSFER", "read"),
        ("MOODIST_COMPILE_OP_TRANSFER", "write"),
        ("MOODIST_COMPILE_OP_SLOTS", "1"),
        ("MOODIST_COMPILE_OP_SLOT_BYTES", "4096"),
    ]
    for dtype, op_name, numel in [(torch.bfloat16, "sum", 1 << 20), (torch.float32, "avg", 3 << 20)]:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(4321 + ctx.rank)
        x = torch.randn(numel, generator=gen, device="cuda").to(dtype)
        region = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
        results = []
        for name, value in variants:
            with environment(name, value), environment("MOODIST_COMPILE_OP_NVLS", "0"):
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


def nvls_counts(ctx, pg, used):
    """Every rank's number of launches that used NVLS (used: this rank's), as a list."""
    counts = torch.empty(ctx.world_size, dtype=torch.int64, device="cuda")
    pg._allgather_base(counts, torch.tensor([used], dtype=torch.int64, device="cuda")).wait()
    return counts.tolist()


# Guard elements before and after outputs that are views into larger buffers (whole 16-byte vectors for every dtype):
# stores past an output's end (e.g. by a multicast store that ignored its predicate) change them
GUARD = 64


def guarded_outputs(pg, numel, dtype, count):
    """count buffers of the communication pool with GUARD elements before and after numel, and the views of the
    numel elements in the middle (the outputs; at the same place on every rank that allocates them alike)."""
    bigs = [moodist.comm_empty(numel + 2 * GUARD, dtype=dtype, group=pg) for _ in range(count)]
    return bigs, [big[GUARD:GUARD + numel] for big in bigs]


def guards_intact(big, numel, value):
    return bool((big[:GUARD] == value).all()) and bool((big[GUARD + numel:] == value).all())


@test
def test_all_reduce_nvls(ctx: TestContext):
    """All-reduces, a reduce-scatter and a reduce to one rank of buffers of the communication pool (at the same place on
    every rank), with NVLS where the group supports it (multimem.ld_reduce: the switches combine the values, in an order
    of their own, 16-bit floats in float32 and within 1 ulp of the exact sum): random values give results within
    rounding of the reduce loop's (MOODIST_COMPILE_OP_NVLS=0; integers the same, sums that overflow too: both wrap), the
    same bits on every rank and in every run, and nothing stored past the outputs (guard elements around them); NaN,
    infinities and -0 as in test_reduce_special_values (with NVLS, a sum of -0 only is +0; min and max of floating point
    types do not use NVLS, whose min and max do not propagate NaN, nor averages of float16, whose float16 sum could
    overflow: averages of 40000 here). Sizes from one 16-byte vector to several items per warp. With
    MOODIST_TEST_EXPECT_NVLS=1, the launches must use NVLS where it applies (see EXPECT_NVLS), and not where not."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    whole = lambda numel: [TensorRegion(offset=[0], shape=[numel], device="cuda")]  # noqa: E731
    # (dtype, reduce op, elements of an output, kind, inputs: random, or "overflow": sums that overflow, "large":
    # float16 averages whose sum overflows float16)
    cases = [
        (torch.bfloat16, "sum", (1 << 20) + 8 * n, "all_reduce", "random"),
        (torch.float32, "avg", 3 << 18, "all_reduce", "random"),
        (torch.bfloat16, "avg", 40000, "all_reduce", "random"),
        (torch.float16, "sum", 8, "all_reduce", "random"),
        (torch.float64, "sum", 4096, "all_reduce", "random"),
        (torch.float64, "avg", 2000, "all_reduce", "random"),
        (torch.int32, "sum", 1 << 16, "all_reduce", "random"),
        (torch.int32, "min", 1000, "all_reduce", "random"),
        (torch.int32, "max", 1 << 16, "all_reduce", "random"),
        (torch.int64, "sum", 2000, "all_reduce", "random"),
        (torch.int64, "min", 8 * 1000, "all_reduce", "random"),
        (torch.int64, "max", 3000, "all_reduce", "random"),
        (torch.int32, "sum", 4096, "all_reduce", "overflow"),
        (torch.int64, "sum", 4096, "all_reduce", "overflow"),
        (torch.float16, "avg", 4096, "all_reduce", "large"),
        (torch.float32, "sum", 40000, "reduce_scatter", "random"),
        (torch.bfloat16, "sum", 1 << 18, "reduce_one", "random"),
    ]
    tolerance = {torch.bfloat16: (2**-7, 1e-3), torch.float16: (2**-10, 1e-4), torch.float32: (1e-5, 1e-5),
                 torch.float64: (1e-12, 1e-12)}
    for dtype, op_name, numel, kind, values in cases:
        # (reduce to one rank: the last rank's output; every rank allocates the same buffers, at the same offsets)
        has_output = kind != "reduce_one" or ctx.rank == n - 1
        in_numel = n * numel if kind == "reduce_scatter" else numel
        inputs = whole(in_numel)
        if kind == "reduce_scatter":
            outputs = [TensorRegion(offset=[ctx.rank * numel], shape=[numel], device="cuda")]
        else:
            outputs = whole(numel) if has_output else []
        op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs, reduce=op_name)
        with environment("MOODIST_COMPILE_OP_NVLS", "0"):
            reference_op = moodist.compile_op(pg, dtype=dtype, inputs=inputs, outputs=outputs, reduce=op_name)
        gen = torch.Generator(device="cuda")
        gen.manual_seed(77 + ctx.rank)
        x = moodist.comm_empty(in_numel, dtype=dtype, group=pg)
        bigs, outs = guarded_outputs(pg, numel, dtype, 2)
        if values == "overflow":
            x.fill_((1 << (30 if dtype == torch.int32 else 62)) + ctx.rank)
        elif values == "large":
            x.fill_(40000)
        elif dtype.is_floating_point:
            x.copy_(torch.randn(in_numel, generator=gen, device="cuda"))
        else:
            x.copy_(torch.randint(-1000, 1000, (in_numel,), generator=gen, device="cuda"))
        reference = torch.empty(numel, dtype=dtype, device="cuda")
        launches0 = pg._get_option("nvls_launches")
        for big, out in zip(bigs, outs):
            big.fill_(99)
            op([x], [out] if has_output else []).wait()
        used = pg._get_option("nvls_launches") - launches0
        reference_op([x], [reference] if has_output else []).wait()
        errors = []
        what = f"{kind} reduce={op_name} {dtype} numel {numel} ({values})"
        bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[outs[0].element_size()]
        if has_output:
            if not torch.equal(outs[0].view(bits), outs[1].view(bits)):
                errors.append(f"{what}: two runs gave different bits")
            if dtype.is_floating_point:
                rtol, atol = tolerance[dtype]
                if not torch.allclose(outs[0].double(), reference.double(), rtol=rtol, atol=atol):
                    worst = (outs[0].double() - reference.double()).abs().max().item()
                    errors.append(f"{what}: off from the reduce loop's result by up to {worst}")
            elif not torch.equal(outs[0], reference):
                errors.append(f"{what}: not the reduce loop's result")
            if values == "large" and not bool((outs[0] == 40000).all()):
                errors.append(f"{what}: not the average 40000, e.g. {outs[0][:4].tolist()}")
            if not all(guards_intact(big, numel, 99) for big in bigs):
                errors.append(f"{what}: stores past the output")
        if kind == "all_reduce":
            everyone = torch.empty(n * numel, dtype=dtype, device="cuda")
            pg._allgather_base(everyone, outs[0]).wait()
            if not all(torch.equal(everyone[r * numel:(r + 1) * numel].view(bits), outs[0].view(bits))
                       for r in range(n)):
                errors.append(f"{what}: the ranks have different bits")
        if EXPECT_NVLS and n >= (8 if kind == "all_reduce" else 12):
            counts = nvls_counts(ctx, pg, used)
            # The ranks that execute reductions: every rank in a reduce-scatter and in an all-reduce of at least 16
            # bytes per rank, the last rank in a reduce to it, else some
            if kind == "reduce_scatter" or (kind == "all_reduce" and numel * x.element_size() >= 16 * n):
                executors = list(range(n))
            elif kind == "reduce_one":
                executors = [n - 1]
            else:
                executors = [r for r in range(n) if counts[r]]
            if dtype == torch.float16 and op_name == "avg":
                if any(counts):
                    errors.append(f"{what}: launches used NVLS (per rank {counts}), which float16 averages must not")
            elif not executors or any(counts[r] != 2 for r in executors):
                errors.append(f"{what}: launches that used NVLS per rank {counts}")
        raise_on_all_ranks(ctx, pg, errors)
        del x, outs, bigs
    # Special values, in the pool (where NVLS applies)
    check_special_values(ctx, pg, comm=True)


@test
def test_all_reduce_nvls_edges(ctx: TestContext):
    """NVLS reductions with several edges per rank: an all-reduce of two tensors and a reduce-scatter, in edges of at
    most 4 KiB (MOODIST_COMPILE_OP_MAX_EDGE_BYTES=4096) from pieces that are not multiples of 4 KiB, so that the
    kernel's items step from edge to edge, end edges part way, and its check has several entries per warp. Buffers of
    the communication pool: results within rounding of the reduce loop's, the same bits on every rank, nothing stored
    past the outputs; with MOODIST_TEST_EXPECT_NVLS=1, every rank's launches used NVLS (all-reduce from 8 ranks,
    reduce-scatter from 12)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    dtype = torch.float32
    na = 100000  # (pieces of 400000 / n bytes)
    nb = 3000
    nrs = 3000  # (the reduce-scatter's cells: 12000 bytes, three edges)
    errors = []
    with environment("MOODIST_COMPILE_OP_MAX_EDGE_BYTES", "4096"):
        regions = [TensorRegion(offset=[0], shape=[na], device="cuda", tensor_id="a"),
                   TensorRegion(offset=[0], shape=[nb], device="cuda", tensor_id="b")]
        ar = moodist.compile_op(pg, dtype=dtype, inputs=regions, outputs=regions, reduce="sum")
        rs_inputs = [TensorRegion(offset=[0], shape=[n * nrs], device="cuda")]
        rs_outputs = [TensorRegion(offset=[ctx.rank * nrs], shape=[nrs], device="cuda")]
        rs = moodist.compile_op(pg, dtype=dtype, inputs=rs_inputs, outputs=rs_outputs, reduce="sum")
        with environment("MOODIST_COMPILE_OP_NVLS", "0"):
            ar_ref = moodist.compile_op(pg, dtype=dtype, inputs=regions, outputs=regions, reduce="sum")
            rs_ref = moodist.compile_op(pg, dtype=dtype, inputs=rs_inputs, outputs=rs_outputs, reduce="sum")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(91 + ctx.rank)
    for name, op, ref, in_sizes, out_sizes, kind in [("all-reduce", ar, ar_ref, [na, nb], [na, nb], "all_reduce"),
                                                     ("reduce-scatter", rs, rs_ref, [n * nrs], [nrs], "rs")]:
        xs = [moodist.comm_empty(s, dtype=dtype, group=pg) for s in in_sizes]
        for x in xs:
            x.copy_(torch.randn(x.numel(), generator=gen, device="cuda"))
        guarded = [guarded_outputs(pg, s, dtype, 1) for s in out_sizes]
        bigs = [g[0][0] for g in guarded]
        outs = [g[1][0] for g in guarded]
        for big in bigs:
            big.fill_(99)
        refs = [torch.empty(s, dtype=dtype, device="cuda") for s in out_sizes]
        launches0 = pg._get_option("nvls_launches")
        op(xs, outs).wait()
        torch.cuda.synchronize()
        used = pg._get_option("nvls_launches") - launches0
        ref(xs, refs).wait()
        for i, (out, r, big) in enumerate(zip(outs, refs, bigs)):
            if not torch.allclose(out, r, rtol=1e-5, atol=1e-5):
                errors.append(f"{name}, output {i}: off from the reduce loop's result by up to "
                              f"{(out - r).abs().max().item()}")
            if not guards_intact(big, out.numel(), 99):
                errors.append(f"{name}, output {i}: stores past the output")
            if kind == "all_reduce":
                everyone = torch.empty(n * out.numel(), dtype=dtype, device="cuda")
                pg._allgather_base(everyone, out.contiguous()).wait()
                if not all(torch.equal(everyone[q * out.numel():(q + 1) * out.numel()], out) for q in range(n)):
                    errors.append(f"{name}, output {i}: the ranks have different bits")
        if EXPECT_NVLS and n >= (8 if kind == "all_reduce" else 12):
            counts = nvls_counts(ctx, pg, used)
            if any(c != 1 for c in counts):
                errors.append(f"{name}: launches that used NVLS per rank {counts}")
        del xs, outs, bigs, guarded
    raise_on_all_ranks(ctx, pg, errors)


@test
def test_reduce_nvls_mismatch(ctx: TestContext):
    """NVLS reductions fall back to the reduce loop when a rank's tensor is not where NVLS needs it: the last rank's
    input outside the communication pool, its input at another offset of the pool, its output outside the pool (an
    all-reduce's multicast stores would write elsewhere on it), and in a reduce-scatter its input outside the pool.
    Every rank allocates the pool buffers all the same (so that the pools stay alike). No launch may use NVLS, on any
    rank (a multimem.ld_reduce from a member without memory at the address ends the process), and the results are
    right; then with every tensor in place, NVLS where the group supports it (MOODIST_TEST_EXPECT_NVLS=1)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    dtype = torch.float32
    numel = 1 << 16
    last = ctx.rank == n - 1
    region = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
    ar = moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce="sum")
    rs = moodist.compile_op(pg, dtype=dtype, inputs=[TensorRegion(offset=[0], shape=[n * numel], device="cuda")],
                            outputs=[TensorRegion(offset=[ctx.rank * numel], shape=[numel], device="cuda")],
                            reduce="sum")

    def values(r, count):
        # small integers: exact sums
        return ((torch.arange(count, device="cuda") * 3 + r * 7) % 11 - 5).to(dtype)

    errors = []
    for case in ["input outside the pool", "input at another offset", "output outside the pool",
                 "reduce-scatter input outside the pool", "all in place"]:
        rs_case = case.startswith("reduce-scatter")
        in_numel = n * numel if rs_case else numel
        # (the pool buffers stay allocated until the end of the case, on every rank)
        comm_x = moodist.comm_empty(in_numel, dtype=dtype, group=pg)
        comm_x2 = moodist.comm_empty(in_numel, dtype=dtype, group=pg)
        comm_out = moodist.comm_empty(numel, dtype=dtype, group=pg)
        x = comm_x
        out = comm_out
        if last and case.endswith("input outside the pool"):
            x = torch.empty(in_numel, dtype=dtype, device="cuda")
        elif last and case == "input at another offset":
            x = comm_x2
        if last and case == "output outside the pool":
            out = torch.empty(numel, dtype=dtype, device="cuda")
        x.copy_(values(ctx.rank, in_numel))
        out.fill_(99)
        launches0 = pg._get_option("nvls_launches")
        (rs if rs_case else ar)([x], [out]).wait()
        torch.cuda.synchronize()
        counts = nvls_counts(ctx, pg, pg._get_option("nvls_launches") - launches0)
        everyone = sum(values(r, in_numel) for r in range(n))
        expected = everyone[ctx.rank * numel:(ctx.rank + 1) * numel] if rs_case else everyone
        if not torch.equal(out, expected):
            errors.append(f"{case}: wrong result, {(out != expected).sum().item()} elements")
        if case != "all in place" and any(counts):
            errors.append(f"{case}: launches used NVLS (per rank {counts})")
        if case == "all in place" and EXPECT_NVLS and n >= 8 and not all(counts):
            errors.append(f"{case}: launches that used NVLS per rank {counts}")
        del x, out, comm_x, comm_x2, comm_out
    raise_on_all_ranks(ctx, pg, errors)


@test
def test_reduce_writes(ctx: TestContext):
    """Reductions across NVLink domains with writes instead of reads (MOODIST_COMPILE_OP_TRANSFER=write): the ranks
    that compute partial results and shares write them into the scratch tensors of the ranks that need them, and
    signal their kernels. Reduce-scatter (up to parts), all-reduce of all sizes, in place, inputs or outputs on some
    ranks only, many partial results per pair of ranks, and in two rounds (64 MiB, and in place with inputs on two
    ranks: copies of the inputs)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    with environment("MOODIST_COMPILE_OP_TRANSFER", "write"):
        for numel in [64, 100000, 5 << 19]:
            spec = {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}
            run_reduce(ctx, pg, "sum", torch.float32, spec)
        for numel, combinations in [(4096, SOME), (8, SOME), ((1 << 20) + 64, SOME)]:
            spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
            for dtype, op_name in combinations:
                run_reduce(ctx, pg, op_name, dtype, spec, salt=numel)
        numel = 300000
        whole = [((0,), (numel,))]
        spec = {"0": ([numel], [whole] * n, [whole] * n)}
        run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=2)
        first_two = [whole if r < 2 else [] for r in range(n)]
        first_last = [whole if r in (0, n - 1) else [] for r in range(n)]
        first_eight = [whole if r < 8 else [] for r in range(n)]
        for ins, outs in [(first_two, [whole] * n), (first_last, [whole] * n), ([whole] * n, first_eight)]:
            run_reduce(ctx, pg, "avg", torch.bfloat16, {"0": ([numel], ins, outs)})
        spec = {}
        for t in range(40):
            numel = 64 * (t % 3 + 1)
            spec[f"t{t}"] = ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])
        run_reduce(ctx, pg, "avg", torch.float32, spec)
        with environment("MOODIST_COMPILE_OP_ROUNDS", "2"):
            numel = (64 << 20) // 2
            spec = {"0": ([numel], [[((0,), (numel,))]] * n, [[((0,), (numel,))]] * n)}
            run_reduce(ctx, pg, "avg", torch.bfloat16, spec)
            numel = 1 << 18
            whole = [((0,), (numel,))]
            spec = {"0": ([numel], [whole if r in (0, n - 1) else [] for r in range(n)], [whole] * n)}
            run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=3)


@test
def test_reduce_rings(ctx: TestContext):
    """Reductions across NVLink domains with one slot per ring of the scratch tensors, of 4 KiB
    (MOODIST_COMPILE_OP_SLOTS=1, MOODIST_COMPILE_OP_SLOT_BYTES=4096; by default slots of up to 2 MiB, about 4 MiB per
    ring): parts of 4 KiB (in two rounds, shares), and every partial result, input, copy and share that crosses domains
    takes its slot only once the one before it there has been used, many times per op. Reduce-scatter (up to 128
    parts), reduce to one, all-reduce in one round and in two (in place, inputs on two ranks: copies, outputs on 8
    ranks), many tensors, an all-reduce and a reduce-scatter in one op; then 2 slots of 8 KiB."""
    pg = create_process_group(ctx)
    n = ctx.world_size

    def whole(numel):
        return [((0,), (numel,))]

    with environment("MOODIST_COMPILE_OP_SLOTS", "1"), environment("MOODIST_COMPILE_OP_SLOT_BYTES", "4096"):
        for numel in [64, 20000, 1 << 17]:
            spec = {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}
            run_reduce(ctx, pg, "sum", torch.float32, spec, salt=numel)
        numel = 100000
        outs = [[] for _ in range(n)]
        outs[n - 1] = whole(numel)
        run_reduce(ctx, pg, "avg", torch.bfloat16, {"0": ([numel], [whole(numel)] * n, outs)})
        for rounds in ["1", "2"]:
            with environment("MOODIST_COMPILE_OP_ROUNDS", rounds):
                spec = {"0": ([numel], [whole(numel)] * n, [whole(numel)] * n)}
                run_reduce(ctx, pg, "avg", torch.bfloat16, spec)
                run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=2)
                first_last = [whole(numel) if r in (0, n - 1) else [] for r in range(n)]
                spec = {"0": ([numel], first_last, [whole(numel)] * n)}
                run_reduce(ctx, pg, "sum", torch.float32, spec, in_place=True, repeat=2)
                first_eight = [whole(numel) if r < 8 else [] for r in range(n)]
                run_reduce(ctx, pg, "max", torch.int32, {"0": ([numel], [whole(numel)] * n, first_eight)})
        spec = {}
        for t in range(40):
            numel = 64 * (t % 3 + 1)
            spec[f"t{t}"] = ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])
        run_reduce(ctx, pg, "avg", torch.float32, spec)
        w = ([8, 4096], [[((0, 0), (8, 4096))]] * n, [[((0, 0), (8, 4096))]] * n)
        b = ([n * 4096], [[((0,), (n * 4096,))]] * n, [[((r * 4096,), (4096,))] for r in range(n)])
        run_reduce(ctx, pg, "sum", torch.float32, {"w": w, "b": b}, repeat=2)
    with environment("MOODIST_COMPILE_OP_SLOTS", "2"), environment("MOODIST_COMPILE_OP_SLOT_BYTES", "8192"):
        numel = 1 << 18
        run_reduce(ctx, pg, "sum", torch.float32, {"0": ([numel], [whole(numel)] * n, [whole(numel)] * n)})
        spec = {"0": ([n * 50000], [[((0,), (n * 50000,))]] * n, [[((r * 50000,), (50000,))] for r in range(n)])}
        run_reduce(ctx, pg, "avg", torch.bfloat16, spec)


@test
def test_reduce_uneven_rings(ctx: TestContext):
    """Reductions across NVLink domains whose ranks lay out their rings differently (by default, a rank with more rings
    gives each fewer slots): a rank sets the credits of the slots that it reads to generations of the rank that
    computes them, whose ring can go through more generations than the rings of the reading rank. An all-reduce of
    64 KiB and a reduce of 128 MiB to rank 0 in one op (the ranks that compute the partial results of the other domains
    for rank 0 have more rings than rank 0), and an all-reduce of 128 MiB with outputs on all ranks but the last two
    (the executors of their domain, fewer than in the others, have more rings), in one round and in two."""
    pg = create_process_group(ctx)
    n = ctx.world_size

    def whole(numel):
        return [((0,), (numel,))]

    small = 16384
    numel = 32 << 20
    to_first = [whole(numel)] + [[] for _ in range(n - 1)]
    spec = {"w": ([small], [whole(small)] * n, [whole(small)] * n), "b": ([numel], [whole(numel)] * n, to_first)}
    run_reduce(ctx, pg, "sum", torch.float32, spec)
    most = [whole(numel) if r < n - 2 else [] for r in range(n)]
    for rounds in ["1", "2"]:
        with environment("MOODIST_COMPILE_OP_ROUNDS", rounds):
            run_reduce(ctx, pg, "sum", torch.float32, {"0": ([numel], [whole(numel)] * n, most)})


@test
def test_reduce_ring(ctx: TestContext):
    """Reductions along a ring of the NVLink domains wherever they can be (MOODIST_COMPILE_OP_RING=1; by default with 3
    or more domains of the same size, for a rank's one cell with sources in every domain, of 6 MiB or more): the domain
    after the executor's computes its partial result, each next one adds its own, and the executor its domain's. With
    fewer domains, plain reductions. Reduce-scatter of every dtype and op, in parts, in slots of 4 KiB (many parts, many
    uses of a slot) with groups of 1, 2, 4, 8 and 16 parts (MOODIST_COMPILE_OP_RING_GROUP), reduce to one rank, a
    reduce-scatter and an all-reduce in one op (the all-reduce not along a ring), two reduce-scattered tensors (two such
    cells per rank: none along a ring), a reduce-scatter with a second tensor reduced to rank 0 (rank 0 has two cells,
    the others go along rings: the ranks at rank 0's position compute steps of rings and partial results for rank 0),
    domains with one source (rings that start from a copy of an input); then by default, 1 and 8 MiB per rank."""
    pg = create_process_group(ctx)
    n = ctx.world_size

    def reduce_scatter(numel):
        return {"0": ([n * numel], [[((0,), (n * numel,))]] * n, [[((r * numel,), (numel,))] for r in range(n)])}

    def whole(numel):
        return [((0,), (numel,))]

    with environment("MOODIST_COMPILE_OP_RING", "1"):
        for dtype, op_name in all_combinations():
            run_reduce(ctx, pg, op_name, dtype, reduce_scatter(64))
        for numel in [100000, 5 << 19]:
            for dtype, op_name in SOME:
                run_reduce(ctx, pg, op_name, dtype, reduce_scatter(numel), salt=numel, repeat=2)
        with environment("MOODIST_COMPILE_OP_SLOT_BYTES", "4096"):
            for group in ["1", "2", "4", "8", "16"]:
                with environment("MOODIST_COMPILE_OP_RING_GROUP", group):
                    run_reduce(ctx, pg, "sum", torch.float32, reduce_scatter(20000), repeat=2)
                    run_reduce(ctx, pg, "avg", torch.bfloat16, reduce_scatter(1 << 16))
        for numel in [64, 300000]:
            to_last = [[] for _ in range(n - 1)] + [whole(numel)]
            run_reduce(ctx, pg, "sum", torch.float32, {"0": ([numel], [whole(numel)] * n, to_last)})
        w = ([8, 4096], [[((0, 0), (8, 4096))]] * n, [[((0, 0), (8, 4096))]] * n)
        run_reduce(ctx, pg, "sum", torch.float32, {"w": w, "b": reduce_scatter(50000)["0"]}, repeat=2)
        run_reduce(ctx, pg, "avg", torch.float32, {"a": reduce_scatter(1000)["0"], "b": reduce_scatter(3000)["0"]})
        to_first = [whole(1000)] + [[] for _ in range(n - 1)]
        for numel in [3000, 300000]:
            b = ([1000], [whole(1000)] * n, to_first)
            run_reduce(ctx, pg, "sum", torch.float32, {"a": reduce_scatter(numel)["0"], "b": b}, salt=numel, repeat=2)
        # Domains (nodes here) with one source: the odd ones, where only the first rank has inputs; a ring whose first
        # step is there starts from a copy of that input, which the next step adds as an input, not as a partial result.
        # (Three sources in the others: 2 per domain on average, so that averages of the test's inputs are exact.)
        local = int(os.environ.get("LOCAL_WORLD_SIZE", n))
        some = [whole(n * 1000) if r % local == 0 or (r // local % 2 == 0 and r % local < 3) else [] for r in range(n)]
        for op_name in ["sum", "avg", "max"]:
            run_reduce(ctx, pg, op_name, torch.float32, {"0": ([n * 1000], some, reduce_scatter(1000)["0"][2])})
    for numel in [1 << 18, 2 << 20]:
        run_reduce(ctx, pg, "sum", torch.float32, reduce_scatter(numel))
        run_reduce(ctx, pg, "avg", torch.bfloat16, reduce_scatter(numel * 2))


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
    the sum is close to the float64 sum. Without NVLS (MOODIST_COMPILE_OP_NVLS=0): whether a launch uses it depends on
    where its tensors are, and the two outputs here may be in place on every rank or not (see test_all_reduce_nvls)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 200000
    with environment("MOODIST_COMPILE_OP_NVLS", "0"):
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


def check_special_values(ctx, pg, comm):
    """All-reduces of NaN, infinities and zeros, of every floating point type and op: min and max propagate NaN, sums
    follow IEEE (inf + -inf = NaN), and a sum of -0 only is -0 (the reduce loop's sums start from -0), or +0 where the
    launch used NVLS (the switches' sums start from +0). comm: the tensors from the communication pool (at the same
    place on every rank: NVLS where the group supports it, for sums and averages but those of float16, which with
    MOODIST_TEST_EXPECT_NVLS=1 must use it, and nothing else), else from the pool as a test allocates them."""
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
    errors = []
    for dtype in FLOAT_DTYPES:
        x64 = vals.to(dtype).double()
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
            op = moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce=op_name)
            if comm:
                x = moodist.comm_empty(numel, dtype=dtype, group=pg)
                out = moodist.comm_empty(numel, dtype=dtype, group=pg)
                x.copy_(vals[ctx.rank].to(dtype))
                out.fill_(99)
            else:
                x = vals[ctx.rank].to(dtype).clone()
                out = torch.full((numel,), 99, dtype=dtype, device="cuda")
            launches0 = pg._get_option("nvls_launches")
            op([x], [out]).wait()
            used = torch.tensor([pg._get_option("nvls_launches") - launches0], dtype=torch.int64, device="cuda")
            everyone = torch.empty(n, dtype=torch.int64, device="cuda")
            pg._allgather_base(everyone, used).wait()
            if op_name in ("sum", "avg") and bool((everyone != 0).any()):
                expected[2] = 0.0
            # (with MOODIST_TEST_EXPECT_NVLS=1, the comm buffers' launches use NVLS where it applies: sums, averages but
            # those of float16)
            nvls_applies = op_name == "sum" or (op_name == "avg" and dtype != torch.float16)
            if comm and EXPECT_NVLS and n >= 8 and bool((everyone != 0).any()) != nvls_applies:
                errors.append(f"reduce={op_name} {dtype} (comm buffers): launches that used NVLS per rank "
                              f"{everyone.tolist()}, expected {'some' if nvls_applies else 'none'}")
            # NaN where expected, and elsewhere the same bits (-0 is not 0)
            same_nan = torch.equal(out.isnan(), expected.isnan())
            a = torch.where(out.isnan(), torch.zeros_like(out), out).view(bits[dtype])
            b = torch.where(expected.isnan(), torch.zeros_like(expected), expected).view(bits[dtype])
            if not (same_nan and torch.equal(a, b)):
                errors.append(f"reduce={op_name} {dtype}{' (comm buffers)' if comm else ''}: got {out[:8].tolist()}, "
                              f"expected {expected[:8].tolist()}")
            del x, out
    raise_on_all_ranks(ctx, pg, errors)


@test
def test_reduce_special_values(ctx: TestContext):
    """NaN, infinities and zeros in all-reduces (see check_special_values)."""
    pg = create_process_group(ctx)
    check_special_values(ctx, pg, comm=False)


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


@test
def test_reduce_interleaved(ctx: TestContext):
    """Reductions interleaved with the process group's collectives on one stream: 400 ops without waiting in between,
    with random delays between the calls on every rank. Each op's words in the concurrency slots (tensor lists, step
    values, notifications, arrival values) are reused by the op 16 later (the step values by ops of any kind), and in
    a reduction a rank exchanges data over RDMA only with some ranks (here also one of the inputs of two ranks, in
    which most ranks only receive). See also test_reduce_two_streams."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    numel = 4096
    whole = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
    full = [TensorRegion(offset=[0], shape=[n * numel], device="cuda")]
    shard = [TensorRegion(offset=[ctx.rank * numel], shape=[numel], device="cuda")]
    two = ctx.rank in (0, n - 1)
    all_reduce = moodist.compile_op(pg, dtype=torch.float32, inputs=whole, outputs=whole, reduce="sum")
    reduce_scatter = moodist.compile_op(pg, dtype=torch.float32, inputs=full, outputs=shard, reduce="sum")
    two_inputs = moodist.compile_op(pg, dtype=torch.float32, inputs=whole if two else [], outputs=whole, reduce="sum")
    value = float(ctx.rank + 1)
    total = float(n * (n + 1) // 2)
    total_two = float(sum(r + 1 for r in {0, n - 1}))
    gathered = torch.arange(1, n + 1, dtype=torch.float32, device="cuda").repeat_interleave(numel)

    def filled(numel):
        return torch.full((numel,), value, device="cuda")

    kinds = random.Random(1)  # the same ops on every rank
    delays = random.Random(ctx.rank)
    checks = []
    for i in range(400):
        if delays.random() < 0.05:
            time.sleep(delays.uniform(0, 0.003))
        kind = kinds.randrange(7)
        if kind == 0:
            y = torch.empty(numel, device="cuda")
            all_reduce([filled(numel)], [y])
            checks.append(("compile_op all-reduce", y, total))
        elif kind == 1:
            y = torch.empty(numel, device="cuda")
            reduce_scatter([filled(n * numel)], [y])
            checks.append(("compile_op reduce-scatter", y, total))
        elif kind == 2:
            y = torch.empty(numel, device="cuda")
            two_inputs([filled(numel)] if two else [], [y])
            checks.append(("compile_op reduction of two inputs", y, total_two))
        elif kind == 3:
            t = filled(numel)
            pg.allreduce([t]).wait()
            checks.append(("all-reduce", t, total))
        elif kind == 4:
            t = torch.empty(n * numel, device="cuda")
            pg._allgather_base(t, filled(numel)).wait()
            checks.append(("all-gather", t, gathered))
        elif kind == 5:
            t = torch.empty(numel, device="cuda")
            pg._reduce_scatter_base(t, filled(n * numel)).wait()
            checks.append(("reduce-scatter", t, total))
        else:
            root = i % n
            t = filled(numel)
            pg.broadcast(t, root).wait()
            checks.append((f"broadcast from {root}", t, float(root + 1)))
    torch.cuda.synchronize()
    errors = []
    for i, (name, t, expected) in enumerate(checks):
        if not isinstance(expected, torch.Tensor):
            expected = torch.full_like(t, expected)
        if not torch.equal(t, expected):
            bad = (t != expected).nonzero()
            first = bad[0].item()
            errors.append(
                f"op {i} ({name}): {len(bad)} of {t.numel()} wrong, first at {first}: {t[first].item()} expected "
                f"{expected[first].item()}"
            )
    raise_on_all_ranks(ctx, pg, errors)


@test
def test_reduce_two_streams(ctx: TestContext):
    """Reductions on one stream, the process group's reduce-scatters and all-reduces on another. An op starts on a
    rank once the rank is done with the previous op of its concurrency slot (16 ops earlier), while other ranks may
    still be in it, here in a collective on the other stream that waits for exact step values from every rank (the
    process group's reduce-scatter across nodes: each reader tells each source that it is done). A reduction across
    NVLink domains writes into the memory of ranks it exchanges nothing with too (arrival values, see WorkCustom),
    which must not overwrite such a step value before the collective has read it, or the collective waits forever.
    Each round: a large reduce-scatter or all-reduce, which takes longer than the reductions and which the ranks finish
    at different times (random delays before it), then 16 small reductions, the last one or two of which reuse the
    collective's concurrency slots (a reduce-scatter takes one, an all-reduce two: those of its reduce-scatter and
    its all-gather). The reduce-scatter's RDMA goes between ranks of the same local rank on different nodes only,
    which with 2 nodes are also the reduction's peers: this needs 3 nodes or more, and a rank that is late to read
    the collective's step values (its CPU thread preempted, say); with step values as arrival values, a run on 4
    nodes in which one rank waited 3 ms before reading them hung. The all-reduces' all-gathers (rings) also check
    that the reductions' tensor lists do not overwrite a ring's (see Group::customDyns): on 64 GPUs (16 nodes) they
    did, and a rank read from its ring neighbor with another rank's address and key (a remote access error)."""
    pg = create_process_group(ctx)
    n = ctx.world_size
    side = torch.cuda.Stream()
    shard = max((32 << 20) // n, 4096)  # float32 elements per rank (128 MiB per reduce-scatter in all)
    numel = 4096
    whole = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
    all_reduce = moodist.compile_op(pg, dtype=torch.float32, inputs=whole, outputs=whole, reduce="sum")
    value = float(ctx.rank + 1)
    total = float(n * (n + 1) // 2)
    delays = random.Random(ctx.rank)
    checks = []  # (name, a bool tensor: whether the output was right, computed on the op's stream)
    for i in range(100):
        if delays.random() < 0.3:
            time.sleep(delays.uniform(0, 0.003))
        # (nothing orders the two streams but the ops' concurrency slots)
        with torch.cuda.stream(side):
            x = torch.full((n * shard,), value, device="cuda")
            if i % 2 == 0:
                y = torch.empty(shard, device="cuda")
                pg._reduce_scatter_base(y, x).wait()
                checks.append((f"round {i}: reduce-scatter", (y == total).all()))
            else:
                pg.allreduce([x]).wait()  # (a reduce-scatter and an all-gather)
                checks.append((f"round {i}: all-reduce", (x == total).all()))
            del x
        for j in range(16):
            y = torch.empty(numel, device="cuda")
            all_reduce([torch.full((numel,), value, device="cuda")], [y])
            checks.append((f"round {i}: compile_op all-reduce {j}", (y == total).all()))
    torch.cuda.synchronize()
    errors = [f"{name}: wrong result" for name, ok in checks if not ok.item()]
    raise_on_all_ranks(ctx, pg, errors)
