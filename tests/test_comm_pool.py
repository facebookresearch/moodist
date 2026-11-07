"""
Tests for moodist's communication pools (moodist.comm_empty, the process group's allocate_tensor): a buffer gets the
same offset in the group's pool on every rank when all ranks allocate and free in the same order, whatever else they
allocate; compile_op ops store to such buffers with NVLink multicast (NVLS) where the group supports it (NVLink domains
of at least 16 ranks; an all-gather across domains stores to each domain with multicast stores too), and to each rank
otherwise.

Set MOODIST_TEST_EXPECT_NVLS=1 when the group's NVLink domains support NVLS, to also check that the launches use it
(copies in domains of 16 ranks or more, all of the same size).
"""

import os

import torch
import moodist
from moodist import TensorRegion
from framework import TestContext, test, create_process_group

moodist.enable_cuda_allocator()

EXPECT_NVLS = os.environ.get("MOODIST_TEST_EXPECT_NVLS") == "1"


def nvlink_domains(pg, ctx):
    """Every rank's NVLink domain (its first rank), as a list per rank. (A collective: the process group's all-gather,
    which may use NVLS too, so not between two reads of nvls_launches.)"""
    return [row[0] for row in gather_ints(pg, ctx, [pg._get_option("nvlink_domain")])]


def expect_nvls(ctx, pg, domains=None):
    """Whether the all-gathers must use NVLS: copies use it in NVLink domains of 16 ranks or more, across domains too
    if every rank of a domain stores as many chunks (domains of the same size, whose proxies each read the same)."""
    if not EXPECT_NVLS or ctx.world_size < 16:
        return False
    domains = domains if domains is not None else nvlink_domains(pg, ctx)
    sizes = [domains.count(d) for d in set(domains)]
    return min(sizes) >= 16 and len(set(sizes)) == 1


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


def allgather_op(pg, ctx, numel, dtype=torch.float32):
    return moodist.compile_op(
        pg,
        dtype=dtype,
        inputs=[TensorRegion(offset=[ctx.rank * numel], shape=[numel], device="cuda")],
        outputs=[TensorRegion(offset=[0], shape=[numel * ctx.world_size], device="cuda")],
    )


def expected_allgather(ctx, numel, it):
    return (torch.arange(ctx.world_size, device="cuda", dtype=torch.float32) * 1000 + it).repeat_interleave(numel)


@test
def test_comm_pool_same_offsets(ctx: TestContext):
    """Comm buffers get the same offsets on every rank, with rank-dependent allocations from the main pool in
    between, frees, and reuse of freed memory."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    sizes = [1 << 10, 3 << 20, 100, 64 << 20, 4096, 1 << 20, 17 << 20]
    keep = []
    offsets = []
    for i, n in enumerate(sizes):
        # main-pool noise of a different size on every rank
        noise = torch.empty((ctx.rank + 1) * 12345 + i, dtype=torch.uint8, device="cuda")
        t = moodist.comm_empty(n, dtype=torch.uint8, group=pg)
        offsets.append(moodist.comm_pool_offset(t))
        keep.append(t)
        del noise
    ctx.assert_true(all(o >= 0 for o in offsets), f"offsets {offsets}")
    ctx.assert_true(len(set(offsets)) == len(offsets), f"overlapping offsets {offsets}")
    # free some, allocate again (the same order on every rank)
    del keep[1], keep[3]
    for n in [2 << 20, 50 << 20, 512]:
        t = moodist.comm_empty(n, dtype=torch.uint8, group=pg)
        offsets.append(moodist.comm_pool_offset(t))
        keep.append(t)
    # a view inside a comm buffer has the offset of its start
    v = keep[0][100:200]
    offsets.append(moodist.comm_pool_offset(v))
    ctx.assert_true(offsets[-1] == moodist.comm_pool_offset(keep[0]) + 100, f"view offset {offsets[-1]}")
    # not in the pool
    ctx.assert_true(moodist.comm_pool_offset(torch.empty(1000, device="cuda")) == -1, "main pool tensor")
    assert_same_on_all_ranks(pg, ctx, offsets, "comm pool offsets")


@test
def test_comm_pool_streams(ctx: TestContext):
    """A buffer freed while another stream uses it: the next allocation of that memory waits for that stream on the
    GPU (the offsets stay the same on every rank, the data is not overwritten early)."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    # A size no other test uses, larger than a slab: a gets a slab of its own, which b then gets again
    n = (400 << 20) // 4
    side = torch.cuda.Stream()
    a = moodist.comm_empty(n, dtype=torch.float32, group=pg)
    a.fill_(1.0)
    off_a = moodist.comm_pool_offset(a)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # a long kernel that reads a, then copies it out
        torch.cuda._sleep(int(50e6))
        out = a.clone()
    a.record_stream(side)
    del a
    # reuses a's memory: must not overwrite it before the side stream's copy has read it
    b = moodist.comm_empty(n, dtype=torch.float32, group=pg)
    off_b = moodist.comm_pool_offset(b)
    b.fill_(2.0)
    torch.cuda.synchronize()
    ctx.assert_true(off_b == off_a, f"the freed memory was not reused: {off_a} then {off_b}")
    ctx.assert_true(bool((out == 1.0).all()), "the side stream read memory that was reused early")
    ctx.assert_true(bool((b == 2.0).all()), "b")
    assert_same_on_all_ranks(pg, ctx, [off_a, off_b], "offsets")


@test
def test_comm_pool_allgather(ctx: TestContext):
    """An all-gather into comm buffers allocated after compiling (where buffers from the main pool are at different
    offsets on different ranks); with NVLS where the group supports it."""
    if ctx.world_size < 2:
        return
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for numel in [16384, 1 << 20]:
        op = allgather_op(pg, ctx, numel)
        inp = torch.empty(numel, device="cuda", dtype=torch.float32)
        launches0 = pg._get_option("nvls_launches")
        iterations = 8
        for it in range(iterations):
            out = moodist.comm_empty(numel * ctx.world_size, dtype=torch.float32, group=pg)
            out.fill_(-1)
            inp.fill_(ctx.rank * 1000 + it)
            op([inp], [out]).wait()
            ok = bool((out == expected_allgather(ctx, numel, it)).all())
            ctx.assert_true(ok, f"numel {numel} iteration {it}: wrong result")
            del out
        torch.cuda.synchronize()
        used = pg._get_option("nvls_launches") - launches0
        if expect_nvls(ctx, pg):
            ctx.assert_true(used == iterations, f"numel {numel}: {used} of {iterations} launches used NVLS")


@test
def test_comm_pool_allgather_outside(ctx: TestContext):
    """Every rank's output outside the pool: the launches run the op's unicast kernel (no NVLS), correctly; then every
    rank's in the pool: its NVLS kernel (where supported). (One rank's outside the pool alone is an error: see
    test_nvls_errors.py.)"""
    if ctx.world_size < 2:
        return
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    domains = nvlink_domains(pg, ctx)
    numel = 1 << 18
    op = allgather_op(pg, ctx, numel)
    inp = torch.empty(numel, device="cuda", dtype=torch.float32)
    launches0 = pg._get_option("nvls_launches")
    for it in range(4):
        out = torch.empty(numel * ctx.world_size, device="cuda", dtype=torch.float32)
        inp.fill_(ctx.rank * 1000 + it)
        op([inp], [out]).wait()
        ctx.assert_true(bool((out == expected_allgather(ctx, numel, it)).all()), f"outside iteration {it}")
        del out
    torch.cuda.synchronize()
    used = pg._get_option("nvls_launches") - launches0
    ctx.assert_true(used == 0, f"{used} launches used NVLS with the outputs outside the pool")
    launches0 = pg._get_option("nvls_launches")
    for it in range(4):
        out = moodist.comm_empty(numel * ctx.world_size, dtype=torch.float32, group=pg)
        inp.fill_(ctx.rank * 1000 + 10 + it)
        op([inp], [out]).wait()
        ctx.assert_true(bool((out == expected_allgather(ctx, numel, 10 + it)).all()), f"iteration {it}")
        del out
    torch.cuda.synchronize()
    used = pg._get_option("nvls_launches") - launches0
    if expect_nvls(ctx, pg, domains):
        ctx.assert_true(used == 4, f"{used} of 4 launches used NVLS")


@test
def test_comm_pool_allgather_sizes(ctx: TestContext):
    """All-gathers into comm buffers, out of place and in place (the input a view of the output): of 9 MiB per rank,
    whose transfers across NVLink domains are split into chunks (from 8 MiB, see proxyRemoteEdges in compile_op.cc),
    and of 1 MiB + 16 bytes per rank (a last NVLS chunk of 16 bytes); with NVLS where the group supports it."""
    if ctx.world_size < 2:
        return
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    for numel in [(9 << 20) // 4, (1 << 18) + 4]:
        op = allgather_op(pg, ctx, numel)
        inp = torch.empty(numel, device="cuda", dtype=torch.float32)
        launches0 = pg._get_option("nvls_launches")
        iterations = 4
        for it in range(iterations):
            out = moodist.comm_empty(numel * ctx.world_size, dtype=torch.float32, group=pg)
            out.fill_(-1)
            in_place = it % 2 == 1
            src = out[ctx.rank * numel : (ctx.rank + 1) * numel] if in_place else inp
            src.fill_(ctx.rank * 1000 + it)
            op([src], [out]).wait()
            ok = bool((out == expected_allgather(ctx, numel, it)).all())
            ctx.assert_true(ok, f"numel {numel} iteration {it} (in place: {in_place}): wrong result")
            del out, src
        torch.cuda.synchronize()
        used = pg._get_option("nvls_launches") - launches0
        if expect_nvls(ctx, pg):
            ctx.assert_true(used == iterations, f"numel {numel}: {used} of {iterations} launches used NVLS")


@test
def test_comm_pool_broadcast(ctx: TestContext):
    """The process group's broadcast of comm buffers, from the first and the last rank: correct; across NVLink domains,
    the proxy that reads the tensor in each domain but the root's stores it to the other ranks of its domain with NVLS
    (where supported; the ranks with nothing to store do not count launches, so per domain, the most launches of one of
    its ranks)."""
    if ctx.world_size < 2:
        return
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    domains = nvlink_domains(pg, ctx)
    check_nvls = expect_nvls(ctx, pg, domains) and len(set(domains)) > 1
    for root in sorted({0, ctx.world_size - 1}):
        for numel in [1 << 14, (9 << 20) // 4]:
            t = moodist.comm_empty(numel, dtype=torch.float32, group=pg)
            launches0 = pg._get_option("nvls_launches")
            iterations = 4
            for it in range(iterations):
                t.fill_(root * 1000 + it if ctx.rank == root else -1)
                pg.broadcast(t, root).wait()
                ok = bool((t == root * 1000 + it).all())
                ctx.assert_true(ok, f"root {root} numel {numel} iteration {it}: wrong result")
            torch.cuda.synchronize()
            used = pg._get_option("nvls_launches") - launches0
            del t
            if check_nvls:
                rows = gather_ints(pg, ctx, [used])
                for d in sorted(set(domains) - {domains[root]}):
                    most = max(rows[r][0] for r in range(ctx.world_size) if domains[r] == d)
                    ctx.assert_true(
                        most == iterations,
                        f"root {root} numel {numel}: {most} of {iterations} launches used NVLS in the domain of rank {d}",
                    )


@test
def test_comm_pool_large(ctx: TestContext):
    """Buffers larger than a slab (256 MiB), in a new slab each."""
    if ctx.world_size < 2:
        return
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    numel = (300 << 20) // 4 // ctx.world_size
    op = allgather_op(pg, ctx, numel)
    inp = torch.empty(numel, device="cuda", dtype=torch.float32)
    offsets = []
    for it in range(2):
        out = moodist.comm_empty(numel * ctx.world_size, dtype=torch.float32, group=pg)
        offsets.append(moodist.comm_pool_offset(out))
        inp.fill_(ctx.rank * 1000 + it)
        op([inp], [out]).wait()
        ctx.assert_true(bool((out == expected_allgather(ctx, numel, it)).all()), f"iteration {it}")
        del out
    assert_same_on_all_ranks(pg, ctx, offsets, "offsets")


@test
def test_comm_pool_backend_allocate_tensor(ctx: TestContext):
    """The process group's allocate_tensor (FSDP2's hook) allocates from the pool."""
    torch.cuda.set_device(ctx.local_rank)
    pg = create_process_group(ctx)
    device = torch.device("cuda", ctx.local_rank)
    backend = pg._get_backend(device)
    ctx.assert_true(backend.supports_tensor_alloc(device), "supports_tensor_alloc")
    t = backend.allocate_tensor(1000, dtype=torch.bfloat16, device=device)
    ctx.assert_true(t.shape == (1000,) and t.dtype == torch.bfloat16 and t.is_cuda, f"{t.shape} {t.dtype}")
    off = moodist.comm_pool_offset(t)
    ctx.assert_true(off >= 0, f"offset {off}")
    assert_same_on_all_ranks(pg, ctx, [off], "offset")
