"""
Tests for moodist's CUDA allocator as PyTorch's allocator (moodist.enable_cuda_allocator): reuse of freed memory on its
own stream and on others (with the waits that needs), statistics, out-of-memory errors, PyTorch memory pools and CUDA
graph capture, and the mapped-memory limit with pressure mode (last: it leaves the allocator in pressure mode).

Set MOODIST_DEBUG_ALLOCATOR=1 to also check the allocator's data structures after every call.
"""

import ctypes
import gc
import time

import torch
import torch.distributed as dist

import moodist
from framework import TestContext, test

# A GPU busy-wait of a few tenths of a second: work still running when the host frees memory
SLEEP_CYCLES = 500_000_000

# PyTorch 2.13+ marks its captures (markCaptureBegin/End): routing to a pool without a capture does not keep the
# allocator from syncing the device then
CAPTURE_MARKS = tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 13)


def setup(ctx: TestContext):
    torch.cuda.set_device(ctx.local_rank)
    moodist.enable_cuda_allocator()
    torch.cuda.synchronize()


def nbytes(t):
    return t.numel() * t.element_size()


def overlaps(t, ptr, size):
    return t.data_ptr() < ptr + size and ptr < t.data_ptr() + nbytes(t)


def allocate_until_overlap(ptr, size, make, limit=200):
    """Allocates with make() until an allocation overlaps [ptr, ptr + size) (best fit takes other free regions of the
    stream first); returns it, and keeps the others alive in the returned list"""
    keep = []
    for _ in range(limit):
        t = make()
        if overlaps(t, ptr, size):
            return t, keep
        keep.append(t)
    raise AssertionError("no allocation got the freed memory back")


@test
def test_cuda_allocator_stats(ctx: TestContext):
    """torch.cuda.memory_allocated / max_memory_allocated / memory_reserved / memory_stats report the allocator's."""
    setup(ctx)
    a0 = torch.cuda.memory_allocated()
    t = torch.empty(10 << 20, dtype=torch.uint8, device="cuda")
    ctx.assert_true(torch.cuda.memory_allocated() - a0 >= 10 << 20, f"{torch.cuda.memory_allocated()} {a0}")
    ctx.assert_true(torch.cuda.memory_reserved() >= torch.cuda.memory_allocated())
    ctx.assert_true(torch.cuda.max_memory_allocated() >= torch.cuda.memory_allocated())
    stats = torch.cuda.memory_stats()
    ctx.assert_true(stats["num_device_alloc"] >= 1, str(stats))
    ctx.assert_true(stats["allocation.all.current"] >= 1, str(stats))
    ctx.assert_true(stats["requested_bytes.all.current"] >= 10 << 20, str(stats))
    del t
    ctx.assert_equal(torch.cuda.memory_allocated(), a0)
    ctx.assert_true(torch.cuda.max_memory_allocated() >= a0 + (10 << 20))
    torch.cuda.reset_peak_memory_stats()
    ctx.assert_equal(torch.cuda.max_memory_allocated(), a0)


@test
def test_cuda_allocator_oom_error(ctx: TestContext):
    """An allocation that cannot be satisfied raises torch.OutOfMemoryError, and the allocator works on."""
    setup(ctx)
    ooms = torch.cuda.memory_stats()["num_ooms"]
    try:
        torch.empty(1 << 44, dtype=torch.uint8, device="cuda")
    except torch.OutOfMemoryError as e:
        ctx.assert_true("out of memory" in str(e), str(e))
    else:
        raise AssertionError("expected torch.OutOfMemoryError")
    ctx.assert_equal(torch.cuda.memory_stats()["num_ooms"], ooms + 1)
    t = torch.ones(1000, device="cuda")
    ctx.assert_equal(t.sum().item(), 1000.0)


@test
def test_cuda_allocator_stream_reuse(ctx: TestContext):
    """Freed memory is reused by allocations on the stream it was allocated on; while memory can be mapped, allocations
    on other streams get other memory (no waits)."""
    setup(ctx)
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    n = 3 << 20 | 4096
    with torch.cuda.stream(s1):
        a = torch.empty(n, dtype=torch.uint8, device="cuda")
        pa, size = a.data_ptr(), nbytes(a)
        del a
    with torch.cuda.stream(s2):
        others = [torch.empty(n, dtype=torch.uint8, device="cuda") for _ in range(4)]
        for t in others:
            ctx.assert_false(overlaps(t, pa, size), "an allocation on another stream got s1's freed memory")
    with torch.cuda.stream(s1):
        b, keep = allocate_until_overlap(pa, size, lambda: torch.empty(n, dtype=torch.uint8, device="cuda"))
    del b, keep, others
    torch.cuda.synchronize()


@test
def test_cuda_allocator_record_stream_wait(ctx: TestContext):
    """Memory that another stream used (record_stream) is reused on its own stream only after that stream's work."""
    setup(ctx)
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    n = (1 << 20) + 1024
    t = torch.zeros(n, device="cuda")
    p, size = t.data_ptr(), nbytes(t)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        t.fill_(1.0)
    t.record_stream(side)
    del t
    u, keep = allocate_until_overlap(p, size, lambda: torch.empty(n, device="cuda"))
    u.fill_(2.0)
    torch.cuda.synchronize()
    ctx.assert_true(bool((u == 2.0).all()), "the allocation did not wait for the stream that used the memory")
    del u, keep


@test
def test_cuda_allocator_destroyed_stream_handle(ctx: TestContext):
    """A stream that frees memory and is destroyed while its work still runs: the driver gives its handle to the next
    stream created, which must not take that memory without waiting for that work."""
    setup(ctx)
    cuda = ctypes.CDLL("libcuda.so.1")

    def create():
        s = ctypes.c_void_p()
        ctx.assert_equal(cuda.cuStreamCreate(ctypes.byref(s), 1), 0)
        return s.value

    def destroy(s):
        ctx.assert_equal(cuda.cuStreamDestroy_v2(ctypes.c_void_p(s)), 0)

    n = (4 << 20) + 256
    for _ in range(3):
        s1 = create()
        with torch.cuda.stream(torch.cuda.ExternalStream(s1)):
            t = torch.zeros(n, device="cuda")
            torch.cuda._sleep(SLEEP_CYCLES)
            t.fill_(1.0)
        del t
        destroy(s1)
        s2 = create()
        with torch.cuda.stream(torch.cuda.ExternalStream(s2)):
            u = torch.empty(n, device="cuda")
            u.fill_(2.0)
        torch.cuda.synchronize()
        ctx.assert_true(bool((u == 2.0).all()), "a new stream took a destroyed stream's memory without waiting")
        del u
        destroy(s2)


@test
def test_cuda_allocator_raw_alloc(ctx: TestContext):
    """torch.cuda.caching_allocator_alloc (raw_alloc_with_stream) and caching_allocator_delete."""
    setup(ctx)
    s = torch.cuda.Stream()
    a0 = torch.cuda.memory_allocated()
    ptr = torch.cuda.caching_allocator_alloc(1 << 20, stream=s)
    ctx.assert_true(ptr != 0)
    ctx.assert_true(torch.cuda.memory_allocated() >= a0 + (1 << 20))
    torch.cuda.caching_allocator_delete(ptr)
    ctx.assert_equal(torch.cuda.memory_allocated(), a0)


def graph_case(ctx, mode):
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    g = torch.cuda.CUDAGraph()
    graph_memory = []
    with torch.cuda.graph(g, capture_error_mode=mode):
        tmp = x * 2
        graph_memory.append((tmp.data_ptr(), nbytes(tmp)))
        y = tmp + 1
        del tmp
        tmp2 = y * 3  # (may reuse tmp's memory, within the capture)
        graph_memory.append((tmp2.data_ptr(), nbytes(tmp2)))
        z = tmp2 - y
        del tmp2
    eager = [torch.full((n,), 7.0, device="cuda") for _ in range(8)]
    for e in eager:
        for p, size in graph_memory:
            ctx.assert_false(overlaps(e, p, size), "an eager allocation got memory the graph uses")
    for _ in range(3):
        x.normal_()
        g.replay()
        torch.cuda.synchronize()
        ctx.assert_true(torch.allclose(z, (x * 2 + 1) * 2), f"capture mode {mode}: wrong result")
    for e in eager:
        ctx.assert_true(bool((e == 7.0).all()), f"capture mode {mode}: a replay wrote an eager tensor")
    del g, y, z, eager
    torch.cuda.synchronize()


@test
def test_cuda_graph_private_pool(ctx: TestContext):
    """Memory allocated during a capture belongs to the graph (its private pool): eager allocations do not get it while
    the graph lives, and replays compute correct results."""
    setup(ctx)
    for mode in ("global", "thread_local", "relaxed"):
        graph_case(ctx, mode)


@test
def test_cuda_graph_multi_stream(ctx: TestContext):
    """A capture that forks onto another stream: memory freed on one capture stream and reused on another waits for
    it within the graph (an edge), and replays are correct."""
    setup(ctx)
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    side = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            a = x * 2
        main.wait_stream(side)
        b = a + 1
        a.record_stream(main)
        del a
        with torch.cuda.stream(side):
            c = torch.empty(n, device="cuda")  # (may reuse a's memory: after b, via an edge)
            c.copy_(x)
            c.mul_(3)
        main.wait_stream(side)
        d = b + c
    for _ in range(3):
        x.normal_()
        g.replay()
        torch.cuda.synchronize()
        ctx.assert_true(torch.allclose(d, x * 2 + 1 + x * 3), "wrong result")
    del g, b, c, d
    torch.cuda.synchronize()


@test
def test_cuda_graph_pool_release(ctx: TestContext):
    """A graph's private pool returns its memory to the main pool once the graph is destroyed (at the next capture,
    whose torch.cuda.graph calls empty_cache): capturing and destroying graphs over and over does not map more and more
    memory."""
    setup(ctx)
    n = 32 << 20
    x = torch.ones(n, dtype=torch.uint8, device="cuda")
    reserved = []
    for i in range(12):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            big = x * 2
            out = big.sum()
        g.replay()
        g.replay()
        ctx.assert_equal(out.item(), 2 * n)
        del g, big, out
        gc.collect()
        reserved.append(torch.cuda.memory_reserved())
    ctx.assert_true(reserved[-1] == reserved[1], f"memory reserved keeps growing: {reserved}")
    del x


@test
def test_cuda_graph_readonly_graph(ctx: TestContext):
    """A graph that only uses memory of its (shared) pool, allocating nothing, may still be running when its pool is
    released: the pool's memory goes back to the main pool only after that replay."""
    setup(ctx)
    n = 4 << 20
    x = torch.ones(n, device="cuda")
    pool = torch.cuda.graph_pool_handle()
    g1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool):
        y = x * 2
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, pool=pool):
        torch.cuda._sleep(SLEEP_CYCLES)
        y.add_(1)  # in place: g2 allocates and frees nothing
    g1.replay()
    torch.cuda.synchronize()
    capture_stream = torch.cuda.graph.default_capture_stream
    py, size = y.data_ptr(), nbytes(y)
    g2.replay()  # on the current stream: sleeps, then writes y
    del y, g1
    gc.collect()
    del g2  # the pool's last use: released while g2's replay runs
    gc.collect()
    torch.cuda.empty_cache()  # (returns the pool's memory, after g2's replay)
    with torch.cuda.stream(capture_stream):
        z, keep = allocate_until_overlap(py, size, lambda: torch.full((n,), 7.0, device="cuda"))
    torch.cuda.synchronize()
    ctx.assert_true(bool((z == 7.0).all()), "a graph's replay wrote memory that its released pool gave back")
    del z, keep


def reused_on_new_stream(ptr, size, n, count=16):
    """Whether one of count allocations of n bytes on a new stream overlaps [ptr, ptr + size): with no freed memory of
    its own, the stream takes the best fits of the memory streams share (where released pools' memory goes)"""
    cuda = ctypes.CDLL("libcuda.so.1")
    s = ctypes.c_void_p()
    assert cuda.cuStreamCreate(ctypes.byref(s), 1) == 0
    found = False
    with torch.cuda.stream(torch.cuda.ExternalStream(s.value)):
        ts = []
        for _ in range(count):
            ts.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
            if overlaps(ts[-1], ptr, size):
                found = True
                break
        del ts  # (before the stream is destroyed)
    torch.cuda.synchronize()
    cuda.cuStreamDestroy_v2(s)
    return found


def destroyed_graph_memory(n):
    """A graph whose pool has free memory (a temporary of n bytes, and its output, freed), destroyed: the temporary's
    address and size"""
    x = torch.ones(n, dtype=torch.uint8, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        tmp = x + 1
        p, size = tmp.data_ptr(), nbytes(tmp)
        y = tmp * 2
        del tmp
    g.replay()
    torch.cuda.synchronize()
    del y, g
    gc.collect()
    return p, size


@test
def test_cuda_graph_release_at_empty_cache(ctx: TestContext):
    """A destroyed graph's memory goes back to the main pool at torch.cuda.empty_cache (after a device sync), as with
    PyTorch's allocator; not before."""
    setup(ctx)
    n = (9 << 20) + 4096
    p, size = destroyed_graph_memory(n)
    ctx.assert_false(reused_on_new_stream(p, size, n), "a released pool's memory was reused without a device sync")
    torch.cuda.empty_cache()
    ctx.assert_true(reused_on_new_stream(p, size, n), "empty_cache did not return a released pool's memory")


@test
def test_cuda_graph_destroy_without_sync(ctx: TestContext):
    """Destroying a graph does not wait for the device (as with PyTorch's allocator): with a long kernel running on
    another stream, it returns long before the kernel ends."""
    setup(ctx)
    x = torch.ones(1 << 20, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = x * 2
        z = y + 1
    del y  # (the pool has free memory: a sync could return it)
    g.replay()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    t0 = time.perf_counter()
    with torch.cuda.stream(side):
        torch.cuda._sleep(4 * SLEEP_CYCLES)
    del g, z
    gc.collect()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    ctx.assert_true(t1 - t0 < 0.5 * (t2 - t0), f"destroying a graph took {t1 - t0:.3f} s of the kernel's {t2 - t0:.3f}")
    del x


@test
def test_cuda_graph_release_in_mempool_region(ctx: TestContext):
    """torch.cuda.empty_cache inside a torch.cuda.use_mem_pool region: with PyTorch 2.13+ (which marks its captures) it
    returns a destroyed graph's memory; before, the region keeps the allocator from syncing (it could be a capture's)
    until it ends."""
    setup(ctx)
    if not hasattr(torch.cuda, "MemPool"):
        ctx.log("no torch.cuda.MemPool, skipping")
        return
    n = (9 << 20) + 4096
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool):
        p, size = destroyed_graph_memory(n)
        torch.cuda.empty_cache()
    if not CAPTURE_MARKS:
        ctx.assert_false(reused_on_new_stream(p, size, n), "a released pool's memory was reused without a device sync")
        torch.cuda.empty_cache()
    ctx.assert_true(reused_on_new_stream(p, size, n), "empty_cache did not return a released pool's memory")
    del pool


@test
def test_cuda_graph_shared_pool(ctx: TestContext):
    """Two graphs captured into one pool (torch.cuda.graph_pool_handle), replayed in capture order."""
    setup(ctx)
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    pool = torch.cuda.graph_pool_handle()
    g1 = torch.cuda.CUDAGraph()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g1, pool=pool):
        tmp = x + 1
        y = tmp * 2
        del tmp
    with torch.cuda.graph(g2, pool=pool):
        tmp = y + 1
        z = tmp * 3
        del tmp
    for _ in range(3):
        x.normal_()
        g1.replay()
        g2.replay()
        torch.cuda.synchronize()
        ctx.assert_true(torch.allclose(z, ((x + 1) * 2 + 1) * 3), "wrong result")
    del g1, g2, y, z
    torch.cuda.synchronize()


@test
def test_cuda_mempool_nested(ctx: TestContext):
    """Nested torch.cuda.use_mem_pool: allocations go to the innermost pool, as with PyTorch's allocator."""
    setup(ctx)
    if not hasattr(torch.cuda, "MemPool"):
        ctx.log("no torch.cuda.MemPool, skipping")
        return
    a = torch.cuda.MemPool()
    b = torch.cuda.MemPool()
    n = 3 << 20
    with torch.cuda.use_mem_pool(a):
        with torch.cuda.use_mem_pool(b):
            x = torch.empty(n, dtype=torch.uint8, device="cuda")
    p = x.data_ptr()
    del x  # (into the pool it came from)
    with torch.cuda.use_mem_pool(a):
        ya = torch.empty(n, dtype=torch.uint8, device="cuda")
    with torch.cuda.use_mem_pool(b):
        yb = torch.empty(n, dtype=torch.uint8, device="cuda")
    ctx.assert_equal(yb.data_ptr(), p, "the nested allocation did not come from the inner pool")
    ctx.assert_true(ya.data_ptr() != p)
    del ya, yb, a, b


@test
def test_cuda_mempool(ctx: TestContext):
    """torch.cuda.MemPool with torch.cuda.use_mem_pool: allocations in the context come from the pool, others from the
    main pool, and the pool's memory goes back to the main pool when the pool is gone."""
    setup(ctx)
    if not hasattr(torch.cuda, "MemPool"):
        ctx.log("no torch.cuda.MemPool, skipping")
        return
    pool = torch.cuda.MemPool()
    ctx.assert_equal(pool.use_count(), 1)
    n = 5 << 20
    with torch.cuda.use_mem_pool(pool):
        ctx.assert_equal(pool.use_count(), 2)
        a = torch.empty(n, dtype=torch.uint8, device="cuda")
        pa, size = a.data_ptr(), nbytes(a)
        del a
        b = torch.empty(n, dtype=torch.uint8, device="cuda")
        ctx.assert_equal(b.data_ptr(), pa, "the pool's freed memory was not reused in the pool")
    ctx.assert_equal(pool.use_count(), 1)
    c = torch.empty(n, dtype=torch.uint8, device="cuda")
    ctx.assert_false(overlaps(c, pa, size), "a main-pool allocation got the pool's memory")
    del b
    # the pool keeps its freed memory while it exists
    d = torch.empty(n, dtype=torch.uint8, device="cuda")
    ctx.assert_false(overlaps(d, pa, size), "a main-pool allocation got the pool's freed memory")
    del pool
    gc.collect()
    del c, d
    torch.cuda.synchronize()


@test
def test_cuda_graph_nccl(ctx: TestContext):
    """A capture with NCCL collectives (PyTorch's NCCL process group) and moodist's allocator."""
    setup(ctx)
    if not dist.is_nccl_available():
        ctx.log("NCCL not available, skipping")
        return
    store = ctx.create_store("nccl_graph")
    pg = dist.ProcessGroupNCCL(store, ctx.rank, ctx.world_size)
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    pg.allreduce([x.clone()]).wait()  # (communicator set up outside the capture)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        y = x * 2
        pg.allreduce([y]).wait()
        z = y + 1
    for _ in range(3):
        x.fill_(float(ctx.rank + 1))
        g.replay()
        torch.cuda.synchronize()
        expected = 2 * sum(range(1, ctx.world_size + 1)) + 1
        ctx.assert_true(bool((z == expected).all()), f"wrong result {z[:4]} expected {expected}")
    del g, y, z
    torch.cuda.synchronize()
    pg.shutdown() if hasattr(pg, "shutdown") else None


@test
def test_cuda_graph_unrouted_capture(ctx: TestContext):
    """A stream capture that PyTorch does not route to a pool (cuStreamBeginCapture directly; not supported), with a
    tensor freed inside it: memory pools and graph captures still work afterwards. (Nothing is allocated inside: an
    allocation on a capturing stream that PyTorch does not route invalidates the capture with some drivers.)"""
    setup(ctx)
    cuda = ctypes.CDLL("libcuda.so.1")
    s = torch.cuda.Stream()
    x = torch.ones(1 << 20, device="cuda")
    with torch.cuda.stream(s):
        t = x * 2
        y = torch.empty_like(t)
    torch.cuda.synchronize()
    handle = ctypes.c_void_p(s.cuda_stream)
    ctx.assert_equal(cuda.cuStreamBeginCapture_v2(handle, 2), 0)  # relaxed mode
    try:
        with torch.cuda.stream(s):
            torch.add(t, 1, out=y)
            del t  # (its event is recorded in the capture)
    finally:
        graph = ctypes.c_void_p()
        ended = cuda.cuStreamEndCapture(handle, ctypes.byref(graph))
        if graph.value:
            cuda.cuGraphDestroy(graph)
    ctx.assert_equal(ended, 0)
    torch.cuda.synchronize()
    del y
    if hasattr(torch.cuda, "MemPool"):
        pool = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(pool):
            z = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")  # (takes main-pool memory: queries events)
        del z, pool
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        w = x * 3
    g.replay()
    torch.cuda.synchronize()
    ctx.assert_true(bool((w == 3.0).all()))
    del g, w, x


@test
def test_cuda_graph_conditional_node(ctx: TestContext):
    """A conditional node (PyTorch 2.12+): its body is captured on a stream of its own, which PyTorch 2.13+ destroys
    right after the body's capture. Memory allocated in the body and freed later (during the capture, and after it)
    must not make the allocator call the driver with that stream's handle."""
    setup(ctx)
    if not hasattr(torch._C._CUDAGraph, "begin_capture_to_if_node"):
        ctx.log("no conditional nodes in this PyTorch, skipping")
        return
    n = 1 << 20
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    pred = torch.ones((), dtype=torch.bool, device="cuda")
    y = torch.zeros(n, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        g.begin_capture_to_if_node(pred)
        tmp = x * 2  # in the body, on its stream
        out1 = tmp + 1
        out2 = tmp + 2
        del tmp  # freed in the body
        y.copy_(out1)
        y.add_(out2)
        g.end_capture_to_conditional_node()
        del out1  # freed during the capture, after the body (whose stream may be gone)
        z = y * 3
    del out2  # freed after the capture
    gc.collect()
    for value in (True, False, True):
        pred.fill_(value)
        before = y.clone()
        x.add_(1)
        g.replay()
        torch.cuda.synchronize()
        expected = (x * 2 + 1) + (x * 2 + 2) if value else before
        ctx.assert_true(torch.equal(y, expected), f"wrong body result (predicate {value})")
        ctx.assert_true(torch.equal(z, y * 3), "wrong result after the conditional node")
    # streams created now may get the body stream's handle
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        u = torch.empty(n, device="cuda")
        u.fill_(1.0)
    torch.cuda.synchronize()
    del g, y, z, u


@test
def test_cuda_graph_release_under_limit(ctx: TestContext):
    """With mapping limited, a destroyed graph's memory (waiting for a device sync) goes back to the main pool before
    an allocation would fail."""
    setup(ctx)
    n = (64 << 20) + 4096
    p, size = destroyed_graph_memory(n)
    total = torch.cuda.mem_get_info()[1]
    reserved = torch.cuda.memory_reserved()
    got = []
    try:
        torch.cuda.set_per_process_memory_fraction(reserved / total)
        while True:
            try:
                t = torch.empty(n, dtype=torch.uint8, device="cuda")
            except torch.OutOfMemoryError:
                raise AssertionError("out of memory while a released pool's memory waited for a device sync")
            got.append(t)
            if overlaps(t, p, size):
                break
        ctx.assert_equal(torch.cuda.memory_reserved(), reserved, "memory mapped beyond the limit")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    del got


@test
def test_cuda_graph_invalidated_capture(ctx: TestContext):
    """A capture invalidated by an operation it forbids: allocations and frees until it ends work (frees wait for its
    end), and later captures work. (Late in the file, and no random numbers here or after: a failed capture leaves
    PyTorch's generator in capture state, and before PyTorch 2.13 its pool routing too, which keeps released pools
    from being synced and returned for good.)"""
    setup(ctx)
    n = 1 << 20
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            a = x * 2
            try:
                torch.cuda.synchronize()  # (forbidden in a capture: invalidates it)
            except Exception:
                pass
            b = torch.empty(n, device="cuda")
            del a, b
    except Exception:
        pass
    else:
        raise AssertionError("the capture did not fail")
    del g
    gc.collect()
    torch.cuda.synchronize()
    eager = torch.full((n,), 7.0, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        tmp = x * 2
        y = tmp + 1
        del tmp
    for i in range(3):
        x.add_(1)
        g.replay()
        torch.cuda.synchronize()
        ctx.assert_true(torch.equal(y, x * 2 + 1), "wrong result after a failed capture")
    ctx.assert_true(bool((eager == 7.0).all()))
    del g, y, eager
    torch.cuda.synchronize()


@test
def test_cuda_allocator_pressure_mode(ctx: TestContext):
    """With mapping limited (torch.cuda.set_per_process_memory_fraction), allocations take free memory of other streams
    (pressure mode): first memory that needs no wait, then memory whose last use is still running, after that use;
    once no free memory fits, torch.OutOfMemoryError. (Last: the allocator stays in pressure mode.)"""
    setup(ctx)
    total = torch.cuda.mem_get_info()[1]
    chunk = total // 64
    a = torch.cuda.Stream()
    b = torch.cuda.Stream()
    n = 8 << 20
    r0 = torch.cuda.memory_reserved()
    with torch.cuda.stream(a):
        blocks = []
        while torch.cuda.memory_reserved() - r0 < 2 * chunk and len(blocks) < 10000:
            blocks.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
    reserved = torch.cuda.memory_reserved()
    try:
        torch.cuda.set_per_process_memory_fraction(reserved / total)
        with torch.cuda.stream(a):
            torch.cuda._sleep(SLEEP_CYCLES)
            for t in blocks:
                t.fill_(1)
        spans = [(t.data_ptr(), nbytes(t)) for t in blocks]
        del blocks
        got = []
        with torch.cuda.stream(b):
            while True:
                try:
                    got.append(torch.empty(n, dtype=torch.uint8, device="cuda"))
                except torch.OutOfMemoryError:
                    break
            for t in got:
                t.fill_(2)
        torch.cuda.synchronize()
        for t in got:
            ctx.assert_true(bool((t == 2).all()), "an allocation did not wait for the stream that used its memory")
        # (all but fragments left at the end)
        reused = sum(any(overlaps(t, p, size) for t in got) for p, size in spans)
        ctx.assert_true(reused >= len(spans) - 2, f"only {reused} of {len(spans)} freed blocks of the other stream reused")
        ctx.assert_equal(torch.cuda.memory_reserved(), reserved, "memory mapped beyond the limit")
        count = len(got)
        del got
        # memory freed on several streams is reused across them
        for s in (a, b, a):
            with torch.cuda.stream(s):
                again = [torch.empty(n, dtype=torch.uint8, device="cuda") for _ in range(count)]
                for t in again:
                    t.fill_(3)
                del again
        torch.cuda.synchronize()
        ctx.assert_equal(torch.cuda.memory_reserved(), reserved, "memory mapped beyond the limit")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
