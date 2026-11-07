"""
Tests for ranks whose tensors are not where an op's NVLS kernel needs them. A launch of an op that has an NVLS kernel
runs it when the tensors that NVLS uses are in the communication pool (moodist.comm_empty, the process group's
allocate_tensor), else the op's unicast kernel (see launchCustomOp in processgroup.cc); the NVLS kernel checks that the
tensors are at the same place of the pool on every rank of the NVLink domain, and the two kernels recognize each other's
launches. Ranks that disagree (a tensor in the pool on some ranks only, or at different offsets of it) get an error: the
kernel reports it in host memory, marks its launch as failed for its peers (which then fail too, also those that check
nothing themselves, e.g. the ranks of a reduce to another rank) and traps (the CUDA context fails), the CPU thread logs
the report, the process group option "kernel_error" returns its code, and the next launch raises it.

Every case runs in a child process per rank (this file run as a script, with a process group of its own), which ends
with a failed context; the parent checks what the child reports. With MOODIST_TEST_EXPECT_NVLS=1 and NVLink domains of
16 ranks or more (copies) or 12 (reductions across domains; 8 for all-reduces in one domain), the mismatched cases must
fail on every rank of every domain (the last rank of each domain has the mismatched tensor); otherwise (no NVLS kernel)
they must run without error, with the right results. One more case runs without error either way: a buffer of the pool
at the address of a freed tensor outside the pool, which an op launched with before (its NVLS words, cached by address,
must not stay those of the freed tensor).
"""

import os
import subprocess
import sys
from datetime import timedelta

import torch
import moodist
from moodist import TensorRegion

if __name__ != "__main__":
    from framework import TestContext, test, create_process_group

EXPECT_NVLS = os.environ.get("MOODIST_TEST_EXPECT_NVLS") == "1"

# (case, what the NVLS kernel launches report: 1, a tensor at another offset of the pool; 2, a peer launched the other
# kernel; 0, no error). Any rank may report 3 instead (a peer failed first). The reduce cases reduce to rank 0, which
# alone checks the inputs (the others' kernels only wait for it).
CASES = [
    ("allgather_outside", 2),
    ("allgather_offset", 1),
    ("allreduce_input_outside", 2),
    ("allreduce_input_offset", 1),
    ("reducescatter_input_outside", 2),
    ("reduce_input_outside", 2),
    ("reduce_input_offset", 1),
    ("allreduce_cached_address", 0),
]


def values(r, count, dtype):
    # small integers: exact sums
    return ((torch.arange(count, device="cuda") * 3 + r * 7) % 11 - 5).to(dtype)


def run_case(pg, case, rank, world, mismatched):
    """Runs the case's op; returns (output, expected, buffers to keep until the op is done, a note) on this rank."""
    numel = 1 << 14
    dtype = torch.float32
    # (the mismatched rank's extra buffer of the pool, allocated first: the op's buffers at other offsets of the pool)
    keep = []
    if case.startswith("allgather"):
        op = moodist.compile_op(
            pg,
            dtype=dtype,
            inputs=[TensorRegion(offset=[rank * numel], shape=[numel], device="cuda")],
            outputs=[TensorRegion(offset=[0], shape=[numel * world], device="cuda")],
        )
        if mismatched and case.endswith("offset"):
            keep.append(moodist.comm_empty(1 << 20, dtype=torch.uint8, group=pg))
        out = moodist.comm_empty(numel * world, dtype=dtype, group=pg)
        if mismatched and case.endswith("outside"):
            out = torch.empty(numel * world, dtype=dtype, device="cuda")
        inp = values(rank, numel, dtype)
        op([inp], [out]).wait()
        expected = torch.cat([values(r, numel, dtype) for r in range(world)])
        return out, expected, keep, ""
    if case == "allreduce_cached_address":
        # An all-reduce in place, first on a tensor outside the pool (a view of a large one, at its first 2 MiB
        # boundary), then on a buffer of the pool that a new slab puts at the same address (the large tensor freed: the
        # pool takes its memory for the slab, which starts at the first boundary of the slab alignment), the same on
        # every rank: the second launch uses NVLS where the group has it
        numel = (32 << 20) // 4
        region = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
        op = moodist.compile_op(pg, dtype=dtype, inputs=region, outputs=region, reduce="sum")
        big = torch.empty((512 << 20) // 4, dtype=dtype, device="cuda")
        skip = (-big.data_ptr()) % (2 << 20) // big.element_size()
        x = big[skip:skip + numel]
        address = x.data_ptr()
        x.copy_(values(rank, numel, dtype))
        op([x], [x]).wait()
        torch.cuda.synchronize()
        del x, big
        comm = moodist.comm_empty(numel, dtype=dtype, group=pg)
        same = comm.data_ptr() == address
        comm.copy_(values(rank, numel, dtype))
        launches0 = pg._get_option("nvls_launches")
        op([comm], [comm]).wait()
        torch.cuda.synchronize()
        used = pg._get_option("nvls_launches") - launches0
        expected = sum(values(r, numel, dtype) for r in range(world))
        return comm, expected, keep, f"SAME_ADDRESS: {int(same)} NVLS_USED: {used}"
    rs = case.startswith("reducescatter")
    to_one = case.startswith("reduce_")
    in_numel = numel * world if rs else numel
    if rs:
        outputs = [TensorRegion(offset=[rank * numel], shape=[numel], device="cuda")]
    elif to_one:
        outputs = [TensorRegion(offset=[0], shape=[numel], device="cuda")] if rank == 0 else []
    else:
        outputs = [TensorRegion(offset=[0], shape=[numel], device="cuda")]
    op = moodist.compile_op(pg, dtype=dtype, inputs=[TensorRegion(offset=[0], shape=[in_numel], device="cuda")],
                            outputs=outputs, reduce="sum")
    if mismatched and case.endswith("offset"):
        keep.append(moodist.comm_empty(1 << 20, dtype=torch.uint8, group=pg))
    x = moodist.comm_empty(in_numel, dtype=dtype, group=pg)
    if mismatched and case.endswith("outside"):
        x = torch.empty(in_numel, dtype=dtype, device="cuda")
    out = moodist.comm_empty(numel, dtype=dtype, group=pg)
    x.copy_(values(rank, in_numel, dtype))
    op([x], [out] if outputs else []).wait()
    everyone = sum(values(r, in_numel, dtype) for r in range(world))
    if to_one and rank != 0:
        # (no output here)
        return out, out, keep, ""
    expected = everyone[rank * numel:(rank + 1) * numel] if rs else everyone
    return out, expected, keep, ""


def child_main(argv):
    """The child of a case: argv = case, master address, port, rank, world size, local rank, mismatched (0/1). Prints
    RESULT: ok, RESULT: wrong, or RESULT: error <message>, then KERNEL_ERROR: <code>, and NOTE: (case-specific), and
    exits at once (the context may have failed)."""
    case, addr, port, rank, world, local_rank, mismatched = argv
    port, rank, world, local_rank, mismatched = int(port), int(rank), int(world), int(local_rank), mismatched == "1"
    torch.cuda.set_device(local_rank)
    moodist.enable_cuda_allocator()
    store = moodist.TcpStore(addr, port, "nvls_errors_child", world, rank, timedelta(seconds=180))
    pg = moodist.MoodistProcessGroup(store, rank, world)
    note = ""
    try:
        out, expected, keep, note = run_case(pg, case, rank, world, mismatched)
        torch.cuda.synchronize()
        del keep
        result = "ok" if torch.equal(out, expected) else "wrong"
    except Exception as e:  # noqa: BLE001 (a failed context: CUDA's error)
        result = f"error {str(e).splitlines()[0] if str(e) else type(e).__name__}"
    print(f"RESULT: {result}", flush=True)
    print(f"KERNEL_ERROR: {pg._get_option('kernel_error')}", flush=True)
    print(f"NOTE: {note}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_child(ctx, case, index, mismatched):
    """Runs the case's child on this rank (every rank does, at once); returns (result, kernel error code, note,
    output)."""
    port = ctx.master_port + 5000 + 20 * ctx._current_test_id + index
    args = [sys.executable, os.path.abspath(__file__), "--child", case, ctx.master_addr, str(port), str(ctx.rank),
            str(ctx.world_size), str(ctx.local_rank), "1" if mismatched else "0"]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=300)
        output = p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
        return "timeout", 0, "", output
    lines = output.splitlines()
    result = next((line[len("RESULT: "):] for line in lines if line.startswith("RESULT: ")), "none")
    code = next((int(line[len("KERNEL_ERROR: "):]) for line in lines if line.startswith("KERNEL_ERROR: ")), -1)
    note = next((line[len("NOTE: "):] for line in lines if line.startswith("NOTE: ")), "")
    return result, code, note, output


if __name__ != "__main__":

    def nvlink_domains(pg, ctx):
        t = torch.tensor([pg._get_option("nvlink_domain")], dtype=torch.int64, device="cuda")
        out = torch.empty(ctx.world_size, dtype=torch.int64, device="cuda")
        pg._allgather_base(out, t).wait()
        return out.tolist()

    @test
    def test_nvls_errors(ctx: TestContext):
        """Every case (see CASES), each in children: with an NVLS kernel, every rank's launch reports the error (the
        last rank of each NVLink domain has its tensor outside the pool, or at another offset), with its code or 3 (a
        peer failed first); without, the right results. The cached address case: the right results, and with NVLS,
        its second launch used it (if the buffer of the pool did land at the freed tensor's address)."""
        if ctx.world_size < 2:
            return
        torch.cuda.set_device(ctx.local_rank)
        pg = create_process_group(ctx)
        domains = nvlink_domains(pg, ctx)
        last_of_domain = max(r for r in range(ctx.world_size) if domains[r] == domains[ctx.rank]) == ctx.rank
        sizes = [domains.count(d) for d in set(domains)]
        one_domain = len(sizes) == 1
        errors = []
        for index, (case, code) in enumerate(CASES):
            # Whether the case's op can have an NVLS kernel: copies in domains of 16 ranks or more (all of the same
            # size), all-reduces in one domain from 8 ranks, other reductions from 12
            if case.startswith("allgather"):
                possible = min(sizes) >= 16 and len(set(sizes)) == 1
            else:
                possible = min(sizes) >= (8 if one_domain and case.startswith("allreduce") else 12)
            # (the mismatched rank of the reduce cases is never the root, rank 0)
            result, reported, note, output = run_child(ctx, case, index, last_of_domain and ctx.rank != 0)
            tail = output[-1500:]
            succeeded = result == "ok" and reported == 0
            if code == 0:
                if ctx.rank == 0:
                    ctx.log(f"{case}: {note}")
                if not succeeded:
                    errors.append(f"{case}: expected success, got result {result!r}, kernel error {reported}; child "
                                  f"output tail: {tail!r}")
                elif possible and EXPECT_NVLS and note.startswith("SAME_ADDRESS: 1") and "NVLS_USED: 1" not in note:
                    errors.append(f"{case}: the launch on a buffer of the pool at a cached address did not use NVLS "
                                  f"({note})")
                continue
            failed = reported in (code, 3) and result.startswith("error")
            logged = "moodist: rank" in output
            if possible and EXPECT_NVLS:
                if not failed or not logged:
                    errors.append(f"{case}: expected error {code} (or 3), got result {result!r}, kernel error "
                                  f"{reported}{'' if logged else ' (no report logged)'}; child output tail: {tail!r}")
            elif not possible or EXPECT_NVLS:
                if not succeeded:
                    errors.append(f"{case}: expected success, got result {result!r}, kernel error {reported}; child "
                                  f"output tail: {tail!r}")
            elif not succeeded and not failed:
                # (NVLS not known to be there: either, but nothing else)
                errors.append(f"{case}: got result {result!r}, kernel error {reported}; child output tail: {tail!r}")
        ctx.assert_true(not errors, "; ".join(errors))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        child_main(sys.argv[2:])
