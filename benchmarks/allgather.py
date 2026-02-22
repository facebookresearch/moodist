"""
Benchmark: all-gather latency and bandwidth.

Run directly with torchrun, one backend at a time:

    torchrun --nproc-per-node 8 benchmarks/allgather.py --backend nccl
    torchrun --nproc-per-node 8 benchmarks/allgather.py --backend moodist
    torchrun --nproc-per-node 8 benchmarks/allgather.py --backend moodist_compile_op

    torchrun --nproc-per-node 2 benchmarks/allgather.py --backend nccl --sizes 1K,1M,64M
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


def parse_size(s: str) -> int:
    s = s.strip().upper()
    for suffix, mult in [("G", 1024**3), ("M", 1024**2), ("K", 1024)]:
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def format_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.0f}G"
    elif n >= 1024**2:
        return f"{n / 1024**2:.0f}M"
    elif n >= 1024:
        return f"{n / 1024:.0f}K"
    return str(n)


DEFAULT_SIZES = [1024, 4096, 16384, 65536, 262144, 1024**2, 4 * 1024**2,
                 16 * 1024**2, 64 * 1024**2, 256 * 1024**2]


def bench_allgather(backend: str, sizes: list[int], iterations: int, warmup: int,
                    profile: bool = False):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if backend in ("moodist", "moodist_compile_op"):
        import moodist
        dist.init_process_group(backend="moodist")
        moodist.enable_cuda_allocator()
    else:
        dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"All-gather benchmark: {backend}, {world_size} GPUs, "
              f"{iterations} iterations, {warmup} warmup")
        print()
        print(f"{'size':>8}  {'med(us)':>8} {'p10(us)':>8} {'p90(us)':>8} {'algbw':>8} {'busbw':>8}")
        print("-" * 50)

    # Pre-compile compile_op for all sizes (compile once, run many)
    compiled_ops = {}
    if backend == "moodist_compile_op":
        from moodist import TensorRegion
        for size in sizes:
            numel = size // 4
            inputs = [TensorRegion(offset=[rank * numel], shape=[numel], device="cuda")]
            outputs = [TensorRegion(offset=[0], shape=[numel * world_size], device="cuda")]
            compiled_ops[size] = moodist.compile_op(
                dist.group.WORLD, dtype=torch.float32, inputs=inputs, outputs=outputs)

    for size in sizes:
        numel = size // 4
        input_tensor = torch.full((numel,), rank + 1.0, device="cuda", dtype=torch.float32)
        output_tensor = torch.empty(numel * world_size, device="cuda", dtype=torch.float32)

        if backend == "moodist_compile_op":
            op = compiled_ops[size]
            run = lambda: op([input_tensor], [output_tensor]).wait()
        else:
            run = lambda: dist.all_gather_into_tensor(output_tensor, input_tensor)

        # Warmup
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        # Correctness check
        for r in range(world_size):
            chunk = output_tensor[r * numel:(r + 1) * numel]
            expected = r + 1.0
            if not torch.all(chunk == expected):
                bad = (chunk != expected).sum().item()
                print(f"RANK {rank}: CORRECTNESS FAILURE at size {format_size(size)}, "
                      f"chunk {r}: {bad}/{numel} elements wrong", file=sys.stderr)
                sys.exit(1)

        # Profiling
        if profile:
            trace_dir = f"traces/{backend}"
            os.makedirs(trace_dir, exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                for _ in range(10):
                    run()
            torch.cuda.synchronize()
            prof.export_chrome_trace(
                f"{trace_dir}/allgather_{format_size(size)}_rank{rank}.json")
            if rank == 0:
                print(f"  trace: {trace_dir}/allgather_{format_size(size)}_rank{rank}.json")

        # Measured iterations
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

        for i in range(iterations):
            start_events[i].record()
            run()
            end_events[i].record()

        torch.cuda.synchronize()

        times_us = sorted(s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events))
        n = len(times_us)
        median = times_us[n // 2]
        p10 = times_us[n // 10]
        p90 = times_us[n * 9 // 10]
        total_bytes = size * world_size
        algbw = (total_bytes / 1e9) / (median / 1e6) if median > 0 else 0
        busbw = algbw * (world_size - 1) / world_size

        if rank == 0:
            print(f"{format_size(size):>8}  {median:>8.1f} {p10:>8.1f} {p90:>8.1f} {algbw:>8.2f} {busbw:>8.2f}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="All-gather benchmark")
    parser.add_argument("--backend", required=True,
                        choices=["nccl", "moodist", "moodist_compile_op"])
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated sizes per rank, e.g. '1K,1M,64M'")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--profile", action="store_true",
                        help="Generate Chrome traces in traces/<backend>/")
    args = parser.parse_args()

    sizes = [parse_size(s) for s in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES
    bench_allgather(args.backend, sizes, args.iterations, args.warmup, args.profile)


if __name__ == "__main__":
    main()
