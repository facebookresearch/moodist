"""
Benchmark: pure NVLink copy bandwidth (no local-to-local copy).

Like allgather, but each rank only receives the chunks from other GPUs,
skipping the self-copy. This isolates NVLink transfer performance.

Uses compile_op to express the pattern:
  - Each rank provides its chunk as input
  - Each rank's outputs cover only the non-self portions

    torchrun --nproc-per-node 8 benchmarks/nvlink_copy.py
    torchrun --nproc-per-node 2 benchmarks/nvlink_copy.py --sizes 1K,1M,64M
"""

import argparse
from datetime import timedelta
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


def bench_nvlink_copy(sizes: list[int], iterations: int, warmup: int,
                      profile: bool = False, world_size: int = None):
    rank = int(os.environ["RANK"])
    if world_size is not None:
        if rank >= world_size:
            return
    else:
        world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    import moodist
    from moodist import TensorRegion
    dist.init_process_group(backend="moodist", init_method="moodist://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]), rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
    #dist.init_process_group(backend="moodist", init_method="tcp://%s:%s" % (os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]), rank=rank, world_size=world_size, timeout=timedelta(seconds=20))
    moodist.enable_cpu_allocator()
    moodist.enable_cuda_allocator()

    if rank == 0:
        print(f"NVLink copy benchmark (no local copy): {world_size} GPUs, "
              f"{iterations} iterations, {warmup} warmup")
        print()
        print(f"{'size':>8}  {'med(us)':>8} {'p10(us)':>8} {'p90(us)':>8} {'algbw':>8} {'busbw':>8}")
        print("-" * 58)

    # Pre-compile for all sizes
    compiled_ops = {}
    for size in sizes:
        numel = size // 4

        # Input: this rank's chunk in the global coordinate space
        inputs = [TensorRegion(offset=[rank * numel], shape=[numel], device="cuda")]

        # Outputs: everything EXCEPT this rank's chunk
        outputs = []
        if rank > 0:
            outputs.append(TensorRegion(
                offset=[0], shape=[rank * numel], device="cuda"))
        if rank < world_size - 1:
            outputs.append(TensorRegion(
                offset=[(rank + 1) * numel],
                shape=[(world_size - rank - 1) * numel],
                device="cuda"))

        compiled_ops[size] = moodist.compile_op(
            dist.group.WORLD, dtype=torch.float32,
            inputs=inputs, outputs=outputs)

    for size in sizes:
        numel = size // 4
        input_tensor = torch.full((numel,), rank + 1.0,
                                  device="cuda", dtype=torch.float32)

        # Build output tensors matching the output regions
        output_tensors = []
        if rank > 0:
            output_tensors.append(
                torch.empty(rank * numel, device="cuda", dtype=torch.float32))
        if rank < world_size - 1:
            output_tensors.append(
                torch.empty((world_size - rank - 1) * numel,
                            device="cuda", dtype=torch.float32))

        op = compiled_ops[size]
        run = lambda: op([input_tensor], output_tensors).wait()

        input_tensor_0 = input_tensor.clone()

        # Warmup
        for iteration in range(warmup):
            for t in output_tensors:
                t.fill_(42)
            oi = input_tensor_0.clone()
            input_tensor = oi.clone()
            oi.fill_(500)
            if rank == 1:
                assert torch.all(input_tensor == input_tensor.clone())
            run()
            input_tensor.zero_()
            output_tensors = [o.clone() for o in output_tensors]

            # Correctness check
            # output_tensors[0] (if rank > 0): chunks from ranks 0..rank-1
            # output_tensors[1] (if rank < world_size-1): chunks from ranks rank+1..N-1
            oidx = 0
            if rank > 0:
                out = output_tensors[oidx]
                for r in range(rank):
                    chunk = out[r * numel:(r + 1) * numel]
                    expected = r + 1.0
                    if not torch.all(chunk == expected):
                        bad = (chunk != expected).sum().item()
                        print(f"RANK {rank}: iteration {iteration} CORRECTNESS FAILURE at size "
                            f"{format_size(size)}, chunk {r}: "
                            f"{bad}/{numel} elements wrong", file=sys.stderr)
                        sys.exit(1)
                oidx += 1
            if rank < world_size - 1:
                out = output_tensors[oidx]
                for r in range(rank + 1, world_size):
                    local_r = r - rank - 1
                    chunk = out[local_r * numel:(local_r + 1) * numel]
                    expected = r + 1.0
                    if not torch.all(chunk == expected):
                        bad = (chunk != expected).sum().item()
                        print(f"RANK {rank}: iteration {iteration} CORRECTNESS FAILURE at size "
                            f"{format_size(size)}, chunk {r}: "
                            f"{bad}/{numel} elements wrong", file=sys.stderr)
                        sys.exit(1)

        torch.cuda.synchronize()

        # Profiling
        if profile:
            trace_dir = "traces/nvlink_copy"
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
                f"{trace_dir}/nvlink_copy_{format_size(size)}_rank{rank}.json")
            if rank == 0:
                print(f"  trace: {trace_dir}/nvlink_copy_{format_size(size)}_rank{rank}.json")

        # Measured iterations
        start_events = [torch.cuda.Event(enable_timing=True)
                        for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True)
                      for _ in range(iterations)]

        for i in range(iterations):
            start_events[i].record()
            for _ in range(16):
                run()
            end_events[i].record()

        torch.cuda.synchronize()

        times_us = sorted(
            s.elapsed_time(e) * 1000 / 16
            for s, e in zip(start_events, end_events))
        n = len(times_us)
        median = times_us[n // 2]
        p10 = times_us[n // 10]
        p90 = times_us[n * 9 // 10]
        # Same bandwidth formula as allgather for direct comparison
        total_bytes = size * world_size
        algbw = (total_bytes / 1e9) / (median / 1e6) if median > 0 else 0
        busbw = algbw * (world_size - 1) / world_size

        if rank == 0:
            print(f"{format_size(size):>8}  {median:>8.1f} {p10:>8.1f} "
                  f"{p90:>8.1f} {algbw:>8.2f} {busbw:>8.2f}")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="NVLink copy benchmark (no local copy)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated sizes per rank, e.g. '1K,1M,64M'")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--profile", action="store_true",
                        help="Generate Chrome traces in traces/nvlink_copy/")
    parser.add_argument("--world_size", type=int, default=None)
    args = parser.parse_args()

    sizes = ([parse_size(s) for s in args.sizes.split(",")]
             if args.sizes else DEFAULT_SIZES)
    bench_nvlink_copy(sizes, args.iterations, args.warmup, args.profile, args.world_size)


if __name__ == "__main__":
    main()
