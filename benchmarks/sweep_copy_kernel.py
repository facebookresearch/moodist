"""
Sweep copy kernel configurations and print a comparison table.

Usage:
    python benchmarks/sweep_copy_kernel.py --gpus 2
    python benchmarks/sweep_copy_kernel.py --gpus 8
    python benchmarks/sweep_copy_kernel.py --gpus 8 --sizes 64M,256M
    python benchmarks/sweep_copy_kernel.py --gpus 8 --sizes 256M --iterations 100

Runs the allgather benchmark across multiple kernel configs and prints results.
"""

import argparse
import os
import re
import subprocess
import sys


def parse_size(s):
    s = s.strip().upper()
    for suffix, mult in [("G", 1024**3), ("M", 1024**2), ("K", 1024)]:
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def format_size(n):
    if n >= 1024**3:
        return f"{n / 1024**3:.0f}G"
    elif n >= 1024**2:
        return f"{n / 1024**2:.0f}M"
    elif n >= 1024:
        return f"{n / 1024:.0f}K"
    return str(n)


def run_benchmark(gpus, sizes_str, iterations, warmup, env_overrides):
    """Run allgather benchmark, return dict of {size_str: {med, busbw, ...}}."""
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [
        "torchrun", "--nproc-per-node", str(gpus),
        "benchmarks/allgather.py",
        "--backend", "moodist_compile_op",
        "--sizes", sizes_str,
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env)
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Show first meaningful error line
        for line in stderr.split("\n"):
            if "error" in line.lower() or "Error" in line or "FAIL" in line:
                print(f"  FAILED: {line.strip()}", file=sys.stderr)
                break
        else:
            print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        return None

    # Parse output: "   256M    123.4    100.2    150.3    12.34    11.23"
    results = {}
    for line in result.stdout.split("\n"):
        line = line.strip()
        if not line or line.startswith("All-gather") or line.startswith("size") or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                size_label = parts[0]
                med = float(parts[1])
                busbw = float(parts[5])
                results[size_label] = {"med_us": med, "busbw": busbw}
            except (ValueError, IndexError):
                continue

    return results if results else None


def main():
    parser = argparse.ArgumentParser(description="Sweep copy kernel configurations")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--sizes", type=str, default="1M,4M,16M,64M,256M")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--nccl", action="store_true", help="Include NCCL baseline")
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    sizes_str = ",".join(sizes)

    # Define configurations to sweep
    configs = []

    # v1 baseline
    configs.append(("v1", {"MOODIST_COPY_KERNEL": "v1"}))

    # v7 sweep: depth x block_size x load_first
    for depth in [1, 2, 3, 4, 8, 16, 32]:
        for bs in [256, 512, 768, 1024]:
            for lf in [False, True]:
                label = f"v7 d={depth} bs={bs}"
                if lf:
                    label += " lf"
                env = {
                    "MOODIST_COPY_KERNEL": "v7",
                    "MOODIST_COPY_DEPTH": str(depth),
                    "MOODIST_COPY_BLOCK_SIZE": str(bs),
                }
                if lf:
                    env["MOODIST_COPY_LOAD_FIRST"] = "1"
                configs.append((label, env))

    if args.nccl:
        configs.append(("nccl", {}))

    print(f"Sweeping {len(configs)} configs, {args.gpus} GPUs, sizes: {sizes_str}")
    print(f"iterations={args.iterations}, warmup={args.warmup}")
    print()

    # Collect all results
    all_results = {}
    for i, (label, env) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {label} ...", end="", flush=True)

        if label == "nccl":
            # Special case: run NCCL via the benchmark directly
            cmd_env = os.environ.copy()
            cmd = [
                "torchrun", "--nproc-per-node", str(args.gpus),
                "benchmarks/allgather.py",
                "--backend", "nccl",
                "--sizes", sizes_str,
                "--iterations", str(args.iterations),
                "--warmup", str(args.warmup),
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120, env=cmd_env)
                results = {}
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("All-gather") or line.startswith("size") or line.startswith("-"):
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            results[parts[0]] = {"med_us": float(parts[1]), "busbw": float(parts[5])}
                        except (ValueError, IndexError):
                            continue
                all_results[label] = results if results else None
            except Exception as e:
                all_results[label] = None
                print(f" ERROR: {e}")
                continue
        else:
            all_results[label] = run_benchmark(
                args.gpus, sizes_str, args.iterations, args.warmup, env)

        if all_results[label]:
            # Print quick summary
            busbws = [f"{s}={v['busbw']:.0f}" for s, v in all_results[label].items()]
            print(f" {', '.join(busbws)}")
        else:
            print(" FAILED")

    # Print result table
    print()
    print("=" * 80)
    print(f"Results: {args.gpus} GPUs, busbw (GB/s)")
    print("=" * 80)

    # Header
    col_w = max(len(s) for s in sizes) + 2
    label_w = max((len(l) for l in all_results), default=20)
    header = f"{'config':<{label_w}}"
    for s in sizes:
        header += f"  {s:>{col_w}}"
    print(header)
    print("-" * len(header))

    # Find best busbw per size for highlighting
    best = {}
    for s in sizes:
        best_val = 0
        for label, results in all_results.items():
            if results and s in results:
                best_val = max(best_val, results[s]["busbw"])
        best[s] = best_val

    # Rows
    for label, results in all_results.items():
        row = f"{label:<{label_w}}"
        if results is None:
            row += "  FAILED"
        else:
            for s in sizes:
                if s in results:
                    bw = results[s]["busbw"]
                    marker = " *" if bw >= best[s] - 0.01 else "  "
                    row += f"  {bw:>{col_w - 2}.1f}{marker}"
                else:
                    row += f"  {'---':>{col_w}}"
        print(row)

    print()
    print("* = best for that size")


if __name__ == "__main__":
    main()
