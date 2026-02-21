#!/usr/bin/env python3
"""Sweep MOODIST_COPY_KERNEL values and compare latencies.

Usage:
    python benchmarks/copy_kernel_sweep.py [nproc] [--sizes SIZES] [--iterations N]

Examples:
    python benchmarks/copy_kernel_sweep.py 2
    python benchmarks/copy_kernel_sweep.py 8 --sizes 64,1K,64K,1M,64M
    python benchmarks/copy_kernel_sweep.py 2 --iterations 500
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Sweep copy kernel versions")
    parser.add_argument("nproc", type=int, nargs="?", default=2,
                        help="Number of GPUs (default: 2)")
    parser.add_argument("--sizes", type=str,
                        default="64,256,1K,4K,16K,64K,256K,1M,4M,16M,64M",
                        help="Comma-separated sizes per rank")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--versions", type=str, default="v0,v1,v2",
                        help="Comma-separated kernel versions to test")
    args = parser.parse_args()

    versions = args.versions.split(",")

    for ver in versions:
        env = {**os.environ, "MOODIST_COPY_KERNEL": ver}
        cmd = [
            "torchrun",
            f"--nproc-per-node={args.nproc}",
            "benchmarks/allgather.py",
            "--backend", "moodist_compile_op",
            "--sizes", args.sizes,
            "--iterations", str(args.iterations),
            "--warmup", str(args.warmup),
        ]
        print(f"\n{'=' * 60}")
        print(f"  MOODIST_COPY_KERNEL={ver}")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  [FAILED with exit code {result.returncode}]")
        print()


if __name__ == "__main__":
    main()
