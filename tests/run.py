#!/usr/bin/env python3
"""
Test runner for moodist tests.

Usage:
    # Run all tests with torchrun:
    torchrun --nproc-per-node 8 tests/run.py

    # Run specific test files:
    torchrun --nproc-per-node 8 tests/run.py test_store.py test_queue.py

    # Run a specific test by name:
    torchrun --nproc-per-node 8 tests/run.py test_store.py::test_store_basic_set_get

    # Run single-process (for non-distributed tests like serialize):
    python tests/run.py test_serialize.py

    # With slurm:
    srun ... python tests/run.py

    # Output each rank to separate log files:
    torchrun --nproc-per-node 8 tests/run.py --per-rank-logs
    # Creates test_logs/rank_0.log, test_logs/rank_1.log, etc.
"""

import importlib.util
import os
import sys
from pathlib import Path

# Add tests directory to path so test files can import framework
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from framework import TestRunner, create_context_from_env, get_tests, clear_tests, clear_process_group_cache


def setup_per_rank_logging(rank: int, log_dir: str):
    """Redirect stdout/stderr to per-rank log files."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"rank_{rank}.log"

    # Open file and redirect at the OS level to capture C++ output too
    f = open(log_file, "w", buffering=1)  # Line buffered
    # Redirect OS-level file descriptors so C++ writes are captured
    os.dup2(f.fileno(), sys.stdout.fileno())
    os.dup2(f.fileno(), sys.stderr.fileno())
    sys.stdout = f
    sys.stderr = f
    return f


def load_test_module(path: Path):
    """Load a test module and return it."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so pickle/deserialize can find classes defined in test files
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def discover_test_files(tests_dir: Path) -> list[Path]:
    """Find all test_*.py files in the tests directory."""
    return sorted(tests_dir.glob("test_*.py"))


def parse_test_arg(arg: str) -> tuple[str, str | None]:
    """Parse a test argument like 'test_store.py::test_name' into (file, test_name)."""
    if "::" in arg:
        file_part, test_name = arg.split("::", 1)
        return file_part, test_name
    return arg, None


def main():
    ctx = create_context_from_env()

    # Parse --per-rank-logs option before anything else
    log_file = None
    args = sys.argv[1:]
    per_rank_logs = False
    remaining_args = []
    for arg in args:
        if arg == "--per-rank-logs":
            per_rank_logs = True
        else:
            remaining_args.append(arg)

    if per_rank_logs:
        log_file = setup_per_rank_logging(ctx.rank, "test_logs")
        ctx.per_rank_logs = True

    if ctx.rank == 0 or per_rank_logs:
        print(f"Running tests with {ctx.world_size} processes")
        print(f"Master: {ctx.master_addr}:{ctx.master_port}")
        if per_rank_logs:
            print(f"Per-rank logs: test_logs/rank_*.log")
        print()

    # Determine which test files to run
    test_filter = None  # Optional filter for specific test name
    if remaining_args:
        # Parse arguments - support file.py::test_name syntax
        test_files = []
        for arg in remaining_args:
            file_part, test_name = parse_test_arg(arg)
            test_files.append(tests_dir / file_part)
            if test_name:
                test_filter = test_name
    else:
        # Discover all test files
        test_files = discover_test_files(tests_dir)

    if (ctx.rank == 0 or per_rank_logs) and not test_files:
        print("No test files found!")
        sys.exit(1)

    all_passed = True
    runner = TestRunner(ctx)

    for test_file in test_files:
        if not test_file.exists():
            if ctx.rank == 0 or per_rank_logs:
                print(f"Test file not found: {test_file}")
            all_passed = False
            continue

        if ctx.rank == 0 or per_rank_logs:
            print(f"=== {test_file.name} ===")

        # Clear any previously registered tests and cached process groups
        clear_tests()
        clear_process_group_cache()

        # Load the test module (this registers tests via @test decorator)
        try:
            load_test_module(test_file)
        except Exception as e:
            if ctx.rank == 0 or per_rank_logs:
                print(f"  Failed to load: {e}")
            all_passed = False
            continue

        # Run all tests from this module
        tests = get_tests()
        if not tests:
            if ctx.rank == 0 or per_rank_logs:
                print("  (no tests found)")
            continue

        # Filter tests if a specific test name was requested
        if test_filter:
            tests = [(name, fn) for name, fn in tests if name == test_filter]
            if not tests:
                if ctx.rank == 0 or per_rank_logs:
                    print(f"  Test '{test_filter}' not found")
                all_passed = False
                continue

        runner.run_all(tests)

    # Final summary
    if not runner.summarize():
        all_passed = False

    if log_file:
        log_file.close()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
