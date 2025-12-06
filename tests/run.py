#!/usr/bin/env python3
"""
Test runner for moodist tests.

Usage:
    # Run all tests with torchrun:
    torchrun --nproc-per-node 8 tests/run.py

    # Run specific test files:
    torchrun --nproc-per-node 8 tests/run.py test_store.py test_queue.py

    # Run single-process (for non-distributed tests like serialize):
    python tests/run.py test_serialize.py

    # With slurm:
    srun ... python tests/run.py
"""

import importlib.util
import os
import sys
from pathlib import Path

# Add tests directory to path so test files can import framework
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from framework import TestRunner, create_context_from_env, get_tests, clear_tests


def load_test_module(path: Path):
    """Load a test module and return it."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_test_files(tests_dir: Path) -> list[Path]:
    """Find all test_*.py files in the tests directory."""
    return sorted(tests_dir.glob("test_*.py"))


def main():
    ctx = create_context_from_env()

    if ctx.rank == 0:
        print(f"Running tests with {ctx.world_size} processes")
        print(f"Master: {ctx.master_addr}:{ctx.master_port}")
        print()

    # Determine which test files to run
    if len(sys.argv) > 1:
        # Specific files provided
        test_files = [tests_dir / arg for arg in sys.argv[1:]]
    else:
        # Discover all test files
        test_files = discover_test_files(tests_dir)

    if ctx.rank == 0 and not test_files:
        print("No test files found!")
        sys.exit(1)

    all_passed = True
    runner = TestRunner(ctx)

    for test_file in test_files:
        if not test_file.exists():
            if ctx.rank == 0:
                print(f"Test file not found: {test_file}")
            all_passed = False
            continue

        if ctx.rank == 0:
            print(f"=== {test_file.name} ===")

        # Clear any previously registered tests
        clear_tests()

        # Load the test module (this registers tests via @test decorator)
        try:
            load_test_module(test_file)
        except Exception as e:
            if ctx.rank == 0:
                print(f"  Failed to load: {e}")
            all_passed = False
            continue

        # Run all tests from this module
        tests = get_tests()
        if not tests:
            if ctx.rank == 0:
                print("  (no tests found)")
            continue

        runner.run_all(tests)

    # Final summary
    if not runner.summarize():
        all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
