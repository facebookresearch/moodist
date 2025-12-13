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

    # Set timeout for entire test run (captures stack trace on timeout):
    torchrun --nproc-per-node 8 tests/run.py --timeout 60
    # If py-spy is installed, uses it for detailed stack traces including C++ frames
"""

import importlib.util
import os
import signal
import subprocess
import sys
import time
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


def get_rank_from_env() -> int:
    """Get rank from environment variables (torchrun or slurm style)."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    else:
        return 0


def capture_stack_trace(pid: int) -> str:
    """Capture stack trace of a process using py-spy and pstack."""
    output = []

    # py-spy for Python + native frames on Python threads
    try:
        result = subprocess.run(
            ["py-spy", "dump", "--native", "--pid", str(pid)],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            output.append("=== py-spy (Python threads) ===")
            output.append(result.stdout)
        else:
            output.append(f"py-spy failed (exit {result.returncode}): {result.stderr}")
    except FileNotFoundError:
        output.append("(py-spy not installed)")
    except subprocess.TimeoutExpired:
        output.append("(py-spy timed out)")
    except Exception as e:
        output.append(f"(py-spy failed: {e})")

    # pstack/gstack for all threads including C++ background threads
    for cmd in ["pstack", "gstack"]:
        try:
            result = subprocess.run(
                [cmd, str(pid)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                output.append(f"\n=== {cmd} (all threads) ===")
                output.append(result.stdout)
                break
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            output.append(f"({cmd} timed out)")
            break
        except Exception as e:
            output.append(f"({cmd} failed: {e})")
            break
    else:
        # Fallback to gdb if pstack/gstack not available
        try:
            result = subprocess.run(
                ["gdb", "-batch", "-ex", "thread apply all bt", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                output.append("\n=== gdb (all threads) ===")
                output.append(result.stdout)
        except FileNotFoundError:
            output.append("(gdb not installed)")
        except subprocess.TimeoutExpired:
            output.append("(gdb timed out)")
        except Exception as e:
            output.append(f"(gdb failed: {e})")

    return "\n".join(output) if output else "(no stack trace tools available)"


def run_with_timeout(timeout: float, remaining_args: list[str], per_rank_logs: bool) -> int:
    """Fork and run tests in child process with timeout monitoring from parent.

    Returns exit code (0 = success, 1 = failure, 2 = timeout).
    """
    # Fork before any CUDA initialization
    pid = os.fork()

    if pid == 0:
        # Child process: run tests normally
        # Re-exec without --timeout to avoid infinite recursion
        new_args = [sys.executable, sys.argv[0]] + remaining_args
        os.execv(sys.executable, new_args)
        # execv doesn't return on success
        sys.exit(1)

    # Parent process: monitor child with timeout
    start_time = time.monotonic()
    deadline = start_time + timeout

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Timeout! Set up logging to a separate timeout log file
            log_file = None
            if per_rank_logs:
                rank = get_rank_from_env()
                log_path = Path("test_logs")
                log_path.mkdir(parents=True, exist_ok=True)
                log_file_path = log_path / f"timeout_rank_{rank}.log"
                log_file = open(log_file_path, "w", buffering=1)
                os.dup2(log_file.fileno(), sys.stdout.fileno())
                os.dup2(log_file.fileno(), sys.stderr.fileno())
                sys.stdout = log_file
                sys.stderr = log_file

            # Capture stack trace and kill child
            print(f"\n\033[33m{'='*60}\033[0m")
            print(f"\033[33mTIMEOUT after {timeout}s - capturing stack trace...\033[0m")
            print(f"\033[33m{'='*60}\033[0m\n")
            sys.stdout.flush()

            stack_trace = capture_stack_trace(pid)
            print(stack_trace)
            sys.stdout.flush()

            print(f"\n\033[33mKilling child process {pid}...\033[0m")
            sys.stdout.flush()
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except OSError:
                pass

            if log_file:
                log_file.close()
            return 2  # Timeout exit code

        # Wait for child with a short timeout so we can check deadline
        try:
            wait_pid, status = os.waitpid(pid, os.WNOHANG)
            if wait_pid == pid:
                # Child exited normally
                if os.WIFEXITED(status):
                    return os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    sig = os.WTERMSIG(status)
                    print(f"Child killed by signal {sig}")
                    return 1
                else:
                    return 1
        except ChildProcessError:
            # Child already gone
            return 1

        # Sleep briefly before checking again
        time.sleep(0.1)


def run_tests(ctx, per_rank_logs: bool, test_files: list[Path], test_filter: str | None, log_file) -> bool:
    """Run the actual tests. Returns True if all passed."""
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

    return all_passed


def main():
    # Parse options before creating context (so we can fork early for timeout)
    args = sys.argv[1:]
    per_rank_logs = False
    test_timeout = None
    remaining_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--per-rank-logs":
            per_rank_logs = True
            remaining_args.append(arg)
        elif arg == "--timeout":
            i += 1
            if i < len(args):
                test_timeout = float(args[i])
            else:
                print("Error: --timeout requires a value in seconds", file=sys.stderr)
                sys.exit(1)
        elif arg.startswith("--timeout="):
            test_timeout = float(arg.split("=", 1)[1])
        else:
            remaining_args.append(arg)
        i += 1

    # If timeout is set, fork and monitor
    if test_timeout is not None:
        exit_code = run_with_timeout(test_timeout, remaining_args, per_rank_logs)
        sys.exit(exit_code)

    # No timeout - run tests directly
    log_file = None

    # Set up per-rank logging BEFORE creating context (which creates barrier store)
    if per_rank_logs:
        rank = get_rank_from_env()
        log_file = setup_per_rank_logging(rank, "test_logs")

    ctx = create_context_from_env()

    if per_rank_logs:
        ctx.per_rank_logs = True

    if ctx.rank == 0 or per_rank_logs:
        print(f"Running tests with {ctx.world_size} processes")
        print(f"Master: {ctx.master_addr}:{ctx.master_port}")
        if per_rank_logs:
            print(f"Per-rank logs: test_logs/rank_*.log")
        print()

    # Determine which test files to run
    test_filter = None  # Optional filter for specific test name
    test_args = [arg for arg in remaining_args if not arg.startswith("--")]
    if test_args:
        # Parse arguments - support file.py::test_name syntax
        test_files = []
        for arg in test_args:
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

    all_passed = run_tests(ctx, per_rank_logs, test_files, test_filter, log_file)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
