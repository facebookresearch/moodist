"""
Minimal test framework for moodist distributed tests.

Usage:
    from framework import TestContext, test

    @test
    def test_something(ctx: TestContext):
        ctx.assert_equal(1 + 1, 2)

    # Or for distributed tests:
    @test
    def test_distributed(ctx: TestContext):
        store = ctx.create_store()
        store.set(f"key_{ctx.rank}", b"value")
        ctx.barrier()
        # All ranks can now see all keys
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Optional

import torch


@dataclass
class TestContext:
    """Context passed to each test function."""
    rank: int
    world_size: int
    local_rank: int
    master_addr: str
    master_port: int
    barrier_store: "moodist.TcpStore"  # Created at startup, shared across all tests

    _barrier_count: int = field(default=0, repr=False)
    _test_store_count: int = field(default=0, repr=False)
    _current_test_id: int = field(default=0, repr=False)
    _keepalive: list = field(default_factory=list, repr=False)  # Keep objects alive until barrier

    def _set_test_id(self, test_id: int):
        """Called by TestRunner before each test to set a consistent test ID."""
        self._current_test_id = test_id
        self._test_store_count = 0

    def _cleanup(self):
        """Called after test barrier to release kept-alive objects."""
        self._keepalive.clear()

    def keep_alive(self, obj):
        """Keep an object alive until after the post-test barrier.

        Use this for stores, process groups, queues, etc. that may have
        pending operations when the test function returns.
        """
        self._keepalive.append(obj)
        return obj

    def create_store(self, key: str = "test", timeout: timedelta = timedelta(seconds=30)):
        """Create a TcpStore for this test. All ranks will use the same port."""
        import moodist
        # Port is based on test ID and store count within the test, ensuring consistency
        self._test_store_count += 1
        port = self.master_port + 1000 + self._current_test_id * 100 + self._test_store_count
        store = moodist.TcpStore(
            self.master_addr, port, key, self.world_size, self.rank, timeout
        )
        # Keep a reference so the store isn't destroyed until after the barrier
        self._keepalive.append(store)
        return store

    def barrier(self):
        """Barrier using the shared barrier store."""
        barrier_id = self._barrier_count
        self._barrier_count += 1
        self.barrier_store.set(f"barrier_{barrier_id}_{self.rank}", b"1")
        for r in range(self.world_size):
            self.barrier_store.get(f"barrier_{barrier_id}_{r}")

    def log(self, msg: str):
        """Print a message prefixed with rank info."""
        print(f"[rank {self.rank}/{self.world_size}] {msg}", flush=True)

    def assert_true(self, condition: bool, msg: str = ""):
        if not condition:
            raise AssertionError(f"assert_true failed: {msg}" if msg else "assert_true failed")

    def assert_false(self, condition: bool, msg: str = ""):
        if condition:
            raise AssertionError(f"assert_false failed: {msg}" if msg else "assert_false failed")

    def assert_equal(self, a, b, msg: str = ""):
        if a != b:
            raise AssertionError(f"assert_equal failed: {a!r} != {b!r}" + (f" ({msg})" if msg else ""))

    def assert_raises(self, exc_type: type, fn: Callable, *args, **kwargs):
        """Assert that fn(*args, **kwargs) raises exc_type."""
        try:
            fn(*args, **kwargs)
        except exc_type:
            return
        except Exception as e:
            raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")
        raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised")


# Registry of test functions
_tests: list[tuple[str, Callable[[TestContext], None]]] = []


def test(fn: Callable[[TestContext], None]) -> Callable[[TestContext], None]:
    """Decorator to register a test function."""
    _tests.append((fn.__name__, fn))
    return fn


def test_devices(*devices: str):
    """Decorator to run a test for each specified device.

    Usage:
        @test_devices("cpu", "cuda")
        def test_something(ctx: TestContext, device: str):
            tensor = torch.tensor([1.0], device=device)
            ...

    This registers test_something_cpu and test_something_cuda.
    """
    def decorator(fn: Callable[[TestContext, str], None]):
        for device in devices:
            def make_wrapper(dev):
                def wrapper(ctx: TestContext):
                    return fn(ctx, dev)
                return wrapper
            wrapper = make_wrapper(device)
            wrapper.__name__ = f"{fn.__name__}_{device}"
            _tests.append((wrapper.__name__, wrapper))
        return fn
    return decorator


def test_cpu_cuda(fn: Callable[[TestContext, str], None]):
    """Shortcut for @test_devices("cpu", "cuda")."""
    return test_devices("cpu", "cuda")(fn)


def get_tests() -> list[tuple[str, Callable[[TestContext], None]]]:
    """Return list of (name, function) for all registered tests."""
    return _tests.copy()


def clear_tests():
    """Clear the test registry (useful for testing the framework itself)."""
    _tests.clear()


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None


class TestRunner:
    """Runs tests and collects results."""

    def __init__(self, ctx: TestContext):
        self.ctx = ctx
        self.results: list[TestResult] = []
        self._test_counter: int = 0

    def run_test(self, name: str, fn: Callable[[TestContext], None]) -> TestResult:
        """Run a single test, return result."""
        # Set a unique test ID so all ranks use consistent ports
        self.ctx._set_test_id(self._test_counter)
        self._test_counter += 1

        # Barrier before test to ensure all ranks start together
        self.ctx.barrier()

        start = time.monotonic()
        error = None

        try:
            fn(self.ctx)
        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        duration = time.monotonic() - start
        passed = error is None

        # Barrier after test to ensure no rank exits early and destroys stores
        try:
            self.ctx.barrier()
        except Exception:
            pass  # Don't mask test failures with barrier failures

        # Now it's safe to cleanup - all ranks have passed the barrier
        self.ctx._cleanup()

        result = TestResult(name=name, passed=passed, duration=duration, error=error)
        self.results.append(result)
        return result

    def run_all(self, tests: Optional[list[tuple[str, Callable]]] = None):
        """Run all registered tests (or provided list)."""
        if tests is None:
            tests = get_tests()

        for name, fn in tests:
            result = self.run_test(name, fn)
            status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"

            # Only rank 0 prints summary line
            if self.ctx.rank == 0:
                print(f"  {name:<50} {status}  ({result.duration:.3f}s)")

            if not result.passed and self.ctx.rank == 0:
                # Print error details
                print(f"    Error: {result.error}")

    def summarize(self) -> bool:
        """Print summary, return True if all passed."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        failed = total - passed

        if self.ctx.rank == 0:
            print()
            if failed == 0:
                print(f"\033[32m{passed}/{total} tests passed - ALL OK\033[0m")
            else:
                print(f"\033[31m{passed}/{total} tests passed - {failed} FAILED\033[0m")

        return failed == 0


def create_context_from_env() -> TestContext:
    """Create TestContext from environment variables (torchrun or slurm)."""
    import subprocess
    import moodist

    if "WORLD_SIZE" in os.environ:
        # torchrun style
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ:
        # slurm style
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        )
        master_addr = hostnames.split()[0].decode("utf-8")
        master_port = 29500  # default
        local_rank = int(os.environ["SLURM_LOCALID"])
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        # Single process mode for local testing
        master_addr = "127.0.0.1"
        master_port = 29500
        local_rank = 0
        rank = 0
        world_size = 1

    # Create barrier store at startup - all ranks are synchronized by torchrun/slurm
    barrier_store = moodist.TcpStore(
        master_addr, master_port + 999, "barrier", world_size, rank,
        timedelta(seconds=120)
    )

    return TestContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        barrier_store=barrier_store,
    )


def create_process_group(ctx: TestContext):
    """Helper to create a ProcessGroup for testing.

    Creates a new MoodistProcessGroup with its own store. The process group
    is automatically kept alive until after the post-test barrier.
    """
    from datetime import timedelta
    import torch
    import moodist

    store = ctx.create_store(key="pg", timeout=timedelta(seconds=60))
    torch.cuda.set_device(ctx.local_rank)
    pg = moodist.MoodistProcessGroup(store, ctx.rank, ctx.world_size)
    ctx.keep_alive(pg)
    return pg
