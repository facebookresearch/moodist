"""
Tests for moodist.TcpStore

TcpStore is the foundational component for distributed coordination.
These tests verify basic key-value operations across multiple ranks.
"""

from datetime import timedelta

import moodist
from framework import TestContext, test


@test
def test_noop(ctx: TestContext):
    """Dummy test that does nothing - tests barrier store shutdown only."""
    pass


@test
def test_store_basic_set_get(ctx: TestContext):
    """Each rank sets its own key and retrieves it."""
    store = ctx.create_store()

    key = f"mykey_{ctx.rank}"
    value = f"value_from_rank_{ctx.rank}".encode()

    store.set(key, value)
    result = store.get(key)

    ctx.assert_equal(result, value)


@test
def test_store_cross_rank_visibility(ctx: TestContext):
    """Verify that keys set by one rank are visible to all other ranks."""
    store = ctx.create_store()

    # Each rank sets a unique key
    my_key = f"rank_{ctx.rank}_key"
    my_value = f"hello_from_{ctx.rank}".encode()
    store.set(my_key, my_value)

    # Barrier to ensure all ranks have set their keys
    ctx.barrier()

    # Each rank reads all other ranks' keys
    for r in range(ctx.world_size):
        key = f"rank_{r}_key"
        expected = f"hello_from_{r}".encode()
        result = store.get(key)
        ctx.assert_equal(result, expected, f"reading key from rank {r}")


@test
def test_store_wait_for_keys(ctx: TestContext):
    """Test wait() blocking until keys are available."""
    store = ctx.create_store()

    # Rank 0 will wait for a key that rank 1 sets (if we have multiple ranks)
    if ctx.world_size > 1:
        if ctx.rank == 0:
            # Wait for the key from rank 1
            store.wait(["key_from_rank_1"])
            # Now we should be able to get it
            result = store.get("key_from_rank_1")
            ctx.assert_equal(result, b"value_1")
        elif ctx.rank == 1:
            # Set the key (rank 0 is waiting for it)
            store.set("key_from_rank_1", b"value_1")
    else:
        # Single rank: just verify wait works on existing key
        store.set("solo_key", b"solo_value")
        store.wait(["solo_key"])
        result = store.get("solo_key")
        ctx.assert_equal(result, b"solo_value")

    ctx.barrier()


@test
def test_store_check_keys(ctx: TestContext):
    """Test check() returns True only when all keys exist."""
    store = ctx.create_store()

    # Initially, check should return False for non-existent keys
    exists = store.check([f"nonexistent_key_{ctx.rank}"])
    ctx.assert_false(exists, "check should return False for non-existent key")

    # Set a key
    my_key = f"check_test_{ctx.rank}"
    store.set(my_key, b"exists")

    # Now check should return True for our own key
    exists = store.check([my_key])
    ctx.assert_true(exists, "check should return True after key is set")

    ctx.barrier()

    # After barrier, all keys should exist - check all of them
    all_keys = [f"check_test_{r}" for r in range(ctx.world_size)]
    exists = store.check(all_keys)
    ctx.assert_true(exists, "check should return True for all keys after barrier")


@test
def test_store_overwrite_key(ctx: TestContext):
    """Test that setting a key twice overwrites the value."""
    store = ctx.create_store()

    key = f"overwrite_test_{ctx.rank}"

    store.set(key, b"first_value")
    result1 = store.get(key)
    ctx.assert_equal(result1, b"first_value")

    store.set(key, b"second_value")
    result2 = store.get(key)
    ctx.assert_equal(result2, b"second_value")


@test
def test_store_binary_values(ctx: TestContext):
    """Test storing and retrieving binary data with null bytes."""
    store = ctx.create_store()

    key = f"binary_test_{ctx.rank}"
    # Value with null bytes and various byte values
    value = bytes([0, 1, 2, 255, 254, 0, 128, 64, 0])

    store.set(key, value)
    result = store.get(key)

    ctx.assert_equal(result, value)


@test
def test_store_empty_value(ctx: TestContext):
    """Test storing and retrieving empty value."""
    store = ctx.create_store()

    key = f"empty_test_{ctx.rank}"
    value = b""

    store.set(key, value)
    result = store.get(key)

    ctx.assert_equal(result, value)


@test
def test_store_large_value(ctx: TestContext):
    """Test storing and retrieving a larger value."""
    store = ctx.create_store()

    key = f"large_test_{ctx.rank}"
    # 1MB of data
    value = bytes(range(256)) * 4096

    store.set(key, value)
    result = store.get(key)

    ctx.assert_equal(len(result), len(value))
    ctx.assert_equal(result, value)


@test
def test_store_many_keys(ctx: TestContext):
    """Test setting and getting many keys."""
    store = ctx.create_store()

    num_keys = 100

    # Set many keys
    for i in range(num_keys):
        key = f"manykeys_{ctx.rank}_{i}"
        value = f"value_{i}".encode()
        store.set(key, value)

    # Read them all back
    for i in range(num_keys):
        key = f"manykeys_{ctx.rank}_{i}"
        expected = f"value_{i}".encode()
        result = store.get(key)
        ctx.assert_equal(result, expected, f"key {i}")


@test
def test_store_timeout_on_missing_key(ctx: TestContext):
    """Test that get() times out on a non-existent key."""
    # Create store with short timeout - use create_store for consistent ports
    store = ctx.create_store(key="timeout_test", timeout=timedelta(milliseconds=500))

    # This should raise an exception due to timeout
    ctx.assert_raises(
        RuntimeError,
        store.get,
        "this_key_does_not_exist"
    )


@test
def test_store_graceful_shutdown(ctx: TestContext):
    """Test that stores shut down gracefully without 'store was shut down' errors.

    This verifies the two-phase shutdown protocol:
    1. All ranks signal shutdown
    2. All ranks wait for peers to acknowledge
    3. All ranks signal exit
    4. No spurious errors during coordinated shutdown
    """
    import time

    # Create store directly (not via create_store) so we have full control
    # over when it's destroyed - create_store keeps a reference in _keepalive
    store = moodist.TcpStore(
        ctx.master_addr,
        ctx.master_port + 1900,  # Use unique port to avoid conflicts
        "shutdown_test",
        ctx.world_size,
        ctx.rank,
        timedelta(seconds=30),
    )

    # Do some operations to ensure the store is fully connected
    store.set(f"key_{ctx.rank}", f"value_{ctx.rank}".encode())
    ctx.barrier()

    # Verify all keys are readable
    for r in range(ctx.world_size):
        result = store.get(f"key_{r}")
        ctx.assert_equal(result, f"value_{r}".encode())

    # Ensure all ranks finish their gets before any rank starts shutdown
    ctx.barrier()

    # Stagger the shutdown slightly - rank 0 destroys first, others follow
    # This tests that the two-phase protocol handles asymmetric shutdown timing
    # Cap at 500ms to stay well under the 2s timeout even for large world sizes
    time.sleep(min(0.01 * ctx.rank, 0.5))

    # Explicitly delete the store to trigger shutdown
    # The two-phase protocol should prevent "store was shut down on rank X" errors
    del store

    # If we get here without exceptions, the graceful shutdown worked
    ctx.barrier()


@test
def test_store_clone(ctx: TestContext):
    """Test that clone() shares the underlying store and refcounting works."""
    import time

    store = moodist.TcpStore(
        ctx.master_addr,
        ctx.master_port + 2100,
        "clone_test",
        ctx.world_size,
        ctx.rank,
        timedelta(seconds=10),
    )

    # Set a key via original
    store.set(f"original_{ctx.rank}", b"from_original")

    # Clone the store
    cloned = store.clone()

    # Both should see the key
    ctx.assert_equal(store.get(f"original_{ctx.rank}"), b"from_original")
    ctx.assert_equal(cloned.get(f"original_{ctx.rank}"), b"from_original")

    # Set via clone, read via original
    cloned.set(f"cloned_{ctx.rank}", b"from_clone")
    ctx.assert_equal(store.get(f"cloned_{ctx.rank}"), b"from_clone")

    ctx.barrier()

    # Delete original - clone should still work
    del store
    cloned.set(f"after_original_deleted_{ctx.rank}", b"still_works")
    ctx.assert_equal(cloned.get(f"after_original_deleted_{ctx.rank}"), b"still_works")

    ctx.barrier()

    # Now delete clone - this should trigger shutdown
    del cloned

    ctx.barrier()


@test
def test_store_wait_multiple_keys(ctx: TestContext):
    """Test wait() on multiple keys that appear at different times."""
    import time

    if ctx.world_size < 2:
        return

    store = ctx.create_store(key="wait_multi")

    if ctx.rank == 0:
        # Set keys with delays
        store.set("multi_key_1", b"value1")
        time.sleep(0.1)
        store.set("multi_key_2", b"value2")
        time.sleep(0.1)
        store.set("multi_key_3", b"value3")
    elif ctx.rank == 1:
        # Wait for all three keys at once
        store.wait(["multi_key_1", "multi_key_2", "multi_key_3"])
        # All should be available now
        ctx.assert_equal(store.get("multi_key_1"), b"value1")
        ctx.assert_equal(store.get("multi_key_2"), b"value2")
        ctx.assert_equal(store.get("multi_key_3"), b"value3")

    ctx.barrier()


@test
def test_store_wait_with_timeout(ctx: TestContext):
    """Test wait() with explicit timeout parameter."""
    store = ctx.create_store(key="wait_timeout")

    # Set a key first
    store.set(f"exists_{ctx.rank}", b"value")

    # Wait with explicit short timeout - should succeed immediately
    store.wait([f"exists_{ctx.rank}"], timedelta(milliseconds=100))

    # Wait for non-existent key with short timeout - should fail
    ctx.assert_raises(
        RuntimeError,
        lambda: store.wait(["nonexistent_key_xyz"], timedelta(milliseconds=100)),
    )


@test
def test_store_empty_key(ctx: TestContext):
    """Test storing with empty key name."""
    store = ctx.create_store(key="empty_key_test")

    # Empty key should work (or raise a clear error)
    try:
        store.set("", b"empty_key_value")
        result = store.get("")
        ctx.assert_equal(result, b"empty_key_value")
    except RuntimeError:
        # If empty keys are not allowed, that's acceptable too
        pass


@test
def test_store_long_key(ctx: TestContext):
    """Test storing with very long key name."""
    store = ctx.create_store(key="long_key_test")

    # 10KB key name
    long_key = "k" * 10240 + f"_{ctx.rank}"

    store.set(long_key, b"long_key_value")
    result = store.get(long_key)
    ctx.assert_equal(result, b"long_key_value")


@test
def test_store_special_chars_in_key(ctx: TestContext):
    """Test keys with special characters."""
    store = ctx.create_store(key="special_chars")

    test_cases = [
        (f"spaces in key {ctx.rank}", b"spaces"),
        (f"slashes/in/key/{ctx.rank}", b"slashes"),
        (f"unicode_\u00e9\u00e0\u00fc_{ctx.rank}", b"unicode"),
        (f"newline\nkey_{ctx.rank}", b"newline"),
        (f"tab\tkey_{ctx.rank}", b"tab"),
    ]

    for key, value in test_cases:
        store.set(key, value)
        result = store.get(key)
        ctx.assert_equal(result, value, f"key: {repr(key)}")


@test
def test_store_concurrent_writes_same_key(ctx: TestContext):
    """Test multiple ranks writing to the same key concurrently."""
    if ctx.world_size < 2:
        return

    store = ctx.create_store(key="concurrent_write")

    # All ranks write to the same key simultaneously
    shared_key = "shared_key"
    my_value = f"value_from_rank_{ctx.rank}".encode()

    store.set(shared_key, my_value)

    ctx.barrier()

    # After barrier, read the key - should have some valid value
    # (we can't predict which rank "won", but it should be one of them)
    result = store.get(shared_key)
    valid_values = [f"value_from_rank_{r}".encode() for r in range(ctx.world_size)]
    ctx.assert_true(
        result in valid_values,
        f"Expected one of {valid_values}, got {result}"
    )


@test
def test_store_operations_after_shutdown_error(ctx: TestContext):
    """Test behavior when trying to use store after receiving shutdown error."""
    import time

    if ctx.world_size < 2:
        return

    store = moodist.TcpStore(
        ctx.master_addr,
        ctx.master_port + 2200,
        "ops_after_shutdown",
        ctx.world_size,
        ctx.rank,
        timedelta(seconds=5),
    )

    store.set(f"key_{ctx.rank}", b"value")
    ctx.barrier()

    if ctx.rank == 0:
        # Rank 0 shuts down
        del store
        time.sleep(2.0)
    else:
        time.sleep(0.5)
        # First operation should get shutdown error
        try:
            store.wait(["never_exists"])
        except RuntimeError as e:
            ctx.assert_true("shut down" in str(e).lower(), f"Expected shutdown error, got: {e}")

        # Subsequent operations should also fail gracefully (not crash)
        try:
            store.set("after_shutdown", b"value")
        except RuntimeError:
            pass  # Expected to fail

        try:
            store.get("key_0")
        except RuntimeError:
            pass  # Expected to fail

        del store

    ctx.barrier()


@test
def test_store_unexpected_shutdown_error(ctx: TestContext):
    """Test that unexpected shutdown still produces 'store was shut down' error."""
    import time

    if ctx.world_size < 2:
        return

    # Create store directly (not via create_store) so we have full control
    # over when it's destroyed - create_store keeps a reference in _keepalive
    store = moodist.TcpStore(
        ctx.master_addr,
        ctx.master_port + 2000,  # Use unique port to avoid conflicts
        "unexpected_shutdown_test",
        ctx.world_size,
        ctx.rank,
        timedelta(seconds=10),  # Longer than 2s shutdown grace period
    )

    # Ensure store is connected
    store.set(f"key_{ctx.rank}", f"value_{ctx.rank}".encode())
    ctx.barrier()

    if ctx.rank == 0:
        del store
        time.sleep(3.0)
    elif ctx.rank == 1:
        time.sleep(0.1)
        try:
            store.wait(["key_that_was_never_set"])
            ctx.assert_true(False, "Expected RuntimeError but wait() returned")
        except RuntimeError as e:
            ctx.assert_true(
                "shut down" in str(e).lower(),
                f"Expected 'shut down' in error message, got: {e}"
            )
        del store
    else:
        time.sleep(0.5)
        del store

    ctx.barrier()
