"""
Tests for moodist.TcpStore

TcpStore is the foundational component for distributed coordination.
These tests verify basic key-value operations across multiple ranks.
"""

from datetime import timedelta

import moodist
from framework import TestContext, test


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
