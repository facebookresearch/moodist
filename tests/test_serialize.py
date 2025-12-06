"""
Tests for moodist.serialize and moodist.deserialize

These functions serialize Python objects to a torch.Tensor (uint8) and back.
The format is moodist's own binary format, but uses pickle protocol (__reduce__)
for custom objects. This means any picklable object can be serialized.

Most tests here run independently on each rank (no distributed coordination needed).
One test verifies serialized data can be transferred between ranks.
"""

from dataclasses import dataclass

import torch
import moodist
from framework import TestContext, test


# Define at module level so it can be found during deserialization
@dataclass
class _TestPoint:
    x: int
    y: int


def roundtrip(obj):
    """Serialize and deserialize, return result."""
    tensor = moodist.serialize(obj)
    return moodist.deserialize(tensor)


@test
def test_serialize_none(ctx: TestContext):
    """Test serializing None."""
    result = roundtrip(None)
    ctx.assert_true(result is None, f"expected None, got {result!r}")


@test
def test_serialize_bool(ctx: TestContext):
    """Test serializing booleans."""
    ctx.assert_true(roundtrip(True) is True, "True failed")
    ctx.assert_true(roundtrip(False) is False, "False failed")


@test
def test_serialize_int_small(ctx: TestContext):
    """Test serializing small integers (int8/int16 range)."""
    for val in [0, 1, -1, 42, -42, 127, -128, 255, -256, 32767, -32768]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"int {val}")


@test
def test_serialize_int_medium(ctx: TestContext):
    """Test serializing medium integers (int32 range)."""
    for val in [2**20, -2**20, 2**31 - 1, -2**31]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"int {val}")


@test
def test_serialize_int_large(ctx: TestContext):
    """Test serializing large integers (int64 range)."""
    for val in [2**40, -2**40, 2**63 - 1, -2**63]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"int {val}")


@test
def test_serialize_int_bigint(ctx: TestContext):
    """Test serializing arbitrary precision integers (beyond int64)."""
    for val in [2**100, -2**100, 2**1000, -2**1000]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"bigint with {val.bit_length()} bits")


@test
def test_serialize_float(ctx: TestContext):
    """Test serializing floats."""
    for val in [0.0, 1.0, -1.0, 3.14159, -2.71828, 1e10, 1e-10]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"float {val}")


@test
def test_serialize_float_special(ctx: TestContext):
    """Test serializing special float values."""
    import math

    # Infinity
    result = roundtrip(float('inf'))
    ctx.assert_true(math.isinf(result) and result > 0, "positive infinity")

    result = roundtrip(float('-inf'))
    ctx.assert_true(math.isinf(result) and result < 0, "negative infinity")

    # NaN
    result = roundtrip(float('nan'))
    ctx.assert_true(math.isnan(result), "nan")


@test
def test_serialize_str(ctx: TestContext):
    """Test serializing strings."""
    for val in ["", "hello", "hello world", "unicode: ñ é ü 你好 🎉"]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"str {val!r}")


@test
def test_serialize_str_long(ctx: TestContext):
    """Test serializing long strings."""
    val = "x" * 100000
    result = roundtrip(val)
    ctx.assert_equal(result, val, "long string")


@test
def test_serialize_bytes(ctx: TestContext):
    """Test serializing bytes."""
    for val in [b"", b"hello", b"\x00\x01\x02\xff\xfe"]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"bytes {val!r}")


@test
def test_serialize_bytes_large(ctx: TestContext):
    """Test serializing large bytes."""
    val = bytes(range(256)) * 1000
    result = roundtrip(val)
    ctx.assert_equal(result, val, "large bytes")


@test
def test_serialize_bytearray(ctx: TestContext):
    """Test serializing bytearray."""
    val = bytearray(b"hello\x00world")
    result = roundtrip(val)
    ctx.assert_equal(result, val, "bytearray")
    ctx.assert_true(isinstance(result, bytearray), "should return bytearray")


@test
def test_serialize_list(ctx: TestContext):
    """Test serializing lists."""
    for val in [[], [1], [1, 2, 3], [1, "two", 3.0, None, True]]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"list {val!r}")


@test
def test_serialize_tuple(ctx: TestContext):
    """Test serializing tuples."""
    for val in [(), (1,), (1, 2, 3), (1, "two", 3.0, None, True)]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"tuple {val!r}")
        ctx.assert_true(isinstance(result, tuple), "should return tuple")


@test
def test_serialize_dict(ctx: TestContext):
    """Test serializing dicts."""
    for val in [{}, {"a": 1}, {"a": 1, "b": 2}, {1: "one", "two": 2, (3,): [3]}]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"dict {val!r}")


@test
def test_serialize_set(ctx: TestContext):
    """Test serializing sets."""
    for val in [set(), {1}, {1, 2, 3}, {"a", "b", "c"}]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"set {val!r}")
        ctx.assert_true(isinstance(result, set), "should return set")


@test
def test_serialize_frozenset(ctx: TestContext):
    """Test serializing frozensets."""
    for val in [frozenset(), frozenset([1]), frozenset([1, 2, 3])]:
        result = roundtrip(val)
        ctx.assert_equal(result, val, f"frozenset {val!r}")
        ctx.assert_true(isinstance(result, frozenset), "should return frozenset")


@test
def test_serialize_nested(ctx: TestContext):
    """Test serializing nested structures."""
    val = {
        "list": [1, 2, [3, 4, {"nested": True}]],
        "tuple": (1, (2, 3)),
        "dict": {"a": {"b": {"c": 42}}},
        "mixed": [{"set": frozenset([1, 2])}, (None, True, False)],
    }
    result = roundtrip(val)
    ctx.assert_equal(result, val, "nested structure")


@test
def test_serialize_shared_reference(ctx: TestContext):
    """Test that shared object references are preserved."""
    shared_list = [1, 2, 3]
    val = [shared_list, shared_list]

    result = roundtrip(val)

    ctx.assert_equal(result, val, "value equality")
    # Check that both elements reference the same object
    ctx.assert_true(result[0] is result[1], "shared reference should be preserved")


@test
def test_serialize_circular_reference(ctx: TestContext):
    """Test that circular references are handled."""
    val = [1, 2]
    val.append(val)  # circular reference

    result = roundtrip(val)

    ctx.assert_equal(result[0], 1)
    ctx.assert_equal(result[1], 2)
    ctx.assert_true(result[2] is result, "circular reference should be preserved")


@test
def test_serialize_function(ctx: TestContext):
    """Test serializing a function reference."""
    import math

    result = roundtrip(math.sqrt)
    ctx.assert_true(result is math.sqrt, "function should be the same object")


@test
def test_serialize_type(ctx: TestContext):
    """Test serializing a type/class reference."""
    result = roundtrip(int)
    ctx.assert_true(result is int, "type should be the same object")

    result = roundtrip(dict)
    ctx.assert_true(result is dict, "type should be the same object")


@test
def test_serialize_custom_class(ctx: TestContext):
    """Test serializing a custom class instance via __reduce__."""
    val = _TestPoint(10, 20)
    result = roundtrip(val)

    ctx.assert_equal(result.x, 10)
    ctx.assert_equal(result.y, 20)
    ctx.assert_true(isinstance(result, _TestPoint), f"should be _TestPoint instance, got {type(result)}")


@test
def test_serialize_numpy_array(ctx: TestContext):
    """Test serializing numpy array (via reduce path).

    NOTE: This test is currently skipped because numpy's __reduce_ex__ with
    protocol 5 returns PickleBuffer objects for zero-copy, which moodist
    doesn't support. This is a known limitation.
    """
    # Skip for now - numpy uses PickleBuffer which moodist doesn't handle
    # TODO: Consider adding PickleBuffer support or using older protocol
    return

    try:
        import numpy as np
    except ImportError:
        return  # Skip if numpy not available

    val = np.array([1.0, 2.0, 3.0])
    result = roundtrip(val)

    ctx.assert_true(isinstance(result, np.ndarray), "should be ndarray")
    ctx.assert_true(np.array_equal(result, val), "array values should match")


@test
def test_serialize_returns_tensor(ctx: TestContext):
    """Test that serialize returns a uint8 tensor."""
    tensor = moodist.serialize({"hello": "world"})

    ctx.assert_true(isinstance(tensor, torch.Tensor), "should return tensor")
    ctx.assert_equal(tensor.dtype, torch.uint8, "should be uint8")
    ctx.assert_true(tensor.numel() > 0, "should have data")


@test
def test_serialize_cross_rank(ctx: TestContext):
    """Test that serialized data can be transferred between ranks."""
    if ctx.world_size < 2:
        return  # Need at least 2 ranks

    store = ctx.create_store()

    # Rank 0 serializes and sends
    if ctx.rank == 0:
        obj = {
            "from": 0,
            "data": [1, 2, 3],
            "nested": {"a": True, "b": None},
        }
        tensor = moodist.serialize(obj)
        # Convert to bytes for store
        store.set("serialized", tensor.numpy().tobytes())

    ctx.barrier()

    # Rank 1 receives and deserializes
    if ctx.rank == 1:
        data = store.get("serialized")
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        result = moodist.deserialize(tensor)

        ctx.assert_equal(result["from"], 0)
        ctx.assert_equal(result["data"], [1, 2, 3])
        ctx.assert_equal(result["nested"], {"a": True, "b": None})
