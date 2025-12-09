"""
Benchmark for moodist.serialize focusing on ObjMap performance.

ObjMap is used to track object references during serialization:
- Maps PyObject* -> offset (to detect repeated objects)
- The hash table uses linear probing with 4-slot limit

This benchmark tests various patterns that stress the hash table:
1. Many unique objects (tests insert performance)
2. Repeated references (tests lookup performance)
3. Objects of uniform size (adversarial for naive hash)
"""

import time
import pickle
import torch
import moodist


def benchmark(fn, iterations=100, warmup=10):
    """Run a benchmark, return median time in milliseconds."""
    # Warmup - important for JIT, allocations, cache warming
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    return median, p10, p90


def run_benchmark(name, data, include_deserialize=False):
    """Run moodist and pickle benchmarks on the same data."""
    # Pre-serialize for deserialization benchmarks
    moodist_serialized = moodist.serialize(data)
    pickle_serialized = pickle.dumps(data)

    # Serialize benchmarks
    moodist_ser, moodist_ser_p10, moodist_ser_p90 = benchmark(
        lambda: moodist.serialize(data)
    )
    pickle_ser, _, _ = benchmark(lambda: pickle.dumps(data))

    # Size comparison
    moodist_size = len(bytes(moodist_serialized))
    pickle_size = len(pickle_serialized)

    result = {
        "name": name,
        "moodist_ser": moodist_ser,
        "moodist_ser_p10": moodist_ser_p10,
        "moodist_ser_p90": moodist_ser_p90,
        "pickle_ser": pickle_ser,
        "speedup_ser": pickle_ser / moodist_ser,
        "size_ratio": moodist_size / pickle_size,
    }

    if include_deserialize:
        moodist_de, moodist_de_p10, moodist_de_p90 = benchmark(
            lambda: moodist.deserialize(moodist_serialized)
        )
        pickle_de, _, _ = benchmark(lambda: pickle.loads(pickle_serialized))
        result.update({
            "moodist_de": moodist_de,
            "moodist_de_p10": moodist_de_p10,
            "moodist_de_p90": moodist_de_p90,
            "pickle_de": pickle_de,
            "speedup_de": pickle_de / moodist_de,
        })

    return result


def main():
    print("=" * 100)
    print("moodist vs pickle Benchmark")
    print("=" * 100)
    print()

    # Define all benchmark workloads
    # Each is (name, data, include_deserialize)
    workloads = [
        ("many_unique_strings (10k)", [f"string_{i}" for i in range(10000)], False),
        ("many_unique_ints (10k)", [i + 1000000 for i in range(10000)], False),
        ("repeated_refs (100x100)", [f"shared_{i % 100}" for i in range(10000)], False),
        ("uniform_3char (10k)", [f"a{i:02d}" for i in range(10000)], False),
        ("uniform_9char (10k)", [f"str{i:06d}" for i in range(10000)], False),
        ("nested_dicts (5^5)", make_nested(5, 5), False),
        ("wide_dict (10k keys)", {f"key_{i}": i for i in range(10000)}, True),
        ("list_of_tuples (10k)", [(f"input_{i}", i) for i in range(10000)], True),
        ("mixed_types (10k)", make_mixed_data(), True),
    ]

    # Print header
    print(f"{'Benchmark':<25} {'moodist':>8} {'p10':>6} {'p90':>6} {'pickle':>8} {'speedup':>8} {'size':>6}")
    print("-" * 100)

    results = []
    for name, data, include_de in workloads:
        try:
            r = run_benchmark(name, data, include_de)
            results.append(r)

            # Print serialize results
            print(f"{name:<25} {r['moodist_ser']:>7.2f}ms {r['moodist_ser_p10']:>5.2f}ms {r['moodist_ser_p90']:>5.2f}ms "
                  f"{r['pickle_ser']:>7.2f}ms {r['speedup_ser']:>7.2f}x {r['size_ratio']:>5.2f}x")

            # Print deserialize results if included
            if include_de:
                print(f"  {'(deserialize)':<23} {r['moodist_de']:>7.2f}ms {r['moodist_de_p10']:>5.2f}ms {r['moodist_de_p90']:>5.2f}ms "
                      f"{r['pickle_de']:>7.2f}ms {r['speedup_de']:>7.2f}x")

        except Exception as e:
            print(f"{name:<25} ERROR: {e}")

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    # Check uniform size patterns
    uniform_times = [r["moodist_ser"] for r in results if "uniform" in r["name"]]
    other_times = [r["moodist_ser"] for r in results if "uniform" not in r["name"]]

    if uniform_times and other_times:
        avg_uniform = sum(uniform_times) / len(uniform_times)
        avg_other = sum(other_times) / len(other_times)
        ratio = avg_uniform / avg_other if avg_other > 0 else 0

        print(f"Avg uniform size:    {avg_uniform:.2f} ms")
        print(f"Avg other patterns:  {avg_other:.2f} ms")
        print(f"Ratio (uniform/other): {ratio:.2f}x")

        if ratio > 2.0:
            print("\nWARNING: Uniform size patterns are significantly slower!")
            print("This may indicate hash collision issues.")
        else:
            print("\nOK: No significant slowdown for uniform size patterns.")

    # Overall speedup
    avg_speedup = sum(r["speedup_ser"] for r in results) / len(results)
    avg_size = sum(r["size_ratio"] for r in results) / len(results)
    print(f"\nAvg serialize speedup vs pickle: {avg_speedup:.2f}x")
    print(f"Avg size ratio (moodist/pickle): {avg_size:.2f}x")


def make_nested(depth, width):
    """Create nested dict structure."""
    if depth == 0:
        return "leaf"
    return {f"k{i}": make_nested(depth - 1, width) for i in range(width)}


def make_mixed_data():
    """Create mixed type data."""
    data = []
    for i in range(2500):
        data.extend([
            i,                      # int
            f"str_{i}",             # string
            float(i) / 1000,        # float
            {"k": i, "v": str(i)},  # dict
        ])
    return data


if __name__ == "__main__":
    main()
