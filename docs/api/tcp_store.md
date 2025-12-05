# moodist.TcpStore

A distributed, decentralized key-value store for process group coordination.

## Class Signature

```python
class TcpStore(torch.distributed.Store):
    def __init__(
        self,
        hostname: str,
        port: int,
        key: str,
        world_size: int,
        rank: int,
        timeout: timedelta
    )
```

## Overview

`TcpStore` is a distributed hash table (DHT) implementation of PyTorch's `Store` interface. Unlike PyTorch's standard `TCPStore` which uses a centralized master-worker architecture, Moodist's `TcpStore` distributes data across all ranks, providing better scalability for large-scale distributed training.

**Key Features:**
- Decentralized architecture - no single point of failure or bottleneck
- Distributed key storage using consistent hashing
- Mesh network topology with automatic routing
- Auto-reconnection and keepalive for fault tolerance
- Compatible with PyTorch's distributed initialization

**Comparison with PyTorch's TCPStore:**

| Feature | PyTorch TCPStore | Moodist TcpStore |
|---------|------------------|------------------|
| Architecture | Centralized (master on rank 0) | Decentralized (distributed across all ranks) |
| Key storage | All keys on master | Keys hashed across ranks |
| Scalability | Master can become bottleneck | Load distributed evenly |
| Fault tolerance | Master failure is fatal | Mesh topology with reconnection |
| Network topology | Star (all connect to master) | Mesh (ring + cross-connections) |

## Constructor Parameters

### `hostname`
**Type:** `str`

The hostname or IP address of rank 0. All ranks use this to bootstrap the connection process, but unlike PyTorch's TCPStore, rank 0 is not a centralized master - it just serves as the initial rendezvous point.

### `port`
**Type:** `int`

The port number for the initial connection to rank 0.

### `key`
**Type:** `str`

A unique key identifying this store instance. Multiple process groups can coexist by using different keys. The key is combined with hostname, port, and world_size to create a unique store identifier.

### `world_size`
**Type:** `int`

The total number of ranks participating in this store.

### `rank`
**Type:** `int`

The rank of this process (0 to world_size - 1).

### `timeout`
**Type:** `timedelta`

The default timeout for store operations (get, set, wait, etc.).

## Methods

### `set(key, value)`

Store a key-value pair.

**Parameters:**
- `key`: `str` - The key to store
- `value`: `bytes` or `list[int]` - The value to store

The key is hashed to determine which rank stores it. The operation blocks until the value is stored and acknowledged.

### `get(key)`

Retrieve a value by key.

**Parameters:**
- `key`: `str` - The key to retrieve

**Returns:** `bytes` - The stored value

**Raises:** `RuntimeError` if the key doesn't exist or timeout expires

The request is routed to the rank that owns the key based on consistent hashing.

### `wait(keys, timeout=None)`

Wait for keys to be set.

**Parameters:**
- `keys`: `list[str]` - Keys to wait for
- `timeout`: `timedelta` (optional) - Override the default timeout

Blocks until all specified keys have been set by some rank.

### `check(keys)`

Check if keys exist.

**Parameters:**
- `keys`: `list[str]` - Keys to check

**Returns:** `bool` - `True` if all keys exist

### Unimplemented Methods

The following methods from PyTorch's Store interface are not implemented and will raise `RuntimeError`:

- `add(key, value)` - Atomic increment
- `deleteKey(key)` - Delete a key
- `getNumKeys()` - Count total keys

## Architecture

### Distributed Hash Table

Keys are distributed across ranks using consistent hashing:

```
storage_rank = hash(key) % world_size
```

This means:
- Each rank stores approximately `1/world_size` of all keys
- No single rank becomes a bottleneck
- Key lookups are routed to the owning rank

### Mesh Network Topology

Instead of all ranks connecting to a central master, `TcpStore` builds a mesh network:

1. **Ring connections**: Each rank connects to rank `(rank + 1) % world_size`
2. **Cross-connections**: Additional random connections to reduce path lengths

This creates a graph where any rank can reach any other rank in a small number of hops. Messages are routed using shortest-path routing tables computed at startup.

### Connection Protocol

1. **UDP Discovery**: Ranks send UDP packets to find each other and exchange TCP addresses
2. **TCP Data Transfer**: Reliable message passing over TCP connections
3. **Keepalive**: Periodic heartbeats detect and recover from connection failures
4. **Auto-reconnection**: Failed connections are automatically re-established

### Message Routing

When a rank needs to access a key stored on another rank:

1. Compute the destination rank from the key hash
2. Look up the first hop in the routing table
3. Send the message to the first hop
4. Intermediate ranks forward the message toward the destination
5. The destination processes the request and routes the response back

## Examples

### Example 1: Basic Usage

```python
from datetime import timedelta
import moodist

# Create store on each rank
store = moodist.TcpStore(
    hostname="master-node",
    port=29500,
    key="my_training_job",
    world_size=8,
    rank=rank,
    timeout=timedelta(seconds=300)
)

# Store a value (will be routed to owning rank)
if rank == 0:
    store.set("config", b'{"learning_rate": 0.001}')

# All ranks can retrieve it
store.wait(["config"])
config = store.get("config")
```

### Example 2: Using with PyTorch Distributed

```python
import torch.distributed as dist
import moodist

# TcpStore can be used for distributed initialization
store = moodist.TcpStore(
    hostname="master-node",
    port=29500,
    key="default_pg",
    world_size=world_size,
    rank=rank,
    timeout=timedelta(minutes=5)
)

# Use with init_process_group
dist.init_process_group(
    backend="moodist",
    store=store,
    rank=rank,
    world_size=world_size
)
```

### Example 3: Coordination Between Ranks

```python
from datetime import timedelta
import moodist

store = moodist.TcpStore(
    hostname="master-node",
    port=29500,
    key="sync_job",
    world_size=4,
    rank=rank,
    timeout=timedelta(seconds=60)
)

# Each rank signals when ready
store.set(f"rank_{rank}_ready", b"1")

# Wait for all ranks
store.wait([f"rank_{i}_ready" for i in range(4)])
print(f"Rank {rank}: All ranks ready!")
```

## Performance Considerations

### Advantages of Distributed Design

- **No bottleneck**: In PyTorch's TCPStore, all operations go through rank 0. With thousands of ranks, this becomes a bottleneck. Moodist's TcpStore distributes the load.

- **Parallel operations**: Multiple ranks can perform store operations simultaneously without contention (as long as they access different keys).

- **Reduced latency for nearby keys**: Keys owned by nearby ranks (in the mesh topology) have lower access latency.

### Trade-offs

- **Multi-hop routing**: Accessing a key may require routing through intermediate ranks, adding latency compared to direct access.

- **Startup overhead**: Building the mesh topology requires initial coordination.

- **Not all methods implemented**: `add()`, `deleteKey()`, and `getNumKeys()` are not available.

## Timeout Diagnostics

When a `get()` or `wait()` operation times out, TcpStore provides detailed diagnostic information to help identify the cause. This is especially useful in large distributed systems where connectivity issues may be complex.

### Example Error Message

```
RuntimeError: Moodist Store get(my_key): timed out after 30 seconds
  No TCP connection to rank 5 (first hop to target rank 12)
  [T+0.0s, 30.1s ago] Rank 3: TCP connection to rank 0 established
  [T+0.1s, 30.0s ago] Rank 3: TCP connection to rank 4 established
  [T+2.5s, 27.6s ago] Rank 3: TCP connection to rank 5 lost: Connection reset by peer
  [T+15.0s, 15.1s ago] Rank 3: no TCP connection to rank 5 (addresses known)
```

### Understanding the Diagnostics

The error message includes:

1. **Local connection status**: The first line after the timeout message describes the local rank's connection state to the relevant peer (typically the first hop toward the target rank).

2. **Diagnostic reports from other ranks**: Subsequent lines show connection events reported by intermediate ranks in the mesh topology. Each line includes:
   - `[T+X.Xs, Y.Ys ago]`: Two timestamps - time since the store was created, and time before the timeout
   - `Rank N`: The rank reporting the diagnostic
   - The event description (connection established, lost, waiting, etc.)

### Diagnostic Event Types

- **TCP connection to rank N established**: A connection to another rank was successfully made
- **TCP connection to rank N lost: \<error\>**: A previously working connection failed, with the error reason
- **no TCP connection to rank N (addresses known)**: Cannot connect but have the peer's addresses
- **waiting for rank N to reconnect**: Had a connection before, waiting for it to recover
- **waiting for rank N to connect**: Never had a connection to this rank yet
- **message pending to rank N for Xs (TCP connected)**: Connected but message not being processed

### How Diagnostics Work

The diagnostic system uses UDP to exchange connection state information between ranks:

1. When a request is made, the originating rank becomes the "diagnostic source"
2. Intermediate ranks that encounter connection issues report back to the diagnostic source via UDP
3. These reports are collected and included in the timeout error message
4. Reports are deduplicated and filtered to show the most relevant information

This distributed diagnostic approach means you can identify connection problems anywhere in the mesh topology, not just on the local rank.

## See Also

- [`moodist.MoodistProcessGroup`](process_group.md) - Process group management
- [PyTorch Distributed Store](https://pytorch.org/docs/stable/distributed.html#torch.distributed.Store) - Base Store interface
