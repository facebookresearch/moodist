# TcpStore Internals

This document covers the internal implementation of `moodist.TcpStore`. For the user-facing API, see [docs/api/tcp_store.md](../api/tcp_store.md).

## Overview

TcpStore is a decentralized distributed key-value store. Key implementation details:

- **Mesh network topology** with spanning tree routing
- **Consistent hashing** for key distribution across ranks
- **Two-phase shutdown protocol** for graceful termination
- **UDP diagnostics** for timeout debugging

Code location: `store.cc`

## Mesh Network Topology

### Topology Construction

The mesh is built in `findPaths()`:

- **Small clusters (worldSize <= 4)**: Simple ring topology where each rank connects only to neighbors `(rank ± 1) % worldSize`. This makes debugging easier with small process counts.

- **Large clusters (worldSize > 4)**: Ring plus random shortcut connections for better connectivity and shorter path lengths.

```cpp
// Ring connections (always)
for (uint32_t i : range(worldSize)) {
  addedge(i, (i + 1) % worldSize);
}

// Random shortcuts (worldSize > 4 only)
if (worldSize > 4) {
  // Add random non-neighbor, non-self connections
}
```

### Routing

Each rank computes shortest paths to all other ranks using BFS. The `firsthop[r]` array stores the first edge to use when routing to rank `r`.

For broadcasts, each edge maintains a `broadcastForwardFor` set - the ranks whose broadcasts should be forwarded through that edge. This creates a spanning tree rooted at each source rank.

## Message Types

Key message types for shutdown:

| Message | Purpose |
|---------|---------|
| `messageShutdown` | Phase 1: Rank signals intent to shut down |
| `messageExit` | Phase 2: Rank is actually exiting |

Other message types include `messageSet`, `messageGet`, `messageWait`, `messageCheck`, `messageDone`, etc.

## Two-Phase Shutdown Protocol

### Problem

Without coordination, when one rank's store is destroyed:
1. It closes TCP connections
2. Other ranks with pending operations (get, wait) see connection errors
3. Users get confusing "connection reset" errors instead of meaningful messages

### Solution: Two-Phase Shutdown

**Phase 1 - Shutdown Signal (`messageShutdown`):**
1. When a rank's store destructor is called, it broadcasts `messageShutdown` to all ranks
2. Each rank that receives `messageShutdown` records the sender in `shutdownStates[rank].shutdown = true`
3. Each rank also triggers its own shutdown and broadcasts `messageShutdown`
4. The initiating rank waits (up to 2 seconds) for all peers to acknowledge shutdown

**Phase 2 - Exit Signal (`messageExit`):**
1. After all ranks have acknowledged shutdown (or timeout), the rank broadcasts `messageExit`
2. Each rank that receives `messageExit` marks `shutdownStates[rank].exited = true`
3. The rank wakes all pending waits with an appropriate error
4. The rank waits (up to 2 seconds) for all peers to acknowledge exit
5. Brief delay (50ms) to allow exit messages to propagate before closing connections

### Graceful vs Unexpected Shutdown

**Graceful shutdown:** All ranks call `shutdown()` (via destructor). The two-phase protocol completes normally, pending waits receive a generic "store was shut down" error.

**Unexpected shutdown:** A rank exits without calling `shutdown()` (crash, early termination). Other ranks receive `messageExit` without prior `messageShutdown`. In this case:
- `exitSourceRank` is set to the offending rank
- Error messages say "store was shut down on rank X" to help debugging

```cpp
if (!shuttingDown) {
  // Peer exited while we haven't started shutdown - unexpected
  if (!exitSourceRank) {
    exitSourceRank = hdr.sourceRank;
  }
}
// Mark ourselves as exited, wake waits, broadcast exit
exited = true;
for (auto& v : waits) {
  exitWait(v.second);
}
broadcastMessage(messageExit);
```

### State Tracking

Per-rank shutdown state:
```cpp
struct ShutdownState {
  bool shutdown = false;  // Phase 1: received messageShutdown from this rank
  bool exited = false;    // Phase 2: received messageExit from this rank
};
Vector<ShutdownState> shutdownStates;  // Indexed by rank
```

Local state:
```cpp
bool shuttingDown = false;           // We've started shutdown
bool exited = false;                 // We've exited
std::optional<uint32_t> exitSourceRank;  // Rank that caused unexpected exit (for error messages)
```

## Broadcast Routing

Messages can be broadcast to all ranks using spanning tree routing. The `MessageHeader` has a `broadcast` flag - when set, each rank automatically forwards the message to its subtree before processing.

**`broadcastMessage(messageType, args...)`** - Initiate a broadcast from this rank:
- Sets `broadcast = true` in the header
- Sends to all direct edges (neighbors in the mesh)
- Variadic template allows passing additional payload after the header

**`forwardBroadcast(sourceRank, buffer, senderEdge)`** - Forward a received broadcast:
- Called automatically in `processMessage` when `hdr.broadcast` is true
- Makes a copy of the original buffer for each forwarding edge
- Patches `destinationRank` in each copy before sending
- Forwards to edges that have `sourceRank` in their `broadcastForwardFor` set
- `senderEdge` prevents sending back to the edge we received from

The spanning tree structure guarantees each rank receives the broadcast exactly once - no deduplication needed.

## Key Distribution

Keys are distributed using consistent hashing:

```cpp
uint32_t storeRank(std::string_view key) {
  return stringHash(key) % worldSize;
}
```

Operations on a key are routed to the owning rank. The `firsthop` table determines the next hop toward any destination.

## Connection Management

- **TCP connections** for reliable message delivery
- **UDP** for diagnostics and discovery
- **Keepalives** detect failed connections
- **Auto-reconnection** re-establishes failed connections

## Diagnostics

When operations timeout, detailed diagnostics are collected:
- Local connection status to the first hop
- Reports from intermediate ranks about their connection states
- Timestamps relative to store creation and timeout

See the API doc for details on diagnostic output format.

## Thread Safety

The store uses a `SpinMutex` for the main data structures. Operations that block (like waiting for shutdown acknowledgments) unlock the mutex during sleep.

## Testing

Shutdown behavior is tested in `tests/test_store.py`:
- `test_store_graceful_shutdown`: Verifies coordinated shutdown without errors
- `test_store_unexpected_shutdown_error`: Verifies unexpected exit produces proper error messages

Note: These tests create stores directly with `moodist.TcpStore()` rather than using `ctx.create_store()` to have full control over store lifetime (avoiding the `_keepalive` reference in the test framework).
