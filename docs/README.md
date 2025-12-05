# Moodist API Documentation

Welcome to the Moodist API documentation. Moodist is a PyTorch extension library that implements a high-performance process group for distributed computing, built on RDMA.

## API Reference

### Core Functions

- **[`compile_op`](api/compile_op.md)** - Compile custom collective operations for arbitrary data movement patterns between ranks
- **[`Queue`](api/queue.md)** - Distributed queue for inter-rank communication of tensors and objects

### Infrastructure

- **[`TcpStore`](api/tcp_store.md)** - Distributed, decentralized key-value store for process group coordination

### Coming Soon

Documentation for additional Moodist features:
- `MoodistProcessGroup` - Process group management
- `serialize` / `deserialize` - Object serialization for distributed communication
- `enable_profiling`, `enable_cuda_allocator`, `enable_cpu_allocator` - Performance tuning
- `cuda_copy` - CUDA tensor operations

## Getting Started

For installation instructions and basic usage, see the main [README](../README.md).

## Examples

All API documentation pages include practical examples. For a comprehensive guide to `compile_op` usage patterns, see the [compile_op examples section](api/compile_op.md#examples).
