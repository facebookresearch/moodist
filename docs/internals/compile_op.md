# compile_op Internals

This document describes how `compile_op` turns a description of input and output tensor regions into a data
movement operation: the compile phase, the generated CUDA kernel, the runtime path, and known issues and planned
work. For the user-facing API, see [api/compile_op.md](../api/compile_op.md).

**Key files:**
- `py/moodist/compile.py` - Python entry point: validates the specs, exchanges them between ranks, calls
  `compile_op_full`
- `core/compile_op.cc`, `core/compile_op.h` - the compile phase: regions, cells, edges, graph variants, kernel
  tuning
- `core/ptx_codegen.cc` - `generateKernel`: the copy kernel of one rank, as PTX
- `core/ptx.h`, `core/ptx.cc` - the PTX DSL (typed registers and operators, scope macros for control flow)
- `core/compile_op_kernel.cc` - PTX JIT (`cuModuleLoadDataEx`)
- `core/processgroup.cc` - `compileOpFull` (sets up the compile context), `customOp` (runs a compiled op)
- `core/cputhread.cc` - `WorkCustom`: the RDMA part of an op, run by the CPU thread

## Data model

`compile_op::Graph` (compile_op.h):

- A **node** is a byte offset in one tensor of one rank, `(rank, tensorIndex, offset)`, with a device and a
  `filled` flag (the data is there when the op starts, as opposed to being written by an edge during the op).
  Node ids are assigned by the rank that owns the node, so a node has the same id on every rank.
- An **edge** copies `bytes` from its source node to one or more destination nodes. `executorRank` is the rank
  whose kernel performs the copy. `cellIndex` identifies the piece of the global tensor.
- Tensor indices on a rank: its inputs, then its input copies (contiguous copies of input pieces that are not
  contiguous in the input), then its outputs, then its output copies.

## Compile phase

Every step is collective: all ranks of the group take part (through per-rank queues and barriers).

1. **Spec exchange** (compile.py): every rank gathers `(rank, dtype, inputs, outputs)` from all ranks, so every
   rank knows every region of every rank.
2. **Phase 1**: each rank intersects its own inputs with every output (same `tensor_id`) and sends the
   intersections to the rank of the output.
3. **Phase 2**, cell decomposition: per `tensor_id`, the boundaries of all regions of all ranks cut each
   dimension into intervals, and a cell is one box of the resulting grid (cells have global indices). For each
   cell of its outputs, a rank looks for the covering inputs: none is a gap error, more than one an overlap error
   unless `reduce="any"` (which picks one at random).
4. **Phase 3**: one read request per cell, to the rank that owns the chosen input.
5. **Phase 4**: that rank resolves the offset. A piece that is not contiguous in the input becomes an input copy
   (deduplicated).
6. **Phase 5**: the output's rank creates one edge per cell, executed by itself: the source is the input (or
   input copy) node, the destination its output node, or an output copy if the piece is not contiguous in the
   output. Then the ranks exchange node ids.
7. **Proxy pass** (`proxyRemoteEdges`; `MOODIST_COMPILE_OP_PROXY=0` disables it): data that several ranks of an
   NVLink domain (`cudaRanks`, see the next step) need from a rank outside the domain should cross the network
   once. The ranks of each domain exchange their edges from outside the domain and group them by source node. In a
   group with two or more destinations (CUDA, sizes and destination offsets multiples of 16 bytes, one destination
   per rank), one destination rank, the proxy, keeps its edge, which will be an RDMA read into its own output, and
   the edges of the other destinations now read from the proxy's node instead, which makes them kernel edges. The
   proxy is preferably the destination with the same local rank as the source (whose NICs are on the same rail),
   otherwise the one that proxies the fewest bytes so far; every member of the domain computes the same
   assignment. The proxy's RDMA edge gets a signal index (`rdmaSignal`), and the proxy gets a relay edge: no
   sources, the proxy's node as destination, the same signal index. The relay edge makes the proxy's kernel wait
   until the CPU thread signals that the read has completed, and then mark the node as filled; the copies from the
   node are ordinary dependent edges.
8. **Kernel / RDMA split**: an edge goes to the kernel if all its nodes are CUDA, all its ranks are kernel peers
   (`cudaRanks`: the ranks whose memory this rank can map, i.e. CUDA IPC peers on the same node, or all ranks of a
   multi-node NVLink fabric domain such as NVL72), and, for process group collectives built on compile_op
   (scatter, gather), its size and offsets are multiples of 16 bytes. Every other edge becomes an `ibRead`, an
   RDMA read by the CPU thread of the destination rank (with the edge's signal index). Kernel tuning and launch
   are collective, so the kernel path is used only if every rank has at least one kernel edge; otherwise all
   edges become ibReads, and the proxy pass is undone (the copies from a proxy's node would race with the proxy's
   own read).
9. **16-byte check**: kernel edges must have sizes and offsets that are multiples of 16 bytes (the kernel uses TMA
   bulk copies). The ranks exchange their first violation and raise together.
10. **Graph variants** (`MOODIST_COMPILE_OP_GRAPHS`, default `multicast`), each a `GraphBuilder` pass over the
   kernel edges:
   - `multicast`: the edges that read the same source node are merged into one edge with several destinations,
     executed by the source rank: it reads each chunk once and writes it to all its destinations.
   - `read` (`buildNothing`): every edge stays with its destination rank, which reads from the source.
   - `ring_read`, `ring_write`: the destinations of a cell form a chain, starting at the source rank, and each
     rank copies the cell from the previous rank's output (read: the receiver executes the copy, write: the
     sender). A proxy's relay edge heads the chain of its cell.

   Edges have unfilled sources, i.e. depend on other edges, in the ring graphs and for proxied transfers (the
   copies from a proxy's node). Relay edges stay with the proxy in every variant.

   `finalize` then sends every edge to every rank that appears in it, sorts the edges into `cudaEdges` and
   `rdmaEdges` (always empty at the moment, because of the split above), and builds the tensor mappings: which of
   its tensors each peer must map, and in which slot of the peer's address table they go.
11. **Tuning** (`tuneKernels`): the copy engine `grouped` (`MOODIST_COMPILE_OP_ENGINES`) x block size {128, 256} x
    shared memory {64 KiB, 226 KiB} x grid size {8, 12, 16}, for every graph variant (12 settings per variant and
    engine). Each setting is generated and JIT
    compiled, then run on zeroed scratch tensors: 12 warmup launches, then 2 timed batches of 12 launches. The
    ranks exchange their timings and take the minimum over ranks per setting. The kernel of the fastest setting
    is generated and compiled once more for the op. Tuning launches run without the CPU thread, so in tuning
    kernels, relay edges do not wait for a signal.

## The kernel (`generateKernel`)

One kernel per op and rank: `compile_op_copy(params {stepValue, concurrencyIndex}, tensorAddrs[], mappedAddrs[])`,
launched in the remote memory synchronization domain. Tensor addresses are kernel arguments, so a compiled op runs
on new tensors at every call.

- **Buffers** (device memory, one slot per concurrency index, mapped by all kernel peers; cached per group and
  reused by later ops): `addresses` (peer tensor addresses, per block), `syncs` (exit sync, one row per block),
  `depFilled` (per node and block).
- **Parity**: every (concurrency index, block) keeps a bit that flips at each launch. Exchanged addresses carry it
  in bit 63, so a reader can tell the current launch's entries from the previous one's.
- **Address exchange**: on entry, each rank writes the addresses at which its peers have mapped its tensors (from
  the IPC mapper, per call) into the peers' `addresses` tables. A rank that needs a remote node's address spins
  until the entry carries the current parity.
- **Copy engines** (`EmitMultiCopy::emit(src, dsts, bytes)`): TMA bulk loads into shared memory (completion via
  mbarrier), then one TMA bulk store per destination, so a multicast edge is read once and written N times. The
  chunks of an edge are assigned round-robin to (buffer, block). Engines: `grouped` (groups of 128 threads sharing
  a double-buffered pipeline; the default), `simple`, `buffered` and `lockstep` (one warp per pipeline; tuned only
  if selected with `MOODIST_COMPILE_OP_ENGINES`). Unaligned addresses or sizes trap.
- **Scheduling**: a rank executes the edges whose executor it is. If all their sources are filled, the copies run
  in sequence. Otherwise a polling loop goes over the edges: thread 0 checks whether the source is ready
  (`depFilled` of the node for this block >= stepValue) and whether the edge is already done, `bar.red` broadcasts
  the decision, the block copies, and thread 0 signals. Dependencies are per block: block b of a consumer waits
  for block b of the producer, which copied the same chunks, as all ranks run the same configuration.
- **Signaling**: after an edge, thread 0 writes stepValue into the `depFilled` slots of its destination nodes on
  its own rank, on the destination ranks, and on the destination ranks of edges that read those nodes. In the
  sequential path, only edges that others depend on are signaled. Before copying from a source that another
  agent wrote during the op (a peer GPU, or a NIC), thread 0 executes `fence.acquire.sys`.
- **Relay edges** (proxied transfers): thread 0 polls the signal of the CPU thread, slot 16 + `rdmaSignal` of
  `cpuOutBuffer` for the concurrency index, until it reaches stepValue, then signals the node as filled like after
  a copy. No bar.red and no copy.
- **Exit**: an all-to-all step value exchange among the kernel peers (`execSync`), so that no rank finishes while
  peers may still read or write its tensors.

## Runtime (`customOp`, `WorkCustom`)

- `customOp` checks the tensors (shape, dtype, device; for user ops also that CUDA tensors come from the moodist
  CUDA allocator and are 16-byte aligned when the op has a kernel), makes the input copies
  (`narrow().contiguous()`) and allocates the output copies (copied into the outputs when the op is waited on).
- Stream order: if any rank needs RDMA, write `cpuIn` = stepValue (the CPU thread may start), then the kernel, then
  wait until `cpuOut` >= stepValue.
- The CPU thread (`WorkCustom`, only if some rank has ibReads) publishes the rank's input list (addresses, sizes,
  rkeys; CUDA tensors are registered on first use) to all ranks, registers the outputs, collects all ranks' input
  lists, waits for `cpuIn`, runs a step value barrier with all ranks (all inputs are ready), issues its ibReads
  (reads of 4 MiB or more split into one part per NIC; shuffled, at most 2 in flight per NIC), waits for them,
  runs a second barrier with all ranks (nobody reads this rank's inputs any more) and writes `cpuOut`.
- The kernel and the CPU thread run concurrently. Besides `cpuIn` and `cpuOut` on the stream, the CPU thread
  signals the kernel when the read of a proxied transfer has completed (all its parts): `writeCpuOut` writes the
  step value into slot 16 + signal index of `cpuOutBuffer` (a loopback RDMA write through the first NIC, as the
  all-gather does for its chunks).

## Environment variables

- `MOODIST_COMPILE_OP_GRAPHS` - the graph variants to tune over (see the compile phase). Same on all ranks.
- `MOODIST_COMPILE_OP_ENGINES` - the copy engines to tune over, comma separated: `grouped` (default), `simple`,
  `buffered`, `lockstep`. Same on all ranks.
- `MOODIST_COMPILE_OP_PROXY=0` - no proxy pass: every rank reads all its data from outside its NVLink domain
  itself. Same on all ranks.
- `MOODIST_FABRIC=0` - ignore a multi-node NVLink fabric: compile_op reaches other nodes over RDMA, as between
  separate NVLink domains. Same on all ranks.
- `MOODIST_DUMP_KERNELS` - write each op's final PTX to `moodist-compile-op-kernels-rank<rank>-op<id>.ptx`.

## Performance notes

All-gather on two GB300 nodes with 4 GPUs each, in one NVL72 domain (October 2026; per-rank size, median time per
call, algorithm bandwidth in parentheses):

| | 1 KiB | 1 MiB | 16 MiB | 256 MiB |
|---|---|---|---|---|
| `multicast` | 16 us | 27 us | 207 us | 2.96 ms (725 GB/s) |
| `read` | 28 us | 48 us | 563 us | 10.7 ms (201 GB/s) |
| `ring_read` | 60 us | 92 us | 313 us | 3.68 ms (584 GB/s) |
| `ring_write` | 53 us | 81 us | 258 us | 3.15 ms (681 GB/s) |
| NCCL (NVLink fabric) | 85 us | 98 us | | 2.91 ms (739 GB/s) |

With `MOODIST_FABRIC=0`, i.e. two NVLink domains connected by RoCE (4 NICs per GPU):

| | 1 KiB | 1 MiB | 16 MiB | 256 MiB |
|---|---|---|---|---|
| `multicast`, proxy pass (one RDMA read per chunk and node, then NVLink) | 74 us | 121 us | 429 us | 6.63 ms (324 GB/s) |
| `ring_write`, proxy pass | 79 us | 121 us | 438 us | 6.84 ms (314 GB/s) |
| `multicast`, `MOODIST_COMPILE_OP_PROXY=0` (every rank reads all chunks of the other node) | 85 us | 211 us | 1.78 ms | 26.5 ms (81 GB/s) |
| the same before reads were split over the NICs | 82 us | 206 us | 2.19 ms | 33.9 ms (63 GB/s) |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 92 us | 102 us | 440 us | 5.42 ms (396 GB/s) |

Compile time grows with the number of edges, because the copy code of each edge is generated inline. All-gather
where each rank's input is K separate 4 KiB pieces (K edges per rank with `multicast`), one node with 4 GPUs:

| K | PTX | compile_op | time per call |
|---|---|---|---|
| 1 | 11 KB | 1.2 s | 11 us |
| 16 | 97 KB | 4.7 s | 22 us |
| 64 | 384 KB | 13 s | 57 us |
| 256 | 1.6 MB | 81 s | 204 us |

Nearly all of the compile time is the JIT of the 12 tuning candidates.

## Known issues and TODO

### Correctness
- Some compile errors are raised only by the rank that finds them (gaps and overlaps in phase 2, device
  mismatches), and the other ranks then wait forever in the next collective step. The 16-byte check shows the
  pattern to follow: exchange the first error and raise on all ranks.
- The kernel path is all or nothing: if some rank has no kernel edge, every edge of every rank goes over RDMA
  (e.g. when a rank receives nothing, or only into CPU tensors). Ranks without kernel edges should take part in
  tuning without a kernel instead.
- Proxied transfers: the CPU thread signals the kernel as soon as the read's completion arrives, with a loopback
  RDMA write through the first NIC. That the read's data is visible in GPU memory before the signal relies on
  timing rather than on PCIe ordering (the read may have gone through another NIC, and MRs allow relaxed
  ordering); the all-gather's chunk signals do the same. A flush (a small RDMA read of the destination through
  the NIC of the read, before the signal) would make it explicit.
- `tuneKernels` combines the ranks' timings with the minimum; the maximum (slowest rank) or the sum fits a
  collective better.
- `computeTarget` returns `sm_100a` for every compute capability from 10 up except 10.3, so devices such as sm_110
  or sm_120 get a target their driver cannot load. The copy kernel does not seem to need arch-specific (`a`)
  instructions, so a plain target would do.

### Performance
- Compile time (see above): the copy code is inlined per edge, and the JIT of 12 candidates per graph variant
  dominates. A table-driven edge loop (edge descriptors in constant or global memory, one copy loop) would make
  the code size independent of the number of edges. JIT of the candidates on several threads would cut the rest
  (the PTX differs per rank, so the work cannot be split across ranks).
- The polling loop does one block-wide `bar.red` per edge and pass; with many edges, a warp-wide scan for the next
  ready edge would be cheaper.
- Proxied transfers are not pipelined: the proxy's kernel starts the NVLink copies only once the whole RDMA read
  has completed (6.6 ms against NCCL's 5.4 ms for the 256 MiB all-gather above). Edges the proxy pass leaves
  alone (CPU destinations, unaligned pieces, one destination per domain) are read by each destination rank.
- `WorkCustom` runs two barriers with all ranks per op, including ranks without RDMA work.
- Pieces that are not contiguous go through separate copies before and after the op.
- After tuning, the ranks log their graphs one after the other, with two barriers per rank.

### Cleanup
- `docs/api/compile_op.md`: the examples predate `TensorRegion`'s `device` field.
- Kernel profiling (`Profiling` in `ptx_codegen.cc`) is compiled out with `if (false)`; an environment variable
  would make it usable without a rebuild.

### Mixed NVLink and RDMA paths: next steps
The proxy pass (compile phase step 7) sends data needed by several ranks of an NVLink domain over the network once.
(Before the edge-based model, `computeNvlinkPlan` in processgroup.cc, removed in c57c8f4, computed a similar plan
with gateway reads and local copies, but it was never executed.) Next:
1. Pipelining: large proxied transfers in chunks, with a signal per chunk checked in the copy engine's chunk loop
   (rather than one edge per chunk, given the per-edge code size), so that the NVLink copies overlap the RDMA read.
2. The write variant: the source writes the data and the signal into the proxy's memory.
3. An explicit flush before the signal (see Correctness).

### Other features
- Unaligned copies (align down, shift the bytes, handle the unaligned tail; needs `fence.proxy.async.shared::cta`
  between generic shared memory writes and TMA stores) or a staging buffer, to lift the 16-byte requirement.
- Reductions (`reduce="sum"`), which need float math in the PTX DSL.
- TMA tensor maps for strided regions, instead of copies of non-contiguous pieces.
