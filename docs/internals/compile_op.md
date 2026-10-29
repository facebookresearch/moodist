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
   proxy is a destination with the same local rank as the source if there is one (whose NICs are on the same
   rail; a domain that spans several nodes has one per node), and among the candidates the one that proxies the
   fewest bytes so far; every member of the domain computes the same assignment. Each domain takes the sources in
   rank order starting after its own first rank, which is also the order in which the CPU thread issues a
   proxy's reads (a few at a time): the domains then read from different sources at the same time instead of all
   from the same ones first. The proxy's RDMA edge gets a signal index (`rdmaSignal`), and the proxy gets a relay edge: no
   sources, the proxy's node as destination, the same signal index. The relay edge makes the proxy's kernel wait
   until the CPU thread signals that the read has completed, and then mark the node as filled; the copies from the
   node are ordinary dependent edges. Transfers of 8 MiB or more are split into chunks of at least 4 MiB (at most
   32): each chunk has its own read, signal, relay edge and copies (`chunkIndex`; the chunk nodes get ids that
   their owner never assigns, numbered in the same order by every member of the domain), so that the copies of a
   chunk start as soon as it has arrived.
8. **Kernel / RDMA split**: an edge goes to the kernel if all its nodes are CUDA, all its ranks are kernel peers
   (`cudaRanks`: the ranks whose memory this rank can map, i.e. CUDA IPC peers on the same node, or all ranks of a
   multi-node NVLink fabric domain such as NVL72), and, for process group collectives built on compile_op
   (scatter, gather), its size and offsets are multiples of 16 bytes. Every other edge becomes an `ibRead`, an
   RDMA read by the CPU thread of the destination rank (with the edge's signal index). Kernel tuning and launch
   are collective, so the kernel path is used only if every rank has at least one kernel edge; otherwise all
   edges become ibReads, and the proxy pass is undone (the copies from a proxy's node would race with the proxy's
   own read). The ibReads are sorted by source so that each source has about one reader at a time: in the same
   exchange, the ranks learn each other's domain (its first rank), and the rank at position p of domain i reads
   the domains after its own in turn, each from position p (mod the domain's size) on, i.e. (domain i + 1 + a,
   position p + b) in step (a, b); reads from its own domain come last. (A rotation by rank, (source - rank) mod
   size, does not work: the ranks of a domain skip its own ranks, so they all start on the same source.) In the
   next exchange (whether any rank needs RDMA), each rank also describes its reads without a signal: whether it
   reads from every rank outside its domain (or from every other rank), and the least and the most it reads from
   one rank. If every rank with such reads does, with amounts within 1/16 of each other, the steps all take equally
   long and a rank reads one source at a time (`readSources` 1); otherwise 8 at a time (see the RDMA read notes
   under Performance).
9. **16-byte check**: kernel edges must have sizes and offsets that are multiples of 16 bytes (the kernel uses TMA
   bulk copies). The ranks exchange their first violation and raise together.
10. **Graph variants** (`MOODIST_COMPILE_OP_GRAPHS`, default `multicast`), each a `GraphBuilder` pass over the
   kernel edges:
   - `multicast`: the edges that read the same source node are merged into one edge with several destinations,
     executed by the source rank: it reads each chunk once and writes it to all its destinations.
   - `read` (`buildNothing`): every edge stays with its destination rank, which reads from the source.
   - `ring_read`, `ring_write`: the destinations of a cell form a chain, starting at the source rank, and each
     rank copies the cell from the previous rank's output (read: the receiver executes the copy, write: the
     sender). A proxy's relay edge heads the chain of its cell (of each chunk, for chunked transfers).

   Edges have unfilled sources, i.e. depend on other edges, in the ring graphs and for proxied transfers (the
   copies from a proxy's node). Relay edges stay with the proxy in every variant.

   `finalize` then sends every edge to every rank that appears in it, sorts the edges into `cudaEdges` and
   `rdmaEdges` (always empty at the moment, because of the split above), and builds the tensor mappings: which of
   its tensors each peer must map, and in which slot of the peer's address table they go.
11. **Tuning** (`tuneKernels`): the grid size {4, 8, 12, 16} (blocks of 128 threads with one pipeline of 32 KiB
    chunks in 226 KiB of shared memory), for every graph variant (4 settings per variant). Each setting is
    generated and JIT compiled, then run on zeroed scratch tensors: 12 warmup launches, then 2 timed batches of 12
    launches. The ranks exchange their timings and take the minimum over ranks per setting. The fastest setting
    wins, with a penalty of 0.5% per block: the kernel occupies its SMs, which the computation it overlaps with
    lacks meanwhile, so a bigger grid has to be clearly faster to be chosen (as it is for copies limited by the
    SMs' bandwidth). The kernel of the chosen setting is generated and compiled once more for the op. Tuning
    launches run without the CPU thread, so in tuning kernels, relay edges do not wait for a signal.

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
- **Copy engine** (`PipeRing`): each block runs pipelines (the first lane of each of its first warps; one by
  default), each with a ring of stages of 32 KiB in shared memory. A chunk is copied by a TMA bulk load into a
  stage (completion via mbarrier), then one TMA bulk store of the stage per destination, so a multicast edge is
  read once and written N times. A TMA copy is latency bound: an SM reaches its NVLink bandwidth (about 52 GB/s for
  one store per load, up to 65 GB/s for many, on GB300) only with about 100 KB of loads in flight. So a pipeline
  keeps up to (stages - 2) loads in flight ahead of its stores: after issuing the stores of chunk k, it waits until
  those of chunk k - 2 have read their stage (`cp.async.bulk.wait_group.read 2`) and loads the next chunk there.
  The copy rates of the SMs vary (by up to half under load), hence the dynamic distribution below. Edges and
  addresses must be 16-byte aligned (checked when the op is compiled).
- **Tables**: the edges a rank executes are described by tables in one initialized global array of the module:
  edge setups (flags, source, bytes, destinations, signals), node setups (tensor slot and offset), signal setups
  (`depFilled` slots to write when an edge is done) and the address exchange setups. The kernel loops over them at
  runtime, so its code size does not depend on the number of edges (edges are limited to 4 GiB each).
- **Scheduling**: a rank executes the edges whose executor it is. If no edge waits (all sources filled, no relay
  edges), every block first resolves the addresses of all edges into a table in shared memory (in global memory if
  it exceeds 32 KiB), and the pipelines of all blocks then take items from one sequence: an item is a chunk of an
  edge and a group of its destinations (with few chunks, an edge's destinations are split into groups, for about
  4 items per pipeline, so that the stores of an all-gather of a few chunks to many ranks spread over the SMs).
  Items are claimed from a counter in global memory (one per concurrency slot and launch parity), the first claim
  of a pipeline taking enough items to fill its ring, the later ones one item, claimed when the pipeline starts on
  the previous one. Edges are in destination rank order rotated to start at the executing rank (so that in an
  all-to-all, the ranks do not all write to the same rank at once). If some edge waits, in rounds: warp 0 scans the
  edges that are not done, 32 at a time (one per lane, `vote.ballot`); a source is ready when its `depFilled` slot
  for this block has reached stepValue. Relay edges whose signal has arrived are finished on the spot by the lane
  that saw it. The first ready copy edge is handed to the block through shared memory (one `bar.sync` per round);
  the block copies its chunks c with c % grid size = block index (the pipelines of the block taking them in turn)
  and thread 0 marks it done (done bits in shared memory). Relay edges come first in the table, so that the scan
  always reaches them. Dependencies are per block: block b of a consumer waits for block b of the producer, which
  copied the same chunks, as all ranks run the same configuration.
- **Signaling**: after an edge that some edge reads (known from the peers' lists of read nodes, as the reader may
  be an edge this rank does not see), the thread that saw it done (thread 0 after a copy) writes stepValue into
  the `depFilled` slots of its destination nodes on its own rank, on the destination ranks, and on the destination
  ranks of edges that read those nodes, after `fence.release.sys`. Without waits, any block may copy any item of
  an edge: the pipeline that completes the edge's last item (a per-edge counter that wraps around after every
  launch) writes the slots of all blocks. The lane that sees a dependency met executes
  `fence.acquire.sys` before the selection barrier; before copying from a peer's tensor, every thread does.
- **Relay edges** (proxied transfers): ready once the CPU thread's signal, slot 16 + `rdmaSignal` of
  `cpuOutBuffer` for the concurrency index, has reached stepValue (always ready when tuning); nothing to copy.
- **Exit**: an all-to-all step value exchange among the kernel peers (`execSync`), so that no rank finishes while
  peers may still read or write its tensors.
- **NVLS** (NVLink SHARP multicast; groups of at least 16 ranks in one NVLink domain, ops not behind a process group
  collective): an edge to every rank of the group, at the same tensor and offset on each (an all-gather's), can
  store each chunk once to the group's multicast mapping, which the switches copy to every GPU. The group's
  multicast object (`Group::setupNvls`, collective, at the first op that uses it) covers moodist's whole allocator
  pool: every allocator chunk is bound at its offset in the pool when it is mapped (`allocatorAddChunkListener`), so
  offset o of every rank's pool is at the mapping's base + o; a counter area follows. A rank uses NVLS if every edge
  it executes is such an edge (no waits) and every rank that does stores the same number of chunks (agreed in
  `generateKernel`). Each launch checks the destinations: a peer marks the address it publishes with bit 62
  (`nvlsPoolBit`) if the tensor is in its pool, and the low 40 bits are then the tensor's pool offset (peers map a
  rank's pool at a 1 TiB aligned base); if every destination is at the local destination's offset, block 0 stores
  the chunks with one multicast store each (chunks of 64 KiB, through a ring of its own over the stage memory: per
  chunk, the window and the counter cost time) and the other blocks only take part in the address exchange and the
  exit sync; otherwise the launch stores to each destination as usual. Multicast stores at full speed from all ranks
  overload the switches' links to the GPUs, which then deliver about 2/3 of their bandwidth, so the ranks advance in
  step: an op has a step counter per concurrency slot in the counter area; once the stores of a chunk have completed,
  a rank adds 1 to it through the multicast mapping (`multimem.red`, relaxed: a release waits for all outstanding
  stores), and it stores chunk c once its local copy of the counter shows every rank's chunks up to c - 8 complete.
  The counter only grows; `customOp` passes its value before the launch (launches of the slot times the chunks of
  all ranks); a launch that does not use NVLS adds its chunks at the end.

## Runtime (`customOp`, `WorkCustom`)

- `customOp` checks the tensors (shape, dtype, device; for user ops also that CUDA tensors come from the moodist
  CUDA allocator and are 16-byte aligned when the op has a kernel), makes the input copies
  (`narrow().contiguous()`) and allocates the output copies (copied into the outputs when the op is waited on).
- Stream order: if any rank needs RDMA, write `cpuIn` = stepValue (the CPU thread may start), then the kernel, then
  wait until `cpuOut` >= stepValue.
- The op runs on the caller's current CUDA stream (no internal stream). Everything enqueued after it on that stream
  sees the complete op, RDMA part included, so ops without CPU tensors are done for the caller when the call
  returns. Each op uses one of 16 concurrency slots (its own `cpuIn`, `cpuOut` and step value exchange areas): on
  the stream, the whole op waits for the slot's event, which the slot's previous op (possibly on another stream)
  recorded when it was done, and records it when done itself. `wait()` (or dropping the future) makes the current
  stream wait for that event if it is not the op's stream (if the slot has been reused since, for the later op,
  which only waits longer), then makes the output copies. Ops with CPU tensors, or compiled with `cpu_sync=True`,
  block in `wait()` until the CPU thread is done. (moodist's process group collectives also run on the current
  stream, but their `Work.wait()` does nothing, so they do not join another stream.) Between April and October
  2026 (95a48ec) `wait()` blocked for every op with an RDMA part; see the latency noise note under Known issues.
- The CPU thread (`WorkCustom`, only if some rank has ibReads) publishes the rank's input list (addresses, sizes,
  rkeys; CUDA tensors are registered on first use) to all ranks, registers the outputs, collects all ranks' input
  lists, waits for `cpuIn`, runs a step value barrier with all ranks (all inputs are ready), issues its ibReads
  (split into parts of about 128 KiB, and if the op has fewer reads than NICs, smaller reads too, into parts of at
  least 64 KiB; reads with a signal first, in signal order, i.e. the chunks of a transfer in order, then the
  others in the order of the compile phase, `readSources` at a time: with more than one, the parts of that many
  reads go round robin, and a read whose parts have all been issued makes room for the next; each part on the NIC
  with the fewest in flight, up to 8), waits for them, runs a second barrier with all ranks (nobody reads this
  rank's inputs any more) and writes `cpuOut`. The parts of a read with a signal share it: the signal is written
  when the last part has completed. Small parts keep a rank on one or two sources at a time (with `readSources`
  1), which with the read order keeps the ranks in step (see the RDMA read notes under Performance).
- The kernel and the CPU thread run concurrently. Besides `cpuIn` and `cpuOut` on the stream, the CPU thread
  signals the kernel when the read of a proxied transfer has completed (all its parts): `writeCpuOut` writes the
  step value into slot 16 + signal index of `cpuOutBuffer` (a loopback RDMA write through the first NIC, as the
  all-gather does for its chunks).

- NVLS ops (see the kernel): `customOp` sets bit 62 of the addresses it publishes for tensors in moodist's pool
  (when the peer's mapping of the pool is 1 TiB aligned), and passes the op's step counter value.

## Environment variables

- `MOODIST_COMPILE_OP_GRAPHS` - the graph variants to tune over (see the compile phase). Same on all ranks.
- `MOODIST_COMPILE_OP_PROXY=0` - no proxy pass: every rank reads all its data from outside its NVLink domain
  itself. Same on all ranks.
- `MOODIST_FABRIC=0` - ignore a multi-node NVLink fabric: compile_op reaches other nodes over RDMA, as between
  separate NVLink domains. Same on all ranks.
- `MOODIST_IB_DATA_DIRECT=0` - ignore the Data Direct DMA functions of ConnectX-8 NICs (not specific to compile_op;
  see the RDMA bandwidth note under Performance).
- `MOODIST_DUMP_KERNELS` - write each op's final PTX to `moodist-compile-op-kernels-rank<rank>-op<id>.ptx`.

## Tracing

`moodist.enable_profiling(True)` makes the CPU thread record Chrome trace spans; after
`moodist.enable_profiling(False)`, its next traced point writes them to `moodist-trace-cpu-<group>-<rank>.json` in
the current directory (so run one more op). For compile_op the spans are `WorkCustom`'s phases, named with the op's
step value: `start`, `wait dyn N` (waiting for rank N's input list), `wait kernel` (for the stream to reach the
op), `step exchange`, `wait step N`, `reads`, `wait read completions` and `wait reads of N` (rank N still reading),
where only the ranks actually waited for appear; and each NIC's reads (`read N`, from rank N). The CPU thread also
records how long each op waited from the call until the CPU thread picked it up, and its sleeps, stalls of 20 us
or more and moves between cores, with the core numbers. Timestamps are the system clock, so the files of all
ranks can be merged into one trace (align the nodes' clocks, e.g. on the end of `step exchange`, which all ranks
leave within a few microseconds).

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

With `MOODIST_FABRIC=0`, i.e. two NVLink domains connected by RoCE (one ConnectX-8 card with 4 ports of 200 Gb/s
per GPU):

| | 1 KiB | 1 MiB | 16 MiB | 256 MiB |
|---|---|---|---|---|
| `multicast`, proxy pass (each rank's data read once per node, then NVLink) | 35 us | 85 us | 233 us | 2.89 ms (742 GB/s) |
| `multicast`, `MOODIST_COMPILE_OP_PROXY=0` (every rank reads all chunks of the other node) | 44 us | 171 us | 1.40 ms | 21.3 ms (101 GB/s) |
| moodist's all-gather (process group) | 40 us | 68 us | 264 us | 3.51 ms (613 GB/s) |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 95 us | 101 us | 441 us | 5.42 ms (396 GB/s) |

(The 1 MiB column above predates the splitting of small reads over the NICs: each proxy reads one 1 MiB piece,
which then went through a single NIC, a 200 Gb/s port. Measured on other nodes of the same type, splitting it took
1 MiB from 82 to 52 us, 512 KiB from 58 to 45 us and 2 MiB from 131 to 68 us; with four domains (three reads per
proxy) 1 MiB from 101 to 94 us and 2 MiB from 160 to 143 us.)

One GPU per node (each GPU receives one piece over the network): compile_op 87, 93 and 95 GB/s per GPU at 64 MiB,
256 MiB and 1 GiB per rank, moodist's all-gather 90, 94 and 95 GB/s, NCCL 73, 81 and 84 GB/s.

Two nodes in different NVL72 domains (in different racks): compile_op 42 us, 92 us, 241 us and 2.90 ms (740 GB/s),
moodist's all-gather 40 us, 79 us, 277 us and 3.51 ms, NCCL 81 us, 123 us, 481 us and 5.63 ms (381 GB/s).

Earlier states (two nodes, `MOODIST_FABRIC=0`). The first row is with `wait()` blocking until the CPU thread is done
(see the latency noise note below); the others are also measured with the large buffers off page boundaries (see
the RDMA notes under Performance) and, except in the second row, without Data Direct:

| | 1 KiB | 1 MiB | 16 MiB | 256 MiB |
|---|---|---|---|---|
| `multicast`, proxy pass, `wait()` blocking | 75 us | 119 us | 259 us | 2.91 ms (738 GB/s) |
| `multicast`, proxy pass, with Data Direct | 73 us | 117 us | 266 us | 4.66 ms (461 GB/s) |
| `multicast`, proxy pass | 76 us | 119 us | 388 us | 5.12 ms (419 GB/s) |
| `read`, proxy pass | 76 us | 122 us | 386 us | 5.17 ms (416 GB/s) |
| `ring_write`, proxy pass | 77 us | 126 us | 387 us | 5.20 ms (413 GB/s) |
| `multicast`, proxy pass without chunks (each transfer one read) | 74 us | 121 us | 429 us | 6.63 ms (324 GB/s) |
| `multicast`, `MOODIST_COMPILE_OP_PROXY=0` | 85 us | 211 us | 1.78 ms | 26.5 ms (81 GB/s) |
| the same before reads were split over the NICs | 82 us | 206 us | 2.19 ms | 33.9 ms (63 GB/s) |
| moodist's all-gather (process group) | 38 us | 70 us | | 5.51 ms (390 GB/s) |

(NCCL without Data Direct, `NCCL_IB_DATA_DIRECT=0`: 8.53 ms at 256 MiB.)

16 GPUs: four nodes in two NVL72 domains of two nodes each (in different racks):

| | 1 KiB | 1 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|
| `multicast`, proxy pass | 46 us | 101 us | 410 us | 1.54 ms | 5.95 ms (722 GB/s) |
| `multicast`, `MOODIST_COMPILE_OP_PROXY=0` | 48 us | 267 us | 2.59 ms | 10.4 ms | 41.2 ms (104 GB/s) |
| moodist's all-gather (process group) | 57 us | 147 us | 692 us | 2.40 ms | 9.21 ms (466 GB/s) |
| NCCL | 91 us | 220 us | 644 us | 1.64 ms | 6.18 ms (695 GB/s) |

With `MOODIST_FABRIC=0`, i.e. four domains of one node:

| | 1 KiB | 1 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|
| `multicast`, proxy pass | 46 us | 107 us | 597 us | 2.18 ms | 9.67 ms (444 GB/s) |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 97 us | 225 us | 1.01 ms | 2.91 ms | 11.1 ms (386 GB/s) |

With two domains of two nodes, the large sizes are bound by NVLink (each GPU receives 14 of the 16 pieces over
NVLink); with four domains, by the proxies' RDMA reads (each proxy reads 3 pieces, at 85-95 GB/s). With `wait()`
blocking until the CPU thread was done (see the latency noise note under Known issues), 1 KiB took 87-89 us in runs
without tails, 1 MiB 135-136 us and 16 MiB 412 us (two domains) and 653 us (four domains). Earlier states at
256 MiB (large buffers off page boundaries, without Data Direct): two domains 6.07 ms, or 11.4 ms with the first
destination of the source's local rank as proxy (one node of each domain proxied everything); four domains
17.4 ms, or 28.7 ms with the same read order in every domain; moodist's all-gather 15.2 ms (11.6 ms with Data
Direct). The tables above predate the RDMA read order and 128 KiB parts (see below).

32 GPUs: eight nodes in one NVL72 domain, with `MOODIST_FABRIC=0`, i.e. eight domains of one node over RoCE. For
all-to-all the size is what each rank sends in total (a 32nd to each rank):

| all-to-all | 4 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|---|---|---|
| compile_op | 68 us | 71 us | 104 us | 259 us | 771 us | 2.57 ms | 10.2 ms |
| compile_op before the read order and 128 KiB parts | 83 us | 95 us | 143 us | 365 us | 1.28 ms | 4.16 ms | 16.6 ms |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 139 us | 137 us | 163 us | 282 us | 932 us | 2.93 ms | 11.2 ms |

| all-gather | 1 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|---|
| `multicast`, proxy pass | 56 us | 155 us | 403 us | 1.31 ms | 5.09 ms | 20.3 ms |
| the same before | 55 us | 173 us | 518 us | 1.40 ms | 5.01 ms | 22.6 ms |
| `MOODIST_COMPILE_OP_PROXY=0` | 68 us | 442 us | 1.38 ms | 5.09 ms | 20.5 ms | 84.7 ms |
| the same before | 83 us | 669 us | 2.13 ms | 8.28 ms | 33.6 ms | 135 ms |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 102 us | 364 us | 661 us | 1.86 ms | 5.66 ms | 22.2 ms |

At 1 GiB, the all-to-all reads 92 GB/s per GPU (of 100), NCCL moves 84. Over the NVLink fabric (one domain, no
RDMA), all-to-all takes 22-23 us up to 1 MiB, 44.5 us at 16 MiB, 117 us at 64 MiB, 404 us at 256 MiB and 1.53 ms
at 1 GiB (NCCL 118-236 us up to 1 MiB, then 165 us, 142 us, 434 us and 1.63 ms).

All-gather over the NVLink fabric with NVLS (see the kernel: one SM per GPU stores each chunk once, with
multicast) and without it (unicast stores from 12-16 SMs, the state before NVLS), outputs at the same pool offset
on every rank:

| all-gather | 64 KiB | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|---|---|
| 32 GPUs (8 nodes, one NVL72 domain), NVLS | 19.6 us | 27.4 us | 62.1 us | 208 us | 789 us | 3.10 ms | 12.40 ms |
| 32 GPUs, unicast | 19.9 us | 31.5 us | 72.9 us | 239 us | 826 us | 3.20 ms | 12.69 ms |
| 32 GPUs, NVLS with 32 KiB chunks | 17.8 us | 28.8 us | 63.9 us | 215 us | 811 us | 3.19 ms | 12.73 ms |
| 16 GPUs (4 nodes), NVLS | 17.8 us | 20.4 us | 38.6 us | 110 us | 405 us | 1.56 ms | 6.19 ms |
| 16 GPUs, unicast | 16.8 us | 23.6 us | 43.5 us | 126 us | 408 us | 1.54 ms | 6.11 ms |

Up to 128 KiB, runs differ by 1-2 us, more than NVLS changes. Large sizes are bound by what the GPUs' NVLink links
deliver to them, and the switch also delivers each multicast chunk to its sender: a GPU receives N/(N-1) times
what unicast stores bring it, 3.2% more on 32 ranks and 6.7% more on 16, where NVLS is slower from 64 MiB on.
Unicast stores are faster when the ranks' outputs are at different offsets: on 32 GPUs, 256 MiB took 12.22 ms with
the outputs at offsets 64-192 KiB apart (NVLS needs them at the same offset: 12.38 ms in the same run), and 16 MiB
814 us. Without the step counter, multicast stores from 32 ranks took 17-19 ms at 256 MiB (see the kernel). On 4
GPUs, and on 64 GPUs in two NVL72 domains (29.4 ms at 256 MiB), ops do not use NVLS and take what they took before.

RDMA reads, measured with per-read logs (issue and completion time of every read on every rank, merged) on the
1 GiB all-to-all above. The reads used to be split into one part per NIC (from 4 MiB on), with 2 in flight per NIC,
in random order: 16.6 ms. In the order of the compile phase (each source one reader per step), 14.8 ms: each NIC
still read from two sources at a time, the reads finished at different rates, and the ranks drifted apart until a
source port had no reader 13% of the time and three or more readers 30% of the time. With one read in flight per
NIC, the ranks stayed in step (one reader per source port 97% of the time, 23 GB/s per 200 Gb/s port) and took
11.6 ms, but every read cost a round trip (about 13 us from post to completion besides the transfer). 128 KiB parts
with 8 in flight per NIC (1 MiB) keep a rank on one or two sources at a time and hide the round trip: 10.2 ms, or
12.2 ms in random order. Other part sizes and counts with 768 KiB to 1.5 MiB in flight were within 1-7% at 1 GiB,
but larger parts were slower at 16-64 MiB (fewer parts per source), and smaller ones at 1 GiB (64 KiB with 12 in
flight: 12.1 ms). The CPU thread posts a read in about 300 ns: at 128 KiB and 92 GB/s per GPU, 0.7 reads per us,
which keeps it about 45% busy (its loop took 185 ns per iteration instead of about 100). The small parts load the
network less: per 150 calls of the 1 GiB all-to-all, 2.1 million ECN-marked packets (144 million with the large
parts in random order, 108 million with NCCL) and 0.6 million CNPs (9.6 and 12.4 million). The NIC already split
the large reads into 256 KiB requests, so the read requests on the wire only doubled.

With unequal amounts per pair of ranks, one source at a time leaves sources idle: readers pile up on the larger
transfers. All-to-all-v on the same 32 GPUs, 256 MiB per rank on average, each pair of ranks its own range of the
index space; the bound is the most any rank sends or receives across nodes, at 92 GB/s:

| sizes per pair | bound | before the read order | one source at a time | 8 at a time | NCCL |
|---|---|---|---|---|---|
| equal | 2.55 ms | 4.16 ms | 2.59 ms | 3.01 ms | 2.93 ms |
| lognormal, sigma 1 | 4.10 ms | 6.12 ms | 6.58 ms | 5.05 ms | 4.39 ms |
| lognormal, sigma 2 | 7.40 ms | 12.1 ms | 9.02 ms | 8.17 ms | 8.00 ms |
| one source sends 4 times as much | 10.2 ms | 11.4 ms | 10.8 ms | 10.2 ms | 4.46 ms |
| one destination receives 4 times as much | 10.2 ms | 11.9 ms | 13.1 ms | 10.7 ms | 10.9 ms |

compile_op reads one source at a time only when all ranks read equal amounts from all remote ranks (see the compile
phase), otherwise 8 at a time: 4 at a time was up to 4% slower, all at once no faster and less steady (with many
sources per reader port, congestion control steps in: 5-7 times as many ECN-marked packets). A fluid model of the
reads (max-min fair rates over the GPUs' send and receive bandwidth) agrees: one source at a time 1.68 and 1.27
times the bound for the two lognormal cases, 4 or more at a time within 3%. NCCL also sends a GPU's data through the
NICs of the other GPUs of its node (PXN), which is why it beats the per-GPU bound when one source sends more than the
others (over the node's NICs, the bound is 4.47 ms there).

Compile time does not depend on the number of edges: the kernel reads its edges from tables. All-gather where each
rank's input is K separate 4 KiB pieces (K edges per rank with `multicast`), one node with 4 GPUs:

| K | PTX | compile_op | time per call |
|---|---|---|---|
| 1 | 15 KB | 1.4 s | 12 us |
| 16 | 21 KB | 1.5 s | 24 us |
| 64 | 40 KB | 1.3 s | 65 us |
| 256 | 121 KB | 1.7 s | 229 us |

Most of the PTX of large ops is the tables (initializers in decimal). Nearly all of the compile time was the JIT of
the 12 tuning candidates. The kernel that generated the copy code of each edge inline took 1.2 s, 4.7 s, 13 s and
81 s (PTX of 11 KB to 1.6 MB), and was about 12% faster per edge (11, 22, 57 and 204 us).

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
- The kernel takes the tensor addresses (and the addresses at which peers map them) as kernel parameters, which are
  limited to 32 KB: an op fails to JIT with about 4000 (tensor, peer) mappings (e.g. 1024 input tensors on 4
  ranks). Passing them in a device buffer would lift the limit.
- Kernel edges must be smaller than 4 GiB (32-bit sizes in the kernel's tables; checked). Splitting larger pieces
  into several edges when the edges are created would lift the limit.
- `computeTarget` returns `sm_100a` for every compute capability from 10 up except 10.3, so devices such as sm_110
  or sm_120 get a target their driver cannot load. The copy kernel does not seem to need arch-specific (`a`)
  instructions, so a plain target would do.

### Performance
- NVLS (see the kernel and the all-gather table under Performance notes). Open: the rank threshold (16; one node of
  4 GPUs lost from 4M on, 8 and 12 are not measured; 16 ranks lose 1-2% from 64 MiB on), groups that span several
  NVLink domains (one multicast object per domain, for the edges within a domain, including the proxies' stores of
  what they read: 64 GPUs in two domains take 29.4 ms at 256 MiB, where the GPUs' NVLink ingress allows about 25),
  destination sets smaller than the group (an object per set), ranks that store different numbers of chunks (the
  window assumes the same), outputs that are not at the same offset on every rank (a pool for communication
  buffers), and multicast reductions (`multimem.ld_reduce`, `multimem.red`) for reduce-scatter and all-reduce.
- Unicast stores to outputs at the same offset on every rank are 2-4% slower at large sizes than to outputs at
  different offsets (all-gather on 32 GPUs, see the NVLS table), maybe because all copies of a chunk then have the
  same address bits; not investigated.
- Compile time is mostly the JIT of the tuning candidates (4 per graph variant). JIT on several threads would cut
  it (the PTX differs per rank, so the work cannot be split across ranks).
- With waits (relay edges, ring graphs), the block copies one edge at a time and drains its pipelines at the end
  of each edge, and a dependency is per block and edge. Per-chunk ready flags would let the pipelines take chunks
  across edges as without waits.
- The fixed cost of a launch is about 6 us on 4 GPUs (64 KiB shift): of it, `fence.release.sys` before the exit
  step value exchange takes 2-3 us (any system-scope fence takes about 1.8 us on GB300 even with nothing
  outstanding, a GPU-scope one 0.3 us; under load it waits for the copies of the other blocks), the address
  exchange and resolution about 1.5 us, reading the launch parity 0.5 us. Back-to-back launches of an empty kernel
  take 2 us each.
- Edges the proxy pass leaves alone (CPU destinations, unaligned pieces, one destination per domain) are read by
  each destination rank.
- Regions of different sizes in one index space fragment the op: phase 2 cuts each dimension at the boundaries of
  all regions of all ranks, and neighboring cells are not merged into one edge. An all-to-all-v laid out as
  [source, destination, max size], with a different size per pair, cuts every pair at the sizes of all others:
  on 32 GPUs about 13,000 RDMA reads of a few KiB per rank and call at 256 MiB, which made the CPU thread the
  bottleneck (2.4 times the bound instead of 1.2-1.6). Merging cells that are contiguous in both the input and the
  output would avoid it; until then, such ops should give each piece its own range of the index space.
- A GPU's RDMA traffic goes through its own NICs only. NCCL (PXN) also routes a GPU's data through the NICs of the
  other GPUs of its node, over NVLink first, and was 2.3 times faster when one source sent 4 times as much as the
  others (see the all-to-all-v table). compile_op would need a staging copy in another GPU's memory.
- `WorkCustom` runs two barriers with all ranks per op, including ranks without RDMA work.
- RDMA bandwidth on GB300 (one ConnectX-8 card with 4 ports of 200 Gb/s per GPU, i.e. 100 GB/s): the card's PCIe
  link to the CPU is Gen5 x16, and without Data Direct the NICs reach GPU memory through it, at about 52 GB/s per GPU.
  Each card also has a Data Direct DMA function behind its own PCIe switch, to which one GPU is attached (Gen6 x16).
  moodist uses it like NCCL (`NCCL_IB_DATA_DIRECT`): the NIC selection pairs each GPU with the card whose function is
  behind the GPU's switch (`dataDirectPath` in group.cc), and `RdmaIb::regMrCuda` registers that GPU's memory with
  `mlx5dv_reg_dmabuf_mr(..., MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT)` on a dmabuf mapped through PCIe
  (`CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE`; the default mapping goes through the CPU, which the function
  reaches only through its switch's x1 link: 0.4 GB/s, as with the wrong card). With one GPU per node, all-gather
  then reaches 95 GB/s per GPU from 256 MiB per rank (NCCL: 84 GB/s).
- RDMA reads into a destination that does not start at a page boundary are much slower: packets are cut from the start
  of the read, so each one straddles two pages of the destination (GB300, 1 GiB read: 11.3 ms with the destination at
  a page boundary, 14.3 ms with it 1 KiB into a page, 16.5 ms with the source 1 KiB into a page too; the source alone
  off a boundary costs nothing). `readData` reads the start of reads of 64 KiB or more up to the next page boundary
  separately, and the CUDA allocator starts allocations of 64 KiB or more at a page boundary. Before both, any small
  tensor allocated earlier (e.g. a benchmark's 1 KiB step) shifted all later buffers off page boundaries, which
  capped reads at about 64 GB/s per GPU (or 16 GB/s per queue pair: a second queue pair per NIC hid it). RDMA writes
  (scattered by the remote NIC into its memory) probably have the same problem; not measured.
- With reads split into one part per NIC (before the 128 KiB parts), more reads in flight per NIC made the proxied
  reads slower without Data Direct (2: 17.1 ms, 4: 19.9 ms, 8: 21.9 ms at 256 MiB with four domains): each NIC then
  read from several sources at a time (see the RDMA read note). With 128 KiB parts, 8 in flight are faster than the
  old 2 without Data Direct too (32 GPUs, `MOODIST_IB_DATA_DIRECT=0`, 256 MiB per rank: all-gather 34.1 ms
  instead of 38.1, all-to-all 4.30 ms instead of 5.09; NCCL 4.67 ms).
- Latency noise (fixed in October 2026): from 95a48ec on, `wait()` (and dropping the future) blocked until the CPU
  thread had finished every op with an RDMA part, although the stream already waits for it. The caller could not
  run ahead, so after each op the CPU thread ran out of work and slept, and every op had to wake it up on every
  rank. On the GB300 nodes a wake-up occasionally left the thread queued for 0.3-3 ms (the kernel's per-thread
  run-queue wait: 90-170 ms per 300 ops in noisy runs, about 0 in clean ones), behind another thread on the core
  it had slept on: the CPU thread of another rank of the same NUMA node, or one of the processes outside the job
  (16 cores of each node were busy with them). With 16 GPUs some rank was late on most ops, and a run tended to
  stay noisy, as the threads kept sleeping on the same cores. 1 KiB all-gather, 16 GPUs: median 87 us when clean,
  otherwise up to about 600 us with a p90 of 1-2 ms (p90 over median 2.8 over the runs of one day, with p90 over
  1.5 x median in 47 of 79 runs; NCCL and moodist's all-gather 1.1 and 1.04, never over 1.5). Pinning the Python
  thread and the CPU thread to cores of their own also removed it (78-80 us), pinning either alone did not; Python
  garbage collection and the order of the reads played no part. Now the op is done for the caller at the call
  (see Runtime), so the CPU thread has the next ops queued while the GPU works: 46 us with a p90 of 52 us. Traces
  (see Tracing) show the phases of each op and which rank the others waited for.
- The CPU thread still sleeps when it has nothing to do. A caller that waits for each op on the CPU (ops with CPU
  tensors or `cpu_sync`, or a `torch.cuda.synchronize()` per op) wakes it for every op, with the same occasional
  multi-millisecond wake-ups (moodist's all-gather in such a loop: 300-700 us instead of 58 us on 16 GPUs). Spinning
  for about 200 us before sleeping removed them in that test.
- Pieces that are not contiguous go through separate copies before and after the op.
- After tuning, the ranks log their graphs one after the other, with two barriers per rank.

### Cleanup
- `docs/api/compile_op.md`: the examples predate `TensorRegion`'s `device` field.
- Kernel profiling (`Profiling` in `ptx_codegen.cc`) is compiled out with `if (false)`; an environment variable
  would make it usable without a rebuild.

### Mixed NVLink and RDMA paths: next steps
The proxy pass (compile phase step 7) sends data needed by several ranks of an NVLink domain over the network once,
in chunks for large transfers. (Before the edge-based model, `computeNvlinkPlan` in processgroup.cc, removed in
c57c8f4, computed a similar plan with gateway reads and local copies, but it was never executed.) Next:
1. The write variant: the source writes the data and the signal into the proxy's memory.
2. An explicit flush before the signal (see Correctness).
3. A ring for the RDMA part where the op allows it, as the process group's all-gather does (each rank reads from one
   fixed neighbor, which relays the data of the ranks before it): fixed neighbors and synchronization with neighbors
   only, instead of `WorkCustom`'s exchanges with all ranks. It needs a heuristic for when it applies, and
   measurements against the proxy pass.

### Other features
- Unaligned copies (align down, shift the bytes, handle the unaligned tail; needs `fence.proxy.async.shared::cta`
  between generic shared memory writes and TMA stores) or a staging buffer, to lift the 16-byte requirement.
- Reductions (`reduce="sum"`), which need float math in the PTX DSL.
- TMA tensor maps for strided regions, instead of copies of non-contiguous pieces.
