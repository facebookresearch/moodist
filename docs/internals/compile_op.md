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
- An **edge** copies `bytes` from its source node to one or more destination nodes, or in a reduction combines
  several source nodes into its destination nodes. `executorRank` is the rank whose kernel performs the copy.
  `cellIndex` identifies the piece of the global tensor.
- Tensor indices on a rank: its inputs, then its input copies (contiguous copies of input pieces that are not
  contiguous in the input), then its outputs, then its output copies, and for reductions across NVLink domains or in
  two phases, a scratch tensor (see steps 7 and 11).

## Compile phase

Every step is collective: all ranks of the group take part (through per-rank queues and barriers).

1. **Spec exchange** (compile.py): every rank gathers `(rank, dtype, inputs, outputs, reduce)` from all ranks, so
   every rank knows every region of every rank, and checks that all ranks gave the same dtype and reduce op
   (`compile` checks this again, for the C++ API, in its first exchange).

   Errors are raised on every rank: a rank that raised alone would leave the others waiting in the next
   exchange (and the queues out of step). A rank that finds an error sends it in the next exchange, and every rank
   raises the first one (lowest rank): compile.py's checks of a rank's own arguments in the spec exchange (where a
   rank with an error of its own raises that one, and the others raise the first one's message, as a `TypeError`, a
   `ValueError` or else a `RuntimeError`), the regions (the device index of a rank's own regions, which only it
   knows) in `compile`'s first exchange, the gaps and overlaps of a rank's outputs (Phase 2) with the read requests
   of Phase 3, the reductions' errors with the agreement on the kernel path (step 9), and the 16-byte check (step 10)
   in an exchange of its own, and the errors in generating and compiling the kernels (step 12: in the tuning, a
   rank's kernels have its own edges, e.g. too many for a configuration's shared memory, and ranks that no kernel
   edge touches generate none; `raiseFirstError`). Failures inside the steps (e.g. running out of memory in the
   tuning) still leave the other ranks waiting, for at most the process group's timeout (see Known issues).
2. **Phase 1**: each rank intersects its own inputs with every output (same `tensor_id`) and sends the
   intersections to the rank of the output.
3. **Phase 2**, cell decomposition: per `tensor_id`, the boundaries of all regions of all ranks cut each
   dimension into intervals, and a cell is one box of the resulting grid (cells have global indices). For each
   cell of its outputs, a rank looks for the covering inputs: none is a gap error, more than one an overlap error
   unless `reduce="any"` (which picks one at random) or a reduction (`"sum"`, `"avg"`, `"min"`, `"max"`), which
   takes all of them, in rank order. A reduced element is computed once, wherever its outputs are
   (`reductionPieces`): a cell with several destination ranks (the ranks whose outputs cover it, e.g. every rank in
   an all-reduce) is split into one piece per destination rank, byte ranges of whole 16-byte units of the cell in
   row-major order (which is how its inputs and outputs, or their contiguous copies, hold it), and destination rank
   (p + cell index) mod m computes piece p: it reads the piece from every source, combines them and stores the
   result to every destination, which is a reduce-scatter and an all-gather in one pass. A cell with one destination
   rank (e.g. in a reduce-scatter) is computed by that rank. A cell with destination ranks in several NVLink domains
   (the ranks learn each other's domain, its first rank, in the first exchange of `compile`), e.g. in an all-reduce
   across domains, is computed once per domain with destinations: it has as many pieces as the domain with the most
   destination ranks has, and in each such domain, destination rank (p + cell index) mod m_d of its m_d computes
   piece p for the destinations there (step 7); or in two rounds, which `reductionPieces` picks per cell, these
   executors compute a share of piece p each and gather the others' (step 7). Every destination rank of a cell
   computes the same pieces, from the same regions. In a cell whose sources and destinations are not all in one
   NVLink domain, pieces are split further into parts (with the same executors; at most 32, more only where parts
   would exceed 1 GiB, see Phase 5, or a slot of the rings of step 7), so that the partial results of other domains
   (step 7) arrive in parts: pieces of 8 MiB or more into parts of at least 4 MiB, and with destinations in several
   domains, into parts of at least 32 MiB / m (from 1 to 4 MiB; m: the most destination ranks of a domain), since
   such an executor writes each part to the m destinations of its domain, which with many of them takes longer than
   reading the part, and smaller parts then overlap the two better (see the measurements in `reductionPieces`). With
   reads (the default, see step 7), what a part sends across domains (the part, or in two rounds a share of it) must
   fit in a slot of the rings of the scratch tensors, 2 MiB (`MOODIST_COMPILE_OP_SLOT_BYTES`): a reduce-scatter of
   256 MiB per rank has 128 parts of 2 MiB instead of 32 of 8 MiB. (In two rounds, the last share of a smaller last
   part can be up to a page larger: the slots of its rings are then that much larger.)
4. **Phase 3**: one read request per cell and source, to the rank that owns the input.
5. **Phase 4**: that rank resolves the offset. A piece that is not contiguous in the input becomes an input copy
   (deduplicated).
6. **Phase 5**: the output's rank creates one edge per cell, executed by itself: the source is the input (or
   input copy) node, the destination its output node, or an output copy if the piece is not contiguous in the
   output. A reduced cell gets an edge per piece instead, with all the sources (in rank order) and the piece's
   executor. No edge is larger than 1 GiB (the kernel's tables hold 32-bit sizes): a larger cell becomes an edge
   per part of 1 GiB, and a larger piece several parts (in `reductionPieces`), the part's index being the edge's
   `chunkIndex` (the proxy pass numbers its chunks within the part's, and the ring graphs chain each part
   separately). Then the ranks exchange node ids.
7. **Reductions** (`mergeReductions`, `crossDomainReductions`): the edges of a piece of a reduced cell, one built
   by each destination rank of the cell (they have the same sources and the same executor), are merged into one
   edge with every destination, executed by that rank. Its sources start at the executor's own (if any) and continue
   in rank order from there, and so do its destinations: the ranks then read from (and write to) different ranks at
   the same time, and the kernel combines the sources in this order.

   Across NVLink domains, an executor reads the sources of its own domain itself, and one value from each other
   domain over RDMA: the input, if the domain has one source, otherwise the domain's partial result, which a rank of
   that domain computes. A piece with one executor (its destinations are in one domain: a reduce-scatter, a reduce to
   one rank) gets its partial results from ranks with the executor's local rank (NICs on the same rail), the one at
   the executor's position among the ranks of its domain with that local rank, so that with domains of the same shape
   every rank computes as many partial results. The executors send their requests to the partial ranks
   (`PartialRequest`), which add an edge per partial result (an item), from the domain's sources (in rank order from
   their own) to a slot of a ring of their scratch tensor (see Rings below), with a done flag (see the kernel and
   Runtime), and answer with the slot, its notification slot, the generation of its use and its credit signal. The
   executor's edge then combines its domain's sources, the inputs of other domains and the partial results (in the
   order of their domains' first ranks), the last two from slots of its scratch tensor that ibReads fill (with a
   `signalIndex`, and for partial results a `waitSlot`: read once the partial rank has sent its notification);
   `sourceWaits` holds what the kernel waits for before it loads each source. Partial results are of the op's dtype
   (16-bit floats are rounded once per domain), and `avg` divides them by the number of all inputs already
   (`divisor`; the executor divides what it combined of the inputs, then adds the `partialSources`), which keeps sums
   of many 16-bit values in range.

   A piece with destinations in several domains (e.g. an all-reduce) has an executor in each of them (Phase 2), and
   these must store the same bits: they combine the same values in the same order, the inputs of the domains with one
   source, then the partial results of the others, in the order of the domains. An executor computes the partial
   result of its own domain itself, an item for itself and for the other executors, and combines it as rounded, from
   a slot of its self ring, once its kernel has stored all of it (the kernel waits for its own done flag). All
   executors get the partial result of a domain from the same rank, which computes it once (the requests for the
   same sources and range are merged, and the edge stores it once, to a slot that all of them read): the domain's
   executor, or in a domain without destinations, its source rank (p + cell index) mod (their number). A domain whose
   one source is on one of its own destination ranks contributes a copy of that input: in an in-place op the input is
   the output that the domain's executor stores the result to, while the executors of other domains still read it
   over RDMA. So the executor first copies it (an item with one source, which the kernel stores as loaded), and every
   executor, itself included, combines the copy, in the input's place (`avg` divides it with the inputs). Every
   domain's partial result thus crosses to every other domain with destinations: with k such domains of n GPUs, an
   all-reduce in one round moves (k - 1) / n of the tensor per GPU over RDMA each way (1/32 with 2 domains of 32 GPUs,
   but 15/4 with 16 domains of 4).

   Within each domain, these edges use NVLS where the domain can (see NVLS runs under the kernel): a partial result
   computed from the domain's inputs, and an executor's own domain's inputs in a reduce-scatter or a reduce to one
   rank, load with `multimem.ld_reduce` (the switches combine the domain's values), and an all-reduce's executor
   stores its result to its domain's ranks with one multicast store per chunk. The switches' partial results are
   rounded within 1 ulp (see NVLS reductions), and every executor combines the same rounded values, as before.

   In two rounds, the executors of a part split it into k shares (byte ranges of whole 16-byte units, whole pages
   from 64 KiB per share on; `shareRange`), and the executor of the j-th domain with destinations owns share j
   (`AllReducePiece::shareOwners`): a reduce-scatter among the executors, then an all-gather. The contributions of
   the domains are the same as in one round (from the same partial ranks), but an owner asks only for its share of
   each (the partial ranks compute an item per share), and its edge combines them in the same order (the same bits as
   in one round), stores its share to the destinations of its domain and to a slot that the other owners read (in
   rings of their own, for shares), and sets its done flag: the other owners request the share (`ResultRequest`, by
   cell and chunk index, in the exchange of the partial results; answered like a partial result) and read it once
   notified, and an edge per other share copies it (one source, stored as loaded) to the destinations of their domain.
   Shares have rings and notification slots of their own: the owners read each other's partial results to compute
   their shares, so an owner waiting for a share to read a partial result could wait for an owner that waits for
   it. An owner reads (n_s - 1) / k + (k - 1) / k of a part (n_s domains with sources) instead of n_s - 1
   parts: 2 (k - 1) / (k n) of the tensor per GPU for an all-reduce (15/32 instead of 15/4 with 16 domains of 4).
   In-place ops stay safe: an output's share is overwritten by its domain's owner once its own partial result (or
   copy) is done, which is all that reads its domain's inputs for that share, and by a copy of another share once
   that share is done, which implies the same. `reductionPieces` picks two rounds for a cell if they save an executor
   at least 2 MiB of reads (25 us at 80 GB/s, about what the second round's notification and read add):
   (bytes / m) (k - 1) (n_s - 2) / k, so never with two domains (the same reads, one more round), with 16 domains of 4
   from about 640 KiB on, with 3 domains of 4 from 12 MiB; `MOODIST_COMPILE_OP_ROUNDS=1` or `2` overrides it (2:
   wherever every share gets at least one unit, with two domains too).

   Rings: the items (partial results, copies, shares) and the data that the CPU threads read go through slots of the
   scratch tensors (a CUDA tensor of the op's dtype that `customOp` allocates per call, the last tensor;
   `tuneKernels` allocates it too). A rank's kernel stores the items that it computes to rings of slots, one per kind
   (partial results and copies, shares) and set of ranks that read them (itself among them for the items that it uses
   itself): an item that several ranks read is stored once. A rank's CPU thread reads into rings of slots, one per
   kind and rank that it reads from (inputs included). A ring has slots for about 4 MiB (2 slots of 2 MiB parts, up
   to 16 of smaller shares, which need more in flight: all-reduce of 256 MiB across 16 domains of 4 GPUs, 2.17 ms
   with 2 slots, 1.88 with 4), or for 32 MiB / their number if a rank has fewer than 8 rings (the items go in bursts
   of the slots, see below: all-reduce of 256 MiB across 2 domains of 32 GPUs, 8 parts of 1 MiB per executor, 0.88 ms
   with 4 slots, 0.85 with 8; `MOODIST_COMPILE_OP_SLOTS` sets another number), fewer if it has fewer uses, fewer
   where a rank reads many rings of another rank's (each has an equal share of the 64 notification slots per pair of
   ranks; a few rings, more only where the sets of ranks that read the items vary, e.g. with domains of different
   sizes), and fewer where a rank's signals or done flags would not do (`layoutRings` halves the largest rings until
   they do). The uses of a ring take its slots in turn: use i takes slot i mod n with generation i / n (n slots), and
   the slot's signals, done flags and notification slots (one per reader and slot) are set to the step value +
   generation (`Graph::Signal`). The CPU thread writes such values from a table of them per concurrency slot, which
   has the generations of the uses of the rank's own slots and those of its credits (below), which are the computing
   rank's: each rank lays out its own rings, and a computing rank's ring can have fewer slots than the rings of a
   rank that reads it (e.g. if it has more rings), and so more generations. A slot is used again once its previous
   use is over: a rank's kernel stores an item to a slot once every rank that reads it has read the previous one (a
   credit per reader: its CPU thread sets a signal of the computing rank's once its read has completed,
   `IbRead::creditSignal`, which the kernel waits for before it stores, `destinationWaits`) and the rank's own edge
   that used it is done (a second done flag), and a CPU thread reads into a slot once its kernel has used the
   previous contents (the done flag that the edge that used them sets, `IbRead::freeFlag`); a slot's last use in the
   op needs neither (no credits, no such done flags). A slot holds the largest of its uses, a part or a share, of at
   most 2 MiB (Phase 2), so a rank's scratch tensor holds a few slots per peer instead of everything that it
   exchanges (before rings, it held every partial result whole for the whole op: a reduce-scatter across k domains
   took about 2 (k - 1) times a rank's output, 7.5 GiB at 256 MiB per rank with 16 domains, now 4 (k - 1) slots of
   2 MiB, 120 MiB; an all-reduce in one round about k / m of the tensor, in two rounds about 3 / m, now about 4 MiB
   per ring, at most 32 MiB in all with fewer than 8 rings). An op's generations are less than 3840 (step values
   advance by 0x1000 per op), so a slot can have up to 3839 uses (about 15 GiB through a ring of 2 slots of 2 MiB; an
   op with more raises an error).

   No rank waits for a rank that waits for it: every edge has a key that every rank computes the same, and a rank's
   kernel takes its edges in key order (see the kernel): the chunk index; then the pieces with executors in several
   domains before the others, by cell and piece for the former, by the index of the executor's edge among its edges of
   the chunk for the latter; the items before the executor's edges that use them. Every wait is for an edge with a
   smaller key, or for a transfer that waits only for such edges: an executor's edge waits for its items, which come
   before it; an item waits for the previous use of its slot, whose edges come before it (the uses of a ring are in key
   order on both of its ranks); and the copies of the shares of a part come after the next part's edge (blocks that wait
   for the shares of a part then wait behind the share of the next part, which they can compute meanwhile, so that the
   other owners get it early), before the shares that wait for them (the previous uses of their slots). An item goes as
   early as its slot lets it (other ranks wait for it): the first uses of a ring come first (in their order), the others
   right after the edges of an earlier use, use j after those of use j - 3 (two uses ahead of its own; at most n - 1
   with n slots, n - 2 if the rank's own edges use the ring, so that the item does not wait right behind the edge that
   it waits for), so that the transfers of the items of a part overlap the edges before it. Right before the edges that
   use them, the items of a part were computed only once the edges of the part before were done, which waited for that
   part's items from the other ranks, and the ranks went from part to part in step (reduce-scatter of 256 MiB across 16
   domains of 4 GPUs: 60 ms instead of 48). In two rounds, items and edges then alternate, and the shares that the other
   owners wait for come early (all-reduce of 256 MiB across 16 domains of 4 GPUs: 1.86 ms, 2.00 with all first uses
   first). The items of a ring that the rank's own edges use in one round (the partial results of its own domain in an
   all-reduce) go in bursts of n instead: uses bn to bn + n - 1 after the edges of use bn - 1, so that the kernel mostly
   loads, then stores, as when every item had a region of its own (all-reduce of 256 MiB across 2 domains of 32 GPUs:
   0.88 ms, 0.91 with the items two uses ahead; but reduce-scatter of 256 MiB across 16 domains of 4 GPUs in bursts: 56
   ms); but not where the rank's domain can use NVLS for both, from 16 ranks (see NVLS runs in the kernel): its kernel
   loads the items with `multimem.ld_reduce`, which the GPUs' NVLink egress limits, and stores the edges' results with
   multicast stores, which their ingress limits, so that interleaved, both directions work at once (all-reduce of 256
   MiB across 2 domains of 32 GB300: 705 us two uses ahead, 780 us in bursts of 8). Within a chunk, a partial rank
   computes the items of pieces with one executor for its own executor first, then for those of the domain after its
   own, starting at its own position (where their keys tie), and an executor issues
   its reads per part from the domain before its own on, the order in which they are computed: at any time, every
   partial rank then has one reader, and every executor one source (reduce-scatter with 16 domains of 4 GPUs: 47.5 ms at
   256 MiB, 68.7 ms with the partial ranks serving the executors in rank order and the executors reading in domain
   order; see Performance notes). The items of pieces with executors in several domains are in piece order on every
   rank, which an item that several executors read needs.

   Rings of the domains (`MOODIST_COMPILE_OP_RING`, with reads): an RDMA read from one source at a time is faster than
   from several at once (each GPU receiving 15 parts of 256 MiB over RDMA, 16 domains of 4 GB300: 95 GB/s from one rank
   at a time, 86 from two, 82 from eight, as the partial results of a reduce-scatter came; with several senders per
   receiver, the switches mark packets for congestion, and the NICs slow down). So with 3 or more domains of the same
   size, a cell with one executor (a reduce-scatter's, a reduce to one rank's) and sources in every domain can go along
   a ring of the domains instead: the domain after the executor's computes its partial result (step 0; with one source,
   a copy of it, which the next step adds as an input: an average divides it), the next one adds its own to it (step 1:
   an item whose sources are its domain's and the item of step 0, read over RDMA), and so on around the ring, and the
   executor adds its domain's sources to the last one. Every step is computed by the rank at the executor's position on
   its rail (`pickPartialRank`), so that every rank reads from one rank (at its position in the domain before its own)
   and is read by one. Each step rounds its running sum to the op's dtype: as many roundings as with a partial result
   per domain, but of sums that grow along the ring, which in 16-bit types is less accurate. The executor asks for every
   step at once: a step's request names the rank of the step before (`ringFrom`) and the rank that reads the item
   (`ringTo`), to which the computing rank sends its answer (`RingAnswer`). The items of a rank's steps go through one
   ring (read by the next domain's rank), and it reads into one ring (from the domain before). Their keys are (group of
   parts, step, cell, part), with groups of 8 parts (`MOODIST_COMPILE_OP_RING_GROUP`): the links carry the items of a
   group in order, and the item of the step before that a rank needs came 8 parts earlier. A ring has at least half as
   many slots as its two largest consecutive rows (the uses of one group and step) have uses, which keeps every wait for
   a smaller key (an item waits for its slot's previous use to be read by the rank after, into a slot whose previous use
   is at least two rows back). With groups of 8, that is the 8 slots of 2 MiB per ring that the scratch budget gives a
   rank's two rings anyway (groups of 4 were 5% slower at 256 MiB, groups of 16 1% faster with twice the scratch). A
   rank credits the rank before it for every item that it reads; on the queue pairs of its reads from that rank, the
   credits slowed the reads down, so they go through queue pairs of their own (see Runtime). A rank takes the ring for
   its cell if it has one such cell (a rank with two reduce-scattered tensors has two and takes none), of 6 MiB or more:
   below that, the steps one after the other take longer than the reads from every domain at once (see Performance
   notes). `MOODIST_COMPILE_OP_RING=1` takes it at any size, `0` never.

   With writes (`MOODIST_COMPILE_OP_TRANSFER=write`; reads by default), the data goes the other way: an executor
   allocates a region of its scratch tensor and the signal for each item (or share) that it needs when it makes its
   request, and sends them with it (`WriteTarget`). The rank that computes it gets a `CustomOpDescriptor::IbWrite`
   instead of a notification: its CPU thread writes the item there once its kernel has stored it, then sets the
   signal, which the executor's kernel waits for (see Runtime, Signals). No notification, and nothing to do for the
   executor's CPU thread: one transfer instead of a notification and a read. Every item and every write has a region
   of its own (no rings: the scratch tensors hold everything, as before rings). The writes go in the order in which
   the kernel computes their data (the shares of two rounds each to the owners of the domains after its own first).
   The edges, their order and the bits are those of reads. The inputs of domains with one source are read either way.

   Reductions of CPU tensors are an error (step 9).
8. **Proxy pass** (`proxyRemoteEdges`; `MOODIST_COMPILE_OP_PROXY=0` disables it; not for reductions, whose kernel
   takes no relay edges): data that several ranks of an
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
   node are ordinary dependent edges (in the `multicast` graph, where the proxy executes them, they wait for the
   signal themselves, in the kernel's item sequence; see the kernel's Scheduling). Transfers of 8 MiB or more are
   split into chunks of at least 4 MiB (at most
   32): each chunk has its own read, signal, relay edge and copies (`chunkIndex`; the chunk nodes get ids that
   their owner never assigns, numbered in the same order by every member of the domain), so that the copies of a
   chunk start as soon as it has arrived.
9. **Kernel / RDMA split**: an edge goes to the kernel if all its nodes are CUDA, all its ranks are kernel peers
   (`cudaRanks`: the ranks whose memory this rank can map, i.e. CUDA IPC peers on the same node, or all ranks of a
   multi-node NVLink fabric domain such as NVL72), and, for process group collectives built on compile_op
   (scatter, gather), its size and offsets are multiples of 16 bytes. Every other edge becomes an `ibRead`, an
   RDMA read by the CPU thread of the destination rank (with the edge's signal index). Kernel tuning and launch
   are collective, so the ranks agree on whether the kernel path is used at all: it is if any rank has a kernel
   edge. A rank without one (e.g. one that only sends, or only receives over RDMA) still takes part in the tuning,
   and runs a kernel if the graph gives it kernel edges to execute (with `multicast`, a rank that only sends executes
   the copies of its inputs) or the kernel edges of other ranks touch its tensors (its kernel then only publishes
   their addresses and takes part in the exit sync). Without the kernel path, all edges become ibReads, and the proxy
   pass is undone (the copies from a proxy's node would race with the proxy's own read). Reductions run in the
   kernel only: an edge with several sources that the kernel cannot take (CPU tensors; those of CUDA tensors in other
   NVLink domains are kernel edges after step 7) is an error, on every rank. (Until October 2026, the kernel path was used only if every rank
   had a kernel edge, so e.g. a gather, or a transfer between two ranks of a larger group, went entirely over RDMA.)
   The ibReads are sorted by source so that each source has about one reader at a time: in the same
   exchange, the ranks learn each other's domain (its first rank), and the rank at position p of domain i reads
   the domains after its own in turn, each from position p (mod the domain's size) on, i.e. (domain i + 1 + a,
   position p + b) in step (a, b); reads from its own domain come last. (A rotation by rank, (source - rank) mod
   size, does not work: the ranks of a domain skip its own ranks, so they all start on the same source.) In the
   next exchange (whether any rank needs RDMA), each rank also describes its reads without a signal: whether it
   reads from every rank outside its domain (or from every other rank), and the least and the most it reads from
   one rank. If every rank with such reads does, with amounts within 1/16 of each other, the steps all take equally
   long and a rank reads one source at a time (`readSources` 1); otherwise 8 at a time (see the RDMA read notes
   under Performance).
10. **16-byte check**: kernel edges must have sizes and offsets that are multiples of 16 bytes (the kernel uses TMA
   bulk copies). The ranks exchange their first violation and raise together.
11. **Graph variants** (`MOODIST_COMPILE_OP_GRAPHS`, default `multicast`), each a `GraphBuilder` pass over the
   kernel edges:
   - `multicast`: the edges that read the same source node are merged into one edge with several destinations,
     executed by the source rank: it reads each chunk once and writes it to all its destinations.
   - `read` (`buildNothing`): every edge stays with its destination rank, which reads from the source.
   - `ring_read`, `ring_write`: the destinations of a cell form a chain, starting at the source rank, and each
     rank copies the cell from the previous rank's output (read: the receiver executes the copy, write: the
     sender). A proxy's relay edge heads the chain of its cell (of each chunk, for chunked transfers).

   Edges have unfilled sources, i.e. depend on other edges, in the ring graphs and for proxied transfers (the
   copies from a proxy's node). Relay edges stay with the proxy in every variant.

   Reductions use the `multicast` graph only, which leaves their edges (merged in step 7) with their executors: as
   it is, and in two phases (`twoPhaseReductions`) if some rank has a piece of at least 2 MiB that combines
   several sources into several destinations in one NVLink domain (a piece of an all-reduce that takes no part in a
   reduction across domains). Such an edge becomes an edge that computes the piece into a region of the scratch
   tensor (adding one if the op has none), with a done flag (`doneFlags`), and an edge with one source that copies it
   from there to the destinations once that is set (`sourceWaits`). The kernel computes all such pieces before it
   copies any (the copies come after all other edges), so the blocks first read all the sources, then write
   all the destinations, instead of both in every item. The same bits (the copies store the rounded result as it
   is), and in place too (a piece's sources and destinations are those of its executor's two edges, the second of
   which waits for the first). Tuning times the two graphs against each other (step 12), and the op takes the scratch
   tensor of the one chosen; `MOODIST_COMPILE_OP_PHASES=1` or `2` keeps one of them (2: pieces of any size). Two
   phases win from 128 MiB on 32 GPUs (see Performance notes); below 2 MiB per piece they were slower, and the
   variant doubles the kernels that tuning generates and times. Not where the kernels will use NVLS, as far as
   compile can tell (see the kernel; unless `MOODIST_COMPILE_OP_PHASES=2`): every rank can (the group, dtype and
   reduce op allow it), and the ranks all of whose edges have the shape for it have some reductions, up to
   `nvlsReductionMaxBytes`. Those run in one pass, faster than either (an all-reduce of 256 MiB on 32 GPUs: 551 us,
   against 842-892 us in one pass and 894 us in two phases without NVLS), and an op whose launches cannot use NVLS
   (its tensors not in place) then runs the reduce loop in one pass. In the same exchange, the ranks learn whether
   any rank's edges could use NVLS (NVLS reductions or runs, see the kernel), and if so set up the multicast objects
   of the group's domains (`Group::setupNvls`, collective: here, where every rank takes part, and not in
   `generateKernel`, where the ranks of a domain decide among themselves).

   `finalize` then sends every edge to every rank that appears in it, sorts the edges into `cudaEdges` and
   `rdmaEdges` (always empty at the moment, because of the split above), and builds the tensor mappings: which of
   its tensors each peer must map, and in which slot of the peer's address table they go.
12. **Tuning** (`tuneKernels`): the grid size {4, 8, 12, 16} (blocks of 128 threads with one pipeline of 32 KiB
    chunks in 226 KiB of shared memory; reductions: {8, 16, 24, 32} with blocks of 256 threads, as their blocks
    also combine the data, and reading from other GPUs takes more SMs than writing), for every graph variant (4
    settings per variant; `MOODIST_COMPILE_OP_CONFIGS` replaces these configurations, which every rank checks
    against the reduce loop's needs for reductions). Each setting is generated and JIT compiled (errors are raised on
    every rank, see step 1), then run on zeroed scratch tensors: 12 warmup launches, then 2 timed batches of 12
    launches. The ranks exchange their timings and take the slowest rank's per setting (of the ranks with a kernel):
    an op takes as long as its slowest rank, e.g. the executors of a reduction across domains, which combine more
    than the others. The fastest setting wins, with a penalty of 0.5% per block: the kernel occupies its SMs, which
    the computation it overlaps with lacks meanwhile, so a bigger grid has to be clearly faster to be chosen (as it
    is for copies limited by the SMs' bandwidth). The kernel of the chosen setting is generated and compiled once
    more for the op (its errors too are raised on every rank). The tuning launches wait for each other across the
    ranks, for at most the process group's timeout (the process then ends: a kernel cannot be stopped). Tuning
    launches run without the CPU thread, so in tuning kernels, relay edges and the reduce loop's waits for RDMA
    signals (sources from other domains, credits) do not wait; the reduce loop's done flags and its waits for them
    work as in the op's launches (the pieces of two phases, the items that a rank uses itself and their slots), in
    `cpuInBuffer`, which the step value reset before and after the tuning launches (`resetBarriers`) clears: the
    tuning launches number their steps from 1 on, and their flags would otherwise mislead the op's launches.

## The kernel (`generateKernel`)

One kernel per op and rank: `compile_op_copy(params {stepValue, concurrencyIndex}, tensorAddrs[], mappedAddrs[])`,
launched in the remote memory synchronization domain. Tensor addresses are kernel arguments, so a compiled op runs
on new tensors at every call. Kernel parameters are limited to 32764 bytes, which about 4000 tensors and (tensor,
peer) mappings fill (e.g. an all-gather of 1400 tensors on 2 ranks): an op whose address arrays could exceed 32000
bytes (counting NVLS's) has them in a global array of its module instead, with a copy per concurrency slot, and the
module's second kernel, `compile_op_params`, copies them there from its own parameters (31 KiB per launch) before
every launch, on the same stream (`launchCompileOpKernel`). The ops of a slot run one after the other on the GPU
(customOp), so the copy stays until the kernel has read it; tuning fills every slot's copy once per candidate.

- **Buffers** (device memory, one slot per concurrency index, mapped by all kernel peers; cached per group and
  reused by later ops): `addresses` (peer tensor addresses, per block), `syncs` (exit sync, one row per block),
  `depFilled` (per node and block).
- **Parity**: every (concurrency index, block) keeps a bit that flips at each launch. Exchanged words (addresses,
  and in NVLS ops identities) carry it as a tag, bit 63 with parity 1 and bit 62 with parity 0, so a reader can tell
  the current launch's entries from the previous one's, and never takes a word of 0 for an entry: the step value
  reset (`resetStepValue`, at every compile and when the step values wrap) clears these tables, also those of ops in
  use, whose parity goes on (with parity 0 as a tag of 0, a launch after a reset could read a cleared entry as an
  address of 0 when a peer was late).
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
  runtime, so its code size does not depend on the number of edges (they hold 32-bit sizes: compile_op makes no
  edge larger than 1 GiB, see Phase 5).
- **Scheduling**: a rank executes the edges whose executor it is. If no edge waits (all sources filled, no relay
  edges), every block first resolves the addresses of all edges into a table in shared memory (in global memory if
  it exceeds 32 KiB), and the pipelines of all blocks then take items from one sequence: an item is a chunk of an
  edge and a group of its destinations (with few chunks, an edge's destinations are split into groups, for about
  4 items per pipeline, so that the stores of an all-gather of a few chunks to many ranks spread over the SMs).
  Items are claimed from a counter in global memory (one per concurrency slot and launch parity), the first claim
  of a pipeline taking enough items to fill its ring, the later ones one item, claimed when the pipeline starts on
  the previous one. Edges are in the order of the rank they write to rotated to start at the executing rank (so
  that in an all-to-all, the ranks do not all write to the same rank at once), or for edges that read another
  rank's node into the executing rank (the `read` graphs), of the rank they read from (all-gather on 32 GPUs, 64 MiB
  per rank, 16 blocks: 74 GB/s per GPU in destination order, where every rank read rank 0 first, 631 GB/s rotated
  by source). A proxy's copies of what it reads (compile phase step 8; in the `multicast` graph, the proxy executes
  the copies from its node) go through the sequence too, if every edge of the rank that waits, waits for a relay edge
  of the rank, and no other edge reads the relays' nodes or waits for the copies (`relayItems`): the relay edges are
  left out, the copies that wait come after the others, in the order of their relays' signals (the order in which the
  CPU thread issues the reads), and a pipeline loads an item of such a copy once the relay's signal has arrived (see
  Relay edges; then `fence.acquire.sys` and `fence.proxy.async.global`). Until then, it stores the items that it has
  loaded, and waits for the signal only when it has none left; the first claims share the items that do not wait.
  (In the rounds below, all-gather on 64 GPUs in two NVL72 domains, 256 MiB per rank: 30.2 ms, 560 GB/s per GPU, the
  blocks' static split of each edge leaving the slowest SMs last; in the sequence 25.3 ms, 668 GB/s; see Performance
  notes.) Otherwise, if some edge waits, in rounds: warp 0 scans the
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
- **Relay edges** (proxied transfers): ready once the words of signal `rdmaSignal` in `Group::rdmaSignalBuffer` for
  the concurrency index (one per NIC, see Runtime, Signals) have all reached stepValue (always ready when tuning);
  nothing to copy. In the item sequence (`relayItems`, see Scheduling), the copies from the relay's node wait for the
  signal themselves (`edgeRelayWait`, the signal in `PipeEdge::waits`), and nothing marks the node filled.
- **Reductions** (`reduceLoop`): a rank that executes a reduction (an edge with several sources, or a part of one
  across NVLink domains, see compile phase step 7) runs this loop instead of the copy pipelines. The blocks (of 256
  threads) take items one at a time from the sequence (an item: a chunk of an edge, with all its destinations).
  Thread 0 loads the chunk of each source of an item in turn with bulk loads, into a ring of 32 KiB stages (ahead
  across items as far as the ring allows), and every thread combines each source's chunk, as it arrives, into its
  accumulators (16 bytes per thread and step), in the order of the edge's sources: in `float32` for `float32`,
  `float16` and `bfloat16`, in `float64`, `int32` and `int64` for those; `avg` divides once the inputs among the
  sources are combined (by the number of inputs of the result; the partial results of other domains come after
  them, already divided). The threads write the result into one of two result buffers (rounding 16-bit floats once,
  to nearest even), and thread 0 stores it to every destination with bulk stores (the stores of the item before may
  still be reading the other buffer). An item with one source (a cell that one input covers) is stored as loaded.
  Sums start from -0 (x + -0 = x for every x), floating point min and max propagate NaN. (Within an NVLink domain,
  a launch may combine the sources in the switches instead, see NVLS reductions and NVLS runs below.) After thread 0's
  stores, warp 0 synchronizes (`bar.warp.sync`) before the next item's block barriers: the ptxas of CUDA 13.0 can fail
  to reconverge a warp after a branch of one thread (see Known issues).

  The blocks take the edges in the order of the graph: across domains, the key order of compile phase step 7, in
  which the items that other ranks wait for come ahead of the edges that use them, and in two phases the pieces
  before their copies (the order of peers that the copy pipelines use does not apply to reductions, whose items read
  all their sources and store to all their destinations). (Two sequences with blocks of their own, for the edges that
  wait for no source and for the others, were slower: fewer blocks computed the data that every rank waits for, and
  the blocks of the second waited meanwhile; all-reduce of 64 MiB across 2 domains of 32 GPUs: 309 us instead of
  266.) A block that has stored an item of an edge with done flags waits for its stores to complete and counts the
  item (a per-edge counter, which wraps around after every launch), or counts it right away if the flags only free
  slots that the edge has loaded (`doneWhenLoaded`: an executor's edge, unless it stores a share, and a copy of a
  share; the item's loads are complete); the block that counts the last one sets the edge's done flags (slots 16 +
  index of `cpuInBuffer`, host memory, for the CPU thread and for the kernel itself) to the step value + their
  generations (from a table of the module). A source that the CPU thread reads from another domain is loaded once the
  read's signal (in `Group::rdmaSignalBuffer`) has arrived, an item that the kernel computes for itself (an
  executor's own partial result) once its done flag is set, and an item is computed once the slots that it stores to
  are free (its destination waits, before its first source: a credit from the rank that has read the slot's previous
  use, or the done flag of the edge that has used a self slot); a word of the wait table holds the index, the
  generation (the value to wait for is the step value + generation) and whether it is a done flag. Until then,
  thread 0 stops loading there, and the block goes on with the stages it has; it waits only when it has combined
  every stage loaded before (at the start of a stage). Its earlier items are then complete, and every wait is for an
  edge that comes earlier in its rank's order, or for a transfer that waits only for such edges (see compile phase
  step 7), so the blocks of all ranks make progress.
- **Exit**: an all-to-all step value exchange among the kernel peers (`execSync`), so that no rank finishes while
  peers may still read or write its tensors.
- **NVLS** (NVLink SHARP multicast; copies in groups of at least 16 ranks that are one NVLink domain, ops not behind
  a process group collective): an edge to every rank of the group, at the same tensor and offset on each (an
  all-gather's), can store each chunk once to the group's multicast mapping, which the switches copy to every GPU.
  Each NVLink domain of the group has two multicast objects of its ranks (`Group::setupNvls`, collective over the
  group, at the first op that can use them; a multicast object cannot span domains: the domain's first rank creates
  them, the others import them, and a rank alone in its domain has none). One covers moodist's whole allocator pool:
  every allocator chunk is bound at its offset in the pool when it is mapped (`allocatorAddChunkListener`), so offset
  o of the pool of every rank of the domain is at the mapping's base + o; a counter area follows. The other covers
  the group's communication pool (`allocatorCommPoolCreate`; `moodist.comm_empty`): every slab of the pool is bound
  at its logical offset when the pool adds it, so a buffer that every rank allocates in the same order is at the same
  place of that object on every rank of the domain. A rank uses NVLS for copies if every edge it executes is such
  an edge (no waits) and every rank that does stores the same number of chunks (agreed in `generateKernel`). Each
  launch checks the destinations: with the addresses, the ranks publish their tensors' NVLS identities (the object
  and the offset in it: `nvlsIdentity` in processgroup.cc) into a table of each peer's with the layout of the
  address table. If every destination has the local destination's identity, block 0 stores the chunks with one
  multicast store each (to the multicast address of the local destination's tensor, which `customOp` passes with the
  identities in a fourth kernel parameter, or through the parameter buffer; chunks of 64 KiB, through a ring of its
  own over the stage memory: per chunk, the window and the counter cost time) and the other blocks only take part in
  the address exchange and the exit sync; otherwise the launch stores to each destination as usual. Multicast stores
  at full speed from all ranks overload the switches' links to the GPUs, which then deliver about 2/3 of their
  bandwidth, so the ranks advance in step: an op has a step counter per concurrency slot in the counter area; once
  the stores of a chunk have completed, a rank adds 1 to it through the multicast mapping (`multimem.red`, relaxed: a
  release waits for all outstanding stores), and it stores chunk c once its local copy of the counter shows every
  rank's chunks up to c - 8 complete. The counter only grows; `customOp` passes its value before the launch (launches
  of the slot times the chunks of all ranks); a launch that does not use NVLS adds its chunks at the end.
- **NVLS reductions** (within an NVLink domain: all-reduces in domains of at least 8 ranks, reductions with one
  destination of at least 12; sums of every floating point type, averages of `float32`, `float64` and `bfloat16`,
  sums, min and max of integers, `int32` only up to 4 MiB per op): a reduction with a source on every rank of the
  executor's domain, at the same tensor and offset on each (`nvlsReductionShape`: a piece of an all-reduce, a cell of
  a reduce-scatter or of a reduce to one rank, in a group that is one domain or within each domain), loads each
  16-byte vector of its result with `multimem.ld_reduce` from the multicast address of its local source: the switches
  read the values of every member of the multicast object and combine them (16-bit floats in `float32`,
  `.acc::f32`, rounded as described below; one instruction per vector, two or four for `float64`, `int32` and
  `int64`, whose forms have no vectors). With a destination on every rank of the domain (an all-reduce's), the result
  goes to all of them with one `multimem.st` to the multicast address of the local destination: each GPU then sends
  its input once and receives the result once, (1 + 1/N) times the data instead of 2 (N - 1) / N with unicast pieces;
  otherwise it goes to the local destination. A rank does so (`nvlsReduceLoop`) if every edge it executes is such an
  edge (`nvlsReduceLocal`; no step counters). The ranks of a domain decide together, with the copies' decision: a
  rank whose kernel peers are not its domain's ranks says so, which they all hear. `Group::setupNvls` is collective
  over the group, so `compile` calls it, where every rank takes part, if some rank's edges could use NVLS (the
  kernels then only use its result). Every launch checks the identities of every source and every multicast
  destination first (the NVLS edge table has an entry per node list and NVLS slot): a `multimem.ld_reduce` at an
  offset where a member has no memory bound makes the reader's context fail with CUDA_ERROR_CONTAINED. A launch that
  cannot use NVLS runs the reduce loop. The warps of the first blocks claim items of 4 KiB (8 vectors per lane) of the
  NVLS sequence (`nvlsFirst`) one at a time from counters of their own (used by alternate launches, like the copies'
  claim counters); the blocks are enough for about 4 MiB in flight across the op's GPUs, by each rank's share of the
  reduced bytes (4 blocks of 256 threads with 32 GPUs, 8 with 16: in the probe, more were slower; the root of a
  reduce to one rank takes all of its grid, which limits it, see Known issues). The vectors past an edge's end load
  its last vector again and store nothing: no multimem instruction is under a PTX predicate (see Known issues), and
  kernels whose edges have no multicast stores have no `multimem.st`. Differences from the reduce loop: the switches'
  sums start from +0 (a sum of -0 only is +0); their sums of 16-bit floats are within 1 ulp of the exact sum, but
  not always the nearest value, which the reduce loop's are (on 8 GB300, 24-26% of the elements of sums of random
  values, positive or of either sign, were 1 ulp off, with a mean error below 0.01 ulp); an average of `bfloat16`
  divides that sum (exactly with a power of 2 ranks); and min and max of floating point types and averages of
  `float16` do not use NVLS (`multimem.ld_reduce` has no min and max of `float32` and `float64`, those of `float16`
  and `bfloat16` return the other value where one is NaN, and the switches round a `float16` sum to `float16`, which
  overflows from 65504 / N, where the reduce loop divides the `float32` sum). Integer sums wrap, as in the reduce
  loop (`int32` and `int64`, measured). An all-reduce gets the same bits on every rank (one multicast store per
  vector), and its NVLS launches the same bits in every run (random values, 32 GB300; a launch whose tensors are not
  in place gets the reduce loop's). The tuning launches do not use NVLS (their scratch tensors are not in the pools
  at the same offsets), so it is decided by rule; reductions that will use it have no two-phase variant (compile
  phase step 11).
- **NVLS runs** (the reduce loop's edges, e.g. those of reductions across NVLink domains, compile phase step 7: the
  same domains, dtypes, reduce ops and limits as NVLS reductions): the first N sources of an edge (N: the ranks of
  the executor's domain) can be a run, a source on every rank of the domain at the same tensor and offset, filled
  and without waits (the domain's inputs, which come first: an executor's own domain's in a reduce-scatter or a
  reduce to one rank, those of a partial result that a rank computes for other domains), and its first N
  destinations, a destination on every rank of the domain at the same place (the result of an all-reduce in the
  domain); `nvlsReductionRuns`: both from 8 ranks (as an all-reduce of `nvlsReduceLoop`), sources alone from 12 (as
  with one destination), destinations alone from 16 (as copies with multicast stores: across 2 domains of 8 GB300, an
  all-reduce of 256 MiB took 874 us with multicast stores of its results, 772 us without). A rank
  whose kernel uses the reduce loop has runs if any of its edges has one (`nvlsRunsLocal`, decided with the
  others, see NVLS reductions; its edges otherwise as before). A run has an NVLS slot (after the destinations: the
  sources' run, then the destinations') and an entry in the NVLS edge table, and the launch checks it like the
  others; a run whose check fails gets 0 in its slot, and the edge takes it as N sources (or destinations) as usual,
  each run on its own. Thread 0 loads a run of sources as one stage without data (an arrive on the stage's
  mbarrier, no bulk load), and the threads load their vectors of the chunk with `multimem.ld_reduce` from the
  multicast address when they get to it, as one source (vectors past the chunk's end load its last vector again);
  thread 0 stores to a run of destinations with one bulk store to the multicast address of the local destination,
  then to the destinations after it. On 32 GB300 in one domain, all-reduce through the reduce loop with runs:
  16 MiB 62 us, 64 MiB 180 us, 256 MiB 571 us (`nvlsReduceLoop`: 51, 151, 549 us). Across 2 domains of 32 GB300,
  an executor's partial result of its own domain and those it computes for the other domain's executors load with
  `multimem.ld_reduce`, and the all-reduce's results go to its domain with multicast stores: see Performance notes.
  (Partial results with done flags wait for the completion of their block's bulk stores, earlier multicast stores
  included; storing them from the threads' registers instead changed nothing measurable, 709 / 708 us at 256 MiB.)

## Runtime (`customOp`, `WorkCustom`)

- `customOp` checks the tensors (shape, dtype, device; for user ops also that CUDA tensors come from the moodist
  CUDA allocator and are 16-byte aligned when the op has a kernel), makes the input copies
  (`narrow().contiguous()`), allocates the output copies (copied into the outputs when the op is waited on) and the
  scratch tensor of reductions across NVLink domains or in two phases (on the op's stream: its memory is reused only
  by work after the op on that stream, which comes after the op's RDMA transfers into and from it). A rank with a
  scratch tensor counts as one with CUDA tensors, also if it has no tensors of its own in the op (a partial rank need
  not have any): its stream must wait for the CPU thread too, or a new tensor on the stream could take the scratch
  tensor's memory while other ranks still read from it or write into it.
- Stream order (a rank with CUDA tensors, when any rank needs RDMA): write `cpuIn` = stepValue (the CPU thread may
  start), then the kernel, then wait until `cpuOut` >= stepValue.
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
- The CPU thread (`WorkCustom`, only if some rank has ibReads or ibWrites) works with the rank's peers in the op
  (`CustomOpDescriptor::peers`: the ranks it reads from, writes to or notifies, and those that read from it, write to
  it or notify it; the compile phase finds them in an exchange it already has). It publishes the rank's input list
  (addresses, sizes, rkeys; CUDA tensors are registered on first use; the inputs, the input copies and the scratch
  tensor) to its peers, into words of its own (`Group::customDyns`: the process group's all-gather ring keeps the
  lists of a rank's ring neighbor in its list slots 0 and 1, which a compile_op of rank 0 or 1 16 ops later overwrote
  while the rank still read from its neighbor), registers the outputs, collects its peers' input lists, waits for
  `cpuIn`, exchanges step values with its peers (their inputs are ready), issues its ibReads (split into parts of
  about 128 KiB, and if the op has fewer reads than NICs, smaller reads too, into parts of at least 64 KiB; reads
  with a signal first, in the order of the compile phase (a proxy's in signal order, i.e. the chunks of a transfer in
  order; a reduction's in the order of the executor's edges), then the others in the order of the compile phase,
  `readSources` at a time: with more than one, the parts of that many reads go round robin, and a read
  whose parts have all been issued makes room for the next; each part on the NIC with the fewest in flight, up to 8),
  waits for them (and for its reads of the peers' input lists of more than two tensors, also of the lists that it
  does not use, of peers that only read from it or write to it: their completions could otherwise come after the op
  had ended and its `WorkCustom` was reused), runs a second barrier with its peers (nobody reads this rank's inputs
  any more), waits until all other ranks have reached the op too (see Slot reuse below), and writes `cpuOut`. The
  parts of a read with a signal share it: the signal is written when the last part has completed. Small parts keep a
  rank on one or two sources at a time (with `readSources` 1), which with the read order keeps the ranks in step (see
  the RDMA read notes under Performance).
  Reductions across NVLink domains (compile phase step 7): a read of an item waits until its notification has arrived
  (slot `waitSlot` of the partial rank's slots in `Group::customNotifications` is at least the step value +
  `waitGeneration`), and a read into a slot of a ring until the kernel has used the slot's previous contents (its
  done flag `freeFlag`); it lets the reads after it go first meanwhile: the next part of the first read that is ready
  goes next (the reads before it go on as soon as they are ready, which keeps the order; the reads with parts left
  form a list). The search looks past at most 64 reads that wait, and the forwarding of notifications (below) past at
  most 64 whose flags are not set: with rings, most reads and notifications of an op wait for most of it (for the
  slots of the parts before them), and passes over all of them took longer than the transfers (reduce-scatter of
  256 MiB across 16 domains of 4 GPUs: 1920 reads and as many notifications per rank, 67 ms instead of 60); they wait
  only for the ones before them (and for other ranks), so those further on can wait too. Once a read of an item has
  completed, the CPU thread also sets the item's credit signal in the partial rank (`creditSignal`), which frees its
  slot there. The CPU thread of a partial rank meanwhile watches its kernel's done flags: once an item's flag has the
  value of the slot's use (step value + generation), it writes that value into the item's notification slot of the
  executor (`writeNotification`; an item that the executors of several domains read has a slot and a notification for
  each). The credits go through a second queue pair per peer and NIC (`IbCommon::controlQps`, `Rdma::postWriteControl`),
  not the one of the reads from that rank: small writes to a rank through the queue pair of reads from it slow the reads
  down (a GPU reading 3.75 GiB from one rank: 95 GB/s; with a write of 4 bytes to it per part of 128 KiB, 29 GB/s; with
  one per 16 parts, 78 GB/s), and in a ring of the domains (step 7) a rank credits the rank that it reads from for every
  item (reduce-scatter of 256 MiB across 16 domains of 4 GPUs: 52.1 ms with the credits on the queue pairs of the reads,
  45.6 ms without, with groups of 4 parts). All credits to a signal go through the same queue pair, so they arrive in
  order. The control queue pairs have send queues of 256 work requests (the others 2048), and `RdmaIb` keeps at most 256
  control writes in flight per NIC. The notifications stay on the queue pair of the reads: through the control queue
  pair, they made no difference at large sizes, and the times of small ops varied more (reduce-scatter of 1 MiB across 2
  domains of 32 GPUs: 167 us at the median instead of 159, the fastest 10% 140 us instead of 156). The values that the
  CPU thread writes come from a table per concurrency slot (`generationValue`), which an op fills when it starts (the
  slot's previous op has finished its writes from it), for the generations of the uses of its rank's slots and those of
  its credits, the partial ranks' (`CustomOpDescriptor::numGenerations`). The notifications have words of their own: in
  the `inStepValueDeviceChunk` words, where they were before, the process group's all-gather and reduce-scatter rings
  write step values under the indices of other ranks into their send neighbor's memory without waiting for it to reach
  the op (the broadcast rings only into ranks that have reached it), so a ring 16 ops later on another stream could tell
  a rank still in the op that a partial result was ready before it was (the all-gather ring, the only one of the two in
  use, writes a rank's index only once that rank is in the ring op: that rank has then finished the compile_op, which it
  does only once the reads from it are over).
  With writes (`MOODIST_COMPILE_OP_TRANSFER=write`), the CPU thread of a partial rank watches the flags of its
  `IbWrite`s instead: once one is set (and the destination's input list has arrived), it splits the write into parts
  like a read (about 128 KiB, round robin over the NICs from one that rotates per write) and queues them on their
  NICs; each NIC posts its queued parts up to the same limit of 8 parts (of reads and writes) in flight, and once its
  parts of a write have completed, it sets its words of the write's signal in the destination (see Signals below).
  The writes go before the reads in each round of the loop: other ranks' kernels wait for them. The second barrier
  then also tells a rank that nobody writes into its scratch tensor any more.
- Slot reuse (`WorkCustom`): the step values that ranks write into each other's memory per concurrency slot are
  reused by the op 16 later, of any kind, and some process group collectives wait for exact values in them, so a rank
  must not write there for a later op while another still waits in the earlier one: a rank writes its step value into
  another rank's memory only once that rank's input list for the same op has arrived (it is then done with the slot's
  previous op). The wait for all other ranks at the end of the op (for their arrival values, see below) keeps every
  rank within an op of every other, as when all ranks waited for each other's reads. The process group collectives
  need that: they write their lists into the memory of other ranks without waiting for them to reach the op (the
  all-gather ring also writes step values under the indices of other ranks, and keeps its neighbor's lists in list
  slots 0 and 1, which the other collectives use for ranks 0 and 1), which is safe only if those ranks have finished
  the slot's previous op by then; on one stream, the collectives that wait for all ranks (barriers, all-gathers,
  reduce-scatters, all-reduces) keep every rank within an op of every other too, but the peers alone would let a rank
  get any number of ops ahead of the ranks that it does not exchange with. Every rank writes an arrival value (its
  step value and op id) into the memory of the ranks that are not its peers, a few at a time while its reads go on,
  into words of their own (`Group::cpuArrivalValues`). Only `WorkCustom` reads them, and only with `>=`, so a rank
  can write them without knowing whether the other rank is done with the slot's previous op (which may be a process
  group collective waiting for an exact step value); a larger value is a later op's (that rank has finished this
  one). With the op's step value, the op ids must match (the peers compare theirs in their input lists).
- Signals (`writeRdmaSignal`): a kernel waits for the data of an RDMA transfer (a proxied transfer's read, a
  reduction's source from another NVLink domain) until its signal in `Group::rdmaSignalBuffer` is set: 16 bytes per
  signal index, a word per NIC, all of which must have the step value (+ the generation of the use of a slot of a
  ring; a credit, which follows no data, sets all words at once). A NIC's writes into GPU memory reach the GPU
  in the order the NIC issued them unless the later one is relaxed-ordered (PCIe lets a write pass earlier ones only
  with relaxed ordering, which the data's registrations allow, for speed), and a GPU whose GPUDirect RDMA writes
  ordering is "owner" (`CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING`; GB300) lets its kernels see them in
  that order. So each NIC that carried some of the data writes its word (and the words of the NICs next to it that
  carried none, see `signalWords`) once its data is in place: for a read, a loopback write once all parts have
  completed; for writes, a write to the destination once that NIC's parts have completed (by then the destination's
  NIC has received all of them, even if the network delivered them out of order, and issued their writes into its
  memory, which the signal's write cannot pass). The signal buffer has a registration of its own without relaxed
  ordering (`rdmaSignalMr`, `Rdma::regMrCuda`), on the same path as the data's: registered through Data Direct like
  the data, or like it without (`RdmaMr::dataDirect`). Elsewhere, or with `MOODIST_RDMA_FLUSH=1`, the CPU thread
  flushes the data first (`flushRead`): a read of one byte of it through every NIC that carried some (PCIe does not
  let a read pass the writes before it), a loopback read for the reads of a signal, a read of the destination for
  writes (on the same connection right after the last part, which the peer executes after them, `Rdma::inOrder`; on
  EFA, whose SRD queue pairs deliver out of order, once the parts have completed), then sets all words with one
  write. For writes, the destination decides: if its signals cannot follow the data (its registrations differ, its
  GPU does not order the writes, or it has `MOODIST_RDMA_FLUSH=1`), it asks the writers to flush (`Header::flags`).
  Signals without flushes rely on the NIC relaxing the order only of the writes into registrations that allow it: a
  NIC set to relax the order of all its writes needs `MOODIST_RDMA_FLUSH=1` on every rank. The reads without a signal
  (into CUDA tensors) are flushed once they have all completed, through each NIC that carried some (a read for each
  kind of registration that its data went into, through Data Direct or not), before the CPU thread lets the stream go
  on (`cpuOut`; it is one word, which the stream waits for). Data in CPU tensors needs no flush (the completion,
  written to host memory too, does not pass it).
- The kernel and the CPU thread run concurrently. Besides `cpuIn` and `cpuOut` on the stream, the CPU threads
  signal the kernel when the data of an RDMA transfer is in place or a slot of a ring is free (see Signals), and the
  kernel signals the CPU thread when it has stored an item or used the contents of a slot (done flags, `cpuInBuffer`
  slots 16 + index).

- NVLS ops (see the kernel): `customOp` passes the NVLS identities and multicast addresses of the tensors (in the
  communication pool, at logical offset o: (2 << 60) | o; else in moodist's pool, at offset o: (1 << 60) | o) and the
  op's step counter value (0 for reductions, which have none); an identity refers to the rank's NVLink domain's
  objects, which the launches compare within the domain. Block 0 of a launch that used NVLS (copies or reductions;
  with NVLS runs, a launch whose every run could) counts it in the group's counter area (the process group option
  `nvls_launches`; the read-only option `nvlink_domain` gives the first rank of the rank's domain, for tests). With
  `MOODIST_COMPILE_OP_PROFILE`, the launch that writes the kernel's profile does so once the CPU thread has the op's
  RDMA part (its kernel may wait for the transfers: writing it right after the launch deadlocked reductions across
  domains).
- Communication pools (`allocatorCommPoolCreate`; per group, `Group::getCommPool`; `moodist.comm_empty` and the
  process group's `allocate_tensor`): allocations get logical offsets that depend only on the sequence of
  allocations and frees of the pool, so that ranks that allocate the same buffers in the same order get the same
  offsets. The pool places them with the allocator's region machinery (best fit, lowest offset first) in logical
  coordinates; memory freed while streams other than the allocation's still used it (record_stream) carries an
  event per stream, and an allocation of it on another stream waits for those events on the GPU, not on the host
  (which would make the placement depend on timing). The memory comes from the allocator's pool in slabs of at least
  256 MiB (one 2 MiB aligned allocation each, at any address, never freed), each at its own logical range with a 2
  MiB gap before it, so that no free region spans two slabs. Without communication: a rank binds a new slab into its
  domain's multicast object when it adds it (before handing out memory in it), and the kernel's address exchange
  orders that before any peer's multicast stores to it.

## Environment variables

- `MOODIST_COMPILE_OP_GRAPHS` - the graph variants to tune over (see the compile phase). Same on all ranks.
- `MOODIST_COMPILE_OP_PROXY=0` - no proxy pass: every rank reads all its data from outside its NVLink domain
  itself. Same on all ranks.
- `MOODIST_FABRIC=0` - ignore a multi-node NVLink fabric: compile_op reaches other nodes over RDMA, as between
  separate NVLink domains. Same on all ranks.
- `MOODIST_IB_DATA_DIRECT=0` - ignore the Data Direct DMA functions of ConnectX-8 NICs (not specific to compile_op;
  see the RDMA bandwidth note under Performance).
- `MOODIST_DUMP_KERNELS` - write each op's final PTX to `moodist-compile-op-kernels-rank<rank>-op<id>.ptx`.
- `MOODIST_COMPILE_OP_CONFIGS="blockSize:sharedMemory:gridSize[:chunkSize[:pipelines]],..."` - the kernel
  configurations to tune over, instead of the default ones (see Tuning); e.g. `128:231424:16` for one 16-block
  configuration (which also skips the tuning of the others). For benchmarks and experiments. Same on all ranks.
- `MOODIST_COMPILE_OP_PROFILE=N` - kernel profiling (see Tracing).
- `MOODIST_COMPILE_OP_ROUNDS=1|2` - all-reduces across NVLink domains in one round, or in two wherever they can be,
  instead of as the cost model picks (`auto`, the default; see compile phase step 7). Same on all ranks.
- `MOODIST_COMPILE_OP_PHASES=1|2` - reductions in one NVLink domain in one pass, or in two phases (pieces of any
  size), instead of as tuning picks (`auto`, the default; see compile phase step 11). Same on all ranks.
- `MOODIST_COMPILE_OP_TRANSFER=read|write` - the partial results and shares of reductions across NVLink domains go
  over RDMA reads by the ranks that need them (`read`, the default), or writes by the ranks that compute them (see
  compile phase step 7). Same on all ranks.
- `MOODIST_COMPILE_OP_SLOTS=N`, `MOODIST_COMPILE_OP_SLOT_BYTES=N` - with reads, the slots per ring of the scratch
  tensors of reductions across NVLink domains (1 to 16; by default enough for about 4 MiB, or for 32 MiB / the rank's
  number of rings if that is more, from 2 to 16), and the most that a slot holds (a multiple of 4096; by default 2
  MiB), which also limits their parts (see compile phase steps 3 and 7), for tests: one slot of 4 KiB uses every slot
  many times. Same on all ranks.
- `MOODIST_COMPILE_OP_RING=auto|1|0` - with reads, reductions along a ring of the NVLink domains (see compile phase
  step 7) where they are faster (`auto`, the default: with 3 or more domains of the same size, a rank's one cell with
  one executor and sources in every domain, of 6 MiB or more), wherever they can be (`1`), or never (`0`). Same on
  all ranks.
- `MOODIST_COMPILE_OP_RING_GROUP=N` - the parts per group in the order of such a ring (1 to 16; by default 8), for
  tests and measurements. Same on all ranks.
- `MOODIST_RDMA_FLUSH=1` - the data of RDMA transfers is flushed before their signals even where the signals can
  follow it (see Runtime, Signals), for tests and comparisons. Read when the CPU thread starts; may differ between
  ranks (for writes, the destination's applies).
- `MOODIST_COMPILE_OP_MAX_EDGE_BYTES=N` - the largest edge (a multiple of 4096 up to 2 GiB; by default 1 GiB, see
  Phase 5), for tests: small values split regions into many edges. Same on all ranks.
- `MOODIST_COMPILE_OP_PARAM_BUFFER=1` - the kernel takes its address arrays from its parameter buffer, as ops too
  large for kernel parameters do (see The kernel), for tests.
- `MOODIST_COMPILE_OP_NVLS=0` - no NVLS (copies and reductions, see The kernel), for tests and comparisons. Read
  when an op is compiled; a rank that says no makes the op use none.
- `MOODIST_COMPILE_OP_NVLS_BLOCKS=N` - the blocks that do the NVLS reductions of every rank that does any (at most the
  grid; by default by each rank's share, see The kernel), for measurements.
- `MOODIST_IB_TIMEOUT` - the RDMA transport's ACK timeout exponent (not specific to compile_op; default 20, as
  NCCL's: on ConnectX-8, about 64 s of an unanswered request before the error that ends the job; see ib_common.cc).
- `MOODIST_TIMEOUT` - the process group's timeout in seconds (not specific to compile_op; it overrides the one that
  PyTorch gives the process group, by default 30 minutes): how long compile_op's exchanges, barriers and tuning
  launches wait for the other ranks.

## Tracing

`moodist.enable_profiling(True)` makes the CPU thread record Chrome trace spans; after
`moodist.enable_profiling(False)`, its next traced point writes them to `moodist-trace-cpu-<group>-<rank>.json` in
the current directory (so run one more op). For compile_op the spans are `WorkCustom`'s phases, named with the op's
step value: `start`, `wait dyn N` (waiting for the input list of peer N, or of the rank itself), `wait kernel` (for
the stream to reach the op), `step exchange`, `wait step N` (peer N's step value), `reads`,
`wait read and write completions`, `wait reads of N` (peer N still reading or writing), `wait arrival of N` (rank N,
not a peer, has not reached the op) and `wait flushes`, where only the ranks actually waited for appear; and each
NIC's reads and writes (`read N`, from rank N, and `write N`, to rank N). The CPU thread also records how long each
op waited from the call until the CPU thread picked it up, and its sleeps, stalls of 20 us or more and moves between
cores, with the core numbers. Timestamps are the system clock, so the files of all ranks can be merged into one trace
(align the nodes' clocks, e.g. on the end of `step exchange`, which a rank and its peers leave within a few
microseconds).

With `MOODIST_COMPILE_OP_PROFILE=N`, the kernel of every op compiled from then on records a timestamp at each
marker in `generateKernel` (`profiling.enter`: the phases of the kernel, such as the address exchange, the
address resolution, the copy loop and the exit sync), and the op's N-th launch synchronizes the device and writes
them to `moodist-trace-cuda-<group>-<rank>-op<id>.json`, one track per thread (pid = block). Timestamps are SM clock
cycles since the thread started, shown as microseconds. It costs 128 KiB of device memory per thread of the kernel.

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

(The `read` row predates the order of edges by the rank they read from (see Scheduling): on the same kind of nodes at
the end of October, `read` took 17 us, 26 us, 213 us and 3.24 ms (580 GB/s), against 15 us, 37 us, 657 us and
11.2 ms before it, and `multicast` 15 us, 27 us, 193 us and 2.81 ms.)

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
GPUs, and on 64 GPUs in two NVL72 domains (29.4 ms at 256 MiB then; 25.3 ms with the proxies' copies in the item
sequence, see Scheduling), ops do not use NVLS and take what they took before.

All-gather on 64 GPUs in two NVL72 domains of 32 (`bfloat16`, per-rank size, median; a GPU receives 63 pieces, the 32
from the other domain through the proxies of compile phase step 8), with the proxies' copies in rounds and in the
item sequence (see the kernel's Scheduling; November 2026, the same 16 nodes):

| | 64 KiB | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|---|---|
| rounds | 68 us | 79 us | 139 us | 479 us | 1.87 ms | 7.59 ms | 30.2 ms (560 GB/s) |
| item sequence | 51 us | 69 us | 141 us | 440 us | 1.61 ms | 6.28 ms | 25.3 ms (668 GB/s) |

(GB/s: what a GPU receives, per second.) On 128 GPUs in four domains of 32, 16 MiB took 3.71 ms in rounds and 3.20 ms
in the sequence, 64 MiB 14.8 and 12.7 ms, 256 MiB 58.7 and 49.6 ms (581 and 687 GB/s). A first version, in which a
pipeline waited for the relay's signal as soon as it took an item of the relay's copies (the stores of the items that
it had loaded waiting too), and the first claims took enough items to fill the rings as without waits, took 152 us at
1 MiB: of the 64 items, the first claims of half the 16 pipelines took the 32 of the rank's own input, those of the
other half the 32 of the chunk from the other domain, which they then waited for.

Reductions (`reduce="sum"`, `bfloat16`) on 32 GPUs in one NVL72 domain, against NCCL and moodist's process group
collectives (which reach the other nodes over RDMA). Reduce-scatter: the size is each rank's output (its input is 32
times as large); all-reduce: the tensor's size:

| | 64 KiB | 1 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|
| reduce-scatter, compile_op | 29 us | 69 us | 838 us | 3.26 ms | 12.94 ms |
| reduce-scatter, NCCL | 115 us | 137 us | 809 us | 3.23 ms | 12.30 ms |
| reduce-scatter, NCCL with `NCCL_MNNVL_ENABLE=0` | 316 us | 392 us | 1.88 ms | 5.69 ms | 22.2 ms |
| reduce-scatter, moodist's process group | 139 us | 236 us | 2.52 ms | 9.65 ms | 39.7 ms |
| all-reduce, compile_op | 30 us | 46 us | 72 us | 222 us | 937 us |
| all-reduce, compile_op, two phases from 128 MiB on (October 30) | 31 us | 46 us | 72 us | 226 us | 888 us |
| all-reduce, NCCL | 163 us | 132 us | 132 us | 334 us | 900 us |
| all-reduce, NCCL with `NCCL_MNNVL_ENABLE=0` | 84 us | 153 us | 974 us | 1.66 ms | 5.04 ms |
| all-reduce, moodist's process group | 158 us | 194 us | 321 us | 703 us | 2.12 ms |

The large reduce-scatters read from the other GPUs at 620-640 GB/s per GPU (16 blocks), as fast as the all-gather
writes. NCCL (`NCCL_DEBUG_SUBSYS=TUNING`) used NVLS (`multimem` reductions) for the all-reduce at every size, and
for the reduce-scatter at 64 KiB and 1 MiB only (ring from 16 MiB on). With `NCCL_NVLS_ENABLE=0` (ring; a rerun of the default in
the same session came within 3% of the table from 1 MiB on): reduce-scatter 110 us, 110 us, 811 us, 3.24 ms and
12.3 ms; all-reduce 149 us, 143 us, 169 us, 313 us and 1.05 ms. compile_op's all-reduce, without NVLS, is faster
than NCCL's without at every size, and was 4% slower than NCCL's with it at 256 MiB, before the reductions in two
phases (compile phase step 11). One pass / two phases on the same 32 GPUs (another session; tuning, the default,
picked the faster one at every size): 64 MiB 223 / 232 us, 128 MiB 470 / 437 us, 256 MiB 937 / 894 us, 512 MiB 1.86
/ 1.65 ms (558 and 629 GB/s); below 64 MiB two phases were 8-17% slower. On 16 GPUs: 64 MiB 226 / 227 us, 256 MiB
881 / 825 us.

With NVLS reductions (November 2026, see the kernel; tensors at the same offsets of the pool on every rank), on 32
GPUs of another NVL72 domain, against the reduce loop (the build before them) and NCCL in the same session
(`bfloat16`; `float32` takes the same time per byte):

| all-reduce | 64 KiB | 256 KiB | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|---|---|---|---|
| compile_op, NVLS | 18 us | 23 us | 21 us | 26 us | 51 us | 152 us | 551 us | 2.15 ms |
| compile_op, reduce loop | 31 us | 33 us | 46 us | 50 us | 71 us | 225 us | 842 us | 3.22 ms |
| NCCL | 132 us | 142 us | 130 us | 227 us | 223 us | 335 us | 907 us | 2.87 ms |

| reduce-scatter (each rank's output) | 64 KiB | 1 MiB | 16 MiB | 64 MiB |
|---|---|---|---|---|
| compile_op, NVLS | 20 us | 62 us | 742 us | 2.92 ms |
| compile_op, reduce loop | 29 us | 69 us | 840 us | 3.27 ms |
| NCCL | 213 us | 138 us | 800 us | 3.12 ms |

At 256 MiB, the all-reduce moves (1 + 1/32) times the tensor through each GPU's NVLink, at about 500 GB/s each way
(the loads and the stores of a warp do not overlap, see Known issues); the reduce-scatter sends each GPU's whole input
into the switches at 736 GB/s (2 GiB per GPU at 64 MiB), where the reduce loop read 31/32 of it at 636 GB/s.

Fewer GPUs (in the same domain; NVLS / reduce loop, buffers from the communication pool):

| | 8 GPUs | 12 GPUs | 16 GPUs |
|---|---|---|---|
| all-reduce, 64 KiB | 18 / 19 us | 18 / 22 us | 17 / 23 us |
| all-reduce, 16 MiB | 53 / 60 us | 56 / 66 us | 52 / 68 us |
| all-reduce, 256 MiB | 568 / 714 us | 606 / 801 us | 555 / 797 us |
| reduce-scatter, 64 KiB per rank | 17 / 18 us | 17 / 20 us | 18 / 21 us |
| reduce-scatter, 64 MiB per rank | 753 / 723 us | 1113 / 1131 us | 1473 / 1587 us |

With one destination (reduce-scatter, reduce to one rank), each GPU sends its whole input into the switches, N / (N -
1) times what the other ranks read with the reduce loop: 14% more on 8 GPUs, where it was 4% slower, hence NVLS for
those from 12 ranks on. A reduce to one rank (`bfloat16`, 32 GPUs): 64 KiB 19 / 29 us, 1 MiB 26 / 59 us, 16 MiB 170 /
699 us, 256 MiB 2.50 / 10.91 ms (NCCL 151 us, 157 us, 136 us, 0.63 ms; see Known issues); on 16 GPUs, 256 MiB 1.85 /
5.28 ms. Other types on 32 GPUs (all-reduce, NVLS / reduce loop): `float64` sum 64 KiB 19 / 55 us, 16 MiB 68 / 83 us,
256 MiB 818 / 859 us; `int64` sum 19 / 31 us, 69 / 70 us, 818 / 894 us; `int32` sum (four 4-byte loads per vector)
19 / 30 us, 1 MiB 29 / 46 us, 4 MiB 45 / 50 us, 8 MiB 69 / 53 us, 256 MiB 1554 / 894 us, hence NVLS for `int32` up to
4 MiB only (`int32` max took the same as its sum).

Reduce-scatter (`bfloat16`) on 64 GPUs in two NVL72 domains of 32 (across domains, see compile phase step 7), and
with `MOODIST_FABRIC=0`, i.e. 16 domains of one node:

| reduce-scatter, 64 GPUs | 64 KiB | 1 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|
| compile_op, two domains | 111 us | 154 us | 1.65 ms | 6.46 ms | 25.7 ms |
| NCCL | 419 us | 764 us | 2.24 ms | 6.51 ms | 25.0 ms |
| moodist's process group | 245 us | 486 us | 5.53 ms | 21.4 ms | 87.8 ms |
| compile_op, 16 domains | 124 us | 299 us | 3.19 ms | 12.4 ms | 47.5 ms |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 622 us | 853 us | 3.72 ms | 11.4 ms | 44.9 ms |

With two domains, the large sizes are bound by NVLink, as NCCL's: a GPU reads its domain's inputs twice, for the
partial result of its peer in the other domain and for its own output (655 GB/s per GPU, as in one domain). With 16
domains, by RDMA: a GPU reads 15 partial results of the size of its output, at about 85 GB/s at 256 MiB. Before the
orders of the partial results and the reads (step 7), the 16 domains took 443 us, 4.65 ms, 16.8 ms and 68.7 ms from
1 MiB on; with 1 MiB parts instead of 128 KiB (see the RDMA notes), 442 us, 3.99 ms, 13.9 ms and 50.7 ms. NCCL's
reduce-scatter on the 64 GPUs is a ring at every size, NVLS or not (`NCCL_NVLS_ENABLE=0`: 420 us, 747 us, 2.02 ms,
6.71 ms and 26.2 ms). Along rings of the domains (step 7, from 6 MiB per rank on), the 16 domains take 2.72 ms,
10.9 ms and 43.5 ms at 16, 64 and 256 MiB (see the ring table below).

All-reduce (`bfloat16`) on the same 64 GPUs (one session):

| all-reduce, 64 GPUs | 64 KiB | 1 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|
| compile_op, two domains | 111 us | 96 us | 123 us | 261 us | 846 us |
| NCCL (`NVLS_TREE` from 1 MiB on) | 151 us | 423 us | 408 us | 738 us | 1.24 ms |
| NCCL with `NCCL_NVLS_ENABLE=0` (tree and ring) | 140 us | 210 us | 311 us | 596 us | 1.73 ms |
| compile_op, 16 domains, one round | 101 us | 153 us | 872 us | 3.20 ms | 12.5 ms |
| compile_op, 16 domains, two rounds (the default from 1 MiB on) | 137 us | 143 us | 258 us | 607 us | 1.91 ms |
| NCCL with `NCCL_MNNVL_ENABLE=0` | 140 us | 193 us | 1.17 ms | 1.99 ms | 5.40 ms |

With two domains, an executor computes its domain's partial result of its piece (reading 1/32 of the tensor from
each GPU of its domain), reads the other domain's over RDMA (1/32 of the tensor) and stores the sum to the 32 GPUs of
its domain: the NVLink traffic of an all-reduce in one domain, in two phases, and the RDMA in between, in 1 MiB parts
(Phase 2). The phases take less time than the single pass of one domain, where every item reads its 32 sources and
stores to 32 destinations: 846 us against 937 us on 32 GPUs in the same session at 256 MiB, with the same grid of
24 blocks (the NVLink-only tuning launches: 10.0 against 11.2 ms for 12), which suggests the same split within one
domain (see Reductions: next steps). NCCL's choice, `NVLS_TREE`, is slower than its tree and ring from 1 MiB to
64 MiB. With 16 domains in one round, every GPU reads 15 partial results of a quarter of the tensor over RDMA, 3.75
times the tensor, at about 80 GB/s at 256 MiB, and the large sizes took 1.6 (64 MiB) to 2.3 times (256 MiB) as long
as NCCL's (moodist's process group: 2.58 ms at 256 MiB with two domains). In two rounds (compile phase step 7), a
GPU reads 15/16 of a quarter of the tensor twice: 1.91 ms at 256 MiB, 2.8 times faster than NCCL, and 6.5 times
faster than one round; the second round costs about 35 us at small sizes. With two domains, one round is faster at
every size (two rounds, forced: 125 us, 135 us, 163 us, 310 us and 988 us): both read as much, and the second round
adds a notification and a read.

With NVLS within each domain (November 2026, NVLS runs, see the kernel; buffers from the communication pool, so that
every rank's tensors are in place; `bfloat16`, median, one session), on the same 64 GPUs in two domains of 32:

| two domains of 32, NVLS / without (`MOODIST_COMPILE_OP_NVLS=0`) | 64 KiB | 1 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|---|---|
| all-reduce | 58 / 68 us | 62 / 85 us | 109 / 117 us | 266 / 284 us | 710 / 851 us | 2.39 / 3.28 ms |
| reduce-scatter (per rank) | 65 / 70 us | 159 / 158 us | 1.47 / 1.66 ms | 5.81 / 6.50 ms | | |
| reduce to one rank | | 79 / 104 us | 261 / 736 us | 1.01 / 2.80 ms | 3.95 / 10.83 ms | |

An all-reduce's executor computes its domain's partial result of its piece with `multimem.ld_reduce` (each GPU's
input goes into the switches once) and stores the results with multicast stores (each GPU receives them once): the
NVLink traffic of an NVLS all-reduce in one domain, which takes 551 us at 256 MiB there; across the domains, the
partial results go over RDMA in between (see Known issues, Performance, for what the rest costs). With the partial
results of the executor's own domain in bursts of 8, as without NVLS, 256 MiB took 780 us (all loads, then all
stores); interleaved (two uses ahead, compile phase step 7), 705-710 us. A reduce-scatter's executor combines its
domain's inputs with `multimem.ld_reduce` (and the partial results for the other domain's executors are computed the
same way): its egress, its whole input for every executor, was the limit before too. A reduce to one rank takes its
domain's sources with `multimem.ld_reduce` instead of reading them all, and is then bound by its RDMA read of the
other domain's partial result (64 MiB in 1.01 ms, about 66 GB/s through the root's NIC). Across two domains of 8 GPUs
(2 nodes each), reductions use no NVLS (partial results with `multimem.ld_reduce` from 12 ranks per domain on, results
with multicast stores alone from 16, see NVLS runs; the multicast stores of an all-reduce's results took 874 us at
256 MiB, 772 us without).

Since `WorkCustom` exchanges input lists and waits for reads with the op's peers only (see Runtime; before, with all
ranks: each op posted about 190 small RDMA writes per rank on 64 GPUs, which kept the CPU thread busy for most of a
small op), small ops across domains are faster (`bfloat16`, median, the previous build in the same session in
parentheses): with two domains of 32, all-reduce 64 KiB 63 us (106), 1 MiB 82 us (101), 4 MiB 87 us (118),
reduce-scatter 64 KiB 68 us (114); with 16 domains of 4 (`MOODIST_FABRIC=0`), all-reduce 64 KiB 79 us (101), 1 MiB
119 us (144), 4 MiB 144 us (170), 16 MiB 245 us (263), reduce-scatter 64 KiB 86 us (110). Larger ops take about
the same time (two domains: all-reduce 64 MiB 241 us, 256 MiB 852 us; reduce-scatter 1 MiB 157 us, 256 MiB 25.7 ms),
except with 16 domains at 256 MiB: the all-reduce is 1.7% faster (1.91 ms), the reduce-scatter 1.5% slower at the
median (47.9 ms instead of 47.2; the fastest 10% 46.5 ms instead of 44.4, the slowest 10% 55 ms instead of 63). The
barrier with all ranks at the end of each op started the next op's reads on all ranks at once, which their order
(every source one reader per step) needs: with a temporary switch back to it for the final barrier, the
reduce-scatter took 46.4 ms and the all-reduce 1.94 ms.

The cost model (step 7) against `MOODIST_COMPILE_OP_ROUNDS=1` and `2` (`bfloat16` all-reduce, k domains of 4 GPUs
with `MOODIST_FABRIC=0`; one round / two rounds, in us; the model's crossover in the last column):

| k | 1 MiB | 2 MiB | 4 MiB | 8 MiB | 16 MiB | 64 MiB | 256 MiB | two rounds from |
|---|---|---|---|---|---|---|---|---|
| 3 | 53 / 78 | 61 / 94 | 79 / 98 | 123 / 137 | 207 / 186 | 482 / 409 | 1614 / 1366 | 12 MiB |
| 4 | 59 / 82 | 69 / 100 | 91 / 109 | 145 / 127 | 256 / 200 | 667 / 442 | 2340 / 1487 | 5.3 MiB |
| 8 | 91 / 93 | 106 / 111 | 156 / 156 | 262 / 166 | 475 / 204 | 1483 / 516 | 5561 / 1650 | 1.5 MiB |
| 16 | 152 / 143 | | 306 / 168 | | 877 / 258 | 3180 / 607 | 12500 / 1907 | 640 KiB |

The model picks the faster variant everywhere but with 8 domains at 2 MiB (5% slower than one round).

Reads against writes (`MOODIST_COMPILE_OP_TRANSFER`, step 7) and signals that follow their data against flushes
(`MOODIST_RDMA_FLUSH=1`, see Runtime), 16 domains of 4 GB300 with `MOODIST_FABRIC=0` (`bfloat16`, one build):

| 16 domains of 4 | AR 1 MiB | AR 16 MiB | AR 64 MiB | AR 256 MiB | RS 4 MiB | RS 16 MiB | RS 64 MiB | RS 256 MiB |
|---|---|---|---|---|---|---|---|---|
| reads | 144 us | 263 us | 604 us | 1.95 ms | 872 us | 3.17 ms | 12.5 ms | 46.2 ms |
| reads, flushes | 152 us | 271 us | 650 us | 2.11 ms | 894 us | 3.18 ms | 12.5 ms | 46.2 ms |
| writes | 139 us | 230 us | 531 us | 1.89 ms | 824 us | 3.28 ms | 17.8 ms | 89.9 ms |
| writes, flushes | 166 us | 261 us | 574 us | 2.04 ms | 841 us | 3.34 ms | 15.7 ms | 90.2 ms |

The reads take what they took before the signals and flushes existed. A flush costs a stage of transfers about
5 us with reads (a loopback read) and 10-15 us with writes (a read over the network), and an all-reduce in two
rounds, whose rounds wait for each other's signals, up to 8% at 64-256 MiB. Writes save the notification and the
executor's CPU thread: 3-13% for all-reduces across 16 domains from 16 MiB on, 5% for reduce-scatters of 1-4 MiB.
Large reduce-scatters are much slower with writes: every GPU sends 15 partial results the size of its output (3.75
GiB at 256 MiB), in a rotation that gives every receiver one sender at a time as long as the ranks stay in step.
With writes, ranks out of step put two senders on one receiver, whose switch port then queues, and the NICs' congestion
control cuts both senders' rates (14.4 million ECN-marked packets per 20 ops at 256 MiB, against 1.45 million with
reads), which the transfers that follow feel too: 3% slower at 16 MiB, 20-40% at 64 MiB, twice as slow at 256 MiB.
With reads, two readers on one source share its link, and nothing queues in the network. More writes in flight per
NIC (2 MiB instead of 1) brought 64 MiB level with reads, but left 256 MiB at 79 ms. On 64 GPUs in two domains of 32,
reads and writes took the same time from 16 MiB on (an all-reduce of 256 MiB 847 us, a reduce-scatter 25.7 ms).

Reductions along rings of the domains (step 7) against the reads from every domain at once
(`MOODIST_COMPILE_OP_RING=1` / `0`), `bfloat16`, k domains of 4 GB300 with `MOODIST_FABRIC=0`, size per rank
(groups of 8 parts; ring / reads from every domain, in us):

| | 1 MiB | 2 MiB | 4 MiB | 6 MiB | 8 MiB | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|---|---|---|---|---|
| reduce-scatter, k = 4 | 151 / 102 | 207 / 141 | 252 / 227 | 295 / 334 | 343 / 414 | 598 / 768 | 2369 / 2914 | 9417 / 11360 |
| reduce-scatter, k = 8 | 286 / 165 | 429 / 250 | 516 / 432 | 612 / 608 | 708 / 789 | 1310 / 1513 | 5212 / 5774 | 20838 / 22823 |
| reduce-scatter, k = 16 | 605 / 319 | 908 / 488 | 1045 / 866 | 1275 / 1256 | 1454 / 1672 | 2726 / 3177 | 10869 / 12271 | 43454 / 48889 |
| reduce to rank 0, k = 4 | 106 / 74 | 151 / 113 | 183 / 180 | | 242 / 312 | | 898 / 2124 | |
| reduce to rank 0, k = 8 | 156 / 121 | 201 / 201 | 231 / 353 | | 277 / 654 | | 957 / 4881 | |
| reduce to rank 0, k = 16 | 422 / 223 | 620 / 395 | 651 / 717 | | 694 / 1365 | 705 / 2658 | 998 / 10414 | 3119 / 41423 |

In a reduce-scatter, every GPU reads k - 1 partial results of the size of its output either way: along the rings
from one source (93 GB/s at 256 MiB, without ECN-marked packets), otherwise from several at once (82 GB/s, with
congestion marks). But the ring's k - 1 steps go one after the other, each with a notification and a read (about
40 us per step at 1 MiB), which takes longer at small sizes: the ring is 2-4% slower at 5 MiB per rank, within 2% at
6 MiB (12% faster with 4 domains), and `auto` takes it from 6 MiB on. To one rank, the reads from every domain at
once all go into the one GPU, (k - 1) times its output, while the ring's steps are spread over k - 1 GPUs: faster
from 4 MiB on with 8 and 16 domains (from 8 MiB with 4), 13 times faster at 256 MiB with 16 domains (only the GPUs
at the executor's position work; see Reductions: next steps). Also from 6 MiB on, for simplicity.

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
- Failures inside the collective steps of the compile phase that only one rank hits (e.g. running out of memory for
  the tuning's scratch tensors) are raised by that rank only. The other ranks wait for it until the process group's
  timeout (`MOODIST_TIMEOUT` overrides it): waiting in an exchange (`CompileContext::receive`), they then raise an
  error, after which the group's compile_op raises too (its exchanges are out of step); waiting in a barrier or for
  the tuning launches, they end the process instead (as after running out of memory for the scratch tensors: the
  others wait in the tuning's barrier). The errors that step 1 lists are raised on all ranks. The failing rank could
  tell the others through the queues, so that they fail at once.
- The process group's own collectives (all-gather, reduce-scatter, broadcast and reduce in cputhread.cc) signal their
  kernels as soon as the completions of their RDMA reads arrive (`writeCpuOut`), through the first NIC into memory
  registered with relaxed ordering, so the signal may land before data that another NIC wrote, or that the same NIC
  wrote before it (a write with relaxed ordering may pass earlier ones; compile_op's signals follow their data, see
  Runtime).
- `computeTarget` returns `sm_100a` for every compute capability from 10 up except 10.3, so devices such as sm_110
  or sm_120 get a target their driver cannot load. The copy kernel does not seem to need arch-specific (`a`)
  instructions, so a plain target would do.
- The kernels' PTX is compiled by the driver's JIT, whose ptxas is that of the driver's CUDA version. The ptxas of
  CUDA 13.0 (driver 580) drops the predicate of a `multimem.st` or `multimem.red` under a PTX predicate: the
  instruction always executes, and the comparison is removed as dead code (13.2 keeps it; seen in November 2026 with a
  predicated multicast store of the NVLS reductions, which then stored past the ends of edges: all-reduces of 4-64 KiB
  overwrote other pieces on every rank, reduce-scatters wrote past their outputs). Under a branch, which it turns
  into a predicated instruction itself, they work: the kernel has no multimem instruction under a PTX predicate (see
  the helpers in ptx.h). CUDA 13.2 Update 2 also fixes a compiler bug present since 12.8 (stale registers after
  failed reconvergence in kernels with nested divergence, which the kernels have), which drivers before 595.71 have.
  Probably that (or the like) hung a variant of the reduce loop (November 2026, not kept) whose thread 0 alone
  counted an item's done flags after the item's last block barrier: every all-reduce of 64 MiB across 2 domains of
  32 GB300 hung in the tuning launches of its 24-block setting, and with a `bar.warp.sync` after thread 0's code
  never; the reduce loop now has one there (see Reductions in the kernel). Compiling the PTX with a ptxas of moodist's
  own (the nvPTXCompiler library of a known version) instead of the driver's would make the compiler independent of
  the driver.

### Performance
- NVLS (see the kernel and the all-gather table under Performance notes). Open: the rank threshold (16; one node of
  4 GPUs lost from 4M on, 8 and 12 are not measured; 16 ranks lose 1-2% from 64 MiB on) could become a tuning
  candidate instead (one block, against the unicast grids with their block penalty), which needs tuning buffers from
  the communication pool. The switch delivers each multicast chunk to its sender too: a multicast store reaches
  exactly the members of the object that have memory bound at its offset (a member without, sender or receiver, is
  skipped, without an error; GB300, October 2026), so one object per sender, bound by every rank but the sender,
  would skip it, the local copy then done by the idle blocks. Further: groups that span several NVLink domains (one
  multicast object per domain, for the edges within a domain, including the proxies' stores of what they read: 64
  GPUs in two domains take 25.3 ms at 256 MiB with unicast stores, about what the GPUs' NVLink ingress allows, so
  there NVLS would mostly save SMs), destination sets
  smaller than the group (which ranks bind a range decides who receives it), and ranks that store different numbers
  of chunks (the window assumes the same).
- NVLS reductions (see the kernel and the tables under Performance notes). Open: a reduce to one rank combines
  everything on the root, whose rate depends on its loads in flight, which its blocks limit (at most the grid, tuned
  for the reduce loop; the grid must be the same on every rank): on 16 GPUs, 256 MiB took 1.82 ms with the tuned
  grid, 3.46 ms with 8 blocks, 1.30 ms with 32 and 1.29 ms with 64 (about 208 GB/s from about 1 MiB in flight),
  against 5.28 ms with the reduce loop (on 32 GPUs with the tuned grid 2.50 ms, against 10.91 ms; NCCL 0.63 ms). More
  vectors in flight per warp on the root would reach that rate with any grid; splitting its cell among the sources,
  each computing a piece and storing it to the root (see Reductions: next steps, item 5), would use every GPU's rate
  (deferred: reductions to one rank are not a common collective). The blocks (by rule, from the probe; tuning them
  needs tuning buffers from the communication pool, as above), the overlap of loads and stores (an all-reduce of
  256 MiB on 32 GPUs, in the probe: loads alone 381 us, stores alone 391 us, both 551 us; a warp loads an item, then
  stores it). The thresholds (8 ranks, 12 with one destination; `int32` up to 4 MiB) were measured on GB300 only, the
  12 on reduce-scatters, where NVLS sends each GPU's whole input into the switches: reductions to one rank on 8 to 11
  ranks, where it saves the root's ingress, were not measured, nor reductions across domains of 8 to 31 GPUs.
- NVLS across NVLink domains (NVLS runs, see the kernel and Performance notes). Open: an all-reduce across 2 domains
  of 32 GB300 takes about 705 us at 256 MiB, against 551 us in one domain with the same NVLink traffic per GPU (each
  GPU's input into the switches once, the results out once). A kernel profile (`MOODIST_COMPILE_OP_PROFILE`, rank 0,
  24 blocks) had the blocks wait for sources about 256 us each (the other domain's partial results, which the RDMA
  read brings 1 MiB at a time once the other domain's kernel has stored all of a part, and the last parts' at the
  end), load with `multimem.ld_reduce` for about 210 us (3 to 65 us per 32 KiB item, the GPUs' egress shared by every
  rank's loads), and wait for stores for about 130 us. More blocks did not help (16: 684 us, 32: 680, 48: 728, 64:
  720), smaller parts a little (256 KiB parts with the items 6 uses ahead: 672 us; 64 MiB unchanged). To try: blocks
  of their own for the items without source waits (the partial results, which then never wait behind an edge that
  waits for the other domain), or items interleaved finer than whole parts. Reductions to one rank across domains are
  now bound by the root's RDMA read of the other domain's partial result (64 MiB in 1.02 ms: one NIC); a ring of
  NICs, or every GPU of the domain reading a piece, would use more.
- Unicast stores to outputs at the same offset on every rank are 2-4% slower at large sizes than to outputs at
  different offsets (all-gather on 32 GPUs, see the NVLS table), maybe because all copies of a chunk then have the
  same address bits; not investigated.
- Compile time is mostly the JIT of the tuning candidates (4 per graph variant). JIT on several threads would cut
  it (the PTX differs per rank, so the work cannot be split across ranks).
- With waits for other edges (the ring graphs, and in the `read` graph the copies from a proxy's node), the block
  copies one edge at a time and drains its pipelines at the end of each edge, and a dependency is per block and
  edge. Per-chunk ready flags would let the pipelines take chunks across edges as without waits (as the copies that
  wait only for relay edges do, see Scheduling).
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
- `WorkCustom` still sends an arrival value to every rank that is not its peer per op (on 64 GPUs, about 60 of the
  about 70 small RDMA writes of an op), so that every rank stays within an op of every other (see Runtime). They go
  out while the reads go on, off the critical path, but keep the CPU thread busy. The process group collectives only
  need every rank to be done with an op before another starts the op 16 later (in the same concurrency slot):
  arrival values every few ops, with a wait for an earlier op's, would save most of them, but let the ranks drift
  further apart, which the read order does not like (how much the ops need them for speed has not been measured).
- Ranks that call different ops at the same point (a bug in the caller) usually fail with "Mismatched custom
  operation" if both ops are compile_op ops (the peers compare the op ids in their input lists, the other ranks in
  their arrival values), but can also wait forever: two ranks of which only one counts the other as its peer wait for
  each other (one for an input list, which goes only to peers, the other for an arrival value, which goes only to the
  other ranks), as do the ranks of an op with RDMA for those of one without (which run no `WorkCustom`), and a rank
  that waits forever compares no further op ids. A compile_op op and a process group collective wait for each other
  forever: the input lists of compile_op have words of their own, so neither sees the other's (before, a rank in a
  compile_op op that waited for a peer's list failed on a direct collective's list in that peer's slot, and on a
  ring's if the peer was rank 0 or 1). `WorkCustom` could watch the process group's list slots while it waits: a list
  there with the op's step value means a mismatch.
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
- Code layout (no longer visible since October 2026): small ops across two NVL72 domains (64 GPUs, all-reduce of
  64 KiB to 4 MiB) depended on the build's code layout while `WorkCustom` exchanged with all ranks. 9c67f3a's build
  took 91 us at 64 KiB (median; p90 100). Every build of it with code shifted took 102-109 us (p90 about 120; the
  fastest 10% the same): a `nop` in `RdmaIb`'s constructor, 8 bytes of padding at the end of `RdmaIb` with an
  initializer (one store more), a member more in `CustomOpDescriptor` or in `Rdma`, unused code in the CPU thread's
  loop, or the same source built with `-falign-functions=64`. Data layout alone made no difference (the same padding
  without an initializer, which leaves the code as it is; every malloc rounded up to whole cache lines by an
  LD_PRELOAD shim, in both builds), nor did a changed log message: 9c67f3a's build was the lucky one. In CPU-thread
  traces of the two, the slow build's phases that post RDMA writes took 6-7% longer (`start`, which sent the input
  list to all 64 ranks: 18.9 instead of 17.9 us; `step exchange`, the step value to the 63 others: 17.0 instead of
  15.9 us), and its waits for the last rank of a barrier 2-3 times as long. Each op posted about 190 small RDMA
  writes per rank, which kept the CPU thread busy for most of the op, so a few percent of its speed became 20% of the
  op's time (`perf` is not usable on these nodes, so the code that the layout slowed down is unknown). With the
  exchanges limited to the op's peers (see Runtime), builds that differ by a `nop` take the same time (63 us at
  64 KiB, p90 67).
- Small reductions across many NVLink domains: on 256 GB300 in 64 domains of 4 (`MOODIST_FABRIC=0`), an all-reduce
  of 64 KiB takes 316 us and one of 1 MiB 439 us, against NCCL's 161 and 263 us (`NCCL_MNNVL_ENABLE=0`); with 16
  domains of 4, 79 and 119 us. In one round, every executor reads 63 partial results, every partial rank notifies 63
  executors, and the step exchange and the barrier at the end of the op go to about 66 peers, all posted one after the
  other by the CPU thread. In a CPU-thread trace of an op of 64 KiB (393 us per op while traced), the step exchange
  took 27 us, the wait for the peers' step values 22, the 62 notifications 49, the 63 reads 177 (mostly waiting for
  their notifications), the completions 34 and the barrier with the peers 21, all of which grow with the number of
  domains. Rings of the domains do not help small ops: a hop (a kernel stores an item, its CPU thread notifies the
  reader, whose CPU thread reads it and signals its kernel) takes about 30-40 us (the reduce-scatter rings under
  Performance notes, at 1 MiB per rank), and an all-reduce along a ring of 64 domains would take 2 x 63 of them in a
  row. Fewer hops and fewer transfers per rank would: recursive doubling among the executors of a piece (6 rounds with
  64 domains), or exchanges within groups of 8 domains (2 rounds of 7 transfers per executor), with each hop an RDMA
  write of the data and its signal rather than a notification and a read. Which to use from which size on is a
  question for the tuning (see Tuning the RDMA side).

### Cleanup
- `docs/api/compile_op.md`: the examples predate `TensorRegion`'s `device` field.

### Mixed NVLink and RDMA paths: next steps
The proxy pass (compile phase step 8) sends data needed by several ranks of an NVLink domain over the network once,
in chunks for large transfers. (Before the edge-based model, `computeNvlinkPlan` in processgroup.cc, removed in
c57c8f4, computed a similar plan with gateway reads and local copies, but it was never executed.) Next:
1. The write variant: the source writes the data and the signal into the proxy's memory (as reductions across
   domains can, see compile phase step 7).
2. A ring for the RDMA part where the op allows it, as the process group's all-gather does (each rank reads from one
   fixed neighbor, which relays the data of the ranks before it): one source per rank, and its two neighbors as its
   only peers in `WorkCustom` (see Runtime). It needs a heuristic for when it applies, and measurements against the
   proxy pass. Reductions across domains have rings for cells with one executor (compile phase step 7: reduce-scatters
   9-22% faster from 8 MiB per rank on, with 4 to 16 domains), and a ring has no incast (with writes, incast makes
   large reduce-scatters twice as slow, see Performance notes).

### Reductions: next steps
1. Rings for writes (compile phase step 7): with writes, every item and every write still has a region of its own,
   so the scratch tensors hold everything that crosses domains (a reduce-scatter across k domains about 2 (k - 1)
   times a rank's output). Rings would work for writes too: a writer knows when its own slot is free (once its write
   has completed), but the receiving rank's slots need credits back to the writer (its kernel has used them). In two
   rounds, the copies of other shares could also read them into the outputs instead (the RDMA read into the
   executor's own output, the kernel copying it to the others of its domain).
2. Across NVLink domains (compile phase step 7): reads or writes (`MOODIST_COMPILE_OP_TRANSFER`, reads by default).
   Writes are faster for all-reduces across many domains from 16 MiB on and much slower for large reduce-scatters (see
   Performance notes); which to use when (e.g. writes for all-reduces in two rounds) is open. In two rounds, owners
   could write their shares straight into the other owners' outputs instead of their scratch tensors. With 16 domains,
   the reduce-scatter along rings is 27%, 5% and 3% faster than NCCL at 16, 64 and 256 MiB (above; NCCL from an earlier
   session). Small reductions across domains take 63 us on 64 GPUs in two domains (all-reduce of 64 KiB; 79 us with 16
   domains of 4), against 30 us in one domain of 32 (two rounds about 35 us more). The cost model of two rounds
   (Performance notes) picks two rounds a little early with 8 domains; its parts are those of one round (Phase 2), which
   were not tuned for two rounds. CPU tensors (an error): the CPU thread could combine them (`CpuReductions` in
   cputhread.cc, which the process group's CPU reduce-scatter uses).
3. All-reduce in one domain in two phases (compile phase step 11): 5-11% faster from 128 MiB on 32 GPUs, but the
   copies of a piece wait for all of it, and the kernel computes every piece before it copies any. In parts (as
   across domains, Phase 2), the copies of the first parts could overlap the computation of the last ones (across
   two domains, with 1 MiB parts, 846 us at 256 MiB on 64 GPUs; in two phases in one domain of 32, 888 us).
4. NVLS: done in one domain (November 2026, see NVLS reductions in the kernel) and within the domains of
   reductions across domains (one multicast object per domain; see NVLS runs in the kernel; open items under Known
   issues, Performance).
5. Reduce to one rank: the root reads every input itself (pieces only split cells with several destination
   ranks). Splitting such cells among the sources (each computing a piece and writing it to the root) would bring
   the root's NVLink ingress from (N - 1) times to about twice the data. With NVLS, the root's ingress is the result
   only, but it combines all of it alone, limited by its loads in flight (see Known issues, Performance): split, every
   GPU would combine a piece.
6. The process group's reduce-scatter and all-reduce on compile_op (and its all-gather), e.g. for FSDP2: needs ops
   parameterized by size (compiling takes 1-2 s, once per size), and `avg` (FSDP2's default `ReduceOp.AVG` is a
   fatal error in moodist's process group reduce-scatter today).
7. Rings of the domains (compile phase step 7) where there are none yet: a reduce to one rank uses only the GPUs at
   the executor's position (one ring of k - 1 steps; a ring per position, each for a part of the cell that a rank of
   the executor's domain then writes to it over NVLink, would use every NIC); ranks with several cells with one
   executor (e.g. two reduce-scattered tensors: the steps of several cells in one ring); domains of different sizes;
   writes (item 1). All-gathers, and the proxy pass, could go along rings too (see Mixed NVLink and RDMA paths), not
   all-to-alls (every pair of ranks exchanges different data).
8. Fewer connections: every rank connects to every rank, with two queue pairs per peer and NIC (send queues of 2048
   and 256 work requests; setting them up takes 0.8 s with 64 ranks of 4 NICs, 0.55 s before the second), but a rank
   of a ring exchanges with its two neighbors (and its own domain) only, and the reductions across domains with the
   ranks of its rail. At large world sizes, connecting only the pairs of ranks that an op needs (when it is compiled)
   would save most of them.

### Tuning the RDMA side
The tuning (compile phase step 12) times kernels only, without the CPU thread. How the data crosses NVLink domains is
decided by rules: one round or two (the cost model of `reductionPieces`, 5% off with 8 domains at 2 MiB), rings of the
domains or reads from every domain (from 6 MiB per rank, with 3 or more domains of the same size), reads or writes
(`MOODIST_COMPILE_OP_TRANSFER`: reads unless set, though writes are 3-13% faster for all-reduces across 16 domains from
16 MiB on), one source at a time or 8 (`readSources`), and fixed sizes (the proxies' chunks, the parts, the slots of the
rings). Each choice has a crossover that depends on the number and the sizes of the domains, the sizes of the op and
the hardware, and more choices are coming (e.g. small all-reduces across many domains in fewer hops, see Performance).
A second stage of the tuning could time them instead (discussed in November 2026, not built):
- Stage 1, as now: the kernel configuration of each graph, NVLink only. The two stages are mostly independent: a plan
  of the RDMA side has graphs of its own, whose kernels are tuned for it. Plans other than the default could try fewer
  configurations (the default's grid size and its neighbors).
- Stage 2: each candidate plan runs as a real op (its kernel, the CPU thread, RDMA) on scratch tensors of the op's
  shapes, a few calls to warm up (which absorb the skew between the ranks, e.g. 60-140 ms out of a barrier on 256
  GPUs), then timed calls. The ranks exchange their medians as in the kernel tuning, the slowest rank's time counts, and
  a plan other than the default has to win by a few percent, so that noise does not decide (stalls of the CPU thread
  of 0.2-1 ms are common on 256 GPUs).
- The cost is in compiling the plans (their exchanges and kernel JIT: an op takes 2.6-4.6 s to compile on 256 GPUs),
  not in timing them (10 calls of an all-gather of 256 MiB per rank take 0.25 s). So rules decide what is tuned: each
  plan says where it applies (sizes, numbers and sizes of domains), a cost model ranks the plans, and only those within
  about 25% of the best estimate are compiled and timed; far from a crossover, nothing is. The decisions can be cached
  per op signature and topology, in the process and possibly in a file.
- Plans that combine the values in a different order (rings, recursive doubling) give different bits in 16-bit types.
  That is acceptable: the reductions where precision matters are mostly in `float32`, and the least precise plan is as
  precise as NCCL's or more.
- A compiled candidate keeps everything that it allocated (moodist frees nothing): signals, notification slots, kernel
  modules, and NVLS step counters, of which a group has few. Fine with two or three candidates per op, but candidates
  should not take NVLS counters.
- First use: the small all-reduces across many domains in fewer hops, against one round near their crossover; then
  reads against writes, rings against reads, one round against two.

### Other features
- Unaligned copies (align down, shift the bytes, handle the unaligned tail; needs `fence.proxy.async.shared::cta`
  between generic shared memory writes and TMA stores) or a staging buffer, to lift the 16-byte requirement.
- TMA tensor maps for strided regions, instead of copies of non-contiguous pieces.
