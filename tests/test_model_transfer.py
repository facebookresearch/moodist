"""
Tests for model transfer between trainer and worker process groups.

This tests the scenario where:
- Trainers have model parameters with FSDP-style sharding (Shard)
- Workers have the same parameters with different sharding (may be Replicate or different Shard)
- Parameters are transferred from trainers to workers

The test runs with multiple trainer/worker ratios to exercise different configurations.
"""

from datetime import timedelta
import hashlib
import random

import moodist
from moodist import TensorRegion
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from framework import TestContext, test

moodist.enable_cpu_allocator()

# Ratios of trainers to total world_size
TRAINER_RATIOS = [0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9]

# Seed for deterministic assignment of trainer shards to workers
ASSIGNMENT_SEED = 42

# Number of parameters to batch per compile_op call
COMPILE_OP_BATCH_SIZE = 8


def _cleanup_distributed():
    """Clean up PyTorch distributed state."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _compute_split(world_size: int, trainer_ratio: float) -> tuple[int, int]:
    """Compute (num_trainers, num_workers) for a given ratio."""
    num_trainers = round(trainer_ratio * world_size)
    # Clamp to ensure at least 1 trainer and 1 worker
    num_trainers = max(1, min(world_size - 1, num_trainers))
    num_workers = world_size - num_trainers
    return num_trainers, num_workers


def _name_hash(name: str) -> int:
    """Compute a deterministic hash from parameter name."""
    # Use first 6 hex digits for a manageable offset (0 to ~16 million)
    return int(hashlib.md5(name.encode()).hexdigest()[:6], 16)


def _make_deterministic_tensor(
    name: str, shape: tuple, device: str, dtype: torch.dtype
) -> torch.Tensor:
    """Create a deterministic tensor based on parameter name."""
    offset = _name_hash(name)
    numel = 1
    for s in shape:
        numel *= s
    return torch.arange(0, numel, device=device, dtype=dtype).view(shape) + offset


def _compute_shard_for_rank(
    shape: tuple[int, ...],
    placements: list,
    rank: int,
    group_size: int,
) -> tuple[list[int], list[int]]:
    """Compute the offset and shape of the shard for a given rank.

    Args:
        shape: The global tensor shape
        placements: List of placements (e.g., [Shard(0)])
        rank: The rank within the group
        group_size: The size of the group

    Returns:
        (offset, local_shape) where offset is the starting index in each dim
        and local_shape is the size in each dim for this rank's shard.
    """
    offset = [0] * len(shape)
    local_shape = list(shape)

    for p in placements:
        if isinstance(p, Shard):
            dim = p.dim
            nelem = local_shape[dim]
            chunk_size = (nelem + group_size - 1) // group_size
            o = min(chunk_size * rank, nelem)
            n = min(chunk_size, nelem - o)
            offset[dim] = o
            local_shape[dim] = n

    return offset, local_shape


def _compute_transfer_assignment(
    param_names: list[str],
    num_trainers: int,
    num_workers: int,
    seed: int,
) -> dict[tuple[str, int], int]:
    """Compute deterministic assignment of (param_name, trainer_rank) -> worker_rank.

    Uses seeded RNG for reproducible but random-looking assignment.

    Args:
        param_names: List of parameter names
        num_trainers: Number of trainer ranks
        num_workers: Number of worker ranks
        seed: Random seed for reproducibility

    Returns:
        Dict mapping (param_name, trainer_rank) -> worker_rank
    """
    rng = random.Random(seed)
    assignment = {}

    for name in param_names:
        for trainer_rank in range(num_trainers):
            worker_rank = rng.randrange(num_workers)
            assignment[(name, trainer_rank)] = worker_rank

    return assignment


def _create_moodist_group(
    store, rank: int, world_size: int, ranks: list[int], prefix: str
):
    """Create a moodist ProcessGroup for a subset of ranks.

    Args:
        store: The outer store to create a PrefixStore from
        rank: This process's global rank
        world_size: Total world size (unused, kept for clarity)
        ranks: List of global ranks in this group
        prefix: Prefix for the store

    Returns:
        MoodistProcessGroup if this rank is in the group, None otherwise
    """
    if rank not in ranks:
        return None

    ranks = list(sorted(ranks))
    group_rank = ranks.index(rank)
    group_size = len(ranks)

    prefix_store = dist.PrefixStore(prefix, store)
    return moodist.create_moodist_backend(
        prefix_store, group_rank, group_size, timedelta(seconds=30)
    )


class ModelTransfer:
    """Handles model parameter transfer from trainers to workers."""

    def __init__(
        self,
        all_group,
        rank: int,
        num_trainers: int,
        num_workers: int,
        param_names: list[str],
        named_parameters: dict[str, torch.Tensor],
        workers_group=None,
        log_fn=None,
    ):
        self.all_group = all_group
        self.rank = rank
        self.num_trainers = num_trainers
        self.num_workers = num_workers
        self.param_names = param_names
        self.named_parameters = named_parameters
        self.workers_group = workers_group
        self.log_fn = log_fn or (lambda msg: None)

        self.trainer_ranks = list(range(num_trainers))
        self.worker_ranks = list(range(num_trainers, num_trainers + num_workers))
        self.is_trainer = rank < num_trainers

        # Compute transfer assignment (deterministic, same on all ranks)
        self.assignment = _compute_transfer_assignment(
            param_names, num_trainers, num_workers, ASSIGNMENT_SEED
        )

        # Create one queue per worker
        all_ranks = list(range(num_trainers + num_workers))
        self.worker_queues: dict[int, moodist.Queue] = {}
        for worker_global_rank in self.worker_ranks:
            worker_index_in_all = all_ranks.index(worker_global_rank)
            queue = moodist.Queue(all_group, location=worker_index_in_all)
            self.worker_queues[worker_global_rank] = queue

        # For workers: count expected chunks
        if not self.is_trainer:
            self.expected_chunks = sum(
                1
                for (name, trainer_rank), dst_worker in self.assignment.items()
                if self.worker_ranks[dst_worker] == rank
            )
            self.chunk_metadata: list[
                tuple[str, int, list[int], list[int], torch.dtype]
            ] = []
            # Batched ops: list of (op, batch_params) where batch_params is
            # list of (name, input_keys, output_info)
            self.compiled_batches: list[tuple] = []

    def initialize(self):
        """Initialize transfer - trainers send metadata to workers."""
        if self.is_trainer:
            trainer_rank = self.rank

            for name in self.param_names:
                p = self.named_parameters[name]

                # Get placement info
                if isinstance(p, DTensor):
                    placements = list(p.placements)
                    global_shape = tuple(p.shape)
                    dtype = p.dtype
                    # Compute offset and shape for this trainer's shard
                    offset, local_shape = _compute_shard_for_rank(
                        global_shape, placements, trainer_rank, self.num_trainers
                    )
                else:
                    # Non-DTensor (replicated) - treat as full tensor
                    global_shape = tuple(p.shape)
                    dtype = p.dtype
                    offset = [0] * len(global_shape)
                    local_shape = list(global_shape)

                # Skip empty shards
                if 0 in local_shape:
                    continue

                # Find assigned worker
                worker_index = self.assignment[(name, trainer_rank)]
                worker_global_rank = self.worker_ranks[worker_index]

                # Send metadata
                queue = self.worker_queues[worker_global_rank]
                metadata = (name, trainer_rank, offset, local_shape, dtype)
                queue.put_object(metadata)

            # self.log_fn(f"Trainer {trainer_rank}: sent metadata for {len(self.param_names)} parameters")

        else:
            # Worker: receive metadata
            worker_global_rank = self.rank
            queue = self.worker_queues[worker_global_rank]

            for _ in range(self.expected_chunks):
                metadata = queue.get_object()
                name, trainer_rank, offset, shape, dtype = metadata
                self.chunk_metadata.append((name, trainer_rank, offset, shape, dtype))

            # self.log_fn(f"Worker {worker_global_rank}: received metadata for {len(self.chunk_metadata)} chunks")

            # Group chunks by parameter name
            chunks_by_param: dict[
                str, list[tuple[int, list[int], list[int], torch.dtype]]
            ] = {}
            for name, trainer_rank, offset, shape, dtype in self.chunk_metadata:
                if name not in chunks_by_param:
                    chunks_by_param[name] = []
                chunks_by_param[name].append((trainer_rank, offset, shape, dtype))

            # Batch parameters for compile_op calls
            for batch_start in range(0, len(self.param_names), COMPILE_OP_BATCH_SIZE):
                batch_names = self.param_names[batch_start:batch_start + COMPILE_OP_BATCH_SIZE]

                # Collect all inputs and outputs for this batch
                all_inputs = []
                all_outputs = []
                batch_params = []  # (name, input_keys, output_info) per param
                batch_dtype = None

                for name in batch_names:
                    p = self.named_parameters[name]

                    # Get global shape and dtype from worker's parameter
                    if isinstance(p, DTensor):
                        global_shape = tuple(p.shape)
                        dtype = p.dtype
                        placements = list(p.placements)
                        worker_rank_in_group = self.rank - self.num_trainers
                        output_offset, output_shape = _compute_shard_for_rank(
                            global_shape, placements, worker_rank_in_group, self.num_workers
                        )
                    else:
                        # Non-DTensor (replicated) - worker wants full tensor
                        global_shape = tuple(p.shape)
                        dtype = p.dtype
                        output_offset = [0] * len(global_shape)
                        output_shape = list(global_shape)

                    # All params in batch must have same dtype
                    if batch_dtype is None:
                        batch_dtype = dtype
                    assert batch_dtype == dtype, f"Mixed dtypes in batch: {batch_dtype} vs {dtype}"

                    # Inputs: chunks this worker received for this parameter
                    input_keys = []
                    if name in chunks_by_param:
                        for trainer_rank, offset, shape, _ in chunks_by_param[name]:
                            all_inputs.append(TensorRegion(offset=offset, shape=shape, device="cpu", tensor_id=name))
                            input_keys.append((name, trainer_rank))

                    # Output: this worker's shard
                    all_outputs.append(TensorRegion(offset=output_offset, shape=output_shape, device="cpu", tensor_id=name))
                    output_info = (output_offset, output_shape)

                    batch_params.append((name, input_keys, output_info))

                # Compile the batched op
                op = moodist.compile_op(
                    self.workers_group,
                    dtype=batch_dtype,
                    inputs=all_inputs,
                    outputs=all_outputs,
                )
                self.compiled_batches.append((op, batch_params))

    def send(self):
        """Trainer sends parameter chunks to assigned workers."""
        assert self.is_trainer, "send() should only be called by trainers"

        trainer_rank = self.rank

        for name in self.param_names:
            p = self.named_parameters[name]

            # Get local shard
            if isinstance(p, DTensor):
                local_tensor = p.to_local()
            else:
                local_tensor = p

            # Skip empty shards
            if local_tensor.numel() == 0:
                continue

            # Find assigned worker (convert to global rank)
            worker_index = self.assignment[(name, trainer_rank)]
            worker_global_rank = self.worker_ranks[worker_index]

            # Send to worker's queue (use transaction for atomicity)
            queue = self.worker_queues[worker_global_rank]
            with queue.transaction() as t:
                t.put_object((name, trainer_rank))
                t.put_tensor(local_tensor)

        # self.log_fn(f"Trainer {trainer_rank}: sent all chunks")

    def receive(self) -> dict[str, torch.Tensor]:
        """Worker receives parameter chunks from trainers and redistributes.

        Returns:
            Dict mapping param_name -> output tensor
        """
        assert not self.is_trainer, "receive() should only be called by workers"

        worker_global_rank = self.rank
        queue = self.worker_queues[worker_global_rank]

        received_chunks: dict[tuple[str, int], torch.Tensor] = {}

        for _ in range(self.expected_chunks):
            name, trainer_rank = queue.get_object()
            tensor = queue.get_tensor()
            received_chunks[(name, trainer_rank)] = tensor

        # self.log_fn(f"Worker {worker_global_rank}: received {len(received_chunks)} chunks")

        # Run batched compiled ops to redistribute chunks among workers
        output_tensors: dict[str, torch.Tensor] = {}

        handles = []

        for op, batch_params in self.compiled_batches:
            # Gather all inputs for this batch (in order)
            inputlist = []
            for name, input_keys, output_info in batch_params:
                for key in input_keys:
                    inputlist.append(received_chunks[key])

            # Create all output tensors for this batch (in order)
            outputlist = []
            for name, input_keys, output_info in batch_params:
                output_offset, output_shape = output_info
                p = self.named_parameters[name]
                dtype = p.dtype if isinstance(p, DTensor) else p.dtype
                t = torch.empty(output_shape, dtype=dtype, device="cpu")
                outputlist.append(t)
                output_tensors[name] = t

            # Run the batched op
            handles.append(op(inputlist, outputlist))

        for h in handles:
            h.wait()

        # self.log_fn(f"Worker {worker_global_rank}: redistributed {len(output_tensors)} parameters")

        return output_tensors


def _run_model_transfer_test(ctx: TestContext, trainer_ratio: float):
    """Run a model transfer test with the given trainer ratio."""
    if ctx.world_size < 2:
        ctx.log(f"Skipping: need at least 2 ranks, got {ctx.world_size}")
        return

    num_trainers, num_workers = _compute_split(ctx.world_size, trainer_ratio)
    is_trainer = ctx.rank < num_trainers

    ctx.log(
        f"ratio={trainer_ratio}, trainers={num_trainers}, workers={num_workers}, "
        f"role={'trainer' if is_trainer else 'worker'}"
    )

    # Create outer store for coordination
    outer_store = ctx.create_store(key=f"model_transfer_{int(trainer_ratio * 100)}")

    try:
        torch.cuda.set_device(ctx.local_rank)

        # Determine inner group parameters
        if is_trainer:
            list(range(num_trainers))
            inner_rank = ctx.rank
            inner_size = num_trainers
            prefix = "trainers"
        else:
            list(range(num_trainers, ctx.world_size))
            inner_rank = ctx.rank - num_trainers
            inner_size = num_workers
            prefix = "workers"

        # Create inner store and initialize process group
        inner_store = dist.PrefixStore(prefix, outer_store)

        dist.init_process_group(
            backend="moodist",
            store=inner_store,
            rank=inner_rank,
            world_size=inner_size,
            timeout=timedelta(seconds=30),
        )

        # Create all_group for final barrier (all ranks)
        all_group = _create_moodist_group(
            outer_store,
            ctx.rank,
            ctx.world_size,
            list(range(ctx.world_size)),
            "all_group",
        )
        ctx.assert_true(all_group is not None, "all_group should include this rank")

        # Create device mesh for inner group
        device = "cuda"
        mesh = dist.device_mesh.init_device_mesh(
            device,
            (inner_size,),
            mesh_dim_names=("dp",),
        )

        # Create parameters with appropriate sharding
        named_parameters: list[tuple[str, torch.Tensor]] = []

        if is_trainer:
            # Trainers: all parameters sharded

            # Large 2D parameter (like tok_embeddings) - sharded on dim 0
            name = "tok_embeddings.weight"
            t = _make_deterministic_tensor(name, (1024, 256), device, torch.float32)
            t = distribute_tensor(t, mesh, [Shard(0)])
            named_parameters.append((name, t))

            # Several layer parameters
            for layer_idx in range(40):
                # Large 2D weight (like attention.wq) - sharded
                name = f"layers.{layer_idx}.attention.wq.weight"
                t = _make_deterministic_tensor(name, (512, 512), device, torch.float32)
                t = distribute_tensor(t, mesh, [Shard(0)])
                named_parameters.append((name, t))

                # Small 1D norm weight - sharded
                name = f"layers.{layer_idx}.attention_norm.weight"
                t = _make_deterministic_tensor(name, (512,), device, torch.float32)
                t = distribute_tensor(t, mesh, [Shard(0)])
                named_parameters.append((name, t))

            # Final norm (small 1D) - sharded
            name = "norm.weight"
            t = _make_deterministic_tensor(name, (512,), device, torch.float32)
            t = distribute_tensor(t, mesh, [Shard(0)])
            named_parameters.append((name, t))
        else:
            # Workers: large params sharded, small params replicated

            # Large 2D parameter - sharded on workers too
            t = torch.zeros(1024 * 256, device=device, dtype=torch.float32).view(
                1024, 256
            )
            t = distribute_tensor(t, mesh, [Shard(0)])
            named_parameters.append(("tok_embeddings.weight", t))

            # Layer parameters
            for layer_idx in range(40):
                # Large 2D weight - sharded
                t = torch.zeros(512 * 512, device=device, dtype=torch.float32).view(
                    512, 512
                )
                t = distribute_tensor(t, mesh, [Shard(0)])
                named_parameters.append((f"layers.{layer_idx}.attention.wq.weight", t))

                # Small 1D norm - REPLICATED on workers (they want the full tensor)
                t = torch.zeros(512, device=device, dtype=torch.float32)
                # No DTensor wrapping - workers want full tensor
                named_parameters.append(
                    (f"layers.{layer_idx}.attention_norm.weight", t)
                )

            # Final norm - REPLICATED on workers
            t = torch.zeros(512, device=device, dtype=torch.float32)
            named_parameters.append(("norm.weight", t))

        # Get sorted parameter names (must be consistent across all ranks)
        param_names = sorted([name for name, _ in named_parameters])
        named_parameters_dict = {name: p for name, p in named_parameters}

        # Create model transfer handler
        # Workers need their inner process group for compile_op
        workers_group = None
        if not is_trainer:
            workers_group = dist.distributed_c10d._world.default_pg

        model_transfer = ModelTransfer(
            all_group=all_group,
            rank=ctx.rank,
            num_trainers=num_trainers,
            num_workers=num_workers,
            param_names=param_names,
            named_parameters=named_parameters_dict,
            workers_group=workers_group,
            log_fn=ctx.log,
        )

        all_group.barrier()

        model_transfer.initialize()

        for iteration in range(8):
            all_group.barrier()
            if is_trainer:
                model_transfer.send()
                for name, p in named_parameters:
                    p += 1
            else:
                output_tensors = model_transfer.receive()

                # Copy received shards into DTensor local storage
                for name, local_tensor in output_tensors.items():
                    p = named_parameters_dict[name]
                    if isinstance(p, DTensor):
                        moodist.cuda_copy(p.to_local(), local_tensor)
                    else:
                        moodist.cuda_copy(p, local_tensor)

                # Allgather to get full tensors
                full_tensors: dict[str, torch.Tensor] = {}
                for name, p in named_parameters_dict.items():
                    if isinstance(p, DTensor):
                        full_tensors[name] = p.full_tensor()
                    else:
                        full_tensors[name] = p

                # Verify received tensors match expected values
                for name, received in full_tensors.items():
                    p = named_parameters_dict[name]
                    shape = tuple(p.shape)
                    expected = (
                        _make_deterministic_tensor(name, shape, device, torch.float32)
                        + iteration
                    )
                    if not torch.equal(received, expected):
                        raise AssertionError(
                            f"Tensor mismatch for {name}: "
                            f"received shape {received.shape}, expected shape {expected.shape}"
                        )

        # Final barrier
        all_group.barrier()

        ctx.log("test completed successfully")

    finally:
        _cleanup_distributed()


def _make_test(trainer_ratio: float):
    """Create a test function for a given trainer ratio."""

    def test_fn(ctx: TestContext):
        _run_model_transfer_test(ctx, trainer_ratio)

    # Name like test_model_transfer_ratio_10, test_model_transfer_ratio_25, etc.
    ratio_name = f"{int(trainer_ratio * 100):02d}"
    test_fn.__name__ = f"test_model_transfer_ratio_{ratio_name}"
    return test_fn


# Register tests for each ratio
for _ratio in TRAINER_RATIOS:
    _test_fn = _make_test(_ratio)
    test(_test_fn)
