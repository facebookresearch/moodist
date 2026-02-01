import weakref

import torch

from .queue import Queue


weak_group = weakref.WeakValueDictionary()
weak_queue = weakref.WeakKeyDictionary()


class Name(str):
    pass


def _is_dtensor(x):
    """Check if x is a DTensor without hard dependency on torch.distributed.tensor."""
    return hasattr(x, 'placements') and hasattr(x, 'device_mesh') and hasattr(x, 'to_local')


def _get_shard_metadata(dtensor):
    """Extract offset and local shape for this rank's shard of the DTensor.

    Matches PyTorch's Shard.local_shard_size_and_offset logic for consistency.
    """
    from torch.distributed.tensor import Shard

    device_mesh = dtensor.device_mesh
    placements = dtensor.placements
    global_shape = list(dtensor.shape)
    offsets = [0] * len(global_shape)
    local_shape = list(global_shape)

    coord = device_mesh.get_coordinate()
    if coord is None:
        # This rank is not part of this mesh - return empty shard
        return {'offset': offsets, 'shape': [0] * len(global_shape)}

    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            global_size = global_shape[shard_dim]
            num_chunks = device_mesh.size(mesh_dim)
            rank = coord[mesh_dim]

            chunk_size = (global_size + num_chunks - 1) // num_chunks
            shard_start = chunk_size * rank

            if global_size < shard_start:
                # Empty shard - past the end
                offsets[shard_dim] = global_size
                local_shape[shard_dim] = 0
            else:
                offsets[shard_dim] = shard_start
                local_shape[shard_dim] = (
                    min(global_size, shard_start + chunk_size) - shard_start
                )

    return {'offset': offsets, 'shape': local_shape}


def _process_tensor_specs(specs, holder):
    """
    Process a list of tensor specifications (dicts or DTensors).

    Args:
        specs: List of dicts or DTensors
        holder: Dict to store/validate ndim and dtype

    Returns:
        List of dicts with 'offset' and 'shape' keys
    """
    if specs is None:
        return None

    processed = []
    for x in specs:
        if _is_dtensor(x):
            # Extract and validate dtype
            x_dtype = x.dtype
            x_ndim = len(x.shape)

            if holder.get('dtype') is None:
                holder['dtype'] = x_dtype
                holder['ndim'] = x_ndim
            else:
                if holder['dtype'] != x_dtype:
                    raise ValueError(
                        f"All DTensors must have the same dtype, got {holder['dtype']} and {x_dtype}"
                    )
                if holder.get('ndim') is not None and holder['ndim'] != x_ndim:
                    raise ValueError(
                        f"All DTensors must have the same ndim, got {holder['ndim']} and {x_ndim}"
                    )
                holder['ndim'] = x_ndim

            processed.append(_get_shard_metadata(x))
        else:
            # Assume it's a dict
            processed.append(x)

    return processed


def compile_op(group, dtype=None, inputs=None, outputs=None, reduce=None):
    """Compile a custom collective operation for distributed tensor communication.

    This function creates an optimized collective operation that transfers data between
    processes in a distributed group. It's a generalization of standard collective
    operations (like all_gather, reduce_scatter, etc.) that allows arbitrary input/output
    patterns across ranks.

    The function coordinates all ranks to exchange their input/output specifications,
    validates consistency across ranks, and compiles an optimized operation that handles
    the specified data movement patterns.

    Args:
        group: A MoodistProcessGroup instance representing the distributed process group.
        dtype: The PyTorch data type (torch.dtype) for the operation (e.g., torch.float32).
               All ranks must specify the same dtype. Can be omitted if using DTensors.
        inputs: Optional list of input tensor specifications. Each element can be either:
                - A dict with 'offset' and 'shape' keys specifying the slice in global coordinates
                - A DTensor, from which the offset and shape are derived automatically
                If None, this rank contributes no inputs to the operation.
        outputs: Optional list of output tensor specifications. Same format as inputs.
                 If None, this rank receives no outputs from the operation.
        reduce: How to handle overlapping inputs. Options:
                - None (default): Error if inputs overlap
                - "any": Pick any source for overlapping regions (for replicated data)

    Returns:
        A compiled custom operation object that can be used to efficiently execute the
        specified collective communication pattern.

    Raises:
        ValueError: If dtype is not provided (and not derivable from DTensors),
                   input/output specifications are malformed, or ranks specify inconsistent
                   dtypes.
        TypeError: If dtype is not a torch.dtype, or input/output specifications have
                   wrong types.

    Example:
        >>> # Using dict specifications:
        >>> # Rank 0 sends data at offset [0, 0] with shape [2, 4]
        >>> # Rank 1 receives data at offset [0, 0] with shape [2, 4]
        >>> import torch
        >>> import moodist
        >>> group = moodist.find_process_group("my_group")
        >>>
        >>> if group.rank() == 0:
        >>>     inputs = [{'offset': [0, 0], 'shape': [2, 4]}]
        >>>     outputs = None
        >>> else:
        >>>     inputs = None
        >>>     outputs = [{'offset': [0, 0], 'shape': [2, 4]}]
        >>>
        >>> op = moodist.compile_op(
        >>>     group,
        >>>     dtype=torch.float32,
        >>>     inputs=inputs,
        >>>     outputs=outputs
        >>> )
        >>>
        >>> # Using DTensors (dtype derived automatically):
        >>> op = moodist.compile_op(
        >>>     group,
        >>>     inputs=[input_dtensor],
        >>>     outputs=[output_dtensor]
        >>> )

    Note:
        - This function performs collective synchronization (barriers and queue operations)
          and must be called by all ranks in the group.
        - Input/output regions can overlap, enabling operations like scatter, gather,
          all-gather, reduce-scatter, and custom patterns.
        - The function uses an internal queue for coordination, which is cached per group.
    """
    # Process DTensors and extract dtype if not provided
    holder = {'ndim': None, 'dtype': dtype}

    inputs = _process_tensor_specs(inputs, holder)
    outputs = _process_tensor_specs(outputs, holder)

    ndim = holder['ndim']
    dtype = holder['dtype']

    if dtype is None:
        raise ValueError("dtype must be provided or derivable from DTensors")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")

    # Derive ndim from inputs/outputs if not derived from DTensors
    if ndim is None:
        if inputs:
            for x in inputs:
                if 'offset' in x and x['offset']:
                    ndim = len(x['offset'])
                    break
        if ndim is None and outputs:
            for x in outputs:
                if 'offset' in x and x['offset']:
                    ndim = len(x['offset'])
                    break

    name = Name(group.moodist_name() + ".{compile_collective_queue}")
    if name not in weak_group:
        queue = Queue(group, range(group.size()), name=name)
        weak_queue[name] = queue
        weak_group[name] = group
    queue = weak_queue.get(name)
    assert isinstance(queue, Queue)

    def check(l):
        nonlocal ndim
        if not isinstance(l, (tuple, list)):
            raise TypeError(f"inputs/outputs must be a tuple or list, got {type(l).__name__}")
        for x in l:
            if not isinstance(x, dict):
                raise TypeError(f"each input/output spec must be a dict, got {type(x).__name__}")
            for n in ("offset", "shape"):
                if n not in x:
                    raise ValueError(f"'{n}' is missing for an input or output")
                v = x[n]
                if not isinstance(v, (tuple, list)):
                    raise TypeError(f"'{n}' must be a tuple or list, got {type(v).__name__}")
                # Set or validate ndim
                if ndim is None:
                    ndim = len(v)
                elif len(v) != ndim:
                    raise ValueError(
                        f"expected '{n}' with {ndim} dimensions, but got {len(v)}"
                    )
                for i, z in enumerate(v):
                    if not isinstance(z, int):
                        raise TypeError(f"{n}[{i}] must be an int, got {type(z).__name__}")
        return tuple((tuple(x["offset"]), tuple(x["shape"])) for x in l)

    if inputs is not None:
        inputs = check(inputs)
    if outputs is not None:
        outputs = check(outputs)

    assert queue.empty()
    group.barrier()

    info = (group.rank(), ndim, dtype, inputs, outputs)
    queue.put_object(info)

    all_inputs = []
    all_outputs = []

    for _ in range(group.size()):
        source_rank, n_ndim, ndtype, ninput, noutput = queue.get_object()
        # Validate ndim consistency (only if both ranks have data)
        if n_ndim is not None and ndim is not None and n_ndim != ndim:
            raise ValueError(
                f"moodist.compile_op: Ranks specified different ndim: {ndim} vs {n_ndim}"
            )
        if ndtype != dtype:
            raise ValueError(
                f"moodist.compile_op: Ranks specified different dtypes: {dtype} vs {ndtype}"
            )

        if ninput is not None:
            for o, s in ninput:
                all_inputs.append((source_rank, o, s))
        if noutput is not None:
            for o, s in noutput:
                all_outputs.append((source_rank, o, s))

    assert queue.empty()
    group.barrier()

    return group.compile_op_full(dtype, all_inputs, all_outputs, reduce=reduce)
