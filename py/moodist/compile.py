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


def _process_tensor_specs(specs, shape_holder):
    """
    Process a list of tensor specifications (dicts or DTensors).

    Args:
        specs: List of dicts or DTensors
        shape_holder: Dict to store/validate global shape and dtype

    Returns:
        List of dicts with 'offset' and 'shape' keys
    """
    if specs is None:
        return None

    processed = []
    for x in specs:
        if _is_dtensor(x):
            # Extract and validate global shape/dtype
            x_shape = tuple(x.shape)
            x_dtype = x.dtype

            if shape_holder.get('shape') is None:
                shape_holder['shape'] = x_shape
                shape_holder['dtype'] = x_dtype
            else:
                if shape_holder['shape'] != x_shape:
                    raise ValueError(
                        f"All DTensors must have the same global shape, got {shape_holder['shape']} and {x_shape}"
                    )
                if shape_holder['dtype'] != x_dtype:
                    raise ValueError(
                        f"All DTensors must have the same dtype, got {shape_holder['dtype']} and {x_dtype}"
                    )

            processed.append(_get_shard_metadata(x))
        else:
            # Assume it's a dict
            processed.append(x)

    return processed


def compile_op(group, shape=None, dtype=None, inputs=None, outputs=None):
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
        shape: The global tensor shape as a tuple/list of integers (e.g., (batch, height, width)).
               All ranks must specify the same shape. Can be omitted if using DTensors.
        dtype: The PyTorch data type (torch.dtype) for the operation (e.g., torch.float32).
               All ranks must specify the same dtype. Can be omitted if using DTensors.
        inputs: Optional list of input tensor specifications. Each element can be either:
                - A dict with 'offset' and 'shape' keys specifying the slice in global coordinates
                - A DTensor, from which the offset and shape are derived automatically
                If None, this rank contributes no inputs to the operation.
        outputs: Optional list of output tensor specifications. Same format as inputs.
                 If None, this rank receives no outputs from the operation.

    Returns:
        A compiled custom operation object that can be used to efficiently execute the
        specified collective communication pattern.

    Raises:
        ValueError: If shape/dtype are not provided (and not derivable from DTensors),
                   input/output specifications are malformed, or ranks specify inconsistent
                   shapes or dtypes.
        TypeError: If dtype is not a torch.dtype, shape contains non-integers, or
                  input/output specifications have wrong types.

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
        >>>     shape=[2, 4],
        >>>     dtype=torch.float32,
        >>>     inputs=inputs,
        >>>     outputs=outputs
        >>> )
        >>>
        >>> # Using DTensors (shape and dtype derived automatically):
        >>> op = moodist.compile_op(
        >>>     group,
        >>>     inputs=[input_dtensor],
        >>>     outputs=[output_dtensor]
        >>> )

    Note:
        - This function performs collective synchronization (barriers and queue operations)
          and must be called by all ranks in the group.
        - The offset and shape dimensions must match the global shape's dimensionality.
        - Input/output regions can overlap, enabling operations like scatter, gather,
          all-gather, reduce-scatter, and custom patterns.
        - The function uses an internal queue for coordination, which is cached per group.
    """
    # Process DTensors and extract shape/dtype if not provided
    shape_holder = {'shape': tuple(shape) if shape is not None else None,
                    'dtype': dtype}

    inputs = _process_tensor_specs(inputs, shape_holder)
    outputs = _process_tensor_specs(outputs, shape_holder)

    shape = shape_holder['shape']
    dtype = shape_holder['dtype']

    if shape is None:
        raise ValueError("shape must be provided or derivable from DTensors")
    if dtype is None:
        raise ValueError("dtype must be provided or derivable from DTensors")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")

    shape = tuple(shape)
    for i, x in enumerate(shape):
        if not isinstance(x, int):
            raise TypeError(f"shape[{i}] must be an int, got {type(x).__name__}")

    name = Name(group.moodist_name() + ".{compile_collective_queue}")
    if name not in weak_group:
        queue = Queue(group, range(group.size()), name=name)
        weak_queue[name] = queue
        weak_group[name] = group
    queue = weak_queue.get(name)
    assert isinstance(queue, Queue)

    def check(l):
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
                if len(v) != len(shape):
                    raise ValueError(
                        f"expected '{n}' with {len(shape)} dimensions, but got {len(v)}"
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

    info = (group.rank(), shape, dtype, inputs, outputs)
    queue.put_object(info)

    all_inputs = []
    all_outputs = []

    for _ in range(group.size()):
        source_rank, nshape, ndtype, ninput, noutput = queue.get_object()
        if nshape != shape:
            raise ValueError(
                f"moodist.compile_op: Ranks specified different shapes: {shape} vs {nshape}"
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

    return group.compile_op_full(shape, dtype, all_inputs, all_outputs)
