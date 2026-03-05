import weakref
from dataclasses import dataclass
from typing import List, Union

import torch

from .queue import Queue
from .sharding import dtensor_shards


weak_group = weakref.WeakValueDictionary()
weak_queue = weakref.WeakKeyDictionary()


class Name(str):
    pass


@dataclass
class TensorRegion:
    """Specification for an input or output tensor region in compile_op.

    Attributes:
        offset: Position in the global tensor (list of ints, one per dimension).
        shape: Shape of this region (list of ints, one per dimension).
        device: Device for this region. Can be a torch.device or string like
            "cpu", "cuda", or "cuda:N". If cuda:N is specified, it must match
            the process group's CUDA device.
        tensor_id: Identifier for grouping regions. Regions with different tensor_ids
            are treated as separate logical tensors and can have different ndims.
            Defaults to "0".
    """
    offset: List[int]
    shape: List[int]
    device: Union[str, torch.device]
    tensor_id: str = "0"


def specs_from_dtensor(dtensor, tensor_id: str = "0") -> List[TensorRegion]:
    """Convert a DTensor's shards to TensorRegions for compile_op.

    Uses dtensor_shards() to compute the chunks this rank owns, then wraps
    them as TensorRegion objects with the specified tensor_id.

    Args:
        dtensor: A PyTorch DTensor.
        tensor_id: Identifier for this tensor (default "0").

    Returns:
        List of TensorRegion objects for this rank's chunks.
    """
    shards = dtensor_shards(dtensor)
    device = dtensor.device
    return [TensorRegion(s.global_offset, s.shape, device, tensor_id) for s in shards]


def _is_dtensor(x):
    """Check if x is a DTensor without hard dependency on torch.distributed.tensor."""
    return hasattr(x, 'placements') and hasattr(x, 'device_mesh') and hasattr(x, 'to_local')


def _process_tensor_specs(specs, holder):
    """
    Process a list of tensor specifications (TensorSpec, dict, or DTensor).

    Args:
        specs: List of TensorSpec, dicts, or DTensors
        holder: Dict to store/validate dtype (from DTensors)

    Returns:
        List of TensorSpec objects, or None if specs is None
    """
    if specs is None:
        return None

    processed = []
    for x in specs:
        if isinstance(x, TensorRegion):
            processed.append(x)
        elif _is_dtensor(x):
            # Extract and validate dtype from DTensor
            x_dtype = x.dtype

            if holder.get('dtype') is None:
                holder['dtype'] = x_dtype
            elif holder['dtype'] != x_dtype:
                raise ValueError(
                    f"All DTensors must have the same dtype, got {holder['dtype']} and {x_dtype}"
                )

            # Convert DTensor to TensorRegions (may produce multiple regions)
            # Use default tensor_id "0" for DTensors passed directly
            dtensor_regions = specs_from_dtensor(x, tensor_id="0")
            processed.extend(dtensor_regions)
        elif isinstance(x, dict):
            # Convert dict to TensorRegion
            offset = x.get('offset')
            shape = x.get('shape')
            device = x.get('device')
            if device is None:
                raise ValueError("'device' is required for dict-style tensor specifications")
            tensor_id = x.get('tensor_id', "0")
            # Convert tensor_id to string if it's an int
            if isinstance(tensor_id, int):
                tensor_id = str(tensor_id)
            processed.append(TensorRegion(offset=offset, shape=shape, device=device, tensor_id=tensor_id))
        else:
            raise TypeError(
                f"each input/output spec must be a TensorRegion, dict, or DTensor, "
                f"got {type(x).__name__}"
            )

    return processed


def compile_op(group, dtype=None, inputs=None, outputs=None, reduce=None, cpu_sync=False):
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
                - A TensorRegion object (requires device field)
                - A dict with 'offset', 'shape', 'device', and optionally 'tensor_id' keys
                - A DTensor, from which specs are derived automatically
                If None, this rank contributes no inputs to the operation.
        outputs: Optional list of output tensor specifications. Same format as inputs.
                 If None, this rank receives no outputs from the operation.
        reduce: How to handle overlapping inputs. Options:
                - None (default): Error if inputs overlap
                - "any": Pick any source for overlapping regions (for replicated data)
        cpu_sync: If True, force CPU-side synchronization before CUDA operations.
                  This avoids potential deadlocks when CUDA has device-wide syncs pending.
                  Default is False.

    Returns:
        A compiled custom operation object that can be used to efficiently execute the
        specified collective communication pattern.

    Raises:
        ValueError: If dtype is not provided (and not derivable from DTensors),
                   input/output specifications are malformed, or ranks specify inconsistent
                   dtypes or ndims within a tensor_id group.
        TypeError: If dtype is not a torch.dtype, or input/output specifications have
                   wrong types.

    Example:
        >>> # Using dict specifications:
        >>> import torch
        >>> import moodist
        >>> group = moodist.find_process_group("my_group")
        >>>
        >>> if group.rank() == 0:
        >>>     inputs = [{'offset': [0, 0], 'shape': [2, 4], 'device': 'cuda'}]
        >>>     outputs = None
        >>> else:
        >>>     inputs = None
        >>>     outputs = [{'offset': [0, 0], 'shape': [2, 4], 'device': 'cuda'}]
        >>>
        >>> op = moodist.compile_op(
        >>>     group,
        >>>     dtype=torch.float32,
        >>>     inputs=inputs,
        >>>     outputs=outputs
        >>> )
        >>>
        >>> # Using TensorRegion with tensor_id for batching multiple tensors:
        >>> from moodist import TensorRegion
        >>> inputs = [
        >>>     TensorRegion(offset=[0, 0], shape=[10, 10], device="cuda", tensor_id="weight"),
        >>>     TensorRegion(offset=[0], shape=[128], device="cuda", tensor_id="bias"),
        >>> ]
        >>>
        >>> # Using DTensors (dtype and device derived automatically):
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
        - Different tensor_ids can have different ndims (e.g., 2D weight vs 1D bias).
        - The function uses an internal queue for coordination, which is cached per group.
    """
    # Process specs and extract dtype if not provided
    holder = {'dtype': dtype}

    inputs = _process_tensor_specs(inputs, holder)
    outputs = _process_tensor_specs(outputs, holder)

    dtype = holder['dtype']

    if dtype is None:
        raise ValueError("dtype must be provided or derivable from DTensors")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")

    name = Name(group.moodist_name() + ".{compile_collective_queue}")
    if name not in weak_group:
        queue = Queue(group, range(group.size()), name=name)
        weak_queue[name] = queue
        weak_group[name] = group
    queue = weak_queue.get(name)
    assert isinstance(queue, Queue)

    def check(specs):
        """Validate TensorRegions and convert to tuples for serialization."""
        if not isinstance(specs, (tuple, list)):
            raise TypeError(f"inputs/outputs must be a tuple or list, got {type(specs).__name__}")
        result = []
        for spec in specs:
            if not isinstance(spec, TensorRegion):
                raise TypeError(f"each input/output spec must be a TensorRegion, got {type(spec).__name__}")
            for name, value in [("offset", spec.offset), ("shape", spec.shape)]:
                if value is None:
                    raise ValueError(f"'{name}' is missing for an input or output")
                if not isinstance(value, (tuple, list)):
                    raise TypeError(f"'{name}' must be a tuple or list, got {type(value).__name__}")
                for i, z in enumerate(value):
                    if not isinstance(z, int):
                        raise TypeError(f"{name}[{i}] must be an int, got {type(z).__name__}")
            if len(spec.offset) != len(spec.shape):
                raise ValueError(
                    f"offset and shape must have same length, got {len(spec.offset)} and {len(spec.shape)}"
                )
            # Convert device to string for serialization
            device_str = str(spec.device) if spec.device is not None else None
            if device_str is None:
                raise ValueError("'device' is required for TensorRegion")
            result.append((tuple(spec.offset), tuple(spec.shape), spec.tensor_id, device_str))
        return tuple(result)

    if inputs is not None:
        inputs = check(inputs)
    if outputs is not None:
        outputs = check(outputs)

    assert queue.empty()
    group.barrier()

    info = (group.rank(), dtype, inputs, outputs)
    queue.put_object(info)

    all_inputs = []
    all_outputs = []

    for _ in range(group.size()):
        source_rank, ndtype, ninput, noutput = queue.get_object()
        if ndtype != dtype:
            raise ValueError(
                f"moodist.compile_op: Ranks specified different dtypes: {dtype} vs {ndtype}"
            )

        if ninput is not None:
            for offset, shape, tensor_id, device in ninput:
                all_inputs.append((source_rank, offset, shape, tensor_id, device))
        if noutput is not None:
            for offset, shape, tensor_id, device in noutput:
                all_outputs.append((source_rank, offset, shape, tensor_id, device))

    # Validate per-tensor_id ndim consistency
    tensor_id_ndim = {}
    for items, kind in [(all_inputs, "input"), (all_outputs, "output")]:
        for rank, offset, shape, tensor_id, device in items:
            ndim = len(offset)
            if tensor_id in tensor_id_ndim:
                if tensor_id_ndim[tensor_id] != ndim:
                    raise ValueError(
                        f"moodist.compile_op: Inconsistent ndim for tensor_id '{tensor_id}': "
                        f"expected {tensor_id_ndim[tensor_id]}, got {ndim} (in {kind} from rank {rank})"
                    )
            else:
                tensor_id_ndim[tensor_id] = ndim

    assert queue.empty()
    group.barrier()

    return group.compile_op_full(dtype, all_inputs, all_outputs, reduce=reduce, cpu_sync=cpu_sync)
