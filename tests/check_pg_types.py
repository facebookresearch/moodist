#!/usr/bin/env python3
"""
Standalone script to inspect ProcessGroup types with moodist backend.

Run with: torchrun --nproc_per_node=2 check_pg_types.py
"""

import gc
import torch
import torch.distributed as dist
from datetime import timedelta


def main():
    rank = int(__import__("os").environ.get("RANK", 0))
    world_size = int(__import__("os").environ.get("WORLD_SIZE", 1))

    # Only print from rank 0
    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    # Initialize with moodist
    import moodist
    store = moodist.TcpStore(
        "127.0.0.1",
        29500,
        "check_types",
        world_size,
        rank,
        timedelta(seconds=30),
    )

    dist.init_process_group(
        backend="moodist",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)

    log("=" * 60)
    log("DEFAULT PROCESS GROUP")
    log("=" * 60)

    default_pg = dist.distributed_c10d._world.default_pg
    log(f"default_pg = {default_pg}")
    log(f"type(default_pg) = {type(default_pg)}")
    log(f"type(default_pg).__name__ = {type(default_pg).__name__}")
    log(f"type(default_pg).__module__ = {type(default_pg).__module__}")
    log(f"id(default_pg) = {id(default_pg)}")

    ProcessGroup = torch.distributed.ProcessGroup
    log(f"\nisinstance(default_pg, ProcessGroup) = {isinstance(default_pg, ProcessGroup)}")
    log(f"issubclass(type(default_pg), ProcessGroup) = {issubclass(type(default_pg), ProcessGroup)}")

    log(f"\nhasattr(default_pg, 'moodist_name') = {hasattr(default_pg, 'moodist_name')}")
    if hasattr(default_pg, 'moodist_name'):
        log(f"default_pg.moodist_name() = {default_pg.moodist_name()}")

    log("\n" + "=" * 60)
    log("NEW_GROUP (all ranks)")
    log("=" * 60)

    # Create a new group with all ranks
    new_pg = dist.new_group(list(range(world_size)))
    log(f"new_pg = {new_pg}")
    log(f"type(new_pg) = {type(new_pg)}")
    log(f"type(new_pg).__name__ = {type(new_pg).__name__}")
    log(f"type(new_pg).__module__ = {type(new_pg).__module__}")
    log(f"id(new_pg) = {id(new_pg)}")

    log(f"\nisinstance(new_pg, ProcessGroup) = {isinstance(new_pg, ProcessGroup)}")
    log(f"issubclass(type(new_pg), ProcessGroup) = {issubclass(type(new_pg), ProcessGroup)}")

    log(f"\nhasattr(new_pg, 'moodist_name') = {hasattr(new_pg, 'moodist_name')}")
    if hasattr(new_pg, 'moodist_name'):
        log(f"new_pg.moodist_name() = {new_pg.moodist_name()}")

    log(f"\nnew_pg is default_pg = {new_pg is default_pg}")
    log(f"type(new_pg) is type(default_pg) = {type(new_pg) is type(default_pg)}")

    if world_size >= 2:
        log("\n" + "=" * 60)
        log("NEW_GROUP (subset - rank 0 only)")
        log("=" * 60)

        # Create a group with just rank 0
        subset_pg = dist.new_group([0])
        log(f"subset_pg = {subset_pg}")
        log(f"type(subset_pg) = {type(subset_pg)}")
        if subset_pg is not None and subset_pg != dist.GroupMember.NON_GROUP_MEMBER:
            log(f"type(subset_pg).__name__ = {type(subset_pg).__name__}")
            log(f"hasattr(subset_pg, 'moodist_name') = {hasattr(subset_pg, 'moodist_name')}")
        else:
            log(f"(rank {rank} is not in this group)")

    log("\n" + "=" * 60)
    log("_world REGISTRIES")
    log("=" * 60)

    log(f"\n_world.pg_map keys:")
    for pg in dist.distributed_c10d._world.pg_map:
        log(f"  {type(pg).__name__} @ {id(pg)}")

    log(f"\n_world.pg_group_ranks keys:")
    for pg in dist.distributed_c10d._world.pg_group_ranks:
        log(f"  {type(pg).__name__} @ {id(pg)}")

    log("\n" + "=" * 60)
    log("DEVICE MESH")
    log("=" * 60)

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("dp",),
    )
    dim_group = device_mesh.get_group(0)

    log(f"dim_group = {dim_group}")
    log(f"type(dim_group) = {type(dim_group)}")
    log(f"type(dim_group).__name__ = {type(dim_group).__name__}")
    log(f"id(dim_group) = {id(dim_group)}")

    log(f"\ndim_group is default_pg = {dim_group is default_pg}")
    log(f"type(dim_group) is type(default_pg) = {type(dim_group) is type(default_pg)}")

    log(f"\nhasattr(dim_group, 'moodist_name') = {hasattr(dim_group, 'moodist_name')}")

    # Check if group_name works on wrapper (should read from shared C++ object)
    log(f"\ndefault_pg.group_name = {default_pg.group_name}")
    log(f"dim_group.group_name = {dim_group.group_name}")
    log(f"group_name matches: {default_pg.group_name == dim_group.group_name}")

    log(f"\ndim_group in _world.pg_group_ranks = {dim_group in dist.distributed_c10d._world.pg_group_ranks}")
    log(f"default_pg in _world.pg_group_ranks = {default_pg in dist.distributed_c10d._world.pg_group_ranks}")

    # Try get_rank on dim_group
    log("\n" + "=" * 60)
    log("LOOKUPS")
    log("=" * 60)
    try:
        r = dist.get_rank(dim_group)
        log(f"dist.get_rank(dim_group) = {r}")
    except Exception as e:
        log(f"dist.get_rank(dim_group) FAILED: {e}")

    try:
        r = dist.get_rank(default_pg)
        log(f"dist.get_rank(default_pg) = {r}")
    except Exception as e:
        log(f"dist.get_rank(default_pg) FAILED: {e}")

    try:
        r = dist.get_rank(new_pg)
        log(f"dist.get_rank(new_pg) = {r}")
    except Exception as e:
        log(f"dist.get_rank(new_pg) FAILED: {e}")

    try:
        b = dist.get_backend(dim_group)
        log(f"dist.get_backend(dim_group) = {b}")
    except Exception as e:
        log(f"dist.get_backend(dim_group) FAILED: {e}")

    try:
        b = dist.get_backend(default_pg)
        log(f"dist.get_backend(default_pg) = {b}")
    except Exception as e:
        log(f"dist.get_backend(default_pg) FAILED: {e}")

    log("\n" + "=" * 60)
    log("CLEANUP TEST - DESTROY SINGLE GROUP")
    log("=" * 60)

    log(f"\nBEFORE destroying new_pg:")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    log(f"  pg_group_ranks keys:")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    # Find wrapper for new_pg (should have same C++ pointer, different Python id)
    new_pg_id = id(new_pg)
    log(f"\n  new_pg id: {new_pg_id}")

    # Identify wrappers before
    wrappers_before = [
        (pg, id(pg)) for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"
    ]
    log(f"  Wrappers before: {[(type(w).__name__, wid) for w, wid in wrappers_before]}")

    log(f"\nCalling dist.destroy_process_group(new_pg)...")
    dist.destroy_process_group(new_pg)

    log(f"\nAFTER destroy_process_group(new_pg) (before gc.collect):")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    log(f"\nCalling gc.collect()...")
    gc.collect()

    log(f"\nAFTER gc.collect (still holding new_pg reference):")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    # The problem: we still hold a reference to new_pg, so it can't be GC'd
    log(f"\nDeleting local new_pg reference...")
    del new_pg
    gc.collect()

    log(f"\nAFTER del new_pg + gc.collect:")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    # Check if new_pg's wrapper leaked
    wrappers_after = [
        (pg, id(pg)) for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"
    ]
    log(f"\n  Wrappers after: {[(type(w).__name__, wid) for w, wid in wrappers_after]}")

    # We started with 2 wrappers, destroyed 1 group, should have 1 wrapper left
    if len(wrappers_after) > 1:
        log(f"  WARNING: Possible wrapper leak! Expected 1 wrapper, got {len(wrappers_after)}")
    else:
        log(f"  OK: {len(wrappers_after)} wrapper(s) remaining (expected 1 for default_pg)")

    log("\n" + "=" * 60)
    log("CLEANUP TEST - DESTROY ALL")
    log("=" * 60)

    log(f"\nBEFORE destroy_process_group:")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    log(f"  pg_group_ranks keys:")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    # Identify wrappers vs MoodistProcessGroup
    wrappers_before = []
    moodist_pgs_before = []
    for pg in dist.distributed_c10d._world.pg_group_ranks:
        if type(pg).__name__ == "ProcessGroup":
            wrappers_before.append(pg)
        elif type(pg).__name__ == "MoodistProcessGroup":
            moodist_pgs_before.append(pg)

    log(f"\n  Wrappers (ProcessGroup): {len(wrappers_before)}")
    log(f"  MoodistProcessGroups: {len(moodist_pgs_before)}")

    # Keep a reference to wrapper to check if it survives
    wrapper_ids_before = [id(w) for w in wrappers_before]
    log(f"  Wrapper ids: {wrapper_ids_before}")

    log(f"\nCalling dist.destroy_process_group()...")
    dist.destroy_process_group()

    log(f"\nAFTER destroy_process_group (before gc.collect):")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    log(f"\nCalling gc.collect()...")
    gc.collect()

    log(f"\nAFTER gc.collect:")
    log(f"  len(pg_group_ranks) = {len(dist.distributed_c10d._world.pg_group_ranks)}")
    for pg in list(dist.distributed_c10d._world.pg_group_ranks.keys()):
        log(f"    {type(pg).__name__} @ {id(pg)}")

    # Check if wrappers survived
    wrappers_after = [
        pg for pg in dist.distributed_c10d._world.pg_group_ranks
        if type(pg).__name__ == "ProcessGroup"
    ]
    log(f"\n  Wrappers remaining: {len(wrappers_after)}")
    if wrappers_after:
        log(f"  WARNING: Wrapper leak detected!")
        for w in wrappers_after:
            log(f"    {type(w).__name__} @ {id(w)}")

    log("\nDone.")


if __name__ == "__main__":
    main()
