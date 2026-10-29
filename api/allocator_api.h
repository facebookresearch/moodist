// Copyright (c) Meta Platforms, Inc. and affiliates.

// CUDA Allocator API for moodist.
// Uses ABI-safe types at the library boundary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

// Forward declare CUstream in global namespace for cuda.h compatibility
struct CUstream_st;
typedef CUstream_st* CUstream;

namespace moodist::cuda {
using CUstream = ::CUstream;
} // namespace moodist::cuda

namespace moodist {

using namespace cuda;

// Forward declare CudaAllocatorImpl (opaque type in core)
struct CudaAllocatorImpl;

// Allocation result - returned by cudaAllocatorImplAllocate
struct CudaAllocation {
  uintptr_t ptr;    // Allocated memory pointer
  void* cleanupCtx; // Context for deallocation (pass to cudaAllocatorImplFree)
  int deviceIndex;  // Device index (may differ from input if auto-detected)
};

// CoreApi functions for CUDA allocator

// Create a new CudaAllocatorImpl
CudaAllocatorImpl* createCudaAllocatorImpl();

// Set the global allocator impl pointer (for allocator::owns/mappedRegion)
void setCudaAllocatorImpl(CudaAllocatorImpl* impl);

// Allocate memory
// - bytes: number of bytes to allocate (will be aligned internally)
// - stream: CUDA stream for allocation
// - deviceIndex: CUDA device index (-1 to auto-detect from current context)
// Returns: CudaAllocation with ptr, cleanupCtx, and actual deviceIndex
CudaAllocation cudaAllocatorImplAllocate(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex);

// Free memory using cleanup context (the main deallocation path)
// - cleanupCtx: context returned from cudaAllocatorImplAllocate
// - stream: the stream that was active when the allocation was last used
void cudaAllocatorImplFree(void* cleanupCtx, CUstream stream);

// Record that a stream has used an allocation (for recordStream)
// This must be called when torch's recordStream is invoked
void cudaAllocatorImplRecordStream(CudaAllocatorImpl* impl, uintptr_t ptr, CUstream stream);

// Query if an address is owned by the allocator
bool allocatorOwns(uintptr_t address);

// Get the mapped region containing an address
std::pair<uintptr_t, size_t> allocatorMappedRegion(uintptr_t address);

// Free callback API - allows registering callbacks to be invoked when memory is freed
// Returns: handle that can be passed to allocatorRemoveFreeCallback
void* allocatorAddFreeCallback(CudaAllocatorImpl* impl, uintptr_t baseAddress, void* callbackFn);

// Remove a free callback
void allocatorRemoveFreeCallback(CudaAllocatorImpl* impl, uintptr_t baseAddress, void* handle);

// Get the device index the allocator is initialized on (-1 if not yet initialized)
int cudaAllocatorImplGetDevice(CudaAllocatorImpl* impl);

// Get the reserved VA base address (0 if not initialized)
uintptr_t allocatorGetReservedBase();

// Size of the allocator's VA reservation: chunk offsets (and so pool offsets) are below it. Peers reserve the same
// size for the mapping of a rank's pool (see IpcMapper).
constexpr size_t allocatorReserveSize = (size_t)1 << 40;

// Get number of cuMemCreate chunks
size_t allocatorGetChunkCount();

// Call fn(ctx, handle, offset, size) for every chunk (cuMemCreate handle, offset within the reserved range, size): the
// existing ones now, the ones mapped later as they are mapped (before any memory in them is handed out). fn runs with
// the allocator's lock held, so it must not allocate.
void allocatorAddChunkListener(void (*fn)(void* ctx, unsigned long long handle, size_t offset, size_t size), void* ctx);

// Get chunk info by index: handle, offset within reserved range, size.
// Returns true if index is valid, false otherwise.
bool allocatorGetChunk(size_t index, unsigned long long* outHandle, size_t* outOffset, size_t* outSize);

// Check if the allocator supports fabric handles
bool allocatorSupportsFabric();

// Communication pools. A communication pool places its allocations in a logical address space of its own,
// deterministically: the same sequence of allocations and frees (sizes and order) gives the same logical offsets in
// every process, whatever else the processes allocate and whenever their GPUs finish with freed memory. (An allocation
// that reuses freed memory makes its stream wait on the GPU for the streams that used the memory, instead of the host
// checking whether they have, which depends on timing.) The memory comes from the allocator's pool, in slabs: one
// contiguous allocation per slab, at any address, each at its own range of logical offsets (never adjacent, so that no
// allocation spans two slabs). A process group binds every slab at its logical offset into a multicast object of its
// own (see allocatorCommPoolAddSlabListener), so that a buffer that all ranks of the group allocate in the same order
// is at the same offset of that object on every rank. Slabs are never returned to the pool.
struct CommPool;

// Logical offsets are below this (the size of the multicast object they are bound into)
constexpr size_t commPoolLogicalSize = (size_t)1 << 40;

CommPool* allocatorCommPoolCreate();

// Allocate from the pool (bytes are rounded up to 4 KiB; the memory is 4 KiB aligned) for use on stream. The
// cleanupCtx is freed with cudaAllocatorImplFree, like those of cudaAllocatorImplAllocate.
CudaAllocation allocatorCommPoolAllocate(CommPool* pool, size_t bytes, CUstream stream, int deviceIndex);

// Call fn(ctx, logicalOffset, handle, memOffset, size) for every piece of every slab of the pool (the part of a slab
// in one cuMemCreate chunk: handle, the offset in it, size; logicalOffset is the piece's): for the existing slabs
// now, for new ones when they are added, before any memory in them is handed out. fn returns whether it bound the
// piece; the memory of a slab counts as in the pool for allocatorCommPoolOffset only if every piece was bound. fn runs
// with the allocator's lock held, so it must not allocate. All offsets and sizes are multiples of 2 MiB.
void allocatorCommPoolAddSlabListener(CommPool* pool,
    bool (*fn)(void* ctx, size_t logicalOffset, unsigned long long handle, size_t memOffset, size_t size), void* ctx);

// The logical offset of [address, address + bytes) if it lies in one slab of the pool whose pieces are all bound (see
// allocatorCommPoolAddSlabListener), else -1
int64_t allocatorCommPoolOffset(CommPool* pool, uintptr_t address, size_t bytes);

// The same for whichever communication pool [address, address + bytes) is in (memory is in at most one), or -1
int64_t allocatorAnyCommPoolOffset(uintptr_t address, size_t bytes);

} // namespace moodist
