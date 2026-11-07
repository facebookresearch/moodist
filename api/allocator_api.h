// Copyright (c) Meta Platforms, Inc. and affiliates.

// CUDA Allocator API for moodist.
// Uses ABI-safe types at the library boundary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
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

// Thrown by the allocation functions when the allocator cannot get enough memory from the device (other errors throw
// std::runtime_error). The wrapper turns it into PyTorch's OutOfMemoryError.
struct CudaOutOfMemoryError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// CoreApi functions for CUDA allocator

// The allocator is global: the CudaAllocatorImpl* arguments below are ignored (pass nullptr). These two remain for
// the CoreApi layout: createCudaAllocatorImpl returns nullptr, setCudaAllocatorImpl does nothing.
CudaAllocatorImpl* createCudaAllocatorImpl();
void setCudaAllocatorImpl(CudaAllocatorImpl* impl);

// Allocate memory for use on stream (see the top of allocator.cc): from the private pool that allocations on stream
// are routed to (see cudaAllocatorBeginAllocateToPool), if any, else from the main pool
// - bytes: number of bytes to allocate (will be aligned internally)
// - stream: CUDA stream for allocation
// - deviceIndex: CUDA device index (-1 to auto-detect from current context)
// Returns: CudaAllocation with ptr, cleanupCtx, and actual deviceIndex. Throws CudaOutOfMemoryError if out of memory.
CudaAllocation cudaAllocatorImplAllocate(CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex);

// Free memory using cleanup context (the main deallocation path). The memory is reused only by work that comes after
// the work queued by then on the stream it was allocated on and on the streams recorded for it
// (cudaAllocatorImplRecordStream).
// - cleanupCtx: context returned from cudaAllocatorImplAllocate (or allocatorCommPoolAllocate)
// - stream: unused
void cudaAllocatorImplFree(void* cleanupCtx, CUstream stream);

// Record that a stream has used an allocation (for recordStream)
// This must be called when torch's recordStream is invoked
void cudaAllocatorImplRecordStream(CudaAllocatorImpl* impl, uintptr_t ptr, CUstream stream);

// A statistic of the allocator, like PyTorch's c10::CachingAllocator::Stat: its current value, its peak, and the sums
// of its increases and decreases (peak and sums since the last reset, see cudaAllocatorResetStats)
struct CudaAllocatorStat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

// The allocator's statistics, for PyTorch's DeviceStats (torch.cuda.memory_stats). The arrays of 3 are for all
// allocations, those of at most 1 MiB, and larger ones (PyTorch's aggregate, small pool and large pool).
struct CudaAllocatorStats {
  CudaAllocatorStat allocations[3];    // allocations of the main pool and of private pools, not freed yet
  CudaAllocatorStat allocatedBytes[3]; // their sizes as allocated (rounded up, with alignment padding)
  CudaAllocatorStat requestedBytes[3]; // their sizes as requested
  CudaAllocatorStat reservedBytes;     // memory mapped (never unmapped)
  CudaAllocatorStat segments;          // chunks mapped
  int64_t numOoms = 0;                 // allocations that failed for lack of memory
  int64_t numMaps = 0;                 // chunks mapped
  size_t largestFreeBlock = 0;         // the largest free region of the main pool
};

void cudaAllocatorGetStats(CudaAllocatorStats* out);
// The most memory the allocator maps (0: no limit, the default): once that much is mapped, allocations reuse free
// memory (see pressure mode in allocator.cc) or fail. Memory mapped already stays mapped.
void cudaAllocatorSetMemoryLimit(size_t bytes);
// Resets the peaks to the current values, and/or the sums (and numOoms, numMaps) to zero
void cudaAllocatorResetStats(bool peaks, bool accumulated);

// Private pools: PyTorch's memory pools (MempoolId_t {id0, id1}), e.g. those of CUDA graphs. A pool exists from its
// first cudaAllocatorCreateOrIncrefPool or cudaAllocatorBeginAllocateToPool (which count a use each) until
// cudaAllocatorReleasePool has released all uses; its free memory, and memory still allocated from it once freed, then
// go back to the main pool after a device sync, which the allocator does on cudaAllocatorEmptyCache and before an
// allocation would enter pressure mode or fail, if no capture is in progress (as PyTorch's allocator, with cudaFree).
// Between cudaAllocatorBeginAllocateToPool and cudaAllocatorEndAllocateToPool, allocations on the streams that
// filter(filterCtx, stream) returns true for come from the pool (PyTorch's beginAllocateToPool; the most recent scope
// first); the allocator calls destroyFilterCtx(filterCtx) at the end. filter is called with the allocator's lock held.
// See allocator.cc ("CUDA graphs") for allocations during stream capture.
using CudaAllocatorPoolFilter = bool (*)(void* filterCtx, CUstream stream);
void cudaAllocatorBeginAllocateToPool(
    uint64_t id0, uint64_t id1, CudaAllocatorPoolFilter filter, void* filterCtx, void (*destroyFilterCtx)(void*));
void cudaAllocatorEndAllocateToPool(uint64_t id0, uint64_t id1);
void cudaAllocatorReleasePool(uint64_t id0, uint64_t id1);
void cudaAllocatorCreateOrIncrefPool(uint64_t id0, uint64_t id1);
// The pool's uses (0 if there is no such pool)
int cudaAllocatorGetPoolUseCount(uint64_t id0, uint64_t id1);
// PyTorch's emptyCache: released pools' memory goes back to the main pool (after a device sync) if no capture is in
// progress. Memory stays mapped.
void cudaAllocatorEmptyCache();
// PyTorch's capture marks (torch 2.13+): markCaptureBegin right after cudaStreamBeginCapture and markCaptureEnd right
// after cudaStreamEndCapture, for every capture (conditional nodes' bodies too); a capture's streams still exist at its
// end mark (PyTorch 2.13 destroys a conditional node's stream after it, before endAllocateToPool).
// cudaAllocatorSetCaptureMarks(true) says that PyTorch makes them: then routing to a pool without a capture
// (torch.cuda.use_mem_pool, cudagraph trees' warmup) does not keep the allocator from syncing the device.
void cudaAllocatorSetCaptureMarks(bool enabled);
void cudaAllocatorMarkCaptureBegin();
void cudaAllocatorMarkCaptureEnd();

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

// Get chunk info by index: handle, offset within reserved range, size.
// Returns true if index is valid, false otherwise.
bool allocatorGetChunk(size_t index, unsigned long long* outHandle, size_t* outOffset, size_t* outSize);

// Check if the allocator supports fabric handles
bool allocatorSupportsFabric();

// Communication pools. A communication pool places its allocations in a logical address space of its own,
// deterministically: the same sequence of allocations and frees (sizes, streams and order) gives the same logical
// offsets in every process, whatever else the processes allocate and whenever their GPUs finish with freed memory.
// (Freed memory is reused by allocations on the stream it was allocated on, else the pool takes new memory; see the
// top of allocator.cc. Out of device memory, an allocation fails rather than take memory freed on other streams, whose
// choice would depend on timing.) The memory comes from the allocator's pool, in slabs:
// one contiguous allocation per slab, at any address, each at its own range of logical offsets (never adjacent, so that
// no allocation spans two slabs). A process group binds every slab at its logical offset into a multicast object of its
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

// How many slabs the pool has added so far (without the allocator's lock). An address keeps its slab (slabs are never
// freed), but memory at an address outside the pool can be in it once a slab has been added: a new slab is taken from
// the main pool's memory, at any address, also one where memory of the main pool was before.
uint64_t allocatorCommPoolSlabCount(CommPool* pool);

// The same for whichever communication pool [address, address + bytes) is in (memory is in at most one), or -1
int64_t allocatorAnyCommPoolOffset(uintptr_t address, size_t bytes);

} // namespace moodist
