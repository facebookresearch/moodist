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

} // namespace moodist
