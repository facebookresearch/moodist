// Copyright (c) Meta Platforms, Inc. and affiliates.

// CUDA Allocator API for moodist.
// Uses ABI-safe types at the library boundary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

namespace moodist {

// Forward declare CudaAllocatorImpl (opaque type in core)
struct CudaAllocatorImpl;

// Factory function to create CudaAllocatorImpl
CudaAllocatorImpl* createCudaAllocatorImpl();

// Set the global allocator impl pointer (for allocator::owns/mappedRegion)
void setCudaAllocatorImpl(CudaAllocatorImpl* impl);

// Allocate memory
// Returns: pointer to allocated memory, sets *cleanupCtx for deallocation
// deviceIndex: CUDA device index (from c10::cuda::current_device in wrapper)
// stream: CUDA stream for allocation
uintptr_t cudaAllocatorImplAllocate(
    CudaAllocatorImpl* impl, size_t bytes, CUstream stream, int deviceIndex, void** cleanupCtx);

// Deallocate memory
// streams: array of streams that have used this allocation
// streamCount: number of streams in the array
void cudaAllocatorImplDeallocate(
    CudaAllocatorImpl* impl, uintptr_t ptr, size_t bytes, CUstream* streams, size_t streamCount);

// Deallocate memory using cleanup context
void cudaAllocatorImplFree(void* cleanupCtx);

// Record a stream use for an allocation (for recordStream)
void cudaAllocatorImplRecordStream(CudaAllocatorImpl* impl, uintptr_t ptr, CUstream stream);

// Query functions (work with global impl)
bool allocatorOwns(uintptr_t address);
std::pair<uintptr_t, size_t> allocatorMappedRegion(uintptr_t address);

// FreeCallback API
void* allocatorAddFreeCallback(uintptr_t baseAddress, void* callbackCtx);
void allocatorRemoveFreeCallback(uintptr_t baseAddress, void* handle);

} // namespace moodist
