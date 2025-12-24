// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace moodist {

// Forward declare CudaAllocatorImpl (opaque type defined in core)
struct CudaAllocatorImpl;

namespace allocator {

// Query if an address is owned by the moodist allocator
bool owns(uintptr_t address);

// Get the mapped region containing an address
std::pair<uintptr_t, size_t> mappedRegion(uintptr_t address);

} // namespace allocator

#ifdef MOODIST_WRAPPER
// Enable the moodist CUDA allocator as PyTorch's CUDA allocator
void enableCudaAllocator();
#endif

} // namespace moodist
