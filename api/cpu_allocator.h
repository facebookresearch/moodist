// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace moodist {

namespace cpu_allocator {
bool owns(uintptr_t);
std::pair<uintptr_t, size_t> regionAt(uintptr_t);
bool owns(const void*);
std::pair<uintptr_t, size_t> regionAt(const void*);

void* moo_alloc(size_t);
void moo_free(void*);

} // namespace cpu_allocator

// CoreApi functions for CPU allocator
// cpuAllocatorAlloc: allocates bytes, returns ptr and sets *cleanupCtx for deleter
// cpuAllocatorFree: deleter function (void(*)(void*)) suitable for DataPtr
void* cpuAllocatorAlloc(size_t bytes, void** cleanupCtx);
void cpuAllocatorFree(void* cleanupCtx);

#ifdef MOODIST_WRAPPER
// Enable the moodist CPU allocator as PyTorch's CPU allocator
void enableCpuAllocator();
#endif

} // namespace moodist
