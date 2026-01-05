// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup API for moodist.
// Uses ApiHandle pattern for refcounted ownership across the API boundary.
// Uses TensorPtr for tensor operations - works in both wrapper and core.

#pragma once

#include "tensor_ptr.h"
#include "types.h"

#include <span>

namespace moodist {

using namespace cuda;

// Factory function - returns ApiHandle (ownership via RVO)
// c10dStore is opaque pointer to c10d::Store - core calls back via WrapperApi for store ops
api::ProcessGroupHandle createProcessGroup(void* c10dStore, int rank, int size);

// Destroy function - called by ApiHandle when refcount reaches 0
void processGroupDestroy(api::ProcessGroup* pg);

// Optional: explicitly trigger shutdown before last reference is dropped
void processGroupShutdown(api::ProcessGroup* pg);

// Accessors
int processGroupRank(api::ProcessGroup* pg);
int processGroupSize(api::ProcessGroup* pg);
bool processGroupGetPreferKernelLess(api::ProcessGroup* pg);
void processGroupSetPreferKernelLess(api::ProcessGroup* pg, bool value);

// Generic option get/set by name
// Throws std::runtime_error for unknown options
// Bool options: returns/accepts 0 or 1
// Int options: -1 = auto, >=0 = value
int64_t processGroupGetOption(api::ProcessGroup* pg, const char* name);
void processGroupSetOption(api::ProcessGroup* pg, const char* name, int64_t value);

// Collective operations - all logic handled in core
void processGroupAllGather(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, CUstream stream);

void processGroupReduceScatter(api::ProcessGroup* pg, TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp,
    CUstream stream, float premulValue);

void processGroupAllreduce(
    api::ProcessGroup* pg, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue);

void processGroupBroadcast(api::ProcessGroup* pg, TensorPtr& tensor, int sourceRank, CUstream stream);

void processGroupReduce(api::ProcessGroup* pg, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream);

void processGroupBarrier(api::ProcessGroup* pg);

void processGroupScatter(
    api::ProcessGroup* pg, std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream);

void processGroupGather(
    api::ProcessGroup* pg, std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream);

void processGroupAllToAll(
    api::ProcessGroup* pg, std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream);

void processGroupCudaBarrier(api::ProcessGroup* pg, CUstream stream);

// Profiling
void setProfilingEnabled(bool enabled);
bool getProfilingEnabled();

} // namespace moodist
