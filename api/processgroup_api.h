// Copyright (c) Meta Platforms, Inc. and affiliates.

// ProcessGroup API for moodist.
// Uses TensorPtr for tensor operations - works in both wrapper and core.

#pragma once

#include "tensor_ptr.h"

#include <cuda.h>
#include <span>

namespace moodist {

// Forward declarations
struct ProcessGroupImpl;

// ProcessGroupImpl creation/destruction
// c10dStore is opaque pointer to c10d::Store - core calls back via WrapperApi for store ops
ProcessGroupImpl* createProcessGroupImpl(void* c10dStore, int rank, int size);
void processGroupImplAddRef(ProcessGroupImpl* impl);
void processGroupImplDecRef(ProcessGroupImpl* impl);

// Accessors
int processGroupImplRank(ProcessGroupImpl* impl);
int processGroupImplSize(ProcessGroupImpl* impl);

// Collective operations - all logic handled in core
void processGroupImplAllGather(ProcessGroupImpl* impl, TensorPtr& output, const TensorPtr& input, CUstream stream);

void processGroupImplReduceScatter(ProcessGroupImpl* impl, TensorPtr& output, const TensorPtr& input, ReduceOp reduceOp,
    CUstream stream, float premulValue);

void processGroupImplAllreduce(
    ProcessGroupImpl* impl, TensorPtr& tensor, ReduceOp reduceOp, CUstream stream, float premulValue);

void processGroupImplBroadcast(ProcessGroupImpl* impl, TensorPtr& tensor, int sourceRank, CUstream stream);

void processGroupImplReduce(
    ProcessGroupImpl* impl, TensorPtr& tensor, int destRank, ReduceOp reduceOp, CUstream stream);

void processGroupImplBarrier(ProcessGroupImpl* impl);

void processGroupImplScatter(
    ProcessGroupImpl* impl, std::span<TensorPtr> inputs, TensorPtr& output, int sourceRank, CUstream stream);

void processGroupImplGather(
    ProcessGroupImpl* impl, std::span<TensorPtr> outputs, const TensorPtr& input, int destRank, CUstream stream);

void processGroupImplAllToAll(
    ProcessGroupImpl* impl, std::span<TensorPtr> outputs, std::span<TensorPtr> inputs, CUstream stream);

void processGroupImplCudaBarrier(ProcessGroupImpl* impl, CUstream stream);

void processGroupImplShutdown(ProcessGroupImpl* impl);

} // namespace moodist
