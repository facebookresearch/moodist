// Copyright (c) Meta Platforms, Inc. and affiliates.

// compile_op implementation
// See compile_op.h for design overview
//
// Five-phase protocol:
// Phase 1: Exchange logical mappings (sender → receiver)
// Phase 2: Overlap resolution via cell decomposition (local)
// Phase 3: Send read requests (receiver → sender)
// Phase 4: Process requests, generate copies (sender)
// Phase 5: Finalize and generate reads (receiver)

#include "compile_op.h"
#include "queue.h"
#include "serialization.h"

#include <algorithm>
#include <cstring>
#include <memory>

namespace moodist {
namespace compile_op {

namespace {

// Helper to serialize data to a CPU TensorPtr for queue operations
template<typename... T>
TensorPtr serializeToTensorPtr(const T&... v) {
  auto buffer = serializeToBuffer(v...);
  size_t size = buffer->size();
  // Create a 1D uint8 CPU tensor and copy data into it
  int64_t shape = static_cast<int64_t>(size);
  CHECK(moodist::wrapperApi.tensorEmpty != nullptr);
  Tensor* t = moodist::wrapperApi.tensorEmpty(&shape, 1, DType::UInt8, -1);
  CHECK(t != nullptr);
  // Copy serialized data into tensor
  void* tensorData = moodist::wrapperApi.tensorDataPtr(t);
  std::memcpy(tensorData, buffer->data(), size);
  return TensorPtr(t);
}

// Helper to deserialize from TensorPtr
template<typename... T>
void deserializeFromTensorPtr(const TensorPtr& tensor, T&... result) {
  void* ptr = tensor.data_ptr();
  size_t size = static_cast<size_t>(tensor.numel());
  deserializeBuffer(ptr, size, result...);
}

} // namespace

std::shared_ptr<CustomOpDescriptor> compile(
    const CompileContext& ctx,
    DType dtype,
    std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> inputs,
    std::span<const std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>> outputs,
    ReduceOp reduce) {

  const size_t rank = ctx.rank;
  const size_t size = ctx.size;
  size_t itemsize = wrapperApi.dtypeSize(dtype);

  // Derive ndim from inputs/outputs
  // If this rank has no inputs/outputs, ndim doesn't matter (no Coords created)
  int ndim = 0;
  for (const auto& [r, off, sh] : inputs) {
    if (!off.empty()) {
      ndim = static_cast<int>(off.size());
      break;
    }
  }
  if (ndim == 0) {
    for (const auto& [r, off, sh] : outputs) {
      if (!off.empty()) {
        ndim = static_cast<int>(off.size());
        break;
      }
    }
  }

  // Validate input/output dimensions are consistent
  for (size_t i : indices(inputs)) {
    const auto& [r, off, sh] = inputs[i];
    if (ndim > 0 && (off.size() != static_cast<size_t>(ndim) || sh.size() != static_cast<size_t>(ndim))) {
      throw std::runtime_error(fmt::sprintf(
          "moodist.compile_op: input %zu has wrong dimensions (offset size %zu, shape size %zu, expected %d)",
          i, off.size(), sh.size(), ndim));
    }
    if (off.size() != sh.size()) {
      throw std::runtime_error(fmt::sprintf(
          "moodist.compile_op: input %zu has mismatched offset/shape sizes (%zu vs %zu)",
          i, off.size(), sh.size()));
    }
  }
  for (size_t i : indices(outputs)) {
    const auto& [r, off, sh] = outputs[i];
    if (ndim > 0 && (off.size() != static_cast<size_t>(ndim) || sh.size() != static_cast<size_t>(ndim))) {
      throw std::runtime_error(fmt::sprintf(
          "moodist.compile_op: output %zu has wrong dimensions (offset size %zu, shape size %zu, expected %d)",
          i, off.size(), sh.size(), ndim));
    }
    if (off.size() != sh.size()) {
      throw std::runtime_error(fmt::sprintf(
          "moodist.compile_op: output %zu has mismatched offset/shape sizes (%zu vs %zu)",
          i, off.size(), sh.size()));
    }
  }

  // Parse inputs and outputs into TensorDescr structs
  Vector<TensorDescr> inputDescrs;
  Vector<TensorDescr> outputDescrs;
  Vector<size_t> inputIndexCounter(size);
  Vector<size_t> outputIndexCounter(size);

  for (const auto& [r, off, sh] : inputs) {
    TensorDescr d;
    d.rank = r;
    d.index = inputIndexCounter[r]++;
    d.offset = Coord(ndim);
    d.shape = Coord(ndim);
    std::ranges::copy(off, d.offset.begin());
    std::ranges::copy(sh, d.shape.begin());
    d.numel = numel(d.shape);
    inputDescrs.push_back(std::move(d));
  }

  for (const auto& [r, off, sh] : outputs) {
    TensorDescr d;
    d.rank = r;
    d.index = outputIndexCounter[r]++;
    d.offset = Coord(ndim);
    d.shape = Coord(ndim);
    std::ranges::copy(off, d.offset.begin());
    std::ranges::copy(sh, d.shape.begin());
    d.numel = numel(d.shape);
    outputDescrs.push_back(std::move(d));
  }

  // Build lookup tables
  Vector<Vector<size_t>> inputsPerRank(size);
  Vector<Vector<size_t>> outputsPerRank(size);

  for (size_t i : indices(inputDescrs)) {
    inputsPerRank[inputDescrs[i].rank].push_back(i);
  }
  for (size_t i : indices(outputDescrs)) {
    outputsPerRank[outputDescrs[i].rank].push_back(i);
  }

  // Collect my inputs (inputs owned by this rank)
  Vector<size_t> myInputIndices = inputsPerRank[rank];

  // ============================================================================
  // Phase 1: Find intersections and exchange logical mappings
  // ============================================================================

  // For each output (from any rank), find intersecting inputs from myInputs
  // Create LogicalInput entries and group by destination rank
  Vector<Vector<LogicalInput>> logicalInputsToSend(size);

  for (size_t outIdx : indices(outputDescrs)) {
    const TensorDescr& out = outputDescrs[outIdx];
    if (out.numel == 0) {
      continue;
    }

    // Check each of my inputs for intersection with this output
    for (size_t myInpIdx : myInputIndices) {
      const TensorDescr& inp = inputDescrs[myInpIdx];

      // Check intersection
      bool intersects = true;
      for (int d = 0; d < ndim; ++d) {
        if (inp.offset[d] >= out.offset[d] + out.shape[d] ||
            out.offset[d] >= inp.offset[d] + inp.shape[d]) {
          intersects = false;
          break;
        }
      }

      if (!intersects) {
        continue;
      }

      // Compute intersection region
      LogicalInput li;
      li.offset = Coord(ndim);
      li.shape = Coord(ndim);
      for (int d = 0; d < ndim; ++d) {
        int64_t lo = std::max(inp.offset[d], out.offset[d]);
        int64_t hi = std::min(inp.offset[d] + inp.shape[d], out.offset[d] + out.shape[d]);
        li.offset[d] = lo;
        li.shape[d] = hi - lo;
      }
      li.inputRank = rank;
      li.inputIndex = inp.index;
      li.outputRank = out.rank;
      li.outputIndex = out.index;

      logicalInputsToSend[out.rank].push_back(std::move(li));
    }
  }

  // Barrier before queue operations
  ctx.barrier();

  // Send logical inputs to each destination rank
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(logicalInputsToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive logical inputs from all source ranks
  Vector<LogicalInput> receivedInputs;
  for (size_t i : range(size)) {
    auto [tensor, qsize] = ctx.queues[rank]->get();
    Vector<LogicalInput> batch;
    deserializeFromTensorPtr(tensor, batch);

    for (LogicalInput& li : batch) {
      CHECK(li.outputRank == rank);
      receivedInputs.push_back(std::move(li));
    }
  }

  // Debug logging
  log.debug("compile_op phase1: rank %zu, sent to %zu ranks, received %zu logical inputs\n",
      rank, size, receivedInputs.size());

  // ============================================================================
  // Phase 2: Overlap resolution via cell decomposition
  // ============================================================================

  // Group received inputs by output index
  size_t numMyOutputs = outputsPerRank[rank].size();
  Vector<Vector<size_t>> inputsPerOutput(numMyOutputs);
  for (size_t i : indices(receivedInputs)) {
    inputsPerOutput[receivedInputs[i].outputIndex].push_back(i);
  }

  // Result: for each output, a list of (cell, source input index) pairs
  // Each cell will become a ReadRequest in Phase 3
  struct CellSource {
    Coord offset;
    Coord shape;
    size_t sourceIdx;  // Index into receivedInputs
  };
  Vector<Vector<CellSource>> cellsPerOutput(numMyOutputs);

  for (size_t outIdx : range(numMyOutputs)) {
    size_t globalOutIdx = outputsPerRank[rank][outIdx];
    const TensorDescr& out = outputDescrs[globalOutIdx];
    const auto& inputIdxs = inputsPerOutput[outIdx];

    if (out.numel == 0) {
      continue;
    }

    // Collect boundaries in each dimension from output and all covering inputs
    Vector<Vector<int64_t>> boundaries(ndim);
    for (int d = 0; d < ndim; ++d) {
      boundaries[d].push_back(out.offset[d]);
      boundaries[d].push_back(out.offset[d] + out.shape[d]);
      for (size_t i : inputIdxs) {
        const LogicalInput& inp = receivedInputs[i];
        boundaries[d].push_back(inp.offset[d]);
        boundaries[d].push_back(inp.offset[d] + inp.shape[d]);
      }
      std::sort(boundaries[d].begin(), boundaries[d].end());
      boundaries[d].erase(std::unique(boundaries[d].begin(), boundaries[d].end()), boundaries[d].end());
    }

    // Compute total number of cells
    Vector<size_t> numIntervals(ndim);
    size_t totalCells = 1;
    for (int d = 0; d < ndim; ++d) {
      numIntervals[d] = boundaries[d].size() - 1;
      totalCells *= numIntervals[d];
    }

    // Iterate through all cells
    std::string gapDetails;
    std::string overlapDetails;
    size_t gapCount = 0;
    size_t overlapCount = 0;
    constexpr size_t maxReportedRegions = 5;

    for (size_t cellIdx = 0; cellIdx < totalCells; ++cellIdx) {
      // Convert cellIdx to multi-dimensional index and compute cell offset/shape
      Coord cellOffset(ndim);
      Coord cellShape(ndim);
      size_t remaining = cellIdx;
      for (int d = ndim - 1; d >= 0; --d) {
        size_t intervalIdx = remaining % numIntervals[d];
        remaining /= numIntervals[d];
        cellOffset[d] = boundaries[d][intervalIdx];
        cellShape[d] = boundaries[d][intervalIdx + 1] - boundaries[d][intervalIdx];
      }

      // Check if cell is inside output
      bool insideOutput = true;
      for (int d = 0; d < ndim; ++d) {
        if (cellOffset[d] < out.offset[d] ||
            cellOffset[d] + cellShape[d] > out.offset[d] + out.shape[d]) {
          insideOutput = false;
          break;
        }
      }
      if (!insideOutput) {
        continue;
      }

      // Find all inputs that cover this cell
      Vector<size_t> coveringInputs;
      for (size_t i : inputIdxs) {
        const LogicalInput& inp = receivedInputs[i];
        bool covers = true;
        for (int d = 0; d < ndim; ++d) {
          if (cellOffset[d] < inp.offset[d] ||
              cellOffset[d] + cellShape[d] > inp.offset[d] + inp.shape[d]) {
            covers = false;
            break;
          }
        }
        if (covers) {
          coveringInputs.push_back(i);
        }
      }

      if (coveringInputs.empty()) {
        // Gap detected
        gapCount++;
        if (gapCount <= maxReportedRegions) {
          gapDetails += fmt::sprintf("\n  output[%zu]: missing region at [%s] shape [%s]", outIdx,
              fmt::to_string(fmt::join(cellOffset, ",")).c_str(),
              fmt::to_string(fmt::join(cellShape, ",")).c_str());
        }
      } else if (coveringInputs.size() > 1) {
        // Overlap detected
        if (reduce == ReduceOp::None) {
          overlapCount++;
          if (overlapCount <= maxReportedRegions) {
            const LogicalInput& a = receivedInputs[coveringInputs[0]];
            const LogicalInput& b = receivedInputs[coveringInputs[1]];
            overlapDetails += fmt::sprintf(
                "\n  output[%zu]: overlap at [%s] shape [%s] between rank %u and rank %u",
                outIdx,
                fmt::to_string(fmt::join(cellOffset, ",")).c_str(),
                fmt::to_string(fmt::join(cellShape, ",")).c_str(),
                a.inputRank, b.inputRank);
          }
        } else {
          // reduce == Any: pick one randomly
          size_t pickIdx = random<size_t>(0, coveringInputs.size() - 1);
          CellSource cs;
          cs.offset = std::move(cellOffset);
          cs.shape = std::move(cellShape);
          cs.sourceIdx = coveringInputs[pickIdx];
          cellsPerOutput[outIdx].push_back(std::move(cs));
        }
      } else {
        // Exactly one covering input
        CellSource cs;
        cs.offset = std::move(cellOffset);
        cs.shape = std::move(cellShape);
        cs.sourceIdx = coveringInputs[0];
        cellsPerOutput[outIdx].push_back(std::move(cs));
      }
    }

    // Report errors
    if (gapCount > 0) {
      std::string msg = "moodist.compile_op: missing input coverage";
      if (gapCount > maxReportedRegions) {
        msg += fmt::sprintf(" (%zu regions, showing first %zu):", gapCount, maxReportedRegions);
      } else {
        msg += ":";
      }
      msg += gapDetails;
      throw std::runtime_error(msg);
    }

    if (overlapCount > 0) {
      std::string msg = "moodist.compile_op: overlapping inputs detected";
      if (overlapCount > maxReportedRegions) {
        msg += fmt::sprintf(" (%zu regions, showing first %zu):", overlapCount, maxReportedRegions);
      } else {
        msg += ":";
      }
      msg += overlapDetails;
      throw std::runtime_error(msg);
    }
  }

  log.debug("compile_op phase2: rank %zu, completed cell decomposition\n", rank);

  // ============================================================================
  // Phase 3: Send read requests (receiver → sender)
  // ============================================================================

  // Convert cells to ReadRequests, grouped by source rank
  Vector<Vector<ReadRequest>> requestsToSend(size);

  for (size_t outIdx : range(numMyOutputs)) {
    for (const auto& cell : cellsPerOutput[outIdx]) {
      const LogicalInput& src = receivedInputs[cell.sourceIdx];

      ReadRequest req;
      req.offset = cell.offset;
      req.shape = cell.shape;
      req.requesterRank = static_cast<uint32_t>(rank);
      req.inputIndex = src.inputIndex;
      req.outputIndex = static_cast<uint32_t>(outIdx);

      requestsToSend[src.inputRank].push_back(std::move(req));
    }
  }

  // Barrier before queue operations
  ctx.barrier();

  // Send read requests to each source rank
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(requestsToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive read requests from all ranks (as a sender)
  Vector<ReadRequest> receivedRequests;
  for (size_t i : range(size)) {
    (void)i;
    auto [tensor, qsize] = ctx.queues[rank]->get();
    Vector<ReadRequest> batch;
    deserializeFromTensorPtr(tensor, batch);
    for (auto& req : batch) {
      receivedRequests.push_back(std::move(req));
    }
  }

  log.debug("compile_op phase3: rank %zu, received %zu read requests\n", rank, receivedRequests.size());

  // ============================================================================
  // Phase 4: Process requests, generate copies, send responses (sender)
  // ============================================================================

  // Track copies needed for non-contiguous reads from my inputs
  struct InputCopy {
    uint32_t inputIndex;  // Which of my inputs
    Coord offset;         // Offset within that input
    Coord shape;          // Shape of the region
  };
  Vector<InputCopy> inputCopies;

  // Process each request and build responses
  Vector<Vector<ReadResponse>> responsesToSend(size);

  for (const auto& req : receivedRequests) {
    // Find my input
    CHECK(req.inputIndex < myInputIndices.size());
    size_t globalInputIdx = myInputIndices[req.inputIndex];
    const TensorDescr& inp = inputDescrs[globalInputIdx];

    // Compute offset within input (request offset is in global coords)
    Coord relOffset = req.offset - inp.offset;

    // Check if this region is contiguous within the input tensor
    bool isContiguous = contiguous(req.shape, inp.shape);

    ReadResponse resp;
    resp.requesterRank = req.requesterRank;
    resp.outputIndex = req.outputIndex;
    resp.requestOffset = req.offset;
    resp.requestShape = req.shape;
    resp.senderRank = static_cast<uint32_t>(rank);

    if (isContiguous) {
      // Can read directly from input
      resp.tensorIndex = req.inputIndex;
      resp.inputOffset = linearOffset(relOffset, inp.shape);
      resp.tensorShape = inp.shape;
      resp.isCopy = false;
    } else {
      // Need to make a contiguous copy
      uint32_t copyIndex = static_cast<uint32_t>(myInputIndices.size() + inputCopies.size());

      InputCopy copy;
      copy.inputIndex = req.inputIndex;
      copy.offset = relOffset;
      copy.shape = req.shape;
      inputCopies.push_back(std::move(copy));

      resp.tensorIndex = copyIndex;
      resp.inputOffset = 0;  // Copy is contiguous, starts at 0
      resp.tensorShape = req.shape;  // Copy has same shape as request
      resp.isCopy = true;
    }

    responsesToSend[req.requesterRank].push_back(std::move(resp));
  }

  // Barrier before sending responses
  ctx.barrier();

  // Send responses to each requester
  for (size_t destRank : range(size)) {
    TensorPtr tensor = serializeToTensorPtr(responsesToSend[destRank]);
    ctx.queues[destRank]->put(std::move(tensor), 0);
  }

  // Receive responses from all ranks
  Vector<ReadResponse> receivedResponses;
  for (size_t i : range(size)) {
    (void)i;
    auto [tensor, qsize] = ctx.queues[rank]->get();
    Vector<ReadResponse> batch;
    deserializeFromTensorPtr(tensor, batch);
    for (auto& resp : batch) {
      receivedResponses.push_back(std::move(resp));
    }
  }

  log.debug("compile_op phase4: rank %zu, %zu input copies, received %zu responses\n",
      rank, inputCopies.size(), receivedResponses.size());

  // ============================================================================
  // Phase 5: Finalize and build CustomOpDescriptor (receiver)
  // ============================================================================

  auto op = std::make_shared<CustomOpDescriptor>();
  op->id = (*ctx.nextOpId)++;
  op->dtype = dtype;

  // Populate inputs (my inputs) - sizes in bytes
  for (size_t idx : myInputIndices) {
    op->inputs.push_back(itemsize * inputDescrs[idx].numel);
    const Coord& sh = inputDescrs[idx].shape;
    op->inputShapes.emplace_back(sh.begin(), sh.end());
  }

  // Populate outputs (my outputs) - sizes in bytes
  for (size_t idx : outputsPerRank[rank]) {
    op->outputs.push_back(itemsize * outputDescrs[idx].numel);
    const Coord& sh = outputDescrs[idx].shape;
    op->outputShapes.emplace_back(sh.begin(), sh.end());
  }

  // Add inputCopies from Phase 4
  for (const auto& copy : inputCopies) {
    CustomOpDescriptor::Copy c;
    c.index = copy.inputIndex;
    c.offset = {copy.offset.begin(), copy.offset.end()};
    c.shape = {copy.shape.begin(), copy.shape.end()};
    op->inputCopies.push_back(std::move(c));
  }

  // Track output copies needed for non-contiguous writes
  struct OutputCopy {
    uint32_t outputIndex;
    Coord offset;  // Relative to output
    Coord shape;
  };
  Vector<OutputCopy> outputCopyList;

  // Process responses and build reads
  for (const auto& resp : receivedResponses) {
    // Find my output
    CHECK(resp.outputIndex < outputsPerRank[rank].size());
    size_t globalOutputIdx = outputsPerRank[rank][resp.outputIndex];
    const TensorDescr& out = outputDescrs[globalOutputIdx];

    // Compute output offset (request offset is in global coords)
    Coord relOutOffset = resp.requestOffset - out.offset;
    size_t outputOffset = linearOffset(relOutOffset, out.shape);

    // Check if this region is contiguous within the output tensor
    bool outputContiguous = contiguous(resp.requestShape, out.shape);

    // Compute bytes
    size_t bytes = itemsize * numel(resp.requestShape);

    // Check if input side is contiguous (for the Read entry)
    bool inputContiguous = contiguous(resp.requestShape, resp.tensorShape);

    CustomOpDescriptor::Read read;
    read.rank = resp.senderRank;
    read.inputIndex = resp.tensorIndex;
    read.outputIndex = resp.outputIndex;
    read.inputOffset = itemsize * resp.inputOffset;  // Convert to bytes
    read.bytes = bytes;

    if (outputContiguous) {
      read.outputOffset = itemsize * outputOffset;  // Convert to bytes
    } else {
      // Need to copy to a contiguous region, then write to output
      uint32_t copyIdx = static_cast<uint32_t>(outputsPerRank[rank].size() + outputCopyList.size());

      OutputCopy oc;
      oc.outputIndex = resp.outputIndex;
      oc.offset = relOutOffset;
      oc.shape = resp.requestShape;
      outputCopyList.push_back(std::move(oc));

      // Read into the copy buffer (which will be contiguous)
      read.outputIndex = copyIdx;
      read.outputOffset = 0;
    }

    // For now, use simple reads (no NVLink optimization)
    // TODO: Add NVLink path (gatewayReads, localCopies, localInputCopies)
    op->reads.push_back(read);

    (void)inputContiguous; // Used by execution path, not stored in descriptor
  }

  // Add outputCopies
  for (const auto& oc : outputCopyList) {
    CustomOpDescriptor::Copy c;
    c.index = oc.outputIndex;
    c.offset = {oc.offset.begin(), oc.offset.end()};
    c.shape = {oc.shape.begin(), oc.shape.end()};
    op->outputCopies.push_back(std::move(c));
  }

  // Final barrier
  ctx.barrier();

  log.debug("compile_op phase5: rank %zu, %zu reads, %zu output copies\n",
      rank, op->reads.size(), op->outputCopies.size());

  return op;
}

} // namespace compile_op
} // namespace moodist
