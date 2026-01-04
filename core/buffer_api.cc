// Copyright (c) Meta Platforms, Inc. and affiliates.

// Buffer API implementation - no Python/pybind11 dependency.
// These functions allow external libraries (like serialize) to use core's
// internal allocator for Buffer objects.

#include "api/types.h"
#include "buffer.h"

namespace moodist {

api::Buffer* bufferAllocate(size_t nbytes) {
  return Buffer::allocate(nbytes);
}

void* serializeBufferPtr(api::Buffer* buf) {
  return static_cast<Buffer*>(buf)->data();
}

size_t serializeBufferSize(api::Buffer* buf) {
  return static_cast<Buffer*>(buf)->size();
}

void bufferSetSize(api::Buffer* buf, size_t size) {
  static_cast<Buffer*>(buf)->msize = size;
}

void bufferDestroy(api::Buffer* buf) {
  Buffer::deallocate(static_cast<Buffer*>(buf));
}

// destroy() implementation for api::Buffer - called by ApiHandle destructor
namespace api {
void destroy(Buffer* buffer) {
  moodist::bufferDestroy(buffer);
}
} // namespace api

} // namespace moodist
