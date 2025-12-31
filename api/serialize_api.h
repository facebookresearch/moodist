// Copyright (c) Meta Platforms, Inc. and affiliates.

// Serialization API for moodist.
// These functions are passed via CoreApi struct, not looked up via dlsym,
// so they use default (hidden) visibility like all other API functions.

#pragma once

#include "types.h"

#include <Python.h>
#include <cstddef>

namespace moodist {

// Serialize object, returns ApiHandle<api::Buffer> with refcount=1
api::BufferHandle serializeObjectImpl(PyObject* o);

// Access buffer data
void* serializeBufferPtr(api::Buffer* buf);
size_t serializeBufferSize(api::Buffer* buf);

// Destroy buffer (called by ApiHandle when refcount reaches zero)
void bufferDestroy(api::Buffer* buf);

// Deserialize from raw pointer/size (returns new reference)
PyObject* deserializeObjectImpl(const void* ptr, size_t len);

} // namespace moodist
