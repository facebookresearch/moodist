// Copyright (c) Meta Platforms, Inc. and affiliates.

// Serialization C API for moodist.
// This header can be included by both core library and wrapper.

#pragma once

#include "types.h"

#include <Python.h>
#include <cstddef>

#ifndef MOODIST_API
#define MOODIST_API __attribute__((visibility("default")))
#endif

namespace moodist {

// Serialize object, returns api::Buffer* with refcount=1 (caller must call bufferDestroy)
MOODIST_API api::Buffer* serializeObjectImpl(PyObject* o);

// Access buffer data
MOODIST_API void* serializeBufferPtr(api::Buffer* buf);
MOODIST_API size_t serializeBufferSize(api::Buffer* buf);

// Destroy buffer (called by ApiHandle when refcount reaches zero)
MOODIST_API void bufferDestroy(api::Buffer* buf);

// Deserialize from raw pointer/size (returns new reference)
MOODIST_API PyObject* deserializeObjectImpl(const void* ptr, size_t len);

} // namespace moodist
