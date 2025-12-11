// Copyright (c) Meta Platforms, Inc. and affiliates.

// Serialization C API for moodist.
// This header can be included by both core library and wrapper.

#pragma once

#include <Python.h>
#include <cstddef>

#ifndef MOODIST_API
#define MOODIST_API __attribute__((visibility("default")))
#endif

namespace moodist {

// Forward declare Buffer (opaque type)
struct Buffer;

// Serialize object, returns Buffer* with refcount=1 (caller must call serializeBufferDecRef)
MOODIST_API Buffer* serializeObjectImpl(PyObject* o);

// Access buffer data
MOODIST_API void* serializeBufferPtr(Buffer* buf);
MOODIST_API size_t serializeBufferSize(Buffer* buf);

// Ref counting - prevent destructor issues by using explicit functions
MOODIST_API void serializeBufferAddRef(Buffer* buf);
MOODIST_API void serializeBufferDecRef(Buffer* buf);

// Deserialize from raw pointer/size (returns new reference)
MOODIST_API PyObject* deserializeObjectImpl(const void* ptr, size_t len);

} // namespace moodist
