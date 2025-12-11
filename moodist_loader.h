// Copyright (c) Meta Platforms, Inc. and affiliates.

// Header for loading the moodist API in the wrapper library

#pragma once

#include "moodist_api.h"

namespace moodist {

// Get the MoodistAPI struct, loading libmoodist.so if needed
// This is the wrapper-side function that uses dlopen/dlsym
MoodistAPI* getMoodistAPI();

} // namespace moodist
