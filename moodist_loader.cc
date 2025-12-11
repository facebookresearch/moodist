// Copyright (c) Meta Platforms, Inc. and affiliates.

// Loads libmoodist.so via dlopen and gets the API struct

#include "moodist_api.h"

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace moodist {

namespace {

void* libHandle = nullptr;
MoodistAPI* api = nullptr;

std::string getLibraryPath() {
  // Get path to this module (_C.so)
  Dl_info info;
  if (dladdr((void*)&getLibraryPath, &info) == 0) {
    throw std::runtime_error("Failed to get _C.so path via dladdr");
  }

  std::string path(info.dli_fname);
  size_t lastSlash = path.rfind('/');
  if (lastSlash == std::string::npos) {
    return "libmoodist.so";
  }

  // Try same directory first
  std::string sameDirPath = path.substr(0, lastSlash + 1) + "libmoodist.so";
  if (access(sameDirPath.c_str(), F_OK) == 0) {
    return sameDirPath;
  }

  // Try parent directory (for multi-version wheels)
  std::string parentPath = path.substr(0, lastSlash);
  lastSlash = parentPath.rfind('/');
  if (lastSlash != std::string::npos) {
    std::string parentDirPath = parentPath.substr(0, lastSlash + 1) + "libmoodist.so";
    if (access(parentDirPath.c_str(), F_OK) == 0) {
      return parentDirPath;
    }
  }

  // Fall back to just the name (let dlopen search)
  return "libmoodist.so";
}

} // namespace

MoodistAPI* getMoodistAPI() {
  if (api != nullptr) {
    return api;
  }

  std::string libPath = getLibraryPath();

  libHandle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    throw std::runtime_error(std::string("Failed to load libmoodist.so: ") + dlerror());
  }

  auto getAPI = reinterpret_cast<MoodistAPI* (*)(uint32_t)>(dlsym(libHandle, "moodistGetAPI"));
  if (getAPI == nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error(std::string("Failed to find moodistGetAPI: ") + dlerror());
  }

  api = getAPI(kMoodistAPIVersion);
  if (api == nullptr) {
    dlclose(libHandle);
    libHandle = nullptr;
    throw std::runtime_error("moodist API version mismatch: _C.so and libmoodist.so are incompatible");
  }

  if (api->magic != kMoodistBuildMagic) {
    dlclose(libHandle);
    libHandle = nullptr;
    api = nullptr;
    throw std::runtime_error("moodist build magic mismatch: _C.so and libmoodist.so are from different builds");
  }

  return api;
}

} // namespace moodist
