/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "function.h"

#include <memory>
#include <string>

namespace moodist {

namespace async {

void setCurrentThreadName(std::string name);

struct SchedulerFifoImpl;

struct SchedulerFifo {

  std::unique_ptr<SchedulerFifoImpl> impl_;

  SchedulerFifo();
  SchedulerFifo(size_t nThreads);
  ~SchedulerFifo();

  void run(Function<void()> f) noexcept;
  void setMaxThreads(size_t nThreads);
  size_t getMaxThreads() const;
  bool isInThread() const noexcept;
};

} // namespace async
} // namespace moodist
