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

struct SchedulerImpl;

struct Scheduler {

  std::unique_ptr<SchedulerImpl> impl_;

  Scheduler();
  Scheduler(size_t nThreads);
  ~Scheduler();

  void run(Function<void()> f) noexcept;
  void setMaxThreads(size_t nThreads);
  size_t getMaxThreads() const;
  bool isInThread() const noexcept;
  void setName(std::string name);
};

} // namespace async
} // namespace moodist
