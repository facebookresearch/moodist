#pragma once

#include "fmt/printf.h"

#include <mutex>

namespace moodist {

enum class LogLevel {
  LOG_NONE,
  LOG_ERROR,
  LOG_INFO,
  LOG_VERBOSE,
  LOG_DEBUG,
};

constexpr auto LOG_NONE = LogLevel::LOG_NONE;
constexpr auto LOG_ERROR = LogLevel::LOG_ERROR;
constexpr auto LOG_INFO = LogLevel::LOG_INFO;
constexpr auto LOG_VERBOSE = LogLevel::LOG_VERBOSE;
constexpr auto LOG_DEBUG = LogLevel::LOG_DEBUG;

inline LogLevel currentLogLevel = LOG_INFO;

inline std::mutex logMutex;

template<typename... Args>
[[gnu::cold]] void logat(LogLevel level, const char* fmt, Args&&... args) {
  std::lock_guard l(logMutex);
  FILE* ftarget = stdout;
  FILE* fother = stderr;
  if (level == LOG_ERROR) {
    std::swap(ftarget, fother);
  }
  fflush(fother);
  time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto* tm = std::localtime(&now);
  char buf[0x40];
  std::strftime(buf, sizeof(buf), "%d-%m-%Y %H:%M:%S", tm);
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  if (!s.empty() && s.back() == '\n') {
    fmt::fprintf(ftarget, "%s moodist: %s", buf, s);
  } else {
    fmt::fprintf(ftarget, "%s moodist: %s\n", buf, s);
  }
  fflush(ftarget);
}

inline struct Log {
  template<typename... Args>
  void error(const char* fmt, Args&&... args) {
    logat(LOG_ERROR, fmt, std::forward<Args>(args)...);
  }
  template<typename... Args>
  void info(const char* fmt, Args&&... args) {
    if (currentLogLevel >= LOG_INFO) {
      [[unlikely]];
      logat(LOG_INFO, fmt, std::forward<Args>(args)...);
    }
  }
  template<typename... Args>
  void verbose(const char* fmt, Args&&... args) {
    if (currentLogLevel >= LOG_VERBOSE) {
      [[unlikely]];
      logat(LOG_VERBOSE, fmt, std::forward<Args>(args)...);
    }
  }
  template<typename... Args>
  void debug(const char* fmt, Args&&... args) {
    if (currentLogLevel >= LOG_DEBUG) {
      [[unlikely]];
      logat(LOG_DEBUG, fmt, std::forward<Args>(args)...);
    }
  }
} log;

template<typename... Args>
[[noreturn]] [[gnu::cold]] void fatal(const char* fmt, Args&&... args) {
  auto s = fmt::sprintf(fmt, std::forward<Args>(args)...);
  log.error(" -- MOODIST FATAL ERROR --\n%s\n", s);
  std::quick_exit(1);
}

} // namespace moodist
