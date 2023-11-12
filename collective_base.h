#pragma once

#include "common.h"
#include "group.h"

#include "fmt/printf.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

namespace moodist {

struct ProxyInfo {
  size_t source;
  size_t destination;
  size_t destinationPeerIndex;
};

struct ProxyDestinationInfo {
  size_t source;
  size_t proxy;
  size_t proxyPeerIndex;
};

struct Group;

struct CollectiveBase {
  Group* group;

  CollectiveBase(Group* group) : group(group){};
  virtual ~CollectiveBase() {}

  const size_t rank = group->rank;
  const size_t size = group->size;

  std::vector<std::string> supportedTypes = {"float", "double", "int32_t", "int64_t"};
  std::vector<size_t> supportedTypeSizes = {4, 8, 4, 8};
  std::vector<std::string> supportedReductions = {"rsum", "rmin", "rmax", "ravg"};

  static std::string replace(std::string str, std::vector<std::pair<std::string, std::string>> vars) {
    std::sort(vars.begin(), vars.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
    size_t pos = 0;
    while (pos != str.size()) {
      std::string key;
      std::string value;
      size_t index = std::string::npos;
      for (auto& v : vars) {
        auto i = str.find(v.first, pos);
        if (i == std::string::npos) {
          continue;
        }
        if (i < index) {
          key = v.first;
          value = v.second;
          index = i;
        }
      }
      if (index == std::string::npos) {
        break;
      }
      str.replace(index, key.size(), value);
      pos = index + value.size();
    }
    return str;
  }
  template<typename Vars>
  static void extractvars(Vars& vars) {}
  template<typename Vars, typename Key, typename Value, typename... Args>
  static void extractvars(Vars& vars, Key key, Value value, Args... args) {
    if constexpr (std::is_integral_v<Value>) {
      if (value > 65536) {
        vars.emplace_back(key, fmt::sprintf("%#x", value));
      } else {
        vars.emplace_back(key, std::to_string(value));
      }
    } else {
      vars.emplace_back(key, value);
    }
    extractvars(vars, args...);
  }
  template<typename... Args>
  static std::string replace(std::string str, Args... args) {
    static_assert(sizeof...(Args) % 2 == 0);
    std::vector<std::pair<std::string, std::string>> vars;
    extractvars(vars, args...);
    return replace(str, vars);
  }

  std::atomic_int varCounter = 0;
  std::string getVar(std::string name) {
    for (auto& v : name) {
      if (!identifierChar(v) && (v < '0' || v > '9')) {
        v = '_';
      }
    }
    return fmt::sprintf("v%d_%s", varCounter++, name);
  }

  bool identifierChar(char c) {
    if (c >= 'A' && c <= 'Z') {
      return true;
    }
    if (c >= 'a' && c <= 'z') {
      return true;
    }
    if (c == '_') {
      return true;
    }
    return false;
  }

  std::string makeVars(std::string input) {
    const char* c = input.c_str();
    std::vector<std::pair<std::string, std::string>> replacements;
    while (*c) {
      if (*c == '%') {
        ++c;
        if (*c == '%') {
          ++c;
        } else {
          const char* p = c;
          while (*c && identifierChar(*c)) {
            ++c;
          }
          std::string name(p, c - p);
          if (name != "" && name != "d" && name != "s" && name != "u" && name != "ld" && name != "lu" && name != "lx" &&
              name != "p") {
            replacements.emplace_back('%' + name, getVar(name));
          }
        }
      } else {
        ++c;
      }
    }
    return replace(input, replacements);
  }

  std::string indent(const std::string& s, int times = 1) {
    std::string r;
    const char* c = s.c_str();
    for (int line = 1; *c; ++line) {
      r += fmt::sprintf("  ", line);
      const char* p = c;
      while (*c && *c != '\n') {
        ++c;
      }
      if (*c) {
        ++c;
      }
      r.append(p, c - p);
    }
    if (times > 1) {
      return indent(r, times - 1);
    }
    return r;
  }

  std::string autoindent(const std::string& s) {
    std::string r;
    const char* c = s.c_str();
    int level = 0;
    for (int line = 1; *c; ++line) {
      while (*c == ' ') {
        ++c;
      }
      const char* p = c;
      if (*c == '}') {
        --level;
        ++c;
      }
      for (int i = 0; i < level; ++i) {
        r += "  ";
      }
      if (*c == '/' && c[1] == '/') {
        while (*c && *c != '\n') {
          ++c;
        }
      } else {
        while (*c && *c != '\n') {
          if (*c == '{') {
            ++level;
          } else if (*c == '}') {
            --level;
          }
          ++c;
        }
      }
      if (*c) {
        ++c;
      }
      r.append(p, c - p);
    }
    return r;
  }

  std::string addLineCountComments(const std::string& s) {
    std::string r;
    const char* c = s.c_str();
    bool isInComment = false;
    for (int line = 1; *c; ++line) {
      if (isInComment) {
        r += fmt::sprintf(" * % 5d *  ", line);
      } else {
        r += fmt::sprintf("/* % 5d */ ", line);
      }
      const char* p = c;
      while (*c && *c != '\n') {
        if (c[0] == '/' && c[1] == '*') {
          isInComment = true;
        } else if (c[0] == '*' && c[1] == '/') {
          isInComment = false;
        }
        ++c;
      }
      if (*c) {
        ++c;
      }
      r.append(p, c - p);
    }
    return r;
  }

  std::string intrinsics(const std::string& s) {
    std::string r;
    const char* c = s.c_str();
    const char* p = c;
    int n = 0;
    while (*c) {
      if (identifierChar(*c)) {
        const char* s = c;
        ++c;
        while (identifierChar(*c)) {
          ++c;
        }
        if (*c == '(') {
          std::string_view name(s, c - s);
          ++c;
          const char* pargs = c;
          while (*c && *c != ')') {
            ++c;
          }
          std::string_view args(pargs, c - pargs);
          if (*c) {
            ++c;
          }
          std::string_view original(s, c - s);
          if (name == "syncthreads" && args == "") {
            r.append(p, s - p);
            r += replace(
                R"(/* $original */ asm volatile ("barrier.sync $n;" :: ); )", "$original", original, "$n", ++n % 16,
                "$rank", rank);
            p = c;
          } else if (name == "barriernumber" && args == "") {
            r.append(p, s - p);
            r += replace("(/* $original */ $n", "$original", original, "$n", ++n % 16);
            p = c;
          }
        }
        continue;
      }
      if (c[0] == '/' && c[1] == '*') {
        while (*c) {
          if (c[0] == '*' && c[1] == '/') {
            ++c;
            break;
          }
          ++c;
        }
      }
      ++c;
    }
    r.append(p, c - p);
    return r;
  }

  std::string cast(std::string type, uintptr_t address) {
    return fmt::sprintf("((%s)%#x)", type, address);
  }
  std::string cast(std::string type, std::string address) {
    return replace("(($type)($address))", "$type", type, "$address", address);
  }

  std::string concurrencyIndex(uintptr_t base, size_t itembytes, size_t offset = 0) {
    return replace(
        "($base + $itembytes * concurrencyIndex + $offset)", "$base", base, "$itembytes", itembytes, "$offset", offset);
  }
  std::string concurrencyIndex(const AllocatedArray& arr, size_t offset = 0) {
    return concurrencyIndex(arr.buffer.cudaPointer, arr.itembytes, offset);
  }
  std::string concurrencyIndex(const PeerArrayRef& arr, size_t offset = 0) {
    return concurrencyIndex(arr.base, arr.itembytes, offset);
  }

  size_t getPeerIndex(size_t rank) {
    return group->getPeerIndex(rank);
  }
};

} // namespace moodist
