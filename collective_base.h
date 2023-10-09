#pragma once

#include "group.h"

#include "fmt/printf.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

namespace moodist {

struct Group;

struct CollectiveBase {
  Group* group;

  CollectiveBase(Group* group) : group(group){};
  virtual ~CollectiveBase() {}

  const size_t rank = group->rank;
  const size_t size = group->size;

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
        r += fmt::sprintf(" * % 4d *  ", line);
      } else {
        r += fmt::sprintf("/* % 4d */ ", line);
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

  std::string emitCopy(
      std::vector<std::string> sources, std::vector<std::vector<std::string>> destinations, std::string bytes,
      size_t gridSize, size_t blockSize, std::string threadIndex, std::string blockIndex,
      std::string dataChunkFinishedCallback) {
    TORCH_CHECK(sources.size() == destinations.size());
    size_t stride = gridSize * blockSize;
    auto genwrites = [&](std::string& writes, std::string type, int typesize,
                         const std::vector<std::string>& temporaries) {
      for (size_t i = 0; i != destinations.size(); ++i) {
        for (auto& v : destinations[i]) {
          writes += fmt::sprintf("write_peer((%s*)(void*)%s, %s);\n", type, v, temporaries[i]);
          writes += fmt::sprintf("%s = %s + %u;\n", v, v, typesize * stride);
        }
      }
    };
    auto readwrite = [&](std::string& reads, std::string& writes, std::string type, int typesize) {
      std::vector<std::string> temporaries;
      for (auto& v : sources) {
        std::string temp = getVar("temporary");
        temporaries.push_back(temp);
        reads += fmt::sprintf("%s %s = read_peer((%s*)(void*)%s);\n", type, temp, type, v);
        reads += fmt::sprintf("%s = %s + %u;\n", v, v, typesize * stride);
      }
      genwrites(writes, type, typesize, temporaries);
    };
    auto addOffset = [&](int typesize) {
      std::string s;
      for (auto& v : sources) {
        s += fmt::sprintf("%s += %d * %%offset;\n", v, typesize);
      }
      for (auto& list : destinations) {
        for (auto& v : list) {
          s += fmt::sprintf("%s += %d * %%offset;\n", v, typesize);
        }
      }
      return s;
    };
    std::function<std::string(std::string, int, int, int)> genCopy = [&](std::string type, int typesize,
                                                                         int unrollOuter, int unrollInner) {
      std::string s = R"(
$pre
  while (%bytes >= $loopbytes) {
    size_t %loopcount = %bytes / $loopbytes;
    $datachunkpre
    $preloop
#pragma unroll 1
    for (size_t %i = 0; %i != %loopcount - 1; ++%i) {
      __syncwarp();
      $loop
    }
    $postloop
    %bytes -= $loopbytes * %loopcount;
    $datachunkpost
  }
$post
)";

      std::string datachunkpre;
      std::string datachunkpost;
      if (!dataChunkFinishedCallback.empty()) {
        datachunkpre = R"(
          %loopcount = ((%bytes + $dataChunks - 1) / $dataChunks + $loopbytes - 1) / $loopbytes;
          )";
        datachunkpost = R"(
          if (%bytes <= %nextChunkBytes) {
            %nextChunkBytes -= %chunkBytes;
            ++%currentChunkIndex;
            uint32_t currentChunkIndex = %currentChunkIndex;
            $dataChunkFinishedCallback
          })";
      }
      s = replace(s, "$datachunkpre", datachunkpre, "$datachunkpost", datachunkpost);
      std::string pre = addOffset(typesize);
      std::string preloop;
      std::string loop;
      std::string postloop;
      for (int i = 0; i != unrollOuter; ++i) {
        for (size_t n = 0; n != destinations.size(); ++n) {
          for (int i = 0; i != unrollInner; ++i) {
            auto& src = sources[n];
            std::string temp = getVar("temporary");
            std::string read = replace(
                R"($temp = read_peer_large_$typesize(($type*)(void*)$src); $src += $typesize * $stride;
                )",
                "$type", type, "$temp", temp, "$src", src, "$typesize", typesize, "$stride", stride);
            preloop += type + " " + read;
            for (auto& dst : destinations[n]) {
              std::string write = replace(
                  R"(write_peer(($type*)(void*)$dst, $temp); $dst += $typesize * $stride;
                  )",
                  "$type", type, "$temp", temp, "$dst", dst, "$typesize", typesize, "$stride", stride);
              postloop += write;
              loop += write;
            }
            loop += read;
          }
        }
      }
      std::string post;
      for (auto& v : sources) {
        post += fmt::sprintf("%s -= %d * %%offset;\n", v, typesize);
      }
      for (auto& list : destinations) {
        for (auto& v : list) {
          post += fmt::sprintf("%s -= %d * %%offset;\n", v, typesize);
        }
      }
      size_t loopbytes = typesize * unrollOuter * unrollInner * stride;
      return replace(
          s, "$loop", loop, "$pre", pre, "$post", post, "$loopbytes", loopbytes, "$preloop", preloop, "$postloop",
          postloop);
    };
    std::string s;
    s += fmt::sprintf(
        "/* copy  bytes: %s  gridSize: %d  blockSize: %d  threadIndex: %s  blockIndex: %s */\n", bytes, gridSize,
        blockSize, threadIndex, blockIndex);
    s += R"(
  size_t %bytes = $bytes;
  uint32_t %threadIndex = $threadIndex;
  uint32_t %blockIndex = $blockIndex;
  uint32_t %offset = $blockSize * %blockIndex + %threadIndex;
  $datachunkpre
  $defs
  $copy1
  $copy2
  $copy3
  if (%offset < (uint32_t)%bytes) {
    $copylastbyte
  }
  $datachunkpost
)";
    std::string datachunkpre;
    std::string datachunkpost;
    if (!dataChunkFinishedCallback.empty()) {
      datachunkpre = R"(
        uint32_t %currentChunkIndex = 0;
        size_t %chunkBytes = %bytes / $dataChunks;
        size_t %nextChunkBytes = %bytes - %chunkBytes;)";
      datachunkpost = R"(
        if (%currentChunkIndex > $dataChunks) {
          printf("currentChunkIndex is %%d!\n", %currentChunkIndex);
          __trap();
        }
        if (%currentChunkIndex != $dataChunks) {
          %currentChunkIndex = $dataChunks;
          uint32_t currentChunkIndex = %currentChunkIndex;
          $dataChunkFinishedCallback
        })";
    }
    s = replace(s, "$datachunkpre", datachunkpre, "$datachunkpost", datachunkpost);
    std::string defs;
    for (auto& v : sources) {
      auto var = getVar(v);
      defs += fmt::sprintf("uintptr_t %s = (uintptr_t)(void*)(%s);\n", var, v);
      v = var;
    }
    for (auto& list : destinations) {
      for (auto& v : list) {
        auto var = getVar(v);
        defs += fmt::sprintf("uintptr_t %s = (uintptr_t)(void*)(%s);\n", var, v);
        // defs += replace(R"(if (threadIdx.x == 0) printf("rank $rank got dst %%#llx from $v\n", (uintptr_t)$var);
        // )", "$var", var, "$v", v);
        v = var;
      }
    }
    // 16 uint4 uses 64 registers. we could go higher to hide more latency,
    // but it seems to not improve performance
    std::string copy1 = genCopy("uint4", 16, 1, 4);
    std::string copy2 = genCopy("unsigned int", 4, 1, 1);
    std::string copy3 = genCopy("unsigned char", 1, 1, 1);
    std::string copylastbyte = addOffset(1);
    readwrite(copylastbyte, copylastbyte, "unsigned char", 1);
    s = replace(
        s, "$bytes", bytes, "$defs", defs, "$copy1", copy1, "$copy2", copy2, "$copy3", copy3, "$copylastbyte",
        copylastbyte);
    s = replace(s, "$dataChunkFinishedCallback", dataChunkFinishedCallback);
    s = replace(s, "$threadIndex", threadIndex, "$blockIndex", blockIndex, "$copyGridSize", gridSize);
    s = makeVars(s);
    return s;
  }

  std::string emitCopyGroupedByBlock(
      std::vector<std::string> sources, std::vector<std::vector<std::string>> destinations, std::string bytes,
      size_t gridSize, size_t blockSize, std::vector<std::string> dataChunkFinishedCallbacks) {
    TORCH_CHECK(sources.size() == destinations.size());
    TORCH_CHECK(dataChunkFinishedCallbacks.size() == sources.size());
    std::string s;
    s += "/* copy group by block */\n";
    s += R"(
  uint32_t %threadIndex = threadIdx.x;
  uint32_t %blockIndex = blockIdx.x;
  
  uint32_t %index = %blockIndex %% $numSources;
$copies
)";
    std::string copies;
    size_t remainingGridSize = gridSize;
    std::vector<size_t> copyGridSizes;
    copyGridSizes.resize(sources.size());
    for (size_t i = 0; i != gridSize; ++i) {
      ++copyGridSizes.at(i % sources.size());
    }
    for (size_t i = 0; i != sources.size(); ++i) {
      size_t div = (sources.size() - i);
      size_t currentGridSize = (remainingGridSize + div - 1) / div;
      TORCH_CHECK(currentGridSize == copyGridSizes.at(i));
      remainingGridSize -= currentGridSize;
      std::string copy = emitCopy(
          {sources[i]}, {destinations[i]}, bytes, currentGridSize, blockSize, "$copyThreadIndex", "$copyBlockIndex",
          dataChunkFinishedCallbacks[i]);
      copies += replace(
          R"(
        if (%index == $i) {
          uint32_t %copyBlockIndex = %blockIndex / $numSources;
          uint32_t %copyThreadIndex = %threadIndex;
          $copy
        }
      )",
          "$i", i, "$copy", copy);
    }
    s = replace(s, "$copies", copies);
    s = replace(
        s, "$numSources", sources.size(), "$copyThreadIndex", "%copyThreadIndex", "$copyBlockIndex", "%copyBlockIndex");
    s = makeVars(s);
    return s;
  }

  std::string cast(std::string type, uintptr_t address) {
    return fmt::sprintf("((%s)%#x)", type, address);
  }

  size_t getPeerIndex(size_t rank) {
    return group->getPeerIndex(rank);
  }
};

} // namespace moodist
