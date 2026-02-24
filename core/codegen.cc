// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "codegen.h"
#include "common.h"
#include "compile_op_kernel.h"
#include "group.h"

namespace moodist {
namespace codegen {

// ---------------------------------------------------------------------------
// Thread-local builder state
// ---------------------------------------------------------------------------

static thread_local Builder* activeBuilder = nullptr;

Builder& builder() {
  CHECK(activeBuilder != nullptr);
  return *activeBuilder;
}

BuilderScope::BuilderScope() {
  prev = activeBuilder;
  activeBuilder = &b;
}

BuilderScope::~BuilderScope() {
  activeBuilder = prev;
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

void Builder::emit(const std::string& line) {
  lines.push_back(std::string(indentLevel * 2, ' ') + line);
}

void Builder::emitBlank() {
  lines.emplace_back();
}

void Builder::indent() {
  ++indentLevel;
}

void Builder::dedent() {
  CHECK(indentLevel > 0);
  --indentLevel;
}

Var Builder::makeVar(const char* type) {
  std::string name = "v" + std::to_string(varCounter++);
  return Var(std::move(name), type);
}

Var Builder::makeVar(const char* type, const char* name) {
  return Var(name, type);
}

std::string Builder::finalize() const {
  std::string result;
  for (auto& line : lines) {
    result += line;
    result += '\n';
  }
  return result;
}

// ---------------------------------------------------------------------------
// Expr helpers
// ---------------------------------------------------------------------------

Expr hex(uintptr_t value) {
  return Expr(fmt::sprintf("0x%lxUL", value));
}

Expr cast(const char* type, const Expr& e) {
  return Expr(std::string("((") + type + ")(" + e.str + "))");
}

Expr as(const char* type, const Expr& addr) {
  return Expr(std::string("((") + type + "*)(" + addr.str + "))");
}

Expr deref(const char* type, const Expr& addr) {
  return Expr(std::string("*((") + type + "*)(" + addr.str + "))");
}

Expr addrof(const char* type, const Expr& e) {
  return Expr(std::string("((") + type + "*)(&(" + e.str + ")))");
}

Expr call(const char* func) {
  return Expr(std::string(func) + "()");
}

Expr call(const char* func, const Expr& a) {
  return Expr(std::string(func) + "(" + a.str + ")");
}

Expr call(const char* func, const Expr& a, const Expr& b) {
  return Expr(std::string(func) + "(" + a.str + ", " + b.str + ")");
}

Expr call(const char* func, const Expr& a, const Expr& b, const Expr& c) {
  return Expr(std::string(func) + "(" + a.str + ", " + b.str + ", " + c.str + ")");
}

Expr call(const char* func, const Expr& a, const Expr& b, const Expr& c, const Expr& d) {
  return Expr(std::string(func) + "(" + a.str + ", " + b.str + ", " + c.str + ", " + d.str + ")");
}

void stmt(const char* func) {
  builder().emit(std::string(func) + "();");
}

void stmt(const char* func, const Expr& a) {
  builder().emit(std::string(func) + "(" + a.str + ");");
}

void stmt(const char* func, const Expr& a, const Expr& b) {
  builder().emit(std::string(func) + "(" + a.str + ", " + b.str + ");");
}

void stmt(const char* func, const Expr& a, const Expr& b, const Expr& c) {
  builder().emit(std::string(func) + "(" + a.str + ", " + b.str + ", " + c.str + ");");
}

Expr ldcv(const Expr& addr) {
  return call("__ldcv", addr);
}
void stwt(const Expr& addr, const Expr& val) {
  stmt("__stwt", addr, val);
}
void threadfence_system() {
  stmt("__threadfence_system");
}
void syncthreads() {
  emit("asm volatile (\"barrier.sync 0;\" :: );");
}

// ---------------------------------------------------------------------------
// Var
// ---------------------------------------------------------------------------

void Var::operator=(const Expr& rhs) {
  builder().emit(str + " = " + rhs.str + ";");
}

void Var::operator+=(const Expr& rhs) {
  builder().emit(str + " += " + rhs.str + ";");
}

void Var::operator-=(const Expr& rhs) {
  builder().emit(str + " -= " + rhs.str + ";");
}

void Var::operator*=(const Expr& rhs) {
  builder().emit(str + " *= " + rhs.str + ";");
}

void Var::operator/=(const Expr& rhs) {
  builder().emit(str + " /= " + rhs.str + ";");
}

void Var::operator%=(const Expr& rhs) {
  builder().emit(str + " %= " + rhs.str + ";");
}

void Var::operator&=(const Expr& rhs) {
  builder().emit(str + " &= " + rhs.str + ";");
}

void Var::operator|=(const Expr& rhs) {
  builder().emit(str + " |= " + rhs.str + ";");
}

void Var::operator^=(const Expr& rhs) {
  builder().emit(str + " ^= " + rhs.str + ";");
}

void Var::operator<<=(const Expr& rhs) {
  builder().emit(str + " <<= " + rhs.str + ";");
}

void Var::operator>>=(const Expr& rhs) {
  builder().emit(str + " >>= " + rhs.str + ";");
}

// ---------------------------------------------------------------------------
// Free functions: decl, emit
// ---------------------------------------------------------------------------

Var decl(const char* type) {
  auto& b = builder();
  Var v = b.makeVar(type);
  b.emit(std::string(type) + " " + v.str + ";");
  return v;
}

Var decl(const char* type, const Expr& init) {
  auto& b = builder();
  Var v = b.makeVar(type);
  b.emit(std::string(type) + " " + v.str + " = " + init.str + ";");
  return v;
}

Var decl(const char* type, const char* name) {
  auto& b = builder();
  Var v = b.makeVar(type, name);
  b.emit(std::string(type) + " " + v.str + ";");
  return v;
}

Var decl(const char* type, const char* name, const Expr& init) {
  auto& b = builder();
  Var v = b.makeVar(type, name);
  b.emit(std::string(type) + " " + v.str + " = " + init.str + ";");
  return v;
}

void emit(const std::string& stmt) {
  builder().emit(stmt);
}

void emitBlank() {
  builder().emitBlank();
}

// ---------------------------------------------------------------------------
// Control flow: ScopeGuard
// ---------------------------------------------------------------------------

ScopeGuard::~ScopeGuard() {
  if (!closed) {
    auto& b = builder();
    b.dedent();
    b.emit("}");
    closed = true;
  }
}

ScopeGuard _If(const Expr& cond) {
  auto& b = builder();
  b.emit("if (" + cond.str + ") {");
  b.indent();
  return ScopeGuard();
}

ScopeGuard _Else() {
  auto& b = builder();
  // Previous ScopeGuard already dedented and emitted "}" — replace it
  b.lines.pop_back();
  b.emit("} else {");
  b.indent();
  return ScopeGuard();
}

ScopeGuard _ElseIf(const Expr& cond) {
  auto& b = builder();
  b.lines.pop_back();
  b.emit("} else if (" + cond.str + ") {");
  b.indent();
  return ScopeGuard();
}

ScopeGuard _While(const Expr& cond) {
  auto& b = builder();
  b.emit("while (" + cond.str + ") {");
  b.indent();
  return ScopeGuard();
}

ScopeGuard _Block() {
  auto& b = builder();
  b.emit("{");
  b.indent();
  return ScopeGuard();
}

// ---------------------------------------------------------------------------
// Control flow: ForRange
// ---------------------------------------------------------------------------

ForRange::ForRange(const Expr& end) : start_(0), end_(end), step_(1), type_("uint32_t") {}

ForRange::ForRange(const Expr& start, const Expr& end) : start_(start), end_(end), step_(1), type_("uint32_t") {}

ForRange::ForRange(const Expr& start, const Expr& end, const Expr& step)
    : start_(start), end_(end), step_(step), type_("uint32_t") {}

ForRange::ForRange(const char* type, const Expr& start, const Expr& end)
    : start_(start), end_(end), step_(1), type_(type) {}

ForRange::ForRange(const char* type, const Expr& start, const Expr& end, const Expr& step)
    : start_(start), end_(end), step_(step), type_(type) {}

ForRange::Iter ForRange::begin() {
  auto& b = builder();
  Var v = b.makeVar(type_);
  b.emit("for (" + std::string(type_) + " " + v.str + " = " + start_.str + "; " + v.str + " < " + end_.str + "; " +
         v.str + " += " + step_.str + ") {");
  b.indent();
  return Iter(std::move(v), false);
}

ForRange::Iter::~Iter() {
  if (ownsClose) {
    auto& b = builder();
    b.dedent();
    b.emit("}");
  }
}

} // namespace codegen

std::string generateCopyKernel(Group* group, size_t gridSize, size_t blockSize, int depth) {
  using namespace codegen;
  BuilderScope scope;

  size_t rank = group->rank;
  const auto& peerIndices = group->peerIndices;
  bool noSync = std::getenv("MOODIST_PROFILE_NOSYNC") && !strcmp(std::getenv("MOODIST_PROFILE_NOSYNC"), "1");

  int BS = (int)blockSize;
  int LB = (int)(blockSize * 16);
  int GS = (int)gridSize;

  // ---- Preamble ----
  emit("using uintptr_t = unsigned long;");
  emit("using uint64_t = unsigned long;");
  emit("using uint32_t = unsigned int;");
  emit("using uint16_t = unsigned short;");
  emit("using uint8_t = unsigned char;");
  emit("using int32_t = int;");
  emit("using int64_t = long;");
  emit("using size_t = unsigned long;");
  emitBlank();
  emit("namespace {");
  emitBlank();
  emit("__device__ void syncthreads() { asm volatile (\"barrier.sync 0;\" :: ); }");
  emitBlank();
  emit("struct CopyDescriptor { uintptr_t src; uintptr_t dst; uint32_t bytes; };");
  emitBlank();
  emit("struct CompileOpCopyParameters {");
  builder().indent();
  emit("uint32_t stepValue;");
  emit("uint32_t concurrencyIndex;");
  emit("uint32_t numDescriptors;");
  emit("uint32_t _pad;");
  emit(fmt::sprintf("CopyDescriptor descriptors[%u];", kMaxCopyDescriptors));
  builder().dedent();
  emit("};");
  emitBlank();
  emit(fmt::sprintf("__device__ uint32_t entryCounter[%zu];", Group::maxConcurrency));
  emit(fmt::sprintf("__device__ uint32_t exitCounter[%zu];", Group::maxConcurrency));
  emitBlank();

  // ---- copy_descriptor device function ----
  // All sizes/counts are uint32_t. Only src/dst pointers are 64-bit.
  emit("__device__ void copy_descriptor(");
  emit("    const uint32_t tid, const uint32_t blockIndex, const uint32_t numBlocks,");
  emit("    uintptr_t src, uintptr_t dst, uint32_t bytes) {");
  builder().indent();

  auto src = expr("src");
  auto dst = expr("dst");
  auto numBlocks = expr("numBlocks");

  emit(fmt::sprintf("const uint32_t tid16 = tid * 16;"));
  emit(fmt::sprintf("const uint32_t loopBytes = %du;", LB));
  auto tid16 = expr("tid16");
  auto loopBytes = expr("loopBytes");
  emitBlank();

  // Head alignment
  {
    Var srcMod = decl(u32, "srcMod", cast(u32, src & 15));
    Var dstMod = decl(u32, "dstMod", cast(u32, dst & 15));
    IF(srcMod == dstMod && srcMod != 0 && expr("bytes") >= 16) {
      Var headBytes = decl(u32, "headBytes", 16 - srcMod);
      IF(headBytes > expr("bytes")) {
        headBytes = expr("bytes");
      }
      for (Var i : ForRange(u32, expr("tid"), headBytes, Expr(BS))) {
        emit(deref(u8, dst + i).str + " = " + deref("const uint8_t", src + i).str + ";");
      }
      emit("src += headBytes;");
      emit("dst += headBytes;");
      emit("bytes -= headBytes;");
    }
  }
  emitBlank();

  Var aligned = decl(Bool, "aligned", ((src | dst) & 15) == 0);
  emitBlank();

  IF(aligned && expr("bytes") >= loopBytes) {
    Var count = decl(u32, "count", expr("bytes") / loopBytes);
    Var i = decl(u32, "i", expr("blockIndex"));
    emitBlank();

    // Declare carry registers
    for (int k = 0; k < depth; k++) {
      decl(uint4, fmt::sprintf("v%d", k).c_str());
    }
    emitBlank();

    // Prime: load first `depth` elements
    for (int k = 0; k < depth; k++) {
      auto vk = expr(fmt::sprintf("v%d", k).c_str());
      IF(i + k * numBlocks < count) {
        emit(vk.str + " = " + ldcv(as(uint4, src + (i + k * numBlocks) * loopBytes + tid16)).str + ";");
      }
    }
    emitBlank();

    // Main loop: store-load pairs
    WHILE(i + (2 * depth - 1) * numBlocks < count) {
      for (int j = 0; j < depth; j++) {
        auto vj = expr(fmt::sprintf("v%d", j).c_str());
        stwt(as(uint4, dst + (i + j * numBlocks) * loopBytes + tid16), vj);
        emit(vj.str + " = " + ldcv(as(uint4, src + (i + (j + depth) * numBlocks) * loopBytes + tid16)).str + ";");
      }
      i += depth * numBlocks;
    }
    emitBlank();

    // Drain
    for (int k = 0; k < depth; k++) {
      auto vk = expr(fmt::sprintf("v%d", k).c_str());
      IF(i + k * numBlocks < count) {
        stwt(as(uint4, dst + (i + k * numBlocks) * loopBytes + tid16), vk);
      }
    }
    i += depth * numBlocks;
    emitBlank();

    // Tail
    WHILE(i < count) {
      Var tv = decl(uint4, "tv", ldcv(as(uint4, src + i * loopBytes + tid16)));
      stwt(as(uint4, dst + i * loopBytes + tid16), tv);
      i += numBlocks;
    }
    emitBlank();

    emit("uint32_t done = count * loopBytes;");
    emit("src += done;");
    emit("dst += done;");
    emit("bytes -= done;");
  }
  emitBlank();

  // Remaining aligned uint4 elements
  IF(aligned) {
    Var remaining16 = decl(u32, "remaining16", expr("bytes") / 16);
    for (Var i : ForRange(u32, expr("tid"), remaining16, Expr(BS))) {
      Var val = decl(uint4, "val", ldcv(as("const uint4", src + i * 16)));
      stwt(as(uint4, dst + i * 16), val);
    }
    emit("uint32_t done16 = remaining16 * 16;");
    emit("src += done16;");
    emit("dst += done16;");
    emit("bytes -= done16;");
  }
  emitBlank();

  // Tail: byte copy
  for (Var i : ForRange(u32, expr("tid"), expr("bytes"), Expr(BS))) {
    emit(deref(u8, dst + i).str + " = " + deref("const uint8_t", src + i).str + ";");
  }

  builder().dedent();
  emit("}");
  emitBlank();
  emit("} // namespace");
  emitBlank();

  // ---- Main kernel ----
  emit(fmt::sprintf("extern \"C\" __global__ void __launch_bounds__(%d, 1)", BS));
  emit("compile_op_copy(CompileOpCopyParameters params) {");
  builder().indent();

  auto stepValue = expr("stepValue");
  auto concurrencyIndex = expr("concurrencyIndex");

  emit("const uint32_t stepValue = params.stepValue;");
  emit("const uint32_t concurrencyIndex = params.concurrencyIndex;");
  emitBlank();

  // ---- Entry barrier ----
  IF(expr("threadIdx.x") == 0) {
    IF(call("atomicInc", expr("&entryCounter[concurrencyIndex]"), Expr(GS - 1)) == 0) {
      if (!noSync) {
        for (size_t i : peerIndices) {
          auto& arr = group->peerCudaStepValue[i];
          Expr addr = hex(arr.base) + hex(arr.itembytes) * concurrencyIndex + hex(sizeof(uint32_t) * rank);
          emit(deref("volatile uint32_t", addr).str + " = stepValue;");
        }
      }
      threadfence_system();
    }
    if (!noSync) {
      for (size_t i : peerIndices) {
        Expr addr = hex(group->cudaStepValue.buffer.cudaPointer) +
                    hex(group->cudaStepValue.itembytes) * concurrencyIndex + hex(sizeof(uint32_t) * group->ipcRanks[i]);
        WHILE(deref("volatile uint32_t", addr) < stepValue) {}
      }
    }
  }
  syncthreads();
  emitBlank();

  // ---- Descriptor loop (staggered for peer diversity) ----
  Var tid = decl(u32, "tid", expr("threadIdx.x"));
  auto params = expr("params");

  for (Var dd : ForRange(u32, 0, params.dot("numDescriptors"))) {
    Var d = decl(u32, "d", (expr("blockIdx.x") + dd) % params.dot("numDescriptors"));
    Expr desc = params.dot("descriptors")[d];
    emit(fmt::sprintf("copy_descriptor(tid, blockIdx.x, %d, ", GS) + desc.dot("src").str + ", " + desc.dot("dst").str +
         ", " + desc.dot("bytes").str + ");");
  }
  emitBlank();

  // ---- Exit barrier ----
  threadfence_system();
  syncthreads();
  IF(expr("threadIdx.x") == 0 &&
      call("atomicInc", expr("&exitCounter[concurrencyIndex]"), Expr(GS - 1)) == Expr(GS - 1)) {
    if (!noSync) {
      for (size_t i : peerIndices) {
        auto& arr = group->peerCudaCopyDone[i];
        Expr addr =
            hex(arr.base) + hex(arr.itembytes) * concurrencyIndex + hex(sizeof(uint32_t) * group->peerMyRemoteIndex[i]);
        emit(deref("volatile uint32_t", addr).str + " = stepValue;");
      }
      for (size_t i : peerIndices) {
        Expr addr = hex(group->cudaCopyDone.buffer.cudaPointer) +
                    hex(group->cudaCopyDone.itembytes) * concurrencyIndex + hex(sizeof(uint32_t) * i);
        WHILE(deref("volatile uint32_t", addr) < stepValue) {}
      }
    }
  }

  builder().dedent();
  emit("}");

  return scope.finalize();
}

} // namespace moodist
