// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moodist {
namespace codegen {

struct Builder;

// Access the active builder (must be within a BuilderScope).
Builder& builder();

// ---------------------------------------------------------------------------
// Expr: a string-based expression in generated code.
//
// Operator overloading builds expression strings:
//   Expr a("x"), b("y");
//   Expr c = a + b * 2;  // c.str == "(x + (y * 2))"
//
// Implicit conversion from integer types allows mixing C++ constants
// with codegen expressions:
//   Var i = decl("size_t");
//   Expr addr = src + i * 16;  // "src" + "i" * 16
// ---------------------------------------------------------------------------

struct Expr {
  std::string str;

  Expr() = default;
  explicit Expr(std::string s) : str(std::move(s)) {}

  // Implicit from integer types for natural arithmetic with C++ constants.
  Expr(int v) : str(std::to_string(v)) {}
  Expr(unsigned v) : str(std::to_string(v) + "u") {}
  Expr(long v) : str(std::to_string(v)) {}
  Expr(unsigned long v) : str(std::to_string(v) + "u") {}

  // Arithmetic
  friend Expr operator+(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " + " + b.str + ")");
  }
  friend Expr operator-(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " - " + b.str + ")");
  }
  friend Expr operator*(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " * " + b.str + ")");
  }
  friend Expr operator/(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " / " + b.str + ")");
  }
  friend Expr operator%(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " % " + b.str + ")");
  }

  // Comparison
  friend Expr operator<(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " < " + b.str + ")");
  }
  friend Expr operator>(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " > " + b.str + ")");
  }
  friend Expr operator<=(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " <= " + b.str + ")");
  }
  friend Expr operator>=(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " >= " + b.str + ")");
  }
  friend Expr operator==(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " == " + b.str + ")");
  }
  friend Expr operator!=(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " != " + b.str + ")");
  }

  // Logical
  friend Expr operator&&(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " && " + b.str + ")");
  }
  friend Expr operator||(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " || " + b.str + ")");
  }
  Expr operator!() const {
    return Expr("(!" + str + ")");
  }

  // Bitwise
  friend Expr operator&(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " & " + b.str + ")");
  }
  friend Expr operator|(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " | " + b.str + ")");
  }
  friend Expr operator^(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " ^ " + b.str + ")");
  }
  Expr operator~() const {
    return Expr("(~" + str + ")");
  }

  // Unary minus
  Expr operator-() const {
    return Expr("(-" + str + ")");
  }

  // Shift
  friend Expr operator<<(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " << " + b.str + ")");
  }
  friend Expr operator>>(const Expr& a, const Expr& b) {
    return Expr("(" + a.str + " >> " + b.str + ")");
  }

  // Member/index access
  Expr operator[](const Expr& index) const {
    return Expr(str + "[" + index.str + "]");
  }
  Expr dot(const char* member) const {
    return Expr(str + "." + member);
  }
  Expr arrow(const char* member) const {
    return Expr(str + "->" + member);
  }

  // Ternary: cond.select(a, b) => "(cond ? a : b)"
  Expr select(const Expr& a, const Expr& b) const {
    return Expr("(" + str + " ? " + a.str + " : " + b.str + ")");
  }
};

// A named expression (e.g., a builtin like threadIdx.x).
inline Expr expr(const char* s) {
  return Expr(std::string(s));
}

// ---------------------------------------------------------------------------
// Common type names
// ---------------------------------------------------------------------------

constexpr const char* u8 = "uint8_t";
constexpr const char* u16 = "uint16_t";
constexpr const char* u32 = "uint32_t";
constexpr const char* u64 = "uint64_t";
constexpr const char* i32 = "int32_t";
constexpr const char* i64 = "int64_t";
constexpr const char* uptr = "uintptr_t";
constexpr const char* sz = "size_t";
constexpr const char* Bool = "bool";
constexpr const char* uint4 = "uint4";

// Hex literal (for addresses baked into generated code).
Expr hex(uintptr_t value);

// Type cast: "(type)(expr)"
Expr cast(const char* type, const Expr& e);

// Pointer cast: "(type*)(expr)"
Expr as(const char* type, const Expr& addr);

// Pointer cast + dereference: "*(type*)(expr)"
Expr deref(const char* type, const Expr& addr);

// Address-of with cast: "(type*)(&expr)" — rarely needed.
Expr addrof(const char* type, const Expr& e);

// Function calls (1-4 args)
Expr call(const char* func);
Expr call(const char* func, const Expr& a);
Expr call(const char* func, const Expr& a, const Expr& b);
Expr call(const char* func, const Expr& a, const Expr& b, const Expr& c);
Expr call(const char* func, const Expr& a, const Expr& b, const Expr& c, const Expr& d);

// Emit a function call as a statement (void return).
void stmt(const char* func);
void stmt(const char* func, const Expr& a);
void stmt(const char* func, const Expr& a, const Expr& b);
void stmt(const char* func, const Expr& a, const Expr& b, const Expr& c);

// CUDA intrinsics
Expr ldcv(const Expr& addr);
void stwt(const Expr& addr, const Expr& val);
void threadfence_system();
void syncthreads();

// ---------------------------------------------------------------------------
// Var: a named variable in generated code.
//
// All assignment operators emit codegen statements (not C++ assignments).
//   Var x = decl("uint32_t");        // emits "uint32_t v0;"
//   x = 5;                           // emits "v0 = 5;"
//   x += i * 16;                     // emits "v0 += (v1 * 16);"
//
// Copy constructor copies the name (both refer to the same generated var).
//   Var y = x;  // y.str == x.str, no codegen emitted
// ---------------------------------------------------------------------------

struct Var : Expr {
  std::string typeName;

  Var() = default;
  Var(const Var&) = default;
  Var(Var&&) noexcept = default;

  // All operator= forms emit codegen assignment.
  void operator=(const Expr& rhs);
  void operator=(const Var& rhs) {
    operator=(static_cast<const Expr&>(rhs));
  }
  void operator=(Var&& rhs) {
    operator=(static_cast<const Expr&>(rhs));
  }

  void operator+=(const Expr& rhs);
  void operator-=(const Expr& rhs);
  void operator*=(const Expr& rhs);
  void operator/=(const Expr& rhs);
  void operator%=(const Expr& rhs);
  void operator&=(const Expr& rhs);
  void operator|=(const Expr& rhs);
  void operator^=(const Expr& rhs);
  void operator<<=(const Expr& rhs);
  void operator>>=(const Expr& rhs);

private:
  friend struct Builder;
  Var(std::string name, std::string type) : Expr(std::move(name)), typeName(std::move(type)) {}
};

// Declare a variable. Emits the declaration to the active builder.
// Auto-generates a unique name (v0, v1, ...) unless a name is given.
Var decl(const char* type);
Var decl(const char* type, const Expr& init);
Var decl(const char* type, const char* name);
Var decl(const char* type, const char* name, const Expr& init);

// Emit a raw statement or blank line.
void emit(const std::string& stmt);
void emitBlank();

// ---------------------------------------------------------------------------
// Builder: manages code generation state.
//
// Maintains a flat list of lines. Braces provide structure; finalize()
// applies auto-indentation based on { and }.
// ---------------------------------------------------------------------------

struct Builder {
  std::vector<std::string> lines;
  int varCounter = 0;
  int indentLevel = 0;

  void emit(const std::string& line);
  void emitBlank();
  void indent();
  void dedent();
  Var makeVar(const char* type);
  Var makeVar(const char* type, const char* name);

  std::string finalize() const;
};

// RAII scope to activate a builder (thread-local).
struct BuilderScope {
  Builder b;
  Builder* prev;

  BuilderScope();
  ~BuilderScope();

  std::string finalize() const {
    return b.finalize();
  }
};

// ---------------------------------------------------------------------------
// Control flow: IF / ELSE / ELIF / WHILE
//
// Uses the range-based for loop trick: the macro expands to a for loop
// that iterates exactly once. RAII ensures matching braces.
//
//   IF (aligned && bytes >= loopBytes) {
//       // body
//   } ELSE {
//       // body
//   }
//
//   WHILE (i + depth * numBlocks < count) {
//       // body
//   }
// ---------------------------------------------------------------------------

struct ScopeGuard {
  bool closed = false;

  ScopeGuard() = default;
  ScopeGuard(ScopeGuard&& o) noexcept : closed(o.closed) {
    o.closed = true;
  }
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ~ScopeGuard();

  struct Iter {
    bool done;
    int operator*() const {
      return 0;
    }
    Iter& operator++() {
      done = true;
      return *this;
    }
    bool operator!=(const Iter& o) const {
      return done != o.done;
    }
  };

  Iter begin() {
    return {false};
  }
  Iter end() {
    return {true};
  }
};

ScopeGuard _If(const Expr& cond);
ScopeGuard _Else();
ScopeGuard _ElseIf(const Expr& cond);
ScopeGuard _While(const Expr& cond);
ScopeGuard _Block(); // anonymous { }

#define IF(cond) for ([[maybe_unused]] auto _cg_scope_ : ::moodist::codegen::_If(cond))
#define ELSE for ([[maybe_unused]] auto _cg_scope_ : ::moodist::codegen::_Else())
#define ELIF(cond) for ([[maybe_unused]] auto _cg_scope_ : ::moodist::codegen::_ElseIf(cond))
#define WHILE(cond) for ([[maybe_unused]] auto _cg_scope_ : ::moodist::codegen::_While(cond))

// ---------------------------------------------------------------------------
// ForRange: range-based for loop in generated code.
//
//   for (Var i : ForRange(0, count)) {
//       // body using i
//   }
//
//   for (Var i : ForRange("size_t", tid, count, numBlocks)) {
//       // strided loop
//   }
// ---------------------------------------------------------------------------

struct ForRange {
  Expr start_, end_, step_;
  const char* type_;

  ForRange(const Expr& end);
  ForRange(const Expr& start, const Expr& end);
  ForRange(const Expr& start, const Expr& end, const Expr& step);
  ForRange(const char* type, const Expr& start, const Expr& end);
  ForRange(const char* type, const Expr& start, const Expr& end, const Expr& step);

  struct Iter {
    Var var;
    bool done;
    bool ownsClose;

    Iter(Var v, bool d) : var(std::move(v)), done(d), ownsClose(!d) {}
    Iter(Iter&& o) noexcept : var(std::move(o.var)), done(o.done), ownsClose(o.ownsClose) {
      o.ownsClose = false;
    }
    Iter(const Iter&) = delete;
    ~Iter();

    Var& operator*() {
      return var;
    }
    Iter& operator++() {
      done = true;
      return *this;
    }
    bool operator!=(const Iter& o) const {
      return done != o.done;
    }
  };

  Iter begin();
  Iter end() {
    return Iter(Var(), true);
  }
};

} // namespace codegen

struct Group; // forward declare

// ---------------------------------------------------------------------------
// Kernel generation building blocks.
//
// These functions append to the active codegen::Builder. They are composable:
// call emitPreamble + emitCopyFunction + emitMainKernel to build a complete
// kernel, or mix and match for tuning kernels.
// ---------------------------------------------------------------------------

// Emit type aliases, struct definitions, device counters, syncthreads helper.
// Must be called first — provides the types used by everything else.
void emitPreamble();

// Emit a __device__ copy function with pipelined uint4 loads/stores.
// functionName: name of the generated function (e.g. "copy_descriptor")
// blockSize: threads per block (baked in as loopBytes constant)
// depth: pipeline depth (number of in-flight uint4 loads)
void emitCopyFunction(const char* functionName, size_t blockSize, int depth);

// Emit entry barrier code (stepValue writes + waits).
// Must be called inside an `if (threadIdx.x == 0)` block.
void emitEntryBarrier(Group* group, size_t gridSize);

// Emit exit barrier code (copyDone writes + waits).
// Must be called inside an `if (threadIdx.x == 0 && lastBlock)` block.
void emitExitBarrier(Group* group);

// Emit the complete main kernel with barriers and descriptor loop.
void emitMainKernel(Group* group, size_t gridSize, size_t blockSize, const char* kernelName = "compile_op_copy",
    const char* copyFnName = "copy_descriptor");

// Generate a complete copy kernel source string. Convenience wrapper
// that calls emitPreamble + emitCopyFunction + emitMainKernel.
std::string generateCopyKernel(Group* group, size_t gridSize, size_t blockSize, int depth);

} // namespace moodist
