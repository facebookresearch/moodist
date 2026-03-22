// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "common.h"

#include <array>
#include <bit>
#include <memory>
#include <string>
#include <vector>

namespace moodist {
namespace ptx {

// Register type classes — determines prefix and .reg declaration type.
enum class RegType { Pred, B16, B32, B64 };

// A typed PTX register. Stores its full name for easy use in instruction strings.
struct Reg {
  RegType type;
  int id;
  std::string name; // "%p0", "%rs0", "%r0", "%rd0"
};

// A basic block: labeled sequence of instructions.
struct Block {
  std::string label;
  std::vector<std::string> insts;

  void emit(const std::string& inst);
};

// A PTX function/kernel with registers, parameters, and basic blocks.
struct Function {
  std::string name;
  int maxThreads = 0; // for .maxntid (0 = omit)

  // Parameters — stored as declaration strings like ".param .u64 name_param_0"
  std::vector<std::string> paramDecls;
  void addParam(const char* type);         // e.g. ".u64", ".u32"
  void addParamBytes(int align, int size); // .param .align A .b8 name[SIZE]
  // Returns the parameter name for use in ld.param, e.g. "name_param_0"
  std::string param(int index) const;

  // Shared memory declarations — emitted after register declarations
  std::vector<std::string> sharedDecls;
  // Declare a shared memory array. Returns the symbol name.
  std::string addShared(int align, int sizeBytes, const char* suffix = nullptr);

  // Register allocation — one counter per type class
  int regCounts[4] = {};
  Reg reg(RegType type);

  // DSL register allocation state
  // std::vector<int> freeRegs[4];
  int regHighWater[4] = {};

  // Label/branch target counters
  int labelCounter = 0;
  int brxCounter = 0;

  // Basic blocks — owned by the function, serialized in order
  std::vector<std::unique_ptr<Block>> blocks;
  Block* newBlock(const char* label);

  // Serialize the function body to a PTX string (entry point + body)
  std::string finalize() const;
};

// Top-level PTX module containing globals and functions.
struct Module {
  std::string version = "8.8";
  std::string target;
  int addressSize = 64;

  std::vector<std::string> globals;
  std::vector<std::unique_ptr<Function>> functions;
  std::vector<std::string> topLevelComments;

  Function* newFunction(const char* name);

  // Declare a global variable. Returns the symbol name.
  // Example: addGlobal(".u64", 256, "counters") → ".global .u64 counters[256]"
  std::string addGlobal(const char* type, int count, const char* name);

  // Declare an initialized constant memory array. Returns the symbol name.
  // Data is embedded directly in the PTX as a byte initializer list.
  // align: byte alignment (must be power of 2)
  // data/size: raw bytes to embed
  // name: symbol name
  std::string addConst(int align, const void* data, size_t size, const char* name);

  // Serialize the complete module to a PTX string
  std::string finalize() const;
};

// ---------------------------------------------------------------------------
// TLS emit targets — set these before emitting instructions
// ---------------------------------------------------------------------------

void setModule(Module* m);
void setFunction(Function* f);
Function* getFunction();
void setBlock(Block* b);

// Register allocation via currentFunction
Reg reg(RegType type);

// ---------------------------------------------------------------------------
// DSL layer — typed values, RAII registers, operators, control flow
// ---------------------------------------------------------------------------

// Typed value — determines instruction selection (signedness + width).
enum class ValType { Pred, U16, S16, U32, S32, U64, S64 };

// Map ValType to RegType for register allocation.
RegType regTypeFor(ValType type);

struct Value {
  Reg reg; // only valid when kind == Register
  ValType type;
  bool valid = false;

  enum Kind { None, Register, Immediate };
  Kind kind = None;
  std::string immStr; // string representation for immediates
  std::optional<int64_t> immValue;

  // --- Constructors ---
  Value() = default;
  explicit Value(ValType type);       // allocate register (existing Val behavior)
  Value(ValType type, std::string s); // immediate, untyped (e.g. "%tid.x")
  Value(const Reg& r, ValType type);  // wrap existing Reg as register Value

  Value(int64_t v) {
    *this = imm(ValType::S64, v);
  }

  // --- String representation ---
  // Returns register name for registers, literal string for immediates.
  const std::string& str() const;

  static Value imm(ValType, int64_t);

  // --- RAII ---
  ~Value();
  Value(Value&& o);
  Value(const Value& o) {
    *this = o;
  }
  Value& operator=(Value&& o);
  Value& operator=(int64_t v);      // emit mov with immediate
  Value& operator=(const Value& o); // emit mov

  // --- Implicit conversion to Reg (for lvalue use) ---
  operator const Reg&() const;

  // --- Arithmetic — result has same type as *this ---
  Value operator+(const Value& b) const;
  Value operator-(const Value& b) const;
  Value operator*(const Value& b) const;
  Value operator/(const Value& b) const;
  Value operator%(const Value& b) const;

  // --- Bitwise — result has same type as *this ---
  Value operator&(const Value& b) const;
  Value operator|(const Value& b) const;
  Value operator^(const Value& b) const;
  Value operator~() const;
  Value operator<<(const Value& b) const;
  Value operator>>(const Value& b) const;

  // --- Comparison — result is Pred ---
  Value operator<(const Value& b) const;
  Value operator<=(const Value& b) const;
  Value operator>(const Value& b) const;
  Value operator>=(const Value& b) const;
  Value operator==(const Value& b) const;
  Value operator!=(const Value& b) const;

  // --- Predicate negation ---
  Value operator!() const;

  // --- Compound assignment — modifies in place ---
  void operator+=(const Value& b);
  void operator-=(const Value& b);
  void operator*=(const Value& b);
  void operator/=(const Value& b);
  void operator%=(const Value& b);
  void operator^=(const Value& b);
  void operator&=(const Value& b);
  void operator|=(const Value& b);
  void operator<<=(const Value& b);
  void operator>>=(const Value& b);

  Value operator-() const;
};

// Backward compatibility alias
using Val = Value;

// Forward declaration for Operand proxy type used in TValue arithmetic operators.
template<ValType T>
struct Operand;

// Typed PTX value — wraps Value with compile-time type constraints.
// Integer constructors always allocate a register (safe for loop variables).
// Arithmetic operators accept Operand<T>, which is implicitly constructable from
// integers (immediate, no register) or TValue<T> (borrows existing register).
// Comparisons return plain Value (pred). Implicit conversion to const Value& avoids mov.
template<ValType T>
struct TValue {
  Value inner;

  TValue() : inner(T) {}
  TValue(Value v) : inner(std::move(v)) {
    if (inner.type != T) {
      throw std::runtime_error("ptx::TValue: type mismatch");
    }
  }
  // Integer constructors always allocate a register + emit mov
  TValue(int64_t v) : inner(T) {
    inner = v;
  }

  operator const Value&() const {
    return inner;
  }

  TValue& operator=(Value v) {
    inner = std::move(v);
    return *this;
  }
  TValue& operator=(int64_t v) {
    inner = v;
    return *this;
  }

  TValue operator+(const Operand<T>& b) const;
  TValue operator-(const Operand<T>& b) const;
  TValue operator*(const Operand<T>& b) const;
  TValue operator/(const Operand<T>& b) const;
  TValue operator%(const Operand<T>& b) const;
  TValue operator&(const Operand<T>& b) const;
  TValue operator|(const Operand<T>& b) const;
  TValue operator^(const Operand<T>& b) const;
  TValue operator~() const {
    return ~inner;
  }
  TValue operator<<(const Operand<T>& b) const;
  TValue operator>>(const Operand<T>& b) const;

  void operator+=(const Operand<T>& b);
  void operator-=(const Operand<T>& b);
  void operator*=(const Operand<T>& b);
  void operator/=(const Operand<T>& b);
  void operator%=(const Operand<T>& b);
  void operator&=(const Operand<T>& b);
  void operator|=(const Operand<T>& b);
  void operator^=(const Operand<T>& b);
  void operator<<=(const Operand<T>& b);
  void operator>>=(const Operand<T>& b);

  Value operator<(const Operand<T>& b) const;
  Value operator<=(const Operand<T>& b) const;
  Value operator>(const Operand<T>& b) const;
  Value operator>=(const Operand<T>& b) const;
  Value operator==(const Operand<T>& b) const;
  Value operator!=(const Operand<T>& b) const;

  TValue operator-() const;
};

// Operand proxy — used as argument type for TValue arithmetic operators.
// Integers become immediates (no register allocation); TValues borrow their register.
template<ValType T>
struct Operand {
  Value owned;      // owns the Value for integer immediates or moved-in temporaries
  const Value* ref; // always points to the operand value

  Operand(int64_t v) : owned(Value::imm(T, v)), ref(&owned) {}
  Operand(const Value& v) : ref(&v) {}                               // borrow any Value
  Operand(const TValue<T>& v) : ref(&v.inner) {}                     // borrow existing register
  Operand(TValue<T>&& v) : owned(std::move(v.inner)), ref(&owned) {} // take ownership of temporary

  const Value& get() const {
    return *ref;
  }
};

// TValue operator definitions (after Operand is complete)
template<ValType T>
TValue<T> TValue<T>::operator+(const Operand<T>& b) const {
  return inner + b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator-(const Operand<T>& b) const {
  return inner - b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator*(const Operand<T>& b) const {
  return inner * b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator/(const Operand<T>& b) const {
  auto& bv = b.get();
  if ((inner.type == ValType::U32 || inner.type == ValType::U64) && bv.kind == Value::Immediate && bv.immValue) {
    uint64_t uv = *bv.immValue;
    if ((int64_t)uv > 0 && std::popcount(uv) == 1) {
      return inner >> std::countr_zero(uv);
    }
  }
  return inner / b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator%(const Operand<T>& b) const {
  auto& bv = b.get();
  if ((inner.type == ValType::U32 || inner.type == ValType::U64) && bv.kind == Value::Immediate && bv.immValue) {
    uint64_t uv = *bv.immValue;
    if ((int64_t)uv > 0 && std::popcount(uv) == 1) {
      return inner & (uv - 1);
    }
  }
  return inner % b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator&(const Operand<T>& b) const {
  return inner & b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator|(const Operand<T>& b) const {
  return inner | b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator^(const Operand<T>& b) const {
  return inner ^ b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator<<(const Operand<T>& b) const {
  return inner << b.get();
}
template<ValType T>
TValue<T> TValue<T>::operator>>(const Operand<T>& b) const {
  return inner >> b.get();
}
template<ValType T>
void TValue<T>::operator+=(const Operand<T>& b) {
  inner += b.get();
}
template<ValType T>
void TValue<T>::operator-=(const Operand<T>& b) {
  inner -= b.get();
}
template<ValType T>
void TValue<T>::operator*=(const Operand<T>& b) {
  inner *= b.get();
}
template<ValType T>
void TValue<T>::operator/=(const Operand<T>& b) {
  inner /= b.get();
}
template<ValType T>
void TValue<T>::operator%=(const Operand<T>& b) {
  inner %= b.get();
}
template<ValType T>
void TValue<T>::operator&=(const Operand<T>& b) {
  inner &= b.get();
}
template<ValType T>
void TValue<T>::operator|=(const Operand<T>& b) {
  inner |= b.get();
}
template<ValType T>
void TValue<T>::operator^=(const Operand<T>& b) {
  inner ^= b.get();
}
template<ValType T>
void TValue<T>::operator<<=(const Operand<T>& b) {
  inner <<= b.get();
}
template<ValType T>
void TValue<T>::operator>>=(const Operand<T>& b) {
  inner >>= b.get();
}
template<ValType T>
Value TValue<T>::operator<(const Operand<T>& b) const {
  return inner < b.get();
}
template<ValType T>
Value TValue<T>::operator<=(const Operand<T>& b) const {
  return inner <= b.get();
}
template<ValType T>
Value TValue<T>::operator>(const Operand<T>& b) const {
  return inner > b.get();
}
template<ValType T>
Value TValue<T>::operator>=(const Operand<T>& b) const {
  return inner >= b.get();
}
template<ValType T>
Value TValue<T>::operator==(const Operand<T>& b) const {
  return inner == b.get();
}
template<ValType T>
Value TValue<T>::operator!=(const Operand<T>& b) const {
  return inner != b.get();
}

template<ValType T>
TValue<T> TValue<T>::operator-() const {
  return -inner;
}

using u32 = TValue<ValType::U32>;
using s32 = TValue<ValType::S32>;
using u64 = TValue<ValType::U64>;

template<ValType T>
inline TValue<T> operator+(int64_t a, const TValue<T>& b) {
  return Value::imm(T, a) + b;
}
template<ValType T>
inline TValue<T> operator*(int64_t a, const TValue<T>& b) {
  return Value::imm(T, a) * b;
}

template<ValType T>
inline TValue<ValType::Pred> operator<(int64_t a, const TValue<T>& b) {
  return Value::imm(T, a) < b;
}

inline Value to_u16() {
  return Value(ValType::U16);
}
inline Value to_u16(const Value& v) {
  Value r(ValType::U16);
  r = v;
  return r;
}
inline Value to_u32() {
  return Value(ValType::U32);
}
inline Value to_u32(const Value& v) {
  Value r(ValType::U32);
  r = v;
  return r;
}
inline Value to_u64() {
  return Value(ValType::U64);
}
inline Value to_u64(const Value& v) {
  Value r(ValType::U64);
  r = v;
  return r;
}

// ---------------------------------------------------------------------------
// Free-standing instruction helpers (emit to currentBlock)
// ---------------------------------------------------------------------------

// Arithmetic
void add_u32(const Value& d, const Value& a, const Value& b);
void add_s32(const Value& d, const Value& a, const Value& b);
void add_u64(const Value& d, const Value& a, const Value& b);
void add_s64(const Value& d, const Value& a, const Value& b);
void sub_u32(const Value& d, const Value& a, const Value& b);
void sub_s32(const Value& d, const Value& a, const Value& b);
void sub_u64(const Value& d, const Value& a, const Value& b);
void mul_lo_u32(const Value& d, const Value& a, const Value& b);
void mul_lo_s32(const Value& d, const Value& a, const Value& b);
void mul_lo_u64(const Value& d, const Value& a, const Value& b);
void mul_lo_s64(const Value& d, const Value& a, const Value& b);
void mul_wide_u32(const Value& d, const Value& a, const Value& b);
void mul_wide_s32(const Value& d, const Value& a, const Value& b);
void div_u32(const Value& d, const Value& a, const Value& b);
void div_s32(const Value& d, const Value& a, const Value& b);
void rem_u32(const Value& d, const Value& a, const Value& b);
void rem_s32(const Value& d, const Value& a, const Value& b);
void rem_u64(const Value& d, const Value& a, const Value& b);
void min_u32(const Value& d, const Value& a, const Value& b);
void min_s32(const Value& d, const Value& a, const Value& b);
void max_u32(const Value& d, const Value& a, const Value& b);
void max_s32(const Value& d, const Value& a, const Value& b);
inline Value min_u32(const Value& a, const Value& b) {
  Value result(ValType::U32);
  min_u32(result, a, b);
  return result;
}

// Bitwise
void and_pred(const Value& d, const Value& a, const Value& b);
void or_pred(const Value& d, const Value& a, const Value& b);
void and_b32(const Value& d, const Value& a, const Value& b);
void and_b64(const Value& d, const Value& a, const Value& b);
void or_b32(const Value& d, const Value& a, const Value& b);
void or_b64(const Value& d, const Value& a, const Value& b);
void xor_b32(const Value& d, const Value& a, const Value& b);
void not_b32(const Value& d, const Value& a);
void shl_b32(const Value& d, const Value& a, const Value& b);
void shl_b64(const Value& d, const Value& a, const Value& b);
void shr_u32(const Value& d, const Value& a, const Value& b);
void shr_s32(const Value& d, const Value& a, const Value& b);

void neg_s32(const Value& d, const Value& a);

// Comparison
void setp_eq_u32(const Value& p, const Value& a, const Value& b);
void setp_ne_u32(const Value& p, const Value& a, const Value& b);
void setp_lt_u32(const Value& p, const Value& a, const Value& b);
void setp_le_u32(const Value& p, const Value& a, const Value& b);
void setp_gt_u32(const Value& p, const Value& a, const Value& b);
void setp_ge_u32(const Value& p, const Value& a, const Value& b);
void setp_eq_s32(const Value& p, const Value& a, const Value& b);
void setp_ne_s32(const Value& p, const Value& a, const Value& b);
void setp_lt_s32(const Value& p, const Value& a, const Value& b);
void setp_ge_s32(const Value& p, const Value& a, const Value& b);
void setp_ne_s64(const Value& p, const Value& a, const Value& b);
void setp_eq_u64(const Value& p, const Value& a, const Value& b);
void setp_ne_u64(const Value& p, const Value& a, const Value& b);
void setp_lt_s64(const Value& p, const Value& a, const Value& b);
void setp_ge_s64(const Value& p, const Value& a, const Value& b);

// Move
void mov_u32(const Value& d, const Value& a);
void mov_u64(const Value& d, const Value& a);
void mov_b64(const Value& d, const Value& a);

// Load
void ld_param_u32(const Value& d, const std::string& paramName);
void ld_param_u64(const Value& d, const std::string& paramName);
void ld_param_u32(const Value& d, const Value& addr, int offset);
void ld_param_u64(const Value& d, const Value& addr, int offset);
inline u64 ld_param_u64(const Value& addr, int offset) {
  u64 result;
  ld_param_u64(result, addr, offset);
  return result;
}
void ld_global_u32(const Value& d, const Value& addr);
inline Value ld_global_u32(const Value& addr) {
  u32 result;
  ld_global_u32(result, addr);
  return result;
}
void ld_global_u64(const Value& d, const Value& addr);
inline u64 ld_global_u64(const u64& addr) {
  u64 result;
  ld_global_u64(result, addr);
  return result;
}
void ld_const_u32(const Value& d, const Value& addr);
inline u32 ld_const_u32(const Value& addr) {
  u32 result;
  ld_const_u32(result, addr);
  return result;
}
void ld_const_u64(const Value& d, const Value& addr);
inline u64 ld_const_u64(const Value& addr) {
  u64 result;
  ld_const_u64(result, addr);
  return result;
}
void ld_const_u8(const Value& d, const Value& addr);
inline Value ld_const_u8(const Value& addr) {
  Value result(ValType::U32);
  ld_const_u8(result, addr);
  return result;
}
void ld_global_volatile_u32(const Value& d, const Value& addr);
void ld_global_relaxed_sys_u32(const Value& d, const Value& addr);
inline u32 ld_global_relaxed_sys_u32(const Value& addr) {
  u32 result;
  ld_global_relaxed_sys_u32(result, addr);
  return result;
}
void ld_global_relaxed_sys_u64(const Value& d, const Value& addr);
inline u64 ld_global_relaxed_sys_u64(const Value& addr) {
  u64 result;
  ld_global_relaxed_sys_u64(result, addr);
  return result;
}
void ld_global_acquire_sys_u32(const Value& d, const Value& addr);
inline Value ld_global_acquire_sys_u32(const Value& addr) {
  Value result(ValType::U32);
  ld_global_acquire_sys_u32(result, addr);
  return result;
}
void fence_proxy();
void fence();
void fence_acquire_sys();
void fence_release_sys();
void fence_acquire_gpu();
void fence_release_gpu();
void st_global_relaxed_sys_u32(const Value& addr, const Value& val);
void st_global_relaxed_sys_u64(const Value& addr, const Value& val);
void st_global_release_sys_u32(const Value& addr, const Value& val);
void ld_global_cv_v4_u32(const Value& d0, const Value& d1, const Value& d2, const Value& d3, const Value& addr);
void ld_u8(const Value& d, const Value& addr);
void ld_shared_acquire_cta_u32(const Value& d, const Value& addr);
inline Value ld_shared_acquire_cta_u32(const Value& addr) {
  Value result(ValType::U32);
  ld_shared_acquire_cta_u32(result, addr);
  return result;
}
void ld_shared_v4_u32(std::array<Value, 4>& v, const Value& addr);
void st_shared_v4_u32(const Value& addr, const std::array<Value, 4>& v);

u64 ld_global_cv_u64(const u64& addr);
void st_global_wt_u64(const u64& addr, const u64& value);

// Store
void st_global_u32(const Value& addr, const Value& val);
void st_global_u64(const Value& addr, const Value& val);
void st_global_volatile_u32(const Value& addr, const Value& val);
void st_global_wt_v4_u32(const Value& addr, const Value& s0, const Value& s1, const Value& s2, const Value& s3);
void st_u8(const Value& addr, const Value& val);
void st_shared_release_cta_u32(const Value& addr, const Value& val);

// Conversion
void cvt_u64_u32(const Value& d, const Value& a);
void cvt_u32_u64(const Value& d, const Value& a);

// Control flow
struct Label;
void bra(const Block* target);
void bra(const Value& pred, const Block* target);
void bra_div(const Value& pred, const Block* target);
void bra_not(const Value& pred, const Block* target);
void brx_idx(const Value& index, const std::vector<Label>& targets, bool divergent = false);
void ret();

// Atomic
void atom_global_inc_u32(const Value& d, const Value& addr, const Value& b);
void atom_global_relaxed_inc_u32(const Value& d, const Value& addr, const Value& b);
// Atomic add (global, all semantics × {cta,gpu,sys} scopes)
Value atom_global_relaxed_cta_add_u32(const Value& addr, const Value& b);
Value atom_global_relaxed_gpu_add_u32(const Value& addr, const Value& b);
Value atom_global_relaxed_sys_add_u32(const Value& addr, const Value& b);
Value atom_global_acquire_cta_add_u32(const Value& addr, const Value& b);
Value atom_global_acquire_gpu_add_u32(const Value& addr, const Value& b);
Value atom_global_acquire_sys_add_u32(const Value& addr, const Value& b);
Value atom_global_release_cta_add_u32(const Value& addr, const Value& b);
Value atom_global_release_gpu_add_u32(const Value& addr, const Value& b);
Value atom_global_release_sys_add_u32(const Value& addr, const Value& b);
Value atom_global_acq_rel_cta_add_u32(const Value& addr, const Value& b);
Value atom_global_acq_rel_gpu_add_u32(const Value& addr, const Value& b);
Value atom_global_acq_rel_sys_add_u32(const Value& addr, const Value& b);
// Atomic add (shared::cta, all semantics, cta scope)
Value atom_shared_relaxed_cta_add_u32(const Value& addr, const Value& b);
Value atom_shared_acquire_cta_add_u32(const Value& addr, const Value& b);
Value atom_shared_release_cta_add_u32(const Value& addr, const Value& b);
Value atom_shared_acq_rel_cta_add_u32(const Value& addr, const Value& b);

// Synchronization
void barrier_sync(int n = 0);
void membar_sys();
void warp_sync(uint32_t membermask = 0xFFFFFFFF);

// Trap
void trap();

// Raw emit (escape hatch)
void emit(const std::string& inst);

// Test function: generates simple kernels and returns the PTX string.
std::string ptxTest(const char* target = "sm_90a");

// RAII function scope — sets TLS context, manages register free-lists.
// On destruction, writes register high-water marks to Function::regCounts[].
struct FunctionScope {
  Function* fn;
  FunctionScope(Function* f);
  ~FunctionScope();
  FunctionScope(const FunctionScope&) = delete;
  FunctionScope& operator=(const FunctionScope&) = delete;
};

// Create a new block with auto-generated label, append to current function,
// and set as current block.
Block* activateNewBlock(const char* prefix);

// Control flow scope guard — iterates exactly once via range-for trick.
// Owns a pending block that is activated when the scope closes.
struct ScopeGuard {
  std::unique_ptr<Block> pendingBlock;
  Block* backEdgeTarget = nullptr;             // WHILE: emit bra before activating exit
  moodist::Function<void(ScopeGuard&)> stepFn; // FOR: emit step before back-edge
  bool closed = false;

  std::vector<Label> caseLabels;

  ScopeGuard() = default;
  ScopeGuard(ScopeGuard&& o) noexcept;
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ~ScopeGuard();
};

ScopeGuard _If(const Value& pred);
ScopeGuard _IfD(const Value& pred); // divergent IF: uses bra (not bra.uni)
ScopeGuard _Else();
ScopeGuard _WhileImpl(Block* header, const Value& pred);
ScopeGuard _Skip();

ScopeGuard _Switch(const u32& index, bool divergent);
ScopeGuard _Case(ScopeGuard& scope, uint32_t value);

template<typename F>
ScopeGuard _While(F&& condFn) {
  Block* header = activateNewBlock("while");
  Value cond = condFn();
  return _WhileImpl(header, cond);
}

template<typename CondFn, typename StepFn>
ScopeGuard _For(CondFn&& condFn, StepFn&& stepFn) {
  Block* header = activateNewBlock("for");
  Value pred = condFn();
  auto sg = _WhileImpl(header, pred);
  sg.stepFn = std::forward<StepFn>(stepFn);
  return sg;
}

#define SKIP if ([[maybe_unused]] auto _ptx_scope_ = ::moodist::ptx::_Skip(); true)

#define IF(pred) if ([[maybe_unused]] auto _ptx_scope_ = ::moodist::ptx::_If(pred); true)
#define IF_D(pred) if ([[maybe_unused]] auto _ptx_scope_ = ::moodist::ptx::_IfD(pred); true)
#define ELSE if ([[maybe_unused]] auto _ptx_scope_ = ::moodist::ptx::_Else(); true)
#define WHILE(cond)                                                                                                    \
  if ([[maybe_unused]] auto _ptx_while_scope_ = ::moodist::ptx::_While([&]() {                                         \
        return (cond);                                                                                                 \
      });                                                                                                              \
      true)
#define BREAK ::moodist::ptx::bra(_ptx_while_scope_.pendingBlock.get())
#define CONTINUE ::moodist::ptx::bra(_ptx_while_scope_.backEdgeTarget)
#define FOR(init, cond, step)                                                                                          \
  if (init; true)                                                                                                      \
    if ([[maybe_unused]] auto _ptx_while_scope_ = ::moodist::ptx::_For(                                                \
            [&]() {                                                                                                    \
              return (cond);                                                                                           \
            },                                                                                                         \
            [&](ScopeGuard&) {                                                                                         \
              step;                                                                                                    \
            });                                                                                                        \
        true)

#define SWITCH(index) if ([[maybe_unused]] auto _ptx_switch_scope = ::moodist::ptx::_Switch(index, false); true)
#define SWITCH_D(index) if ([[maybe_unused]] auto _ptx_switch_scope = ::moodist::ptx::_Switch(index, true); true)
#define CASE(index) if ([[maybe_unused]] auto _ptx_case_scope = ::moodist::ptx::_Case(_ptx_switch_scope, index); true)

// Predicated execution — all instructions in body are prefixed with @pred.

// Labels — declare anywhere, GOTO to jump, LABEL to place.
// Supports both forward jumps (GOTO before LABEL) and backward jumps/loops (GOTO after LABEL).
//   Label tail("tail");
//   ...
//   GOTO(tail);
//   ...
//   LABEL(tail);  // subsequent instructions go here
struct Label {
  std::unique_ptr<Block> block;
  Block* rawPtr = nullptr; // stable pointer, valid before and after LABEL
  Label();
  Label(const char* name);
  Label(const Label&) = delete;
  Label& operator=(const Label&) = delete;
  Label(Label&&) = default;
  Label& operator=(Label&&) = default;
};
void activateLabel(Label& label);

#define GOTO(label) ::moodist::ptx::bra((label).rawPtr)
#define GOTO_IF(pred, label) ::moodist::ptx::bra(pred, (label).rawPtr)
#define GOTO_IF_D(pred, label) ::moodist::ptx::bra_div(pred, (label).rawPtr)
#define GOTO_IF_NOT(pred, label) ::moodist::ptx::bra_not(pred, (label).rawPtr)
#define LABEL(label) ::moodist::ptx::activateLabel(label)

// Predicated execution — all instructions in body are prefixed with @pred.
struct PredGuard {
  PredGuard(const Value& pred);
  ~PredGuard();
  PredGuard(const PredGuard&) = delete;
  PredGuard& operator=(const PredGuard&) = delete;
  PredGuard(PredGuard&&) = delete;
};
#define PRED(pred) if ([[maybe_unused]] auto _ptx_pred_ = ::moodist::ptx::PredGuard(pred); true)

// Special registers
u32 threadIdx_x();
u32 blockIdx_x();
u32 blockDim_x();
u64 clock64();

// Parameter loading — emits ld.param, returns typed Value
Value loadParam(int index, ValType type);

// Parameter struct access — for .param .b8 byte-array params
Value paramBase(int index); // mov.b64 of param address, returns U64
Value loadParamField(const Value& base, int offset, ValType type);

// Type conversion
Value widen(const Value& v);  // U32→U64, S32→S64
Value narrow(const Value& v); // U64→U32, S64→S32
inline u64 widen(const u32& v) {
  return widen(v.inner);
}
inline u32 narrow(const u64& v) {
  return narrow(v.inner);
}

// Memory operations
void storeGlobal(const Value& addr, const Value& val);
Value loadGlobalVolatile(const Value& addr, ValType type);
void storeGlobalVolatile(const Value& addr, const Value& val);
Value loadGlobalAcquireSys(const Value& addr, ValType type);
void storeGlobalRelaxedSys(const Value& addr, const Value& val);
void storeGlobalReleaseSys(const Value& addr, const Value& val);

// Vectorized load/store (4 x u32)
void ld_global_cv_v4_u32(std::array<Value, 4>& v, const Value& addr);
void ld_global_nc_v4_u32(std::array<Value, 4>& v, const Value& addr);
void ld_global_cs_v4_u32(std::array<Value, 4>& v, const Value& addr);
void ld_v4_u32(std::array<Value, 4>& v, const Value& addr); // generic addressing (no cache qualifier)
void st_global_wt_v4_u32(const Value& addr, const std::array<Value, 4>& v);
void ldcv_v4(Value& v0, Value& v1, Value& v2, Value& v3, const Value& addr);
void ldcv_v4(std::array<Value, 4>& v, const Value& addr);
void ldnc_v4(Value& v0, Value& v1, Value& v2, Value& v3, const Value& addr);
void ldnc_v4(std::array<Value, 4>& v, const Value& addr);
void ldcs_v4(Value& v0, Value& v1, Value& v2, Value& v3, const Value& addr);
void ldcs_v4(std::array<Value, 4>& v, const Value& addr);
void ld_plain_v4(std::array<Value, 4>& v, const Value& addr); // generic addressing, no cache qualifier
void stwt_v4(const Value& addr, const Value& v0, const Value& v1, const Value& v2, const Value& v3);
void stwt_v4(const Value& addr, const std::array<Value, 4>& v);

// Atomic operations
Value atomicInc(const Value& addr, const Value& modulo);

// Global variable address
Value globalAddr(const char* name); // mov.u64 of global symbol address, returns U64
Value constAddr(const char* name);  // mov.u32 of const symbol address, returns U32
Value sharedAddr(const char* name); // mov.u32 of shared symbol address, returns U32 (native shared-space address)
Value addShared(int align, int sizeBytes, const char* suffix = nullptr);   // declare shared mem, return U64 address
Value addShared32(int align, int sizeBytes, const char* suffix = nullptr); // declare shared mem, return U32 address
u64 addGlobalVar(int align, int sizeBytes, const char* suffix = nullptr);  // declare global var, return U64 address
Value addConst32(
    int align, const void* data, size_t size, const char* name); // declare initialized const, return U32 address

template<typename T>
Value addConst32(const std::vector<T>& vec, const char* name) {
  if (vec.empty()) {
    char zero = 0;
    return addConst32(alignof(T), &zero, 1, name);
  }
  return addConst32(alignof(T), vec.data(), vec.size() * sizeof(T), name);
}

// Hex immediate — for baking GPU addresses into PTX
// Value hexImm(uintptr_t value);

// mbarrier operations (sm_90+)
void mbarrier_init(const Value& addr, int count);
void mbarrier_inval(const Value& addr);
Value mbarrier_arrive(const Value& addr); // returns state token (u64)
Value mbarrier_arrive_noComplete(
    const Value& addr); // arrive with .noComplete hint (caller guarantees this won't complete the phase)
void mbarrier_expect_tx(const Value& addr, const Value& txCount);
Value mbarrier_try_wait_parity(const Value& addr, const Value& phaseParity); // returns pred
inline void mbarrier_wait_parity(const Value& addr, const Value& phaseParity) {
  WHILE(!mbarrier_try_wait_parity(addr, phaseParity)) {}
}

// cp.async.bulk (sm_90+)
void cp_async_bulk_shared_global(const Value& dst, const Value& src, const Value& size, const Value& mbar);
void cp_async_bulk_global_shared(const Value& dst, const Value& src, const Value& size);
void multimem_cp_async_bulk_global_shared(const Value& dst, const Value& src, const Value& size);
void cp_async_bulk_prefetch_l2(const Value& src, const Value& size);
void cp_async_bulk_commit_group();
void cp_async_bulk_wait_group(int n);
void cp_async_bulk_wait_group_read(int n);

// Named barrier with explicit thread count (bar.sync barrierID, threadCount)
void bar_sync(int barrierId, int threadCount);
void bar_sync(const Value& barrierId, int threadCount);
// bar.red reductions — return count/predicate result in d
Value bar_red_popc(int barrierId, int threadCount, const Value& pred);
Value bar_red_popc(const Value& barrierId, int threadCount, const Value& pred);
Value bar_red_and(int barrierId, int threadCount, const Value& pred);
Value bar_red_and(const Value& barrierId, int threadCount, const Value& pred);
Value bar_red_or(int barrierId, int threadCount, const Value& pred);
Value bar_red_or(const Value& barrierId, int threadCount, const Value& pred);
// shfl.sync — warp shuffle (sm_70+). b=source lane/offset, c=clamp/segment, membermask=participant mask.
// Broadcast from lane 0: shfl_sync_idx_b32(val, 0, 0x1f, 0xffffffff)
Value shfl_sync_idx_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask = 0xffffffff);
Value shfl_sync_up_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask = 0xffffffff);
Value shfl_sync_down_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask = 0xffffffff);
Value shfl_sync_bfly_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask = 0xffffffff);
void bar_arrive(int barrierId, int threadCount);
void bar_arrive(const Value& barrierId, int threadCount);

// Shared memory address conversion
Value cvta_shared(const Value& sharedAddr); // convert shared addr to generic

} // namespace ptx
} // namespace moodist
