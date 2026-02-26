// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <array>
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

// Instruction operand — wraps a register, immediate, or special name.
struct Operand {
  std::string str;
  Operand(const Reg& r) : str(r.name) {}
  Operand(int32_t v) : str(std::to_string(v)) {}
  Operand(uint32_t v) : str(std::to_string(v)) {}
  Operand(int64_t v) : str(std::to_string(v)) {}
  Operand(uint64_t v) : str(std::to_string(v)) {}
  Operand(const char* s) : str(s) {} // for "%tid.x" etc.
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

  // Basic blocks — owned by the function, serialized in order
  std::vector<std::unique_ptr<Block>> blocks;
  Block* newBlock(const char* label);

  // Serialize the function body to a PTX string (entry point + body)
  std::string finalize() const;
};

// Top-level PTX module containing globals and functions.
struct Module {
  std::string version = "8.7";
  std::string target;
  int addressSize = 64;

  std::vector<std::string> globals;
  std::vector<std::unique_ptr<Function>> functions;

  Function* newFunction(const char* name);

  // Serialize the complete module to a PTX string
  std::string finalize() const;
};

// ---------------------------------------------------------------------------
// TLS emit targets — set these before emitting instructions
// ---------------------------------------------------------------------------

void setModule(Module* m);
void setFunction(Function* f);
void setBlock(Block* b);

// Register allocation via currentFunction
Reg reg(RegType type);

// ---------------------------------------------------------------------------
// Free-standing instruction helpers (emit to currentBlock)
// ---------------------------------------------------------------------------

// Arithmetic
void add_u32(const Reg& d, const Operand& a, const Operand& b);
void add_s32(const Reg& d, const Operand& a, const Operand& b);
void add_u64(const Reg& d, const Operand& a, const Operand& b);
void add_s64(const Reg& d, const Operand& a, const Operand& b);
void sub_u32(const Reg& d, const Operand& a, const Operand& b);
void sub_s32(const Reg& d, const Operand& a, const Operand& b);
void mul_lo_u32(const Reg& d, const Operand& a, const Operand& b);
void mul_lo_s32(const Reg& d, const Operand& a, const Operand& b);
void mul_lo_u64(const Reg& d, const Operand& a, const Operand& b);
void mul_lo_s64(const Reg& d, const Operand& a, const Operand& b);
void mul_wide_u32(const Reg& d, const Operand& a, const Operand& b);
void mul_wide_s32(const Reg& d, const Operand& a, const Operand& b);
void div_u32(const Reg& d, const Operand& a, const Operand& b);
void div_s32(const Reg& d, const Operand& a, const Operand& b);
void rem_u32(const Reg& d, const Operand& a, const Operand& b);
void rem_s32(const Reg& d, const Operand& a, const Operand& b);
void min_u32(const Reg& d, const Operand& a, const Operand& b);
void min_s32(const Reg& d, const Operand& a, const Operand& b);
void max_u32(const Reg& d, const Operand& a, const Operand& b);
void max_s32(const Reg& d, const Operand& a, const Operand& b);

// Bitwise
void and_pred(const Reg& d, const Operand& a, const Operand& b);
void or_pred(const Reg& d, const Operand& a, const Operand& b);
void and_b32(const Reg& d, const Operand& a, const Operand& b);
void and_b64(const Reg& d, const Operand& a, const Operand& b);
void or_b32(const Reg& d, const Operand& a, const Operand& b);
void or_b64(const Reg& d, const Operand& a, const Operand& b);
void xor_b32(const Reg& d, const Operand& a, const Operand& b);
void not_b32(const Reg& d, const Operand& a);
void shl_b32(const Reg& d, const Operand& a, const Operand& b);
void shl_b64(const Reg& d, const Operand& a, const Operand& b);
void shr_u32(const Reg& d, const Operand& a, const Operand& b);
void shr_s32(const Reg& d, const Operand& a, const Operand& b);

// Comparison
void setp_eq_u32(const Reg& p, const Operand& a, const Operand& b);
void setp_ne_u32(const Reg& p, const Operand& a, const Operand& b);
void setp_lt_u32(const Reg& p, const Operand& a, const Operand& b);
void setp_le_u32(const Reg& p, const Operand& a, const Operand& b);
void setp_gt_u32(const Reg& p, const Operand& a, const Operand& b);
void setp_ge_u32(const Reg& p, const Operand& a, const Operand& b);
void setp_eq_s32(const Reg& p, const Operand& a, const Operand& b);
void setp_ne_s32(const Reg& p, const Operand& a, const Operand& b);
void setp_lt_s32(const Reg& p, const Operand& a, const Operand& b);
void setp_ge_s32(const Reg& p, const Operand& a, const Operand& b);
void setp_ne_s64(const Reg& p, const Operand& a, const Operand& b);
void setp_lt_s64(const Reg& p, const Operand& a, const Operand& b);
void setp_ge_s64(const Reg& p, const Operand& a, const Operand& b);

// Move
void mov_u32(const Reg& d, const Operand& a);
void mov_u64(const Reg& d, const Operand& a);
void mov_b64(const Reg& d, const Operand& a);

// Load
void ld_param_u32(const Reg& d, const std::string& paramName);
void ld_param_u64(const Reg& d, const std::string& paramName);
void ld_param_u32(const Reg& d, const Reg& addr, int offset);
void ld_param_u64(const Reg& d, const Reg& addr, int offset);
void ld_global_u32(const Reg& d, const Operand& addr);
void ld_global_volatile_u32(const Reg& d, const Operand& addr);
void ld_global_cv_v4_u32(const Reg& d0, const Reg& d1, const Reg& d2, const Reg& d3, const Operand& addr);
void ld_u8(const Reg& d, const Operand& addr);

// Store
void st_global_u32(const Operand& addr, const Operand& val);
void st_global_volatile_u32(const Operand& addr, const Operand& val);
void st_global_wt_v4_u32(const Operand& addr, const Reg& s0, const Reg& s1, const Reg& s2, const Reg& s3);
void st_u8(const Operand& addr, const Operand& val);

// Conversion
void cvt_u64_u32(const Reg& d, const Operand& a);
void cvt_u32_u64(const Reg& d, const Operand& a);

// Control flow
void bra(const Block* target);
void bra(const Reg& pred, const Block* target);
void bra_not(const Reg& pred, const Block* target);
void ret();

// Atomic
void atom_global_inc_u32(const Reg& d, const Operand& addr, const Operand& b);

// Synchronization
void barrier_sync(int n = 0);
void membar_sys();
void warp_sync(uint32_t membermask = 0xFFFFFFFF);

// Raw emit (escape hatch)
void emit(const std::string& inst);

// Test function: generates simple kernels and returns the PTX string.
std::string ptxTest(const char* target = "sm_90a");

// ---------------------------------------------------------------------------
// DSL layer — typed values, RAII registers, operators, control flow
// ---------------------------------------------------------------------------

// Typed value — determines instruction selection (signedness + width).
enum class ValType { Pred, U16, S16, U32, S32, U64, S64 };

// Map ValType to RegType for register allocation.
RegType regTypeFor(ValType type);

// RAII register — allocates from free-list on construct, returns on destruct.
// Implicitly converts to Reg (for destination params) and Operand (for source params).
struct Val {
  Reg reg;
  ValType type;
  bool valid = false;

  Val() = default;
  explicit Val(ValType type);
  ~Val();
  Val(Val&& o) noexcept;
  Val(const Val& o) noexcept {
    *this = o;
  }
  Val& operator=(Val&& o) noexcept;
  Val& operator=(int64_t v);    // emit mov with immediate
  Val& operator=(const Val& o); // emit mov (same register type required)

  operator const Reg&() const {
    return reg;
  }
  operator Operand() const {
    return Operand(reg);
  }

  // Arithmetic — result has same type as *this
  Val operator+(const Operand& b) const;
  Val operator-(const Operand& b) const;
  Val operator*(const Operand& b) const;
  Val operator/(const Operand& b) const;
  Val operator%(const Operand& b) const;

  // Bitwise — result has same type as *this
  Val operator&(const Operand& b) const;
  Val operator|(const Operand& b) const;
  Val operator^(const Operand& b) const;
  Val operator~() const;
  Val operator<<(const Operand& b) const;
  Val operator>>(const Operand& b) const;

  // Comparison — result is Pred
  Val operator<(const Operand& b) const;
  Val operator<=(const Operand& b) const;
  Val operator>(const Operand& b) const;
  Val operator>=(const Operand& b) const;
  Val operator==(const Operand& b) const;
  Val operator!=(const Operand& b) const;

  // Predicate negation
  Val operator!() const;

  // Compound assignment — modifies in place
  void operator+=(const Operand& b);
  void operator-=(const Operand& b);
  void operator*=(const Operand& b);
  void operator/=(const Operand& b);
  void operator%=(const Operand& b);
  void operator^=(const Operand& b);
};

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
  Block* backEdgeTarget = nullptr; // WHILE: emit bra before activating exit
  bool closed = false;

  ScopeGuard() = default;
  ScopeGuard(ScopeGuard&& o) noexcept;
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

ScopeGuard _If(const Reg& pred);
ScopeGuard _Else();
ScopeGuard _WhileImpl(Block* header, const Reg& pred);

template<typename F>
ScopeGuard _While(F&& condFn) {
  Block* header = activateNewBlock("while");
  Val pred = condFn();
  return _WhileImpl(header, pred);
}

#define IF(pred) for ([[maybe_unused]] auto _ptx_scope_ : ::moodist::ptx::_If(pred))
#define ELSE for ([[maybe_unused]] auto _ptx_scope_ : ::moodist::ptx::_Else())
#define WHILE(cond)                                                                                                    \
  for ([[maybe_unused]] auto _ptx_scope_ : ::moodist::ptx::_While([&]() {                                              \
         return (cond);                                                                                                \
       }))

// Special registers
Val threadIdx_x();
Val blockIdx_x();
Val blockDim_x();

// Parameter loading — emits ld.param, returns typed Val
Val loadParam(int index, ValType type);

// Parameter struct access — for .param .b8 byte-array params
Val paramBase(int index); // mov.b64 of param address, returns U64
Val loadParamField(const Val& base, int offset, ValType type);

// Type conversion
Val widen(const Val& v);  // U32→U64, S32→S64
Val narrow(const Val& v); // U64→U32, S64→S32

// Memory operations
void storeGlobal(const Val& addr, const Val& val);
Val loadGlobalVolatile(const Val& addr, ValType type);
void storeGlobalVolatile(const Val& addr, const Val& val);
Val loadGlobalAcquireSys(const Val& addr, ValType type);
void storeGlobalRelaxedSys(const Val& addr, const Val& val);
void storeGlobalReleaseSys(const Val& addr, const Val& val);

// Vectorized load/store (4 x u32)
void ld_global_cv_v4_u32(std::array<Val, 4>& v, const Operand& addr);
void ld_global_nc_v4_u32(std::array<Val, 4>& v, const Operand& addr);
void ld_global_cs_v4_u32(std::array<Val, 4>& v, const Operand& addr);
void ld_v4_u32(std::array<Val, 4>& v, const Operand& addr); // generic addressing (no cache qualifier)
void st_global_wt_v4_u32(const Operand& addr, const std::array<Val, 4>& v);
void ldcv_v4(Val& v0, Val& v1, Val& v2, Val& v3, const Val& addr);
void ldcv_v4(std::array<Val, 4>& v, const Val& addr);
void ldnc_v4(Val& v0, Val& v1, Val& v2, Val& v3, const Val& addr);
void ldnc_v4(std::array<Val, 4>& v, const Val& addr);
void ldcs_v4(Val& v0, Val& v1, Val& v2, Val& v3, const Val& addr);
void ldcs_v4(std::array<Val, 4>& v, const Val& addr);
void ld_plain_v4(std::array<Val, 4>& v, const Val& addr); // generic addressing, no cache qualifier
void stwt_v4(const Val& addr, const Val& v0, const Val& v1, const Val& v2, const Val& v3);
void stwt_v4(const Val& addr, const std::array<Val, 4>& v);

// Atomic operations
Val atomicInc(const Val& addr, const Operand& modulo);

// Global variable address
Val globalAddr(const char* name); // mov.u64 of global symbol address, returns U64

// Hex immediate — for baking GPU addresses into PTX
Operand hexImm(uintptr_t value);

// mbarrier operations (sm_90+)
void mbarrier_init(const Val& addr, int count);
Val mbarrier_arrive(const Val& addr); // returns state token (u64)
void mbarrier_expect_tx(const Val& addr, const Val& txCount);
Val mbarrier_try_wait_parity(const Val& addr, const Val& phaseParity); // returns pred

// cp.async.bulk (sm_90+)
void cp_async_bulk_shared_global(const Val& dst, const Val& src, const Val& size, const Val& mbar);
void cp_async_bulk_global_shared(const Val& dst, const Val& src, const Val& size);
void cp_async_bulk_commit_group();
void cp_async_bulk_wait_group(int n);

// Shared memory address conversion
Val cvta_shared(const Val& sharedAddr); // convert shared addr to generic

} // namespace ptx
} // namespace moodist
