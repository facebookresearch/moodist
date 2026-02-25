// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx.h"

#include <cstdio>
#include <stdexcept>

namespace moodist {
namespace ptx {

// ---------------------------------------------------------------------------
// TLS state
// ---------------------------------------------------------------------------

static thread_local Module* currentModule = nullptr;
static thread_local Function* currentFunction = nullptr;
static thread_local Block* currentBlock = nullptr;

void setModule(Module* m) {
  currentModule = m;
}
void setFunction(Function* f) {
  currentFunction = f;
}
void setBlock(Block* b) {
  currentBlock = b;
}

Reg reg(RegType type) {
  return currentFunction->reg(type);
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

void Block::emit(const std::string& inst) {
  insts.push_back(inst);
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

static const char* regPrefix(RegType type) {
  switch (type) {
  case RegType::Pred:
    return "%p";
  case RegType::B16:
    return "%rs";
  case RegType::B32:
    return "%r";
  case RegType::B64:
    return "%rd";
  }
  return "";
}

static const char* regDeclType(RegType type) {
  switch (type) {
  case RegType::Pred:
    return ".pred";
  case RegType::B16:
    return ".b16";
  case RegType::B32:
    return ".b32";
  case RegType::B64:
    return ".b64";
  }
  return "";
}

Reg Function::reg(RegType type) {
  int id = regCounts[(int)type]++;
  Reg r;
  r.type = type;
  r.id = id;
  r.name = std::string(regPrefix(type)) + std::to_string(id);
  return r;
}

void Function::addParam(const char* type) {
  int index = (int)paramDecls.size();
  std::string paramName = name + "_param_" + std::to_string(index);
  paramDecls.push_back(std::string(".param ") + type + " " + paramName);
}

void Function::addParamBytes(int align, int size) {
  int index = (int)paramDecls.size();
  std::string paramName = name + "_param_" + std::to_string(index);
  paramDecls.push_back(
      ".param .align " + std::to_string(align) + " .b8 " + paramName + "[" + std::to_string(size) + "]");
}

std::string Function::param(int index) const {
  return name + "_param_" + std::to_string(index);
}

Block* Function::newBlock(const char* label) {
  auto b = std::make_unique<Block>();
  b->label = label;
  Block* ptr = b.get();
  blocks.push_back(std::move(b));
  return ptr;
}

std::string Function::finalize() const {
  std::string s;

  // Entry point and parameters
  s += ".visible .entry " + name + "(\n";
  for (int i = 0; i < (int)paramDecls.size(); i++) {
    s += "    " + paramDecls[i];
    if (i + 1 < (int)paramDecls.size()) {
      s += ",";
    }
    s += "\n";
  }
  s += ")\n";

  if (maxThreads > 0) {
    s += ".maxntid " + std::to_string(maxThreads) + ", 1, 1\n";
  }

  s += "{\n";

  // Register declarations — only emit types that were used
  const RegType types[] = {RegType::Pred, RegType::B16, RegType::B32, RegType::B64};
  for (auto type : types) {
    int count = regCounts[(int)type];
    if (count > 0) {
      s += "    .reg ";
      s += regDeclType(type);
      s += " ";
      s += regPrefix(type);
      s += "<" + std::to_string(count) + ">;\n";
    }
  }

  s += "\n";

  // Blocks
  for (int bi = 0; bi < (int)blocks.size(); bi++) {
    const Block& block = *blocks[bi];

    if (bi > 0) {
      // Non-entry blocks emit their label
      s += "    " + block.label + ":\n";
    }

    for (const auto& inst : block.insts) {
      s += "    " + inst + ";\n";
    }

    if (bi + 1 < (int)blocks.size()) {
      s += "\n";
    }
  }

  s += "\n}\n";
  return s;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

Function* Module::newFunction(const char* name) {
  auto f = std::make_unique<Function>();
  f->name = name;
  Function* ptr = f.get();
  functions.push_back(std::move(f));
  return ptr;
}

std::string Module::finalize() const {
  std::string s;
  s += ".version " + version + "\n";
  s += ".target " + target + "\n";
  s += ".address_size " + std::to_string(addressSize) + "\n";
  s += "\n";

  for (const auto& g : globals) {
    s += g + "\n";
  }
  if (!globals.empty()) {
    s += "\n";
  }

  for (const auto& f : functions) {
    s += f->finalize();
  }

  return s;
}

// ---------------------------------------------------------------------------
// Instruction formatting helpers (private)
// ---------------------------------------------------------------------------

static void emitInst(const std::string& inst) {
  currentBlock->emit(inst);
}

static std::string fmt3(const char* op, const Reg& d, const Operand& a, const Operand& b) {
  return std::string(op) + " " + d.name + ", " + a.str + ", " + b.str;
}

static std::string fmt2(const char* op, const Reg& d, const Operand& a) {
  return std::string(op) + " " + d.name + ", " + a.str;
}

// ---------------------------------------------------------------------------
// Instruction helpers
// ---------------------------------------------------------------------------

// Arithmetic
void add_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("add.u32", d, a, b));
}
void add_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("add.s32", d, a, b));
}
void add_u64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("add.u64", d, a, b));
}
void add_s64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("add.s64", d, a, b));
}
void sub_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("sub.u32", d, a, b));
}
void sub_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("sub.s32", d, a, b));
}
void mul_lo_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("mul.lo.u32", d, a, b));
}
void mul_lo_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("mul.lo.s32", d, a, b));
}
void mul_lo_u64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("mul.lo.u64", d, a, b));
}
void mul_lo_s64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("mul.lo.s64", d, a, b));
}
void mul_wide_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("mul.wide.u32", d, a, b));
}
void mul_wide_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("mul.wide.s32", d, a, b));
}
void div_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("div.u32", d, a, b));
}
void div_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("div.s32", d, a, b));
}
void rem_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("rem.u32", d, a, b));
}
void rem_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("rem.s32", d, a, b));
}
void min_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("min.u32", d, a, b));
}
void min_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("min.s32", d, a, b));
}
void max_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("max.u32", d, a, b));
}
void max_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("max.s32", d, a, b));
}

// Bitwise
void and_pred(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("and.pred", d, a, b));
}
void or_pred(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("or.pred", d, a, b));
}
void and_b32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("and.b32", d, a, b));
}
void and_b64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("and.b64", d, a, b));
}
void or_b32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("or.b32", d, a, b));
}
void or_b64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("or.b64", d, a, b));
}
void xor_b32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("xor.b32", d, a, b));
}
void not_b32(const Reg& d, const Operand& a) {
  emitInst(fmt2("not.b32", d, a));
}
void shl_b32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("shl.b32", d, a, b));
}
void shl_b64(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("shl.b64", d, a, b));
}
void shr_u32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("shr.u32", d, a, b));
}
void shr_s32(const Reg& d, const Operand& a, const Operand& b) {
  emitInst(fmt3("shr.s32", d, a, b));
}

// Comparison
void setp_eq_u32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.eq.u32", p, a, b));
}
void setp_ne_u32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.ne.u32", p, a, b));
}
void setp_lt_u32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.lt.u32", p, a, b));
}
void setp_le_u32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.le.u32", p, a, b));
}
void setp_gt_u32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.gt.u32", p, a, b));
}
void setp_ge_u32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.ge.u32", p, a, b));
}
void setp_eq_s32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.eq.s32", p, a, b));
}
void setp_ne_s32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.ne.s32", p, a, b));
}
void setp_lt_s32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.lt.s32", p, a, b));
}
void setp_ge_s32(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.ge.s32", p, a, b));
}
void setp_ne_s64(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.ne.s64", p, a, b));
}
void setp_lt_s64(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.lt.s64", p, a, b));
}
void setp_ge_s64(const Reg& p, const Operand& a, const Operand& b) {
  emitInst(fmt3("setp.ge.s64", p, a, b));
}

// Move
void mov_u32(const Reg& d, const Operand& a) {
  emitInst(fmt2("mov.u32", d, a));
}
void mov_u64(const Reg& d, const Operand& a) {
  emitInst(fmt2("mov.u64", d, a));
}
void mov_b64(const Reg& d, const Operand& a) {
  emitInst(fmt2("mov.b64", d, a));
}

// Load
void ld_param_u32(const Reg& d, const std::string& paramName) {
  emitInst("ld.param.u32 " + d.name + ", [" + paramName + "]");
}
void ld_param_u64(const Reg& d, const std::string& paramName) {
  emitInst("ld.param.u64 " + d.name + ", [" + paramName + "]");
}
void ld_param_u32(const Reg& d, const Reg& addr, int offset) {
  emitInst("ld.param.u32 " + d.name + ", [" + addr.name + "+" + std::to_string(offset) + "]");
}
void ld_param_u64(const Reg& d, const Reg& addr, int offset) {
  emitInst("ld.param.u64 " + d.name + ", [" + addr.name + "+" + std::to_string(offset) + "]");
}
void ld_global_u32(const Reg& d, const Operand& addr) {
  emitInst("ld.global.u32 " + d.name + ", [" + addr.str + "]");
}
void ld_global_volatile_u32(const Reg& d, const Operand& addr) {
  emitInst("ld.global.volatile.u32 " + d.name + ", [" + addr.str + "]");
}
void ld_global_cv_v4_u32(const Reg& d0, const Reg& d1, const Reg& d2, const Reg& d3, const Operand& addr) {
  emitInst(
      "ld.global.cv.v4.u32 {" + d0.name + ", " + d1.name + ", " + d2.name + ", " + d3.name + "}, [" + addr.str + "]");
}
void ld_u8(const Reg& d, const Operand& addr) {
  emitInst("ld.u8 " + d.name + ", [" + addr.str + "]");
}

// Store
void st_global_u32(const Operand& addr, const Operand& val) {
  emitInst("st.global.u32 [" + addr.str + "], " + val.str);
}
void st_global_volatile_u32(const Operand& addr, const Operand& val) {
  emitInst("st.global.volatile.u32 [" + addr.str + "], " + val.str);
}
void st_global_wt_v4_u32(const Operand& addr, const Reg& s0, const Reg& s1, const Reg& s2, const Reg& s3) {
  emitInst(
      "st.global.wt.v4.u32 [" + addr.str + "], {" + s0.name + ", " + s1.name + ", " + s2.name + ", " + s3.name + "}");
}
void st_u8(const Operand& addr, const Operand& val) {
  emitInst("st.u8 [" + addr.str + "], " + val.str);
}

// Conversion
void cvt_u64_u32(const Reg& d, const Operand& a) {
  emitInst(fmt2("cvt.u64.u32", d, a));
}
void cvt_u32_u64(const Reg& d, const Operand& a) {
  emitInst(fmt2("cvt.u32.u64", d, a));
}

// Control flow
void bra(const Block* target) {
  emitInst("bra " + target->label);
}
void bra(const Reg& pred, const Block* target) {
  emitInst("@" + pred.name + " bra " + target->label);
}
void bra_not(const Reg& pred, const Block* target) {
  emitInst("@!" + pred.name + " bra " + target->label);
}
void ret() {
  emitInst("ret");
}

// Atomic
void atom_global_inc_u32(const Reg& d, const Operand& addr, const Operand& b) {
  emitInst("atom.global.inc.u32 " + d.name + ", [" + addr.str + "], " + b.str);
}

// Synchronization
void barrier_sync(int n) {
  emitInst("barrier.sync " + std::to_string(n));
}
void membar_sys() {
  emitInst("membar.sys");
}

// Raw emit
void emit(const std::string& inst) {
  emitInst(inst);
}

// ---------------------------------------------------------------------------
// DSL layer
// ---------------------------------------------------------------------------

// TLS state for DSL register allocation
static thread_local std::vector<int> freeRegs[4];
static thread_local int regHighWater[4] = {};
static thread_local int labelCounter = 0;

// --- ValType ---

RegType regTypeFor(ValType type) {
  switch (type) {
  case ValType::Pred:
    return RegType::Pred;
  case ValType::U16:
  case ValType::S16:
    return RegType::B16;
  case ValType::U32:
  case ValType::S32:
    return RegType::B32;
  case ValType::U64:
  case ValType::S64:
    return RegType::B64;
  }
  return RegType::B32;
}

// --- Val ---

[[noreturn]] static void unsupported(const char* op, ValType type) {
  char buf[128];
  snprintf(buf, sizeof(buf), "ptx: unsupported ValType %d for %s", (int)type, op);
  throw std::runtime_error(buf);
}

Val::Val(ValType t) : type(t) {
  RegType rt = regTypeFor(t);
  auto& free = freeRegs[(int)rt];
  int id;
  if (!free.empty()) {
    id = free.back();
    free.pop_back();
  } else {
    id = regHighWater[(int)rt]++;
  }
  reg.type = rt;
  reg.id = id;
  reg.name = std::string(regPrefix(rt)) + std::to_string(id);
  valid = true;
}

Val::~Val() {
  if (valid) {
    freeRegs[(int)reg.type].push_back(reg.id);
  }
}

Val::Val(Val&& o) noexcept : reg(std::move(o.reg)), type(o.type), valid(o.valid) {
  o.valid = false;
}

Val& Val::operator=(Val&& o) noexcept {
  if (this != &o) {
    if (valid) {
      freeRegs[(int)reg.type].push_back(reg.id);
    }
    reg = std::move(o.reg);
    type = o.type;
    valid = o.valid;
    o.valid = false;
  }
  return *this;
}

Val& Val::operator=(int64_t v) {
  Operand imm(v);
  switch (regTypeFor(type)) {
  case RegType::B32:
    mov_u32(reg, imm);
    break;
  case RegType::B64:
    mov_u64(reg, imm);
    break;
  default:
    unsupported("=(imm)", type);
  }
  return *this;
}

// --- Operators ---

Val Val::operator+(const Operand& b) const {
  Val result(type);
  switch (type) {
  case ValType::U32:
    add_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    add_s32(result.reg, reg, b);
    break;
  case ValType::U64:
    add_u64(result.reg, reg, b);
    break;
  case ValType::S64:
    add_s64(result.reg, reg, b);
    break;
  default:
    unsupported("+", type);
  }
  return result;
}

Val Val::operator-(const Operand& b) const {
  Val result(type);
  switch (type) {
  case ValType::U32:
    sub_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    sub_s32(result.reg, reg, b);
    break;
  default:
    unsupported("-", type);
  }
  return result;
}

Val Val::operator*(const Operand& b) const {
  Val result(type);
  switch (type) {
  case ValType::U32:
    mul_lo_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    mul_lo_s32(result.reg, reg, b);
    break;
  case ValType::U64:
    mul_lo_u64(result.reg, reg, b);
    break;
  case ValType::S64:
    mul_lo_s64(result.reg, reg, b);
    break;
  default:
    unsupported("*", type);
  }
  return result;
}

Val Val::operator/(const Operand& b) const {
  Val result(type);
  switch (type) {
  case ValType::U32:
    div_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    div_s32(result.reg, reg, b);
    break;
  default:
    unsupported("/", type);
  }
  return result;
}

Val Val::operator%(const Operand& b) const {
  Val result(type);
  switch (type) {
  case ValType::U32:
    rem_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    rem_s32(result.reg, reg, b);
    break;
  default:
    unsupported("%", type);
  }
  return result;
}

Val Val::operator&(const Operand& b) const {
  Val result(type);
  switch (regTypeFor(type)) {
  case RegType::Pred:
    and_pred(result.reg, reg, b);
    break;
  case RegType::B32:
    and_b32(result.reg, reg, b);
    break;
  case RegType::B64:
    and_b64(result.reg, reg, b);
    break;
  default:
    unsupported("&", type);
  }
  return result;
}

Val Val::operator|(const Operand& b) const {
  Val result(type);
  switch (regTypeFor(type)) {
  case RegType::Pred:
    or_pred(result.reg, reg, b);
    break;
  case RegType::B32:
    or_b32(result.reg, reg, b);
    break;
  case RegType::B64:
    or_b64(result.reg, reg, b);
    break;
  default:
    unsupported("|", type);
  }
  return result;
}

Val Val::operator^(const Operand& b) const {
  Val result(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    xor_b32(result.reg, reg, b);
    break;
  default:
    unsupported("^", type);
  }
  return result;
}

Val Val::operator~() const {
  Val result(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    not_b32(result.reg, reg);
    break;
  default:
    unsupported("~", type);
  }
  return result;
}

Val Val::operator<<(const Operand& b) const {
  Val result(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    shl_b32(result.reg, reg, b);
    break;
  case RegType::B64:
    shl_b64(result.reg, reg, b);
    break;
  default:
    unsupported("<<", type);
  }
  return result;
}

Val Val::operator>>(const Operand& b) const {
  Val result(type);
  switch (type) {
  case ValType::U32:
    shr_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    shr_s32(result.reg, reg, b);
    break;
  default:
    unsupported(">>", type);
  }
  return result;
}

Val Val::operator<(const Operand& b) const {
  Val result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_lt_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    setp_lt_s32(result.reg, reg, b);
    break;
  case ValType::S64:
    setp_lt_s64(result.reg, reg, b);
    break;
  default:
    unsupported("<", type);
  }
  return result;
}

Val Val::operator<=(const Operand& b) const {
  Val result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_le_u32(result.reg, reg, b);
    break;
  default:
    unsupported("<=", type);
  }
  return result;
}

Val Val::operator>(const Operand& b) const {
  Val result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_gt_u32(result.reg, reg, b);
    break;
  default:
    unsupported(">", type);
  }
  return result;
}

Val Val::operator>=(const Operand& b) const {
  Val result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_ge_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    setp_ge_s32(result.reg, reg, b);
    break;
  case ValType::S64:
    setp_ge_s64(result.reg, reg, b);
    break;
  default:
    unsupported(">=", type);
  }
  return result;
}

Val Val::operator==(const Operand& b) const {
  Val result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_eq_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    setp_eq_s32(result.reg, reg, b);
    break;
  default:
    unsupported("==", type);
  }
  return result;
}

Val Val::operator!=(const Operand& b) const {
  Val result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_ne_u32(result.reg, reg, b);
    break;
  case ValType::S32:
    setp_ne_s32(result.reg, reg, b);
    break;
  case ValType::S64:
    setp_ne_s64(result.reg, reg, b);
    break;
  default:
    unsupported("!=", type);
  }
  return result;
}

void Val::operator+=(const Operand& b) {
  switch (type) {
  case ValType::U32:
    add_u32(reg, reg, b);
    break;
  case ValType::S32:
    add_s32(reg, reg, b);
    break;
  case ValType::U64:
    add_u64(reg, reg, b);
    break;
  case ValType::S64:
    add_s64(reg, reg, b);
    break;
  default:
    unsupported("+=", type);
  }
}

void Val::operator-=(const Operand& b) {
  switch (type) {
  case ValType::U32:
    sub_u32(reg, reg, b);
    break;
  case ValType::S32:
    sub_s32(reg, reg, b);
    break;
  default:
    unsupported("-=", type);
  }
}

void Val::operator*=(const Operand& b) {
  switch (type) {
  case ValType::U32:
    mul_lo_u32(reg, reg, b);
    break;
  case ValType::S32:
    mul_lo_s32(reg, reg, b);
    break;
  case ValType::U64:
    mul_lo_u64(reg, reg, b);
    break;
  case ValType::S64:
    mul_lo_s64(reg, reg, b);
    break;
  default:
    unsupported("*=", type);
  }
}

void Val::operator/=(const Operand& b) {
  switch (type) {
  case ValType::U32:
    div_u32(reg, reg, b);
    break;
  case ValType::S32:
    div_s32(reg, reg, b);
    break;
  default:
    unsupported("/=", type);
  }
}

void Val::operator%=(const Operand& b) {
  switch (type) {
  case ValType::U32:
    rem_u32(reg, reg, b);
    break;
  case ValType::S32:
    rem_s32(reg, reg, b);
    break;
  default:
    unsupported("%=", type);
  }
}

// --- FunctionScope ---

FunctionScope::FunctionScope(Function* f) : fn(f) {
  setFunction(f);
  for (int i = 0; i < 4; i++) {
    freeRegs[i].clear();
    regHighWater[i] = 0;
  }
  labelCounter = 0;
}

FunctionScope::~FunctionScope() {
  for (int i = 0; i < 4; i++) {
    fn->regCounts[i] = regHighWater[i];
  }
  setFunction(nullptr);
}

// --- Block activation ---

static std::string genLabel(const char* prefix) {
  return std::string("L_") + prefix + "_" + std::to_string(labelCounter++);
}

static void activateBlock(std::unique_ptr<Block> block) {
  Block* ptr = block.get();
  currentFunction->blocks.push_back(std::move(block));
  setBlock(ptr);
}

Block* activateNewBlock(const char* prefix) {
  auto block = std::make_unique<Block>();
  block->label = genLabel(prefix);
  Block* ptr = block.get();
  currentFunction->blocks.push_back(std::move(block));
  setBlock(ptr);
  return ptr;
}

// --- ScopeGuard ---

ScopeGuard::ScopeGuard(ScopeGuard&& o) noexcept
    : pendingBlock(std::move(o.pendingBlock)), backEdgeTarget(o.backEdgeTarget), closed(o.closed) {
  o.closed = true;
}

ScopeGuard::~ScopeGuard() {
  if (!closed) {
    if (backEdgeTarget) {
      bra(backEdgeTarget);
    }
    if (pendingBlock) {
      activateBlock(std::move(pendingBlock));
    }
  }
}

// --- Control flow ---

ScopeGuard _If(const Reg& pred) {
  ScopeGuard sg;
  // Create skip block (forward reference, not yet in function)
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("endif");
  // Emit conditional branch: skip body when pred is false
  bra_not(pred, sg.pendingBlock.get());
  // Create and activate body block
  activateNewBlock("then");
  return sg;
}

ScopeGuard _Else() {
  ScopeGuard sg;
  // Create endif block (forward reference)
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("endif");
  // Insert unconditional branch to endif at end of previous block (then block)
  auto& blocks = currentFunction->blocks;
  Block* prevBlock = blocks[blocks.size() - 2].get();
  prevBlock->emit("bra " + sg.pendingBlock->label);
  // Current block (the IF's skip target) becomes the else block — already active
  return sg;
}

ScopeGuard _WhileImpl(Block* header, const Reg& pred) {
  ScopeGuard sg;
  // Create exit block (forward reference, not yet in function)
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("endwhile");
  sg.backEdgeTarget = header;
  // Emit conditional branch: exit loop when pred is false
  bra_not(pred, sg.pendingBlock.get());
  // Create and activate body block
  activateNewBlock("while_body");
  return sg;
}

// --- Convenience functions ---

Val threadIdx_x() {
  Val v(ValType::U32);
  mov_u32(v.reg, "%tid.x");
  return v;
}

Val blockIdx_x() {
  Val v(ValType::U32);
  mov_u32(v.reg, "%ctaid.x");
  return v;
}

Val blockDim_x() {
  Val v(ValType::U32);
  mov_u32(v.reg, "%ntid.x");
  return v;
}

Val loadParam(int index, ValType type) {
  Val v(type);
  std::string paramName = currentFunction->param(index);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_param_u32(v.reg, paramName);
    break;
  case RegType::B64:
    ld_param_u64(v.reg, paramName);
    break;
  default:
    unsupported("loadParam", type);
  }
  return v;
}

Val paramBase(int index) {
  Val v(ValType::U64);
  mov_b64(v.reg, Operand(currentFunction->param(index).c_str()));
  return v;
}

Val loadParamField(const Val& base, int offset, ValType type) {
  Val v(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_param_u32(v.reg, base.reg, offset);
    break;
  case RegType::B64:
    ld_param_u64(v.reg, base.reg, offset);
    break;
  default:
    unsupported("loadParamField", type);
  }
  return v;
}

Val widen(const Val& v) {
  ValType wide;
  switch (v.type) {
  case ValType::U32:
    wide = ValType::U64;
    break;
  case ValType::S32:
    wide = ValType::S64;
    break;
  default:
    unsupported("widen", v.type);
  }
  Val result(wide);
  cvt_u64_u32(result.reg, v.reg);
  return result;
}

void storeGlobal(const Val& addr, const Val& val) {
  switch (regTypeFor(val.type)) {
  case RegType::B32:
    st_global_u32(addr.reg, val.reg);
    break;
  default:
    unsupported("storeGlobal", val.type);
  }
}

Val loadGlobalVolatile(const Val& addr, ValType type) {
  Val v(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_global_volatile_u32(v.reg, addr.reg);
    break;
  default:
    unsupported("loadGlobalVolatile", type);
  }
  return v;
}

void storeGlobalVolatile(const Val& addr, const Val& val) {
  switch (regTypeFor(val.type)) {
  case RegType::B32:
    st_global_volatile_u32(addr.reg, val.reg);
    break;
  default:
    unsupported("storeGlobalVolatile", val.type);
  }
}

void ldcv_v4(Val& v0, Val& v1, Val& v2, Val& v3, const Val& addr) {
  v0 = Val(ValType::U32);
  v1 = Val(ValType::U32);
  v2 = Val(ValType::U32);
  v3 = Val(ValType::U32);
  ld_global_cv_v4_u32(v0, v1, v2, v3, addr.reg);
}

void stwt_v4(const Val& addr, const Val& v0, const Val& v1, const Val& v2, const Val& v3) {
  st_global_wt_v4_u32(addr.reg, v0, v1, v2, v3);
}

Val atomicInc(const Val& addr, const Operand& modulo) {
  Val result(ValType::U32);
  atom_global_inc_u32(result.reg, addr.reg, modulo);
  return result;
}

// ---------------------------------------------------------------------------
// ptxTest
// ---------------------------------------------------------------------------

std::string ptxTest(const char* target) {
  Module mod;
  mod.target = target;
  setModule(&mod);

  // --- Test 1: simple tid write ---
  auto* fn = mod.newFunction("test_write_tid");
  {
    FunctionScope fnScope(fn);
    fn->maxThreads = 256;
    fn->addParam(".u64"); // output pointer
    fn->addParam(".u32"); // count

    activateNewBlock("entry");

    auto outPtr = loadParam(0, ValType::U64);
    auto count = loadParam(1, ValType::U32);
    auto tid = threadIdx_x();

    IF(tid < count) {
      auto addr = outPtr + widen(tid) * 4;
      storeGlobal(addr, tid);
    }

    ret();
  }

  // --- Test 2: vectorized copy (4 x u32 per thread) ---
  auto* fn2 = mod.newFunction("test_copy_v4");
  {
    FunctionScope fnScope(fn2);
    fn2->maxThreads = 256;
    fn2->addParam(".u64"); // dst pointer
    fn2->addParam(".u64"); // src pointer
    fn2->addParam(".u32"); // count (number of u32 elements)

    activateNewBlock("entry");

    auto dst = loadParam(0, ValType::U64);
    auto src = loadParam(1, ValType::U64);
    auto count = loadParam(2, ValType::U32);

    // Global thread ID
    auto tid = threadIdx_x();
    auto bid = blockIdx_x();
    auto bdim = blockDim_x();
    auto gid = bid * bdim + tid;

    // Each thread handles 4 elements
    auto base = gid * 4;

    IF(base < count) {
      // Byte offset for v4 u32 (16 bytes per group)
      auto byteOff = widen(base) * 4;
      auto srcAddr = src + byteOff;
      auto dstAddr = dst + byteOff;

      // Load 4 x u32 from src
      Val v0(ValType::U32);
      Val v1(ValType::U32);
      Val v2(ValType::U32);
      Val v3(ValType::U32);
      ld_global_cv_v4_u32(v0, v1, v2, v3, srcAddr);

      // Store 4 x u32 to dst
      st_global_wt_v4_u32(dstAddr, v0, v1, v2, v3);
    }
    ELSE {
      // Out of bounds — write zeros to dst
      auto byteOff = widen(base) * 4;
      auto dstAddr = dst + byteOff;
      Val zero(ValType::U32);
      zero = 0;
      st_global_wt_v4_u32(dstAddr, zero, zero, zero, zero);
    }

    // Second bounds check — flush after copy
    IF(gid == 0) {
      membar_sys();
    }

    ret();
  }

  // --- Test 3: strided loop (WHILE test) ---
  auto* fn3 = mod.newFunction("test_strided_copy");
  {
    FunctionScope fnScope(fn3);
    fn3->maxThreads = 256;
    fn3->addParam(".u64"); // dst pointer
    fn3->addParam(".u64"); // src pointer
    fn3->addParam(".u32"); // count (number of u32 elements)

    activateNewBlock("entry");

    auto dst = loadParam(0, ValType::U64);
    auto src = loadParam(1, ValType::U64);
    auto count = loadParam(2, ValType::U32);

    auto tid = threadIdx_x();
    auto bdim = blockDim_x();
    auto bid = blockIdx_x();
    auto i = bid * bdim + tid;

    WHILE(i < count) {
      auto byteOff = widen(i) * 4;
      auto srcAddr = src + byteOff;
      auto dstAddr = dst + byteOff;
      auto val = Val(ValType::U32);
      ld_global_u32(val, srcAddr);
      st_global_u32(dstAddr, val);
      i += bdim;
    }

    ret();
  }

  // --- Test 4: struct param loading ---
  // Mimics loading from CompileOpCopyParameters:
  //   offset 0: u32 stepValue
  //   offset 4: u32 concurrencyIndex
  //   offset 8: u32 numDescriptors
  //   offset 16: CopyDescriptor descriptors[] (each 24 bytes: u64 src, u64 dst, u32 bytes)
  auto* fn4 = mod.newFunction("test_struct_param");
  {
    FunctionScope fnScope(fn4);
    fn4->maxThreads = 256;
    fn4->addParamBytes(8, 4816); // .param .align 8 .b8 param[4816]

    activateNewBlock("entry");

    auto params = paramBase(0);
    auto numDesc = loadParamField(params, 8, ValType::U32);
    auto tid = threadIdx_x();

    // Load first descriptor's src and dst
    auto descBase = params + 16; // offset of descriptors[0]
    auto src = loadParamField(descBase, 0, ValType::U64);
    auto dst = loadParamField(descBase, 8, ValType::U64);
    auto bytes = loadParamField(descBase, 16, ValType::U32);

    // Simple copy: each thread copies one u32
    IF(tid < bytes) {
      auto off = widen(tid);
      auto srcAddr = src + off;
      auto dstAddr = dst + off;
      Val v(ValType::U32);
      ld_global_u32(v, srcAddr);
      st_global_u32(dstAddr, v);
    }

    ret();
  }

  std::string ptx = mod.finalize();
  printf("=== PTX Test Output ===\n%s\n", ptx.c_str());
  return ptx;
}

} // namespace ptx
} // namespace moodist
