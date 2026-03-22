// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ptx.h"

#include <cassert>
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
static thread_local std::string currentPred;

void setModule(Module* m) {
  currentModule = m;
}
void setFunction(Function* f) {
  currentFunction = f;
}
Function* getFunction() {
  return currentFunction;
}
Value addShared(int align, int sizeBytes, const char* suffix) {
  std::string sym = currentFunction->addShared(align, sizeBytes, suffix);
  return globalAddr(sym.c_str());
}
u64 addGlobalVar(int align, int sizeBytes, const char* suffix) {
  static int counter = 0;
  std::string name = std::string("global_") + (suffix ? suffix : std::to_string(counter++));
  std::string decl = ".global .align " + std::to_string(align) + " .b8 " + name + "[" + std::to_string(sizeBytes) + "]";
  currentModule->globals.push_back(decl);
  return globalAddr(name.c_str());
}
Value addConst32(int align, const void* data, size_t size, const char* name) {
  currentModule->addConst(align, data, size, name);
  return constAddr(name);
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

std::string Function::addShared(int align, int sizeBytes, const char* suffix) {
  int index = (int)sharedDecls.size();
  std::string sym = name + "_shared_" + (suffix ? suffix : std::to_string(index));
  sharedDecls.push_back(
      ".shared .align " + std::to_string(align) + " .b8 " + sym + "[" + std::to_string(sizeBytes) + "]");
  return sym;
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

  // Shared memory declarations
  for (const auto& decl : sharedDecls) {
    s += "    " + decl + ";\n";
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

std::string Module::addGlobal(const char* type, int count, const char* name) {
  std::string decl = ".global " + std::string(type) + " " + name;
  if (count > 1) {
    decl += "[" + std::to_string(count) + "]";
  }
  globals.push_back(decl);
  return name;
}

std::string Module::addConst(int align, const void* data, size_t size, const char* name) {
  // Emit: .const .align A .b8 name[SIZE] = {b0, b1, b2, ...};
  std::string decl = ".const .align " + std::to_string(align) + " .b8 " + name + "[" + std::to_string(size) + "] = {";
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; i++) {
    if (i > 0) {
      decl += ',';
    }
    decl += std::to_string(bytes[i]);
  }
  decl += "}";
  globals.push_back(decl);
  return name;
}

std::string Module::finalize() const {
  std::string s;
  s += ".version " + version + "\n";
  s += ".target " + target + "\n";
  s += ".address_size " + std::to_string(addressSize) + "\n";
  s += "\n";

  for (auto& c : topLevelComments) {
    s += "// " + c + "\n";
  }
  s += "\n";

  for (const auto& g : globals) {
    s += g + ";\n";
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
  currentBlock->emit(currentPred + inst);
}

static std::string fmt3(const char* op, const Value& d, const Value& a, const Value& b) {
  return std::string(op) + " " + d.str() + ", " + a.str() + ", " + b.str();
}

static std::string fmt2(const char* op, const Value& d, const Value& a) {
  return std::string(op) + " " + d.str() + ", " + a.str();
}

// ---------------------------------------------------------------------------
// Instruction helpers
// ---------------------------------------------------------------------------

// Arithmetic
void add_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("add.u32", d, a, b));
}
void add_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("add.s32", d, a, b));
}
void add_u64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("add.u64", d, a, b));
}
void add_s64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("add.s64", d, a, b));
}
void sub_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("sub.u32", d, a, b));
}
void sub_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("sub.s32", d, a, b));
}
void sub_u64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("sub.u64", d, a, b));
}
void mul_lo_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("mul.lo.u32", d, a, b));
}
void mul_lo_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("mul.lo.s32", d, a, b));
}
void mul_lo_u64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("mul.lo.u64", d, a, b));
}
void mul_lo_s64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("mul.lo.s64", d, a, b));
}
void mul_wide_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("mul.wide.u32", d, a, b));
}
void mul_wide_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("mul.wide.s32", d, a, b));
}
void div_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("div.u32", d, a, b));
}
void div_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("div.s32", d, a, b));
}
void rem_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("rem.u32", d, a, b));
}
void rem_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("rem.s32", d, a, b));
}
void rem_u64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("rem.u64", d, a, b));
}
void min_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("min.u32", d, a, b));
}
void min_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("min.s32", d, a, b));
}
void max_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("max.u32", d, a, b));
}
void max_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("max.s32", d, a, b));
}

// Bitwise
void and_pred(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("and.pred", d, a, b));
}
void or_pred(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("or.pred", d, a, b));
}
void and_b32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("and.b32", d, a, b));
}
void and_b64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("and.b64", d, a, b));
}
void or_b32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("or.b32", d, a, b));
}
void or_b64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("or.b64", d, a, b));
}
void xor_b32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("xor.b32", d, a, b));
}
void not_b32(const Value& d, const Value& a) {
  emitInst(fmt2("not.b32", d, a));
}
void shl_b32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("shl.b32", d, a, b));
}
void shl_b64(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("shl.b64", d, a, b));
}
void shr_u32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("shr.u32", d, a, b));
}
void shr_s32(const Value& d, const Value& a, const Value& b) {
  emitInst(fmt3("shr.s32", d, a, b));
}

void neg_s32(const Value& d, const Value& a) {
  emitInst(fmt2("neg.s32", d, a));
}

// Comparison
void setp_eq_u32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.eq.u32", p, a, b));
}
void setp_ne_u32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ne.u32", p, a, b));
}
void setp_lt_u32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.lt.u32", p, a, b));
}
void setp_le_u32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.le.u32", p, a, b));
}
void setp_gt_u32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.gt.u32", p, a, b));
}
void setp_ge_u32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ge.u32", p, a, b));
}
void setp_eq_s32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.eq.s32", p, a, b));
}
void setp_ne_s32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ne.s32", p, a, b));
}
void setp_lt_s32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.lt.s32", p, a, b));
}
void setp_ge_s32(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ge.s32", p, a, b));
}
void setp_ne_s64(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ne.s64", p, a, b));
}
void setp_eq_u64(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.eq.u64", p, a, b));
}
void setp_ne_u64(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ne.u64", p, a, b));
}
void setp_lt_s64(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.lt.s64", p, a, b));
}
void setp_ge_s64(const Value& p, const Value& a, const Value& b) {
  emitInst(fmt3("setp.ge.s64", p, a, b));
}

// Move
void mov_u32(const Value& d, const Value& a) {
  emitInst(fmt2("mov.u32", d, a));
}
void mov_u64(const Value& d, const Value& a) {
  emitInst(fmt2("mov.u64", d, a));
}
void mov_b64(const Value& d, const Value& a) {
  emitInst(fmt2("mov.b64", d, a));
}

// Load
void ld_param_u32(const Value& d, const std::string& paramName) {
  emitInst("ld.param.u32 " + d.str() + ", [" + paramName + "]");
}
void ld_param_u64(const Value& d, const std::string& paramName) {
  emitInst("ld.param.u64 " + d.str() + ", [" + paramName + "]");
}
void ld_param_u32(const Value& d, const Value& addr, int offset) {
  emitInst("ld.param.u32 " + d.str() + ", [" + addr.str() + "+" + std::to_string(offset) + "]");
}
void ld_param_u64(const Value& d, const Value& addr, int offset) {
  emitInst("ld.param.u64 " + d.str() + ", [" + addr.str() + "+" + std::to_string(offset) + "]");
}
void ld_global_u32(const Value& d, const Value& addr) {
  emitInst("ld.global.u32 " + d.str() + ", [" + addr.str() + "]");
}
void ld_global_u64(const Value& d, const Value& addr) {
  emitInst("ld.global.u64 " + d.str() + ", [" + addr.str() + "]");
}
void ld_const_u32(const Value& d, const Value& addr) {
  emitInst("ld.const.u32 " + d.str() + ", [" + addr.str() + "]");
}
void ld_const_u64(const Value& d, const Value& addr) {
  emitInst("ld.const.u64 " + d.str() + ", [" + addr.str() + "]");
}
void ld_const_u8(const Value& d, const Value& addr) {
  emitInst("ld.const.u8 " + d.str() + ", [" + addr.str() + "]");
}
void ld_global_volatile_u32(const Value& d, const Value& addr) {
  emitInst("ld.global.volatile.u32 " + d.str() + ", [" + addr.str() + "]");
}
void ld_global_cv_v4_u32(const Value& d0, const Value& d1, const Value& d2, const Value& d3, const Value& addr) {
  emitInst("ld.global.cv.v4.u32 {" + d0.str() + ", " + d1.str() + ", " + d2.str() + ", " + d3.str() + "}, [" +
           addr.str() + "]");
}
void ld_global_cv_v4_u32(std::array<Value, 4>& v, const Value& addr) {
  ld_global_cv_v4_u32(v[0], v[1], v[2], v[3], addr);
}
void ld_shared_v4_u32(std::array<Value, 4>& v, const Value& addr) {
  v[0] = Value(ValType::U32);
  v[1] = Value(ValType::U32);
  v[2] = Value(ValType::U32);
  v[3] = Value(ValType::U32);
  emitInst("ld.shared.v4.u32 {" + v[0].str() + ", " + v[1].str() + ", " + v[2].str() + ", " + v[3].str() + "}, [" +
           addr.str() + "]");
}
void st_shared_v4_u32(const Value& addr, const std::array<Value, 4>& v) {
  emitInst("st.shared.v4.u32 [" + addr.str() + "], {" + v[0].str() + ", " + v[1].str() + ", " + v[2].str() + ", " +
           v[3].str() + "}");
}
void ld_global_nc_v4_u32(const Value& d0, const Value& d1, const Value& d2, const Value& d3, const Value& addr) {
  emitInst("ld.global.nc.v4.u32 {" + d0.str() + ", " + d1.str() + ", " + d2.str() + ", " + d3.str() + "}, [" +
           addr.str() + "]");
}
void ld_global_nc_v4_u32(std::array<Value, 4>& v, const Value& addr) {
  ld_global_nc_v4_u32(v[0], v[1], v[2], v[3], addr);
}
void ld_global_cs_v4_u32(const Value& d0, const Value& d1, const Value& d2, const Value& d3, const Value& addr) {
  emitInst("ld.global.cs.v4.u32 {" + d0.str() + ", " + d1.str() + ", " + d2.str() + ", " + d3.str() + "}, [" +
           addr.str() + "]");
}
void ld_global_cs_v4_u32(std::array<Value, 4>& v, const Value& addr) {
  ld_global_cs_v4_u32(v[0], v[1], v[2], v[3], addr);
}
void ld_v4_u32(const Value& d0, const Value& d1, const Value& d2, const Value& d3, const Value& addr) {
  emitInst("ld.v4.u32 {" + d0.str() + ", " + d1.str() + ", " + d2.str() + ", " + d3.str() + "}, [" + addr.str() + "]");
}
void ld_v4_u32(std::array<Value, 4>& v, const Value& addr) {
  ld_v4_u32(v[0], v[1], v[2], v[3], addr);
}
void ld_u8(const Value& d, const Value& addr) {
  emitInst("ld.u8 " + d.str() + ", [" + addr.str() + "]");
}

u64 ld_global_cv_u64(const u64& addr) {
  u64 r;
  emitInst("ld.global.cv.u64 " + r.inner.str() + ", [" + addr.inner.str() + "]");
  return r;
}
void st_global_wt_u64(const u64& addr, const u64& value) {
  emitInst("st.global.wt.u64 [" + addr.inner.str() + "], " + value.inner.str());
}

// Store
void st_global_u32(const Value& addr, const Value& val) {
  emitInst("st.global.u32 [" + addr.str() + "], " + val.str());
}
void st_global_u64(const Value& addr, const Value& val) {
  emitInst("st.global.u64 [" + addr.str() + "], " + val.str());
}
void st_global_volatile_u32(const Value& addr, const Value& val) {
  emitInst("st.global.volatile.u32 [" + addr.str() + "], " + val.str());
}
void ld_global_relaxed_sys_u32(const Value& d, const Value& addr) {
  emitInst("ld.relaxed.sys.global.u32 " + d.str() + ", [" + addr.str() + "]");
}
void ld_global_relaxed_sys_u64(const Value& d, const Value& addr) {
  emitInst("ld.relaxed.sys.global.u64 " + d.str() + ", [" + addr.str() + "]");
}
void ld_global_acquire_sys_u32(const Value& d, const Value& addr) {
  emitInst("ld.acquire.sys.global.u32 " + d.str() + ", [" + addr.str() + "]");
}
void fence_proxy() {
  emitInst("fence.proxy.async");
}
void fence() {
  emitInst("fence.cta");
}
void fence_acquire_sys() {
  emitInst("fence.acquire.sys");
}
void fence_release_sys() {
  emitInst("fence.release.sys");
}
void fence_acquire_gpu() {
  emitInst("fence.acquire.gpu");
}
void fence_release_gpu() {
  emitInst("fence.release.gpu");
}
void st_global_relaxed_sys_u32(const Value& addr, const Value& val) {
  emitInst("st.relaxed.sys.global.u32 [" + addr.str() + "], " + val.str());
}
void st_global_relaxed_sys_u64(const Value& addr, const Value& val) {
  emitInst("st.relaxed.sys.global.u64 [" + addr.str() + "], " + val.str());
}
void st_global_release_sys_u32(const Value& addr, const Value& val) {
  emitInst("st.release.sys.global.u32 [" + addr.str() + "], " + val.str());
}
void ld_shared_acquire_cta_u32(const Value& d, const Value& addr) {
  emitInst("ld.acquire.cta.shared.u32 " + d.str() + ", [" + addr.str() + "]");
}
void st_shared_release_cta_u32(const Value& addr, const Value& val) {
  emitInst("st.release.cta.shared.u32 [" + addr.str() + "], " + val.str());
}
void st_global_wt_v4_u32(const Value& addr, const Value& s0, const Value& s1, const Value& s2, const Value& s3) {
  emitInst("st.global.wt.v4.u32 [" + addr.str() + "], {" + s0.str() + ", " + s1.str() + ", " + s2.str() + ", " +
           s3.str() + "}");
}
void st_global_wt_v4_u32(const Value& addr, const std::array<Value, 4>& v) {
  st_global_wt_v4_u32(addr, v[0], v[1], v[2], v[3]);
}
void st_u8(const Value& addr, const Value& val) {
  emitInst("st.u8 [" + addr.str() + "], " + val.str());
}

// Conversion
void cvt_u64_u32(const Value& d, const Value& a) {
  emitInst(fmt2("cvt.u64.u32", d, a));
}
void cvt_u32_u64(const Value& d, const Value& a) {
  emitInst(fmt2("cvt.u32.u64", d, a));
}

// Control flow
void bra(const Block* target) {
  emitInst("bra.uni " + target->label);
}
void bra(const Value& pred, const Block* target) {
  emitInst("@" + pred.str() + " bra.uni " + target->label);
}
void bra_div(const Value& pred, const Block* target) {
  emitInst("@" + pred.str() + " bra " + target->label);
}
void bra_not(const Value& pred, const Block* target) {
  emitInst("@!" + pred.str() + " bra.uni " + target->label);
}
void brx_idx(const Value& index, const std::vector<Label>& targets, bool divergent) {
  std::string targetList;
  for (const auto& t : targets) {
    if (!targetList.empty()) {
      targetList += ", ";
    }
    targetList += t.rawPtr->label;
  }
  std::string tsLabel = "ts_" + std::to_string(currentFunction->brxCounter++);
  emitInst(tsLabel + ": .branchtargets " + targetList);
  if (divergent) {
    emitInst("brx.idx " + index.str() + ", " + tsLabel);
  } else {
    emitInst("brx.idx.uni " + index.str() + ", " + tsLabel);
  }
}
void ret() {
  emitInst("ret");
}

// Atomic
void atom_global_inc_u32(const Value& d, const Value& addr, const Value& b) {
  emitInst("atom.global.inc.u32 " + d.str() + ", [" + addr.str() + "], " + b.str());
}
void atom_global_relaxed_inc_u32(const Value& d, const Value& addr, const Value& b) {
  emitInst("atom.relaxed.global.inc.u32 " + d.str() + ", [" + addr.str() + "], " + b.str());
}

static Value atomAdd(const char* space, const char* sem, const char* scope, const Value& addr, const Value& b) {
  Value d(ValType::U32);
  emitInst(std::string("atom.") + sem + "." + scope + "." + space + ".add.u32 " + d.str() + ", [" + addr.str() + "], " +
           b.str());
  return d;
}
Value atom_global_relaxed_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "relaxed", "cta", addr, b);
}
Value atom_global_relaxed_gpu_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "relaxed", "gpu", addr, b);
}
Value atom_global_relaxed_sys_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "relaxed", "sys", addr, b);
}
Value atom_global_acquire_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "acquire", "cta", addr, b);
}
Value atom_global_acquire_gpu_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "acquire", "gpu", addr, b);
}
Value atom_global_acquire_sys_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "acquire", "sys", addr, b);
}
Value atom_global_release_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "release", "cta", addr, b);
}
Value atom_global_release_gpu_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "release", "gpu", addr, b);
}
Value atom_global_release_sys_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "release", "sys", addr, b);
}
Value atom_global_acq_rel_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "acq_rel", "cta", addr, b);
}
Value atom_global_acq_rel_gpu_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "acq_rel", "gpu", addr, b);
}
Value atom_global_acq_rel_sys_add_u32(const Value& addr, const Value& b) {
  return atomAdd("global", "acq_rel", "sys", addr, b);
}
Value atom_shared_relaxed_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("shared::cta", "relaxed", "cta", addr, b);
}
Value atom_shared_acquire_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("shared::cta", "acquire", "cta", addr, b);
}
Value atom_shared_release_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("shared::cta", "release", "cta", addr, b);
}
Value atom_shared_acq_rel_cta_add_u32(const Value& addr, const Value& b) {
  return atomAdd("shared::cta", "acq_rel", "cta", addr, b);
}

// Synchronization
void barrier_sync(int n) {
  emitInst("barrier.sync " + std::to_string(n));
}
void membar_sys() {
  emitInst("membar.sys");
}
void warp_sync(uint32_t membermask) {
  char buf[16];
  snprintf(buf, sizeof(buf), "0x%08x", membermask);
  emitInst(std::string("bar.warp.sync ") + buf);
}

void trap() {
  emitInst("trap");
}

// mbarrier operations
void mbarrier_init(const Value& addr, int count) {
  emitInst("mbarrier.init.shared::cta.b64 [" + addr.str() + "], " + std::to_string(count));
}
void mbarrier_inval(const Value& addr) {
  emitInst("mbarrier.inval.shared::cta.b64 [" + addr.str() + "]");
}

Value mbarrier_arrive(const Value& addr) {
  Value state(ValType::U64);
  emitInst("mbarrier.arrive.shared::cta.b64 " + state.str() + ", [" + addr.str() + "]");
  return state;
}

Value mbarrier_arrive_noComplete(const Value& addr) {
  Value state(ValType::U64);
  emitInst("mbarrier.arrive.noComplete.shared::cta.b64 " + state.str() + ", [" + addr.str() + "], 1");
  return state;
}

void mbarrier_expect_tx(const Value& addr, const Value& txCount) {
  emitInst("mbarrier.expect_tx.shared::cta.b64 [" + addr.str() + "], " + txCount.str());
}

Value mbarrier_try_wait_parity(const Value& addr, const Value& phaseParity) {
  Value result(ValType::Pred);
  emitInst("mbarrier.try_wait.parity.shared::cta.b64 " + result.str() + ", [" + addr.str() + "], " + phaseParity.str());
  return result;
}

// cp.async.bulk operations
void cp_async_bulk_shared_global(const Value& dst, const Value& src, const Value& size, const Value& mbar) {
  emitInst("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [" + dst.str() + "], [" + src.str() + "], " +
           size.str() + ", [" + mbar.str() + "]");
}

void cp_async_bulk_global_shared(const Value& dst, const Value& src, const Value& size) {
  emitInst("cp.async.bulk.global.shared::cta.bulk_group [" + dst.str() + "], [" + src.str() + "], " + size.str());
}

void multimem_cp_async_bulk_global_shared(const Value& dst, const Value& src, const Value& size) {
  emitInst(
      "multimem.cp.async.bulk.global.shared::cta.bulk_group [" + dst.str() + "], [" + src.str() + "], " + size.str());
}

void cp_async_bulk_prefetch_l2(const Value& src, const Value& size) {
  emitInst("cp.async.bulk.prefetch.L2.global [" + src.str() + "], " + size.str());
}

void cp_async_bulk_commit_group() {
  emitInst("cp.async.bulk.commit_group");
}

void cp_async_bulk_wait_group(int n) {
  emitInst("cp.async.bulk.wait_group " + std::to_string(n));
}
void cp_async_bulk_wait_group_read(int n) {
  emitInst("cp.async.bulk.wait_group.read " + std::to_string(n));
}

void bar_sync(int barrierId, int threadCount) {
  emitInst("bar.sync " + std::to_string(barrierId) + ", " + std::to_string(threadCount));
}

void bar_sync(const Value& barrierId, int threadCount) {
  emitInst("bar.sync " + barrierId.str() + ", " + std::to_string(threadCount));
}

static Value barRed(const char* op, const std::string& barrier, int threadCount, const Value& pred) {
  Value d(ValType::U32);
  emitInst("bar.red." + std::string(op) + ".u32 " + d.str() + ", " + barrier + ", " + std::to_string(threadCount) +
           ", " + pred.str());
  return d;
}
static Value barRedPred(const char* op, const std::string& barrier, int threadCount, const Value& pred) {
  Value d(ValType::Pred);
  emitInst("bar.red." + std::string(op) + ".pred " + d.str() + ", " + barrier + ", " + std::to_string(threadCount) +
           ", " + pred.str());
  return d;
}
Value bar_red_popc(int barrierId, int threadCount, const Value& pred) {
  return barRed("popc", std::to_string(barrierId), threadCount, pred);
}
Value bar_red_popc(const Value& barrierId, int threadCount, const Value& pred) {
  return barRed("popc", barrierId.str(), threadCount, pred);
}
Value bar_red_and(int barrierId, int threadCount, const Value& pred) {
  return barRedPred("and", std::to_string(barrierId), threadCount, pred);
}
Value bar_red_and(const Value& barrierId, int threadCount, const Value& pred) {
  return barRedPred("and", barrierId.str(), threadCount, pred);
}
Value bar_red_or(int barrierId, int threadCount, const Value& pred) {
  return barRedPred("or", std::to_string(barrierId), threadCount, pred);
}
Value bar_red_or(const Value& barrierId, int threadCount, const Value& pred) {
  return barRedPred("or", barrierId.str(), threadCount, pred);
}

static Value shflSync(const char* mode, const Value& a, const Value& b, uint32_t c, uint32_t membermask) {
  Value d(ValType::U32);
  char buf[64];
  snprintf(buf, sizeof(buf), "shfl.sync.%s.b32 %s, %s, %s, 0x%x, 0x%x", mode, d.str().c_str(), a.str().c_str(),
      b.str().c_str(), c, membermask);
  emitInst(buf);
  return d;
}
Value shfl_sync_idx_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask) {
  return shflSync("idx", a, b, c, membermask);
}
Value shfl_sync_up_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask) {
  return shflSync("up", a, b, c, membermask);
}
Value shfl_sync_down_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask) {
  return shflSync("down", a, b, c, membermask);
}
Value shfl_sync_bfly_b32(const Value& a, const Value& b, uint32_t c, uint32_t membermask) {
  return shflSync("bfly", a, b, c, membermask);
}

void bar_arrive(int barrierId, int threadCount) {
  emitInst("bar.arrive " + std::to_string(barrierId) + ", " + std::to_string(threadCount));
}

void bar_arrive(const Value& barrierId, int threadCount) {
  emitInst("bar.arrive " + barrierId.str() + ", " + std::to_string(threadCount));
}

// Shared memory address conversion
Value cvta_shared(const Value& sharedAddr) {
  Value result(ValType::U64);
  emitInst("cvta.shared.u64 " + result.str() + ", " + sharedAddr.str());
  return result;
}

// Raw emit
void emit(const std::string& inst) {
  emitInst(inst);
}

// ---------------------------------------------------------------------------
// DSL layer
// ---------------------------------------------------------------------------

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

// --- Value ---

[[noreturn]] static void unsupported(const char* op, ValType type) {
  char buf[128];
  snprintf(buf, sizeof(buf), "ptx: unsupported ValType %d for %s", (int)type, op);
  throw std::runtime_error(buf);
}

Value::Value(ValType t) : type(t), kind(Register) {
  RegType rt = regTypeFor(t);
  // auto& free = currentFunction->freeRegs[(int)rt];
  // int id;
  // if (!free.empty() && false) {
  //   id = free.back();
  //   free.pop_back();
  // } else {
  //   id = currentFunction->regHighWater[(int)rt]++;
  // }
  int id = currentFunction->regHighWater[(int)rt]++;
  reg.type = rt;
  reg.id = id;
  reg.name = std::string(regPrefix(rt)) + std::to_string(id);
  valid = true;
}

Value::Value(ValType type, std::string s) : type(type), kind(Immediate), immStr(s), valid(true) {}

Value::Value(const Reg& r, ValType t) : reg(r), type(t), valid(true), kind(Register) {}

Value Value::imm(ValType type, int64_t v) {
  Value r;
  r.type = type;
  r.kind = Immediate;
  r.immValue = v;
  r.immStr = std::to_string(v);
  r.valid = true;
  return r;
}

const std::string& Value::str() const {
  if (kind == Register) {
    return reg.name;
  }
  return immStr;
}

Value::~Value() {
  if (valid && kind == Register) {
    // currentFunction->freeRegs[(int)reg.type].push_back(reg.id);
  }
}

Value::Value(Value&& o)
    : reg(std::move(o.reg)), type(o.type), valid(o.valid), kind(o.kind), immStr(std::move(o.immStr)) {
  o.valid = false;
  o.kind = None;
}

Value& Value::operator=(Value&& o) {
  if (this != &o) {
    if (valid && kind == Register && o.valid) {
      // Already has a register in emitted PTX — emit mov to preserve it
      if (regTypeFor(type) == regTypeFor(o.type)) {
        switch (regTypeFor(type)) {
        case RegType::B32:
          mov_u32(*this, o);
          break;
        case RegType::B64:
          mov_u64(*this, o);
          break;
        default:
          break;
        }
      } else {
        throw std::runtime_error("bad assignment");
      }
      // Free the source register
      // currentFunction->freeRegs[(int)o.reg.type].push_back(o.reg.id);
      o.valid = false;
      o.kind = None;
    } else {
      // // Uninitialized destination — just transfer ownership
      // if (valid && kind == Register) {
      //   currentFunction->freeRegs[(int)reg.type].push_back(reg.id);
      // }
      reg = std::move(o.reg);
      type = o.type;
      valid = o.valid;
      kind = o.kind;
      immStr = std::move(o.immStr);
      o.valid = false;
      o.kind = None;
    }
  }
  return *this;
}

Value& Value::operator=(int64_t v) {
  switch (regTypeFor(type)) {
  case RegType::Pred:
    emitInst("setp.ne.u32 " + str() + ", " + std::to_string(v) + ", 0");
    break;
  case RegType::B32:
    mov_u32(*this, Value::imm(type, v));
    break;
  case RegType::B64:
    mov_u64(*this, Value::imm(type, v));
    break;
  default:
    unsupported("=(imm)", type);
  }
  return *this;
}

Value& Value::operator=(const Value& o) {
  if (!valid || kind != Register) {
    // Uninitialized: allocate a register matching the source type
    *this = Value(o.type);
  }
  if (o.kind == Register) {
    if (regTypeFor(type) != regTypeFor(o.type)) {
      unsupported("=(Value) type mismatch", type);
    }
    switch (regTypeFor(type)) {
    case RegType::Pred:
      emitInst("mov.pred " + str() + ", " + o.str());
      break;
    case RegType::B32:
      mov_u32(*this, o);
      break;
    case RegType::B64:
      mov_u64(*this, o);
      break;
    default:
      unsupported("=(Value)", type);
    }
  } else {
    // Source is an immediate — emit mov
    switch (regTypeFor(type)) {
    case RegType::Pred:
      emitInst("setp.ne.u32 " + str() + ", " + o.immStr + ", 0");
      break;
    case RegType::B32:
      mov_u32(*this, o);
      break;
    case RegType::B64:
      mov_u64(*this, o);
      break;
    default:
      unsupported("=(Value imm)", type);
    }
  }
  return *this;
}

Value::operator const Reg&() const {
  if (kind != Register) {
    throw std::runtime_error("ptx: Value is not a register (cannot use as destination)");
  }
  return reg;
}

// --- Operators ---

Value Value::operator+(const Value& b) const {
  Value result(type);
  switch (type) {
  case ValType::U32:
    add_u32(result, *this, b);
    break;
  case ValType::S32:
    add_s32(result, *this, b);
    break;
  case ValType::U64:
    add_u64(result, *this, b);
    break;
  case ValType::S64:
    add_s64(result, *this, b);
    break;
  default:
    unsupported("+", type);
  }
  return result;
}

Value Value::operator-(const Value& b) const {
  Value result(type);
  switch (type) {
  case ValType::U32:
    sub_u32(result, *this, b);
    break;
  case ValType::S32:
    sub_s32(result, *this, b);
    break;
  case ValType::U64:
    sub_u64(result, *this, b);
    break;
  default:
    unsupported("-", type);
  }
  return result;
}

Value Value::operator*(const Value& b) const {
  Value result(type);
  switch (type) {
  case ValType::U32:
    mul_lo_u32(result, *this, b);
    break;
  case ValType::S32:
    mul_lo_s32(result, *this, b);
    break;
  case ValType::U64:
    mul_lo_u64(result, *this, b);
    break;
  case ValType::S64:
    mul_lo_s64(result, *this, b);
    break;
  default:
    unsupported("*", type);
  }
  return result;
}

Value Value::operator/(const Value& b) const {
  Value result(type);
  switch (type) {
  case ValType::U32:
    div_u32(result, *this, b);
    break;
  case ValType::S32:
    div_s32(result, *this, b);
    break;
  default:
    unsupported("/", type);
  }
  return result;
}

Value Value::operator%(const Value& b) const {
  Value result(type);
  switch (type) {
  case ValType::U32:
    rem_u32(result, *this, b);
    break;
  case ValType::S32:
    rem_s32(result, *this, b);
    break;
  case ValType::U64:
    rem_u64(result, *this, b);
    break;
  default:
    unsupported("%", type);
  }
  return result;
}

Value Value::operator&(const Value& b) const {
  Value result(type);
  switch (regTypeFor(type)) {
  case RegType::Pred:
    and_pred(result, *this, b);
    break;
  case RegType::B32:
    and_b32(result, *this, b);
    break;
  case RegType::B64:
    and_b64(result, *this, b);
    break;
  default:
    unsupported("&", type);
  }
  return result;
}

Value Value::operator|(const Value& b) const {
  Value result(type);
  switch (regTypeFor(type)) {
  case RegType::Pred:
    or_pred(result, *this, b);
    break;
  case RegType::B32:
    or_b32(result, *this, b);
    break;
  case RegType::B64:
    or_b64(result, *this, b);
    break;
  default:
    unsupported("|", type);
  }
  return result;
}

Value Value::operator^(const Value& b) const {
  Value result(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    xor_b32(result, *this, b);
    break;
  default:
    unsupported("^", type);
  }
  return result;
}

Value Value::operator~() const {
  Value result(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    not_b32(result, *this);
    break;
  default:
    unsupported("~", type);
  }
  return result;
}

Value Value::operator!() const {
  Value result(ValType::Pred);
  switch (regTypeFor(type)) {
  case RegType::Pred:
    emitInst("not.pred " + result.str() + ", " + str());
    break;
  default:
    unsupported("!", type);
  }
  return result;
}

Value Value::operator-() const {
  Value result(ValType::S32);
  switch (regTypeFor(type)) {
  case RegType::B32:
    neg_s32(result, *this);
    break;
  default:
    unsupported("-", type);
  }
  return result;
}

Value Value::operator<<(const Value& b) const {
  Value result(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    shl_b32(result, *this, b);
    break;
  case RegType::B64:
    shl_b64(result, *this, b);
    break;
  default:
    unsupported("<<", type);
  }
  return result;
}

Value Value::operator>>(const Value& b) const {
  Value result(type);
  switch (type) {
  case ValType::U32:
    shr_u32(result, *this, b);
    break;
  case ValType::S32:
    shr_s32(result, *this, b);
    break;
  default:
    unsupported(">>", type);
  }
  return result;
}

Value Value::operator<(const Value& b) const {
  Value result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_lt_u32(result, *this, b);
    break;
  case ValType::S32:
    setp_lt_s32(result, *this, b);
    break;
  case ValType::S64:
    setp_lt_s64(result, *this, b);
    break;
  default:
    unsupported("<", type);
  }
  return result;
}

Value Value::operator<=(const Value& b) const {
  Value result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_le_u32(result, *this, b);
    break;
  default:
    unsupported("<=", type);
  }
  return result;
}

Value Value::operator>(const Value& b) const {
  Value result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_gt_u32(result, *this, b);
    break;
  default:
    unsupported(">", type);
  }
  return result;
}

Value Value::operator>=(const Value& b) const {
  Value result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_ge_u32(result, *this, b);
    break;
  case ValType::S32:
    setp_ge_s32(result, *this, b);
    break;
  case ValType::S64:
    setp_ge_s64(result, *this, b);
    break;
  default:
    unsupported(">=", type);
  }
  return result;
}

Value Value::operator==(const Value& b) const {
  Value result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_eq_u32(result, *this, b);
    break;
  case ValType::S32:
    setp_eq_s32(result, *this, b);
    break;
  case ValType::U64:
    setp_eq_u64(result, *this, b);
    break;
  default:
    unsupported("==", type);
  }
  return result;
}

Value Value::operator!=(const Value& b) const {
  Value result(ValType::Pred);
  switch (type) {
  case ValType::U32:
    setp_ne_u32(result, *this, b);
    break;
  case ValType::S32:
    setp_ne_s32(result, *this, b);
    break;
  case ValType::U64:
    setp_ne_u64(result, *this, b);
    break;
  case ValType::S64:
    setp_ne_s64(result, *this, b);
    break;
  default:
    unsupported("!=", type);
  }
  return result;
}

void Value::operator+=(const Value& b) {
  switch (type) {
  case ValType::U32:
    add_u32(*this, *this, b);
    break;
  case ValType::S32:
    add_s32(*this, *this, b);
    break;
  case ValType::U64:
    add_u64(*this, *this, b);
    break;
  case ValType::S64:
    add_s64(*this, *this, b);
    break;
  default:
    unsupported("+=", type);
  }
}

void Value::operator-=(const Value& b) {
  switch (type) {
  case ValType::U32:
    sub_u32(*this, *this, b);
    break;
  case ValType::S32:
    sub_s32(*this, *this, b);
    break;
  default:
    unsupported("-=", type);
  }
}

void Value::operator*=(const Value& b) {
  switch (type) {
  case ValType::U32:
    mul_lo_u32(*this, *this, b);
    break;
  case ValType::S32:
    mul_lo_s32(*this, *this, b);
    break;
  case ValType::U64:
    mul_lo_u64(*this, *this, b);
    break;
  case ValType::S64:
    mul_lo_s64(*this, *this, b);
    break;
  default:
    unsupported("*=", type);
  }
}

void Value::operator/=(const Value& b) {
  switch (type) {
  case ValType::U32:
    div_u32(*this, *this, b);
    break;
  case ValType::S32:
    div_s32(*this, *this, b);
    break;
  default:
    unsupported("/=", type);
  }
}

void Value::operator%=(const Value& b) {
  switch (type) {
  case ValType::U32:
    rem_u32(*this, *this, b);
    break;
  case ValType::S32:
    rem_s32(*this, *this, b);
    break;
  default:
    unsupported("%=", type);
  }
}

void Value::operator^=(const Value& b) {
  switch (regTypeFor(type)) {
  case RegType::B32:
    xor_b32(*this, *this, b);
    break;
  default:
    unsupported("^=", type);
  }
}

void Value::operator&=(const Value& b) {
  switch (regTypeFor(type)) {
  case RegType::B32:
    and_b32(*this, *this, b);
    break;
  case RegType::B64:
    and_b64(*this, *this, b);
    break;
  default:
    unsupported("&=", type);
  }
}

void Value::operator|=(const Value& b) {
  switch (regTypeFor(type)) {
  case RegType::B32:
    or_b32(*this, *this, b);
    break;
  case RegType::B64:
    or_b64(*this, *this, b);
    break;
  default:
    unsupported("|=", type);
  }
}

void Value::operator<<=(const Value& b) {
  switch (regTypeFor(type)) {
  case RegType::B32:
    shl_b32(*this, *this, b);
    break;
  case RegType::B64:
    shl_b64(*this, *this, b);
    break;
  default:
    unsupported("<<=", type);
  }
}

void Value::operator>>=(const Value& b) {
  switch (regTypeFor(type)) {
  case RegType::B32:
    shr_u32(*this, *this, b);
    break;
  default:
    unsupported(">>=", type);
  }
}

// --- FunctionScope ---

FunctionScope::FunctionScope(Function* f) : fn(f) {
  setFunction(f);
}

FunctionScope::~FunctionScope() {
  for (int i = 0; i < 4; i++) {
    fn->regCounts[i] = fn->regHighWater[i];
  }
  setFunction(nullptr);
}

// --- Block activation ---

static std::string genLabel(const char* prefix) {
  return std::string("L_") + prefix + "_" + std::to_string(currentFunction->labelCounter++);
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
    : pendingBlock(std::move(o.pendingBlock)), backEdgeTarget(o.backEdgeTarget), stepFn(std::move(o.stepFn)),
      closed(o.closed) {
  o.closed = true;
}

ScopeGuard::~ScopeGuard() {
  if (!closed) {
    if (stepFn) {
      stepFn(*this);
    }
    if (backEdgeTarget) {
      bra(backEdgeTarget);
    }
    if (pendingBlock) {
      activateBlock(std::move(pendingBlock));
    }
  }
}

// --- Control flow ---

ScopeGuard _If(const Value& pred) {
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

ScopeGuard _IfD(const Value& pred) {
  ScopeGuard sg;
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("endif");
  // Divergent branch: use bra (not bra.uni) so ptxas inserts reconvergence
  emitInst("@!" + pred.str() + " bra " + sg.pendingBlock->label);
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
  prevBlock->emit("bra.uni " + sg.pendingBlock->label);
  // Current block (the IF's skip target) becomes the else block — already active
  return sg;
}

ScopeGuard _WhileImpl(Block* header, const Value& cond) {
  ScopeGuard sg;
  // Create exit block (forward reference, not yet in function)
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("endwhile");
  sg.backEdgeTarget = header;
  // Convert non-pred to pred if needed (e.g. WHILE(true) → Value(1))
  if (cond.type != ValType::Pred) {
    bra_not(cond != 0, sg.pendingBlock.get());
  } else {
    bra_not(cond, sg.pendingBlock.get());
  }
  // Create and activate body block
  activateNewBlock("while_body");
  return sg;
}

ScopeGuard _Skip() {
  ScopeGuard sg;
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("postskip");
  emitInst("bra.uni " + sg.pendingBlock->label);
  activateNewBlock("skip");
  return sg;
}

ScopeGuard _Switch(const u32& index, bool divergent) {
  ScopeGuard sg;
  sg.pendingBlock = std::make_unique<Block>();
  sg.pendingBlock->label = genLabel("endswitch");

  u32 index_value = index;

  Label label("switch");
  GOTO(label);
  sg.stepFn = [label = std::move(label), index_value = std::move(index_value), divergent](ScopeGuard& sg) mutable {
    LABEL(label);
    for (auto& v : sg.caseLabels) {
      if (v.rawPtr == nullptr) {
        v.rawPtr = &*sg.pendingBlock;
      }
    }
    if (!sg.caseLabels.empty()) {
      IF(index_value < sg.caseLabels.size()) {
        brx_idx(index_value, sg.caseLabels, divergent);
      }
    }
  };
  activateNewBlock("switch_body");
  return sg;
}
ScopeGuard _Case(ScopeGuard& scope, uint32_t value) {
  if (value >= 65536) {
    throw std::runtime_error("case value is to large");
  }
  Label label("case");
  LABEL(label);
  if (scope.caseLabels.size() <= value) {
    scope.caseLabels.resize(value + 1);
    scope.caseLabels[value] = std::move(label);
  }

  ScopeGuard sg;
  sg.backEdgeTarget = &*scope.pendingBlock;
  return sg;
}

// --- Labels ---

Label::Label() {
  block = std::make_unique<Block>();
  block->label = genLabel("label");
  rawPtr = block.get();
}

Label::Label(const char* name) {
  block = std::make_unique<Block>();
  block->label = genLabel(name);
  rawPtr = block.get();
}

void activateLabel(Label& label) {
  assert(label.block && "Label already placed");
  activateBlock(std::move(label.block));
}

// --- Predicated execution ---

PredGuard::PredGuard(const Value& pred) {
  if (!currentPred.empty()) {
    throw std::runtime_error("nested PRED not supported");
  }
  currentPred = "@" + pred.str() + " ";
}

PredGuard::~PredGuard() {
  currentPred.clear();
}

// --- Convenience functions ---

u32 special32(std::string name) {
  u32 r;
  r = Value(ValType::U32, name);
  return r;
}

u64 special64(std::string name) {
  u64 r;
  r = Value(ValType::U64, name);
  return r;
}

u32 threadIdx_x() {
  return special32("%tid.x");
}

u32 blockIdx_x() {
  return special32("%ctaid.x");
}

u32 blockDim_x() {
  return special32("%ntid.x");
}

u64 clock64() {
  return special64("%clock64");
}

Value loadParam(int index, ValType type) {
  Value v(type);
  std::string paramName = currentFunction->param(index);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_param_u32(v, paramName);
    break;
  case RegType::B64:
    ld_param_u64(v, paramName);
    break;
  default:
    unsupported("loadParam", type);
  }
  return v;
}

Value paramBase(int index) {
  Value v(ValType::U32);
  mov_u32(v, Value(ValType::U32, currentFunction->param(index).c_str()));
  return v;
}

Value loadParamField(const Value& base, int offset, ValType type) {
  Value v(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_param_u32(v, base, offset);
    break;
  case RegType::B64:
    ld_param_u64(v, base, offset);
    break;
  default:
    unsupported("loadParamField", type);
  }
  return v;
}

Value widen(const Value& v) {
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
  Value result(wide);
  cvt_u64_u32(result, v);
  return result;
}

Value narrow(const Value& v) {
  ValType n;
  switch (v.type) {
  case ValType::U64:
    n = ValType::U32;
    break;
  case ValType::S64:
    n = ValType::S32;
    break;
  default:
    unsupported("narrow", v.type);
  }
  Value result(n);
  cvt_u32_u64(result, v);
  return result;
}

void storeGlobal(const Value& addr, const Value& val) {
  switch (regTypeFor(val.type)) {
  case RegType::B32:
    st_global_u32(addr, val);
    break;
  default:
    unsupported("storeGlobal", val.type);
  }
}

Value loadGlobalVolatile(const Value& addr, ValType type) {
  Value v(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_global_volatile_u32(v, addr);
    break;
  default:
    unsupported("loadGlobalVolatile", type);
  }
  return v;
}

void storeGlobalVolatile(const Value& addr, const Value& val) {
  switch (regTypeFor(val.type)) {
  case RegType::B32:
    st_global_volatile_u32(addr, val);
    break;
  default:
    unsupported("storeGlobalVolatile", val.type);
  }
}

Value loadGlobalAcquireSys(const Value& addr, ValType type) {
  Value v(type);
  switch (regTypeFor(type)) {
  case RegType::B32:
    ld_global_acquire_sys_u32(v, addr);
    break;
  default:
    unsupported("loadGlobalAcquireSys", type);
  }
  return v;
}

void storeGlobalRelaxedSys(const Value& addr, const Value& val) {
  switch (regTypeFor(val.type)) {
  case RegType::B32:
    st_global_relaxed_sys_u32(addr, val);
    break;
  default:
    unsupported("storeGlobalRelaxedSys", val.type);
  }
}

void storeGlobalReleaseSys(const Value& addr, const Value& val) {
  switch (regTypeFor(val.type)) {
  case RegType::B32:
    st_global_release_sys_u32(addr, val);
    break;
  default:
    unsupported("storeGlobalReleaseSys", val.type);
  }
}

void ldcv_v4(Value& v0, Value& v1, Value& v2, Value& v3, const Value& addr) {
  if (!v0.valid) {
    v0 = Value(ValType::U32);
  }
  if (!v1.valid) {
    v1 = Value(ValType::U32);
  }
  if (!v2.valid) {
    v2 = Value(ValType::U32);
  }
  if (!v3.valid) {
    v3 = Value(ValType::U32);
  }
  ld_global_cv_v4_u32(v0, v1, v2, v3, addr);
}

void ldcv_v4(std::array<Value, 4>& v, const Value& addr) {
  ldcv_v4(v[0], v[1], v[2], v[3], addr);
}

void ldnc_v4(Value& v0, Value& v1, Value& v2, Value& v3, const Value& addr) {
  if (!v0.valid) {
    v0 = Value(ValType::U32);
  }
  if (!v1.valid) {
    v1 = Value(ValType::U32);
  }
  if (!v2.valid) {
    v2 = Value(ValType::U32);
  }
  if (!v3.valid) {
    v3 = Value(ValType::U32);
  }
  ld_global_nc_v4_u32(v0, v1, v2, v3, addr);
}

void ldnc_v4(std::array<Value, 4>& v, const Value& addr) {
  ldnc_v4(v[0], v[1], v[2], v[3], addr);
}

void ldcs_v4(Value& v0, Value& v1, Value& v2, Value& v3, const Value& addr) {
  if (!v0.valid) {
    v0 = Value(ValType::U32);
  }
  if (!v1.valid) {
    v1 = Value(ValType::U32);
  }
  if (!v2.valid) {
    v2 = Value(ValType::U32);
  }
  if (!v3.valid) {
    v3 = Value(ValType::U32);
  }
  ld_global_cs_v4_u32(v0, v1, v2, v3, addr);
}

void ldcs_v4(std::array<Value, 4>& v, const Value& addr) {
  ldcs_v4(v[0], v[1], v[2], v[3], addr);
}

void ld_plain_v4(std::array<Value, 4>& v, const Value& addr) {
  for (auto& vi : v) {
    if (!vi.valid) {
      vi = Value(ValType::U32);
    }
  }
  ld_v4_u32(v, addr);
}

void stwt_v4(const Value& addr, const Value& v0, const Value& v1, const Value& v2, const Value& v3) {
  st_global_wt_v4_u32(addr, v0, v1, v2, v3);
}

void stwt_v4(const Value& addr, const std::array<Value, 4>& v) {
  stwt_v4(addr, v[0], v[1], v[2], v[3]);
}

Value atomicInc(const Value& addr, const Value& modulo) {
  Value result(ValType::U32);
  atom_global_inc_u32(result, addr, modulo);
  return result;
}

Value globalAddr(const char* name) {
  Value v(ValType::U64);
  mov_u64(v, Value(ValType::U64, name));
  return v;
}

Value constAddr(const char* name) {
  Value v(ValType::U32);
  emitInst(std::string("mov.u32 ") + v.str() + ", " + name);
  return v;
}

Value sharedAddr(const char* name) {
  Value v(ValType::U32);
  emitInst(std::string("mov.u32 ") + v.str() + ", " + name);
  return v;
}

Value addShared32(int align, int sizeBytes, const char* suffix) {
  std::string sym = currentFunction->addShared(align, sizeBytes, suffix);
  return sharedAddr(sym.c_str());
}

// Value hexImm(uintptr_t value) {
//   char buf[32];
//   snprintf(buf, sizeof(buf), "0x%lx", (unsigned long)value);
//   return Value(buf);
// }

} // namespace ptx
} // namespace moodist
