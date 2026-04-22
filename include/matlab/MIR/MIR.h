#pragma once

// Core MIR data structures: Value, Op (tag-based), Block, Region, MIRContext.
//
// Structured control flow (MLIR scf-style): IfOp, ForOp, WhileOp carry nested
// Regions rather than branch terminators. That way the lowering pass can emit
// straight from the AST without building a CFG, and a later pass can flatten
// into basic blocks when we port to real MLIR.

#include "matlab/Basic/SourceManager.h"
#include "matlab/Sema/Type.h"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace matlab {
namespace mir {

class Op;
class Block;
class Region;
class MIRContext;

//===----------------------------------------------------------------------===//
// OpKind — every distinct operation kind in our pseudo-dialect.
//===----------------------------------------------------------------------===//

enum class OpKind : uint16_t {
  // Module / function structure
  ModuleOp,
  FuncOp,

  // Constants
  ConstInt,
  ConstFloat,
  ConstComplex,
  ConstString,
  ConstChar,
  ConstLogical,
  ConstColon,  // MATLAB ':' indexing sentinel
  ConstEnd,    // MATLAB 'end' sentinel within subscripts

  // Stack-slot variable model (Alloc/Load/Store — mem2reg fodder)
  Alloc,
  Load,
  Store,

  // Unary arithmetic / logical
  Neg,
  UPlus,
  Not,

  // Binary arithmetic
  Add, Sub,
  MatMul,  MatDiv,  MatLDiv,  MatPow,        // '*' '/' '\' '^'
  EMul,    EDiv,    ELDiv,    EPow,           // '.*' './' '.\' '.^'

  // Comparison
  Eq, Ne, Lt, Le, Gt, Ge,

  // Logical
  BitAnd, BitOr, ShortAnd, ShortOr,

  // Array ops
  Transpose,    // .'
  CTranspose,   // '
  Range,        // a:b or a:step:b
  ConcatRow,    // horizontal concat (comma / space in matrix literal)
  ConcatCol,    // vertical concat (; in matrix literal)
  Subscript,    // a(i,j,...) with regular indexing
  CellSubscript,// c{i}
  FieldAccess,  // s.name

  // Calls
  Call,          // user-defined function (symbol + args)
  CallBuiltin,   // known builtin (symbol + args)
  CallIndirect,  // apply a function handle (callee is a value)

  // Handles
  MakeHandle,    // @name -> handle
  MakeAnon,      // @(...) expr -> handle (with captured region)

  // Structured control flow
  IfOp,          // regions: then, else (else optional/empty)
  ForOp,         // operand: iterable; region: body w/ bb-arg = loop var
  WhileOp,       // regions: cond, body
  Yield,         // implicit terminator of structured regions
  Return,        // early return from a FuncOp
  Break,
  Continue,

  // Matrix/cell literal construction
  MakeMatrix,    // rows as a list of "row ops" — we flatten via ConcatRow/Col
  MakeCell,

  // Struct / dynamic field
  DynamicField,
};

const char *opKindName(OpKind K);

//===----------------------------------------------------------------------===//
// Attribute — simple variant type for inline op metadata (string name, const
// values, symbol refs). Not modelled as first-class TypeAttr / IntAttr classes
// the way MLIR does, but enough for our printer + lowering.
//===----------------------------------------------------------------------===//

struct Attribute {
  enum class Kind : uint8_t { None, Int, Float, Str, Sym, Bool };
  Kind K = Kind::None;
  int64_t I = 0;
  double F = 0.0;
  std::string S;  // also used for Sym

  static Attribute ofInt(int64_t V)   { Attribute A; A.K = Kind::Int;   A.I = V; return A; }
  static Attribute ofFloat(double V)  { Attribute A; A.K = Kind::Float; A.F = V; return A; }
  static Attribute ofStr(std::string V){ Attribute A; A.K = Kind::Str;  A.S = std::move(V); return A; }
  static Attribute ofSym(std::string V){ Attribute A; A.K = Kind::Sym;  A.S = std::move(V); return A; }
  static Attribute ofBool(bool V)     { Attribute A; A.K = Kind::Bool;  A.I = V; return A; }
};

//===----------------------------------------------------------------------===//
// Value — SSA value. Either an Op result or a Block argument.
//===----------------------------------------------------------------------===//

class Value {
public:
  const Type *Ty = nullptr;
  Op *DefiningOp = nullptr;     // non-null if produced by an op
  Block *DefiningBlock = nullptr; // non-null if a block argument
  uint32_t Index = 0;           // position among results / arguments
  uint32_t Id = 0;              // for printing; assigned by MIRContext
  SourceLocation Loc;
};

//===----------------------------------------------------------------------===//
// Op — one operation. Single class with OpKind tag.
//===----------------------------------------------------------------------===//

class Op {
public:
  OpKind K;
  std::vector<Value *> Operands;
  std::vector<Value *> Results;   // owned by MIRContext
  std::vector<Region *> Regions;  // owned by MIRContext
  std::vector<std::pair<std::string, Attribute>> Attrs;
  SourceRange Loc;

  Block *ParentBlock = nullptr;

  Value *result(size_t I = 0) const { return I < Results.size() ? Results[I] : nullptr; }

  void setAttr(std::string Name, Attribute V) {
    for (auto &P : Attrs)
      if (P.first == Name) { P.second = std::move(V); return; }
    Attrs.emplace_back(std::move(Name), std::move(V));
  }
  const Attribute *getAttr(std::string_view Name) const {
    for (auto &P : Attrs) if (P.first == Name) return &P.second;
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Block — contains ops. We use a std::list for cheap insert/splice.
//===----------------------------------------------------------------------===//

class Block {
public:
  std::vector<Value *> Arguments;
  std::list<Op *> Ops;
  Region *Parent = nullptr;
  uint32_t Id = 0;
};

//===----------------------------------------------------------------------===//
// Region — a list of blocks belonging to a single parent op.
//===----------------------------------------------------------------------===//

class Region {
public:
  std::list<Block *> Blocks;
  Op *Parent = nullptr;

  Block *entry() const { return Blocks.empty() ? nullptr : Blocks.front(); }
};

//===----------------------------------------------------------------------===//
// MIRContext — owns all MIR objects.
//===----------------------------------------------------------------------===//

class MIRContext {
public:
  MIRContext();
  ~MIRContext();
  MIRContext(const MIRContext &) = delete;
  MIRContext &operator=(const MIRContext &) = delete;

  Op *newOp(OpKind K);
  Value *newValue(const Type *Ty);
  Block *newBlock();
  Region *newRegion();

  // Convenience: fresh block-argument Value attached to the given block.
  Value *addBlockArgument(Block *B, const Type *Ty);

private:
  std::vector<std::unique_ptr<Op>> Ops;
  std::vector<std::unique_ptr<Value>> Values;
  std::vector<std::unique_ptr<Block>> Blocks;
  std::vector<std::unique_ptr<Region>> Regions;
  uint32_t NextValueId = 0;
  uint32_t NextBlockId = 0;
};

//===----------------------------------------------------------------------===//
// Module — a top-level container holding function ops. A ModuleOp with a
// single region containing one block.
//===----------------------------------------------------------------------===//

struct Module {
  Op *ModuleOp = nullptr;  // op of kind ModuleOp
  Region *Body = nullptr;
  Block *EntryBlock = nullptr;
};

Module createModule(MIRContext &Ctx);

} // namespace mir
} // namespace matlab
