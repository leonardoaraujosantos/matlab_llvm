#pragma once

#include "matlab/MIR/MIR.h"
#include "matlab/Sema/Type.h"

namespace matlab {
namespace mir {

/// Insertion-point-based op builder. Mirrors mlir::OpBuilder in spirit.
class Builder {
public:
  Builder(MIRContext &Ctx, TypeContext &TC)
      : Ctx(Ctx), TC(TC) {}

  MIRContext &context() { return Ctx; }
  TypeContext &types() { return TC; }

  // Insertion point.
  void setInsertionPointToEnd(Block *B) { InsertBlock = B; }
  Block *getInsertionBlock() const { return InsertBlock; }

  // Low-level: create an op and insert it at the insertion point.
  Op *create(OpKind K, std::vector<Value *> Operands = {},
             std::vector<const Type *> ResultTypes = {},
             SourceRange Loc = {});

  // Typed constant helpers.
  Value *constInt(int64_t V, const Type *Ty, SourceRange Loc = {});
  Value *constFloat(double V, const Type *Ty, SourceRange Loc = {});
  Value *constString(std::string V, const Type *Ty, SourceRange Loc = {});
  Value *constChar(std::string V, const Type *Ty, SourceRange Loc = {});
  Value *constLogical(bool V, SourceRange Loc = {});
  Value *constColon(SourceRange Loc = {});
  Value *constEnd(SourceRange Loc = {});

  // Variable slot operations.
  Value *alloc(const Type *Ty, std::string Name = "", SourceRange Loc = {});
  Value *load(Value *Slot, const Type *Ty, SourceRange Loc = {});
  void   store(Value *V, Value *Slot, SourceRange Loc = {});

  // Arithmetic / comparison — builder figures out the result type from Ty arg.
  Value *unary(OpKind K, Value *A, const Type *Ty, SourceRange Loc = {});
  Value *binary(OpKind K, Value *A, Value *B, const Type *Ty, SourceRange Loc = {});

  // Array construction
  Value *range(Value *Start, Value *Step, Value *End, const Type *Ty,
               SourceRange Loc = {});
  Value *concatRow(std::vector<Value *> Elts, const Type *Ty, SourceRange Loc = {});
  Value *concatCol(std::vector<Value *> Elts, const Type *Ty, SourceRange Loc = {});
  Value *subscript(Value *Arr, std::vector<Value *> Idx, const Type *Ty,
                   SourceRange Loc = {});
  Value *cellSubscript(Value *Arr, std::vector<Value *> Idx, const Type *Ty,
                       SourceRange Loc = {});
  Value *fieldAccess(Value *Base, std::string Field, const Type *Ty,
                     SourceRange Loc = {});
  Value *dynamicField(Value *Base, Value *Name, const Type *Ty,
                      SourceRange Loc = {});

  // Calls
  Value *call(std::string Callee, std::vector<Value *> Args,
              const Type *Ty, SourceRange Loc = {});
  Value *callBuiltin(std::string Callee, std::vector<Value *> Args,
                     const Type *Ty, SourceRange Loc = {});
  Value *callIndirect(Value *Handle, std::vector<Value *> Args,
                      const Type *Ty, SourceRange Loc = {});

  // Handles
  Value *makeHandle(std::string Name, const Type *Ty, SourceRange Loc = {});
  // MakeAnon: body region is populated by the caller.
  Op *makeAnon(std::vector<std::string> Params, const Type *Ty,
               SourceRange Loc = {});

  // Structured control flow — returns the op so the caller can populate regions.
  Op *ifOp(Value *Cond, SourceRange Loc = {});
  Op *forOp(std::string VarName, Value *Iter, SourceRange Loc = {});
  Op *whileOp(SourceRange Loc = {});

  // Terminators
  void yield(std::vector<Value *> Vs = {}, SourceRange Loc = {});
  void retOp(std::vector<Value *> Vs = {}, SourceRange Loc = {});
  void breakOp(SourceRange Loc = {});
  void continueOp(SourceRange Loc = {});

  // FuncOp creation
  Op *funcOp(std::string Name, std::vector<const Type *> InTys,
             std::vector<const Type *> OutTys, SourceRange Loc = {});

private:
  MIRContext &Ctx;
  TypeContext &TC;
  Block *InsertBlock = nullptr;
};

} // namespace mir
} // namespace matlab
