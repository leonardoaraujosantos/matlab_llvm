#pragma once

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/MIR/MIR.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

namespace matlab {
namespace mir {

class Builder;

// Lowers a typed AST (post-Sema) into MIR.
//
// Strategy:
//   - A MATLAB script becomes a top-level function @script.
//   - Each TU function becomes a matlab.func op.
//   - User variables are materialized with matlab.alloc in the entry block
//     and accessed via matlab.load / matlab.store. A later pass (mem2reg) can
//     promote those to SSA values with scf block arguments.
//   - Structured control flow (if/for/while) lowers into nested regions on
//     scf.if / scf.for / scf.while ops.
class Lowerer {
public:
  Lowerer(MIRContext &MIRCtx, TypeContext &TC, DiagnosticEngine &Diag);

  // Lower a translation unit. Caller owns the returned Module (stored in
  // MIRContext's arena).
  Module lower(const ::matlab::TranslationUnit &TU);

private:
  MIRContext &MIR;
  TypeContext &TC;
  DiagnosticEngine &Diag;

  // Maps Sema bindings to their slot values in the current function body.
  // (Parameters are also mapped to a slot: we spill the block-arg into a slot
  // on entry so param uses look identical to var uses.)
  std::unordered_map<Binding *, Value *> SlotMap;

  // Lower either a script or a function definition.
  void lowerScript(const ::matlab::Script &S, Op *ModuleBody);
  void lowerFunction(const ::matlab::Function &F, Op *ModuleBody, Builder &B);

  // Statement / expression lowering.
  void lowerBlock(const ::matlab::Block &B, Builder &Bld);
  void lowerStmt(const ::matlab::Stmt &St, Builder &Bld);
  Value *lowerExpr(const ::matlab::Expr &E, Builder &Bld);

  // Helpers
  Value *lowerLValueStore(const ::matlab::Expr &LHS, Value *Rhs, Builder &Bld);
  Value *loadBinding(Binding *B, const Type *ValTy, Builder &Bld,
                     SourceRange Loc);
  Value *getOrCreateSlot(Binding *B, const Type *Ty, Builder &Bld);

  OpKind binOpKind(BinOp Op);
  OpKind unOpKind(UnOp Op);
  OpKind postfixKind(PostfixOp Op);
};

} // namespace mir
} // namespace matlab
