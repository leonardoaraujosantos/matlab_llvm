#pragma once

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Sema/Scope.h"

namespace matlab {

class TypeContext;

class Resolver {
public:
  Resolver(SemaContext &Sema, TypeContext &TC, DiagnosticEngine &Diag);

  // Resolve a whole translation unit in place. After this call:
  //   - Every NameExpr has Ref set (or a diagnostic was emitted)
  //   - Every CallOrIndex has Resolved set to Call or Index
  //   - Every Function has FnScope, ParamRefs, OutputRefs populated
  void resolve(TranslationUnit &TU);

  // Global scope (TU-level functions + registered builtins).
  Scope *globalScope() { return Global; }

private:
  SemaContext &Sema;
  TypeContext &TC;
  DiagnosticEngine &Diag;
  Scope *Global = nullptr;

  void registerBuiltins();
  void registerBuiltin(std::string_view Name);

  Binding *declareFn(Scope *S, Function *F);

  // Two-pass scope construction for a function.
  void collectAssignments(Function &F, Scope *FnScope);
  void collectAssignmentsInBlock(Block &B, Scope *FnScope);
  void collectAssignmentsInStmt(Stmt &S, Scope *FnScope);
  void collectAssignmentsInExpr(Expr &E, Scope *FnScope); // for anon fn vars? no-op

  // Resolution pass.
  void resolveFunction(Function &F, Scope *Parent);
  void resolveBlock(Block &B, Scope *S);
  void resolveStmt(Stmt &St, Scope *S);
  void resolveExpr(Expr &E, Scope *S);
  void resolveLValue(Expr &E, Scope *S); // for assignment LHS
  void resolveCallee(CallOrIndex &C, Scope *S);
};

} // namespace matlab
