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

  // REPL mode: treats unresolved NameExpr references at script scope as
  // implicit Vars (backed by the runtime workspace) instead of emitting
  // an "undefined name" diagnostic. Lets the user type `disp(x)` for a
  // variable defined in a prior input line without each input becoming
  // its own isolated compilation unit.
  void setReplMode(bool V) { ReplMode = V; }

  // REPL workspace kind hook — when ReplMode is on, the Resolver calls
  // this for every newly auto-declared binding to learn the kind of
  // the value previously stored under that name (0=f64, 1=mat, 2=obj,
  // 3=string, -1=missing). The result seeds the binding's
  // InferredType so the dispatch sites that key on type (string disp,
  // strlen, isstring, etc.) light up across compilation boundaries —
  // without this hook a fresh-input `disp(t)` can't see that `t` was
  // assigned a "..." in a prior compile.
  //
  // matlabc/main.cpp installs the hook before each REPL compile;
  // other Sema clients that don't need workspace introspection
  // simply leave it null.
  using WorkspaceKindHookT = int (*)(const char *name, int64_t len);
  void setWorkspaceKindHook(WorkspaceKindHookT H) { WorkspaceKindHook = H; }

  // REPL workspace class-name hook — for kind=2 (matlab_obj*) bindings,
  // returns the class name stored with the instance (looked up from the
  // runtime's class registry via the obj's class_id).  Used by the
  // Resolver to re-pin `Binding::PinnedClass` cross-turn so the
  // `obj(args)` System-Object sugar and dot-method dispatch still
  // route to the class methods on a subsequent REPL input.  Returns
  // null when the name isn't a kind=2 binding or the class hasn't
  // been registered (e.g. a function ran with class registration
  // disabled — pre-2026-05 behavior).
  using WorkspaceClassNameHookT =
      const char *(*)(const char *name, int64_t len, int64_t *name_len_out);
  void setWorkspaceClassNameHook(WorkspaceClassNameHookT H) {
    WorkspaceClassNameHook = H;
  }

private:
  SemaContext &Sema;
  TypeContext &TC;
  DiagnosticEngine &Diag;
  Scope *Global = nullptr;
  bool ReplMode = false;
  WorkspaceKindHookT WorkspaceKindHook = nullptr;
  WorkspaceClassNameHookT WorkspaceClassNameHook = nullptr;

  void registerBuiltins();
  void registerBuiltin(std::string_view Name);

  Binding *declareFn(Scope *S, Function *F);
  Binding *declareClass(Scope *S, ClassDef *C);

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
