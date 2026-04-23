#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/Type.h"

#include <string>

namespace matlab {

Resolver::Resolver(SemaContext &Sema, TypeContext &TC, DiagnosticEngine &Diag)
    : Sema(Sema), TC(TC), Diag(Diag) {
  Global = Sema.newScope(nullptr, "<global>");
  registerBuiltins();
}

void Resolver::registerBuiltin(std::string_view Name) {
  Binding *B = Sema.newBinding();
  Global->declare(Name, BindingKind::Builtin, B);
}

void Resolver::registerBuiltins() {
  // Minimal initial registry. Type inference will special-case some of these
  // to produce concrete shape/dtype results.
  for (const char *N : {
    "zeros", "ones", "eye", "rand", "randn", "magic", "diag",
    "size", "length", "numel", "ndims",
    "reshape", "repmat", "linspace",
    "abs", "sqrt", "exp", "log", "sin", "cos", "tan",
    "min", "max", "sum", "prod", "mean",
    "mtimes", "mldivide", "mrdivide",
    "transpose", "ctranspose",
    "disp", "fprintf", "sprintf", "error", "warning", "input", "clear",
    "keyboard", "pause", "tic", "toc",
    "isempty", "isequal", "find",
    "true", "false",
    "mod", "rem", "floor", "ceil", "round", "fix",
    "double", "single", "int32", "int64", "logical", "char",
    "struct", "cell", "fieldnames",
    "svd", "eig", "inv", "pinv", "det", "rank",
    "arrayfun", "cellfun",
    "nargin", "nargout", "varargin", "varargout",
  }) {
    registerBuiltin(N);
  }
}

Binding *Resolver::declareFn(Scope *S, Function *F) {
  Binding *B = Sema.newBinding();
  Binding *Declared = S->declare(F->Name, BindingKind::Function, B);
  Declared->FuncDef = F;
  F->Self = Declared;
  return Declared;
}

void Resolver::resolve(TranslationUnit &TU) {
  // Register all top-level functions in the global scope *before* resolving
  // their bodies so mutual calls work.
  for (Function *F : TU.Functions) declareFn(Global, F);

  if (TU.ScriptNode) {
    Scope *ScriptScope = Sema.newScope(Global, "<script>");
    // Pre-collect assignments so NameExpr uses can see script-local vars.
    collectAssignmentsInBlock(*TU.ScriptNode->Body, ScriptScope);
    resolveBlock(*TU.ScriptNode->Body, ScriptScope);
  }

  for (Function *F : TU.Functions) resolveFunction(*F, Global);
}

//===----------------------------------------------------------------------===//
// Assignment collection (pre-pass to populate variable bindings).
//===----------------------------------------------------------------------===//

void Resolver::collectAssignments(Function &F, Scope *FnScope) {
  // Parameters and outputs are declared up-front.
  for (auto Name : F.Inputs) {
    if (Name == "~") continue; // placeholder parameter
    Binding *B = Sema.newBinding();
    Binding *D = FnScope->declare(Name, BindingKind::Param, B);
    F.ParamRefs.push_back(D);
  }
  for (auto Name : F.Outputs) {
    Binding *B = Sema.newBinding();
    Binding *D = FnScope->declare(Name, BindingKind::Output, B);
    F.OutputRefs.push_back(D);
  }
  // Register nested functions so calls to them resolve inside the parent body.
  for (Function *N : F.Nested) {
    declareFn(FnScope, N);
  }
  if (F.Body) collectAssignmentsInBlock(*F.Body, FnScope);
}

void Resolver::collectAssignmentsInBlock(Block &B, Scope *FnScope) {
  for (Stmt *S : B.Stmts)
    if (S) collectAssignmentsInStmt(*S, FnScope);
}

void Resolver::collectAssignmentsInStmt(Stmt &S, Scope *FnScope) {
  switch (S.Kind) {
  case NodeKind::AssignStmt: {
    auto &A = static_cast<AssignStmt &>(S);
    for (Expr *L : A.LHS) {
      // Peel off indexing/field-access to find the root name.
      Expr *Root = L;
      while (Root) {
        switch (Root->Kind) {
        case NodeKind::NameExpr: {
          auto *N = static_cast<NameExpr *>(Root);
          Binding *B = Sema.newBinding();
          FnScope->getOrDeclareVar(N->Name, B);
          Root = nullptr;
          break;
        }
        case NodeKind::CallOrIndex:
          Root = static_cast<CallOrIndex *>(Root)->Callee;
          break;
        case NodeKind::CellIndex:
          Root = static_cast<CellIndex *>(Root)->Callee;
          break;
        case NodeKind::FieldAccess:
          Root = static_cast<FieldAccess *>(Root)->Base;
          break;
        case NodeKind::DynamicField:
          Root = static_cast<DynamicField *>(Root)->Base;
          break;
        default:
          Root = nullptr;
          break;
        }
      }
    }
    break;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<ForStmt &>(S);
    if (!F.Var.empty()) {
      Binding *B = Sema.newBinding();
      /* A prior for-loop with the same variable name reuses its
       * binding; keep VarRef pointing at the real one so the lowerer's
       * slot lookup returns the shared slot. */
      F.VarRef = FnScope->getOrDeclareVar(F.Var, B);
    }
    if (F.Body) collectAssignmentsInBlock(*F.Body, FnScope);
    break;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<WhileStmt &>(S);
    if (W.Body) collectAssignmentsInBlock(*W.Body, FnScope);
    break;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<IfStmt &>(S);
    if (I.Then) collectAssignmentsInBlock(*I.Then, FnScope);
    for (auto &EI : I.Elseifs)
      if (EI.Body) collectAssignmentsInBlock(*EI.Body, FnScope);
    if (I.Else) collectAssignmentsInBlock(*I.Else, FnScope);
    break;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<SwitchStmt &>(S);
    for (auto &C : Sw.Cases)
      if (C.Body) collectAssignmentsInBlock(*C.Body, FnScope);
    break;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<TryStmt &>(S);
    if (T.TryBody) collectAssignmentsInBlock(*T.TryBody, FnScope);
    if (!T.CatchVar.empty()) {
      Binding *B = Sema.newBinding();
      FnScope->getOrDeclareVar(T.CatchVar, B);
    }
    if (T.CatchBody) collectAssignmentsInBlock(*T.CatchBody, FnScope);
    break;
  }
  case NodeKind::GlobalDecl: {
    auto &G = static_cast<GlobalDecl &>(S);
    for (auto N : G.Names) {
      Binding *B = Sema.newBinding();
      FnScope->declare(N, BindingKind::Global, B);
    }
    break;
  }
  case NodeKind::PersistentDecl: {
    auto &P = static_cast<PersistentDecl &>(S);
    for (auto N : P.Names) {
      Binding *B = Sema.newBinding();
      FnScope->declare(N, BindingKind::Persistent, B);
    }
    break;
  }
  case NodeKind::CommandStmt: {
    // Commands don't introduce variables.
    break;
  }
  default:
    break;
  }
}

//===----------------------------------------------------------------------===//
// Resolution pass.
//===----------------------------------------------------------------------===//

void Resolver::resolveFunction(Function &F, Scope *Parent) {
  F.FnScope = Sema.newScope(Parent, std::string(F.Name));
  collectAssignments(F, F.FnScope);
  if (F.Body) resolveBlock(*F.Body, F.FnScope);
  for (Function *N : F.Nested) resolveFunction(*N, F.FnScope);
}

void Resolver::resolveBlock(Block &B, Scope *S) {
  for (Stmt *St : B.Stmts) if (St) resolveStmt(*St, S);
}

void Resolver::resolveStmt(Stmt &St, Scope *S) {
  switch (St.Kind) {
  case NodeKind::ExprStmt: {
    auto &E = static_cast<ExprStmt &>(St);
    if (E.E) resolveExpr(*E.E, S);
    break;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<AssignStmt &>(St);
    if (A.RHS) resolveExpr(*A.RHS, S);
    for (Expr *L : A.LHS) resolveLValue(*L, S);
    break;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<IfStmt &>(St);
    if (I.Cond) resolveExpr(*I.Cond, S);
    if (I.Then) resolveBlock(*I.Then, S);
    for (auto &EI : I.Elseifs) {
      if (EI.Cond) resolveExpr(*EI.Cond, S);
      if (EI.Body) resolveBlock(*EI.Body, S);
    }
    if (I.Else) resolveBlock(*I.Else, S);
    break;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<ForStmt &>(St);
    if (F.Iter) resolveExpr(*F.Iter, S);
    if (F.Body) resolveBlock(*F.Body, S);
    break;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<WhileStmt &>(St);
    if (W.Cond) resolveExpr(*W.Cond, S);
    if (W.Body) resolveBlock(*W.Body, S);
    break;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<SwitchStmt &>(St);
    if (Sw.Discriminant) resolveExpr(*Sw.Discriminant, S);
    for (auto &C : Sw.Cases) {
      if (C.Value) resolveExpr(*C.Value, S);
      if (C.Body)  resolveBlock(*C.Body, S);
    }
    break;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<TryStmt &>(St);
    if (T.TryBody) resolveBlock(*T.TryBody, S);
    if (T.CatchBody) resolveBlock(*T.CatchBody, S);
    break;
  }
  case NodeKind::CommandStmt: {
    auto &C = static_cast<CommandStmt &>(St);
    Binding *B = S->lookup(C.Name);
    if (!B) {
      Diag.error(C.Range.Begin,
                 std::string("undefined command or function '") +
                     std::string(C.Name) + "'");
    }
    break;
  }
  default:
    break;
  }
}

void Resolver::resolveLValue(Expr &E, Scope *S) {
  switch (E.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<NameExpr &>(E);
    Binding *B = S->lookup(N.Name);
    if (!B) {
      // Should have been pre-declared by collectAssignments; if not, emit.
      Diag.error(N.Range.Begin,
                 std::string("cannot assign to undeclared name '") +
                     std::string(N.Name) + "'");
      return;
    }
    if (B->Kind == BindingKind::Function || B->Kind == BindingKind::Builtin) {
      Diag.error(N.Range.Begin,
                 std::string("cannot assign to function '") +
                     std::string(N.Name) + "'");
    }
    N.Ref = B;
    B->WrittenTo = true;
    break;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<CallOrIndex &>(E);
    // `a(i) = x` — LHS must be indexing into a variable.
    resolveCallee(C, S);
    if (C.Resolved == CallKind::Call) {
      Diag.error(C.Range.Begin, "cannot assign to function call result");
    }
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<CellIndex &>(E);
    if (C.Callee) resolveLValue(*C.Callee, S);
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<FieldAccess &>(E);
    if (F.Base) resolveLValue(*F.Base, S);
    break;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<DynamicField &>(E);
    if (F.Base) resolveLValue(*F.Base, S);
    if (F.Name) resolveExpr(*F.Name, S);
    break;
  }
  default:
    Diag.error(E.Range.Begin, "expression is not assignable");
  }
}

void Resolver::resolveCallee(CallOrIndex &C, Scope *S) {
  // Resolve the callee first.
  if (C.Callee) resolveExpr(*C.Callee, S);

  // Decide Call vs Index.
  if (auto *N = dynamic_cast<NameExpr *>(C.Callee)) {
    if (N->Ref) {
      switch (N->Ref->Kind) {
      case BindingKind::Var:
      case BindingKind::Param:
      case BindingKind::Output:
      case BindingKind::Global:
      case BindingKind::Persistent:
        C.Resolved = CallKind::Index;
        N->Ref->ReadFrom = true;
        return;
      case BindingKind::Function:
      case BindingKind::Builtin:
      case BindingKind::Import:
        C.Resolved = CallKind::Call;
        return;
      }
    }
    // Unknown name with a callee that looks like an identifier — treat as
    // call and let type inference report it as ambiguous.
    C.Resolved = CallKind::Call;
    return;
  }

  // Non-identifier callee: could be a function handle call, a chained index,
  // etc. Default to Call for handles, else Index.
  if (C.Callee && C.Callee->Ty &&
      C.Callee->Ty->K == Type::Kind::FuncHandle) {
    C.Resolved = CallKind::Call;
  } else {
    C.Resolved = CallKind::Index;
  }
}

void Resolver::resolveExpr(Expr &E, Scope *S) {
  switch (E.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<NameExpr &>(E);
    Binding *B = S->lookup(N.Name);
    if (!B) {
      Diag.error(N.Range.Begin,
                 std::string("undefined name '") + std::string(N.Name) + "'");
      return;
    }
    N.Ref = B;
    B->ReadFrom = true;
    break;
  }
  case NodeKind::BinaryOp: {
    auto &B = static_cast<BinaryOpExpr &>(E);
    if (B.LHS) resolveExpr(*B.LHS, S);
    if (B.RHS) resolveExpr(*B.RHS, S);
    break;
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<UnaryOpExpr &>(E);
    if (U.Operand) resolveExpr(*U.Operand, S);
    break;
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<PostfixOpExpr &>(E);
    if (P.Operand) resolveExpr(*P.Operand, S);
    break;
  }
  case NodeKind::RangeExpr: {
    auto &R = static_cast<RangeExpr &>(E);
    if (R.Start) resolveExpr(*R.Start, S);
    if (R.Step)  resolveExpr(*R.Step, S);
    if (R.End)   resolveExpr(*R.End, S);
    break;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<CallOrIndex &>(E);
    resolveCallee(C, S);
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<CellIndex &>(E);
    if (C.Callee) resolveExpr(*C.Callee, S);
    for (Expr *A : C.Args) if (A) resolveExpr(*A, S);
    break;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<FieldAccess &>(E);
    if (F.Base) resolveExpr(*F.Base, S);
    break;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<DynamicField &>(E);
    if (F.Base) resolveExpr(*F.Base, S);
    if (F.Name) resolveExpr(*F.Name, S);
    break;
  }
  case NodeKind::MatrixLiteral: {
    auto &M = static_cast<MatrixLiteral &>(E);
    for (auto &R : M.Rows)
      for (Expr *C : R) if (C) resolveExpr(*C, S);
    break;
  }
  case NodeKind::CellLiteral: {
    auto &M = static_cast<CellLiteral &>(E);
    for (auto &R : M.Rows)
      for (Expr *C : R) if (C) resolveExpr(*C, S);
    break;
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<AnonFunction &>(E);
    Scope *Inner = Sema.newScope(S, "<anon>");
    A.ParamRefs.clear();
    for (auto P : A.Params) {
      Binding *B = Sema.newBinding();
      Binding *D = Inner->declare(P, BindingKind::Param, B);
      A.ParamRefs.push_back(D);
    }
    if (A.Body) resolveExpr(*A.Body, Inner);
    break;
  }
  case NodeKind::FuncHandle: {
    auto &F = static_cast<FuncHandle &>(E);
    F.Ref = S->lookup(F.Name);
    if (!F.Ref) {
      Diag.error(F.Range.Begin,
                 std::string("undefined function '") + std::string(F.Name) +
                     "' in function handle");
    } else if (F.Ref->Kind != BindingKind::Function &&
               F.Ref->Kind != BindingKind::Builtin) {
      Diag.error(F.Range.Begin,
                 std::string("'") + std::string(F.Name) +
                     "' is not a function");
    }
    break;
  }
  // Literals and EndExpr/ColonExpr need no resolution.
  default:
    break;
  }
}

} // namespace matlab
