//===----------------------------------------------------------------------===//
// CallSiteAnalyzer — see header for design. Two public entry points:
//
//   - walkUserCalls(TU, action) — visits every CallOrIndex resolved to a
//     user-defined Function* and invokes the action callback. Shared by
//     the analyzer (Phase 1) and the Phase 3 monomorphizer.
//   - analyzeCallSites(TU) — returns a map of Function* → distinct
//     per-call-site signatures, built on top of walkUserCalls.
//===----------------------------------------------------------------------===//

#include "matlab/Sema/CallSiteAnalyzer.h"

#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

#include <algorithm>
#include <sstream>

namespace matlab {

std::string typeSignature(const Type *T) {
  if (!T) return "?";
  return T->toString();
}

CallSignature callSignatureFrom(const std::vector<const Type *> &ArgTypes) {
  CallSignature Sig;
  Sig.reserve(ArgTypes.size());
  for (const Type *T : ArgTypes) Sig.push_back(typeSignature(T));
  return Sig;
}

namespace {

// Recursive AST walker. Visits every Expr / Stmt looking for CallOrIndex
// nodes whose Resolved == Call and whose callee resolves to a user
// Function*. For each such call, invokes the action callback. Tracks
// the enclosing Function* so callers can distinguish recursive sites
// (Caller == Callee) from external ones.
class Walker {
public:
  explicit Walker(const UserCallActionWithCaller &Action) : Action(&Action) {}
  explicit Walker(const ClassCallAction &CA) : ClassAct(&CA) {}

  void visit(Function &F) {
    Function *Saved = CurrentFn;
    CurrentFn = &F;
    if (F.Body) visit(*F.Body);
    for (Function *Nested : F.Nested)
      if (Nested) visit(*Nested);
    CurrentFn = Saved;
  }

  void visit(Script &S) { if (S.Body) visit(*S.Body); }

  void visit(ClassDef &C) {
    for (Function *M : C.Methods)
      if (M) visit(*M);
    for (Function *M : C.StaticMethods)
      if (M) visit(*M);
    for (auto &Prop : C.Props)
      if (Prop.Default) visit(*Prop.Default);
  }

  void visit(Stmt &S) {
    switch (S.Kind) {
    case NodeKind::ExprStmt: {
      auto &E = static_cast<ExprStmt &>(S);
      if (E.E) visit(*E.E);
      return;
    }
    case NodeKind::AssignStmt: {
      auto &A = static_cast<AssignStmt &>(S);
      for (Expr *L : A.LHS) if (L) visit(*L);
      if (A.RHS) visit(*A.RHS);
      return;
    }
    case NodeKind::IfStmt: {
      auto &I = static_cast<IfStmt &>(S);
      if (I.Cond) visit(*I.Cond);
      if (I.Then) visit(*I.Then);
      for (auto &E : I.Elseifs) {
        if (E.Cond) visit(*E.Cond);
        if (E.Body) visit(*E.Body);
      }
      if (I.Else) visit(*I.Else);
      return;
    }
    case NodeKind::ForStmt: {
      auto &F = static_cast<ForStmt &>(S);
      if (F.Iter) visit(*F.Iter);
      if (F.Body) visit(*F.Body);
      return;
    }
    case NodeKind::WhileStmt: {
      auto &W = static_cast<WhileStmt &>(S);
      if (W.Cond) visit(*W.Cond);
      if (W.Body) visit(*W.Body);
      return;
    }
    case NodeKind::SwitchStmt: {
      auto &Sw = static_cast<SwitchStmt &>(S);
      if (Sw.Discriminant) visit(*Sw.Discriminant);
      for (auto &C : Sw.Cases) {
        if (C.Value) visit(*C.Value);
        if (C.Body) visit(*C.Body);
      }
      return;
    }
    case NodeKind::TryStmt: {
      auto &T = static_cast<TryStmt &>(S);
      if (T.TryBody) visit(*T.TryBody);
      if (T.CatchBody) visit(*T.CatchBody);
      return;
    }
    case NodeKind::Block: {
      auto &B = static_cast<Block &>(S);
      for (Stmt *Inner : B.Stmts) if (Inner) visit(*Inner);
      return;
    }
    default:
      return; // ReturnStmt / BreakStmt / ContinueStmt / declarations / etc.
    }
  }

  void visit(Expr &E) {
    switch (E.Kind) {
    case NodeKind::BinaryOp: {
      auto &B = static_cast<BinaryOpExpr &>(E);
      if (B.LHS) visit(*B.LHS);
      if (B.RHS) visit(*B.RHS);
      return;
    }
    case NodeKind::UnaryOp: {
      auto &U = static_cast<UnaryOpExpr &>(E);
      if (U.Operand) visit(*U.Operand);
      return;
    }
    case NodeKind::PostfixOp: {
      auto &P = static_cast<PostfixOpExpr &>(E);
      if (P.Operand) visit(*P.Operand);
      return;
    }
    case NodeKind::RangeExpr: {
      auto &R = static_cast<RangeExpr &>(E);
      if (R.Start) visit(*R.Start);
      if (R.Step) visit(*R.Step);
      if (R.End) visit(*R.End);
      return;
    }
    case NodeKind::CallOrIndex: {
      auto &C = static_cast<CallOrIndex &>(E);
      // Walk children first — nested calls in arg positions still need to
      // be discovered (e.g. `sq(twice(5))` records both sites).
      if (C.Callee) visit(*C.Callee);
      for (Expr *A : C.Args) if (A) visit(*A);
      visitCall(C);
      return;
    }
    case NodeKind::CellIndex: {
      auto &C = static_cast<CellIndex &>(E);
      if (C.Callee) visit(*C.Callee);
      for (Expr *A : C.Args) if (A) visit(*A);
      return;
    }
    case NodeKind::FieldAccess: {
      auto &F = static_cast<FieldAccess &>(E);
      if (F.Base) visit(*F.Base);
      return;
    }
    case NodeKind::DynamicField: {
      auto &D = static_cast<DynamicField &>(E);
      if (D.Base) visit(*D.Base);
      if (D.Name) visit(*D.Name);
      return;
    }
    case NodeKind::MatrixLiteral: {
      auto &M = static_cast<MatrixLiteral &>(E);
      for (auto &Row : M.Rows) for (Expr *Cell : Row) if (Cell) visit(*Cell);
      return;
    }
    case NodeKind::CellLiteral: {
      auto &M = static_cast<CellLiteral &>(E);
      for (auto &Row : M.Rows) for (Expr *Cell : Row) if (Cell) visit(*Cell);
      return;
    }
    case NodeKind::AnonFunction: {
      auto &A = static_cast<AnonFunction &>(E);
      if (A.Body) visit(*A.Body);
      return;
    }
    default:
      return; // literals, NameExpr, EndExpr, ColonExpr, FuncHandle
    }
  }

private:
  // Filter and dispatch to the action.
  void visitCall(CallOrIndex &C) {
    if (C.Resolved != CallKind::Call) return;
    if (Action) {
      if (auto *N = dynamic_cast<NameExpr *>(C.Callee))
        if (N->Ref && N->Ref->Kind == BindingKind::Function && N->Ref->FuncDef)
          (*Action)(CurrentFn, C, *N, *N->Ref->FuncDef);
    }
    if (ClassAct) visitClassCall(C);
  }

  // #191 P5 — surface class constructor / instance-method dispatch.
  void visitClassCall(CallOrIndex &C) {
    // Constructor: `ClassName(args)` — a NameExpr callee bound to a class.
    if (auto *N = dynamic_cast<NameExpr *>(C.Callee)) {
      if (N->Ref && N->Ref->Kind == BindingKind::Class && N->Ref->ClassDef)
        (*ClassAct)(CurrentFn, C, *N->Ref->ClassDef, N->Ref->ClassDef->Name);
      return;
    }
    // Instance method: `obj.method(args)` — a FieldAccess callee whose base
    // resolves to a class (an object<Class> type, or a PinnedClass binding),
    // and where some class up the Super chain defines that method.
    if (auto *FA = dynamic_cast<FieldAccess *>(C.Callee)) {
      const ClassDef *Recv = recvClassOf(FA->Base);
      if (Recv && chainHasMethod(Recv, FA->Field))
        (*ClassAct)(CurrentFn, C, *Recv, FA->Field);
    }
  }

  static const ClassDef *recvClassOf(const Expr *Base) {
    if (!Base) return nullptr;
    if (auto *N = dynamic_cast<const NameExpr *>(Base))
      if (N->Ref && N->Ref->PinnedClass) return N->Ref->PinnedClass;
    if (Base->Ty && Base->Ty->K == Type::Kind::Object)
      return static_cast<const ObjectType &>(*Base->Ty).Class;
    return nullptr;
  }
  static bool chainHasMethod(const ClassDef *CD, std::string_view Name) {
    for (const ClassDef *CC = CD; CC; CC = CC->Super)
      for (const Function *M : CC->Methods)
        if (M && M->Name == Name) return true;
    return false;
  }

  const UserCallActionWithCaller *Action = nullptr;
  const ClassCallAction *ClassAct = nullptr;
  Function *CurrentFn = nullptr;
};

} // namespace

void walkUserCallsWithCaller(TranslationUnit &TU,
                             const UserCallActionWithCaller &Action) {
  Walker W(Action);
  if (TU.ScriptNode) W.visit(*TU.ScriptNode);
  for (Function *F : TU.Functions)
    if (F) W.visit(*F);
  for (ClassDef *C : TU.Classes)
    if (C) W.visit(*C);
}

void walkClassCallsWithCaller(TranslationUnit &TU,
                              const ClassCallAction &Action) {
  Walker W(Action);
  if (TU.ScriptNode) W.visit(*TU.ScriptNode);
  for (Function *F : TU.Functions)
    if (F) W.visit(*F);
  for (ClassDef *C : TU.Classes)
    if (C) W.visit(*C);
}

void walkUserCalls(TranslationUnit &TU, const UserCallAction &Action) {
  walkUserCallsWithCaller(
      TU, [&](Function * /*Caller*/, CallOrIndex &C, NameExpr &N,
              Function &F) { Action(C, N, F); });
}

CallSiteAnalysis analyzeCallSites(TranslationUnit &TU) {
  CallSiteAnalysis Out;
  walkUserCalls(TU, [&](CallOrIndex &C, NameExpr & /*N*/, Function &F) {
    Out.Sigs[&F].insert(callSignatureFrom(C.ArgTypes));
  });
  return Out;
}

std::string CallSiteAnalysis::dump() const {
  // Order callees alphabetically by name; ties resolved by pointer value so
  // anonymous helpers (no name) still sort deterministically.
  std::vector<Function *> Order;
  Order.reserve(Sigs.size());
  for (auto &KV : Sigs) Order.push_back(KV.first);
  std::sort(Order.begin(), Order.end(), [](Function *A, Function *B) {
    if (A->Name != B->Name) return A->Name < B->Name;
    return A < B;
  });

  std::ostringstream OS;
  for (Function *F : Order) {
    OS << F->Name << ":\n";
    const auto &Bucket = Sigs.at(F);
    if (Bucket.empty()) {
      OS << "  (no recorded call sites)\n";
      continue;
    }
    for (const CallSignature &Sig : Bucket) {
      OS << "  (";
      for (size_t I = 0; I < Sig.size(); ++I) {
        if (I) OS << ", ";
        OS << Sig[I];
      }
      OS << ")\n";
    }
  }
  return OS.str();
}

} // namespace matlab
