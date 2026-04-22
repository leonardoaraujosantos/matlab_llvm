#include "matlab/Sema/SemaDumper.h"

#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

#include <ostream>
#include <string>

namespace matlab {

namespace {

class Dumper {
public:
  explicit Dumper(std::ostream &OS) : OS(OS) {}

  void dump(const Node &N, unsigned Indent = 0);

private:
  std::ostream &OS;

  void pad(unsigned I) { for (unsigned i = 0; i < I; ++i) OS << "  "; }

  static std::string tyStr(const Expr &E) {
    if (!E.Ty) return "?";
    return E.Ty->toString();
  }

  void dumpExpr(const Expr &E, unsigned I);
  void dumpStmt(const Stmt &S, unsigned I);
  void dumpBlock(const Block &B, unsigned I);
};

void Dumper::dumpExpr(const Expr &E, unsigned I) {
  pad(I);
  switch (E.Kind) {
  case NodeKind::IntegerLiteral:
    OS << "IntLit " << static_cast<const IntegerLiteral &>(E).Text
       << " [" << tyStr(E) << "]\n";
    break;
  case NodeKind::FPLiteral:
    OS << "FPLit " << static_cast<const FPLiteral &>(E).Text
       << " [" << tyStr(E) << "]\n";
    break;
  case NodeKind::ImagLiteral:
    OS << "ImagLit " << static_cast<const ImagLiteral &>(E).Text
       << " [" << tyStr(E) << "]\n";
    break;
  case NodeKind::StringLiteral:
    OS << "StrLit \"" << static_cast<const StringLiteral &>(E).Value
       << "\" [" << tyStr(E) << "]\n";
    break;
  case NodeKind::CharLiteral:
    OS << "CharLit '" << static_cast<const CharLiteral &>(E).Value
       << "' [" << tyStr(E) << "]\n";
    break;
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(E);
    OS << "Name " << N.Name << " ("
       << (N.Ref ? bindingKindName(N.Ref->Kind) : "?")
       << ") [" << tyStr(E) << "]\n";
    break;
  }
  case NodeKind::EndExpr:   OS << "End [" << tyStr(E) << "]\n"; break;
  case NodeKind::ColonExpr: OS << "Colon [" << tyStr(E) << "]\n"; break;
  case NodeKind::BinaryOp: {
    auto &B = static_cast<const BinaryOpExpr &>(E);
    OS << "BinOp " << binOpName(B.Op) << " [" << tyStr(E) << "]\n";
    if (B.LHS) dumpExpr(*B.LHS, I + 1);
    if (B.RHS) dumpExpr(*B.RHS, I + 1);
    break;
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(E);
    OS << "UnaryOp " << unOpName(U.Op) << " [" << tyStr(E) << "]\n";
    if (U.Operand) dumpExpr(*U.Operand, I + 1);
    break;
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<const PostfixOpExpr &>(E);
    OS << "Postfix " << postfixName(P.Op) << " [" << tyStr(E) << "]\n";
    if (P.Operand) dumpExpr(*P.Operand, I + 1);
    break;
  }
  case NodeKind::RangeExpr: {
    auto &R = static_cast<const RangeExpr &>(E);
    OS << "Range [" << tyStr(E) << "]\n";
    if (R.Start) dumpExpr(*R.Start, I + 1);
    if (R.Step)  dumpExpr(*R.Step,  I + 1);
    if (R.End)   dumpExpr(*R.End,   I + 1);
    break;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(E);
    const char *Kind = C.Resolved == CallKind::Call  ? "Call"
                     : C.Resolved == CallKind::Index ? "Index"
                                                     : "CallOrIndex";
    OS << Kind << " [" << tyStr(E) << "]\n";
    if (C.Callee) dumpExpr(*C.Callee, I + 1);
    for (auto *A : C.Args) if (A) dumpExpr(*A, I + 1);
    break;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<const CellIndex &>(E);
    OS << "CellIndex [" << tyStr(E) << "]\n";
    if (C.Callee) dumpExpr(*C.Callee, I + 1);
    for (auto *A : C.Args) if (A) dumpExpr(*A, I + 1);
    break;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<const FieldAccess &>(E);
    OS << "Field ." << F.Field << " [" << tyStr(E) << "]\n";
    if (F.Base) dumpExpr(*F.Base, I + 1);
    break;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<const DynamicField &>(E);
    OS << "DynamicField [" << tyStr(E) << "]\n";
    if (F.Base) dumpExpr(*F.Base, I + 1);
    if (F.Name) dumpExpr(*F.Name, I + 1);
    break;
  }
  case NodeKind::MatrixLiteral: {
    auto &M = static_cast<const MatrixLiteral &>(E);
    OS << "MatrixLit rows=" << M.Rows.size() << " [" << tyStr(E) << "]\n";
    for (size_t R = 0; R < M.Rows.size(); ++R) {
      pad(I + 1); OS << "Row " << R << "\n";
      for (auto *C : M.Rows[R]) if (C) dumpExpr(*C, I + 2);
    }
    break;
  }
  case NodeKind::CellLiteral: {
    auto &M = static_cast<const CellLiteral &>(E);
    OS << "CellLit rows=" << M.Rows.size() << " [" << tyStr(E) << "]\n";
    for (size_t R = 0; R < M.Rows.size(); ++R) {
      pad(I + 1); OS << "Row " << R << "\n";
      for (auto *C : M.Rows[R]) if (C) dumpExpr(*C, I + 2);
    }
    break;
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<const AnonFunction &>(E);
    OS << "AnonFn params=(";
    for (size_t i = 0; i < A.Params.size(); ++i) {
      if (i) OS << ", ";
      OS << A.Params[i];
    }
    OS << ") [" << tyStr(E) << "]\n";
    if (A.Body) dumpExpr(*A.Body, I + 1);
    break;
  }
  case NodeKind::FuncHandle: {
    auto &F = static_cast<const FuncHandle &>(E);
    OS << "Handle @" << F.Name << " ("
       << (F.Ref ? bindingKindName(F.Ref->Kind) : "?") << ") ["
       << tyStr(E) << "]\n";
    break;
  }
  default:
    OS << "?expr " << nodeKindName(E.Kind) << "\n";
  }
}

void Dumper::dumpBlock(const Block &B, unsigned I) {
  pad(I); OS << "Block\n";
  for (auto *S : B.Stmts) if (S) dumpStmt(*S, I + 1);
}

void Dumper::dumpStmt(const Stmt &S, unsigned I) {
  pad(I);
  switch (S.Kind) {
  case NodeKind::ExprStmt: {
    auto &N = static_cast<const ExprStmt &>(S);
    OS << "ExprStmt" << (N.Suppressed ? " suppressed" : "") << "\n";
    if (N.E) dumpExpr(*N.E, I + 1);
    break;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<const AssignStmt &>(S);
    OS << "AssignStmt" << (A.Suppressed ? " suppressed" : "") << "\n";
    pad(I + 1); OS << "LHS\n";
    for (auto *L : A.LHS) if (L) dumpExpr(*L, I + 2);
    pad(I + 1); OS << "RHS\n";
    if (A.RHS) dumpExpr(*A.RHS, I + 2);
    break;
  }
  case NodeKind::IfStmt: {
    auto &N = static_cast<const IfStmt &>(S);
    OS << "IfStmt\n";
    pad(I + 1); OS << "Cond\n";
    if (N.Cond) dumpExpr(*N.Cond, I + 2);
    pad(I + 1); OS << "Then\n";
    if (N.Then) dumpBlock(*N.Then, I + 2);
    for (auto &EI : N.Elseifs) {
      pad(I + 1); OS << "ElseIf\n";
      if (EI.Cond) dumpExpr(*EI.Cond, I + 2);
      if (EI.Body) dumpBlock(*EI.Body, I + 2);
    }
    if (N.Else) {
      pad(I + 1); OS << "Else\n";
      dumpBlock(*N.Else, I + 2);
    }
    break;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<const ForStmt &>(S);
    OS << (F.IsParfor ? "ParforStmt " : "ForStmt ") << F.Var << "\n";
    pad(I + 1); OS << "Iter\n";
    if (F.Iter) dumpExpr(*F.Iter, I + 2);
    if (F.Body) dumpBlock(*F.Body, I + 1);
    break;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<const WhileStmt &>(S);
    OS << "WhileStmt\n";
    pad(I + 1); OS << "Cond\n";
    if (W.Cond) dumpExpr(*W.Cond, I + 2);
    if (W.Body) dumpBlock(*W.Body, I + 1);
    break;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<const SwitchStmt &>(S);
    OS << "SwitchStmt\n";
    pad(I + 1); OS << "Disc\n";
    if (Sw.Discriminant) dumpExpr(*Sw.Discriminant, I + 2);
    for (auto &C : Sw.Cases) {
      pad(I + 1); OS << (C.Value ? "Case\n" : "Otherwise\n");
      if (C.Value) dumpExpr(*C.Value, I + 2);
      if (C.Body)  dumpBlock(*C.Body, I + 2);
    }
    break;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<const TryStmt &>(S);
    OS << "TryStmt";
    if (!T.CatchVar.empty()) OS << " catch-as " << T.CatchVar;
    OS << "\n";
    if (T.TryBody) dumpBlock(*T.TryBody, I + 1);
    if (T.CatchBody) {
      pad(I + 1); OS << "Catch\n";
      dumpBlock(*T.CatchBody, I + 2);
    }
    break;
  }
  case NodeKind::ReturnStmt:   OS << "ReturnStmt\n"; break;
  case NodeKind::BreakStmt:    OS << "BreakStmt\n"; break;
  case NodeKind::ContinueStmt: OS << "ContinueStmt\n"; break;
  case NodeKind::GlobalDecl: {
    auto &G = static_cast<const GlobalDecl &>(S);
    OS << "GlobalDecl";
    for (auto &N : G.Names) OS << ' ' << N;
    OS << "\n";
    break;
  }
  case NodeKind::PersistentDecl: {
    auto &P = static_cast<const PersistentDecl &>(S);
    OS << "PersistentDecl";
    for (auto &N : P.Names) OS << ' ' << N;
    OS << "\n";
    break;
  }
  case NodeKind::CommandStmt: {
    auto &C = static_cast<const CommandStmt &>(S);
    OS << "CommandStmt " << C.Name;
    for (auto &A : C.Args) OS << " '" << A << "'";
    OS << "\n";
    break;
  }
  case NodeKind::Block:
    dumpBlock(static_cast<const Block &>(S), I);
    break;
  default:
    OS << "?stmt " << nodeKindName(S.Kind) << "\n";
  }
}

void Dumper::dump(const Node &N, unsigned I) {
  switch (N.Kind) {
  case NodeKind::TranslationUnit: {
    auto &T = static_cast<const TranslationUnit &>(N);
    pad(I); OS << "TranslationUnit\n";
    if (T.ScriptNode) dump(*T.ScriptNode, I + 1);
    for (auto *F : T.Functions) if (F) dump(*F, I + 1);
    break;
  }
  case NodeKind::Script: {
    auto &S = static_cast<const Script &>(N);
    pad(I); OS << "Script\n";
    if (S.Body) dumpBlock(*S.Body, I + 1);
    break;
  }
  case NodeKind::Function: {
    auto &F = static_cast<const Function &>(N);
    pad(I); OS << "Function " << F.Name << "\n";
    pad(I + 1); OS << "Inputs:";
    for (size_t k = 0; k < F.Inputs.size(); ++k) {
      const Type *T = k < F.ParamRefs.size() && F.ParamRefs[k]
                       ? F.ParamRefs[k]->InferredType : nullptr;
      OS << ' ' << F.Inputs[k] << "[" << (T ? T->toString() : "?") << "]";
    }
    OS << "\n";
    pad(I + 1); OS << "Outputs:";
    for (size_t k = 0; k < F.Outputs.size(); ++k) {
      const Type *T = k < F.OutputRefs.size() && F.OutputRefs[k]
                       ? F.OutputRefs[k]->InferredType : nullptr;
      OS << ' ' << F.Outputs[k] << "[" << (T ? T->toString() : "?") << "]";
    }
    OS << "\n";
    if (F.FnScope) {
      pad(I + 1); OS << "Locals:";
      std::vector<std::string> Names;
      for (auto &[K, B] : F.FnScope->locals()) {
        if (B->Kind != BindingKind::Var) continue;
        Names.push_back(K + "[" +
                        (B->InferredType ? B->InferredType->toString() : "?") +
                        "]");
      }
      std::sort(Names.begin(), Names.end());
      for (auto &N : Names) OS << ' ' << N;
      OS << "\n";
    }
    if (F.Body) dumpBlock(*F.Body, I + 1);
    for (auto *N2 : F.Nested) {
      pad(I + 1); OS << "Nested\n";
      dump(*N2, I + 2);
    }
    break;
  }
  default:
    if (auto *S = dynamic_cast<const Stmt *>(&N)) { dumpStmt(*S, I); break; }
    if (auto *E = dynamic_cast<const Expr *>(&N)) { dumpExpr(*E, I); break; }
    pad(I); OS << "?node " << nodeKindName(N.Kind) << "\n";
  }
}

} // namespace

void dumpSema(std::ostream &OS, const Node &N) {
  Dumper D(OS);
  D.dump(N);
}

} // namespace matlab
