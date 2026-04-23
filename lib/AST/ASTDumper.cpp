#include "matlab/AST/ASTDumper.h"

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

  void pad(unsigned Indent) {
    for (unsigned I = 0; I < Indent; ++I) OS << "  ";
  }

  void dumpExpr(const Expr &E, unsigned Indent);
  void dumpStmt(const Stmt &S, unsigned Indent);
  void dumpBlock(const Block &B, unsigned Indent);
};

void Dumper::dumpExpr(const Expr &E, unsigned Indent) {
  pad(Indent);
  switch (E.Kind) {
  case NodeKind::IntegerLiteral: {
    auto &N = static_cast<const IntegerLiteral &>(E);
    OS << "IntegerLiteral " << N.Text << "\n";
    break;
  }
  case NodeKind::FPLiteral: {
    auto &N = static_cast<const FPLiteral &>(E);
    OS << "FPLiteral " << N.Text << "\n";
    break;
  }
  case NodeKind::ImagLiteral: {
    auto &N = static_cast<const ImagLiteral &>(E);
    OS << "ImagLiteral " << N.Text << "\n";
    break;
  }
  case NodeKind::StringLiteral: {
    auto &N = static_cast<const StringLiteral &>(E);
    OS << "StringLiteral \"" << N.Value << "\"\n";
    break;
  }
  case NodeKind::CharLiteral: {
    auto &N = static_cast<const CharLiteral &>(E);
    OS << "CharLiteral '" << N.Value << "'\n";
    break;
  }
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(E);
    OS << "NameExpr " << N.Name << "\n";
    break;
  }
  case NodeKind::EndExpr:    OS << "EndExpr\n"; break;
  case NodeKind::ColonExpr:  OS << "ColonExpr\n"; break;
  case NodeKind::BinaryOp: {
    auto &N = static_cast<const BinaryOpExpr &>(E);
    OS << "BinaryOp " << binOpName(N.Op) << "\n";
    if (N.LHS) dumpExpr(*N.LHS, Indent + 1);
    if (N.RHS) dumpExpr(*N.RHS, Indent + 1);
    break;
  }
  case NodeKind::UnaryOp: {
    auto &N = static_cast<const UnaryOpExpr &>(E);
    OS << "UnaryOp " << unOpName(N.Op) << "\n";
    if (N.Operand) dumpExpr(*N.Operand, Indent + 1);
    break;
  }
  case NodeKind::PostfixOp: {
    auto &N = static_cast<const PostfixOpExpr &>(E);
    OS << "PostfixOp " << postfixName(N.Op) << "\n";
    if (N.Operand) dumpExpr(*N.Operand, Indent + 1);
    break;
  }
  case NodeKind::RangeExpr: {
    auto &N = static_cast<const RangeExpr &>(E);
    OS << "RangeExpr\n";
    if (N.Start) dumpExpr(*N.Start, Indent + 1);
    if (N.Step)  dumpExpr(*N.Step,  Indent + 1);
    if (N.End)   dumpExpr(*N.End,   Indent + 1);
    break;
  }
  case NodeKind::CallOrIndex: {
    auto &N = static_cast<const CallOrIndex &>(E);
    OS << "CallOrIndex\n";
    if (N.Callee) dumpExpr(*N.Callee, Indent + 1);
    for (auto *A : N.Args) if (A) dumpExpr(*A, Indent + 1);
    break;
  }
  case NodeKind::CellIndex: {
    auto &N = static_cast<const CellIndex &>(E);
    OS << "CellIndex\n";
    if (N.Callee) dumpExpr(*N.Callee, Indent + 1);
    for (auto *A : N.Args) if (A) dumpExpr(*A, Indent + 1);
    break;
  }
  case NodeKind::FieldAccess: {
    auto &N = static_cast<const FieldAccess &>(E);
    OS << "FieldAccess ." << N.Field << "\n";
    if (N.Base) dumpExpr(*N.Base, Indent + 1);
    break;
  }
  case NodeKind::DynamicField: {
    auto &N = static_cast<const DynamicField &>(E);
    OS << "DynamicField\n";
    if (N.Base) dumpExpr(*N.Base, Indent + 1);
    if (N.Name) dumpExpr(*N.Name, Indent + 1);
    break;
  }
  case NodeKind::MatrixLiteral: {
    auto &N = static_cast<const MatrixLiteral &>(E);
    OS << "MatrixLiteral rows=" << N.Rows.size() << "\n";
    for (size_t R = 0; R < N.Rows.size(); ++R) {
      pad(Indent + 1);
      OS << "Row " << R << "\n";
      for (auto *C : N.Rows[R]) if (C) dumpExpr(*C, Indent + 2);
    }
    break;
  }
  case NodeKind::CellLiteral: {
    auto &N = static_cast<const CellLiteral &>(E);
    OS << "CellLiteral rows=" << N.Rows.size() << "\n";
    for (size_t R = 0; R < N.Rows.size(); ++R) {
      pad(Indent + 1);
      OS << "Row " << R << "\n";
      for (auto *C : N.Rows[R]) if (C) dumpExpr(*C, Indent + 2);
    }
    break;
  }
  case NodeKind::AnonFunction: {
    auto &N = static_cast<const AnonFunction &>(E);
    OS << "AnonFunction params=(";
    for (size_t I = 0; I < N.Params.size(); ++I) {
      if (I) OS << ", ";
      OS << N.Params[I];
    }
    OS << ")\n";
    if (N.Body) dumpExpr(*N.Body, Indent + 1);
    break;
  }
  case NodeKind::FuncHandle: {
    auto &N = static_cast<const FuncHandle &>(E);
    OS << "FuncHandle @" << N.Name << "\n";
    break;
  }
  default:
    OS << "<?expr " << nodeKindName(E.Kind) << ">\n";
  }
}

void Dumper::dumpBlock(const Block &B, unsigned Indent) {
  pad(Indent);
  OS << "Block\n";
  for (auto *S : B.Stmts) if (S) dumpStmt(*S, Indent + 1);
}

void Dumper::dumpStmt(const Stmt &S, unsigned Indent) {
  pad(Indent);
  switch (S.Kind) {
  case NodeKind::ExprStmt: {
    auto &N = static_cast<const ExprStmt &>(S);
    OS << "ExprStmt" << (N.Suppressed ? " suppressed" : "") << "\n";
    if (N.E) dumpExpr(*N.E, Indent + 1);
    break;
  }
  case NodeKind::AssignStmt: {
    auto &N = static_cast<const AssignStmt &>(S);
    OS << "AssignStmt" << (N.Suppressed ? " suppressed" : "") << "\n";
    pad(Indent + 1); OS << "LHS\n";
    for (auto *L : N.LHS) if (L) dumpExpr(*L, Indent + 2);
    pad(Indent + 1); OS << "RHS\n";
    if (N.RHS) dumpExpr(*N.RHS, Indent + 2);
    break;
  }
  case NodeKind::IfStmt: {
    auto &N = static_cast<const IfStmt &>(S);
    OS << "IfStmt\n";
    pad(Indent + 1); OS << "Cond\n";
    if (N.Cond) dumpExpr(*N.Cond, Indent + 2);
    pad(Indent + 1); OS << "Then\n";
    if (N.Then) dumpBlock(*N.Then, Indent + 2);
    for (auto &EI : N.Elseifs) {
      pad(Indent + 1); OS << "ElseIf\n";
      if (EI.Cond) dumpExpr(*EI.Cond, Indent + 2);
      if (EI.Body) dumpBlock(*EI.Body, Indent + 2);
    }
    if (N.Else) {
      pad(Indent + 1); OS << "Else\n";
      dumpBlock(*N.Else, Indent + 2);
    }
    break;
  }
  case NodeKind::ForStmt: {
    auto &N = static_cast<const ForStmt &>(S);
    OS << (N.IsParfor ? "ParforStmt " : "ForStmt ") << N.Var << "\n";
    pad(Indent + 1); OS << "Iter\n";
    if (N.Iter) dumpExpr(*N.Iter, Indent + 2);
    if (N.Body) dumpBlock(*N.Body, Indent + 1);
    break;
  }
  case NodeKind::WhileStmt: {
    auto &N = static_cast<const WhileStmt &>(S);
    OS << "WhileStmt\n";
    pad(Indent + 1); OS << "Cond\n";
    if (N.Cond) dumpExpr(*N.Cond, Indent + 2);
    if (N.Body) dumpBlock(*N.Body, Indent + 1);
    break;
  }
  case NodeKind::SwitchStmt: {
    auto &N = static_cast<const SwitchStmt &>(S);
    OS << "SwitchStmt\n";
    pad(Indent + 1); OS << "Disc\n";
    if (N.Discriminant) dumpExpr(*N.Discriminant, Indent + 2);
    for (auto &C : N.Cases) {
      pad(Indent + 1);
      OS << (C.Value ? "Case\n" : "Otherwise\n");
      if (C.Value) dumpExpr(*C.Value, Indent + 2);
      if (C.Body) dumpBlock(*C.Body, Indent + 2);
    }
    break;
  }
  case NodeKind::TryStmt: {
    auto &N = static_cast<const TryStmt &>(S);
    OS << "TryStmt";
    if (!N.CatchVar.empty()) OS << " catch-as " << N.CatchVar;
    OS << "\n";
    if (N.TryBody) dumpBlock(*N.TryBody, Indent + 1);
    if (N.CatchBody) {
      pad(Indent + 1); OS << "Catch\n";
      dumpBlock(*N.CatchBody, Indent + 2);
    }
    break;
  }
  case NodeKind::ReturnStmt:   OS << "ReturnStmt\n"; break;
  case NodeKind::BreakStmt:    OS << "BreakStmt\n"; break;
  case NodeKind::ContinueStmt: OS << "ContinueStmt\n"; break;
  case NodeKind::GlobalDecl: {
    auto &N = static_cast<const GlobalDecl &>(S);
    OS << "GlobalDecl";
    for (auto &Nm : N.Names) OS << ' ' << Nm;
    OS << "\n";
    break;
  }
  case NodeKind::PersistentDecl: {
    auto &N = static_cast<const PersistentDecl &>(S);
    OS << "PersistentDecl";
    for (auto &Nm : N.Names) OS << ' ' << Nm;
    OS << "\n";
    break;
  }
  case NodeKind::ImportStmt: {
    auto &N = static_cast<const ImportStmt &>(S);
    OS << "ImportStmt ";
    for (size_t I = 0; I < N.Path.size(); ++I) {
      if (I) OS << '.';
      OS << N.Path[I];
    }
    if (N.Wildcard) OS << ".*";
    OS << "\n";
    break;
  }
  case NodeKind::CommandStmt: {
    auto &N = static_cast<const CommandStmt &>(S);
    OS << "CommandStmt " << N.Name;
    for (auto &A : N.Args) OS << " '" << A << "'";
    if (N.Suppressed) OS << " suppressed";
    OS << "\n";
    break;
  }
  case NodeKind::Block: {
    dumpBlock(static_cast<const Block &>(S), Indent);
    break;
  }
  default:
    OS << "<?stmt " << nodeKindName(S.Kind) << ">\n";
  }
}

void Dumper::dump(const Node &N, unsigned Indent) {
  switch (N.Kind) {
  case NodeKind::TranslationUnit: {
    auto &T = static_cast<const TranslationUnit &>(N);
    pad(Indent); OS << "TranslationUnit\n";
    if (T.ScriptNode) dump(*T.ScriptNode, Indent + 1);
    for (auto *C : T.Classes) if (C) dump(*C, Indent + 1);
    for (auto *F : T.Functions) if (F) dump(*F, Indent + 1);
    break;
  }
  case NodeKind::ClassDef: {
    auto &C = static_cast<const ClassDef &>(N);
    pad(Indent); OS << "ClassDef " << C.Name;
    if (!C.SuperName.empty()) OS << " < " << C.SuperName;
    OS << "\n";
    for (auto &P : C.Props) {
      pad(Indent + 1); OS << "Property " << P.Name << "\n";
      if (P.Default) dumpExpr(*P.Default, Indent + 2);
    }
    for (auto *F : C.Methods) if (F) dump(*F, Indent + 1);
    for (auto *F : C.StaticMethods) {
      if (!F) continue;
      pad(Indent + 1); OS << "Static\n";
      dump(*F, Indent + 2);
    }
    break;
  }
  case NodeKind::Script: {
    auto &S = static_cast<const Script &>(N);
    pad(Indent); OS << "Script\n";
    if (S.Body) dumpBlock(*S.Body, Indent + 1);
    break;
  }
  case NodeKind::Function: {
    auto &F = static_cast<const Function &>(N);
    pad(Indent); OS << "Function " << F.Name << " in=(";
    for (size_t I = 0; I < F.Inputs.size(); ++I) {
      if (I) OS << ", ";
      OS << F.Inputs[I];
    }
    OS << ") out=(";
    for (size_t I = 0; I < F.Outputs.size(); ++I) {
      if (I) OS << ", ";
      OS << F.Outputs[I];
    }
    OS << ")\n";
    if (F.Body) dumpBlock(*F.Body, Indent + 1);
    for (auto *N2 : F.Nested) {
      pad(Indent + 1); OS << "Nested\n";
      dump(*N2, Indent + 2);
    }
    break;
  }
  default:
    if (auto *S = dynamic_cast<const Stmt *>(&N)) { dumpStmt(*S, Indent); break; }
    if (auto *E = dynamic_cast<const Expr *>(&N)) { dumpExpr(*E, Indent); break; }
    pad(Indent); OS << "<?node " << nodeKindName(N.Kind) << ">\n";
  }
}

} // namespace

void dumpAST(std::ostream &OS, const Node &N) {
  Dumper D(OS);
  D.dump(N);
}

} // namespace matlab
