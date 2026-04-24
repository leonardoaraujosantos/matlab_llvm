#include "matlab/AST/Formatter.h"

#include <ostream>
#include <string>
#include <string_view>

namespace matlab {

namespace {

/* Pretty-print operator precedence — lower binds tighter in output.
 * We emit parens whenever a child expression's precedence is strictly
 * lower than the parent's, preserving the parsed structure even when
 * a different grouping would produce the same numeric result. */
int binOpPrec(BinOp Op) {
  switch (Op) {
  case BinOp::ShortOr:  return 1;
  case BinOp::ShortAnd: return 2;
  case BinOp::Or:       return 3;
  case BinOp::And:      return 4;
  case BinOp::Eq: case BinOp::Ne:
  case BinOp::Lt: case BinOp::Le:
  case BinOp::Gt: case BinOp::Ge: return 5;
  case BinOp::Add: case BinOp::Sub: return 6;
  case BinOp::Mul: case BinOp::Div:
  case BinOp::LeftDiv:
  case BinOp::ElemMul: case BinOp::ElemDiv:
  case BinOp::ElemLeftDiv: return 7;
  case BinOp::Pow: case BinOp::ElemPow: return 8;
  }
  return 0;
}

const char *binOpLex(BinOp Op) {
  switch (Op) {
  case BinOp::Add:         return "+";
  case BinOp::Sub:         return "-";
  case BinOp::Mul:         return "*";
  case BinOp::Div:         return "/";
  case BinOp::LeftDiv:     return "\\";
  case BinOp::Pow:         return "^";
  case BinOp::ElemMul:     return ".*";
  case BinOp::ElemDiv:     return "./";
  case BinOp::ElemLeftDiv: return ".\\";
  case BinOp::ElemPow:     return ".^";
  case BinOp::Eq:          return "==";
  case BinOp::Ne:          return "~=";
  case BinOp::Lt:          return "<";
  case BinOp::Le:          return "<=";
  case BinOp::Gt:          return ">";
  case BinOp::Ge:          return ">=";
  case BinOp::And:         return "&";
  case BinOp::Or:          return "|";
  case BinOp::ShortAnd:    return "&&";
  case BinOp::ShortOr:     return "||";
  }
  return "?";
}

const char *unOpLex(UnOp Op) {
  switch (Op) {
  case UnOp::Plus:  return "+";
  case UnOp::Minus: return "-";
  case UnOp::Not:   return "~";
  }
  return "?";
}

const char *postfixLex(PostfixOp Op) {
  switch (Op) {
  case PostfixOp::CTranspose: return "'";
  case PostfixOp::Transpose:  return ".'";
  }
  return "?";
}

class Formatter {
public:
  explicit Formatter(std::ostream &OS) : OS(OS) {}
  void run(const TranslationUnit &TU);

private:
  std::ostream &OS;
  unsigned Depth = 0;

  void pad() { for (unsigned i = 0; i < Depth; ++i) OS << "    "; }

  /* Expressions. `ParentPrec` is the binding level of the enclosing
   * operator; sub-expressions below that level get wrapped in parens. */
  void emitExpr(const Expr &E, int ParentPrec = 0);
  void emitLiteral(const Expr &E);
  void emitCallArgs(const std::vector<Expr *> &Args);
  void emitMatrixLiteral(const MatrixLiteral &M);
  void emitCellLiteral(const CellLiteral &M);
  void emitAnonFunction(const AnonFunction &A);
  void emitRange(const RangeExpr &R);

  /* Statements / blocks. */
  void emitBlock(const Block &B);
  void emitStmt(const Stmt &S);
  void emitIf(const IfStmt &I);
  void emitFor(const ForStmt &F);
  void emitWhile(const WhileStmt &W);
  void emitSwitch(const SwitchStmt &Sw);
  void emitTry(const TryStmt &T);
  void emitAssign(const AssignStmt &A);
  void emitExprStmt(const ExprStmt &E);
  void emitGlobal(const GlobalDecl &G);
  void emitPersistent(const PersistentDecl &P);
  void emitCommand(const CommandStmt &C);

  /* Top-level. */
  void emitFunction(const Function &F);
  void emitClassDef(const ClassDef &C);

  /* Emit the suffix — `;` when suppressed, nothing otherwise — then a
   * newline. Statement-level entry point. */
  void endStmt(bool Suppressed) {
    if (Suppressed) OS << ';';
    OS << '\n';
  }

  static std::string quoteDouble(std::string_view S) {
    std::string R;
    R.reserve(S.size() + 2);
    R.push_back('"');
    for (char c : S) {
      if (c == '"') { R += "\"\""; continue; }
      R.push_back(c);
    }
    R.push_back('"');
    return R;
  }
  static std::string quoteSingle(std::string_view S) {
    std::string R;
    R.reserve(S.size() + 2);
    R.push_back('\'');
    for (char c : S) {
      if (c == '\'') { R += "''"; continue; }
      R.push_back(c);
    }
    R.push_back('\'');
    return R;
  }
};

void Formatter::emitLiteral(const Expr &E) {
  switch (E.Kind) {
  case NodeKind::IntegerLiteral:
    OS << static_cast<const IntegerLiteral &>(E).Text;
    return;
  case NodeKind::FPLiteral:
    OS << static_cast<const FPLiteral &>(E).Text;
    return;
  case NodeKind::ImagLiteral:
    OS << static_cast<const ImagLiteral &>(E).Text;
    return;
  case NodeKind::StringLiteral:
    OS << quoteDouble(static_cast<const StringLiteral &>(E).Value);
    return;
  case NodeKind::CharLiteral:
    OS << quoteSingle(static_cast<const CharLiteral &>(E).Value);
    return;
  case NodeKind::NameExpr:
    OS << static_cast<const NameExpr &>(E).Name;
    return;
  case NodeKind::EndExpr:
    OS << "end";
    return;
  case NodeKind::ColonExpr:
    OS << ":";
    return;
  default:
    break;
  }
}

void Formatter::emitCallArgs(const std::vector<Expr *> &Args) {
  OS << '(';
  for (size_t i = 0; i < Args.size(); ++i) {
    if (i) OS << ", ";
    if (Args[i]) emitExpr(*Args[i]);
  }
  OS << ')';
}

void Formatter::emitMatrixLiteral(const MatrixLiteral &M) {
  OS << '[';
  for (size_t r = 0; r < M.Rows.size(); ++r) {
    if (r) OS << "; ";
    const auto &Row = M.Rows[r];
    for (size_t c = 0; c < Row.size(); ++c) {
      if (c) OS << ' ';
      if (Row[c]) emitExpr(*Row[c]);
    }
  }
  OS << ']';
}

void Formatter::emitCellLiteral(const CellLiteral &M) {
  OS << '{';
  for (size_t r = 0; r < M.Rows.size(); ++r) {
    if (r) OS << "; ";
    const auto &Row = M.Rows[r];
    for (size_t c = 0; c < Row.size(); ++c) {
      if (c) OS << ", ";
      if (Row[c]) emitExpr(*Row[c]);
    }
  }
  OS << '}';
}

void Formatter::emitAnonFunction(const AnonFunction &A) {
  OS << "@(";
  for (size_t i = 0; i < A.Params.size(); ++i) {
    if (i) OS << ", ";
    OS << A.Params[i];
  }
  OS << ") ";
  if (A.Body) emitExpr(*A.Body);
}

void Formatter::emitRange(const RangeExpr &R) {
  if (R.Start) emitExpr(*R.Start);
  OS << ':';
  if (R.Step) { emitExpr(*R.Step); OS << ':'; }
  if (R.End) emitExpr(*R.End);
}

void Formatter::emitExpr(const Expr &E, int ParentPrec) {
  switch (E.Kind) {
  case NodeKind::IntegerLiteral:
  case NodeKind::FPLiteral:
  case NodeKind::ImagLiteral:
  case NodeKind::StringLiteral:
  case NodeKind::CharLiteral:
  case NodeKind::NameExpr:
  case NodeKind::EndExpr:
  case NodeKind::ColonExpr:
    emitLiteral(E);
    return;
  case NodeKind::BinaryOp: {
    auto &B = static_cast<const BinaryOpExpr &>(E);
    int P = binOpPrec(B.Op);
    bool Paren = P < ParentPrec;
    if (Paren) OS << '(';
    if (B.LHS) emitExpr(*B.LHS, P);
    OS << ' ' << binOpLex(B.Op) << ' ';
    /* Right-associativity for ^ / .^ is emitted by keeping the same
     * precedence on the RHS; everything else is left-associative so
     * the RHS nudges up by 1 to force parens on a same-level peer. */
    int RhsMin = (B.Op == BinOp::Pow || B.Op == BinOp::ElemPow) ? P : P + 1;
    if (B.RHS) emitExpr(*B.RHS, RhsMin);
    if (Paren) OS << ')';
    return;
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(E);
    /* Unary binds tighter than any binary op we emit; no parens
     * needed from ParentPrec, but wrap if the operand itself is a
     * binary op to avoid `- a + b` parse ambiguity. */
    OS << unOpLex(U.Op);
    if (U.Operand) emitExpr(*U.Operand, 9);
    return;
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<const PostfixOpExpr &>(E);
    if (P.Operand) emitExpr(*P.Operand, 9);
    OS << postfixLex(P.Op);
    return;
  }
  case NodeKind::RangeExpr:
    emitRange(static_cast<const RangeExpr &>(E));
    return;
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(E);
    if (C.Callee) emitExpr(*C.Callee, 10);
    emitCallArgs(C.Args);
    return;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<const CellIndex &>(E);
    if (C.Callee) emitExpr(*C.Callee, 10);
    OS << '{';
    for (size_t i = 0; i < C.Args.size(); ++i) {
      if (i) OS << ", ";
      if (C.Args[i]) emitExpr(*C.Args[i]);
    }
    OS << '}';
    return;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<const FieldAccess &>(E);
    if (F.Base) emitExpr(*F.Base, 10);
    OS << '.' << F.Field;
    return;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<const DynamicField &>(E);
    if (F.Base) emitExpr(*F.Base, 10);
    OS << ".(";
    if (F.Name) emitExpr(*F.Name);
    OS << ')';
    return;
  }
  case NodeKind::MatrixLiteral:
    emitMatrixLiteral(static_cast<const MatrixLiteral &>(E));
    return;
  case NodeKind::CellLiteral:
    emitCellLiteral(static_cast<const CellLiteral &>(E));
    return;
  case NodeKind::AnonFunction:
    emitAnonFunction(static_cast<const AnonFunction &>(E));
    return;
  case NodeKind::FuncHandle: {
    auto &F = static_cast<const FuncHandle &>(E);
    OS << '@' << F.Name;
    return;
  }
  default:
    break;
  }
}

void Formatter::emitBlock(const Block &B) {
  for (const Stmt *S : B.Stmts) if (S) emitStmt(*S);
}

void Formatter::emitAssign(const AssignStmt &A) {
  pad();
  if (A.LHS.size() == 1) {
    if (A.LHS[0]) emitExpr(*A.LHS[0]);
  } else {
    OS << '[';
    for (size_t i = 0; i < A.LHS.size(); ++i) {
      if (i) OS << ", ";
      if (A.LHS[i]) emitExpr(*A.LHS[i]);
      else           OS << '~';
    }
    OS << ']';
  }
  OS << " = ";
  if (A.RHS) emitExpr(*A.RHS);
  endStmt(A.Suppressed);
}

void Formatter::emitExprStmt(const ExprStmt &E) {
  pad();
  if (E.E) emitExpr(*E.E);
  endStmt(E.Suppressed);
}

void Formatter::emitIf(const IfStmt &I) {
  pad(); OS << "if ";
  if (I.Cond) emitExpr(*I.Cond);
  OS << '\n';
  ++Depth;
  if (I.Then) emitBlock(*I.Then);
  --Depth;
  for (const auto &EI : I.Elseifs) {
    pad(); OS << "elseif ";
    if (EI.Cond) emitExpr(*EI.Cond);
    OS << '\n';
    ++Depth;
    if (EI.Body) emitBlock(*EI.Body);
    --Depth;
  }
  if (I.Else) {
    pad(); OS << "else\n";
    ++Depth; emitBlock(*I.Else); --Depth;
  }
  pad(); OS << "end\n";
}

void Formatter::emitFor(const ForStmt &F) {
  pad(); OS << (F.IsParfor ? "parfor " : "for ") << F.Var << " = ";
  if (F.Iter) emitExpr(*F.Iter);
  OS << '\n';
  ++Depth;
  if (F.Body) emitBlock(*F.Body);
  --Depth;
  pad(); OS << "end\n";
}

void Formatter::emitWhile(const WhileStmt &W) {
  pad(); OS << "while ";
  if (W.Cond) emitExpr(*W.Cond);
  OS << '\n';
  ++Depth;
  if (W.Body) emitBlock(*W.Body);
  --Depth;
  pad(); OS << "end\n";
}

void Formatter::emitSwitch(const SwitchStmt &Sw) {
  pad(); OS << "switch ";
  if (Sw.Discriminant) emitExpr(*Sw.Discriminant);
  OS << '\n';
  for (const auto &C : Sw.Cases) {
    pad();
    if (C.Value) { OS << "case "; emitExpr(*C.Value); }
    else         { OS << "otherwise"; }
    OS << '\n';
    ++Depth;
    if (C.Body) emitBlock(*C.Body);
    --Depth;
  }
  pad(); OS << "end\n";
}

void Formatter::emitTry(const TryStmt &T) {
  pad(); OS << "try\n";
  ++Depth;
  if (T.TryBody) emitBlock(*T.TryBody);
  --Depth;
  pad();
  if (!T.CatchVar.empty()) OS << "catch " << T.CatchVar << '\n';
  else                     OS << "catch\n";
  ++Depth;
  if (T.CatchBody) emitBlock(*T.CatchBody);
  --Depth;
  pad(); OS << "end\n";
}

void Formatter::emitGlobal(const GlobalDecl &G) {
  pad(); OS << "global";
  for (auto N : G.Names) OS << ' ' << N;
  OS << '\n';
}
void Formatter::emitPersistent(const PersistentDecl &P) {
  pad(); OS << "persistent";
  for (auto N : P.Names) OS << ' ' << N;
  OS << '\n';
}
void Formatter::emitCommand(const CommandStmt &C) {
  pad(); OS << C.Name;
  for (const auto &A : C.Args) OS << ' ' << A;
  endStmt(C.Suppressed);
}

void Formatter::emitStmt(const Stmt &S) {
  switch (S.Kind) {
  case NodeKind::ExprStmt:       emitExprStmt(static_cast<const ExprStmt &>(S)); break;
  case NodeKind::AssignStmt:     emitAssign(static_cast<const AssignStmt &>(S)); break;
  case NodeKind::IfStmt:         emitIf(static_cast<const IfStmt &>(S)); break;
  case NodeKind::ForStmt:        emitFor(static_cast<const ForStmt &>(S)); break;
  case NodeKind::WhileStmt:      emitWhile(static_cast<const WhileStmt &>(S)); break;
  case NodeKind::SwitchStmt:     emitSwitch(static_cast<const SwitchStmt &>(S)); break;
  case NodeKind::TryStmt:        emitTry(static_cast<const TryStmt &>(S)); break;
  case NodeKind::ReturnStmt:     pad(); OS << "return\n"; break;
  case NodeKind::BreakStmt:      pad(); OS << "break\n"; break;
  case NodeKind::ContinueStmt:   pad(); OS << "continue\n"; break;
  case NodeKind::GlobalDecl:     emitGlobal(static_cast<const GlobalDecl &>(S)); break;
  case NodeKind::PersistentDecl: emitPersistent(static_cast<const PersistentDecl &>(S)); break;
  case NodeKind::CommandStmt:    emitCommand(static_cast<const CommandStmt &>(S)); break;
  default: break;
  }
}

void Formatter::emitFunction(const Function &F) {
  pad();
  OS << "function ";
  if (F.Outputs.size() == 1) {
    OS << F.Outputs[0] << " = ";
  } else if (F.Outputs.size() > 1) {
    OS << '[';
    for (size_t i = 0; i < F.Outputs.size(); ++i) {
      if (i) OS << ", ";
      OS << F.Outputs[i];
    }
    OS << "] = ";
  }
  OS << F.Name << '(';
  for (size_t i = 0; i < F.Inputs.size(); ++i) {
    if (i) OS << ", ";
    OS << F.Inputs[i];
  }
  OS << ")\n";
  ++Depth;
  if (F.Body) emitBlock(*F.Body);
  for (const Function *N : F.Nested) if (N) emitFunction(*N);
  --Depth;
  pad(); OS << "end\n";
}

void Formatter::emitClassDef(const ClassDef &C) {
  pad();
  OS << "classdef " << C.Name;
  if (!C.SuperName.empty()) OS << " < " << C.SuperName;
  OS << '\n';
  ++Depth;
  /* Emit properties in source order, starting a new
   * `properties [(Attrs)]` block whenever the attribute signature
   * changes from the previous prop. This preserves declarations like
   *   properties (Dependent)
   *       Area
   *   end
   * rather than collapsing them into a single vanilla block. */
  auto propAttrsStr = [](const ClassProp &P) {
    std::string S;
    auto add = [&](std::string_view K, std::string_view V = {}) {
      if (!S.empty()) S += ", ";
      S += std::string(K);
      if (!V.empty()) { S += '='; S += std::string(V); }
    };
    if (P.Dependent)  add("Dependent");
    if (P.Constant)   add("Constant");
    if (P.IsAbstract) add("Abstract");
    if (!P.Access.empty())    add("Access",    P.Access);
    if (!P.GetAccess.empty()) add("GetAccess", P.GetAccess);
    if (!P.SetAccess.empty()) add("SetAccess", P.SetAccess);
    return S;
  };
  if (!C.Props.empty()) {
    std::string CurAttrs;
    bool Open = false;
    for (size_t i = 0; i < C.Props.size(); ++i) {
      const auto &P = C.Props[i];
      std::string A = propAttrsStr(P);
      if (!Open || A != CurAttrs) {
        if (Open) { --Depth; pad(); OS << "end\n"; }
        pad(); OS << "properties";
        if (!A.empty()) OS << " (" << A << ")";
        OS << '\n';
        ++Depth;
        CurAttrs = A;
        Open = true;
      }
      pad(); OS << P.Name;
      if (P.Default) { OS << " = "; emitExpr(*P.Default); }
      OS << '\n';
    }
    if (Open) { --Depth; pad(); OS << "end\n"; }
  }
  if (!C.EnumMembers.empty()) {
    pad(); OS << "enumeration\n";
    ++Depth;
    for (auto M : C.EnumMembers) { pad(); OS << M << '\n'; }
    --Depth;
    pad(); OS << "end\n";
  }
  if (!C.Methods.empty()) {
    pad(); OS << "methods\n";
    ++Depth;
    for (const Function *M : C.Methods) if (M) emitFunction(*M);
    --Depth;
    pad(); OS << "end\n";
  }
  if (!C.StaticMethods.empty()) {
    pad(); OS << "methods (Static)\n";
    ++Depth;
    for (const Function *M : C.StaticMethods) if (M) emitFunction(*M);
    --Depth;
    pad(); OS << "end\n";
  }
  --Depth;
  pad(); OS << "end\n";
}

void Formatter::run(const TranslationUnit &TU) {
  if (TU.ScriptNode && TU.ScriptNode->Body) emitBlock(*TU.ScriptNode->Body);
  for (const ClassDef *C : TU.Classes) if (C) { OS << '\n'; emitClassDef(*C); }
  for (const Function *F : TU.Functions) if (F) { OS << '\n'; emitFunction(*F); }
}

} // namespace

void formatAST(std::ostream &OS, const TranslationUnit &TU) {
  Formatter F(OS);
  F.run(TU);
}

} // namespace matlab
