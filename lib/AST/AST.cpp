#include "matlab/AST/AST.h"

#include <algorithm>
#include <cstdlib>

namespace matlab {

const char *binOpName(BinOp O) {
  switch (O) {
  case BinOp::Add: return "+";
  case BinOp::Sub: return "-";
  case BinOp::Mul: return "*";
  case BinOp::Div: return "/";
  case BinOp::LeftDiv: return "\\";
  case BinOp::Pow: return "^";
  case BinOp::ElemMul: return ".*";
  case BinOp::ElemDiv: return "./";
  case BinOp::ElemLeftDiv: return ".\\";
  case BinOp::ElemPow: return ".^";
  case BinOp::Eq: return "==";
  case BinOp::Ne: return "~=";
  case BinOp::Lt: return "<";
  case BinOp::Le: return "<=";
  case BinOp::Gt: return ">";
  case BinOp::Ge: return ">=";
  case BinOp::And: return "&";
  case BinOp::Or: return "|";
  case BinOp::ShortAnd: return "&&";
  case BinOp::ShortOr: return "||";
  }
  return "?";
}

const char *unOpName(UnOp O) {
  switch (O) {
  case UnOp::Plus: return "+";
  case UnOp::Minus: return "-";
  case UnOp::Not: return "~";
  }
  return "?";
}

const char *postfixName(PostfixOp O) {
  switch (O) {
  case PostfixOp::CTranspose: return "'";
  case PostfixOp::Transpose: return ".'";
  }
  return "?";
}

const char *nodeKindName(NodeKind K) {
  switch (K) {
#define CASE(X) case NodeKind::X: return #X;
  CASE(IntegerLiteral) CASE(FPLiteral) CASE(ImagLiteral)
  CASE(StringLiteral) CASE(CharLiteral)
  CASE(NameExpr) CASE(EndExpr) CASE(ColonExpr)
  CASE(BinaryOp) CASE(UnaryOp) CASE(PostfixOp) CASE(RangeExpr)
  CASE(CallOrIndex) CASE(CellIndex) CASE(FieldAccess) CASE(DynamicField)
  CASE(MatrixLiteral) CASE(CellLiteral)
  CASE(AnonFunction) CASE(FuncHandle)
  CASE(ExprStmt) CASE(AssignStmt)
  CASE(IfStmt) CASE(ForStmt) CASE(WhileStmt) CASE(SwitchStmt) CASE(TryStmt)
  CASE(ReturnStmt) CASE(BreakStmt) CASE(ContinueStmt)
  CASE(GlobalDecl) CASE(PersistentDecl) CASE(ImportStmt) CASE(CommandStmt)
  CASE(Block)
  CASE(Function) CASE(Script) CASE(ClassDef) CASE(TranslationUnit)
#undef CASE
  }
  return "?";
}

//===----------------------------------------------------------------------===//
// ASTContext
//===----------------------------------------------------------------------===//

static constexpr size_t kSlabSize = 64 * 1024;

ASTContext::ASTContext() = default;

ASTContext::~ASTContext() {
  // Destroy in reverse order of construction.
  for (auto It = Nodes.rbegin(); It != Nodes.rend(); ++It) {
    (*It)->~Node();
  }
}

static size_t alignUp(size_t V, size_t A) { return (V + A - 1) & ~(A - 1); }

void *ASTContext::allocate(size_t Size, size_t Align) {
  if (Size > kSlabSize / 2) {
    // Oversized: give it its own slab.
    Slab S;
    S.Data = std::make_unique<char[]>(Size);
    S.Size = Size;
    S.Used = Size;
    void *P = S.Data.get();
    Slabs.push_back(std::move(S));
    return P;
  }
  if (!Slabs.empty()) {
    Slab &Cur = Slabs.back();
    size_t Start = alignUp(Cur.Used, Align);
    if (Start + Size <= Cur.Size) {
      void *P = Cur.Data.get() + Start;
      Cur.Used = Start + Size;
      return P;
    }
  }
  Slab S;
  S.Data = std::make_unique<char[]>(kSlabSize);
  S.Size = kSlabSize;
  S.Used = Size;
  void *P = S.Data.get();
  Slabs.push_back(std::move(S));
  return P;
}

void ASTContext::registerNode(Node *N) { Nodes.push_back(N); }

std::string_view ASTContext::intern(std::string_view S) {
  auto Str = std::make_unique<std::string>(S);
  std::string_view V = *Str;
  Strings.push_back(std::move(Str));
  return V;
}

} // namespace matlab
