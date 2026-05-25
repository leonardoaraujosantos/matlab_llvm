//===----------------------------------------------------------------------===//
// AST Cloner — see header for design. Deep-copies the AST tree node-by-
// node through ASTContext::make<T>(), preserving source-buffer-backed
// string_views and zeroing Sema-populated pointers.
//===----------------------------------------------------------------------===//

#include "matlab/AST/Cloner.h"

namespace matlab {

namespace {

// Copy the Node-level fields (SourceRange) from Src onto Dst. The Kind
// is set by the constructor; we keep the original SourceRange so
// diagnostics on the clone still point at the user's source.
template <typename DstT, typename SrcT>
DstT *copyNodeBase(DstT *Dst, const SrcT &Src) {
  Dst->Range = Src.Range;
  return Dst;
}

Expr *cloneExprImpl(ASTContext &Ctx, const Expr *E);
Stmt *cloneStmtImpl(ASTContext &Ctx, const Stmt *S);

Block *cloneBlock(ASTContext &Ctx, const Block *B) {
  if (!B) return nullptr;
  Block *N = Ctx.make<Block>();
  copyNodeBase(N, *B);
  N->Stmts.reserve(B->Stmts.size());
  for (Stmt *St : B->Stmts)
    N->Stmts.push_back(cloneStmtImpl(Ctx, St));
  return N;
}

Expr *cloneExprImpl(ASTContext &Ctx, const Expr *E) {
  if (!E) return nullptr;
  switch (E->Kind) {
  case NodeKind::IntegerLiteral: {
    auto *S = static_cast<const IntegerLiteral *>(E);
    auto *N = Ctx.make<IntegerLiteral>();
    copyNodeBase(N, *S);
    N->Text = S->Text;
    return N;
  }
  case NodeKind::FPLiteral: {
    auto *S = static_cast<const FPLiteral *>(E);
    auto *N = Ctx.make<FPLiteral>();
    copyNodeBase(N, *S);
    N->Text = S->Text;
    return N;
  }
  case NodeKind::ImagLiteral: {
    auto *S = static_cast<const ImagLiteral *>(E);
    auto *N = Ctx.make<ImagLiteral>();
    copyNodeBase(N, *S);
    N->Text = S->Text;
    return N;
  }
  case NodeKind::StringLiteral: {
    auto *S = static_cast<const StringLiteral *>(E);
    auto *N = Ctx.make<StringLiteral>();
    copyNodeBase(N, *S);
    N->Value = S->Value;
    return N;
  }
  case NodeKind::CharLiteral: {
    auto *S = static_cast<const CharLiteral *>(E);
    auto *N = Ctx.make<CharLiteral>();
    copyNodeBase(N, *S);
    N->Value = S->Value;
    return N;
  }
  case NodeKind::NameExpr: {
    auto *S = static_cast<const NameExpr *>(E);
    auto *N = Ctx.make<NameExpr>();
    copyNodeBase(N, *S);
    N->Name = S->Name;
    // Ref is intentionally left nullptr — Resolver re-populates it.
    return N;
  }
  case NodeKind::EndExpr: {
    auto *N = Ctx.make<EndExpr>();
    copyNodeBase(N, *E);
    return N;
  }
  case NodeKind::ColonExpr: {
    auto *N = Ctx.make<ColonExpr>();
    copyNodeBase(N, *E);
    return N;
  }
  case NodeKind::BinaryOp: {
    auto *S = static_cast<const BinaryOpExpr *>(E);
    auto *N = Ctx.make<BinaryOpExpr>();
    copyNodeBase(N, *S);
    N->Op = S->Op;
    N->LHS = cloneExprImpl(Ctx, S->LHS);
    N->RHS = cloneExprImpl(Ctx, S->RHS);
    return N;
  }
  case NodeKind::UnaryOp: {
    auto *S = static_cast<const UnaryOpExpr *>(E);
    auto *N = Ctx.make<UnaryOpExpr>();
    copyNodeBase(N, *S);
    N->Op = S->Op;
    N->Operand = cloneExprImpl(Ctx, S->Operand);
    return N;
  }
  case NodeKind::PostfixOp: {
    auto *S = static_cast<const PostfixOpExpr *>(E);
    auto *N = Ctx.make<PostfixOpExpr>();
    copyNodeBase(N, *S);
    N->Op = S->Op;
    N->Operand = cloneExprImpl(Ctx, S->Operand);
    return N;
  }
  case NodeKind::RangeExpr: {
    auto *S = static_cast<const RangeExpr *>(E);
    auto *N = Ctx.make<RangeExpr>();
    copyNodeBase(N, *S);
    N->Start = cloneExprImpl(Ctx, S->Start);
    N->Step  = cloneExprImpl(Ctx, S->Step);
    N->End   = cloneExprImpl(Ctx, S->End);
    return N;
  }
  case NodeKind::CallOrIndex: {
    auto *S = static_cast<const CallOrIndex *>(E);
    auto *N = Ctx.make<CallOrIndex>();
    copyNodeBase(N, *S);
    N->Callee = cloneExprImpl(Ctx, S->Callee);
    N->Args.reserve(S->Args.size());
    for (Expr *A : S->Args) N->Args.push_back(cloneExprImpl(Ctx, A));
    // Resolved + ArgTypes are intentionally left at their default state —
    // Resolver and TypeInference re-populate them on the clone.
    return N;
  }
  case NodeKind::CellIndex: {
    auto *S = static_cast<const CellIndex *>(E);
    auto *N = Ctx.make<CellIndex>();
    copyNodeBase(N, *S);
    N->Callee = cloneExprImpl(Ctx, S->Callee);
    N->Args.reserve(S->Args.size());
    for (Expr *A : S->Args) N->Args.push_back(cloneExprImpl(Ctx, A));
    return N;
  }
  case NodeKind::FieldAccess: {
    auto *S = static_cast<const FieldAccess *>(E);
    auto *N = Ctx.make<FieldAccess>();
    copyNodeBase(N, *S);
    N->Base = cloneExprImpl(Ctx, S->Base);
    N->Field = S->Field;
    return N;
  }
  case NodeKind::DynamicField: {
    auto *S = static_cast<const DynamicField *>(E);
    auto *N = Ctx.make<DynamicField>();
    copyNodeBase(N, *S);
    N->Base = cloneExprImpl(Ctx, S->Base);
    N->Name = cloneExprImpl(Ctx, S->Name);
    return N;
  }
  case NodeKind::MatrixLiteral: {
    auto *S = static_cast<const MatrixLiteral *>(E);
    auto *N = Ctx.make<MatrixLiteral>();
    copyNodeBase(N, *S);
    N->Rows.reserve(S->Rows.size());
    for (const auto &Row : S->Rows) {
      std::vector<Expr *> R;
      R.reserve(Row.size());
      for (Expr *Cell : Row) R.push_back(cloneExprImpl(Ctx, Cell));
      N->Rows.push_back(std::move(R));
    }
    return N;
  }
  case NodeKind::CellLiteral: {
    auto *S = static_cast<const CellLiteral *>(E);
    auto *N = Ctx.make<CellLiteral>();
    copyNodeBase(N, *S);
    N->Rows.reserve(S->Rows.size());
    for (const auto &Row : S->Rows) {
      std::vector<Expr *> R;
      R.reserve(Row.size());
      for (Expr *Cell : Row) R.push_back(cloneExprImpl(Ctx, Cell));
      N->Rows.push_back(std::move(R));
    }
    return N;
  }
  case NodeKind::AnonFunction: {
    auto *S = static_cast<const AnonFunction *>(E);
    auto *N = Ctx.make<AnonFunction>();
    copyNodeBase(N, *S);
    N->Params = S->Params;
    N->Body = cloneExprImpl(Ctx, S->Body);
    // ParamRefs intentionally left empty — Resolver re-populates.
    return N;
  }
  case NodeKind::FuncHandle: {
    auto *S = static_cast<const FuncHandle *>(E);
    auto *N = Ctx.make<FuncHandle>();
    copyNodeBase(N, *S);
    N->Name = S->Name;
    // Ref intentionally left nullptr.
    return N;
  }
  default:
    // Unknown node kind — return nullptr defensively. The caller's TU has
    // been Resolver-validated already, so this branch should be unreachable
    // in practice. We avoid asserting to keep the cloner usable in
    // diagnostic / dump paths.
    return nullptr;
  }
}

Stmt *cloneStmtImpl(ASTContext &Ctx, const Stmt *S) {
  if (!S) return nullptr;
  switch (S->Kind) {
  case NodeKind::ExprStmt: {
    auto *Src = static_cast<const ExprStmt *>(S);
    auto *N = Ctx.make<ExprStmt>();
    copyNodeBase(N, *Src);
    N->E = cloneExprImpl(Ctx, Src->E);
    N->Suppressed = Src->Suppressed;
    return N;
  }
  case NodeKind::AssignStmt: {
    auto *Src = static_cast<const AssignStmt *>(S);
    auto *N = Ctx.make<AssignStmt>();
    copyNodeBase(N, *Src);
    N->LHS.reserve(Src->LHS.size());
    for (Expr *L : Src->LHS) N->LHS.push_back(cloneExprImpl(Ctx, L));
    N->RHS = cloneExprImpl(Ctx, Src->RHS);
    N->Suppressed = Src->Suppressed;
    return N;
  }
  case NodeKind::IfStmt: {
    auto *Src = static_cast<const IfStmt *>(S);
    auto *N = Ctx.make<IfStmt>();
    copyNodeBase(N, *Src);
    N->Cond = cloneExprImpl(Ctx, Src->Cond);
    N->Then = cloneBlock(Ctx, Src->Then);
    N->Elseifs.reserve(Src->Elseifs.size());
    for (const auto &E : Src->Elseifs) {
      ElseIf C;
      C.Cond = cloneExprImpl(Ctx, E.Cond);
      C.Body = cloneBlock(Ctx, E.Body);
      N->Elseifs.push_back(C);
    }
    N->Else = cloneBlock(Ctx, Src->Else);
    return N;
  }
  case NodeKind::ForStmt: {
    auto *Src = static_cast<const ForStmt *>(S);
    auto *N = Ctx.make<ForStmt>();
    copyNodeBase(N, *Src);
    N->Var = Src->Var;
    N->Iter = cloneExprImpl(Ctx, Src->Iter);
    N->Body = cloneBlock(Ctx, Src->Body);
    N->IsParfor = Src->IsParfor;
    // VarRef intentionally left nullptr.
    return N;
  }
  case NodeKind::WhileStmt: {
    auto *Src = static_cast<const WhileStmt *>(S);
    auto *N = Ctx.make<WhileStmt>();
    copyNodeBase(N, *Src);
    N->Cond = cloneExprImpl(Ctx, Src->Cond);
    N->Body = cloneBlock(Ctx, Src->Body);
    return N;
  }
  case NodeKind::SwitchStmt: {
    auto *Src = static_cast<const SwitchStmt *>(S);
    auto *N = Ctx.make<SwitchStmt>();
    copyNodeBase(N, *Src);
    N->Discriminant = cloneExprImpl(Ctx, Src->Discriminant);
    N->Cases.reserve(Src->Cases.size());
    for (const auto &C : Src->Cases) {
      SwitchCase NC;
      NC.Value = cloneExprImpl(Ctx, C.Value);
      NC.Body = cloneBlock(Ctx, C.Body);
      N->Cases.push_back(NC);
    }
    return N;
  }
  case NodeKind::TryStmt: {
    auto *Src = static_cast<const TryStmt *>(S);
    auto *N = Ctx.make<TryStmt>();
    copyNodeBase(N, *Src);
    N->TryBody = cloneBlock(Ctx, Src->TryBody);
    N->CatchVar = Src->CatchVar;
    N->CatchBody = cloneBlock(Ctx, Src->CatchBody);
    // CatchVarRef intentionally left nullptr.
    return N;
  }
  case NodeKind::ReturnStmt: {
    auto *N = Ctx.make<ReturnStmt>();
    copyNodeBase(N, *S);
    return N;
  }
  case NodeKind::BreakStmt: {
    auto *N = Ctx.make<BreakStmt>();
    copyNodeBase(N, *S);
    return N;
  }
  case NodeKind::ContinueStmt: {
    auto *N = Ctx.make<ContinueStmt>();
    copyNodeBase(N, *S);
    return N;
  }
  case NodeKind::GlobalDecl: {
    auto *Src = static_cast<const GlobalDecl *>(S);
    auto *N = Ctx.make<GlobalDecl>();
    copyNodeBase(N, *Src);
    N->Names = Src->Names;
    return N;
  }
  case NodeKind::PersistentDecl: {
    auto *Src = static_cast<const PersistentDecl *>(S);
    auto *N = Ctx.make<PersistentDecl>();
    copyNodeBase(N, *Src);
    N->Names = Src->Names;
    return N;
  }
  case NodeKind::ImportStmt: {
    auto *Src = static_cast<const ImportStmt *>(S);
    auto *N = Ctx.make<ImportStmt>();
    copyNodeBase(N, *Src);
    N->Path = Src->Path;
    N->Wildcard = Src->Wildcard;
    return N;
  }
  case NodeKind::CommandStmt: {
    auto *Src = static_cast<const CommandStmt *>(S);
    auto *N = Ctx.make<CommandStmt>();
    copyNodeBase(N, *Src);
    N->Name = Src->Name;
    N->Args = Src->Args;
    N->Suppressed = Src->Suppressed;
    return N;
  }
  case NodeKind::Block:
    return cloneBlock(Ctx, static_cast<const Block *>(S));
  default:
    return nullptr;
  }
}

Function *cloneFunctionImpl(ASTContext &Ctx, const Function &Src,
                            std::string_view NewName) {
  auto *N = Ctx.make<Function>();
  N->Range = Src.Range;
  N->Name = Ctx.intern(NewName);
  N->Inputs = Src.Inputs;
  N->Outputs = Src.Outputs;
  N->Body = cloneBlock(Ctx, Src.Body);
  N->Nested.reserve(Src.Nested.size());
  for (Function *Nested : Src.Nested) {
    if (!Nested) {
      N->Nested.push_back(nullptr);
      continue;
    }
    // Nested helpers keep their original name. Future phases that want to
    // mangle nested clones can call cloneFunction recursively with a new
    // suffix; the common case is to share the un-mangled name.
    N->Nested.push_back(cloneFunctionImpl(Ctx, *Nested, Nested->Name));
  }
  // FnScope / Self / ParamRefs / OutputRefs intentionally left default
  // — Resolver re-populates them when run over the clone-augmented TU.
  return N;
}

} // namespace

Expr *cloneExpr(ASTContext &Ctx, const Expr *E) {
  return cloneExprImpl(Ctx, E);
}

Stmt *cloneStmt(ASTContext &Ctx, const Stmt *S) {
  return cloneStmtImpl(Ctx, S);
}

Function *cloneFunction(ASTContext &Ctx, const Function &Src,
                        std::string_view NewName) {
  return cloneFunctionImpl(Ctx, Src, NewName);
}

} // namespace matlab
