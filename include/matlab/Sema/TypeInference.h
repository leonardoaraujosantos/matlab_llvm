#pragma once

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

#include <unordered_map>

namespace matlab {

class TypeInference {
public:
  TypeInference(SemaContext &Sema, TypeContext &TC, DiagnosticEngine &Diag);

  // Must run after Resolver::resolve().
  void run(TranslationUnit &TU);

private:
  SemaContext &Sema;
  TypeContext &TC;
  DiagnosticEngine &Diag;

  // Per-binding inferred type in the current flow. We keep one environment
  // per function at a time; joins across branches produce a new map.
  using Env = std::unordered_map<Binding *, const Type *>;

  void runFunction(Function &F);
  void runScript(Script &S);

  // Forward-analyze a block, updating env. Returns the env at block end.
  Env visitBlock(Block &B, Env In);
  Env visitStmt(Stmt &St, Env In);

  // Expression inference. Returns the Type and also writes E.Ty.
  const Type *visit(Expr &E, Env &Env);
  const Type *visitBinary(BinaryOpExpr &B, Env &Env);
  const Type *visitUnary(UnaryOpExpr &U, Env &Env);
  const Type *visitPostfix(PostfixOpExpr &P, Env &Env);
  const Type *visitRange(RangeExpr &R, Env &Env);
  const Type *visitCallOrIndex(CallOrIndex &C, Env &Env);
  const Type *visitCellIndex(CellIndex &C, Env &Env);
  const Type *visitMatrix(MatrixLiteral &M, Env &Env);
  const Type *visitCellLit(CellLiteral &M, Env &Env);
  const Type *visitBuiltinCall(std::string_view Name,
                               const std::vector<Expr *> &Args,
                               Env &Env);

  // Helpers
  Env joinEnv(const Env &A, const Env &B);
  static bool envEqual(const Env &A, const Env &B);
};

} // namespace matlab
