#include "matlab/Sema/TypeInference.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <string>

namespace matlab {

TypeInference::TypeInference(SemaContext &Sema, TypeContext &TC,
                             DiagnosticEngine &Diag)
    : Sema(Sema), TC(TC), Diag(Diag) {
  (void)this->Sema;
  (void)this->Diag;
}

//===----------------------------------------------------------------------===//
// Env helpers
//===----------------------------------------------------------------------===//

TypeInference::Env TypeInference::joinEnv(const Env &A, const Env &B) {
  Env R = A;
  for (auto &[K, V] : B) {
    auto It = R.find(K);
    if (It == R.end()) R[K] = V;
    else               It->second = TC.join(It->second, V);
  }
  // Any binding present in A but not in B stays (it's an "unknown from the
  // other branch" — conservatively keep the known type from A).
  return R;
}

bool TypeInference::envEqual(const Env &A, const Env &B) {
  if (A.size() != B.size()) return false;
  for (auto &[K, V] : A) {
    auto It = B.find(K);
    if (It == B.end() || It->second != V) return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Top-level
//===----------------------------------------------------------------------===//

void TypeInference::run(TranslationUnit &TU) {
  if (TU.ScriptNode) runScript(*TU.ScriptNode);
  for (Function *F : TU.Functions) runFunction(*F);
}

void TypeInference::runScript(Script &S) {
  Env E;
  if (S.Body) visitBlock(*S.Body, std::move(E));
}

void TypeInference::runFunction(Function &F) {
  Env E;
  // Parameters start with Any.
  for (Binding *B : F.ParamRefs) {
    E[B] = TC.any();
    B->InferredType = TC.any();
  }
  // Outputs start unassigned; treat as Any to allow use before assignment
  // analysis downstream.
  for (Binding *B : F.OutputRefs) E[B] = nullptr;

  if (F.Body) {
    Env Out = visitBlock(*F.Body, std::move(E));
    // Copy final types back to bindings.
    for (auto &[B, T] : Out) B->InferredType = T;
  }

  for (Function *N : F.Nested) runFunction(*N);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

TypeInference::Env TypeInference::visitBlock(Block &B, Env In) {
  for (Stmt *S : B.Stmts) {
    if (!S) continue;
    In = visitStmt(*S, std::move(In));
  }
  return In;
}

TypeInference::Env TypeInference::visitStmt(Stmt &St, Env In) {
  switch (St.Kind) {
  case NodeKind::ExprStmt: {
    auto &E = static_cast<ExprStmt &>(St);
    if (E.E) visit(*E.E, In);
    return In;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<AssignStmt &>(St);
    const Type *RhsT = A.RHS ? visit(*A.RHS, In) : TC.any();
    for (Expr *L : A.LHS) {
      if (!L) continue;
      if (auto *N = dynamic_cast<NameExpr *>(L)) {
        if (N->Ref) {
          In[N->Ref] = RhsT;
          N->Ty = RhsT;
        }
      } else {
        // Indexed LHS (e.g. a(i) = x). For now, keep the root's type as-is
        // and annotate the sub-expression with Any.
        visit(*L, In);
      }
    }
    return In;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<IfStmt &>(St);
    if (I.Cond) visit(*I.Cond, In);
    Env Then = I.Then ? visitBlock(*I.Then, In) : In;
    Env Acc = Then;
    for (auto &EI : I.Elseifs) {
      if (EI.Cond) visit(*EI.Cond, In);
      Env B = EI.Body ? visitBlock(*EI.Body, In) : In;
      Acc = joinEnv(Acc, B);
    }
    if (I.Else) {
      Env E = visitBlock(*I.Else, In);
      Acc = joinEnv(Acc, E);
    } else {
      // No else: possible fall-through with original env.
      Acc = joinEnv(Acc, In);
    }
    return Acc;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<ForStmt &>(St);
    const Type *IterT = F.Iter ? visit(*F.Iter, In) : TC.any();
    // Loop variable type: if iter is an array, the loop var is a column of it;
    // for now, we approximate as scalar of the same dtype (common case of
    // `for i = 1:n`).
    const Type *VarT = TC.any();
    if (IterT && IterT->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*IterT);
      VarT = TC.scalar(A.Elt);
    }
    Binding *VarB = nullptr;
    // Find the binding for F.Var in the environment by name.
    for (auto &[B, _] : In) {
      if (B->Name == F.Var) { VarB = B; break; }
    }
    if (VarB) In[VarB] = VarT;
    // Fixpoint over loop body.
    Env Cur = In;
    for (int Iter = 0; Iter < 8; ++Iter) {
      Env Next = F.Body ? visitBlock(*F.Body, Cur) : Cur;
      Next = joinEnv(Next, Cur);
      if (envEqual(Next, Cur)) { Cur = std::move(Next); break; }
      Cur = std::move(Next);
    }
    // Zero-iteration case: join with entering env.
    return joinEnv(Cur, In);
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<WhileStmt &>(St);
    if (W.Cond) visit(*W.Cond, In);
    Env Cur = In;
    for (int Iter = 0; Iter < 8; ++Iter) {
      Env Next = W.Body ? visitBlock(*W.Body, Cur) : Cur;
      Next = joinEnv(Next, Cur);
      if (envEqual(Next, Cur)) { Cur = std::move(Next); break; }
      Cur = std::move(Next);
    }
    return joinEnv(Cur, In);
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<SwitchStmt &>(St);
    if (Sw.Discriminant) visit(*Sw.Discriminant, In);
    Env Acc;
    bool First = true;
    bool HasOtherwise = false;
    for (auto &C : Sw.Cases) {
      if (C.Value) visit(*C.Value, In);
      if (!C.Value) HasOtherwise = true;
      Env B = C.Body ? visitBlock(*C.Body, In) : In;
      if (First) { Acc = B; First = false; }
      else        Acc = joinEnv(Acc, B);
    }
    if (!HasOtherwise) Acc = joinEnv(Acc, In);
    return Acc;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<TryStmt &>(St);
    Env TryE = T.TryBody ? visitBlock(*T.TryBody, In) : In;
    Env CatchE = T.CatchBody ? visitBlock(*T.CatchBody, In) : In;
    return joinEnv(TryE, CatchE);
  }
  case NodeKind::ReturnStmt:
  case NodeKind::BreakStmt:
  case NodeKind::ContinueStmt:
  case NodeKind::GlobalDecl:
  case NodeKind::PersistentDecl:
  case NodeKind::ImportStmt:
  case NodeKind::CommandStmt:
    return In;
  default:
    return In;
  }
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

const Type *TypeInference::visit(Expr &E, Env &Env) {
  const Type *T = nullptr;
  switch (E.Kind) {
  case NodeKind::IntegerLiteral:
  case NodeKind::FPLiteral:
    // MATLAB: numeric literals default to double.
    T = TC.scalar(Dtype::Double);
    break;
  case NodeKind::ImagLiteral:
    T = TC.scalar(Dtype::Complex);
    break;
  case NodeKind::StringLiteral:
    T = TC.stringScalar();
    break;
  case NodeKind::CharLiteral: {
    auto &L = static_cast<CharLiteral &>(E);
    T = TC.arrayOf(Dtype::Char,
                   Shape::matrix(1, static_cast<int64_t>(L.Value.size())));
    break;
  }
  case NodeKind::EndExpr:
    T = TC.scalar(Dtype::Double);
    break;
  case NodeKind::ColonExpr:
    T = TC.any();
    break;
  case NodeKind::NameExpr: {
    auto &N = static_cast<NameExpr &>(E);
    if (N.Ref) {
      auto It = Env.find(N.Ref);
      if (It != Env.end() && It->second) {
        T = It->second;
      } else if (N.Ref->Kind == BindingKind::Function ||
                 N.Ref->Kind == BindingKind::Builtin) {
        T = TC.funcHandle();
      } else {
        T = TC.any();
      }
    } else {
      T = TC.any();
    }
    break;
  }
  case NodeKind::BinaryOp:
    T = visitBinary(static_cast<BinaryOpExpr &>(E), Env); break;
  case NodeKind::UnaryOp:
    T = visitUnary(static_cast<UnaryOpExpr &>(E), Env); break;
  case NodeKind::PostfixOp:
    T = visitPostfix(static_cast<PostfixOpExpr &>(E), Env); break;
  case NodeKind::RangeExpr:
    T = visitRange(static_cast<RangeExpr &>(E), Env); break;
  case NodeKind::CallOrIndex:
    T = visitCallOrIndex(static_cast<CallOrIndex &>(E), Env); break;
  case NodeKind::CellIndex:
    T = visitCellIndex(static_cast<CellIndex &>(E), Env); break;
  case NodeKind::MatrixLiteral:
    T = visitMatrix(static_cast<MatrixLiteral &>(E), Env); break;
  case NodeKind::CellLiteral:
    T = visitCellLit(static_cast<CellLiteral &>(E), Env); break;
  case NodeKind::FieldAccess: {
    auto &F = static_cast<FieldAccess &>(E);
    if (F.Base) visit(*F.Base, Env);
    T = TC.any();
    break;
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<DynamicField &>(E);
    if (F.Base) visit(*F.Base, Env);
    if (F.Name) visit(*F.Name, Env);
    T = TC.any();
    break;
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<AnonFunction &>(E);
    // Body is typed in a nested scope; simple pass without capturing env
    // changes back (closures are immutable captures semantically).
    if (A.Body) visit(*A.Body, Env);
    T = TC.funcHandle();
    break;
  }
  case NodeKind::FuncHandle:
    T = TC.funcHandle();
    break;
  default:
    T = TC.any();
  }
  E.Ty = T;
  return T;
}

const Type *TypeInference::visitBinary(BinaryOpExpr &B, Env &Env) {
  const Type *L = B.LHS ? visit(*B.LHS, Env) : TC.any();
  const Type *R = B.RHS ? visit(*B.RHS, Env) : TC.any();

  switch (B.Op) {
  case BinOp::Add: case BinOp::Sub:
  case BinOp::ElemMul: case BinOp::ElemDiv:
  case BinOp::ElemLeftDiv: case BinOp::ElemPow:
    return TC.broadcastNumeric(L, R);

  case BinOp::Mul: {
    // Matrix multiply: (M x K) * (K x N) -> (M x N); scalar * X broadcasts.
    if (!L || !R || L->K != Type::Kind::Array || R->K != Type::Kind::Array)
      return TC.any();
    auto &LA = static_cast<const ArrayType &>(*L);
    auto &RA = static_cast<const ArrayType &>(*R);
    Dtype D = promoteDtype(LA.Elt, RA.Elt);
    if (D == Dtype::Unknown) return TC.any();
    if (LA.S.K == Shape::Rank::Scalar) return TC.arrayOf(D, RA.S);
    if (RA.S.K == Shape::Rank::Scalar) return TC.arrayOf(D, LA.S);
    if (LA.S.K == Shape::Rank::Matrix && RA.S.K == Shape::Rank::Matrix) {
      int64_t M = LA.S.Dims.size() > 0 ? LA.S.Dims[0] : -1;
      int64_t N = RA.S.Dims.size() > 1 ? RA.S.Dims[1] : -1;
      return TC.arrayOf(D, Shape::matrix(M, N));
    }
    return TC.arrayOf(D, Shape::unknown());
  }
  case BinOp::Div: case BinOp::LeftDiv: {
    if (!L || !R || L->K != Type::Kind::Array || R->K != Type::Kind::Array)
      return TC.any();
    auto &LA = static_cast<const ArrayType &>(*L);
    auto &RA = static_cast<const ArrayType &>(*R);
    Dtype D = promoteDtype(LA.Elt, RA.Elt);
    if (D == Dtype::Unknown) return TC.any();
    if (LA.S.K == Shape::Rank::Scalar && RA.S.K == Shape::Rank::Scalar)
      return TC.scalar(D);
    return TC.arrayOf(D, Shape::unknown());
  }
  case BinOp::Pow: {
    // Scalar^scalar -> scalar. Matrix power has different semantics.
    if (!L || !R || L->K != Type::Kind::Array || R->K != Type::Kind::Array)
      return TC.any();
    auto &LA = static_cast<const ArrayType &>(*L);
    auto &RA = static_cast<const ArrayType &>(*R);
    Dtype D = promoteDtype(LA.Elt, RA.Elt);
    if (LA.S.K == Shape::Rank::Scalar && RA.S.K == Shape::Rank::Scalar)
      return TC.scalar(D);
    return TC.arrayOf(D, LA.S);
  }
  case BinOp::Eq: case BinOp::Ne:
  case BinOp::Lt: case BinOp::Le:
  case BinOp::Gt: case BinOp::Ge: {
    const Type *BT = TC.broadcastNumeric(L, R);
    if (BT && BT->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*BT);
      return TC.arrayOf(Dtype::Logical, A.S);
    }
    return TC.scalar(Dtype::Logical);
  }
  case BinOp::And: case BinOp::Or: {
    const Type *BT = TC.broadcastNumeric(L, R);
    if (BT && BT->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*BT);
      return TC.arrayOf(Dtype::Logical, A.S);
    }
    return TC.scalar(Dtype::Logical);
  }
  case BinOp::ShortAnd: case BinOp::ShortOr:
    return TC.scalar(Dtype::Logical);
  }
  return TC.any();
}

const Type *TypeInference::visitUnary(UnaryOpExpr &U, Env &Env) {
  const Type *T = U.Operand ? visit(*U.Operand, Env) : TC.any();
  if (U.Op == UnOp::Not) {
    if (T && T->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*T);
      return TC.arrayOf(Dtype::Logical, A.S);
    }
    return TC.scalar(Dtype::Logical);
  }
  return T;
}

const Type *TypeInference::visitPostfix(PostfixOpExpr &P, Env &Env) {
  const Type *T = P.Operand ? visit(*P.Operand, Env) : TC.any();
  if (!T || T->K != Type::Kind::Array) return TC.any();
  auto &A = static_cast<const ArrayType &>(*T);
  // Transpose swaps the two dimensions for matrices; scalars unchanged.
  if (A.S.K == Shape::Rank::Scalar) return T;
  if (A.S.K == Shape::Rank::Matrix && A.S.Dims.size() >= 2) {
    return TC.arrayOf(A.Elt, Shape::matrix(A.S.Dims[1], A.S.Dims[0]));
  }
  if (A.S.K == Shape::Rank::Vector && !A.S.Dims.empty()) {
    // Row vs column is not tracked in our Vector rank; return as-is.
    return T;
  }
  return TC.arrayOf(A.Elt, Shape::unknown());
}

const Type *TypeInference::visitRange(RangeExpr &R, Env &Env) {
  if (R.Start) visit(*R.Start, Env);
  if (R.Step)  visit(*R.Step,  Env);
  if (R.End)   visit(*R.End,   Env);
  // a:b produces a row vector of doubles (dynamic length unless we can fold).
  return TC.arrayOf(Dtype::Double, Shape::vector(-1));
}

const Type *TypeInference::visitCellIndex(CellIndex &C, Env &Env) {
  if (C.Callee) visit(*C.Callee, Env);
  for (Expr *A : C.Args) if (A) visit(*A, Env);
  return TC.any();
}

const Type *TypeInference::visitMatrix(MatrixLiteral &M, Env &Env) {
  Dtype D = Dtype::Unknown;
  bool First = true;
  // Count rows / cols approximately (scalar-assumption).
  int64_t Rows = static_cast<int64_t>(M.Rows.size());
  int64_t Cols = -1;
  bool AllScalars = true;
  for (auto &R : M.Rows) {
    int64_t RowCols = 0;
    for (Expr *E : R) {
      if (!E) continue;
      const Type *T = visit(*E, Env);
      if (!T || T->K != Type::Kind::Array) { AllScalars = false; RowCols++; continue; }
      auto &A = static_cast<const ArrayType &>(*T);
      if (A.S.K != Shape::Rank::Scalar) AllScalars = false;
      if (First) { D = A.Elt; First = false; }
      else       D = promoteDtype(D, A.Elt);
      RowCols++;
    }
    if (Cols < 0)                Cols = RowCols;
    else if (Cols != RowCols)    Cols = -1;
  }
  if (First) D = Dtype::Double;
  if (AllScalars && Rows >= 0 && Cols >= 0) {
    if (Rows == 1 && Cols == 1) return TC.scalar(D);
    if (Rows == 1)               return TC.arrayOf(D, Shape::vector(Cols));
    return TC.arrayOf(D, Shape::matrix(Rows, Cols));
  }
  return TC.arrayOf(D == Dtype::Unknown ? Dtype::Double : D, Shape::unknown());
}

const Type *TypeInference::visitCellLit(CellLiteral &M, Env &Env) {
  for (auto &R : M.Rows)
    for (Expr *E : R) if (E) visit(*E, Env);
  return TC.cellAny();
}

//===----------------------------------------------------------------------===//
// Builtins
//===----------------------------------------------------------------------===//

// Parse an integer literal if the expression is one; returns -1 otherwise.
static int64_t foldInt(Expr *E) {
  if (!E) return -1;
  if (auto *L = dynamic_cast<IntegerLiteral *>(E)) {
    try { return std::stoll(std::string(L->Text)); }
    catch (...) { return -1; }
  }
  if (auto *U = dynamic_cast<UnaryOpExpr *>(E)) {
    if (U->Op == UnOp::Minus) {
      int64_t V = foldInt(U->Operand);
      return V >= 0 ? -V : -1;
    }
    if (U->Op == UnOp::Plus)
      return foldInt(U->Operand);
  }
  return -1;
}

const Type *TypeInference::visitBuiltinCall(std::string_view Name,
                                             const std::vector<Expr *> &Args,
                                             Env &Env) {
  // Evaluate argument types first (side-effect: annotate AST).
  std::vector<const Type *> ArgTys;
  ArgTys.reserve(Args.size());
  for (Expr *A : Args) ArgTys.push_back(A ? visit(*A, Env) : TC.any());

  auto constructorOf = [&](Dtype D) -> const Type * {
    // zeros/ones/eye/rand/randn(n)    -> n x n  (except zeros() = scalar)
    // zeros/ones(m, n)                -> m x n
    // zeros/ones(sz) with vector sz   -> unknown shape
    if (Args.empty()) return TC.scalar(D);
    if (Args.size() == 1) {
      int64_t N = foldInt(Args[0]);
      if (N >= 0) return TC.arrayOf(D, Shape::matrix(N, N));
      return TC.arrayOf(D, Shape::unknown());
    }
    if (Args.size() == 2) {
      int64_t M = foldInt(Args[0]);
      int64_t N = foldInt(Args[1]);
      return TC.arrayOf(D, Shape::matrix(M, N));
    }
    return TC.arrayOf(D, Shape::unknown());
  };

  if (Name == "zeros" || Name == "ones" || Name == "eye" ||
      Name == "rand"  || Name == "randn")
    return constructorOf(Dtype::Double);
  if (Name == "true" || Name == "false")
    return constructorOf(Dtype::Logical);

  if (Name == "size") {
    // size(A) -> row vector of length ndims(A); size(A,k) -> scalar double
    if (Args.size() >= 2) return TC.scalar(Dtype::Double);
    return TC.arrayOf(Dtype::Double, Shape::vector(-1));
  }
  if (Name == "length" || Name == "numel" || Name == "ndims")
    return TC.scalar(Dtype::Double);

  if (Name == "linspace") {
    int64_t N = -1;
    if (Args.size() >= 3) N = foldInt(Args[2]);
    return TC.arrayOf(Dtype::Double,
                      N > 0 ? Shape::vector(N) : Shape::vector(-1));
  }

  if (Name == "abs" || Name == "sqrt" || Name == "exp" ||
      Name == "log" || Name == "sin"  || Name == "cos" || Name == "tan") {
    // Element-wise: preserves shape, promotes to floating.
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      Dtype D = isFloating(A.Elt) ? A.Elt : Dtype::Double;
      return TC.arrayOf(D, A.S);
    }
    return TC.scalar(Dtype::Double);
  }
  if (Name == "mod" || Name == "rem" || Name == "floor" ||
      Name == "ceil" || Name == "round" || Name == "fix") {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      return TC.arrayOf(A.Elt, A.S);
    }
    return TC.scalar(Dtype::Double);
  }

  if (Name == "transpose" || Name == "ctranspose") {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      if (A.S.K == Shape::Rank::Matrix && A.S.Dims.size() >= 2)
        return TC.arrayOf(A.Elt, Shape::matrix(A.S.Dims[1], A.S.Dims[0]));
      return TC.arrayOf(A.Elt, A.S);
    }
    return TC.any();
  }

  // Dtype-cast builtins
  if (Name == "double")  return ArgTys.empty() ? TC.scalar(Dtype::Double) :
    ArgTys[0] && ArgTys[0]->K == Type::Kind::Array ?
      TC.arrayOf(Dtype::Double, static_cast<const ArrayType &>(*ArgTys[0]).S)
      : (const Type *)TC.scalar(Dtype::Double);
  if (Name == "single")  return TC.scalar(Dtype::Single);
  if (Name == "int32")   return TC.scalar(Dtype::Int32);
  if (Name == "int64")   return TC.scalar(Dtype::Int64);
  if (Name == "logical") return TC.scalar(Dtype::Logical);
  if (Name == "char")    return TC.arrayOf(Dtype::Char, Shape::unknown());

  if (Name == "disp" || Name == "fprintf" || Name == "warning" ||
      Name == "error") {
    return TC.any(); // effectively void
  }

  return TC.any();
}

const Type *TypeInference::visitCallOrIndex(CallOrIndex &C, Env &Env) {
  // Callee must be visited to annotate its type; treat it specially so we
  // don't box a function reference into Any.
  if (C.Callee) visit(*C.Callee, Env);

  if (C.Resolved == CallKind::Call) {
    if (auto *N = dynamic_cast<NameExpr *>(C.Callee)) {
      if (N->Ref && N->Ref->Kind == BindingKind::Builtin) {
        return visitBuiltinCall(N->Name, C.Args, Env);
      }
      if (N->Ref && N->Ref->Kind == BindingKind::Function && N->Ref->FuncDef) {
        // Visit arguments for side-effect annotation.
        for (Expr *A : C.Args) if (A) visit(*A, Env);
        // Without cross-function inference, return Any. TODO: infer per-call.
        return TC.any();
      }
    }
    for (Expr *A : C.Args) if (A) visit(*A, Env);
    return TC.any();
  }

  // Index: element type of the callee, shape depends on index kind.
  for (Expr *A : C.Args) if (A) visit(*A, Env);
  if (C.Callee && C.Callee->Ty) {
    if (C.Callee->Ty->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*C.Callee->Ty);
      // Approximation: scalar indices -> scalar element; otherwise unknown
      // shape of the same element type.
      bool AllScalar = true;
      for (Expr *Arg : C.Args) {
        if (!Arg || !Arg->Ty || Arg->Ty->K != Type::Kind::Array) {
          AllScalar = false;
          break;
        }
        auto &AT = static_cast<const ArrayType &>(*Arg->Ty);
        if (AT.S.K != Shape::Rank::Scalar) { AllScalar = false; break; }
      }
      if (AllScalar) return TC.scalar(A.Elt);
      return TC.arrayOf(A.Elt, Shape::unknown());
    }
    if (C.Callee->Ty->K == Type::Kind::StringArray) {
      return TC.stringScalar();
    }
  }
  return TC.any();
}

} // namespace matlab
