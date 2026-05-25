#include "matlab/Sema/TypeInference.h"

#include "llvm/ADT/DenseMap.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <optional>
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
  /* Class methods are ordinary free functions at this stage (Resolver
   * registered them as Function bindings with a class-pinned first
   * param). Walk them too so their bodies get the same type-inference
   * treatment as top-level functions — otherwise comparisons like
   * `nargin == 1` inside a method body keep a NoneType result and
   * scf.if lowering breaks. */
  for (ClassDef *C : TU.Classes) {
    for (Function *M : C->Methods) if (M) runFunction(*M);
    for (Function *M : C->StaticMethods) if (M) runFunction(*M);
  }
}

void TypeInference::runScript(Script &S) {
  Env E;
  if (S.Body) visitBlock(*S.Body, std::move(E));
}

void TypeInference::runFunction(Function &F) {
  Env E;
  // Phase 5.6 Stage B — vector function arguments. Before the
  // ordinary body walk, scan the body's AST for `param(k)`
  // constant-index subscript sites. If any parameter is read with
  // a constant integer index, seed its env type with a `Vector(N)`
  // shape (where N is the largest index seen) so subsequent
  // visits of the same param flow as a vector instead of
  // collapsing to a scalar via the assignment `param = fi(param,
  // ...)` re-cast idiom. The element dtype starts as Unknown; the
  // body's own ops (fi-cast, etc.) refine it. Without this seed,
  // `vec_a = fi(vec_a, 1, 16, 8); ... = vec_a(k);` mistypes vec_a
  // as scalar `i16`, the subscript surfaces as
  // `matlab.subscript(scalar, idx)`, and the whole pipeline
  // operates on a malformed shape (Stage B blocker writeup in
  // docs/emit_systemverilog.md).
  llvm::DenseMap<Binding *, int64_t> ParamMaxIdx;
  // Phase 5.6 Stage A.1 — fi-spec propagation across function-arg
  // re-cast sites. When the body contains `fi(param, S, W, F)` (a
  // re-cast of an fi-typed function parameter), we capture the
  // declared output spec and use it as the param's inferred type
  // — the user's intent is "interpret the existing fi value with
  // this exact spec", which only matches if the param itself is
  // already that spec at the call site. Without this, the param
  // stays Any and Lowering can't emit fi.cast attrs that
  // LowerFixedPoint's clamp path needs.
  llvm::DenseMap<Binding *, FixedSpec> ParamFiSpec;
  if (F.Body) {
    std::function<void(const Expr &)> walkExpr;
    std::function<void(const Stmt &)> walkStmt;
    std::function<void(const ::matlab::Block &)> walkBlock;
    walkExpr = [&](const Expr &E0) {
      if (auto *C = dynamic_cast<const CallOrIndex *>(&E0)) {
        // Phase 5.6 Stage A.1: recognize `fi(param, signed, WL,
        // FL)` re-cast where `param` is a NameExpr bound to a
        // function parameter. Capture the declared output spec
        // as the param's inferred fi spec.
        if (auto *FN = dynamic_cast<const NameExpr *>(C->Callee)) {
          if (FN->Ref && FN->Ref->Kind == BindingKind::Builtin &&
              FN->Name == "fi" && C->Args.size() >= 4 && C->Args[0]) {
            if (auto *VN = dynamic_cast<const NameExpr *>(C->Args[0])) {
              if (VN->Ref) {
                for (Binding *PB : F.ParamRefs) {
                  if (PB != VN->Ref) continue;
                  // Need (signed, WL, FL) as compile-time integers.
                  std::function<std::optional<int64_t>(const Expr *)>
                      foldI = [&](const Expr *Ex) -> std::optional<int64_t> {
                    if (!Ex) return std::nullopt;
                    if (auto *L =
                            dynamic_cast<const IntegerLiteral *>(Ex)) {
                      try { return std::stoll(std::string(L->Text)); }
                      catch (...) { return std::nullopt; }
                    }
                    if (auto *U = dynamic_cast<const UnaryOpExpr *>(Ex)) {
                      auto V = foldI(U->Operand);
                      if (!V) return std::nullopt;
                      if (U->Op == UnOp::Minus) return -*V;
                      if (U->Op == UnOp::Plus)  return  *V;
                    }
                    return std::nullopt;
                  };
                  auto Sgn = foldI(C->Args[1]);
                  auto WL  = foldI(C->Args[2]);
                  auto FL  = foldI(C->Args[3]);
                  if (Sgn && WL && FL && *Sgn >= 0 && *WL > 0 &&
                      *WL <= 64 && *FL >= 0 && *FL <= *WL) {
                    FixedSpec Spec;
                    Spec.Signed = (*Sgn != 0);
                    Spec.WordLength = (uint8_t)*WL;
                    Spec.FractionLength = (int8_t)*FL;
                    auto It = ParamFiSpec.find(PB);
                    if (It == ParamFiSpec.end()) {
                      ParamFiSpec[PB] = Spec;
                    }
                  }
                  break;
                }
              }
            }
          }
        }
        // Recognized call/index whose callee is a NameExpr bound
        // to one of the function's parameters. Pull out constant
        // integer indices.
        if (auto *N = dynamic_cast<const NameExpr *>(C->Callee)) {
          if (N->Ref) {
            for (Binding *PB : F.ParamRefs) {
              if (PB == N->Ref) {
                if (C->Args.size() == 1 && C->Args[0]) {
                  // Inline a tiny `foldInt` — `foldIntExpr` lives
                  // later in this TU. Only IntegerLiteral / signed-
                  // unary forms are valid Vector indices.
                  std::function<std::optional<int64_t>(const Expr *)>
                      foldI = [&](const Expr *Ex) -> std::optional<int64_t> {
                    if (!Ex) return std::nullopt;
                    if (auto *L = dynamic_cast<const IntegerLiteral *>(Ex)) {
                      try { return std::stoll(std::string(L->Text)); }
                      catch (...) { return std::nullopt; }
                    }
                    if (auto *U = dynamic_cast<const UnaryOpExpr *>(Ex)) {
                      auto V = foldI(U->Operand);
                      if (!V) return std::nullopt;
                      if (U->Op == UnOp::Minus) return -*V;
                      if (U->Op == UnOp::Plus)  return  *V;
                    }
                    return std::nullopt;
                  };
                  auto K = foldI(C->Args[0]);
                  if (K && *K > 0) {
                    auto It = ParamMaxIdx.find(PB);
                    if (It == ParamMaxIdx.end() || It->second < *K)
                      ParamMaxIdx[PB] = *K;
                  }
                }
                break;
              }
            }
          }
        }
        if (C->Callee) walkExpr(*C->Callee);
        for (Expr *A : C->Args) if (A) walkExpr(*A);
        return;
      }
      if (auto *B = dynamic_cast<const BinaryOpExpr *>(&E0)) {
        if (B->LHS) walkExpr(*B->LHS);
        if (B->RHS) walkExpr(*B->RHS);
        return;
      }
      if (auto *U = dynamic_cast<const UnaryOpExpr *>(&E0)) {
        if (U->Operand) walkExpr(*U->Operand);
        return;
      }
      if (auto *P = dynamic_cast<const PostfixOpExpr *>(&E0)) {
        if (P->Operand) walkExpr(*P->Operand);
        return;
      }
      if (auto *M = dynamic_cast<const MatrixLiteral *>(&E0)) {
        for (auto &Row : M->Rows)
          for (Expr *El : Row) if (El) walkExpr(*El);
        return;
      }
      if (auto *R = dynamic_cast<const RangeExpr *>(&E0)) {
        if (R->Start) walkExpr(*R->Start);
        if (R->Step)  walkExpr(*R->Step);
        if (R->End)   walkExpr(*R->End);
        return;
      }
    };
    walkStmt = [&](const Stmt &S) {
      switch (S.Kind) {
      case NodeKind::ExprStmt: {
        auto &Es = static_cast<const ExprStmt &>(S);
        if (Es.E) walkExpr(*Es.E);
        return;
      }
      case NodeKind::AssignStmt: {
        auto &As = static_cast<const AssignStmt &>(S);
        for (Expr *L : As.LHS) if (L) walkExpr(*L);
        if (As.RHS) walkExpr(*As.RHS);
        return;
      }
      case NodeKind::IfStmt: {
        auto &I = static_cast<const IfStmt &>(S);
        if (I.Cond) walkExpr(*I.Cond);
        if (I.Then) walkBlock(*I.Then);
        for (auto &EI : I.Elseifs) {
          if (EI.Cond) walkExpr(*EI.Cond);
          if (EI.Body) walkBlock(*EI.Body);
        }
        if (I.Else) walkBlock(*I.Else);
        return;
      }
      case NodeKind::ForStmt: {
        auto &Fs = static_cast<const ForStmt &>(S);
        if (Fs.Iter) walkExpr(*Fs.Iter);
        if (Fs.Body) walkBlock(*Fs.Body);
        return;
      }
      case NodeKind::WhileStmt: {
        auto &Ws = static_cast<const WhileStmt &>(S);
        if (Ws.Cond) walkExpr(*Ws.Cond);
        if (Ws.Body) walkBlock(*Ws.Body);
        return;
      }
      default: return;
      }
    };
    walkBlock = [&](const ::matlab::Block &B0) {
      for (Stmt *S : B0.Stmts) if (S) walkStmt(*S);
    };
    walkBlock(*F.Body);
  }

  // Parameters start with Any unless the pre-pass found vector-
  // shape or fi-spec evidence. Vector-shape: param appears as
  // `param(k)` with constant k; the param starts as
  // `arrayOf(Unknown, Vector(N))`. fi-spec: param appears in
  // `fi(param, signed, WL, FL)` re-cast; param starts as
  // `fixedScalar(spec)` (or `fixedArray` when both the vector
  // and fi pieces of evidence are present).
  for (size_t PI = 0; PI < F.ParamRefs.size(); ++PI) {
    Binding *B = F.ParamRefs[PI];
    // Sema-time monomorphization (#38, Phase 5). When the
    // monomorphizer has stamped a concrete arg type for this
    // parameter, honor it as the param's initial env type instead of
    // starting at Any. The stamp lives on the Function (not on the
    // Binding) so it survives Resolver re-runs. Builtin-pre-pass
    // evidence (ParamMaxIdx / ParamFiSpec) still wins when present —
    // it represents stronger in-body usage facts that should refine
    // even concrete stamps.
    if (PI < F.ParamTypeStamps.size() && F.ParamTypeStamps[PI] &&
        F.ParamTypeStamps[PI]->K != Type::Kind::Any) {
      auto VIt = ParamMaxIdx.find(B);
      auto FIt = ParamFiSpec.find(B);
      bool HasEvidence =
          (VIt != ParamMaxIdx.end() && VIt->second >= 1) ||
          (FIt != ParamFiSpec.end());
      if (!HasEvidence) {
        E[B] = F.ParamTypeStamps[PI];
        B->InferredType = F.ParamTypeStamps[PI];
        continue;
      }
    }
    auto VIt = ParamMaxIdx.find(B);
    auto FIt = ParamFiSpec.find(B);
    bool HasVec = VIt != ParamMaxIdx.end() && VIt->second >= 1;
    bool HasFi = FIt != ParamFiSpec.end();
    if (HasVec && HasFi) {
      const Type *T = TC.fixedArray(FIt->second,
                                     Shape::vector(VIt->second));
      E[B] = T;
      B->InferredType = T;
    } else if (HasVec) {
      const Type *T = TC.arrayOf(Dtype::Unknown,
                                  Shape::vector(VIt->second));
      E[B] = T;
      B->InferredType = T;
    } else if (HasFi) {
      const Type *T = TC.fixedScalar(FIt->second);
      E[B] = T;
      B->InferredType = T;
    } else {
      E[B] = TC.any();
      B->InferredType = TC.any();
    }
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
    /* Multi-return refinement: when LHS arity > 1 and the RHS is a
     * call to a known multi-return-matrix builtin, give each LHS the
     * appropriate Array type instead of the single fallback RhsT.
     * Without this, `[xx, yy] = meshgrid(...)` typed both `xx` and `yy`
     * as the same fallback (often Any), and downstream `exp(xx)` then
     * fell through to scalar Double — leading to an arith.mulf(f64,
     * !llvm.ptr) op that crashed the LLVM lowering pipeline. */
    const Type *PerLhsT = nullptr;
    if (A.LHS.size() > 1 && A.RHS &&
        A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *C = static_cast<const CallOrIndex *>(A.RHS);
      if (auto *N = dynamic_cast<const NameExpr *>(C->Callee)) {
        // [X, Y] = meshgrid(x, y), [X, Y] = ndgrid(x, y), and the 3-arg
        // forms — all outputs are real-double matrices.
        if (N->Name == "meshgrid" || N->Name == "ndgrid") {
          PerLhsT = TC.arrayOf(Dtype::Double, Shape::matrix(-1, -1));
        }
      }
    }
    for (Expr *L : A.LHS) {
      if (!L) continue;
      if (auto *N = dynamic_cast<NameExpr *>(L)) {
        if (N->Ref) {
          const Type *T = PerLhsT ? PerLhsT : RhsT;
          In[N->Ref] = T;
          N->Ty = T;
        }
      } else if (auto *C = dynamic_cast<CallOrIndex *>(L);
                 C && C->Args.size() == 1 && C->Args[0] &&
                 C->Args[0]->Kind == NodeKind::ColonExpr) {
        /* `lhs(:) = rhs` — type-preserving assignment. For an fi-typed
         * lhs this is the "cast rhs into lhs's spec" idiom that prevents
         * the FixedSpec from re-inferring (and growing unboundedly) on
         * each iteration of an accumulator loop. We keep the existing
         * type in Env and let LowerFixedPoint insert the explicit cast
         * downstream. For non-fi lhs the behavior also preserves the
         * destination type — same as MATLAB's "assign into existing
         * variable's class" rule. */
        visit(*L, In);
        if (auto *N = dynamic_cast<NameExpr *>(C->Callee); N && N->Ref) {
          auto It = In.find(N->Ref);
          if (It != In.end() && It->second) {
            // Keep existing type; annotate the LHS expression accordingly.
            N->Ty = It->second;
            L->Ty = It->second;
          }
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
    // First try the current env.
    for (auto &[B, _] : In) {
      if (B->Name == F.Var) { VarB = B; break; }
    }
    // Otherwise, walk the body AST for any NameExpr that the resolver has
    // already bound for this loop variable — it's a side-door into Sema's
    // binding table that doesn't require threading a scope pointer down.
    if (!VarB && F.Body) {
      std::function<void(const ::matlab::Block &)> walkBlock;
      std::function<void(const Stmt &)>            walkStmt;
      std::function<void(const Expr &)>            walkExpr;
      walkExpr = [&](const Expr &E) {
        if (VarB) return;
        if (E.Kind == NodeKind::NameExpr) {
          auto &N = static_cast<const NameExpr &>(E);
          if (N.Name == F.Var && N.Ref) VarB = N.Ref;
          return;
        }
        for (unsigned i = 0; i < 8 && !VarB; ++i) (void)i; // dummy to keep tidy
        switch (E.Kind) {
        case NodeKind::BinaryOp: {
          auto &B = static_cast<const BinaryOpExpr &>(E);
          if (B.LHS) walkExpr(*B.LHS);
          if (B.RHS) walkExpr(*B.RHS);
          break;
        }
        case NodeKind::UnaryOp:
          if (auto *U = static_cast<const UnaryOpExpr &>(E).Operand) walkExpr(*U);
          break;
        case NodeKind::PostfixOp:
          if (auto *U = static_cast<const PostfixOpExpr &>(E).Operand) walkExpr(*U);
          break;
        case NodeKind::CallOrIndex: {
          auto &C = static_cast<const CallOrIndex &>(E);
          if (C.Callee) walkExpr(*C.Callee);
          for (auto *A : C.Args) if (A) walkExpr(*A);
          break;
        }
        case NodeKind::RangeExpr: {
          auto &R = static_cast<const RangeExpr &>(E);
          if (R.Start) walkExpr(*R.Start);
          if (R.Step)  walkExpr(*R.Step);
          if (R.End)   walkExpr(*R.End);
          break;
        }
        default: break;
        }
      };
      walkStmt = [&](const Stmt &S) {
        if (VarB) return;
        if (S.Kind == NodeKind::ExprStmt) {
          auto &E = static_cast<const ExprStmt &>(S);
          if (E.E) walkExpr(*E.E);
        } else if (S.Kind == NodeKind::AssignStmt) {
          auto &A = static_cast<const AssignStmt &>(S);
          for (auto *L : A.LHS) if (L) walkExpr(*L);
          if (A.RHS) walkExpr(*A.RHS);
        } else if (S.Kind == NodeKind::IfStmt) {
          auto &I = static_cast<const IfStmt &>(S);
          if (I.Cond) walkExpr(*I.Cond);
          if (I.Then) walkBlock(*I.Then);
          for (auto &EI : I.Elseifs) {
            if (EI.Cond) walkExpr(*EI.Cond);
            if (EI.Body) walkBlock(*EI.Body);
          }
          if (I.Else) walkBlock(*I.Else);
        } else if (S.Kind == NodeKind::ForStmt) {
          auto &FS = static_cast<const ForStmt &>(S);
          if (FS.Iter) walkExpr(*FS.Iter);
          if (FS.Body) walkBlock(*FS.Body);
        } else if (S.Kind == NodeKind::WhileStmt) {
          auto &W = static_cast<const WhileStmt &>(S);
          if (W.Cond) walkExpr(*W.Cond);
          if (W.Body) walkBlock(*W.Body);
        }
      };
      walkBlock = [&](const ::matlab::Block &B) {
        for (auto *S : B.Stmts) { if (!VarB && S) walkStmt(*S); }
      };
      walkBlock(*F.Body);
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
      } else if (N.Ref->Kind == BindingKind::Builtin &&
                 (N.Name == "pi" || N.Name == "eps" || N.Name == "Inf" ||
                  N.Name == "NaN" || N.Name == "inf" || N.Name == "nan" ||
                  N.Name == "realmin" || N.Name == "realmax")) {
        /* Bare nullary numeric constants are scalar doubles, not handles —
         * otherwise `pi * v` (handle * array) infers `any`, which then
         * collapses a following sin/cos to a scalar, dropping the shape. */
        T = TC.scalar(Dtype::Double);
      } else if (N.Ref->Kind == BindingKind::Function ||
                 N.Ref->Kind == BindingKind::Builtin) {
        T = TC.funcHandle();
      } else if (N.Ref->InferredType) {
        /* REPL cross-input persistence: the workspace-kind hook may
         * have already stamped a concrete InferredType for an auto-
         * declared name (kind=0 -> scalar double, kind=3 -> string).
         * Seed Env with it so this read-side visit, which runs before
         * any in-TU assign, sees the right shape. Without this, the
         * load falls through to `any` and the lowering picks the
         * generic matlab_ws_get_mat path even when we know the
         * binding holds an f64. */
        T = N.Ref->InferredType;
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
    const Type *BaseT = F.Base ? visit(*F.Base, Env) : nullptr;
    T = TC.any();
    /* fi property access: WordLength / FractionLength / Signed /
     * IntegerLength / Value / int / double — all known from FixedSpec. */
    if (BaseT && BaseT->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*BaseT);
      if (A.Elt == Dtype::Fixed && A.FxSpec) {
        if (F.Field == "WordLength" || F.Field == "FractionLength" ||
            F.Field == "Signed" || F.Field == "IntegerLength") {
          T = TC.scalar(Dtype::Double);
        } else if (F.Field == "Value") {
          T = TC.arrayOf(Dtype::Double, A.S);
        }
      }
    }
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

// Fixed-Point Designer arithmetic-spec promotion.
//
// FullPrecision rules (User's Guide §3.10–3.14). The default for our scalar
// pipeline. KeepLSB / SpecifyPrecision can override; Phase 1 only ships the
// default and the `lhs(:) = rhs` clamp at assignment sites (see visitStmt).
//
// add/sub: align to the larger FL (left-shift smaller side); WL grows by 1.
// mul:     WL_out = WL_a + WL_b; FL_out = FL_a + FL_b.
// Mixed fi + double: cast the double side to the fi spec, then apply.
namespace {

FixedSpec promoteFixedAdd(const FixedSpec &A, const FixedSpec &B) {
  FixedSpec R;
  R.Signed = A.Signed || B.Signed;
  R.FractionLength = std::max(A.FractionLength, B.FractionLength);
  int LeftShiftA = R.FractionLength - A.FractionLength;
  int LeftShiftB = R.FractionLength - B.FractionLength;
  int WL_A = int(A.WordLength) + LeftShiftA;
  int WL_B = int(B.WordLength) + LeftShiftB;
  int WL = std::max(WL_A, WL_B) + 1;
  if (WL > 64) WL = 64; // clamp at the widest native lane
  R.WordLength = uint8_t(WL);
  // Default fimath modes: inherit the more conservative of the two sides
  // (Saturate over Wrap, Nearest over Floor).
  R.OF = (A.OF == FixedSpec::Overflow::Saturate ||
          B.OF == FixedSpec::Overflow::Saturate)
             ? FixedSpec::Overflow::Saturate
             : FixedSpec::Overflow::Wrap;
  R.RM = (A.RM == FixedSpec::Rounding::Nearest ||
          B.RM == FixedSpec::Rounding::Nearest)
             ? FixedSpec::Rounding::Nearest
             : FixedSpec::Rounding::Floor;
  return R;
}

FixedSpec promoteFixedMul(const FixedSpec &A, const FixedSpec &B) {
  FixedSpec R;
  R.Signed = A.Signed || B.Signed;
  int WL = int(A.WordLength) + int(B.WordLength);
  if (WL > 64) WL = 64;
  R.WordLength = uint8_t(WL);
  int FL = int(A.FractionLength) + int(B.FractionLength);
  if (FL > R.WordLength) FL = R.WordLength;
  R.FractionLength = int8_t(FL);
  R.OF = (A.OF == FixedSpec::Overflow::Saturate ||
          B.OF == FixedSpec::Overflow::Saturate)
             ? FixedSpec::Overflow::Saturate
             : FixedSpec::Overflow::Wrap;
  R.RM = (A.RM == FixedSpec::Rounding::Nearest ||
          B.RM == FixedSpec::Rounding::Nearest)
             ? FixedSpec::Rounding::Nearest
             : FixedSpec::Rounding::Floor;
  return R;
}

} // namespace

const Type *TypeInference::visitBinary(BinaryOpExpr &B, Env &Env) {
  const Type *L = B.LHS ? visit(*B.LHS, Env) : TC.any();
  const Type *R = B.RHS ? visit(*B.RHS, Env) : TC.any();

  // Per-op fi handling (FullPrecision rules; Phase 4 will route through a
  // first-class fimath surface). Falls through to the regular numeric
  // promotion path when neither operand is Fixed.
  auto tryFixedBinop = [&](BinOp Op) -> const Type * {
    if (!L || !R || L->K != Type::Kind::Array || R->K != Type::Kind::Array)
      return nullptr;
    auto &LA = static_cast<const ArrayType &>(*L);
    auto &RA = static_cast<const ArrayType &>(*R);
    if (LA.Elt != Dtype::Fixed && RA.Elt != Dtype::Fixed) return nullptr;
    // Mixed fi + non-fi: cast the non-fi to the fi side's spec.
    FixedSpec SL = LA.FxSpec ? *LA.FxSpec : FixedSpec{};
    FixedSpec SR = RA.FxSpec ? *RA.FxSpec : FixedSpec{};
    if (LA.Elt == Dtype::Fixed && RA.Elt != Dtype::Fixed) SR = SL;
    if (RA.Elt == Dtype::Fixed && LA.Elt != Dtype::Fixed) SL = SR;
    if (!LA.FxSpec && LA.Elt == Dtype::Fixed) return nullptr; // unresolved spec
    if (!RA.FxSpec && RA.Elt == Dtype::Fixed) return nullptr;
    Shape Out = broadcastShape(LA.S, RA.S);
    switch (Op) {
    case BinOp::Add: case BinOp::Sub:
      return TC.fixedArray(promoteFixedAdd(SL, SR), Out);
    case BinOp::ElemMul:
    case BinOp::Mul: // matrix * scalar / scalar * matrix collapses to elt mul
      return TC.fixedArray(promoteFixedMul(SL, SR), Out);
    default:
      return nullptr; // div / pow / cmp deferred — fall through.
    }
  };

  switch (B.Op) {
  case BinOp::Add: case BinOp::Sub:
    if (auto *T = tryFixedBinop(B.Op)) return T;
    return TC.broadcastNumeric(L, R);
  case BinOp::ElemMul:
    if (auto *T = tryFixedBinop(B.Op)) return T;
    return TC.broadcastNumeric(L, R);
  case BinOp::ElemDiv:
  case BinOp::ElemLeftDiv: case BinOp::ElemPow:
    return TC.broadcastNumeric(L, R);

  case BinOp::Mul: {
    // Matrix multiply: (M x K) * (K x N) -> (M x N); scalar * X broadcasts.
    if (!L || !R || L->K != Type::Kind::Array || R->K != Type::Kind::Array)
      return TC.any();
    auto &LA = static_cast<const ArrayType &>(*L);
    auto &RA = static_cast<const ArrayType &>(*R);
    // Scalar fi * scalar fi (the apply_gain shape) — promote spec via the
    // mul rule. Vector/matrix fi mul is Phase 3 territory; for now accept
    // scalar-scalar and let the rest fall through.
    if ((LA.Elt == Dtype::Fixed || RA.Elt == Dtype::Fixed) &&
        LA.S.K == Shape::Rank::Scalar && RA.S.K == Shape::Rank::Scalar) {
      if (auto *T = tryFixedBinop(BinOp::Mul)) return T;
    }
    Dtype D = promoteDtype(LA.Elt, RA.Elt);
    if (D == Dtype::Unknown) return TC.any();
    if (D == Dtype::Fixed)   return TC.any(); // unresolved fi mul shape
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
  /* Unary minus on unsigned fi is a MATLAB error; we surface it at lower
   * stages (LowerFixedPoint will refuse to emit). For signed fi the spec
   * is preserved (Saturate / Wrap on the smallest-negative case is a
   * runtime concern). */
  if (U.Op == UnOp::Minus && T && T->K == Type::Kind::Array) {
    auto &A = static_cast<const ArrayType &>(*T);
    if (A.Elt == Dtype::Fixed && A.FxSpec) return T;
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

// Try to constant-fold an expression tree into an int64_t. Returns nullopt
// if any leaf isn't a plain integer literal (possibly behind a unary +/-).
static std::optional<int64_t> foldIntExpr(const Expr *E) {
  if (!E) return std::nullopt;
  if (auto *L = dynamic_cast<const IntegerLiteral *>(E)) {
    try { return std::stoll(std::string(L->Text)); }
    catch (...) { return std::nullopt; }
  }
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(E)) {
    auto V = foldIntExpr(U->Operand);
    if (!V) return std::nullopt;
    if (U->Op == UnOp::Minus) return -*V;
    if (U->Op == UnOp::Plus)  return  *V;
    return std::nullopt;
  }
  return std::nullopt;
}

const Type *TypeInference::visitRange(RangeExpr &R, Env &Env) {
  if (R.Start) visit(*R.Start, Env);
  if (R.Step)  visit(*R.Step,  Env);
  if (R.End)   visit(*R.End,   Env);

  // Try to fold the length. MATLAB range length = floor((end-start)/step)+1,
  // with step defaulting to 1, and 0 elements if the sign of (end-start)
  // doesn't match step.
  auto FS = foldIntExpr(R.Start);
  auto FE = foldIntExpr(R.End);
  int64_t Step = 1;
  if (R.Step) {
    if (auto S = foldIntExpr(R.Step)) Step = *S;
    else return TC.arrayOf(Dtype::Double, Shape::vector(-1));
  }
  if (FS && FE && Step != 0) {
    int64_t Diff = *FE - *FS;
    int64_t Len = (Step > 0 && Diff < 0) || (Step < 0 && Diff > 0)
                    ? 0
                    : Diff / Step + 1;
    return TC.arrayOf(Dtype::Double, Shape::vector(Len));
  }
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
  // Track the first FixedSpec we see so we can pin the result to it
  // when every fi-typed element shares the same spec. The Phase-1
  // spec-propagation rules apply at binop sites; concat is unambiguous
  // — all elements must already be in the same numerictype.
  std::optional<FixedSpec> FxSpec;
  bool AllSameFxSpec = true;
  int64_t TotalElts = 0;
  for (auto &R : M.Rows) {
    int64_t RowCols = 0;
    for (Expr *E : R) {
      if (!E) continue;
      const Type *T = visit(*E, Env);
      if (!T || T->K != Type::Kind::Array) { AllScalars = false; RowCols++; TotalElts++; continue; }
      auto &A = static_cast<const ArrayType &>(*T);
      if (A.S.K != Shape::Rank::Scalar) AllScalars = false;
      if (First) { D = A.Elt; First = false; }
      else       D = promoteDtype(D, A.Elt);
      // Track per-element fi spec / element count.
      int64_t EltCount = 1;
      if (A.S.K == Shape::Rank::Vector && !A.S.Dims.empty() && A.S.Dims[0] >= 0)
        EltCount = A.S.Dims[0];
      else if (A.S.K == Shape::Rank::Matrix && A.S.Dims.size() == 2 &&
               A.S.Dims[0] >= 0 && A.S.Dims[1] >= 0)
        EltCount = A.S.Dims[0] * A.S.Dims[1];
      else if (A.S.K != Shape::Rank::Scalar)
        EltCount = -1;
      if (EltCount < 0) AllScalars = false; // we still know it's a vector/matrix though
      if (A.Elt == Dtype::Fixed) {
        if (!A.FxSpec) { AllSameFxSpec = false; }
        else if (!FxSpec) FxSpec = A.FxSpec;
        else if (!(*FxSpec == *A.FxSpec)) AllSameFxSpec = false;
      }
      TotalElts += (EltCount > 0 ? EltCount : 0);
      RowCols++;
    }
    if (Cols < 0)                Cols = RowCols;
    else if (Cols != RowCols)    Cols = -1;
  }
  if (First) D = Dtype::Double;
  // Fi result path: the matrix literal carries an fi vector/matrix shape
  // and the FixedSpec from the (matched) elements.
  if (D == Dtype::Fixed && FxSpec && AllSameFxSpec) {
    if (Rows == 1) {
      // Row vector: total length = sum of per-element widths (each
      // scalar contributes 1; sub-vector contributes its length).
      int64_t Len = TotalElts > 0 ? TotalElts : -1;
      return TC.fixedArray(*FxSpec, Shape::vector(Len));
    }
    return TC.fixedArray(*FxSpec, Shape::unknown());
  }
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
  if (Name == "magic") {
    // magic(n) -> n×n matrix of double.
    int64_t N = Args.size() == 1 ? foldInt(Args[0]) : -1;
    return TC.arrayOf(Dtype::Double,
                      N > 0 ? Shape::matrix(N, N) : Shape::unknown());
  }
  if (Name == "diag") {
    // diag(vec_of_len_n) -> n×n matrix; diag(matrix) -> column vector of
    // length min(m,n). Without richer shape info we report dynamic.
    return TC.arrayOf(Dtype::Double, Shape::unknown());
  }

  if (Name == "size") {
    // size(A) -> row vector of length ndims(A); size(A,k) -> scalar double
    if (Args.size() >= 2) return TC.scalar(Dtype::Double);
    return TC.arrayOf(Dtype::Double, Shape::vector(-1));
  }
  if (Name == "length" || Name == "numel" || Name == "ndims")
    return TC.scalar(Dtype::Double);

  /* Propagation Models (docs/comm_toolbox_roadmap.md §3). All scalar-
   * returning closed-form path-loss / Fresnel / diffraction / geographic
   * helpers report `scalar(Double)`. The matrix-returning entries
   * (terrainProfile / coverageGrid / coverageGridMulti) return
   * `arrayOf(Double, unknown)`. The struct-returning `linkBudget` falls
   * through to `any()` — its uses are field accesses, which have their
   * own typing path. */
  if (Name == "fspl" || Name == "pathlossHata" || Name == "pathlossCost231" ||
      Name == "pathlossEgli" || Name == "pathlossEcc33" ||
      Name == "pathlossSui" || Name == "pathlossEricsson9999" ||
      Name == "pathlossRain" || Name == "pathlossGas" ||
      Name == "pathlossFog" || Name == "pathlossCloseIn" ||
      Name == "fresnelZoneRadius" || Name == "fresnelClearance" ||
      Name == "diffractionKnifeEdge" || Name == "diffractionBullington" ||
      Name == "diffractionDeygout" ||
      Name == "haversine" || Name == "bearing" || Name == "vincenty" ||
      Name == "greatCircleDestLat" || Name == "greatCircleDestLon" ||
      Name == "itmPathloss" || Name == "losObstruction" || Name == "losClear" ||
      Name == "sectorPattern" || Name == "cosinePattern" ||
      Name == "gaussianPattern" || Name == "isotropicPattern" ||
      Name == "applyMountAz" || Name == "applyMountEl")
    return TC.scalar(Dtype::Double);
  if (Name == "terrainProfile" || Name == "coverageGrid" ||
      Name == "coverageGridMulti" || Name == "applyMountOrientation")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* COMM Tier-1 (docs/comm_toolbox_roadmap.md §2). Function-form
   * base layer; runtime/runtime_comm.cpp. */
  if (Name == "rngGet" ||
      Name == "biterr" || Name == "biterrK" || Name == "biterrCount" ||
      Name == "symerr" || Name == "symerrCount")
    return TC.scalar(Dtype::Double);
  if (Name == "randi") {
    /* randi(imax) -> scalar; multi-arg forms -> matrix. */
    if (Args.size() <= 1) return TC.scalar(Dtype::Double);
    return TC.arrayOf(Dtype::Double, Shape::unknown());
  }
  if (Name == "randsrc" || Name == "randsrcWeighted" ||
      Name == "randerr" ||
      Name == "int2bit" || Name == "bit2int" ||
      Name == "de2bi" || Name == "bi2de" ||
      Name == "awgn")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Tier-2 digital modulation (docs/comm_toolbox_roadmap.md §4). */
  if (Name == "qfunc" || Name == "berawgn")
    return TC.scalar(Dtype::Double);
  if (Name == "pammod" || Name == "pamdemod" ||
      Name == "pskmod" || Name == "pskdemod" ||
      Name == "qammod" || Name == "qamdemod" ||
      Name == "qamdemodBit" || Name == "qamdemodLlr" ||
      Name == "genqammod" || Name == "genqamdemod" ||
      Name == "rcosdesign" || Name == "gaussdesign" ||
      Name == "scatterplot")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Tier-3 channel coding (docs/comm_toolbox_roadmap.md §5). */
  if (Name == "crcCheck" || Name == "oct2dec")
    return TC.scalar(Dtype::Double);
  if (Name == "crcGenerate" || Name == "crcStrip" ||
      Name == "convenc" || Name == "vitdec" ||
      Name == "hammgenParity" ||
      Name == "hammingEncode" || Name == "hammingDecode" ||
      Name == "intrlv" || Name == "deintrlv")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Tier-4 equalisation / sync / RF impairments + soft Viterbi
   * (docs/comm_toolbox_roadmap.md §6). */
  if (Name == "preambleDetect")
    return TC.scalar(Dtype::Double);
  if (Name == "lms" || Name == "rls" || Name == "cma" || Name == "dfe" ||
      Name == "costasPll" || Name == "symbolSyncMM" ||
      Name == "phaseFreqOffset" || Name == "iqimbal" ||
      Name == "memorylessNl" || Name == "phaseNoise" ||
      Name == "vitdecSoft")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Tier-5 OFDM / fading / MIMO. */
  if (Name == "ofdmmod" || Name == "ofdmdemod" ||
      Name == "rayleighChannel" || Name == "ricianChannel" ||
      Name == "ostbcEncode" || Name == "ostbcCombine" ||
      Name == "mlDetect")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Tier-6 spreading + source coding. */
  if (Name == "pnSequence" || Name == "goldSequence" ||
      Name == "hadamard" || Name == "walshCode" ||
      Name == "quantiz" || Name == "quantizApply" ||
      Name == "lloydsQuant" ||
      Name == "compandMu" || Name == "compandA" ||
      Name == "dpcmEncode" || Name == "dpcmDecode")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Tier-7 LDPC / Turbo / Polar. */
  if (Name == "polarEncode" || Name == "polarSCdecode" ||
      Name == "ldpcEncode"  || Name == "ldpcDecodeMS"  ||
      Name == "turboEncode" || Name == "turboDecode")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  /* Partial Differential Equation Toolbox — see
   * docs/pde_toolbox_roadmap.md.  All return either a struct (typed as
   * Array of Double so the struct ptr lowers to !llvm.ptr) or a matrix.
   * The one scalar return is pde_peak_disp_3d. */
  if (Name == "pde_peak_disp_3d" ||
      Name == "pde_result_num_iters" || Name == "pde_result_resid" ||
      Name == "pde_save_stl" ||
      Name == "spnnz" || Name == "sprows" || Name == "spcols" ||
      Name == "pcg_flag" ||
      Name == "pcg_relres" || Name == "pcg_iter" ||
      /* Optimization Toolbox — see docs/optim_toolbox_roadmap.md.
       * Scalar-returning solvers: root finder (`fzero`), 1-D
       * minimiser (`fminbnd`).  Tier-4 problem-based DAG builders all
       * return a scalar node id. */
      Name == "fzero" || Name == "fminbnd" ||
      Name == "matlab_optim_pb_var" || Name == "matlab_optim_pb_const" ||
      Name == "matlab_optim_pb_add" || Name == "matlab_optim_pb_sub" ||
      Name == "matlab_optim_pb_neg" || Name == "matlab_optim_pb_mul" ||
      Name == "matlab_optim_pb_div" || Name == "matlab_optim_pb_pow" ||
      Name == "matlab_optim_pb_le" || Name == "matlab_optim_pb_ge" ||
      Name == "matlab_optim_pb_eq")
    return TC.scalar(Dtype::Double);

  /* `matlab_optim_pb_solve` and the problem-based `solve(prob)` return
   * the solution column vector — but we deliberately leave them
   * `any`-typed so the receiving slot stays `none` and the matrix-slot
   * retyping pass lifts it to `!llvm.ptr` on the ptr-typed store (a
   * `tensor`-typed slot would not retype the same way).  The SymPP
   * `solve` is gated on a sym first argument and is unaffected. */

  /* `fsolve` — scalar x0 gives a scalar root; a vector x0 gives a
   * vector solution.  Decide from the second argument's shape. */
  if (Name == "fsolve") {
    if (Args.size() >= 2 && ArgTys[1] &&
        ArgTys[1]->K == Type::Kind::Array &&
        static_cast<const ArrayType *>(ArgTys[1])->S.K == Shape::Rank::Scalar)
      return TC.scalar(Dtype::Double);
    return TC.arrayOf(Dtype::Double, Shape::unknown());
  }
  if (Name == "pde_mesh_rect_tri" || Name == "pde_boundary_nodes_rect" ||
      Name == "pde_assemble_poisson_2d" || Name == "pde_apply_dirichlet" ||
      Name == "pde_mesh_cuboid_tet" || Name == "pde_face_nodes" ||
      Name == "pde_assemble_elast_3d" || Name == "pde_face_pressure_3d" ||
      Name == "pde_apply_fixed_3d" || Name == "pde_reshape_disp_3d" ||
      Name == "pde_von_mises_3d" || Name == "pde_node_von_mises_3d" ||
      Name == "pde_sys_K" || Name == "pde_sys_F" || Name == "pde_sys_M" ||
      Name == "pde_mesh_nodes" || Name == "pde_mesh_triangles" ||
      Name == "pde_mesh_tets" || Name == "pde_mesh_faces" ||
      Name == "pde_assemble_transient_2d" || Name == "pde_eigsmall" ||
      Name == "pde_step_forward_euler_2d" || Name == "pde_init_uniform_2d" ||
      Name == "pde_solve_nonlinear_2d" || Name == "pde_result_solution" ||
      Name == "pde_load_stl" || Name == "pde_load_glb" ||
      Name == "sparse" || Name == "speye" || Name == "spdiag" ||
      Name == "sparse_matvec" || Name == "spfull" ||
      Name == "pcg" || Name == "pcg_x" ||
      Name == "pde_assemble_poisson_2d_sparse" ||
      Name == "pde_apply_dirichlet_sparse" ||
      Name == "pde_assemble_elast_3d_sparse" ||
      Name == "pde_apply_fixed_3d_sparse" ||
      Name == "pde_sys_K_sparse" ||
      Name == "pde_voxelize_surface" ||
      Name == "pde_solve_femodel" || Name == "pde_solve" ||
      Name == "pde_kernel_mesh" || Name == "pde_kernel_u" ||
      Name == "pde_kernel_vm" ||
      Name == "pde_set_material" || Name == "pde_set_face_fixed" ||
      Name == "pde_set_face_pressure" || Name == "pde_generate_mesh" ||
      Name == "pde_multicylinder" || Name == "pde_multicylinder_hollow" ||
      Name == "pde_multisphere" ||
      Name == "pde_translate" || Name == "pde_rotate" || Name == "pde_scale" ||
      Name == "pde_set_face_temperature" || Name == "pde_set_face_heat" ||
      Name == "pde_set_face_voltage" || Name == "pde_set_face_charge" ||
      Name == "pde_set_body_heat" || Name == "pde_set_body_charge" ||
      Name == "pde_solve_thermal_steady" ||
      Name == "pde_solve_electrostatic" ||
      Name == "pde_solve_magnetostatic" ||
      Name == "pde_solve_dc_conduction" ||
      Name == "pde_set_face_potential" ||
      Name == "pde_set_face_current" ||
      Name == "pde_set_body_current" ||
      Name == "pde_solve_structural_transient" ||
      Name == "pde_set_time_step" || Name == "pde_set_num_steps" ||
      Name == "pde_kernel_uhist" || Name == "pde_kernel_tlist" ||
      Name == "pde_solve_structural_modal" ||
      Name == "pde_set_num_modes" ||
      Name == "pde_kernel_freqs" ||
      Name == "pde_eig_lanczos_si" ||
      Name == "pde_eig_lanczos_si_full" ||
      Name == "pde_eig_lambda" || Name == "pde_eig_phi" ||
      Name == "pde_solve_structural_frequency" ||
      Name == "pde_set_freq_list" ||
      Name == "pde_kernel_freqlist" ||
      Name == "pde_solve_harmonic_em" ||
      Name == "pde_set_wave_number" ||
      Name == "pde_solve_structural_transient_modal" ||
      Name == "pde_set_rayleigh" || Name == "pde_set_modal_results" ||
      Name == "minres" || Name == "sparse_gmres_ilu0" ||
      Name == "pde_mesh_quadratic" ||
      Name == "pde_assemble_elast_3d_t10" ||
      Name == "pde_face_pressure_3d_t10" ||
      Name == "pde_face_nodes_t10" ||
      Name == "pde_node_von_mises_3d_t10" ||
      Name == "pde_apply_fixed_3d_t10" ||
      Name == "pde_solve_thermal_transient" ||
      Name == "pde_set_initial_temperature" ||
      Name == "pde_set_cell_temperature" ||
      Name == "pde_set_reference_temperature" ||
      Name == "pde_reduce" || Name == "reduce" ||
      Name == "pde_reconstruct_solution" || Name == "reconstructSolution" ||
      Name == "pde_refine_mesh" || Name == "refineMesh" ||
      Name == "pde_adapt_mesh"  || Name == "adaptmesh"  ||
      Name == "pde_solve_structural_static_nl" ||
      Name == "pde_set_multi_coeff" || Name == "pde_solve_multi" ||
      Name == "pde_multi_u" || Name == "pde_multi_v" ||
      Name == "pde_set_interface_face" ||
      Name == "pde_reduce_craig_bampton" ||
      Name == "pde_solve_structural_static_tl" ||
      Name == "pde_refine_mesh_bey" || Name == "refineMeshBey" ||
      Name == "pde_adapt_mesh_marked" ||
      Name == "pde_set_multi_coeff_n" ||
      Name == "pde_solve_multi_n" ||
      Name == "pde_multi_n_u" ||
      Name == "solvepde" || Name == "solvepdeeig" ||
      Name == "specifyCoefficients" ||
      Name == "applyBoundaryCondition" ||
      Name == "pde_assemble_poisson_3d_sparse" ||
      Name == "pde_apply_dirichlet_3d_sparse" ||
      Name == "pde_face_scalar_load_3d" ||
      /* Optimization Toolbox — vector-returning solvers: N-D
       * minimisers (`fminsearch`, `fminunc`), linear programming
       * (`linprog`), non-negative least squares (`lsqnonneg`), and
       * the Tier-2 constrained / least-squares solvers (`fmincon`,
       * `quadprog`, `lsqlin`, `lsqnonlin`, `lsqcurvefit`).  `fsolve`
       * is handled separately — scalar form returns a scalar, N-D
       * form returns a vector. */
      Name == "fminsearch" || Name == "fminunc" ||
      Name == "linprog" || Name == "lsqnonneg" ||
      Name == "fmincon" || Name == "quadprog" || Name == "lsqlin" ||
      Name == "lsqnonlin" || Name == "lsqcurvefit" ||
      /* Tier-3 — MILP, cone, minimax, goal-attainment, semi-infinite
       * all return a solution vector. */
      Name == "intlinprog" || Name == "coneprog" || Name == "fminimax" ||
      Name == "fgoalattain" || Name == "fseminf")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

  if (Name == "linspace") {
    int64_t N = -1;
    if (Args.size() >= 3) N = foldInt(Args[2]);
    return TC.arrayOf(Dtype::Double,
                      N > 0 ? Shape::vector(N) : Shape::vector(-1));
  }

  /* Nullary numeric constants — typed as scalar Double.  Without this they
   * default to `any`, which makes `pi * v` (and `2*pi*x/...`) collapse: the
   * multiply yields `any`, and a following sin/cos then infers a scalar
   * result, dropping the vector shape.  `Inf`/`NaN`/`eps`/realmin/realmax
   * are scalar only in their 0-arg form (the sized forms build matrices). */
  if (Name == "pi" ||
      ((Name == "eps" || Name == "Inf" || Name == "NaN" ||
        Name == "inf" || Name == "nan" ||
        Name == "realmin" || Name == "realmax") && Args.empty())) {
    return TC.scalar(Dtype::Double);
  }

  if (Name == "abs" || Name == "sqrt" || Name == "exp" ||
      Name == "log" || Name == "sin"  || Name == "cos" || Name == "tan" ||
      /* Degree-argument trigonometry — element-wise like sin/cos. */
      Name == "sind"  || Name == "cosd"  || Name == "tand" ||
      Name == "asind" || Name == "acosd" || Name == "atand") {
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

  // Dtype-cast builtins. Shape-preserving for the cases that have a
  // matrix-aware runtime path; the others fall back to a scalar result.
  // Phase 1.1 (Option B) wires int32/uint8 matrix descriptors; the other
  // widths still go scalar until their typed runtime lands.
  auto castShape = [&](Dtype D) -> const Type * {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array)
      return TC.arrayOf(D, static_cast<const ArrayType &>(*ArgTys[0]).S);
    return TC.scalar(D);
  };
  if (Name == "double")  return castShape(Dtype::Double);
  if (Name == "single")  return castShape(Dtype::Single);
  if (Name == "int8")    return TC.scalar(Dtype::Int8);
  if (Name == "int16")   return TC.scalar(Dtype::Int16);
  if (Name == "int32")   return castShape(Dtype::Int32);
  if (Name == "int64")   return TC.scalar(Dtype::Int64);
  if (Name == "uint8")   return castShape(Dtype::UInt8);
  if (Name == "uint16")  return TC.scalar(Dtype::UInt16);
  if (Name == "uint32")  return TC.scalar(Dtype::UInt32);
  if (Name == "uint64")  return TC.scalar(Dtype::UInt64);
  if (Name == "logical") return TC.scalar(Dtype::Logical);
  if (Name == "char")    return TC.arrayOf(Dtype::Char, Shape::unknown());

  //===--- Fixed-Point Designer (fi) -------------------------------------===//
  // fi(value) / fi(value, signed, WL) / fi(value, signed, WL, FL).
  // Constant args fold here (almost always literals in real fi code) so the
  // FixedSpec is concrete by the time MLIR sees it. fi(value, T) and
  // fi(value, T, F) — with numerictype/fimath objects — are Phase-4 surface
  // and fall through to TC.any() for now.
  if (Name == "fi") {
    const Type *ValT = !ArgTys.empty() ? ArgTys[0] : nullptr;
    Shape OutShape = Shape::scalar();
    if (ValT && ValT->K == Type::Kind::Array)
      OutShape = static_cast<const ArrayType &>(*ValT).S;
    FixedSpec Spec; // defaults: signed Q15.16 — MATLAB's fi(value) default
    /* Phase 4: fi(value, T) and fi(value, T, F) — read the spec out of
     * a numerictype object (compile-time). Phase 1 explicit-args form
     * (fi(value, signed, WL, FL)) is the fall-through. */
    bool UsedNumerictype = false;
    if (Args.size() >= 2 && ArgTys.size() >= 2 && ArgTys[1] &&
        ArgTys[1]->K == Type::Kind::Numerictype) {
      auto &NT = static_cast<const NumerictypeType &>(*ArgTys[1]);
      Spec.Signed = NT.Signed;
      Spec.WordLength = NT.WordLength;
      Spec.FractionLength = NT.FractionLength;
      UsedNumerictype = true;
    }
    if (UsedNumerictype && Args.size() >= 3 && ArgTys[2] &&
        ArgTys[2]->K == Type::Kind::Fimath) {
      auto &FM = static_cast<const FimathType &>(*ArgTys[2]);
      Spec.OF = FM.OF;
      Spec.RM = FM.RM;
    }
    if (!UsedNumerictype && Args.size() >= 3) {
      int64_t Sgn = foldInt(Args[1]);
      int64_t WL  = foldInt(Args[2]);
      if (Sgn < 0 || WL <= 0 || WL > 64) return TC.any();
      Spec.Signed = (Sgn != 0);
      Spec.WordLength = uint8_t(WL);
      // Default fraction length per MATLAB fi: WL-1 for signed, WL for
      // unsigned, but only when the user omitted FL.
      Spec.FractionLength = Spec.Signed ? int8_t(WL - 1) : int8_t(WL);
    }
    if (!UsedNumerictype && Args.size() >= 4) {
      int64_t FL = foldInt(Args[3]);
      if (FL < 0 || FL > Spec.WordLength) return TC.any();
      Spec.FractionLength = int8_t(FL);
    }
    if (!UsedNumerictype && Args.size() == 2) {
      /* fi(value, T) where T didn't resolve as a numerictype. Bail. */
      return TC.any();
    }
    return TC.fixedArray(Spec, OutShape);
  }
  // numerictype(signed, WL, FL) — compile-time object carrying the spec.
  if (Name == "numerictype") {
    if (Args.size() == 3) {
      int64_t Sgn = foldInt(Args[0]);
      int64_t WL  = foldInt(Args[1]);
      int64_t FL  = foldInt(Args[2]);
      if (Sgn >= 0 && WL > 0 && WL <= 64 && FL >= 0 && FL <= WL)
        return TC.numerictype(Sgn != 0, uint8_t(WL), int8_t(FL));
    }
    return TC.any();
  }
  // fimath('OverflowAction', 'Saturate'|'Wrap',
  //        'RoundingMethod', 'Floor'|'Nearest'). We accept name-value
  // pairs in any order and ignore unknown names (forward-compatible
  // for properties we don't yet model).
  if (Name == "fimath") {
    FixedSpec::Overflow OF = FixedSpec::Overflow::Saturate;
    FixedSpec::Rounding RM = FixedSpec::Rounding::Floor;
    auto literalText = [](Expr *E) -> std::string {
      if (auto *S = dynamic_cast<StringLiteral *>(E)) return S->Value;
      if (auto *S = dynamic_cast<CharLiteral *>(E))   return S->Value;
      return "";
    };
    for (size_t i = 0; i + 1 < Args.size(); i += 2) {
      std::string K = literalText(Args[i]);
      std::string V = literalText(Args[i + 1]);
      if (K == "OverflowAction") {
        if (V == "Wrap")     OF = FixedSpec::Overflow::Wrap;
        else if (V == "Saturate") OF = FixedSpec::Overflow::Saturate;
      } else if (K == "RoundingMethod") {
        if (V == "Nearest") RM = FixedSpec::Rounding::Nearest;
        else if (V == "Floor") RM = FixedSpec::Rounding::Floor;
        else if (V == "Zero") RM = FixedSpec::Rounding::Zero;
        else if (V == "Convergent") RM = FixedSpec::Rounding::Convergent;
        else if (V == "Ceiling") RM = FixedSpec::Rounding::Ceiling;
      }
    }
    return TC.fimath(OF, RM);
  }
  if (Name == "fipref") return TC.any();
  // int(n) / storedInteger(n) / storedIntegerToDouble(n) — return the native
  // integer behind a fi value. For non-fi inputs, behave like a no-op.
  if (Name == "int" || Name == "storedInteger") {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      if (A.Elt == Dtype::Fixed && A.FxSpec) {
        Dtype D = Dtype::Int64;
        switch (A.FxSpec->storageBits()) {
        case 8:  D = A.FxSpec->Signed ? Dtype::Int8  : Dtype::UInt8;  break;
        case 16: D = A.FxSpec->Signed ? Dtype::Int16 : Dtype::UInt16; break;
        case 32: D = A.FxSpec->Signed ? Dtype::Int32 : Dtype::UInt32; break;
        default: D = A.FxSpec->Signed ? Dtype::Int64 : Dtype::UInt64; break;
        }
        return TC.arrayOf(D, A.S);
      }
    }
    return TC.any();
  }
  if (Name == "storedIntegerToDouble") {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      return TC.arrayOf(Dtype::Double, A.S);
    }
    return TC.scalar(Dtype::Double);
  }
  // reinterpretcast(n, T) — bit-reinterpret stored int as a different
  // numerictype without changing storage. The two specs must have
  // matching WL (the storage lane is the same); FL/signedness can swap.
  if (Name == "reinterpretcast") {
    if (Args.size() < 2 || ArgTys.size() < 2 || !ArgTys[0] || !ArgTys[1])
      return TC.any();
    if (ArgTys[1]->K != Type::Kind::Numerictype) return TC.any();
    auto &NT = static_cast<const NumerictypeType &>(*ArgTys[1]);
    Shape OutShape = Shape::scalar();
    if (ArgTys[0]->K == Type::Kind::Array)
      OutShape = static_cast<const ArrayType &>(*ArgTys[0]).S;
    FixedSpec NS;
    NS.Signed = NT.Signed;
    NS.WordLength = NT.WordLength;
    NS.FractionLength = NT.FractionLength;
    return TC.fixedArray(NS, OutShape);
  }
  // setfimath(n, F) returns a fi with F's overflow/rounding overriding
  // n's; WL/FL stay. removefimath(n) resets both to defaults (Saturate /
  // Floor). For non-fi inputs, both are no-ops.
  if (Name == "removefimath" || Name == "setfimath") {
    if (ArgTys.empty() || !ArgTys[0] || ArgTys[0]->K != Type::Kind::Array)
      return TC.any();
    auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
    if (A.Elt != Dtype::Fixed || !A.FxSpec) return ArgTys[0];
    FixedSpec NS = *A.FxSpec;
    if (Name == "removefimath") {
      NS.OF = FixedSpec::Overflow::Saturate;
      NS.RM = FixedSpec::Rounding::Floor;
    } else {
      // setfimath(n, F): the fimath argument carries the overrides.
      if (Args.size() >= 2 && ArgTys.size() >= 2 && ArgTys[1] &&
          ArgTys[1]->K == Type::Kind::Fimath) {
        auto &FM = static_cast<const FimathType &>(*ArgTys[1]);
        NS.OF = FM.OF;
        NS.RM = FM.RM;
      }
    }
    if (A.S.K == Shape::Rank::Scalar) return TC.fixedScalar(NS);
    return TC.fixedArray(NS, A.S);
  }
  // bin / hex / dec — render a fi as a string. Returns char array.
  if (Name == "bin" || Name == "hex" || Name == "dec")
    return TC.arrayOf(Dtype::Char, Shape::unknown());

  // sum / mean / min / max on a fi array return a fi scalar with the
  // input's spec preserved. (FullPrecision sum widens by log2(N) bits;
  // for the FIR shape that doesn't matter because the accumulator's
  // spec is set explicitly via fi(0, 1, 36, 28) etc.)
  if ((Name == "sum" || Name == "mean" || Name == "min" || Name == "max") &&
      !ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
    auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
    if (A.Elt == Dtype::Fixed && A.FxSpec)
      return TC.fixedScalar(*A.FxSpec);
  }

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
        // Snapshot per-arg inferred types onto the Call node so the Sema-time
        // monomorphizer (#38, Phase 1) can bucket call sites by signature
        // without re-walking the type-inference environment.
        C.ArgTypes.clear();
        C.ArgTypes.reserve(C.Args.size());
        for (Expr *A : C.Args)
          C.ArgTypes.push_back(A ? A->Ty : nullptr);
        // Tier-6 — cross-function return-type propagation. When the
        // callee has been visited earlier in the TU walk (e.g. an
        // inner subsystem helper that was emitted before its
        // outer-subsystem caller), its `OutputRefs[0]->Ty` carries
        // the type Sema inferred for the first output. Use it as
        // the call result. The Embedded Coder lane orders TU
        // entries inner-first so this fires naturally for nested
        // subsystems; ordinary user `.m` files where outers
        // reference inners declared later in the same file still
        // fall back to `Any` (matches the prior behaviour).
        Function *F = N->Ref->FuncDef;
        if (!F->OutputRefs.empty() && F->OutputRefs[0]) {
          if (auto *FTy = F->OutputRefs[0]->InferredType) return FTy;
        }
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
      auto &Arr = static_cast<const ArrayType &>(*C.Callee->Ty);

      // Classify each index: scalar / range-of-known-length / colon-all /
      // unknown-vector. Returns (length, known). `length == -1, known=true`
      // means "use the callee's dim as-is" (colon).
      auto classifyIdx = [&](const Expr *A) -> std::pair<int64_t, bool> {
        if (!A) return {-1, false};
        if (A->Kind == NodeKind::ColonExpr) return {-1, true};
        if (auto *R = dynamic_cast<const RangeExpr *>(A)) {
          if (R->Ty && R->Ty->K == Type::Kind::Array) {
            auto &RT = static_cast<const ArrayType &>(*R->Ty);
            if (RT.S.K == Shape::Rank::Vector && !RT.S.Dims.empty() &&
                RT.S.Dims[0] >= 0)
              return {RT.S.Dims[0], true};
          }
          return {-1, false};
        }
        if (!A->Ty || A->Ty->K != Type::Kind::Array) return {-1, false};
        auto &AT = static_cast<const ArrayType &>(*A->Ty);
        if (AT.S.K == Shape::Rank::Scalar) return {1, true};
        if (AT.S.K == Shape::Rank::Vector && !AT.S.Dims.empty() &&
            AT.S.Dims[0] >= 0)
          return {AT.S.Dims[0], true};
        return {-1, false};
      };

      // Helpers to construct results that preserve the FixedSpec when
      // the source array is fi-typed.
      auto fiScalar = [&]() -> const Type * {
        if (Arr.Elt == Dtype::Fixed && Arr.FxSpec)
          return TC.fixedScalar(*Arr.FxSpec);
        return TC.scalar(Arr.Elt);
      };
      auto fiArray = [&](Shape S) -> const Type * {
        if (Arr.Elt == Dtype::Fixed && Arr.FxSpec)
          return TC.fixedArray(*Arr.FxSpec, std::move(S));
        return TC.arrayOf(Arr.Elt, std::move(S));
      };

      // Bit-slice extension: `x(hi:lo)` on a scalar integer with constant
      // descending range. Only the form `x(hi:lo)` (no explicit step) with
      // hi >= lo >= 0 and hi < bitwidth(x) is recognized. The result type
      // rounds the slice width up to the next native int {1,8,16,32,64};
      // bits above the slice are zero in the result. MATLAB itself treats
      // `x(7:0)` on a scalar as an empty array, so this overlay doesn't
      // shadow valid MATLAB code.
      if (Arr.S.K == Shape::Rank::Scalar && isInteger(Arr.Elt) &&
          C.Args.size() == 1) {
        if (auto *R = dynamic_cast<const RangeExpr *>(C.Args[0])) {
          if (!R->Step) {
            auto FS = foldIntExpr(R->Start);
            auto FE = foldIntExpr(R->End);
            if (FS && FE) {
              int64_t Hi = *FS, Lo = *FE;
              int SrcW = 0;
              switch (Arr.Elt) {
              case Dtype::Int8: case Dtype::UInt8: SrcW = 8; break;
              case Dtype::Int16: case Dtype::UInt16: SrcW = 16; break;
              case Dtype::Int32: case Dtype::UInt32: SrcW = 32; break;
              case Dtype::Int64: case Dtype::UInt64: SrcW = 64; break;
              case Dtype::Logical: SrcW = 1; break;
              case Dtype::Fixed:
                if (Arr.FxSpec) SrcW = (int)Arr.FxSpec->WordLength;
                break;
              default: break;
              }
              int64_t SliceW = Hi - Lo + 1;
              if (SrcW > 0 && Hi >= Lo && Lo >= 0 && Hi < SrcW &&
                  SliceW >= 1 && SliceW <= 64) {
                Dtype RD;
                if      (SliceW == 1)  RD = Dtype::Logical;
                else if (SliceW <= 8)  RD = Dtype::UInt8;
                else if (SliceW <= 16) RD = Dtype::UInt16;
                else if (SliceW <= 32) RD = Dtype::UInt32;
                else                    RD = Dtype::UInt64;
                return TC.scalar(RD);
              }
            }
          }
        }
      }

      // All scalar indices collapse to a scalar element.
      bool AllScalar = true;
      for (const Expr *Arg : C.Args) {
        auto [L, Known] = classifyIdx(Arg);
        if (!(Known && L == 1)) { AllScalar = false; break; }
      }
      if (AllScalar) return fiScalar();

      // Try to recover a ranked result when we're doing 2D subscripting and
      // each index's output length is known (either folded or a colon whose
      // length is the matching callee dim).
      if (C.Args.size() == 2 && Arr.S.K == Shape::Rank::Matrix &&
          Arr.S.Dims.size() == 2) {
        auto [L0, K0] = classifyIdx(C.Args[0]);
        auto [L1, K1] = classifyIdx(C.Args[1]);
        if (K0 && K1) {
          int64_t R = (L0 < 0) ? Arr.S.Dims[0] : L0;
          int64_t Co = (L1 < 0) ? Arr.S.Dims[1] : L1;
          if (R == 1 && Co >= 0)
            return fiArray(Shape::vector(Co));
          if (Co == 1 && R >= 0)
            return fiArray(Shape::matrix(R, 1));
          if (R >= 0 && Co >= 0)
            return fiArray(Shape::matrix(R, Co));
        }
      }
      // One-arg indexing of a vector: return a vector of the index length.
      if (C.Args.size() == 1 && Arr.S.K == Shape::Rank::Vector) {
        auto [L, K] = classifyIdx(C.Args[0]);
        if (K) {
          if (L < 0 && !Arr.S.Dims.empty()) L = Arr.S.Dims[0];
          if (L >= 0) return fiArray(Shape::vector(L));
        }
      }
      return fiArray(Shape::unknown());
    }
    if (C.Callee->Ty->K == Type::Kind::StringArray) {
      return TC.stringScalar();
    }
  }
  return TC.any();
}

} // namespace matlab
