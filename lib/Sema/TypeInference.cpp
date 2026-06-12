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
  /* Class methods are ordinary free functions at this stage (Resolver
   * registered them as Function bindings with a class-pinned first
   * param). Walk them too so their bodies get the same type-inference
   * treatment as top-level functions — otherwise comparisons like
   * `nargin == 1` inside a method body keep a NoneType result and
   * scf.if lowering breaks. */
  auto inferAllFunctions = [&]() {
    for (Function *F : TU.Functions) runFunction(*F);
    for (ClassDef *C : TU.Classes) {
      for (Function *M : C->Methods) if (M) runFunction(*M);
      for (Function *M : C->StaticMethods) if (M) runFunction(*M);
    }
  };
  /* #191 P2.1 — inter-procedural return-type propagation. A call to a user
   * function returns the callee's first-output type (visitCallOrIndex reads
   * OutputRefs[0]->InferredType), but that is only set once the callee's body
   * has been inferred. Previously the script was walked FIRST, so every call
   * from the script body to a local function — the usual MATLAB layout, where
   * functions are defined after the script body — saw a null output type and
   * fell to Any. Infer all function bodies first (pass 1 populates each output
   * type), once more (pass 2 resolves one level of forward inter-function
   * calls: A defined before B but calling B), then the script LAST so its call
   * sites see fully-populated output types. Two passes (not a fixpoint) keep
   * this bounded; deeper transitive forward chains degrade to Any as before
   * rather than looping. */
  inferAllFunctions();
  inferAllFunctions();
  if (TU.ScriptNode) runScript(*TU.ScriptNode);
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
      } else if (auto *FA = dynamic_cast<FieldAccess *>(L);
                 FA && dynamic_cast<NameExpr *>(FA->Base)) {
        // #191 P4.2: `s.field = rhs` accumulates the field's type into a
        // per-binding struct type, so a later `s.field` read recovers it.
        // OpenSet stays true (other fields may exist / be added later).
        auto *BN = static_cast<NameExpr *>(FA->Base);
        const Type *FldT = PerLhsT ? PerLhsT : RhsT;
        if (BN->Ref) {
          std::map<std::string, const Type *> Fields;
          auto It = In.find(BN->Ref);
          if (It != In.end() && It->second &&
              It->second->K == Type::Kind::Struct)
            Fields = static_cast<const StructType &>(*It->second).Fields;
          Fields[std::string(FA->Field)] = FldT;
          const Type *ST = TC.structWith(std::move(Fields), /*OpenSet=*/true);
          In[BN->Ref] = ST;
          BN->Ty = ST;
          L->Ty = FldT;
        } else {
          visit(*L, In);
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
      } else if (N.Ref->Kind == BindingKind::Builtin &&
                 (N.Name == "true" || N.Name == "false")) {
        /* Bare `true` / `false` are logical scalars, not function handles.
         * Mistyping them as @handle is mostly masked today, but #191 P2.1
         * propagates a function's return type to its call sites, so a
         * `function b = f(); b = true; end` would otherwise surface @handle at
         * every `f()` call. (The `true(n)` array form is a call, handled
         * elsewhere.) */
        T = TC.scalar(Dtype::Logical);
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
    // #191 P4.2: a struct field read recovers the type recorded when the
    // field was assigned (`s.x = expr`). Unknown fields stay Any (OpenSet).
    if (BaseT && BaseT->K == Type::Kind::Struct) {
      auto &S = static_cast<const StructType &>(*BaseT);
      auto It = S.Fields.find(std::string(F.Field));
      if (It != S.Fields.end() && It->second) T = It->second;
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
    // #191 P4.1: an anonymous function has a known input arity (its params)
    // and always yields a single output.
    T = TC.funcHandleArity(static_cast<int>(A.Params.size()), 1);
    break;
  }
  case NodeKind::FuncHandle: {
    // #191 P4.1: `@userfn` takes its arity from the resolved function's
    // declared inputs/outputs; `@builtin` (no FuncDef) stays unknown (-1).
    auto &H = static_cast<FuncHandle &>(E);
    if (H.Ref && H.Ref->Kind == BindingKind::Function && H.Ref->FuncDef) {
      const Function *F = H.Ref->FuncDef;
      T = TC.funcHandleArity(static_cast<int>(F->Inputs.size()),
                             static_cast<int>(F->Outputs.size()));
    } else {
      T = TC.funcHandle();
    }
    break;
  }
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

  // #234: a CHAR literal in arithmetic / comparison promotes to its numeric
  // code(s) — `'A' + 1 == 66`, `'a' - 'A' == 32` — unlike a string literal
  // ("A"), which concatenates. A CharLiteral is typed stringScalar by visit(),
  // so without this the result collapses to a string/ptr and the matlab.add
  // can't lower. Re-type a CharLiteral operand as numeric (scalar for a single
  // char, 1xN row for a multi-char literal) for the non-concat operators; the
  // matching lowering converts the operand to its code value. StringLiteral is
  // left alone so `"A" + 1` still concatenates.
  {
    /* ARITHMETIC ops only — NOT comparison.  Char arithmetic (`'A' + 1`) is
     * unambiguously numeric, but a comparison like `strvar == 'x'` (string
     * var vs single char) must keep its existing element/string semantics, so
     * re-typing the char operand numeric there would change behaviour. */
    bool NumericOp =
        B.Op == BinOp::Add || B.Op == BinOp::Sub || B.Op == BinOp::Mul ||
        B.Op == BinOp::Div || B.Op == BinOp::LeftDiv || B.Op == BinOp::Pow ||
        B.Op == BinOp::ElemMul || B.Op == BinOp::ElemDiv ||
        B.Op == BinOp::ElemLeftDiv || B.Op == BinOp::ElemPow;
    if (NumericOp) {
      auto charNumeric = [&](Expr *Op, const Type *T) -> const Type * {
        /* A single-char literal promotes to a scalar code (`'A' + 1`); a
         * multi-char literal promotes to a 1xN row of codes (`'AB' + 1` ->
         * [66 67]). The matching BinaryOp lowering materialises the code(s)
         * (a const_float scalar or a concat_row matrix). A StringLiteral is
         * left alone so `"AB" + 1` still concatenates. */
        if (Op && Op->Kind == NodeKind::CharLiteral) {
          size_t n = static_cast<CharLiteral &>(*Op).Value.size();
          if (n == 1) return TC.scalar(Dtype::Double);
          if (n > 1)
            return TC.arrayOf(Dtype::Double,
                              Shape::matrix(1, static_cast<int64_t>(n)));
        }
        return T;
      };
      L = charNumeric(B.LHS, L);
      R = charNumeric(B.RHS, R);
    }
    /* Comparison: re-type a single-char CharLiteral operand numeric ONLY when
     * the OTHER operand is numeric (`'A' == 65`, `c >= '0'` with c numeric).
     * Gating on a numeric counterpart keeps `strvar == 'x'` and `'x' == 'y'`
     * (both string-typed operands) on their existing string/element path, so
     * no comparison behaviour changes except char-vs-number, which MATLAB
     * does evaluate on codes. */
    bool CmpOp = B.Op == BinOp::Eq || B.Op == BinOp::Ne || B.Op == BinOp::Lt ||
                 B.Op == BinOp::Le || B.Op == BinOp::Gt || B.Op == BinOp::Ge;
    if (CmpOp) {
      auto isNumeric = [](const Type *T) {
        return T && T->K == Type::Kind::Array;
      };
      auto isSingleChar = [](Expr *Op) {
        return Op && Op->Kind == NodeKind::CharLiteral &&
               static_cast<CharLiteral &>(*Op).Value.size() == 1;
      };
      if (isSingleChar(B.LHS) && isNumeric(R)) L = TC.scalar(Dtype::Double);
      else if (isSingleChar(B.RHS) && isNumeric(L)) R = TC.scalar(Dtype::Double);
    }
  }

  // #40: arithmetic involving a class instance dispatches to the class's
  // operator overload (plus / minus / mtimes / ...), which by convention
  // returns an instance of the same class. Preserving the object type keeps
  // chained expressions (`(G + H) * K`) and downstream constructor / call
  // args concrete instead of collapsing to Any. This mirrors the same
  // same-class assumption the resolver already makes when it pins operands
  // (docs/sema.md §3.4). Comparison / logical operators are left alone —
  // their overloads return logical, not the class.
  {
    auto objClass = [](const Type *T) -> const ClassDef * {
      return (T && T->K == Type::Kind::Object)
                 ? static_cast<const ObjectType &>(*T).Class
                 : nullptr;
    };
    switch (B.Op) {
    case BinOp::Add: case BinOp::Sub: case BinOp::Mul: case BinOp::Div:
    case BinOp::LeftDiv: case BinOp::Pow:
    case BinOp::ElemMul: case BinOp::ElemDiv:
    case BinOp::ElemLeftDiv: case BinOp::ElemPow:
      if (const ClassDef *CD = objClass(L)) return TC.objectOf(CD);
      if (const ClassDef *CD = objClass(R)) return TC.objectOf(CD);
      break;
    default:
      break;
    }
  }

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
    // Any when an operand isn't a numeric array: object operands are already
    // handled by the operator-overload path (visitBinary's #40 block above),
    // so reaching here means a string/cell/unresolved operand whose product
    // type can't be determined statically.
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
    if (D == Dtype::Unknown) return TC.any(); // operand dtype unresolved
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
    // Any when an operand isn't a numeric array (object overloads handled
    // earlier; string/cell/unresolved operands have no static quotient type).
    if (!L || !R || L->K != Type::Kind::Array || R->K != Type::Kind::Array)
      return TC.any();
    auto &LA = static_cast<const ArrayType &>(*L);
    auto &RA = static_cast<const ArrayType &>(*R);
    Dtype D = promoteDtype(LA.Elt, RA.Elt);
    if (D == Dtype::Unknown) return TC.any(); // operand dtype unresolved
    if (LA.S.K == Shape::Rank::Scalar && RA.S.K == Shape::Rank::Scalar)
      return TC.scalar(D);
    return TC.arrayOf(D, Shape::unknown());
  }
  case BinOp::Pow: {
    // Scalar^scalar -> scalar. Matrix power has different semantics.
    // Any when an operand isn't a numeric array (object overloads handled
    // earlier; otherwise the power result type is statically unknown).
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
  // Defensive: every BinOp is handled in the switch above, so this is
  // unreachable in practice — Any only if a new operator is added without a
  // case, where an unknown result type is the safe default.
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
  // Postfix here is transpose (' / .'). On a non-array operand (object with a
  // transpose overload, string, unresolved) the result type isn't statically
  // known — object dispatch happens at lowering, so Any is correct.
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
  const Type *CT = C.Callee ? visit(*C.Callee, Env) : nullptr;
  for (Expr *A : C.Args) if (A) visit(*A, Env);
  // #191 P4.3: a brace-index `c{i}` yields the cell's element type when the
  // cell was built homogeneously (visitCellLit recorded ElementUpperBound).
  // A single brace index returns one element; a heterogeneous / untyped cell
  // (ElementUpperBound null) still falls to Any.
  if (CT && CT->K == Type::Kind::Cell) {
    auto &Cell = static_cast<const CellType &>(*CT);
    if (Cell.ElementUpperBound) return Cell.ElementUpperBound;
  }
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
  // #191 P4.3: join the element types so a homogeneous literal (`{1,2,3}`,
  // `{"a","b"}`) carries an element type and a later `c{i}` recovers it. A
  // mixed literal joins to Any and degrades to cellAny() (cellOf handles that).
  const Type *Elt = nullptr;
  bool Any = false;
  for (auto &R : M.Rows)
    for (Expr *E : R)
      if (E) {
        const Type *ET = visit(*E, Env);
        Elt = Elt ? TC.join(Elt, ET) : ET;
        if (!Elt || Elt->K == Type::Kind::Any) Any = true;
      }
  return (Any || !Elt) ? TC.cellAny() : TC.cellOf(Elt);
}

//===----------------------------------------------------------------------===//
// Builtins
//===----------------------------------------------------------------------===//

// #191 P3: an arithmetic operator overload (plus/mtimes/mpower/...) conventionally
// returns an instance of the same class — the same assumption visitBinary's #40
// block makes for the BinaryOp form. After the P3 rewrite turns `a op b` into a
// method call, method-result typing must apply the same convention, because some
// operator method bodies (e.g. tf.mpower's `r=a; while: r=r*a`) have a
// param-dependent output that infers as Any, which would break a chained
// expression `(a^2)+b` at lowering. Comparison operators return logical, not the
// class, so they are excluded.
static bool isArithOperatorMethod(std::string_view N) {
  return N == "plus" || N == "minus" || N == "uminus" || N == "uplus" ||
         N == "mtimes" || N == "mrdivide" || N == "mldivide" || N == "mpower" ||
         N == "times" || N == "rdivide" || N == "ldivide" || N == "power";
}

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

  // #233: strsplit(s[, delim]) -> a cell of string tokens.
  if (Name == "strsplit")
    return TC.cellOf(TC.stringScalar());

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
    // zeros/ones/...(d1, d2, d3, ...) with >= 3 scalar dim args is a rank-N
    // array. Report a positive NDArray rank (one Dims entry per arg, -1 where
    // the extent isn't a foldable constant) so the clone-on-assign gate in
    // Lowering deep-copies the matlab_matN buffer on `B = A` (issue #102).
    // Unknown rank was excluded from that gate, which let `B = A` alias.
    std::vector<int64_t> Dims;
    Dims.reserve(Args.size());
    for (Expr *A : Args) Dims.push_back(foldInt(A));
    return TC.arrayOf(D, Shape::ndarray(std::move(Dims)));
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

  /* File I/O ids/status are scalar doubles. Without this they fall through
   * to `any()`, and in the JIT/-dap workspace lane an `any` scalar reads
   * back as a boxed matlab_mat* (matlab_ws_get_mat) — so `fprintf(fid, …)`
   * sees a ptr where the file-id lowering needs an f64 and the launch fails
   * to compile (#77). fopen -> fid (scalar); fclose/fseek/ftell/feof ->
   * status/position (scalar). (Multi-output `[fid,msg] = fopen(...)` still
   * types fid scalar via the first-output path.) */
  if (Name == "fopen" || Name == "fclose" || Name == "fseek" ||
      Name == "ftell" || Name == "feof" || Name == "frewind")
    return TC.scalar(Dtype::Double);

  /* Continuous wavelet transform — `[wt, f] = cwt(x, fs)` yields a
   * coefficient MATRIX (scales x time), not a scalar. Unmodelled it fell
   * through to `any`, so the first output typed as a scalar; in the
   * JIT/-dap workspace lane the var then read back via matlab_ws_get_f64
   * while the store held a real matlab_mat*, and `size(mag, 2)` had no
   * (f64, dim) lowering. Report a dynamic-shape double matrix. */
  if (Name == "cwt")
    return TC.arrayOf(Dtype::Double, Shape::unknown());

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
      /* Issue #28 — geometry + mesher front door (struct-returning). */
      Name == "multicuboid" || Name == "decsg" || Name == "createpde" ||
      Name == "geometryFromEdges" || Name == "generateMesh" ||
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
      Name == "log" || Name == "log10" || Name == "log2" ||
      Name == "sin"  || Name == "cos" || Name == "tan" ||
      Name == "sinh" || Name == "cosh" || Name == "tanh" ||
      /* sign — element-wise, real result (drops to floating like the rest). */
      Name == "sign" ||
      /* Degree-argument trigonometry — element-wise like sin/cos. */
      Name == "sind"  || Name == "cosd"  || Name == "tand" ||
      Name == "asind" || Name == "acosd" || Name == "atand") {
    /* If the argument is (or reduces to) a class-pinned binding — e.g.
     * a `dlarray` — back off to `any` so the dispatch path picks the
     * classdef method instead of forcing a scalar/array numeric type.
     * Without this the result alloc is typed `f64` while the classdef
     * method returns a pointer, producing an alloc/store type mismatch.
     *
     * Recurse through composite expressions (BinaryOp / UnaryOp / a
     * CallOrIndex of a dlarray-returning builtin) so `sqrt(v + eps_dl)`
     * routes correctly when `v` and `eps_dl` are dlarray. */
    if (!Args.empty()) {
      std::function<bool(const Expr *)> argIsClassPinned =
          [&argIsClassPinned](const Expr *X) -> bool {
        if (!X) return false;
        if (auto *NE = dynamic_cast<const NameExpr *>(X))
          return NE->Ref && NE->Ref->PinnedClass;
        if (auto *Bi = dynamic_cast<const BinaryOpExpr *>(X))
          return argIsClassPinned(Bi->LHS) || argIsClassPinned(Bi->RHS);
        if (auto *U = dynamic_cast<const UnaryOpExpr *>(X))
          return argIsClassPinned(U->Operand);
        if (auto *CX = dynamic_cast<const CallOrIndex *>(X)) {
          /* Direct ctor call on a class — pinned. */
          if (auto *NX = dynamic_cast<const NameExpr *>(CX->Callee)) {
            if (NX->Ref && NX->Ref->Kind == BindingKind::Class &&
                NX->Ref->ClassDef) return true;
          }
          /* Reduce/activation calls (`sqrt(x)`, `mean(x,1)`, ...) where
           * the FIRST arg is class-pinned — the result inherits the
           * pin via the classdef method. */
          for (Expr *A : CX->Args)
            if (argIsClassPinned(A)) return true;
        }
        return false;
      };
      // A class-pinned arg routes through the class's method overload at
      // runtime (`abs`/`sqrt`/… on an object), so the result type is the
      // method's — not statically known here. Typing it as object<Class> is
      // the job of #191 P1.1 (method-call result typing, depends on P2.1);
      // until then Any defers to runtime dispatch (intentional).
      if (argIsClassPinned(Args[0])) return TC.any();
    }
    // Element-wise: preserves shape, promotes to floating.
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      Dtype D = isFloating(A.Elt) ? A.Elt : Dtype::Double;
      return TC.arrayOf(D, A.S);
    }
    return TC.scalar(Dtype::Double);
  }
  if (Name == "mod" || Name == "rem" || Name == "floor" ||
      Name == "ceil" || Name == "round" || Name == "fix" ||
      /* bitshift(a, n) — element-wise on a; preserves a's shape and type. */
      Name == "bitshift") {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      return TC.arrayOf(A.Elt, A.S);
    }
    return TC.scalar(Dtype::Double);
  }
  /* Scalar math builtins added with the runtime _s forms (scalar args). */
  if (Name == "log1p" || Name == "expm1" ||
      Name == "nextpow2" || Name == "hypot" || Name == "nthroot" ||
      Name == "gcd" || Name == "lcm" ||
      Name == "nchoosek")
    return TC.scalar(Dtype::Double);

  /* primes(n) / factor(n) (#235): a row vector; the length is data-dependent
   * so the column count is unknown (-1). Only the 1-arg numeric form is typed
   * here — the symbolic factor(expr, var) is the 2-arg form and is handled by
   * the sym path (exprIsSym), so we don't claim it as numeric. */
  if (Name == "primes" || (Name == "factor" && Args.size() == 1))
    return TC.arrayOf(Dtype::Double, Shape::vector(-1));

  /* isprime / factorial (#235): element-wise, shape-preserving for an
   * array argument; scalar otherwise (matches the shipped _s scalar form). */
  if (Name == "isprime" || Name == "factorial") {
    if (!ArgTys.empty() && ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      return TC.arrayOf(Dtype::Double, A.S);
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
    // transpose of a non-array (object with a transpose overload, or an
    // unresolved arg): result type isn't known statically.
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
  // strfind(s, pat) -> 1xk row vector of 1-based match positions.
  if (Name == "strfind") return TC.arrayOf(Dtype::Double, Shape::vector(-1));
  // str2num(s) -> numeric matrix parsed from the string (#235). Shape is
  // data-dependent (scalar / vector / 2-D), so it's unknown until runtime.
  if (Name == "str2num") return TC.arrayOf(Dtype::Double, Shape::unknown());
  // regexp(s, pat) default form -> 1xk row vector of 1-based match starts
  // (#235). The cell-returning option modes ('match'/'tokens') are not
  // lowered; regexprep returns a string and uses the default (like strrep).
  if (Name == "regexp") return TC.arrayOf(Dtype::Double, Shape::vector(-1));

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
      // signedness / word-length didn't fold to a valid constant — the fi
      // spec is undetermined, so the value's type is unknown.
      if (Sgn < 0 || WL <= 0 || WL > 64) return TC.any();
      Spec.Signed = (Sgn != 0);
      Spec.WordLength = uint8_t(WL);
      // Default fraction length per MATLAB fi: WL-1 for signed, WL for
      // unsigned, but only when the user omitted FL.
      Spec.FractionLength = Spec.Signed ? int8_t(WL - 1) : int8_t(WL);
    }
    if (!UsedNumerictype && Args.size() >= 4) {
      int64_t FL = foldInt(Args[3]);
      // fraction-length didn't fold to a valid constant — fi spec undetermined.
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
    // non-3-arg form or args that didn't fold to a valid spec — the
    // numerictype object can't be modelled at compile time.
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
  // fipref configures fi *display* preferences and has no numeric value /
  // type to model — Any is correct (intentional, not a precision gap).
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
    // int()/storedInteger() on a non-fi / non-array arg: without a fi spec
    // there's no stored-integer width to infer, so the type is unknown.
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
    // missing operands — can't determine the reinterpreted spec.
    if (Args.size() < 2 || ArgTys.size() < 2 || !ArgTys[0] || !ArgTys[1])
      return TC.any();
    // 2nd arg isn't a resolved numerictype — target spec unknown.
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
    // no array operand to carry/strip a fimath — type unknown.
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

  // #191 P2.2 — general (non-fi) reductions. Reduction results lower to boxed
  // matrix pointers, so every typed result here is a ptr-SHAPED type (Matrix
  // rank, never bare Scalar): a Scalar-rank type would make the slot/return f64
  // while the body value is a ptr (the gpu/multiret/bode regressions). Cases:
  //   * sum/prod/mean/median/std/var of a MATRIX -> 1xN row (default dim-1);
  //   * sum/prod/mean/median/std/var of a VECTOR/SCALAR -> 1x1 matrix;
  //   * min/max(x, y) elementwise with a non-scalar result.
  // Single-arg min/max stays Any: it feeds the `[v,i]=min(x)` multi-output
  // index path that relies on the Any result.
  {
    bool isMinMax = (Name == "min" || Name == "max");
    bool isFloatReduce =
        (Name == "mean" || Name == "median" || Name == "std" || Name == "var");
    bool isSumProd = (Name == "sum" || Name == "prod");
    if ((isMinMax || isFloatReduce || isSumProd) && !ArgTys.empty() &&
        ArgTys[0] && ArgTys[0]->K == Type::Kind::Array) {
      auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
      if (A.Elt != Dtype::Fixed) {
        Dtype RD = A.Elt;
        if (isFloatReduce)
          RD = (A.Elt == Dtype::Complex) ? Dtype::Complex : Dtype::Double;
        else if (A.Elt == Dtype::Logical)
          RD = Dtype::Double; // sum/prod/min/max of logical promote to double

        // min(x, y) / max(x, y): elementwise. Take the non-scalar operand's
        // shape (covers max(x, 0) and max(a, b)); only when non-scalar.
        if (isMinMax && ArgTys.size() == 2 && ArgTys[1] &&
            ArgTys[1]->K == Type::Kind::Array) {
          auto &B = static_cast<const ArrayType &>(*ArgTys[1]);
          const Shape &RS = (A.S.K != Shape::Rank::Scalar) ? A.S : B.S;
          if (RS.K != Shape::Rank::Scalar && RS.K != Shape::Rank::Unknown)
            return TC.arrayOf(RD, RS);
        }

        // sum/prod/mean/median/std/var, no explicit dim. Matrix -> 1xN row;
        // vector/scalar -> 1x1 matrix (ptr-shaped, never bare Scalar).
        if (!isMinMax && ArgTys.size() == 1) {
          if (A.S.K == Shape::Rank::Matrix && A.S.Dims.size() >= 2)
            return TC.arrayOf(RD, Shape::matrix(1, A.S.Dims[1]));
          if (A.S.K == Shape::Rank::Vector || A.S.K == Shape::Rank::Scalar)
            return TC.arrayOf(RD, Shape::matrix(1, 1));
        }
      }
    }
  }

  // norm(x) / norm(x, p) / norm(x, 'fro') — always a real scalar. Type it as
  // a 1x1 MATRIX (ptr-shaped, not a bare scalar) for the same reason as the
  // reductions above: the result lowers to a boxed matrix pointer.
  if (Name == "norm" && !ArgTys.empty() && ArgTys[0] &&
      ArgTys[0]->K == Type::Kind::Array)
    return TC.arrayOf(Dtype::Double, Shape::matrix(1, 1));

  // reshape(x, m, n) — same element type, shape from the (foldable) scalar
  // dim args. Result is a matrix (ptr) so no scalar box/unbox concern. The
  // [m n]-vector form and `[]`-placeholder form fall through to Any.
  if (Name == "reshape" && ArgTys.size() == 3 && ArgTys[0] &&
      ArgTys[0]->K == Type::Kind::Array) {
    auto &A = static_cast<const ArrayType &>(*ArgTys[0]);
    int64_t R = foldInt(Args[1]);
    int64_t C = foldInt(Args[2]);
    if (R > 0 && C > 0)
      return TC.arrayOf(A.Elt, Shape::matrix(R, C));
  }

  if (Name == "disp" || Name == "fprintf" || Name == "warning" ||
      Name == "error") {
    return TC.any(); // effectively void
  }

  // Default for any builtin not special-cased above: its return type isn't
  // modelled here, so Any. Builtins whose result feeds further inference
  // should get an explicit case; this is the catch-all safe default.
  // #191 P2.2 probe: name the unmodelled builtins whose args ARE typed
  // (those are the ones poisoning downstream call-arg precision).
  if (::getenv("MATLAB_LLVM_PROBE_ANYBUILTIN")) {
    bool allTyped = !ArgTys.empty();
    for (const Type *T : ArgTys)
      if (!T || T->K == Type::Kind::Any) { allTyped = false; break; }
    if (allTyped)
      fprintf(stderr, "[any-builtin] %.*s (%zu typed args)\n",
              (int)Name.size(), Name.data(), ArgTys.size());
  }
  return TC.any();
}

const Type *TypeInference::visitCallOrIndex(CallOrIndex &C, Env &Env) {
  // Callee must be visited to annotate its type; treat it specially so we
  // don't box a function reference into Any.
  if (C.Callee) visit(*C.Callee, Env);

  if (C.Resolved == CallKind::Call) {
    // #191 P5: snapshot per-arg inferred types onto the call so the Sema-time
    // monomorphizer can bucket the call site by signature. Done for plain
    // function calls below; this helper lets class constructor / instance-
    // method calls share the same machinery (their args must be visited
    // first). Call it once the args have been visited in each branch.
    auto snapshotArgTypes = [&]() {
      C.ArgTypes.clear();
      C.ArgTypes.reserve(C.Args.size());
      for (Expr *A : C.Args) C.ArgTypes.push_back(A ? A->Ty : nullptr);
    };
    if (auto *N = dynamic_cast<NameExpr *>(C.Callee)) {
      // #191 P3 prerequisite — function-style instance-method dispatch:
      // `meth(obj, ...)` where the first argument is a class instance and the
      // class (or a super) defines a method `meth` returns that method's
      // inferred output type, not Any. This mirrors P1.1 (the obj.method()
      // FieldAccess-callee form) for the NameExpr-callee form, and keeps the P3
      // operator rewrite (op -> method call) from dropping the object<Class>
      // result type. Only fires when the class actually has the method, so
      // generic builtins on objects (size(obj), disp(obj), ...) are unaffected.
      if (!C.Args.empty() && C.Args[0]) {
        const Type *A0T = visit(*C.Args[0], Env);
        if (A0T && A0T->K == Type::Kind::Object) {
          const ClassDef *CD = static_cast<const ObjectType &>(*A0T).Class;
          for (const ClassDef *CC = CD; CC; CC = CC->Super)
            for (Function *M : CC->Methods)
              if (M && M->Name == N->Name) {
                for (Expr *A : C.Args) if (A) visit(*A, Env);
                snapshotArgTypes();
                // Arithmetic operator overloads return the class (#40 convention),
                // even when the body's own output infers as Any (param-dependent).
                if (isArithOperatorMethod(N->Name))
                  return TC.objectOf(CD);
                if (!M->OutputRefs.empty() && M->OutputRefs[0] &&
                    M->OutputRefs[0]->InferredType)
                  return M->OutputRefs[0]->InferredType;
                return TC.any();
              }
        }
      }
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
        // The callee's first output type isn't available (defined later in
        // the TU, recursive, or itself Any). Inter-procedural propagation for
        // this case is #191 P2.1; until then Any.
        return TC.any();
      }
      if (N->Ref && N->Ref->Kind == BindingKind::Class &&
          N->Ref->ClassDef) {
        // Constructor call `ClassName(args)`. The result is an instance of
        // that class — give it a concrete object type (#40) instead of the
        // old `Any`, so the constructor monomorphizer can bucket call sites
        // by class and the lowerer maps the value to a matlab_obj* ptr.
        for (Expr *A : C.Args) if (A) visit(*A, Env);
        snapshotArgTypes();
        return TC.objectOf(N->Ref->ClassDef);
      }
    }
    // #191 P1.1: `obj.method(args)` — when the base is a class instance, look
    // up the method on the class and return its inferred first-output type
    // (e.g. `scale` returning `Pt(...)` makes `p.scale(2)` infer object<Pt>).
    // The base was already visited above (visit(*C.Callee) recurses into the
    // FieldAccess base), so its type is annotated. Depends on P2.1 having
    // inferred the method bodies (run() walks class methods before the script).
    if (auto *FA = dynamic_cast<FieldAccess *>(C.Callee)) {
      for (Expr *A : C.Args) if (A) visit(*A, Env);
      snapshotArgTypes();
      const Type *BaseT = FA->Base ? FA->Base->Ty : nullptr;
      if (BaseT && BaseT->K == Type::Kind::Object) {
        const ClassDef *CD = static_cast<const ObjectType &>(*BaseT).Class;
        // Arithmetic operator overloads return the class (#40 convention), even
        // when the method body's own output infers as Any (param-dependent) —
        // keeps a chained operator rewrite `(a^2).plus(b)` typed object<Class>.
        if (CD && isArithOperatorMethod(FA->Field))
          return TC.objectOf(CD);
        if (CD)
          for (const ClassDef *CC = CD; CC; CC = CC->Super)
            for (Function *M : CC->Methods)
              if (M && M->Name == FA->Field) {
                if (!M->OutputRefs.empty() && M->OutputRefs[0] &&
                    M->OutputRefs[0]->InferredType)
                  return M->OutputRefs[0]->InferredType;
                return TC.any();
              }
      }
      return TC.any();
    }
    // Callee isn't a resolved name (e.g. a computed/handle callee, or a
    // call through an expression result) — the return type is unknown.
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
  // Indexing a callee whose type isn't an Array or StringArray (cell, struct,
  // object, or unresolved): the indexed element type isn't modelled here.
  // Struct-field and cell-element typing are #191 P4.2 / P4.3.
  return TC.any();
}

} // namespace matlab
