// DispatchDesynth.cpp — #191 P3. See DispatchDesynth.h / docs.
//
// Rewrites `a <op> b` (where an operand is an instance of an allow-listed
// class with the matching operator method) into the explicit method call
// `a.<opmethod>(b)` — a FieldAccess-callee CallOrIndex. A non-object operand is
// boxed into a one-arg constructor `Class(value)`. The rewritten form is
// type-checked by P1.1 and lowers identically to the previously-synthesized
// `Class__<opmethod>`, so this is a behavior-preserving move of the dispatch
// from lowering time to Sema time (making it visible to P2/P5).

#include "matlab/Sema/DispatchDesynth.h"

#include "matlab/AST/AST.h"
#include "matlab/Sema/Scope.h" // Binding::PinnedClass
#include "matlab/Sema/Type.h"

namespace matlab {
namespace sema {

namespace {

const ClassDef *objClassOf(const Expr *E, bool KeyOffPinnedClass) {
  if (!E)
    return nullptr;
  // Primary signal: Sema stamped the operand's type as `object<Class>`.
  if (E->Ty && E->Ty->K == Type::Kind::Object)
    return static_cast<const ObjectType &>(*E->Ty).Class;
  // Fallback (whole-program / AOT only): a class-typed variable whose value
  // carries only the binding's `PinnedClass`, not an object `Expr->Ty`. This
  // happens when the value comes from a builtin/function that returns the class
  // via the pin side-channel (e.g. `sys = c2d(G, Ts)` — a tf, pinned but not
  // object-typed). The lowering SYNTHESIS path keys off exactly this binding
  // (`pinnedFromExpr`), so mirroring it here lets desynth cover the same
  // operators instead of leaving them to synthesis. Re-running TypeInference
  // after the rewrite (see p3DesynthDispatch) then stamps the rewritten method
  // call with the proper `object<Class>` result type, so a surrounding operator
  // in a later pass sees the object and rewrites too.
  //
  // OFF for cross-turn -repl: there the operand is a cross-turn binding with no
  // object Expr->Ty, and desynthing it into a method call whose base is that
  // binding crashes the dispatch lowering at runtime; the synthesis fallback
  // (taken because the no-op rewrite leaves the BinaryOp in place) is correct.
  if (KeyOffPinnedClass && E->Kind == NodeKind::NameExpr) {
    const auto *NE = static_cast<const NameExpr *>(E);
    if (NE->Ref && NE->Ref->PinnedClass)
      return NE->Ref->PinnedClass;
  }
  return nullptr;
}

// MATLAB operator → method name (same mapping the lowering synthesis uses).
const char *opMethodName(BinOp Op) {
  switch (Op) {
  case BinOp::Add:         return "plus";
  case BinOp::Sub:         return "minus";
  case BinOp::Mul:         return "mtimes";
  case BinOp::Div:         return "mrdivide";
  case BinOp::LeftDiv:     return "mldivide";
  case BinOp::Pow:         return "mpower";
  case BinOp::ElemMul:     return "times";
  case BinOp::ElemDiv:     return "rdivide";
  case BinOp::ElemLeftDiv: return "ldivide";
  case BinOp::ElemPow:     return "power";
  case BinOp::Eq:          return "eq";
  case BinOp::Ne:          return "ne";
  case BinOp::Lt:          return "lt";
  case BinOp::Le:          return "le";
  case BinOp::Gt:          return "gt";
  case BinOp::Ge:          return "ge";
  default:                 return nullptr;
  }
}

// Classes whose 1-arg constructor implements scalar promotion (`G + 2` ==
// `G + tf(2)`). Mirrors the lowering's scalar-mixing allow-list; only these
// box a non-object operand into a constructor call. Other classes (e.g. a
// user Vec2) take the raw scalar — `a * 3` calls `Vec2__mtimes(a, 3)`.
bool boxSafe(std::string_view N) {
  return N == "tf" || N == "ss" || N == "zpk" || N == "pid" || N == "frd" ||
         N == "OptimizationExpression";
}

bool classHasMethod(const ClassDef *CD, std::string_view Name) {
  for (const ClassDef *CC = CD; CC; CC = CC->Super)
    for (const Function *M : CC->Methods)
      if (M && M->Name == Name)
        return true;
  return false;
}

struct Rewriter {
  ASTContext &Ctx;
  const std::set<std::string> &Allow;
  bool KeyOffPinnedClass = false;
  int Count = 0;

  bool allowed(const ClassDef *CD) const {
    return CD && Allow.count(std::string(CD->Name)) > 0;
  }

  const ClassDef *classOf(const Expr *E) const {
    return objClassOf(E, KeyOffPinnedClass);
  }

  // Box a non-object operand into a one-arg ctor `Class(value)`.
  Expr *boxScalar(const ClassDef *CD, Expr *Val) {
    auto *NE = Ctx.make<NameExpr>();
    NE->Name = CD->Name;
    NE->Ref = CD->Self; // class binding, so it resolves as a constructor
    auto *Call = Ctx.make<CallOrIndex>();
    Call->Callee = NE;
    Call->Args.push_back(Val);
    Call->Resolved = CallKind::Call;
    return Call;
  }

  Expr *rewriteExpr(Expr *E) {
    if (!E)
      return E;
    switch (E->Kind) {
    case NodeKind::BinaryOp: {
      auto *B = static_cast<BinaryOpExpr *>(E);
      B->LHS = rewriteExpr(B->LHS);
      B->RHS = rewriteExpr(B->RHS);
      const ClassDef *L = classOf(B->LHS);
      const ClassDef *R = classOf(B->RHS);
      // The FieldAccess base must be an object operand DIRECTLY: rewrite only
      // when the LHS is the class instance, so the base is `B->LHS` (already an
      // object). `obj op X` -> `obj.<op>(X)`, with X boxed only for a box-safe
      // class (`G + 2` -> `G.plus(tf(2))`), raw otherwise (`a * 3` ->
      // `a.mtimes(3)`).
      //
      // The mirror case `X op obj` (scalar on the LHS, e.g. `2 * G`) would need
      // the boxed scalar as the FieldAccess base — `(tf(2)).mtimes(G)` — but the
      // method-dispatch lowering can't yet take a constructor-call base (it
      // segfaults), so that case is left to the lowering synthesis fallback
      // (which boxes both operands and emits Class__op directly). Follow-on:
      // teach the lowering to accept a call-result method base, then rewrite
      // `X op obj` here too.
      const ClassDef *Obj = nullptr;
      Expr *Base = nullptr;
      Expr *Arg = nullptr;
      if (L) {
        Obj = L;
        Base = B->LHS;
        Arg = (R || !boxSafe(L->Name)) ? B->RHS : boxScalar(L, B->RHS);
      }
      if (Obj && allowed(Obj)) {
        if (const char *M = opMethodName(B->Op)) {
          if (classHasMethod(Obj, M)) {
            auto *FA = Ctx.make<FieldAccess>();
            FA->Base = Base;
            FA->Field = M; // string-literal storage, stable
            auto *Call = Ctx.make<CallOrIndex>();
            Call->Callee = FA;
            Call->Args.push_back(Arg);
            Call->Resolved = CallKind::Call;
            // Carry the original operator's inferred type onto the rewritten
            // call. The pass rewrites bottom-up and TypeInference hasn't re-run
            // yet, so a freshly-created CallOrIndex has Ty == null; without this
            // a PARENT operator (`(a*b) / c`) would see a null-typed operand,
            // skip the rewrite, and lower as a raw matrix op on object pointers
            // (matlab.matdiv on two tf objects -> crash). The rewritten method
            // call has the same result type as the operator it replaces.
            Call->Ty = B->Ty;
            FA->Ty = B->Ty;
            ++Count;
            return Call;
          }
        }
      }
      return B;
    }
    case NodeKind::UnaryOp: {
      auto *U = static_cast<UnaryOpExpr *>(E);
      U->Operand = rewriteExpr(U->Operand);
      // #191 P3: de-synthesize a unary operator on a class instance into an
      // explicit 0-arg method call (`-obj` -> `obj.uminus()`), mirroring the
      // BinaryOp path so it is visible to P2/P5 instead of synthesized at
      // lowering (Lowering.cpp UnaryOp class-pinned path). The method-dispatch
      // lowering passes the base as the implicit first arg, emitting
      // Owner__uminus(obj) — identical to the synthesis it replaces.
      const ClassDef *Obj = classOf(U->Operand);
      if (Obj && allowed(Obj)) {
        const char *M = nullptr;
        switch (U->Op) {
        case UnOp::Minus: M = "uminus"; break;
        case UnOp::Plus:  M = "uplus";  break;
        case UnOp::Not:   M = "not";    break;
        }
        if (M && classHasMethod(Obj, M)) {
          auto *FA = Ctx.make<FieldAccess>();
          FA->Base = U->Operand;
          FA->Field = M; // string-literal storage, stable
          auto *Call = Ctx.make<CallOrIndex>();
          Call->Callee = FA;
          Call->Resolved = CallKind::Call;
          // Carry the operator's inferred type so a parent operator sees a
          // typed operand (see the BinaryOp note above).
          Call->Ty = U->Ty;
          FA->Ty = U->Ty;
          ++Count;
          return Call;
        }
      }
      return U;
    }
    case NodeKind::PostfixOp: {
      auto *P = static_cast<PostfixOpExpr *>(E);
      P->Operand = rewriteExpr(P->Operand);
      return P;
    }
    case NodeKind::RangeExpr: {
      auto *Rg = static_cast<RangeExpr *>(E);
      Rg->Start = rewriteExpr(Rg->Start);
      Rg->Step = rewriteExpr(Rg->Step);
      Rg->End = rewriteExpr(Rg->End);
      return Rg;
    }
    case NodeKind::CallOrIndex: {
      auto *C = static_cast<CallOrIndex *>(E);
      C->Callee = rewriteExpr(C->Callee);
      for (auto &A : C->Args)
        A = rewriteExpr(A);
      return C;
    }
    case NodeKind::CellIndex: {
      auto *C = static_cast<CellIndex *>(E);
      C->Callee = rewriteExpr(C->Callee);
      for (auto &A : C->Args)
        A = rewriteExpr(A);
      return C;
    }
    case NodeKind::FieldAccess: {
      auto *F = static_cast<FieldAccess *>(E);
      F->Base = rewriteExpr(F->Base);
      return F;
    }
    case NodeKind::DynamicField: {
      auto *F = static_cast<DynamicField *>(E);
      F->Base = rewriteExpr(F->Base);
      F->Name = rewriteExpr(F->Name);
      return F;
    }
    case NodeKind::MatrixLiteral: {
      auto *M = static_cast<MatrixLiteral *>(E);
      for (auto &Row : M->Rows)
        for (auto &Elt : Row)
          Elt = rewriteExpr(Elt);
      return M;
    }
    case NodeKind::CellLiteral: {
      auto *M = static_cast<CellLiteral *>(E);
      for (auto &Row : M->Rows)
        for (auto &Elt : Row)
          Elt = rewriteExpr(Elt);
      return M;
    }
    case NodeKind::AnonFunction: {
      auto *A = static_cast<AnonFunction *>(E);
      A->Body = rewriteExpr(A->Body);
      return A;
    }
    default:
      return E;
    }
  }

  void rewriteBlock(Block *B) {
    if (!B)
      return;
    for (Stmt *S : B->Stmts)
      rewriteStmt(S);
  }

  void rewriteStmt(Stmt *S) {
    if (!S)
      return;
    switch (S->Kind) {
    case NodeKind::ExprStmt:
      static_cast<ExprStmt *>(S)->E = rewriteExpr(static_cast<ExprStmt *>(S)->E);
      break;
    case NodeKind::AssignStmt: {
      auto *A = static_cast<AssignStmt *>(S);
      for (auto &L : A->LHS)
        L = rewriteExpr(L);
      A->RHS = rewriteExpr(A->RHS);
      break;
    }
    case NodeKind::IfStmt: {
      auto *I = static_cast<IfStmt *>(S);
      I->Cond = rewriteExpr(I->Cond);
      rewriteBlock(I->Then);
      for (auto &EI : I->Elseifs) {
        EI.Cond = rewriteExpr(EI.Cond);
        rewriteBlock(EI.Body);
      }
      rewriteBlock(I->Else);
      break;
    }
    case NodeKind::ForStmt: {
      auto *F = static_cast<ForStmt *>(S);
      F->Iter = rewriteExpr(F->Iter);
      rewriteBlock(F->Body);
      break;
    }
    case NodeKind::WhileStmt: {
      auto *W = static_cast<WhileStmt *>(S);
      W->Cond = rewriteExpr(W->Cond);
      rewriteBlock(W->Body);
      break;
    }
    case NodeKind::SwitchStmt: {
      auto *Sw = static_cast<SwitchStmt *>(S);
      Sw->Discriminant = rewriteExpr(Sw->Discriminant);
      for (auto &Cs : Sw->Cases) {
        Cs.Value = rewriteExpr(Cs.Value);
        rewriteBlock(Cs.Body);
      }
      break;
    }
    case NodeKind::TryStmt: {
      auto *T = static_cast<TryStmt *>(S);
      rewriteBlock(T->TryBody);
      rewriteBlock(T->CatchBody);
      break;
    }
    case NodeKind::Block:
      rewriteBlock(static_cast<Block *>(S));
      break;
    default:
      break;
    }
  }

  void rewriteFunction(Function *F) {
    if (F)
      rewriteBlock(F->Body);
  }
};

} // namespace

int desynthDispatch(ASTContext &Ctx, TranslationUnit &TU,
                    const std::set<std::string> &AllowClasses,
                    bool KeyOffPinnedClass) {
  if (AllowClasses.empty())
    return 0;
  Rewriter R{Ctx, AllowClasses, KeyOffPinnedClass};
  if (TU.ScriptNode && TU.ScriptNode->Body)
    R.rewriteBlock(TU.ScriptNode->Body);
  for (Function *F : TU.Functions)
    R.rewriteFunction(F);
  for (ClassDef *C : TU.Classes) {
    for (Function *M : C->Methods)
      R.rewriteFunction(M);
    for (Function *M : C->StaticMethods)
      R.rewriteFunction(M);
  }
  return R.Count;
}

} // namespace sema
} // namespace matlab
