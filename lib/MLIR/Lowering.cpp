#include "matlab/MLIR/Lowering.h"

#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/MLIR/Context.h"
#include "matlab/MLIR/TypeMapper.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_os_ostream.h"

#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace matlab {
namespace mlirgen {

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

int64_t foldInt(const Expr *E) {
  if (!E) return 0;
  if (auto *L = dynamic_cast<const IntegerLiteral *>(E)) {
    try { return std::stoll(std::string(L->Text)); } catch (...) { return 0; }
  }
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(E)) {
    if (U->Op == UnOp::Minus) return -foldInt(U->Operand);
    if (U->Op == UnOp::Plus)  return  foldInt(U->Operand);
  }
  return 0;
}

double foldFloat(const Expr *E) {
  if (!E) return 0.0;
  if (auto *L = dynamic_cast<const FPLiteral *>(E)) {
    try { return std::stod(std::string(L->Text)); } catch (...) { return 0.0; }
  }
  if (dynamic_cast<const IntegerLiteral *>(E)) return (double)foldInt(E);
  return 0.0;
}

/* Walk an anon function body collecting NameExpr bindings that refer to
 * values defined OUTSIDE the anon — i.e. captures. Params are filtered
 * out (they resolve against the block args), and so are builtins and
 * user functions (which don't need a capture slot; their call lowering
 * routes through the @name path or a direct call).
 *
 * Out is populated with bindings in first-seen order; Seen deduplicates
 * across multiple references to the same capture. Unknown expression
 * kinds simply aren't recursed into — a capture hiding inside an
 * unrecognised expr will still be lowered as a fresh lazy slot at the
 * read site, which loses the value but doesn't crash. */
void collectCaptures(const Expr *E,
                     const std::vector<Binding *> &Params,
                     std::vector<Binding *> &Out,
                     std::unordered_set<Binding *> &Seen) {
  if (!E) return;
  switch (E->Kind) {
  case NodeKind::NameExpr: {
    auto *N = static_cast<const NameExpr *>(E);
    if (!N->Ref) return;
    for (Binding *P : Params) if (P == N->Ref) return;
    if (N->Ref->Kind == BindingKind::Builtin ||
        N->Ref->Kind == BindingKind::Function) return;
    if (!Seen.insert(N->Ref).second) return;
    Out.push_back(N->Ref);
    return;
  }
  case NodeKind::BinaryOp: {
    auto *B = static_cast<const BinaryOpExpr *>(E);
    collectCaptures(B->LHS, Params, Out, Seen);
    collectCaptures(B->RHS, Params, Out, Seen);
    return;
  }
  case NodeKind::UnaryOp: {
    auto *U = static_cast<const UnaryOpExpr *>(E);
    collectCaptures(U->Operand, Params, Out, Seen);
    return;
  }
  case NodeKind::PostfixOp: {
    auto *P = static_cast<const PostfixOpExpr *>(E);
    collectCaptures(P->Operand, Params, Out, Seen);
    return;
  }
  case NodeKind::RangeExpr: {
    auto *R = static_cast<const RangeExpr *>(E);
    collectCaptures(R->Start, Params, Out, Seen);
    collectCaptures(R->Step,  Params, Out, Seen);
    collectCaptures(R->End,   Params, Out, Seen);
    return;
  }
  case NodeKind::CallOrIndex: {
    auto *C = static_cast<const CallOrIndex *>(E);
    collectCaptures(C->Callee, Params, Out, Seen);
    for (const Expr *A : C->Args) collectCaptures(A, Params, Out, Seen);
    return;
  }
  case NodeKind::MatrixLiteral: {
    auto *M = static_cast<const MatrixLiteral *>(E);
    for (auto &Row : M->Rows)
      for (const Expr *A : Row) collectCaptures(A, Params, Out, Seen);
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// Lowerer
//===----------------------------------------------------------------------===//

class Lowerer {
public:
  Lowerer(mlir::MLIRContext &MCtx, TypeContext &TC, DiagnosticEngine &Diag,
          const SourceManager *SM = nullptr)
      : MCtx(MCtx), TC(TC), Diag(Diag), SM(SM), B(&MCtx) {
    (void)this->Diag;
    (void)this->SM;
  }

  mlir::ModuleOp lower(const TranslationUnit &TU);

private:
  mlir::MLIRContext &MCtx;
  TypeContext &TC;
  DiagnosticEngine &Diag;
  const SourceManager *SM;
  mlir::OpBuilder B;

  // Per-function: binding -> slot (Value result of matlab.alloc).
  std::unordered_map<Binding *, mlir::Value> Slots;

  // Bindings known to hold function handles. Populated when we see an
  // assignment whose RHS is an AnonFunction / FuncHandle. Used at
  // CallOrIndex lowering time to emit matlab.call_indirect instead of
  // matlab.subscript for `f(x)` where f is a handle variable.
  //
  // The vector is the list of capture SPILL SLOTS (in the outer function)
  // that must be loaded and prepended to each call_indirect's argument
  // list — so that @(x) x + k calls still see the value k had at @ time.
  // Empty vector = no captures (plain @name handles or capture-free anons).
  std::unordered_map<Binding *, std::vector<mlir::Value>> HandleBindings;

  // Side map populated inside the AnonFunction lowering so the enclosing
  // AssignStmt can link the resulting capture slot list to the LHS binding.
  // Keyed by the AnonFunction AST node; cleared after use.
  std::unordered_map<const AnonFunction *,
                     std::vector<mlir::Value>> PendingCaptures;

  // Stack of (base, dim) contexts for `end` resolution inside subscripts.
  // Each entry represents the subscript arg currently being lowered:
  //   base = the matrix being indexed (already-lowered SSA value)
  //   dim  = 1-based position of this arg in the subscript.
  // When an EndExpr is lowered, the top of the stack provides operands for
  // the emitted matlab.end op so the tensor-ops pass can rewrite it to a
  // runtime matlab_end_of_dim call.
  std::vector<std::pair<mlir::Value, int64_t>> SubscriptCtx;

  //--- location / type helpers
  mlir::Location loc(SourceLocation L) const;
  mlir::Location loc(SourceRange R) const { return loc(R.Begin); }
  mlir::Type mirTy(const Type *T) const;

  //--- emission helpers
  mlir::Value emitUnreg(llvm::StringRef OpName,
                        llvm::ArrayRef<mlir::Value> Operands,
                        mlir::Type ResultType, mlir::Location Loc,
                        llvm::ArrayRef<mlir::NamedAttribute> Attrs = {});

  mlir::Operation *emitUnregOp(llvm::StringRef OpName,
                               llvm::ArrayRef<mlir::Value> Operands,
                               llvm::ArrayRef<mlir::Type> ResultTypes,
                               mlir::Location Loc,
                               llvm::ArrayRef<mlir::NamedAttribute> Attrs = {},
                               unsigned NumRegions = 0);

  mlir::Value emitAlloc(const Type *T, llvm::StringRef Name, mlir::Location Loc);
  mlir::Value emitLoad(mlir::Value Slot, mlir::Type Ty, mlir::Location Loc);
  void        emitStore(mlir::Value V, mlir::Value Slot, mlir::Location Loc);

  //--- top-level
  void lowerScript(const Script &S, mlir::ModuleOp M);
  void lowerFunction(const Function &F, mlir::ModuleOp M);

  //--- blocks / stmts / exprs
  void lowerBlock(const ::matlab::Block &B);
  void lowerStmt(const Stmt &St);
  mlir::Value lowerExpr(const Expr &E);
  void lowerLValueStore(const Expr &LHS, mlir::Value Rhs);

  //--- op-kind translation
  llvm::StringRef binOpName(BinOp O);
  llvm::StringRef unOpName(UnOp O);
  llvm::StringRef postfixName(PostfixOp O);

  mlir::Value loadBinding(Binding *Bnd, const Type *ValTy, mlir::Location L);
  mlir::Value getOrCreateSlot(Binding *Bnd, const Type *T, llvm::StringRef N,
                              mlir::Location L);
};

//===----------------------------------------------------------------------===//
// Helpers impl
//===----------------------------------------------------------------------===//

mlir::Location Lowerer::loc(SourceLocation L) const {
  if (!SM || !L.isValid())
    return mlir::UnknownLoc::get(&MCtx);
  auto LC = SM->getLineColumn(L);
  mlir::StringAttr File = mlir::StringAttr::get(&MCtx, SM->getName(L.File));
  return mlir::FileLineColLoc::get(File, LC.Line, LC.Column);
}

mlir::Type Lowerer::mirTy(const Type *T) const {
  return mapType(MCtx, T);
}

mlir::Operation *Lowerer::emitUnregOp(llvm::StringRef OpName,
                                     llvm::ArrayRef<mlir::Value> Operands,
                                     llvm::ArrayRef<mlir::Type> ResultTypes,
                                     mlir::Location Loc,
                                     llvm::ArrayRef<mlir::NamedAttribute> Attrs,
                                     unsigned NumRegions) {
  mlir::OperationState State(Loc, OpName);
  State.addOperands(Operands);
  State.addTypes(ResultTypes);
  for (auto &A : Attrs) State.addAttribute(A.getName(), A.getValue());
  for (unsigned i = 0; i < NumRegions; ++i) State.addRegion();
  return B.create(State);
}

mlir::Value Lowerer::emitUnreg(llvm::StringRef OpName,
                               llvm::ArrayRef<mlir::Value> Operands,
                               mlir::Type ResultType, mlir::Location Loc,
                               llvm::ArrayRef<mlir::NamedAttribute> Attrs) {
  mlir::Operation *Op = emitUnregOp(OpName, Operands, {ResultType}, Loc, Attrs);
  return Op->getResult(0);
}

mlir::Value Lowerer::emitAlloc(const Type *T, llvm::StringRef Name,
                               mlir::Location Loc) {
  mlir::Type MT = mirTy(T);
  mlir::NamedAttribute NameAttr(
      mlir::StringAttr::get(&MCtx, "name"),
      mlir::StringAttr::get(&MCtx, Name));
  return emitUnreg("matlab.alloc", {}, MT, Loc, {NameAttr});
}

mlir::Value Lowerer::emitLoad(mlir::Value Slot, mlir::Type Ty,
                              mlir::Location Loc) {
  return emitUnreg("matlab.load", {Slot}, Ty, Loc);
}

void Lowerer::emitStore(mlir::Value V, mlir::Value Slot, mlir::Location Loc) {
  emitUnregOp("matlab.store", {V, Slot}, {}, Loc);
}

//===----------------------------------------------------------------------===//
// Op-name mapping
//===----------------------------------------------------------------------===//

llvm::StringRef Lowerer::binOpName(BinOp O) {
  switch (O) {
  case BinOp::Add:          return "matlab.add";
  case BinOp::Sub:          return "matlab.sub";
  case BinOp::Mul:          return "matlab.matmul";
  case BinOp::Div:          return "matlab.matdiv";
  case BinOp::LeftDiv:      return "matlab.matldiv";
  case BinOp::Pow:          return "matlab.matpow";
  case BinOp::ElemMul:      return "matlab.emul";
  case BinOp::ElemDiv:      return "matlab.ediv";
  case BinOp::ElemLeftDiv:  return "matlab.eldiv";
  case BinOp::ElemPow:      return "matlab.epow";
  case BinOp::Eq:           return "matlab.eq";
  case BinOp::Ne:           return "matlab.ne";
  case BinOp::Lt:           return "matlab.lt";
  case BinOp::Le:           return "matlab.le";
  case BinOp::Gt:           return "matlab.gt";
  case BinOp::Ge:           return "matlab.ge";
  case BinOp::And:          return "matlab.and";
  case BinOp::Or:           return "matlab.or";
  case BinOp::ShortAnd:     return "matlab.short_and";
  case BinOp::ShortOr:      return "matlab.short_or";
  }
  return "matlab.add";
}

llvm::StringRef Lowerer::unOpName(UnOp O) {
  switch (O) {
  case UnOp::Plus:  return "matlab.uplus";
  case UnOp::Minus: return "matlab.neg";
  case UnOp::Not:   return "matlab.not";
  }
  return "matlab.neg";
}

llvm::StringRef Lowerer::postfixName(PostfixOp O) {
  switch (O) {
  case PostfixOp::CTranspose: return "matlab.ctranspose";
  case PostfixOp::Transpose:  return "matlab.transpose";
  }
  return "matlab.transpose";
}

//===----------------------------------------------------------------------===//
// Slot handling
//===----------------------------------------------------------------------===//

mlir::Value Lowerer::getOrCreateSlot(Binding *Bnd, const Type *T,
                                     llvm::StringRef N, mlir::Location L) {
  auto It = Slots.find(Bnd);
  if (It != Slots.end()) return It->second;

  // Allocate at the start of the current func's entry block.
  auto *InsBlock = B.getInsertionBlock();
  mlir::Block *Entry = InsBlock;
  mlir::Operation *P = InsBlock ? InsBlock->getParentOp() : nullptr;
  while (P && !mlir::isa<mlir::func::FuncOp>(P)) {
    auto *PB = P->getBlock();
    P = PB ? PB->getParentOp() : nullptr;
  }
  if (P) {
    auto F = mlir::cast<mlir::func::FuncOp>(P);
    Entry = &F.getBody().front();
  }

  mlir::OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(Entry);
  mlir::Value Slot = emitAlloc(T, N, L);
  Slots[Bnd] = Slot;
  return Slot;
}

mlir::Value Lowerer::loadBinding(Binding *Bnd, const Type *ValTy,
                                 mlir::Location L) {
  if (!Bnd) return emitUnreg("matlab.undef", {}, mirTy(ValTy), L);
  if (Bnd->Kind == BindingKind::Function ||
      Bnd->Kind == BindingKind::Builtin) {
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, std::string(Bnd->Name)));
    return emitUnreg("matlab.make_handle", {}, mirTy(ValTy), L, {Cal});
  }
  auto It = Slots.find(Bnd);
  if (It == Slots.end()) {
    // Declared but never stored — materialize a slot lazily.
    mlir::Value S = getOrCreateSlot(Bnd, ValTy, Bnd->Name, L);
    return emitLoad(S, mirTy(ValTy), L);
  }
  // Prefer the slot's own type when it's more concrete than what Sema
  // inferred (Sema falls back to `any`/NoneType for values it can't
  // specialize; the slot may have been created with a concrete scalar
  // type, e.g. the f64 spill slot of an anon function's block arg).
  mlir::Type LoadTy = mirTy(ValTy);
  mlir::Type SlotTy = It->second.getType();
  if (mlir::isa<mlir::NoneType>(LoadTy) &&
      !mlir::isa<mlir::NoneType>(SlotTy))
    LoadTy = SlotTy;
  return emitLoad(It->second, LoadTy, L);
}

//===----------------------------------------------------------------------===//
// Top-level
//===----------------------------------------------------------------------===//

mlir::ModuleOp Lowerer::lower(const TranslationUnit &TU) {
  mlir::ModuleOp M = mlir::ModuleOp::create(mlir::UnknownLoc::get(&MCtx));
  B.setInsertionPointToEnd(M.getBody());

  if (TU.ScriptNode) lowerScript(*TU.ScriptNode, M);
  for (const Function *F : TU.Functions) if (F) lowerFunction(*F, M);

  return M;
}

void Lowerer::lowerScript(const Script &S, mlir::ModuleOp M) {
  mlir::OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToEnd(M.getBody());

  auto FnTy = mlir::FunctionType::get(&MCtx, {}, {});
  auto Fn = mlir::func::FuncOp::create(loc(S.Range), "script", FnTy);
  B.insert(Fn);

  auto *Entry = Fn.addEntryBlock();
  B.setInsertionPointToEnd(Entry);
  Slots.clear();

  if (S.Body) lowerBlock(*S.Body);

  mlir::func::ReturnOp::create(B, loc(S.Range));
}

void Lowerer::lowerFunction(const Function &F, mlir::ModuleOp M) {
  mlir::OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToEnd(M.getBody());

  // Build parameter / result type vectors from Sema-inferred types.
  llvm::SmallVector<mlir::Type, 4> InTys, OutTys;
  for (Binding *P : F.ParamRefs)
    InTys.push_back(mirTy(P && P->InferredType ? P->InferredType : TC.any()));
  for (Binding *O : F.OutputRefs)
    OutTys.push_back(mirTy(O && O->InferredType ? O->InferredType : TC.any()));

  auto FnTy = mlir::FunctionType::get(&MCtx, InTys, OutTys);
  auto Fn = mlir::func::FuncOp::create(loc(F.Range),
                                       std::string(F.Name), FnTy);
  B.insert(Fn);

  auto *Entry = Fn.addEntryBlock();
  B.setInsertionPointToEnd(Entry);

  Slots.clear();

  // Spill parameters into slots.
  for (size_t i = 0; i < F.ParamRefs.size(); ++i) {
    Binding *Bnd = F.ParamRefs[i];
    if (!Bnd) continue;
    const Type *T = Bnd->InferredType ? Bnd->InferredType : TC.any();
    mlir::Value Slot = emitAlloc(T, Bnd->Name, loc(F.Range));
    Slots[Bnd] = Slot;
    emitStore(Entry->getArgument(i), Slot, loc(F.Range));
  }
  // Pre-allocate output slots.
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bnd = F.OutputRefs[i];
    if (!Bnd) continue;
    const Type *T = Bnd->InferredType ? Bnd->InferredType : TC.any();
    mlir::Value Slot = emitAlloc(T, Bnd->Name, loc(F.Range));
    Slots[Bnd] = Slot;
  }
  // Pre-allocate local var slots so allocas stay at the function prologue.
  if (F.FnScope) {
    std::vector<std::pair<std::string, Binding *>> Locals;
    for (auto &[K, Bnd] : F.FnScope->locals())
      if (Bnd && Bnd->Kind == BindingKind::Var) Locals.emplace_back(K, Bnd);
    std::sort(Locals.begin(), Locals.end(),
              [](const auto &A, const auto &B) { return A.first < B.first; });
    for (auto &[N, Bnd] : Locals) {
      if (Slots.count(Bnd)) continue;
      const Type *T = Bnd->InferredType ? Bnd->InferredType : TC.any();
      mlir::Value Slot = emitAlloc(T, N, loc(F.Range));
      Slots[Bnd] = Slot;
    }
  }

  if (F.Body) lowerBlock(*F.Body);

  // Implicit return: load each output slot and return.
  llvm::SmallVector<mlir::Value, 4> Rets;
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bnd = F.OutputRefs[i];
    if (!Bnd) continue;
    mlir::Value Slot = Slots[Bnd];
    Rets.push_back(emitLoad(Slot, OutTys[i], loc(F.Range)));
  }
  mlir::func::ReturnOp::create(B, loc(F.Range), Rets);

  // Nested functions: emit at module level.
  for (const Function *N : F.Nested) if (N) lowerFunction(*N, M);
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

void Lowerer::lowerBlock(const ::matlab::Block &Blk) {
  for (const Stmt *S : Blk.Stmts) if (S) lowerStmt(*S);
}

void Lowerer::lowerStmt(const Stmt &St) {
  switch (St.Kind) {
  case NodeKind::ExprStmt: {
    auto &E = static_cast<const ExprStmt &>(St);
    if (E.E) lowerExpr(*E.E);
    return;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<const AssignStmt &>(St);
    /* If RHS is an anonymous function or a function handle, tag the LHS
     * binding so later reads through it call the handle rather than
     * trying to subscript a matrix. */
    bool RhsIsHandle = A.RHS && (A.RHS->Kind == NodeKind::AnonFunction ||
                                 A.RHS->Kind == NodeKind::FuncHandle);

    /* Multi-return call: [V, D] = eig(A). If the LHS arity is > 1 and
     * the RHS is a call to a builtin that has a multi-return variant,
     * emit a matlab.call_builtin with N result types and a nargout
     * attribute so LowerTensorOps can dispatch to the right runtime
     * entry. Each LHS then gets its own result. */
    if (A.LHS.size() > 1 && A.RHS &&
        A.RHS->Kind == NodeKind::CallOrIndex) {
      auto *C = static_cast<const CallOrIndex *>(A.RHS);
      auto *Callee = dynamic_cast<const NameExpr *>(C->Callee);
      if (Callee && Callee->Ref &&
          Callee->Ref->Kind == BindingKind::Builtin) {
        llvm::SmallVector<mlir::Value, 4> Args;
        for (const Expr *Arg : C->Args)
          if (Arg) Args.push_back(lowerExpr(*Arg));
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, std::string(Callee->Name)));
        mlir::NamedAttribute NO(
            mlir::StringAttr::get(&MCtx, "nargout"),
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(&MCtx, 64),
                (int64_t)A.LHS.size()));
        llvm::SmallVector<mlir::Type, 4> Rtys(
            A.LHS.size(), mlir::NoneType::get(&MCtx));
        mlir::Operation *Op = emitUnregOp("matlab.call_builtin", Args,
                                           Rtys, loc(A.Range), {Cal, NO});
        for (size_t i = 0;
             i < A.LHS.size() && i < (size_t)Op->getNumResults(); ++i) {
          if (A.LHS[i])
            lowerLValueStore(*A.LHS[i], Op->getResult(i));
        }
        return;
      }
    }

    mlir::Value Rhs = A.RHS ? lowerExpr(*A.RHS) : mlir::Value{};
    if (RhsIsHandle) {
      /* Pick up capture spill slots left by the AnonFunction lowering
       * (empty vector for @name and capture-free anons). */
      std::vector<mlir::Value> Caps;
      if (A.RHS->Kind == NodeKind::AnonFunction) {
        auto *AF = static_cast<const AnonFunction *>(A.RHS);
        auto It = PendingCaptures.find(AF);
        if (It != PendingCaptures.end()) {
          Caps = std::move(It->second);
          PendingCaptures.erase(It);
        }
      }
      for (const Expr *L : A.LHS) {
        if (auto *N = dynamic_cast<const NameExpr *>(L))
          if (N->Ref) HandleBindings[N->Ref] = Caps;
      }
    }
    for (const Expr *L : A.LHS) if (L) lowerLValueStore(*L, Rhs);
    /* Implicit display: MATLAB prints the result of a statement that
     * doesn't end in a semicolon. We handle the common case of a single
     * named LHS (x = expr). Skip when: the rhs is a handle (we've
     * spilled it; disp would try to matrix-print a function pointer),
     * the rhs's type is NoneType (void call result), or the LHS isn't
     * a single NameExpr.
     *
     * Formatted as two disp calls — "x =" then the value — so it lines
     * up with MATLAB's '%NAME =\n<value>' layout without needing a new
     * runtime entry. */
    if (!A.Suppressed && Rhs && !RhsIsHandle &&
        !mlir::isa<mlir::NoneType>(Rhs.getType()) &&
        A.LHS.size() == 1 && A.LHS[0] &&
        A.LHS[0]->Kind == NodeKind::NameExpr) {
      auto *N = static_cast<const NameExpr *>(A.LHS[0]);
      std::string Label = std::string(N->Name) + " =";
      mlir::NamedAttribute LV(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::StringAttr::get(&MCtx, Label));
      mlir::Value LabelV = emitUnreg("matlab.const_char", {},
                                      mlir::NoneType::get(&MCtx),
                                      loc(A.Range), {LV});
      mlir::NamedAttribute Cal(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "disp"));
      emitUnregOp("matlab.call_builtin", {LabelV},
                  {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
      emitUnregOp("matlab.call_builtin", {Rhs},
                  {mlir::NoneType::get(&MCtx)}, loc(A.Range), {Cal});
    }
    return;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<const IfStmt &>(St);
    mlir::OpBuilder::InsertionGuard G(B);
    mlir::Value Cond = I.Cond
        ? lowerExpr(*I.Cond)
        : emitUnreg("matlab.const_logical", {},
                    mlir::IntegerType::get(&MCtx, 1), loc(I.Range));

    // scf.if with withElseRegion=true auto-inserts scf.yield terminators; we
    // insert BEFORE those when emitting our body.
    auto IfOp = mlir::scf::IfOp::create(B, loc(I.Range),
                                        /*resultTypes=*/mlir::TypeRange{},
                                        Cond, /*withElseRegion=*/true);
    mlir::Block *ThenB = &IfOp.getThenRegion().front();
    mlir::Block *ElseB = &IfOp.getElseRegion().front();

    B.setInsertionPoint(ThenB->getTerminator());
    if (I.Then) lowerBlock(*I.Then);

    // Chain elseifs into nested scf.ifs in the else region.
    mlir::Block *ElseCursor = ElseB;
    for (auto &EI : I.Elseifs) {
      B.setInsertionPoint(ElseCursor->getTerminator());
      mlir::Value Cond2 = EI.Cond
          ? lowerExpr(*EI.Cond)
          : emitUnreg("matlab.const_logical", {},
                      mlir::IntegerType::get(&MCtx, 1), loc(I.Range));
      auto Inner = mlir::scf::IfOp::create(
          B, loc(I.Range), mlir::TypeRange{}, Cond2, /*withElseRegion=*/true);
      B.setInsertionPoint(Inner.getThenRegion().front().getTerminator());
      if (EI.Body) lowerBlock(*EI.Body);
      ElseCursor = &Inner.getElseRegion().front();
    }

    B.setInsertionPoint(ElseCursor->getTerminator());
    if (I.Else) lowerBlock(*I.Else);
    return;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<const ForStmt &>(St);
    mlir::Value Iter = F.Iter
        ? lowerExpr(*F.Iter)
        : emitUnreg("matlab.undef", {}, mirTy(TC.any()), loc(F.Range));

    // Loop-var element type: if iter is ranked, take its element type.
    mlir::Type ElemTy = mlir::NoneType::get(&MCtx);
    if (Iter) {
      auto IterTy = Iter.getType();
      if (auto RT = mlir::dyn_cast<mlir::RankedTensorType>(IterTy))
        ElemTy = RT.getElementType();
      else if (auto UT = mlir::dyn_cast<mlir::UnrankedTensorType>(IterTy))
        ElemTy = UT.getElementType();
    }

    // Build a matlab.for with one region, one block with an i-var argument.
    mlir::NamedAttribute VarAttr(
        mlir::StringAttr::get(&MCtx, "var"),
        mlir::StringAttr::get(&MCtx, std::string(F.Var)));
    // Save the outer insertion point *before* createBlock moves it.
    mlir::OpBuilder::InsertionGuard G(B);

    mlir::Operation *ForOp = emitUnregOp(
        F.IsParfor ? "matlab.parfor" : "matlab.for",
        {Iter}, {}, loc(F.Range), {VarAttr}, /*NumRegions=*/1);
    auto &Region = ForOp->getRegion(0);
    mlir::Block *Body = B.createBlock(&Region, Region.end(), {ElemTy}, {loc(F.Range)});

    /* Find Sema binding for F.Var. First-time references to the loop
     * variable happen INSIDE the body (via NameExpr) and would otherwise
     * allocate the slot lazily at read time — too late for our induction-
     * store. Resolve F.VarRef (populated by the Resolver) and
     * pre-allocate the slot before emitting the store. */
    Binding *VarBind = F.VarRef;
    if (!VarBind)
      for (auto &[Bnd, _] : Slots)
        if (Bnd->Name == F.Var) { VarBind = Bnd; break; }

    B.setInsertionPointToEnd(Body);
    if (VarBind) {
      mlir::Value Slot = getOrCreateSlot(VarBind, TC.scalar(Dtype::Double),
                                         VarBind->Name, loc(F.Range));
      B.setInsertionPointToEnd(Body);
      emitStore(Body->getArgument(0), Slot, loc(F.Range));
    }
    if (F.Body) lowerBlock(*F.Body);
    emitUnregOp("matlab.yield", {}, {}, loc(F.Range));
    return;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<const WhileStmt &>(St);
    mlir::OpBuilder::InsertionGuard G(B);

    // matlab.while has two regions: cond (yields i1) and body.
    mlir::Operation *Op = emitUnregOp("matlab.while", {}, {}, loc(W.Range), {},
                                      /*NumRegions=*/2);
    mlir::Block *Cond = B.createBlock(&Op->getRegion(0), Op->getRegion(0).end(),
                                      {}, {});
    mlir::Block *Body = B.createBlock(&Op->getRegion(1), Op->getRegion(1).end(),
                                      {}, {});

    B.setInsertionPointToEnd(Cond);
    mlir::Value C = W.Cond
        ? lowerExpr(*W.Cond)
        : emitUnreg("matlab.const_logical", {},
                    mlir::IntegerType::get(&MCtx, 1), loc(W.Range));
    emitUnregOp("matlab.yield", {C}, {}, loc(W.Range));

    B.setInsertionPointToEnd(Body);
    if (W.Body) lowerBlock(*W.Body);
    emitUnregOp("matlab.yield", {}, {}, loc(W.Range));
    return;
  }
  case NodeKind::SwitchStmt: {
    auto &Sw = static_cast<const SwitchStmt &>(St);
    mlir::Value Disc = Sw.Discriminant
        ? lowerExpr(*Sw.Discriminant)
        : emitUnreg("matlab.undef", {}, mirTy(TC.any()), loc(Sw.Range));
    bool HasOtherwise = false;
    for (auto &C : Sw.Cases) {
      if (!C.Value) { HasOtherwise = true; continue; }
      mlir::Value V = lowerExpr(*C.Value);
      mlir::Value Cond = emitUnreg("matlab.eq", {Disc, V},
                                   mlir::IntegerType::get(&MCtx, 1),
                                   loc(Sw.Range));
      mlir::OpBuilder::InsertionGuard G(B);
      auto IfOp = mlir::scf::IfOp::create(B, loc(Sw.Range), mlir::TypeRange{},
                                          Cond, /*withElseRegion=*/true);
      B.setInsertionPoint(IfOp.getThenRegion().front().getTerminator());
      if (C.Body) lowerBlock(*C.Body);
    }
    if (HasOtherwise) {
      for (auto &C : Sw.Cases) {
        if (C.Value) continue;
        if (C.Body) lowerBlock(*C.Body);
        break;
      }
    }
    (void)Disc;
    return;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<const TryStmt &>(St);
    if (T.TryBody) lowerBlock(*T.TryBody);
    return;
  }
  case NodeKind::ReturnStmt:
    mlir::func::ReturnOp::create(B, loc(St.Range));
    return;
  case NodeKind::BreakStmt:
    emitUnregOp("matlab.break", {}, {}, loc(St.Range));
    return;
  case NodeKind::ContinueStmt:
    emitUnregOp("matlab.continue", {}, {}, loc(St.Range));
    return;
  case NodeKind::GlobalDecl:
  case NodeKind::PersistentDecl:
  case NodeKind::ImportStmt:
    return;
  case NodeKind::CommandStmt: {
    auto &C = static_cast<const CommandStmt &>(St);
    /* `clear A B C` maps each named variable's slot to an empty matrix.
     * We resolve each arg to a binding by name inside the current scope
     * (walking SlotMap for an entry with that name). Unmatched args are
     * silently ignored, matching MATLAB's behavior when clearing an
     * undefined name.
     *
     * `clear` with no args clears all variables in the current scope. */
    if (C.Name == "clear") {
      mlir::NamedAttribute NameAttr(
          mlir::StringAttr::get(&MCtx, "callee"),
          mlir::StringAttr::get(&MCtx, "matlab_empty_mat"));
      auto emitClearSlot = [&](mlir::Value Slot) {
        /* Call matlab_empty_mat(), store its result into the slot. We
         * emit as matlab.call_builtin so the tensor-ops pass picks it
         * up and converts it to a real llvm.call in due course. */
        auto PtrT = mlir::NoneType::get(&MCtx);  /* will be retyped */
        mlir::Value Empty = emitUnreg("matlab.call_builtin", {},
                                       PtrT, loc(C.Range), {NameAttr});
        emitStore(Empty, Slot, loc(C.Range));
      };
      if (C.Args.empty()) {
        for (auto &P : Slots) emitClearSlot(P.second);
      } else {
        for (auto &A : C.Args) {
          for (auto &P : Slots) {
            if (P.first->Name == A) { emitClearSlot(P.second); break; }
          }
        }
      }
      return;
    }

    llvm::SmallVector<mlir::Value, 4> Args;
    for (auto &A : C.Args) {
      mlir::NamedAttribute VA(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::StringAttr::get(&MCtx, A));
      Args.push_back(emitUnreg("matlab.const_str", {},
                               mlir::NoneType::get(&MCtx),
                               loc(C.Range), {VA}));
    }
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, std::string(C.Name)));
    emitUnregOp("matlab.call_builtin", Args,
                {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// LValue store
//===----------------------------------------------------------------------===//

void Lowerer::lowerLValueStore(const Expr &LHS, mlir::Value Rhs) {
  switch (LHS.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(LHS);
    if (!N.Ref) return;
    const Type *T = LHS.Ty ? LHS.Ty : TC.any();
    mlir::Value Slot = getOrCreateSlot(N.Ref, T, N.Name, loc(N.Range));
    if (Rhs) emitStore(Rhs, Slot, loc(N.Range));
    return;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(LHS);
    llvm::SmallVector<mlir::Value, 4> Os;
    mlir::Value Base;
    if (C.Callee) {
      Base = lowerExpr(*C.Callee);
      Os.push_back(Base);
    }
    // Push subscript context so any `end` inside an index expression
    // resolves to size(Base, dim).
    for (size_t a = 0; a < C.Args.size(); ++a) {
      const Expr *Arg = C.Args[a];
      if (!Arg) continue;
      if (Base) SubscriptCtx.push_back({Base, (int64_t)(a + 1)});
      Os.push_back(lowerExpr(*Arg));
      if (Base) SubscriptCtx.pop_back();
    }
    if (Rhs) Os.push_back(Rhs);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, "__subscript_store"));
    emitUnregOp("matlab.call_builtin", Os,
                {mlir::NoneType::get(&MCtx)}, loc(C.Range), {Cal});
    return;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<const FieldAccess &>(LHS);
    mlir::Value Base = F.Base ? lowerExpr(*F.Base) : mlir::Value{};
    llvm::SmallVector<mlir::Value, 2> Os;
    if (Base) Os.push_back(Base);
    if (Rhs) Os.push_back(Rhs);
    mlir::NamedAttribute FN(
        mlir::StringAttr::get(&MCtx, "field"),
        mlir::StringAttr::get(&MCtx, std::string(F.Field)));
    mlir::NamedAttribute Store(
        mlir::StringAttr::get(&MCtx, "store"),
        mlir::BoolAttr::get(&MCtx, true));
    emitUnregOp("matlab.field", Os,
                {mlir::NoneType::get(&MCtx)}, loc(F.Range), {FN, Store});
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

mlir::Value Lowerer::lowerExpr(const Expr &E) {
  mlir::Location L = loc(E.Range);
  mlir::Type RT = mirTy(E.Ty ? E.Ty : TC.any());

  switch (E.Kind) {
  case NodeKind::IntegerLiteral: {
    int64_t V = foldInt(&E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::IntegerAttr::get(
                                mlir::IntegerType::get(&MCtx, 64), V));
    return emitUnreg("matlab.const_int", {}, RT, L, {A});
  }
  case NodeKind::FPLiteral: {
    double V = foldFloat(&E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::FloatAttr::get(
                                mlir::Float64Type::get(&MCtx), V));
    return emitUnreg("matlab.const_float", {}, RT, L, {A});
  }
  case NodeKind::ImagLiteral: {
    auto &I = static_cast<const ImagLiteral &>(E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::StringAttr::get(&MCtx, std::string(I.Text)));
    return emitUnreg("matlab.const_complex", {}, RT, L, {A});
  }
  case NodeKind::StringLiteral: {
    auto &S = static_cast<const StringLiteral &>(E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::StringAttr::get(&MCtx, S.Value));
    return emitUnreg("matlab.const_str", {}, RT, L, {A});
  }
  case NodeKind::CharLiteral: {
    auto &S = static_cast<const CharLiteral &>(E);
    mlir::NamedAttribute A(mlir::StringAttr::get(&MCtx, "value"),
                            mlir::StringAttr::get(&MCtx, S.Value));
    return emitUnreg("matlab.const_char", {}, RT, L, {A});
  }
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(E);
    return loadBinding(N.Ref, E.Ty ? E.Ty : TC.any(), L);
  }
  case NodeKind::EndExpr: {
    // If we're inside a subscript arg, emit matlab.end with (base, dim)
    // operands so LowerTensorOps can rewrite it to matlab_end_of_dim.
    // Otherwise fall back to the zero-operand form — it won't survive
    // later passes, but the parser already errors on end-outside-indexing
    // so this path is really only reachable for weird IR.
    if (!SubscriptCtx.empty()) {
      auto [Base, Dim] = SubscriptCtx.back();
      mlir::NamedAttribute VA(
          mlir::StringAttr::get(&MCtx, "value"),
          mlir::FloatAttr::get(mlir::Float64Type::get(&MCtx),
                                (double)Dim));
      mlir::Value DimV = emitUnreg("matlab.const_float", {},
                                   mlir::Float64Type::get(&MCtx), L, {VA});
      return emitUnreg("matlab.end", {Base, DimV}, RT, L);
    }
    return emitUnreg("matlab.end", {}, RT, L);
  }
  case NodeKind::ColonExpr:
    return emitUnreg("matlab.colon", {}, RT, L);
  case NodeKind::BinaryOp: {
    auto &Bi = static_cast<const BinaryOpExpr &>(E);
    mlir::Value LHS = Bi.LHS ? lowerExpr(*Bi.LHS) : mlir::Value{};
    mlir::Value RHS = Bi.RHS ? lowerExpr(*Bi.RHS) : mlir::Value{};
    return emitUnreg(binOpName(Bi.Op), {LHS, RHS}, RT, L);
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(E);
    mlir::Value A = U.Operand ? lowerExpr(*U.Operand) : mlir::Value{};
    return emitUnreg(unOpName(U.Op), {A}, RT, L);
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<const PostfixOpExpr &>(E);
    mlir::Value A = P.Operand ? lowerExpr(*P.Operand) : mlir::Value{};
    return emitUnreg(postfixName(P.Op), {A}, RT, L);
  }
  case NodeKind::RangeExpr: {
    auto &R = static_cast<const RangeExpr &>(E);
    llvm::SmallVector<mlir::Value, 3> Os;
    if (R.Start) Os.push_back(lowerExpr(*R.Start));
    if (R.Step)  Os.push_back(lowerExpr(*R.Step));
    if (R.End)   Os.push_back(lowerExpr(*R.End));
    mlir::NamedAttribute HS(
        mlir::StringAttr::get(&MCtx, "has_step"),
        mlir::BoolAttr::get(&MCtx, R.Step != nullptr));
    return emitUnreg("matlab.range", Os, RT, L, {HS});
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(E);
    if (C.Resolved == CallKind::Call) {
      auto *N = dynamic_cast<const NameExpr *>(C.Callee);
      llvm::SmallVector<mlir::Value, 4> Args;
      for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A));
      if (N && N->Ref) {
        mlir::NamedAttribute Cal(
            mlir::StringAttr::get(&MCtx, "callee"),
            mlir::StringAttr::get(&MCtx, std::string(N->Name)));
        return emitUnreg(N->Ref->Kind == BindingKind::Builtin
                              ? "matlab.call_builtin" : "matlab.call",
                          Args, RT, L, {Cal});
      }
      mlir::Value CV = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
      llvm::SmallVector<mlir::Value, 4> Os;
      Os.push_back(CV);
      for (auto V : Args) Os.push_back(V);
      return emitUnreg("matlab.call_indirect", Os, RT, L);
    }
    // Index
    // Detect the "call through a handle" case: if the callee is a NameExpr
    // whose binding was assigned from @(x)... / @name, emit a
    // matlab.call_indirect instead of a matlab.subscript.
    bool IsHandleCall = false;
    const std::vector<mlir::Value> *CapSlots = nullptr;
    if (auto *NE = dynamic_cast<const NameExpr *>(C.Callee))
      if (NE->Ref) {
        auto It = HandleBindings.find(NE->Ref);
        if (It != HandleBindings.end()) {
          IsHandleCall = true;
          CapSlots = &It->second;
        }
      }

    mlir::Value Arr = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
    llvm::SmallVector<mlir::Value, 4> Idx;
    Idx.push_back(Arr);
    /* For an anon call with captures, the outlined function's signature
     * is (captures..., explicit_args...). We load each capture spill
     * slot (captured-at-@-time value) and prepend them to the arg list
     * before the user-written arguments. The slot's own type gives the
     * load type — handles both scalar (f64) and matrix-pointer captures
     * (matlab.alloc of a tensor type lowers to a ptr-typed slot later). */
    if (IsHandleCall && CapSlots) {
      for (mlir::Value Spill : *CapSlots)
        Idx.push_back(emitLoad(Spill, Spill.getType(), L));
    }
    // Lower each arg with subscript context pushed so any EndExpr inside
    // resolves to size(Arr, thisDim). Context is per-arg so that sibling
    // args don't leak each other's dim.
    for (size_t a = 0; a < C.Args.size(); ++a) {
      const Expr *Arg = C.Args[a];
      if (!Arg) continue;
      if (!IsHandleCall)
        SubscriptCtx.push_back({Arr, (int64_t)(a + 1)});
      Idx.push_back(lowerExpr(*Arg));
      if (!IsHandleCall) SubscriptCtx.pop_back();
    }
    if (IsHandleCall)
      return emitUnreg("matlab.call_indirect", Idx, RT, L);
    mlir::NamedAttribute NA(
        mlir::StringAttr::get(&MCtx, "nindices"),
        mlir::IntegerAttr::get(mlir::IntegerType::get(&MCtx, 64),
                               (int64_t)C.Args.size()));
    return emitUnreg("matlab.subscript", Idx, RT, L, {NA});
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<const CellIndex &>(E);
    mlir::Value Arr = C.Callee ? lowerExpr(*C.Callee) : mlir::Value{};
    llvm::SmallVector<mlir::Value, 4> Idx;
    Idx.push_back(Arr);
    for (const Expr *A : C.Args) if (A) Idx.push_back(lowerExpr(*A));
    return emitUnreg("matlab.cell_subscript", Idx, RT, L);
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<const FieldAccess &>(E);
    mlir::Value Base = F.Base ? lowerExpr(*F.Base) : mlir::Value{};
    mlir::NamedAttribute FA(
        mlir::StringAttr::get(&MCtx, "field"),
        mlir::StringAttr::get(&MCtx, std::string(F.Field)));
    return emitUnreg("matlab.field", {Base}, RT, L, {FA});
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<const DynamicField &>(E);
    mlir::Value Base = F.Base ? lowerExpr(*F.Base) : mlir::Value{};
    mlir::Value Name = F.Name ? lowerExpr(*F.Name) : mlir::Value{};
    return emitUnreg("matlab.dyn_field", {Base, Name}, RT, L);
  }
  case NodeKind::MatrixLiteral: {
    auto &M = static_cast<const MatrixLiteral &>(E);
    bool SingleRow = M.Rows.size() == 1;
    llvm::SmallVector<mlir::Value, 4> Rows;
    for (auto &R : M.Rows) {
      llvm::SmallVector<mlir::Value, 4> Cs;
      for (const Expr *C : R) if (C) Cs.push_back(lowerExpr(*C));
      // For a single-row literal the concat_row *is* the matrix result, so
      // give it the sema-inferred type. Multi-row literals feed concat_col
      // and the row type stays opaque.
      mlir::Type RowTy = SingleRow ? RT : mlir::NoneType::get(&MCtx);
      mlir::Value Row = emitUnreg("matlab.concat_row", Cs, RowTy, L);
      Rows.push_back(Row);
    }
    if (SingleRow) return Rows.front();
    return emitUnreg("matlab.concat_col", Rows, RT, L);
  }
  case NodeKind::CellLiteral: {
    auto &M = static_cast<const CellLiteral &>(E);
    llvm::SmallVector<mlir::Value, 4> Rows;
    for (auto &R : M.Rows) {
      llvm::SmallVector<mlir::Value, 4> Cs;
      for (const Expr *C : R) if (C) Cs.push_back(lowerExpr(*C));
      mlir::Value Row = emitUnreg("matlab.make_cell", Cs,
                                  mlir::NoneType::get(&MCtx), L);
      Rows.push_back(Row);
    }
    if (Rows.size() == 1) return Rows.front();
    return emitUnreg("matlab.make_cell", Rows, RT, L);
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<const AnonFunction &>(E);
    std::string Joined;
    for (size_t i = 0; i < A.Params.size(); ++i) {
      if (i) Joined += ",";
      Joined += std::string(A.Params[i]);
    }
    mlir::NamedAttribute PA(
        mlir::StringAttr::get(&MCtx, "params"),
        mlir::StringAttr::get(&MCtx, Joined));

    /* Detect captures: free variables in the body that aren't params,
     * builtins, or user functions. These become additional leading
     * block args + matlab.make_anon operands so each call_indirect can
     * thread the captured values through to the outlined llvm.func. */
    std::vector<Binding *> Captures;
    std::unordered_set<Binding *> Seen;
    if (A.Body) collectCaptures(A.Body, A.ParamRefs, Captures, Seen);

    auto F64Ty = mlir::Float64Type::get(&MCtx);

    /* Materialize each capture at the @-site. Capture element type comes
     * from the binding's Sema-inferred type: scalar -> f64, tensor ->
     * ptr (via the slot's tensor type which LowerTensorOps later
     * retypes). The outer slot load, the spill slot, the make_anon
     * operand and the corresponding anon-region block argument all
     * share this capture type. */
    llvm::SmallVector<mlir::Value, 4> CaptureVals;
    llvm::SmallVector<mlir::Type, 4>  CaptureTys;
    std::vector<mlir::Value> CaptureSpills;
    for (Binding *Bnd : Captures) {
      const Type *BTy = Bnd->InferredType ? Bnd->InferredType
                                          : TC.scalar(Dtype::Double);
      /* Prefer the outer slot's concrete MLIR type: Sema-level
       * InferredType is often still `any` for script-scope matrix
       * assignments even though the slot was allocated with a real
       * tensor type by the matrix-literal store earlier. */
      mlir::Value OuterSlot = getOrCreateSlot(Bnd, BTy, Bnd->Name, L);
      mlir::Type MTy = OuterSlot.getType();
      if (mlir::isa<mlir::NoneType>(MTy)) MTy = F64Ty;
      mlir::Value Cur = emitLoad(OuterSlot, MTy, L);
      /* Spill slot mirrors the outer slot's type so call-site reloads
       * see the same shape. */
      mlir::Value SpillSlot;
      if (MTy == F64Ty) {
        SpillSlot = emitAlloc(TC.scalar(Dtype::Double), Bnd->Name, L);
      } else {
        /* Emit a raw matlab.alloc with the concrete MLIR type; other
         * paths can't synthesize the Sema Type* for arbitrary tensors. */
        mlir::NamedAttribute NA(
            mlir::StringAttr::get(&MCtx, "name"),
            mlir::FlatSymbolRefAttr::get(&MCtx, std::string(Bnd->Name)));
        SpillSlot = emitUnreg("matlab.alloc", {}, MTy, L, {NA});
      }
      emitStore(Cur, SpillSlot, L);
      CaptureVals.push_back(Cur);
      CaptureTys.push_back(MTy);
      CaptureSpills.push_back(SpillSlot);
    }
    PendingCaptures[&A] = CaptureSpills;

    mlir::OpBuilder::InsertionGuard G(B);
    /* Block args: [captures (typed per capture)..., params (f64)...]. */
    llvm::SmallVector<mlir::Type> ArgTys;
    ArgTys.append(CaptureTys.begin(), CaptureTys.end());
    for (size_t i = 0; i < A.Params.size(); ++i) ArgTys.push_back(F64Ty);
    llvm::SmallVector<mlir::Location> ArgLocs(Captures.size() +
                                              A.Params.size(), L);
    mlir::Operation *Op = emitUnregOp("matlab.make_anon", CaptureVals, {RT},
                                      L, {PA}, /*NumRegions=*/1);
    mlir::Block *Body = B.createBlock(&Op->getRegion(0),
                                      Op->getRegion(0).end(),
                                      ArgTys, ArgLocs);
    B.setInsertionPointToEnd(Body);

    /* Swap in a fresh Slots map for the body. Captures AND params both
     * get inner spill slots whose type is the block arg's type.
     * Captures reuse the block arg's type (tensor or f64); params are
     * f64 (scalar-only for v1). */
    auto Saved = Slots;
    Slots.clear();
    for (size_t i = 0; i < Captures.size(); ++i) {
      mlir::Type MTy = Body->getArgument(i).getType();
      mlir::Value Slot;
      if (MTy == F64Ty) {
        Slot = emitAlloc(TC.scalar(Dtype::Double), Captures[i]->Name, L);
      } else {
        mlir::NamedAttribute NA(
            mlir::StringAttr::get(&MCtx, "name"),
            mlir::FlatSymbolRefAttr::get(
                &MCtx, std::string(Captures[i]->Name)));
        Slot = emitUnreg("matlab.alloc", {}, MTy, L, {NA});
      }
      Slots[Captures[i]] = Slot;
      emitStore(Body->getArgument(i), Slot, L);
    }
    for (size_t i = 0; i < A.ParamRefs.size(); ++i) {
      Binding *Bnd = A.ParamRefs[i];
      if (!Bnd) continue;
      mlir::Value Slot = emitAlloc(TC.scalar(Dtype::Double), Bnd->Name, L);
      Slots[Bnd] = Slot;
      emitStore(Body->getArgument(Captures.size() + i), Slot, L);
    }

    mlir::Value V = A.Body ? lowerExpr(*A.Body) : mlir::Value{};
    llvm::SmallVector<mlir::Value, 1> Ys;
    if (V) Ys.push_back(V);
    emitUnregOp("matlab.yield", Ys, {}, L);

    Slots = std::move(Saved);
    return Op->getResult(0);
  }
  case NodeKind::FuncHandle: {
    auto &F = static_cast<const FuncHandle &>(E);
    mlir::NamedAttribute Cal(
        mlir::StringAttr::get(&MCtx, "callee"),
        mlir::StringAttr::get(&MCtx, std::string(F.Name)));
    return emitUnreg("matlab.make_handle", {}, RT, L, {Cal});
  }
  default:
    return emitUnreg("matlab.undef", {}, RT, L);
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

mlir::ModuleOp lowerToMLIR(Context &Ctx,
                           TypeContext &TC,
                           DiagnosticEngine &Diag,
                           const TranslationUnit &TU) {
  Lowerer L(Ctx.get(), TC, Diag);
  return L.lower(TU);
}

void printModule(std::ostream &OS, mlir::ModuleOp M) {
  std::string S;
  llvm::raw_string_ostream RS(S);
  M.print(RS);
  RS.flush();
  OS << S;
}

} // namespace mlirgen
} // namespace matlab
