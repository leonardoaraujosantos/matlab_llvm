#include "matlab/MIR/Lowering.h"

#include "matlab/MIR/Builder.h"
#include "matlab/MIR/MIR.h"

#include <cstdlib>
#include <string>

namespace matlab {
namespace mir {

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

Lowerer::Lowerer(MIRContext &MIRCtx, TypeContext &TC, DiagnosticEngine &Diag)
    : MIR(MIRCtx), TC(TC), Diag(Diag) {
  (void)this->Diag;
}

//===----------------------------------------------------------------------===//
// Op kind mapping
//===----------------------------------------------------------------------===//

OpKind Lowerer::binOpKind(BinOp O) {
  switch (O) {
  case BinOp::Add:          return OpKind::Add;
  case BinOp::Sub:          return OpKind::Sub;
  case BinOp::Mul:          return OpKind::MatMul;
  case BinOp::Div:          return OpKind::MatDiv;
  case BinOp::LeftDiv:      return OpKind::MatLDiv;
  case BinOp::Pow:          return OpKind::MatPow;
  case BinOp::ElemMul:      return OpKind::EMul;
  case BinOp::ElemDiv:      return OpKind::EDiv;
  case BinOp::ElemLeftDiv:  return OpKind::ELDiv;
  case BinOp::ElemPow:      return OpKind::EPow;
  case BinOp::Eq:           return OpKind::Eq;
  case BinOp::Ne:           return OpKind::Ne;
  case BinOp::Lt:           return OpKind::Lt;
  case BinOp::Le:           return OpKind::Le;
  case BinOp::Gt:           return OpKind::Gt;
  case BinOp::Ge:           return OpKind::Ge;
  case BinOp::And:          return OpKind::BitAnd;
  case BinOp::Or:           return OpKind::BitOr;
  case BinOp::ShortAnd:     return OpKind::ShortAnd;
  case BinOp::ShortOr:      return OpKind::ShortOr;
  }
  return OpKind::Add;
}

OpKind Lowerer::unOpKind(UnOp O) {
  switch (O) {
  case UnOp::Plus:  return OpKind::UPlus;
  case UnOp::Minus: return OpKind::Neg;
  case UnOp::Not:   return OpKind::Not;
  }
  return OpKind::Neg;
}

OpKind Lowerer::postfixKind(PostfixOp O) {
  switch (O) {
  case PostfixOp::CTranspose: return OpKind::CTranspose;
  case PostfixOp::Transpose:  return OpKind::Transpose;
  }
  return OpKind::Transpose;
}

//===----------------------------------------------------------------------===//
// Slot handling
//===----------------------------------------------------------------------===//

Value *Lowerer::getOrCreateSlot(Binding *B, const Type *Ty, Builder &Bld) {
  auto It = SlotMap.find(B);
  if (It != SlotMap.end()) return It->second;
  // Allocate the slot in the entry block of the current function.
  Block *Saved = Bld.getInsertionBlock();
  Block *Entry = Saved;
  if (Entry) {
    Region *R = Entry->Parent;
    while (R) {
      Op *P = R->Parent;
      if (!P) break;
      if (P->K == OpKind::FuncOp) { Entry = R->Blocks.front(); break; }
      if (P->ParentBlock) R = P->ParentBlock->Parent;
      else break;
    }
  }
  // Emit Alloc at entry-block start. We approximate "start" by appending; since
  // we allocate slots only at first encounter, order is still deterministic.
  Bld.setInsertionPointToEnd(Entry);
  const Type *SlotTy = Ty ? Ty : TC.any();
  Value *Slot = Bld.alloc(SlotTy, std::string(B->Name));
  SlotMap[B] = Slot;
  Bld.setInsertionPointToEnd(Saved);
  return Slot;
}

Value *Lowerer::loadBinding(Binding *B, const Type *ValTy, Builder &Bld,
                            SourceRange Loc) {
  if (!B) return Bld.constInt(0, TC.any(), Loc);
  if (B->Kind == BindingKind::Function || B->Kind == BindingKind::Builtin) {
    // Treat bare function name use in value-context as a handle.
    return Bld.makeHandle(std::string(B->Name), TC.funcHandle(), Loc);
  }
  Value *Slot = getOrCreateSlot(B, ValTy, Bld);
  return Bld.load(Slot, ValTy ? ValTy : TC.any(), Loc);
}

//===----------------------------------------------------------------------===//
// Top-level lowering
//===----------------------------------------------------------------------===//

Module Lowerer::lower(const ::matlab::TranslationUnit &TU) {
  Module M = createModule(MIR);
  Builder B(MIR, TC);
  B.setInsertionPointToEnd(M.EntryBlock);

  if (TU.ScriptNode) {
    lowerScript(*TU.ScriptNode, M.ModuleOp);
  }
  for (const Function *F : TU.Functions)
    if (F) lowerFunction(*F, M.ModuleOp, B);

  return M;
}

void Lowerer::lowerScript(const ::matlab::Script &S, Op *ModuleBody) {
  Builder B(MIR, TC);
  B.setInsertionPointToEnd(ModuleBody->Regions[0]->Blocks.front());
  // Scripts become @script() -> () — no inputs/outputs for now.
  Op *Fn = B.funcOp("script", {}, {}, S.Range);
  Block *Entry = Fn->Regions[0]->Blocks.front();
  B.setInsertionPointToEnd(Entry);
  SlotMap.clear();
  if (S.Body) lowerBlock(*S.Body, B);
  B.retOp({}, {});
}

void Lowerer::lowerFunction(const ::matlab::Function &F, Op *ModuleBody, Builder & /*Outer*/) {
  Builder B(MIR, TC);
  B.setInsertionPointToEnd(ModuleBody->Regions[0]->Blocks.front());

  // Collect input/output types from Sema's bindings.
  std::vector<const Type *> InTys, OutTys;
  for (Binding *P : F.ParamRefs)
    InTys.push_back(P && P->InferredType ? P->InferredType : TC.any());
  for (Binding *O : F.OutputRefs)
    OutTys.push_back(O && O->InferredType ? O->InferredType : TC.any());

  Op *Fn = B.funcOp(std::string(F.Name), InTys, OutTys, F.Range);
  Block *Entry = Fn->Regions[0]->Blocks.front();
  B.setInsertionPointToEnd(Entry);

  SlotMap.clear();

  // Materialize a slot for each parameter and spill the block-arg into it.
  for (size_t i = 0; i < F.ParamRefs.size(); ++i) {
    Binding *Bind = F.ParamRefs[i];
    if (!Bind) continue;
    Value *Slot = B.alloc(InTys[i], std::string(Bind->Name));
    B.store(Entry->Arguments[i], Slot);
    SlotMap[Bind] = Slot;
  }
  // Allocate slots for outputs (initial value is undef — MATLAB assigns them
  // during the body).
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bind = F.OutputRefs[i];
    if (!Bind) continue;
    Value *Slot = B.alloc(OutTys[i], std::string(Bind->Name));
    SlotMap[Bind] = Slot;
  }
  // Pre-allocate slots for all local variables so their allocs land at the
  // function prologue rather than mid-body (avoids def-after-use in textual
  // output and mirrors the MLIR "allocas-at-entry" convention).
  if (F.FnScope) {
    std::vector<std::pair<std::string, Binding *>> Locals;
    for (auto &[K, Bind] : F.FnScope->locals())
      if (Bind && Bind->Kind == BindingKind::Var) Locals.emplace_back(K, Bind);
    std::sort(Locals.begin(), Locals.end(),
              [](const auto &A, const auto &B) { return A.first < B.first; });
    for (auto &[N, Bind] : Locals) {
      if (SlotMap.count(Bind)) continue;
      const Type *T = Bind->InferredType ? Bind->InferredType : TC.any();
      Value *Slot = B.alloc(T, N);
      SlotMap[Bind] = Slot;
    }
  }

  if (F.Body) lowerBlock(*F.Body, B);

  // Implicit return: load each output slot and emit a matlab.return.
  std::vector<Value *> Rets;
  for (size_t i = 0; i < F.OutputRefs.size(); ++i) {
    Binding *Bind = F.OutputRefs[i];
    if (!Bind) continue;
    Value *Slot = SlotMap[Bind];
    Rets.push_back(B.load(Slot, OutTys[i]));
  }
  B.retOp(std::move(Rets));
  // Nested functions: lower as siblings in the module scope for now.
  for (const Function *N : F.Nested) if (N) lowerFunction(*N, ModuleBody, B);
}

//===----------------------------------------------------------------------===//
// Statement lowering
//===----------------------------------------------------------------------===//

void Lowerer::lowerBlock(const ::matlab::Block &B, Builder &Bld) {
  for (const Stmt *S : B.Stmts) if (S) lowerStmt(*S, Bld);
}

void Lowerer::lowerStmt(const ::matlab::Stmt &St, Builder &Bld) {
  switch (St.Kind) {
  case NodeKind::ExprStmt: {
    auto &E = static_cast<const ExprStmt &>(St);
    if (E.E) lowerExpr(*E.E, Bld);
    return;
  }
  case NodeKind::AssignStmt: {
    auto &A = static_cast<const AssignStmt &>(St);
    Value *Rhs = A.RHS ? lowerExpr(*A.RHS, Bld) : nullptr;
    for (const Expr *L : A.LHS) if (L) lowerLValueStore(*L, Rhs, Bld);
    return;
  }
  case NodeKind::IfStmt: {
    auto &I = static_cast<const IfStmt &>(St);
    Value *Cond = I.Cond ? lowerExpr(*I.Cond, Bld) : Bld.constLogical(true);
    Op *If = Bld.ifOp(Cond, I.Range);

    Block *ThenEntry = If->Regions[0]->Blocks.front();
    Block *ElseEntry = If->Regions[1]->Blocks.front();

    Block *Save = Bld.getInsertionBlock();
    Bld.setInsertionPointToEnd(ThenEntry);
    if (I.Then) lowerBlock(*I.Then, Bld);
    Bld.yield();

    // Chain elseifs as nested if/else within the else region.
    Block *ElseCursor = ElseEntry;
    for (size_t idx = 0; idx < I.Elseifs.size(); ++idx) {
      Bld.setInsertionPointToEnd(ElseCursor);
      Value *Cond2 = I.Elseifs[idx].Cond
                         ? lowerExpr(*I.Elseifs[idx].Cond, Bld)
                         : Bld.constLogical(true);
      Op *Inner = Bld.ifOp(Cond2);
      Block *ThenB = Inner->Regions[0]->Blocks.front();
      Block *ElseB = Inner->Regions[1]->Blocks.front();
      Bld.setInsertionPointToEnd(ThenB);
      if (I.Elseifs[idx].Body) lowerBlock(*I.Elseifs[idx].Body, Bld);
      Bld.yield();
      ElseCursor = ElseB;
    }
    Bld.setInsertionPointToEnd(ElseCursor);
    if (I.Else) lowerBlock(*I.Else, Bld);
    Bld.yield();
    // Close each outer else-if chain with a yield.
    // (Already done inside the loop.)

    Bld.setInsertionPointToEnd(Save);
    return;
  }
  case NodeKind::ForStmt: {
    auto &F = static_cast<const ForStmt &>(St);
    Value *Iter = F.Iter ? lowerExpr(*F.Iter, Bld) : Bld.constInt(0, TC.any());
    Op *Loop = Bld.forOp(std::string(F.Var), Iter, F.Range);

    // Body region takes one block argument: the loop variable.
    Block *Body = Loop->Regions[0]->Blocks.front();
    // Derive element type from iter, fall back to Any.
    const Type *ElemTy = TC.any();
    if (Iter && Iter->Ty && Iter->Ty->K == Type::Kind::Array) {
      auto &AT = static_cast<const ArrayType &>(*Iter->Ty);
      ElemTy = TC.scalar(AT.Elt);
    }
    Value *LoopVal = MIR.addBlockArgument(Body, ElemTy);

    // Find the binding for F.Var within the current function scope via the
    // SlotMap by name match. If it has a slot, store into it; otherwise create.
    Binding *VarBind = nullptr;
    for (auto &[B, _] : SlotMap)
      if (B->Name == F.Var) { VarBind = B; break; }

    Block *Save = Bld.getInsertionBlock();
    Bld.setInsertionPointToEnd(Body);
    if (VarBind) {
      Value *Slot = getOrCreateSlot(VarBind, ElemTy, Bld);
      Bld.store(LoopVal, Slot);
    }
    if (F.Body) lowerBlock(*F.Body, Bld);
    Bld.yield();
    Bld.setInsertionPointToEnd(Save);
    return;
  }
  case NodeKind::WhileStmt: {
    auto &W = static_cast<const WhileStmt &>(St);
    Op *Loop = Bld.whileOp(W.Range);
    Block *Save = Bld.getInsertionBlock();
    // Cond region: evaluate the condition, yield it as the single value.
    Block *Cond = Loop->Regions[0]->Blocks.front();
    Bld.setInsertionPointToEnd(Cond);
    Value *C = W.Cond ? lowerExpr(*W.Cond, Bld) : Bld.constLogical(true);
    Bld.yield({C});
    // Body region.
    Block *Body = Loop->Regions[1]->Blocks.front();
    Bld.setInsertionPointToEnd(Body);
    if (W.Body) lowerBlock(*W.Body, Bld);
    Bld.yield();
    Bld.setInsertionPointToEnd(Save);
    return;
  }
  case NodeKind::ReturnStmt:   Bld.retOp(); return;
  case NodeKind::BreakStmt:    Bld.breakOp(); return;
  case NodeKind::ContinueStmt: Bld.continueOp(); return;
  case NodeKind::GlobalDecl:
  case NodeKind::PersistentDecl:
  case NodeKind::ImportStmt:
    return; // no MIR emission yet
  case NodeKind::CommandStmt: {
    auto &C = static_cast<const CommandStmt &>(St);
    std::vector<Value *> Args;
    for (auto &A : C.Args)
      Args.push_back(Bld.constString(A, TC.stringScalar()));
    Bld.callBuiltin(std::string(C.Name), std::move(Args), TC.any());
    return;
  }
  case NodeKind::SwitchStmt: {
    // Lower switch as a chain of scf.if ops.
    auto &Sw = static_cast<const SwitchStmt &>(St);
    Value *Disc = Sw.Discriminant ? lowerExpr(*Sw.Discriminant, Bld)
                                  : Bld.constInt(0, TC.any());
    Block *Cursor = Bld.getInsertionBlock();
    bool HasOtherwise = false;
    Op *FirstIf = nullptr;
    Block *ElseCursor = nullptr;
    for (auto &C : Sw.Cases) {
      if (!C.Value) { HasOtherwise = true; continue; }
      Value *V = lowerExpr(*C.Value, Bld);
      Value *Cond = Bld.binary(OpKind::Eq, Disc, V,
                               TC.scalar(Dtype::Logical));
      Op *If = Bld.ifOp(Cond);
      if (!FirstIf) FirstIf = If;
      Block *Then = If->Regions[0]->Blocks.front();
      Block *Else = If->Regions[1]->Blocks.front();
      Bld.setInsertionPointToEnd(Then);
      if (C.Body) lowerBlock(*C.Body, Bld);
      Bld.yield();
      Bld.setInsertionPointToEnd(Else);
      ElseCursor = Else;
    }
    if (HasOtherwise) {
      for (auto &C : Sw.Cases) {
        if (C.Value) continue;
        if (C.Body) lowerBlock(*C.Body, Bld);
        break;
      }
    }
    if (ElseCursor) { Bld.yield(); Bld.setInsertionPointToEnd(Cursor); }
    (void)FirstIf;
    return;
  }
  case NodeKind::TryStmt: {
    auto &T = static_cast<const TryStmt &>(St);
    // No real exception model yet — just lower the try body.
    if (T.TryBody) lowerBlock(*T.TryBody, Bld);
    (void)T.CatchBody;
    return;
  }
  default:
    return;
  }
}

//===----------------------------------------------------------------------===//
// L-value store
//===----------------------------------------------------------------------===//

Value *Lowerer::lowerLValueStore(const ::matlab::Expr &LHS, Value *Rhs, Builder &Bld) {
  switch (LHS.Kind) {
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(LHS);
    if (!N.Ref) return Rhs;
    const Type *Ty = LHS.Ty ? LHS.Ty : TC.any();
    Value *Slot = getOrCreateSlot(N.Ref, Ty, Bld);
    if (Rhs) Bld.store(Rhs, Slot);
    return Rhs;
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(LHS);
    // a(i, j, ...) = rhs — model as a "subscript-store" via a call_builtin
    // placeholder for now. A proper op can be added later.
    std::vector<Value *> Os;
    if (C.Callee) Os.push_back(lowerExpr(*C.Callee, Bld));
    for (const Expr *A : C.Args) if (A) Os.push_back(lowerExpr(*A, Bld));
    if (Rhs) Os.push_back(Rhs);
    Bld.callBuiltin("__subscript_store", std::move(Os), TC.any());
    return Rhs;
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<const CellIndex &>(LHS);
    std::vector<Value *> Os;
    if (C.Callee) Os.push_back(lowerExpr(*C.Callee, Bld));
    for (const Expr *A : C.Args) if (A) Os.push_back(lowerExpr(*A, Bld));
    if (Rhs) Os.push_back(Rhs);
    Bld.callBuiltin("__cell_store", std::move(Os), TC.any());
    return Rhs;
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<const FieldAccess &>(LHS);
    Value *Base = F.Base ? lowerExpr(*F.Base, Bld) : nullptr;
    std::vector<Value *> Os;
    if (Base) Os.push_back(Base);
    if (Rhs) Os.push_back(Rhs);
    Op *O = Bld.create(OpKind::FieldAccess, std::move(Os), {TC.any()}, F.Range);
    O->setAttr("field", Attribute::ofSym(std::string(F.Field)));
    O->setAttr("store", Attribute::ofBool(true));
    return Rhs;
  }
  default:
    return Rhs;
  }
}

//===----------------------------------------------------------------------===//
// Expression lowering
//===----------------------------------------------------------------------===//

static int64_t foldInt(const Expr *E) {
  if (!E) return 0;
  if (auto *L = dynamic_cast<const IntegerLiteral *>(E)) {
    try { return std::stoll(std::string(L->Text)); }
    catch (...) { return 0; }
  }
  if (auto *U = dynamic_cast<const UnaryOpExpr *>(E)) {
    if (U->Op == UnOp::Minus) return -foldInt(U->Operand);
    if (U->Op == UnOp::Plus)  return foldInt(U->Operand);
  }
  return 0;
}

static double foldFloat(const Expr *E) {
  if (!E) return 0.0;
  if (auto *L = dynamic_cast<const FPLiteral *>(E)) {
    try { return std::stod(std::string(L->Text)); }
    catch (...) { return 0.0; }
  }
  if (dynamic_cast<const IntegerLiteral *>(E)) {
    return (double)foldInt(E);
  }
  return 0.0;
}

Value *Lowerer::lowerExpr(const ::matlab::Expr &E, Builder &Bld) {
  const Type *Ty = E.Ty ? E.Ty : TC.any();
  switch (E.Kind) {
  case NodeKind::IntegerLiteral:
    return Bld.constInt(foldInt(&E), Ty, E.Range);
  case NodeKind::FPLiteral:
    return Bld.constFloat(foldFloat(&E), Ty, E.Range);
  case NodeKind::ImagLiteral: {
    auto &L = static_cast<const ImagLiteral &>(E);
    Op *O = Bld.create(OpKind::ConstComplex, {}, {Ty}, L.Range);
    O->setAttr("value", Attribute::ofStr(std::string(L.Text)));
    return O->result();
  }
  case NodeKind::StringLiteral: {
    auto &L = static_cast<const StringLiteral &>(E);
    return Bld.constString(L.Value, Ty, L.Range);
  }
  case NodeKind::CharLiteral: {
    auto &L = static_cast<const CharLiteral &>(E);
    return Bld.constChar(L.Value, Ty, L.Range);
  }
  case NodeKind::NameExpr: {
    auto &N = static_cast<const NameExpr &>(E);
    return loadBinding(N.Ref, Ty, Bld, N.Range);
  }
  case NodeKind::EndExpr:   return Bld.constEnd(E.Range);
  case NodeKind::ColonExpr: return Bld.constColon(E.Range);
  case NodeKind::BinaryOp: {
    auto &B = static_cast<const BinaryOpExpr &>(E);
    Value *L = B.LHS ? lowerExpr(*B.LHS, Bld) : nullptr;
    Value *R = B.RHS ? lowerExpr(*B.RHS, Bld) : nullptr;
    return Bld.binary(binOpKind(B.Op), L, R, Ty, B.Range);
  }
  case NodeKind::UnaryOp: {
    auto &U = static_cast<const UnaryOpExpr &>(E);
    Value *A = U.Operand ? lowerExpr(*U.Operand, Bld) : nullptr;
    return Bld.unary(unOpKind(U.Op), A, Ty, U.Range);
  }
  case NodeKind::PostfixOp: {
    auto &P = static_cast<const PostfixOpExpr &>(E);
    Value *A = P.Operand ? lowerExpr(*P.Operand, Bld) : nullptr;
    return Bld.unary(postfixKind(P.Op), A, Ty, P.Range);
  }
  case NodeKind::RangeExpr: {
    auto &R = static_cast<const RangeExpr &>(E);
    Value *S = R.Start ? lowerExpr(*R.Start, Bld) : nullptr;
    Value *Step = R.Step ? lowerExpr(*R.Step, Bld) : nullptr;
    Value *End = R.End ? lowerExpr(*R.End, Bld) : nullptr;
    return Bld.range(S, Step, End, Ty, R.Range);
  }
  case NodeKind::CallOrIndex: {
    auto &C = static_cast<const CallOrIndex &>(E);
    if (C.Resolved == CallKind::Call) {
      auto *N = dynamic_cast<const NameExpr *>(C.Callee);
      if (N && N->Ref) {
        std::vector<Value *> Args;
        for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A, Bld));
        if (N->Ref->Kind == BindingKind::Builtin)
          return Bld.callBuiltin(std::string(N->Name), std::move(Args), Ty, C.Range);
        return Bld.call(std::string(N->Name), std::move(Args), Ty, C.Range);
      }
      // Indirect call (computed callee, handle-valued).
      Value *CV = C.Callee ? lowerExpr(*C.Callee, Bld) : nullptr;
      std::vector<Value *> Args;
      for (const Expr *A : C.Args) if (A) Args.push_back(lowerExpr(*A, Bld));
      return Bld.callIndirect(CV, std::move(Args), Ty, C.Range);
    }
    // Index.
    Value *Arr = C.Callee ? lowerExpr(*C.Callee, Bld) : nullptr;
    std::vector<Value *> Idx;
    for (const Expr *A : C.Args) if (A) Idx.push_back(lowerExpr(*A, Bld));
    return Bld.subscript(Arr, std::move(Idx), Ty, C.Range);
  }
  case NodeKind::CellIndex: {
    auto &C = static_cast<const CellIndex &>(E);
    Value *Arr = C.Callee ? lowerExpr(*C.Callee, Bld) : nullptr;
    std::vector<Value *> Idx;
    for (const Expr *A : C.Args) if (A) Idx.push_back(lowerExpr(*A, Bld));
    return Bld.cellSubscript(Arr, std::move(Idx), Ty, C.Range);
  }
  case NodeKind::FieldAccess: {
    auto &F = static_cast<const FieldAccess &>(E);
    Value *B = F.Base ? lowerExpr(*F.Base, Bld) : nullptr;
    return Bld.fieldAccess(B, std::string(F.Field), Ty, F.Range);
  }
  case NodeKind::DynamicField: {
    auto &F = static_cast<const DynamicField &>(E);
    Value *B = F.Base ? lowerExpr(*F.Base, Bld) : nullptr;
    Value *N = F.Name ? lowerExpr(*F.Name, Bld) : nullptr;
    return Bld.dynamicField(B, N, Ty, F.Range);
  }
  case NodeKind::MatrixLiteral: {
    auto &M = static_cast<const MatrixLiteral &>(E);
    std::vector<Value *> Rows;
    for (auto &R : M.Rows) {
      std::vector<Value *> Cs;
      for (const Expr *C : R) if (C) Cs.push_back(lowerExpr(*C, Bld));
      Value *RowV = Bld.concatRow(std::move(Cs), TC.any());
      Rows.push_back(RowV);
    }
    if (Rows.size() == 1) return Rows.front();
    return Bld.concatCol(std::move(Rows), Ty, M.Range);
  }
  case NodeKind::CellLiteral: {
    auto &M = static_cast<const CellLiteral &>(E);
    std::vector<Value *> Rows;
    for (auto &R : M.Rows) {
      std::vector<Value *> Cs;
      for (const Expr *C : R) if (C) Cs.push_back(lowerExpr(*C, Bld));
      Op *O = Bld.create(OpKind::MakeCell, std::move(Cs), {TC.any()});
      O->setAttr("row", Attribute::ofBool(true));
      Rows.push_back(O->result());
    }
    if (Rows.size() == 1) return Rows.front();
    Op *O = Bld.create(OpKind::MakeCell, std::move(Rows), {Ty}, M.Range);
    O->setAttr("row", Attribute::ofBool(false));
    return O->result();
  }
  case NodeKind::AnonFunction: {
    auto &A = static_cast<const AnonFunction &>(E);
    std::vector<std::string> P;
    for (auto S : A.Params) P.emplace_back(S);
    Op *O = Bld.makeAnon(std::move(P), Ty, A.Range);
    // Populate the body region by evaluating the body inside it.
    Block *Save = Bld.getInsertionBlock();
    Block *Body = O->Regions[0]->Blocks.front();
    Bld.setInsertionPointToEnd(Body);
    Value *V = A.Body ? lowerExpr(*A.Body, Bld) : nullptr;
    Bld.yield(V ? std::vector<Value *>{V} : std::vector<Value *>{});
    Bld.setInsertionPointToEnd(Save);
    return O->result();
  }
  case NodeKind::FuncHandle: {
    auto &F = static_cast<const FuncHandle &>(E);
    return Bld.makeHandle(std::string(F.Name), Ty, F.Range);
  }
  default:
    return Bld.constInt(0, TC.any());
  }
}

} // namespace mir
} // namespace matlab
