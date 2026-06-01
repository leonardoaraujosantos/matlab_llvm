// Lowers sequential matlab.for / matlab.while into scf.while constructs
// so the MLIR conversion pipeline (scf → cf → llvm) can finish translation.
//
// Scope (v1):
//   - matlab.for whose iterator is produced by a matlab.range op (i.e. the
//     common `for i = a:b` / `for i = a:step:b` form). Extracted start,
//     step (default 1.0) and end become the scf.while driver; the loop
//     variable's f64 block argument is substituted by the scf.while's
//     induction value.
//   - matlab.while with cond + body regions that each end in matlab.yield.
//     The cond's yield operand flows into scf.condition; the body's yield
//     is dropped (scf.while body yields no carry values).
//
// Loops with iterators that aren't a matlab.range — e.g. `for c = M`
// iterating over columns of a matrix — are left intact. A later pass can
// add that case by lowering to a scf.for over 1..cols with a column-slice.
//
// Semantics notes:
//   - MATLAB's `for i = a:s:b` walks a<=b when s>0 and a>=b when s<0. We
//     select the appropriate comparison via arith.select on the sign of
//     the step so the runtime behavior matches for both directions.
//   - The induction variable's outer slot store (emitted by the frontend
//     as matlab.store of the block arg into the loop-var slot) is cloned
//     verbatim inside the body, now storing the scf.while's induction
//     value. This keeps the matlab.load path in the body wiring unchanged.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

/* Extract (start, step, end) from a matlab.range producer.  Returns
 * false if the op isn't a range or the step can't be synthesised.
 *
 * Accepts two shapes:
 *   (1) The MATLAB-dialect op `matlab.range(start, end)` or
 *       `matlab.range(start, step, end)` with `has_step` attr.  This
 *       is the frontend-emitted form before LowerTensorOps runs.
 *   (2) The post-LowerTensorOps form `llvm.call @matlab_range(start,
 *       step, end) : (f64, f64, f64) -> !llvm.ptr`.  Issue #36 moves
 *       LowerTensorOps earlier in the pipeline; the matlab.range op
 *       is consumed before LowerSeqLoops runs, so the iter operand of
 *       a surviving matlab.for is the lowered runtime call. */
bool extractRange(Value V, Value &Start, Value &Step, Value &End,
                  OpBuilder &B, Location Loc) {
  Operation *Def = V.getDefiningOp();
  if (!Def) return false;
  auto F64 = B.getF64Type();

  if (isMatlabOp(Def, "matlab.range")) {
    unsigned N = Def->getNumOperands();
    auto HasStepAttr = Def->getAttrOfType<BoolAttr>("has_step");
    bool HasStep = HasStepAttr && HasStepAttr.getValue();
    if (HasStep && N == 3) {
      Start = Def->getOperand(0);
      Step  = Def->getOperand(1);
      End   = Def->getOperand(2);
    } else if (!HasStep && N == 2) {
      Start = Def->getOperand(0);
      End   = Def->getOperand(1);
      Step  = arith::ConstantOp::create(B, Loc, B.getF64FloatAttr(1.0));
    } else {
      return false;
    }
  } else if (auto Call = dyn_cast<LLVM::CallOp>(Def)) {
    /* Post-LowerTensorOps form: matlab_range(start, step, end) -> ptr.
     * LowerTensorOps always emits the 3-arg variant (step=1.0 when
     * the source had no step) so we don't need a 2-arg case. */
    auto Callee = Call.getCallee();
    if (!Callee || *Callee != "matlab_range") return false;
    if (Call.getNumOperands() != 3) return false;
    Start = Call.getOperand(0);
    Step  = Call.getOperand(1);
    End   = Call.getOperand(2);
  } else {
    return false;
  }

  /* The bounds are usually f64. But a workspace scalar used as a loop bound
   * (`for i = 1:(nstate-1)`, nstate read back as a boxed matlab_mat*) can
   * arrive as a ptr or a non-f64 scalar. A range bound is scalar in MATLAB,
   * so coerce each to f64 — unboxing a ptr via matlab_mat_to_scalar — rather
   * than leaving the matlab.for unconverted. (#77) */
  auto PtrTy = LLVM::LLVMPointerType::get(B.getContext());
  auto coerce = [&](Value &Vv) -> bool {
    Type T = Vv.getType();
    if (T == F64) return true;
    if (T == PtrTy) {
      auto Mod = Def->getParentOfType<ModuleOp>();
      auto Fn = Mod.lookupSymbol<LLVM::LLVMFuncOp>("matlab_mat_to_scalar");
      if (!Fn) {
        OpBuilder::InsertionGuard G(B);
        B.setInsertionPointToStart(Mod.getBody());
        auto Ty = LLVM::LLVMFunctionType::get(F64, {PtrTy});
        Fn = LLVM::LLVMFuncOp::create(B, Loc, "matlab_mat_to_scalar", Ty);
        Fn.setLinkage(LLVM::Linkage::External);
      }
      Vv = LLVM::CallOp::create(B, Loc, Fn, ValueRange{Vv}).getResult();
      return true;
    }
    if (auto IT = dyn_cast<IntegerType>(T)) {
      Vv = (IT.getWidth() == 1)
               ? (Value)arith::UIToFPOp::create(B, Loc, F64, Vv)
               : (Value)arith::SIToFPOp::create(B, Loc, F64, Vv);
      return true;
    }
    if (isa<Float32Type>(T)) {
      Vv = arith::ExtFOp::create(B, Loc, F64, Vv);
      return true;
    }
    return false;
  };
  if (!coerce(Start) || !coerce(Step) || !coerce(End)) return false;
  return true;
}

bool lowerForOp(Operation *ForOp) {
  if (ForOp->getNumRegions() != 1) return false;
  if (ForOp->getNumOperands() < 1 || ForOp->getNumOperands() > 2)
    return false;
  Region &Body = ForOp->getRegion(0);
  if (!Body.hasOneBlock()) return false;
  Block &BB = Body.front();
  if (BB.getNumArguments() != 1) return false;

  OpBuilder B(ForOp);
  Location L = ForOp->getLoc();
  auto F64 = B.getF64Type();

  Value Iter = ForOp->getOperand(0);
  /* Optional second operand: did_break i1 slot. When present, the
   * scf.while cond also checks !did_break so a break inside the body
   * exits the loop immediately on the next cond check. */
  Value BreakSlot;
  if (ForOp->getNumOperands() == 2) BreakSlot = ForOp->getOperand(1);

  /* Matrix-iterate form: `for n = M` where M is a 1-D static-shape
   * tensor.  MATLAB iterates `n` over each column of M; for a row
   * vector each column is a scalar and the frontend types BB.arg(0)
   * as the element type.  Synthesise a 1-based scf.while of length N
   * and substitute BB.arg(0) with a 1-indexed matlab.subscript pull
   * from M on each iteration.  LowerTensorOps converts the subscript
   * to the matrix runtime entry on its later pass.
   *
   * Scope (v1, issue #23): 1-D ranked tensor with static shape and
   * scalar element type.  Covers the issue's `for n = [128 256 512]`
   * pattern and the GPU benchmark fixture.  2-D matrix-column
   * iteration (block arg becomes a column slice) is a follow-on. */
  Value Start, Step, End;
  bool MatrixIter = false;
  bool MatrixIterPtr = false;          /* iter is matlab_mat * (post-LT) */
  Type MatrixIterElemTy;
  auto PtrTy = LLVM::LLVMPointerType::get(B.getContext());
  if (!extractRange(Iter, Start, Step, End, B, L)) {
    /* Two matrix-iter shapes:
     *   (a) Pre-LowerTensorOps: `tensor<NxT>` with static shape.  N
     *       is known at compile time; emit matlab.subscript(M, IV)
     *       and let LowerTensorOps convert it on its later sweep.
     *   (b) Post-LowerTensorOps (issue #36 ordering): `!llvm.ptr`
     *       (matlab_mat *).  N is unknown statically — call
     *       matlab_size_dim(M, 2.0) at runtime for the bound, and
     *       per-iteration call matlab_subscript1_s(M, IV) → f64. */
    auto IterTy = Iter.getType();
    if (auto T = mlir::dyn_cast<RankedTensorType>(IterTy)) {
      if (!T.hasStaticShape() || T.getShape().size() != 1)
        return false;
      MatrixIterElemTy = T.getElementType();
      if (BB.getArgument(0).getType() != MatrixIterElemTy) return false;
      int64_t N = T.getShape()[0];
      if (N <= 0) return false;
      MatrixIter = true;
      Start = arith::ConstantOp::create(B, L, B.getF64FloatAttr(1.0));
      End   = arith::ConstantOp::create(
          B, L, B.getF64FloatAttr((double)N));
      Step  = arith::ConstantOp::create(B, L, B.getF64FloatAttr(1.0));
    } else if (IterTy == PtrTy) {
      /* Body arg must be f64 — for ptr we read scalar elements via
       * the runtime subscript helper which returns f64. */
      if (BB.getArgument(0).getType() != F64) return false;
      MatrixIterElemTy = F64;
      MatrixIter = true;
      MatrixIterPtr = true;
      /* End = matlab_size_dim(M, 2.0) — number of columns. */
      auto Mod = ForOp->getParentOfType<ModuleOp>();
      auto FnSym = Mod.lookupSymbol<LLVM::LLVMFuncOp>("matlab_size_dim");
      if (!FnSym) {
        OpBuilder::InsertionGuard G(B);
        B.setInsertionPointToStart(Mod.getBody());
        auto Ty = LLVM::LLVMFunctionType::get(F64, {PtrTy, F64});
        FnSym = LLVM::LLVMFuncOp::create(B, ForOp->getLoc(),
                                          "matlab_size_dim", Ty);
        FnSym.setLinkage(LLVM::Linkage::External);
      }
      Value Two = arith::ConstantOp::create(B, L, B.getF64FloatAttr(2.0));
      auto Call = LLVM::CallOp::create(B, L, FnSym, ValueRange{Iter, Two});
      End = Call.getResult();
      Start = arith::ConstantOp::create(B, L, B.getF64FloatAttr(1.0));
      Step  = arith::ConstantOp::create(B, L, B.getF64FloatAttr(1.0));
    } else {
      return false;
    }
  }
  /* Remember the matlab.range producer so we can erase it below if its
   * only user was this matlab.for. Leaving it in place would cause
   * LowerTensorOps to emit a dead matlab_range() runtime call. */
  Operation *RangeProducer = MatrixIter ? nullptr : Iter.getDefiningOp();

  /* scf.while carrying one f64 induction value (%iv). */
  auto W = scf::WhileOp::create(B, L, TypeRange{F64}, ValueRange{Start});

  /* ---- cond region ----------------------------------------------------- */
  {
    Block *Cond = B.createBlock(&W.getBefore(), W.getBefore().end(),
                                TypeRange{F64}, {L});
    OpBuilder::InsertionGuard G(B);
    B.setInsertionPointToEnd(Cond);
    Value IV = Cond->getArgument(0);
    Value Zero = arith::ConstantOp::create(B, L, B.getF64FloatAttr(0.0));
    Value PosStep = arith::CmpFOp::create(
        B, L, arith::CmpFPredicate::OGT, Step, Zero);
    Value LeCmp = arith::CmpFOp::create(
        B, L, arith::CmpFPredicate::OLE, IV, End);
    Value GeCmp = arith::CmpFOp::create(
        B, L, arith::CmpFPredicate::OGE, IV, End);
    Value C = arith::SelectOp::create(B, L, PosStep, LeCmp, GeCmp);
    if (BreakSlot) {
      auto I1 = B.getI1Type();
      OperationState St(L, "matlab.load");
      St.addOperands(BreakSlot);
      St.addTypes(I1);
      Operation *LoadOp = B.create(St);
      Value BV = LoadOp->getResult(0);
      Value True = arith::ConstantOp::create(B, L, I1,
          B.getIntegerAttr(I1, 1));
      Value NotBr = arith::XOrIOp::create(B, L, BV, True);
      C = arith::AndIOp::create(B, L, C, NotBr);
    }
    scf::ConditionOp::create(B, L, C, ValueRange{IV});
  }

  /* ---- body region ---------------------------------------------------- */
  {
    Block *NewBody = B.createBlock(&W.getAfter(), W.getAfter().end(),
                                   TypeRange{F64}, {L});
    OpBuilder::InsertionGuard G(B);
    B.setInsertionPointToEnd(NewBody);
    Value IV = NewBody->getArgument(0);

    /* Clone each op from the original matlab.for body, mapping its block
     * argument (the original induction value) to the new scf IV.
     * matlab.yield at the end is replaced by arith.addf (step) + scf.yield.
     *
     * Matrix-iter form: BB.arg(0) is the per-iteration element of M,
     * not the index — emit matlab.subscript(M, IV) (1-based) and bind
     * the result to BB.arg(0) instead of IV.  matlab.subscript stays in
     * MATLAB dialect; LowerTensorOps converts it to a runtime call on
     * a later pass once M's tensor type has been lowered to ptr. */
    IRMapping Map;
    if (MatrixIter && MatrixIterPtr) {
      /* Post-LowerTensorOps ptr arm: emit a runtime subscript call. */
      auto Mod = ForOp->getParentOfType<ModuleOp>();
      auto SubFn =
          Mod.lookupSymbol<LLVM::LLVMFuncOp>("matlab_subscript1_s");
      if (!SubFn) {
        OpBuilder::InsertionGuard G(B);
        B.setInsertionPointToStart(Mod.getBody());
        auto Ty = LLVM::LLVMFunctionType::get(F64, {PtrTy, F64});
        SubFn = LLVM::LLVMFuncOp::create(B, ForOp->getLoc(),
                                          "matlab_subscript1_s", Ty);
        SubFn.setLinkage(LLVM::Linkage::External);
      }
      auto Call = LLVM::CallOp::create(B, L, SubFn, ValueRange{Iter, IV});
      Map.map(BB.getArgument(0), Call.getResult());
    } else if (MatrixIter) {
      OperationState SS(L, "matlab.subscript");
      SS.addOperands(ValueRange{Iter, IV});
      SS.addTypes(TypeRange{MatrixIterElemTy});
      SS.addAttribute("nindices", B.getI64IntegerAttr(1));
      Operation *SubOp = B.create(SS);
      Map.map(BB.getArgument(0), SubOp->getResult(0));
    } else {
      Map.map(BB.getArgument(0), IV);
    }
    for (Operation &Op : BB) {
      if (isMatlabOp(&Op, "matlab.yield")) continue;
      B.clone(Op, Map);
    }
    Value Next = arith::AddFOp::create(B, L, IV, Step);
    scf::YieldOp::create(B, L, ValueRange{Next});
  }

  ForOp->erase();
  if (RangeProducer && RangeProducer->use_empty() &&
      isMatlabOp(RangeProducer, "matlab.range"))
    RangeProducer->erase();
  return true;
}

bool lowerWhileOp(Operation *WhileOp) {
  if (WhileOp->getNumRegions() != 2) return false;
  Region &CondR = WhileOp->getRegion(0);
  Region &BodyR = WhileOp->getRegion(1);
  if (!CondR.hasOneBlock() || !BodyR.hasOneBlock()) return false;
  Block &CondBB = CondR.front();
  Block &BodyBB = BodyR.front();

  OpBuilder B(WhileOp);
  Location L = WhileOp->getLoc();

  /* scf.while with no carried values — iter_args is empty. */
  auto W = scf::WhileOp::create(B, L, TypeRange{}, ValueRange{});

  /* ---- cond region: clone ops, replace matlab.yield's operand with
   *      scf.condition's operand. ------------------------------------- */
  {
    Block *Cond = B.createBlock(&W.getBefore(), W.getBefore().end(),
                                TypeRange{}, {});
    OpBuilder::InsertionGuard G(B);
    B.setInsertionPointToEnd(Cond);
    IRMapping Map;
    bool Ok = true;
    for (Operation &Op : CondBB) {
      if (isMatlabOp(&Op, "matlab.yield")) {
        if (Op.getNumOperands() != 1) { Ok = false; break; }
        Value C = Map.lookupOrDefault(Op.getOperand(0));
        /* scf.condition expects i1 — coerce if the yielded value is an
         * integer or float logical. For our lowerer, conditions are i1. */
        if (!mlir::isa<IntegerType>(C.getType()) ||
            mlir::cast<IntegerType>(C.getType()).getWidth() != 1) {
          auto F64 = B.getF64Type();
          if (C.getType() == F64) {
            Value Zero = arith::ConstantOp::create(
                B, L, B.getF64FloatAttr(0.0));
            C = arith::CmpFOp::create(
                B, L, arith::CmpFPredicate::ONE, C, Zero);
          } else {
            /* Fallback for ptr-typed conds (matlab_mat *) that fixupIfCond
             * didn't intercept upstream — emit a runtime truth call so we
             * don't leave a malformed scf.while behind. */
            auto PtrTy = LLVM::LLVMPointerType::get(B.getContext());
            auto I8 = B.getIntegerType(8);
            if (C.getType() == PtrTy) {
              auto Mod = WhileOp->getParentOfType<ModuleOp>();
              auto Fn = Mod.lookupSymbol<LLVM::LLVMFuncOp>("matlab_mat_truth");
              if (!Fn) {
                OpBuilder::InsertionGuard G(B);
                B.setInsertionPointToStart(Mod.getBody());
                Fn = LLVM::LLVMFuncOp::create(
                    B, Mod.getLoc(), "matlab_mat_truth",
                    LLVM::LLVMFunctionType::get(I8, {PtrTy}));
              }
              auto Call = LLVM::CallOp::create(B, L, Fn, ValueRange{C});
              Value I8V = Call.getResult();
              Value Zero8 = arith::ConstantOp::create(
                  B, L, I8, B.getIntegerAttr(I8, 0));
              C = arith::CmpIOp::create(
                  B, L, arith::CmpIPredicate::ne, I8V, Zero8);
            } else {
              Ok = false; break;
            }
          }
        }
        scf::ConditionOp::create(B, L, C, ValueRange{});
        continue;
      }
      B.clone(Op, Map);
    }
    if (!Ok) {
      /* Bail-out cleanup: erase the partially-built scf.while so we
       * don't leave verifier-rejected IR. The original matlab.while
       * stays in place; downstream passes will surface the real error. */
      W.erase();
      return false;
    }
  }

  /* ---- body region: clone ops, replace matlab.yield with scf.yield. --- */
  {
    Block *Body = B.createBlock(&W.getAfter(), W.getAfter().end(),
                                TypeRange{}, {});
    OpBuilder::InsertionGuard G(B);
    B.setInsertionPointToEnd(Body);
    IRMapping Map;
    for (Operation &Op : BodyBB) {
      if (isMatlabOp(&Op, "matlab.yield")) continue;
      B.clone(Op, Map);
    }
    scf::YieldOp::create(B, L);
  }

  WhileOp->erase();
  return true;
}

} // namespace

bool runLowerSeqLoops(ModuleOp M) {
  SmallVector<Operation *> Fors, Whiles;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.for"))   Fors.push_back(Op);
    if (isMatlabOp(Op, "matlab.while")) Whiles.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Fors)   if (lowerForOp(Op))   Changed = true;
  for (Operation *Op : Whiles) if (lowerWhileOp(Op)) Changed = true;
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
