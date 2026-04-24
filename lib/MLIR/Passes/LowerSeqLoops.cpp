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

/* Extract (start, step, end) from a matlab.range producer. Returns
 * false if the op isn't a range or the step can't be synthesised. */
bool extractRange(Value V, Value &Start, Value &Step, Value &End,
                  OpBuilder &B, Location Loc) {
  Operation *Def = V.getDefiningOp();
  if (!isMatlabOp(Def, "matlab.range")) return false;
  auto F64 = B.getF64Type();
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
  /* All three values must already be f64 (they are, because the frontend
   * emits matlab.range with f64 operands for numeric ranges). */
  if (Start.getType() != F64 || Step.getType() != F64 ||
      End.getType() != F64) return false;
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

  Value Start, Step, End;
  if (!extractRange(Iter, Start, Step, End, B, L)) return false;
  /* Remember the matlab.range producer so we can erase it below if its
   * only user was this matlab.for. Leaving it in place would cause
   * LowerTensorOps to emit a dead matlab_range() runtime call. */
  Operation *RangeProducer = Iter.getDefiningOp();

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
     * matlab.yield at the end is replaced by arith.addf (step) + scf.yield. */
    IRMapping Map;
    Map.map(BB.getArgument(0), IV);
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
    for (Operation &Op : CondBB) {
      if (isMatlabOp(&Op, "matlab.yield")) {
        if (Op.getNumOperands() != 1) return false;
        Value C = Map.lookupOrDefault(Op.getOperand(0));
        /* scf.condition expects i1 — coerce if the yielded value is an
         * integer or float logical. For our lowerer, conditions are i1. */
        if (!mlir::isa<IntegerType>(C.getType()) ||
            mlir::cast<IntegerType>(C.getType()).getWidth() != 1) {
          /* If it's f64 (rare — matlab.const_logical should be i1), cast
           * through arith.cmpf != 0. */
          auto F64 = B.getF64Type();
          if (C.getType() == F64) {
            Value Zero = arith::ConstantOp::create(
                B, L, B.getF64FloatAttr(0.0));
            C = arith::CmpFOp::create(
                B, L, arith::CmpFPredicate::ONE, C, Zero);
          } else {
            return false;
          }
        }
        scf::ConditionOp::create(B, L, C, ValueRange{});
        continue;
      }
      B.clone(Op, Map);
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
