// Shared canonical-for-loop matcher used by HWLegalize and
// EmitSystemVerilog. The post-LowerSeqLoops shape of a MATLAB
// `for i = init:end` (or `init:step:end`) is:
//
//   scf.while %iv = %init {
//     %cmp = arith.cmpf <ole|oge>, %iv, %end
//     scf.condition (%cmp) %iv : f64
//   } do {
//   ^bb0(%iv: f64):
//     ...body...
//     %next = arith.addf %iv, %step
//     scf.yield %next : f64
//   }
//
// EmitC.cpp implements its own copy of this matcher; we mirror its
// invariants here so the two emitters agree on what counts as a
// for-loop. Phase 2 of the SV backend additionally requires Init /
// End / Step to be `arith.constant`s — that check is the caller's
// responsibility (it lets HWLegalize emit a precise diagnostic for
// data-dependent bounds).

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace matlab {
namespace mlirgen {

bool matchHWForLoop(mlir::Operation *Op, HWForLoopInfo &Info) {
  auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Op);
  if (!W) return false;
  if (W->getNumOperands() != 1 || W.getNumResults() != 1) return false;
  if (!W.getOperand(0).getType().isF64()) return false;

  // Before region: arith.cmpf %iv, %end + scf.condition %cmp %iv.
  if (!W.getBefore().hasOneBlock()) return false;
  mlir::Block &BB = W.getBefore().front();
  if (BB.getNumArguments() != 1) return false;
  mlir::Value BeforeIv = BB.getArgument(0);
  if (!BeforeIv.getType().isF64()) return false;

  auto Cond = mlir::dyn_cast<mlir::scf::ConditionOp>(BB.getTerminator());
  if (!Cond) return false;
  // Condition must be the cmpf result and the carried value must be
  // the iv itself (passthrough into the after region).
  if (Cond.getNumOperands() < 2) return false;
  if (Cond.getArgs().size() != 1) return false;
  if (Cond.getArgs()[0] != BeforeIv) return false;
  auto Cmp = Cond.getCondition().getDefiningOp<mlir::arith::CmpFOp>();
  if (!Cmp) return false;
  if (Cmp.getLhs() != BeforeIv) return false;
  bool Dec = false;
  switch (Cmp.getPredicate()) {
  case mlir::arith::CmpFPredicate::OLE:
  case mlir::arith::CmpFPredicate::ULE:
    Dec = false; break;
  case mlir::arith::CmpFPredicate::OGE:
  case mlir::arith::CmpFPredicate::UGE:
    Dec = true; break;
  default:
    return false;
  }
  Info.End = Cmp.getRhs();

  // After region: body... + arith.addf %iv, %step + scf.yield %next.
  if (!W.getAfter().hasOneBlock()) return false;
  mlir::Block &AB = W.getAfter().front();
  if (AB.getNumArguments() != 1) return false;
  mlir::Value Iv = AB.getArgument(0);
  if (!Iv.getType().isF64()) return false;
  Info.Iv = Iv;

  auto Yield = mlir::dyn_cast<mlir::scf::YieldOp>(AB.getTerminator());
  if (!Yield) return false;
  if (Yield.getNumOperands() != 1) return false;
  auto Add = Yield.getOperand(0).getDefiningOp<mlir::arith::AddFOp>();
  if (!Add) return false;
  if (Add.getLhs() != Iv) return false;
  Info.Step = Add.getRhs();

  Info.Init = W.getOperand(0);
  Info.IsDecreasing = Dec;
  return true;
}

} // namespace mlirgen
} // namespace matlab
