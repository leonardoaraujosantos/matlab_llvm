// Phase 4.5.2 — replace verifier-placeholder
// `unrealized_conversion_cast` ops on scf.if conditions with the
// real `arith.cmpi ne, src, 0` (integer source) or `arith.cmpf one,
// src, 0.0` (float source) once type-flow has refined the source.
//
// The MIR-to-MLIR Lowering pass cannot always emit a real cmpi/cmpf
// at the scf.if site because the cond may be a `none`-typed load
// (function param / slot whose type only lands after a few rounds
// of scalar-to-arith + user-call refinement). Lowering inserts the
// cast as a verifier placeholder; this pass cleans it up.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

namespace matlab {
namespace mlirgen {

bool runRefineIfConds(mlir::ModuleOp M) {
  bool Ok = true;
  llvm::SmallVector<mlir::UnrealizedConversionCastOp, 8> Worklist;
  M.walk([&](mlir::UnrealizedConversionCastOp Op) {
    // Only consider casts that produce i1 — those are the scf.if
    // placeholders. Any other unrealized cast is somebody else's
    // and we leave it for ReconcileUnrealizedCasts later.
    if (Op.getNumResults() != 1 || Op.getNumOperands() != 1) return;
    auto IT = mlir::dyn_cast<mlir::IntegerType>(Op.getResult(0).getType());
    if (!IT || IT.getWidth() != 1) return;
    Worklist.push_back(Op);
  });
  for (auto Op : Worklist) {
    mlir::Value Src = Op.getOperand(0);
    mlir::Type ST = Src.getType();
    mlir::OpBuilder B(Op);
    mlir::Value Replaced;
    if (auto IST = mlir::dyn_cast<mlir::IntegerType>(ST)) {
      if (IST.getWidth() == 1) {
        // Source is already i1 — pass through.
        Replaced = Src;
      } else {
        mlir::Value Zero = mlir::arith::ConstantOp::create(
            B, Op.getLoc(), IST, mlir::IntegerAttr::get(IST, 0));
        Replaced = mlir::arith::CmpIOp::create(
            B, Op.getLoc(), mlir::arith::CmpIPredicate::ne, Src, Zero);
      }
    } else if (auto FT = mlir::dyn_cast<mlir::FloatType>(ST)) {
      mlir::Value Zero = mlir::arith::ConstantOp::create(
          B, Op.getLoc(), FT, mlir::FloatAttr::get(FT, 0.0));
      Replaced = mlir::arith::CmpFOp::create(
          B, Op.getLoc(), mlir::arith::CmpFPredicate::ONE, Src, Zero);
    } else {
      mlir::emitError(Op.getLoc())
          << "if-condition has unrefined type '" << ST
          << "'; cannot lower to i1 (the source slot or function "
             "argument's type didn't propagate)";
      Ok = false;
      continue;
    }
    Op.getResult(0).replaceAllUsesWith(Replaced);
    Op.erase();
  }
  return Ok;
}

} // namespace mlirgen
} // namespace matlab
