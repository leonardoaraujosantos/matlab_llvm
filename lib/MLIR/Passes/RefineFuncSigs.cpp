// Walk every `func.func` and patch its declared result types from
// the body's `func.return` operand types. Then refresh any
// `func.call` site whose result type now disagrees with the
// callee's signature.
//
// LowerUserCalls already does this internally as part of its
// signature-refinement loop, but only when its "Compatible" gate
// passes. There are pipeline points (the early verify check after
// the WantClean / LowerScalarsToArith batch) that need a
// signature-only refresh — e.g. when LowerScalarsToArith rewrote
// `matlab.make_handle{callee="false"}` to `arith.constant 0 : i1`,
// the func.return now produces i1 but the function declares
// `-> none`, and the verifier fails before the next user-call
// iteration runs.
//
// This pass is idempotent: applying it repeatedly is safe.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

bool runRefineFuncSigs(mlir::ModuleOp M) {
  // 1. Patch func.func result types from func.return operand types.
  M.walk([&](mlir::func::FuncOp Fn) {
    if (Fn.empty()) return;
    llvm::SmallVector<mlir::Type, 4> NewResults(
        Fn.getFunctionType().getResults().begin(),
        Fn.getFunctionType().getResults().end());
    bool Changed = false;
    Fn.walk([&](mlir::func::ReturnOp Ret) {
      if (Ret.getNumOperands() != NewResults.size()) return;
      for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
        auto Old = NewResults[i];
        auto New = Ret.getOperand(i).getType();
        if (mlir::isa<mlir::NoneType>(Old) && Old != New) {
          NewResults[i] = New;
          Changed = true;
        }
      }
    });
    if (Changed) {
      auto Ty = mlir::FunctionType::get(
          Fn.getContext(), Fn.getFunctionType().getInputs(), NewResults);
      Fn.setFunctionType(Ty);
    }
  });

  // 2. Refresh `func.call` sites whose result type now disagrees
  //    with the callee's signature.
  llvm::SmallVector<mlir::func::CallOp, 8> Stale;
  M.walk([&](mlir::func::CallOp Call) {
    auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
    if (!Tgt) return;
    auto SigR = Tgt.getFunctionType().getResults();
    if (Call.getNumResults() != SigR.size()) return;
    for (unsigned i = 0; i < SigR.size(); ++i) {
      if (Call.getResult(i).getType() != SigR[i]) {
        Stale.push_back(Call);
        return;
      }
    }
  });
  for (auto Call : Stale) {
    auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
    if (!Tgt) continue;
    auto SigR = Tgt.getFunctionType().getResults();
    mlir::OpBuilder B(Call);
    auto Nc = mlir::func::CallOp::create(
        B, Call.getLoc(), SigR, Call.getCallee(), Call.getOperands());
    for (unsigned i = 0; i < SigR.size(); ++i)
      Call.getResult(i).replaceAllUsesWith(Nc.getResult(i));
    Call.erase();
  }
  return true;
}

} // namespace mlirgen
} // namespace matlab
