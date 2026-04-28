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
  // 0. Patch func.func arg types from call-site operand types.
  //    Late refinements (RefineSlotTypes / LowerScalarSlots) can
  //    retype call-site operands AFTER the LowerUserCalls fixpoint
  //    has settled. Without this catch-up, a func.func still
  //    declares `none` for an arg whose call sites pass i1 — the
  //    verifier rejects the resulting func.call.
  M.walk([&](mlir::func::CallOp Call) {
    auto Fn = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
    if (!Fn || Fn.empty()) return;
    auto FT = Fn.getFunctionType();
    if (Call.getNumOperands() != FT.getNumInputs()) return;
    llvm::SmallVector<mlir::Type, 4> NewIn(FT.getInputs().begin(),
                                            FT.getInputs().end());
    bool Changed = false;
    for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
      auto CallTy = Call.getOperand(i).getType();
      if (mlir::isa<mlir::NoneType>(NewIn[i]) && NewIn[i] != CallTy) {
        NewIn[i] = CallTy;
        Changed = true;
      }
    }
    if (!Changed) return;
    auto NewFT = mlir::FunctionType::get(Fn.getContext(),
                                          NewIn, FT.getResults());
    Fn.setFunctionType(NewFT);
    auto &Entry = Fn.getBody().front();
    for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
      if (Entry.getArgument(i).getType() != NewIn[i])
        Entry.getArgument(i).setType(NewIn[i]);
    }
  });

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

  // 2. Refresh `func.call` sites whose result types or operand
  //    types now disagree with the callee's signature. The func.call
  //    op stores its result types verbatim; if the callee was
  //    retyped (either by step 0 above for inputs or step 1 for
  //    outputs) the existing call is stale and must be re-emitted.
  llvm::SmallVector<mlir::func::CallOp, 8> Stale;
  M.walk([&](mlir::func::CallOp Call) {
    auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
    if (!Tgt) return;
    auto SigR = Tgt.getFunctionType().getResults();
    auto SigIn = Tgt.getFunctionType().getInputs();
    if (Call.getNumResults() != SigR.size()) return;
    for (unsigned i = 0; i < SigR.size(); ++i) {
      if (Call.getResult(i).getType() != SigR[i]) {
        Stale.push_back(Call);
        return;
      }
    }
    if (Call.getNumOperands() == SigIn.size()) {
      for (unsigned i = 0; i < SigIn.size(); ++i) {
        if (Call.getOperand(i).getType() != SigIn[i]) {
          Stale.push_back(Call);
          return;
        }
      }
    }
  });
  for (auto Call : Stale) {
    auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
    if (!Tgt) continue;
    auto SigR = Tgt.getFunctionType().getResults();
    auto SigIn = Tgt.getFunctionType().getInputs();
    // Only re-emit if every operand type now matches the (possibly
    // freshly-refined) signature. If they don't match, the call
    // can't be patched here and the verifier will surface the
    // root-cause mismatch.
    if (Call.getNumOperands() != SigIn.size()) continue;
    bool OpMatch = true;
    for (unsigned i = 0; i < SigIn.size(); ++i)
      if (Call.getOperand(i).getType() != SigIn[i]) {
        OpMatch = false; break;
      }
    if (!OpMatch) continue;
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
