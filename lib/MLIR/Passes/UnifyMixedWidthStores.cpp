// Mixed-width store unifier — Tier-5f.
//
// The SV pipeline's fi-saturate path produces i64 intermediates for
// fi-multiplies on persistent fetches, while pragma-typed function
// args + fi-typed literals stay at the declared width (i32 for
// Q16.16).  When both feed stores to the SAME `matlab.alloc` slot
// (e.g. a `clamp` slot written by an `if upper / elseif lower /
// else passthrough` saturation), the slot ends up with mixed-width
// stores:
//
//   "matlab.store"(%rail_i32, %slot) : (i32, none) -> ()
//   "matlab.store"(%pass_i64, %slot) : (i64, none) -> ()
//
// `runHWLegalize` then rejects the function with "result has
// unsynthesizable type" because the slot's load returns `none`
// (the alloc slot couldn't be retyped to either i32 or i64
// consistently).
//
// This pass walks every `matlab.alloc` and finds the widest
// integer store width across all its stores; narrower stores get
// `arith.extsi` widened, all `matlab.load`s on the slot get their
// result type retyped to the unified width, and downstream
// `func.return` ops + the enclosing `func.func` signature get
// updated.  After this pass the SV pipeline sees a consistent
// width per slot and synth-checks pass.
//
// Slots whose stores are NOT all integers (e.g. mixed int + float)
// are left alone — those failures are real type bugs that should
// surface at HWLegalize anyway.
//
// Runs between `runLowerScalarSlots` and `runHWLegalize` in the
// SV emit lane (see `tools/matlabc/main.cpp`).

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isMatlabOp(mlir::Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

// Find all matlab.store ops targeting Slot. Returns the stores and
// (out param) the union of integer widths seen across their values.
// Returns false if any non-integer store is found (mixed int+float
// is a real bug elsewhere; leave the slot alone).
bool collectStores(mlir::Value Slot,
                   llvm::SmallVectorImpl<mlir::Operation *> &Stores,
                   llvm::SmallVectorImpl<unsigned> &Widths) {
  for (mlir::OpOperand &Use : Slot.getUses()) {
    mlir::Operation *U = Use.getOwner();
    if (!isMatlabOp(U, "matlab.store")) continue;
    if (U->getOperand(1) != Slot) continue;
    auto IT = mlir::dyn_cast<mlir::IntegerType>(U->getOperand(0).getType());
    if (!IT) return false;
    Stores.push_back(U);
    Widths.push_back(IT.getWidth());
  }
  return true;
}

void collectLoads(mlir::Value Slot,
                  llvm::SmallVectorImpl<mlir::Operation *> &Loads) {
  for (mlir::OpOperand &Use : Slot.getUses()) {
    mlir::Operation *U = Use.getOwner();
    if (isMatlabOp(U, "matlab.load")) Loads.push_back(U);
  }
}

// Update the enclosing func.func's result types after a load got
// retyped. If any `func.return` operand was the (old) load and we
// retyped it, the function's result type at the same index gets
// updated to match. Idempotent.
void refreshEnclosingFuncSig(mlir::func::FuncOp F) {
  if (F.empty()) return;
  // Walk every func.return; for each, recompute its operand types
  // and overwrite the function's return-type vector if any slot
  // changed.
  llvm::SmallVector<mlir::Type> NewResults(
      F.getFunctionType().getResults().begin(),
      F.getFunctionType().getResults().end());
  bool Changed = false;
  F.walk([&](mlir::func::ReturnOp Ret) {
    for (unsigned I = 0;
         I < Ret.getNumOperands() && I < NewResults.size(); ++I) {
      auto Cur = NewResults[I];
      auto New = Ret.getOperand(I).getType();
      if (Cur != New) {
        NewResults[I] = New;
        Changed = true;
      }
    }
  });
  if (Changed) {
    F.setFunctionType(mlir::FunctionType::get(
        F.getContext(), F.getFunctionType().getInputs(), NewResults));
  }
}

} // namespace

bool runUnifyMixedWidthStores(mlir::ModuleOp M) {
  bool AnyChanged = false;
  M.walk([&](mlir::Operation *Op) {
    if (!isMatlabOp(Op, "matlab.alloc")) return;
    if (Op->getNumResults() != 1) return;
    mlir::Value Slot = Op->getResult(0);

    llvm::SmallVector<mlir::Operation *, 4> Stores;
    llvm::SmallVector<unsigned, 4> Widths;
    if (!collectStores(Slot, Stores, Widths)) return;
    if (Stores.size() < 2) return;
    unsigned Max = 0;
    bool AnyMismatch = false;
    for (unsigned W : Widths) {
      if (W > Max) Max = W;
    }
    for (unsigned W : Widths) if (W != Max) AnyMismatch = true;
    if (!AnyMismatch) return;

    // Widen every narrower store via `arith.extsi`. We assume signed
    // widening — matches the SV pipeline's `arith.extsi` everywhere
    // else (fi-saturate path always sign-extends from the natural
    // operand width).
    auto WideTy = mlir::IntegerType::get(M.getContext(), Max);
    for (mlir::Operation *S : Stores) {
      mlir::Value V = S->getOperand(0);
      auto IT = mlir::cast<mlir::IntegerType>(V.getType());
      if (IT.getWidth() == Max) continue;
      mlir::OpBuilder B(S);
      mlir::Value Wide =
          mlir::arith::ExtSIOp::create(B, S->getLoc(), WideTy, V);
      S->setOperand(0, Wide);
    }

    // Retype the alloc result + every load on this slot. The
    // alloc's result still names the "slot address" but its TYPE
    // tracks the unified element type so HWLegalize's
    // synth-check accepts (`none` is rejected as unsynthesisable).
    Slot.setType(WideTy);

    llvm::SmallVector<mlir::Operation *, 4> Loads;
    collectLoads(Slot, Loads);
    for (mlir::Operation *L : Loads) {
      if (L->getNumResults() != 1) continue;
      mlir::Value R = L->getResult(0);
      if (R.getType() == WideTy) continue;
      R.setType(WideTy);
    }
    AnyChanged = true;

    // Refresh the enclosing function signature in case the load
    // feeds a func.return.
    if (auto F = Op->getParentOfType<mlir::func::FuncOp>())
      refreshEnclosingFuncSig(F);
  });
  return AnyChanged;
}

} // namespace mlirgen
} // namespace matlab
