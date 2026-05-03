// Slot-type inference for `matlab.alloc` ops still typed `none`
// after the LowerUserCalls iteration loop runs. The user-call
// pass's `propagateScalarTypes` does this same work but only for
// functions still reachable via a `matlab.call` site — once those
// collapse to `func.call` (which happens in the same pass on the
// first iteration that refines a signature), the slot-retype
// logic stops running. This standalone pass picks up where that
// left off.
//
// The retype rule mirrors `propagateScalarTypes` exactly: if every
// `matlab.store` writing to a `none`-typed slot has the same
// concrete scalar primitive value type, the slot's result gets
// retyped and every `matlab.load` reading from it follows.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isScalarPrim(mlir::Type T) {
  if (mlir::isa<mlir::IntegerType>(T)) return true;
  if (mlir::isa<mlir::FloatType>(T)) return true;
  /* Phase 6 — sym values are !llvm.ptr (matlab_sym* / matlab_symmat*).
   * Treat as scalar primitives so RefineSlotTypes can promote slots
   * that store sym/symmat values stored from Symbolic Math Toolbox
   * builtins. Without this, slots created by `syms x` lowering stay
   * none-typed forever and survive to EmitC as unsupported matlab.alloc. */
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(T)) return true;
  return false;
}

bool isMatlabOp(mlir::Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

} // namespace

bool runRefineSlotTypes(mlir::ModuleOp M) {
  bool ChangedAny = false;
  bool Iterating = true;
  // Iterate until fixpoint — retyping one slot may unblock another
  // (a load from slot A whose type just became concrete now feeds a
  // store into slot B that was previously polymorphic).
  while (Iterating) {
    Iterating = false;
    M.walk([&](mlir::Operation *Op) {
      if (!isMatlabOp(Op, "matlab.alloc")) return;
      if (Op->getNumResults() != 1) return;
      mlir::Value Slot = Op->getResult(0);
      if (!mlir::isa<mlir::NoneType>(Slot.getType())) return;
      mlir::Type Stored;
      bool Consistent = true;
      bool Any = false;
      for (mlir::OpOperand &Use : Slot.getUses()) {
        mlir::Operation *U = Use.getOwner();
        if (isMatlabOp(U, "matlab.store") &&
            U->getNumOperands() == 2 &&
            U->getOperand(1) == Slot) {
          mlir::Type T = U->getOperand(0).getType();
          if (!isScalarPrim(T)) { Consistent = false; break; }
          if (!Any) { Stored = T; Any = true; }
          else if (Stored != T) { Consistent = false; break; }
        }
      }
      if (!Any || !Consistent) return;
      // Retype the slot.
      Slot.setType(Stored);
      // Retype any `none`-typed `matlab.load` reading from this
      // slot. Loads with a concrete result type that already
      // matches stay; loads with a different concrete type are
      // user-visible bugs and we leave them for downstream passes
      // to surface.
      for (mlir::OpOperand &Use : Slot.getUses()) {
        mlir::Operation *U = Use.getOwner();
        if (!isMatlabOp(U, "matlab.load")) continue;
        if (U->getNumResults() != 1) continue;
        mlir::Value LoadRes = U->getResult(0);
        if (mlir::isa<mlir::NoneType>(LoadRes.getType()))
          LoadRes.setType(Stored);
      }
      ChangedAny = true;
      Iterating = true;
    });
  }
  return ChangedAny;
}

} // namespace mlirgen
} // namespace matlab
