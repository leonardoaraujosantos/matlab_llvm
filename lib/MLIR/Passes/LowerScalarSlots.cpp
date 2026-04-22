// Convert every `matlab.alloc` with a scalar-primitive result type (f64,
// f32, iN) into an `llvm.alloca`, and every `matlab.load` / `matlab.store`
// to it into `llvm.load` / `llvm.store`. Any alloc whose result type is
// `none` or aggregate (tensor, cell, struct) is left alone — those still
// need a richer ABI.
//
// This runs after SlotPromotion (which removes intra-block slots) and
// after LowerUserCalls type-refinement, so the surviving scalar allocs
// are exactly the ones that survived both passes: cross-block or
// function-body locals whose types are now concretely f64.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {
using namespace mlir;

bool isMatlabOp(Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

bool isScalarPrimitive(Type T) {
  return mlir::isa<Float64Type, Float32Type, IntegerType>(T);
}

} // namespace

bool runLowerScalarSlots(ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto I64 = IntegerType::get(Ctx, 64);
  bool Changed = false;

  llvm::SmallVector<Operation *> Allocs;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.alloc") && Op->getNumResults() == 1 &&
        isScalarPrimitive(Op->getResult(0).getType()))
      Allocs.push_back(Op);
  });

  for (Operation *Alloc : Allocs) {
    Type ElemTy = Alloc->getResult(0).getType();
    OpBuilder B(Alloc);
    Value One = LLVM::ConstantOp::create(B, Alloc->getLoc(), I64,
                                          B.getI64IntegerAttr(1));
    Value Ptr = LLVM::AllocaOp::create(B, Alloc->getLoc(), PtrTy, ElemTy,
                                        One, /*alignment=*/0);

    // Rewrite loads and stores that use this slot. Walk a copy because
    // erase mutates the use list.
    llvm::SmallVector<Operation *> ToErase;
    for (OpOperand &Use : Alloc->getResult(0).getUses()) {
      Operation *U = Use.getOwner();
      if (isMatlabOp(U, "matlab.load") && U->getNumOperands() == 1 &&
          U->getNumResults() == 1) {
        B.setInsertionPoint(U);
        Value V = LLVM::LoadOp::create(B, U->getLoc(), ElemTy, Ptr);
        U->getResult(0).replaceAllUsesWith(V);
        ToErase.push_back(U);
      } else if (isMatlabOp(U, "matlab.store") && U->getNumOperands() == 2 &&
                 U->getOperand(1) == Alloc->getResult(0)) {
        Value V = U->getOperand(0);
        if (V.getType() != ElemTy) {
          // Operand type drift after earlier refinements — skip this slot.
          ToErase.clear();
          goto NextAlloc;
        }
        B.setInsertionPoint(U);
        LLVM::StoreOp::create(B, U->getLoc(), V, Ptr);
        ToErase.push_back(U);
      } else {
        // Unexpected user; skip this slot conservatively.
        ToErase.clear();
        goto NextAlloc;
      }
    }
    for (Operation *U : ToErase) U->erase();
    Alloc->erase();
    Changed = true;
  NextAlloc:;
  }

  return Changed;
}

} // namespace mlirgen
} // namespace matlab
