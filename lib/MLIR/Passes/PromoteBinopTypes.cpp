// lib/MLIR/Passes/PromoteBinopTypes.cpp — propagate operand types
// through `matlab.{add,sub,emul,ediv,matmul}` results when one
// operand is `!llvm.ptr` (a matrix) and the result is still
// `none`-typed.  Companion to PromoteNoneParams.cpp and
// RefineSlotTypes.cpp.
//
// The Lowerer initially types every binop result as `none` when
// Sema couldn't infer.  RefineSlotTypes can promote a slot from
// the type of values stored to it — but only if those values are
// already typed.  Binop results stay `none` until something else
// types them, breaking the type-flow chain:
//
//     %x = matlab.call "gpuArray_rand"(...) : (...) -> none   // x: none
//     %a = matlab.load %a_slot : (f64) -> f64                  // scalar
//     %z = matlab.emul %a, %x  : (f64, none) -> none           // z: none
//     gather(%z)                                                // operand none
//
// After LowerTensorOps rewrites gpuArray_rand to an llvm.call
// returning ptr, the `%x` Value's type becomes ptr — but matlab.emul's
// result type stays `none`, so the slot retype never propagates and
// `gather` fails to dispatch.  This pass closes the gap: walk every
// matlab.* binop with `none` result, check operand types, and if any
// operand is now ptr-typed, retype the result to ptr.  Cascades
// through subsequent matlab.store → slot → matlab.load → next binop
// when run iteratively (the pipeline already does this).
//
// **Idempotent + iteration-safe**: only mutates results that are
// still `none`.  If both operands are scalar f64 and the result is
// `none`, we leave it for `runRefineSlotTypes` / `runLowerScalarsToArith`
// to handle (the existing scalar-propagation lanes).
//
// Returns true if any op was retyped (so the caller can iterate
// until fixpoint).

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

bool isMatlabBinop(Operation *Op) {
  if (!Op) return false;
  StringRef N = Op->getName().getStringRef();
  return N == "matlab.add" || N == "matlab.sub" ||
         N == "matlab.matmul" || N == "matlab.emul" ||
         N == "matlab.ediv";
}

}  // namespace

bool runPromoteBinopTypes(mlir::ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);

  bool Changed = false;
  M.walk([&](Operation *Op) {
    if (!isMatlabBinop(Op)) return;
    if (Op->getNumResults() != 1) return;
    if (Op->getNumOperands() != 2) return;
    if (!mlir::isa<NoneType>(Op->getResult(0).getType())) return;

    Type T0 = Op->getOperand(0).getType();
    Type T1 = Op->getOperand(1).getType();
    /* If either operand is ptr-typed (a matrix), the result is also
     * a matrix (ptr).  This matches the runtime behavior of
     * matlab_add_mm / matlab_emul_ms / matlab_emul_sm etc. — any
     * scalar-by-matrix or matrix-by-matrix binop returns a matrix. */
    if (T0 == PtrTy || T1 == PtrTy) {
      Op->getResult(0).setType(PtrTy);
      Changed = true;
      return;
    }
    /* Tensor operands also indicate matrix semantics; retype to the
     * tensor type that's present. */
    if (mlir::isa<RankedTensorType, UnrankedTensorType>(T0)) {
      Op->getResult(0).setType(T0);
      Changed = true;
      return;
    }
    if (mlir::isa<RankedTensorType, UnrankedTensorType>(T1)) {
      Op->getResult(0).setType(T1);
      Changed = true;
      return;
    }
  });

  /* Propagate to slots: when a binop now produces ptr and is stored
   * into a `none`-typed slot, retype the slot + its load results.
   * Mirrors RefineSlotTypes for slots that the existing pass missed. */
  M.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.store") || Op->getNumOperands() != 2) return;
    Value Val = Op->getOperand(0);
    Value Slot = Op->getOperand(1);
    if (Val.getType() != PtrTy) return;
    if (!mlir::isa<NoneType>(Slot.getType())) return;
    /* Retype the slot itself + every load that reads from it. */
    Slot.setType(PtrTy);
    Changed = true;
    for (Operation *U : Slot.getUsers()) {
      if (isMatlabOp(U, "matlab.load") && U->getNumResults() == 1 &&
          mlir::isa<NoneType>(U->getResult(0).getType())) {
        U->getResult(0).setType(PtrTy);
      }
    }
  });

  /* Also propagate to matlab.call_builtin results: when a call's
   * runtime symbol is one we know returns ptr (gpuArray.X factories,
   * gather, etc.) AND the result is still `none`, retype it.  This
   * unblocks downstream slot retypes without needing LowerTensorOps
   * to have already fired.
   *
   * Conservative — we check the `callee` attribute string against a
   * known-ptr-return whitelist matching the pde_table entries. */
  static const llvm::StringRef PtrReturningCalls[] = {
    "gpuArray_rand", "gpuArray_randn", "gpuArray_zeros",
    "gpuArray_ones", "gpuArray_eye", "gpuArray_linspace",
    "gpuArray", "gather", "arrayfun",
    "matlab_gpucoder_matmatkernel", "matlab_stencilfun",
    "matlab_gpucoder_sort", "matlab_arrayfun",
  };
  M.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    if (Op->getNumResults() != 1) return;
    if (!mlir::isa<NoneType>(Op->getResult(0).getType())) return;
    auto Cal = Op->getAttrOfType<StringAttr>("callee");
    if (!Cal) return;
    StringRef CN = Cal.getValue();
    for (auto Known : PtrReturningCalls) {
      if (CN == Known) {
        Op->getResult(0).setType(PtrTy);
        Changed = true;
        break;
      }
    }
  });

  return Changed;
}

}  // namespace mlirgen
}  // namespace matlab
