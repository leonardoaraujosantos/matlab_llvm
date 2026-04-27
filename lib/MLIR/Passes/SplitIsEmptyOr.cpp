// Pre-HWStateInfer normalization: `if isempty(c) || X ... end`
// rewriting into two separate scf.if guards (the HDL Coder
// canonical form). HWStateInfer's matcher requires the isempty
// result to feed exactly one cmpf that gates one scf.if — the
// joined `||` form fails that check, but the user-written joined
// form is common enough (literal examples/hdl/mealy_fsm.m and
// moore_fsm.m use it) that it's worth handling automatically
// rather than requiring a user-side rewrite.
//
// The transformation:
//
//   %ie = llvm.call @matlab_persistent_isempty(%idx) : (i32) -> f64
//   %or = matlab.short_or(%ie, %X) : (f64, T) -> i1
//   scf.if %or { ...body... }
//
// becomes:
//
//   %ie = llvm.call @matlab_persistent_isempty(%idx) : (i32) -> f64
//   %cmp = arith.cmpf one, %ie, 0.0 : f64
//   scf.if %cmp { ...body... }
//   scf.if %X' { ...body cloned... }   // %X' truthy-coerced
//
// The body is duplicated (both arms execute the same init). The
// `%or` op and any cmpf consuming it are erased.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isMatlabOpName(mlir::Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

bool isPersistentIsEmptyCall(mlir::Operation *Op) {
  if (!Op) return false;
  if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    auto Sym = C.getCallee();
    return Sym && *Sym == "matlab_persistent_isempty";
  }
  if (Op->getName().getStringRef() == "matlab.call_builtin") {
    auto S = Op->getAttrOfType<mlir::StringAttr>("callee");
    return S && S.getValue() == "matlab_persistent_isempty";
  }
  return false;
}

/// Coerce an arbitrary value to i1 in front of an scf.if. The
/// existing Lowerer::fixupIfCond knows the same logic; we redo it
/// here because we're operating on already-lowered IR.
mlir::Value coerceToI1(mlir::OpBuilder &B, mlir::Value V,
                       mlir::Location LC) {
  mlir::Type T = V.getType();
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    if (IT.getWidth() == 1) return V;
    auto Z = mlir::arith::ConstantOp::create(
        B, LC, IT, mlir::IntegerAttr::get(IT, 0));
    return mlir::arith::CmpIOp::create(
        B, LC, mlir::arith::CmpIPredicate::ne, V, Z);
  }
  if (mlir::isa<mlir::Float64Type, mlir::Float32Type>(T)) {
    auto FT = mlir::cast<mlir::FloatType>(T);
    auto Z = mlir::arith::ConstantOp::create(
        B, LC, FT, mlir::FloatAttr::get(FT, 0.0));
    return mlir::arith::CmpFOp::create(
        B, LC, mlir::arith::CmpFPredicate::ONE, V, Z);
  }
  // None / unknown — fall back to the unrealized-cast placeholder
  // that RefineIfConds will fix up later.
  if (mlir::isa<mlir::NoneType>(T)) {
    auto I1 = B.getI1Type();
    return mlir::UnrealizedConversionCastOp::create(
               B, LC, mlir::TypeRange{I1}, mlir::ValueRange{V})
        .getResult(0);
  }
  return V;
}

/// True when the only consumer of `V` (transitively through any
/// number of cmp* ops with a single use) is a single scf.if
/// condition. Returns the scf.if op and the immediate user-of-V
/// (which the caller will edit to swap in the isempty operand).
mlir::scf::IfOp findGuardingIf(mlir::Value V) {
  if (!V.hasOneUse()) return nullptr;
  return mlir::dyn_cast<mlir::scf::IfOp>(V.use_begin()->getOwner());
}

} // namespace

bool runSplitIsEmptyOr(mlir::ModuleOp M) {
  llvm::SmallVector<mlir::Operation *, 4> Worklist;
  M.walk([&](mlir::Operation *Op) {
    if (!isMatlabOpName(Op, "matlab.short_or")) return;
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1) return;
    // One operand must be (or transitively cmp) an isempty call.
    bool L = isPersistentIsEmptyCall(Op->getOperand(0).getDefiningOp());
    bool R = isPersistentIsEmptyCall(Op->getOperand(1).getDefiningOp());
    if (!L && !R) return;
    auto If = findGuardingIf(Op->getResult(0));
    if (!If) return;
    Worklist.push_back(Op);
  });

  for (mlir::Operation *Op : Worklist) {
    auto IfOp = findGuardingIf(Op->getResult(0));
    if (!IfOp) continue;  // mutated by an earlier iteration
    bool LIsIE = isPersistentIsEmptyCall(Op->getOperand(0).getDefiningOp());
    mlir::Value IEVal = LIsIE ? Op->getOperand(0) : Op->getOperand(1);
    mlir::Value Other = LIsIE ? Op->getOperand(1) : Op->getOperand(0);

    mlir::OpBuilder B(IfOp);
    mlir::Location LC = IfOp.getLoc();

    // 1. Build the isempty-only condition (cmpf one, %ie, 0.0). The
    //    Lowerer normally emits this for `if isempty(c)`; we redo it
    //    by hand because we're after lowering.
    mlir::Value IECond = coerceToI1(B, IEVal, LC);

    // 2. Replace the IfOp's condition operand with the new IECond.
    IfOp.getConditionMutable().assign(IECond);

    // 3. Insert the cloned `if X` right after the original. Build
    //    its condition expression at the right insertion point: a
    //    fresh builder positioned right after IfOp inserts the
    //    coerce ops first, then the clone consumes them.
    mlir::OpBuilder Post(IfOp);
    Post.setInsertionPointAfter(IfOp);
    mlir::Value OtherCond = coerceToI1(Post, Other, LC);
    mlir::IRMapping Mapping;
    auto Clone = mlir::cast<mlir::scf::IfOp>(Post.clone(*IfOp, Mapping));
    Clone.getConditionMutable().assign(OtherCond);

    // 4. Erase the OR.
    Op->erase();
  }
  return true;
}

} // namespace mlirgen
} // namespace matlab
