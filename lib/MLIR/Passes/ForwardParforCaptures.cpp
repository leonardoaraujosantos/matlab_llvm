// ForwardParforCaptures — bridge pass between SlotPromotion and
// OutlineParfor that closes the "outer-scope literal captured into a
// parfor body" gap documented in issue #20.
//
// Motivation
// ----------
// MATLAB's most common parfor idiom is to pre-compute a scalar (problem
// size, tolerance, iteration count, …) outside the loop and read it
// inside:
//
//     W = 800;
//     parfor j = 1:N
//         total = total + j * W;     % `W` captured from outer scope
//     end
//
// Today SlotPromotion (the intra-block mem2reg-lite pass) refuses to
// promote `W`'s slot the moment it sees a use in a different block
// (the parfor body's block).  The outer matlab.alloc then survives,
// and OutlineParfor's capture analysis rejects it:
//
//     parfor: body captures value of unsupported defining op
//     'matlab.alloc'
//
// because matlab.alloc is not on the cloneable-external allowlist.
//
// Strategy (Phase 1 of issue #20's fix plan)
// ------------------------------------------
// For each matlab.parfor in the module, for each matlab.load inside its
// body whose slot is defined OUTSIDE the parfor:
//
//   1. Locate every matlab.store of that slot.
//   2. Confirm they all live in an ancestor block (i.e. they're outer-
//      scope writes, not inner mutations).
//   3. Confirm every store is BEFORE the parfor in program order — i.e.
//      no in-between writes between the store(s) and the parfor.
//   4. Confirm every store's value operand resolves to a constant
//      literal (arith::ConstantOp, LLVM::ConstantOp, matlab.const_int,
//      matlab.const_float).  Multiple stores are OK if they agree.
//   5. Clone the literal inside the parfor body and RAUW the load.
//
// The outer slot is intentionally left alone: callers may still want
// to read `W` after the parfor, and rewriting / erasing the slot here
// would force us to re-prove non-aliasing with post-parfor code.  The
// outliner's capture analysis sees only cloned constants on the
// in-body operand chain and succeeds.
//
// Out of scope for this pass (Phase 2/3 follow-ons documented in #20):
//   - Runtime-computed scalars (state[] ABI extension).
//   - Matrix-typed captures (also unblocks GPU Coder T2.B-D).
//   - Stores that are not literal constants but whose value is defined
//     outside-and-dominates-the-parfor and is otherwise cloneable.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

/// A value qualifies as a "cloneable literal" if its defining op produces
/// a constant with no operands — safe to clone into any region without
/// pulling in further dependencies.
bool isCloneableLiteral(Value V) {
  Operation *Def = V.getDefiningOp();
  if (!Def) return false;
  if (Def->getNumOperands() != 0) return false;
  if (isa<arith::ConstantOp>(Def)) return true;
  if (isa<LLVM::ConstantOp>(Def)) return true;
  if (isMatlabOp(Def, "matlab.const_int")) return true;
  if (isMatlabOp(Def, "matlab.const_float")) return true;
  return false;
}

/// True if two literal-producing ops yield the same constant value.
/// Conservative: compares the producing op's value-bearing attribute.
bool literalsAgree(Operation *A, Operation *B) {
  if (!A || !B) return false;
  if (A == B) return true;
  if (A->getName() != B->getName()) return false;
  // Compare the canonical "value" attribute that constant-flavored ops
  // carry.  Both arith.constant and LLVM::ConstantOp store it under
  // "value"; matlab.const_int / matlab.const_float likewise.
  auto AV = A->getAttr("value");
  auto BV = B->getAttr("value");
  return AV && BV && AV == BV;
}

/// True if `Earlier` appears before `Later` in their shared block.
/// Caller must guarantee same parent block.
bool isBefore(Operation *Earlier, Operation *Later) {
  if (Earlier->getBlock() != Later->getBlock()) return false;
  return Earlier->isBeforeInBlock(Later);
}

/// Try to rewrite a single in-body load by forwarding the slot's
/// pre-parfor literal value.  Returns true on rewrite.
bool tryForwardOneLoad(Operation *Load, Operation *Parfor, OpBuilder &B) {
  if (!isMatlabOp(Load, "matlab.load") || Load->getNumOperands() != 1)
    return false;
  Value Slot = Load->getOperand(0);
  Operation *AllocOp = Slot.getDefiningOp();
  if (!isMatlabOp(AllocOp, "matlab.alloc")) return false;

  /* The slot must live OUTSIDE the parfor.  The cheapest reliable check
   * is: the alloc's block must be an ancestor of the parfor's block.
   * A nested-region alloc (inside the parfor body) is the trivial intra-
   * region case and SlotPromotion handles it on a later iteration. */
  if (AllocOp->getBlock() == Parfor->getBlock())
    ; /* parent block is fine — most common shape */
  Region *AllocRegion = AllocOp->getParentRegion();
  if (!AllocRegion->isAncestor(Parfor->getParentRegion()))
    return false;
  if (AllocRegion == Load->getParentRegion()) return false;

  /* Walk all stores of the slot.  Every store must:
   *   (a) live in the same region as the alloc (no inner-region mutations),
   *   (b) precede the parfor (or some ancestor of it) in program order,
   *   (c) store a cloneable literal.
   * If multiple stores agree on the same constant, we still forward. */
  Operation *ChosenLiteral = nullptr;
  for (OpOperand &Use : Slot.getUses()) {
    Operation *U = Use.getOwner();
    if (!isMatlabOp(U, "matlab.store") || U->getNumOperands() != 2)
      continue;
    if (U->getOperand(1) != Slot) return false; /* slot flows as value */
    if (U->getParentRegion() != AllocRegion) return false;

    /* Find the ancestor of the parfor that lives in AllocRegion to
     * compare program order against the store. */
    Operation *ParforAnchor = Parfor;
    while (ParforAnchor && ParforAnchor->getParentRegion() != AllocRegion)
      ParforAnchor = ParforAnchor->getParentOp();
    if (!ParforAnchor) return false;
    if (!isBefore(U, ParforAnchor)) return false;

    Value StoredVal = U->getOperand(0);
    if (!isCloneableLiteral(StoredVal)) return false;
    Operation *LitDef = StoredVal.getDefiningOp();
    if (!ChosenLiteral) ChosenLiteral = LitDef;
    else if (!literalsAgree(ChosenLiteral, LitDef)) return false;
  }
  if (!ChosenLiteral) return false;

  /* Clone the literal at the load site, replace, erase load. */
  B.setInsertionPoint(Load);
  Operation *Cloned = B.clone(*ChosenLiteral);
  /* Type-match guard: if the cloned literal's result type differs from
   * the load's result type (e.g. f64 literal vs none-typed load that
   * later passes will refine), bail rather than introduce a verifier
   * mismatch. */
  if (Cloned->getResult(0).getType() != Load->getResult(0).getType()) {
    Cloned->erase();
    return false;
  }
  Load->getResult(0).replaceAllUsesWith(Cloned->getResult(0));
  Load->erase();
  return true;
}

} // namespace

unsigned runForwardParforCaptures(mlir::ModuleOp M) {
  llvm::SmallVector<Operation *> Parfors;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.parfor")) Parfors.push_back(Op);
  });
  if (Parfors.empty()) return 0;

  OpBuilder B(M.getContext());
  unsigned Rewritten = 0;
  for (Operation *Parfor : Parfors) {
    if (Parfor->getNumRegions() != 1) continue;
    Region &Body = Parfor->getRegion(0);
    /* Snapshot loads first — tryForwardOneLoad mutates the IR. */
    llvm::SmallVector<Operation *> Loads;
    Body.walk([&](Operation *Op) {
      if (isMatlabOp(Op, "matlab.load")) Loads.push_back(Op);
    });
    for (Operation *L : Loads) {
      if (tryForwardOneLoad(L, Parfor, B)) ++Rewritten;
    }
  }
  return Rewritten;
}

} // namespace mlirgen
} // namespace matlab
