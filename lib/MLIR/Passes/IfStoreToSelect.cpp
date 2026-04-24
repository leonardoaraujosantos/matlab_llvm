// Rewrite `scf.if` whose only effect is storing one of two pre-computed
// SSA values into the same slot, into a single `arith.select` + store.
//
//     scf.if %cond {
//       llvm.store %a, %slot
//     } else {
//       llvm.store %b, %slot
//     }
//
//   =>
//
//     %s = arith.select %cond, %a, %b
//     llvm.store %s, %slot
//
// Combined with Mem2RegLite this collapses the slot entirely when both
// arms are the slot's only writers. We intentionally require each arm
// to contain exactly one op (the store) plus the auto-yield, so we don't
// need to reason about hoisting pure ops out of an if region.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {
using namespace mlir;

// Return the (single) store op in `region` if the region's only block
// contains exactly: one store op and one terminator (scf.yield).
// Returns nullptr otherwise.
LLVM::StoreOp lonelyStore(Region &R) {
  if (!R.hasOneBlock()) return {};
  Block &B = R.front();
  // Expect exactly one non-terminator op + the yield.
  if (B.empty()) return {};
  Operation &First = B.front();
  auto Store = dyn_cast<LLVM::StoreOp>(First);
  if (!Store) return {};
  Operation *Next = First.getNextNode();
  if (!Next || !isa<scf::YieldOp>(Next)) return {};
  if (Next->getNextNode()) return {};
  return Store;
}

bool tryRewrite(scf::IfOp If) {
  if (If.getNumResults() != 0) return false;
  if (If.getElseRegion().empty()) return false;

  LLVM::StoreOp ThenSt = lonelyStore(If.getThenRegion());
  LLVM::StoreOp ElseSt = lonelyStore(If.getElseRegion());
  if (!ThenSt || !ElseSt) return false;
  if (ThenSt.getAddr() != ElseSt.getAddr()) return false;

  Value ThenVal = ThenSt.getValue();
  Value ElseVal = ElseSt.getValue();
  if (ThenVal.getType() != ElseVal.getType()) return false;

  // Both SSA values must be defined outside the scf.if so we can reference
  // them from before the op. Block-argument values are defined by their
  // owning block — check that they originate outside the if's regions.
  auto OutsideIf = [&](Value V) {
    Operation *Def = V.getDefiningOp();
    if (Def)
      return !If->isAncestor(Def);
    // Block argument: the block must not be nested inside the scf.if.
    Block *OwnBlock = cast<BlockArgument>(V).getOwner();
    return !If->isAncestor(OwnBlock->getParentOp());
  };
  if (!OutsideIf(ThenVal) || !OutsideIf(ElseVal)) return false;
  if (!OutsideIf(ThenSt.getAddr())) return false;

  OpBuilder B(If);
  Value Sel = arith::SelectOp::create(B, If.getLoc(), If.getCondition(),
                                      ThenVal, ElseVal);
  LLVM::StoreOp::create(B, If.getLoc(), Sel, ThenSt.getAddr());
  If.erase();
  return true;
}

bool runOnRegion(Region &R) {
  bool Changed = false;
  // Collect candidate ifs first — rewriting mutates block iteration.
  llvm::SmallVector<scf::IfOp> Ifs;
  R.walk([&](scf::IfOp Op) { Ifs.push_back(Op); });
  for (scf::IfOp Op : Ifs)
    Changed |= tryRewrite(Op);
  return Changed;
}

} // namespace

bool runIfStoreToSelect(ModuleOp M) {
  bool Changed = false;
  for (Operation &Op : M.getBody()->getOperations()) {
    if (auto F = dyn_cast<func::FuncOp>(Op))
      Changed |= runOnRegion(F.getBody());
    else if (auto F = dyn_cast<LLVM::LLVMFuncOp>(Op))
      Changed |= runOnRegion(F.getBody());
  }
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
