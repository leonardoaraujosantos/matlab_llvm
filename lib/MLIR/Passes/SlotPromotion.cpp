// Intra-block slot promotion for matlab.alloc / matlab.load / matlab.store.
//
// Strategy (simple but high-impact):
//   1. For each matlab.alloc in a block:
//      - Collect its users. If any user is outside this block or isn't a
//        matlab.load / matlab.store of the slot, skip it.
//      - Gather stores and loads in program order within the block.
//   2. Walk the block's op list. Maintain a current-value map: slot -> last
//      stored Value. When we see a store, update the map. When we see a load
//      whose operand is a promotable slot with a known current value, replace
//      the load's result with that value and erase the load.
//   3. After the walk, if all loads of the slot have been replaced and no
//      uses remain besides the slot's own stores, erase the stores and the
//      alloc.
//
// This handles the common case where a function's entry block has many allocs
// + initial parameter spills + straight-line assignments before any control
// flow. Slots that are referenced inside scf.if / matlab.for / matlab.while
// bodies are left alone — proper mem2reg for those needs iter_args support,
// which is Phase 6 territory.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isMatlabOp(mlir::Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

/// Returns true if every use of `Slot` is a matlab.load/matlab.store in the
/// same block, and in the case of store it's the *slot* operand (operand #1),
/// not the value operand.
bool isLocalSlot(mlir::Value Slot, mlir::Block *HomeBlock) {
  for (mlir::OpOperand &Use : Slot.getUses()) {
    mlir::Operation *Owner = Use.getOwner();
    if (Owner->getBlock() != HomeBlock) return false;
    if (isMatlabOp(Owner, "matlab.load")) {
      if (Owner->getNumOperands() != 1 || Owner->getOperand(0) != Slot)
        return false;
      continue;
    }
    if (isMatlabOp(Owner, "matlab.store")) {
      // Operand 0 is the value, Operand 1 is the slot. If the slot flows as
      // operand 0, that means someone stored the slot itself — bail.
      if (Owner->getNumOperands() != 2) return false;
      if (Owner->getOperand(1) != Slot) return false;
      continue;
    }
    return false;
  }
  return true;
}

void promoteBlock(mlir::Block &BB, bool &Changed) {
  // Candidate slots defined in this block.
  llvm::SmallVector<mlir::Operation *, 16> Allocs;
  for (mlir::Operation &Op : BB) {
    if (isMatlabOp(&Op, "matlab.alloc") && Op.getNumResults() == 1)
      Allocs.push_back(&Op);
  }

  // Filter to locally-used slots.
  llvm::SmallPtrSet<mlir::Operation *, 16> Promotable;
  for (mlir::Operation *A : Allocs) {
    if (isLocalSlot(A->getResult(0), &BB)) Promotable.insert(A);
  }
  if (Promotable.empty()) return;

  // Walk the block in order, maintaining the latest stored Value per slot.
  llvm::DenseMap<mlir::Value, mlir::Value> Current;

  // Loads we've SSA-replaced — safe to erase unconditionally since their
  // results now have no users.
  llvm::SmallVector<mlir::Operation *, 32> LoadsToErase;
  // Stores keyed by their slot's defining op. We only erase these once the
  // slot is confirmed fully promoted — a mid-block type-mismatched load
  // demotes the slot, and in that case the surviving load needs the store
  // to keep providing its value.
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<mlir::Operation *, 4>> StoresPerSlot;

  for (mlir::Operation &Op : llvm::make_early_inc_range(BB)) {
    if (isMatlabOp(&Op, "matlab.store")) {
      mlir::Value Val = Op.getOperand(0);
      mlir::Value Slot = Op.getOperand(1);
      if (auto *Def = Slot.getDefiningOp();
          Def && Promotable.count(Def)) {
        Current[Slot] = Val;
        StoresPerSlot[Def].push_back(&Op);
      }
      continue;
    }
    if (isMatlabOp(&Op, "matlab.load")) {
      mlir::Value Slot = Op.getOperand(0);
      if (auto *Def = Slot.getDefiningOp();
          Def && Promotable.count(Def)) {
        auto It = Current.find(Slot);
        if (It != Current.end()) {
          mlir::Value V = It->second;
          // Only replace if the value type matches the load result type.
          if (V.getType() == Op.getResult(0).getType()) {
            Op.getResult(0).replaceAllUsesWith(V);
            LoadsToErase.push_back(&Op);
            continue;
          }
        }
        // Load with no prior store (undef use) or mismatched type — don't
        // promote this slot. The staged stores for this slot will NOT be
        // erased, so the surviving load keeps reading the right value.
        Promotable.erase(Def);
        Current.erase(Slot);
      }
      continue;
    }
  }

  // Erase replaced loads (their results have no users after RAUW).
  for (mlir::Operation *Op : LoadsToErase) Op->erase();
  // For slots that stayed promotable, erase staged stores and the alloc.
  for (mlir::Operation *Alloc : Promotable) {
    auto It = StoresPerSlot.find(Alloc);
    if (It != StoresPerSlot.end())
      for (mlir::Operation *S : It->second) S->erase();
    if (Alloc->getResult(0).use_empty()) {
      Alloc->erase();
      Changed = true;
    }
  }
}

} // namespace

bool runSlotPromotion(mlir::ModuleOp M) {
  bool Changed = false;
  // Iterate a couple of rounds in case promotion unblocks further promotion
  // (e.g. slots that were stored-to a previously-promoted value).
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool Local = false;
    M.walk([&](mlir::func::FuncOp Fn) {
      for (mlir::Block &BB : Fn.getBody())
        promoteBlock(BB, Local);
    });
    if (!Local) break;
    Changed = true;
  }
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
