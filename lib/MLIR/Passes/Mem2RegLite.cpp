// Promote `llvm.alloca`s that behave like SSA values back into SSA.
//
// After LowerScalarSlots, we commonly end up with allocas that are written
// exactly once (e.g. parameter spills `store %arg, %slot`) and read many
// times. EmitC renders these as a `T slot = 0; void* p = &slot;
// *(T*)p = value;` prelude followed by `*(T*)p` at every read site — noisy
// and bears no resemblance to MATLAB source.
//
// This pass rewrites a conservative subset: when an alloca has exactly
// one store in the entry block (dominating all reads), and every other
// user is a plain `llvm.load` of its address, we replace every load with
// the stored value and erase the store + alloca.
//
// Multi-store allocas (scf.if merge points, loop accumulators) are left
// for EmitC to render normally — those genuinely need a mutable C local.
//
// Runs after LowerIO and before EmitC/LowerToLLVMIR.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {
using namespace mlir;

// Try to promote one alloca. Returns true if the alloca was erased.
bool tryPromote(LLVM::AllocaOp Alloca) {
  // The address escaping into anything other than load/store disqualifies
  // us: casts, calls, GEPs could all observe or mutate the slot.
  LLVM::StoreOp TheStore;
  llvm::SmallVector<LLVM::LoadOp> Loads;
  for (OpOperand &Use : Alloca->getUses()) {
    Operation *U = Use.getOwner();
    if (auto S = dyn_cast<LLVM::StoreOp>(U)) {
      // Must be a store TO the slot, not a store OF the slot's ptr.
      if (S.getAddr() != Alloca.getResult()) return false;
      if (TheStore) return false;  // more than one store
      TheStore = S;
      continue;
    }
    if (auto L = dyn_cast<LLVM::LoadOp>(U)) {
      if (L.getAddr() != Alloca.getResult()) return false;
      Loads.push_back(L);
      continue;
    }
    return false;  // unexpected user
  }

  if (!TheStore) return false;  // write-only / read-only: leave alone

  // The store must dominate every load. We rely on the structural
  // invariant that (a) the alloca sits in the entry block of its function
  // and (b) the store is in the same block at a position before any load
  // in that block. Loads in later / nested blocks are automatically
  // dominated because the entry block dominates all reachable blocks in
  // a structured scf.* region.
  Block *AllocBlock = Alloca->getBlock();
  if (TheStore->getBlock() != AllocBlock) return false;

  // Scan the alloc-block: if we see a load of this slot before the store,
  // the load would observe the zero-init instead of the stored value —
  // bail.
  for (auto It = AllocBlock->begin(); It != AllocBlock->end(); ++It) {
    if (&*It == TheStore.getOperation()) break;
    if (auto L = dyn_cast<LLVM::LoadOp>(&*It)) {
      if (L.getAddr() == Alloca.getResult()) return false;
    }
  }

  // The value being stored must dominate the store itself — trivially
  // true, since it's already an operand. Good to go.
  Value StoredVal = TheStore.getValue();
  for (LLVM::LoadOp L : Loads) {
    // Type check: load result must match the stored value's type.
    if (L.getResult().getType() != StoredVal.getType()) return false;
  }
  // Carry the alloca's user-source variable name forward onto the
  // defining op of the stored value, so downstream emitters can
  // surface readable names instead of falling back to a fresh `vN` id.
  // Mirrors what SlotPromotion does at the MATLAB-dialect level for
  // type-matched promotions; here we cover the LLVM-dialect case where
  // the slot survived earlier passes due to a `none`-typed alloc and
  // got Mem2Reg'd later. Don't clobber a name a previous pass set.
  if (auto NA =
          Alloca->getAttrOfType<StringAttr>("matlab.name")) {
    if (Operation *Def = StoredVal.getDefiningOp())
      if (!Def->hasAttr("matlab.name"))
        Def->setAttr("matlab.name", NA);
  }
  for (LLVM::LoadOp L : Loads) {
    L.getResult().replaceAllUsesWith(StoredVal);
    L.erase();
  }
  TheStore.erase();
  Alloca.erase();
  return true;
}

bool runOnFunction(Region &Body) {
  if (Body.empty()) return false;
  Block &Entry = Body.front();
  llvm::SmallVector<LLVM::AllocaOp> Candidates;
  for (Operation &Op : Entry) {
    if (auto A = dyn_cast<LLVM::AllocaOp>(Op)) {
      // Only scalar slots (element type = primitive). Aggregate allocas
      // (matrix-literal buffers from LowerTensorOps) have semantic element
      // structure we don't want to alias-analyze.
      Type ET = A.getElemType();
      if (!isa<Float64Type, Float32Type, IntegerType,
               LLVM::LLVMPointerType>(ET)) continue;
      Candidates.push_back(A);
    }
  }
  bool Changed = false;
  for (LLVM::AllocaOp A : Candidates)
    Changed |= tryPromote(A);
  return Changed;
}

} // namespace

bool runMem2RegLite(ModuleOp M) {
  bool Changed = false;
  for (Operation &Op : M.getBody()->getOperations()) {
    if (auto F = dyn_cast<func::FuncOp>(Op))
      Changed |= runOnFunction(F.getBody());
    else if (auto F = dyn_cast<LLVM::LLVMFuncOp>(Op))
      Changed |= runOnFunction(F.getBody());
  }
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
