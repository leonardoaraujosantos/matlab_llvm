// Outlines each matlab.parfor body into a private func.func and replaces
// the parfor op with an llvm.call to matlab_parfor_dispatch. The dispatcher
// (in runtime/matlab_runtime.c) spawns one pthread per iteration.
//
// v1 scope: only supports bodies that reference the loop variable, arith
// constants, and module-level symbols (llvm.mlir.global / addressof /
// function symbols). Bodies that capture stack-allocated slots, block
// arguments from outside, or other per-call values are rejected — a future
// pass will extend this with a state-struct ABI.
//
// The matlab.parfor iter operand must be a matlab.range with two or three
// f64 operands; we pass (start, step, end) to the dispatcher.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <iostream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

LLVM::LLVMFuncOp getOrInsertRTDecl(OpBuilder &B, ModuleOp M, StringRef Name,
                                   Type Result, ArrayRef<Type> Args) {
  if (auto Existing = M.lookupSymbol<LLVM::LLVMFuncOp>(Name)) return Existing;
  OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(M.getBody());
  auto Ty = LLVM::LLVMFunctionType::get(Result, Args);
  auto F = LLVM::LLVMFuncOp::create(B, M.getLoc(), Name, Ty);
  F.setLinkage(LLVM::Linkage::External);
  return F;
}

// An operation defined outside the parfor body is "cloneable" if it's a
// no-side-effect value producer whose ops don't depend on any parfor-external
// values themselves (constants, addressof). Return true on acceptance.
bool isCloneableExternal(Operation *Op) {
  if (!Op) return false;
  if (isa<arith::ConstantOp>(Op)) return true;
  if (isa<LLVM::ConstantOp>(Op)) return true;
  if (isa<LLVM::ZeroOp>(Op)) return true;
  if (isa<LLVM::AddressOfOp>(Op)) return true;
  return false;
}

// Outline one matlab.parfor op. Returns true on success.
bool outlineParfor(Operation *Parfor, unsigned Id) {
  Location Loc = Parfor->getLoc();
  MLIRContext *Ctx = Parfor->getContext();
  ModuleOp Module = Parfor->getParentOfType<ModuleOp>();
  OpBuilder B(Ctx);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto VoidTy = LLVM::LLVMVoidType::get(Ctx);

  // Extract (start, step, end) from the matlab.range feeding the parfor.
  if (Parfor->getNumOperands() != 1) return false;
  Value Iter = Parfor->getOperand(0);
  Operation *Range = Iter.getDefiningOp();
  if (!isMatlabOp(Range, "matlab.range")) {
    std::cerr << "parfor: iter operand is not a matlab.range — skipping\n";
    return false;
  }

  Value Start, Step, End;
  if (Range->getNumOperands() == 2) {
    Start = Range->getOperand(0);
    End = Range->getOperand(1);
    Step = nullptr;
  } else if (Range->getNumOperands() == 3) {
    Start = Range->getOperand(0);
    Step = Range->getOperand(1);
    End = Range->getOperand(2);
  } else {
    return false;
  }
  if (Start.getType() != F64 || End.getType() != F64 ||
      (Step && Step.getType() != F64)) {
    std::cerr << "parfor: range operands must all be f64 — skipping\n";
    return false;
  }

  // The body region: single block, first arg is the induction variable.
  if (Parfor->getNumRegions() != 1) return false;
  Region &Body = Parfor->getRegion(0);
  if (!Body.hasOneBlock()) {
    std::cerr << "parfor: body has multiple blocks — not supported\n";
    return false;
  }
  Block &BodyBlock = Body.front();
  if (BodyBlock.getNumArguments() != 1 ||
      BodyBlock.getArgument(0).getType() != F64) {
    std::cerr << "parfor: body must have a single f64 induction arg\n";
    return false;
  }
  Value IV = BodyBlock.getArgument(0);

  // Redirect loads/stores that reference the outer loop-variable slot to
  // the block argument directly. The frontend lowering stores %iv into an
  // outer slot at body entry, then every `i` read in the body is a
  // matlab.load from that slot. For parfor we need each thread to read the
  // value MATLAB's semantics require ("for parfor each iteration gets its
  // own i"), not the last value written by some other thread — so we
  // bypass the slot entirely inside the body.
  StringRef VarName;
  if (auto VA = Parfor->getAttrOfType<StringAttr>("var"))
    VarName = VA.getValue();
  if (!VarName.empty()) {
    llvm::SmallVector<Operation *, 4> StoresToErase;
    for (Operation &Op : BodyBlock) {
      if (isMatlabOp(&Op, "matlab.load") && Op.getNumOperands() == 1) {
        Operation *SlotDef = Op.getOperand(0).getDefiningOp();
        if (isMatlabOp(SlotDef, "matlab.alloc")) {
          auto NameA = SlotDef->getAttrOfType<StringAttr>("name");
          if (NameA && NameA.getValue() == VarName) {
            // RAUW unconditionally. The load's result type may be `none`
            // (Sema couldn't infer) while IV is f64 — that's fine because
            // the downstream consumer is an unregistered matlab.call_builtin
            // which doesn't enforce type equality on operands, and once the
            // disp is rewritten by LowerIO the signature matches the f64
            // runtime entry point exactly.
            Op.getResult(0).replaceAllUsesWith(IV);
          }
        }
      }
    }
    for (Operation &Op : BodyBlock) {
      if (isMatlabOp(&Op, "matlab.store") && Op.getNumOperands() == 2) {
        Operation *SlotDef = Op.getOperand(1).getDefiningOp();
        if (isMatlabOp(SlotDef, "matlab.alloc")) {
          auto NameA = SlotDef->getAttrOfType<StringAttr>("name");
          if (NameA && NameA.getValue() == VarName)
            StoresToErase.push_back(&Op);
        }
      }
    }
    for (Operation *S : StoresToErase) S->erase();
    llvm::SmallVector<Operation *, 4> DeadLoads;
    for (Operation &Op : BodyBlock)
      if (isMatlabOp(&Op, "matlab.load") && Op.use_empty())
        DeadLoads.push_back(&Op);
    for (Operation *L : DeadLoads) L->erase();
  }

  // --- Reduction detection -------------------------------------------------
  // Find patterns of the form
  //     %v   = matlab.load(%slot)                          // slot is outer
  //     %sum = matlab.add(%v, %rhs)  (or matlab.add(%rhs, %v))
  //            matlab.store(%sum, %slot)
  // where %rhs is safe to evaluate per-iteration (doesn't transitively load
  // from %slot itself). Each such %slot becomes a "reduction variable" that
  // the body contributes to via mutex-protected matlab_reduce_add_f64 calls.
  struct Reduction {
    Operation *AllocOp;   // outer matlab.alloc — the slot
    Operation *LoadOp;    // matlab.load in body
    Operation *AddOp;     // matlab.add in body
    Operation *StoreOp;   // matlab.store in body
    Value Rhs;            // the non-load addend
  };
  llvm::SmallVector<Reduction> Reductions;
  llvm::DenseSet<Operation *> ReductionBodyOps;
  {
    for (Operation &Op : BodyBlock) {
      if (!isMatlabOp(&Op, "matlab.store") || Op.getNumOperands() != 2)
        continue;
      Value Stored = Op.getOperand(0);
      Value Slot   = Op.getOperand(1);
      Operation *AddOp = Stored.getDefiningOp();
      bool IsAdd = AddOp && (isMatlabOp(AddOp, "matlab.add") ||
                             isa<arith::AddFOp>(AddOp) ||
                             isa<arith::AddIOp>(AddOp));
      if (!IsAdd || AddOp->getNumOperands() != 2)
        continue;
      Value A = AddOp->getOperand(0), BV = AddOp->getOperand(1);
      Operation *LoadOp = nullptr; Value Rhs;
      auto tryMatch = [&](Value MaybeLoad, Value MaybeRhs) {
        Operation *L = MaybeLoad.getDefiningOp();
        if (isMatlabOp(L, "matlab.load") && L->getOperand(0) == Slot) {
          LoadOp = L; Rhs = MaybeRhs;
        }
      };
      tryMatch(A, BV);
      if (!LoadOp) tryMatch(BV, A);
      if (!LoadOp) continue;
      Operation *AllocOp = Slot.getDefiningOp();
      if (!isMatlabOp(AllocOp, "matlab.alloc")) continue;
      if (AllocOp->getBlock() == &BodyBlock) continue;  // must be outer
      // Reject if %rhs depends on %v (then it's not a reduction).
      llvm::DenseSet<Operation *> Seen;
      std::function<bool(Value)> DependsOnLoad = [&](Value V) -> bool {
        if (V == LoadOp->getResult(0)) return true;
        Operation *D = V.getDefiningOp();
        if (!D || !Seen.insert(D).second) return false;
        for (Value Op : D->getOperands())
          if (DependsOnLoad(Op)) return true;
        return false;
      };
      if (DependsOnLoad(Rhs)) continue;
      Reductions.push_back({AllocOp, LoadOp, AddOp, &Op, Rhs});
      ReductionBodyOps.insert(LoadOp);
      ReductionBodyOps.insert(AddOp);
      ReductionBodyOps.insert(&Op);
    }
  }

  // --- Capture analysis (now excluding reduction chains) ------------------
  llvm::DenseSet<Value> DefinedInside;
  DefinedInside.insert(IV);
  for (Operation &Op : BodyBlock) {
    for (Value R : Op.getResults()) DefinedInside.insert(R);
  }
  // The reduction slot values are consumed by the reduction chain itself;
  // their replacement (a ptr loaded from state) will be created later. For
  // the purpose of capture analysis they are allowed.
  llvm::DenseSet<Value> ReductionSlots;
  for (auto &R : Reductions) ReductionSlots.insert(R.AllocOp->getResult(0));

  llvm::SmallVector<Operation *> ExternsToClone;
  llvm::DenseSet<Operation *> ExternSet;
  for (Operation &Op : BodyBlock) {
    if (ReductionBodyOps.count(&Op)) continue; // will be replaced
    for (Value Operand : Op.getOperands()) {
      if (DefinedInside.count(Operand)) continue;
      if (ReductionSlots.count(Operand)) continue;
      Operation *Def = Operand.getDefiningOp();
      if (!Def) {
        std::cerr << "parfor: body captures a block argument from outside — "
                     "not supported\n";
        return false;
      }
      if (!isCloneableExternal(Def)) {
        std::cerr << "parfor: body captures value of unsupported defining op '"
                  << Def->getName().getStringRef().str() << "'\n";
        return false;
      }
      if (ExternSet.insert(Def).second) ExternsToClone.push_back(Def);
    }
  }

  // --- Create outlined function ------------------------------------------
  // Signature: (f64 iv, ptr state). `state` points to a packed array of
  // reduction-variable pointers (or null if there are no reductions).
  std::string Name = ("__parfor_body_" + llvm::Twine(Id)).str();
  B.setInsertionPointToEnd(Module.getBody());
  auto FnTy = LLVM::LLVMFunctionType::get(VoidTy, {F64, PtrTy});
  auto Fn = LLVM::LLVMFuncOp::create(B, Loc, Name, FnTy);
  Fn.setLinkage(LLVM::Linkage::Internal);
  Block *Entry = Fn.addEntryBlock(B);
  Value InnerIV = Entry->getArgument(0);
  Value State   = Entry->getArgument(1);

  B.setInsertionPointToEnd(Entry);

  // Load each reduction pointer from state[k]. state is a `ptr` pointing to
  // an array of `ptr`. For k-th reduction: `gep ptr, [k]`, then `load ptr`.
  auto ArrayOfPtr = LLVM::LLVMArrayType::get(
      PtrTy, static_cast<unsigned>(Reductions.size()));
  llvm::SmallVector<Value> InnerRedPtrs(Reductions.size());
  for (size_t k = 0; k < Reductions.size(); ++k) {
    Value IdxK = LLVM::ConstantOp::create(
        B, Loc, IntegerType::get(Ctx, 64), B.getI64IntegerAttr((int64_t)k));
    Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, State,
                                    ValueRange{IdxK});
    InnerRedPtrs[k] = LLVM::LoadOp::create(B, Loc, PtrTy, Gep);
  }

  // Clone externals first so their results are available.
  IRMapping Mapping;
  Mapping.map(IV, InnerIV);
  for (Operation *Ext : ExternsToClone) B.clone(*Ext, Mapping);

  // Clone the body. For reduction ops, insert a runtime reduce call using
  // the slot pointer from state. Everything else clones normally.
  auto Reduce = getOrInsertRTDecl(B, Module, "matlab_reduce_add_f64", VoidTy,
                                  {PtrTy, F64});
  for (Operation &Op : BodyBlock) {
    if (isMatlabOp(&Op, "matlab.yield")) continue;
    if (ReductionBodyOps.count(&Op)) {
      if (isMatlabOp(&Op, "matlab.store")) {
        // This is the store terminator of a reduction — emit the call here.
        for (size_t k = 0; k < Reductions.size(); ++k) {
          if (Reductions[k].StoreOp != &Op) continue;
          // Resolve Rhs in the cloned context.
          Value ClonedRhs = Mapping.lookupOrDefault(Reductions[k].Rhs);
          LLVM::CallOp::create(B, Loc, Reduce,
                               ValueRange{InnerRedPtrs[k], ClonedRhs});
          break;
        }
      }
      // Skip cloning load/add/store — they're replaced by the call above.
      continue;
    }
    B.clone(Op, Mapping);
  }
  LLVM::ReturnOp::create(B, Loc, ValueRange{});

  // --- Convert reduction slots to llvm.alloca / llvm.load/store ----------
  // For each reduction var, the outer matlab.alloc becomes an llvm.alloca
  // of f64 and every external (non-body) load/store on it becomes
  // llvm.load / llvm.store. The reduction chain inside the body was removed
  // above, so the only remaining uses are outside the parfor.
  for (auto &R : Reductions) {
    Operation *Alloc = R.AllocOp;
    // Put the alloca where the matlab.alloc was.
    B.setInsertionPoint(Alloc);
    Value One = LLVM::ConstantOp::create(
        B, Alloc->getLoc(), IntegerType::get(Ctx, 64),
        B.getI64IntegerAttr(1));
    Value NewPtr = LLVM::AllocaOp::create(
        B, Alloc->getLoc(), PtrTy, F64, One, /*alignment=*/0);

    // Rewrite all remaining uses of the old slot value.
    Value OldSlot = Alloc->getResult(0);
    llvm::SmallVector<Operation *, 4> UsersToErase;
    for (OpOperand &Use : llvm::make_early_inc_range(OldSlot.getUses())) {
      Operation *U = Use.getOwner();
      if (isMatlabOp(U, "matlab.load") && U->getNumOperands() == 1) {
        B.setInsertionPoint(U);
        Value V = LLVM::LoadOp::create(B, U->getLoc(), F64, NewPtr);
        U->getResult(0).replaceAllUsesWith(V);
        UsersToErase.push_back(U);
      } else if (isMatlabOp(U, "matlab.store") && U->getNumOperands() == 2 &&
                 U->getOperand(1) == OldSlot) {
        B.setInsertionPoint(U);
        LLVM::StoreOp::create(B, U->getLoc(), U->getOperand(0), NewPtr);
        UsersToErase.push_back(U);
      }
    }
    for (Operation *U : UsersToErase) U->erase();
    Alloc->erase();
    // Point the Reduction record at the new ptr def for state building.
    R.AllocOp = NewPtr.getDefiningOp();
  }

  // --- Replace the parfor op --------------------------------------------
  B.setInsertionPoint(Parfor);
  Value StepV = Step ? Step
                     : static_cast<Value>(arith::ConstantOp::create(
                           B, Loc, F64, B.getF64FloatAttr(1.0)));
  Value FnPtr = LLVM::AddressOfOp::create(B, Loc, PtrTy, Fn.getName());

  // Build the state array on the stack and store each reduction pointer.
  Value StateArg;
  if (Reductions.empty()) {
    StateArg = LLVM::ZeroOp::create(B, Loc, PtrTy);
  } else {
    Value One = LLVM::ConstantOp::create(
        B, Loc, IntegerType::get(Ctx, 64), B.getI64IntegerAttr(1));
    StateArg = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrayOfPtr, One,
                                      /*alignment=*/0);
    for (size_t k = 0; k < Reductions.size(); ++k) {
      Value IdxK = LLVM::ConstantOp::create(
          B, Loc, IntegerType::get(Ctx, 64), B.getI64IntegerAttr((int64_t)k));
      Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateArg,
                                      ValueRange{IdxK});
      LLVM::StoreOp::create(B, Loc, Reductions[k].AllocOp->getResult(0), Gep);
    }
  }

  auto Dispatch = getOrInsertRTDecl(
      B, Module, "matlab_parfor_dispatch", VoidTy,
      {F64, F64, F64, PtrTy, PtrTy});
  LLVM::CallOp::create(B, Loc, Dispatch,
                       ValueRange{Start, StepV, End, FnPtr, StateArg});

  Parfor->erase();
  // If the matlab.range result is now unused, drop it to keep the IR tidy.
  if (Iter.use_empty() && Range) Range->erase();
  return true;
}

} // namespace

unsigned runOutlineParfor(ModuleOp M) {
  llvm::SmallVector<Operation *> Parfors;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.parfor")) Parfors.push_back(Op);
  });
  unsigned Outlined = 0;
  for (Operation *P : Parfors) {
    if (outlineParfor(P, Outlined)) ++Outlined;
  }

  // Sweep dead matlab.alloc / matlab.range ops now orphaned by the redirect.
  // Iterate to fixpoint so chains drop together.
  for (;;) {
    llvm::SmallVector<Operation *, 8> Dead;
    M.walk([&](Operation *Op) {
      if ((isMatlabOp(Op, "matlab.alloc") ||
           isMatlabOp(Op, "matlab.range")) &&
          Op->use_empty())
        Dead.push_back(Op);
    });
    if (Dead.empty()) break;
    for (Operation *D : Dead) D->erase();
  }
  return Outlined;
}

} // namespace mlirgen
} // namespace matlab
