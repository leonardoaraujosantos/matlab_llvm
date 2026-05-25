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

  // Extract (start, step, end) from the iter producer feeding the parfor.
  // Two shapes (see LowerSeqLoops::extractRange for details — same story):
  //   (a) matlab.range(start[, step], end) — pre-LowerTensorOps form.
  //   (b) llvm.call @matlab_range(start, step, end) -> ptr — the form
  //       that survives when LowerTensorOps ran earlier (issue #36).
  if (Parfor->getNumOperands() != 1) return false;
  Value Iter = Parfor->getOperand(0);
  Operation *Range = Iter.getDefiningOp();
  Value Start, Step, End;
  if (isMatlabOp(Range, "matlab.range")) {
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
  } else if (auto Call = dyn_cast<LLVM::CallOp>(Range)) {
    auto Callee = Call.getCallee();
    if (!Callee || *Callee != "matlab_range") {
      std::cerr << "parfor: iter operand is not a matlab.range — skipping\n";
      return false;
    }
    if (Call.getNumOperands() != 3) return false;
    Start = Call.getOperand(0);
    Step  = Call.getOperand(1);
    End   = Call.getOperand(2);
  } else {
    std::cerr << "parfor: iter operand is not a matlab.range — skipping\n";
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
    // Walk the parfor region transitively — nested for/while loops can
    // also reference the IV slot (e.g. `for px = 1:N; ... = ... + py * px`
    // where `py` is the parfor IV).
    Region &PR = Parfor->getRegion(0);
    llvm::SmallVector<Operation *, 4> StoresToErase;
    PR.walk([&](Operation *Op) {
      if (isMatlabOp(Op, "matlab.load") && Op->getNumOperands() == 1) {
        Operation *SlotDef = Op->getOperand(0).getDefiningOp();
        if (isMatlabOp(SlotDef, "matlab.alloc")) {
          auto NameA = SlotDef->getAttrOfType<StringAttr>("name");
          if (NameA && NameA.getValue() == VarName) {
            // RAUW unconditionally. The load's result type may be `none`
            // (Sema couldn't infer) while IV is f64 — that's fine because
            // the downstream consumer is an unregistered matlab.call_builtin
            // which doesn't enforce type equality on operands, and once the
            // disp is rewritten by LowerIO the signature matches the f64
            // runtime entry point exactly.
            Op->getResult(0).replaceAllUsesWith(IV);
          }
        }
      }
    });
    PR.walk([&](Operation *Op) {
      if (isMatlabOp(Op, "matlab.store") && Op->getNumOperands() == 2) {
        Operation *SlotDef = Op->getOperand(1).getDefiningOp();
        if (isMatlabOp(SlotDef, "matlab.alloc")) {
          auto NameA = SlotDef->getAttrOfType<StringAttr>("name");
          if (NameA && NameA.getValue() == VarName)
            StoresToErase.push_back(Op);
        }
      }
    });
    for (Operation *S : StoresToErase) S->erase();
    llvm::SmallVector<Operation *, 4> DeadLoads;
    PR.walk([&](Operation *Op) {
      if (isMatlabOp(Op, "matlab.load") && Op->use_empty())
        DeadLoads.push_back(Op);
    });
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

  // --- Iteration-local outer-slot detection (issue #20 follow-on) ---------
  // The MATLAB frontend allocates every named variable as a `matlab.alloc`
  // in the enclosing scope, even for purely-body-local intermediates like
  // `cim = ...; total = total + cim;` inside a parfor (and any helper
  // slots a nested for loop introduces, e.g. an inner `for px = 1:N`
  // adds its own px slot).  Such slots are NOT captures (they're written
  // in the body) and NOT reductions, but their alloc lives outside the
  // body — so the operand-check loop below would reject them.  Identify
  // them here and arrange to clone the alloc into the outlined function
  // entry so each iteration has its own private slot.
  Region &ParforRegion = Parfor->getRegion(0);
  llvm::SmallVector<Operation *> IterLocalAllocs;
  llvm::DenseSet<Value> IterLocalSlots;
  {
    llvm::DenseSet<Value> CandidateSlots;
    ParforRegion.walk([&](Operation *Op) {
      if (!isMatlabOp(Op, "matlab.store") || Op->getNumOperands() != 2)
        return;
      Value Slot = Op->getOperand(1);
      Operation *SlotDef = Slot.getDefiningOp();
      if (!isMatlabOp(SlotDef, "matlab.alloc")) return;
      // Skip allocs that already live anywhere inside the parfor region.
      if (ParforRegion.isAncestor(SlotDef->getParentRegion())) return;
      CandidateSlots.insert(Slot);
    });
    for (Value Slot : CandidateSlots) {
      // Iteration-local iff every use of the slot lives somewhere inside
      // the parfor region (transitively — nested for / if / while are
      // fine).
      bool AllInRegion = true;
      for (OpOperand &Use : Slot.getUses()) {
        Region *UR = Use.getOwner()->getParentRegion();
        if (!UR || !ParforRegion.isAncestor(UR)) {
          AllInRegion = false;
          break;
        }
      }
      if (!AllInRegion) continue;
      IterLocalAllocs.push_back(Slot.getDefiningOp());
      IterLocalSlots.insert(Slot);
    }
  }

  // --- Capture analysis (now excluding reduction chains) ------------------
  // Walk transitively so values produced inside nested for/while loops
  // count as defined-inside the parfor region.
  llvm::DenseSet<Value> DefinedInside;
  DefinedInside.insert(IV);
  ParforRegion.walk([&](Operation *Op) {
    for (Value R : Op->getResults()) DefinedInside.insert(R);
    for (Region &R : Op->getRegions())
      for (Block &Blk : R)
        for (BlockArgument BA : Blk.getArguments())
          DefinedInside.insert(BA);
  });
  // The reduction slot values are consumed by the reduction chain itself;
  // their replacement (a ptr loaded from state) will be created later. For
  // the purpose of capture analysis they are allowed.
  llvm::DenseSet<Value> ReductionSlots;
  for (auto &R : Reductions) ReductionSlots.insert(R.AllocOp->getResult(0));

  // --- Read-only scalar captures (issue #20, Phase 2) ---------------------
  // For each matlab.load(%outer_slot) inside the body whose slot is defined
  // outside the parfor and is NOT a reduction slot, capture the slot's
  // value once at the call site and pass it through the state[] array.
  // The body's loads get RAUW'd to the state-loaded value via IRMapping.
  //
  // Scope (this PR): scalar f64 captures only.  Matrix (ptr / tensor)
  // captures require running OutlineParfor after LowerTensorOps so the
  // slot is already `ptr`-typed — pass-ordering rework is the next slice
  // of the issue.  See PR description for the documented carve-out.
  //
  // A slot qualifies as a Phase-2 capture only if the body NEVER stores
  // to it.  Slots written from inside the body are either reductions
  // (handled by ReductionBodyOps above) or write-by-row patterns
  // (deferred — pragma support is the next slice of issue #20).
  // Excluding written-to slots here keeps a reduction the detector
  // missed from being silently mis-treated as a read-only capture;
  // the existing "body captures value of unsupported defining op" path
  // will surface it as before.
  struct Capture {
    Value Slot;                                     // outer matlab.alloc
    Type CaptureType;                               // f64 today
    llvm::SmallVector<Operation *, 2> Loads;        // matlab.load ops in body
  };
  // Collect stores transitively — nested for/while loops inside the body
  // contribute stores too (their own IV slot writes, intermediate locals).
  llvm::DenseSet<Value> SlotsWrittenInBody;
  ParforRegion.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.store") && Op->getNumOperands() == 2)
      SlotsWrittenInBody.insert(Op->getOperand(1));
  });
  llvm::SmallVector<Capture> Captures;
  llvm::DenseMap<Value, size_t> SlotToCaptureIdx;
  llvm::DenseSet<Operation *> CaptureLoadOps;
  llvm::DenseSet<Value> CaptureSlots;
  // Walk the parfor region transitively so nested-for loads of outer slots
  // (e.g. `for px = 1:W` capturing W from outside) are surfaced too.
  ParforRegion.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.load") || Op->getNumOperands() != 1) return;
    if (ReductionBodyOps.count(Op)) return;
    Value Slot = Op->getOperand(0);
    Operation *SlotDef = Slot.getDefiningOp();
    if (!isMatlabOp(SlotDef, "matlab.alloc")) return;
    // Inner slot (defined anywhere inside the parfor region) — handled
    // separately (reduction / IterLocal / body-defined SSA).
    if (ParforRegion.isAncestor(SlotDef->getParentRegion())) return;
    if (ReductionSlots.count(Slot)) return;
    if (IterLocalSlots.count(Slot)) return;
    if (SlotsWrittenInBody.count(Slot)) return;
    Type T = Op->getResult(0).getType();
    // Capture-supported types.  v1 (45ed8fb) was f64-only.  #20-A widens to
    // every scalar integer/float type that fits in a ptr-sized state[]
    // slot — they're stored at their natural LLVM width into the 8-byte
    // ptr-array element.  #20-B accepts tensor types (matrix capture) via
    // an unrealized_conversion_cast to/from PtrTy bracketing the state[]
    // slot store/load — LowerTensorOps resolves the cast when it converts
    // the tensor to a matlab_mat * carrier.
    //
    // Safety note for tensor captures: the SlotsWrittenInBody check above
    // guarantees the captured *slot* isn't reassigned in the body.  Writes
    // through the captured pointer (e.g. `A(i,j) = x` on a shared matrix)
    // are the user's responsibility — same contract as MATLAB's parfor.
    bool ScalarOk = (mlir::isa<FloatType>(T) || mlir::isa<IntegerType>(T)) &&
                    T.getIntOrFloatBitWidth() <= 64;
    bool PtrOk    = mlir::isa<LLVM::LLVMPointerType>(T);
    bool TensorOk = mlir::isa<RankedTensorType, UnrankedTensorType>(T);
    if (!ScalarOk && !PtrOk && !TensorOk) return;
    auto It = SlotToCaptureIdx.find(Slot);
    if (It == SlotToCaptureIdx.end()) {
      SlotToCaptureIdx[Slot] = Captures.size();
      Captures.push_back({Slot, T, {Op}});
    } else {
      Captures[It->second].Loads.push_back(Op);
    }
    CaptureLoadOps.insert(Op);
    CaptureSlots.insert(Slot);
  });

  llvm::SmallVector<Operation *> ExternsToClone;
  llvm::DenseSet<Operation *> ExternSet;
  bool RejectCapture = false;
  // Walk operands transitively too so the rejection check sees nested-for
  // loads of outer values.
  ParforRegion.walk([&](Operation *Op) {
    if (RejectCapture) return;
    if (ReductionBodyOps.count(Op)) return; // will be replaced
    if (CaptureLoadOps.count(Op)) return;   // will be replaced via state[]
    for (Value Operand : Op->getOperands()) {
      if (DefinedInside.count(Operand)) continue;
      if (ReductionSlots.count(Operand)) continue;
      if (CaptureSlots.count(Operand)) continue;  /* handled via state[] */
      if (IterLocalSlots.count(Operand))
        continue;  /* alloc cloned inside the outlined function */
      // Block args of the parfor region itself (e.g. the IV / a nested-
      // for's IV) are produced inside the region; treat as defined-inside.
      if (auto BA = mlir::dyn_cast<BlockArgument>(Operand)) {
        if (ParforRegion.isAncestor(BA.getOwner()->getParent())) continue;
        std::cerr << "parfor: body captures a block argument from outside — "
                     "not supported\n";
        RejectCapture = true;
        return;
      }
      Operation *Def = Operand.getDefiningOp();
      if (!Def) {
        std::cerr << "parfor: body captures a block argument from outside — "
                     "not supported\n";
        RejectCapture = true;
        return;
      }
      if (!isCloneableExternal(Def)) {
        std::cerr << "parfor: body captures value of unsupported defining op '"
                  << Def->getName().getStringRef().str() << "'\n";
        RejectCapture = true;
        return;
      }
      if (ExternSet.insert(Def).second) ExternsToClone.push_back(Def);
    }
  });
  if (RejectCapture) return false;

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
  // an array of `ptr`-sized slots.  Layout:
  //   state[0..n_red-1]              : reduction-variable pointers
  //   state[n_red..n_red+n_cap-1]    : captured scalar values (Phase 2)
  // For k-th reduction: `gep ptr, [k]`, then `load ptr`.
  size_t NRed = Reductions.size();
  size_t NCap = Captures.size();
  auto ArrayOfPtr = LLVM::LLVMArrayType::get(
      PtrTy, static_cast<unsigned>(NRed + NCap));
  llvm::SmallVector<Value> InnerRedPtrs(NRed);
  for (size_t k = 0; k < NRed; ++k) {
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
  // Clone iteration-local matlab.alloc ops into the outlined entry so
  // each thread gets its own slot.  The original outer alloc is still
  // emitted but ends up dead (its only uses were inside the parfor body,
  // which we're erasing); LowerScalarSlots / LowerTensorOps will sweep
  // it on a later pass iteration.
  for (Operation *A : IterLocalAllocs) B.clone(*A, Mapping);

  // Load each capture from state[NRed + k] and map every original in-body
  // matlab.load result to it.
  //
  // For scalar / ptr captures the load is directly typed (state[] slot's
  // 8-byte width fits any scalar ≤ 64 bits).  For tensor types (#20-B
  // matrix capture) we always load PtrTy from state[] and map the body's
  // matlab.load result (typed tensor) to the PtrTy value — matlab.subscript
  // and friends are operand-type-generic in their signatures, so the cloned
  // body verifies, and LowerTensorOps' subscript-rewrite (which gates on
  // operand 0 being PtrTy) then fires as it would in the non-parfor flow
  // after matlab.alloc retypes to llvm.alloca.
  for (size_t k = 0; k < NCap; ++k) {
    Value IdxK = LLVM::ConstantOp::create(
        B, Loc, IntegerType::get(Ctx, 64),
        B.getI64IntegerAttr((int64_t)(NRed + k)));
    Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, State,
                                    ValueRange{IdxK});
    Type CapTy = Captures[k].CaptureType;
    bool CapTyIsLLVM = mlir::isa<LLVM::LLVMPointerType>(CapTy) ||
                       ((mlir::isa<FloatType>(CapTy) ||
                         mlir::isa<IntegerType>(CapTy)) &&
                        CapTy.getIntOrFloatBitWidth() <= 64);
    Value Loaded;
    if (CapTyIsLLVM) {
      Loaded = LLVM::LoadOp::create(B, Loc, CapTy, Gep);
    } else {
      Loaded = LLVM::LoadOp::create(B, Loc, PtrTy, Gep);
    }
    for (Operation *L : Captures[k].Loads)
      Mapping.map(L->getResult(0), Loaded);
  }

  // Clone the body. For reduction ops, insert a runtime reduce call using
  // the slot pointer from state. Everything else clones normally.
  auto Reduce = getOrInsertRTDecl(B, Module, "matlab_reduce_add_f64", VoidTy,
                                  {PtrTy, F64});
  for (Operation &Op : BodyBlock) {
    if (isMatlabOp(&Op, "matlab.yield")) continue;
    if (CaptureLoadOps.count(&Op)) continue;  /* replaced by state-load */
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

  // Build the state array on the stack:
  //   state[0..NRed-1]              : reduction-variable pointers
  //   state[NRed..NRed+NCap-1]      : captured scalar values (Phase 2)
  Value StateArg;
  if (NRed == 0 && NCap == 0) {
    StateArg = LLVM::ZeroOp::create(B, Loc, PtrTy);
  } else {
    Value One = LLVM::ConstantOp::create(
        B, Loc, IntegerType::get(Ctx, 64), B.getI64IntegerAttr(1));
    StateArg = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrayOfPtr, One,
                                      /*alignment=*/0);
    for (size_t k = 0; k < NRed; ++k) {
      Value IdxK = LLVM::ConstantOp::create(
          B, Loc, IntegerType::get(Ctx, 64), B.getI64IntegerAttr((int64_t)k));
      Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateArg,
                                      ValueRange{IdxK});
      LLVM::StoreOp::create(B, Loc, Reductions[k].AllocOp->getResult(0), Gep);
    }
    /* Capture values: emit one matlab.load per unique outer slot right
     * before the dispatch, then llvm.store into state[NRed+k].  matlab.load
     * here remains for downstream LowerScalarSlots / LowerTensorOps to
     * convert into a typed LLVM load against the slot's final lowering.
     *
     * Non-LLVM types (tensor — #20-B matrix capture) are cast to PtrTy via
     * unrealized_conversion_cast before storing into the ptr-sized state[]
     * slot; LowerTensorOps later resolves the cast when the tensor becomes
     * a matlab_mat *. */
    for (size_t k = 0; k < NCap; ++k) {
      Value Slot = Captures[k].Slot;
      Type CapTy = Captures[k].CaptureType;
      OperationState LoadState(Loc, "matlab.load");
      LoadState.addOperands(ValueRange{Slot});
      LoadState.addTypes(TypeRange{CapTy});
      Operation *LoadOp = B.create(LoadState);
      Value CapVal = LoadOp->getResult(0);
      bool CapTyIsLLVM = mlir::isa<LLVM::LLVMPointerType>(CapTy) ||
                         ((mlir::isa<FloatType>(CapTy) ||
                           mlir::isa<IntegerType>(CapTy)) &&
                          CapTy.getIntOrFloatBitWidth() <= 64);
      if (!CapTyIsLLVM) {
        CapVal = UnrealizedConversionCastOp::create(B, Loc, PtrTy, CapVal)
                     .getResult(0);
      }
      Value IdxK = LLVM::ConstantOp::create(
          B, Loc, IntegerType::get(Ctx, 64),
          B.getI64IntegerAttr((int64_t)(NRed + k)));
      Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateArg,
                                      ValueRange{IdxK});
      LLVM::StoreOp::create(B, Loc, CapVal, Gep);
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
