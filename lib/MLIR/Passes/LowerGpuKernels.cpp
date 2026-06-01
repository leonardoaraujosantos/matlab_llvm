// lib/MLIR/Passes/LowerGpuKernels.cpp — lower matlab.gpu.kernel ops.
//
// Two lanes selected by the MATLAB_GPU_OUTLINE env var:
//
//   (default)       Simple rewrite to `matlab.for`.  The kernel body
//                   continues to run in-place on the host (the CPU-debug
//                   lane).  All slot captures — output arrays, outer
//                   scalars, kernel-local temporaries — are visible
//                   via the existing matlab.alloc / matlab.load /
//                   matlab.store machinery.  Numerically correct for
//                   every shape `coder.gpu.kernelfun` can be applied to.
//
//   MATLAB_GPU_OUTLINE=1
//                   Real array-capture outline (issue #24).  Split into
//                   two phases around LowerTensorOps:
//
//                     EARLY (runOutlineGpuKernels, before LowerSeqLoops):
//                       classify each kernel's outer captures on the
//                       pre-lowering matlab.* IR.  A kernel is CLAIMED
//                       when every capture's defining op is a
//                       `matlab.alloc` slot (output array OR scalar) or a
//                       cloneable external (constant / addressof) — i.e.
//                       no raw outer block-argument operands.  Claimed
//                       kernels get the induction-variable slot folded
//                       away (loads → block arg, stores erased), are
//                       tagged `matlab.gpu.outline`, and LEFT IN PLACE so
//                       the downstream LowerSeqLoops / LowerTensorOps
//                       passes lower their body in situ (those passes
//                       Mod.walk into the kernel region).  Unclaimed
//                       kernels — and every kernel on the default lane —
//                       are rewritten to `matlab.for` immediately.
//
//                     LATE (runOutlineGpuKernelsLate, after LowerTensorOps):
//                       the claimed kernel's body is now plain
//                       ptr/arith/scf/runtime-call IR — output arrays are
//                       `!llvm.ptr` to `matlab_mat`, scalar slots are
//                       `llvm.alloca`.  Lift the body into an
//                       `llvm.func void(f64 iv, ptr state)`: every
//                       captured pointer slot and scalar value is packed
//                       into a `state` array of `!llvm.ptr` and reloaded
//                       at the top of the outlined function.  No
//                       `unrealized_conversion_cast` is needed — the
//                       capture is already a pointer.  This is the
//                       canonical source the Tier-2/3/4 EmitMetal/CUDA/
//                       OpenCL passes walk to print kernel source.
//
// **Why the split**.  The pre-lowering outliner had to cast captured
// tensors to `ptr` and back, and that `unrealized_conversion_cast`
// defeated LowerTensorOps's operand matching (`__subscript_store`
// requires a `ptr` base).  Running the lift AFTER LowerTensorOps removes
// the cast entirely.
//
// A `coder.gpu.kernelfun`-tagged body references three classes of
// outer slot, each handled by the LATE lift:
//   1. Output / input arrays (`matlab.alloc` → `ptr` to `matlab_mat`):
//      the shared, written-across-iterations result.  Passed through
//      `state` as a shared pointer.
//   2. Outer scalars set BEFORE the kernel and only read inside
//      (`re_min = -2.0;`): cloned per-invocation and seeded from the
//      outer value, so the read-only value is preserved.
//   3. Kernel-local temporaries first WRITTEN inside (`cr = ...;`):
//      logically per-iteration.  Also cloned per-invocation (the seed
//      from its outer slot is harmless — it is overwritten before read).
// Cloning every scalar slot + seeding from the outer value is correct on
// BOTH the sequential CPU dispatch lane (matlab_gpu_launch_kernel's host
// fallback) AND a truly-parallel device backend, so no definite-
// assignment analysis is needed to split bucket-2 from bucket-3.
//
// The kernel-analysis warning (T7.5) runs regardless of lane.

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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdlib>
#include <cstring>
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

/* Attribute marking a kernel CLAIMED by the EARLY phase for the LATE
 * lift.  Left in place across LowerSeqLoops / LowerTensorOps. */
constexpr llvm::StringLiteral kOutlineTag = "matlab.gpu.outline";

bool isCloneableExternal(Operation *Op) {
  if (!Op) return false;
  if (isa<arith::ConstantOp>(Op)) return true;
  if (isa<LLVM::ConstantOp>(Op)) return true;
  if (isa<LLVM::ZeroOp>(Op)) return true;
  if (isa<LLVM::AddressOfOp>(Op)) return true;
  return false;
}

void rewriteToMatlabFor(Operation *K, OpBuilder &B) {
  B.setInsertionPoint(K);
  llvm::SmallVector<Value, 2> Operands(K->getOperands());
  llvm::SmallVector<NamedAttribute> Attrs(K->getAttrs());
  Attrs.emplace_back(StringAttr::get(K->getContext(), "matlab.gpu.target"),
                     StringAttr::get(K->getContext(), "cpu"));
  OperationState State(K->getLoc(), "matlab.for");
  State.addOperands(Operands);
  State.addAttributes(Attrs);
  State.addRegion();
  Operation *NewFor = Operation::create(State);
  NewFor->getRegion(0).takeBody(K->getRegion(0));
  B.insert(NewFor);
  K->erase();
}

/* Fold the induction-variable slot inside the kernel body: replace
 * every `matlab.load` of the `var` slot with the block-argument IV and
 * erase the matching `matlab.store`s.  Runs on the pre-lowering matlab.*
 * IR (EARLY phase) so the IV slot never reaches the lowered body and
 * isn't mistaken for a captured scalar slot in the LATE phase.  Safe to
 * run only on a kernel the early analysis has already CLAIMED. */
void foldInductionVarSlot(Operation *K) {
  StringRef VarName;
  if (auto VA = K->getAttrOfType<StringAttr>("var")) VarName = VA.getValue();
  if (VarName.empty()) return;
  Region &Body = K->getRegion(0);
  Value IV = Body.front().getArgument(0);

  auto slotName = [&](Value Slot) -> StringRef {
    Operation *D = Slot.getDefiningOp();
    if (!isMatlabOp(D, "matlab.alloc")) return {};
    if (auto NA = D->getAttrOfType<StringAttr>("name")) return NA.getValue();
    return {};
  };

  // Walk the whole region (the IV-slot may be read inside nested loops,
  // e.g. `cr = ...(i-1)...` under an inner for-j) so every reference is
  // replaced, not just those in the immediate body block.
  Body.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.load") && Op->getNumOperands() == 1 &&
        slotName(Op->getOperand(0)) == VarName)
      Op->getResult(0).replaceAllUsesWith(IV);
  });

  llvm::SmallVector<Operation *, 4> Dead;
  Body.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.store") && Op->getNumOperands() == 2 &&
        slotName(Op->getOperand(1)) == VarName)
      Dead.push_back(Op);
  });
  Body.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.load") && Op->use_empty())
      Dead.push_back(Op);
  });
  for (Operation *D : Dead) D->erase();
}

/* EARLY-phase analysis (read-only).  Decide whether the real outliner
 * can CLAIM this kernel for the LATE lift.
 *
 * The LATE lift runs after LowerTensorOps and clones the kernel body
 * into an llvm.func.  The lowering passes DO descend into the kernel
 * region for control flow (an inner matlab.for/while lowers to scf in
 * place) but NOT for scalar-slot promotion, so by the late point the
 * body holds plain ptr/arith/scf/runtime-call ops plus still-`matlab.*`
 * scalar slot load/stores — both of which the lift handles:
 *   - ARRAY captures are `matlab.alloc` slots that lowered to a
 *     `ptr`-to-matlab_mat; passed through state as shared pointers
 *     (no tensor↔ptr cast);
 *   - SCALAR captures are still-`matlab.alloc` f64 slots; cloned
 *     per-invocation in the outlined function and seeded from their
 *     outer value (see outlineLowered);
 *   - cloneable externals (constants) are cloned in.
 *
 * A kernel is therefore CLAIMABLE when it has the canonical range +
 * single-f64-IV shape and every cross-region operand is a `matlab.alloc`
 * slot, the folded IV slot, or a cloneable external.  It DECLINES (and
 * falls back to the numerically-correct `matlab.for` CPU rewrite) on a
 * raw outer block-argument capture or a still-nested `matlab.gpu.kernel`
 * (e.g. the inner per-row kernel of a `coder.gpu.kernelfun` nest, which
 * captures the outer kernel's induction argument — it is declined and
 * rewritten before the outer kernel is processed).
 *
 * NOTE on parallelism: cloning every scalar slot per-invocation and
 * seeding from the outer value is correct on the sequential CPU dispatch
 * lane AND on a truly-parallel backend (a bucket-3 per-iteration temp
 * gets private storage and is overwritten before read; a bucket-2 outer
 * read-only value is preserved) — so no definite-assignment analysis is
 * required.  Makes NO mutations. */
bool kernelIsClaimable(Operation *K) {
  MLIRContext *Ctx = K->getContext();
  auto F64 = Float64Type::get(Ctx);

  if (K->getNumOperands() < 1) return false;
  Operation *Range = K->getOperand(0).getDefiningOp();
  if (!isMatlabOp(Range, "matlab.range")) return false;
  unsigned RN = Range->getNumOperands();
  if (RN != 2 && RN != 3) return false;
  for (Value RO : Range->getOperands())
    if (RO.getType() != F64) return false;

  if (K->getNumRegions() != 1) return false;
  Region &Body = K->getRegion(0);
  if (!Body.hasOneBlock()) return false;
  Block &BodyBlock = Body.front();
  if (BodyBlock.getNumArguments() != 1 ||
      BodyBlock.getArgument(0).getType() != F64)
    return false;

  // The induction-variable slot is folded away in the EARLY phase, so
  // ignore loads/stores against it during the claim check.
  StringRef VarName;
  if (auto VA = K->getAttrOfType<StringAttr>("var")) VarName = VA.getValue();
  auto slotName = [&](Value V) -> StringRef {
    Operation *D = V.getDefiningOp();
    if (!isMatlabOp(D, "matlab.alloc")) return {};
    if (auto NA = D->getAttrOfType<StringAttr>("name")) return NA.getValue();
    return {};
  };
  auto isVarSlot = [&](Value V) -> bool {
    return !VarName.empty() && slotName(V) == VarName;
  };

  // Everything defined inside the kernel region: the entry IV, every op
  // result, and every NESTED block argument (inner matlab.for / while
  // induction + iter args) — so a use of one isn't mistaken for an outer
  // capture.
  llvm::DenseSet<Value> DefinedInside;
  DefinedInside.insert(BodyBlock.getArgument(0));
  Body.walk([&](Operation *Op) {
    for (Value R : Op->getResults()) DefinedInside.insert(R);
    for (Region &Rg : Op->getRegions())
      for (Block &Bl : Rg)
        for (Value A : Bl.getArguments()) DefinedInside.insert(A);
  });

  bool Ok = true;
  Body.walk([&](Operation *Op) {
    if (!Ok) return;
    // A still-nested kernel (should already be rewritten by the time the
    // enclosing kernel is processed) can't be lifted — decline.
    if (isMatlabOp(Op, "matlab.gpu.kernel")) {
      Ok = false;
      return;
    }
    for (Value Operand : Op->getOperands()) {
      if (DefinedInside.count(Operand)) continue;
      Operation *Def = Operand.getDefiningOp();
      if (!Def) { Ok = false; return; }            // genuine outer block arg
      if (isCloneableExternal(Def)) continue;
      if (isVarSlot(Operand)) continue;            // folded away early
      // Any matlab.alloc slot is liftable: tensor/array slots pass through
      // state as shared pointers; scalar slots are cloned per-invocation
      // and initialised from their outer value.
      if (isMatlabOp(Def, "matlab.alloc")) continue;
      Ok = false;                                  // non-slot outer SSA — decline
    }
  });
  return Ok;
}

/* EARLY-phase claim.  Fold the induction slot, then rebuild the kernel
 * op with the loop bounds (start, step, end) appended as three explicit
 * f64 operands.  The original `matlab.range` operand is lowered to a
 * runtime call by LowerTensorOps before the LATE phase runs, so the LATE
 * lift can no longer recover the bounds from it — the appended operands
 * are plain f64 SSA values that survive lowering unchanged.  Tags the
 * rebuilt op `matlab.gpu.outline`. */
void claimKernelForLateOutline(Operation *K, OpBuilder &B) {
  MLIRContext *Ctx = K->getContext();
  auto F64 = Float64Type::get(Ctx);
  foldInductionVarSlot(K);

  Operation *Range = K->getOperand(0).getDefiningOp();
  Value Start, Step, End;
  if (Range->getNumOperands() == 3) {
    Start = Range->getOperand(0);
    Step = Range->getOperand(1);
    End = Range->getOperand(2);
  } else {
    Start = Range->getOperand(0);
    End = Range->getOperand(1);
  }
  B.setInsertionPoint(K);
  if (!Step)
    Step = arith::ConstantOp::create(B, K->getLoc(), F64,
                                     B.getF64FloatAttr(1.0));

  llvm::SmallVector<Value> Operands(K->getOperands());
  Operands.push_back(Start);
  Operands.push_back(Step);
  Operands.push_back(End);
  llvm::SmallVector<NamedAttribute> Attrs(K->getAttrs());
  OperationState State(K->getLoc(), "matlab.gpu.kernel");
  State.addOperands(Operands);
  State.addAttributes(Attrs);
  State.addRegion();
  Operation *NewK = Operation::create(State);
  NewK->setAttr(kOutlineTag, UnitAttr::get(Ctx));
  NewK->getRegion(0).takeBody(K->getRegion(0));
  B.insert(NewK);
  K->erase();
}

/* LATE-phase lift.  The kernel body is now plain ptr/arith/scf/runtime-
 * call IR.  Outline it into `llvm.func void(f64 iv, ptr state)`, packing
 * every captured value into a `state` array of `!llvm.ptr`:
 *   - pointer captures (output arrays, scalar slots that survived as
 *     `llvm.alloca`) go into the slot directly;
 *   - non-pointer captures (a scalar SSA value) are spilled to a fresh
 *     stack slot whose address goes into `state`, and reloaded inside.
 * Cloneable externals are cloned into the outlined entry block.  Returns
 * false (leaving the kernel for the simple rewrite) only on an
 * unexpected shape the EARLY claim should already have excluded. */
bool outlineLowered(Operation *K, unsigned KernelId) {
  Location Loc = K->getLoc();
  MLIRContext *Ctx = K->getContext();
  ModuleOp Module = K->getParentOfType<ModuleOp>();
  OpBuilder B(Ctx);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto VoidTy = LLVM::LLVMVoidType::get(Ctx);
  auto I32 = IntegerType::get(Ctx, 32);
  auto I64 = IntegerType::get(Ctx, 64);

  // claimKernelForLateOutline appended (start, step, end) as the last
  // three operands.  Operand 0 is the original range producer (now a
  // lowered runtime call) and is ignored.
  unsigned NOps = K->getNumOperands();
  if (NOps < 4) return false;  // claimKernelForLateOutline appends 3 bounds
  Value Start = K->getOperand(NOps - 3);
  Value Step = K->getOperand(NOps - 2);
  Value End = K->getOperand(NOps - 1);

  Region &Body = K->getRegion(0);
  if (!Body.hasOneBlock()) return false;
  Block &BodyBlock = Body.front();
  Value IV = BodyBlock.getArgument(0);

  // Collect captures (values defined outside the region, used inside),
  // in a deterministic discovery order, plus the cloneable externals.
  // DefinedInside spans the entry IV, all op results, and all nested
  // block arguments (inner loop induction / iter args).
  llvm::DenseSet<Value> DefinedInside;
  DefinedInside.insert(IV);
  Body.walk([&](Operation *Op) {
    for (Value R : Op->getResults()) DefinedInside.insert(R);
    for (Region &Rg : Op->getRegions())
      for (Block &Bl : Rg)
        for (Value A : Bl.getArguments()) DefinedInside.insert(A);
  });

  llvm::SmallVector<Operation *> ExternsToClone;
  llvm::DenseSet<Operation *> ExternSet;
  llvm::SmallVector<Value> Captures;
  llvm::DenseSet<Value> CaptureSet;
  bool Bad = false;
  Body.walk([&](Operation *Op) {
    for (Value Operand : Op->getOperands()) {
      if (DefinedInside.count(Operand)) continue;
      if (CaptureSet.count(Operand)) continue;
      Operation *Def = Operand.getDefiningOp();
      if (!Def) { Bad = true; return; }
      if (isCloneableExternal(Def)) {
        if (ExternSet.insert(Def).second) ExternsToClone.push_back(Def);
        continue;
      }
      Captures.push_back(Operand);
      CaptureSet.insert(Operand);
    }
  });
  if (Bad) return false;

  // Classify each capture.  Two liftable shapes:
  //   * ARRAY slot — already a `!llvm.ptr` (an `llvm.alloca` holding a
  //     matlab_mat*).  Shared across iterations: the slot pointer is
  //     passed through state and the body reads/writes the shared
  //     matrix through it.
  //   * SCALAR slot — a still-`matlab.alloc`-typed f64 slot (its uses
  //     live in the kernel region so the scalar-slot promotion never
  //     fired).  Cloned per-invocation into the outlined function and
  //     initialised from the slot's outer value, so a bucket-3
  //     per-iteration temporary gets private storage and a bucket-2
  //     outer read-only value is preserved — correct on both the
  //     sequential CPU lane and a truly-parallel backend, with no
  //     definite-assignment analysis required.
  auto isScalarSlot = [&](Value V) -> bool {
    return isMatlabOp(V.getDefiningOp(), "matlab.alloc") && V.getType() == F64;
  };
  for (Value Cap : Captures)
    if (Cap.getType() != PtrTy && !isScalarSlot(Cap))
      return false;  // unexpected shape — EARLY claim should exclude it

  // Build an unregistered matlab.* op at the current insertion point.
  auto makeMatOp = [&](StringRef OpName, ValueRange Ins,
                       TypeRange Outs) -> Operation * {
    OperationState St(Loc, OpName);
    St.addOperands(Ins);
    St.addTypes(Outs);
    Operation *Op = Operation::create(St);
    B.insert(Op);
    return Op;
  };

  std::string Name = ("__gpu_kernel_" + llvm::Twine(KernelId)).str();
  B.setInsertionPointToEnd(Module.getBody());
  auto FnTy = LLVM::LLVMFunctionType::get(VoidTy, {F64, PtrTy});
  auto Fn = LLVM::LLVMFuncOp::create(B, Loc, Name, FnTy);
  Fn.setLinkage(LLVM::Linkage::External);
  Block *Entry = Fn.addEntryBlock(B);
  Value InnerIV = Entry->getArgument(0);
  Value StateArg = Entry->getArgument(1);
  B.setInsertionPointToEnd(Entry);

  // Reload each capture from state[k] at the top of the outlined func.
  llvm::DenseMap<Value, Value> Remap;
  for (size_t k = 0; k < Captures.size(); ++k) {
    Value IdxK = LLVM::ConstantOp::create(B, Loc, I64,
                                          B.getI64IntegerAttr((int64_t)k));
    Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateArg,
                                    ValueRange{IdxK});
    Value Slot = LLVM::LoadOp::create(B, Loc, PtrTy, Gep);
    if (isScalarSlot(Captures[k])) {
      // Clone the matlab.alloc as a function-local scalar slot and seed
      // it from the (spilled) outer value passed through state.
      Operation *Local = B.clone(*Captures[k].getDefiningOp());
      Value OuterVal = LLVM::LoadOp::create(B, Loc, F64, Slot);
      makeMatOp("matlab.store", {OuterVal, Local->getResult(0)}, {});
      Remap[Captures[k]] = Local->getResult(0);
    } else {
      // Array slot — shared pointer.
      Remap[Captures[k]] = Slot;
    }
  }

  IRMapping Mapping;
  Mapping.map(IV, InnerIV);
  for (auto &P : Remap) Mapping.map(P.first, P.second);
  for (Operation *Ext : ExternsToClone) B.clone(*Ext, Mapping);
  for (Operation &Op : BodyBlock) {
    if (isMatlabOp(&Op, "matlab.yield")) continue;
    B.clone(Op, Mapping);
  }
  LLVM::ReturnOp::create(B, Loc, ValueRange{});

  // Build the state array + dispatch call at the original kernel site.
  B.setInsertionPoint(K);
  Value StepV = Step;  // always set by claimKernelForLateOutline
  Value FnPtr = LLVM::AddressOfOp::create(B, Loc, PtrTy, Fn.getName());

  Value StateOuter;
  if (Captures.empty()) {
    StateOuter = LLVM::ZeroOp::create(B, Loc, PtrTy);
  } else {
    auto ArrTy = LLVM::LLVMArrayType::get(
        PtrTy, static_cast<unsigned>(Captures.size()));
    Value One = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
    StateOuter = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrTy, One,
                                        /*alignment=*/0);
    for (size_t k = 0; k < Captures.size(); ++k) {
      Value Cap = Captures[k];
      Value SlotPtr;
      if (Cap.getType() == PtrTy) {
        // Array slot — pass the slot pointer itself (shared).
        SlotPtr = Cap;
      } else {
        // Scalar slot — read its current outer value and spill it to a
        // stack slot whose address goes through state; the outlined
        // function seeds its private clone from it.
        Operation *Ld = makeMatOp("matlab.load", {Cap}, {F64});
        Value OuterVal = Ld->getResult(0);
        Value One2 =
            LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
        SlotPtr = LLVM::AllocaOp::create(B, Loc, PtrTy, F64, One2,
                                         /*alignment=*/0);
        LLVM::StoreOp::create(B, Loc, OuterVal, SlotPtr);
      }
      Value IdxK = LLVM::ConstantOp::create(B, Loc, I64,
                                            B.getI64IntegerAttr((int64_t)k));
      Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateOuter,
                                      ValueRange{IdxK});
      LLVM::StoreOp::create(B, Loc, SlotPtr, Gep);
    }
  }

  Value KernelIdV = LLVM::ConstantOp::create(
      B, Loc, I32, B.getI32IntegerAttr((int32_t)KernelId));
  auto Dispatch = getOrInsertRTDecl(B, Module, "matlab_gpu_launch_kernel",
                                    VoidTy, {F64, F64, F64, PtrTy, PtrTy, I32});
  LLVM::CallOp::create(B, Loc, Dispatch,
                       ValueRange{Start, StepV, End, FnPtr, StateOuter,
                                  KernelIdV});

  K->erase();
  return true;
}

bool wantRealOutline() {
  const char *S = std::getenv("MATLAB_GPU_OUTLINE");
  return S && std::strcmp(S, "1") == 0;
}

}  // namespace

/* Kernel-analysis warnings (T7.5). */
void emitKernelAnalysisWarnings(Operation *Kernel) {
  unsigned BreakLike = 0;
  Kernel->walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.break") || isMatlabOp(Op, "matlab.continue")) {
      ++BreakLike;
    } else if (isMatlabOp(Op, "matlab.store") && Op->getNumOperands() == 2) {
      Operation *SlotDef = Op->getOperand(1).getDefiningOp();
      if (isMatlabOp(SlotDef, "matlab.alloc")) {
        if (auto NA = SlotDef->getAttrOfType<FlatSymbolRefAttr>("name")) {
          auto N = NA.getValue();
          if (N == "__did_break" || N == "__did_continue") ++BreakLike;
        } else if (auto SA = SlotDef->getAttrOfType<StringAttr>("name")) {
          auto N = SA.getValue();
          if (N == "__did_break" || N == "__did_continue") ++BreakLike;
        }
      }
    }
  });
  if (BreakLike > 0) {
    std::cerr << "warning: coder.gpu.kernel body contains break/continue — "
              << "GPU Coder UG p. 6-23 flags these as kernel-incompatible "
              << "(CPU-debug lane tolerates them; real device backends "
              << "will require a refactor).\n";
  }
}

/* EARLY phase.  On the default lane (or for any kernel the outliner
 * can't claim), rewrite `matlab.gpu.kernel` → `matlab.for` immediately.
 * Under MATLAB_GPU_OUTLINE=1, a claimable kernel instead has its
 * induction slot folded, is tagged `matlab.gpu.outline`, and is LEFT IN
 * PLACE so LowerSeqLoops / LowerTensorOps lower its body before the LATE
 * lift in runOutlineGpuKernelsLate. */
unsigned runOutlineGpuKernels(ModuleOp M) {
  llvm::SmallVector<Operation *> Kernels;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.gpu.kernel")) Kernels.push_back(Op);
  });
  unsigned Handled = 0;
  OpBuilder B(M.getContext());
  bool TryOutline = wantRealOutline();
  for (Operation *K : Kernels) {
    emitKernelAnalysisWarnings(K);
    if (TryOutline && kernelIsClaimable(K)) {
      claimKernelForLateOutline(K, B);
      ++Handled;
      continue;
    }
    rewriteToMatlabFor(K, B);
    ++Handled;
  }
  return Handled;
}

/* LATE phase — see header.  Lifts every kernel the EARLY phase tagged. */
unsigned runOutlineGpuKernelsLate(ModuleOp M) {
  llvm::SmallVector<Operation *> Kernels;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.gpu.kernel") && Op->hasAttr(kOutlineTag))
      Kernels.push_back(Op);
  });
  unsigned Outlined = 0;
  OpBuilder B(M.getContext());
  for (Operation *K : Kernels) {
    K->removeAttr(kOutlineTag);
    if (outlineLowered(K, Outlined)) {
      ++Outlined;
      continue;
    }
    // EARLY claimed it but the lowered shape was unexpected — fall back
    // to the CPU rewrite so the program still compiles correctly.
    rewriteToMatlabFor(K, B);
  }
  return Outlined;
}

}  // namespace mlirgen
}  // namespace matlab
