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
//                   Real outline of the body into an llvm.func with
//                   signature `void(double iv, void *state)`.  Captures
//                   only tensor-typed `matlab.alloc` results (output
//                   arrays); falls back to the simple rewrite if the
//                   body has any scalar capture the outliner can't
//                   classify.  The outlined function is the canonical
//                   source the Tier-2/3/4 EmitMetal/CUDA/OpenCL passes
//                   walk to print kernel source.
//
// **Why the outliner is conservative (the architectural finding)**.
// A `coder.gpu.kernelfun`-tagged body references three classes of
// outer slot:
//
//   1. Output arrays (tensor-typed `matlab.alloc`): need to be visible
//      across iterations.  ABI: pass the ptr through state.  Backends
//      bind this as MTLBuffer / CUdeviceptr / cl_mem.
//   2. Outer scalars set BEFORE the kernel (e.g. `n_grid = 64;
//      re_min = -2.0;`): read-only inside.  ABI options:
//        a) inline as constant if the source is a static literal,
//        b) pass as additional scalar args via an extended state ABI,
//        c) load once before launch and pass packed.
//   3. Kernel-local temporaries (e.g. `cr = ...; zi = ...; k = k+1;`
//      with first-use being a write): per-iteration private storage.
//      ABI: clone the matlab.alloc into the outlined function's entry
//      block; treat as local.
//
// Distinguishing (2) from (3) requires a **definite-assignment scope
// analysis** on the body that the project's Sema doesn't expose yet.
// Without it, lifting a temp like `cr` through state corrupts results
// across iterations (every thread reads/writes the same outer slot).
//
// The conservative outliner accepts kernels whose ONLY outer-defined
// references are:
//   - tensor-typed `matlab.alloc` results (bucket 1 — passed via state)
//   - cloneable externals (arith.constant / llvm.constant /
//     llvm.mlir.zero / llvm.mlir.addressof — cloned into the outlined
//     function)
// Anything else (esp. f64-typed `matlab.alloc` from a bucket-2 or -3
// scalar) makes the outliner decline and the simple rewrite handles
// the kernel correctly on the CPU lane.
//
// **What's left for future work**:
//   - Definite-assignment analysis to split bucket-2 vs bucket-3 scalar
//     slots.
//   - Extended state ABI carrying mixed (ptr, f64) entries.
//   - The Tier-2/3/4 emit passes don't strictly need this outliner —
//     they can emit kernel source by walking the rewritten matlab.for
//     body directly (with a small wrapper around the output-array
//     bind step).  Documented in
//     docs/gpu_coder_progress.md as the recommended path.
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

bool isCloneableExternal(Operation *Op) {
  if (!Op) return false;
  if (isa<arith::ConstantOp>(Op)) return true;
  if (isa<LLVM::ConstantOp>(Op)) return true;
  if (isa<LLVM::ZeroOp>(Op)) return true;
  if (isa<LLVM::AddressOfOp>(Op)) return true;
  return false;
}

/* Tensor-typed `matlab.alloc` results are the only captures the
 * conservative outliner handles — see the file header. */
bool isTensorArrayCapture(Operation *Def, Value V) {
  if (!Def) return false;
  if (!isMatlabOp(Def, "matlab.alloc")) return false;
  return mlir::isa<RankedTensorType, UnrankedTensorType>(V.getType());
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

/* Real outliner — conservative variant.  Returns true on success;
 * false when the body has shapes outside the supported bucket so the
 * caller can fall back to the simple rewrite. */
bool outlineToLLVMFunc(Operation *K, unsigned KernelId) {
  Location Loc = K->getLoc();
  MLIRContext *Ctx = K->getContext();
  ModuleOp Module = K->getParentOfType<ModuleOp>();
  OpBuilder B(Ctx);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto VoidTy = LLVM::LLVMVoidType::get(Ctx);
  auto I32 = IntegerType::get(Ctx, 32);
  auto I64 = IntegerType::get(Ctx, 64);

  if (K->getNumOperands() < 1) return false;
  Value Iter = K->getOperand(0);
  Operation *Range = Iter.getDefiningOp();
  if (!isMatlabOp(Range, "matlab.range")) return false;

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
      (Step && Step.getType() != F64))
    return false;

  if (K->getNumRegions() != 1) return false;
  Region &Body = K->getRegion(0);
  if (!Body.hasOneBlock()) return false;
  Block &BodyBlock = Body.front();
  if (BodyBlock.getNumArguments() != 1 ||
      BodyBlock.getArgument(0).getType() != F64)
    return false;
  Value IV = BodyBlock.getArgument(0);

  StringRef VarName;
  if (auto VA = K->getAttrOfType<StringAttr>("var"))
    VarName = VA.getValue();
  if (!VarName.empty()) {
    llvm::SmallVector<Operation *, 4> StoresToErase;
    for (Operation &Op : BodyBlock) {
      if (isMatlabOp(&Op, "matlab.load") && Op.getNumOperands() == 1) {
        Operation *SlotDef = Op.getOperand(0).getDefiningOp();
        if (isMatlabOp(SlotDef, "matlab.alloc")) {
          auto NameA = SlotDef->getAttrOfType<StringAttr>("name");
          if (NameA && NameA.getValue() == VarName)
            Op.getResult(0).replaceAllUsesWith(IV);
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

  // Capture analysis — accept ONLY tensor-typed matlab.alloc captures
  // plus cloneable externals.  Decline anything else.
  llvm::DenseSet<Value> DefinedInside;
  DefinedInside.insert(IV);
  for (Operation &Op : BodyBlock)
    for (Value R : Op.getResults()) DefinedInside.insert(R);

  llvm::SmallVector<Operation *> ExternsToClone;
  llvm::DenseSet<Operation *> ExternSet;
  llvm::SmallVector<Value> Captures;
  llvm::DenseSet<Value> CaptureSet;
  for (Operation &Op : BodyBlock) {
    for (Value Operand : Op.getOperands()) {
      if (DefinedInside.count(Operand)) continue;
      if (CaptureSet.count(Operand)) continue;
      Operation *Def = Operand.getDefiningOp();
      if (!Def) return false;  // outer block argument — decline
      if (isCloneableExternal(Def)) {
        if (ExternSet.insert(Def).second) ExternsToClone.push_back(Def);
      } else if (isTensorArrayCapture(Def, Operand)) {
        Captures.push_back(Operand);
        CaptureSet.insert(Operand);
      } else {
        // Bucket-2 / bucket-3 scalar slot (or unsupported op) — decline.
        return false;
      }
    }
  }

  std::string Name = ("__gpu_kernel_" + llvm::Twine(KernelId)).str();
  B.setInsertionPointToEnd(Module.getBody());
  auto FnTy = LLVM::LLVMFunctionType::get(VoidTy, {F64, PtrTy});
  auto Fn = LLVM::LLVMFuncOp::create(B, Loc, Name, FnTy);
  Fn.setLinkage(LLVM::Linkage::External);
  Block *Entry = Fn.addEntryBlock(B);
  Value InnerIV = Entry->getArgument(0);
  Value StateArg = Entry->getArgument(1);

  B.setInsertionPointToEnd(Entry);

  llvm::DenseMap<Value, Value> CaptureRemap;
  for (size_t k = 0; k < Captures.size(); ++k) {
    Value IdxK = LLVM::ConstantOp::create(
        B, Loc, I64, B.getI64IntegerAttr((int64_t)k));
    Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateArg,
                                    ValueRange{IdxK});
    Value LoadedPtr = LLVM::LoadOp::create(B, Loc, PtrTy, Gep);
    Type OrigTy = Captures[k].getType();
    if (OrigTy == PtrTy) {
      CaptureRemap[Captures[k]] = LoadedPtr;
    } else {
      auto Cast = UnrealizedConversionCastOp::create(B, Loc, OrigTy,
                                                     LoadedPtr);
      CaptureRemap[Captures[k]] = Cast.getResult(0);
    }
  }

  IRMapping Mapping;
  Mapping.map(IV, InnerIV);
  for (auto &P : CaptureRemap) Mapping.map(P.first, P.second);
  for (Operation *Ext : ExternsToClone) B.clone(*Ext, Mapping);
  for (Operation &Op : BodyBlock) {
    if (isMatlabOp(&Op, "matlab.yield")) continue;
    B.clone(Op, Mapping);
  }
  LLVM::ReturnOp::create(B, Loc, ValueRange{});

  B.setInsertionPoint(K);
  Value StepV = Step ? Step
                     : static_cast<Value>(arith::ConstantOp::create(
                           B, Loc, F64, B.getF64FloatAttr(1.0)));
  Value FnPtr = LLVM::AddressOfOp::create(B, Loc, PtrTy, Fn.getName());

  Value StateOuter;
  if (Captures.empty()) {
    StateOuter = LLVM::ZeroOp::create(B, Loc, PtrTy);
  } else {
    auto ArrTy = LLVM::LLVMArrayType::get(PtrTy,
                                          static_cast<unsigned>(Captures.size()));
    Value One = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
    StateOuter = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrTy, One,
                                        /*alignment=*/0);
    for (size_t k = 0; k < Captures.size(); ++k) {
      Value IdxK = LLVM::ConstantOp::create(
          B, Loc, I64, B.getI64IntegerAttr((int64_t)k));
      Value Gep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, StateOuter,
                                      ValueRange{IdxK});
      Value PtrVal;
      if (Captures[k].getType() == PtrTy) {
        PtrVal = Captures[k];
      } else {
        auto Cast = UnrealizedConversionCastOp::create(B, Loc, PtrTy,
                                                      Captures[k]);
        PtrVal = Cast.getResult(0);
      }
      LLVM::StoreOp::create(B, Loc, PtrVal, Gep);
    }
  }

  Value KernelIdV = LLVM::ConstantOp::create(B, Loc, I32,
                                             B.getI32IntegerAttr((int32_t)KernelId));
  auto Dispatch = getOrInsertRTDecl(
      B, Module, "matlab_gpu_launch_kernel", VoidTy,
      {F64, F64, F64, PtrTy, PtrTy, I32});
  LLVM::CallOp::create(B, Loc, Dispatch,
                       ValueRange{Start, StepV, End, FnPtr, StateOuter,
                                  KernelIdV});

  K->erase();
  if (Iter.use_empty() && Range) Range->erase();
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

unsigned runOutlineGpuKernels(ModuleOp M) {
  llvm::SmallVector<Operation *> Kernels;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.gpu.kernel")) Kernels.push_back(Op);
  });
  unsigned Rewritten = 0;
  OpBuilder B(M.getContext());
  bool TryOutline = wantRealOutline();
  for (Operation *K : Kernels) {
    emitKernelAnalysisWarnings(K);
    if (TryOutline && outlineToLLVMFunc(K, Rewritten)) {
      ++Rewritten;
      continue;
    }
    rewriteToMatlabFor(K, B);
    ++Rewritten;
  }
  return Rewritten;
}

}  // namespace mlirgen
}  // namespace matlab
