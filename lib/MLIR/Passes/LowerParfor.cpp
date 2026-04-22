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

  // Capture analysis: collect values used inside the body that are defined
  // outside. For each, the defining op must be cloneable; otherwise we bail.
  llvm::DenseSet<Value> DefinedInside;
  DefinedInside.insert(IV);
  for (Operation &Op : BodyBlock) {
    for (Value R : Op.getResults()) DefinedInside.insert(R);
  }
  llvm::SmallVector<Operation *> ExternsToClone;
  llvm::DenseSet<Operation *> ExternSet;
  for (Operation &Op : BodyBlock) {
    for (Value Operand : Op.getOperands()) {
      if (DefinedInside.count(Operand)) continue;
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

  // Create the outlined function as an llvm.func so llvm.mlir.addressof
  // can reference it directly. The body still contains arith.* and llvm.call
  // ops; the final conversion-to-LLVM pipeline handles the rest.
  std::string Name = ("__parfor_body_" + llvm::Twine(Id)).str();
  B.setInsertionPointToEnd(Module.getBody());
  auto FnTy = LLVM::LLVMFunctionType::get(VoidTy, {F64});
  auto Fn = LLVM::LLVMFuncOp::create(B, Loc, Name, FnTy);
  Fn.setLinkage(LLVM::Linkage::Internal);
  Block *Entry = Fn.addEntryBlock(B);

  // Clone body ops into the entry block.
  IRMapping Mapping;
  Mapping.map(IV, Entry->getArgument(0));

  B.setInsertionPointToEnd(Entry);
  // Clone externals first so their results are available when body ops refer
  // to them. Any external captured by more than one body op is cloned once.
  for (Operation *Ext : ExternsToClone) B.clone(*Ext, Mapping);

  // Clone every op except the region's terminator (matlab.yield).
  for (Operation &Op : BodyBlock) {
    if (isMatlabOp(&Op, "matlab.yield")) continue;
    B.clone(Op, Mapping);
  }
  LLVM::ReturnOp::create(B, Loc, ValueRange{});

  // Replace the parfor with: default step if missing, take addressof outlined
  // fn, call matlab_parfor_dispatch(start, step, end, bodyPtr).
  B.setInsertionPoint(Parfor);
  Value StepV = Step ? Step
                     : static_cast<Value>(arith::ConstantOp::create(
                           B, Loc, F64, B.getF64FloatAttr(1.0)));

  Value FnPtr = LLVM::AddressOfOp::create(B, Loc, PtrTy, Fn.getName());
  auto Dispatch = getOrInsertRTDecl(
      B, Module, "matlab_parfor_dispatch", VoidTy,
      {F64, F64, F64, PtrTy});
  LLVM::CallOp::create(B, Loc, Dispatch,
                       ValueRange{Start, StepV, End, FnPtr});

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
