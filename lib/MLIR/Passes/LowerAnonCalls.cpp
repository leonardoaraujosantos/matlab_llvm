// Outlines matlab.make_anon bodies into llvm.func declarations and
// rewrites call sites through matlab.call_indirect into direct llvm.call
// through a function pointer.
//
// The AST->MLIR frontend emits each anonymous function `@(x, y) body` as
//   %handle = matlab.make_anon() ({
//     ^bb0(%arg0: f64, %arg1: f64):
//        <body ops>
//        matlab.yield %v
//   }) {params = "x,y"} : () -> <none>
//
// This pass:
//   1. Finds every matlab.make_anon whose body has a single block with all
//      f64 parameters, matlab.yield as its terminator, and no capture of
//      values defined outside the body.
//   2. Creates an llvm.func @__anon_N with matching (f64..) -> f64 sig
//      (or a wider sig if the yielded value is a ptr).
//   3. Moves the body's ops into the new function; the matlab.yield is
//      replaced with llvm.return.
//   4. Replaces the matlab.make_anon with llvm.mlir.addressof @__anon_N.
//   5. Rewrites each matlab.call_indirect whose callee operand is the
//      resulting ptr into an llvm.call through that pointer.
//
// Captures (body references to a value defined outside the region) are
// flagged and we skip outlining. A future pass can extend this to pass a
// captured-state struct like LowerParfor does for reductions.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

/* Propagate f64 operand types through the common unregistered matlab.*
 * scalar arithmetic ops whose result is currently `none`. Sema leaves anon-
 * body ops untyped; once the block args flow concrete types in, the loads
 * / matmul / add / etc. can all be retyped to f64. */
void propagateScalarTypesInLLVMFunc(LLVM::LLVMFuncOp Fn) {
  auto F64 = Float64Type::get(Fn.getContext());
  bool Changed = true;
  while (Changed) {
    Changed = false;
    Fn.walk([&](Operation *Op) {
      StringRef N = Op->getName().getStringRef();
      auto canRefine = [&](Type Cur, Type Proposed) {
        return mlir::isa<NoneType>(Cur) && Proposed == F64;
      };
      if ((N == "matlab.matmul" || N == "matlab.matdiv" ||
           N == "matlab.add"    || N == "matlab.sub"   ||
           N == "matlab.emul"   || N == "matlab.ediv")
          && Op->getNumOperands() == 2 && Op->getNumResults() == 1) {
        Type A = Op->getOperand(0).getType();
        Type B = Op->getOperand(1).getType();
        Type R = Op->getResult(0).getType();
        Type Concrete;
        if (A == F64) Concrete = A;
        else if (B == F64) Concrete = B;
        if (Concrete && canRefine(R, Concrete)) {
          Op->getResult(0).setType(Concrete);
          Changed = true;
        }
      }
      if (N == "matlab.load" && Op->getNumOperands() == 1 &&
          Op->getNumResults() == 1) {
        Type SlotTy = Op->getOperand(0).getType();
        Type R      = Op->getResult(0).getType();
        if (SlotTy == F64 && canRefine(R, SlotTy)) {
          Op->getResult(0).setType(SlotTy);
          Changed = true;
        }
      }
    });
  }
}

bool outlineAnon(Operation *Anon, unsigned Id) {
  ModuleOp Mod = Anon->getParentOfType<ModuleOp>();
  MLIRContext *Ctx = Anon->getContext();
  OpBuilder B(Ctx);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);

  if (Anon->getNumRegions() != 1) return false;
  Region &Body = Anon->getRegion(0);
  if (!Body.hasOneBlock()) return false;
  Block &BodyBlock = Body.front();

  /* Accept only f64 block args (scalar params) for v1. */
  llvm::SmallVector<Type> ArgTys;
  for (BlockArgument A : BodyBlock.getArguments()) {
    if (A.getType() != F64) return false;
    ArgTys.push_back(A.getType());
  }

  /* Find the terminator. Must be matlab.yield with at most one operand. */
  Operation *Term = BodyBlock.getTerminator();
  if (!isMatlabOp(Term, "matlab.yield")) return false;
  if (Term->getNumOperands() > 1) return false;

  /* Determine return type from the yielded value (or void). */
  SmallVector<Type> ResultTys;
  Value YieldVal = Term->getNumOperands() > 0
                       ? Term->getOperand(0) : Value{};
  if (YieldVal) ResultTys.push_back(YieldVal.getType());

  /* Capture check: every value used inside the body must be either defined
   * inside or be a module-level symbol. For v1 we bail on anything else. */
  llvm::DenseSet<Value> Inside;
  for (BlockArgument A : BodyBlock.getArguments()) Inside.insert(A);
  for (Operation &Op : BodyBlock)
    for (Value R : Op.getResults()) Inside.insert(R);
  for (Operation &Op : BodyBlock) {
    for (Value V : Op.getOperands()) {
      if (Inside.count(V)) continue;
      Operation *Def = V.getDefiningOp();
      if (!Def) {
        std::cerr << "anon: outside-scope block arg captured — skipping\n";
        return false;
      }
      /* Allow constants and addressof — these can be re-cloned later. */
      if (Def->getBlock() != Mod.getBody() &&
          !isa<LLVM::ConstantOp, LLVM::AddressOfOp, LLVM::ZeroOp>(Def)) {
        std::cerr << "anon: captures op '"
                  << Def->getName().getStringRef().str()
                  << "' — skipping\n";
        return false;
      }
    }
  }

  /* Create the outlined llvm.func at module scope. */
  OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToEnd(Mod.getBody());
  std::string Name = ("__anon_" + llvm::Twine(Id)).str();
  auto FnType = LLVM::LLVMFunctionType::get(
      ResultTys.empty() ? (Type)LLVM::LLVMVoidType::get(Ctx) : ResultTys[0],
      ArgTys);
  auto Fn = LLVM::LLVMFuncOp::create(B, Anon->getLoc(), Name, FnType);
  Fn.setLinkage(LLVM::Linkage::Internal);
  Block *Entry = Fn.addEntryBlock(B);

  /* Clone the body ops into the entry block, mapping block args. */
  IRMapping Map;
  for (unsigned i = 0; i < BodyBlock.getNumArguments(); ++i)
    Map.map(BodyBlock.getArgument(i), Entry->getArgument(i));

  B.setInsertionPointToEnd(Entry);
  for (Operation &Op : BodyBlock) {
    if (isMatlabOp(&Op, "matlab.yield")) continue;
    B.clone(Op, Map);
  }
  /* Propagate concrete f64 types through the body before emitting the
   * return; Sema left anon-body ops typed `none` and we need real types
   * for the LLVM conversion pipeline. */
  propagateScalarTypesInLLVMFunc(Fn);

  /* Replace yield with llvm.return. If the yielded value is still `none`
   * after propagation, the body uses ops we can't lower (matrix output
   * from a scalar anon, say) — bail and drop the outlined function. */
  if (YieldVal) {
    Value Mapped = Map.lookupOrDefault(YieldVal);
    if (mlir::isa<NoneType>(Mapped.getType())) {
      Fn.erase();
      std::cerr << "anon: body yields `none` — unsupported\n";
      return false;
    }
    if (Mapped.getType() != FnType.getReturnType()) {
      auto NewType = LLVM::LLVMFunctionType::get(Mapped.getType(), ArgTys);
      Fn.setFunctionType(NewType);
      FnType = NewType;
      ResultTys[0] = Mapped.getType();
    }
    LLVM::ReturnOp::create(B, Anon->getLoc(), ValueRange{Mapped});
  } else {
    LLVM::ReturnOp::create(B, Anon->getLoc(), ValueRange{});
  }

  /* Replace the make_anon with llvm.mlir.addressof @__anon_N. */
  B.setInsertionPoint(Anon);
  Value Addr = LLVM::AddressOfOp::create(B, Anon->getLoc(), PtrTy, Name);
  Anon->getResult(0).replaceAllUsesWith(Addr);
  Anon->erase();

  /* Rewrite every matlab.call_indirect whose callee is this addressof into
   * an llvm.call through the ptr. We resolve call_indirect ops by walking
   * the addressof value's uses.
   *
   * The signature we use is the outlined function's signature; if a call
   * site's arg types don't match we skip it. */
  SmallVector<Operation *> Calls;
  for (Operation *User : Addr.getUsers()) {
    if (isMatlabOp(User, "matlab.call_indirect") &&
        User->getNumOperands() >= 1 && User->getOperand(0) == Addr)
      Calls.push_back(User);
  }
  for (Operation *Call : Calls) {
    if (Call->getNumOperands() - 1 != ArgTys.size()) continue;
    bool OK = true;
    SmallVector<Value, 4> Args;
    for (unsigned i = 0; i < ArgTys.size(); ++i) {
      Value V = Call->getOperand(i + 1);
      if (V.getType() != ArgTys[i]) { OK = false; break; }
      Args.push_back(V);
    }
    if (!OK) continue;

    B.setInsertionPoint(Call);
    /* Build llvm.call through a function pointer. We construct the op
     * via OperationState because the LLVM dialect's typed builder has a
     * 24-parameter signature that's painful to use directly. The
     * 'var_callee_type' attribute distinguishes indirect from direct
     * calls; operandSegmentSizes records the (call-args, bundle) split. */
    OperationState State(Call->getLoc(), "llvm.call");
    SmallVector<Value> Ops;
    Ops.push_back(Addr);
    for (auto V : Args) Ops.push_back(V);
    State.addOperands(Ops);
    if (!ResultTys.empty()) State.addTypes(ResultTys);
    /* No callee symbol => this is an indirect call. We intentionally
     * do NOT set var_callee_type (that's only for variadic functions);
     * operandSegmentSizes still records the (call-args, bundle) split. */
    State.addAttribute("op_bundle_sizes",
                       DenseI32ArrayAttr::get(Ctx, {}));
    State.addAttribute("operandSegmentSizes",
                       DenseI32ArrayAttr::get(Ctx,
                           {static_cast<int32_t>(Ops.size()), 0}));
    Operation *NewCall = B.create(State);
    if (!ResultTys.empty() && Call->getNumResults() >= 1 &&
        NewCall->getNumResults() >= 1) {
      Call->getResult(0).replaceAllUsesWith(NewCall->getResult(0));
    }
    Call->erase();
  }

  return true;
}

/* Rewrite matlab.make_handle {callee = "<name>"} into an llvm.mlir.addressof
 * of the runtime's scalar entry for <name> (matlab_<name>_s). The downstream
 * call_indirect rewrite then lowers `f(x)` calls through that pointer into a
 * plain llvm.call. Only the scalar-signature builtins that exist in the
 * runtime are handled; anything else is left alone and will surface as a
 * verifier/translate error, which is preferable to silently wrong code. */
bool rewriteMakeHandle(ModuleOp M) {
  static const llvm::StringMap<StringRef> ScalarRt = {
    {"sin",  "matlab_sin_s"},  {"cos",  "matlab_cos_s"},
    {"tan",  "matlab_tan_s"},  {"exp",  "matlab_exp_s"},
    {"log",  "matlab_log_s"},  {"sqrt", "matlab_sqrt_s"},
    {"abs",  "matlab_abs_s"},
  };
  MLIRContext *Ctx = M.getContext();
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  OpBuilder B(Ctx);
  bool Changed = false;

  SmallVector<Operation *> Handles;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.make_handle")) Handles.push_back(Op);
  });
  for (Operation *H : Handles) {
    auto Attr = H->getAttrOfType<StringAttr>("callee");
    if (!Attr) continue;
    auto It = ScalarRt.find(Attr.getValue());
    if (It == ScalarRt.end()) continue;

    /* Declare the runtime function if missing. Signature: (f64) -> f64. */
    StringRef RtName = It->second;
    LLVM::LLVMFuncOp Fn = M.lookupSymbol<LLVM::LLVMFuncOp>(RtName);
    if (!Fn) {
      OpBuilder::InsertionGuard G(B);
      B.setInsertionPointToStart(M.getBody());
      auto Ty = LLVM::LLVMFunctionType::get(F64, {F64});
      Fn = LLVM::LLVMFuncOp::create(B, H->getLoc(), RtName, Ty);
      Fn.setLinkage(LLVM::Linkage::External);
    }

    B.setInsertionPoint(H);
    Value Addr = LLVM::AddressOfOp::create(B, H->getLoc(), PtrTy, RtName);
    H->getResult(0).replaceAllUsesWith(Addr);

    /* Rewrite each matlab.call_indirect whose callee is this handle into
     * an llvm.call of matlab_<name>_s with the single f64 arg. */
    SmallVector<Operation *> Calls;
    for (Operation *User : Addr.getUsers()) {
      if (isMatlabOp(User, "matlab.call_indirect") &&
          User->getNumOperands() >= 1 && User->getOperand(0) == Addr)
        Calls.push_back(User);
    }
    for (Operation *Call : Calls) {
      if (Call->getNumOperands() != 2) continue;  /* handle + 1 arg */
      Value Arg = Call->getOperand(1);
      if (Arg.getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Arg});
      if (Call->getNumResults() >= 1 && NC->getNumResults() >= 1)
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
    }
    H->erase();
    Changed = true;
  }
  return Changed;
}

/* Trace a call_indirect's callee operand back to a matlab.make_handle,
 * returning the handle's callee name on success. We follow one level of
 * matlab.load <- matlab.store <- make_handle when the handle flows
 * through a slot (`f = @sq; f(3)`). Direct handle operands work too. */
StringRef traceHandleCallee(Value Callee) {
  Operation *Def = Callee.getDefiningOp();
  if (!Def) return {};
  if (isMatlabOp(Def, "matlab.make_handle")) {
    if (auto A = Def->getAttrOfType<StringAttr>("callee"))
      return A.getValue();
    return {};
  }
  if (isMatlabOp(Def, "matlab.load") && Def->getNumOperands() == 1) {
    Value Slot = Def->getOperand(0);
    StringRef Found;
    for (OpOperand &Use : Slot.getUses()) {
      Operation *U = Use.getOwner();
      if (!isMatlabOp(U, "matlab.store")) continue;
      if (U->getNumOperands() != 2 || U->getOperand(1) != Slot) continue;
      Operation *SrcDef = U->getOperand(0).getDefiningOp();
      if (!isMatlabOp(SrcDef, "matlab.make_handle")) return {};
      auto A = SrcDef->getAttrOfType<StringAttr>("callee");
      if (!A) return {};
      if (Found.empty()) Found = A.getValue();
      else if (Found != A.getValue()) return {};
    }
    return Found;
  }
  return {};
}

/* Rewrite matlab.call_indirect through a handle whose callee resolves
 * to a func.func in the module as a direct matlab.call. This runs
 * BEFORE the LowerUserCalls/LowerScalarsToArith fixpoint so the
 * emitted matlab.call picks up type refinement the same way as a
 * syntactic `sq(3)`. */
bool rewriteUserCallIndirect(ModuleOp M) {
  SmallVector<Operation *> Calls;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call_indirect")) Calls.push_back(Op);
  });
  bool Changed = false;
  OpBuilder B(M.getContext());
  for (Operation *Call : Calls) {
    if (Call->getNumOperands() < 1) continue;
    StringRef Name = traceHandleCallee(Call->getOperand(0));
    if (Name.empty()) continue;
    if (!M.lookupSymbol<func::FuncOp>(Name)) continue;

    B.setInsertionPoint(Call);
    SmallVector<Value> Args(Call->operand_begin() + 1,
                            Call->operand_end());
    OperationState State(Call->getLoc(),
                          OperationName("matlab.call", M.getContext()));
    State.addOperands(Args);
    State.addTypes(Call->getResultTypes());
    State.addAttribute("callee", StringAttr::get(M.getContext(), Name));
    Operation *Direct = B.create(State);

    if (Call->getNumResults() >= 1 && Direct->getNumResults() >= 1)
      Call->getResult(0).replaceAllUsesWith(Direct->getResult(0));
    Call->erase();
    Changed = true;
  }

  /* Clean up orphaned make_handle/slot/load/store chains whose calls we
   * just replaced. We iterate a few rounds because erasing a load may
   * newly orphan a store, and erasing a store may newly orphan an
   * alloc. */
  for (int Round = 0; Round < 4; ++Round) {
    SmallVector<Operation *> Dead;
    M.walk([&](Operation *Op) {
      if (Op->getNumResults() == 1 && Op->getResult(0).use_empty() &&
          (isMatlabOp(Op, "matlab.make_handle") ||
           isMatlabOp(Op, "matlab.load"))) {
        Dead.push_back(Op);
        return;
      }
      if (isMatlabOp(Op, "matlab.store") && Op->getNumOperands() == 2) {
        /* A store is dead if its slot has no remaining load users. */
        Value Slot = Op->getOperand(1);
        bool HasLoad = false;
        for (Operation *U : Slot.getUsers())
          if (isMatlabOp(U, "matlab.load")) { HasLoad = true; break; }
        if (!HasLoad) Dead.push_back(Op);
      }
      if (isMatlabOp(Op, "matlab.alloc") && Op->getNumResults() == 1 &&
          Op->getResult(0).use_empty())
        Dead.push_back(Op);
    });
    if (Dead.empty()) break;
    for (Operation *Op : Dead) Op->erase();
  }
  return Changed;
}

} // namespace

bool runLowerAnonCalls(ModuleOp M) {
  /* User-function handles (`f = @sq; f(3)`) — rewrite to direct
   * matlab.call so the LowerUserCalls fixpoint refines sq's signature
   * the same way it handles a syntactic sq(3). Must run before the
   * scalar-math handle rewrite so those remain addressof+llvm.call. */
  bool Changed = rewriteUserCallIndirect(M);

  /* make_handle first so @sin-style handles resolve to addressof before the
   * anon outliner inspects call_indirect sites. */
  Changed |= rewriteMakeHandle(M);

  SmallVector<Operation *> Anons;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.make_anon")) Anons.push_back(Op);
  });
  unsigned Id = 0;
  for (Operation *A : Anons)
    if (outlineAnon(A, Id++)) Changed = true;
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
