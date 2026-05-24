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
  auto PtrTy = LLVM::LLVMPointerType::get(Fn.getContext());
  auto isPtrLike = [&](Type T) {
    return T == PtrTy || mlir::isa<RankedTensorType>(T) ||
           mlir::isa<UnrankedTensorType>(T);
  };
  bool Changed = true;
  while (Changed) {
    Changed = false;
    Fn.walk([&](Operation *Op) {
      StringRef N = Op->getName().getStringRef();
      auto canRefine = [&](Type Cur, Type Proposed) {
        return mlir::isa<NoneType>(Cur) && (Proposed == F64 ||
                                            isPtrLike(Proposed));
      };
      if ((N == "matlab.matmul" || N == "matlab.matdiv" ||
           N == "matlab.add"    || N == "matlab.sub"   ||
           N == "matlab.emul"   || N == "matlab.ediv")
          && Op->getNumOperands() == 2 && Op->getNumResults() == 1) {
        Type A = Op->getOperand(0).getType();
        Type B = Op->getOperand(1).getType();
        Type R = Op->getResult(0).getType();
        /* Mixed scalar×matrix or matrix×scalar: result shape follows
         * the non-scalar operand. Pure scalar×scalar yields f64. */
        Type Concrete;
        if (isPtrLike(A) || isPtrLike(B)) {
          Concrete = isPtrLike(A) ? A : B;
        } else if (A == F64) {
          Concrete = A;
        } else if (B == F64) {
          Concrete = B;
        }
        if (Concrete && canRefine(R, Concrete)) {
          Op->getResult(0).setType(Concrete);
          Changed = true;
        }
      }
      if (N == "matlab.load" && Op->getNumOperands() == 1 &&
          Op->getNumResults() == 1) {
        Type SlotTy = Op->getOperand(0).getType();
        Type R      = Op->getResult(0).getType();
        if ((SlotTy == F64 || isPtrLike(SlotTy)) &&
            canRefine(R, SlotTy)) {
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

  /* Accept f64 (scalar params/captures) and ptr (matrix captures) block
   * args. Tensor-typed block args get retyped to ptr first — by this
   * point LowerTensorOps hasn't run yet, so matrix captures arrive as
   * tensor<...xf64>. We rewrite those to ptr on the block arg and on
   * all uses as a ptr-typed load. */
  llvm::SmallVector<Type> ArgTys;
  for (BlockArgument A : BodyBlock.getArguments()) {
    Type T = A.getType();
    if (T == F64) {
      ArgTys.push_back(T);
      continue;
    }
    if (mlir::isa<RankedTensorType>(T) ||
        mlir::isa<UnrankedTensorType>(T)) {
      A.setType(PtrTy);
      ArgTys.push_back(PtrTy);
      continue;
    }
    if (T == PtrTy) { ArgTys.push_back(T); continue; }
    return false;
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
    /* If the yield value is still tensor-typed (a matrix anon), the
     * runtime ABI passes matrices as !llvm.ptr; LowerTensorOps will
     * retype the producer ops afterwards. Pre-commit the signature to
     * ptr so the post-LowerTensorOps llvm.return types agree. */
    Type RetTy = Mapped.getType();
    if (mlir::isa<RankedTensorType>(RetTy) ||
        mlir::isa<UnrankedTensorType>(RetTy))
      RetTy = PtrTy;
    if (RetTy != FnType.getReturnType()) {
      auto NewType = LLVM::LLVMFunctionType::get(RetTy, ArgTys);
      Fn.setFunctionType(NewType);
      FnType = NewType;
      ResultTys.clear();
      ResultTys.push_back(RetTy);
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
    if (It == ScalarRt.end()) {
      /* Fall through: maybe it's a user-defined func.func in the
       * module.  Resolve `@name` → llvm.mlir.addressof so arrayfun /
       * matlab_gpucoder_reduce / etc. get a real function pointer.
       * The runtime call ABI expects `double (double)` for unary or
       * `double (double, double)` for binary — caller's responsibility
       * to match. */
      auto FuncSym = M.lookupSymbol<func::FuncOp>(Attr.getValue());
      if (!FuncSym) continue;
      /* Use func.constant to materialise a function reference to the
       * func.func, then bitcast/unrealized_conversion_cast to ptr for
       * the make_handle's downstream ptr consumers. */
      B.setInsertionPoint(H);
      auto FuncRef = func::ConstantOp::create(
          B, H->getLoc(), FuncSym.getFunctionType(),
          FuncSym.getSymName());
      auto Cast = mlir::UnrealizedConversionCastOp::create(
          B, H->getLoc(), PtrTy, FuncRef.getResult());
      H->getResult(0).replaceAllUsesWith(Cast.getResult(0));
      H->erase();
      Changed = true;
      continue;
    }

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

/* Second-chance rewrite for matlab.call_indirect ops that survived the
 * first LowerAnonCalls run because their operand types hadn't yet been
 * retyped from tensor to ptr. By the time LowerTensorOps has run, the
 * operands at the call site are ptr and can match the outlined
 * llvm.func's signature. We don't walk through slot indirection here —
 * if the user stored the handle in a slot, LowerAnonCalls's first run
 * already folded those paths before outlining. */
bool runLowerAnonCallsPost(ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  OpBuilder B(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto isTensorLike = [](Type T) {
    return mlir::isa<RankedTensorType>(T) || mlir::isa<UnrankedTensorType>(T);
  };
  SmallVector<Operation *> Calls;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call_indirect")) Calls.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Call : Calls) {
    if (Call->getNumOperands() < 1) continue;
    /* Trace the callee operand back to an llvm.mlir.addressof.  The
     * handle may be the direct defining op, or it may flow through a
     * slot — and at this point in the pipeline that slot's load/store
     * pair can still be either `matlab.load`/`matlab.store` (the anon
     * outliner replaced the make_anon with an addressof but left the
     * surrounding slot ops untouched) or already-lowered `llvm.load`/
     * `llvm.store`.  Handle both. */
    Value Callee = Call->getOperand(0);
    Operation *Def = Callee.getDefiningOp();
    StringRef FnName;
    auto srcAddrName = [](Value Src) -> StringRef {
      if (auto Addr = dyn_cast_or_null<LLVM::AddressOfOp>(Src.getDefiningOp()))
        return Addr.getGlobalName();
      return {};
    };
    if (auto Addr = dyn_cast_or_null<LLVM::AddressOfOp>(Def)) {
      FnName = Addr.getGlobalName();
    } else if (Def && (Def->getName().getStringRef() == "llvm.load" ||
                       isMatlabOp(Def, "matlab.load")) &&
               Def->getNumOperands() == 1) {
      Value Slot = Def->getOperand(0);
      for (OpOperand &Use : Slot.getUses()) {
        Operation *U = Use.getOwner();
        bool IsStore = U->getName().getStringRef() == "llvm.store" ||
                       isMatlabOp(U, "matlab.store");
        if (!IsStore) continue;
        if (U->getNumOperands() != 2 || U->getOperand(1) != Slot) continue;
        StringRef N = srcAddrName(U->getOperand(0));
        if (N.empty()) { FnName = {}; break; }
        if (FnName.empty()) FnName = N;
        else if (FnName != N) { FnName = {}; break; }
      }
    }
    if (FnName.empty()) continue;
    auto Fn = M.lookupSymbol<LLVM::LLVMFuncOp>(FnName);
    if (!Fn) continue;
    auto FnType = Fn.getFunctionType();
    if (Call->getNumOperands() - 1 != FnType.getNumParams()) continue;
    bool OK = true;
    SmallVector<Value> Args;
    B.setInsertionPoint(Call);
    for (unsigned i = 0; i < FnType.getNumParams(); ++i) {
      Value V = Call->getOperand(i + 1);
      Type Want = FnType.getParamType(i);
      if (V.getType() == Want) {
        Args.push_back(V);
      } else if (Want == PtrTy && isTensorLike(V.getType())) {
        /* The outlined function takes a ptr but the call site still
         * has a tensor-typed value (e.g. the result of a builtin call
         * that hasn't been converted yet) — bridge with an
         * unrealized_conversion_cast, the same ptr/tensor ABI bridge
         * the PDE / Optim dispatch tables use. */
        auto Cast = mlir::UnrealizedConversionCastOp::create(
            B, Call->getLoc(), PtrTy, V);
        Args.push_back(Cast.getResult(0));
      } else {
        OK = false;
        break;
      }
    }
    if (!OK) continue;

    B.setInsertionPoint(Call);
    OperationState State(Call->getLoc(), "llvm.call");
    SmallVector<Value> Ops;
    Ops.push_back(Callee);
    for (auto V : Args) Ops.push_back(V);
    State.addOperands(Ops);
    if (!mlir::isa<LLVM::LLVMVoidType>(FnType.getReturnType()))
      State.addTypes({FnType.getReturnType()});
    State.addAttribute("op_bundle_sizes",
                       DenseI32ArrayAttr::get(Ctx, {}));
    State.addAttribute("operandSegmentSizes",
                       DenseI32ArrayAttr::get(Ctx,
                           {static_cast<int32_t>(Ops.size()), 0}));
    Operation *NewCall = B.create(State);
    if (Call->getNumResults() >= 1 && NewCall->getNumResults() >= 1)
      Call->getResult(0).replaceAllUsesWith(NewCall->getResult(0));
    Call->erase();
    Changed = true;
  }
  return Changed;
}

/* Trace a value back to a matlab.make_anon op. The anon may either be
 * the value's direct defining op, or stored into a slot whose load
 * defines the value. Returns nullptr if the value doesn't trace to a
 * unique make_anon. */
static Operation *traceToMakeAnon(Value V) {
  Operation *Def = V.getDefiningOp();
  if (!Def) return nullptr;
  if (isMatlabOp(Def, "matlab.make_anon")) return Def;
  if (isMatlabOp(Def, "matlab.load") && Def->getNumOperands() == 1) {
    Value Slot = Def->getOperand(0);
    Operation *Found = nullptr;
    for (Operation *U : Slot.getUsers()) {
      if (!isMatlabOp(U, "matlab.store")) continue;
      if (U->getNumOperands() != 2 || U->getOperand(1) != Slot) continue;
      Operation *Src = U->getOperand(0).getDefiningOp();
      if (!isMatlabOp(Src, "matlab.make_anon")) return nullptr;
      if (Found && Found != Src) return nullptr;
      Found = Src;
    }
    return Found;
  }
  return nullptr;
}

/* Pre-pass for vector-y ode45/ode23. Sema types anonymous-function block
 * args as f64 by default. When the anon is passed to ode45 / ode23 with
 * a matrix-typed y0, retype the second block arg (y) to ptr so the
 * outlined function correctly takes a matrix and the vector-dispatch
 * path in LowerTensorOps fires. The y-slot alloc and any loads of that
 * slot are cascade-retyped so the body's printed type signatures stay
 * consistent. */
static bool retypeAnonsForVectorODE(ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  bool Changed = false;
  SmallVector<Operation *> Calls;
  M.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto Cn = Op->getAttrOfType<StringAttr>("callee");
    if (!Cn) return;
    StringRef Name = Cn.getValue();
    if (Name != "ode45" && Name != "ode23" && Name != "ode23s") return;
    if (Op->getNumOperands() < 3) return;
    Calls.push_back(Op);
  });
  for (Operation *Call : Calls) {
    Type Y0Ty = Call->getOperand(2).getType();
    bool Y0IsMatrix = mlir::isa<RankedTensorType>(Y0Ty) ||
                      mlir::isa<UnrankedTensorType>(Y0Ty) ||
                      Y0Ty == PtrTy;
    if (!Y0IsMatrix) continue;
    Operation *Anon = traceToMakeAnon(Call->getOperand(0));
    if (!Anon || Anon->getNumRegions() != 1) continue;
    Region &Body = Anon->getRegion(0);
    if (!Body.hasOneBlock()) continue;
    Block &Entry = Body.front();
    if (Entry.getNumArguments() < 2) continue;
    BlockArgument YArg = Entry.getArgument(1);
    if (YArg.getType() != F64) continue;       /* already retyped or wrong shape */

    YArg.setType(PtrTy);
    /* Cascade: retype the slot the arg gets stored into, plus loads. */
    for (Operation *U : YArg.getUsers()) {
      if (!isMatlabOp(U, "matlab.store") || U->getNumOperands() != 2) continue;
      if (U->getOperand(0) != YArg) continue;
      Value Slot = U->getOperand(1);
      Operation *SlotDef = Slot.getDefiningOp();
      if (isMatlabOp(SlotDef, "matlab.alloc"))
        SlotDef->getResult(0).setType(PtrTy);
      for (Operation *SU : Slot.getUsers()) {
        if (isMatlabOp(SU, "matlab.load") && SU->getNumResults() == 1)
          SU->getResult(0).setType(PtrTy);
      }
    }
    Changed = true;
  }
  return Changed;
}

/* Retype one block arg of the anonymous function reachable from
 * `handle` to ptr, cascading the change through the prologue slot the
 * argument is stored into and that slot's loads.  Returns true if a
 * change was made.  Shared by the Optimization-Toolbox pre-passes that
 * give an objective / constraint handle a vector (matrix) argument.   */
static bool retypeMakeAnonArg(Operation *Anon, unsigned argIdx) {
  if (!Anon || Anon->getNumRegions() != 1) return false;
  MLIRContext *Ctx = Anon->getContext();
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  Region &Body = Anon->getRegion(0);
  if (!Body.hasOneBlock()) return false;
  Block &Entry = Body.front();
  if (Entry.getNumArguments() <= argIdx) return false;
  BlockArgument Arg = Entry.getArgument(argIdx);
  if (Arg.getType() != F64) return false;     /* already retyped / wrong shape */

  Arg.setType(PtrTy);
  for (Operation *U : Arg.getUsers()) {
    if (!isMatlabOp(U, "matlab.store") || U->getNumOperands() != 2) continue;
    if (U->getOperand(0) != Arg) continue;
    Value Slot = U->getOperand(1);
    Operation *SlotDef = Slot.getDefiningOp();
    if (isMatlabOp(SlotDef, "matlab.alloc"))
      SlotDef->getResult(0).setType(PtrTy);
    for (Operation *SU : Slot.getUsers()) {
      if (isMatlabOp(SU, "matlab.load") && SU->getNumResults() == 1)
        SU->getResult(0).setType(PtrTy);
    }
  }
  return true;
}

static bool retypeAnonArg(Value handle, unsigned argIdx) {
  return retypeMakeAnonArg(traceToMakeAnon(handle), argIdx);
}

/* REPL-only pre-pass.  In the static -emit-* pipeline an anon used as a
 * vector objective is reached from its solver call site
 * (`retypeAnonsForVectorObjective` traces `fminunc(ros, …)` back to the
 * `make_anon`).  In the REPL each line is its own module, so the turn
 * that types `ros = @(x) … x(1) … x(2) …` has no solver call to key
 * off — the anon would be outlined with an f64 `x` and its
 * `matlab.subscript` reads would survive untranslatable.
 *
 * Heuristic: if a `make_anon` block argument is itself used as the base
 * of a `matlab.subscript` (directly, or through the prologue
 * store→slot→load it is spilled into), the parameter is being indexed
 * — it is array-shaped — so retype it to ptr.  This fixes the ABI at
 * the *definition* site, independent of any call site, so the handle
 * stored in the workspace already has the `double(*)(matlab_mat*)`
 * shape the solver expects on a later turn. */
static bool retypeAnonsBySubscriptedParam(ModuleOp M) {
  bool Changed = false;
  SmallVector<Operation *> Anons;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.make_anon")) Anons.push_back(Op);
  });
  auto F64 = Float64Type::get(M.getContext());
  for (Operation *Anon : Anons) {
    if (Anon->getNumRegions() != 1) continue;
    Region &Body = Anon->getRegion(0);
    if (!Body.hasOneBlock()) continue;
    Block &Entry = Body.front();
    for (unsigned ai = 0; ai < Entry.getNumArguments(); ++ai) {
      BlockArgument Arg = Entry.getArgument(ai);
      if (Arg.getType() != F64) continue;
      /* Does this arg feed a matlab.subscript as its base value?  Check
       * both the direct use and the spilled-slot load chain. */
      bool Subscripted = false;
      auto feedsSubscript = [&](Value V) {
        for (Operation *U : V.getUsers())
          if (isMatlabOp(U, "matlab.subscript") && U->getNumOperands() >= 1 &&
              U->getOperand(0) == V)
            return true;
        return false;
      };
      if (feedsSubscript(Arg)) {
        Subscripted = true;
      } else {
        for (Operation *U : Arg.getUsers()) {
          if (!isMatlabOp(U, "matlab.store") || U->getNumOperands() != 2)
            continue;
          if (U->getOperand(0) != Arg) continue;
          for (Operation *SU : U->getOperand(1).getUsers())
            if (isMatlabOp(SU, "matlab.load") && SU->getNumResults() == 1 &&
                feedsSubscript(SU->getResult(0)))
              Subscripted = true;
        }
      }
      if (Subscripted)
        Changed |= retypeMakeAnonArg(Anon, ai);
    }
  }
  return Changed;
}

/* Pre-pass for the Optimization-Toolbox solvers whose objective /
 * constraint / model handles take a vector argument.  Sema types
 * anonymous-function block args as f64 by default; retype the relevant
 * ones to ptr so the outlined function takes a matrix and the body's
 * `x(i)` subscripts lower against a ptr base.  Mirrors
 * retypeAnonsForVectorODE.  Per solver:
 *   fminsearch / fminunc / lsqnonlin — handle is operand 0, retype arg 0
 *   fmincon  — objective operand 0 (arg 0); 9-arg form also has the
 *              nonlcon handle at operand 8 (arg 0)
 *   lsqcurvefit — model handle fun(params, xdata): operand 0, args 0 & 1
 *   fsolve   — operand 0; only the N-D form (vector x0 at operand 1)
 *              has a vector argument, the scalar form keeps f64        */
static bool retypeAnonsForVectorObjective(ModuleOp M) {
  auto PtrTy = LLVM::LLVMPointerType::get(M.getContext());
  bool Changed = false;
  SmallVector<Operation *> Calls;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call_builtin")) Calls.push_back(Op);
  });
  for (Operation *Call : Calls) {
    auto Cn = Call->getAttrOfType<StringAttr>("callee");
    if (!Cn) continue;
    StringRef Name = Cn.getValue();
    unsigned nops = Call->getNumOperands();
    if ((Name == "fminsearch" || Name == "fminunc" || Name == "lsqnonlin" ||
         Name == "fminimax" || Name == "fgoalattain" ||
         /* Global Optimization Tier-1 — objective handle at operand 0,
          * both the user-facing names and the runtime symbols (the
          * latter after the Lowering hook has rewritten the call). */
         Name == "ga" || Name == "particleswarm" || Name == "simulannealbnd" ||
         Name == "patternsearch" || Name == "surrogateopt" ||
         Name == "gamultiobj" || Name == "paretosearch" ||
         Name == "matlab_gads_ga" || Name == "matlab_gads_ga_opts" ||
         Name == "matlab_gads_particleswarm" ||
         Name == "matlab_gads_simulannealbnd" ||
         Name == "matlab_gads_patternsearch" ||
         Name == "matlab_gads_surrogateopt" ||
         Name == "matlab_gads_gamultiobj" ||
         Name == "matlab_gads_paretosearch" ||
         /* Stats Tier-6: bayesopt objective handle (operand 0). */
         Name == "matlab_stats_bayesopt" ||
         /* Tier-2: createOptimProblem stashes the objective handle (op 0
          * of make_problem) into the thread-local; retype it to ptr. */
         Name == "matlab_gads_make_problem") &&
        nops >= 1) {
      /* Single vector-argument objective / vector-residual handle. */
      Changed |= retypeAnonArg(Call->getOperand(0), 0);
    } else if (Name == "fseminf" && nops >= 3) {
      /* fseminf(@fun, x0, @seminfcon, ...): `fun` is a vector
       * objective (operand 0, arg 0); `seminfcon` is the per-point
       * handle fcon(x, w) at operand 2 — only its first arg `x` is a
       * vector, `w` stays a scalar, exactly like lsqcurvefit's model. */
      Changed |= retypeAnonArg(Call->getOperand(0), 0);
      Changed |= retypeAnonArg(Call->getOperand(2), 0);
    } else if (Name == "lsqcurvefit" && nops >= 1) {
      /* Model handle fun(params, t): only `params` is a vector — it is
       * indexed in the body, so retyping is safe.  `t` is passed as a
       * scalar (the model is evaluated per data point), so it stays
       * f64 and element-wise model expressions lower cleanly. */
      Changed |= retypeAnonArg(Call->getOperand(0), 0);
    } else if (Name == "fmincon" && nops >= 1) {
      Changed |= retypeAnonArg(Call->getOperand(0), 0);
      if (nops == 9)
        Changed |= retypeAnonArg(Call->getOperand(8), 0);
    } else if (Name == "fsolve" && nops == 2) {
      Type X0Ty = Call->getOperand(1).getType();
      bool x0Vector = mlir::isa<RankedTensorType>(X0Ty) ||
                      mlir::isa<UnrankedTensorType>(X0Ty) ||
                      X0Ty == PtrTy;
      if (x0Vector) Changed |= retypeAnonArg(Call->getOperand(0), 0);
    } else if (Name == "nlmpcmove" && nops == 5) {
      /* MPC Tier-5: nlmpcmove(nlobj, x, lastu, r, @stateFn) — the
       * StateFcn handle's sole block arg is the packed `zxu = [x; u]`
       * column vector.  Retype so the body's `zxu(i, 1)` subscripts
       * lower against a ptr base. */
      Changed |= retypeAnonArg(Call->getOperand(4), 0);
    } else if (Name == "matlab_nlmpc_move" && nops == 5) {
      /* Same as above but at the runtime-symbol level (the call
       * site after the Tier-5 Lowering hook has run). */
      Changed |= retypeAnonArg(Call->getOperand(4), 0);
    } else if (Name == "greyest" && nops == 4) {
      /* Ident Tier-4: greyest(data, par0, @structfn, nx) — the
       * structure handle's sole block arg is the parameter vector,
       * subscripted (par(i)) in the body, so retype to ptr. */
      Changed |= retypeAnonArg(Call->getOperand(2), 0);
    } else if ((Name == "matlab_ident_greyest" ||
                Name == "matlab_ident_nlgreyest") && nops == 5) {
      /* Same at the runtime-symbol level: (nl)greyest(model, data, par0,
       * @structfn, nx) — handle is operand 3. */
      Changed |= retypeAnonArg(Call->getOperand(3), 0);
    } else if ((Name == "matlab_ident_ekf_predict" ||
                Name == "matlab_ident_ukf_predict") && nops == 2) {
      /* EKF/UKF predict(obj, @StateFcn) — StateFcn(x) handle at op 1. */
      Changed |= retypeAnonArg(Call->getOperand(1), 0);
    } else if ((Name == "matlab_ident_ekf_correct" ||
                Name == "matlab_ident_ukf_correct") && nops == 3) {
      /* EKF/UKF correct(obj, @MeasFcn, y) — MeasFcn(x) handle at op 1. */
      Changed |= retypeAnonArg(Call->getOperand(1), 0);
    }
  }
  return Changed;
}

bool runLowerAnonCalls(ModuleOp M, bool ReplMode) {
  /* User-function handles (`f = @sq; f(3)`) — rewrite to direct
   * matlab.call so the LowerUserCalls fixpoint refines sq's signature
   * the same way it handles a syntactic sq(3). Must run before the
   * scalar-math handle rewrite so those remain addressof+llvm.call. */
  bool Changed = rewriteUserCallIndirect(M);

  /* Vector-y ode45/ode23: retype anon `y` block args before outlining. */
  Changed |= retypeAnonsForVectorODE(M);

  /* Vector-objective fminsearch/fminunc: retype anon `x` block arg. */
  Changed |= retypeAnonsForVectorObjective(M);

  /* REPL only: an anon defined on one line and used as a solver
   * objective on a later line has no in-module call site to trace, so
   * retype any subscripted parameter to ptr at the definition site. */
  if (ReplMode)
    Changed |= retypeAnonsBySubscriptedParam(M);

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
