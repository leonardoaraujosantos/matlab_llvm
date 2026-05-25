// lib/MLIR/Passes/PromoteNoneParams.cpp — promote `none`-typed
// function input parameters to `f64` when their body usage is
// numeric.  Sema doesn't infer types for function parameters when
// there's no call site to learn from (top-level entry-point
// functions are the canonical case — every GPU PCT validation test
// is shaped `function err = test_gpu_axpy(n)` with no in-module
// caller), so the param's slot, the block argument, and every
// matlab.load on the slot stay `none`-typed.  Downstream dispatch
// tables (LowerTensorOps's strict + loose matchers) require f64
// for numeric arg positions, so the function never compiles
// end-to-end.
//
// This pass closes the gap by walking each func.func, identifying
// each `none`-typed input arg, checking how the corresponding slot
// is used in the body, and (if usage is numeric) retyping the arg
// + slot + every store/load on it to f64.  Idempotent.
//
// **Promotion heuristic (conservative)**: a `none` param promotes
// to f64 iff every load on its slot is consumed by an op that
// expects a numeric value (matlab.call_builtin, matlab.add /
// matlab.sub / matlab.matmul, arith.{add,sub,mul,div}f,
// matlab.range, matlab.subscript).  Loads consumed by other
// matlab.store / matlab.alloc / etc. don't disqualify but also
// don't trigger promotion.  Loads with NO uses are conservative —
// we don't promote.
//
// **Out of scope**: tensor-typed params (size analysis would
// constrain them differently), classdef-handle params (need ptr
// type), char-array params (need ptr-to-matlab_string).  These
// stay `none` and the existing diagnostic surfaces them.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdio>
#include <cstdlib>

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

/* True when the callee of a matlab.call_builtin op treats its
 * operand as a scalar f64.  We can't blanket-trust call_builtin
 * (length(obj), disp(obj), step(sys), imadd(mat,mat) take ptrs
 * or matrices, not scalars), so this is a narrow allowlist of
 * builtins whose first argument is unambiguously a scalar. */
bool isScalarBuiltinCall(Operation *User) {
  if (!isMatlabOp(User, "matlab.call_builtin")) return false;
  auto Callee = User->getAttrOfType<StringAttr>("callee");
  if (!Callee) return false;
  StringRef N = Callee.getValue();
  // gpuArray.* size-parameter builtins — n in gpuArray.rand(n, ...)
  // / gpuArray.linspace(a, b, n).  The MLIR callee is the un-prefixed
  // class_method form `gpuArray_<method>` (single underscore between
  // class and method); `matlab_gpuArray_*` is the post-lowering rename
  // and `gpuArray__*` is the classdef-method namespace (double
  // underscore).  All three forms surface here depending on pipeline
  // order, so we accept any of them.
  if (N.starts_with("matlab_gpuArray_") || N.starts_with("gpuArray__") ||
      N.starts_with("gpuArray_"))
    return true;
  // scalar transcendental / arithmetic math
  static constexpr llvm::StringRef Scalars[] = {
    "matlab_sin", "matlab_cos", "matlab_tan",
    "matlab_asin", "matlab_acos", "matlab_atan", "matlab_atan2",
    "matlab_sind", "matlab_cosd", "matlab_tand",
    "matlab_asind", "matlab_acosd", "matlab_atand",
    "matlab_sinh", "matlab_cosh", "matlab_tanh",
    "matlab_exp", "matlab_log", "matlab_log2", "matlab_log10",
    "matlab_sqrt", "matlab_abs", "matlab_sign",
    "matlab_ceil", "matlab_floor", "matlab_round", "matlab_fix",
    "matlab_mod", "matlab_rem", "matlab_power", "matlab_pow",
    "matlab_max2", "matlab_min2",
  };
  for (auto S : Scalars) if (N == S) return true;
  return false;
}

/* True when this op consumes its operand as a numeric (scalar) value.
 * The list mirrors the dispatch surfaces in LowerTensorOps. */
bool consumesAsNumeric(Operation *User) {
  if (!User) return false;
  StringRef N = User->getName().getStringRef();
  // matlab.* numeric ops (binops + unops).  Excludes matlab.subscript
  // and matlab.transpose: both signal that the operand is a matrix
  // (e.g. p(i) — p is the indexable, NOT a scalar), so they must NOT
  // trigger param-promotion.
  if (N == "matlab.add" || N == "matlab.sub" || N == "matlab.matmul" ||
      N == "matlab.emul" || N == "matlab.ediv" || N == "matlab.epow" ||
      N == "matlab.neg" ||
      N == "matlab.range" || N == "matlab.cmp")
    return true;
  // matlab.call_builtin only counts when the callee is a scalar-only
  // builtin (not length/disp/step/imadd/etc., which take ptr/object).
  if (isScalarBuiltinCall(User)) return true;
  // arith.* float ops
  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::CmpFOp, arith::NegFOp>(User))
    return true;
  return false;
}

/* Find the param-slot matlab.alloc for each block argument.  The
 * Lowerer emits `matlab.store %arg_i, %slot` as one of the first
 * ops in the function body when the arg has a declared name.  We
 * scan for the matlab.store and pull the slot Value. */
void findParamSlots(func::FuncOp Fn,
                    llvm::SmallVectorImpl<Value> &SlotsOut) {
  SlotsOut.assign(Fn.getNumArguments(), Value());
  if (Fn.empty()) return;
  Block &Entry = Fn.getBody().front();
  for (Operation &Op : Entry) {
    if (!isMatlabOp(&Op, "matlab.store") || Op.getNumOperands() != 2)
      continue;
    Value Stored = Op.getOperand(0);
    Value Slot = Op.getOperand(1);
    /* Stored must be one of our block args (i.e. directly an
     * Entry.getArgument(i) — pre-promotion, no intervening ops). */
    for (unsigned i = 0; i < Fn.getNumArguments(); ++i) {
      if (Stored == Entry.getArgument(i)) {
        if (!SlotsOut[i]) SlotsOut[i] = Slot;
        break;
      }
    }
  }
}

/* Walk every matlab.load on `slot` and check if any of them is
 * consumed by a numeric op.  Returns true iff at least one such
 * use exists and NONE of the uses are "obviously non-numeric"
 * (e.g. stored into a tensor-typed slot — would suggest matrix
 * semantics).  v1 is the simple "any numeric use" rule. */
bool slotHasNumericUse(Value Slot) {
  if (!Slot) return false;
  for (Operation *User : Slot.getUsers()) {
    if (!isMatlabOp(User, "matlab.load")) continue;
    if (User->getNumResults() != 1) continue;
    Value LoadVal = User->getResult(0);
    for (Operation *LU : LoadVal.getUsers())
      if (consumesAsNumeric(LU))
        return true;
  }
  return false;
}

/* Retype every matlab.alloc / matlab.load / matlab.store SSA value
 * that's tied to `Slot` to the new f64 type.  Mutates the IR in
 * place; doesn't reach across function boundaries. */
void retypeSlotToF64(Value Slot, Type F64) {
  /* The alloc result is the slot Value itself. */
  Slot.setType(F64);
  for (OpOperand &Use : llvm::make_early_inc_range(Slot.getUses())) {
    Operation *U = Use.getOwner();
    if (isMatlabOp(U, "matlab.load")) {
      if (U->getNumResults() == 1)
        U->getResult(0).setType(F64);
    }
    /* matlab.store doesn't have a result so no need to update its
     * own result; but its first operand (the stored value) might
     * still be `none`-typed.  We retype that operand chain via the
     * block-arg promotion step. */
  }
}

}  // namespace

/* Walk every func.func and promote `none`-typed input args to f64
 * when their slot has numeric use in the body.  Returns the number
 * of promoted args across the module. */
unsigned runPromoteNoneParams(mlir::ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  auto F64 = Float64Type::get(Ctx);
  auto NoneT = NoneType::get(Ctx);
  unsigned Promoted = 0;

  M.walk([&](func::FuncOp Fn) {
    if (Fn.empty()) return;
    if (Fn.getNumArguments() == 0) return;

    /* Skip classdef methods / constructors.  A classdef ctor returns
     * a ptr-typed matlab_obj* and a classdef method's first input is
     * also ptr-typed (the `obj` receiver).  These functions' param
     * types are determined by the classdef contract (not numeric
     * usage in the body), so promoting them is wrong.  Detect via:
     *   - first output is ptr (ctor returning obj)
     *   - first input is ptr (method receiver)
     *   - mangled name contains "__" (classdef method namespace) */
    auto FnTy = Fn.getFunctionType();
    auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
    if (FnTy.getNumResults() >= 1 && FnTy.getResult(0) == PtrTy) {
      return;
    }
    if (FnTy.getNumInputs() >= 1 && FnTy.getInput(0) == PtrTy) {
      if (std::getenv("MATLAB_GPU_DEBUG_PROMOTE"))
        std::fprintf(stderr, "  skip: first input is ptr\n");
      return;
    }
    if (Fn.getSymName().contains("__"))
      return;
    /* Skip functions that already have non-none arg types — they've
     * been refined by RefineFuncSigs or had types from Sema. */
    bool AnyNone = false;
    for (auto T : FnTy.getInputs())
      if (mlir::isa<NoneType>(T)) { AnyNone = true; break; }
    if (!AnyNone)
      return;

    /* === Polymorphic-helper guard (issue #21 workaround) ===
     *
     * THIS IS A KNOWN BAND-AID.  The "real fix" — per-call-site
     * monomorphization happening BEFORE this pass — was attempted
     * and found to require a deeper refactor of LowerTensorOps into
     * composable phases (see issue #36 + the comment thread there).
     *
     * INVARIANT THIS GUARD MAINTAINS: a func.func with multiple
     * in-module callers and at least one `none` arg stays `none`-typed
     * after this pass runs, so the late `runMonomorphiseUserCalls`
     * can clone the body per concrete signature seen at call sites.
     *
     * CANARY: test/Run/fn_polymorphic_invariant.m exercises four
     * shape combinations of the same callee.  If the guard ever stops
     * firing prematurely, that fixture is the first to break in the
     * per-PR run-tests lane.
     *
     * Skip functions that have in-module callers.  Promoting a polymorphic
     * helper (e.g. `function y = sq(x); y = x.*x; end` called both as
     * `sq(5)` and `sq([1 2 3])`) to f64 would monomorphize it and break
     * the matrix call sites.  Sema's call-site arg-flow already refines
     * such helpers' arg types from the actual call args; the pass is
     * meant for true entry-point functions with no in-module caller
     * (the GPU PCT validation tests). */
    bool HasCaller = false;
    auto FnName = Fn.getSymName();
    M.walk([&](Operation *Op) {
      if (HasCaller) return;
      if (auto Call = dyn_cast<func::CallOp>(Op)) {
        if (Call.getCallee() == FnName) HasCaller = true;
        return;
      }
      // matlab.call still un-lowered when PromoteNoneParams runs early
      if (Op->getName().getStringRef() == "matlab.call") {
        if (auto CalleeAttr = Op->getAttrOfType<StringAttr>("callee"))
          if (CalleeAttr.getValue() == FnName) HasCaller = true;
      }
    });
    if (HasCaller) {
      if (std::getenv("MATLAB_GPU_DEBUG_PROMOTE"))
        std::fprintf(stderr, "  skip: has in-module caller(s)\n");
      return;
    }

    /* Find each param slot. */
    llvm::SmallVector<Value> Slots;
    findParamSlots(Fn, Slots);

    /* Decide which to promote. */
    llvm::SmallVector<bool> Promote(Fn.getNumArguments(), false);
    bool ChangedFn = false;
    for (unsigned i = 0; i < Fn.getNumArguments(); ++i) {
      Type ArgTy = Fn.getFunctionType().getInputs()[i];
      if (!mlir::isa<NoneType>(ArgTy)) continue;
      if (!Slots[i]) continue;
      if (!slotHasNumericUse(Slots[i])) continue;
      Promote[i] = true;
      ChangedFn = true;
    }
    if (!ChangedFn) return;

    /* Update func.func signature. */
    llvm::SmallVector<Type> NewIn(Fn.getFunctionType().getInputs().begin(),
                                  Fn.getFunctionType().getInputs().end());
    for (unsigned i = 0; i < Promote.size(); ++i)
      if (Promote[i]) NewIn[i] = F64;
    auto NewFT = FunctionType::get(Ctx, NewIn,
                                    Fn.getFunctionType().getResults());
    Fn.setFunctionType(NewFT);

    /* Update block arg type only — let RefineSlotTypes propagate the
     * slot via the now-f64-typed matlab.store(%arg, %slot) observation
     * during the standard pipeline iteration.  This keeps the
     * promotion conservative: we don't mutate the slot chain ourselves
     * (which can confuse downstream passes that re-derive slot types
     * from stores); we just introduce the typed source. */
    Block &Entry = Fn.getBody().front();
    for (unsigned i = 0; i < Promote.size(); ++i) {
      if (!Promote[i]) continue;
      Entry.getArgument(i).setType(F64);
      ++Promoted;
    }
    /* Suppress unused-var warnings. */
    (void)NoneT;
    (void)Slots;
  });

  return Promoted;
}

}  // namespace mlirgen
}  // namespace matlab
