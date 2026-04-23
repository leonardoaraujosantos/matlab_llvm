// Lowering of user-defined function calls to MLIR func.call, with
// single-site monomorphization of the callee's signature.
//
// Our frontend emits `matlab.call @fname(args)` for user-defined calls and
// every user func.func starts with `none`-typed parameters and (if the
// function has outputs) `none`-typed results. The MLIR->LLVM conversion
// pipeline has no way to lower `matlab.call` or `none`, so this pass:
//
//   1. Groups every matlab.call in the module by callee symbol.
//   2. For each callee func.func: if all call sites pass consistent arg
//      types and the current signature has any `none`-typed parameters,
//      retype the parameters (and the entry-block arguments) to match.
//      Similarly infer the result types from the operand types of the
//      func.return inside the body.
//   3. Rewrites every matlab.call into a func.call with the refined
//      signature. Consumers of the old (none-typed) result get
//      transparently updated to the new typed value via RAUW.
//
// The result is that programs like `y = sq(3.0); disp(y);` — or the parfor
// variant `parfor i = 1:3; work(i); end` — lower cleanly to LLVM IR.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

// A type is "refinable from none" if the current type is NoneType and the
// proposed type is anything else. We only retype in this direction — never
// narrow an already-concrete type.
bool canRefineTo(Type Old, Type New) {
  if (!Old || !New) return false;
  if (Old == New) return false;
  return mlir::isa<NoneType>(Old);
}

bool isScalarPrimitive(Type T) {
  return mlir::isa<Float64Type, Float32Type, IntegerType>(T);
}

// Forward-propagate concrete operand types into result types for the
// unregistered scalar matlab.* ops we model. Run after retyping block
// arguments so the entry-block arg's f64 type flows through matlab.matmul
// etc. into the eventual func.return operand type, enabling return-type
// refinement.
void propagateScalarTypes(func::FuncOp Fn) {
  /* For self-recursion closure: when `fib(n-1) + fib(n-2)` appears, both
   * operands of the add are matlab.call results to the enclosing function,
   * currently typed `none`. If the enclosing function has a single scalar
   * input type, speculate that the recursive-call results share that type
   * so the add (and downstream return) can refine — which then retypes the
   * signature's return and closes the loop. */
  StringRef SelfName = Fn.getSymName();
  Type SelfInScalar;
  if (Fn.getFunctionType().getNumInputs() >= 1 &&
      isScalarPrimitive(Fn.getFunctionType().getInput(0)))
    SelfInScalar = Fn.getFunctionType().getInput(0);

  auto isSelfCallNoneResult = [&](Value V) -> bool {
    Operation *Def = V.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.call")) return false;
    auto CA = Def->getAttrOfType<StringAttr>("callee");
    if (!CA || CA.getValue() != SelfName) return false;
    return mlir::isa<NoneType>(V.getType());
  };

  bool Changed = true;
  while (Changed) {
    Changed = false;
    Fn.walk([&](Operation *Op) {
      StringRef N = Op->getName().getStringRef();
      // Binary ops on matching scalar operand types.
      if ((N == "matlab.matmul" || N == "matlab.matdiv" ||
           N == "matlab.add"    || N == "matlab.sub"   ||
           N == "matlab.emul"   || N == "matlab.ediv")
          && Op->getNumOperands() == 2 && Op->getNumResults() == 1) {
        Type A = Op->getOperand(0).getType();
        Type B = Op->getOperand(1).getType();
        Type R = Op->getResult(0).getType();
        Type Concrete;
        if (isScalarPrimitive(A)) Concrete = A;
        else if (isScalarPrimitive(B)) Concrete = B;
        /* Self-recursion speculation: both operands are self-calls with
         * `none` results, and we know the function's scalar input type. */
        if (!Concrete && SelfInScalar &&
            isSelfCallNoneResult(Op->getOperand(0)) &&
            isSelfCallNoneResult(Op->getOperand(1)))
          Concrete = SelfInScalar;
        if (Concrete && canRefineTo(R, Concrete)) {
          // Optimistically refine the result to the concrete side even if
          // only one operand is concretely typed. The other is typically a
          // `none`-typed return from a recursive call; refining the result
          // lets the function's return type settle to Concrete, which then
          // feeds back into the recursive call site and closes the loop.
          Op->getResult(0).setType(Concrete);
          Changed = true;
        }
      }
      // Unary ops preserving operand type.
      if ((N == "matlab.neg" || N == "matlab.uplus")
          && Op->getNumOperands() == 1 && Op->getNumResults() == 1) {
        Type A = Op->getOperand(0).getType();
        Type R = Op->getResult(0).getType();
        if (isScalarPrimitive(A) && canRefineTo(R, A)) {
          Op->getResult(0).setType(A);
          Changed = true;
        }
      }
      // matlab.load through a slot whose type is now concrete can produce
      // a concretely-typed value. The slot's "type" is the alloc's result
      // type, which we only trust as a hint — if the load result is none
      // but the slot result is concrete and they disagree, refine.
      if (N == "matlab.load" && Op->getNumOperands() == 1 &&
          Op->getNumResults() == 1) {
        Type SlotTy = Op->getOperand(0).getType();
        Type R      = Op->getResult(0).getType();
        if (isScalarPrimitive(SlotTy) && canRefineTo(R, SlotTy)) {
          Op->getResult(0).setType(SlotTy);
          Changed = true;
        }
      }
      // A matlab.alloc with `none` result whose every matlab.store consumer
      // stores a value of the same concrete scalar type can be retyped.
      // This is what lifts a `none`-typed slot to f64 once parameter
      // spills flow concrete types into it.
      if (N == "matlab.alloc" && Op->getNumResults() == 1 &&
          mlir::isa<NoneType>(Op->getResult(0).getType())) {
        Type Stored;
        bool Consistent = true;
        bool Any = false;
        for (OpOperand &Use : Op->getResult(0).getUses()) {
          Operation *U = Use.getOwner();
          if (isMatlabOp(U, "matlab.store") && U->getNumOperands() == 2 &&
              U->getOperand(1) == Op->getResult(0)) {
            Type T = U->getOperand(0).getType();
            if (!isScalarPrimitive(T)) { Consistent = false; break; }
            if (!Any) { Stored = T; Any = true; }
            else if (Stored != T) { Consistent = false; break; }
          }
        }
        if (Any && Consistent) {
          Op->getResult(0).setType(Stored);
          Changed = true;
        }
      }
    });
  }
}

// Update existing func.call ops whose result types no longer match their
// callee's current signature.
void refreshFuncCalls(ModuleOp M) {
  llvm::SmallVector<func::CallOp> Stale;
  M.walk([&](func::CallOp Call) {
    auto Fn = M.lookupSymbol<func::FuncOp>(Call.getCallee());
    if (!Fn) return;
    auto Sig = Fn.getFunctionType().getResults();
    if (Call.getNumResults() != Sig.size()) { Stale.push_back(Call); return; }
    for (unsigned i = 0; i < Call.getNumResults(); ++i) {
      if (Call.getResult(i).getType() != Sig[i]) {
        Stale.push_back(Call); return;
      }
    }
  });
  for (auto Call : Stale) {
    auto Fn = M.lookupSymbol<func::FuncOp>(Call.getCallee());
    if (!Fn) continue;
    OpBuilder B(Call);
    auto NewCall = func::CallOp::create(B, Call.getLoc(),
                                         Fn.getFunctionType().getResults(),
                                         Call.getCallee(),
                                         Call.getOperands());
    unsigned NCopy = std::min((unsigned)Call.getNumResults(),
                              (unsigned)NewCall.getNumResults());
    for (unsigned i = 0; i < NCopy; ++i)
      Call.getResult(i).replaceAllUsesWith(NewCall.getResult(i));
    Call.erase();
  }
}

} // namespace

/* Lower every matlab.nargin / matlab.nargout inside each func.func to
 * an arith.constant. The value is taken from the optional
 * 'matlab.nargin_value' / 'matlab.nargout_value' attribute on the
 * enclosing func (set by the monomorphiser when cloning for a
 * particular call-site arity), or the function's declared arity as a
 * fallback. */
bool runLowerNarginNargout(ModuleOp M) {
  auto F64 = Float64Type::get(M.getContext());
  bool Changed = false;
  M.walk([&](func::FuncOp Fn) {
    int64_t NarginVal = (int64_t)Fn.getFunctionType().getNumInputs();
    int64_t NargoutVal = (int64_t)Fn.getFunctionType().getNumResults();
    if (auto A = Fn->getAttrOfType<IntegerAttr>("matlab.nargin_value"))
      NarginVal = A.getValue().getSExtValue();
    if (auto A = Fn->getAttrOfType<IntegerAttr>("matlab.nargout_value"))
      NargoutVal = A.getValue().getSExtValue();
    SmallVector<Operation *> Dead;
    Fn.walk([&](Operation *Op) {
      StringRef N = Op->getName().getStringRef();
      if (N != "matlab.nargin" && N != "matlab.nargout") return;
      if (Op->getNumResults() != 1) return;
      OpBuilder B(Op);
      int64_t V = (N == "matlab.nargin") ? NarginVal : NargoutVal;
      auto C = arith::ConstantOp::create(
          B, Op->getLoc(), F64, FloatAttr::get(F64, (double)V));
      Op->getResult(0).replaceAllUsesWith(C.getResult());
      Dead.push_back(Op);
    });
    for (Operation *Op : Dead) { Op->erase(); Changed = true; }
  });
  /* Script-level matlab.nargin/nargout (outside a func.func) fold to 0. */
  M.walk([&](Operation *Op) {
    StringRef N = Op->getName().getStringRef();
    if (N != "matlab.nargin" && N != "matlab.nargout") return;
    if (Op->getNumResults() != 1) return;
    OpBuilder B(Op);
    auto C = arith::ConstantOp::create(
        B, Op->getLoc(), F64, FloatAttr::get(F64, 0.0));
    Op->getResult(0).replaceAllUsesWith(C.getResult());
    Op->erase();
    Changed = true;
  });
  return Changed;
}

/* Clone a user function per distinct concrete call-site signature when
 * it's polymorphically used. Runs after LowerTensorOps so call sites
 * have collapsed tensor types to !llvm.ptr — the only real dispatch
 * distinction remaining is f64 vs ptr, with f64 covering scalar calls
 * and ptr covering any matrix shape. */
bool runMonomorphiseUserCalls(ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  auto I64 = IntegerType::get(Ctx, 64);
  /* Gather matlab.call sites that remain (untyped-path). */
  llvm::DenseMap<StringRef, llvm::SmallVector<Operation *>> Sites;
  M.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call")) return;
    auto CA = Op->getAttrOfType<StringAttr>("callee");
    if (!CA) return;
    Sites[CA.getValue()].push_back(Op);
  });
  bool Changed = false;
  for (auto &[Name, S] : Sites) {
    auto Fn = M.lookupSymbol<func::FuncOp>(Name);
    if (!Fn) continue;
    unsigned DeclArity = Fn.getFunctionType().getNumInputs();
    /* Bucket by (user-visible arity, operand-type tuple). UserArity
     * captures what the programmer wrote at the call site (number of
     * args before the frontend's variadic-packing step), so nargin
     * inside the clone reflects the original call. ActualArity ==
     * Tys.size() is the operand count after packing and drives
     * signature sizing. For non-variadic callees the two agree. */
    struct SigKey {
      unsigned UserArity = 0;
      llvm::SmallVector<Type, 4> Tys;
      bool operator==(const SigKey &O) const {
        return UserArity == O.UserArity && Tys == O.Tys;
      }
    };
    struct SigHash {
      size_t operator()(const SigKey &K) const {
        size_t h = std::hash<unsigned>{}(K.UserArity);
        for (Type T : K.Tys)
          h ^= mlir::hash_value(T) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
      }
    };
    std::unordered_map<SigKey, llvm::SmallVector<Operation *>, SigHash>
        Buckets;
    for (Operation *C : S) {
      unsigned N = C->getNumOperands();
      if (N > DeclArity) continue; /* too many args — user error, skip */
      SigKey K;
      if (auto UA = C->getAttrOfType<IntegerAttr>("user_arity"))
        K.UserArity = (unsigned)UA.getValue().getSExtValue();
      else
        K.UserArity = N;
      bool AllConcrete = true;
      for (unsigned i = 0; i < N; ++i) {
        Type T = C->getOperand(i).getType();
        if (mlir::isa<NoneType>(T)) AllConcrete = false;
        K.Tys.push_back(T);
      }
      if (!AllConcrete) continue;
      Buckets[K].push_back(C);
    }
    if (Buckets.empty()) continue;

    /* The original function keeps serving the bucket whose UserArity
     * matches its current nargin_value (or the declared arity if
     * unset). If no bucket matches, ALL buckets clone — we don't want
     * to overwrite the original's nargin_value when it's already in
     * use by earlier-converted func.call sites. */
    int64_t CurrentNargin = (int64_t)DeclArity;
    if (auto A = Fn->getAttrOfType<IntegerAttr>("matlab.nargin_value"))
      CurrentNargin = A.getValue().getSExtValue();
    SigKey KeepKey;
    bool HaveKeep = false;
    for (auto &[K, _] : Buckets) {
      if ((int64_t)K.UserArity == CurrentNargin) {
        KeepKey = K; HaveKeep = true; break;
      }
    }

    OpBuilder B(Ctx);
    unsigned CloneIdx = 0;
    for (auto &[K, BSites] : Buckets) {
      bool IsKeep = HaveKeep && (K == KeepKey);
      func::FuncOp Target;
      if (IsKeep) {
        Target = Fn;
      } else {
        std::string CloneName = (Name.str() + "__s" +
                                 std::to_string(CloneIdx++));
        while (M.lookupSymbol(CloneName)) CloneName += "_";
        B.setInsertionPointToEnd(M.getBody());
        Target = mlir::cast<func::FuncOp>(B.clone(*Fn.getOperation()));
        Target.setSymName(CloneName);
        for (Operation *C : BSites)
          C->setAttr("callee", StringAttr::get(Ctx, CloneName));
        Changed = true;
      }

      /* nargin_value = user-visible arity (pre-packing). */
      Target->setAttr(
          "matlab.nargin_value",
          IntegerAttr::get(I64, (int64_t)K.UserArity));

      /* Signature / block args match the ACTUAL operand count
       * (post-packing), which is Tys.size(). Trailing declared
       * params beyond that are padded to f64 at the call site; in
       * the signature we default them to f64 too. */
      unsigned ActualArity = (unsigned)K.Tys.size();
      SmallVector<Type, 4> AllInputs(K.Tys.begin(), K.Tys.end());
      auto F64 = Float64Type::get(Ctx);
      for (unsigned i = ActualArity; i < DeclArity; ++i)
        AllInputs.push_back(F64);
      if (!Target.empty()) {
        Block &Entry = Target.getBody().front();
        for (unsigned i = 0; i < Entry.getNumArguments() &&
                             i < AllInputs.size(); ++i) {
          Entry.getArgument(i).setType(AllInputs[i]);
        }
      }
      Target.setFunctionType(FunctionType::get(Ctx, AllInputs,
          Target.getFunctionType().getResults()));
    }
  }
  return Changed;
}

bool runLowerUserCalls(ModuleOp M) {
  MLIRContext *Ctx = M.getContext();
  bool Changed = false;

  // --- 1. Gather matlab.call sites, grouped by callee -------------------
  llvm::DenseMap<StringRef, llvm::SmallVector<Operation *>> Calls;
  M.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call")) return;
    auto CA = Op->getAttrOfType<StringAttr>("callee");
    if (!CA) return;
    Calls[CA.getValue()].push_back(Op);
  });


  // --- 2. Retype signatures --------------------------------------------
  for (auto &[Name, Sites] : Calls) {
    auto Fn = M.lookupSymbol<func::FuncOp>(Name);
    if (!Fn) continue;
    auto OldType = Fn.getFunctionType();

    // a) Check arg-type consensus across sites. Skip if arity disagrees.
    unsigned NumIn = OldType.getNumInputs();
    if (NumIn == 0 || Sites.empty()) {
      // Still useful for return-type refinement below, so don't `continue`.
    }
    bool Compatible = true;
    llvm::SmallVector<Type, 4> NewInputs(OldType.getInputs().begin(),
                                          OldType.getInputs().end());
    for (Operation *C : Sites) {
      if (C->getNumOperands() != NumIn) { Compatible = false; break; }
      for (unsigned i = 0; i < NumIn; ++i) {
        Type CallTy = C->getOperand(i).getType();
        Type ExistingNew = NewInputs[i];
        if (canRefineTo(ExistingNew, CallTy)) {
          NewInputs[i] = CallTy;
        } else if (ExistingNew != CallTy &&
                   !mlir::isa<NoneType>(CallTy)) {
          // Two concrete types disagree — skip retyping this arg.
          Compatible = false;
          break;
        }
      }
      if (!Compatible) break;
    }

    // b) Infer result types from func.return operand types if currently `none`.
    llvm::SmallVector<Type, 4> NewResults(OldType.getResults().begin(),
                                           OldType.getResults().end());
    Fn.walk([&](func::ReturnOp Ret) {
      if (Ret.getNumOperands() != NewResults.size()) return;
      for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
        Type RetTy = Ret.getOperand(i).getType();
        if (canRefineTo(NewResults[i], RetTy))
          NewResults[i] = RetTy;
      }
    });

    /* Always run body-local scalar type propagation FIRST (before the
     * Compatible-gated exit): it refines matlab.alloc / matlab.load
     * result types from store operand types regardless of whether
     * the call-site consensus passed. This matters for per-arity
     * clones whose sites carry fewer args than declared — they flip
     * Compatible to false, but the clone's body still needs its
     * slot types propagated from its (already-retyped) block args. */
    propagateScalarTypes(Fn);
    NewResults.assign(Fn.getFunctionType().getResults().begin(),
                      Fn.getFunctionType().getResults().end());
    Fn.walk([&](func::ReturnOp Ret) {
      if (Ret.getNumOperands() != NewResults.size()) return;
      for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
        Type RetTy = Ret.getOperand(i).getType();
        if (canRefineTo(NewResults[i], RetTy))
          NewResults[i] = RetTy;
      }
    });

    // c) Apply signature update if anything actually changed.
    if (!Compatible) continue;
    bool InputsChanged = false, ResultsChanged = false;
    for (unsigned i = 0; i < NumIn; ++i)
      if (NewInputs[i] != OldType.getInput(i)) InputsChanged = true;
    for (unsigned i = 0; i < NewResults.size(); ++i)
      if (NewResults[i] != OldType.getResult(i)) ResultsChanged = true;

    /* (propagateScalarTypes moved above the Compatible-gated exit so
     * per-arity clones still get their body types refined.) */
    propagateScalarTypes(Fn);
    NewResults.assign(Fn.getFunctionType().getResults().begin(),
                      Fn.getFunctionType().getResults().end());
    Fn.walk([&](func::ReturnOp Ret) {
      if (Ret.getNumOperands() != NewResults.size()) return;
      for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
        Type RetTy = Ret.getOperand(i).getType();
        if (canRefineTo(NewResults[i], RetTy))
          NewResults[i] = RetTy;
      }
    });
    ResultsChanged = false;
    for (unsigned i = 0; i < NewResults.size(); ++i)
      if (NewResults[i] != Fn.getFunctionType().getResult(i))
        ResultsChanged = true;

    if (!InputsChanged && !ResultsChanged) continue;

    // Apply param changes first (they enable body propagation), then
    // propagate scalar types through the body, then re-read return types.
    if (InputsChanged) {
      auto InterimType =
          FunctionType::get(Ctx, NewInputs, OldType.getResults());
      Fn.setFunctionType(InterimType);
      if (!Fn.empty()) {
        Block &Entry = Fn.getBody().front();
        for (unsigned i = 0; i < NumIn; ++i) {
          if (Entry.getArgument(i).getType() != NewInputs[i])
            Entry.getArgument(i).setType(NewInputs[i]);
        }
      }
      propagateScalarTypes(Fn);

      // Re-examine return types after propagation.
      NewResults.assign(Fn.getFunctionType().getResults().begin(),
                        Fn.getFunctionType().getResults().end());
      Fn.walk([&](func::ReturnOp Ret) {
        if (Ret.getNumOperands() != NewResults.size()) return;
        for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
          Type RetTy = Ret.getOperand(i).getType();
          if (canRefineTo(NewResults[i], RetTy))
            NewResults[i] = RetTy;
        }
      });
      ResultsChanged = false;
      for (unsigned i = 0; i < NewResults.size(); ++i)
        if (NewResults[i] != Fn.getFunctionType().getResult(i))
          ResultsChanged = true;
    }

    if (ResultsChanged) {
      auto FinalType = FunctionType::get(
          Ctx, Fn.getFunctionType().getInputs(), NewResults);
      Fn.setFunctionType(FinalType);
    }
    Changed = true;
  }

  // Existing func.call ops may now have stale result types; re-emit them.
  refreshFuncCalls(M);

  // --- 3. Convert matlab.call → func.call ------------------------------
  // Re-collect in case earlier updates invalidated references.
  llvm::SmallVector<Operation *> ToErase;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call")) ToErase.push_back(Op);
  });
  for (Operation *Call : ToErase) {
    auto CA = Call->getAttrOfType<StringAttr>("callee");
    if (!CA) continue;
    auto Fn = M.lookupSymbol<func::FuncOp>(CA.getValue());
    if (!Fn) continue;
    auto FnTy = Fn.getFunctionType();
    unsigned N = Call->getNumOperands();
    unsigned M_ = FnTy.getNumInputs();
    if (N > M_) continue; /* too many args — skip */

    /* Skip this call if the user-visible arity (from the frontend's
     * variadic-packing attribute) doesn't match the callee's
     * matlab.nargin_value. The monomorphiser will clone the callee
     * with a matching nargin_value attribute first. Without this
     * skip, two calls to a variadic function with different user
     * arities but the same post-packing operand types would both
     * get converted to func.call of the same function — losing the
     * per-callsite nargin distinction. */
    if (auto UA = Call->getAttrOfType<IntegerAttr>("user_arity")) {
      int64_t CallUA = UA.getValue().getSExtValue();
      int64_t FnUA = (int64_t)M_;
      if (auto FA = Fn->getAttrOfType<IntegerAttr>("matlab.nargin_value"))
        FnUA = FA.getValue().getSExtValue();
      if (CallUA != FnUA) continue;
    }

    // Only convert if the first N operand types match the leading N
    // inputs of the callee's signature exactly. Missing trailing args
    // (fewer-arg calls) get padded with 0.0 or null-ptr depending on
    // the declared type.
    bool OK = true;
    for (unsigned i = 0; i < N; ++i) {
      if (FnTy.getInput(i) != Call->getOperand(i).getType()) {
        OK = false; break;
      }
    }
    if (!OK) continue;

    OpBuilder B(Call);
    llvm::SmallVector<Value, 4> Args(Call->operand_begin(),
                                     Call->operand_end());
    /* Pad missing trailing args. Scalar f64 -> 0.0; ptr-typed -> null
     * pointer constant. Any other declared type we leave unpadded and
     * skip the rewrite — later passes or a proper error will surface. */
    for (unsigned i = N; i < M_; ++i) {
      Type T = FnTy.getInput(i);
      if (mlir::isa<Float64Type>(T)) {
        Args.push_back(arith::ConstantOp::create(
            B, Call->getLoc(), T, FloatAttr::get(T, 0.0)));
      } else if (T == LLVM::LLVMPointerType::get(M.getContext())) {
        Args.push_back(LLVM::ZeroOp::create(
            B, Call->getLoc(), T));
      } else {
        OK = false; break;
      }
    }
    if (!OK) continue;
    auto NewCall = func::CallOp::create(B, Call->getLoc(),
                                        FnTy.getResults(),
                                        CA.getValue(), Args);

    // Replace uses of old matlab.call results. If result-arity changed,
    // use the min of both.
    unsigned NCopy = std::min((unsigned)Call->getNumResults(),
                              (unsigned)NewCall->getNumResults());
    for (unsigned i = 0; i < NCopy; ++i) {
      Call->getResult(i).replaceAllUsesWith(NewCall->getResult(i));
    }
    Call->erase();
    Changed = true;
  }

  return Changed;
}

} // namespace mlirgen
} // namespace matlab
