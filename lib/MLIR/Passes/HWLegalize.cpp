// Synthesizability gate for the SystemVerilog (ASIC) backend.
//
// Walks the post-LowerIO module and emits a source-located error for every
// construct that cannot be mapped to inferable RTL. The pass is run in
// both `-emit-systemverilog` (gate-then-emit) and `-check-synthesizable`
// (gate-only) modes — emission never silently produces broken RTL.
//
// Phase 1 scope (see docs/emit_systemverilog.md → "Synthesizability gate"
// → coverage table). The categories enforced here:
//
//   - any surviving `llvm.call @matlab_*` runtime call (I/O, parfor, eval,
//     cell/struct, file I/O — all unsynthesizable)
//   - recursion (direct or indirect call-graph cycle)
//   - `scf.while` (Phase 1 rejects all; later phases relax for FSM form)
//   - `llvm.alloca` (Phase 1 is scalar-only; arrays are Phase 2)
//   - any user-function parameter / result of `f64` / `f32` (floating
//     point requires explicit `fi(...)` policy)
//   - function handles / anonymous functions surviving to emission
//
// Each diagnostic carries an MLIR `Location` that traces back to the
// original `.m` line through the locs propagated during lowering.

#include "matlab/MLIR/Passes/Passes.h"
#include "matlab/Basic/SourceManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

/// Return true if T is a synthesizable-scalar Phase-1 type: i1 / i8 /
/// i16 / i32 / i64. Floating-point and pointer types are rejected at
/// the boundary; other integer widths (e.g. i7) are not produced by the
/// existing pipeline so we don't enumerate them.
bool isSynthScalar(mlir::Type T) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    unsigned W = IT.getWidth();
    return W == 1 || W == 8 || W == 16 || W == 32 || W == 64;
  }
  return false;
}

/// True for the top-level driver wrapper produced by lowering. Pre-
/// LowerIO it's named `@script`; LowerIO renames it to `@main` so the
/// module is directly linkable. The synthesizability gate runs after
/// LowerIO, so the live name is `@main`. Both names are accepted to
/// stay robust against pipeline reordering.
bool isScriptFunc(mlir::func::FuncOp F) {
  llvm::StringRef N = F.getSymName();
  return N == "script" || N == "main";
}

/// Detect a call-graph cycle reachable from `Start`. Used to reject
/// direct (`f` calls `f`) and indirect (`f` → `g` → `f`) recursion.
bool hasCycleFrom(mlir::ModuleOp M, mlir::func::FuncOp Start) {
  llvm::SmallVector<mlir::func::FuncOp, 8> Stack;
  llvm::DenseSet<mlir::Operation *> OnStack;
  llvm::DenseSet<mlir::Operation *> Visited;
  std::function<bool(mlir::func::FuncOp)> Dfs =
      [&](mlir::func::FuncOp F) -> bool {
    if (OnStack.contains(F.getOperation())) return true;
    if (Visited.contains(F.getOperation())) return false;
    Visited.insert(F.getOperation());
    OnStack.insert(F.getOperation());
    bool Cycle = false;
    F.walk([&](mlir::func::CallOp Call) {
      if (Cycle) return;
      auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
      if (!Tgt) return;
      if (Dfs(Tgt)) Cycle = true;
    });
    OnStack.erase(F.getOperation());
    return Cycle;
  };
  return Dfs(Start);
}

} // namespace

bool runHWLegalize(mlir::ModuleOp M, const matlab::SourceManager * /*SM*/) {
  bool Ok = true;

  // The synthesizability gate runs on every user-defined function —
  // `@script` (the top-level driver wrapper produced by lowering) is
  // not a synthesizable unit and is intentionally skipped. The same
  // applies to its descendants: a `disp(...)` call lives in `@script`
  // and is normal there; the same call inside a user function is a
  // hard error.
  auto walkUserFuncs = [&](auto Fn) {
    M.walk([&](mlir::func::FuncOp F) {
      if (F.empty()) return;
      if (isScriptFunc(F)) return;
      Fn(F);
    });
  };

  // 1. Reject any surviving runtime call inside a user function. After
  //    LowerIO every matlab.* op has either been replaced by arith /
  //    scf / func / llvm.* or is a runtime call — those calls are the
  //    unsynthesizable surface.
  //
  //    Exception (Phase 3): the persistent-variable state ABI
  //    (`matlab_persistent_isempty`, `matlab_global_get_f64`,
  //    `matlab_global_set_f64`) is recognized and rewritten to
  //    inferable registers by HWStateInfer + the SV emitter. Allow
  //    those calls here; reject everything else.
  walkUserFuncs([&](mlir::func::FuncOp F) {
    F.walk([&](mlir::LLVM::CallOp Call) {
      auto Callee = Call.getCallee();
      if (!Callee) {
        mlir::emitError(Call.getLoc())
            << "indirect call is not synthesizable";
        Ok = false;
        return;
      }
      llvm::StringRef Name = *Callee;
      if (Name == "matlab_persistent_isempty" ||
          Name == "matlab_global_get_f64" ||
          Name == "matlab_global_set_f64")
        return;  // recognized state ABI — handled by HWStateInfer
      if (Name.starts_with("matlab_")) {
        mlir::emitError(Call.getLoc())
            << "runtime call '" << Name
            << "' has no synthesizable form (I/O, dynamic allocation, "
               "and runtime dispatch are not RTL-inferable)";
        Ok = false;
      }
    });
  });

  // 1b. Validate every persistent in every user function. Mismatched
  //     widths, missing isempty initializers, and other shape errors
  //     surface here — `gatherHWPersistentState` emits the
  //     diagnostic and returns false.
  walkUserFuncs([&](mlir::func::FuncOp F) {
    llvm::SmallVector<HWPersistentInfo, 4> Persists;
    if (!gatherHWPersistentState(F.getOperation(), Persists))
      Ok = false;
  });

  // 2. `scf.while` falls into two cases after LowerSeqLoops:
  //    a) the canonical bounded for-loop pattern (init/end/step shape
  //       produced by `for i = init:end`) — accepted in Phase 2 when
  //       all three bounds are compile-time constants
  //    b) a true data-dependent while — rejected (Phase 4 territory,
  //       needs FSM extraction)
  //    Phase 2 also checks: the induction variable is *not* used
  //    inside the body as a datapath value. Lowering integer-typed
  //    body uses of `i` is a Phase 4 enhancement; for now, an unused
  //    `i` (the common counter-bumping case) is the only legal shape.
  walkUserFuncs([&](mlir::func::FuncOp F) {
    F.walk([&](mlir::scf::WhileOp W) {
      HWForLoopInfo Info;
      if (!matchHWForLoop(W, Info)) {
        mlir::emitError(W.getLoc())
            << "data-dependent while-loop is not synthesizable in "
            << "Phase 2 (needs explicit FSM form; see "
            << "docs/emit_systemverilog.md)";
        Ok = false;
        return;
      }
      auto IsConst = [](mlir::Value V) {
        return V.getDefiningOp<mlir::arith::ConstantOp>() != nullptr;
      };
      if (!IsConst(Info.Init) || !IsConst(Info.End) || !IsConst(Info.Step)) {
        mlir::emitError(W.getLoc())
            << "for-loop bounds must be compile-time constants in "
            << "Phase 2 (data-dependent trip counts need FSM form)";
        Ok = false;
        return;
      }
      if (!Info.Iv.use_empty()) {
        // The induction variable's only legal uses are inside the
        // matcher's recognized cmpf and addf — those live in the
        // before-region's terminator chain and the after-region's
        // tail respectively. We accept the iv whenever its uses are
        // exactly those structural roles. Anything else (e.g. a body
        // op reading %iv as an operand) is a datapath use.
        for (mlir::OpOperand &U : Info.Iv.getUses()) {
          mlir::Operation *User = U.getOwner();
          if (mlir::isa<mlir::arith::CmpFOp, mlir::arith::AddFOp,
                        mlir::scf::YieldOp,
                        mlir::scf::ConditionOp>(User))
            continue;
          mlir::emitError(User->getLoc())
              << "induction variable used as datapath value is not "
              << "supported in Phase 2 (loop body must not consume "
              << "the loop counter; later phases will lower it to an "
              << "integer)";
          Ok = false;
          break;
        }
      }
    });
  });

  // 3. `llvm.alloca` element types: scalar primitives render as
  //    `logic` declarations + blocking assignments inside
  //    `always_comb`; static arrays of scalar primitives (Phase 4.5.4
  //    `LowerStaticFiArrays`) render as `logic [W-1:0] arr [N];` with
  //    indexed access. Anything else (struct element, runtime ptr) is
  //    out of scope.
  walkUserFuncs([&](mlir::func::FuncOp F) {
    F.walk([&](mlir::LLVM::AllocaOp A) {
      mlir::Type ET = A.getElemType();
      if (isSynthScalar(ET)) return;  // scalar slot
      if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(ET)) {
        if (isSynthScalar(Arr.getElementType())) return;  // static array
      }
      mlir::emitError(A.getLoc())
          << "stack allocation has unsynthesizable element type "
          << "(must be a scalar i1 / i8 / i16 / i32 / i64 or a static "
             "1-D array of those)";
      Ok = false;
    });
  });

  // 4. Walk every user-defined function (non-`@script`) and check
  //    parameter / result types, recursion, and inferred-latch hazards.
  bool HasUserFunc = false;
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;          // forward decl
    if (isScriptFunc(F)) return;    // top-level driver, not synthesizable
    HasUserFunc = true;

    // 4a. Recursion.
    if (hasCycleFrom(M, F)) {
      mlir::emitError(F.getLoc())
          << "function '" << F.getSymName()
          << "' is part of a recursive call cycle "
          << "(recursion is not synthesizable)";
      Ok = false;
    }

    // 4b. Parameter types.
    auto FT = F.getFunctionType();
    for (auto [I, T] : llvm::enumerate(FT.getInputs())) {
      if (!isSynthScalar(T)) {
        mlir::emitError(F.getLoc())
            << "function '" << F.getSymName() << "' parameter " << I
            << " has unsynthesizable type (only i1 / i8 / i16 / i32 / "
               "i64 are supported in Phase 1; floating-point requires "
               "explicit fi(...) conversion)";
        Ok = false;
      }
    }
    // 4c. Result types. Phase 3 relaxation: a function whose result
    //     directly returns a recognized persistent get (`return %g`
    //     where `%g = llvm.call @matlab_global_get_f64(...)`) is
    //     accepted even though the get's type is f64 — the SV
    //     emitter routes the return through the register's native
    //     integer width.
    llvm::SmallVector<HWPersistentInfo, 4> Persists;
    gatherHWPersistentState(F.getOperation(), Persists);
    auto IsPersistentGet = [&](mlir::Value V) {
      auto *Op = V.getDefiningOp();
      if (!Op) return false;
      auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op);
      if (!Call) return false;
      auto C = Call.getCallee();
      return C && *C == "matlab_global_get_f64";
    };
    for (auto [I, T] : llvm::enumerate(FT.getResults())) {
      if (isSynthScalar(T)) continue;
      // Look at every func.return in F; if every result-operand for
      // this index is a persistent get, it's OK.
      bool AllOK = true;
      bool Any = false;
      F.walk([&](mlir::func::ReturnOp R) {
        if (R.getNumOperands() <= I) return;
        Any = true;
        if (!IsPersistentGet(R.getOperand(I))) AllOK = false;
      });
      if (Any && AllOK) continue;
      mlir::emitError(F.getLoc())
          << "function '" << F.getSymName() << "' result " << I
          << " has unsynthesizable type";
      Ok = false;
    }
  });

  if (!HasUserFunc) {
    mlir::emitError(M.getLoc())
        << "no synthesizable functions found "
        << "(top-level scripts are not synthesizable; define a "
        << "`function y = name(...)` whose inputs/outputs are typed "
        << "via fi(...))";
    Ok = false;
  }

  return Ok;
}

} // namespace mlirgen
} // namespace matlab
