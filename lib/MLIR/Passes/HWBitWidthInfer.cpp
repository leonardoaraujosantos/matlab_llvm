// Bit-width inference / verification for the SystemVerilog backend.
//
// Phase 1 keeps inference inline with verification — every SSA value
// reachable from a synthesizable function must have a type the SV
// emitter can render: `i1` (→ `logic`) or `i8 / i16 / i32 / i64` (→
// `logic [W-1:0]`, signedness inferred at op level). Anything else
// (including `f64`, `none`, pointer, vector, tensor) is rejected here
// so the emitter only sees well-formed types.
//
// The per-value `sv.type` discardable attribute attachment described in
// docs/emit_systemverilog.md is deferred to Phase 5 — it becomes
// load-bearing only once fixed-point binary-point tracking lands.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isAcceptedType(mlir::Type T) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    unsigned W = IT.getWidth();
    return W == 1 || W == 8 || W == 16 || W == 32 || W == 64;
  }
  return false;
}

bool isScriptFunc(mlir::func::FuncOp F) {
  llvm::StringRef N = F.getSymName();
  return N == "script" || N == "main";
}

} // namespace

bool runHWBitWidthInfer(mlir::ModuleOp M,
                        const matlab::SourceManager * /*SM*/) {
  bool Ok = true;
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;
    if (isScriptFunc(F)) return;
    F.walk([&](mlir::Operation *Op) {
      // Skip the func.func op itself; its result types are the function
      // results, validated by HWLegalize.
      if (mlir::isa<mlir::func::FuncOp>(Op)) return;
      // The slot trio (alloca / load / store / gep) carries
      // `!llvm.ptr` results that are conceptually slot / element
      // addresses, not datapath values. Skip them — the SV emitter
      // renders the slot or array as a `logic [W-1:0]` (or
      // `[W-1:0] arr [N]`) declaration whose width comes from the
      // alloca's element type (validated in HWLegalize).
      if (mlir::isa<mlir::LLVM::AllocaOp,
                    mlir::LLVM::LoadOp,
                    mlir::LLVM::StoreOp,
                    mlir::LLVM::GEPOp>(Op)) {
        if (auto Ld = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
          // The loaded value (typed scalar) does need to be checked.
          if (!isAcceptedType(Ld.getResult().getType())) {
            mlir::emitError(Op->getLoc())
                << "load result type '" << Ld.getResult().getType()
                << "' is not synthesizable in Phase 1";
            Ok = false;
          }
        }
        return;
      }
      // For-loop control-flow ops (Phase 2): the canonical bounded
      // for-loop pattern uses an `f64` induction variable inside
      // `arith.cmpf` / `arith.addf` / `scf.condition` / `scf.yield`.
      // These are structural — they emit no datapath value the SV
      // emitter renders directly (the iv lowers to an SV `int` at
      // emission time). Skip the structural roles so the f64 type
      // doesn't trip the `isAcceptedType` check.
      //
      // Phase 4: cmpf can also appear as the datapath state-equality
      // check `cmpf oeq, get_f64(st), <case_label>` — an i1 result
      // that the SV emitter must declare normally. Distinguish the
      // two via the consumer: cmpf used by scf.condition is
      // structural; cmpf used by anything else is datapath.
      if (mlir::isa<mlir::scf::WhileOp, mlir::scf::ConditionOp,
                    mlir::scf::YieldOp>(Op))
        return;
      auto IsForLoopStructural = [](mlir::Operation *Op) {
        if (!Op->getParentOp() ||
            !mlir::isa<mlir::scf::WhileOp>(Op->getParentOp()))
          return false;
        if (mlir::isa<mlir::arith::CmpFOp>(Op)) {
          for (mlir::OpOperand &U : Op->getResult(0).getUses())
            if (mlir::isa<mlir::scf::ConditionOp>(U.getOwner()))
              return true;
          return false;
        }
        if (mlir::isa<mlir::arith::AddFOp>(Op)) {
          for (mlir::OpOperand &U : Op->getResult(0).getUses())
            if (mlir::isa<mlir::scf::YieldOp>(U.getOwner()))
              return true;
          return false;
        }
        return false;
      };
      if (mlir::isa<mlir::arith::CmpFOp, mlir::arith::AddFOp>(Op) &&
          IsForLoopStructural(Op))
        return;
      // arith.constant of f64 is also structural in Phase 2 — it
      // appears only as init/end/step of a recognized for-loop. If a
      // user-function body actually uses an f64 constant in the
      // datapath, HWLegalize already caught it (the parameter type
      // check rejects f64), so this is safe.
      if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
        if (mlir::isa<mlir::FloatType>(C.getType()))
          return;
      }
      // Phase 3: persistent-variable runtime calls produce f64 / none
      // results (the runtime ABI), but the SV emitter routes their
      // uses to integer-typed register signals. Skip type-checking
      // their results here. The state ABI is split between
      // `llvm.call` (isempty / get) and `matlab.call_builtin` (set);
      // both forms surface here.
      if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
        auto C = Call.getCallee();
        if (C && (*C == "matlab_persistent_isempty" ||
                  *C == "matlab_global_get_f64" ||
                  *C == "matlab_global_set_f64"))
          return;
      }
      if (Op->getName().getStringRef() == "matlab.call_builtin") {
        if (auto S = Op->getAttrOfType<mlir::StringAttr>("callee")) {
          llvm::StringRef N = S.getValue();
          if (N == "matlab_persistent_isempty" ||
              N == "matlab_global_get_f64" ||
              N == "matlab_global_set_f64")
            return;
        }
      }
      for (mlir::Value V : Op->getResults()) {
        if (!isAcceptedType(V.getType())) {
          mlir::emitError(Op->getLoc())
              << "value of type '" << V.getType()
              << "' is not synthesizable "
              << "(supported: i1 / i8 / i16 / i32 / i64)";
          Ok = false;
          return;
        }
      }
    });
  });
  return Ok;
}

} // namespace mlirgen
} // namespace matlab
