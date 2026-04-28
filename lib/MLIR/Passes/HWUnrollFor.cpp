// Phase 5.6 Stage F.2 — unroll constant-bound canonical for-
// loops at the MLIR level so downstream passes (Stage F's
// persistent fi-array lowering, in particular) see constant
// subscript indices instead of the f64 iv.
//
// The SV emitter ALREADY renders unrolled-shape for-loops as
// SV `for (int i = ...)` constructs that the synth tool inlines
// — that's the Phase 2 default. But the MLIR-level subscript
// `arr(i)` inside the body uses the iv as a runtime value, which
// breaks Stage F's per-element rewrite for persistent fi-arrays.
//
// This pass IR-level-unrolls every `scf.while` that matches the
// canonical for-loop shape with compile-time constant bounds.
// The body is cloned N times via `mapper`-driven IRMapping, with
// the iv block-arg replaced per-iteration by an `arith.constant
// : f64` of the concrete iteration value. The original loop
// (and its before-region cmpf, after-region addf, and condition)
// is erased after the clones are inlined.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

bool readF64Const(mlir::Value V, double &Out) {
  auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return false;
  if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
    Out = FA.getValueAsDouble();
    return true;
  }
  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
    Out = (double)IA.getInt();
    return true;
  }
  return false;
}

bool tryUnroll(mlir::scf::WhileOp W) {
  HWForLoopInfo Info;
  if (!matchHWForLoop(W, Info)) {
    if (getenv("DEBUG_UNROLL"))
      llvm::errs() << "[unroll] not canonical for-loop\n";
    return false;
  }
  double Init, End, Step;
  if (!readF64Const(Info.Init, Init)) {
    if (getenv("DEBUG_UNROLL"))
      llvm::errs() << "[unroll] init not const\n";
    return false;
  }
  if (!readF64Const(Info.End, End)) {
    if (getenv("DEBUG_UNROLL"))
      llvm::errs() << "[unroll] end not const\n";
    return false;
  }
  if (!readF64Const(Info.Step, Step)) {
    if (getenv("DEBUG_UNROLL"))
      llvm::errs() << "[unroll] step not const\n";
    return false;
  }
  if (Step == 0.0) return false;
  if (getenv("DEBUG_UNROLL"))
    llvm::errs() << "[unroll] init=" << Init << " end=" << End
                 << " step=" << Step << "\n";

  // Compute the concrete iteration values. Cap trip count at a
  // conservative bound to avoid IR blow-up; HWLegalize already
  // warns above 64.
  llvm::SmallVector<double, 16> Vals;
  const int kMaxTrip = 4096;
  if (!Info.IsDecreasing) {
    if (Step <= 0.0) return false;
    for (double V = Init; V <= End + 1e-9 && (int)Vals.size() < kMaxTrip;
         V += Step)
      Vals.push_back(V);
  } else {
    if (Step >= 0.0) return false;
    for (double V = Init; V >= End - 1e-9 && (int)Vals.size() < kMaxTrip;
         V += Step)
      Vals.push_back(V);
  }
  if ((int)Vals.size() == kMaxTrip) return false;
  if (Vals.empty()) {
    // Zero trips: just erase the loop.
    W.erase();
    return true;
  }

  mlir::OpBuilder B(W);
  mlir::Location L = W.getLoc();
  auto F64 = mlir::Float64Type::get(W.getContext());
  // For each iteration value, clone the after-region's body with
  // the iv (block-arg) substituted by an arith.constant.
  mlir::Block &After = W.getAfter().front();
  mlir::Value Iv = After.getArgument(0);

  // Identify the iv-spill slot (if any). Lowering emits at the
  // top of the loop body:
  //   store(iv, %slot)        — defining op may be in the alloca's
  //                              parent function; the store is in
  //                              the after-region.
  //   ... = load(%slot)       — used by body subscripts
  // The spill chain is what makes the body's `arr(i)` subscripts
  // reference the iv via an `f64` load instead of via the iv
  // block-arg directly. Without folding the load back to the
  // per-iteration constant, the subscript indices stay non-const
  // and downstream Stage F bails. Recognize the slot via the
  // single iv-store inside the after-region.
  mlir::Operation *IvSpillSlot = nullptr;
  mlir::Operation *IvSpillStore = nullptr;
  for (mlir::Operation &Op : After) {
    if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(&Op)) {
      if (S.getValue() == Iv) {
        if (auto *Def = S.getAddr().getDefiningOp()) {
          IvSpillSlot = Def;
          IvSpillStore = &Op;
          break;
        }
      }
    }
  }

  for (double V : Vals) {
    mlir::IRMapping Mapping;
    mlir::Value IvConst = mlir::arith::ConstantOp::create(B, L, F64,
        mlir::FloatAttr::get(F64, V));
    Mapping.map(Iv, IvConst);
    // Clone every op in the after-region except: (a) the
    // terminating scf.yield, (b) the iv-step addf feeding the
    // yield, (c) the iv-spill store (if recognized — its only
    // semantic effect was forwarding the iv into a slot for
    // later loads, which we substitute directly), (d) loads of
    // the iv-spill slot (replaced with IvConst via Mapping).
    for (mlir::Operation &Op : After) {
      if (mlir::isa<mlir::scf::YieldOp>(Op)) continue;
      if (auto Add = mlir::dyn_cast<mlir::arith::AddFOp>(Op)) {
        bool OnlyYieldUser = true;
        for (mlir::Operation *U : Add->getUsers()) {
          if (!mlir::isa<mlir::scf::YieldOp>(U)) {
            OnlyYieldUser = false; break;
          }
        }
        if (OnlyYieldUser) continue;
      }
      // Skip the iv-spill store entirely.
      if (IvSpillStore && &Op == IvSpillStore) continue;
      // Replace iv-spill loads with the per-iteration IvConst.
      if (IvSpillSlot) {
        if (auto Ld = mlir::dyn_cast<mlir::LLVM::LoadOp>(&Op)) {
          if (auto *AddrDef = Ld.getAddr().getDefiningOp()) {
            if (AddrDef == IvSpillSlot) {
              Mapping.map(Ld.getResult(), IvConst);
              continue;
            }
          }
        }
      }
      B.clone(Op, Mapping);
    }
  }
  W.erase();
  // After the loop is gone, the iv-spill slot's stores die with
  // the loop body. The slot's alloca lives outside the loop. If
  // it now has no remaining users, erase it — otherwise an f64
  // alloca survives to HWLegalize and trips the "non-synthesizable
  // alloca" check. Leave the slot if any user (load) outside the
  // loop body still exists (e.g. a debug print or the user
  // referenced `i` after the loop).
  if (IvSpillSlot && IvSpillSlot->getResult(0).use_empty())
    IvSpillSlot->erase();
  return true;
}

} // namespace

/// Fold every `arith.fptosi(arith.constant : f64)` pair to a
/// single `arith.constant : i*` of the truncated integer value.
/// Stage D's `buildGepIndex` emits an `arith.fptosi` to convert
/// the loop iv (f64) to an integer for `llvm.getelementptr`. After
/// unroll, the iv becomes an `arith.constant : f64`; the fptosi
/// chain remains as an unfolded pair that the SV emitter doesn't
/// recognize. This local fold keeps each iteration's GEP index a
/// pure integer constant, which feeds Stage F's `readF64Const` /
/// `readIntConst` machinery cleanly.
void foldConstFpToSi(mlir::ModuleOp M) {
  llvm::SmallVector<mlir::arith::FPToSIOp, 8> Worklist;
  M.walk([&](mlir::arith::FPToSIOp Op) { Worklist.push_back(Op); });
  for (auto Op : Worklist) {
    auto C = Op.getIn().getDefiningOp<mlir::arith::ConstantOp>();
    if (!C) continue;
    auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue());
    if (!FA) continue;
    int64_t V = (int64_t)FA.getValueAsDouble();
    auto IT = mlir::dyn_cast<mlir::IntegerType>(Op.getResult().getType());
    if (!IT) continue;
    mlir::OpBuilder B(Op);
    auto NewC = mlir::arith::ConstantOp::create(B, Op.getLoc(), IT,
        mlir::IntegerAttr::get(IT, V));
    Op.getResult().replaceAllUsesWith(NewC.getResult());
    Op.erase();
    if (C.getResult().use_empty()) C.erase();
  }
}

bool runHWUnrollFor(mlir::ModuleOp M) {
  llvm::SmallVector<mlir::scf::WhileOp, 4> Worklist;
  M.walk([&](mlir::scf::WhileOp W) { Worklist.push_back(W); });
  // Iterate until no new loops match — the unroller can expose
  // nested loops that previously had non-const operands depending
  // on outer-loop ivs (rare in our SV path but cheap to handle).
  bool Changed = true;
  while (Changed) {
    Changed = false;
    llvm::SmallVector<mlir::scf::WhileOp, 4> Next;
    for (mlir::scf::WhileOp W : Worklist) {
      if (tryUnroll(W)) Changed = true;
      else Next.push_back(W);
    }
    Worklist = std::move(Next);
    if (Changed) {
      // Re-collect: a cloned body may contain new scf.while ops.
      Worklist.clear();
      M.walk([&](mlir::scf::WhileOp W) { Worklist.push_back(W); });
    }
  }
  // Final cleanup: fold every `fptosi(constant)` pair so the
  // unrolled body has integer-constant GEP indices.
  foldConstFpToSi(M);
  return true;
}

} // namespace mlirgen
} // namespace matlab
