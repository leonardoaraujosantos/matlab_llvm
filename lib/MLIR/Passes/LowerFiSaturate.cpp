// Phase 5.1 — fixed-point saturate semantics.
//
// Replaces every runtime-call `matlab_fi_sat_s64(val, W)` and
// `matlab_fi_sat_u64(val, W)` with an explicit clamp circuit
// built from `arith.cmpi` + `arith.select`. The SV emitter
// renders the chain as ternary expressions (`(val > MAX) ? MAX
// : (val < MIN ? MIN : val)`) which synthesize to a comparator
// + 2-way mux per bound.
//
// This replaces the earlier "passthrough" DCE step in
// LowerStaticFiArrays.cpp, which was correct only for
// Wrap-mode fi (the trunci downstream produces the same value
// as the saturate for non-overflowing inputs). For Saturate
// mode (the MATLAB Coder default), passthrough silently
// changed semantics on overflow. The explicit clamp gives the
// user-asked Saturate semantic regardless.
//
// Width W is read from the second operand of the runtime call,
// which is always an i8 constant in our pipeline. If W is 0 or
// ≥64 (the trivial / no-clamp cases handled by the runtime),
// the call still rewrites — to a constant 0 (W==0) or
// passthrough (W>=64) — to keep downstream IR uniform.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

/// Read an integer from an arith.constant or llvm.mlir.constant.
bool readIntC(mlir::Value V, int64_t &Out) {
  if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
      Out = IA.getInt();
      return true;
    }
  }
  if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
      Out = IA.getInt();
      return true;
    }
  }
  return false;
}

void rewriteOne(mlir::LLVM::CallOp Call, bool Signed) {
  if (Call->getNumOperands() != 2 || Call->getNumResults() != 1) return;
  mlir::Value Val = Call.getOperand(0);
  mlir::Value Wv = Call.getOperand(1);
  int64_t W;
  if (!readIntC(Wv, W)) return;

  mlir::OpBuilder B(Call);
  mlir::Location L = Call.getLoc();
  mlir::Type Ty = Val.getType();

  // Peel `arith.extsi narrow → wide` so the clamp emits at the
  // narrower width whenever the bounds fit there. Two sub-cases:
  //
  //   1. W >= narrow_width  (saturate range ⊇ input range)
  //      → clamp is a no-op; replace with the wide input.
  //   2. W < narrow_width AND bounds fit in narrow
  //      → emit clamp at narrow, then sign-extend back to wide
  //        for downstream consumers. Avoids the i64 intermediate
  //        whose upper bits Verilator flags as UNUSEDSIGNAL.
  if (auto E = Val.getDefiningOp<mlir::arith::ExtSIOp>(); E && Signed) {
    auto NIT = mlir::dyn_cast<mlir::IntegerType>(E.getIn().getType());
    auto WIT = mlir::dyn_cast<mlir::IntegerType>(Ty);
    if (NIT && WIT && NIT.getWidth() < WIT.getWidth() &&
        W >= 0 && (uint64_t)W >= NIT.getWidth()) {
      Call.getResult().replaceAllUsesWith(Val);
      Call.erase();
      return;
    }
    // W < NIT.getWidth(): clamp at the narrow width.
    if (NIT && WIT && NIT.getWidth() < WIT.getWidth() &&
        W > 0 && (uint64_t)W < NIT.getWidth()) {
      mlir::Type NarrowTy = NIT;
      mlir::Value Narrow = E.getIn();
      // Compute bounds at narrow width.
      int64_t Max = ((int64_t)1 << (W - 1)) - 1;
      int64_t Min = -((int64_t)1 << (W - 1));
      mlir::OpBuilder NB(Call);
      auto NMax = mlir::arith::ConstantOp::create(
          NB, L, NarrowTy, mlir::IntegerAttr::get(NarrowTy, Max));
      auto NMin = mlir::arith::ConstantOp::create(
          NB, L, NarrowTy, mlir::IntegerAttr::get(NarrowTy, Min));
      auto Gt = mlir::arith::CmpIOp::create(
          NB, L, mlir::arith::CmpIPredicate::sgt, Narrow, NMax);
      auto Lt = mlir::arith::CmpIOp::create(
          NB, L, mlir::arith::CmpIPredicate::slt, Narrow, NMin);
      auto Inner = mlir::arith::SelectOp::create(NB, L, Lt, NMin, Narrow);
      auto Clamped = mlir::arith::SelectOp::create(NB, L, Gt, NMax, Inner);
      // Sign-extend back to the original wide type. Downstream
      // `trunci wide→narrow'` chains collapse via the
      // extsi/trunci fold.
      auto Wide = mlir::arith::ExtSIOp::create(NB, L, Ty, Clamped);
      Call.getResult().replaceAllUsesWith(Wide);
      Call.erase();
      return;
    }
  }

  auto ConstI = [&](int64_t V) {
    return mlir::arith::ConstantOp::create(
        B, L, Ty, mlir::IntegerAttr::get(Ty, V));
  };

  mlir::Value Out;
  if (W <= 0) {
    Out = ConstI(0);
  } else if (W >= 64) {
    Out = Val;
  } else if (Signed) {
    int64_t Max = ((int64_t)1 << (W - 1)) - 1;
    int64_t Min = -((int64_t)1 << (W - 1));
    auto MaxV = ConstI(Max);
    auto MinV = ConstI(Min);
    auto GtMax = mlir::arith::CmpIOp::create(
        B, L, mlir::arith::CmpIPredicate::sgt, Val, MaxV);
    auto LtMin = mlir::arith::CmpIOp::create(
        B, L, mlir::arith::CmpIPredicate::slt, Val, MinV);
    auto Inner = mlir::arith::SelectOp::create(B, L, LtMin, MinV, Val);
    Out = mlir::arith::SelectOp::create(B, L, GtMax, MaxV, Inner);
  } else {
    // Unsigned: only an upper bound (no underflow possible — the
    // value is already non-negative in the caller's semantics).
    int64_t Max = (W == 64) ? -1
                            : (int64_t)((uint64_t(1) << W) - 1);
    auto MaxV = ConstI(Max);
    // Compare unsigned (`ugt`) — even though `Val` is signless,
    // the saturate's semantic is unsigned.
    auto GtMax = mlir::arith::CmpIOp::create(
        B, L, mlir::arith::CmpIPredicate::ugt, Val, MaxV);
    Out = mlir::arith::SelectOp::create(B, L, GtMax, MaxV, Val);
  }
  Call.getResult().replaceAllUsesWith(Out);
  Call.erase();
}

} // namespace

bool runLowerFiSaturate(mlir::ModuleOp M) {
  llvm::SmallVector<std::pair<mlir::LLVM::CallOp, bool>, 8> Worklist;
  M.walk([&](mlir::LLVM::CallOp C) {
    auto Sym = C.getCallee();
    if (!Sym) return;
    if (*Sym == "matlab_fi_sat_s64") Worklist.push_back({C, true});
    else if (*Sym == "matlab_fi_sat_u64") Worklist.push_back({C, false});
  });
  for (auto &[C, Signed] : Worklist) rewriteOne(C, Signed);

  // After the peel + clamp rewrite, the original `arith.extsi
  // narrow → wide` op feeding the saturate may be unused.
  // Erase any extsi (or extui) op whose result has no remaining
  // consumers to keep the prelude clean and avoid Verilator's
  // "Signal is not used" warning on the dead i64 intermediate.
  bool ChangedDce = true;
  while (ChangedDce) {
    ChangedDce = false;
    llvm::SmallVector<mlir::Operation *, 8> Dead;
    M.walk([&](mlir::Operation *Op) {
      if (!mlir::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp>(Op)) return;
      if (Op->getNumResults() != 1) return;
      if (!Op->getResult(0).use_empty()) return;
      Dead.push_back(Op);
    });
    for (mlir::Operation *Op : Dead) { Op->erase(); ChangedDce = true; }
  }

  // Collapse `arith.trunci (W → N) of arith.extsi (M → W)` chains
  // to a single op. The peel branch above leaves `extsi narrow →
  // wide` ops whose only consumer is then `trunci wide → narrow`;
  // without this fold, Verilator flags the wide intermediate's
  // upper bits as UNUSEDSIGNAL. Mirrors the same fold in
  // LowerStaticFiArrays — duplicated here because that pass runs
  // before us and won't see the patterns LowerFiSaturate
  // produces.
  bool ChangedFold = true;
  while (ChangedFold) {
    ChangedFold = false;
    llvm::SmallVector<mlir::arith::TruncIOp, 8> Truncs;
    M.walk([&](mlir::arith::TruncIOp T) { Truncs.push_back(T); });
    for (mlir::arith::TruncIOp T : Truncs) {
      auto E = T.getIn().getDefiningOp<mlir::arith::ExtSIOp>();
      if (!E) continue;
      mlir::Value Src = E.getIn();
      auto SrcIT = mlir::dyn_cast<mlir::IntegerType>(Src.getType());
      auto DstIT = mlir::dyn_cast<mlir::IntegerType>(T.getResult().getType());
      if (!SrcIT || !DstIT) continue;
      mlir::OpBuilder B(T);
      mlir::Value New;
      if (SrcIT.getWidth() == DstIT.getWidth()) {
        New = Src;
      } else if (SrcIT.getWidth() < DstIT.getWidth()) {
        New = mlir::arith::ExtSIOp::create(B, T.getLoc(), DstIT, Src);
      } else {
        New = mlir::arith::TruncIOp::create(B, T.getLoc(), DstIT, Src);
      }
      T.getResult().replaceAllUsesWith(New);
      T.erase();
      if (E.getResult().use_empty()) E.erase();
      ChangedFold = true;
    }
  }
  return true;
}

} // namespace mlirgen
} // namespace matlab
