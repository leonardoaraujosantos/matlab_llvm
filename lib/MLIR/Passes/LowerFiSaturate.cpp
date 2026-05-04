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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"
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
      // Tag the outer (final) SelectOp so the SV emitter can hoist
      // a per-(input-width, sat-width) helper function instead of
      // rendering the cmp/cmp/select/select chain inline. The
      // inner SelectOp is left bare — emitter walks the operand
      // structure to extract the original input. See the matching
      // render path in EmitSystemVerilog (B1).
      Clamped->setAttr(
          "matlab.fi_sat_w",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(Call.getContext(), 32), W));
      Clamped->setAttr(
          "matlab.fi_sat_signed",
          mlir::BoolAttr::get(Call.getContext(), Signed));
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
  // Tag the outer SelectOp(s) so the SV emitter can hoist a
  // per-(input-width, sat-width) helper. Skip when W <= 0 (Out is
  // a constant) or W >= 64 (Out is the input passed through);
  // those cases produce no SelectOp.
  if (W > 0 && W < 64) {
    if (auto Sel = Out.getDefiningOp<mlir::arith::SelectOp>()) {
      Sel->setAttr(
          "matlab.fi_sat_w",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(Call.getContext(), 32), W));
      Sel->setAttr(
          "matlab.fi_sat_signed",
          mlir::BoolAttr::get(Call.getContext(), Signed));
    }
  }
  Call.getResult().replaceAllUsesWith(Out);
  Call.erase();
}

/// Walk forward from a tagged saturation `Sel` looking for a
/// destination value with a `matlab.name` (or `name`) attr — a
/// store-target slot or a func.return result. Handles two
/// transparent forwarding shapes that show up in the LowerFiSaturate
/// output: `arith.extsi` (the narrow-peel wrapper) and chained
/// saturations (this Sel's Out is the input of the next Sel).
/// Returns the first base name found, or empty if the chain ends
/// at an anonymous consumer.
std::string walkSatDest(mlir::arith::SelectOp Sel) {
  llvm::SmallPtrSet<mlir::Operation *, 16> Visited;
  llvm::SmallVector<mlir::Value, 4> Frontier;
  Frontier.push_back(Sel.getResult());
  while (!Frontier.empty()) {
    mlir::Value Cur = Frontier.pop_back_val();
    for (mlir::Operation *U : Cur.getUsers()) {
      if (!Visited.insert(U).second) continue;
      if (auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(U)) {
        if (auto *Addr = St.getAddr().getDefiningOp()) {
          if (auto N = Addr->getAttrOfType<mlir::StringAttr>("matlab.name"))
            return N.getValue().str();
          if (auto N = Addr->getAttrOfType<mlir::StringAttr>("name"))
            return N.getValue().str();
        }
        continue;
      }
      // Persistent set runtime call → use the persistent register
      // name as the destination. Both the LLVM-typed
      // `matlab_persistent_set_*` and the matlab.call_builtin
      // `matlab_global_set_*` shapes carry a `persistent_name`
      // string attr from earlier lowering.
      {
        auto getStrAttr = [&](const char *Name) -> mlir::StringAttr {
          return U->getAttrOfType<mlir::StringAttr>(Name);
        };
        llvm::StringRef Callee;
        if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(U)) {
          if (auto S = C.getCallee()) Callee = *S;
        } else if (auto C2 = U->getAttrOfType<mlir::StringAttr>("callee")) {
          Callee = C2.getValue();
        }
        if (Callee.starts_with("matlab_persistent_set") ||
            Callee.starts_with("matlab_global_set")) {
          if (auto N = getStrAttr("persistent_name"))
            return N.getValue().str();
        }
      }
      if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(U)) {
        for (unsigned I = 0; I < R.getNumOperands(); ++I) {
          if (R.getOperand(I) != Cur) continue;
          auto F = R->getParentOfType<mlir::func::FuncOp>();
          if (!F) continue;
          if (auto N = F.getResultAttrOfType<mlir::StringAttr>(
                  I, "matlab.name"))
            return N.getValue().str();
        }
        continue;
      }
      // Pass through ExtSI (narrow-peel wrapper) and any SelectOp
      // (tagged outer of a chained sat, OR untagged Inner whose
      // own user is the chain's outer). CmpI is sat-internal too —
      // walking through its result lands on the SelectOp that
      // consumes it, which is the chain outer.
      if (mlir::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                    mlir::arith::TruncIOp, mlir::arith::SelectOp,
                    mlir::arith::CmpIOp>(U)) {
        for (mlir::Value R : U->getResults()) Frontier.push_back(R);
        continue;
      }
      // Shifts are the canonical "fractional re-quantize" the fi
      // pipeline emits between cascaded saturations (`sat → >>> N
      // → sat`). Walk through when Cur is the shifted value
      // (operand 0); shift-amount uses lead off a different chain.
      if (mlir::isa<mlir::arith::ShRSIOp, mlir::arith::ShRUIOp,
                    mlir::arith::ShLIOp>(U)) {
        if (U->getOperand(0) == Cur)
          for (mlir::Value R : U->getResults()) Frontier.push_back(R);
        continue;
      }
      // Pure arith binops in a fi datapath chain — `add+sat+add+
      // sat` (vector_processor's dot product / mag_sq) lifts the
      // destination name across the binop joins so upstream
      // intermediates inherit the eventual `<dest>_pre` naming
      // instead of staying as `vN_1` placeholders.
      if (mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                    mlir::arith::MulIOp, mlir::arith::AndIOp,
                    mlir::arith::OrIOp, mlir::arith::XOrIOp>(U)) {
        for (mlir::Value R : U->getResults()) Frontier.push_back(R);
        continue;
      }
      // Same idea for the unregistered matlab.* binops that
      // survive through to the SV emitter.
      llvm::StringRef N = U->getName().getStringRef();
      if (N == "matlab.add" || N == "matlab.sub" ||
          N == "matlab.matmul" || N == "matlab.emul") {
        for (mlir::Value R : U->getResults()) Frontier.push_back(R);
        continue;
      }
    }
  }
  return "";
}

/// Attach context-derived `matlab.name` attrs to each tagged
/// saturating SelectOp and to its pre-clamp input value's defining
/// op. Skips ops that already carry a name.
void nameSatChains(mlir::ModuleOp M) {
  M.walk([&](mlir::arith::SelectOp Sel) {
    if (!Sel->hasAttr("matlab.fi_sat_w")) return;
    std::string Base = walkSatDest(Sel);
    if (Base.empty()) return;
    auto Sgn = Sel->getAttrOfType<mlir::BoolAttr>("matlab.fi_sat_signed");
    bool Signed = Sgn && Sgn.getValue();
    // Sat output: only set when the SelectOp itself doesn't have
    // a source name yet (e.g. inherited from a pre-existing matlab
    // attribute upstream).
    if (!Sel->hasAttr("matlab.name") && !Sel->hasAttr("name")) {
      Sel->setAttr("matlab.name",
                   mlir::StringAttr::get(Sel.getContext(), Base + "_sat"));
    }
    // Walk to the pre-clamp input value.
    mlir::Value In;
    if (Signed) {
      if (auto Inner =
              Sel.getFalseValue().getDefiningOp<mlir::arith::SelectOp>())
        In = Inner.getFalseValue();
    } else {
      In = Sel.getFalseValue();
    }
    if (!In) return;
    if (auto *VOp = In.getDefiningOp()) {
      if (!VOp->hasAttr("matlab.name") && !VOp->hasAttr("name")) {
        VOp->setAttr("matlab.name",
                     mlir::StringAttr::get(VOp->getContext(),
                                           Base + "_pre"));
      }
    }
  });
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
  // C1 — propagate destination names onto tagged saturating
  // SelectOps and their pre-clamp input ops. Runs after the DCE
  // and trunci/extsi fold so the forward walk follows the final
  // shape of the IR (no dead ExtSI links to confuse the chain).
  nameSatChains(M);
  return true;
}

} // namespace mlirgen
} // namespace matlab
