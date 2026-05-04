// Slot-type inference for `matlab.alloc` ops still typed `none`
// after the LowerUserCalls iteration loop runs. The user-call
// pass's `propagateScalarTypes` does this same work but only for
// functions still reachable via a `matlab.call` site — once those
// collapse to `func.call` (which happens in the same pass on the
// first iteration that refines a signature), the slot-retype
// logic stops running. This standalone pass picks up where that
// left off.
//
// The retype rule mirrors `propagateScalarTypes` exactly: if every
// `matlab.store` writing to a `none`-typed slot has the same
// concrete scalar primitive value type, the slot's result gets
// retyped and every `matlab.load` reading from it follows.

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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isScalarPrim(mlir::Type T) {
  if (mlir::isa<mlir::IntegerType>(T)) return true;
  if (mlir::isa<mlir::FloatType>(T)) return true;
  /* Phase 6 — sym values are !llvm.ptr (matlab_sym* / matlab_symmat*).
   * Treat as scalar primitives so RefineSlotTypes can promote slots
   * that store sym/symmat values stored from Symbolic Math Toolbox
   * builtins. Without this, slots created by `syms x` lowering stay
   * none-typed forever and survive to EmitC as unsupported matlab.alloc. */
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(T)) return true;
  return false;
}

bool isMatlabOp(mlir::Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

} // namespace

/// Build a per-function map of persistent_name → (width, signed) by
/// scanning every `matlab_global_set_f64` / `matlab_persistent_set_*`
/// call's `fi_wl` / `fi_signed` attrs. The runtime ABI's get-call
/// returns f64 regardless of the register's actual width, so a slot
/// whose every store is `f64 = matlab_global_get_f64(reg)` ends up
/// f64-typed. This map lets us recover the underlying integer width
/// from the (well-known) set sites.
struct PersistInfo {
  unsigned Width = 0;
  bool Signed = true;
};
static llvm::StringMap<PersistInfo> gatherPersistInfo(mlir::func::FuncOp F) {
  llvm::StringMap<PersistInfo> Map;
  F.walk([&](mlir::Operation *Op) {
    auto Name = Op->getAttrOfType<mlir::StringAttr>("persistent_name");
    if (!Name) return;
    // The set-site call has the register's `fi_wl` and
    // `fi_signed` attributes (Lowering / LowerFixedPoint forward
    // them onto the typed-set call). Skip is-empty / get sites
    // (no fi_wl).
    auto WL = Op->getAttrOfType<mlir::IntegerAttr>("fi_wl");
    if (!WL) return;
    auto Sgn = Op->getAttrOfType<mlir::IntegerAttr>("fi_signed");
    PersistInfo PI;
    PI.Width = (unsigned)WL.getInt();
    PI.Signed = Sgn ? Sgn.getInt() != 0 : true;
    auto It = Map.find(Name.getValue());
    if (It == Map.end()) {
      Map[Name.getValue()] = PI;
    } else if (It->second.Width != PI.Width) {
      // Inconsistent width across set sites — leave unresolved
      // (downstream legalize will diagnose).
      It->second.Width = 0;
    }
  });
  return Map;
}

/// HW-aware slot type refinement. When a slot's only stores carry
/// values produced by `matlab_global_get_f64` calls referencing a
/// persistent register with a known integer width, retype the slot
/// to that width. The runtime ABI's f64 return is a transport
/// detail; the user's source semantics treat the value as the
/// register's typed integer. After retype, insert `arith.fptosi`
/// casts at each store and `arith.sitofp` at any surviving f64
/// consumers (matlab.load result type stays f64 if a downstream
/// op needs it that way; we conservatively narrow the load too).
///
/// Unblocks the regfile-class pattern: `if raddr == K; rdata =
/// rN; end` where each rN is `fi(0, 1, 16, 0)`. Without this, the
/// slot stays f64 and HWLegalize rejects the function's f64
/// return type.
static bool refineHWSlotsFromPersists(mlir::ModuleOp M) {
  bool Changed = false;
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;
    auto PMap = gatherPersistInfo(F);
    if (PMap.empty()) return;
    F.walk([&](mlir::Operation *Op) {
      // Operate on `matlab.alloc` whose declared type is f64 (the
      // post-RefineSlotTypes shape: every store was f64 so the
      // slot got refined to f64). This pass tightens to the
      // common-source register's integer width.
      if (!isMatlabOp(Op, "matlab.alloc")) return;
      if (Op->getNumResults() != 1) return;
      mlir::Value Slot = Op->getResult(0);
      if (!mlir::isa<mlir::Float64Type>(Slot.getType())) return;
      // Walk every store; require that each store's value is the
      // result of a `matlab_global_get_f64` call (or an arith.add
      // / mul / sub of such values — the typical mux-out pattern
      // doesn't go through arithmetic so we keep it strict).
      // Collect the union of register widths referenced.
      llvm::SmallVector<mlir::Operation *, 4> Stores;
      llvm::SmallVector<mlir::Operation *, 4> Loads;
      bool Compatible = true;
      unsigned Width = 0;
      bool Signed = true;
      for (mlir::OpOperand &Use : Slot.getUses()) {
        mlir::Operation *U = Use.getOwner();
        if (isMatlabOp(U, "matlab.store") && U->getOperand(1) == Slot) {
          Stores.push_back(U);
          mlir::Value V = U->getOperand(0);
          auto *VOp = V.getDefiningOp();
          if (!VOp) { Compatible = false; break; }
          auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(VOp);
          if (!C) { Compatible = false; break; }
          auto Sym = C.getCallee();
          if (!Sym || *Sym != "matlab_global_get_f64") {
            Compatible = false;
            break;
          }
          auto Name =
              C->getAttrOfType<mlir::StringAttr>("persistent_name");
          if (!Name) { Compatible = false; break; }
          auto It = PMap.find(Name.getValue());
          if (It == PMap.end() || It->second.Width == 0) {
            Compatible = false;
            break;
          }
          if (Width == 0) {
            Width = It->second.Width;
            Signed = It->second.Signed;
          } else if (Width != It->second.Width) {
            // Mixed widths feeding this slot — keep f64 and let
            // downstream pick up.
            Compatible = false;
            break;
          }
          continue;
        }
        if (isMatlabOp(U, "matlab.load")) { Loads.push_back(U); continue; }
        Compatible = false;
        break;
      }
      if (!Compatible || Stores.empty() || Width == 0) return;
      auto NewTy =
          mlir::IntegerType::get(F.getContext(), Width);
      Slot.setType(NewTy);
      // Retype each store's value: replace the f64 with an
      // arith.fptosi (or fptoui) of the f64. Downstream the
      // emitter's existing arith.fptosi-of-persist-get path
      // unwraps to the typed register signal.
      mlir::OpBuilder B(F.getContext());
      for (mlir::Operation *St : Stores) {
        mlir::Value V = St->getOperand(0);
        B.setInsertionPoint(St);
        mlir::Value Cast;
        if (Signed)
          Cast = mlir::arith::FPToSIOp::create(B, St->getLoc(),
                                                NewTy, V);
        else
          Cast = mlir::arith::FPToUIOp::create(B, St->getLoc(),
                                                NewTy, V);
        St->setOperand(0, Cast);
      }
      // Retype every load result; the f64 result type was
      // synchronized with the slot. Downstream consumers expect
      // an integer.
      for (mlir::Operation *Ld : Loads) {
        if (Ld->getNumResults() != 1) continue;
        mlir::Value R = Ld->getResult(0);
        if (mlir::isa<mlir::Float64Type>(R.getType()))
          R.setType(NewTy);
      }
      Changed = true;
    });
  });
  return Changed;
}

bool runRefineSlotTypes(mlir::ModuleOp M) {
  bool ChangedAny = false;
  bool Iterating = true;
  // Iterate until fixpoint — retyping one slot may unblock another
  // (a load from slot A whose type just became concrete now feeds a
  // store into slot B that was previously polymorphic).
  while (Iterating) {
    Iterating = false;
    M.walk([&](mlir::Operation *Op) {
      if (!isMatlabOp(Op, "matlab.alloc")) return;
      if (Op->getNumResults() != 1) return;
      mlir::Value Slot = Op->getResult(0);
      if (!mlir::isa<mlir::NoneType>(Slot.getType())) return;
      mlir::Type Stored;
      bool Consistent = true;
      bool Any = false;
      for (mlir::OpOperand &Use : Slot.getUses()) {
        mlir::Operation *U = Use.getOwner();
        if (isMatlabOp(U, "matlab.store") &&
            U->getNumOperands() == 2 &&
            U->getOperand(1) == Slot) {
          mlir::Type T = U->getOperand(0).getType();
          if (!isScalarPrim(T)) { Consistent = false; break; }
          if (!Any) { Stored = T; Any = true; }
          else if (Stored != T) { Consistent = false; break; }
        }
      }
      if (!Any || !Consistent) return;
      // Retype the slot.
      Slot.setType(Stored);
      // Retype any `none`-typed `matlab.load` reading from this
      // slot. Loads with a concrete result type that already
      // matches stay; loads with a different concrete type are
      // user-visible bugs and we leave them for downstream passes
      // to surface.
      for (mlir::OpOperand &Use : Slot.getUses()) {
        mlir::Operation *U = Use.getOwner();
        if (!isMatlabOp(U, "matlab.load")) continue;
        if (U->getNumResults() != 1) continue;
        mlir::Value LoadRes = U->getResult(0);
        if (mlir::isa<mlir::NoneType>(LoadRes.getType()))
          LoadRes.setType(Stored);
      }
      ChangedAny = true;
      Iterating = true;
    });
  }
  // HW-aware second pass: tighten f64-typed slots whose stores all
  // come from typed persistent gets. Runs after the same-type
  // refinement loop above so the f64 baseline is already in place.
  if (refineHWSlotsFromPersists(M)) ChangedAny = true;
  return ChangedAny;
}

} // namespace mlirgen
} // namespace matlab
