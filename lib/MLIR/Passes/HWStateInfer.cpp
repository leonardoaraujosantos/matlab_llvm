// Phase 3 persistent-variable recognition for the SystemVerilog
// (ASIC) backend. The pre-LowerIO MATLAB pattern:
//
//     persistent c;
//     if isempty(c)
//         c = <reset>;
//     end
//     ... user body ...     // reads / writes via runtime ABI
//     out = c;
//
// lowers to the fixed shape documented in `gatherHWPersistentState`.
// This pass recognizes that shape, gathers per-variable metadata,
// and lets the SV emitter render each one as an inferable
// `always_ff`-driven register.
//
// The pass itself does not mutate the IR. It returns metadata that
// `HWLegalize` (to validate) and `EmitSystemVerilog` (to render)
// both consume — keeping a single point of truth for "what counts as
// a persistent register."

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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

/// Names of the runtime ABI calls that carry persistent-variable
/// state in the post-lowering IR.
constexpr llvm::StringLiteral IsEmptyName = "matlab_persistent_isempty";
constexpr llvm::StringLiteral GetName     = "matlab_global_get_f64";
constexpr llvm::StringLiteral SetName     = "matlab_global_set_f64";

bool isCallTo(mlir::LLVM::CallOp Call, llvm::StringRef Sym) {
  auto C = Call.getCallee();
  return C && *C == Sym;
}

/// True for an `llvm.call @<Sym>(...)` or a `matlab.call_builtin`
/// op whose `callee` attr equals `Sym`. The persistent ABI is split
/// between the two op forms after the existing pipeline runs:
/// `_get_f64` and `_persistent_isempty` survive as `llvm.call`,
/// while `_set_f64` survives as `matlab.call_builtin` — the SV path
/// recognizes both shapes.
bool isCallTo(mlir::Operation *Op, llvm::StringRef Sym) {
  if (auto LC = mlir::dyn_cast<mlir::LLVM::CallOp>(Op))
    return isCallTo(LC, Sym);
  // matlab.call_builtin is unregistered; match by op name + callee
  // string attr.
  if (Op->getName().getStringRef() != "matlab.call_builtin") return false;
  auto S = Op->getAttrOfType<mlir::StringAttr>("callee");
  return S && S.getValue() == Sym;
}

/// Read the `persistent_name` string attr from a call op. Empty if
/// the attr is missing — every recognized site carries it.
llvm::StringRef persistentName(mlir::Operation *Op) {
  if (auto S = Op->getAttrOfType<mlir::StringAttr>("persistent_name"))
    return S.getValue();
  return {};
}

/// Read the integer index operand (always operand 0 of an isempty /
/// get / set call). The lowering uses an `arith.constant` for it.
bool readIndex(mlir::Value V, int32_t &Out) {
  auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return false;
  auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue());
  if (!IA) return false;
  Out = (int32_t)IA.getInt();
  return true;
}

/// Pull the integer width + sign from a typed value. Caller has
/// already confirmed the value's type is an integer (the runtime ABI
/// for `_set_f64` accepts one integer operand carrying the user's
/// typed payload).
bool integerWidthSign(mlir::Type T, unsigned &Width, bool &Signed) {
  auto IT = mlir::dyn_cast<mlir::IntegerType>(T);
  if (!IT) return false;
  Width = IT.getWidth();
  // MLIR's arith integer types are signless. We default signed=true
  // for multi-bit ints — the front-end's int8/int16/... casts are
  // always signed, and uint8/... aren't carried in the type system
  // today (the lowering emits arith.* on signless ints regardless).
  // Phase 3 just records this default; future work can route a
  // `matlab.unsigned` attr through if needed.
  Signed = (Width != 1);
  return true;
}

} // namespace

bool gatherHWPersistentState(mlir::Operation *FuncOp,
                             llvm::SmallVectorImpl<HWPersistentInfo> &Out) {
  auto F = mlir::dyn_cast<mlir::func::FuncOp>(FuncOp);
  if (!F) return true; // nothing to gather

  // First pass — collect every recognized call site, keyed by index.
  struct Bucket {
    std::string Name;
    mlir::Operation *IsEmpty = nullptr;
    llvm::SmallVector<mlir::Operation *, 4> Gets;
    llvm::SmallVector<mlir::Operation *, 4> Sets;
  };
  llvm::DenseMap<int32_t, Bucket> Buckets;

  bool Ok = true;
  // Walk every op (not just llvm.call) — the existing pipeline leaves
  // `_set_f64` as `matlab.call_builtin` while `_get_f64` and
  // `_persistent_isempty` are lowered to `llvm.call`. Both shapes are
  // recognized.
  F.walk([&](mlir::Operation *Op) {
    if (Op->getNumOperands() < 1) return;
    int32_t Idx;
    if (!readIndex(Op->getOperand(0), Idx)) return;

    if (isCallTo(Op, IsEmptyName)) {
      Buckets[Idx].IsEmpty = Op;
      return;
    }
    if (isCallTo(Op, GetName)) {
      Buckets[Idx].Gets.push_back(Op);
      llvm::StringRef Nm = persistentName(Op);
      if (!Nm.empty()) Buckets[Idx].Name = Nm.str();
      return;
    }
    if (isCallTo(Op, SetName)) {
      Buckets[Idx].Sets.push_back(Op);
      llvm::StringRef Nm = persistentName(Op);
      if (!Nm.empty()) Buckets[Idx].Name = Nm.str();
      return;
    }
  });

  if (Buckets.empty()) return true;  // no persistents, nothing to do

  // Second pass — validate each bucket and fill HWPersistentInfo.
  for (auto &Pair : Buckets) {
    int32_t Idx = Pair.first;
    Bucket &B = Pair.second;
    HWPersistentInfo Info;
    Info.Idx = Idx;
    Info.Name = B.Name.empty() ? ("persist" + std::to_string(Idx)) : B.Name;

    if (!B.IsEmpty) {
      mlir::emitError(F.getLoc())
          << "persistent variable '" << Info.Name
          << "' is missing the canonical `if isempty(...) ... end` "
             "initializer (required for synthesizable reset value)";
      Ok = false;
      continue;
    }

    // Locate the isempty guard. The cmpf consuming the isempty result
    // must feed an scf.if; that if's then-region must contain exactly
    // one set call (the reset init).
    mlir::Value IECmpVal;
    {
      // The cmpf consuming the isempty f64 result. (Always llvm.call
      // in the post-LowerIO IR, but we use a generic lookup so the
      // matcher remains robust if the lowering shape shifts.)
      if (B.IsEmpty->getNumResults() != 1) {
        mlir::emitError(B.IsEmpty->getLoc())
            << "isempty call must produce one result";
        Ok = false; continue;
      }
      mlir::Value IEResult = B.IsEmpty->getResult(0);
      if (!IEResult.hasOneUse()) {
        mlir::emitError(B.IsEmpty->getLoc())
            << "isempty result must have exactly one use "
               "(the canonical cmpf one, ..., 0.0)";
        Ok = false; continue;
      }
      mlir::Operation *Cmp = IEResult.use_begin()->getOwner();
      auto CF = mlir::dyn_cast<mlir::arith::CmpFOp>(Cmp);
      if (!CF || !CF.getResult().hasOneUse()) {
        mlir::emitError(Cmp->getLoc())
            << "isempty result must feed an arith.cmpf with one use";
        Ok = false; continue;
      }
      IECmpVal = CF.getResult();
    }
    // The user of the cmpf must be an scf.if; that if's then-region
    // contains the set-call init.
    mlir::Operation *CmpUser = IECmpVal.use_begin()->getOwner();
    auto Guard = mlir::dyn_cast<mlir::scf::IfOp>(CmpUser);
    if (!Guard) {
      mlir::emitError(CmpUser->getLoc())
          << "isempty cmpf must feed an scf.if guard";
      Ok = false; continue;
    }
    Info.IsEmptyGuard = Guard;

    // Find the set call inside the then-region — that's the reset
    // value source. The set-call inside the guard is *also* in our
    // Sets list; remove it (it's the init, not part of the next-state
    // logic).
    mlir::Operation *InitSet = nullptr;
    for (mlir::Operation &Op : Guard.getThenRegion().front()) {
      if (!isCallTo(&Op, SetName)) continue;
      if (Op.getNumOperands() < 1) continue;
      int32_t IIdx;
      if (!readIndex(Op.getOperand(0), IIdx)) continue;
      if (IIdx != Idx) continue;
      InitSet = &Op;
      break;
    }
    if (!InitSet) {
      mlir::emitError(Guard.getLoc())
          << "persistent variable '" << Info.Name
          << "': isempty guard must contain a single init "
             "assignment (`c = <reset>;`)";
      Ok = false; continue;
    }
    if (InitSet->getNumOperands() < 2) {
      mlir::emitError(InitSet->getLoc())
          << "init set call has wrong arity";
      Ok = false; continue;
    }
    Info.ResetValue = InitSet->getOperand(1);
    // Width + sign come from the init value's type — that's the
    // user-declared register width. Intermediate set sites may carry
    // wider arith results (e.g. `c + 1` is i9 in a fi-tagged add);
    // the SV emitter truncates back to the register width at
    // assignment.
    {
      unsigned W; bool Sg;
      if (!integerWidthSign(Info.ResetValue.getType(), W, Sg)) {
        mlir::emitError(InitSet->getLoc())
            << "persistent variable '" << Info.Name
            << "' has non-integer reset value — only integer registers "
               "are synthesizable";
        Ok = false; continue;
      }
      Info.Width = W;
      Info.Signed = Sg;
      // Override signedness + width from the `fi_*` attrs the
      // frontend attached to the reset value's defining op, when
      // present. MLIR's arith integer types are signless and width
      // is the storage class (i8/i16/i32/i64); the user's
      // `fi(value, signed, W, F)` declaration carries both as
      // attrs. Without this override `fi(_, 0, W, F)` would
      // render as `logic signed [STOR-1:0]` instead of the user-
      // declared `logic [W-1:0]`. Walk through `arith.trunci` /
      // `arith.extsi` / `arith.extui` adapters that the lowering
      // sometimes inserts between the typed source op and the
      // set call.
      auto pickFiAttrs = [](mlir::Operation *Op,
                            mlir::BoolAttr &SignedOut,
                            mlir::IntegerAttr &WLOut) {
        for (mlir::Operation *Cur = Op; Cur; ) {
          if (auto SA = Cur->getAttrOfType<mlir::IntegerAttr>("fi_signed"))
            SignedOut = mlir::BoolAttr::get(Cur->getContext(),
                                             SA.getInt() != 0);
          if (auto BA = Cur->getAttrOfType<mlir::BoolAttr>("fi_signed"))
            SignedOut = BA;
          if (auto WLA = Cur->getAttrOfType<mlir::IntegerAttr>("fi_wl"))
            WLOut = WLA;
          if (SignedOut && WLOut) return;
          // Adapter chain: trunci / extsi / extui pass-through.
          if (mlir::isa<mlir::arith::TruncIOp, mlir::arith::ExtSIOp,
                        mlir::arith::ExtUIOp>(Cur)) {
            Cur = Cur->getOperand(0).getDefiningOp();
            continue;
          }
          break;
        }
      };
      mlir::BoolAttr FiSignedAttr;
      mlir::IntegerAttr FiWLAttr;
      // Preferred source: the persistent set call carries the
      // binding's user-declared fi spec (Lowering attaches
      // fi_signed / fi_wl / fi_fl on every `_persistent_set_ptr`
      // / `_global_set_f64` call for a Persistent binding). This
      // is the "ground truth" for the register's declared spec —
      // the user wrote `fi(_, signed, W, F)` and that's what
      // they want emitted, regardless of how the datapath
      // intermediates grow.
      pickFiAttrs(InitSet, FiSignedAttr, FiWLAttr);
      // Fall back: scan regular set sites for fi attrs on the
      // call. Same source (Lowering tags every set), but if the
      // pipeline ever drops the InitSet's attrs we still get a
      // best-effort recovery from the datapath-update sets.
      for (mlir::Operation *S : B.Sets) {
        if (FiSignedAttr && FiWLAttr) break;
        if (S == InitSet) continue;
        pickFiAttrs(S, FiSignedAttr, FiWLAttr);
      }
      // Final fallback: walk the reset value's defining op + any
      // upstream adapter ops (trunci/extsi/extui). Useful for
      // bindings whose set call lost its attrs through some
      // pipeline step that hasn't been audited.
      if ((!FiSignedAttr || !FiWLAttr)) {
        if (auto *Def = Info.ResetValue.getDefiningOp())
          pickFiAttrs(Def, FiSignedAttr, FiWLAttr);
      }
      if (FiSignedAttr) Info.Signed = FiSignedAttr.getValue();
      if (FiWLAttr) {
        unsigned UserWL = (unsigned)FiWLAttr.getInt();
        if (UserWL > 0 && UserWL <= Info.Width) Info.Width = UserWL;
      }
    }
    if (B.Sets.size() <= 1) {
      // Only the init set was found — register has no datapath
      // updates and would be a constant.
      mlir::emitError(InitSet->getLoc())
          << "persistent variable '" << Info.Name
          << "' has no writes outside the reset initializer "
             "(register would be a constant)";
      Ok = false; continue;
    }
    // Filter the init set out of the public Sets list — it lives in
    // the always_ff's reset branch, not in always_comb.
    for (mlir::Operation *S : B.Sets) {
      if (S != InitSet) Info.Sets.push_back(S);
    }
    Info.Gets = std::move(B.Gets);
    Out.push_back(std::move(Info));
  }
  return Ok;
}

} // namespace mlirgen
} // namespace matlab
