// Phase 5.6 Stage F — persistent fi-array lowering for the SV
// backend.
//
// Recognizes the canonical persistent fi-array shift register
// pattern:
//
//   persistent buf;
//   if isempty(buf)
//       buf = fi(zeros(1, N), S, W, F);
//   end
//   buf = [<scalar>, buf(1:N-1)];          % shift-register write
//   ... = buf(k) ...                       % constant-k reads
//
// And rewrites the runtime ABI used by the lowering
// (`matlab_persistent_isempty/get_ptr/set_ptr`) into N parallel
// scalar persistents that the existing `HWStateInfer` recognition
// + SV emitter render as N parallel always_ff registers.
//
// Synthesizes per-element indices `idx*100 + k` (k in [0..N-1]).
// Each synthetic scalar persistent gets its own clone of the
// original isempty / cmpf / scf.if guard, with a single
// `_global_set_f64(idx*100 + k, init_k)` inside the then-region
// (init_k = 0 for zeros-init, the literal coefficient for a
// future literal-init reset).
//
// Each `_persistent_get_ptr(idx) → subscript1_s(_, k_const)`
// chain becomes one `_global_get_f64(idx*100 + (k_const-1))`.
// Each `_persistent_set_ptr(idx, p)` (p = static `llvm.alloca [N
// x iW]` from Stage E) becomes N `_global_set_f64(idx*100 + k,
// gep+load(p, k))` calls.
//
// Stage F v1 scope:
//   - 1-D arrays only (rows = 1 in the zeros init).
//   - Constant-index reads only (loop-iv reads need Stage D
//     post-unroll, which already produces constant indices for
//     bounded for-loops; non-unrollable iv reads bail).
//   - Single-arg isempty pattern (no `isempty(c) || reset` —
//     SplitIsEmptyOr already canonicalizes that earlier).
//   - The zeros init is recognized via `matlab_mat_i64_zeros`
//     as init_p; literal-init resets (e.g. `buf = fi([1, 2, 3,
//     4], ...)`) are deferred to v2.
//
// Returns true on success. Bails (leaves IR unchanged) for any
// pattern that doesn't match — HWLegalize then rejects the
// surviving runtime calls.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
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

/// Match either `matlab.call_builtin` (with `callee` string attr)
/// or `llvm.call @<sym>(...)` against a runtime symbol name. The
/// SV pipeline leaves some calls in unregistered form and lowers
/// others to `llvm.call`; the matcher accepts both shapes.
bool isCallTo(mlir::Operation *Op, llvm::StringRef Callee) {
  if (!Op) return false;
  if (Op->getName().getStringRef() == "matlab.call_builtin") {
    auto S = Op->getAttrOfType<mlir::StringAttr>("callee");
    return S && S.getValue() == Callee;
  }
  if (auto LC = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    auto C = LC.getCallee();
    return C && *C == Callee;
  }
  return false;
}
// Back-compat name kept for the existing call sites in this TU.
bool isMatlabCallBuiltin(mlir::Operation *Op, llvm::StringRef Callee) {
  return isCallTo(Op, Callee);
}

bool readI32Const(mlir::Value V, int32_t &Out) {
  if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
      Out = (int32_t)IA.getInt();
      return true;
    }
  }
  if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
      Out = (int32_t)IA.getInt();
      return true;
    }
  }
  return false;
}

bool readF64Const(mlir::Value V, double &Out) {
  if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
      Out = FA.getValueAsDouble();
      return true;
    }
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
      Out = (double)IA.getInt();
      return true;
    }
  }
  if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>()) {
    auto VAttr = C.getValue();
    if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(VAttr)) {
      Out = FA.getValueAsDouble();
      return true;
    }
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(VAttr)) {
      Out = (double)IA.getInt();
      return true;
    }
  }
  return false;
}

/// Extract element k from `Ptr` — assumed to be the result of an
/// `llvm.alloca [N x iW]` allocated earlier in the same function
/// (typically by Stage E's concat lowering). Returns the loaded
/// scalar of type `iW`. Returns null on shape mismatch.
mlir::Value extractElement(mlir::OpBuilder &B, mlir::Location L,
                           mlir::Value Ptr, int64_t K, int64_t N,
                           unsigned ElemW) {
  auto &Ctx = *B.getContext();
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(&Ctx);
  auto ElemTy = mlir::IntegerType::get(&Ctx, ElemW);
  auto ArrTy = mlir::LLVM::LLVMArrayType::get(ElemTy, N);
  auto I32 = mlir::IntegerType::get(&Ctx, 32);
  auto IdxC = mlir::LLVM::ConstantOp::create(B, L, I32,
      mlir::IntegerAttr::get(I32, K));
  auto Gep = mlir::LLVM::GEPOp::create(B, L, PtrTy, ArrTy, Ptr,
      mlir::ArrayRef<mlir::LLVM::GEPArg>{0, IdxC.getRes()});
  auto Ld = mlir::LLVM::LoadOp::create(B, L, ElemTy, Gep.getRes());
  return Ld.getRes();
}

/// Per-element synthetic persistent ids carve out a private id
/// range so they don't collide with the user's original scalar
/// persistents. Original persistent indices start at 0 and grow
/// densely (1, 2, 3, ...); without this offset, splitting an
/// array at idx 0 with N=4 would emit synthetic ids 0..3 — which
/// alias with the next three scalar persistents. Assumes the user
/// won't have ≥ kSyntheticBase persistents in a single function
/// (1000 is far more than any realistic source program declares).
static constexpr int32_t kSyntheticBase = 1000;

/// Rewrite one persistent fi-array bucket. Returns true on success
/// (IR mutated); false leaves the IR untouched.
///
/// `IsEmpty` may be null when the bucket shares a guard with
/// another bucket (the FIR-asic-pipelined idiom: one
/// `if isempty(delay_line) || reset` block initializes both
/// `delay_line` and `reg_products` — only the first has an
/// isempty call, the rest are just init-sets inside the same
/// scf.if). In that case the per-element guard chain is emitted
/// fresh (each synthetic scalar persistent gets its own
/// `_persistent_isempty(idx_k)` chain, identical reset to 0)
/// and the original guard erase is left to the lead bucket.
bool rewriteOne(mlir::func::FuncOp F, int32_t Idx,
                llvm::StringRef Name,
                llvm::StringRef PersistentFn,
                mlir::Operation *IsEmpty,
                mlir::Operation *InitSet,
                llvm::ArrayRef<mlir::Operation *> RegularSets,
                llvm::ArrayRef<mlir::Operation *> Gets) {
  if (!InitSet) return false;

  auto *Ctx = F.getContext();
  // Validate the init: zeros call with constant (1, N) shape.
  if (InitSet->getNumOperands() < 2) return false;
  mlir::Value InitP = InitSet->getOperand(1);
  auto *InitDef = InitP.getDefiningOp();
  if (!InitDef ||
      !isCallTo(InitDef, "matlab_mat_i64_zeros")) {
    // u64 variant or unsupported init shape.
    if (!InitDef ||
        !isCallTo(InitDef, "matlab_mat_u64_zeros")) {
      return false;
    }
  }
  if (InitDef->getNumOperands() != 2) {
    return false;
  }
  double Rows, Cols;
  if (!readF64Const(InitDef->getOperand(0), Rows)) {
    return false;
  }
  if (!readF64Const(InitDef->getOperand(1), Cols)) {
    return false;
  }
  if (Rows < 1.0 || Cols < 1.0) {
    return false;
  }
  int64_t Nr = (int64_t)Rows;
  int64_t Nc = (int64_t)Cols;
  int64_t N = Nr * Nc;

  // Element width / sign from the `fi_*` attrs the frontend
  // attached to the zeros call. (Same convention as Stage C/D/E.)
  // Capture into immortal NamedAttribute copies — the InitDef op
  // dies along with the guard's then-region when we erase the
  // guard below; reading attrs through it after that is UB.
  // The fi_* attrs survive on the matlab.call_builtin form but get
  // stripped during the conversion to llvm.call (which is where the
  // pipeline shape lands by the time Stage F runs). Derive the
  // storage class from the data shape instead:
  //
  //   1. RegularSet's `_persistent_set_ptr(idx, p)` where p is an
  //      `llvm.alloca [N x iW]` (Stage E shift-register write) →
  //      iW is the storage class. Most reliable for arrays with
  //      shift-register writes (delay_line / reg_buf idiom).
  //   2. A Get's `__subscript_store(get, k, val)` write-through
  //      (FIR-asic-pipelined `reg_products(i) = ...` idiom) → val's
  //      type is the storage class.
  //   3. Fall back to `fi_wl` on InitDef if it's still around (it's
  //      not on the post-pipeline llvm.call shape but the matcher
  //      stays robust if the lowering shape shifts).
  //   4. Otherwise default 16, matching the original Stage F v1
  //      assumption.
  unsigned WL = 16;
  bool Signed = true;
  if (auto WLA = InitDef->getAttrOfType<mlir::IntegerAttr>("fi_wl"))
    WL = (unsigned)WLA.getInt();
  if (auto SA = InitDef->getAttrOfType<mlir::IntegerAttr>("fi_signed"))
    Signed = SA.getInt() != 0;
  unsigned ElemW = WL <= 8 ? 8 : (WL <= 16 ? 16 : (WL <= 32 ? 32 : 64));
  // Probe RegularSets for an alloca-backed write.
  for (mlir::Operation *Set : RegularSets) {
    if (Set->getNumOperands() < 2) continue;
    mlir::Value P = Set->getOperand(1);
    auto Alloca = P.getDefiningOp<mlir::LLVM::AllocaOp>();
    if (!Alloca) continue;
    auto ArrT = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(
        Alloca.getElemType());
    if (!ArrT) continue;
    if (auto IT =
            mlir::dyn_cast<mlir::IntegerType>(ArrT.getElementType())) {
      ElemW = IT.getWidth();
      break;
    }
  }
  // Probe Gets for a __subscript_store write-through.
  for (mlir::Operation *G : Gets) {
    if (G->getNumResults() != 1) continue;
    for (mlir::Operation *U : G->getResult(0).getUsers()) {
      if (!isCallTo(U, "__subscript_store")) continue;
      if (U->getNumOperands() < 3) continue;
      if (auto IT = mlir::dyn_cast<mlir::IntegerType>(
              U->getOperand(2).getType())) {
        ElemW = IT.getWidth();
        break;
      }
    }
  }
  auto ElemTy = mlir::IntegerType::get(Ctx, ElemW);
  auto F64 = mlir::Float64Type::get(Ctx);
  auto I32 = mlir::IntegerType::get(Ctx, 32);
  // Snapshot fi_* attrs so we can keep using them after erasure.
  mlir::IntegerAttr CapWL =
      InitDef->getAttrOfType<mlir::IntegerAttr>("fi_wl");
  mlir::IntegerAttr CapFL =
      InitDef->getAttrOfType<mlir::IntegerAttr>("fi_fl");
  mlir::IntegerAttr CapSn =
      InitDef->getAttrOfType<mlir::IntegerAttr>("fi_signed");
  mlir::IntegerAttr CapOf =
      InitDef->getAttrOfType<mlir::IntegerAttr>("fi_of");
  mlir::IntegerAttr CapRm =
      InitDef->getAttrOfType<mlir::IntegerAttr>("fi_rm");
  auto attachFiAttrs = [&](mlir::OperationState &S) {
    if (CapWL) S.addAttribute("fi_wl", CapWL);
    if (CapFL) S.addAttribute("fi_fl", CapFL);
    if (CapSn) S.addAttribute("fi_signed", CapSn);
    if (CapOf) S.addAttribute("fi_of", CapOf);
    if (CapRm) S.addAttribute("fi_rm", CapRm);
  };
  (void)Signed;

  // Locate the cmpf + scf.if structure consuming the isempty
  // result. Mirror HWStateInfer's matcher.
  //
  // When the bucket has no IsEmpty (it shares a guard with another
  // bucket), the per-element guard chain is emitted just before
  // the InitSet's enclosing scf.if (we walk up to find it). The
  // original cmpf/if/isempty erase is left to the lead bucket.
  mlir::Operation *Cmp = nullptr;
  mlir::arith::CmpFOp CF;
  mlir::scf::IfOp Guard;
  if (IsEmpty) {
    if (IsEmpty->getNumResults() != 1) return false;
    if (!IsEmpty->getResult(0).hasOneUse()) return false;
    Cmp = IsEmpty->getResult(0).use_begin()->getOwner();
    CF = mlir::dyn_cast<mlir::arith::CmpFOp>(Cmp);
    if (!CF || !CF.getResult().hasOneUse()) return false;
    mlir::Operation *CmpUser = CF.getResult().use_begin()->getOwner();
    Guard = mlir::dyn_cast<mlir::scf::IfOp>(CmpUser);
    if (!Guard) return false;
  } else {
    // Walk up from the InitSet to find the enclosing scf.if.
    mlir::Operation *Cur = InitSet->getParentOp();
    while (Cur) {
      if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Cur)) {
        Guard = If;
        break;
      }
      Cur = Cur->getParentOp();
    }
    if (!Guard) return false;
  }

  // Synthesize per-element scalar persistents. For each k in
  // [0..N-1], insert a `_global_set_f64(idx_k, 0)` call inside
  // the existing guard's then-region (replacing the array's
  // `_persistent_set_ptr(idx, zeros)`). Sibling scalar
  // persistents that share the same `if isempty || reset` guard
  // (the FIR-asic-pipelined idiom) keep working through their
  // existing `_global_set_f64(idx, 0)` inits — HWStateInfer
  // sees a uniform set of scalar inits inside one guard and
  // handles all of them with the standard scalar path.
  mlir::OpBuilder B(Guard);
  mlir::Location L = Guard.getLoc();
  auto stringAttr = [&](llvm::StringRef S) {
    return mlir::StringAttr::get(Ctx, S);
  };
  // For each k, emit a fresh `_persistent_isempty(idx_k) → cmpf
  // → scf.if { _global_set_f64(idx_k, 0) }` chain right before
  // the original guard. Each synthetic per-element scalar
  // persistent gets its own canonical init-guard shape, which
  // HWStateInfer recognizes uniformly. Sibling scalar persistents
  // (reg_acc, reg_output) keep their existing inits inside the
  // original guard untouched.
  auto buildIsEmptyChain = [&](int32_t IdxK,
                               std::function<void(mlir::OpBuilder &,
                                                  mlir::Location)> InitBody) {
    mlir::Value KConst = mlir::arith::ConstantOp::create(B, L, I32,
        mlir::IntegerAttr::get(I32, IdxK));
    mlir::OperationState IES(L, "matlab.call_builtin");
    IES.addOperands({KConst});
    IES.addTypes({F64});
    IES.addAttribute("callee", stringAttr("matlab_persistent_isempty"));
    auto *IECall = B.create(IES);
    mlir::Value Zero = mlir::arith::ConstantOp::create(B, L, F64,
        mlir::FloatAttr::get(F64, 0.0));
    auto Cmp = mlir::arith::CmpFOp::create(B, L,
        mlir::arith::CmpFPredicate::ONE,
        IECall->getResult(0), Zero);
    auto NewGuard = mlir::scf::IfOp::create(B, L, Cmp.getResult(),
        /*withElseRegion=*/false);
    mlir::Block *Then = NewGuard.thenBlock();
    mlir::OpBuilder TB(Then->getTerminator());
    InitBody(TB, L);
  };
  for (int64_t k = 0; k < N; ++k) {
    int32_t IdxK = kSyntheticBase + Idx * 100 + (int32_t)k;
    buildIsEmptyChain(IdxK, [&](mlir::OpBuilder &TB, mlir::Location IL) {
      mlir::Value KConst = mlir::arith::ConstantOp::create(TB, IL, I32,
          mlir::IntegerAttr::get(I32, IdxK));
      mlir::Value InitVal = mlir::arith::ConstantOp::create(TB, IL,
          ElemTy, mlir::IntegerAttr::get(ElemTy, 0));
      mlir::OperationState SS(IL, "matlab.call_builtin");
      SS.addOperands({KConst, InitVal});
      SS.addTypes({mlir::NoneType::get(Ctx)});
      SS.addAttribute("callee", stringAttr("matlab_global_set_f64"));
      SS.addAttribute("persistent_fn", stringAttr(PersistentFn));
      SS.addAttribute("persistent_name",
          stringAttr((Name.str() + "_" + std::to_string(k)).c_str()));
      attachFiAttrs(SS);
      (void)TB.create(SS);
    });
  }
  // Erase this bucket's `_persistent_set_ptr(idx, zeros_call)`
  // and the upstream `zeros_call`/`__subscript_store` chain.
  // The Guard scf.if itself stays for now — sibling buckets that
  // share the same guard (FIR-asic-pipelined idiom: one
  // `if isempty(delay_line) || reset` block initializes both
  // `delay_line` AND `reg_products`) need it intact. A final
  // cleanup pass in `runLowerPersistentFiArrays` erases the
  // (now empty) guard + its cmpf/isempty chain after every
  // bucket has been processed.
  if (auto *ZerosCall = InitDef) {
    InitSet->erase();
    if (ZerosCall->getResult(0).use_empty()) ZerosCall->erase();
  }

  // For each get site (`_persistent_get_ptr(idx) →
  // subscript1_s(_, k_const)`), replace with `_global_get_f64(
  // idx*100 + k - 1)` — k in source is 1-based, k_synth is
  // 0-based. Bail if any get's only consumer isn't a
  // recognized constant subscript read.
  for (mlir::Operation *Get : Gets) {
    if (Get->getNumResults() != 1) return false;
    // Two consumer shapes are accepted on a `_persistent_get_ptr`
    // result:
    //
    //   (a) `matlab_mat_i64_subscript1_s(get, k)` — element READ.
    //       Replaced with `_global_get_f64(idx*100 + k-1)`.
    //   (b) `__subscript_store(get, k, val)` — element WRITE
    //       (Phase 5.6 closure: the FIR-asic-pipelined idiom
    //       `reg_products(i) = ...` writes back THROUGH the
    //       persistent ptr without a `_persistent_set_ptr`
    //       follow-up). Replaced with `_global_set_f64(idx*100
    //       + k-1, val)`.
    llvm::SmallVector<mlir::Operation *, 4> SubReads;
    llvm::SmallVector<mlir::Operation *, 4> SubWrites;
    for (mlir::Operation *U : Get->getResult(0).getUsers()) {
      if (isCallTo(U, "matlab_mat_i64_subscript1_s") ||
          isCallTo(U, "matlab_mat_u64_subscript1_s") ||
          isCallTo(U, "matlab_mat_i64_subscript2_s") ||
          isCallTo(U, "matlab_mat_u64_subscript2_s")) {
        SubReads.push_back(U);
        continue;
      }
      if (isCallTo(U, "__subscript_store")) {
        SubWrites.push_back(U);
        continue;
      }
      return false;
    }
    auto I64 = mlir::IntegerType::get(Ctx, 64);
    // Detect `add(int_val, const 1)` pattern (the canonical 1-based
    // index `addr + 1` from user code). Returns the int_val operand,
    // skipping the +1 — comparisons against the 0-based register
    // index `k` then need no offset adjustment. Accepts both
    // `matlab.add` (operand types may differ — e.g. i8 + f64) and
    // already-lowered `arith.addi` / `arith.addf`.
    auto extractZeroBased = [](mlir::Value KRT) -> mlir::Value {
      auto *D = KRT.getDefiningOp();
      if (!D) return mlir::Value{};
      llvm::StringRef N = D->getName().getStringRef();
      bool IsAdd = N == "matlab.add" || N == "arith.addi" ||
                   N == "arith.addf";
      if (!IsAdd || D->getNumOperands() != 2) return mlir::Value{};
      auto isConstOne = [](mlir::Value V) -> bool {
        auto *C = V.getDefiningOp();
        if (!C) return false;
        if (auto Co = mlir::dyn_cast<mlir::arith::ConstantOp>(C)) {
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Co.getValue()))
            return IA.getInt() == 1;
          if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(Co.getValue()))
            return FA.getValueAsDouble() == 1.0;
        }
        if (auto Co = mlir::dyn_cast<mlir::LLVM::ConstantOp>(C)) {
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Co.getValue()))
            return IA.getInt() == 1;
          if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(Co.getValue()))
            return FA.getValueAsDouble() == 1.0;
        }
        if (auto VA = C->getAttrOfType<mlir::IntegerAttr>("value"))
          return VA.getInt() == 1;
        if (auto VA = C->getAttrOfType<mlir::FloatAttr>("value"))
          return VA.getValueAsDouble() == 1.0;
        return false;
      };
      if (isConstOne(D->getOperand(1))) return D->getOperand(0);
      if (isConstOne(D->getOperand(0))) return D->getOperand(1);
      return mlir::Value{};
    };
    // Coerce a 1-based index value to i32. Accepts f64 (the runtime
    // ABI), any IntegerType, or `none`-typed values whose defining
    // op is `matlab.add(typed_int, 1)` (the canonical user pattern).
    // Returns null on unsupported types. Records the original 1-based
    // op so the caller can erase it after the expansion if it becomes
    // dead (otherwise the orphan `matlab.add(i8, f64) -> none` trips
    // HWBitWidthInfer's "value of type 'none' is not synthesizable"
    // gate).
    auto coerceKToI32 = [&](mlir::OpBuilder &SB, mlir::Location Loc,
                            mlir::Value KRT,
                            int64_t &OffsetOut,
                            mlir::Operation *&PeeledAddOut) -> mlir::Value {
      OffsetOut = 1;  // default: source is 1-based; compare against (k+1)
      PeeledAddOut = nullptr;
      // Try to peel off `+ 1` first — keeps the comparison constants
      // matching the user-visible 0-based register indices and
      // typically gives a typed integer source we can use directly.
      if (mlir::Value Z = extractZeroBased(KRT)) {
        PeeledAddOut = KRT.getDefiningOp();
        OffsetOut = 0;
        KRT = Z;
      }
      if (auto IT = mlir::dyn_cast<mlir::IntegerType>(KRT.getType())) {
        if (IT.getWidth() == 32) return KRT;
        if (IT.getWidth() < 32)
          return mlir::arith::ExtUIOp::create(SB, Loc, I32, KRT);
        return mlir::arith::TruncIOp::create(SB, Loc, I32, KRT);
      }
      if (mlir::isa<mlir::Float64Type>(KRT.getType()))
        return mlir::arith::FPToUIOp::create(SB, Loc, I32, KRT);
      return mlir::Value{};
    };
    auto buildGet = [&](mlir::OpBuilder &SB, mlir::Location Loc,
                        int64_t k) -> mlir::Value {
      int32_t IdxK = kSyntheticBase + Idx * 100 + (int32_t)k;
      mlir::Value KConst = mlir::arith::ConstantOp::create(SB, Loc, I32,
          mlir::IntegerAttr::get(I32, IdxK));
      mlir::OperationState GS(Loc, "matlab.call_builtin");
      GS.addOperands({KConst});
      GS.addTypes({I64});
      GS.addAttribute("callee", stringAttr("matlab_global_get_f64"));
      GS.addAttribute("persistent_fn", stringAttr(PersistentFn));
      GS.addAttribute("persistent_name",
          stringAttr((Name.str() + "_" + std::to_string(k)).c_str()));
      attachFiAttrs(GS);
      return SB.create(GS)->getResult(0);
    };
    auto buildSet = [&](mlir::OpBuilder &SB, mlir::Location Loc,
                        int64_t k, mlir::Value Val) {
      int32_t IdxK = kSyntheticBase + Idx * 100 + (int32_t)k;
      mlir::Value KConst = mlir::arith::ConstantOp::create(SB, Loc, I32,
          mlir::IntegerAttr::get(I32, IdxK));
      mlir::OperationState SS(Loc, "matlab.call_builtin");
      SS.addOperands({KConst, Val});
      SS.addTypes({mlir::NoneType::get(Ctx)});
      SS.addAttribute("callee", stringAttr("matlab_global_set_f64"));
      SS.addAttribute("persistent_fn", stringAttr(PersistentFn));
      SS.addAttribute("persistent_name",
          stringAttr((Name.str() + "_" + std::to_string(k)).c_str()));
      attachFiAttrs(SS);
      (void)SB.create(SS);
    };

    for (mlir::Operation *Sub : SubReads) {
      if (Sub->getNumResults() != 1) return false;
      bool Is2D = Sub->getNumOperands() == 3;
      if (!Is2D && Sub->getNumOperands() != 2) return false;
      mlir::OpBuilder SB(Sub);
      auto Loc = Sub->getLoc();
      // 2-D constant access: `arr(i, j)` with both i and j folding
      // to integer constants. Flatten to 1-D row-major (1-based):
      // flat = (i - 1) * Nc + (j - 1). Currently only constant 2-D
      // access is supported; runtime 2-D would need an N-input
      // mux per dimension and is deferred.
      if (Is2D) {
        double ID, JD;
        if (!readF64Const(Sub->getOperand(1), ID)) return false;
        if (!readF64Const(Sub->getOperand(2), JD)) return false;
        int64_t I = (int64_t)ID, J = (int64_t)JD;
        if (I < 1 || I > Nr || J < 1 || J > Nc) return false;
        int64_t Flat = (I - 1) * Nc + (J - 1);
        mlir::Value Val = buildGet(SB, Loc, Flat);
        Sub->getResult(0).replaceAllUsesWith(Val);
        Sub->erase();
        continue;
      }
      double KD;
      if (readF64Const(Sub->getOperand(1), KD)) {
        // Constant-k fast path: single _global_get_f64.
        int64_t K = (int64_t)KD;
        if (K < 1 || K > N) return false;
        mlir::Value Val = buildGet(SB, Loc, K - 1);
        Sub->getResult(0).replaceAllUsesWith(Val);
        Sub->erase();
        continue;
      }
      // Runtime-k read: emit a select cascade fed by N parallel
      // _global_get_f64 reads. Comparisons use either the 0-based
      // source index (when `+1` was peeled off) or the 1-based index
      // against (k_const + 1). The select chain folds back to a
      // single SV `case`/`if-elseif` block.
      int64_t Offset = 1;
      mlir::Operation *PeeledAdd = nullptr;
      mlir::Value KInt = coerceKToI32(SB, Loc, Sub->getOperand(1),
                                       Offset, PeeledAdd);
      if (!KInt) return false;
      // Build the cascade at ElemTy width — trunci every per-element
      // get from i64 (the runtime ABI) first so the final cascade and
      // the consumer slot end up at the register's actual width
      // (otherwise Mem2Reg propagates i64 all the way to the function
      // return and `rdata` shows up as `[63:0]` instead of the user's
      // declared fi width).
      llvm::SmallVector<mlir::Value, 8> Elems;
      for (int64_t k = 0; k < N; ++k) {
        mlir::Value V = buildGet(SB, Loc, k);
        if (V.getType() != ElemTy) {
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
            if (IT.getWidth() > ElemW)
              V = mlir::arith::TruncIOp::create(SB, Loc, ElemTy, V);
            else if (IT.getWidth() < ElemW)
              V = mlir::arith::ExtSIOp::create(SB, Loc, ElemTy, V);
          }
        }
        Elems.push_back(V);
      }
      mlir::Value Result = Elems[N - 1];
      for (int64_t k = N - 2; k >= 0; --k) {
        mlir::Value KCmp = mlir::arith::ConstantOp::create(SB, Loc, I32,
            mlir::IntegerAttr::get(I32, (int64_t)(k + Offset)));
        mlir::Value Eq = mlir::arith::CmpIOp::create(SB, Loc,
            mlir::arith::CmpIPredicate::eq, KInt, KCmp);
        Result = mlir::arith::SelectOp::create(SB, Loc, Eq,
            Elems[k], Result);
      }
      // Replace Sub's i64-typed result with our narrower-typed cascade.
      // Existing consumers (a downstream `arith.trunci` from i64 to
      // ElemW, or a typed slot store) will pick up the new type via
      // RAUW; the slot retype loop below handles the alloc/load
      // signatures.
      mlir::Value Bridge = Result;
      Sub->getResult(0).replaceAllUsesWith(Bridge);
      // Retype any matlab.alloc / llvm.alloca consumer slot if it
      // was allocated as `!llvm.ptr` (Sema saw the fi-array slice
      // as opaque) and now sees a uniformly typed integer store.
      // RefineSlotTypes only handles None / f64 baseline, not ptr.
      for (mlir::OpOperand &Use : Bridge.getUses()) {
        mlir::Operation *U = Use.getOwner();
        if (U->getName().getStringRef() != "matlab.store") continue;
        if (U->getNumOperands() != 2 || U->getOperand(0) != Bridge) continue;
        mlir::Value Slot = U->getOperand(1);
        if (Slot.getType() == Bridge.getType()) continue;
        if (auto *AOp = Slot.getDefiningOp()) {
          if (AOp->getName().getStringRef() != "matlab.alloc") continue;
          bool Uniform = true;
          for (mlir::OpOperand &SU : Slot.getUses()) {
            mlir::Operation *SOp = SU.getOwner();
            if (SOp->getName().getStringRef() == "matlab.store" &&
                SOp->getOperand(0).getType() != Bridge.getType()) {
              Uniform = false; break;
            }
          }
          if (!Uniform) continue;
          Slot.setType(Bridge.getType());
          for (mlir::OpOperand &SU : Slot.getUses()) {
            mlir::Operation *SOp = SU.getOwner();
            if (SOp->getName().getStringRef() == "matlab.load" &&
                SOp->getNumResults() == 1)
              SOp->getResult(0).setType(Bridge.getType());
          }
        }
      }
      Sub->erase();
      // Drop the now-dead `matlab.add(typed_int, 1)` we peeled off —
      // it has no users after Sub erased its only consumer, and its
      // `none` result type would otherwise fail HWBitWidthInfer's
      // "value of type 'none' is not synthesizable" gate.
      if (PeeledAdd && PeeledAdd->use_empty()) PeeledAdd->erase();
      (void)F64;
    }
    // Phase 5.6 closure: per-element WRITES on the get-ptr
    // (`reg_products(i) = ...`). Each becomes a fresh
    // `_global_set_f64(idx_k, val)` — or an N-way decoded write
    // when k is a runtime expression.
    auto coerceVal = [&](mlir::OpBuilder &SB, mlir::Location Loc,
                         mlir::Value Val) -> mlir::Value {
      if (Val.getType() == ElemTy) return Val;
      if (auto VIT = mlir::dyn_cast<mlir::IntegerType>(Val.getType())) {
        if (VIT.getWidth() > ElemW)
          return mlir::arith::TruncIOp::create(SB, Loc, ElemTy, Val);
        if (VIT.getWidth() < ElemW)
          return mlir::arith::ExtSIOp::create(SB, Loc, ElemTy, Val);
        return Val;
      }
      return mlir::Value{};
    };
    for (mlir::Operation *Sub : SubWrites) {
      // 3-arg: __subscript_store(p, k, val) — 1-D
      // 4-arg: __subscript_store(p, i, j, val) — 2-D
      bool Is2DWrite = Sub->getNumOperands() == 4;
      if (!Is2DWrite && Sub->getNumOperands() != 3) return false;
      mlir::OpBuilder SB(Sub);
      auto Loc = Sub->getLoc();
      // 2-D constant write: flatten (i, j) to row-major index.
      if (Is2DWrite) {
        double ID, JD;
        if (!readF64Const(Sub->getOperand(1), ID)) return false;
        if (!readF64Const(Sub->getOperand(2), JD)) return false;
        int64_t I = (int64_t)ID, J = (int64_t)JD;
        if (I < 1 || I > Nr || J < 1 || J > Nc) return false;
        int64_t Flat = (I - 1) * Nc + (J - 1);
        mlir::Value Val = coerceVal(SB, Loc, Sub->getOperand(3));
        if (!Val) return false;
        buildSet(SB, Loc, Flat, Val);
        Sub->erase();
        continue;
      }
      double KD;
      if (readF64Const(Sub->getOperand(1), KD)) {
        // Constant-k fast path: single _global_set_f64.
        int64_t K = (int64_t)KD;
        if (K < 1 || K > N) return false;
        mlir::Value Val = coerceVal(SB, Loc, Sub->getOperand(2));
        if (!Val) return false;
        buildSet(SB, Loc, K - 1, Val);
        Sub->erase();
        continue;
      }
      // Runtime-k write: emit N decoded writes. Each register k
      // gets `next_k = (k_int == k+offset) ? val : cur_k` followed by
      // an unconditional set. The SV emitter renders this as a
      // 2-way mux feeding each register's always_ff. Synthesis
      // tools recognize the pattern and infer per-register decode.
      int64_t Offset = 1;
      mlir::Operation *PeeledAdd = nullptr;
      mlir::Value KInt = coerceKToI32(SB, Loc, Sub->getOperand(1),
                                       Offset, PeeledAdd);
      if (!KInt) return false;
      mlir::Value Val = coerceVal(SB, Loc, Sub->getOperand(2));
      if (!Val) return false;
      for (int64_t k = 0; k < N; ++k) {
        mlir::Value Cur = buildGet(SB, Loc, k);
        // Trunc the i64 ABI return down to ElemTy so the select
        // operand types match.
        if (Cur.getType() != ElemTy) {
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(Cur.getType())) {
            if (IT.getWidth() > ElemW)
              Cur = mlir::arith::TruncIOp::create(SB, Loc, ElemTy, Cur);
            else if (IT.getWidth() < ElemW)
              Cur = mlir::arith::ExtSIOp::create(SB, Loc, ElemTy, Cur);
          } else return false;
        }
        mlir::Value KCmp = mlir::arith::ConstantOp::create(SB, Loc, I32,
            mlir::IntegerAttr::get(I32, (int64_t)(k + Offset)));
        mlir::Value Eq = mlir::arith::CmpIOp::create(SB, Loc,
            mlir::arith::CmpIPredicate::eq, KInt, KCmp);
        mlir::Value Next = mlir::arith::SelectOp::create(SB, Loc, Eq,
            Val, Cur);
        buildSet(SB, Loc, k, Next);
      }
      Sub->erase();
      if (PeeledAdd && PeeledAdd->use_empty()) PeeledAdd->erase();
    }
    Get->erase();
  }

  // For each regular set (`_persistent_set_ptr(idx, p)`), p is the
  // result of an `llvm.alloca [N x iW]` (from Stage E folding).
  // Emit N `_global_set_f64(idx_k, gep+load(p, k))` calls.
  for (mlir::Operation *Set : RegularSets) {
    if (Set->getNumOperands() != 2) return false;
    mlir::Value P = Set->getOperand(1);
    mlir::Operation *PDef = P.getDefiningOp();
    if (!PDef) {
      return false;
    }

    mlir::OpBuilder SB(Set);
    // Two source shapes are accepted:
    //
    //   (a) `llvm.alloca [N x iW]` — the result of Stage E's
    //       concat rewrite folded by `tryRewrite`. Per-element
    //       values come from GEP+load on the alloca.
    //
    //   (b) `llvm.call @matlab_mat_i64_zeros(1, N) +
    //        __subscript_store(zeros, k, val) ...` — Stage E
    //       expanded the concat to this shape but the standard
    //       zeros-folding bailed because the zeros result feeds
    //       `_persistent_set_ptr` (which isn't a recognized
    //       user). Per-element values come from the matching
    //       __subscript_store siblings.
    auto Alloca = mlir::dyn_cast<mlir::LLVM::AllocaOp>(PDef);
    auto ZerosCall = mlir::dyn_cast<mlir::LLVM::CallOp>(PDef);
    if (Alloca) {
      auto ArrT = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(
          Alloca.getElemType());
      if (!ArrT || (int64_t)ArrT.getNumElements() != N) {
        return false;
      }
      if (ArrT.getElementType() != ElemTy) {
        return false;
      }
      for (int64_t k = 0; k < N; ++k) {
        mlir::Value Val = extractElement(SB, Set->getLoc(), P, k, N, ElemW);
        int32_t IdxK = kSyntheticBase + Idx * 100 + (int32_t)k;
        mlir::Value KConst = mlir::arith::ConstantOp::create(SB,
            Set->getLoc(), I32, mlir::IntegerAttr::get(I32, IdxK));
        mlir::OperationState SS(Set->getLoc(), "matlab.call_builtin");
        SS.addOperands({KConst, Val});
        SS.addTypes({mlir::NoneType::get(Ctx)});
        SS.addAttribute("callee", stringAttr("matlab_global_set_f64"));
        SS.addAttribute("persistent_fn", stringAttr(PersistentFn));
        SS.addAttribute("persistent_name",
            stringAttr((Name.str() + "_" + std::to_string(k)).c_str()));
        attachFiAttrs(SS);
        (void)SB.create(SS);
      }
      Set->erase();
      continue;
    }
    if (!ZerosCall) {
      return false;
    }
    auto ZerosCallee = ZerosCall.getCallee();
    if (!ZerosCallee ||
        (*ZerosCallee != "matlab_mat_i64_zeros" &&
         *ZerosCallee != "matlab_mat_u64_zeros")) {
      return false;
    }
    // Walk the zeros-call's users to find per-element
    // __subscript_stores. Build a map k -> stored value.
    llvm::DenseMap<int64_t, mlir::Value> Elems;
    llvm::SmallVector<mlir::Operation *, 8> StoreOps;
    for (mlir::Operation *U : ZerosCall->getUsers()) {
      if (U == Set) continue;  // the persistent_set_ptr we're rewriting
      if (!isCallTo(U, "__subscript_store")) {
        return false;
      }
      if (U->getNumOperands() != 3) return false;
      double KD;
      if (!readF64Const(U->getOperand(1), KD)) return false;
      int64_t K = (int64_t)KD;
      if (K < 1 || K > N) return false;
      Elems[K - 1] = U->getOperand(2);
      StoreOps.push_back(U);
    }
    // For each k in [0..N-1], emit a `_global_set_f64` with the
    // corresponding stored value. Missing slots fall back to 0
    // (the implicit zeros-init value).
    for (int64_t k = 0; k < N; ++k) {
      mlir::Value Val;
      auto It = Elems.find(k);
      if (It != Elems.end()) {
        Val = It->second;
        // The per-element value's type may be wider than the
        // storage class (e.g. i64 from a slice-load). Trunc /
        // bitcast to the element type so the synthetic `_set_f64`
        // call carries the user's typed payload.
        if (Val.getType() != ElemTy) {
          if (auto VIT =
                  mlir::dyn_cast<mlir::IntegerType>(Val.getType())) {
            if (VIT.getWidth() > ElemW)
              Val = mlir::arith::TruncIOp::create(SB, Set->getLoc(),
                  ElemTy, Val);
            else if (VIT.getWidth() < ElemW)
              Val = mlir::arith::ExtSIOp::create(SB, Set->getLoc(),
                  ElemTy, Val);
          } else return false;
        }
      } else {
        Val = mlir::arith::ConstantOp::create(SB, Set->getLoc(), ElemTy,
            mlir::IntegerAttr::get(ElemTy, 0));
      }
      int32_t IdxK = kSyntheticBase + Idx * 100 + (int32_t)k;
      mlir::Value KConst = mlir::arith::ConstantOp::create(SB,
          Set->getLoc(), I32, mlir::IntegerAttr::get(I32, IdxK));
      mlir::OperationState SS(Set->getLoc(), "matlab.call_builtin");
      SS.addOperands({KConst, Val});
      SS.addTypes({mlir::NoneType::get(Ctx)});
      SS.addAttribute("callee", stringAttr("matlab_global_set_f64"));
      SS.addAttribute("persistent_fn", stringAttr(PersistentFn));
      SS.addAttribute("persistent_name",
          stringAttr((Name.str() + "_" + std::to_string(k)).c_str()));
      attachFiAttrs(SS);
      (void)SB.create(SS);
    }
    Set->erase();
    // Erase the now-unused __subscript_stores + zeros call.
    for (mlir::Operation *U : StoreOps) U->erase();
    if (ZerosCall->getResult(0).use_empty()) ZerosCall->erase();
  }
  return true;
}

} // namespace

bool runLowerPersistentFiArrays(mlir::ModuleOp M) {
  // Walk every user function and group persistent_set_ptr /
  // persistent_get_ptr / persistent_isempty calls by index.
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;

    struct Bucket {
      std::string Name;
      std::string PersistentFn;
      mlir::Operation *IsEmpty = nullptr;
      mlir::Operation *InitSet = nullptr;
      llvm::SmallVector<mlir::Operation *, 4> RegularSets;
      llvm::SmallVector<mlir::Operation *, 4> Gets;
    };
    llvm::DenseMap<int32_t, Bucket> Buckets;

    F.walk([&](mlir::Operation *Op) {
      if (!isCallTo(Op, "matlab_persistent_isempty") &&
          !isCallTo(Op, "matlab_persistent_get_ptr") &&
          !isCallTo(Op, "matlab_persistent_set_ptr"))
        return;
      if (Op->getNumOperands() < 1) return;
      int32_t Idx;
      if (!readI32Const(Op->getOperand(0), Idx)) return;
      auto &Bk = Buckets[Idx];
      if (auto N = Op->getAttrOfType<mlir::StringAttr>("persistent_name"))
        Bk.Name = N.getValue().str();
      if (auto N = Op->getAttrOfType<mlir::StringAttr>("persistent_fn"))
        Bk.PersistentFn = N.getValue().str();
      if (isMatlabCallBuiltin(Op, "matlab_persistent_isempty")) {
        Bk.IsEmpty = Op;
      } else if (isMatlabCallBuiltin(Op, "matlab_persistent_get_ptr")) {
        Bk.Gets.push_back(Op);
      } else {
        // _persistent_set_ptr — distinguish init vs regular by
        // ancestor: init-set lives inside an scf.if region whose
        // condition derives from an isempty.
        bool IsInit = false;
        mlir::Operation *Cur = Op->getParentOp();
        while (Cur) {
          if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Cur)) {
            // Check if the if's condition cmpf consumes an
            // isempty result.
            mlir::Value Cond = If.getCondition();
            if (auto CF =
                    Cond.getDefiningOp<mlir::arith::CmpFOp>()) {
              mlir::Operation *Lhs = CF.getLhs().getDefiningOp();
              if (Lhs &&
                  isMatlabCallBuiltin(Lhs, "matlab_persistent_isempty")) {
                IsInit = true;
                break;
              }
            }
          }
          Cur = Cur->getParentOp();
        }
        if (IsInit) Bk.InitSet = Op;
        else Bk.RegularSets.push_back(Op);
      }
    });

    for (auto &Pair : Buckets) {
      Bucket &Bk = Pair.second;
      if (getenv("DEBUG_F"))
        llvm::errs() << "[F] idx=" << Pair.first
                     << " hasIsEmpty=" << !!Bk.IsEmpty
                     << " hasInit=" << !!Bk.InitSet
                     << " gets=" << Bk.Gets.size()
                     << " regSets=" << Bk.RegularSets.size() << "\n";
      // Fall back to a synthesized name when LowerTensorOps's
      // matlab.call_builtin → llvm.call conversion drops the
      // persistent_name attr (it's preserved on _global_set_f64
      // sites for scalar persistents but the persistent_set_ptr
      // family loses it during the runtime-call conversion).
      if (Bk.Name.empty())
        Bk.Name = "buf" + std::to_string(Pair.first);
      if (Bk.PersistentFn.empty())
        Bk.PersistentFn = F.getSymName().str();
      if (!Bk.InitSet) continue;
      bool Ok = rewriteOne(F, Pair.first, Bk.Name, Bk.PersistentFn,
                           Bk.IsEmpty, Bk.InitSet, Bk.RegularSets,
                           Bk.Gets);
      if (getenv("DEBUG_F"))
        llvm::errs() << "[F]   rewriteOne idx=" << Pair.first
                     << " ok=" << Ok << "\n";
      (void)Ok;
    }

    // Original-guard cleanup: the `if isempty(delay_line) || reset`
    // guard now contains scalar-persistent inits (reg_acc=0,
    // reg_output=0) plus dead leftover ops. Pull each scalar
    // `_global_set_f64(idx, val)` out and wrap it in its OWN
    // `_persistent_isempty(idx) → cmpf → scf.if` chain so
    // HWStateInfer's per-persistent matcher accepts each of them.
    // Then erase the original guard (now structurally empty).
    F.walk([&](mlir::scf::IfOp Guard) {
      // Identify guards whose then-region only contains a chain
      // of `_global_set_f64` calls + the implicit yield. Walk the
      // condition to confirm it consumes a `_persistent_isempty`
      // somewhere — direct cmpf or via a `matlab.short_or` /
      // similar logical chain.
      if (!Guard.getThenRegion().hasOneBlock()) return;
      mlir::Block &TB = Guard.getThenRegion().front();
      llvm::SmallVector<mlir::Operation *, 4> ScalarSets;
      bool OnlyScalarSets = true;
      for (mlir::Operation &TOp : TB) {
        if (mlir::isa<mlir::scf::YieldOp>(TOp)) continue;
        if (isCallTo(&TOp, "matlab_global_set_f64")) {
          ScalarSets.push_back(&TOp);
          continue;
        }
        OnlyScalarSets = false;
        break;
      }
      if (!OnlyScalarSets) return;
      // Skip guards that already have the canonical one-set shape —
      // those are either the original simple `if isempty(c)` guard
      // (which HWStateInfer accepts as-is) or a fresh per-element
      // guard we built earlier in `rewriteOne`. Re-processing them
      // here would extract & re-clone the single set into a new
      // guard, leaving orphaned cmpf chains that fail
      // HWStateInfer's `isempty result must feed an arith.cmpf
      // with one use` matcher. Only multi-set guards (the FIR-
      // asic-pipelined `if isempty(c) || reset` idiom that
      // initializes 2+ sibling persistents in one block) need the
      // per-set extraction.
      if (ScalarSets.size() < 2) return;
      // Confirm the condition involves an isempty call —
      // otherwise leave the guard alone (it's a regular user
      // `if`).
      bool CondHasIsEmpty = false;
      llvm::SmallVector<mlir::Operation *, 4> CondWork;
      if (auto *Def = Guard.getCondition().getDefiningOp())
        CondWork.push_back(Def);
      while (!CondWork.empty()) {
        mlir::Operation *Op = CondWork.pop_back_val();
        if (isCallTo(Op, "matlab_persistent_isempty")) {
          CondHasIsEmpty = true;
          break;
        }
        for (mlir::Value V : Op->getOperands())
          if (auto *D = V.getDefiningOp()) CondWork.push_back(D);
      }
      if (!CondHasIsEmpty) return;
      // For each scalar set, build a fresh isempty/cmpf/scf.if
      // chain right before the original guard.
      mlir::OpBuilder OB(Guard);
      mlir::Location GL = Guard.getLoc();
      auto F64 = mlir::Float64Type::get(Guard.getContext());
      auto I32 = mlir::IntegerType::get(Guard.getContext(), 32);
      for (mlir::Operation *Set : ScalarSets) {
        if (Set->getNumOperands() != 2) continue;
        int32_t SetIdx;
        if (!readI32Const(Set->getOperand(0), SetIdx)) continue;
        // Read the init value's stored integer type and value
        // (the scalar persistent's reset constant). Has to be
        // fetched up front because the operand's defining op
        // lives inside the original guard's then-region and
        // dies when we erase Guard below.
        mlir::Value OrigVal = Set->getOperand(1);
        auto OrigValIT =
            mlir::dyn_cast<mlir::IntegerType>(OrigVal.getType());
        if (!OrigValIT) continue;
        int64_t InitInt = 0;
        if (auto C = OrigVal.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto IA =
                  mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
            InitInt = IA.getInt();
        }
        // Capture the persistent_name / fi_* attrs so the cloned
        // call carries the same metadata.
        llvm::SmallVector<mlir::NamedAttribute, 8> SetAttrs;
        for (auto &A : Set->getAttrs()) SetAttrs.push_back(A);

        mlir::Value KConst = mlir::arith::ConstantOp::create(OB, GL, I32,
            mlir::IntegerAttr::get(I32, SetIdx));
        mlir::OperationState IES(GL, "matlab.call_builtin");
        IES.addOperands({KConst});
        IES.addTypes({F64});
        IES.addAttribute("callee",
            mlir::StringAttr::get(Guard.getContext(),
                                   "matlab_persistent_isempty"));
        auto *IECall = OB.create(IES);
        mlir::Value FZero = mlir::arith::ConstantOp::create(OB, GL, F64,
            mlir::FloatAttr::get(F64, 0.0));
        auto Cmp = mlir::arith::CmpFOp::create(OB, GL,
            mlir::arith::CmpFPredicate::ONE,
            IECall->getResult(0), FZero);
        auto NewGuard = mlir::scf::IfOp::create(OB, GL, Cmp.getResult(),
            /*withElseRegion=*/false);
        mlir::Block *Then = NewGuard.thenBlock();
        // Build a fresh `_global_set_f64(idx_const, init_const)`
        // inside the new guard's then-region. We can't move the
        // original op because its operand const is defined
        // inside the about-to-be-erased original guard.
        mlir::OpBuilder TB(Then->getTerminator());
        mlir::Value KVNew = mlir::arith::ConstantOp::create(TB, GL, I32,
            mlir::IntegerAttr::get(I32, SetIdx));
        mlir::Value VNew = mlir::arith::ConstantOp::create(TB, GL,
            OrigValIT, mlir::IntegerAttr::get(OrigValIT, InitInt));
        mlir::OperationState SS(GL, "matlab.call_builtin");
        SS.addOperands({KVNew, VNew});
        SS.addTypes({mlir::NoneType::get(Guard.getContext())});
        for (auto &A : SetAttrs) SS.addAttribute(A.getName(), A.getValue());
        (void)TB.create(SS);
      }
      Guard.erase();
    });

    // DCE + empty-guard cleanup, alternated to fixpoint. The two
    // feed each other: erasing an empty `if (cmpf)` orphan-erases
    // its cmpf + isempty, which then unblocks any other ops that
    // still reference them. Without alternation, the original
    // `if isempty(c)` guard (now empty after Stage F erased its
    // body) stays alive on the first DCE pass — its cmpf has one
    // use (the guard) so DCE skips it — and HWStateInfer later
    // rejects the dangling cmpf with "isempty result must feed an
    // arith.cmpf with one use" because by then the empty guard
    // got erased without re-DCE'ing the cmpf.
    bool Changed = true;
    while (Changed) {
      Changed = false;
      // Empty-guard sweep first — frees up cmpf uses that the DCE
      // walker can then reap.
      llvm::SmallVector<mlir::scf::IfOp, 4> Empty;
      F.walk([&](mlir::scf::IfOp If) {
        if (!If.getThenRegion().hasOneBlock()) return;
        mlir::Block &TB = If.getThenRegion().front();
        for (mlir::Operation &TOp : TB) {
          if (!mlir::isa<mlir::scf::YieldOp>(TOp)) return;
        }
        if (!If.getElseRegion().empty()) {
          mlir::Block &EB = If.getElseRegion().front();
          for (mlir::Operation &EOp : EB) {
            if (!mlir::isa<mlir::scf::YieldOp>(EOp)) return;
          }
        }
        if (If.getNumResults() != 0) return;
        Empty.push_back(If);
      });
      for (auto If : Empty) {
        If.erase();
        Changed = true;
      }
      // DCE leftover orphan ops (cmpf / isempty / short_or chains
      // whose only user was the just-erased guard).
      llvm::SmallVector<mlir::Operation *, 8> Dead;
      F.walk([&](mlir::Operation *Op) {
        if (Op->getNumResults() != 1) return;
        if (!Op->getResult(0).use_empty()) return;
        if (mlir::isa<mlir::arith::CmpFOp, mlir::arith::CmpIOp,
                      mlir::arith::ConstantOp>(Op)) {
          Dead.push_back(Op);
          return;
        }
        if (isCallTo(Op, "matlab_persistent_isempty")) {
          Dead.push_back(Op);
          return;
        }
        if (Op->getName().getStringRef() == "matlab.short_or" ||
            Op->getName().getStringRef() == "matlab.short_and") {
          Dead.push_back(Op);
          return;
        }
      });
      for (mlir::Operation *Op : Dead) {
        Op->erase();
        Changed = true;
      }
    }

    // Final fixup: when a `_global_get_f64(idx)` result flows
    // into a `matlab.store(f64, iN_slot)`, fold the slot away —
    // the slot's loads are routed to the register signal name
    // by the SV emitter through `exprFor`'s persistent-get
    // recognition. Without this fixup the type mismatch on the
    // store breaks SlotPromotion and the bare `matlab.alloc`
    // survives to SV emit (unsupported op).
    //
    // Pattern (after Stage F's array→scalar rewrite):
    //   %v = llvm.call @matlab_global_get_f64(%idx) : f64
    //   matlab.store(%v, %slot : iN)
    //   ... %ld = matlab.load(%slot) : iN ...
    //
    // Rewrite: forward each `matlab.load(%slot)` to the
    // corresponding `_global_get_f64` result (matches what the
    // SV emitter does for direct persistent-get reads). Erase
    // the store + alloc.
    llvm::SmallVector<mlir::Operation *, 4> DeadStores;
    llvm::SmallVector<mlir::Operation *, 4> DeadAllocs;
    F.walk([&](mlir::Operation *St) {
      if (St->getName().getStringRef() != "matlab.store") return;
      if (St->getNumOperands() != 2) return;
      mlir::Value V = St->getOperand(0);
      mlir::Value Slot = St->getOperand(1);
      if (!mlir::isa<mlir::Float64Type>(V.getType())) return;
      if (!mlir::isa<mlir::IntegerType>(Slot.getType())) return;
      auto *Def = V.getDefiningOp();
      if (!Def) return;
      if (!isCallTo(Def, "matlab_global_get_f64")) return;
      auto *SlotDef = Slot.getDefiningOp();
      if (!SlotDef ||
          SlotDef->getName().getStringRef() != "matlab.alloc")
        return;
      // Validate every other user of the slot is a matlab.load
      // — anything else means the slot has multiple writers
      // and the rewrite isn't safe.
      bool Ok = true;
      llvm::SmallVector<mlir::Operation *, 4> Loads;
      for (mlir::Operation *U : SlotDef->getResult(0).getUsers()) {
        if (U == St) continue;
        if (U->getName().getStringRef() == "matlab.load" &&
            U->getNumResults() == 1) {
          Loads.push_back(U);
          continue;
        }
        Ok = false;
        break;
      }
      if (!Ok) return;
      // Forward every load to the get's result.
      for (mlir::Operation *L : Loads) {
        L->getResult(0).replaceAllUsesWith(V);
        L->erase();
      }
      DeadStores.push_back(St);
      DeadAllocs.push_back(SlotDef);
    });
    for (mlir::Operation *Op : DeadStores) Op->erase();
    for (mlir::Operation *Op : DeadAllocs) Op->erase();
  });
  return true;
}

} // namespace mlirgen
} // namespace matlab
