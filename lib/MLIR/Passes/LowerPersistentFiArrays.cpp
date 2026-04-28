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

/// Rewrite one persistent fi-array bucket. Returns true on success
/// (IR mutated); false leaves the IR untouched.
bool rewriteOne(mlir::func::FuncOp F, int32_t Idx,
                llvm::StringRef Name,
                llvm::StringRef PersistentFn,
                mlir::Operation *IsEmpty,
                mlir::Operation *InitSet,
                llvm::ArrayRef<mlir::Operation *> RegularSets,
                llvm::ArrayRef<mlir::Operation *> Gets) {
  if (!IsEmpty || !InitSet) return false;

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
  if (Rows != 1.0 || Cols < 1.0) {
    return false;
  }
  int64_t N = (int64_t)Cols;

  // Element width / sign from the `fi_*` attrs the frontend
  // attached to the zeros call. (Same convention as Stage C/D/E.)
  // Capture into immortal NamedAttribute copies — the InitDef op
  // dies along with the guard's then-region when we erase the
  // guard below; reading attrs through it after that is UB.
  unsigned WL = 16;
  bool Signed = true;
  if (auto WLA = InitDef->getAttrOfType<mlir::IntegerAttr>("fi_wl"))
    WL = (unsigned)WLA.getInt();
  if (auto SA = InitDef->getAttrOfType<mlir::IntegerAttr>("fi_signed"))
    Signed = SA.getInt() != 0;
  unsigned ElemW = WL <= 8 ? 8 : (WL <= 16 ? 16 : (WL <= 32 ? 32 : 64));
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
  if (IsEmpty->getNumResults() != 1) {
    return false;
  }
  if (!IsEmpty->getResult(0).hasOneUse()) {
    return false;
  }
  mlir::Operation *Cmp = IsEmpty->getResult(0).use_begin()->getOwner();
  auto CF = mlir::dyn_cast<mlir::arith::CmpFOp>(Cmp);
  if (!CF || !CF.getResult().hasOneUse()) {
    return false;
  }
  mlir::Operation *CmpUser = CF.getResult().use_begin()->getOwner();
  auto Guard = mlir::dyn_cast<mlir::scf::IfOp>(CmpUser);
  if (!Guard) {
    return false;
  }

  // Synthesize per-element scalar persistents. For each k in
  // [0..N-1]:
  //   - Inject a fresh `_persistent_isempty(idx_k) → cmpf → if {
  //     _global_set_f64(idx_k, 0) }` chain right before the
  //     original guard.
  //   - Replace the original guard's `_persistent_set_ptr(idx,
  //     zeros)` with nothing (the per-k init guards above
  //     handle it).
  mlir::OpBuilder B(Guard);
  mlir::Location L = Guard.getLoc();
  auto stringAttr = [&](llvm::StringRef S) {
    return mlir::StringAttr::get(Ctx, S);
  };
  auto buildIsEmptyChain = [&](int32_t IdxK,
                               std::function<void(mlir::OpBuilder &,
                                                  mlir::Location)> InitBody) {
    // _persistent_isempty(idx_k)
    mlir::Value KConst = mlir::arith::ConstantOp::create(B, L, I32,
        mlir::IntegerAttr::get(I32, IdxK));
    mlir::OperationState IES(L, "matlab.call_builtin");
    IES.addOperands({KConst});
    IES.addTypes({F64});
    IES.addAttribute("callee", stringAttr("matlab_persistent_isempty"));
    auto *IECall = B.create(IES);
    // cmpf one, ie, 0.0
    mlir::Value Zero = mlir::arith::ConstantOp::create(B, L, F64,
        mlir::FloatAttr::get(F64, 0.0));
    auto Cmp = mlir::arith::CmpFOp::create(B, L,
        mlir::arith::CmpFPredicate::ONE,
        IECall->getResult(0), Zero);
    // scf.if with then-region only. The IfOp builder auto-
    // inserts a terminating `scf.yield` in the then-block; we
    // insert our InitBody BEFORE that terminator so the block
    // ends correctly with a single yield.
    auto NewGuard = mlir::scf::IfOp::create(B, L, Cmp.getResult(),
        /*withElseRegion=*/false);
    mlir::Block *Then = NewGuard.thenBlock();
    mlir::OpBuilder TB(Then->getTerminator());
    InitBody(TB, L);
  };
  for (int64_t k = 0; k < N; ++k) {
    int32_t IdxK = Idx * 100 + (int32_t)k;
    buildIsEmptyChain(IdxK, [&](mlir::OpBuilder &TB, mlir::Location IL) {
      // _global_set_f64(idx_k, init_k_typed) with init = 0.
      mlir::Value KConst = mlir::arith::ConstantOp::create(TB, IL, I32,
          mlir::IntegerAttr::get(I32, IdxK));
      mlir::Value InitVal = mlir::arith::ConstantOp::create(TB, IL, ElemTy,
          mlir::IntegerAttr::get(ElemTy, 0));
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
  // Erase the original isempty guard chain entirely. Its uses
  // were the original isempty result → cmpf → if; we just emitted
  // N replacement chains above. The guard's then-region holds the
  // original `_persistent_set_ptr(idx, zeros_call)` and the
  // zeros_call itself; both die with the guard.
  Guard.erase();
  CF.erase();
  IsEmpty->erase();

  // For each get site (`_persistent_get_ptr(idx) →
  // subscript1_s(_, k_const)`), replace with `_global_get_f64(
  // idx*100 + k - 1)` — k in source is 1-based, k_synth is
  // 0-based. Bail if any get's only consumer isn't a
  // recognized constant subscript read.
  for (mlir::Operation *Get : Gets) {
    if (Get->getNumResults() != 1) return false;
    llvm::SmallVector<mlir::Operation *, 4> SubReads;
    for (mlir::Operation *U : Get->getResult(0).getUsers()) {
      if (!isMatlabCallBuiltin(U, "matlab_mat_i64_subscript1_s") &&
          !isMatlabCallBuiltin(U, "matlab_mat_u64_subscript1_s")) {
        return false;
      }
      SubReads.push_back(U);
    }
    for (mlir::Operation *Sub : SubReads) {
      if (Sub->getNumOperands() != 2 || Sub->getNumResults() != 1) {
        return false;
      }
      double KD;
      if (!readF64Const(Sub->getOperand(1), KD)) {
        return false;
      }
      int64_t K = (int64_t)KD;
      if (K < 1 || K > N) {
        return false;
      }
      int32_t IdxK = Idx * 100 + (int32_t)(K - 1);
      mlir::OpBuilder SB(Sub);
      mlir::Value KConst = mlir::arith::ConstantOp::create(SB,
          Sub->getLoc(), I32, mlir::IntegerAttr::get(I32, IdxK));
      // Emit `_global_get_f64` with i64 return type matching the
      // original `subscript1_s` shape so downstream `arith.trunci`
      // consumers fold without an explicit fp→int cast. The SV
      // emitter doesn't care about the declared return type — it
      // routes the call's result through the register's signal
      // name regardless.
      auto I64 = mlir::IntegerType::get(Ctx, 64);
      mlir::OperationState GS(Sub->getLoc(), "matlab.call_builtin");
      GS.addOperands({KConst});
      GS.addTypes({I64});
      GS.addAttribute("callee", stringAttr("matlab_global_get_f64"));
      GS.addAttribute("persistent_fn", stringAttr(PersistentFn));
      GS.addAttribute("persistent_name",
          stringAttr((Name.str() + "_" + std::to_string(K - 1)).c_str()));
      attachFiAttrs(GS);
      auto *NewGet = SB.create(GS);
      Sub->getResult(0).replaceAllUsesWith(NewGet->getResult(0));
      Sub->erase();
      (void)F64;
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
        int32_t IdxK = Idx * 100 + (int32_t)k;
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
      int32_t IdxK = Idx * 100 + (int32_t)k;
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
      // Fall back to a synthesized name when LowerTensorOps's
      // matlab.call_builtin → llvm.call conversion drops the
      // persistent_name attr (it's preserved on _global_set_f64
      // sites for scalar persistents but the persistent_set_ptr
      // family loses it during the runtime-call conversion).
      if (Bk.Name.empty())
        Bk.Name = "buf" + std::to_string(Pair.first);
      if (Bk.PersistentFn.empty())
        Bk.PersistentFn = F.getSymName().str();
      if (!Bk.IsEmpty || !Bk.InitSet) continue;
      bool Ok = rewriteOne(F, Pair.first, Bk.Name, Bk.PersistentFn,
                           Bk.IsEmpty, Bk.InitSet, Bk.RegularSets,
                           Bk.Gets);
      (void)Ok;
    }
  });
  return true;
}

} // namespace mlirgen
} // namespace matlab
