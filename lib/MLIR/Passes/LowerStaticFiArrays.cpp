// Phase 4.5.4 — rewrite the canonical `fi(zeros(1, N), S, W, F)`
// pattern from runtime-call form into a stack-allocated
// `llvm.alloca !llvm.array<N x iW>` with `getelementptr` +
// `load` / `store` access. Lets the SV emitter render the array
// as `logic [W-1:0] arr [N];` with `arr[i] = v;` / `v = arr[i];`
// access, matching the static-array shape from Phase 2's
// for-loop fixtures.
//
// The rewrite is conservative and pattern-matched: each
// `llvm.call @matlab_mat_i64_zeros` is treated independently. If
// any of its uses don't match the recognized pattern (constant
// index on every store, constant index on every read, uniform
// integer element type from the stores), the call is left in
// place and HWLegalize rejects it downstream as a runtime call.
//
// Phase 4.5.4 v1 scope:
//   - 1-D vectors only (rows = 1, cols = N constant)
//   - constant integer indices on both reads and writes
//   - element type uniform across all stores

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isMatlabOpName(mlir::Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

bool getCalleeStr(mlir::Operation *Op, std::string &Out) {
  if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    auto S = C.getCallee();
    if (!S) return false;
    Out = S->str();
    return true;
  }
  if (Op->getName().getStringRef() == "matlab.call_builtin") {
    auto S = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!S) return false;
    Out = S.getValue().str();
    return true;
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
  // `llvm.mlir.constant` from LowerTensorOps's helper for default
  // step values.
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

bool readIntConst(mlir::Value V, int64_t &Out) {
  auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return false;
  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
    Out = IA.getInt();
    return true;
  }
  if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
    Out = (int64_t)FA.getValueAsDouble();
    return true;
  }
  return false;
}

struct Site {
  // The `llvm.call @matlab_mat_i64_zeros` whose result is the
  // array-pointer source. May be consumed directly by subscript ops
  // OR through a `matlab.alloc` slot (which is what
  // `LowerScalarSlots` would normally promote, but it bails on
  // !llvm.ptr-typed slots since they aren't scalar primitives).
  mlir::LLVM::CallOp ZerosCall;
  // Constant N (= rows * cols).
  int64_t N = 0;
  // Element width (e.g. 16 for fi(_, 1, 16, 0)).
  unsigned ElemW = 0;
  // The matlab.alloc slot that wraps the zeros result, if present
  // (the lowering inserts one for any runtime-pointer-typed
  // variable). Erased at the end of rewrite.
  mlir::Operation *Slot = nullptr;
  // Every `matlab.call_builtin __subscript_store` site.
  llvm::SmallVector<mlir::Operation *, 4> Stores;
  // Every `llvm.call @matlab_mat_i64_subscript1_s` site.
  llvm::SmallVector<mlir::LLVM::CallOp, 4> Reads;
  // Every load of the slot.  Erased at the end.
  llvm::SmallVector<mlir::Operation *, 4> SlotLoads;
  // Every store INTO the slot.  Erased at the end.
  llvm::SmallVector<mlir::Operation *, 4> SlotStores;
};

/// Walk every use of `Ptr` and the slot (if any) and gather the
/// recognized uses. Returns false if any use doesn't fit the
/// pattern.
/// True when `User` looks like a "store of `Ptr` into some slot" — both
/// the pre-pipeline matlab.store form and the post-LowerScalarSlots
/// llvm.store form. Returns the slot's defining op via `OutSlot` on
/// success (the slot is a matlab.alloc or llvm.alloca).
bool isSlotStoreOfPtr(mlir::OpOperand &U, mlir::Operation *&OutSlot) {
  mlir::Operation *User = U.getOwner();
  // matlab.store(ptr, slot)
  if (isMatlabOpName(User, "matlab.store") &&
      User->getNumOperands() == 2 && U.getOperandNumber() == 0) {
    auto *Alloc = User->getOperand(1).getDefiningOp();
    if (!isMatlabOpName(Alloc, "matlab.alloc") &&
        !mlir::isa_and_nonnull<mlir::LLVM::AllocaOp>(Alloc))
      return false;
    OutSlot = Alloc;
    return true;
  }
  // llvm.store(ptr, slot) — same pattern after slot-typing.
  if (auto SOp = mlir::dyn_cast<mlir::LLVM::StoreOp>(User)) {
    if (U.getOperandNumber() != 0) return false;
    auto *Alloc = SOp.getAddr().getDefiningOp();
    if (!isMatlabOpName(Alloc, "matlab.alloc") &&
        !mlir::isa_and_nonnull<mlir::LLVM::AllocaOp>(Alloc))
      return false;
    OutSlot = Alloc;
    return true;
  }
  return false;
}

bool gatherUses(mlir::Value Ptr, Site &S) {
  // First, check direct uses of the zeros result.
  for (mlir::OpOperand &U : Ptr.getUses()) {
    mlir::Operation *User = U.getOwner();
    mlir::Operation *Slot = nullptr;
    if (isSlotStoreOfPtr(U, Slot)) {
      // The pointer is being stored INTO a slot.
      S.SlotStores.push_back(User);
      if (S.Slot && S.Slot != Slot) return false;
      S.Slot = Slot;
      continue;
    }
    std::string Callee;
    if (!getCalleeStr(User, Callee)) return false;
    if (Callee == "__subscript_store" && U.getOperandNumber() == 0) {
      S.Stores.push_back(User);
      continue;
    }
    if (Callee == "matlab_mat_i64_subscript1_s" &&
        U.getOperandNumber() == 0) {
      S.Reads.push_back(mlir::cast<mlir::LLVM::CallOp>(User));
      continue;
    }
    return false;
  }
  // If the zeros result was stored into a slot, also walk the slot's
  // load uses to find indirect subscript_store / subscript1_s sites.
  if (S.Slot) {
    for (mlir::OpOperand &U : S.Slot->getResult(0).getUses()) {
      mlir::Operation *User = U.getOwner();
      // Phase 5.6 Stage D: accept either the matlab.store/load pair
      // or the post-LowerScalarSlots llvm.store/load pair — slot
      // typing may have already retyped the slot to !llvm.ptr by
      // the time this pass runs.
      if (isMatlabOpName(User, "matlab.store") ||
          mlir::isa<mlir::LLVM::StoreOp>(User)) continue;
      bool IsLoad = isMatlabOpName(User, "matlab.load") ||
                    mlir::isa<mlir::LLVM::LoadOp>(User);
      if (IsLoad) {
        S.SlotLoads.push_back(User);
        for (mlir::OpOperand &LU : User->getResult(0).getUses()) {
          mlir::Operation *LUO = LU.getOwner();
          std::string Callee;
          if (!getCalleeStr(LUO, Callee)) return false;
          if (Callee == "__subscript_store" &&
              LU.getOperandNumber() == 0) {
            S.Stores.push_back(LUO);
            continue;
          }
          if (Callee == "matlab_mat_i64_subscript1_s" &&
              LU.getOperandNumber() == 0) {
            S.Reads.push_back(mlir::cast<mlir::LLVM::CallOp>(LUO));
            continue;
          }
          return false;
        }
        continue;
      }
      return false;
    }
  }
  return true;
}

bool inferElemWidth(const Site &S, unsigned &OutW) {
  OutW = 0;
  for (mlir::Operation *St : S.Stores) {
    if (St->getNumOperands() < 3) return false;
    mlir::Type T = St->getOperand(2).getType();
    auto IT = mlir::dyn_cast<mlir::IntegerType>(T);
    if (!IT) return false;
    if (OutW == 0) OutW = IT.getWidth();
    else if (OutW != IT.getWidth()) return false;
  }
  if (OutW == 0) {
    // No stores yet — could happen for a read-only persistent zeros
    // (rare). Default to i64 (the runtime ABI's storage width).
    OutW = 64;
  }
  return true;
}

bool tryRewrite(mlir::LLVM::CallOp Zeros) {
  if (Zeros.getNumOperands() != 2 || Zeros.getNumResults() != 1)
    return false;
  double Rows, Cols;
  if (!readF64Const(Zeros.getOperand(0), Rows)) return false;
  if (!readF64Const(Zeros.getOperand(1), Cols)) return false;
  // Phase 4.5.4 v1: 1-D only (rows == 1).
  if (Rows != 1.0) return false;
  if (Cols < 1.0) return false;
  int64_t N = (int64_t)Cols;

  Site S;
  S.ZerosCall = Zeros;
  S.N = N;
  if (!gatherUses(Zeros.getResult(), S)) return false;
  if (!inferElemWidth(S, S.ElemW)) return false;

  // Phase 5.6 Stage D: store/read indices may be either compile-time
  // constants OR an SSA value — typically a for-loop induction
  // variable, since the most common shape is `for i = 1:N; arr(i)
  // ...; end`. Constant indices fold to a fixed GEP offset; SSA
  // indices lower to `arith.fptosi(idx) - 1` (1-based → 0-based)
  // feeding the GEP. The validation here only checks arity; the
  // rewrite path below handles both shapes.
  for (mlir::Operation *St : S.Stores) {
    if (St->getNumOperands() < 3) return false;
  }
  for (mlir::LLVM::CallOp Rd : S.Reads) {
    if (Rd.getNumOperands() != 2) return false;
  }

  mlir::OpBuilder B(Zeros);
  mlir::MLIRContext *Ctx = Zeros.getContext();
  mlir::Location L = Zeros.getLoc();
  auto ElemTy = mlir::IntegerType::get(Ctx, S.ElemW);
  auto ArrTy = mlir::LLVM::LLVMArrayType::get(ElemTy, S.N);
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(Ctx);
  auto I32 = mlir::IntegerType::get(Ctx, 32);
  auto I64 = mlir::IntegerType::get(Ctx, 64);

  // Stack-allocate `[N x iW]`.
  auto One = mlir::LLVM::ConstantOp::create(B, L, I64,
      mlir::IntegerAttr::get(I64, 1));
  auto Alloca = mlir::LLVM::AllocaOp::create(B, L, PtrTy, ArrTy, One, 0);
  // Zero-init: emit `arr[i] = 0` for every i. (memset.intrinsic
  // would also work but adds an LLVM intrinsic dependency.)
  auto Zero = mlir::arith::ConstantOp::create(B, L, ElemTy,
      mlir::IntegerAttr::get(ElemTy, 0));
  for (int64_t I = 0; I < S.N; ++I) {
    auto IdxC = mlir::LLVM::ConstantOp::create(B, L, I32,
        mlir::IntegerAttr::get(I32, I));
    auto Gep = mlir::LLVM::GEPOp::create(
        B, L, PtrTy, ArrTy, Alloca.getRes(),
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, IdxC.getRes()});
    mlir::LLVM::StoreOp::create(B, L, Zero, Gep.getRes());
  }

  // Helper: build the 0-based integer index value for a GEP, from
  // the runtime call's 1-based index operand. For a compile-time
  // constant the GEP gets a static i32; for an SSA value (e.g. a
  // for-loop iv typed f64), we emit `fptosi(idx) - 1`.
  auto buildGepIndex = [&](mlir::OpBuilder &Bldr, mlir::Location IL,
                           mlir::Value RawIdx) -> mlir::Value {
    int64_t K;
    if (readIntConst(RawIdx, K)) {
      return mlir::LLVM::ConstantOp::create(Bldr, IL, I32,
          mlir::IntegerAttr::get(I32, K - 1)).getRes();
    }
    // Non-constant: convert to integer and adjust to 0-based.
    mlir::Value IntIdx;
    if (mlir::isa<mlir::FloatType>(RawIdx.getType())) {
      IntIdx = mlir::arith::FPToSIOp::create(Bldr, IL, I32, RawIdx);
    } else if (auto IT = mlir::dyn_cast<mlir::IntegerType>(RawIdx.getType())) {
      if (IT.getWidth() == 32) IntIdx = RawIdx;
      else if (IT.getWidth() < 32)
        IntIdx = mlir::arith::ExtSIOp::create(Bldr, IL, I32, RawIdx);
      else
        IntIdx = mlir::arith::TruncIOp::create(Bldr, IL, I32, RawIdx);
    } else {
      return {};
    }
    auto OneI32 = mlir::arith::ConstantOp::create(Bldr, IL, I32,
        mlir::IntegerAttr::get(I32, 1));
    return mlir::arith::SubIOp::create(Bldr, IL, IntIdx, OneI32).getResult();
  };

  // Replace each store with GEP + store.
  for (mlir::Operation *St : S.Stores) {
    mlir::OpBuilder SB(St);
    mlir::Value GepIdx = buildGepIndex(SB, St->getLoc(), St->getOperand(1));
    if (!GepIdx) return false;
    auto Gep = mlir::LLVM::GEPOp::create(
        SB, St->getLoc(), PtrTy, ArrTy, Alloca.getRes(),
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, GepIdx});
    mlir::Value V = St->getOperand(2);
    // The runtime ABI accepted `none`-typed values via call_builtin;
    // by Phase 4.5.4 the value should already have a refined integer
    // type. If it doesn't, bail.
    if (!mlir::isa<mlir::IntegerType>(V.getType())) return false;
    mlir::LLVM::StoreOp::create(SB, St->getLoc(), V, Gep.getRes());
    St->erase();
  }

  // Replace each read with GEP + load. The runtime ABI returned i64
  // (with sign-extension) for storage uniformity; we keep that
  // behavior only when a consumer actually needs i64. Most
  // consumers in the lowered IR are `arith.trunci` back to the
  // element width — for those, replacing the trunci's input
  // directly with the load result avoids declaring i64 temps that
  // immediately get truncated away (Verilator flags those upper
  // bits as UNUSEDSIGNAL).
  for (mlir::LLVM::CallOp Rd : S.Reads) {
    mlir::OpBuilder RB(Rd);
    mlir::Value GepIdx = buildGepIndex(RB, Rd.getLoc(), Rd.getOperand(1));
    if (!GepIdx) return false;
    auto Gep = mlir::LLVM::GEPOp::create(
        RB, Rd.getLoc(), PtrTy, ArrTy, Alloca.getRes(),
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, GepIdx});
    auto Ld = mlir::LLVM::LoadOp::create(
        RB, Rd.getLoc(), ElemTy, Gep.getRes());
    mlir::Value LoadVal = Ld.getRes();
    // Walk the read's uses. Anything that's an `arith.trunci` back
    // to an integer ≤ ElemW gets its trunci elided (replace with
    // load result, possibly truncated). Other consumers see a
    // sign-extended i64 to keep ABI compatibility.
    llvm::SmallVector<mlir::Operation *, 4> ToErase;
    bool AnyOther = false;
    for (mlir::OpOperand &U : Rd.getResult().getUses()) {
      mlir::Operation *Use = U.getOwner();
      if (auto Tr = mlir::dyn_cast<mlir::arith::TruncIOp>(Use)) {
        unsigned TW = mlir::cast<mlir::IntegerType>(
            Tr.getResult().getType()).getWidth();
        if (TW <= S.ElemW) {
          mlir::Value V = LoadVal;
          if (TW < S.ElemW) {
            mlir::OpBuilder TB(Tr);
            auto NarrowTy = mlir::IntegerType::get(Ctx, TW);
            V = mlir::arith::TruncIOp::create(
                TB, Tr.getLoc(), NarrowTy, V);
          }
          Tr.getResult().replaceAllUsesWith(V);
          ToErase.push_back(Tr);
          continue;
        }
      }
      AnyOther = true;
    }
    for (mlir::Operation *Op : ToErase) Op->erase();
    if (AnyOther) {
      // Some consumer needs the i64 form — produce it and replace.
      mlir::Value Wide = LoadVal;
      if (S.ElemW < 64)
        Wide = mlir::arith::ExtSIOp::create(RB, Rd.getLoc(), I64, Wide);
      Rd.getResult().replaceAllUsesWith(Wide);
    }
    Rd.erase();
  }

  // Erase any slot loads / stores / the slot alloc itself.
  for (mlir::Operation *Op : S.SlotLoads) {
    Op->getResult(0).replaceAllUsesWith(Alloca.getRes());
    Op->erase();
  }
  for (mlir::Operation *Op : S.SlotStores) Op->erase();
  if (S.Slot) S.Slot->erase();

  Zeros->erase();
  return true;
}

} // namespace

/// Phase 5.6 Stage B — function-arg vector lowering.
///
/// When a func.func has a parameter whose declared type is
/// `!llvm.ptr` and that arg carries the `matlab.array_n` /
/// `matlab.fi_wl` attribute set (attached by `Lowering` when the
/// parameter's Sema-inferred type is a vector fi-array), the arg
/// IS the static-array source. There's no `matlab_mat_i64_zeros`
/// to rewrite — the storage was allocated by the caller (a Stage
/// C literal-init shortcut produces exactly this shape). The body's
/// `matlab_mat_i64_subscript1_s` reads on the arg pointer get
/// rewritten to direct GEP+load on that pointer.
///
/// Also drops the no-op `matlab.fi.cast(load_arg_slot) → ptr`
/// re-cast pattern that the user-written `vec_a = fi(vec_a, S, W,
/// F)` produces. The re-cast is identity for a vector arg whose
/// type already matches; LowerFixedPoint can't lower it (its
/// constructor branch only takes f64 input) so it would otherwise
/// trip HWLegalize as an unhandled op.
bool tryRewriteArg(mlir::BlockArgument Arg, int64_t N, unsigned ElemW) {
  if (N <= 0 || ElemW == 0 || ElemW > 64) return false;
  mlir::MLIRContext *Ctx = Arg.getContext();
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(Ctx);
  auto ElemTy = mlir::IntegerType::get(Ctx, ElemW);
  auto ArrTy = mlir::LLVM::LLVMArrayType::get(ElemTy, N);
  auto I32 = mlir::IntegerType::get(Ctx, 32);
  auto I64 = mlir::IntegerType::get(Ctx, 64);

  // Walk the arg's direct uses + the slot it lives in (if any) to
  // gather subscript reads. Mirrors `gatherUses` but starts at a
  // function arg rather than a `matlab_mat_i64_zeros` call.
  llvm::SmallVector<mlir::LLVM::CallOp, 4> Reads;
  llvm::SmallVector<mlir::Operation *, 4> SlotLoads;
  llvm::SmallVector<mlir::Operation *, 4> SlotStores;
  llvm::SmallVector<mlir::Operation *, 4> NoOpReCasts;
  mlir::Operation *Slot = nullptr;
  for (mlir::OpOperand &U : Arg.getUses()) {
    mlir::Operation *User = U.getOwner();
    mlir::Operation *S = nullptr;
    if (isSlotStoreOfPtr(U, S)) {
      SlotStores.push_back(User);
      if (Slot && Slot != S) return false;
      Slot = S;
      continue;
    }
    // Any other direct use of the arg (e.g. a subscript_s call
    // with the arg directly) — handle below in the unified loop.
    std::string Callee;
    if (!getCalleeStr(User, Callee)) return false;
    if (Callee == "matlab_mat_i64_subscript1_s" &&
        U.getOperandNumber() == 0) {
      Reads.push_back(mlir::cast<mlir::LLVM::CallOp>(User));
      continue;
    }
    return false;
  }
  if (Slot) {
    for (mlir::OpOperand &U : Slot->getResult(0).getUses()) {
      mlir::Operation *User = U.getOwner();
      if (isMatlabOpName(User, "matlab.store") ||
          mlir::isa<mlir::LLVM::StoreOp>(User)) continue;
      bool IsLoad = isMatlabOpName(User, "matlab.load") ||
                    mlir::isa<mlir::LLVM::LoadOp>(User);
      if (!IsLoad) return false;
      SlotLoads.push_back(User);
      for (mlir::OpOperand &LU : User->getResult(0).getUses()) {
        mlir::Operation *LUO = LU.getOwner();
        // The body's `vec_a = fi(vec_a, S, W, F)` re-cast on a
        // vector arg surfaces as `matlab.fi.cast(load) → ptr`
        // whose result feeds another `matlab.store/llvm.store`
        // back into the same slot. Recognize and erase as a no-
        // op (the re-cast IS identity when the input is already
        // typed as the target fi spec, which the call site
        // guarantees by passing a typed array).
        if (isMatlabOpName(LUO, "matlab.fi.cast") &&
            LUO->getNumResults() == 1) {
          // Validate: result is a ptr being stored back into the
          // same slot.
          for (mlir::Operation *CU : LUO->getResult(0).getUsers()) {
            mlir::Operation *S = nullptr;
            mlir::OpOperand *Op0 = nullptr;
            for (mlir::OpOperand &O : CU->getOpOperands())
              if (O.get() == LUO->getResult(0)) { Op0 = &O; break; }
            if (Op0 && isSlotStoreOfPtr(*Op0, S) && S == Slot) {
              NoOpReCasts.push_back(LUO);
              SlotStores.push_back(CU);
              break;
            }
          }
          continue;
        }
        std::string Callee;
        if (!getCalleeStr(LUO, Callee)) return false;
        if (Callee == "matlab_mat_i64_subscript1_s" &&
            LU.getOperandNumber() == 0) {
          Reads.push_back(mlir::cast<mlir::LLVM::CallOp>(LUO));
          continue;
        }
        return false;
      }
    }
  }
  if (Reads.empty()) return false;

  // Helper mirrors the in-place version in `tryRewrite`.
  auto buildGepIndex = [&](mlir::OpBuilder &Bldr, mlir::Location IL,
                           mlir::Value RawIdx) -> mlir::Value {
    int64_t K;
    if (readIntConst(RawIdx, K)) {
      return mlir::LLVM::ConstantOp::create(Bldr, IL, I32,
          mlir::IntegerAttr::get(I32, K - 1)).getRes();
    }
    mlir::Value IntIdx;
    if (mlir::isa<mlir::FloatType>(RawIdx.getType()))
      IntIdx = mlir::arith::FPToSIOp::create(Bldr, IL, I32, RawIdx);
    else if (auto IT = mlir::dyn_cast<mlir::IntegerType>(RawIdx.getType())) {
      if (IT.getWidth() == 32) IntIdx = RawIdx;
      else if (IT.getWidth() < 32)
        IntIdx = mlir::arith::ExtSIOp::create(Bldr, IL, I32, RawIdx);
      else
        IntIdx = mlir::arith::TruncIOp::create(Bldr, IL, I32, RawIdx);
    } else return {};
    auto OneI32 = mlir::arith::ConstantOp::create(Bldr, IL, I32,
        mlir::IntegerAttr::get(I32, 1));
    return mlir::arith::SubIOp::create(Bldr, IL, IntIdx, OneI32).getResult();
  };

  // For each read: GEP + load on the arg pointer, replacing the
  // runtime call. Mirrors the consumer-rewrite in `tryRewrite`.
  for (mlir::LLVM::CallOp Rd : Reads) {
    mlir::OpBuilder RB(Rd);
    mlir::Value GepIdx = buildGepIndex(RB, Rd.getLoc(), Rd.getOperand(1));
    if (!GepIdx) return false;
    auto Gep = mlir::LLVM::GEPOp::create(
        RB, Rd.getLoc(), PtrTy, ArrTy, Arg,
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, GepIdx});
    auto Ld = mlir::LLVM::LoadOp::create(RB, Rd.getLoc(), ElemTy, Gep.getRes());
    mlir::Value LoadVal = Ld.getRes();
    llvm::SmallVector<mlir::Operation *, 4> ToErase;
    bool AnyOther = false;
    for (mlir::OpOperand &U : Rd.getResult().getUses()) {
      mlir::Operation *Use = U.getOwner();
      if (auto Tr = mlir::dyn_cast<mlir::arith::TruncIOp>(Use)) {
        unsigned TW = mlir::cast<mlir::IntegerType>(
            Tr.getResult().getType()).getWidth();
        if (TW <= ElemW) {
          mlir::Value V = LoadVal;
          if (TW < ElemW) {
            mlir::OpBuilder TB(Tr);
            auto NarrowTy = mlir::IntegerType::get(Ctx, TW);
            V = mlir::arith::TruncIOp::create(TB, Tr.getLoc(), NarrowTy, V);
          }
          Tr.getResult().replaceAllUsesWith(V);
          ToErase.push_back(Tr);
          continue;
        }
      }
      AnyOther = true;
    }
    for (mlir::Operation *Op : ToErase) Op->erase();
    if (AnyOther) {
      mlir::Value Wide = LoadVal;
      if (ElemW < 64)
        Wide = mlir::arith::ExtSIOp::create(RB, Rd.getLoc(), I64, Wide);
      Rd.getResult().replaceAllUsesWith(Wide);
    }
    Rd.erase();
  }

  // Erase no-op re-casts and any slot store/load chains they
  // belonged to. A surviving slot load with no remaining uses
  // also goes; the slot itself stays if other uses exist.
  for (mlir::Operation *Op : NoOpReCasts) Op->erase();
  for (mlir::Operation *Op : SlotStores) Op->erase();
  for (mlir::Operation *Op : SlotLoads) {
    if (Op->getResult(0).use_empty()) Op->erase();
    else Op->getResult(0).replaceAllUsesWith(Arg);
  }
  // The slot's own store of the original arg → ptr is now
  // dead; collapse the slot to point directly at the arg by
  // replacing remaining slot loads (above). If the slot has any
  // other uses we leave it; downstream cleanup handles it.
  if (Slot && Slot->getResult(0).use_empty()) Slot->erase();
  return true;
}

/// Phase 5.6 Stage E — rewrite a `matlab_mat_i64_concat_row(a, b,
/// ...)` whose every operand has a statically-known length into a
/// fresh `matlab_mat_i64_zeros(1, N) + N __subscript_store` chain.
/// The existing zeros-folding path (`tryRewrite` above) then turns
/// the chain into an `llvm.alloca [N x iW]` + per-element stores.
///
/// Recognized operand shapes:
///   - `matlab_mat_i64_from_scalar(val_i64)` — single element. The
///     scalar value gets stored at the corresponding output slot,
///     truncated/extended to the storage class as needed.
///   - `matlab_mat_i64_slice1(src_ptr, range_tensor)` where the
///     range is a `matlab.range(start, end)` with both bounds
///     compile-time constants. Each element in the slice is read
///     via a fresh `subscript1_s(src_ptr, k)` call, then stored at
///     the corresponding output slot. The fold path's later
///     `tryRewrite` on src_ptr (zeros-allocated) collapses both
///     the read and write sides to GEP+load/store on the source's
///     and destination's allocas.
///
/// Returns true on success. Drops the original concat / slice /
/// from_scalar / range ops if they have no remaining uses.
bool tryRewriteConcat(mlir::LLVM::CallOp Concat) {
  auto Sym = Concat.getCallee();
  if (!Sym || *Sym != "matlab_mat_i64_concat_row") return false;
  if (Concat.getNumResults() != 1) return false;

  struct OpInfo {
    enum Kind { Scalar, Slice } K;
    mlir::Value SrcVal;
    int64_t Start = 0;
    int64_t Len = 1;
    mlir::Operation *DefOp = nullptr;
  };
  llvm::SmallVector<OpInfo, 4> OpInfos;
  int64_t TotalN = 0;
  for (mlir::Value Arg : Concat.getOperands()) {
    auto *Def = Arg.getDefiningOp();
    if (!Def) return false;
    auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Def);
    if (!Call) return false;
    auto Callee = Call.getCallee();
    if (!Callee) return false;
    OpInfo Oi;
    Oi.DefOp = Def;
    if (*Callee == "matlab_mat_i64_from_scalar" ||
        *Callee == "matlab_mat_u64_from_scalar") {
      if (Call.getNumOperands() != 1) return false;
      Oi.K = OpInfo::Scalar;
      Oi.SrcVal = Call.getOperand(0);
      Oi.Len = 1;
    } else if (*Callee == "matlab_mat_i64_zeros" ||
               *Callee == "matlab_mat_u64_zeros") {
      // A `zeros(1, N)` source (typically from a literal-init
      // array `fi([c1, c2, ..., cN], ...)` after Stage C lowering)
      // contributes N elements: reads are emitted on the zeros'
      // result pointer at indices [1..N]. The downstream
      // `tryRewrite` on the same zeros call then folds both the
      // existing __subscript_stores AND our new subscript reads
      // into the same `llvm.alloca [N x iW]`.
      if (Call.getNumOperands() != 2) return false;
      double Rows, Cols;
      if (!readF64Const(Call.getOperand(0), Rows)) return false;
      if (!readF64Const(Call.getOperand(1), Cols)) return false;
      if (Rows != 1.0 || Cols < 1.0) return false;
      Oi.K = OpInfo::Slice;
      Oi.SrcVal = Call.getResult();
      Oi.Start = 1;
      Oi.Len = (int64_t)Cols;
    } else if (*Callee == "matlab_mat_i64_slice1" ||
               *Callee == "matlab_mat_u64_slice1") {
      if (Call.getNumOperands() != 2) return false;
      // Post-LowerTensorOps shape: the range arg is a `llvm.call
      // @matlab_range(start, step, end) → ptr` (always 3 args; step
      // defaults to 1.0 when source had no explicit step).
      auto RngCall = Call.getOperand(1).getDefiningOp<mlir::LLVM::CallOp>();
      if (!RngCall) return false;
      auto RngCallee = RngCall.getCallee();
      if (!RngCallee || *RngCallee != "matlab_range") return false;
      if (RngCall.getNumOperands() != 3) return false;
      double Start, Step, End;
      if (!readF64Const(RngCall.getOperand(0), Start)) return false;
      if (!readF64Const(RngCall.getOperand(1), Step)) return false;
      if (!readF64Const(RngCall.getOperand(2), End)) return false;
      // Stage E v1: only unit-step ranges. Strided slices need
      // per-element index arithmetic; defer.
      if (Step != 1.0) return false;
      if (Start < 1 || End < Start) return false;
      Oi.K = OpInfo::Slice;
      Oi.SrcVal = Call.getOperand(0);
      Oi.Start = (int64_t)Start;
      Oi.Len = (int64_t)(End - Start + 1);
    } else {
      return false;
    }
    OpInfos.push_back(Oi);
    TotalN += Oi.Len;
  }
  if (TotalN < 1) return false;

  mlir::OpBuilder B(Concat);
  mlir::Location L = Concat.getLoc();
  auto &Ctx = *Concat.getContext();
  auto F64 = mlir::Float64Type::get(&Ctx);
  auto I64 = mlir::IntegerType::get(&Ctx, 64);
  auto PtrTy = mlir::LLVM::LLVMPointerType::get(&Ctx);
  // Inherit the fi-spec attrs from the concat call so the
  // synthesized zeros / __subscript_store ops carry the same
  // metadata (LowerStaticFiArrays' existing logic + downstream
  // emitter use these).
  llvm::SmallVector<mlir::NamedAttribute, 8> ConcatAttrs;
  for (auto &E0 : Concat->getAttrs()) {
    if (E0.getName().getValue() == "callee") continue;
    ConcatAttrs.push_back(E0);
  }
  // Storage class width from fi_wl.
  unsigned ElemW = 16;
  bool Signed = true;
  if (auto WLA = Concat->getAttrOfType<mlir::IntegerAttr>("fi_wl")) {
    unsigned W = (unsigned)WLA.getInt();
    ElemW = W <= 8 ? 8 : (W <= 16 ? 16 : (W <= 32 ? 32 : 64));
  }
  if (auto SA = Concat->getAttrOfType<mlir::IntegerAttr>("fi_signed"))
    Signed = SA.getInt() != 0;
  auto IT = mlir::IntegerType::get(&Ctx, ElemW);

  // Step 1: emit matlab_mat_{i,u}64_zeros(1, N) for the result.
  llvm::StringRef ZerosCallee = Signed ? "matlab_mat_i64_zeros"
                                       : "matlab_mat_u64_zeros";
  auto getOrInsert = [&](llvm::StringRef Name, mlir::Type Ret,
                         mlir::ArrayRef<mlir::Type> Args)
      -> mlir::LLVM::LLVMFuncOp {
    auto M = Concat->getParentOfType<mlir::ModuleOp>();
    if (auto Existing = M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(Name))
      return Existing;
    mlir::OpBuilder::InsertionGuard G(B);
    B.setInsertionPointToStart(M.getBody());
    auto Ty = mlir::LLVM::LLVMFunctionType::get(Ret, Args);
    auto Fn = mlir::LLVM::LLVMFuncOp::create(B, M.getLoc(), Name, Ty);
    Fn.setLinkage(mlir::LLVM::Linkage::External);
    return Fn;
  };
  auto ZerosFn = getOrInsert(ZerosCallee, PtrTy, {F64, F64});
  mlir::Value RowsV = mlir::arith::ConstantOp::create(B, L, F64,
      mlir::FloatAttr::get(F64, 1.0));
  mlir::Value ColsV = mlir::arith::ConstantOp::create(B, L, F64,
      mlir::FloatAttr::get(F64, (double)TotalN));
  auto ZerosCall = mlir::LLVM::CallOp::create(B, L, ZerosFn,
      mlir::ValueRange{RowsV, ColsV});
  for (auto &E0 : ConcatAttrs)
    ZerosCall->setAttr(E0.getName(), E0.getValue());
  mlir::Value NewArr = ZerosCall.getResult();

  // Step 2: per-element __subscript_store. For Slice operands we
  // also emit per-element subscript1_s reads on the source ptr.
  llvm::StringRef SubReadCallee = Signed ? "matlab_mat_i64_subscript1_s"
                                         : "matlab_mat_u64_subscript1_s";
  int64_t DstK = 1;
  for (auto &Oi : OpInfos) {
    if (Oi.K == OpInfo::Scalar) {
      // Adjust scalar value to the storage type.
      mlir::Value V = Oi.SrcVal;
      if (auto VIT = mlir::dyn_cast<mlir::IntegerType>(V.getType())) {
        if (VIT.getWidth() != ElemW) {
          if (VIT.getWidth() < ElemW)
            V = Signed ? (mlir::Value)mlir::arith::ExtSIOp::create(B, L, IT, V)
                       : (mlir::Value)mlir::arith::ExtUIOp::create(B, L, IT, V);
          else
            V = mlir::arith::TruncIOp::create(B, L, IT, V);
        }
      } else {
        return false;
      }
      mlir::Value KV = mlir::arith::ConstantOp::create(B, L, F64,
          mlir::FloatAttr::get(F64, (double)DstK));
      mlir::OperationState St(L, "matlab.call_builtin");
      St.addOperands({NewArr, KV, V});
      St.addTypes({mlir::NoneType::get(&Ctx)});
      St.addAttribute("callee",
          mlir::StringAttr::get(&Ctx, "__subscript_store"));
      for (auto &E0 : ConcatAttrs)
        St.addAttribute(E0.getName(), E0.getValue());
      (void)B.create(St);
      DstK++;
      continue;
    }
    // Slice: emit per-element reads on Oi.SrcVal at indices
    // [Start..Start+Len-1], stored at consecutive output slots.
    auto SubReadFn = getOrInsert(SubReadCallee, I64, {PtrTy, F64});
    for (int64_t i = 0; i < Oi.Len; ++i) {
      int64_t SrcK = Oi.Start + i;
      mlir::Value SrcKV = mlir::arith::ConstantOp::create(B, L, F64,
          mlir::FloatAttr::get(F64, (double)SrcK));
      auto Rd = mlir::LLVM::CallOp::create(B, L, SubReadFn,
          mlir::ValueRange{Oi.SrcVal, SrcKV});
      for (auto &E0 : ConcatAttrs)
        Rd->setAttr(E0.getName(), E0.getValue());
      // Trunc the i64 to the storage class for the store.
      mlir::Value V = Rd.getResult();
      if (ElemW < 64)
        V = mlir::arith::TruncIOp::create(B, L, IT, V);
      mlir::Value DstKV = mlir::arith::ConstantOp::create(B, L, F64,
          mlir::FloatAttr::get(F64, (double)DstK));
      mlir::OperationState St(L, "matlab.call_builtin");
      St.addOperands({NewArr, DstKV, V});
      St.addTypes({mlir::NoneType::get(&Ctx)});
      St.addAttribute("callee",
          mlir::StringAttr::get(&Ctx, "__subscript_store"));
      for (auto &E0 : ConcatAttrs)
        St.addAttribute(E0.getName(), E0.getValue());
      (void)B.create(St);
      DstK++;
    }
  }

  // Step 3: rewrite the concat result to point at the new zeros'
  // pointer; erase the concat + recursively-dead helper ops.
  Concat.getResult().replaceAllUsesWith(NewArr);
  Concat.erase();
  // Iterative DCE: erase any op whose results have all become
  // dead. Keep going until nothing further changes — handles the
  // chain `from_scalar(extsi(fi.cast))` and `slice1(load,
  // matlab_range(c1, c2, c3))` whose intermediate ops drop out
  // one level at a time.
  llvm::SmallVector<mlir::Operation *, 8> Seeds;
  for (auto &Oi : OpInfos) if (Oi.DefOp) Seeds.push_back(Oi.DefOp);
  bool Changed = true;
  while (Changed) {
    Changed = false;
    llvm::SmallVector<mlir::Operation *, 8> NextSeeds;
    for (mlir::Operation *Op : Seeds) {
      if (!Op) continue;
      bool AllDead = true;
      for (mlir::Value R : Op->getResults())
        if (!R.use_empty()) { AllDead = false; break; }
      if (!AllDead) continue;
      // Op is dead — remember its operand-defs and erase.
      for (mlir::Value V : Op->getOperands())
        if (auto *D = V.getDefiningOp()) NextSeeds.push_back(D);
      Op->erase();
      Changed = true;
    }
    Seeds = std::move(NextSeeds);
  }
  return true;
}

bool runLowerStaticFiArrays(mlir::ModuleOp M) {
  // Phase 5.6 Stage E: pre-fold concat-of-statically-shaped-
  // operands into a `matlab_mat_i64_zeros + N __subscript_store`
  // chain so the standard zeros-folding path below picks them up.
  llvm::SmallVector<mlir::LLVM::CallOp, 4> Concats;
  M.walk([&](mlir::LLVM::CallOp C) {
    auto Sym = C.getCallee();
    if (!Sym) return;
    if (*Sym == "matlab_mat_i64_concat_row" ||
        *Sym == "matlab_mat_u64_concat_row")
      Concats.push_back(C);
  });
  for (mlir::LLVM::CallOp C : Concats) (void)tryRewriteConcat(C);

  llvm::SmallVector<mlir::LLVM::CallOp, 4> Worklist;
  M.walk([&](mlir::LLVM::CallOp C) {
    auto Sym = C.getCallee();
    if (!Sym) return;
    if (*Sym != "matlab_mat_i64_zeros") return;
    Worklist.push_back(C);
  });
  for (mlir::LLVM::CallOp C : Worklist) {
    (void)tryRewrite(C);
  }

  // Phase 5.6 Stage B: also rewrite vector function args whose
  // declared type is `!llvm.ptr` and which carry the
  // `matlab.array_n` / `matlab.fi_wl` attributes attached by
  // `Lowering` for inferred-vector parameters. The arg's
  // pointer is treated as the static-array source.
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;
    auto FT = F.getFunctionType();
    for (unsigned I = 0; I < FT.getNumInputs(); ++I) {
      auto N = F.getArgAttrOfType<mlir::IntegerAttr>(I, "matlab.array_n");
      auto WL = F.getArgAttrOfType<mlir::IntegerAttr>(I, "matlab.fi_wl");
      if (!N || !WL) continue;
      mlir::BlockArgument Arg = F.getArgument(I);
      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(Arg.getType())) continue;
      // Pick the storage class — same rule as Stage C: smallest
      // native int that fits the WL.
      unsigned W = (unsigned)WL.getInt();
      unsigned StorBits = W <= 8 ? 8 : (W <= 16 ? 16 : (W <= 32 ? 32 : 64));
      (void)tryRewriteArg(Arg, N.getInt(), StorBits);
    }
  });

  // Dead-code-eliminate runtime-call helpers that survived the
  // rewrite without consumers. The `__subscript_store` lowering
  // wraps each scalar value in a `matlab_mat_from_scalar` runtime
  // call before passing it through; once we've erased the
  // subscript_store sites those wrappers are dead. They allocate
  // heap memory at runtime, so they're not pure in the
  // strict-MLIR sense, but for synthesis they're noise — we erase
  // them rather than let HWLegalize reject them.
  llvm::SmallVector<mlir::LLVM::CallOp, 8> DeadWrappers;
  M.walk([&](mlir::LLVM::CallOp C) {
    auto Sym = C.getCallee();
    if (!Sym) return;
    if (*Sym != "matlab_mat_from_scalar" &&
        *Sym != "matlab_mat_i64_from_scalar")
      return;
    if (C->getNumResults() != 1) return;
    if (!C->getResult(0).use_empty()) return;
    DeadWrappers.push_back(C);
  });
  for (mlir::LLVM::CallOp C : DeadWrappers) C.erase();

  // (Phase 5.1 moved the matlab_fi_sat_s64 / _u64 handling into
  // LowerFiSaturate, which emits an explicit clamp circuit
  // instead of the simple passthrough DCE that was here. The
  // earlier passthrough was correct only for Wrap-mode fi —
  // the explicit clamp gives correct Saturate semantics
  // regardless of overflow mode and produces identical results
  // for in-range values, so existing fixtures continue to
  // function unchanged.)

  // Collapse `arith.trunci(W → N) of arith.extsi(M → W)` chains
  // produced by the saturate-replacement step into a single
  // narrowing/widening op. Without this, the fi-arithmetic
  // widening leaves dead intermediate i64 signals that Verilator's
  // UNUSEDSIGNAL warning flags. Iterate to fixpoint — chains of
  // multiple extsi/trunci collapse step by step.
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
