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
bool gatherUses(mlir::Value Ptr, Site &S) {
  // First, check direct uses of the zeros result.
  for (mlir::OpOperand &U : Ptr.getUses()) {
    mlir::Operation *User = U.getOwner();
    if (isMatlabOpName(User, "matlab.store") &&
        User->getNumOperands() == 2 && U.getOperandNumber() == 0) {
      // The pointer is being stored INTO a slot.
      S.SlotStores.push_back(User);
      mlir::Value SlotV = User->getOperand(1);
      mlir::Operation *Alloc = SlotV.getDefiningOp();
      if (!isMatlabOpName(Alloc, "matlab.alloc")) return false;
      if (S.Slot && S.Slot != Alloc) return false;
      S.Slot = Alloc;
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
      if (isMatlabOpName(User, "matlab.store")) continue;  // slot store
      if (isMatlabOpName(User, "matlab.load")) {
        S.SlotLoads.push_back(User);
        // Walk the load's uses.
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

  // Every store and read index must be a compile-time integer.
  for (mlir::Operation *St : S.Stores) {
    if (St->getNumOperands() < 3) return false;
    int64_t I;
    if (!readIntConst(St->getOperand(1), I)) return false;
    if (I < 1 || I > N) return false;
  }
  for (mlir::LLVM::CallOp Rd : S.Reads) {
    if (Rd.getNumOperands() != 2) return false;
    int64_t I;
    if (!readIntConst(Rd.getOperand(1), I)) return false;
    if (I < 1 || I > N) return false;
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

  // Replace each store with GEP + store.
  for (mlir::Operation *St : S.Stores) {
    int64_t Idx;
    readIntConst(St->getOperand(1), Idx);  // already validated
    mlir::OpBuilder SB(St);
    auto IdxC = mlir::LLVM::ConstantOp::create(SB, St->getLoc(), I32,
        mlir::IntegerAttr::get(I32, Idx - 1));
    auto Gep = mlir::LLVM::GEPOp::create(
        SB, St->getLoc(), PtrTy, ArrTy, Alloca.getRes(),
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, IdxC.getRes()});
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
    int64_t Idx;
    readIntConst(Rd.getOperand(1), Idx);
    mlir::OpBuilder RB(Rd);
    auto IdxC = mlir::LLVM::ConstantOp::create(RB, Rd.getLoc(), I32,
        mlir::IntegerAttr::get(I32, Idx - 1));
    auto Gep = mlir::LLVM::GEPOp::create(
        RB, Rd.getLoc(), PtrTy, ArrTy, Alloca.getRes(),
        mlir::ArrayRef<mlir::LLVM::GEPArg>{0, IdxC.getRes()});
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

bool runLowerStaticFiArrays(mlir::ModuleOp M) {
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
