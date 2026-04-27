// Phase 5.4 — constant-coefficient multiplier rewrite.
//
// Walks every `arith.muli` op. When one operand is a compile-time
// `arith.constant`, rewrite to the matching shift-add tree using
// the simple-CSD coefficient patterns:
//
//     x * 0            → 0
//     x * 1            → x
//     x * -1           → 0 - x
//     x * 2^k          → x << k
//     x * -(2^k)       → 0 - (x << k)
//     x * (2^k - 1)    → (x << k) - x      (×3, ×7, ×15, ×31, ...)
//     x * (2^k + 1)    → (x << k) + x      (×5, ×9, ×17, ×33, ...)
//
// Other constants stay as ordinary `muli`. Full Booth / CSD
// recoding for arbitrary coefficients is a v2 follow-up; v1
// captures the patterns that account for most DSP coefficients
// users actually write (FIR filters' simple-fraction coefficients
// after fi-quantization typically land on these shapes).
//
// This is an SV-pipeline-only pass — the C/Python/TS backends
// emit `*` directly and the user expects matching semantics
// there.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isPow2(uint64_t U) { return U != 0 && (U & (U - 1)) == 0; }

unsigned log2u(uint64_t U) {
  unsigned R = 0;
  while ((U >>= 1) != 0) ++R;
  return R;
}

bool getIntConst(mlir::Value V, int64_t &Out) {
  auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return false;
  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
    Out = IA.getInt();
    return true;
  }
  return false;
}

/// Try to rewrite one `arith.muli`. Returns true if rewritten.
bool tryRewrite(mlir::arith::MulIOp Op) {
  mlir::Value VarOp;
  int64_t K;
  if (getIntConst(Op.getRhs(), K)) {
    VarOp = Op.getLhs();
  } else if (getIntConst(Op.getLhs(), K)) {
    VarOp = Op.getRhs();
  } else {
    return false;
  }
  mlir::Type Ty = Op.getType();
  if (!mlir::isa<mlir::IntegerType>(Ty)) return false;
  mlir::OpBuilder B(Op);
  mlir::Location L = Op.getLoc();

  // Handy: build a typed constant of the result type.
  auto ConstI = [&](int64_t V) -> mlir::Value {
    return mlir::arith::ConstantOp::create(
        B, L, Ty, mlir::IntegerAttr::get(Ty, V));
  };
  auto Shift = [&](mlir::Value V, unsigned ShAmt) -> mlir::Value {
    if (ShAmt == 0) return V;
    auto S = ConstI((int64_t)ShAmt);
    return mlir::arith::ShLIOp::create(B, L, V, S);
  };
  auto Sub = [&](mlir::Value A, mlir::Value B_) -> mlir::Value {
    return mlir::arith::SubIOp::create(B, L, A, B_);
  };
  auto Add = [&](mlir::Value A, mlir::Value B_) -> mlir::Value {
    return mlir::arith::AddIOp::create(B, L, A, B_);
  };
  auto Replace = [&](mlir::Value New) {
    Op.getResult().replaceAllUsesWith(New);
    Op.erase();
  };

  if (K == 0) {
    Replace(ConstI(0));
    return true;
  }
  if (K == 1) {
    Replace(VarOp);
    return true;
  }
  if (K == -1) {
    Replace(Sub(ConstI(0), VarOp));
    return true;
  }

  bool Neg = K < 0;
  uint64_t AbsK = Neg ? (uint64_t)(-K) : (uint64_t)K;

  // Power of 2: x << k (or 0 - (x << k)).
  if (isPow2(AbsK)) {
    unsigned Log = log2u(AbsK);
    mlir::Value V = Shift(VarOp, Log);
    if (Neg) V = Sub(ConstI(0), V);
    Replace(V);
    return true;
  }
  // 2^k - 1 (positive only): (x << k) - x.
  if (!Neg && isPow2(AbsK + 1)) {
    unsigned Log = log2u(AbsK + 1);
    mlir::Value V = Sub(Shift(VarOp, Log), VarOp);
    Replace(V);
    return true;
  }
  // 2^k + 1 (positive, AbsK ≥ 3): (x << k) + x.
  if (!Neg && AbsK >= 3 && isPow2(AbsK - 1)) {
    unsigned Log = log2u(AbsK - 1);
    mlir::Value V = Add(Shift(VarOp, Log), VarOp);
    Replace(V);
    return true;
  }
  return false;
}

} // namespace

bool runConstMulCSD(mlir::ModuleOp M) {
  llvm::SmallVector<mlir::arith::MulIOp, 8> Worklist;
  M.walk([&](mlir::arith::MulIOp Op) { Worklist.push_back(Op); });
  for (auto Op : Worklist) (void)tryRewrite(Op);
  return true;
}

} // namespace mlirgen
} // namespace matlab
