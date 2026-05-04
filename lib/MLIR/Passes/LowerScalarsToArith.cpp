// Partial lowering of scalar matlab.* ops to the arith dialect.
//
// Only rewrites ops whose operands and results are scalar primitive types
// (f64, f32, i1, iN). Array / tensor ops are left for Phase 6+.
//
// Uses MLIR's greedy pattern rewriter. Patterns match on operation-name
// strings since the matlab dialect has no registered Op classes yet.

#include "matlab/MLIR/Passes/Passes.h"

#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {

bool isScalarFloat(mlir::Type T) {
  return mlir::isa<mlir::Float32Type, mlir::Float64Type>(T);
}
bool isScalarInt(mlir::Type T) {
  return mlir::isa<mlir::IntegerType>(T);
}

/// Shared helper for matching an unregistered matlab.* op by name and
/// asserting a single-result, N-operand scalar shape.
struct NameMatch : public mlir::RewritePattern {
  llvm::StringRef Target;
  NameMatch(llvm::StringRef From, mlir::MLIRContext *Ctx,
            mlir::PatternBenefit B = 1)
      : mlir::RewritePattern(From, B, Ctx), Target(From) {}
};

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

/// matlab.const_float : () -> fK  {value = F}  →  arith.constant F : fK
struct ConstFloatToArith : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumResults() != 1) return mlir::failure();
    mlir::Type Ty = Op->getResult(0).getType();
    if (!isScalarFloat(Ty)) return mlir::failure();
    auto V = Op->getAttrOfType<mlir::FloatAttr>("value");
    if (!V) return mlir::failure();
    auto FT = mlir::cast<mlir::FloatType>(Ty);
    auto Attr = mlir::FloatAttr::get(FT, V.getValueAsDouble());
    R.replaceOpWithNewOp<mlir::arith::ConstantOp>(Op, Ty, Attr);
    return mlir::success();
  }
};

/// matlab.const_int : () -> fK  {value = I : i64}
///   (we always type integer literals as double in Sema)
///   →  arith.constant (double)I : fK
///
/// For matlab.const_int : () -> iN, emit an integer arith.constant instead.
struct ConstIntToArith : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumResults() != 1) return mlir::failure();
    mlir::Type Ty = Op->getResult(0).getType();
    auto V = Op->getAttrOfType<mlir::IntegerAttr>("value");
    if (!V) return mlir::failure();
    if (auto FT = mlir::dyn_cast<mlir::FloatType>(Ty)) {
      auto Attr = mlir::FloatAttr::get(FT, (double)V.getInt());
      R.replaceOpWithNewOp<mlir::arith::ConstantOp>(Op, Ty, Attr);
      return mlir::success();
    }
    if (auto IT = mlir::dyn_cast<mlir::IntegerType>(Ty)) {
      auto Attr = mlir::IntegerAttr::get(IT, V.getInt());
      R.replaceOpWithNewOp<mlir::arith::ConstantOp>(Op, Ty, Attr);
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// matlab.const_logical : () -> i1 {value = bool}  →  arith.constant bool : i1
struct ConstLogicalToArith : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumResults() != 1) return mlir::failure();
    auto I1 = mlir::dyn_cast<mlir::IntegerType>(Op->getResult(0).getType());
    if (!I1 || I1.getWidth() != 1) return mlir::failure();
    auto V = Op->getAttrOfType<mlir::BoolAttr>("value");
    if (!V) return mlir::failure();
    auto Attr = mlir::IntegerAttr::get(I1, V.getValue() ? 1 : 0);
    R.replaceOpWithNewOp<mlir::arith::ConstantOp>(Op, I1, Attr);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// `int<N>` / `uint<N>` constant casts
//===----------------------------------------------------------------------===//

/// `uint8(0)`, `int16(5)`, etc. with a compile-time-constant operand
/// today route through a runtime call (`matlab_uint8_s` / `matlab_
/// int16_s` / ...) — unsynthesizable. When the operand is a literal
/// `arith.constant` (float or integer), fold the cast to a typed
/// `arith.constant` of the target integer width. The cast still goes
/// through the runtime when its operand is a runtime value, so this
/// only fires on the constant case.
struct IntCastConstantFold : public NameMatch {
  IntCastConstantFold(mlir::MLIRContext *Ctx)
      : NameMatch("matlab.call_builtin", Ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!C) return mlir::failure();
    llvm::StringRef N = C.getValue();
    unsigned W;
    bool Signed;
    if      (N == "int8")   { W = 8;  Signed = true;  }
    else if (N == "int16")  { W = 16; Signed = true;  }
    else if (N == "int32")  { W = 32; Signed = true;  }
    else if (N == "int64")  { W = 64; Signed = true;  }
    else if (N == "uint8")  { W = 8;  Signed = false; }
    else if (N == "uint16") { W = 16; Signed = false; }
    else if (N == "uint32") { W = 32; Signed = false; }
    else if (N == "uint64") { W = 64; Signed = false; }
    else return mlir::failure();
    if (Op->getNumOperands() != 1 || Op->getNumResults() != 1)
      return mlir::failure();
    auto CstOp = Op->getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
    if (!CstOp) return mlir::failure();
    int64_t Val;
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(CstOp.getValue()))
      Val = IA.getInt();
    else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(CstOp.getValue()))
      Val = (int64_t)FA.getValueAsDouble();
    else
      return mlir::failure();
    // Saturate to the target width — matches the runtime helpers
    // (matlab_int8_s / matlab_uint8_s / ...), which clamp rather than
    // wrap. uint8(-5) is 0, uint8(300) is 255, int8(200) is 127.
    auto Ty = mlir::IntegerType::get(R.getContext(), W);
    int64_t Trunc;
    if (Signed) {
      int64_t Lo = (W >= 64) ? std::numeric_limits<int64_t>::min()
                              : -(int64_t(1) << (W - 1));
      int64_t Hi = (W >= 64) ? std::numeric_limits<int64_t>::max()
                              : ((int64_t(1) << (W - 1)) - 1);
      Trunc = Val < Lo ? Lo : (Val > Hi ? Hi : Val);
    } else {
      uint64_t Hi = (W >= 64) ? ~uint64_t(0) : ((uint64_t(1) << W) - 1);
      Trunc = Val < 0 ? 0
                      : ((uint64_t)Val > Hi ? (int64_t)Hi : Val);
    }
    auto Attr = mlir::IntegerAttr::get(Ty, Trunc);
    auto NewOp = R.replaceOpWithNewOp<mlir::arith::ConstantOp>(Op, Ty, Attr);
    /* Tag unsigned-cast results so a downstream int→f64 conversion
     * (LowerIO's disp dispatch, EmitC's printf widening, etc.) widens
     * with UIToFPOp instead of SIToFPOp. Without this, uint8(255) bits
     * = 0xFF would convert as signed i8 = -1 → -1.0, losing the
     * saturation contract. */
    if (!Signed) NewOp->setAttr("matlab.unsigned", R.getUnitAttr());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// `true` / `false` literal handles
//===----------------------------------------------------------------------===//

/// Sema registers `true` / `false` as builtin function names, so a
/// MATLAB statement like `overflow = false;` lowers to a
/// `matlab.make_handle` op with `callee = "true"` or `callee =
/// "false"`. The downstream pipeline can't distinguish these from
/// real function-handle constructions, so they survive as `none`-
/// typed handles and break type-flow.
///
/// Rewrite them to `arith.constant 1 : i1` / `arith.constant 0 : i1`
/// so they become well-typed boolean constants. Matches by callee
/// string attr (the make_handle op has `none` result type by
/// default; we replace it with an i1 op so consumers see the
/// concrete boolean type).
struct TrueFalseHandleToArith : public NameMatch {
  TrueFalseHandleToArith(mlir::MLIRContext *Ctx)
      : NameMatch("matlab.make_handle", Ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!C) return mlir::failure();
    int64_t Val;
    if (C.getValue() == "true") Val = 1;
    else if (C.getValue() == "false") Val = 0;
    else return mlir::failure();
    auto I1 = R.getI1Type();
    auto Attr = mlir::IntegerAttr::get(I1, Val);
    R.replaceOpWithNewOp<mlir::arith::ConstantOp>(Op, I1, Attr);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Unary
//===----------------------------------------------------------------------===//

struct NegToArith : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumOperands() != 1 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Value A = Op->getOperand(0);
    mlir::Type Ty = Op->getResult(0).getType();
    if (A.getType() != Ty) return mlir::failure();
    if (isScalarFloat(Ty)) {
      R.replaceOpWithNewOp<mlir::arith::NegFOp>(Op, A);
      return mlir::success();
    }
    if (isScalarInt(Ty)) {
      auto Zero = mlir::arith::ConstantOp::create(
          R, Op->getLoc(), Ty, mlir::IntegerAttr::get(Ty, 0));
      R.replaceOpWithNewOp<mlir::arith::SubIOp>(Op, Zero, A);
      return mlir::success();
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Binary arithmetic
//===----------------------------------------------------------------------===//

template <typename FOp, typename IOp>
struct BinArithToArith : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Value A = Op->getOperand(0);
    mlir::Value B = Op->getOperand(1);
    // Prefer result type when set; fall back to operand type when the
    // result type is still `none` (Sema doesn't always propagate the
    // result type for unregistered matlab.* ops, especially when one
    // operand was refined late by user-call lowering or by an earlier
    // bitop rewrite).
    mlir::Type Ty = Op->getResult(0).getType();
    if (mlir::isa<mlir::NoneType>(Ty)) Ty = A.getType();
    if (A.getType() != Ty || B.getType() != Ty) return mlir::failure();
    if (isScalarFloat(Ty)) {
      R.replaceOpWithNewOp<FOp>(Op, A, B);
      return mlir::success();
    }
    if constexpr (!std::is_same_v<IOp, void>) {
      if (isScalarInt(Ty)) {
        R.replaceOpWithNewOp<IOp>(Op, A, B);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Bitwise builtins (bitand / bitor / bitxor / bitcmp / bitshift)
//===----------------------------------------------------------------------===//

/// Lowers `matlab.call_builtin @<bitop>(a, b)` to `arith.<op>i` when
/// the OPERAND types are matching scalar integers. The original
/// matlab.call_builtin result type is `none` (Sema doesn't propagate
/// builtin return types through call_builtin); the lowering picks
/// up the operand integer type and uses it for the new arith op.
/// Downstream consumers see a typed result and continue lowering.
template <typename IOp>
struct BinaryBitwiseBuiltin : public NameMatch {
  llvm::StringRef Callee;
  BinaryBitwiseBuiltin(llvm::StringRef CalleeName, mlir::MLIRContext *Ctx)
      : NameMatch("matlab.call_builtin", Ctx), Callee(CalleeName) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!C || C.getValue() != Callee) return mlir::failure();
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Type ATy = Op->getOperand(0).getType();
    mlir::Type BTy = Op->getOperand(1).getType();
    if (!isScalarInt(ATy) || ATy != BTy) return mlir::failure();
    R.replaceOpWithNewOp<IOp>(Op, ATy, Op->getOperand(0), Op->getOperand(1));
    return mlir::success();
  }
};

/// Lowers `matlab.call_builtin @bitshift(a, k)` (k a compile-time
/// constant) to `arith.shli` (k > 0) or `arith.shrui` (k < 0).
/// The original call_builtin's result type is `none` because Sema
/// doesn't propagate; the lowered op picks up the value operand's
/// integer type so downstream passes see a concrete-typed result.
struct BitshiftBuiltin : public NameMatch {
  BitshiftBuiltin(mlir::MLIRContext *Ctx)
      : NameMatch("matlab.call_builtin", Ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!C || C.getValue() != "bitshift") return mlir::failure();
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Type Ty = Op->getOperand(0).getType();
    if (!isScalarInt(Ty)) return mlir::failure();
    // Read the shift amount as a compile-time constant.
    mlir::Value Amt = Op->getOperand(1);
    int64_t K = 0;
    bool Known = false;
    if (auto Cst = Amt.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Cst.getValue())) {
        K = IA.getInt();
        Known = true;
      } else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(Cst.getValue())) {
        K = (int64_t)FA.getValueAsDouble();
        Known = true;
      }
    }
    if (!Known) return mlir::failure();
    bool Left = (K >= 0);
    int64_t Mag = Left ? K : -K;
    auto Cst = mlir::arith::ConstantOp::create(
        R, Op->getLoc(), Ty, mlir::IntegerAttr::get(Ty, Mag));
    if (Left) {
      R.replaceOpWithNewOp<mlir::arith::ShLIOp>(
          Op, Op->getOperand(0), Cst);
    } else {
      // Phase 1 SV target: unsigned right shift. Could split into
      // arith vs logical based on operand signedness if needed.
      R.replaceOpWithNewOp<mlir::arith::ShRUIOp>(
          Op, Op->getOperand(0), Cst);
    }
    return mlir::success();
  }
};

/// Lowers `matlab.call_builtin @bitcmp(a)` (bitwise NOT) to
/// `arith.xori a, -1`. Single-operand on a scalar integer.
struct BitCmpBuiltin : public NameMatch {
  BitCmpBuiltin(mlir::MLIRContext *Ctx)
      : NameMatch("matlab.call_builtin", Ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!C || C.getValue() != "bitcmp") return mlir::failure();
    if (Op->getNumOperands() != 1 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Type Ty = Op->getOperand(0).getType();
    if (!isScalarInt(Ty)) return mlir::failure();
    auto Cst = mlir::arith::ConstantOp::create(
        R, Op->getLoc(), Ty,
        mlir::IntegerAttr::get(Ty, -1));
    R.replaceOpWithNewOp<mlir::arith::XOrIOp>(Op, Op->getOperand(0), Cst);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

/// Lowers matlab.{eq,ne,lt,le,gt,ge} on scalar operands to arith.cmp{f,i}.
template <mlir::arith::CmpFPredicate FPred, mlir::arith::CmpIPredicate IPred>
struct CmpToArith : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Value A = Op->getOperand(0);
    mlir::Value B = Op->getOperand(1);
    mlir::Type OperandTy = A.getType();
    if (OperandTy != B.getType()) return mlir::failure();
    // Result must be i1.
    auto ResI = mlir::dyn_cast<mlir::IntegerType>(Op->getResult(0).getType());
    if (!ResI || ResI.getWidth() != 1) return mlir::failure();
    if (isScalarFloat(OperandTy)) {
      R.replaceOpWithNewOp<mlir::arith::CmpFOp>(Op, FPred, A, B);
      return mlir::success();
    }
    if (isScalarInt(OperandTy)) {
      R.replaceOpWithNewOp<mlir::arith::CmpIOp>(Op, IPred, A, B);
      return mlir::success();
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// matlab.matmul on scalars — degenerate: scalar * scalar is mul.
//===----------------------------------------------------------------------===//

struct ScalarMatMulToMulf : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Value A = Op->getOperand(0);
    mlir::Value B = Op->getOperand(1);
    mlir::Type Ty = Op->getResult(0).getType();
    if (A.getType() != Ty || B.getType() != Ty) return mlir::failure();
    if (isScalarFloat(Ty)) {
      R.replaceOpWithNewOp<mlir::arith::MulFOp>(Op, A, B);
      return mlir::success();
    }
    if (isScalarInt(Ty)) {
      R.replaceOpWithNewOp<mlir::arith::MulIOp>(Op, A, B);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct ScalarMatDivToDivf : public NameMatch {
  using NameMatch::NameMatch;
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Value A = Op->getOperand(0);
    mlir::Value B = Op->getOperand(1);
    mlir::Type Ty = Op->getResult(0).getType();
    if (A.getType() != Ty || B.getType() != Ty) return mlir::failure();
    if (isScalarFloat(Ty)) {
      R.replaceOpWithNewOp<mlir::arith::DivFOp>(Op, A, B);
      return mlir::success();
    }
    return mlir::failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

bool runLowerScalarsToArith(mlir::ModuleOp M) {
  mlir::MLIRContext *Ctx = M.getContext();
  mlir::RewritePatternSet Patterns(Ctx);

  Patterns.add<ConstFloatToArith>("matlab.const_float", Ctx);
  Patterns.add<ConstIntToArith>("matlab.const_int", Ctx);
  Patterns.add<ConstLogicalToArith>("matlab.const_logical", Ctx);
  Patterns.add<TrueFalseHandleToArith>(Ctx);
  Patterns.add<IntCastConstantFold>(Ctx);
  Patterns.add<NegToArith>("matlab.neg", Ctx);
  // Elementwise and "matmul-as-scalar-mul" both collapse on scalars.
  Patterns.add<BinArithToArith<mlir::arith::AddFOp, mlir::arith::AddIOp>>(
      "matlab.add", Ctx);
  Patterns.add<BinArithToArith<mlir::arith::SubFOp, mlir::arith::SubIOp>>(
      "matlab.sub", Ctx);
  Patterns.add<BinArithToArith<mlir::arith::MulFOp, mlir::arith::MulIOp>>(
      "matlab.emul", Ctx);
  Patterns.add<BinArithToArith<mlir::arith::DivFOp, void>>(
      "matlab.ediv", Ctx);
  Patterns.add<ScalarMatMulToMulf>("matlab.matmul", Ctx);
  Patterns.add<ScalarMatDivToDivf>("matlab.matdiv", Ctx);

  // Bitwise builtins. Only fire when operand types are concrete
  // matching scalar integers — type-refinement-driven, like the rest
  // of this pass. Once function signatures are refined by repeated
  // user-call lowering, bitand/bitor/bitxor sites collapse to their
  // arith counterparts, which then participate in the next round of
  // type propagation through their result type.
  Patterns.add<BinaryBitwiseBuiltin<mlir::arith::AndIOp>>("bitand", Ctx);
  Patterns.add<BinaryBitwiseBuiltin<mlir::arith::OrIOp>>("bitor", Ctx);
  Patterns.add<BinaryBitwiseBuiltin<mlir::arith::XOrIOp>>("bitxor", Ctx);
  Patterns.add<BitCmpBuiltin>(Ctx);
  Patterns.add<BitshiftBuiltin>(Ctx);

  using namespace mlir::arith;
  Patterns.add<CmpToArith<CmpFPredicate::OEQ, CmpIPredicate::eq>>(
      "matlab.eq", Ctx);
  Patterns.add<CmpToArith<CmpFPredicate::ONE, CmpIPredicate::ne>>(
      "matlab.ne", Ctx);
  Patterns.add<CmpToArith<CmpFPredicate::OLT, CmpIPredicate::slt>>(
      "matlab.lt", Ctx);
  Patterns.add<CmpToArith<CmpFPredicate::OLE, CmpIPredicate::sle>>(
      "matlab.le", Ctx);
  Patterns.add<CmpToArith<CmpFPredicate::OGT, CmpIPredicate::sgt>>(
      "matlab.gt", Ctx);
  Patterns.add<CmpToArith<CmpFPredicate::OGE, CmpIPredicate::sge>>(
      "matlab.ge", Ctx);

  mlir::GreedyRewriteConfig Config;
  (void)mlir::applyPatternsGreedily(M, std::move(Patterns), Config);
  return true;
}

} // namespace mlirgen
} // namespace matlab
