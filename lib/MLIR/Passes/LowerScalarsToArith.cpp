// Partial lowering of scalar matlab.* ops to the arith dialect.
//
// Only rewrites ops whose operands and results are scalar primitive types
// (f64, f32, i1, iN). Array / tensor ops are left for Phase 6+.
//
// Uses MLIR's greedy pattern rewriter. Patterns match on operation-name
// strings since the matlab dialect has no registered Op classes yet.

#include "matlab/MLIR/Passes/Passes.h"

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
    mlir::Type Ty = Op->getResult(0).getType();
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
