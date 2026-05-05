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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    // Accept arith.constant or any frontend constant op carrying a
    // `value` attribute (matlab.const_int, matlab.const_real, the
    // unregistered uint*-cast leaves the original frontend
    // constant intact). Without this, `uint16(0)` fed by a
    // `matlab.const_real 0.0` would survive as a runtime call
    // because the operand isn't an arith.constant — only `uint8`
    // happens to fold today, so multi-cycle multiplier-style
    // designs that need wider casts trip on un-typed adds.
    mlir::Operation *CstOp = Op->getOperand(0).getDefiningOp();
    if (!CstOp) return mlir::failure();
    int64_t Val;
    bool Got = false;
    if (auto AC = mlir::dyn_cast<mlir::arith::ConstantOp>(CstOp)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(AC.getValue())) {
        Val = IA.getInt(); Got = true;
      } else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(AC.getValue())) {
        Val = (int64_t)FA.getValueAsDouble(); Got = true;
      }
    }
    if (!Got) {
      // Frontend const op with a `value` attribute (matlab.const_int,
      // matlab.const_real). Read directly from the attribute.
      if (auto IA = CstOp->getAttrOfType<mlir::IntegerAttr>("value")) {
        Val = IA.getInt(); Got = true;
      } else if (auto FA = CstOp->getAttrOfType<mlir::FloatAttr>("value")) {
        Val = (int64_t)FA.getValueAsDouble(); Got = true;
      }
    }
    if (!Got) return mlir::failure();
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

/// Lowers `matlab.subscript(typed_int, matlab.range(hi_const, lo_const))`
/// — the bit-slice extension `x(hi:lo)` — to `arith.shrui` +
/// `arith.trunci` + `arith.andi`. Fires only when:
///   - the value operand has a typed scalar integer type (the snapshot
///     pattern + RefineSlotTypes anchors this in the HW pipeline), and
///   - the index is a `matlab.range` op with two folded i/f constant
///     bounds and no step, with hi >= lo >= 0 and hi < bitwidth(src).
/// The result type is the rounded-up next-native width (1, 8, 16, 32,
/// or 64); the andi mask collapses when slice_w is one of these widths.
struct BitsliceFromSubscript : public NameMatch {
  BitsliceFromSubscript(mlir::MLIRContext *Ctx)
      : NameMatch("matlab.subscript", Ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    if (Op->getNumOperands() != 2 || Op->getNumResults() != 1)
      return mlir::failure();
    auto NA = Op->getAttrOfType<mlir::IntegerAttr>("nindices");
    if (!NA || NA.getInt() != 1) return mlir::failure();
    mlir::Value V = Op->getOperand(0);
    auto SrcTy = mlir::dyn_cast<mlir::IntegerType>(V.getType());
    if (!SrcTy) return mlir::failure();
    mlir::Value Idx = Op->getOperand(1);
    auto *RangeOp = Idx.getDefiningOp();
    if (!RangeOp) return mlir::failure();
    auto foldOpInt = [](mlir::Value V) -> std::optional<int64_t> {
      auto *D = V.getDefiningOp();
      if (!D) return std::nullopt;
      if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D)) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
          return IA.getInt();
        if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue()))
          return (int64_t)FA.getValueAsDouble();
      }
      // matlab.const_int / matlab.const_float (pre-arith form).
      if (auto VA = D->getAttrOfType<mlir::IntegerAttr>("value"))
        return VA.getInt();
      if (auto VA = D->getAttrOfType<mlir::FloatAttr>("value"))
        return (int64_t)VA.getValueAsDouble();
      // mlir::LLVM::ConstantOp (post-arith-to-LLVM form).
      if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D)) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
          return IA.getInt();
        if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue()))
          return (int64_t)FA.getValueAsDouble();
      }
      return std::nullopt;
    };
    int64_t Hi = 0, Lo = 0;
    auto N = RangeOp->getName().getStringRef();
    if (N == "matlab.range") {
      if (RangeOp->getNumOperands() < 2) return mlir::failure();
      if (auto HasStep =
              RangeOp->getAttrOfType<mlir::BoolAttr>("has_step"))
        if (HasStep.getValue()) return mlir::failure();
      auto FH = foldOpInt(RangeOp->getOperand(0));
      auto FL = foldOpInt(RangeOp->getOperand(1));
      if (!FH || !FL) return mlir::failure();
      Hi = *FH; Lo = *FL;
    } else if (N == "llvm.call") {
      auto Cal = RangeOp->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
      if (!Cal || Cal.getValue() != "matlab_range") return mlir::failure();
      if (RangeOp->getNumOperands() != 3) return mlir::failure();
      auto FH = foldOpInt(RangeOp->getOperand(0));
      auto FStep = foldOpInt(RangeOp->getOperand(1));
      auto FL = foldOpInt(RangeOp->getOperand(2));
      if (!FH || !FStep || !FL) return mlir::failure();
      // Implicit (no-step) ranges are emitted with step=1 by
      // LowerTensorOps. Anything else means the user wrote an
      // explicit step — not the bit-slice idiom.
      if (*FStep != 1) return mlir::failure();
      Hi = *FH; Lo = *FL;
    } else {
      return mlir::failure();
    }
    int64_t SliceW = Hi - Lo + 1;
    unsigned SrcW = SrcTy.getWidth();
    if (Hi < Lo || Lo < 0 || Hi >= (int64_t)SrcW ||
        SliceW < 1 || SliceW > 64)
      return mlir::failure();
    unsigned ResW;
    if      (SliceW == 1)  ResW = 1;
    else if (SliceW <= 8)  ResW = 8;
    else if (SliceW <= 16) ResW = 16;
    else if (SliceW <= 32) ResW = 32;
    else                   ResW = 64;
    auto L = Op->getLoc();
    mlir::Value Cur = V;
    if (Lo > 0) {
      auto Sh = mlir::arith::ConstantOp::create(
          R, L, SrcTy, mlir::IntegerAttr::get(SrcTy, Lo));
      Cur = mlir::arith::ShRUIOp::create(R, L, Cur, Sh);
    }
    auto ResTy = mlir::IntegerType::get(R.getContext(), ResW);
    if (ResW < SrcW)
      Cur = mlir::arith::TruncIOp::create(R, L, ResTy, Cur);
    else if (ResW > SrcW)
      Cur = mlir::arith::ExtUIOp::create(R, L, ResTy, Cur);
    if ((unsigned)SliceW < ResW) {
      uint64_t Mask = (SliceW == 64) ? ~0ULL : ((1ULL << SliceW) - 1ULL);
      auto MaskC = mlir::arith::ConstantOp::create(
          R, L, ResTy, mlir::IntegerAttr::get(ResTy, (int64_t)Mask));
      Cur = mlir::arith::AndIOp::create(R, L, Cur, MaskC);
    }
    R.replaceOp(Op, Cur);
    return mlir::success();
  }
};

/// Lowers `matlab.call_builtin @bitslice(value) {hi, lo, src_width}`
/// (the bit-slice extension `x(hi:lo)`) to `arith.shrui` +
/// `arith.trunci` + `arith.andi`. Result type is the rounded-up
/// next-native width (1, 8, 16, 32, or 64). When slice_w is one of
/// the native widths, the andi mask collapses (mask = all-ones)
/// and is skipped. When lo == 0, the shrui collapses too.
struct BitsliceBuiltin : public NameMatch {
  BitsliceBuiltin(mlir::MLIRContext *Ctx)
      : NameMatch("matlab.call_builtin", Ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &R) const override {
    auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
    if (!C || C.getValue() != "bitslice") return mlir::failure();
    if (Op->getNumOperands() != 1 || Op->getNumResults() != 1)
      return mlir::failure();
    mlir::Value V = Op->getOperand(0);
    auto SrcTy = mlir::dyn_cast<mlir::IntegerType>(V.getType());
    if (!SrcTy) return mlir::failure();
    auto HiA = Op->getAttrOfType<mlir::IntegerAttr>("hi");
    auto LoA = Op->getAttrOfType<mlir::IntegerAttr>("lo");
    if (!HiA || !LoA) return mlir::failure();
    int64_t Hi = HiA.getInt();
    int64_t Lo = LoA.getInt();
    int64_t SliceW = Hi - Lo + 1;
    if (SliceW < 1 || SliceW > 64) return mlir::failure();
    // Pick the result width: round up to next native size.
    unsigned ResW;
    if      (SliceW == 1)  ResW = 1;
    else if (SliceW <= 8)  ResW = 8;
    else if (SliceW <= 16) ResW = 16;
    else if (SliceW <= 32) ResW = 32;
    else                   ResW = 64;
    auto L = Op->getLoc();
    mlir::Value Cur = V;
    // shrui by lo if lo > 0; the source-width carries through.
    if (Lo > 0) {
      auto Sh = mlir::arith::ConstantOp::create(
          R, L, SrcTy, mlir::IntegerAttr::get(SrcTy, Lo));
      Cur = mlir::arith::ShRUIOp::create(R, L, Cur, Sh);
    }
    // Truncate (or extend / no-op) to result width. Trunci only
    // narrows; extui widens; if widths match, skip.
    auto ResTy = mlir::IntegerType::get(R.getContext(), ResW);
    if (ResW < SrcTy.getWidth())
      Cur = mlir::arith::TruncIOp::create(R, L, ResTy, Cur);
    else if (ResW > SrcTy.getWidth())
      Cur = mlir::arith::ExtUIOp::create(R, L, ResTy, Cur);
    // Mask if slice_w is narrower than result_w (non-aligned slice).
    if ((unsigned)SliceW < ResW) {
      uint64_t Mask = (SliceW == 64) ? ~0ULL : ((1ULL << SliceW) - 1ULL);
      auto MaskC = mlir::arith::ConstantOp::create(
          R, L, ResTy, mlir::IntegerAttr::get(ResTy, (int64_t)Mask));
      Cur = mlir::arith::AndIOp::create(R, L, Cur, MaskC);
    }
    R.replaceOp(Op, Cur);
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
    // Original strict path: result type must already match both
    // operands. Common after the user-call refinement loop.
    if (A.getType() == Ty && B.getType() == Ty) {
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
    // Relaxed path: the result type may still be `none` (the
    // frontend doesn't propagate result types for matlab.* ops);
    // if both operands are the same scalar primitive type, use
    // that as the result. Catches the common post-Stage-F shape
    // where typed loads feed a still-`none`-typed matmul.
    if (A.getType() == B.getType()) {
      if (isScalarFloat(A.getType())) {
        R.replaceOpWithNewOp<mlir::arith::MulFOp>(Op, A, B);
        return mlir::success();
      }
      if (isScalarInt(A.getType())) {
        R.replaceOpWithNewOp<mlir::arith::MulIOp>(Op, A, B);
        return mlir::success();
      }
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
  Patterns.add<BitsliceBuiltin>(Ctx);
  Patterns.add<BitsliceFromSubscript>(Ctx);

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
