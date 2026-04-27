#include "matlab/MLIR/TypeMapper.h"

#include "matlab/Sema/Type.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

namespace matlab {
namespace mlirgen {

mlir::Type mapElementType(mlir::MLIRContext &Ctx, Dtype D) {
  mlir::Builder B(&Ctx);
  switch (D) {
  case Dtype::Double:  return B.getF64Type();
  case Dtype::Single:  return B.getF32Type();
  case Dtype::Complex:
    /* Complex values flow through the runtime as matlab_mat_c* (even
     * scalars — stored as 1x1). Map to !llvm.ptr directly so slots,
     * loads, stores, and call sites stay well-typed end-to-end. */
    return mlir::LLVM::LLVMPointerType::get(&Ctx);
  /* Use signless integers throughout — the arith dialect's ops require
   * signless operands and results. The signedness rides on producing /
   * consuming ops as semantic info (e.g. the FixedSpec attribute, or
   * the choice of arith.extsi vs arith.extui). */
  case Dtype::Int8:    return B.getIntegerType(8);
  case Dtype::Int16:   return B.getIntegerType(16);
  case Dtype::Int32:   return B.getIntegerType(32);
  case Dtype::Int64:   return B.getIntegerType(64);
  case Dtype::UInt8:   return B.getIntegerType(8);
  case Dtype::UInt16:  return B.getIntegerType(16);
  case Dtype::UInt32:  return B.getIntegerType(32);
  case Dtype::UInt64:  return B.getIntegerType(64);
  case Dtype::Logical: return B.getI1Type();
  case Dtype::Char:    return B.getI8Type();
  case Dtype::Fixed:
    /* Without the FixedSpec we can't pick a precise width; default to i64
     * so downstream LowerFixedPoint sees room for the widest stored class.
     * Sites that have the spec on hand should use mapType (which routes
     * through the ArrayType.FxSpec) for the precise i8/i16/i32/i64. */
    return B.getIntegerType(64, /*isSigned=*/true);
  case Dtype::Unknown: return B.getF64Type(); // fall back to double
  }
  return B.getF64Type();
}

mlir::Type mapShapedType(mlir::MLIRContext &Ctx, const Shape &S, mlir::Type Elt) {
  switch (S.K) {
  case Shape::Rank::Scalar:
  case Shape::Rank::Unknown:
    // Rank::Unknown -> we don't know the rank. Use unranked tensor.
    if (S.K == Shape::Rank::Scalar) return Elt;
    return mlir::UnrankedTensorType::get(Elt);
  case Shape::Rank::Vector:
  case Shape::Rank::Matrix:
  case Shape::Rank::NDArray: {
    llvm::SmallVector<int64_t, 4> Dims;
    Dims.reserve(S.Dims.size());
    for (int64_t D : S.Dims) {
      Dims.push_back(D >= 0 ? D : mlir::ShapedType::kDynamic);
    }
    return mlir::RankedTensorType::get(Dims, Elt);
  }
  }
  return Elt;
}

mlir::Type mapType(mlir::MLIRContext &Ctx, const matlab::Type *T) {
  mlir::Builder B(&Ctx);
  if (!T) return B.getNoneType();
  switch (T->K) {
  case matlab::Type::Kind::Any: return B.getNoneType();
  case matlab::Type::Kind::Array: {
    auto &A = static_cast<const ArrayType &>(*T);
    mlir::Type Elt;
    if (A.Elt == Dtype::Fixed && A.FxSpec) {
      /* Pick the smallest native integer that can hold WL bits. The
       * signedness is semantic info that rides on the op's fi_signed
       * attribute — we use *signless* MLIR integer types here so the
       * value flows through arith.* ops cleanly (arith requires signless
       * operands). LowerFixedPoint reads the signedness from the
       * attribute when picking extsi/extui, shrsi/shrui, etc. */
      uint8_t Bits = A.FxSpec->storageBits();
      Elt = B.getIntegerType(Bits == 0 ? 64 : Bits);
    } else {
      Elt = mapElementType(Ctx, A.Elt);
    }
    return mapShapedType(Ctx, A.S, Elt);
  }
  case matlab::Type::Kind::StringArray:
    // MATLAB strings as opaque handles for now — represent as !matlab.string.
    return mlir::NoneType::get(&Ctx);
  case matlab::Type::Kind::Cell:
  case matlab::Type::Kind::Struct:
  case matlab::Type::Kind::FuncHandle:
    return mlir::NoneType::get(&Ctx);
  }
  return B.getNoneType();
}

} // namespace mlirgen
} // namespace matlab
