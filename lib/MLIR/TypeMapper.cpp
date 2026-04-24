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
  case Dtype::Int8:    return B.getIntegerType(8,  /*isSigned=*/true);
  case Dtype::Int16:   return B.getIntegerType(16, true);
  case Dtype::Int32:   return B.getIntegerType(32, true);
  case Dtype::Int64:   return B.getIntegerType(64, true);
  case Dtype::UInt8:   return B.getIntegerType(8,  /*isSigned=*/false);
  case Dtype::UInt16:  return B.getIntegerType(16, false);
  case Dtype::UInt32:  return B.getIntegerType(32, false);
  case Dtype::UInt64:  return B.getIntegerType(64, false);
  case Dtype::Logical: return B.getI1Type();
  case Dtype::Char:    return B.getI8Type();
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
    mlir::Type Elt = mapElementType(Ctx, A.Elt);
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
