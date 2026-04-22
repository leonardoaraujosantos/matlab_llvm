#pragma once

// Maps matlab::Type (Sema) to mlir::Type.
//
// Scalars   -> f64 / f32 / complex / i8..i64 / u8..u64 / i1 / i8 (char)
// Arrays    -> tensor<...x ELEM>; unknown rank -> tensor<*xELEM>
// any       -> mlir::NoneType (placeholder; type inference should replace
//              these when it can)

namespace mlir { class MLIRContext; class Type; }

namespace matlab {

class Type;  // Sema
enum class Dtype : unsigned char;
struct Shape;

namespace mlirgen {

mlir::Type mapElementType(mlir::MLIRContext &Ctx, Dtype D);
mlir::Type mapShapedType(mlir::MLIRContext &Ctx, const Shape &S, mlir::Type Elt);

/// Map a Sema type to an MLIR type. Returns `none` for unknown / `any`.
mlir::Type mapType(mlir::MLIRContext &Ctx, const matlab::Type *T);

} // namespace mlirgen
} // namespace matlab
