#pragma once

namespace mlir { class ModuleOp; }

namespace matlab {
namespace mlirgen {

/// Intra-block slot promotion: matlab.alloc / matlab.load / matlab.store
/// chains that live in a single block are promoted to SSA values.
/// Returns true if anything changed.
bool runSlotPromotion(mlir::ModuleOp M);

/// Partial lowering of scalar matlab.* ops to arith.* / arith.constant.
/// Only rewrites ops whose operands and results are primitive MLIR types
/// (f32/f64/integer/i1). Array / tensor ops are left for later phases.
bool runLowerScalarsToArith(mlir::ModuleOp M);

} // namespace mlirgen
} // namespace matlab
