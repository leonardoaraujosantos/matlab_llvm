#pragma once

#include "mlir/IR/Dialect.h"

namespace matlab {
namespace mlirgen {

/// Minimal `matlab` MLIR dialect.
///
/// Phase 5 keeps this deliberately thin: no registered Op classes (we still
/// emit `matlab.*` as unregistered operations), but the dialect *is*
/// registered so `mlir::verify()` recognizes the namespace instead of
/// complaining about unknown dialects. `allowUnknownOperations()` opts the
/// dialect into a permissive verification model so Phase 5 passes can rewrite
/// ops without first translating every kind to a proper Op class.
///
/// Phase 6+ will turn the main ops into C++ classes with proper verifiers.
class MatlabDialect : public mlir::Dialect {
public:
  explicit MatlabDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "matlab"; }
};

} // namespace mlirgen
} // namespace matlab

MLIR_DECLARE_EXPLICIT_TYPE_ID(::matlab::mlirgen::MatlabDialect)
