#include "matlab/MLIR/Dialect/MatlabDialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

namespace matlab {
namespace mlirgen {

MatlabDialect::MatlabDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx,
                    mlir::TypeID::get<MatlabDialect>()) {
  // No registered ops yet — accept anything in the matlab.* namespace.
  allowUnknownOperations();
}

} // namespace mlirgen
} // namespace matlab

MLIR_DEFINE_EXPLICIT_TYPE_ID(::matlab::mlirgen::MatlabDialect)
