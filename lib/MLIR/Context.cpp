#include "matlab/MLIR/Context.h"

#include "matlab/MLIR/Dialect/MatlabDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

namespace matlab {
namespace mlirgen {

Context::Context() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  MatlabDialect>();

  Ctx = new mlir::MLIRContext(registry);
  // The matlab dialect is registered and accepts unknown ops inside its
  // namespace. Ops in other namespaces still need their dialect loaded.
  Ctx->loadAllAvailableDialects();
}

Context::~Context() { delete Ctx; }

} // namespace mlirgen
} // namespace matlab
