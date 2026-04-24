// Emits Python source from an MLIR ModuleOp whose ops have already been
// lowered to a small, closed set: func / arith / scf / cf / llvm.call /
// llvm.alloca / llvm.load / llvm.store / llvm.mlir.global /
// llvm.mlir.addressof plus outlined llvm.func bodies (parfor / anonymous
// functions).
//
// The emitted file imports `matlab_runtime` (a NumPy-backed Python shim)
// and runs on CPython 3.10+. Companion to EmitC.cpp.

#include "matlab/MLIR/Passes/Passes.h"
#include "matlab/Basic/SourceManager.h"

#include "mlir/IR/BuiltinOps.h"

namespace matlab {
namespace mlirgen {

std::string emitPython(mlir::ModuleOp, bool, const matlab::SourceManager *) {
  return "# stub\n";
}

} // namespace mlirgen
} // namespace matlab
