#pragma once

#include "matlab/AST/AST.h"

#include <iosfwd>

namespace mlir { class ModuleOp; }

namespace matlab {

class TypeContext;
class DiagnosticEngine;

namespace mlirgen {

class Context;

/// Lowers a typed AST into an mlir::ModuleOp. The caller is responsible for
/// keeping the Context alive as long as the module is used.
mlir::ModuleOp lowerToMLIR(Context &Ctx,
                           TypeContext &TC,
                           DiagnosticEngine &Diag,
                           const TranslationUnit &TU);

/// Dump an mlir::ModuleOp using the MLIR printer.
void printModule(std::ostream &OS, mlir::ModuleOp M);

} // namespace mlirgen
} // namespace matlab
