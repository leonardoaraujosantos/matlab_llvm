#pragma once

#include "matlab/AST/AST.h"

#include <iosfwd>

namespace mlir { class ModuleOp; }

namespace matlab {

class SourceManager;
class TypeContext;
class DiagnosticEngine;

namespace mlirgen {

class Context;

/// Lowers a typed AST into an mlir::ModuleOp. The caller is responsible for
/// keeping the Context alive as long as the module is used. If `SM` is
/// non-null, generated ops carry FileLineColLoc locations derived from it
/// so downstream tooling (EmitC's #line directives, debug info) can map
/// back to the original .m source.
///
/// When `ReplMode` is true, top-level script-level Var reads and writes
/// are rerouted through matlab_ws_get_* / matlab_ws_set_* runtime calls
/// instead of using func-local slots. This makes variables persistent
/// across JIT invocations — the shape the REPL needs so a user typing
/// `x = 1` followed by `disp(x)` actually sees the earlier assignment.
mlir::ModuleOp lowerToMLIR(Context &Ctx,
                           TypeContext &TC,
                           DiagnosticEngine &Diag,
                           const TranslationUnit &TU,
                           const SourceManager *SM = nullptr,
                           bool ReplMode = false,
                           bool DebugMode = false);

/// Dump an mlir::ModuleOp using the MLIR printer.
void printModule(std::ostream &OS, mlir::ModuleOp M);

} // namespace mlirgen
} // namespace matlab
