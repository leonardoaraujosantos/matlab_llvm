#pragma once

#include "matlab/Flowchart/Loader.h"

namespace matlab {

class ASTContext;
class DiagnosticEngine;
class SourceManager;
class TranslationUnit;

namespace flowchart {

/// Options controlling how `buildAST` resolves block references that
/// reach beyond the FlowDoc itself. Currently only `custom` blocks
/// consult this — `library_id` references are resolved against
/// `BlockSearchPath` in order, and `path` references are resolved
/// relative to the `.mflow` file's location.
struct BuildOptions {
  std::vector<std::string> BlockSearchPath;
  std::string MflowDirectory; // dir containing the .mflow; used for `data.path`
};

/// Walk a validated `FlowDoc` and synthesize a `TranslationUnit` whose
/// `ScriptNode` body is the entry flow's control-flow chain expanded into
/// MATLAB statements, and whose `Functions` list contains every
/// `function`-kind sub-flow plus any custom-block-provided functions.
/// The returned TU is owned by `Ctx`; per-block synthetic MATLAB buffers
/// ("<flow:NODEID>") are added to `SM` so any downstream diagnostics
/// still resolve to a concrete file:line:column.
TranslationUnit *buildAST(const FlowDoc &Doc, ASTContext &Ctx,
                          SourceManager &SM, DiagnosticEngine &Diag,
                          const BuildOptions &Opts = {});

} // namespace flowchart
} // namespace matlab
