#pragma once

#include "matlab/Flowchart/Loader.h"

#include <string>
#include <unordered_map>

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

/// Reverse lookup from `(file_id, line)` in the original `.mflow` to
/// the block id that produced the statement at that position. Used by
/// the DAP server's `stackTrace` handler to surface the originating
/// block id in each frame's name so the IDE can highlight the active
/// block on the canvas (Phase 8a). Key encoding: high 32 bits =
/// `file_id`, low 32 bits = 1-based line number.
struct BlockLineMap {
  std::unordered_map<int64_t, std::string> Lookup;
  static int64_t key(uint32_t File, uint32_t Line) {
    return (static_cast<int64_t>(File) << 32) | static_cast<int64_t>(Line);
  }
};

/// Walk a validated `FlowDoc` and synthesize a `TranslationUnit` whose
/// `ScriptNode` body is the entry flow's control-flow chain expanded into
/// MATLAB statements, and whose `Functions` list contains every
/// `function`-kind sub-flow plus any custom-block-provided functions.
/// The returned TU is owned by `Ctx`; per-block synthetic MATLAB buffers
/// ("<flow:NODEID>") are added to `SM` so any downstream diagnostics
/// still resolve to a concrete file:line:column.
///
/// When `OutBlockMap` is non-null, every node's `(file_id, line)`
/// position is recorded there with its block id. The DAP / LSP
/// servers use the map to surface block context on debug stops.
TranslationUnit *buildAST(const FlowDoc &Doc, ASTContext &Ctx,
                          SourceManager &SM, DiagnosticEngine &Diag,
                          const BuildOptions &Opts = {},
                          BlockLineMap *OutBlockMap = nullptr);

} // namespace flowchart
} // namespace matlab
