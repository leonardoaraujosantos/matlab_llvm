#pragma once

#include <iosfwd>

namespace matlab {

class TranslationUnit;

namespace flowchart {

/// Pretty-print a `TranslationUnit` as a `.mflow` JSON document. The
/// inverse of `loadMflow` + `buildAST`: every Stmt / Function / Block
/// in the TU lowers to a flowchart block, control flow is encoded
/// via per-port edges, and every non-`program` Function becomes a
/// `function`-kind sub-flow. Auto-generates `ui.position` so the
/// IDE can render the diagram on first open; the IDE may rewrite the
/// positions on save.
///
/// The output is a single canonical JSON object (one field per line,
/// 2-space indent) so successive `-emit-mflow` runs of the same TU
/// produce byte-identical output, suitable for source-control diffs
/// and golden tests.
void emitMflow(std::ostream &OS, const TranslationUnit &TU);

} // namespace flowchart
} // namespace matlab
