#pragma once

#include <iosfwd>

namespace matlab {

class TranslationUnit;

namespace flowchart {

struct FlowDoc; // from Loader.h

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
///
/// When `PreserveLayoutFrom` is non-null (Phase 8d), the emitter
/// reads `ui.position` from that reference document and copies it
/// onto every newly-emitted node whose `id` matches a node in the
/// reference. Unmatched nodes (newly added blocks) fall back to the
/// auto-layout column. Stable id generation across runs
/// (`n_<kind>_<n>` ordering) makes the merge work as long as the
/// program shape is unchanged; layout edits the IDE made between
/// the two emissions are preserved.
void emitMflow(std::ostream &OS, const TranslationUnit &TU,
               const FlowDoc *PreserveLayoutFrom = nullptr);

} // namespace flowchart
} // namespace matlab
