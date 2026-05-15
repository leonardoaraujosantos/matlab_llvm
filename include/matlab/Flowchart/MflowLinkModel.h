#pragma once

#include "matlab/Basic/SourceManager.h"
#include "matlab/Flowchart/Loader.h"

#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace matlab {

class DiagnosticEngine;

namespace flowchart {

//===----------------------------------------------------------------------===//
// MflowLinkModel — the flat signal-flow IR.
//
// `SignalFlowLowering` lowers a signal-flow `FlowDoc` (settings.kind ==
// "signal_flow") to this structure: the single source of truth handed
// to the simulation runtime (§7), the codegen lane (§9), and the model
// advisor. See `docs/mflow_link_roadmap.md` §6.1.
//
// A control-flow `.mflow` never reaches this path — it lowers through
// `GraphToAST` to a statement AST instead.
//===----------------------------------------------------------------------===//

// Sample-time class of a block — how the scheduler (§7.1) treats it.
enum class SampleTimeClass {
  Continuous,    // owns continuous state; integrated by the ODE solver
  Discrete,      // fires on `period + offset`
  Constant,      // emits a fixed value; evaluated once at startTime
  FixedInMinor,  // pure algebra / time function; rides the continuous step
};

const char *sampleTimeClassName(SampleTimeClass C);

struct MflBlock {
  // Flat block id. Top-level blocks keep their `.mflow` node id;
  // blocks inlined from a subsystem are prefixed `subsysId/` (§6.2).
  std::string Id;
  std::string Kind;                          // a `signal_*` kind
  std::map<std::string, std::string> Params; // resolved scalars (raw text)
  SampleTimeClass SampleClass = SampleTimeClass::FixedInMinor;
  double SamplePeriod = 0.0;   // seconds; meaningful for Discrete
  double SampleOffset = 0.0;   // seconds; meaningful for Discrete
  int ContStateCount = 0;      // continuous states this block owns
  int DiscStateCount = 0;      // discrete states this block owns
  // A loop-breaker's output in the current step does not depend on
  // its input in the same step (Integrator / Unit Delay / ZOH, and a
  // strictly-proper Transfer Fcn / State-Space). Its outgoing edges
  // are dropped from the execution-order sort graph (§6.3).
  bool IsLoopBreaker = false;
  bool LogSignal = false;      // `data.log_signal` — stream this output
  // Item-1 (Tier-I) — signal width per port. Default = 1 (scalar).
  // A block's *output width* is what downstream blocks read; its
  // *input widths* are filled in by width inference at lowering
  // time from the upstream edges. `OutWidth == 0` means
  // "scalar-or-inherit" (the block contributes its own output
  // width once its inputs are known — e.g. Gain inherits from its
  // input, Mux sums its inputs).
  int OutWidth = 1;
  // For multi-input blocks the widths line up with `Inputs_` in
  // `MflowLinkSim` once edges are resolved. The lowering doesn't
  // try to populate this — the runtime fills it on first use.
  std::vector<int> InWidths;
  // Tier F — conditional subsystem gate. Empty for an always-enabled
  // block; a flat block id (the source signal driving the enable)
  // for a block inlined from a `signal_enabled_subsystem` or for a
  // block that carries an explicit `data.enable_block`. At runtime,
  // `MflowLinkSim::evalAll` skips this block's evaluator (and zeros
  // its derivative slice) whenever the source's output is ≤ 0.
  std::string EnableSource;
  // Tier F carve-out — when true, the gate is an *edge* condition:
  // the block fires for exactly one major step on the rising edge
  // (0 → 1 transition) of `EnableSource`. Stamped onto every leaf
  // inlined from a `signal_triggered_subsystem`. Mutually exclusive
  // with the level-gated `signal_enabled_subsystem` semantics.
  bool EnableEdgeTriggered = false;
  SourceLocation Loc;          // the originating `.mflow` node
};

struct MflEdge {
  std::string Id;
  std::string FromBlock, FromPort;
  std::string ToBlock, ToPort;
};

// One entry per block that registers a zero-crossing predicate
// (Switch / Saturation / Relay …) — the runtime brackets the root
// when the predicate flips sign between two integrator steps (§7.3).
struct MflZeroCrossing {
  std::string BlockId;
  std::string Kind;
};

// Item-2 (§7.4) — algebraic-loop record. One per direct-feedthrough
// cycle the lowering kept (and didn't break with an
// Integrator/UnitDelay/ZOH). At runtime, `MflowLinkSim` re-iterates
// every block index in `Members` each major step until the outputs
// reach a fixed point under the configured tolerance. The members
// are stored in their execution-order index (into `Blocks`) so the
// solver can walk them in a deterministic order.
struct MflAlgebraicLoop {
  std::vector<size_t> Members;
};

struct MflowLinkModel {
  std::string EntryName;                  // entry flow name
  std::vector<MflBlock> Blocks;
  std::vector<MflEdge> Edges;
  // Topological order over data edges, feedback resolved through the
  // loop-breaker blocks. Indices into `Blocks`. Every block appears
  // exactly once.
  std::vector<size_t> ExecOrder;
  std::vector<MflZeroCrossing> ZeroCrossings;
  // Item-2 — algebraic loops that the lowering accepted (because
  // `settings.solver.algebraicLoopMethod != "off"`). The runtime
  // solves them via fixed-point iteration each major step. Empty
  // when the model has no direct-feedthrough cycles.
  std::vector<MflAlgebraicLoop> AlgebraicLoops;
  int ContStateCount = 0;                 // sum over blocks
  int DiscStateCount = 0;
  SolverConfig Solver;                    // resolved from settings.solver
  SnapshotConfig Snapshot;                // resolved from settings.snapshot

  const MflBlock *findBlock(const std::string &Id) const;
};

// Lower a signal-flow `FlowDoc` to the flat IR. On any error — a
// non-signal-flow document, an unknown / not-yet-supported block kind,
// a subsystem cycle, or an algebraic loop — reports through `Diag` and
// returns `std::nullopt`.
std::optional<MflowLinkModel> lowerSignalFlow(const FlowDoc &Doc,
                                              DiagnosticEngine &Diag);

// Pretty-print the IR in a stable, line-oriented form — the body of
// `matlabc -simulate --dry-run` and the Tier-B golden tests.
void dumpMflowLinkModel(std::ostream &OS, const MflowLinkModel &M);

} // namespace flowchart
} // namespace matlab
