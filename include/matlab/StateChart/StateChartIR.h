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

namespace statechart {

//===----------------------------------------------------------------------===//
// StateChartIR — the resolved hierarchical IR for an mStateflow chart.
//
// `buildChartModel` lowers a state-chart `FlowDoc` (settings.kind ==
// "state_chart") to this structure. From here Lowering (§6.3) emits
// MATLAB AST that the matlabc front-end consumes; the runtime
// (`runtime_mstateflow.cpp`) consumes the same struct at simulate
// time. See docs/mStateflow_roadmap.md §6.2.
//
// Action bodies are kept as RAW MATLAB SOURCE STRINGS at this layer;
// MATLAB-front-end parsing happens in Lowering so chart-IR clients
// (the IDE, codegen, the runtime) don't pull in MatlabParse.
//===----------------------------------------------------------------------===//

enum class Decomposition {
  Or,    // exclusive (one substate active) — Stateflow's default
  And,   // parallel (all substates concurrently active)
  Leaf,  // atomic — no substates
};

const char *decompositionName(Decomposition D);

enum class ContainerStyle {
  State,    // workhorse
  Subchart, // collapsed nested chart view
  Box,      // visual grouping without semantic state
};

const char *containerStyleName(ContainerStyle C);

enum class JunctionKind {
  Connective,  // flow branch/merge
  History,     // remembers last active sibling of its parent
  Entry,       // entry port on a compound state's boundary
  Exit,        // exit port on a compound state's boundary
  Default,     // source of a default-transition (Stateflow bullet+stub)
};

const char *junctionKindName(JunctionKind K);

enum class TransitionKind {
  Outer,    // crosses the source state's boundary
  Inner,    // self-loop that does NOT exit/re-enter the source
  Default,  // emanates from a `junction_default` source
};

const char *transitionKindName(TransitionKind K);

//===----------------------------------------------------------------------===//
// Action — entry / during / exit / on-event bodies.
//
// At chart-IR layer, the body is plain MATLAB source. Lowering parses
// it through the matlab front-end so type checking and codegen work
// uniformly across the rest of matlabc.
//===----------------------------------------------------------------------===//

struct Action {
  std::string Source;          // raw MATLAB source, may be empty
  SourceLocation Loc;
  bool empty() const { return Source.empty(); }
};

//===----------------------------------------------------------------------===//
// TransitionLabel — the four-field decomposition of a raw edge label
//   `event[guard]{condAction}/transAction`
// Each field is optional. Parsed by `parseTransitionLabel` (in
// StateChartIR.cpp); empty strings mean the field was absent.
//===----------------------------------------------------------------------===//

struct TransitionLabel {
  std::string Raw;             // the original label exactly as written
  std::string Event;           // identifier (may be empty)
  std::string Guard;           // MATLAB expression source (may be empty)
  std::string CondAction;      // MATLAB statement source (may be empty)
  std::string TransAction;     // MATLAB statement source (may be empty)
  SourceLocation Loc;
};

//===----------------------------------------------------------------------===//
// ChartJunction — non-state graph nodes (connective / history / entry /
// exit / default).
//===----------------------------------------------------------------------===//

struct ChartJunction {
  std::string Id;
  JunctionKind Kind;
  std::string ParentId;        // empty for chart root
  SourceLocation Loc;
};

//===----------------------------------------------------------------------===//
// ChartState — atomic OR compound; the discriminator is `Decomp`.
//===----------------------------------------------------------------------===//

struct ChartState {
  std::string Id;
  std::string Label;
  std::string ParentId;                 // empty for chart-root states
  Decomposition Decomp = Decomposition::Leaf;
  ContainerStyle Container = ContainerStyle::State;
  bool IsInitial = false;               // OR-children only
  bool HasHistory = false;              // compound OR parents only
  bool Atomic = false;                  // codegen hint
  // Set on every direct substate of an AND parent. Default = 0 when
  // the loader-side validator passes (it requires AND children to
  // declare it).
  std::optional<int> ExecutionOrder;

  Action Entry, During, Exit;
  // Event-name → handler body. Order preserved (declaration order)
  // for stable codegen.
  std::vector<std::pair<std::string, Action>> OnEvent;

  // Sibling lists kept in source-order for stable diagnostics + dump.
  std::vector<std::string> ChildStateIds;
  std::vector<std::string> ChildJunctionIds;

  SourceLocation Loc;
};

//===----------------------------------------------------------------------===//
// Transition — source/dest are `ChartState` OR `ChartJunction` ids.
//===----------------------------------------------------------------------===//

struct Transition {
  std::string Id;
  std::string SourceId;
  std::string DestId;
  int Priority = 1;
  TransitionKind Kind = TransitionKind::Outer;
  TransitionLabel Label;
  SourceLocation Loc;
};

//===----------------------------------------------------------------------===//
// Chart — one per state-chart `Flow`.
//===----------------------------------------------------------------------===//

struct ChartFunction {
  std::string Id;              // the node id that hosts the call-site
  std::string Name;            // identifier referenced from action bodies
  std::vector<std::string> Inputs;
  std::vector<std::string> Outputs;
  std::string Body;            // raw MATLAB source (matlab variant)
  SourceLocation Loc;
};

struct Chart {
  std::string Name;            // matches the originating Flow.Name
  flowchart::Signature Sig;
  flowchart::SymbolTable Symbols;

  std::map<std::string, ChartState>    States;
  std::map<std::string, ChartJunction> Junctions;
  std::vector<Transition> Transitions;
  std::vector<ChartFunction> Functions;

  std::vector<std::string> RootStateIds;
  std::vector<std::string> RootJunctionIds;

  // Super-step iteration cap. Mirrors Stateflow's chart property
  // `kMaxIterations` (§2-41); the runtime raises a warning when it
  // saturates rather than spinning forever.
  int MaxIterations = 1000;

  SourceLocation Loc;

  const ChartState    *findState(const std::string &Id) const;
  const ChartJunction *findJunction(const std::string &Id) const;
};

struct ChartModel {
  std::string EntryName;
  std::vector<Chart> Charts;

  const Chart *findChart(const std::string &Name) const;
  const Chart *entryChart() const { return findChart(EntryName); }
};

//===----------------------------------------------------------------------===//
// Front door — translate a state-chart FlowDoc into a ChartModel.
// On any structural error reports through `Diag` and returns nullopt.
// The Loader has already validated parent resolution / decomposition /
// default-transition multiplicity / AND-execution-order, so failures
// here are semantic (e.g. transition endpoint resolution, duplicate
// symbols, unknown junction kind).
//===----------------------------------------------------------------------===//

std::optional<ChartModel> buildChartModel(const flowchart::FlowDoc &Doc,
                                          DiagnosticEngine &Diag);

// Parse a raw transition-label string into its four sub-fields. Public
// because the IDE-side parser (Tier 1 UI) needs to reach the same
// splitter for live edit-time feedback.
TransitionLabel parseTransitionLabel(const std::string &Raw,
                                     SourceLocation Loc);

// Stable line-oriented dump used by `matlabc -dump-chart` (Tier 4d)
// and the golden-file tests.
void dumpChartModel(std::ostream &OS, const ChartModel &M);

} // namespace statechart
} // namespace matlab
