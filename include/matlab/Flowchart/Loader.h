#pragma once

#include "matlab/Basic/SourceManager.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace matlab {

class DiagnosticEngine;

namespace flowchart {

//===----------------------------------------------------------------------===//
// In-memory representation of a parsed `.mflow` file.
//
// Field shapes match `docs/flowchart_frontend.md` §2. Numeric values are
// kept as strings (zoom, x, y, ...) because the loader doesn't evaluate
// them — Phase 2's AST builder will reparse the user-facing expression
// strings (`data.expression`, `data.cond`, ...) through the existing
// matlab Lexer + Parser.
//===----------------------------------------------------------------------===//

struct Port {
  std::string Id;
  SourceLocation Loc;       // points at the port's `id` value in the JSON
};

struct Endpoint {
  std::string Node;
  std::string Port;
  SourceLocation Loc;       // points at the endpoint object
};

struct Edge {
  std::string Id;
  std::string Kind;         // "control" | "data"
  Endpoint From;
  Endpoint To;
  SourceLocation Loc;       // points at the edge object
};

struct Node {
  std::string Id;
  std::string Kind;
  std::string Label;
  std::map<std::string, std::string> Data;
  // Array-valued data fields, e.g. `data.inputs = ["x", "k"]` on a
  // custom block. Stored separately from scalar `Data` so callers
  // that only care about strings (the Phase 2/3/4 block kinds) don't
  // need to inspect a variant.
  std::map<std::string, std::vector<std::string>> DataArrays;
  // mflowLink: the `data.params` nested object the IDE writes for
  // every `signal_*` block (Matlab_llvm_ide `SignalFlowParamSpec`).
  // Kept separate from `Data` because the generic `data` parser only
  // flattens scalar fields and skips nested objects. Scalars stored
  // as raw text, exactly like `Data`. Empty on a control-flow node.
  std::map<std::string, std::string> Params;
  std::map<std::string, SourceLocation> DataLocs; // per-field byte location
  std::map<std::string, SourceLocation> ParamLocs; // per-param byte location
  std::vector<Port> InPorts;
  std::vector<Port> OutPorts;
  SourceLocation Loc;       // points at the opening `{` of the node object
  // Points at the closing `}` of the node object. Together with Loc
  // this gives the Stmt range a synthesised .mflow block covers, so
  // every line the block spans gets registered as a valid breakpoint
  // row (not just the line of the opening brace).
  SourceLocation LocEnd;
  // `ui.position.{x, y}` from the IDE-saved file. The compile path
  // doesn't use these — they're round-trip-only. The Phase 8d
  // `-emit-mflow --preserve-layout` reads them so a re-emit keeps
  // the user's hand-placed positions for unchanged blocks.
  // `HasUiPosition` distinguishes "no position recorded in source"
  // from "position is (0, 0)".
  bool HasUiPosition = false;
  int UiX = 0;
  int UiY = 0;

  bool hasData(std::string_view Key) const {
    return Data.find(std::string(Key)) != Data.end();
  }
  const std::string *getData(std::string_view Key) const {
    auto It = Data.find(std::string(Key));
    return It == Data.end() ? nullptr : &It->second;
  }
  const std::vector<std::string> *getDataArray(std::string_view Key) const {
    auto It = DataArrays.find(std::string(Key));
    return It == DataArrays.end() ? nullptr : &It->second;
  }
  bool hasParam(std::string_view Key) const {
    return Params.find(std::string(Key)) != Params.end();
  }
  const std::string *getParam(std::string_view Key) const {
    auto It = Params.find(std::string(Key));
    return It == Params.end() ? nullptr : &It->second;
  }
};

struct Signature {
  std::vector<std::string> Inputs;
  std::vector<std::string> Outputs;
};

struct Flow {
  std::string Id;
  std::string Kind;         // "program" | "function"
  std::string Name;
  Signature Sig;
  std::vector<Node> Nodes;
  std::vector<Edge> Edges;
  SourceLocation Loc;       // points at the flow object
};

//===----------------------------------------------------------------------===//
// mflowLink (signal-flow) settings — `settings.solver` / `settings.snapshot`.
//
// Only meaningful when `Settings::Kind == "signal_flow"`; absent on a
// control-flow document. Field names mirror the IDE's `SolverConfig` /
// `SnapshotConfig` (Matlab_llvm_ide `FlowchartModels.swift`). See
// `docs/mflow_link_roadmap.md` §5.1. The on-disk JSON keys are
// camelCase (`startTime`, `relTol`, …) — the loader matches the IDE's
// `JSONEncoder` output verbatim.
//===----------------------------------------------------------------------===//

struct SolverConfig {                       // settings.solver
  std::string Type      = "variable_step";  // "fixed_step" | "variable_step"
  std::string Algorithm = "ode45";          // ode45|ode23|ode23s|ode15s|euler|heun
  double StartTime = 0.0, StopTime = 10.0;
  std::string MaxStep = "auto", MinStep = "auto";   // "auto" | <seconds>
  double RelTol = 1e-3, AbsTol = 1e-6;
  bool   ZeroCrossing = true;
  std::string AlgebraicLoopMethod = "trust_region"; // trust_region|newton|off
};

struct SnapshotConfig {                     // settings.snapshot
  bool Enabled = true;
  int  Depth   = 256;
  std::string Fields = "states";            // "states"|"states+inputs"|"all"
};

struct Settings {
  bool ColumnMajor = true;
  std::string DefaultNumericType = "double";
  std::string SourceLanguage;
  // mflowLink: which `.mflow` dialect this document is. Absent on
  // disk ⇒ "control_flow" (the historical default — control-flow
  // files are byte-for-byte unaffected). "signal_flow" selects the
  // block-diagram simulation lane.
  std::string Kind = "control_flow";
  std::optional<SolverConfig>   Solver;
  std::optional<SnapshotConfig> Snapshot;
};

struct FlowDoc {
  FileID File = 0;
  std::string Schema;       // expected: "matforge.flowchart"
  std::string Version;      // expected: "0.1.0"
  std::string Entry;        // name of the entry flow
  Settings Settings;
  std::vector<Flow> Flows;

  // Returns the flow whose `name` matches Entry, or nullptr if missing.
  const Flow *entryFlow() const;
  const Flow *findFlow(std::string_view Name) const;

  // True for an mflowLink (signal-flow) document. Control-flow docs —
  // every `.mflow` shipped before mflowLink — return false.
  bool isSignalFlow() const { return Settings.Kind == "signal_flow"; }
};

//===----------------------------------------------------------------------===//
// Loader entry points
//===----------------------------------------------------------------------===//

// Parse + validate a `.mflow` file already loaded into the SourceManager.
// On any error reports through `Diag` and returns std::nullopt; warnings
// (e.g. unreachable palette nodes) don't suppress the result.
std::optional<FlowDoc> loadMflow(const SourceManager &SM, FileID File,
                                 DiagnosticEngine &Diag);

// Convenience: load a path through the SourceManager + parse.
std::optional<FlowDoc> loadMflowFromPath(SourceManager &SM,
                                         const std::string &Path,
                                         DiagnosticEngine &Diag);

// Pretty-print a FlowDoc in a stable, line-oriented form for golden tests.
// Independent of any std::cout / std::cerr so callers can route to either.
void dumpFlowDoc(std::ostream &OS, const FlowDoc &Doc);

} // namespace flowchart
} // namespace matlab
