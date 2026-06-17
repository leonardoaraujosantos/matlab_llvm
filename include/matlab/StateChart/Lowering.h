#pragma once

#include "matlab/StateChart/StateChartIR.h"

#include <iosfwd>
#include <optional>
#include <string>

namespace matlab {

class DiagnosticEngine;

namespace statechart {

//===----------------------------------------------------------------------===//
// Chart IR → MATLAB lowering.
//
// Emits a single MATLAB source string per chart. The output is a
// callable `function [outputs, state] = <chart>_tick(state, inputs,
// events)` plus an `<chart>_init()` helper, plus one region helper
// per OR-region in the chart. The Lowering owns the symbol → state-
// field rewrite for action bodies; the matlabc front-end consumes
// the output exactly as it would any hand-written .m file.
//
// Strategy:
//   - Active state is `state.regions.<regionId> = '<substateId>'`.
//     One OR region per compound state (and one for chart-root).
//   - Chart locals + inputs + outputs live under `state.locals.X`.
//   - Events live under `state.events.X` (booleans, set by caller).
//   - Super-step is a fixed-point loop bounded by Chart.MaxIterations
//     with a runtime warning on saturation.
//   - Transitions inside one region evaluate in priority order. The
//     first fire exits the source substate (running its exit chain),
//     runs the transition's condAction + transAction, enters the
//     destination (running its entry chain), and sets `fired=true`.
//   - AND parents step every child region per super-step iteration
//     in `executionOrder`.
//
// The lowering is intentionally syntactic: action / guard bodies are
// re-emitted with chart-symbol identifiers rewritten to `state.locals.*`
// references. The lowering doesn't try to type-check actions; that
// happens when matlabc compiles the emitted .m.
//===----------------------------------------------------------------------===//

enum class LoweringTarget {
  // Default: software execution. Persistent-scalar form with a
  // top-level driver, while-loop super-step, and float-typed locals.
  // Targets matlabc's `-emit-matlab` / `-emit-llvm` / `-emit-c`
  // lanes.
  Software,
  // Synthesizable HDL. No top-level driver, one-pass tick (no
  // super-step inner loop — every clock advances one transition
  // attempt per region), per-variable `if isempty(X)` initialisers,
  // explicit integer / fixed-point types so the SV emit pipeline
  // can pick widths.
  SystemVerilog,
};

struct LoweringOptions {
  LoweringTarget Target = LoweringTarget::Software;
  // SV-only: bit width for state codes + integer-typed locals.
  // 16-bit covers up to 65k states; chart symbol locals reuse the
  // same width unless the caller overrides via fi-spec.
  int IntegerWidth = 16;
  // Software target only: prepend a 5-tick demo driver so the lowered
  // .m is a runnable script-with-functions. Disabled when the output
  // is consumed as an importable module (e.g. the cocotb Python
  // reference) — running the demo at import would dirty the chart's
  // persistent state before the cosim resets it.
  bool IncludeDemoDriver = true;
};

struct LoweringResult {
  std::string MatlabSource;          // full generated .m text
  std::string TickFunction;          // name of the chart-tick entry function
  std::string InitFunction;          // name of the chart-init entry function
};

// Emit a runnable MATLAB program for the chart's entry flow. Returns
// nullopt on any unsupported feature.
std::optional<LoweringResult> lowerChartToMatlab(const Chart &C,
                                                 DiagnosticEngine &Diag);
std::optional<LoweringResult> lowerChartToMatlab(const Chart &C,
                                                 DiagnosticEngine &Diag,
                                                 const LoweringOptions &Opts);

} // namespace statechart
} // namespace matlab
