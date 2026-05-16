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

struct LoweringResult {
  std::string MatlabSource;          // full generated .m text
  std::string TickFunction;          // name of the chart-tick entry function
  std::string InitFunction;          // name of the chart-init entry function
};

// Emit a runnable MATLAB program for the chart's entry flow. Returns
// nullopt on any unsupported feature (Tier 4b carves out: super-
// transitions across hierarchy levels, temporal operators, in(),
// history junctions outside the most-common pattern, and chart-fn
// nodes — those are deferred to later tiers).
std::optional<LoweringResult> lowerChartToMatlab(const Chart &C,
                                                 DiagnosticEngine &Diag);

} // namespace statechart
} // namespace matlab
