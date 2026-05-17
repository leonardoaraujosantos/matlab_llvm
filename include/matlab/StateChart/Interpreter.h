#pragma once

#include "matlab/StateChart/StateChartIR.h"

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace matlab::statechart {

//===----------------------------------------------------------------------===//
// ChartInterpreter — direct in-process simulator for a Chart IR.
//
// The MATLAB lowering (lib/StateChart/Lowering.cpp) is the codegen
// path: it produces compilable MATLAB that matlabc then routes
// through MIR / LLVM / emit-* lanes. The interpreter is a *parallel*
// path used by the live-debug surface:
//
//   - `-simulate` for a state-chart `.mflow` runs an interpreter
//     trace so the CLI prints a deterministic sequence of stateEnter
//     / stateExit / transitionFired events rather than just the
//     initial active configuration.
//   - The chart-namespaced DAP server (`runStateChartDap`) hosts an
//     interpreter; client `stateChart/stepSuperStep` / `stepTransition`
//     calls drive a real super-step; breakpoints pause exactly where
//     they're set; `stateChart/emit` injects an event for the next
//     step.
//
// The interpreter handles the subset of MATLAB that chart actions
// and guards actually use — assignments, arithmetic, relational
// operators, boolean operators, parenthesised sub-expressions, the
// constants `true` / `false` / `pi` / `Inf` / `NaN`, plus the
// builtin chart predicate `in(stateId)`. More complex expressions
// (function calls, struct/cell access, multi-arg ops) fall back to
// a diagnostic — the user can always run the lowered MATLAB path
// through the JIT for full-fidelity execution.
//===----------------------------------------------------------------------===//

struct ChartTraceEvent {
  enum class Kind {
    SuperStepBegin,
    SuperStepEnd,
    StateEnter,
    StateExit,
    TransitionFired,
    EventBroadcast,
    Breakpoint,
    MaxIterations,
  };
  Kind K;
  std::string Id;            // state id / transition id / event name
  std::string Src, Dst;      // transition src/dst (else empty)
  std::string EventName;     // for transitionFired: the triggering event (may be empty)
  int Iteration = 0;
  bool Quiescent = false;
  // Breakpoint payload — what kind of breakpoint stopped the run.
  // One of "stateEnter" / "stateExit" / "transition" / "" (none).
  std::string BreakpointReason;
};

class ChartInterpreter {
public:
  explicit ChartInterpreter(const Chart &C);

  // Enter the chart's initial configuration: walk default substates
  // top-down, fire entry actions, record one StateEnter per state
  // entered. Idempotent — calling twice is a no-op after the first.
  std::vector<ChartTraceEvent> initialize();

  // Broadcast an event for the upcoming super-step. The event stays
  // set until the next `superStep` (or `stepTransition`) clears it.
  void emit(const std::string &EventName);

  // Run a single super-step (priority-ordered transition fire loop)
  // until either:
  //   - the chart quiesces (no transition fired this iteration), or
  //   - a breakpoint fires, or
  //   - the iteration count hits `MaxIterations` (emits a
  //     MaxIterations trace event).
  // Returns trace events generated, in fire order.
  std::vector<ChartTraceEvent> superStep();

  // Run until the next transition fires, then halt. Equivalent to a
  // single fired step (no quiescence pass). Useful for the DAP
  // `stateChart/stepTransition` verb.
  std::vector<ChartTraceEvent> stepTransition();

  bool isActive(const std::string &StateId) const;
  std::vector<std::string> activeStates() const;

  // Local-data accessors — let driver scripts / the REPL set inputs
  // before stepping and read outputs after.
  void setLocal(const std::string &Name, double Value);
  std::optional<double> getLocal(const std::string &Name) const;
  // Enumerate every local data slot currently held by the interpreter.
  // Used by the IDE introspection DAP path so the inspector / Active-
  // State pane can render live values during a pause without
  // re-walking the chart IR.
  std::vector<std::pair<std::string, double>> allLocals() const;

  // Breakpoint surface. `setX` replaces the entire breakpoint set
  // (DAP convention).
  void setStateEnterBreakpoints(const std::vector<std::string> &Ids);
  void setStateExitBreakpoints(const std::vector<std::string> &Ids);
  void setTransitionBreakpoints(const std::vector<std::string> &Ids);
  // Symbol-change watchpoints — pause when any of these locals is
  // written by an action. The trace event carries the symbol name in
  // `Id` and "symbolChange" as the BreakpointReason; the new value is
  // observable via `getLocal`. Tier-5 surface (right-click symbol in
  // Symbols pane → "Break on change").
  void setSymbolBreakpoints(const std::vector<std::string> &Names);

  // Operating-point save/restore (Tier 6). Snapshots are kept inside
  // the interpreter; `restore` is value-overwrite, not patch.
  struct Snapshot {
    std::unordered_map<std::string, std::string> Regions;
    std::unordered_map<std::string, double>      Locals;
    std::unordered_map<std::string, std::string> History;
  };
  Snapshot snapshot() const;
  void     restore(const Snapshot &);

  // Configurable super-step cap (mirrors Chart.MaxIterations).
  int MaxIterations;

private:
  const Chart &C_;
  std::unordered_map<std::string, std::string> Regions_;
  std::unordered_map<std::string, double>      Locals_;
  std::unordered_set<std::string>              Events_;
  // History junctions: when a substate is exited, record its id
  // against its parent so a future re-entry that goes through a
  // history junction can target the recorded substate.
  std::unordered_map<std::string, std::string> History_;
  bool Initialized_ = false;

  std::unordered_set<std::string> StateEnterBP_;
  std::unordered_set<std::string> StateExitBP_;
  std::unordered_set<std::string> TransitionBP_;
  std::unordered_set<std::string> SymbolBP_;
  // Trace accumulator used during execAction so a `state.locals.X`
  // write inside a chart body can push a Breakpoint trace event
  // alongside the rest of the super-step trace. Reset per call.
  std::vector<ChartTraceEvent> *ActionTrace_ = nullptr;
  // Tier-N temporal-operator state. TickCount_ advances once per
  // super-step; EntryTimes_[id] records the tick value when state id
  // was last entered. `after`/`before`/`every`/`at` consult them.
  int TickCount_ = 0;
  std::unordered_map<std::string, int> EntryTimes_;
  // Counter-style temporal: `temporalCount(event)` per (state, event)
  // pair. Increments once per super-step in which the owner is
  // active and the event was broadcast. Cleared on state entry. The
  // parser looks up entries by (Interp_.actionOwner(), event-name).
  std::map<std::pair<std::string, std::string>, int> TempCounts_;
  // `duration(expr)` per (state, raw-expr-text) pair. `first` is the
  // active flag (1 while the inner expression holds continuously);
  // `second` is the tick_count snapshot when the current run began.
  // Cleared on state entry. The parser captures the raw expression
  // text between balanced parens via the lexer's slice() helper, and
  // routes evaluation through `durationOf` below — which both reads
  // and maintains the slot, since the interpreter resolves duration
  // calls lazily on every guard / cond evaluation rather than via a
  // pre-super-step maintenance pass.
  std::map<std::pair<std::string, std::string>,
           std::pair<int, int>> Durations_;
  // Owning state id for the action currently being evaluated. Set
  // by execAction / evalGuard before invoking the Parser so the
  // temporal-operator builtins can compute (TickCount_ -
  // EntryTimes_[owner]).
  std::string ActionOwner_;

  // The accumulator passed through the step plumbing — when a
  // breakpoint hits, we stash its reason here and short-circuit
  // the rest of the super-step.
  bool BreakpointPending_ = false;

  // --- step plumbing ------------------------------------------------
  void initialEnter(std::vector<ChartTraceEvent> &Out);
  bool stepChartRoot(std::vector<ChartTraceEvent> &Out,
                     bool StopOnFirstTransition);
  bool stepStateContext(const std::string &StateId,
                        std::vector<ChartTraceEvent> &Out,
                        bool StopOnFirstTransition);
  bool stepActiveSubstate(const std::string &SubstateId,
                          std::vector<ChartTraceEvent> &Out,
                          bool StopOnFirstTransition);
  void fireTransition(const Transition &T,
                      std::vector<ChartTraceEvent> &Out);
  // Walks transition `T` through any connective / entry / exit / history
  // junctions in its destination chain. Returns the final-state id and
  // the ordered list of segments whose trans actions must run on
  // commit. nullopt means the chain can't reach a state (any junction
  // has no outgoing whose guard succeeds) — caller backtracks.
  struct ResolvedPath {
    std::string FinalStateId;
    std::vector<const Transition *> Segments;  // includes `T` itself
  };
  std::optional<ResolvedPath> resolvePath(const Transition &T);
  bool walkPath(const std::string &NodeId,
                std::vector<const Transition *> &Segs,
                std::string &FinalState);

  void enterState(const std::string &Id,
                  std::vector<ChartTraceEvent> &Out);
  void exitState(const std::string &Id,
                 std::vector<ChartTraceEvent> &Out);

  std::string initialSubstateOf(const ChartState &S) const;
  std::vector<const Transition *> outgoingFrom(const std::string &Id) const;

  bool evalGuard(const Transition &T);
  void execAction(const std::string &Source);
  double evalExpression(const std::string &Source);
public:
  // Temporal-operator helpers — public so the inline Parser inside
  // Interpreter.cpp can reach them without a friend declaration.
  int tickCount() const { return TickCount_; }
  int entryTimeOf(const std::string &Id) const {
    auto It = EntryTimes_.find(Id);
    return It == EntryTimes_.end() ? 0 : It->second;
  }
  const std::string &actionOwner() const { return ActionOwner_; }
  int tempCountOf(const std::string &State,
                  const std::string &Event) const {
    auto It = TempCounts_.find({State, Event});
    return It == TempCounts_.end() ? 0 : It->second;
  }
  // Lazy evaluator for `duration(expr)`. Called every time the parser
  // reaches a duration call site with the inner expression's current
  // bool value and its raw source text. The slot transitions are:
  //   - cond true,  !active → active=1, start=tickCount, return 0
  //   - cond true,   active → return tickCount - start
  //   - cond false,         → active=0, return 0
  // Multiple reads in the same super-step yield the same value as
  // long as `cond` evaluates the same way (Stateflow's "actions are
  // side-effect-free in cond expressions" assumption).
  int durationOf(const std::string &State,
                 const std::string &Expr, bool CondHolds);
private:
};

} // namespace matlab::statechart
