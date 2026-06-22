#pragma once

#include "matlab/Flowchart/MflowLinkModel.h"

#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace matlab {

class DiagnosticEngine;

namespace flowchart {

//===----------------------------------------------------------------------===//
// MflowLinkSim — the in-process signal-flow simulation runtime.
//
// Owns the live state of an `MflowLinkModel` and advances time through
// it. The roadmap §7 imagines a `runtime/runtime_mflowlink.cpp`
// generated-C++/dlopen runtime; this is the Tier-C interpreter that
// gets the run-pause-step path online with the same observable
// behaviour, against the same IR. The generated-C++ lane is Tier G
// (`-emit-mflowlink-cpp`) — see `docs/mflow_link_roadmap.md`.
//
// Tier-C support set: Constant, Step, Sine, Gain, Sum, Product,
// Abs, Saturation, Integrator, Transfer Fcn (strictly proper),
// State-Space (D = 0), Scope, Display, To Workspace, Terminator,
// Mux/Demux/Switch (algebra only — zero-crossing root-finding is
// Tier E). Discrete blocks (Unit Delay / ZOH) are Tier E.
//===----------------------------------------------------------------------===//

// Tier-H carve-out — parse-only validation of a `signal_matlab_fcn`
// expression. Returns an empty string on success, or a human-readable
// error message on syntax failure. The result of a successful call is
// equivalent to what `MflowLinkSim`'s constructor would parse, so a
// caller can do the parse + error-report at lowering time and trust
// the runtime cache will succeed too.
std::string validateMatlabFcnExpression(const std::string &Expr);

// Item-4 — parse-only validation of a `signal_matlab_fcn` block's
// `params.function_body`. Returns empty on success, or a human-
// readable error otherwise. Same contract as
// `validateMatlabFcnExpression`: the lowering pass invokes this so
// syntax errors surface with a sourced diagnostic and the block id.
std::string validateMatlabFunctionBody(const std::string &Source);

//===----------------------------------------------------------------------===//
// §17.5 #8 carve-out — true MLIR JIT for `signal_matlab_fcn` bodies.
//
// MflowLinkSim ships with a scalar-only AST interpreter as the always-
// available fallback. When a richer evaluator is wanted (multi-return,
// matrix indexing, vector ops, nested user-function calls), a host
// binary (today: matlabc) can install a JIT factory here; the
// simulator's constructor consults it for every `signal_matlab_fcn`
// block and — on a successful compile — uses the JIT'd entrypoint at
// runtime instead of walking the AST.
//
// Layering: this struct keeps MatlabFlowchart's static-library
// dependency closure unchanged. The factory's implementation can pull
// in MatlabSema / MatlabMLIR / LLVM ORC freely; only the three
// function-pointer slots below are crossed at link time.
//
// Failure is silent and per-block — `Compile` returning null leaves
// `Err` populated for the lowering diagnostics layer to surface, and
// the affected block falls back to the AST interpreter.
//===----------------------------------------------------------------------===//
struct MatlabFcnJit {
  // Opaque per-block compiled-state handle. The host binary defines
  // the concrete layout (an `mlir::ExecutionEngine` plus the resolved
  // entry-point function pointer is the natural choice).
  struct Handle;

  // Compile a `function y = foo(u1, u2, ...)` body string for an
  // entry that takes `NumInputs` scalar inputs and returns a single
  // scalar. Returns the new handle (caller owns it via `Release`),
  // or null on failure with `Err` set. `NumInputs` is bounded by the
  // simulator at the wire-arity ceiling — 8 today.
  using CompileFn = Handle *(*)(const std::string &Body,
                                unsigned NumInputs,
                                std::string &Err);
  // Invoke a previously-compiled handle. `Inputs` has length
  // `NumInputs` (the same value passed to `Compile`); returns the
  // entry-point's scalar output.
  using CallFn    = double (*)(Handle *H, const double *Inputs,
                               unsigned NumInputs);
  // Tear down a handle (frees JIT'd code memory + the engine).
  using ReleaseFn = void   (*)(Handle *H);

  CompileFn Compile = nullptr;
  CallFn    Call    = nullptr;
  ReleaseFn Release = nullptr;
};

// Install / clear the process-wide JIT factory. Calls after this
// affect subsequent `MflowLinkSim` constructions; already-built
// simulators keep whatever they captured at construction time.
void setMatlabFcnJit(MatlabFcnJit Jit);

// Retrieve the currently installed factory. Returns a default-
// constructed (all-null) struct when no JIT is registered. Used by
// `MflowLinkSim`'s constructor to capture the factory snapshot.
const MatlabFcnJit &currentMatlabFcnJit();

// Parsed expression tree for a `signal_matlab_fcn` block. Public so
// the runtime cache (`MflowLinkSim::MatlabFcnCache_`) can hold
// `unique_ptr<MatlabFcnTree>` without dragging in an opaque-type
// dance. Only `MflowLinkSim.cpp` builds and walks these — callers
// shouldn't construct them directly.
struct MatlabFcnTree {
  enum class K { Num, Var, Neg, Bin, Call };
  K Kind = K::Num;
  double Num = 0.0;
  std::string Name;
  char Op = '+';
  std::vector<std::unique_ptr<MatlabFcnTree>> Children;
};

class MflowLinkSim {
public:
  // Forward declaration of the parsed-function state struct. Defined
  // in MflowLinkSim.cpp — held opaquely here so callers outside the
  // .cpp don't need to drag in the matlab AST / Lexer / Parser
  // headers.
  struct MatlabFunctionState;

  // One per-step recorded sample of a logged signal (`data.log_signal:
  // true`, or a Scope / To Workspace block). Streamed to CSV by the
  // `-simulate` driver, and — once Tier D lands — over the DAP
  // `signalSample` event.
  struct LogSample {
    double T = 0.0;
    double Value = 0.0;
  };

  explicit MflowLinkSim(const MflowLinkModel &M);
  ~MflowLinkSim();

  // (Re)initialise the state vector from per-block initial conditions,
  // clear the log buffers, set t = startTime.
  void reset();

  // Take one major step with the configured fixed step size, running
  // the evaluators in topological order and advancing the continuous
  // state by classic RK4. Returns the step size actually taken.
  double stepMajor();

  // Run from startTime to stopTime by repeated stepMajor().
  void runToCompletion();

  // CSV: header `t,<logged block id>,...` followed by one row per
  // recorded sample. The block ids are quoted only if they need it.
  void writeCsv(std::ostream &OS) const;

  double currentTime() const { return T_; }
  size_t majorStepsTaken() const { return MajorSteps_; }
  const MflowLinkModel &model() const { return M_; }

  // Diagnostic-style introspection — used by `-simulate --dump-state`
  // and the Tier-D DAP `variables` request.
  // Returns { logged-block-id → current output value }.
  std::vector<std::pair<std::string, double>> currentLoggedOutputs() const;

  // #354 — source-line breakpoints inside a MATLAB Function block. Arm a set of
  // 1-based body lines on a `signal_matlab_fcn` block; when the interpreter
  // reaches one during a step, it records the hit, which the simulate-DAP polls
  // via consumeSourceBreakpointHit() to pause the run.
  void setSourceBreakpoints(const std::string &BlockId,
                            const std::vector<int> &Lines);
  void clearSourceBreakpoints() { SourceBreakpoints_.clear(); }
  // Returns { blockId, line } of the most recent source-line hit and clears it;
  // line < 0 means "no hit".
  struct SourceHit {
    std::string BlockId;
    int Line = -1;
    std::map<std::string, double> Vars; // body locals captured at the hit line
  };
  SourceHit consumeSourceBreakpointHit();

  //===-------------------------------------------------------------===//
  // Tier-D snapshot ring (§7.5).
  //
  // Push a frozen copy of `(T_, Y_, MajorSteps_)` onto a fixed-depth
  // FIFO at the START of every major step; pop to step backwards.
  // Snapshots are taken automatically inside `stepMajor`; the public
  // surface here lets the DAP server inspect / control the ring.
  //===-------------------------------------------------------------===//
  bool stepBackMajor();
  size_t snapshotDepth() const { return Snapshots_.size(); }
  size_t snapshotCapacity() const { return SnapshotCap_; }

  //===-------------------------------------------------------------===//
  // Tier-E — block-level stepping (§7.1 — "the IDE can step block-by-
  // block through one major step").
  //
  // `BlockCursor_` is an index in [0, ExecOrder.size()] pointing at
  // the *next* block the simulation will execute. A `stepBlock` call
  // advances the cursor by one — the block itself has already been
  // evaluated as part of the most recent `stepMajor`/`reset`; the
  // cursor is the IDE's hook for highlighting one block at a time
  // and reading per-block intermediate values. Once the cursor
  // reaches the end, the next `stepBlock` commits the major step.
  //===-------------------------------------------------------------===//
  size_t blockCursor() const { return BlockCursor_; }
  // Walks through the topo-sorted block list. Returns the id of the
  // block that just became active; empty if the cursor wrapped and
  // the call instead advanced time via stepMajor.
  std::string stepBlock();
  // Reverses stepBlock — pulls the cursor back by one; at cursor 0 it
  // step-backs the most recent major step and lands the cursor at
  // ExecOrder.size().
  std::string stepBackBlock();
  // Current block-cursor active id, or empty when cursor is at end /
  // start of step (no block currently "active").
  std::string activeBlockId() const;

  //===-------------------------------------------------------------===//
  // Tier-E — zero-crossings (§7.3).
  //
  // After each major step, every block in `M_.ZeroCrossings` has its
  // predicate re-evaluated; if the sign flipped from the start of
  // the step, the simulator bisects the major-step interval to
  // bracket the crossing and re-records the state at that time. The
  // observed crossings are surfaced to the DAP server through
  // `consumeZeroCrossings`, which returns and clears the queue.
  //===-------------------------------------------------------------===//
  struct CrossingEvent {
    std::string BlockId;
    double T;
  };
  std::vector<CrossingEvent> consumeZeroCrossings();

  //===-------------------------------------------------------------===//
  // Item-2 — algebraic-loop solver diagnostics.
  //
  // When the runtime's fixed-point solver fails to converge inside
  // its iteration cap, the loop's member ids are queued here. The
  // DAP server drains the queue every step and surfaces them as
  // `stopped { reason: "algebraic loop did not converge" }`. In
  // `-simulate` (non-DAP) mode the queue is collected at end of
  // run and printed to stderr.
  //===-------------------------------------------------------------===//
  struct AlgebraicLoopFailure {
    double T;
    std::vector<size_t> Members; // indices into M_.Blocks
  };
  std::vector<AlgebraicLoopFailure> consumeAlgebraicLoopFailures();

private:
  const MflowLinkModel &M_;
  // Per-block input wiring: Inputs_[i] is the list of (sourceBlock,
  // sourcePort, destPort) tuples feeding block i. Most blocks emit a
  // single scalar in Out_[i]; multi-output blocks (state_space with a
  // multi-row C, a MATLAB Function with [y1,y2,...]) additionally publish
  // their named output ports in PortOut_[i] so a consumer wired from
  // `out2` reads the right value (#344 / #345).
  struct InputEdge {
    size_t SrcBlock;     // index into M_.Blocks
    std::string SrcPort; // the output port on the source block (may be empty)
    std::string DstPort; // the port on *this* block the value lands on
  };
  std::vector<std::vector<InputEdge>> Inputs_;
  // Named secondary outputs per block (port id -> value). Empty for the
  // single-output common case; a consumer falls back to Out_[srcBlock]
  // when the source port isn't present here.
  std::vector<std::map<std::string, double>> PortOut_;

  // Each block with continuous state owns a contiguous slice of Y_.
  // StateOffset_[i] is the start of block i's slice; ContStateCount
  // on the block gives its length.
  std::vector<size_t> StateOffset_;
  // Tier F — index into `M_.Blocks` of the gate source for each
  // block, or -1 when always enabled. Resolved once at construction
  // from `MflBlock::EnableSource` (a flat block id). `evalAll` skips
  // block I when Gate_[I] ≥ 0 and Out_[Gate_[I]] ≤ 0.
  std::vector<int> Gate_;
  // Tier F carve-out — true when the gate is edge-triggered (the
  // block fires for exactly one major step on a rising edge of the
  // gate). The level-vs-edge distinction is read from
  // `MflBlock::EnableEdgeTriggered` at construction time.
  std::vector<bool> GateEdge_;
  // Previous-step value of every block's output, used to detect
  // rising edges on gate signals. Updated at the end of each major
  // step (after the final evalAll) so the *next* step's evalAll
  // sees the just-completed step's outputs as "previous".
  std::vector<double> PrevOut_;

  // Cached per-block plant data the evaluator would otherwise reparse
  // on every step (transfer-fcn numerator/denominator coefficients,
  // state-space matrices, …). Keeps the inner loop cheap.
  struct TFCoeffs {
    bool Valid = false;
    std::vector<double> Num; // highest-order coeff first
    std::vector<double> Den; // highest-order coeff first
  };
  std::vector<TFCoeffs> TFCache_;

  // Tier-H per-block side state. Indexed by block index; only the
  // entries for blocks of the matching kind are meaningful.
  // - `TransportBuf_[I]` is the circular history for a
  //   `signal_transport_delay`. Element `.first` is `t`, `.second`
  //   is the input value at that `t`. The runtime appends one
  //   sample per major step and linear-interpolates the output at
  //   `T_ - delay`.
  struct DelayHistory {
    double Delay = 0.0;
    std::vector<std::pair<double, double>> Samples;
    double InitialOutput = 0.0;
  };
  std::vector<DelayHistory> TransportBuf_;
  // - `Lookup1D_[I]` and `Lookup2D_[I]` are the resolved breakpoint
  //   tables. Filled once at construction so the evaluator doesn't
  //   reparse the comma-separated `params.tableData` / `breakpoints*`
  //   strings every step.
  struct Lookup1D {
    bool Valid = false;
    std::vector<double> X, Y;
  };
  struct Lookup2D {
    bool Valid = false;
    std::vector<double> X, Y;       // breakpoints
    std::vector<double> Z;          // row-major, size = X.size()·Y.size()
  };
  std::vector<Lookup1D> Lookup1DCache_;
  std::vector<Lookup2D> Lookup2DCache_;
  // - `NoiseSeed_` is the per-block RNG state (xorshift64). Reseeded
  //   on `reset()` from `params.seed`.
  std::vector<uint64_t> NoiseSeed_;
  // - `DigitalLatch_` holds the registered/held output of the clocked
  //   HDL blocks (signal_dff / signal_tff / signal_counter). They sample
  //   on a `clk` rising edge (PrevOut_ vs Out_) and otherwise hold, so the
  //   value lives outside the continuous/discrete state vectors. Seeded on
  //   `reset()` from `params.initialValue` (#343).
  std::vector<double> DigitalLatch_;
  // - `HdlMem_[I]` is the per-block array state for the HDL memory blocks
  //   (#343): the shift chain of a `signal_shift_register`, the storage of a
  //   `signal_ram`, or the (read-only) contents of a `signal_rom`. Sized and
  //   seeded on `reset()`; shift/write happen once per major step on a `clk`
  //   posedge in commitDigitalRegisters().
  std::vector<std::vector<double>> HdlMem_;
  // - `ErrAccum_` / `TotAccum_` hold the running error / symbol counts for
  //   each `signal_error_rate` block (Communications BER sink, #343). They
  //   accumulate once per major step in commitDigitalRegisters() (NOT in
  //   evalAll, which runs multiple times per RK4 step); the block's output is
  //   the running ratio `ErrAccum_/TotAccum_`. Reset to 0 on `reset()`.
  std::vector<double> ErrAccum_;
  std::vector<double> TotAccum_;
  // - `RunCount_` / `RunMean_` / `RunM2_` hold the Welford streaming-statistics
  //   state for each `signal_running_stats` block (#343). Updated once per
  //   major step (in commitDigitalRegisters); the output is the running mean,
  //   variance, or std per `params.stat`. Reset to 0 on `reset()`.
  std::vector<double> RunCount_;
  std::vector<double> RunMean_;
  std::vector<double> RunM2_;
  // - `Kalman_[I]` is the per-block discrete Kalman-filter state for a
  //   `signal_kalman` block (#343): the parsed A/C/Q/R/B matrices (flat
  //   row-major) plus the running estimate `X` (length N) and error covariance
  //   `Pc` (N×N). Built once at `reset()` from the block's matrix params; the
  //   predict+update recursion runs once per major step. `Valid` is false when
  //   the matrices don't conform (the block then passes its estimate through
  //   unchanged). The N-vector estimate is published via Out_/VecOut_.
  struct KalmanState {
    int N = 0;   // state dimension (rows of A)
    int Mz = 0;  // measurement dimension (rows of C)
    int P = 0;   // control dimension (cols of B); 0 ⇒ no control input
    std::vector<double> A, C, Q, R, B; // flat row-major
    std::vector<double> X;             // current estimate, length N
    std::vector<double> Pc;            // current covariance, N×N
    bool Valid = false;
  };
  std::vector<KalmanState> Kalman_;
  // - `MatlabFcnCache_[I]` holds the parsed expression tree for a
  //   `signal_matlab_fcn` block with `params.expression`. Empty
  //   `unique_ptr` means "not a matlab_fcn block" or "the block
  //   uses the function_body path instead" or "parse failed".
  std::vector<std::unique_ptr<MatlabFcnTree>> MatlabFcnCache_;
  // - `MatlabFnCache_[I]` holds the parsed-function state for a
  //   `signal_matlab_fcn` block that uses `params.function_body`
  //   instead of `params.expression`. Item-4 MVP walks the parsed
  //   AST with a small scalar interpreter — full JIT integration
  //   is a follow-up. `MatlabFunctionState` is forward-declared
  //   publicly above; defined in the .cpp.
  std::vector<std::unique_ptr<MatlabFunctionState>> MatlabFnCache_;
  // §17.5 #8 — per-block JIT handle. Non-null iff the installed
  // `MatlabFcnJit` factory successfully compiled this block's
  // `params.function_body`. When set, the runtime evaluator dispatches
  // through the JIT'd entrypoint instead of `runMatlabFunction`. The
  // matching `NumInputs` arity is captured in `MatlabFnJitArity_[I]`
  // (resolved once at construction from the connected `u1..uN`
  // ports plus the function-body signature).
  std::vector<MatlabFcnJit::Handle *> MatlabFnJit_;
  std::vector<unsigned>               MatlabFnJitArity_;
  // Snapshot of the JIT factory captured at construction time. Held
  // by-value (three function pointers) so a later `setMatlabFcnJit`
  // call doesn't change semantics mid-run.
  MatlabFcnJit MatlabFcnJitOps_{};

  double T_ = 0.0;
  size_t MajorSteps_ = 0;
  // #354 — armed source-line breakpoints per matlab_fcn block, and the most
  // recent hit recorded by the interpreter during a step.
  std::map<std::string, std::set<int>> SourceBreakpoints_;
  SourceHit LastSourceHit_;
  double StepSize_ = 0.01;
  // Item-2 — adaptive step state for the Dormand-Prince RK4(5)
  // integrator. CurrentAdaptiveH_ is the last accepted step size;
  // it seeds the next step (warm-start). True only when
  // `settings.solver.type == "variable_step"` AND the algorithm is
  // an adaptive one (ode45 / ode23). Fixed-step mode falls back to
  // classic RK4 at `StepSize_`.
  bool   AdaptiveSolver_ = false;
  // §17.5 #3 — implicit BDF1 lane for stiff systems. Gated on
  // `settings.solver.algorithm == "ode15s"`. Fixed-step, uses
  // `StepSize_`. L-stable, so steps can be much larger than
  // DOPRI5's stability bound on stiff problems.
  bool   Implicit_ = false;
  double CurrentAdaptiveH_ = 0.01;
  // §17.5 #7 — effective tolerances after applying per-flow
  // solver overrides. Equal to `M_.Solver.{Rel,Abs}Tol` when no
  // block tightens them.
  double EffectiveRelTol_ = 1e-3;
  double EffectiveAbsTol_ = 1e-6;
  // Scalar / first-element output per block. Every evaluator writes
  // this slot whether the block is scalar or vector — downstream
  // scalar readers (the vast majority of evaluators) read it
  // unchanged, so the Tier-I introduction of vectors doesn't
  // ripple through every Out_[X] reference.
  std::vector<double> Out_;
  // Item-1 — per-block output *vector*. Populated for blocks whose
  // OutWidth_[I] > 1; held but empty for scalar blocks. Vector-
  // aware evaluators (Mux concat, Demux split, vector
  // signal_constant, scope/log when width > 1) reach in here;
  // every other reader still hits the scalar Out_[I].
  std::vector<int>                 OutWidth_;
  // §17.5 #9 — per-block 2-D shape (rows, cols). Mirrors
  // MflBlock::OutRows / OutCols stamped by the lowering's shape
  // inference. For 1-D vectors / scalars, OutRows=1 and OutCols
  // equals OutWidth so element-wise broadcast loops over OutWidth
  // unchanged.
  std::vector<int>                 OutRows_;
  std::vector<int>                 OutCols_;
  // mflow-nd-signals — per-block N-D shape (rank 1–6) over the same flat
  // VecOut_ buffer. `prod(OutShape_[I]) == OutWidth_[I]`. OutRows_/OutCols_
  // remain the 2-D projection for the legacy paths.
  std::vector<std::vector<int>>    OutShape_;
  std::vector<std::vector<double>> VecOut_;
  std::vector<double> Y_;             // continuous state, length M_.ContStateCount
  // Discrete state — one scalar slot per `Unit Delay` / `ZOH`, indexed
  // by `DiscStateOffset_[i]`. Treated by the evaluator as the current
  // *latched* output of the block; updated on tick by the scheduler.
  std::vector<double> Z_;
  std::vector<size_t> DiscStateOffset_;
  // Unit Delay needs a one-tick lag: at tick t, the new input is
  // staged into `Znext_`, then at the *end* of the tick (after every
  // discrete block at this time has been processed) the latch is
  // committed via `Z_ := Znext_`. ZOH writes both at once (no lag).
  std::vector<double> Znext_;
  // §17.5 #4 — previous-tick input value per discrete block.
  // signal_discrete_integrator (Forward Euler / Trapezoidal) needs
  // `u[n]` when computing the state update for tick n+1; our
  // scheduler fires at end-of-step so the `Out_` read at tick time
  // is `u[n+1]`. This vector stashes `u[n]` between ticks.
  std::vector<double> DiscPrevU_;
  // §17.5 #5 — u-history buffer for signal_discrete_filter. One
  // vector per block; size matches the filter order so any
  // numerator coefficient (b₀..b_M) can multiply its corresponding
  // delayed input u[n-k]. Empty for non-filter blocks.
  std::vector<std::vector<double>> FirHistory_;
  // Per-block next-fire time. `+∞` for blocks that don't have a
  // discrete sample-time; a finite value (multiple of `SamplePeriod`)
  // for Unit Delay / ZOH / future Discrete blocks.
  std::vector<double> NextFire_;

  // Block-level stepping cursor — see `stepBlock` above.
  size_t BlockCursor_ = 0;

  // Zero-crossing tracker — `ZCSign_[k]` is the predicate sign for
  // `M_.ZeroCrossings[k]` at the start of the current major step.
  // Refreshed at the end of every major step; flips trigger a
  // `CrossingEvent` push to `ZCQueue_`.
  std::vector<int> ZCSign_;
  std::vector<CrossingEvent> ZCQueue_;
  // Item-2 — algebraic-loop convergence failures since the last
  // `consumeAlgebraicLoopFailures()` call.
  std::vector<AlgebraicLoopFailure> AlgLoopFailures_;

  // Buffer for the logged-signal CSV: one *column* per element of
  // a logged block (scalar blocks ⇒ one column; vector blocks ⇒
  // OutWidth columns named `<id>[0]`, `<id>[1]`, …). LogBlocks_[k]
  // is the originating block index for column k, and
  // LogElements_[k] is the element offset inside that block's
  // vector output.
  std::vector<std::string> LogNames_;
  std::vector<size_t> LogBlocks_;
  std::vector<int>    LogElements_;
  std::vector<std::vector<LogSample>> LogColumns_;

  // Snapshot ring (§7.5). One entry per past major step, capped at
  // `SnapshotCap_` (default 256 from settings.snapshot.depth). The
  // back is the most-recent snapshot — what stepBackMajor pops.
  struct Snapshot {
    double T;
    size_t MajorSteps;
    std::vector<double> Y;
    std::vector<double> Out;
    std::vector<double> Z;          // discrete state
    std::vector<double> DiscPrevU;  // §17.5 #4 — prev-tick input
    std::vector<std::vector<double>> FirHistory; // §17.5 #5
    std::vector<double> PrevOut;    // for Tier-F edge-trigger detection
    std::vector<double> NextFire;   // scheduler queue
    std::vector<int> ZCSign;        // zero-crossing signs
    size_t LogRows;     // truncate logs back to this size on restore
  };
  std::vector<Snapshot> Snapshots_;
  size_t SnapshotCap_ = 256;
  void pushSnapshot();
  // True when the user has step-back'd one or more times: the next
  // forward step writes "over" the rewritten future, so log rows past
  // the restored count must be dropped on the next stepMajor.
  bool LogsTruncated_ = false;

  // Evaluate every block in execution order using `State` as the
  // active continuous state vector and time `T`. Writes per-block
  // outputs into `Out_`. If `Deriv` is non-null, fills it with the
  // state derivative; otherwise it's a pure output evaluation (used
  // for logging at the end of a major step).
  void evalAll(double T, const double *State, double *Deriv);

  // RHS for the global continuous state — wraps `evalAll`.
  void derivative(double T, const double *State, double *Deriv);

  // Record one sample per logged block at the current time.
  void logSample();

  // Fire every discrete block whose `NextFire_[i] ≤ T_ + eps`,
  // latching new outputs into Z_ / Znext_ and advancing the per-
  // block next-fire time by the configured sample period.
  void fireDiscreteTicks();

  // Run the once-per-major-step relay state-update sweep. Looks at
  // each `signal_relay`'s current input and, with hysteresis-aware
  // logic, may flip its latched bit. Idempotent if called more than
  // once at the same time — the predicate uses the *latched* state,
  // not transient evalAll values.
  void commitRelayState();

  // Predicate sign for `M_.ZeroCrossings[k]` at the current outputs:
  // returns +1, 0, or -1. The predicate depends on the block's kind
  // (Saturation: input vs. its rails; Switch: control vs. threshold).
  int predicateSign(size_t K) const;

  // Bisect the major step that *just* ran to bracket the zero-
  // crossing on predicate `K`. Returns the bisected time; on the way
  // it also updates Y_ to the state at that time. Used by stepMajor
  // when a sign flip is detected.
  double bisectZeroCrossing(size_t K, double TStart,
                            const std::vector<double> &YStart,
                            double TEnd);

  // Resolve `params.<key>` to a double, falling back to `Def`.
  static double paramD(const MflBlock &B, const char *Key, double Def);
};

} // namespace flowchart
} // namespace matlab
