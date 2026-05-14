#pragma once

#include "matlab/Flowchart/MflowLinkModel.h"

#include <iosfwd>
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

class MflowLinkSim {
public:
  // One per-step recorded sample of a logged signal (`data.log_signal:
  // true`, or a Scope / To Workspace block). Streamed to CSV by the
  // `-simulate` driver, and — once Tier D lands — over the DAP
  // `signalSample` event.
  struct LogSample {
    double T = 0.0;
    double Value = 0.0;
  };

  explicit MflowLinkSim(const MflowLinkModel &M);

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

private:
  const MflowLinkModel &M_;
  // Per-block input wiring: Inputs_[i] is the list of (sourceBlock,
  // sourcePortIgnored, destPort) tuples feeding block i. Outputs are
  // assumed scalar in Tier C — every block emits one `double`.
  struct InputEdge {
    size_t SrcBlock;     // index into M_.Blocks
    std::string DstPort; // the port on *this* block the value lands on
  };
  std::vector<std::vector<InputEdge>> Inputs_;

  // Each block with continuous state owns a contiguous slice of Y_.
  // StateOffset_[i] is the start of block i's slice; ContStateCount
  // on the block gives its length.
  std::vector<size_t> StateOffset_;

  // Cached per-block plant data the evaluator would otherwise reparse
  // on every step (transfer-fcn numerator/denominator coefficients,
  // state-space matrices, …). Keeps the inner loop cheap.
  struct TFCoeffs {
    bool Valid = false;
    std::vector<double> Num; // highest-order coeff first
    std::vector<double> Den; // highest-order coeff first
  };
  std::vector<TFCoeffs> TFCache_;

  double T_ = 0.0;
  size_t MajorSteps_ = 0;
  double StepSize_ = 0.01;
  // Current outputs, one scalar per block — written by `evalOutputs`,
  // read by downstream blocks via the Inputs_ wiring.
  std::vector<double> Out_;
  std::vector<double> Y_;             // continuous state, length M_.ContStateCount

  // Buffer for the logged-signal CSV: log column names in stable order
  // (block-id), parallel sample arrays.
  std::vector<std::string> LogNames_;
  std::vector<size_t> LogBlocks_;     // index into M_.Blocks for each column
  std::vector<std::vector<LogSample>> LogColumns_;

  // Snapshot ring (§7.5). One entry per past major step, capped at
  // `SnapshotCap_` (default 256 from settings.snapshot.depth). The
  // back is the most-recent snapshot — what stepBackMajor pops.
  struct Snapshot {
    double T;
    size_t MajorSteps;
    std::vector<double> Y;
    std::vector<double> Out;
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

  // Resolve `params.<key>` to a double, falling back to `Def`.
  static double paramD(const MflBlock &B, const char *Key, double Def);
};

} // namespace flowchart
} // namespace matlab
