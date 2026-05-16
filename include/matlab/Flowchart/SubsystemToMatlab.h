#pragma once

#include "matlab/AST/AST.h"
#include "matlab/Flowchart/Loader.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace matlab {

class DiagnosticEngine;

namespace flowchart {

//===----------------------------------------------------------------------===//
// SubsystemToMatlab — mflowLink Embedded Coder, Tier 1.
//
// Lower a named `Flow` (one whose nodes include `signal_inport` /
// `signal_outport` boundary tags) to a single `matlab::Function` AST
// node. Inports become the function's args (sorted by their `id` so
// `u1`/`u2`/... map deterministically), outports become its returns,
// and every internal block emits one MATLAB assign statement in topo
// order.
//
// The emitted function feeds straight into the existing matlab_llvm
// `-emit-{c,cpp,python,typescript,systemverilog}` lanes — see
// `docs/embedded_coder_roadmap.md` §3 for the architecture rationale
// (Path A: subsystem → MATLAB AST → reuse the existing 25-pass MLIR
// pipeline + per-target emitters).
//
// Tier-1 block coverage (stateless): signal_constant, signal_gain,
// signal_sum, signal_product, signal_abs, signal_saturation,
// signal_math_fcn, signal_trig_fcn, signal_relop, signal_logical,
// signal_compare_to_zero, signal_compare_to_constant, signal_mux,
// signal_demux, signal_reshape, signal_switch, signal_multiport_switch,
// signal_merge.  Stateful + continuous + HDL handled by Tiers 3–5.
//
// Block kinds outside Tier-1 coverage are reported as a sourced
// diagnostic; the pass returns nullptr.
//===----------------------------------------------------------------------===//

// Per-port fixed-point spec for HDL emit. Default `Q16.16` signed
// covers most embedded-controller datapaths; user overrides per port
// via `--fi-spec <name>=Q<W>.<F>` on the CLI.
struct FixedPointSpec {
  int Width = 32;
  int Frac  = 16;
  bool Signed = true;
};

// Tier-4 + Tier-5 emit options.
//
// `TargetRate` (Tier 4): global sample period (seconds) the codegen
// lane discretises every continuous block to; 0.0 = unset, fall back
// to per-block `sample_time` / `Ts` or `settings.solver.maxStep`.
//
// `RejectContinuous` (Tier 5): when true, continuous-time blocks
// (`signal_integrator` / `signal_transfer_fcn` / etc.) emit a
// sourced error instead of auto-discretising. SV emit sets this
// because hardware has no implicit floating-point integrator —
// the user must replace continuous blocks with their discrete
// counterparts in the source `.mflow` explicitly.
//
// `StateAsPersistent` (Tier 5): when true, stateful blocks emit
// state as MATLAB `persistent` variables (with `if isempty(...)`
// initialisation guarded by a `reset` arg) instead of as function
// args + returns. This routes through the existing matlab_llvm
// SV pipeline's persistent → register lowering and keeps subsystem
// state internal to the synthesised module.
//
// `FiDefault` / `FiSpecs`: default fixed-point format + per-port
// overrides for the synthesised function's args / returns. Only
// honoured by the HDL emit lane.
// `DiscretizeMethod` (Tier-5i): how `signal_integrator` /
// `signal_transfer_fcn` / `signal_zero_pole` / `signal_state_space`
// blocks are discretised when a sample rate is picked.
//   - "forward_euler" (default): s -> (z-1)/Ts. Strictly proper
//     continuous TFs stay strictly proper in discrete form (no direct
//     feedthrough) — preserves the loop-breaker topo-sort invariant.
//   - "tustin": s -> (2/Ts)·(z-1)/(z+1) (a.k.a. bilinear). Discrete TF
//     gains a direct-feedthrough term n_n*u[k] in the output equation.
//     Realisation switches to Direct Form II Transposed; same state
//     count (= continuous denominator degree). Better frequency
//     fidelity over the Nyquist band; users running the result inside
//     a feedback loop must ensure another delay element (Unit Delay /
//     discrete integrator) breaks the cycle.
struct SubsystemEmitOptions {
  double TargetRate = 0.0;
  std::string DiscretizeMethod = "forward_euler";
  bool RejectContinuous = false;
  bool StateAsPersistent = false;
  FixedPointSpec FiDefault{};
  std::map<std::string, FixedPointSpec> FiSpecs;
  // Tier-7e — whole-diagram tick count override. 0 = inherit from
  // `settings.solver.stopTime / TargetRate`. Honoured only by
  // `buildDiagramTU` (the standalone whole-diagram emit); the per-
  // subsystem lane has no time loop.
  int TickCount = 0;
  // Tier-7e — sink-log decimation. The diagram's time loop still
  // runs every tick, but `signal_scope` / `signal_to_workspace`
  // assignments fire only every Nth tick. Useful when stopTime is
  // long and the log would otherwise be enormous. 1 = no
  // decimation (log every tick); 0 falls back to 1.
  int LogDecimation = 1;
};

// Lower a named subsystem to a `matlab::Function`. The returned node
// is owned by `AST`. Returns nullptr on failure (with diagnostic).
matlab::Function *lowerSubsystemToMatlab(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts = {});

// Wrap the lowered subsystem in a synthesised `TranslationUnit` plus
// a concrete-typed driver call so the downstream `-emit-*` lanes
// type-refine the function's args / return to `double`. The TU shape:
//
//     _unused = <SubsystemName>(0.0, 0.0, ...);   % driver
//
//     function [y1, ...] = <SubsystemName>(u1, ...)
//       ... body ...
//     end
//
// The trailing driver call is harmless at runtime (Python: a module-
// level no-op binding; C++: a no-op `main()`); the per-target emit
// lanes use it strictly for type inference. Caller owns the TU via
// the supplied `ASTContext`.
matlab::TranslationUnit *buildSubsystemTU(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts = {});

//===----------------------------------------------------------------------===//
// Tier-7 — whole-diagram emit. Takes a `.mflow` model whose entry
// flow is a "program" (no boundary inports/outports — sources and
// sinks ARE the graph) and synthesises a top-level `simulate()`
// function that runs the diagram for the model's configured
// number of ticks. Sources (signal_sine / signal_step / etc.)
// become per-tick generators computed from the current time; sinks
// (signal_scope / signal_to_workspace) accumulate per-tick values
// into per-block log arrays returned alongside `simulate()`'s
// other outputs.
//
// The downstream `-emit-{c,cpp,python,typescript}` lanes consume
// the TU exactly the same way as per-subsystem emit: only the
// shape of the synthesised function differs. SystemVerilog
// whole-diagram is deliberately NOT supported (the SV emit lane
// stays per-subsystem; whole-diagram simulation lives on the
// host).
matlab::TranslationUnit *buildDiagramTU(
    const FlowDoc &Doc,
    const std::string &EntryFlowName,
    matlab::ASTContext &AST,
    matlab::DiagnosticEngine &Diag,
    const SubsystemEmitOptions &Opts = {});

//===----------------------------------------------------------------------===//
// Tier-7d — whole-diagram cocotb SIL emit. Render a Python testbench
// that exercises a pre-emitted SystemVerilog DUT (from one of the
// entry flow's `signal_subsystem` blocks) while running the rest of
// the diagram host-side as the SIL reference. Each tick: the host
// drives the DUT's inputs with Q<W>.<F> packed values, samples the
// DUT outputs on the clock edge, decodes them, and compares against
// the reference subsystem's Python emit. Sample mismatches abort
// the test; the harness writes a CSV of `(t, dut_y*, ref_y*, err)`
// rows on success.
//
// The caller orchestrates the SV + Python reference emits (via
// `matlabc -emit-systemverilog --subsystem` and `-emit-python
// --subsystem`); this function only renders the testbench text.
struct DiagramCocotbOptions {
  // One DUT pinned to one entry-flow signal_subsystem block. The
  // single-DUT case (Tier-7d MVP) is the common form; multi-DUT
  // (`--dut a,b,c`) is the same plumbing with a wrapper SV that
  // instantiates each DUT side-by-side with its ports prefixed by
  // the block id. Each DUT compares independently against its own
  // Python reference.
  struct DutSpec {
    std::string BlockId;           // entry-flow block id
    std::string ModuleName;        // SV module name (per-DUT)
    std::string RefModule;         // Python module name for the ref
    std::string RefClass;          // Python class name in that module
    std::vector<std::string> InputPorts;   // ordered (per-DUT)
    std::vector<std::string> OutputPorts;  // ordered (per-DUT)
    bool Sequential = false;       // DUT has clk + rst_n / reset?
  };
  // The list of DUTs. Single-DUT models populate this with one
  // entry; the harness emit loops over it for drive + sample +
  // compare. Empty is a programming error caught at emit time.
  std::vector<DutSpec> Duts;
  // SIL comparison tolerance (decoded units).
  double Tolerance = 1.0 / 65536.0;
  // Q<W>.<F> packing for every DUT port.
  int FiWidth = 32;
  int FiFrac  = 16;
  bool FiSigned = true;
  // Cycles between drive and compare: matches the SV pipeline depth.
  int Latency = 0;
  // Multi-DUT wrapper SV module name. Empty when there's only one
  // DUT (the harness uses the DUT's own module directly).
  std::string WrapperModule;
  // Tier-7d follow-up — non-DUT nested `signal_subsystem` blocks on
  // the host side. Each entry pins one entry-flow block id to its
  // Python helper class (the per-subsystem `-emit-python` output
  // alongside the test file). The harness's HostModel instantiates
  // one helper-class instance per entry and calls `step(...)` at
  // the block's site in the topo walk; that step carries the
  // helper's persistent state across ticks. Without these, host-
  // side stateful subsystems would emit a "Tier-7d follow-up"
  // diagnostic.
  struct HostHelper {
    std::string BlockId;        // entry-flow block id
    std::string ModuleName;     // sibling Python module to import
    std::string ClassName;      // helper class name inside it
    std::vector<std::string> InputPorts;   // ordered public inputs
    std::vector<std::string> OutputPorts;  // ordered public outputs
  };
  std::vector<HostHelper> HostHelpers;
};

// Returns the test_<entry>.py contents on success, or std::nullopt on
// error (diagnostic emitted via Diag).  The caller writes the string
// to disk alongside the SV + reference Python + Makefile.
std::optional<std::string> emitDiagramCocotbHarness(
    const FlowDoc &Doc,
    const std::string &EntryFlowName,
    const DiagramCocotbOptions &Opts,
    matlab::DiagnosticEngine &Diag);

//===----------------------------------------------------------------------===//
// Tier-2 class wrapper — per-target shim that bundles the functional
// `step(...)` into a class/struct holding the persistent state slots.
//
// Computed independently of `lowerSubsystemToMatlab` so the matlabc
// driver can append it to whichever emit-* lane's output. Returns the
// metadata + ready-to-emit class source for the given target.
//===----------------------------------------------------------------------===//
struct SubsystemMeta {
  std::string Name;                            // canonical entry name
  std::vector<std::string> InputNames;         // u1, u2, ... (public)
  std::vector<std::string> OutputNames;        // y1, y2, ... (public)
  std::vector<std::string> StateArgNames;      // s_<id>
  std::vector<std::string> StateReturnNames;   // s_<id>_next
  // Tier-4 — per-state-slot initial value (matches index of
  // StateArgNames). 0.0 for blocks without an `initialCondition` /
  // `initialOutput` param; otherwise the user-supplied IC. The class
  // wrapper emits these as the default-init values for the member
  // fields so a freshly-constructed object has the same starting
  // state as the simulator's t=0 snapshot.
  std::vector<double>      StateInitVals;
};

// Compute the public metadata from the named subsystem. Returns an
// empty optional if the subsystem can't be lowered (the diagnostic
// is the same one `lowerSubsystemToMatlab` would emit). The class
// wrapper sample uses InputNames + OutputNames + State*Names to
// place each member field / step-method arg.
std::optional<SubsystemMeta> describeSubsystem(
    const FlowDoc &Doc,
    const std::string &SubsystemName,
    matlab::DiagnosticEngine &Diag);

// Render a class/struct wrapper around the functional `step(...)` for
// the target language. Targets: "python" / "cpp" / "c" / "typescript".
// Returns the source text to append after the emit-* lane's output.
// The wrapper exposes:
//   - a default constructor that zero-initialises every state slot;
//   - a `step(u1, ..., uN)` method that calls the functional form,
//     latches the next-state into member fields, and returns the
//     y-tuple (or single y when M == 1).
std::string emitSubsystemClassWrapper(const SubsystemMeta &Meta,
                                       const std::string &Target);

} // namespace flowchart
} // namespace matlab
