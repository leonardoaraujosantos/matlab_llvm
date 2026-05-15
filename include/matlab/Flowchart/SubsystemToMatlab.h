#pragma once

#include "matlab/AST/AST.h"
#include "matlab/Flowchart/Loader.h"

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

// Tier-4 emit options. `TargetRate` is the global sample period
// (seconds) the codegen lane discretises every continuous block to;
// 0.0 = unset, fall back to per-block `sample_time` / `Ts` or
// `settings.solver.maxStep`. `DiscretizeMethod` is currently
// "backward_euler" (the simulator's signal_discrete_integrator
// default — uses the current input). Future: "trapezoidal" /
// "forward_euler".
struct SubsystemEmitOptions {
  double TargetRate = 0.0;
  std::string DiscretizeMethod = "backward_euler";
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
