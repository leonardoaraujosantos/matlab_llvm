#include "matlab/AST/AST.h"
#include "matlab/AST/ASTDumper.h"
#include "matlab/AST/Formatter.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Flowchart/ASTToGraph.h"
#include "matlab/Flowchart/GraphToAST.h"
#include "matlab/Flowchart/SubsystemToMatlab.h"
#include "matlab/Flowchart/Loader.h"
#include "matlab/Flowchart/MflowLinkModel.h"
#include "matlab/Flowchart/MflowLinkSim.h"
#include "matlab/StateChart/Interpreter.h"
#include "matlab/StateChart/Lowering.h"
#include "matlab/StateChart/StateChartIR.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"
#include "matlab/MIR/Lowering.h"
#include "matlab/MIR/MIR.h"
#include "matlab/MIR/Printer.h"
#if MATLAB_LLVM_WITH_MLIR
#include "matlab/MLIR/Context.h"
#include "matlab/MLIR/Lowering.h"
#include "matlab/MLIR/Passes/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"

#include <cerrno>
/* MC layer for the DAP `disassemble` request — host-triple's
 * disassembler tables turn JIT-emitted bytes back into text
 * without a full lldb integration. */
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"
#include <climits>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/RewriteDspSoForSv.h"
#include "matlab/AST/Cloner.h"
#include "matlab/Sema/CallSiteAnalyzer.h"
#include "matlab/Sema/Monomorphize.h"
#include "matlab/Sema/SemaDumper.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"
#include "matlab/Sema/TypeInference.h"
#include "matlab/Sema/DispatchDesynth.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <termios.h>
#include <unistd.h>
#include <filesystem>
#include <algorithm>
#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace matlab;

// §17.5 #8 — installer for the MLIR-JIT factory consumed by
// `MflowLinkSim` to evaluate signal_matlab_fcn bodies. Defined in
// tools/matlabc/MflowLinkJit.cpp; forward-declared here so main()
// can register it before any -simulate path runs.
namespace matlab { namespace flowchart {
void installMflowLinkJit();

#if MATLAB_LLVM_WITH_MLIR
// Tier 5 — stamp `hdl.ports` ArrayAttr on a subsystem's
// synthesised function so `runApplyPortTypePragmas` types the
// args/return at the user-requested fi format. Bypasses the
// source-text scan (`runScanHWPragmas`) — our AST is hand-built
// and the SourceManager doesn't carry `% hdl:` comments.
//
// Builds one DictionaryAttr per public input + output, plus a
// `bool` entry for the implicit `reset` arg when the subsystem
// is stateful. Default spec is signed Q16.16; overrides come from
// the `--fi-spec port=Q<W>.<F>` CLI flag.
static void stampSubsystemPortPragmas(
    mlir::ModuleOp M,
    const std::string &FnName,
    const std::vector<std::string> &Inputs,
    const std::vector<std::string> &Outputs,
    const FixedPointSpec &DefaultFi,
    const std::map<std::string, FixedPointSpec> &Overrides,
    bool HasReset) {
  auto &Ctx = *M.getContext();
  auto I64 = mlir::IntegerType::get(&Ctx, 64);
  auto buildEntry = [&](const std::string &Name,
                         const FixedPointSpec &Spec,
                         bool BoolKind = false) -> mlir::DictionaryAttr {
    llvm::SmallVector<mlir::NamedAttribute> Fields;
    Fields.push_back({mlir::StringAttr::get(&Ctx, "name"),
                      mlir::StringAttr::get(&Ctx, Name)});
    Fields.push_back({mlir::StringAttr::get(&Ctx, "kind"),
                      mlir::StringAttr::get(
                          &Ctx, BoolKind ? "bool" : "fi")});
    Fields.push_back({mlir::StringAttr::get(&Ctx, "signed"),
                      mlir::BoolAttr::get(&Ctx,
                                           BoolKind ? false : Spec.Signed)});
    Fields.push_back({mlir::StringAttr::get(&Ctx, "width"),
                      mlir::IntegerAttr::get(I64,
                                              BoolKind ? 1 : Spec.Width)});
    Fields.push_back({mlir::StringAttr::get(&Ctx, "frac"),
                      mlir::IntegerAttr::get(I64,
                                              BoolKind ? 0 : Spec.Frac)});
    return mlir::DictionaryAttr::get(&Ctx, Fields);
  };
  llvm::SmallVector<mlir::Attribute> Ports;
  auto specOf = [&](const std::string &N) -> FixedPointSpec {
    auto It = Overrides.find(N);
    return It == Overrides.end() ? DefaultFi : It->second;
  };
  for (auto &N : Inputs)  Ports.push_back(buildEntry(N, specOf(N)));
  if (HasReset) Ports.push_back(buildEntry("reset", DefaultFi, /*BoolKind=*/true));
  for (auto &N : Outputs) Ports.push_back(buildEntry(N, specOf(N)));
  auto PortsAttr = mlir::ArrayAttr::get(&Ctx, Ports);
  // Find the function by name and stamp.
  M.walk([&](mlir::func::FuncOp F) {
    if (F.getSymName() == FnName)
      F->setAttr("hdl.ports", PortsAttr);
  });
}
#endif

} }

namespace {
struct Options {
  enum class Mode { DumpTokens, DumpAST, EmitSema, DumpCallSites,
                    TestAstClone, TestMonomorphize,
                    EmitMIR, EmitMLIR,
                    EmitLLVM, EmitC, EmitCpp, EmitPython, EmitTypeScript,
                    EmitFiReport, EmitSystemVerilog, CheckSynthesizable,
                    EmitHardwareReport, EmitCocotb,
                    EmitCuda, EmitMetal, EmitOpenCL,
                    DumpFlow, DumpChart, EmitMatlab, EmitMflow,
                    EmitMflowLinkCpp, EmitTrace,
                    Check, Repl, Format, Dap, Simulate };
  Mode Mode = Mode::Check;
  bool Opt = false;
  /* mflowLink: `-simulate --dry-run` lowers a signal-flow .mflow to
   * the MflowLinkModel IR and prints the sorted execution order
   * without booting the simulation runtime — the Tier-B smoke lane
   * (docs/mflow_link_roadmap.md §14). */
  bool DryRun = false;
  /* mflowLink: `-simulate --dap` boots a DAP server on stdio (Tier-D).
   * Pauses at entry, then accepts the §10 verb set: stepMajor /
   * stepBackMajor / continue / pause / disconnect, plus the snapshot
   * ring (settings.snapshot.depth). */
  bool SimulateDap = false;
  /* `-emit-c` / `-emit-cpp` default to NOT emitting `#line` directives
   * — the cleaner output is what most users want for hand-reading the
   * generated C / C++. Pass `-line` to opt back in when you need
   * `lldb` / `gdb` to step from the compiled binary back into the
   * original .m source. `-no-line` is still accepted (and is now a
   * no-op for C / C++ since it matches the default) for any scripts
   * that have been passing it explicitly. Python emission has no
   * `#line` mechanism so the flag is silently ignored there. */
  bool NoLine = false;
  bool EmitLine = false;
  bool Doxygen = false;
  bool CppAuto = false;
  /* When true, lowering injects matlab_dbg_hook(file_id, line) at the
   * start of every statement. Enabled implicitly for -dap; exposed via
   * -g for tests and tooling that want to inspect the injected hooks
   * in the emitted MLIR / C / C++ without standing up a DAP session. */
  bool Debug = false;
  /* SystemVerilog reset convention. ASIC default is async-assert /
   * sync-deassert, active-low — `posedge clk or negedge rst_n` with
   * the reset arm `if (!rst_n)`. Sync-active-high / sync-active-low
   * are also supported via the `-sv-reset=...` flag for teams that
   * prefer sync-only reset trees. Phase 1 / Phase 2 modules without
   * persistent state do not consume this. */
  enum class SvResetKind { AsyncLow, SyncHigh, SyncLow };
  SvResetKind SvReset = SvResetKind::AsyncLow;
  /* FSM state-encoding policy for `-emit-systemverilog`. Binary
   * is the default (smallest register, synth tool re-encodes
   * anyway). One-hot picks single-bit-per-state for fastest
   * decode; gray picks reflected-binary for single-bit-transition
   * adjacency. */
  enum class SvFSMEncoding { Binary, OneHot, Gray };
  SvFSMEncoding SvFSMEnc = SvFSMEncoding::Binary;
  /* Phase 5.4 — constant-coefficient multiplier rewrite. `auto`
   * (default) enables the simple-CSD shift-add patterns; `off`
   * disables. Reserved values like `csd` / `fcsd` map to the
   * same v1 implementation today (full CSD recoding is a
   * follow-up). */
  bool SvConstMulOpt = true;
  std::string InputPath;
  /* Additional input files. When multiple `.m` files are passed, the
   * driver concatenates their contents in CLI order — the first file
   * (kept in InputPath for backward compat with single-file modes) is
   * the script entry; subsequent files contribute function definitions
   * referenced from the script. Lets a `test_<mod>.m` driver compile
   * together with its `<mod>.m` definition without manual splicing. */
  std::vector<std::string> ExtraInputs;
  /* Embedded Coder, Tier 1 — `--subsystem <name>` flag. When set with
   * a `-emit-{c,cpp,python,typescript,...}` mode and a `.mflow` input,
   * the driver runs `lowerSubsystemToMatlab` to synthesise a MATLAB
   * `function` AST from the named flow and feeds it into the regular
   * emit-* pipeline. See `docs/embedded_coder_roadmap.md` §7. Empty =
   * no subsystem-emit; the input is processed as a normal MATLAB or
   * `.mflow` source. */
  std::string Subsystem;
  /* Embedded Coder, Tier 2 — `--state-form={class,function}` flag.
   * Default = `class` for software targets: the emit-* output gets
   * a target-specific class/struct wrapper appended that carries
   * the subsystem's state slots as member fields and exposes a
   * mutating `step(u)` method. `function` leaves the raw functional
   * `[y, s_next] = step(u, s)` form alone.  SystemVerilog ignores
   * this flag (state lives in registers natively). */
  std::string StateForm = "class";
  /* Embedded Coder, Tier 4 — `--target-rate <Ts>` flag.  Sample
   * period (seconds) at which any continuous block in the chosen
   * subsystem gets auto-discretised (Forward Euler today; see
   * docs/embedded_coder_roadmap.md Tier 4).  When 0.0, the lowering
   * falls back to the block's `data.sample_time` / `params.Ts` /
   * `settings.solver.maxStep`, in that order. */
  double TargetRate = 0.0;
  /* Tier-7e — `--ticks <N>` / `--decimation <N>` flags for the
   * standalone whole-diagram emit. `Ticks` overrides the time-loop
   * count (otherwise `stopTime / Ts`); `Decimation` gates sink-log
   * writes so a long simulation can produce a manageable log array.
   * Both default to 0 / 1 respectively — the historical behaviour. */
  int Ticks = 0;
  int Decimation = 1;
  /* Embedded Coder, Tier 5 — `--fi-spec <port>=Q<W>.<F>` flag,
   * repeatable. Per-port fixed-point width / fraction overrides
   * for the SV emit lane. Without an override, every port emits
   * as the default fi spec (signed Q16.16 — 32 bits, 16 fractional).
   * Format examples:
   *   --fi-spec u=Q16.16         (signed)
   *   --fi-spec idx=UQ8.0        (unsigned, 8-bit integer)
   *   --fi-spec sig=Q32.24       (high-precision)
   */
  std::vector<std::string> FiSpecs;
  /* Embedded Coder, Tier-5i — `--discretize=forward_euler|tustin`
   * flag. Controls how continuous-time blocks (signal_integrator,
   * signal_transfer_fcn, signal_zero_pole, signal_state_space) are
   * mapped to discrete form. Default `forward_euler` preserves the
   * pre-Tier-5i behaviour (strict-proper continuous TF stays
   * strict-proper). `tustin` (bilinear) gives better frequency
   * fidelity at the cost of direct feedthrough in the output
   * equation — same state count, DF2T realisation. */
  std::string Discretize = "forward_euler";
  /* Block-library search path for `.mflow` custom blocks (Phase 4b).
   * Resolution order: command-line `--block-path DIR` entries (in CLI
   * order) followed by colon-separated entries from the
   * `MATFORGE_BLOCK_PATH` environment variable. The first directory
   * that contains a matching `.m` file wins. Ignored for non-`.mflow`
   * inputs. */
  std::vector<std::string> BlockPath;
  /* Phase 8d: when emitting a `.mflow` (`-emit-mflow` mode), copy
   * `ui.position` from this reference file for every node whose id
   * matches. Unmatched nodes (newly added blocks) fall back to the
   * column auto-layout. Empty by default — re-emitting in place
   * still produces auto-layout positions unless `--preserve-layout`
   * is supplied. */
  std::string PreserveLayoutPath;
  /* `-emit-cocotb` knobs. Output dir defaults to `<basename>_cocotb`
   * alongside the input file when CocotbOutDir is empty. Vector
   * count is the number of stimulus samples the testbench drives
   * before signing off; combinational modules use one sample per
   * cycle, sequential modules one sample per posedge.
   *
   * `CocotbLatency` aligns the comparison window for pipelined
   * modules: DUT output at cycle k+L is compared against the
   * reference's output for cycle k. Mirrors MathWorks HDL Verifier's
   * `Latency` parameter — same role, same effect. Pre-fill cycles
   * (0..L-1) drive inputs but skip comparison; post-fill cycles
   * (N..N+L-1) drive zeros to flush the pipeline so every recorded
   * reference output gets matched against a DUT sample. Default 0
   * matches the v1 behaviour (combinational + immediate-update
   * sequential). */
  std::string CocotbOutDir;
  int CocotbVectors = 100;
  int CocotbLatency = 0;
  /* True iff the user passed `-cocotb-latency=N` on the command
   * line (any value, including 0). Lets a `% cocotb: latency(N)`
   * source pragma supply a default per-fixture value while still
   * letting the CLI override when the user knows better. */
  bool CocotbLatencyExplicit = false;
  /* Seed for the harness's `random.seed(...)` call. Default 42 so
   * the harness output is byte-stable across runs (golden-diff
   * friendly). User overrides via `-cocotb-seed=N` to explore
   * different randomization schedules without editing the
   * generated harness. */
  int CocotbSeed = 42;
  /* Tier-7d — `-emit-cocotb FILE.mflow --dut <block>` mode.  The
   * named entry-flow block (must be a `signal_subsystem`) maps to
   * the SV DUT; the rest of the diagram (sources, sinks, non-DUT
   * internals) is rendered as a host-side Python class that the
   * cocotb testbench drives + samples each tick. Empty = stick
   * with the existing per-subsystem cocotb harness. */
  std::string CocotbDut;
  /* Tier-7d — comparison tolerance, decoded units (per output
   * port). Default 1 LSB at Q16.16 = 1/65536. */
  double CocotbTolerance = 1.0 / 65536.0;
  bool CocotbToleranceExplicit = false;
};

int usage(const char *Prog) {
  std::cerr << "usage: " << Prog
            << " [-dump-tokens | -dump-ast | -emit-sema |\n"
               "             -dump-call-sites | -emit-mir |\n"
               "             -emit-mlir | -emit-llvm | -emit-c | -emit-cpp |\n"
               "             -emit-python | -emit-typescript |\n"
               "             -emit-systemverilog | -check-synthesizable |\n"
               "             -emit-hardware-report | -dump-flow |\n"
               "             -dump-chart | -emit-trace |\n"
               "             -emit-matlab |\n"
               "             -format | -repl | -dap]\n"
               "            [-no-line | -line] [-doxygen] [-cpp-auto] [-g]  FILE.m\n";
  return 64;
}

bool parseArgs(int Argc, char **Argv, Options &Opts, const char *&Prog) {
  Prog = Argv[0];
  for (int I = 1; I < Argc; ++I) {
    std::string_view A = Argv[I];
    if (A == "-dump-tokens") Opts.Mode = Options::Mode::DumpTokens;
    else if (A == "-dump-ast") Opts.Mode = Options::Mode::DumpAST;
    else if (A == "-emit-sema") Opts.Mode = Options::Mode::EmitSema;
    /* Phase 1 of the Sema-time monomorphization epic (issue #38): dump the
     * per-callee call-site signature buckets produced by CallSiteAnalyzer.
     * Pure analysis — no IR is emitted. Used by test/Sema goldens to gate
     * the analyzer's bucketing logic before Phase 3 starts consuming it. */
    else if (A == "-dump-call-sites") Opts.Mode = Options::Mode::DumpCallSites;
    /* Phase 2 of #38: round-trip the AST cloner. After the parse, each
     * top-level function is cloned with a `__clone` suffix and the clone
     * is inserted next to the original in the TU. Then Sema + dumpSema
     * run. If the cloner is correct, original and clone show structurally
     * identical Sema state (apart from the renamed function). */
    else if (A == "-test-ast-clone") Opts.Mode = Options::Mode::TestAstClone;
    /* Phases 3+4 of #38: run the monomorphization driver to fixpoint.
     * For each user function called with >1 distinct signatures, the
     * pass deep-clones the body once per non-canonical signature and
     * rewrites the corresponding call sites to dispatch to the clone.
     * Phase 4 closes the loop by re-running Sema after each round so
     * call sites inside cloned bodies can be discovered + retargeted.
     * Mode outputs the post-mono Sema dump + an iteration summary. */
    else if (A == "-test-monomorphize") Opts.Mode = Options::Mode::TestMonomorphize;
    else if (A == "-emit-mir") Opts.Mode = Options::Mode::EmitMIR;
    else if (A == "-emit-mlir") Opts.Mode = Options::Mode::EmitMLIR;
    else if (A == "-emit-llvm") Opts.Mode = Options::Mode::EmitLLVM;
    else if (A == "-emit-c") Opts.Mode = Options::Mode::EmitC;
    else if (A == "-emit-cpp") Opts.Mode = Options::Mode::EmitCpp;
    else if (A == "-emit-python") Opts.Mode = Options::Mode::EmitPython;
    else if (A == "-emit-typescript" || A == "-emit-ts")
      Opts.Mode = Options::Mode::EmitTypeScript;
    /* GPU Coder Tier-6 AOT emit lanes — one MATLAB source produces a
     * self-contained standalone bundle for the chosen device target.
     * The kernel-bodies are CPU-equivalent (T1's rewrite-to-`matlab.for`
     * runs through EmitC) wrapped in a per-target host driver.  When
     * the array-capture outliner ships (next session keystone), each
     * `coder.gpu.kernelfun` body will be extracted into a real device
     * kernel; until then the bundles are the GPU-API surface + CPU body
     * + correct toolchain wiring. */
    else if (A == "-emit-cuda")   Opts.Mode = Options::Mode::EmitCuda;
    else if (A == "-emit-metal")  Opts.Mode = Options::Mode::EmitMetal;
    else if (A == "-emit-opencl") Opts.Mode = Options::Mode::EmitOpenCL;
    else if (A == "-emit-fixed-point-report" || A == "-emit-fi-report")
      Opts.Mode = Options::Mode::EmitFiReport;
    else if (A == "-emit-systemverilog" || A == "-emit-sv")
      Opts.Mode = Options::Mode::EmitSystemVerilog;
    else if (A == "-check-synthesizable")
      Opts.Mode = Options::Mode::CheckSynthesizable;
    else if (A == "-emit-hardware-report" || A == "-emit-hw-report")
      Opts.Mode = Options::Mode::EmitHardwareReport;
    else if (A == "-dump-flow") Opts.Mode = Options::Mode::DumpFlow;
    else if (A == "-dump-chart") Opts.Mode = Options::Mode::DumpChart;
    else if (A == "-emit-trace") Opts.Mode = Options::Mode::EmitTrace;
    else if (A == "-emit-matlab" || A == "-emit-m")
      Opts.Mode = Options::Mode::EmitMatlab;
    else if (A == "-emit-mflow" || A == "-emit-flow")
      Opts.Mode = Options::Mode::EmitMflow;
    else if (A == "-emit-mflowlink-cpp" || A == "-emit-signal-flow-cpp")
      Opts.Mode = Options::Mode::EmitMflowLinkCpp;
    else if (A == "-emit-cocotb")
      Opts.Mode = Options::Mode::EmitCocotb;
    else if (A.size() > 16 && A.substr(0, 16) == "-cocotb-vectors=") {
      Opts.CocotbVectors = std::atoi(std::string(A.substr(16)).c_str());
      if (Opts.CocotbVectors <= 0) {
        std::cerr << "-cocotb-vectors must be a positive integer\n";
        return false;
      }
    }
    else if (A.size() > 12 && A.substr(0, 12) == "-cocotb-out=")
      Opts.CocotbOutDir = std::string(A.substr(12));
    else if (A.size() > 16 && A.substr(0, 16) == "-cocotb-latency=") {
      Opts.CocotbLatency = std::atoi(std::string(A.substr(16)).c_str());
      if (Opts.CocotbLatency < 0) {
        std::cerr << "-cocotb-latency must be a non-negative integer\n";
        return false;
      }
      Opts.CocotbLatencyExplicit = true;
    }
    else if (A.size() > 13 && A.substr(0, 13) == "-cocotb-seed=") {
      Opts.CocotbSeed = std::atoi(std::string(A.substr(13)).c_str());
    }
    else if (A == "--dut" || A == "-dut") {
      if (++I >= Argc) {
        std::cerr << "--dut requires an argument\n";
        return false;
      }
      Opts.CocotbDut = Argv[I];
    }
    else if (A.size() > 6 && A.substr(0, 6) == "--dut=")
      Opts.CocotbDut = std::string(A.substr(6));
    else if (A.size() > 18 && A.substr(0, 18) == "-cocotb-tolerance=") {
      try {
        Opts.CocotbTolerance = std::stod(std::string(A.substr(18)));
        Opts.CocotbToleranceExplicit = true;
      } catch (...) {
        std::cerr << "-cocotb-tolerance must be a non-negative number\n";
        return false;
      }
    }
    else if (A == "-repl") Opts.Mode = Options::Mode::Repl;
    else if (A == "-format") Opts.Mode = Options::Mode::Format;
    else if (A == "-dap") Opts.Mode = Options::Mode::Dap;
    else if (A == "-simulate") Opts.Mode = Options::Mode::Simulate;
    else if (A == "--dry-run" || A == "-dry-run") Opts.DryRun = true;
    else if (A == "--sim-dap" || A == "-sim-dap") Opts.SimulateDap = true;
    else if (A.size() > 12 && A.substr(0, 12) == "--subsystem=")
      Opts.Subsystem = std::string(A.substr(12));
    else if (A == "--subsystem" || A == "-subsystem") {
      if (++I >= Argc) {
        std::cerr << "--subsystem requires an argument\n";
        return false;
      }
      Opts.Subsystem = Argv[I];
    }
    else if (A.size() > 13 && A.substr(0, 13) == "--state-form=")
      Opts.StateForm = std::string(A.substr(13));
    else if (A.size() > 14 && A.substr(0, 14) == "--target-rate=") {
      try { Opts.TargetRate = std::stod(std::string(A.substr(14))); }
      catch (...) {
        std::cerr << "--target-rate must be a positive number\n";
        return false;
      }
    }
    else if (A == "--target-rate" || A == "-target-rate") {
      if (++I >= Argc) {
        std::cerr << "--target-rate requires an argument\n";
        return false;
      }
      try { Opts.TargetRate = std::stod(Argv[I]); }
      catch (...) {
        std::cerr << "--target-rate must be a positive number\n";
        return false;
      }
    }
    else if (A.size() > 8 && A.substr(0, 8) == "--ticks=")
      Opts.Ticks = std::atoi(std::string(A.substr(8)).c_str());
    else if (A == "--ticks" || A == "-ticks") {
      if (++I >= Argc) {
        std::cerr << "--ticks requires a positive integer\n";
        return false;
      }
      Opts.Ticks = std::atoi(Argv[I]);
    }
    else if (A.size() > 13 && A.substr(0, 13) == "--decimation=")
      Opts.Decimation = std::atoi(std::string(A.substr(13)).c_str());
    else if (A == "--decimation" || A == "-decimation") {
      if (++I >= Argc) {
        std::cerr << "--decimation requires a positive integer\n";
        return false;
      }
      Opts.Decimation = std::atoi(Argv[I]);
    }
    else if (A.size() > 10 && A.substr(0, 10) == "--fi-spec=")
      Opts.FiSpecs.push_back(std::string(A.substr(10)));
    else if (A == "--fi-spec" || A == "-fi-spec") {
      if (++I >= Argc) {
        std::cerr << "--fi-spec requires an argument (port=Q<W>.<F>)\n";
        return false;
      }
      Opts.FiSpecs.push_back(Argv[I]);
    }
    else if (A.size() > 13 && A.substr(0, 13) == "--discretize=") {
      Opts.Discretize = std::string(A.substr(13));
      if (Opts.Discretize != "forward_euler" &&
          Opts.Discretize != "tustin") {
        std::cerr << "--discretize must be `forward_euler` or `tustin`\n";
        return false;
      }
    }
    else if (A == "--discretize" || A == "-discretize") {
      if (++I >= Argc) {
        std::cerr << "--discretize requires an argument "
                  << "(forward_euler|tustin)\n";
        return false;
      }
      Opts.Discretize = Argv[I];
      if (Opts.Discretize != "forward_euler" &&
          Opts.Discretize != "tustin") {
        std::cerr << "--discretize must be `forward_euler` or `tustin`\n";
        return false;
      }
    }
    else if (A == "-opt" || A == "-O") Opts.Opt = true;
    else if (A == "-no-line" || A == "--no-line") Opts.NoLine = true;
    else if (A == "-line" || A == "--line") Opts.EmitLine = true;
    else if (A == "-doxygen" || A == "--doxygen") Opts.Doxygen = true;
    else if (A == "-cpp-auto" || A == "--cpp-auto") Opts.CppAuto = true;
    else if (A == "-g" || A == "--debug-hooks") Opts.Debug = true;
    else if (A == "-sv-reset=async-low")
      Opts.SvReset = Options::SvResetKind::AsyncLow;
    else if (A == "-sv-reset=sync-high")
      Opts.SvReset = Options::SvResetKind::SyncHigh;
    else if (A == "-sv-reset=sync-low")
      Opts.SvReset = Options::SvResetKind::SyncLow;
    else if (A == "-sv-fsm-encoding=binary")
      Opts.SvFSMEnc = Options::SvFSMEncoding::Binary;
    else if (A == "-sv-fsm-encoding=one-hot" ||
             A == "-sv-fsm-encoding=one_hot")
      Opts.SvFSMEnc = Options::SvFSMEncoding::OneHot;
    else if (A == "-sv-fsm-encoding=gray")
      Opts.SvFSMEnc = Options::SvFSMEncoding::Gray;
    else if (A == "-sv-const-mul=off")
      Opts.SvConstMulOpt = false;
    else if (A == "-sv-const-mul=auto" || A == "-sv-const-mul=csd" ||
             A == "-sv-const-mul=on")
      Opts.SvConstMulOpt = true;
    else if (A == "-h" || A == "--help") return false;
    else if (A == "--block-path" || A == "-block-path") {
      if (I + 1 >= Argc) {
        std::cerr << "--block-path requires a directory argument\n";
        return false;
      }
      Opts.BlockPath.push_back(Argv[++I]);
    }
    else if (A.size() > 13 && A.substr(0, 13) == "--block-path=") {
      Opts.BlockPath.push_back(std::string(A.substr(13)));
    }
    else if (A == "--preserve-layout" || A == "-preserve-layout") {
      if (I + 1 >= Argc) {
        std::cerr << "--preserve-layout requires a file argument\n";
        return false;
      }
      Opts.PreserveLayoutPath = Argv[++I];
    }
    else if (A.size() > 18 && A.substr(0, 18) == "--preserve-layout=") {
      Opts.PreserveLayoutPath = std::string(A.substr(18));
    }
    else if (!A.empty() && A[0] == '-') {
      std::cerr << "unknown flag: " << A << "\n";
      return false;
    } else {
      if (Opts.InputPath.empty())
        Opts.InputPath = std::string(A);
      else
        Opts.ExtraInputs.push_back(std::string(A));
    }
  }
  /* -repl doesn't take a file. Everything else does.
   * -dap may receive the program path via DAP `launch`, so a CLI
   * path is optional there too. */
  if (Opts.Mode == Options::Mode::Repl) return true;
  if (Opts.Mode == Options::Mode::Dap) return true;
  return !Opts.InputPath.empty();
}

void dumpTokens(const SourceManager &SM, const std::vector<Token> &Ts) {
  for (const auto &T : Ts) {
    auto LC = SM.getLineColumn(T.Loc);
    std::cout << LC.Line << ':' << LC.Column << "\t"
              << tokenKindName(T.Kind);
    if (T.Kind != TokenKind::newline && T.Kind != TokenKind::eof)
      std::cout << "\t'" << T.Text << "'";
    std::cout << '\n';
  }
}

#if MATLAB_LLVM_WITH_MLIR
/* --- REPL -----------------------------------------------------------------
 *
 * Accumulate input, parse + Sema + lower with ReplMode=true, run the same
 * pass pipeline the -emit-llvm path uses, JIT with mlir::ExecutionEngine,
 * invoke the generated `script` function. Variables live in a module-
 * global matlab_struct inside the runtime so they persist across
 * invocations. The JIT resolves matlab_* and matlab_ws_* symbols against
 * the running matlabc process — the runtime is linked into the
 * executable at build time for this purpose. */

int blockDepth(const std::vector<Token> &Toks) {
  int d = 0;
  /* #260: `properties` / `methods` / `events` / `enumeration` are
   * context-sensitive — they open a block (with a matching `end`) ONLY inside
   * a classdef.  Outside one they are ordinary functions (`p =
   * properties(obj)`), so counting them unconditionally would make the REPL
   * accumulator wait forever for a non-existent `end`.  Track whether a
   * classdef is open and only treat them as block-openers then; without this
   * the accumulator submits a partial `classdef ... properties ... end` after
   * the *properties* `end` (the classdef's own `end` still pending) and the
   * parser errors (`'end' is only valid inside indexing` / `unexpected
   * 'properties'`). */
  bool inClassdef = false;
  /* An `end` inside `()` / `[]` / `{}` is an indexing `end` (`a(2:end)`,
   * `x{end}`), NOT a block terminator.  Counting it as a block close drops the
   * depth early and makes the accumulator submit a still-open if/for/while
   * block — and a prepended classdef prelude then lands inside that open block
   * ("unexpected 'classdef' in expression", #260 Symptom B).  Track bracket
   * nesting and only treat a depth-0 `end` as a block close. */
  int bracket = 0;
  for (const auto &T : Toks) {
    switch (T.Kind) {
    case TokenKind::l_paren:
    case TokenKind::l_square:
    case TokenKind::l_brace:
      ++bracket; break;
    case TokenKind::r_paren:
    case TokenKind::r_square:
    case TokenKind::r_brace:
      if (bracket > 0) --bracket; break;
    case TokenKind::kw_classdef:
      inClassdef = true; ++d; break;
    case TokenKind::kw_properties:
    case TokenKind::kw_methods:
    case TokenKind::kw_events:
    case TokenKind::kw_enumeration:
      if (inClassdef) ++d; break;
    case TokenKind::kw_if:
    case TokenKind::kw_for:
    case TokenKind::kw_while:
    case TokenKind::kw_switch:
    case TokenKind::kw_try:
    case TokenKind::kw_function:
    case TokenKind::kw_parfor:
      ++d; break;
    case TokenKind::kw_end:
      if (bracket == 0) --d; break;
    default: break;
    }
  }
  return d < 0 ? 0 : d;
}

/* Bracket / paren / brace depth across the token stream.  Used by
 * the REPL to keep buffering input while an expression is mid-flight
 * (e.g. a `(...)` arg list or a `[...]` literal spanning multiple
 * lines).  Returns 0 when all brackets are balanced. */
int bracketDepth(const std::vector<Token> &Toks) {
  int p = 0, sq = 0, br = 0;
  for (const auto &T : Toks) {
    switch (T.Kind) {
    case TokenKind::l_paren:  ++p;  break;
    case TokenKind::r_paren:  --p;  break;
    case TokenKind::l_square: ++sq; break;
    case TokenKind::r_square: --sq; break;
    case TokenKind::l_brace:  ++br; break;
    case TokenKind::r_brace:  --br; break;
    default: break;
    }
  }
  int total = (p < 0 ? 0 : p) + (sq < 0 ? 0 : sq) + (br < 0 ? 0 : br);
  return total;
}

/* True if the raw input text ends with a MATLAB line continuation
 * (`...` after stripping a trailing `%` line-comment and whitespace).
 * The lexer consumes `...` silently — it does not emit an ellipsis
 * token in the canonical case — so the REPL has to inspect the raw
 * source to know that the user is mid-statement. */
bool hasTrailingEllipsis(const std::string &Src) {
  /* Walk back from the end, skipping CR / LF / spaces / tabs and any
   * trailing line-comment (`% ...`).  If we land on a `...` triple,
   * the statement is unfinished. */
  int n = (int)Src.size();
  int i = n - 1;
  /* Strip trailing newline / whitespace. */
  while (i >= 0 && (Src[i] == '\n' || Src[i] == '\r' ||
                     Src[i] == ' ' || Src[i] == '\t'))
    --i;
  /* Strip a trailing line-comment if present.  Scan backwards from
   * the current end position to the nearest preceding newline; if
   * that segment starts with `%`, treat it as a comment and clip. */
  if (i >= 0) {
    int line_start = i;
    while (line_start > 0 && Src[line_start - 1] != '\n') --line_start;
    /* Look for the first `%` on this line that is NOT inside a quoted
     * string.  We don't have a full lex state here; the simple
     * heuristic of "first %" matches the textbook case and is the
     * pattern shown in the REPL transcript. */
    for (int j = line_start; j <= i; ++j) {
      if (Src[j] == '%') { i = j - 1; break; }
    }
    /* Re-strip trailing whitespace before the comment we just dropped. */
    while (i >= 0 && (Src[i] == ' ' || Src[i] == '\t'))
      --i;
  }
  return i >= 2 && Src[i] == '.' && Src[i - 1] == '.' && Src[i - 2] == '.';
}

/* Format every diagnostic in `Diag` as a single multi-line string,
 * one diag per line in `<file>:<line>:<col>: <level>: <message>`
 * shape. Used by the DAP evaluate handler to carry compile errors
 * into the response so the IDE's watch row can show the actual
 * cause instead of "see debug console". The shape mirrors what
 * Diag.printAll() emits to stderr — keeps the user's mental model
 * consistent across the two surfaces. */
std::string formatDiagnostics(const SourceManager &SM,
                               const DiagnosticEngine &Diag) {
  std::string Out;
  for (const Diagnostic &D : Diag.diagnostics()) {
    auto LC = SM.getLineColumn(D.Loc);
    if (D.Loc.isValid())
      Out += SM.getName(D.Loc.File);
    else
      Out += "<input>";
    if (LC.Line) {
      Out += ":";
      Out += std::to_string(LC.Line);
      Out += ":";
      Out += std::to_string(LC.Column);
    }
    Out += ": ";
    switch (D.Level) {
    case DiagLevel::Error:   Out += "error: ";   break;
    case DiagLevel::Warning: Out += "warning: "; break;
    case DiagLevel::Note:    Out += "note: ";    break;
    }
    Out += D.Message;
    Out += "\n";
  }
  return Out;
}

/* Run a REPL input through the full Lex → Parse → Sema → MLIR →
 * JIT pipeline. Returns 0 on success, 1 on any diagnostic-level
 * error. When `DiagOut` is non-null, captured diagnostics are
 * formatted into that string in addition to being printed to
 * stderr — used by the DAP evaluate handler to surface compile
 * errors in the watch box without forcing the user to scan the
 * debug console. */
/* Resolver workspace-kind hook — defined further down (after the
 * runtime-introspection externs), forward-declared here so the REPL
 * compile entries below can install it on their Resolver. */
extern "C" int replWorkspaceKindHook(const char *name, int64_t len);
/* Companion hook for kind=2 bindings — returns the class name of the
 * runtime obj stored under `name` (or null when the binding isn't a
 * class instance / the class isn't registered).  Defined alongside
 * replWorkspaceKindHook below. */
extern "C" const char *replWorkspaceClassNameHook(const char *name,
                                                   int64_t len,
                                                   int64_t *len_out);
/* Companion hook for kind=13 bindings — returns the handle's stored
 * return-kind (#119).  Defined alongside replWorkspaceKindHook below. */
extern "C" int replWorkspaceHandleSigHook(const char *name, int64_t len);

/* matlabc binary directory — captured once in main() so the REPL
 * prelude-search helpers can find runtime/*.m files relative to the
 * binary location without threading argv[0] through every call site.
 * Empty when matlabc was invoked without a discoverable on-disk path
 * (rare; falls back to no preludes in that case). */
static std::string g_MatlabcBinDir;

/* User-defined function persistence across REPL turns. The REPL
 * accumulator always submits a `function ... end` block on its own
 * (block depth flips back to 0 right after `end`), so the function
 * definition lands in TU N while its call site lands in TU N+1.
 * matlabc's resolver / type inference / monomorphisation can't link
 * the two — the call site degrades to a workspace load + array
 * index, the function body keeps `none`-typed args, and the JIT
 * fails to translate either to LLVM IR.
 *
 * We close the gap by stashing each successfully-parsed top-level
 * function's source verbatim. On every subsequent REPL turn, the
 * prelude builder scans the user input for mentions of stashed
 * names and prepends matching function sources back in. The turn
 * compiles function + call site in the same TU — Sema refines arg
 * types from call-site shapes (same as static `-emit-c`), the call
 * lowers as a real `matlab.call`, and the JIT succeeds.
 *
 * Map: function name (lowercased? — MATLAB is case-sensitive in
 * function names per §15-2, so we keep exact case) → source text.
 * A redefinition in a later turn overwrites the entry. */
static std::map<std::string, std::string> g_ReplUserFunctions;

/* Build the SO / CST classdef prelude content to prepend to a REPL
 * input.  Same trigger logic as the static-input path (see
 * `userMentionsCommClasses` / `userMentionsCstClasses` below): scan
 * the source for whole-word class names with `(` or `=` follow-up,
 * stripping line-comments; for each hit, locate the prelude file
 * under `<bin>/../runtime/` and concatenate its contents.
 *
 * Returns the combined prelude text (possibly empty).  REPL caller
 * prepends this to the user input before lexing — adds the
 * `classdef` blocks to the same TU so Sema sees the class names. */
static std::string buildReplPrelude(const std::string &Src);

#ifdef MATLAB_LLVM_WITH_PLOT
/* IDE integration (Matlab_llvm_ide): defined in runtime/plot/c_api.cpp.
 * Streams every open figure as a sentinel-bracketed base64 PNG to stdout
 * when MATLAB_LLVM_IDE_FIGURES=1 is set; no-op otherwise. Forward decl
 * here so we can call it from runRepl after each input without dragging
 * the full matlab_plot.h surface into this TU. */
extern "C" void matlab_ide_emit_all_figures(void);
#endif

/* #77: shared in-process (JIT) software-lowering pipeline.
 *
 * The REPL (`runReplInput`) and the DAP launch (`compileProgram`) both
 * lower a module for the ExecutionEngine, and each used to carry its own
 * hand-written copy of the pass list.  The copies drifted from the static
 * `-emit-*` pipeline — most importantly they never ran `runRefineSlotTypes`,
 * so `none`-typed slots were never refined to their concrete stored type
 * and any builtin whose lowering keys off a concrete operand type (e.g.
 * `disp(f(0))` where `f` is a handle) failed to match, surviving as an
 * un-lowered `matlab.*` op that `validateAllMatlabOpsLowered` then rejected
 * ("failed to compile program").  They also lacked the trailing
 * `LowerTensorOps`+`RefineFuncSigs` convergence loop and `LowerStaticFiArrays`.
 *
 * This is the single source of truth for both JIT callers, modelled on the
 * software portion of the static pipeline's tail (the `WantFullPipeline`
 * branch in `main`).  It always lowers for software execution (f64 lane,
 * never the HW integer-width lane) and uses ReplMode anon lowering, which
 * is what both JIT callers need.  Callers keep their own pre-amble
 * (`lowerToMLIR` + verify) and post-steps (validate / classdef-method drop /
 * stripMatlabFuncAttrs / ExecutionEngine create). */
/* Iteratively erase classdef method `func.func`s that have no remaining
 * symbol uses. Shared by the JIT/REPL/-dap lowering (runJitSoftwareLowering)
 * and mirrors the identical strip the AOT path runs in lowerToLLVMIR.
 * Only `matlab.class_name`-tagged funcs are considered, so ordinary
 * library/helper functions are never removed; the fixpoint loop drops
 * methods reachable only from other (already-dead) methods. */
static void dropUncalledClassMethods(mlir::ModuleOp M) {
  auto SymTbl = mlir::SymbolTable(M);
  bool Changed = true;
  while (Changed) {
    Changed = false;
    llvm::SmallVector<mlir::Operation *> Drop;
    M.walk([&](mlir::func::FuncOp F) {
      if (!F->hasAttr("matlab.class_name")) return;
      auto Sym = F.getSymNameAttr();
      if (auto Uses = SymTbl.getSymbolUses(Sym, M))
        if (Uses->empty()) Drop.push_back(F);
    });
    for (auto *Op : Drop) {
      SymTbl.erase(Op);
      Changed = true;
    }
  }
}

static void runJitSoftwareLowering(mlir::ModuleOp M) {
  using namespace mlirgen;
  runSlotPromotion(M);
  // fi ops must lower before the generic scalar-to-arith pass (else the
  // matlab.add/matmul carrying fi attrs fold to plain arith and lose the
  // spec metadata).  See docs/emit_fixed_point.md.
  runLowerFixedPoint(M);
  runLowerScalarsToArith(M);
  runSlotPromotion(M);
  // Patch func.func sigs from refined return-op types so the verifier /
  // func-to-llvm conversion doesn't trip on a body that now produces e.g.
  // i1 while the signature still declares `-> none`.
  runRefineFuncSigs(M);
  runPromoteNoneParams(M);
  for (int Iter = 0; Iter < 4; ++Iter)
    if (!runPromoteBinopTypes(M)) break;
  // #77: seed slot/load types from the now-typed entry-block args before
  // the outliner — without this RefineSlotTypes (which the JIT copies
  // omitted) slots stay `none` and the body never gets concretely typed.
  runRefineSlotTypes(M);
  // Forward outer-scope literal captures into parfor bodies before the
  // outliner (issue #20 common case), then outline parfor / GPU kernels.
  runForwardParforCaptures(M);
  runOutlineParfor(M);
  runOutlineGpuKernels(M);
  runLowerSeqLoops(M);
  // ReplMode anon lowering — the JIT callers differ from the static path
  // here (REPL-mode codegen).  Outlines anon bodies so handles become
  // plain function pointers and call_indirect sites collapse.
  runLowerAnonCalls(M, /*ReplMode=*/true);
  for (int Iter = 0; Iter < 8; ++Iter) {
    bool A = runLowerScalarsToArith(M);
    bool B = runLowerUserCalls(M);
    if (!A && !B) break;
  }
  // #77: a param-bound for-loop (`for k = 1:n`) couldn't lower in the first
  // runLowerSeqLoops (the param was still `none`); the fixpoint just refined
  // it — refine the slot and re-run seq-loop lowering before LowerTensorOps
  // consumes the matlab.range producer.
  runRefineSlotTypes(M);
  runLowerSeqLoops(M);
  runLowerTensorOps(M);
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool A = runLowerScalarsToArith(M);
    bool B = runLowerUserCalls(M);
    if (!A && !B) break;
  }
  // #77: propagate freshly-typed call results through binops + slot chains
  // (a chained `gather(a .* x + b)`), iterating to fixpoint.
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool Pb = runPromoteBinopTypes(M);
    runRefineSlotTypes(M);
    if (!Pb) break;
  }
  runLowerTensorOps(M);
  // LATE GPU outline (issue #24) — see the AOT path for rationale.
  if (runOutlineGpuKernelsLate(M)) {
    runRefineSlotTypes(M);
    runLowerSeqLoops(M);
    runLowerTensorOps(M);
    for (int Iter = 0; Iter < 4; ++Iter) {
      bool A = runLowerScalarsToArith(M);
      bool B = runLowerUserCalls(M);
      if (!A && !B) break;
    }
    runLowerTensorOps(M);
  }
  // Second LowerFixedPoint sweep — picks up matlab_mat_*_slice1 / _concat_row
  // sites that needed their tensor operand retyped to ptr first.
  runLowerFixedPoint(M);
  // Second-chance anon-call rewrite: a matlab.call_indirect that survived the
  // first LowerAnonCalls because its matrix operands were still tensor-typed
  // can now match the outlined function's (ptr, ...) signature.  This is what
  // lowers a vector-objective anon (`@(x) x(1)^2 + x(2)^2`) passed to a
  // solver; without it the `matlab.subscript` reads of `x(i)` survive.
  if (runLowerAnonCallsPost(M)) {
    runLowerTensorOps(M);
    for (int Iter = 0; Iter < 4; ++Iter) {
      bool A = runLowerScalarsToArith(M);
      bool B = runLowerUserCalls(M);
      if (!A && !B) break;
    }
    runLowerTensorOps(M);
  }
  // Multi-callsite monomorphisation (matrix-typed / arity-varying /
  // varargin callees).  compileProgram lacked this entirely.
  // #191 P5 scaffolding: MATLAB_LLVM_NO_LATE_MONO=1 bypasses the late
  // pass so the true Sema-only failure set can be measured (and, once
  // P3+P5 land, locked at zero).
  if (!std::getenv("MATLAB_LLVM_NO_LATE_MONO") && runMonomorphiseUserCalls(M)) {
    for (int Iter = 0; Iter < 4; ++Iter) {
      bool A = runLowerScalarsToArith(M);
      bool B = runLowerUserCalls(M);
      if (!A && !B) break;
    }
    runLowerTensorOps(M);
    // Refresh each func.func signature from the types now flowing through
    // its func.return (LowerTensorOps rewrote clone bodies, not signatures).
    M.walk([&](mlir::func::FuncOp Fn) {
      if (Fn.empty()) return;
      llvm::SmallVector<mlir::Type, 4> NewResults(
          Fn.getFunctionType().getResults().begin(),
          Fn.getFunctionType().getResults().end());
      bool Changed = false;
      Fn.walk([&](mlir::func::ReturnOp Ret) {
        if (Ret.getNumOperands() != NewResults.size()) return;
        for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
          auto Old = NewResults[i];
          auto New = Ret.getOperand(i).getType();
          if (mlir::isa<mlir::NoneType>(Old) && Old != New) {
            NewResults[i] = New;
            Changed = true;
          }
        }
      });
      if (Changed)
        Fn.setFunctionType(mlir::FunctionType::get(
            Fn.getContext(), Fn.getFunctionType().getInputs(), NewResults));
    });
    // Stale func.call result types need patching to match.
    M.walk([&](mlir::func::CallOp Call) {
      auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
      if (!Tgt) return;
      auto SigR = Tgt.getFunctionType().getResults();
      if (Call.getNumResults() != SigR.size()) return;
      bool Mismatch = false;
      for (unsigned i = 0; i < SigR.size(); ++i)
        if (Call.getResult(i).getType() != SigR[i]) { Mismatch = true; break; }
      if (!Mismatch) return;
      mlir::OpBuilder CB(Call);
      auto Nc = mlir::func::CallOp::create(CB, Call.getLoc(), SigR,
                                            Call.getCallee(),
                                            Call.getOperands());
      for (unsigned i = 0; i < SigR.size(); ++i)
        Call.getResult(i).replaceAllUsesWith(Nc.getResult(i));
      Call.erase();
    });
    runLowerTensorOps(M);
  }
  runLowerFixedPoint(M);
  runLowerNarginNargout(M);
  // #77: refine `none`-typed slots whose stores agree on a concrete type
  // BEFORE LowerStaticFiArrays / LowerScalarSlots so the retyped slots get
  // promoted; then rewrite `fi(zeros(1,N),...)` chains to stack allocas.
  runRefineSlotTypes(M);
  runLowerStaticFiArrays(M);
  runRefineFuncSigs(M);
  // #77: matrix-returning user functions only settle their tensor->ptr
  // result type in RefineFuncSigs; re-run LowerTensorOps so the caller's
  // slot (fed by that call) retypes to ptr and its A(i,j)/A+1 uses lower.
  // Iterate with RefineFuncSigs so chained matrix-returning calls converge.
  // This loop also re-lowers the REPL's ptr-sticky `matlab_ws_set_obj`
  // store once its stored value is concretely ptr-typed.
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool Changed = runLowerTensorOps(M);
    runRefineFuncSigs(M);
    if (!Changed) break;
  }
  // #148: an `if <string-predicate>(a,b)` (contains/startsWith/strcmp/...)
  // condition is `none`-typed at MIR-to-MLIR lowering, so fixupIfCond left a
  // verifier-placeholder unrealized_conversion_cast on the scf.if. The
  // builtin's result only refines to f64 in the LowerTensorOps loop above, so
  // resolve those placeholders now (cast -> arith.cmpf one, 0.0). The SV-emit
  // pipeline runs this separately; this covers the AOT / JIT / REPL / -dap
  // path (this lowering function is shared by all of them).
  runRefineIfConds(M);
  // Promote any surviving scalar-primitive matlab.alloc to llvm.alloca.
  runLowerScalarSlots(M);
#ifdef MATLAB_LLVM_WITH_PLOT
  runLowerPlot(M);
#endif
  runLowerIO(M);
  /* #77: drop uncalled classdef method bodies before the leftover-op
   * validation + LLVM conversion. The merged toolbox classdef prelude
   * pulls *every* method of a referenced class into the module, but
   * Sema only refines a method's param types when a call site drives
   * them — an uncalled method (e.g. dlarray's `conv2d` when the program
   * only uses `relu`/`softmax`) keeps `none`-typed args, so its internal
   * runtime calls never match a lowering shape and would trip the
   * conversion as "unsupported call shape". The AOT path strips these in
   * lowerToLLVMIR; the JIT/REPL/-dap path needs the same. Iterates so a
   * method only reachable from another dead method also drops. Internal
   * sibling calls keep transitively-reachable methods live; non-classdef
   * library functions don't carry `matlab.class_name`, so they're never
   * touched. */
  dropUncalledClassMethods(M);
  /* #77: DebugMode mirrors every named store to the per-frame LOCALS
   * table via `matlab_dbg_frame_set(name, value)` (emitStore). With the
   * toolbox classdef prelude now merged for `-dap`, those mirrors are
   * also emitted inside library classdef method bodies, where many
   * internal temporaries stay `none`-typed (matlab_obj-flavoured values
   * that never refine to f64/ptr/int). Such a mirror has no runtime
   * variant and nothing to display, but it survives LowerTensorOps'
   * type-dispatch (which punts none-typed operands) all the way to the
   * conversion pipeline, where it dies as "unsupported call shape".
   * Drop these residual un-typeable mirrors — user-visible locals
   * resolve to concrete types and lower normally above. The mirror's
   * result is an unused void token, so erasing is safe. */
  {
    llvm::SmallVector<mlir::Operation *, 16> Dead;
    M.walk([&](mlir::Operation *Op) {
      if (Op->getName().getStringRef() != "matlab.call_builtin") return;
      auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
      if (!C || C.getValue() != "matlab_dbg_frame_set") return;
      if (Op->getNumOperands() < 2) return;
      if (!mlir::isa<mlir::NoneType>(Op->getOperand(1).getType())) return;
      bool ResultsUnused = true;
      for (mlir::Value R : Op->getResults())
        if (!R.use_empty()) { ResultsUnused = false; break; }
      if (ResultsUnused) Dead.push_back(Op);
    });
    for (mlir::Operation *Op : Dead) Op->erase();
  }
  if (getenv("MATLABC_JIT_DUMP")) mlirgen::printModule(std::cerr, M);
}


// #191 P3 — run the dispatch de-synthesis AST rewrite (operator/method dispatch
// on instances of the allow-listed classes -> explicit method-call nodes) after
// the first Resolver+TypeInference, then re-type the synthesized nodes. The
// allow-list grows one class per PR; the lowering synthesis remains the
// identical fallback for operands whose object type isn't on Expr->Ty (e.g.
// cross-turn -repl), so this is behavior-preserving. See
// docs/sema_p3_dispatch_desynth.md.
// KeyOffPinnedClass enables the PinnedClass operand-class fallback in the
// desynth pass — wanted in whole-program (AOT) compilation where P2/P5 run,
// but unsafe in cross-turn -repl (the rewritten method call's base is a
// cross-turn binding the dispatch lowering segfaults on; synthesis handles it).
static void p3DesynthDispatch(matlab::ASTContext &Ctx,
                              matlab::TranslationUnit &TU,
                              matlab::TypeInference &Inf,
                              bool KeyOffPinnedClass) {
  static const std::set<std::string> kP3Classes = {"Vec2", "tf",  "ss",
                                                    "zpk",  "pid", "frd"};
  // Iterate to a fixpoint. The first pass rewrites the leaf operators (some
  // recovered only from a binding's PinnedClass, so their result type isn't
  // known yet); re-running TypeInference stamps each rewritten method call
  // with its object<Class> result type, which lets the NEXT pass rewrite a
  // surrounding operator whose operand is that call. Without the loop a
  // chained `a*b + c` on PinnedClass-only operands leaves the outer `+` to
  // the lowering synthesis fallback. Bounded so a non-converging case can't
  // spin; in practice it settles in 2–3 passes.
  for (int Iter = 0; Iter < 8; ++Iter) {
    if (matlab::sema::desynthDispatch(Ctx, TU, kP3Classes, KeyOffPinnedClass) ==
        0)
      break;
    Inf.run(TU);
  }
}

int runReplInput(mlirgen::Context &MCtx, const std::string &Src, int Id,
                 std::string *DiagOut = nullptr) {
  /* If the input mentions any CST / Comm classdef name, prepend the
   * matching prelude file(s) before parsing.  The classdef blocks
   * land in the same TU as the user's script, so Sema can resolve
   * the class name and the lowering can emit ctor / method bodies.
   * Falls back to the user input verbatim when nothing matches —
   * non-classdef REPL turns pay no overhead. */
  /* Prelude goes AFTER the user input — MATLAB script files mix
   * top-level statements with classdef blocks in that order, and the
   * static-input path uses the same shape (see the loader around
   * `Combined += "\n"; ... PreludePaths`). */
  std::string Prelude = buildReplPrelude(Src);
  std::string Combined = Prelude.empty() ? Src : Src + "\n" + Prelude;
  SourceManager SM;
  FileID F = SM.addBuffer("<repl:" + std::to_string(Id) + ">", Combined);
  DiagnosticEngine Diag(SM);
  auto onFail = [&] {
    if (DiagOut) *DiagOut = formatDiagnostics(SM, Diag);
    Diag.printAll();
  };
  Lexer Lx(SM, F, Diag);
  auto Toks = Lx.tokenize();

  ASTContext AstCtx;
  Parser P(std::move(Toks), AstCtx, Diag);
  TranslationUnit *TU = P.parseFile();
  if (!TU || Diag.hasErrors()) {
    onFail();
    return 1;
  }

  /* Stash every top-level function defined in this turn so a later
   * turn that mentions its name pulls it back via buildReplPrelude.
   * `Combined = Src + "\n" + Prelude` lays Src first, so a function
   * defined in the user's input has Range offsets within Src; a
   * function pulled from the prelude has offsets past `Src.size() +
   * 1`. We skip the latter — don't re-store prelude content. */
  if (TU) {
    std::string_view Buf = SM.getBuffer(F);
    size_t SrcEnd = Src.size();
    for (Function *Fn : TU->Functions) {
      if (!Fn || Fn->Name.empty()) continue;
      if (Fn->Range.Begin.File != F || Fn->Range.End.File != F) continue;
      uint32_t Beg = Fn->Range.Begin.Offset;
      uint32_t End = Fn->Range.End.Offset;
      if (End <= Beg || End > Buf.size()) continue;
      if (Beg >= SrcEnd) continue;  // prelude-sourced — skip
      g_ReplUserFunctions[std::string(Fn->Name)] =
          std::string(Buf.substr(Beg, End - Beg));
    }
  }

  SemaContext Sema;
  TypeContext TC;
  Resolver R(Sema, TC, Diag);
  R.setReplMode(true);
  R.setWorkspaceKindHook(&replWorkspaceKindHook);
  R.setWorkspaceClassNameHook(&replWorkspaceClassNameHook);
  R.setWorkspaceHandleSigHook(&replWorkspaceHandleSigHook);
  R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  Inf.run(*TU);
  // Interactive -repl / JIT: keep the PinnedClass fallback OFF — its rewrite
  // crashes the cross-turn dispatch lowering (see p3DesynthDispatch).
  p3DesynthDispatch(AstCtx, *TU, Inf, /*KeyOffPinnedClass=*/false);
  if (Diag.hasErrors()) {
    onFail();
    return 1;
  }

  auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU, &SM, /*ReplMode=*/true);
  if (Diag.hasErrors() || mlir::failed(mlir::verify(M))) {
    onFail();
    std::cerr << "error: REPL MLIR verification failed\n";
    return 1;
  }

  /* #77: shared in-process software lowering (was an inline copy that
   * had drifted from the static pipeline — missing runRefineSlotTypes
   * etc.).  Now the single source of truth for REPL + DAP launch. */
  runJitSoftwareLowering(M);
  /* (Uncalled classdef methods are dropped inside runJitSoftwareLowering
   * via dropUncalledClassMethods — see #77.) */

  if (mlir::failed(mlir::verify(M))) {
    std::cerr << "error: REPL MLIR verification failed after passes\n";
    return 1;
  }

  /* Function-definition-only input — turns where the user typed a
   * `function ... end` block but no script statements. The module
   * has func.func defs (likely `none`-typed) but no script `main`
   * to execute. We've already stashed the function source in
   * g_ReplUserFunctions for later turns, so there's nothing to
   * actually do here. Skip the JIT entirely; trying to translate
   * uncalled `none`-typed funcs would fail noisily for no benefit. */
  {
    bool HasScript = TU && TU->ScriptNode != nullptr;
    if (!HasScript) return 0;
  }

  /* Same conversion-to-LLVM-dialect pipeline that lowerToLLVMIR runs.
   * We do it here rather than calling lowerToLLVMIR so ExecutionEngine
   * can consume the module directly instead of via an intermediate
   * textual LLVM IR round-trip. */
  /* Reject leftover matlab.* ops before the conversion pipeline —
   * SCFToControlFlow / FuncToLLVM / etc. SIGSEGV on them when an
   * unrecognized call shape leaves a `matlab.subscript` /
   * `matlab.const_char` / `matlab.undef` on a `none`-typed value
   * chain. See the comment on validateAllMatlabOpsLowered for the
   * cascade we're preventing. */
  if (mlir::failed(mlirgen::validateAllMatlabOpsLowered(M)))
    return 1;

  mlir::PassManager PM(&MCtx.get());
  PM.addPass(mlir::createCanonicalizerPass());
  PM.addPass(mlir::createSCFToControlFlowPass());
  PM.addPass(mlir::createConvertControlFlowToLLVMPass());
  PM.addPass(mlir::createArithToLLVMConversionPass());
  PM.addPass(mlir::createConvertFuncToLLVMPass());
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(PM.run(M))) {
    std::cerr << "error: REPL MLIR-to-LLVM conversion pipeline failed\n";
    return 1;
  }

  /* The JIT path (ExecutionEngine::create below) is stricter than the
   * `-emit-llvm` translator about unknown llvm.func parameter attrs —
   * the matlab.name / matlab.fi_* attrs we stamp for EmitC / SV need
   * to be stripped before translation or it errors with "Unhandled
   * parameter attribute". */
  mlirgen::stripMatlabFuncAttrs(M);

  if (getenv("MATLABC_REPL_DUMP")) {
    mlirgen::printModule(std::cerr, M);
  }

  mlir::ExecutionEngineOptions EngineOpts;
  EngineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  auto EngineOrErr = mlir::ExecutionEngine::create(M, EngineOpts);
  if (!EngineOrErr) {
    std::cerr << "error: ExecutionEngine::create failed: "
              << llvm::toString(EngineOrErr.takeError()) << "\n";
    return 1;
  }
  auto &Engine = *EngineOrErr;
  /* Look up the raw symbol rather than going through invoke<>. The
   * template invoke builds `_mlir_ciface_<name>` and then invokePacked
   * prepends another `_mlir_` layer for the packed wrapper — our
   * script doesn't need packed arg marshalling, so we just cast the
   * raw symbol to a function pointer and call it.
   *
   * LowerIO renames `script` to `main` and changes its return to i32;
   * we match that here. A REPL script has no user-visible return
   * value either way. */
  auto FnOrErr = Engine->lookup("main");
  if (!FnOrErr) {
    std::cerr << "error: lookup(\"main\") failed: "
              << llvm::toString(FnOrErr.takeError()) << "\n";
    return 1;
  }
  using Thunk = int (*)(void);
  auto Fn = reinterpret_cast<Thunk>(*FnOrErr);
  (void)Fn();
  /* Keep this turn's ExecutionEngine alive for the rest of the REPL
   * session.  A turn that does `f = @(x) ...` stores a function
   * pointer into *this* turn's JIT'd code in the workspace; a later
   * turn (`fminunc(f, ...)`) calls back through it.  If the engine
   * were destroyed at end of turn that code memory would be freed and
   * the cross-turn call would jump into reclaimed memory.  The MLIR
   * context outlives all turns (runRepl owns it), so parking the
   * engines in a session-lifetime vector is sufficient.  Memory grows
   * per turn, which is acceptable for an interactive session. */
  static std::vector<std::unique_ptr<mlir::ExecutionEngine>> g_ReplEngines;
  g_ReplEngines.push_back(std::move(Engine));
  return 0;
}

/* ===========================================================================
 * REPL help command
 *
 * Table-driven. `help` without args prints a grouped topic index; `help <name>`
 * prints a detailed entry. Intercepted in the REPL loop BEFORE the compile
 * pipeline — help isn't a real builtin on the Sema side, it's a REPL UX
 * affordance (matching MATLAB's own `help` command shape).
 * =========================================================================*/

struct HelpEntry {
  const char *name;
  const char *group;
  const char *sig;
  const char *desc;
  const char *examples;
};

static const HelpEntry HelpTable[] = {
  // ---- FFT / complex ----
  {"fft", "FFT",
   "Y = fft(X)",
   "DFT of a real or complex vector / matrix column. Pure-C Cooley-Tukey.",
   "fft([1 2 3 4])\n"
   "   10+0i  -2+2i  -2+0i  -2-2i\n"
   "% round-trip:\n"
   "ifft(fft([1 2 3 4]))\n"
   "   1  2  3  4"},
  {"ifft", "FFT",
   "X = ifft(Y)",
   "Inverse DFT. Applies a 1/N scale per MATLAB's convention.",
   "ifft(fft([1 2 3 4]))  % recovers the input up to rounding"},
  {"fft2", "FFT",
   "Y = fft2(X)",
   "2-D DFT. Applies fft along rows then columns (separable transform).",
   "fft2(eye(4))  % identity → all-ones 4x4 complex matrix"},
  {"ifft2", "FFT",
   "X = ifft2(Y)",
   "Inverse 2-D DFT.",
   "ifft2(fft2(magic(4)))  % recovers magic(4)"},
  {"conj", "Complex",
   "c = conj(z)",
   "Complex conjugate. Polymorphic — identity on real input.",
   "conj(3 + 4i)      % 3 - 4i\n"
   "conj([1+2i  3-1i]) % [1-2i  3+1i]"},
  {"real", "Complex",
   "r = real(z)",
   "Real part of a complex value. Returns a real matrix.",
   "real(3 + 4i)      % 3\n"
   "real(fft([1 2 3 4]))"},
  {"imag", "Complex",
   "i = imag(z)",
   "Imaginary part. Returns a real matrix (zeros for real input).",
   "imag(3 + 4i)      % 4"},
  {"angle", "Complex",
   "phi = angle(z)",
   "Argument of a complex value in radians.",
   "angle(1 + 1i)     % 0.7854 (π/4)"},
  {"abs", "Complex",
   "m = abs(x)",
   "Magnitude. Real fast path; complex path uses hypot(re,im).",
   "abs(-3)          % 3\n"
   "abs(3 + 4i)      % 5"},

  // ---- Linear algebra ----
  {"inv", "Linear algebra",
   "B = inv(A)",
   "Matrix inverse via LU with partial pivoting. Real-only today.",
   "A = [4 3; 6 3];\n"
   "inv(A)\n"
   "   -0.5    0.5\n"
   "    1     -0.667"},
  {"det", "Linear algebra",
   "d = det(A)",
   "Determinant. Falls out of the LU pivoting sign.",
   "det([1 2; 3 4])   % -2"},
  {"svd", "Linear algebra",
   "s = svd(A)",
   "Singular values (column vector). `[U,S,V]` form is a roadmap item.",
   "svd(magic(4))    % [34, 17.889, 4.472, 0]"},
  {"eig", "Linear algebra",
   "v = eig(A)\n       [V, D] = eig(A)",
   "Eigenvalues (1-return) or eigenvectors + diagonal (2-return). Jacobi; symmetric input.",
   "eig([2 -1 0; -1 2 -1; 0 -1 2])\n"
   "[V, D] = eig([4 1; 1 3]);\n"
   "V * D * V'       % reconstructs the input"},
  {"lu", "Linear algebra",
   "[L, U] = lu(A)",
   "LU factorization via Doolittle with partial pivoting.",
   "[L, U] = lu([4 3; 6 3]);\n"
   "L * U            % recovers the input"},
  {"qr", "Linear algebra",
   "[Q, R] = qr(A)",
   "QR via modified Gram-Schmidt with reorthogonalization. m ≥ n.",
   "[Q, R] = qr([1 2; 3 4; 5 6]);\n"
   "Q' * Q           % identity (up to rounding)"},
  {"chol", "Linear algebra",
   "R = chol(A)",
   "Cholesky factor (upper). Input must be positive-definite.",
   "R = chol([4 2; 2 3]);\n"
   "R' * R           % recovers the input"},
  {"pinv", "Linear algebra",
   "B = pinv(A)",
   "Moore-Penrose pseudo-inverse via normal equations.",
   "A = [1 2; 3 4; 5 6];\n"
   "pinv(A) * A      % identity 2x2 (up to rounding)"},
  {"norm", "Linear algebra",
   "n = norm(A)",
   "Frobenius norm.",
   "norm([3 4])      % 5\n"
   "norm(eye(3))     % sqrt(3)"},
  {"trace", "Linear algebra",
   "t = trace(A)",
   "Sum of diagonal entries.",
   "trace(magic(4))  % 34"},
  {"kron", "Linear algebra",
   "K = kron(A, B)",
   "Kronecker product.",
   "kron(eye(2), [1 2; 3 4])"},

  // ---- Creation / shape ----
  {"zeros", "Creation",
   "A = zeros(n)\n       A = zeros(m, n)\n       A = zeros(m, n, p)",
   "Matrix (or 3-D array) of zeros.",
   "zeros(3)\n"
   "zeros(2, 3)"},
  {"ones", "Creation",
   "A = ones(n)\n       A = ones(m, n)",
   "Matrix of ones.",
   "ones(3)\n"
   "ones(2, 3)"},
  {"eye", "Creation",
   "A = eye(n)\n       A = eye(m, n)",
   "Identity matrix (non-square form supported).",
   "eye(4)"},
  {"rand", "Creation",
   "A = rand(n)\n       A = rand(m, n)",
   "Uniform random on [0, 1). Deterministic seed per invocation.",
   "rand(3)"},
  {"randn", "Creation",
   "A = randn(n)\n       A = randn(m, n)",
   "Standard-normal random (Box-Muller).",
   "randn(2, 5)"},
  {"magic", "Creation",
   "A = magic(n)",
   "Magic square of order n.",
   "magic(4)"},
  {"linspace", "Creation",
   "v = linspace(a, b, n)",
   "n evenly-spaced points from a to b, endpoints inclusive.",
   "linspace(0, 1, 5)\n"
   "   0  0.25  0.5  0.75  1"},
  {"diag", "Creation",
   "d = diag(A)\n       D = diag(v)",
   "Matrix → diagonal vector, or vector → diagonal matrix.",
   "diag([1 2 3])\n"
   "diag([1 2; 3 4])  % [1; 4]"},
  {"reshape", "Shape",
   "B = reshape(A, m, n)",
   "Reshape keeping element order (column-major).",
   "reshape(1:6, 2, 3)"},
  {"repmat", "Shape",
   "B = repmat(A, m, n)",
   "Tile A m-by-n times.",
   "repmat([1 2], 2, 3)"},
  {"transpose", "Shape",
   "B = A'   % ctranspose (complex-conjugate)\n       B = A.'  % transpose (no conjugate)",
   "Matrix transpose. `'` conjugates for complex matrices; `.'` does not.",
   "A = [1+1i 2; 3 4];\n"
   "A'               % conjugate transpose\n"
   "A.'              % plain transpose"},
  {"size", "Shape",
   "s = size(A)\n       [m, n] = size(A)\n       k = size(A, dim)",
   "Matrix dimensions. Three forms: row vector, multi-return, single-dim.",
   "[m, n] = size([1 2 3; 4 5 6])   % m=2, n=3"},
  {"length", "Shape",
   "n = length(A)",
   "Longest dimension.",
   "length([1 2 3 4])   % 4"},
  {"numel", "Shape",
   "n = numel(A)",
   "Total number of elements.",
   "numel(eye(3))       % 9"},

  // ---- Reductions ----
  {"sum", "Reduction",
   "s = sum(A)\n       s = sum(A, dim)",
   "Column-wise sum (default); dimension-aware variant.",
   "sum([1 2 3 4])     % 10\n"
   "sum(magic(4), 1)   % row vector of column sums"},
  {"prod", "Reduction",
   "p = prod(A)\n       p = prod(A, dim)",
   "Column-wise product; dimension-aware variant.",
   "prod(1:5)          % 120"},
  {"mean", "Reduction",
   "m = mean(A)\n       m = mean(A, dim)",
   "Column-wise mean; dimension-aware variant.",
   "mean([1 2 3 4])    % 2.5"},
  {"min", "Reduction",
   "m = min(A)\n       m = min(A, B)\n       m = min(A, [], dim)",
   "Column-wise min (default), elementwise min of two, or dim-aware.",
   "min([3 1 4 1 5])   % 1\n"
   "min([1 5], [3 2])  % [1 2]"},
  {"max", "Reduction",
   "m = max(A)\n       m = max(A, B)",
   "Column-wise max; elementwise-of-two; dim-aware.",
   "max([3 1 4 1 5])   % 5"},
  {"cumsum", "Reduction",
   "c = cumsum(A)\n       c = cumsum(A, dim)",
   "Running sum.",
   "cumsum([1 2 3 4])  % [1 3 6 10]"},
  {"sort", "Search",
   "s = sort(A)",
   "Column-wise ascending sort.",
   "sort([3 1 4 1 5 9 2 6])"},
  {"find", "Search",
   "i = find(A)",
   "Linear indices of non-zero entries.",
   "find([0 1 0 1 1])  % [2; 4; 5]"},
  {"unique", "Search",
   "u = unique(A)",
   "Unique sorted entries.",
   "unique([3 1 4 1 5 9 2 6 5 3])"},

  // ---- I/O ----
  {"disp", "I/O",
   "disp(x)",
   "Print a value without a label. Polymorphic (scalar / matrix / complex / string).",
   "disp(pi)\n"
   "disp([1 2 3])\n"
   "disp(3 + 4i)"},
  {"fprintf", "I/O",
   "fprintf(fmt, a, b, ...)",
   "C-style formatted print. Up to 4 numeric args in v1.",
   "fprintf('%d + %d = %d\\n', 2, 3, 5)\n"
   "fprintf('%.4f\\n', pi)"},
  {"sprintf", "I/O",
   "s = sprintf(fmt, ...)",
   "Format to a string instead of stdout.",
   "s = sprintf('%.2f', pi);\n"
   "disp(s)            % \"3.14\""},
  {"error", "I/O",
   "error(msg)",
   "Throw a runtime error. Caught by surrounding try/catch if any.",
   "try\n"
   "   error('boom')\n"
   "catch ME\n"
   "   disp(ME.message)\n"
   "end"},

  // ---- Control flow ----
  {"for", "Control",
   "for i = start:step:end\n         body\n       end",
   "Range-based loop. Step is optional (defaults to 1).",
   "for i = 1:5\n"
   "   disp(i);\n"
   "end"},
  {"while", "Control",
   "while cond\n         body\n       end",
   "Conditional loop.",
   "i = 1;\n"
   "while i <= 5\n"
   "   disp(i); i = i + 1;\n"
   "end"},
  {"if", "Control",
   "if cond, body\n       elseif cond, body\n       else body\n       end",
   "Conditional. `elseif` / `else` optional.",
   "x = 3;\n"
   "if x > 0, disp('pos'); elseif x == 0, disp('zero'); else disp('neg'); end"},
  {"parfor", "Control",
   "parfor i = start:end\n         body\n       end",
   "Parallel for — pthread per iteration. Reductions (`x = x + i`) get a mutex.",
   "x = 0;\n"
   "parfor i = 1:10\n"
   "   x = x + i;\n"
   "end\n"
   "disp(x)  % 55"},
  {"try", "Control",
   "try\n         body\n       catch ME\n         body\n       end",
   "Catch runtime errors. `ME.message` holds the thrown string.",
   "try\n"
   "   error('oops')\n"
   "catch ME\n"
   "   disp(ME.message)\n"
   "end"},
  {"function", "Control",
   "function y = f(x) ... end\n       function [u, v] = g(x) ... end",
   "User-defined function. Multi-return via `[a, b]` on LHS.",
   "function y = sq(x)\n"
   "   y = x * x;\n"
   "end"},
  {"classdef", "OOP",
   "classdef Name\n         properties ... end\n         methods ... end\n       end",
   "User-defined class. Supports inheritance, operator overloading, Dependent props, enums.",
   "classdef Vec2\n"
   "   properties, x, y, end\n"
   "   methods\n"
   "      function obj = Vec2(a, b), obj.x=a; obj.y=b; end\n"
   "   end\n"
   "end"},

  // ---- Constants ----
  {"pi", "Constants",
   "pi",
   "π (3.14159265358979…). Folds to arith.constant at emit time.",
   "sin(pi)   % ~0\n"
   "2 * pi    % 6.2832"},
  {"e", "Constants",
   "e",
   "Euler's number (2.71828…).",
   "e^2       % 7.389"},
  {"Inf", "Constants",
   "Inf",
   "Positive infinity.",
   "Inf > 1e300     % 1"},
  {"NaN", "Constants",
   "NaN",
   "Not-a-number.",
   "NaN == NaN     % 0 (per IEEE 754)"},
  {"eps", "Constants",
   "eps",
   "Machine epsilon for double (2.22e-16).",
   "eps             % 2.2204e-16"},

  // ---- REPL ----
  {"who", "REPL",
   "who",
   "List names in the current workspace.",
   "x = 1;  y = [1 2 3];\n"
   "who     % x, y"},
  {"whos", "REPL",
   "whos",
   "List names + size + class.",
   "A = magic(4);\n"
   "whos"},
  {"clear", "REPL",
   "clear           % wipe the whole workspace\n       clear x         % remove one name",
   "Workspace purge. Command syntax or function syntax both work.",
   "clear x\n"
   "clear"},
  {"dbg", "REPL",
   "dbg(x)\n       dbg(x, 'label')",
   "Source-located debug print to stderr. Works in REPL and compiled code.",
   "A = [1 2; 3 4];\n"
   "dbg(A)\n"
   "dbg(A * 3, 'scaled')"},
  {"help", "REPL",
   "help\n       help <topic>",
   "This command. `help` with no argument lists all topics.",
   "help\n"
   "help fft\n"
   "help classdef"},
  {"ver", "REPL",
   "ver\n       ver <name>\n       version",
   "Product version + the shipped-toolbox inventory (with each toolbox's "
   "shipped tier range). `ver <name>` filters by case-insensitive substring.",
   "ver\n"
   "ver robotics\n"
   "version"},
  {"exit", "REPL",
   "exit\n       quit",
   "Leave the REPL. Ctrl-D does the same.",
   "exit"},
};

static std::string trimLR(std::string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return std::string(s.substr(a, b - a));
}

static void printHelpTopic(const HelpEntry &e) {
  std::cout << "\n  " << e.name << "\n  "
            << std::string(std::strlen(e.name), '=') << "\n\n";
  std::cout << "  GROUP:     " << e.group << "\n\n";
  std::cout << "  SYNOPSIS\n    " << e.sig << "\n\n";
  std::cout << "  DESCRIPTION\n    " << e.desc << "\n\n";
  std::cout << "  EXAMPLES\n    ";
  for (const char *p = e.examples; *p; ++p) {
    std::cout << *p;
    if (*p == '\n' && *(p + 1)) std::cout << "    ";
  }
  std::cout << "\n\n";
}

static void printHelpOverview() {
  std::cout << "\n  matlab_llvm REPL help\n"
            << "  =====================\n\n"
            << "  Usage:\n"
            << "    help               — this overview\n"
            << "    help <topic>       — detailed help on a topic\n"
            << "    ver                — product version + shipped toolboxes\n\n"
            << "  Topics (grouped):\n\n";
  // Group by `group` field, preserving first-seen order.
  std::vector<const char *> groups;
  for (const auto &e : HelpTable) {
    bool seen = false;
    for (auto g : groups) if (g == e.group || std::strcmp(g, e.group) == 0) {
      seen = true; break;
    }
    if (!seen) groups.push_back(e.group);
  }
  for (const char *g : groups) {
    std::cout << "  " << g << "\n   ";
    size_t col = 4;
    for (const auto &e : HelpTable) {
      if (std::strcmp(e.group, g) != 0) continue;
      size_t entryLen = std::strlen(e.name) + 2;
      if (col + entryLen > 70) {
        std::cout << "\n   ";
        col = 4;
      }
      std::cout << " " << e.name;
      col += entryLen;
    }
    std::cout << "\n\n";
  }
}

/* ---- `ver` — product version + shipped-toolbox inventory --------------- *
 * matlab_llvm's analogue of MATLAB's `ver`.  The version's minor number
 * tracks the shipped-toolbox count (bump alongside the README badge). */
static const char *kProductVersion = "0.25.3";
static const char *kProductTagline =
    "a MATLAB compiler + runtime on MLIR / LLVM";

struct ToolboxInfo {
  const char *name;     // display name
  const char *tiers;    // shipped tier / status summary
};

/* Keep in sync with the README "Shipped Toolboxes" table. */
static const ToolboxInfo kToolboxes[] = {
  {"Signal Processing",                 "Tier 1-3"},
  {"Control System",                    "Tier 1-4"},
  {"Communications",                    "Tier 1-7"},
  {"RF",                                "Tier 1-4 + Verilog-A"},
  {"Antenna (subset)",                  "ANT Tier-2"},
  {"Propagation Models",                "PROP Tier 1-3"},
  {"Optimization",                      "Tier 1-5"},
  {"Model Predictive Control",          "Tier 1-6"},
  {"System Identification",             "Tier 1-6"},
  {"Global Optimization",               "Tier 1-6"},
  {"Statistics and Machine Learning",   "Tier 1-6"},
  {"Image Processing",                  "Tier 1-6"},
  {"Curve Fitting",                     "Tier 1-6"},
  {"DSP System",                        "Tier 1-6 + DSP HDL 7-8"},
  {"Wavelet",                           "Tier 1-6"},
  {"Partial Differential Equation",     "Tier 1-4"},
  {"Symbolic Math",                     "Tier 1-4 (SymPP)"},
  {"Stateflow (mStateflow)",            "backend + DAP"},
  {"Financial",                         "Tier 1-7"},
  {"Econometrics",                      "Tier 1-6"},
  {"Fixed-Point Designer",              "Tier 1-5"},
  {"Sensor Fusion and Tracking",        "Tier 1-6"},
  {"Robotics System",                   "Tier 1-6"},
  {"Navigation",                        "Tier 1-6"},
  {"Deep Learning",                     "Tier 1-4 complete (dlarray + autodiff + training + LSTM/GRU/bilstm/lstmp + attention + embedding)"},
};

static void printVersion(const std::string &filter) {
  constexpr int N = static_cast<int>(sizeof(kToolboxes) / sizeof(kToolboxes[0]));
  if (filter.empty()) {
    std::cout << "\n--------------------------------------------------------------------------\n";
    std::cout << "  matlab_llvm — " << kProductTagline << "\n";
    std::cout << "  Version: " << kProductVersion << "   ·   " << N
              << " toolbox surfaces shipped\n";
    std::cout << "  Codegen: LLVM | C | C++ | Python | TypeScript | SystemVerilog | cocotb | Verilog-A\n";
    std::cout << "--------------------------------------------------------------------------\n";
  }
  // Column-aligned toolbox list (filtered by case-insensitive substring).
  std::string fl = filter;
  for (char &c : fl) c = static_cast<char>(std::tolower((unsigned char)c));
  int shown = 0;
  for (const auto &t : kToolboxes) {
    if (!fl.empty()) {
      std::string nl = t.name;
      for (char &c : nl) c = static_cast<char>(std::tolower((unsigned char)c));
      if (nl.find(fl) == std::string::npos) continue;
    }
    std::cout << "  " << t.name;
    int pad = 38 - static_cast<int>(std::strlen(t.name));
    for (int i = 0; i < pad; ++i) std::cout << ' ';
    std::cout << t.tiers << "\n";
    ++shown;
  }
  if (!fl.empty() && shown == 0)
    std::cout << "  no toolbox matching '" << filter << "'.\n";
  std::cout << "\n";
}

/* Returns true if the line was handled as a `ver` / `version` command. */
static bool tryHandleVer(const std::string &rawLine) {
  std::string s = trimLR(rawLine);
  while (!s.empty() && (s.back() == ';' || std::isspace((unsigned char)s.back())))
    s.pop_back();
  if (s.empty()) return false;
  auto stripQuotes = [](std::string t) {
    t = trimLR(t);
    if (t.size() >= 2 &&
        ((t.front() == '\'' && t.back() == '\'') ||
         (t.front() == '"' && t.back() == '"')))
      t = t.substr(1, t.size() - 2);
    return trimLR(t);
  };
  if (s == "ver" || s == "version") { printVersion(""); return true; }
  /* command form: `ver signal` */
  if (s.size() > 4 && (s[3] == ' ' || s[3] == '\t') && s.compare(0, 3, "ver") == 0) {
    printVersion(stripQuotes(s.substr(4)));
    return true;
  }
  /* function form: `ver('signal')` / `ver()` */
  if (s.size() >= 5 && s.compare(0, 4, "ver(") == 0 && s.back() == ')') {
    printVersion(stripQuotes(s.substr(4, s.size() - 5)));
    return true;
  }
  return false;
}

/* Returns true if the line was handled as a help command (caller should
 * skip the compile pipeline for it). */
/* REPL convenience: intercept `loadStateChart('foo.mflow')` (or the
 * double-quoted form) so the user can drop a .mflow file straight
 * into a REPL session without the two-step
 *   $ matlabc -emit-matlab foo.mflow > foo.m
 *   >> run('foo.m')
 * dance. We shell out to the same matlabc binary (resolved via
 * g_MatlabcBinDir captured in main()) with -emit-matlab, capture
 * stdout, and feed the result through runReplInput so the emitted
 * `<chart>_tick` function lands in the live REPL session.
 *
 * Returns the number of REPL pipeline turns consumed (0 if the line
 * isn't a loadStateChart call; ≥1 otherwise). The caller is
 * responsible for incrementing the REPL counter and skipping
 * normal-line handling. */
static int tryHandleLoadStateChart(const std::string &rawLine,
                                   mlirgen::Context &MCtx,
                                   int &Counter);

static bool tryHandleHelp(const std::string &rawLine) {
  std::string s = trimLR(rawLine);
  /* tolerate trailing ";" (MATLAB suppression) and whitespace */
  while (!s.empty() && (s.back() == ';' || std::isspace((unsigned char)s.back())))
    s.pop_back();
  if (s.empty()) return false;

  /* Plain `help` */
  if (s == "help") { printHelpOverview(); return true; }

  /* `help <topic>` — command syntax */
  auto tryTopic = [](const std::string &topic) {
    std::string t = trimLR(topic);
    /* strip optional quotes (function-call form: help('fft')) */
    if (t.size() >= 2 &&
        ((t.front() == '\'' && t.back() == '\'') ||
         (t.front() == '"' && t.back() == '"')))
      t = t.substr(1, t.size() - 2);
    t = trimLR(t);
    if (t.empty()) { printHelpOverview(); return true; }
    for (const auto &e : HelpTable) {
      if (t == e.name) { printHelpTopic(e); return true; }
    }
    std::cout << "  no help entry for '" << t
              << "'. Type 'help' for the topic index.\n";
    return true;
  };

  /* command form: `help fft` */
  if (s.size() > 5 && (s[4] == ' ' || s[4] == '\t') &&
      s.compare(0, 4, "help") == 0) {
    return tryTopic(s.substr(5));
  }
  /* function form: `help(fft)` or `help('fft')` */
  if (s.size() > 6 && s.compare(0, 5, "help(") == 0 && s.back() == ')') {
    return tryTopic(s.substr(5, s.size() - 6));
  }
  return false;
}

/* ===========================================================================
 * REPL line editor
 *
 * Raw-mode termios when stdin is a TTY: arrow keys for history (↑ / ↓),
 * cursor movement (← / →), Home/End, Backspace/Delete, Ctrl-A/E/U/K/L,
 * Ctrl-C (discard line), Ctrl-D (exit on empty; delete-char otherwise).
 * Falls back to std::getline when stdin is piped (scripted REPL input,
 * CI, heredocs).
 * =========================================================================*/

class ReplLineEditor {
public:
  ReplLineEditor() : TtyMode(isatty(STDIN_FILENO)) {
    if (TtyMode) tcgetattr(STDIN_FILENO, &OrigTermios);
  }
  ~ReplLineEditor() { restoreTermios(); }

  void addHistory(const std::string &line) {
    if (line.empty()) return;
    if (!History.empty() && History.back() == line) return;
    History.push_back(line);
    if (History.size() > kMaxHistory) History.erase(History.begin());
  }

  std::optional<std::string> readLine(const char *prompt) {
    if (!TtyMode) return readLineCooked(prompt);
    return readLineRaw(prompt);
  }

private:
  static constexpr size_t kMaxHistory = 500;
  bool TtyMode;
  struct termios OrigTermios;
  std::vector<std::string> History;

  void restoreTermios() {
    if (TtyMode) tcsetattr(STDIN_FILENO, TCSAFLUSH, &OrigTermios);
  }

  std::optional<std::string> readLineCooked(const char *prompt) {
    /* Non-TTY stdin = the caller is a pipe, a heredoc, an editor /
     * IDE harness, or a CI script — none of which need our `>> `
     * prompt characters. Writing them anyway forces every consumer
     * to filter them back out (the matlab_llvm_ide Command Window
     * was double-printing prompts because of exactly this), and it
     * mismatches how Python / Node / irb / every well-behaved REPL
     * handle pipe-driven input. We still flush the no-op so the
     * stream stays in lock-step with std::cin reads.
     *
     * Interactive use is unaffected: the TTY branch in `readLine`
     * dispatches to readLineRaw, which keeps writing the prompt
     * (and re-drawing it on history scroll, line edits, etc.). */
    (void)prompt;
    std::string Line;
    if (!std::getline(std::cin, Line)) {
      std::cout << '\n';
      return std::nullopt;
    }
    return Line;
  }

  static void writeStr(const char *s) { (void)!write(STDOUT_FILENO, s, std::strlen(s)); }
  static void writeStr(const std::string &s) { (void)!write(STDOUT_FILENO, s.data(), s.size()); }

  std::optional<std::string> readLineRaw(const char *prompt) {
    struct termios raw = OrigTermios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    std::string Buf;
    size_t Cursor = 0;
    int HistIdx = (int)History.size();
    std::string Saved;  /* in-progress edit when browsing history */

    auto redraw = [&]() {
      std::string out = "\r\x1b[K";
      out += prompt;
      out += Buf;
      if (Cursor < Buf.size()) {
        out += "\x1b[";
        out += std::to_string(Buf.size() - Cursor);
        out += "D";
      }
      writeStr(out);
    };
    writeStr(prompt);

    auto leave = [&](std::optional<std::string> r) {
      tcsetattr(STDIN_FILENO, TCSAFLUSH, &OrigTermios);
      return r;
    };

    while (true) {
      char c;
      ssize_t n = read(STDIN_FILENO, &c, 1);
      if (n <= 0) { writeStr("\n"); return leave(std::nullopt); }

      /* Ctrl-D: EOF on empty line, delete-char-forward otherwise. */
      if (c == 4) {
        if (Buf.empty()) { writeStr("\n"); return leave(std::nullopt); }
        if (Cursor < Buf.size()) { Buf.erase(Cursor, 1); redraw(); }
        continue;
      }
      /* Ctrl-C: discard line; return empty string so the caller re-prompts. */
      if (c == 3) { writeStr("^C\n"); return leave(std::string{}); }
      /* Enter. */
      if (c == '\r' || c == '\n') { writeStr("\n"); return leave(Buf); }
      /* Backspace. */
      if (c == 127 || c == 8) {
        if (Cursor > 0) { Buf.erase(Cursor - 1, 1); --Cursor; redraw(); }
        continue;
      }
      /* Ctrl-A / Ctrl-E: line start / end. */
      if (c == 1)  { Cursor = 0;          redraw(); continue; }
      if (c == 5)  { Cursor = Buf.size(); redraw(); continue; }
      /* Ctrl-U / Ctrl-K: kill to start / to end. */
      if (c == 21) { Buf.erase(0, Cursor); Cursor = 0; redraw(); continue; }
      if (c == 11) { Buf.erase(Cursor);                   redraw(); continue; }
      /* Ctrl-L: clear screen. */
      if (c == 12) { writeStr("\x1b[2J\x1b[H"); redraw(); continue; }

      /* ESC-prefixed escape sequence (arrow keys, Home, End, Delete, ...). */
      if (c == 27) {
        char seq[3] = {0, 0, 0};
        if (read(STDIN_FILENO, &seq[0], 1) != 1) continue;
        if (read(STDIN_FILENO, &seq[1], 1) != 1) continue;
        if (seq[0] != '[' && seq[0] != 'O') continue;
        switch (seq[1]) {
        case 'A':  /* ↑ — previous history */
          if (HistIdx == (int)History.size()) Saved = Buf;
          if (HistIdx > 0) {
            --HistIdx;
            Buf = History[HistIdx];
            Cursor = Buf.size();
            redraw();
          }
          break;
        case 'B':  /* ↓ — next history */
          if (HistIdx < (int)History.size()) {
            ++HistIdx;
            Buf = (HistIdx == (int)History.size()) ? Saved : History[HistIdx];
            Cursor = Buf.size();
            redraw();
          }
          break;
        case 'C':  /* → */
          if (Cursor < Buf.size()) { ++Cursor; redraw(); }
          break;
        case 'D':  /* ← */
          if (Cursor > 0) { --Cursor; redraw(); }
          break;
        case 'H':  /* Home (some terminals) */
          Cursor = 0; redraw();
          break;
        case 'F':  /* End (some terminals) */
          Cursor = Buf.size(); redraw();
          break;
        case '1':  /* Home (ESC[1~) or ESC[7~ */
        case '7':
          read(STDIN_FILENO, &seq[2], 1);  /* eat the '~' */
          Cursor = 0; redraw();
          break;
        case '4':  /* End (ESC[4~) */
        case '8':
          read(STDIN_FILENO, &seq[2], 1);
          Cursor = Buf.size(); redraw();
          break;
        case '3':  /* Delete (ESC[3~) */
          read(STDIN_FILENO, &seq[2], 1);
          if (Cursor < Buf.size()) { Buf.erase(Cursor, 1); redraw(); }
          break;
        default:
          break;
        }
        continue;
      }
      /* Printable. */
      if ((unsigned char)c >= 32 && c != 127) {
        Buf.insert(Cursor, 1, c);
        ++Cursor;
        redraw();
      }
    }
  }
};

/* Definition for the forward decl up near tryHandleHelp(). */
static int tryHandleLoadStateChart(const std::string &rawLine,
                                   mlirgen::Context &MCtx,
                                   int &Counter) {
  std::string s = trimLR(rawLine);
  while (!s.empty() &&
         (s.back() == ';' || std::isspace((unsigned char)s.back())))
    s.pop_back();
  if (s.empty()) return 0;
  static const std::string Prefix = "loadStateChart(";
  if (s.size() <= Prefix.size() ||
      s.compare(0, Prefix.size(), Prefix) != 0 ||
      s.back() != ')')
    return 0;
  std::string Arg = s.substr(Prefix.size(),
                             s.size() - Prefix.size() - 1);
  Arg = trimLR(Arg);
  if (Arg.size() >= 2 &&
      ((Arg.front() == '\'' && Arg.back() == '\'') ||
       (Arg.front() == '"'  && Arg.back() == '"')))
    Arg = Arg.substr(1, Arg.size() - 2);
  if (Arg.empty()) {
    std::cerr << "loadStateChart: missing .mflow path argument\n";
    return 1;
  }
  std::string Bin = g_MatlabcBinDir.empty()
                       ? std::string("matlabc")
                       : g_MatlabcBinDir + "/matlabc";
  /* Quote the path defensively — spaces, single quotes, and shell
   * metas in .mflow filenames otherwise break the popen() call. */
  std::string Quoted;
  Quoted.reserve(Arg.size() + 8);
  Quoted += "'";
  for (char Ch : Arg) {
    if (Ch == '\'') Quoted += "'\\''";
    else Quoted += Ch;
  }
  Quoted += "'";
  std::string Cmd = Bin + " -emit-matlab " + Quoted + " 2>&1";
  FILE *P = popen(Cmd.c_str(), "r");
  if (!P) {
    std::cerr << "loadStateChart: failed to invoke matlabc\n";
    return 1;
  }
  std::string Captured;
  char Buf[4096];
  size_t N;
  while ((N = fread(Buf, 1, sizeof(Buf), P)) > 0)
    Captured.append(Buf, N);
  int Rc = pclose(P);
  if (Rc != 0) {
    std::cerr << "loadStateChart: matlabc -emit-matlab failed (exit "
              << WEXITSTATUS(Rc) << ")\n";
    if (!Captured.empty()) std::cerr << Captured;
    return 1;
  }
  /* Feed the captured MATLAB source through the REPL pipeline so
   * its `<chart>_tick` (and any sibling chart functions) register
   * in the live session. The chart-tick fn carries persistent
   * state, so subsequent calls advance the chart in-place — when
   * called from inside the same REPL input. The included demo
   * driver at the top of the emitted .m runs 5 ticks and prints
   * the chart's outputs, which is usually enough to verify the
   * lowering. Driving the chart programmatically across REPL
   * turns hits matlabc's current cross-unit function-call gap;
   * use `matlabc -emit-c` / AOT for production drivers. */
  (void)runReplInput(MCtx, Captured, Counter++);
  std::cout << "loadStateChart: emitted " << Arg
            << " — demo driver ran above; the chart's `<name>_tick` "
               "is now stashed in the REPL's user-function table and "
               "can be called directly on subsequent turns."
            << std::endl;
  return 1;
}

int runRepl() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlirgen::Context MCtx;
  mlir::registerBuiltinDialectTranslation(MCtx.get());
  mlir::registerLLVMDialectTranslation(MCtx.get());

  std::cerr << "matlabc REPL (experimental). Ctrl-D or `exit` to quit. "
               "Type `help` for commands.\n";
  ReplLineEditor Editor;
  std::string Accum;
  int Counter = 0;
  while (true) {
    const char *Prompt = Accum.empty() ? ">> " : "   ";
    auto LineOpt = Editor.readLine(Prompt);
    if (!LineOpt) { std::cout << '\n'; break; }
    std::string Line = *LineOpt;

    if (Accum.empty() && (Line == "exit" || Line == "quit" ||
                          Line == "exit;" || Line == "quit;"))
      break;

    /* Help is a REPL-side UX affordance — not a real Sema builtin. Catch
     * it at the top level, before we feed the line into the pipeline. */
    if (Accum.empty() && tryHandleHelp(Line)) {
      Editor.addHistory(Line);
      continue;
    }
    /* `ver` / `version` — product version + shipped-toolbox inventory. */
    if (Accum.empty() && tryHandleVer(Line)) {
      Editor.addHistory(Line);
      continue;
    }
    /* loadStateChart('foo.mflow') is a REPL-only shortcut that wraps
     * the two-step `matlabc -emit-matlab + run` workflow. */
    if (Accum.empty() && tryHandleLoadStateChart(Line, MCtx, Counter)) {
      Editor.addHistory(Line);
      continue;
    }

    Editor.addHistory(Line);
    Accum += Line;
    Accum += '\n';

    /* Lex once to decide if we have a complete balanced input.
     * Three things can keep the REPL collecting more lines:
     *   - block-level keywords still open (if / for / function / ...),
     *   - paren / bracket / brace depth > 0,
     *   - the last meaningful chars on the trailing line are `...`
     *     (MATLAB line-continuation; the lexer eats it silently). */
    SourceManager SM;
    FileID F = SM.addBuffer("<repl>", Accum);
    DiagnosticEngine Diag(SM);
    Lexer Lx(SM, F, Diag);
    auto Toks = Lx.tokenize();
    if (blockDepth(Toks)   > 0) continue;
    if (bracketDepth(Toks) > 0) continue;
    if (hasTrailingEllipsis(Accum)) continue;

    (void)runReplInput(MCtx, Accum, Counter++);
#ifdef MATLAB_LLVM_WITH_PLOT
    /* IDE integration (Matlab_llvm_ide): when the harness sets
     * MATLAB_LLVM_IDE_FIGURES=1, stream every open figure as a
     * sentinel-bracketed base64 PNG over stdout after each REPL
     * input so plot()/figure()/saveas() show up live in the Plots
     * panel without the user needing to call drawnow. The runtime
     * function self-gates on the env var, so this is a cheap no-op
     * for non-IDE invocations. */
    matlab_ide_emit_all_figures();
#endif
    Accum.clear();
  }
  return 0;
}

/* --- DAP (Debug Adapter Protocol) ----------------------------------------
 *
 * `matlabc -dap [FILE.m]` speaks DAP over stdio (JSON-RPC 2.0 with
 * Content-Length framing — same wire shape as LSP). A client (VS Code /
 * nvim-dap / etc.) launches matlabc and then sends:
 *
 *   initialize -> launch (or attach) -> setBreakpoints ...
 *   -> configurationDone  (at this point we spawn the worker thread)
 *   -> on every `stopped` event: stackTrace / scopes / variables /
 *      continue | next | stepIn | stepOut
 *   -> disconnect
 *
 * The worker thread JIT-executes the compiled `main` symbol. The module is
 * built with ReplMode=true (so top-level vars go through matlab_ws_*;
 * inspection piggybacks on the same workspace struct the REPL uses) and
 * DebugMode=true (every statement prefixed by matlab_dbg_hook). The hook
 * checks the runtime breakpoint table + step mode and pthread_cond_waits
 * on the debugger-side condvar when it decides to pause.
 *
 * Scope for v1:
 *   - Full step into / step out across user functions: the lowerer
 *     wraps each user-function body with matlab_dbg_enter_frame /
 *     matlab_dbg_leave_frame, so stackTrace reports the live call
 *     chain instead of a single <script> frame.
 *   - Locals scope = the REPL workspace snapshot. */

/* Prototypes for the runtime DAP API. Defined in matlab_runtime.c and
 * linked into matlabc for this path. */
extern "C" {
void matlab_dbg_enable(int stop_on_entry);
void matlab_dbg_register_file(int32_t file_id, const char *name,
                               int64_t name_len);
const char *matlab_dbg_file_name(int32_t file_id, int64_t *len_out);
void matlab_dbg_clear_breakpoints_in_file(int32_t file_id);
int  matlab_dbg_add_breakpoint(int32_t file_id, int32_t line);
void matlab_dbg_resume(int action);
void matlab_dbg_get_pause(int32_t *file_id, int32_t *line);
int  matlab_dbg_frame_count(void);
int  matlab_dbg_frame_at(int i, int32_t *file_id, int32_t *line,
                          const char **fn_name);
void matlab_dbg_wait_for_pause(void);
int  matlab_dbg_is_paused(void);
int  matlab_dbg_ws_count(void);
const char *matlab_dbg_ws_name(int i, int64_t *len_out);
int  matlab_dbg_ws_kind(int i);
int  matlab_dbg_ws_is_mat3(int i);
double matlab_dbg_ws_f64(int i);
void  *matlab_dbg_ws_ptr(int i);
/* matlab_obj_class_id : matlab_obj* -> int32_t (returned as double).
 * matlab_dbg_class_name(id, &len): class registry lookup, returns the
 * UTF-8 name registered by the lowering (or NULL).  Both are used by
 * buildReplPrelude to walk workspace kind=2 bindings and re-load the
 * matching classdef prelude on subsequent REPL turns. */
double matlab_obj_class_id(void *o);
const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out);
/* #116: kind-stable class-name lookup for a workspace variable, captured at
 * obj-store time (matlab_ws_set_obj).  Survives cross-turn ClassId
 * reassignment, unlike matlab_obj_class_id -> matlab_dbg_class_name. */
const char *matlab_dbg_ws_obj_class_name(const char *name, int64_t len,
                                         int64_t *out_len);
void  matlab_ws_set_f64(const char *name, int64_t len, double v);
void  matlab_ws_set_mat(const char *name, int64_t len, struct matlab_mat *m);
void  matlab_ws_set_string(const char *name, int64_t len, void *s);
void  matlab_ws_clear_one(const char *name, int64_t len);
/* matlab_string accessors. The descriptor layout is private to the
 * runtime; the REPL/DAP side reads bytes through these helpers so
 * formatVar / typeForVar stay layout-agnostic. */
const char *matlab_string_get_data(void *s, int64_t *len_out);
/* Phase 6 — Symbolic Math Toolbox. matlab_dbg_sym_str pretty-prints a
 * matlab_sym* via SymPP and caches the buffer thread-locally. Returns
 * NULL/0-len when the build wasn't configured with MATLAB_LLVM_WITH_SYM
 * (the runtime ships a stub that returns NULL in that case). */
const char *matlab_dbg_sym_str(void *s, int64_t *len_out);
/* Phase 6.1 — symbolic matrix variant; pretty-prints a matlab_symmat*. */
const char *matlab_dbg_symmat_str(void *m, int64_t *len_out);
int64_t     matlab_string_get_len (void *s);

/* Resolver workspace-kind hook (Resolver::WorkspaceKindHookT). Used
 * to seed cross-input REPL bindings: when the resolver auto-declares
 * a name that wasn't assigned in the current TU, this returns the
 * kind under which a prior input bound it (or -1 if absent). The
 * Resolver maps kind=3 to InferredType=stringScalar so disp/strlen/
 * isstring dispatch fires across compilation boundaries. Defined
 * after main.cpp's runtime-introspection externs because it has to
 * walk matlab_dbg_ws_count/_name/_kind. */
extern "C" int32_t matlab_ws_is_videowriter(const char *name, int64_t len);
extern "C" int32_t matlab_ws_is_timetable(const char *name, int64_t len);
extern "C" int replWorkspaceKindHook(const char *name, int64_t len) {
  int n = matlab_dbg_ws_count();
  for (int i = 0; i < n; ++i) {
    int64_t got = 0;
    const char *gn = matlab_dbg_ws_name(i, &got);
    if (got == len && gn && memcmp(gn, name, (size_t)len) == 0) {
      /* #236: a VideoWriter handle is stored under a generic kind (mat/obj),
       * which would re-stamp the binding wrong; the name registry overrides
       * it with kind 15 so the resolver marks the binding IsVideoWriter. */
      if (matlab_ws_is_videowriter(name, len)) return 15;
      /* #259: a timetable is stored generically; the name registry overrides
       * the storage kind with 17 so the Resolver re-stamps the binding
       * IsTimetable (summary/head/TT.col route to the timetable path). */
      if (matlab_ws_is_timetable(name, len)) return 17;
      /* #116: a 3-D array round-trips under the generic mat kind=1, losing
       * its rank for the next turn's Sema.  Report a distinct kind so the
       * Resolver re-stamps the binding 3-D (Binding::IsThreeD) and the N-D
       * subscript store/read detectors fire cross-turn. */
      if (matlab_dbg_ws_is_mat3(i)) return 16;
      return matlab_dbg_ws_kind(i);
    }
  }
  return -1;
}

/* Resolver workspace class-name hook — when a name resolves to a
 * kind=2 (matlab_obj*) binding, look up its class_id via the
 * runtime's obj header and translate to a class name through the
 * registry populated by the Lowering's `matlab_dbg_register_class`
 * emission.  Returns nullptr (and leaves `*len_out` untouched) for
 * non-obj bindings or unregistered class_ids. */
extern "C" const char *replWorkspaceClassNameHook(const char *name,
                                                   int64_t len,
                                                   int64_t *len_out) {
  int n = matlab_dbg_ws_count();
  for (int i = 0; i < n; ++i) {
    int64_t got = 0;
    const char *gn = matlab_dbg_ws_name(i, &got);
    if (got != len || !gn || memcmp(gn, name, (size_t)len) != 0) continue;
    if (matlab_dbg_ws_kind(i) != 2) return nullptr;
    void *o = matlab_dbg_ws_ptr(i);
    if (!o) return nullptr;
    /* #116: prefer the class name captured at store time — the obj's stored
     * class_id is from its turn of construction and a later turn's registry
     * may map that id to a different class (per-TU positional ids). */
    int64_t snLen = 0;
    const char *sn = matlab_dbg_ws_obj_class_name(name, len, &snLen);
    if (sn && snLen > 0) { if (len_out) *len_out = snLen; return sn; }
    int32_t cid = (int32_t)matlab_obj_class_id(o);
    return matlab_dbg_class_name(cid, len_out);
  }
  return nullptr;
}

/* Resolver workspace function-handle signature hook (#119) — returns the
 * stored return-kind (-1 unknown / 0 scalar / 1 matrix) for a kind=13
 * binding so a cross-turn `f(vec)` with a matrix argument dispatches to
 * the matrix-argument trampoline with the right result type. */
extern "C" int32_t matlab_ws_get_handle_sig(const char *name, int64_t len);
extern "C" int replWorkspaceHandleSigHook(const char *name, int64_t len) {
  return (int)matlab_ws_get_handle_sig(name, len);
}

/* Definition of the forward declaration earlier in this file — kept
 * here next to replWorkspaceKindHook because both consume
 * `g_MatlabcBinDir` (set in main()) and share the per-class scanning
 * idiom with the static-input prelude wiring. */
static std::string buildReplPrelude(const std::string &Src) {
  if (g_MatlabcBinDir.empty()) return std::string();
  /* Comment-strip pass — drop everything from `%` to end-of-line so a
   * comment line like `% tf is short for transfer function` doesn't
   * pull the prelude in. */
  std::string Stripped;
  Stripped.reserve(Src.size());
  bool InComment = false;
  for (char c : Src) {
    if (c == '\n') {
      InComment = false;
      Stripped.push_back(c);
      continue;
    }
    if (c == '%') InComment = true;
    if (!InComment) Stripped.push_back(c);
  }
  /* A comment-only / blank turn references nothing, so it never needs
   * a classdef prelude.  Bail early: appending the prelude to such a
   * turn would make `classdef` the first non-comment token, which
   * flips the parser into single-classdef-file mode and trips
   * "stray tokens after classdef" on the second classdef in a
   * multi-class umbrella file (optim_classdefs.m has three). */
  {
    bool AnyCode = false;
    for (char c : Stripped)
      if (!std::isspace((unsigned char)c)) { AnyCode = true; break; }
    if (!AnyCode) return std::string();
  }
  auto wordHit = [&](const char *Name, char Follow1, char Follow2) -> bool {
    size_t NL = std::strlen(Name);
    size_t P = 0;
    while ((P = Stripped.find(Name, P)) != std::string::npos) {
      bool LeftWord = (P > 0) && (std::isalnum((unsigned char)Stripped[P-1]) ||
                                    Stripped[P-1] == '_');
      if (!LeftWord && P + NL <= Stripped.size()) {
        char R = (P + NL < Stripped.size()) ? Stripped[P + NL] : '\0';
        if (R == Follow1) return true;
        size_t Q = P + NL;
        while (Q < Stripped.size() && (Stripped[Q] == ' ' || Stripped[Q] == '\t'))
          Q++;
        if (Q < Stripped.size() && Stripped[Q] == Follow2) {
          /* Single `=` (not `==`) means assignment. */
          if (Follow2 != '=' || Q + 1 >= Stripped.size() ||
              Stripped[Q+1] != '=') return true;
        }
        /* No-paren constructor on the RHS: `m = occupancyMap;` (#79.1).
         * A bare class name at end-of-line or followed only by a
         * statement terminator counts as a mention, mirroring the AOT
         * prelude scanner. */
        if (Q >= Stripped.size() || Stripped[Q] == ';' ||
            Stripped[Q] == ',' || Stripped[Q] == '\n' || Stripped[Q] == '\r')
          return true;
      }
      P += NL;
    }
    return false;
  };
  auto mentions = [&](const char *Name) -> bool {
    /* Call shape `Name(`, assignment shape `Name =`, or a bare RHS
     * mention `Name;` (no-paren constructor, #79.1). */
    return wordHit(Name, '(', '=');
  };
  /* CST prelude classes: tf lives in cst_classdefs.m (shares
   * cst_polyadd helpers); ss / zpk / pid / frd have per-class files.
   *
   * The 2026-05 runtime reorganization moved per-toolbox `.m` files
   * into `runtime/toolbox/<name>/`. The candidate list below probes
   * both the flat layout (legacy / unused) and every toolbox subdir
   * so a single Leaf lookup finds the file wherever it lives. */
  static const char *kToolboxDirs[] = {
    "comm", "rf", "optim", "mpc", "ident", "gads", "pde", "prop", "sym",
    "stateflow", "antenna", "control", "stats", "images", "curvefit",
    "dsp", "gpu", "finance", "econ", "fusion", "robotics", "navigation",
    "dlnet", "rl", "bioinfo",
  };
  std::vector<std::string> Files;
  auto add = [&](const std::string &Leaf) {
    std::vector<std::string> Cands;
    /* Flat layout (legacy fallback + share/ install layout). */
    Cands.push_back(g_MatlabcBinDir + "/../runtime/" + Leaf);
    Cands.push_back(g_MatlabcBinDir + "/runtime/" + Leaf);
    Cands.push_back(g_MatlabcBinDir + "/../share/matlabc/runtime/" + Leaf);
    /* Per-toolbox subdirs (post-reorg layout). */
    for (const char *Tb : kToolboxDirs) {
      Cands.push_back(g_MatlabcBinDir + "/../runtime/toolbox/" + Tb + "/" + Leaf);
      Cands.push_back(g_MatlabcBinDir + "/runtime/toolbox/" + std::string(Tb) + "/" + Leaf);
      Cands.push_back(g_MatlabcBinDir + "/../share/matlabc/runtime/toolbox/" + Tb + "/" + Leaf);
    }
    for (auto &C : Cands) {
      std::ifstream Fp(C);
      if (Fp) {
        /* Dedup — several Want entries can map to the same umbrella
         * prelude file (e.g. optimvar / optimproblem / Optimization*
         * all live in optim_classdefs.m). */
        for (auto &F : Files) if (F == C) return;
        Files.push_back(C);
        if (std::getenv("MATLAB_LLVM_DEBUG_PRELUDE"))
          std::fprintf(stderr, "[prelude] add: %s\n", C.c_str());
        return;
      }
    }
    if (std::getenv("MATLAB_LLVM_DEBUG_PRELUDE"))
      std::fprintf(stderr, "[prelude] NOT FOUND: %s\n", Leaf.c_str());
  };
  /* Per-class wants — each class's prelude file is pulled in only
   * when the user input (or the workspace) actually mentions it.
   * Same per-class file layout as the CST classes: keeps uncalled
   * method bodies out of the TU, which would otherwise survive the
   * lowering passes with `none`-typed args (Sema only refines types
   * for methods that have call sites) and trip the LLVM translation
   * step. */
  struct Want { bool active = false; const char *Name; const char *File; };
  Want Cls[] = {
    /* CST — tf lives in the umbrella `cst_classdefs.m` (shares
     * cst_polyadd / cst_polysub helpers); the others have their
     * own per-class files. */
    {false, "tf",                       "cst_classdefs.m"},
    /* Bioinformatics Tier-4 — the phytree classdef lives in
     * bioinfo_classdefs.m; seqlinkage / seqneighjoin build it. */
    {false, "phytree",                  "bioinfo_classdefs.m"},
    {false, "seqlinkage",               "bioinfo_classdefs.m"},
    {false, "seqneighjoin",             "bioinfo_classdefs.m"},
    {false, "DataMatrix",               "bioinfo_classdefs.m"},
    {false, "ss",                       "cst_class_ss.m"},
    {false, "zpk",                      "cst_class_zpk.m"},
    {false, "pid",                      "cst_class_pid.m"},
    {false, "frd",                      "cst_class_frd.m"},
    /* Comm SO surface — one file per class.  Convolutional / Viterbi /
     * OFDM / channel SOs need matrix-typed property storage and
     * matrix-arg method bodies (currently typed as `none` until the
     * field-type-inference work lands); deferred to a follow-on slice. */
    {false, "CommCRCGenerator",         "comm_class_crc_generator.m"},
    {false, "CommCRCDetector",          "comm_class_crc_detector.m"},
    /* Antenna Toolbox catalog (ANT-Tier-1, geometry-only).  v1 ships
     * the catalog classes with scalar properties; pattern / impedance
     * / sparameters methods require the wire-MoM solver (ANT-Tier-2)
     * and land in a follow-on slice. */
    {false, "AntDipole",   "ant_class_dipole.m"},
    {false, "AntMonopole", "ant_class_monopole.m"},
    /* RF Toolbox catalog (RF-Tier-1, skeleton).  v1 ships scalar
     * properties (NumPorts / Impedance); the Parameters / Frequencies
     * cube + Touchstone reader land in a follow-on slice once
     * matrix-typed classdef property storage is in. */
    {false, "RFSparameters", "rf_class_sparameters.m"},
    /* RF Toolbox sibling network-parameter classdefs (RF-Tier-1
     * follow-on).  Each is a property-holder skeleton paralleling
     * RFSparameters; population from sparamS2y / sparamS2z / sparamS2h
     * / sparamS2abcd happens via direct assignment of the runtime
     * helper's struct return. */
    {false, "RFYparameters",     "rf_class_yparameters.m"},
    {false, "RFZparameters",     "rf_class_zparameters.m"},
    {false, "RFHparameters",     "rf_class_hparameters.m"},
    {false, "RFGparameters",     "rf_class_gparameters.m"},
    {false, "RFAbcdparameters",  "rf_class_abcdparameters.m"},
    {false, "RFTparameters",     "rf_class_tparameters.m"},
    /* RF circuit hierarchy (RF-Tier-4 partial).  Amplifier / mixer /
     * passive blocks cascade through rfbudgetFriis via per-block
     * NF / Gain / IP3 columns; the classdef is the user-facing
     * property holder. */
    {false, "RFCktAmplifier",    "rf_class_amplifier.m"},
    {false, "RFCktMixer",        "rf_class_mixer.m"},
    {false, "RFCktPassive",      "rf_class_passive.m"},
    {false, "RFCktCascade",      "rf_class_cascade.m"},
    {false, "RFCktParallel",     "rf_class_parallel.m"},
    {false, "RFCktSeries",       "rf_class_series.m"},
    {false, "RFCktShunt",        "rf_class_shunt.m"},
    {false, "RFRational",        "rf_class_rfrational.m"},
    /* RF Propagation site descriptors (MathWorks txsite / rxsite
     * shape).  Constructed via the kwarg-sugar — every property is
     * scalar or string (no matrices), so the catalog skeleton
     * works today without additional infrastructure. */
    {false, "TxSite", "rf_class_txsite.m"},
    {false, "RxSite", "rf_class_rxsite.m"},
    {false, "PropagationModel", "rf_class_propagationmodel.m"},
    /* Optimization Toolbox problem-based API (Tier-4).  The umbrella
     * `optim_classdefs.m` holds the OptimizationExpression /
     * OptimizationProblem classdefs plus the `optimvar` /
     * `optimproblem` factory functions; any mention pulls it in (the
     * `add` helper dedups the repeated file). */
    {false, "optimvar",               "optim_classdefs.m"},
    {false, "optimproblem",           "optim_classdefs.m"},
    {false, "OptimizationExpression", "optim_classdefs.m"},
    {false, "OptimizationProblem",    "optim_classdefs.m"},
    /* MPC Toolbox Tier-1 — `mpc_classdefs.m` holds the `mpc` and
     * `mpcstate` classdefs.  Any of these mentions pulls the file in
     * (the prelude builder dedups). */
    {false, "mpc",        "mpc_classdefs.m"},
    {false, "mpcstate",   "mpc_classdefs.m"},
    {false, "mpcmove",    "mpc_classdefs.m"},
    /* Nonlinear MPC (Tier-5) — the `nlmpc` classdef + `nlmpcmove` live in
     * the same umbrella. A program that only mentions nlmpc/nlmpcmove (not
     * the linear `mpc`) still needs the file, and `nlmpcmove` does NOT
     * match the `mpcmove` mention (word-boundary: the `l` prefix). Without
     * this the nlmpc classdef is absent, the binding never gets a
     * PinnedClass, and the Tier-5 nlmpcmove lowering hook (which keys on
     * the `nlmpc` class pin) never fires under -dap/-repl. */
    {false, "nlmpc",      "mpc_classdefs.m"},
    {false, "nlmpcmove",  "mpc_classdefs.m"},
    /* System Identification Toolbox Tier-1 — umbrella
     * `ident_classdefs.m` holds the `iddata` + `idpoly` classes; the
     * estimator/method names (arx / ar / compare / ...) pull it in
     * too so a follow-up REPL turn that only mentions `compare(...)`
     * still has the classdefs in scope. */
    {false, "iddata",     "ident_classdefs.m"},
    {false, "idpoly",     "ident_classdefs.m"},
    {false, "idss",       "ident_classdefs.m"},
    {false, "idgrey",     "ident_classdefs.m"},
    {false, "greyest",    "ident_classdefs.m"},
    {false, "impulseest", "ident_classdefs.m"},
    {false, "forecast",   "ident_classdefs.m"},
    {false, "idfrd",      "ident_classdefs.m"},
    {false, "etfe",       "ident_classdefs.m"},
    {false, "spa",        "ident_classdefs.m"},
    {false, "extendedKalmanFilter",  "ident_classdefs.m"},
    {false, "unscentedKalmanFilter", "ident_classdefs.m"},
    {false, "correct",    "ident_classdefs.m"},
    {false, "recursiveLS",  "ident_classdefs.m"},
    {false, "recursiveARX", "ident_classdefs.m"},
    {false, "idnlgrey",   "ident_classdefs.m"},
    {false, "nlgreyest",  "ident_classdefs.m"},
    {false, "arxOptions", "ident_classdefs.m"},
    {false, "getcov",     "ident_classdefs.m"},
    {false, "getpvec",    "ident_classdefs.m"},
    {false, "setpvec",    "ident_classdefs.m"},
    {false, "n4sid",      "ident_classdefs.m"},
    {false, "ssest",      "ident_classdefs.m"},
    {false, "tfest",      "ident_classdefs.m"},
    {false, "arx",        "ident_classdefs.m"},
    {false, "ar",         "ident_classdefs.m"},
    {false, "armax",      "ident_classdefs.m"},
    {false, "oe",         "ident_classdefs.m"},
    {false, "bj",         "ident_classdefs.m"},
    {false, "iv4",        "ident_classdefs.m"},
    {false, "delayest",   "ident_classdefs.m"},
    {false, "compare",    "ident_classdefs.m"},
    {false, "predict",    "ident_classdefs.m"},
    {false, "resid",      "ident_classdefs.m"},
    {false, "goodnessOfFit", "ident_classdefs.m"},
    /* Sensor Fusion and Tracking Toolbox — `fusion_classdefs.m` holds the
     * `quaternion` value-type + tracking filters (trackingKF/EKF/UKF) +
     * inertial sensor models (imuSensor/gpsSensor) + orientation/pose
     * fusion filters (ahrsfilter/imufilter/complementaryFilter/insfilterMARG).
     * Predict/correct method names are already pulled in by ident; the
     * mentions below cover the strictly-fusion surface. */
    {false, "quaternion",          "fusion_classdefs.m"},
    {false, "trackingKF",          "fusion_classdefs.m"},
    {false, "trackingEKF",         "fusion_classdefs.m"},
    {false, "trackingUKF",         "fusion_classdefs.m"},
    {false, "objectDetection",     "fusion_classdefs.m"},
    {false, "imuSensor",           "fusion_classdefs.m"},
    {false, "gpsSensor",           "fusion_classdefs.m"},
    {false, "ahrsfilter",          "fusion_classdefs.m"},
    {false, "imufilter",           "fusion_classdefs.m"},
    {false, "complementaryFilter", "fusion_classdefs.m"},
    {false, "insfilterMARG",       "fusion_classdefs.m"},
    {false, "ecompass",            "fusion_classdefs.m"},
    {false, "slerp",               "fusion_classdefs.m"},
    {false, "rotatepoint",         "fusion_classdefs.m"},
    {false, "rotateframe",         "fusion_classdefs.m"},
    {false, "quat2eul",            "fusion_classdefs.m"},
    {false, "eul2quat",            "fusion_classdefs.m"},
    {false, "quat2rotm",           "fusion_classdefs.m"},
    {false, "rotm2quat",           "fusion_classdefs.m"},
    {false, "allanvar",            "fusion_classdefs.m"},
    {false, "constvel",            "fusion_classdefs.m"},
    {false, "constacc",            "fusion_classdefs.m"},
    {false, "constturn",           "fusion_classdefs.m"},
    {false, "cvmeas",              "fusion_classdefs.m"},
    {false, "cameas",              "fusion_classdefs.m"},
    {false, "ctmeas",              "fusion_classdefs.m"},
    {false, "initcvekf",           "fusion_classdefs.m"},
    {false, "initctekf",           "fusion_classdefs.m"},
    {false, "waypointTrajectory",  "fusion_classdefs.m"},
    {false, "lookupPose",          "fusion_classdefs.m"},
    {false, "lla2ned",             "fusion_classdefs.m"},
    {false, "ned2lla",             "fusion_classdefs.m"},
    {false, "assignmunkres",       "fusion_classdefs.m"},
    {false, "trackerGNN",          "fusion_classdefs.m"},
    {false, "objectTrack",         "fusion_classdefs.m"},
    {false, "numConfirmed",        "fusion_classdefs.m"},
    {false, "trackFuser",          "fusion_classdefs.m"},
    {false, "trackGOSPAMetric",    "fusion_classdefs.m"},
    {false, "trackOSPAMetric",     "fusion_classdefs.m"},
    {false, "trackErrorMetrics",   "fusion_classdefs.m"},
    {false, "rtsSmoother",         "fusion_classdefs.m"},
    /* Robotics System Toolbox — `robotics_classdefs.m` umbrella. */
    {false, "se3",                       "robotics_classdefs.m"},
    {false, "so3",                       "robotics_classdefs.m"},
    {false, "rigidBodyTree",             "robotics_classdefs.m"},
    {false, "addBody",                   "robotics_classdefs.m"},
    {false, "getTransform",              "robotics_classdefs.m"},
    {false, "geometricJacobian",         "robotics_classdefs.m"},
    {false, "homeConfiguration",         "robotics_classdefs.m"},
    {false, "randomConfiguration",       "robotics_classdefs.m"},
    {false, "loadrobot",                 "robotics_classdefs.m"},
    {false, "inverseKinematics",         "robotics_classdefs.m"},
    {false, "constraintPoseTarget",      "robotics_classdefs.m"},
    {false, "trvec2tform",               "robotics_classdefs.m"},
    {false, "tform2trvec",               "robotics_classdefs.m"},
    {false, "rotm2tform",                "robotics_classdefs.m"},
    {false, "tform2rotm",                "robotics_classdefs.m"},
    {false, "eul2tform",                 "robotics_classdefs.m"},
    {false, "tform2eul",                 "robotics_classdefs.m"},
    {false, "axang2rotm",                "robotics_classdefs.m"},
    {false, "rotm2axang",                "robotics_classdefs.m"},
    {false, "axang2tform",               "robotics_classdefs.m"},
    {false, "tform2axang",               "robotics_classdefs.m"},
    {false, "quat2tform",                "robotics_classdefs.m"},
    {false, "tform2quat",                "robotics_classdefs.m"},
    {false, "homtrans",                  "robotics_classdefs.m"},
    {false, "wrapToPi",                  "robotics_classdefs.m"},
    {false, "wrapTo2Pi",                 "robotics_classdefs.m"},
    {false, "vecnorm",                   "robotics_classdefs.m"},
    {false, "cubicpolytraj",             "robotics_classdefs.m"},
    {false, "trapveltraj",               "robotics_classdefs.m"},
    {false, "transformtraj",             "robotics_classdefs.m"},
    {false, "massMatrix",                "robotics_classdefs.m"},
    {false, "inverseDynamics",           "robotics_classdefs.m"},
    {false, "forwardDynamics",           "robotics_classdefs.m"},
    {false, "gravityTorque",             "robotics_classdefs.m"},
    {false, "velocityProduct",           "robotics_classdefs.m"},
    {false, "centerOfMass",              "robotics_classdefs.m"},
    {false, "importrobot",               "robotics_classdefs.m"},
    {false, "generalizedInverseKinematics","robotics_classdefs.m"},
    {false, "constraintPositionTarget",  "robotics_classdefs.m"},
    {false, "constraintOrientationTarget","robotics_classdefs.m"},
    {false, "constraintJointBounds",     "robotics_classdefs.m"},
    {false, "collisionCylinder",         "robotics_classdefs.m"},
    {false, "collisionCapsule",          "robotics_classdefs.m"},
    {false, "differentialDriveKinematics","robotics_classdefs.m"},
    {false, "unicycleKinematics",        "robotics_classdefs.m"},
    {false, "bicycleKinematics",         "robotics_classdefs.m"},
    {false, "ackermannKinematics",       "robotics_classdefs.m"},
    {false, "derivative",                "robotics_classdefs.m"},
    {false, "binaryOccupancyMap",        "robotics_classdefs.m"},
    {false, "mobileRobotPRM",            "robotics_classdefs.m"},
    {false, "controllerPurePursuit",     "robotics_classdefs.m"},
    {false, "setOccupancy",              "robotics_classdefs.m"},
    {false, "getOccupancy",              "robotics_classdefs.m"},
    {false, "checkOccupancy",            "robotics_classdefs.m"},
    {false, "findpath",                  "robotics_classdefs.m"},
    {false, "collisionBox",              "robotics_classdefs.m"},
    {false, "collisionSphere",           "robotics_classdefs.m"},
    {false, "checkCollision",            "robotics_classdefs.m"},
    {false, "manipulatorRRT",            "robotics_classdefs.m"},
    {false, "plan",                      "robotics_classdefs.m"},
    /* Navigation Toolbox — `navigation_classdefs.m` umbrella (Tiers 1–4). */
    {false, "occupancyMap",              "navigation_classdefs.m"},
    {false, "stateSpaceSE2",             "navigation_classdefs.m"},
    {false, "stateSpaceDubins",          "navigation_classdefs.m"},
    {false, "validatorOccupancyMap",     "navigation_classdefs.m"},
    {false, "navPath",                   "navigation_classdefs.m"},
    {false, "plannerRRT",                "navigation_classdefs.m"},
    {false, "plannerRRTStar",            "navigation_classdefs.m"},
    {false, "plannerAStarGrid",          "navigation_classdefs.m"},
    {false, "lidarScan",                 "navigation_classdefs.m"},
    {false, "lidarSLAM",                 "navigation_classdefs.m"},
    {false, "poseGraph",                 "navigation_classdefs.m"},
    {false, "isStateValid",              "navigation_classdefs.m"},
    {false, "isMotionValid",             "navigation_classdefs.m"},
    {false, "matchScans",                "navigation_classdefs.m"},
    {false, "optimizePoseGraph",         "navigation_classdefs.m"},
    {false, "addRelativePose",           "navigation_classdefs.m"},
    {false, "shortenpath",               "navigation_classdefs.m"},
    {false, "sampleUniform",             "navigation_classdefs.m"},
    {false, "controllerVFH",             "navigation_classdefs.m"},
    {false, "monteCarloLocalization",    "navigation_classdefs.m"},
    {false, "stateEstimatorPF",          "navigation_classdefs.m"},
    {false, "gnssSensor",                "navigation_classdefs.m"},
    {false, "referencePathFrenet",       "navigation_classdefs.m"},
    {false, "trajectoryGeneratorFrenet", "navigation_classdefs.m"},
    {false, "getStateEstimate",          "navigation_classdefs.m"},
    {false, "global2frenet",             "navigation_classdefs.m"},
    {false, "frenet2global",             "navigation_classdefs.m"},
    {false, "gnssconstellation",         "navigation_classdefs.m"},
    {false, "receiverposition",          "navigation_classdefs.m"},
    /* Reinforcement Learning Toolbox — `rl_classdefs.m` umbrella (Tier 1). */
    {false, "rlPredefinedEnv",           "rl_classdefs.m"},
    {false, "rlMDPEnv",                  "rl_classdefs.m"},
    {false, "rlFiniteSetSpec",           "rl_classdefs.m"},
    {false, "rlNumericSpec",             "rl_classdefs.m"},
    {false, "rlFunctionEnv",             "rl_classdefs.m"},
    {false, "rlTable",                   "rl_classdefs.m"},
    {false, "rlQValueFunction",          "rl_classdefs.m"},
    {false, "rlQAgent",                  "rl_classdefs.m"},
    {false, "rlSARSAAgent",              "rl_classdefs.m"},
    {false, "rlDQNAgent",                "rl_classdefs.m"},
    {false, "rlPGAgent",                 "rl_classdefs.m"},
    {false, "rlDDPGAgent",               "rl_classdefs.m"},
    {false, "rlTD3Agent",                "rl_classdefs.m"},
    {false, "rlPPOAgent",                "rl_classdefs.m"},
    {false, "rlSACAgent",                "rl_classdefs.m"},
    {false, "rlGRPOAgent",               "rl_classdefs.m"},
    {false, "rlTRPOAgent",               "rl_classdefs.m"},
    {false, "rlMaxQPolicy",              "rl_classdefs.m"},
    {false, "getAction",                 "rl_classdefs.m"},
    {false, "getMaxQValue",              "rl_classdefs.m"},
    {false, "getGreedyPolicy",           "rl_classdefs.m"},
    {false, "rlQAgentOptions",           "rl_classdefs.m"},
    {false, "rlSARSAAgentOptions",       "rl_classdefs.m"},
    {false, "rlOptimizerOptions",        "rl_classdefs.m"},
    {false, "rlTrainingOptions",         "rl_classdefs.m"},
    {false, "rlSimulationOptions",       "rl_classdefs.m"},
    {false, "getObservationInfo",        "rl_classdefs.m"},
    {false, "getActionInfo",             "rl_classdefs.m"},
    {false, "getCritic",                 "rl_classdefs.m"},
    {false, "getLearnableParameters",    "rl_classdefs.m"},
    /* Deep Learning Toolbox — `dlnet_classdefs.m` (dlarray + autodiff). */
    {false, "dlarray",                   "dlnet_classdefs.m"},
    {false, "dlgradient",                "dlnet_classdefs.m"},
    {false, "extractdata",               "dlnet_classdefs.m"},
    {false, "relu",                      "dlnet_classdefs.m"},
    {false, "sigmoid",                   "dlnet_classdefs.m"},
    {false, "softmax",                   "dlnet_classdefs.m"},
    {false, "crossentropy",              "dlnet_classdefs.m"},
    {false, "mse",                       "dlnet_classdefs.m"},
    {false, "lstm",                      "dlnet_classdefs.m"},
    {false, "embed",                     "dlnet_classdefs.m"},
    {false, "gru",                       "dlnet_classdefs.m"},
    {false, "bilstm",                    "dlnet_classdefs.m"},
    {false, "lstmp",                     "dlnet_classdefs.m"},
    /* DL Phase 1 small ops.  `sqrt` is a generic builtin -- not a trigger.
     * The DL-only activation names trigger the dlnet prelude. */
    {false, "leakyrelu",                 "dlnet_classdefs.m"},
    {false, "gelu",                      "dlnet_classdefs.m"},
    {false, "swish",                     "dlnet_classdefs.m"},
    {false, "softplus",                  "dlnet_classdefs.m"},
    {false, "elu",                       "dlnet_classdefs.m"},
    /* Tier C: rank-4 batched conv + reshape + pool + BN + LN. */
    {false, "conv2d_batch",              "dlnet_classdefs.m"},
    {false, "conv2d_full",               "dlnet_classdefs.m"},
    {false, "maxpool2d",                 "dlnet_classdefs.m"},
    {false, "avgpool2d",                 "dlnet_classdefs.m"},
    {false, "batchnorm",                 "dlnet_classdefs.m"},
    {false, "layernorm",                 "dlnet_classdefs.m"},
    {false, "batchnorm_eval",            "dlnet_classdefs.m"},
    {false, "groupnorm",                 "dlnet_classdefs.m"},
    {false, "batchnorm_train",           "dlnet_classdefs.m"},
    {false, "instancenorm",              "dlnet_classdefs.m"},
    {false, "rmsnorm",                   "dlnet_classdefs.m"},
    /* Global Optimization Toolbox Tier-2 — `gads_classdefs.m` holds the
     * MultiStart + GlobalSearch solver objects.  (`run` is too generic
     * to trigger on; the solver-object mentions pull the prelude.) */
    {false, "MultiStart",        "gads_classdefs.m"},
    {false, "GlobalSearch",      "gads_classdefs.m"},
    {false, "createOptimProblem","gads_classdefs.m"},
    {false, "optimoptions",     "gads_classdefs.m"},
    /* Statistics Toolbox Tier-1 distribution objects. */
    {false, "makedist",          "stats_classdefs.m"},
    {false, "fitdist",           "stats_classdefs.m"},
    {false, "ProbDistUnivParam", "stats_classdefs.m"},
    {false, "fitlm",             "stats_classdefs.m"},
    {false, "fitglm",            "stats_classdefs.m"},
    {false, "LinearModel",       "stats_classdefs.m"},
    {false, "fitcknn",           "stats_classdefs.m"},
    {false, "fitcnb",            "stats_classdefs.m"},
    {false, "fitcdiscr",         "stats_classdefs.m"},
    {false, "fitctree",          "stats_classdefs.m"},
    {false, "fitcsvm",           "stats_classdefs.m"},
    {false, "fitcecoc",          "stats_classdefs.m"},
    {false, "ClassificationModel","stats_classdefs.m"},
    {false, "fitcensemble",       "stats_classdefs.m"},
    {false, "TreeBagger",         "stats_classdefs.m"},
    {false, "affine2d",          "image_classdefs.m"},
    {false, "projective2d",      "image_classdefs.m"},
    {false, "imref2d",           "image_classdefs.m"},
    {false, "fitgeotform2d",     "image_classdefs.m"},
    /* Curve Fitting Toolbox Tier-1 — `curvefit_classdefs.m` holds the
     * `cfit` fitted-model object (+ `fittype` / `fitoptions` carriers).
     * `fit` is the builtin entry; any mention pulls the umbrella in. */
    {false, "fit",               "curvefit_classdefs.m"},
    {false, "cfit",              "curvefit_classdefs.m"},
    {false, "sfit",              "curvefit_classdefs.m"},
    {false, "fittype",           "curvefit_classdefs.m"},
    {false, "fitoptions",        "curvefit_classdefs.m"},
    {false, "coeffvalues",       "curvefit_classdefs.m"},
    {false, "ppform",            "curvefit_classdefs.m"},
    {false, "spline",            "curvefit_classdefs.m"},
    {false, "pchip",             "curvefit_classdefs.m"},
    {false, "ppmak",             "curvefit_classdefs.m"},
    {false, "fnder",             "curvefit_classdefs.m"},
    {false, "fnint",             "curvefit_classdefs.m"},
    /* Financial Toolbox Tier-3 — Portfolio classdef. Any mention of
     * a Portfolio method or the constructor pulls in the umbrella. */
    {false, "Portfolio",            "finance_classdefs.m"},
    {false, "PortfolioCVaR",        "finance_classdefs.m"},
    {false, "PortfolioMAD",         "finance_classdefs.m"},
    {false, "creditscorecard",      "finance_classdefs.m"},
    {false, "fitmodel",             "finance_classdefs.m"},
    {false, "probdefault",          "finance_classdefs.m"},
    {false, "setScenarios",         "finance_classdefs.m"},
    {false, "gbm",                  "finance_classdefs.m"},
    {false, "cir",                  "finance_classdefs.m"},
    {false, "hwv",                  "finance_classdefs.m"},
    {false, "simByEuler",           "finance_classdefs.m"},
    {false, "simBySolution",        "finance_classdefs.m"},
    {false, "setAssetMoments",      "finance_classdefs.m"},
    {false, "setBounds",            "finance_classdefs.m"},
    {false, "setBudget",            "finance_classdefs.m"},
    {false, "setEquality",          "finance_classdefs.m"},
    {false, "setInequality",        "finance_classdefs.m"},
    {false, "setDefaultConstraints","finance_classdefs.m"},
    {false, "estimateFrontier",     "finance_classdefs.m"},
    {false, "estimatePortMoments",  "finance_classdefs.m"},
    {false, "estimatePortReturn",   "finance_classdefs.m"},
    {false, "estimatePortRisk",     "finance_classdefs.m"},
    {false, "estimateMaxSharpeRatio","finance_classdefs.m"},
    {false, "estimateAssetMoments", "finance_classdefs.m"},
    {false, "estimateFrontierByReturn","finance_classdefs.m"},
    {false, "estimateFrontierByRisk", "finance_classdefs.m"},
    /* Econometrics Toolbox model objects (econ_classdefs.m). */
    {false, "arima",                  "econ_classdefs.m"},
    {false, "garch",                  "econ_classdefs.m"},
    {false, "egarch",                 "econ_classdefs.m"},
    {false, "gjr",                    "econ_classdefs.m"},
    {false, "varm",                   "econ_classdefs.m"},
    {false, "ssm",                    "econ_classdefs.m"},
    {false, "dssm",                   "econ_classdefs.m"},
    {false, "bayeslm",                "econ_classdefs.m"},
    {false, "dtmc",                   "econ_classdefs.m"},
    /* mStateflow Tier 4c — `mstateflow_helpers.m` exposes the small
     * MATLAB-level surface (emit / save-op / restore-op / active /
     * reset) that lets a REPL session drive a chart_tick function
     * without ceremony. Pulled in on any `mstateflow_*` mention. */
    {false, "mstateflow_emit",         "mstateflow_helpers.m"},
    {false, "mstateflow_save_op",      "mstateflow_helpers.m"},
    {false, "mstateflow_restore_op",   "mstateflow_helpers.m"},
    {false, "mstateflow_active",       "mstateflow_helpers.m"},
    {false, "mstateflow_reset",        "mstateflow_helpers.m"},
    {false, "mstateflow_push_history", "mstateflow_helpers.m"},
    {false, "mstateflow_pop_history",  "mstateflow_helpers.m"},
    /* mStateflow Tier 4f — `stateChart` classdef is a thin wrapper
     * around the persistent-scalar `<chart>_tick(in1, ..., ev_e1,
     * ...)` function the lowering emits. The previous classdef
     * design (state-struct + init_fn) is gone with the
     * persistent-scalar refactor (2026-05); the wrapper now just
     * captures a function handle + a `reset` that nukes
     * persistents. */
    {false, "stateChart",             "stateflow_classdefs.m"},
    /* DSP System Toolbox — `dsp_classdefs.m` umbrella.  Both the dotted
     * package form (matched in turn-0 source text, e.g.
     * `dsp.FIRFilter(...)`) and the flat classdef name (matched against a
     * persisted workspace object's class name in later REPL turns) point
     * at the same file; the loader dedupes. */
    {false, "dsp.FIRFilter",     "dsp_classdefs.m"},
    {false, "dsp.IIRFilter",     "dsp_classdefs.m"},
    {false, "dsp.BiquadFilter",  "dsp_classdefs.m"},
    {false, "dsp.SOSFilter",     "dsp_classdefs.m"},
    {false, "dsp.Delay",         "dsp_classdefs.m"},
    {false, "dsp.LMSFilter",     "dsp_classdefs.m"},
    {false, "dsp.RLSFilter",     "dsp_classdefs.m"},
    {false, "dsp.FIRDecimator",  "dsp_classdefs.m"},
    {false, "dsp.FIRInterpolator", "dsp_classdefs.m"},
    {false, "dsp.CICDecimator",  "dsp_classdefs.m"},
    {false, "dsp.CICInterpolator", "dsp_classdefs.m"},
    {false, "dsp.SampleRateConverter", "dsp_classdefs.m"},
    {false, "dsp.Channelizer",   "dsp_classdefs.m"},
    {false, "dsp.ChannelSynthesizer", "dsp_classdefs.m"},
    {false, "dsp_FIRFilter",     "dsp_classdefs.m"},
    {false, "dsp_IIRFilter",     "dsp_classdefs.m"},
    {false, "dsp_BiquadFilter",  "dsp_classdefs.m"},
    {false, "dsp_SOSFilter",     "dsp_classdefs.m"},
    {false, "dsp_Delay",         "dsp_classdefs.m"},
    {false, "dsp_LMSFilter",     "dsp_classdefs.m"},
    {false, "dsp_RLSFilter",     "dsp_classdefs.m"},
    {false, "dsp_FIRDecimator",  "dsp_classdefs.m"},
    {false, "dsp_FIRInterpolator", "dsp_classdefs.m"},
    {false, "dsp_CICDecimator",  "dsp_classdefs.m"},
    {false, "dsp_CICInterpolator", "dsp_classdefs.m"},
    {false, "dsp_SampleRateConverter", "dsp_classdefs.m"},
    {false, "dsp_Channelizer",   "dsp_classdefs.m"},
    {false, "dsp_ChannelSynthesizer", "dsp_classdefs.m"},
    /* T5 dotted + flat. */
    {false, "dsp.SineWave",                "dsp_classdefs.m"},
    {false, "dsp.NCO",                     "dsp_classdefs.m"},
    {false, "dsp.Chirp",                   "dsp_classdefs.m"},
    {false, "dsp.MovingAverage",           "dsp_classdefs.m"},
    {false, "dsp.MovingRMS",               "dsp_classdefs.m"},
    {false, "dsp.MovingMaximum",           "dsp_classdefs.m"},
    {false, "dsp.MovingMinimum",           "dsp_classdefs.m"},
    {false, "dsp.MovingStandardDeviation", "dsp_classdefs.m"},
    {false, "dsp.PeakFinder",              "dsp_classdefs.m"},
    {false, "dsp.DCBlocker",               "dsp_classdefs.m"},
    {false, "dsp.ZeroCrossingDetector",    "dsp_classdefs.m"},
    {false, "dsp.SpectrumEstimator",       "dsp_classdefs.m"},
    {false, "dsp.AsyncBuffer",             "dsp_classdefs.m"},
    {false, "dsp_SineWave",                "dsp_classdefs.m"},
    {false, "dsp_NCO",                     "dsp_classdefs.m"},
    {false, "dsp_Chirp",                   "dsp_classdefs.m"},
    {false, "dsp_MovingAverage",           "dsp_classdefs.m"},
    {false, "dsp_MovingRMS",               "dsp_classdefs.m"},
    {false, "dsp_MovingMaximum",           "dsp_classdefs.m"},
    {false, "dsp_MovingMinimum",           "dsp_classdefs.m"},
    {false, "dsp_MovingStandardDeviation", "dsp_classdefs.m"},
    {false, "dsp_PeakFinder",              "dsp_classdefs.m"},
    {false, "dsp_DCBlocker",               "dsp_classdefs.m"},
    {false, "dsp_ZeroCrossingDetector",    "dsp_classdefs.m"},
    {false, "dsp_SpectrumEstimator",       "dsp_classdefs.m"},
    {false, "dsp_AsyncBuffer",             "dsp_classdefs.m"},
    /* T6 dotted + flat. */
    {false, "dsp.LevinsonSolver",  "dsp_classdefs.m"},
    {false, "dsp.NotchPeakFilter", "dsp_classdefs.m"},
    {false, "dsp.LowpassFilter",   "dsp_classdefs.m"},
    {false, "dsp.HighpassFilter",  "dsp_classdefs.m"},
    {false, "dsp_LevinsonSolver",  "dsp_classdefs.m"},
    {false, "dsp_NotchPeakFilter", "dsp_classdefs.m"},
    {false, "dsp_LowpassFilter",   "dsp_classdefs.m"},
    {false, "dsp_HighpassFilter",  "dsp_classdefs.m"},
    /* DSP HDL Toolbox — Tier-7/8 simulation surface. */
    {false, "dsphdl.FIRFilter",    "dsphdl_classdefs.m"},
    {false, "dsphdl.BiquadFilter", "dsphdl_classdefs.m"},
    {false, "dsphdl.SineWave",     "dsphdl_classdefs.m"},
    {false, "dsphdl.NCO",          "dsphdl_classdefs.m"},
    {false, "dsphdl.FIRDecimator", "dsphdl_classdefs.m"},
    {false, "dsphdl.CICDecimator", "dsphdl_classdefs.m"},
    {false, "dsphdl_FIRFilter",    "dsphdl_classdefs.m"},
    {false, "dsphdl_BiquadFilter", "dsphdl_classdefs.m"},
    {false, "dsphdl_SineWave",     "dsphdl_classdefs.m"},
    {false, "dsphdl_NCO",          "dsphdl_classdefs.m"},
    {false, "dsphdl_FIRDecimator", "dsphdl_classdefs.m"},
    {false, "dsphdl_CICDecimator", "dsphdl_classdefs.m"},
    /* GPU Coder — host-side carriers.  Source-text mentions of any
     * gpuArray / gather / coder.gpuConfig / gpuDevice form pull in
     * gpu_classdefs.m which ships the gpuArray + coder_gpuConfig
     * handle classdefs.  See docs/gpu_coder_roadmap.md §1 (T1.4). */
    {false, "gpuArray",         "gpu_classdefs.m"},
    {false, "gather",           "gpu_classdefs.m"},
    {false, "existsOnGPU",      "gpu_classdefs.m"},
    {false, "gpuDevice",        "gpu_classdefs.m"},
    {false, "coder.gpuConfig",  "gpu_config_classdefs.m"},
    {false, "coder_gpuConfig",  "gpu_config_classdefs.m"},
    /* GPU Coder Tier-5 design-pattern helpers — gpucoder.reduce /
     * matrixMatrixKernel / stencilfun / sort all live as C runtime
     * functions (runtime/toolbox/gpu/runtime_gpu_helpers.cpp) routed
     * through the LowerTensorOps dispatch table.  No classdef prelude
     * file is needed; the function-handle ABI delivers the user's @f
     * to the runtime entry. */
  };
  /* Source-mention scan: turn-0-style detection. */
  for (auto &W : Cls) if (mentions(W.Name)) W.active = true;

  /* Workspace-mention scan: subsequent REPL turns may not re-mention
   * the class name (e.g. `crc = CommCRCGenerator(1)` in turn 0, then
   * just `crc(1)` in turn 1).  Walk every kind=2 workspace binding,
   * resolve its class_id → class name, and union into the active set
   * so the next TU still has the classdef in scope. */
  int n = matlab_dbg_ws_count();
  for (int i = 0; i < n; ++i) {
    if (matlab_dbg_ws_kind(i) != 2) continue;
    void *o = matlab_dbg_ws_ptr(i);
    if (!o) continue;
    /* #116: prefer the store-time class name (kind-stable) over the volatile
     * class_id -> registry lookup, which can resolve to the wrong class on a
     * later turn (per-TU positional ClassIds). */
    int64_t inLen = 0;
    const char *vn = matlab_dbg_ws_name(i, &inLen);
    int64_t cnLen = 0;
    const char *cn = (vn && inLen > 0)
        ? matlab_dbg_ws_obj_class_name(vn, inLen, &cnLen) : nullptr;
    if (!cn || cnLen <= 0) {
      int32_t cid = (int32_t)matlab_obj_class_id(o);
      cn = matlab_dbg_class_name(cid, &cnLen);
    }
    if (!cn || cnLen <= 0) continue;
    std::string_view CN(cn, (size_t)cnLen);
    for (auto &W : Cls)
      if (CN == W.Name) { W.active = true; break; }
  }

  for (auto &W : Cls) if (W.active) add(W.File);
  std::string Out;
  for (auto &P : Files) {
    std::ifstream In(P);
    if (!In) continue;
    std::ostringstream Buf;
    Buf << In.rdbuf();
    if (!Out.empty()) Out += '\n';
    Out += "% --- prelude ";
    Out += P;
    Out += " ---\n";
    Out += Buf.str();
  }
  /* User-defined function persistence — append the source of any
   * stashed function whose name shows up in this turn's input. The
   * scan is *transitive*: once a function is pulled in, its own
   * source is scanned for further mentions, since one user function
   * may call another. The classic case is the chart_tick's helper
   * funcs (e.g. `gate_tick` calling `openGate`) — both need to be
   * in the same TU. We loop until quiescence. */
  /* Detect names redefined in the current input. A `function ... NAME(`
   * signature should suppress that name's stashed prelude copy — otherwise
   * the verifier sees two `func.func @NAME` and trips on "redefinition of
   * symbol". The new definition will be re-captured by runReplInput on the
   * way out, so future turns see the latest version. */
  std::set<std::string> RedefinedHere;
  {
    size_t I = 0;
    while ((I = Stripped.find("function", I)) != std::string::npos) {
      bool LeftWord = (I > 0) &&
                      (std::isalnum((unsigned char)Stripped[I-1]) ||
                       Stripped[I-1] == '_');
      if (LeftWord) { I += 8; continue; }
      size_t J = I + 8;
      /* Scan the function signature up to the first `(` for a name token.
       * Forms supported: `function NAME(...)`, `function out = NAME(...)`,
       * `function [a,b] = NAME(...)`. */
      size_t Open = Stripped.find('(', J);
      if (Open == std::string::npos) { I += 8; break; }
      std::string Sig = Stripped.substr(J, Open - J);
      /* Take whatever follows the last `=`, else the whole prefix. */
      auto Eq = Sig.rfind('=');
      std::string Tail = (Eq == std::string::npos) ? Sig
                                                     : Sig.substr(Eq + 1);
      /* Strip whitespace, brackets. */
      std::string Name;
      for (char c : Tail) {
        if (std::isalnum((unsigned char)c) || c == '_') Name += c;
        else if (!Name.empty()) break;
      }
      if (!Name.empty()) RedefinedHere.insert(Name);
      I = Open + 1;
    }
  }
  std::set<std::string> Wanted;
  for (auto &P : g_ReplUserFunctions) {
    if (RedefinedHere.count(P.first)) continue;
    if (mentions(P.first.c_str())) Wanted.insert(P.first);
  }
  bool Grew = true;
  while (Grew) {
    Grew = false;
    for (auto &P : g_ReplUserFunctions) {
      if (Wanted.count(P.first)) continue;
      /* Scan every already-wanted function's body for mentions. */
      bool Hit = false;
      auto scanBody = [&](const std::string &Body, const std::string &Name) {
        size_t NL = Name.size();
        size_t Pos = 0;
        while ((Pos = Body.find(Name, Pos)) != std::string::npos) {
          bool LeftWord = (Pos > 0) && (std::isalnum((unsigned char)Body[Pos-1]) ||
                                          Body[Pos-1] == '_');
          if (!LeftWord && Pos + NL < Body.size()) {
            size_t Q = Pos + NL;
            while (Q < Body.size() && (Body[Q] == ' ' || Body[Q] == '\t')) Q++;
            if (Q < Body.size() && Body[Q] == '(') return true;
          }
          Pos += NL;
        }
        return false;
      };
      for (auto &W : Wanted) {
        auto It = g_ReplUserFunctions.find(W);
        if (It == g_ReplUserFunctions.end()) continue;
        if (scanBody(It->second, P.first)) { Hit = true; break; }
      }
      if (Hit) { Wanted.insert(P.first); Grew = true; }
    }
  }
  for (auto &Name : Wanted) {
    auto It = g_ReplUserFunctions.find(Name);
    if (It == g_ReplUserFunctions.end()) continue;
    if (!Out.empty()) Out += '\n';
    Out += "% --- repl-user-fn ";
    Out += Name;
    Out += " ---\n";
    Out += It->second;
    if (!It->second.empty() && It->second.back() != '\n') Out += '\n';
  }
  if (Out.empty()) return std::string();
  return Out;
}
int  matlab_dbg_add_breakpoint_ex(int32_t file_id, int32_t line,
                                   const char *cond, int64_t cond_len,
                                   const char *log,  int64_t log_len);
/* Same as _ex plus a hit-count gate. hit_op encoding:
 *   0 = no gate (default; same as _ex)
 *   1 = ==     2 = >=     3 = >     4 = % (every Nth) */
int  matlab_dbg_add_breakpoint_ex2(int32_t file_id, int32_t line,
                                    const char *cond, int64_t cond_len,
                                    const char *log,  int64_t log_len,
                                    int hit_op, int64_t hit_target);
int  matlab_dbg_breakpoint_meta(int idx, const char **cond, int64_t *cond_len,
                                 const char **log, int64_t *log_len,
                                 int *disabled);
/* Per-bp (file_id, line) accessor — used by reverseContinue to
 * check whether a rewound line lands on an active bp. */
int  matlab_dbg_breakpoint_at(int idx, int32_t *file_id, int32_t *line);
void matlab_dbg_disable_condition(int idx);
int  matlab_dbg_get_pause_bp(void);
/* Per-frame Locals — written by the lowering's mirror calls in
 * DebugMode after every store to a named slot. The DAP server reads
 * these to render `Locals` for any frame in the call stack. The
 * frame_idx convention here matches matlab_dbg.frames[]: 0 is the
 * outermost / script frame, n_frames-1 is the innermost. */
int  matlab_dbg_frame_locals_count(int frame_idx);
const char *matlab_dbg_frame_local_name(int frame_idx, int i,
                                         int64_t *len_out);
int  matlab_dbg_frame_local_kind(int frame_idx, int i);
double matlab_dbg_frame_local_f64(int frame_idx, int i);
void  *matlab_dbg_frame_local_ptr(int frame_idx, int i);
/* Class-instance support. matlab_dbg_class_name resolves the class_id
 * tag stamped on a matlab_obj* by matlab_obj_new. The introspection
 * accessors (_obj_field_*) walk the obj's struct prefix so the DAP
 * server can expand a class instance into one row per property. */
const char *matlab_dbg_class_name(int32_t class_id, int64_t *len_out);
int32_t matlab_dbg_obj_class_id_of(void *obj);
int  matlab_dbg_obj_field_count(void *obj);
const char *matlab_dbg_obj_field_name(void *obj, int i, int64_t *len_out);
int  matlab_dbg_obj_field_kind(void *obj, int i);
double matlab_dbg_obj_field_f64(void *obj, int i);
void *matlab_dbg_obj_field_ptr(void *obj, int i);
void matlab_ws_set_obj(const char *name, int64_t len, void *obj);

/* DAP completeness extras. Each one is a thin reader over state the
 * runtime already maintains (executable lines, function table, error
 * snapshot) — added so the DAP server doesn't need to re-walk MLIR
 * or re-parse the AST to answer breakpointLocations / exceptionInfo /
 * setFunctionBreakpoints requests.
 *
 * matlab_dbg_executable_lines: writes up to `cap` line numbers into
 * `out` (the lines a breakpoint can land on for this file) and
 * returns the total count. Pass `out=NULL, cap=0` to query the count
 * without copying.
 *
 * matlab_dbg_lookup_function: name → (file_id, first body line). 0
 * on miss, 1 on hit.
 *
 * matlab_dbg_set_pause_on_error: when non-zero, the runtime hook
 * pauses on the first hook fired after matlab_set_error sets the
 * flag, surfacing the failing frame to the DAP client.
 *
 * matlab_dbg_last_error_msg / err_frame_count / err_frame_at: read
 * the snapshot captured by matlab_set_error_msg before the unwind.
 * Same shape as matlab_dbg_frame_at but indexes the err_frames[]
 * array instead of the live frames[] stack. */
/* Toggle "pause on error" — when on, the runtime hook surfaces a
 * pause on the first hook fired after matlab_set_error. */
void matlab_dbg_set_pause_on_error(int on);
/* Read the message captured by matlab_set_error_msg before the
 * unwind. NULL/0-len when no error has fired this session. */
const char *matlab_dbg_last_error_msg(int64_t *len_out);
/* True iff the most recent pause came from a `keyboard` call (not a
 * breakpoint, step, or pause request). The DAP server uses this to
 * surface stop reason="entry". */
int matlab_dbg_was_paused_from_keyboard(void);

/* Data breakpoints (write watchpoints). The runtime maintains a
 * per-name watch list and the matlab_ws_set_* / matlab_dbg_frame_set_*
 * sites trip a pause on a name match.
 *
 * `add_watchpoint`: appends or refreshes by id. scope is 0 (any),
 * 1 (script-ws only), 2 (innermost-frame only); v1 always passes 0.
 * `clear_watchpoints`: drops the whole list (the DAP request always
 * carries a fresh full list, so clear-then-add is the simplest impl).
 * `last_watchpoint_id`: id of the watch that tripped the most recent
 * pause, or 0; mirrors hitBreakpointIds for line bps.
 * `was_paused_from_watch`: stop-reason discriminator. */
int matlab_dbg_add_watchpoint(const char *name, int64_t name_len,
                               int32_t scope, int32_t id);
/* Same as add_watchpoint but with explicit access kind:
 *   0 = write only (default; back-compat with the original API)
 *   1 = read only
 *   2 = read+write
 * Read watchpoints fire on matlab_ws_get_* in JIT'd REPL-mode
 * code; frame-local reads go through stack slots and aren't
 * visible to the runtime watch table. */
int matlab_dbg_add_watchpoint_ex(const char *name, int64_t name_len,
                                  int32_t scope, int32_t id,
                                  int32_t access);
void matlab_dbg_clear_watchpoints(void);
int32_t matlab_dbg_last_watchpoint_id(void);
int matlab_dbg_was_paused_from_watch(void);

/* Thread enumeration. Populated lazily as parfor / other workers
 * call into the debug runtime; the main script worker is thread
 * id 1. The DAP `threads` request reports this list; `stopped`
 * events carry the originating thread id. */
int     matlab_dbg_thread_count(void);
int32_t matlab_dbg_thread_id_at(int idx);
int32_t matlab_dbg_paused_thread_id(void);

/* Reverse stepping. Pops one statement's worth of undo records
 * from the runtime's undo log, applies them to revert variable
 * writes, and returns:
 *   1  -> rewound to a statement boundary (out_file_id, out_line
 *         get the resume location)
 *   0  -> log exhausted; nothing rewound
 *  -1  -> hit an irreversible-op marker (out_msg explains)
 * The runtime owns the undo log; the DAP server treats this as
 * an opaque "rewind one step" operation. */
int matlab_dbg_step_back(int32_t *out_file_id, int32_t *out_line,
                         char *out_msg, int64_t msg_cap);

/* Rewound-state query + redo walker. After matlab_dbg_step_back,
 * the JIT thread is still parked one statement past the rewound
 * caret; the DAP server consults matlab_dbg_is_rewound on every
 * forward step and, while true, routes through
 * matlab_dbg_step_forward_redo instead of resuming the JIT. The
 * redo function walks the undo log forward, re-applying each
 * record's post-write state, until either a same-frame boundary
 * is reached or the recorded future is exhausted (caught up to
 * the JIT's parked position). Return values mirror step_back:
 *    1 = landed on a boundary; out_file_id/out_line carry it.
 *    0 = caught up — the caller should resume the JIT normally.
 *   -1 = hit an irreversible-op marker (out_msg explains). */
int matlab_dbg_is_rewound(void);
int matlab_dbg_step_forward_redo(int32_t *out_file_id, int32_t *out_line,
                                  char *out_msg, int64_t msg_cap);

/* readMemory / writeMemory accessors. Hand out a memoryReference
 * (hex pointer string) per matrix-variable row; the DAP server
 * decodes it back to a buffer pointer for the read. Bounded by
 * matlab_dbg_mat_data_bytes so a malformed request can't walk
 * past the buffer. Complex matrices return NULL (their re/im
 * pair can't be summarised through a single pointer). */
void   *matlab_dbg_mat_data_ptr(void *mat);
int64_t matlab_dbg_mat_data_bytes(void *mat);
/* Existing in matlab_runtime.c — re-declared here for the DAP server.
 * `matlab_err_traceback_*` reads the snapshot frames captured at the
 * point matlab_set_error fired, so it survives the unwind. */
int  matlab_err_traceback_count(void);
int  matlab_err_traceback_at(int i, int32_t *file_id, int32_t *line,
                              const char **fn_name);
}

/* Forward declarations from matlab_runtime.c so we can format matrices
 * into human-readable "1x3 double" strings for the DAP `variables`
 * response without duplicating the display logic. */
struct matlab_mat;
struct matlab_mat_c;
struct matlab_mat3;
struct matlab_mat_u8;
struct matlab_mat_i32;
extern "C" int64_t matlab_dbg_mat_rows(struct matlab_mat *m);
extern "C" int64_t matlab_dbg_mat_cols(struct matlab_mat *m);
extern "C" double matlab_dbg_mat_get(struct matlab_mat *m, int64_t i, int64_t j);
extern "C" int64_t matlab_mat_u8_rows (struct matlab_mat_u8  *m);
extern "C" int64_t matlab_mat_u8_cols (struct matlab_mat_u8  *m);
extern "C" int64_t matlab_mat_i32_rows(struct matlab_mat_i32 *m);
extern "C" int64_t matlab_mat_i32_cols(struct matlab_mat_i32 *m);
/* Discriminator: 1 = real 2-D matlab_mat, 2 = matlab_mat_c (complex),
 * 3 = matlab_mat3 (3-D). The DAP server stores a kind=1 ws/frame
 * value as a `void *` because all three share the same LLVM type;
 * matlab_dbg_mat_kind reads the magic byte at offset 0 to dispatch. */
extern "C" int32_t matlab_dbg_mat_kind(const void *p);
extern "C" int64_t matlab_dbg_mat_c_rows(const struct matlab_mat_c *m);
extern "C" int64_t matlab_dbg_mat_c_cols(const struct matlab_mat_c *m);
extern "C" double matlab_dbg_mat_c_re(const struct matlab_mat_c *m,
                                       int64_t i, int64_t j);
extern "C" double matlab_dbg_mat_c_im(const struct matlab_mat_c *m,
                                       int64_t i, int64_t j);
extern "C" int64_t matlab_dbg_mat3_rows(const struct matlab_mat3 *m);
extern "C" int64_t matlab_dbg_mat3_cols(const struct matlab_mat3 *m);
extern "C" int64_t matlab_dbg_mat3_depth(const struct matlab_mat3 *m);
extern "C" double matlab_dbg_mat3_get(const struct matlab_mat3 *m,
                                       int64_t i, int64_t j, int64_t k);
/* Tier C — matN (rank >= 4) reflection. */
extern "C" int32_t matlab_dbg_matN_ndims(const void *p);
extern "C" int64_t matlab_dbg_matN_dim(const void *p, int32_t k_1based);
extern "C" int64_t matlab_dbg_matN_numel(const void *p);
extern "C" double  matlab_dbg_matN_get_lin(const void *p, int64_t lin_zero_based);

/* Phase 5 heterogeneous types — workspace rows for kind 6 (table) /
 * 9 (categorical) / 10 (datetime) / 11 (duration) need their own
 * shape/format paths. We don't include the runtime header here (the
 * DAP server has its own LLVM/MLIR includes that don't play with
 * matlab_runtime.h's C-style guards); forward-declare what we need. */
struct matlab_table_s;        typedef struct matlab_table_s        matlab_table;
struct matlab_categorical_s;  typedef struct matlab_categorical_s  matlab_categorical;
struct matlab_datetime_s;     typedef struct matlab_datetime_s     matlab_datetime;
struct matlab_duration_s;     typedef struct matlab_duration_s     matlab_duration;
extern "C" double      matlab_table_height(matlab_table *t);
extern "C" double      matlab_table_width (matlab_table *t);
extern "C" const char *matlab_table_column_name(matlab_table *t,
                                                 int32_t idx,
                                                 int64_t *out_len);
extern "C" void       *matlab_table_column_data(matlab_table *t, int32_t idx);
extern "C" int32_t     matlab_table_column_kind_idx(matlab_table *t,
                                                     int32_t idx);
extern "C" double      matlab_categorical_length (matlab_categorical *c);
extern "C" double      matlab_categorical_numcats(matlab_categorical *c);
extern "C" double      matlab_duration_to_seconds(matlab_duration *d);

/* mStateflow Tier 4c — forward decls for the chart-runtime snapshot
 * helpers in runtime/runtime_mstateflow.cpp. File-scope so name
 * lookup from inside namespace dap resolves to the global C symbol. */
extern "C" {
int    mstateflow_snapshot_save_blob(const char *Name, const void *Data,
                                     size_t Len);
size_t mstateflow_snapshot_size(const char *Name);
int    mstateflow_snapshot_copy(const char *Name, void *Out, size_t Cap);
void   mstateflow_snapshot_reset(void);
int    mstateflow_snapshot_count(void);
const char *mstateflow_snapshot_name(int Idx);
size_t mstateflow_snapshot_name_size(int Idx);
}

namespace dap {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

/* DAP resume actions — must match matlab_dbg_action in the runtime. */
enum Action { RUN = 0, CONTINUE = 1, STEP_OVER = 2, STEP_IN = 3,
              STEP_OUT = 4, STOP = 5 };

pthread_mutex_t WriteMu = PTHREAD_MUTEX_INITIALIZER;

/* The real stdout FD saved before we redirect stdout to the pipe
 * reserved for the debuggee. All DAP frames go back through this. */
int OriginalStdoutFd = -1;
/* The read end of the pipe the debuggee writes to. Forwarded to the
 * client as `output` events. */
int DebuggeeOutFd = -1;
/* Same pair for stderr — keeps Diag prints (compile / lower errors
 * from REPL eval, error()-traceback emissions) out of the DAP
 * channel while still surfacing them in the IDE's debug console. */
int OriginalStderrFd = -1;
int DebuggeeErrFd = -1;

/* Module-wide state threaded through worker / server / reader. */
struct Shared {
  std::string ProgramPath;   /* absolute / CLI-supplied path */
  std::unique_ptr<mlir::ExecutionEngine> Engine;
  /* JIT-resolved address of `main` — the first instruction of the
   * compiled program. The DAP `disassemble` request uses this as
   * the implicit base when the IDE asks to disassemble "from the
   * top" (no memoryReference supplied). Set in workerMain right
   * before the call, so it's available for any request that comes
   * in while the worker is paused. */
  void *MainAddr = nullptr;
  int32_t FileId = 1;
  pthread_t Worker;
  bool WorkerStarted = false;
  bool WorkerExited = false;
  pthread_mutex_t Mu = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t Cv = PTHREAD_COND_INITIALIZER;
  int NextSeq = 1;
  /* Mapping from canonicalized source path to the runtime's file_id.
   * Populated at compileProgram() with every file the SourceManager
   * loaded, then consulted by setBreakpoints to look up the id for
   * the source the IDE asked about. Keys are realpath()-resolved so
   * "./examples/factorial.m" and "/abs/.../factorial.m" collapse. */
  std::unordered_map<std::string, int32_t> PathToFileId;
  /* Per-file set of line numbers a breakpoint can land on. Populated
   * during compileProgram by walking every statement in the script
   * body and every function body. The DAP `breakpointLocations`
   * request reads from this so the IDE can grey out lines that
   * aren't valid bp targets. The set is approximate — it lists every
   * statement's start line, which is a superset of the lines the
   * MLIR lowering's `matlab_dbg_hook` actually fires on, so the bp
   * install (`setBreakpoints`) is still authoritative for whether a
   * given line resolves. */
  std::unordered_map<int32_t, std::set<int32_t>> BpLocations;
  /* Per-file alias map: lines a Stmt covers but doesn't START on,
   * pointing back to the Stmt's canonical begin line (where the
   * runtime hook actually fires). Populated alongside `BpLocations`
   * during the AST walk in compileProgram. Used by:
   *   - `breakpointLocations` — alias keys are returned as valid bp
   *     candidates, so the IDE highlights every line a `.mflow`
   *     block's JSON spans (not just the line of its `{`).
   *   - `setBreakpoints` — when the user clicks an alias line, we
   *     install the bp at the canonical line so the runtime hook can
   *     match it. Without this, a click on `data.expression` of a
   *     `display` block would either silently snap forward to the
   *     next block (pre-fix) or register a never-firing bp (after
   *     expanding BpLocations alone).  */
  std::unordered_map<int32_t, std::unordered_map<int32_t, int32_t>> BpAliases;
  /* Function name -> (file_id, first body line). Built at
   * compileProgram time from the TU's Function list (top-level +
   * nested) so the DAP `setFunctionBreakpoints` request can install
   * a line breakpoint at the function's entry by name. */
  struct FnEntry { int32_t FileId = 0; int32_t Line = 0; };
  std::unordered_map<std::string, FnEntry> FunctionTable;
  /* Phase 8a: `(file_id, line)` → originating block id, populated
   * for `.mflow` entry points by `flowchart::buildAST`. The DAP
   * `stackTrace` handler appends the block id to each frame's name
   * so the IDE can highlight the active block on the canvas. Empty
   * for `.m` entry points and unused outside `.mflow` programs. */
  std::unordered_map<int64_t, std::string> BlockByLine;
  /* Phase 8c: extra block-library search-path entries supplied by
   * the IDE through DAP `initialize`'s `initializationOptions.blockPath`
   * (a JSON array of strings). Threaded into
   * `BuildOptions::BlockSearchPath` ahead of `MATFORGE_BLOCK_PATH`
   * env-var entries when compiling `.mflow` programs, so a project
   * can configure block libraries through its DAP launch
   * configuration without setting environment variables on the
   * matlabc subprocess. Empty for `.m` programs and unused. */
  std::vector<std::string> BlockPathFromIDE;
  /* Phase 8b: the block id we were stopped on when the user
   * issued the most recent `next` (step over). Set by the `next`
   * handler before it calls `matlab_dbg_resume(STEP_OVER)`; the
   * monitor thread reads it on every step-pause to suppress
   * stops that landed inside the same block (e.g. an
   * `expression` block whose `data.expression` parses to two
   * Stmts steps once across both). Cleared on `continue` /
   * `stepIn` / `stepOut` so per-statement granularity returns
   * for those modes. Empty for `.m` programs (BlockByLine is
   * empty and the monitor's lookup always misses). */
  std::string StepOverBlockId;
  /* Breakpoints set against a path the runtime hasn't registered
   * yet (e.g. setBreakpoints arrived before launch / compileProgram).
   * Held here keyed by canonical path, replayed when the path
   * later registers. Each entry mirrors the DAP request payload so
   * we can re-verify with the same condition / logMessage /
   * hitCondition the IDE sent originally. */
  struct PendingBp {
    std::string Path;
    int32_t Line = 0;
    std::string Condition;
    std::string LogMessage;
    std::string HitCondition;
  };
  std::vector<PendingBp> PendingBps;
  /* Class methods grouped by class name. Built at compileProgram
   * from each ClassDef's `Methods` + `StaticMethods` lists. Used
   * by the `variables` expansion of a class instance to surface
   * "method rows" alongside property rows — the IDE's debugger
   * panel renders properties under a value icon and methods under
   * a function icon (presentationHint.kind="method").
   *
   * The inheritance chain is followed via ClassParent (class name
   * -> super-class name) so a `Savings < Account` instance lists
   * its own Rate property + Savings constructor *and* inherits
   * Account's deposit method. */
  struct MethodEntry {
    std::string Name;
    int32_t FileId = 0;
    int32_t Line = 0;
    bool Static = false;
    std::vector<std::string> Inputs;
    std::vector<std::string> Outputs;
    std::string DefiningClass;   /* for "inherited from X" hint */
  };
  std::unordered_map<std::string, std::vector<MethodEntry>> ClassMethods;
  /* ClassName -> direct superclass name. Empty when no `< Super`
   * clause. Walked iteratively to gather inherited methods. */
  std::unordered_map<std::string, std::string> ClassParent;
  /* Counter bumped every time a continue/next/stepIn/stepOut request
   * is processed. The monitor records the pre-resume value when it
   * blocks for the client's response and exits its inner wait once
   * the counter has advanced. This is robust to the worker re-pausing
   * inside the wait window — without the counter we'd see paused
   * flip 1→0→1 and conclude the resume hadn't happened. */
  uint64_t ResumeGen = 0;
  /* Serialise step requests against monitor delivery.
   *
   * StepsRequested counts step (next/stepIn/stepOut) requests
   * issued by the server to the runtime. StopsEmitted counts
   * `stopped` events the monitor has actually published to the
   * client. The handler waits for StopsEmitted >= StepsRequested
   * before bumping StepsRequested and issuing a new resume — this
   * prevents back-to-back step clicks (faster than the worker can
   * pause + the monitor can emit) from coalescing into fewer
   * stops and leaving the IDE thinking the program is still
   * running. Continue/pause/breakpoint stops bump StopsEmitted
   * too; that is intentional and harmless (the handler condition
   * is "monitor has caught up to my requests"). */
  uint64_t StepsRequested = 0;
  uint64_t StopsEmitted = 0;
  /* Set by the monitor under G.Mu the moment it observes a pause
   * (between exiting the outer wait and dispatching the stopped
   * event), cleared after the stopped event has been sent and
   * StopsEmitted bumped. The step handler's wait predicate also
   * tests this flag, so a `next` arriving mid-delivery cannot
   * race ahead of the monitor and resume the worker before the
   * IDE has been told about the pause. Without this, the monitor
   * would re-check is_paused() outside the lock, see 0 (because
   * the racing handler had already resumed), and silently skip
   * the stopped event — losing one step per race. */
  bool MonitorBusy = false;
};

Shared G;

/* Lexicographic line read from stdin. DAP/LSP headers are CRLF-
 * terminated. Read bytes directly so we don't get stuck in cin's
 * line buffering across the header/body boundary. */
std::optional<std::string> readFrame() {
  size_t ContentLength = 0;
  std::string Line;
  while (true) {
    Line.clear();
    int c;
    while ((c = std::cin.get()) != EOF) {
      if (c == '\r') {
        if (std::cin.peek() == '\n') std::cin.get();
        break;
      }
      if (c == '\n') break;
      Line.push_back((char)c);
    }
    if (c == EOF) return std::nullopt;
    if (Line.empty()) break;
    const char Key[] = "Content-Length:";
    if (Line.compare(0, sizeof Key - 1, Key) == 0) {
      const char *s = Line.c_str() + sizeof Key - 1;
      while (*s == ' ' || *s == '\t') ++s;
      ContentLength = (size_t)std::strtoul(s, nullptr, 10);
    }
  }
  if (ContentLength == 0) return std::string{};
  std::string Body(ContentLength, '\0');
  std::cin.read(&Body[0], (std::streamsize)ContentLength);
  if (std::cin.gcount() != (std::streamsize)ContentLength) return std::nullopt;
  return Body;
}

/* Write a DAP frame to the saved original stdout FD (the debuggee
 * owns the "plumbing" stdout and we mustn't stomp on its output). */
void writeFrame(const Value &V) {
  std::string Body;
  llvm::raw_string_ostream OS(Body);
  OS << V;
  OS.flush();
  std::string Hdr = "Content-Length: " + std::to_string(Body.size()) +
                     "\r\n\r\n";
  pthread_mutex_lock(&WriteMu);
  (void)!write(OriginalStdoutFd, Hdr.data(), Hdr.size());
  (void)!write(OriginalStdoutFd, Body.data(), Body.size());
  pthread_mutex_unlock(&WriteMu);
}

int seq() { return G.NextSeq++; }

void sendResponse(int64_t RequestSeq, llvm::StringRef Command, bool Success,
                  Value Body) {
  Object O{
    {"seq", seq()},
    {"type", "response"},
    {"request_seq", RequestSeq},
    {"success", Success},
    {"command", Command},
  };
  if (Success) {
    O["body"] = std::move(Body);
  } else {
    /* On failure, DAP puts the error payload in `message` + `body`. */
    O["message"] = std::move(Body);
  }
  writeFrame(Value(std::move(O)));
}

void sendEvent(llvm::StringRef Event, Value Body = Object{}) {
  Object O{
    {"seq", seq()},
    {"type", "event"},
    {"event", Event},
    {"body", std::move(Body)},
  };
  writeFrame(Value(std::move(O)));
}

/* Helpers -----------------------------------------------------------*/

/* Absolute path for the DAP `source.path` field. The client typically
 * sends file URIs ("file:///abs/path"); we stored the path as given
 * via the CLI or `launch.program` — emit it verbatim. */
std::string absPath(const std::string &P) { return P; }

/* Resolve a path to an absolute, symlink-collapsed form for use as a
 * key in PathToFileId. Returns the original string when realpath()
 * fails (e.g. a phantom path the IDE supplied for a file that no
 * longer exists). The resulting key is what every lookup in the map
 * compares against; canonicalising both sides means relative,
 * symlinked, and trailing-slash-equivalent paths all collapse. */
std::string canonPath(const std::string &P) {
  if (P.empty()) return P;
  char Resolved[PATH_MAX];
  if (realpath(P.c_str(), Resolved)) return std::string(Resolved);
  return P;
}

Object sourceObj() {
  Object O;
  O["name"] = G.ProgramPath.substr(G.ProgramPath.find_last_of('/') + 1);
  O["path"] = absPath(G.ProgramPath);
  return O;
}

/* Build a DAP source object for a specific runtime file_id by
 * resolving the path through matlab_dbg_file_name. Falls back to
 * the entry-point's source when the id is unknown (e.g. a frame
 * whose file_id was never registered). */
Object sourceObjForFile(int32_t Fid) {
  int64_t Len = 0;
  const char *Name = matlab_dbg_file_name(Fid, &Len);
  if (!Name || Len == 0) return sourceObj();
  std::string Path(Name, (size_t)Len);
  Object O;
  O["name"] = Path.substr(Path.find_last_of('/') + 1);
  O["path"] = absPath(Path);
  return O;
}

/* Single MLIR context shared by the program JIT and any condition /
 * log-point evaluator runs the monitor thread fires off. mlir::
 * MLIRContext isn't thread-safe, but the worker only touches it
 * during compileProgram; afterward the JIT'd code runs against
 * a finalized engine and the monitor thread is the sole consumer. */
mlirgen::Context &sharedDapContext() {
  static mlirgen::Context Ctx;
  static bool Inited = false;
  if (!Inited) {
    mlir::registerBuiltinDialectTranslation(Ctx.get());
    mlir::registerLLVMDialectTranslation(Ctx.get());
    /* Without an explicit handler MLIR drops diagnostics on the floor,
     * so a failed verification or conversion becomes a silent
     * "failed to compile program" with an empty stderr. Forward every
     * MLIR diagnostic to std::cerr so future regressions are
     * actually debuggable. */
    Ctx.get().getDiagEngine().registerHandler(
        [](mlir::Diagnostic &D) {
          std::cerr << "matlabc: ";
          if (auto FLC = mlir::dyn_cast<mlir::FileLineColLoc>(D.getLocation()))
            std::cerr << FLC.getFilename().str() << ':' << FLC.getLine()
                      << ':' << FLC.getColumn() << ": ";
          std::cerr << D.str() << '\n';
          for (auto &N : D.getNotes()) std::cerr << "  note: " << N.str() << '\n';
          return mlir::success();
        });
    Inited = true;
  }
  return Ctx;
}

/* Forward decls -- runReplInput is defined above in the same
 * anonymous namespace; matlab_ws_* are runtime entries linked into
 * matlabc. The conditional-breakpoint evaluator below calls into
 * both. */
extern "C" {
double matlab_ws_get_f64(const char *name, int64_t len);
double matlab_ws_has(const char *name, int64_t len);
}

/* Counter so each condition / log eval gets a unique <repl:N> file
 * name in error messages. */
int NextEvalId = 1000000;

/* Bridge a runtime function frame's mini-workspace into matlab_ws
 * for the duration of a REPL eval, then reverse the bridge.
 *
 * Used by:
 *   - evaluate (watch / hover / repl), parameterised by frameId
 *   - cond/log breakpoint evaluators, always against the innermost
 *     paused frame so a bp inside `compute(a, b)` can have a
 *     condition like `a > 5`
 *
 * Bridge mechanics:
 *   - Snapshot every matlab_ws name that collides with a frame
 *     local (PreExisting) so we can restore the original value.
 *   - Stamp the frame locals onto matlab_ws via the kind-specific
 *     setter (set_f64 / set_mat / set_obj).
 *   - On reverse: clear stamped names that didn't pre-exist, then
 *     restore the pre-existing ones.
 *
 * Script frame (rt index 0) needs no bridging — its locals are
 * already in matlab_ws + frame_locals[0] which the JIT accesses
 * directly. The constructor simply returns without doing work. */
struct FrameBridge {
  struct WsBackup { std::string name; int kind; double f64; void *ptr; };
  std::vector<WsBackup> Backup;
  std::unordered_set<std::string> PreExisting;
  std::vector<std::string> Stamped;
  bool Active = false;

  void stamp(int RtFrameIdx) {
    if (RtFrameIdx <= 0) return;
    Active = true;
    int N = matlab_dbg_ws_count();
    for (int i = 0; i < N; ++i) {
      int64_t Nlen = 0;
      const char *Nm = matlab_dbg_ws_name(i, &Nlen);
      if (!Nm) continue;
      PreExisting.insert(std::string(Nm, (size_t)Nlen));
    }
    int FN = matlab_dbg_frame_locals_count(RtFrameIdx);
    for (int i = 0; i < FN; ++i) {
      int64_t Nlen = 0;
      const char *Nm = matlab_dbg_frame_local_name(RtFrameIdx, i, &Nlen);
      if (!Nm) continue;
      std::string Nstr(Nm, (size_t)Nlen);
      if (PreExisting.count(Nstr)) {
        int wsN = matlab_dbg_ws_count();
        for (int j = 0; j < wsN; ++j) {
          int64_t WL = 0;
          const char *WN = matlab_dbg_ws_name(j, &WL);
          if (!WN || (size_t)WL != Nstr.size() ||
              std::memcmp(WN, Nstr.data(), Nstr.size()) != 0)
            continue;
          int K = matlab_dbg_ws_kind(j);
          WsBackup B{Nstr, K, 0.0, nullptr};
          if (K == 0) B.f64 = matlab_dbg_ws_f64(j);
          else if (K == 1 || K == 2 || K == 3) B.ptr = matlab_dbg_ws_ptr(j);
          Backup.push_back(std::move(B));
          break;
        }
      }
      int K = matlab_dbg_frame_local_kind(RtFrameIdx, i);
      if (K == 0) {
        matlab_ws_set_f64(Nstr.data(), (int64_t)Nstr.size(),
                           matlab_dbg_frame_local_f64(RtFrameIdx, i));
      } else if (K == 1) {
        matlab_ws_set_mat(Nstr.data(), (int64_t)Nstr.size(),
            (struct matlab_mat *)matlab_dbg_frame_local_ptr(
                RtFrameIdx, i));
      } else if (K == 2) {
        matlab_ws_set_obj(Nstr.data(), (int64_t)Nstr.size(),
            matlab_dbg_frame_local_ptr(RtFrameIdx, i));
      } else if (K == 3) {
        matlab_ws_set_string(Nstr.data(), (int64_t)Nstr.size(),
            matlab_dbg_frame_local_ptr(RtFrameIdx, i));
      }
      Stamped.push_back(std::move(Nstr));
    }
  }
  void restore() {
    if (!Active) return;
    for (const std::string &Nstr : Stamped) {
      if (!PreExisting.count(Nstr))
        matlab_ws_clear_one(Nstr.data(), (int64_t)Nstr.size());
    }
    for (const WsBackup &B : Backup) {
      if (B.kind == 0)
        matlab_ws_set_f64(B.name.data(), (int64_t)B.name.size(), B.f64);
      else if (B.kind == 1)
        matlab_ws_set_mat(B.name.data(), (int64_t)B.name.size(),
                           (struct matlab_mat *)B.ptr);
      else if (B.kind == 2)
        matlab_ws_set_obj(B.name.data(), (int64_t)B.name.size(), B.ptr);
      else if (B.kind == 3)
        matlab_ws_set_string(B.name.data(), (int64_t)B.name.size(), B.ptr);
    }
    Backup.clear();
    PreExisting.clear();
    Stamped.clear();
    Active = false;
  }
};

/* Returns the runtime index of the innermost frame, or -1 if no
 * function frame is on the stack (paused inside the script body or
 * pre-launch). The script frame is rt index 0, so >= 1 means we're
 * inside a user function and the bridge is meaningful. */
int innermostFunctionFrameIdx() {
  int Total = matlab_dbg_frame_count();
  if (Total <= 1) return -1;
  return Total - 1;
}

/* Try to evaluate `expr` as a MATLAB scalar in the current
 * workspace. Wraps it in `__matlab_dbg_cond = (expr);` and runs the
 * full REPL pipeline; the result lands in matlab_ws under that name.
 *
 * Bridges the innermost user-function frame (when there is one) so
 * conditions on a bp inside a function body can reference function
 * locals — without the bridge, only script-scope vars are visible.
 *
 * Returns 1 if the expression evaluated to a non-zero scalar, 0 if
 * it evaluated to zero, and -1 if the eval failed (parse error,
 * undefined name, etc). The caller can use -1 to disable the
 * condition so subsequent hits don't keep retrying. */
int evalConditionInWorkspace(const std::string &Expr) {
  FrameBridge FB;
  FB.stamp(innermostFunctionFrameIdx());
  std::string Src = "__matlab_dbg_cond = (" + Expr + ");";
  int Rc = runReplInput(sharedDapContext(), Src, NextEvalId++);
  const char Name[] = "__matlab_dbg_cond";
  int Result = -1;
  if (Rc == 0 && matlab_ws_has(Name, (int64_t)(sizeof Name - 1)) != 0.0) {
    double V = matlab_ws_get_f64(Name, (int64_t)(sizeof Name - 1));
    Result = V != 0.0 ? 1 : 0;
  }
  matlab_ws_clear_one(Name, (int64_t)(sizeof Name - 1));
  FB.restore();
  return Result;
}

/* Walk a logMessage template, substituting `{name}` placeholders
 * with the matching workspace variable's printed form. v1 only
 * resolves bare identifiers — anything more complex (`{a + b}` or
 * `{x(1)}`) is left as the literal substring so the user gets a
 * clear hint to simplify. The output goes through formatVar so
 * matrices become "RxC double" without dumping the whole buffer. */
std::string formatVar(int Kind, int WsIdx);
std::string interpolateLogMessage(const std::string &Tmpl) {
  std::string Out;
  Out.reserve(Tmpl.size());
  for (size_t i = 0; i < Tmpl.size();) {
    if (Tmpl[i] == '{') {
      auto End = Tmpl.find('}', i + 1);
      if (End != std::string::npos) {
        std::string Inner = Tmpl.substr(i + 1, End - i - 1);
        /* Trim whitespace. */
        size_t s = 0, e = Inner.size();
        while (s < e && std::isspace((unsigned char)Inner[s])) ++s;
        while (e > s && std::isspace((unsigned char)Inner[e - 1])) --e;
        Inner = Inner.substr(s, e - s);
        bool IsIdent = !Inner.empty() && (std::isalpha((unsigned char)Inner[0]) || Inner[0] == '_');
        for (size_t k = 1; IsIdent && k < Inner.size(); ++k)
          if (!std::isalnum((unsigned char)Inner[k]) && Inner[k] != '_')
            IsIdent = false;
        if (IsIdent) {
          int N = matlab_dbg_ws_count();
          int Found = -1, Kind = -1;
          for (int j = 0; j < N; ++j) {
            int64_t Nlen = 0;
            const char *Nm = matlab_dbg_ws_name(j, &Nlen);
            if ((size_t)Nlen == Inner.size() &&
                std::memcmp(Nm, Inner.data(), (size_t)Nlen) == 0) {
              Found = j; Kind = matlab_dbg_ws_kind(j);
              break;
            }
          }
          if (Found >= 0) {
            Out += formatVar(Kind, Found);
          } else {
            Out += "{"; Out += Inner; Out += "}";
          }
          i = End + 1;
          continue;
        }
        /* Non-identifier expressions: pass through verbatim. */
        Out += Tmpl.substr(i, End - i + 1);
        i = End + 1;
        continue;
      }
    }
    Out += Tmpl[i++];
  }
  return Out;
}

/* Forward decl: defined alongside the variables-row helpers further
 * down. compileProgram needs it to emit `breakpoint` events for any
 * pending bps that resolved to executable lines after the path
 * registry was populated. */
int64_t encodeBpId(int32_t file_id, int32_t line);

/* Build + JIT the program, store into G.Engine, register its file
 * with the runtime. Returns true on success. */
bool compileProgram() {
  SourceManager SM;
  FileID F = SM.loadFile(G.ProgramPath);
  if (F == 0) {
    std::cerr << "matlabc -dap: cannot open " << G.ProgramPath << "\n";
    return false;
  }
  G.FileId = (int32_t)F;
  G.PathToFileId.clear();
  G.BpLocations.clear();
  G.BpAliases.clear();
  G.FunctionTable.clear();
  G.BlockByLine.clear();
  G.ClassMethods.clear();
  G.ClassParent.clear();

  /* Register every file the SourceManager knows about with the
   * runtime's debug table. Today only the entry-point is loaded;
   * once Sema starts pulling sibling .m files in to resolve
   * cross-file calls they'll appear here automatically and
   * cross-file breakpoints will Just Work.
   *
   * Phase 6: synthesised per-block buffers added by the flowchart
   * builder (`<flow:NODEID>` etc.) are filtered out — they're not
   * real files the IDE can open, and the AST's source ranges have
   * already been remapped to the .mflow's byte offsets so DAP
   * breakpoints set on `.mflow` lines fire correctly. */
  auto registerSMFile = [](FileID Fid, const std::string &Name) {
    if (!Name.empty() && Name.front() == '<') return;
    matlab_dbg_register_file((int32_t)Fid, Name.data(),
                              (int64_t)Name.size());
    G.PathToFileId[canonPath(Name)] = (int32_t)Fid;
  };
  for (size_t i = 1; i <= SM.numFiles(); ++i)
    registerSMFile((FileID)i, SM.getName((FileID)i));

  /* Detect .mflow inputs and route through the flowchart frontend
   * instead of the MATLAB lexer/parser. The resulting TU feeds the
   * same Sema + MLIR pipeline below, so every DAP capability that
   * works on .m files (breakpoints, step in/out/over, evaluate,
   * setVariable, multi-frame stack trace) works on .mflow files
   * too — block-line breakpoints fire because GraphToAST tags
   * each statement's Range.Begin with the originating block's
   * .mflow byte offset (see lib/Flowchart/GraphToAST.cpp). */
  auto endsWith = [](const std::string &S, std::string_view Suf) {
    return S.size() >= Suf.size() &&
           std::string_view(S).substr(S.size() - Suf.size()) == Suf;
  };
  bool IsFlow = endsWith(G.ProgramPath, ".mflow");

  DiagnosticEngine Diag(SM);
  ASTContext AstCtx;
  TranslationUnit *TU = nullptr;

  if (IsFlow) {
    matlab::flowchart::BuildOptions BO;
    /* `data.path` on custom blocks resolves relative to the .mflow
     * file's containing directory. */
    auto Slash = G.ProgramPath.find_last_of("/\\");
    if (Slash != std::string::npos)
      BO.MflowDirectory = G.ProgramPath.substr(0, Slash);
    /* `library_id` block search path. Resolution order matches
     * `matlabc -dap`'s symmetry with the standalone CLI:
     *   1. IDE-supplied entries from `initialize`'s
     *      `initializationOptions.blockPath` (Phase 8c).
     *   2. `MATFORGE_BLOCK_PATH` env-var entries (colon-separated).
     * First match wins; the loader walks the list in order. */
    for (const auto &P : G.BlockPathFromIDE)
      BO.BlockSearchPath.push_back(P);
    if (const char *Env = std::getenv("MATFORGE_BLOCK_PATH")) {
      std::string E = Env;
      size_t Start = 0;
      while (Start <= E.size()) {
        size_t Sep = E.find(':', Start);
        std::string Part = (Sep == std::string::npos)
                               ? E.substr(Start)
                               : E.substr(Start, Sep - Start);
        if (!Part.empty()) BO.BlockSearchPath.push_back(std::move(Part));
        if (Sep == std::string::npos) break;
        Start = Sep + 1;
      }
    }
    auto Doc = matlab::flowchart::loadMflow(SM, F, Diag);
    if (Doc) {
      // Phase 8a: collect the (file_id, line) → block-id map so
      // stackTrace can surface the originating block id in each
      // frame name. The map is shared in `G.BlockByLine` so the
      // stackTrace handler can read it without re-running the
      // builder.
      matlab::flowchart::BlockLineMap BlockMap;
      TU = matlab::flowchart::buildAST(*Doc, AstCtx, SM, Diag, BO,
                                       &BlockMap);
      G.BlockByLine = std::move(BlockMap.Lookup);
    }
  } else {
    Lexer Lx(SM, F, Diag);
    auto Toks = Lx.tokenize();
    Parser P(std::move(Toks), AstCtx, Diag);
    TU = P.parseFile();
  }
  if (!TU || Diag.hasErrors()) { Diag.printAll(); return false; }

  /* #77: inject the classdef prelude — the same toolbox classdef bodies
   * (`tf`, `cfit`, `dlnetwork`, ...) that `-emit-*` and the REPL pull in.
   * Without it a `-dap` launch of any classdef-using example fails to
   * compile because the class name resolves as undefined, even though the
   * same file builds and runs via `-emit-llvm`.  Rather than rewrite the
   * entry-point buffer (which would shift line numbers and break breakpoint
   * mapping), we parse the prelude as a separate TU and merge its classdefs
   * / helper functions in — exactly how the sibling-`.m` merge below works.
   * `.mflow` inputs don't use the MATLAB classdef prelude. */
  if (!IsFlow) {
    std::string Src(SM.getBuffer(F));
    std::string Prelude = buildReplPrelude(Src);
    if (!Prelude.empty()) {
      FileID PF = SM.addBuffer("<dap-prelude>", Prelude);
      DiagnosticEngine PDiag(SM);
      Lexer PLx(SM, PF, PDiag);
      auto PToks = PLx.tokenize();
      Parser PP(std::move(PToks), AstCtx, PDiag);
      /* The prelude is a flat concatenation of several toolbox classdefs
       * (+ helper functions) with no script body. Parse it in
       * multiple-unit mode — the default one-classdef-per-file rule
       * would error "stray tokens after classdef" at the 2nd classdef
       * and `!PDiag.hasErrors()` below would then silently drop the
       * *entire* prelude, leaving every classdef method unlowered and
       * collapsing operator/method dispatch (the root cause of the #77
       * "-dap can't launch classdef-using examples" failures). */
      TranslationUnit *PTU = PP.parseFile(/*AllowMultipleUnits=*/true);
      if (PTU && !PDiag.hasErrors()) {
        for (auto *Fn : PTU->Functions) TU->Functions.push_back(Fn);
        for (auto *Cls : PTU->Classes) TU->Classes.push_back(Cls);
      }
    }
  }

  /* Multi-file breakpoints: walk the entry-point's directory for
   * sibling .m files, parse each, and merge any function-only or
   * classdef-only siblings into the main TU. The merge gives Sema /
   * lowering visibility into helpers defined alongside the entry
   * point, which in turn lets each helper file's lines emit hooks
   * carrying the correct file_id — so an IDE breakpoint set on
   * `helper.m:5` resolves through G.PathToFileId and fires when the
   * compiled helper executes that line.
   *
   * Only function-/classdef-only siblings are pulled in: a sibling
   * that has a script body (top-level statements) is treated as its
   * own entry-point candidate and skipped to avoid stitching in
   * unrelated executable code from neighbouring scripts (the
   * test/Debug/ corpus has many such files).
   *
   * Per-file diagnostics are dropped on parse failure so a malformed
   * sibling doesn't tank the launch — the entry point still
   * compiles. The same shared ASTContext is reused so node lifetimes
   * align with the main TU.
   *
   * Skipped for .mflow entries — flowchart programs reference helper
   * functions through `function`-kind sub-flows or `custom` blocks
   * (with their own search-path resolution), not through ad-hoc
   * sibling `.m` files in the same directory. */
  if (!IsFlow) {
    namespace fs = std::filesystem;
    fs::path EntryPath = fs::path(G.ProgramPath);
    std::error_code EC;
    fs::path Dir = fs::canonical(EntryPath, EC).parent_path();
    if (!EC && fs::exists(Dir, EC)) {
      std::vector<std::string> Siblings;
      for (auto It = fs::directory_iterator(Dir, EC);
           !EC && It != fs::directory_iterator(); ++It) {
        if (!It->is_regular_file()) continue;
        if (It->path().extension() != ".m") continue;
        std::string SP = It->path().string();
        /* Skip the entry point itself — it's already loaded. */
        fs::path Cand = fs::canonical(It->path(), EC);
        if (EC) continue;
        fs::path EntryCanon = fs::canonical(EntryPath, EC);
        if (EC) continue;
        if (Cand == EntryCanon) continue;
        Siblings.push_back(SP);
      }
      /* Sort for deterministic file_id assignment across runs — the
       * IDs are exposed via DAP `source.path` so a stable ordering
       * keeps log lines comparable. */
      std::sort(Siblings.begin(), Siblings.end());
      for (const std::string &SP : Siblings) {
        FileID SF = SM.loadFile(SP);
        if (SF == 0) continue;
        DiagnosticEngine SibDiag(SM);
        Lexer SibLx(SM, SF, SibDiag);
        auto SibToks = SibLx.tokenize();
        Parser SibP(std::move(SibToks), AstCtx, SibDiag);
        TranslationUnit *SibTU = SibP.parseFile();
        if (!SibTU || SibDiag.hasErrors()) continue;
        /* Skip siblings that have a script body — they're scripts in
         * their own right, not function-file helpers. */
        bool HasScriptBody = SibTU->ScriptNode &&
                              SibTU->ScriptNode->Body &&
                              !SibTU->ScriptNode->Body->Stmts.empty();
        if (HasScriptBody) continue;
        for (auto *Fn : SibTU->Functions) TU->Functions.push_back(Fn);
        for (auto *Cls : SibTU->Classes) TU->Classes.push_back(Cls);
      }
      /* Re-sync the path → file_id table now that SM has more entries.
       * This loop runs again at the bottom of the registration block;
       * doing it here keeps both sides consistent if the resolver
       * needs to see the auxiliary files (it shouldn't, but defensive
       * cheap). */
      for (size_t i = 1; i <= SM.numFiles(); ++i)
        registerSMFile((FileID)i, SM.getName((FileID)i));
    }
  }

  /* Walk the parsed TU to populate G.BpLocations and G.FunctionTable
   * — the data the breakpointLocations / setFunctionBreakpoints DAP
   * requests answer from. Does NOT need Sema to have run; we only
   * touch syntactic info (statement source ranges, function names,
   * body block heads). The walker recurses into nested if/for/while/
   * switch/try blocks so a breakpoint set on a line inside a loop
   * body lights up correctly even though the loop's outer Range
   * already covered the line. */
  {
    auto stmtLine = [&](Stmt *S) -> std::pair<int32_t, int32_t> {
      if (!S) return {0, 0};
      auto LC = SM.getLineColumn(S->Range.Begin);
      return {(int32_t)S->Range.Begin.File, (int32_t)LC.Line};
    };
    /* Register the Stmt's canonical begin line in `BpLocations` and
     * any intermediate lines (Begin.Line, End.Line] in `BpAliases`,
     * pointing back to the begin line. Two-tier on purpose:
     *   - The runtime hook fires once per Stmt, on Range.Begin's
     *     line, so the breakpoint INSTALL has to use that line or
     *     the bp will never match.
     *   - But the IDE still wants to let users click any line a
     *     Stmt spans (especially the human-readable `data.expression`
     *     line of a `.mflow` block, not its anonymous opening `{`).
     * The aliases bridge the gap: setBreakpoints rewrites a clicked
     * alias line to its canonical begin line at install time.
     *
     * Constrained to the case where Begin and End share a file id —
     * synthetic `<flow:NODEID>` buffers and any other cross-file
     * Range.End fall back to begin-only behaviour. Capped at 1024
     * lines defensively so a runaway range can't balloon the maps. */
    auto recordStmt = [&](Stmt *S) {
      auto FL = stmtLine(S);
      if (FL.first == 0 || FL.second == 0) return;
      G.BpLocations[FL.first].insert(FL.second);
      if (!S || !S->Range.End.isValid()) return;
      if (S->Range.End.File != S->Range.Begin.File) return;
      uint32_t EndLine = SM.getLineColumn(S->Range.End).Line;
      if (EndLine <= (uint32_t)FL.second) return;
      uint32_t Span = EndLine - (uint32_t)FL.second;
      if (Span > 1024) return;
      auto &AliasMap = G.BpAliases[FL.first];
      for (uint32_t L = (uint32_t)FL.second + 1; L <= EndLine; ++L) {
        /* Don't overwrite an alias already pointing at a closer
         * (later) Stmt: nested control-flow blocks share lines with
         * their outer block, and the inner Stmt's canonical line is
         * what the user expects to break on. AST walk order is
         * outer-then-inner, so the inner write wins via this guard. */
        auto It = AliasMap.find((int32_t)L);
        if (It == AliasMap.end() || It->second < FL.second)
          AliasMap[(int32_t)L] = FL.second;
      }
    };
    std::function<void(Block *)> walkBlock;
    walkBlock = [&](Block *B) {
      if (!B) return;
      for (Stmt *S : B->Stmts) {
        if (!S) continue;
        recordStmt(S);
        switch (S->Kind) {
        case NodeKind::IfStmt: {
          auto *IF = static_cast<IfStmt *>(S);
          walkBlock(IF->Then);
          for (auto &E : IF->Elseifs) walkBlock(E.Body);
          walkBlock(IF->Else);
          break;
        }
        case NodeKind::ForStmt:
          walkBlock(static_cast<ForStmt *>(S)->Body);
          break;
        case NodeKind::WhileStmt:
          walkBlock(static_cast<WhileStmt *>(S)->Body);
          break;
        case NodeKind::SwitchStmt: {
          auto *SW = static_cast<SwitchStmt *>(S);
          for (auto &C : SW->Cases) walkBlock(C.Body);
          break;
        }
        case NodeKind::TryStmt: {
          auto *TS = static_cast<TryStmt *>(S);
          walkBlock(TS->TryBody);
          walkBlock(TS->CatchBody);
          break;
        }
        case NodeKind::Block:
          walkBlock(static_cast<Block *>(S));
          break;
        default:
          break;
        }
      }
    };
    if (TU->ScriptNode) walkBlock(TU->ScriptNode->Body);
    std::function<void(Function *)> walkFn;
    walkFn = [&](Function *Fn) {
      if (!Fn || !Fn->Body) return;
      /* Function table: name → (file_id, first body line). The first
       * body line is the natural breakpoint target for "stop on
       * entry to fn"; if the body is empty, fall back to the
       * function declaration's own start line. */
      int32_t Fid = 0, Ln = 0;
      if (!Fn->Body->Stmts.empty()) {
        auto FL = stmtLine(Fn->Body->Stmts.front());
        Fid = FL.first; Ln = FL.second;
      }
      if (Fid == 0 || Ln == 0) {
        auto LC = SM.getLineColumn(Fn->Range.Begin);
        Fid = (int32_t)Fn->Range.Begin.File;
        Ln = (int32_t)LC.Line;
      }
      G.FunctionTable[std::string(Fn->Name)] = {Fid, Ln};
      walkBlock(Fn->Body);
      for (Function *Nested : Fn->Nested) walkFn(Nested);
    };
    for (Function *Fn : TU->Functions) walkFn(Fn);
    /* Class methods are also breakpoint targets. Each method's body
     * lives in its own Function, attached to the class via Methods.
     * The runtime hooks fire from method bodies the same way they
     * do from free functions.
     *
     * Methods are registered under three keys so the IDE can resolve
     * them with whichever form the user typed:
     *   - bare name:        "deposit"
     *   - dotted form:      "Account.deposit"
     *   - qualified form:   "Account/deposit"  (matches MATLAB's own UI)
     * The bare-name overwrite is intentional — if two classes share a
     * method name (`Account.deposit` and `Savings.deposit`), the last
     * one wins as the bare-name target, but the dotted/qualified
     * forms always disambiguate. Static methods and constructors get
     * the same treatment. */
    auto registerMethod = [&](const std::string &ClassName, Function *Fn,
                              bool Static) {
      if (!Fn || Fn->Name.empty()) return;
      walkFn(Fn);
      /* walkFn already wrote the bare-name entry into FunctionTable.
       * Re-read it so the dotted / qualified aliases point at the same
       * (file_id, line) pair. */
      auto It = G.FunctionTable.find(std::string(Fn->Name));
      if (It == G.FunctionTable.end()) return;
      Shared::FnEntry E = It->second;
      G.FunctionTable[ClassName + "." + std::string(Fn->Name)] = E;
      G.FunctionTable[ClassName + "/" + std::string(Fn->Name)] = E;
      /* Also populate ClassMethods for the variables-row surface. The
       * MethodEntry captures parameter names so the "value column"
       * can render a signature like `@deposit(obj, amt)` instead of
       * a bare name; the IDE renders methods with a function icon
       * via presentationHint. */
      Shared::MethodEntry ME;
      ME.Name = std::string(Fn->Name);
      ME.FileId = E.FileId;
      ME.Line = E.Line;
      ME.Static = Static;
      ME.DefiningClass = ClassName;
      ME.Inputs.reserve(Fn->Inputs.size());
      for (auto N : Fn->Inputs) ME.Inputs.push_back(std::string(N));
      ME.Outputs.reserve(Fn->Outputs.size());
      for (auto N : Fn->Outputs) ME.Outputs.push_back(std::string(N));
      G.ClassMethods[ClassName].push_back(std::move(ME));
    };
    for (ClassDef *C : TU->Classes) {
      if (!C) continue;
      std::string CN(C->Name);
      if (!C->SuperName.empty())
        G.ClassParent[CN] = std::string(C->SuperName);
      for (Function *M : C->Methods)       registerMethod(CN, M, false);
      for (Function *M : C->StaticMethods) registerMethod(CN, M, true);
    }
  }

  SemaContext Sema;
  TypeContext TC;
  Resolver R(Sema, TC, Diag);
  R.setReplMode(true);
  R.setWorkspaceKindHook(&replWorkspaceKindHook);
  R.setWorkspaceClassNameHook(&replWorkspaceClassNameHook);
  R.setWorkspaceHandleSigHook(&replWorkspaceHandleSigHook);
  R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  Inf.run(*TU);
  // Interactive -repl / JIT: keep the PinnedClass fallback OFF — its rewrite
  // crashes the cross-turn dispatch lowering (see p3DesynthDispatch).
  p3DesynthDispatch(AstCtx, *TU, Inf, /*KeyOffPinnedClass=*/false);
  if (Diag.hasErrors()) { Diag.printAll(); return false; }

  /* Keep MLIR context alive for the lifetime of the ExecutionEngine
   * AND for any subsequent breakpoint-condition evaluations the
   * monitor thread runs through runReplInput. Static-local on first
   * call, registers translations once, reused thereafter. */
  mlirgen::Context &MCtx = sharedDapContext();

  auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU, &SM,
                                /*ReplMode=*/true, /*DebugMode=*/true);
  if (Diag.hasErrors() || mlir::failed(mlir::verify(M))) {
    Diag.printAll();
    std::cerr << "matlabc -dap: MLIR verification failed\n";
    return false;
  }

  /* #77: shared in-process software lowering — same single source of
   * truth the REPL uses.  Previously compileProgram carried the
   * thinnest of the three copies (no runRefineSlotTypes, no
   * runLowerAnonCallsPost, no runMonomorphiseUserCalls), which is why
   * DAP launch failed to compile 133 AOT-passing examples. */
  runJitSoftwareLowering(M);

  if (getenv("MATLABC_DAP_DUMP")) mlirgen::printModule(std::cerr, M);
  /* Verify after the matlab-pass batch. The check that used to live
   * after the LLVM-conversion pipeline could only catch failures the
   * conversion attributed cleanly; surfacing a verifier error here
   * pinpoints which matlab pass left a stale op or signature. */
  if (mlir::failed(mlir::verify(M))) {
    std::cerr << "matlabc -dap: MLIR verification failed after matlab passes\n";
    if (!getenv("MATLABC_DAP_DUMP")) mlirgen::printModule(std::cerr, M);
    return false;
  }

  /* Reject leftover matlab.* ops before the conversion pipeline.
   * See the runReplInput site above for the cascade we're
   * preventing. */
  if (mlir::failed(mlirgen::validateAllMatlabOpsLowered(M)))
    return false;

  mlir::PassManager PM(&MCtx.get());
  PM.addPass(mlir::createCanonicalizerPass());
  PM.addPass(mlir::createSCFToControlFlowPass());
  PM.addPass(mlir::createConvertControlFlowToLLVMPass());
  PM.addPass(mlir::createArithToLLVMConversionPass());
  PM.addPass(mlir::createConvertFuncToLLVMPass());
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(PM.run(M))) {
    std::cerr << "matlabc -dap: MLIR-to-LLVM conversion pipeline failed\n";
    if (!getenv("MATLABC_DAP_DUMP")) mlirgen::printModule(std::cerr, M);
    return false;
  }

  /* The JIT path used by `-dap` (ExecutionEngine::create below) errors
   * on unknown llvm.func parameter attrs — strip our matlab.* attrs
   * before translation. Same fix as the `-repl` path. */
  mlirgen::stripMatlabFuncAttrs(M);

  /* Forward decl so the pending-breakpoints replay below + monitorMain
   * (further down) can build `breakpoint` events / stopped events with
   * stable bp ids. The encoder is defined alongside the variables-row
   * helpers further down. */
  extern int64_t encodeBpId(int32_t file_id, int32_t line);

  mlir::ExecutionEngineOptions EngineOpts;
  EngineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  auto EngineOrErr = mlir::ExecutionEngine::create(M, EngineOpts);
  if (!EngineOrErr) {
    std::cerr << "matlabc -dap: ExecutionEngine::create failed: "
              << llvm::toString(EngineOrErr.takeError()) << "\n";
    return false;
  }
  G.Engine = std::move(*EngineOrErr);
  /* Replay any pending breakpoints whose path now resolves through
   * G.PathToFileId. Snap to nearest executable line, install via
   * the runtime, and emit a `breakpoint` event with reason="changed"
   * so the IDE updates the gutter glyph from "unverified" to
   * "verified". Bps that still don't resolve stay queued — a future
   * compileProgram (e.g. after `restart`) gets another chance. */
  if (!G.PendingBps.empty()) {
    std::vector<Shared::PendingBp> StillPending;
    for (Shared::PendingBp &P : G.PendingBps) {
      auto It = G.PathToFileId.find(P.Path);
      if (It == G.PathToFileId.end()) {
        StillPending.push_back(std::move(P));
        continue;
      }
      int32_t PFid = It->second;
      int32_t Line = P.Line;
      auto BL = G.BpLocations.find(PFid);
      if (BL != G.BpLocations.end() && BL->second.count(Line) == 0) {
        auto Snap = BL->second.lower_bound(Line);
        if (Snap == BL->second.end()) continue;
        Line = *Snap;
      }
      int HitOp = 0;
      int64_t HitTarget = 0;
      if (!P.HitCondition.empty()) {
        llvm::StringRef HC(P.HitCondition);
        HC = HC.trim();
        if (HC.consume_front(">=")) HitOp = 2;
        else if (HC.consume_front("==")) HitOp = 1;
        else if (HC.consume_front(">"))  HitOp = 3;
        else if (HC.consume_front("%"))  HitOp = 4;
        else HitOp = 1;
        HC = HC.trim();
        int64_t N = 0;
        if (!HC.getAsInteger(10, N) && N > 0) HitTarget = N;
        else HitOp = 0;
      }
      bool OK = matlab_dbg_add_breakpoint_ex2(
          PFid, Line,
          P.Condition.empty() ? nullptr : P.Condition.data(),
          (int64_t)P.Condition.size(),
          P.LogMessage.empty() ? nullptr : P.LogMessage.data(),
          (int64_t)P.LogMessage.size(),
          HitOp, HitTarget);
      if (!OK) continue;
      Object Bp{
        {"verified", true},
        {"line", (int64_t)Line},
        {"id", encodeBpId(PFid, Line)},
        {"source", Object{
          {"name", P.Path}, {"path", P.Path},
        }},
      };
      sendEvent("breakpoint",
                Object{{"reason", "changed"},
                       {"breakpoint", std::move(Bp)}});
    }
    G.PendingBps = std::move(StillPending);
  }
  return true;
}

/* Worker thread: invokes the JIT'd `main`. Sets WorkerExited + wakes
 * the monitor loop on return. */

void *workerMain(void *) {
  /* #112: never dereference a null engine. The worker is only spawned
   * once `compileProgram` succeeded (see configurationDone), but guard
   * here too so a failed launch can never fault in
   * `ExecutionEngine::lookup` — that was the null-engine SIGSEGV on the
   * compile-failure path. */
  if (!G.Engine) {
    pthread_mutex_lock(&G.Mu);
    G.WorkerExited = true;
    pthread_cond_broadcast(&G.Cv);
    pthread_mutex_unlock(&G.Mu);
    return nullptr;
  }
  auto FnOrErr = G.Engine->lookup("main");
  if (FnOrErr) {
    G.MainAddr = (void *)*FnOrErr;
    using Thunk = int (*)(void);
    auto Fn = reinterpret_cast<Thunk>(*FnOrErr);
    (void)Fn();
  } else {
    std::cerr << "matlabc -dap: lookup(\"main\") failed: "
              << llvm::toString(FnOrErr.takeError()) << "\n";
  }
  pthread_mutex_lock(&G.Mu);
  G.WorkerExited = true;
  pthread_cond_broadcast(&G.Cv);
  pthread_mutex_unlock(&G.Mu);
  return nullptr;
}

/* Monitor thread: waits for either a pause or worker exit, and emits
 * the matching DAP event. Loops until the worker exits. When the
 * pause came from a conditional or log-point breakpoint we filter
 * here — log messages get emitted as `output` events and the
 * worker is resumed without ever telling the IDE we stopped; failing
 * conditions silently resume too. The IDE only sees a `stopped`
 * event for "real" pauses (step, plain bp, or true condition). */
/* #137: the worker and monitor threads run the full MLIR lowering / JIT — the
 * monitor compiles breakpoint *conditions* through runReplInput → lowerToMLIR,
 * which recurses through Lowerer::lowerExpr over a whole REPL TU (prelude +
 * workspace rehydration). The default secondary-thread stack (~512 KB on
 * macOS) overflows that on a non-trivial prelude (SIGBUS in ___chkstk_darwin),
 * intermittently crashing -dap on a conditional breakpoint. Spawn them with an
 * 8 MB stack — matching the main thread's headroom. */
static int dapSpawnBigStack(pthread_t *t, void *(*fn)(void *), void *arg) {
  pthread_attr_t attr;
  if (pthread_attr_init(&attr) != 0)
    return pthread_create(t, nullptr, fn, arg);
  pthread_attr_setstacksize(&attr, 8 * 1024 * 1024);
  int rc = pthread_create(t, &attr, fn, arg);
  pthread_attr_destroy(&attr);
  return rc;
}

void *monitorMain(void *) {
  bool Debug = getenv("MATLABC_DAP_TRACE") != nullptr;
  while (true) {
    pthread_mutex_lock(&G.Mu);
    while (!G.WorkerExited && !matlab_dbg_is_paused())
      pthread_cond_wait(&G.Cv, &G.Mu);
    bool Exited = G.WorkerExited;
    bool Paused = matlab_dbg_is_paused();
    /* Claim the pause atomically: server-side step handlers gate on
     * MonitorBusy in waitForStepReady, so once we set this flag no
     * `next` can race ahead and resume the worker before we've
     * sent the stopped event for the current pause. Set/cleared
     * strictly under G.Mu — see the Shared::MonitorBusy comment. */
    if (Paused && !Exited) G.MonitorBusy = true;
    pthread_mutex_unlock(&G.Mu);

    if (Paused && !Exited) {
      int32_t Fid = 0, Ln = 0;
      matlab_dbg_get_pause(&Fid, &Ln);
      int BpIdx = matlab_dbg_get_pause_bp();
      const char *Cond = nullptr, *Log = nullptr;
      int64_t CondLen = 0, LogLen = 0;
      int CondDisabled = 0;
      if (BpIdx >= 0)
        matlab_dbg_breakpoint_meta(BpIdx, &Cond, &CondLen, &Log, &LogLen,
                                    &CondDisabled);

      bool Suppress = false;

      // Phase 8b: silently re-step while the step-over remains
      // inside the same .mflow block. Only fires when:
      //   - The runtime says this is a non-bp pause (BpIdx < 0).
      //   - The user issued `next` (G.StepOverBlockId is set).
      //   - The new (file_id, line) maps to the same block id we
      //     started the step from.
      // Re-issuing STEP_OVER walks one more statement; we loop via
      // the outer `while (true)` because resume + cv broadcast
      // wakes the worker, which fires another hook and re-enters
      // this same monitor body.
      if (BpIdx < 0 && !G.StepOverBlockId.empty()
          && !G.BlockByLine.empty()) {
        int64_t Key = (static_cast<int64_t>((uint32_t)Fid) << 32) |
                      static_cast<int64_t>((uint32_t)Ln);
        auto It = G.BlockByLine.find(Key);
        if (It != G.BlockByLine.end() && It->second == G.StepOverBlockId) {
          if (Debug) {
            std::fprintf(stderr,
                "[monitor] same-block step (block=%s line=%d) — "
                "auto-stepping\n", G.StepOverBlockId.c_str(), (int)Ln);
            std::fflush(stderr);
          }
          matlab_dbg_resume(STEP_OVER);
          pthread_mutex_lock(&G.Mu);
          /* No stopped event will fire for this auto-stepped pause,
           * so clear MonitorBusy here — leaving it set would keep
           * the server's waitForStepReady blocked forever. */
          G.MonitorBusy = false;
          pthread_cond_broadcast(&G.Cv);
          pthread_mutex_unlock(&G.Mu);
          continue;
        }
      }

      if (Log && LogLen > 0) {
        /* Log point: emit an output event with the interpolated
         * template, never tell the IDE we stopped. The worker is
         * blocked inside matlab_dbg_hook; we resume it ourselves.
         *
         * Bridge function-frame locals so `{a}` resolves to the
         * function's parameter when the bp fires inside a function
         * body — same machinery as the conditional-bp evaluator. */
        std::string Tmpl(Log, (size_t)LogLen);
        FrameBridge FB;
        FB.stamp(innermostFunctionFrameIdx());
        std::string Msg = interpolateLogMessage(Tmpl);
        FB.restore();
        Msg += "\n";
        sendEvent("output", Object{{"category", "console"},
                                    {"output", Msg}});
        Suppress = true;
      } else if (CondDisabled) {
        /* Eval failed earlier — silently suppress without re-trying
         * the JIT pipeline. The diagnostic was already printed. */
        Suppress = true;
      } else if (Cond && CondLen > 0) {
        /* Conditional breakpoint: evaluate against the workspace.
         * eval == 0 → user expression was false; suppress the stop.
         * eval == -1 → eval failed; mark the condition disabled so
         * we don't keep paying the JIT cost for a broken expr. */
        std::string Expr(Cond, (size_t)CondLen);
        int Result = evalConditionInWorkspace(Expr);
        if (Result == -1) {
          std::fprintf(stderr,
                       "[matlabc -dap] condition disabled at line %d: %s\n",
                       (int)Ln, Expr.c_str());
          matlab_dbg_disable_condition(BpIdx);
          Suppress = true;
        } else if (Result == 0) {
          Suppress = true;
        }
      }

      if (Suppress) {
        if (Debug) {
          std::fprintf(stderr, "[monitor] suppressed pause at %d\n", Ln);
          std::fflush(stderr);
        }
        matlab_dbg_resume(CONTINUE);
        pthread_mutex_lock(&G.Mu);
        /* Suppressed pauses (logpoint, conditional-bp false) don't
         * emit a stopped event — clear MonitorBusy here so the
         * server's step waiters aren't held up indefinitely. */
        G.MonitorBusy = false;
        pthread_cond_broadcast(&G.Cv);
        pthread_mutex_unlock(&G.Mu);
      } else {
        if (Debug) {
          std::fprintf(stderr, "[monitor] stopped at %d\n", Ln);
          std::fflush(stderr);
        }
        /* Snapshot the current resume generation BEFORE sending the
         * stopped event. The continue/step handlers bump it under
         * G.Mu, so we exit the inner wait the moment the client has
         * acted — even if the worker has already re-paused at the
         * next breakpoint by then. Without this, a paused→resume→
         * paused sequence inside the wait window would mask the
         * client's resume and leave us blocked forever. */
        pthread_mutex_lock(&G.Mu);
        uint64_t MyGen = G.ResumeGen;
        pthread_mutex_unlock(&G.Mu);
        /* The runtime sets cur_bp_idx >= 0 only when a breakpoint
         * matched; step / pause comes through with BpIdx == -1.
         * Surface that as the DAP-standard "step" reason so the IDE
         * renders the right icon and doesn't imply the user has an
         * unexpected breakpoint sitting on the current line. */
        /* Stop reason precedence:
         *   - bp matched (BpIdx >= 0)             -> "breakpoint"
         *   - data-bp tripped (watchpoint write)  -> "data breakpoint"
         *   - keyboard() call from user code      -> "entry"
         *   - everything else (step / pause)      -> "step"
         * The runtime exposes per-source flags (paused_from_watch,
         * paused_from_keyboard); reading them here is race-free
         * because the worker is currently parked on the condvar. */
        const char *Reason;
        bool FromWatch = matlab_dbg_was_paused_from_watch();
        if (BpIdx >= 0) Reason = "breakpoint";
        else if (FromWatch) Reason = "data breakpoint";
        else if (matlab_dbg_was_paused_from_keyboard()) Reason = "entry";
        else Reason = "step";
        /* threadId reports the runtime-assigned id of the worker
         * that hit the pause. For the main script (no parfor),
         * this is always 1; for parfor bodies, each spawned
         * pthread gets its own sequential id (2, 3, ...) on first
         * hook fire. Falls back to 1 pre-registration so old
         * tests / clients keep working. */
        int32_t StopThreadId = matlab_dbg_paused_thread_id();
        if (StopThreadId == 0) StopThreadId = 1;
        Object Body{
          {"reason", Reason},
          {"threadId", (int64_t)StopThreadId},
          {"allThreadsStopped", true},
          {"line", (int64_t)Ln},
        };
        /* hitBreakpointIds: when the pause was triggered by a
         * matched breakpoint (BpIdx >= 0), surface the bp's id (the
         * same id we returned in setBreakpoints / setFunctionBreakpoints)
         * so the IDE can highlight the row that fired. Single-element
         * array because our hook stops on the first match — we don't
         * coalesce same-line bps. Data breakpoints use the same
         * field with the watchpoint's id (returned in
         * setDataBreakpoints) so the IDE highlights the watched
         * variable's row. */
        if (BpIdx >= 0) {
          Array Ids;
          Ids.push_back(encodeBpId(Fid, Ln));
          Body["hitBreakpointIds"] = std::move(Ids);
        } else if (FromWatch) {
          int32_t WId = matlab_dbg_last_watchpoint_id();
          if (WId != 0) {
            Array Ids;
            Ids.push_back((int64_t)WId);
            Body["hitBreakpointIds"] = std::move(Ids);
          }
        }
#ifdef MATLAB_LLVM_WITH_PLOT
        /* Flush any figure the JIT created between the previous
         * resume and this re-pause — a script that does plot(x,y)
         * between two breakpoints should land the figure in the
         * Plots panel at the second stop, not at thread exit. */
        matlab_ide_emit_all_figures();
#endif
        sendEvent("stopped", Value(std::move(Body)));
        pthread_mutex_lock(&G.Mu);
        /* Mark the stop as delivered so any step handler that's
         * blocked in waitForStepReady wakes up and proceeds. The
         * counter is monotonic and cumulative across all stop
         * sources (step / breakpoint / pause). */
        G.StopsEmitted++;
        G.MonitorBusy = false;
        pthread_cond_broadcast(&G.Cv);
        while (G.ResumeGen == MyGen && !G.WorkerExited)
          pthread_cond_wait(&G.Cv, &G.Mu);
        pthread_mutex_unlock(&G.Mu);
        if (Debug) {
          std::fprintf(stderr, "[monitor] resumed\n");
          std::fflush(stderr);
        }
      }
    }

    if (Exited) break;
  }
  /* `thread` event with reason="exited" mirrors the "started" event we
   * fire on configurationDone — keeps adapters that track the live
   * thread set in sync. */
  sendEvent("thread",
            Object{{"reason", "exited"}, {"threadId", (int64_t)1}});
  sendEvent("exited", Object{{"exitCode", 0}});
  sendEvent("terminated");
  return nullptr;
}

/* Reader thread: forwards debuggee stdout to DAP `output` events. */
void *stdoutReaderMain(void *) {
  char Buf[4096];
  while (true) {
    ssize_t n = read(DebuggeeOutFd, Buf, sizeof Buf);
    if (n <= 0) break;
    Object Body{
      {"category", "stdout"},
      {"output", std::string(Buf, (size_t)n)},
    };
    sendEvent("output", Value(std::move(Body)));
  }
  return nullptr;
}

/* Same as stdoutReaderMain for stderr. Diagnostics from the REPL
 * JIT (parser / type / lowering errors) and the error()-traceback
 * printer write here; the IDE's debug console renders them with the
 * `stderr` category styling so users can tell error output from
 * normal program output at a glance.
 *
 * Tee'd to OriginalStderrFd: unlike stdout (which the JIT'd disp/
 * fprintf "owns" exclusively for DAP forwarding), stderr is what
 * spawning callers — including our test harness — read for failure
 * context. Keeping the original stream alive preserves
 * `subprocess.stderr` capture and CI logs while still forwarding
 * the same bytes to the IDE as `output` events. */
void *stderrReaderMain(void *) {
  char Buf[4096];
  while (true) {
    ssize_t n = read(DebuggeeErrFd, Buf, sizeof Buf);
    if (n <= 0) break;
    if (OriginalStderrFd >= 0)
      (void)!write(OriginalStderrFd, Buf, (size_t)n);
    Object Body{
      {"category", "stderr"},
      {"output", std::string(Buf, (size_t)n)},
    };
    sendEvent("output", Value(std::move(Body)));
  }
  return nullptr;
}

/* A separate signalling path so the monitor wakes when the worker
 * goes from "running" to "paused". We set paused=1 inside the hook
 * under the runtime's mutex; here we poll via matlab_dbg_is_paused
 * inside our own mutex so the condvar wakeup is well-defined.
 *
 * This is a lightweight thread that just periodically checks. We
 * could instead extend the runtime API to signal G.Cv directly, but
 * that would couple the runtime to the DAP server. A 20ms poll is
 * below the threshold of perceptible latency for human-driven
 * stepping and keeps the runtime decoupled. */
void *pauseWatcherMain(void *) {
  struct timespec ts = {0, 20 * 1000 * 1000};
  while (true) {
    pthread_mutex_lock(&G.Mu);
    bool Exited = G.WorkerExited;
    /* Unconditional broadcast: the monitor's inner "wait for resume"
     * loop also needs a wakeup on the paused=1 -> paused=0 transition,
     * not just on 0 -> 1. Broadcasting every tick keeps both loops
     * responsive without coupling the runtime to G.Cv. */
    pthread_cond_broadcast(&G.Cv);
    pthread_mutex_unlock(&G.Mu);
    if (Exited) break;
    nanosleep(&ts, nullptr);
  }
  return nullptr;
}

/* Object-ref registry. Each class instance the IDE asks to expand
 * gets a small integer handle in this vector; we hand the handle
 * back as the row's variablesReference so the next `variables`
 * request can find the matlab_obj* again. The registry is process-
 * lifetime — entries pile up across pauses but the obj pointers stay
 * valid as long as their owning slot is alive (script-frame for the
 * REPL workspace, function-frame for per-frame Locals). The base is
 * picked above the existing 1 / 1000+ ranges so the encodings don't
 * collide. */
constexpr int64_t ObjRefBase = 100000;
std::vector<void *> ObjRefs;

int64_t registerObjRef(void *obj) {
  if (!obj) return 0;
  ObjRefs.push_back(obj);
  return ObjRefBase + (int64_t)(ObjRefs.size() - 1);
}

void *lookupObjRef(int64_t ref) {
  if (ref < ObjRefBase || ref >= ObjRefBase + 100000) return nullptr;
  size_t idx = (size_t)(ref - ObjRefBase);
  if (idx >= ObjRefs.size()) return nullptr;
  return ObjRefs[idx];
}

/* Matrix-ref registry. Mirror of ObjRefs but for matlab_mat *
 * pointers — every matrix row in LOCALS / WATCH / property children
 * gets a handle here so the IDE can drill into the cells via the
 * standard DAP `variables` request. The base sits above ObjRefs's
 * window so a stray ref doesn't accidentally route to the wrong
 * registry. As with ObjRefs the matrix pointer is borrowed from the
 * owning slot (function-frame mini-ws or matlab_ws); the slot
 * outlives any client read because the runtime is paused while the
 * DAP server is responding. */
/* DAP `variables` rows that carry a `variablesReference` can also
 * advertise the *kind* of children to expect via `indexedVariables`
 * (for numeric grids — matrices) and `namedVariables` (for property
 * sets — class instances). Matrix-viewer / variable-inspector panels
 * use these counts to lay out a grid widget or a property table
 * without first paging through children. Both fields are optional
 * per the spec — we set them when we know the count cheaply. */
/* Encode a (file_id, line) pair as a stable DAP breakpoint id. The
 * IDE round-trips ids opaquely (setBreakpoints → stopped's
 * hitBreakpointIds), so any deterministic mapping that's unique
 * across the session works. file_id * 1e6 + line keeps function and
 * line breakpoints in the same id space without a separate registry,
 * and is reversible for debugging. Caps the line number at <1M. */
constexpr int64_t BpIdLineWidth = 1000000;
int64_t encodeBpId(int32_t file_id, int32_t line) {
  return (int64_t)file_id * BpIdLineWidth + (int64_t)line;
}

int64_t matIndexedCount(struct matlab_mat *Mraw) {
  if (!Mraw) return 0;
  int32_t Kind = matlab_dbg_mat_kind(Mraw);
  if (Kind == 2) {
    auto *M = (struct matlab_mat_c *)Mraw;
    return matlab_dbg_mat_c_rows(M) * matlab_dbg_mat_c_cols(M);
  }
  if (Kind == 3) {
    auto *M = (struct matlab_mat3 *)Mraw;
    return matlab_dbg_mat3_rows(M) * matlab_dbg_mat3_cols(M)
         * matlab_dbg_mat3_depth(M);
  }
  if (Kind == 4) return matlab_dbg_matN_numel(Mraw);
  int64_t r = matlab_dbg_mat_rows(Mraw);
  int64_t c = matlab_dbg_mat_cols(Mraw);
  if (r <= 0 || c <= 0) return 0;
  return r * c;
}

/* Multi-cell test used to gate `variablesReference` assignment.
 * A 1x1 real matrix unboxes to its scalar value in the parent row
 * and gets no expansion; same logic for a 1x1 complex (rendered
 * as "re+im*i"). 3-D arrays are always drillable — there's no
 * scalar shape they unbox to. Centralised here so every site that
 * decides whether to call registerMatRef agrees on the rule. */
bool matIsMultiCell(struct matlab_mat *Mraw) {
  if (!Mraw) return false;
  int32_t Kind = matlab_dbg_mat_kind(Mraw);
  if (Kind == 2) {
    auto *M = (struct matlab_mat_c *)Mraw;
    return matlab_dbg_mat_c_rows(M) != 1 || matlab_dbg_mat_c_cols(M) != 1;
  }
  if (Kind == 3) return true;
  if (Kind == 4) return matlab_dbg_matN_numel(Mraw) > 1;
  return matlab_dbg_mat_rows(Mraw) != 1 || matlab_dbg_mat_cols(Mraw) != 1;
}

/* Total namedVariables for a class instance — properties (from the
 * obj's struct prefix) plus methods walked across the inheritance
 * chain via G.ClassParent, with overrides de-duped by name (so
 * `Savings.deposit` shadowing `Account.deposit` counts as one row).
 * IDEs use namedVariables as a sizing hint for the property pane;
 * undercounting makes the pane stop scrolling before the last
 * method row. */
int64_t objNamedCount(void *obj) {
  if (!obj) return 0;
  int64_t Total = matlab_dbg_obj_field_count(obj);
  int32_t cid = matlab_dbg_obj_class_id_of(obj);
  int64_t cnLen = 0;
  const char *cn = matlab_dbg_class_name(cid, &cnLen);
  if (!cn || cnLen <= 0) return Total;
  std::string ClassName(cn, (size_t)cnLen);
  std::unordered_set<std::string> Seen;
  for (std::string Cur = ClassName; !Cur.empty();) {
    auto MIt = G.ClassMethods.find(Cur);
    if (MIt != G.ClassMethods.end()) {
      for (const Shared::MethodEntry &ME : MIt->second)
        if (Seen.insert(ME.Name).second) ++Total;
    }
    auto PIt = G.ClassParent.find(Cur);
    if (PIt == G.ClassParent.end()) break;
    Cur = PIt->second;
  }
  return Total;
}

constexpr int64_t MatRefBase = 200000;
std::vector<void *> MatRefs;

int64_t registerMatRef(void *mat) {
  if (!mat) return 0;
  MatRefs.push_back(mat);
  return MatRefBase + (int64_t)(MatRefs.size() - 1);
}

void *lookupMatRef(int64_t ref) {
  if (ref < MatRefBase) return nullptr;
  size_t idx = (size_t)(ref - MatRefBase);
  if (idx >= MatRefs.size()) return nullptr;
  return MatRefs[idx];
}

/* Phase 5.3 — table variablesReference registry. Kept distinct from
 * MatRefs because the drill-in handler must dispatch on table-vs-mat:
 * walking a table's columns goes through matlab_table_column_* and
 * the per-column formatter, walking a matrix's cells goes through
 * matlab_dbg_mat_get. Mixing the two registries would let the existing
 * `if (Ref >= MatRefBase)` branch grab a table pointer and crash
 * (matlab_dbg_mat_get reads off the head of a matlab_table_s, which
 * is the same crash the original bug surfaced).
 *
 * Window layout (existing constants in this TU):
 *   1                       legacy "script Locals" alias
 *   [1000, 50000)           frame ids (DAP frame_id + 1000)
 *   [TableRefBase, ObjRefBase) = [50000, 100000)   table refs
 *   [ObjRefBase, MatRefBase)   = [100000, 200000)  class-instance refs
 *   [MatRefBase, ...)          = [200000, ...)     matrix refs
 *
 * 50000 slots is far more than any real session needs but keeps the
 * encoding stable. The variables-handler dispatch inserts the table
 * branch before the ObjRefBase check so a table ref can't fall through
 * into the ObjRef path. */
constexpr int64_t TableRefBase = 50000;
std::vector<matlab_table *> TableRefs;

int64_t registerTableRef(matlab_table *t) {
  if (!t) return 0;
  TableRefs.push_back(t);
  return TableRefBase + (int64_t)(TableRefs.size() - 1);
}

matlab_table *lookupTableRef(int64_t ref) {
  if (ref < TableRefBase || ref >= ObjRefBase) return nullptr;
  size_t idx = (size_t)(ref - TableRefBase);
  if (idx >= TableRefs.size()) return nullptr;
  return TableRefs[idx];
}

/* Memory-region registry for the DAP `readMemory` / `writeMemory`
 * requests. Whenever we hand out a memoryReference on a matrix
 * variable row, we also record (data_ptr, byte_count) here so the
 * read/write handler can validate the request against a known
 * buffer instead of trusting the IDE's hex string blindly. The
 * registry is keyed by the data pointer itself — duplicate entries
 * just refresh the byte_count.
 *
 * Without this gate, `readMemory({memoryReference: "0xdeadbeef",
 * count: 1MB})` would happily walk out-of-bounds memory; a stray
 * IDE request from a paused-but-stale debug session is the
 * realistic failure mode. */
struct MemRegion { void *Ptr; int64_t Bytes; };
std::vector<MemRegion> MemRegions;

void registerMemRegion(void *Ptr, int64_t Bytes) {
  if (!Ptr || Bytes <= 0) return;
  for (auto &R : MemRegions) {
    if (R.Ptr == Ptr) { R.Bytes = Bytes; return; }
  }
  MemRegions.push_back({Ptr, Bytes});
}

const MemRegion *lookupMemRegion(void *Ptr) {
  for (const auto &R : MemRegions)
    if (R.Ptr == Ptr) return &R;
  return nullptr;
}

/* Base64 encode/decode for the DAP readMemory/writeMemory payload
 * (the `data` field carries raw bytes as base64 per spec). Tiny
 * standalone implementation — pulling in a third-party codec for
 * <30 lines of code wasn't worth the dependency. */
std::string b64Encode(const uint8_t *Data, size_t N) {
  static const char Tbl[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string Out;
  Out.reserve(((N + 2) / 3) * 4);
  for (size_t i = 0; i < N; i += 3) {
    uint32_t v = (uint32_t)Data[i] << 16;
    if (i + 1 < N) v |= (uint32_t)Data[i + 1] << 8;
    if (i + 2 < N) v |= (uint32_t)Data[i + 2];
    Out.push_back(Tbl[(v >> 18) & 0x3F]);
    Out.push_back(Tbl[(v >> 12) & 0x3F]);
    Out.push_back(i + 1 < N ? Tbl[(v >> 6) & 0x3F] : '=');
    Out.push_back(i + 2 < N ? Tbl[v & 0x3F]      : '=');
  }
  return Out;
}

std::vector<uint8_t> b64Decode(const std::string &S) {
  static int8_t Inv[256] = {0};
  static bool Init = false;
  if (!Init) {
    for (int i = 0; i < 256; ++i) Inv[i] = -1;
    const char *Tbl =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; i < 64; ++i) Inv[(uint8_t)Tbl[i]] = (int8_t)i;
    Init = true;
  }
  std::vector<uint8_t> Out;
  Out.reserve((S.size() / 4) * 3);
  uint32_t v = 0;
  int bits = 0;
  for (char c : S) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
    int8_t d = Inv[(uint8_t)c];
    if (d < 0) continue;
    v = (v << 6) | (uint32_t)d;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      Out.push_back((uint8_t)((v >> bits) & 0xFF));
    }
  }
  return Out;
}

/* Side-effect helper: when a variable row is about to surface a
 * matrix, stash the matrix's data buffer in MemRegions and return
 * the formatted memoryReference string. The IDE's `readMemory` /
 * `writeMemory` requests later decode the hex back to a pointer
 * and we re-validate against MemRegions to bound the I/O.
 *
 * Returns an empty string for matrices we can't expose (1x1 real
 * — already unboxed; complex — has separate re/im buffers). The
 * caller skips the memoryReference field in those cases so the
 * row stays clean. */
std::string registerMatMemRef(void *Mraw) {
  void *Data = matlab_dbg_mat_data_ptr(Mraw);
  int64_t Bytes = matlab_dbg_mat_data_bytes(Mraw);
  if (!Data || Bytes <= 0) return std::string();
  registerMemRegion(Data, Bytes);
  /* matlab_dbg_mat_data_ptr already filtered by kind; format and
   * return. */
  char Buf[32];
  snprintf(Buf, sizeof Buf, "0x%llx",
           (unsigned long long)(uintptr_t)Data);
  return Buf;
}

/* Lazily-built disassembler holder for the DAP `disassemble`
 * request. Construction is non-trivial (target lookup, MCInfo +
 * MCRegisterInfo + MCInstrInfo + MCSubtargetInfo + AsmInfo +
 * MCContext + MCDisassembler + MCInstPrinter all have to be
 * created in dependency order), so we cache the whole stack on
 * first request and reuse it.
 *
 * Single-target: the host triple is whatever the JIT is running
 * on. We don't try to support cross-disassembly because the IDE
 * is always asking about the in-process JIT'd code. */
struct DisasmHolder {
  bool Inited = false;
  bool Available = false;
  std::string ErrMsg;
  std::unique_ptr<llvm::MCRegisterInfo> MRI;
  std::unique_ptr<llvm::MCAsmInfo> MAI;
  std::unique_ptr<llvm::MCInstrInfo> MII;
  std::unique_ptr<llvm::MCSubtargetInfo> STI;
  std::unique_ptr<llvm::MCContext> Ctx;
  std::unique_ptr<llvm::MCDisassembler> Dis;
  std::unique_ptr<llvm::MCInstPrinter> Printer;
};
DisasmHolder &disasmHolder() {
  static DisasmHolder H;
  if (H.Inited) return H;
  H.Inited = true;
  /* Lazy init — see runDap() comment. Idempotent. */
  llvm::InitializeNativeTargetDisassembler();
  std::string Triple = llvm::sys::getDefaultTargetTriple();
  std::string LookupErr;
  const llvm::Target *T = llvm::TargetRegistry::lookupTarget(Triple, LookupErr);
  if (!T) {
    H.ErrMsg = "MCTarget lookup failed for " + Triple + ": " + LookupErr;
    return H;
  }
  H.MRI.reset(T->createMCRegInfo(Triple));
  if (!H.MRI) { H.ErrMsg = "createMCRegInfo failed"; return H; }
  llvm::MCTargetOptions MCOpts;
  H.MAI.reset(T->createMCAsmInfo(*H.MRI, Triple, MCOpts));
  if (!H.MAI) { H.ErrMsg = "createMCAsmInfo failed"; return H; }
  H.MII.reset(T->createMCInstrInfo());
  if (!H.MII) { H.ErrMsg = "createMCInstrInfo failed"; return H; }
  H.STI.reset(T->createMCSubtargetInfo(
      Triple, llvm::sys::getHostCPUName(), ""));
  if (!H.STI) { H.ErrMsg = "createMCSubtargetInfo failed"; return H; }
  H.Ctx.reset(new llvm::MCContext(
      llvm::Triple(Triple), H.MAI.get(), H.MRI.get(), H.STI.get()));
  H.Dis.reset(T->createMCDisassembler(*H.STI, *H.Ctx));
  if (!H.Dis) { H.ErrMsg = "createMCDisassembler failed"; return H; }
  /* Asm-printer flavour 0 is the target's default; AT&T-vs-Intel
   * is x86-only and the current users don't care. */
  H.Printer.reset(T->createMCInstPrinter(
      llvm::Triple(Triple), /*SyntaxVariant=*/0,
      *H.MAI, *H.MII, *H.MRI));
  if (!H.Printer) { H.ErrMsg = "createMCInstPrinter failed"; return H; }
  H.Available = true;
  return H;
}

/* Inverse of the inline pointer-to-hex formatting in
 * registerMatMemRef. Returns nullptr on malformed input. */
void *parseMemRef(const std::string &S) {
  if (S.size() < 3) return nullptr;
  size_t off = 0;
  if (S[0] == '0' && (S[1] == 'x' || S[1] == 'X')) off = 2;
  uintptr_t V = 0;
  for (size_t i = off; i < S.size(); ++i) {
    char c = S[i];
    int d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
    else return nullptr;
    V = (V << 4) | (uintptr_t)d;
  }
  return (void *)V;
}

/* Format a single matrix row for display alongside its
 * variablesReference. 1x1 matrices unbox to the scalar (matches
 * matlab_struct_get_f64 and what users want to see in a counter
 * variable); everything else gets the `RxC double` shape summary so
 * the disclosure arrow in the IDE has a meaningful preview before
 * it's clicked. */
/* Render the shape header that lands in the parent row's `value`
 * column — `2x3 double`, `2x3 complex`, `2x3x4 double`. 1x1 real
 * matrices unbox to the scalar value (matches what the watch box
 * naturally expects); 1x1 complex unbox to "re+im*i" form. */
std::string formatMatShape(struct matlab_mat *Mraw) {
  if (!Mraw) return "[]";
  int32_t Kind = matlab_dbg_mat_kind(Mraw);
  if (Kind == 2) {
    auto *M = (struct matlab_mat_c *)Mraw;
    int64_t R = matlab_dbg_mat_c_rows(M);
    int64_t C = matlab_dbg_mat_c_cols(M);
    if (R == 1 && C == 1) {
      double re = matlab_dbg_mat_c_re(M, 1, 1);
      double im = matlab_dbg_mat_c_im(M, 1, 1);
      char Buf[64];
      if (im >= 0)
        snprintf(Buf, sizeof Buf, "%g+%gi", re, im);
      else
        snprintf(Buf, sizeof Buf, "%g-%gi", re, -im);
      return Buf;
    }
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%lldx%lld complex",
             (long long)R, (long long)C);
    return Buf;
  }
  if (Kind == 3) {
    auto *M = (struct matlab_mat3 *)Mraw;
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%lldx%lldx%lld double",
             (long long)matlab_dbg_mat3_rows(M),
             (long long)matlab_dbg_mat3_cols(M),
             (long long)matlab_dbg_mat3_depth(M));
    return Buf;
  }
  if (Kind == 4) {
    /* Tier C: render the dims tuple as "AxBxCx... double".  Most matN
     * values stay within 4-6 axes; cap at 8 to bound the buffer. */
    int32_t nd = matlab_dbg_matN_ndims(Mraw);
    if (nd <= 0) return "[]";
    if (nd > 8) nd = 8;
    char Buf[128];
    int n = 0;
    for (int32_t k = 1; k <= nd; ++k) {
      int64_t d = matlab_dbg_matN_dim(Mraw, k);
      n += snprintf(Buf + n, sizeof(Buf) - (size_t)n,
                    k == 1 ? "%lld" : "x%lld", (long long)d);
      if (n >= (int)sizeof(Buf) - 12) break;
    }
    snprintf(Buf + n, sizeof(Buf) - (size_t)n, " double");
    return Buf;
  }
  int64_t R = matlab_dbg_mat_rows(Mraw);
  int64_t C = matlab_dbg_mat_cols(Mraw);
  if (R == 1 && C == 1) {
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%g", matlab_dbg_mat_get(Mraw, 1, 1));
    return Buf;
  }
  char Buf[64];
  snprintf(Buf, sizeof Buf, "%lldx%lld double",
           (long long)R, (long long)C);
  return Buf;
}

/* Cap matrix expansion at 256 children so a watchful IDE doesn't
 * pull a 1000x1000 grid in one shot. The trailing "..." row makes
 * the truncation visible. Children layout:
 *   - real 2-D 1xN row vector  -> linear "(j)" labels.
 *   - real 2-D Mx1 col vector  -> linear "(i)" labels.
 *   - real 2-D MxN matrix      -> "(i,j)" labels in row-major order.
 *   - complex MxN              -> "(i,j)" labels with value
 *                                  rendered as "re+im*i" so a single
 *                                  child row carries both parts.
 *   - 3-D MxNxP                -> "(i,j,k)" labels, slice-major
 *                                  iteration so cells with the same
 *                                  k group together.
 * 1x1 matrices have no children — the parent row already shows the
 * scalar (or `re+im*i`) via formatMatShape. */
constexpr size_t MatExpandCap = 256;

void appendMatChildren(Array &Vs, struct matlab_mat *Mraw) {
  if (!Mraw) return;
  int32_t Kind = matlab_dbg_mat_kind(Mraw);

  size_t emitted = 0;
  auto emitTruncated = [&] {
    Vs.push_back(Object{
      {"name", std::string("…")},
      {"value", std::string("(truncated)")},
      {"variablesReference", (int64_t)0},
    });
  };
  auto emit = [&](std::string label, std::string val,
                  const char *Type) {
    Vs.push_back(Object{
      {"name", std::move(label)},
      {"value", std::move(val)},
      {"type", std::string(Type)},
      {"variablesReference", (int64_t)0},
    });
    ++emitted;
  };

  if (Kind == 2) {
    auto *M = (struct matlab_mat_c *)Mraw;
    int64_t R = matlab_dbg_mat_c_rows(M);
    int64_t C = matlab_dbg_mat_c_cols(M);
    if (R == 1 && C == 1) return;
    for (int64_t i = 1; i <= R; ++i) {
      for (int64_t j = 1; j <= C; ++j) {
        if (emitted >= MatExpandCap) { emitTruncated(); return; }
        char LabelBuf[64];
        snprintf(LabelBuf, sizeof LabelBuf, "(%lld,%lld)",
                 (long long)i, (long long)j);
        double re = matlab_dbg_mat_c_re(M, i, j);
        double im = matlab_dbg_mat_c_im(M, i, j);
        char ValBuf[80];
        if (im >= 0)
          snprintf(ValBuf, sizeof ValBuf, "%g+%gi", re, im);
        else
          snprintf(ValBuf, sizeof ValBuf, "%g-%gi", re, -im);
        emit(LabelBuf, ValBuf, "complex");
      }
    }
    return;
  }

  if (Kind == 3) {
    auto *M = (struct matlab_mat3 *)Mraw;
    int64_t R = matlab_dbg_mat3_rows(M);
    int64_t C = matlab_dbg_mat3_cols(M);
    int64_t D = matlab_dbg_mat3_depth(M);
    /* Slice-major: outermost loop is k so all (i,j) of slice k
     * appear contiguously. Matches how MATLAB's whos / disp render
     * 3-D arrays page by page. */
    for (int64_t k = 1; k <= D; ++k) {
      for (int64_t i = 1; i <= R; ++i) {
        for (int64_t j = 1; j <= C; ++j) {
          if (emitted >= MatExpandCap) { emitTruncated(); return; }
          char LabelBuf[64];
          snprintf(LabelBuf, sizeof LabelBuf, "(%lld,%lld,%lld)",
                   (long long)i, (long long)j, (long long)k);
          char ValBuf[64];
          snprintf(ValBuf, sizeof ValBuf, "%g",
                   matlab_dbg_mat3_get(M, i, j, k));
          emit(LabelBuf, ValBuf, "double");
        }
      }
    }
    return;
  }

  if (Kind == 4) {
    /* matN drill: walk the flat buffer in row-major-extended order,
     * de-linearising each linear index back into the (i1, i2, ..., in)
     * tuple via the dims tuple read off via matlab_dbg_matN_dim. */
    int32_t nd = matlab_dbg_matN_ndims(Mraw);
    if (nd <= 0) return;
    if (nd > 8) nd = 8;
    int64_t dims[8] = {0};
    int64_t total = 1;
    for (int32_t k = 0; k < nd; ++k) {
      dims[k] = matlab_dbg_matN_dim(Mraw, k + 1);
      total *= dims[k];
    }
    int64_t idx[8] = {0};
    for (int64_t lin = 0; lin < total; ++lin) {
      if (emitted >= MatExpandCap) { emitTruncated(); return; }
      char LabelBuf[96];
      int n = 0;
      n += snprintf(LabelBuf + n, sizeof(LabelBuf) - (size_t)n, "(");
      for (int32_t k = 0; k < nd; ++k) {
        n += snprintf(LabelBuf + n, sizeof(LabelBuf) - (size_t)n,
                      k == 0 ? "%lld" : ",%lld",
                      (long long)(idx[k] + 1));
      }
      n += snprintf(LabelBuf + n, sizeof(LabelBuf) - (size_t)n, ")");
      char ValBuf[64];
      snprintf(ValBuf, sizeof ValBuf, "%g",
               matlab_dbg_matN_get_lin(Mraw, lin));
      emit(LabelBuf, ValBuf, "double");
      /* Advance idx — rightmost varies fastest, mirroring storage order. */
      for (int32_t k = nd - 1; k >= 0; --k) {
        if (++idx[k] < dims[k]) break;
        idx[k] = 0;
      }
    }
    return;
  }

  /* Real 2-D matlab_mat path. */
  int64_t R = matlab_dbg_mat_rows(Mraw);
  int64_t C = matlab_dbg_mat_cols(Mraw);
  if (R == 1 && C == 1) return;
  bool RowVec = (R == 1);
  bool ColVec = (C == 1);
  for (int64_t i = 1; i <= R; ++i) {
    for (int64_t j = 1; j <= C; ++j) {
      if (emitted >= MatExpandCap) { emitTruncated(); return; }
      char LabelBuf[64];
      if (RowVec)      snprintf(LabelBuf, sizeof LabelBuf, "(%lld)", (long long)j);
      else if (ColVec) snprintf(LabelBuf, sizeof LabelBuf, "(%lld)", (long long)i);
      else             snprintf(LabelBuf, sizeof LabelBuf, "(%lld,%lld)",
                                  (long long)i, (long long)j);
      char ValBuf[64];
      snprintf(ValBuf, sizeof ValBuf, "%g",
               matlab_dbg_mat_get(Mraw, i, j));
      emit(LabelBuf, ValBuf, "double");
    }
  }
}

/* Render a class instance as `1x1 ClassName`, falling back to the
 * raw class_id when the registry hasn't been populated (DebugMode
 * off path; shouldn't happen for -dap launches but the runtime is
 * the source of truth so the formatter handles it gracefully). */
std::string formatObj(void *obj) {
  if (!obj) return "[]";
  int32_t cid = matlab_dbg_obj_class_id_of(obj);
  int64_t cnLen = 0;
  const char *cn = matlab_dbg_class_name(cid, &cnLen);
  std::string clsName;
  if (cn && cnLen > 0) clsName.assign(cn, (size_t)cnLen);
  else                  clsName = "<class " + std::to_string(cid) + ">";
  return std::string("1x1 ") + clsName;
}

/* DAP `type` field. Drives the IDE's TYPE column and hover tooltips.
 * MATLAB-style canonical names: scalar/matrix as `double`, classes as
 * the class name, "..." literals as `string`. The runtime kind enum
 * (0=f64, 1=mat, 2=obj, 3=string) maps directly. */
std::string typeForVar(int Kind, void *Ptr) {
  (void)Ptr;
  if (Kind == 0) return "double";
  if (Kind == 1) return "double";
  if (Kind == 2) {
    if (!Ptr) return "object";
    int32_t cid = matlab_dbg_obj_class_id_of(Ptr);
    int64_t cnLen = 0;
    const char *cn = matlab_dbg_class_name(cid, &cnLen);
    if (cn && cnLen > 0) return std::string(cn, (size_t)cnLen);
    return "object";
  }
  if (Kind == 3) return "string";
  if (Kind == 4) return "uint8";
  if (Kind == 5) return "int32";
  /* Phase 5 heterogeneous types. The IDE's Workspace pane keys the
   * TYPE column off these strings and the TABLE VIEWER tab routes
   * on `table` specifically. */
  if (Kind == 6) return "table";
  if (Kind == 7) return "sym";
  if (Kind == 8) return "sym matrix";
  if (Kind == 9)  return "categorical";
  if (Kind == 10) return "datetime";
  if (Kind == 11) return "duration";
  return "any";
}

/* Format a variable for the DAP `variables` response. Matrices get
 * a shape summary ("1x3 double") except 1x1 matrices, which unbox
 * to the scalar value — matches matlab_struct_get_f64's auto-unbox
 * and is also what users want to see in the watch panel for a
 * counter-style variable. Class instances render as `1x1 ClassName`;
 * the LOCALS handler attaches a variablesReference so the row
 * expands into one child per property. */
std::string formatVar(int Kind, int WsIdx) {
  if (Kind == 0) {
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%g", matlab_dbg_ws_f64(WsIdx));
    return Buf;
  }
  if (Kind == 1) {
    return formatMatShape((struct matlab_mat *)matlab_dbg_ws_ptr(WsIdx));
  }
  if (Kind == 4 || Kind == 5) {
    /* Phase 1.1.F: typed-int matrices show as "MxN uint8" / "MxN int32"
     * in the DAP variable view. The IDE row label keeps the lane visible
     * even though the values format identically (no "double" trailing). */
    void *Ptr = matlab_dbg_ws_ptr(WsIdx);
    int64_t R = 0, C = 0;
    if (Kind == 4) {
      auto *M = (matlab_mat_u8 *)Ptr;
      R = matlab_mat_u8_rows(M); C = matlab_mat_u8_cols(M);
    } else {
      auto *M = (matlab_mat_i32 *)Ptr;
      R = matlab_mat_i32_rows(M); C = matlab_mat_i32_cols(M);
    }
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%lldx%lld %s",
             (long long)R, (long long)C,
             Kind == 4 ? "uint8" : "int32");
    return Buf;
  }
  if (Kind == 2) {
    return formatObj(matlab_dbg_ws_ptr(WsIdx));
  }
  if (Kind == 3) {
    /* Render `"abc"` for string vars, with the actual bytes drawn
     * through the opaque accessor so this side never has to know
     * the matlab_string_s layout. Empty / NULL renders as `""`. */
    void *S = matlab_dbg_ws_ptr(WsIdx);
    int64_t SL = 0;
    const char *SD = matlab_string_get_data(S, &SL);
    std::string Out;
    Out.reserve((size_t)SL + 2);
    Out.push_back('"');
    Out.append(SD, (size_t)SL);
    Out.push_back('"');
    return Out;
  }
  if (Kind == 7) {
    /* Symbolic Math Toolbox — render via matlab_dbg_sym_str (a thin
     * wrapper around matlab_sym_str that returns SymPP's pretty form).
     * Returns "<unset sym>" if the runtime wasn't built with sym
     * support, mirroring the static-analysis-friendly fallback. */
    void *S = matlab_dbg_ws_ptr(WsIdx);
    int64_t SL = 0;
    const char *SD = matlab_dbg_sym_str(S, &SL);
    if (!SD || SL == 0) return "<unset sym>";
    return std::string(SD, (size_t)SL);
  }
  if (Kind == 8) {
    /* Phase 6.1 — symbolic matrix. Same shape as kind=7 but routed
     * through matlab_dbg_symmat_str (wraps matlab_symmat_str). */
    void *M = matlab_dbg_ws_ptr(WsIdx);
    int64_t SL = 0;
    const char *SD = matlab_dbg_symmat_str(M, &SL);
    if (!SD || SL == 0) return "<unset sym matrix>";
    return std::string(SD, (size_t)SL);
  }
  if (Kind == 6) {
    /* Phase 5.3 — table. Render the same `NxM table` summary MATLAB's
     * Workspace pane shows; the IDE's TABLE VIEWER tab paints the full
     * grid by reading inspectionRaw (captured disp(t) text). */
    auto *T = (matlab_table *)matlab_dbg_ws_ptr(WsIdx);
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%lldx%lld table",
             (long long)matlab_table_height(T),
             (long long)matlab_table_width(T));
    return Buf;
  }
  if (Kind == 9) {
    /* Phase 5.2 — categorical. 1-D vector; render `Nx1 categorical`. */
    auto *C = (matlab_categorical *)matlab_dbg_ws_ptr(WsIdx);
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%lldx1 categorical",
             (long long)matlab_categorical_length(C));
    return Buf;
  }
  if (Kind == 10) {
    /* Phase 5.1 — datetime. Scalar wrapper around seconds-since-epoch;
     * render as `1x1 datetime` (drilling further is a follow-up; for
     * now the row is a leaf). */
    return "1x1 datetime";
  }
  if (Kind == 11) {
    /* Phase 5.1 — duration. Show the value too — `<n> sec duration` —
     * because durations are tiny and inlining the numeric span is more
     * useful than `1x1 duration`. */
    auto *D = (matlab_duration *)matlab_dbg_ws_ptr(WsIdx);
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%g sec duration",
             matlab_duration_to_seconds(D));
    return Buf;
  }
  if (Kind == 13) {
    /* Function handle (kind=13). The stored value is a raw code pointer
     * — never dereference it as data; render a stable type label. */
    return "@function_handle";
  }
  return "<unknown>";
}

/* Handlers -----------------------------------------------------------*/

bool handleRequest(const Object &Msg) {
  auto Cmd = Msg.getString("command");
  const Value *SeqV = Msg.get("seq");
  int64_t ReqSeq = SeqV && SeqV->getAsInteger() ? *SeqV->getAsInteger() : 0;
  const Object *Args = Msg.getObject("arguments");
  Object Empty;
  if (!Args) Args = &Empty;
  if (!Cmd) return true;

  if (*Cmd == "initialize") {
    /* Phase 8c: read the IDE-supplied
     * `initializationOptions.blockPath` (string array). Forwarded
     * to `BuildOptions::BlockSearchPath` for `.mflow` programs at
     * compileProgram time. The IDE typically sets this from a
     * project setting (e.g. `${workspaceFolder}/blocks`). Stored
     * on `G` rather than passed through the launch handler since
     * `initializationOptions` arrives before `launch` and stays
     * stable across compileProgram restarts. */
    G.BlockPathFromIDE.clear();
    if (const Object *InitOpts =
            Args->getObject("initializationOptions")) {
      if (const Array *BP = InitOpts->getArray("blockPath")) {
        for (const auto &V : *BP) {
          if (auto S = V.getAsString())
            G.BlockPathFromIDE.emplace_back(*S);
        }
      }
    }
    /* Exception-breakpoint filters drive the IDE's "Pause on Errors"
     * / "Pause on Caught Errors" toggles. We expose a single filter
     * `error` that maps to MATLAB's error() flag — when enabled, the
     * runtime hook pauses the worker on the next statement after the
     * flag is set so the user can inspect the failing frame. */
    Array ExcFilters;
    ExcFilters.push_back(Object{
      {"filter", "error"},
      {"label", "MATLAB error()"},
      {"default", false},
      {"description", "Pause when matlab_set_error fires (uncaught error)."},
    });
    Object Caps{
      {"supportsConfigurationDoneRequest", true},
      /* Function breakpoints resolve a function name against the
       * compiled translation unit's function table and install a
       * line breakpoint at the function's first body line. */
      {"supportsFunctionBreakpoints", true},
      /* Conditional breakpoints + log points evaluate at script-frame
       * scope only (they read the workspace through matlab_ws_*).
       * Conditions inside user-function frames see <script>'s vars
       * but not the function's locals — Option B (per-function slot
       * tables) is the planned follow-up. */
      {"supportsConditionalBreakpoints", true},
      {"supportsHitConditionalBreakpoints", true},
      {"supportsLogPoints", true},
      /* setVariable + setExpression both reuse the REPL-JIT
       * assignment path: wrap as `<lhs> = (<rhs>);` and run through
       * runReplInput. Any MATLAB expression on the RHS works. */
      {"supportsSetVariable", true},
      {"supportsSetExpression", true},
      /* No state recorder yet, so reverse stepping and step-back are
       * advertised as unsupported. The handlers respond
       * success=false with a clear "requires recorder" message
       * rather than the unknown-request fallthrough. */
      {"supportsStepBack", true},
      {"supportsRestartFrame", false},
      {"supportsRestartRequest", true},
      {"supportsGotoTargetsRequest", false},
      {"supportsStepInTargetsRequest", true},
      {"supportsCompletionsRequest", true},
      {"supportsModulesRequest", true},
      {"supportsLoadedSourcesRequest", true},
      {"supportsTerminateRequest", true},
      {"supportsTerminateThreadsRequest", true},
      {"supportTerminateDebuggee", true},
      {"supportsExceptionInfoRequest", true},
      {"supportsBreakpointLocationsRequest", true},
      {"exceptionBreakpointFilters", std::move(ExcFilters)},
      /* Memory / disassembly / data-watchpoints / instruction
       * breakpoints all need infrastructure (JIT-frame addressing,
       * watchpoint instrumentation, native disassembly) that this
       * MVP doesn't ship. The corresponding handlers respond with
       * success=false + a precise reason. */
      {"supportsDataBreakpoints", true},
      {"supportsReadMemoryRequest", true},
      {"supportsWriteMemoryRequest", true},
      {"supportsDisassembleRequest", true},
      {"supportsInstructionBreakpoints", false},
      {"supportsSteppingGranularity", false},
      {"supportsCancelRequest", false},
      /* `evaluate` powers watch / hover / debug-console expressions.
       * v1 evaluates against the script-level workspace plus the
       * script frame's mini-ws; function-frame locals aren't visible
       * to the evaluator yet (the per-frame mini-ws is read by
       * `variables` but not bridged into runReplInput). */
      {"supportsEvaluateForHovers", true},
    };
    sendResponse(ReqSeq, *Cmd, true, Value(std::move(Caps)));
    sendEvent("initialized");
    return true;
  }

  if (*Cmd == "launch" || *Cmd == "attach") {
    /* `program` (launch) overrides the CLI-supplied path. */
    auto Prog = Args->getString("program");
    if (Prog && !Prog->empty()) G.ProgramPath = Prog->str();
    auto StopOnEntry = Args->getBoolean("stopOnEntry");
    bool SoE = StopOnEntry.value_or(false);

    if (G.ProgramPath.empty()) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("no program path supplied"));
      return true;
    }
    /* #77: make the program path absolute BEFORE the chdir below.  The
     * handler chdir's into the program's directory (so the script's
     * relative file reads behave like build_and_run.sh), but
     * G.ProgramPath was left relative — so compileProgram's loadFile then
     * looked for `<dir>/<dir>/<file>` from the new cwd, returned 0, and the
     * launch failed with the generic "failed to compile program".  This was
     * the silent root cause behind the DAP-only failures for every example
     * launched by a relative path. */
    {
      std::error_code AbsEC;
      std::filesystem::path AbsP =
          std::filesystem::absolute(G.ProgramPath, AbsEC);
      if (!AbsEC) G.ProgramPath = AbsP.lexically_normal().string();
    }
    /* Resolve the JIT'd program's working directory so relative
     * file reads (readtable("foo.csv") etc.) behave the same way
     * they do under `build_and_run.sh && ./out`, which exec's the
     * binary from the script's folder. Order of precedence:
     *   1. DAP `launch.cwd` if the IDE supplied one (matches the
     *      vscode-debug spec — many IDEs send the workspace folder).
     *   2. Otherwise, the directory containing the program file —
     *      keeps relative paths in the script working without IDE
     *      cooperation.
     * A failed chdir is non-fatal (logged via the DAP `output` channel
     * upstream of this function); the launch still proceeds because
     * the user may be invoking the program with absolute paths. */
    std::string TargetCwd;
    auto CwdArg = Args->getString("cwd");
    if (CwdArg && !CwdArg->empty()) {
      TargetCwd = CwdArg->str();
    } else {
      auto Slash = G.ProgramPath.find_last_of("/\\");
      if (Slash != std::string::npos && Slash > 0)
        TargetCwd = G.ProgramPath.substr(0, Slash);
    }
    if (!TargetCwd.empty()) {
      if (chdir(TargetCwd.c_str()) != 0) {
        std::cerr << "matlabc -dap: chdir(" << TargetCwd
                  << ") failed: " << std::strerror(errno) << "\n";
      }
    }
    if (!compileProgram()) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("failed to compile program"));
      return true;
    }
    matlab_dbg_enable(SoE ? 1 : 0);
    sendResponse(ReqSeq, *Cmd, true, Object{});
    return true;
  }

  if (*Cmd == "setBreakpoints") {
    const Object *Src = Args->getObject("source");
    if (!Src) {
      sendResponse(ReqSeq, *Cmd, false, Value("no source"));
      return true;
    }
    /* Resolve the IDE-supplied source.path against our path → file_id
     * table. A miss means the JIT didn't load that file (yet) — we
     * still respond successfully with verified=false for each
     * breakpoint so the IDE doesn't tear down the connection, but
     * nothing gets added to the runtime table. */
    auto SrcPath = Src->getString("path");
    std::string CanonSrc = SrcPath ? canonPath(SrcPath->str())
                                   : std::string();
    int32_t Fid = 0;
    if (!CanonSrc.empty()) {
      auto It = G.PathToFileId.find(CanonSrc);
      if (It != G.PathToFileId.end()) Fid = It->second;
    }
    if (Fid != 0) {
      /* Wipe prior breakpoints for this file and replay the request. */
      matlab_dbg_clear_breakpoints_in_file(Fid);
    }
    /* Drop any prior pending entries for this path — the IDE's
     * setBreakpoints semantics replace, not append. After this clear,
     * we'll re-queue the new list below if the path is still unknown. */
    if (!CanonSrc.empty()) {
      G.PendingBps.erase(
        std::remove_if(G.PendingBps.begin(), G.PendingBps.end(),
                       [&](const Shared::PendingBp &P) {
                         return P.Path == CanonSrc;
                       }),
        G.PendingBps.end());
    }
    /* Look up the per-file executable-line set populated by the AST
     * walker in compileProgram. Used to snap user-picked lines onto
     * the nearest bp-eligible row when they click on a blank /
     * comment-only line — better UX than silently failing to
     * verify. */
    const std::set<int32_t> *ExecLines = nullptr;
    if (Fid != 0) {
      auto BL = G.BpLocations.find(Fid);
      if (BL != G.BpLocations.end()) ExecLines = &BL->second;
    }
    const std::unordered_map<int32_t, int32_t> *AliasLines = nullptr;
    if (Fid != 0) {
      auto AL = G.BpAliases.find(Fid);
      if (AL != G.BpAliases.end()) AliasLines = &AL->second;
    }
    const Array *Bps = Args->getArray("breakpoints");
    Array Verified;
    if (Bps) {
      for (const auto &B : *Bps) {
        const Object *BO = B.getAsObject();
        if (!BO) continue;
        auto Ln = BO->getInteger("line");
        if (!Ln) continue;
        int32_t Requested = (int32_t)*Ln;
        int32_t Resolved = Requested;
        std::string Msg;
        bool Snapped = false;
        if (Fid == 0) {
          Msg = "source not loaded by compileProgram";
        } else if (ExecLines) {
          if (ExecLines->count(Requested) == 0) {
            /* Aliased line first: a click inside a Stmt's span (most
             * commonly a `.mflow` block's body lines) maps back to
             * the Stmt's canonical begin line, which is where the
             * runtime hook actually fires. */
            int32_t Alias = 0;
            if (AliasLines) {
              auto AI = AliasLines->find(Requested);
              if (AI != AliasLines->end()) Alias = AI->second;
            }
            if (Alias != 0) {
              Resolved = Alias;
              Snapped = true;
              Msg = "snapped to start of enclosing block";
            } else {
              /* Snap forward to the next executable line. Forward
               * only — snapping backward would land before the
               * user's intent for a click in a blank-line gap
               * between two statements. */
              auto It = ExecLines->lower_bound(Requested);
              if (It != ExecLines->end()) {
                Resolved = *It;
                Snapped = true;
                Msg = "snapped to next executable line";
              } else {
                Msg = "no executable line at or after this row";
              }
            }
          }
        }
        bool OK = false;
        if (Fid != 0 && (!Msg.size() || Snapped)) {
          /* condition / logMessage are optional in the DAP spec;
           * when present, route through the _ex form so the runtime
           * stores the strings alongside the (file_id, line) pair
           * for the monitor thread to read once the bp matches. */
          auto Cond = BO->getString("condition");
          auto Log  = BO->getString("logMessage");
          auto Hit  = BO->getString("hitCondition");
          std::string CS = Cond ? Cond->str() : std::string();
          std::string LS = Log  ? Log->str()  : std::string();
          /* Parse `hitCondition` into (op, target). DAP doesn't
           * specify the syntax beyond "an expression that determines
           * how many hits are ignored" — VS Code accepts a bare
           * integer (== N), `>=N`, `>N`, and `%N`. We support all
           * four; anything else falls back to op=0 (no gate) plus a
           * message field so the user knows their input was ignored. */
          int HitOp = 0;
          int64_t HitTarget = 0;
          if (Hit && !Hit->empty()) {
            llvm::StringRef HC = *Hit;
            HC = HC.trim();
            if (HC.consume_front(">=")) HitOp = 2;
            else if (HC.consume_front("==")) HitOp = 1;
            else if (HC.consume_front(">"))  HitOp = 3;
            else if (HC.consume_front("%"))  HitOp = 4;
            else HitOp = 1;  /* bare "100" = stop on the 100th hit */
            HC = HC.trim();
            int64_t N = 0;
            if (!HC.getAsInteger(10, N) && N > 0) {
              HitTarget = N;
            } else {
              HitOp = 0;
              if (Msg.empty())
                Msg = "ignored unparseable hitCondition";
            }
          }
          OK = matlab_dbg_add_breakpoint_ex2(
              Fid, Resolved,
              CS.empty() ? nullptr : CS.data(), (int64_t)CS.size(),
              LS.empty() ? nullptr : LS.data(), (int64_t)LS.size(),
              HitOp, HitTarget);
          if (!OK && Msg.empty())
            Msg = "breakpoint table full";
        }
        /* Path didn't resolve at request time — queue the bp so we
         * can re-verify it once compileProgram registers the path
         * (e.g. setBreakpoints arrived before launch, which the DAP
         * spec allows). The IDE sees verified=false in this response;
         * when the path later registers, we emit a `breakpoint`
         * event with reason="changed" carrying verified=true. */
        if (Fid == 0 && !CanonSrc.empty()) {
          Shared::PendingBp P;
          P.Path = CanonSrc;
          P.Line = Requested;
          if (auto Cond = BO->getString("condition"))
            P.Condition = Cond->str();
          if (auto Log = BO->getString("logMessage"))
            P.LogMessage = Log->str();
          if (auto Hit = BO->getString("hitCondition"))
            P.HitCondition = Hit->str();
          G.PendingBps.push_back(std::move(P));
          if (Msg.empty())
            Msg = "source not loaded yet — bp queued for replay";
        }
        Object Out{
          {"verified", OK},
          {"line", (int64_t)Resolved},
        };
        if (OK) Out["id"] = encodeBpId(Fid, Resolved);
        if (!Msg.empty()) Out["message"] = Msg;
        Verified.push_back(std::move(Out));
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"breakpoints", std::move(Verified)}});
    return true;
  }

  if (*Cmd == "configurationDone") {
    sendResponse(ReqSeq, *Cmd, true, Object{});
    /* #112: only start the worker if `launch` actually produced a JIT
     * engine. If compilation failed (G.Engine is null), the worker
     * would dereference a null engine in `ExecutionEngine::lookup` and
     * SIGSEGV. Instead, emit a `terminated` event so the IDE ends the
     * session cleanly — the failure diagnostics were already delivered
     * on the `launch` response and via the stderr `output` channel. */
    if (!G.Engine) {
      sendEvent("terminated");
      return true;
    }
    pthread_mutex_lock(&G.Mu);
    bool JustStarted = false;
    if (!G.WorkerStarted) {
      dapSpawnBigStack(&G.Worker, workerMain, nullptr);  /* #137: 8 MB stack */
      G.WorkerStarted = true;
      JustStarted = true;
      /* Detach; we use G.WorkerExited to know when it's done. */
      pthread_detach(G.Worker);
      /* Spawn the helper threads after the worker is kicked. */
      pthread_t Mon, Watcher, Rdr;
      dapSpawnBigStack(&Mon, monitorMain, nullptr);  /* #137: 8 MB stack */
      pthread_detach(Mon);
      pthread_create(&Watcher, nullptr, pauseWatcherMain, nullptr);
      pthread_detach(Watcher);
      pthread_create(&Rdr, nullptr, stdoutReaderMain, nullptr);
      pthread_detach(Rdr);
      if (DebuggeeErrFd >= 0) {
        pthread_t ErrRdr;
        pthread_create(&ErrRdr, nullptr, stderrReaderMain, nullptr);
        pthread_detach(ErrRdr);
      }
    }
    pthread_mutex_unlock(&G.Mu);
    if (JustStarted) {
      /* `process` advertises the debuggee identity to the IDE — useful
       * for adapters that show "Attached to <name> (pid: ...)" in
       * their status bar. We're a JIT host so there is no separate
       * pid to advertise; report ours. */
      sendEvent("process", Object{
        {"name", G.ProgramPath},
        {"systemProcessId", (int64_t)getpid()},
        {"isLocalProcess", true},
        {"startMethod", "launch"},
      });
      /* `thread` started: single MATLAB worker. The id matches what
       * `threads` returns and what `stopped`/`continued` events
       * carry. */
      sendEvent("thread",
                Object{{"reason", "started"}, {"threadId", (int64_t)1}});
      /* `loadedSource` per registered file gives the IDE a
       * source-tree view (multi-file launches show every sibling .m
       * that was auto-loaded, not just the entry point). */
      for (const auto &Kv : G.PathToFileId) {
        sendEvent("loadedSource", Object{
          {"reason", "new"},
          {"source", Object{
            {"name", Kv.first},
            {"path", Kv.first},
            {"sourceReference", (int64_t)0},
          }},
        });
      }
    }
    return true;
  }

  if (*Cmd == "threads") {
    /* Enumerate registered threads from the runtime. Thread id 1
     * is the main script worker (lazy-registered on its first
     * hook fire); ids 2..N are parfor workers, in spawn order.
     *
     * Pre-launch the table is empty — return a synthetic single
     * "main" entry so the IDE renders the threads pane instead
     * of falling back to "no threads". The synthetic id matches
     * what the runtime would assign on first hook fire, so a
     * pre-launch threads response stays consistent with the
     * post-launch view. */
    Array Ts;
    int N = matlab_dbg_thread_count();
    if (N == 0) {
      Ts.push_back(Object{{"id", 1}, {"name", "main"}});
    } else {
      for (int i = 0; i < N; ++i) {
        int32_t Id = matlab_dbg_thread_id_at(i);
        std::string Name = (Id == 1) ? "main"
                                      : "parfor-" + std::to_string(Id - 1);
        Ts.push_back(Object{{"id", (int64_t)Id}, {"name", std::move(Name)}});
      }
    }
    sendResponse(ReqSeq, *Cmd, true, Object{{"threads", std::move(Ts)}});
    return true;
  }

  if (*Cmd == "stackTrace") {
    int N = matlab_dbg_frame_count();
    Array Frames;
    int FrameId = 0;
    for (int i = 0; i < N; ++i) {
      int32_t Fid = 0, Ln = 0;
      const char *FnName = nullptr;
      if (!matlab_dbg_frame_at(i, &Fid, &Ln, &FnName)) break;
      // Phase 8a: when the entry-point is a `.mflow`, append the
      // originating block id to the frame name so the IDE can
      // highlight the active block on the canvas. Format:
      // `<frame> [block:n_kind_n]`. No-op for `.m` programs (the
      // map is empty).
      std::string Name = FnName ? FnName : "<frame>";
      if (!G.BlockByLine.empty()) {
        int64_t Key = (static_cast<int64_t>((uint32_t)Fid) << 32) |
                      static_cast<int64_t>((uint32_t)Ln);
        auto It = G.BlockByLine.find(Key);
        if (It != G.BlockByLine.end()) {
          Name += " [block:";
          Name += It->second;
          Name += "]";
        }
      }
      Object Fr{
        {"id", FrameId++},
        {"name", std::move(Name)},
        {"line", (int64_t)Ln},
        {"column", (int64_t)1},
        {"source", sourceObjForFile(Fid)},
      };
      Frames.push_back(std::move(Fr));
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"stackFrames", std::move(Frames)},
                        {"totalFrames", (int64_t)N}});
    return true;
  }

  if (*Cmd == "scopes") {
    /* DAP `scopes` is parameterised by the frame the IDE is asking
     * about. Return one Locals scope whose variablesReference encodes
     * the frame so the matching `variables` request knows which slice
     * of the runtime to read. Encoding: 1000 + DAP_frame_id, where
     * DAP frame ids are 0 = innermost / top-of-stack (matches what
     * stackTrace publishes). The legacy ref `1` is preserved as an
     * alias for the script-level workspace so any IDE / test that
     * hardcodes it keeps working. */
    Array Sc;
    auto FrameId = Args->getInteger("frameId");
    int64_t DapFrameId = FrameId.value_or(0);
    int64_t Ref = 1000 + DapFrameId;
    Sc.push_back(Object{
      {"name", "Locals"},
      {"variablesReference", Ref},
      {"expensive", false},
    });
    sendResponse(ReqSeq, *Cmd, true, Object{{"scopes", std::move(Sc)}});
    return true;
  }

  if (*Cmd == "variables") {
    /* Decode the variablesReference. The DAP frame_id (0 = innermost)
     * maps to the runtime's frames[] array (0 = outermost) via the
     * inverse: runtime_idx = n_frames - 1 - dap_frame_id. The script
     * frame (runtime_idx == 0) gets a merged view: matlab_ws (REPL-
     * mode'd script assignments) plus frame_locals[0] (loop induction
     * vars and other slot-stored values). Function frames just use
     * their per-frame mini-ws. */
    auto VR = Args->getInteger("variablesReference");
    Array Vs;
    int64_t Ref = VR.value_or(0);
    int RtFrameIdx = -1;          /* -1 means "script ws only" */
    bool MergeScriptWs = false;
    /* Matrix expansion: same pattern as the obj path below, but the
     * children are scalar cells instead of properties. The handle
     * came from a kind=1 row (LOCALS, watch eval, or an obj
     * property that holds a matrix); we resolve it back to the
     * matlab_mat* and walk its row-major buffer via the runtime
     * accessor. The window check is `< MatRefBase + 100000` so an
     * out-of-range ref doesn't accidentally hit MatRefs when the
     * caller meant ObjRefs (or vice versa). */
    if (Ref >= MatRefBase) {
      auto *M = (struct matlab_mat *)lookupMatRef(Ref);
      if (M) appendMatChildren(Vs, M);
      sendResponse(ReqSeq, *Cmd, true,
                   Object{{"variables", std::move(Vs)}});
      return true;
    }
    /* Phase 5.3 — table-column expansion. The variablesReference came
     * from a kind=6 row (workspace pane); resolve to matlab_table*
     * and emit one row per column using the per-kind formatter for
     * the column's data pointer. CRITICAL: this branch must come
     * before the ObjRefBase check below — TableRefBase < ObjRefBase
     * by design, but the obj-branch's `Ref >= ObjRefBase` test would
     * still miss table refs, so the actual safety here is that
     * table refs never leak into MatRefs / ObjRefs (the lookup helpers
     * gate by window). */
    if (Ref >= TableRefBase && Ref < ObjRefBase) {
      auto *T = lookupTableRef(Ref);
      if (T) {
        int32_t NCols = (int32_t)matlab_table_width(T);
        int64_t NRows = (int64_t)matlab_table_height(T);
        for (int32_t ci = 0; ci < NCols; ++ci) {
          int64_t NameLen = 0;
          const char *NameP = matlab_table_column_name(T, ci, &NameLen);
          if (!NameP) continue;
          std::string ColName(NameP, (size_t)NameLen);
          int32_t CK = matlab_table_column_kind_idx(T, ci);
          /* MATLAB_TABLE_KIND_NUMERIC=0  → matlab_mat *
           * MATLAB_TABLE_KIND_STRING=1   → matlab_string ** array
           * MATLAB_TABLE_KIND_DATETIME=2 → matlab_datetime ** array
           * For v1 we render NUMERIC columns through formatMatShape +
           * registerMatRef so the user can drill into the cells (the
           * existing matrix viewer path). STRING / DATETIME columns
           * render with a shape summary only — drilling into pointer
           * arrays would need a new dispatch and isn't needed to clear
           * the workspace-crash bug. */
          std::string Val;
          std::string Type;
          int64_t ChildRef = 0;
          int64_t IndexedHint = 0;
          if (CK == 0) {
            auto *M = (struct matlab_mat *)matlab_table_column_data(T, ci);
            Val = formatMatShape(M);
            Type = "double";
            if (M && matIsMultiCell(M)) {
              ChildRef = registerMatRef(M);
              IndexedHint = matIndexedCount(M);
            }
          } else if (CK == 1) {
            char Buf[64];
            snprintf(Buf, sizeof Buf, "%lldx1 string", (long long)NRows);
            Val = Buf;
            Type = "string";
          } else if (CK == 2) {
            char Buf[64];
            snprintf(Buf, sizeof Buf, "%lldx1 datetime", (long long)NRows);
            Val = Buf;
            Type = "datetime";
          } else {
            Val = "<unknown column kind>";
            Type = "any";
          }
          Object Row{
            {"name", std::move(ColName)},
            {"value", std::move(Val)},
            {"type", std::move(Type)},
            {"variablesReference", ChildRef},
          };
          if (IndexedHint > 0) Row["indexedVariables"] = IndexedHint;
          Vs.push_back(std::move(Row));
        }
      }
      sendResponse(ReqSeq, *Cmd, true,
                   Object{{"variables", std::move(Vs)}});
      return true;
    }
    /* Object-property expansion: when the IDE clicks the disclosure
     * arrow on a class-instance row, the request comes back with the
     * variablesReference we previously handed out. Resolve it back to
     * a matlab_obj* and emit one row per property. */
    if (Ref >= ObjRefBase) {
      void *obj = lookupObjRef(Ref);
      if (obj) {
        int N = matlab_dbg_obj_field_count(obj);
        for (int i = 0; i < N; ++i) {
          int64_t Nlen = 0;
          const char *Nm = matlab_dbg_obj_field_name(obj, i, &Nlen);
          if (!Nm) continue;
          int K = matlab_dbg_obj_field_kind(obj, i);
          std::string Val;
          int64_t ChildRef = 0;
          int64_t IndexedHint = 0;
          int64_t NamedHint = 0;
          std::string MemRef;
          if (K == 0) {
            char Buf[64];
            snprintf(Buf, sizeof Buf, "%g",
                     matlab_dbg_obj_field_f64(obj, i));
            Val = Buf;
          } else if (K == 1) {
            auto *M = (struct matlab_mat *)matlab_dbg_obj_field_ptr(obj, i);
            Val = formatMatShape(M);
            /* Multi-cell matrix properties are drillable too — the
             * Matrix Viewer / Variable Inspector can chase the ref
             * down without a separate eval. The memoryReference
             * exposes the data buffer so the IDE's memory view
             * can dump raw bytes. */
            if (M && matIsMultiCell(M)) {
              ChildRef = registerMatRef(M);
              IndexedHint = matIndexedCount(M);
              MemRef = registerMatMemRef(M);
            }
          } else if (K == 2) {
            void *child = matlab_dbg_obj_field_ptr(obj, i);
            Val = formatObj(child);
            if (child) {
              ChildRef = registerObjRef(child);
              NamedHint = objNamedCount(child);
            }
          } else {
            Val = "<unknown>";
          }
          Object Row{
            {"name", std::string(Nm, (size_t)Nlen)},
            {"value", Val},
            {"type", typeForVar(K, K == 2 ? matlab_dbg_obj_field_ptr(obj, i)
                                          : nullptr)},
            {"variablesReference", ChildRef},
          };
          if (IndexedHint > 0) Row["indexedVariables"] = IndexedHint;
          if (NamedHint > 0) Row["namedVariables"] = NamedHint;
          if (!MemRef.empty()) Row["memoryReference"] = MemRef;
          Vs.push_back(std::move(Row));
        }
        /* Method rows. After the property rows, emit one entry per
         * method declared on the obj's class (resolved via the
         * runtime's class_id table) and on every superclass walked
         * via G.ClassParent. Methods are leaves (variablesReference=0)
         * — there's no "expand a method" affordance — but the IDE
         * renders them with a function icon via
         * `presentationHint.kind="method"`.
         *
         * The value column shows a compact signature (`@deposit(obj,
         * amt)`) so users can see arity at a glance without
         * jumping to the source. Methods inherited from a parent
         * class get a "(inherited from X)" suffix on the value to
         * disambiguate from the obj's own methods.
         *
         * Duplicate-name handling: a derived class can override a
         * parent method (`Savings.deposit` shadows `Account.deposit`).
         * We track seen names while walking the chain so the
         * override wins and the parent entry is suppressed. */
        int32_t cid = matlab_dbg_obj_class_id_of(obj);
        int64_t cnLen = 0;
        const char *cn = matlab_dbg_class_name(cid, &cnLen);
        if (cn && cnLen > 0) {
          std::string ClassName(cn, (size_t)cnLen);
          std::unordered_set<std::string> SeenMethods;
          for (std::string Cur = ClassName; !Cur.empty();) {
            auto MIt = G.ClassMethods.find(Cur);
            if (MIt != G.ClassMethods.end()) {
              for (const Shared::MethodEntry &ME : MIt->second) {
                if (!SeenMethods.insert(ME.Name).second) continue;
                std::string Sig = "@" + ME.Name + "(";
                for (size_t k = 0; k < ME.Inputs.size(); ++k) {
                  if (k) Sig += ", ";
                  Sig += ME.Inputs[k];
                }
                Sig += ")";
                if (Cur != ClassName) {
                  Sig += "  (inherited from ";
                  Sig += Cur;
                  Sig += ")";
                }
                std::string TypeLabel = ME.Static ? "static method"
                                                  : "method";
                Object Row{
                  {"name", ME.Name},
                  {"value", Sig},
                  {"type", TypeLabel},
                  {"variablesReference", (int64_t)0},
                  /* DAP `presentationHint` controls the IDE's row
                   * glyph. `kind: "method"` selects the function
                   * icon; `attributes: ["readOnly"]` suppresses the
                   * inline-edit affordance (you can't reassign a
                   * method on an instance through the watch UI). */
                  {"presentationHint", Object{
                    {"kind", "method"},
                    {"attributes", Array{Value("readOnly")}},
                    {"visibility", "public"},
                  }},
                };
                Vs.push_back(std::move(Row));
              }
            }
            auto PIt = G.ClassParent.find(Cur);
            if (PIt == G.ClassParent.end()) break;
            Cur = PIt->second;
          }
        }
      }
      sendResponse(ReqSeq, *Cmd, true,
                   Object{{"variables", std::move(Vs)}});
      return true;
    }
    if (Ref == 1) {
      /* Legacy ref. Behave as before: return matlab_ws contents only.
       * Existing tests that hardcode `1` continue to work. */
      MergeScriptWs = true;
    } else if (Ref >= 1000) {
      int DapFrameId = (int)(Ref - 1000);
      int Total = matlab_dbg_frame_count();
      RtFrameIdx = Total - 1 - DapFrameId;
      if (RtFrameIdx < 0 || RtFrameIdx >= Total) RtFrameIdx = -1;
      /* The outermost frame is the script — merge matlab_ws into its
       * Locals view. Inner function frames only show their own mini-ws. */
      if (RtFrameIdx == 0) MergeScriptWs = true;
    }

    /* Track names we've already emitted so the merge doesn't report
     * the same variable twice when matlab_ws and the script-frame
     * mini-ws both happen to carry it. matlab_ws wins (it's the most
     * authoritative for top-level assignments under ReplMode). */
    std::unordered_set<std::string> Seen;
    if (MergeScriptWs) {
      int N = matlab_dbg_ws_count();
      for (int i = 0; i < N; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_ws_name(i, &Nlen);
        int K = matlab_dbg_ws_kind(i);
        std::string Nstr(Nm, (size_t)Nlen);
        Seen.insert(Nstr);
        /* Class instances get a variablesReference so the IDE can
         * expand them. Matrix rows get one too, so the IDE can drill
         * into the cells via the standard `variables(ref)` path —
         * 1x1 matrices are skipped because formatMatShape already
         * unboxes them to the scalar value. */
        int64_t ChildRef = 0;
        int64_t IndexedCount = 0;
        int64_t NamedCount = 0;
        std::string MemRef;
        if (K == 2) {
          if (void *obj = matlab_dbg_ws_ptr(i)) {
            ChildRef = registerObjRef(obj);
            NamedCount = objNamedCount(obj);
          }
        } else if (K == 1) {
          auto *M = (struct matlab_mat *)matlab_dbg_ws_ptr(i);
          if (M && matIsMultiCell(M)) {
            ChildRef = registerMatRef(M);
            IndexedCount = matIndexedCount(M);
            MemRef = registerMatMemRef(M);
          }
        } else if (K == 6) {
          /* Phase 5.3 — table row gets a variablesReference into the
           * TableRefs registry so clicking the disclosure arrow drops
           * into the column-walking branch in the `variables` handler,
           * not the matlab_dbg_mat_get cell walker (which would crash
           * on a misinterpreted matlab_table_s). */
          if (auto *T = (matlab_table *)matlab_dbg_ws_ptr(i)) {
            ChildRef = registerTableRef(T);
            NamedCount = (int64_t)matlab_table_width(T);
          }
        }
        Object Row{
          {"name", Nstr},
          {"value", formatVar(K, i)},
          {"type", typeForVar(K, K == 2 ? matlab_dbg_ws_ptr(i) : nullptr)},
          {"variablesReference", ChildRef},
        };
        if (IndexedCount > 0) Row["indexedVariables"] = IndexedCount;
        if (NamedCount > 0) Row["namedVariables"] = NamedCount;
        if (!MemRef.empty()) Row["memoryReference"] = MemRef;
        Vs.push_back(std::move(Row));
      }
    }
    if (RtFrameIdx >= 0) {
      int N = matlab_dbg_frame_locals_count(RtFrameIdx);
      for (int i = 0; i < N; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_frame_local_name(RtFrameIdx, i, &Nlen);
        if (!Nm) continue;
        std::string Nstr(Nm, (size_t)Nlen);
        if (Seen.count(Nstr)) continue;
        int K = matlab_dbg_frame_local_kind(RtFrameIdx, i);
        /* Inline format: scalars print as "%g", matrices as "RxC
         * double" (with 1x1 unboxed). Mirrors formatVar for ws but
         * pulls values from the per-frame accessors. */
        std::string Val;
        int64_t ChildRef = 0;
        int64_t IndexedCount = 0;
        int64_t NamedCount = 0;
        std::string MemRef;
        if (K == 0) {
          char Buf[64];
          double V = matlab_dbg_frame_local_f64(RtFrameIdx, i);
          snprintf(Buf, sizeof Buf, "%g", V);
          Val = Buf;
        } else if (K == 1) {
          auto *M = (struct matlab_mat *)matlab_dbg_frame_local_ptr(
              RtFrameIdx, i);
          Val = formatMatShape(M);
          /* Same gating as the matlab_ws merge above: only multi-cell
           * matrices get a child ref, scalars are leaves. */
          if (M && matIsMultiCell(M)) {
            ChildRef = registerMatRef(M);
            IndexedCount = matIndexedCount(M);
            MemRef = registerMatMemRef(M);
          }
        } else if (K == 2) {
          void *obj = matlab_dbg_frame_local_ptr(RtFrameIdx, i);
          Val = formatObj(obj);
          if (obj) {
            ChildRef = registerObjRef(obj);
            NamedCount = objNamedCount(obj);
          }
        } else {
          Val = "<unknown>";
        }
        Object Row{
          {"name", Nstr},
          {"value", Val},
          {"type", typeForVar(K, K == 2 ? matlab_dbg_frame_local_ptr(
                                              RtFrameIdx, i)
                                        : nullptr)},
          {"variablesReference", ChildRef},
        };
        if (IndexedCount > 0) Row["indexedVariables"] = IndexedCount;
        if (NamedCount > 0) Row["namedVariables"] = NamedCount;
        if (!MemRef.empty()) Row["memoryReference"] = MemRef;
        Vs.push_back(std::move(Row));
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"variables", std::move(Vs)}});
    return true;
  }

  if (*Cmd == "setVariable") {
    /* Mutate a workspace variable from the watch box. We piggyback on
     * the REPL JIT pipeline that conditional breakpoints already use:
     * wrap the user's input as `<name> = (<value>);` and run it
     * through Lex → Parse → Sema → MLIR → JIT against the persistent
     * workspace struct. Any valid MATLAB expression on the RHS works
     * — scalars, matrix literals (`[1 2; 3 4]`), strings, struct
     * accessors, function calls — without us having to re-parse them
     * here. After the assignment lands, we re-read the variable's
     * formatted value for the response so the IDE's watch box shows
     * what actually got stored. */
    auto NameOpt = Args->getString("name");
    auto ValOpt = Args->getString("value");
    if (!NameOpt || !ValOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("setVariable requires name and value"));
      return true;
    }
    std::string NameStr = NameOpt->str();
    std::string ValStr = ValOpt->str();
    /* Defense-in-depth: validate the name is a plain identifier so a
     * malformed `name` like `x); system(...` can't smuggle extra
     * statements into the assignment we're about to JIT. The REPL
     * pipeline would catch syntax errors anyway, but failing fast
     * here keeps the error message tight ("not a valid identifier")
     * instead of reflecting a parser diagnostic. */
    auto IsIdent = [](const std::string &S) {
      if (S.empty()) return false;
      char c0 = S[0];
      if (!(std::isalpha((unsigned char)c0) || c0 == '_')) return false;
      for (size_t i = 1; i < S.size(); ++i) {
        char c = S[i];
        if (!(std::isalnum((unsigned char)c) || c == '_')) return false;
      }
      return true;
    };
    if (!IsIdent(NameStr)) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("name is not a valid identifier"));
      return true;
    }
    /* The runReplInput pipeline operates at script scope and writes
     * to the workspace via matlab_ws_set_*, exactly the same path the
     * scenario's normal assignments use. Wrap with a single trailing
     * semicolon to suppress implicit display so the IDE doesn't see a
     * spurious `output` event for what should be a silent mutation. */
    std::string Src = NameStr + " = (" + ValStr + ");";
    int Rc = runReplInput(sharedDapContext(), Src, NextEvalId++);
    if (Rc != 0) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("setVariable expression failed to compile"));
      return true;
    }
    /* Re-read the variable's stored kind/value to render the response.
     * If the assignment somehow didn't land (e.g. RHS produced a
     * void), fall back to "<unset>" rather than emitting an empty
     * value the IDE would render as a blank cell. */
    int N = matlab_dbg_ws_count();
    int Found = -1, Kind = -1;
    for (int i = 0; i < N; ++i) {
      int64_t Nlen = 0;
      const char *Nm = matlab_dbg_ws_name(i, &Nlen);
      if ((size_t)Nlen == NameStr.size() &&
          std::memcmp(Nm, NameStr.data(), (size_t)Nlen) == 0) {
        Found = i; Kind = matlab_dbg_ws_kind(i);
        break;
      }
    }
    std::string Display = (Found >= 0) ? formatVar(Kind, Found)
                                       : std::string("<unset>");
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"value", Display}});
    return true;
  }

  if (*Cmd == "evaluate") {
    /* DAP `evaluate` is what powers the watch panel, hover-eval, and
     * the debug console. Implementation: wrap the user's expression
     * as `__matlab_dbg_eval = (<expr>);` and run it through the same
     * REPL JIT pipeline conditional breakpoints already use. The
     * result lands in matlab_ws under that name; we read the kind
     * back and format with the same formatVar that powers the
     * `variables` response, so a watch on `[1 2; 3 4]` shows up as
     * "2x2 double" and a watch on `x + 1` shows the scalar.
     *
     * Frame-scoped eval (item 6 in the plan): when the IDE supplies a
     * frameId pointing at a non-script frame, we bridge that frame's
     * mini-workspace into matlab_ws for the duration of the eval.
     * The bridge is reversible: snapshot every pre-existing matlab_ws
     * entry, stamp the frame locals on top, run the eval, then
     * restore. Names that didn't exist pre-stamp get cleared via
     * matlab_ws_clear_one so eval doesn't leak function locals into
     * the persistent script workspace. */
    /* Worker-state gate. `runReplInput` shares matlab_ws with the
     * JIT'd program, so evaluating while the worker is mid-execution
     * races on the workspace and the JIT engine state. We allow eval
     * in three states:
     *   - Pre-launch  (worker not yet started; ws is empty)
     *   - Paused      (worker stopped at a breakpoint; safe by design)
     *   - Post-exit   (worker finished; ws is a stable snapshot)
     * The "running, not paused" case is the unsafe one. */
    {
      pthread_mutex_lock(&G.Mu);
      bool Running = G.WorkerStarted && !G.WorkerExited;
      pthread_mutex_unlock(&G.Mu);
      if (Running && !matlab_dbg_is_paused()) {
        sendResponse(ReqSeq, *Cmd, false,
                     Value("evaluate is only valid while the program is "
                           "paused or has exited"));
        return true;
      }
    }

    auto ExprOpt = Args->getString("expression");
    if (!ExprOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("evaluate requires an expression"));
      return true;
    }
    std::string Expr = ExprOpt->str();

    /* DAP `context` distinguishes the watch panel / hover from the REPL
     * console. Watch and hover want a value to display, so we wrap as
     * `__matlab_dbg_eval = (<expr>);` and read the result back. REPL
     * wants statement-level execution: `disp(T)`, `clear x`,
     * `T(2,2) = 99` — so we run the input verbatim and let stdout flow
     * out via the existing pipe redirect → DAP `output` events. */
    auto CtxOpt = Args->getString("context");
    bool IsRepl = CtxOpt && *CtxOpt == "repl";

    /* Watch-mode auto-promotion: certain inputs are statement-shaped
     * void calls that cannot survive the `__matlab_dbg_eval = (...);`
     * wrap — the assignment-of-void crashes deep in the lowering /
     * JIT (SIGSEGV, no diagnostic). Detect them up front by extracting
     * the leading identifier and matching against a known set of
     * void-returning builtins; route those through the REPL branch
     * (run verbatim, no wrap) and return `result="<void>"` to the
     * IDE so the watch row shows a clear placeholder instead of a
     * dropped connection.
     *
     * False positives on the list are safe — the worst case is a
     * watch on `disp(A)` showing "<void>" while disp's output flows
     * out as DAP `output` events. False negatives crash matlabc, so
     * the list errs on the inclusive side. */
    auto isVoidStatement = [](llvm::StringRef S) {
      while (!S.empty() && (S.front() == ' ' || S.front() == '\t'))
        S = S.drop_front();
      size_t i = 0;
      while (i < S.size() &&
             (std::isalnum((unsigned char)S[i]) || S[i] == '_'))
        ++i;
      if (i == 0) return false;
      llvm::StringRef Name = S.substr(0, i);
      llvm::StringRef Rest = S.drop_front(i);
      while (!Rest.empty() && (Rest.front() == ' ' || Rest.front() == '\t'))
        Rest = Rest.drop_front();
      /* Statement form (`clear x`, `who`, `whos`) — bare name not
       * followed by `(` qualifies if it's in the void-statement set. */
      bool IsCallForm = !Rest.empty() && Rest.front() == '(';
      static const llvm::StringRef VoidCalls[] = {
        "disp", "fprintf", "printf", "error", "warning", "assert",
        "dbg", "plot", "figure", "hold", "axis", "title", "xlabel",
        "ylabel", "legend", "save", "load", "drawnow", "pause",
        "clf", "cla", "close", "set", "delete", "addpath", "rmpath",
        "clear", "who", "whos",
      };
      static const llvm::StringRef VoidStatements[] = {
        "clear", "who", "whos", "drawnow", "pause", "clf", "cla",
        "close", "hold", "dbcont", "dbstop", "dbquit", "dbup",
        "dbdown",
      };
      if (IsCallForm) {
        for (auto V : VoidCalls) if (Name == V) return true;
        return false;
      }
      for (auto V : VoidStatements) if (Name == V) return true;
      return false;
    };
    bool VoidPromoted = false;
    if (!IsRepl && isVoidStatement(Expr)) {
      IsRepl = true;
      VoidPromoted = true;
    }

    if (IsRepl) {
      /* Trim outer whitespace only — preserve a trailing `;` because in
       * MATLAB it suppresses the implicit display of an assignment's
       * result, and that user intent is meaningful in the REPL. */
      while (!Expr.empty() &&
             (Expr.back() == ' ' || Expr.back() == '\t' ||
              Expr.back() == '\n' || Expr.back() == '\r'))
        Expr.pop_back();
      while (!Expr.empty() &&
             (Expr.front() == ' ' || Expr.front() == '\t'))
        Expr.erase(Expr.begin());
      /* runReplInput's lexer/parser assume the input ends with a
       * newline (the standalone REPL appends `\n` after each line of
       * stdin input). Without it, parser recovery on a malformed input
       * walks past EOF and trips a libc++ length_error in some
       * downstream string op, aborting the process. Append it
       * unconditionally — it's a no-op for already-well-formed inputs
       * and keeps malformed ones contained to a clean diagnostic. */
      if (!Expr.empty()) Expr.push_back('\n');
    } else {
      /* Watch / hover: strip trailing whitespace AND `;` — the wrap we
       * add below injects its own terminator. */
      while (!Expr.empty() &&
             (Expr.back() == ' ' || Expr.back() == '\t' ||
              Expr.back() == '\n' || Expr.back() == ';'))
        Expr.pop_back();
    }
    if (Expr.empty()) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("evaluate received an empty expression"));
      return true;
    }

    /* Resolve frameId -> runtime frame index. DAP frame ids are
     * innermost-first; the runtime indexes outermost-first. The
     * script frame (rt index 0) doesn't need bridging — its locals
     * are already in matlab_ws + frame_locals[0] which the REPL JIT
     * accesses directly. */
    auto FrameIdOpt = Args->getInteger("frameId");
    int RtFrameIdx = -1;
    if (FrameIdOpt) {
      int Total = matlab_dbg_frame_count();
      int DapFrameId = (int)*FrameIdOpt;
      int Idx = Total - 1 - DapFrameId;
      if (Idx > 0 && Idx < Total) RtFrameIdx = Idx;
    }

    /* Bridge the requested frame's locals into matlab_ws for the
     * eval, then reverse the bridge afterward. Same helper used by
     * the cond/log breakpoint evaluators; see FrameBridge above. */
    FrameBridge FB;
    FB.stamp(RtFrameIdx);

    const char EvalName[] = "__matlab_dbg_eval";
    std::string Src = IsRepl
                       ? Expr
                       : (std::string(EvalName) + " = (" + Expr + ");");
    std::string DiagText;
    int Rc = runReplInput(sharedDapContext(), Src, NextEvalId++, &DiagText);

#ifdef MATLAB_LLVM_WITH_PLOT
    /* Flush any figure the eval just created — without this the
     * figure stays parked in the per-thread Figure registry until
     * the JIT worker thread exits at session teardown, so the IDE's
     * Plots panel stays empty for the duration of the debug session.
     * Runs on both success and error paths (a partial eval that
     * managed to plot before failing still flushes); the runtime
     * self-gates on MATLAB_LLVM_IDE_FIGURES and fflushes stdout. */
    matlab_ide_emit_all_figures();
#endif

    /* Read the result before any restoration so we can format it. The
     * REPL path skips this entirely — its "result" is whatever the user's
     * statement printed via disp/fprintf, which already streamed out as
     * `output` events through the stdout-redirect pipe. */
    std::string Display;
    std::string EvalType;
    int64_t EvalRef = 0;
    int64_t EvalIndexed = 0;
    int64_t EvalNamed = 0;
    bool RcOk = (Rc == 0);
    if (RcOk && !IsRepl) {
      int N = matlab_dbg_ws_count();
      int Found = -1, Kind = -1;
      int64_t EvalLen = (int64_t)(sizeof EvalName - 1);
      for (int i = 0; i < N; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_ws_name(i, &Nlen);
        if (Nlen == EvalLen &&
            std::memcmp(Nm, EvalName, (size_t)Nlen) == 0) {
          Found = i; Kind = matlab_dbg_ws_kind(i);
          break;
        }
      }
      /* Class-instance promotion. The REPL JIT compiling
       * `__matlab_dbg_eval = (<expr>);` doesn't know that the RHS is a
       * class instance — its Sema is fresh and has no view into the
       * workspace's existing bindings — so the result lands with
       * kind=1 (matlab_mat) even when the underlying pointer is a
       * matlab_obj. Detect that here by sweeping every currently
       * tracked obj pointer (matlab_ws kind=2 plus every frame's
       * mini-ws kind=2) and matching against the eval result's ptr.
       * On a hit we know the value is a class instance and switch the
       * display + variablesReference to the obj path. */
      if (Found >= 0 && Kind == 1) {
        void *EvalPtr = matlab_dbg_ws_ptr(Found);
        auto isKnownObj = [&](void *p) -> bool {
          if (!p) return false;
          int wsN = matlab_dbg_ws_count();
          for (int j = 0; j < wsN; ++j)
            if (matlab_dbg_ws_kind(j) == 2 &&
                matlab_dbg_ws_ptr(j) == p)
              return true;
          int fc = matlab_dbg_frame_count();
          for (int f = 0; f < fc; ++f) {
            int fn = matlab_dbg_frame_locals_count(f);
            for (int j = 0; j < fn; ++j)
              if (matlab_dbg_frame_local_kind(f, j) == 2 &&
                  matlab_dbg_frame_local_ptr(f, j) == p)
                return true;
          }
          return false;
        };
        if (isKnownObj(EvalPtr)) Kind = 2;
      }
      Display = (Found >= 0)
                ? (Kind == 2
                   ? formatObj(matlab_dbg_ws_ptr(Found))
                   : formatVar(Kind, Found))
                : std::string("<void>");
      /* Hand back a variablesReference for class-instance eval
       * results so the IDE can expand a watched object inline (the
       * obj pointer survives the matlab_ws_clear_one below — the
       * underlying obj is owned by the originating slot, not by the
       * workspace's name binding). Multi-cell matrix results get
       * the same treatment via the MatRefs registry so the watch
       * box can drill into a `[1 2; 3 4]` literal or an `A * B`
       * expression. */
      if (Found >= 0 && Kind == 2) {
        if (void *obj = matlab_dbg_ws_ptr(Found)) {
          EvalRef = registerObjRef(obj);
          EvalNamed = objNamedCount(obj);
        }
      } else if (Found >= 0 && Kind == 1) {
        auto *M = (struct matlab_mat *)matlab_dbg_ws_ptr(Found);
        if (M && matIsMultiCell(M)) {
          EvalRef = registerMatRef(M);
          EvalIndexed = matIndexedCount(M);
        }
      }
      if (Found >= 0)
        EvalType = typeForVar(Kind,
            Kind == 2 ? matlab_dbg_ws_ptr(Found) : nullptr);
    }

    /* Clear the eval-result holder first so it doesn't pile up
     * across many evaluate calls, then reverse the frame bridge
     * (clears stamped names, restores pre-existing values). */
    matlab_ws_clear_one(EvalName, (int64_t)(sizeof EvalName - 1));
    FB.restore();

    if (!RcOk) {
      /* Captured diagnostics (parser / type / lowering errors)
       * become the response message so the IDE's watch row shows
       * the actual cause — first line in the cell, full text in
       * the hover tooltip. The same bytes also reached stderr via
       * Diag.printAll(); the stderr-forwarding pipe surfaces them
       * in the debug console for users who prefer scrolling
       * through the full text there. */
      std::string Msg = DiagText;
      /* Trim trailing newlines so the IDE's single-line message
       * field doesn't render an awkward blank trailing row. */
      while (!Msg.empty() && (Msg.back() == '\n' || Msg.back() == '\r'))
        Msg.pop_back();
      if (Msg.empty())
        Msg = IsRepl ? "REPL input failed to run"
                     : "evaluate expression failed to compile";
      sendResponse(ReqSeq, *Cmd, false, Value(Msg));
      return true;
    }
    /* If the watch handler auto-promoted a void statement to the
     * REPL path, the IDE still expects a value-shaped response.
     * Render `<void>` so the watch row shows a clear placeholder
     * instead of an empty cell — the actual side effect (printed
     * output) flowed through the DAP `output` event channel. */
    if (VoidPromoted) Display = "<void>";
    Object Body{{"result", Display},
                {"variablesReference", EvalRef}};
    if (!EvalType.empty()) Body["type"] = EvalType;
    if (EvalIndexed > 0) Body["indexedVariables"] = EvalIndexed;
    if (EvalNamed > 0) Body["namedVariables"] = EvalNamed;
    sendResponse(ReqSeq, *Cmd, true, std::move(Body));
    return true;
  }

  auto nudgeMonitor = [] {
    pthread_mutex_lock(&G.Mu);
    /* Bump the generation so the monitor's inner wait, which
     * snapshots ResumeGen before sleeping, exits the moment the
     * client has acted. The broadcast wakes the wait. */
    G.ResumeGen++;
    pthread_cond_broadcast(&G.Cv);
    pthread_mutex_unlock(&G.Mu);
  };

  /* Serialise the next step request against the monitor's
   * delivery of the *previous* step's `stopped` event.
   *
   * Why: the IDE may pipeline `next` clicks faster than the
   * worker can pause + the monitor can emit a `stopped` event.
   * Without serialisation, a second `next` arriving while the
   * worker is still mid-step from the first issues a redundant
   * `matlab_dbg_resume` (paused was already 0; no-op) but still
   * bumps `ResumeGen`. The monitor's stopped/resume accounting
   * drifts: fewer `stopped` events fire than the IDE expects,
   * and the user sees "click Step Over a few times — eventually
   * it stops pausing, I have to hit Pause manually."
   *
   * Fix: count step requests issued (`StepsRequested`) and
   * stops the monitor has actually delivered (`StopsEmitted`).
   * A new step request waits until `StopsEmitted` has caught up
   * — i.e. the previous step's `stopped` event has been sent —
   * before bumping its own counter and issuing the resume. One
   * step request maps to exactly one `stopped` event.
   *
   * The pauseWatcher broadcasts `G.Cv` every 20 ms, so the wait
   * also wakes promptly on pause-state transitions; the timeout
   * is a safety bound for malformed sequences (e.g. a step
   * issued against a worker that has already exited).
   *
   * Returns true if a step slot was acquired (the step handler
   * should proceed); false if the worker exited or the wait
   * timed out (the handler should bail with an error response). */
  auto waitForStepReady = [](int timeoutMs = 5000) -> bool {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += timeoutMs / 1000;
    deadline.tv_nsec += (long)(timeoutMs % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
      deadline.tv_sec += 1;
      deadline.tv_nsec -= 1000000000L;
    }
    pthread_mutex_lock(&G.Mu);
    /* Invariant: at rest (between steps), StopsEmitted is exactly
     * one more than StepsRequested — the +1 accounts for the
     * stop that brought us back from the previous step (or the
     * initial breakpoint / stopOnEntry stop). Waiting for
     * `StopsEmitted > StepsRequested` ensures the monitor has
     * delivered the stopped event for the previous step before
     * we issue a new resume. Using `>` instead of `>=` is the
     * difference between "this step has paused" and "the
     * stopped event has actually been sent." */
    while (!G.WorkerExited &&
           (G.MonitorBusy ||
            G.StopsEmitted <= G.StepsRequested ||
            !matlab_dbg_is_paused())) {
      int rc = pthread_cond_timedwait(&G.Cv, &G.Mu, &deadline);
      if (rc == ETIMEDOUT) break;
    }
    bool ready = !G.WorkerExited &&
                 !G.MonitorBusy &&
                 G.StopsEmitted > G.StepsRequested &&
                 matlab_dbg_is_paused();
    if (ready) G.StepsRequested++;
    pthread_mutex_unlock(&G.Mu);
    return ready;
  };
  /* --- Source / file inspection --------------------------------------- */

  if (*Cmd == "loadedSources") {
    /* Return one Source object per .m file the SourceManager loaded
     * during compileProgram (entry point + auto-loaded siblings).
     * Mirrors the `loadedSource` events we fire on configurationDone
     * so a late-attaching client can still build a complete source
     * tree via this poll. */
    Array Ss;
    for (const auto &Kv : G.PathToFileId) {
      Ss.push_back(Object{
        {"name", Kv.first},
        {"path", Kv.first},
        {"sourceReference", (int64_t)0},
      });
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"sources", std::move(Ss)}});
    return true;
  }

  if (*Cmd == "source") {
    /* Read file content from disk by path. Used by IDEs that don't
     * have direct file-system access (remote-debug or container
     * scenarios). Local debug sessions short-circuit this — the IDE
     * already has the .m file open. */
    auto SrcObj = Args->getObject("source");
    std::string Path;
    if (SrcObj) {
      if (auto P = SrcObj->getString("path")) Path = P->str();
    }
    if (Path.empty()) {
      sendResponse(ReqSeq, *Cmd, false, Value("source requires a path"));
      return true;
    }
    std::ifstream In(Path);
    if (!In) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("source: cannot open " + Path));
      return true;
    }
    std::string Content((std::istreambuf_iterator<char>(In)),
                        std::istreambuf_iterator<char>());
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"content", std::move(Content)},
                        {"mimeType", "text/x-matlab"}});
    return true;
  }

  if (*Cmd == "modules") {
    /* No shared-library / dynamically-loaded module concept in our
     * JIT model — every .m file goes through compileProgram into a
     * single ExecutionEngine. Return an empty list so module-aware
     * IDEs render an empty Modules pane instead of falling back to
     * the unknown-handler reply. */
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"modules", Array{}}, {"totalModules", (int64_t)0}});
    return true;
  }

  /* --- Breakpoint variants -------------------------------------------- */

  if (*Cmd == "breakpointLocations") {
    /* Return every breakpointable line in [startLine, endLine] for
     * a given source. Server-side: G.BpLocations is populated at
     * compileProgram time by walking the AST (one entry per
     * statement start line). The actual hook may not fire on every
     * recorded line (the lowering normalises hook lines past blank
     * / comment-only rows), but setBreakpoints is authoritative for
     * whether a given line resolves — this request is only there to
     * tell the IDE which rows to highlight as candidates. */
    auto SrcObj = Args->getObject("source");
    auto StartLineOpt = Args->getInteger("line");
    auto EndLineOpt = Args->getInteger("endLine");
    int32_t Fid = 0;
    if (SrcObj) {
      if (auto P = SrcObj->getString("path")) {
        auto It = G.PathToFileId.find(canonPath(P->str()));
        if (It != G.PathToFileId.end()) Fid = It->second;
      }
    }
    int64_t Start = StartLineOpt.value_or(1);
    int64_t End = EndLineOpt.value_or(Start);
    Array Locs;
    std::set<int32_t> Reported;
    auto It = G.BpLocations.find(Fid);
    if (It != G.BpLocations.end()) {
      for (int32_t L : It->second) {
        if ((int64_t)L >= Start && (int64_t)L <= End) {
          if (Reported.insert(L).second)
            Locs.push_back(Object{{"line", (int64_t)L},
                                  {"column", (int64_t)1}});
        }
      }
    }
    /* Alias keys: lines a Stmt covers but doesn't start on. The IDE
     * highlights every JSON line of a `.mflow` block (not just the
     * line of the opening `{`); setBreakpoints rewrites these back
     * to the canonical begin line at install time. */
    auto AIt = G.BpAliases.find(Fid);
    if (AIt != G.BpAliases.end()) {
      for (auto &P : AIt->second) {
        int32_t L = P.first;
        if ((int64_t)L >= Start && (int64_t)L <= End) {
          if (Reported.insert(L).second)
            Locs.push_back(Object{{"line", (int64_t)L},
                                  {"column", (int64_t)1}});
        }
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"breakpoints", std::move(Locs)}});
    return true;
  }

  if (*Cmd == "setFunctionBreakpoints") {
    /* Resolve each function name against G.FunctionTable (populated
     * at compileProgram time) and install a line breakpoint at the
     * function's first body line. Unknown names come back with
     * verified=false rather than dropping the connection. */
    const Array *Bps = Args->getArray("breakpoints");
    Array Verified;
    if (Bps) {
      for (const auto &V : *Bps) {
        const Object *B = V.getAsObject();
        std::string Nm = B && B->getString("name")
                              ? B->getString("name")->str()
                              : std::string();
        auto It = G.FunctionTable.find(Nm);
        if (It != G.FunctionTable.end() && It->second.FileId != 0) {
          matlab_dbg_add_breakpoint(It->second.FileId, It->second.Line);
          int64_t Nlen = 0;
          const char *Path = matlab_dbg_file_name(It->second.FileId, &Nlen);
          Object Out{{"verified", true},
                     {"line", (int64_t)It->second.Line},
                     {"id", encodeBpId(It->second.FileId,
                                       It->second.Line)}};
          if (Path) {
            Out["source"] = Object{
              {"name", std::string(Path, (size_t)Nlen)},
              {"path", std::string(Path, (size_t)Nlen)},
            };
          }
          Verified.push_back(std::move(Out));
        } else {
          Verified.push_back(Object{
            {"verified", false},
            {"message", "no function named '" + Nm + "' in compiled program"},
          });
        }
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"breakpoints", std::move(Verified)}});
    return true;
  }

  if (*Cmd == "setExceptionBreakpoints") {
    /* Toggle the runtime's "pause on error()" filter. The IDE sends
     * the active filter list; we look for our `error` filter and
     * forward the on/off state to the runtime. Filters we don't
     * recognise are ignored silently — the spec says we MUST NOT
     * fail the request because the IDE may not know which filters
     * apply to the current session. */
    const Array *Filters = Args->getArray("filters");
    bool ErrorOn = false;
    if (Filters) {
      for (const auto &V : *Filters) {
        if (auto S = V.getAsString()) {
          if (*S == "error") ErrorOn = true;
        }
      }
    }
    matlab_dbg_set_pause_on_error(ErrorOn ? 1 : 0);
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"breakpoints", Array{}}});
    return true;
  }

  if (*Cmd == "dataBreakpointInfo") {
    /* The IDE asks "can I set a data breakpoint on this name?"
     * before sending setDataBreakpoints. We accept any plain
     * identifier — the runtime's watch table is keyed by name, so
     * resolution is trivial. The returned `dataId` is the same
     * string we'll receive back in setDataBreakpoints; encoding the
     * name itself keeps the round-trip stable across IDE restarts.
     *
     * `accessTypes` tells the IDE which kinds of watch the user
     * can pick. We expose only "write" because read watchpoints
     * would need every matlab_ws_get_* / frame_local_* call to
     * gate against the watch list — measurable hot-path cost we
     * don't want to pay until someone needs it. */
    auto NameOpt = Args->getString("name");
    if (!NameOpt || NameOpt->empty()) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("dataBreakpointInfo requires a name"));
      return true;
    }
    std::string Nm = NameOpt->str();
    /* Defensive identifier check — same as setVariable. The watch
     * table treats the name as a literal byte string, so a stray
     * `;` or backslash wouldn't cause harm, but rejecting non-
     * identifiers gives a cleaner error than letting the watch
     * silently never match. */
    auto IsIdent = [](const std::string &S) {
      if (S.empty()) return false;
      char c0 = S[0];
      if (!(std::isalpha((unsigned char)c0) || c0 == '_')) return false;
      for (size_t i = 1; i < S.size(); ++i)
        if (!(std::isalnum((unsigned char)S[i]) || S[i] == '_'))
          return false;
      return true;
    };
    if (!IsIdent(Nm)) {
      sendResponse(ReqSeq, *Cmd, true, Object{
        {"dataId", Value(nullptr)},
        {"description", "name is not a plain identifier"},
      });
      return true;
    }
    /* Both read and write access types are supported. Read
     * watchpoints fire on matlab_ws_get_* in JIT'd REPL-mode
     * code; user-function-frame reads (`compute(a, b)` reading
     * `a`) bypass the runtime API — they go through stack slots
     * the JIT loads directly — so a read-watch on a function
     * local is silently invisible. The IDE doesn't have a way
     * to express that scope distinction, so we just advertise
     * the access kinds and document the limitation. */
    Array AccessTypes;
    AccessTypes.push_back(Value("read"));
    AccessTypes.push_back(Value("write"));
    AccessTypes.push_back(Value("readWrite"));
    sendResponse(ReqSeq, *Cmd, true, Object{
      {"dataId", Nm},                     /* dataId == the name */
      {"description", "watch on " + Nm},
      {"accessTypes", std::move(AccessTypes)},
      {"canPersist", true},
    });
    return true;
  }

  if (*Cmd == "setDataBreakpoints") {
    /* Replace-the-whole-list semantics, same as setBreakpoints.
     * The IDE always passes the full active set; we wipe the
     * runtime's watch table and re-add each entry. ID encoding
     * uses a simple hash of the name so cleared-then-readded
     * watches keep stable hitBreakpointIds-style references. */
    matlab_dbg_clear_watchpoints();
    const Array *Bps = Args->getArray("breakpoints");
    Array Verified;
    if (Bps) {
      for (const auto &V : *Bps) {
        const Object *B = V.getAsObject();
        if (!B) continue;
        auto DataId = B->getString("dataId");
        if (!DataId || DataId->empty()) {
          Verified.push_back(Object{
            {"verified", false},
            {"message", "missing dataId"},
          });
          continue;
        }
        std::string Nm = DataId->str();
        auto AT = B->getString("accessType");
        std::string Access = AT ? AT->str() : std::string("write");
        /* Map the DAP accessType string to the runtime's int
         * encoding (0=write, 1=read, 2=readWrite). Unknown values
         * default to write — same behaviour as omitting accessType. */
        int32_t AccessKind;
        if (Access == "read")            AccessKind = 1;
        else if (Access == "readWrite")  AccessKind = 2;
        else                              AccessKind = 0;
        /* Stable id derived from the name. djb2 hash truncated
         * to 31 bits so we never collide with the encodeBpId
         * line-bp space (which uses file_id*1e6 + line). The
         * runtime stores it verbatim and surfaces it on trip. */
        uint32_t H = 5381;
        for (char c : Nm) H = (H * 33u) ^ (uint8_t)c;
        int32_t Id = (int32_t)(H & 0x7FFFFFFFu);
        bool OK = matlab_dbg_add_watchpoint_ex(
            Nm.data(), (int64_t)Nm.size(),
            /*scope=*/0, Id, AccessKind);
        Object Out{{"verified", OK}};
        if (OK) Out["id"] = (int64_t)Id;
        else Out["message"] = "watchpoint table full";
        Verified.push_back(std::move(Out));
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"breakpoints", std::move(Verified)}});
    return true;
  }

  if (*Cmd == "setInstructionBreakpoints") {
    /* Instruction breakpoints address into native code by absolute
     * memory location. The JIT'd image isn't exposed at that
     * granularity — there's no public mapping from line to native
     * PC. Refuse. */
    sendResponse(ReqSeq, *Cmd, false,
                 Value("instruction breakpoints are unsupported: the JIT "
                       "image is not addressable at the instruction level"));
    return true;
  }

  /* --- Evaluation extras ---------------------------------------------- */

  if (*Cmd == "completions") {
    /* Return the union of (a) workspace names whose prefix matches,
     * (b) frame locals (via the supplied frameId, if any), and (c)
     * builtin function names. Ranking: ws + frame names first
     * (user-defined > builtins), each in alphabetical order, capped
     * at 64 entries to keep the response small. */
    auto TextOpt = Args->getString("text");
    auto ColOpt = Args->getInteger("column");
    std::string Text = TextOpt ? TextOpt->str() : std::string();
    /* DAP `column` is 1-based and points one past the last typed
     * char — the prefix is everything from the last non-identifier
     * char up to (column - 1). */
    int64_t Col = ColOpt.value_or((int64_t)Text.size() + 1);
    if (Col < 1) Col = 1;
    if ((size_t)Col > Text.size() + 1) Col = (int64_t)(Text.size() + 1);
    int64_t Start = Col - 1;
    while (Start > 0) {
      char c = Text[(size_t)(Start - 1)];
      if (!(std::isalnum((unsigned char)c) || c == '_')) break;
      --Start;
    }
    std::string Prefix = Text.substr((size_t)Start,
                                      (size_t)(Col - 1 - Start));

    auto FrameIdOpt = Args->getInteger("frameId");
    int RtFrameIdx = -1;
    if (FrameIdOpt) {
      int Total = matlab_dbg_frame_count();
      int Idx = Total - 1 - (int)*FrameIdOpt;
      if (Idx >= 0 && Idx < Total) RtFrameIdx = Idx;
    }

    std::set<std::string> Names;
    int Nws = matlab_dbg_ws_count();
    for (int i = 0; i < Nws; ++i) {
      int64_t L = 0;
      const char *N = matlab_dbg_ws_name(i, &L);
      if (N) Names.insert(std::string(N, (size_t)L));
    }
    if (RtFrameIdx >= 0) {
      int Nf = matlab_dbg_frame_locals_count(RtFrameIdx);
      for (int i = 0; i < Nf; ++i) {
        int64_t L = 0;
        const char *N = matlab_dbg_frame_local_name(RtFrameIdx, i, &L);
        if (N) Names.insert(std::string(N, (size_t)L));
      }
    }
    /* Builtins: a small curated set covers the common REPL surface.
     * Alphabetical order keeps the response stable across runs. */
    static const char *Builtins[] = {
      "abs", "ceil", "clear", "cos", "det", "diag", "disp", "eig",
      "exp", "eye", "fft", "find", "floor", "fprintf", "imag", "inv",
      "isempty", "isequal", "length", "log", "max", "mean", "min",
      "ndims", "numel", "ode23", "ode45", "ones", "prod", "rand", "randn", "real",
      "reshape", "round", "sin", "size", "sort", "sqrt", "sum", "svd",
      "tan", "transpose", "who", "whos", "zeros",
    };
    for (const char *B : Builtins) Names.insert(B);

    Array Targets;
    int Cap = 64;
    for (const std::string &N : Names) {
      if (N.size() < Prefix.size()) continue;
      if (N.compare(0, Prefix.size(), Prefix) != 0) continue;
      Targets.push_back(Object{
        {"label", N},
        {"text", N},
        {"start", (int64_t)Start},
        {"length", (int64_t)Prefix.size()},
      });
      if ((int)Targets.size() >= Cap) break;
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"targets", std::move(Targets)}});
    return true;
  }

  if (*Cmd == "setExpression") {
    /* setVariable mutates by name; setExpression mutates by lvalue
     * expression (e.g. `s.field` or `A(2,3)`). Both share the same
     * REPL-JIT assignment path — we just pass the lvalue through as
     * the LHS without the identifier-only guard setVariable applies.
     * The compiler diagnostics catch malformed lvalues. */
    auto LhsOpt = Args->getString("expression");
    auto ValOpt = Args->getString("value");
    if (!LhsOpt || !ValOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("setExpression requires expression and value"));
      return true;
    }
    std::string Lhs = LhsOpt->str();
    std::string Rhs = ValOpt->str();
    std::string Src = Lhs + " = (" + Rhs + ");";
    int Rc = runReplInput(sharedDapContext(), Src, NextEvalId++);
    if (Rc != 0) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("setExpression failed; see debug console for details"));
      return true;
    }
    /* Read the stored value back by re-evaluating the same lvalue.
     * For `s.field = computeSomething()` the user wants to see the
     * computed result in the watch row, not the literal text
     * "computeSomething()". The readback uses the same
     * `__matlab_dbg_eval = (<lhs>);` wrap as the watch path —
     * arbitrary lvalues are valid expressions, so they round-trip
     * cleanly.
     *
     * If the readback fails (e.g. the LHS is itself only valid as
     * an assignment target — uncommon but possible for some
     * future indexing forms), fall back to echoing the raw RHS
     * rather than failing the whole request: the assignment did
     * land, we just couldn't render the result. */
    const char ReadName[] = "__matlab_dbg_eval";
    std::string ReadSrc = std::string(ReadName) + " = (" + Lhs + ");";
    int ReadRc = runReplInput(sharedDapContext(), ReadSrc, NextEvalId++);
    std::string Display = Rhs;
    int64_t ReadRef = 0;
    int64_t ReadIndexed = 0;
    int64_t ReadNamed = 0;
    std::string ReadType;
    if (ReadRc == 0) {
      int N = matlab_dbg_ws_count();
      int Found = -1, Kind = -1;
      int64_t Elen = (int64_t)(sizeof ReadName - 1);
      for (int i = 0; i < N; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_ws_name(i, &Nlen);
        if (Nlen == Elen &&
            std::memcmp(Nm, ReadName, (size_t)Nlen) == 0) {
          Found = i; Kind = matlab_dbg_ws_kind(i);
          break;
        }
      }
      if (Found >= 0) {
        Display = formatVar(Kind, Found);
        ReadType = typeForVar(Kind,
            Kind == 2 ? matlab_dbg_ws_ptr(Found) : nullptr);
        if (Kind == 2) {
          if (void *obj = matlab_dbg_ws_ptr(Found)) {
            ReadRef = registerObjRef(obj);
            ReadNamed = matlab_dbg_obj_field_count(obj);
          }
        } else if (Kind == 1) {
          auto *M = (struct matlab_mat *)matlab_dbg_ws_ptr(Found);
          if (M && matIsMultiCell(M)) {
            ReadRef = registerMatRef(M);
            ReadIndexed = matIndexedCount(M);
          }
        }
      }
      matlab_ws_clear_one(ReadName, Elen);
    }
    Object Body{{"value", Display},
                {"variablesReference", ReadRef}};
    if (!ReadType.empty()) Body["type"] = ReadType;
    if (ReadIndexed > 0) Body["indexedVariables"] = ReadIndexed;
    if (ReadNamed > 0) Body["namedVariables"] = ReadNamed;
    sendResponse(ReqSeq, *Cmd, true, std::move(Body));
    return true;
  }

  if (*Cmd == "exceptionInfo") {
    /* Surface the most recent matlab error()'s message + frame
     * snapshot for the IDE's exception-info hover. The runtime
     * snapshot is captured at error() time inside matlab_set_error,
     * so this response reflects the failing frame even after the
     * worker has unwound past it. */
    int64_t MsgLen = 0;
    const char *Msg = matlab_dbg_last_error_msg(&MsgLen);
    std::string Body(Msg ? std::string(Msg, (size_t)MsgLen)
                          : std::string("(no error recorded)"));
    int Nf = matlab_err_traceback_count();
    std::string Stack;
    for (int i = 0; i < Nf; ++i) {
      int32_t Fid = 0, Ln = 0;
      const char *FnName = nullptr;
      if (!matlab_err_traceback_at(i, &Fid, &Ln, &FnName)) break;
      int64_t Plen = 0;
      const char *Path = matlab_dbg_file_name(Fid, &Plen);
      char LineBuf[32];
      snprintf(LineBuf, sizeof LineBuf, ":%d", (int)Ln);
      Stack += "  at ";
      Stack += FnName ? FnName : "<frame>";
      Stack += " (";
      Stack += Path ? std::string(Path, (size_t)Plen) : "<file>";
      Stack += LineBuf;
      Stack += ")\n";
    }
    Object Details{{"message", Body}};
    if (!Stack.empty()) Details["stackTrace"] = Stack;
    sendResponse(ReqSeq, *Cmd, true, Object{
      {"exceptionId", "matlab.error"},
      {"description", Body},
      {"breakMode", "always"},
      {"details", std::move(Details)},
    });
    return true;
  }

  /* --- Goto / restart / step-in targets ------------------------------- */

  if (*Cmd == "stepInTargets") {
    /* MATLAB's call sites are simple — at most one user-defined call
     * per statement — so the IDE's "step into a specific call"
     * picker doesn't have anything to choose between. Return one
     * target that maps back to the regular stepIn behaviour. */
    Array Ts;
    Ts.push_back(Object{
      {"id", (int64_t)1},
      {"label", "step into next call"},
    });
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"targets", std::move(Ts)}});
    return true;
  }

  if (*Cmd == "gotoTargets" || *Cmd == "goto") {
    /* Goto requires moving the program counter to an arbitrary line
     * within the current frame — possible in interpreters but our
     * compiled-and-JIT model has no PC manipulation primitive. */
    sendResponse(ReqSeq, *Cmd, false,
                 Value("goto is unsupported: the JIT exposes no "
                       "in-frame PC manipulation primitive"));
    return true;
  }

  if (*Cmd == "restartFrame") {
    /* Restarting a frame would require rolling the runtime's
     * matlab_ws back to the frame's entry state and re-entering —
     * we don't snapshot at function entry, so refusing is the only
     * honest answer. */
    sendResponse(ReqSeq, *Cmd, false,
                 Value("restartFrame is unsupported: the runtime does not "
                       "snapshot per-frame workspace at function entry"));
    return true;
  }

  if (*Cmd == "restart") {
    /* Per the DAP spec, the canonical implementation is to send a
     * `terminated` event with `restart: true` and let the client
     * follow up with a fresh `launch`. That keeps the
     * tear-down/rebuild logic in one place (the launch handler)
     * instead of duplicating compileProgram + worker spawn here. */
    matlab_dbg_resume(STOP);
    sendResponse(ReqSeq, *Cmd, true, Object{});
    sendEvent("terminated", Object{{"restart", true}});
    return true;
  }

  /* --- Reverse stepping ---------------------------------------------- */

  if (*Cmd == "stepBack") {
    /* Pop one statement's worth of undo records from the runtime
     * log, applying each in reverse to revert variable writes.
     * The runtime returns the resume line (or an irreversible-op
     * message). We mirror the forward-step UX: emit a `continued`
     * event acknowledging the move, then a `stopped` event with
     * reason="step" at the rewound line so the IDE highlights it.
     *
     * If the log is exhausted (n_undo == 0), respond with
     * success=true but emit a `stopped` reason="step" at the
     * current line so the IDE doesn't hang on a missing event. */
    int32_t Fid = 0, Ln = 0;
    char Msg[256];
    Msg[0] = '\0';
    int Rc = matlab_dbg_step_back(&Fid, &Ln, Msg, sizeof Msg);
    if (Rc == -1) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value(std::string("stepBack: ") + Msg));
      return true;
    }
    sendResponse(ReqSeq, *Cmd, true, Object{});
    sendEvent("continued",
              Object{{"threadId", (int64_t)1},
                     {"allThreadsContinued", true}});
    if (Rc == 1) {
      sendEvent("stopped", Object{
        {"reason", "step"},
        {"threadId", (int64_t)1},
        {"allThreadsStopped", true},
        {"line", (int64_t)Ln},
      });
    } else {
      /* Log was empty — we've rewound past the very first
       * statement. Emit reason="entry" with a description so the
       * IDE renders the stop with the program-start glyph rather
       * than a generic step. */
      sendEvent("stopped", Object{
        {"reason", "entry"},
        {"description", "stepBack: undo log exhausted"},
        {"threadId", (int64_t)1},
        {"allThreadsStopped", true},
      });
    }
    return true;
  }

  if (*Cmd == "reverseContinue") {
    /* Spec: "reverse-continue back to a breakpoint, exception, or
     * the program start". Walk stepBack until one of:
     *   - the rewound (file_id, line) matches an active breakpoint
     *     -> stop with reason="breakpoint" + hitBreakpointIds
     *   - stepBack returns Rc=-1 (irreversible op marker)
     *     -> stop with reason="exception" + description
     *   - stepBack returns Rc=0 (log exhausted)
     *     -> stop with reason="entry" + description
     *   - safety cap hit (10k iterations) — defensive against a
     *     pathological undo log
     *
     * The bp scan uses matlab_dbg_breakpoint_at to read each
     * (file_id, line) directly. Linear over n_bp on every rewound
     * line; n_bp is small in practice. */
    sendResponse(ReqSeq, *Cmd, true, Object{});
    sendEvent("continued",
              Object{{"threadId", (int64_t)1},
                     {"allThreadsContinued", true}});
    constexpr int RcBpHit = 1, RcIrrev = -1, RcEmpty = 0;
    constexpr int SafetyCap = 10000;
    for (int iter = 0; iter < SafetyCap; ++iter) {
      int32_t Fid = 0, Ln = 0;
      char Msg[256];
      Msg[0] = '\0';
      int Rc = matlab_dbg_step_back(&Fid, &Ln, Msg, sizeof Msg);
      if (Rc == RcBpHit) {
        /* Did we land on a bp line? Walk every active bp and
         * compare; first match wins. */
        for (int i = 0;; ++i) {
          int32_t BpFid = 0, BpLn = 0;
          if (!matlab_dbg_breakpoint_at(i, &BpFid, &BpLn)) break;
          if (BpFid == Fid && BpLn == Ln) {
            Object Body{
              {"reason", "breakpoint"},
              {"threadId", (int64_t)1},
              {"allThreadsStopped", true},
              {"line", (int64_t)Ln},
            };
            Array Ids;
            Ids.push_back(encodeBpId(BpFid, BpLn));
            Body["hitBreakpointIds"] = std::move(Ids);
            sendEvent("stopped", Value(std::move(Body)));
            return true;
          }
        }
        /* No bp hit — keep walking back. */
        continue;
      }
      if (Rc == RcIrrev) {
        sendEvent("stopped", Object{
          {"reason", "exception"},
          {"description", std::string(Msg)},
          {"threadId", (int64_t)1},
          {"allThreadsStopped", true},
        });
        return true;
      }
      /* Rc == RcEmpty: log exhausted; stop at program start. */
      (void)RcEmpty;
      sendEvent("stopped", Object{
        {"reason", "entry"},
        {"description", "reverseContinue: undo log exhausted"},
        {"threadId", (int64_t)1},
        {"allThreadsStopped", true},
      });
      return true;
    }
    /* Safety cap exceeded — emit a stopped event so the IDE
     * doesn't hang waiting on us. */
    sendEvent("stopped", Object{
      {"reason", "step"},
      {"description", "reverseContinue: safety cap reached"},
      {"threadId", (int64_t)1},
      {"allThreadsStopped", true},
    });
    return true;
  }
  if (*Cmd == "readMemory") {
    /* Decode the memoryReference back to a buffer pointer and read
     * `count` bytes starting at `offset`. The buffer must have been
     * registered via registerMemRegion (matrix data buffers are the
     * only thing we hand out today) — this gates the read against
     * a known size so a malformed request can't walk past the end.
     *
     * Per DAP spec, the response carries:
     *   - address: the requested memoryReference (echoed back)
     *   - data: base64 of the bytes actually read
     *   - unreadableBytes: count we couldn't satisfy (clipped at
     *     the buffer end)
     * IDEs use the truncation field to render "..." past the end. */
    auto MemRefOpt = Args->getString("memoryReference");
    auto OffsetOpt = Args->getInteger("offset");
    auto CountOpt  = Args->getInteger("count");
    if (!MemRefOpt || !CountOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("readMemory requires memoryReference and count"));
      return true;
    }
    void *Base = parseMemRef(MemRefOpt->str());
    const MemRegion *R = lookupMemRegion(Base);
    if (!R) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("memoryReference does not point at a registered "
                         "buffer (only matrix data buffers are exposed)"));
      return true;
    }
    int64_t Offset = OffsetOpt.value_or(0);
    int64_t Count  = *CountOpt;
    if (Offset < 0 || Count < 0) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("readMemory offset/count must be non-negative"));
      return true;
    }
    /* Cap reads at 1MB so a runaway request can't allocate gigs
     * of base64. The IDE retries with smaller chunks if it really
     * wants more — the memory-view widgets all do this anyway. */
    constexpr int64_t MaxRead = 1024 * 1024;
    if (Count > MaxRead) Count = MaxRead;
    int64_t Start = Offset;
    int64_t End = Offset + Count;
    int64_t Unreadable = 0;
    if (Start > R->Bytes) { Start = R->Bytes; Unreadable = Count; }
    if (End > R->Bytes) {
      Unreadable += End - R->Bytes;
      End = R->Bytes;
    }
    int64_t Avail = End - Start;
    if (Avail < 0) Avail = 0;
    std::string Data = b64Encode(
        (const uint8_t *)R->Ptr + Start, (size_t)Avail);
    Object Body{
      {"address", MemRefOpt->str()},
      {"data", std::move(Data)},
    };
    if (Unreadable > 0) Body["unreadableBytes"] = Unreadable;
    sendResponse(ReqSeq, *Cmd, true, std::move(Body));
    return true;
  }

  if (*Cmd == "writeMemory") {
    /* Inverse of readMemory. Same registration check — only buffers
     * we previously handed out via memoryReference are writable. The
     * IDE sends `data` as base64 plus an offset; we decode and
     * memcpy into the buffer (clipped at the buffer end so a long
     * write can't smash adjacent state). */
    auto MemRefOpt = Args->getString("memoryReference");
    auto OffsetOpt = Args->getInteger("offset");
    auto DataOpt   = Args->getString("data");
    if (!MemRefOpt || !DataOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("writeMemory requires memoryReference and data"));
      return true;
    }
    void *Base = parseMemRef(MemRefOpt->str());
    const MemRegion *R = lookupMemRegion(Base);
    if (!R) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("memoryReference does not point at a registered "
                         "buffer"));
      return true;
    }
    int64_t Offset = OffsetOpt.value_or(0);
    if (Offset < 0 || Offset > R->Bytes) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("writeMemory offset out of range"));
      return true;
    }
    auto Bytes = b64Decode(DataOpt->str());
    int64_t Avail = R->Bytes - Offset;
    int64_t N = (int64_t)Bytes.size();
    int64_t BytesWritten = N <= Avail ? N : Avail;
    int64_t BytesIgnored = N - BytesWritten;
    if (BytesWritten > 0)
      std::memcpy((uint8_t *)R->Ptr + Offset, Bytes.data(),
                  (size_t)BytesWritten);
    Object Body{{"bytesWritten", BytesWritten}};
    if (BytesIgnored > 0) Body["offset"] = (int64_t)0;
    sendResponse(ReqSeq, *Cmd, true, std::move(Body));
    return true;
  }

  if (*Cmd == "disassemble") {
    /* Walk JIT-emitted machine code instruction-by-instruction
     * using the host triple's MCDisassembler. The IDE supplies
     * a memoryReference (must be JIT-emitted code — we accept the
     * `main` entry point we cached, plus any pointer the IDE has
     * seen via a prior disassemble response) plus an instruction
     * count. We disassemble forward from there until count is met
     * or the next instruction fails to decode (we fall back to a
     * `.byte` row in that case so the response stays well-formed).
     *
     * No bounds-checking against a "code region table" the way
     * readMemory uses MemRegions — we don't track JIT'd code
     * segment extents on the server side, so the IDE has to be
     * sensible about its memoryReference. The disassembler will
     * eventually fail gracefully on garbage bytes. */
    DisasmHolder &H = disasmHolder();
    if (!H.Available) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("disassembler unavailable: " + H.ErrMsg));
      return true;
    }
    auto MRefOpt = Args->getString("memoryReference");
    auto CountOpt = Args->getInteger("instructionCount");
    if (!CountOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("disassemble requires instructionCount"));
      return true;
    }
    /* Default to the JIT main entry point when memoryReference is
     * empty or missing — matches what users expect from a "show me
     * the code" request without prior context. */
    void *Base = nullptr;
    if (MRefOpt && !MRefOpt->empty()) Base = parseMemRef(MRefOpt->str());
    if (!Base) Base = G.MainAddr;
    if (!Base) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("disassemble: no memoryReference and the JIT "
                         "main entry isn't resolved yet (worker hasn't "
                         "started)"));
      return true;
    }
    int64_t Offset = Args->getInteger("offset").value_or(0);
    int64_t InstrOffset =
        Args->getInteger("instructionOffset").value_or(0);
    int64_t Count = *CountOpt;
    if (Count <= 0 || Count > 4096) Count = 4096;

    auto *Cursor = (const uint8_t *)Base + Offset;
    /* Forward-decode-then-skip is the cheapest way to honour
     * instructionOffset: the disassembler is the source of truth
     * for instruction lengths so we can't pre-compute a stride. */
    Array Instrs;
    auto emitInstr = [&](uint64_t Addr, llvm::ArrayRef<uint8_t> Bytes,
                         const std::string &Text) {
      char AddrBuf[32];
      snprintf(AddrBuf, sizeof AddrBuf, "0x%llx",
               (unsigned long long)Addr);
      /* Convert stack buffers to std::string before stuffing them
       * into the Object literal. llvm::json::Value's `const char *`
       * brace-init overload picks StringRef (no copy), and the
       * stack buffer goes away when this lambda returns —
       * serialising later reads garbage. The std::string overload
       * does copy. */
      std::string AddrStr(AddrBuf);
      std::string ByteStr;
      for (size_t i = 0; i < Bytes.size(); ++i) {
        char B[4];
        snprintf(B, sizeof B, "%02x", Bytes[i]);
        if (i) ByteStr += ' ';
        ByteStr += B;
      }
      Instrs.push_back(Object{
        {"address", std::move(AddrStr)},
        {"instructionBytes", std::move(ByteStr)},
        {"instruction", Text},
      });
    };
    auto stepOne = [&](const uint8_t *&P, bool DoEmit) -> bool {
      llvm::MCInst Inst;
      uint64_t Sz = 0;
      llvm::ArrayRef<uint8_t> View(P, /*max-x86-insn=*/15);
      auto Result = H.Dis->getInstruction(Inst, Sz, View,
                                            (uint64_t)(uintptr_t)P,
                                            llvm::nulls());
      if (Result == llvm::MCDisassembler::Success && Sz > 0) {
        if (DoEmit) {
          std::string TextBuf;
          llvm::raw_string_ostream TS(TextBuf);
          H.Printer->printInst(&Inst, (uint64_t)(uintptr_t)P, "",
                                *H.STI, TS);
          TS.flush();
          /* Trim leading whitespace the printer's tab-prefix produces. */
          size_t s = 0;
          while (s < TextBuf.size() &&
                 (TextBuf[s] == ' ' || TextBuf[s] == '\t')) ++s;
          emitInstr((uint64_t)(uintptr_t)P,
                    llvm::ArrayRef<uint8_t>(P, (size_t)Sz),
                    TextBuf.substr(s));
        }
        P += Sz;
        return true;
      }
      /* Decode failed — emit one .byte row so the IDE can still
       * render something, and step forward by 1 to recover.
       * Stops the response from collapsing to "everything failed"
       * on a single un-decoded byte. */
      if (DoEmit)
        emitInstr((uint64_t)(uintptr_t)P,
                  llvm::ArrayRef<uint8_t>(P, 1),
                  ".byte (decode failed)");
      P += 1;
      return false;
    };
    /* Skip InstrOffset instructions before emitting (positive only;
     * negative offsets would need a backward-decoder which is
     * non-trivial on variable-length archs — refuse cleanly). */
    if (InstrOffset < 0) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("disassemble: negative instructionOffset is "
                         "unsupported (variable-length arch)"));
      return true;
    }
    for (int64_t i = 0; i < InstrOffset; ++i) stepOne(Cursor, false);
    for (int64_t i = 0; i < Count; ++i) stepOne(Cursor, true);
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"instructions", std::move(Instrs)}});
    return true;
  }

  if (*Cmd == "locations") {
    /* `locations` maps a memoryReference back to a Source +
     * (line, column). We don't maintain a PC -> .m line table, so
     * this stays refused. The DWARF emitted by `-emit-llvm -g`
     * covers the native-debugging case for users who need that
     * mapping. */
    sendResponse(ReqSeq, *Cmd, false,
                 Value("locations is unsupported: no PC -> .m source "
                       "mapping is maintained for JIT'd code"));
    return true;
  }

  /* `continued` events let adapters that resume the worker out-of-band
   * (e.g. via a remote-restart UX) stay in sync. We emit one for every
   * resume request even though the spec says we MAY skip it for
   * client-initiated resumes — emitting unconditionally keeps the
   * `stopped` ↔ `continued` ordering symmetric and matches what
   * VS Code's debug UI prefers. allThreadsContinued is true because
   * MATLAB execution is single-threaded. */
  auto emitContinued = [&] {
    sendEvent("continued",
              Object{{"threadId", (int64_t)1},
                     {"allThreadsContinued", true}});
  };

  /* Forward step in a rewound state: walk the recorded future
   * via matlab_dbg_step_forward_redo instead of waking the JIT.
   * The JIT is parked one statement past the rewound caret;
   * resuming it directly would skip the rewound region (e.g.
   * stepBack to line 17 → next lands at line 20 with line 19's
   * writes applied, since that's where the JIT actually is).
   * Returns:
   *    1 → landed on a same-frame boundary; emit stopped event,
   *        return true so the handler is done.
   *    0 → caught up to the JIT's parked position; caller must
   *        fall through to a normal matlab_dbg_resume(action).
   *   -1 → hit an irreversible-op marker; surface the runtime's
   *        message via the response and emit stopped at the
   *        prior caret with reason="exception".
   *
   * For DAP `continue`: we loop redo-step until caught up (the
   * full recorded future re-applies, no per-line bp checks
   * during replay), then resume the JIT normally — the JIT will
   * hit the next live bp from its parked position onward. */
  auto emitStoppedAtRedo = [&](int32_t Ln) {
    sendEvent("stopped", Object{
      {"reason", "step"},
      {"threadId", (int64_t)1},
      {"allThreadsStopped", true},
      {"line", (int64_t)Ln},
    });
  };

  if (*Cmd == "continue") {
    /* Drain the redo log first so a continue after a stepBack
     * gets the user back to live JIT execution before resuming. */
    while (matlab_dbg_is_rewound()) {
      int32_t Fid = 0, Ln = 0;
      char Msg[256]; Msg[0] = '\0';
      int Rc = matlab_dbg_step_forward_redo(&Fid, &Ln, Msg, sizeof Msg);
      if (Rc == 0) break; /* caught up */
      if (Rc == -1) {
        /* Hit an irreversible marker: stop here with the runtime's
         * message and let the user decide to stepBack or restart. */
        sendResponse(ReqSeq, *Cmd, true,
                     Object{{"allThreadsContinued", true}});
        emitContinued();
        sendEvent("stopped", Object{
          {"reason", "exception"},
          {"description", std::string(Msg)},
          {"threadId", (int64_t)1},
          {"allThreadsStopped", true},
          {"line", (int64_t)Ln},
        });
        return true;
      }
      /* Rc == 1: a same-frame boundary. Keep replaying — the
       * user asked for `continue`, not `next`. The replay does
       * NOT re-trigger breakpoints on already-recorded lines;
       * once we're caught up the JIT will hit the next live bp. */
    }
    G.StepOverBlockId.clear();
    matlab_dbg_resume(CONTINUE);
    nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"allThreadsContinued", true}});
    emitContinued();
    return true;
  }
  if (*Cmd == "next") {
    /* Serialise rapid-fire next clicks against monitor delivery
     * so each click produces exactly one stopped event. See
     * waitForStepReady above. If the wait fails (worker exited
     * or timeout) we still send a success response — the IDE will
     * notice via the missing stopped / via a `terminated` event. */
    waitForStepReady();
    if (matlab_dbg_is_rewound()) {
      int32_t Fid = 0, Ln = 0;
      char Msg[256]; Msg[0] = '\0';
      int Rc = matlab_dbg_step_forward_redo(&Fid, &Ln, Msg, sizeof Msg);
      if (Rc == 1) {
        sendResponse(ReqSeq, *Cmd, true, Object{});
        emitContinued();
        emitStoppedAtRedo(Ln);
        return true;
      }
      if (Rc == -1) {
        sendResponse(ReqSeq, *Cmd, false,
                     Value(std::string("next: ") + Msg));
        return true;
      }
      /* Rc == 0: caught up. Fall through to JIT resume. */
    }
    /* Phase 8b: snapshot the current block id BEFORE issuing
     * STEP_OVER. The monitor uses this to suppress per-statement
     * stops that land in the same block (e.g. an `expression`
     * block parsing to two Stmts produces two hooks but should
     * present as one step). Empty for `.m` programs. */
    G.StepOverBlockId.clear();
    if (!G.BlockByLine.empty()) {
      int32_t Fid = 0, Ln = 0;
      matlab_dbg_get_pause(&Fid, &Ln);
      int64_t Key = (static_cast<int64_t>((uint32_t)Fid) << 32) |
                    static_cast<int64_t>((uint32_t)Ln);
      auto It = G.BlockByLine.find(Key);
      if (It != G.BlockByLine.end()) G.StepOverBlockId = It->second;
    }
    matlab_dbg_resume(STEP_OVER); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    emitContinued();
    return true;
  }
  if (*Cmd == "stepIn") {
    /* See waitForStepReady above — same rationale as `next`. */
    waitForStepReady();
    if (matlab_dbg_is_rewound()) {
      int32_t Fid = 0, Ln = 0;
      char Msg[256]; Msg[0] = '\0';
      int Rc = matlab_dbg_step_forward_redo(&Fid, &Ln, Msg, sizeof Msg);
      if (Rc == 1) {
        sendResponse(ReqSeq, *Cmd, true, Object{});
        emitContinued();
        emitStoppedAtRedo(Ln);
        return true;
      }
      if (Rc == -1) {
        sendResponse(ReqSeq, *Cmd, false,
                     Value(std::string("stepIn: ") + Msg));
        return true;
      }
    }
    /* Phase 8b: stepIn / stepOut deliberately keep per-statement
     * granularity. The block-skip behaviour is opt-in via STEP_OVER
     * only — stepIn is for descending into a callee, stepOut for
     * leaving the current frame; in both cases the user wants to
     * see every statement they cross. */
    G.StepOverBlockId.clear();
    matlab_dbg_resume(STEP_IN); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    emitContinued();
    return true;
  }
  if (*Cmd == "stepOut") {
    /* See waitForStepReady above — same rationale as `next`. */
    waitForStepReady();
    if (matlab_dbg_is_rewound()) {
      int32_t Fid = 0, Ln = 0;
      char Msg[256]; Msg[0] = '\0';
      int Rc = matlab_dbg_step_forward_redo(&Fid, &Ln, Msg, sizeof Msg);
      if (Rc == 1) {
        sendResponse(ReqSeq, *Cmd, true, Object{});
        emitContinued();
        emitStoppedAtRedo(Ln);
        return true;
      }
      if (Rc == -1) {
        sendResponse(ReqSeq, *Cmd, false,
                     Value(std::string("stepOut: ") + Msg));
        return true;
      }
    }
    G.StepOverBlockId.clear();
    matlab_dbg_resume(STEP_OUT); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    emitContinued();
    return true;
  }

  if (*Cmd == "pause") {
    /* Ask the runtime to stop at the next hook. */
    matlab_dbg_resume(STEP_IN); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    return true;
  }

  /* Lifecycle teardown. The DAP spec separates `terminate` (graceful:
   * ask the debuggee to wind down, with a chance to restart) from
   * `disconnect` (forceful: detach and exit). We honour both:
   *
   *   - `terminate` asks the runtime to stop, sends a `terminated`
   *     event, and keeps the DAP server loop alive so the client may
   *     follow up with `restart` or `disconnect`. The
   *     `terminateDebuggee` arg on `disconnect` (DAP default = true
   *     for launch sessions) is unused — we always stop the worker.
   *
   *   - `disconnect` stops the worker AND exits the request loop, so
   *     the matlabc process winds down. Matches the behaviour the
   *     test suite already relied on. */
  if (*Cmd == "terminate" || *Cmd == "terminateThreads") {
    matlab_dbg_resume(STOP);
    sendResponse(ReqSeq, *Cmd, true, Object{});
    sendEvent("terminated");
    return true;
  }
  if (*Cmd == "disconnect") {
    matlab_dbg_resume(STOP);
    sendResponse(ReqSeq, *Cmd, true, Object{});
    return false; /* tell the loop to exit */
  }

  /* Unknown: return success with empty body so the client doesn't
   * hang waiting on a mandatory-but-unimplemented request. DAP
   * doesn't define a MethodNotFound the same way LSP does. */
  sendResponse(ReqSeq, *Cmd, true, Object{});
  return true;
}

int runDap(const std::string &CLIPath) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  /* The disassembler init is deferred to first use (see
   * disasmHolder() below). On some LLVM builds calling it during
   * startup interacts badly with MLIR's already-completed target
   * registration and trips a SIGTRAP; deferring keeps startup
   * clean and only pays the init cost when a `disassemble`
   * request actually arrives. */

  /* Redirect stdout to a pipe so matlab_disp_* etc. from the JIT'd
   * program don't corrupt the DAP channel. */
  int Pipe[2];
  if (pipe(Pipe) != 0) {
    std::cerr << "matlabc -dap: pipe() failed\n";
    return 1;
  }
  OriginalStdoutFd = dup(STDOUT_FILENO);
  if (OriginalStdoutFd < 0) {
    std::cerr << "matlabc -dap: dup(stdout) failed\n";
    return 1;
  }
  if (dup2(Pipe[1], STDOUT_FILENO) < 0) {
    std::cerr << "matlabc -dap: dup2 failed\n";
    return 1;
  }
  close(Pipe[1]);
  DebuggeeOutFd = Pipe[0];
  /* The captured stdout is now a pipe, not a tty.  glibc (Linux)
   * full-buffers a non-tty stdout, so JIT'd `disp()` / `fprintf` output
   * would sit in the 4 KB buffer and never reach the DAP reader until a
   * fill or process exit — the REPL `evaluate` output events time out on
   * Linux (macOS libc flushes more eagerly, so it passed locally).  Force
   * line-buffering so each printed line surfaces to the pipe promptly. */
  setvbuf(stdout, nullptr, _IOLBF, 0);

  /* Same redirect for stderr. The DAP server's own diagnostics still
   * need an unredirected stderr — std::cerr lines emitted before
   * runDap reach the parent process directly. After this point any
   * fprintf(stderr, ...) goes through our pipe and surfaces as
   * `output` events with `category: "stderr"`. */
  int ErrPipe[2];
  if (pipe(ErrPipe) != 0) {
    std::cerr << "matlabc -dap: stderr pipe() failed\n";
    return 1;
  }
  OriginalStderrFd = dup(STDERR_FILENO);
  if (OriginalStderrFd < 0) {
    std::cerr << "matlabc -dap: dup(stderr) failed\n";
    return 1;
  }
  if (dup2(ErrPipe[1], STDERR_FILENO) < 0) {
    /* Best-effort: if redirect fails, just log and proceed without
     * stderr capture — the rest of the server is still functional. */
    (void)!write(OriginalStderrFd, "matlabc -dap: stderr dup2 failed\n", 33);
    close(ErrPipe[0]);
    close(ErrPipe[1]);
  } else {
    close(ErrPipe[1]);
    DebuggeeErrFd = ErrPipe[0];
  }

  /* Spawn the stderr-pipe reader BEFORE we accept any DAP requests so
   * compile-time diagnostics emitted during `launch` (which runs
   * before configurationDone) make it back to the IDE as `output`
   * events and to the parent's stderr — instead of sitting in the
   * pipe buffer until the worker thread is started. Without this
   * early spawn a `failed to compile program` response was effectively
   * undebuggable because the actual MLIR / verifier error never
   * surfaced. */
  if (DebuggeeErrFd >= 0) {
    pthread_t EarlyErrRdr;
    pthread_create(&EarlyErrRdr, nullptr, stderrReaderMain, nullptr);
    pthread_detach(EarlyErrRdr);
  }

  G.ProgramPath = CLIPath;
  std::ios::sync_with_stdio(false);

  bool Debug = getenv("MATLABC_DAP_TRACE") != nullptr;
  while (true) {
    auto Msg = readFrame();
    if (!Msg) break;
    if (Msg->empty()) continue;
    if (Debug) std::fprintf(stderr, "[server] recv: %s\n",
                             Msg->substr(0, 120).c_str());
    auto Parsed = llvm::json::parse(*Msg);
    if (!Parsed) { llvm::consumeError(Parsed.takeError()); continue; }
    const Object *Root = Parsed->getAsObject();
    if (!Root) continue;
    auto Ty = Root->getString("type");
    if (!Ty || *Ty != "request") continue;
    if (!handleRequest(*Root)) break;
  }

  /* #112: before returning (the caller hard-exits the process), make
   * sure the worker thread is no longer touching the ExecutionEngine.
   * The server loop can exit on stdin EOF or `disconnect` while the
   * worker is still inside `ExecutionEngine::lookup("main")` (lazy ORC
   * materialization) or running the JIT'd program. Returning straight
   * into process teardown then races the worker against engine /
   * MLIR-context destruction — the SIGSEGV in `ExecutionEngine::lookup`
   * / `~ExecutionEngine` that this issue reported.
   *
   * Ask the runtime to stop (unblocks a stopOnEntry / breakpoint pause)
   * and wait, bounded, for the worker to set WorkerExited. The wait is
   * capped so a pathological program that ignores STOP can't hang the
   * adapter shutdown — in that case we fall through to the caller's
   * hard-exit as before. */
  pthread_mutex_lock(&G.Mu);
  bool MustWait = G.WorkerStarted && !G.WorkerExited;
  pthread_mutex_unlock(&G.Mu);
  if (MustWait) {
    matlab_dbg_resume(STOP);
    struct timespec Deadline;
    clock_gettime(CLOCK_REALTIME, &Deadline);
    Deadline.tv_sec += 5;
    pthread_mutex_lock(&G.Mu);
    while (!G.WorkerExited) {
      if (pthread_cond_timedwait(&G.Cv, &G.Mu, &Deadline) == ETIMEDOUT)
        break;
    }
    pthread_mutex_unlock(&G.Mu);
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// mflowLink — `matlabc -simulate --dap model.mflow`
//
// Tier-D DAP server for the signal-flow simulation lane
// (docs/mflow_link_roadmap.md §8 + §10). Reuses the JSON-RPC framing
// (`readFrame` / `writeFrame` / `sendResponse` / `sendEvent`) but
// dispatches a completely separate request set on top of an in-process
// `MflowLinkSim`. The matlab-program DAP path (runDap above) is left
// untouched.
//
// Three things that make this loop simpler than runDap:
//   1. No JIT'd child process → no pipe + dup2 dance. We still grab
//      OriginalStdoutFd so writeFrame's `write(OriginalStdoutFd, ...)`
//      reaches the DAP client.
//   2. The simulator is synchronous; pausing means "the request loop
//      is waiting for the next request". No worker thread, no
//      matlab_dbg_resume — `continue` just runs the loop in this
//      thread.
//   3. Source mapping is single-frame: every stopped event reports a
//      single virtual frame named "mflowLink" with the current
//      simulation time as `line`. The IDE renders the active block
//      via `simulationActiveBlock` events instead.
//===----------------------------------------------------------------------===//

namespace {
// `stopped` reason → DAP body. Convenience over building the Object
// inline at every call site.
llvm::json::Value mflStoppedBody(const char *Reason,
                                 const char *Description = nullptr) {
  llvm::json::Object O{
    {"reason", Reason},
    {"threadId", 1},
    {"allThreadsStopped", true},
  };
  if (Description) O["description"] = Description;
  return llvm::json::Value(std::move(O));
}

void mflEmitTimeEvent(double T, size_t MajorStep) {
  sendEvent("simulationTime",
            llvm::json::Object{{"t", T},
                               {"majorStep", static_cast<int64_t>(MajorStep)}});
}

void mflEmitSampleEvents(const matlab::flowchart::MflowLinkSim &Sim) {
  double T = Sim.currentTime();
  for (auto &P : Sim.currentLoggedOutputs()) {
    sendEvent("signalSample",
              llvm::json::Object{{"blockId", P.first},
                                 {"t", T},
                                 {"value", P.second}});
  }
}

// Drain the simulator's zero-crossing queue (§10 zeroCrossing event).
// Called after every forward-progress operation (continue, stepMajor,
// stepBlock) so the IDE's active-block halo / log can react.
void mflEmitZeroCrossings(matlab::flowchart::MflowLinkSim &Sim) {
  for (auto &E : Sim.consumeZeroCrossings()) {
    sendEvent("zeroCrossing",
              llvm::json::Object{{"blockId", E.BlockId}, {"t", E.T}});
  }
}

// Drain the simulator's algebraic-loop failure queue. Tier-I Item-2
// runs Newton / trust-region iterations on direct-feedthrough cycles;
// when a cycle fails to converge within the 50-iteration cap, the
// runtime records the offending member set + simulation time. The
// IDE captures these into MflowLinkSimulation.recentAlgebraicLoopFailures
// and shows a chrome-strip badge. Block indices are translated to the
// IR-side string ids so the IDE doesn't need to know about the
// runtime's internal layout.
void mflEmitAlgebraicLoopFailures(
    matlab::flowchart::MflowLinkSim &Sim,
    const matlab::flowchart::MflowLinkModel &Model) {
  for (auto &F : Sim.consumeAlgebraicLoopFailures()) {
    llvm::json::Array Members;
    for (size_t Idx : F.Members) {
      if (Idx < Model.Blocks.size())
        Members.push_back(Model.Blocks[Idx].Id);
    }
    sendEvent("algebraicLoopFailure",
              llvm::json::Object{
                {"t", F.T},
                {"members", llvm::json::Value(std::move(Members))}});
  }
}

// Tag the IDE's canvas with the cursor's currently-active block.
// Empty id ⇒ no block is highlighted (the cursor is at end-of-step,
// pre-stepBlock or post-major-commit).
void mflEmitActiveBlock(const std::string &BlockId) {
  sendEvent("simulationActiveBlock",
            llvm::json::Object{{"nodeId", BlockId}});
}
} // namespace

int runMflowLinkDap(const std::string &Path) {
  // The simulation runs entirely in-process — no child stdout to
  // shield — but writeFrame still routes everything through
  // OriginalStdoutFd, so capture stdout here. (Pipe redirection of
  // stdin/stdout/stderr is not needed: the simulator never writes
  // to stdout, and the runtime has no printf-style output.)
  OriginalStdoutFd = dup(STDOUT_FILENO);
  if (OriginalStdoutFd < 0) {
    std::cerr << "matlabc -simulate --dap: dup(stdout) failed\n";
    return 1;
  }

  matlab::SourceManager FlowSM;
  matlab::DiagnosticEngine FlowDiag(FlowSM);
  auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Path, FlowDiag);
  if (!Doc) {
    FlowDiag.printAll();
    return 1;
  }
  if (!Doc->isSignalFlow()) {
    std::cerr << Path
              << ": -simulate --dap requires a signal-flow .mflow\n";
    return 1;
  }
  auto Model = matlab::flowchart::lowerSignalFlow(*Doc, FlowDiag);
  FlowDiag.printAll();
  if (!Model) return 1;

  matlab::flowchart::MflowLinkSim Sim(*Model);
  Sim.reset();

  bool ConfDone = false;
  bool Debug = getenv("MATLABC_DAP_TRACE") != nullptr;

  //===-------------------------------------------------------------===//
  // Tier-F — DAP-server-local breakpoint state.
  //
  // Per-session: replaced wholesale by each `setTimeBreakpoints` /
  // `setSignalBreakpoints` request (DAP convention is "this list is
  // now the entire set"). `Hit` is sticky-per-pass — a fired
  // breakpoint won't fire again until the simulator is reset or the
  // user disarms it by sending a new list.
  //===-------------------------------------------------------------===//
  struct TimeBP { double T; bool Hit; };
  std::vector<TimeBP> TimeBreakpoints_;
  struct SignalBP {
    std::string BlockId;
    std::string Condition; // raw expression; parsed by `signalCondMet`
    bool Hit;
  };
  std::vector<SignalBP> SignalBreakpoints_;

  //===-------------------------------------------------------------===//
  // Tier-F — tiny condition evaluator.
  //
  // Accepts the shapes the IDE roadmap §10 calls out:
  //   value <op> N
  //   abs(value) <op> N
  // where <op> ∈ { >, <, >=, <=, ==, != }. Returns false on any
  // parse failure — the user's `setSignalBreakpoints` list is
  // already ack'd (so the IDE knows we accepted the request), and a
  // mis-parsed expression silently never fires rather than spamming
  // diagnostics.
  //===-------------------------------------------------------------===//
  auto signalCondMet = [](const std::string &Expr, double V) -> bool {
    std::string S = Expr;
    // Strip whitespace.
    S.erase(std::remove_if(S.begin(), S.end(),
                           [](char C) { return C == ' ' || C == '\t'; }),
            S.end());
    // Identify `value` vs `abs(value)`.
    double Lhs;
    size_t Pos;
    if (S.rfind("abs(value)", 0) == 0) {
      Lhs = std::fabs(V);
      Pos = 10;
    } else if (S.rfind("value", 0) == 0) {
      Lhs = V;
      Pos = 5;
    } else {
      return false;
    }
    // Operator.
    std::string Op;
    if (S.compare(Pos, 2, ">=") == 0 || S.compare(Pos, 2, "<=") == 0 ||
        S.compare(Pos, 2, "==") == 0 || S.compare(Pos, 2, "!=") == 0) {
      Op = S.substr(Pos, 2);
      Pos += 2;
    } else if (Pos < S.size() && (S[Pos] == '>' || S[Pos] == '<')) {
      Op = S.substr(Pos, 1);
      Pos += 1;
    } else {
      return false;
    }
    double Rhs;
    try {
      Rhs = std::stod(S.substr(Pos));
    } catch (...) {
      return false;
    }
    if (Op == ">")  return Lhs >  Rhs;
    if (Op == "<")  return Lhs <  Rhs;
    if (Op == ">=") return Lhs >= Rhs;
    if (Op == "<=") return Lhs <= Rhs;
    if (Op == "==") return Lhs == Rhs;
    if (Op == "!=") return Lhs != Rhs;
    return false;
  };

  // Returns the first breakpoint that just fired, formatted as
  // `(reason-description, description)` for the `stopped` event;
  // empty description when nothing fired. Side-effect: marks the
  // matching breakpoint as `Hit` so it doesn't refire on the next
  // step.
  auto checkBreakpoints = [&]() -> std::string {
    // Time breakpoints first — they're cheaper to test and the
    // expected "stop at t = 5s" UX wants priority over signal
    // breakpoints that happen to coincide.
    for (auto &BP : TimeBreakpoints_) {
      if (!BP.Hit && Sim.currentTime() >= BP.T - 1e-9) {
        BP.Hit = true;
        std::ostringstream OS;
        OS << "t=" << BP.T;
        return OS.str();
      }
    }
    auto Outputs = Sim.currentLoggedOutputs();
    for (auto &BP : SignalBreakpoints_) {
      if (BP.Hit) continue;
      double V = 0.0;
      bool Found = false;
      for (auto &P : Outputs)
        if (P.first == BP.BlockId) { V = P.second; Found = true; break; }
      if (!Found) continue;
      if (signalCondMet(BP.Condition, V)) {
        BP.Hit = true;
        std::ostringstream OS;
        OS << BP.BlockId << " " << BP.Condition << " (=" << V << ")";
        return OS.str();
      }
    }
    return std::string{};
  };

  std::ios::sync_with_stdio(false);

  while (true) {
    auto Msg = readFrame();
    if (!Msg) break;
    if (Msg->empty()) continue;
    if (Debug)
      std::fprintf(stderr, "[sim-dap] recv: %s\n",
                   Msg->substr(0, 160).c_str());
    auto Parsed = llvm::json::parse(*Msg);
    if (!Parsed) {
      llvm::consumeError(Parsed.takeError());
      continue;
    }
    const Object *Root = Parsed->getAsObject();
    if (!Root) continue;
    auto Ty = Root->getString("type");
    if (!Ty || *Ty != "request") continue;
    auto Cmd = Root->getString("command");
    if (!Cmd) continue;
    int64_t Seq = Root->getInteger("seq").value_or(0);

    if (*Cmd == "initialize") {
      Object Caps{
        {"supportsConfigurationDoneRequest", true},
        {"supportsStepBack", true},
        {"supportsRestartRequest", true},
        {"supportsTerminateRequest", true},
      };
      sendResponse(Seq, *Cmd, true, Value(std::move(Caps)));
      sendEvent("initialized");
      continue;
    }
    if (*Cmd == "launch" || *Cmd == "attach") {
      sendResponse(Seq, *Cmd, true, Object{});
      continue;
    }
    if (*Cmd == "configurationDone") {
      sendResponse(Seq, *Cmd, true, Object{});
      // Boot stopped at startTime per roadmap §8.
      mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
      mflEmitSampleEvents(Sim);
      sendEvent("stopped", mflStoppedBody("entry"));
      ConfDone = true;
      continue;
    }
    if (*Cmd == "setBreakpoints" || *Cmd == "setExceptionBreakpoints") {
      // Source-file breakpoints don't apply to a block-diagram —
      // ack with an empty `breakpoints` array.
      sendResponse(Seq, *Cmd, true, Object{{"breakpoints", Array{}}});
      continue;
    }
    if (*Cmd == "setTimeBreakpoints") {
      // Tier-F (§10): pause when simulation time crosses any of a
      // user-specified list. Replaces (not appends) — DAP convention.
      Array Out;
      TimeBreakpoints_.clear();
      auto Args = Root->getObject("arguments");
      if (Args) {
        if (auto *Arr = Args->getArray("times")) {
          for (auto &V : *Arr) {
            const Object *BP = V.getAsObject();
            if (!BP) continue;
            std::optional<double> T;
            if (auto N = BP->getNumber("t")) T = *N;
            else if (auto N = BP->getInteger("t")) T = static_cast<double>(*N);
            if (!T) continue;
            TimeBreakpoints_.push_back({*T, /*Hit=*/false});
            Out.push_back(Object{{"verified", true}, {"t", *T}});
          }
        }
      }
      sendResponse(Seq, *Cmd, true,
                   Object{{"breakpoints", Value(std::move(Out))}});
      continue;
    }
    if (*Cmd == "setSignalBreakpoints") {
      // Tier-F (§10): watch a signal output, pause when a condition
      // becomes true. The IDE roadmap's per-edge `breakpoint` schema
      // is still unbuilt, so we accept either the edge form
      // (`{ "edgeId": "e7", "condition": "..." }`) — resolved when
      // edge metadata lands — or a direct `{ "blockId": "b", ... }`
      // form that names the source block of the watched signal.
      Array Out;
      SignalBreakpoints_.clear();
      auto Args = Root->getObject("arguments");
      if (Args) {
        if (auto *Arr = Args->getArray("breakpoints")) {
          for (auto &V : *Arr) {
            const Object *BP = V.getAsObject();
            if (!BP) continue;
            std::string Block, Cond;
            if (auto S = BP->getString("blockId")) Block = std::string(*S);
            else if (auto S = BP->getString("edgeId"))
              Block = std::string(*S); // best-effort until edge meta ships
            if (auto S = BP->getString("condition"))
              Cond = std::string(*S);
            if (Block.empty() || Cond.empty()) continue;
            SignalBreakpoints_.push_back({Block, Cond, /*Hit=*/false});
            Out.push_back(Object{{"verified", true},
                                 {"blockId", Block},
                                 {"condition", Cond}});
          }
        }
      }
      sendResponse(Seq, *Cmd, true,
                   Object{{"breakpoints", Value(std::move(Out))}});
      continue;
    }
    if (*Cmd == "threads") {
      Array Ts{Object{{"id", 1}, {"name", "mflowLink"}}};
      sendResponse(Seq, *Cmd, true,
                   Object{{"threads", Value(std::move(Ts))}});
      continue;
    }
    if (*Cmd == "stackTrace") {
      Array Frames{Object{
        {"id", 1},
        {"name", "mflowLink"},
        {"line", 1},
        {"column", 1},
      }};
      sendResponse(Seq, *Cmd, true,
                   Object{{"stackFrames", Value(std::move(Frames))},
                          {"totalFrames", 1}});
      continue;
    }
    if (*Cmd == "scopes") {
      Array Sc{Object{{"name", "Signals"},
                      {"variablesReference", 1},
                      {"expensive", false}}};
      sendResponse(Seq, *Cmd, true,
                   Object{{"scopes", Value(std::move(Sc))}});
      continue;
    }
    if (*Cmd == "variables") {
      Array Vars;
      Vars.push_back(Object{{"name", "t"},
                            {"value", std::to_string(Sim.currentTime())},
                            {"variablesReference", 0}});
      Vars.push_back(Object{
        {"name", "majorStep"},
        {"value", std::to_string(Sim.majorStepsTaken())},
        {"variablesReference", 0}});
      for (auto &P : Sim.currentLoggedOutputs()) {
        Vars.push_back(Object{{"name", P.first},
                              {"value", std::to_string(P.second)},
                              {"variablesReference", 0}});
      }
      sendResponse(Seq, *Cmd, true,
                   Object{{"variables", Value(std::move(Vars))}});
      continue;
    }
    if (*Cmd == "evaluate") {
      // Tier-F: real expression evaluator. Echo a short status so the
      // IDE's hover / watch panes don't sit on `<no value>`.
      auto Args = Root->getObject("arguments");
      std::string Expr;
      if (Args)
        if (auto S = Args->getString("expression")) Expr = std::string(*S);
      std::string Body = "t=" + std::to_string(Sim.currentTime()) +
                         " majorStep=" + std::to_string(Sim.majorStepsTaken());
      (void)Expr;
      sendResponse(Seq, *Cmd, true,
                   Object{{"result", Body}, {"variablesReference", 0}});
      continue;
    }

    auto stepN = [&](int N) {
      for (int K = 0; K < N; ++K) {
        double H = Sim.stepMajor();
        if (H <= 0.0) break;
      }
      mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
      mflEmitSampleEvents(Sim);
      mflEmitZeroCrossings(Sim);
      mflEmitAlgebraicLoopFailures(Sim, *Model);
      sendEvent("snapshotTaken",
                Object{{"majorStep",
                        static_cast<int64_t>(Sim.majorStepsTaken())},
                       {"depth",
                        static_cast<int64_t>(Sim.snapshotDepth())}});
    };

    if (*Cmd == "continue") {
      // Run to stopTime in this thread (synchronous — the matlabc
      // process is the simulator). Stream the per-step events as we
      // go so the IDE's signal scopes stay live.
      sendResponse(Seq, *Cmd, true,
                   Object{{"allThreadsContinued", true}});
      bool ZCStopped = false;
      std::string BPDesc;     // Tier-F: non-empty when a breakpoint fired
      size_t Before = Sim.majorStepsTaken();
      while (Sim.currentTime() < Model->Solver.StopTime - 1e-12) {
        size_t MStart = Sim.majorStepsTaken();
        double H = Sim.stepMajor();
        if (H <= 0.0 && Sim.majorStepsTaken() == MStart) break;
        // Zero-crossings always surface (no throttle — they're rare
        // by definition and the user wants to see them).
        auto Crossings = Sim.consumeZeroCrossings();
        for (auto &E : Crossings)
          sendEvent("zeroCrossing",
                    Object{{"blockId", E.BlockId}, {"t", E.T}});
        // Drain the Tier-I Item-2 algebraic-loop failure queue.
        // Surface as a custom event AND, on the first failure, flip
        // the continue loop into a stopped(reason="algebraic loop")
        // state so the user has a chance to inspect before time
        // marches on. Same policy as the zero-crossing pause.
        mflEmitAlgebraicLoopFailures(Sim, *Model);
        // Throttle: emit a sample event every 16th step to avoid
        // flooding (§10 — IDE-side throttle target).
        if ((Sim.majorStepsTaken() % 16) == 0) {
          mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
          mflEmitSampleEvents(Sim);
        }
        // Tier-F: pause on the first hit time / signal breakpoint.
        BPDesc = checkBreakpoints();
        if (!BPDesc.empty()) break;
        if (!Crossings.empty()) {
          // Stop on the first crossing inside `continue` so the IDE
          // can land the user there — matches Simulink's "pause on
          // zero-crossing" behaviour. Forward-stepping resumes the
          // run; the IDE controls the policy.
          ZCStopped = true;
          break;
        }
      }
      (void)Before;
      mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
      mflEmitSampleEvents(Sim);
      const char *Reason = !BPDesc.empty() ? "breakpoint"
                            : (ZCStopped   ? "crossing" : "step");
      const char *Default = ZCStopped ? "zero crossing" : "stopTime reached";
      sendEvent("stopped",
                mflStoppedBody(Reason, !BPDesc.empty() ? BPDesc.c_str()
                                                      : Default));
      continue;
    }
    if (*Cmd == "pause") {
      // The loop is already paused between requests; ack and re-stop.
      sendResponse(Seq, *Cmd, true, Object{});
      sendEvent("stopped", mflStoppedBody("pause"));
      continue;
    }
    if (*Cmd == "next" || *Cmd == "stepIn" || *Cmd == "stepMajor") {
      sendResponse(Seq, *Cmd, true, Object{});
      stepN(1);
      sendEvent("stopped", mflStoppedBody("step"));
      continue;
    }
    if (*Cmd == "stepOut") {
      // No call stack to step out of — fall through to a single step.
      sendResponse(Seq, *Cmd, true, Object{});
      stepN(1);
      sendEvent("stopped", mflStoppedBody("step"));
      continue;
    }
    if (*Cmd == "reverseContinue" || *Cmd == "stepBack" ||
        *Cmd == "stepBackMajor") {
      sendResponse(Seq, *Cmd, true, Object{});
      bool Ok = Sim.stepBackMajor();
      mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
      mflEmitSampleEvents(Sim);
      // The cursor is reset at end-of-step after a step-back — clear
      // any IDE-side active-block highlight.
      mflEmitActiveBlock(Sim.activeBlockId());
      const char *Desc = Ok ? "step-back" : "snapshot ring empty";
      sendEvent("stopped", mflStoppedBody("step", Desc));
      continue;
    }
    if (*Cmd == "stepBlock") {
      // Tier-E: advance the cursor one block within the current major
      // step. When the cursor wraps, `Sim.stepBlock()` commits the
      // major step and returns "" — emit the post-commit time/sample
      // events in that case.
      sendResponse(Seq, *Cmd, true, Object{});
      size_t MajorBefore = Sim.majorStepsTaken();
      std::string Active = Sim.stepBlock();
      if (Sim.majorStepsTaken() != MajorBefore) {
        // A major step was committed by this stepBlock call.
        mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
        mflEmitSampleEvents(Sim);
        mflEmitZeroCrossings(Sim);
      }
      mflEmitActiveBlock(Active);
      sendEvent("stopped", mflStoppedBody("step"));
      continue;
    }
    if (*Cmd == "stepBackBlock") {
      // Tier-E: pull the cursor back one block. At cursor 0, this
      // pops a major step from the snapshot ring and lands the
      // cursor at end-of-step.
      sendResponse(Seq, *Cmd, true, Object{});
      size_t MajorBefore = Sim.majorStepsTaken();
      std::string Active = Sim.stepBackBlock();
      if (Sim.majorStepsTaken() != MajorBefore) {
        mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
        mflEmitSampleEvents(Sim);
      }
      mflEmitActiveBlock(Active);
      sendEvent("stopped", mflStoppedBody("step"));
      continue;
    }
    if (*Cmd == "resetSimulation" || *Cmd == "restart") {
      Sim.reset();
      // Re-arm every breakpoint so a resumed run after `restart`
      // can pause at the same instants as before.
      for (auto &BP : TimeBreakpoints_)   BP.Hit = false;
      for (auto &BP : SignalBreakpoints_) BP.Hit = false;
      sendResponse(Seq, *Cmd, true, Object{});
      mflEmitTimeEvent(Sim.currentTime(), Sim.majorStepsTaken());
      mflEmitSampleEvents(Sim);
      sendEvent("stopped", mflStoppedBody("entry", "reset"));
      continue;
    }
    if (*Cmd == "configureSolver") {
      // Tier-F: live tuning of relTol / maxStep. Ack so the IDE pane
      // doesn't error; the live-tuning hook into MflowLinkSim is a
      // small follow-up.
      sendResponse(Seq, *Cmd, true, Object{});
      continue;
    }
    if (*Cmd == "terminate" || *Cmd == "terminateThreads") {
      sendResponse(Seq, *Cmd, true, Object{});
      sendEvent("terminated");
      continue;
    }
    if (*Cmd == "disconnect") {
      sendResponse(Seq, *Cmd, true, Object{});
      break;
    }

    // Unknown request — ack empty so the client doesn't stall.
    (void)ConfDone;
    sendResponse(Seq, *Cmd, true, Object{});
  }
  return 0;
}

/* ===========================================================================
 * mStateflow — `matlabc -simulate --dap model.mflow` for a state-chart
 *
 * Tier 4e of docs/mStateflow_roadmap.md. Reuses the same JSON-RPC
 * framing as `runMflowLinkDap`, namespaces every chart-specific verb
 * under `stateChart/`. The session owns a `ChartInterpreter` — every
 * `stepSuperStep` / `stepTransition` / `emit` mutates real chart
 * state, and the resulting trace is streamed back as
 * `stateChart/{stateEnter, stateExit, transitionFired, ...}` events.
 * Breakpoints set via `stateChart/setStateBreakpoints` /
 * `setTransitionBreakpoints` fire from inside the step and cause a
 * `stopped` event matching DAP convention.
 * ========================================================================= */
int runStateChartDap(const std::string &Path) {
  OriginalStdoutFd = dup(STDOUT_FILENO);
  if (OriginalStdoutFd < 0) {
    std::cerr << "matlabc -simulate --dap: dup(stdout) failed\n";
    return 1;
  }

  matlab::SourceManager FlowSM;
  matlab::DiagnosticEngine FlowDiag(FlowSM);
  auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Path, FlowDiag);
  if (!Doc) { FlowDiag.printAll(); return 1; }
  if (!Doc->isStateChart()) {
    std::cerr << Path
              << ": runStateChartDap requires a state-chart .mflow\n";
    return 1;
  }
  auto Model = matlab::statechart::buildChartModel(*Doc, FlowDiag);
  FlowDiag.printAll();
  if (!Model) return 1;
  const matlab::statechart::Chart *Entry = Model->entryChart();
  if (!Entry && !Model->Charts.empty()) Entry = &Model->Charts.front();
  if (!Entry) {
    std::cerr << Path << ": state-chart document has no charts\n";
    return 1;
  }

  matlab::statechart::ChartInterpreter Interp(*Entry);

  // Stream the result of a step (initialise / super-step / step-
  // transition) as DAP events in fire order. A `stopped` event is
  // emitted on the first Breakpoint trace event so the IDE can
  // freeze the canvas.
  auto streamTrace =
      [&](const std::vector<matlab::statechart::ChartTraceEvent> &Trace) {
        using K = matlab::statechart::ChartTraceEvent::Kind;
        bool StoppedSent = false;
        for (auto &E : Trace) {
          switch (E.K) {
          case K::SuperStepBegin:
            sendEvent("stateChart/superStepBegin",
                      Object{{"iteration", (int64_t)E.Iteration}, {"t", 0.0}});
            break;
          case K::SuperStepEnd:
            sendEvent("stateChart/superStepEnd",
                      Object{{"iteration", (int64_t)E.Iteration},
                             {"quiescent", E.Quiescent},
                             {"t", 0.0}});
            break;
          case K::StateEnter:
            sendEvent("stateChart/stateEnter",
                      Object{{"id", E.Id}, {"t", 0.0}});
            break;
          case K::StateExit:
            sendEvent("stateChart/stateExit",
                      Object{{"id", E.Id}, {"t", 0.0}});
            break;
          case K::TransitionFired: {
            Object Body{
              {"id", E.Id}, {"src", E.Src}, {"dst", E.Dst}, {"t", 0.0}};
            if (!E.EventName.empty()) Body["eventName"] = E.EventName;
            sendEvent("stateChart/transitionFired", std::move(Body));
            break;
          }
          case K::EventBroadcast:
            sendEvent("stateChart/eventBroadcast",
                      Object{{"name", E.Id}, {"t", 0.0}});
            break;
          case K::Breakpoint:
            if (!StoppedSent) {
              sendEvent("stopped",
                        Object{{"reason", "breakpoint"},
                               {"description", E.BreakpointReason + " " +
                                                   E.Id},
                               {"threadId", (int64_t)1}});
              StoppedSent = true;
            }
            break;
          case K::MaxIterations:
            sendEvent("stateChart/maxIterations",
                      Object{{"iteration", (int64_t)E.Iteration}, {"t", 0.0}});
            break;
          }
        }
      };

  // Serialised snapshot blob format: simple text "key=value\n" lines —
  // enough to round-trip Regions + Locals + History between save and
  // restore. The C-side ring stores it as bytes; the chart-DAP side
  // re-parses on restore.
  auto serializeSnapshot =
      [&](const matlab::statechart::ChartInterpreter::Snapshot &S) {
        std::string Out;
        Out += "R\n";
        for (auto &P : S.Regions)
          Out += "r " + P.first + " " + P.second + "\n";
        Out += "L\n";
        for (auto &P : S.Locals)
          Out += "l " + P.first + " " + std::to_string(P.second) + "\n";
        Out += "H\n";
        for (auto &P : S.History)
          Out += "h " + P.first + " " + P.second + "\n";
        return Out;
      };
  auto deserializeSnapshot =
      [](const std::string &Blob)
      -> matlab::statechart::ChartInterpreter::Snapshot {
        matlab::statechart::ChartInterpreter::Snapshot S;
        size_t i = 0;
        while (i < Blob.size()) {
          size_t nl = Blob.find('\n', i);
          if (nl == std::string::npos) nl = Blob.size();
          std::string Line = Blob.substr(i, nl - i);
          i = nl + 1;
          if (Line.size() < 2) continue;
          char Tag = Line[0];
          if (Tag != 'r' && Tag != 'l' && Tag != 'h') continue;
          size_t sp = Line.find(' ', 2);
          if (sp == std::string::npos) continue;
          std::string K = Line.substr(2, sp - 2);
          std::string V = Line.substr(sp + 1);
          if (Tag == 'r') S.Regions[K] = V;
          else if (Tag == 'l') {
            try { S.Locals[K] = std::stod(V); } catch (...) {}
          } else if (Tag == 'h') S.History[K] = V;
        }
        return S;
      };

  bool ConfDone = false;
  std::ios::sync_with_stdio(false);
  while (true) {
    auto Msg = readFrame();
    if (!Msg) break;
    if (getenv("MATLABC_DAP_TRACE")) {
      std::fprintf(stderr, "[chart-dap] frame: len=%zu \"%s\"\n",
                   Msg->size(),
                   Msg->substr(0, 80).c_str());
    }
    if (Msg->empty()) continue;
    auto Parsed = llvm::json::parse(*Msg);
    if (!Parsed) { llvm::consumeError(Parsed.takeError()); continue; }
    const Object *O = Parsed->getAsObject();
    if (!O) continue;
    auto Ty = O->getString("type");
    if (!Ty || *Ty != "request") continue;
    auto Cmd = O->getString("command");
    int64_t Seq = O->getInteger("seq").value_or(0);
    if (!Cmd) continue;
    if (getenv("MATLABC_DAP_TRACE")) {
      std::fprintf(stderr, "[chart-dap] recv: command=%s seq=%lld\n",
                   Cmd->str().c_str(), (long long)Seq);
    }

    if (*Cmd == "initialize") {
      Object Caps{
        {"supportsConfigurationDoneRequest", true},
        {"supportsStateChartProtocol", true},
      };
      sendResponse(Seq, *Cmd, true, std::move(Caps));
      sendEvent("initialized");
      continue;
    }
    if (*Cmd == "launch" || *Cmd == "attach") {
      sendResponse(Seq, *Cmd, true, Object{});
      continue;
    }
    if (*Cmd == "configurationDone") {
      sendResponse(Seq, *Cmd, true, Object{});
      ConfDone = true;
      streamTrace(Interp.initialize());
      continue;
    }
    if (*Cmd == "disconnect" || *Cmd == "terminate") {
      sendResponse(Seq, *Cmd, true, Object{});
      break;
    }

    // --- stateChart-namespaced verbs ---------------------------------
    if (*Cmd == "stateChart/setStateBreakpoints") {
      std::vector<std::string> Enter, Exit;
      const Object *Args = O->getObject("arguments");
      if (Args) {
        bool OnEnter = Args->getBoolean("onEnter").value_or(true);
        bool OnExit  = Args->getBoolean("onExit").value_or(false);
        const Array *Ids = Args->getArray("ids");
        if (Ids) {
          for (auto &V : *Ids) {
            auto S = V.getAsString();
            if (!S) continue;
            if (OnEnter) Enter.push_back(S->str());
            if (OnExit)  Exit.push_back(S->str());
          }
        }
      }
      Interp.setStateEnterBreakpoints(Enter);
      Interp.setStateExitBreakpoints(Exit);
      sendResponse(Seq, *Cmd, true,
                   Object{{"verified", true},
                          {"count", (int64_t)Enter.size() +
                                    (int64_t)Exit.size()}});
      continue;
    }
    if (*Cmd == "stateChart/setTransitionBreakpoints") {
      std::vector<std::string> Ids;
      const Object *Args = O->getObject("arguments");
      if (Args) {
        const Array *Arr = Args->getArray("ids");
        if (Arr) {
          for (auto &V : *Arr) {
            auto S = V.getAsString();
            if (!S) continue;
            Ids.push_back(S->str());
          }
        }
      }
      Interp.setTransitionBreakpoints(Ids);
      sendResponse(Seq, *Cmd, true,
                   Object{{"verified", true}, {"count", (int64_t)Ids.size()}});
      continue;
    }
    if (*Cmd == "stateChart/setSymbolBreakpoints") {
      // Tier 5 — right-click a symbol in the Symbols pane → "Break
      // on change". Pause when any of these locals' value is updated
      // by an entry / during / exit / cond / trans / on-event action.
      std::vector<std::string> Names;
      const Object *Args = O->getObject("arguments");
      if (Args) {
        const Array *Arr = Args->getArray("names");
        if (Arr) {
          for (auto &V : *Arr) {
            auto S = V.getAsString();
            if (!S) continue;
            Names.push_back(S->str());
          }
        }
      }
      Interp.setSymbolBreakpoints(Names);
      sendResponse(Seq, *Cmd, true,
                   Object{{"verified", true}, {"count", (int64_t)Names.size()}});
      continue;
    }
    if (*Cmd == "stateChart/emit") {
      const Object *Args = O->getObject("arguments");
      auto Name = Args ? Args->getString("name") : std::nullopt;
      if (Name) Interp.emit(Name->str());
      sendResponse(Seq, *Cmd, true,
                   Object{{"name", Name ? Name->str() : std::string()}});
      continue;
    }
    if (*Cmd == "stateChart/setLocal") {
      const Object *Args = O->getObject("arguments");
      auto Name = Args ? Args->getString("name") : std::nullopt;
      auto Val  = Args ? Args->getNumber("value") : std::nullopt;
      if (Name && Val) Interp.setLocal(Name->str(), *Val);
      sendResponse(Seq, *Cmd, true, Object{});
      continue;
    }
    if (*Cmd == "stateChart/getActive") {
      Array Ids;
      for (auto &Id : Interp.activeStates()) Ids.push_back(Id);
      sendResponse(Seq, *Cmd, true, Object{{"ids", std::move(Ids)}});
      continue;
    }
    if (*Cmd == "stateChart/getLocals") {
      // Snapshot every local + its current value. Pairs with the
      // Active-State pane's live-inspector row during pause.
      Object Locals;
      for (auto &P : Interp.allLocals()) Locals[P.first] = P.second;
      sendResponse(Seq, *Cmd, true,
                   Object{{"locals", std::move(Locals)}});
      continue;
    }
    if (*Cmd == "stateChart/listStates") {
      // Flat list of every state in the chart. Parent / decomposition
      // / initial-flag fields let the IDE recreate the hierarchy tree
      // without re-parsing the .mflow source.
      Array Out;
      for (auto &P : Entry->States) {
        const matlab::statechart::ChartState &S = P.second;
        Object O{
          {"id", S.Id},
          {"label", S.Label},
          {"parent", S.ParentId},
          {"decomposition",
              std::string(matlab::statechart::decompositionName(S.Decomp))},
          {"isInitial", S.IsInitial},
          {"hasHistory", S.HasHistory},
          {"atomic", S.Atomic},
        };
        if (S.ExecutionOrder)
          O["executionOrder"] = (int64_t)*S.ExecutionOrder;
        Out.push_back(std::move(O));
      }
      sendResponse(Seq, *Cmd, true, Object{{"states", std::move(Out)}});
      continue;
    }
    if (*Cmd == "stateChart/listJunctions") {
      // Every connective / history / entry / exit / default junction
      // in the chart. The IDE renders glyphs from this without
      // re-parsing the FlowDoc.
      Array Out;
      for (auto &P : Entry->Junctions) {
        const matlab::statechart::ChartJunction &J = P.second;
        Out.push_back(Object{
          {"id", J.Id},
          {"kind",
              std::string(matlab::statechart::junctionKindName(J.Kind))},
          {"parent", J.ParentId},
        });
      }
      sendResponse(Seq, *Cmd, true, Object{{"junctions", std::move(Out)}});
      continue;
    }
    if (*Cmd == "stateChart/listTransitions") {
      // One entry per transition with its parsed label fields. The IDE
      // breakpoint UI uses this to populate "fire on" pickers.
      Array Out;
      for (auto &T : Entry->Transitions) {
        Object O{
          {"id", T.Id},
          {"src", T.SourceId},
          {"dst", T.DestId},
          {"kind",
              std::string(matlab::statechart::transitionKindName(T.Kind))},
          {"priority", (int64_t)T.Priority},
        };
        if (!T.Label.Raw.empty())         O["label"]      = T.Label.Raw;
        if (!T.Label.Event.empty())       O["event"]      = T.Label.Event;
        if (!T.Label.Guard.empty())       O["guard"]      = T.Label.Guard;
        if (!T.Label.CondAction.empty())  O["condAction"] = T.Label.CondAction;
        if (!T.Label.TransAction.empty()) O["transAction"]= T.Label.TransAction;
        Out.push_back(std::move(O));
      }
      sendResponse(Seq, *Cmd, true, Object{{"transitions", std::move(Out)}});
      continue;
    }
    if (*Cmd == "stateChart/listEvents") {
      // Symbol table → events bucket. Lets the IDE populate the
      // "broadcast" combobox in the Command Window.
      Array Out;
      for (auto &E : Entry->Symbols.Events) {
        Object O{{"name", E.Name}};
        if (!E.Scope.empty())   O["scope"]   = E.Scope;
        if (!E.Trigger.empty()) O["trigger"] = E.Trigger;
        Out.push_back(std::move(O));
      }
      sendResponse(Seq, *Cmd, true, Object{{"events", std::move(Out)}});
      continue;
    }
    if (*Cmd == "stateChart/listSymbols") {
      // Every data + message slot the chart declares. The Symbols
      // inspector tab reads this; the IDE doesn't need to re-parse
      // the .mflow JSON itself.
      auto emitSym = [&](const matlab::flowchart::Symbol &S,
                         const char *Bucket) {
        Object O{{"name", S.Name}, {"bucket", std::string(Bucket)}};
        if (!S.Scope.empty())   O["scope"]   = S.Scope;
        if (!S.Type.empty())    O["type"]    = S.Type;
        if (!S.Units.empty())   O["units"]   = S.Units;
        if (!S.Initial.empty()) O["initial"] = S.Initial;
        return O;
      };
      Array Out;
      for (auto &S : Entry->Symbols.Data)     Out.push_back(emitSym(S, "data"));
      for (auto &S : Entry->Symbols.Messages) Out.push_back(emitSym(S, "message"));
      sendResponse(Seq, *Cmd, true, Object{{"symbols", std::move(Out)}});
      continue;
    }
    if (*Cmd == "stateChart/listSnapshots") {
      // Names currently in the runtime snapshot ring. Sizes included
      // so the IDE can show a footprint column in the snapshots panel.
      Array Out;
      int N = ::mstateflow_snapshot_count();
      for (int I = 0; I < N; ++I) {
        const char *Nm = ::mstateflow_snapshot_name(I);
        size_t Sz = ::mstateflow_snapshot_name_size(I);
        if (!Nm) continue;
        Out.push_back(Object{{"name", std::string(Nm)},
                             {"bytes", (int64_t)Sz}});
      }
      sendResponse(Seq, *Cmd, true, Object{{"snapshots", std::move(Out)}});
      continue;
    }
    if (*Cmd == "stateChart/stepSuperStep") {
      auto Trace = Interp.superStep();
      streamTrace(Trace);
      // Cheap count for the IDE: how many transitions fired.
      int64_t Fired = 0;
      bool Quiescent = false;
      using TK = matlab::statechart::ChartTraceEvent::Kind;
      for (auto &E : Trace) {
        if (E.K == TK::TransitionFired) ++Fired;
        if (E.K == TK::SuperStepEnd)    Quiescent = E.Quiescent;
      }
      sendResponse(Seq, *Cmd, true,
                   Object{{"quiescent", Quiescent}, {"firedCount", Fired}});
      continue;
    }
    if (*Cmd == "stateChart/stepTransition") {
      auto Trace = Interp.stepTransition();
      streamTrace(Trace);
      int64_t Fired = 0;
      using TK = matlab::statechart::ChartTraceEvent::Kind;
      for (auto &E : Trace) if (E.K == TK::TransitionFired) ++Fired;
      sendResponse(Seq, *Cmd, true,
                   Object{{"fired", Fired > 0}, {"firedCount", Fired}});
      continue;
    }
    if (*Cmd == "stateChart/saveOperatingPoint") {
      const Object *Args = O->getObject("arguments");
      auto Name = Args ? Args->getString("name") : std::nullopt;
      std::string N = Name ? Name->str() : std::string("default");
      std::string Blob = serializeSnapshot(Interp.snapshot());
      ::mstateflow_snapshot_save_blob(N.c_str(), Blob.data(), Blob.size());
      sendResponse(Seq, *Cmd, true,
                   Object{{"name", N}, {"bytes", (int64_t)Blob.size()}});
      continue;
    }
    if (*Cmd == "stateChart/restoreOperatingPoint") {
      const Object *Args = O->getObject("arguments");
      auto Name = Args ? Args->getString("name") : std::nullopt;
      std::string N = Name ? Name->str() : std::string("default");
      size_t Sz = ::mstateflow_snapshot_size(N.c_str());
      if (Sz == 0) {
        sendResponse(Seq, *Cmd, false,
                     Object{{"name", N},
                            {"reason", "unknown snapshot"}});
        continue;
      }
      std::string Blob(Sz, '\0');
      ::mstateflow_snapshot_copy(N.c_str(), Blob.data(), Sz);
      Interp.restore(deserializeSnapshot(Blob));
      sendResponse(Seq, *Cmd, true, Object{{"name", N}, {"restored", true}});
      // Refresh the IDE's view of active state via a synthetic
      // super-step boundary (no events fired, just announce the new
      // active configuration).
      sendEvent("stateChart/superStepBegin",
                Object{{"iteration", (int64_t)0}, {"t", 0.0}});
      for (auto &Id : Interp.activeStates())
        sendEvent("stateChart/stateEnter",
                  Object{{"id", Id}, {"t", 0.0}});
      sendEvent("stateChart/superStepEnd",
                Object{{"iteration", (int64_t)0},
                       {"quiescent", true},
                       {"t", 0.0}});
      continue;
    }

    // Unknown command — ack with success to keep the IDE happy.
    (void)ConfDone;
    sendResponse(Seq, *Cmd, true, Object{});
  }
  return 0;
}

} // namespace dap
#endif

#if MATLAB_LLVM_WITH_MLIR
/* ===========================================================================
 * `-emit-cocotb` v1
 *
 * Generates a CocoTB testbench directory next to the input .m file (or at
 * `-cocotb-out=DIR`):
 *
 *     <stem>_cocotb/
 *         <stem>.sv         — the SV DUT (matlabc -emit-systemverilog)
 *         <stem>_ref.py     — the Python reference model (-emit-python)
 *         test_<stem>.py    — the CocoTB harness (random-vector lockstep)
 *         cocotb_fi.py      — fi pack / unpack helpers (mirror of runtime/)
 *         Makefile          — Verilator + CocoTB invocation
 *
 * The harness drives N random vectors through the DUT and the reference
 * Python model and asserts cycle-by-cycle equality. Combinational vs
 * sequential is detected by scanning the lowered IR for
 * `matlab_persistent_*` calls. v1 only handles function-only `.m` files
 * (the typical HDL shape); script-with-driver is rejected with a hint.
 * =========================================================================*/

/* Embedded copy of `runtime/cocotb_fi.py`. We ship it inside the binary
 * so the harness directory is self-contained — the user can move it
 * anywhere and `make` from there without depending on the source tree
 * layout. Kept in sync with the canonical file under `runtime/`. */
static const char *kCocotbFiHelperPy =
R"COCOTB_FI("""Fixed-point pack / unpack helpers for the matlabc CocoTB harness.
Mirror of runtime/cocotb_fi.py; regenerated by matlabc -emit-cocotb."""


def pack_fi(value, signed: bool, wl: int, fl: int) -> int:
    if wl <= 0:
        raise ValueError(f"pack_fi: WL must be positive, got {wl}")
    raw = round(float(value) * (1 << fl))
    if signed:
        lo = -(1 << (wl - 1))
        hi = (1 << (wl - 1)) - 1
    else:
        lo = 0
        hi = (1 << wl) - 1
    if raw < lo:
        raw = lo
    elif raw > hi:
        raw = hi
    if signed and raw < 0:
        raw += 1 << wl
    return raw


def unpack_fi(bits, signed: bool, wl: int, fl: int) -> float:
    try:
        raw = int(bits)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unpack_fi: signal value not resolvable: {bits!r}") from exc
    if wl <= 0:
        raise ValueError(f"unpack_fi: WL must be positive, got {wl}")
    raw &= (1 << wl) - 1
    if signed and (raw & (1 << (wl - 1))):
        raw -= 1 << wl
    return raw / (1 << fl)


def fi_range(signed: bool, wl: int, fl: int):
    if signed:
        lo_raw = -(1 << (wl - 1))
        hi_raw = (1 << (wl - 1)) - 1
    else:
        lo_raw = 0
        hi_raw = (1 << wl) - 1
    scale = 1 << fl
    return lo_raw / scale, hi_raw / scale
)COCOTB_FI";

struct CocotbPortSpec {
  std::string Name;
  bool Signed;
  unsigned WL;
  unsigned FL;
  std::string Kind; // "fi" or "bool"
  /* `% cocotb: hold(<name>, <cycles>)` from the source file pins
   * this input's value across <cycles> stimulus iterations. Used
   * for multi-stage pipelined DUTs (sequential_processor) where
   * the SV samples a given input at cycle k+L while the reference
   * consumes input[k] — without holding the input stable across
   * those L cycles, the comparison legitimately diverges per random
   * gain. 0 means "advance every cycle" (the default). */
  int HoldCycles = 0;
  /* Unpacked-array length. 0 = scalar port (the default), >0 =
   * `logic [W-1:0] name [N]` shape. The harness drives /
   * reads element-by-element via `dut.<name>[k]` and the Python
   * reference receives a list. Detected from the SV port-list
   * `name [N]` suffix during port parsing. */
  int ArrayLen = 0;
  /* `% cocotb: stimulus(<name>, <kind>, ...)` pragma — picks a
   * deterministic per-cycle value generator for this input.
   * Default `Random` keeps the v2 behaviour. The other kinds let
   * users drive impulse / constant / ramp patterns through
   * pipelined DUTs whose per-call reference can't keep up with
   * random per-cycle inputs (the multi-stage cascade in
   * sequential_processor / fir_asic_pipelined). */
  enum class StimKind { Random, Impulse, Constant, Ramp, Range };
  StimKind Stim = StimKind::Random;
  double StimArg1 = 0.0;   // impulse: value@0 / constant: value / ramp: start / range: lo
  double StimArg2 = 0.0;   // ramp: stride / range: hi
};

struct CocotbFuncSpec {
  std::string Name;
  std::vector<CocotbPortSpec> Inputs;
  std::vector<CocotbPortSpec> Outputs;
  bool Sequential = false; // has clk/rst_n on the SV side
};

struct CocotbStim {
  CocotbPortSpec::StimKind Kind = CocotbPortSpec::StimKind::Random;
  double Arg1 = 0.0;
  double Arg2 = 0.0;
};

struct CocotbPragmaScan {
  std::map<std::string, int> Holds;
  std::map<std::string, CocotbStim> Stim;
  /* `% cocotb: latency(N)` — the harness's pipeline-latency value
   * lifted into the source so a fixture's right-by-default L
   * lives next to the design. Equivalent to passing
   * `-cocotb-latency=N` on the command line. The CLI flag wins
   * when the user passes it explicitly (Options::CocotbLatencyExplicit). */
  std::optional<int> Latency;
  /* `% cocotb: cover(<port>, min_bins=N)` — fail the test when the
   * named port hit fewer than N distinct values across the random
   * stimulus. Catches "vectors only exercised half the FSM
   * states" silently. Per-port; multiple coverage pragmas allowed.
   * Output ports also accepted — applies to whichever side the
   * port name matches. */
  std::map<std::string, int> CoverMinBins;
  /* `% cocotb: cover_pairs(<port>, min_pairs=N)` — fail the test
   * when the named port saw fewer than N distinct (prev, curr)
   * consecutive value pairs. Transition coverage for FSMs and
   * any port whose semantics depend on edges, not just static
   * value distribution. Empty pairs (first cycle has no prev)
   * are not counted. */
  std::map<std::string, int> CoverPairsMin;
  /* `% cocotb: cover_range(<port>)` — fail the test when the
   * named port did not see every value in its full fi range
   * [lo..hi]. Only meaningful for narrow ports (WL <= 8); for
   * wider ports the universe is too large to be exhaustive.
   * Stored as a set (no per-pragma threshold; the threshold is
   * the full range size). */
  std::set<std::string> CoverRange;
};

/* Scan the original .m source for `% cocotb:` directives and
 * collect the (input-name → spec) maps. Tolerant of leading
 * whitespace and optional spaces around commas, single-line shape
 * only. Recognised directives:
 *
 *   % cocotb: hold(<name>, <cycles>)
 *   % cocotb: stimulus(<name>, impulse, <value>)
 *   % cocotb: stimulus(<name>, constant, <value>)
 *   % cocotb: stimulus(<name>, ramp, <start>, <stride>)
 *   % cocotb: latency(<cycles>)
 *
 * Unrecognised directives are silently ignored (forward-compat
 * with future v3 items). */
static CocotbPragmaScan
scanCocotbPragmas(const std::string &SrcPath) {
  CocotbPragmaScan Out;
  std::ifstream F(SrcPath);
  if (!F) return Out;
  std::string Src((std::istreambuf_iterator<char>(F)),
                   std::istreambuf_iterator<char>());
  auto skipWs = [](std::string &S) {
    while (!S.empty() && std::isspace((unsigned char)S.front()))
      S.erase(S.begin());
  };
  auto trim = [&](std::string &S) {
    skipWs(S);
    while (!S.empty() && std::isspace((unsigned char)S.back()))
      S.pop_back();
  };
  /* Split a comma-separated argument list. Returns the trimmed
   * tokens. No nesting / strings — sufficient for our pragma
   * surface. */
  auto splitArgs = [&trim](std::string Body) -> std::vector<std::string> {
    std::vector<std::string> Args;
    std::string Cur;
    for (char C : Body) {
      if (C == ',') {
        std::string A = Cur; trim(A); Args.push_back(A);
        Cur.clear();
      } else Cur += C;
    }
    if (!Cur.empty() || !Body.empty()) {
      std::string A = Cur; trim(A); Args.push_back(A);
    }
    return Args;
  };

  size_t Pos = 0;
  while (Pos < Src.size()) {
    size_t LineEnd = Src.find('\n', Pos);
    if (LineEnd == std::string::npos) LineEnd = Src.size();
    std::string Line = Src.substr(Pos, LineEnd - Pos);
    Pos = LineEnd + 1;
    auto Pct = Line.find('%');
    if (Pct == std::string::npos) continue;
    std::string Tail = Line.substr(Pct + 1);
    skipWs(Tail);
    if (Tail.compare(0, 7, "cocotb:") != 0) continue;
    Tail.erase(0, 7);
    skipWs(Tail);

    /* Identify the directive head (up to the open paren). */
    size_t Open = Tail.find('(');
    size_t Close = Tail.find(')');
    if (Open == std::string::npos || Close == std::string::npos ||
        Close < Open)
      continue;
    std::string Head = Tail.substr(0, Open);
    std::string Body = Tail.substr(Open + 1, Close - Open - 1);
    trim(Head);
    auto Args = splitArgs(Body);

    if (Head == "hold" && Args.size() == 2) {
      int Cy = std::atoi(Args[1].c_str());
      if (!Args[0].empty() && Cy >= 0) Out.Holds[Args[0]] = Cy;
      continue;
    }
    if (Head == "stimulus" && Args.size() >= 3) {
      CocotbStim S;
      const std::string &Kind = Args[1];
      if (Kind == "impulse" && Args.size() >= 3) {
        S.Kind = CocotbPortSpec::StimKind::Impulse;
        S.Arg1 = std::strtod(Args[2].c_str(), nullptr);
      } else if (Kind == "constant" && Args.size() >= 3) {
        S.Kind = CocotbPortSpec::StimKind::Constant;
        S.Arg1 = std::strtod(Args[2].c_str(), nullptr);
      } else if (Kind == "ramp" && Args.size() >= 4) {
        S.Kind = CocotbPortSpec::StimKind::Ramp;
        S.Arg1 = std::strtod(Args[2].c_str(), nullptr);
        S.Arg2 = std::strtod(Args[3].c_str(), nullptr);
      } else if (Kind == "range" && Args.size() >= 4) {
        S.Kind = CocotbPortSpec::StimKind::Range;
        S.Arg1 = std::strtod(Args[2].c_str(), nullptr);
        S.Arg2 = std::strtod(Args[3].c_str(), nullptr);
      } else continue;
      if (!Args[0].empty()) Out.Stim[Args[0]] = S;
      continue;
    }
    /* `% cocotb: range(<name>, <lo>, <hi>)` shorthand — equivalent to
     * `stimulus(<name>, range, <lo>, <hi>)`. Lets a DUT whose port pragma
     * declares `fi, signed, WL, FL` with FL > 0 constrain the random
     * stimulus to the natural real-value range; the SV erases FL from the
     * port list, so the harness would otherwise draw values up to ±2^(WL-1)
     * which overflow the SV's mid-computation truncation differently from
     * the Python reference's saturate-and-grow.  See the DL HDL H3 note in
     * docs/deep_learning_toolbox_roadmap.md.  */
    if (Head == "range" && Args.size() >= 3) {
      CocotbStim S;
      S.Kind = CocotbPortSpec::StimKind::Range;
      S.Arg1 = std::strtod(Args[1].c_str(), nullptr);
      S.Arg2 = std::strtod(Args[2].c_str(), nullptr);
      if (!Args[0].empty()) Out.Stim[Args[0]] = S;
      continue;
    }
    if (Head == "latency" && Args.size() == 1) {
      int L = std::atoi(Args[0].c_str());
      if (L >= 0) Out.Latency = L;
      continue;
    }
    if (Head == "cover" && Args.size() >= 2) {
      // Accept `cover(<port>, N)` and `cover(<port>, min_bins=N)`.
      // Strip an optional `min_bins=` prefix from the second arg.
      std::string Spec = Args[1];
      const std::string Key = "min_bins=";
      if (Spec.compare(0, Key.size(), Key) == 0)
        Spec.erase(0, Key.size());
      int N = std::atoi(Spec.c_str());
      if (!Args[0].empty() && N > 0) Out.CoverMinBins[Args[0]] = N;
      continue;
    }
    if (Head == "cover_pairs" && Args.size() >= 2) {
      // `cover_pairs(<port>, N)` or `cover_pairs(<port>, min_pairs=N)`
      std::string Spec = Args[1];
      const std::string Key = "min_pairs=";
      if (Spec.compare(0, Key.size(), Key) == 0)
        Spec.erase(0, Key.size());
      int N = std::atoi(Spec.c_str());
      if (!Args[0].empty() && N > 0) Out.CoverPairsMin[Args[0]] = N;
      continue;
    }
    if (Head == "cover_range" && Args.size() >= 1) {
      // `cover_range(<port>)` — no threshold, the range size is
      // implied by the port's fi width.
      if (!Args[0].empty()) Out.CoverRange.insert(Args[0]);
      continue;
    }
  }
  return Out;
}

/* Parse the SV file's `module <name> (...)` port list and rebuild a
 * CocotbFuncSpec from it. Source of truth for port types — the SV
 * emitter has already refined signed-ness, width, and direction by
 * the time it reaches the port list, so we don't have to replay the
 * pipeline in-process to derive the same info.
 *
 * Recognized port shapes (one per line, possibly with trailing comma):
 *   `input  logic signed [W-1:0] name,`
 *   `input  logic [W-1:0] name,`
 *   `input  logic name,`             (1-bit, e.g. `clk`, `rst_n`, bool)
 *   `output ...` (same forms)
 *
 * Synthetic clk / rst_n are filtered into Spec.Sequential rather than
 * appearing as harness-driven INPUTS. FL is always 0 here — the SV
 * layer has erased the fractional-bit information; the harness uses
 * it only for the random vector range, where Q.0 is the safe default.
 * Real fi precision still flows through the reference Python model
 * because that uses matlab_runtime's own fi semantics. */
static std::optional<CocotbFuncSpec>
parseCocotbSpecFromSv(const std::string &SvPath,
                      const std::string &Stem) {
  std::ifstream F(SvPath);
  if (!F) return std::nullopt;
  std::string Src((std::istreambuf_iterator<char>(F)),
                   std::istreambuf_iterator<char>());
  // Locate the first `module <name> (...);` block.
  size_t MStart = Src.find("module ");
  if (MStart == std::string::npos) return std::nullopt;
  size_t Open = Src.find('(', MStart);
  size_t Close = Src.find(");", Open);
  if (Open == std::string::npos || Close == std::string::npos)
    return std::nullopt;

  // Module name lives between `module ` and `(`.
  std::string Header = Src.substr(MStart + 7, Open - (MStart + 7));
  // Trim leading/trailing whitespace.
  while (!Header.empty() && std::isspace((unsigned char)Header.back()))
    Header.pop_back();
  while (!Header.empty() && std::isspace((unsigned char)Header.front()))
    Header.erase(Header.begin());
  CocotbFuncSpec S;
  S.Name = Header.empty() ? Stem : Header;

  std::string Ports = Src.substr(Open + 1, Close - Open - 1);
  // Tokenise by `,` at top level. SV's port list isn't nested with
  // parens or braces, so a flat split is fine.
  std::vector<std::string> Lines;
  {
    std::string Cur;
    for (char C : Ports) {
      if (C == ',') {
        Lines.push_back(Cur);
        Cur.clear();
      } else {
        Cur += C;
      }
    }
    if (!Cur.empty()) Lines.push_back(Cur);
  }

  auto trim = [](std::string &S) {
    while (!S.empty() && std::isspace((unsigned char)S.back())) S.pop_back();
    while (!S.empty() && std::isspace((unsigned char)S.front()))
      S.erase(S.begin());
  };

  for (auto &L : Lines) {
    trim(L);
    if (L.empty()) continue;
    bool IsInput;
    if (L.compare(0, 6, "input ") == 0) { IsInput = true;  L.erase(0, 6); }
    else if (L.compare(0, 7, "output ") == 0) { IsInput = false; L.erase(0, 7); }
    else continue;
    trim(L);
    // Drop the leading `logic` keyword (always present in our emit).
    if (L.compare(0, 6, "logic ") == 0) L.erase(0, 6);
    trim(L);
    bool Signed = false;
    if (L.compare(0, 7, "signed ") == 0) { Signed = true; L.erase(0, 7); }
    trim(L);
    unsigned WL = 1;
    // Optional `[W-1:0]`.
    if (!L.empty() && L.front() == '[') {
      size_t RB = L.find(']');
      if (RB == std::string::npos) continue;
      std::string Bits = L.substr(1, RB - 1);
      // Bits is `<HI>:<LO>`; parse <HI> + 1 = WL.
      size_t Colon = Bits.find(':');
      if (Colon == std::string::npos) continue;
      WL = (unsigned)std::atoi(Bits.substr(0, Colon).c_str()) + 1;
      L.erase(0, RB + 1);
      trim(L);
    }
    // Remainder is the port name (until end of line, possibly with
    // a `[N]` array-length suffix for unpacked vector ports).
    std::string Name;
    size_t I = 0;
    for (; I < L.size(); ++I) {
      char C = L[I];
      if (std::isalnum((unsigned char)C) || C == '_') Name += C;
      else break;
    }
    if (Name.empty()) continue;
    // Trailing whitespace then `[N]` → vector / unpacked-array port.
    // v3.3: capture N and emit per-element drive / read.
    int ArrayLen = 0;
    while (I < L.size() && std::isspace((unsigned char)L[I])) ++I;
    if (I < L.size() && L[I] == '[') {
      ++I;
      std::string Num;
      while (I < L.size() && std::isdigit((unsigned char)L[I]))
        Num += L[I++];
      if (I < L.size() && L[I] == ']') ++I;
      ArrayLen = std::atoi(Num.c_str());
      if (ArrayLen <= 0) ArrayLen = 0; // unparsable — fall back to scalar
    }

    // clk / rst_n are synthetic — flip Sequential and skip the port.
    if (Name == "clk" || Name == "rst_n") {
      S.Sequential = true;
      continue;
    }
    CocotbPortSpec P;
    P.Name = Name;
    P.Signed = Signed;
    P.WL = WL;
    P.FL = 0;
    P.Kind = (WL == 1) ? "bool" : "fi";
    P.ArrayLen = ArrayLen;
    if (IsInput) S.Inputs.push_back(P);
    else        S.Outputs.push_back(P);
  }
  return S;
}


/* ===========================================================================
 * v3.2 — Tester-driven stimulus extraction (`test_<stem>.m`)
 *
 * When `-emit-cocotb FILE.m` finds a sibling `test_<stem>.m`, the
 * harness can replay the tester's hand-picked stimulus instead of
 * driving random vectors. This is the open-source equivalent of
 * MathWorks's "Simulink Coder Test Bench" workflow: the same
 * MATLAB test driver that the user runs by hand also drives the
 * SV DUT.
 *
 * Approach: AST-walk `test_<stem>.m` and unroll the stimulus loop
 * into a flat list of input tuples. The harness then replays them
 * cycle-by-cycle, calling the Python reference with the same
 * tuple to compute the expected output (no double-implementation
 * — the tester's MATLAB code is the spec; the device's own
 * `<stem>_ref.py` is what evaluates it).
 *
 * Recognised tester shapes (covers test_mealy / test_fsm_moore /
 * test_counter / test_mux):
 *
 *   1. Single device call, no loop:
 *        a = fi(...); b = uint8(...); ...
 *        result = device(a, b, ...);
 *
 *   2. Vector-driven loop:
 *        bits = [<row literal>]; reset = false;
 *        for i = 1:length(bits)
 *            r = device(bits(i), reset);
 *        end
 *
 *   3. Fixed-count loop with conditional inputs:
 *        for i = 1:N
 *            if i == K, reset = true;
 *            else      reset = false;
 *            end
 *            r = device(reset);
 *        end
 *
 * Anything outside this set returns std::nullopt and the caller
 * falls back to random vectors with a diagnostic. Adding more
 * shapes is the v3 follow-up — start small. The four existing
 * `test_*.m` fixtures all extract cleanly.
 * =========================================================================*/

namespace {

struct EvalVal {
  enum class K { None, Num, Bool, Vec } Kind = K::None;
  double Num = 0.0;
  bool Bool = false;
  std::vector<double> Vec;
};

/* Format a Python literal from an evaluated value. Bools render as
 * Python `True` / `False`; integers render without a decimal point so
 * `pack_fi` doesn't have to round trivial cases; non-integer doubles
 * use full precision. */
static std::string pyLit(const EvalVal &V) {
  switch (V.Kind) {
    case EvalVal::K::Bool: return V.Bool ? "True" : "False";
    case EvalVal::K::Num: {
      if (V.Num == std::trunc(V.Num) &&
          std::abs(V.Num) < 1e15) {
        char Buf[64];
        snprintf(Buf, sizeof Buf, "%lld", (long long)V.Num);
        return Buf;
      }
      char Buf[64];
      snprintf(Buf, sizeof Buf, "%.17g", V.Num);
      return Buf;
    }
    default: return "0";
  }
}

class TesterStimulus {
public:
  std::optional<std::vector<std::vector<std::string>>>
  extract(matlab::TranslationUnit *TU, llvm::StringRef DeviceName) {
    DeviceName_ = DeviceName.str();
    if (!TU || !TU->ScriptNode || !TU->ScriptNode->Body) return std::nullopt;
    Tuples_.clear();
    Sym_.clear();
    if (!walkBlock(TU->ScriptNode->Body)) return std::nullopt;
    if (Tuples_.empty()) return std::nullopt;
    return Tuples_;
  }

private:
  std::string DeviceName_;
  std::vector<std::vector<std::string>> Tuples_;
  std::map<std::string, EvalVal> Sym_;

  bool walkBlock(matlab::Block *B) {
    if (!B) return false;
    for (auto *S : B->Stmts) {
      if (!walkStmt(S)) return false;
    }
    return true;
  }

  bool walkStmt(matlab::Stmt *S) {
    using namespace matlab;
    if (auto *AS = dynamic_cast<AssignStmt *>(S)) {
      /* Both `r = device(...)` and `[a, b] = device(...)` shapes
       * record the same stimulus tuple — only the call args matter
       * for driving the DUT. Single-LHS assignments may also bind a
       * literal value to a name, which we capture for later
       * substitution in the loop body. */
      if (auto *RC = dynamic_cast<CallOrIndex *>(AS->RHS)) {
        if (auto *Cn = dynamic_cast<NameExpr *>(RC->Callee)) {
          if (Cn->Name == DeviceName_) {
            return recordCall(RC);
          }
        }
      }
      if (AS->LHS.size() == 1) {
        if (auto *LN = dynamic_cast<NameExpr *>(AS->LHS[0])) {
          EvalVal V;
          if (evalExpr(AS->RHS, V))
            Sym_[std::string(LN->Name)] = V;
        }
      }
      return true;
    }
    if (auto *ES = dynamic_cast<ExprStmt *>(S)) {
      if (auto *RC = dynamic_cast<CallOrIndex *>(ES->E)) {
        if (auto *Cn = dynamic_cast<NameExpr *>(RC->Callee)) {
          if (Cn->Name == DeviceName_) {
            return recordCall(RC);
          }
        }
      }
      return true; // ignore fprintf / disp / etc.
    }
    if (auto *FS = dynamic_cast<ForStmt *>(S)) {
      auto *RE = dynamic_cast<RangeExpr *>(FS->Iter);
      if (!RE) return false;
      double Lo, Hi;
      if (!evalNumber(RE->Start, Lo)) return false;
      if (!evalNumber(RE->End, Hi)) return false;
      double Step = 1.0;
      if (RE->Step && !evalNumber(RE->Step, Step)) return false;
      if (Step == 0.0) return false;
      std::string IvName(FS->Var);
      EvalVal SavedI; bool HadI = Sym_.count(IvName);
      if (HadI) SavedI = Sym_[IvName];
      for (double i = Lo;
           (Step > 0) ? (i <= Hi + 1e-9) : (i >= Hi - 1e-9);
           i += Step) {
        EvalVal IV; IV.Kind = EvalVal::K::Num; IV.Num = i;
        Sym_[IvName] = IV;
        if (!walkBlock(FS->Body)) return false;
      }
      if (HadI) Sym_[IvName] = SavedI;
      else Sym_.erase(IvName);
      return true;
    }
    if (auto *IS = dynamic_cast<IfStmt *>(S)) {
      EvalVal CV;
      if (!evalExpr(IS->Cond, CV)) return false;
      bool Truthy = (CV.Kind == EvalVal::K::Bool && CV.Bool) ||
                    (CV.Kind == EvalVal::K::Num && CV.Num != 0.0);
      if (Truthy) return walkBlock(IS->Then);
      for (auto &EI : IS->Elseifs) {
        EvalVal EV;
        if (!evalExpr(EI.Cond, EV)) return false;
        bool ET = (EV.Kind == EvalVal::K::Bool && EV.Bool) ||
                  (EV.Kind == EvalVal::K::Num && EV.Num != 0.0);
        if (ET) return walkBlock(EI.Body);
      }
      if (IS->Else) return walkBlock(IS->Else);
      return true;
    }
    /* Anything else in the body: ignore but proceed (e.g. a stray
     * disp or whatever). The extraction is best-effort — we only
     * fail when we can't even recognise the structure. */
    return true;
  }

  bool recordCall(matlab::CallOrIndex *C) {
    std::vector<std::string> Tuple;
    Tuple.reserve(C->Args.size());
    for (auto *Arg : C->Args) {
      EvalVal V;
      if (!evalExpr(Arg, V)) return false;
      Tuple.push_back(pyLit(V));
    }
    Tuples_.push_back(std::move(Tuple));
    return true;
  }

  bool evalNumber(matlab::Expr *E, double &Out) {
    EvalVal V;
    if (!evalExpr(E, V)) return false;
    if (V.Kind == EvalVal::K::Num) { Out = V.Num; return true; }
    if (V.Kind == EvalVal::K::Bool) { Out = V.Bool ? 1.0 : 0.0; return true; }
    return false;
  }

  bool evalExpr(matlab::Expr *E, EvalVal &Out) {
    using namespace matlab;
    if (!E) return false;
    if (auto *I = dynamic_cast<IntegerLiteral *>(E)) {
      Out.Kind = EvalVal::K::Num;
      Out.Num = std::strtod(std::string(I->Text).c_str(), nullptr);
      return true;
    }
    if (auto *F = dynamic_cast<FPLiteral *>(E)) {
      Out.Kind = EvalVal::K::Num;
      Out.Num = std::strtod(std::string(F->Text).c_str(), nullptr);
      return true;
    }
    if (auto *N = dynamic_cast<NameExpr *>(E)) {
      if (N->Name == "true") { Out.Kind = EvalVal::K::Bool; Out.Bool = true; return true; }
      if (N->Name == "false") { Out.Kind = EvalVal::K::Bool; Out.Bool = false; return true; }
      auto It = Sym_.find(std::string(N->Name));
      if (It == Sym_.end()) return false;
      Out = It->second;
      return true;
    }
    if (auto *U = dynamic_cast<UnaryOpExpr *>(E)) {
      EvalVal V;
      if (!evalExpr(U->Operand, V)) return false;
      if (V.Kind != EvalVal::K::Num) return false;
      switch (U->Op) {
        case UnOp::Plus:  Out = V; return true;
        case UnOp::Minus: Out.Kind = EvalVal::K::Num; Out.Num = -V.Num; return true;
        case UnOp::Not:   Out.Kind = EvalVal::K::Bool; Out.Bool = (V.Num == 0.0); return true;
      }
      return false;
    }
    if (auto *B = dynamic_cast<BinaryOpExpr *>(E)) {
      EvalVal L, R;
      if (!evalExpr(B->LHS, L) || !evalExpr(B->RHS, R)) return false;
      auto N = [](const EvalVal &V) -> double {
        if (V.Kind == EvalVal::K::Num) return V.Num;
        if (V.Kind == EvalVal::K::Bool) return V.Bool ? 1.0 : 0.0;
        return 0.0;
      };
      switch (B->Op) {
        case BinOp::Add: Out.Kind = EvalVal::K::Num; Out.Num = N(L) + N(R); return true;
        case BinOp::Sub: Out.Kind = EvalVal::K::Num; Out.Num = N(L) - N(R); return true;
        case BinOp::Mul: Out.Kind = EvalVal::K::Num; Out.Num = N(L) * N(R); return true;
        case BinOp::Div: Out.Kind = EvalVal::K::Num; Out.Num = (N(R)==0?0:N(L)/N(R)); return true;
        case BinOp::Eq: Out.Kind = EvalVal::K::Bool; Out.Bool = (N(L) == N(R)); return true;
        case BinOp::Ne: Out.Kind = EvalVal::K::Bool; Out.Bool = (N(L) != N(R)); return true;
        case BinOp::Lt: Out.Kind = EvalVal::K::Bool; Out.Bool = (N(L) <  N(R)); return true;
        case BinOp::Le: Out.Kind = EvalVal::K::Bool; Out.Bool = (N(L) <= N(R)); return true;
        case BinOp::Gt: Out.Kind = EvalVal::K::Bool; Out.Bool = (N(L) >  N(R)); return true;
        case BinOp::Ge: Out.Kind = EvalVal::K::Bool; Out.Bool = (N(L) >= N(R)); return true;
        default: return false;
      }
    }
    if (auto *M = dynamic_cast<MatrixLiteral *>(E)) {
      Out.Kind = EvalVal::K::Vec;
      Out.Vec.clear();
      for (auto &Row : M->Rows) {
        for (auto *Cell : Row) {
          EvalVal CV;
          if (!evalExpr(Cell, CV)) return false;
          if (CV.Kind == EvalVal::K::Num) Out.Vec.push_back(CV.Num);
          else if (CV.Kind == EvalVal::K::Bool) Out.Vec.push_back(CV.Bool ? 1.0 : 0.0);
          else return false;
        }
      }
      return true;
    }
    if (auto *C = dynamic_cast<CallOrIndex *>(E)) {
      auto *Cn = dynamic_cast<NameExpr *>(C->Callee);
      if (!Cn) return false;
      llvm::StringRef Cname = Cn->Name;
      /* `length(vec)` → vector size as a Num. */
      if (Cname == "length" && C->Args.size() == 1) {
        EvalVal Inner;
        if (!evalExpr(C->Args[0], Inner)) return false;
        if (Inner.Kind != EvalVal::K::Vec) return false;
        Out.Kind = EvalVal::K::Num;
        Out.Num = (double)Inner.Vec.size();
        return true;
      }
      /* Type-cast / fi-constructor calls: just take the first arg's
       * numeric value. The harness does its own pack_fi based on the
       * SV port spec, so the wrapper's WL/FL args here are vestigial. */
      if (Cname == "fi" || Cname == "int8" || Cname == "int16" ||
          Cname == "int32" || Cname == "int64" || Cname == "uint8" ||
          Cname == "uint16" || Cname == "uint32" || Cname == "uint64" ||
          Cname == "double" || Cname == "single" || Cname == "logical") {
        if (C->Args.empty()) return false;
        return evalExpr(C->Args[0], Out);
      }
      /* `vec(i)` — index a known vector by a 1-based scalar. */
      auto It = Sym_.find(std::string(Cname));
      if (It != Sym_.end() && It->second.Kind == EvalVal::K::Vec &&
          C->Args.size() == 1) {
        EvalVal Idx;
        if (!evalExpr(C->Args[0], Idx)) return false;
        if (Idx.Kind != EvalVal::K::Num) return false;
        int64_t K = (int64_t)Idx.Num - 1;
        if (K < 0 || (size_t)K >= It->second.Vec.size()) return false;
        Out.Kind = EvalVal::K::Num;
        Out.Num = It->second.Vec[K];
        return true;
      }
      return false;
    }
    return false;
  }
};

/* True when `B` (or any nested block) contains a top-level call to
 * the named device function. Used to disambiguate between multiple
 * `test_*.m` scripts in the same directory. */
static bool blockCallsDevice(matlab::Block *B,
                              const std::string &DeviceName) {
  using namespace matlab;
  if (!B) return false;
  for (auto *S : B->Stmts) {
    auto matchesCall = [&](Expr *E) -> bool {
      auto *C = dynamic_cast<CallOrIndex *>(E);
      if (!C) return false;
      auto *N = dynamic_cast<NameExpr *>(C->Callee);
      return N && std::string(N->Name) == DeviceName;
    };
    if (auto *AS = dynamic_cast<AssignStmt *>(S)) {
      if (matchesCall(AS->RHS)) return true;
    } else if (auto *ES = dynamic_cast<ExprStmt *>(S)) {
      if (matchesCall(ES->E)) return true;
    } else if (auto *IS = dynamic_cast<IfStmt *>(S)) {
      if (blockCallsDevice(IS->Then, DeviceName)) return true;
      for (auto &EI : IS->Elseifs)
        if (blockCallsDevice(EI.Body, DeviceName)) return true;
      if (blockCallsDevice(IS->Else, DeviceName)) return true;
    } else if (auto *FS = dynamic_cast<ForStmt *>(S)) {
      if (blockCallsDevice(FS->Body, DeviceName)) return true;
    }
  }
  return false;
}

/* Locate a tester `.m` file next to the input and parse it into a
 * fresh TU. Strategy:
 *   1. Try the strict convention: `<input-dir>/test_<stem>.m`.
 *   2. If missing, scan `<input-dir>/test_*.m` and pick the first
 *      whose script body contains a call to the DUT function name —
 *      handles the existing examples/hdl/ naming that doesn't strictly
 *      mirror the function name (test_mealy.m for mealy_fsm.m,
 *      test_mux.m for mux_4to_1_16bit.m, etc.).
 * Returns nullptr on miss. Parse failures fall through silently —
 * the harness emit then falls back to random vectors with a
 * diagnostic. */
static matlab::TranslationUnit *
loadTesterTU(matlab::SourceManager &SM, matlab::ASTContext &AstCtx,
             const std::string &InputPath,
             const std::string &Stem,
             const std::string &DeviceName) {
  auto Slash = InputPath.find_last_of('/');
  std::string Parent = (Slash == std::string::npos) ? "."
                       : InputPath.substr(0, Slash);

  auto tryParse = [&](const std::string &Path) -> matlab::TranslationUnit * {
    struct stat St;
    if (stat(Path.c_str(), &St) != 0 || !S_ISREG(St.st_mode))
      return nullptr;
    matlab::FileID FID = SM.loadFile(Path);
    if (FID == 0) return nullptr;
    matlab::DiagnosticEngine Diag(SM);
    matlab::Lexer Lx(SM, FID, Diag);
    auto Toks = Lx.tokenize();
    matlab::Parser P(std::move(Toks), AstCtx, Diag);
    auto *TU = P.parseFile();
    if (!TU || Diag.hasErrors()) return nullptr;
    return TU;
  };

  /* (1) Strict convention. */
  if (auto *TU = tryParse(Parent + "/test_" + Stem + ".m"))
    return TU;

  /* (2) Heuristic: any `test_*.m` whose body calls our device. */
  DIR *D = opendir(Parent.c_str());
  if (!D) return nullptr;
  std::vector<std::string> Candidates;
  while (auto *Ent = readdir(D)) {
    std::string Name = Ent->d_name;
    if (Name.size() < 7 || Name.substr(0, 5) != "test_") continue;
    if (Name.size() < 3 || Name.substr(Name.size() - 2) != ".m") continue;
    Candidates.push_back(Name);
  }
  closedir(D);
  std::sort(Candidates.begin(), Candidates.end());
  for (auto &Name : Candidates) {
    auto *TU = tryParse(Parent + "/" + Name);
    if (!TU) continue;
    if (TU->ScriptNode &&
        blockCallsDevice(TU->ScriptNode->Body, DeviceName))
      return TU;
  }
  return nullptr;
}

} // namespace

/* Render the harness Python source from a single function spec.
 * Combinational modules sample after a 1ns settle; sequential
 * modules drive a 10ns clock, pulse `rst_n` low for 2 cycles at
 * startup, then sample on every posedge against the reference
 * model whose own per-call state (Python function-attribute
 * persistents) tracks the DUT's.
 *
 * `Latency` (>=0) aligns the comparison for pipelined DUTs: the
 * reference is called at cycle k, but DUT outputs aren't sampled
 * for comparison until cycle k+L. The loop runs N+L iterations —
 * the trailing L cycles drive zero inputs (pipeline flush) so
 * every recorded reference output gets matched against a DUT
 * sample. L=0 reproduces the v1 lockstep behaviour. */
static std::string
renderCocotbHarness(const CocotbFuncSpec &S, const std::string &Stem,
                    int Vectors, int Latency, int Seed,
                    const std::vector<std::vector<std::string>> *Stimulus,
                    const std::map<std::string, int> &CoverMinBins,
                    const std::map<std::string, int> &CoverPairsMin,
                    const std::set<std::string> &CoverRange) {
  std::string Out;
  auto append = [&](const std::string &S) { Out += S; };
  auto pyTuple = [](const CocotbPortSpec &P) -> std::string {
    bool Sn = P.Kind == "bool" ? false : P.Signed;
    /* (name, signed, wl, fl, array_len). array_len = 0 means
     * scalar; >0 means an unpacked array of that length, driven
     * and sampled per-element via `dut.<name>[k]`. */
    return "(\"" + P.Name + "\", " + (Sn ? "True" : "False") + ", "
         + std::to_string(P.WL) + ", " + std::to_string(P.FL) + ", "
         + std::to_string(P.ArrayLen) + ")";
  };

  append("# Generated by matlabc -emit-cocotb. Do not edit.\n");
  append("# DUT: " + S.Name + "  ");
  append(S.Sequential ? "[sequential]\n" : "[combinational]\n");
  append("\n");
  append("import json\n");
  append("import os\n");
  append("import random\n");
  append("import sys\n");
  append("\n");
  append("import cocotb\n");
  if (S.Sequential) {
    append("from cocotb.clock import Clock\n");
    append("from cocotb.triggers import RisingEdge, Timer\n");
  } else {
    append("from cocotb.triggers import Timer\n");
  }
  append("\n");
  append("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n");
  append("from cocotb_fi import pack_fi, unpack_fi, fi_range\n");
  append("from " + Stem + "_ref import " + S.Name + "\n");
  append("\n");
  append("INPUTS = [\n");
  for (auto &P : S.Inputs) append("    " + pyTuple(P) + ",\n");
  append("]\n");
  append("OUTPUTS = [\n");
  for (auto &P : S.Outputs) append("    " + pyTuple(P) + ",\n");
  append("]\n");
  append("\n");
  /* Helper functions for scalar / unpacked-array port driving and
   * sampling. Keeps the test body small and uniform regardless of
   * whether a port is scalar (`alen == 0`) or a vector
   * (`logic ... [N]`, `alen == N`). */
  append("def _real(v):\n");
  append("    if isinstance(v, bool): return 1.0 if v else 0.0\n");
  append("    return float(v)\n\n");
  append("def _drive(dut, name, val, signed, wl, fl, alen):\n");
  append("    if alen == 0:\n");
  append("        getattr(dut, name).value = pack_fi(_real(val), signed, wl, fl)\n");
  append("    else:\n");
  append("        for k in range(alen):\n");
  append("            getattr(dut, name)[k].value = pack_fi(_real(val[k]), "
         "signed, wl, fl)\n\n");
  append("def _read(dut, name, signed, wl, fl, alen):\n");
  append("    if alen == 0:\n");
  append("        return unpack_fi(int(getattr(dut, name).value), signed, wl, fl)\n");
  append("    return [unpack_fi(int(getattr(dut, name)[k].value), signed, wl, fl) "
         "for k in range(alen)]\n\n");
  append("def _eq(dut_val, ref_val, tol=1e-9, wl=0, fl=0):\n");
  append("    if isinstance(dut_val, list):\n");
  append("        ref_seq = ref_val if isinstance(ref_val, (list, tuple)) else [ref_val]\n");
  append("        if len(dut_val) != len(ref_seq): return False\n");
  append("        return all(_eq(a, b, tol, wl, fl) for a, b in zip(dut_val, ref_seq))\n");
  append("    if abs(float(dut_val) - float(ref_val)) <= tol:\n");
  append("        return True\n");
  /* Sign-interpretation tolerance. The SV emitter doesn't carry the
   * source-side signedness for output ports — its default is
   * `output logic signed [W-1:0]`, while the Python ref's wrap may
   * choose unsigned. The bit pattern is identical; only the
   * harness-side decode differs. Compare the two values modulo
   * 2^WL when both are integer-typed and FL = 0 (the common case
   * for HDL ports), which collapses sign ambiguity. */
  append("    if wl and 1 <= wl <= 64 and fl == 0:\n");
  append("        try:\n");
  append("            mask = (1 << wl) - 1\n");
  append("            return (int(dut_val) & mask) == (int(ref_val) & mask)\n");
  append("        except (TypeError, ValueError):\n");
  append("            return False\n");
  append("    return False\n\n");
  append("def _gen_random(signed, wl, fl, alen):\n");
  append("    if alen == 0:\n");
  append("        lo, hi = fi_range(signed, wl, fl)\n");
  append("        v_packed = pack_fi(random.uniform(lo, hi), signed, wl, fl)\n");
  append("        v_real = unpack_fi(v_packed, signed, wl, fl)\n");
  append("        return v_packed, v_real\n");
  append("    lo, hi = fi_range(signed, wl, fl)\n");
  append("    packed = [pack_fi(random.uniform(lo, hi), signed, wl, fl) "
         "for _ in range(alen)]\n");
  append("    real = [unpack_fi(p, signed, wl, fl) for p in packed]\n");
  append("    return packed, real\n\n");
  /* v3.7 — coverage tracker. Records per-port min / max / count /
   * mean over the run, plus a value histogram for ports narrow
   * enough that one bucket per value is reasonable (WL <= 8). At
   * test end the harness writes `coverage.txt` next to cocotb's
   * own results — a quick "did the random vectors actually
   * exercise the input space" check, useful when chasing
   * unexpected coverage holes. */
  append("class _Coverage:\n");
  append("    def __init__(self, INPUTS, OUTPUTS):\n");
  append("        self.in_specs = INPUTS\n");
  append("        self.out_specs = OUTPUTS\n");
  append("        self.in_stat = [self._fresh() for _ in INPUTS]\n");
  append("        self.out_stat = [self._fresh() for _ in OUTPUTS]\n");
  append("    def _fresh(self):\n");
  append("        return {'count': 0, 'min': None, 'max': None, "
         "'sum': 0.0, 'hist': {}, 'pairs': set(), 'prev': None}\n");
  /* `pairs` holds distinct (prev, curr) tuples for transition
   * (cover_pairs) coverage; `prev` holds the previous scalar
   * value seen on the port so each new sample produces one new
   * candidate pair. Only scalar ports get pair tracking — for
   * unpacked-array ports we'd need to pair element-wise per slot
   * which is rarely useful; can be added if a fixture wants it. */
  append("    def _record(self, stat, val, narrow):\n");
  append("        stat['count'] += 1\n");
  append("        stat['sum'] += float(val)\n");
  append("        if stat['min'] is None or val < stat['min']: "
         "stat['min'] = val\n");
  append("        if stat['max'] is None or val > stat['max']: "
         "stat['max'] = val\n");
  append("        if narrow:\n");
  append("            key = float(val)\n");
  append("            stat['hist'][key] = stat['hist'].get(key, 0) + 1\n");
  append("        if stat['prev'] is not None:\n");
  append("            stat['pairs'].add((float(stat['prev']), float(val)))\n");
  append("        stat['prev'] = val\n");
  append("    def record_input(self, k, value):\n");
  append("        name, signed, wl, fl, alen = self.in_specs[k]\n");
  append("        narrow = wl <= 8\n");
  append("        if alen > 0:\n");
  append("            for v in value: self._record(self.in_stat[k], v, narrow)\n");
  append("        else:\n");
  append("            self._record(self.in_stat[k], value, narrow)\n");
  append("    def record_output(self, j, value):\n");
  append("        name, signed, wl, fl, alen = self.out_specs[j]\n");
  append("        narrow = wl <= 8\n");
  append("        if alen > 0:\n");
  append("            for v in value: self._record(self.out_stat[j], v, narrow)\n");
  append("        else:\n");
  append("            self._record(self.out_stat[j], value, narrow)\n");
  append("    def write(self, path):\n");
  append("        try:\n");
  append("            with open(path, 'w') as f:\n");
  append("                self._write(f)\n");
  append("        except OSError:\n");
  append("            pass  # coverage is best-effort, never fail the test\n");
  append("    def _write(self, f):\n");
  append("        f.write('# matlabc -emit-cocotb coverage report\\n')\n");
  append("        f.write('# Generated by test_<stem>.py at end of run.\\n\\n')\n");
  append("        f.write('## Inputs\\n')\n");
  append("        self._write_section(f, self.in_specs, self.in_stat)\n");
  append("        f.write('\\n## Outputs\\n')\n");
  append("        self._write_section(f, self.out_specs, self.out_stat)\n");
  append("    def _write_section(self, f, specs, stats):\n");
  append("        for spec, stat in zip(specs, stats):\n");
  append("            name, signed, wl, fl, alen = spec\n");
  append("            shape = f'[{alen}]' if alen > 0 else ''\n");
  append("            sign = 'signed' if signed else 'unsigned'\n");
  append("            f.write(f'\\n  {name}{shape} : {sign} {wl} bits, FL={fl}\\n')\n");
  append("            if stat['count'] == 0:\n");
  append("                f.write('    (no samples)\\n')\n");
  append("                continue\n");
  append("            mean = stat['sum'] / stat['count']\n");
  append("            f.write(f'    samples={stat[\"count\"]}  '\n");
  append("                    f'min={stat[\"min\"]}  max={stat[\"max\"]}  '\n");
  append("                    f'mean={mean:.6g}\\n')\n");
  append("            if stat['hist']:\n");
  append("                items = sorted(stat['hist'].items())\n");
  append("                f.write('    histogram:\\n')\n");
  append("                for k, v in items:\n");
  append("                    bar = '#' * min(40, max(1, v))\n");
  append("                    f.write(f'      {k:>10g}  {v:>5d}  {bar}\\n')\n");
  append("            if stat['pairs']:\n");
  append("                f.write(f'    transition pairs: {len(stat[\"pairs\"])}\\n')\n");
  append("\n");

  append("@cocotb.test()\n");
  append("async def test_" + S.Name + "(dut):\n");
  append("    \"\"\"Random-vector lockstep verification of " + S.Name +
         " against the reference Python model. Generated by "
         "matlabc -emit-cocotb.\"\"\"\n");
  /* Seed and vector count are baked in at emit time but overridable
   * via env so triage / sweep workflows don't have to re-emit. The
   * actual values used are logged so they show up in the cocotb log
   * for any failure. */
  append("    _seed = int(os.environ.get(\"COCOTB_SEED\", \"" +
         std::to_string(Seed) + "\"))\n");
  append("    _N    = int(os.environ.get(\"COCOTB_VECTORS\", \"" +
         std::to_string(Vectors) + "\"))\n");
  append("    random.seed(_seed)\n");
  append("    N = _N\n");
  append("    cocotb.log.info(f\"matlabc harness: seed={_seed} "
         "vectors={N}\")\n");
  /* Replay-from-trail mode. When COCOTB_REPLAY_ARGS is set, load
   * the JSONL trail and use its values for every cycle (overriding
   * random / stim / tester values). The per-cycle override happens
   * right before the comparator runs — see the inner loop. N is
   * forced to len(REPLAY) so the harness exits cleanly when the
   * trail ends; latency / sequential bookkeeping continue to work
   * unchanged. */
  append("    REPLAY = None\n");
  append("    _replay_path = os.environ.get(\"COCOTB_REPLAY_ARGS\")\n");
  append("    if _replay_path and os.path.exists(_replay_path):\n");
  append("        with open(_replay_path) as _rf:\n");
  append("            REPLAY = [json.loads(_l) for _l in _rf "
         "if _l.strip()]\n");
  append("        N = len(REPLAY)\n");
  append("        cocotb.log.info(f\"matlabc harness: replaying "
         "{N} cycle(s) from {_replay_path}\")\n");
  /* args_trail accumulates per-cycle (cycle, args dict) for
   * dumping at end-of-test. Always populated, even on
   * non-failing runs — gives users a deterministic trail to
   * convert into a regression test or seed a manual stimulus. */
  append("    args_trail = []\n");
  if (S.Sequential) {
    append("    cocotb.start_soon(Clock(dut.clk, 10, units=\"ns\").start())\n");
    append("    dut.rst_n.value = 0\n");
    /* Initial inputs to known values so the always_ff doesn't sample
     * X / Z on the first pre-reset cycle. */
    append("    for name, signed, wl, fl, alen in INPUTS:\n");
    append("        if alen == 0:\n");
    append("            getattr(dut, name).value = 0\n");
    append("        else:\n");
    append("            for k in range(alen):\n");
    append("                getattr(dut, name)[k].value = 0\n");
    append("    await RisingEdge(dut.clk)\n");
    append("    await RisingEdge(dut.clk)\n");
    append("    dut.rst_n.value = 1\n");
  }
  append("    LATENCY = " + std::to_string(Latency) + "\n");
  append("    failures = 0\n");
  append("    coverage = _Coverage(INPUTS, OUTPUTS)\n");
  append("    compared = 0\n");
  /* The reference is called at cycle k; the DUT's response to that
   * cycle's inputs surfaces L cycles later. Recorded refs sit in
   * a FIFO until their corresponding DUT sample comes due. We do
   * NOT flush the pipeline with zero inputs at the tail — for
   * stateful DUTs (FIR with feedback through delay_line, etc.)
   * driving zeros would corrupt the late-cycle DUT samples.
   * Instead we just skip comparison for the first L cycles
   * (pipeline warm-up) and run for N cycles total, yielding
   * N-L valid comparisons. Mirrors HDL Verifier's `Latency`
   * parameter contract: drive N stimulus, compare from cycle L. */
  append("    pending_refs = []  # FIFO of (cycle, ref_tuple, args)\n");
  if (Stimulus) {
    /* Tester-driven mode (v3.2): the input vector list was extracted
     * from a sibling `test_<stem>.m` and embedded as STIMULUS below.
     * Each iteration drives the next tuple instead of generating
     * random values. */
    append("    STIMULUS = [\n");
    for (auto &Tup : *Stimulus) {
      append("        (");
      for (size_t I = 0; I < Tup.size(); ++I) {
        if (I) append(", ");
        append(Tup[I]);
      }
      if (Tup.size() == 1) append(",");
      append("),\n");
    }
    append("    ]\n");
    append("    N = len(STIMULUS)\n");
    append("    for i, args in enumerate(STIMULUS):\n");
    append("        py_args = list(args)\n");
    append("        for j, (name, signed, wl, fl, alen) in enumerate(INPUTS):\n");
    append("            _drive(dut, name, py_args[j], signed, wl, fl, alen)\n");
    append("            coverage.record_input(j, _real(py_args[j]) "
           "if alen == 0 else [_real(x) for x in py_args[j]])\n");
  } else {
    /* Per-input hold cycles from `% cocotb: hold(<name>, N)` pragmas.
     * `HOLD[k] = N` means input k stays at the same value for N
     * consecutive iterations before a fresh random value is drawn.
     * 0 / 1 both mean "advance every iteration" (the default). */
    append("    HOLD = [");
    for (size_t I = 0; I < S.Inputs.size(); ++I) {
      if (I) append(", ");
      int H = S.Inputs[I].HoldCycles;
      if (H < 1) H = 1;
      append(std::to_string(H));
    }
    append("]\n");
    /* Per-input stimulus shape from `% cocotb: stimulus(...)` pragmas.
     * Tuple (kind, arg1, arg2) where kind is one of:
     *   "random"   — fresh draw per cycle (default).
     *   "impulse"  — arg1 at cycle 0, zeros after.
     *   "constant" — arg1 every cycle.
     *   "ramp"     — arg1 + cycle*arg2.
     * Unblocks pipelined DUTs whose per-call reference doesn't
     * agree with random per-cycle inputs (sequential_processor). */
    append("    STIM = [\n");
    for (auto &P : S.Inputs) {
      const char *K = "random";
      switch (P.Stim) {
        case CocotbPortSpec::StimKind::Random:   K = "random"; break;
        case CocotbPortSpec::StimKind::Impulse:  K = "impulse"; break;
        case CocotbPortSpec::StimKind::Constant: K = "constant"; break;
        case CocotbPortSpec::StimKind::Ramp:     K = "ramp"; break;
        case CocotbPortSpec::StimKind::Range:    K = "range"; break;
      }
      char Buf[128];
      snprintf(Buf, sizeof Buf,
               "        (\"%s\", %.17g, %.17g),\n",
               K, P.StimArg1, P.StimArg2);
      append(Buf);
    }
    append("    ]\n");
    append("    held_real = [None] * len(INPUTS)\n");
    append("    held_packed = [None] * len(INPUTS)\n");
    append("    cycles_left = [0] * len(INPUTS)\n");
    append("    def _stim_value(i, kind, a1, a2, signed, wl, fl, alen):\n");
    append("        if kind == 'impulse':\n");
    append("            v = a1 if i == 0 else 0.0\n");
    append("            if alen == 0:\n");
    append("                p = pack_fi(v, signed, wl, fl)\n");
    append("                return p, unpack_fi(p, signed, wl, fl)\n");
    append("            packed = [pack_fi(v, signed, wl, fl) for _ in range(alen)]\n");
    append("            return packed, [unpack_fi(p, signed, wl, fl) for p in packed]\n");
    append("        if kind == 'constant':\n");
    append("            v = a1\n");
    append("            if alen == 0:\n");
    append("                p = pack_fi(v, signed, wl, fl)\n");
    append("                return p, unpack_fi(p, signed, wl, fl)\n");
    append("            packed = [pack_fi(v, signed, wl, fl) for _ in range(alen)]\n");
    append("            return packed, [unpack_fi(p, signed, wl, fl) for p in packed]\n");
    append("        if kind == 'ramp':\n");
    append("            v = a1 + i * a2\n");
    append("            if alen == 0:\n");
    append("                p = pack_fi(v, signed, wl, fl)\n");
    append("                return p, unpack_fi(p, signed, wl, fl)\n");
    append("            packed = [pack_fi(v + j * a2, signed, wl, fl) for j in range(alen)]\n");
    append("            return packed, [unpack_fi(p, signed, wl, fl) for p in packed]\n");
    append("        if kind == 'range':\n");
    /* `range` constrains the random stimulus to a real-value window
     * (a1=lo, a2=hi).  Used when SV erases the fi FL on the port list
     * and the natural fi_range would let stimulus overflow the SV's
     * mid-computation truncation.  See DL HDL H3. */
    append("            lo, hi = a1, a2\n");
    append("            if alen == 0:\n");
    append("                v = random.uniform(lo, hi)\n");
    append("                p = pack_fi(v, signed, wl, fl)\n");
    append("                return p, unpack_fi(p, signed, wl, fl)\n");
    append("            packed = [pack_fi(random.uniform(lo, hi), signed, wl, fl) for _ in range(alen)]\n");
    append("            return packed, [unpack_fi(p, signed, wl, fl) for p in packed]\n");
    append("        # default: random\n");
    append("        return _gen_random(signed, wl, fl, alen)\n");
    append("    for i in range(N):\n");
    append("        py_args = []\n");
    append("        for k, (name, signed, wl, fl, alen) in enumerate(INPUTS):\n");
    append("            kind, a1, a2 = STIM[k]\n");
    append("            if kind != 'random':\n");
    append("                # Deterministic stimulus — recompute every cycle.\n");
    append("                held_packed[k], held_real[k] = _stim_value("
           "i, kind, a1, a2, signed, wl, fl, alen)\n");
    append("            elif cycles_left[k] == 0:\n");
    append("                held_packed[k], held_real[k] = "
           "_gen_random(signed, wl, fl, alen)\n");
    append("                cycles_left[k] = HOLD[k]\n");
    append("            if kind == 'random':\n");
    append("                cycles_left[k] -= 1\n");
    /* Drive: scalar uses .value =, vector uses [k].value = per element. */
    append("            if alen == 0:\n");
    append("                getattr(dut, name).value = held_packed[k]\n");
    append("            else:\n");
    append("                for kk in range(alen):\n");
    append("                    getattr(dut, name)[kk].value = held_packed[k][kk]\n");
    /* Reference args: scalar Q.0 → int, scalar Q.F → float, vector → list. */
    append("            if alen > 0:\n");
    append("                py_args.append([int(x) if fl == 0 else x "
           "for x in held_real[k]])\n");
    append("            else:\n");
    append("                py_args.append(int(held_real[k]) if fl == 0 "
           "else held_real[k])\n");
    append("            coverage.record_input(k, held_real[k])\n");
  }
  /* Replay-mode override + args trail capture. After the per-input
   * loop has built py_args (in either branch) and driven the DUT,
   * if REPLAY is set we re-pack the recorded values for this
   * cycle and re-drive — slight redundancy but keeps the override
   * logic in one spot rather than tangled across both branches.
   * args_trail captures the final py_args (post-override) for
   * dump at end-of-test. */
  append("        if REPLAY is not None and i < len(REPLAY):\n");
  append("            _rec = REPLAY[i].get(\"args\", {})\n");
  append("            for k, (name, signed, wl, fl, alen) in enumerate(INPUTS):\n");
  append("                if name not in _rec: continue\n");
  append("                v = _rec[name]\n");
  append("                if alen == 0:\n");
  append("                    py_args[k] = (int(v) if fl == 0 else float(v))\n");
  append("                    getattr(dut, name).value = "
         "pack_fi(_real(v), signed, wl, fl)\n");
  append("                else:\n");
  append("                    py_args[k] = list(v)\n");
  append("                    for kk in range(alen):\n");
  append("                        getattr(dut, name)[kk].value = "
         "pack_fi(_real(v[kk]), signed, wl, fl)\n");
  append("        _trail_args = {}\n");
  append("        for k, (name, signed, wl, fl, alen) in enumerate(INPUTS):\n");
  append("            if alen > 0:\n");
  append("                _trail_args[name] = [int(x) if fl == 0 "
         "else float(x) for x in py_args[k]]\n");
  append("            else:\n");
  append("                _trail_args[name] = (int(py_args[k]) if fl == 0 "
         "else float(py_args[k]))\n");
  append("        args_trail.append({\"cycle\": i, \"args\": _trail_args})\n");
  if (S.Sequential) {
    /* Sequential DUTs need TWO sample windows per cycle:
     *   - pre-edge: combinational outputs reflecting f(old_state,
     *     new_inputs). Mealy-style outputs match here (their value
     *     for the cycle's input is computed with the still-valid
     *     state).
     *   - post-edge: registered outputs and Moore-style
     *     combinational outputs (depend only on the just-latched
     *     state). Most outputs match here.
     * The harness samples both, then per-output accepts whichever
     * matches the reference. Mealy alignment without per-output
     * kind metadata.
     *
     * Drive inputs first, then a 1ns settle so always_comb
     * propagates the new inputs through the still-valid old state
     * — that's the pre-edge sample. await RisingEdge advances the
     * state; another 1ns settle lets always_comb re-evaluate with
     * the new state — that's the post-edge sample. */
    append("        await Timer(1, units=\"ns\")\n");
    append("        pre_samples = []\n");
    append("        for j, (name, signed, wl, fl, alen) in enumerate(OUTPUTS):\n");
    append("            try:\n");
    append("                pre_samples.append(_read(dut, name, signed, wl, fl, alen))\n");
    append("            except Exception:\n");
    append("                pre_samples.append(None)\n");
    append("        await RisingEdge(dut.clk)\n");
    append("        await Timer(1, units=\"ns\")\n");
  } else {
    append("        await Timer(1, units=\"ns\")\n");
  }
  append("        ref = " + S.Name + "(*py_args)\n");
  append("        if not isinstance(ref, tuple):\n");
  append("            ref = (ref,)\n");
  if (S.Sequential)
    append("        pending_refs.append((i, ref, list(py_args), "
           "list(pre_samples)))\n");
  else
    append("        pending_refs.append((i, ref, list(py_args), None))\n");
  append("        if i >= LATENCY:\n");
  append("            popped = pending_refs.pop(0)\n");
  append("            ref_cycle, ref, ref_args, pre_samp = popped\n");
  append("            compared += 1\n");
  append("            for j, (name, signed, wl, fl, alen) in enumerate(OUTPUTS):\n");
  append("                try:\n");
  append("                    post_val = _read(dut, name, signed, wl, fl, alen)\n");
  append("                except Exception as exc:\n");
  append("                    cocotb.log.error(f\"#{ref_cycle} {name}: cannot "
         "read DUT signal: {exc}\")\n");
  append("                    failures += 1\n");
  append("                    continue\n");
  append("                coverage.record_output(j, post_val)\n");
  append("                ref_val = ref[j]\n");
  /* For sequential modules, accept either the pre-edge or
   * post-edge sample as a match. The pre-edge value is captured
   * before the rising edge, post-edge after. Mealy outputs match
   * pre-edge; Moore / counter / FIR outputs match post-edge. */
  append("                pre_val = pre_samp[j] if pre_samp else None\n");
  append("                post_match = _eq(post_val, ref_val, wl=wl, fl=fl)\n");
  append("                pre_match = (pre_val is not None and "
         "_eq(pre_val, ref_val, wl=wl, fl=fl))\n");
  append("                if post_match or pre_match:\n");
  append("                    continue\n");
  /* A2: enriched mismatch diagnostics. Decode each value as fi
   * (so signed widths show their effective range, e.g. `42 [i16
   * range -32768..32767]`), and surface canonical fault hints
   * when the divergence shape matches a known root cause. The
   * raw post/pre/ref/args line stays first for easy grep / log
   * scraping; the hint lines follow indented for readability. */
  append("                fi_tag = (lambda v: f\"{int(v)} ["
         "{'signed' if signed else 'unsigned'} {wl}b "
         "0x{int(v) & ((1<<wl)-1):X}]\" if isinstance(v, (int, float)) "
         "and abs(float(v)) < (1<<63) else str(v))\n");
  append("                hints = []\n");
  /* Sign-interpretation hint — DUT and ref are bit-equivalent
   * modulo 2^WL but the harness reads them with different
   * signedness. Already auto-equated by the modulo-WL fallback
   * in _eq, but if it hits and still fails, surface the bits. */
  append("                try:\n");
  append("                    if wl and 1 <= wl <= 64:\n");
  append("                        m = (1 << wl) - 1\n");
  append("                        if (int(post_val) & m) == (int(ref_val) & m):\n");
  append("                            hints.append(\"sign-interpretation: "
         "bits match modulo 2^{wl}\")\n");
  append("                except (TypeError, ValueError): pass\n");
  /* Saturation-suspected hint — DUT pinned to ±max while ref
   * is well outside the legal range. */
  append("                try:\n");
  append("                    if wl and 2 <= wl <= 64:\n");
  append("                        if signed:\n");
  append("                            hi = (1 << (wl - 1)) - 1\n");
  append("                            lo = -(1 << (wl - 1))\n");
  append("                        else:\n");
  append("                            hi = (1 << wl) - 1\n");
  append("                            lo = 0\n");
  append("                        if int(post_val) in (hi, lo) and "
         "(int(ref_val) > hi or int(ref_val) < lo):\n");
  append("                            hints.append(\"saturation suspected: "
         "DUT pinned to {hi if int(post_val) == hi else lo}, ref outside "
         "[{lo}..{hi}]\")\n");
  append("                except (TypeError, ValueError): pass\n");
  /* Latency-suspected hint — ref pre value (pre_samp) matches
   * post (i.e. DUT lagged one cycle behind). */
  append("                if pre_val is not None and pre_val == ref_val and "
         "pre_val != post_val:\n");
  append("                    hints.append(\"latency suspected: pre-edge "
         "sample matched ref; consider increasing latency by 1\")\n");
  append("                cocotb.log.error(\n");
  append("                    f\"#{ref_cycle} {name}: post={post_val} "
         "pre={pre_val} ref={ref_val} args={ref_args}\")\n");
  append("                cocotb.log.error(\"  decoded: post=\" + "
         "fi_tag(post_val) + \" ref=\" + fi_tag(ref_val))\n");
  append("                for h in hints:\n");
  append("                    cocotb.log.error(\"  hint: \" + h)\n");
  /* VCD pointer — the Makefile already enables --trace, so
   * dump.vcd lives next to the run. Print on first failure so
   * the user sees the path immediately rather than digging it
   * out of the cocotb working dir. */
  append("                if failures == 0:\n");
  append("                    _dut_path = os.path.dirname("
         "os.path.abspath(__file__))\n");
  append("                    cocotb.log.error(\"  trace: \" + "
         "_dut_path + \"/dump.vcd (open in GTKWave / Surfer)\")\n");
  append("                failures += 1\n");
  /* v3.7 — write coverage report next to the cocotb run output.
   * Best-effort: a write failure (read-only fs, permissions)
   * doesn't fail the test. Filename is fixed (`coverage.txt`) in
   * the harness directory so the Makefile's per-stem layout
   * keeps multiple harnesses' reports from clobbering each
   * other for the standard `just test-cocotb` workflow. */
  append("    coverage.write(os.path.join(os.path.dirname("
         "os.path.abspath(__file__)), 'coverage.txt'))\n");
  /* args_trail dump — best-effort, mirrors coverage.txt. The
   * trail is the canonical reproducer for any failure: feed it
   * back via COCOTB_REPLAY_ARGS to drive the same input sequence
   * deterministically, even after editing the source or
   * regenerating with a different seed. Always written, not just
   * on failure, so a successful run also captures a known-good
   * regression pin. */
  append("    try:\n");
  append("        _tp = os.path.join(os.path.dirname("
         "os.path.abspath(__file__)), 'args_trail.jsonl')\n");
  append("        with open(_tp, 'w') as _tf:\n");
  append("            for _r in args_trail:\n");
  append("                _tf.write(json.dumps(_r) + '\\n')\n");
  append("    except OSError:\n");
  append("        pass\n");
  append("    assert failures == 0, "
         "f\"{failures} mismatch(es) across {compared} compared cycles "
         "(latency={LATENCY}, total stimulus N={N})\"\n");
  // Coverage gate (`% cocotb: cover(<port>, min_bins=N)`). After
  // the comparison sweep, count distinct values seen on each
  // covered port and assert the count meets the threshold. Catches
  // "random vectors only exercised half the FSM states" silently.
  if (!CoverMinBins.empty()) {
    append("    cover_failures = []\n");
    for (auto &Kv : CoverMinBins) {
      const std::string &Nm = Kv.first;
      int Min = Kv.second;
      append("    for k, spec in enumerate(INPUTS):\n");
      append("        if spec[0] == \"" + Nm + "\":\n");
      append("            seen = len(coverage.in_stat[k]['hist']) "
             "if coverage.in_stat[k]['hist'] else "
             "(coverage.in_stat[k]['count'] and 1 or 0)\n");
      append("            if seen < " + std::to_string(Min) + ":\n");
      append("                cover_failures.append("
             "f\"input '" + Nm + "' hit {seen} bin(s), expected >= "
             + std::to_string(Min) + "\")\n");
      append("    for j, spec in enumerate(OUTPUTS):\n");
      append("        if spec[0] == \"" + Nm + "\":\n");
      append("            seen = len(coverage.out_stat[j]['hist']) "
             "if coverage.out_stat[j]['hist'] else "
             "(coverage.out_stat[j]['count'] and 1 or 0)\n");
      append("            if seen < " + std::to_string(Min) + ":\n");
      append("                cover_failures.append("
             "f\"output '" + Nm + "' hit {seen} bin(s), expected >= "
             + std::to_string(Min) + "\")\n");
    }
    append("    assert not cover_failures, "
           "\"coverage gate failed:\\n  \" + \"\\n  \".join(cover_failures)\n");
  }
  /* cover_pairs(<port>, min_pairs=N): assert >= N distinct
   * (prev, curr) consecutive pairs were seen on the named port.
   * Transition coverage — what FSMs need beyond raw bin counts. */
  if (!CoverPairsMin.empty()) {
    append("    pair_failures = []\n");
    for (auto &Kv : CoverPairsMin) {
      const std::string &Nm = Kv.first;
      int Min = Kv.second;
      append("    for k, spec in enumerate(INPUTS):\n");
      append("        if spec[0] == \"" + Nm + "\":\n");
      append("            seen = len(coverage.in_stat[k]['pairs'])\n");
      append("            if seen < " + std::to_string(Min) + ":\n");
      append("                pair_failures.append("
             "f\"input '" + Nm + "' hit {seen} pair(s), expected >= "
             + std::to_string(Min) + "\")\n");
      append("    for j, spec in enumerate(OUTPUTS):\n");
      append("        if spec[0] == \"" + Nm + "\":\n");
      append("            seen = len(coverage.out_stat[j]['pairs'])\n");
      append("            if seen < " + std::to_string(Min) + ":\n");
      append("                pair_failures.append("
             "f\"output '" + Nm + "' hit {seen} pair(s), expected >= "
             + std::to_string(Min) + "\")\n");
    }
    append("    assert not pair_failures, "
           "\"transition coverage gate failed:\\n  \" + "
           "\"\\n  \".join(pair_failures)\n");
  }
  /* cover_range(<port>): assert every value in the port's full
   * fi range [lo..hi] was observed. Only meaningful for narrow
   * ports; we cap at WL <= 8 (256 values) and produce a build-
   * time-equivalent runtime check (the harness asserts the WL
   * limit so a misuse fails loudly instead of silently passing). */
  if (!CoverRange.empty()) {
    append("    range_failures = []\n");
    for (auto &Nm : CoverRange) {
      append("    for k, spec in enumerate(INPUTS):\n");
      append("        if spec[0] == \"" + Nm + "\":\n");
      append("            _, _signed, _wl, _fl, _ = spec\n");
      append("            assert _wl <= 8, "
             "\"cover_range only supported for WL <= 8 ports "
             "(port '" + Nm + "' is wider)\"\n");
      append("            assert _fl == 0, "
             "\"cover_range only supported for integer ports "
             "(port '" + Nm + "' has FL != 0)\"\n");
      append("            lo, hi = fi_range(_signed, _wl, _fl)\n");
      append("            seen = set(coverage.in_stat[k]['hist'].keys())\n");
      append("            expected = set(float(v) for v in "
             "range(int(lo), int(hi) + 1))\n");
      append("            missing = sorted(expected - seen)\n");
      append("            if missing:\n");
      append("                range_failures.append("
             "f\"input '" + Nm + "' missing {len(missing)} value(s) "
             "in [{int(lo)}..{int(hi)}]: \" + str(missing[:8]) + "
             "(' ...' if len(missing) > 8 else ''))\n");
      append("    for j, spec in enumerate(OUTPUTS):\n");
      append("        if spec[0] == \"" + Nm + "\":\n");
      append("            _, _signed, _wl, _fl, _ = spec\n");
      append("            assert _wl <= 8, "
             "\"cover_range only supported for WL <= 8 ports "
             "(port '" + Nm + "' is wider)\"\n");
      append("            assert _fl == 0, "
             "\"cover_range only supported for integer ports "
             "(port '" + Nm + "' has FL != 0)\"\n");
      append("            lo, hi = fi_range(_signed, _wl, _fl)\n");
      append("            seen = set(coverage.out_stat[j]['hist'].keys())\n");
      append("            expected = set(float(v) for v in "
             "range(int(lo), int(hi) + 1))\n");
      append("            missing = sorted(expected - seen)\n");
      append("            if missing:\n");
      append("                range_failures.append("
             "f\"output '" + Nm + "' missing {len(missing)} value(s) "
             "in [{int(lo)}..{int(hi)}]: \" + str(missing[:8]) + "
             "(' ...' if len(missing) > 8 else ''))\n");
    }
    append("    assert not range_failures, "
           "\"range coverage gate failed:\\n  \" + "
           "\"\\n  \".join(range_failures)\n");
  }
  return Out;
}

static std::string renderCocotbMakefile(const std::string &Stem,
                                         const std::string &DutName) {
  std::string M;
  M += "# Generated by matlabc -emit-cocotb. Do not edit.\n";
  M += "TOPLEVEL_LANG ?= verilog\n";
  M += "SIM           ?= verilator\n";
  M += "EXTRA_ARGS    += --trace --trace-structs\n";
  // The emitted SV relies on Verilog's implicit width extension in
  // mixed-width arithmetic (e.g. acc + x); Verilator 5.x escalates that to a
  // fatal WIDTHEXPAND/WIDTHTRUNC lint, so silence the width-style warnings.
  M += "EXTRA_ARGS    += -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC\n";
  M += "VERILOG_SOURCES = $(PWD)/" + Stem + ".sv\n";
  M += "TOPLEVEL = " + DutName + "\n";
  M += "MODULE   = test_" + Stem + "\n";
  M += "\n";
  M += "include $(shell cocotb-config --makefiles)/Makefile.sim\n";
  M += "\n";
  /* Sweep target: run N seeds (1..N) back-to-back, log per-seed
   * pass/fail, and report which seeds failed. Useful for catching
   * edge-case bugs the default seed=42 happens to miss. The
   * harness reads COCOTB_SEED from env (see the emitted
   * test_<stem>.py), so we just vary the env per invocation; the
   * Verilator build is reused since the SV is unchanged. Override
   * count with `make sweep N=50`. */
  M += ".PHONY: sweep\n";
  M += "sweep:\n";
  M += "\t@N=$${N:-20}; FAIL=\"\"; PASS=0; \\\n";
  M += "\techo \"sweep: $$N seeds on " + Stem + "\"; \\\n";
  M += "\tfor s in `seq 1 $$N`; do \\\n";
  M += "\t  if COCOTB_SEED=$$s $(MAKE) -s 2>/dev/null >/dev/null; then \\\n";
  M += "\t    printf \"  seed=%-4d PASS\\n\" $$s; PASS=$$((PASS+1)); \\\n";
  M += "\t  else \\\n";
  M += "\t    printf \"  seed=%-4d FAIL\\n\" $$s; FAIL=\"$$FAIL $$s\"; \\\n";
  M += "\t  fi; \\\n";
  M += "\tdone; \\\n";
  M += "\tif [ -n \"$$FAIL\" ]; then \\\n";
  M += "\t  echo \"\"; echo \"sweep: FAIL on seeds:$$FAIL\"; \\\n";
  M += "\t  echo \"replay one with: COCOTB_SEED=<n> make\"; exit 1; \\\n";
  M += "\telse echo \"sweep: $$PASS/$$N seeds passed\"; fi\n";
  M += "\n";
  /* Replay target: drive the exact stimulus sequence captured in
   * args_trail.jsonl from the previous run. Useful for: turning a
   * failing run into a deterministic regression repro, or pinning
   * a known-good trail before editing the source. Override the
   * trail file with `make replay TRAIL=other_trail.jsonl`. */
  M += ".PHONY: replay\n";
  M += "replay:\n";
  M += "\t@TRAIL=$${TRAIL:-args_trail.jsonl}; \\\n";
  M += "\tif [ ! -f \"$$TRAIL\" ]; then \\\n";
  M += "\t  echo \"replay: $$TRAIL not found; run \\`make\\` first to "
       "produce one\"; exit 1; \\\n";
  M += "\tfi; \\\n";
  M += "\techo \"replay: driving stimulus from $$TRAIL\"; \\\n";
  M += "\tCOCOTB_REPLAY_ARGS=$$PWD/$$TRAIL $(MAKE) -s\n";
  return M;
}

/* mkdir -p equivalent. The harness lives at most one directory deep
 * relative to the input (`<input-dir>/<stem>_cocotb/`), so we only
 * need to create one level. ENOENT on the parent is left for the
 * caller to surface — matches the GNU `mkdir` default behaviour. */
static int mkdirIfNeeded(const std::string &Dir) {
  struct stat St;
  if (stat(Dir.c_str(), &St) == 0) {
    if (S_ISDIR(St.st_mode)) return 0;
    std::cerr << "error: " << Dir << " exists but is not a directory\n";
    return 1;
  }
  if (mkdir(Dir.c_str(), 0755) < 0) {
    std::perror(("mkdir " + Dir).c_str());
    return 1;
  }
  return 0;
}

static std::string shellQuote(const std::string &S) {
  std::string Q = "'";
  for (char C : S) {
    if (C == '\'') Q += "'\\''";
    else Q += C;
  }
  Q += "'";
  return Q;
}

static int runMatlabcEmit(const std::string &Self, const std::string &Mode,
                          const std::string &Input, const std::string &Out) {
  std::string Cmd = shellQuote(Self) + " " + Mode + " " +
                    shellQuote(Input) + " > " + shellQuote(Out);
  int Rc = std::system(Cmd.c_str());
  if (Rc != 0) {
    std::cerr << "error: command failed (rc=" << Rc << "): " << Cmd << "\n";
    return 1;
  }
  return 0;
}

static int writeStringToFile(const std::string &Path, const std::string &S) {
  std::ofstream F(Path);
  if (!F) {
    std::cerr << "error: cannot write " << Path << "\n";
    return 1;
  }
  F << S;
  return 0;
}

// Tier-7d — whole-diagram cocotb SIL emit. Driven from
// `matlabc -emit-cocotb <model.mflow> --dut <block>`. We self-invoke
// `-emit-systemverilog --subsystem <dut-flow>` to get the SV, then
// `-emit-python --subsystem <dut-flow>` for the host reference,
// parse the SV's port list as the source of truth for DUT signals,
// and finally call `emitDiagramCocotbHarness` to render the
// testbench Python file.
//
// Returns 0 on success, non-zero on any failure (with diagnostics
// already printed). Mirrors `emitCocotbHarness`'s I/O contract so the
// EmitCocotb dispatch can call either depending on the input shape.
static int emitCocotbHarnessForDiagram(const char *Self,
                                        const Options &Opts,
                                        matlab::flowchart::FlowDoc &Doc,
                                        DiagnosticEngine &Diag) {
  const matlab::flowchart::Flow *Entry = Doc.findFlow(Doc.Entry);
  if (!Entry) {
    std::cerr << "error: -emit-cocotb: entry flow \"" << Doc.Entry
              << "\" not found\n";
    return 1;
  }

  // Parse `--dut a,b,c` (comma-separated) into a list of block ids.
  // Single-DUT runs supply one id; multi-DUT exercises the wrapper
  // SV generator below.
  std::vector<std::string> DutBlockIds;
  {
    const std::string &S = Opts.CocotbDut;
    size_t Pos = 0;
    while (Pos <= S.size()) {
      size_t Sep = S.find(',', Pos);
      std::string Part = (Sep == std::string::npos) ? S.substr(Pos)
                                                    : S.substr(Pos, Sep - Pos);
      while (!Part.empty() && std::isspace((unsigned char)Part.front()))
        Part.erase(Part.begin());
      while (!Part.empty() && std::isspace((unsigned char)Part.back()))
        Part.pop_back();
      if (!Part.empty()) DutBlockIds.push_back(Part);
      if (Sep == std::string::npos) break;
      Pos = Sep + 1;
    }
  }
  if (DutBlockIds.empty()) {
    std::cerr << "error: -emit-cocotb: --dut requires at least one "
                 "block id (comma-separated list for multi-DUT)\n";
    return 1;
  }

  // Output dir: <model_stem>_cocotb (or user override).
  std::string Input = Opts.InputPath;
  std::string Stem = Input;
  size_t Slash = Stem.find_last_of('/');
  if (Slash != std::string::npos) Stem = Stem.substr(Slash + 1);
  size_t Dot = Stem.find_last_of('.');
  if (Dot != std::string::npos) Stem = Stem.substr(0, Dot);
  std::string OutDir = Opts.CocotbOutDir;
  if (OutDir.empty()) {
    std::string Parent = ".";
    if (Slash != std::string::npos) Parent = Input.substr(0, Slash);
    OutDir = Parent + "/" + Stem + "_cocotb";
  }
  if (mkdirIfNeeded(OutDir) != 0) return 1;

  auto camel = [](const std::string &S) {
    std::string Out; bool Up = true;
    for (char C : S) {
      if (C == '_' || C == '-') { Up = true; continue; }
      Out.push_back(Up ? std::toupper((unsigned char)C) : C);
      Up = false;
    }
    return Out;
  };

  // Per-DUT resolution: locate the block + its referenced flow,
  // self-invoke -emit-sv + -emit-python, parse the SV port list.
  struct DutCtx {
    std::string BlockId;
    const matlab::flowchart::Node *Node;
    const matlab::flowchart::Flow *Flow;
    std::optional<CocotbFuncSpec> Spec;
    std::string DutStem;
  };
  std::vector<DutCtx> DutCtxs;
  for (const auto &BId : DutBlockIds) {
    DutCtx D;
    D.BlockId = BId;
    D.Node = nullptr;
    for (const auto &N : Entry->Nodes) {
      if (N.Id == BId) { D.Node = &N; break; }
    }
    if (!D.Node) {
      std::cerr << "error: -emit-cocotb: --dut block \"" << BId
                << "\" not found in entry flow \"" << Doc.Entry << "\"\n";
      return 1;
    }
    if (D.Node->Kind != "signal_subsystem") {
      std::cerr << "error: -emit-cocotb: --dut block \"" << BId
                << "\" must be a signal_subsystem (kind is \""
                << D.Node->Kind << "\")\n";
      return 1;
    }
    const std::string *FlowId = D.Node->getData("flow_id");
    if (!FlowId || FlowId->empty()) {
      std::cerr << "error: -emit-cocotb: --dut block \"" << BId
                << "\" missing data.flow_id\n";
      return 1;
    }
    D.Flow = nullptr;
    for (const auto &F : Doc.Flows) {
      if (F.Id == *FlowId) { D.Flow = &F; break; }
    }
    if (!D.Flow) {
      std::cerr << "error: -emit-cocotb: flow with id \"" << *FlowId
                << "\" (referenced by --dut block \"" << BId
                << "\") not found\n";
      return 1;
    }
    D.DutStem = D.Flow->Name;
    std::string SVPath = OutDir + "/" + D.DutStem + ".sv";
    std::string RefPyPath = OutDir + "/" + D.DutStem + "_ref.py";
    std::string DutFlag = "--subsystem " + shellQuote(D.Flow->Name);
    if (runMatlabcEmit(Self, "-emit-systemverilog " + DutFlag, Input,
                       SVPath) != 0)
      return 1;
    if (runMatlabcEmit(Self, "-emit-python " + DutFlag, Input,
                       RefPyPath) != 0)
      return 1;
    D.Spec = parseCocotbSpecFromSv(SVPath, D.DutStem);
    if (!D.Spec) {
      std::cerr << "error: -emit-cocotb: failed to parse port list from "
                << SVPath << "\n";
      return 1;
    }
    DutCtxs.push_back(std::move(D));
  }
  bool MultiDut = DutCtxs.size() > 1;

  // Build DiagramCocotbOptions.
  matlab::flowchart::DiagramCocotbOptions DCO;
  DCO.Tolerance = Opts.CocotbTolerance;
  DCO.FiFrac = 16;
  if (!DutCtxs.front().Spec->Inputs.empty()) {
    DCO.FiWidth = (int)DutCtxs.front().Spec->Inputs.front().WL;
    DCO.FiSigned = DutCtxs.front().Spec->Inputs.front().Signed;
  }
  DCO.Latency = Opts.CocotbLatency;
  for (const auto &D : DutCtxs) {
    matlab::flowchart::DiagramCocotbOptions::DutSpec Spec;
    Spec.BlockId = D.BlockId;
    Spec.ModuleName = D.Spec->Name;
    Spec.RefModule = D.DutStem + "_ref";
    Spec.RefClass = camel(D.Flow->Name);
    Spec.Sequential = D.Spec->Sequential;
    for (const auto &P : D.Spec->Inputs) {
      if (P.Name == "reset") continue;
      Spec.InputPorts.push_back(P.Name);
    }
    for (const auto &P : D.Spec->Outputs) Spec.OutputPorts.push_back(P.Name);
    DCO.Duts.push_back(std::move(Spec));
  }

  // Multi-DUT — synthesise a wrapper SV that instantiates every
  // DUT side-by-side. Each DUT's ports are exposed at the wrapper
  // boundary prefixed with `<block_id>__` so cocotb addresses them
  // by `dut.<block_id>__<port>`. clk / rst_n / reset (when any DUT
  // is sequential) are wrapper-level signals fanned out to each
  // DUT instance.
  std::string WrapperModule;
  if (MultiDut) {
    WrapperModule = Stem + "_wrapper";
    DCO.WrapperModule = WrapperModule;
    bool AnySeq = false;
    for (const auto &D : DutCtxs) if (D.Spec->Sequential) AnySeq = true;
    std::ostringstream W;
    W << "// Generated by matlabc -emit-cocotb (multi-DUT wrapper).\n";
    W << "// Instantiates "
      << DutCtxs.size() << " DUTs side-by-side and re-exposes their\n";
    W << "// I/O at the wrapper boundary prefixed with `<block_id>__`.\n\n";
    W << "module " << WrapperModule << " (\n";
    bool First = true;
    auto comma = [&]() { if (!First) W << ",\n"; First = false; };
    if (AnySeq) {
      comma(); W << "    input  logic clk";
      comma(); W << "    input  logic rst_n";
      comma(); W << "    input  logic reset";
    }
    for (const auto &D : DutCtxs) {
      for (const auto &P : D.Spec->Inputs) {
        if (P.Name == "reset") continue;
        comma();
        W << "    input  logic";
        if (P.Signed) W << " signed";
        if (P.WL > 1) W << " [" << (P.WL - 1) << ":0]";
        W << " " << D.BlockId << "__" << P.Name;
      }
      for (const auto &P : D.Spec->Outputs) {
        comma();
        W << "    output logic";
        if (P.Signed) W << " signed";
        if (P.WL > 1) W << " [" << (P.WL - 1) << ":0]";
        W << " " << D.BlockId << "__" << P.Name;
      }
    }
    W << "\n);\n\n";
    for (const auto &D : DutCtxs) {
      W << "    " << D.Spec->Name << " " << D.BlockId << "_inst (\n";
      bool FirstP = true;
      auto pcomma = [&]() { if (!FirstP) W << ",\n"; FirstP = false; };
      if (D.Spec->Sequential) {
        pcomma(); W << "        .clk(clk)";
        pcomma(); W << "        .rst_n(rst_n)";
        // Only wire .reset(reset) if the DUT actually has a `reset` port.
        bool HasReset = false;
        for (const auto &P : D.Spec->Inputs)
          if (P.Name == "reset") { HasReset = true; break; }
        if (HasReset) { pcomma(); W << "        .reset(reset)"; }
      }
      for (const auto &P : D.Spec->Inputs) {
        if (P.Name == "reset") continue;
        pcomma();
        W << "        ." << P.Name << "(" << D.BlockId << "__" << P.Name << ")";
      }
      for (const auto &P : D.Spec->Outputs) {
        pcomma();
        W << "        ." << P.Name << "(" << D.BlockId << "__" << P.Name << ")";
      }
      W << "\n    );\n\n";
    }
    W << "endmodule\n";
    if (writeStringToFile(OutDir + "/" + WrapperModule + ".sv",
                          W.str()) != 0)
      return 1;
  }

  // Tier-7d follow-up: emit a per-subsystem Python reference for
  // every non-DUT `signal_subsystem` in the entry flow (host
  // helpers). Self-invoke `-emit-python --subsystem <flow>` for
  // each and populate DCO.HostHelpers.
  for (const auto &N : Entry->Nodes) {
    bool IsDut = false;
    for (const auto &D : DutCtxs) if (D.Node == &N) { IsDut = true; break; }
    if (IsDut) continue;
    if (N.Kind != "signal_subsystem") continue;
    const std::string *FId = N.getData("flow_id");
    if (!FId || FId->empty()) {
      std::cerr << "error: -emit-cocotb: host-side signal_subsystem \""
                << N.Id << "\" missing data.flow_id\n";
      return 1;
    }
    const matlab::flowchart::Flow *HFlow = nullptr;
    for (const auto &F : Doc.Flows)
      if (F.Id == *FId) { HFlow = &F; break; }
    if (!HFlow) {
      std::cerr << "error: -emit-cocotb: host-side signal_subsystem \""
                << N.Id << "\" references unknown flow_id \""
                << *FId << "\"\n";
      return 1;
    }
    std::string HelperPy = OutDir + "/" + HFlow->Name + "_ref.py";
    std::string HFlag = "--subsystem " + shellQuote(HFlow->Name);
    if (runMatlabcEmit(Self, "-emit-python " + HFlag, Input,
                       HelperPy) != 0)
      return 1;
    matlab::flowchart::DiagramCocotbOptions::HostHelper H;
    H.BlockId    = N.Id;
    H.ModuleName = HFlow->Name + "_ref";
    H.ClassName  = camel(HFlow->Name);
    for (const auto &P : HFlow->Nodes) {
      if (P.Kind == "signal_inport")  H.InputPorts.push_back(P.Id);
      if (P.Kind == "signal_outport") H.OutputPorts.push_back(P.Id);
    }
    DCO.HostHelpers.push_back(std::move(H));
  }

  auto TestPy =
      matlab::flowchart::emitDiagramCocotbHarness(Doc, Doc.Entry, DCO,
                                                   Diag);
  if (!TestPy) return 1;

  std::string TestPath = OutDir + "/test_" + Stem + ".py";
  if (writeStringToFile(TestPath, *TestPy) != 0) return 1;
  if (writeStringToFile(OutDir + "/cocotb_fi.py",
                         std::string(kCocotbFiHelperPy)) != 0)
    return 1;
  // Makefile points at every DUT's SV (+ wrapper for multi-DUT) and
  // the diagram-level test module.
  std::string MF;
  MF += "# Generated by matlabc -emit-cocotb. Do not edit.\n";
  MF += "TOPLEVEL_LANG ?= verilog\n";
  MF += "SIM           ?= verilator\n";
  MF += "EXTRA_ARGS    += --trace --trace-structs\n";
  // Silence Verilator 5.x's fatal width-style lint (the emitted SV uses
  // Verilog's implicit width extension in mixed-width arithmetic).
  MF += "EXTRA_ARGS    += -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC\n";
  MF += "VERILOG_SOURCES =";
  for (const auto &D : DutCtxs) MF += " $(PWD)/" + D.DutStem + ".sv";
  if (MultiDut) MF += " $(PWD)/" + WrapperModule + ".sv";
  MF += "\n";
  MF += std::string("TOPLEVEL = ") +
        (MultiDut ? WrapperModule : DutCtxs.front().Spec->Name) + "\n";
  MF += "MODULE   = test_" + Stem + "\n";
  MF += "\n";
  MF += "include $(shell cocotb-config --makefiles)/Makefile.sim\n";
  if (writeStringToFile(OutDir + "/Makefile", MF) != 0) return 1;

  // Copy matlab_runtime.py over for the reference module.
  auto findRuntimePy = [&]() -> std::string {
    std::string SelfStr(Self);
    auto last = SelfStr.find_last_of('/');
    std::string Bin = (last == std::string::npos) ? "."
                                                    : SelfStr.substr(0, last);
    char Real[PATH_MAX];
    if (realpath(Bin.c_str(), Real)) Bin = Real;
    std::vector<std::string> Cands = {
      /* Post-2026-05 layout: runtime/shim/. */
      Bin + "/../runtime/shim/matlab_runtime.py",
      Bin + "/runtime/shim/matlab_runtime.py",
      Bin + "/../share/matlabc/runtime/shim/matlab_runtime.py",
      /* Legacy flat layout (fallback for older installs). */
      Bin + "/../runtime/matlab_runtime.py",
      Bin + "/runtime/matlab_runtime.py",
      Bin + "/../share/matlabc/runtime/matlab_runtime.py",
    };
    for (auto &C : Cands) {
      std::ifstream F(C);
      if (F) return C;
    }
    return std::string();
  };
  std::string RuntimePy = findRuntimePy();
  if (RuntimePy.empty()) {
    std::cerr << "warning: -emit-cocotb: couldn't locate matlab_runtime.py "
                 "next to matlabc; the reference model won't import. "
                 "Copy it manually into "
              << OutDir << "/ or set PYTHONPATH to its directory.\n";
  } else {
    std::ifstream Src(RuntimePy);
    std::string Body((std::istreambuf_iterator<char>(Src)),
                     std::istreambuf_iterator<char>());
    if (writeStringToFile(OutDir + "/matlab_runtime.py", Body) != 0)
      return 1;
  }

  std::cerr << "matlabc: wrote whole-diagram cocotb SIL harness to "
            << OutDir << " (";
  for (size_t I = 0; I < DutCtxs.size(); ++I) {
    if (I) std::cerr << ", ";
    std::cerr << "DUT[" << I << "]: " << DutCtxs[I].BlockId << " → "
              << DutCtxs[I].Flow->Name;
  }
  if (MultiDut) std::cerr << ", wrapper=" << WrapperModule;
  if (Opts.CocotbLatency > 0)
    std::cerr << ", latency=" << Opts.CocotbLatency;
  std::cerr << ", tol=" << DCO.Tolerance << ")\n";
  return 0;
}

int emitCocotbHarness(const char *Self, const Options &Opts,
                      const TranslationUnit &TU, TypeContext &TC,
                      DiagnosticEngine &Diag, SourceManager &SM) {
  // Compute stem + output dir.
  std::string Input = Opts.InputPath;
  std::string Stem = Input;
  size_t Slash = Stem.find_last_of('/');
  if (Slash != std::string::npos) Stem = Stem.substr(Slash + 1);
  size_t Dot = Stem.find_last_of('.');
  if (Dot != std::string::npos) Stem = Stem.substr(0, Dot);

  std::string OutDir = Opts.CocotbOutDir;
  if (OutDir.empty()) {
    std::string Parent = ".";
    if (Slash != std::string::npos) Parent = Input.substr(0, Slash);
    OutDir = Parent + "/" + Stem + "_cocotb";
  }
  if (mkdirIfNeeded(OutDir) != 0) return 1;

  // Drive the existing emit modes via a self-invocation. Cleaner than
  // re-running the lowering pipeline twice in-process — each emit
  // pass mutates the Module destructively, so a single shared lower
  // wouldn't survive both.
  std::string SVPath = OutDir + "/" + Stem + ".sv";
  std::string PyPath = OutDir + "/" + Stem + "_ref.py";
  if (runMatlabcEmit(Self, "-emit-systemverilog", Input, SVPath) != 0)
    return 1;
  if (runMatlabcEmit(Self, "-emit-python", Input, PyPath) != 0)
    return 1;

  // Source of truth for port specs: parse the SV port list we just
  // emitted. The SV emitter has already type-refined every signal
  // (input / output, signed / unsigned, width); rebuilding the same
  // info from the lowered MLIR Module here would mean replaying most
  // of the SV pipeline. The port list is canonical and trivial to
  // parse (newline + comma delimited inside the `module(...);` block).
  auto Spec = parseCocotbSpecFromSv(SVPath, Stem);
  if (!Spec) {
    std::cerr << "error: emit-cocotb: failed to parse port list from "
              << SVPath << "\n";
    return 1;
  }

  /* Apply any `% cocotb:` pragmas from the source onto the matching
   * input port. Mismatched names are reported but ignored (stale
   * pragma after a rename). */
  auto Pragmas = scanCocotbPragmas(Input);
  for (auto &P : Spec->Inputs) {
    auto HIt = Pragmas.Holds.find(P.Name);
    if (HIt != Pragmas.Holds.end()) P.HoldCycles = HIt->second;
    auto SIt = Pragmas.Stim.find(P.Name);
    if (SIt != Pragmas.Stim.end()) {
      P.Stim = SIt->second.Kind;
      P.StimArg1 = SIt->second.Arg1;
      P.StimArg2 = SIt->second.Arg2;
    }
  }
  auto warnOrphan = [&](const std::string &Name, const char *Tag) {
    bool Found = false;
    for (auto &P : Spec->Inputs)
      if (P.Name == Name) { Found = true; break; }
    if (!Found)
      std::cerr << "warning: emit-cocotb: `% cocotb: " << Tag << "(" << Name
                << ", ...)` doesn't match any input port; ignored.\n";
  };
  for (auto &Kv : Pragmas.Holds) warnOrphan(Kv.first, "hold");
  for (auto &Kv : Pragmas.Stim)  warnOrphan(Kv.first, "stimulus");
  for (auto &Kv : Pragmas.CoverMinBins) {
    bool Found = false;
    for (auto &P : Spec->Inputs)
      if (P.Name == Kv.first) { Found = true; break; }
    if (!Found)
      for (auto &P : Spec->Outputs)
        if (P.Name == Kv.first) { Found = true; break; }
    if (!Found)
      std::cerr << "warning: emit-cocotb: `% cocotb: cover(" << Kv.first
                << ", ...)` doesn't match any input or output port; "
                   "ignored.\n";
  }

  /* `% cocotb: latency(N)` applies when the user didn't pass
   * `-cocotb-latency=` on the command line — same convention the
   * rest of the pragma surface uses (CLI wins for explicit overrides,
   * pragma fills the per-fixture default). The effective latency is
   * threaded through the rest of this function; `Opts` is const so
   * we keep a local. */
  int EffectiveLatency = Opts.CocotbLatency;
  if (!Opts.CocotbLatencyExplicit && Pragmas.Latency)
    EffectiveLatency = *Pragmas.Latency;

  /* v3.2: when a sibling `test_<stem>.m` exists, replay its
   * stimulus instead of driving random vectors. The extractor is
   * best-effort — if the tester uses a shape outside the recognised
   * patterns (loop with literal-vector indexing, fixed-count loop
   * with iv conditionals, single device call) it returns nullopt
   * and we fall back to random with a diagnostic. */
  std::optional<std::vector<std::vector<std::string>>> Stimulus;
  {
    matlab::SourceManager TesterSM;
    matlab::ASTContext TesterCtx;
    auto *TTU = loadTesterTU(TesterSM, TesterCtx, Input, Stem, Spec->Name);
    if (TTU) {
      TesterStimulus Ext;
      Stimulus = Ext.extract(TTU, Spec->Name);
      if (!Stimulus) {
        std::cerr << "warning: emit-cocotb: found test_" << Stem
                  << ".m but couldn't extract its stimulus shape — "
                     "falling back to random vectors. Recognised shapes "
                     "are documented in docs/emit_cocotb.md (v3.2).\n";
      }
    }
  }

  std::string Harness = renderCocotbHarness(*Spec, Stem, Opts.CocotbVectors,
                                             EffectiveLatency,
                                             Opts.CocotbSeed,
                                             Stimulus ? &*Stimulus : nullptr,
                                             Pragmas.CoverMinBins,
                                             Pragmas.CoverPairsMin,
                                             Pragmas.CoverRange);
  std::string Makefile = renderCocotbMakefile(Stem, Spec->Name);

  if (writeStringToFile(OutDir + "/test_" + Stem + ".py", Harness) != 0)
    return 1;
  if (writeStringToFile(OutDir + "/Makefile", Makefile) != 0)
    return 1;
  if (writeStringToFile(OutDir + "/cocotb_fi.py",
                         std::string(kCocotbFiHelperPy)) != 0)
    return 1;

  /* The Python reference (`<stem>_ref.py`) imports `matlab_runtime`
   * — fi semantics, persistent state, runtime saturate / quantize.
   * Copy that module into the harness dir so the user can `cd` and
   * `make` without depending on the source-tree layout. We resolve
   * the source via the matlabc binary's path: the build dir layout
   * is `<repo>/build/matlabc` with the runtime at `<repo>/runtime/`,
   * so two parent steps from the binary plus `runtime/matlab_runtime.py`
   * is the canonical guess. Falls back to the install-side neighbour
   * when the build-dir guess misses. */
  auto findRuntimePy = [&]() -> std::string {
    std::string SelfStr(Self);
    auto last = SelfStr.find_last_of('/');
    std::string Bin = (last == std::string::npos) ? "." : SelfStr.substr(0, last);
    /* Resolve symlinks / relative components in argv[0] so the
     * `<bin>/../runtime/...` walk lands on the source tree even
     * when the user invoked matlabc via a build symlink. */
    char Real[PATH_MAX];
    if (realpath(Bin.c_str(), Real)) Bin = Real;
    std::vector<std::string> Cands = {
      /* Post-2026-05 layout: runtime/shim/. */
      Bin + "/../runtime/shim/matlab_runtime.py",
      Bin + "/runtime/shim/matlab_runtime.py",
      Bin + "/../share/matlabc/runtime/shim/matlab_runtime.py",
      /* Legacy flat layout (fallback for older installs). */
      Bin + "/../runtime/matlab_runtime.py",
      Bin + "/runtime/matlab_runtime.py",
      Bin + "/../share/matlabc/runtime/matlab_runtime.py",
    };
    for (auto &C : Cands) {
      std::ifstream F(C);
      if (F) return C;
    }
    return std::string();
  };
  std::string RuntimePy = findRuntimePy();
  if (RuntimePy.empty()) {
    std::cerr << "warning: emit-cocotb: couldn't locate matlab_runtime.py "
                 "next to matlabc; the reference model won't import. "
                 "Copy it manually into "
              << OutDir << "/ or set PYTHONPATH to its directory.\n";
  } else {
    std::ifstream Src(RuntimePy);
    std::string Body((std::istreambuf_iterator<char>(Src)),
                     std::istreambuf_iterator<char>());
    if (writeStringToFile(OutDir + "/matlab_runtime.py", Body) != 0)
      return 1;
  }

  std::cerr << "matlabc: wrote CocoTB harness to " << OutDir
            << " (" << Spec->Inputs.size() << " inputs, "
            << Spec->Outputs.size() << " outputs, "
            << (Spec->Sequential ? "sequential" : "combinational");
  if (Stimulus)
    std::cerr << ", " << Stimulus->size() << " vectors from test_"
              << Stem << ".m";
  else
    std::cerr << ", " << Opts.CocotbVectors << " random vectors";
  if (EffectiveLatency > 0)
    std::cerr << ", latency=" << EffectiveLatency;
  std::cerr << ")\n";

  /* v3.4: auto-detect a pipeline-depth hint by counting `persistent`
   * declarations in the source. The full register-chain depth is
   * harder to reason about (some DUTs have non-pipelined parallel
   * persistents), so this is informational only — printed when
   * neither the CLI flag nor a pragma supplied a latency and the
   * count suggests a pipelined design. The user still picks the
   * right L; this just nudges them toward "non-zero L is
   * probably needed". */
  if (Spec->Sequential && EffectiveLatency == 0) {
    std::ifstream Sf(Input);
    std::string Body((std::istreambuf_iterator<char>(Sf)),
                      std::istreambuf_iterator<char>());
    int PersistCount = 0;
    size_t P = 0;
    while ((P = Body.find("persistent", P)) != std::string::npos) {
      bool At = (P == 0 || !std::isalnum((unsigned char)Body[P-1]));
      bool Brk = (P + 10 >= Body.size() ||
                  !std::isalnum((unsigned char)Body[P+10]));
      if (At && Brk) ++PersistCount;
      P += 10;
    }
    /* B4 — precise auto-latency. Two complementary hints:
     *
     * 1. Scalar-persistent chain. N independent persistents that
     *    feed each other in source order produce a visible delay
     *    of N - 1 cycles (MATLAB blocking semantics: the body's
     *    `stage1 = in; stage2 = stage1` reads stage1's same-cycle
     *    written value, so the SV's two-flop chain shows up as a
     *    one-cycle delay against the Python ref).
     *
     * 2. fi-array shift register. `fi(zeros(1, N), ...)` declares
     *    an N-element shift register; Stage F splits it into N
     *    parallel scalar persistents, and the natural pipeline
     *    depth from input to the last tap is N. Scan source for
     *    `zeros(1, N)` (or `zeros(1,N)`) literals and pick the
     *    largest N as the shift-register depth.
     *
     * The hint reports the larger of the two estimates — fixtures
     * with both shapes (sync_2ff is shape 1, sequential_processor
     * is shape 2, fir_asic_pipelined is shape 2 with reinforcing
     * shape 1) end up with the right L. */
    int ZerosDepth = 0;
    {
      size_t Q = 0;
      while ((Q = Body.find("zeros(", Q)) != std::string::npos) {
        Q += 6;  // skip "zeros("
        // Look for `1,` (with optional whitespace) and pick out N.
        size_t Cur = Q;
        while (Cur < Body.size() && std::isspace((unsigned char)Body[Cur]))
          ++Cur;
        if (Cur < Body.size() && Body[Cur] == '1') {
          ++Cur;
          while (Cur < Body.size() && std::isspace((unsigned char)Body[Cur]))
            ++Cur;
          if (Cur < Body.size() && Body[Cur] == ',') {
            ++Cur;
            while (Cur < Body.size() &&
                   std::isspace((unsigned char)Body[Cur]))
              ++Cur;
            int N = 0;
            while (Cur < Body.size() &&
                   std::isdigit((unsigned char)Body[Cur])) {
              N = N * 10 + (Body[Cur] - '0');
              ++Cur;
            }
            if (N > 1 && N > ZerosDepth) ZerosDepth = N;
          }
        }
      }
    }
    int ChainEstimate = (PersistCount >= 2) ? (PersistCount - 1) : 0;
    int Suggested = std::max(ChainEstimate, ZerosDepth);
    if (Suggested >= 1) {
      std::cerr << "       hint: ";
      if (ZerosDepth > 0)
        std::cerr << ZerosDepth << "-tap fi-array shift register"
                  << (PersistCount >= 2
                      ? std::string(" (+ ") + std::to_string(PersistCount)
                          + " scalar persistents)"
                      : std::string())
                  << " — pipelined";
      else
        std::cerr << PersistCount << " `persistent` decls — pipelined";
      std::cerr << "; if outputs are registered, add `% cocotb: latency("
                << Suggested << ")` near the `% hdl: port(...)` "
                   "lines, or pass `-cocotb-latency=" << Suggested
                << "` on the CLI.\n";
    }
  }
  return 0;
}
#endif // MATLAB_LLVM_WITH_MLIR

} // namespace

int main(int Argc, char **Argv) {
  Options Opts;
  const char *Prog = Argv[0];
  if (!parseArgs(Argc, Argv, Opts, Prog)) return usage(Prog);

  /* Capture the matlabc binary directory so `buildReplPrelude` (and any
   * other path-relative helper) can locate `runtime/*.m` without
   * threading argv[0] through their signatures.  realpath() resolves
   * symlinks so a `/usr/local/bin/matlabc` symlink to the build tree
   * still points back at the source's `runtime/` directory. */
  {
    std::string SelfStr(Argv[0]);
    auto last = SelfStr.find_last_of('/');
    std::string Bin = (last == std::string::npos)
                          ? std::string(".") : SelfStr.substr(0, last);
    char Real[PATH_MAX];
    if (realpath(Bin.c_str(), Real)) Bin = Real;
    g_MatlabcBinDir = Bin;
  }

#if MATLAB_LLVM_WITH_MLIR
  // §17.5 #8 — register the MLIR-JIT factory for signal_matlab_fcn
  // blocks. This is what makes `-simulate` route a block's
  // `params.function_body` through the full lex/parse/lower/JIT
  // pipeline instead of the scalar AST interpreter. Idempotent.
  matlab::flowchart::installMflowLinkJit();
  if (Opts.Mode == Options::Mode::Repl) return runRepl();
  if (Opts.Mode == Options::Mode::Dap) {
    int Rc = dap::runDap(Opts.InputPath);
    /* #77: a one-shot `-dap` session leaves an ORC-JIT compile thread
     * that may still be lazily materializing a referenced-but-uncalled
     * symbol in the background. Returning here runs the C++ static
     * destructors, which tear down LLVM's global state (MCContext, the
     * ExecutionSession) out from under that worker thread — it then
     * SIGSEGVs in AsmPrinter/MCContext::getOrCreateSymbol at process
     * exit (every program does this, e.g. fibonacci.m; the `terminated`
     * event has already been delivered so the debug session itself is
     * fine, but the process exits with signal 11 and races launch
     * tests). The session is fully complete by now — all DAP frames and
     * program output have been written — so flush and hard-exit,
     * skipping the racy static teardown entirely. */
    std::fflush(stdout);
    std::fflush(stderr);
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(Rc);
  }
#else
  if (Opts.Mode == Options::Mode::Repl ||
      Opts.Mode == Options::Mode::Dap) {
    std::cerr << "error: matlabc was built without MLIR support; "
                 "REPL / DAP are unavailable\n";
    return 1;
  }
#endif

  if (Opts.Mode == Options::Mode::DumpFlow) {
    SourceManager FlowSM;
    DiagnosticEngine FlowDiag(FlowSM);
    auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Opts.InputPath,
                                                    FlowDiag);
    if (Doc) matlab::flowchart::dumpFlowDoc(std::cout, *Doc);
    FlowDiag.printAll();
    return FlowDiag.hasErrors() ? 1 : 0;
  }

  /* mStateflow Tier 0/4a — `matlabc -dump-chart chart.mflow` loads a
   * state-chart `.mflow`, lowers it to the resolved chart IR, and
   * prints a stable text dump for golden-file tests. Errors out when
   * the input is not a state-chart document. */
  if (Opts.Mode == Options::Mode::DumpChart) {
    SourceManager FlowSM;
    DiagnosticEngine FlowDiag(FlowSM);
    auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Opts.InputPath,
                                                    FlowDiag);
    if (Doc) {
      if (!Doc->isStateChart()) {
        std::cerr << Opts.InputPath
                  << ": -dump-chart requires a state-chart .mflow "
                     "(got settings.kind=\""
                  << Doc->Settings.Kind << "\")\n";
        return 1;
      }
      auto Model = matlab::statechart::buildChartModel(*Doc, FlowDiag);
      if (Model) matlab::statechart::dumpChartModel(std::cout, *Model);
    }
    FlowDiag.printAll();
    return FlowDiag.hasErrors() ? 1 : 0;
  }

  /* mStateflow Tier-N polish — `matlabc -emit-trace chart.mflow`
   * drives the chart through one initialise + N super-steps via the
   * C++ interpreter and prints each ChartTraceEvent as a one-line
   * JSON record to stdout. Useful for offline trace analysis
   * pipelines + golden-file regression of the chart's deterministic
   * step sequence (DAP path is the live-interactive surface; this
   * is its "one-shot dump" sibling). The number of post-init steps
   * defaults to 5 to match the -emit-matlab demo driver; users can
   * pipe stdout into jq for filtering. */
  if (Opts.Mode == Options::Mode::EmitTrace) {
    SourceManager FlowSM;
    DiagnosticEngine FlowDiag(FlowSM);
    auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Opts.InputPath,
                                                    FlowDiag);
    if (!Doc) { FlowDiag.printAll(); return 1; }
    if (!Doc->isStateChart()) {
      std::cerr << Opts.InputPath
                << ": -emit-trace requires a state-chart .mflow "
                   "(got settings.kind=\""
                << Doc->Settings.Kind << "\")\n";
      return 1;
    }
    auto Model = matlab::statechart::buildChartModel(*Doc, FlowDiag);
    FlowDiag.printAll();
    if (!Model || FlowDiag.hasErrors()) return 1;
    if (Model->Charts.empty()) {
      std::cerr << Opts.InputPath
                << ": -emit-trace: chart model has no charts\n";
      return 1;
    }
    matlab::statechart::ChartInterpreter Interp(Model->Charts.front());
    auto jsonEscape = [](const std::string &S) {
      std::string Out;
      Out.reserve(S.size() + 2);
      for (char C : S) {
        switch (C) {
          case '"':  Out += "\\\""; break;
          case '\\': Out += "\\\\"; break;
          case '\n': Out += "\\n";  break;
          case '\r': Out += "\\r";  break;
          case '\t': Out += "\\t";  break;
          default:
            if ((unsigned char)C < 0x20)
              Out += "?";  // strip control chars
            else
              Out += C;
        }
      }
      return Out;
    };
    auto kindName = [](matlab::statechart::ChartTraceEvent::Kind K) {
      using K_ = matlab::statechart::ChartTraceEvent::Kind;
      switch (K) {
        case K_::SuperStepBegin:    return "superStepBegin";
        case K_::SuperStepEnd:      return "superStepEnd";
        case K_::StateEnter:        return "stateEnter";
        case K_::StateExit:         return "stateExit";
        case K_::TransitionFired:   return "transitionFired";
        case K_::EventBroadcast:    return "eventBroadcast";
        case K_::Breakpoint:        return "breakpoint";
        case K_::MaxIterations:     return "maxIterations";
      }
      return "unknown";
    };
    auto emitEvents = [&](const std::vector<
                              matlab::statechart::ChartTraceEvent> &Evs) {
      for (auto &E : Evs) {
        std::cout << "{\"kind\":\"" << kindName(E.K) << "\"";
        if (!E.Id.empty())
          std::cout << ",\"id\":\"" << jsonEscape(E.Id) << "\"";
        if (!E.Src.empty())
          std::cout << ",\"src\":\"" << jsonEscape(E.Src) << "\"";
        if (!E.Dst.empty())
          std::cout << ",\"dst\":\"" << jsonEscape(E.Dst) << "\"";
        if (!E.EventName.empty())
          std::cout << ",\"eventName\":\""
                    << jsonEscape(E.EventName) << "\"";
        if (E.K == matlab::statechart::ChartTraceEvent::Kind::SuperStepBegin
            || E.K == matlab::statechart::ChartTraceEvent::Kind::SuperStepEnd
            || E.K == matlab::statechart::ChartTraceEvent::Kind::MaxIterations)
          std::cout << ",\"iteration\":" << E.Iteration;
        if (E.K == matlab::statechart::ChartTraceEvent::Kind::SuperStepEnd)
          std::cout << ",\"quiescent\":"
                    << (E.Quiescent ? "true" : "false");
        if (!E.BreakpointReason.empty())
          std::cout << ",\"reason\":\""
                    << jsonEscape(E.BreakpointReason) << "\"";
        std::cout << "}\n";
      }
    };
    // Initialise + 5 super-steps. Subsequent super-steps run with
    // the default driver (no events, no local injections), so the
    // trace is deterministic across machines. Drivers that want to
    // inject inputs should use the DAP path.
    emitEvents(Interp.superStep());
    for (int I = 0; I < 4; ++I) emitEvents(Interp.superStep());
    return 0;
  }

  /* mflowLink Tier G — `matlabc -emit-mflowlink-cpp model.mflow` emits
   * a self-contained C++ file to stdout: the original .mflow JSON
   * embedded as a raw string literal plus a small `main()` that
   * builds the IR through the existing `loadMflow` / `lowerSignalFlow`
   * machinery, runs `MflowLinkSim::runToCompletion`, and writes the
   * logged-signal CSV. The user compiles it against the matlab_llvm
   * Flowchart static libs to produce a deployable simulator that does
   * not need the original .mflow at runtime. See
   * `docs/mflow_link_roadmap.md` §9 and `runtime/build_mflowlink.sh`. */
  if (Opts.Mode == Options::Mode::EmitMflowLinkCpp) {
    // Read the raw .mflow bytes — we'll embed them verbatim, so the
    // generated binary loads through the same parser path as
    // `matlabc -simulate`, with zero risk of round-trip drift.
    std::ifstream In(Opts.InputPath, std::ios::binary);
    if (!In) {
      std::cerr << Opts.InputPath << ": cannot open .mflow file\n";
      return 1;
    }
    std::stringstream Buf;
    Buf << In.rdbuf();
    std::string Bytes = Buf.str();

    // Pre-validate by parsing + lowering. A user who runs this on a
    // broken model wants the error here, not in the generated binary
    // an hour later in CI.
    {
      SourceManager FlowSM;
      DiagnosticEngine FlowDiag(FlowSM);
      auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Opts.InputPath,
                                                      FlowDiag);
      if (!Doc) { FlowDiag.printAll(); return 1; }
      if (!Doc->isSignalFlow()) {
        std::cerr << Opts.InputPath
                  << ": -emit-mflowlink-cpp requires a signal-flow .mflow "
                     "(settings.kind == \"signal_flow\")\n";
        return 1;
      }
      auto Model = matlab::flowchart::lowerSignalFlow(*Doc, FlowDiag);
      FlowDiag.printAll();
      if (!Model) return 1;
    }

    // Pick a raw-string delimiter that doesn't appear in the file.
    // Conventional `)json"` survives almost every .mflow; if some
    // user pastes that literal sequence into a `params.title` we
    // walk up to a unique tag.
    std::string Tag = "json";
    while (Bytes.find(std::string(")") + Tag + "\"") != std::string::npos)
      Tag += "_";

    std::cout <<
        "// Auto-generated by `matlabc -emit-mflowlink-cpp`. Do not edit\n"
        "// by hand — regenerate from the source .mflow instead.\n"
        "//\n"
        "// Compile:\n"
        "//   clang++ -std=c++17 -I <matlab_llvm>/include this.cpp \\\n"
        "//       -L <matlab_llvm>/build -lMatlabFlowchart -lMatlabBasic\n"
        "//\n"
        "// Run: ./a.out > out.csv\n"
        "//\n"
        "// Or use `runtime/build_mflowlink.sh <this.cpp>` from the\n"
        "// matlab_llvm checkout to build + run in one step.\n"
        "\n"
        "#include \"matlab/Basic/Diagnostic.h\"\n"
        "#include \"matlab/Basic/SourceManager.h\"\n"
        "#include \"matlab/Flowchart/Loader.h\"\n"
        "#include \"matlab/Flowchart/MflowLinkModel.h\"\n"
        "#include \"matlab/Flowchart/MflowLinkSim.h\"\n"
        "\n"
        "#include <iostream>\n"
        "#include <string>\n"
        "\n"
        "static const char *MFLOWLINK_MODEL_JSON = R\"" << Tag << "(\n";
    std::cout << Bytes;
    std::cout <<
        ")" << Tag << "\";\n"
        "\n"
        "int main(int argc, char **argv) {\n"
        "  (void)argc; (void)argv;\n"
        "  matlab::SourceManager SM;\n"
        "  matlab::FileID F = SM.addBuffer(\""
        << Opts.InputPath << "\", MFLOWLINK_MODEL_JSON);\n"
        "  matlab::DiagnosticEngine Diag(SM);\n"
        "  auto Doc = matlab::flowchart::loadMflow(SM, F, Diag);\n"
        "  if (!Doc) { Diag.printAll(); return 1; }\n"
        "  auto Model = matlab::flowchart::lowerSignalFlow(*Doc, Diag);\n"
        "  Diag.printAll();\n"
        "  if (!Model) return 1;\n"
        "  matlab::flowchart::MflowLinkSim Sim(*Model);\n"
        "  Sim.runToCompletion();\n"
        "  Sim.writeCsv(std::cout);\n"
        "  return 0;\n"
        "}\n";
    return 0;
  }

  /* mflowLink — `matlabc -simulate model.mflow`. Lowers a signal-flow
   * `.mflow` to the MflowLinkModel IR (lib/Flowchart/SignalFlowLowering.cpp).
   * With `--dry-run` it prints the sorted execution order and exits —
   * the Tier-B smoke lane. The interactive runtime (DAP-server mode,
   * step / step-back) is Tier C/D of docs/mflow_link_roadmap.md. */
  if (Opts.Mode == Options::Mode::Simulate) {
    SourceManager FlowSM;
    DiagnosticEngine FlowDiag(FlowSM);
    auto Doc = matlab::flowchart::loadMflowFromPath(FlowSM, Opts.InputPath,
                                                    FlowDiag);
    if (!Doc) {
      FlowDiag.printAll();
      return 1;
    }
    if (Doc->isStateChart()) {
      /* mStateflow Tier 4d — `-simulate` on a state-chart .mflow
       * lowers to MATLAB and prints a smoke trace: N super-step
       * iterations starting from the initial active configuration,
       * one line per tick showing the active region-vector. Real
       * driver scripts (REPL / IDE) get their interactivity through
       * the `<chart>_tick` function the lowering emits — this lane
       * just confirms the chart simulates end-to-end.  --dry-run
       * short-circuits to the chart IR dump (same as `-dump-chart`)
       * so the existing CTest goldens cover this path too. */
#if MATLAB_LLVM_WITH_MLIR
      // Tier 4e — `-simulate --sim-dap` on a chart .mflow boots the
      // chart-namespaced DAP server.
      if (Opts.SimulateDap) {
        return dap::runStateChartDap(Opts.InputPath);
      }
#else
      if (Opts.SimulateDap) {
        std::cerr << "matlabc was built without MLIR support; "
                     "-simulate --sim-dap is unavailable\n";
        return 1;
      }
#endif
      auto Model = matlab::statechart::buildChartModel(*Doc, FlowDiag);
      FlowDiag.printAll();
      if (!Model) return 1;
      if (Opts.DryRun) {
        matlab::statechart::dumpChartModel(std::cout, *Model);
        return 0;
      }
      const matlab::statechart::Chart *Entry = Model->entryChart();
      if (!Entry && !Model->Charts.empty()) Entry = &Model->Charts.front();
      if (!Entry) {
        std::cerr << Opts.InputPath
                  << ": state-chart document has no charts\n";
        return 1;
      }
      auto Lowered = matlab::statechart::lowerChartToMatlab(*Entry, FlowDiag);
      FlowDiag.printAll();
      if (!Lowered) return 1;
      // Drive the C++ chart interpreter and print a deterministic
      // trace. By default we run the initial entry chain plus N
      // super-steps with no external events; the chart settles into
      // its quiescent post-entry configuration and we dump the
      // active-state vector. Real driver scripts get richer control
      // through the chart_tick MATLAB lowering or the chart DAP
      // server (`-simulate --sim-dap`).
      matlab::statechart::ChartInterpreter Interp(*Entry);
      std::cout << "% mStateflow -simulate trace for chart \""
                << Entry->Name << "\"\n";
      auto printEvent =
          [&](const matlab::statechart::ChartTraceEvent &E) {
            using K = matlab::statechart::ChartTraceEvent::Kind;
            switch (E.K) {
            case K::SuperStepBegin:
              std::cout << "[superStepBegin iter=" << E.Iteration << "]\n";
              break;
            case K::SuperStepEnd:
              std::cout << "[superStepEnd iter=" << E.Iteration
                        << " quiescent=" << (E.Quiescent ? "true" : "false")
                        << "]\n";
              break;
            case K::StateEnter:
              std::cout << "+enter " << E.Id << "\n";
              break;
            case K::StateExit:
              std::cout << "-exit  " << E.Id << "\n";
              break;
            case K::TransitionFired:
              std::cout << ">fire  " << E.Id << " (" << E.Src << " -> "
                        << E.Dst;
              if (!E.EventName.empty()) std::cout << " on " << E.EventName;
              std::cout << ")\n";
              break;
            case K::EventBroadcast:
              std::cout << "*event " << E.Id << "\n";
              break;
            case K::Breakpoint:
              std::cout << "!break " << E.BreakpointReason << " " << E.Id
                        << "\n";
              break;
            case K::MaxIterations:
              std::cout << "!warn  super-step did not converge within "
                        << E.Iteration << " iterations\n";
              break;
            }
          };
      auto Init = Interp.initialize();
      for (auto &E : Init) printEvent(E);
      // Run one no-event super-step to exercise during actions /
      // self-firing transitions. Charts that need external events
      // sit quiescent here, which is the right CLI behaviour.
      auto Step = Interp.superStep();
      for (auto &E : Step) printEvent(E);
      // Final active-state vector for at-a-glance verification.
      std::cout << "active@end:";
      for (auto &Id : Interp.activeStates()) std::cout << " " << Id;
      std::cout << "\n";
      return 0;
    }
    if (!Doc->isSignalFlow()) {
      std::cerr << Opts.InputPath
                << ": -simulate requires a signal-flow .mflow "
                   "(settings.kind == \"signal_flow\")\n";
      return 1;
    }
    auto Model = matlab::flowchart::lowerSignalFlow(*Doc, FlowDiag);
    FlowDiag.printAll();
    if (!Model) return 1;
    if (Opts.DryRun) {
      matlab::flowchart::dumpMflowLinkModel(std::cout, *Model);
      return 0;
    }
#if MATLAB_LLVM_WITH_MLIR
    // Tier-D DAP-server lane — pauses at entry, accepts the §10 verb
    // set, drives the same MflowLinkSim instance with snapshot-ring
    // step-back. Reuses the JSON-RPC framing the matlab-program DAP
    // server already established (`OriginalStdoutFd`, `readFrame`,
    // `sendResponse`, `sendEvent`).
    if (Opts.SimulateDap) {
      // The DAP code lives behind MATLAB_LLVM_WITH_MLIR like the rest
      // of the dap:: namespace; we don't need MLIR for the simulator
      // itself, but the framing helpers are gated this way today.
      return dap::runMflowLinkDap(Opts.InputPath);
    }
#else
    if (Opts.SimulateDap) {
      std::cerr << "matlabc was built without MLIR support; "
                   "-simulate --sim-dap is unavailable\n";
      return 1;
    }
#endif
    // Tier-C interpreter (lib/Flowchart/MflowLinkSim.cpp). Runs from
    // solver.startTime to solver.stopTime and streams the logged
    // signals as CSV on stdout.
    matlab::flowchart::MflowLinkSim Sim(*Model);
    Sim.runToCompletion();
    Sim.writeCsv(std::cout);
    return 0;
  }

  /* CST stdlib prelude: classdef definitions for `tf` / `ss` / `zpk`
   * / `pid` / `frd` model objects and any other "intrinsic" classes
   * the toolbox surface relies on. Located the same way as the
   * cocotb runtime: walk up from argv[0] to find `runtime/`. Empty
   * string when not found — silently skipped so non-CST tests keep
   * working. The prelude is also skipped when the user input doesn't
   * mention any of the class names it provides: an unused classdef
   * compiles down to a func.func body whose `none`-typed slots no
   * downstream pass can resolve, leaving stale `matlab.call_builtin`
   * ops that fail LLVM-IR translation. The textual scan is a
   * cheap whole-word check against the source — false positives
   * (e.g. a user comment `% tf is short for transfer function`)
   * just pay the parse cost, not a correctness bug. */
  auto findCstPrelude = [&]() -> std::string {
    std::string SelfStr(Argv[0]);
    auto last = SelfStr.find_last_of('/');
    std::string Bin = (last == std::string::npos) ? "." : SelfStr.substr(0, last);
    char Real[PATH_MAX];
    if (realpath(Bin.c_str(), Real)) Bin = Real;
    std::vector<std::string> Cands = {
      Bin + "/../runtime/cst_classdefs.m",
      Bin + "/runtime/cst_classdefs.m",
      Bin + "/../share/matlabc/runtime/cst_classdefs.m",
    };
    for (auto &C : Cands) {
      std::ifstream Fp(C);
      if (Fp) return C;
    }
    return std::string();
  };
  /* Per-class detection: scan the user input for whole-word `<name>(`
   * or `<name> =` (single-`=` assignment) patterns. Returns a vector
   * of matched class names. Comments are stripped first so a `% tf
   * is short for transfer function` line doesn't pull the prelude in. */
  auto userMentionsCstClasses =
      [](const std::string &Path) -> std::vector<std::string> {
    std::vector<std::string> Found;
    std::ifstream In(Path);
    if (!In) return Found;
    std::ostringstream Buf;
    Buf << In.rdbuf();
    std::string SrcRaw = Buf.str();
    std::string Src;
    Src.reserve(SrcRaw.size());
    bool InComment = false;
    for (char c : SrcRaw) {
      if (c == '\n') {
        InComment = false;
        Src.push_back(c);
        continue;
      }
      if (c == '%') InComment = true;
      if (!InComment) Src.push_back(c);
    }
    static const char *Names[] = { "tf", "ss", "zpk", "pid", "frd" };
    for (const char *N : Names) {
      size_t NL = std::strlen(N);
      size_t P = 0;
      bool Hit = false;
      while ((P = Src.find(N, P)) != std::string::npos && !Hit) {
        bool LeftWord = (P > 0) && (std::isalnum((unsigned char)Src[P-1]) ||
                                     Src[P-1] == '_');
        if (!LeftWord && P + NL < Src.size()) {
          char Right = Src[P + NL];
          if (Right == '(') { Hit = true; break; }
          size_t Q = P + NL;
          while (Q < Src.size() && (Src[Q] == ' ' || Src[Q] == '\t')) Q++;
          if (Q < Src.size() && Src[Q] == '=') {
            if (Q + 1 >= Src.size() || Src[Q+1] != '=') { Hit = true; break; }
          }
        }
        P += NL;
      }
      if (Hit) Found.push_back(N);
    }
    return Found;
  };
  /* The per-class prelude file lookup: tf lives in the umbrella
   * `cst_classdefs.m` (it shares the `cst_polyadd` / `cst_polysub`
   * helpers); the other classes have their own `cst_class_<name>.m`
   * files so the unused-classdef bodies (with `none`-typed slots
   * that no downstream pass refines) don't get pulled in for
   * tf-only programs. */
  auto findClassPrelude = [&](const std::string &ClsName) -> std::string {
    std::string SelfStr(Argv[0]);
    auto last = SelfStr.find_last_of('/');
    std::string Bin = (last == std::string::npos) ? "." : SelfStr.substr(0, last);
    char Real[PATH_MAX];
    if (realpath(Bin.c_str(), Real)) Bin = Real;
    std::string Leaf = (ClsName == "tf") ? "cst_classdefs.m"
                                          : ("cst_class_" + ClsName + ".m");
    /* CST classdefs live in runtime/toolbox/control/ post-2026-05. */
    std::vector<std::string> Cands = {
      Bin + "/../runtime/toolbox/control/" + Leaf,
      Bin + "/runtime/toolbox/control/" + Leaf,
      Bin + "/../share/matlabc/runtime/toolbox/control/" + Leaf,
      /* Legacy flat-layout fallback. */
      Bin + "/../runtime/" + Leaf,
      Bin + "/runtime/" + Leaf,
      Bin + "/../share/matlabc/runtime/" + Leaf,
    };
    for (auto &C : Cands) {
      std::ifstream Fp(C);
      if (Fp) return C;
    }
    return std::string();
  };
  std::vector<std::string> PreludePaths;
  for (const std::string &Cls : userMentionsCstClasses(Opts.InputPath)) {
    std::string P = findClassPrelude(Cls);
    if (!P.empty()) PreludePaths.push_back(std::move(P));
  }
  /* Communications Toolbox System Object preludes — per-class files
   * mirroring the CST pattern.  Each class lives in its own
   * `comm_class_<name>.m` file so an uncalled class's method bodies
   * (whose `none`-typed params Sema can't refine without a call
   * site) don't get pulled into the TU. */
  /* Comm + Antenna SO catalog scan.  Same shape as the CST scan
   * above — whole-word / call-site match for each registered class
   * name; per-class prelude file lookup via the explicit table
   * below. */
  auto userMentionsExtClasses =
      [](const std::string &Path) -> std::vector<std::string> {
    std::vector<std::string> Found;
    std::ifstream In(Path);
    if (!In) return Found;
    std::ostringstream Buf;
    Buf << In.rdbuf();
    std::string SrcRaw = Buf.str();
    std::string Src;
    Src.reserve(SrcRaw.size());
    bool InComment = false;
    for (char c : SrcRaw) {
      if (c == '\n') {
        InComment = false;
        Src.push_back(c);
        continue;
      }
      if (c == '%') InComment = true;
      if (!InComment) Src.push_back(c);
    }
    static const char *Names[] = {
      /* Comm SO surface. */
      "CommCRCGenerator", "CommCRCDetector",
      /* Antenna catalog (ANT-Tier-1). */
      "AntDipole", "AntMonopole",
      /* RF catalog (RF-Tier-1). */
      "RFSparameters",
      /* RF sibling network-parameter classdefs. */
      "RFYparameters", "RFZparameters", "RFHparameters",
      "RFGparameters", "RFAbcdparameters", "RFTparameters",
      /* RF circuit hierarchy. */
      "RFCktAmplifier", "RFCktMixer", "RFCktPassive",
      "RFCktCascade", "RFCktParallel", "RFCktSeries", "RFCktShunt",
      "RFRational",
      /* RF Propagation site descriptors. */
      "TxSite", "RxSite", "PropagationModel",
      /* PDE Toolbox classdef façade — see runtime/pde_classdefs.m.
       * Any one detection pulls the umbrella file in, which contains
       * all of femodel + materialProperties + faceBC/edgeBC/vertexBC
       * + faceLoad/edgeLoad/vertexLoad/cellLoad + result classes
       * (StaticStructuralResults / StationaryResults /
       * pdeDisplacement) + the solve/generateMesh dispatch + setter
       * helpers. */
      "femodel", "materialProperties",
      "faceBC", "edgeBC", "vertexBC",
      "faceLoad", "edgeLoad", "vertexLoad", "cellLoad",
      "StaticStructuralResults", "StationaryResults",
      "ThermalResults", "ElectrostaticResults",
      "MagneticResults", "DCConductionResults",
      "TransientStructuralResults", "ModalStructuralResults",
      "FrequencyStructuralResults", "HarmonicEMResults",
      /* Optimization Toolbox problem-based API — umbrella file
       * `optim_classdefs.m` (OptimizationExpression / Optimization
       * Problem / EquationProblem classdefs + the optimvar /
       * optimproblem / eqnproblem factories). */
      "optimvar", "optimintvar", "optimproblem", "eqnproblem",
      "OptimizationExpression", "OptimizationProblem", "EquationProblem",
      /* MPC Toolbox Tier-1/2/3/4 — umbrella `mpc_classdefs.m` (mpc +
       * mpcstate + mpcmoveopt + explicitMPC classdefs + factories).
       * Mentions of any pull in the file. */
      "mpc", "mpcstate", "mpcmove", "mpcmoveopt",
      "explicitMPC", "generateExplicitMPC", "mpcmoveExplicit",
      "mpcActiveSetSolver", "mpcmoveFinite",
      "nlmpc", "nlmpcmove",
      "mpcsimopt", "setEstimator", "getEstimator", "review",
      /* System Identification Toolbox Tier-1 — umbrella
       * `ident_classdefs.m` (iddata + idpoly).  Any estimator /
       * container / method mention pulls the file in. */
      "iddata", "idpoly", "idss", "idgrey", "n4sid", "ssest", "tfest", "greyest",
      "impulseest", "forecast", "idfrd", "etfe", "spa",
      "extendedKalmanFilter", "unscentedKalmanFilter", "correct",
      "recursiveLS", "recursiveARX", "idnlgrey", "nlgreyest",
      "arxOptions", "getcov", "getpvec", "setpvec",
      "MultiStart", "GlobalSearch", "createOptimProblem", "optimoptions",
      "makedist", "fitdist", "ProbDistUnivParam", "fitlm", "fitglm", "LinearModel",
      "fitcknn", "fitcnb", "fitcdiscr", "fitctree", "fitcsvm", "fitcecoc", "ClassificationModel",
      "fitcensemble", "TreeBagger",
      "affine2d", "projective2d", "imref2d", "fitgeotform2d",
      /* Curve Fitting Toolbox Tier-1 — `curvefit_classdefs.m` umbrella. */
      "fit", "cfit", "sfit", "fittype", "fitoptions", "coeffvalues",
      "ppform", "spline", "pchip", "ppmak", "fnder", "fnint",
      /* Bioinformatics Toolbox Tier-4/6 — `bioinfo_classdefs.m`. */
      "phytree", "seqlinkage", "seqneighjoin", "DataMatrix",
      /* DSP System Toolbox — `dsp_classdefs.m` umbrella.  The parser folds
       * `dsp.Foo` -> `dsp_Foo`; the source-text scan keys on the dotted
       * package form the user actually wrote. */
      "dsp.FIRFilter", "dsp.IIRFilter", "dsp.BiquadFilter",
      "dsp.SOSFilter", "dsp.Delay", "dsp.LMSFilter", "dsp.RLSFilter",
      "dsp.FIRDecimator", "dsp.FIRInterpolator", "dsp.CICDecimator",
      "dsp.CICInterpolator", "dsp.SampleRateConverter",
      "dsp.Channelizer", "dsp.ChannelSynthesizer",
      /* DSP Tier-5 — sources / stats / detectors / spectral / buffering. */
      "dsp.SineWave", "dsp.NCO", "dsp.Chirp",
      "dsp.MovingAverage", "dsp.MovingRMS", "dsp.MovingMaximum",
      "dsp.MovingMinimum", "dsp.MovingStandardDeviation",
      "dsp.PeakFinder", "dsp.DCBlocker",
      "dsp.ZeroCrossingDetector", "dsp.SpectrumEstimator", "dsp.AsyncBuffer",
      /* DSP Tier-6 — linalg + polish filter SOs. */
      "dsp.LevinsonSolver", "dsp.NotchPeakFilter",
      "dsp.LowpassFilter", "dsp.HighpassFilter",
      /* DSP HDL Tier-7/8 simulation surface — `dsphdl_classdefs.m`. */
      "dsphdl.FIRFilter", "dsphdl.BiquadFilter",
      "dsphdl.SineWave", "dsphdl.NCO",
      "dsphdl.FIRDecimator", "dsphdl.CICDecimator",
      "arx", "ar", "armax", "oe", "bj",
      "iv4", "delayest", "compare", "predict", "resid", "goodnessOfFit",
      /* GPU Coder host-side carriers — see gpu_classdefs.m. */
      "gpuArray", "gather", "existsOnGPU", "gpuDevice",
      "coder.gpuConfig", "coder_gpuConfig",
      /* Financial Toolbox Tier-3 — Portfolio classdef. */
      "Portfolio", "setAssetMoments", "setBounds", "setBudget",
      "setDefaultConstraints", "estimateFrontier", "estimatePortMoments",
      "estimateMaxSharpeRatio", "estimateAssetMoments",
      "estimateFrontierByReturn", "estimateFrontierByRisk",
      "estimatePortReturn", "estimatePortRisk",
      /* Financial Toolbox Tier-4 — credit scorecard classdef. */
      "creditscorecard", "fitmodel", "probdefault",
      /* Financial Toolbox Tier-5 — CVaR / MAD portfolio classdefs. */
      "PortfolioCVaR", "PortfolioMAD", "setScenarios",
      "setProbabilityLevel", "estimatePortVaR",
      /* Financial Toolbox Tier-6 — SDE Monte Carlo. */
      "gbm", "cir", "hwv", "simByEuler", "simBySolution",
      /* Econometrics Toolbox model objects (econ_classdefs.m). */
      "arima", "garch", "egarch", "gjr", "varm", "ssm", "dssm",
      "bayeslm", "dtmc",
      /* Sensor Fusion and Tracking Toolbox — `fusion_classdefs.m` umbrella. */
      "quaternion", "trackingKF", "trackingEKF", "trackingUKF",
      "objectDetection", "imuSensor", "gpsSensor", "ahrsfilter",
      "imufilter", "complementaryFilter", "insfilterMARG",
      "ecompass", "slerp", "rotatepoint", "rotateframe",
      "quat2eul", "eul2quat", "quat2rotm", "rotm2quat",
      "allanvar", "constvel", "constacc", "constturn",
      "cvmeas", "cameas", "ctmeas", "initcvekf", "initctekf",
      "waypointTrajectory", "lookupPose", "lla2ned", "ned2lla",
      "assignmunkres", "trackerGNN", "objectTrack", "numConfirmed",
      "trackFuser", "trackGOSPAMetric", "trackOSPAMetric",
      "trackErrorMetrics", "rtsSmoother",
      /* Robotics System Toolbox — `robotics_classdefs.m` umbrella. */
      "se3", "so3", "rigidBodyTree", "addBody",
      "getTransform", "geometricJacobian",
      "homeConfiguration", "randomConfiguration", "loadrobot",
      "inverseKinematics", "constraintPoseTarget",
      "trvec2tform", "tform2trvec", "rotm2tform", "tform2rotm",
      "eul2tform", "tform2eul", "axang2rotm", "rotm2axang",
      "axang2tform", "tform2axang", "quat2tform", "tform2quat",
      "homtrans", "wrapToPi", "wrapTo2Pi", "vecnorm",
      "cubicpolytraj", "trapveltraj", "transformtraj",
      "massMatrix", "inverseDynamics",
      "forwardDynamics", "gravityTorque", "velocityProduct", "centerOfMass",
      "importrobot", "generalizedInverseKinematics",
      "constraintPositionTarget", "constraintOrientationTarget",
      "constraintJointBounds", "collisionCylinder", "collisionCapsule",
      "differentialDriveKinematics", "unicycleKinematics",
      "bicycleKinematics", "ackermannKinematics", "derivative",
      "binaryOccupancyMap",
      "mobileRobotPRM", "controllerPurePursuit",
      "setOccupancy", "getOccupancy", "checkOccupancy", "findpath",
      "collisionBox", "collisionSphere", "checkCollision",
      "manipulatorRRT", "plan",
      /* Navigation Toolbox — `navigation_classdefs.m` umbrella. */
      "occupancyMap", "stateSpaceSE2", "stateSpaceDubins",
      "validatorOccupancyMap", "navPath", "plannerRRT", "plannerRRTStar",
      "plannerAStarGrid", "lidarScan", "lidarSLAM", "poseGraph",
      "isStateValid", "isMotionValid", "matchScans", "optimizePoseGraph",
      "addRelativePose", "shortenpath", "sampleUniform",
      "controllerVFH", "monteCarloLocalization", "stateEstimatorPF",
      "gnssSensor", "referencePathFrenet", "trajectoryGeneratorFrenet",
      "getStateEstimate", "global2frenet", "frenet2global",
      "gnssconstellation", "receiverposition",
      /* Deep Learning Toolbox — `dlnet_classdefs.m`. */
      "dlarray", "dlgradient", "extractdata", "relu", "sigmoid",
      "softmax", "crossentropy", "mse", "lstm", "embed",
      "gru", "bilstm", "lstmp",
      "leakyrelu", "gelu", "swish", "softplus", "elu",
      "conv2d_batch", "conv2d_full", "maxpool2d", "avgpool2d", "batchnorm",
      "layernorm", "batchnorm_eval", "groupnorm", "batchnorm_train",
      "instancenorm", "rmsnorm",
      /* Reinforcement Learning Toolbox — `rl_classdefs.m` (Tier 1). */
      "rlPredefinedEnv", "rlMDPEnv", "rlFiniteSetSpec", "rlNumericSpec",
      "rlFunctionEnv", "rlTable", "rlQValueFunction", "rlQAgent",
      "rlSARSAAgent", "rlDQNAgent", "rlPGAgent",
      "rlDDPGAgent", "rlTD3Agent", "rlPPOAgent", "rlSACAgent", "rlGRPOAgent", "rlTRPOAgent",
      "rlQAgentOptions", "rlSARSAAgentOptions",
      "rlOptimizerOptions", "rlTrainingOptions", "rlSimulationOptions",
      "rlMaxQPolicy",
      "getObservationInfo", "getActionInfo", "getCritic",
      "getLearnableParameters", "getAction", "getMaxQValue", "getGreedyPolicy",
      /* GPU Coder T5 design-pattern helpers — runtime entries, no
       * prelude file needed.  Listed here only for the AOT-prelude
       * scanner's awareness (no leaf to map). */
    };
    for (const char *N : Names) {
      size_t NL = std::strlen(N);
      size_t P = 0;
      bool Hit = false;
      while ((P = Src.find(N, P)) != std::string::npos && !Hit) {
        bool LeftWord = (P > 0) && (std::isalnum((unsigned char)Src[P-1]) ||
                                     Src[P-1] == '_');
        if (!LeftWord && P + NL <= Src.size()) {
          char Right = (P + NL < Src.size()) ? Src[P + NL] : '\0';
          if (Right == '(') { Hit = true; break; }
          size_t Q = P + NL;
          while (Q < Src.size() && (Src[Q] == ' ' || Src[Q] == '\t')) Q++;
          if (Q < Src.size() && Src[Q] == '=') {
            if (Q + 1 >= Src.size() || Src[Q+1] != '=') { Hit = true; break; }
          }
          /* No-paren constructor on the RHS: `m = occupancyMap;` (#79.1).
           * A bare class name at end-of-source or followed only by a
           * statement terminator (`;`, `,`, newline) is also a mention,
           * so the classdef prelude is pulled in. Matching terminators
           * (not any non-word char) avoids treating `occupancyMap.foo`
           * as a hit; a spurious prelude load would be harmless anyway. */
          if (Q >= Src.size() || Src[Q] == ';' || Src[Q] == ',' ||
              Src[Q] == '\n' || Src[Q] == '\r') { Hit = true; break; }
        }
        P += NL;
      }
      if (Hit) Found.push_back(N);
    }
    return Found;
  };
  /* Per-class file name lookup — explicit table since the file names
   * are intentionally friendly (`comm_class_crc_generator.m`, not
   * the mechanical `comm_class_crcgenerator.m`). */
  auto extClassLeaf = [](llvm::StringRef ClsName) -> std::string {
    if (ClsName == "CommCRCGenerator")
      return "comm_class_crc_generator.m";
    if (ClsName == "CommCRCDetector")
      return "comm_class_crc_detector.m";
    if (ClsName == "AntDipole")
      return "ant_class_dipole.m";
    if (ClsName == "AntMonopole")
      return "ant_class_monopole.m";
    if (ClsName == "RFSparameters")
      return "rf_class_sparameters.m";
    if (ClsName == "RFYparameters")
      return "rf_class_yparameters.m";
    if (ClsName == "RFZparameters")
      return "rf_class_zparameters.m";
    if (ClsName == "RFHparameters")
      return "rf_class_hparameters.m";
    if (ClsName == "RFGparameters")
      return "rf_class_gparameters.m";
    if (ClsName == "RFAbcdparameters")
      return "rf_class_abcdparameters.m";
    if (ClsName == "RFTparameters")
      return "rf_class_tparameters.m";
    if (ClsName == "RFCktAmplifier")
      return "rf_class_amplifier.m";
    if (ClsName == "RFCktMixer")
      return "rf_class_mixer.m";
    if (ClsName == "RFCktPassive")
      return "rf_class_passive.m";
    if (ClsName == "RFCktCascade")
      return "rf_class_cascade.m";
    if (ClsName == "RFCktParallel")
      return "rf_class_parallel.m";
    if (ClsName == "RFCktSeries")
      return "rf_class_series.m";
    if (ClsName == "RFCktShunt")
      return "rf_class_shunt.m";
    if (ClsName == "RFRational")
      return "rf_class_rfrational.m";
    if (ClsName == "TxSite")
      return "rf_class_txsite.m";
    if (ClsName == "RxSite")
      return "rf_class_rxsite.m";
    if (ClsName == "PropagationModel")
      return "rf_class_propagationmodel.m";
    /* PDE Toolbox classdef façade — all classes share one umbrella
     * file `pde_classdefs.m` (mirrors the cst_classdefs.m pattern).
     * The first match short-circuits the rest, so loading the
     * umbrella once when ANY of these names is detected is
     * idempotent. */
    if (ClsName == "femodel" ||
        ClsName == "materialProperties" ||
        ClsName == "faceBC" ||
        ClsName == "edgeBC" ||
        ClsName == "vertexBC" ||
        ClsName == "faceLoad" ||
        ClsName == "edgeLoad" ||
        ClsName == "vertexLoad" ||
        ClsName == "cellLoad" ||
        ClsName == "StaticStructuralResults" ||
        ClsName == "StationaryResults" ||
        ClsName == "ThermalResults" ||
        ClsName == "ElectrostaticResults" ||
        ClsName == "MagneticResults" ||
        ClsName == "DCConductionResults" ||
        ClsName == "TransientStructuralResults" ||
        ClsName == "ModalStructuralResults" ||
        ClsName == "FrequencyStructuralResults" ||
        ClsName == "HarmonicEMResults")
      return "pde_classdefs.m";
    /* Optimization Toolbox problem-based API — umbrella file shared
     * by all detection names (deduped by PreludePaths below). */
    if (ClsName == "optimvar" || ClsName == "optimintvar" ||
        ClsName == "optimproblem" || ClsName == "eqnproblem" ||
        ClsName == "OptimizationExpression" ||
        ClsName == "OptimizationProblem" ||
        ClsName == "EquationProblem")
      return "optim_classdefs.m";
    /* MPC Toolbox umbrella. */
    if (ClsName == "mpc" || ClsName == "mpcstate" ||
        ClsName == "mpcmove" || ClsName == "mpcmoveopt" ||
        ClsName == "explicitMPC" ||
        ClsName == "generateExplicitMPC" ||
        ClsName == "mpcmoveExplicit" ||
        ClsName == "mpcActiveSetSolver" ||
        ClsName == "mpcmoveFinite" ||
        ClsName == "nlmpc" || ClsName == "nlmpcmove" ||
        ClsName == "mpcsimopt" || ClsName == "setEstimator" ||
        ClsName == "getEstimator" || ClsName == "review")
      return "mpc_classdefs.m";
    /* System Identification Toolbox umbrella. */
    if (ClsName == "iddata" || ClsName == "idpoly" || ClsName == "idss" ||
        ClsName == "idgrey" || ClsName == "greyest" ||
        ClsName == "impulseest" || ClsName == "forecast" ||
        ClsName == "idfrd" || ClsName == "etfe" || ClsName == "spa" ||
        ClsName == "extendedKalmanFilter" || ClsName == "unscentedKalmanFilter" ||
        ClsName == "correct" ||
        ClsName == "recursiveLS" || ClsName == "recursiveARX" ||
        ClsName == "idnlgrey" || ClsName == "nlgreyest" ||
        ClsName == "arxOptions" || ClsName == "getcov" ||
        ClsName == "getpvec" || ClsName == "setpvec" ||
        ClsName == "n4sid" || ClsName == "ssest" || ClsName == "tfest" ||
        ClsName == "arx" || ClsName == "ar" ||
        ClsName == "armax" || ClsName == "oe" || ClsName == "bj" ||
        ClsName == "iv4" || ClsName == "delayest" ||
        ClsName == "compare" || ClsName == "predict" ||
        ClsName == "resid" || ClsName == "goodnessOfFit")
      return "ident_classdefs.m";
    /* Global Optimization Toolbox Tier-2. */
    if (ClsName == "MultiStart" || ClsName == "GlobalSearch" ||
        ClsName == "createOptimProblem" || ClsName == "optimoptions")
      return "gads_classdefs.m";
    if (ClsName == "ProbDistUnivParam" || ClsName == "makedist" ||
        ClsName == "fitdist" || ClsName == "LinearModel" ||
        ClsName == "fitlm" || ClsName == "fitglm" ||
        ClsName == "ClassificationModel" || ClsName == "fitcknn" ||
        ClsName == "fitcnb" || ClsName == "fitcdiscr" || ClsName == "fitctree" ||
        ClsName == "fitcsvm" || ClsName == "fitcecoc" ||
        ClsName == "fitcensemble" || ClsName == "TreeBagger")
      return "stats_classdefs.m";
    if (ClsName == "affine2d" || ClsName == "projective2d" ||
        ClsName == "imref2d" || ClsName == "fitgeotform2d")
      return "image_classdefs.m";
    /* Curve Fitting Toolbox umbrella. */
    if (ClsName == "fit" || ClsName == "cfit" || ClsName == "sfit" ||
        ClsName == "fittype" || ClsName == "fitoptions" ||
        ClsName == "coeffvalues" || ClsName == "ppform" ||
        ClsName == "spline" || ClsName == "pchip" || ClsName == "ppmak" ||
        ClsName == "fnder" || ClsName == "fnint")
      return "curvefit_classdefs.m";
    /* Bioinformatics Toolbox Tier-4/6 — phytree + DataMatrix classdefs. */
    if (ClsName == "phytree" || ClsName == "seqlinkage" ||
        ClsName == "seqneighjoin" || ClsName == "DataMatrix")
      return "bioinfo_classdefs.m";
    /* DSP System Toolbox umbrella — any `dsp.*` package class. */
    if (ClsName.starts_with("dsp."))
      return "dsp_classdefs.m";
    /* DSP HDL Toolbox umbrella — any `dsphdl.*` package class. */
    if (ClsName.starts_with("dsphdl."))
      return "dsphdl_classdefs.m";
    /* GPU Coder host-side carriers — single umbrella file holding
     * gpuArray + coder_gpuConfig classdefs and gather/existsOnGPU/
     * gpuDevice free functions.  See docs/gpu_coder_roadmap.md T1.4. */
    if (ClsName == "gpuArray" || ClsName == "gather" ||
        ClsName == "existsOnGPU" || ClsName == "gpuDevice")
      return "gpu_classdefs.m";
    if (ClsName == "coder.gpuConfig" || ClsName == "coder_gpuConfig")
      return "gpu_config_classdefs.m";
    /* Financial Toolbox Tier-3 — Portfolio classdef umbrella. */
    if (ClsName == "Portfolio" ||
        ClsName == "setAssetMoments" || ClsName == "setBounds" ||
        ClsName == "setBudget" || ClsName == "setDefaultConstraints" ||
        ClsName == "estimateFrontier" ||
        ClsName == "estimateFrontierByReturn" ||
        ClsName == "estimateFrontierByRisk" ||
        ClsName == "estimatePortMoments" ||
        ClsName == "estimatePortReturn" ||
        ClsName == "estimatePortRisk" ||
        ClsName == "estimateMaxSharpeRatio" ||
        ClsName == "estimateAssetMoments" ||
        ClsName == "creditscorecard" || ClsName == "fitmodel" ||
        ClsName == "probdefault" ||
        ClsName == "PortfolioCVaR" || ClsName == "PortfolioMAD" ||
        ClsName == "setScenarios" || ClsName == "setProbabilityLevel" ||
        ClsName == "estimatePortVaR" ||
        ClsName == "gbm" || ClsName == "cir" || ClsName == "hwv" ||
        ClsName == "simByEuler" || ClsName == "simBySolution")
      return "finance_classdefs.m";
    /* Econometrics Toolbox model objects — econ_classdefs.m umbrella. */
    if (ClsName == "arima" || ClsName == "garch" ||
        ClsName == "egarch" || ClsName == "gjr" || ClsName == "varm" ||
        ClsName == "ssm" || ClsName == "dssm" ||
        ClsName == "bayeslm" || ClsName == "dtmc")
      return "econ_classdefs.m";
    /* Sensor Fusion and Tracking Toolbox — single umbrella file with the
     * `quaternion` value type, tracking filters, sensor models, and fusion
     * filters. */
    if (ClsName == "quaternion" || ClsName == "trackingKF" ||
        ClsName == "trackingEKF" || ClsName == "trackingUKF" ||
        ClsName == "objectDetection" || ClsName == "imuSensor" ||
        ClsName == "gpsSensor" || ClsName == "ahrsfilter" ||
        ClsName == "imufilter" || ClsName == "complementaryFilter" ||
        ClsName == "insfilterMARG" || ClsName == "ecompass" ||
        ClsName == "slerp" || ClsName == "rotatepoint" ||
        ClsName == "rotateframe" || ClsName == "quat2eul" ||
        ClsName == "eul2quat" || ClsName == "quat2rotm" ||
        ClsName == "rotm2quat" || ClsName == "allanvar" ||
        ClsName == "constvel" || ClsName == "constacc" ||
        ClsName == "constturn" || ClsName == "cvmeas" ||
        ClsName == "cameas" || ClsName == "ctmeas" ||
        ClsName == "initcvekf" || ClsName == "initctekf" ||
        ClsName == "waypointTrajectory" || ClsName == "lookupPose" ||
        ClsName == "lla2ned" || ClsName == "ned2lla" ||
        ClsName == "assignmunkres" || ClsName == "trackerGNN" ||
        ClsName == "objectTrack" || ClsName == "numConfirmed" ||
        ClsName == "trackFuser" || ClsName == "trackGOSPAMetric" ||
        ClsName == "trackOSPAMetric" || ClsName == "trackErrorMetrics" ||
        ClsName == "rtsSmoother")
      return "fusion_classdefs.m";
    /* Robotics System Toolbox umbrella. */
    if (ClsName == "se3" || ClsName == "so3" ||
        ClsName == "rigidBodyTree" || ClsName == "addBody" ||
        ClsName == "getTransform" || ClsName == "geometricJacobian" ||
        ClsName == "homeConfiguration" || ClsName == "randomConfiguration" ||
        ClsName == "loadrobot" ||
        ClsName == "inverseKinematics" || ClsName == "constraintPoseTarget" ||
        ClsName == "trvec2tform" || ClsName == "tform2trvec" ||
        ClsName == "rotm2tform" || ClsName == "tform2rotm" ||
        ClsName == "eul2tform" || ClsName == "tform2eul" ||
        ClsName == "axang2rotm" || ClsName == "rotm2axang" ||
        ClsName == "axang2tform" || ClsName == "tform2axang" ||
        ClsName == "quat2tform" || ClsName == "tform2quat" ||
        ClsName == "homtrans" || ClsName == "wrapToPi" ||
        ClsName == "wrapTo2Pi" || ClsName == "vecnorm" ||
        ClsName == "cubicpolytraj" || ClsName == "trapveltraj" ||
        ClsName == "transformtraj" ||
        ClsName == "massMatrix" || ClsName == "inverseDynamics" ||
        ClsName == "forwardDynamics" || ClsName == "gravityTorque" ||
        ClsName == "velocityProduct" || ClsName == "centerOfMass" ||
        ClsName == "importrobot" ||
        ClsName == "generalizedInverseKinematics" ||
        ClsName == "constraintPositionTarget" ||
        ClsName == "constraintOrientationTarget" ||
        ClsName == "constraintJointBounds" ||
        ClsName == "collisionCylinder" || ClsName == "collisionCapsule" ||
        ClsName == "differentialDriveKinematics" ||
        ClsName == "unicycleKinematics" || ClsName == "bicycleKinematics" ||
        ClsName == "ackermannKinematics" || ClsName == "derivative" ||
        ClsName == "binaryOccupancyMap" || ClsName == "mobileRobotPRM" ||
        ClsName == "controllerPurePursuit" ||
        ClsName == "setOccupancy" || ClsName == "getOccupancy" ||
        ClsName == "checkOccupancy" || ClsName == "findpath" ||
        ClsName == "collisionBox" || ClsName == "collisionSphere" ||
        ClsName == "checkCollision" || ClsName == "manipulatorRRT" ||
        ClsName == "plan")
      return "robotics_classdefs.m";
    /* Navigation Toolbox umbrella. */
    if (ClsName == "occupancyMap" || ClsName == "stateSpaceSE2" ||
        ClsName == "stateSpaceDubins" || ClsName == "validatorOccupancyMap" ||
        ClsName == "navPath" || ClsName == "plannerRRT" ||
        ClsName == "plannerRRTStar" || ClsName == "plannerAStarGrid" ||
        ClsName == "lidarScan" || ClsName == "lidarSLAM" ||
        ClsName == "poseGraph" || ClsName == "isStateValid" ||
        ClsName == "isMotionValid" || ClsName == "matchScans" ||
        ClsName == "optimizePoseGraph" || ClsName == "addRelativePose" ||
        ClsName == "shortenpath" || ClsName == "sampleUniform" ||
        ClsName == "controllerVFH" || ClsName == "monteCarloLocalization" ||
        ClsName == "stateEstimatorPF" || ClsName == "gnssSensor" ||
        ClsName == "referencePathFrenet" ||
        ClsName == "trajectoryGeneratorFrenet" ||
        ClsName == "getStateEstimate" || ClsName == "global2frenet" ||
        ClsName == "frenet2global" || ClsName == "gnssconstellation" ||
        ClsName == "receiverposition")
      return "navigation_classdefs.m";
    /* Deep Learning Toolbox umbrella. */
    if (ClsName == "dlarray" || ClsName == "dlgradient" ||
        ClsName == "extractdata" || ClsName == "relu" ||
        ClsName == "sigmoid" || ClsName == "softmax" ||
        ClsName == "crossentropy" || ClsName == "mse" ||
        ClsName == "lstm" || ClsName == "embed" ||
        ClsName == "gru" || ClsName == "bilstm" || ClsName == "lstmp" ||
        ClsName == "leakyrelu" || ClsName == "gelu" || ClsName == "swish" ||
        ClsName == "softplus" || ClsName == "elu" ||
        ClsName == "conv2d_batch" || ClsName == "conv2d_full" ||
        ClsName == "maxpool2d" || ClsName == "avgpool2d" ||
        ClsName == "batchnorm" || ClsName == "layernorm" ||
        ClsName == "batchnorm_eval" || ClsName == "groupnorm" ||
        ClsName == "batchnorm_train" || ClsName == "instancenorm" ||
        ClsName == "rmsnorm")
      return "dlnet_classdefs.m";
    if (ClsName == "rlPredefinedEnv" || ClsName == "rlMDPEnv" ||
        ClsName == "rlFiniteSetSpec" || ClsName == "rlNumericSpec" ||
        ClsName == "rlFunctionEnv" || ClsName == "rlTable" ||
        ClsName == "rlQValueFunction" || ClsName == "rlQAgent" ||
        ClsName == "rlSARSAAgent" || ClsName == "rlDQNAgent" ||
        ClsName == "rlDDPGAgent" || ClsName == "rlTD3Agent" ||
        ClsName == "rlPPOAgent" || ClsName == "rlSACAgent" ||
        ClsName == "rlGRPOAgent" || ClsName == "rlTRPOAgent" ||
        ClsName == "rlPGAgent" || ClsName == "rlQAgentOptions" ||
        ClsName == "rlSARSAAgentOptions" || ClsName == "rlOptimizerOptions" ||
        ClsName == "rlTrainingOptions" || ClsName == "rlSimulationOptions" ||
        ClsName == "rlMaxQPolicy" ||
        ClsName == "getObservationInfo" || ClsName == "getActionInfo" ||
        ClsName == "getCritic" || ClsName == "getLearnableParameters" ||
        ClsName == "getAction" || ClsName == "getMaxQValue" ||
        ClsName == "getGreedyPolicy")
      return "rl_classdefs.m";
    /* GPU Coder T5 design-pattern helpers are C runtime entries; no
     * classdef file to pull in. */
    return std::string();
  };
  /* DSP System-Object -> flat-fi source rewrite for the
   * -emit-systemverilog pipeline (Category-1 v1 of LowerDspSystemObjects).
   *
   * When the user writes a synthesizable dsp.FIRFilter SO + step
   * pattern inside a `%#codegen` function, rewrite the source to the
   * flat fi-array + persistent shift-register + MAC equivalent BEFORE
   * the prelude scan runs.  After the rewrite, the source no longer
   * mentions `dsp.FIRFilter`, so the prelude detection skips
   * dsp_classdefs.m and the SV pipeline never sees the SO machinery
   * (matlab_obj_* / matlab_dsp_iir_step et al. — those are
   * non-synthesizable and would fail HWLegalize).
   *
   * Bails (no-op) for any non-matching shape; the SV pipeline then
   * fires its normal "no synthesizable form" error, which is the right
   * failure mode for non-canonical inputs.  See
   * docs/dsp_so_to_sv_bridge.md for the full design + the eventual
   * MLIR-pass form. */
  if (Opts.Mode == Options::Mode::EmitSystemVerilog ||
      Opts.Mode == Options::Mode::CheckSynthesizable ||
      Opts.Mode == Options::Mode::EmitCocotb ||
      Opts.Mode == Options::Mode::EmitHardwareReport) {
    std::ifstream InSrc(Opts.InputPath);
    if (InSrc) {
      std::ostringstream Buf;
      Buf << InSrc.rdbuf();
      std::string Rewritten = matlab::sema::rewriteDspSoForSv(Buf.str());
      if (!Rewritten.empty()) {
        char Tmpl[] = "/tmp/matlabc-dsp-so-rewrite-XXXXXX.m";
        int Fd = mkstemps(Tmpl, 2);
        if (Fd >= 0) {
          ssize_t W = ::write(Fd, Rewritten.data(),
                              static_cast<size_t>(Rewritten.size()));
          ::close(Fd);
          if (W == static_cast<ssize_t>(Rewritten.size()))
            Opts.InputPath = Tmpl;  /* downstream sees the rewritten source */
        }
      }
    }
  }
  for (const std::string &Cls : userMentionsExtClasses(Opts.InputPath)) {
    std::string Leaf = extClassLeaf(Cls);
    std::string SelfStr(Argv[0]);
    auto last = SelfStr.find_last_of('/');
    std::string Bin = (last == std::string::npos) ? "." : SelfStr.substr(0, last);
    char Real[PATH_MAX];
    if (realpath(Bin.c_str(), Real)) Bin = Real;
    /* 2026-05 reorganization: probe runtime/toolbox/<name>/ subdirs
     * before falling back to the legacy flat layout. */
    static const char *kToolboxDirs[] = {
      "comm", "rf", "optim", "mpc", "ident", "gads", "pde", "prop", "sym",
      "stateflow", "antenna", "control", "stats", "images", "curvefit",
      "dsp", "gpu", "finance", "econ", "fusion", "robotics", "navigation",
      "dlnet", "rl", "bioinfo",
    };
    std::vector<std::string> Cands;
    for (const char *Tb : kToolboxDirs) {
      Cands.push_back(Bin + "/../runtime/toolbox/" + Tb + "/" + Leaf);
      Cands.push_back(Bin + "/runtime/toolbox/" + std::string(Tb) + "/" + Leaf);
      Cands.push_back(Bin + "/../share/matlabc/runtime/toolbox/" + Tb + "/" + Leaf);
    }
    Cands.push_back(Bin + "/../runtime/" + Leaf);
    Cands.push_back(Bin + "/runtime/" + Leaf);
    Cands.push_back(Bin + "/../share/matlabc/runtime/" + Leaf);
    for (auto &C : Cands) {
      std::ifstream Fp(C);
      if (Fp) {
        /* Dedupe: PDE umbrella `pde_classdefs.m` maps many class
         * names to a single file.  Loading it twice would duplicate
         * the function symbols (setMaterialProperties / setFaceBC /
         * solve / ...). */
        if (std::find(PreludePaths.begin(), PreludePaths.end(), C)
            == PreludePaths.end()) {
          PreludePaths.push_back(C);
        }
        break;
      }
    }
  }
  /* Back-compat single-path variable for the loader below; the
   * concat path now iterates PreludePaths. */
  std::string PreludePath = PreludePaths.empty() ? "" : PreludePaths.front();

  SourceManager SM;
  FileID F = 0;
  /* Always go through the concat path when a prelude is found. The
   * single-file fast path stays for .mflow / no-prelude builds. */
  if (Opts.ExtraInputs.empty() && PreludePaths.empty()) {
    F = SM.loadFile(Opts.InputPath);
    if (F == 0) {
      std::cerr << Opts.InputPath << ": cannot open file\n";
      return 1;
    }
  } else {
    /* Multi-file input — concatenate the optional CST prelude +
     * `Opts.InputPath` + every `ExtraInputs` path in CLI order with
     * `\n` separators, surface to the rest of the pipeline as one
     * synthetic buffer. The combined name is the primary input's
     * path (so diagnostics still mention a recognizable file) and
     * per-file `% --- file <path> ---` markers separate the regions. */
    std::string Combined;
    auto Append = [&](const std::string &P) -> bool {
      std::ifstream In(P, std::ios::binary);
      if (!In) {
        std::cerr << P << ": cannot open file\n";
        return false;
      }
      std::ostringstream Buf;
      Buf << In.rdbuf();
      if (!Combined.empty()) Combined += '\n';
      Combined += "% --- file ";
      Combined += P;
      Combined += " ---\n";
      Combined += Buf.str();
      return true;
    };
    /* Ordering: script files can mix top-level statements with classdef
     * blocks — script first, prelude last (the convention from
     * class_inherit.m / class_operators.m / etc.).  But a function-
     * defining file (starts with `function NAME(...)`) cannot be
     * followed by a classdef — the parser emits "stray tokens after
     * function definitions".  So PREPEND the prelude in that case.
     * Heuristic: peek the first non-comment, non-whitespace token of
     * the primary input and check for `function` or `classdef`. */
    bool PrimaryIsFunctionOrClass = false;
    {
      std::ifstream In(Opts.InputPath, std::ios::binary);
      if (In) {
        std::ostringstream Buf;
        Buf << In.rdbuf();
        std::string S = Buf.str();
        std::string Trim;
        Trim.reserve(S.size());
        bool InComment = false;
        for (char c : S) {
          if (c == '\n') { InComment = false; Trim.push_back(c); continue; }
          if (c == '%') InComment = true;
          if (!InComment) Trim.push_back(c);
        }
        size_t i = 0;
        while (i < Trim.size() && std::isspace(static_cast<unsigned char>(Trim[i])))
          ++i;
        auto starts = [&](const char *Kw) -> bool {
          size_t L = std::strlen(Kw);
          if (i + L > Trim.size()) return false;
          if (std::memcmp(Trim.data() + i, Kw, L) != 0) return false;
          char R = (i + L < Trim.size()) ? Trim[i + L] : ' ';
          return !(std::isalnum(static_cast<unsigned char>(R)) || R == '_');
        };
        PrimaryIsFunctionOrClass = starts("function") || starts("classdef");
      }
    }
    if (PrimaryIsFunctionOrClass) {
      /* Prelude classdefs first, then user file(s) so the function /
       * classdef definitions sit at the end of the combined buffer. */
      for (const auto &P : PreludePaths)
        if (!Append(P)) return 1;
      if (!Append(Opts.InputPath)) return 1;
      for (const auto &P : Opts.ExtraInputs)
        if (!Append(P)) return 1;
    } else {
      if (!Append(Opts.InputPath)) return 1;
      for (const auto &P : Opts.ExtraInputs)
        if (!Append(P)) return 1;
      for (const auto &P : PreludePaths)
        if (!Append(P)) return 1;
    }
    F = SM.addBuffer(Opts.InputPath, std::move(Combined));
  }

  DiagnosticEngine Diag(SM);

  /* `.mflow` inputs (the MatForge IDE flowchart format) bypass the
   * MATLAB lexer/parser and synthesize an AST directly from the
   * flowchart graph — see docs/flowchart_frontend.md. The resulting
   * TranslationUnit feeds the same Sema + MLIR pipeline below, so
   * every existing `-emit-*` mode works on `.mflow` inputs too. */
  auto endsWith = [](const std::string &S, std::string_view Suf) {
    return S.size() >= Suf.size() &&
           std::string_view(S).substr(S.size() - Suf.size()) == Suf;
  };
  bool IsFlow = endsWith(Opts.InputPath, ".mflow");

  ASTContext Ctx;
  TranslationUnit *TU = nullptr;
  std::vector<Token> Toks;

  if (IsFlow) {
    matlab::flowchart::BuildOptions BO;
    BO.BlockSearchPath = Opts.BlockPath;
    /* Append entries from MATFORGE_BLOCK_PATH (colon-separated). CLI
     * `--block-path` wins on first hit since it's listed first. */
    if (const char *Env = std::getenv("MATFORGE_BLOCK_PATH")) {
      std::string E = Env;
      size_t Start = 0;
      while (Start <= E.size()) {
        size_t Sep = E.find(':', Start);
        std::string Part = (Sep == std::string::npos)
                               ? E.substr(Start)
                               : E.substr(Start, Sep - Start);
        if (!Part.empty()) BO.BlockSearchPath.push_back(std::move(Part));
        if (Sep == std::string::npos) break;
        Start = Sep + 1;
      }
    }
    /* Custom-block `data.path` is resolved relative to the .mflow
     * file's containing directory. */
    {
      auto LastSlash = Opts.InputPath.find_last_of("/\\");
      if (LastSlash != std::string::npos)
        BO.MflowDirectory = Opts.InputPath.substr(0, LastSlash);
    }
    auto Doc = matlab::flowchart::loadMflow(SM, F, Diag);
    if (Doc) {
      /* mStateflow Tier 4 — a state-chart .mflow lowers to MATLAB
       * source (Lowering.cpp), which is then fed through the existing
       * lex/parse/sema/codegen pipeline so every -emit-* lane works
       * without further special-casing. -emit-matlab prints the
       * lowered source verbatim and exits; other modes drop the
       * lowered MATLAB into the SourceManager and continue. */
      if (Doc->isStateChart()) {
        auto Model = matlab::statechart::buildChartModel(*Doc, Diag);
        if (!Model) { Diag.printAll(); return 1; }
        const matlab::statechart::Chart *Entry = Model->entryChart();
        if (!Entry && !Model->Charts.empty()) Entry = &Model->Charts.front();
        if (!Entry) {
          std::cerr << Opts.InputPath
                    << ": state-chart document has no charts\n";
          return 1;
        }
        // SV / hardware-report / synthesizable-check lanes need the
        // HDL-friendly chart form (one-pass tick, integer types, per-
        // var isempty initialisers). Software lanes get the default.
        matlab::statechart::LoweringOptions LowOpts;
        switch (Opts.Mode) {
        case Options::Mode::EmitSystemVerilog:
        case Options::Mode::CheckSynthesizable:
        case Options::Mode::EmitHardwareReport:
        case Options::Mode::EmitCocotb:
          LowOpts.Target = matlab::statechart::LoweringTarget::SystemVerilog;
          break;
        default: break;
        }
        // Debug hook — `MATLABC_DUMP_SV_LOWER=1` prints the SV-target
        // MATLAB source to stderr so we can see what the SV pipeline
        // sees. Useful while iterating on the lowering.
        if (LowOpts.Target ==
                matlab::statechart::LoweringTarget::SystemVerilog &&
            getenv("MATLABC_DUMP_SV_LOWER")) {
          auto Dump = matlab::statechart::lowerChartToMatlab(*Entry, Diag,
                                                              LowOpts);
          if (Dump) std::cerr << Dump->MatlabSource;
        }
        auto Lowered =
            matlab::statechart::lowerChartToMatlab(*Entry, Diag, LowOpts);
        if (!Lowered) { Diag.printAll(); return 1; }
        if (Opts.Mode == Options::Mode::EmitMatlab ||
            Opts.Mode == Options::Mode::Format) {
          std::cout << Lowered->MatlabSource;
          Diag.printAll();
          return Diag.hasErrors() ? 1 : 0;
        }
        // For every other mode (-emit-c / -emit-cpp / -emit-mir / ...)
        // re-enter the standard .m pipeline on the lowered source.
        FileID Synth = SM.addBuffer(Opts.InputPath + ".lowered.m",
                                    Lowered->MatlabSource);
        Lexer Lx(SM, Synth, Diag);
        Toks = Lx.tokenize();
        Parser P(std::move(Toks), Ctx, Diag);
        TU = P.parseFile();
        if (Opts.Mode == Options::Mode::DumpTokens) {
          Diag.printAll();
          return Diag.hasErrors() ? 1 : 0;
        }
        // Fall through into the post-load `if (Opts.Mode ==
        // DumpAST)` / sema / MLIR / emit-* dispatch below.
        goto state_chart_lowered;
      }
      // Embedded Coder, Tier 7d — whole-diagram cocotb SIL emit.
      // `matlabc -emit-cocotb <model.mflow> --dut <block>` short-
      // circuits the regular AST + Sema + MLIR pipeline: the entry
      // flow is walked directly, the named DUT block's referenced
      // flow gets its own SV + Python emits via self-invocation,
      // and a top-level cocotb testbench is rendered from the
      // diagram's wiring. No host TU is needed because the test
      // harness drives the DUT and computes the host model in
      // Python directly.
      if (Opts.Mode == Options::Mode::EmitCocotb &&
          !Opts.CocotbDut.empty() && Doc->isSignalFlow()) {
        if (emitCocotbHarnessForDiagram(Argv[0], Opts, *Doc, Diag) != 0)
          return 1;
        Diag.printAll();
        return Diag.hasErrors() ? 1 : 0;
      }
      // Embedded Coder, Tier 1 — when the user supplies `--subsystem
      // <name>` against a signal-flow .mflow, run the dedicated
      // SubsystemToMatlab lowering instead of the control-flow
      // buildAST. Produces a single-function TU + driver call for
      // the downstream emit-* lanes (docs/embedded_coder_roadmap.md
      // §3, §6).
      if (!Opts.Subsystem.empty() && Doc->isSignalFlow()) {
        matlab::flowchart::SubsystemEmitOptions SO;
        SO.TargetRate = Opts.TargetRate;
        SO.DiscretizeMethod = Opts.Discretize;
        // Tier 5 — when the user is emitting SystemVerilog, switch
        // the lowering into HDL mode: continuous blocks become a
        // sourced error (no implicit auto-discretisation), state
        // emits as MATLAB `persistent` variables (which the SV
        // pipeline lowers to clocked registers), and the function
        // gains a `reset` arg for power-up clear.
        bool IsSV =
            Opts.Mode == Options::Mode::EmitSystemVerilog ||
            Opts.Mode == Options::Mode::CheckSynthesizable ||
            Opts.Mode == Options::Mode::EmitHardwareReport ||
            Opts.Mode == Options::Mode::EmitCocotb;
        // --state-form=persistent forces HDL-style state regardless
        // of target mode — useful for `-dump-ast` / `-emit-matlab`
        // inspection of the SV-shaped function.
        if (Opts.StateForm == "persistent") IsSV = true;
        if (IsSV) {
          SO.RejectContinuous = true;
          SO.StateAsPersistent = true;
        }
        // Parse `--fi-spec port=Q<W>.<F>` entries.
        // Form: name=[U]Q<W>.<F>. Default sign = signed (`Q...`);
        // unsigned prefix is `UQ...`.
        for (const auto &S : Opts.FiSpecs) {
          auto Eq = S.find('=');
          if (Eq == std::string::npos) continue;
          std::string Name = S.substr(0, Eq);
          std::string Spec = S.substr(Eq + 1);
          matlab::flowchart::FixedPointSpec FS;
          size_t Cur = 0;
          if (Spec.size() > 1 && (Spec[0] == 'U' || Spec[0] == 'u') &&
              (Spec[1] == 'Q' || Spec[1] == 'q')) {
            FS.Signed = false; Cur = 2;
          } else if (!Spec.empty() &&
                     (Spec[0] == 'Q' || Spec[0] == 'q')) {
            FS.Signed = true; Cur = 1;
          }
          auto Dot = Spec.find('.', Cur);
          if (Dot != std::string::npos) {
            try {
              FS.Width = std::stoi(Spec.substr(Cur, Dot - Cur));
              FS.Frac  = std::stoi(Spec.substr(Dot + 1));
            } catch (...) {
              std::cerr << "ignoring malformed --fi-spec \"" << S
                        << "\" (expected name=[U]Q<W>.<F>)\n";
              continue;
            }
          }
          SO.FiSpecs[Name] = FS;
        }
        TU = matlab::flowchart::buildSubsystemTU(*Doc, Opts.Subsystem,
                                                  Ctx, Diag, SO);
      } else if (Doc->isSignalFlow() && Doc->isSignalFlow() &&
                 (Opts.Mode == Options::Mode::EmitPython ||
                  Opts.Mode == Options::Mode::EmitC ||
                  Opts.Mode == Options::Mode::EmitCpp ||
                  Opts.Mode == Options::Mode::EmitTypeScript ||
                  Opts.Mode == Options::Mode::EmitMatlab)) {
        // Tier-7 — whole-diagram emit. When the user emits a
        // signal-flow .mflow WITHOUT `--subsystem` and targets a
        // software language, route through DiagramToMatlab. The
        // entry flow's sources / sinks / subsystems get bundled
        // into a `simulate()` function that runs the time loop
        // and returns per-sink log arrays. SV is excluded — the
        // SV lane stays per-subsystem.
        matlab::flowchart::SubsystemEmitOptions SO;
        SO.TargetRate = Opts.TargetRate;
        SO.DiscretizeMethod = Opts.Discretize;
        SO.TickCount     = Opts.Ticks;
        SO.LogDecimation = Opts.Decimation;
        TU = matlab::flowchart::buildDiagramTU(*Doc, Doc->Entry, Ctx,
                                                Diag, SO);
      } else {
        TU = matlab::flowchart::buildAST(*Doc, Ctx, SM, Diag, BO);
      }
    }
    if (Opts.Mode == Options::Mode::DumpTokens) {
      Diag.printAll();
      std::cerr << "warning: -dump-tokens does not apply to .mflow input\n";
      return Diag.hasErrors() ? 1 : 0;
    }
  } else {
    Lexer Lx(SM, F, Diag);
    Toks = Lx.tokenize();

    if (Opts.Mode == Options::Mode::DumpTokens) {
      dumpTokens(SM, Toks);
      Diag.printAll();
      return Diag.hasErrors() ? 1 : 0;
    }

    Parser P(std::move(Toks), Ctx, Diag);
    TU = P.parseFile();
  }
  // mStateflow Tier 4 — landing pad for the state-chart lowering
  // path. After it injects the lowered .m into SM and produces a TU,
  // control jumps here so the downstream sema / MLIR / emit-* dispatch
  // applies uniformly with control-flow / signal-flow inputs.
  state_chart_lowered: ;

  if (Opts.Mode == Options::Mode::DumpAST) {
    if (TU) dumpAST(std::cout, *TU);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::Format ||
      Opts.Mode == Options::Mode::EmitMatlab) {
    if (TU) formatAST(std::cout, *TU);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::EmitMflow) {
    /* Phase 8d: when `--preserve-layout PATH` is supplied, load the
     * reference document (possibly the same file we're re-emitting)
     * and pass it to the emitter so node `ui.position`s are
     * preserved across re-emits for unchanged blocks. */
    std::optional<matlab::flowchart::FlowDoc> RefDoc;
    if (TU && !Opts.PreserveLayoutPath.empty()) {
      SourceManager RefSM;
      DiagnosticEngine RefDiag(RefSM);
      RefDoc = matlab::flowchart::loadMflowFromPath(
          RefSM, Opts.PreserveLayoutPath, RefDiag);
      if (RefDiag.hasErrors()) {
        RefDiag.printAll();
        std::cerr << "warning: --preserve-layout target is malformed; "
                     "falling back to auto-layout\n";
        RefDoc.reset();
      }
    }
    if (TU) matlab::flowchart::emitMflow(
        std::cout, *TU, RefDoc ? &*RefDoc : nullptr);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  // Phase 2 of #38 — AST cloner round-trip. Inserts a deep-cloned copy of
  // each top-level function (suffix `__clone`) into the TU before Sema
  // runs, so Resolver + TypeInference see both originals and clones in
  // the same TU. If the cloner is correct, both halves resolve and infer
  // types independently and the dump shows structurally identical bindings.
  if (Opts.Mode == Options::Mode::TestAstClone && TU) {
    std::vector<Function *> Clones;
    Clones.reserve(TU->Functions.size());
    for (Function *F : TU->Functions) {
      if (!F) continue;
      std::string NewName = std::string(F->Name) + "__clone";
      Clones.push_back(cloneFunction(Ctx, *F, NewName));
    }
    for (Function *C : Clones)
      TU->Functions.push_back(C);
  }

  // Sema
  SemaContext Sema;
  TypeContext TC;
  Resolver R(Sema, TC, Diag);
  if (TU) R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  if (TU) Inf.run(*TU);
  // Whole-program (AOT) compile — the path -emit-llvm and the production
  // Sema-time monomorphizer (below) consume. Enable the PinnedClass fallback
  // so operators on pinned-but-not-object operands (e.g. a c2d-result tf) are
  // desynthed too, making them visible to P2/P5.
  if (TU) p3DesynthDispatch(Ctx, *TU, Inf, /*KeyOffPinnedClass=*/true);

  if (Opts.Mode == Options::Mode::EmitSema) {
    if (TU) dumpSema(std::cout, *TU);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::DumpCallSites) {
    // Phase 1 of #38: surface the per-callee signature buckets discovered
    // by TypeInference. No further pipeline stages run.
    if (TU) {
      CallSiteAnalysis Sites = analyzeCallSites(*TU);
      std::cout << Sites.dump();
    }
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::TestAstClone) {
    // Phase 2 of #38: clones were appended before Sema ran. Now dump the
    // augmented TU via the standard SemaDumper — golden compares original
    // vs `<name>__clone` bindings/types for structural equivalence.
    if (TU) dumpSema(std::cout, *TU);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::TestMonomorphize) {
    // Phases 3+4 of #38: run the monomorphizer to fixpoint. We re-use the
    // already-instantiated Sema / TC objects across iterations — Resolver
    // and TypeInference are idempotent on a re-run (Scope::declare returns
    // existing bindings, Refs/Ty pointers are overwritten by the new
    // walk). The callback below is invoked between iterations to refresh
    // ArgTypes on cloned bodies before the next analyze step.
    if (TU) {
      auto runSemaPass = [&]() {
        Resolver R2(Sema, TC, Diag);
        R2.resolve(*TU);
        TypeInference Inf2(Sema, TC, Diag);
        // Two passes — see comment on the env-gated path below.
        Inf2.run(*TU);
        // -test-monomorphize diagnostic mode: keep the PinnedClass fallback OFF
        // so the SemaMono golden tracks the Expr->Ty-only rewrite.
        p3DesynthDispatch(Ctx, *TU, Inf2, /*KeyOffPinnedClass=*/false);
        Inf2.run(*TU);
      };
      MonomorphizeStats S =
          runMonomorphize(*TU, Ctx, TC, runSemaPass, /*MaxIters=*/8);
      dumpSema(std::cout, *TU);
      std::cout << "monomorphize: iterations=" << S.Iterations
                << " clones=" << S.ClonesCreated
                << " rewrites=" << S.CallSitesRewritten
                << " converged=" << (S.Converged ? "true" : "false")
                << "\n";
    }
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  // Phase 5/6 of #38 — Sema-time monomorphization. Runs the
  // clone-and-stamp fixpoint loop so each user Function in the TU
  // ends up with concrete arg types for its surviving call-site
  // signature. The AST→MLIR lowerer then picks up the concrete types
  // naturally, replacing the late-pipeline PromoteNoneParams +
  // runMonomorphiseUserCalls workarounds. ON by default; set
  // MATLAB_LLVM_SEMA_MONO=0 to disable for bisecting downstream issues.
  //
  // Skipped on the HW emit lanes (SystemVerilog / CocoTB / synth check
  // / hardware report / fi report). Those flows have explicit port-
  // width contracts written by the user — fi types declared via
  // `numerictype(...)` and op-by-op width growth rules — that
  // Sema-mono's call-site-driven type propagation would override (e.g.
  // `y = a + b` of two int16 args widens to int17 at the function
  // boundary). Keeping the HW lane on the late MLIR pipeline preserves
  // golden output for the 79 EmitSV fixtures (port widths, saturation
  // shapes, body arithmetic).
  //
  // Sema dumps and diagnostic modes above (`-emit-sema`,
  // `-dump-call-sites`, `-test-ast-clone`, `-test-monomorphize`) have
  // already returned by this point so they see the pre-mono Sema state.
  // Test-monomorphize runs the same driver as part of its own flow.
  // #191 P5 measurement: enumerate the class constructor / instance-method
  // call signatures the late MLIR monomorphiser owns and a future Sema-time
  // class-mono must absorb. Gated, inert by default. `<Class>::<method>` plus
  // the arg-type signature; "ctor" marks a constructor (method == class name).
  if (TU && std::getenv("MATLAB_LLVM_PROBE_CLASSMONO")) {
    matlab::walkClassCallsWithCaller(
        *TU, [](matlab::Function *, matlab::CallOrIndex &C,
                const matlab::ClassDef &Recv, std::string_view Method) {
          bool IsCtor = (Method == Recv.Name);
          // argc = syntactic arity (C.Args). sigTypes = how many entries
          // TypeInference stamped onto C.ArgTypes — currently 0 for class
          // calls (TypeInference only populates ArgTypes for BindingKind::
          // Function), which a Sema-time class-mono must fix to bucket them.
          fprintf(stderr, "[class-mono] %.*s::%.*s%s argc=%zu sigTypes=%zu\n",
                  (int)Recv.Name.size(), Recv.Name.data(),
                  (int)Method.size(), Method.data(), IsCtor ? " ctor" : "",
                  C.Args.size(), C.ArgTypes.size());
        });
  }

  if (TU) {
    bool IsHwEmit =
        Opts.Mode == Options::Mode::EmitSystemVerilog ||
        Opts.Mode == Options::Mode::CheckSynthesizable ||
        Opts.Mode == Options::Mode::EmitHardwareReport ||
        Opts.Mode == Options::Mode::EmitCocotb ||
        Opts.Mode == Options::Mode::EmitFiReport;
    // Default: on for software lanes, off for HW lanes (port-width
    // contracts). The env var overrides either way: =1 forces on, =0
    // forces off — useful for bisecting and for the gated test sweep.
    bool MonoEnabled = !IsHwEmit;
    if (const char *Env = std::getenv("MATLAB_LLVM_SEMA_MONO"))
      MonoEnabled = !(*Env == '\0' || std::string_view(Env) == "0");
    /* Issue #75 — per-source opt-in for HW emit.  The pragma
     *   `% hdl: precise_fi`
     * (anywhere in the source) enables Sema-mono on the HW lane for
     * that file, which threads natural fi-growth widths through the
     * matlab.matmul / matlab.add result types so EmitSystemVerilog +
     * EmitPython produce mathematically equivalent fixed-point
     * arithmetic.  Without the pragma the HW lane stays on its
     * legacy lossy-truncation path (preserves the 79 existing EmitSV
     * goldens whose hand-rolled fi expressions were authored against
     * that behaviour).  Detected by a simple text scan over the
     * source file -- the canonical pragma scan in ScanHWPragmas runs
     * later in the pipeline. */
    if (!MonoEnabled && IsHwEmit && !Opts.InputPath.empty()) {
      std::ifstream PF(Opts.InputPath);
      if (PF) {
        std::string Src((std::istreambuf_iterator<char>(PF)),
                         std::istreambuf_iterator<char>());
        // Match `% hdl: precise_fi` with optional surrounding whitespace
        // and optional trailing arguments / comments.  Conservative
        // substring search keeps the check O(n) and tolerant of
        // line-wrapping artefacts.
        if (Src.find("hdl: precise_fi") != std::string::npos)
          MonoEnabled = true;
      }
    }
    if (MonoEnabled) {
      auto runSemaPass = [&]() {
        Resolver R2(Sema, TC, Diag);
        R2.resolve(*TU);
        TypeInference Inf2(Sema, TC, Diag);
        // Two passes: TypeInference visits the script before the
        // functions, so the script's call sites cannot see refined
        // OutputRefs[i]->InferredType on the first pass. The second
        // pass propagates the just-refined function output types into
        // script-level expressions (and into other functions' callers
        // when the call graph has forward references).
        Inf2.run(*TU);
        Inf2.run(*TU);
      };
      (void)runMonomorphize(*TU, Ctx, TC, runSemaPass, /*MaxIters=*/8);
    }
  }

  if (Opts.Mode == Options::Mode::EmitFiReport) {
    /* Walk every Sema-typed binding in the TU and print a one-line
     * summary for fi values. Modeled after MathWorks Coder's
     * type-proposal report — surfaces WL/FL/signedness/overflow per
     * binding. The intent is a low-cost sanity check before deploying
     * fi code: catch unexpected widenings, missing (:) clamps,
     * unintended Wrap modes. */
    auto modeName = [](FixedSpec::Overflow O) -> const char * {
      return O == FixedSpec::Overflow::Wrap ? "Wrap" : "Saturate";
    };
    auto roundName = [](FixedSpec::Rounding R) -> const char * {
      switch (R) {
      case FixedSpec::Rounding::Floor:      return "Floor";
      case FixedSpec::Rounding::Nearest:    return "Nearest";
      case FixedSpec::Rounding::Zero:       return "Zero";
      case FixedSpec::Rounding::Convergent: return "Convergent";
      case FixedSpec::Rounding::Ceiling:    return "Ceiling";
      }
      return "?";
    };
    auto printBinding = [&](const std::string &Scope, const std::string &Name,
                            const Type *T) {
      if (!T || T->K != Type::Kind::Array) return;
      auto &A = static_cast<const ArrayType &>(*T);
      if (A.Elt != Dtype::Fixed || !A.FxSpec) return;
      auto &S = *A.FxSpec;
      std::cout << "  " << (Scope.empty() ? "" : Scope + ".") << Name
                << " : " << (S.Signed ? "signed" : "unsigned")
                << " WL=" << int(S.WordLength)
                << " FL=" << int(S.FractionLength)
                << " IL=" << S.integerLength()
                << " " << modeName(S.OF)
                << "/" << roundName(S.RM)
                << " shape=" << A.S.toString()
                << "\n";
    };
    if (TU) {
      std::cout << "fixed-point report — " << Opts.InputPath << "\n";
      /* Script-level bindings live in the global Resolver scope rather
       * than on a Script node directly; we walk every function's
       * inferred bindings, plus any script-scope vars surfaced through
       * the resolver. For Phase 1 we just walk the functions — script
       * coverage is a follow-up. */
      for (Function *F : TU->Functions) {
        if (!F) continue;
        bool HeaderPrinted = false;
        auto reportOne = [&](std::string_view N, Binding *B) {
          if (!B || !B->InferredType) return;
          if (B->InferredType->K != Type::Kind::Array) return;
          auto &A = static_cast<const ArrayType &>(*B->InferredType);
          if (A.Elt != Dtype::Fixed) return;
          if (!HeaderPrinted) {
            std::cout << "[" << F->Name << "]\n";
            HeaderPrinted = true;
          }
          printBinding(std::string(F->Name), std::string(N), B->InferredType);
        };
        /* Walk inputs, then locals, then outputs. The display order
         * matches the function signature reading direction. */
        for (size_t i = 0; i < F->ParamRefs.size(); ++i)
          reportOne(F->Inputs[i], F->ParamRefs[i]);
        if (F->FnScope) {
          for (auto &[N, B] : F->FnScope->locals())
            if (B->Kind == BindingKind::Var)
              reportOne(N, B);
        }
        for (size_t i = 0; i < F->OutputRefs.size(); ++i)
          reportOne(F->Outputs[i], F->OutputRefs[i]);
      }
    }
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::EmitMIR) {
    mir::MIRContext MIRCtx;
    mir::Lowerer L(MIRCtx, TC, Diag);
    if (TU) {
      mir::Module M = L.lower(*TU);
      mir::printModule(std::cout, M);
    }
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

#if MATLAB_LLVM_WITH_MLIR
  if (Opts.Mode == Options::Mode::EmitCocotb) {
    if (!TU) return 1;
    if (emitCocotbHarness(Argv[0], Opts, *TU, TC, Diag, SM) != 0) return 1;
    return 0;
  }
  /* GPU Coder Tier-6 — AOT emit standalone bundle.  Each target writes
   * three files:
   *   <stem>_kernel.<dialect>   — device kernel template (.cu/.metal/.cl)
   *   <stem>_main.<host-ext>    — host driver (.cpp / .mm)
   *   Makefile                  — toolchain wiring (nvcc / xcrun metal / clang)
   * The kernel body uses the CPU equivalent of the user's function (the
   * T1 LowerGpuKernels rewrite-to-`matlab.for` lane lowered through
   * EmitC).  Tier-2/3/4 will swap the kernel template for the real
   * outlined body once the array-capture extension to the outliner
   * lands. */
  if (Opts.Mode == Options::Mode::EmitCuda  ||
      Opts.Mode == Options::Mode::EmitMetal ||
      Opts.Mode == Options::Mode::EmitOpenCL) {
    if (!TU) return 1;
    /* Stem from the input file basename (no extension). */
    std::string Stem = Opts.InputPath;
    auto slash = Stem.find_last_of('/');
    if (slash != std::string::npos) Stem = Stem.substr(slash + 1);
    auto dot = Stem.find_last_of('.');
    if (dot != std::string::npos) Stem = Stem.substr(0, dot);
    /* Output directory: <stem>_<target> next to the input. */
    std::string Target;
    std::string KernelExt;
    std::string HostExt = "cpp";
    std::string ToolchainComment;
    std::string CompileCmd;
    std::string BuildExtras;
    std::string KernelDecl;
    std::string KernelBody;
    std::string ThreadIdLine;
    if (Opts.Mode == Options::Mode::EmitCuda) {
      Target = "cuda";   KernelExt = "cu";
      ToolchainComment = "# Requires CUDA Toolkit (nvcc + libcudart + libcublas).";
      CompileCmd = "nvcc -O2 -std=c++17";
      BuildExtras = " -lcublas -lcufft -lcusolver";
      KernelDecl =
          "__global__ __launch_bounds__(256, 1) void "
          + Stem + "_kernel(const double *X, double *Y, int n)";
      ThreadIdLine =
          "  int tid = blockIdx.x * blockDim.x + threadIdx.x;";
    } else if (Opts.Mode == Options::Mode::EmitMetal) {
      Target = "metal";  KernelExt = "metal";  HostExt = "mm";
      ToolchainComment = "# Requires Xcode (xcrun metal / metallib) + clang++.";
      CompileCmd = "clang++ -O2 -std=c++20 -fobjc-arc";
      BuildExtras = " -framework Metal -framework MetalPerformanceShaders -framework Foundation";
      KernelDecl =
          "kernel void " + Stem + "_kernel(\n"
          "  device const double *X [[buffer(0)]],\n"
          "  device       double *Y [[buffer(1)]],\n"
          "  constant uint &n        [[buffer(2)]],\n"
          "  uint tid                [[thread_position_in_grid]])";
      ThreadIdLine = "  /* tid is the kernel param. */";
    } else {  /* EmitOpenCL */
      Target = "opencl"; KernelExt = "cl";
      ToolchainComment = "# Requires OpenCL ICD loader (libOpenCL) + clang++.";
      CompileCmd = "clang++ -O2 -std=c++17";
      BuildExtras = " -lOpenCL";
      KernelDecl =
          "__kernel void " + Stem + "_kernel(\n"
          "    __global const double *X,\n"
          "    __global       double *Y,\n"
          "    const int n)";
      ThreadIdLine = "  int tid = get_global_id(0);";
    }
    /* Pick the output dir. */
    std::string OutDir = Stem + "_" + Target;
    std::error_code Ec;
    /* Use mkdir(2) — std::filesystem available in C++17 but the rest of
     * the tool uses cstdio shapes; mkdir/EEXIST is simpler. */
    if (::mkdir(OutDir.c_str(), 0755) != 0 && errno != EEXIST) {
      std::cerr << "error: -emit-" << Target << ": cannot create "
                << OutDir << ": " << std::strerror(errno) << "\n";
      return 1;
    }
    /* Descriptor of the first emitted CUDA kernel — drives the
     * NVRTC-based host driver + Makefile below.  Stays empty (kernelCount
     * 0) for Metal/OpenCL and when no kernel was emitted. */
    mlirgen::GpuKernelInfo GpuInfo;
    /* Kernel file. */
    {
      std::string Path = OutDir + "/" + Stem + "_kernel." + KernelExt;
      std::ofstream OF(Path);
      if (!OF) {
        std::cerr << "error: cannot write " << Path << "\n";
        return 1;
      }
      /* T2.B / T3 / T4 — for each Metal/CUDA/OpenCL emit mode, build
       * a one-shot MLIR module and walk the matlab.gpu.kernel ops via
       * the target-specific emit pass.  Result is the real kernel
       * source translated from the user's MATLAB body.  Falls back to
       * the identity placeholder if the MLIR build errors or no
       * matlab.gpu.kernel ops were found. */
      std::string KernelSource;
      if (TU) {
        mlirgen::Context MCtx;
        DiagnosticEngine TmpDiag(SM);
        auto M = mlirgen::lowerToMLIR(MCtx, TC, TmpDiag, *TU, &SM,
                                       /*ReplMode=*/false,
                                       /*DebugMode=*/false);
        if (!TmpDiag.hasErrors()) {
          if (Opts.Mode == Options::Mode::EmitMetal)
            KernelSource = mlirgen::emitMetalKernels(M, Stem, &GpuInfo);
          else if (Opts.Mode == Options::Mode::EmitCuda)
            KernelSource = mlirgen::emitCudaKernels(M, Stem, &GpuInfo);
          else if (Opts.Mode == Options::Mode::EmitOpenCL)
            KernelSource = mlirgen::emitOpenCLKernels(M, Stem, &GpuInfo);
        }
      }
      if (!KernelSource.empty()) {
        OF << "/* " << Path << "\n"
           << " * Generated by matlabc -emit-" << Target << "\n"
           << " * Source: " << Opts.InputPath << "\n"
           << " *\n"
           << " * T2.B/T3/T4: body translated op-by-op from the user's\n"
           << " * coder.gpu.kernelfun-tagged MATLAB body.  Unsupported\n"
           << " * op shapes inline a FALLBACK comment + identity body. */\n\n"
           << KernelSource;
      } else {
        OF << "/* " << Path << "\n"
           << " * Generated by matlabc -emit-" << Target << "\n"
           << " * Source: " << Opts.InputPath << "\n"
           << " * \n"
           << " * GPU Coder Tier-6 v1 kernel template.  The body below is\n"
           << " * the placeholder that future per-target outliners will\n"
           << " * fill in with the user's `coder.gpu.kernelfun`-tagged\n"
           << " * function body.  Today the host driver calls the CPU\n"
           << " * equivalent (see " << Stem << "_main.cpp).\n"
           << " */\n\n"
           << KernelDecl << " {\n"
           << ThreadIdLine << "\n"
           << "  if (tid >= n) return;\n"
           << "  /* TODO: outlined kernel body lands here (blocked on the\n"
           << "   * array-capture extension to LowerGpuKernels). */\n"
           << "  Y[tid] = X[tid];  /* identity kernel — replace */\n"
           << "}\n";
      }
    }
    /* Host driver. */
    {
      std::string Path = OutDir + "/" + Stem + "_main." + HostExt;
      std::ofstream OF(Path);
      if (!OF) {
        std::cerr << "error: cannot write " << Path << "\n";
        return 1;
      }
      OF << "/* " << Path << "\n"
         << " * Generated by matlabc -emit-" << Target << "\n"
         << " * Source: " << Opts.InputPath << "\n"
         << " * \n";
      if (Opts.Mode == Options::Mode::EmitCuda) {
        OF << " * GPU Coder Tier-3 CUDA host driver.  JIT-compiles\n"
           << " * " << Stem << "_kernel.cu via NVRTC, allocates the output\n"
           << " * buffer, launches the kernel via the CUDA driver API, and\n"
           << " * prints the result.  No nvcc required.\n"
           << " */\n\n";
      } else {
        OF << " * GPU Coder Tier-6 host driver.  Allocates device buffers,\n"
           << " * uploads X, launches " << Stem << "_kernel, downloads Y.\n"
           << " * The driver demonstrates the host-side dispatch shape;\n"
           << " * full device-kernel runtime semantics arrive in T2.B-D\n"
           << " * (Metal) / T3 (CUDA) / T4 (OpenCL).\n"
           << " */\n\n";
      }
      if (Opts.Mode == Options::Mode::EmitCuda) {
        bool runnable = (GpuInfo.kernelCount == 1 && GpuInfo.hasOutput &&
                         !GpuInfo.bailed);
        if (!runnable) {
          /* Body not fully translatable (multiple kernels, no output, or
           * a FALLBACK identity body) — emit a compilable stub that says
           * so rather than a driver that would launch garbage. */
          OF << "#include <cstdio>\n\n"
             << "/* This program's coder.gpu.kernelfun body did not fully\n"
             << " * translate to a single device kernel (see the FALLBACK\n"
             << " * note in " << Stem << "_kernel.cu).  The bundle is\n"
             << " * emission-only for this input; the host driver below is\n"
             << " * a no-op placeholder. */\n"
             << "int main() {\n"
             << "  std::printf(\"" << Stem
             << ": kernel not fully translated; emission-only bundle.\\n\");\n"
             << "  return 0;\n"
             << "}\n";
        } else {
          /* NVRTC-based driver: read the emitted .cu, JIT-compile it for
           * the local device's compute capability, and launch the real
           * kernel (" << GpuInfo.name << ") via the driver API.  No nvcc
           * needed.  Scalar captures are set to a demo value (2.0); grid
           * size n comes from argv[1] (default 16). */
          OF << "#include <cstdio>\n"
             << "#include <cstdlib>\n"
             << "#include <fstream>\n"
             << "#include <sstream>\n"
             << "#include <string>\n"
             << "#include <vector>\n"
             << "#include <cuda.h>\n"
             << "#include <nvrtc.h>\n\n"
             << "static void ck(CUresult r, const char *w) {\n"
             << "  if (r) { const char *s = 0; cuGetErrorString(r, &s);\n"
             << "    std::fprintf(stderr, \"CUDA %s: %s\\n\", w, s ? s : \"?\");\n"
             << "    std::exit(1); }\n"
             << "}\n\n"
             << "int main(int argc, char **argv) {\n"
             // `mlc_n` (not `n`) to avoid colliding with a scalar capture
             // literally named `n` (e.g. Mandelbrot's grid size).
             << "  int mlc_n = argc > 1 ? std::atoi(argv[1]) : 16;\n"
             << "  ck(cuInit(0), \"init\");\n"
             << "  CUdevice dev; ck(cuDeviceGet(&dev, 0), \"device\");\n"
             << "  int major = 0, minor = 0;\n"
             << "  cuDeviceGetAttribute(&major, "
                "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);\n"
             << "  cuDeviceGetAttribute(&minor, "
                "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);\n"
             << "  CUcontext ctx; ck(cuCtxCreate(&ctx, 0, dev), \"context\");\n"
             << "  std::ifstream ifs(\"" << Stem << "_kernel.cu\");\n"
             << "  std::stringstream ss; ss << ifs.rdbuf();\n"
             << "  std::string src = ss.str();\n"
             << "  if (src.empty()) { std::fprintf(stderr, \"cannot read "
             << Stem << "_kernel.cu — run this program from the bundle "
             << "directory\\n\"); return 1; }\n"
             << "  nvrtcProgram prog;\n"
             << "  if (nvrtcCreateProgram(&prog, src.c_str(), \"" << Stem
             << "_kernel.cu\", 0, 0, 0) != NVRTC_SUCCESS) return 1;\n"
             << "  char arch[40];\n"
             << "  std::snprintf(arch, sizeof(arch), "
                "\"--gpu-architecture=sm_%d%d\", major, minor);\n"
             << "  const char *opts[] = {arch};\n"
             << "  if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS) {\n"
             << "    size_t ls = 0; nvrtcGetProgramLogSize(prog, &ls);\n"
             << "    std::vector<char> lg(ls ? ls : 1); "
                "nvrtcGetProgramLog(prog, lg.data());\n"
             << "    std::fprintf(stderr, \"NVRTC:\\n%s\\n\", lg.data()); "
                "return 1;\n"
             << "  }\n"
             << "  size_t ps = 0; nvrtcGetPTXSize(prog, &ps);\n"
             << "  std::vector<char> ptx(ps); nvrtcGetPTX(prog, ptx.data());\n"
             << "  CUmodule mod; ck(cuModuleLoadData(&mod, ptx.data()), "
                "\"module\");\n"
             << "  CUfunction fn; ck(cuModuleGetFunction(&fn, mod, \""
             << GpuInfo.name << "\"), \"function\");\n"
             << "  size_t total = "
             << (GpuInfo.twoD ? "(size_t)mlc_n * mlc_n" : "(size_t)mlc_n")
             << ";\n"
             << "  CUdeviceptr d_out;\n"
             << "  ck(cuMemAlloc(&d_out, total * sizeof(double)), \"alloc\");\n";
          /* Demo scalar captures (const double in the kernel signature). */
          for (const auto &s : GpuInfo.scalarArgs)
            OF << "  double " << s << " = (double)mlc_n;  /* demo capture */\n";
          if (GpuInfo.twoD) {
            /* 2-D flattened grid: out, captures, then nrows + ncols. */
            OF << "  int nrows = mlc_n, ncols = mlc_n;\n"
               << "  void *args[] = { &d_out";
            for (const auto &s : GpuInfo.scalarArgs) OF << ", &" << s;
            OF << ", &nrows, &ncols };\n"
               << "  unsigned bx = 16, by = 16;\n"
               << "  unsigned gx = (mlc_n + bx - 1) / bx, "
                  "gy = (mlc_n + by - 1) / by;\n"
               << "  ck(cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, 0, "
                  "args, 0), \"launch\");\n";
          } else {
            OF << "  int n_grid = mlc_n;\n"
               << "  void *args[] = { &d_out";
            for (const auto &s : GpuInfo.scalarArgs) OF << ", &" << s;
            OF << ", &n_grid };\n"
               << "  int block = 256, grid = (mlc_n + block - 1) / block;\n"
               << "  ck(cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, 0, "
                  "args, 0), \"launch\");\n";
          }
          OF << "  ck(cuCtxSynchronize(), \"sync\");\n"
             << "  std::vector<double> h_out(total);\n"
             << "  ck(cuMemcpyDtoH(h_out.data(), d_out, "
                "total * sizeof(double)), \"d2h\");\n"
             << "  double sum = 0;\n"
             << "  for (size_t i = 0; i < total; ++i) sum += h_out[i];\n"
             << "  std::printf(\"" << Stem << ": checksum = %.4f\\n\", sum);\n"
             << "  cuMemFree(d_out);\n"
             << "  return 0;\n"
             << "}\n";
        }
      } else if (Opts.Mode == Options::Mode::EmitMetal) {
        bool runnable = (GpuInfo.kernelCount == 1 && GpuInfo.hasOutput &&
                         !GpuInfo.bailed);
        if (!runnable) {
          /* Body not fully translatable (multiple kernels, no output, or
           * a FALLBACK identity body) — emit a compilable stub rather
           * than a driver that would launch garbage. */
          OF << "#include <cstdio>\n\n"
             << "/* This program's coder.gpu.kernelfun body did not fully\n"
             << " * translate to a single device kernel (see the FALLBACK\n"
             << " * note in " << Stem << "_kernel.metal).  The bundle is\n"
             << " * emission-only for this input. */\n"
             << "int main() {\n"
             << "  std::printf(\"" << Stem
             << ": kernel not fully translated; emission-only bundle.\\n\");\n"
             << "  return 0;\n"
             << "}\n";
        } else {
          /* Driver matching the emitted kernel's ABI: out at buffer(0),
           * scalar captures (set to a demo value 2.0) at buffer(1..N),
           * dispatched over n (1-D) or n×n (2-D, the flattened
           * for-i×for-j grid).  n comes from argv[1] (default 16).  The
           * 2-D leading dimension is the grid's i-extent (gsz.y), so no
           * extra argument is needed. */
          OF << "#import <Metal/Metal.h>\n"
             << "#import <Foundation/Foundation.h>\n"
             << "#include <cstdio>\n"
             << "#include <cstdlib>\n\n"
             << "int main(int argc, char **argv) {\n"
             << "  @autoreleasepool {\n"
             // `mlc_n` (not `n`) to avoid colliding with a scalar capture
             // literally named `n` (e.g. Mandelbrot's grid size).
             << "    int mlc_n = argc > 1 ? atoi(argv[1]) : 16;\n"
             << "    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();\n"
             << "    if (!dev) { std::fprintf(stderr, \"no Metal device\\n\"); return 1; }\n"
             << "    id<MTLCommandQueue> q = [dev newCommandQueue];\n"
             << "    NSError *err = nil;\n"
             << "    NSString *src = [NSString stringWithContentsOfFile:@\""
             << Stem << "_kernel.metal\"\n"
             << "                                              encoding:NSUTF8StringEncoding error:&err];\n"
             << "    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];\n"
             << "    if (!lib) { std::fprintf(stderr, \"MSL compile: %s\\n\", err.localizedDescription.UTF8String); return 1; }\n"
             << "    id<MTLFunction> fn = [lib newFunctionWithName:@\""
             << GpuInfo.name << "\"];\n"
             << "    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];\n"
             << "    size_t total = "
             << (GpuInfo.twoD ? "(size_t)mlc_n * mlc_n" : "(size_t)mlc_n")
             << ";\n"
             << "    id<MTLBuffer> by = [dev newBufferWithLength:total*sizeof(float) options:MTLResourceStorageModeShared];\n";
          /* Demo scalar captures (constant float& at buffer 1..N). */
          for (const auto &s : GpuInfo.scalarArgs)
            OF << "    float " << s << " = (float)mlc_n;  /* demo capture */\n";
          OF << "    id<MTLCommandBuffer> cmdbuf = [q commandBuffer];\n"
             << "    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];\n"
             << "    [enc setComputePipelineState:pso];\n"
             << "    [enc setBuffer:by offset:0 atIndex:0];\n";
          unsigned BufIdx = 1;
          for (const auto &s : GpuInfo.scalarArgs)
            OF << "    [enc setBytes:&" << s << " length:sizeof(float) atIndex:"
               << BufIdx++ << "];\n";
          if (GpuInfo.twoD)
            OF << "    [enc dispatchThreads:MTLSizeMake(mlc_n, mlc_n, 1)\n"
               << "           threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];\n";
          else
            OF << "    [enc dispatchThreads:MTLSizeMake(mlc_n, 1, 1)\n"
               << "           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];\n";
          OF << "    [enc endEncoding];\n"
             << "    [cmdbuf commit];\n"
             << "    [cmdbuf waitUntilCompleted];\n"
             << "    float *out = (float *)[by contents];\n"
             << "    double sum = 0;\n"
             << "    for (size_t i = 0; i < total; ++i) sum += out[i];\n"
             << "    std::printf(\"" << Stem
             << ": checksum = %.4f\\n\", sum);\n"
             << "  }\n"
             << "  return 0;\n"
             << "}\n";
        }
      } else {  /* OpenCL */
        bool runnable = (GpuInfo.kernelCount == 1 && GpuInfo.hasOutput &&
                         !GpuInfo.bailed);
        if (!runnable) {
          OF << "#include <cstdio>\n\n"
             << "/* This program's coder.gpu.kernelfun body did not fully\n"
             << " * translate to a single device kernel (see the FALLBACK\n"
             << " * note in " << Stem << "_kernel.cl).  The bundle is\n"
             << " * emission-only for this input. */\n"
             << "int main() {\n"
             << "  std::printf(\"" << Stem
             << ": kernel not fully translated; emission-only bundle.\\n\");\n"
             << "  return 0;\n"
             << "}\n";
        } else {
          /* Header-free-capable driver: use the platform's OpenCL headers
           * when present (end users with an SDK), else hand-declare the
           * minimal OpenCL 1.2 API so the bundle compiles against just the
           * ICD loader (libOpenCL) with no SDK installed.  Reads the .cl
           * at runtime, builds it, and launches the real kernel
           * (" << GpuInfo.name << ").  Scalar captures are set to a demo
           * value (2.0); grid size n comes from argv[1] (default 16). */
          OF << "#include <cstdio>\n"
             << "#include <cstdlib>\n"
             << "#include <cstddef>\n"
             << "#include <cstdint>\n"
             << "#include <fstream>\n"
             << "#include <sstream>\n"
             << "#include <string>\n\n"
             << "#if defined(__has_include)\n"
             << "#  if __has_include(<CL/cl.h>)\n"
             << "#    define CL_TARGET_OPENCL_VERSION 120\n"
             << "#    include <CL/cl.h>\n"
             << "#    define MATLAB_HAVE_CL_HEADER 1\n"
             << "#  elif __has_include(<OpenCL/opencl.h>)\n"
             << "#    include <OpenCL/opencl.h>\n"
             << "#    define MATLAB_HAVE_CL_HEADER 1\n"
             << "#  endif\n"
             << "#endif\n"
             << "#ifndef MATLAB_HAVE_CL_HEADER\n"
             << "extern \"C\" {\n"
             << "typedef int cl_int; typedef unsigned int cl_uint;\n"
             << "typedef unsigned long cl_ulong;\n"
             << "typedef void *cl_platform_id; typedef void *cl_device_id;\n"
             << "typedef void *cl_context; typedef void *cl_command_queue;\n"
             << "typedef void *cl_program; typedef void *cl_kernel;\n"
             << "typedef void *cl_mem; typedef cl_ulong cl_mem_flags;\n"
             << "typedef cl_ulong cl_device_type; typedef cl_uint cl_bool;\n"
             << "cl_int clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);\n"
             << "cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, "
                "cl_device_id*, cl_uint*);\n"
             << "cl_context clCreateContext(const intptr_t*, cl_uint, "
                "const cl_device_id*, void*, void*, cl_int*);\n"
             << "cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, "
                "cl_ulong, cl_int*);\n"
             << "cl_program clCreateProgramWithSource(cl_context, cl_uint, "
                "const char**, const size_t*, cl_int*);\n"
             << "cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, "
                "const char*, void*, void*);\n"
             << "cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, "
                "size_t, void*, size_t*);\n"
             << "cl_kernel clCreateKernel(cl_program, const char*, cl_int*);\n"
             << "cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, "
                "cl_int*);\n"
             << "cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);\n"
             << "cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, "
                "cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, "
                "const void*, void*);\n"
             << "cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, "
                "size_t, size_t, void*, cl_uint, const void*, void*);\n"
             << "cl_int clFinish(cl_command_queue);\n"
             << "}\n"
             << "#define CL_DEVICE_TYPE_DEFAULT 1UL\n"
             << "#define CL_MEM_READ_WRITE 1UL\n"
             << "#define CL_TRUE 1\n"
             << "#define CL_PROGRAM_BUILD_LOG 0x1183\n"
             << "#endif\n\n"
             << "int main(int argc, char **argv) {\n"
             // `mlc_n` (not `n`) to avoid colliding with a scalar capture
             // literally named `n` (e.g. Mandelbrot's grid size).
             << "  int mlc_n = argc > 1 ? std::atoi(argv[1]) : 16;\n"
             << "  cl_platform_id plat; cl_uint np = 0;\n"
             << "  if (clGetPlatformIDs(1, &plat, &np) != 0 || np == 0) {\n"
             << "    std::fprintf(stderr, \"no OpenCL platform\\n\"); return 1; }\n"
             << "  cl_device_id dev; cl_uint nd = 0;\n"
             << "  if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, &nd)"
                " != 0 || nd == 0) {\n"
             << "    std::fprintf(stderr, \"no OpenCL device\\n\"); return 1; }\n"
             << "  cl_int err = 0;\n"
             << "  cl_context ctx = clCreateContext(0, 1, &dev, 0, 0, &err);\n"
             << "  cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, "
                "&err);\n"
             << "  std::ifstream ifs(\"" << Stem << "_kernel.cl\");\n"
             << "  std::stringstream ss; ss << ifs.rdbuf();\n"
             << "  std::string src = ss.str();\n"
             << "  if (src.empty()) { std::fprintf(stderr, \"cannot read "
             << Stem << "_kernel.cl — run from the bundle directory\\n\"); "
                "return 1; }\n"
             << "  const char *csrc = src.c_str(); size_t slen = src.size();\n"
             << "  cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, "
                "&slen, &err);\n"
             << "  if (clBuildProgram(prog, 1, &dev, \"\", 0, 0) != 0) {\n"
             << "    char log[8192] = {0};\n"
             << "    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, "
                "sizeof(log), log, 0);\n"
             << "    std::fprintf(stderr, \"OpenCL build:\\n%s\\n\", log); "
                "return 1; }\n"
             << "  cl_kernel kernel = clCreateKernel(prog, \"" << GpuInfo.name
             << "\", &err);\n"
             << "  size_t total = "
             << (GpuInfo.twoD ? "(size_t)mlc_n * mlc_n" : "(size_t)mlc_n")
             << ";\n"
             << "  cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, "
                "total * sizeof(double), 0, &err);\n";
          for (const auto &s : GpuInfo.scalarArgs)
            OF << "  double " << s << " = (double)mlc_n;  /* demo capture */\n";
          /* The 2-D kernel keeps the trailing n_grid arg (unused in the
           * body — the leading dim comes from get_global_size(1)), so the
           * argument list is identical for 1-D and 2-D; only the NDRange
           * dimensionality and the buffer size differ. */
          OF << "  int n_grid = mlc_n;\n"
             << "  cl_uint ai = 0;\n"
             << "  clSetKernelArg(kernel, ai++, sizeof(cl_mem), &d_out);\n";
          for (const auto &s : GpuInfo.scalarArgs)
            OF << "  clSetKernelArg(kernel, ai++, sizeof(double), &" << s
               << ");\n";
          OF << "  clSetKernelArg(kernel, ai++, sizeof(int), &n_grid);\n";
          if (GpuInfo.twoD)
            OF << "  size_t gws[2] = { (size_t)mlc_n, (size_t)mlc_n };\n"
               << "  cl_int le = clEnqueueNDRangeKernel(queue, kernel, 2, 0, "
                  "gws, 0, 0, 0, 0);\n";
          else
            OF << "  size_t gws = (size_t)mlc_n;\n"
               << "  cl_int le = clEnqueueNDRangeKernel(queue, kernel, 1, 0, "
                  "&gws, 0, 0, 0, 0);\n";
          OF << "  clFinish(queue);\n"
             << "  if (le != 0) { std::fprintf(stderr, \"launch rc=%d\\n\", le); "
                "return 1; }\n"
             << "  double *h_out = new double[total];\n"
             << "  clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, "
                "total * sizeof(double), h_out, 0, 0, 0);\n"
             << "  double sum = 0;\n"
             << "  for (size_t i = 0; i < total; ++i) sum += h_out[i];\n"
             << "  std::printf(\"" << Stem << ": checksum = %.4f\\n\", sum);\n"
             << "  delete[] h_out;\n"
             << "  return 0;\n"
             << "}\n";
        }
      }
    }
    /* Makefile. */
    {
      std::string Path = OutDir + "/Makefile";
      std::ofstream OF(Path);
      OF << "# " << Path << "\n"
         << "# Generated by matlabc -emit-" << Target << "\n";
      if (Opts.Mode == Options::Mode::EmitCuda) {
        /* nvcc-free build: the host driver JIT-compiles the kernel via
         * NVRTC at runtime, so we only link the driver (libcuda) + NVRTC
         * (libnvrtc) — no CUDA Toolkit / nvcc required.  CUDA_INC /
         * CUDA_LIBS are overridable so a pip-wheel CUDA install (no
         * system .so symlinks) can be passed in:
         *   make CUDA_INC="-I/path/include" \
         *        CUDA_LIBS="/path/libnvrtc.so.12 /usr/lib/.../libcuda.so" */
        OF << "# nvcc-free: host driver JIT-compiles the kernel via NVRTC.\n\n"
           << "CXX      ?= g++\n"
           << "CUDA_INC ?=\n"
           << "CUDA_LIBS ?= -lnvrtc -lcuda\n\n"
           << "TARGET = " << Stem << "_" << Target << "\n"
           << "SRC    = " << Stem << "_main." << HostExt << "\n\n"
           << "$(TARGET): $(SRC) " << Stem << "_kernel." << KernelExt << "\n"
           << "\t$(CXX) -O2 -std=c++17 $(SRC) $(CUDA_INC) -o $(TARGET) "
              "$(CUDA_LIBS)\n\n"
           << "clean:\n"
           << "\trm -f $(TARGET)\n";
      } else if (Opts.Mode == Options::Mode::EmitOpenCL) {
        /* The host driver reads + builds the .cl at runtime via the ICD
         * loader, so it only needs libOpenCL.  No OpenCL SDK headers are
         * required (the driver hand-declares the API when none are
         * found).  OPENCL_LIB is overridable for ICD loaders shipped
         * without an unversioned .so symlink:
         *   make OPENCL_LIB="/lib/x86_64-linux-gnu/libOpenCL.so.1" */
        OF << "# SDK-free: host driver builds the .cl via the OpenCL ICD.\n\n"
           << "CXX        ?= g++\n"
           << "OPENCL_INC ?=\n"
           << "OPENCL_LIB ?= -lOpenCL\n\n"
           << "TARGET = " << Stem << "_" << Target << "\n"
           << "SRC    = " << Stem << "_main." << HostExt << "\n\n"
           << "$(TARGET): $(SRC) " << Stem << "_kernel." << KernelExt << "\n"
           << "\t$(CXX) -O2 -std=c++17 $(SRC) $(OPENCL_INC) -o $(TARGET) "
              "$(OPENCL_LIB)\n\n"
           << "clean:\n"
           << "\trm -f $(TARGET)\n";
      } else {
        OF << ToolchainComment << "\n\n"
           << "TARGET = " << Stem << "_" << Target << "\n"
           << "SRC    = " << Stem << "_main." << HostExt << "\n\n"
           << "$(TARGET): $(SRC) " << Stem << "_kernel." << KernelExt << "\n"
           << "\t" << CompileCmd << " $(SRC) -o $(TARGET)" << BuildExtras
           << "\n\n"
           << "clean:\n"
           << "\trm -f $(TARGET)\n";
      }
    }
    /* README — name the artifacts. */
    {
      std::ofstream OF(OutDir + "/README.md");
      OF << "# " << Stem << " — " << Target << " bundle\n\n"
         << "Generated by `matlabc -emit-" << Target << " "
         << Opts.InputPath << "`.\n\n"
         << "Build:\n```\nmake\n```\n\n"
         << "Files:\n"
         << "- `" << Stem << "_kernel." << KernelExt << "` — device kernel\n"
         << "- `" << Stem << "_main." << HostExt << "` — host driver\n"
         << "- `Makefile` — toolchain wiring\n";
    }
    std::cerr << "matlabc -emit-" << Target << ": wrote bundle to "
              << OutDir << "/\n";
    return 0;
  }
  if (Opts.Mode == Options::Mode::EmitMLIR ||
      Opts.Mode == Options::Mode::EmitLLVM ||
      Opts.Mode == Options::Mode::EmitC ||
      Opts.Mode == Options::Mode::EmitCpp ||
      Opts.Mode == Options::Mode::EmitPython ||
      Opts.Mode == Options::Mode::EmitTypeScript ||
      Opts.Mode == Options::Mode::EmitSystemVerilog ||
      Opts.Mode == Options::Mode::CheckSynthesizable ||
      Opts.Mode == Options::Mode::EmitHardwareReport) {
    mlirgen::Context MCtx;
    if (TU) {
      auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU, &SM,
                                    /*ReplMode=*/false,
                                    /*DebugMode=*/Opts.Debug);
      /* Sema-emitted errors (undefined name 'X', missing required
       * arg, etc.) get collected in Diag during MIR/MLIR generation.
       * Without this check, malformed IR with placeholder ops
       * (matlab.undef / matlab.call_indirect to a never-defined
       * symbol / matlab.subscript on a none-typed value) propagates
       * through the lowering passes and SIGSEGVs the
       * SCFToControlFlow / FuncToLLVM passes that try to introspect
       * operand types. Bail clean here. */
      if (Diag.hasErrors()) {
        Diag.printAll();
        return 1;
      }
      if (mlir::failed(mlir::verify(M))) {
        std::cerr << "error: MLIR verification failed after lowering\n";
        return 1;
      }
      // Opt/Run paths always clean up slots and scalars.
      bool WantFullPipeline = Opts.Mode == Options::Mode::EmitLLVM ||
                              Opts.Mode == Options::Mode::EmitC ||
                              Opts.Mode == Options::Mode::EmitCpp ||
                              Opts.Mode == Options::Mode::EmitPython ||
                              Opts.Mode == Options::Mode::EmitTypeScript ||
                              Opts.Mode == Options::Mode::EmitSystemVerilog ||
                              Opts.Mode == Options::Mode::CheckSynthesizable ||
                              Opts.Mode == Options::Mode::EmitHardwareReport;
      bool WantClean = Opts.Opt || WantFullPipeline;
      // `% hdl: port(...)` pragmas were originally an SV-only mechanism
      // for typing function-only files without a driver. The same
      // information is also useful to the C / C++ / Python /
      // TypeScript paths — without it, a function-only `.m` file
      // emits as a function with `none`-typed args that none of the
      // downstream emitters can lower (matlab.alloc survives). Run
      // the scan + apply for every WantFullPipeline mode; it's
      // idempotent and a no-op on files without `% hdl:` comments.
      // The SV-specific pragma surface (fsm_encoding, input_pipeline,
      // ...) is re-scanned further down in the SV branch.
      /* SV / hardware emit modes have their own integer-width inference
       * (HWBitWidthInfer infers i1/i16/i32 widths from chart annotations).
       * The PromoteNoneParams + PromoteBinopTypes f64-propagation lanes
       * for GPU validate against scalar-double MATLAB semantics — applying
       * them to chart-lowered .m files mis-promotes integer-typed signals
       * (l_airflow et al) to f64 and breaks HWLegalize.  Skip the f64
       * lanes entirely for HW-targeted compiles. */
      bool IsHwEmit =
          Opts.Mode == Options::Mode::EmitSystemVerilog ||
          Opts.Mode == Options::Mode::CheckSynthesizable ||
          Opts.Mode == Options::Mode::EmitHardwareReport;
      if (WantFullPipeline) {
        mlirgen::runScanHWPragmas(M, &SM);
        // Tier 5 — for a subsystem emit, ScanHWPragmas runs over the
        // .mflow JSON buffer and finds nothing (the synthesised AST
        // has no `% hdl:` comments). Stamp the port-type attributes
        // programmatically from the public inputs/outputs + the
        // `--fi-spec` overrides so `runApplyPortTypePragmas` (next
        // line) types the function args/return at the requested
        // fi format. Software targets (Python/C/C++/TS) don't go
        // through this branch — they're handled by emit-* via Tier 1.
        bool IsSV =
            Opts.Mode == Options::Mode::EmitSystemVerilog ||
            Opts.Mode == Options::Mode::CheckSynthesizable ||
            Opts.Mode == Options::Mode::EmitHardwareReport ||
            Opts.Mode == Options::Mode::EmitCocotb;
        if (!Opts.Subsystem.empty() && IsSV) {
          // Re-load the .mflow doc to get the public-port list. (Doc
          // pointer above is out of scope here — recompute.)
          SourceManager FlowSM3;
          DiagnosticEngine FlowDiag3(FlowSM3);
          auto Doc3 = matlab::flowchart::loadMflowFromPath(
              FlowSM3, Opts.InputPath, FlowDiag3);
          if (Doc3) {
            auto Meta = matlab::flowchart::describeSubsystem(
                *Doc3, Opts.Subsystem, FlowDiag3);
            if (Meta) {
              matlab::flowchart::FixedPointSpec Default{};
              std::map<std::string,
                       matlab::flowchart::FixedPointSpec> Overrides;
              // Replicate the CLI parse from §9740 — kept inline to
              // avoid a third copy of the same logic.
              for (const auto &S : Opts.FiSpecs) {
                auto Eq = S.find('=');
                if (Eq == std::string::npos) continue;
                std::string Name = S.substr(0, Eq);
                std::string Spec = S.substr(Eq + 1);
                matlab::flowchart::FixedPointSpec FS;
                size_t Cur = 0;
                if (Spec.size() > 1 &&
                    (Spec[0] == 'U' || Spec[0] == 'u') &&
                    (Spec[1] == 'Q' || Spec[1] == 'q')) {
                  FS.Signed = false; Cur = 2;
                } else if (!Spec.empty() &&
                           (Spec[0] == 'Q' || Spec[0] == 'q')) {
                  FS.Signed = true; Cur = 1;
                }
                auto Dot = Spec.find('.', Cur);
                if (Dot != std::string::npos) {
                  try {
                    FS.Width = std::stoi(Spec.substr(Cur, Dot - Cur));
                    FS.Frac  = std::stoi(Spec.substr(Dot + 1));
                  } catch (...) { continue; }
                }
                Overrides[Name] = FS;
              }
              bool HasReset = !Meta->StateArgNames.empty();
              matlab::flowchart::stampSubsystemPortPragmas(
                  M, Meta->Name, Meta->InputNames, Meta->OutputNames,
                  Default, Overrides, HasReset);
              // Tier-6 — nested subsystem helpers also need their
              // ports stamped so the SV pipeline can lower each
              // function's args / returns at the right fi width
              // (otherwise the call's return type stays `none` and
              // downstream fi-casts fall back to the constructor
              // path that doesn't synthesise). Walk every flow that
              // has signal_inport / signal_outport boundary ports
              // and stamp its hdl.ports too.
              for (const auto &OtherFlow : Doc3->Flows) {
                if (OtherFlow.Name == Opts.Subsystem) continue;
                bool HasBoundary = false;
                for (const auto &N : OtherFlow.Nodes) {
                  if (N.Kind == "signal_inport" ||
                      N.Kind == "signal_outport") {
                    HasBoundary = true;
                    break;
                  }
                }
                if (!HasBoundary) continue;
                auto OMeta = matlab::flowchart::describeSubsystem(
                    *Doc3, OtherFlow.Name, FlowDiag3);
                if (!OMeta) continue;
                bool ORst = !OMeta->StateArgNames.empty();
                matlab::flowchart::stampSubsystemPortPragmas(
                    M, OMeta->Name, OMeta->InputNames,
                    OMeta->OutputNames, Default, Overrides, ORst);
              }
            }
          }
        }
        if (!mlirgen::runApplyPortTypePragmas(M)) {
          Diag.printAll();
          return 1;
        }
        /* PromoteNoneParams runs AFTER ApplyPortTypePragmas so any HDL
         * function whose `% hdl: port(x, 'int16')` pragma typed its arg
         * to i16 / i8 / i32 is already non-none and my pass skips it
         * (the AnyNone check returns false).  Plain functions (GPU
         * PCT et al.) without port pragmas keep `none` args, which my
         * pass promotes to f64.
         *
         * Must still run BEFORE SlotPromotion (next, inside WantClean)
         * which collapses the matlab.store(%arg, %slot) anchor my
         * detector uses.  HW-emit modes skip the whole f64 lane
         * because their integer types come from the pragma + HW
         * width-infer passes. */
        if (!IsHwEmit) mlirgen::runPromoteNoneParams(M);
        // Seed slot/load types from the now-typed entry-block args
        // BEFORE SlotPromotion runs in WantClean. SlotPromotion
        // only fires when the value type matches the load result
        // type; without this RefineSlotTypes pass, all slots stay
        // `none`-typed and the body never gets concretely typed
        // (LowerUserCalls only runs propagateScalarTypes on funcs
        // with active matlab.call sites, which a no-caller bare
        // function lacks).
        mlirgen::runRefineSlotTypes(M);
      }
      if (WantClean) {
        mlirgen::runSlotPromotion(M);
        // See docs/emit_fixed_point.md — fi ops must lower before arith.
        mlirgen::runLowerFixedPoint(M);
        mlirgen::runLowerScalarsToArith(M);
        mlirgen::runSlotPromotion(M);
        // Patch func.func signatures from refined return-op types so
        // the verifier doesn't trip on `make_handle("false") →
        // arith.constant : i1` rewrites whose function still
        // declares `-> none`. Idempotent.
        mlirgen::runRefineFuncSigs(M);
        if (mlir::failed(mlir::verify(M))) {
          std::cerr << "error: MLIR verification failed after passes\n";
          return 1;
        }
      }
      if (WantFullPipeline) {
        // Outline parfor first — that way the induction variable flows as a
        // direct block argument (f64) into disp/fprintf rather than via an
        // outer slot that would still be `none`-typed at LowerIO time.
        if (!IsHwEmit) {
          mlirgen::runPromoteNoneParams(M);
          for (int Iter = 0; Iter < 4; ++Iter)
            if (!mlirgen::runPromoteBinopTypes(M)) break;
        }
        /* Forward outer-scope literal captures into parfor bodies
         * before the outliner — closes the matlab.alloc-capture
         * rejection from issue #20 for the common literal case. */
        mlirgen::runForwardParforCaptures(M);
        mlirgen::runOutlineParfor(M);
        mlirgen::runOutlineGpuKernels(M);
        // Lower sequential matlab.for / matlab.while into scf.while so
        // the MLIR conversion pipeline can finish translation. Must run
        // before LowerTensorOps (which would erase the matlab.range
        // producer the for-lowering relies on) and after OutlineParfor
        // (which consumes matlab.parfor).
        mlirgen::runLowerSeqLoops(M);
        // Outline anonymous-function bodies into llvm.funcs so their
        // handles become plain function pointers and call_indirect sites
        // collapse to direct llvm.calls.
        mlirgen::runLowerAnonCalls(M);
        // Iterate scalar-to-arith + user-call lowering to a fixpoint so
        // type refinement propagates across chained user calls. Each
        // iteration: LowerScalarsToArith folds scalar ops that became
        // matchable after previous arg/result retyping; LowerUserCalls
        // refines func.func signatures from call-site types and converts
        // matlab.call -> func.call only where operand types now match.
        // Bounded iteration count protects against pathological loops.
        for (int Iter = 0; Iter < 8; ++Iter) {
          bool A = mlirgen::runLowerScalarsToArith(M);
          bool B = mlirgen::runLowerUserCalls(M);
          if (!A && !B) break;
        }
        // A param-bound for-loop inside a function (`for k = 1:n`, n a
        // parameter) couldn't lower in the first runLowerSeqLoops above: the
        // param — hence the range bound — was still `none`-typed then. The
        // scalar/user-call fixpoint just refined the param to f64; refine
        // the param's slot/loads to match and re-run seq-loop lowering so
        // the f64 range bound now extracts. Must stay before LowerTensorOps
        // (which consumes the matlab.range producer). Idempotent: already-
        // lowered loops are gone, so this only catches the deferred ones.
        mlirgen::runRefineSlotTypes(M);
        mlirgen::runLowerSeqLoops(M);
        // Lower every tensor-producing matlab.* op to a runtime call
        // against the matrix runtime (matlab_zeros / matlab_add_mm /
        // matlab_transpose / ...). After this runs, matrix values in the
        // IR are !llvm.ptr to heap-allocated matlab_mat descriptors, and
        // disp on a matrix ptr routes to matlab_disp_mat.
        mlirgen::runLowerTensorOps(M);
        /* After LowerTensorOps has retyped any slots whose stores are
         * ptr-typed (class-instance slots, cell / struct slots), the
         * call-site loads feeding into user-method calls change type
         * from `none` to `ptr`. Re-run the scalar+user-call fixpoint
         * so the method-call matlab.call sites now match their
         * func.func signatures and get converted to func.call. */
        for (int Iter = 0; Iter < 4; ++Iter) {
          bool A = mlirgen::runLowerScalarsToArith(M);
          bool B = mlirgen::runLowerUserCalls(M);
          if (!A && !B) break;
        }
        /* Propagate the freshly-typed call results through binops +
         * slot chains so a chained `gather(a .* x + b)` pattern routes
         * correctly.  Iterate to fixpoint.  Skipped for HW emit modes
         * — see comment at the EARLY invocation site for rationale. */
        if (!IsHwEmit) {
          for (int Iter = 0; Iter < 4; ++Iter) {
            bool Pb = mlirgen::runPromoteBinopTypes(M);
            mlirgen::runRefineSlotTypes(M);
            if (!Pb) break;
          }
        }
        mlirgen::runLowerTensorOps(M);
        // LATE GPU outline (issue #24): any matlab.gpu.kernel the early
        // runOutlineGpuKernels CLAIMED (MATLAB_GPU_OUTLINE=1) was left in
        // place — array captures are now `ptr` to matlab_mat and scalar
        // slots are `llvm.alloca`, so the body lifts into a standalone
        // llvm.func with plain pointer/scalar state (no tensor↔ptr cast).
        // The lowering passes above do NOT descend into the kernel region
        // (an unregistered op), so the lifted body can still hold
        // matlab.for / matlab.load / matlab.store ops; re-run the seq-loop
        // + tensor + scalar fixpoint so they lower now that the body lives
        // in a real func.
        if (mlirgen::runOutlineGpuKernelsLate(M)) {
          mlirgen::runRefineSlotTypes(M);
          mlirgen::runLowerSeqLoops(M);
          mlirgen::runLowerTensorOps(M);
          for (int Iter = 0; Iter < 4; ++Iter) {
            bool A = mlirgen::runLowerScalarsToArith(M);
            bool B = mlirgen::runLowerUserCalls(M);
            if (!A && !B) break;
          }
          mlirgen::runLowerTensorOps(M);
        }
        // Second LowerFixedPoint sweep — picks up matlab.call_builtin
        // @matlab_mat_*_slice1 / _concat_row sites that needed their
        // tensor operand retyped to ptr by LowerTensorOps first.
        mlirgen::runLowerFixedPoint(M);
        // Second-chance anon call rewrite: any matlab.call_indirect that
        // survived the first LowerAnonCalls because its matrix operands
        // were still tensor-typed can now match the outlined function's
        // (ptr, ...) signature after LowerTensorOps retyped the slots.
        if (mlirgen::runLowerAnonCallsPost(M)) {
          // The newly-lowered llvm.call producing a ptr may now be the
          // operand of an un-lowered matlab.call_builtin @disp (etc.).
          // Re-run LowerTensorOps so disp(ptr) routes to matlab_disp_mat.
          mlirgen::runLowerTensorOps(M);
        }
        // Multi-callsite monomorphisation. Phase 5 of #38 moved
        // scalar-polymorphic helpers to a Sema-time clone (each call
        // site already dispatches to its own per-type specialisation
        // by this point), but the late pass is still load-bearing for:
        //   - matrix-typed call sites (Sema-mono defers ptr-shape sigs
        //     until LowerTensorOps materialises tensor literals);
        //   - arity-varying callees (nargin per-arity clones, e.g.
        //     `add2(5, 7)` + `add2(5)` against the same body);
        //   - varargin / varargout (per-arity bucket clones the cell
        //     pack/unpack shape).
        // Sema-mono's growPlan skips these classes explicitly, so the
        // pass below sees a strictly smaller workload than before
        // Phase 5 landed (most run-tests fixtures pass through this
        // call without any cloning) but it's NOT redundant — dropping
        // it regresses fn_polymorphic_invariant + fn_nargin_callsite +
        // varargout_basic. The Phase 6 cleanup that retires this call
        // entirely needs the matrix-ptr and arity-varying classes
        // absorbed Sema-side first; documented as a follow-up.
        // #191 P5 scaffolding: MATLAB_LLVM_NO_LATE_MONO=1 bypasses it.
        if (!std::getenv("MATLAB_LLVM_NO_LATE_MONO") &&
            mlirgen::runMonomorphiseUserCalls(M)) {
          for (int Iter = 0; Iter < 4; ++Iter) {
            bool A = mlirgen::runLowerScalarsToArith(M);
            bool B = mlirgen::runLowerUserCalls(M);
            if (!A && !B) break;
          }
          mlirgen::runLowerTensorOps(M);
          // Final sweep: refresh each func.func's signature from the
          // types that actually flow through its func.return. Needed
          // because LowerTensorOps rewrote the body but didn't touch
          // the enclosing function's return type.
          M.walk([&](mlir::func::FuncOp Fn) {
            if (Fn.empty()) return;
            llvm::SmallVector<mlir::Type, 4> NewResults(
                Fn.getFunctionType().getResults().begin(),
                Fn.getFunctionType().getResults().end());
            bool Changed = false;
            Fn.walk([&](mlir::func::ReturnOp Ret) {
              if (Ret.getNumOperands() != NewResults.size()) return;
              for (unsigned i = 0; i < Ret.getNumOperands(); ++i) {
                auto Old = NewResults[i];
                auto New = Ret.getOperand(i).getType();
                if (mlir::isa<mlir::NoneType>(Old) && Old != New) {
                  NewResults[i] = New;
                  Changed = true;
                }
              }
            });
            if (Changed) {
              auto Ty = mlir::FunctionType::get(
                  Fn.getContext(),
                  Fn.getFunctionType().getInputs(), NewResults);
              Fn.setFunctionType(Ty);
            }
          });
          // Stale func.call ops need their result types patched too.
          M.walk([&](mlir::func::CallOp Call) {
            auto Tgt = M.lookupSymbol<mlir::func::FuncOp>(
                Call.getCallee());
            if (!Tgt) return;
            auto SigR = Tgt.getFunctionType().getResults();
            if (Call.getNumResults() != SigR.size()) return;
            bool Mismatch = false;
            for (unsigned i = 0; i < SigR.size(); ++i)
              if (Call.getResult(i).getType() != SigR[i]) {
                Mismatch = true; break;
              }
            if (!Mismatch) return;
            mlir::OpBuilder CB(Call);
            auto Nc = mlir::func::CallOp::create(CB, Call.getLoc(),
                                                  SigR, Call.getCallee(),
                                                  Call.getOperands());
            for (unsigned i = 0; i < SigR.size(); ++i)
              Call.getResult(i).replaceAllUsesWith(Nc.getResult(i));
            Call.erase();
          });
          // After patching call results, any disp(ptr) sites that were
          // previously fed by a none-typed func.call now see a ptr
          // operand and need LowerTensorOps's matlab_disp_mat dispatch.
          mlirgen::runLowerTensorOps(M);
        }
        // Lower matlab.nargin / matlab.nargout placeholders to
        // arith.constant. Runs AFTER the monomorphiser so per-arity
        // clones see their own call-site arity rather than the
        // function's declared arity.
        mlirgen::runLowerNarginNargout(M);
        // Phase 4.5.1: refine `none`-typed `matlab.alloc` slots whose
        // every store agrees on a concrete scalar type. Must run
        // BEFORE LowerScalarSlots so the just-retyped slots get
        // promoted to llvm.alloca on the same pass.
        mlirgen::runRefineSlotTypes(M);
        // Phase 4.5.4: rewrite `fi(zeros(1, N), ...)` runtime-call
        // chains into stack-allocated `llvm.alloca <[N x iW]>` with
        // GEP + load/store access. Must also run before
        // LowerScalarSlots so the slot wrapping the array pointer
        // is erased.
        mlirgen::runLowerStaticFiArrays(M);
        // Patch func.func signatures from the refined return types.
        mlirgen::runRefineFuncSigs(M);
        // Matrix-returning user functions: RefineFuncSigs above is the only
        // pass that settles their `tensor -> ptr` result type (LowerUserCalls'
        // canRefineTo refuses an already-concrete tensor result, so the
        // func.call stayed tensor through the fixpoint loops). Now that the
        // call result is ptr, re-run LowerTensorOps so the caller's slot —
        // fed by that call — is retyped to ptr and its `A(i,j[,k])` / `A+1`
        // matrix uses finally lower (otherwise they survive as un-lowered
        // matlab.subscript / matlab.add). Iterate with RefineFuncSigs so
        // chained matrix-returning calls converge. Mirrors the REPL/JIT path.
        //
        // HDL lanes are excluded: this extra LowerTensorOps round over-lowers
        // a matrix concat like `[fi(x), delay_line(1:3)]` into a runtime
        // matlab_mat_from_scalar call, which the SystemVerilog backend expects
        // to still be in array form (an unpacked-array shift). The
        // matrix-returning-function fix is a software/execute concern only.
        bool HwLane =
            Opts.Mode == Options::Mode::EmitSystemVerilog ||
            Opts.Mode == Options::Mode::CheckSynthesizable ||
            Opts.Mode == Options::Mode::EmitHardwareReport ||
            Opts.Mode == Options::Mode::EmitCocotb ||
            Opts.Mode == Options::Mode::EmitFiReport;
        if (!HwLane) {
          for (int Iter = 0; Iter < 4; ++Iter) {
            bool Changed = mlirgen::runLowerTensorOps(M);
            mlirgen::runRefineFuncSigs(M);
            if (!Changed) break;
          }
        }
        // #148: resolve verifier-placeholder unrealized_conversion_cast on
        // scf.if conditions (e.g. `if contains(a,b)` / `if strcmp(a,b)` —
        // `none`-typed at lowering, refined to f64 by the loop above) before
        // any emitter sees them. Without this the cast survives into the
        // emit-c/cpp/python/typescript backends as an "unsupported op" and
        // into emit-llvm translation as a failure. Idempotent.
        mlirgen::runRefineIfConds(M);
        // After user-call refinement, any surviving matlab.alloc whose
        // result type is now a scalar primitive can be promoted to
        // llvm.alloca. This catches function-body locals that weren't
        // promoted by SlotPromotion (because they're used across blocks).
        mlirgen::runLowerScalarSlots(M);
#ifdef MATLAB_LLVM_WITH_PLOT
        mlirgen::runLowerPlot(M);
#endif
  mlirgen::runLowerIO(M);
        if (Opts.Mode == Options::Mode::EmitC ||
            Opts.Mode == Options::Mode::EmitCpp ||
            Opts.Mode == Options::Mode::EmitPython ||
            Opts.Mode == Options::Mode::EmitTypeScript) {
          // For HDL files that use a `persistent` fi-array (e.g.
          // `persistent delay_line; ...; delay_line = [x, delay_line(1:3)]`),
          // the runtime ptr semantics survive into the C/Python/TS
          // backends. Reuse Stage F (originally written for SV) to
          // rewrite the array into N parallel scalar persistents,
          // each handled cleanly by the existing per-emitter
          // `matlab_global_{get,set}_f64 + persistent_name` paths.
          // Bit-identical output to the SV version, which matches
          // what a CocoTB harness needs for its reference model.
          // The same per-element __subscript_store sites the SV
          // path consumes are also rewritten — without it, those
          // ops would survive untranslated into the emitter.
          mlirgen::runHWUnrollFor(M);
          mlirgen::runLowerPersistentFiArrays(M);
          mlirgen::runRefineSlotTypes(M);
          mlirgen::runLowerScalarSlots(M);
          // DCE dead `matlab.alloc` chains. Stage F's rewrite (and
          // some lowering shapes) can leave a matlab.alloc whose
          // only users are matlab.store / matlab.load with all load
          // results unused — a synthetic slot wrapping a return port
          // that's also written through a separate llvm.alloca.
          // LowerScalarSlots refuses to promote these when the
          // store's value type drifts from the alloc's elem type
          // (e.g. f64 stored into an iN slot). Erase them outright
          // so the emitter doesn't trip on the surviving matlab.alloc.
          {
            llvm::SmallVector<mlir::Operation *, 8> Dead;
            M.walk([&](mlir::Operation *Op) {
              if (Op->getName().getStringRef() != "matlab.alloc") return;
              if (Op->getNumResults() != 1) return;
              for (mlir::Operation *U : Op->getResult(0).getUsers()) {
                auto N = U->getName().getStringRef();
                if (N != "matlab.store" && N != "matlab.load") return;
                if (N == "matlab.load" && !U->getResult(0).use_empty())
                  return;
              }
              Dead.push_back(Op);
            });
            for (mlir::Operation *Op : Dead) {
              llvm::SmallVector<mlir::Operation *, 4> Users;
              for (mlir::Operation *U : Op->getResult(0).getUsers())
                Users.push_back(U);
              for (mlir::Operation *U : Users) U->erase();
              Op->erase();
            }
          }
          // Stage F doesn't catch every persistent fi-array shape
          // (e.g. fir_asic_pipelined's `reg_products(i) = ...` write-
          // through chain when the array isn't an isempty-initialised
          // shift register). Any surviving `matlab.call_builtin
          // @__subscript_store(arr, idx_f64, val_intN)` site here is
          // a runtime-managed array element write — lower it to
          // `llvm.call @matlab_mat_{i,u}64_set1_s(arr, idx_f64,
          // sext/zext val to i64)` so the C / Python / TS backends
          // emit a concrete runtime call rather than tripping the
          // unsupported-op fallback. The SV path doesn't need this
          // (it never gets here — its branch handles Stage F's
          // misses through HWLegalize's diagnostic).
          {
            auto &Ctx = *M.getContext();
            auto F64 = mlir::Float64Type::get(&Ctx);
            auto I64 = mlir::IntegerType::get(&Ctx, 64);
            auto VoidTy = mlir::LLVM::LLVMVoidType::get(&Ctx);
            auto PtrTy = mlir::LLVM::LLVMPointerType::get(&Ctx);
            auto getOrInsert = [&](llvm::StringRef Name)
                -> mlir::LLVM::LLVMFuncOp {
              if (auto E = M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(Name))
                return E;
              mlir::OpBuilder MB(&Ctx);
              MB.setInsertionPointToStart(M.getBody());
              auto Ty = mlir::LLVM::LLVMFunctionType::get(
                  VoidTy, {PtrTy, F64, I64});
              auto Fn = mlir::LLVM::LLVMFuncOp::create(
                  MB, M.getLoc(), Name, Ty);
              Fn.setLinkage(mlir::LLVM::Linkage::External);
              return Fn;
            };
            llvm::SmallVector<mlir::Operation *, 8> Survivors;
            M.walk([&](mlir::Operation *Op) {
              if (Op->getName().getStringRef() != "matlab.call_builtin")
                return;
              auto C = Op->getAttrOfType<mlir::StringAttr>("callee");
              if (!C || C.getValue() != "__subscript_store") return;
              if (Op->getNumOperands() != 3) return;
              if (Op->getOperand(0).getType() != PtrTy) return;
              if (Op->getOperand(1).getType() != F64) return;
              if (!mlir::isa<mlir::IntegerType>(
                      Op->getOperand(2).getType())) return;
              Survivors.push_back(Op);
            });
            for (mlir::Operation *Op : Survivors) {
              mlir::Value Base = Op->getOperand(0);
              mlir::Value Idx  = Op->getOperand(1);
              mlir::Value Val  = Op->getOperand(2);
              auto VIT = mlir::cast<mlir::IntegerType>(Val.getType());
              bool Signed = true;
              if (auto SA = Op->getAttrOfType<mlir::IntegerAttr>(
                      "fi_signed"))
                Signed = SA.getInt() != 0;
              mlir::OpBuilder OB(Op);
              mlir::Value V64 = Val;
              if (VIT.getWidth() < 64) {
                V64 = Signed
                    ? (mlir::Value)mlir::arith::ExtSIOp::create(
                          OB, Op->getLoc(), I64, Val)
                    : (mlir::Value)mlir::arith::ExtUIOp::create(
                          OB, Op->getLoc(), I64, Val);
              } else if (VIT.getWidth() > 64) {
                V64 = mlir::arith::TruncIOp::create(
                    OB, Op->getLoc(), I64, Val);
              }
              auto Fn = getOrInsert(Signed ? "matlab_mat_i64_set1_s"
                                           : "matlab_mat_u64_set1_s");
              mlir::LLVM::CallOp::create(
                  OB, Op->getLoc(), Fn,
                  mlir::ValueRange{Base, Idx, V64});
              Op->erase();
            }
          }
          // Fold `if/else/store-to-same-slot` into `arith.select` first,
          // then squash single-store allocas back into SSA so the emitted
          // C doesn't drag a `T slot = 0; void* p = &slot;` prelude for
          // every parameter spill / function-local constant. Keeps the
          // LLVM path untouched (it has its own mem2reg on the backend).
          mlirgen::runIfStoreToSelect(M);
          mlirgen::runMem2RegLite(M);
          // Final signature catch-up: Mem2RegLite / IfStoreToSelect
          // can promote slots and rewrite arms in ways that retype
          // call-site operands. RefineFuncSigs's input-side
          // refinement (step 0) catches the leftover none → typed
          // gap so the verifier doesn't reject a stale func.call.
          mlirgen::runRefineFuncSigs(M);
          if (getenv("DUMP_BEFORE_C")) mlirgen::printModule(std::cerr, M);
          // Verify the module right before emission so a malformed IR
          // state is surfaced with a clear error rather than as a cryptic
          // cc/c++ compile failure on the emitted source.
          if (mlir::failed(mlir::verify(M))) {
            std::cerr
                << "error: MLIR verification failed before C emission\n";
            return 1;
          }
          std::string Src;
          if (Opts.Mode == Options::Mode::EmitPython) {
            Src = mlirgen::emitPython(M, Opts.NoLine, &SM);
          } else if (Opts.Mode == Options::Mode::EmitTypeScript) {
            Src = mlirgen::emitTypeScript(M, Opts.NoLine, &SM);
          } else {
            /* C / C++ default to suppressing `#line`. `-line` opts back
             * in; `-no-line` is the (now-redundant) explicit form of
             * the default. Both flags together is harmless — line
             * directives are emitted only when EmitLine is set. */
            bool NoLineForC = !Opts.EmitLine;
            Src = mlirgen::emitC(
                M, Opts.Mode == Options::Mode::EmitCpp, NoLineForC,
                Opts.Doxygen, Opts.CppAuto, &SM);
          }
          if (Src.empty()) return 1;
          // Embedded Coder, Tier 2 — append a per-target class
          // wrapper around the functional `step(...)` when the
          // user is emitting a `.mflow` subsystem AND opted into
          // the (default) `class` state form. Skipped for
          // `--state-form=function` and for non-subsystem emits.
          if (!Opts.Subsystem.empty() && Opts.StateForm == "class") {
            std::string TargetKey;
            switch (Opts.Mode) {
              case Options::Mode::EmitPython:     TargetKey = "python"; break;
              case Options::Mode::EmitCpp:        TargetKey = "cpp"; break;
              case Options::Mode::EmitC:          TargetKey = "c"; break;
              case Options::Mode::EmitTypeScript: TargetKey = "typescript"; break;
              default: break;
            }
            if (!TargetKey.empty()) {
              SourceManager FlowSM2;
              DiagnosticEngine FlowDiag2(FlowSM2);
              auto Doc2 = matlab::flowchart::loadMflowFromPath(
                  FlowSM2, Opts.InputPath, FlowDiag2);
              if (Doc2) {
                auto Meta = matlab::flowchart::describeSubsystem(
                    *Doc2, Opts.Subsystem, FlowDiag2);
                if (Meta) {
                  Src += matlab::flowchart::emitSubsystemClassWrapper(
                      *Meta, TargetKey);
                }
              }
            }
          }
          std::cout << Src;
        } else if (Opts.Mode == Options::Mode::EmitSystemVerilog ||
                   Opts.Mode == Options::Mode::CheckSynthesizable ||
                   Opts.Mode == Options::Mode::EmitHardwareReport) {
          // Phase 4 v2.6: scan `% hdl: <directive>(<args>)`
          // pragmas inside each user function and attach as
          // string attributes on the func.func. The SV emitter
          // checks them for per-function overrides (e.g.
          // `hdl.fsm_encoding` overrides the CLI-wide
          // `-sv-fsm-encoding` flag).
          mlirgen::runScanHWPragmas(M, &SM);

          // Pre-HWStateInfer normalization: split `if isempty(c) ||
          // X ... end` into the canonical two-guard form
          // (`if isempty(c)` + `if X`, both cloned bodies) so the
          // HWStateInfer matcher's single-use-isempty constraint
          // accepts the literal HDL Coder mealy/moore idiom.
          mlirgen::runSplitIsEmptyOr(M);

          // Phase 5.6 Stage F.2: unroll constant-bound canonical
          // for-loops at the IR level. Stage F's per-element
          // persistent-fi-array rewrite needs constant subscript
          // indices on every read; without IR-level unrolling
          // the body of `for i = 1:N; arr(i) ...; end` keeps
          // the f64 iv as the subscript index and Stage F bails.
          mlirgen::runHWUnrollFor(M);

          // Phase 5.6 Stage F: lower persistent fi-array shift-
          // register patterns into N parallel scalar persistents.
          // Runs after `LowerStaticFiArrays` (so the next-cycle
          // pointer is a static `llvm.alloca [N x iW]`) but before
          // `HWStateInfer` so the synthetic scalar persistents
          // surface as recognized state.
          mlirgen::runLowerPersistentFiArrays(M);
          // Stage F's rewrite can leave behind a matlab.alloc
          // slot (e.g. `y` in `y = reg_output`) that wasn't
          // around to be promoted by the earlier
          // LowerScalarSlots pass. Re-run RefineSlotTypes +
          // LowerScalarSlots + Mem2RegLite so those slots end
          // up as llvm.alloca / get folded out.
          mlirgen::runRefineSlotTypes(M);
          // RefineSlotTypes' HW-aware second pass may retype a
          // slot from f64 to iN (the typed register width); the
          // function's result type and any func.return need to
          // pick up the change. Re-run RefineFuncSigs to keep
          // signatures consistent before LowerScalarSlots
          // materializes the alloca.
          mlirgen::runRefineFuncSigs(M);
          // Re-run a slot-type / scalar-lowering fixpoint after
          // Stage F. Each iteration: refine slots from typed
          // stores, run scalar lowering (which now sees newly-
          // typed loads), refresh slots again. Without this
          // interleave, `matlab.call_builtin @bitxor (i8, none)`
          // stays a runtime call because the second operand's
          // load was still `none`-typed during LSE2A's first
          // pass. Bounded — convergence is usually 2 rounds.
          for (int Iter = 0; Iter < 4; ++Iter) {
            bool A = mlirgen::runLowerScalarsToArith(M);
            bool B = mlirgen::runRefineSlotTypes(M);
            if (!A && !B) break;
          }
          mlirgen::runRefineFuncSigs(M);
          mlirgen::runLowerScalarSlots(M);
          mlirgen::runMem2RegLite(M);
          // Tier-5f — unify mixed-width integer stores into the
          // same matlab.alloc slot (the saturation+persistent
          // pattern: i32 rails + i64 passthrough). Sign-extends
          // narrower stores and retypes the slot's loads so
          // HWLegalize sees a consistent width.  Idempotent on
          // already-uniform slots.
          if (mlirgen::runUnifyMixedWidthStores(M)) {
            // The unified slot now has a concrete integer type;
            // LowerScalarSlots can lower it to llvm.alloca on a
            // re-run.  Then Mem2RegLite picks up the alloca that
            // becomes a single-writer scalar.
            mlirgen::runLowerScalarSlots(M);
            mlirgen::runMem2RegLite(M);
          }
          if (getenv("DUMP_AFTER_F")) mlirgen::printModule(std::cerr, M);

          // Phase 5.1: replace runtime-call `matlab_fi_sat_s64` /
          // `_u64` saturate helpers with explicit clamp circuits
          // (cmpi + select chain). Earlier the SV pipeline DCE'd
          // these via passthrough, which was correct only for
          // Wrap-mode fi; the explicit clamp gives correct
          // Saturate semantics regardless and synthesizes to a
          // small comparator + 2-way mux per bound.
          mlirgen::runLowerFiSaturate(M);

          // Phase 5.4: rewrite constant-coefficient multiplications
          // to shift-add trees (`x*7 → (x<<3) - x`). Default-on for
          // the SV pipeline; `-sv-const-mul=off` disables. Runs only
          // for SV emit / report / check-synth — other backends
          // emit `*` directly to match user-side semantics.
          if (Opts.SvConstMulOpt) mlirgen::runConstMulCSD(M);

          // Phase 4.5.2: replace any `unrealized_conversion_cast`
          // placeholder on scf.if conditions (inserted at MIR-to-MLIR
          // lowering when the cond was `none`-typed) with a real
          // `arith.cmpi ne` / `arith.cmpf one` against zero, now
          // that operand types have refined.
          mlirgen::runRefineIfConds(M);

          // Same pre-emit cleanup as EmitC: fold `if/else` stores into
          // `arith.select` and promote single-store allocas. Required so
          // scalar combinational programs surface to the SV emitter as
          // pure dataflow rather than a load/store dance.
          mlirgen::runIfStoreToSelect(M);
          mlirgen::runMem2RegLite(M);
          if (mlir::failed(mlir::verify(M))) {
            std::cerr
                << "error: MLIR verification failed before SV emission\n";
            return 1;
          }
          // Synthesizability gate. Runs in both `-emit-systemverilog` and
          // `-check-synthesizable` modes — emission never silently
          // produces broken RTL. See docs/emit_systemverilog.md.
          bool Ok = mlirgen::runHWLegalize(M, &SM);
          if (Ok) Ok = mlirgen::runHWBitWidthInfer(M, &SM);
          if (Opts.Mode == Options::Mode::CheckSynthesizable) {
            // Also run the SV emitter in dry-run mode so FSM-time
            // diagnostics (Phase 4 v2.3 ambiguity checks) fire
            // alongside HWLegalize's gate. Discard the rendered
            // SV — `-check-synthesizable` writes no stdout. The
            // dry-run still has to materialize the string because
            // the emitter's gather step is integral to its run().
            if (Ok) {
              std::string Dry = mlirgen::emitSystemVerilog(
                  M, &SM, mlirgen::HWResetKind::AsyncLow,
                  mlirgen::HWFSMEncoding::Binary);
              if (Dry.empty()) Ok = false;
            }
            Diag.printAll();
            return Ok ? 0 : 1;
          }
          if (Opts.Mode == Options::Mode::EmitHardwareReport) {
            // Phase 5.5 — emit a Markdown summary of the post-
            // pipeline IR's resource shape. Same gate as
            // `-emit-systemverilog`, then walk the module and
            // print operator counts / register info / FSM info.
            if (!Ok) {
              Diag.printAll();
              return 1;
            }
            mlirgen::emitHardwareReport(M, std::cout, &SM);
            Diag.printAll();
            return 0;
          }
          if (!Ok) {
            Diag.printAll();
            return 1;
          }
          mlirgen::HWResetKind R = mlirgen::HWResetKind::AsyncLow;
          switch (Opts.SvReset) {
          case Options::SvResetKind::AsyncLow:
            R = mlirgen::HWResetKind::AsyncLow; break;
          case Options::SvResetKind::SyncHigh:
            R = mlirgen::HWResetKind::SyncHigh; break;
          case Options::SvResetKind::SyncLow:
            R = mlirgen::HWResetKind::SyncLow; break;
          }
          mlirgen::HWFSMEncoding FE = mlirgen::HWFSMEncoding::Binary;
          switch (Opts.SvFSMEnc) {
          case Options::SvFSMEncoding::Binary:
            FE = mlirgen::HWFSMEncoding::Binary; break;
          case Options::SvFSMEncoding::OneHot:
            FE = mlirgen::HWFSMEncoding::OneHot; break;
          case Options::SvFSMEncoding::Gray:
            FE = mlirgen::HWFSMEncoding::Gray; break;
          }
          std::string Src = mlirgen::emitSystemVerilog(M, &SM, R, FE);
          if (Src.empty()) return 1;
          std::cout << Src;
        } else {
          /* `-g` on the -emit-llvm path turns on DWARF emission so the
           * resulting LLVM IR carries `!dbg` metadata. clang's downstream
           * codegen turns those into a DWARF section, and lldb / gdb can
           * then step through the original `.m` source after compiling
           * the IR with `clang -x ir -g foo.ll -o foo`. */
          std::string LL = mlirgen::lowerToLLVMIR(M, Opts.Debug);
          if (LL.empty()) return 1;
          std::cout << LL;
        }
      } else {
        mlirgen::printModule(std::cout, M);
      }
    }
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }
#endif

  Diag.printAll();
  return Diag.hasErrors() ? 1 : 0;
}
