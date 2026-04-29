#include "matlab/AST/AST.h"
#include "matlab/AST/ASTDumper.h"
#include "matlab/AST/Formatter.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Flowchart/GraphToAST.h"
#include "matlab/Flowchart/Loader.h"
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#endif
#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/SemaDumper.h"
#include "matlab/Sema/Scope.h"
#include "matlab/Sema/Type.h"
#include "matlab/Sema/TypeInference.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

namespace {
struct Options {
  enum class Mode { DumpTokens, DumpAST, EmitSema, EmitMIR, EmitMLIR,
                    EmitLLVM, EmitC, EmitCpp, EmitPython, EmitTypeScript,
                    EmitFiReport, EmitSystemVerilog, CheckSynthesizable,
                    EmitHardwareReport,
                    DumpFlow, EmitMatlab,
                    Check, Repl, Format, Dap };
  Mode Mode = Mode::Check;
  bool Opt = false;
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
  /* Block-library search path for `.mflow` custom blocks (Phase 4b).
   * Resolution order: command-line `--block-path DIR` entries (in CLI
   * order) followed by colon-separated entries from the
   * `MATFORGE_BLOCK_PATH` environment variable. The first directory
   * that contains a matching `.m` file wins. Ignored for non-`.mflow`
   * inputs. */
  std::vector<std::string> BlockPath;
};

int usage(const char *Prog) {
  std::cerr << "usage: " << Prog
            << " [-dump-tokens | -dump-ast | -emit-sema | -emit-mir |\n"
               "             -emit-mlir | -emit-llvm | -emit-c | -emit-cpp |\n"
               "             -emit-python | -emit-typescript |\n"
               "             -emit-systemverilog | -check-synthesizable |\n"
               "             -emit-hardware-report | -dump-flow |\n"
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
    else if (A == "-emit-mir") Opts.Mode = Options::Mode::EmitMIR;
    else if (A == "-emit-mlir") Opts.Mode = Options::Mode::EmitMLIR;
    else if (A == "-emit-llvm") Opts.Mode = Options::Mode::EmitLLVM;
    else if (A == "-emit-c") Opts.Mode = Options::Mode::EmitC;
    else if (A == "-emit-cpp") Opts.Mode = Options::Mode::EmitCpp;
    else if (A == "-emit-python") Opts.Mode = Options::Mode::EmitPython;
    else if (A == "-emit-typescript" || A == "-emit-ts")
      Opts.Mode = Options::Mode::EmitTypeScript;
    else if (A == "-emit-fixed-point-report" || A == "-emit-fi-report")
      Opts.Mode = Options::Mode::EmitFiReport;
    else if (A == "-emit-systemverilog" || A == "-emit-sv")
      Opts.Mode = Options::Mode::EmitSystemVerilog;
    else if (A == "-check-synthesizable")
      Opts.Mode = Options::Mode::CheckSynthesizable;
    else if (A == "-emit-hardware-report" || A == "-emit-hw-report")
      Opts.Mode = Options::Mode::EmitHardwareReport;
    else if (A == "-dump-flow") Opts.Mode = Options::Mode::DumpFlow;
    else if (A == "-emit-matlab" || A == "-emit-m")
      Opts.Mode = Options::Mode::EmitMatlab;
    else if (A == "-repl") Opts.Mode = Options::Mode::Repl;
    else if (A == "-format") Opts.Mode = Options::Mode::Format;
    else if (A == "-dap") Opts.Mode = Options::Mode::Dap;
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
  for (const auto &T : Toks) {
    switch (T.Kind) {
    case TokenKind::kw_if:
    case TokenKind::kw_for:
    case TokenKind::kw_while:
    case TokenKind::kw_switch:
    case TokenKind::kw_try:
    case TokenKind::kw_function:
    case TokenKind::kw_classdef:
    case TokenKind::kw_parfor:
      ++d; break;
    case TokenKind::kw_end:
      --d; break;
    default: break;
    }
  }
  return d < 0 ? 0 : d;
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
int runReplInput(mlirgen::Context &MCtx, const std::string &Src, int Id,
                 std::string *DiagOut = nullptr) {
  SourceManager SM;
  FileID F = SM.addBuffer("<repl:" + std::to_string(Id) + ">", Src);
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

  SemaContext Sema;
  TypeContext TC;
  Resolver R(Sema, TC, Diag);
  R.setReplMode(true);
  R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  Inf.run(*TU);
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

  mlirgen::runSlotPromotion(M);
  // Rewrite Fixed-Point Designer (`fi`) ops into integer-shift sequences
  // BEFORE the generic scalar-to-arith pass — otherwise the matlab.add /
  // matlab.matmul that carry fi attributes get folded to plain arith.addi
  // / arith.muli and lose the spec metadata. See docs/emit_fixed_point.md.
  mlirgen::runLowerFixedPoint(M);
  mlirgen::runLowerScalarsToArith(M);
  mlirgen::runSlotPromotion(M);
  mlirgen::runOutlineParfor(M);
  mlirgen::runLowerSeqLoops(M);
  mlirgen::runLowerAnonCalls(M);
  for (int Iter = 0; Iter < 8; ++Iter) {
    bool A = mlirgen::runLowerScalarsToArith(M);
    bool B = mlirgen::runLowerUserCalls(M);
    if (!A && !B) break;
  }
  mlirgen::runLowerTensorOps(M);
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool A = mlirgen::runLowerScalarsToArith(M);
    bool B = mlirgen::runLowerUserCalls(M);
    if (!A && !B) break;
  }
  mlirgen::runLowerTensorOps(M);
  // Second LowerFixedPoint sweep — picks up matlab.call_builtin
  // @matlab_mat_*_slice1 / _concat_row sites that needed their tensor
  // operand retyped to ptr by LowerTensorOps first.
  mlirgen::runLowerFixedPoint(M);
  mlirgen::runLowerNarginNargout(M);
  mlirgen::runLowerScalarSlots(M);
  mlirgen::runLowerIO(M);

  if (mlir::failed(mlir::verify(M))) {
    std::cerr << "error: REPL MLIR verification failed after passes\n";
    return 1;
  }

  /* Same conversion-to-LLVM-dialect pipeline that lowerToLLVMIR runs.
   * We do it here rather than calling lowerToLLVMIR so ExecutionEngine
   * can consume the module directly instead of via an intermediate
   * textual LLVM IR round-trip. */
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
            << "    help <topic>       — detailed help on a topic\n\n"
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

/* Returns true if the line was handled as a help command (caller should
 * skip the compile pipeline for it). */
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
    std::cout << prompt << std::flush;
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

    Editor.addHistory(Line);
    Accum += Line;
    Accum += '\n';

    /* Lex once to decide if we have a complete balanced input. */
    SourceManager SM;
    FileID F = SM.addBuffer("<repl>", Accum);
    DiagnosticEngine Diag(SM);
    Lexer Lx(SM, F, Diag);
    auto Toks = Lx.tokenize();
    if (blockDepth(Toks) > 0) continue;  /* need more input */

    (void)runReplInput(MCtx, Accum, Counter++);
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
double matlab_dbg_ws_f64(int i);
void  *matlab_dbg_ws_ptr(int i);
void  matlab_ws_set_f64(const char *name, int64_t len, double v);
void  matlab_ws_set_mat(const char *name, int64_t len, struct matlab_mat *m);
void  matlab_ws_clear_one(const char *name, int64_t len);
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
extern "C" int64_t matlab_dbg_mat_rows(struct matlab_mat *m);
extern "C" int64_t matlab_dbg_mat_cols(struct matlab_mat *m);
extern "C" double matlab_dbg_mat_get(struct matlab_mat *m, int64_t i, int64_t j);
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
  /* Function name -> (file_id, first body line). Built at
   * compileProgram time from the TU's Function list (top-level +
   * nested) so the DAP `setFunctionBreakpoints` request can install
   * a line breakpoint at the function's entry by name. */
  struct FnEntry { int32_t FileId = 0; int32_t Line = 0; };
  std::unordered_map<std::string, FnEntry> FunctionTable;
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
          else if (K == 1 || K == 2) B.ptr = matlab_dbg_ws_ptr(j);
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
  G.FunctionTable.clear();
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
    /* `library_id` resolution honours the same env var the CLI
     * accepts. CLI `--block-path` isn't reachable from the DAP
     * launch surface yet — track via initializationOptions in a
     * follow-up. */
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
    if (Doc)
      TU = matlab::flowchart::buildAST(*Doc, AstCtx, SM, Diag, BO);
  } else {
    Lexer Lx(SM, F, Diag);
    auto Toks = Lx.tokenize();
    Parser P(std::move(Toks), AstCtx, Diag);
    TU = P.parseFile();
  }
  if (!TU || Diag.hasErrors()) { Diag.printAll(); return false; }

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
    auto recordStmt = [&](Stmt *S) {
      auto FL = stmtLine(S);
      if (FL.first != 0 && FL.second != 0)
        G.BpLocations[FL.first].insert(FL.second);
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
  R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  Inf.run(*TU);
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


  mlirgen::runSlotPromotion(M);
  // Rewrite Fixed-Point Designer (`fi`) ops into integer-shift sequences
  // BEFORE the generic scalar-to-arith pass — otherwise the matlab.add /
  // matlab.matmul that carry fi attributes get folded to plain arith.addi
  // / arith.muli and lose the spec metadata. See docs/emit_fixed_point.md.
  mlirgen::runLowerFixedPoint(M);
  mlirgen::runLowerScalarsToArith(M);
  mlirgen::runSlotPromotion(M);
  mlirgen::runOutlineParfor(M);
  mlirgen::runLowerSeqLoops(M);
  mlirgen::runLowerAnonCalls(M);
  for (int Iter = 0; Iter < 8; ++Iter) {
    bool A = mlirgen::runLowerScalarsToArith(M);
    bool B = mlirgen::runLowerUserCalls(M);
    if (!A && !B) break;
  }
  mlirgen::runLowerTensorOps(M);
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool A = mlirgen::runLowerScalarsToArith(M);
    bool B = mlirgen::runLowerUserCalls(M);
    if (!A && !B) break;
  }
  mlirgen::runLowerTensorOps(M);
  // Second LowerFixedPoint sweep — picks up matlab.call_builtin
  // @matlab_mat_*_slice1 / _concat_row sites that needed their tensor
  // operand retyped to ptr by LowerTensorOps first.
  mlirgen::runLowerFixedPoint(M);
  mlirgen::runLowerNarginNargout(M);
  mlirgen::runLowerScalarSlots(M);
  mlirgen::runLowerIO(M);

  if (getenv("MATLABC_DAP_DUMP")) mlirgen::printModule(std::cerr, M);

  mlir::PassManager PM(&MCtx.get());
  PM.addPass(mlir::createCanonicalizerPass());
  PM.addPass(mlir::createSCFToControlFlowPass());
  PM.addPass(mlir::createConvertControlFlowToLLVMPass());
  PM.addPass(mlir::createArithToLLVMConversionPass());
  PM.addPass(mlir::createConvertFuncToLLVMPass());
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(PM.run(M))) {
    std::cerr << "matlabc -dap: MLIR-to-LLVM conversion pipeline failed\n";
    return false;
  }

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
void *monitorMain(void *) {
  bool Debug = getenv("MATLABC_DAP_TRACE") != nullptr;
  while (true) {
    pthread_mutex_lock(&G.Mu);
    while (!G.WorkerExited && !matlab_dbg_is_paused())
      pthread_cond_wait(&G.Cv, &G.Mu);
    bool Exited = G.WorkerExited;
    pthread_mutex_unlock(&G.Mu);

    if (matlab_dbg_is_paused()) {
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
        sendEvent("stopped", Value(std::move(Body)));
        pthread_mutex_lock(&G.Mu);
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
 * the class name. The runtime kind enum (0=f64, 1=mat, 2=obj) maps
 * directly. */
std::string typeForVar(int Kind, void *Ptr) {
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
  if (Kind == 2) {
    return formatObj(matlab_dbg_ws_ptr(WsIdx));
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
            /* Snap forward to the next executable line. Forward
             * only — snapping backward would land before the user's
             * intent for a click in a blank-line gap between two
             * statements. */
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
    pthread_mutex_lock(&G.Mu);
    bool JustStarted = false;
    if (!G.WorkerStarted) {
      pthread_create(&G.Worker, nullptr, workerMain, nullptr);
      G.WorkerStarted = true;
      JustStarted = true;
      /* Detach; we use G.WorkerExited to know when it's done. */
      pthread_detach(G.Worker);
      /* Spawn the helper threads after the worker is kicked. */
      pthread_t Mon, Watcher, Rdr;
      pthread_create(&Mon, nullptr, monitorMain, nullptr);
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
      Object Fr{
        {"id", FrameId++},
        {"name", FnName ? FnName : "<frame>"},
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
    auto It = G.BpLocations.find(Fid);
    if (It != G.BpLocations.end()) {
      for (int32_t L : It->second) {
        if ((int64_t)L >= Start && (int64_t)L <= End) {
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
      "ndims", "numel", "ones", "prod", "rand", "randn", "real",
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
    matlab_dbg_resume(CONTINUE);
    nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"allThreadsContinued", true}});
    emitContinued();
    return true;
  }
  if (*Cmd == "next") {
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
    matlab_dbg_resume(STEP_OVER); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    emitContinued();
    return true;
  }
  if (*Cmd == "stepIn") {
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
    matlab_dbg_resume(STEP_IN); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    emitContinued();
    return true;
  }
  if (*Cmd == "stepOut") {
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
  return 0;
}

} // namespace dap
#endif
} // namespace

int main(int Argc, char **Argv) {
  Options Opts;
  const char *Prog = Argv[0];
  if (!parseArgs(Argc, Argv, Opts, Prog)) return usage(Prog);

#if MATLAB_LLVM_WITH_MLIR
  if (Opts.Mode == Options::Mode::Repl) return runRepl();
  if (Opts.Mode == Options::Mode::Dap) return dap::runDap(Opts.InputPath);
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

  SourceManager SM;
  FileID F = 0;
  if (Opts.ExtraInputs.empty()) {
    F = SM.loadFile(Opts.InputPath);
    if (F == 0) {
      std::cerr << Opts.InputPath << ": cannot open file\n";
      return 1;
    }
  } else {
    /* Multi-file input — concatenate `Opts.InputPath` + every
     * `ExtraInputs` path in CLI order with `\n` separators, surface
     * to the rest of the pipeline as one synthetic buffer. The
     * combined name is the primary input's path (so diagnostics
     * still mention a recognizable file) and per-file `% --- file
     * <path> ---` markers separate the regions. */
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
    if (!Append(Opts.InputPath)) return 1;
    for (const auto &P : Opts.ExtraInputs)
      if (!Append(P)) return 1;
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
    if (Doc)
      TU = matlab::flowchart::buildAST(*Doc, Ctx, SM, Diag, BO);
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

  // Sema
  SemaContext Sema;
  TypeContext TC;
  Resolver R(Sema, TC, Diag);
  if (TU) R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  if (TU) Inf.run(*TU);

  if (Opts.Mode == Options::Mode::EmitSema) {
    if (TU) dumpSema(std::cout, *TU);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
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
      bool IsSVPath = Opts.Mode == Options::Mode::EmitSystemVerilog ||
                      Opts.Mode == Options::Mode::CheckSynthesizable ||
                      Opts.Mode == Options::Mode::EmitHardwareReport;
      if (IsSVPath) {
        // Phase 5.6.1: scan `% hdl: port(...)` pragmas + apply them
        // to func signatures BEFORE the refinement iteration so a
        // function-only `.m` file (no typed driver) gets its port
        // widths from the pragma and the rest of the pipeline sees
        // typed args naturally. ScanHWPragmas is idempotent; the
        // SV-specific re-scan further down picks up the rest of the
        // pragma surface (fsm_encoding, input_pipeline, ...).
        mlirgen::runScanHWPragmas(M, &SM);
        if (!mlirgen::runApplyPortTypePragmas(M)) {
          Diag.printAll();
          return 1;
        }
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
        mlirgen::runOutlineParfor(M);
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
        mlirgen::runLowerTensorOps(M);
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
        // Multi-callsite monomorphisation: if a user function is called
        // with both scalar and matrix args (sq(5) + sq([1 2 3])) we
        // clone it per concrete signature so each specialisation
        // retypes independently. Runs AFTER LowerTensorOps when
        // operand types have collapsed to f64 / !llvm.ptr — matrix
        // shapes share the ptr sig. If any clones were made, iterate
        // the user-call + tensor-op fixpoint once more so the clones
        // get their signatures refined and their bodies retyped.
        if (mlirgen::runMonomorphiseUserCalls(M)) {
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
        // After user-call refinement, any surviving matlab.alloc whose
        // result type is now a scalar primitive can be promoted to
        // llvm.alloca. This catches function-body locals that weren't
        // promoted by SlotPromotion (because they're used across blocks).
        mlirgen::runLowerScalarSlots(M);
        mlirgen::runLowerIO(M);
        if (Opts.Mode == Options::Mode::EmitC ||
            Opts.Mode == Options::Mode::EmitCpp ||
            Opts.Mode == Options::Mode::EmitPython ||
            Opts.Mode == Options::Mode::EmitTypeScript) {
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
          mlirgen::runLowerScalarSlots(M);
          mlirgen::runMem2RegLite(M);
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
