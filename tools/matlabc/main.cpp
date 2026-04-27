#include "matlab/AST/AST.h"
#include "matlab/AST/ASTDumper.h"
#include "matlab/AST/Formatter.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
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
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <termios.h>
#include <unistd.h>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace matlab;

namespace {
struct Options {
  enum class Mode { DumpTokens, DumpAST, EmitSema, EmitMIR, EmitMLIR,
                    EmitLLVM, EmitC, EmitCpp, EmitPython, EmitTypeScript,
                    EmitFiReport, EmitSystemVerilog, CheckSynthesizable,
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
  std::string InputPath;
};

int usage(const char *Prog) {
  std::cerr << "usage: " << Prog
            << " [-dump-tokens | -dump-ast | -emit-sema | -emit-mir |\n"
               "             -emit-mlir | -emit-llvm | -emit-c | -emit-cpp |\n"
               "             -emit-python | -emit-typescript |\n"
               "             -emit-systemverilog | -check-synthesizable |\n"
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
    else if (A == "-h" || A == "--help") return false;
    else if (!A.empty() && A[0] == '-') {
      std::cerr << "unknown flag: " << A << "\n";
      return false;
    } else {
      if (!Opts.InputPath.empty()) return false;
      Opts.InputPath = std::string(A);
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

int runReplInput(mlirgen::Context &MCtx, const std::string &Src, int Id) {
  SourceManager SM;
  FileID F = SM.addBuffer("<repl:" + std::to_string(Id) + ">", Src);
  DiagnosticEngine Diag(SM);
  Lexer Lx(SM, F, Diag);
  auto Toks = Lx.tokenize();

  ASTContext AstCtx;
  Parser P(std::move(Toks), AstCtx, Diag);
  TranslationUnit *TU = P.parseFile();
  if (!TU || Diag.hasErrors()) {
    Diag.printAll();
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
    Diag.printAll();
    return 1;
  }

  auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU, &SM, /*ReplMode=*/true);
  if (Diag.hasErrors() || mlir::failed(mlir::verify(M))) {
    Diag.printAll();
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
int  matlab_dbg_breakpoint_meta(int idx, const char **cond, int64_t *cond_len,
                                 const char **log, int64_t *log_len,
                                 int *disabled);
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
}

/* Forward declarations from matlab_runtime.c so we can format matrices
 * into human-readable "1x3 double" strings for the DAP `variables`
 * response without duplicating the display logic. */
struct matlab_mat;
extern "C" int64_t matlab_dbg_mat_rows(struct matlab_mat *m);
extern "C" int64_t matlab_dbg_mat_cols(struct matlab_mat *m);
extern "C" double matlab_dbg_mat_get(struct matlab_mat *m, int64_t i, int64_t j);

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

/* Module-wide state threaded through worker / server / reader. */
struct Shared {
  std::string ProgramPath;   /* absolute / CLI-supplied path */
  std::unique_ptr<mlir::ExecutionEngine> Engine;
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

/* Try to evaluate `expr` as a MATLAB scalar in the current
 * workspace. Wraps it in `__matlab_dbg_cond = (expr);` and runs the
 * full REPL pipeline; the result lands in matlab_ws under that name.
 *
 * Returns 1 if the expression evaluated to a non-zero scalar, 0 if
 * it evaluated to zero, and -1 if the eval failed (parse error,
 * undefined name, etc). The caller can use -1 to disable the
 * condition so subsequent hits don't keep retrying. */
int evalConditionInWorkspace(const std::string &Expr) {
  std::string Src = "__matlab_dbg_cond = (" + Expr + ");";
  int Rc = runReplInput(sharedDapContext(), Src, NextEvalId++);
  if (Rc != 0) return -1;
  const char Name[] = "__matlab_dbg_cond";
  if (matlab_ws_has(Name, (int64_t)(sizeof Name - 1)) == 0.0) return -1;
  double V = matlab_ws_get_f64(Name, (int64_t)(sizeof Name - 1));
  return V != 0.0 ? 1 : 0;
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

  /* Register every file the SourceManager knows about with the
   * runtime's debug table. Today only the entry-point is loaded;
   * once Sema starts pulling sibling .m files in to resolve
   * cross-file calls they'll appear here automatically and
   * cross-file breakpoints will Just Work. */
  auto registerSMFile = [](FileID Fid, const std::string &Name) {
    matlab_dbg_register_file((int32_t)Fid, Name.data(),
                              (int64_t)Name.size());
    G.PathToFileId[canonPath(Name)] = (int32_t)Fid;
  };
  for (size_t i = 1; i <= SM.numFiles(); ++i)
    registerSMFile((FileID)i, SM.getName((FileID)i));

  DiagnosticEngine Diag(SM);
  Lexer Lx(SM, F, Diag);
  auto Toks = Lx.tokenize();
  ASTContext AstCtx;
  Parser P(std::move(Toks), AstCtx, Diag);
  TranslationUnit *TU = P.parseFile();
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
   * align with the main TU. */
  {
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

  mlir::ExecutionEngineOptions EngineOpts;
  EngineOpts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  auto EngineOrErr = mlir::ExecutionEngine::create(M, EngineOpts);
  if (!EngineOrErr) {
    std::cerr << "matlabc -dap: ExecutionEngine::create failed: "
              << llvm::toString(EngineOrErr.takeError()) << "\n";
    return false;
  }
  G.Engine = std::move(*EngineOrErr);
  return true;
}

/* Worker thread: invokes the JIT'd `main`. Sets WorkerExited + wakes
 * the monitor loop on return. */
void *workerMain(void *) {
  auto FnOrErr = G.Engine->lookup("main");
  if (FnOrErr) {
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
         * blocked inside matlab_dbg_hook; we resume it ourselves. */
        std::string Tmpl(Log, (size_t)LogLen);
        std::string Msg = interpolateLogMessage(Tmpl);
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
        const char *Reason = (BpIdx >= 0) ? "breakpoint" : "step";
        Object Body{
          {"reason", Reason},
          {"threadId", 1},
          {"allThreadsStopped", true},
          {"line", (int64_t)Ln},
        };
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

/* Format a single matrix row for display alongside its
 * variablesReference. 1x1 matrices unbox to the scalar (matches
 * matlab_struct_get_f64 and what users want to see in a counter
 * variable); everything else gets the `RxC double` shape summary so
 * the disclosure arrow in the IDE has a meaningful preview before
 * it's clicked. */
std::string formatMatShape(struct matlab_mat *M) {
  if (!M) return "[]";
  int64_t R = matlab_dbg_mat_rows(M);
  int64_t C = matlab_dbg_mat_cols(M);
  if (R == 1 && C == 1) {
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%g", matlab_dbg_mat_get(M, 1, 1));
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
 *   - 1xN row vector  -> linear "(j)" labels.
 *   - Mx1 col vector  -> linear "(i)" labels.
 *   - MxN matrix      -> "(i,j)" labels in row-major order.
 * 1x1 matrices have no children — the parent row already shows the
 * scalar via formatMatShape. */
constexpr size_t MatExpandCap = 256;

void appendMatChildren(Array &Vs, struct matlab_mat *M) {
  if (!M) return;
  int64_t R = matlab_dbg_mat_rows(M);
  int64_t C = matlab_dbg_mat_cols(M);
  if (R == 1 && C == 1) return;
  bool RowVec = (R == 1);
  bool ColVec = (C == 1);
  size_t emitted = 0;
  auto emit = [&](std::string label, double v) {
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%g", v);
    Vs.push_back(Object{
      {"name", std::move(label)},
      {"value", std::string(Buf)},
      {"variablesReference", (int64_t)0},
    });
    ++emitted;
  };
  for (int64_t i = 1; i <= R; ++i) {
    for (int64_t j = 1; j <= C; ++j) {
      if (emitted >= MatExpandCap) {
        Vs.push_back(Object{
          {"name", std::string("…")},
          {"value", std::string("(truncated)")},
          {"variablesReference", (int64_t)0},
        });
        return;
      }
      char LabelBuf[64];
      if (RowVec)      snprintf(LabelBuf, sizeof LabelBuf, "(%lld)", (long long)j);
      else if (ColVec) snprintf(LabelBuf, sizeof LabelBuf, "(%lld)", (long long)i);
      else             snprintf(LabelBuf, sizeof LabelBuf, "(%lld,%lld)",
                                  (long long)i, (long long)j);
      emit(LabelBuf, matlab_dbg_mat_get(M, i, j));
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
    Object Caps{
      {"supportsConfigurationDoneRequest", true},
      {"supportsFunctionBreakpoints", false},
      /* Conditional breakpoints + log points evaluate at script-frame
       * scope only (they read the workspace through matlab_ws_*).
       * Conditions inside user-function frames see <script>'s vars
       * but not the function's locals — Option B (per-function slot
       * tables) is the planned follow-up. */
      {"supportsConditionalBreakpoints", true},
      {"supportsLogPoints", true},
      /* setVariable accepts scalar (f64) values today; matrices,
       * strings, structs, and cells are rejected with a clear
       * error message in the response body. */
      {"supportsSetVariable", true},
      {"supportsStepBack", false},
      {"supportsTerminateRequest", true},
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
    int32_t Fid = 0;
    if (SrcPath) {
      auto It = G.PathToFileId.find(canonPath(SrcPath->str()));
      if (It != G.PathToFileId.end()) Fid = It->second;
    }
    if (Fid != 0) {
      /* Wipe prior breakpoints for this file and replay the request. */
      matlab_dbg_clear_breakpoints_in_file(Fid);
    }
    const Array *Bps = Args->getArray("breakpoints");
    Array Verified;
    if (Bps) {
      for (const auto &B : *Bps) {
        const Object *BO = B.getAsObject();
        if (!BO) continue;
        auto Ln = BO->getInteger("line");
        if (!Ln) continue;
        bool OK = false;
        if (Fid != 0) {
          /* condition / logMessage are optional in the DAP spec;
           * when present, route through the _ex form so the runtime
           * stores the strings alongside the (file_id, line) pair
           * for the monitor thread to read once the bp matches. */
          auto Cond = BO->getString("condition");
          auto Log  = BO->getString("logMessage");
          std::string CS = Cond ? Cond->str() : std::string();
          std::string LS = Log  ? Log->str()  : std::string();
          OK = matlab_dbg_add_breakpoint_ex(
              Fid, (int32_t)*Ln,
              CS.empty() ? nullptr : CS.data(), (int64_t)CS.size(),
              LS.empty() ? nullptr : LS.data(), (int64_t)LS.size());
        }
        Verified.push_back(Object{
          {"verified", OK},
          {"line", *Ln},
        });
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"breakpoints", std::move(Verified)}});
    return true;
  }

  if (*Cmd == "configurationDone") {
    sendResponse(ReqSeq, *Cmd, true, Object{});
    pthread_mutex_lock(&G.Mu);
    if (!G.WorkerStarted) {
      pthread_create(&G.Worker, nullptr, workerMain, nullptr);
      G.WorkerStarted = true;
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
    }
    pthread_mutex_unlock(&G.Mu);
    return true;
  }

  if (*Cmd == "threads") {
    Array Ts;
    Ts.push_back(Object{{"id", 1}, {"name", "main"}});
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
             * down without a separate eval. */
            if (M && (matlab_dbg_mat_rows(M) != 1 ||
                      matlab_dbg_mat_cols(M) != 1))
              ChildRef = registerMatRef(M);
          } else if (K == 2) {
            void *child = matlab_dbg_obj_field_ptr(obj, i);
            Val = formatObj(child);
            if (child) ChildRef = registerObjRef(child);
          } else {
            Val = "<unknown>";
          }
          Vs.push_back(Object{
            {"name", std::string(Nm, (size_t)Nlen)},
            {"value", Val},
            {"variablesReference", ChildRef},
          });
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
        if (K == 2) {
          if (void *obj = matlab_dbg_ws_ptr(i)) ChildRef = registerObjRef(obj);
        } else if (K == 1) {
          auto *M = (struct matlab_mat *)matlab_dbg_ws_ptr(i);
          if (M && (matlab_dbg_mat_rows(M) != 1 ||
                    matlab_dbg_mat_cols(M) != 1))
            ChildRef = registerMatRef(M);
        }
        Vs.push_back(Object{
          {"name", Nstr},
          {"value", formatVar(K, i)},
          {"variablesReference", ChildRef},
        });
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
          if (M && (matlab_dbg_mat_rows(M) != 1 ||
                    matlab_dbg_mat_cols(M) != 1))
            ChildRef = registerMatRef(M);
        } else if (K == 2) {
          void *obj = matlab_dbg_frame_local_ptr(RtFrameIdx, i);
          Val = formatObj(obj);
          if (obj) ChildRef = registerObjRef(obj);
        } else {
          Val = "<unknown>";
        }
        Vs.push_back(Object{
          {"name", Nstr},
          {"value", Val},
          {"variablesReference", ChildRef},
        });
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
    auto ExprOpt = Args->getString("expression");
    if (!ExprOpt) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("evaluate requires an expression"));
      return true;
    }
    std::string Expr = ExprOpt->str();
    /* The DAP spec allows `expression` to be a statement-level command
     * in the "repl" context. Strip a single trailing semicolon if the
     * user typed one — our wrap injects its own. */
    while (!Expr.empty() &&
           (Expr.back() == ' ' || Expr.back() == '\t' ||
            Expr.back() == '\n' || Expr.back() == ';'))
      Expr.pop_back();
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

    /* Snapshot the pre-eval state of every matlab_ws entry whose name
     * collides with what we're about to stamp from the frame's
     * mini-ws. We use a struct-of-arrays so the snapshot survives the
     * subsequent set_f64/set_mat calls without dangling pointers
     * (matlab_ws may reorganize internally on insert).
     *
     * Two sets of names are tracked:
     *   - PreExisting: names already in matlab_ws before stamping.
     *     Restored with their original kind/value after eval.
     *   - Stamped: names we wrote during stamping. After eval, any
     *     stamped name not also PreExisting gets matlab_ws_clear_one'd
     *     so the script workspace doesn't keep function locals. */
    struct WsBackup { std::string name; int kind; double f64; void *ptr; };
    std::vector<WsBackup> Backup;
    std::unordered_set<std::string> PreExisting;
    std::vector<std::string> Stamped;
    if (RtFrameIdx > 0) {
      int N = matlab_dbg_ws_count();
      for (int i = 0; i < N; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_ws_name(i, &Nlen);
        if (!Nm) continue;
        std::string Nstr(Nm, (size_t)Nlen);
        PreExisting.insert(Nstr);
      }
      int FN = matlab_dbg_frame_locals_count(RtFrameIdx);
      for (int i = 0; i < FN; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_frame_local_name(RtFrameIdx, i, &Nlen);
        if (!Nm) continue;
        std::string Nstr(Nm, (size_t)Nlen);
        /* Capture the prior matlab_ws value (if any) so we can put it
         * back. Re-look-up by name rather than caching above so we
         * pick up the current kind/value rather than a stale one in
         * case the ws got reorganized. */
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
          double V = matlab_dbg_frame_local_f64(RtFrameIdx, i);
          matlab_ws_set_f64(Nstr.data(), (int64_t)Nstr.size(), V);
        } else if (K == 1) {
          void *P = matlab_dbg_frame_local_ptr(RtFrameIdx, i);
          /* matlab_ws_set_mat takes the matrix descriptor pointer
           * verbatim; the struct stays owned by the JIT's slot for
           * the lifetime of the frame, which covers our eval. */
          matlab_ws_set_mat(Nstr.data(), (int64_t)Nstr.size(),
                             (struct matlab_mat *)P);
        } else if (K == 2) {
          /* Class instance: stamp via matlab_ws_set_obj so the
           * workspace remembers it as kind=2. The eval JIT only
           * needs to see the obj pointer; method dispatch routes
           * through matlab_obj_* the same way it does in the
           * compiled code, so `obj.method()` and `obj.Prop` work
           * inside watch expressions. */
          void *P = matlab_dbg_frame_local_ptr(RtFrameIdx, i);
          matlab_ws_set_obj(Nstr.data(), (int64_t)Nstr.size(), P);
        }
        Stamped.push_back(std::move(Nstr));
      }
    }

    const char EvalName[] = "__matlab_dbg_eval";
    std::string Src = std::string(EvalName) + " = (" + Expr + ");";
    int Rc = runReplInput(sharedDapContext(), Src, NextEvalId++);

    /* Read the result before any restoration so we can format it. */
    std::string Display;
    int64_t EvalRef = 0;
    bool RcOk = (Rc == 0);
    if (RcOk) {
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
        if (void *obj = matlab_dbg_ws_ptr(Found))
          EvalRef = registerObjRef(obj);
      } else if (Found >= 0 && Kind == 1) {
        auto *M = (struct matlab_mat *)matlab_dbg_ws_ptr(Found);
        if (M && (matlab_dbg_mat_rows(M) != 1 ||
                  matlab_dbg_mat_cols(M) != 1))
          EvalRef = registerMatRef(M);
      }
    }

    /* Restore matlab_ws to its pre-stamp state. Order matters: clear
     * the freshly-stamped names first (so they don't shadow the
     * restored values), then re-set the pre-existing ones. The eval
     * result holder is also cleared so it doesn't pile up in the
     * workspace across many evaluate calls. */
    matlab_ws_clear_one(EvalName, (int64_t)(sizeof EvalName - 1));
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

    if (!RcOk) {
      sendResponse(ReqSeq, *Cmd, false,
                   Value("evaluate expression failed to compile"));
      return true;
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"result", Display},
                        {"variablesReference", EvalRef}});
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
  if (*Cmd == "continue") {
    matlab_dbg_resume(CONTINUE);
    nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"allThreadsContinued", true}});
    return true;
  }
  if (*Cmd == "next") {
    matlab_dbg_resume(STEP_OVER); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{}); return true;
  }
  if (*Cmd == "stepIn") {
    matlab_dbg_resume(STEP_IN); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{}); return true;
  }
  if (*Cmd == "stepOut") {
    matlab_dbg_resume(STEP_OUT); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{}); return true;
  }

  if (*Cmd == "pause") {
    /* Ask the runtime to stop at the next hook. */
    matlab_dbg_resume(STEP_IN); nudgeMonitor();
    sendResponse(ReqSeq, *Cmd, true, Object{});
    return true;
  }

  if (*Cmd == "terminate" || *Cmd == "disconnect") {
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

  SourceManager SM;
  FileID F = SM.loadFile(Opts.InputPath);
  if (F == 0) {
    std::cerr << Opts.InputPath << ": cannot open file\n";
    return 1;
  }

  DiagnosticEngine Diag(SM);
  Lexer Lx(SM, F, Diag);
  auto Toks = Lx.tokenize();

  if (Opts.Mode == Options::Mode::DumpTokens) {
    dumpTokens(SM, Toks);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  ASTContext Ctx;
  Parser P(std::move(Toks), Ctx, Diag);
  TranslationUnit *TU = P.parseFile();

  if (Opts.Mode == Options::Mode::DumpAST) {
    if (TU) dumpAST(std::cout, *TU);
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }

  if (Opts.Mode == Options::Mode::Format) {
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
      Opts.Mode == Options::Mode::CheckSynthesizable) {
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
                              Opts.Mode == Options::Mode::CheckSynthesizable;
      bool WantClean = Opts.Opt || WantFullPipeline;
      if (WantClean) {
        mlirgen::runSlotPromotion(M);
        // See docs/emit_fixed_point.md — fi ops must lower before arith.
        mlirgen::runLowerFixedPoint(M);
        mlirgen::runLowerScalarsToArith(M);
        mlirgen::runSlotPromotion(M);
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
                   Opts.Mode == Options::Mode::CheckSynthesizable) {
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
            // Print Diag and exit; no SV output. Exit 0 on clean.
            Diag.printAll();
            return Ok ? 0 : 1;
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
          std::string Src = mlirgen::emitSystemVerilog(M, &SM, R);
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
