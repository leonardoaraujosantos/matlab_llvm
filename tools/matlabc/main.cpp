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

#include <iostream>
#include <string>
#include <string_view>

using namespace matlab;

namespace {
struct Options {
  enum class Mode { DumpTokens, DumpAST, EmitSema, EmitMIR, EmitMLIR,
                    EmitLLVM, EmitC, EmitCpp, Check, Repl, Format, Dap };
  Mode Mode = Mode::Check;
  bool Opt = false;
  bool NoLine = false;
  bool Doxygen = false;
  bool CppAuto = false;
  std::string InputPath;
};

int usage(const char *Prog) {
  std::cerr << "usage: " << Prog
            << " [-dump-tokens | -dump-ast | -emit-sema | -emit-mir |\n"
               "             -emit-mlir | -emit-llvm | -emit-c | -emit-cpp |\n"
               "             -format | -repl | -dap]\n"
               "            [-no-line] [-doxygen] [-cpp-auto]  FILE.m\n";
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
    else if (A == "-repl") Opts.Mode = Options::Mode::Repl;
    else if (A == "-format") Opts.Mode = Options::Mode::Format;
    else if (A == "-dap") Opts.Mode = Options::Mode::Dap;
    else if (A == "-opt" || A == "-O") Opts.Opt = true;
    else if (A == "-no-line" || A == "--no-line") Opts.NoLine = true;
    else if (A == "-doxygen" || A == "--doxygen") Opts.Doxygen = true;
    else if (A == "-cpp-auto" || A == "--cpp-auto") Opts.CppAuto = true;
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

int runRepl() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlirgen::Context MCtx;
  mlir::registerBuiltinDialectTranslation(MCtx.get());
  mlir::registerLLVMDialectTranslation(MCtx.get());

  std::cerr << "matlabc REPL (experimental). Ctrl-D or `exit` to quit.\n";
  std::string Accum;
  int Counter = 0;
  while (true) {
    std::cout << (Accum.empty() ? ">> " : "   ") << std::flush;
    std::string Line;
    if (!std::getline(std::cin, Line)) { std::cout << '\n'; break; }
    if (Accum.empty() && (Line == "exit" || Line == "quit" ||
                          Line == "exit;" || Line == "quit;"))
      break;
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
 *   - script-level stepping (user-function entry/exit hooks aren't
 *     emitted yet, so stepIn past a call falls through to stepOver
 *     semantically).
 *   - Locals scope = the REPL workspace snapshot. Stack trace has one
 *     frame ("<script>") plus any frames matlab_dbg_enter_frame has
 *     pushed (currently only done by future user-function work). */

/* Prototypes for the runtime DAP API. Defined in matlab_runtime.c and
 * linked into matlabc for this path. */
extern "C" {
void matlab_dbg_enable(int stop_on_entry);
void matlab_dbg_register_file(int32_t file_id, const char *name,
                               int64_t name_len);
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
}

/* Forward declarations from matlab_runtime.c so we can format matrices
 * into human-readable "1x3 double" strings for the DAP `variables`
 * response without duplicating the display logic. */
struct matlab_mat;
extern "C" int64_t matlab_dbg_mat_rows(struct matlab_mat *m);
extern "C" int64_t matlab_dbg_mat_cols(struct matlab_mat *m);

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

Object sourceObj() {
  Object O;
  O["name"] = G.ProgramPath.substr(G.ProgramPath.find_last_of('/') + 1);
  O["path"] = absPath(G.ProgramPath);
  return O;
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
  matlab_dbg_register_file(G.FileId, G.ProgramPath.data(),
                            (int64_t)G.ProgramPath.size());

  DiagnosticEngine Diag(SM);
  Lexer Lx(SM, F, Diag);
  auto Toks = Lx.tokenize();
  ASTContext AstCtx;
  Parser P(std::move(Toks), AstCtx, Diag);
  TranslationUnit *TU = P.parseFile();
  if (!TU || Diag.hasErrors()) { Diag.printAll(); return false; }

  SemaContext Sema;
  TypeContext TC;
  Resolver R(Sema, TC, Diag);
  R.setReplMode(true);
  R.resolve(*TU);
  TypeInference Inf(Sema, TC, Diag);
  Inf.run(*TU);
  if (Diag.hasErrors()) { Diag.printAll(); return false; }

  /* Keep MLIR context alive for the lifetime of the ExecutionEngine.
   * We leak it into a static — the process exits on disconnect so
   * there's no lifecycle to manage beyond that. */
  static mlirgen::Context MCtx;
  static bool Inited = false;
  if (!Inited) {
    mlir::registerBuiltinDialectTranslation(MCtx.get());
    mlir::registerLLVMDialectTranslation(MCtx.get());
    Inited = true;
  }

  auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU, &SM,
                                /*ReplMode=*/true, /*DebugMode=*/true);
  if (Diag.hasErrors() || mlir::failed(mlir::verify(M))) {
    Diag.printAll();
    std::cerr << "matlabc -dap: MLIR verification failed\n";
    return false;
  }


  mlirgen::runSlotPromotion(M);
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
 * the matching DAP event. Loops until the worker exits. */
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
      if (Debug) std::fprintf(stderr, "[monitor] stopped at %d\n", Ln);
      Object Body{
        {"reason", "breakpoint"},
        {"threadId", 1},
        {"allThreadsStopped", true},
        {"line", (int64_t)Ln},
      };
      sendEvent("stopped", Value(std::move(Body)));
      /* Wait until the worker is no longer paused (client resumed). */
      pthread_mutex_lock(&G.Mu);
      while (matlab_dbg_is_paused() && !G.WorkerExited)
        pthread_cond_wait(&G.Cv, &G.Mu);
      pthread_mutex_unlock(&G.Mu);
      if (Debug) std::fprintf(stderr, "[monitor] resumed\n");
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

/* Format a variable for the DAP `variables` response. Matrices get
 * a shape summary ("1x3 double"), scalars get the f64 value. */
std::string formatVar(int Kind, int WsIdx) {
  if (Kind == 0) {
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%g", matlab_dbg_ws_f64(WsIdx));
    return Buf;
  }
  if (Kind == 1) {
    auto *M = (struct matlab_mat *)matlab_dbg_ws_ptr(WsIdx);
    if (!M) return "[]";
    int64_t R = matlab_dbg_mat_rows(M);
    int64_t C = matlab_dbg_mat_cols(M);
    char Buf[64];
    snprintf(Buf, sizeof Buf, "%lldx%lld double",
             (long long)R, (long long)C);
    return Buf;
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
      {"supportsConditionalBreakpoints", false},
      {"supportsSetVariable", false},
      {"supportsStepBack", false},
      {"supportsTerminateRequest", true},
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
    /* Wipe prior breakpoints for this file and replay the request. */
    matlab_dbg_clear_breakpoints_in_file(G.FileId);
    const Array *Bps = Args->getArray("breakpoints");
    Array Verified;
    if (Bps) {
      for (const auto &B : *Bps) {
        const Object *BO = B.getAsObject();
        if (!BO) continue;
        auto Ln = BO->getInteger("line");
        if (!Ln) continue;
        bool OK = matlab_dbg_add_breakpoint(G.FileId, (int32_t)*Ln);
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
        {"source", sourceObj()},
      };
      Frames.push_back(std::move(Fr));
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"stackFrames", std::move(Frames)},
                        {"totalFrames", (int64_t)N}});
    return true;
  }

  if (*Cmd == "scopes") {
    Array Sc;
    Sc.push_back(Object{
      {"name", "Locals"},
      {"variablesReference", 1},
      {"expensive", false},
    });
    sendResponse(ReqSeq, *Cmd, true, Object{{"scopes", std::move(Sc)}});
    return true;
  }

  if (*Cmd == "variables") {
    auto VR = Args->getInteger("variablesReference");
    Array Vs;
    if (VR && *VR == 1) {
      int N = matlab_dbg_ws_count();
      for (int i = 0; i < N; ++i) {
        int64_t Nlen = 0;
        const char *Nm = matlab_dbg_ws_name(i, &Nlen);
        int K = matlab_dbg_ws_kind(i);
        Vs.push_back(Object{
          {"name", std::string(Nm, (size_t)Nlen)},
          {"value", formatVar(K, i)},
          {"variablesReference", 0},
        });
      }
    }
    sendResponse(ReqSeq, *Cmd, true,
                 Object{{"variables", std::move(Vs)}});
    return true;
  }

  auto nudgeMonitor = [] {
    pthread_mutex_lock(&G.Mu);
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
      Opts.Mode == Options::Mode::EmitCpp) {
    mlirgen::Context MCtx;
    if (TU) {
      auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU, &SM);
      if (mlir::failed(mlir::verify(M))) {
        std::cerr << "error: MLIR verification failed after lowering\n";
        return 1;
      }
      // Opt/Run paths always clean up slots and scalars.
      bool WantFullPipeline = Opts.Mode == Options::Mode::EmitLLVM ||
                              Opts.Mode == Options::Mode::EmitC ||
                              Opts.Mode == Options::Mode::EmitCpp;
      bool WantClean = Opts.Opt || WantFullPipeline;
      if (WantClean) {
        mlirgen::runSlotPromotion(M);
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
            Opts.Mode == Options::Mode::EmitCpp) {
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
          std::string Src = mlirgen::emitC(
              M, Opts.Mode == Options::Mode::EmitCpp, Opts.NoLine,
              Opts.Doxygen, Opts.CppAuto, &SM);
          if (Src.empty()) return 1;
          std::cout << Src;
        } else {
          std::string LL = mlirgen::lowerToLLVMIR(M);
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
