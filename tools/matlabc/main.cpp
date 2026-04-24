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
#include "llvm/Support/TargetSelect.h"
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
                    EmitLLVM, EmitC, EmitCpp, Check, Repl, Format };
  Mode Mode = Mode::Check;
  bool Opt = false;
  std::string InputPath;
};

int usage(const char *Prog) {
  std::cerr << "usage: " << Prog
            << " [-dump-tokens | -dump-ast | -emit-sema | -emit-mir |\n"
               "             -emit-mlir | -emit-llvm | -emit-c | -emit-cpp |\n"
               "             -format | -repl] FILE.m\n";
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
    else if (A == "-opt" || A == "-O") Opts.Opt = true;
    else if (A == "-h" || A == "--help") return false;
    else if (!A.empty() && A[0] == '-') {
      std::cerr << "unknown flag: " << A << "\n";
      return false;
    } else {
      if (!Opts.InputPath.empty()) return false;
      Opts.InputPath = std::string(A);
    }
  }
  /* -repl doesn't take a file. Everything else does. */
  if (Opts.Mode == Options::Mode::Repl) return true;
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
#endif
} // namespace

int main(int Argc, char **Argv) {
  Options Opts;
  const char *Prog = Argv[0];
  if (!parseArgs(Argc, Argv, Opts, Prog)) return usage(Prog);

#if MATLAB_LLVM_WITH_MLIR
  if (Opts.Mode == Options::Mode::Repl) return runRepl();
#else
  if (Opts.Mode == Options::Mode::Repl) {
    std::cerr << "error: matlabc was built without MLIR support; "
                 "REPL is unavailable\n";
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
          // Verify the module right before emission so a malformed IR
          // state is surfaced with a clear error rather than as a cryptic
          // cc/c++ compile failure on the emitted source.
          if (mlir::failed(mlir::verify(M))) {
            std::cerr
                << "error: MLIR verification failed before C emission\n";
            return 1;
          }
          std::string Src = mlirgen::emitC(
              M, Opts.Mode == Options::Mode::EmitCpp);
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
