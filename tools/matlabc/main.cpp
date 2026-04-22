#include "matlab/AST/AST.h"
#include "matlab/AST/ASTDumper.h"
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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
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
  enum class Mode { DumpTokens, DumpAST, EmitSema, EmitMIR, EmitMLIR, Check };
  Mode Mode = Mode::Check;
  bool Opt = false;
  std::string InputPath;
};

int usage(const char *Prog) {
  std::cerr << "usage: " << Prog
            << " [-dump-tokens | -dump-ast | -emit-sema | -emit-mir] FILE.m\n";
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
} // namespace

int main(int Argc, char **Argv) {
  Options Opts;
  const char *Prog = Argv[0];
  if (!parseArgs(Argc, Argv, Opts, Prog)) return usage(Prog);

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
  if (Opts.Mode == Options::Mode::EmitMLIR) {
    mlirgen::Context MCtx;
    if (TU) {
      auto M = mlirgen::lowerToMLIR(MCtx, TC, Diag, *TU);
      if (mlir::failed(mlir::verify(M))) {
        std::cerr << "error: MLIR verification failed after lowering\n";
        return 1;
      }
      if (Opts.Opt) {
        mlirgen::runSlotPromotion(M);
        mlirgen::runLowerScalarsToArith(M);
        mlirgen::runSlotPromotion(M); // second sweep in case lowering unblocks
        if (mlir::failed(mlir::verify(M))) {
          std::cerr << "error: MLIR verification failed after passes\n";
          return 1;
        }
      }
      mlirgen::printModule(std::cout, M);
    }
    Diag.printAll();
    return Diag.hasErrors() ? 1 : 0;
  }
#endif

  Diag.printAll();
  return Diag.hasErrors() ? 1 : 0;
}
