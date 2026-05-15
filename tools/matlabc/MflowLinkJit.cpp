//===----------------------------------------------------------------------===//
// §17.5 #8 carve-out — true MLIR JIT for `signal_matlab_fcn` bodies.
//
// `MflowLinkSim` ships with a scalar-only AST interpreter as the always-
// available fallback (lib/Flowchart/MflowLinkSim.cpp). This TU plugs the
// optional JIT factory the simulator consults at construction: each
// signal_matlab_fcn block whose `params.function_body` compiles through
// the regular matlab_llvm pipeline gets evaluated by its JIT'd
// entrypoint instead of the AST walker. Bodies that hit a corner of
// the language the JIT doesn't cope with (typically things like
// `n`-as-loop-bound for triple-helper chains where the type fixpoint
// stalls) silently fall back to the interpreter so the simulation
// keeps running.
//
// Layering: lives in matlabc rather than `MatlabFlowchart` so the
// flowchart library doesn't acquire MLIR / LLVM-ORC as a transitive
// dependency. `installMflowLinkJit()` is invoked once from main()
// before any `-simulate` path runs.
//
// Wrapper synthesis: given a user body
//
//   function y = compute(u1, u2)
//     ...
//   end
//
// we emit a small driver TU:
//
//   __priming = mflowlink_jit_entry(0.0, 0.0);   % type-refinement
//
//   function y = mflowlink_jit_entry(u1, u2)
//     y = compute(u1, u2);
//   end
//
//   <user body verbatim>
//
// The driver call site forces the entry-function's arg slots to f64,
// which back-propagates through `compute` and any helpers the user
// declared inline. `main` is emitted but never invoked — we look up
// `mflowlink_jit_entry` by name and cast its address to a flat
// `(double, ...) → double` function pointer.
//===----------------------------------------------------------------------===//

#include "matlab/Flowchart/MflowLinkSim.h"

#include "matlab/AST/AST.h"
#include "matlab/Basic/Diagnostic.h"
#include "matlab/Basic/SourceManager.h"
#include "matlab/Lex/Lexer.h"
#include "matlab/Parse/Parser.h"
#include "matlab/Sema/Resolver.h"
#include "matlab/Sema/Type.h"
#include "matlab/Sema/TypeInference.h"

#if MATLAB_LLVM_WITH_MLIR

#include "matlab/MLIR/Context.h"
#include "matlab/MLIR/Lowering.h"
#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

#include <memory>
#include <sstream>
#include <string>

namespace matlab::flowchart {

namespace {

//===-----------------------------------------------------------------===//
// Per-block JIT handle. Holds the MLIR context, the engine that owns
// the compiled code, and the resolved entry-point function pointer.
//===-----------------------------------------------------------------===//
struct JitHandle {
  std::unique_ptr<matlab::mlirgen::Context> Ctx;
  std::unique_ptr<mlir::ExecutionEngine> Engine;
  void *EntryAddr = nullptr;
  unsigned NumInputs = 0;
};

//===-----------------------------------------------------------------===//
// Extract `<name>` from `function [<outs>] = <name>(<args>)`. Handles
// both the single-output `function y = foo(x)` form and the
// bracket-list `function [a, b] = foo(x)` form. Returns an empty
// string on failure; the caller falls back to the AST interpreter.
//===-----------------------------------------------------------------===//
std::string extractFunctionName(const std::string &Source) {
  // The simple line-scan: find the first `function ` keyword, then
  // walk past `[...] = ` or `name = ` to land on the function name
  // immediately before the `(`.
  size_t Pos = Source.find("function");
  if (Pos == std::string::npos) return {};
  Pos += sizeof("function") - 1;
  // Skip whitespace.
  while (Pos < Source.size() && (Source[Pos] == ' ' || Source[Pos] == '\t'))
    ++Pos;
  // Optional output spec: skip up to `=`.
  size_t Eq = Source.find('=', Pos);
  size_t Lp = Source.find('(', Pos);
  if (Lp == std::string::npos) return {};
  size_t NameEnd = Lp;
  size_t NameStart = (Eq != std::string::npos && Eq < Lp) ? Eq + 1 : Pos;
  while (NameStart < NameEnd &&
         (Source[NameStart] == ' ' || Source[NameStart] == '\t'))
    ++NameStart;
  size_t E = NameEnd;
  while (E > NameStart &&
         (Source[E - 1] == ' ' || Source[E - 1] == '\t'))
    --E;
  if (NameStart >= E) return {};
  return Source.substr(NameStart, E - NameStart);
}

//===-----------------------------------------------------------------===//
// Build the TU source the JIT compiles.
//===-----------------------------------------------------------------===//
std::string synthesizeWrapper(const std::string &Body,
                              unsigned NumInputs,
                              const std::string &FnName) {
  std::ostringstream OS;

  // 1) Driver call — forces the entry function's args to f64. The
  //    result is bound to a throwaway variable so it doesn't trip
  //    the parser's "stray tokens after function definitions" rule.
  OS << "__mflowlink_priming = mflowlink_jit_entry(";
  for (unsigned i = 0; i < NumInputs; ++i) {
    if (i) OS << ", ";
    OS << "0.0";
  }
  OS << ");\n\n";

  // 2) Wrapper function — single line of work, delegates to the
  //    user-named function so type inference back-propagates through
  //    the call site (user function's args become f64).
  OS << "function y = mflowlink_jit_entry(";
  for (unsigned i = 0; i < NumInputs; ++i) {
    if (i) OS << ", ";
    OS << "u" << (i + 1);
  }
  OS << ")\n  y = " << FnName << "(";
  for (unsigned i = 0; i < NumInputs; ++i) {
    if (i) OS << ", ";
    OS << "u" << (i + 1);
  }
  OS << ");\nend\n\n";

  // 3) User function body verbatim.
  OS << Body;
  if (!Body.empty() && Body.back() != '\n') OS << "\n";
  return OS.str();
}

//===-----------------------------------------------------------------===//
// Run the MLIR static `-emit-llvm` pipeline through to the JIT.
// Returns the constructed ExecutionEngine + the resolved entry
// address, or null on failure with `Err` populated.
//===-----------------------------------------------------------------===//
bool runPipeline(mlir::ModuleOp M, std::string &Err) {
  using namespace mlir;
  using namespace matlab::mlirgen;
  if (failed(verify(M))) { Err = "MLIR verification failed"; return false; }

  // Bootstrap: apply pragma scan + slot-type seeding so function-only
  // entries pick up concrete types where the static -emit-* path
  // does it.
  runScanHWPragmas(M, /*SM=*/nullptr);
  if (!runApplyPortTypePragmas(M)) { Err = "port-pragma apply failed"; return false; }
  runRefineSlotTypes(M);

  runSlotPromotion(M);
  runLowerFixedPoint(M);
  runLowerScalarsToArith(M);
  runSlotPromotion(M);
  runRefineFuncSigs(M);
  if (failed(verify(M))) { Err = "verification failed after slot promo"; return false; }

  runOutlineParfor(M);
  runLowerSeqLoops(M);
  runLowerAnonCalls(M, /*ReplMode=*/false);
  for (int Iter = 0; Iter < 8; ++Iter) {
    bool A = runLowerScalarsToArith(M);
    bool B = runLowerUserCalls(M);
    if (!A && !B) break;
  }
  runLowerTensorOps(M);
  for (int Iter = 0; Iter < 4; ++Iter) {
    bool A = runLowerScalarsToArith(M);
    bool B = runLowerUserCalls(M);
    if (!A && !B) break;
  }
  runLowerTensorOps(M);
  if (runLowerAnonCallsPost(M)) {
    runLowerTensorOps(M);
    for (int Iter = 0; Iter < 4; ++Iter) {
      bool A = runLowerScalarsToArith(M);
      bool B = runLowerUserCalls(M);
      if (!A && !B) break;
    }
    runLowerTensorOps(M);
  }
  if (runMonomorphiseUserCalls(M)) {
    for (int Iter = 0; Iter < 4; ++Iter) {
      bool A = runLowerScalarsToArith(M);
      bool B = runLowerUserCalls(M);
      if (!A && !B) break;
    }
    runLowerTensorOps(M);
  }
  runLowerFixedPoint(M);
  runLowerNarginNargout(M);
  runLowerScalarSlots(M);
  runRefineFuncSigs(M);
  runLowerTensorOps(M);
  runLowerIO(M);

  if (failed(verify(M))) { Err = "verification failed after passes"; return false; }

  // Convert to LLVM dialect — same set as the REPL pipeline.
  PassManager PM(M.getContext());
  PM.addPass(createCanonicalizerPass());
  PM.addPass(createSCFToControlFlowPass());
  PM.addPass(createConvertControlFlowToLLVMPass());
  PM.addPass(createArithToLLVMConversionPass());
  PM.addPass(createConvertFuncToLLVMPass());
  PM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(PM.run(M))) {
    Err = "MLIR-to-LLVM conversion pipeline failed";
    return false;
  }

  stripMatlabFuncAttrs(M);
  return true;
}

MatlabFcnJit::Handle *compileImpl(const std::string &Body,
                                  unsigned NumInputs,
                                  std::string &Err) {
  if (NumInputs > 8) {
    Err = "JIT supports at most 8 inputs (got "
        + std::to_string(NumInputs) + ")";
    return nullptr;
  }
  std::string FnName = extractFunctionName(Body);
  if (FnName.empty()) {
    Err = "could not extract function name from body";
    return nullptr;
  }
  // Defensive: the wrapper's entry name and the user's name must
  // differ — otherwise we'd shadow the user function.
  if (FnName == "mflowlink_jit_entry") {
    Err = "user function must not be named `mflowlink_jit_entry`";
    return nullptr;
  }

  std::string Source = synthesizeWrapper(Body, NumInputs, FnName);

  // Lex + Parse + Sema in a fresh source manager so diagnostics
  // never leak into the surrounding -simulate run.
  matlab::SourceManager SM;
  matlab::FileID F = SM.addBuffer("<mflowlink-jit>", Source);
  matlab::DiagnosticEngine Diag(SM);

  matlab::Lexer Lex(SM, F, Diag);
  auto Tokens = Lex.tokenize();
  if (Diag.hasErrors()) { Err = "lex error in synthesized TU"; return nullptr; }

  matlab::ASTContext AST;
  matlab::Parser Parser(std::move(Tokens), AST, Diag);
  auto *TU = Parser.parseFile();
  if (!TU || Diag.hasErrors()) { Err = "parse error in synthesized TU"; return nullptr; }

  matlab::SemaContext Sema;
  matlab::TypeContext TC;
  matlab::Resolver R(Sema, TC, Diag);
  R.resolve(*TU);
  matlab::TypeInference Inf(Sema, TC, Diag);
  Inf.run(*TU);
  if (Diag.hasErrors()) { Err = "sema error in synthesized TU"; return nullptr; }

  // Fresh MLIR context per handle — keeps lifetime bookkeeping
  // simple (Engine owns the module, Handle owns the context, both
  // live until Release).
  auto MCtx = std::make_unique<matlab::mlirgen::Context>();
  mlir::registerBuiltinDialectTranslation(MCtx->get());
  mlir::registerLLVMDialectTranslation(MCtx->get());

  auto M = matlab::mlirgen::lowerToMLIR(*MCtx, TC, Diag, *TU,
                                        &SM, /*ReplMode=*/false);
  if (Diag.hasErrors()) { Err = "MLIR lowering failed"; return nullptr; }

  if (!runPipeline(M, Err)) return nullptr;

  // JIT.
  mlir::ExecutionEngineOptions Opts;
  Opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;
  auto EngineOrErr = mlir::ExecutionEngine::create(M, Opts);
  if (!EngineOrErr) {
    Err = "ExecutionEngine::create failed: "
        + llvm::toString(EngineOrErr.takeError());
    return nullptr;
  }
  auto Engine = std::move(*EngineOrErr);

  auto AddrOrErr = Engine->lookup("mflowlink_jit_entry");
  if (!AddrOrErr) {
    Err = "lookup(mflowlink_jit_entry) failed: "
        + llvm::toString(AddrOrErr.takeError());
    return nullptr;
  }

  auto *H = new JitHandle();
  // Note: the Ctx unique_ptr declared in JitHandle uses an incomplete
  // type from MflowLinkSim's POV — that's why this TU defines the
  // struct privately and only exposes a `MatlabFcnJit::Handle *`.
  H->Ctx       = std::move(MCtx);
  H->Engine    = std::move(Engine);
  H->EntryAddr = *AddrOrErr;
  H->NumInputs = NumInputs;
  return reinterpret_cast<MatlabFcnJit::Handle *>(H);
}

double callImpl(MatlabFcnJit::Handle *Opaque, const double *In, unsigned N) {
  auto *H = reinterpret_cast<JitHandle *>(Opaque);
  if (!H || !H->EntryAddr) return 0.0;
  (void)N; // arity captured at compile time; trusted by the caller.
  switch (H->NumInputs) {
  case 0: {
    using FT = double (*)();
    return reinterpret_cast<FT>(H->EntryAddr)();
  }
  case 1: {
    using FT = double (*)(double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0]);
  }
  case 2: {
    using FT = double (*)(double, double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1]);
  }
  case 3: {
    using FT = double (*)(double, double, double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1], In[2]);
  }
  case 4: {
    using FT = double (*)(double, double, double, double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1], In[2], In[3]);
  }
  case 5: {
    using FT = double (*)(double, double, double, double, double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1], In[2], In[3], In[4]);
  }
  case 6: {
    using FT = double (*)(double, double, double, double, double, double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1], In[2], In[3],
                                              In[4], In[5]);
  }
  case 7: {
    using FT = double (*)(double, double, double, double, double, double,
                          double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1], In[2], In[3],
                                              In[4], In[5], In[6]);
  }
  case 8: {
    using FT = double (*)(double, double, double, double, double, double,
                          double, double);
    return reinterpret_cast<FT>(H->EntryAddr)(In[0], In[1], In[2], In[3],
                                              In[4], In[5], In[6], In[7]);
  }
  }
  return 0.0;
}

void releaseImpl(MatlabFcnJit::Handle *Opaque) {
  auto *H = reinterpret_cast<JitHandle *>(Opaque);
  delete H;  // Engine + Ctx both freed via their unique_ptrs.
}

} // namespace

// Installer — main() calls this once before any -simulate path runs.
// Idempotent: replacing the factory mid-process is allowed but
// already-built simulators keep their captured snapshot.
void installMflowLinkJit() {
  // The native target + dialect translations have already been
  // initialised by the REPL / DAP startup paths in matlabc/main.cpp.
  // For pure `-simulate` runs (no REPL, no DAP), the lookup of an
  // LLVM target needs the same one-time init; we replicate it here
  // because main() may not reach the REPL setup before -simulate
  // takes off.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  setMatlabFcnJit({compileImpl, callImpl, releaseImpl});
}

} // namespace matlab::flowchart

#else  // !MATLAB_LLVM_WITH_MLIR

namespace matlab::flowchart {
// MLIR disabled at build time — the JIT is a no-op; MflowLinkSim
// keeps using the AST interpreter on every signal_matlab_fcn block.
void installMflowLinkJit() {}
} // namespace matlab::flowchart

#endif
