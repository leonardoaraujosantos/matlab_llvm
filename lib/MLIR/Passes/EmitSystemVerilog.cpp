// Emits synthesizable SystemVerilog (ASIC target) from an MLIR ModuleOp
// whose ops have been driven through the same lowering pipeline as
// `-emit-c` plus the SV-specific gate (HWLegalize / HWBitWidthInfer).
//
// Phase 1 scope: scalar combinational only — one `module` per user
// `func.func`, one `always_comb` body, ports named after MATLAB
// parameter / result names, integer datapath only (i1 / i8 / i16 /
// i32 / i64). Registers, FSMs, and RAM inference are later phases.
//
// IR shape consumed (matches what EmitC.cpp consumes after LowerIO +
// IfStoreToSelect + Mem2RegLite):
//
//   func.func body ::= seq of (
//       arith.constant
//     | arith.{add,sub,mul,divs,divu,rems,remu,andi,ori,xori,
//              shli,shrsi,shrui}i
//     | arith.cmpi
//     | arith.select
//     | arith.{extsi,extui,trunci}
//     | scf.if (with or without results)
//     | scf.yield
//     | llvm.alloca / llvm.load / llvm.store    -- scalar slots
//     | func.return
//   )
//
// Anything outside this set causes a hard `error: emit-sv:` diagnostic
// rather than silent passthrough. Hardware emission has zero tolerance
// for silent fallback.

#include "matlab/MLIR/Passes/Passes.h"
#include "matlab/Basic/SourceManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

class Emitter {
public:
  Emitter(std::ostream &OS, const matlab::SourceManager *SM,
          HWResetKind Reset, HWFSMEncoding FSMEnc)
      : OS(OS), SM(SM), Reset(Reset), FSMEnc(FSMEnc) {
    (void)this->SM;
  }

  bool run(mlir::ModuleOp M);

private:
  // --- Top-level ---------------------------------------------------------
  bool emitModuleForFunc(mlir::func::FuncOp F);
  void emitProlog();

  // --- Naming ------------------------------------------------------------
  // Stable identifier for an SSA value. Looks up an explicit MATLAB
  // name first (matlab.name attr on the producing op or block-arg
  // attr), falls back to a fresh `vN`. A given Value always maps to
  // the same string within one Emitter invocation.
  std::string name(mlir::Value V);
  std::string freshName(const char *Prefix = "v");
  std::string sanitize(llvm::StringRef In);

  // --- Type rendering ----------------------------------------------------
  // Width of an integer MLIR type. Caller has already verified that T
  // is one of i1 / i8 / i16 / i32 / i64 (HWBitWidthInfer enforces).
  unsigned widthOf(mlir::Type T);
  // Render a SystemVerilog port / signal type for an MLIR integer
  // type. `Signed` controls the explicit `signed` qualifier — Phase 1
  // defaults all multi-bit integer ports to `signed` so MATLAB's
  // signed comparisons (`<`, `>=`, ...) lower to SV's signed
  // comparisons without `$signed(...)` wrappers. `i1` always renders
  // as bare `logic` (no width, no sign).
  std::string svType(mlir::Type T, bool Signed = true);

  // --- Body emission -----------------------------------------------------
  void emitBody(mlir::func::FuncOp F);
  void emitAlwaysFF();
  void emitFSMTypedefs();
  void emitPipelineDecls(mlir::func::FuncOp F);
  void emitPipelineFF();
  void emitFSMCase(mlir::scf::IfOp Head, int Indent);
  /// Phase 4 v2 — FSM cascade detection. Walks every `scf.if`
  /// whose condition is `cmpf oeq, persistent_get, const` and
  /// peels the else-chain to collect a list of `(case_const,
  /// then_region)` pairs. Heads (top-of-cascade ifs) are stored
  /// in `FSMs`; inner cascade ifs are added to `Suppress` and
  /// `CascadeInner` so the regular emit dispatch skips them.
  void gatherFSMs(mlir::func::FuncOp F);
  void emitOp(mlir::Operation &Op, int Indent);
  void emitRegion(mlir::Region &R, int Indent);
  void emitBlock(mlir::Block &B, int Indent);
  void declarePrelude(mlir::func::FuncOp F);

  // --- Op handlers -------------------------------------------------------
  void emitArithConstant(mlir::arith::ConstantOp C, int Indent);
  void emitBinop(mlir::Operation &Op, llvm::StringRef SvOp, int Indent);
  void emitUnaryNeg(mlir::Operation &Op, int Indent);
  void emitCmp(mlir::arith::CmpIOp C, int Indent);
  void emitCmpF(mlir::arith::CmpFOp C, int Indent);
  void emitSelect(mlir::arith::SelectOp S, int Indent);
  void emitExtTrunc(mlir::Operation &Op, int Indent);
  void emitScfIf(mlir::scf::IfOp If, int Indent);
  void emitScfYield(mlir::scf::YieldOp Y, int Indent);
  void emitScfWhile(mlir::scf::WhileOp W, int Indent);
  void emitAlloca(mlir::LLVM::AllocaOp A, int Indent);
  void emitLoad(mlir::LLVM::LoadOp L, int Indent);
  void emitStore(mlir::LLVM::StoreOp S, int Indent);
  void emitGEP(mlir::LLVM::GEPOp G, int Indent);
  void emitReturn(mlir::func::ReturnOp R, int Indent);

  // --- Helpers -----------------------------------------------------------
  void indent(int N) { for (int i = 0; i < N; ++i) OS << "    "; }
  void fail(llvm::StringRef Msg) {
    if (!Failed)
      std::cerr << "error: emit-sv: " << Msg.str() << "\n";
    Failed = true;
  }
  // Render an SV literal for an integer constant. Width comes from V's
  // type; signedness flag controls the `'sd` vs `'d` suffix and
  // negative-value rendering.
  std::string intLiteral(int64_t Val, mlir::Type T, bool Signed);
  // Render the expression form of V. For an SSA value with an explicit
  // name slot, returns the name. For unnamed pure ops (constants,
  // inline-friendly arith), returns the inline expression. The returned
  // string is parenthesized when the surrounding precedence requires it.
  std::string exprFor(mlir::Value V);
  // Comma-separated SV port-list lines for the function. Output ports
  // come last so callers reading the module declaration see "inputs ...
  // -> outputs". Result names are taken from the function's `result_names`
  // attribute when present, otherwise `y`, `y1`, `y2`, ... .
  void emitPortList(mlir::func::FuncOp F);
  // True when V is the result of an op the emitter inlines at use site.
  bool isInlineable(mlir::Value V);
  /// Phase 4 v2 — when one operand of a comparison or assignment
  /// is a recognized FSM register (persistent_get whose RegIndex
  /// is in FSMs), render its peer constant as the matching enum
  /// literal instead of a raw integer literal. Returns the enum
  /// literal name on match, empty string on miss.
  std::string fsmEnumLiteralForConstAgainst(mlir::Value RegSide,
                                             mlir::Value ConstSide);

  // --- State -------------------------------------------------------------
  std::ostream &OS;
  const matlab::SourceManager *SM;
  HWResetKind Reset;
  HWFSMEncoding FSMEnc;
  llvm::DenseMap<mlir::Value, std::string> Names;
  llvm::StringSet<> Used;
  unsigned NextFresh = 0;
  bool Failed = false;
  // Per-function: argument names (parallel to F.getArguments()).
  std::vector<std::string> ArgNames;
  // Per-function: output names (parallel to F's result types).
  // OutNames is the SV port name (`y`, `y1`, ...) that the
  // module declaration uses. OutWriteNames is the name the
  // always_comb body writes to: same as OutNames when there's
  // no output pipelining; `<port>_d0` when output_pipeline > 0
  // so the body feeds the pipeline-register chain instead of
  // the port directly.
  std::vector<std::string> OutNames;
  std::vector<std::string> OutWriteNames;
  // Per-function: SSA values that should be declared as `logic` at
  // module scope. Filled by declarePrelude.
  std::vector<mlir::Value> PreludeDecls;
  // Per-function: persistent registers (Phase 3). Empty for stateless
  // functions. Each entry carries the recognized get/set sites the
  // emitter routes through register signals.
  llvm::SmallVector<HWPersistentInfo, 4> Persists;
  // Quick lookup: get-call op → index into Persists. The result of
  // each recognized get_f64 renders as the register's current-value
  // signal name.
  llvm::DenseMap<mlir::Operation *, unsigned> GetSiteToReg;
  // Quick lookup: set-call op → index into Persists. Each renders as
  // an assignment to the register's `_next` signal.
  llvm::DenseMap<mlir::Operation *, unsigned> SetSiteToReg;
  // Ops the emitter must skip during always_comb body emission. Used
  // to suppress the isempty-guarded scf.if (its init becomes the
  // reset value) and the cmpf+isempty trio that feeds it.
  llvm::DenseSet<mlir::Operation *> Suppress;
  // Phase 4.5.4: GEP ops that resolve to `arr_name[idx_expr]`. The
  // load/store consuming the GEP renders the indexed access
  // directly. The map's value is the SV expression (e.g.
  // `"v[2]"`).
  llvm::DenseMap<mlir::Operation *, std::string> GepAddr;
  // Phase 5.2 v1: per-function port-pipeline stage counts read
  // from `hdl.input_pipeline` / `hdl.output_pipeline` pragma
  // attributes. 0 means "no pipelining on that side". When
  // either is non-zero, the emitter routes the always_comb
  // body through internal pre-/post-pipeline signals and adds
  // a dedicated always_ff that shifts the register chain.
  int InputPipelineN = 0;
  int OutputPipelineN = 0;

  // Phase 4 v2: FSM cascades recognized in the function body.
  // Each entry is one cascade — the head scf.if op, the persistent
  // register's index in `Persists`, the per-case (const, region)
  // pairs in source order, and the optional default else-region.
  struct HWFSMInfo {
    unsigned RegIndex;
    mlir::Operation *Head = nullptr;
    llvm::SmallVector<std::pair<int64_t, mlir::Region *>, 6> Cases;
    mlir::Region *DefaultRegion = nullptr;
    // SV-side enum-literal names per case const, plus the reset
    // literal for the always_ff. Filled at gather time.
    llvm::SmallVector<std::string, 6> CaseNames;
    std::string ResetName;
    // SV state-register-type name (e.g. "state_t").
    std::string EnumType;
  };
  llvm::SmallVector<HWFSMInfo, 2> FSMs;
  // Map from any cascade scf.if op (head OR inner) to the FSM
  // index. Inner ifs are also added to `Suppress` so they don't
  // get rendered twice.
  llvm::DenseMap<mlir::Operation *, unsigned> CascadeOp;
};

unsigned Emitter::widthOf(mlir::Type T) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T))
    return IT.getWidth();
  return 0;
}

std::string Emitter::svType(mlir::Type T, bool Signed) {
  unsigned W = widthOf(T);
  if (W == 0) return "/* unknown */";
  if (W == 1) return "logic";
  std::ostringstream S;
  S << "logic";
  if (Signed) S << " signed";
  S << " [" << (W - 1) << ":0]";
  return S.str();
}

std::string Emitter::sanitize(llvm::StringRef In) {
  std::string Out;
  Out.reserve(In.size());
  for (char c : In) {
    if (std::isalnum((unsigned char)c) || c == '_')
      Out.push_back(c);
    else
      Out.push_back('_');
  }
  if (Out.empty() || std::isdigit((unsigned char)Out[0]))
    Out.insert(Out.begin(), '_');
  return Out;
}

std::string Emitter::freshName(const char *Prefix) {
  while (true) {
    std::string Cand = std::string(Prefix) + std::to_string(NextFresh++);
    if (Used.insert(Cand).second) return Cand;
  }
}

std::string Emitter::name(mlir::Value V) {
  auto It = Names.find(V);
  if (It != Names.end()) return It->second;
  // Try the producing op's `matlab.name` attr.
  std::string Cand;
  if (auto *Op = V.getDefiningOp()) {
    if (auto S = Op->getAttrOfType<mlir::StringAttr>("matlab.name"))
      Cand = sanitize(S.getValue());
    if (Cand.empty())
      if (auto S = Op->getAttrOfType<mlir::StringAttr>("name"))
        Cand = sanitize(S.getValue());
  }
  if (Cand.empty()) Cand = freshName();
  // Disambiguate against the live identifier set.
  if (Used.contains(Cand)) {
    std::string Base = Cand;
    unsigned I = 1;
    while (Used.contains(Cand))
      Cand = Base + "_" + std::to_string(I++);
  }
  Used.insert(Cand);
  Names[V] = Cand;
  return Cand;
}

std::string Emitter::intLiteral(int64_t Val, mlir::Type T, bool Signed) {
  unsigned W = widthOf(T);
  std::ostringstream S;
  if (W == 1) {
    S << (Val ? "1'b1" : "1'b0");
    return S.str();
  }
  if (Signed) {
    // SystemVerilog grammar: a sized literal cannot embed the sign in
    // its value digits (`16'sd-32000` is a syntax error). Emit
    // negative values as `-16'sd<abs>`. Special-case INT64_MIN which
    // has no positive representation in the same width.
    if (Val < 0) {
      uint64_t Abs = (Val == std::numeric_limits<int64_t>::min())
                         ? uint64_t(1) << 63
                         : (uint64_t)(-Val);
      S << "-" << W << "'sd" << Abs;
    } else {
      S << W << "'sd" << Val;
    }
  } else {
    // Mask to width to avoid sign-extension surprises.
    uint64_t Mask = (W >= 64) ? ~uint64_t(0) : ((uint64_t(1) << W) - 1);
    S << W << "'d" << ((uint64_t)Val & Mask);
  }
  return S.str();
}

bool Emitter::isInlineable(mlir::Value V) {
  // Phase 1 takes the predictable-RTL path: every named producer gets
  // a top-level declaration and every op writes its result by name.
  // Inlining is reserved for a later quality pass once the golden
  // tests pin the verbose form down.
  (void)V;
  return false;
}

std::string Emitter::exprFor(mlir::Value V) {
  if (auto *Op = V.getDefiningOp()) {
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
        bool Signed = true;
        // i1 is unsigned-style.
        if (auto IT = mlir::dyn_cast<mlir::IntegerType>(C.getType()))
          Signed = (IT.getWidth() != 1);
        return intLiteral(IA.getInt(), C.getType(), Signed);
      }
    }
    // Phase 4.5.4: `llvm.mlir.constant` (used as the GEP index in
    // the static-fi-array lowering) renders as a plain integer
    // literal so `<arr>[<idx>]` falls out cleanly. Indices are
    // unsized so SV self-determined-width rules pick the right
    // shape.
    if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Op)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
        std::ostringstream S; S << IA.getInt(); return S.str();
      }
    }
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
      // The frontend emits MATLAB switch-case labels as `f64`
      // constants regardless of the discriminator's actual integer
      // type, producing comparisons like `arith.cmpi i8, f64`. Render
      // the f64 here as an unsized SV integer literal — SV's
      // self-determined-width rules promote it to match the integer
      // side. (We accept `widthwarn` from Verilator on this rather
      // than guess a width without surrounding-op context.)
      if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
        double D = FA.getValueAsDouble();
        int64_t I = (int64_t)D;
        if ((double)I == D) {
          std::ostringstream S; S << I; return S.str();
        }
      }
    }
    // Phase 3: a persistent-get call result reads as the register's
    // current-value signal, regardless of the get's declared f64 ABI
    // type. The synth tool / SV semantics handle implicit
    // sign-extension into wider expressions.
    auto It = GetSiteToReg.find(Op);
    if (It != GetSiteToReg.end())
      return Persists[It->second].Name;
  }
  return name(V);
}

void Emitter::emitPortList(mlir::func::FuncOp F) {
  auto FT = F.getFunctionType();
  // Reserve the synthesized clock + reset names BEFORE arg/output
  // name resolution so a user arg named `rst` / `rst_n` / `clk` gets
  // suffixed (`rst_`) and doesn't shadow the system port. Phase 5.2
  // adds port-pipeline registers as another reason to need a clock.
  bool NeedsClock = !Persists.empty() ||
                    InputPipelineN > 0 || OutputPipelineN > 0;
  if (NeedsClock) {
    Used.insert("clk");
    if (Reset == HWResetKind::AsyncLow || Reset == HWResetKind::SyncLow)
      Used.insert("rst_n");
    else
      Used.insert("rst");
  }
  // Argument names: prefer matlab.name on the entry-block arg
  // (lowering attaches it via `function_arg_name`), fall back to
  // `arg<i>`.
  ArgNames.clear();
  ArgNames.reserve(FT.getNumInputs());
  for (unsigned I = 0; I < FT.getNumInputs(); ++I) {
    std::string Nm;
    if (auto S = F.getArgAttrOfType<mlir::StringAttr>(I, "matlab.name"))
      Nm = sanitize(S.getValue());
    if (Nm.empty()) Nm = "arg" + std::to_string(I);
    while (Used.contains(Nm)) Nm += "_";
    Used.insert(Nm);
    ArgNames.push_back(Nm);
    // Phase 5.2: when input_pipeline > 0, route the body's
    // references through the last-stage pipeline register
    // (`<arg>_dN`). The port itself stays at `<arg>`; the
    // input-pipeline always_ff drives the register chain.
    if (InputPipelineN > 0) {
      Names[F.getArgument(I)] = Nm + "_d" + std::to_string(InputPipelineN);
    } else {
      Names[F.getArgument(I)] = Nm;
    }
  }
  // Output names. Phase 1 looks for an `sv.result_names` array attr
  // (an extension point for later) and otherwise uses `y`, `y1`, ...
  OutNames.clear();
  OutNames.reserve(FT.getNumResults());
  for (unsigned I = 0; I < FT.getNumResults(); ++I) {
    std::string Nm = (I == 0) ? "y" : ("y" + std::to_string(I));
    while (Used.contains(Nm)) Nm += "_";
    Used.insert(Nm);
    OutNames.push_back(Nm);
  }

  // Print the port list. Phase 3: prepend clk + reset port when the
  // function has any persistent state. Phase 5.2: also when port
  // pipelining is on.
  bool First = true;
  if (NeedsClock) {
    OS << "    input  logic clk";
    First = false;
    if (Reset == HWResetKind::AsyncLow || Reset == HWResetKind::SyncLow) {
      OS << ",\n    input  logic rst_n";
    } else {
      OS << ",\n    input  logic rst";
    }
  }
  for (unsigned I = 0; I < FT.getNumInputs(); ++I) {
    if (!First) OS << ",\n";
    First = false;
    OS << "    input  " << svType(FT.getInput(I)) << " " << ArgNames[I];
  }
  for (unsigned I = 0; I < FT.getNumResults(); ++I) {
    if (!First) OS << ",\n";
    First = false;
    // For results that come from a persistent get, the function's
    // declared result type is the runtime ABI's f64 — render the SV
    // port at the register's actual integer width instead.
    mlir::Type T = FT.getResult(I);
    // Find a func.return op and inspect its operand[I]; if it's a
    // recognized persistent get, use that register's width.
    F.walk([&](mlir::func::ReturnOp R) {
      if (R.getNumOperands() <= I) return;
      auto *Op = R.getOperand(I).getDefiningOp();
      if (!Op) return;
      auto It = GetSiteToReg.find(Op);
      if (It == GetSiteToReg.end()) return;
      auto &P = Persists[It->second];
      T = mlir::IntegerType::get(F.getContext(), P.Width);
    });
    OS << "    output " << svType(T) << " " << OutNames[I];
  }
  OS << "\n";

  // Phase 5.2: after the port list is printed, populate
  // OutWriteNames — what the always_comb body actually writes
  // to. With no pipeline this is just OutNames; with
  // output_pipeline > 0 it's `<port>_d0` so the body feeds the
  // pipeline-register chain that drives the port.
  OutWriteNames = OutNames;
  if (OutputPipelineN > 0) {
    for (auto &N : OutWriteNames) N += "_d0";
  }
}

void Emitter::declarePrelude(mlir::func::FuncOp F) {
  // Walk the body and collect every SSA value that needs a `logic`
  // declaration. Skip block-arguments of the entry block (those are
  // ports); skip values produced by ops we render inline (constants);
  // skip values produced by `llvm.alloca` (they're slot addresses,
  // not signals — the slot's element type defines the actual signal
  // we declare).
  PreludeDecls.clear();
  // Map alloca SSA value -> its declared signal name. The corresponding
  // load/store ops will reuse that name.
  // Actually the names map already handles this — we just need to make
  // sure we *declare* the slot name with its element type, and that
  // load/store route through the same name.

  llvm::DenseMap<mlir::Value, mlir::Type> SlotElemTy;
  // Helper: true when Op or any ancestor is on the suppression list.
  // We use this to skip ops that live inside the isempty if-guard
  // (the init set call in particular — its result type is `none`,
  // and the prelude has nothing meaningful to declare for it).
  auto IsSuppressedOrInside = [&](mlir::Operation *Op) {
    for (mlir::Operation *Cur = Op; Cur; Cur = Cur->getParentOp())
      if (Suppress.contains(Cur)) return true;
    return false;
  };
  // Pre-pass — find single-use fi-tagged binops whose only consumer
  // is a recognized persistent set; mark them suppressed so the
  // prelude doesn't declare their wide intermediate values. The set
  // site emits the binop's expression inline. Doing this here, before
  // the main walk, keeps `Suppress` consistent for both the prelude
  // and the body.
  for (auto &P : Persists) {
    for (mlir::Operation *Set : P.Sets) {
      if (Set->getNumOperands() < 2) continue;
      mlir::Value Val = Set->getOperand(1);
      auto *VOp = Val.getDefiningOp();
      if (!VOp) continue;
      llvm::StringRef N = VOp->getName().getStringRef();
      bool IsArith = (N == "matlab.add" || N == "matlab.sub" ||
                      N == "matlab.matmul" || N == "matlab.emul");
      if (IsArith && Val.hasOneUse()) Suppress.insert(VOp);
    }
  }
  F.walk([&](mlir::Operation *Op) {
    // Phase 3 — suppress declaration of values whose producer is
    // routed to a register signal (persistent get) or whose op is on
    // the suppression list (or nested inside one).
    if (IsSuppressedOrInside(Op)) return;
    if (GetSiteToReg.contains(Op)) return;
    if (SetSiteToReg.contains(Op)) return;
    if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Op)) {
      // Slot signal name comes from the alloca's `name` attr.
      mlir::Value Slot = A.getResult();
      // Force an entry in Names so loads/stores share it.
      (void)name(Slot);
      SlotElemTy[Slot] = A.getElemType();
      // We'll emit the declaration when we encounter the alloca during
      // body emission, but it's cleaner to declare slots up front.
      PreludeDecls.push_back(Slot);
      return;
    }
    // `llvm.store`, `scf.yield`, `func.return` produce no SSA result,
    // so nothing to declare. `llvm.load` *does* produce an SSA result
    // (the loaded value) and falls through to the generic decl path.
    if (mlir::isa<mlir::LLVM::StoreOp, mlir::scf::YieldOp,
                  mlir::func::ReturnOp>(Op))
      return;
    // Phase 4.5.4: `llvm.getelementptr` produces an `!llvm.ptr` that
    // we never reify as a named SV signal — the load/store consumer
    // renders the indexed access expression directly via the
    // GepAddr side-table.
    if (mlir::isa<mlir::LLVM::GEPOp>(Op))
      return;
    // For-loop control-flow ops produce structural results (the
    // for-iv, the cmpf result, the addf next-value) that the SV
    // emitter renders as part of the SV `for (int i = ...)` head and
    // the loop pattern. They never become datapath signals.
    //
    // arith.cmpf and arith.addf, however, ALSO appear in datapath
    // contexts after Phase 4 lands FSM emission: the state-equality
    // check `switch (st) case 0: ...` lowers via the runtime ABI
    // through `arith.cmpf oeq, get_f64(st), 0.0 : f64` whose result
    // is a real datapath i1. Skip these ops only when they're part
    // of a recognized for-loop pattern (i.e., they live in the
    // before-region's terminator chain or the after-region's tail).
    auto IsForLoopStructural = [](mlir::Operation *Op) {
      mlir::Operation *Parent = Op->getParentOp();
      if (!Parent) return false;
      auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Parent);
      if (!W) return false;
      // Cmpf used by scf.condition (the for-loop's terminator) is
      // structural; cmpf elsewhere is datapath.
      if (mlir::isa<mlir::arith::CmpFOp>(Op)) {
        for (mlir::OpOperand &U : Op->getResult(0).getUses())
          if (mlir::isa<mlir::scf::ConditionOp>(U.getOwner()))
            return true;
        return false;
      }
      // Addf used by scf.yield (after-region tail) is structural.
      if (mlir::isa<mlir::arith::AddFOp>(Op)) {
        for (mlir::OpOperand &U : Op->getResult(0).getUses())
          if (mlir::isa<mlir::scf::YieldOp>(U.getOwner()))
            return true;
        return false;
      }
      return false;
    };
    if (mlir::isa<mlir::scf::WhileOp, mlir::scf::ConditionOp>(Op))
      return;
    if (mlir::isa<mlir::arith::CmpFOp, mlir::arith::AddFOp>(Op) &&
        IsForLoopStructural(Op))
      return;
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
      // Constants render inline; no prelude entry.
      (void)C;
      return;
    }
    if (mlir::isa<mlir::LLVM::ConstantOp>(Op)) {
      // `llvm.mlir.constant` is consumed as an alloca size operand —
      // never appears in the datapath, so no prelude entry.
      return;
    }
    for (mlir::Value V : Op->getResults()) {
      // Reserve a name now so it appears in the prelude.
      (void)name(V);
      PreludeDecls.push_back(V);
    }
  });

  for (mlir::Value V : PreludeDecls) {
    auto It = SlotElemTy.find(V);
    mlir::Type T = (It != SlotElemTy.end()) ? It->second : V.getType();
    // Phase 4.5.4: array element type → `logic [W-1:0] arr [N];`.
    if (auto Arr = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(T)) {
      OS << "    " << svType(Arr.getElementType()) << " "
         << name(V) << " [" << Arr.getNumElements() << "];\n";
      continue;
    }
    OS << "    " << svType(T) << " " << name(V) << ";\n";
  }
  // Phase 3: declare a `<reg>` and `<reg>_next` pair for every
  // persistent register. The register signal carries the current
  // (clocked) value; the `_next` signal carries the combinational
  // next-state expression that always_ff samples on the next edge.
  // Phase 4 v2 — when a register has been recognized as an FSM
  // state (its index appears in FSMs[].RegIndex), declare both
  // signals at the typedef-enum type instead of the raw `logic
  // [W-1:0]`. The synth tool sees state encoding intent and the
  // user-facing source reads cleanly.
  for (unsigned R = 0; R < Persists.size(); ++R) {
    auto &P = Persists[R];
    std::string Ty;
    bool IsFSM = false;
    for (auto &FI : FSMs) {
      if (FI.RegIndex == R) { Ty = FI.EnumType; IsFSM = true; break; }
    }
    if (!IsFSM) {
      auto T = mlir::IntegerType::get(F.getContext(), P.Width);
      Ty = svType(T);
    }
    OS << "    " << Ty << " " << P.Name << ";\n";
    OS << "    " << Ty << " " << P.Name << "_next;\n";
  }
  if (!PreludeDecls.empty() || !Persists.empty()) OS << "\n";
}

void Emitter::emitArithConstant(mlir::arith::ConstantOp C, int Indent) {
  // Constants are rendered inline at use site by exprFor. Emit nothing.
  (void)C; (void)Indent;
}

void Emitter::emitBinop(mlir::Operation &Op, llvm::StringRef SvOp,
                        int Indent) {
  if (Op.getNumOperands() != 2 || Op.getNumResults() != 1) {
    fail("binop with unexpected arity");
    return;
  }
  indent(Indent);
  OS << name(Op.getResult(0)) << " = "
     << exprFor(Op.getOperand(0)) << " " << SvOp.str() << " "
     << exprFor(Op.getOperand(1)) << ";\n";
}

void Emitter::emitUnaryNeg(mlir::Operation &Op, int Indent) {
  // arith doesn't have a `negi`; the lowering emits `subi 0, x` or
  // `muli x, -1`. We don't see a direct neg in the IR. Reserved for
  // future use.
  (void)Op; (void)Indent;
  fail("emitUnaryNeg called unexpectedly");
}

std::string Emitter::fsmEnumLiteralForConstAgainst(mlir::Value RegSide,
                                                    mlir::Value ConstSide) {
  auto *Op = RegSide.getDefiningOp();
  if (!Op) return std::string();
  auto It = GetSiteToReg.find(Op);
  if (It == GetSiteToReg.end()) return std::string();
  unsigned RegIdx = It->second;
  const HWFSMInfo *Fsm = nullptr;
  for (auto &FI : FSMs) {
    if (FI.RegIndex == RegIdx) { Fsm = &FI; break; }
  }
  if (!Fsm) return std::string();
  auto C = ConstSide.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return std::string();
  int64_t V;
  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
    V = IA.getInt();
  } else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
    V = (int64_t)FA.getValueAsDouble();
  } else {
    return std::string();
  }
  for (size_t i = 0; i < Fsm->Cases.size(); ++i)
    if (Fsm->Cases[i].first == V) return Fsm->CaseNames[i];
  return std::string();
}

void Emitter::emitCmp(mlir::arith::CmpIOp C, int Indent) {
  llvm::StringRef SvOp;
  switch (C.getPredicate()) {
  case mlir::arith::CmpIPredicate::eq:  SvOp = "==";  break;
  case mlir::arith::CmpIPredicate::ne:  SvOp = "!=";  break;
  case mlir::arith::CmpIPredicate::slt: SvOp = "<";   break;
  case mlir::arith::CmpIPredicate::sle: SvOp = "<=";  break;
  case mlir::arith::CmpIPredicate::sgt: SvOp = ">";   break;
  case mlir::arith::CmpIPredicate::sge: SvOp = ">=";  break;
  case mlir::arith::CmpIPredicate::ult: SvOp = "<";   break;
  case mlir::arith::CmpIPredicate::ule: SvOp = "<=";  break;
  case mlir::arith::CmpIPredicate::ugt: SvOp = ">";   break;
  case mlir::arith::CmpIPredicate::uge: SvOp = ">=";  break;
  }
  indent(Indent);
  std::string LExpr = exprFor(C.getLhs());
  std::string RExpr = exprFor(C.getRhs());
  if (auto E = fsmEnumLiteralForConstAgainst(C.getLhs(), C.getRhs());
      !E.empty()) {
    RExpr = E;
  } else if (auto E = fsmEnumLiteralForConstAgainst(C.getRhs(), C.getLhs());
             !E.empty()) {
    LExpr = E;
  }
  OS << name(C.getResult()) << " = " << LExpr << " " << SvOp.str() << " "
     << RExpr << ";\n";
}

void Emitter::emitCmpF(mlir::arith::CmpFOp C, int Indent) {
  // Phase 4: arith.cmpf surfaces in the FSM lowering — `switch (st)
  // case <const>` becomes a chain of `arith.cmpf oeq, get_f64(st),
  // <case_const>`. The LHS is a recognized persistent get (routed
  // through `exprFor` to the register signal name) and the RHS is
  // an integer-valued f64 constant (rendered as a plain integer
  // literal via the same `exprFor` path used for switch case
  // labels). All ordered/unordered variants pick the same SV
  // operator since unordered makes no sense on integer-shaped data.
  llvm::StringRef SvOp;
  switch (C.getPredicate()) {
  case mlir::arith::CmpFPredicate::OEQ:
  case mlir::arith::CmpFPredicate::UEQ: SvOp = "=="; break;
  case mlir::arith::CmpFPredicate::ONE:
  case mlir::arith::CmpFPredicate::UNE: SvOp = "!="; break;
  case mlir::arith::CmpFPredicate::OLT:
  case mlir::arith::CmpFPredicate::ULT: SvOp = "<"; break;
  case mlir::arith::CmpFPredicate::OLE:
  case mlir::arith::CmpFPredicate::ULE: SvOp = "<="; break;
  case mlir::arith::CmpFPredicate::OGT:
  case mlir::arith::CmpFPredicate::UGT: SvOp = ">"; break;
  case mlir::arith::CmpFPredicate::OGE:
  case mlir::arith::CmpFPredicate::UGE: SvOp = ">="; break;
  default:
    fail("unsupported arith.cmpf predicate (only ord/unord eq/ne/lt/le/gt/ge)");
    return;
  }
  indent(Indent);
  // FSM-aware: if the LHS is a recognized persistent get for an
  // FSM register and the RHS is a const matching one of the case
  // values, render `<state> == S<n>` instead of `<state> == 8'sdN`.
  std::string LExpr = exprFor(C.getLhs());
  std::string RExpr = exprFor(C.getRhs());
  if (auto E = fsmEnumLiteralForConstAgainst(C.getLhs(), C.getRhs());
      !E.empty()) {
    RExpr = E;
  } else if (auto E = fsmEnumLiteralForConstAgainst(C.getRhs(), C.getLhs());
             !E.empty()) {
    LExpr = E;
  }
  OS << name(C.getResult()) << " = "
     << LExpr << " " << SvOp.str() << " " << RExpr << ";\n";
}

void Emitter::emitSelect(mlir::arith::SelectOp S, int Indent) {
  indent(Indent);
  OS << name(S.getResult()) << " = "
     << exprFor(S.getCondition()) << " ? "
     << exprFor(S.getTrueValue()) << " : "
     << exprFor(S.getFalseValue()) << ";\n";
}

void Emitter::emitExtTrunc(mlir::Operation &Op, int Indent) {
  // SystemVerilog: extending or truncating a packed integer is a
  // direct width cast. We emit `<W>'($signed(x))` for sign-ext,
  // `<W>'(x)` for zero-ext / truncate.
  bool Signed = mlir::isa<mlir::arith::ExtSIOp>(Op);
  unsigned W = widthOf(Op.getResult(0).getType());
  indent(Indent);
  OS << name(Op.getResult(0)) << " = " << W << "'(";
  if (Signed) OS << "$signed(";
  OS << exprFor(Op.getOperand(0));
  if (Signed) OS << ")";
  OS << ");\n";
}

void Emitter::emitScfIf(mlir::scf::IfOp If, int Indent) {
  // Phase 1 supports both shapes:
  //   - scf.if without results: pure side-effecting branches that store
  //     to slots. Renders as `if (cond) begin ... end else begin ... end`
  //     inside `always_comb`.
  //   - scf.if with results: the values yielded by each arm assign the
  //     `if`'s SSA results. Renders as the same construct, with each
  //     arm writing the result name(s).
  indent(Indent);
  OS << "if (" << exprFor(If.getCondition()) << ") begin\n";
  emitRegion(If.getThenRegion(), Indent + 1);
  // The else region is "empty" in MLIR terms when no false-branch was
  // written. MLIR still synthesizes a single block containing just an
  // implicit `scf.yield` for the no-result form, so `getElseRegion()`
  // is non-empty even for `if cond { body }`. Skip the else-branch
  // emission when its only op is the auto-yield.
  bool ElseIsEmpty = If.getElseRegion().empty();
  if (!ElseIsEmpty) {
    auto &EB = If.getElseRegion().front();
    auto NumOps = std::distance(EB.begin(), EB.end());
    if (NumOps == 1 && mlir::isa<mlir::scf::YieldOp>(EB.front()) &&
        EB.front().getNumOperands() == 0)
      ElseIsEmpty = true;
  }
  if (!ElseIsEmpty) {
    indent(Indent);
    OS << "end else begin\n";
    emitRegion(If.getElseRegion(), Indent + 1);
  }
  indent(Indent);
  OS << "end\n";
}

void Emitter::emitScfYield(mlir::scf::YieldOp Y, int Indent) {
  // Yield can come from inside an scf.if (assign parent results) or
  // from inside the after-region of a recognized for-loop (no datapath
  // effect — the addf step is part of the loop pattern). The latter is
  // skipped silently.
  auto *Parent = Y->getParentOp();
  if (mlir::isa<mlir::scf::WhileOp>(Parent)) {
    // Loop yield. The matched for-loop pattern absorbs the addf+yield
    // into the for-head; nothing to emit here.
    return;
  }
  auto If = mlir::dyn_cast<mlir::scf::IfOp>(Parent);
  if (!If) {
    if (Y.getNumOperands() != 0)
      fail("scf.yield outside scf.if not supported in Phase 1");
    return;
  }
  if (Y.getNumOperands() != If->getNumResults()) {
    fail("scf.yield arity mismatch");
    return;
  }
  for (unsigned I = 0; I < Y.getNumOperands(); ++I) {
    indent(Indent);
    OS << name(If->getResult(I)) << " = "
       << exprFor(Y.getOperand(I)) << ";\n";
  }
}

void Emitter::emitScfWhile(mlir::scf::WhileOp W, int Indent) {
  // Phase 2: only the canonical bounded for-loop shape is accepted —
  // HWLegalize already rejected everything else with a precise
  // diagnostic. Reach for the same matcher, render as a synthesizable
  // SV `for` with an integer counter. ASIC synthesis tools fully
  // unroll constant-bound for-loops inside `always_comb`; the explicit
  // for-form keeps the source readable and the emitter simple.
  HWForLoopInfo Info;
  if (!matchHWForLoop(W, Info)) {
    fail("scf.while did not match canonical for-loop pattern at "
         "emit time (HWLegalize should have rejected earlier)");
    return;
  }
  auto ToInt = [](mlir::Value V) -> int64_t {
    auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
    if (!C) return 0;
    auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue());
    if (!FA) return 0;
    double D = FA.getValueAsDouble();
    return (int64_t)D;
  };
  int64_t InitV = ToInt(Info.Init);
  int64_t EndV = ToInt(Info.End);
  int64_t StepV = ToInt(Info.Step);
  std::string IvName;
  if (auto *Op = W.getOperation()) {
    if (auto S = Op->getAttrOfType<mlir::StringAttr>("matlab.name"))
      IvName = sanitize(S.getValue());
  }
  if (IvName.empty()) IvName = freshName("i");
  // Avoid colliding with prelude-declared signals.
  while (Used.contains(IvName)) IvName += "_";
  Used.insert(IvName);

  indent(Indent);
  if (Info.IsDecreasing) {
    // For descending ranges (`for i = init:-1:end`) the matched
    // arith.addf is `iv + (negative step)`. Render as
    // `iv = iv - |step|` so the SV head reads naturally.
    int64_t Mag = StepV < 0 ? -StepV : StepV;
    OS << "for (int " << IvName << " = " << InitV << "; "
       << IvName << " >= " << EndV << "; "
       << IvName << " = " << IvName << " - " << Mag
       << ") begin\n";
  } else {
    OS << "for (int " << IvName << " = " << InitV << "; "
       << IvName << " <= " << EndV << "; "
       << IvName << " = " << IvName << " + " << StepV
       << ") begin\n";
  }
  // Emit the after-region's body, but stop before the trailing
  // arith.addf + scf.yield (those are part of the loop pattern and
  // already encoded in the for-head).
  mlir::Block &AB = W.getAfter().front();
  for (mlir::Operation &Op : AB.getOperations()) {
    if (Failed) break;
    // Skip the addf %iv, %step that feeds the yield.
    if (auto Add = mlir::dyn_cast<mlir::arith::AddFOp>(Op)) {
      if (Add.getLhs() == Info.Iv) continue;
    }
    if (mlir::isa<mlir::scf::YieldOp>(Op)) continue;
    emitOp(Op, Indent + 1);
  }
  indent(Indent);
  OS << "end\n";
}

void Emitter::emitAlloca(mlir::LLVM::AllocaOp A, int Indent) {
  // The signal was already declared in the prelude. Nothing to emit
  // inside `always_comb`.
  (void)A; (void)Indent;
}

void Emitter::emitGEP(mlir::LLVM::GEPOp G, int Indent) {
  // Phase 4.5.4: a `getelementptr [N x iW], %arr[0, %idx]` resolves
  // to the SV indexed-access expression `<arr>[<idx>]`. We don't
  // emit a statement for it — instead we record the address-string
  // in `GepAddr` so the consuming load/store renders the indexed
  // access directly.
  (void)Indent;
  if (G.getDynamicIndices().empty()) return;  // unexpected shape
  // Two indices total: the array's "outer" 0 (compile-time) plus
  // the per-element index. Our static-array lowering only ever
  // emits exactly that shape; bail otherwise.
  if (G.getDynamicIndices().size() != 1) return;
  std::string Arr = name(G.getBase());
  std::string Idx = exprFor(G.getDynamicIndices()[0]);
  std::string Expr = Arr + "[" + Idx + "]";
  GepAddr[G.getOperation()] = Expr;
}

void Emitter::emitLoad(mlir::LLVM::LoadOp L, int Indent) {
  // If the address is a GEP we've recorded, render `<arr>[<idx>]`
  // as the source. Otherwise this is a plain slot load.
  std::string AddrExpr;
  if (auto *AddrOp = L.getAddr().getDefiningOp()) {
    auto It = GepAddr.find(AddrOp);
    if (It != GepAddr.end()) AddrExpr = It->second;
  }
  if (AddrExpr.empty()) AddrExpr = name(L.getAddr());
  indent(Indent);
  OS << name(L.getResult()) << " = " << AddrExpr << ";\n";
}

void Emitter::emitStore(mlir::LLVM::StoreOp S, int Indent) {
  std::string AddrExpr;
  if (auto *AddrOp = S.getAddr().getDefiningOp()) {
    auto It = GepAddr.find(AddrOp);
    if (It != GepAddr.end()) AddrExpr = It->second;
  }
  if (AddrExpr.empty()) AddrExpr = name(S.getAddr());
  indent(Indent);
  OS << AddrExpr << " = " << exprFor(S.getValue()) << ";\n";
}

void Emitter::emitReturn(mlir::func::ReturnOp R, int Indent) {
  // Drive the output ports.
  if (R.getNumOperands() != OutNames.size()) {
    fail("func.return arity mismatch");
    return;
  }
  for (unsigned I = 0; I < R.getNumOperands(); ++I) {
    indent(Indent);
    std::string Expr = exprFor(R.getOperand(I));
    // Phase 4 v2: when the return value is a recognized FSM
    // register's get-call result, the rendered expression is the
    // enum-typed register name. The output port stays at the
    // original raw integer width (e.g. `logic signed [7:0] y1`),
    // so we need an explicit width cast — Verilator otherwise
    // flags the assignment as WIDTHEXPAND. The cast width comes
    // from the persistent register's underlying integer width.
    if (auto *Op = R.getOperand(I).getDefiningOp()) {
      auto It = GetSiteToReg.find(Op);
      if (It != GetSiteToReg.end()) {
        auto &P = Persists[It->second];
        bool IsFSM = false;
        for (auto &FI : FSMs)
          if (FI.RegIndex == It->second) { IsFSM = true; break; }
        if (IsFSM) {
          std::ostringstream W;
          W << P.Width << "'(" << Expr << ")";
          Expr = W.str();
        }
      }
    }
    OS << OutWriteNames[I] << " = " << Expr << ";\n";
  }
}

void Emitter::emitOp(mlir::Operation &Op, int Indent) {
  using namespace mlir;

  // Phase 3 — suppress ops the persistent-state matcher captured.
  // The isempty if-guard, its cmpf, and the isempty call all feed
  // only the reset path; they're emitted inside always_ff and have
  // no place in always_comb.
  if (Suppress.contains(&Op)) return;

  // Phase 3 — recognized persistent set call → assignment to the
  // register's `_next` signal. Inline the value expression when its
  // producer is a single-use fi-tagged matlab arith op so the
  // emitted SV avoids a wide intermediate temp (which Verilator
  // flags as WIDTHEXPAND when its declared width exceeds the
  // operand widths).
  {
    auto It = SetSiteToReg.find(&Op);
    if (It != SetSiteToReg.end()) {
      auto &P = Persists[It->second];
      mlir::Value Val = Op.getOperand(1);
      // Phase 4 v2: an FSM register's `_next = ...` assignment with
      // a constant value renders as the enum literal (e.g. `S1`),
      // not a raw integer literal. Drops out cleanly because the
      // const came from the matched cascade case-label.
      const HWFSMInfo *Fsm = nullptr;
      for (auto &FI : FSMs) {
        if (FI.RegIndex == It->second) { Fsm = &FI; break; }
      }
      if (Fsm) {
        if (auto C = Val.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
            int64_t V = IA.getInt();
            for (size_t i = 0; i < Fsm->Cases.size(); ++i) {
              if (Fsm->Cases[i].first == V) {
                indent(Indent);
                OS << P.Name << "_next = " << Fsm->CaseNames[i] << ";\n";
                return;
              }
            }
          }
        }
      }
      // Build the SV expression for `Val` — single-use fi arith ops
      // are rendered inline; everything else falls through to a
      // named reference.
      std::string ValExpr;
      if (auto *VOp = Val.getDefiningOp()) {
        llvm::StringRef N = VOp->getName().getStringRef();
        bool IsArith = (N == "matlab.add" || N == "matlab.sub" ||
                        N == "matlab.matmul" || N == "matlab.emul");
        if (IsArith && Val.hasOneUse() && VOp->getNumOperands() == 2) {
          // Mark the producer for skip — its result is folded into
          // this set-site expression.
          Suppress.insert(VOp);
          llvm::StringRef Sv = "+";
          if (N == "matlab.sub") Sv = "-";
          if (N == "matlab.matmul" || N == "matlab.emul") Sv = "*";
          ValExpr = exprFor(VOp->getOperand(0)) + " " + Sv.str() +
                    " " + exprFor(VOp->getOperand(1));
        }
      }
      if (ValExpr.empty()) ValExpr = exprFor(Val);

      indent(Indent);
      OS << P.Name << "_next = ";
      // The set's RHS may be wider than the register (e.g. an fi
      // i8+i8 add yields i9 → i16 in MLIR's signless type system).
      // Truncate to the register width via an SV size cast so the
      // assignment is unambiguous.
      unsigned VW = 0;
      if (auto IT = mlir::dyn_cast<mlir::IntegerType>(Val.getType()))
        VW = IT.getWidth();
      if (VW > P.Width) OS << P.Width << "'(";
      OS << ValExpr;
      if (VW > P.Width) OS << ")";
      OS << ";\n";
      return;
    }
  }

  // Phase 3 — recognized persistent get call has no statement-level
  // effect. Its uses route through the register signal name via
  // exprFor; the call op itself is skipped.
  if (GetSiteToReg.contains(&Op)) return;

  // Recognized bitwise builtins on integer / fi operands. The
  // frontend lowers `bitand/bitor/bitxor/bitshift/bitcmp(...)` to a
  // `matlab.call_builtin` site with the matching `callee` attr and
  // `none`-or-integer-typed operands; the SV emitter renders them as
  // direct bitwise operators. `bitshift(a, k)` with positive `k`
  // shifts left; negative `k` shifts right (arith for signed).
  if (Op.getName().getStringRef() == "matlab.call_builtin") {
    auto Callee = Op.getAttrOfType<mlir::StringAttr>("callee");
    if (Callee) {
      llvm::StringRef N = Callee.getValue();
      llvm::StringRef Sv;
      bool Unary = false;
      if (N == "bitand") Sv = "&";
      else if (N == "bitor")  Sv = "|";
      else if (N == "bitxor") Sv = "^";
      else if (N == "bitcmp") { Sv = "~"; Unary = true; }
      if (!Sv.empty()) {
        unsigned ExpectOperands = Unary ? 1 : 2;
        if (Op.getNumOperands() != ExpectOperands ||
            Op.getNumResults() != 1) {
          fail(("unsupported arity on " + N + " in SV emitter").str());
          return;
        }
        indent(Indent);
        OS << name(Op.getResult(0)) << " = ";
        if (Unary) {
          OS << Sv.str() << exprFor(Op.getOperand(0));
        } else {
          OS << exprFor(Op.getOperand(0)) << " " << Sv.str() << " "
             << exprFor(Op.getOperand(1));
        }
        OS << ";\n";
        return;
      }
      if (N == "bitshift") {
        if (Op.getNumOperands() != 2 || Op.getNumResults() != 1) {
          fail("unsupported arity on bitshift in SV emitter");
          return;
        }
        // Positive shift = left, negative = arithmetic right. The
        // shift amount may be a positive or negative compile-time
        // constant; check.
        mlir::Value Amt = Op.getOperand(1);
        bool IsLeft = true;
        int64_t AmtV = 0;
        bool AmtKnown = false;
        if (auto C = Amt.getDefiningOp<mlir::arith::ConstantOp>()) {
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
            AmtV = IA.getInt(); AmtKnown = true;
          } else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
            AmtV = (int64_t)FA.getValueAsDouble(); AmtKnown = true;
          }
        }
        if (AmtKnown && AmtV < 0) { IsLeft = false; AmtV = -AmtV; }
        indent(Indent);
        OS << name(Op.getResult(0)) << " = " << exprFor(Op.getOperand(0));
        if (IsLeft) OS << " << ";
        else        OS << " >>> ";
        if (AmtKnown) OS << AmtV;
        else          OS << exprFor(Amt);
        OS << ";\n";
        return;
      }
    }
  }

  // Phase 3 — recognized scalar `matlab.*` arithmetic / comparison
  // ops that survive LowerFixedPoint with an f64 operand (typically
  // because one operand is a persistent get). The SV emitter renders
  // them as if the f64 input were already the register's integer
  // value (`exprFor` routes persistent-get results to the register
  // signal name).
  {
    llvm::StringRef OpName = Op.getName().getStringRef();
    llvm::StringRef Sv;
    if (OpName == "matlab.add") Sv = "+";
    else if (OpName == "matlab.sub") Sv = "-";
    else if (OpName == "matlab.matmul" || OpName == "matlab.emul") Sv = "*";
    else if (OpName == "matlab.gt") Sv = ">";
    else if (OpName == "matlab.ge") Sv = ">=";
    else if (OpName == "matlab.lt") Sv = "<";
    else if (OpName == "matlab.le") Sv = "<=";
    else if (OpName == "matlab.eq") Sv = "==";
    else if (OpName == "matlab.ne") Sv = "!=";
    if (!Sv.empty()) {
      if (Op.getNumOperands() != 2 || Op.getNumResults() != 1) {
        fail(("unsupported arity on " + OpName + " in SV emitter").str());
        return;
      }
      indent(Indent);
      std::string LExpr = exprFor(Op.getOperand(0));
      std::string RExpr = exprFor(Op.getOperand(1));
      // FSM-aware: render the constant peer as the enum literal
      // when comparing/using a recognized FSM register's get.
      // Only do this for equality/inequality — other ops (`+`, `-`,
      // `*`) on enum literals are still raw integer arithmetic.
      if (OpName == "matlab.eq" || OpName == "matlab.ne") {
        if (auto E = fsmEnumLiteralForConstAgainst(
                Op.getOperand(0), Op.getOperand(1));
            !E.empty()) {
          RExpr = E;
        } else if (auto E = fsmEnumLiteralForConstAgainst(
                       Op.getOperand(1), Op.getOperand(0));
                   !E.empty()) {
          LExpr = E;
        }
      }
      OS << name(Op.getResult(0)) << " = " << LExpr << " " << Sv.str() << " "
         << RExpr << ";\n";
      return;
    }
  }

  if (auto C = dyn_cast<arith::ConstantOp>(Op)) {
    emitArithConstant(C, Indent); return;
  }
  if (isa<arith::AddIOp>(Op)) { emitBinop(Op, "+", Indent); return; }
  if (isa<arith::SubIOp>(Op)) { emitBinop(Op, "-", Indent); return; }
  if (isa<arith::MulIOp>(Op)) { emitBinop(Op, "*", Indent); return; }
  if (isa<arith::DivSIOp>(Op)) { emitBinop(Op, "/", Indent); return; }
  if (isa<arith::DivUIOp>(Op)) { emitBinop(Op, "/", Indent); return; }
  if (isa<arith::RemSIOp>(Op)) { emitBinop(Op, "%", Indent); return; }
  if (isa<arith::RemUIOp>(Op)) { emitBinop(Op, "%", Indent); return; }
  if (isa<arith::AndIOp>(Op)) { emitBinop(Op, "&", Indent); return; }
  if (isa<arith::OrIOp>(Op))  { emitBinop(Op, "|", Indent); return; }
  if (isa<arith::XOrIOp>(Op)) { emitBinop(Op, "^", Indent); return; }
  if (isa<arith::ShLIOp>(Op))  { emitBinop(Op, "<<",  Indent); return; }
  if (isa<arith::ShRSIOp>(Op)) { emitBinop(Op, ">>>", Indent); return; }
  if (isa<arith::ShRUIOp>(Op)) { emitBinop(Op, ">>",  Indent); return; }
  if (auto C = dyn_cast<arith::CmpIOp>(Op)) { emitCmp(C, Indent); return; }
  if (auto C = dyn_cast<arith::CmpFOp>(Op)) { emitCmpF(C, Indent); return; }
  if (auto S = dyn_cast<arith::SelectOp>(Op)) { emitSelect(S, Indent); return; }
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(Op)) {
    emitExtTrunc(Op, Indent); return;
  }
  if (auto If = dyn_cast<scf::IfOp>(Op)) {
    // Phase 4 v2 — head of an FSM cascade renders as `unique case`;
    // inner cascade ifs are already absorbed into the head's case
    // arms and emit nothing (their THEN regions are walked by
    // emitFSMCase).
    auto FIt = CascadeOp.find(&Op);
    if (FIt != CascadeOp.end()) {
      auto &FI = FSMs[FIt->second];
      if (FI.Head == &Op) emitFSMCase(If, Indent);
      // Inner cascade ifs: skip silently.
      return;
    }
    emitScfIf(If, Indent);
    return;
  }
  if (auto Y = dyn_cast<scf::YieldOp>(Op)) { emitScfYield(Y, Indent); return; }
  if (auto W = dyn_cast<scf::WhileOp>(Op)) { emitScfWhile(W, Indent); return; }
  if (auto A = dyn_cast<LLVM::AllocaOp>(Op)) { emitAlloca(A, Indent); return; }
  if (auto G = dyn_cast<LLVM::GEPOp>(Op)) { emitGEP(G, Indent); return; }
  if (auto L = dyn_cast<LLVM::LoadOp>(Op)) { emitLoad(L, Indent); return; }
  if (auto S = dyn_cast<LLVM::StoreOp>(Op)) { emitStore(S, Indent); return; }
  if (auto R = dyn_cast<func::ReturnOp>(Op)) { emitReturn(R, Indent); return; }

  // `llvm.mlir.constant` is produced as the size operand of an alloca
  // (always 1 in our pipeline, scalar slot). It has no datapath
  // semantics — the slot's `logic` declaration carries everything.
  if (isa<LLVM::ConstantOp>(Op)) return;

  std::ostringstream Err;
  Err << "unsupported op in emitter: " << Op.getName().getStringRef().str();
  fail(Err.str());
}

void Emitter::emitBlock(mlir::Block &B, int Indent) {
  for (auto &Op : B.getOperations()) {
    if (Failed) return;
    emitOp(Op, Indent);
  }
}

void Emitter::emitRegion(mlir::Region &R, int Indent) {
  for (auto &B : R) {
    if (Failed) return;
    emitBlock(B, Indent);
  }
}

void Emitter::emitBody(mlir::func::FuncOp F) {
  // Single always_comb block. Phase 1 has no clocked logic.
  OS << "    always_comb begin\n";
  // Latch guard: pre-assign every always_comb-driven signal at the
  // top so no code path leaves it unassigned. This is the canonical
  // SV idiom for combinational temps and is recognized by every
  // synth tool as a default-assignment pattern (no latch inferred).
  // Without it, Verilator (correctly) flags any signal that's
  // conditionally written in only some branches.
  for (mlir::Value V : PreludeDecls) {
    indent(2);
    // Phase 4.5.4: arrays use the SV `'{default: '0}` literal so
    // every element zero-inits.
    if (auto *Op = V.getDefiningOp()) {
      if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Op)) {
        if (mlir::isa<mlir::LLVM::LLVMArrayType>(A.getElemType())) {
          OS << name(V) << " = '{default: '0};\n";
          continue;
        }
      }
    }
    OS << name(V) << " = '0;\n";
  }
  // Phase 3: each persistent register's `_next` signal defaults to
  // the current register value — i.e. "hold by default". The user's
  // body may then conditionally overwrite it. Without this, branches
  // that don't assign `_next` would infer a latch.
  for (auto &P : Persists) {
    indent(2);
    OS << P.Name << "_next = " << P.Name << ";\n";
  }
  // Likewise drive every output port to 0 by default — `func.return`
  // will overwrite later, but if the function has any conditional
  // structure the same latch-inference rule applies. Use
  // OutWriteNames so output_pipeline=N writes feed `<out>_d0`
  // (the pre-pipeline signal) instead of the actual port.
  for (const auto &Out : OutWriteNames) {
    indent(2);
    OS << Out << " = '0;\n";
  }
  emitRegion(F.getBody(), 2);
  OS << "    end\n";
}

bool Emitter::emitModuleForFunc(mlir::func::FuncOp F) {
  // Phase 4 v2.6: per-function FSM encoding override. If the
  // function carries an `hdl.fsm_encoding` string attribute
  // (set by ScanHWPragmas from a `% hdl: fsm_encoding('...')`
  // comment), use that encoding instead of the CLI-wide flag.
  // Saved + restored around the function so different functions
  // in the same module can use different encodings.
  HWFSMEncoding SavedFSMEnc = FSMEnc;
  if (auto Attr = F->getAttrOfType<mlir::StringAttr>("hdl.fsm_encoding")) {
    llvm::StringRef V = Attr.getValue();
    if (V == "binary") FSMEnc = HWFSMEncoding::Binary;
    else if (V == "one_hot" || V == "one-hot")
      FSMEnc = HWFSMEncoding::OneHot;
    else if (V == "gray") FSMEnc = HWFSMEncoding::Gray;
    else
      mlir::emitWarning(F.getLoc())
          << "unrecognized hdl.fsm_encoding value '" << V
          << "' (expected 'binary', 'one_hot', or 'gray')";
  }

  // Phase 5.2: per-function port-pipeline stage counts.
  InputPipelineN = 0;
  OutputPipelineN = 0;
  auto parsePipelineAttr = [&](llvm::StringRef Name, int &OutN) {
    if (auto A = F->getAttrOfType<mlir::StringAttr>(Name)) {
      int V = 0;
      try { V = std::stoi(A.getValue().str()); }
      catch (...) {
        mlir::emitWarning(F.getLoc())
            << Name << " value '" << A.getValue() << "' is not an integer";
        return;
      }
      if (V < 0) {
        mlir::emitWarning(F.getLoc())
            << Name << " stage count must be ≥ 0, got " << V;
        return;
      }
      OutN = V;
    }
  };
  parsePipelineAttr("hdl.input_pipeline", InputPipelineN);
  parsePipelineAttr("hdl.output_pipeline", OutputPipelineN);

  // Reset per-function state.
  Names.clear();
  Used.clear();
  NextFresh = 0;
  ArgNames.clear();
  OutNames.clear();
  PreludeDecls.clear();
  Persists.clear();
  GetSiteToReg.clear();
  SetSiteToReg.clear();
  Suppress.clear();
  GepAddr.clear();
  FSMs.clear();
  CascadeOp.clear();

  // Collect persistent registers for this function. HWLegalize already
  // validated these — gathering can only fail if the IR mutated since;
  // bail with a hard error in that case.
  if (!gatherHWPersistentState(F.getOperation(), Persists)) {
    fail("persistent state shape changed since HWLegalize");
    return false;
  }
  // Reserve register signal names ahead of port-list emission so they
  // don't collide with module / arg / output names.
  for (unsigned R = 0; R < Persists.size(); ++R) {
    auto &P = Persists[R];
    std::string Base = sanitize(P.Name);
    while (Used.contains(Base)) Base += "_";
    Used.insert(Base);
    P.Name = Base;
    std::string NextSig = Base + "_next";
    while (Used.contains(NextSig)) NextSig += "_";
    Used.insert(NextSig);
    // Stash the chosen `_next` name in a side-channel via the get/set
    // tables: GetSiteToReg/SetSiteToReg map to the register index, and
    // we look up `Persists[idx].Name` / `Persists[idx].Name + "_next"`
    // at emit time. (We keep `_next` implicit — `Name + "_next"` is
    // computed as needed.)
    for (auto *Op : P.Gets) GetSiteToReg[Op] = R;
    for (auto *Op : P.Sets) SetSiteToReg[Op] = R;
    // The isempty guard is suppressed during always_comb emission
    // (its init becomes the reset value). Suppress the cmpf and
    // isempty call too — they feed only the guard.
    if (P.IsEmptyGuard) {
      Suppress.insert(P.IsEmptyGuard);
      // The cmpf operand of the guard is the cmpf op.
      if (auto IfOp = mlir::dyn_cast<mlir::scf::IfOp>(P.IsEmptyGuard))
        if (auto *Cmp = IfOp.getCondition().getDefiningOp())
          Suppress.insert(Cmp);
    }
    // Suppress the isempty call itself.
    F.walk([&](mlir::Operation *Op) {
      if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
        auto C = Call.getCallee();
        if (C && *C == "matlab_persistent_isempty") Suppress.insert(Op);
      }
    });
  }

  // Phase 4 v2 — recognize FSM cascades on persistent registers
  // BEFORE the prelude / body emit so the cascade ifs are added to
  // Suppress and the register's declaration uses the enum type.
  gatherFSMs(F);

  std::string ModName = sanitize(F.getSymName());
  // Module name conflicts with a register/port name only in adversarial
  // input; not worth a full uniquifier — collisions are caught by the
  // synth tool downstream.
  (void)ModName;

  OS << "module " << sanitize(F.getSymName()) << " (\n";
  emitPortList(F);
  OS << ");\n\n";
  // FSM typedef enums declared at module scope, before signal decls.
  if (!FSMs.empty()) {
    emitFSMTypedefs();
    OS << "\n";
  }
  declarePrelude(F);
  // Phase 5.2 — port-pipeline register declarations.
  emitPipelineDecls(F);
  emitBody(F);
  emitAlwaysFF();
  // Phase 5.2 — pipeline shift register + assign-out drivers.
  emitPipelineFF();
  OS << "\nendmodule\n";

  // Restore the CLI-wide FSM encoding for the next function.
  FSMEnc = SavedFSMEnc;
  return !Failed;
}

// Helper: try to match an scf.if condition as a state-equality
// check `<cmp> <persistent_get>, <const>` and return (RegIndex,
// CaseValue) on success. Two surface shapes are accepted:
//
//   - `arith.cmpf oeq, get_f64(reg), <const>` — the canonical
//     shape from a `switch state` lowering when the case labels
//     are float-typed (the front-end emits switch labels as f64
//     regardless of the discriminator type).
//   - `matlab.eq(get_f64(reg), <const>)` — the shape that
//     survives when the case labels are typed integers (e.g.
//     uint8(N) folded to i8 constants) and the LHS f64 ABI never
//     converted, leaving the unregistered matlab.eq op for the
//     emitter to handle.
static bool matchStateEq(mlir::Value Cond,
                         const llvm::DenseMap<mlir::Operation *, unsigned>
                             &GetSiteToReg,
                         unsigned &OutReg, int64_t &OutVal) {
  mlir::Value Lhs;
  mlir::Value Rhs;
  if (auto Cmp = Cond.getDefiningOp<mlir::arith::CmpFOp>()) {
    if (Cmp.getPredicate() != mlir::arith::CmpFPredicate::OEQ) return false;
    Lhs = Cmp.getLhs();
    Rhs = Cmp.getRhs();
  } else if (auto *Op = Cond.getDefiningOp()) {
    if (Op->getName().getStringRef() != "matlab.eq") return false;
    if (Op->getNumOperands() != 2) return false;
    Lhs = Op->getOperand(0);
    Rhs = Op->getOperand(1);
  } else {
    return false;
  }
  auto *L = Lhs.getDefiningOp();
  if (!L) return false;
  auto It = GetSiteToReg.find(L);
  if (It == GetSiteToReg.end()) return false;
  auto C = Rhs.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return false;
  if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue())) {
    OutVal = (int64_t)FA.getValueAsDouble();
  } else if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
    OutVal = IA.getInt();
  } else {
    return false;
  }
  OutReg = It->second;
  return true;
}

void Emitter::gatherFSMs(mlir::func::FuncOp F) {
  llvm::DenseSet<mlir::Operation *> Inner;  // skip these as heads
  F.walk([&](mlir::scf::IfOp If) {
    if (Inner.contains(If.getOperation())) return;
    unsigned RegIdx;
    int64_t CaseVal;
    if (!matchStateEq(If.getCondition(), GetSiteToReg, RegIdx, CaseVal))
      return;
    HWFSMInfo Info;
    Info.RegIndex = RegIdx;
    Info.Head = If.getOperation();
    Info.Cases.push_back({CaseVal, &If.getThenRegion()});

    mlir::scf::IfOp Cur = If;
    while (true) {
      mlir::Region &Else = Cur.getElseRegion();
      if (Else.empty() || !Else.hasOneBlock()) break;
      mlir::Block &EB = Else.front();
      // Look for the pattern `<cmpf-supporting ops>; scf.if <cmpf>;
      // scf.yield` — the next-cascade scf.if is the LAST non-yield
      // op in the else block; the ops before it are the cmpf chain
      // feeding its condition. The cascade continues only when the
      // last non-yield op is a state-eq scf.if matching the same
      // register, AND no other side-effecting ops exist between
      // the cmpf chain and that scf.if.
      mlir::scf::IfOp Next = nullptr;
      bool ExtraSideEffects = false;
      for (mlir::Operation &Op : EB) {
        if (mlir::isa<mlir::scf::YieldOp>(Op)) continue;
        if (auto NIf = mlir::dyn_cast<mlir::scf::IfOp>(Op)) {
          if (Next) { ExtraSideEffects = true; break; }
          Next = NIf;
          continue;
        }
        // Non-if ops in the else region: only allow pure operands
        // feeding the next scf.if's condition (cmpf / cmpi /
        // constants / matlab.eq for the mixed-type integer-literal
        // case-label form / a recognized persistent-get re-read,
        // which is what the IR uses to look up the state register
        // for each cascade tail). Anything else (a store, an
        // unrelated call, ...) means the else has its own side
        // effects and is the default arm.
        bool IsPureCmp =
            mlir::isa<mlir::arith::CmpFOp, mlir::arith::CmpIOp,
                      mlir::arith::ConstantOp,
                      mlir::LLVM::ConstantOp>(Op) ||
            Op.getName().getStringRef() == "matlab.eq";
        bool IsPersistentGet = false;
        if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
          auto Sym = Call.getCallee();
          if (Sym && *Sym == "matlab_global_get_f64")
            IsPersistentGet = true;
        }
        if (!IsPureCmp && !IsPersistentGet) {
          ExtraSideEffects = true;
          break;
        }
      }
      unsigned NextReg;
      int64_t NextVal;
      if (Next && !ExtraSideEffects &&
          matchStateEq(Next.getCondition(), GetSiteToReg,
                        NextReg, NextVal) &&
          NextReg == RegIdx) {
        Info.Cases.push_back({NextVal, &Next.getThenRegion()});
        Inner.insert(Next.getOperation());
        Cur = Next;
        continue;
      }
      // Else-region is the default arm (anything else).
      bool ElseEmpty = true;
      for (mlir::Operation &Op : EB) {
        if (!mlir::isa<mlir::scf::YieldOp>(Op)) { ElseEmpty = false; break; }
      }
      if (!ElseEmpty) Info.DefaultRegion = &Else;
      break;
    }

    if (Info.Cases.size() < 2) return;  // not really a cascade

    // Phase 4 v2.3 — ambiguity diagnostics. Three checks, each
    // with low false-positive rate and high real-bug rate.
    //
    // (1) Duplicate case labels. The user wrote two `case <c>`
    //     arms with the same constant in the same switch — the
    //     later arm is unreachable, definite bug.
    {
      llvm::SmallSet<int64_t, 4> Seen;
      for (auto &[Val, _Region] : Info.Cases) {
        if (!Seen.insert(Val).second) {
          mlir::emitError(Info.Head->getLoc())
              << "FSM cascade on persistent '" << Persists[RegIdx].Name
              << "' has a duplicate case label '" << Val
              << "' — the second arm is unreachable. "
              << "Remove the duplicate or distinguish the constants.";
          Failed = true;
          break;
        }
      }
    }

    // (Skipped: "state written but never matched in any case
    // arm" check. False-positives when a Moore-style output-decode
    // cascade only covers a subset of states and routes the rest
    // through its default arm — which is a perfectly valid pattern
    // even though no `case X` exists for some constants the state
    // register can hold.)

    // (2) Empty case arm — a recognized cascade arm whose body
    //     is empty (just an scf.yield) suggests the user meant
    //     to do something but didn't. Deliberately empty arms
    //     are usually written as `case Sx, /* fall through */`
    //     in HDL Coder style, which we don't yet support; for
    //     now flag empty arms as suspect.
    for (size_t i = 0; i < Info.Cases.size(); ++i) {
      auto &[Val, Region] = Info.Cases[i];
      if (!Region) continue;  // synthesized reset arm
      bool Empty = true;
      for (mlir::Block &B : *Region) {
        for (mlir::Operation &Op : B) {
          if (mlir::isa<mlir::scf::YieldOp>(Op)) continue;
          Empty = false; break;
        }
        if (!Empty) break;
      }
      if (Empty) {
        mlir::emitError(Info.Head->getLoc())
            << "FSM cascade on persistent '" << Persists[RegIdx].Name
            << "' has an empty `case " << Val
            << "` arm — the user wrote `case " << Val
            << "` with no body, which is almost always unintended. "
               "Add a body or remove the arm.";
        Failed = true;
      }
    }

    // Generate enum-literal names. v1 uses S0/S1/.../SN based on the
    // order constants appear; the case constants are remembered so we
    // can map the persistent's reset value to the right name.
    auto &P = Persists[Info.RegIndex];
    Info.EnumType = P.Name + "_t";
    for (auto &[Val, Region] : Info.Cases) {
      Info.CaseNames.push_back("S" + std::to_string(Val));
    }
    // If the default region exists but no case matched the reset
    // value, add a synthetic enum literal for the reset state too —
    // SV requires the enum literal to be defined before use.
    int64_t ResetVal = 0;
    if (auto *RV = P.ResetValue.getDefiningOp()) {
      if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(RV)) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
          ResetVal = IA.getInt();
      }
    }
    bool Found = false;
    for (auto &[Val, Region] : Info.Cases) {
      if (Val == ResetVal) { Found = true; break; }
    }
    if (!Found) {
      Info.Cases.insert(Info.Cases.begin(), {ResetVal, nullptr});
      Info.CaseNames.insert(Info.CaseNames.begin(),
                            "S" + std::to_string(ResetVal));
    }
    Info.ResetName = "S" + std::to_string(ResetVal);

    unsigned Idx = (unsigned)FSMs.size();
    CascadeOp[Info.Head] = Idx;
    for (mlir::Operation *I : Inner) CascadeOp[I] = Idx;
    // Suppress the cmpf ops feeding the cascade scf.ifs — they're
    // structural to the case lowering and shouldn't render as
    // separate `v_N = state == X` assignments. Don't add the
    // cascade scf.ifs themselves to Suppress: their bodies are
    // case arms that we still want to walk for prelude
    // declarations.
    auto SuppressIfCmp = [&](mlir::scf::IfOp If) {
      if (auto *Op = If.getCondition().getDefiningOp()) {
        if (mlir::isa<mlir::arith::CmpFOp>(Op) ||
            Op->getName().getStringRef() == "matlab.eq")
          Suppress.insert(Op);
      }
    };
    SuppressIfCmp(mlir::cast<mlir::scf::IfOp>(Info.Head));
    for (mlir::Operation *I : Inner)
      SuppressIfCmp(mlir::cast<mlir::scf::IfOp>(I));
    FSMs.push_back(std::move(Info));
  });
}

void Emitter::emitFSMTypedefs() {
  // A function may contain multiple cascade-shaped switch
  // statements on the same persistent register (e.g. the
  // canonical Moore output-decode `if state == S2 then ... else
  // ...` after the state-transition cascade). Each is rendered as
  // its own `unique case`, but they share the underlying enum
  // type, so emit each typedef only once per register.
  llvm::DenseSet<unsigned> Emitted;
  for (auto &F : FSMs) {
    if (!Emitted.insert(F.RegIndex).second) continue;
    unsigned N = (unsigned)F.Cases.size();

    // Compute encoded value per state and the underlying width.
    // - Binary  : sequential, width = ⌈log2(N)⌉ (≥1).
    // - OneHot  : one bit per state, width = N.
    // - Gray    : reflected-binary gray code, width = ⌈log2(N)⌉.
    unsigned W = 1;
    llvm::SmallVector<uint64_t, 8> Values(N);
    if (FSMEnc == HWFSMEncoding::OneHot) {
      W = N;
      for (unsigned i = 0; i < N; ++i)
        Values[i] = (uint64_t)1 << i;
    } else {
      while ((1u << W) < N) ++W;
      for (unsigned i = 0; i < N; ++i) {
        if (FSMEnc == HWFSMEncoding::Gray)
          Values[i] = i ^ (i >> 1);
        else
          Values[i] = i;  // Binary
      }
    }

    indent(1);
    OS << "typedef enum logic";
    if (W > 1) OS << " [" << (W - 1) << ":0]";
    OS << " {";
    bool ExplicitVals = (FSMEnc != HWFSMEncoding::Binary);
    for (unsigned i = 0; i < F.CaseNames.size(); ++i) {
      if (i) OS << ", ";
      OS << F.CaseNames[i];
      if (ExplicitVals) {
        OS << " = " << W << "'d" << Values[i];
      }
    }
    OS << "} " << F.EnumType << ";\n";
  }
}

void Emitter::emitFSMCase(mlir::scf::IfOp Head, int Indent) {
  auto It = CascadeOp.find(Head.getOperation());
  if (It == CascadeOp.end()) {
    fail("emitFSMCase called on unrecognized scf.if");
    return;
  }
  auto &F = FSMs[It->second];
  auto &P = Persists[F.RegIndex];
  indent(Indent);
  OS << "unique case (" << P.Name << ")\n";
  for (size_t i = 0; i < F.Cases.size(); ++i) {
    auto &[Val, Region] = F.Cases[i];
    indent(Indent + 1);
    OS << F.CaseNames[i] << ": begin\n";
    if (Region) {
      emitRegion(*Region, Indent + 2);
    }
    indent(Indent + 1);
    OS << "end\n";
  }
  if (F.DefaultRegion) {
    indent(Indent + 1);
    OS << "default: begin\n";
    emitRegion(*F.DefaultRegion, Indent + 2);
    indent(Indent + 1);
    OS << "end\n";
  } else {
    // Always emit a default for `unique case` so synth tools don't
    // warn about incompletely-decoded inputs. The default reasserts
    // the hold-by-default `<reg>_next = <reg>` (already set at the
    // top of always_comb), so no explicit body needed.
    indent(Indent + 1);
    OS << "default: ;\n";
  }
  indent(Indent);
  OS << "endcase\n";
}

/// Phase 5.2: needs clk + rst_n ports when port pipelining is on
/// or there are persistent registers.
static bool wantsClock(int InputPipelineN, int OutputPipelineN,
                       size_t PersistsCount) {
  return InputPipelineN > 0 || OutputPipelineN > 0 || PersistsCount > 0;
}

void Emitter::emitPipelineDecls(mlir::func::FuncOp F) {
  if (InputPipelineN <= 0 && OutputPipelineN <= 0) return;
  // Input pipeline: each input port `<arg>` gets stages
  // `<arg>_d1, <arg>_d2, ..., <arg>_dN`. The body's references
  // to the port (already named in ArgNames[I]) are routed to
  // `<arg>_dN` via the Names map override below.
  auto FT = F.getFunctionType();
  if (InputPipelineN > 0) {
    OS << "    // Phase 5.2 input-pipeline registers (";
    OS << InputPipelineN << " stage" << (InputPipelineN > 1 ? "s" : "");
    OS << ").\n";
    for (unsigned I = 0; I < FT.getNumInputs(); ++I) {
      const std::string &Base = ArgNames[I];
      for (int S = 1; S <= InputPipelineN; ++S) {
        OS << "    " << svType(FT.getInput(I)) << " "
           << Base << "_d" << S << ";\n";
      }
    }
  }
  // Output pipeline: each output port `<out>` gets stages
  // `<out>_d0` (combinational pre-pipe), `<out>_d1, ..., <out>_dN`.
  // The combinational body writes to `<out>_d0`; the always_ff
  // shifts d0 → d1 → … → dN; an `assign <out> = <out>_dN;`
  // drives the port. (The hold-by-default `<out> = '0;` line in
  // emitBody already pre-assigns; we redirect it to `<out>_d0`.)
  if (OutputPipelineN > 0) {
    OS << "    // Phase 5.2 output-pipeline registers (";
    OS << OutputPipelineN << " stage" << (OutputPipelineN > 1 ? "s" : "");
    OS << ").\n";
    for (unsigned I = 0; I < FT.getNumResults(); ++I) {
      mlir::Type T = FT.getResult(I);
      // Effective port type may differ from FT for FSM returns;
      // re-derive via the per-function trick used in emitPortList
      // (Phase 4 v2.6). For pipeline declarations we use the
      // function-result type — port pipelining is post-FSM.
      F.walk([&](mlir::func::ReturnOp R) {
        if (R.getNumOperands() <= I) return;
        auto *Op = R.getOperand(I).getDefiningOp();
        if (!Op) return;
        auto It = GetSiteToReg.find(Op);
        if (It == GetSiteToReg.end()) return;
        auto &P = Persists[It->second];
        T = mlir::IntegerType::get(F.getContext(), P.Width);
      });
      const std::string &Base = OutNames[I];
      for (int S = 0; S <= OutputPipelineN; ++S) {
        OS << "    " << svType(T) << " "
           << Base << "_d" << S << ";\n";
      }
    }
  }
  OS << "\n";
}

void Emitter::emitPipelineFF() {
  if (InputPipelineN <= 0 && OutputPipelineN <= 0) return;
  // One synchronous block shifting both pipelines on every
  // posedge clk; the chosen reset policy applies.
  switch (Reset) {
  case HWResetKind::AsyncLow:
    OS << "    always_ff @(posedge clk or negedge rst_n) begin\n";
    OS << "        if (!rst_n) begin\n"; break;
  case HWResetKind::SyncHigh:
    OS << "    always_ff @(posedge clk) begin\n";
    OS << "        if (rst) begin\n"; break;
  case HWResetKind::SyncLow:
    OS << "    always_ff @(posedge clk) begin\n";
    OS << "        if (!rst_n) begin\n"; break;
  }
  // Reset every pipeline stage to 0.
  if (InputPipelineN > 0) {
    for (size_t I = 0; I < ArgNames.size(); ++I) {
      const std::string &Base = ArgNames[I];
      for (int S = 1; S <= InputPipelineN; ++S) {
        indent(3);
        OS << Base << "_d" << S << " <= '0;\n";
      }
    }
  }
  if (OutputPipelineN > 0) {
    for (size_t I = 0; I < OutNames.size(); ++I) {
      const std::string &Base = OutNames[I];
      for (int S = 1; S <= OutputPipelineN; ++S) {
        indent(3);
        OS << Base << "_d" << S << " <= '0;\n";
      }
    }
  }
  OS << "        end else begin\n";
  // Shift each pipeline.
  if (InputPipelineN > 0) {
    for (size_t I = 0; I < ArgNames.size(); ++I) {
      const std::string &Base = ArgNames[I];
      indent(3);
      OS << Base << "_d1 <= " << Base << ";\n";
      for (int S = 2; S <= InputPipelineN; ++S) {
        indent(3);
        OS << Base << "_d" << S << " <= " << Base << "_d" << (S - 1)
           << ";\n";
      }
    }
  }
  if (OutputPipelineN > 0) {
    for (size_t I = 0; I < OutNames.size(); ++I) {
      const std::string &Base = OutNames[I];
      for (int S = 1; S <= OutputPipelineN; ++S) {
        indent(3);
        OS << Base << "_d" << S << " <= " << Base << "_d" << (S - 1)
           << ";\n";
      }
    }
  }
  OS << "        end\n";
  OS << "    end\n\n";
  // Drive output ports from the last-stage pipeline register.
  if (OutputPipelineN > 0) {
    for (size_t I = 0; I < OutNames.size(); ++I) {
      OS << "    assign " << OutNames[I]
         << " = " << OutNames[I] << "_d" << OutputPipelineN << ";\n";
    }
    OS << "\n";
  }
}

void Emitter::emitAlwaysFF() {
  if (Persists.empty()) return;
  OS << "\n";
  // One always_ff block per reset domain. Phase 3 has a single domain
  // (clk + chosen reset polarity / synchronicity). All registers are
  // driven from the same block.
  switch (Reset) {
  case HWResetKind::AsyncLow:
    OS << "    always_ff @(posedge clk or negedge rst_n) begin\n";
    OS << "        if (!rst_n) begin\n";
    break;
  case HWResetKind::SyncHigh:
    OS << "    always_ff @(posedge clk) begin\n";
    OS << "        if (rst) begin\n";
    break;
  case HWResetKind::SyncLow:
    OS << "    always_ff @(posedge clk) begin\n";
    OS << "        if (!rst_n) begin\n";
    break;
  }
  for (unsigned R = 0; R < Persists.size(); ++R) {
    auto &P = Persists[R];
    indent(3);
    // Phase 4 v2: FSM register's reset value is the enum literal
    // for the reset-state, not a raw integer literal.
    bool IsFSM = false;
    std::string ResetExpr;
    for (auto &FI : FSMs) {
      if (FI.RegIndex == R) {
        ResetExpr = FI.ResetName;
        IsFSM = true;
        break;
      }
    }
    if (!IsFSM) ResetExpr = exprFor(P.ResetValue);
    OS << P.Name << " <= " << ResetExpr << ";\n";
  }
  OS << "        end else begin\n";
  for (auto &P : Persists) {
    indent(3);
    OS << P.Name << " <= " << P.Name << "_next;\n";
  }
  OS << "        end\n";
  OS << "    end\n";
}

void Emitter::emitProlog() {
  OS << "// Generated by matlabc -emit-systemverilog. Do not edit.\n";
  OS << "// Target: ASIC, vendor-neutral synthesizable SystemVerilog.\n";
  OS << "// Phase 1 — scalar combinational only.\n\n";
}

bool Emitter::run(mlir::ModuleOp M) {
  emitProlog();
  bool First = true;
  M.walk([&](mlir::func::FuncOp F) {
    if (Failed) return;
    if (F.empty()) return;
    {
      llvm::StringRef N = F.getSymName();
      if (N == "script" || N == "main") return; // skip top-level driver
    }
    if (!First) OS << "\n";
    First = false;
    if (!emitModuleForFunc(F)) return;
  });
  return !Failed;
}

} // namespace

std::string emitSystemVerilog(mlir::ModuleOp M,
                              const matlab::SourceManager *SM,
                              HWResetKind Reset, HWFSMEncoding FSMEnc) {
  std::ostringstream OS;
  Emitter E(OS, SM, Reset, FSMEnc);
  if (!E.run(M)) return std::string();
  return OS.str();
}

} // namespace mlirgen
} // namespace matlab
