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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
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
  // D1 — port-width lint hint. For each integer-typed input port
  // with width > 1, walks transitive uses through stores/loads of
  // its spill slot and checks if every leaf use is a boolean
  // predicate (`arith.cmpi eq/ne` or `matlab.eq/matlab.ne` against
  // a constant). On match, emits a stderr warning suggesting the
  // user retype the source as `bool` / 1-bit fi so the SV port
  // shrinks to a single bit. Informational; emission still
  // succeeds.
  void emitPortHints(mlir::func::FuncOp F);
  // True when every (transitive) use of `V` lands at a boolean
  // predicate against a constant. Walks through `matlab.store` /
  // `llvm.store` to a spill slot and follows that slot's loads.
  bool isBooleanOnlyPort(mlir::Value V);

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
  // B1 — saturation rendering. `collectSatHelpers` walks the
  // function for any `arith.select` carrying the
  // `matlab.fi_sat_w` / `matlab.fi_sat_signed` attrs (set by
  // LowerFiSaturate) and records the (signed, sat-width,
  // input-width) tuples used. `emitSatHelpers` declares one
  // `function automatic` per unique tuple inside the module so
  // the body can call `sat_<s|u><IW>_b<W>(x)` instead of
  // inlining the cmp/cmp/select/select ternary chain.
  void collectSatHelpers(mlir::func::FuncOp F);
  void emitSatHelpers();
  // Walk the saturating outer SelectOp's operand structure to
  // find the original (pre-clamp) input value the helper takes.
  // Signed sat: input is the inner SelectOp's FalseValue.
  // Unsigned sat: input is the outer SelectOp's FalseValue.
  mlir::Value satHelperInput(mlir::arith::SelectOp Sel, bool Signed);
  // Build the helper's SV name for a given (signed, sat-width,
  // input-width) triple. Pure formatting; same triple →
  // identical name in both `collectSatHelpers` and the call-site
  // rendering paths.
  std::string satHelperName(bool Signed, unsigned SatW,
                            unsigned InputW);
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
  // Try to match an if/else-if chain where every condition is
  // `<disc> == <const>` against the SAME `<disc>` value. On match,
  // emit `unique case (<disc>) ... endcase` and return true. The
  // synth tool then realizes this as a parallel mux instead of the
  // priority cascade implied by nested if/else, which improves
  // timing and matches the user's source-level `switch ... case`.
  // Returns false (caller falls back to nested if/else) for chains
  // with non-equality / non-uniform-discriminator conditions and
  // for short chains (1 case) that don't benefit from the case
  // form.
  bool tryEmitSwitchCase(mlir::scf::IfOp Head, int Indent);
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
  // If V is an integer-constant op whose value fits in a literal of
  // width W and the requested signedness, return the literal text
  // re-rendered at that width (e.g. `4'sd0` instead of `4'(8'sd0)`).
  // Returns std::nullopt when V is not a foldable constant or its
  // value doesn't fit, leaving the caller to wrap with the existing
  // `W'(...)` size-cast form.
  std::optional<std::string>
  tryReemitConstAtWidth(mlir::Value V, unsigned W, bool Signed);
  // S1 — pick the rendered width of `V` as it would appear in SV.
  // For values that route through a persist-get this is the
  // register's `Persists[].Width` (the SV signal's actual width),
  // not the IR's possibly-wider integer type. Returns 0 when no
  // narrower width is known.
  unsigned renderedWidthOf(mlir::Value V);
  // Render `V` as an SV expression in the context of an adjacent
  // operand whose rendered width is `OtherW`. When `V` is a
  // small-magnitude constant whose value fits in `OtherW` bits,
  // re-emit at that width so a `count_reg + 1` reads as
  // `count_reg + 4'd1` instead of `count_reg + 8'sd1`. Falls
  // back to plain `exprFor` otherwise.
  std::string exprForInContext(mlir::Value V, unsigned OtherW,
                                bool OtherSigned);
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
  /// Render the inline-form expression for a pure arith op that
  /// `isInlineable` returned true for. Operands recursively go
  /// through `exprFor`, so a tree of inlineable ops collapses
  /// into a single readable expression. Always parenthesized so
  /// the surrounding precedence is unambiguous.
  std::string renderInlineExpr(mlir::Operation *Op);

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

  // T3 — when an output port is narrowed below the IR's return type
  // (the runtime ABI returned i64 but the user's `fi(_, signed,
  // W, F)` cap is W bits), record the narrowed width per result
  // index so emitReturn wraps the RHS with an explicit `<W>'(...)`
  // size cast, matching the port's logic width and silencing
  // Verilator's WIDTHTRUNC.
  std::vector<unsigned> OutNarrowedW;

  // Phase 5.6.2b — source-comment forwarding. Tracks the last
  // source line per file that the emitter has already considered
  // (either emitted or scanned for comments), so the next op's
  // leading-comment scan starts at the right place. Reset per
  // function so each module's body starts a fresh scan window.
  // The scan is also scoped to the current function's line range
  // (CommentMinLine .. CommentMaxLine) so script-header comments
  // and other file-level prose outside the function body don't
  // leak into the emitted module.
  llvm::StringMap<uint32_t> LastEmittedLine;
  // Phase 5.6.4 — trailing-comment scan also looks at the line of
  // the most recently emitted op for a `% ...` after non-WS code,
  // emitting it as `// ...` on the next line. `LastTailEmittedLine`
  // tracks which lines have already had their trailing comment
  // processed so we don't emit duplicates when several ops share
  // a source line.
  llvm::StringMap<uint32_t> LastTailEmittedLine;
  std::string CommentFile;
  uint32_t CommentMinLine = 0;
  uint32_t CommentMaxLine = 0;

  // Phase 5.6.3 — slot-output collapse. When a stack slot has the
  // same `matlab.name` as an output port and every load of it
  // feeds the func.return that drives that port, the slot and the
  // port can share one signal — no separate `data_out_1` scratch
  // signal, no `data_out = data_out_1;` epilogue. This map tracks
  // the matched alloca → result-index pairs; populated once per
  // function in `emitModuleForFunc`.
  llvm::DenseMap<mlir::Operation *, unsigned> SlotMergedToOut;
  // Helper: scan the source range (LastEmittedLine[file] .. line-1]
  // for the op at `Loc` and emit any `% ...` comment-only lines
  // there as `// ...` SV comments at the given indent. No-op when
  // SM is null or the op has no FileLineColLoc.
  void emitLeadingCommentsBefore(mlir::Location Loc, int Indent);

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
  // B1 — per-function set of saturation-helper specs needed by the
  // body. Populated by `collectSatHelpers` from
  // `matlab.fi_sat_w` attrs on `arith.select`. Each entry is
  // (signed, sat-width, input-width). Stored as a sorted
  // dedup'd vector so emit order is stable across runs.
  struct SatHelperKey {
    bool Signed;
    unsigned SatW;
    unsigned InputW;
    bool operator==(const SatHelperKey &O) const {
      return Signed == O.Signed && SatW == O.SatW && InputW == O.InputW;
    }
    bool operator<(const SatHelperKey &O) const {
      if (Signed != O.Signed) return Signed < O.Signed;
      if (InputW != O.InputW) return InputW < O.InputW;
      return SatW < O.SatW;
    }
  };
  std::vector<SatHelperKey> SatHelpers;
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

unsigned Emitter::renderedWidthOf(mlir::Value V) {
  // Same logic as `name()`'s register-get short-circuit: a
  // persistent get reads as the register's typed signal whose
  // width is `Persists[].Width`, not the IR-level f64/i64 ABI
  // type. Trace one hop through a recognized get-call op.
  if (auto *Op = V.getDefiningOp()) {
    auto It = GetSiteToReg.find(Op);
    if (It != GetSiteToReg.end()) {
      return Persists[It->second].Width;
    }
  }
  return widthOf(V.getType());
}

std::string Emitter::exprForInContext(mlir::Value V, unsigned OtherW,
                                       bool OtherSigned) {
  if (OtherW == 0) return exprFor(V);
  // Only re-render when V is a foldable constant. Non-constants
  // already have their own concrete width and the SV
  // self-determined-width rule handles the promotion.
  auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return exprFor(V);
  if (auto Lit = tryReemitConstAtWidth(V, OtherW, OtherSigned))
    return *Lit;
  return exprFor(V);
}

std::optional<std::string>
Emitter::tryReemitConstAtWidth(mlir::Value V, unsigned W, bool Signed) {
  if (W == 0) return std::nullopt;
  auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
  if (!C) return std::nullopt;
  auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue());
  if (!IA) return std::nullopt;
  int64_t Val = IA.getInt();
  if (Signed) {
    int64_t Lo = (W >= 64) ? std::numeric_limits<int64_t>::min()
                           : -(int64_t(1) << (W - 1));
    int64_t Hi = (W >= 64) ? std::numeric_limits<int64_t>::max()
                           : ((int64_t(1) << (W - 1)) - 1);
    if (Val < Lo || Val > Hi) return std::nullopt;
  } else {
    if (Val < 0) return std::nullopt;
    uint64_t Hi =
        (W >= 64) ? ~uint64_t(0) : ((uint64_t(1) << W) - 1);
    if ((uint64_t)Val > Hi) return std::nullopt;
  }
  auto Ty = mlir::IntegerType::get(V.getContext(), W);
  return intLiteral(Val, Ty, Signed);
}

bool Emitter::isInlineable(mlir::Value V) {
  // Inline pure single-use arith ops at their use site so the
  // emitted SV reads as ordinary expressions rather than the
  // dataflow trace `vN_1 = ...; vM_1 = vN_1 op ...; ...`. Single
  // use because computing twice would expand the netlist; same
  // block because lifting an op into a control-flow region (or
  // across a yield) would change the surrounding latch-guard
  // dance.
  auto *Op = V.getDefiningOp();
  if (!Op) return false;
  if (Op->getNumResults() != 1) return false;
  if (!V.hasOneUse()) return false;
  // Don't inline values that the FSM/persistent path consumes
  // specially — those routes need the named signal at the use
  // site (register signal, _next signal, FSM enum literal).
  if (GetSiteToReg.contains(Op)) return false;
  if (SetSiteToReg.contains(Op)) return false;
  if (Suppress.contains(Op)) return false;
  // Only same-block uses. Crossing into an scf.if region body
  // would inline an unconditional computation under a guard,
  // which is harmless for pure ops but masks the structure.
  // Crossing OUT of a region (a yield consumer) would be even
  // less local. Keep the rule simple: definer and the sole
  // user must share a parent block.
  mlir::Operation *User = (*V.getUsers().begin());
  if (User->getBlock() != Op->getBlock()) return false;
  llvm::StringRef N = Op->getName().getStringRef();
  // Pure datapath arith ops.
  if (N == "arith.addi" || N == "arith.subi" ||
      N == "arith.muli" ||
      N == "arith.andi" || N == "arith.ori"  || N == "arith.xori" ||
      N == "arith.shli" || N == "arith.shrsi"|| N == "arith.shrui"||
      N == "arith.divsi"|| N == "arith.divui"||
      N == "arith.remsi"|| N == "arith.remui")
    return true;
  // Comparisons and selects.
  if (mlir::isa<mlir::arith::CmpIOp, mlir::arith::CmpFOp,
                mlir::arith::SelectOp>(Op))
    return true;
  // Width casts.
  if (N == "arith.extsi" || N == "arith.extui" ||
      N == "arith.trunci")
    return true;
  // FP→int conversions appear only as the typed re-cast
  // LowerScalarSlots inserts for `slot_iN = persist_get_f64`;
  // exprFor's fptosi branch unwraps them to the typed register
  // signal, so they inline at the use site cleanly.
  if (N == "arith.fptosi" || N == "arith.fptoui")
    return true;
  // matlab.* surviving binops (handled by the dispatcher's matlab
  // fallback) — same inlining contract.
  if (N == "matlab.add" || N == "matlab.sub" ||
      N == "matlab.matmul" || N == "matlab.emul" ||
      N == "matlab.eq" || N == "matlab.ne" ||
      N == "matlab.lt" || N == "matlab.le" ||
      N == "matlab.gt" || N == "matlab.ge" ||
      N == "matlab.short_or" || N == "matlab.short_and")
    return true;
  // Plain LLVM scalar load of a slot — single-use loads inline as
  // the slot name (or `<arr>[<idx>]` for GEP-based loads). Keeps
  // the canonical "spill-load-spill" patter from showing up as a
  // chain of `vN_1 = slot;` aliases right before each consumer.
  if (mlir::isa<mlir::LLVM::LoadOp>(Op))
    return true;
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
    // type. When the IR-level type is wider than the register's
    // storage width AND a consumer of the get's result actually
    // uses the wider value (e.g. `addi(i64, get_i64)` from Stage
    // F's array-persistent rewrite where the get returns i64 to
    // match the runtime ABI but the underlying register is sized
    // to `fi_wl`), wrap with an explicit `W'($signed(...))` so SV
    // sees a properly sign-extended operand. Verilator otherwise
    // flags WIDTHEXPAND and the SV semantics would zero-extend the
    // raw bits, corrupting negative values. Skip the wrap when
    // every consumer is an `arith.trunci` back down to the
    // register width — the trunci already discards the wider bits.
    auto It = GetSiteToReg.find(Op);
    if (It != GetSiteToReg.end()) {
      auto &P = Persists[It->second];
      unsigned ResultW = widthOf(Op->getResult(0).getType());
      bool NeedsExtend = false;
      if (ResultW > P.Width && P.Width > 0) {
        for (mlir::Operation *U : Op->getResult(0).getUsers()) {
          // arith.trunci's SV rendering is `WT'(EXPR)` which is
          // already an explicit width cast — no extra sign-extend
          // wrapping needed regardless of the trunci's target
          // width (the cast handles the conversion). Same goes
          // for arith.extsi / arith.extui downstream.
          if (mlir::isa<mlir::arith::TruncIOp, mlir::arith::ExtSIOp,
                        mlir::arith::ExtUIOp>(U)) continue;
          NeedsExtend = true;
          break;
        }
      }
      if (NeedsExtend) {
        std::ostringstream S;
        S << ResultW << "'($signed(" << P.Name << "))";
        return S.str();
      }
      return P.Name;
    }
    // Phase 5.6.3: pure single-use arith ops inline at use site.
    // Recursively builds a parenthesized expression so trees of
    // inlineable ops collapse into one readable line.
    if (isInlineable(V))
      return renderInlineExpr(Op);
  }
  return name(V);
}

/// Drop a single matching outer paren pair from `S` so a top-level
/// inline expression embedded in an unambiguous context (RHS of
/// assignment, body of `if (...)`, `case` selector) doesn't render
/// with double parens. Conservative: only strips when the entire
/// string is `(...)` with the open paren at index 0 and the close
/// paren at the very end and they match.
static std::string stripOuterParens(const std::string &S) {
  if (S.size() < 2 || S.front() != '(' || S.back() != ')') return S;
  int Depth = 0;
  for (size_t I = 0; I < S.size(); ++I) {
    if (S[I] == '(') ++Depth;
    else if (S[I] == ')') {
      --Depth;
      if (Depth == 0 && I + 1 != S.size()) return S;
    }
  }
  return S.substr(1, S.size() - 2);
}

std::string Emitter::renderInlineExpr(mlir::Operation *Op) {
  llvm::StringRef N = Op->getName().getStringRef();
  // Load: render as the slot name (or the GEP address string for
  // array element loads). No parens — a load is just a reference.
  if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
    if (auto *AddrOp = L.getAddr().getDefiningOp()) {
      auto It = GepAddr.find(AddrOp);
      if (It != GepAddr.end()) return It->second;
    }
    return name(L.getAddr());
  }
  // FSM-aware rendering for cmp ops mirrors emitCmp{,F}: when the
  // peer side of a comparison against a recognized FSM register's
  // get-call resolves to a case value, render the enum literal
  // instead of the raw integer constant.
  auto fsmEnum = [&](mlir::Value RegSide, mlir::Value ConstSide,
                     std::string &Out) -> bool {
    auto E = fsmEnumLiteralForConstAgainst(RegSide, ConstSide);
    if (E.empty()) return false;
    Out = E;
    return true;
  };
  if (auto C = mlir::dyn_cast<mlir::arith::CmpIOp>(Op)) {
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
    std::string L = exprFor(C.getLhs());
    std::string R = exprFor(C.getRhs());
    if (!fsmEnum(C.getLhs(), C.getRhs(), R))
      fsmEnum(C.getRhs(), C.getLhs(), L);
    std::ostringstream S; S << "(" << L << " " << SvOp.str() << " " << R << ")";
    return S.str();
  }
  if (auto C = mlir::dyn_cast<mlir::arith::CmpFOp>(Op)) {
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
    default: return name(Op->getResult(0));  // unsupported pred
    }
    std::string L = exprFor(C.getLhs());
    std::string R = exprFor(C.getRhs());
    if (!fsmEnum(C.getLhs(), C.getRhs(), R))
      fsmEnum(C.getRhs(), C.getLhs(), L);
    std::ostringstream S; S << "(" << L << " " << SvOp.str() << " " << R << ")";
    return S.str();
  }
  if (auto S = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
    // B1 — saturation pattern collapses to a helper-function call.
    if (auto Wattr =
            S->getAttrOfType<mlir::IntegerAttr>("matlab.fi_sat_w")) {
      bool Sgn = S->getAttrOfType<mlir::BoolAttr>(
                       "matlab.fi_sat_signed").getValue();
      unsigned SatW = (unsigned)Wattr.getInt();
      unsigned InputW = widthOf(S.getResult().getType());
      mlir::Value In = satHelperInput(S, Sgn);
      std::ostringstream Out;
      Out << satHelperName(Sgn, SatW, InputW) << "(" << exprFor(In)
          << ")";
      return Out.str();
    }
    std::ostringstream Out;
    Out << "(" << exprFor(S.getCondition()) << " ? "
        << exprFor(S.getTrueValue()) << " : "
        << exprFor(S.getFalseValue()) << ")";
    return Out.str();
  }
  if (N == "arith.extsi" || N == "arith.extui" || N == "arith.trunci") {
    bool Signed = (N == "arith.extsi");
    unsigned W = widthOf(Op->getResult(0).getType());
    // S5 — collapse chained same-direction extends. An `arith.extsi
    // i16 → i32` feeding `arith.extsi i32 → i64` is semantically a
    // single `extsi i16 → i64`; rendering it as
    // `64'($signed(32'($signed(x))))` reads as a redundant cast
    // chain. Skip the inner cast at emit time when the operand is
    // a same-direction ext.
    mlir::Value Src = Op->getOperand(0);
    while (auto *In = Src.getDefiningOp()) {
      llvm::StringRef IN = In->getName().getStringRef();
      bool SameDir = (Signed && IN == "arith.extsi") ||
                     (!Signed && N == "arith.extui" &&
                      IN == "arith.extui") ||
                     (N == "arith.trunci" && IN == "arith.trunci");
      if (!SameDir) break;
      // For ext: child must widen to a width ≤ this op's width.
      // For trunci: child must narrow to a width ≥ this op's width.
      unsigned ChildW = widthOf(In->getResult(0).getType());
      if (N == "arith.trunci") {
        if (ChildW < W) break;
      } else if (ChildW > W) {
        break;
      }
      Src = In->getOperand(0);
    }
    std::ostringstream Out;
    Out << W << "'(";
    if (Signed) Out << "$signed(";
    // S5 — the outer cast `W'(...)` (and inner `$signed(...)` for
    // sign-ext) already provides paren grouping; strip a redundant
    // outer-paren level off the operand expression so the chain
    // `64'($signed((32'($signed(a)) * 32'($signed(b)))))` reads as
    // `64'($signed(32'($signed(a)) * 32'($signed(b))))`. Identical
    // gates, less visual nesting.
    Out << stripOuterParens(exprFor(Src));
    if (Signed) Out << ")";
    Out << ")";
    return Out.str();
  }
  // `arith.fptosi` of an `f64` persistent-get result is the
  // canonical "typed read of a register" pattern — the runtime
  // ABI returns f64 from `matlab_global_get_*`, but the SV
  // emitter renders the get as a typed register signal of the
  // register's actual width. The cast-to-iN is therefore inert
  // at the SV level: just unwrap to the underlying typed signal.
  // Without this, body-level assignments like
  // `state_display = current_state` (lowered through a slot
  // with the cast inserted by LowerScalarSlots) would render
  // the cast as an unsupported op.
  if (N == "arith.fptosi" || N == "arith.fptoui") {
    if (auto *In = Op->getOperand(0).getDefiningOp()) {
      auto It = GetSiteToReg.find(In);
      if (It != GetSiteToReg.end()) {
        unsigned W = widthOf(Op->getResult(0).getType());
        std::string Inner = exprFor(Op->getOperand(0));
        // Always wrap in an explicit `<W>'(...)` size cast: the
        // SV-rendered register signal's effective width depends on
        // whether the persist is FSM-encoded (enum bits) or a raw
        // integer register (P.Width), and the cast target is the
        // user-facing slot/output width — they may differ. The
        // size cast is a no-op when widths match and the canonical
        // width-conversion idiom otherwise.
        if (W == 0) return Inner;
        bool IsFSM = false;
        for (auto &FI : FSMs)
          if (FI.RegIndex == It->second) { IsFSM = true; break; }
        // Non-FSM and same-width: skip the cast for cleaner SV.
        if (!IsFSM && Persists[It->second].Width == W) return Inner;
        std::ostringstream Out;
        Out << W << "'(" << Inner << ")";
        return Out.str();
      }
    }
  }
  // `arith.xori(x, all-ones)` is the canonical lowering of
  // `bitcmp(x)` / bitwise NOT. Rendering as `x ^ -1` is
  // synthesizable but reads wrong — `~x` is the SV idiom for a
  // NOT gate and what every hand-written RTL author writes.
  // Match either operand position (constant folding can't reorder
  // a non-commutative op, but xori IS commutative — be safe).
  auto isAllOnesIntConst = [](mlir::Value V) -> bool {
    auto *D = V.getDefiningOp();
    if (!D) return false;
    llvm::APInt Val;
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D)) {
      auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue());
      if (!IA) return false;
      Val = IA.getValue();
    } else if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D)) {
      auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue());
      if (!IA) return false;
      Val = IA.getValue();
    } else {
      return false;
    }
    return Val.isAllOnes();
  };
  if (N == "arith.xori" && Op->getNumOperands() == 2) {
    if (isAllOnesIntConst(Op->getOperand(1)))
      return "~" + exprFor(Op->getOperand(0));
    if (isAllOnesIntConst(Op->getOperand(0)))
      return "~" + exprFor(Op->getOperand(1));
  }
  // Const-fold integer arithmetic on two integer constants. The
  // pipeline often leaves `(N - 1)` style 1-based-to-0-based index
  // conversions where `N` is a per-iteration constant from the
  // unrolled for-loop; rendering them as `(32'sd1 - 32'sd1)` is
  // synthesizable but visually noisy. Fold here so subscript reads
  // emit `arr[0]` instead of `arr[(32'sd1 - 32'sd1)]`. Bounded to
  // `addi/subi/muli` on signless integer constants — anything else
  // falls through to the binop renderer below.
  if (Op->getNumOperands() == 2 &&
      (N == "arith.addi" || N == "arith.subi" || N == "arith.muli")) {
    auto getIntConst = [](mlir::Value V, int64_t &Out) -> bool {
      auto *D = V.getDefiningOp();
      if (!D) return false;
      if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D)) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
          Out = IA.getInt();
          return true;
        }
      }
      if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D)) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
          Out = IA.getInt();
          return true;
        }
      }
      return false;
    };
    int64_t LhsC, RhsC;
    if (getIntConst(Op->getOperand(0), LhsC) &&
        getIntConst(Op->getOperand(1), RhsC)) {
      int64_t Folded = N == "arith.addi" ? LhsC + RhsC :
                       N == "arith.subi" ? LhsC - RhsC : LhsC * RhsC;
      auto IT = mlir::dyn_cast<mlir::IntegerType>(
          Op->getResult(0).getType());
      if (IT) {
        // Narrow to the result width's representable range to
        // mirror SV's wraparound semantics on overflow. Then format
        // via the existing intLiteral helper for consistent `W'sdN`
        // / `W'dN` rendering.
        return intLiteral(Folded, IT, /*Signed=*/true);
      }
    }
  }
  // Binary arith ops with an SV operator equivalent.
  llvm::StringRef SvOp;
  if (N == "arith.addi" || N == "matlab.add") SvOp = "+";
  else if (N == "arith.subi" || N == "matlab.sub") SvOp = "-";
  else if (N == "arith.muli" || N == "matlab.matmul" ||
           N == "matlab.emul") SvOp = "*";
  else if (N == "arith.andi") SvOp = "&";
  else if (N == "arith.ori")  SvOp = "|";
  else if (N == "arith.xori") SvOp = "^";
  else if (N == "arith.shli") SvOp = "<<";
  else if (N == "arith.shrsi") SvOp = ">>>";
  else if (N == "arith.shrui") SvOp = ">>";
  else if (N == "arith.divsi" || N == "arith.divui") SvOp = "/";
  else if (N == "arith.remsi" || N == "arith.remui") SvOp = "%";
  else if (N == "matlab.eq") SvOp = "==";
  else if (N == "matlab.ne") SvOp = "!=";
  else if (N == "matlab.lt") SvOp = "<";
  else if (N == "matlab.le") SvOp = "<=";
  else if (N == "matlab.gt") SvOp = ">";
  else if (N == "matlab.ge") SvOp = ">=";
  else if (N == "matlab.short_or") SvOp = "||";
  else if (N == "matlab.short_and") SvOp = "&&";
  if (!SvOp.empty() && Op->getNumOperands() == 2) {
    bool IsCmp = (N == "matlab.eq" || N == "matlab.ne" ||
                  N == "matlab.lt" || N == "matlab.le" ||
                  N == "matlab.gt" || N == "matlab.ge");
    // S1 — context-aware constant rendering.
    unsigned LW = renderedWidthOf(Op->getOperand(0));
    unsigned RW = renderedWidthOf(Op->getOperand(1));
    std::string L = exprForInContext(Op->getOperand(0), RW, true);
    std::string R = exprForInContext(Op->getOperand(1), LW, true);
    if (IsCmp && (N == "matlab.eq" || N == "matlab.ne")) {
      if (auto E = fsmEnumLiteralForConstAgainst(
              Op->getOperand(0), Op->getOperand(1)); !E.empty())
        R = E;
      else if (auto E = fsmEnumLiteralForConstAgainst(
                   Op->getOperand(1), Op->getOperand(0)); !E.empty())
        L = E;
    }
    // T3-related — width-extend operands when the result is wider
    // than either operand's rendered width (the runtime ABI's f64
    // get followed by a narrowing render leaves operand widths
    // smaller than the matlab.* op's result width). Insert
    // `<W>'($signed(...))` so SV's self-determined-width rule
    // doesn't expand-warn. Skip cmps (i1 result) and i1-operand
    // short-circuit booleans.
    bool ResIsArith =
        (N == "matlab.add" || N == "matlab.sub" ||
         N == "matlab.matmul" || N == "matlab.emul");
    if (ResIsArith) {
      unsigned ResW = widthOf(Op->getResult(0).getType());
      auto wrap = [&](std::string &S, unsigned W) {
        if (W == 0 || W >= ResW) return;
        std::ostringstream Wr;
        Wr << ResW << "'($signed(" << S << "))";
        S = Wr.str();
      };
      wrap(L, LW);
      wrap(R, RW);
    }
    // S3 — flatten redundant parens for same-operator chains.
    // `((a && b) && c)` becomes `(a && b && c)`. When an operand
    // is itself a parenthesized chain of the SAME operator, the
    // associativity of `&&` / `||` / `+` / `*` / `&` / `|` / `^`
    // makes the inner parens redundant; SV parses left-to-right
    // and the gates collapse to the same circuit.
    auto stripIfSameOp = [&](std::string &S, mlir::Value V) {
      auto *D = V.getDefiningOp();
      if (!D) return;
      if (D->getName().getStringRef() != N) return;
      if (S.size() < 2 || S.front() != '(' || S.back() != ')')
        return;
      S = S.substr(1, S.size() - 2);
    };
    bool Associative = (N == "matlab.short_and" ||
                        N == "matlab.short_or" || N == "arith.addi" ||
                        N == "arith.muli" || N == "arith.andi" ||
                        N == "arith.ori" || N == "arith.xori" ||
                        N == "matlab.add" || N == "matlab.matmul" ||
                        N == "matlab.emul");
    if (Associative) {
      stripIfSameOp(L, Op->getOperand(0));
      stripIfSameOp(R, Op->getOperand(1));
    }
    std::ostringstream Out;
    Out << "(" << L << " " << SvOp.str() << " " << R << ")";
    return Out.str();
  }
  // Unrecognized — fall back to the named result.
  return name(Op->getResult(0));
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
  // Output names. Phase 5.6.2a: prefer the MATLAB return-variable name
  // (set by Lowering as `matlab.name` on each func result) so a
  // signature like `[data_out, overflow] = alu_16bit(...)` emits the
  // ports `output ... data_out, output ... overflow` instead of the
  // generic `y, y1, ...` fallback.
  OutNames.clear();
  OutNames.reserve(FT.getNumResults());
  OutNarrowedW.assign(FT.getNumResults(), 0);
  for (unsigned I = 0; I < FT.getNumResults(); ++I) {
    std::string Nm;
    if (auto S = F.getResultAttrOfType<mlir::StringAttr>(I, "matlab.name"))
      Nm = sanitize(S.getValue());
    if (Nm.empty()) Nm = (I == 0) ? "y" : ("y" + std::to_string(I));
    while (Used.contains(Nm)) Nm += "_";
    Used.insert(Nm);
    OutNames.push_back(Nm);
  }
  // Phase 5.6.3: bind each merged alloca's signal name to its
  // matched output port name BEFORE declarePrelude runs. Stores
  // into the slot then render `<port> = ...;` directly.
  for (auto &Pair : SlotMergedToOut) {
    mlir::Value SlotVal = Pair.first->getResult(0);
    Names[SlotVal] = OutNames[Pair.second];
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
    // Phase 5.6 Stage B: a `!llvm.ptr` arg with `matlab.array_n` +
    // `matlab.fi_wl` attrs is a vector port. Render as
    // `input logic [W-1:0] name [N]` (signedness from the
    // `matlab.fi_signed` attr, default to signed for fi).
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(FT.getInput(I))) {
      auto NA = F.getArgAttrOfType<mlir::IntegerAttr>(I, "matlab.array_n");
      auto WLA = F.getArgAttrOfType<mlir::IntegerAttr>(I, "matlab.fi_wl");
      if (NA && WLA) {
        unsigned W = (unsigned)WLA.getInt();
        bool Signed = true;
        if (auto SA =
                F.getArgAttrOfType<mlir::IntegerAttr>(I, "matlab.fi_signed"))
          Signed = SA.getInt() != 0;
        auto IT = mlir::IntegerType::get(F.getContext(), W);
        OS << "    input  " << svType(IT, Signed) << " " << ArgNames[I]
           << " [" << NA.getInt() << "]";
        continue;
      }
    }
    // Scalar (non-vector) port. Honor the `% hdl: port(name, fi,
    // unsigned, W, F)` pragma's signedness via the `matlab.fi_signed`
    // arg attr (set by ApplyPortTypePragmas). MLIR IntegerType is
    // signless; defaulting to signed in `svType(Type)` over-claims
    // for unsigned ports declared by pragma.
    bool ScalarSigned = true;
    if (auto SA =
            F.getArgAttrOfType<mlir::IntegerAttr>(I, "matlab.fi_signed"))
      ScalarSigned = SA.getInt() != 0;
    OS << "    input  " << svType(FT.getInput(I), ScalarSigned)
       << " " << ArgNames[I];
  }
  for (unsigned I = 0; I < FT.getNumResults(); ++I) {
    if (!First) OS << ",\n";
    First = false;
    // For results that come from a persistent get, the function's
    // declared result type is the runtime ABI's f64 — render the SV
    // port at the register's actual integer width instead. Also
    // honor the register's signedness (`fi(_, 0, _, _)` → unsigned)
    // so the output port matches the user's declared spec.
    mlir::Type T = FT.getResult(I);
    bool ResSigned = true;
    bool SawPersist = false;
    // T3 — narrow the port to the user's saturation cap when the
    // return value's defining chain culminates in a tagged
    // saturating SelectOp. The runtime ABI's i64 result type
    // doesn't reflect the user's `fi(_, signed, W, F)` width spec;
    // walking back through ext/trunc adapters to the outer sat
    // SelectOp recovers the real width and signedness from the
    // `matlab.fi_sat_w` / `matlab.fi_sat_signed` attrs that
    // LowerFiSaturate set.
    F.walk([&](mlir::func::ReturnOp R) {
      if (R.getNumOperands() <= I) return;
      mlir::Value V = R.getOperand(I);
      for (int Hop = 0; V && Hop < 6; ++Hop) {
        auto *D = V.getDefiningOp();
        if (!D) return;
        if (auto S = mlir::dyn_cast<mlir::arith::SelectOp>(D)) {
          if (auto Wattr = S->getAttrOfType<mlir::IntegerAttr>(
                  "matlab.fi_sat_w")) {
            unsigned W = (unsigned)Wattr.getInt();
            if (W > 0) {
              bool Sgn = true;
              if (auto SA = S->getAttrOfType<mlir::BoolAttr>(
                      "matlab.fi_sat_signed"))
                Sgn = SA.getValue();
              T = mlir::IntegerType::get(F.getContext(), W);
              ResSigned = Sgn;
              SawPersist = true;
              OutNarrowedW[I] = W;
              return;
            }
          }
        }
        if (mlir::isa<mlir::arith::FPToSIOp, mlir::arith::FPToUIOp,
                      mlir::arith::TruncIOp, mlir::arith::ExtSIOp,
                      mlir::arith::ExtUIOp>(D)) {
          V = D->getOperand(0);
          continue;
        }
        break;
      }
    });
    F.walk([&](mlir::func::ReturnOp R) {
      if (SawPersist) return;
      if (R.getNumOperands() <= I) return;
      auto *Op = R.getOperand(I).getDefiningOp();
      if (!Op) return;
      // Direct: return value IS a persistent get.
      if (auto It = GetSiteToReg.find(Op); It != GetSiteToReg.end()) {
        auto &P = Persists[It->second];
        T = mlir::IntegerType::get(F.getContext(), P.Width);
        ResSigned = P.Signed;
        SawPersist = true;
        return;
      }
      // Indirect: post-B1 fix shape — return is `llvm.load %slot`
      // where the only `llvm.store %v, %slot` writes a value
      // produced by `arith.fptosi (persist_get_f64 -> iN)`. Walk
      // through to recover the register's signedness so the port
      // matches the source-level `uint8`/`int8` declaration
      // (e.g. moore_fsm's `state_display` MUST be unsigned, the
      // default-signed rule overrides the source).
      mlir::Value V = R.getOperand(I);
      for (int Hop = 0; V && Hop < 4; ++Hop) {
        auto *D = V.getDefiningOp();
        if (!D) return;
        if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(D)) {
          mlir::Value Slot = L.getAddr();
          mlir::Operation *LastStore = nullptr;
          for (mlir::Operation *U : Slot.getUsers()) {
            if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(U))
              if (S.getAddr() == Slot) LastStore = S;
          }
          if (!LastStore) return;
          V = LastStore->getOperand(0);
          continue;
        }
        if (mlir::isa<mlir::arith::FPToSIOp, mlir::arith::FPToUIOp,
                      mlir::arith::TruncIOp, mlir::arith::ExtSIOp,
                      mlir::arith::ExtUIOp>(D)) {
          V = D->getOperand(0);
          continue;
        }
        if (auto It = GetSiteToReg.find(D); It != GetSiteToReg.end()) {
          auto &P = Persists[It->second];
          T = mlir::IntegerType::get(F.getContext(), P.Width);
          ResSigned = P.Signed;
          SawPersist = true;
          return;
        }
        return;
      }
    });
    // For non-persistent results (e.g. `r = a + b` where r is just
    // a return value), trace back through the func.return op's
    // operand defining op chain to find an `fi_signed` attr. The
    // frontend tags fi-aware ops (matlab.fi.const, matlab.add,
    // matlab.sub, matlab.fi.cast, etc.) with the spec they
    // produce; LowerFixedPoint carries those attrs onto the
    // arith.constant / etc. that survives. Without this thread
    // an unsigned `r = a + b` would render `output logic signed
    // [W-1:0] r` even though both operands and the result are
    // declared unsigned.
    if (!SawPersist) {
      F.walk([&](mlir::func::ReturnOp R) {
        if (R.getNumOperands() <= I) return;
        mlir::Value V = R.getOperand(I);
        for (int Hop = 0; V && Hop < 8; ++Hop) {
          // Block arguments (entry-block func args): look at the
          // arg attrs the call-refinement / pragma path threaded.
          // For `r = a + b`, after recursing into the addi's LHS
          // we land on arg0; reading `matlab.fi_signed` there
          // closes the trace.
          if (auto BA = mlir::dyn_cast<mlir::BlockArgument>(V)) {
            auto *Owner = BA.getOwner();
            auto Fn = mlir::dyn_cast<mlir::func::FuncOp>(
                Owner->getParentOp());
            if (!Fn) return;
            unsigned ArgI = BA.getArgNumber();
            if (auto IA = Fn.getArgAttrOfType<mlir::IntegerAttr>(
                    ArgI, "matlab.fi_signed"))
              ResSigned = IA.getInt() != 0;
            return;
          }
          auto *D = V.getDefiningOp();
          if (!D) return;
          if (auto SA = D->getAttr("fi_signed")) {
            if (auto BA = mlir::dyn_cast<mlir::BoolAttr>(SA))
              ResSigned = BA.getValue();
            else if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(SA))
              ResSigned = IA.getInt() != 0;
            return;
          }
          // T1 — `matlab.unsigned` unit attr is set on
          // arith.constant ops produced by IntCastConstantFold for
          // uint8 / uint16 / uint32 / uint64 literal casts. Honour
          // it so a function that returns `uint8(N)` shows up as an
          // unsigned port.
          if (D->hasAttr("matlab.unsigned")) {
            ResSigned = false;
            return;
          }
          // matlab.load: trace through last store value.
          if (D->getName().getStringRef() == "matlab.load" &&
              D->getNumOperands() == 1) {
            mlir::Value Slot = D->getOperand(0);
            mlir::Operation *LastStore = nullptr;
            for (mlir::Operation *U : Slot.getUsers()) {
              if (U->getName().getStringRef() == "matlab.store" &&
                  U->getNumOperands() == 2 && U->getOperand(1) == Slot)
                LastStore = U;
            }
            if (!LastStore) return;
            V = LastStore->getOperand(0);
            continue;
          }
          // llvm.load: same trace, post-LowerScalarSlots shape.
          if (auto LL = mlir::dyn_cast<mlir::LLVM::LoadOp>(D)) {
            mlir::Value Slot = LL.getAddr();
            mlir::Operation *LastStore = nullptr;
            for (mlir::Operation *U : Slot.getUsers()) {
              if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(U))
                if (S.getAddr() == Slot) LastStore = S;
            }
            if (!LastStore) return;
            V = LastStore->getOperand(0);
            continue;
          }
          if (mlir::isa<mlir::arith::TruncIOp, mlir::arith::ExtSIOp,
                        mlir::arith::ExtUIOp,
                        mlir::arith::FPToSIOp, mlir::arith::FPToUIOp>(D)) {
            V = D->getOperand(0);
            continue;
          }
          // Binary arith ops on signless ints — the result spec
          // typically matches the LHS. Recurse one operand.
          if (D->getNumOperands() == 2 &&
              mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                        mlir::arith::MulIOp, mlir::arith::AndIOp,
                        mlir::arith::OrIOp, mlir::arith::XOrIOp>(D)) {
            V = D->getOperand(0);
            continue;
          }
          return;
        }
      });
    }
    OS << "    output " << svType(T, ResSigned) << " " << OutNames[I];
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
      // Phase 5.6.3: a slot merged into an output port shares the
      // port's signal name (already bound in `emitPortList`); no
      // separate declaration or prelude pre-init.
      if (SlotMergedToOut.contains(A.getOperation())) return;
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
      // Phase 5.6.3: inlineable values render at use site; no
      // top-level `logic` declaration and no prelude pre-init.
      if (isInlineable(V)) continue;
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
      Ty = svType(T, P.Signed);
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
  // Inlined at use site — no statement of our own.
  if (isInlineable(Op.getResult(0))) return;
  // S1 — when one operand is a constant and the other has a known
  // narrower rendered width, narrow the constant's literal width
  // so `count_reg + 8'sd1` reads as `count_reg + 4'd1`.
  unsigned LW = renderedWidthOf(Op.getOperand(0));
  unsigned RW = renderedWidthOf(Op.getOperand(1));
  std::string LExpr = exprForInContext(Op.getOperand(0), RW, true);
  std::string RExpr = exprForInContext(Op.getOperand(1), LW, true);
  indent(Indent);
  OS << name(Op.getResult(0)) << " = "
     << LExpr << " " << SvOp.str() << " " << RExpr << ";\n";
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
  if (isInlineable(C.getResult())) return;
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
  unsigned LW = renderedWidthOf(C.getLhs());
  unsigned RW = renderedWidthOf(C.getRhs());
  std::string LExpr = exprForInContext(C.getLhs(), RW, true);
  std::string RExpr = exprForInContext(C.getRhs(), LW, true);
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
  if (isInlineable(C.getResult())) return;
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
  if (isInlineable(S.getResult())) return;
  // B1 — saturation pattern collapses to a helper-function call.
  if (auto Wattr =
          S->getAttrOfType<mlir::IntegerAttr>("matlab.fi_sat_w")) {
    bool Sgn = S->getAttrOfType<mlir::BoolAttr>(
                     "matlab.fi_sat_signed").getValue();
    unsigned SatW = (unsigned)Wattr.getInt();
    unsigned InputW = widthOf(S.getResult().getType());
    mlir::Value In = satHelperInput(S, Sgn);
    indent(Indent);
    OS << name(S.getResult()) << " = "
       << satHelperName(Sgn, SatW, InputW) << "(" << exprFor(In)
       << ");\n";
    return;
  }
  indent(Indent);
  OS << name(S.getResult()) << " = "
     << exprFor(S.getCondition()) << " ? "
     << exprFor(S.getTrueValue()) << " : "
     << exprFor(S.getFalseValue()) << ";\n";
}

void Emitter::emitExtTrunc(mlir::Operation &Op, int Indent) {
  if (isInlineable(Op.getResult(0))) return;
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

bool Emitter::tryEmitSwitchCase(mlir::scf::IfOp Head, int Indent) {
  // Match a chain of `if (<disc> == c0) ... else if (<disc> == c1)
  // ... else if (<disc> == cN) ... else <default>`. Every cmp must
  // be against the SAME `<disc>` and use equality (`arith.cmpi eq`
  // or `matlab.eq`). Constants must reduce to integer values via
  // the existing `exprFor` rendering. The chain is followed via
  // the else-region: it must contain exactly one inner scf.if (and
  // a yield), or it's the default arm.
  struct CaseEntry { std::string ConstExpr; mlir::Region *Body; };
  llvm::SmallVector<CaseEntry, 8> Cases;
  mlir::Region *Default = nullptr;
  mlir::Value Disc;
  mlir::scf::IfOp Cur = Head;
  while (Cur) {
    mlir::Value Cond = Cur.getCondition();
    mlir::Operation *Def = Cond.getDefiningOp();
    if (!Def) return false;
    mlir::Value LhsV;
    mlir::Value RhsV;
    if (auto Cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(Def)) {
      if (Cmp.getPredicate() != mlir::arith::CmpIPredicate::eq)
        return false;
      LhsV = Cmp.getLhs();
      RhsV = Cmp.getRhs();
    } else if (Def->getName().getStringRef() == "matlab.eq" &&
               Def->getNumOperands() == 2) {
      LhsV = Def->getOperand(0);
      RhsV = Def->getOperand(1);
    } else {
      return false;
    }
    // The discriminator must be the same SSA value across every
    // arm. Don't try to be clever about which side is the const —
    // require LHS to be the discriminator and RHS to be the
    // constant. (Both sides being constants would have folded
    // earlier; both being non-const isn't a `switch` shape.)
    if (!Disc) Disc = LhsV;
    else if (Disc != LhsV) return false;
    // RHS must be a constant the emitter can render. Use `exprFor`
    // with a one-shot inlineability check: if RHS doesn't have a
    // defining op or isn't a recognized constant op, bail.
    auto *RDef = RhsV.getDefiningOp();
    if (!RDef) return false;
    bool IsConst = mlir::isa<mlir::arith::ConstantOp,
                              mlir::LLVM::ConstantOp>(RDef) ||
                   RDef->getName().getStringRef() == "matlab.const_int";
    if (!IsConst) return false;
    Cases.push_back({stripOuterParens(exprFor(RhsV)),
                     &Cur.getThenRegion()});
    // Else region: either empty (no default), a chained scf.if
    // (continues the cascade — the next cmp's defining ops live
    // alongside the inner if in the same block, so we look for
    // "exactly one scf.if and one yield, plus zero or more
    // pure operand-chain ops"), or arbitrary body (default arm).
    auto &ER = Cur.getElseRegion();
    if (ER.empty()) { Cur = nullptr; break; }
    auto &EB = ER.front();
    // No-op else (auto-yield only) — no default arm.
    if (std::next(EB.begin()) == EB.end() &&
        mlir::isa<mlir::scf::YieldOp>(EB.front()) &&
        EB.front().getNumOperands() == 0) {
      Cur = nullptr;
      break;
    }
    // Find the (single) inner scf.if. Multiple scf.ifs or zero
    // scf.ifs both fail this match (the latter falls through to
    // the default-arm path).
    mlir::scf::IfOp NextIf;
    bool MultipleIfs = false;
    for (mlir::Operation &TOp : EB) {
      if (auto Inner = mlir::dyn_cast<mlir::scf::IfOp>(&TOp)) {
        if (NextIf) { MultipleIfs = true; break; }
        NextIf = Inner;
      }
    }
    if (!MultipleIfs && NextIf) {
      // Verify the inner if is the LAST non-yield op (so we don't
      // mis-handle `else { sideEffectStore; if (...) {...} }`,
      // which is a real default-arm + nested-if rather than a
      // chain). The yield must immediately follow the inner if.
      auto &Term = EB.back();
      mlir::Operation *PrevToTerm = nullptr;
      for (mlir::Operation &TOp : EB)
        if (&TOp != &Term) PrevToTerm = &TOp;
      if (PrevToTerm == NextIf.getOperation() &&
          mlir::isa<mlir::scf::YieldOp>(Term) &&
          Term.getNumOperands() == 0) {
        Cur = NextIf;
        continue;
      }
    }
    // Anything else in the else region — treat as the default arm.
    Default = &ER;
    Cur = nullptr;
    break;
  }
  // 2+ cases is the bar — a single `if x == c` doesn't benefit
  // from the case form and would just be noisier than the existing
  // if/else rendering.
  if (Cases.size() < 2) return false;
  // If the default region holds the `IfStoreToSelect`-folded
  // remnant of the last user case (`store(select(disc == const,
  // X, Y), slot)`), unfold it back into one more case + a
  // simpler default. Detection: the default's only op is a
  // store whose value is `arith.select(<disc> == <const>, X, Y)`.
  // The select form arises when the last `if (sel == 3) y = in3
  // else y = in0` got folded; preserving the case form keeps the
  // RTL semantically aligned with the user's `switch sel` source.
  std::string ExtraCaseConst;
  std::string ExtraCaseRhs;
  std::string DefaultRhs;
  std::string DefaultLhsName;
  bool DefaultUnfolded = false;
  if (Default && !Default->empty()) {
    auto &DB = Default->front();
    // Find the (single) store op; permit operand-chain ops feeding
    // the stored value (e.g. the matlab.eq + arith.select that
    // IfStoreToSelect produced from the deepest if/else fold).
    mlir::Operation *Store = nullptr;
    int StoreCount = 0;
    for (mlir::Operation &TOp : DB) {
      if (TOp.getName().getStringRef() == "matlab.store" ||
          mlir::isa<mlir::LLVM::StoreOp>(&TOp)) {
        Store = &TOp;
        ++StoreCount;
      }
    }
    if (StoreCount == 1 && Store && Store->getNumOperands() == 2) {
      mlir::Value StVal = Store->getOperand(0);
      mlir::Value StSlot = Store->getOperand(1);
      auto Sel = StVal.getDefiningOp<mlir::arith::SelectOp>();
      if (Sel) {
        mlir::Operation *SCmp = Sel.getCondition().getDefiningOp();
        mlir::Value SLhs;
        mlir::Value SRhs;
        if (SCmp) {
          if (auto C = mlir::dyn_cast<mlir::arith::CmpIOp>(SCmp)) {
            if (C.getPredicate() == mlir::arith::CmpIPredicate::eq) {
              SLhs = C.getLhs(); SRhs = C.getRhs();
            }
          } else if (SCmp->getName().getStringRef() == "matlab.eq" &&
                     SCmp->getNumOperands() == 2) {
            SLhs = SCmp->getOperand(0);
            SRhs = SCmp->getOperand(1);
          }
        }
        // Same discriminator as the rest of the chain, RHS const.
        if (SLhs == Disc && SRhs) {
          auto *RDef = SRhs.getDefiningOp();
          bool IsConst = RDef &&
              (mlir::isa<mlir::arith::ConstantOp,
                          mlir::LLVM::ConstantOp>(RDef) ||
               RDef->getName().getStringRef() == "matlab.const_int");
          if (IsConst) {
            ExtraCaseConst = stripOuterParens(exprFor(SRhs));
            ExtraCaseRhs = stripOuterParens(exprFor(Sel.getTrueValue()));
            DefaultRhs = stripOuterParens(exprFor(Sel.getFalseValue()));
            DefaultLhsName = name(StSlot);
            DefaultUnfolded = true;
          }
        }
      }
    }
  }
  indent(Indent);
  OS << "unique case (" << stripOuterParens(exprFor(Disc)) << ")\n";
  for (auto &E : Cases) {
    indent(Indent + 1);
    OS << E.ConstExpr << ": begin\n";
    emitRegion(*E.Body, Indent + 2);
    indent(Indent + 1);
    OS << "end\n";
  }
  if (DefaultUnfolded) {
    indent(Indent + 1);
    OS << ExtraCaseConst << ": begin\n";
    indent(Indent + 2);
    OS << DefaultLhsName << " = " << ExtraCaseRhs << ";\n";
    indent(Indent + 1);
    OS << "end\n";
    indent(Indent + 1);
    OS << "default: begin\n";
    indent(Indent + 2);
    OS << DefaultLhsName << " = " << DefaultRhs << ";\n";
    indent(Indent + 1);
    OS << "end\n";
  } else if (Default) {
    indent(Indent + 1);
    OS << "default: begin\n";
    emitRegion(*Default, Indent + 2);
    indent(Indent + 1);
    OS << "end\n";
  }
  indent(Indent);
  OS << "endcase\n";
  return true;
}

void Emitter::emitScfIf(mlir::scf::IfOp If, int Indent) {
  // Phase 1 supports both shapes:
  //   - scf.if without results: pure side-effecting branches that store
  //     to slots. Renders as `if (cond) begin ... end else begin ... end`
  //     inside `always_comb`.
  //   - scf.if with results: the values yielded by each arm assign the
  //     `if`'s SSA results. Renders as the same construct, with each
  //     arm writing the result name(s).
  //
  // Switch-case detection runs first: a `switch sel; case 0 ... case 1
  // ... otherwise` source pattern lowers through Sema to a chain of
  // nested `scf.if (sel == cN)`. Rendering that as `unique case
  // (sel)` instead of nested if/else lets the synth tool realize a
  // parallel mux. Only triggers on result-less ifs (the chain that
  // store to shared slots — the typical `data_out = ...` case);
  // result-yielding ifs are rare and don't compose with `unique case`
  // semantics anyway.
  if (If->getNumResults() == 0 && tryEmitSwitchCase(If, Indent))
    return;
  indent(Indent);
  OS << "if (" << stripOuterParens(exprFor(If.getCondition()))
     << ") begin\n";
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
       << stripOuterParens(exprFor(Y.getOperand(I))) << ";\n";
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
  // Phase 5.6.3: single-use loads inline at their consumer via
  // exprFor → renderInlineExpr; emit nothing here.
  if (isInlineable(L.getResult())) return;
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
  OS << AddrExpr << " = " << stripOuterParens(exprFor(S.getValue()))
     << ";\n";
}

void Emitter::emitReturn(mlir::func::ReturnOp R, int Indent) {
  // Drive the output ports.
  if (R.getNumOperands() != OutNames.size()) {
    fail("func.return arity mismatch");
    return;
  }
  for (unsigned I = 0; I < R.getNumOperands(); ++I) {
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
    std::string Rhs = stripOuterParens(Expr);
    // T3 — when the output port is narrowed below the IR's return
    // type (`OutNarrowedW[I]` > 0), wrap the RHS with an explicit
    // `<W>'(...)` size cast so Verilator's WIDTHTRUNC stays
    // silent. Skip the cast when Rhs is the merged-slot port
    // assignment (see below — that path already aliases names).
    if (I < OutNarrowedW.size() && OutNarrowedW[I] > 0 &&
        Rhs != OutWriteNames[I]) {
      std::ostringstream W;
      W << OutNarrowedW[I] << "'(" << Rhs << ")";
      Rhs = W.str();
    }
    // Phase 5.6.3: a slot-merged output port assigns itself
    // here; the body already wrote it through the merged signal
    // name. Suppress the `port = port;` no-op.
    if (Rhs == OutWriteNames[I]) continue;
    indent(Indent);
    OS << OutWriteNames[I] << " = " << Rhs << ";\n";
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
          // S1 — context-aware constant rendering for the inlined
          // operands. `count_reg + 8'sd1` becomes `count_reg + 4'd1`
          // when count_reg is a 4-bit register and the IR's i8
          // constant fits in 4 bits.
          unsigned LW = renderedWidthOf(VOp->getOperand(0));
          unsigned RW = renderedWidthOf(VOp->getOperand(1));
          ValExpr =
              exprForInContext(VOp->getOperand(0), RW, true) + " " +
              Sv.str() + " " +
              exprForInContext(VOp->getOperand(1), LW, true);
        }
      }
      // The set's RHS may be wider than the register (e.g. an fi
      // i8+i8 add yields i9 → i16 in MLIR's signless type system).
      // Truncate to the register width via an SV size cast so the
      // assignment is unambiguous.
      unsigned VW = 0;
      if (auto IT = mlir::dyn_cast<mlir::IntegerType>(Val.getType()))
        VW = IT.getWidth();
      // Constant fold: when the size-cast would wrap a plain
      // constant, re-render the literal at the register width
      // instead — `4'(8'sd0)` becomes `4'sd0`, identical hardware
      // with less visual noise. Skip when ValExpr was already
      // populated by the inline-arith path above.
      if (ValExpr.empty() && VW > P.Width) {
        if (auto Lit = tryReemitConstAtWidth(Val, P.Width, P.Signed)) {
          indent(Indent);
          OS << P.Name << "_next = " << *Lit << ";\n";
          return;
        }
      }
      if (ValExpr.empty()) ValExpr = exprFor(Val);

      indent(Indent);
      OS << P.Name << "_next = ";
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
    // Phase 5.6 closure: short-circuit boolean operators on
    // i1 operands lower as plain SV `||` / `&&`. Used by the
    // canonical `(x > hi) || (x < lo)` overflow-check idiom.
    else if (OpName == "matlab.short_or") Sv = "||";
    else if (OpName == "matlab.short_and") Sv = "&&";
    if (!Sv.empty()) {
      if (Op.getNumOperands() != 2 || Op.getNumResults() != 1) {
        fail(("unsupported arity on " + OpName + " in SV emitter").str());
        return;
      }
      // Phase 5.6.3: when the result is single-use and the
      // consumer is in the same block, the value renders inline at
      // the use site via exprFor → renderInlineExpr.
      if (isInlineable(Op.getResult(0))) return;
      indent(Indent);
      unsigned LW = renderedWidthOf(Op.getOperand(0));
      unsigned RW = renderedWidthOf(Op.getOperand(1));
      std::string LExpr =
          exprForInContext(Op.getOperand(0), RW, true);
      std::string RExpr =
          exprForInContext(Op.getOperand(1), LW, true);
      // Width-extend operands to the result type when one or both
      // are rendered narrower than the result. `c1 = int3 - comb1_d`
      // where int3, comb1_d are 22-bit registers but c1 is 32-bit:
      // wrap both operands with `32'($signed(...))` so Verilator's
      // self-determined-width rule doesn't flag WIDTHEXPAND.
      // Skip for comparisons (result is 1-bit) and short-circuit
      // booleans (operands are already i1).
      bool ResIsArith =
          (OpName == "matlab.add" || OpName == "matlab.sub" ||
           OpName == "matlab.matmul" || OpName == "matlab.emul");
      if (ResIsArith) {
        unsigned ResW = widthOf(Op.getResult(0).getType());
        auto wrap = [&](std::string &S, unsigned W) {
          if (W == 0 || W >= ResW) return;
          std::ostringstream Wr;
          Wr << ResW << "'($signed(" << S << "))";
          S = Wr.str();
        };
        wrap(LExpr, LW);
        wrap(RExpr, RW);
      }
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
  // `arith.fptosi` / `arith.fptoui` show up only as the typed
  // re-cast LowerScalarSlots inserts for a `slot_iN = persist_get`
  // where the runtime ABI returned f64. exprFor's fptosi branch
  // unwraps the cast to the underlying typed register signal.
  if (isa<arith::FPToSIOp, arith::FPToUIOp>(Op)) {
    if (isInlineable(Op.getResult(0))) return;
    indent(Indent);
    OS << name(Op.getResult(0)) << " = " << exprFor(Op.getResult(0))
       << ";\n";
    return;
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

void Emitter::emitLeadingCommentsBefore(mlir::Location Loc, int Indent) {
  if (!SM) return;
  auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(Loc);
  if (!FL) return;
  uint32_t Line = FL.getLine();
  if (Line == 0) return;
  llvm::StringRef File = FL.getFilename().getValue();
  // Scope the scan to the current function's body range. This
  // drops file-header / script-driver comments that aren't part
  // of the function being emitted.
  if (CommentFile.empty() || File != CommentFile) return;
  matlab::FileID FID = SM->findFileByName(File.str());
  if (FID == 0) return;
  uint32_t &Last = LastEmittedLine[File];
  uint32_t &Tail = LastTailEmittedLine[File];
  // Helper: extract `% ...` text from one source line. Returns
  // empty string when the line has no comment-text we want to
  // emit (no `%` at all, comment-only line that's a `#codegen` /
  // `hdl:` / `--- file ` marker, or empty body after stripping).
  // `LeadingOnly` toggles between "leading-only" comments (full-
  // line `%`-prefixed, used for the standalone-comment case) and
  // "trailing-only" comments (`%` after non-WS code on a line
  // that's otherwise a statement).
  auto extractCommentText = [&](uint32_t L, bool TrailingOnly) -> std::string {
    if (L < CommentMinLine || L > CommentMaxLine) return {};
    auto Txt = SM->getLineText(FID, L);
    llvm::StringRef LR(Txt.data(), Txt.size());
    // Find the first `%` not in a quoted string.
    bool InSingle = false, InDouble = false;
    size_t Pos = llvm::StringRef::npos;
    for (size_t I = 0; I < LR.size(); ++I) {
      char C = LR[I];
      if (!InDouble && C == '\'' && !InSingle) {
        InSingle = true; continue;
      }
      if (InSingle && C == '\'') { InSingle = false; continue; }
      if (!InSingle && C == '"' && !InDouble) { InDouble = true; continue; }
      if (InDouble && C == '"') { InDouble = false; continue; }
      if (InSingle || InDouble) continue;
      if (C == '%') { Pos = I; break; }
    }
    if (Pos == llvm::StringRef::npos) return {};
    // Determine HasCode: non-WS before the `%`.
    bool HasCode = false;
    for (size_t I = 0; I < Pos; ++I)
      if (!std::isspace((unsigned char)LR[I])) { HasCode = true; break; }
    if (TrailingOnly && !HasCode) return {};
    if (!TrailingOnly && HasCode) return {};
    llvm::StringRef Tx = LR.substr(Pos + 1);
    while (!Tx.empty() && std::isspace((unsigned char)Tx.front()))
      Tx = Tx.drop_front();
    while (!Tx.empty() && std::isspace((unsigned char)Tx.back()))
      Tx = Tx.drop_back();
    if (Tx.empty()) return {};
    if (Tx == "#codegen") return {};
    {
      llvm::StringRef T = Tx;
      if (T.starts_with("hdl:")) return {};
      if (T.starts_with("--- file ")) return {};
    }
    return Tx.str();
  };
  // Phase 5.6.4: emit ONLY comments that immediately precede this
  // op's source line. "Immediately precede" means: every line
  // between the comment and the op's line is itself a blank line,
  // a comment, or a pragma. This avoids dumping comments for
  // source lines that got folded away by Stage F or other
  // optimizations, which previously caused all leading comments
  // to bunch up at the top of `always_comb`.
  //
  // Skipped (too-distant) comments still advance Last/Tail so a
  // subsequent op never re-considers them. A comment that's
  // semantically orphaned (its target source line was folded) is
  // simply dropped — preferable to mis-attaching it to an
  // unrelated downstream op.
  auto isBlankOrCommentOrPragma = [&](uint32_t L) -> bool {
    auto Txt = SM->getLineText(FID, L);
    llvm::StringRef LR(Txt.data(), Txt.size());
    bool HasCode = false;
    bool InSingle = false, InDouble = false;
    for (size_t I = 0; I < LR.size(); ++I) {
      char C = LR[I];
      if (!InDouble && C == '\'' && !InSingle) {
        InSingle = true; continue;
      }
      if (InSingle && C == '\'') { InSingle = false; continue; }
      if (!InSingle && C == '"' && !InDouble) { InDouble = true; continue; }
      if (InDouble && C == '"') { InDouble = false; continue; }
      if (InSingle || InDouble) continue;
      if (C == '%') break; // rest is a comment
      if (!std::isspace((unsigned char)C)) { HasCode = true; break; }
    }
    return !HasCode;
  };
  uint32_t StartLine = std::max(Tail + 1, CommentMinLine);
  uint32_t EndLine = std::min(Line, CommentMaxLine);
  for (uint32_t L = StartLine; L <= EndLine; ++L) {
    if (L > Line) break;
    std::string Text = extractCommentText(L, /*TrailingOnly=*/false);
    if (Text.empty())
      Text = extractCommentText(L, /*TrailingOnly=*/true);
    if (Text.empty()) continue;
    // Check whether every line strictly between L and Line is
    // blank/comment/pragma. If so this comment is attached to the
    // current op; otherwise it belongs to a folded-away predecessor
    // and we drop it.
    bool Adjacent = true;
    for (uint32_t M = L + 1; M < Line; ++M) {
      if (!isBlankOrCommentOrPragma(M)) { Adjacent = false; break; }
    }
    if (!Adjacent) continue;
    indent(Indent);
    OS << "// " << Text << "\n";
  }
  Tail = std::max(Tail, EndLine);
  Last = std::max(Last, Line);
}

// Heuristic: ops that emit no visible SV. Comment-attachment skips
// these so their associated source comments fall through to the
// next visible op instead of bunching up against an invisible
// constant or yield.
static bool isInvisibleEmitOp(mlir::Operation &Op) {
  if (mlir::isa<mlir::scf::YieldOp>(&Op)) return true;
  if (mlir::isa<mlir::arith::ConstantOp,
                mlir::LLVM::ConstantOp>(&Op)) return true;
  // Persistent-state runtime ABI calls — these get suppressed by
  // the SV emitter (state inferred to register signals); they
  // emit no visible statement.
  if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(&Op)) {
    auto Cl = C.getCallee();
    if (Cl && (*Cl == "matlab_persistent_isempty" ||
               *Cl == "matlab_persistent_get_ptr" ||
               *Cl == "matlab_persistent_set_ptr" ||
               *Cl == "matlab_global_get_f64")) {
      return true;
    }
  }
  // matlab.call_builtin → matlab_global_set_f64 IS the persistent
  // register write which renders as `<reg>_next = <val>;` —
  // visible. Other builtins (subscript_store, etc.) also render.
  // Don't blanket-suppress them.
  // matlab.alloc / llvm.alloca are also invisible to comment-
  // attachment (they emit their declarations in the prelude).
  if (mlir::isa<mlir::LLVM::AllocaOp>(&Op)) return true;
  if (Op.getName().getStringRef() == "matlab.alloc") return true;
  // Inline-only arith ops never emit a statement of their own —
  // they're folded into the use site by exprFor.
  if (mlir::isa<mlir::arith::CmpIOp, mlir::arith::CmpFOp,
                mlir::arith::AddIOp, mlir::arith::SubIOp,
                mlir::arith::MulIOp, mlir::arith::AndIOp,
                mlir::arith::OrIOp, mlir::arith::XOrIOp,
                mlir::arith::ShLIOp, mlir::arith::ShRSIOp,
                mlir::arith::ShRUIOp, mlir::arith::SelectOp,
                mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                mlir::arith::TruncIOp,
                mlir::arith::FPToSIOp, mlir::arith::FPToUIOp>(&Op)) {
    if (Op.getNumResults() == 1 && !Op.getResult(0).use_empty()) {
      // If every user of this result is in the same block, it'll
      // be inlined at use sites and this op emits no statement.
      bool AllSameBlock = true;
      for (mlir::Operation *U : Op.getResult(0).getUsers()) {
        if (U->getBlock() != Op.getBlock()) { AllSameBlock = false; break; }
      }
      if (AllSameBlock) return true;
    }
  }
  // matlab.eq / etc. are also inline.
  llvm::StringRef N = Op.getName().getStringRef();
  if (N == "matlab.eq" || N == "matlab.ne" || N == "matlab.lt" ||
      N == "matlab.le" || N == "matlab.gt" || N == "matlab.ge")
    return true;
  return false;
}

void Emitter::emitBlock(mlir::Block &B, int Indent) {
  for (auto &Op : B.getOperations()) {
    if (Failed) return;
    // Skip suppressed (HWStateInfer / FSM-recognized) ops too —
    // they emit no visible RTL; their comments should attach to
    // the next visible op instead.
    if (!isInvisibleEmitOp(Op) && !Suppress.contains(&Op))
      emitLeadingCommentsBefore(Op.getLoc(), Indent);
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
  // Latch guard: pre-assign always_comb-driven signals at the top so
  // no code path leaves them unassigned. This is the canonical SV
  // idiom for combinational temps and is recognized by every synth
  // tool as a default-assignment pattern (no latch inferred). Without
  // it, Verilator (correctly) flags any signal that's conditionally
  // written in only some branches.
  //
  // Optimization: a `signal = '0;` prelude entry is *dead* when the
  // signal is also assigned at the top level of the always_comb body
  // (i.e. unconditionally, before any read can observe the prelude
  // value). Detect those cases up front and suppress the redundant
  // line — the body's top-level write provides the same latch-safety
  // guarantee with less visual noise.
  llvm::DenseSet<mlir::Value> UncondVals;
  llvm::DenseSet<unsigned> UncondOutputs;
  // Outputs whose alloca is slot-merged into the port skip the
  // explicit `port = expr` write in emitReturn; the body's
  // top-level stores to the merged signal are what actually drive
  // the port. Those stores get picked up via UncondVals below.
  llvm::DenseSet<unsigned> SlotMergedOuts;
  for (auto &Pair : SlotMergedToOut) SlotMergedOuts.insert(Pair.second);
  if (!F.getBody().empty()) {
    mlir::Block &Entry = F.getBody().front();
    for (mlir::Operation &Op : Entry) {
      if (mlir::isa<mlir::func::ReturnOp>(Op)) {
        for (unsigned I = 0, N = OutWriteNames.size(); I < N; ++I)
          if (!SlotMergedOuts.contains(I)) UncondOutputs.insert(I);
        continue;
      }
      // Persistent register set-sites at top level *are* unconditional
      // writes to `<reg>_next`, but suppressing the `_next = <reg>;`
      // prelude removes the only implicit read of `<reg>` for
      // registers whose get-sites are otherwise routed through stores
      // that the emitter doesn't materialize (a separate gap, see
      // examples/hdl/fir_asic_pipelined). Keep the prelude — its cost
      // is one line per register, and it doubles as a hold-by-default
      // fallback for any branch that skips `_next`.
      if (SetSiteToReg.contains(&Op)) continue;
      if (GetSiteToReg.contains(&Op)) continue;
      if (Suppress.contains(&Op)) continue;
      // A whole-slot store at top level overwrites the slot's
      // signal unconditionally. Stores via GEP touch only one
      // element of an array and don't prove full coverage, so
      // those keep the prelude `'{default: '0}`.
      if (auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
        mlir::Value Slot = St.getAddr();
        if (auto *SOp = Slot.getDefiningOp())
          if (mlir::isa<mlir::LLVM::GEPOp>(SOp)) continue;
        UncondVals.insert(Slot);
        // A top-level whole-slot store to a slot that's been
        // merged into an output port also drives that port
        // unconditionally — surface the index so the output's
        // prelude `'0` can be suppressed.
        if (auto *SOp = Slot.getDefiningOp()) {
          auto OIt = SlotMergedToOut.find(SOp);
          if (OIt != SlotMergedToOut.end())
            UncondOutputs.insert(OIt->second);
        }
        continue;
      }
      // Address-producing ops (alloca creates an uninitialized
      // slot; GEP computes an offset) carry no data write, so
      // their results don't satisfy the latch-guard guarantee.
      if (mlir::isa<mlir::LLVM::AllocaOp, mlir::LLVM::GEPOp>(Op))
        continue;
      for (mlir::Value V : Op.getResults()) UncondVals.insert(V);
    }
  }
  for (mlir::Value V : PreludeDecls) {
    if (UncondVals.contains(V)) continue;
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
  // (the pre-pipeline signal) instead of the actual port. Suppress
  // when the return op writes the output unconditionally.
  for (unsigned I = 0; I < OutWriteNames.size(); ++I) {
    if (UncondOutputs.contains(I)) continue;
    indent(2);
    OS << OutWriteNames[I] << " = '0;\n";
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
  LastEmittedLine.clear();
  LastTailEmittedLine.clear();
  SlotMergedToOut.clear();
  // Phase 5.6.3: detect allocas whose `matlab.name` matches an
  // output port's `matlab.name` AND whose only loads feed the
  // function's func.return. For each match, record the merge so
  // (a) the alloca's signal name aliases the port, (b) the
  // prelude skips its declaration, (c) the return suppresses the
  // self-assign `port = port;`.
  {
    auto FT = F.getFunctionType();
    llvm::StringMap<unsigned> OutByName;
    for (unsigned I = 0; I < FT.getNumResults(); ++I) {
      if (auto S = F.getResultAttrOfType<mlir::StringAttr>(I, "matlab.name"))
        OutByName[S.getValue()] = I;
    }
    if (!OutByName.empty()) {
      F.walk([&](mlir::LLVM::AllocaOp A) {
        auto NA = A->getAttrOfType<mlir::StringAttr>("matlab.name");
        if (!NA) return;
        // The slot's `name` attr may have come from the slot's
        // alloc (matlab.alloc → llvm.alloca lowering). Look up
        // matching output result.
        auto It = OutByName.find(NA.getValue());
        if (It == OutByName.end()) return;
        // Validate: every user of the alloca must be a store-to
        // or load-of the slot — no aliased pointer escapes (GEP,
        // bitcast, function call, etc.) that could observe or
        // mutate the slot through a different path. We require
        // at least one final load that feeds func.return so the
        // emitter knows which read drives the port. Intermediate
        // loads (datapath reads inside the body — alu_16bit's
        // overflow check reads `data_out` after assigning it) are
        // OK: SV `always_comb` blocking semantics make a later
        // `port = expr;` followed by `if (port < 0) ...` read the
        // just-written value, identical to the slot semantics.
        bool Ok = true;
        bool SawReturnLoad = false;
        for (auto *U : A->getUsers()) {
          if (mlir::isa<mlir::LLVM::StoreOp>(U)) continue;
          if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(U)) {
            for (auto *LU : L->getUsers()) {
              if (mlir::isa<mlir::func::ReturnOp>(LU))
                SawReturnLoad = true;
            }
            continue;
          }
          // Anything else (a non-load, non-store user) bails.
          Ok = false;
          break;
        }
        if (!Ok || !SawReturnLoad) return;
        SlotMergedToOut[A.getOperation()] = It->second;
      });
    }
  }
  // Phase 5.6.2b: discover the function's body line range so the
  // comment-forwarding scan can drop anything outside it (script-
  // header prose, leftover driver comments, etc.). Seed from the
  // FuncOp's own location (the `function` keyword line) so a
  // function-leading comment immediately above the signature gets
  // included while file-level prose above does not.
  CommentFile.clear();
  CommentMinLine = ~0u;
  CommentMaxLine = 0;
  if (auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(F.getLoc())) {
    CommentFile = FL.getFilename().str();
    CommentMinLine = FL.getLine();
    CommentMaxLine = FL.getLine();
  }
  F.walk([&](mlir::Operation *Op) {
    if (auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(Op->getLoc())) {
      uint32_t L = FL.getLine();
      if (L == 0) return;
      if (CommentFile.empty()) CommentFile = FL.getFilename().str();
      else if (FL.getFilename().getValue() != CommentFile) return;
      CommentMinLine = std::min(CommentMinLine, L);
      CommentMaxLine = std::max(CommentMaxLine, L);
    }
  });

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
    // Suppress the isempty call itself. Accept both `llvm.call`
    // and `matlab.call_builtin` shapes — Stage F's synthetic
    // per-element persistents emit the latter form, scalar
    // persistents lower to the former by pipeline time.
    F.walk([&](mlir::Operation *Op) {
      if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
        auto C = Call.getCallee();
        if (C && *C == "matlab_persistent_isempty") Suppress.insert(Op);
        return;
      }
      if (Op->getName().getStringRef() == "matlab.call_builtin") {
        auto S = Op->getAttrOfType<mlir::StringAttr>("callee");
        if (S && S.getValue() == "matlab_persistent_isempty")
          Suppress.insert(Op);
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

  // D1 — emit lint hints (stderr) before the module body. Runs
  // here so the user sees the warning paired with the port list
  // they're about to inspect.
  emitPortHints(F);

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
  // B1 — gather saturation-helper specs from `arith.select`
  // attrs set by LowerFiSaturate, then emit one `function
  // automatic` per unique (signed, sat-width, input-width)
  // tuple. Placed between declarations and always_comb so the
  // body's calls to `sat_<...>(x)` resolve against an in-scope
  // declaration without a forward reference.
  collectSatHelpers(F);
  if (!SatHelpers.empty()) {
    OS << "\n";
    emitSatHelpers();
  }
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

std::string Emitter::satHelperName(bool Signed, unsigned SatW,
                                   unsigned InputW) {
  std::ostringstream S;
  S << "sat_" << (Signed ? "s" : "u") << InputW << "_b" << SatW;
  return S.str();
}

mlir::Value Emitter::satHelperInput(mlir::arith::SelectOp Sel,
                                    bool Signed) {
  // Mirror the operand structure produced by `LowerFiSaturate`:
  //   Signed → outer = select(GtMax, MaxV, Inner)
  //            inner = select(LtMin, MinV, Val)
  //   Unsigned → outer = select(GtMax, MaxV, Val)
  if (Signed) {
    if (auto Inner =
            Sel.getFalseValue().getDefiningOp<mlir::arith::SelectOp>())
      return Inner.getFalseValue();
  }
  return Sel.getFalseValue();
}

void Emitter::collectSatHelpers(mlir::func::FuncOp F) {
  SatHelpers.clear();
  std::set<SatHelperKey> Seen;
  F.walk([&](mlir::arith::SelectOp Sel) {
    auto W = Sel->getAttrOfType<mlir::IntegerAttr>("matlab.fi_sat_w");
    if (!W) return;
    auto S = Sel->getAttrOfType<mlir::BoolAttr>("matlab.fi_sat_signed");
    if (!S) return;
    SatHelperKey K{S.getValue(), (unsigned)W.getInt(),
                   widthOf(Sel.getResult().getType())};
    if (K.SatW == 0 || K.InputW == 0) return;
    if (Seen.insert(K).second) SatHelpers.push_back(K);
  });
  std::sort(SatHelpers.begin(), SatHelpers.end());
}

void Emitter::emitSatHelpers() {
  if (SatHelpers.empty()) return;
  // Build each helper's signature + body as raw SV text. The
  // helper takes the input at its full width; the saturation
  // clamps the value to fit `K.SatW` bits but the result type is
  // the same as the input (caller would otherwise need to cast).
  for (const auto &K : SatHelpers) {
    std::string Ty = "logic";
    if (K.Signed) Ty += " signed";
    Ty += " [" + std::to_string(K.InputW - 1) + ":0]";
    std::string Name = satHelperName(K.Signed, K.SatW, K.InputW);
    auto SLit = [&](int64_t V) {
      std::ostringstream O;
      if (V < 0) {
        uint64_t Abs =
            (V == std::numeric_limits<int64_t>::min())
                ? (uint64_t(1) << 63)
                : (uint64_t)(-V);
        O << "-" << K.InputW << "'sd" << Abs;
      } else {
        O << K.InputW << "'sd" << V;
      }
      return O.str();
    };
    auto ULit = [&](uint64_t V) {
      std::ostringstream O;
      O << K.InputW << "'d" << V;
      return O.str();
    };
    // Parameter name `arg` chosen to avoid VARHIDDEN collisions
    // with module ports — modules commonly name a 16-bit input
    // port `x`, which would otherwise shadow the helper's input.
    indent(1);
    OS << "function automatic " << Ty << " " << Name
       << "(input " << Ty << " arg);\n";
    if (K.Signed) {
      int64_t Max = (K.SatW < 64) ? ((int64_t)1 << (K.SatW - 1)) - 1
                                  : std::numeric_limits<int64_t>::max();
      int64_t Min = (K.SatW < 64) ? -((int64_t)1 << (K.SatW - 1))
                                  : std::numeric_limits<int64_t>::min();
      indent(2);
      OS << "if (arg > " << SLit(Max) << ") " << Name << " = "
         << SLit(Max) << ";\n";
      indent(2);
      OS << "else if (arg < " << SLit(Min) << ") " << Name << " = "
         << SLit(Min) << ";\n";
      indent(2);
      OS << "else " << Name << " = arg;\n";
    } else {
      uint64_t Max = (K.SatW < 64) ? ((uint64_t(1) << K.SatW) - 1)
                                   : ~uint64_t(0);
      indent(2);
      OS << "if (arg > " << ULit(Max) << ") " << Name << " = "
         << ULit(Max) << ";\n";
      indent(2);
      OS << "else " << Name << " = arg;\n";
    }
    indent(1);
    OS << "endfunction\n\n";
  }
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
    if (!IsFSM) {
      // The reset value typically renders at the storage class
      // width (e.g. `8'sd0` for an i8-stored register), but the
      // register itself is declared at the user-declared width
      // (`P.Width`, from the fi spec). Constant reset values fold
      // to a literal at the register width so the always_ff reads
      // `count_reg <= 4'sd0;` instead of `count_reg <= 4'(8'sd0);`.
      // Non-constants fall back to the explicit `<W>'(<expr>)` SV
      // size-cast idiom so Verilator doesn't flag WIDTHTRUNC.
      unsigned RVW = widthOf(P.ResetValue.getType());
      if (RVW > P.Width && P.Width > 0) {
        if (auto Lit =
                tryReemitConstAtWidth(P.ResetValue, P.Width, P.Signed)) {
          ResetExpr = *Lit;
        } else {
          ResetExpr = exprFor(P.ResetValue);
          std::ostringstream Cast;
          Cast << P.Width << "'(" << ResetExpr << ")";
          ResetExpr = Cast.str();
        }
      } else {
        ResetExpr = exprFor(P.ResetValue);
      }
    }
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
  const char *RC = "async-low (assert async, deassert sync)";
  if (Reset == HWResetKind::SyncHigh) RC = "sync-high";
  else if (Reset == HWResetKind::SyncLow) RC = "sync-low";
  OS << "// Reset convention: " << RC << ".\n\n";
}

bool Emitter::isBooleanOnlyPort(mlir::Value V) {
  // Walk transitively through stores/loads of a spill slot (the
  // canonical entry-block shape: each port arg is stored to a
  // matlab.alloc / llvm.alloca, every datapath read goes through
  // a load). A "boolean predicate" use is `arith.cmpi eq/ne` or
  // `matlab.eq/matlab.ne` against a constant on the other side.
  // Anything else (an arith.add, a select, a return, a store to a
  // non-spill slot, ...) makes the port a real datapath signal —
  // give up and report `false`.
  llvm::SmallPtrSet<mlir::Value, 8> Visited;
  std::function<bool(mlir::Value)> Walk = [&](mlir::Value Cur) -> bool {
    if (!Visited.insert(Cur).second) return true;
    if (Cur.use_empty()) return false; // unused → don't warn
    bool AnyClassified = false;
    for (mlir::OpOperand &Use : Cur.getUses()) {
      mlir::Operation *U = Use.getOwner();
      llvm::StringRef N = U->getName().getStringRef();
      // Boolean predicate cmpi eq/ne against a constant.
      bool IsCmp = false;
      if (auto C = mlir::dyn_cast<mlir::arith::CmpIOp>(U)) {
        auto P = C.getPredicate();
        IsCmp = (P == mlir::arith::CmpIPredicate::eq ||
                 P == mlir::arith::CmpIPredicate::ne);
      } else if (N == "matlab.eq" || N == "matlab.ne") {
        IsCmp = true;
      }
      if (IsCmp) {
        // Only `cmp <port>, 0` or `cmp <port>, 1` qualify as
        // boolean predicates — anything else (a multi-way `case
        // (sel)` discriminator like `sel == 2`, `sel == 5`, ...)
        // is enum-shaped, not boolean. Reject so an N-way mux
        // selector doesn't trip the bool-port hint.
        bool BoolPeer = false;
        for (mlir::Value O : U->getOperands()) {
          if (O == Cur) continue;
          auto *Op = O.getDefiningOp();
          if (!Op) continue;
          int64_t K = 0;
          bool Got = false;
          // arith.constant + llvm.mlir.constant carry their value
          // via getValue(); both can be IntegerAttr (i*) or
          // FloatAttr (f64 — matlab.eq's RHS, post-lowering, is
          // f64-typed via the runtime ABI).
          mlir::Attribute Val;
          if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op))
            Val = C.getValue();
          else if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Op))
            Val = C.getValue();
          else
            Val = Op->getAttr("value");
          if (auto IA = mlir::dyn_cast_or_null<mlir::IntegerAttr>(Val)) {
            K = IA.getInt();
            Got = true;
          } else if (auto FA =
                         mlir::dyn_cast_or_null<mlir::FloatAttr>(Val)) {
            double D = FA.getValueAsDouble();
            if (D == 0.0 || D == 1.0) {
              K = (int64_t)D;
              Got = true;
            }
          }
          if (Got && (K == 0 || K == 1)) {
            BoolPeer = true;
            break;
          }
        }
        if (!BoolPeer) return false;
        AnyClassified = true;
        continue;
      }
      // Spill: matlab.store / llvm.store — Cur must be the value
      // (operand 0), not the slot. Walk every load of the slot.
      if ((N == "matlab.store" ||
           mlir::isa<mlir::LLVM::StoreOp>(U)) &&
          U->getNumOperands() >= 2 &&
          U->getOperand(0) == Cur) {
        mlir::Value Slot = U->getOperand(1);
        for (mlir::Operation *SU : Slot.getUsers()) {
          llvm::StringRef SN = SU->getName().getStringRef();
          if (SN == "matlab.load" ||
              mlir::isa<mlir::LLVM::LoadOp>(SU)) {
            for (mlir::Value R : SU->getResults())
              if (!Walk(R)) return false;
          }
          // Stores back into the slot don't count as new uses.
        }
        AnyClassified = true;
        continue;
      }
      return false;
    }
    return AnyClassified;
  };
  return Walk(V);
}

void Emitter::emitPortHints(mlir::func::FuncOp F) {
  for (unsigned I = 0; I < F.getNumArguments(); ++I) {
    mlir::BlockArgument Arg = F.getArgument(I);
    auto IT = mlir::dyn_cast<mlir::IntegerType>(Arg.getType());
    if (!IT || IT.getWidth() <= 1) continue;
    if (!isBooleanOnlyPort(Arg)) continue;
    std::string PortName;
    if (auto N =
            F.getArgAttrOfType<mlir::StringAttr>(I, "matlab.name"))
      PortName = N.getValue().str();
    if (PortName.empty()) PortName = "arg" + std::to_string(I);
    llvm::errs() << F.getLoc() << ": warning: input port '"
                 << PortName << "' is " << IT.getWidth()
                 << " bits wide but only used as a boolean — "
                 << "consider declaring it as `% hdl: port("
                 << PortName << ", bool)` (or `fi unsigned 1 0`) "
                 << "so the SV port renders as 1-bit `logic "
                 << PortName << "` instead of `logic ["
                 << (IT.getWidth() - 1) << ":0] " << PortName
                 << "`\n";
  }
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
  /* Phase 6: symbolic computation is fundamentally unsynthesizable —
   * matlab_sym_* expects an MPFR/GMP runtime that has no FPGA mapping.
   * Diagnose at the start with a clear hardware-context error so the
   * user gets the right hint (drop the sym, use fi). */
  bool HasSym = false;
  M.walk([&](mlir::Operation *Op) {
    if (HasSym) return;
    if (auto Cal = Op->getAttrOfType<mlir::StringAttr>("callee"))
      if (Cal.getValue().starts_with("matlab_sym_")) { HasSym = true; return; }
    if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op))
      if (F.getName().starts_with("matlab_sym_")) HasSym = true;
  });
  if (HasSym) {
    llvm::errs() << "error: Symbolic Math Toolbox operations "
                 << "(matlab_sym_*) are not synthesizable — use fi for "
                 << "fixed-point hardware models, or remove the "
                 << "symbolic operations\n";
    return {};
  }
  std::ostringstream OS;
  Emitter E(OS, SM, Reset, FSMEnc);
  if (!E.run(M)) return std::string();
  return OS.str();
}

} // namespace mlirgen
} // namespace matlab
