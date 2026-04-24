// Emits C (or C++) source from an MLIR ModuleOp whose ops have already been
// lowered to a small, closed set: func / arith / scf / cf / llvm.call /
// llvm.alloca / llvm.load / llvm.store / llvm.mlir.global / llvm.mlir.addressof
// plus outlined llvm.func bodies (parfor / anonymous functions).
//
// The emitter walks the module, assigns every mlir::Value a stable C
// identifier, and prints statements as it visits ops in source order.
// The output is intended to be linked against runtime/matlab_runtime.c.

#include "matlab/MLIR/Passes/Passes.h"
#include "matlab/Basic/SourceManager.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <iostream>
#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

/// Metadata for an scf.while op that matched the canonical for-loop shape
/// produced by LowerSeqLoops. Populated by Emitter::scanForLoopPatterns
/// before emission so the scf.while handler can print a C-style
/// `for (T iv = init; iv OP end; iv += step)` instead of a while loop.
struct ForLoopInfo {
  // Loop bounds (SSA values — stringified with stmtExpr at emit time).
  mlir::Value Init;
  mlir::Value End;
  mlir::Value Step;
  // True when the surviving condition is `arith.cmpf OGE iv, end` (i.e.
  // the loop counts down). Drives `>=` vs `<=` in the for-head.
  bool IsDecreasing = false;
  // Ops absorbed into the for-head / for-increment; emitOp short-circuits
  // when it sees them in SuppressedOps.
  mlir::Operation *AddOp = nullptr;     // arith.addf iv, step
  mlir::Operation *YieldOp = nullptr;   // scf.yield %add
  // When non-null, the first op of the after-block is `llvm.store iv, slot`
  // where `slot` is an alloca carrying the MATLAB loop-var name. We reuse
  // that slot's identifier as the C induction variable.
  mlir::Operation *BindStore = nullptr;
  mlir::Operation *SlotAlloca = nullptr;
  // C identifier for the induction variable. Either the MATLAB loop-var
  // name lifted from SlotAlloca, or a fresh `vN` when no fusion applies.
  std::string IvName;
  // True when the slot's entire use-set lives inside this for-loop — safe
  // to elide the slot's prolog declaration and scope the IV to the for.
  bool FuseSlot = false;
};

/// Per-module emission state: symbol table for SSA values and a single
/// output stream. Kept in a class because the recursive descent over
/// regions needs to share state across helpers.
class Emitter {
public:
  Emitter(std::ostream &OS, bool Cpp, bool NoLine, bool Doxygen,
          bool CppAuto, const matlab::SourceManager *SM)
      : OS(OS), Cpp(Cpp), NoLine(NoLine), Doxygen(Doxygen),
        CppAuto(Cpp && CppAuto), SM(SM) {}

  bool run(mlir::ModuleOp M);

private:
  // --- Naming / types -----------------------------------------------------
  std::string name(mlir::Value V);
  std::string freshName(const char *Prefix = "v");
  std::string uniqueName(llvm::StringRef Hint);
  std::string sanitizeIdent(llvm::StringRef In);
  std::string cTypeOf(mlir::Type T);
  std::string cTypeOfValue(mlir::Value V) { return cTypeOf(V.getType()); }

  // --- Region / block printing -------------------------------------------
  void emitRegion(mlir::Region &R, int Indent);
  void emitBlock(mlir::Block &B, int Indent);
  void emitOp(mlir::Operation &Op, int Indent);

  // --- Top-level ---------------------------------------------------------
  void emitGlobal(mlir::LLVM::GlobalOp G);
  void emitFuncFunc(mlir::func::FuncOp F);
  void emitLLVMFunc(mlir::LLVM::LLVMFuncOp F);
  void emitProlog();
  // Walk the module once to populate HasParfor / NeedsStdio / NeedsIostream
  // so the prolog can emit the right `#include`s and the body emitter can
  // decide whether to substitute matlab_disp_* calls for stdio / iostream
  // equivalents.
  void precomputeModuleProperties(mlir::ModuleOp M);
  // Try to emit `llvm.call @matlab_disp_*` as a direct puts/printf/cout
  // call when the module has no parfor (the runtime's mutex is dead
  // weight for single-threaded output) and, for disp_str, the string
  // argument is a compile-time literal. Returns true if the call was
  // handled — caller skips its own emission. Otherwise false.
  bool tryEmitIOSubstitution(mlir::LLVM::CallOp Call, int Indent);
  // Escape a raw byte sequence for embedding in a `"..."` C/C++ literal.
  // Only handles ASCII-safe input; callers should fall back to the
  // runtime call when non-printable bytes are present.
  static bool writeQuotedStringLiteral(std::ostream &OS,
                                       llvm::StringRef Raw);

  // --- Helpers -----------------------------------------------------------
  void indent(int N) { for (int i = 0; i < N; ++i) OS << "  "; }
  std::string constStr(mlir::LLVM::GlobalOp G);
  void fail(llvm::StringRef Msg) {
    if (!Failed)
      std::cerr << "error: emit-c: " << Msg.str() << "\n";
    Failed = true;
  }
  void emitLineDirective(mlir::Location L, int Indent);
  // Scan leading `%` comments in the source that precede `Line` (exclusive)
  // down to `AfterLine` (exclusive). Emit them as `// ...` lines at the
  // given indent. No-op when SM is null or the named file isn't in it.
  // Returns true if any output was emitted (comments or preserved blanks).
  // When `FunctionHeader` is true AND the Doxygen flag is set, the block
  // is rendered as `/**\n * line\n * line\n */` instead of `//` lines.
  bool emitLeadingComments(llvm::StringRef FullPath, int AfterLine,
                           int Line, int Indent, bool FunctionHeader = false);

  // --- Single-use inlining ----------------------------------------------
  // Return the C expression to use when referring to V: either the inline
  // expression built lazily from its producer, or its declared identifier
  // (for values the emitter already materialized as a C local).
  std::string exprFor(mlir::Value V);
  // exprFor, with one outermost layer of balanced parens stripped when
  // present. Use only at statement-level positions where the surrounding
  // syntax already delimits the expression (return value, rhs of `=`,
  // if/while condition, function arguments). Never call from inside
  // another composed expression — inner parens there enforce precedence.
  std::string stmtExpr(mlir::Value V);
  // Walk a region, marking producers that should be skipped during emission
  // and whose uses should inline the expression instead. Does NOT build
  // the strings — those are synthesised on demand by buildInlineExpr so
  // operand references resolve against names that are only chosen at
  // emission time (e.g. slot_p for llvm.alloca results).
  void computeInlines(mlir::Region &R);
  // Walk a region and tag scf.while ops that match the canonical for-loop
  // shape produced by LowerSeqLoops. Populates ForPatterns and
  // SuppressedOps so the scf.while emitter can print `for (...; ...; ...)`
  // and the absorbed ops (arith.addf, scf.yield, and — when fusing — the
  // iv-binding llvm.store and the slot's llvm.alloca) are skipped.
  void scanForLoopPatterns(mlir::Region &R);
  // Helper for scanForLoopPatterns: try to match one scf.while against the
  // for-loop shape. Writes Info on success.
  bool matchForPattern(mlir::scf::WhileOp W, ForLoopInfo &Info);
  // Walk a region and tag every break / continue flag slot, the false
  // stores that only exist to initialise or reset them, and the scf.if
  // ops the emitter should collapse into `break;` / `continue;` one-
  // liners or into inline then-region emission. Populates the four maps
  // above plus SuppressedOps.
  void scanBreakContinueFlags(mlir::Region &R);
  // Is V the expression `xori(load(flag_slot), const true)` — i.e. the
  // inverted break or continue flag value the loop condition / body
  // guard reads? Used both by the scf.while emitter (to strip such
  // operands out of the while condition) and by the scf.if scan (to
  // recognise the body guard).
  bool isFlagInversion(mlir::Value V);
  // Walk a conjunction expression and collect the operands that are NOT
  // break/continue flag inversions. Used to rewrite a loop condition
  // like `cond & !did_break` down to just `cond`.
  void gatherNonFlagConjuncts(mlir::Value V,
      llvm::SmallVectorImpl<mlir::Value> &Out);
  // Is Op's result safe to inline at its use?
  bool canInline(mlir::Operation &Op);
  // Build the inline expression for an inlineable Op, recursively
  // resolving operand references via exprFor (which re-enters this
  // function for any operand whose producer is also inlined).
  bool buildInlineExpr(mlir::Operation &Op, std::string &Expr);

  std::ostream &OS;
  bool Cpp;
  bool NoLine;
  bool Doxygen;  // wrap function-leading comments in /** ... */
  bool CppAuto;  // use `auto` for call-result locals in C++ mode
  const matlab::SourceManager *SM;  // nullable; comments scan disabled if null
  bool Failed = false;

  // Module-level properties, computed once up front so the prolog (header
  // includes, runtime externs) and the body emission agree. Populated by
  // precomputeModuleProperties before prolog/body emission.
  bool HasParfor = false;  // any call to matlab_parfor_dispatch?
  bool NeedsStdio = false; // will emit puts() / printf()
  bool NeedsIostream = false; // will emit std::cout
  // Runtime llvm.func declarations that still have at least one call site
  // surviving the IO substitutions. Used to prune dead externs from the
  // prolog so the emitted file doesn't declare functions it never calls.
  llvm::StringSet<> LiveRuntimeFuncs;
  // `llvm.mlir.global` symbols that are still referenced by a surviving
  // (non-substituted) callsite. Globals whose only users fed the
  // substituted `matlab_disp_str(literal)` path are dead after emission
  // and get pruned from the prolog.
  llvm::StringSet<> LiveGlobals;

  llvm::DenseMap<mlir::Value, std::string> Names;
  llvm::DenseMap<mlir::Operation *, std::string> GlobalStrs;  // global -> C name
  llvm::StringSet<> UsedNames;  // identifiers already claimed.
  // Per-function: SSA values whose producer is skipped; the cached
  // expression to substitute at use sites.
  llvm::DenseMap<mlir::Value, std::string> InlineExprs;
  llvm::DenseSet<mlir::Operation *> InlinedOps;
  // Allocas whose entire use set is plain load/store (no call / GEP /
  // cast consumer). For these, the pointer indirection is skipped and
  // stores/loads are emitted as direct reads/writes of the slot's
  // identifier: `y = 1;` rather than `*(double*)y_p = 1;`. Maps
  // alloca-op -> the C identifier of the backing slot.
  llvm::DenseMap<mlir::Operation *, std::string> DirectSlots;
  // Subset of DirectSlots whose declaration was postponed so it can be
  // merged with the first store into `T slot = val;`. An alloca joins
  // this set when its first same-block user is a StoreOp, and is erased
  // from it once that store has emitted the combined declaration.
  llvm::DenseSet<mlir::Operation *> DirectSlotDefer;
  // scf.while ops that matched the for-loop shape from LowerSeqLoops.
  // Emission consults this map to print a C-style for-head instead of a
  // while, and to route inline expression bindings through the chosen IV
  // name rather than a fresh `vN`.
  llvm::DenseMap<mlir::Operation *, ForLoopInfo> ForPatterns;
  // Ops that have been folded into an enclosing for-loop head/increment
  // (arith.addf, scf.yield) or elided by slot fusion (iv-binding
  // llvm.store). emitOp returns without printing when it encounters one
  // of these.
  llvm::DenseSet<mlir::Operation *> SuppressedOps;
  // Loop-var slot allocas whose `double i = 0;` prolog declaration should
  // be skipped — the for-loop that owns the slot declares the induction
  // variable in its own `for (...)` init clause, and no code outside that
  // loop reads the slot. The alloca handler still runs enough to register
  // a DirectSlots entry aliased to the chosen IV name so in-body loads
  // resolve correctly.
  llvm::DenseSet<mlir::Operation *> FusedForSlots;
  // Alloca op -> C identifier chosen for the fused IV. Shared across
  // multiple for-loops claiming the same slot so each `for (double i =
  // ...; ...)` agrees on the name.
  llvm::DenseMap<mlir::Operation *, std::string> FusedForSlotName;
  // Break / continue flag slot allocas (matlab.name = "__did_break" or
  // "__did_continue"). The frontend allocates these whenever a loop body
  // contains break/continue and threads the i1 state through stores,
  // loads, and loop-condition XORs. The emitter un-does that lowering:
  // skip the alloca, skip the false-store initialisations and continue
  // resets, strip flag-inversions out of loop conditions, un-wrap the
  // body guard, and re-emit flag stores as `break;` / `continue;`.
  llvm::DenseSet<mlir::Operation *> BreakFlagSlots;
  llvm::DenseSet<mlir::Operation *> ContinueFlagSlots;
  // scf.if ops whose entire body is a single flag-set followed by an
  // (elided) scf.yield; re-emit as a one-liner `if (cond) break;` (or
  // continue;). Value is the literal keyword to print.
  llvm::DenseMap<mlir::Operation *, const char *> FlagIfKind;
  // scf.if ops whose condition is purely a break/continue flag check
  // (`!did_break`, `!did_continue`, or their conjunction) guarding a
  // real body; emit the then-region inline at the parent scope so the
  // body doesn't sit under a now-meaningless `if (true) { ... }`.
  llvm::DenseSet<mlir::Operation *> InlinedIfs;
  int NextId = 0;

  // Most recent #line directive emitted — used to dedupe.
  std::string LastLineFile;
  int LastLineNum = -1;
  // True at the very start of a block's body (immediately after its
  // opening `{`). Forward-jump blank lines are suppressed until the
  // first real statement has been emitted — otherwise every block would
  // open with a gratuitous blank line below the brace.
  bool AtBlockStart = true;

  // Trailing same-line comment for the current line, pending emission.
  // Set when we move LastLineNum forward to a line that has `foo; % bar`
  // shape; flushed on the next line advance or block exit so `// bar`
  // lands right below the C output for that line.
  int PendingTrailingLine = -1;      // -1 = no pending
  int PendingTrailingIndent = 0;
  std::string PendingTrailingText;
  // Lines whose trailing comment has already been flushed in the current
  // function. Prevents double-emit when an op's location bounces back to
  // an earlier line (e.g. the synthesized `return 0;` in `main` reusing
  // the script's first-stmt location).
  llvm::DenseSet<int> TrailingEmittedLines;
  void flushPendingTrailing();
};

// ---------------------------------------------------------------------------
// Naming / types
// ---------------------------------------------------------------------------

std::string Emitter::freshName(const char *Prefix) {
  for (;;) {
    std::string S = Prefix;
    S += std::to_string(NextId++);
    if (UsedNames.insert(S).second) return S;
  }
}

// Make a C identifier out of a free-form MATLAB variable name. Allowed
// chars: [A-Za-z0-9_]. Illegal chars become '_'. If the first char is a
// digit, prepend an underscore. Empty / all-bad input falls back to "v".
std::string Emitter::sanitizeIdent(llvm::StringRef In) {
  std::string Out;
  Out.reserve(In.size() + 1);
  for (char C : In) {
    if ((C >= 'A' && C <= 'Z') || (C >= 'a' && C <= 'z') ||
        (C >= '0' && C <= '9') || C == '_') {
      Out += C;
    } else {
      Out += '_';
    }
  }
  if (Out.empty()) Out = "v";
  else if (Out[0] >= '0' && Out[0] <= '9') Out = "_" + Out;
  return Out;
}

// Give a locally unique C identifier derived from Hint. If Hint collides
// with a previously-used name, append _2 / _3 / ... until free.
std::string Emitter::uniqueName(llvm::StringRef Hint) {
  std::string Base = sanitizeIdent(Hint);
  if (UsedNames.insert(Base).second) return Base;
  for (int k = 2; ; ++k) {
    std::string Cand = Base + "_" + std::to_string(k);
    if (UsedNames.insert(Cand).second) return Cand;
  }
}

std::string Emitter::name(mlir::Value V) {
  auto It = Names.find(V);
  if (It != Names.end()) return It->second;
  std::string N = freshName();
  Names[V] = N;
  return N;
}

// Format an IntegerAttr as a C integer literal. i1 values are emitted as
// 0 / 1 (unsigned) so they work correctly when XOR'd against bool operands
// (IntegerAttr::getInt sign-extends i1 `true` to -1, which breaks
// `bool ^ -1` logical-NOT semantics when inlined as an expression).
static std::string formatIntAttr(mlir::IntegerAttr IA) {
  char Buf[64];
  auto T = mlir::dyn_cast<mlir::IntegerType>(IA.getType());
  if (T && T.getWidth() == 1) {
    snprintf(Buf, sizeof(Buf), "%u",
             (unsigned)(IA.getValue().getZExtValue() & 1u));
    return Buf;
  }
  snprintf(Buf, sizeof(Buf), "%lld", (long long)IA.getInt());
  return Buf;
}

// Format a FloatAttr as a C double literal. `%.17g` round-trips every
// finite double, but prints integer-valued numbers without a decimal
// (`42`, `-1`) which are C ints — dangerous in `printf("%g\n", ...)`
// varargs context and noisy in generated source. Append `.0` when the
// formatted text has no `.`, `e`, `E`, `n` (NaN/inf) so the type stays
// double without bloating common fractional values like `3.14`.
//
// Use the shortest precision that round-trips to the same double — the
// textbook algorithm (bump precision from 1 until strtod reproduces the
// source bits). `%.17g` is safe but ugly: `1e-12` becomes
// `9.9999999999999998e-13`, which is correct but unreadable.
static std::string formatFloatAttr(mlir::FloatAttr FA) {
  char Buf[64];
  double D = FA.getValueAsDouble();
  // Try every `%g` precision from 1..17, keep whichever round-trips AND
  // is shortest — `%.1g` of 10 is "1e+01" (roundtrips but verbose),
  // `%.2g` is "10" (roundtrips, shorter). Ties favour lower precision.
  std::string S;
  for (int P = 1; P <= 17; ++P) {
    snprintf(Buf, sizeof(Buf), "%.*g", P, D);
    double Back = 0.0;
    if (sscanf(Buf, "%lf", &Back) != 1 || Back != D) continue;
    std::string Cand = Buf;
    if (S.empty() || Cand.size() < S.size()) S = std::move(Cand);
  }
  if (S.empty()) {
    snprintf(Buf, sizeof(Buf), "%.17g", D);
    S = Buf;
  }
  bool HasDotOrExp = false;
  for (char C : S) {
    if (C == '.' || C == 'e' || C == 'E' || C == 'n') {
      HasDotOrExp = true;
      break;
    }
  }
  if (!HasDotOrExp) S += ".0";
  return S;
}

// Return the C expression to substitute when referring to V.
//  - If V was declared (Names has an entry), return that identifier.
//  - Else if V has an inline expression cached, return it.
//  - Else if V's producer was marked inlineable, build the expression
//    lazily NOW (this point comes after all prior non-inlined ops have
//    already been emitted, so operand names are stable).
//  - Else fall back to name() (auto-allocates a fresh v-id).
std::string Emitter::exprFor(mlir::Value V) {
  auto NI = Names.find(V);
  if (NI != Names.end()) return NI->second;
  auto II = InlineExprs.find(V);
  if (II != InlineExprs.end()) return II->second;
  if (mlir::Operation *Def = V.getDefiningOp()) {
    if (InlinedOps.count(Def)) {
      std::string Expr;
      if (buildInlineExpr(*Def, Expr)) {
        InlineExprs[V] = Expr;
        return Expr;
      }
    }
  }
  return name(V);
}

// Strip one outermost layer of balanced parens from `E` if present. Safe
// only at statement-level positions — inner parens carry precedence and
// cannot be removed without analysis.
static std::string dropOuterParens(std::string E) {
  if (E.size() < 2 || E.front() != '(' || E.back() != ')') return E;
  int Depth = 0;
  for (size_t i = 0; i < E.size(); ++i) {
    if (E[i] == '(') ++Depth;
    else if (E[i] == ')') {
      --Depth;
      // If the opening paren closes before the end, the first `(` is NOT
      // the partner of the final `)` — stripping would change meaning
      // (`(a) + (b)` → `a) + (b`). Bail.
      if (Depth == 0 && i + 1 < E.size()) return E;
    }
  }
  return E.substr(1, E.size() - 2);
}

std::string Emitter::stmtExpr(mlir::Value V) {
  return dropOuterParens(exprFor(V));
}

// Is Op's result safe to inline into its use?
bool Emitter::canInline(mlir::Operation &Op) {
  using namespace mlir;
  if (Op.getNumResults() != 1) return false;
  Value V = Op.getResult(0);

  // Constants / zero / addressof: always inline regardless of use count.
  // Pure, zero-cost, no operand ordering concerns.
  if (isa<LLVM::ConstantOp, arith::ConstantOp, LLVM::ZeroOp,
          LLVM::AddressOfOp>(Op))
    return true;

  // Everything else requires single-use AND a same-block user. The
  // same-block restriction is a conservative approximation of dominance
  // that works for our snapshot shape (scf regions, no cf.br).
  if (!V.hasOneUse()) return false;
  Operation *User = V.getUses().begin()->getOwner();
  if (User->getBlock() != Op.getBlock()) return false;

  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::AddIOp, arith::SubIOp, arith::MulIOp,
          arith::AndIOp, arith::OrIOp,  arith::XOrIOp,
          arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
          arith::SIToFPOp, arith::UIToFPOp,
          arith::FPToSIOp, arith::FPToUIOp,
          arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::TruncFOp, arith::ExtFOp,
          LLVM::GEPOp>(Op))
    return true;

  // llvm.load: no intervening store to the same address AND no call
  // between producer and use in the block. Alloca'd slots don't escape
  // emitted code; stores through the exact same SSA value are the only
  // thing that can change the loaded value.
  if (auto L = dyn_cast<LLVM::LoadOp>(Op)) {
    Value AddrV = L.getAddr();
    Block *BB = Op.getBlock();
    for (auto It = ++Block::iterator(&Op);
         It != BB->end() && &*It != User; ++It) {
      if (auto S = dyn_cast<LLVM::StoreOp>(&*It))
        if (S.getAddr() == AddrV) return false;
      if (isa<LLVM::CallOp, func::CallOp>(&*It)) return false;
    }
    return true;
  }

  // Function calls: inlining moves the call's text to the use site. The
  // call's side effects still run at the use's position; they ran before
  // the use in the original order too, so the net timing is unchanged —
  // PROVIDED no store or other call sits between producer and user. Any
  // intervening side-effecting op could observe or be observed by the
  // call, so we conservatively refuse to hop over them. Indirect calls
  // (no callee attribute) can't be textually inlined cleanly because
  // the function pointer cast is already a mouthful.
  if (auto C = dyn_cast<func::CallOp>(Op)) {
    Block *BB = Op.getBlock();
    for (auto It = ++Block::iterator(&Op);
         It != BB->end() && &*It != User; ++It) {
      if (isa<LLVM::StoreOp>(*It)) return false;
      if (isa<LLVM::CallOp, func::CallOp>(*It)) return false;
    }
    (void)C;
    return true;
  }
  if (auto C = dyn_cast<LLVM::CallOp>(Op)) {
    if (!C.getCallee()) return false;
    Block *BB = Op.getBlock();
    for (auto It = ++Block::iterator(&Op);
         It != BB->end() && &*It != User; ++It) {
      if (isa<LLVM::StoreOp>(*It)) return false;
      if (isa<LLVM::CallOp, func::CallOp>(*It)) return false;
    }
    return true;
  }

  return false;
}

// Build the C expression for an inlineable op's result. Operands
// resolve via exprFor, which recurses into further inlineable producers.
bool Emitter::buildInlineExpr(mlir::Operation &Op, std::string &Expr) {
  using namespace mlir;
  if (auto C = dyn_cast<LLVM::ConstantOp>(Op)) {
    auto A = C.getValue();
    if (auto IA = dyn_cast<IntegerAttr>(A)) {
      Expr = formatIntAttr(IA); return true;
    }
    if (auto FA = dyn_cast<FloatAttr>(A)) {
      Expr = formatFloatAttr(FA); return true;
    }
    return false;
  }
  if (auto C = dyn_cast<arith::ConstantOp>(Op)) {
    auto A = C.getValue();
    if (auto FA = dyn_cast<FloatAttr>(A)) {
      Expr = formatFloatAttr(FA); return true;
    }
    if (auto IA = dyn_cast<IntegerAttr>(A)) {
      Expr = formatIntAttr(IA); return true;
    }
    return false;
  }
  if (isa<LLVM::ZeroOp>(Op)) { Expr = "0"; return true; }
  if (auto A = dyn_cast<LLVM::AddressOfOp>(Op)) {
    // Array globals (string literals) and function symbols both decay to
    // a pointer when referenced by name, so `(void*)name` is sufficient —
    // no `&`, no outer paren wrapping. The C-style cast handles both the
    // array→pointer decay (for const-qualified string buffers) and the
    // function→function-pointer decay (outlined parfor / anon callees).
    Expr = "(void*)" + A.getGlobalName().str();
    return true;
  }
  auto bin = [&](const char *cc) {
    Expr = "(" + exprFor(Op.getOperand(0)) + " " + cc + " "
         + exprFor(Op.getOperand(1)) + ")";
    return true;
  };
  if (isa<arith::AddFOp>(Op)) return bin("+");
  if (isa<arith::SubFOp>(Op)) return bin("-");
  if (isa<arith::MulFOp>(Op)) return bin("*");
  if (isa<arith::DivFOp>(Op)) return bin("/");
  if (isa<arith::AddIOp>(Op)) return bin("+");
  if (isa<arith::SubIOp>(Op)) return bin("-");
  if (isa<arith::MulIOp>(Op)) return bin("*");
  // For i1 results use logical operators — the MATLAB frontend routes
  // `&&` / `||` / `!` through arith, but emitting them as bitwise `&` /
  // `|` / `^` in C is both visually confusing and silently skips short-
  // circuit evaluation that C code depends on.
  if (auto AI = dyn_cast<arith::AndIOp>(Op)) {
    if (AI.getResult().getType().isInteger(1)) return bin("&&");
    return bin("&");
  }
  if (auto OI = dyn_cast<arith::OrIOp>(Op)) {
    if (OI.getResult().getType().isInteger(1)) return bin("||");
    return bin("|");
  }
  if (auto XI = dyn_cast<arith::XOrIOp>(Op)) {
    if (XI.getResult().getType().isInteger(1)) {
      // `x ^ true` is logical-not; emit it that way so callers see
      // `!flag` instead of `flag ^ 1`. For XOR against false the value
      // is x, for other i1 XORs fall back to `!=` (the i1 XOR semantics).
      auto isConst = [](mlir::Value V, uint64_t &Out) {
        auto *D = V.getDefiningOp();
        if (!D) return false;
        if (auto C = dyn_cast<arith::ConstantOp>(D)) {
          if (auto IA = dyn_cast<IntegerAttr>(C.getValue())) {
            Out = IA.getValue().getZExtValue();
            return true;
          }
        }
        if (auto C = dyn_cast<LLVM::ConstantOp>(D)) {
          if (auto IA = dyn_cast<IntegerAttr>(C.getValue())) {
            Out = IA.getValue().getZExtValue();
            return true;
          }
        }
        return false;
      };
      uint64_t K = 0;
      if (isConst(Op.getOperand(1), K) && K == 1) {
        Expr = "!" + exprFor(Op.getOperand(0));
        return true;
      }
      if (isConst(Op.getOperand(0), K) && K == 1) {
        Expr = "!" + exprFor(Op.getOperand(1));
        return true;
      }
      return bin("!=");  // generic i1 xor == "not equal"
    }
    return bin("^");
  }
  if (auto C = dyn_cast<arith::CmpFOp>(Op)) {
    const char *cc = "==";
    switch (C.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ: cc = "=="; break;
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE: cc = "!="; break;
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT: cc = "<"; break;
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE: cc = "<="; break;
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT: cc = ">"; break;
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE: cc = ">="; break;
      default: return false;
    }
    Expr = "(" + exprFor(C.getLhs()) + " " + cc + " " + exprFor(C.getRhs()) + ")";
    return true;
  }
  if (auto C = dyn_cast<arith::CmpIOp>(Op)) {
    const char *cc = "==";
    switch (C.getPredicate()) {
      case arith::CmpIPredicate::eq:  cc = "=="; break;
      case arith::CmpIPredicate::ne:  cc = "!="; break;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult: cc = "<"; break;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule: cc = "<="; break;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt: cc = ">"; break;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge: cc = ">="; break;
    }
    Expr = "(" + exprFor(C.getLhs()) + " " + cc + " " + exprFor(C.getRhs()) + ")";
    return true;
  }
  if (auto S = dyn_cast<arith::SelectOp>(Op)) {
    Expr = "(" + exprFor(S.getCondition()) + " ? "
         + exprFor(S.getTrueValue()) + " : "
         + exprFor(S.getFalseValue()) + ")";
    return true;
  }
  Value V = Op.getResult(0);
  if (isa<arith::SIToFPOp, arith::UIToFPOp>(Op)) {
    Expr = "((double)" + exprFor(Op.getOperand(0)) + ")"; return true;
  }
  if (isa<arith::FPToSIOp, arith::FPToUIOp,
          arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::TruncFOp, arith::ExtFOp>(Op)) {
    Expr = "((" + cTypeOfValue(V) + ")" + exprFor(Op.getOperand(0)) + ")";
    return true;
  }
  if (auto L = dyn_cast<LLVM::LoadOp>(Op)) {
    // Inlining from a direct-slot alloca: the slot's identifier IS the
    // value's expression — no `(*(T*)ptr)` wrapping.
    if (auto *D = L.getAddr().getDefiningOp()) {
      auto It = DirectSlots.find(D);
      if (It != DirectSlots.end()) { Expr = It->second; return true; }
    }
    Expr = "(*(" + cTypeOfValue(V) + "*)" + exprFor(L.getAddr()) + ")";
    return true;
  }
  if (auto G = dyn_cast<LLVM::GEPOp>(Op)) {
    std::string ElTy = cTypeOf(G.getElemType());
    std::string E = "((void*)(((" + ElTy + "*)" + exprFor(G.getBase()) + ")";
    for (auto Idx : G.getIndices()) {
      if (auto Vv = llvm::dyn_cast<mlir::Value>(Idx))
        E += " + " + exprFor(Vv);
      else if (auto A = llvm::dyn_cast<IntegerAttr>(Idx))
        E += " + " + std::to_string(A.getInt());
    }
    E += "))";
    Expr = E;
    return true;
  }
  // Function calls inline as `callee(arg, arg, ...)`. Argument positions
  // are outermost — strip any superfluous outer parens from each operand
  // so `fact((n - 1))` reads as `fact(n - 1)`.
  if (auto C = dyn_cast<func::CallOp>(Op)) {
    std::string E = C.getCallee().str() + "(";
    for (unsigned i = 0; i < C.getNumOperands(); ++i) {
      if (i) E += ", ";
      E += dropOuterParens(exprFor(C.getOperand(i)));
    }
    E += ")";
    Expr = E;
    return true;
  }
  if (auto C = dyn_cast<LLVM::CallOp>(Op)) {
    if (!C.getCallee()) return false;
    std::string E = C.getCallee()->str() + "(";
    for (unsigned i = 0; i < C.getNumOperands(); ++i) {
      if (i) E += ", ";
      E += dropOuterParens(exprFor(C.getOperand(i)));
    }
    E += ")";
    Expr = E;
    return true;
  }
  return false;
}

void Emitter::computeInlines(mlir::Region &R) {
  for (auto &B : R.getBlocks()) {
    for (auto &Op : B.getOperations()) {
      if (canInline(Op)) InlinedOps.insert(&Op);
      for (auto &SubR : Op.getRegions()) computeInlines(SubR);
    }
  }
}

// Does the before-region of W consist of only the scf.condition terminator
// plus ops the emitter will inline into the condition expression? Mirrors
// the BeforeIsCondOnly check inside the scf.while emitter so the
// for-loop pre-scan and that emitter always agree on when the "natural"
// while-shape is emittable.
static bool beforeRegionIsCondOnly(mlir::scf::WhileOp W,
    const llvm::DenseSet<mlir::Operation *> &InlinedOps) {
  for (auto &Inner : W.getBefore().front().getOperations()) {
    if (mlir::isa<mlir::scf::ConditionOp>(Inner)) continue;
    if (InlinedOps.count(&Inner)) continue;
    return false;
  }
  return true;
}

bool Emitter::matchForPattern(mlir::scf::WhileOp W, ForLoopInfo &Info) {
  // Shape expected from LowerSeqLoops::lowerForOp: one f64 iter-arg, the
  // condition forwards the iter-arg unchanged, the condition itself is a
  // single arith.cmpf OLE|OGE iv, end, and the after-block ends with
  // arith.addf iv, step + scf.yield %add. Everything else is left as a
  // plain while.
  if (W.getInits().size() != 1) return false;
  mlir::Block &Before = W.getBefore().front();
  mlir::Block &After = W.getAfter().front();
  if (Before.getNumArguments() != 1 || After.getNumArguments() != 1)
    return false;
  auto F64 = mlir::Float64Type::get(W.getContext());
  if (Before.getArgument(0).getType() != F64) return false;
  if (After.getArgument(0).getType() != F64) return false;

  if (!beforeRegionIsCondOnly(W, InlinedOps)) return false;
  auto Cond = mlir::cast<mlir::scf::ConditionOp>(Before.getTerminator());
  // The condition must forward exactly the before-region's IV so the
  // after-block's IV is identified with the one being compared.
  if (Cond.getArgs().size() != 1) return false;
  if (Cond.getArgs()[0] != Before.getArgument(0)) return false;

  // Strip break / continue flag inversions out of the conjunction before
  // pattern-matching. A for-loop body with `break;` or `continue;` adds
  // `& !did_break` to the scf.condition; after un-lowering to C break /
  // continue statements the inversion goes away, so what's left has to
  // reduce to a single cmpf on the IV.
  llvm::SmallVector<mlir::Value, 2> CondParts;
  gatherNonFlagConjuncts(Cond.getCondition(), CondParts);
  if (CondParts.size() != 1) return false;
  auto Cmp = CondParts[0].getDefiningOp<mlir::arith::CmpFOp>();
  if (!Cmp) return false;
  if (Cmp.getLhs() != Before.getArgument(0)) return false;
  auto Pred = Cmp.getPredicate();
  if (Pred != mlir::arith::CmpFPredicate::OLE &&
      Pred != mlir::arith::CmpFPredicate::OGE) return false;
  Info.End = Cmp.getRhs();
  Info.IsDecreasing = (Pred == mlir::arith::CmpFPredicate::OGE);

  // After-block: the last two ops must be arith.addf + scf.yield, with the
  // addf's lhs being the after-block iv and the yield returning %addf.
  if (After.getOperations().size() < 2) return false;
  auto &TailYield = After.back();
  auto Yld = mlir::dyn_cast<mlir::scf::YieldOp>(TailYield);
  if (!Yld || Yld.getResults().size() != 1) return false;
  auto *AddRaw = Yld.getResults()[0].getDefiningOp();
  auto Add = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(AddRaw);
  if (!Add) return false;
  if (Add.getLhs() != After.getArgument(0)) return false;
  // The add must be second-to-last — i.e. the increment op that immediately
  // precedes the yield — so eliding both leaves no orphan ops in the body.
  if (Add->getNextNode() != Yld.getOperation()) return false;

  Info.Init = W.getInits()[0];
  Info.Step = Add.getRhs();
  Info.AddOp = Add.getOperation();
  Info.YieldOp = Yld.getOperation();
  return true;
}

void Emitter::scanForLoopPatterns(mlir::Region &R) {
  // First pass: identify for-pattern scf.while ops and propose a bind-store
  // + slot alloca for each. Allocas may be claimed by multiple for-loops
  // (e.g. `for i = 1:10 ... end; for i = 1:3 ... end;` reuses the `i`
  // slot across two scf.while ops); we collect the list so the second
  // pass can decide whether every use of the slot is inside at least one
  // claimant's after-region.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 2>>
      SlotClaimants;

  R.walk([&](mlir::scf::WhileOp W) {
    ForLoopInfo Info;
    if (!matchForPattern(W, Info)) return;

    mlir::Block &After = W.getAfter().front();
    mlir::Value Iv = After.getArgument(0);
    for (auto &Op : After.getOperations()) {
      if (&Op == Info.AddOp || &Op == Info.YieldOp) break;
      auto Store = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op);
      if (!Store) continue;
      if (Store.getValue() != Iv) continue;
      auto Alloca = Store.getAddr().getDefiningOp<mlir::LLVM::AllocaOp>();
      if (!Alloca) break;
      auto NA = Alloca->getAttrOfType<mlir::StringAttr>("matlab.name");
      if (!NA) break;
      Info.BindStore = Store.getOperation();
      Info.SlotAlloca = Alloca.getOperation();
      SlotClaimants[Info.SlotAlloca].push_back(W.getOperation());
      break;
    }
    ForPatterns[W.getOperation()] = std::move(Info);
  });

  // Second pass: a slot is fusable when every use lives inside at least
  // one of its claimants' after-regions. If any use escapes to function
  // scope (or to an unrelated region), fusion would drop a value that
  // code outside the loop still reads.
  llvm::DenseMap<mlir::Operation *, bool> SlotFusable;
  for (auto &Entry : SlotClaimants) {
    mlir::Operation *Slot = Entry.first;
    auto &Claimants = Entry.second;
    bool OK = true;
    for (auto &Use : Slot->getUses()) {
      mlir::Operation *User = Use.getOwner();
      bool InsideAny = false;
      for (mlir::Operation *W : Claimants) {
        mlir::Region *Loop = &W->getRegion(1);  // after-region
        mlir::Region *P = User->getParentRegion();
        while (P) {
          if (P == Loop) { InsideAny = true; break; }
          P = P->getParentRegion();
        }
        if (InsideAny) break;
      }
      if (!InsideAny) { OK = false; break; }
    }
    SlotFusable[Slot] = OK;
  }

  // Third pass: commit names and populate SuppressedOps / FusedForSlots.
  // The trailing arith.addf and scf.yield are always absorbed; the
  // iv-binding store and the slot's decl-emission are only skipped when
  // fusion is possible.
  for (auto &KV : ForPatterns) {
    ForLoopInfo &Info = KV.second;
    SuppressedOps.insert(Info.AddOp);
    SuppressedOps.insert(Info.YieldOp);

    bool Fuse = Info.SlotAlloca && SlotFusable.lookup(Info.SlotAlloca);
    if (Fuse) {
      Info.FuseSlot = true;
      auto It = FusedForSlotName.find(Info.SlotAlloca);
      if (It == FusedForSlotName.end()) {
        auto NA = Info.SlotAlloca->getAttrOfType<mlir::StringAttr>(
            "matlab.name");
        std::string N = uniqueName(NA.getValue());
        FusedForSlotName[Info.SlotAlloca] = N;
        Info.IvName = N;
      } else {
        Info.IvName = It->second;
      }
      FusedForSlots.insert(Info.SlotAlloca);
      SuppressedOps.insert(Info.BindStore);
    } else {
      Info.FuseSlot = false;
      Info.IvName = freshName();
    }
  }
}

// Is `V` a compile-time constant integer with value `Want`?
static bool isConstInt(mlir::Value V, uint64_t Want) {
  auto *D = V.getDefiningOp();
  if (!D) return false;
  if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D))
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getValue().getZExtValue() == Want;
  if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D))
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      return IA.getValue().getZExtValue() == Want;
  return false;
}

bool Emitter::isFlagInversion(mlir::Value V) {
  // Names notwithstanding, this also strips any subexpression we can
  // statically prove reduces to `true` in an i1 conjunction — constants,
  // folds of the frontend's `xor(x, true)` idiom, or a read of a flag
  // slot we're about to re-lower to a keyword. All of those are safe to
  // drop from `cond && <strippable>` without changing meaning.
  if (!V.getType().isInteger(1)) return false;
  if (isConstInt(V, 1)) return true;
  auto Xor = V.getDefiningOp<mlir::arith::XOrIOp>();
  if (!Xor) return false;
  if (!Xor.getResult().getType().isInteger(1)) return false;
  mlir::Value Flag;
  if (isConstInt(Xor.getRhs(), 1)) Flag = Xor.getLhs();
  else if (isConstInt(Xor.getLhs(), 1)) Flag = Xor.getRhs();
  else return false;
  // xor(const-false, const-true) = true. Mem2RegLite often produces this
  // shape when the flag slot has no actual store-of-true (i.e. the body
  // has a `continue;` but no `break;`, or vice versa): the pass promotes
  // the zero-initialised slot out, leaving a constant fold that's fully
  // statically true.
  if (isConstInt(Flag, 0)) return true;
  auto Load = Flag.getDefiningOp<mlir::LLVM::LoadOp>();
  if (!Load) return false;
  auto *Addr = Load.getAddr().getDefiningOp();
  if (!Addr) return false;
  return BreakFlagSlots.count(Addr) || ContinueFlagSlots.count(Addr);
}

void Emitter::gatherNonFlagConjuncts(mlir::Value V,
    llvm::SmallVectorImpl<mlir::Value> &Out) {
  if (isFlagInversion(V)) return;
  if (auto And = V.getDefiningOp<mlir::arith::AndIOp>()) {
    // Only walk into ands that produce i1 — otherwise we'd risk splitting
    // a user's bitwise `&` into misleading pieces.
    if (And.getResult().getType().isInteger(1)) {
      gatherNonFlagConjuncts(And.getLhs(), Out);
      gatherNonFlagConjuncts(And.getRhs(), Out);
      return;
    }
  }
  Out.push_back(V);
}

void Emitter::scanBreakContinueFlags(mlir::Region &R) {
  // Phase 1: identify the break / continue flag slots by their
  // matlab.name attribute (LowerScalarSlots forwards it from the
  // frontend's __did_break / __did_continue symbol).
  R.walk([&](mlir::LLVM::AllocaOp A) {
    auto NA = A->getAttrOfType<mlir::StringAttr>("matlab.name");
    if (!NA) return;
    if (NA.getValue() == "__did_break") {
      BreakFlagSlots.insert(A.getOperation());
      SuppressedOps.insert(A.getOperation());
    } else if (NA.getValue() == "__did_continue") {
      ContinueFlagSlots.insert(A.getOperation());
      SuppressedOps.insert(A.getOperation());
    }
  });

  if (BreakFlagSlots.empty() && ContinueFlagSlots.empty()) return;

  // Phase 2: classify every store into a flag slot. A constant-false
  // store is either the pre-loop initialisation or the end-of-iteration
  // continue reset — both are elided. A constant-true store only ever
  // appears inside a scf.if's then-region from the frontend.
  R.walk([&](mlir::LLVM::StoreOp S) {
    auto *Addr = S.getAddr().getDefiningOp();
    if (!Addr) return;
    bool IsBreak = BreakFlagSlots.count(Addr) > 0;
    bool IsCont = ContinueFlagSlots.count(Addr) > 0;
    if (!IsBreak && !IsCont) return;
    if (isConstInt(S.getValue(), 0)) {
      SuppressedOps.insert(S.getOperation());
    }
  });

  // Phase 3: recognise the scf.if shapes the frontend emits around these
  // flags and decide how to collapse them.
  //  - `scf.if cond { <const>; store true → flag } else {}` becomes
  //    `if (cond) break;` (or continue). The constant-true producer gets
  //    absorbed too so it doesn't dangle as a useless `bool v = 1;`.
  //  - `scf.if (!flag [ & !flag]) { ... } else {}` is the post-break /
  //    post-continue guard that exists only so the lowered stores take
  //    effect; since we now emit real break/continue statements, we can
  //    drop the if wrapper and emit the body inline.
  R.walk([&](mlir::scf::IfOp If) {
    if (!If.getElseRegion().empty()) {
      bool ElseTrivial = true;
      for (auto &Blk : If.getElseRegion().getBlocks()) {
        for (auto &Inner : Blk.getOperations()) {
          if (mlir::isa<mlir::scf::YieldOp>(Inner)) continue;
          ElseTrivial = false;
          break;
        }
        if (!ElseTrivial) break;
      }
      if (!ElseTrivial) return;
    }
    if (If.getNumResults() != 0) return;

    // Break/continue short-circuit: single store of true into a flag slot
    // is the only real op in the then-region.
    {
      mlir::LLVM::StoreOp FlagStore;
      bool OtherRealOps = false;
      for (auto &Blk : If.getThenRegion().getBlocks()) {
        for (auto &Inner : Blk.getOperations()) {
          if (mlir::isa<mlir::scf::YieldOp>(Inner)) continue;
          if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Inner)) {
            auto *Addr = S.getAddr().getDefiningOp();
            bool Flag = Addr && (BreakFlagSlots.count(Addr) ||
                                 ContinueFlagSlots.count(Addr));
            if (Flag && isConstInt(S.getValue(), 1) && !FlagStore) {
              FlagStore = S;
              continue;
            }
          }
          if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Inner)) {
            // The `const true` feeding the store is pure; tolerate it.
            (void)C;
            continue;
          }
          if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Inner)) {
            (void)C;
            continue;
          }
          OtherRealOps = true;
        }
      }
      if (FlagStore && !OtherRealOps) {
        auto *Addr = FlagStore.getAddr().getDefiningOp();
        FlagIfKind[If.getOperation()] =
            BreakFlagSlots.count(Addr) ? "break" : "continue";
        SuppressedOps.insert(FlagStore.getOperation());
        // `const true` producer is still emitted as part of inline
        // expression resolution (it's the 1 in `store 1 -> flag`), but
        // with the store suppressed the producer has no user and the
        // emitter's inlining pass will drop it naturally.
        return;
      }
    }

    // Pure flag-guard: every conjunct in the condition strips out as
    // statically-true. The if existed only to gate the body on break /
    // continue flags the frontend threaded through; now that we re-lower
    // those stores into keyword statements, the if is dead weight.
    llvm::SmallVector<mlir::Value, 2> Kept;
    gatherNonFlagConjuncts(If.getCondition(), Kept);
    if (Kept.empty())
      InlinedIfs.insert(If.getOperation());
  });
}

// Emit a `#line N "file"` directive at the given indent level if the
// location is a FileLineColLoc and hasn't been emitted already for this
// (file, line) pair.
// Classify one line in the source buffer.
namespace {
enum class LineKind { Blank, Comment, Code };
struct LineInfo { LineKind Kind; std::string_view Body; };
LineInfo classifyLine(std::string_view Text) {
  size_t I = 0;
  while (I < Text.size() && (Text[I] == ' ' || Text[I] == '\t')) ++I;
  if (I == Text.size()) return {LineKind::Blank, {}};
  if (Text[I] != '%')   return {LineKind::Code, {}};
  ++I;  // skip leading '%'
  if (I < Text.size() && Text[I] == ' ') ++I;
  return {LineKind::Comment, Text.substr(I)};
}

// Walk a code line looking for the first `%` that is NOT inside a string
// literal. Returns std::string_view::npos when the line has no trailing
// comment. Mirrors the MATLAB lexer's string / transpose disambiguation:
//   - `"..."`   : always a string, `\` escapes next character
//   - `'...'`   : string unless preceded by an identifier / number / `)` / `]`
//                 / `.` (in which case `'` is the conjugate-transpose op)
//   - `%`       : anywhere outside strings starts a line comment
size_t findTrailingCommentStart(std::string_view S) {
  unsigned char PrevNonSpace = 0;  // 0 at start-of-line: quote is NOT transpose
  size_t I = 0;
  while (I < S.size()) {
    unsigned char C = S[I];
    if (C == '%') return I;
    if (C == '"') {
      // Walk past the string body. Handle `""` as an escaped quote.
      ++I;
      while (I < S.size() && S[I] != '"') {
        if (S[I] == '\\' && I + 1 < S.size()) I += 2;
        else ++I;
      }
      if (I < S.size()) ++I;
      PrevNonSpace = '"';
      continue;
    }
    if (C == '\'') {
      bool IsTranspose =
          std::isalnum(PrevNonSpace) || PrevNonSpace == ')' ||
          PrevNonSpace == ']' || PrevNonSpace == '.' || PrevNonSpace == '_';
      ++I;
      if (!IsTranspose) {
        // Char literal: skip to next unescaped `'`. MATLAB uses `''` to
        // embed a single quote inside a char array.
        while (I < S.size()) {
          if (S[I] == '\'') {
            if (I + 1 < S.size() && S[I + 1] == '\'') { I += 2; continue; }
            ++I; break;
          }
          ++I;
        }
      }
      PrevNonSpace = '\'';
      continue;
    }
    if (!std::isspace(C)) PrevNonSpace = C;
    ++I;
  }
  return std::string_view::npos;
}
} // namespace

void Emitter::flushPendingTrailing() {
  if (PendingTrailingLine < 0) return;
  indent(PendingTrailingIndent);
  OS << "// " << PendingTrailingText << "\n";
  TrailingEmittedLines.insert(PendingTrailingLine);
  PendingTrailingLine = -1;
  PendingTrailingIndent = 0;
  PendingTrailingText.clear();
}

bool Emitter::emitLeadingComments(llvm::StringRef FullPath, int AfterLine,
                                  int Line, int Indent, bool FunctionHeader) {
  if (!SM || FullPath.empty() || Line <= 0) return false;
  matlab::FileID F = SM->findFileByName(std::string_view(FullPath));
  if (F == 0) return false;

  // First, find the earliest line `L0` in (AfterLine, Line) such that every
  // line in [L0, Line) is either a `%` comment or blank. That block is the
  // current stmt's leading prelude. Anything further back that's `Code` has
  // already been consumed by a previous emit (or is an SSA-only stmt that
  // produced no output).
  int Start = Line;
  for (int L = Line - 1; L > AfterLine; --L) {
    auto Info = classifyLine(SM->getLineText(F, (uint32_t)L));
    if (Info.Kind == LineKind::Code) break;
    Start = L;
  }
  if (Start == Line) return false;  // nothing in the prelude

  // Doxygen-style rendering for function-leading blocks: `/**\n *
  // line\n ... */`. The opening `/**` and closing `*/` frame the block,
  // blank lines in the source become blank ` *` lines inside the block.
  // Only triggered when the flag is on AND this is a function-header
  // context — inline comments between statements keep the `//` style.
  if (Doxygen && FunctionHeader) {
    // Trim leading blanks inside the block so `/**\n *\n * text ...`
    // doesn't gain a gratuitous empty row.
    while (Start < Line) {
      auto Info = classifyLine(SM->getLineText(F, (uint32_t)Start));
      if (Info.Kind != LineKind::Blank) break;
      ++Start;
    }
    if (Start == Line) return false;
    indent(Indent);
    OS << "/**\n";
    bool LastWasBlank = false;
    for (int L = Start; L < Line; ++L) {
      auto Info = classifyLine(SM->getLineText(F, (uint32_t)L));
      if (Info.Kind == LineKind::Blank) {
        if (LastWasBlank) continue;
        indent(Indent); OS << " *\n";
        LastWasBlank = true;
        continue;
      }
      indent(Indent);
      OS << " * " << Info.Body << "\n";
      LastWasBlank = false;
    }
    indent(Indent);
    OS << " */\n";
    return true;
  }

  // Walk Start..Line-1 emitting blanks and comments in source order.
  // Suppress leading blanks (so a block that begins with `// comment`
  // doesn't get a blank row above it right after `{`) and collapse
  // consecutive blanks into a single empty line.
  bool EmittedAny = false;
  bool CanEmitBlank = !AtBlockStart;
  bool LastWasBlank = false;
  for (int L = Start; L < Line; ++L) {
    auto Info = classifyLine(SM->getLineText(F, (uint32_t)L));
    if (Info.Kind == LineKind::Blank) {
      if (!CanEmitBlank || LastWasBlank) continue;
      OS << "\n";
      LastWasBlank = true;
      EmittedAny = true;
      continue;
    }
    // Comment.
    indent(Indent);
    OS << "// " << Info.Body << "\n";
    CanEmitBlank = true;
    LastWasBlank = false;
    EmittedAny = true;
  }
  return EmittedAny;
}

void Emitter::emitLineDirective(mlir::Location L, int Indent) {
  // Unwrap NameLoc / FusedLoc to reach the underlying FileLineColLoc if
  // present. Ops produced by builders often carry wrapped locations.
  mlir::FileLineColLoc FL;
  if ((FL = mlir::dyn_cast<mlir::FileLineColLoc>(L))) {
    // direct.
  } else if (auto NL = mlir::dyn_cast<mlir::NameLoc>(L)) {
    FL = mlir::dyn_cast<mlir::FileLineColLoc>(NL.getChildLoc());
  } else if (auto FuL = mlir::dyn_cast<mlir::FusedLoc>(L)) {
    for (auto Sub : FuL.getLocations())
      if ((FL = mlir::dyn_cast<mlir::FileLineColLoc>(Sub))) break;
  }
  if (!FL) return;
  std::string FullPath = FL.getFilename().str();
  int Line = static_cast<int>(FL.getLine());
  if (FullPath.empty() || Line <= 0) return;
  // Emit only the basename so the generated .c is portable across
  // build machines and doesn't leak an absolute path. Debuggers resolve
  // #line filenames against the compilation directory.
  std::string File = FullPath;
  if (auto Slash = File.find_last_of("/\\"); Slash != std::string::npos)
    File = File.substr(Slash + 1);
  if (File == LastLineFile && Line == LastLineNum) {
    // Same MATLAB line, same file — possibly re-emitted at a deeper indent
    // (e.g. the function signature fires at indent 0, then the body opens
    // and an op on the same line fires at indent 1). If we already queued
    // a trailing for this line, upgrade its indent to match the current
    // context so it lines up with the body statement it annotates.
    if (PendingTrailingLine == Line && Indent > PendingTrailingIndent)
      PendingTrailingIndent = Indent;
    return;
  }

  // Crossing a line boundary: flush any trailing comment on the line we're
  // leaving first so `x = 1; // trailing` lands directly under its stmt.
  flushPendingTrailing();

  bool SameFile = !LastLineFile.empty() && File == LastLineFile;
  bool ForwardJump = SameFile && Line > LastLineNum;

  // Scan for MATLAB `%` comments in the interval (LastLineNum, Line).
  // Forward-within-the-same-file constraint keeps us from replaying
  // comments when an op's location bounces backward (e.g. the scf.if
  // closing brace inheriting the if-header's line). The scan ALSO emits
  // preserved blank lines (interleaved with the comments) so its output
  // matches the source's paragraph structure — we then skip the naive
  // blank-preservation fallback below if it emitted anything.
  bool ScanEmitted = false;
  if (ForwardJump) {
    ScanEmitted = emitLeadingComments(FullPath, LastLineNum, Line, Indent);
  } else if (LastLineFile.empty()) {
    // Fresh function entry or cross-file jump: scan a bounded stretch back
    // so the function's leading doc-comment block shows up above the
    // signature / first statement. Pass FunctionHeader=true so Doxygen
    // mode can render the block as `/** ... */` rather than `//` lines.
    ScanEmitted = emitLeadingComments(FullPath, Line - 64, Line, Indent,
                                      /*FunctionHeader=*/true);
  }

  // Blank-line preservation fallback: if the scan didn't emit anything
  // (no comments or blanks in the interval, or cross-file / first-op
  // case), still reproduce a single blank for a multi-line forward
  // jump so paragraph-style breaks survive even when no `%` sits in
  // between. Never right after a `{`.
  if (!ScanEmitted && !AtBlockStart && ForwardJump &&
      Line > LastLineNum + 1) {
    OS << "\n";
  }
  AtBlockStart = false;
  LastLineFile = File;
  LastLineNum = Line;

  // Scan this line for a trailing `% comment` past the stmt body; stash
  // it as pending so it lands right after this line's C output when we
  // next cross a line boundary (or at block exit). Skip lines we've
  // already emitted a trailing for — a synthesized op that bounces back
  // to an earlier line must not replay the comment.
  if (SM && !TrailingEmittedLines.count(Line)) {
    matlab::FileID FID =
        SM->findFileByName(std::string_view(FullPath));
    if (FID != 0) {
      std::string_view Text = SM->getLineText(FID, (uint32_t)Line);
      size_t Pos = findTrailingCommentStart(Text);
      if (Pos != std::string_view::npos) {
        size_t Body = Pos + 1;
        if (Body < Text.size() && Text[Body] == ' ') ++Body;
        std::string_view Trailing = Text.substr(Body);
        // Ignore pure whitespace after `%` — nothing meaningful to emit.
        size_t NonSpace = Trailing.find_first_not_of(" \t");
        if (NonSpace != std::string_view::npos) {
          PendingTrailingLine = Line;
          PendingTrailingIndent = Indent;
          PendingTrailingText.assign(Trailing);
        }
      }
    }
  }

  if (NoLine) return;
  // Only emit `#line` on forward progression (or the first emit of a
  // function / cross-file switch). A synthesized op that bounces the
  // location back to an earlier line would otherwise re-point the
  // debugger backward, which is useless noise. `SameFile` with
  // backward motion is suppressed — the prior `#line` is still in
  // effect for those lines.
  if (SameFile && !ForwardJump) return;
  indent(Indent);
  OS << "#line " << Line << " \"" << File << "\"\n";
}

std::string Emitter::cTypeOf(mlir::Type T) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    unsigned W = IT.getWidth();
    if (W == 1)  return "bool";
    if (W == 8)  return "int8_t";
    if (W == 16) return "int16_t";
    if (W == 32) return "int32_t";
    if (W == 64) return "int64_t";
    return "int64_t";
  }
  if (mlir::isa<mlir::Float32Type>(T)) return "float";
  if (mlir::isa<mlir::Float64Type>(T)) return "double";
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(T)) return "void*";
  if (mlir::isa<mlir::IndexType>(T)) return "int64_t";
  // Fallback: opaque pointer.
  return "void*";
}

// ---------------------------------------------------------------------------
// Globals (string constants produced by LowerIO via llvm.mlir.global)
// ---------------------------------------------------------------------------

std::string Emitter::constStr(mlir::LLVM::GlobalOp G) {
  // The value attribute is a StringAttr with the raw bytes.
  auto Val = G.getValueAttr();
  if (!Val) return "";
  if (auto S = mlir::dyn_cast<mlir::StringAttr>(Val)) {
    return S.getValue().str();
  }
  return "";
}

void Emitter::emitGlobal(mlir::LLVM::GlobalOp G) {
  std::string N = G.getSymName().str();
  GlobalStrs[G.getOperation()] = N;
  std::string Raw = constStr(G);
  UsedNames.insert(N);

  // If every byte is printable ASCII (or a common whitespace escape),
  // emit as a quoted string literal — far more readable than a byte array
  // when inspecting emitted .c files by hand.
  bool ASCIISafe = true;
  for (unsigned char C : Raw) {
    if (C >= 0x20 && C < 0x7F) continue;
    if (C == '\n' || C == '\t' || C == '\r') continue;
    ASCIISafe = false;
    break;
  }
  if (ASCIISafe) {
    // Drop the explicit array size — C++ requires room for the implicit
    // null terminator, and the runtime never reads past Raw.size() anyway
    // (the length is passed as a separate int64_t argument).
    OS << "static const char " << N << "[] = \"";
    for (unsigned char C : Raw) {
      switch (C) {
        case '\\': OS << "\\\\"; break;
        case '"':  OS << "\\\""; break;
        case '\n': OS << "\\n"; break;
        case '\t': OS << "\\t"; break;
        case '\r': OS << "\\r"; break;
        default:   OS << (char)C; break;
      }
    }
    OS << "\";\n";
    return;
  }
  // Byte-array fallback — unsigned char so bytes > 127 don't trip C++
  // narrowing conversion warnings on non-ASCII content.
  OS << "static const unsigned char " << N << "[" << Raw.size() << "] = {";
  for (size_t i = 0; i < Raw.size(); ++i) {
    if (i) OS << ",";
    OS << (int)(unsigned char)Raw[i];
  }
  OS << "};\n";
}

// ---------------------------------------------------------------------------
// Prolog / decls
// ---------------------------------------------------------------------------

void Emitter::emitProlog() {
  OS << "// Generated by matlabc " << (Cpp ? "-emit-cpp" : "-emit-c")
     << ". Do not edit.\n";
  OS << "#include <stdint.h>\n";
  if (!Cpp) OS << "#include <stdbool.h>\n";
  // IO-substitution headers. When the module has no parfor (no mutex
  // coordination needed) we can collapse the matlab_disp_* runtime calls
  // into direct stdio / iostream output, which reads as hand-written C
  // / modern C++. The flags were computed in precomputeModuleProperties.
  if (NeedsStdio)    OS << "#include <stdio.h>\n";
  if (NeedsIostream) OS << "#include <iostream>\n";
  OS << "\n";
  // Runtime function prototypes are emitted per-module below, with void*
  // for all pointer params so the same declaration works for C and C++
  // (C linkage handles the type bridging to the runtime's typed params).
}

void Emitter::precomputeModuleProperties(mlir::ModuleOp M) {
  HasParfor = false;
  LiveRuntimeFuncs.clear();
  LiveGlobals.clear();
  bool AnyDispStrLiteral = false;
  bool AnyDispScalar = false;
  // First pass: detect parfor. IO substitution is gated on its absence,
  // so we need it settled before deciding whether a call survives.
  M.walk([&](mlir::LLVM::CallOp C) {
    if (auto Callee = C.getCallee())
      if (*Callee == "matlab_parfor_dispatch") HasParfor = true;
  });
  // Second pass: for each call, decide whether it'll be substituted. If
  // it survives, mark its callee live so the extern block keeps it.
  M.walk([&](mlir::LLVM::CallOp C) {
    auto Callee = C.getCallee();
    if (!Callee) return;
    llvm::StringRef Name = *Callee;
    bool Substituted = false;
    if (!HasParfor) {
      if (Name == "matlab_disp_str" && C.getNumOperands() >= 1) {
        // Literal-addressof: first operand produced by llvm.mlir.addressof
        // of a global whose value is a StringAttr.
        if (auto Addr = C.getOperand(0)
                .getDefiningOp<mlir::LLVM::AddressOfOp>()) {
          auto ParentMod = C->getParentOfType<mlir::ModuleOp>();
          if (auto G = ParentMod.lookupSymbol<mlir::LLVM::GlobalOp>(
                  Addr.getGlobalName()))
            if (auto S = mlir::dyn_cast_or_null<mlir::StringAttr>(
                    G.getValueAttr())) {
              // Must be printable for the substitution to fire.
              bool Printable = true;
              for (unsigned char Ch : S.getValue()) {
                bool OK = (Ch >= 0x20 && Ch < 0x7F) || Ch == '\n' ||
                          Ch == '\t' || Ch == '\r';
                if (!OK) { Printable = false; break; }
              }
              if (Printable) {
                Substituted = true;
                AnyDispStrLiteral = true;
              }
            }
        }
      }
      if (Name == "matlab_disp_f64" || Name == "matlab_disp_i64") {
        Substituted = true;
        AnyDispScalar = true;
      }
    }
    if (!Substituted) {
      LiveRuntimeFuncs.insert(Name);
      // A surviving call keeps its operand-producing globals live.
      for (mlir::Value Opnd : C.getOperands()) {
        if (auto A = Opnd.getDefiningOp<mlir::LLVM::AddressOfOp>())
          LiveGlobals.insert(A.getGlobalName());
      }
    }
  });
  // addressof uses outside `llvm.call` (indirect dispatch, parfor bodies
  // loaded via function-pointer) also keep their global live. Walk every
  // addressof and check: if any user is NOT an llvm.call we'd have
  // substituted, mark the global live. This catches things we didn't
  // see in the call-walk above.
  M.walk([&](mlir::LLVM::AddressOfOp A) {
    for (mlir::Operation *U : A.getResult().getUsers()) {
      auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(U);
      if (!Call) {
        // Non-call consumer (e.g. func-ptr table) — definitely live.
        LiveGlobals.insert(A.getGlobalName());
        return;
      }
      // Already handled above in the call-walk; skip.
    }
  });
  if (AnyDispStrLiteral || AnyDispScalar) {
    if (Cpp) NeedsIostream = true;
    else     NeedsStdio = true;
  }
}

bool Emitter::writeQuotedStringLiteral(std::ostream &OS,
                                       llvm::StringRef Raw) {
  for (unsigned char C : Raw) {
    if (C >= 0x20 && C < 0x7F) continue;
    if (C == '\n' || C == '\t' || C == '\r') continue;
    return false;  // non-printable — caller falls back.
  }
  OS << '"';
  for (unsigned char C : Raw) {
    switch (C) {
      case '\\': OS << "\\\\"; break;
      case '"':  OS << "\\\""; break;
      case '\n': OS << "\\n"; break;
      case '\t': OS << "\\t"; break;
      case '\r': OS << "\\r"; break;
      default:   OS << (char)C; break;
    }
  }
  OS << '"';
  return true;
}

bool Emitter::tryEmitIOSubstitution(mlir::LLVM::CallOp Call, int Indent) {
  if (HasParfor) return false;
  auto Callee = Call.getCallee();
  if (!Callee) return false;
  llvm::StringRef Name = *Callee;

  if (Name == "matlab_disp_str") {
    // Requires the address operand to come from a GlobalOp we can
    // reconstruct the string content from. Non-literal disp (dynamic
    // strings) keeps the runtime call so semantics are preserved.
    if (Call.getNumOperands() < 1) return false;
    auto Addr = Call.getOperand(0).getDefiningOp<mlir::LLVM::AddressOfOp>();
    if (!Addr) return false;
    auto ParentMod = Call->getParentOfType<mlir::ModuleOp>();
    auto G = ParentMod.lookupSymbol<mlir::LLVM::GlobalOp>(Addr.getGlobalName());
    if (!G) return false;
    auto S = mlir::dyn_cast_or_null<mlir::StringAttr>(G.getValueAttr());
    if (!S) return false;
    llvm::StringRef Raw = S.getValue();
    // Pre-check: non-printable chars force fallback because we can't
    // embed them verbatim in a `"..."` literal without escape gymnastics
    // we don't support yet.
    for (unsigned char C : Raw) {
      bool OK = (C >= 0x20 && C < 0x7F) || C == '\n' || C == '\t' || C == '\r';
      if (!OK) return false;
    }
    indent(Indent);
    if (Cpp) {
      OS << "std::cout << ";
      writeQuotedStringLiteral(OS, Raw);
      OS << " << '\\n';\n";
    } else {
      OS << "puts(";
      writeQuotedStringLiteral(OS, Raw);
      OS << ");\n";
    }
    return true;
  }

  if (Name == "matlab_disp_f64" && Call.getNumOperands() == 1) {
    // Integer-valued doubles are formatted with a trailing `.0` (see
    // formatFloatAttr), so `printf("%g\n", <expr>)` is type-correct
    // without an explicit `(double)` cast. If the arg ends up being a
    // non-literal SSA value of type double, no cast is needed either —
    // it already has the right type.
    indent(Indent);
    if (Cpp) {
      OS << "std::cout << " << stmtExpr(Call.getOperand(0)) << " << '\\n';\n";
    } else {
      OS << "printf(\"%g\\n\", " << stmtExpr(Call.getOperand(0)) << ");\n";
    }
    return true;
  }

  if (Name == "matlab_disp_i64" && Call.getNumOperands() == 1) {
    indent(Indent);
    if (Cpp) {
      OS << "std::cout << " << stmtExpr(Call.getOperand(0)) << " << '\\n';\n";
    } else {
      OS << "printf(\"%lld\\n\", (long long)" << stmtExpr(Call.getOperand(0))
         << ");\n";
    }
    return true;
  }

  return false;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

bool Emitter::run(mlir::ModuleOp M) {
  precomputeModuleProperties(M);
  emitProlog();

  // -- Pre-emission checks: every defined function must have 0 or 1 results.
  // The printer has no story for multi-result returns (no pass emits them
  // today; guarding is cheap). Fail fast rather than emit broken C.
  for (auto &Op : M.getBody()->getOperations()) {
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      unsigned N = F.getFunctionType().getNumResults();
      if (N > 1) {
        fail(("func.func @" + F.getSymName() +
              " has " + std::to_string(N) +
              " results; emitter supports at most 1").str());
        return false;
      }
    } else if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      // LLVM funcs return a single type (possibly void); no additional check.
    }
  }

  // Pass 0: `extern "C"` runtime prototypes for every llvm.func that's
  // only a declaration (the matlab_* entries imported by LowerIO /
  // LowerTensorOps / LowerParfor). Buffer first so an empty section
  // (every runtime call got IO-substituted) is elided completely —
  // no dangling `extern "C" {}` + comment pair.
  {
    std::ostringstream Buf;
    for (auto &Op : M.getBody()->getOperations()) {
      auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op);
      if (!F) continue;
      if (!F.getBody().empty()) continue;
      UsedNames.insert(F.getSymName().str());
      if (LiveRuntimeFuncs.find(F.getSymName()) == LiveRuntimeFuncs.end())
        continue;
      auto FT = F.getFunctionType();
      std::string RetTy =
          mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())
              ? std::string("void")
              : cTypeOf(FT.getReturnType());
      Buf << "extern " << RetTy << " " << F.getSymName().str() << "(";
      for (unsigned i = 0; i < FT.getNumParams(); ++i) {
        if (i) Buf << ", ";
        Buf << cTypeOf(FT.getParamType(i));
      }
      if (FT.getNumParams() == 0) Buf << "void";
      Buf << ");\n";
    }
    if (!Buf.str().empty()) {
      OS << "// Runtime prototypes (linked against runtime/matlab_runtime.c).\n";
      if (Cpp) OS << "extern \"C\" {\n";
      OS << Buf.str();
      if (Cpp) OS << "} // extern \"C\"\n";
      OS << "\n";
    }
  }

  // Pass 1: llvm.mlir.global string constants. Skip globals that are dead
  // after IO substitution (their only users fed a substituted disp_str
  // callsite). Also skip the heading when no global survives — the
  // cleanest output for parfor-free programs leaves no trace of the
  // runtime's internal constants.
  {
    bool AnyLive = false;
    for (auto &Op : M.getBody()->getOperations()) {
      auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op);
      if (!G) continue;
      if (LiveGlobals.count(G.getSymName())) { AnyLive = true; break; }
    }
    if (AnyLive) {
      OS << "// Module-level string constants.\n";
      for (auto &Op : M.getBody()->getOperations()) {
        auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op);
        if (!G) continue;
        if (!LiveGlobals.count(G.getSymName())) continue;
        emitGlobal(G);
      }
      OS << "\n";
    }
  }

  // Pass 2: forward-declare every defined function so call ordering doesn't
  // matter. Reserve the function's symbol name so body-local identifiers
  // can't collide (important now that locals may inherit MATLAB names).
  // Buffered so the "// Forward declarations." heading only appears when
  // there's at least one non-main function to declare.
  {
    std::ostringstream Buf;
    for (auto &Op : M.getBody()->getOperations()) {
      if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
        if (F.getBody().empty()) continue;
        UsedNames.insert(F.getSymName().str());
        if (F.getSymName() == "main") continue;
        auto FT = F.getFunctionType();
        std::string RetTy = FT.getNumResults() == 0
                                ? std::string("void")
                                : cTypeOf(FT.getResult(0));
        Buf << "static " << RetTy << " " << F.getSymName().str() << "(";
        for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
          if (i) Buf << ", ";
          Buf << cTypeOf(FT.getInput(i));
        }
        if (FT.getNumInputs() == 0) Buf << "void";
        Buf << ");\n";
      } else if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op)) {
        if (F.getBody().empty()) continue;
        UsedNames.insert(F.getSymName().str());
        if (F.getSymName() == "main") continue;
        auto FT = F.getFunctionType();
        std::string RetTy =
            mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())
                ? std::string("void")
                : cTypeOf(FT.getReturnType());
        Buf << "static " << RetTy << " " << F.getSymName().str() << "(";
        for (unsigned i = 0; i < FT.getNumParams(); ++i) {
          if (i) Buf << ", ";
          Buf << cTypeOf(FT.getParamType(i));
        }
        if (FT.getNumParams() == 0) Buf << "void";
        Buf << ");\n";
      }
    }
    if (!Buf.str().empty()) {
      OS << "// Forward declarations.\n";
      OS << Buf.str();
      OS << "\n";
    }
  }

  /* Pass 2b (C++ only): reconstruct idiomatic class { ... }; blocks
   * from the class-method metadata the frontend stamped on each
   * func.func. The blocks are inline trampolines around the flat
   * ClassName__method functions — the class body itself holds a
   * single void* (matlab_obj*) and dispatches all methods through
   * the corresponding flat function. Users who want proper OOP
   * syntax in their handwritten glue code can then write
   * `c = a + b;` with full type-checking, without us needing to
   * rewrite the emitted main() body to match. */
  if (Cpp) {
    struct MethodInfo {
      std::string Mangled;  // flat function name
      std::string Name;     // user-visible method name
      std::string Kind;     // "ctor" / "method" / "static"
      mlir::func::FuncOp Fn;
    };
    std::unordered_map<std::string, std::vector<MethodInfo>> ByClass;
    std::unordered_map<std::string, std::string> SuperOf;
    std::vector<std::string> ClassOrder;
    for (auto &Op : M.getBody()->getOperations()) {
      auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op);
      if (!F) continue;
      auto ClsA = F->getAttrOfType<mlir::StringAttr>("matlab.class_name");
      auto NameA = F->getAttrOfType<mlir::StringAttr>("matlab.method_name");
      auto KindA = F->getAttrOfType<mlir::StringAttr>("matlab.method_kind");
      if (!ClsA || !NameA || !KindA) continue;
      std::string Cls = ClsA.getValue().str();
      if (!ByClass.count(Cls)) ClassOrder.push_back(Cls);
      MethodInfo MI{F.getSymName().str(), NameA.getValue().str(),
                    KindA.getValue().str(), F};
      ByClass[Cls].push_back(std::move(MI));
      if (auto SA = F->getAttrOfType<mlir::StringAttr>("matlab.class_super"))
        SuperOf[Cls] = SA.getValue().str();
    }

    auto opNameFor = [](llvm::StringRef N) -> llvm::StringRef {
      if (N == "plus")     return "operator+";
      if (N == "minus")    return "operator-";
      if (N == "mtimes")   return "operator*";
      if (N == "times")    return "operator*";
      if (N == "mrdivide") return "operator/";
      if (N == "rdivide")  return "operator/";
      if (N == "eq")       return "operator==";
      if (N == "ne")       return "operator!=";
      if (N == "lt")       return "operator<";
      if (N == "le")       return "operator<=";
      if (N == "gt")       return "operator>";
      if (N == "ge")       return "operator>=";
      return llvm::StringRef();
    };

    if (!ClassOrder.empty()) {
      OS << "// User-defined classes (wrappers over the flat ClassName__method "
            "functions).\n";
      for (const auto &Cls : ClassOrder) {
        OS << "class " << Cls;
        if (SuperOf.count(Cls)) OS << " : public " << SuperOf[Cls];
        OS << " {\npublic:\n";
        OS << "  void *_impl;\n";
        OS << "  explicit " << Cls << "(void *impl = nullptr) : _impl(impl) {}\n";
        for (auto &MI : ByClass[Cls]) {
          auto FT = MI.Fn.getFunctionType();
          if (MI.Kind == "ctor") {
            /* User-defined constructor: mirrors the flat ctor
             * signature, delegates to it for _impl. */
            OS << "  " << Cls << "(";
            for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
              if (i) OS << ", ";
              OS << cTypeOf(FT.getInput(i)) << " a" << i;
            }
            OS << ") : _impl(" << MI.Mangled << "(";
            for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
              if (i) OS << ", ";
              OS << "a" << i;
            }
            OS << ")) {}\n";
            continue;
          }
          if (MI.Kind == "static") {
            std::string RetTy = FT.getNumResults() == 0
                ? std::string("void")
                : cTypeOf(FT.getResult(0));
            OS << "  static " << RetTy << " " << MI.Name << "(";
            for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
              if (i) OS << ", ";
              OS << cTypeOf(FT.getInput(i)) << " a" << i;
            }
            OS << ") { "
               << (FT.getNumResults() == 0 ? "" : "return ")
               << MI.Mangled << "(";
            for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
              if (i) OS << ", ";
              OS << "a" << i;
            }
            OS << "); }\n";
            continue;
          }
          /* Instance method. First param is the receiver (_impl);
           * emit a member function taking the remaining inputs. */
          if (FT.getNumInputs() < 1) continue;
          llvm::StringRef OpName = opNameFor(MI.Name);
          /* Sanitise dots in the method name (get.Area -> get_Area)
           * so the emitted identifier is valid C++. */
          std::string SafeName = MI.Name;
          for (char &c : SafeName) if (c == '.') c = '_';
          std::string Target = OpName.empty() ? SafeName : OpName.str();
          std::string RetTyS = FT.getNumResults() == 0
              ? std::string("void")
              : cTypeOf(FT.getResult(0));
          /* Wrap a ptr-typed result back into the same class for
           * operator overloads — so `a + b` reads as a Vec2 at the
           * use site rather than a naked void*. */
          bool WrapResult = (!OpName.empty() &&
                              FT.getNumResults() == 1 &&
                              RetTyS == std::string("void*") &&
                              OpName != "operator==" &&
                              OpName != "operator!=" &&
                              OpName != "operator<" &&
                              OpName != "operator<=" &&
                              OpName != "operator>" &&
                              OpName != "operator>=");
          std::string EffRetTy = WrapResult ? Cls : RetTyS;
          OS << "  " << EffRetTy << " " << Target << "(";
          for (unsigned i = 1; i < FT.getNumInputs(); ++i) {
            if (i > 1) OS << ", ";
            mlir::Type T = FT.getInput(i);
            bool SecondIsSameClass = (!OpName.empty() && i == 1 &&
                                       cTypeOf(T) == std::string("void*"));
            if (SecondIsSameClass)
              OS << "const " << Cls << " &a" << i;
            else
              OS << cTypeOf(T) << " a" << i;
          }
          OS << ")";
          OS << " { ";
          if (FT.getNumResults() > 0) {
            if (WrapResult) OS << "return " << Cls << "(";
            else             OS << "return ";
          }
          OS << MI.Mangled << "(_impl";
          for (unsigned i = 1; i < FT.getNumInputs(); ++i) {
            OS << ", ";
            mlir::Type T = FT.getInput(i);
            bool SecondIsSameClass = (!OpName.empty() && i == 1 &&
                                       cTypeOf(T) == std::string("void*"));
            if (SecondIsSameClass) OS << "a" << i << "._impl";
            else                    OS << "a" << i;
          }
          OS << ")";
          if (WrapResult) OS << ")";
          OS << "; }\n";
        }
        OS << "};\n";
      }
      OS << "\n";
    }
  }

  // Pass 3: emit function bodies.
  for (auto &Op : M.getBody()->getOperations()) {
    if (Failed) break;
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      emitFuncFunc(F);
    } else if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      emitLLVMFunc(F);
    }
  }

  return !Failed;
}

void Emitter::emitFuncFunc(mlir::func::FuncOp F) {
  NextId = 0;  // Reset local SSA counter so each function restarts at v0.
  InlineExprs.clear();
  InlinedOps.clear();
  DirectSlots.clear();
  ForPatterns.clear();
  SuppressedOps.clear();
  FusedForSlots.clear();
  FusedForSlotName.clear();
  BreakFlagSlots.clear();
  ContinueFlagSlots.clear();
  FlagIfKind.clear();
  InlinedIfs.clear();
  // A new function is its own blank-line frame: don't let the previous
  // function's final line number influence whether we emit a blank above
  // the signature (the `}\n\n` end-of-function separator already handles it).
  LastLineFile.clear();
  LastLineNum = -1;
  PendingTrailingLine = -1;
  PendingTrailingText.clear();
  TrailingEmittedLines.clear();
  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());
  auto FT = F.getFunctionType();
  bool IsMain = F.getSymName() == "main";
  std::string RetTy;
  if (IsMain) {
    RetTy = "int";
  } else {
    RetTy = FT.getNumResults() == 0 ? std::string("void")
                                     : cTypeOf(FT.getResult(0));
  }
  emitLineDirective(F.getLoc(), 0);
  OS << (IsMain ? "" : "static ") << RetTy << " " << F.getSymName().str()
     << "(";
  auto &Entry = F.getBody().front();
  for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
    if (i) OS << ", ";
    auto Arg = Entry.getArgument(i);
    // Prefer the matlab.name arg-attr attached at AST->MLIR lowering so
    // the emitted signature mirrors the source (`fact(double n)` rather
    // than `fact(double v15)`). Fall back to a fresh v-counter when the
    // attr is missing (e.g. outlined parfor / anon bodies).
    std::string N;
    if (auto NA = F.getArgAttrOfType<mlir::StringAttr>(i, "matlab.name"))
      N = uniqueName(NA.getValue());
    else
      N = freshName();
    Names[Arg] = N;
    OS << cTypeOf(FT.getInput(i)) << " " << N;
  }
  if (FT.getNumInputs() == 0) OS << "void";
  OS << ") {\n";
  emitRegion(F.getBody(), 1);
  OS << "}\n\n";
}

void Emitter::emitLLVMFunc(mlir::LLVM::LLVMFuncOp F) {
  NextId = 0;
  InlineExprs.clear();
  InlinedOps.clear();
  DirectSlots.clear();
  DirectSlotDefer.clear();
  ForPatterns.clear();
  SuppressedOps.clear();
  FusedForSlots.clear();
  FusedForSlotName.clear();
  BreakFlagSlots.clear();
  ContinueFlagSlots.clear();
  FlagIfKind.clear();
  InlinedIfs.clear();
  LastLineFile.clear();
  LastLineNum = -1;
  PendingTrailingLine = -1;
  PendingTrailingText.clear();
  TrailingEmittedLines.clear();
  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());
  auto FT = F.getFunctionType();
  std::string RetTy =
      mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())
          ? std::string("void")
          : cTypeOf(FT.getReturnType());
  emitLineDirective(F.getLoc(), 0);
  OS << "static " << RetTy << " " << F.getSymName().str() << "(";
  auto &Entry = F.getBody().front();
  for (unsigned i = 0; i < FT.getNumParams(); ++i) {
    if (i) OS << ", ";
    auto Arg = Entry.getArgument(i);
    std::string N = freshName();
    Names[Arg] = N;
    OS << cTypeOf(FT.getParamType(i)) << " " << N;
  }
  if (FT.getNumParams() == 0) OS << "void";
  OS << ") {\n";
  emitRegion(F.getBody(), 1);
  OS << "}\n\n";
}

// ---------------------------------------------------------------------------
// Region / block / op dispatch
// ---------------------------------------------------------------------------

void Emitter::emitRegion(mlir::Region &R, int Indent) {
  // At the snapshot point every region is single-block (scf structural,
  // no cf.br in user code). Multi-block regions would need goto lowering.
  for (auto &B : R.getBlocks())
    emitBlock(B, Indent);
}

void Emitter::emitBlock(mlir::Block &B, int Indent) {
  AtBlockStart = true;
  for (auto &Op : B.getOperations())
    emitOp(Op, Indent);
  // Flush any trailing comment left pending on the last statement —
  // no subsequent line-directive call will cross its line boundary.
  flushPendingTrailing();
}

// ---------------------------------------------------------------------------
// Per-op emission. Initial version: stub that dumps the op mnemonic so we
// can see what survives into the snapshot. We'll replace case-by-case.
// ---------------------------------------------------------------------------

void Emitter::emitOp(mlir::Operation &Op, int Indent) {
  llvm::StringRef Name = Op.getName().getStringRef();

  // If the analysis decided to inline this op's result at its use site,
  // skip the declaration entirely. The consumer will substitute the
  // cached expression via exprFor().
  if (InlinedOps.count(&Op)) return;

  // Ops absorbed into a surrounding for-loop head (arith.addf, scf.yield)
  // or elided by slot fusion (iv-binding llvm.store, the slot's alloca).
  // The scf.while emitter has already accounted for them in the for-head
  // or for-increment string.
  if (SuppressedOps.count(&Op)) return;

  // Emit a #line directive if this op has a FileLineColLoc that differs
  // from the last directive we printed. Deduped inside emitLineDirective,
  // so constants / pure expression ops don't pollute the output. Skip
  // for llvm.alloca: the emitter hoists those to function-body top but
  // their location points to each variable's first use, which drags the
  // leading-comments scan forward and steals comments from real ops.
  if (!mlir::isa<mlir::LLVM::AllocaOp>(Op))
    emitLineDirective(Op.getLoc(), Indent);

  // --- llvm.mlir.zero / llvm.mlir.null -------------------------------
  if (mlir::isa<mlir::LLVM::ZeroOp>(Op)) {
    std::string N = this->name(Op.getResult(0));
    indent(Indent);
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = 0;\n";
    return;
  }

  // --- llvm.mlir.constant ---------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Op)) {
    std::string N = this->name(C.getResult());
    std::string Ty = cTypeOfValue(C.getResult());
    indent(Indent);
    OS << Ty << " " << N << " = ";
    auto V = C.getValue();
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V)) {
      OS << formatIntAttr(IA);
    } else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V)) {
      OS << formatFloatAttr(FA);
    } else {
      OS << "0 /* unknown const */";
    }
    OS << ";\n";
    return;
  }

  // --- arith.constant --------------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
    std::string N = this->name(C.getResult());
    std::string Ty = cTypeOfValue(C.getResult());
    indent(Indent);
    OS << Ty << " " << N << " = ";
    auto V = C.getValue();
    if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V)) {
      OS << formatFloatAttr(FA);
    } else if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V)) {
      OS << formatIntAttr(IA);
    } else {
      OS << "0 /* unknown const */";
    }
    OS << ";\n";
    return;
  }

  // --- func.return / llvm.return --------------------------------------
  if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
    indent(Indent);
    if (R.getNumOperands() == 0) OS << "return;\n";
    else OS << "return " << this->stmtExpr(R.getOperand(0)) << ";\n";
    return;
  }
  if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
    indent(Indent);
    if (R.getNumOperands() == 0) OS << "return;\n";
    else OS << "return " << this->stmtExpr(R.getOperand(0)) << ";\n";
    return;
  }

  // --- llvm.call / func.call ------------------------------------------
  if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    // IO substitution: when no parfor could race on stdout, rewrite
    // matlab_disp_str(literal) / matlab_disp_f64 / matlab_disp_i64 as
    // direct puts / printf / std::cout. Keeps the runtime call when
    // parfor is present (the mutex matters) or when the string arg
    // isn't a compile-time literal.
    if (tryEmitIOSubstitution(Call, Indent)) return;
    indent(Indent);
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult());
      std::string Ty = CppAuto ? "auto" : cTypeOfValue(Call.getResult());
      OS << Ty << " " << N << " = ";
    }
    if (auto Callee = Call.getCallee()) {
      OS << Callee->str() << "(";
      for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
        if (i) OS << ", ";
        OS << this->stmtExpr(Call.getOperand(i));
      }
      OS << ");\n";
    } else {
      // Indirect call: first operand is the function pointer, rest are args.
      // Cast the pointer to the correct function type built from the call's
      // operand and result signatures.
      std::string RetTy = Call.getNumResults() == 1
                              ? cTypeOfValue(Call.getResult())
                              : std::string("void");
      OS << "((" << RetTy << "(*)(";
      for (unsigned i = 1; i < Call.getNumOperands(); ++i) {
        if (i > 1) OS << ", ";
        OS << cTypeOfValue(Call.getOperand(i));
      }
      if (Call.getNumOperands() == 1) OS << "void";
      OS << "))" << this->exprFor(Call.getOperand(0)) << ")(";
      for (unsigned i = 1; i < Call.getNumOperands(); ++i) {
        if (i > 1) OS << ", ";
        OS << this->stmtExpr(Call.getOperand(i));
      }
      OS << ");\n";
    }
    return;
  }
  if (auto Call = mlir::dyn_cast<mlir::func::CallOp>(Op)) {
    indent(Indent);
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult(0));
      std::string Ty = CppAuto ? "auto" : cTypeOfValue(Call.getResult(0));
      OS << Ty << " " << N << " = ";
    }
    OS << Call.getCallee().str() << "(";
    for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
      if (i) OS << ", ";
      OS << this->stmtExpr(Call.getOperand(i));
    }
    OS << ");\n";
    return;
  }

  // --- llvm.mlir.addressof --------------------------------------------
  if (auto A = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(Op)) {
    std::string N = this->name(A.getResult());
    indent(Indent);
    OS << "void* " << N << " = (void*)" << A.getGlobalName().str() << ";\n";
    return;
  }

  // --- arith binary ops on floats -------------------------------------
  auto emitBinF = [&](const char *CC) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = "
       << this->exprFor(Op.getOperand(0)) << " " << CC << " "
       << this->exprFor(Op.getOperand(1)) << ";\n";
  };
  if (mlir::isa<mlir::arith::AddFOp>(Op)) { emitBinF("+"); return; }
  if (mlir::isa<mlir::arith::SubFOp>(Op)) { emitBinF("-"); return; }
  if (mlir::isa<mlir::arith::MulFOp>(Op)) { emitBinF("*"); return; }
  if (mlir::isa<mlir::arith::DivFOp>(Op)) { emitBinF("/"); return; }
  if (mlir::isa<mlir::arith::AddIOp>(Op)) { emitBinF("+"); return; }
  if (mlir::isa<mlir::arith::SubIOp>(Op)) { emitBinF("-"); return; }
  if (mlir::isa<mlir::arith::MulIOp>(Op)) { emitBinF("*"); return; }

  // --- arith.cmpf / cmpi ----------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::arith::CmpFOp>(Op)) {
    const char *CC = "==";
    switch (C.getPredicate()) {
      case mlir::arith::CmpFPredicate::OEQ:
      case mlir::arith::CmpFPredicate::UEQ: CC = "=="; break;
      case mlir::arith::CmpFPredicate::ONE:
      case mlir::arith::CmpFPredicate::UNE: CC = "!="; break;
      case mlir::arith::CmpFPredicate::OLT:
      case mlir::arith::CmpFPredicate::ULT: CC = "<"; break;
      case mlir::arith::CmpFPredicate::OLE:
      case mlir::arith::CmpFPredicate::ULE: CC = "<="; break;
      case mlir::arith::CmpFPredicate::OGT:
      case mlir::arith::CmpFPredicate::UGT: CC = ">"; break;
      case mlir::arith::CmpFPredicate::OGE:
      case mlir::arith::CmpFPredicate::UGE: CC = ">="; break;
      default: break;
    }
    indent(Indent);
    std::string N = this->name(C.getResult());
    OS << "bool " << N << " = " << this->exprFor(C.getLhs()) << " " << CC
       << " " << this->exprFor(C.getRhs()) << ";\n";
    return;
  }
  if (auto C = mlir::dyn_cast<mlir::arith::CmpIOp>(Op)) {
    const char *CC = "==";
    switch (C.getPredicate()) {
      case mlir::arith::CmpIPredicate::eq:  CC = "=="; break;
      case mlir::arith::CmpIPredicate::ne:  CC = "!="; break;
      case mlir::arith::CmpIPredicate::slt:
      case mlir::arith::CmpIPredicate::ult: CC = "<"; break;
      case mlir::arith::CmpIPredicate::sle:
      case mlir::arith::CmpIPredicate::ule: CC = "<="; break;
      case mlir::arith::CmpIPredicate::sgt:
      case mlir::arith::CmpIPredicate::ugt: CC = ">"; break;
      case mlir::arith::CmpIPredicate::sge:
      case mlir::arith::CmpIPredicate::uge: CC = ">="; break;
    }
    indent(Indent);
    std::string N = this->name(C.getResult());
    OS << "bool " << N << " = " << this->exprFor(C.getLhs()) << " " << CC
       << " " << this->exprFor(C.getRhs()) << ";\n";
    return;
  }

  // --- arith casts ----------------------------------------------------
  if (mlir::isa<mlir::arith::SIToFPOp, mlir::arith::UIToFPOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = (double)"
       << this->exprFor(Op.getOperand(0)) << ";\n";
    return;
  }
  if (mlir::isa<mlir::arith::FPToSIOp, mlir::arith::FPToUIOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = ("
       << cTypeOfValue(Op.getResult(0)) << ")"
       << this->exprFor(Op.getOperand(0)) << ";\n";
    return;
  }
  if (mlir::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                mlir::arith::TruncIOp, mlir::arith::TruncFOp,
                mlir::arith::ExtFOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = ("
       << cTypeOfValue(Op.getResult(0)) << ")"
       << this->exprFor(Op.getOperand(0)) << ";\n";
    return;
  }

  // --- llvm.alloca / load / store -------------------------------------
  if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Op)) {
    // Loop-var slot absorbed by a fused for-loop: no `double i = 0;` at
    // function scope — the for-loop's init clause declares the IV. We
    // still register the slot under DirectSlots so in-body loads of the
    // slot expand to the IV's name, and mark the value as direct so
    // stores/loads short-circuit the pointer trampoline.
    if (FusedForSlots.count(A.getOperation())) {
      const std::string &N = FusedForSlotName[A.getOperation()];
      Names[A.getResult()] = N;
      DirectSlots[A.getOperation()] = N;
      return;
    }
    // If LowerScalarSlots / LowerTensorOps propagated the original
    // matlab.alloc `name` attribute, use it as the slot identifier so the
    // emitted C mirrors the MATLAB source (total_slot rather than v3_slot).
    std::string Hint;
    if (auto NA = A->getAttrOfType<mlir::StringAttr>("matlab.name"))
      Hint = NA.getValue().str();
    std::string N, SlotName;
    if (!Hint.empty()) {
      // Common collision: a MATLAB param `n` is spilled into a slot named
      // `n`, but the func arg already claimed `n`. Prefer "<hint>_slot"
      // over the numeric "_2" suffix uniqueName would otherwise produce.
      std::string Sane = sanitizeIdent(Hint);
      if (UsedNames.find(Sane) != UsedNames.end())
        SlotName = uniqueName(Sane + "_slot");
      else
        SlotName = uniqueName(Sane);
    } else {
      SlotName = uniqueName("slot");
    }
    // Two shapes appear:
    //   1) alloca<T> with ArraySize=1 — a scalar slot (LowerScalarSlots).
    //   2) alloca<!llvm.array<N x T>> with ArraySize=1 — a contiguous buffer
    //      for a matrix literal (LowerTensorOps::materializeMat).
    mlir::Type ET = A.getElemType();
    bool IsArray = mlir::isa<mlir::LLVM::LLVMArrayType>(ET);

    // Scan uses: if every consumer is a plain llvm.load / llvm.store that
    // takes the alloca as its address (not the stored value), we can skip
    // the `void* N = &slot;` trampoline and have stores/loads address the
    // slot by name directly. Arrays still need the pointer form because
    // GEPs on them index through it.
    bool DirectSlot = !IsArray;
    for (mlir::OpOperand &Use : A->getUses()) {
      mlir::Operation *U = Use.getOwner();
      if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(U)) {
        if (L.getAddr() != A.getResult()) { DirectSlot = false; break; }
        continue;
      }
      if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(U)) {
        if (S.getAddr() != A.getResult()) { DirectSlot = false; break; }
        continue;
      }
      DirectSlot = false;
      break;
    }

    if (IsArray) {
      auto AT = mlir::cast<mlir::LLVM::LLVMArrayType>(ET);
      std::string ElTy = cTypeOf(AT.getElementType());
      uint64_t N0 = AT.getNumElements();
      N = uniqueName(SlotName + "_p");
      Names[A.getResult()] = N;
      indent(Indent);
      OS << ElTy << " " << SlotName << "[" << N0 << "] = {0};\n";
      indent(Indent);
      OS << "void* " << N << " = (void*)" << SlotName << ";\n";
      return;
    }

    std::string ElTy = cTypeOf(ET);
    if (DirectSlot) {
      // No pointer variable. Bind the alloca's result SSA value to the
      // slot's name — downstream code checks DirectSlots before treating
      // it as a real pointer.
      Names[A.getResult()] = SlotName;
      DirectSlots[A.getOperation()] = SlotName;

      // If the first user in this block is a StoreOp and we see no load
      // of the slot before that store, defer the declaration so the store
      // can emit `T slot = val;` in one line. Works only for same-block
      // first-write ordering — stores in nested scf regions can't hoist
      // the declaration out of their parent block.
      bool DeferDecl = false;
      mlir::Block *ABlock = A->getBlock();
      for (auto It = mlir::Block::iterator(A->getNextNode());
           It != ABlock->end(); ++It) {
        mlir::Operation *Use = nullptr;
        bool Relevant = false;
        for (mlir::OpOperand &U : A.getResult().getUses()) {
          if (U.getOwner() == &*It) { Use = U.getOwner(); Relevant = true; break; }
        }
        if (!Relevant) continue;
        DeferDecl = mlir::isa<mlir::LLVM::StoreOp>(Use);
        break;
      }
      if (DeferDecl) {
        DirectSlotDefer.insert(A.getOperation());
        return;  // the store will emit `T slot = val;`
      }

      indent(Indent);
      OS << ElTy << " " << SlotName << " = 0;\n";
      return;
    }
    N = uniqueName(SlotName + "_p");
    Names[A.getResult()] = N;
    indent(Indent);
    OS << ElTy << " " << SlotName << " = 0;\n";
    indent(Indent);
    OS << "void* " << N << " = (void*)&" << SlotName << ";\n";
    return;
  }

  // --- llvm.getelementptr ---------------------------------------------
  if (auto G = mlir::dyn_cast<mlir::LLVM::GEPOp>(Op)) {
    std::string N = this->name(G.getResult());
    std::string ElTy = cTypeOf(G.getElemType());
    indent(Indent);
    // Flatten every index into a single pointer offset on the declared
    // element type. This matches the LLVM semantics for typed GEP when the
    // source is a plain buffer. For the matrix-literal case we only ever
    // see a single i64 index, so this is tight.
    OS << "void* " << N << " = (void*)(((" << ElTy << "*)"
       << this->exprFor(G.getBase()) << ")";
    // Inline constant indices; SSA indices use their current expression.
    for (auto Idx : G.getIndices()) {
      if (auto V = llvm::dyn_cast<mlir::Value>(Idx)) {
        OS << " + " << this->exprFor(V);
      } else if (auto A = llvm::dyn_cast<mlir::IntegerAttr>(Idx)) {
        OS << " + " << A.getInt();
      }
    }
    OS << ");\n";
    return;
  }
  if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
    // Direct-slot fast path: the address is an alloca we've optimized out
    // of the pointer world. The load becomes a plain read of the slot's
    // identifier — `double n_v = n;` — so subsequent expressions look
    // like hand-written C.
    mlir::Operation *AddrDef = L.getAddr().getDefiningOp();
    bool IsDirect = AddrDef && DirectSlots.count(AddrDef);
    std::string N;
    if (auto *D = AddrDef) {
      if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(D)) {
        if (auto NA = A->getAttrOfType<mlir::StringAttr>("matlab.name")) {
          N = uniqueName(NA.getValue().str() + "_v");
          Names[L.getResult()] = N;
        }
      }
    }
    if (N.empty()) N = this->name(L.getResult());
    indent(Indent);
    std::string ResTy = cTypeOfValue(L.getResult());
    if (IsDirect) {
      OS << ResTy << " " << N << " = " << DirectSlots[AddrDef] << ";\n";
    } else {
      OS << ResTy << " " << N << " = *(" << ResTy << "*)"
         << this->exprFor(L.getAddr()) << ";\n";
    }
    return;
  }
  if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
    mlir::Operation *AddrDef = S.getAddr().getDefiningOp();
    bool IsDirect = AddrDef && DirectSlots.count(AddrDef);
    indent(Indent);
    if (IsDirect) {
      // Merge the deferred slot declaration with its first store so the
      // generated C reads `double y = a + b;` rather than `double y = 0;`
      // followed later by `y = a + b;`.
      if (DirectSlotDefer.erase(AddrDef)) {
        std::string Ty = cTypeOfValue(S.getValue());
        OS << Ty << " " << DirectSlots[AddrDef] << " = "
           << this->stmtExpr(S.getValue()) << ";\n";
      } else {
        OS << DirectSlots[AddrDef] << " = "
           << this->stmtExpr(S.getValue()) << ";\n";
      }
    } else {
      std::string Ty = cTypeOfValue(S.getValue());
      OS << "*(" << Ty << "*)" << this->exprFor(S.getAddr()) << " = "
         << this->stmtExpr(S.getValue()) << ";\n";
    }
    return;
  }

  // --- scf.if ---------------------------------------------------------
  if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Op)) {
    // Break / continue short-circuit: `scf.if cond { flag := true }` came
    // from a MATLAB `if cond; break; end`. Re-emit as a one-liner.
    auto FK = FlagIfKind.find(&Op);
    if (FK != FlagIfKind.end()) {
      indent(Indent);
      OS << "if (" << this->stmtExpr(If.getCondition()) << ") " << FK->second
         << ";\n";
      return;
    }
    // Pure flag-guard wrapping real body: skip the `if (...)` and emit
    // the then-region inline at the parent's indent.
    if (InlinedIfs.count(&Op)) {
      emitRegion(If.getThenRegion(), Indent);
      return;
    }
    // Declare result locals (one per scf.if result) so yield can assign
    // to them and uses outside the if can reference the names directly.
    for (unsigned i = 0; i < If.getNumResults(); ++i) {
      std::string N = this->name(If.getResult(i));
      indent(Indent);
      OS << cTypeOfValue(If.getResult(i)) << " " << N << " = 0;\n";
    }
    indent(Indent);
    OS << "if (" << this->stmtExpr(If.getCondition()) << ") {\n";
    emitRegion(If.getThenRegion(), Indent + 1);
    // Skip an empty-or-trivial else branch when the if carries no result
    // locals (`if cond; ...; end` from MATLAB always materialises an else
    // block containing only an scf.yield — printing `} else { }` is noise).
    bool ElseIsTrivial = true;
    if (!If.getElseRegion().empty()) {
      for (auto &Blk : If.getElseRegion().getBlocks()) {
        for (auto &Inner : Blk.getOperations()) {
          if (mlir::isa<mlir::scf::YieldOp>(Inner)) continue;
          if (InlinedOps.count(&Inner)) continue;
          if (SuppressedOps.count(&Inner)) continue;
          ElseIsTrivial = false;
          break;
        }
        if (!ElseIsTrivial) break;
      }
    }
    bool HasResults = If.getNumResults() > 0;
    if (!If.getElseRegion().empty() && (HasResults || !ElseIsTrivial)) {
      indent(Indent);
      OS << "} else {\n";
      emitRegion(If.getElseRegion(), Indent + 1);
    }
    indent(Indent);
    OS << "}\n";
    return;
  }

  // scf.yield inside an scf.if: assign yielded values to the parent's
  // result slots (same names already allocated above).
  if (auto Y = mlir::dyn_cast<mlir::scf::YieldOp>(Op)) {
    auto *Parent = Op.getParentOp();
    if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Parent)) {
      for (unsigned i = 0; i < Y.getNumOperands(); ++i) {
        indent(Indent);
        OS << this->name(If.getResult(i)) << " = "
           << this->stmtExpr(Y.getOperand(i)) << ";\n";
      }
      return;
    }
    if (auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Parent)) {
      // After-region yield: update iter locals (shared names with iter-args).
      for (unsigned i = 0; i < Y.getNumOperands(); ++i) {
        auto BA = W.getBefore().front().getArgument(i);
        indent(Indent);
        OS << this->name(BA) << " = " << this->stmtExpr(Y.getOperand(i))
           << ";\n";
      }
      return;
    }
    // Unknown parent — just drop.
    return;
  }

  // --- scf.while ------------------------------------------------------
  if (auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Op)) {
    auto &Before = W.getBefore().front();
    auto &After = W.getAfter().front();

    // Did the pre-scan flag this loop as a canonical MATLAB for-loop?
    // When so, take the for-head shortcut: declare the IV in the `for(...)`
    // init clause, fold the trailing arith.addf into the increment, and
    // drop the redundant "i = v0;" bind store when slot fusion applies.
    auto FPIt = ForPatterns.find(W.getOperation());
    if (FPIt != ForPatterns.end()) {
      const ForLoopInfo &Info = FPIt->second;
      // Bind the IV name both where the condition is computed (before) and
      // where it's consumed (after). The before-arg never actually appears
      // in the output because the cmpf is inlined into the for-head, but
      // routing its name avoids freshName() collisions downstream.
      Names[Before.getArgument(0)] = Info.IvName;
      InlineExprs[After.getArgument(0)] = Info.IvName;
      // The scf.while's outer result (rarely read by the MATLAB frontend
      // when a for-loop is lowered) mirrors the IV value at exit.
      if (W.getNumResults() == 1)
        InlineExprs[W.getResult(0)] = Info.IvName;

      // Normalise the step to detect the `i -= K` polish for negative
      // literal steps (5:-1:1 reads nicer as `k -= 1.0` than `k += -1.0`).
      bool StepIsNegLit = false;
      std::string StepExpr;
      if (auto *Def = Info.Step.getDefiningOp()) {
        mlir::FloatAttr FA;
        if (auto CA = mlir::dyn_cast<mlir::arith::ConstantOp>(Def))
          FA = mlir::dyn_cast<mlir::FloatAttr>(CA.getValue());
        else if (auto CL = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Def))
          FA = mlir::dyn_cast<mlir::FloatAttr>(CL.getValue());
        if (FA && FA.getValueAsDouble() < 0.0) {
          StepIsNegLit = true;
          auto NegBuilder = mlir::Builder(W.getContext());
          StepExpr = formatFloatAttr(
              NegBuilder.getF64FloatAttr(-FA.getValueAsDouble()));
        }
      }
      if (!StepIsNegLit)
        StepExpr = this->stmtExpr(Info.Step);

      const char *CmpStr = Info.IsDecreasing ? ">=" : "<=";
      const char *StepOp = StepIsNegLit ? "-=" : "+=";
      indent(Indent);
      OS << "for (" << cTypeOf(Before.getArgument(0).getType())
         << " " << Info.IvName << " = " << this->stmtExpr(Info.Init)
         << "; " << Info.IvName << " " << CmpStr << " "
         << this->stmtExpr(Info.End) << "; " << Info.IvName << " "
         << StepOp << " " << StepExpr << ") {\n";
      for (auto &Inner : After.getOperations())
        emitOp(Inner, Indent + 1);
      indent(Indent);
      OS << "}\n";
      return;
    }

    // Declare one mutable local per iter-arg, initialized from the while
    // operand, and bind the before-block arg to that name so references
    // inside the before region resolve to it.
    for (unsigned i = 0; i < W.getInits().size(); ++i) {
      auto BA = Before.getArgument(i);
      std::string N = freshName();
      Names[BA] = N;
      indent(Indent);
      OS << cTypeOf(BA.getType()) << " " << N << " = "
         << this->stmtExpr(W.getInits()[i]) << ";\n";
    }
    // Result locals for scf.while's outer results: mirror iter-arg names
    // (same storage is used on exit). Bind result SSA values to the same
    // names so callers downstream see the right identifiers.
    for (unsigned i = 0; i < W.getNumResults(); ++i) {
      auto BA = Before.getArgument(i);
      Names[W.getResult(i)] = Names[BA];
    }

    // If the before-region consists only of the scf.condition terminator
    // (and possibly some inlined-only producers whose output is absorbed
    // at the condition-use site), we can emit the natural `while (cond)`
    // loop shape rather than the `while (1) { ...; if (!cond) break; }`
    // intermediate form. Any non-inlined work in the before-region must
    // run every iteration and cannot be hoisted above the loop, so we
    // fall back to the intermediate form when such work exists.
    bool BeforeIsCondOnly = true;
    for (auto &Inner : Before.getOperations()) {
      if (mlir::isa<mlir::scf::ConditionOp>(Inner)) continue;
      if (InlinedOps.count(&Inner)) continue;
      BeforeIsCondOnly = false;
      break;
    }

    // Loop-condition rendering: strip break/continue-flag inversions so
    // a user-source `while cond` with a break inside doesn't emit as
    // `while (cond && !did_break)` — the flag state is gone once we
    // re-lower the store into a `break;` statement. An entirely
    // stripped condition collapses to `1` (true).
    auto emitStrippedCond = [&](mlir::Value V) -> std::string {
      llvm::SmallVector<mlir::Value, 2> Parts;
      gatherNonFlagConjuncts(V, Parts);
      if (Parts.empty()) return "1";
      std::string Out;
      for (unsigned i = 0; i < Parts.size(); ++i) {
        if (i) Out += " && ";
        Out += this->stmtExpr(Parts[i]);
      }
      return Out;
    };

    if (BeforeIsCondOnly) {
      auto Cond = mlir::cast<mlir::scf::ConditionOp>(Before.getTerminator());
      // Bind after-block args to the forwarded values' current expression
      // BEFORE emitting the condition — downstream reads resolve via
      // InlineExprs. (The value name is resolved once here; body reads
      // pick up the bound expression consistently.)
      for (unsigned i = 0; i < Cond.getArgs().size(); ++i) {
        auto AA = After.getArgument(i);
        InlineExprs[AA] = this->exprFor(Cond.getArgs()[i]);
      }
      indent(Indent);
      OS << "while (" << emitStrippedCond(Cond.getCondition()) << ") {\n";
      for (auto &Inner : After.getOperations())
        emitOp(Inner, Indent + 1);
      indent(Indent);
      OS << "}\n";
      return;
    }

    indent(Indent);
    OS << "while (1) {\n";

    // Emit the before region body, then the scf.condition terminator as
    // `if (!cond) break;`, forwarding values into the after-block args.
    for (auto &Inner : Before.getOperations()) {
      if (auto Cond = mlir::dyn_cast<mlir::scf::ConditionOp>(Inner)) {
        indent(Indent + 1);
        OS << "if (!(" << emitStrippedCond(Cond.getCondition())
           << ")) break;\n";
        // Bind after-block args to the forwarded values' current expression.
        // Using InlineExprs here means if the forwarded value was inlined,
        // the after-block transparently re-expands to the expression string;
        // if it was declared, we re-use the name.
        for (unsigned i = 0; i < Cond.getArgs().size(); ++i) {
          auto AA = After.getArgument(i);
          InlineExprs[AA] = this->exprFor(Cond.getArgs()[i]);
        }
        continue;
      }
      emitOp(Inner, Indent + 1);
    }
    // Emit after region ops (block args already bound above).
    for (auto &Inner : After.getOperations())
      emitOp(Inner, Indent + 1);

    indent(Indent);
    OS << "}\n";
    return;
  }

  // --- arith.select ---------------------------------------------------
  if (auto S = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
    indent(Indent);
    std::string N = this->name(S.getResult());
    OS << cTypeOfValue(S.getResult()) << " " << N << " = "
       << this->exprFor(S.getCondition()) << " ? "
       << this->exprFor(S.getTrueValue()) << " : "
       << this->exprFor(S.getFalseValue()) << ";\n";
    return;
  }

  // --- arith bitwise / logical ops on integers / i1 -------------------
  if (auto AI = mlir::dyn_cast<mlir::arith::AndIOp>(Op)) {
    emitBinF(AI.getResult().getType().isInteger(1) ? "&&" : "&");
    return;
  }
  if (auto OI = mlir::dyn_cast<mlir::arith::OrIOp>(Op)) {
    emitBinF(OI.getResult().getType().isInteger(1) ? "||" : "|");
    return;
  }
  if (auto XI = mlir::dyn_cast<mlir::arith::XOrIOp>(Op)) {
    if (XI.getResult().getType().isInteger(1)) {
      // Same special-case as the inline path: `x ^ true` prints as `!x`
      // so loop-break predicates read naturally.
      auto isOne = [](mlir::Value V) {
        auto *D = V.getDefiningOp();
        if (!D) return false;
        if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D))
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
            return IA.getValue().getZExtValue() == 1;
        if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D))
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
            return IA.getValue().getZExtValue() == 1;
        return false;
      };
      mlir::Value Other;
      if (isOne(Op.getOperand(1))) Other = Op.getOperand(0);
      else if (isOne(Op.getOperand(0))) Other = Op.getOperand(1);
      if (Other) {
        indent(Indent);
        std::string N = this->name(Op.getResult(0));
        OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = !"
           << this->exprFor(Other) << ";\n";
        return;
      }
      emitBinF("!=");  // generic i1 XOR is logical inequality
      return;
    }
    emitBinF("^");
    return;
  }

  // --- Fallback: unknown op — refuse to emit rather than silently drop it.
  // A silent drop is dangerous for zero-result side-effect ops (the program
  // would compile but produce wrong output). For ops with results, the
  // downstream "undeclared identifier" error was our only signal; now
  // we surface the root cause at emit time with the MLIR op name.
  indent(Indent);
  OS << "/* UNSUPPORTED: " << Name.str() << " */\n";
  fail(("unsupported op in emitter: " + Name).str());
}

} // namespace

std::string emitC(mlir::ModuleOp M, bool Cpp, bool NoLine, bool Doxygen,
                  bool CppAuto, const matlab::SourceManager *SM) {
  std::ostringstream OSS;
  Emitter E(OSS, Cpp, NoLine, Doxygen, CppAuto, SM);
  if (!E.run(M)) return {};
  return OSS.str();
}

} // namespace mlirgen
} // namespace matlab
