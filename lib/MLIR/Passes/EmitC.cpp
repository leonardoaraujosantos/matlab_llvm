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
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

/// One MATLAB classdef and its discovered scalar properties. The C++
/// emitter lifts these to a real C++ class with named member fields and
/// translated method bodies — so `acc.Balance` is a direct field access
/// rather than a `matlab_obj_get_f64(acc, "Balance", 7)` runtime call.
struct CppClassDef {
  std::string Super;
  // Property names in encounter order (deduped). Currently every
  // property is `double`-typed.
  std::vector<std::string> Properties;
  std::vector<mlir::func::FuncOp> Ctors;
  std::vector<mlir::func::FuncOp> Methods;  // non-ctor (regular / get / op / static)
};

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
  std::string cTypeOfValue(mlir::Value V) {
    if (Cpp && isMatrixValue(V)) return "Matrix";
    return cTypeOf(V.getType());
  }

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
  void indent(int N) {
    flushPendingLine();
    for (int i = 0; i < N; ++i) OS << "  ";
  }
  std::string constStr(mlir::LLVM::GlobalOp G);
  void fail(llvm::StringRef Msg) {
    if (!Failed)
      std::cerr << "error: emit-c: " << Msg.str() << "\n";
    Failed = true;
  }
  void emitLineDirective(mlir::Location L, int Indent);
  // Write any pending `#line` directive to OS. Called by indent() so a
  // content line always flushes the line directive that precedes it.
  void flushPendingLine();
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

  // ----- Classdef → real C++ class translation --------------------------
  // Walk the module, group classdef methods by `matlab.class_name`,
  // discover scalar properties from `obj_set_f64` / `obj_get_f64` calls
  // in method bodies. Populates Classes and ClassMethodFuncs.
  void collectClassdefs(mlir::ModuleOp M);
  // Walk every SSA value in the module and tag it with a class type
  // when the value originates from a class-ctor call result, an alloca
  // that received a class-typed store, or a load from a tracked alloca.
  // Iterated to a fixpoint so transitive store→load propagation lands.
  void populateClassValueTypes(mlir::ModuleOp M);
  // Emit one classdef as a real C++ class block with named member
  // fields, default+parameterised constructors, and translated method
  // bodies that use `Field` (bare) instead of runtime hash calls.
  void emitCppClass(llvm::StringRef ClassName, const CppClassDef &CD);
  // Emit a single class method (ctor / regular / get / operator /
  // static) inside the open class block at indent 1.
  void emitCppMethod(mlir::func::FuncOp F, llvm::StringRef ClassName);
  // Properties of a class plus everything inherited up the super chain.
  llvm::StringSet<> inheritedProperties(llvm::StringRef ClassName) const;
  // Rewrite `matlab_obj_get_f64(self, "X", _)` and
  // `matlab_obj_set_f64(self, "X", _, v)` to direct member access when
  // the receiver is the active class-method's `this` or an SSA value
  // tagged with a class type.
  bool tryRewriteObjGet(mlir::LLVM::CallOp C, std::string &Out);
  bool tryRewriteObjSet(mlir::LLVM::CallOp C, std::string &Out);
  // Rewrite a call to a classdef method as the matching C++ syntax —
  // `Class(args)`, `obj.method(args)`, `a == b`, `obj.X`, `Class::method(args)`.
  bool tryRewriteAsClassCall(llvm::StringRef Callee,
                              mlir::ValueRange Operands,
                              std::string &Out);
  // Look up the C++ class type assigned to V (from a ctor result, an
  // alloca-tracked load, etc.), or empty if V isn't a class instance.
  llvm::StringRef classTypeOf(mlir::Value V) const;

  // Same fixpoint propagation as populateClassValueTypes, but for runtime
  // matrix values. Marks SSA values whose defining call returns
  // `matlab_mat*` and propagates through alloca / store / load and
  // through user-function args/results.
  void populateMatrixValueTypes(mlir::ModuleOp M);
  bool isMatrixReturningRuntimeFn(llvm::StringRef Name) const;
  bool isMatrixValue(mlir::Value V) const { return MatrixValues.count(V); }
  // Try to rewrite a matlab_* runtime call as a `Matrix` operator/method
  // expression. Returns true if Out was filled.
  bool tryRewriteAsMatrixCall(llvm::StringRef Callee,
                               mlir::ValueRange Operands,
                               std::string &Out);

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
  // Snapshot of UsedNames after the prolog has finished registering
  // module-scope symbols (function names, runtime externs, globals).
  // emitFuncFunc / emitLLVMFunc reset UsedNames to this snapshot so
  // every function body starts with a clean local namespace; otherwise
  // a parameter `x` claimed by `mySq` would force `myCube` to use `x_2`.
  llvm::StringSet<> UsedNamesAfterProlog;
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
  // A `#line N "file"` is buffered as pending and only written when the
  // next real content arrives (via indent() or an explicit flush). If a
  // second emitLineDirective fires before the pending one is flushed,
  // the new line replaces the old one — collapses the doubled-#line
  // pattern that classdef ctors used to produce when a static-folded
  // `if nargin == N` left an empty branch behind.
  bool PendingLineActive = false;
  int PendingLineLine = 0;
  std::string PendingLineFile;
  int PendingLineIndent = 0;
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

  // ----- Classdef → real C++ class translation --------------------------
  // Populated by collectClassdefs() before any function emission and
  // consumed by the class-block emit + every call-site rewrite.
  llvm::StringMap<CppClassDef> Classes;
  // Set of flat-function symbol names ("BankAccount__deposit") whose
  // bodies are now emitted inside a class block. The free-function
  // emission pass skips these so we don't get duplicate definitions.
  llvm::StringSet<> ClassMethodFuncs;
  // Active class-method context — set while emitting a class method body.
  // Drives obj_get/set_f64 → bare-field-name rewriting in emitOp.
  bool InClassMethodBody = false;
  mlir::Value ClassMethodSelf;
  std::string ClassMethodClassName;
  // SSA values known to hold instances of a particular class. Set when
  // we see a ctor call's result; propagated through alloca / store /
  // load chains. Drives variable-type emission and call-site rewrites.
  llvm::DenseMap<mlir::Value, std::string> ClassValueType;
  llvm::DenseMap<mlir::Operation *, std::string> ClassAllocaType;
  // SSA values known to hold a `matlab_mat *` (a runtime matrix) — drives
  // C++ type emission as `Matrix` and the operator/method rewrites that
  // turn `matlab_matmul_mm(A, B)` back into `A * B`. Only used in Cpp
  // mode; the C lane keeps `void*`.
  llvm::DenseSet<mlir::Value> MatrixValues;
  llvm::DenseSet<mlir::Operation *> MatrixAllocas;
  // True when at least one runtime call surfaced a Matrix-typed value —
  // gates emission of the `#include "matlab_runtime.hpp"` line and the
  // suppression of the per-runtime-fn extern "C" block.
  bool UsesMatrixWrapper = false;
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
  // Prefer the user-source variable name carried through SlotPromotion +
  // LowerTensorOps + Mem2RegLite (any of which set `matlab.name` on the
  // defining op of a value bound to a named slot). Falling back to a
  // fresh `vN` only when no hint is available keeps the generated C/C++
  // readable: `void* A = matlab_mat_from_buf(...)` instead of
  // `void* v0 = matlab_mat_from_buf(...)`.
  if (auto *D = V.getDefiningOp()) {
    if (auto NA = D->getAttrOfType<mlir::StringAttr>("matlab.name")) {
      std::string N = uniqueName(NA.getValue());
      Names[V] = N;
      return N;
    }
  }
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

  // Is this LLVM call to a runtime helper that's known to be a pure
  // read with no side effects? Such calls don't block inlining of an
  // earlier value across them. The list mirrors EmitPython's allowlist
  // and is intentionally small — adding the wrong helper here would
  // visibly reorder side effects in the generated source.
  auto isPureReadCall = [](Operation &Op2) -> bool {
    auto C = dyn_cast<LLVM::CallOp>(Op2);
    if (!C || !C.getCallee()) return false;
    StringRef N = *C.getCallee();
    if (!N.starts_with("matlab_")) return false;
    StringRef S = N.drop_front(strlen("matlab_"));
    return S == "obj_get_f64" || S == "size" || S == "size_dim" ||
           S == "numel" || S == "numel3" || S == "length" ||
           S == "ndims" || S == "isempty" || S == "isnumeric" ||
           S == "isscalar" || S == "ismatrix" || S == "isvector" ||
           S == "isstruct" || S == "isfield" || S == "iscell" ||
           S == "isstring" || S == "string_len";
  };

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
      if (isa<func::CallOp>(&*It)) return false;
      if (isa<LLVM::CallOp>(&*It)) {
        if (!isPureReadCall(*It)) return false;
      }
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
      if (isa<func::CallOp>(*It)) return false;
      if (isa<LLVM::CallOp>(*It)) {
        if (!isPureReadCall(*It)) return false;
      }
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
      if (isa<func::CallOp>(*It)) return false;
      if (isa<LLVM::CallOp>(*It)) {
        if (!isPureReadCall(*It)) return false;
      }
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
    if (tryRewriteAsClassCall(C.getCallee(), C.getOperands(), Expr))
      return true;
    if (tryRewriteAsMatrixCall(C.getCallee(), C.getOperands(), Expr))
      return true;
    std::string E = C.getCallee().str() + "(";
    for (unsigned i = 0; i < C.getNumOperands(); ++i) {
      if (i) E += ", ";
      E += dropOuterParens(exprFor(C.getOperand(i)));
    }
    E += ")";
    // C++ mode: an unrewritten matrix-returning call (matlab_zeros / ones /
    // eye / rand / …) returns void* at the C ABI level. When the result
    // flows into ostream<< or any context expecting a typed Matrix, wrap
    // it in `Matrix(...)` so overload resolution picks the friend `<<`
    // and not ostream's pointer overload.
    if (Cpp && C.getNumResults() == 1 && isMatrixValue(C.getResult(0)))
      E = "Matrix(" + E + ")";
    Expr = E;
    return true;
  }
  if (auto C = dyn_cast<LLVM::CallOp>(Op)) {
    if (!C.getCallee()) return false;
    // Class-field read: obj_get_f64(self, "X", _) → bare field name (or
    // `instance.X` outside the class method). Tried first so it pre-empts
    // the generic call form below.
    if (tryRewriteObjGet(C, Expr)) return true;
    // Class-method call: emit `Class(args)` / `obj.method(args)` /
    // `a OP b` / `obj.X` / `Class::method(args)`.
    if (tryRewriteAsClassCall(*C.getCallee(), C.getOperands(), Expr))
      return true;
    // Matrix runtime call: `matlab_matmul_mm(A, B)` → `(A * B)`, etc.
    if (tryRewriteAsMatrixCall(*C.getCallee(), C.getOperands(), Expr))
      return true;
    std::string E = C.getCallee()->str() + "(";
    for (unsigned i = 0; i < C.getNumOperands(); ++i) {
      if (i) E += ", ";
      E += dropOuterParens(exprFor(C.getOperand(i)));
    }
    E += ")";
    if (Cpp && C.getNumResults() == 1 && isMatrixValue(C.getResult()))
      E = "Matrix(" + E + ")";
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

// ---------------------------------------------------------------------------
// Classdef → real C++ class translation
// ---------------------------------------------------------------------------

llvm::StringRef Emitter::classTypeOf(mlir::Value V) const {
  auto It = ClassValueType.find(V);
  if (It != ClassValueType.end()) return It->second;
  return {};
}

// Walk a string-global addressof and return the literal text, or {}.
static std::optional<std::string> getStringGlobalLit(
    mlir::Value V, mlir::ModuleOp M) {
  auto *D = V.getDefiningOp();
  auto Addr = mlir::dyn_cast_or_null<mlir::LLVM::AddressOfOp>(D);
  if (!Addr) return std::nullopt;
  auto G = M.lookupSymbol<mlir::LLVM::GlobalOp>(Addr.getGlobalName());
  if (!G) return std::nullopt;
  auto S = mlir::dyn_cast_or_null<mlir::StringAttr>(G.getValueAttr());
  if (!S) return std::nullopt;
  return S.getValue().str();
}

// True iff `Name` is a plain C/C++ identifier (and not a C++ keyword).
static bool isValidCppIdentifier(llvm::StringRef Name) {
  if (Name.empty()) return false;
  unsigned char C0 = (unsigned char)Name[0];
  if (!(std::isalpha(C0) || C0 == '_')) return false;
  for (char C : Name)
    if (!std::isalnum((unsigned char)C) && C != '_') return false;
  static constexpr const char *Kw[] = {
    "alignas","alignof","and","auto","bool","break","case","catch","char",
    "class","const","constexpr","continue","decltype","default","delete",
    "do","double","else","enum","explicit","export","extern","false",
    "float","for","friend","goto","if","inline","int","long","mutable",
    "namespace","new","not","nullptr","operator","or","private","protected",
    "public","register","return","short","signed","sizeof","static","struct",
    "switch","template","this","throw","true","try","typedef","typeid",
    "typename","union","unsigned","using","virtual","void","volatile","while",
    "xor",
  };
  for (auto *K : Kw) if (Name == K) return false;
  return true;
}

// Map MATLAB operator method names to the matching C++ operator spelling.
static llvm::StringRef cppOperatorSpellingFor(llvm::StringRef Name) {
  if (Name == "eq")     return "==";
  if (Name == "ne")     return "!=";
  if (Name == "lt")     return "<";
  if (Name == "le")     return "<=";
  if (Name == "gt")     return ">";
  if (Name == "ge")     return ">=";
  if (Name == "plus")   return "+";
  if (Name == "minus")  return "-";
  if (Name == "mtimes") return "*";
  if (Name == "times")  return "*";
  if (Name == "mrdivide") return "/";
  if (Name == "rdivide")  return "/";
  return {};
}

// Map a MATLAB method name (possibly with `get.` prefix or operator
// alias) to the C++ method identifier we emit. Operators get the dunder
// `operator<spelling>` form; get-properties keep the `get_` prefix
// joined with the property; everything else gets dot-stripped.
static std::string cppMethodIdentFor(llvm::StringRef Name) {
  if (Name.starts_with("get.")) {
    std::string S = "get_";
    S += Name.drop_front(4).str();
    return S;
  }
  if (auto Op = cppOperatorSpellingFor(Name); !Op.empty())
    return ("operator" + Op).str();
  std::string S = Name.str();
  for (char &C : S) if (C == '.') C = '_';
  return S;
}

void Emitter::collectClassdefs(mlir::ModuleOp M) {
  for (auto &Op : M.getBody()->getOperations()) {
    auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op);
    if (!F || F.getBody().empty()) continue;
    auto CN = F->getAttrOfType<mlir::StringAttr>("matlab.class_name");
    if (!CN) continue;
    auto Kind = F->getAttrOfType<mlir::StringAttr>("matlab.method_kind");
    auto &CD = Classes[CN.getValue()];
    if (auto SC = F->getAttrOfType<mlir::StringAttr>("matlab.class_super"))
      if (CD.Super.empty()) CD.Super = SC.getValue().str();
    if (Kind && Kind.getValue() == "ctor")
      CD.Ctors.push_back(F);
    else
      CD.Methods.push_back(F);
    ClassMethodFuncs.insert(F.getSymName().str());

    // Discover properties from obj_set_f64 / obj_get_f64 calls.
    F.getBody().walk([&](mlir::LLVM::CallOp C) {
      if (!C.getCallee()) return;
      llvm::StringRef Callee = *C.getCallee();
      if (Callee != "matlab_obj_set_f64" && Callee != "matlab_obj_get_f64")
        return;
      if (C.getNumOperands() < 2) return;
      auto Lit = getStringGlobalLit(C.getOperand(1), M);
      if (!Lit || !isValidCppIdentifier(*Lit)) return;
      for (auto &P : CD.Properties) if (P == *Lit) return;
      CD.Properties.push_back(*Lit);
    });
  }
}

void Emitter::populateClassValueTypes(mlir::ModuleOp M) {
  auto markCallResult = [&](mlir::Operation *Op, mlir::Value Result,
                             std::optional<llvm::StringRef> CalleeName)
                            -> bool {
    if (!CalleeName) return false;
    if (!ClassMethodFuncs.count(*CalleeName)) return false;
    auto F = M.lookupSymbol<mlir::func::FuncOp>(*CalleeName);
    if (!F) return false;
    auto Kind = F->getAttrOfType<mlir::StringAttr>("matlab.method_kind");
    if (!Kind) return false;
    auto CN = F->getAttrOfType<mlir::StringAttr>("matlab.class_name");
    if (!CN) return false;
    // ctors return their class.
    bool TagIt = (Kind.getValue() == "ctor");
    // Same-class operator overloads (`plus`, `minus`, `mtimes` …) that
    // return `!llvm.ptr` also produce a class-typed value at the use
    // site. Comparison ops (eq / ne / lt …) return `f64` so they're
    // skipped naturally by the FuncType check below.
    if (!TagIt) {
      auto MN = F->getAttrOfType<mlir::StringAttr>("matlab.method_name");
      auto FT = F.getFunctionType();
      if (MN && FT.getNumResults() == 1 &&
          mlir::isa<mlir::LLVM::LLVMPointerType>(FT.getResult(0))) {
        auto Op = cppOperatorSpellingFor(MN.getValue());
        if (Op == "+" || Op == "-" || Op == "*" || Op == "/")
          TagIt = true;
      }
    }
    if (!TagIt) return false;
    auto It = ClassValueType.find(Result);
    if (It != ClassValueType.end() && It->second == CN.getValue().str())
      return false;
    ClassValueType[Result] = CN.getValue().str();
    return true;
  };

  bool Changed = true;
  while (Changed) {
    Changed = false;
    M.walk([&](mlir::Operation *Op) {
      if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
        if (C.getCallee() && C.getNumResults() == 1)
          Changed |= markCallResult(Op, C.getResult(),
                                     llvm::StringRef(*C.getCallee()));
      }
      if (auto C = mlir::dyn_cast<mlir::func::CallOp>(Op)) {
        if (C.getNumResults() == 1)
          Changed |= markCallResult(Op, C.getResult(0),
                                     llvm::StringRef(C.getCallee()));
      }
      // Store of class-typed value into alloca.
      if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
        auto CT = classTypeOf(S.getValue());
        if (CT.empty()) return;
        auto *Def = S.getAddr().getDefiningOp();
        auto A = mlir::dyn_cast_or_null<mlir::LLVM::AllocaOp>(Def);
        if (!A) return;
        auto &T = ClassAllocaType[A.getOperation()];
        if (T != CT) { T = CT.str(); Changed = true; }
      }
      // Load from class-typed alloca.
      if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
        auto *Def = L.getAddr().getDefiningOp();
        auto A = mlir::dyn_cast_or_null<mlir::LLVM::AllocaOp>(Def);
        if (!A) return;
        auto It = ClassAllocaType.find(A.getOperation());
        if (It == ClassAllocaType.end()) return;
        auto &T = ClassValueType[L.getResult()];
        if (T != It->second) { T = It->second; Changed = true; }
      }
    });
  }
}

// Set of runtime functions that return `matlab_mat *`. Drives both the
// matrix-value propagation pass and the call-rewrite layer that maps
// these calls back to operator/method form.
static llvm::StringRef MatrixReturningFns[] = {
    "matlab_mat_from_buf", "matlab_mat_from_scalar", "matlab_empty_mat",
    "matlab_zeros", "matlab_ones", "matlab_eye", "matlab_magic",
    "matlab_rand", "matlab_randn", "matlab_range", "matlab_repmat",
    "matlab_matmul_mm", "matlab_inv", "matlab_mldivide_mm",
    "matlab_mrdivide_mm", "matlab_svd", "matlab_eig", "matlab_eig_V",
    "matlab_eig_D", "matlab_transpose", "matlab_diag", "matlab_reshape",
    "matlab_matpow",
    "matlab_add_mm", "matlab_sub_mm", "matlab_emul_mm", "matlab_ediv_mm",
    "matlab_epow_mm", "matlab_add_ms", "matlab_sub_ms", "matlab_emul_ms",
    "matlab_ediv_ms", "matlab_epow_ms", "matlab_add_sm", "matlab_sub_sm",
    "matlab_emul_sm", "matlab_ediv_sm", "matlab_epow_sm",
    "matlab_gt_mm", "matlab_ge_mm", "matlab_lt_mm", "matlab_le_mm",
    "matlab_eq_mm", "matlab_ne_mm", "matlab_gt_ms", "matlab_ge_ms",
    "matlab_lt_ms", "matlab_le_ms", "matlab_eq_ms", "matlab_ne_ms",
    "matlab_gt_sm", "matlab_ge_sm", "matlab_lt_sm", "matlab_le_sm",
    "matlab_eq_sm", "matlab_ne_sm",
    "matlab_neg_m", "matlab_exp_m", "matlab_log_m", "matlab_sin_m",
    "matlab_cos_m", "matlab_tan_m", "matlab_sqrt_m", "matlab_abs_m",
    "matlab_asin_m", "matlab_acos_m", "matlab_atan_m", "matlab_sinh_m",
    "matlab_cosh_m", "matlab_tanh_m", "matlab_log2_m", "matlab_log10_m",
    "matlab_sign_m", "matlab_floor_m", "matlab_ceil_m", "matlab_round_m",
    "matlab_fix_m", "matlab_atan2_m",
    "matlab_sum", "matlab_prod", "matlab_mean", "matlab_min", "matlab_max",
    "matlab_min_mm", "matlab_max_mm",
    "matlab_sum_dim", "matlab_prod_dim", "matlab_mean_dim",
    "matlab_min_dim", "matlab_max_dim",
    "matlab_cumsum", "matlab_cumprod",
    "matlab_cumsum_dim", "matlab_cumprod_dim",
    "matlab_size", "matlab_slice1", "matlab_slice2", "matlab_find",
    "matlab_erase_rows", "matlab_erase_cols",
    "matlab_struct_get_mat", "matlab_cell_get_mat", "matlab_obj_get_mat",
    "matlab_ws_get_mat",
    "matlab_linspace", "matlab_chol", "matlab_pinv", "matlab_permute",
    "matlab_rot90", "matlab_kron", "matlab_squeeze",
    "matlab_lu_L", "matlab_lu_U", "matlab_qr_Q", "matlab_qr_R",
    "matlab_flip", "matlab_fliplr", "matlab_flipud",
    "matlab_horzcat", "matlab_vertcat",
    "matlab_sort", "matlab_sortrows",
    "matlab_intersect", "matlab_ismember", "matlab_setdiff",
    "matlab_union", "matlab_unique", "matlab_ind2sub",
    "matlab_load_mat", "matlab_fread",
    // Complex / FFT — runtime returns `matlab_mat_c *` or real
    // `matlab_mat *`, both representable as a Matrix wrapper since the
    // wrapper only holds a `void *`.
    "matlab_complex_scalar", "matlab_mat_c_from_real",
    "matlab_mat_c_from_buf", "matlab_conj_c", "matlab_neg_c",
    "matlab_real_c", "matlab_imag_c", "matlab_angle_c", "matlab_abs_c",
    "matlab_add_cc", "matlab_sub_cc", "matlab_emul_cc", "matlab_ediv_cc",
    "matlab_matmul_cc", "matlab_transpose_c", "matlab_ctranspose_c",
    "matlab_fft_c", "matlab_ifft_c", "matlab_fft2_c", "matlab_ifft2_c",
};

bool Emitter::isMatrixReturningRuntimeFn(llvm::StringRef Name) const {
  for (auto &S : MatrixReturningFns)
    if (Name == S) return true;
  return false;
}

void Emitter::populateMatrixValueTypes(mlir::ModuleOp M) {
  if (!Cpp) return;
  // Seed: every result of a matrix-returning runtime call.
  // Plus parameters/results of MATLAB functions that pass matrices through
  // (we can detect these once propagation hits a func.return / func.call).
  bool Changed = true;
  while (Changed) {
    Changed = false;
    M.walk([&](mlir::Operation *Op) {
      if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
        if (C.getCallee() && C.getNumResults() == 1 &&
            isMatrixReturningRuntimeFn(*C.getCallee()))
          if (MatrixValues.insert(C.getResult()).second) Changed = true;
      }
      // Store of matrix-typed value into alloca → mark alloca, then any
      // load from the alloca becomes matrix-typed in the next pass.
      if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
        if (!isMatrixValue(S.getValue())) return;
        auto *Def = S.getAddr().getDefiningOp();
        auto A = mlir::dyn_cast_or_null<mlir::LLVM::AllocaOp>(Def);
        if (!A) return;
        if (MatrixAllocas.insert(A.getOperation()).second) Changed = true;
      }
      if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
        auto *Def = L.getAddr().getDefiningOp();
        auto A = mlir::dyn_cast_or_null<mlir::LLVM::AllocaOp>(Def);
        if (!A || !MatrixAllocas.count(A.getOperation())) return;
        if (MatrixValues.insert(L.getResult()).second) Changed = true;
      }
      // Yield in scf.if: if any operand is matrix-typed, the parent's
      // matching result is matrix-typed.
      if (auto Y = mlir::dyn_cast<mlir::scf::YieldOp>(Op)) {
        auto *Parent = Op->getParentOp();
        if (auto If = mlir::dyn_cast_or_null<mlir::scf::IfOp>(Parent)) {
          for (unsigned i = 0; i < Y.getNumOperands(); ++i)
            if (isMatrixValue(Y.getOperand(i)))
              if (MatrixValues.insert(If.getResult(i)).second) Changed = true;
        }
      }
      // Indirect call to an anon body: `((cast)__anon_N)(args...)` where
      // the first operand is `llvm.mlir.addressof @__anon_N`. Treat as
      // a direct call to that function for arg/result propagation —
      // otherwise capture-bound matrix args stay typed `void*` in the
      // anon body's signature and any matrix op in the body fails to
      // compile.
      if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
        if (!Call.getCallee()) {
          if (auto AO = Call.getOperand(0)
                            .getDefiningOp<mlir::LLVM::AddressOfOp>()) {
            if (auto Callee = M.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
                    AO.getGlobalName())) {
              if (!Callee.getBody().empty()) {
                auto &Entry = Callee.getBody().front();
                // Operands index 1.. correspond to block args 0..
                for (unsigned i = 1; i < Call.getNumOperands() &&
                                      (i - 1) < Entry.getNumArguments();
                     ++i) {
                  if (isMatrixValue(Call.getOperand(i))) {
                    auto BA = Entry.getArgument(i - 1);
                    if (MatrixValues.insert(BA).second) Changed = true;
                  }
                }
                // Return propagation: if the callee returns matrix, the
                // call result at this site is matrix-typed.
                if (Call.getNumResults() == 1) {
                  Callee.walk([&](mlir::LLVM::ReturnOp R) {
                    if (R.getNumOperands() == 1 &&
                        isMatrixValue(R.getOperand(0)))
                      if (MatrixValues.insert(Call.getResult()).second)
                        Changed = true;
                  });
                }
              }
            }
          }
        }
      }
      // Function call: if a callee's return is propagated as matrix,
      // mark the call result. Symmetric for arguments → block args.
      if (auto Call = mlir::dyn_cast<mlir::func::CallOp>(Op)) {
        auto Callee =
            M.lookupSymbol<mlir::func::FuncOp>(Call.getCallee());
        if (!Callee) return;
        if (Callee.getBody().empty()) return;
        // Result propagation: if the callee's return statement yields a
        // matrix-typed value, callers see Matrix.
        Callee.walk([&](mlir::func::ReturnOp R) {
          for (unsigned i = 0; i < R.getNumOperands() &&
                                i < Call.getNumResults();
               ++i) {
            if (isMatrixValue(R.getOperand(i)))
              if (MatrixValues.insert(Call.getResult(i)).second)
                Changed = true;
          }
        });
        // Argument propagation: if the call passes a Matrix in arg #i,
        // the callee's block-arg #i is Matrix-typed.
        if (Callee.getBody().getNumArguments() == Call.getNumOperands()) {
          for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
            if (isMatrixValue(Call.getOperand(i))) {
              auto BA = Callee.getBody().front().getArgument(i);
              if (MatrixValues.insert(BA).second) Changed = true;
            }
          }
        }
      }
    });
  }
  if (!MatrixValues.empty()) UsesMatrixWrapper = true;
}

bool Emitter::tryRewriteAsMatrixCall(llvm::StringRef Callee,
                                       mlir::ValueRange Operands,
                                       std::string &Out) {
  if (!Cpp) return false;
  // Scalar-returning matrix methods. The first operand must be a tracked
  // matrix value — otherwise the runtime call isn't operating on a
  // wrapper instance and we shouldn't dot-call into it.
  auto firstIsMatrix = [&]() {
    return !Operands.empty() && isMatrixValue(Operands[0]);
  };
  if (firstIsMatrix() && Operands.size() == 1) {
    auto opnd0 = dropOuterParens(this->exprFor(Operands[0]));
    if (Callee == "matlab_numel")  { Out = opnd0 + ".numel()"; return true; }
    if (Callee == "matlab_length") { Out = opnd0 + ".length()"; return true; }
    if (Callee == "matlab_det")    { Out = opnd0 + ".det()"; return true; }
  }
  if (!isMatrixReturningRuntimeFn(Callee)) return false;
  // Helpers: pretty operand spelling without superfluous outer parens.
  auto opnd = [&](unsigned i) {
    return dropOuterParens(this->exprFor(Operands[i]));
  };
  auto bin = [&](llvm::StringRef Op) -> std::string {
    return "(" + opnd(0) + " " + Op.str() + " " + opnd(1) + ")";
  };
  auto m1 = [&](llvm::StringRef Method) -> std::string {
    return opnd(0) + "." + Method.str() + "()";
  };
  auto m2 = [&](llvm::StringRef Method) -> std::string {
    return opnd(0) + "." + Method.str() + "(" + opnd(1) + ")";
  };
  if (Operands.size() == 2) {
    if (Callee == "matlab_matmul_mm") { Out = bin("*"); return true; }
    if (Callee == "matlab_add_mm")    { Out = bin("+"); return true; }
    if (Callee == "matlab_sub_mm")    { Out = bin("-"); return true; }
    if (Callee == "matlab_emul_mm")     { Out = m2("emul"); return true; }
    if (Callee == "matlab_ediv_mm")     { Out = m2("ediv"); return true; }
    if (Callee == "matlab_epow_mm")     { Out = m2("epow"); return true; }
    if (Callee == "matlab_mldivide_mm") { Out = m2("mldivide"); return true; }
    if (Callee == "matlab_mrdivide_mm") { Out = m2("mrdivide"); return true; }
    if (Callee == "matlab_add_ms")    { Out = bin("+"); return true; }
    if (Callee == "matlab_sub_ms")    { Out = bin("-"); return true; }
    if (Callee == "matlab_emul_ms")   { Out = bin("*"); return true; }
    if (Callee == "matlab_ediv_ms")   { Out = bin("/"); return true; }
    if (Callee == "matlab_add_sm")    { Out = bin("+"); return true; }
    if (Callee == "matlab_sub_sm")    { Out = bin("-"); return true; }
    if (Callee == "matlab_emul_sm")   { Out = bin("*"); return true; }
    if (Callee == "matlab_ediv_sm")   { Out = bin("/"); return true; }
  }
  if (Operands.size() == 1) {
    if (Callee == "matlab_transpose") { Out = m1("t"); return true; }
    if (Callee == "matlab_inv")       { Out = m1("inv"); return true; }
    if (Callee == "matlab_diag")      { Out = m1("diag"); return true; }
    if (Callee == "matlab_sum")       { Out = m1("sum"); return true; }
    if (Callee == "matlab_prod")      { Out = m1("prod"); return true; }
    if (Callee == "matlab_mean")      { Out = m1("mean"); return true; }
    if (Callee == "matlab_min")       { Out = m1("min"); return true; }
    if (Callee == "matlab_max")       { Out = m1("max"); return true; }
    if (Callee == "matlab_sqrt_m")    { Out = m1("sqrt"); return true; }
    if (Callee == "matlab_abs_m")     { Out = m1("abs"); return true; }
    if (Callee == "matlab_exp_m")     { Out = m1("exp"); return true; }
    if (Callee == "matlab_log_m")     { Out = m1("log"); return true; }
    if (Callee == "matlab_sin_m")     { Out = m1("sin"); return true; }
    if (Callee == "matlab_cos_m")     { Out = m1("cos"); return true; }
    if (Callee == "matlab_tan_m")     { Out = m1("tan"); return true; }
    if (Callee == "matlab_neg_m")     { Out = "(-" + opnd(0) + ")"; return true; }
    if (Callee == "matlab_eig")       { Out = m1("eig"); return true; }
    if (Callee == "matlab_eig_V")     { Out = m1("eigV"); return true; }
    if (Callee == "matlab_eig_D")     { Out = m1("eigD"); return true; }
    if (Callee == "matlab_svd")       { Out = m1("svd"); return true; }
  }
  // matlab_mat_from_buf((void*)slot, m, n) — caller side. The slot
  // expression is already an inlined "(void*)slotName" cast. Strip it
  // back to the pointer expression and emit `Matrix(slot, m, n)` via
  // the explicit pointer ctor (Matrix's matlab_mat* ctor accepts the
  // implicit-converted pointer).
  if (Callee == "matlab_mat_from_buf" && Operands.size() == 3) {
    std::string Buf = this->exprFor(Operands[0]);
    Out = "Matrix(" + Buf + ", " + this->exprFor(Operands[1]) +
          ", " + this->exprFor(Operands[2]) + ")";
    return true;
  }
  return false;
}

llvm::StringSet<> Emitter::inheritedProperties(
    llvm::StringRef ClassName) const {
  llvm::StringSet<> Out;
  llvm::StringRef Cur = ClassName;
  // Walk the super chain; every property the parent (transitively)
  // declares is inherited and must NOT be redeclared on the child.
  while (true) {
    auto It = Classes.find(Cur);
    if (It == Classes.end()) break;
    if (It->second.Super.empty()) break;
    auto Sup = Classes.find(It->second.Super);
    if (Sup == Classes.end()) break;
    for (auto &P : Sup->second.Properties) Out.insert(P);
    Cur = It->second.Super;
  }
  return Out;
}

bool Emitter::tryRewriteObjGet(mlir::LLVM::CallOp C, std::string &Out) {
  // Only the C++ path emits real classes; C stays on the runtime hash.
  if (!Cpp) return false;
  if (!C.getCallee() || *C.getCallee() != "matlab_obj_get_f64") return false;
  if (C.getNumOperands() < 2) return false;
  auto M = C->getParentOfType<mlir::ModuleOp>();
  auto Lit = getStringGlobalLit(C.getOperand(1), M);
  if (!Lit || !isValidCppIdentifier(*Lit)) return false;
  mlir::Value Recv = C.getOperand(0);
  // Inside a class method, the receiver is the method's `this`; emit
  // bare `Field` (member name resolves against `this`).
  if (InClassMethodBody && Recv == ClassMethodSelf) {
    Out = *Lit;
    return true;
  }
  // Outside a class method, the receiver must be a tracked class
  // instance for the rewrite to be safe.
  if (!classTypeOf(Recv).empty()) {
    Out = exprFor(Recv) + "." + *Lit;
    return true;
  }
  return false;
}

bool Emitter::tryRewriteObjSet(mlir::LLVM::CallOp C, std::string &Out) {
  if (!Cpp) return false;
  if (!C.getCallee() || *C.getCallee() != "matlab_obj_set_f64") return false;
  if (C.getNumOperands() < 4) return false;
  auto M = C->getParentOfType<mlir::ModuleOp>();
  auto Lit = getStringGlobalLit(C.getOperand(1), M);
  if (!Lit || !isValidCppIdentifier(*Lit)) return false;
  mlir::Value Recv = C.getOperand(0);
  std::string Lhs;
  if (InClassMethodBody && Recv == ClassMethodSelf) {
    Lhs = *Lit;
  } else if (!classTypeOf(Recv).empty()) {
    Lhs = stmtExpr(Recv) + "." + *Lit;
  } else {
    return false;
  }
  Out = Lhs + " = " + stmtExpr(C.getOperand(3));
  return true;
}

bool Emitter::tryRewriteAsClassCall(llvm::StringRef Callee,
                                     mlir::ValueRange Operands,
                                     std::string &Out) {
  if (!Cpp) return false;
  if (!ClassMethodFuncs.count(Callee)) return false;
  // Resolve the func op via any parent module in scope.
  mlir::func::FuncOp F;
  for (auto &KV : Classes) {
    for (auto Fn : KV.second.Ctors)
      if (Fn.getSymName() == Callee) { F = Fn; break; }
    if (F) break;
    for (auto Fn : KV.second.Methods)
      if (Fn.getSymName() == Callee) { F = Fn; break; }
    if (F) break;
  }
  if (!F) return false;
  auto CN = F->getAttrOfType<mlir::StringAttr>("matlab.class_name");
  auto Kind = F->getAttrOfType<mlir::StringAttr>("matlab.method_kind");
  auto MN = F->getAttrOfType<mlir::StringAttr>("matlab.method_name");
  if (!CN) return false;
  std::string Cls = CN.getValue().str();

  if (Kind && Kind.getValue() == "ctor") {
    std::string E = Cls + "(";
    for (unsigned i = 0; i < Operands.size(); ++i) {
      if (i) E += ", ";
      E += stmtExpr(Operands[i]);
    }
    E += ")";
    Out = E;
    return true;
  }
  if (Kind && Kind.getValue() == "static") {
    std::string Name = MN ? MN.getValue().str() : "";
    if (Name.empty()) return false;
    std::string E = Cls + "::" + Name + "(";
    for (unsigned i = 0; i < Operands.size(); ++i) {
      if (i) E += ", ";
      E += stmtExpr(Operands[i]);
    }
    E += ")";
    Out = E;
    return true;
  }
  if (MN && MN.getValue().starts_with("get.")) {
    if (Operands.size() != 1) return false;
    if (classTypeOf(Operands[0]).empty()) return false;
    Out = exprFor(Operands[0]) + "." +
          ("get_" + MN.getValue().drop_front(4)).str() + "()";
    return true;
  }
  if (MN) {
    if (auto Op = cppOperatorSpellingFor(MN.getValue()); !Op.empty()) {
      if (Operands.size() != 2) return false;
      if (classTypeOf(Operands[0]).empty()) return false;
      Out = "(" + exprFor(Operands[0]) + " " + Op.str() + " "
          + exprFor(Operands[1]) + ")";
      return true;
    }
  }
  if (Operands.empty()) return false;
  if (classTypeOf(Operands[0]).empty()) return false;
  std::string Name = MN ? cppMethodIdentFor(MN.getValue()) : "";
  if (Name.empty()) return false;
  std::string E = exprFor(Operands[0]) + "." + Name + "(";
  for (unsigned i = 1; i < Operands.size(); ++i) {
    if (i > 1) E += ", ";
    E += stmtExpr(Operands[i]);
  }
  E += ")";
  Out = E;
  return true;
}

void Emitter::emitCppClass(llvm::StringRef ClassName,
                            const CppClassDef &CD) {
  llvm::StringSet<> Inh = inheritedProperties(ClassName);
  llvm::SmallVector<llvm::StringRef, 8> Own;
  for (auto &P : CD.Properties)
    if (!Inh.count(P)) Own.push_back(P);

  OS << "class " << ClassName.str();
  if (!CD.Super.empty()) OS << " : public " << CD.Super;
  OS << " {\npublic:\n";
  for (auto &P : Own) OS << "  double " << P.str() << ";\n";
  if (!Own.empty() || !CD.Ctors.empty() || !CD.Methods.empty()) OS << "\n";
  // Always emit a default ctor — needed when the parent class chain
  // requires one for `Foo() = default;` style construction by main /
  // by std::array etc. and to keep field zero-init explicit.
  OS << "  " << ClassName.str() << "() = default;\n";
  for (auto F : CD.Ctors)  emitCppMethod(F, ClassName);
  for (auto F : CD.Methods) emitCppMethod(F, ClassName);
  OS << "};\n\n";
}

void Emitter::emitCppMethod(mlir::func::FuncOp F,
                             llvm::StringRef ClassName) {
  // Per-method emitter state — mirrors emitFuncFunc's reset.
  NextId = 0;
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
  Names.clear();
  UsedNames = UsedNamesAfterProlog;
  // Reserve `this` so locals can't accidentally shadow it. The existing
  // class name is reserved at module scope already.
  UsedNames.insert("this");
  LastLineFile.clear();
  LastLineNum = -1;
  PendingTrailingLine = -1;
  PendingTrailingText.clear();
  TrailingEmittedLines.clear();

  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());

  auto FT = F.getFunctionType();
  auto Kind = F->getAttrOfType<mlir::StringAttr>("matlab.method_kind");
  auto MN   = F->getAttrOfType<mlir::StringAttr>("matlab.method_name");
  bool IsCtor   = Kind && Kind.getValue() == "ctor";
  bool IsStatic = Kind && Kind.getValue() == "static";
  bool IsBinaryOp = MN && !cppOperatorSpellingFor(MN.getValue()).empty() &&
                    FT.getNumInputs() == 2;
  bool IsGet = MN && MN.getValue().starts_with("get.");

  auto &Entry = F.getBody().front();

  // Bind `self` (or `other` for binary ops). Static methods bind no
  // implicit receiver — every parameter is a real argument.
  if (!IsCtor && !IsStatic && FT.getNumInputs() >= 1) {
    Names[Entry.getArgument(0)] = "this";
    InClassMethodBody = true;
    ClassMethodSelf = Entry.getArgument(0);
    ClassMethodClassName = ClassName.str();
  }
  if (IsBinaryOp && FT.getNumInputs() == 2 &&
      mlir::isa<mlir::LLVM::LLVMPointerType>(FT.getInput(1))) {
    UsedNames.insert("other");
    Names[Entry.getArgument(1)] = "other";
    // The binary-op `other` is a same-class instance, so obj_get_f64
    // on it should rewrite to `other.Field`.
    ClassValueType[Entry.getArgument(1)] = ClassName.str();
  }
  // Constructor: locate `matlab_obj_new`, alias its result to `this`,
  // suppress that call + the trailing `return obj_new_result`.
  mlir::Operation *CtorObjNew = nullptr;
  if (IsCtor) {
    F.getBody().walk([&](mlir::LLVM::CallOp C) {
      if (CtorObjNew) return;
      if (C.getCallee() && *C.getCallee() == "matlab_obj_new" &&
          C.getNumResults() == 1) {
        CtorObjNew = C.getOperation();
        Names[C.getResult()] = "this";
        SuppressedOps.insert(C.getOperation());
      }
    });
    if (CtorObjNew) {
      F.getBody().walk([&](mlir::Operation *Op) {
        if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
          if (R.getNumOperands() == 1 &&
              R.getOperand(0) == CtorObjNew->getResult(0))
            SuppressedOps.insert(Op);
        }
      });
      InClassMethodBody = true;
      ClassMethodSelf = CtorObjNew->getResult(0);
      ClassMethodClassName = ClassName.str();
    }
  }

  // For binary-op methods whose second arg is a same-class !llvm.ptr,
  // emit it as `const Class &other`. When the second arg is a scalar
  // (f64 / int) the operator takes a plain value (e.g. `Vec2 * double`).
  bool SecondIsSameClass = false;
  if (IsBinaryOp && FT.getNumInputs() == 2) {
    auto SecondTy = FT.getInput(1);
    SecondIsSameClass =
        mlir::isa<mlir::LLVM::LLVMPointerType>(SecondTy);
  }

  // For methods whose IR return type is `!llvm.ptr` and that belong to a
  // class, the body produces a class instance — emit the C++ return type
  // as the class name (so `Vec2 operator+(...)` returns a Vec2 directly).
  // Comparison-style operators (eq/ne/lt/le/gt/ge) keep their MATLAB
  // 1.0/0.0 return type.
  bool MethodReturnsThisClass = false;
  if (!IsCtor && FT.getNumResults() == 1 &&
      mlir::isa<mlir::LLVM::LLVMPointerType>(FT.getResult(0))) {
    if (MN) {
      auto Op = cppOperatorSpellingFor(MN.getValue());
      if (Op == "+" || Op == "-" || Op == "*" || Op == "/")
        MethodReturnsThisClass = true;
    }
  }

  // Header line.
  indent(1);
  if (IsStatic) OS << "static ";
  if (IsCtor) {
    OS << ClassName.str() << "(";
  } else {
    std::string RetTy;
    if (FT.getNumResults() == 0)        RetTy = "void";
    else if (MethodReturnsThisClass)    RetTy = ClassName.str();
    else                                RetTy = cTypeOf(FT.getResult(0));
    std::string MethodName;
    if (IsGet)
      MethodName = ("get_" + MN.getValue().drop_front(4)).str();
    else if (auto Op = MN ? cppOperatorSpellingFor(MN.getValue())
                          : llvm::StringRef();
             !Op.empty())
      MethodName = ("operator" + Op).str();
    else if (MN)
      MethodName = cppMethodIdentFor(MN.getValue());
    else
      MethodName = F.getSymName().str();
    OS << RetTy << " " << MethodName << "(";
  }

  // Parameters: skip the implicit `this` (first arg for non-ctor / non-static).
  bool FirstParam = true;
  unsigned StartArg = (IsCtor || IsStatic) ? 0 : 1;
  for (unsigned i = StartArg; i < FT.getNumInputs(); ++i) {
    if (!FirstParam) OS << ", ";
    FirstParam = false;
    auto Arg = Entry.getArgument(i);
    std::string N;
    if (i == 1 && IsBinaryOp && SecondIsSameClass) {
      OS << "const " << ClassName.str() << " &other";
      Names[Arg] = "other";
      continue;
    }
    if (auto NA = F.getArgAttrOfType<mlir::StringAttr>(i, "matlab.name"))
      N = uniqueName(NA.getValue());
    else
      N = freshName();
    Names[Arg] = N;
    OS << cTypeOf(FT.getInput(i)) << " " << N;
  }
  OS << ")";

  // Constructor member-init-list shortcut. When the body is just a flat
  // sequence of `obj_set_f64(this, "X", _, value)` calls (modulo the
  // suppressed obj_new + return, plus inlined constants/addressofs),
  // emit `: Field(arg), Field(arg) {}` instead of body-assignment form.
  // For subclasses with inherited fields the leading stores can become
  // a base-ctor call when their order matches the parent ctor's params.
  if (IsCtor && Cpp) {
    auto ClsIt = Classes.find(ClassName);
    auto M = F->getParentOfType<mlir::ModuleOp>();
    if (ClsIt != Classes.end()) {
      llvm::StringSet<> Inh = inheritedProperties(ClassName);
      llvm::SmallVector<std::pair<std::string, std::string>, 8> SetCalls;
      // The frontend wraps ctor bodies in `if nargin == N` checks; with
      // `nargin` baked in by the polymorphic dispatch the condition
      // folds to a literal true. Recognise that shape so the analyser
      // can recurse into the live branch.
      auto isStaticTrue = [&](mlir::Value V) -> bool {
        if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>())
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
            return IA.getValue().getZExtValue() != 0;
        if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>())
          if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
            return IA.getValue().getZExtValue() != 0;
        if (auto Cmp = V.getDefiningOp<mlir::arith::CmpFOp>()) {
          auto getLit = [](mlir::Value W, double &Dst) {
            auto C = W.getDefiningOp<mlir::arith::ConstantOp>();
            if (!C) return false;
            auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue());
            if (!FA) return false;
            Dst = FA.getValueAsDouble();
            return true;
          };
          double L, R;
          if (!getLit(Cmp.getLhs(), L) || !getLit(Cmp.getRhs(), R))
            return false;
          using P = mlir::arith::CmpFPredicate;
          switch (Cmp.getPredicate()) {
            case P::OEQ: case P::UEQ: return L == R;
            case P::ONE: case P::UNE: return L != R;
            case P::OLT: case P::ULT: return L <  R;
            case P::OLE: case P::ULE: return L <= R;
            case P::OGT: case P::UGT: return L >  R;
            case P::OGE: case P::UGE: return L >= R;
            default: return false;
          }
        }
        return false;
      };

      std::function<bool(mlir::Block &)> analyse =
          [&](mlir::Block &Blk) -> bool {
        for (auto &Op2 : Blk.getOperations()) {
          if (SuppressedOps.count(&Op2)) continue;
          if (InlinedOps.count(&Op2)) continue;
          if (mlir::isa<mlir::arith::ConstantOp, mlir::LLVM::ConstantOp,
                        mlir::LLVM::ZeroOp, mlir::LLVM::AddressOfOp>(Op2))
            continue;
          if (mlir::isa<mlir::arith::CmpFOp, mlir::arith::CmpIOp>(Op2))
            continue;  // condition computation for the static-folded if
          if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op2)) {
            if (R.getNumOperands() == 0) continue;
            return false;
          }
          if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op2)) {
            if (R.getNumOperands() == 0) continue;
            return false;
          }
          if (auto Y = mlir::dyn_cast<mlir::scf::YieldOp>(Op2)) {
            if (Y.getNumOperands() == 0) continue;
            return false;
          }
          if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(&Op2)) {
            if (!isStaticTrue(If.getCondition())) return false;
            if (!analyse(If.getThenRegion().front())) return false;
            continue;
          }
          if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(&Op2)) {
            if (!C.getCallee()) return false;
            llvm::StringRef N = *C.getCallee();
            // matlab.nargin lowering: a const that's already inlined.
            if (N == "matlab_obj_set_f64" && C.getNumOperands() >= 4 &&
                C.getOperand(0) == ClassMethodSelf) {
              auto Lit = getStringGlobalLit(C.getOperand(1), M);
              if (!Lit || !isValidCppIdentifier(*Lit)) return false;
              SetCalls.push_back({*Lit, stmtExpr(C.getOperand(3))});
              continue;
            }
            return false;
          }
          // matlab.nargin op — surfaces as a builtin callop; covered above.
          // Any other surviving op disqualifies the init-list shortcut.
          return false;
        }
        return true;
      };

      bool Clean = analyse(Entry);
      if (Clean && !SetCalls.empty()) {
        // Split into base-ctor inits and own inits. The leading stores
        // whose field names are inherited go through the parent's ctor
        // call when their order matches the parent ctor's declared
        // parameter order; everything else lands in the own-init list.
        std::string BaseCall;
        llvm::SmallVector<std::pair<std::string, std::string>, 4> OwnInits;
        unsigned NextStore = 0;
        if (!ClsIt->second.Super.empty() && !Inh.empty()) {
          auto SupIt = Classes.find(ClsIt->second.Super);
          if (SupIt != Classes.end() && !SupIt->second.Ctors.empty()) {
            auto SupCtor = SupIt->second.Ctors.front();
            auto SupFT = SupCtor.getFunctionType();
            unsigned NParams = SupFT.getNumInputs();
            // The first NParams stores must match the parent ctor's
            // properties in declaration order. The simplest mapping
            // assumes the parent's ctor stores them in the order its
            // properties are declared.
            if (SetCalls.size() >= NParams && NParams > 0 &&
                SupIt->second.Properties.size() >= NParams) {
              bool Match = true;
              for (unsigned i = 0; i < NParams; ++i) {
                if (SetCalls[i].first != SupIt->second.Properties[i]) {
                  Match = false; break;
                }
              }
              if (Match) {
                BaseCall = ClsIt->second.Super + "(";
                for (unsigned i = 0; i < NParams; ++i) {
                  if (i) BaseCall += ", ";
                  BaseCall += SetCalls[i].second;
                }
                BaseCall += ")";
                NextStore = NParams;
              }
            }
          }
        }
        // The remaining stores must all set own (non-inherited)
        // properties — otherwise we'd shadow a base-class field.
        bool AllOwn = true;
        for (unsigned i = NextStore; i < SetCalls.size(); ++i) {
          if (Inh.count(SetCalls[i].first)) { AllOwn = false; break; }
          OwnInits.push_back(SetCalls[i]);
        }
        if (AllOwn) {
          OS << " : ";
          bool First = true;
          if (!BaseCall.empty()) { OS << BaseCall; First = false; }
          for (auto &P : OwnInits) {
            if (!First) OS << ", ";
            First = false;
            OS << P.first << "(" << P.second << ")";
          }
          OS << " {}\n";
          InClassMethodBody = false;
          ClassMethodSelf = mlir::Value();
          ClassMethodClassName.clear();
          return;
        }
      }
    }
  }

  OS << " {\n";

  // Trivial body? Empty / just-return / ctor with only obj_new+return.
  bool BodyIsTrivial = true;
  for (auto &Op : Entry.getOperations()) {
    if (SuppressedOps.count(&Op)) continue;
    if (InlinedOps.count(&Op)) continue;
    if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
      if (R.getNumOperands() == 0) continue;
      BodyIsTrivial = false; break;
    }
    if (mlir::isa<mlir::arith::ConstantOp, mlir::LLVM::ConstantOp,
                  mlir::LLVM::ZeroOp, mlir::LLVM::AddressOfOp>(Op))
      continue;
    BodyIsTrivial = false;
    break;
  }
  if (!BodyIsTrivial) emitRegion(F.getBody(), 2);

  indent(1);
  OS << "}\n";

  // Reset class-method context.
  InClassMethodBody = false;
  ClassMethodSelf = mlir::Value();
  ClassMethodClassName.clear();
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
  // Buffer instead of writing immediately. flushPendingLine() in
  // indent() (the entry point for every content emit) writes it just
  // before the actual statement; back-to-back line directives with no
  // statement between collapse to the most recent one.
  PendingLineActive = true;
  PendingLineLine = Line;
  PendingLineFile = File;
  PendingLineIndent = Indent;
}

void Emitter::flushPendingLine() {
  if (!PendingLineActive) return;
  for (int i = 0; i < PendingLineIndent; ++i) OS << "  ";
  OS << "#line " << PendingLineLine << " \"" << PendingLineFile << "\"\n";
  PendingLineActive = false;
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
  OS << (Cpp ? "#include <cstdint>\n" : "#include <stdint.h>\n");
  if (!Cpp) OS << "#include <stdbool.h>\n";
  // IO-substitution headers. When the module has no parfor (no mutex
  // coordination needed) we can collapse the matlab_disp_* runtime calls
  // into direct stdio / iostream output, which reads as hand-written C
  // / modern C++. The flags were computed in precomputeModuleProperties.
  if (NeedsStdio)    OS << "#include <stdio.h>\n";
  if (NeedsIostream) OS << "#include <iostream>\n";
  // C++ Matrix wrapper: when any matrix-typed value flows through the
  // module, pull in the wrapper header. It transitively includes the
  // C ABI header, so the per-fn extern "C" block becomes redundant
  // (we still emit it for non-matrix runtime fns).
  if (Cpp && UsesMatrixWrapper)
    OS << "#include \"matlab_runtime.hpp\"\n";
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
    // C++ classdef rewrites: obj_get/set_f64 inside a class method
    // becomes direct member access; obj_new in a ctor is suppressed.
    // Outside a class method, obj_get/set on a tracked class instance
    // also rewrites. None of these survive to the runtime, so they
    // shouldn't pull in `extern matlab_obj_*` prototypes.
    if (Cpp) {
      auto inClassMethod = [&]() -> bool {
        auto P = C->getParentOfType<mlir::func::FuncOp>();
        return P && ClassMethodFuncs.count(P.getSymName());
      };
      if (Name == "matlab_obj_new" && inClassMethod())
        Substituted = true;
      if ((Name == "matlab_obj_get_f64" || Name == "matlab_obj_set_f64") &&
          C.getNumOperands() >= 2) {
        if (inClassMethod()) {
          Substituted = true;
        } else if (!classTypeOf(C.getOperand(0)).empty()) {
          Substituted = true;
        }
      }
    }
    if (!HasParfor && !Substituted) {
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
  // The Matrix-wrapper rewrites emit `std::cout << M;` for matlab_disp_mat
  // and friends. Even if the program has no other disp call, the wrapper
  // path still needs <iostream>.
  if (Cpp) {
    M.walk([&](mlir::LLVM::CallOp C) {
      if (!C.getCallee()) return;
      if (*C.getCallee() == "matlab_disp_mat") NeedsIostream = true;
    });
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
      // `<<` binds tighter than `==` / `<` / etc. in C++, so any
      // operator-overload expression has to keep its outer parens here
      // — use exprFor (paren-preserving) instead of stmtExpr.
      OS << "std::cout << " << exprFor(Call.getOperand(0)) << " << '\\n';\n";
    } else {
      OS << "printf(\"%g\\n\", " << stmtExpr(Call.getOperand(0)) << ");\n";
    }
    return true;
  }

  if (Name == "matlab_disp_i64" && Call.getNumOperands() == 1) {
    indent(Indent);
    if (Cpp) {
      OS << "std::cout << " << exprFor(Call.getOperand(0)) << " << '\\n';\n";
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
  // Pre-pass: collect classdef methods into Classes so the C++ class
  // block emitter has the property list and ClassMethodFuncs is
  // populated when the free-function pass tries to skip class methods.
  // Runs before precomputeModuleProperties so the live-runtime
  // detection knows which obj_* / class-method calls will be rewritten
  // and shouldn't pull in extern prototypes for them.
  collectClassdefs(M);
  // Tag every SSA value that holds a class instance (ctor result,
  // load from a class-typed alloca, etc.) so call-site rewrites + the
  // alloca / variable-type emit can resolve to the right class.
  if (Cpp) populateClassValueTypes(M);
  if (Cpp) populateMatrixValueTypes(M);
  precomputeModuleProperties(M);
  emitProlog();
  // Anything already in UsedNames after the prolog is module-scope
  // (function names, runtime externs, globals); freeze that snapshot so
  // each function body can start with a fresh local namespace.
  UsedNamesAfterProlog = UsedNames;

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
    // Set of runtime functions whose declaration is already provided by
    // matlab_runtime.hpp — when the wrapper is in use, redeclaring them
    // in the per-module extern "C" block is just noise.
    auto wrapperCovers = [&](llvm::StringRef N) -> bool {
      if (!Cpp || !UsesMatrixWrapper) return false;
      static llvm::StringRef Covered[] = {
          "matlab_mat_from_buf", "matlab_disp_mat", "matlab_matmul_mm",
          "matlab_add_mm", "matlab_sub_mm", "matlab_emul_mm",
          "matlab_ediv_mm", "matlab_epow_mm", "matlab_add_ms",
          "matlab_sub_ms", "matlab_emul_ms", "matlab_ediv_ms",
          "matlab_add_sm", "matlab_sub_sm", "matlab_emul_sm",
          "matlab_ediv_sm", "matlab_transpose", "matlab_inv",
          "matlab_diag", "matlab_sum", "matlab_prod", "matlab_mean",
          "matlab_min", "matlab_max", "matlab_sqrt_m", "matlab_abs_m",
          "matlab_exp_m", "matlab_log_m", "matlab_sin_m",
          "matlab_cos_m", "matlab_tan_m", "matlab_numel",
          "matlab_length", "matlab_det", "matlab_mldivide_mm",
          "matlab_mrdivide_mm", "matlab_eig", "matlab_eig_V",
          "matlab_eig_D", "matlab_svd", "matlab_neg_m",
          "matlab_reshape", "matlab_repmat", "matlab_matpow",
      };
      for (auto &S : Covered)
        if (N == S) return true;
      return false;
    };
    for (auto &Op : M.getBody()->getOperations()) {
      auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op);
      if (!F) continue;
      if (!F.getBody().empty()) continue;
      UsedNames.insert(F.getSymName().str());
      if (LiveRuntimeFuncs.find(F.getSymName()) == LiveRuntimeFuncs.end())
        continue;
      if (wrapperCovers(F.getSymName())) continue;
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
        // C++ class methods are emitted inside the class block — the
        // flat function never gets a definition, so its forward decl
        // would dangle.
        if (Cpp && ClassMethodFuncs.count(F.getSymName())) continue;
        auto FT = F.getFunctionType();
        std::string RetTy = FT.getNumResults() == 0
                                ? std::string("void")
                                : cTypeOf(FT.getResult(0));
        // Lift to Matrix when any propagated return value is matrix-typed
        // — keep forward decl in sync with the function definition's
        // signature emission below.
        if (Cpp && FT.getNumResults() == 1) {
          bool AnyMatrix = false;
          F.walk([&](mlir::func::ReturnOp R) {
            if (R.getNumOperands() == 1 && isMatrixValue(R.getOperand(0)))
              AnyMatrix = true;
          });
          if (AnyMatrix) RetTy = "Matrix";
        }
        Buf << "static " << RetTy << " " << F.getSymName().str() << "(";
        auto &Entry = F.getBody().front();
        for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
          if (i) Buf << ", ";
          std::string ParamTy =
              (Cpp && isMatrixValue(Entry.getArgument(i)))
                  ? "Matrix"
                  : cTypeOf(FT.getInput(i));
          Buf << ParamTy;
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
        if (Cpp && !mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())) {
          bool AnyMatrix = false;
          F.walk([&](mlir::LLVM::ReturnOp R) {
            if (R.getNumOperands() == 1 && isMatrixValue(R.getOperand(0)))
              AnyMatrix = true;
          });
          if (AnyMatrix) RetTy = "Matrix";
        }
        Buf << "static " << RetTy << " " << F.getSymName().str() << "(";
        auto &Entry = F.getBody().front();
        for (unsigned i = 0; i < FT.getNumParams(); ++i) {
          if (i) Buf << ", ";
          std::string ParamTy =
              (Cpp && isMatrixValue(Entry.getArgument(i)))
                  ? "Matrix"
                  : cTypeOf(FT.getParamType(i));
          Buf << ParamTy;
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

  /* Pass 2b (C++ only): emit each MATLAB classdef as a real C++ class
   * with named member fields and translated method bodies (so
   * `acc.Balance` is a direct field access, not a runtime hash call).
   * Inheritance requires parents to be declared first; we walk in
   * topological order so a `Savings : public BankAccount` always sees
   * its base class declared above it. */
  if (Cpp && !Classes.empty()) {
    llvm::SmallVector<llvm::StringRef, 8> ClassOrder;
    llvm::StringSet<> Emitted;
    bool Progress = true;
    while (Progress) {
      Progress = false;
      for (auto &KV : Classes) {
        if (Emitted.count(KV.first())) continue;
        if (!KV.second.Super.empty() && !Emitted.count(KV.second.Super))
          continue;
        ClassOrder.push_back(KV.first());
        Emitted.insert(KV.first());
        Progress = true;
      }
    }
    // Any leftover (forward-declared super not in module) emits anyway
    // so the class isn't silently dropped.
    for (auto &KV : Classes)
      if (!Emitted.count(KV.first())) ClassOrder.push_back(KV.first());
    for (llvm::StringRef Name : ClassOrder)
      emitCppClass(Name, Classes[Name]);
  }

  // Pass 3: emit function bodies. Class methods are already emitted
  // inside their class block, so skip the flat-function form for them.
  for (auto &Op : M.getBody()->getOperations()) {
    if (Failed) break;
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      if (Cpp && ClassMethodFuncs.count(F.getSymName())) continue;
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
  // Restore the module-scope name set so locals from prior functions
  // don't shadow names this function would prefer (e.g. parameter `x`).
  UsedNames = UsedNamesAfterProlog;
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
  // Lift the result type to Matrix when the function's return values are
  // matrix-typed in the propagated map — keeps signatures and bodies in
  // sync (the body's `return X.emul(X)` only type-checks if RetTy is
  // Matrix, not void*).
  if (Cpp && FT.getNumResults() == 1 &&
      !F.getBody().empty()) {
    bool AnyMatrix = false;
    F.walk([&](mlir::func::ReturnOp R) {
      if (R.getNumOperands() != 1) return;
      if (isMatrixValue(R.getOperand(0))) AnyMatrix = true;
    });
    if (AnyMatrix) RetTy = "Matrix";
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
    // Param emits as Matrix when the propagated map flagged the block-arg
    // as matrix-typed (caller passed a Matrix in this position).
    std::string ParamTy =
        (Cpp && isMatrixValue(Arg)) ? "Matrix" : cTypeOf(FT.getInput(i));
    OS << ParamTy << " " << N;
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
  // Restore the module-scope name set so locals from prior functions
  // don't shadow names this function would prefer (e.g. parameter `x`).
  UsedNames = UsedNamesAfterProlog;
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
  // Lift to Matrix in C++ mode when any return value is matrix-typed —
  // mirrors the func::FuncOp branch above. LLVM functions reach here
  // for outlined parfor / anon bodies.
  if (Cpp && !mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())) {
    bool AnyMatrix = false;
    F.walk([&](mlir::LLVM::ReturnOp R) {
      if (R.getNumOperands() == 1 && isMatrixValue(R.getOperand(0)))
        AnyMatrix = true;
    });
    if (AnyMatrix) RetTy = "Matrix";
  }
  emitLineDirective(F.getLoc(), 0);
  OS << "static " << RetTy << " " << F.getSymName().str() << "(";
  auto &Entry = F.getBody().front();
  for (unsigned i = 0; i < FT.getNumParams(); ++i) {
    if (i) OS << ", ";
    auto Arg = Entry.getArgument(i);
    std::string N = freshName();
    Names[Arg] = N;
    std::string ParamTy =
        (Cpp && isMatrixValue(Arg)) ? "Matrix" : cTypeOf(FT.getParamType(i));
    OS << ParamTy << " " << N;
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
  // A bare `return;` (no operand) at the very end of a function body is
  // C-redundant — control falls off naturally. Skip it for void
  // returns; keep `return X;` everywhere.
  auto isTrailingVoid = [](mlir::Operation *Op) {
    if (Op->getNumOperands() != 0) return false;
    if (Op != &Op->getBlock()->back()) return false;
    mlir::Operation *Parent = Op->getParentOp();
    if (!Parent) return false;
    if (!mlir::isa<mlir::func::FuncOp, mlir::LLVM::LLVMFuncOp>(Parent))
      return false;
    return true;
  };
  if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
    if (isTrailingVoid(R.getOperation())) return;
    indent(Indent);
    if (R.getNumOperands() == 0) OS << "return;\n";
    else OS << "return " << this->stmtExpr(R.getOperand(0)) << ";\n";
    return;
  }
  if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
    if (isTrailingVoid(R.getOperation())) return;
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
    // Class-field write: `obj_set_f64(obj, "X", _, v)` → `obj.X = v;`.
    // Statement-form only (the call returns void).
    if (Call.getCallee() && *Call.getCallee() == "matlab_obj_set_f64" &&
        Call.getNumResults() == 0) {
      std::string Rewrite;
      if (tryRewriteObjSet(Call, Rewrite)) {
        indent(Indent);
        OS << Rewrite << ";\n";
        return;
      }
    }
    indent(Indent);
    // For class-typed call results, try the rewrites first so we can
    // emit direct-init syntax (`Class name(args);`) when the rewritten
    // expression matches a `Class(args)` ctor call.
    bool BoundResult = (Call.getNumResults() == 1);
    std::string ResultName, ResultTy;
    if (BoundResult) {
      ResultName = this->name(Call.getResult());
      ResultTy = CppAuto ? "auto" : cTypeOfValue(Call.getResult());
      if (auto CT = classTypeOf(Call.getResult()); !CT.empty())
        ResultTy = CT.str();
    }
    if (auto Callee = Call.getCallee()) {
      // Class-field read with binding: `name = obj_get_f64(obj, "X", _)`.
      // The inline-expression path is handled by buildInlineExpr; this
      // branch covers the rare non-inlined case.
      if (*Callee == "matlab_obj_get_f64" && BoundResult) {
        std::string Rewrite;
        if (tryRewriteObjGet(Call, Rewrite)) {
          OS << ResultTy << " " << ResultName << " = "
             << Rewrite << ";\n";
          return;
        }
      }
      // Class-method call (constructor / regular / get / operator /
      // static). When the callee resolves to a classdef method the
      // result is C++-class-typed.
      {
        std::string Rewrite;
        if (tryRewriteAsClassCall(*Callee, Call.getOperands(), Rewrite)) {
          if (BoundResult) {
            // Direct-init: `Class name(args)` instead of
            // `Class name = Class(args)` when the rewrite is exactly
            // a ctor call with the same type as the binding.
            std::string CtorPrefix = ResultTy + "(";
            if (Cpp && Rewrite.size() > CtorPrefix.size() &&
                Rewrite.compare(0, CtorPrefix.size(), CtorPrefix) == 0 &&
                Rewrite.back() == ')') {
              std::string Args = Rewrite.substr(
                  CtorPrefix.size(),
                  Rewrite.size() - CtorPrefix.size() - 1);
              OS << ResultTy << " " << ResultName << "(" << Args << ");\n";
            } else {
              OS << ResultTy << " " << ResultName << " = "
                 << Rewrite << ";\n";
            }
          } else {
            OS << Rewrite << ";\n";
          }
          return;
        }
      }
      // Matrix-runtime statement-level rewrites.
      if (Cpp && *Callee == "matlab_disp_mat" &&
          Call.getNumOperands() == 1 && Call.getNumResults() == 0) {
        OS << "std::cout << " << exprFor(Call.getOperand(0)) << ";\n";
        return;
      }
      {
        std::string Rewrite;
        if (tryRewriteAsMatrixCall(*Callee, Call.getOperands(), Rewrite)) {
          if (BoundResult) {
            // Direct-init for `Matrix(...)` ctor expressions, mirroring
            // the class-call path so we get `Matrix A(slot, m, n)` not
            // `Matrix A = Matrix(slot, m, n)`.
            std::string CtorPrefix = ResultTy + "(";
            if (Rewrite.size() > CtorPrefix.size() &&
                Rewrite.compare(0, CtorPrefix.size(), CtorPrefix) == 0 &&
                Rewrite.back() == ')') {
              std::string Args = Rewrite.substr(
                  CtorPrefix.size(),
                  Rewrite.size() - CtorPrefix.size() - 1);
              OS << ResultTy << " " << ResultName << "(" << Args << ");\n";
            } else {
              OS << ResultTy << " " << ResultName << " = "
                 << Rewrite << ";\n";
            }
          } else {
            OS << Rewrite << ";\n";
          }
          return;
        }
      }
      if (BoundResult) OS << ResultTy << " " << ResultName << " = ";
      OS << Callee->str() << "(";
      for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
        if (i) OS << ", ";
        OS << this->stmtExpr(Call.getOperand(i));
      }
      OS << ");\n";
    } else {
      // Indirect call: first operand is the function pointer, rest are args.
      // When the pointer is just an addressof of a known function symbol
      // (the LowerAnonCalls shape — `llvm.mlir.addressof @__anon_N`), skip
      // the function-pointer cast wrapping and emit a plain direct call:
      // `__anon_0(5.0, 3.0)` reads exactly like a hand-written C call,
      // unlike `((double(*)(double, double))(void*)__anon_0)(5.0, 3.0)`.
      if (BoundResult) OS << ResultTy << " " << ResultName << " = ";
      auto AddrOf =
          Call.getOperand(0).getDefiningOp<mlir::LLVM::AddressOfOp>();
      if (AddrOf) {
        OS << AddrOf.getGlobalName().str() << "(";
        for (unsigned i = 1; i < Call.getNumOperands(); ++i) {
          if (i > 1) OS << ", ";
          OS << this->stmtExpr(Call.getOperand(i));
        }
        OS << ");\n";
        return;
      }
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
    bool BoundResult = (Call.getNumResults() == 1);
    std::string ResultName, ResultTy;
    if (BoundResult) {
      ResultName = this->name(Call.getResult(0));
      ResultTy = CppAuto ? "auto" : cTypeOfValue(Call.getResult(0));
      if (auto CT = classTypeOf(Call.getResult(0)); !CT.empty())
        ResultTy = CT.str();
    }
    {
      std::string Rewrite;
      if (tryRewriteAsClassCall(Call.getCallee(), Call.getOperands(),
                                 Rewrite)) {
        if (BoundResult) {
          std::string CtorPrefix = ResultTy + "(";
          if (Cpp && Rewrite.size() > CtorPrefix.size() &&
              Rewrite.compare(0, CtorPrefix.size(), CtorPrefix) == 0 &&
              Rewrite.back() == ')') {
            std::string Args = Rewrite.substr(
                CtorPrefix.size(),
                Rewrite.size() - CtorPrefix.size() - 1);
            OS << ResultTy << " " << ResultName << "(" << Args << ");\n";
          } else {
            OS << ResultTy << " " << ResultName << " = "
               << Rewrite << ";\n";
          }
        } else {
          OS << Rewrite << ";\n";
        }
        return;
      }
    }
    {
      std::string Rewrite;
      if (tryRewriteAsMatrixCall(Call.getCallee(), Call.getOperands(),
                                  Rewrite)) {
        if (BoundResult) {
          std::string CtorPrefix = ResultTy + "(";
          if (Rewrite.size() > CtorPrefix.size() &&
              Rewrite.compare(0, CtorPrefix.size(), CtorPrefix) == 0 &&
              Rewrite.back() == ')') {
            std::string Args = Rewrite.substr(
                CtorPrefix.size(),
                Rewrite.size() - CtorPrefix.size() - 1);
            OS << ResultTy << " " << ResultName << "(" << Args << ");\n";
          } else {
            OS << ResultTy << " " << ResultName << " = "
               << Rewrite << ";\n";
          }
        } else {
          OS << Rewrite << ";\n";
        }
        return;
      }
    }
    if (BoundResult) OS << ResultTy << " " << ResultName << " = ";
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

      // Matrix-literal init coming out of LowerTensorOps::materializeMat
      // is a strict pattern: GEP[k] + store cover indices 0..N-1 in
      // straight-line order, indexed by inlinable llvm.constant ops, and
      // every stored value is a constant or otherwise stmtExpr-able.
      // Squash the whole sequence into a single C99 initializer list.
      auto getConstIdx = [](mlir::LLVM::GEPOp G,
                             uint64_t &Out) -> bool {
        auto Idxs = G.getIndices();
        if (std::distance(Idxs.begin(), Idxs.end()) != 1) return false;
        auto Raw = *Idxs.begin();
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(Raw)) {
          Out = IA.getValue().getZExtValue();
          return true;
        }
        if (auto V = mlir::dyn_cast<mlir::Value>(Raw)) {
          if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>())
            if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
              Out = IA.getValue().getZExtValue();
              return true;
            }
          if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>())
            if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
              Out = IA.getValue().getZExtValue();
              return true;
            }
        }
        return false;
      };
      llvm::SmallVector<mlir::Value, 16> InitVals(N0);
      llvm::SmallVector<mlir::Operation *, 32> AbsorbedOps;
      bool InitOK = N0 > 0;
      uint64_t Filled = 0;
      // Walk the alloca's uses: every consumer must be a store TO the
      // alloca (idx 0 with no GEP) OR a GEP-with-constant-index whose
      // only user is a store. We tolerate any interleaved layout, since
      // the materializeMat sequence gets reshuffled by canonicalization
      // (idx 0 collapses to the bare alloca address).
      for (mlir::OpOperand &Use : A->getUses()) {
        if (!InitOK) break;
        mlir::Operation *U = Use.getOwner();
        if (auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(U)) {
          if (St.getAddr() != A.getResult()) { InitOK = false; break; }
          if (InitVals[0]) { InitOK = false; break; }  // duplicate idx 0
          InitVals[0] = St.getValue();
          AbsorbedOps.push_back(St.getOperation());
          ++Filled;
          continue;
        }
        if (auto Gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(U)) {
          if (Gep.getBase() != A.getResult()) { InitOK = false; break; }
          uint64_t Idx;
          if (!getConstIdx(Gep, Idx) || Idx >= N0 || InitVals[Idx]) {
            InitOK = false; break;
          }
          // The GEP must have exactly one user, a store of a value into
          // the GEPed pointer.
          if (!Gep.getResult().hasOneUse()) { InitOK = false; break; }
          auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(
              *Gep.getResult().getUsers().begin());
          if (!St || St.getAddr() != Gep.getResult()) {
            InitOK = false; break;
          }
          InitVals[Idx] = St.getValue();
          AbsorbedOps.push_back(Gep.getOperation());
          AbsorbedOps.push_back(St.getOperation());
          ++Filled;
          continue;
        }
        // Anything other than store / GEP — e.g. the matlab_mat_from_buf
        // call consuming the buffer pointer — is the legitimate post-init
        // user, not part of the init sequence.
        if (mlir::isa<mlir::LLVM::CallOp>(U)) continue;
        InitOK = false;
        break;
      }
      if (InitOK && Filled == N0)
        for (auto V : InitVals) if (!V) { InitOK = false; break; }
      if (InitOK && Filled == N0) {
        indent(Indent);
        OS << ElTy << " " << SlotName << "[" << N0 << "] = {";
        for (uint64_t i = 0; i < N0; ++i) {
          if (i) OS << ", ";
          OS << this->stmtExpr(InitVals[i]);
        }
        OS << "};\n";
        // Skip the `void* slot_p = (void*)slot;` trampoline — bind the
        // alloca's SSA value to the inline cast so consumers read
        // `matlab_mat_from_buf((void*)slot, 3.0, 3.0)` directly.
        InlineExprs[A.getResult()] = "(void*)" + SlotName;
        for (auto *Op2 : AbsorbedOps) SuppressedOps.insert(Op2);
        return;
      }

      // Fallback: zero-init buffer + a separate trampoline pointer when
      // the init pattern doesn't match (e.g. a non-constant element).
      N = uniqueName(SlotName + "_p");
      Names[A.getResult()] = N;
      indent(Indent);
      OS << ElTy << " " << SlotName << "[" << N0 << "] = {0};\n";
      indent(Indent);
      OS << "void* " << N << " = (void*)" << SlotName << ";\n";
      return;
    }

    std::string ElTy = cTypeOf(ET);
    // Class-typed allocas emit as `ClassName slot;` (default-constructed)
    // instead of `void* slot = 0;`. The slot's SSA value still routes
    // through DirectSlots — store/load expand to plain assignments /
    // reads. The store-form will emit `ClassName slot = val;` when
    // DeferDecl applies, in which case skip the bare declaration here.
    auto AllocaCT = ClassAllocaType.find(A.getOperation());
    if (Cpp && AllocaCT != ClassAllocaType.end()) {
      ElTy = AllocaCT->second;
    }
    // Matrix-typed allocas use the same direct-slot path as class
    // allocas: `Matrix slot;` (default-constructed null) instead of
    // `void* slot = 0;`. The wrapper's default ctor is null-safe.
    bool IsMatrixSlot = Cpp && MatrixAllocas.count(A.getOperation());
    if (IsMatrixSlot) ElTy = "Matrix";
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
      // Class-typed slot: `BankAccount acc;` (default-init). Bare
      // doubles still get `= 0;` to avoid an uninitialized-read trap.
      if ((Cpp && AllocaCT != ClassAllocaType.end()) || IsMatrixSlot)
        OS << ElTy << " " << SlotName << ";\n";
      else
        OS << ElTy << " " << SlotName << " = 0;\n";
      return;
    }
    N = uniqueName(SlotName + "_p");
    Names[A.getResult()] = N;
    indent(Indent);
    if (Cpp && AllocaCT != ClassAllocaType.end())
      OS << ElTy << " " << SlotName << ";\n";
    else
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
        // Class-typed alloca: emit `BankAccount slot = ctor(...);`.
        if (auto It = ClassAllocaType.find(AddrDef);
            Cpp && It != ClassAllocaType.end())
          Ty = It->second;
        else if (auto CT = classTypeOf(S.getValue()); !CT.empty())
          Ty = CT.str();
        std::string Expr = this->stmtExpr(S.getValue());
        // Direct-init syntax: when the RHS expression is exactly a
        // `Ty(args)` ctor call, rewrite to `Ty name(args);` for
        // idiomatic C++ — drops the visible repeated type name.
        std::string CtorPrefix = Ty + "(";
        if (Cpp && Expr.size() > CtorPrefix.size() &&
            Expr.compare(0, CtorPrefix.size(), CtorPrefix) == 0 &&
            Expr.back() == ')') {
          std::string Args = Expr.substr(CtorPrefix.size(),
                                          Expr.size() - CtorPrefix.size() - 1);
          OS << Ty << " " << DirectSlots[AddrDef] << "(" << Args << ");\n";
        } else {
          OS << Ty << " " << DirectSlots[AddrDef] << " = "
             << Expr << ";\n";
        }
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
    // Constant-folded condition: when the if-cond reduces to a literal
    // true or false (the typical `if nargin >= 2` shape after type
    // refinement leaves `if (2.0 == 2.0)` for a 2-arg function), drop
    // the if and emit only the live branch's body. Catches both the
    // arith and llvm constant op shapes.
    auto isConstBool = [&](mlir::Value V, bool &Out) -> bool {
      if (auto C = V.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
          Out = IA.getValue().getZExtValue() != 0;
          return true;
        }
      }
      if (auto C = V.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
          Out = IA.getValue().getZExtValue() != 0;
          return true;
        }
      }
      // arith.cmpf with two literal float operands of the same value
      // also folds (LowerNarginNargout leaves `cmpf eq, 2.0, 2.0`).
      if (auto Cmp = V.getDefiningOp<mlir::arith::CmpFOp>()) {
        auto getLit = [](mlir::Value W, double &Dst) {
          auto C = W.getDefiningOp<mlir::arith::ConstantOp>();
          if (!C) return false;
          auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue());
          if (!FA) return false;
          Dst = FA.getValueAsDouble();
          return true;
        };
        double L, R;
        if (!getLit(Cmp.getLhs(), L) || !getLit(Cmp.getRhs(), R))
          return false;
        switch (Cmp.getPredicate()) {
          case mlir::arith::CmpFPredicate::OEQ:
          case mlir::arith::CmpFPredicate::UEQ: Out = (L == R); break;
          case mlir::arith::CmpFPredicate::ONE:
          case mlir::arith::CmpFPredicate::UNE: Out = (L != R); break;
          case mlir::arith::CmpFPredicate::OLT:
          case mlir::arith::CmpFPredicate::ULT: Out = (L <  R); break;
          case mlir::arith::CmpFPredicate::OLE:
          case mlir::arith::CmpFPredicate::ULE: Out = (L <= R); break;
          case mlir::arith::CmpFPredicate::OGT:
          case mlir::arith::CmpFPredicate::UGT: Out = (L >  R); break;
          case mlir::arith::CmpFPredicate::OGE:
          case mlir::arith::CmpFPredicate::UGE: Out = (L >= R); break;
          default: return false;
        }
        return true;
      }
      return false;
    };
    bool CB;
    if (If.getNumResults() == 0 && isConstBool(If.getCondition(), CB)) {
      mlir::Region &Live = CB ? If.getThenRegion() : If.getElseRegion();
      if (!Live.empty()) emitRegion(Live, Indent);
      return;
    }

    // The `if cond { ... } else if cond { ... } ...` chain MATLAB
    // generates from `elseif` / `switch` lowers to a tower of nested
    // scf.if ops where each else region holds exactly one inner scf.if
    // and no other real ops. Detect that shape and unwrap into a flat
    // C `} else if (...) { ... }` cascade so the output reads as the
    // MATLAB programmer wrote it instead of the right-leaning lowering.
    auto isElseIfChain = [&](mlir::scf::IfOp Outer,
                             mlir::scf::IfOp &Inner) -> bool {
      if (Outer.getNumResults() != 0) return false;
      if (Outer.getElseRegion().empty()) return false;
      mlir::scf::IfOp Found;
      for (auto &Blk : Outer.getElseRegion().getBlocks()) {
        for (auto &Inner2 : Blk.getOperations()) {
          if (mlir::isa<mlir::scf::YieldOp>(Inner2)) continue;
          if (InlinedOps.count(&Inner2)) continue;
          if (SuppressedOps.count(&Inner2)) continue;
          if (auto N = mlir::dyn_cast<mlir::scf::IfOp>(Inner2)) {
            if (Found) return false;  // more than one real op
            Found = N;
            continue;
          }
          return false;
        }
      }
      if (!Found) return false;
      Inner = Found;
      return true;
    };

    auto thenIsTrivial = [&](mlir::scf::IfOp X) {
      // Then-region only used by emitOp's normal path; for the else-if
      // chain we don't need this. Kept here for symmetry / readability.
      (void)X;
      return false;
    };
    (void)thenIsTrivial;

    auto elseIsTrivial = [&](mlir::scf::IfOp X) {
      if (X.getElseRegion().empty()) return true;
      for (auto &Blk : X.getElseRegion().getBlocks()) {
        for (auto &Inner : Blk.getOperations()) {
          if (mlir::isa<mlir::scf::YieldOp>(Inner)) continue;
          if (InlinedOps.count(&Inner)) continue;
          if (SuppressedOps.count(&Inner)) continue;
          return false;
        }
      }
      return true;
    };

    // Switch re-emission. A run of `else if (x == c1) ... else if (x == c2)
    // ...` against the same SSA value with integer-valued double literals
    // came from a MATLAB `switch x; case c1; ... case cN; otherwise; end`.
    // Re-emit it as a real C/C++ `switch ((int)x) { ... }` so the original
    // intent survives lowering. We try this both at the top of an if-chain
    // (whole chain is a switch) and partway through (the suffix is the
    // switch — outer non-`==` arms stay as if/else-if).
    auto matchEqLit = [&](mlir::Value Cond, mlir::Value &Var,
                          int64_t &Lit) -> bool {
      auto Cmp = Cond.getDefiningOp<mlir::arith::CmpFOp>();
      if (!Cmp) return false;
      auto P = Cmp.getPredicate();
      if (P != mlir::arith::CmpFPredicate::OEQ &&
          P != mlir::arith::CmpFPredicate::UEQ)
        return false;
      auto getIntLit = [](mlir::Value V, int64_t &Out) {
        auto C = V.getDefiningOp<mlir::arith::ConstantOp>();
        if (!C) return false;
        auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue());
        if (!FA) return false;
        double D = FA.getValueAsDouble();
        if (!std::isfinite(D)) return false;
        int64_t I = (int64_t)D;
        if ((double)I != D) return false;
        Out = I;
        return true;
      };
      // Accept both `x == lit` and `lit == x`.
      if (getIntLit(Cmp.getRhs(), Lit)) { Var = Cmp.getLhs(); return true; }
      if (getIntLit(Cmp.getLhs(), Lit)) { Var = Cmp.getRhs(); return true; }
      return false;
    };

    auto tryCollectSwitch =
        [&](mlir::scf::IfOp Start, mlir::Value &Var,
            llvm::SmallVectorImpl<std::pair<int64_t, mlir::Region *>> &Cases,
            mlir::scf::IfOp &End) -> bool {
      Cases.clear();
      if (Start.getNumResults() != 0) return false;
      mlir::Value V0;
      int64_t L0;
      if (!matchEqLit(Start.getCondition(), V0, L0)) return false;
      Var = V0;
      Cases.push_back({L0, &Start.getThenRegion()});
      mlir::scf::IfOp Cur = Start;
      mlir::scf::IfOp Inner;
      while (isElseIfChain(Cur, Inner)) {
        mlir::Value V;
        int64_t L;
        if (!matchEqLit(Inner.getCondition(), V, L) || V != Var)
          return false;
        Cases.push_back({L, &Inner.getThenRegion()});
        Cur = Inner;
      }
      End = Cur;
      return Cases.size() >= 2;
    };

    auto emitSwitch =
        [&](mlir::Value Var,
            llvm::ArrayRef<std::pair<int64_t, mlir::Region *>> Cases,
            mlir::scf::IfOp End, unsigned Ind) {
          indent(Ind);
          OS << "switch ((int)" << this->stmtExpr(Var) << ") {\n";
          for (auto &Pair : Cases) {
            indent(Ind);
            OS << "case " << Pair.first << ":\n";
            emitRegion(*Pair.second, Ind + 1);
            indent(Ind + 1);
            OS << "break;\n";
          }
          if (!End.getElseRegion().empty() && !elseIsTrivial(End)) {
            indent(Ind);
            OS << "default:\n";
            emitRegion(End.getElseRegion(), Ind + 1);
            indent(Ind + 1);
            OS << "break;\n";
          }
          indent(Ind);
          OS << "}\n";
        };

    {
      llvm::SmallVector<std::pair<int64_t, mlir::Region *>, 8> Cases;
      mlir::Value SwVar;
      mlir::scf::IfOp End;
      if (tryCollectSwitch(If, SwVar, Cases, End)) {
        emitSwitch(SwVar, Cases, End, Indent);
        return;
      }
    }

    indent(Indent);
    OS << "if (" << this->stmtExpr(If.getCondition()) << ") {\n";
    emitRegion(If.getThenRegion(), Indent + 1);

    mlir::scf::IfOp Cur = If;
    mlir::scf::IfOp Inner;
    while (isElseIfChain(Cur, Inner)) {
      // Mid-chain switch suffix: from this Inner onward the chain is a
      // pure `==` cascade — emit `else { switch ... }` and stop.
      llvm::SmallVector<std::pair<int64_t, mlir::Region *>, 8> Cases;
      mlir::Value SwVar;
      mlir::scf::IfOp End;
      if (tryCollectSwitch(Inner, SwVar, Cases, End)) {
        indent(Indent);
        OS << "} else {\n";
        emitSwitch(SwVar, Cases, End, Indent + 1);
        indent(Indent);
        OS << "}\n";
        return;
      }
      indent(Indent);
      OS << "} else if (" << this->stmtExpr(Inner.getCondition()) << ") {\n";
      emitRegion(Inner.getThenRegion(), Indent + 1);
      Cur = Inner;
    }

    if (!Cur.getElseRegion().empty() &&
        (Cur.getNumResults() > 0 || !elseIsTrivial(Cur))) {
      indent(Indent);
      OS << "} else {\n";
      emitRegion(Cur.getElseRegion(), Indent + 1);
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
