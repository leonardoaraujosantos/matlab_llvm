// Emits Python source from an MLIR ModuleOp whose ops have already been
// lowered to a small, closed set: func / arith / scf / cf / llvm.call /
// llvm.alloca / llvm.load / llvm.store / llvm.mlir.global /
// llvm.mlir.addressof plus outlined llvm.func bodies (parfor / anonymous
// functions).
//
// Companion to EmitC.cpp: structure mirrors that emitter closely but
// targets Python (no types, native strings, `and`/`or`, ternary `a if c
// else b`). The emitted file imports `matlab_runtime` (NumPy-backed
// shim) and runs on CPython 3.10+.

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
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cctype>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

/// Metadata for an scf.while op that matched the canonical for-loop shape
/// (one f64 iter-arg, condition `iv <= end` or `iv >= end`, body ends with
/// `iv += step ; yield`). Mirrors the C emitter's ForLoopInfo so the same
/// loops collapse to a clean `for iv in range(...)` / `for iv in
/// rt.frange(...)` in Python.
struct ForLoopInfo {
  mlir::Value Init;
  mlir::Value End;
  mlir::Value Step;
  bool IsDecreasing = false;
  // The arith.addf and scf.yield at the tail of the after-region; absorbed
  // into the for-head and never emitted as their own statements.
  mlir::Operation *AddOp = nullptr;
  mlir::Operation *YieldOp = nullptr;
  // The store-to-slot that copies the iter-arg into the user-visible
  // induction variable, plus the slot's alloca. Both are skipped when the
  // loop owns the slot exclusively (FuseSlot = true).
  mlir::Operation *BindStore = nullptr;
  mlir::Operation *SlotAlloca = nullptr;
  bool FuseSlot = false;
  // The Python identifier used for the induction variable in the emitted
  // `for iv in ...:` head.
  std::string IvName;
};

class Emitter {
public:
  Emitter(std::ostream &OS, bool NoLine, const matlab::SourceManager *SM)
      : OS(OS), NoLine(NoLine), SM(SM) {}

  bool run(mlir::ModuleOp M);

private:
  // --- Naming ------------------------------------------------------------
  std::string name(mlir::Value V);
  std::string freshName(const char *Prefix = "v");
  std::string uniqueName(llvm::StringRef Hint);
  std::string sanitizeIdent(llvm::StringRef In);

  // --- Region / block printing ------------------------------------------
  void emitRegion(mlir::Region &R, int Indent);
  void emitBlock(mlir::Block &B, int Indent);
  void emitOp(mlir::Operation &Op, int Indent);

  // --- Top-level --------------------------------------------------------
  void emitGlobal(mlir::LLVM::GlobalOp G);
  void emitFuncFunc(mlir::func::FuncOp F);
  void emitLLVMFunc(mlir::LLVM::LLVMFuncOp F);
  void emitProlog();
  void precomputeModuleProperties(mlir::ModuleOp M);

  // --- Helpers ----------------------------------------------------------
  void indent(int N) { for (int i = 0; i < N; ++i) OS << "    "; }
  std::string constStr(mlir::LLVM::GlobalOp G);
  void fail(llvm::StringRef Msg) {
    if (!Failed)
      std::cerr << "error: emit-python: " << Msg.str() << "\n";
    Failed = true;
  }
  // Advance the source-line tracker to match `L`, emitting any leading
  // `%` comments / preserved blank lines from the MATLAB source.
  void advanceTo(mlir::Location L, int Indent);
  bool emitLeadingComments(llvm::StringRef FullPath, int AfterLine,
                           int Line, int Indent, bool FunctionHeader = false);

  // --- Single-use inlining ----------------------------------------------
  std::string exprFor(mlir::Value V);
  std::string stmtExpr(mlir::Value V);
  void computeInlines(mlir::Region &R);
  bool canInline(mlir::Operation &Op);
  bool buildInlineExpr(mlir::Operation &Op, std::string &Expr);

  // --- Break/continue flag un-lowering ----------------------------------
  // Walk a region and tag every break / continue flag slot, the false
  // stores that initialise/reset them, and the scf.if ops that should
  // collapse into native `break` / `continue` one-liners.
  void scanBreakContinueFlags(mlir::Region &R);
  bool isFlagInversion(mlir::Value V);
  void gatherNonFlagConjuncts(mlir::Value V,
      llvm::SmallVectorImpl<mlir::Value> &Out);

  // --- For-loop pattern detection ---------------------------------------
  // Match an scf.while produced by LowerSeqLoops::lowerForOp. Populates
  // ForPatterns / SuppressedOps / FusedForSlots so the scf.while emitter
  // can print `for iv in range(...):` (or `rt.frange(...)`) instead of the
  // 4-line while-emulated shape.
  bool matchForPattern(mlir::scf::WhileOp W, ForLoopInfo &Info);
  void scanForLoopPatterns(mlir::Region &R);

  // Try to evaluate V as a compile-time integer-valued literal (covers
  // both arith.constant f64 / i64 and llvm.mlir.constant). Returns true
  // and writes Out when the value is exactly an integer.
  static bool tryEvalIntLiteral(mlir::Value V, long long &Out);
  // Compile-time triple-check: are all of init/end/step integer literals?
  static bool forBoundsAreIntLiterals(const ForLoopInfo &Info,
                                      long long &Init, long long &End,
                                      long long &Step);

  // --- Callee remap: llvm.call @matlab_foo -> rt.foo --------------------
  static std::string remapRuntimeCallee(llvm::StringRef Name);

  // True when the runtime helper named (without `matlab_` prefix) takes a
  // string-length operand we can drop in Python (where strings carry
  // their own length).
  static bool calleeHasDroppableLengthArg(llvm::StringRef Suffix,
                                          unsigned &LengthArgIdx);

  // True when V is a constant condition we can fold at emit time.
  // Returns 1 for static-true, 0 for static-false, -1 for "not foldable".
  static int evalConstCond(mlir::Value V);

  std::ostream &OS;
  bool NoLine;
  const matlab::SourceManager *SM;
  bool Failed = false;

  llvm::DenseMap<mlir::Value, std::string> Names;
  llvm::DenseMap<mlir::Operation *, std::string> GlobalStrs;
  llvm::StringSet<> UsedNames;
  llvm::DenseMap<mlir::Value, std::string> InlineExprs;
  llvm::DenseSet<mlir::Operation *> InlinedOps;
  // Allocas whose entire use set is plain load/store. For these, the
  // pointer indirection collapses to a plain Python variable. Maps
  // alloca-op -> the Python identifier of the backing slot.
  llvm::DenseMap<mlir::Operation *, std::string> DirectSlots;
  // Alloca ops that hold arrays (from matrix-literal materialization);
  // they render as `name = [0.0] * N`, with stores/loads going via
  // indexing.
  llvm::DenseMap<mlir::Operation *, std::string> ArraySlots;
  // Subset of ArraySlots whose backing buffer was already emitted as a
  // collapsed Python list literal (`slot = [v0, v1, ...]`) — the
  // GEP/store ops that filled it have been absorbed and must not emit
  // their own statements. Keyed off SuppressedOps below.
  // Ops whose emission has been folded into a different op (e.g. the
  // GEP+store pairs that filled a matrix literal). Skip these in
  // emitOp().
  llvm::DenseSet<mlir::Operation *> SuppressedOps;
  // Break / continue flag slot allocas (matlab.name = "__did_break" /
  // "__did_continue"). The frontend lowers MATLAB `break`/`continue`
  // through these flags; the emitter un-lowers them back to native
  // Python keywords.
  llvm::DenseSet<mlir::Operation *> BreakFlagSlots;
  llvm::DenseSet<mlir::Operation *> ContinueFlagSlots;
  // scf.if ops whose entire body is a single flag-store; re-emit as
  // `if cond: break` (or continue). Value is the keyword to print.
  llvm::DenseMap<mlir::Operation *, const char *> FlagIfKind;
  // scf.if ops whose condition is purely a break/continue flag check
  // (`!did_break [& !did_continue]`) guarding a real body; emit the
  // then-region inline at the parent's indent.
  llvm::DenseSet<mlir::Operation *> InlinedIfs;
  // String globals (llvm.mlir.global with a StringAttr value), keyed off
  // the global's symbol name. When the AddressOf op for one of these is
  // inlined, return the Python literal directly so the emitted source
  // reads `rt.disp_str("Hello")` instead of `rt.disp_str(__matlab_str0)`.
  llvm::StringMap<std::string> StringGlobalLits;
  // Globals (any kind) that have already been inlined at every use site;
  // skip the top-of-file `name = "..."` declaration for them.
  llvm::StringSet<> SuppressedGlobals;
  // For-loop pattern matches (key: scf.while op).
  llvm::DenseMap<mlir::Operation *, ForLoopInfo> ForPatterns;
  // Slots whose alloca is owned by a for-loop's induction variable; the
  // alloca itself is suppressed and the slot name is the loop's IV name.
  llvm::DenseSet<mlir::Operation *> FusedForSlots;
  llvm::DenseMap<mlir::Operation *, std::string> FusedForSlotName;
  int NextId = 0;

  // Most recent line emitted; used to suppress duplicate line markers
  // and to drive leading-comment emission on forward jumps.
  std::string LastLineFile;
  int LastLineNum = -1;
  bool AtBlockStart = true;

  // Top-level script body (the body of `@main`) is hoisted to module
  // scope. While emitting that body this flag is true so we don't add
  // a surrounding `def main()`.
  bool InMainHoist = false;
};

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

std::string Emitter::freshName(const char *Prefix) {
  for (;;) {
    std::string S = Prefix;
    S += std::to_string(NextId++);
    if (UsedNames.insert(S).second) return S;
  }
}

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

// ---------------------------------------------------------------------------
// Literal formatting
// ---------------------------------------------------------------------------

static std::string formatIntAttr(mlir::IntegerAttr IA) {
  auto T = mlir::dyn_cast<mlir::IntegerType>(IA.getType());
  if (T && T.getWidth() == 1) {
    return (IA.getValue().getZExtValue() & 1u) ? "True" : "False";
  }
  char Buf[64];
  snprintf(Buf, sizeof(Buf), "%lld", (long long)IA.getInt());
  return Buf;
}

static std::string formatFloatAttr(mlir::FloatAttr FA) {
  // Pick the shortest precision (1..17) whose `%g` text round-trips
  // exactly back to D — that's what Python's `repr(0.05)` does, and what
  // a user reading the generated source expects (`0.05`, not
  // `0.050000000000000003`). We also avoid scientific notation when a
  // longer decimal form is available — `10.0` reads better than `1e+01`.
  double D = FA.getValueAsDouble();
  char Buf[64];
  // `%.17g` is the conservative reference: any value round-trips at 17.
  snprintf(Buf, sizeof(Buf), "%.17g", D);
  std::string Ref17 = Buf;
  bool Ref17HasE = (Ref17.find('e') != std::string::npos ||
                    Ref17.find('E') != std::string::npos);

  std::string S;
  for (int P = 1; P <= 17; ++P) {
    snprintf(Buf, sizeof(Buf), "%.*g", P, D);
    if (strtod(Buf, nullptr) != D) continue;
    bool ThisHasE = (strchr(Buf, 'e') || strchr(Buf, 'E'));
    // Reject a shorter form that switches to scientific notation when
    // the conservative reference renders cleanly in decimal.
    if (!Ref17HasE && ThisHasE) continue;
    S = Buf;
    break;
  }
  if (S.empty()) S = Ref17;
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

static std::string dropOuterParens(std::string E) {
  if (E.size() < 2 || E.front() != '(' || E.back() != ')') return E;
  int Depth = 0;
  for (size_t i = 0; i < E.size(); ++i) {
    if (E[i] == '(') ++Depth;
    else if (E[i] == ')') {
      --Depth;
      if (Depth == 0 && i + 1 < E.size()) return E;
    }
  }
  return E.substr(1, E.size() - 2);
}

// ---------------------------------------------------------------------------
// Runtime-symbol remap (`matlab_foo` -> `rt.foo`)
// ---------------------------------------------------------------------------
std::string Emitter::remapRuntimeCallee(llvm::StringRef Name) {
  // Every matlab_* runtime entry maps to `rt.<suffix>`. Keep non-matlab_
  // names verbatim so user-outlined helpers still resolve.
  if (Name.starts_with("matlab_")) {
    std::string Suf = Name.drop_front(strlen("matlab_")).str();
    // Suffixes that collide with Python keywords need an underscore so
    // `rt.assert(...)` parses. Kept in-sync with the runtime's
    // keyword-adjacent helpers.
    if (Suf == "assert" || Suf == "is" || Suf == "in" || Suf == "del" ||
        Suf == "if" || Suf == "else" || Suf == "elif" || Suf == "not" ||
        Suf == "and" || Suf == "or" || Suf == "lambda" || Suf == "pass" ||
        Suf == "class" || Suf == "def" || Suf == "return" ||
        Suf == "from" || Suf == "import" || Suf == "try" || Suf == "except"||
        Suf == "finally" || Suf == "raise" || Suf == "with" ||
        Suf == "while" || Suf == "for" || Suf == "yield" ||
        Suf == "global" || Suf == "nonlocal" ||
        Suf == "None" || Suf == "True" || Suf == "False")
      Suf += "_";
    return "rt." + Suf;
  }
  return Name.str();
}

// ---------------------------------------------------------------------------
// Single-use inlining
// ---------------------------------------------------------------------------

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

std::string Emitter::stmtExpr(mlir::Value V) {
  return dropOuterParens(exprFor(V));
}

bool Emitter::canInline(mlir::Operation &Op) {
  using namespace mlir;
  if (Op.getNumResults() != 1) return false;
  Value V = Op.getResult(0);

  // Always-inline: constants / zero / addressof.
  if (isa<LLVM::ConstantOp, arith::ConstantOp, LLVM::ZeroOp,
          LLVM::AddressOfOp>(Op))
    return true;

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
  // earlier value across them.
  auto isPureReadCall = [](Operation &Op2) -> bool {
    auto C = dyn_cast<LLVM::CallOp>(Op2);
    if (!C || !C.getCallee()) return false;
    StringRef N = *C.getCallee();
    if (!N.starts_with("matlab_")) return false;
    StringRef S = N.drop_front(strlen("matlab_"));
    // Conservative allowlist of known-pure runtime accessors. Adding
    // the wrong helper here would visibly reorder side effects in the
    // generated source, so we keep this list small and explicit.
    return S == "obj_get_f64" || S == "size" || S == "size_dim" ||
           S == "numel" || S == "numel3" || S == "length" ||
           S == "ndims" || S == "isempty" || S == "isnumeric" ||
           S == "isscalar" || S == "ismatrix" || S == "isvector" ||
           S == "isstruct" || S == "isfield" || S == "iscell" ||
           S == "isstring" || S == "string_len";
  };

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

// Is the value's type `i1` (as opposed to a wider integer)?
static bool isI1(mlir::Type T) {
  auto IT = mlir::dyn_cast<mlir::IntegerType>(T);
  return IT && IT.getWidth() == 1;
}

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
    // String globals fold to a Python literal at the use site so the
    // emitted source reads `rt.disp_str("Hello")` instead of
    // `rt.disp_str(__matlab_str0)`. Function-symbol addressofs (function
    // handles) keep the bare name so they decay to the Python callable.
    auto It = StringGlobalLits.find(A.getGlobalName());
    if (It != StringGlobalLits.end()) {
      Expr = It->second;
      SuppressedGlobals.insert(A.getGlobalName());
      return true;
    }
    Expr = A.getGlobalName().str();
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
  // Bitwise vs logical split on i1. `and`/`or` on bools is shorter;
  // `^` doesn't have a short logical form, so emit `!=` (works because
  // `True != False`).
  if (auto A = dyn_cast<arith::AndIOp>(Op)) {
    if (isI1(A.getType())) return bin("and");
    return bin("&");
  }
  if (auto O = dyn_cast<arith::OrIOp>(Op)) {
    if (isI1(O.getType())) return bin("or");
    return bin("|");
  }
  if (auto X = dyn_cast<arith::XOrIOp>(Op)) {
    if (isI1(X.getType())) return bin("!=");
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
    // Common shape from MATLAB's logical-to-double coercion: select(c,
    // 1.0, 0.0). Fold to `float(c)` instead of the ternary so the
    // emitted source reads like the MATLAB expression that produced it.
    auto isLit = [](Value V, double Want) -> bool {
      if (auto *D = V.getDefiningOp()) {
        FloatAttr FA;
        if (auto C = dyn_cast<arith::ConstantOp>(D))
          FA = dyn_cast<FloatAttr>(C.getValue());
        else if (auto C = dyn_cast<LLVM::ConstantOp>(D))
          FA = dyn_cast<FloatAttr>(C.getValue());
        return FA && FA.getValueAsDouble() == Want;
      }
      return false;
    };
    if (isLit(S.getTrueValue(), 1.0) && isLit(S.getFalseValue(), 0.0)) {
      Expr = "float(" + dropOuterParens(exprFor(S.getCondition())) + ")";
      return true;
    }
    // Python ternary: `a if c else b`.
    Expr = "(" + exprFor(S.getTrueValue()) + " if "
         + dropOuterParens(exprFor(S.getCondition())) + " else "
         + exprFor(S.getFalseValue()) + ")";
    return true;
  }
  if (isa<arith::SIToFPOp, arith::UIToFPOp>(Op)) {
    Expr = "float(" + dropOuterParens(exprFor(Op.getOperand(0))) + ")";
    return true;
  }
  if (isa<arith::FPToSIOp, arith::FPToUIOp>(Op)) {
    Expr = "int(" + dropOuterParens(exprFor(Op.getOperand(0))) + ")";
    return true;
  }
  // extsi/extui/trunci/extf/truncf are no-ops in Python.
  if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::TruncFOp, arith::ExtFOp>(Op)) {
    Expr = exprFor(Op.getOperand(0));
    return true;
  }
  if (auto L = dyn_cast<LLVM::LoadOp>(Op)) {
    if (auto *D = L.getAddr().getDefiningOp()) {
      auto It = DirectSlots.find(D);
      if (It != DirectSlots.end()) { Expr = It->second; return true; }
    }
    // Non-direct slot: the address itself IS the value (typically a
    // GEP expression we materialised as an index). For GEPs, the load
    // resolves to `base[idx]` syntax; the GEP's inline expression
    // already carries that form.
    Expr = exprFor(L.getAddr());
    return true;
  }
  if (auto G = dyn_cast<LLVM::GEPOp>(Op)) {
    // GEP into an array slot: produce `base[idx]` for a single-index
    // GEP. Multi-index GEPs are uncommon in our snapshot (only
    // matrix-literal buffers). Fall back to the first index, summed.
    std::string Base;
    if (auto *D = G.getBase().getDefiningOp()) {
      auto It = ArraySlots.find(D);
      if (It != ArraySlots.end()) Base = It->second;
    }
    if (Base.empty()) Base = exprFor(G.getBase());
    std::string Idx;
    bool First = true;
    for (auto I : G.getIndices()) {
      std::string Term;
      if (auto Vv = llvm::dyn_cast<mlir::Value>(I))
        Term = dropOuterParens(exprFor(Vv));
      else if (auto IA = llvm::dyn_cast<mlir::IntegerAttr>(I))
        Term = std::to_string(IA.getInt());
      else continue;
      if (First) { Idx = Term; First = false; }
      else       { Idx = "(" + Idx + " + " + Term + ")"; }
    }
    if (Idx.empty()) Idx = "0";
    Expr = Base + "[" + Idx + "]";
    return true;
  }
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
    std::string Callee = remapRuntimeCallee(*C.getCallee());
    unsigned LengthIdx = ~0u;
    bool DropLen = false;
    if (C.getCallee()->starts_with("matlab_")) {
      unsigned Idx;
      if (calleeHasDroppableLengthArg(
              C.getCallee()->drop_front(strlen("matlab_")), Idx)) {
        LengthIdx = Idx;
        DropLen = true;
      }
    }
    std::string E = Callee + "(";
    bool First = true;
    for (unsigned i = 0; i < C.getNumOperands(); ++i) {
      if (DropLen && i == LengthIdx) continue;
      if (!First) E += ", ";
      First = false;
      E += dropOuterParens(exprFor(C.getOperand(i)));
    }
    E += ")";
    Expr = E;
    return true;
  }
  return false;
}

// Is V a constant integer with the given value?
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
  // True if V reduces to a static `True` once break/continue flags are
  // un-lowered: a constant true, the xor(load flag, true) idiom, or the
  // xor(const-false, const-true) Mem2RegLite leaves when a flag has no
  // store-of-true.
  if (!V.getType().isInteger(1)) return false;
  if (isConstInt(V, 1)) return true;
  auto Xor = V.getDefiningOp<mlir::arith::XOrIOp>();
  if (!Xor) return false;
  if (!Xor.getResult().getType().isInteger(1)) return false;
  mlir::Value Flag;
  if (isConstInt(Xor.getRhs(), 1)) Flag = Xor.getLhs();
  else if (isConstInt(Xor.getLhs(), 1)) Flag = Xor.getRhs();
  else return false;
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
    if (And.getResult().getType().isInteger(1)) {
      gatherNonFlagConjuncts(And.getLhs(), Out);
      gatherNonFlagConjuncts(And.getRhs(), Out);
      return;
    }
  }
  Out.push_back(V);
}

// ---------------------------------------------------------------------------
// For-loop pattern detection (mirrors EmitC.cpp::matchForPattern)
// ---------------------------------------------------------------------------

bool Emitter::tryEvalIntLiteral(mlir::Value V, long long &Out) {
  auto *D = V.getDefiningOp();
  if (!D) return false;
  mlir::Attribute A;
  if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D)) A = C.getValue();
  else if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D)) A = C.getValue();
  else return false;
  if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(A)) {
    Out = IA.getInt();
    return true;
  }
  if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(A)) {
    double D = FA.getValueAsDouble();
    long long I = (long long)D;
    if ((double)I == D) { Out = I; return true; }
    return false;
  }
  return false;
}

bool Emitter::forBoundsAreIntLiterals(const ForLoopInfo &Info, long long &Init,
                                       long long &End, long long &Step) {
  return tryEvalIntLiteral(Info.Init, Init) &&
         tryEvalIntLiteral(Info.End, End) &&
         tryEvalIntLiteral(Info.Step, Step);
}

bool Emitter::matchForPattern(mlir::scf::WhileOp W, ForLoopInfo &Info) {
  // Same shape as EmitC: one f64 iter-arg, before-region forwards it
  // unchanged through scf.condition, condition is a single cmpf OLE/OGE,
  // after-region ends with arith.addf iv, step + scf.yield %addf.
  if (W.getInits().size() != 1) return false;
  mlir::Block &Before = W.getBefore().front();
  mlir::Block &After = W.getAfter().front();
  if (Before.getNumArguments() != 1 || After.getNumArguments() != 1)
    return false;
  auto F64 = mlir::Float64Type::get(W.getContext());
  if (Before.getArgument(0).getType() != F64) return false;
  if (After.getArgument(0).getType() != F64) return false;

  for (auto &Inner : Before.getOperations()) {
    if (mlir::isa<mlir::scf::ConditionOp>(Inner)) continue;
    if (InlinedOps.count(&Inner)) continue;
    return false;
  }
  auto Cond = mlir::cast<mlir::scf::ConditionOp>(Before.getTerminator());
  if (Cond.getArgs().size() != 1) return false;
  if (Cond.getArgs()[0] != Before.getArgument(0)) return false;

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

  if (After.getOperations().size() < 2) return false;
  auto Yld = mlir::dyn_cast<mlir::scf::YieldOp>(&After.back());
  if (!Yld || Yld.getResults().size() != 1) return false;
  auto *AddRaw = Yld.getResults()[0].getDefiningOp();
  auto Add = mlir::dyn_cast_or_null<mlir::arith::AddFOp>(AddRaw);
  if (!Add) return false;
  if (Add.getLhs() != After.getArgument(0)) return false;
  if (Add->getNextNode() != Yld.getOperation()) return false;

  Info.Init = W.getInits()[0];
  Info.Step = Add.getRhs();
  Info.AddOp = Add.getOperation();
  Info.YieldOp = Yld.getOperation();
  return true;
}

void Emitter::scanForLoopPatterns(mlir::Region &R) {
  // Mirrors EmitC: a slot is fusable when every use of the slot lives
  // inside one of its claimants' after-regions. Two consecutive loops
  // reusing the same `i` slot can both fuse when neither references `i`
  // outside its own body.
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

  llvm::DenseMap<mlir::Operation *, bool> SlotFusable;
  for (auto &Entry : SlotClaimants) {
    mlir::Operation *Slot = Entry.first;
    auto &Claimants = Entry.second;
    bool OK = true;
    for (auto &Use : Slot->getUses()) {
      mlir::Operation *User = Use.getOwner();
      bool InsideAny = false;
      for (mlir::Operation *W : Claimants) {
        mlir::Region *Loop = &W->getRegion(1);
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

// ---------------------------------------------------------------------------
// Constant condition folding
// ---------------------------------------------------------------------------

int Emitter::evalConstCond(mlir::Value V) {
  // Fold `arith.cmpi/cmpf C1, C2` and the boolean constants directly.
  // The polymorphic-dispatch layer bakes nargin into a constant comparison
  // (`if 2 == 2:`) which we want to drop entirely.
  if (auto *D = V.getDefiningOp()) {
    if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(D)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
        auto T = mlir::dyn_cast<mlir::IntegerType>(IA.getType());
        if (T && T.getWidth() == 1) return (IA.getInt() & 1) ? 1 : 0;
      }
    }
    if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(D)) {
      if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue())) {
        auto T = mlir::dyn_cast<mlir::IntegerType>(IA.getType());
        if (T && T.getWidth() == 1) return (IA.getInt() & 1) ? 1 : 0;
      }
    }
    if (auto Ci = mlir::dyn_cast<mlir::arith::CmpIOp>(D)) {
      long long L, R;
      if (tryEvalIntLiteral(Ci.getLhs(), L) &&
          tryEvalIntLiteral(Ci.getRhs(), R)) {
        switch (Ci.getPredicate()) {
          case mlir::arith::CmpIPredicate::eq:  return L == R ? 1 : 0;
          case mlir::arith::CmpIPredicate::ne:  return L != R ? 1 : 0;
          case mlir::arith::CmpIPredicate::slt:
          case mlir::arith::CmpIPredicate::ult: return L <  R ? 1 : 0;
          case mlir::arith::CmpIPredicate::sle:
          case mlir::arith::CmpIPredicate::ule: return L <= R ? 1 : 0;
          case mlir::arith::CmpIPredicate::sgt:
          case mlir::arith::CmpIPredicate::ugt: return L >  R ? 1 : 0;
          case mlir::arith::CmpIPredicate::sge:
          case mlir::arith::CmpIPredicate::uge: return L >= R ? 1 : 0;
        }
      }
    }
    if (auto Cf = mlir::dyn_cast<mlir::arith::CmpFOp>(D)) {
      long long L, R;
      if (tryEvalIntLiteral(Cf.getLhs(), L) &&
          tryEvalIntLiteral(Cf.getRhs(), R)) {
        switch (Cf.getPredicate()) {
          case mlir::arith::CmpFPredicate::OEQ:
          case mlir::arith::CmpFPredicate::UEQ: return L == R ? 1 : 0;
          case mlir::arith::CmpFPredicate::ONE:
          case mlir::arith::CmpFPredicate::UNE: return L != R ? 1 : 0;
          case mlir::arith::CmpFPredicate::OLT:
          case mlir::arith::CmpFPredicate::ULT: return L <  R ? 1 : 0;
          case mlir::arith::CmpFPredicate::OLE:
          case mlir::arith::CmpFPredicate::ULE: return L <= R ? 1 : 0;
          case mlir::arith::CmpFPredicate::OGT:
          case mlir::arith::CmpFPredicate::UGT: return L >  R ? 1 : 0;
          case mlir::arith::CmpFPredicate::OGE:
          case mlir::arith::CmpFPredicate::UGE: return L >= R ? 1 : 0;
          default: return -1;
        }
      }
    }
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Runtime call ABI: which helpers carry a droppable string-length operand?
// ---------------------------------------------------------------------------

bool Emitter::calleeHasDroppableLengthArg(llvm::StringRef Suffix,
                                          unsigned &LengthArgIdx) {
  // The C ABI passes (str_ptr, str_len) for any helper that takes a
  // user-string. Python strings carry their length, so the length operand
  // is dead weight — drop it. The runtime keeps an optional `n=None`
  // parameter for back-compat with hand-written callers.
  // Raw-string helpers carry an explicit (ptr, i64-length) tail. The
  // matlab_string variants used by file I/O wrap the buffer in a
  // descriptor and don't take a separate length, so they're absent
  // from this list.
  if (Suffix == "disp_str")      { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_str")   { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64")   { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64_2") { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64_3") { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64_4") { LengthArgIdx = 1; return true; }
  if (Suffix == "input_num")     { LengthArgIdx = 1; return true; }
  // Object field accessors carry the (name_ptr, name_len) pair as their
  // 2nd / 3rd operands; the Python runtime ignores name_len.
  if (Suffix == "obj_get_f64")   { LengthArgIdx = 2; return true; }
  if (Suffix == "obj_set_f64")   { LengthArgIdx = 2; return true; }
  return false;
}

void Emitter::scanBreakContinueFlags(mlir::Region &R) {
  // Phase 1: identify the break / continue flag slots.
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

  // Phase 2: any const-false store into a flag slot is the pre-loop
  // initialisation or end-of-iteration continue reset — elided.
  R.walk([&](mlir::LLVM::StoreOp S) {
    auto *Addr = S.getAddr().getDefiningOp();
    if (!Addr) return;
    bool IsFlag = BreakFlagSlots.count(Addr) || ContinueFlagSlots.count(Addr);
    if (!IsFlag) return;
    if (isConstInt(S.getValue(), 0)) {
      SuppressedOps.insert(S.getOperation());
    }
  });

  // Phase 3: classify scf.if ops.
  //  - `scf.if cond { flag := true }` → `if cond: break/continue`.
  //  - `scf.if (!flag [& !flag]) { ... }` → emit body inline.
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

    // Single store-of-true into a flag slot in the then-region.
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
          if (mlir::isa<mlir::arith::ConstantOp, mlir::LLVM::ConstantOp>(Inner))
            continue;
          OtherRealOps = true;
        }
      }
      if (FlagStore && !OtherRealOps) {
        auto *Addr = FlagStore.getAddr().getDefiningOp();
        FlagIfKind[If.getOperation()] =
            BreakFlagSlots.count(Addr) ? "break" : "continue";
        SuppressedOps.insert(FlagStore.getOperation());
        return;
      }
    }

    // Pure flag-guard: every conjunct in the condition reduces to true.
    llvm::SmallVector<mlir::Value, 2> Kept;
    gatherNonFlagConjuncts(If.getCondition(), Kept);
    if (Kept.empty())
      InlinedIfs.insert(If.getOperation());
  });
}

void Emitter::computeInlines(mlir::Region &R) {
  for (auto &B : R.getBlocks()) {
    for (auto &Op : B.getOperations()) {
      if (canInline(Op)) InlinedOps.insert(&Op);
      for (auto &SubR : Op.getRegions()) computeInlines(SubR);
    }
  }
}

// ---------------------------------------------------------------------------
// Comment / blank-line propagation from the MATLAB source
// ---------------------------------------------------------------------------

namespace {
enum class LineKind { Blank, Comment, Code };
struct LineInfo { LineKind Kind; std::string_view Body; };
LineInfo classifyLine(std::string_view Text) {
  size_t I = 0;
  while (I < Text.size() && (Text[I] == ' ' || Text[I] == '\t')) ++I;
  if (I == Text.size()) return {LineKind::Blank, {}};
  if (Text[I] != '%')   return {LineKind::Code, {}};
  ++I;
  if (I < Text.size() && Text[I] == ' ') ++I;
  return {LineKind::Comment, Text.substr(I)};
}
} // namespace

bool Emitter::emitLeadingComments(llvm::StringRef FullPath, int AfterLine,
                                  int Line, int Indent, bool) {
  if (!SM || FullPath.empty() || Line <= 0) return false;
  matlab::FileID F = SM->findFileByName(std::string_view(FullPath));
  if (F == 0) return false;

  int Start = Line;
  for (int L = Line - 1; L > AfterLine; --L) {
    auto Info = classifyLine(SM->getLineText(F, (uint32_t)L));
    if (Info.Kind == LineKind::Code) break;
    Start = L;
  }
  if (Start == Line) return false;

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
    indent(Indent);
    OS << "# " << Info.Body << "\n";
    CanEmitBlank = true;
    LastWasBlank = false;
    EmittedAny = true;
  }
  return EmittedAny;
}

void Emitter::advanceTo(mlir::Location L, int Indent) {
  mlir::FileLineColLoc FL;
  if ((FL = mlir::dyn_cast<mlir::FileLineColLoc>(L))) {
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
  std::string File = FullPath;
  if (auto Slash = File.find_last_of("/\\"); Slash != std::string::npos)
    File = File.substr(Slash + 1);
  if (File == LastLineFile && Line == LastLineNum) return;

  bool SameFile = !LastLineFile.empty() && File == LastLineFile;
  bool ForwardJump = SameFile && Line > LastLineNum;

  bool ScanEmitted = false;
  if (ForwardJump) {
    ScanEmitted = emitLeadingComments(FullPath, LastLineNum, Line, Indent);
  } else if (LastLineFile.empty()) {
    ScanEmitted = emitLeadingComments(FullPath, Line - 64, Line, Indent,
                                      /*FunctionHeader=*/true);
  }
  if (!ScanEmitted && !AtBlockStart && ForwardJump &&
      Line > LastLineNum + 1) {
    OS << "\n";
  }
  AtBlockStart = false;
  LastLineFile = File;
  LastLineNum = Line;
  (void)NoLine;  // Reserved for future use; Python has no #line directive.
}

// ---------------------------------------------------------------------------
// Globals (string constants from LowerIO via llvm.mlir.global)
// ---------------------------------------------------------------------------

std::string Emitter::constStr(mlir::LLVM::GlobalOp G) {
  auto Val = G.getValueAttr();
  if (!Val) return "";
  if (auto S = mlir::dyn_cast<mlir::StringAttr>(Val))
    return S.getValue().str();
  return "";
}

// Build a Python double-quoted string literal for `Raw` (with escapes).
static std::string buildPyStringLit(llvm::StringRef Raw) {
  std::string Out;
  Out.reserve(Raw.size() + 2);
  Out += '"';
  for (unsigned char C : Raw) {
    switch (C) {
      case '\\': Out += "\\\\"; break;
      case '"':  Out += "\\\""; break;
      case '\n': Out += "\\n"; break;
      case '\t': Out += "\\t"; break;
      case '\r': Out += "\\r"; break;
      default:
        if (C >= 0x20 && C < 0x7F) Out += (char)C;
        else {
          char Buf[8];
          snprintf(Buf, sizeof(Buf), "\\x%02x", (unsigned)C);
          Out += Buf;
        }
        break;
    }
  }
  Out += '"';
  return Out;
}

void Emitter::emitGlobal(mlir::LLVM::GlobalOp G) {
  std::string N = G.getSymName().str();
  GlobalStrs[G.getOperation()] = N;
  std::string Raw = constStr(G);
  UsedNames.insert(N);
  OS << N << " = ";
  OS << '"';
  for (unsigned char C : Raw) {
    switch (C) {
      case '\\': OS << "\\\\"; break;
      case '"':  OS << "\\\""; break;
      case '\n': OS << "\\n"; break;
      case '\t': OS << "\\t"; break;
      case '\r': OS << "\\r"; break;
      default:
        if (C >= 0x20 && C < 0x7F) OS << (char)C;
        else {
          char Buf[8];
          snprintf(Buf, sizeof(Buf), "\\x%02x", (unsigned)C);
          OS << Buf;
        }
        break;
    }
  }
  OS << "\"\n";
}

// ---------------------------------------------------------------------------
// Prolog
// ---------------------------------------------------------------------------

void Emitter::emitProlog() {
  OS << "# Generated by matlabc -emit-python. Do not edit.\n";
  OS << "import matlab_runtime as rt\n";
  OS << "import numpy as np\n";
  OS << "\n";
}

void Emitter::precomputeModuleProperties(mlir::ModuleOp) {
  // Python emitter has no per-module toggles today.
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

bool Emitter::run(mlir::ModuleOp M) {
  if (mlir::failed(mlir::verify(M))) {
    fail("MLIR verification failed before Python emission");
    return false;
  }
  precomputeModuleProperties(M);
  emitProlog();

  // Pre-emission: every defined function must have 0 or 1 results.
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
    }
  }

  // Pass 1: register every string global's Python literal so AddressOf
  // inlining can fold it directly. The top-of-file `__matlab_strN = "..."`
  // declarations the C/C++ backend emits are unnecessary in Python — the
  // literal lives at the use site instead, which reads more naturally.
  for (auto &Op : M.getBody()->getOperations()) {
    if (auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op)) {
      auto Val = G.getValueAttr();
      if (mlir::dyn_cast_or_null<mlir::StringAttr>(Val)) {
        StringGlobalLits[G.getSymName()] = buildPyStringLit(constStr(G));
        UsedNames.insert(G.getSymName().str());
      }
    }
  }

  // Pass 2: reserve the symbol of every defined function so locals don't
  // shadow them, and emit bodies. `@main` is emitted last so top-level
  // script code references already-defined functions.
  mlir::func::FuncOp MainFn;
  for (auto &Op : M.getBody()->getOperations()) {
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      UsedNames.insert(F.getSymName().str());
      if (F.getSymName() == "main") { MainFn = F; continue; }
    } else if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      UsedNames.insert(F.getSymName().str());
    }
  }

  for (auto &Op : M.getBody()->getOperations()) {
    if (Failed) break;
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      if (F == MainFn) continue;  // emit last
      emitFuncFunc(F);
    } else if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      emitLLVMFunc(F);
    }
  }
  if (!Failed && MainFn) emitFuncFunc(MainFn);

  return !Failed;
}

void Emitter::emitFuncFunc(mlir::func::FuncOp F) {
  NextId = 0;
  InlineExprs.clear();
  InlinedOps.clear();
  DirectSlots.clear();
  ArraySlots.clear();
  SuppressedOps.clear();
  BreakFlagSlots.clear();
  ContinueFlagSlots.clear();
  FlagIfKind.clear();
  InlinedIfs.clear();
  ForPatterns.clear();
  FusedForSlots.clear();
  FusedForSlotName.clear();
  LastLineFile.clear();
  LastLineNum = -1;
  // Function-local UsedNames scope: every non-main function gets a fresh
  // namespace so its parameters & locals stay short (`x`, `i`) instead of
  // accumulating `_2`, `_3` suffixes from earlier functions. The
  // module-level reservations (other functions, non-suppressed globals)
  // are restored on exit so calls to peer functions still resolve.
  llvm::StringSet<> SavedUsed;
  bool IsMain = F.getSymName() == "main";
  if (!IsMain) {
    SavedUsed = UsedNames;
    UsedNames.clear();
    // Re-reserve the symbol of every defined function and string global.
    for (auto &Op : F->getParentOfType<mlir::ModuleOp>().getBody()
                     ->getOperations()) {
      if (auto Fn = mlir::dyn_cast<mlir::func::FuncOp>(Op))
        UsedNames.insert(Fn.getSymName().str());
      else if (auto Fn = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op))
        UsedNames.insert(Fn.getSymName().str());
      else if (auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op))
        UsedNames.insert(G.getSymName().str());
    }
  }

  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());
  auto FT = F.getFunctionType();

  // Hoist @main's body to module scope — mirrors the behaviour of
  // LowerIO renaming @script -> @main so scripts become top-level code
  // in the generated Python.
  if (IsMain) {
    InMainHoist = true;
    emitRegion(F.getBody(), 0);
    InMainHoist = false;
    return;
  }

  // User function.
  OS << "def " << F.getSymName().str() << "(";
  auto &Entry = F.getBody().front();
  for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
    if (i) OS << ", ";
    auto Arg = Entry.getArgument(i);
    std::string N;
    if (auto NA = F.getArgAttrOfType<mlir::StringAttr>(i, "matlab.name"))
      N = uniqueName(NA.getValue());
    else
      N = freshName();
    Names[Arg] = N;
    OS << N;
  }
  OS << "):\n";
  if (F.getBody().front().empty() ||
      (F.getBody().front().getOperations().size() == 1 &&
       mlir::isa<mlir::func::ReturnOp>(&F.getBody().front().front()))) {
    // Empty-or-just-return body: emit a `pass`.
    indent(1);
    OS << "pass\n\n";
    if (!IsMain) UsedNames = std::move(SavedUsed);
    return;
  }
  emitRegion(F.getBody(), 1);
  OS << "\n";
  if (!IsMain) UsedNames = std::move(SavedUsed);
}

void Emitter::emitLLVMFunc(mlir::LLVM::LLVMFuncOp F) {
  NextId = 0;
  InlineExprs.clear();
  InlinedOps.clear();
  DirectSlots.clear();
  ArraySlots.clear();
  SuppressedOps.clear();
  BreakFlagSlots.clear();
  ContinueFlagSlots.clear();
  FlagIfKind.clear();
  InlinedIfs.clear();
  ForPatterns.clear();
  FusedForSlots.clear();
  FusedForSlotName.clear();
  LastLineFile.clear();
  LastLineNum = -1;
  llvm::StringSet<> SavedUsed = UsedNames;
  UsedNames.clear();
  for (auto &Op : F->getParentOfType<mlir::ModuleOp>().getBody()
                   ->getOperations()) {
    if (auto Fn = mlir::dyn_cast<mlir::func::FuncOp>(Op))
      UsedNames.insert(Fn.getSymName().str());
    else if (auto Fn = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op))
      UsedNames.insert(Fn.getSymName().str());
    else if (auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op))
      UsedNames.insert(G.getSymName().str());
  }
  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());
  auto FT = F.getFunctionType();
  OS << "def " << F.getSymName().str() << "(";
  auto &Entry = F.getBody().front();
  for (unsigned i = 0; i < FT.getNumParams(); ++i) {
    if (i) OS << ", ";
    auto Arg = Entry.getArgument(i);
    std::string N = freshName();
    Names[Arg] = N;
    OS << N;
  }
  OS << "):\n";
  if (F.getBody().front().empty()) {
    indent(1); OS << "pass\n\n";
    UsedNames = std::move(SavedUsed);
    return;
  }
  emitRegion(F.getBody(), 1);
  OS << "\n";
  UsedNames = std::move(SavedUsed);
}

// ---------------------------------------------------------------------------
// Region / block / op dispatch
// ---------------------------------------------------------------------------

void Emitter::emitRegion(mlir::Region &R, int Indent) {
  for (auto &B : R.getBlocks())
    emitBlock(B, Indent);
}

void Emitter::emitBlock(mlir::Block &B, int Indent) {
  AtBlockStart = true;
  // Track whether this block emitted any real (non-comment) statement; if
  // not and the block is an inner region body, we need a `pass` placeholder
  // for Python's grammar. We don't know the parent here cheaply, so let
  // the parent emit `pass` when its region turned out empty — easier.
  for (auto &Op : B.getOperations())
    emitOp(Op, Indent);
}

// Count real (non-condition, non-inlined-no-op) statements a block will
// emit. scf.yield with one or more operands counts as that many
// statements, since each operand becomes one `result_var = expr` line in
// the parent's scope.
static int countEmittedStmts(mlir::Block &B,
                             const llvm::DenseSet<mlir::Operation *> &Inlined,
                             const llvm::DenseSet<mlir::Operation *> &Suppressed) {
  int N = 0;
  for (auto &Op : B.getOperations()) {
    if (mlir::isa<mlir::scf::ConditionOp>(&Op)) continue;
    if (auto Y = mlir::dyn_cast<mlir::scf::YieldOp>(&Op)) {
      N += (int)Y.getNumOperands();
      continue;
    }
    if (Inlined.count(&Op)) continue;
    if (Suppressed.count(&Op)) continue;
    ++N;
  }
  return N;
}

void Emitter::emitOp(mlir::Operation &Op, int Indent) {
  llvm::StringRef Name = Op.getName().getStringRef();

  if (InlinedOps.count(&Op)) return;
  if (SuppressedOps.count(&Op)) return;

  advanceTo(Op.getLoc(), Indent);

  // --- llvm.mlir.zero / llvm.mlir.null --------------------------------
  if (mlir::isa<mlir::LLVM::ZeroOp>(Op)) {
    std::string N = this->name(Op.getResult(0));
    indent(Indent);
    OS << N << " = 0\n";
    return;
  }

  // --- llvm.mlir.constant ---------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Op)) {
    std::string N = this->name(C.getResult());
    indent(Indent);
    OS << N << " = ";
    auto V = C.getValue();
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V))      OS << formatIntAttr(IA);
    else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V))   OS << formatFloatAttr(FA);
    else OS << "0  # unknown const";
    OS << "\n";
    return;
  }

  // --- arith.constant --------------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
    std::string N = this->name(C.getResult());
    indent(Indent);
    OS << N << " = ";
    auto V = C.getValue();
    if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V))        OS << formatFloatAttr(FA);
    else if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V)) OS << formatIntAttr(IA);
    else OS << "0  # unknown const";
    OS << "\n";
    return;
  }

  // --- func.return / llvm.return --------------------------------------
  if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
    if (InMainHoist) {
      // Top-level main body: drop the final `return` — Python module
      // scope has no `return`. The sentinel `0` the pipeline produces
      // as the script's exit status is dropped with it.
      return;
    }
    if (R.getNumOperands() == 0) {
      // Void return at the end of the body is the implicit Python
      // function exit; only emit `return` when it sits before more code.
      if (R->getNextNode() == nullptr &&
          R->getBlock() == &R->getParentRegion()->back())
        return;
      indent(Indent); OS << "return\n";
    } else {
      indent(Indent);
      OS << "return " << this->stmtExpr(R.getOperand(0)) << "\n";
    }
    return;
  }
  if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
    if (InMainHoist) return;
    if (R.getNumOperands() == 0) {
      if (R->getNextNode() == nullptr &&
          R->getBlock() == &R->getParentRegion()->back())
        return;
      indent(Indent); OS << "return\n";
    } else {
      indent(Indent);
      OS << "return " << this->stmtExpr(R.getOperand(0)) << "\n";
    }
    return;
  }

  // --- llvm.call / func.call ------------------------------------------
  if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    indent(Indent);
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult());
      OS << N << " = ";
    }
    if (auto Callee = Call.getCallee()) {
      // For user-string runtime helpers, drop the (str_ptr, str_len) tail
      // length operand — Python strings carry their length, and the
      // matching runtime stub keeps `n` optional for back-compat.
      unsigned LengthIdx = ~0u;
      bool DropLen = false;
      if (Callee->starts_with("matlab_")) {
        unsigned Idx;
        if (calleeHasDroppableLengthArg(
                Callee->drop_front(strlen("matlab_")), Idx)) {
          LengthIdx = Idx;
          DropLen = true;
        }
      }
      // `rt.disp_str("literal")` is byte-identical to `print("literal")`,
      // so collapse it. Detect by callee + a single (post-length-drop)
      // operand that traces to a string-global addressof. We keep the
      // runtime call for `disp_str(<variable>)` since the runtime path
      // also handles non-`str` payloads.
      auto isStringLiteralOperand = [&](mlir::Value V) -> bool {
        if (auto *D = V.getDefiningOp())
          if (auto A = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(D))
            return StringGlobalLits.count(A.getGlobalName()) > 0;
        return false;
      };
      if ((*Callee == "matlab_disp_str" ||
           *Callee == "matlab_string_disp") &&
          Call.getNumResults() == 0 && Call.getNumOperands() >= 1 &&
          isStringLiteralOperand(Call.getOperand(0))) {
        OS << "print(" << this->stmtExpr(Call.getOperand(0)) << ")\n";
        return;
      }
      OS << remapRuntimeCallee(*Callee) << "(";
      bool First = true;
      for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
        if (DropLen && i == LengthIdx) continue;
        if (!First) OS << ", ";
        First = false;
        OS << this->stmtExpr(Call.getOperand(i));
      }
      OS << ")\n";
    } else {
      // Indirect call: first operand is the callable.
      OS << this->exprFor(Call.getOperand(0)) << "(";
      for (unsigned i = 1; i < Call.getNumOperands(); ++i) {
        if (i > 1) OS << ", ";
        OS << this->stmtExpr(Call.getOperand(i));
      }
      OS << ")\n";
    }
    return;
  }
  if (auto Call = mlir::dyn_cast<mlir::func::CallOp>(Op)) {
    indent(Indent);
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult(0));
      OS << N << " = ";
    }
    OS << Call.getCallee().str() << "(";
    for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
      if (i) OS << ", ";
      OS << this->stmtExpr(Call.getOperand(i));
    }
    OS << ")\n";
    return;
  }

  // --- llvm.mlir.addressof --------------------------------------------
  if (auto A = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(Op)) {
    std::string N = this->name(A.getResult());
    indent(Indent);
    auto It = StringGlobalLits.find(A.getGlobalName());
    if (It != StringGlobalLits.end()) {
      OS << N << " = " << It->second << "\n";
      SuppressedGlobals.insert(A.getGlobalName());
    } else {
      OS << N << " = " << A.getGlobalName().str() << "\n";
    }
    return;
  }

  // --- arith binary ops ------------------------------------------------
  auto emitBin = [&](const char *CC) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << N << " = " << this->exprFor(Op.getOperand(0)) << " " << CC << " "
       << this->exprFor(Op.getOperand(1)) << "\n";
  };
  if (mlir::isa<mlir::arith::AddFOp>(Op)) { emitBin("+"); return; }
  if (mlir::isa<mlir::arith::SubFOp>(Op)) { emitBin("-"); return; }
  if (mlir::isa<mlir::arith::MulFOp>(Op)) { emitBin("*"); return; }
  if (mlir::isa<mlir::arith::DivFOp>(Op)) { emitBin("/"); return; }
  if (mlir::isa<mlir::arith::AddIOp>(Op)) { emitBin("+"); return; }
  if (mlir::isa<mlir::arith::SubIOp>(Op)) { emitBin("-"); return; }
  if (mlir::isa<mlir::arith::MulIOp>(Op)) { emitBin("*"); return; }

  if (auto A = mlir::dyn_cast<mlir::arith::AndIOp>(Op)) {
    if (isI1(A.getType())) { emitBin("and"); return; }
    emitBin("&"); return;
  }
  if (auto O = mlir::dyn_cast<mlir::arith::OrIOp>(Op)) {
    if (isI1(O.getType())) { emitBin("or"); return; }
    emitBin("|"); return;
  }
  if (auto X = mlir::dyn_cast<mlir::arith::XOrIOp>(Op)) {
    if (isI1(X.getType())) { emitBin("!="); return; }
    emitBin("^"); return;
  }

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
    OS << N << " = " << this->exprFor(C.getLhs()) << " " << CC
       << " " << this->exprFor(C.getRhs()) << "\n";
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
    OS << N << " = " << this->exprFor(C.getLhs()) << " " << CC
       << " " << this->exprFor(C.getRhs()) << "\n";
    return;
  }

  // --- arith casts ----------------------------------------------------
  if (mlir::isa<mlir::arith::SIToFPOp, mlir::arith::UIToFPOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << N << " = float(" << this->stmtExpr(Op.getOperand(0)) << ")\n";
    return;
  }
  if (mlir::isa<mlir::arith::FPToSIOp, mlir::arith::FPToUIOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << N << " = int(" << this->stmtExpr(Op.getOperand(0)) << ")\n";
    return;
  }
  // ext/trunc are no-ops in Python.
  if (mlir::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                mlir::arith::TruncIOp, mlir::arith::TruncFOp,
                mlir::arith::ExtFOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << N << " = " << this->stmtExpr(Op.getOperand(0)) << "\n";
    return;
  }

  // --- arith.select ---------------------------------------------------
  if (auto S = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
    indent(Indent);
    std::string N = this->name(S.getResult());
    OS << N << " = " << this->exprFor(S.getTrueValue()) << " if "
       << this->exprFor(S.getCondition()) << " else "
       << this->exprFor(S.getFalseValue()) << "\n";
    return;
  }

  // --- llvm.alloca / load / store -------------------------------------
  if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Op)) {
    std::string Hint;
    if (auto NA = A->getAttrOfType<mlir::StringAttr>("matlab.name"))
      Hint = NA.getValue().str();
    std::string SlotName;
    if (!Hint.empty()) {
      std::string Sane = sanitizeIdent(Hint);
      if (UsedNames.find(Sane) != UsedNames.end())
        SlotName = uniqueName(Sane + "_slot");
      else
        SlotName = uniqueName(Sane);
    } else {
      SlotName = uniqueName("slot");
    }
    mlir::Type ET = A.getElemType();
    bool IsArray = mlir::isa<mlir::LLVM::LLVMArrayType>(ET);

    if (IsArray) {
      auto AT = mlir::cast<mlir::LLVM::LLVMArrayType>(ET);
      uint64_t N0 = AT.getNumElements();
      Names[A.getResult()] = SlotName;
      ArraySlots[A.getOperation()] = SlotName;

      // Matrix-literal init pattern from LowerTensorOps::materializeMat:
      // every alloca user is a GEP-with-constant-index-then-store (or a
      // direct store at idx 0). When all N indices 0..N-1 are filled
      // exactly once with stmtExpr-able values, collapse into a single
      // `slot = [v0, v1, ...]` Python list literal and absorb the
      // GEP/store ops.
      auto getConstIdx = [](mlir::LLVM::GEPOp G, uint64_t &Out) -> bool {
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
        // The legitimate post-init consumer (e.g. matlab_mat_from_buf).
        if (mlir::isa<mlir::LLVM::CallOp>(U)) continue;
        InitOK = false;
        break;
      }
      if (InitOK && Filled == N0)
        for (auto V : InitVals) if (!V) { InitOK = false; break; }
      if (InitOK && Filled == N0) {
        indent(Indent);
        OS << SlotName << " = [";
        for (uint64_t i = 0; i < N0; ++i) {
          if (i) OS << ", ";
          OS << this->stmtExpr(InitVals[i]);
        }
        OS << "]\n";
        for (auto *Op2 : AbsorbedOps) SuppressedOps.insert(Op2);
        return;
      }

      // Fallback: zero-filled list, individual stores follow.
      indent(Indent);
      OS << SlotName << " = [0.0] * " << N0 << "\n";
      return;
    }

    // For-loop fused slot: the surrounding `for iv in range(...):`
    // emitter declares the IV directly. Skip the alloca and any
    // pre-store; future loads/stores resolve to the IV name.
    if (FusedForSlots.count(A.getOperation())) {
      auto NIt = FusedForSlotName.find(A.getOperation());
      if (NIt != FusedForSlotName.end()) {
        Names[A.getResult()] = NIt->second;
        DirectSlots[A.getOperation()] = NIt->second;
      }
      return;
    }

    // Scalar slot: every use we've seen from Mem2RegLite is load/store.
    // Bind the alloca SSA value to the slot name; downstream load/store
    // treat it as a direct Python variable.
    Names[A.getResult()] = SlotName;
    DirectSlots[A.getOperation()] = SlotName;

    // Drop the pre-declaration `slot = 0` when the slot's first use is
    // an unconditional store. Python doesn't need pre-declaration —
    // assignment creates the binding — so the zero-init is dead weight
    // when overwritten before any read.
    //
    // Conservative dominance check: walk the parent block in program
    // order; at each op, look for any operand that is the slot. If the
    // op directly stores into the slot, the init is dead. If the op
    // directly loads, keep the init. If the op has a nested region
    // (scf.if / scf.while) that uses the slot, the use sits under a
    // conditional / loop and may be skipped at runtime — keep the init
    // so post-loop reads bind safely.
    // True when every dynamic execution of `B` performs at least one
    // unconditional Store into SlotV before reading from it (or before
    // returning). Recurses through scf.if when both branches store; that
    // covers MATLAB return-slot patterns like `if c; y = a; else y = b;
    // end` where the return slot is fully assigned regardless of branch.
    std::function<bool(mlir::Block &, mlir::Value)> blockAlwaysStores =
        [&](mlir::Block &B, mlir::Value SlotV) -> bool {
      for (auto &Op2 : B) {
        if (auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op2))
          if (St.getAddr() == SlotV) return true;
        if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Op2)) {
          if (If.getElseRegion().empty()) continue;
          if (blockAlwaysStores(If.getThenRegion().front(), SlotV) &&
              blockAlwaysStores(If.getElseRegion().front(), SlotV))
            return true;
        }
      }
      return false;
    };

    bool DropInit = false;
    {
      mlir::Block *ABlock = A->getBlock();
      mlir::Value SlotV = A.getResult();
      for (auto It = mlir::Block::iterator(A->getNextNode());
           It != ABlock->end(); ++It) {
        mlir::Operation *DirectUser = nullptr;
        for (mlir::OpOperand &U : SlotV.getUses()) {
          if (U.getOwner() == &*It) { DirectUser = U.getOwner(); break; }
        }
        if (DirectUser) {
          if (auto St = mlir::dyn_cast<mlir::LLVM::StoreOp>(DirectUser)) {
            if (St.getAddr() == SlotV) { DropInit = true; break; }
          }
          break;  // load (or other consumer) — keep the init.
        }
        bool NestedUse = false;
        for (auto &Reg : It->getRegions()) {
          Reg.walk([&](mlir::Operation *Inner) {
            for (mlir::Value Opnd : Inner->getOperands())
              if (Opnd == SlotV) NestedUse = true;
          });
          if (NestedUse) break;
        }
        if (NestedUse) {
          // Allow the nested-use case when the op is an scf.if whose
          // both branches unconditionally store into the slot — the
          // pre-init is then dead weight.
          if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(&*It)) {
            if (!If.getElseRegion().empty() &&
                blockAlwaysStores(If.getThenRegion().front(), SlotV) &&
                blockAlwaysStores(If.getElseRegion().front(), SlotV)) {
              DropInit = true;
            }
          }
          break;
        }
      }
    }
    if (DropInit) return;

    indent(Indent);
    OS << SlotName << " = 0\n";
    return;
  }

  if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
    mlir::Operation *AddrDef = L.getAddr().getDefiningOp();
    std::string N;
    if (AddrDef) {
      if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(AddrDef)) {
        if (auto NA = A->getAttrOfType<mlir::StringAttr>("matlab.name")) {
          N = uniqueName(NA.getValue().str() + "_v");
          Names[L.getResult()] = N;
        }
      }
    }
    if (N.empty()) N = this->name(L.getResult());
    indent(Indent);
    if (AddrDef && DirectSlots.count(AddrDef)) {
      OS << N << " = " << DirectSlots[AddrDef] << "\n";
    } else if (AddrDef && ArraySlots.count(AddrDef)) {
      // Array-slot load should always come through a GEP; the GEP
      // inlining handles the indexing. A raw load of the array pointer
      // doesn't make sense here — just bind to the slot name.
      OS << N << " = " << ArraySlots[AddrDef] << "\n";
    } else {
      // Non-direct (e.g. GEP address): the address expression IS the
      // place to load from — emit `N = <addr_expr>`. Works for
      // `arr[idx]` shaped GEPs.
      OS << N << " = " << this->stmtExpr(L.getAddr()) << "\n";
    }
    return;
  }
  if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
    mlir::Operation *AddrDef = S.getAddr().getDefiningOp();
    indent(Indent);
    if (AddrDef && DirectSlots.count(AddrDef)) {
      OS << DirectSlots[AddrDef] << " = " << this->stmtExpr(S.getValue())
         << "\n";
    } else if (AddrDef && ArraySlots.count(AddrDef)) {
      // Store straight to the array alloca's base pointer = element 0.
      // LLVM allows writing through an alloca address without a GEP; in
      // Python this needs explicit indexing.
      OS << ArraySlots[AddrDef] << "[0] = "
         << this->stmtExpr(S.getValue()) << "\n";
    } else {
      // Non-direct: address is a GEP-like expression (`arr[idx]`).
      OS << this->stmtExpr(S.getAddr()) << " = "
         << this->stmtExpr(S.getValue()) << "\n";
    }
    return;
  }

  // --- llvm.getelementptr ---------------------------------------------
  if (auto G = mlir::dyn_cast<mlir::LLVM::GEPOp>(Op)) {
    // If not inlined, materialise as an expression local (same text as
    // the inline form).
    std::string Base;
    if (auto *D = G.getBase().getDefiningOp()) {
      auto It = ArraySlots.find(D);
      if (It != ArraySlots.end()) Base = It->second;
    }
    if (Base.empty()) Base = this->exprFor(G.getBase());
    std::string Idx;
    bool First = true;
    for (auto I : G.getIndices()) {
      std::string Term;
      if (auto Vv = llvm::dyn_cast<mlir::Value>(I))
        Term = dropOuterParens(this->exprFor(Vv));
      else if (auto IA = llvm::dyn_cast<mlir::IntegerAttr>(I))
        Term = std::to_string(IA.getInt());
      else continue;
      if (First) { Idx = Term; First = false; }
      else       { Idx = "(" + Idx + " + " + Term + ")"; }
    }
    if (Idx.empty()) Idx = "0";
    std::string N = this->name(G.getResult());
    indent(Indent);
    OS << N << " = " << Base << "[" << Idx << "]\n";
    return;
  }

  // --- scf.if ---------------------------------------------------------
  if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Op)) {
    // Break/continue short-circuit: `scf.if cond { flag := true }` came
    // from a MATLAB `if cond; break; end`. Re-emit as a one-liner.
    auto FK = FlagIfKind.find(&Op);
    if (FK != FlagIfKind.end()) {
      indent(Indent);
      OS << "if " << this->stmtExpr(If.getCondition()) << ":\n";
      indent(Indent + 1);
      OS << FK->second << "\n";
      return;
    }
    // Pure flag-guard wrapping real body: skip the `if (...)` and emit
    // the then-region inline at the parent's indent.
    if (InlinedIfs.count(&Op)) {
      emitRegion(If.getThenRegion(), Indent);
      return;
    }
    // Static-condition folding. Polymorphic dispatch in the frontend
    // bakes nargin into a constant comparison (`if 2 == 2:`); detect
    // that here and emit only the live branch.
    int Folded = evalConstCond(If.getCondition());
    if (Folded == 1) {
      emitRegion(If.getThenRegion(), Indent);
      return;
    }
    if (Folded == 0) {
      if (!If.getElseRegion().empty())
        emitRegion(If.getElseRegion(), Indent);
      return;
    }

    // Detect an else-region whose only meaningful op is a single nested
    // scf.if (the shape MATLAB `elseif` lowers to). Walk the chain so
    // a cascade of elseifs collapses to one `elif <cond>:` per branch
    // instead of a deepening `else: if ...` ladder.
    auto findElseElif = [&](mlir::scf::IfOp Parent)
        -> mlir::scf::IfOp {
      if (Parent.getElseRegion().empty()) return {};
      auto &EBlock = Parent.getElseRegion().front();
      mlir::scf::IfOp Inner;
      for (auto &Inn : EBlock.getOperations()) {
        if (mlir::isa<mlir::scf::YieldOp>(&Inn)) {
          // Void if (no results): yield is a no-op. Result-bearing if:
          // the yield must forward exactly the inner if's results so
          // chaining is safe.
          if (auto Y = mlir::dyn_cast<mlir::scf::YieldOp>(Inn)) {
            if (Y.getNumOperands() == 0) continue;
            if (!Inner) return {};
            if (Y.getNumOperands() != Inner.getNumResults()) return {};
            for (unsigned i = 0; i < Y.getNumOperands(); ++i)
              if (Y.getOperand(i) != Inner.getResult(i)) return {};
            continue;
          }
        }
        if (InlinedOps.count(&Inn)) continue;
        if (SuppressedOps.count(&Inn)) continue;
        if (auto NestedIf = mlir::dyn_cast<mlir::scf::IfOp>(Inn)) {
          if (Inner) return {};  // multiple ifs — not a clean elif
          // Skip elif folding for the special break/continue forms;
          // those need their own scf.if emit path.
          if (FlagIfKind.count(NestedIf.getOperation())) return {};
          if (InlinedIfs.count(NestedIf.getOperation())) return {};
          Inner = NestedIf;
          continue;
        }
        return {};  // some other op rules out clean elif folding
      }
      return Inner;
    };

    indent(Indent);
    OS << "if " << this->stmtExpr(If.getCondition()) << ":\n";
    if (countEmittedStmts(If.getThenRegion().front(), InlinedOps,
                           SuppressedOps) == 0) {
      indent(Indent + 1); OS << "pass\n";
    } else {
      emitRegion(If.getThenRegion(), Indent + 1);
    }

    // Walk the else-chain, folding each cleanly-chainable nested scf.if
    // into an `elif`. The inner if's result SSA values are aliased to
    // the parent's result-locals so each branch's yield writes to the
    // shared variable.
    mlir::scf::IfOp Cur = If;
    while (true) {
      mlir::scf::IfOp Next = findElseElif(Cur);
      if (!Next) break;
      // Statically-folded inner condition: lift the live branch up to
      // the parent's indent and stop chaining.
      int InnerFold = evalConstCond(Next.getCondition());
      if (InnerFold == 1) {
        for (unsigned i = 0; i < Next.getNumResults(); ++i)
          Names[Next.getResult(i)] = this->name(Cur.getResult(i));
        emitRegion(Next.getThenRegion(), Indent);
        return;
      }
      if (InnerFold == 0) {
        for (unsigned i = 0; i < Next.getNumResults(); ++i)
          Names[Next.getResult(i)] = this->name(Cur.getResult(i));
        if (!Next.getElseRegion().empty())
          emitRegion(Next.getElseRegion(), Indent);
        return;
      }
      // Alias the inner if's result SSA values to the parent's result-
      // local names so the inner yield assigns to the shared variable.
      for (unsigned i = 0; i < Next.getNumResults(); ++i)
        Names[Next.getResult(i)] = this->name(Cur.getResult(i));
      indent(Indent);
      OS << "elif " << this->stmtExpr(Next.getCondition()) << ":\n";
      if (countEmittedStmts(Next.getThenRegion().front(), InlinedOps,
                             SuppressedOps) == 0) {
        indent(Indent + 1); OS << "pass\n";
      } else {
        emitRegion(Next.getThenRegion(), Indent + 1);
      }
      Cur = Next;
    }

    if (!Cur.getElseRegion().empty() &&
        countEmittedStmts(Cur.getElseRegion().front(), InlinedOps,
                           SuppressedOps) > 0) {
      indent(Indent);
      OS << "else:\n";
      emitRegion(Cur.getElseRegion(), Indent + 1);
    }
    return;
  }

  // scf.yield inside scf.if / scf.while: assign to the parent's result
  // / iter-arg locals.
  if (auto Y = mlir::dyn_cast<mlir::scf::YieldOp>(Op)) {
    auto *Parent = Op.getParentOp();
    if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Parent)) {
      for (unsigned i = 0; i < Y.getNumOperands(); ++i) {
        indent(Indent);
        OS << this->name(If.getResult(i)) << " = "
           << this->stmtExpr(Y.getOperand(i)) << "\n";
      }
      return;
    }
    if (auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Parent)) {
      for (unsigned i = 0; i < Y.getNumOperands(); ++i) {
        auto BA = W.getBefore().front().getArgument(i);
        indent(Indent);
        OS << this->name(BA) << " = " << this->stmtExpr(Y.getOperand(i))
           << "\n";
      }
      return;
    }
    return;
  }

  // --- scf.while ------------------------------------------------------
  if (auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Op)) {
    auto &Before = W.getBefore().front();
    auto &After = W.getAfter().front();

    // Did the pre-scan flag this loop as a canonical MATLAB for-loop?
    auto FPIt = ForPatterns.find(W.getOperation());
    if (FPIt != ForPatterns.end()) {
      const ForLoopInfo &Info = FPIt->second;
      Names[Before.getArgument(0)] = Info.IvName;
      InlineExprs[After.getArgument(0)] = Info.IvName;
      if (W.getNumResults() == 1)
        InlineExprs[W.getResult(0)] = Info.IvName;

      // Prefer Python's int `range(...)` when init/end/step are integer
      // literals; otherwise fall back to a runtime helper so floating-
      // point steps and runtime-computed bounds still iterate correctly.
      long long IInit, IEnd, IStep;
      bool IntForm = forBoundsAreIntLiterals(Info, IInit, IEnd, IStep) &&
                     IStep != 0;
      indent(Indent);
      if (IntForm) {
        long long Stop = (IStep > 0) ? IEnd + 1 : IEnd - 1;
        OS << "for " << Info.IvName << " in range("
           << IInit << ", " << Stop;
        if (IStep != 1) OS << ", " << IStep;
        OS << "):\n";
      } else {
        OS << "for " << Info.IvName << " in rt.frange("
           << this->stmtExpr(Info.Init) << ", "
           << this->stmtExpr(Info.End) << ", "
           << this->stmtExpr(Info.Step) << "):\n";
      }
      // Body: emit every after-region op except the absorbed addf and
      // the trailing yield (already in SuppressedOps via scanForLoopPatterns).
      int EmittedStmts = 0;
      for (auto &Inner : After.getOperations()) {
        if (mlir::isa<mlir::scf::YieldOp>(&Inner)) continue;
        if (InlinedOps.count(&Inner)) continue;
        if (SuppressedOps.count(&Inner)) continue;
        ++EmittedStmts;
      }
      if (EmittedStmts == 0) {
        indent(Indent + 1); OS << "pass\n";
      } else {
        for (auto &Inner : After.getOperations())
          emitOp(Inner, Indent + 1);
      }
      return;
    }

    for (unsigned i = 0; i < W.getInits().size(); ++i) {
      auto BA = Before.getArgument(i);
      std::string N = freshName();
      Names[BA] = N;
      indent(Indent);
      OS << N << " = " << this->stmtExpr(W.getInits()[i]) << "\n";
    }
    for (unsigned i = 0; i < W.getNumResults(); ++i) {
      auto BA = Before.getArgument(i);
      Names[W.getResult(i)] = Names[BA];
    }

    bool BeforeIsCondOnly = true;
    for (auto &Inner : Before.getOperations()) {
      if (mlir::isa<mlir::scf::ConditionOp>(Inner)) continue;
      if (InlinedOps.count(&Inner)) continue;
      BeforeIsCondOnly = false;
      break;
    }

    // Render a loop condition with break/continue-flag inversions
    // stripped: `while cond & !did_break` → `while cond` once we're
    // emitting native `break` / `continue`. An entirely stripped
    // condition collapses to `True`.
    auto emitStrippedCond = [&](mlir::Value V) -> std::string {
      llvm::SmallVector<mlir::Value, 2> Parts;
      gatherNonFlagConjuncts(V, Parts);
      if (Parts.empty()) return "True";
      std::string Out;
      for (unsigned i = 0; i < Parts.size(); ++i) {
        if (i) Out += " and ";
        Out += this->stmtExpr(Parts[i]);
      }
      return Out;
    };

    if (BeforeIsCondOnly) {
      auto Cond = mlir::cast<mlir::scf::ConditionOp>(Before.getTerminator());
      for (unsigned i = 0; i < Cond.getArgs().size(); ++i) {
        auto AA = After.getArgument(i);
        InlineExprs[AA] = this->exprFor(Cond.getArgs()[i]);
      }
      indent(Indent);
      OS << "while " << emitStrippedCond(Cond.getCondition()) << ":\n";
      int EmittedStmts = 0;
      for (auto &Inner : After.getOperations()) {
        if (!InlinedOps.count(&Inner) &&
            !mlir::isa<mlir::scf::YieldOp, mlir::scf::ConditionOp>(&Inner))
          ++EmittedStmts;
      }
      if (EmittedStmts == 0 && After.getOperations().size() <= 1) {
        indent(Indent + 1); OS << "pass\n";
      } else {
        for (auto &Inner : After.getOperations())
          emitOp(Inner, Indent + 1);
      }
      return;
    }

    indent(Indent);
    OS << "while True:\n";
    for (auto &Inner : Before.getOperations()) {
      if (auto Cond = mlir::dyn_cast<mlir::scf::ConditionOp>(Inner)) {
        std::string CondStr = emitStrippedCond(Cond.getCondition());
        if (CondStr != "True") {
          indent(Indent + 1);
          OS << "if not (" << CondStr << "): break\n";
        }
        for (unsigned i = 0; i < Cond.getArgs().size(); ++i) {
          auto AA = After.getArgument(i);
          InlineExprs[AA] = this->exprFor(Cond.getArgs()[i]);
        }
        continue;
      }
      emitOp(Inner, Indent + 1);
    }
    for (auto &Inner : After.getOperations())
      emitOp(Inner, Indent + 1);
    return;
  }

  // --- Fallback -------------------------------------------------------
  indent(Indent);
  OS << "# UNSUPPORTED: " << Name.str() << "\n";
  fail(("unsupported op in emitter: " + Name).str());
}

} // namespace

std::string emitPython(mlir::ModuleOp M, bool NoLine,
                       const matlab::SourceManager *SM) {
  std::ostringstream OSS;
  Emitter E(OSS, NoLine, SM);
  if (!E.run(M)) return {};
  return OSS.str();
}

} // namespace mlirgen
} // namespace matlab
