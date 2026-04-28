// Emits TypeScript source from an MLIR ModuleOp whose ops have already
// been lowered to a small, closed set: func / arith / scf / cf / llvm.call
// / llvm.alloca / llvm.load / llvm.store / llvm.mlir.global /
// llvm.mlir.addressof plus outlined llvm.func bodies (parfor / anonymous
// functions).
//
// Companion to EmitPython.cpp: structure mirrors that emitter closely
// but targets TypeScript. The emitted file imports `matlab_runtime` (a
// numpy-ts-backed shim) and runs on Bun / `tsx` / Node + `ts-node`.
//
// Differences from the Python emitter that matter at the source level:
//   - Statements end with `;`.
//   - Block delimiters are `{ }` instead of indentation.
//   - Variable declarations use `let`.
//   - Function declarations use `function name(...) { ... }`.
//   - Class / `extends` / `constructor` / `static` / `get` keywords.
//   - TypeScript can't overload binary operators, so matrix `A + B`
//     becomes `A.add(B)` (NDArray method) instead of `A + B`.
//   - User-defined operator methods (`plus` / `eq` / ...) keep their
//     MATLAB names; call sites become `a.plus(b)` / `a.eq(b)`.

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
#include "llvm/ADT/SetVector.h"
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

/// One class method captured during the pre-pass. Same layout as
/// EmitPython.cpp's ClassMethodInfo, with TypeScript-specific naming.
struct ClassMethodInfo {
  mlir::func::FuncOp Func;
  std::string ClassName;
  enum Kind { Ctor, Method, Get, Operator, Static } Kind = Method;
  std::string EmitName;   // TypeScript method name (e.g. `constructor`,
                          // `deposit`, `Overdrawn`, `eq`).
  std::string OpSpelling; // Unused — kept for parity with EmitPython.
};

struct ClassDef {
  std::string Super;
  llvm::SmallVector<ClassMethodInfo, 8> Methods;
};

struct ForLoopInfo {
  mlir::Value Init;
  mlir::Value End;
  mlir::Value Step;
  bool IsDecreasing = false;
  mlir::Operation *AddOp = nullptr;
  mlir::Operation *YieldOp = nullptr;
  mlir::Operation *BindStore = nullptr;
  mlir::Operation *SlotAlloca = nullptr;
  bool FuseSlot = false;
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
  void indent(int N) { for (int i = 0; i < N; ++i) OS << "  "; }
  std::string constStr(mlir::LLVM::GlobalOp G);
  void fail(llvm::StringRef Msg) {
    if (!Failed)
      std::cerr << "error: emit-typescript: " << Msg.str() << "\n";
    Failed = true;
  }
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
  void scanBreakContinueFlags(mlir::Region &R);
  bool isFlagInversion(mlir::Value V);
  void gatherNonFlagConjuncts(mlir::Value V,
      llvm::SmallVectorImpl<mlir::Value> &Out);

  // --- For-loop pattern detection ---------------------------------------
  bool matchForPattern(mlir::scf::WhileOp W, ForLoopInfo &Info);
  void scanForLoopPatterns(mlir::Region &R);

  static bool tryEvalIntLiteral(mlir::Value V, long long &Out);
  static bool forBoundsAreIntLiterals(const ForLoopInfo &Info,
                                      long long &Init, long long &End,
                                      long long &Step);

  // --- Callee remap: llvm.call @matlab_foo -> rt.foo --------------------
  static std::string remapRuntimeCallee(llvm::StringRef Name);

  // True when the runtime helper named (without `matlab_` prefix) takes
  // a string-length operand we can drop in TypeScript (where strings
  // carry their own length).
  static bool calleeHasDroppableLengthArg(llvm::StringRef Suffix,
                                          unsigned &LengthArgIdx);

  static int evalConstCond(mlir::Value V);

  // Try to rewrite a `matlab_<suffix>` call as an inline numpy-ts /
  // NDArray-method expression. Returns true on success.
  bool tryRewriteAsNumpy(mlir::LLVM::CallOp C, std::string &Out);

  bool tryRewriteObjGet(mlir::LLVM::CallOp C, std::string &Out);
  bool tryRewriteObjSet(mlir::LLVM::CallOp C, std::string &Out);

  void collectClasses(mlir::ModuleOp M);

  void emitClassBlock(llvm::StringRef ClassName, const ClassDef &CI);

  void emitClassMethod(const ClassMethodInfo &CMI, int Indent);

  // Try to rewrite `<ClassName>__<method>(args...)` as one of:
  //   - constructor:  `new ClassName(args)`
  //   - regular call: `args[0].method(args[1:])`
  //   - operator:     `args[0].method(args[1])` (TS has no operator
  //                   overloading, so we use the method directly)
  //   - get-property: `args[0].PropName`
  bool tryRewriteAsClassCall(llvm::StringRef Callee,
                              mlir::ValueRange Operands,
                              std::string &Out);

  std::ostream &OS;
  bool NoLine;
  const matlab::SourceManager *SM;
  bool Failed = false;

  llvm::DenseMap<mlir::Value, std::string> Names;
  llvm::DenseMap<mlir::Operation *, std::string> GlobalStrs;
  llvm::StringSet<> UsedNames;
  llvm::DenseMap<mlir::Value, std::string> InlineExprs;
  llvm::DenseSet<mlir::Operation *> InlinedOps;
  llvm::DenseMap<mlir::Operation *, std::string> DirectSlots;
  llvm::DenseMap<mlir::Operation *, std::string> ArraySlots;
  llvm::DenseSet<mlir::Operation *> SuppressedOps;
  llvm::DenseSet<mlir::Operation *> BreakFlagSlots;
  llvm::DenseSet<mlir::Operation *> ContinueFlagSlots;
  llvm::DenseMap<mlir::Operation *, const char *> FlagIfKind;
  llvm::DenseSet<mlir::Operation *> InlinedIfs;
  llvm::StringMap<std::string> StringGlobalLits;
  llvm::StringSet<> SuppressedGlobals;
  llvm::DenseMap<mlir::Operation *, ForLoopInfo> ForPatterns;
  llvm::DenseSet<mlir::Operation *> FusedForSlots;
  llvm::DenseMap<mlir::Operation *, std::string> FusedForSlotName;
  llvm::StringMap<ClassDef> Classes;
  llvm::StringMap<ClassMethodInfo> CalleeIndex;
  int NextId = 0;

  std::string LastLineFile;
  int LastLineNum = -1;
  bool AtBlockStart = true;
  std::string PreEmittedCommentFile;
  int PreEmittedCommentLine = 0;

  // Top-level script body (the body of `@main`) is hoisted to module
  // scope. While emitting that body this flag is true so we don't add
  // a surrounding `function main()`.
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

// ---------------------------------------------------------------------------
// Literal formatting
// ---------------------------------------------------------------------------

static std::string formatIntAttr(mlir::IntegerAttr IA, bool Unsigned = false) {
  auto T = mlir::dyn_cast<mlir::IntegerType>(IA.getType());
  if (T && T.getWidth() == 1) {
    return (IA.getValue().getZExtValue() & 1u) ? "true" : "false";
  }
  char Buf[64];
  if (Unsigned) {
    snprintf(Buf, sizeof(Buf), "%llu",
             (unsigned long long)IA.getValue().getZExtValue());
  } else {
    snprintf(Buf, sizeof(Buf), "%lld", (long long)IA.getInt());
  }
  return Buf;
}

static std::string formatFloatAttr(mlir::FloatAttr FA) {
  // Pick the shortest precision (1..17) whose `%g` text round-trips
  // exactly back to D — same approach as EmitPython. JS / TS numbers
  // are IEEE-754 f64, so the round-trip semantics match.
  double D = FA.getValueAsDouble();
  char Buf[64];
  snprintf(Buf, sizeof(Buf), "%.17g", D);
  std::string Ref17 = Buf;
  bool Ref17HasE = (Ref17.find('e') != std::string::npos ||
                    Ref17.find('E') != std::string::npos);
  std::string S;
  for (int P = 1; P <= 17; ++P) {
    snprintf(Buf, sizeof(Buf), "%.*g", P, D);
    if (strtod(Buf, nullptr) != D) continue;
    bool ThisHasE = (strchr(Buf, 'e') || strchr(Buf, 'E'));
    if (!Ref17HasE && ThisHasE) continue;
    S = Buf;
    break;
  }
  if (S.empty()) S = Ref17;
  // TypeScript number literals with no `.` or `e` are still doubles;
  // we don't need to append `.0` like Python. But appending `.0` for
  // integer-valued floats keeps the source visually self-documenting
  // and matches EmitPython's output style — readers can tell at a
  // glance which locals are floats vs ints.
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
  if (Name.starts_with("matlab_")) {
    std::string Suf = Name.drop_front(strlen("matlab_")).str();
    // Suffixes that collide with TypeScript reserved words (or with
    // common globals like `assert`) need a trailing underscore so
    // `rt.assert(...)` doesn't shadow Node's built-in. Kept in-sync
    // with the runtime's keyword-adjacent helpers.
    if (Suf == "assert" || Suf == "delete" || Suf == "in" || Suf == "new" ||
        Suf == "var" || Suf == "let" || Suf == "const" || Suf == "if" ||
        Suf == "else" || Suf == "for" || Suf == "while" || Suf == "do" ||
        Suf == "function" || Suf == "return" || Suf == "class" ||
        Suf == "extends" || Suf == "import" || Suf == "export" ||
        Suf == "from" || Suf == "default" || Suf == "switch" ||
        Suf == "case" || Suf == "break" || Suf == "continue" ||
        Suf == "throw" || Suf == "try" || Suf == "catch" || Suf == "finally" ||
        Suf == "typeof" || Suf == "instanceof" || Suf == "void" ||
        Suf == "yield" || Suf == "this" || Suf == "super" || Suf == "null" ||
        Suf == "true" || Suf == "false" || Suf == "static" ||
        Suf == "with" || Suf == "of" || Suf == "as")
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
  // Unregistered matlab.* binops on scalars — same constraints
  // as the registered arith.* binops above.
  {
    StringRef MN = Op.getName().getStringRef();
    if (Op.getNumOperands() == 2 && Op.getNumResults() == 1 &&
        (MN == "matlab.add" || MN == "matlab.sub" ||
         MN == "matlab.emul" || MN == "matlab.matmul" ||
         MN == "matlab.ediv" || MN == "matlab.matdiv" ||
         MN == "matlab.eq" || MN == "matlab.ne" ||
         MN == "matlab.lt" || MN == "matlab.le" ||
         MN == "matlab.gt" || MN == "matlab.ge" ||
         MN == "matlab.short_or" || MN == "matlab.short_and"))
      return true;
  }

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

static bool isI1(mlir::Type T) {
  auto IT = mlir::dyn_cast<mlir::IntegerType>(T);
  return IT && IT.getWidth() == 1;
}

bool Emitter::buildInlineExpr(mlir::Operation &Op, std::string &Expr) {
  using namespace mlir;
  if (auto C = dyn_cast<LLVM::ConstantOp>(Op)) {
    auto A = C.getValue();
    bool Unsigned = (bool)Op.getAttr("matlab.unsigned");
    if (auto IA = dyn_cast<IntegerAttr>(A)) { Expr = formatIntAttr(IA, Unsigned); return true; }
    if (auto FA = dyn_cast<FloatAttr>(A)) { Expr = formatFloatAttr(FA); return true; }
    return false;
  }
  if (auto C = dyn_cast<arith::ConstantOp>(Op)) {
    auto A = C.getValue();
    bool Unsigned = (bool)Op.getAttr("matlab.unsigned");
    if (auto FA = dyn_cast<FloatAttr>(A)) { Expr = formatFloatAttr(FA); return true; }
    if (auto IA = dyn_cast<IntegerAttr>(A)) { Expr = formatIntAttr(IA, Unsigned); return true; }
    return false;
  }
  if (isa<LLVM::ZeroOp>(Op)) { Expr = "0"; return true; }
  if (auto A = dyn_cast<LLVM::AddressOfOp>(Op)) {
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
  // Shifts from LowerFixedPoint. JS `>>` is arithmetic on i32; for the
  // larger widths LowerFixedPoint emits BigInt-shaped helpers via the
  // runtime calls, so we don't need a BigInt path here.
  if (isa<arith::ShLIOp>(Op))  return bin("<<");
  if (isa<arith::ShRSIOp>(Op)) return bin(">>");
  if (isa<arith::ShRUIOp>(Op)) return bin(">>>");
  // Bitwise vs logical split on i1. TypeScript has `&&` / `||` for
  // booleans; bitwise operators on numbers keep their JS semantics.
  if (auto A = dyn_cast<arith::AndIOp>(Op)) {
    if (isI1(A.getType())) return bin("&&");
    return bin("&");
  }
  if (auto O = dyn_cast<arith::OrIOp>(Op)) {
    if (isI1(O.getType())) return bin("||");
    return bin("|");
  }
  if (auto X = dyn_cast<arith::XOrIOp>(Op)) {
    if (isI1(X.getType())) return bin("!==");
    return bin("^");
  }
  // Unregistered matlab.* binops on scalars — same shape as the
  // registered arith.* binops above. Inlines so chains collapse.
  {
    StringRef MN = Op.getName().getStringRef();
    if (Op.getNumOperands() == 2 && Op.getNumResults() == 1) {
      const char *cc = nullptr;
      if (MN == "matlab.add") cc = "+";
      else if (MN == "matlab.sub") cc = "-";
      else if (MN == "matlab.emul" || MN == "matlab.matmul") cc = "*";
      else if (MN == "matlab.ediv" || MN == "matlab.matdiv") cc = "/";
      else if (MN == "matlab.eq") cc = "===";
      else if (MN == "matlab.ne") cc = "!==";
      else if (MN == "matlab.lt") cc = "<";
      else if (MN == "matlab.le") cc = "<=";
      else if (MN == "matlab.gt") cc = ">";
      else if (MN == "matlab.ge") cc = ">=";
      else if (MN == "matlab.short_or") cc = "||";
      else if (MN == "matlab.short_and") cc = "&&";
      if (cc) return bin(cc);
    }
  }
  if (auto C = dyn_cast<arith::CmpFOp>(Op)) {
    const char *cc = "===";
    switch (C.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ: cc = "==="; break;
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE: cc = "!=="; break;
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
    const char *cc = "===";
    switch (C.getPredicate()) {
      case arith::CmpIPredicate::eq:  cc = "==="; break;
      case arith::CmpIPredicate::ne:  cc = "!=="; break;
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
    // 1.0, 0.0). Fold to `(c ? 1 : 0)` so the emitted source reads
    // closer to the MATLAB expression that produced it. JavaScript /
    // TypeScript has no `float()` cast, so we use the ternary form.
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
      Expr = "(" + dropOuterParens(exprFor(S.getCondition()))
           + " ? 1 : 0)";
      return true;
    }
    Expr = "(" + dropOuterParens(exprFor(S.getCondition())) + " ? "
         + exprFor(S.getTrueValue()) + " : "
         + exprFor(S.getFalseValue()) + ")";
    return true;
  }
  if (isa<arith::SIToFPOp, arith::UIToFPOp>(Op)) {
    // No-op for TS — JS numbers are doubles already.
    Expr = exprFor(Op.getOperand(0));
    return true;
  }
  if (isa<arith::FPToSIOp, arith::FPToUIOp>(Op)) {
    // `| 0` coerces to int32 in JS — close enough for our pipeline,
    // which only runs this on values that already fit in 32 bits.
    Expr = "(" + dropOuterParens(exprFor(Op.getOperand(0))) + " | 0)";
    return true;
  }
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
    Expr = exprFor(L.getAddr());
    return true;
  }
  if (auto G = dyn_cast<LLVM::GEPOp>(Op)) {
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
    if (tryRewriteAsClassCall(C.getCallee(), C.getOperands(), Expr))
      return true;
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
    if (tryRewriteObjGet(C, Expr)) return true;
    if (tryRewriteAsClassCall(*C.getCallee(), C.getOperands(), Expr))
      return true;
    if (tryRewriteAsNumpy(C, Expr)) return true;
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
// For-loop pattern detection (mirrors EmitPython.cpp::matchForPattern)
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
  if (Suffix == "disp_str")      { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_str")   { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64")   { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64_2") { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64_3") { LengthArgIdx = 1; return true; }
  if (Suffix == "fprintf_f64_4") { LengthArgIdx = 1; return true; }
  if (Suffix == "input_num")     { LengthArgIdx = 1; return true; }
  if (Suffix == "obj_get_f64")   { LengthArgIdx = 2; return true; }
  if (Suffix == "obj_set_f64")   { LengthArgIdx = 2; return true; }
  return false;
}

// ---------------------------------------------------------------------------
// numpy-ts rewrite — matrix calls collapse to inline NDArray expressions
// ---------------------------------------------------------------------------
//
// TypeScript can't overload `+` / `*` / `@`, so binary matrix ops use
// the NDArray method form: `A.add(B)` / `A.matmul(B)` / `A.mul(s)`.
// Construction goes through `np.zeros(m, n)` / `np.eye(n)` / etc.,
// matching the numpy_ts module's API.
bool Emitter::tryRewriteAsNumpy(mlir::LLVM::CallOp C, std::string &Out) {
  if (!C.getCallee()) return false;
  llvm::StringRef Full = *C.getCallee();
  if (!Full.starts_with("matlab_")) return false;
  llvm::StringRef Suf = Full.drop_front(strlen("matlab_"));

  auto opnd = [&](unsigned i) {
    return dropOuterParens(this->exprFor(C.getOperand(i)));
  };
  auto wrapped = [&](unsigned i) {
    return this->exprFor(C.getOperand(i));
  };
  // Format an integer-typed dimension argument: a constant integer
  // literal becomes a bare `3`; anything else is wrapped in `(... | 0)`
  // so it's coerced to int32 before reaching the runtime.
  auto fmtIntDim = [&](mlir::Value V) -> std::string {
    long long I;
    if (tryEvalIntLiteral(V, I)) return std::to_string(I);
    return "(" + dropOuterParens(this->exprFor(V)) + " | 0)";
  };
  auto methodBin = [&](const char *Op) {
    Out = wrapped(0) + "." + Op + "(" + opnd(1) + ")";
    return true;
  };
  // Commutative scalar-matrix forms — flip operands so the matrix
  // becomes the receiver and the scalar an argument. This lets the
  // emitted source read as `A.add(2)` instead of the broken `2.add(A)`
  // (numbers don't carry NDArray methods in TS).
  auto methodBinFlip = [&](const char *Op) {
    Out = wrapped(1) + "." + Op + "(" + opnd(0) + ")";
    return true;
  };

  // --- Element-wise / matrix arithmetic -----------------------------
  if (C.getNumOperands() == 2) {
    if (Suf == "matmul_mm") { Out = "np.matmul(" + opnd(0) + ", " + opnd(1) + ")"; return true; }
    if (Suf == "add_mm" || Suf == "add_ms")  return methodBin("add");
    if (Suf == "add_sm")                     return methodBinFlip("add");
    if (Suf == "sub_mm" || Suf == "sub_ms")  return methodBin("sub");
    if (Suf == "emul_mm" || Suf == "emul_ms") return methodBin("mul");
    if (Suf == "emul_sm")                    return methodBinFlip("mul");
    if (Suf == "ediv_mm" || Suf == "ediv_ms") return methodBin("div");
    // sub_sm / ediv_sm aren't commutative with the scalar — leave them
    // on the runtime path so the operand order stays clear.
  }

  // --- Construction --------------------------------------------------
  if (Suf == "mat_from_buf" && C.getNumOperands() == 3) {
    Out = "rt.mat_from_buf(" + opnd(0) + ", " +
          fmtIntDim(C.getOperand(1)) + ", " +
          fmtIntDim(C.getOperand(2)) + ")";
    return true;
  }
  if (Suf == "mat_from_scalar" && C.getNumOperands() == 1) {
    Out = "np.array([[" + opnd(0) + "]])";
    return true;
  }
  if (Suf == "zeros") {
    if (C.getNumOperands() == 1) {
      std::string D = fmtIntDim(C.getOperand(0));
      Out = "np.zeros(" + D + ", " + D + ")";  // MATLAB zeros(n) is n×n.
      return true;
    }
    if (C.getNumOperands() == 2) {
      Out = "np.zeros(" + fmtIntDim(C.getOperand(0)) + ", " +
            fmtIntDim(C.getOperand(1)) + ")";
      return true;
    }
  }
  if (Suf == "ones") {
    if (C.getNumOperands() == 1) {
      std::string D = fmtIntDim(C.getOperand(0));
      Out = "np.ones(" + D + ", " + D + ")";
      return true;
    }
    if (C.getNumOperands() == 2) {
      Out = "np.ones(" + fmtIntDim(C.getOperand(0)) + ", " +
            fmtIntDim(C.getOperand(1)) + ")";
      return true;
    }
  }
  if (Suf == "eye") {
    if (C.getNumOperands() == 1) {
      Out = "np.eye(" + fmtIntDim(C.getOperand(0)) + ")";
      return true;
    }
    if (C.getNumOperands() == 2) {
      Out = "np.eye(" + fmtIntDim(C.getOperand(0)) + ", " +
            fmtIntDim(C.getOperand(1)) + ")";
      return true;
    }
  }

  // --- Linear algebra ------------------------------------------------
  if (C.getNumOperands() == 1) {
    if (Suf == "transpose") {
      Out = wrapped(0) + ".T";
      return true;
    }
    if (Suf == "inv")    { Out = "np.linalg.inv("  + opnd(0) + ")"; return true; }
    if (Suf == "det")    { Out = "np.linalg.det("  + opnd(0) + ")"; return true; }
    if (Suf == "norm")   { Out = "np.linalg.norm(" + opnd(0) + ")"; return true; }
    if (Suf == "trace")  { Out = "np.trace("       + opnd(0) + ")"; return true; }
  }
  if (Suf == "mldivide_mm" && C.getNumOperands() == 2) {
    Out = "np.linalg.solve(" + opnd(0) + ", " + opnd(1) + ")";
    return true;
  }

  // --- Element-wise math (matrix variants) --------------------------
  static constexpr struct { const char *Suf; const char *Np; } UnaryMatNp[] = {
    {"sqrt_m", "np.sqrt"},   {"exp_m", "np.exp"},     {"log_m", "np.log"},
    {"log2_m", "np.log2"},   {"log10_m", "np.log10"},
    {"sin_m",  "np.sin"},    {"cos_m", "np.cos"},     {"tan_m", "np.tan"},
    {"asin_m", "np.asin"},   {"acos_m", "np.acos"},   {"atan_m", "np.atan"},
    {"sinh_m", "np.sinh"},   {"cosh_m", "np.cosh"},   {"tanh_m", "np.tanh"},
    {"abs_m",  "np.abs"},    {"sign_m", "np.sign"},
    {"floor_m","np.floor"},  {"ceil_m", "np.ceil"},   {"round_m", "np.round"},
  };
  for (auto &E : UnaryMatNp) {
    if (Suf == E.Suf && C.getNumOperands() == 1) {
      Out = std::string(E.Np) + "(" + opnd(0) + ")";
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Class field access — `obj_get_f64(obj, "X")` → `obj.X`
// ---------------------------------------------------------------------------

static std::optional<std::string> getFieldNameLit(
    mlir::Value V, const llvm::StringMap<std::string> &StringGlobalLits) {
  auto *D = V.getDefiningOp();
  auto A = mlir::dyn_cast_or_null<mlir::LLVM::AddressOfOp>(D);
  if (!A) return std::nullopt;
  auto It = StringGlobalLits.find(A.getGlobalName());
  if (It == StringGlobalLits.end()) return std::nullopt;
  llvm::StringRef Lit = It->second;
  if (Lit.size() < 2 || Lit.front() != '"' || Lit.back() != '"')
    return std::nullopt;
  std::string Name = Lit.substr(1, Lit.size() - 2).str();
  if (Name.empty()) return std::nullopt;
  unsigned char C0 = (unsigned char)Name[0];
  if (!(std::isalpha(C0) || C0 == '_' || C0 == '$')) return std::nullopt;
  for (char C : Name)
    if (!std::isalnum((unsigned char)C) && C != '_' && C != '$') return std::nullopt;
  // Don't shadow a TypeScript reserved word.
  static constexpr const char *Keywords[] = {
    "break","case","catch","class","const","continue","debugger","default",
    "delete","do","else","enum","export","extends","false","finally","for",
    "function","if","import","in","instanceof","new","null","return","super",
    "switch","this","throw","true","try","typeof","var","void","while","with",
    "yield","let","static","implements","interface","package","private",
    "protected","public","await","async",
  };
  for (auto *KW : Keywords) if (Name == KW) return std::nullopt;
  return Name;
}

bool Emitter::tryRewriteObjGet(mlir::LLVM::CallOp C, std::string &Out) {
  if (!C.getCallee()) return false;
  if (*C.getCallee() != "matlab_obj_get_f64") return false;
  if (C.getNumOperands() < 2) return false;
  auto Field = getFieldNameLit(C.getOperand(1), StringGlobalLits);
  if (!Field) return false;
  Out = exprFor(C.getOperand(0)) + "." + *Field;
  return true;
}

// ---------------------------------------------------------------------------
// Classdef collection + call-site rewrite
// ---------------------------------------------------------------------------

// Map a MATLAB operator method name to the TypeScript method name we
// emit for it. TypeScript can't overload binary operators, so call
// sites become `a.<method>(b)` instead of `a OP b`. Returns true on
// success and writes the method name. We keep the MATLAB names verbatim
// (`plus`, `minus`, ...), which reads as the natural translation.
static bool mapOperatorMethod(llvm::StringRef Name, std::string &Method,
                              std::string &Op) {
  Op.clear();  // Unused for TS — kept for the EmitPython parity shim.
  if (Name == "eq")     { Method = "eq";     return true; }
  if (Name == "ne")     { Method = "ne";     return true; }
  if (Name == "lt")     { Method = "lt";     return true; }
  if (Name == "le")     { Method = "le";     return true; }
  if (Name == "gt")     { Method = "gt";     return true; }
  if (Name == "ge")     { Method = "ge";     return true; }
  if (Name == "plus")   { Method = "plus";   return true; }
  if (Name == "minus")  { Method = "minus";  return true; }
  if (Name == "mtimes" ||
      Name == "times")  { Method = Name.str(); return true; }
  if (Name == "mrdivide" ||
      Name == "rdivide"){ Method = Name.str(); return true; }
  if (Name == "uminus") { Method = "uminus"; return true; }
  return false;
}

void Emitter::collectClasses(mlir::ModuleOp M) {
  for (auto &Op : M.getBody()->getOperations()) {
    auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op);
    if (!F) continue;
    if (F.getBody().empty()) continue;
    auto CN = F->getAttrOfType<mlir::StringAttr>("matlab.class_name");
    if (!CN) continue;

    ClassMethodInfo CMI;
    CMI.Func = F;
    CMI.ClassName = CN.getValue().str();

    auto Kind = F->getAttrOfType<mlir::StringAttr>("matlab.method_kind");
    auto MN   = F->getAttrOfType<mlir::StringAttr>("matlab.method_name");
    llvm::StringRef MName = MN ? MN.getValue() : llvm::StringRef();

    if (Kind && Kind.getValue() == "ctor") {
      CMI.Kind = ClassMethodInfo::Ctor;
      CMI.EmitName = "constructor";
    } else if (Kind && Kind.getValue() == "static") {
      CMI.Kind = ClassMethodInfo::Static;
      CMI.EmitName = MName.empty() ? "" : MName.str();
      if (CMI.EmitName.empty()) continue;
    } else if (MName.starts_with("get.")) {
      CMI.Kind = ClassMethodInfo::Get;
      CMI.EmitName = MName.drop_front(4).str();
    } else if (std::string Mthd, Op; mapOperatorMethod(MName, Mthd, Op)) {
      CMI.Kind = ClassMethodInfo::Operator;
      CMI.EmitName = Mthd;
      CMI.OpSpelling = Op;  // empty — TypeScript uses method-call form
    } else if (!MName.empty()) {
      CMI.Kind = ClassMethodInfo::Method;
      CMI.EmitName = MName.str();
    } else {
      continue;
    }

    auto &CD = Classes[CMI.ClassName];
    if (auto Sup = F->getAttrOfType<mlir::StringAttr>("matlab.class_super"))
      if (CD.Super.empty()) CD.Super = Sup.getValue().str();
    CD.Methods.push_back(CMI);
    CalleeIndex[F.getSymName()] = CMI;
  }
}

bool Emitter::tryRewriteAsClassCall(llvm::StringRef Callee,
                                     mlir::ValueRange Operands,
                                     std::string &Out) {
  auto It = CalleeIndex.find(Callee);
  if (It == CalleeIndex.end()) return false;
  const ClassMethodInfo &CMI = It->second;
  switch (CMI.Kind) {
    case ClassMethodInfo::Ctor: {
      std::string E = "new " + CMI.ClassName + "(";
      for (unsigned i = 0; i < Operands.size(); ++i) {
        if (i) E += ", ";
        E += dropOuterParens(this->stmtExpr(Operands[i]));
      }
      E += ")";
      Out = E;
      return true;
    }
    case ClassMethodInfo::Get: {
      if (Operands.size() != 1) return false;
      Out = exprFor(Operands[0]) + "." + CMI.EmitName;
      return true;
    }
    case ClassMethodInfo::Operator:
    case ClassMethodInfo::Method: {
      if (Operands.empty()) return false;
      std::string E = exprFor(Operands[0]) + "." + CMI.EmitName + "(";
      for (unsigned i = 1; i < Operands.size(); ++i) {
        if (i > 1) E += ", ";
        E += dropOuterParens(this->stmtExpr(Operands[i]));
      }
      E += ")";
      Out = E;
      return true;
    }
    case ClassMethodInfo::Static: {
      std::string E = CMI.ClassName + "." + CMI.EmitName + "(";
      for (unsigned i = 0; i < Operands.size(); ++i) {
        if (i) E += ", ";
        E += dropOuterParens(this->stmtExpr(Operands[i]));
      }
      E += ")";
      Out = E;
      return true;
    }
  }
  return false;
}

bool Emitter::tryRewriteObjSet(mlir::LLVM::CallOp C, std::string &Out) {
  if (!C.getCallee()) return false;
  if (*C.getCallee() != "matlab_obj_set_f64") return false;
  if (C.getNumOperands() < 4) return false;
  auto Field = getFieldNameLit(C.getOperand(1), StringGlobalLits);
  if (!Field) return false;
  Out = dropOuterParens(exprFor(C.getOperand(0))) + "." + *Field +
        " = " + dropOuterParens(stmtExpr(C.getOperand(3)));
  return true;
}

void Emitter::scanBreakContinueFlags(mlir::Region &R) {
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

  R.walk([&](mlir::LLVM::StoreOp S) {
    auto *Addr = S.getAddr().getDefiningOp();
    if (!Addr) return;
    bool IsFlag = BreakFlagSlots.count(Addr) || ContinueFlagSlots.count(Addr);
    if (!IsFlag) return;
    if (isConstInt(S.getValue(), 0)) {
      SuppressedOps.insert(S.getOperation());
    }
  });

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

  std::string FileLeaf = FullPath.str();
  if (auto Slash = FileLeaf.find_last_of("/\\"); Slash != std::string::npos)
    FileLeaf = FileLeaf.substr(Slash + 1);
  if (PreEmittedCommentLine > 0 && FileLeaf == PreEmittedCommentFile)
    AfterLine = std::max(AfterLine, PreEmittedCommentLine);

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
    OS << "// " << Info.Body << "\n";
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
  (void)NoLine;
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

// Build a TypeScript double-quoted string literal for `Raw`.
static std::string buildTsStringLit(llvm::StringRef Raw) {
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
  OS << "const " << N << " = " << buildTsStringLit(Raw) << ";\n";
}

// ---------------------------------------------------------------------------
// Prolog
// ---------------------------------------------------------------------------

void Emitter::emitProlog() {
  // The header line always goes here; the conditional `import` lines
  // are added in the standalone `emitTypeScript()` after we know what
  // the body actually references.
  OS << "// Generated by matlabc -emit-typescript. Do not edit.\n";
}

void Emitter::precomputeModuleProperties(mlir::ModuleOp) {
  // No per-module toggles today.
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

bool Emitter::run(mlir::ModuleOp M) {
  if (mlir::failed(mlir::verify(M))) {
    fail("MLIR verification failed before TypeScript emission");
    return false;
  }
  precomputeModuleProperties(M);
  emitProlog();

  // Multi-return: TypeScript supports tuple types and array
  // destructuring natively (`function f(): [T0, T1]` + `const [a, b]
  // = f();`). The return / call-site emit handles N>1 explicitly.

  // Pass 1: register every string global's TS literal so AddressOf
  // inlining can fold it directly.
  for (auto &Op : M.getBody()->getOperations()) {
    if (auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op)) {
      auto Val = G.getValueAttr();
      if (mlir::dyn_cast_or_null<mlir::StringAttr>(Val)) {
        StringGlobalLits[G.getSymName()] = buildTsStringLit(constStr(G));
        UsedNames.insert(G.getSymName().str());
      }
    }
  }

  // Pass 2: collect MATLAB classdefs.
  collectClasses(M);

  // Pass 3: reserve symbols of every defined function so locals don't
  // shadow them, and emit bodies. `@main` is emitted last.
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
  for (auto &KV : Classes) UsedNames.insert(KV.first().str());

  // Hoist the script's top-of-file `% ...` comment block to module-top.
  if (MainFn && SM) {
    auto extract = [](mlir::Location L) -> mlir::FileLineColLoc {
      if (auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(L)) return FL;
      if (auto NL = mlir::dyn_cast<mlir::NameLoc>(L))
        if (auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(NL.getChildLoc()))
          return FL;
      if (auto FuL = mlir::dyn_cast<mlir::FusedLoc>(L)) {
        for (auto Sub : FuL.getLocations())
          if (auto FL = mlir::dyn_cast<mlir::FileLineColLoc>(Sub))
            return FL;
      }
      return {};
    };
    std::string FullPath;
    int MinLine = 0;
    MainFn.getBody().walk([&](mlir::Operation *Op) {
      if (auto FL = extract(Op->getLoc())) {
        int L = (int)FL.getLine();
        if (L > 0 && (MinLine == 0 || L < MinLine)) {
          MinLine = L;
          FullPath = FL.getFilename().str();
        }
      }
    });
    if (MinLine > 0 && !FullPath.empty()) {
      int Line = MinLine;
      std::string FileLeaf = FullPath;
      if (auto Slash = FileLeaf.find_last_of("/\\");
          Slash != std::string::npos)
        FileLeaf = FileLeaf.substr(Slash + 1);
      matlab::FileID F = SM->findFileByName(std::string_view(FullPath));
      if (F != 0 && Line > 1) {
        int Start = Line;
        for (int L = Line - 1; L >= 1; --L) {
          auto Info = classifyLine(SM->getLineText(F, (uint32_t)L));
          if (Info.Kind == LineKind::Code) { Start = L + 1; break; }
          Start = L;
        }
        if (Start <= 1) {
          bool LastWasBlank = false;
          bool EmittedAny = false;
          for (int L = Start; L < Line; ++L) {
            auto Info = classifyLine(SM->getLineText(F, (uint32_t)L));
            if (Info.Kind == LineKind::Blank) {
              if (LastWasBlank) continue;
              OS << "\n";
              LastWasBlank = true;
              continue;
            }
            if (Info.Kind == LineKind::Comment) {
              OS << "// " << Info.Body << "\n";
              LastWasBlank = false;
              EmittedAny = true;
            }
          }
          if (EmittedAny && !LastWasBlank) OS << "\n";
          PreEmittedCommentFile = FileLeaf;
          PreEmittedCommentLine = Line - 1;
        }
      }
    }
  }

  // Pass 4: emit class blocks. Order matters when a subclass references
  // its parent — emit classes whose Super is already declared first.
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
  for (auto &KV : Classes)
    if (!Emitted.count(KV.first())) ClassOrder.push_back(KV.first());

  for (llvm::StringRef Name : ClassOrder) {
    if (Failed) break;
    emitClassBlock(Name, Classes[Name]);
  }

  // Pass 5: emit free (non-class, non-main) functions.
  for (auto &Op : M.getBody()->getOperations()) {
    if (Failed) break;
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      if (F == MainFn) continue;
      if (F->hasAttr("matlab.class_name")) continue;
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

  llvm::StringSet<> SavedUsed;
  bool IsMain = F.getSymName() == "main";
  if (!IsMain) {
    SavedUsed = UsedNames;
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
  }

  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());
  auto FT = F.getFunctionType();

  if (IsMain) {
    InMainHoist = true;
    emitRegion(F.getBody(), 0);
    InMainHoist = false;
    return;
  }

  /* Walk for persistent variables and stamp module-level `let` decls
   * just before the function's `function` line. The mangled name is
   * `<fn>_<var>` so two functions can each declare `persistent n`
   * without colliding at module scope. Refs inside the function body
   * are rewritten to the mangled name; the verbatim runtime call is
   * suppressed. */
  llvm::SetVector<llvm::StringRef> PersistentNames;
  llvm::DenseMap<llvm::StringRef, std::string> InitExprByName;
  std::string FnSym = F.getSymName().str();
  // Walk both LLVM::CallOp and matlab.call_builtin shapes — TS path
  // doesn't run LowerFixedPoint's matlab_persistent_* → llvm.call
  // sweep, so set calls survive as matlab.call_builtin.
  F.getBody().walk([&](mlir::Operation *Op) {
    auto PN = Op->getAttrOfType<mlir::StringAttr>("persistent_name");
    if (!PN) return;
    PersistentNames.insert(PN.getValue());
    llvm::StringRef Callee;
    if (auto C = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
      if (auto Sym = C.getCallee()) Callee = *Sym;
    } else if (Op->getName().getStringRef() == "matlab.call_builtin") {
      if (auto CA = Op->getAttrOfType<mlir::StringAttr>("callee"))
        Callee = CA.getValue();
    }
    if (Callee == "matlab_global_get_f64" && Op->getNumResults() == 1) {
      this->Names[Op->getResult(0)] = FnSym + "_" + PN.getValue().str();
      SuppressedOps.insert(Op);
    }
  });
  // Tier 1: detect canonical isempty-init pattern. Init value
  // becomes the mangled persistent's `let` initializer.
  F.getBody().walk([&](mlir::LLVM::CallOp IECall) {
    auto Callee = IECall.getCallee();
    if (!Callee || *Callee != "matlab_persistent_isempty") return;
    if (IECall.getNumResults() != 1) return;
    if (!IECall.getResult().hasOneUse()) return;
    auto *CmpUser = IECall.getResult().use_begin()->getOwner();
    auto Cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(CmpUser);
    if (!Cmp || !Cmp.getResult().hasOneUse()) return;
    auto *IfUser = Cmp.getResult().use_begin()->getOwner();
    auto Guard = mlir::dyn_cast<mlir::scf::IfOp>(IfUser);
    if (!Guard || !Guard.getThenRegion().hasOneBlock()) return;
    mlir::Operation *InitSet = nullptr;
    llvm::StringRef PNStr;
    for (mlir::Operation &Op : Guard.getThenRegion().front()) {
      llvm::StringRef Cl;
      if (auto LC = mlir::dyn_cast<mlir::LLVM::CallOp>(&Op)) {
        if (auto C = LC.getCallee()) Cl = *C;
      } else if (Op.getName().getStringRef() == "matlab.call_builtin") {
        if (auto CA = Op.getAttrOfType<mlir::StringAttr>("callee"))
          Cl = CA.getValue();
      } else continue;
      if (Cl != "matlab_global_set_f64") continue;
      auto PN = Op.getAttrOfType<mlir::StringAttr>("persistent_name");
      if (!PN) continue;
      InitSet = &Op;
      PNStr = PN.getValue();
      break;
    }
    if (!InitSet || InitSet->getNumOperands() < 2) return;
    InitExprByName[PNStr] =
        dropOuterParens(this->exprFor(InitSet->getOperand(1)));
    SuppressedOps.insert(IECall.getOperation());
    SuppressedOps.insert(Cmp.getOperation());
    SuppressedOps.insert(Guard.getOperation());
    SuppressedOps.insert(InitSet);
  });
  if (!PersistentNames.empty()) {
    llvm::SmallVector<llvm::StringRef, 4> Sorted(
        PersistentNames.begin(), PersistentNames.end());
    std::sort(Sorted.begin(), Sorted.end());
    for (llvm::StringRef PN : Sorted) {
      auto It = InitExprByName.find(PN);
      std::string Init = (It != InitExprByName.end() && !It->second.empty())
                             ? It->second
                             : "0.0";
      OS << "let " << FnSym << "_" << PN.str() << ": number = "
         << Init << ";\n";
    }
    OS << "\n";
  }

  OS << "function " << F.getSymName().str() << "(";
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
    OS << N + ": any";
  }
  OS << "): any {\n";
  if (F.getBody().front().empty() ||
      (F.getBody().front().getOperations().size() == 1 &&
       mlir::isa<mlir::func::ReturnOp>(&F.getBody().front().front()))) {
    OS << "}\n\n";
    if (!IsMain) UsedNames = std::move(SavedUsed);
    return;
  }
  emitRegion(F.getBody(), 1);
  OS << "}\n\n";
  if (!IsMain) UsedNames = std::move(SavedUsed);
}

void Emitter::emitClassBlock(llvm::StringRef ClassName, const ClassDef &CI) {
  OS << "class " << ClassName.str();
  // `< handle` in MATLAB just makes the class reference-typed; JS / TS
  // objects are reference-typed already, so we drop that base. Real
  // user supers stay.
  bool HasUserSuper = !CI.Super.empty() && CI.Super != "handle";
  if (HasUserSuper) OS << " extends " << CI.Super;
  OS << " {\n";
  // Property declarations: collect all property names from method
  // bodies' obj_set_f64 / obj_get_f64 that resolve to plain identifiers.
  // TypeScript classes need fields declared (or `any`-typed) for
  // `this.X` access without `--strict` errors. We emit each as
  // `<name>: any;` so the rest of the class compiles cleanly.
  llvm::StringSet<> Fields;
  for (const auto &CMI : CI.Methods) {
    mlir::func::FuncOp Func = CMI.Func;
    Func.getBody().walk([&](mlir::LLVM::CallOp Call) {
      if (!Call.getCallee()) return;
      llvm::StringRef Cn = *Call.getCallee();
      if (Cn != "matlab_obj_set_f64" && Cn != "matlab_obj_get_f64") return;
      if (Call.getNumOperands() < 2) return;
      auto Field = getFieldNameLit(Call.getOperand(1), StringGlobalLits);
      if (!Field) return;
      Fields.insert(*Field);
    });
  }
  for (auto &K : Fields) {
    indent(1);
    OS << K.first().str() << ": any;\n";
  }
  if (!Fields.empty()) OS << "\n";
  if (CI.Methods.empty() && Fields.empty()) {
    // No methods, no fields — leave the class body empty.
  } else {
    for (const auto &CMI : CI.Methods)
      emitClassMethod(CMI, /*Indent=*/1);
  }
  OS << "}\n\n";
}

void Emitter::emitClassMethod(const ClassMethodInfo &CMI, int Indent) {
  mlir::func::FuncOp F = CMI.Func;
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
  Names.clear();

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
  for (auto &KV : Classes) UsedNames.insert(KV.first().str());
  UsedNames.insert("this");

  computeInlines(F.getBody());
  scanBreakContinueFlags(F.getBody());
  scanForLoopPatterns(F.getBody());

  auto &Entry = F.getBody().front();
  auto FT = F.getFunctionType();
  bool IsCtor = CMI.Kind == ClassMethodInfo::Ctor;
  bool IsStatic = CMI.Kind == ClassMethodInfo::Static;

  // Bind `this` for instance methods. The TS emitter doesn't pass
  // `self` as a parameter — `this` is the receiver — so the first
  // function arg is consumed as the implicit receiver.
  if (!IsCtor && !IsStatic && FT.getNumInputs() >= 1) {
    Names[Entry.getArgument(0)] = "this";
  }
  // For a constructor, walk the body for the obj_new call. Bind its
  // result to `this` so all the field-store rewrites read as
  // `this.X = ...`. The obj_new itself is suppressed — TypeScript
  // creates `this` automatically when `new ClassName(...)` runs.
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
        if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
          if (R.getNumOperands() == 1 &&
              R.getOperand(0) == CtorObjNew->getResult(0))
            SuppressedOps.insert(Op);
        }
      });
    }
  }

  // Header. Getters use `get name()`. Static methods use `static`.
  // Constructors render as `constructor(...)` with no return type.
  indent(Indent);
  if (CMI.Kind == ClassMethodInfo::Get) OS << "get ";
  else if (IsStatic) OS << "static ";
  OS << CMI.EmitName << "(";
  bool FirstParam = true;
  unsigned StartArg = (IsCtor || IsStatic) ? 0 : 1;
  for (unsigned i = StartArg; i < FT.getNumInputs(); ++i) {
    if (!FirstParam) OS << ", ";
    FirstParam = false;
    auto Arg = Entry.getArgument(i);
    std::string N;
    if (auto NA = F.getArgAttrOfType<mlir::StringAttr>(i, "matlab.name"))
      N = uniqueName(NA.getValue());
    else
      N = freshName();
    Names[Arg] = N;
    OS << N << ": any";
  }
  // Constructors can't carry a return type; everything else gets `: any`.
  if (IsCtor) OS << ") {\n";
  else        OS << "): any {\n";

  bool BodyIsTrivial = true;
  for (auto &Op : Entry.getOperations()) {
    if (SuppressedOps.count(&Op)) continue;
    if (InlinedOps.count(&Op)) continue;
    if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
      if (R.getNumOperands() == 0) continue;
      BodyIsTrivial = false; break;
    }
    if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
      if (R.getNumOperands() == 0) continue;
      BodyIsTrivial = false; break;
    }
    if (mlir::isa<mlir::arith::ConstantOp, mlir::LLVM::ConstantOp,
                  mlir::LLVM::ZeroOp, mlir::LLVM::AddressOfOp>(Op))
      continue;
    BodyIsTrivial = false;
    break;
  }
  // Derived-class constructors must call `super()` before any use of
  // `this`. We don't model MATLAB's super-args today — the in-tree
  // tests inherit from `handle` (already filtered out) or from a
  // user class with a no-arg ctor — so a bare `super()` suffices.
  bool DerivedCtor = false;
  if (IsCtor) {
    auto It = Classes.find(CMI.ClassName);
    if (It != Classes.end() && !It->second.Super.empty() &&
        It->second.Super != "handle")
      DerivedCtor = true;
  }
  if (BodyIsTrivial) {
    if (DerivedCtor) { indent(Indent + 1); OS << "super();\n"; }
    indent(Indent); OS << "}\n";
    UsedNames = std::move(SavedUsed);
    return;
  }

  if (DerivedCtor) { indent(Indent + 1); OS << "super();\n"; }
  emitRegion(F.getBody(), Indent + 1);
  indent(Indent); OS << "}\n";
  UsedNames = std::move(SavedUsed);
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
  OS << "function " << F.getSymName().str() << "(";
  auto &Entry = F.getBody().front();
  for (unsigned i = 0; i < FT.getNumParams(); ++i) {
    if (i) OS << ", ";
    auto Arg = Entry.getArgument(i);
    std::string N;
    if (auto NA = F.getArgAttrOfType<mlir::StringAttr>(i, "matlab.name"))
      N = uniqueName(NA.getValue());
    else
      N = freshName();
    Names[Arg] = N;
    OS << N << ": any";
  }
  OS << "): any {\n";
  if (F.getBody().front().empty()) {
    OS << "}\n\n";
    UsedNames = std::move(SavedUsed);
    return;
  }
  emitRegion(F.getBody(), 1);
  OS << "}\n\n";
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
  for (auto &Op : B.getOperations())
    emitOp(Op, Indent);
}

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
    OS << "let " << N << " = 0;\n";
    return;
  }

  // --- llvm.mlir.constant ---------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::LLVM::ConstantOp>(Op)) {
    std::string N = this->name(C.getResult());
    indent(Indent);
    OS << "const " << N << " = ";
    auto V = C.getValue();
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V))      OS << formatIntAttr(IA);
    else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V))   OS << formatFloatAttr(FA);
    else OS << "0 /* unknown const */";
    OS << ";\n";
    return;
  }

  // --- arith.constant --------------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::arith::ConstantOp>(Op)) {
    std::string N = this->name(C.getResult());
    indent(Indent);
    OS << "const " << N << " = ";
    auto V = C.getValue();
    if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V))        OS << formatFloatAttr(FA);
    else if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V)) OS << formatIntAttr(IA);
    else OS << "0 /* unknown const */";
    OS << ";\n";
    return;
  }

  // --- func.return / llvm.return --------------------------------------
  if (auto R = mlir::dyn_cast<mlir::func::ReturnOp>(Op)) {
    if (InMainHoist) return;
    if (R.getNumOperands() == 0) {
      // Trailing void return: TypeScript falls through, no need to emit
      // a `return;` at the end of a function body.
      if (R->getNextNode() == nullptr &&
          R->getBlock() == &R->getParentRegion()->back())
        return;
      indent(Indent); OS << "return;\n";
    } else if (R.getNumOperands() == 1) {
      indent(Indent);
      OS << "return " << this->stmtExpr(R.getOperand(0)) << ";\n";
    } else {
      // Multi-return: TS tuple — `return [a, b];`.
      indent(Indent);
      OS << "return [";
      for (unsigned i = 0; i < R.getNumOperands(); ++i) {
        if (i) OS << ", ";
        OS << this->stmtExpr(R.getOperand(i));
      }
      OS << "];\n";
    }
    return;
  }
  if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
    if (InMainHoist) return;
    if (R.getNumOperands() == 0) {
      if (R->getNextNode() == nullptr &&
          R->getBlock() == &R->getParentRegion()->back())
        return;
      indent(Indent); OS << "return;\n";
    } else {
      indent(Indent);
      OS << "return " << this->stmtExpr(R.getOperand(0)) << ";\n";
    }
    return;
  }

  // --- llvm.call / func.call ------------------------------------------
  if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
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
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult());
      OS << "const " << N << " = ";
    }
    if (auto Callee = Call.getCallee()) {
      if (*Callee == "matlab_obj_get_f64" && Call.getNumResults() == 1) {
        std::string Rewrite;
        if (tryRewriteObjGet(Call, Rewrite)) {
          OS << dropOuterParens(Rewrite) << ";\n";
          return;
        }
      }
      {
        std::string Rewrite;
        if (tryRewriteAsClassCall(*Callee, Call.getOperands(), Rewrite)) {
          OS << dropOuterParens(Rewrite) << ";\n";
          return;
        }
      }
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
      // `rt.disp_str("literal")` is byte-identical to `console.log("literal")`,
      // so collapse it. Detect by callee + a single (post-length-drop)
      // operand that traces to a string-global addressof.
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
        OS << "console.log(" << this->stmtExpr(Call.getOperand(0)) << ");\n";
        return;
      }
      // `rt.disp_f64(x)` for a constant integer literal collapses to
      // `console.log(<int>)` — TypeScript's number formatting matches
      // C's `%g` for plain integer values.
      if (*Callee == "matlab_disp_f64" && Call.getNumResults() == 0 &&
          Call.getNumOperands() == 1) {
        // We keep the runtime call here — the runtime's formatter
        // already mirrors `%g` and handles NaN / Inf consistently.
        // (Inlined `${x}` would diverge for fractional values.)
      }
      if (*Callee == "matlab_disp_mat" && Call.getNumResults() == 0 &&
          Call.getNumOperands() == 1) {
        OS << "rt.disp_mat(" << this->stmtExpr(Call.getOperand(0)) << ");\n";
        return;
      }
      {
        std::string Rewrite;
        if (tryRewriteAsNumpy(Call, Rewrite)) {
          OS << dropOuterParens(Rewrite) << ";\n";
          return;
        }
      }
      /* Persistent variable write: `matlab_global_set_f64(_, v)` with
       * persistent_name + persistent_fn lowers to `<fn>_<name> = <v>;`.
       * The module-level `let <fn>_<name> = 0.0;` decl was emitted by
       * emitFuncFunc above, before the function definition. */
      if (*Callee == "matlab_global_set_f64" && Call.getNumResults() == 0 &&
          Call.getNumOperands() == 2) {
        auto PN = Call->getAttrOfType<mlir::StringAttr>("persistent_name");
        auto PF = Call->getAttrOfType<mlir::StringAttr>("persistent_fn");
        if (PN && PF) {
          OS << PF.getValue().str() << "_" << PN.getValue().str()
             << " = " << this->stmtExpr(Call.getOperand(1)) << ";\n";
          return;
        }
      }
      OS << remapRuntimeCallee(*Callee) << "(";
      bool First = true;
      for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
        if (DropLen && i == LengthIdx) continue;
        if (!First) OS << ", ";
        First = false;
        OS << this->stmtExpr(Call.getOperand(i));
      }
      OS << ");\n";
    } else {
      // Indirect call: first operand is the callable.
      OS << this->exprFor(Call.getOperand(0)) << "(";
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
      OS << "const " << N << " = ";
    } else if (Call.getNumResults() > 1) {
      // Multi-return: TS array destructuring — `const [a, b] = f(args);`.
      OS << "const [";
      for (unsigned i = 0; i < Call.getNumResults(); ++i) {
        if (i) OS << ", ";
        OS << this->name(Call.getResult(i));
      }
      OS << "] = ";
    }
    {
      std::string Rewrite;
      if (tryRewriteAsClassCall(Call.getCallee(), Call.getOperands(),
                                 Rewrite)) {
        OS << dropOuterParens(Rewrite) << ";\n";
        return;
      }
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
    auto It = StringGlobalLits.find(A.getGlobalName());
    if (It != StringGlobalLits.end()) {
      OS << "const " << N << " = " << It->second << ";\n";
      SuppressedGlobals.insert(A.getGlobalName());
    } else {
      OS << "const " << N << " = " << A.getGlobalName().str() << ";\n";
    }
    return;
  }

  // --- arith binary ops ------------------------------------------------
  auto emitBin = [&](const char *CC) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << "const " << N << " = " << this->exprFor(Op.getOperand(0)) << " " << CC << " "
       << this->exprFor(Op.getOperand(1)) << ";\n";
  };
  if (mlir::isa<mlir::arith::AddFOp>(Op)) { emitBin("+"); return; }
  if (mlir::isa<mlir::arith::SubFOp>(Op)) { emitBin("-"); return; }
  if (mlir::isa<mlir::arith::MulFOp>(Op)) { emitBin("*"); return; }
  if (mlir::isa<mlir::arith::DivFOp>(Op)) { emitBin("/"); return; }
  if (mlir::isa<mlir::arith::AddIOp>(Op)) { emitBin("+"); return; }
  if (mlir::isa<mlir::arith::SubIOp>(Op)) { emitBin("-"); return; }
  if (mlir::isa<mlir::arith::MulIOp>(Op)) { emitBin("*"); return; }
  if (mlir::isa<mlir::arith::ShLIOp>(Op))  { emitBin("<<"); return; }
  if (mlir::isa<mlir::arith::ShRSIOp>(Op)) { emitBin(">>"); return; }
  if (mlir::isa<mlir::arith::ShRUIOp>(Op)) { emitBin(">>>"); return; }
  if (mlir::isa<mlir::arith::BitcastOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << "const " << N << " = " << this->exprFor(Op.getOperand(0)) << ";\n";
    return;
  }

  if (auto A = mlir::dyn_cast<mlir::arith::AndIOp>(Op)) {
    if (isI1(A.getType())) { emitBin("&&"); return; }
    emitBin("&"); return;
  }
  if (auto O = mlir::dyn_cast<mlir::arith::OrIOp>(Op)) {
    if (isI1(O.getType())) { emitBin("||"); return; }
    emitBin("|"); return;
  }
  if (auto X = mlir::dyn_cast<mlir::arith::XOrIOp>(Op)) {
    if (isI1(X.getType())) { emitBin("!=="); return; }
    emitBin("^"); return;
  }

  // --- arith.cmpf / cmpi ----------------------------------------------
  if (auto C = mlir::dyn_cast<mlir::arith::CmpFOp>(Op)) {
    const char *CC = "===";
    switch (C.getPredicate()) {
      case mlir::arith::CmpFPredicate::OEQ:
      case mlir::arith::CmpFPredicate::UEQ: CC = "==="; break;
      case mlir::arith::CmpFPredicate::ONE:
      case mlir::arith::CmpFPredicate::UNE: CC = "!=="; break;
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
    OS << "const " << N << " = " << this->exprFor(C.getLhs()) << " " << CC
       << " " << this->exprFor(C.getRhs()) << ";\n";
    return;
  }
  if (auto C = mlir::dyn_cast<mlir::arith::CmpIOp>(Op)) {
    const char *CC = "===";
    switch (C.getPredicate()) {
      case mlir::arith::CmpIPredicate::eq:  CC = "==="; break;
      case mlir::arith::CmpIPredicate::ne:  CC = "!=="; break;
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
    OS << "const " << N << " = " << this->exprFor(C.getLhs()) << " " << CC
       << " " << this->exprFor(C.getRhs()) << ";\n";
    return;
  }

  // --- Unregistered matlab.* binops ----------------------------------
  // Same scope-extension as EmitC/EmitPython: render frontend matlab.*
  // ops as the equivalent JS/TS operator. Lets HDL-source files that
  // don't depend on persistent state compile to TypeScript too.
  {
    llvm::StringRef MN = Op.getName().getStringRef();
    if (Op.getNumOperands() == 2 && Op.getNumResults() == 1) {
      const char *CC = nullptr;
      if (MN == "matlab.add") CC = "+";
      else if (MN == "matlab.sub") CC = "-";
      else if (MN == "matlab.emul" || MN == "matlab.matmul") CC = "*";
      else if (MN == "matlab.ediv" || MN == "matlab.matdiv") CC = "/";
      else if (MN == "matlab.eq") CC = "===";
      else if (MN == "matlab.ne") CC = "!==";
      else if (MN == "matlab.lt") CC = "<";
      else if (MN == "matlab.le") CC = "<=";
      else if (MN == "matlab.gt") CC = ">";
      else if (MN == "matlab.ge") CC = ">=";
      else if (MN == "matlab.short_or") CC = "||";
      else if (MN == "matlab.short_and") CC = "&&";
      if (CC) { emitBin(CC); return; }
    }
  }

  // --- arith casts ----------------------------------------------------
  if (mlir::isa<mlir::arith::SIToFPOp, mlir::arith::UIToFPOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    // No-op cast — JS numbers are doubles already.
    OS << "const " << N << " = " << this->stmtExpr(Op.getOperand(0)) << ";\n";
    return;
  }
  if (mlir::isa<mlir::arith::FPToSIOp, mlir::arith::FPToUIOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << "const " << N << " = (" << this->stmtExpr(Op.getOperand(0)) << " | 0);\n";
    return;
  }
  if (mlir::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                mlir::arith::TruncIOp, mlir::arith::TruncFOp,
                mlir::arith::ExtFOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << "const " << N << " = " << this->stmtExpr(Op.getOperand(0)) << ";\n";
    return;
  }

  // --- arith.select ---------------------------------------------------
  if (auto S = mlir::dyn_cast<mlir::arith::SelectOp>(Op)) {
    indent(Indent);
    std::string N = this->name(S.getResult());
    OS << "const " << N << " = " << this->exprFor(S.getCondition()) << " ? "
       << this->exprFor(S.getTrueValue()) << " : "
       << this->exprFor(S.getFalseValue()) << ";\n";
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
          if (InitVals[0]) { InitOK = false; break; }
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
        if (mlir::isa<mlir::LLVM::CallOp>(U)) continue;
        InitOK = false;
        break;
      }
      if (InitOK && Filled == N0)
        for (auto V : InitVals) if (!V) { InitOK = false; break; }
      if (InitOK && Filled == N0) {
        indent(Indent);
        OS << "const " << SlotName << " = [";
        for (uint64_t i = 0; i < N0; ++i) {
          if (i) OS << ", ";
          OS << this->stmtExpr(InitVals[i]);
        }
        OS << "];\n";
        for (auto *Op2 : AbsorbedOps) SuppressedOps.insert(Op2);
        return;
      }

      indent(Indent);
      OS << "const " << SlotName << " = new Array(" << N0 << ").fill(0.0);\n";
      return;
    }

    if (FusedForSlots.count(A.getOperation())) {
      auto NIt = FusedForSlotName.find(A.getOperation());
      if (NIt != FusedForSlotName.end()) {
        Names[A.getResult()] = NIt->second;
        DirectSlots[A.getOperation()] = NIt->second;
      }
      return;
    }

    Names[A.getResult()] = SlotName;
    DirectSlots[A.getOperation()] = SlotName;

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
          break;
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
    if (DropInit) {
      // First store will become the binding via the `let` form below.
      // But we still need to declare the variable up-front so reads
      // from outside the conditional see it. Declare without an
      // initializer; TypeScript accepts `let name;` and the first
      // assignment provides the value.
      indent(Indent);
      OS << "let " << SlotName << ";\n";
      return;
    }

    indent(Indent);
    OS << "let " << SlotName << " = 0;\n";
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
      OS << "const " << N << " = " << DirectSlots[AddrDef] << ";\n";
    } else if (AddrDef && ArraySlots.count(AddrDef)) {
      OS << "const " << N << " = " << ArraySlots[AddrDef] << ";\n";
    } else {
      OS << "const " << N << " = " << this->stmtExpr(L.getAddr()) << ";\n";
    }
    return;
  }
  if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
    mlir::Operation *AddrDef = S.getAddr().getDefiningOp();
    indent(Indent);
    if (AddrDef && DirectSlots.count(AddrDef)) {
      OS << DirectSlots[AddrDef] << " = " << this->stmtExpr(S.getValue())
         << ";\n";
    } else if (AddrDef && ArraySlots.count(AddrDef)) {
      OS << ArraySlots[AddrDef] << "[0] = "
         << this->stmtExpr(S.getValue()) << ";\n";
    } else {
      OS << this->stmtExpr(S.getAddr()) << " = "
         << this->stmtExpr(S.getValue()) << ";\n";
    }
    return;
  }

  // --- llvm.getelementptr ---------------------------------------------
  if (auto G = mlir::dyn_cast<mlir::LLVM::GEPOp>(Op)) {
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
    OS << "const " << N << " = " << Base << "[" << Idx << "];\n";
    return;
  }

  // --- scf.if ---------------------------------------------------------
  if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Op)) {
    auto FK = FlagIfKind.find(&Op);
    if (FK != FlagIfKind.end()) {
      indent(Indent);
      OS << "if (" << this->stmtExpr(If.getCondition()) << ") "
         << FK->second << ";\n";
      return;
    }
    if (InlinedIfs.count(&Op)) {
      emitRegion(If.getThenRegion(), Indent);
      return;
    }
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

    auto findElseElif = [&](mlir::scf::IfOp Parent)
        -> mlir::scf::IfOp {
      if (Parent.getElseRegion().empty()) return {};
      auto &EBlock = Parent.getElseRegion().front();
      mlir::scf::IfOp Inner;
      for (auto &Inn : EBlock.getOperations()) {
        if (mlir::isa<mlir::scf::YieldOp>(&Inn)) {
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
          if (Inner) return {};
          if (FlagIfKind.count(NestedIf.getOperation())) return {};
          if (InlinedIfs.count(NestedIf.getOperation())) return {};
          Inner = NestedIf;
          continue;
        }
        return {};
      }
      return Inner;
    };

    indent(Indent);
    OS << "if (" << this->stmtExpr(If.getCondition()) << ") {\n";
    if (countEmittedStmts(If.getThenRegion().front(), InlinedOps,
                           SuppressedOps) > 0) {
      emitRegion(If.getThenRegion(), Indent + 1);
    }
    indent(Indent); OS << "}";

    mlir::scf::IfOp Cur = If;
    while (true) {
      mlir::scf::IfOp Next = findElseElif(Cur);
      if (!Next) break;
      int InnerFold = evalConstCond(Next.getCondition());
      if (InnerFold == 1) {
        for (unsigned i = 0; i < Next.getNumResults(); ++i)
          Names[Next.getResult(i)] = this->name(Cur.getResult(i));
        OS << " else {\n";
        emitRegion(Next.getThenRegion(), Indent + 1);
        indent(Indent); OS << "}\n";
        return;
      }
      if (InnerFold == 0) {
        for (unsigned i = 0; i < Next.getNumResults(); ++i)
          Names[Next.getResult(i)] = this->name(Cur.getResult(i));
        if (!Next.getElseRegion().empty()) {
          OS << " else {\n";
          emitRegion(Next.getElseRegion(), Indent + 1);
          indent(Indent); OS << "}\n";
        } else {
          OS << "\n";
        }
        return;
      }
      for (unsigned i = 0; i < Next.getNumResults(); ++i)
        Names[Next.getResult(i)] = this->name(Cur.getResult(i));
      OS << " else if (" << this->stmtExpr(Next.getCondition()) << ") {\n";
      if (countEmittedStmts(Next.getThenRegion().front(), InlinedOps,
                             SuppressedOps) > 0) {
        emitRegion(Next.getThenRegion(), Indent + 1);
      }
      indent(Indent); OS << "}";
      Cur = Next;
    }

    if (!Cur.getElseRegion().empty() &&
        countEmittedStmts(Cur.getElseRegion().front(), InlinedOps,
                           SuppressedOps) > 0) {
      OS << " else {\n";
      emitRegion(Cur.getElseRegion(), Indent + 1);
      indent(Indent); OS << "}\n";
    } else {
      OS << "\n";
    }
    return;
  }

  // scf.yield inside scf.if / scf.while: assign to the parent's result
  // / iter-arg locals. The parent declares the locals up-front, so the
  // yield writes use bare assignment (no `let`).
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
      for (unsigned i = 0; i < Y.getNumOperands(); ++i) {
        auto BA = W.getBefore().front().getArgument(i);
        indent(Indent);
        OS << this->name(BA) << " = " << this->stmtExpr(Y.getOperand(i))
           << ";\n";
      }
      return;
    }
    return;
  }

  // --- scf.while ------------------------------------------------------
  if (auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Op)) {
    auto &Before = W.getBefore().front();
    auto &After = W.getAfter().front();

    auto FPIt = ForPatterns.find(W.getOperation());
    if (FPIt != ForPatterns.end()) {
      const ForLoopInfo &Info = FPIt->second;
      Names[Before.getArgument(0)] = Info.IvName;
      InlineExprs[After.getArgument(0)] = Info.IvName;
      if (W.getNumResults() == 1)
        InlineExprs[W.getResult(0)] = Info.IvName;

      long long IInit, IEnd, IStep;
      bool IntForm = forBoundsAreIntLiterals(Info, IInit, IEnd, IStep) &&
                     IStep != 0;
      indent(Indent);
      if (IntForm) {
        // C-style `for` loop with integer bounds. Native, no runtime
        // helper needed.
        const char *Cmp = IStep > 0 ? "<=" : ">=";
        const char *StepOp = IStep > 0 ? "+=" : "-=";
        long long AbsStep = IStep > 0 ? IStep : -IStep;
        OS << "for (let " << Info.IvName << " = " << IInit << "; "
           << Info.IvName << " " << Cmp << " " << IEnd << "; "
           << Info.IvName << " " << StepOp << " " << AbsStep << ") {\n";
      } else {
        // Runtime-bound iteration falls through `rt.frange`, a
        // generator that handles negative-step and floating bounds.
        OS << "for (const " << Info.IvName << " of rt.frange("
           << this->stmtExpr(Info.Init) << ", "
           << this->stmtExpr(Info.End) << ", "
           << this->stmtExpr(Info.Step) << ")) {\n";
      }
      int EmittedStmts = 0;
      for (auto &Inner : After.getOperations()) {
        if (mlir::isa<mlir::scf::YieldOp>(&Inner)) continue;
        if (InlinedOps.count(&Inner)) continue;
        if (SuppressedOps.count(&Inner)) continue;
        ++EmittedStmts;
      }
      if (EmittedStmts > 0) {
        for (auto &Inner : After.getOperations())
          emitOp(Inner, Indent + 1);
      }
      indent(Indent); OS << "}\n";
      return;
    }

    for (unsigned i = 0; i < W.getInits().size(); ++i) {
      auto BA = Before.getArgument(i);
      std::string N = freshName();
      Names[BA] = N;
      indent(Indent);
      OS << "let " << N << " = " << this->stmtExpr(W.getInits()[i]) << ";\n";
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

    auto emitStrippedCond = [&](mlir::Value V) -> std::string {
      llvm::SmallVector<mlir::Value, 2> Parts;
      gatherNonFlagConjuncts(V, Parts);
      if (Parts.empty()) return "true";
      std::string Out;
      for (unsigned i = 0; i < Parts.size(); ++i) {
        if (i) Out += " && ";
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
      OS << "while (" << emitStrippedCond(Cond.getCondition()) << ") {\n";
      for (auto &Inner : After.getOperations())
        emitOp(Inner, Indent + 1);
      indent(Indent); OS << "}\n";
      return;
    }

    indent(Indent);
    OS << "while (true) {\n";
    for (auto &Inner : Before.getOperations()) {
      if (auto Cond = mlir::dyn_cast<mlir::scf::ConditionOp>(Inner)) {
        std::string CondStr = emitStrippedCond(Cond.getCondition());
        if (CondStr != "true") {
          indent(Indent + 1);
          OS << "if (!(" << CondStr << ")) break;\n";
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
    indent(Indent); OS << "}\n";
    return;
  }

  // --- Unregistered matlab.call_builtin sites that survive to emit
  // time. The TS pipeline doesn't run LowerFixedPoint's matlab_*
  // → llvm.call sweep, so persistent set calls survive as
  // matlab.call_builtin and need explicit handling.
  if (Name == "matlab.call_builtin") {
    auto CA = Op.getAttrOfType<mlir::StringAttr>("callee");
    if (CA && CA.getValue() == "matlab_global_set_f64" &&
        Op.getNumOperands() == 2) {
      auto PN = Op.getAttrOfType<mlir::StringAttr>("persistent_name");
      auto PF = Op.getAttrOfType<mlir::StringAttr>("persistent_fn");
      if (PN && PF) {
        indent(Indent);
        OS << PF.getValue().str() << "_" << PN.getValue().str()
           << " = " << this->stmtExpr(Op.getOperand(1)) << ";\n";
        return;
      }
    }
  }

  // --- Fallback -------------------------------------------------------
  indent(Indent);
  OS << "// UNSUPPORTED: " << Name.str() << "\n";
  fail(("unsupported op in emitter: " + Name).str());
}

} // namespace

// Locate the rendered first newline so we can splice imports between
// the `// Generated ...` header line and the rest of the body.
static size_t firstNewlineAfterHeader(llvm::StringRef Body) {
  size_t Pos = Body.find('\n');
  return Pos == llvm::StringRef::npos ? Body.size() : Pos + 1;
}

// True iff the body actually references `<prefix>` as a symbol — checked
// via a small word-boundary scan so we don't match the substring inside
// a comment, identifier, or string literal that happens to contain it.
static bool bodyReferencesSymbol(llvm::StringRef Body,
                                 llvm::StringRef Prefix) {
  size_t I = 0;
  while ((I = Body.find(Prefix, I)) != llvm::StringRef::npos) {
    if (I > 0) {
      char C = Body[I - 1];
      if (std::isalnum((unsigned char)C) || C == '_' || C == '.') {
        I += Prefix.size();
        continue;
      }
    }
    // Reject matches inside a `// ...` comment line.
    bool InComment = false;
    for (ssize_t J = (ssize_t)I - 1; J >= 0 && Body[J] != '\n'; --J) {
      if (J >= 1 && Body[J - 1] == '/' && Body[J] == '/') {
        InComment = true; break;
      }
    }
    if (InComment) { I += Prefix.size(); continue; }
    return true;
  }
  return false;
}

std::string emitTypeScript(mlir::ModuleOp M, bool NoLine,
                           const matlab::SourceManager *SM) {
  std::ostringstream OSS;
  Emitter E(OSS, NoLine, SM);
  if (!E.run(M)) return {};
  std::string Body = OSS.str();

  size_t SpliceAt = firstNewlineAfterHeader(Body);
  std::string Imports;
  if (bodyReferencesSymbol(Body, "rt."))
    Imports += "import * as rt from \"./matlab_runtime\";\n";
  if (bodyReferencesSymbol(Body, "np."))
    Imports += "import * as np from \"./numpy_ts\";\n";
  Imports += "\n";

  return Body.substr(0, SpliceAt) + Imports + Body.substr(SpliceAt);
}

} // namespace mlirgen
} // namespace matlab
