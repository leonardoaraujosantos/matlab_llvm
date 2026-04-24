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
#include <iostream>
#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

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

  // --- Callee remap: llvm.call @matlab_foo -> rt.foo --------------------
  static std::string remapRuntimeCallee(llvm::StringRef Name);

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
  char Buf[64];
  double D = FA.getValueAsDouble();
  snprintf(Buf, sizeof(Buf), "%.17g", D);
  std::string S = Buf;
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
    // Addressof of a module-scope global: bare name. Works for both
    // string globals and function symbols (function handles decay to
    // the Python callable by name).
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
    // Python ternary: `a if c else b`.
    Expr = "(" + exprFor(S.getTrueValue()) + " if "
         + exprFor(S.getCondition()) + " else "
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
    std::string E = Callee + "(";
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

  // Pass 1: module-scope globals (string constants).
  bool AnyGlobal = false;
  for (auto &Op : M.getBody()->getOperations()) {
    auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op);
    if (!G) continue;
    emitGlobal(G);
    AnyGlobal = true;
  }
  if (AnyGlobal) OS << "\n";

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
  LastLineFile.clear();
  LastLineNum = -1;
  computeInlines(F.getBody());
  auto FT = F.getFunctionType();
  bool IsMain = F.getSymName() == "main";

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
    return;
  }
  emitRegion(F.getBody(), 1);
  OS << "\n";
}

void Emitter::emitLLVMFunc(mlir::LLVM::LLVMFuncOp F) {
  NextId = 0;
  InlineExprs.clear();
  InlinedOps.clear();
  DirectSlots.clear();
  ArraySlots.clear();
  LastLineFile.clear();
  LastLineNum = -1;
  computeInlines(F.getBody());
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
    indent(1); OS << "pass\n\n"; return;
  }
  emitRegion(F.getBody(), 1);
  OS << "\n";
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

// Count real (non-yield, non-condition, non-inlined-no-op) statements a
// block will emit. Used to decide whether to insert a `pass` placeholder.
static int countEmittedStmts(mlir::Block &B,
                             const llvm::DenseSet<mlir::Operation *> &Inlined) {
  int N = 0;
  for (auto &Op : B.getOperations()) {
    if (mlir::isa<mlir::scf::YieldOp, mlir::scf::ConditionOp>(&Op)) continue;
    if (Inlined.count(&Op)) continue;
    ++N;
  }
  return N;
}

void Emitter::emitOp(mlir::Operation &Op, int Indent) {
  llvm::StringRef Name = Op.getName().getStringRef();

  if (InlinedOps.count(&Op)) return;

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
    indent(Indent);
    if (R.getNumOperands() == 0) OS << "return\n";
    else OS << "return " << this->stmtExpr(R.getOperand(0)) << "\n";
    return;
  }
  if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
    if (InMainHoist) return;
    indent(Indent);
    if (R.getNumOperands() == 0) OS << "return\n";
    else OS << "return " << this->stmtExpr(R.getOperand(0)) << "\n";
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
      OS << remapRuntimeCallee(*Callee) << "(";
      for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
        if (i) OS << ", ";
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
    OS << N << " = " << A.getGlobalName().str() << "\n";
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
      // Python: list of floats is fine for small numeric buffers; the
      // runtime wraps into np.array when handing off to matrix ops.
      Names[A.getResult()] = SlotName;
      ArraySlots[A.getOperation()] = SlotName;
      indent(Indent);
      OS << SlotName << " = [0.0] * " << N0 << "\n";
      return;
    }

    // Scalar slot: every use we've seen from Mem2RegLite is load/store.
    // Bind the alloca SSA value to the slot name; downstream load/store
    // treat it as a direct Python variable.
    Names[A.getResult()] = SlotName;
    DirectSlots[A.getOperation()] = SlotName;
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
    // Declare result locals (one per scf.if result) so yields assign
    // to them.
    for (unsigned i = 0; i < If.getNumResults(); ++i) {
      std::string N = this->name(If.getResult(i));
      indent(Indent);
      OS << N << " = 0\n";
    }
    indent(Indent);
    OS << "if " << this->stmtExpr(If.getCondition()) << ":\n";
    if (countEmittedStmts(If.getThenRegion().front(), InlinedOps) == 0 &&
        If.getThenRegion().front().getOperations().size() <= 1) {
      indent(Indent + 1); OS << "pass\n";
    } else {
      emitRegion(If.getThenRegion(), Indent + 1);
    }
    if (!If.getElseRegion().empty()) {
      indent(Indent);
      OS << "else:\n";
      if (countEmittedStmts(If.getElseRegion().front(), InlinedOps) == 0 &&
          If.getElseRegion().front().getOperations().size() <= 1) {
        indent(Indent + 1); OS << "pass\n";
      } else {
        emitRegion(If.getElseRegion(), Indent + 1);
      }
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

    if (BeforeIsCondOnly) {
      auto Cond = mlir::cast<mlir::scf::ConditionOp>(Before.getTerminator());
      for (unsigned i = 0; i < Cond.getArgs().size(); ++i) {
        auto AA = After.getArgument(i);
        InlineExprs[AA] = this->exprFor(Cond.getArgs()[i]);
      }
      indent(Indent);
      OS << "while " << this->stmtExpr(Cond.getCondition()) << ":\n";
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
        indent(Indent + 1);
        OS << "if not " << this->exprFor(Cond.getCondition())
           << ": break\n";
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
