// Emits C (or C++) source from an MLIR ModuleOp whose ops have already been
// lowered to a small, closed set: func / arith / scf / cf / llvm.call /
// llvm.alloca / llvm.load / llvm.store / llvm.mlir.global / llvm.mlir.addressof
// plus outlined llvm.func bodies (parfor / anonymous functions).
//
// The emitter walks the module, assigns every mlir::Value a stable C
// identifier, and prints statements as it visits ops in source order.
// The output is intended to be linked against runtime/matlab_runtime.c.

#include "matlab/MLIR/Passes/Passes.h"

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

/// Per-module emission state: symbol table for SSA values and a single
/// output stream. Kept in a class because the recursive descent over
/// regions needs to share state across helpers.
class Emitter {
public:
  Emitter(std::ostream &OS, bool Cpp) : OS(OS), Cpp(Cpp) {}

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

  // --- Helpers -----------------------------------------------------------
  void indent(int N) { for (int i = 0; i < N; ++i) OS << "  "; }
  std::string constStr(mlir::LLVM::GlobalOp G);
  void fail(llvm::StringRef Msg) {
    if (!Failed)
      std::cerr << "error: emit-c: " << Msg.str() << "\n";
    Failed = true;
  }
  void emitLineDirective(mlir::Location L, int Indent);

  std::ostream &OS;
  bool Cpp;
  bool Failed = false;

  llvm::DenseMap<mlir::Value, std::string> Names;
  llvm::DenseMap<mlir::Operation *, std::string> GlobalStrs;  // global -> C name
  llvm::StringSet<> UsedNames;  // identifiers already claimed.
  int NextId = 0;

  // Most recent #line directive emitted — used to dedupe.
  std::string LastLineFile;
  int LastLineNum = -1;
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

// Emit a `#line N "file"` directive at the given indent level if the
// location is a FileLineColLoc and hasn't been emitted already for this
// (file, line) pair.
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
  std::string File = FL.getFilename().str();
  int Line = static_cast<int>(FL.getLine());
  if (File.empty() || Line <= 0) return;
  // Emit only the basename so the generated .c is portable across
  // build machines and doesn't leak an absolute path. Debuggers resolve
  // #line filenames against the compilation directory.
  if (auto Slash = File.find_last_of("/\\"); Slash != std::string::npos)
    File = File.substr(Slash + 1);
  if (File == LastLineFile && Line == LastLineNum) return;
  LastLineFile = File;
  LastLineNum = Line;
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
  OS << "// Generated by matlabc -emit-c. Do not edit.\n";
  OS << "#include <stdint.h>\n";
  if (!Cpp) OS << "#include <stdbool.h>\n";
  OS << "\n";
  // Runtime function prototypes are emitted per-module below, with void*
  // for all pointer params so the same declaration works for C and C++
  // (C linkage handles the type bridging to the runtime's typed params).
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

bool Emitter::run(mlir::ModuleOp M) {
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
  // LowerTensorOps / LowerParfor).
  OS << "// Runtime prototypes (linked against runtime/matlab_runtime.c).\n";
  if (Cpp) OS << "extern \"C\" {\n";
  for (auto &Op : M.getBody()->getOperations()) {
    auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op);
    if (!F) continue;
    if (!F.getBody().empty()) continue;  // skip defined funcs.
    UsedNames.insert(F.getSymName().str());
    auto FT = F.getFunctionType();
    std::string RetTy =
        mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())
            ? std::string("void")
            : cTypeOf(FT.getReturnType());
    OS << "extern " << RetTy << " " << F.getSymName().str() << "(";
    for (unsigned i = 0; i < FT.getNumParams(); ++i) {
      if (i) OS << ", ";
      OS << cTypeOf(FT.getParamType(i));
    }
    if (FT.getNumParams() == 0) OS << "void";
    OS << ");\n";
  }
  if (Cpp) OS << "} // extern \"C\"\n";
  OS << "\n";

  // Pass 1: llvm.mlir.global string constants. Reserve symbol names first
  // so body-local identifiers won't collide with them.
  OS << "// Module-level string constants.\n";
  for (auto &Op : M.getBody()->getOperations()) {
    if (auto G = mlir::dyn_cast<mlir::LLVM::GlobalOp>(Op))
      emitGlobal(G);
  }
  OS << "\n";

  // Pass 2: forward-declare every defined function so call ordering doesn't
  // matter. Reserve the function's symbol name so body-local identifiers
  // can't collide (important now that locals may inherit MATLAB names).
  OS << "// Forward declarations.\n";
  for (auto &Op : M.getBody()->getOperations()) {
    if (auto F = mlir::dyn_cast<mlir::func::FuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      UsedNames.insert(F.getSymName().str());
      if (F.getSymName() == "main") continue;  // main has no forward decl.
      auto FT = F.getFunctionType();
      std::string RetTy = FT.getNumResults() == 0
                              ? std::string("void")
                              : cTypeOf(FT.getResult(0));
      OS << "static " << RetTy << " " << F.getSymName().str() << "(";
      for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
        if (i) OS << ", ";
        OS << cTypeOf(FT.getInput(i));
      }
      if (FT.getNumInputs() == 0) OS << "void";
      OS << ");\n";
    } else if (auto F = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(Op)) {
      if (F.getBody().empty()) continue;
      UsedNames.insert(F.getSymName().str());
      if (F.getSymName() == "main") continue;
      auto FT = F.getFunctionType();
      std::string RetTy =
          mlir::isa<mlir::LLVM::LLVMVoidType>(FT.getReturnType())
              ? std::string("void")
              : cTypeOf(FT.getReturnType());
      OS << "static " << RetTy << " " << F.getSymName().str() << "(";
      for (unsigned i = 0; i < FT.getNumParams(); ++i) {
        if (i) OS << ", ";
        OS << cTypeOf(FT.getParamType(i));
      }
      if (FT.getNumParams() == 0) OS << "void";
      OS << ");\n";
    }
  }
  OS << "\n";

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
  for (auto &Op : B.getOperations())
    emitOp(Op, Indent);
}

// ---------------------------------------------------------------------------
// Per-op emission. Initial version: stub that dumps the op mnemonic so we
// can see what survives into the snapshot. We'll replace case-by-case.
// ---------------------------------------------------------------------------

void Emitter::emitOp(mlir::Operation &Op, int Indent) {
  llvm::StringRef Name = Op.getName().getStringRef();

  // Emit a #line directive if this op has a FileLineColLoc that differs
  // from the last directive we printed. Deduped inside emitLineDirective,
  // so constants / pure expression ops don't pollute the output.
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
      OS << IA.getInt();
    } else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(V)) {
      char Buf[64];
      snprintf(Buf, sizeof(Buf), "%.17g", FA.getValueAsDouble());
      OS << Buf;
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
      // Print with enough precision to round-trip.
      double D = FA.getValueAsDouble();
      char Buf[64];
      snprintf(Buf, sizeof(Buf), "%.17g", D);
      OS << Buf;
    } else if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(V)) {
      OS << IA.getInt();
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
    else OS << "return " << this->name(R.getOperand(0)) << ";\n";
    return;
  }
  if (auto R = mlir::dyn_cast<mlir::LLVM::ReturnOp>(Op)) {
    indent(Indent);
    if (R.getNumOperands() == 0) OS << "return;\n";
    else OS << "return " << this->name(R.getOperand(0)) << ";\n";
    return;
  }

  // --- llvm.call / func.call ------------------------------------------
  if (auto Call = mlir::dyn_cast<mlir::LLVM::CallOp>(Op)) {
    indent(Indent);
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult());
      OS << cTypeOfValue(Call.getResult()) << " " << N << " = ";
    }
    if (auto Callee = Call.getCallee()) {
      OS << Callee->str() << "(";
      for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
        if (i) OS << ", ";
        OS << this->name(Call.getOperand(i));
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
      OS << "))" << this->name(Call.getOperand(0)) << ")(";
      for (unsigned i = 1; i < Call.getNumOperands(); ++i) {
        if (i > 1) OS << ", ";
        OS << this->name(Call.getOperand(i));
      }
      OS << ");\n";
    }
    return;
  }
  if (auto Call = mlir::dyn_cast<mlir::func::CallOp>(Op)) {
    indent(Indent);
    if (Call.getNumResults() == 1) {
      std::string N = this->name(Call.getResult(0));
      OS << cTypeOfValue(Call.getResult(0)) << " " << N << " = ";
    }
    OS << Call.getCallee().str() << "(";
    for (unsigned i = 0; i < Call.getNumOperands(); ++i) {
      if (i) OS << ", ";
      OS << this->name(Call.getOperand(i));
    }
    OS << ");\n";
    return;
  }

  // --- llvm.mlir.addressof --------------------------------------------
  if (auto A = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(Op)) {
    std::string N = this->name(A.getResult());
    indent(Indent);
    OS << "void* " << N << " = (void*)&" << A.getGlobalName().str() << ";\n";
    return;
  }

  // --- arith binary ops on floats -------------------------------------
  auto emitBinF = [&](const char *CC) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = "
       << this->name(Op.getOperand(0)) << " " << CC << " "
       << this->name(Op.getOperand(1)) << ";\n";
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
    OS << "bool " << N << " = (" << this->name(C.getLhs()) << " " << CC << " "
       << this->name(C.getRhs()) << ");\n";
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
    OS << "bool " << N << " = (" << this->name(C.getLhs()) << " " << CC << " "
       << this->name(C.getRhs()) << ");\n";
    return;
  }

  // --- arith casts ----------------------------------------------------
  if (mlir::isa<mlir::arith::SIToFPOp, mlir::arith::UIToFPOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = (double)"
       << this->name(Op.getOperand(0)) << ";\n";
    return;
  }
  if (mlir::isa<mlir::arith::FPToSIOp, mlir::arith::FPToUIOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = ("
       << cTypeOfValue(Op.getResult(0)) << ")"
       << this->name(Op.getOperand(0)) << ";\n";
    return;
  }
  if (mlir::isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp,
                mlir::arith::TruncIOp, mlir::arith::TruncFOp,
                mlir::arith::ExtFOp>(Op)) {
    indent(Indent);
    std::string N = this->name(Op.getResult(0));
    OS << cTypeOfValue(Op.getResult(0)) << " " << N << " = ("
       << cTypeOfValue(Op.getResult(0)) << ")"
       << this->name(Op.getOperand(0)) << ";\n";
    return;
  }

  // --- llvm.alloca / load / store -------------------------------------
  if (auto A = mlir::dyn_cast<mlir::LLVM::AllocaOp>(Op)) {
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
      // The pointer value itself still needs a unique identifier.
      N = uniqueName(SlotName + "_p");
    } else {
      N = this->name(A.getResult());
      SlotName = N + "_slot";
    }
    Names[A.getResult()] = N;
    // Two shapes appear:
    //   1) alloca<T> with ArraySize=1 — a scalar slot (LowerScalarSlots).
    //   2) alloca<!llvm.array<N x T>> with ArraySize=1 — a contiguous buffer
    //      for a matrix literal (LowerTensorOps::materializeMat).
    mlir::Type ET = A.getElemType();
    if (auto AT = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(ET)) {
      std::string ElTy = cTypeOf(AT.getElementType());
      uint64_t N0 = AT.getNumElements();
      indent(Indent);
      OS << ElTy << " " << SlotName << "[" << N0 << "] = {0};\n";
      indent(Indent);
      OS << "void* " << N << " = (void*)" << SlotName << ";\n";
      return;
    }
    std::string ElTy = cTypeOf(ET);
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
       << this->name(G.getBase()) << ")";
    // Inline constant indices; SSA indices use their names.
    for (auto Idx : G.getIndices()) {
      if (auto V = llvm::dyn_cast<mlir::Value>(Idx)) {
        OS << " + " << this->name(V);
      } else if (auto A = llvm::dyn_cast<mlir::IntegerAttr>(Idx)) {
        OS << " + " << A.getInt();
      }
    }
    OS << ");\n";
    return;
  }
  if (auto L = mlir::dyn_cast<mlir::LLVM::LoadOp>(Op)) {
    indent(Indent);
    std::string N = this->name(L.getResult());
    std::string ResTy = cTypeOfValue(L.getResult());
    OS << ResTy << " " << N << " = *(" << ResTy << "*)"
       << this->name(L.getAddr()) << ";\n";
    return;
  }
  if (auto S = mlir::dyn_cast<mlir::LLVM::StoreOp>(Op)) {
    indent(Indent);
    std::string Ty = cTypeOfValue(S.getValue());
    OS << "*(" << Ty << "*)" << this->name(S.getAddr()) << " = "
       << this->name(S.getValue()) << ";\n";
    return;
  }

  // --- scf.if ---------------------------------------------------------
  if (auto If = mlir::dyn_cast<mlir::scf::IfOp>(Op)) {
    // Declare result locals (one per scf.if result) so yield can assign
    // to them and uses outside the if can reference the names directly.
    for (unsigned i = 0; i < If.getNumResults(); ++i) {
      std::string N = this->name(If.getResult(i));
      indent(Indent);
      OS << cTypeOfValue(If.getResult(i)) << " " << N << " = 0;\n";
    }
    indent(Indent);
    OS << "if (" << this->name(If.getCondition()) << ") {\n";
    emitRegion(If.getThenRegion(), Indent + 1);
    if (!If.getElseRegion().empty()) {
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
           << this->name(Y.getOperand(i)) << ";\n";
      }
      return;
    }
    if (auto W = mlir::dyn_cast<mlir::scf::WhileOp>(Parent)) {
      // After-region yield: update iter locals (shared names with iter-args).
      for (unsigned i = 0; i < Y.getNumOperands(); ++i) {
        auto BA = W.getBefore().front().getArgument(i);
        indent(Indent);
        OS << this->name(BA) << " = " << this->name(Y.getOperand(i))
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

    // Declare one mutable local per iter-arg, initialized from the while
    // operand, and bind the before-block arg to that name so references
    // inside the before region resolve to it.
    for (unsigned i = 0; i < W.getInits().size(); ++i) {
      auto BA = Before.getArgument(i);
      std::string N = freshName();
      Names[BA] = N;
      indent(Indent);
      OS << cTypeOf(BA.getType()) << " " << N << " = "
         << this->name(W.getInits()[i]) << ";\n";
    }
    // Result locals for scf.while's outer results: mirror iter-arg names
    // (same storage is used on exit). Bind result SSA values to the same
    // names so callers downstream see the right identifiers.
    for (unsigned i = 0; i < W.getNumResults(); ++i) {
      auto BA = Before.getArgument(i);
      Names[W.getResult(i)] = Names[BA];
    }

    indent(Indent);
    OS << "while (1) {\n";

    // Emit the before region body, then the scf.condition terminator as
    // `if (!cond) break;`, forwarding values into the after-block args.
    for (auto &Inner : Before.getOperations()) {
      if (auto Cond = mlir::dyn_cast<mlir::scf::ConditionOp>(Inner)) {
        indent(Indent + 1);
        OS << "if (!" << this->name(Cond.getCondition()) << ") break;\n";
        // Bind after-block args to the forwarded values' names.
        for (unsigned i = 0; i < Cond.getArgs().size(); ++i) {
          auto AA = After.getArgument(i);
          Names[AA] = this->name(Cond.getArgs()[i]);
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
    OS << cTypeOfValue(S.getResult()) << " " << N << " = ("
       << this->name(S.getCondition()) << ") ? "
       << this->name(S.getTrueValue()) << " : "
       << this->name(S.getFalseValue()) << ";\n";
    return;
  }

  // --- arith bitwise / logical ops on integers / i1 -------------------
  if (mlir::isa<mlir::arith::AndIOp>(Op)) { emitBinF("&"); return; }
  if (mlir::isa<mlir::arith::OrIOp>(Op))  { emitBinF("|"); return; }
  if (mlir::isa<mlir::arith::XOrIOp>(Op)) { emitBinF("^"); return; }

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

std::string emitC(mlir::ModuleOp M, bool Cpp) {
  std::ostringstream OSS;
  Emitter E(OSS, Cpp);
  if (!E.run(M)) return {};
  return OSS.str();
}

} // namespace mlirgen
} // namespace matlab
