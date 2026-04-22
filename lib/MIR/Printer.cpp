#include "matlab/MIR/Printer.h"

#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace matlab {
namespace mir {

namespace {

// Maps OpKind to the textual op name ("matlab.add", "scf.if", etc.).
const char *opName(OpKind K) {
  switch (K) {
  case OpKind::ModuleOp:       return "matlab.module";
  case OpKind::FuncOp:         return "matlab.func";
  case OpKind::ConstInt:       return "matlab.const_int";
  case OpKind::ConstFloat:     return "matlab.const_float";
  case OpKind::ConstComplex:   return "matlab.const_complex";
  case OpKind::ConstString:    return "matlab.const_str";
  case OpKind::ConstChar:      return "matlab.const_char";
  case OpKind::ConstLogical:   return "matlab.const_logical";
  case OpKind::ConstColon:     return "matlab.colon";
  case OpKind::ConstEnd:       return "matlab.end";
  case OpKind::Alloc:          return "matlab.alloc";
  case OpKind::Load:           return "matlab.load";
  case OpKind::Store:          return "matlab.store";
  case OpKind::Neg:            return "matlab.neg";
  case OpKind::UPlus:          return "matlab.uplus";
  case OpKind::Not:            return "matlab.not";
  case OpKind::Add:            return "matlab.add";
  case OpKind::Sub:            return "matlab.sub";
  case OpKind::MatMul:         return "matlab.matmul";
  case OpKind::MatDiv:         return "matlab.matdiv";
  case OpKind::MatLDiv:        return "matlab.matldiv";
  case OpKind::MatPow:         return "matlab.matpow";
  case OpKind::EMul:           return "matlab.emul";
  case OpKind::EDiv:           return "matlab.ediv";
  case OpKind::ELDiv:          return "matlab.eldiv";
  case OpKind::EPow:           return "matlab.epow";
  case OpKind::Eq:             return "matlab.eq";
  case OpKind::Ne:             return "matlab.ne";
  case OpKind::Lt:             return "matlab.lt";
  case OpKind::Le:             return "matlab.le";
  case OpKind::Gt:             return "matlab.gt";
  case OpKind::Ge:             return "matlab.ge";
  case OpKind::BitAnd:         return "matlab.and";
  case OpKind::BitOr:          return "matlab.or";
  case OpKind::ShortAnd:       return "matlab.short_and";
  case OpKind::ShortOr:        return "matlab.short_or";
  case OpKind::Transpose:      return "matlab.transpose";
  case OpKind::CTranspose:     return "matlab.ctranspose";
  case OpKind::Range:          return "matlab.range";
  case OpKind::ConcatRow:      return "matlab.concat_row";
  case OpKind::ConcatCol:      return "matlab.concat_col";
  case OpKind::Subscript:      return "matlab.subscript";
  case OpKind::CellSubscript:  return "matlab.cell_subscript";
  case OpKind::FieldAccess:    return "matlab.field";
  case OpKind::Call:           return "matlab.call";
  case OpKind::CallBuiltin:    return "matlab.call_builtin";
  case OpKind::CallIndirect:   return "matlab.call_indirect";
  case OpKind::MakeHandle:     return "matlab.make_handle";
  case OpKind::MakeAnon:       return "matlab.make_anon";
  case OpKind::IfOp:           return "scf.if";
  case OpKind::ForOp:          return "scf.for";
  case OpKind::WhileOp:        return "scf.while";
  case OpKind::Yield:          return "scf.yield";
  case OpKind::Return:         return "matlab.return";
  case OpKind::Break:          return "matlab.break";
  case OpKind::Continue:       return "matlab.continue";
  case OpKind::MakeMatrix:     return "matlab.make_matrix";
  case OpKind::MakeCell:       return "matlab.make_cell";
  case OpKind::DynamicField:   return "matlab.dyn_field";
  }
  return "matlab.unknown";
}

std::string typeStr(const Type *T) {
  return T ? T->toString() : "?";
}

std::string valueName(const Value *V, std::unordered_map<const Value*, int> &Names, int &Next) {
  if (!V) return "%<null>";
  auto It = Names.find(V);
  if (It != Names.end()) return "%" + std::to_string(It->second);
  int Id = Next++;
  Names.emplace(V, Id);
  return "%" + std::to_string(Id);
}

void indent(std::ostream &OS, unsigned N) {
  for (unsigned i = 0; i < N; ++i) OS << "  ";
}

void printAttr(std::ostream &OS, const Attribute &A) {
  switch (A.K) {
  case Attribute::Kind::None:  OS << "none"; break;
  case Attribute::Kind::Int:   OS << A.I; break;
  case Attribute::Kind::Float: OS << A.F; break;
  case Attribute::Kind::Str: {
    OS << '"';
    for (char c : A.S) {
      if (c == '"' || c == '\\') OS << '\\';
      OS << c;
    }
    OS << '"';
    break;
  }
  case Attribute::Kind::Sym:  OS << '@' << A.S; break;
  case Attribute::Kind::Bool: OS << (A.I ? "true" : "false"); break;
  }
}

class Printer {
public:
  Printer(std::ostream &OS) : OS(OS) {}

  void print(const Op &O, unsigned Ind);

private:
  std::ostream &OS;
  std::unordered_map<const Value *, int> Names;
  int NextName = 0;

  std::string name(const Value *V) { return valueName(V, Names, NextName); }

  void printResults(const Op &O);
  void printOperands(const Op &O);
  void printAttrs(const Op &O);
  void printResultTypes(const Op &O);
  void printRegion(const Region *R, unsigned Ind);
  void printBlock(const Block *B, unsigned Ind);
};

void Printer::printResults(const Op &O) {
  if (O.Results.empty()) return;
  for (size_t i = 0; i < O.Results.size(); ++i) {
    if (i) OS << ", ";
    OS << name(O.Results[i]);
  }
  OS << " = ";
}

void Printer::printOperands(const Op &O) {
  if (O.Operands.empty()) { OS << "()"; return; }
  OS << "(";
  for (size_t i = 0; i < O.Operands.size(); ++i) {
    if (i) OS << ", ";
    OS << name(O.Operands[i]);
  }
  OS << ")";
}

void Printer::printAttrs(const Op &O) {
  if (O.Attrs.empty()) return;
  OS << " {";
  for (size_t i = 0; i < O.Attrs.size(); ++i) {
    if (i) OS << ", ";
    OS << O.Attrs[i].first << " = ";
    printAttr(OS, O.Attrs[i].second);
  }
  OS << "}";
}

void Printer::printResultTypes(const Op &O) {
  if (O.Results.empty()) return;
  OS << " : ";
  if (O.Results.size() > 1) OS << "(";
  for (size_t i = 0; i < O.Results.size(); ++i) {
    if (i) OS << ", ";
    OS << typeStr(O.Results[i]->Ty);
  }
  if (O.Results.size() > 1) OS << ")";
}

void Printer::printBlock(const Block *B, unsigned Ind) {
  if (!B) return;
  // Suppress the block label when it's the entry block of a parent op whose
  // signature already displays the arguments (FuncOp, MakeAnon).
  bool IsEntry = false;
  if (B->Parent && B->Parent->Parent) {
    Op *Parent = B->Parent->Parent;
    if ((Parent->K == OpKind::FuncOp || Parent->K == OpKind::MakeAnon) &&
        B == B->Parent->Blocks.front()) {
      IsEntry = true;
      // Still seed Names with the block args so they print with the right IDs.
      for (Value *A : B->Arguments) (void)name(A);
    }
  }
  if (!IsEntry && !B->Arguments.empty()) {
    indent(OS, Ind);
    OS << "^bb" << B->Id << "(";
    for (size_t i = 0; i < B->Arguments.size(); ++i) {
      if (i) OS << ", ";
      OS << name(B->Arguments[i]) << ": " << typeStr(B->Arguments[i]->Ty);
    }
    OS << "):\n";
  }
  for (const Op *O : B->Ops) print(*O, Ind);
}

void Printer::printRegion(const Region *R, unsigned Ind) {
  if (!R) return;
  OS << " {\n";
  for (const Block *B : R->Blocks) printBlock(B, Ind + 1);
  indent(OS, Ind);
  OS << "}";
}

void Printer::print(const Op &O, unsigned Ind) {
  indent(OS, Ind);
  printResults(O);
  OS << opName(O.K);

  // Special-case FuncOp to print a function-signature style.
  if (O.K == OpKind::FuncOp) {
    const Attribute *Name = O.getAttr("name");
    if (Name) OS << " @" << Name->S;
    // Print the entry-block signature as the function's params.
    OS << "(";
    if (!O.Regions.empty() && !O.Regions[0]->Blocks.empty()) {
      const Block *Entry = O.Regions[0]->Blocks.front();
      for (size_t i = 0; i < Entry->Arguments.size(); ++i) {
        if (i) OS << ", ";
        OS << name(Entry->Arguments[i]) << ": " << typeStr(Entry->Arguments[i]->Ty);
      }
    }
    OS << ")";
    printAttrs(O);
  } else {
    printOperands(O);
    printAttrs(O);
    printResultTypes(O);
  }

  for (Region *R : O.Regions) printRegion(R, Ind);
  OS << "\n";
}

} // namespace

void printOp(std::ostream &OS, const Op &O, unsigned Indent) {
  Printer P(OS);
  P.print(O, Indent);
}

void printModule(std::ostream &OS, const Module &M) {
  if (!M.ModuleOp) return;
  Printer P(OS);
  P.print(*M.ModuleOp, 0);
}

} // namespace mir
} // namespace matlab
