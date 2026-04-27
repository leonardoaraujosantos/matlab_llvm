// Phase 5.5 — pre-synthesis hardware report.
//
// Walks the post-SV-pipeline module and emits a Markdown summary
// per user func.func. The report's purpose is visibility before
// the user invokes a downstream synthesis tool: gate counts come
// from the synth tool, but operator counts / register widths /
// FSM states are exactly what the SV emitter would produce, so
// the report is a stable artifact for code review.
//
// Counted quantities (per function):
//   - Inferred class:        combinational / clocked /
//                            FSM-bearing / RAM-using
//   - Input + output ports:  width and signed/unsigned per port
//   - Operators by kind:     add / sub / mul / div / cmp /
//                            shift / bitwise (and/or/xor/not)
//   - Registers:             count and total flip-flop bits
//   - FSMs:                  per-register state count plus
//                            chosen encoding (binary / one-hot /
//                            gray) when set via pragma
//   - RAM:                   not yet detected (Phase 4 RAM
//                            inference is deferred to a future
//                            round)

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <ostream>

namespace matlab {
namespace mlirgen {

namespace {

bool isScriptFunc(mlir::func::FuncOp F) {
  llvm::StringRef N = F.getSymName();
  return N == "script" || N == "main";
}

unsigned widthOf(mlir::Type T) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) return IT.getWidth();
  return 0;
}

std::string formatType(mlir::Type T) {
  if (auto IT = mlir::dyn_cast<mlir::IntegerType>(T)) {
    if (IT.getWidth() == 1) return "logic";
    return "logic signed [" + std::to_string(IT.getWidth() - 1) + ":0]";
  }
  return "?";
}

/// Per-function counters. Plain ints; the report renders them
/// directly as Markdown table rows.
struct FuncCounts {
  unsigned Adds = 0, Subs = 0, Muls = 0, Divs = 0;
  unsigned Cmps = 0, Shifts = 0, BitOps = 0, Selects = 0;
  unsigned Constants = 0;  // useful as a sanity check
  // Width-summed cost: the bit-budget impact of each op when the
  // synth tool unrolls. e.g. one i16 + i16 add adds 16 to AddBits.
  // Approximate; real synth area depends on the cell library.
  unsigned AddBits = 0, SubBits = 0, MulBits = 0;
  // Register info filled from HWPersistentInfo.
  llvm::SmallVector<HWPersistentInfo, 4> Persists;
  // Per-FSM info: register name + state count + encoding.
  struct FSMReport {
    std::string RegName;
    unsigned States = 0;
    std::string Encoding;
  };
  llvm::SmallVector<FSMReport, 4> FSMs;
  bool HasClock = false;
  bool HasReset = false;
  std::string ResetKind;  // "async-low" / "sync-high" / etc.
};

/// Reuse the FSM-cascade matcher logic from EmitSystemVerilog —
/// duplicated here as a free function to avoid coupling. Counts
/// the number of distinct state constants in any cascade rooted
/// at a persistent-get on each register.
unsigned countStatesForRegister(mlir::func::FuncOp F,
                                 mlir::Operation *AnyGetForReg) {
  llvm::SmallSet<int64_t, 8> States;
  // Find all scf.if conditions that compare a persistent get
  // (any of P.Gets — but here we only have one example) against
  // a constant. We look at every cmpf-oeq / matlab.eq result that
  // gates an scf.if; if its LHS's defining op is the same callee
  // as AnyGetForReg's callee + same idx attr, we count.
  if (!AnyGetForReg) return 0;
  llvm::StringRef WantName;
  if (auto S = AnyGetForReg->getAttrOfType<mlir::StringAttr>("persistent_name"))
    WantName = S.getValue();
  if (WantName.empty()) return 0;
  F.walk([&](mlir::scf::IfOp If) {
    auto *CondOp = If.getCondition().getDefiningOp();
    if (!CondOp) return;
    mlir::Value Lhs, Rhs;
    if (auto Cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(CondOp)) {
      if (Cmp.getPredicate() != mlir::arith::CmpFPredicate::OEQ) return;
      Lhs = Cmp.getLhs(); Rhs = Cmp.getRhs();
    } else if (CondOp->getName().getStringRef() == "matlab.eq") {
      if (CondOp->getNumOperands() != 2) return;
      Lhs = CondOp->getOperand(0); Rhs = CondOp->getOperand(1);
    } else {
      return;
    }
    auto *L = Lhs.getDefiningOp();
    if (!L) return;
    auto LSym = L->getAttrOfType<mlir::StringAttr>("persistent_name");
    if (!LSym || LSym.getValue() != WantName) return;
    auto C = Rhs.getDefiningOp<mlir::arith::ConstantOp>();
    if (!C) return;
    int64_t V;
    if (auto IA = mlir::dyn_cast<mlir::IntegerAttr>(C.getValue()))
      V = IA.getInt();
    else if (auto FA = mlir::dyn_cast<mlir::FloatAttr>(C.getValue()))
      V = (int64_t)FA.getValueAsDouble();
    else return;
    States.insert(V);
  });
  return (unsigned)States.size();
}

void countOp(mlir::Operation &Op, FuncCounts &C) {
  using namespace mlir;
  unsigned W = 0;
  if (Op.getNumResults() == 1) W = widthOf(Op.getResult(0).getType());

  if (isa<arith::AddIOp>(Op))    { ++C.Adds; C.AddBits += W; return; }
  if (isa<arith::SubIOp>(Op))    { ++C.Subs; C.SubBits += W; return; }
  if (isa<arith::MulIOp>(Op))    { ++C.Muls; C.MulBits += W; return; }
  if (isa<arith::DivSIOp,
          arith::DivUIOp,
          arith::RemSIOp,
          arith::RemUIOp>(Op))   { ++C.Divs; return; }
  if (isa<arith::CmpIOp,
          arith::CmpFOp>(Op))    { ++C.Cmps; return; }
  if (isa<arith::ShLIOp,
          arith::ShRSIOp,
          arith::ShRUIOp>(Op))   { ++C.Shifts; return; }
  if (isa<arith::AndIOp,
          arith::OrIOp,
          arith::XOrIOp>(Op))    { ++C.BitOps; return; }
  if (isa<arith::SelectOp>(Op))  { ++C.Selects; return; }
  if (isa<arith::ConstantOp,
          LLVM::ConstantOp>(Op)) { ++C.Constants; return; }
  // unregistered matlab.* ops the SV emitter routes specially —
  // fold them into the matching counter for accuracy.
  llvm::StringRef N = Op.getName().getStringRef();
  if (N == "matlab.add") { ++C.Adds; C.AddBits += W; return; }
  if (N == "matlab.sub") { ++C.Subs; C.SubBits += W; return; }
  if (N == "matlab.matmul" || N == "matlab.emul")
                          { ++C.Muls; C.MulBits += W; return; }
  if (N == "matlab.eq" || N == "matlab.ne" || N == "matlab.lt" ||
      N == "matlab.le" || N == "matlab.gt" || N == "matlab.ge")
                          { ++C.Cmps; return; }
  // Other ops (alloca, gep, load, store, scf.if, ...) aren't
  // operator-cell counts — they're structural. Skip.
}

void emitFuncReport(std::ostream &OS, mlir::func::FuncOp F,
                    const FuncCounts &C) {
  // Classify.
  std::string Class = "combinational";
  if (!C.FSMs.empty()) Class = "FSM";
  else if (!C.Persists.empty()) Class = "clocked";

  OS << "## " << F.getSymName().str() << " — " << Class << "\n\n";

  if (C.HasClock || C.HasReset) {
    OS << "- **Clock domain:** clk + " << C.ResetKind << "\n";
  } else {
    OS << "- **Clock domain:** combinational\n";
  }

  // Ports.
  auto FT = F.getFunctionType();
  if (FT.getNumInputs() > 0) {
    OS << "- **Inputs:** ";
    for (unsigned i = 0; i < FT.getNumInputs(); ++i) {
      if (i) OS << ", ";
      OS << "arg" << i << " : " << formatType(FT.getInput(i));
    }
    OS << "\n";
  }
  if (FT.getNumResults() > 0) {
    OS << "- **Outputs:** ";
    for (unsigned i = 0; i < FT.getNumResults(); ++i) {
      if (i) OS << ", ";
      OS << "y" << (i ? std::to_string(i) : "")
         << " : " << formatType(FT.getResult(i));
    }
    OS << "\n";
  }

  // Operators.
  OS << "- **Operators:** "
     << C.Adds << " add"
     << " (" << C.AddBits << "b)"
     << ", " << C.Subs << " sub (" << C.SubBits << "b)"
     << ", " << C.Muls << " mul (" << C.MulBits << "b)"
     << ", " << C.Divs << " div"
     << ", " << C.Cmps << " cmp"
     << ", " << C.Shifts << " shift"
     << ", " << C.BitOps << " bitop"
     << ", " << C.Selects << " mux"
     << "\n";

  // Registers.
  if (!C.Persists.empty()) {
    unsigned TotalBits = 0;
    for (auto &P : C.Persists) TotalBits += P.Width;
    OS << "- **Registers:** " << C.Persists.size()
       << " (" << TotalBits << " flip-flops total)";
    OS << " —";
    bool First = true;
    for (auto &P : C.Persists) {
      OS << (First ? " " : ", ");
      First = false;
      OS << "`" << P.Name << "`:" << P.Width << "b";
    }
    OS << "\n";
  } else {
    OS << "- **Registers:** none\n";
  }

  // FSMs.
  if (!C.FSMs.empty()) {
    OS << "- **FSMs:**";
    bool First = true;
    for (auto &FI : C.FSMs) {
      OS << (First ? " " : ", ");
      First = false;
      OS << "`" << FI.RegName << "` — " << FI.States << " states ("
         << FI.Encoding << ")";
    }
    OS << "\n";
  } else {
    OS << "- **FSMs:** none\n";
  }

  // RAM placeholder — RAM inference is deferred (Phase 4 RAM /
  // Phase 4.5.4 follow-up). Always "none" today.
  OS << "- **RAM:** none\n";

  OS << "\n";
}

} // namespace

bool emitHardwareReport(mlir::ModuleOp M, std::ostream &OS,
                        const matlab::SourceManager *SM) {
  (void)SM;
  OS << "# Hardware Report\n\n";
  OS << "Pre-synthesis estimate. Operator and register counts come\n";
  OS << "from the post-pipeline IR; absolute gate counts and timing\n";
  OS << "must come from your downstream synthesis tool.\n\n";

  unsigned NumFuncs = 0;
  M.walk([&](mlir::func::FuncOp F) {
    if (F.empty()) return;
    if (isScriptFunc(F)) return;
    ++NumFuncs;

    FuncCounts C;
    // Persistents.
    (void)gatherHWPersistentState(F.getOperation(), C.Persists);
    if (!C.Persists.empty()) {
      C.HasClock = true;
      C.HasReset = true;
      C.ResetKind = "rst_n (async-low)";  // matches SV emitter default
    }
    // FSMs — for each persistent, count distinct state-eq case
    // constants. ≥2 = FSM.
    for (auto &P : C.Persists) {
      if (P.Gets.empty()) continue;
      unsigned States = countStatesForRegister(F, P.Gets[0]);
      if (States >= 2) {
        FuncCounts::FSMReport R;
        R.RegName = P.Name;
        R.States = States;
        // Default encoding (CLI flag isn't exposed to the report
        // emitter; report it as "binary or per-pragma").
        if (auto Attr = F->getAttrOfType<mlir::StringAttr>(
                "hdl.fsm_encoding")) {
          R.Encoding = Attr.getValue().str();
        } else {
          R.Encoding = "binary (CLI default)";
        }
        C.FSMs.push_back(std::move(R));
      }
    }

    // Operator counts — walk every op in the function body and
    // tally. Skip the SV-pipeline structural ops (alloca, gep,
    // load, store, scf.if, scf.yield, return) which aren't cells.
    F.walk([&](mlir::Operation *Op) { countOp(*Op, C); });

    emitFuncReport(OS, F, C);
  });

  if (NumFuncs == 0) {
    OS << "_No synthesizable functions found in this module._\n";
  }
  return true;
}

} // namespace mlirgen
} // namespace matlab
