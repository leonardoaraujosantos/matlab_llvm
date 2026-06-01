// lib/MLIR/Passes/EmitOpenCL.cpp — emit OpenCL-C kernel source from
// `matlab.gpu.kernel` ops.  T4 of docs/gpu_coder_roadmap.md.
//
// Forked from EmitCUDA.cpp; same body walker.  Differences:
//   CUDA              OpenCL
//   __global__ void   __kernel void
//   T*                __global T*
//   const T (by val)  const T
//   blockIdx*…+thr…   get_global_id(0)
//   double            double (requires cl_khr_fp64 pragma)
//
// fp64 in OpenCL requires the `cl_khr_fp64` extension.  Most modern
// AMD/Intel/NVIDIA OpenCL stacks support it; older Mali doesn't.
// The emitted source enables it via #pragma — if the device rejects
// the pragma, the build fails cleanly and the user can re-emit with
// `single()` casts in their MATLAB source.

#include "matlab/MLIR/Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

struct OCLEmitCtx {
  std::stringstream Os;
  std::string KernelName;
  Value IV;
  std::string IVName = "iv";
  Value OutputSlot;
  std::string OutputName = "out";
  Value IVSlot;
  /* 2-D kernels (a for-i × for-j nest flattened to a 2-D NDRange).
   * Primary IV = inner j (← dim 0); IV2 = outer i (← dim 1).  NRowsExpr
   * is the exact NDRange extent in dim 1 (OpenCL global sizes are not
   * padded). */
  bool TwoD = false;
  Value IV2;
  std::string IV2Name = "i_iv";
  Value IVSlot2;
  std::string OuterVarName;
  std::string NRowsExpr = "(int)get_global_size(1)";
  llvm::SmallVector<Value> ScalarCaptureOrder;
  llvm::DenseMap<Value, std::string> ScalarCaptureNames;
  /* Outer scalar slots WRITTEN inside the body — per-work-item locals. */
  llvm::SmallVector<Value> OuterLocalOrder;
  llvm::DenseMap<Value, std::string> OuterLocalNames;
  llvm::DenseMap<Value, std::string> LocalNames;
  unsigned LocalCounter = 0;
  llvm::DenseMap<Value, std::string> ValueExpr;
  unsigned TempCounter = 0;
  bool Bailed = false;
  std::string BailReason;
  void bail(StringRef R) {
    Bailed = true;
    if (BailReason.empty()) BailReason = std::string(R);
  }
};

std::string allocName(Operation *Alloc, unsigned Fallback) {
  if (!Alloc) return "slot_" + std::to_string(Fallback);
  if (auto SA = Alloc->getAttrOfType<StringAttr>("name"))
    return std::string(SA.getValue());
  if (auto FA = Alloc->getAttrOfType<FlatSymbolRefAttr>("name"))
    return std::string(FA.getValue());
  return "slot_" + std::to_string(Fallback);
}

void collectCaptures(Block &Body, OCLEmitCtx &Ctx) {
  Operation *KernelOp = Body.getParentOp();
  llvm::DenseSet<Value> InsideDefs;
  InsideDefs.insert(Ctx.IV);
  KernelOp->walk([&](Operation *Op) {
    for (Value R : Op->getResults()) InsideDefs.insert(R);
    for (Region &Rg : Op->getRegions())
      for (Block &Bl : Rg)
        for (Value A : Bl.getArguments()) InsideDefs.insert(A);
  });

  StringRef VarName;
  if (auto VA = KernelOp->getAttrOfType<StringAttr>("var"))
    VarName = VA.getValue();

  auto isScalarSlotTy = [&](Type T) {
    if (mlir::isa<RankedTensorType, UnrankedTensorType>(T)) return false;
    return T == Float64Type::get(KernelOp->getContext()) ||
           mlir::isa<NoneType>(T);
  };

  KernelOp->walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.load") || Op->getNumOperands() != 1) return;
    Value Slot = Op->getOperand(0);
    if (InsideDefs.count(Slot)) return;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc")) return;
    auto SA = Def->getAttrOfType<StringAttr>("name");
    if (!SA) return;
    if (!VarName.empty() && SA.getValue() == VarName) Ctx.IVSlot = Slot;
    if (Ctx.TwoD && !Ctx.OuterVarName.empty() &&
        SA.getValue() == Ctx.OuterVarName)
      Ctx.IVSlot2 = Slot;
  });
  KernelOp->walk([&](Operation *Op) {
    if (Ctx.Bailed) return;
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto Cal = Op->getAttrOfType<StringAttr>("callee");
    if (!Cal || Cal.getValue() != "__subscript_store") return;
    if (Op->getNumOperands() < 3) return;
    Value LoadedTensor = Op->getOperand(0);
    Operation *LoadOp = LoadedTensor.getDefiningOp();
    Value Slot = (LoadOp && isMatlabOp(LoadOp, "matlab.load") &&
                  LoadOp->getNumOperands() == 1)
                     ? LoadOp->getOperand(0) : LoadedTensor;
    if (InsideDefs.count(Slot)) return;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc")) return;
    if (!mlir::isa<RankedTensorType, UnrankedTensorType>(Slot.getType())) return;
    if (Ctx.OutputSlot && Ctx.OutputSlot != Slot) {
      Ctx.bail("multiple output tensors not supported"); return;
    }
    Ctx.OutputSlot = Slot;
  });
  /* Outer scalar slots WRITTEN inside the body — per-thread locals. */
  llvm::DenseSet<Value> StoredSet;
  KernelOp->walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.store") || Op->getNumOperands() != 2) return;
    Value Slot = Op->getOperand(1);
    if (InsideDefs.count(Slot) || Slot == Ctx.IVSlot || Slot == Ctx.IVSlot2)
      return;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc") || !isScalarSlotTy(Slot.getType()))
      return;
    if (StoredSet.insert(Slot).second) {
      Ctx.OuterLocalOrder.push_back(Slot);
      Ctx.OuterLocalNames[Slot] =
          "loc_" + allocName(Def, static_cast<unsigned>(StoredSet.size()));
    }
  });
  llvm::DenseSet<Value> ScalarSet;
  KernelOp->walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.load") || Op->getNumOperands() != 1) return;
    Value Slot = Op->getOperand(0);
    if (InsideDefs.count(Slot)) return;
    if (Slot == Ctx.IVSlot || Slot == Ctx.IVSlot2 || Slot == Ctx.OutputSlot)
      return;
    if (StoredSet.count(Slot)) return;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc") || !isScalarSlotTy(Slot.getType()))
      return;
    if (ScalarSet.insert(Slot).second) {
      Ctx.ScalarCaptureOrder.push_back(Slot);
      Ctx.ScalarCaptureNames[Slot] =
          allocName(Def, static_cast<unsigned>(ScalarSet.size()));
    }
  });
}

std::string exprFor(Value V, OCLEmitCtx &Ctx) {
  if (V == Ctx.IV) return Ctx.IVName;
  if (Ctx.TwoD && V == Ctx.IV2) return Ctx.IV2Name;
  if (V == Ctx.OutputSlot) return Ctx.OutputName;
  auto OIt = Ctx.OuterLocalNames.find(V);
  if (OIt != Ctx.OuterLocalNames.end()) return OIt->second;
  auto SIt = Ctx.ScalarCaptureNames.find(V);
  if (SIt != Ctx.ScalarCaptureNames.end()) return SIt->second;
  auto VIt = Ctx.ValueExpr.find(V);
  if (VIt != Ctx.ValueExpr.end()) return VIt->second;
  auto LIt = Ctx.LocalNames.find(V);
  if (LIt != Ctx.LocalNames.end()) return LIt->second;
  return "/* UNRESOLVED */ 0.0";
}

void emitBody(Block &Body, OCLEmitCtx &Ctx) {
  auto F64 = Float64Type::get(Body.getParentOp()->getContext());
  for (Operation &Op : Body) {
    if (Ctx.Bailed) return;
    StringRef Name = Op.getName().getStringRef();
    if (Name == "matlab.yield") continue;
    if (auto C = dyn_cast<arith::ConstantOp>(&Op)) {
      if (C.getResult().getType() == F64) {
        if (auto Fa = mlir::dyn_cast<FloatAttr>(C.getValue())) {
          std::stringstream Ss; Ss << Fa.getValueAsDouble();
          Ctx.ValueExpr[C.getResult()] = Ss.str(); continue;
        }
      }
      if (auto Ia = mlir::dyn_cast<IntegerAttr>(C.getValue())) {
        std::stringstream Ss; Ss << Ia.getInt() << ".0";
        Ctx.ValueExpr[C.getResult()] = Ss.str(); continue;
      }
      Ctx.bail("unsupported constant"); return;
    }
    if (Name == "matlab.const_int") {
      if (auto Va = Op.getAttrOfType<IntegerAttr>("value")) {
        std::stringstream Ss; Ss << Va.getInt() << ".0";
        Ctx.ValueExpr[Op.getResult(0)] = Ss.str(); continue;
      }
      Ctx.bail("matlab.const_int missing value"); return;
    }
    if (Name == "matlab.const_float") {
      if (auto Va = Op.getAttrOfType<FloatAttr>("value")) {
        std::stringstream Ss; Ss << Va.getValueAsDouble();
        Ctx.ValueExpr[Op.getResult(0)] = Ss.str(); continue;
      }
      Ctx.bail("matlab.const_float missing value"); return;
    }
    if (Name == "matlab.alloc") {
      Value R = Op.getResult(0);
      if (R.getType() != F64) { Ctx.bail("non-f64 local slot"); return; }
      std::string LName = "loc_" + allocName(&Op, Ctx.LocalCounter++);
      Ctx.LocalNames[R] = LName;
      Ctx.Os << "  double " << LName << " = 0.0;\n";
      continue;
    }
    if (Name == "matlab.load" && Op.getNumOperands() == 1) {
      Value Slot = Op.getOperand(0);
      if (Slot == Ctx.IVSlot) Ctx.ValueExpr[Op.getResult(0)] = Ctx.IVName;
      else if (Ctx.TwoD && Slot == Ctx.IVSlot2)
        Ctx.ValueExpr[Op.getResult(0)] = Ctx.IV2Name;
      else Ctx.ValueExpr[Op.getResult(0)] = exprFor(Slot, Ctx);
      continue;
    }
    if (Name == "matlab.store" && Op.getNumOperands() == 2) {
      Value Slot = Op.getOperand(1);
      if (Slot == Ctx.IVSlot) continue;
      if (Ctx.TwoD && Slot == Ctx.IVSlot2) continue;
      if (Slot == Ctx.OutputSlot) continue;
      std::string SlotE = exprFor(Slot, Ctx);
      if (SlotE == Ctx.IVName) continue;
      Ctx.Os << "  " << SlotE << " = " << exprFor(Op.getOperand(0), Ctx)
             << ";\n";
      continue;
    }
    if ((Name == "matlab.add" || Name == "matlab.sub" ||
         Name == "matlab.matmul" || Name == "matlab.mul" ||
         Name == "matlab.div" || Name == "matlab.matdiv") &&
        Op.getNumOperands() == 2) {
      /* Inside a per-work-item kernel body every value is scalar, so a
       * MATLAB `*` (matlab.matmul / matlab.mul) is a scalar multiply and
       * `/` (matlab.matdiv / matlab.div) a scalar divide.  Don't require
       * the result to be typed f64 — unpromoted params make the result
       * `none` (see collectCaptures).  Only bail if an operand is
       * genuinely a tensor (an unsupported in-kernel matmul). */
      if (mlir::isa<RankedTensorType, UnrankedTensorType>(
              Op.getOperand(0).getType()) ||
          mlir::isa<RankedTensorType, UnrankedTensorType>(
              Op.getOperand(1).getType())) {
        Ctx.bail("in-kernel tensor binop not supported");
        return;
      }
      const char *Sym = Name == "matlab.add" ? "+"
                       : Name == "matlab.sub" ? "-"
                       : (Name == "matlab.div" || Name == "matlab.matdiv")
                             ? "/"
                             : "*";
      Ctx.ValueExpr[Op.getResult(0)] = "(" + exprFor(Op.getOperand(0), Ctx) +
                                       " " + Sym + " " +
                                       exprFor(Op.getOperand(1), Ctx) + ")";
      continue;
    }
    if (Name == "matlab.neg" && Op.getNumOperands() == 1) {
      Ctx.ValueExpr[Op.getResult(0)] =
          "(-" + exprFor(Op.getOperand(0), Ctx) + ")";
      continue;
    }
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(&Op)) {
      const char *Sym = isa<arith::AddFOp>(&Op) ? "+"
                     : isa<arith::SubFOp>(&Op) ? "-"
                     : isa<arith::MulFOp>(&Op) ? "*" : "/";
      Ctx.ValueExpr[Op.getResult(0)] = "(" + exprFor(Op.getOperand(0), Ctx) +
                                       " " + Sym + " " +
                                       exprFor(Op.getOperand(1), Ctx) + ")";
      continue;
    }

    /* Scalar relational ops (result i1): matlab.{lt,le,gt,ge,eq,ne}. */
    {
      const char *Rel = Name == "matlab.lt"   ? "<"
                      : Name == "matlab.le"   ? "<="
                      : Name == "matlab.gt"   ? ">"
                      : Name == "matlab.ge"   ? ">="
                      : Name == "matlab.eq"   ? "=="
                      : Name == "matlab.ne"   ? "!=" : nullptr;
      if (Rel && Op.getNumOperands() == 2) {
        Ctx.ValueExpr[Op.getResult(0)] =
            "(" + exprFor(Op.getOperand(0), Ctx) + " " + Rel + " " +
            exprFor(Op.getOperand(1), Ctx) + ")";
        continue;
      }
    }
    /* Logical ops (result i1). */
    {
      const char *Lg = (Name == "matlab.short_and" || Name == "matlab.and")
                           ? "&&"
                       : (Name == "matlab.short_or" || Name == "matlab.or")
                           ? "||" : nullptr;
      if (Lg && Op.getNumOperands() == 2) {
        Ctx.ValueExpr[Op.getResult(0)] =
            "(" + exprFor(Op.getOperand(0), Ctx) + " " + Lg + " " +
            exprFor(Op.getOperand(1), Ctx) + ")";
        continue;
      }
      if ((Name == "matlab.not" || Name == "matlab.short_not") &&
          Op.getNumOperands() == 1) {
        Ctx.ValueExpr[Op.getResult(0)] =
            "(!" + exprFor(Op.getOperand(0), Ctx) + ")";
        continue;
      }
    }

    /* matlab.while — see the EmitMetal emitter for the rationale. */
    if (Name == "matlab.while" && Op.getNumRegions() == 2 &&
        Op.getRegion(0).hasOneBlock() && Op.getRegion(1).hasOneBlock()) {
      Block &Cond = Op.getRegion(0).front();
      Block &Loop = Op.getRegion(1).front();
      emitBody(Cond, Ctx);
      if (Ctx.Bailed) return;
      Operation *Term = Cond.getTerminator();
      if (!Term || Term->getNumOperands() != 1) {
        Ctx.bail("matlab.while condition without a yielded value");
        return;
      }
      Ctx.Os << "  while (" << exprFor(Term->getOperand(0), Ctx) << ") {\n";
      emitBody(Loop, Ctx);
      Ctx.Os << "  }\n";
      continue;
    }

    if (Name == "matlab.call_builtin") {
      auto Cal = Op.getAttrOfType<StringAttr>("callee");
      if (Cal && Cal.getValue() == "__subscript_store" &&
          Op.getNumOperands() == 3) {
        Ctx.Os << "  " << exprFor(Op.getOperand(0), Ctx)
               << "[(int)(" << exprFor(Op.getOperand(1), Ctx) << ") - 1] = "
               << exprFor(Op.getOperand(2), Ctx) << ";\n";
        continue;
      }
      if (Cal && Cal.getValue() == "__subscript_store" &&
          Op.getNumOperands() == 4 && Ctx.TwoD) {
        Ctx.Os << "  " << exprFor(Op.getOperand(0), Ctx) << "[((int)("
               << exprFor(Op.getOperand(1), Ctx) << ") - 1) + ((int)("
               << exprFor(Op.getOperand(2), Ctx) << ") - 1) * " << Ctx.NRowsExpr
               << "] = " << exprFor(Op.getOperand(3), Ctx) << ";\n";
        continue;
      }
      if (Cal && Cal.getValue() == "__subscript_load" &&
          Op.getNumOperands() >= 2) {
        Ctx.ValueExpr[Op.getResult(0)] = exprFor(Op.getOperand(0), Ctx) +
            "[(int)(" + exprFor(Op.getOperand(1), Ctx) + ") - 1]";
        continue;
      }
      Ctx.bail((std::string("unsupported builtin ") +
                std::string(Cal ? Cal.getValue() : "")).c_str());
      return;
    }
    if (Name == "matlab.range" || Name == "matlab.const_char") continue;
    Ctx.bail((std::string("unsupported op ") + std::string(Name)).c_str());
    return;
  }
}

}  // namespace

std::string emitOpenCLKernels(mlir::ModuleOp M, llvm::StringRef Prefix,
                              GpuKernelInfo *Info) {
  std::stringstream OS;
  OS << "/* Generated by matlabc -emit-opencl.\n"
     << " * Body translated op-by-op from matlab.gpu.kernel.\n"
     << " * Unsupported shapes fall back to identity body.\n"
     << " */\n"
     << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";
  unsigned KId = 0;
  M.walk([&](mlir::Operation *Op) {
    if (Op->getName().getStringRef() != "matlab.gpu.kernel") return;
    for (Operation *Anc = Op->getParentOp(); Anc; Anc = Anc->getParentOp())
      if (isMatlabOp(Anc, "matlab.gpu.kernel")) return;  /* inner — skip */
    if (Op->getNumRegions() != 1) return;
    auto &Body = Op->getRegion(0);
    if (!Body.hasOneBlock()) return;
    auto &BB = Body.front();
    if (BB.getNumArguments() != 1) return;

    OCLEmitCtx Ctx;
    Ctx.KernelName = std::string(Prefix) + "_kernel_" + std::to_string(KId++);

    /* Flatten a for-i × for-j nest to a 2-D NDRange (see EmitMetal). */
    Operation *Inner = nullptr;
    for (Operation &O : BB)
      if (isMatlabOp(&O, "matlab.gpu.kernel")) { Inner = &O; break; }
    Block *ComputeBB = &BB;
    if (Inner && Inner->getNumRegions() == 1 &&
        Inner->getRegion(0).hasOneBlock() &&
        Inner->getRegion(0).front().getNumArguments() == 1) {
      ComputeBB = &Inner->getRegion(0).front();
      Ctx.TwoD = true;
      Ctx.IV = ComputeBB->getArgument(0);   /* inner j */
      Ctx.IVName = "j_iv";
      Ctx.IV2 = BB.getArgument(0);          /* outer i */
      Ctx.IV2Name = "i_iv";
      if (auto VA = Op->getAttrOfType<StringAttr>("var"))
        Ctx.OuterVarName = std::string(VA.getValue());
    } else {
      Ctx.IV = BB.getArgument(0);
    }

    collectCaptures(*ComputeBB, Ctx);

    OS << "__kernel void " << Ctx.KernelName << "(\n";
    if (Ctx.OutputSlot)
      OS << "    __global double *" << Ctx.OutputName << ",\n";
    for (auto V : Ctx.ScalarCaptureOrder)
      OS << "    const double " << Ctx.ScalarCaptureNames[V] << ",\n";
    if (Ctx.TwoD) {
      OS << "    const int n_grid)\n{\n"
         << "  int jx = get_global_id(0);\n"
         << "  int iy = get_global_id(1);\n"
         << "  double " << Ctx.IVName << " = (double)jx + 1.0;\n"
         << "  double " << Ctx.IV2Name << " = (double)iy + 1.0;\n";
    } else {
      OS << "    const int n_grid)\n{\n"
         << "  int tid = get_global_id(0);\n"
         << "  if (tid >= n_grid) return;\n"
         << "  double " << Ctx.IVName << " = (double)tid + 1.0;\n";
    }
    for (Value V : Ctx.OuterLocalOrder)
      OS << "  double " << Ctx.OuterLocalNames[V] << " = 0.0;\n";
    std::stringstream BodyOS;
    Ctx.Os.swap(BodyOS);
    emitBody(*ComputeBB, Ctx);
    Ctx.Os.swap(BodyOS);
    if (Ctx.Bailed)
      OS << "  // FALLBACK: " << Ctx.BailReason << ".  Identity body.\n";
    else
      OS << BodyOS.str();
    OS << "}\n\n";

    /* Record the first kernel's shape for the host-driver emitter. */
    if (Info && Info->kernelCount == 0) {
      Info->name = Ctx.KernelName;
      Info->hasOutput = (Ctx.OutputSlot != nullptr);
      Info->bailed = Ctx.Bailed;
      Info->twoD = Ctx.TwoD;
      for (auto V : Ctx.ScalarCaptureOrder)
        Info->scalarArgs.push_back(Ctx.ScalarCaptureNames[V]);
    }
    if (Info) Info->kernelCount++;
  });
  if (KId == 0) OS << "/* No matlab.gpu.kernel ops found. */\n";
  return OS.str();
}

}  // namespace mlirgen
}  // namespace matlab
