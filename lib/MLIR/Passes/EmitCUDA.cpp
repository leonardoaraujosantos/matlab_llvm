// lib/MLIR/Passes/EmitCUDA.cpp — emit CUDA-C kernel source from
// `matlab.gpu.kernel` ops.  T3 of docs/gpu_coder_roadmap.md.
//
// Forked from EmitMetal.cpp; the body walker logic is identical (same
// MLIR shapes), the only differences are the kernel signature, the
// per-thread-id syntax, and the type names.
//
//   Metal           CUDA
//   --------------  --------------
//   kernel void     __global__ void
//   device T*       T*
//   constant T&     const T  (passed by value)
//   uint tid        int tid = blockIdx.x * blockDim.x + threadIdx.x
//   float / half    float / double / __half  (CUDA supports fp64!)
//
// CUDA supports `double` natively (Apple Metal doesn't), so this
// emitter uses `double` throughout to match MATLAB's default
// precision.  The user can write `single()` casts in their MATLAB if
// they want fp32 — that decision is upstream of this emitter.
//
// Cannot validate locally on macOS (no nvcc + no NVIDIA HW).  CI runs
// nvcc --compile on the emitted bundle when a Linux+NVIDIA runner is
// available (deferred — Linux CI lane).

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

struct CudaEmitCtx {
  std::stringstream Os;
  std::string KernelName;
  Value IV;
  std::string IVName = "iv";
  Value OutputSlot;
  std::string OutputName = "out";
  Value IVSlot;
  llvm::SmallVector<Value> ScalarCaptureOrder;
  llvm::DenseMap<Value, std::string> ScalarCaptureNames;
  /* Outer scalar slots WRITTEN inside the body — per-thread locals. */
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

void collectCaptures(Block &Body, CudaEmitCtx &Ctx) {
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

  /* A non-tensor scalar slot type: f64, or `none` for an unpromoted
   * function param (no type-promotion pass runs before -emit-cuda). */
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
    if (auto SA = Def->getAttrOfType<StringAttr>("name"))
      if (!VarName.empty() && SA.getValue() == VarName)
        Ctx.IVSlot = Slot;
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
                     ? LoadOp->getOperand(0)
                     : LoadedTensor;
    if (InsideDefs.count(Slot)) return;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc")) return;
    if (!mlir::isa<RankedTensorType, UnrankedTensorType>(Slot.getType())) return;
    if (Ctx.OutputSlot && Ctx.OutputSlot != Slot) {
      Ctx.bail("multiple output tensors not supported");
      return;
    }
    Ctx.OutputSlot = Slot;
  });

  /* Outer scalar slots WRITTEN inside the body (not the IV) — per-thread
   * locals, collected first so a read+written slot is a local. */
  llvm::DenseSet<Value> StoredSet;
  KernelOp->walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.store") || Op->getNumOperands() != 2) return;
    Value Slot = Op->getOperand(1);
    if (InsideDefs.count(Slot) || Slot == Ctx.IVSlot) return;
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
    if (Slot == Ctx.IVSlot || Slot == Ctx.OutputSlot) return;
    if (StoredSet.count(Slot)) return;  /* it's a local */
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

std::string exprFor(Value V, CudaEmitCtx &Ctx) {
  if (V == Ctx.IV) return Ctx.IVName;
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

void emitBody(Block &Body, CudaEmitCtx &Ctx) {
  auto F64 = Float64Type::get(Body.getParentOp()->getContext());
  for (Operation &Op : Body) {
    if (Ctx.Bailed) return;
    StringRef Name = Op.getName().getStringRef();
    if (Name == "matlab.yield") continue;
    if (auto C = dyn_cast<arith::ConstantOp>(&Op)) {
      if (C.getResult().getType() == F64) {
        if (auto Fa = mlir::dyn_cast<FloatAttr>(C.getValue())) {
          std::stringstream Ss;
          Ss << Fa.getValueAsDouble();
          Ctx.ValueExpr[C.getResult()] = Ss.str();
          continue;
        }
      }
      if (auto Ia = mlir::dyn_cast<IntegerAttr>(C.getValue())) {
        std::stringstream Ss;
        Ss << Ia.getInt() << ".0";
        Ctx.ValueExpr[C.getResult()] = Ss.str();
        continue;
      }
      Ctx.bail("unsupported constant shape");
      return;
    }
    if (Name == "matlab.const_int") {
      if (auto Va = Op.getAttrOfType<IntegerAttr>("value")) {
        std::stringstream Ss;
        Ss << Va.getInt() << ".0";
        Ctx.ValueExpr[Op.getResult(0)] = Ss.str();
        continue;
      }
      Ctx.bail("matlab.const_int missing value");
      return;
    }
    if (Name == "matlab.const_float") {
      if (auto Va = Op.getAttrOfType<FloatAttr>("value")) {
        std::stringstream Ss;
        Ss << Va.getValueAsDouble();
        Ctx.ValueExpr[Op.getResult(0)] = Ss.str();
        continue;
      }
      Ctx.bail("matlab.const_float missing value");
      return;
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
      else Ctx.ValueExpr[Op.getResult(0)] = exprFor(Slot, Ctx);
      continue;
    }
    if (Name == "matlab.store" && Op.getNumOperands() == 2) {
      Value Slot = Op.getOperand(1);
      if (Slot == Ctx.IVSlot) continue;
      if (Slot == Ctx.OutputSlot) continue;
      std::string SlotE = exprFor(Slot, Ctx);
      std::string ValE = exprFor(Op.getOperand(0), Ctx);
      if (SlotE == Ctx.IVName) continue;
      Ctx.Os << "  " << SlotE << " = " << ValE << ";\n";
      continue;
    }
    if ((Name == "matlab.add" || Name == "matlab.sub" ||
         Name == "matlab.matmul" || Name == "matlab.mul" ||
         Name == "matlab.div" || Name == "matlab.matdiv") &&
        Op.getNumOperands() == 2) {
      /* Inside a per-thread kernel body every value is scalar, so a
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
      if (Cal && Cal.getValue() == "__subscript_load" &&
          Op.getNumOperands() >= 2) {
        Ctx.ValueExpr[Op.getResult(0)] =
            exprFor(Op.getOperand(0), Ctx) + "[(int)(" +
            exprFor(Op.getOperand(1), Ctx) + ") - 1]";
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

std::string emitCudaKernels(mlir::ModuleOp M, llvm::StringRef Prefix,
                            GpuKernelInfo *Info) {
  std::stringstream OS;
  OS << "/* Generated by matlabc -emit-cuda.\n"
     << " * Body translated op-by-op from matlab.gpu.kernel.\n"
     << " * Unsupported shapes inline a FALLBACK comment + identity body.\n"
     << " *\n"
     << " * No #include: __global__ / blockIdx / threadIdx are device\n"
     << " * builtins for both nvcc and NVRTC, so this source is\n"
     << " * NVRTC-compilable as-is (the host driver JIT-compiles it).\n"
     << " */\n\n";
  unsigned KId = 0;
  M.walk([&](mlir::Operation *Op) {
    if (Op->getName().getStringRef() != "matlab.gpu.kernel") return;
    if (Op->getNumRegions() != 1) return;
    auto &Body = Op->getRegion(0);
    if (!Body.hasOneBlock()) return;
    auto &BB = Body.front();
    if (BB.getNumArguments() != 1) return;

    CudaEmitCtx Ctx;
    Ctx.IV = BB.getArgument(0);
    Ctx.KernelName = std::string(Prefix) + "_kernel_" + std::to_string(KId++);

    collectCaptures(BB, Ctx);

    OS << "extern \"C\" __global__ void " << Ctx.KernelName << "(\n";
    if (Ctx.OutputSlot) OS << "    double *" << Ctx.OutputName << ",\n";
    for (auto V : Ctx.ScalarCaptureOrder)
      OS << "    const double " << Ctx.ScalarCaptureNames[V] << ",\n";
    OS << "    int n_grid)\n{\n";
    OS << "  int tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    OS << "  if (tid >= n_grid) return;\n";
    OS << "  double " << Ctx.IVName << " = (double)tid + 1.0;\n";
    for (Value V : Ctx.OuterLocalOrder)
      OS << "  double " << Ctx.OuterLocalNames[V] << " = 0.0;\n";

    std::stringstream BodyOS;
    Ctx.Os.swap(BodyOS);
    emitBody(BB, Ctx);
    Ctx.Os.swap(BodyOS);
    if (Ctx.Bailed) {
      OS << "  // FALLBACK: " << Ctx.BailReason << ".  Identity body.\n";
    } else {
      OS << BodyOS.str();
    }
    OS << "}\n\n";

    /* Record the first kernel's shape for the host-driver emitter. */
    if (Info && Info->kernelCount == 0) {
      Info->name = Ctx.KernelName;
      Info->hasOutput = (Ctx.OutputSlot != nullptr);
      Info->bailed = Ctx.Bailed;
      for (auto V : Ctx.ScalarCaptureOrder)
        Info->scalarArgs.push_back(Ctx.ScalarCaptureNames[V]);
    }
    if (Info) Info->kernelCount++;
  });
  if (KId == 0)
    OS << "/* No matlab.gpu.kernel ops found. */\n";
  return OS.str();
}

}  // namespace mlirgen
}  // namespace matlab
