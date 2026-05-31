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
  llvm::DenseSet<Value> InsideDefs;
  InsideDefs.insert(Ctx.IV);
  for (Operation &Op : Body)
    for (Value R : Op.getResults()) InsideDefs.insert(R);

  StringRef VarName;
  if (auto VA = Body.getParentOp()->getAttrOfType<StringAttr>("var"))
    VarName = VA.getValue();

  for (Operation &Op : Body) {
    if (!isMatlabOp(&Op, "matlab.load") || Op.getNumOperands() != 1) continue;
    Value Slot = Op.getOperand(0);
    if (InsideDefs.count(Slot)) continue;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc")) continue;
    if (auto SA = Def->getAttrOfType<StringAttr>("name"))
      if (!VarName.empty() && SA.getValue() == VarName)
        Ctx.IVSlot = Slot;
  }

  for (Operation &Op : Body) {
    if (!isMatlabOp(&Op, "matlab.call_builtin")) continue;
    auto Cal = Op.getAttrOfType<StringAttr>("callee");
    if (!Cal || Cal.getValue() != "__subscript_store") continue;
    if (Op.getNumOperands() < 3) continue;
    Value LoadedTensor = Op.getOperand(0);
    Operation *LoadOp = LoadedTensor.getDefiningOp();
    Value Slot;
    if (LoadOp && isMatlabOp(LoadOp, "matlab.load") &&
        LoadOp->getNumOperands() == 1)
      Slot = LoadOp->getOperand(0);
    else
      Slot = LoadedTensor;
    if (InsideDefs.count(Slot)) continue;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc")) continue;
    if (!mlir::isa<RankedTensorType, UnrankedTensorType>(Slot.getType())) continue;
    if (Ctx.OutputSlot && Ctx.OutputSlot != Slot) {
      Ctx.bail("multiple output tensors not supported");
      return;
    }
    Ctx.OutputSlot = Slot;
  }

  llvm::DenseSet<Value> ScalarSet;
  for (Operation &Op : Body) {
    if (!isMatlabOp(&Op, "matlab.load") || Op.getNumOperands() != 1) continue;
    Value Slot = Op.getOperand(0);
    if (InsideDefs.count(Slot)) continue;
    if (Slot == Ctx.IVSlot) continue;
    if (Slot == Ctx.OutputSlot) continue;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc")) continue;
    /* Scalar capture = any non-tensor alloc loaded inside the kernel
     * that isn't the IV or output slot.  Accept f64 *and* `none`: a
     * function param like `x` reaches this emit path with an
     * unpromoted `none` slot type (no type-promotion pass runs before
     * -emit-cuda), but inside the per-thread kernel body it's a scalar.
     * Treated as `const double` in the kernel signature. */
    Type ST = Slot.getType();
    if (mlir::isa<RankedTensorType, UnrankedTensorType>(ST)) continue;
    if (!(ST == Float64Type::get(Op.getContext()) || mlir::isa<NoneType>(ST)))
      continue;
    if (ScalarSet.insert(Slot).second) {
      Ctx.ScalarCaptureOrder.push_back(Slot);
      Ctx.ScalarCaptureNames[Slot] =
          allocName(Def, static_cast<unsigned>(ScalarSet.size()));
    }
  }
}

std::string exprFor(Value V, CudaEmitCtx &Ctx) {
  if (V == Ctx.IV) return Ctx.IVName;
  if (V == Ctx.OutputSlot) return Ctx.OutputName;
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
         Name == "matlab.matmul" || Name == "matlab.mul") &&
        Op.getNumOperands() == 2) {
      /* Inside a per-thread kernel body every value is scalar, so a
       * MATLAB `*` (matlab.matmul / matlab.mul) is a scalar multiply.
       * Don't require the result to be typed f64 — unpromoted params
       * make the result `none` (see collectCaptures).  Only bail if an
       * operand is genuinely a tensor (an unsupported in-kernel matmul). */
      if (mlir::isa<RankedTensorType, UnrankedTensorType>(
              Op.getOperand(0).getType()) ||
          mlir::isa<RankedTensorType, UnrankedTensorType>(
              Op.getOperand(1).getType())) {
        Ctx.bail("in-kernel tensor binop not supported");
        return;
      }
      const char *Sym = Name == "matlab.add" ? "+"
                       : Name == "matlab.sub" ? "-" : "*";
      Ctx.ValueExpr[Op.getResult(0)] = "(" + exprFor(Op.getOperand(0), Ctx) +
                                       " " + Sym + " " +
                                       exprFor(Op.getOperand(1), Ctx) + ")";
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
                            CudaKernelInfo *Info) {
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
