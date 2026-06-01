// lib/MLIR/Passes/EmitMetal.cpp — emit MSL (Metal Shading Language)
// kernel source from `matlab.gpu.kernel` ops.
//
// T2.B of docs/gpu_coder_roadmap.md.  Walks each `matlab.gpu.kernel`
// op in the module and prints an MSL `kernel void <name>(...)` whose
// body is the per-iteration MATLAB body translated op-by-op.  The
// resulting source feeds the `-emit-metal` AOT bundle (Tier-6) and the
// future `MTLLibrary newLibraryWithSource:` JIT (T2.C).
//
// **Pass position**: must run BEFORE `runOutlineGpuKernels` rewrites
// `matlab.gpu.kernel` → `matlab.for` and BEFORE `LowerSeqLoops` /
// `LowerTensorOps` mangle the body.  The pass walks read-only — it
// extracts source and does NOT mutate the IR — so it can co-exist
// with the simple-rewrite CPU lane (Mandelbrot still ships through
// the rewrite path; this pass just also produces an MSL file).
//
// **Supported shapes (v1)**:
//   - 1-D induction range (matlab.range with f64 start/step/end).
//   - One tensor-typed matlab.alloc output written via
//     matlab.call_builtin("__subscript_store") inside the body.
//   - Scalar captures via matlab.alloc f64 + matlab.load (read-only
//     inside the body — bucket-2 outer-set scalars become MSL
//     `constant float &` kernel args).
//   - Body ops: arith.constant, arith.{add,sub,mul,div}f,
//     matlab.{add,sub,matmul} (scalar f64), matlab.load on scalar
//     local slots, matlab.store on scalar local slots,
//     matlab.call_builtin "__subscript_load" / "__subscript_store"
//     on the output tensor.
//
// **Unsupported v1 shapes** (the emitter detects + bails with a
// diagnostic placeholder body; the bundle still builds via the
// identity kernel template):
//   - Nested loops in the body (Mandelbrot's i+j+while pattern).
//   - if/else / while / break / continue.
//   - Multiple outputs.
//   - Non-f64 scalar types (i1, i32, half — half codegen is T2.D).
//
// **Test gate**: `examples/gpu/test_gpuarray_axpy.m`-shaped patterns
// emit a real MSL kernel body; Mandelbrot still falls back to the
// identity placeholder (Tier-2 follow-up: extend the translator to
// handle nested loops + while).
//
// Output format: a single std::string per kernel containing the
// kernel function source.  The caller (matlabc `-emit-metal`) writes
// this to `<stem>_kernel.metal`.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

/* Per-kernel emit context — collects captures, names, generated SSA
 * names, then walks the body op-by-op to print MSL. */
struct MetalEmitCtx {
  std::stringstream Os;
  std::string KernelName;
  Value IV;                                /* induction f64 block arg */
  std::string IVName = "iv";

  /* 2-D kernels (a `coder.gpu.kernelfun` for-i × for-j nest flattened to
   * a 2-D thread grid): the inner loop's induction var.  TwoD gates the
   * uint2 thread-id signature, the second IV decl, and the 2-D
   * subscript-store leading-dim index.  NRowsExpr is the column-major
   * leading dimension (`(int)gsz.y` — the grid's i-extent). */
  bool TwoD = false;
  Value IV2;
  std::string IV2Name = "j_iv";
  Value IVSlot2;
  std::string OuterVarName;                /* outer (i) loop var, 2-D only */
  std::string NRowsExpr = "(int)gsz.y";

  /* Output tensor SLOT — the outer matlab.alloc Value (matched via
   * the matlab.load that feeds __subscript_store).  At most one v1. */
  Value OutputSlot;
  std::string OutputName = "out";

  /* IV slot — the outer matlab.alloc whose name matches the kernel
   * op's `var` attribute.  Loads on this slot resolve to the IV
   * block argument; stores into it are no-ops. */
  Value IVSlot;

  /* Scalar captures (outer-set f64 slots read inside the body via
   * matlab.load).  Each entry maps the outer matlab.alloc slot Value
   * to its MSL kernel-arg name (e.g. `n_scalar`, `re_min`, …).
   *
   * We DON'T emit MSL device buffers for these — they're constant
   * memory `constant float &`.  Their names come from the
   * `name = "..."` attribute on the matlab.alloc op when present. */
  llvm::SmallVector<Value> ScalarCaptureOrder;
  llvm::DenseMap<Value, std::string> ScalarCaptureNames;

  /* Outer f64 slots that are WRITTEN inside the body (kernel-local
   * temporaries hoisted to the enclosing func's entry by the frontend,
   * e.g. Mandelbrot's `cr`/`zr`/`k`).  They are NOT read-only captures —
   * each becomes a per-thread MSL local declared at the top of the body.
   * Their outer value is irrelevant (overwritten before read), so they
   * are initialised to 0. */
  llvm::SmallVector<Value> OuterLocalOrder;
  llvm::DenseMap<Value, std::string> OuterLocalNames;

  /* Local-slot bookkeeping — matlab.alloc Values defined INSIDE the
   * kernel body but read/written via matlab.load/store.  Become MSL
   * `double local_X = 0.0;` declarations at the top of the body. */
  llvm::DenseMap<Value, std::string> LocalNames;
  unsigned LocalCounter = 0;

  /* SSA Value → MSL expression mapping for inline op chains.  When an
   * op's only use is a single subsequent op, we inline its result as
   * a sub-expression rather than emitting a named local. */
  llvm::DenseMap<Value, std::string> ValueExpr;
  unsigned TempCounter = 0;

  /* True once the body walker hit an unsupported op shape; the caller
   * uses this to decide between the real body and the placeholder. */
  bool Bailed = false;
  std::string BailReason;

  std::string newTemp() {
    return "t" + std::to_string(TempCounter++);
  }
  std::string newLocal(StringRef Hint) {
    std::string N = Hint.empty()
        ? ("local_" + std::to_string(LocalCounter++))
        : ("local_" + std::string(Hint));
    LocalCounter++;
    return N;
  }
  void bail(StringRef Reason) {
    Bailed = true;
    if (BailReason.empty()) BailReason = std::string(Reason);
  }
};

/* Get the user-visible name of a matlab.alloc slot (the `name`
 * attribute on the op).  Returns the auto-name "slot_N" if absent. */
std::string allocName(Operation *Alloc, unsigned Fallback) {
  if (!Alloc) return "slot_" + std::to_string(Fallback);
  if (auto SA = Alloc->getAttrOfType<StringAttr>("name"))
    return std::string(SA.getValue());
  if (auto FA = Alloc->getAttrOfType<FlatSymbolRefAttr>("name"))
    return std::string(FA.getValue());
  return "slot_" + std::to_string(Fallback);
}

/* Walk the body to find captures and identify special slots.
 *
 * The MLIR shape for a `coder.gpu.kernelfun` body is:
 *
 *   "matlab.gpu.kernel"(%range) ({
 *   ^bb0(%iv: f64):
 *     "matlab.store"(%iv, %ivslot)       // IV's own slot store
 *     %a = "matlab.load"(%a_slot)        // outer scalar capture
 *     %i = "matlab.load"(%ivslot)        // re-read IV from slot
 *     %v = "matlab.matmul"(%a, %i)       // compute
 *     %y = "matlab.load"(%y_slot)        // load output tensor
 *     %i2 = "matlab.load"(%ivslot)       // re-read IV
 *     "matlab.call_builtin"(%y, %i2, %v) {callee="__subscript_store"}
 *     "matlab.yield"()
 *   }) {var = "i"}
 *
 * The collector classifies outer slots into:
 *   - IV slot (named by `var` attr) — every matlab.load on it
 *     resolves to the IV (block arg).
 *   - Output slot — the tensor-typed matlab.alloc whose load is the
 *     first operand of __subscript_store inside the body.
 *   - Scalar captures — f64-typed matlab.alloc slots whose loads are
 *     read (but not stored to) inside the body.
 */
void collectCaptures(Block &Body, MetalEmitCtx &Ctx) {
  Operation *KernelOp = Body.getParentOp();
  auto F64 = Float64Type::get(KernelOp->getContext());
  /* A scalar slot type: f64, or `none` for an unpromoted function param
   * (no type-promotion pass runs before -emit-metal; inside the
   * per-thread body it is a scalar).  Matches the CUDA/OpenCL emitters. */
  auto isScalarSlotTy = [&](Type T) {
    if (mlir::isa<RankedTensorType, UnrankedTensorType>(T)) return false;
    return T == F64 || mlir::isa<NoneType>(T);
  };

  /* Everything defined inside the kernel region (results + nested block
   * args) — a load/store on one of these is not an outer-slot access. */
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

  /* IV slot(s) — outer matlab.alloc(s) whose name matches the loop
   * var(s).  In 2-D the inner kernel's `var` names the j slot (IVSlot)
   * and Ctx.OuterVarName names the i slot (IVSlot2). */
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

  /* Output slot: tensor-typed outer matlab.alloc feeding __subscript_store. */
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
    if (!mlir::isa<RankedTensorType, UnrankedTensorType>(Slot.getType()))
      return;
    if (Ctx.OutputSlot && Ctx.OutputSlot != Slot) {
      Ctx.bail("multiple output tensors not supported");
      return;
    }
    Ctx.OutputSlot = Slot;
  });

  /* Outer f64 slots WRITTEN inside the body (excluding the IV slot) are
   * per-thread locals, not read-only captures.  Collect them first so a
   * slot that is both read and written is classified as a local. */
  llvm::DenseSet<Value> StoredSet;
  KernelOp->walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.store") || Op->getNumOperands() != 2) return;
    Value Slot = Op->getOperand(1);
    if (InsideDefs.count(Slot) || Slot == Ctx.IVSlot || Slot == Ctx.IVSlot2)
      return;
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc") || !isScalarSlotTy(Slot.getType())) return;
    if (StoredSet.insert(Slot).second) {
      Ctx.OuterLocalOrder.push_back(Slot);
      Ctx.OuterLocalNames[Slot] =
          "loc_" + allocName(Def, static_cast<unsigned>(StoredSet.size()));
    }
  });

  /* Scalar captures: f64-typed outer matlab.alloc slots that are READ
   * but never written (and not the IV) — read-only `constant` args. */
  llvm::DenseSet<Value> ScalarSet;
  KernelOp->walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.load") || Op->getNumOperands() != 1) return;
    Value Slot = Op->getOperand(0);
    if (InsideDefs.count(Slot)) return;
    if (Slot == Ctx.IVSlot || Slot == Ctx.IVSlot2 || Slot == Ctx.OutputSlot)
      return;
    if (StoredSet.count(Slot)) return;  /* it's a local */
    Operation *Def = Slot.getDefiningOp();
    if (!isMatlabOp(Def, "matlab.alloc") || !isScalarSlotTy(Slot.getType())) return;
    if (ScalarSet.insert(Slot).second) {
      Ctx.ScalarCaptureOrder.push_back(Slot);
      Ctx.ScalarCaptureNames[Slot] =
          allocName(Def, static_cast<unsigned>(ScalarSet.size()));
    }
  });
}

/* Get the MSL expression for a Value.  Handles inlined sub-expressions
 * (ValueExpr map), the induction var, captures, and the output. */
std::string exprFor(Value V, MetalEmitCtx &Ctx) {
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

/* Emit the kernel BODY (statements inside the braces).  Walks ops in
 * order, recognising the supported shapes. */
void emitBody(Block &Body, MetalEmitCtx &Ctx) {
  auto F64 = Float64Type::get(Body.getParentOp()->getContext());
  /* A scalar (non-tensor) result: f64, or `none` for an unpromoted
   * function param flowing through arithmetic (no type-promotion pass
   * runs before -emit-metal). */
  auto scalarish = [&](Type T) {
    return T == F64 || mlir::isa<NoneType>(T);
  };
  for (Operation &Op : Body) {
    if (Ctx.Bailed) return;
    StringRef Name = Op.getName().getStringRef();

    if (Name == "matlab.yield") continue;  /* terminator */

    if (auto C = dyn_cast<arith::ConstantOp>(&Op)) {
      Value V = C.getResult();
      if (V.getType() == F64) {
        if (auto Fa = mlir::dyn_cast<FloatAttr>(C.getValue())) {
          std::stringstream Ss;
          Ss << Fa.getValueAsDouble();
          Ctx.ValueExpr[V] = Ss.str();
          continue;
        }
      }
      if (auto Ia = mlir::dyn_cast<IntegerAttr>(C.getValue())) {
        std::stringstream Ss;
        Ss << Ia.getInt() << ".0";
        Ctx.ValueExpr[V] = Ss.str();
        continue;
      }
      Ctx.bail("unsupported constant shape");
      return;
    }

    if (Name == "matlab.const_int") {
      /* matlab.const_int (f64-typed integer literal) — emit as MSL
       * double literal.  Attribute "value" holds the int. */
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

    /* matlab.alloc inside the body — declare an MSL local slot. */
    if (Name == "matlab.alloc") {
      Value R = Op.getResult(0);
      if (!scalarish(R.getType())) {
        Ctx.bail("non-scalar local slot");
        return;
      }
      std::string LName = "loc_" + allocName(&Op, Ctx.LocalCounter++);
      Ctx.LocalNames[R] = LName;
      Ctx.Os << "  float " << LName << " = 0.0f;\n";
      continue;
    }

    if (Name == "matlab.load" && Op.getNumOperands() == 1) {
      Value Slot = Op.getOperand(0);
      /* Loads on the IV slot return the block-arg IV.  Loads on the
       * output slot are the load that feeds __subscript_store —
       * resolve to the output name (the store-site handler reads
       * Ctx.OutputName directly via exprFor). */
      if (Slot == Ctx.IVSlot) {
        Ctx.ValueExpr[Op.getResult(0)] = Ctx.IVName;
      } else if (Ctx.TwoD && Slot == Ctx.IVSlot2) {
        Ctx.ValueExpr[Op.getResult(0)] = Ctx.IV2Name;
      } else {
        Ctx.ValueExpr[Op.getResult(0)] = exprFor(Slot, Ctx);
      }
      continue;
    }

    if (Name == "matlab.store" && Op.getNumOperands() == 2) {
      /* matlab.store(value, slot) — slot must be a local, an IV slot
       * (no-op; IV is the block arg), or the output slot (no-op; we
       * handle output writes via __subscript_store). */
      Value Slot = Op.getOperand(1);
      if (Slot == Ctx.IVSlot) continue;
      if (Ctx.TwoD && Slot == Ctx.IVSlot2) continue;
      if (Slot == Ctx.OutputSlot) continue;
      std::string SlotE = exprFor(Slot, Ctx);
      std::string ValE = exprFor(Op.getOperand(0), Ctx);
      if (SlotE == Ctx.IVName) continue;
      Ctx.Os << "  " << SlotE << " = " << ValE << ";\n";
      continue;
    }

    /* Binary arithmetic on f64 scalars: matlab.{add,sub,mul,div} and the
     * matrix-spelled scalar forms matlab.{matmul,matdiv}. */
    if ((Name == "matlab.add" || Name == "matlab.sub" ||
         Name == "matlab.mul" || Name == "matlab.div" ||
         Name == "matlab.matmul" || Name == "matlab.matdiv") &&
        Op.getNumOperands() == 2 &&
        scalarish(Op.getResult(0).getType()) &&
        !mlir::isa<RankedTensorType, UnrankedTensorType>(
            Op.getOperand(0).getType()) &&
        !mlir::isa<RankedTensorType, UnrankedTensorType>(
            Op.getOperand(1).getType())) {
      const char *Sym = (Name == "matlab.add")                          ? "+"
                       : (Name == "matlab.sub")                         ? "-"
                       : (Name == "matlab.div" || Name == "matlab.matdiv") ? "/"
                                                                          : "*";
      std::string A = exprFor(Op.getOperand(0), Ctx);
      std::string B = exprFor(Op.getOperand(1), Ctx);
      std::string E = "(" + A + " " + Sym + " " + B + ")";
      Ctx.ValueExpr[Op.getResult(0)] = E;
      continue;
    }

    /* matlab.neg — unary minus on a scalar. */
    if (Name == "matlab.neg" && Op.getNumOperands() == 1 &&
        scalarish(Op.getResult(0).getType())) {
      Ctx.ValueExpr[Op.getResult(0)] =
          "(-" + exprFor(Op.getOperand(0), Ctx) + ")";
      continue;
    }

    /* arith.{add,sub,mul,div}f on f64. */
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(&Op)) {
      const char *Sym = isa<arith::AddFOp>(&Op) ? "+"
                     : isa<arith::SubFOp>(&Op) ? "-"
                     : isa<arith::MulFOp>(&Op) ? "*"
                                               : "/";
      std::string A = exprFor(Op.getOperand(0), Ctx);
      std::string B = exprFor(Op.getOperand(1), Ctx);
      std::string E = "(" + A + " " + Sym + " " + B + ")";
      Ctx.ValueExpr[Op.getResult(0)] = E;
      continue;
    }

    /* Scalar relational ops (result i1): matlab.{lt,le,gt,ge,eq,ne}. */
    {
      const char *Rel = nullptr;
      if (Name == "matlab.lt") Rel = "<";
      else if (Name == "matlab.le") Rel = "<=";
      else if (Name == "matlab.gt") Rel = ">";
      else if (Name == "matlab.ge") Rel = ">=";
      else if (Name == "matlab.eq") Rel = "==";
      else if (Name == "matlab.ne") Rel = "!=";
      if (Rel && Op.getNumOperands() == 2) {
        Ctx.ValueExpr[Op.getResult(0)] =
            "(" + exprFor(Op.getOperand(0), Ctx) + " " + Rel + " " +
            exprFor(Op.getOperand(1), Ctx) + ")";
        continue;
      }
    }

    /* Short-circuit / element-wise logical ops (result i1). */
    {
      const char *Lg = nullptr;
      if (Name == "matlab.short_and" || Name == "matlab.and") Lg = "&&";
      else if (Name == "matlab.short_or" || Name == "matlab.or") Lg = "||";
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

    /* matlab.while — condition region (region 0) yields an i1; body
     * region (region 1) runs while true.  Emit a device-side
     * `while (cond) { body }`.  The condition's loads reference the
     * scalar locals, which mutate in the body, so re-evaluating the
     * inlined expression each iteration is correct. */
    if (Name == "matlab.while" && Op.getNumRegions() == 2 &&
        Op.getRegion(0).hasOneBlock() && Op.getRegion(1).hasOneBlock()) {
      Block &Cond = Op.getRegion(0).front();
      Block &Loop = Op.getRegion(1).front();
      emitBody(Cond, Ctx);  // builds ValueExpr for the condition chain
      if (Ctx.Bailed) return;
      Operation *Term = Cond.getTerminator();
      if (!Term || Term->getNumOperands() != 1) {
        Ctx.bail("matlab.while condition without a yielded value");
        return;
      }
      std::string CondE = exprFor(Term->getOperand(0), Ctx);
      Ctx.Os << "  while (" << CondE << ") {\n";
      emitBody(Loop, Ctx);
      Ctx.Os << "  }\n";
      continue;
    }

    /* matlab.call_builtin "__subscript_store" — Y(i) = v (1-D, operands
     * tensor,i,value) or Y(i,j) = v (2-D, operands tensor,i,j,value).
     * MATLAB is 1-based + column-major; MSL is 0-based: 1-D →
     * `out[(int)i - 1]`, 2-D → `out[(int)(i-1) + (int)(j-1)*nrows]`. */
    if (Name == "matlab.call_builtin") {
      auto Cal = Op.getAttrOfType<StringAttr>("callee");
      if (Cal && Cal.getValue() == "__subscript_store" &&
          Op.getNumOperands() == 3) {
        std::string T = exprFor(Op.getOperand(0), Ctx);
        std::string I = exprFor(Op.getOperand(1), Ctx);
        std::string V = exprFor(Op.getOperand(2), Ctx);
        Ctx.Os << "  " << T << "[(int)(" << I << ") - 1] = " << V << ";\n";
        continue;
      }
      if (Cal && Cal.getValue() == "__subscript_store" &&
          Op.getNumOperands() == 4 && Ctx.TwoD) {
        std::string T = exprFor(Op.getOperand(0), Ctx);
        std::string I = exprFor(Op.getOperand(1), Ctx);
        std::string J = exprFor(Op.getOperand(2), Ctx);
        std::string V = exprFor(Op.getOperand(3), Ctx);
        Ctx.Os << "  " << T << "[((int)(" << I << ") - 1) + ((int)(" << J
               << ") - 1) * " << Ctx.NRowsExpr << "] = " << V << ";\n";
        continue;
      }
      if (Cal && Cal.getValue() == "__subscript_load" &&
          Op.getNumOperands() >= 2) {
        std::string T = exprFor(Op.getOperand(0), Ctx);
        std::string I = exprFor(Op.getOperand(1), Ctx);
        std::string E = T + "[(int)(" + I + ") - 1]";
        Ctx.ValueExpr[Op.getResult(0)] = E;
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

/* Public entry — walk every matlab.gpu.kernel op in M and emit MSL
 * source for each.  Returns the combined source string (one kernel
 * function per op, separated by blank lines).  Unsupported kernels
 * emit a `// FALLBACK: <reason>` comment + a placeholder body so the
 * file still compiles. */
std::string emitMetalKernels(mlir::ModuleOp M, llvm::StringRef Prefix,
                             GpuKernelInfo *Info) {
  std::stringstream OS;
  OS << "/* Generated by matlabc -emit-metal.\n"
     << " * Walks matlab.gpu.kernel ops and translates each body op-by-op\n"
     << " * to MSL.  Unsupported shapes (nested loops / while / multiple\n"
     << " * outputs) fall back to an identity placeholder with a comment\n"
     << " * naming the bail reason.\n"
     << " */\n\n"
     << "#include <metal_stdlib>\n"
     << "using namespace metal;\n\n";

  unsigned KId = 0;
  M.walk([&](mlir::Operation *Op) {
    if (Op->getName().getStringRef() != "matlab.gpu.kernel") return;
    /* Skip a kernel nested inside another — the enclosing kernel emits
     * it as part of a flattened multi-dimensional grid. */
    for (Operation *Anc = Op->getParentOp(); Anc; Anc = Anc->getParentOp())
      if (isMatlabOp(Anc, "matlab.gpu.kernel")) return;
    if (Op->getNumRegions() != 1) return;
    auto &Body = Op->getRegion(0);
    if (!Body.hasOneBlock()) return;
    auto &BB = Body.front();
    if (BB.getNumArguments() != 1) return;

    MetalEmitCtx Ctx;
    Ctx.KernelName = (std::string(Prefix) + "_kernel_" +
                      std::to_string(KId++));

    /* Detect a `coder.gpu.kernelfun` for-i × for-j nest: the body's only
     * real content is a single nested kernel.  Flatten to a 2-D grid —
     * the primary IV is the INNER loop (j ← tid.x), the secondary is the
     * OUTER loop (i ← tid.y), and the column-major leading dimension is
     * the grid's i-extent. */
    Operation *Inner = nullptr;
    for (Operation &O : BB)
      if (isMatlabOp(&O, "matlab.gpu.kernel")) { Inner = &O; break; }
    Block *ComputeBB = &BB;
    if (Inner && Inner->getNumRegions() == 1 &&
        Inner->getRegion(0).hasOneBlock() &&
        Inner->getRegion(0).front().getNumArguments() == 1) {
      ComputeBB = &Inner->getRegion(0).front();
      Ctx.TwoD = true;
      Ctx.IV = ComputeBB->getArgument(0);   /* inner loop j */
      Ctx.IVName = "j_iv";
      Ctx.IV2 = BB.getArgument(0);          /* outer loop i */
      Ctx.IV2Name = "i_iv";
      if (auto VA = Op->getAttrOfType<StringAttr>("var"))
        Ctx.OuterVarName = std::string(VA.getValue());
      Ctx.NRowsExpr = "(int)gsz.y";
    } else {
      Ctx.IV = BB.getArgument(0);
    }

    collectCaptures(*ComputeBB, Ctx);

    /* Signature.  Output goes to buffer(0), scalar captures take
     * buffers 1..N, tid is thread_position_in_grid (uint2 in 2-D). */
    OS << "kernel void " << Ctx.KernelName << "(\n";
    if (Ctx.OutputSlot)
      OS << "    device float *" << Ctx.OutputName
         << " [[buffer(0)]],\n";
    unsigned BufIdx = Ctx.OutputSlot ? 1 : 0;
    for (auto V : Ctx.ScalarCaptureOrder) {
      OS << "    constant float &" << Ctx.ScalarCaptureNames[V]
         << " [[buffer(" << BufIdx++ << ")]],\n";
    }
    if (Ctx.TwoD) {
      OS << "    uint2 gtid [[thread_position_in_grid]],\n"
         << "    uint2 gsz [[threads_per_grid]])\n"
         << "{\n"
         << "  float " << Ctx.IVName << " = float(gtid.x) + 1.0f;\n"
         << "  float " << Ctx.IV2Name << " = float(gtid.y) + 1.0f;\n";
    } else {
      OS << "    uint tid [[thread_position_in_grid]])\n"
         << "{\n"
         << "  float " << Ctx.IVName << " = float(tid) + 1.0f;\n";
    }
    /* Declare per-thread locals (outer f64 slots written in the body). */
    for (Value V : Ctx.OuterLocalOrder)
      OS << "  float " << Ctx.OuterLocalNames[V] << " = 0.0f;\n";

    /* Emit the translated body or the bail placeholder. */
    std::stringstream BodyOS;
    Ctx.Os.swap(BodyOS);
    emitBody(*ComputeBB, Ctx);
    Ctx.Os.swap(BodyOS);

    if (Ctx.Bailed) {
      OS << "  // FALLBACK: " << Ctx.BailReason << ".  Identity kernel.\n";
      if (Ctx.OutputSlot)
        OS << "  // " << Ctx.OutputName
           << "[tid] = (placeholder — outliner gap);\n";
    } else {
      OS << BodyOS.str();
    }
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

  if (KId == 0) {
    OS << "/* No matlab.gpu.kernel ops found in the module — file emitted\n"
       << " * as a stub.  Use `coder.gpu.kernelfun()` inside the function\n"
       << " * body to trigger kernel detection. */\n";
  }
  return OS.str();
}

}  // namespace mlirgen
}  // namespace matlab
