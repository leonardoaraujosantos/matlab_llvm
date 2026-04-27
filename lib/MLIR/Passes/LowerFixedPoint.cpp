// Lowers Fixed-Point Designer (`fi`) ops into integer-shift sequences.
//
// Phase 1 surface (see docs/emit_fixed_point.md §7):
//   - matlab.fi.const  -> arith.constant (the stored integer)
//   - matlab.fi.cast   -> matlab_fi_quantize_{s,u}(double) for the
//                          constructor cast, or shift+saturate+truncate
//                          for the (:) clamp cast (fi -> fi).
//   - matlab.add       -> integer add: extend each operand to the result
//                          width, left-shift the smaller-FL side to align
//                          fraction lengths, addi.
//   - matlab.sub       -> mirrored from add via subi.
//   - matlab.matmul    -> integer mul on extended operands (FullPrecision).
//   - matlab.emul      -> same as matmul on scalars.
//   - matlab.neg       -> arith.subi 0, x.
//
// Each rewrite is gated on the `fi` attribute set by the frontend
// (Lowering.cpp's buildFixedAttrs). Ops without the `fi` attribute are
// left for LowerScalarsToArith to handle the regular numeric way.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace mlir;

namespace matlab {
namespace mlirgen {

namespace {

bool isMatlabOp(Operation *Op, llvm::StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

bool hasFiTag(Operation *Op) {
  return Op && Op->getAttrOfType<IntegerAttr>("fi") != nullptr;
}

struct FiSpec {
  bool Signed = true;
  unsigned WL = 16;
  int FL = 15;
  unsigned OF = 1; // Saturate by default
  unsigned RM = 0; // Floor by default
};

FiSpec readSpec(Operation *Op, llvm::StringRef Prefix) {
  FiSpec S;
  auto getI = [&](llvm::StringRef Suffix, int Default) -> int64_t {
    auto A = Op->getAttrOfType<IntegerAttr>((Prefix + Suffix).str());
    return A ? A.getInt() : (int64_t)Default;
  };
  if (Prefix == "fi") {
    S.Signed = getI("_signed", 1) != 0;
    S.WL = (unsigned)getI("_wl", 16);
    S.FL = (int)getI("_fl", 15);
    S.OF = (unsigned)getI("_of", 1);
    S.RM = (unsigned)getI("_rm", 0);
  } else {
    S.Signed = getI("_signed", 1) != 0;
    S.WL = (unsigned)getI("_wl", 16);
    S.FL = (int)getI("_fl", 15);
  }
  return S;
}

unsigned storageBits(unsigned WL) {
  if (WL == 0) return 8;
  if (WL <= 8) return 8;
  if (WL <= 16) return 16;
  if (WL <= 32) return 32;
  return 64;
}

LLVM::LLVMFuncOp getOrInsertRTDecl(OpBuilder &B, ModuleOp M,
                                   llvm::StringRef Name, Type Result,
                                   ArrayRef<Type> Args) {
  if (auto Existing = M.lookupSymbol<LLVM::LLVMFuncOp>(Name)) return Existing;
  OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(M.getBody());
  auto Ty = LLVM::LLVMFunctionType::get(Result, Args);
  auto F = LLVM::LLVMFuncOp::create(B, M.getLoc(), Name, Ty);
  F.setLinkage(LLVM::Linkage::External);
  return F;
}

/// Produce a Value of the given target integer type (signless), extending
/// or truncating from the source if needed. Signedness only affects the
/// extension flavour (extsi vs extui). MLIR signless integers carry no
/// signedness in the type itself; the op picks how the bits are
/// interpreted.
Value extendOrTruncate(OpBuilder &B, Location L, Value V, unsigned TargetBits,
                       bool Signed) {
  Type T = V.getType();
  // The frontend may have produced si16/ui8 etc. (signed/unsigned MLIR
  // integer types); reduce to a signless integer for arith ops.
  unsigned SrcBits = 0;
  if (auto IT = dyn_cast<IntegerType>(T)) SrcBits = IT.getWidth();
  Type Signless = IntegerType::get(B.getContext(), TargetBits);
  Type SignlessSrc = IntegerType::get(B.getContext(), SrcBits);
  if (T != SignlessSrc) {
    V = arith::BitcastOp::create(B, L, SignlessSrc, V);
  }
  if (TargetBits == SrcBits) return V;
  if (TargetBits > SrcBits) {
    if (Signed) return arith::ExtSIOp::create(B, L, Signless, V);
    return arith::ExtUIOp::create(B, L, Signless, V);
  }
  return arith::TruncIOp::create(B, L, Signless, V);
}

Value emitConstantInt(OpBuilder &B, Location L, unsigned Bits, int64_t Val) {
  Type T = IntegerType::get(B.getContext(), Bits);
  return arith::ConstantOp::create(B, L, T,
                                   IntegerAttr::get(T, Val));
}

/// Saturate a value (`In`, signless integer of width `Bits`) to `WL` bits
/// of `Signed` interpretation by calling matlab_fi_sat_s64 / _u64. The
/// helper takes int64 so we widen first and narrow back.
Value emitSaturate(OpBuilder &B, Location L, ModuleOp M, Value In,
                   unsigned WL, bool Signed) {
  MLIRContext *Ctx = B.getContext();
  Type I64 = IntegerType::get(Ctx, 64);
  Type I8 = IntegerType::get(Ctx, 8);
  Value Wide;
  if (auto IT = dyn_cast<IntegerType>(In.getType());
      IT && IT.getWidth() == 64) {
    Wide = In;
  } else if (Signed) {
    Wide = arith::ExtSIOp::create(B, L, I64, In);
  } else {
    Wide = arith::ExtUIOp::create(B, L, I64, In);
  }
  llvm::StringRef Name = Signed ? "matlab_fi_sat_s64" : "matlab_fi_sat_u64";
  auto Fn = getOrInsertRTDecl(B, M, Name, I64, {I64, I8});
  Value WLV = arith::ConstantOp::create(B, L, I8,
                                        IntegerAttr::get(I8, (int64_t)WL));
  auto Call = LLVM::CallOp::create(B, L, Fn, ValueRange{Wide, WLV});
  Value Out = Call.getResult();
  // Narrow back to the input's original width.
  if (auto IT = dyn_cast<IntegerType>(In.getType())) {
    if (IT.getWidth() < 64) {
      Type T = IntegerType::get(Ctx, IT.getWidth());
      Out = arith::TruncIOp::create(B, L, T, Out);
    }
  }
  return Out;
}

/// Apply the configured rounding mode to shift `In` right by `Shift` bits.
/// Falls back to a plain arithmetic right shift for shift==0 or for
/// rounding modes Phase 1 doesn't ship.
Value emitRoundingShift(OpBuilder &B, Location L, ModuleOp M, Value In,
                        unsigned Shift, unsigned Rounding, bool Signed) {
  if (Shift == 0) return In;
  MLIRContext *Ctx = B.getContext();
  unsigned Bits = 64;
  if (auto IT = dyn_cast<IntegerType>(In.getType())) Bits = IT.getWidth();
  if (Rounding == 1 /* Nearest */) {
    // (x + (1 << (shift-1))) >> shift
    Type T = IntegerType::get(Ctx, Bits);
    int64_t HalfV = ((int64_t)1) << (Shift - 1);
    Value Half = arith::ConstantOp::create(B, L, T,
                                           IntegerAttr::get(T, HalfV));
    Value Adjusted = arith::AddIOp::create(B, L, In, Half);
    Value Sh = emitConstantInt(B, L, Bits, (int64_t)Shift);
    if (Signed) return arith::ShRSIOp::create(B, L, Adjusted, Sh);
    return arith::ShRUIOp::create(B, L, Adjusted, Sh);
  }
  // Floor (and unsupported modes): plain shift.
  Value Sh = emitConstantInt(B, L, Bits, (int64_t)Shift);
  if (Signed) return arith::ShRSIOp::create(B, L, In, Sh);
  return arith::ShRUIOp::create(B, L, In, Sh);
}

//===----------------------------------------------------------------------===//
// Per-op rewrites
//===----------------------------------------------------------------------===//

/// matlab.fi.const : () -> iN  {value = K, fi_*}  ->  arith.constant K : iN
bool rewriteFiConst(Operation *Op) {
  OpBuilder B(Op);
  auto V = Op->getAttrOfType<IntegerAttr>("value");
  if (!V) return false;
  Type Ty = Op->getResult(0).getType();
  // Make sure we're producing a signless integer for arith.constant.
  Type Signless = Ty;
  if (auto IT = dyn_cast<IntegerType>(Ty)) {
    Signless = IntegerType::get(B.getContext(), IT.getWidth());
  }
  Value C = arith::ConstantOp::create(B, Op->getLoc(), Signless,
                                      IntegerAttr::get(Signless, V.getInt()));
  if (Signless != Ty) {
    C = arith::BitcastOp::create(B, Op->getLoc(), Ty, C);
  }
  Op->getResult(0).replaceAllUsesWith(C);
  Op->erase();
  return true;
}

/// matlab.fi.cast : f64 -> i64  {callee = matlab_fi_quantize_*, fi_*}
///   -> llvm.call @matlab_fi_quantize_*(value, WL, FL, OF, RM)
///
/// matlab.fi.cast : iA -> iB  {fi_clamp, fi_*}
///   -> shift right by (FL_in - FL_out), saturate to WL_out bits, truncate.
bool rewriteFiCast(Operation *Op, ModuleOp M) {
  if (Op->getNumOperands() != 1) return false;
  OpBuilder B(Op);
  Location L = Op->getLoc();
  Value In = Op->getOperand(0);
  Type ResTy = Op->getResult(0).getType();

  // Constructor cast (double -> stored integer): emit a runtime quantize.
  if (isa<Float64Type, Float32Type>(In.getType())) {
    auto Callee = Op->getAttrOfType<StringAttr>("callee");
    if (!Callee) return false;
    FiSpec Out = readSpec(Op, "fi");
    MLIRContext *Ctx = B.getContext();
    Type F64 = Float64Type::get(Ctx);
    Type I64 = IntegerType::get(Ctx, 64);
    Type I8 = IntegerType::get(Ctx, 8);
    if (In.getType() != F64) {
      In = arith::ExtFOp::create(B, L, F64, In);
    }
    auto Fn = getOrInsertRTDecl(B, M, Callee.getValue(), I64,
                                 {F64, I8, I8, I8, I8});
    Value WL = arith::ConstantOp::create(B, L, I8,
                                          IntegerAttr::get(I8, (int64_t)Out.WL));
    Value FL = arith::ConstantOp::create(B, L, I8,
                                          IntegerAttr::get(I8, (int64_t)Out.FL));
    Value OF = arith::ConstantOp::create(B, L, I8,
                                          IntegerAttr::get(I8, (int64_t)Out.OF));
    Value RM = arith::ConstantOp::create(B, L, I8,
                                          IntegerAttr::get(I8, (int64_t)Out.RM));
    auto Call = LLVM::CallOp::create(B, L, Fn,
                                      ValueRange{In, WL, FL, OF, RM});
    Value Result = Call.getResult();
    if (ResTy != I64) {
      // Truncate to the storage class.
      unsigned Bits = 64;
      if (auto IT = dyn_cast<IntegerType>(ResTy)) Bits = IT.getWidth();
      Type Signless = IntegerType::get(Ctx, Bits);
      Value Narrow = arith::TruncIOp::create(B, L, Signless, Result);
      if (ResTy != Signless) {
        Narrow = arith::BitcastOp::create(B, L, ResTy, Narrow);
      }
      Result = Narrow;
    }
    Op->getResult(0).replaceAllUsesWith(Result);
    Op->erase();
    return true;
  }

  // (:) clamp cast (fi -> fi rebind): shift, saturate, truncate.
  auto IsClamp = Op->getAttrOfType<IntegerAttr>("fi_clamp");
  if (!IsClamp || !isa<IntegerType>(In.getType()) ||
      !isa<IntegerType>(ResTy))
    return false;
  FiSpec Out = readSpec(Op, "fi");
  // The operand carries its own spec attached by the producing op (binop
  // result). Read fi_lhs_* if present (we conventionally tag the source
  // spec under fi_lhs_* on the cast); fall back to inferring source FL
  // from the operand's bit width minus default.
  unsigned SrcBits = cast<IntegerType>(In.getType()).getWidth();
  // For now we require the cast to carry fi_lhs_* attrs naming the
  // source spec. The frontend's `(:)` lowering does not yet attach
  // these — without them we approximate FL_in == FL_out (no shift), i.e.
  // a pure saturate+truncate. That's the conservative fallback.
  int SrcFL = Out.FL;
  bool SrcSigned = Out.Signed;
  (void)SrcBits;
  if (auto LhsFL = Op->getAttrOfType<IntegerAttr>("fi_lhs_fl"))
    SrcFL = (int)LhsFL.getInt();
  if (auto LhsSn = Op->getAttrOfType<IntegerAttr>("fi_lhs_signed"))
    SrcSigned = LhsSn.getInt() != 0;

  // Step 1: shift right by (SrcFL - DstFL) under the rounding mode if
  // SrcFL > DstFL; left-shift by (DstFL - SrcFL) otherwise.
  Value V = In;
  unsigned WorkBits = std::max(SrcBits,
                               storageBits(Out.WL));
  V = extendOrTruncate(B, L, V, WorkBits, SrcSigned);
  if (SrcFL > Out.FL) {
    V = emitRoundingShift(B, L, M, V,
                          (unsigned)(SrcFL - Out.FL), Out.RM, SrcSigned);
  } else if (SrcFL < Out.FL) {
    Value Sh = emitConstantInt(B, L, WorkBits, Out.FL - SrcFL);
    V = arith::ShLIOp::create(B, L, V, Sh);
  }
  // Step 2: saturate to Out.WL bits if requested.
  if (Out.OF == 1 /* Saturate */) {
    V = emitSaturate(B, L, M, V, Out.WL, Out.Signed);
  }
  // Step 3: narrow to the destination storage class.
  V = extendOrTruncate(B, L, V, storageBits(Out.WL), Out.Signed);
  if (V.getType() != ResTy) {
    V = arith::BitcastOp::create(B, L, ResTy, V);
  }
  Op->getResult(0).replaceAllUsesWith(V);
  Op->erase();
  return true;
}

/// matlab.add / matlab.sub (fi-tagged) -> integer add/sub with FL alignment.
bool rewriteFiAddSub(Operation *Op, ModuleOp M, bool IsSub) {
  if (Op->getNumOperands() != 2) return false;
  OpBuilder B(Op);
  Location L = Op->getLoc();
  Value Lhs = Op->getOperand(0);
  Value Rhs = Op->getOperand(1);
  Type ResTy = Op->getResult(0).getType();
  if (!isa<IntegerType>(Lhs.getType()) ||
      !isa<IntegerType>(Rhs.getType()) ||
      !isa<IntegerType>(ResTy))
    return false;

  FiSpec Out = readSpec(Op, "fi");
  FiSpec Ls = readSpec(Op, "fi_lhs");
  FiSpec Rs = readSpec(Op, "fi_rhs");
  unsigned WorkBits = storageBits(Out.WL);

  Value LE = extendOrTruncate(B, L, Lhs, WorkBits, Ls.Signed);
  Value RE = extendOrTruncate(B, L, Rhs, WorkBits, Rs.Signed);
  if (Ls.FL < Out.FL) {
    Value Sh = emitConstantInt(B, L, WorkBits, Out.FL - Ls.FL);
    LE = arith::ShLIOp::create(B, L, LE, Sh);
  }
  if (Rs.FL < Out.FL) {
    Value Sh = emitConstantInt(B, L, WorkBits, Out.FL - Rs.FL);
    RE = arith::ShLIOp::create(B, L, RE, Sh);
  }
  Value Sum = IsSub
      ? (Value)arith::SubIOp::create(B, L, LE, RE)
      : (Value)arith::AddIOp::create(B, L, LE, RE);
  if (Out.OF == 1 /* Saturate */) {
    Sum = emitSaturate(B, L, M, Sum, Out.WL, Out.Signed);
  }
  // Narrow / extend to result storage class.
  Sum = extendOrTruncate(B, L, Sum, storageBits(Out.WL), Out.Signed);
  if (Sum.getType() != ResTy) {
    Sum = arith::BitcastOp::create(B, L, ResTy, Sum);
  }
  Op->getResult(0).replaceAllUsesWith(Sum);
  Op->erase();
  return true;
}

/// matlab.matmul / matlab.emul (fi-tagged scalar) -> integer mul.
/// Only the scalar shape is handled here. fi arrays are Phase 3 work.
bool rewriteFiMul(Operation *Op, ModuleOp M) {
  if (Op->getNumOperands() != 2) return false;
  OpBuilder B(Op);
  Location L = Op->getLoc();
  Value Lhs = Op->getOperand(0);
  Value Rhs = Op->getOperand(1);
  Type ResTy = Op->getResult(0).getType();
  if (!isa<IntegerType>(Lhs.getType()) ||
      !isa<IntegerType>(Rhs.getType()) ||
      !isa<IntegerType>(ResTy))
    return false;

  FiSpec Out = readSpec(Op, "fi");
  FiSpec Ls = readSpec(Op, "fi_lhs");
  FiSpec Rs = readSpec(Op, "fi_rhs");
  unsigned WorkBits = storageBits(Out.WL);

  Value LE = extendOrTruncate(B, L, Lhs, WorkBits, Ls.Signed);
  Value RE = extendOrTruncate(B, L, Rhs, WorkBits, Rs.Signed);
  Value Prod = arith::MulIOp::create(B, L, LE, RE);
  // Natural FL = Ls.FL + Rs.FL. If output FL is narrower, shift right.
  int NaturalFL = Ls.FL + Rs.FL;
  if (NaturalFL > Out.FL) {
    Prod = emitRoundingShift(B, L, M, Prod,
                             (unsigned)(NaturalFL - Out.FL),
                             Out.RM, Out.Signed);
  } else if (NaturalFL < Out.FL) {
    Value Sh = emitConstantInt(B, L, WorkBits, Out.FL - NaturalFL);
    Prod = arith::ShLIOp::create(B, L, Prod, Sh);
  }
  if (Out.OF == 1 /* Saturate */) {
    Prod = emitSaturate(B, L, M, Prod, Out.WL, Out.Signed);
  }
  Prod = extendOrTruncate(B, L, Prod, storageBits(Out.WL), Out.Signed);
  if (Prod.getType() != ResTy) {
    Prod = arith::BitcastOp::create(B, L, ResTy, Prod);
  }
  Op->getResult(0).replaceAllUsesWith(Prod);
  Op->erase();
  return true;
}

/// matlab.call_builtin @matlab_fi_* / @matlab_mat_i64_* / @matlab_mat_u64_*
/// -> llvm.call. The signature is inferred from the operand and result
/// types directly so we don't need a per-name table here; only the callee
/// prefix matters.
bool rewriteFiCallBuiltin(Operation *Op, ModuleOp M) {
  auto Callee = Op->getAttrOfType<StringAttr>("callee");
  if (!Callee) return false;
  llvm::StringRef N = Callee.getValue();
  if (!N.starts_with("matlab_fi_") &&
      !N.starts_with("matlab_mat_i64_") &&
      !N.starts_with("matlab_mat_u64_") &&
      !N.starts_with("matlab_persistent_"))
    return false;
  // Defer the rewrite if any operand is not yet LLVM-compatible — e.g. a
  // tensor coming out of matlab.range that LowerTensorOps still has to
  // retype to ptr. The pass will be re-run after LowerTensorOps and pick
  // this site up then.
  for (Value V : Op->getOperands()) {
    Type T = V.getType();
    if (isa<RankedTensorType, UnrankedTensorType, NoneType>(T))
      return false;
  }
  OpBuilder B(Op);
  Location L = Op->getLoc();
  llvm::SmallVector<Type, 6> ArgTys;
  for (Value V : Op->getOperands()) ArgTys.push_back(V.getType());
  Type ResTy = Op->getNumResults() == 1
      ? Op->getResult(0).getType()
      : LLVM::LLVMVoidType::get(B.getContext());
  if (auto NT = dyn_cast<NoneType>(ResTy))
    ResTy = LLVM::LLVMVoidType::get(B.getContext());
  auto Fn = getOrInsertRTDecl(B, M, Callee.getValue(), ResTy, ArgTys);
  auto Call = LLVM::CallOp::create(B, L, Fn, Op->getOperands());
  if (Op->getNumResults() == 1 && !isa<LLVM::LLVMVoidType>(ResTy)) {
    Op->getResult(0).replaceAllUsesWith(Call.getResult());
  }
  Op->erase();
  return true;
}

/// matlab.neg (fi-tagged) -> arith.subi 0, x.
bool rewriteFiNeg(Operation *Op) {
  if (Op->getNumOperands() != 1) return false;
  OpBuilder B(Op);
  Location L = Op->getLoc();
  Value In = Op->getOperand(0);
  Type ResTy = Op->getResult(0).getType();
  if (!isa<IntegerType>(In.getType()) || !isa<IntegerType>(ResTy))
    return false;
  unsigned Bits = cast<IntegerType>(In.getType()).getWidth();
  Type Signless = IntegerType::get(B.getContext(), Bits);
  Value Op0 = In;
  if (In.getType() != Signless)
    Op0 = arith::BitcastOp::create(B, L, Signless, In);
  Value Zero = emitConstantInt(B, L, Bits, 0);
  Value Neg = arith::SubIOp::create(B, L, Zero, Op0);
  if (Neg.getType() != ResTy)
    Neg = arith::BitcastOp::create(B, L, ResTy, Neg);
  Op->getResult(0).replaceAllUsesWith(Neg);
  Op->erase();
  return true;
}

} // namespace

bool runLowerFixedPoint(ModuleOp M) {
  llvm::SmallVector<Operation *, 32> Targets;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.fi.const") ||
        isMatlabOp(Op, "matlab.fi.cast")) {
      Targets.push_back(Op);
      return;
    }
    if (isMatlabOp(Op, "matlab.call_builtin")) {
      auto C = Op->getAttrOfType<StringAttr>("callee");
      if (!C) return;
      auto N = C.getValue();
      if (N.starts_with("matlab_fi_") ||
          N.starts_with("matlab_mat_i64_") ||
          N.starts_with("matlab_mat_u64_") ||
          N.starts_with("matlab_persistent_"))
        Targets.push_back(Op);
      /* numerictype / fimath / fipref are pure compile-time fi metadata —
       * Sema reads their args at type-inference time; the constructor
       * call has no runtime presence. Drop the op (and any dead
       * matlab.const_char / matlab.alloc / matlab.store chain feeding /
       * receiving its result will be cleaned up by the regular dead-op
       * sweep). */
      if (N == "numerictype" || N == "fimath" || N == "fipref")
        Targets.push_back(Op);
      return;
    }
    if (!hasFiTag(Op)) return;
    auto N = Op->getName().getStringRef();
    if (N == "matlab.add" || N == "matlab.sub" ||
        N == "matlab.matmul" || N == "matlab.emul" ||
        N == "matlab.neg")
      Targets.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Targets) {
    auto N = Op->getName().getStringRef();
    bool Did = false;
    if (N == "matlab.fi.const") Did = rewriteFiConst(Op);
    else if (N == "matlab.fi.cast") Did = rewriteFiCast(Op, M);
    else if (N == "matlab.call_builtin") {
      auto C = Op->getAttrOfType<StringAttr>("callee");
      if (C && (C.getValue() == "numerictype" ||
                C.getValue() == "fimath" ||
                C.getValue() == "fipref")) {
        /* Compile-time-only constructor — drop the op along with the
         * load/store/alloc chain that the binding sits on, since no
         * runtime value is meaningful. We collect the corpse list before
         * erasing because erase invalidates uses. */
        llvm::SmallVector<Operation *, 8> Corpses;
        if (Op->getNumResults() == 1) {
          for (Operation *U : Op->getResult(0).getUsers()) {
            Corpses.push_back(U);
            // matlab.store consumers also pin a matlab.alloc slot whose
            // only writer was this store; collect the slot so reads
            // (loads / passes-through) are also dropped if their slot
            // becomes empty. The slot is operand 1 of matlab.store.
            if (U->getName().getStringRef() == "matlab.store" &&
                U->getNumOperands() == 2) {
              Value Slot = U->getOperand(1);
              if (auto *D = Slot.getDefiningOp())
                if (D->getName().getStringRef() == "matlab.alloc") {
                  for (Operation *SU : Slot.getUsers())
                    if (SU != U) Corpses.push_back(SU);
                  Corpses.push_back(D);
                }
            }
          }
        }
        for (Operation *Co : Corpses) Co->dropAllUses();
        for (Operation *Co : Corpses) Co->erase();
        Op->erase();
        Did = true;
      } else {
        Did = rewriteFiCallBuiltin(Op, M);
      }
    }
    else if (N == "matlab.add") Did = rewriteFiAddSub(Op, M, /*IsSub=*/false);
    else if (N == "matlab.sub") Did = rewriteFiAddSub(Op, M, /*IsSub=*/true);
    else if (N == "matlab.matmul" || N == "matlab.emul")
      Did = rewriteFiMul(Op, M);
    else if (N == "matlab.neg") Did = rewriteFiNeg(Op);
    Changed |= Did;
  }
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
