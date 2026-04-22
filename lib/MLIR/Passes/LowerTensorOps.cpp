// Lower every tensor-producing / tensor-consuming `matlab.*` op to a call
// against the matrix runtime (`matlab_zeros`, `matlab_add_mm`,
// `matlab_disp_mat`, and friends). After this pass runs, every SSA value
// that used to have a `tensor<…xf64>` type is represented as a `!llvm.ptr`
// pointing to a heap-allocated `matlab_mat` descriptor.
//
// Ordering in the driver: runs after `LowerUserCalls` (so function
// signatures have been refined) and before `LowerIO` / `LowerScalarSlots`.
//
// Scope:
//   - `matlab.call_builtin @{zeros,ones,eye,magic,rand,randn,sum,
//                            transpose,ctranspose,diag,reshape,repmat,
//                            exp,log,sin,cos,tan,sqrt,abs}`
//   - `matlab.concat_row` / `matlab.concat_col` of f64 scalars (literal
//     matrix materialization via `matlab_mat_from_buf`).
//   - `matlab.{transpose,ctranspose,neg,add,sub,emul,ediv,epow}` with
//     tensor operand/result types.
//   - `matlab.call_builtin @disp` with a tensor-typed argument routes to
//     `matlab_disp_mat`. Scalar/string disp is still handled by
//     `LowerIO`.
//   - `matlab.alloc` with tensor result / `matlab.load` / `matlab.store`
//     on such slots get rewritten to `llvm.alloca` of `!llvm.ptr` + plain
//     `llvm.load`/`llvm.store` of pointers.
//
// Anything we don't recognize is left alone — the conversion pipeline
// will surface it cleanly if it reaches translation.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace matlab {
namespace mlirgen {

namespace {
using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

bool isTensorLike(Type T) {
  return mlir::isa<RankedTensorType, UnrankedTensorType>(T);
}

LLVM::LLVMFuncOp getOrInsertRTDecl(OpBuilder &B, ModuleOp M, StringRef Name,
                                   Type Result, ArrayRef<Type> Args) {
  if (auto Existing = M.lookupSymbol<LLVM::LLVMFuncOp>(Name)) return Existing;
  OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(M.getBody());
  auto Ty = LLVM::LLVMFunctionType::get(Result, Args);
  auto F = LLVM::LLVMFuncOp::create(B, M.getLoc(), Name, Ty);
  F.setLinkage(LLVM::Linkage::External);
  return F;
}

//===----------------------------------------------------------------------===//
// Per-op rewrites
//===----------------------------------------------------------------------===//

class TensorLowering {
public:
  TensorLowering(ModuleOp Mod) : Mod(Mod), B(Mod.getContext()) {
    Ctx = Mod.getContext();
    F64 = Float64Type::get(Ctx);
    I64 = IntegerType::get(Ctx, 64);
    PtrTy = LLVM::LLVMPointerType::get(Ctx);
    VoidTy = LLVM::LLVMVoidType::get(Ctx);
  }

  bool run();

private:
  ModuleOp Mod;
  MLIRContext *Ctx;
  OpBuilder B;
  Type F64, I64, PtrTy, VoidTy;

  LLVM::LLVMFuncOp rt(StringRef Name, Type Result, ArrayRef<Type> Args) {
    return getOrInsertRTDecl(B, Mod, Name, Result, Args);
  }

  // --- Slot retyping -----------------------------------------------------
  // Convert matlab.alloc with tensor result to llvm.alloca of !llvm.ptr.
  // Rewrite every matlab.load / matlab.store against it.
  bool retypeMatrixSlots();

  // --- Literal materialization ------------------------------------------
  // concat_row/concat_col whose operands are (eventually) f64 scalars
  // materialize the literal matrix via matlab_mat_from_buf.
  bool rewriteLiterals();

  // --- Builtin calls -----------------------------------------------------
  bool rewriteBuiltinCalls();

  // --- Binary element-wise -----------------------------------------------
  bool rewriteBinaryOps();

  // --- Postfix unary (transpose / ctranspose as ops) ---------------------
  bool rewritePostfix();

  // --- disp(matrix) -----------------------------------------------------
  bool rewriteDispMatrix();

  // --- Unary neg on tensor ----------------------------------------------
  bool rewriteUnaryNeg();

  // --- matlab.range -----------------------------------------------------
  // Lowers a:b or a:step:b to a matlab_range runtime call returning a ptr.
  bool rewriteRange();

  // --- matlab.subscript -------------------------------------------------
  // Scalar subscripting of a matrix pointer: A(i,j), A(i).
  bool rewriteSubscript();

  // Try to gather a contiguous row-major element list from a
  // `matlab.concat_col(concat_row(...), concat_row(...), ...)` chain.
  // Returns (rows, cols, elements) if all leaves are f64 values.
  bool gatherLiteralElements(Operation *ColOrRow, int64_t &Rows, int64_t &Cols,
                             SmallVectorImpl<Value> &Elts);

  // Materialize a matrix from a row-major value list: alloca a double buffer,
  // store each value, call matlab_mat_from_buf.
  Value materializeMat(Location Loc, int64_t Rows, int64_t Cols,
                       ArrayRef<Value> Elts);
};

bool TensorLowering::retypeMatrixSlots() {
  SmallVector<Operation *> Allocs;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.alloc") && Op->getNumResults() == 1 &&
        isTensorLike(Op->getResult(0).getType()))
      Allocs.push_back(Op);
  });

  bool Changed = false;
  for (Operation *Alloc : Allocs) {
    B.setInsertionPoint(Alloc);
    Value One = LLVM::ConstantOp::create(
        B, Alloc->getLoc(), I64, B.getI64IntegerAttr(1));
    // Slot element type is `ptr` (we store a matlab_mat pointer in the slot).
    Value NewSlot = LLVM::AllocaOp::create(B, Alloc->getLoc(), PtrTy, PtrTy,
                                            One, /*alignment=*/0);

    // Rewrite users.
    SmallVector<Operation *> ToErase;
    for (OpOperand &Use : Alloc->getResult(0).getUses()) {
      Operation *U = Use.getOwner();
      if (isMatlabOp(U, "matlab.load") && U->getNumOperands() == 1) {
        B.setInsertionPoint(U);
        Value Val = LLVM::LoadOp::create(B, U->getLoc(), PtrTy, NewSlot);
        U->getResult(0).replaceAllUsesWith(Val);
        ToErase.push_back(U);
      } else if (isMatlabOp(U, "matlab.store") && U->getNumOperands() == 2 &&
                 U->getOperand(1) == Alloc->getResult(0)) {
        Value V = U->getOperand(0);
        // The stored value may still be tensor-typed at this point (another
        // producer hasn't been rewritten yet); leave a placeholder that
        // later iteration will clean up. Skip store if its value isn't ptr.
        if (V.getType() != PtrTy) continue;
        B.setInsertionPoint(U);
        LLVM::StoreOp::create(B, U->getLoc(), V, NewSlot);
        ToErase.push_back(U);
      }
    }
    for (Operation *U : ToErase) U->erase();
    // If any uses remain (unrewritten stores with tensor-typed values),
    // leave the alloc in place for another iteration.
    if (Alloc->getResult(0).use_empty()) {
      Alloc->erase();
      Changed = true;
    }
  }
  return Changed;
}

bool TensorLowering::gatherLiteralElements(Operation *Root, int64_t &Rows,
                                            int64_t &Cols,
                                            SmallVectorImpl<Value> &Elts) {
  if (isMatlabOp(Root, "matlab.concat_row")) {
    // 1×N literal: all operands must be f64 scalars.
    Rows = 1;
    Cols = (int64_t)Root->getNumOperands();
    for (Value V : Root->getOperands()) {
      if (V.getType() != F64) return false;
      Elts.push_back(V);
    }
    return true;
  }
  if (isMatlabOp(Root, "matlab.concat_col")) {
    // M×N literal: each operand is a concat_row of N f64 scalars.
    Rows = (int64_t)Root->getNumOperands();
    Cols = -1;
    for (Value RowV : Root->getOperands()) {
      Operation *Row = RowV.getDefiningOp();
      if (!isMatlabOp(Row, "matlab.concat_row")) return false;
      int64_t RowCols = (int64_t)Row->getNumOperands();
      if (Cols == -1) Cols = RowCols;
      else if (RowCols != Cols) return false;
      for (Value V : Row->getOperands()) {
        if (V.getType() != F64) return false;
        Elts.push_back(V);
      }
    }
    return Cols >= 0;
  }
  return false;
}

Value TensorLowering::materializeMat(Location Loc, int64_t Rows, int64_t Cols,
                                      ArrayRef<Value> Elts) {
  Value One = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
  auto ArrayTy = LLVM::LLVMArrayType::get(
      F64, static_cast<unsigned>(Rows * Cols));
  Value BufPtr = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrayTy, One,
                                         /*alignment=*/0);
  for (int64_t k = 0; k < (int64_t)Elts.size(); ++k) {
    Value Idx = LLVM::ConstantOp::create(B, Loc, I64,
                                          B.getI64IntegerAttr(k));
    Value ElemPtr = LLVM::GEPOp::create(B, Loc, PtrTy, F64, BufPtr,
                                         ValueRange{Idx});
    LLVM::StoreOp::create(B, Loc, Elts[k], ElemPtr);
  }
  auto Fn = rt("matlab_mat_from_buf", PtrTy, {PtrTy, F64, F64});
  Value MVal = LLVM::ConstantOp::create(
      B, Loc, F64, B.getF64FloatAttr((double)Rows));
  Value NVal = LLVM::ConstantOp::create(
      B, Loc, F64, B.getF64FloatAttr((double)Cols));
  auto Call = LLVM::CallOp::create(B, Loc, Fn,
                                    ValueRange{BufPtr, MVal, NVal});
  return Call.getResult();
}

bool TensorLowering::rewriteLiterals() {
  SmallVector<Operation *> Roots;
  Mod.walk([&](Operation *Op) {
    // Only rewrite "outermost" concat ops whose result flows to a non-concat
    // user. For nested concat_col(concat_row(...), ...) we rewrite the col.
    if (!isMatlabOp(Op, "matlab.concat_row") &&
        !isMatlabOp(Op, "matlab.concat_col")) return;
    if (!isTensorLike(Op->getResult(0).getType())) return;
    // If every user is a concat_col/concat_row that will rewrite it as part
    // of their own gather, skip — we only want to rewrite at the root.
    for (OpOperand &Use : Op->getResult(0).getUses()) {
      Operation *U = Use.getOwner();
      if (isMatlabOp(U, "matlab.concat_col")) return;
    }
    Roots.push_back(Op);
  });

  bool Changed = false;
  for (Operation *Op : Roots) {
    int64_t Rows = 0, Cols = 0;
    SmallVector<Value, 16> Elts;
    if (!gatherLiteralElements(Op, Rows, Cols, Elts)) continue;
    B.setInsertionPoint(Op);
    Value M = materializeMat(Op->getLoc(), Rows, Cols, Elts);
    Op->getResult(0).replaceAllUsesWith(M);
    Op->erase();
    Changed = true;
  }
  // Sweep orphaned concat_row ops that fed into rewritten concat_cols.
  SmallVector<Operation *> Dead;
  Mod.walk([&](Operation *Op) {
    if ((isMatlabOp(Op, "matlab.concat_row") ||
         isMatlabOp(Op, "matlab.concat_col")) &&
        Op->use_empty())
      Dead.push_back(Op);
  });
  for (Operation *O : Dead) O->erase();
  if (!Dead.empty()) Changed = true;
  return Changed;
}

//===----------------------------------------------------------------------===//
// Builtin call dispatch
//===----------------------------------------------------------------------===//

struct BuiltinRewrite {
  StringRef RTName;    // runtime symbol to call
  Type ResultTy;        // result type after rewrite
  ArrayRef<Type> ArgTy; // expected arg types; parallel to the call operands
};

bool TensorLowering::rewriteBuiltinCalls() {
  SmallVector<Operation *> Calls;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call_builtin")) Calls.push_back(Op);
  });

  bool Changed = false;
  for (Operation *Call : Calls) {
    auto CA = Call->getAttrOfType<StringAttr>("callee");
    if (!CA) continue;
    StringRef Name = CA.getValue();

    // Table of simple 1- or 2-arg builtins returning either a matrix ptr
    // or an f64 scalar. The call is accepted only if operand types match.
    struct Spec {
      StringRef MLName;
      StringRef RTName;
      // 0 => f64 result, 1 => ptr result
      int ResultKind;
      // Arg kinds: 'f' = f64, 'p' = ptr (matrix)
      StringRef ArgKinds;
    };
    static const Spec Table[] = {
      {"zeros",      "matlab_zeros",      1, "ff"},
      {"ones",       "matlab_ones",       1, "ff"},
      {"eye",        "matlab_eye",        1, "ff"},
      {"magic",      "matlab_magic",      1, "f"},
      {"rand",       "matlab_rand",       1, "ff"},
      {"randn",      "matlab_randn",      1, "ff"},
      {"sum",        "matlab_sum",        0, "p"},
      {"transpose",  "matlab_transpose",  1, "p"},
      {"ctranspose", "matlab_transpose",  1, "p"},
      {"diag",       "matlab_diag",       1, "p"},
      {"reshape",    "matlab_reshape",    1, "pff"},
      {"repmat",     "matlab_repmat",     1, "pff"},
      {"exp",        "matlab_exp_m",      1, "p"},
      {"log",        "matlab_log_m",      1, "p"},
      {"sin",        "matlab_sin_m",      1, "p"},
      {"cos",        "matlab_cos_m",      1, "p"},
      {"tan",        "matlab_tan_m",      1, "p"},
      {"sqrt",       "matlab_sqrt_m",     1, "p"},
      {"abs",        "matlab_abs_m",      1, "p"},
      {"inv",        "matlab_inv",        1, "p"},
      {"det",        "matlab_det",        0, "p"},
    };

    const Spec *S = nullptr;
    for (auto &E : Table) if (E.MLName == Name) { S = &E; break; }
    if (!S) {
      // Scalar variants of exp/log/sin/cos/tan/sqrt/abs when the arg is f64
      // already. Fall through to scalar-path below.
      static const llvm::StringMap<StringRef> Scalar = {
        {"exp", "matlab_exp_s"}, {"log", "matlab_log_s"},
        {"sin", "matlab_sin_s"}, {"cos", "matlab_cos_s"},
        {"tan", "matlab_tan_s"}, {"sqrt", "matlab_sqrt_s"},
        {"abs", "matlab_abs_s"},
      };
      auto It = Scalar.find(Name);
      if (It == Scalar.end()) continue;
      if (Call->getNumOperands() != 1) continue;
      if (Call->getOperand(0).getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt(It->second, F64, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    // Check argument count / types.
    if ((int)Call->getNumOperands() != (int)S->ArgKinds.size()) continue;
    SmallVector<Type, 3> ExpTys;
    bool OK = true;
    for (unsigned i = 0; i < S->ArgKinds.size(); ++i) {
      Type Exp = S->ArgKinds[i] == 'f' ? F64 : PtrTy;
      ExpTys.push_back(Exp);
      Type Got = Call->getOperand(i).getType();
      // Accept tensor-typed args where we expect ptr (we'll convert via a
      // subsequent retype — but only if the value is actually a ptr at
      // runtime). We'll be strict and require ptr now; tensor-typed inputs
      // come from allocs that our slot-retype handled, so by the time we
      // run this they should already be ptr.
      if (Exp == F64 && Got != F64) { OK = false; break; }
      if (Exp == PtrTy && Got != PtrTy) { OK = false; break; }
    }
    if (!OK) continue;

    Type ResTy = S->ResultKind == 0 ? F64 : PtrTy;
    B.setInsertionPoint(Call);
    auto Fn = rt(S->RTName, ResTy, ExpTys);
    auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                    Call->getOperands());
    Call->getResult(0).replaceAllUsesWith(NC.getResult());
    Call->erase();
    Changed = true;
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// Binary ops with tensor arguments.
//
//   matlab.{add,sub,emul,ediv,epow}  — always element-wise (mm/ms/sm).
//   matlab.matmul  — matrix×matrix => matlab_matmul_mm (pure-C naive loop).
//                   scalar×matrix / matrix×scalar => element-wise emul.
//   matlab.matdiv  — A/B (mm) => matlab_mrdivide_mm (pure-C LU solve).
//                   A/s, s/A => element-wise ediv.
//   matlab.matldiv — A\B (mm) => matlab_mldivide_mm (pure-C LU solve).
//                   scalar mixes are rare in user code; we leave them
//                   untouched and the conversion pipeline will surface
//                   any that appear.
//===----------------------------------------------------------------------===//

bool TensorLowering::rewriteBinaryOps() {
  // Element-wise base names keyed by op name.
  struct ElemSpec { StringRef MLName; StringRef Base; };
  static const ElemSpec ElemSpecs[] = {
    {"matlab.add",  "add"},
    {"matlab.sub",  "sub"},
    {"matlab.emul", "emul"},
    {"matlab.ediv", "ediv"},
    {"matlab.epow", "epow"},
  };

  SmallVector<Operation *> Binaries;
  Mod.walk([&](Operation *Op) {
    if (Op->getNumOperands() != 2) return;
    StringRef N = Op->getName().getStringRef();
    if (N == "matlab.matmul" || N == "matlab.matdiv" ||
        N == "matlab.matldiv") {
      Binaries.push_back(Op); return;
    }
    for (auto &S : ElemSpecs)
      if (isMatlabOp(Op, S.MLName)) { Binaries.push_back(Op); return; }
  });

  bool Changed = false;
  for (Operation *Op : Binaries) {
    StringRef ML = Op->getName().getStringRef();
    Value A = Op->getOperand(0), BVal = Op->getOperand(1);
    Type AT = A.getType(), BT = BVal.getType();
    bool AP = AT == PtrTy, BP = BT == PtrTy;
    bool AF = AT == F64,    BF = BT == F64;
    if (!AP && !BP) continue; // scalar-only — LowerScalarsToArith handled it

    B.setInsertionPoint(Op);
    LLVM::LLVMFuncOp Fn;
    SmallVector<Value, 2> Args;

    auto emitElem = [&](StringRef Base) {
      if (AP && BP) {
        Fn = rt(("matlab_" + Base + "_mm").str(), PtrTy, {PtrTy, PtrTy});
        Args = {A, BVal};
      } else if (AP && BF) {
        Fn = rt(("matlab_" + Base + "_ms").str(), PtrTy, {PtrTy, F64});
        Args = {A, BVal};
      } else if (AF && BP) {
        Fn = rt(("matlab_" + Base + "_sm").str(), PtrTy, {F64, PtrTy});
        Args = {A, BVal};
      }
    };

    if (ML == "matlab.matmul") {
      if (AP && BP) {
        Fn = rt("matlab_matmul_mm", PtrTy, {PtrTy, PtrTy});
        Args = {A, BVal};
      } else {
        emitElem("emul"); // scalar * matrix broadcast
      }
    } else if (ML == "matlab.matdiv") {
      if (AP && BP) {
        Fn = rt("matlab_mrdivide_mm", PtrTy, {PtrTy, PtrTy});
        Args = {A, BVal};
      } else {
        emitElem("ediv");
      }
    } else if (ML == "matlab.matldiv") {
      if (AP && BP) {
        Fn = rt("matlab_mldivide_mm", PtrTy, {PtrTy, PtrTy});
        Args = {A, BVal};
      } else {
        continue; // uncommon; don't rewrite
      }
    } else {
      // Element-wise ops from ElemSpecs.
      StringRef Base;
      for (auto &E : ElemSpecs) if (E.MLName == ML) { Base = E.Base; break; }
      if (Base.empty()) continue;
      emitElem(Base);
    }

    if (!Fn) continue;
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn, Args);
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
    Op->erase();
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::rewritePostfix() {
  SmallVector<Operation *> Ops;
  Mod.walk([&](Operation *Op) {
    if (Op->getNumOperands() != 1) return;
    if (!isMatlabOp(Op, "matlab.transpose") &&
        !isMatlabOp(Op, "matlab.ctranspose")) return;
    if (Op->getOperand(0).getType() != PtrTy) return;
    Ops.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Ops) {
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_transpose", PtrTy, {PtrTy});
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Op->getOperand(0)});
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
    Op->erase();
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::rewriteUnaryNeg() {
  SmallVector<Operation *> Ops;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.neg") && Op->getNumOperands() == 1 &&
        Op->getOperand(0).getType() == PtrTy)
      Ops.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Ops) {
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_neg_m", PtrTy, {PtrTy});
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Op->getOperand(0)});
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
    Op->erase();
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::rewriteRange() {
  SmallVector<Operation *> Ranges;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.range") && !isTensorLike(Op->getResult(0).getType()))
      return; // already lowered
    if (isMatlabOp(Op, "matlab.range")) Ranges.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Ranges) {
    unsigned N = Op->getNumOperands();
    if (N != 2 && N != 3) continue;
    // Accept f64 operands; skip otherwise.
    for (unsigned i = 0; i < N; ++i)
      if (Op->getOperand(i).getType() != F64) return false;
    Value Start = Op->getOperand(0);
    Value Step, End;
    if (N == 2) {
      End = Op->getOperand(1);
      B.setInsertionPoint(Op);
      Step = LLVM::ConstantOp::create(B, Op->getLoc(), F64,
                                       B.getF64FloatAttr(1.0));
    } else {
      Step = Op->getOperand(1);
      End  = Op->getOperand(2);
    }
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_range", PtrTy, {F64, F64, F64});
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Start, Step, End});
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
    Op->erase();
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::rewriteSubscript() {
  SmallVector<Operation *> Subs;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.subscript")) Subs.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Subs) {
    // Require ptr as base and f64 indices; scalar result.
    unsigned N = Op->getNumOperands();
    if (N < 2 || N > 3) continue;
    if (Op->getOperand(0).getType() != PtrTy) continue;
    for (unsigned i = 1; i < N; ++i)
      if (Op->getOperand(i).getType() != F64) { N = 0; break; }
    if (N == 0) continue;
    if (Op->getNumResults() != 1 || Op->getResult(0).getType() != F64)
      continue;
    B.setInsertionPoint(Op);
    if (N == 3) {
      auto Fn = rt("matlab_subscript2_s", F64, {PtrTy, F64, F64});
      auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                      ValueRange{Op->getOperand(0),
                                                 Op->getOperand(1),
                                                 Op->getOperand(2)});
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
    } else {
      auto Fn = rt("matlab_subscript1_s", F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                      ValueRange{Op->getOperand(0),
                                                 Op->getOperand(1)});
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
    }
    Op->erase();
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::rewriteDispMatrix() {
  SmallVector<Operation *> Disps;
  Mod.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto CA = Op->getAttrOfType<StringAttr>("callee");
    if (!CA || CA.getValue() != "disp") return;
    if (Op->getNumOperands() != 1) return;
    if (Op->getOperand(0).getType() != PtrTy) return;
    Disps.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Disps) {
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_disp_mat", VoidTy, {PtrTy});
    LLVM::CallOp::create(B, Op->getLoc(), Fn,
                         ValueRange{Op->getOperand(0)});
    Op->erase();
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::run() {
  bool AnyChanged = false;
  // Iterate to a fixpoint. Bound it at a generous cap.
  for (int Iter = 0; Iter < 8; ++Iter) {
    bool Changed = false;
    Changed |= retypeMatrixSlots();
    Changed |= rewriteBuiltinCalls();
    Changed |= rewriteLiterals();
    Changed |= rewriteBinaryOps();
    Changed |= rewritePostfix();
    Changed |= rewriteUnaryNeg();
    Changed |= rewriteRange();
    Changed |= rewriteSubscript();
    Changed |= rewriteDispMatrix();
    if (!Changed) break;
    AnyChanged = true;
  }
  return AnyChanged;
}

} // namespace

bool runLowerTensorOps(ModuleOp M) {
  TensorLowering L(M);
  return L.run();
}

} // namespace mlirgen
} // namespace matlab
