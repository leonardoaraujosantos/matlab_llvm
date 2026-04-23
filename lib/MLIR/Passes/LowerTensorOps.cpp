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

  // --- Indexed store: A(rows, cols) = V, A(idx) = V. The lowering front-end
  // emits these as matlab.call_builtin @__subscript_store(A, i, j, ..., V).
  // Operand count and types drive dispatch to matlab_slice_store{1,2}[_scalar].
  bool rewriteSubscriptStore();

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
    /* Only retype when every user is load/store AND every store's value
     * is already ptr-typed. A partial retype (rewriting loads but not
     * stores, or vice versa) would split the slot between old matlab.alloc
     * and new llvm.alloca, desynchronizing loads from subsequent stores.
     * We'd rather wait another iteration until the literal rewrite and
     * builtin rewrite have produced ptr-typed values everywhere. */
    bool AllCanRetype = true;
    for (OpOperand &Use : Alloc->getResult(0).getUses()) {
      Operation *U = Use.getOwner();
      if (isMatlabOp(U, "matlab.load") && U->getNumOperands() == 1) continue;
      if (isMatlabOp(U, "matlab.store") && U->getNumOperands() == 2 &&
          U->getOperand(1) == Alloc->getResult(0)) {
        if (U->getOperand(0).getType() != PtrTy) {
          AllCanRetype = false; break;
        }
        continue;
      }
      AllCanRetype = false; break;
    }
    if (!AllCanRetype) continue;

    B.setInsertionPoint(Alloc);
    Value One = LLVM::ConstantOp::create(
        B, Alloc->getLoc(), I64, B.getI64IntegerAttr(1));
    Value NewSlot = LLVM::AllocaOp::create(B, Alloc->getLoc(), PtrTy, PtrTy,
                                            One, /*alignment=*/0);

    SmallVector<Operation *> ToErase;
    for (OpOperand &Use : Alloc->getResult(0).getUses()) {
      Operation *U = Use.getOwner();
      if (isMatlabOp(U, "matlab.load")) {
        B.setInsertionPoint(U);
        Value Val = LLVM::LoadOp::create(B, U->getLoc(), PtrTy, NewSlot);
        U->getResult(0).replaceAllUsesWith(Val);
        ToErase.push_back(U);
      } else if (isMatlabOp(U, "matlab.store")) {
        B.setInsertionPoint(U);
        LLVM::StoreOp::create(B, U->getLoc(), U->getOperand(0), NewSlot);
        ToErase.push_back(U);
      }
    }
    for (Operation *U : ToErase) U->erase();
    Alloc->erase();
    Changed = true;
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
    /* Zero-operand concat is an empty literal [] — route to the runtime
     * empty-matrix constructor so downstream disp/isempty/etc. see a
     * real 0×0 matlab_mat*. */
    if (Op->getNumOperands() == 0) {
      B.setInsertionPoint(Op);
      auto Fn = rt("matlab_empty_mat", PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn, ValueRange{});
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
      Op->erase();
      Changed = true;
      continue;
    }
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

    /* matlab_struct_new / set_f64 / set_mat / get_f64 / get_mat /
     * has_field. The frontend emits these as matlab.call_builtin with
     * a const_char for the field name. We materialise the name as an
     * llvm.mlir.global + addressof (ptr + length) and declare the
     * runtime function with the appropriate signature. */
    auto fieldNameAddr =
        [&](Value NameV, int64_t &LenOut) -> Value {
      Operation *Def = NameV.getDefiningOp();
      if (!isMatlabOp(Def, "matlab.const_char")) return Value{};
      auto VA = Def->getAttrOfType<StringAttr>("value");
      if (!VA) return Value{};
      StringRef Text = VA.getValue();
      LenOut = (int64_t)Text.size();
      /* Reuse an existing __matlab_str* global for the same text if
       * LowerIO already created one. */
      LLVM::GlobalOp Found;
      for (auto G : Mod.getOps<LLVM::GlobalOp>()) {
        if (!G.getConstant()) continue;
        auto Attr = mlir::dyn_cast_or_null<StringAttr>(G.getValueAttr());
        if (Attr && Attr.getValue() == Text) { Found = G; break; }
      }
      if (!Found) {
        OpBuilder::InsertionGuard G(B);
        B.setInsertionPointToStart(Mod.getBody());
        auto ArrayTy = LLVM::LLVMArrayType::get(
            IntegerType::get(Ctx, 8),
            static_cast<unsigned>(Text.size()));
        unsigned N = 0;
        std::string SymName;
        do {
          SymName = ("__matlab_str_f" + std::to_string(N++));
        } while (Mod.lookupSymbol(SymName));
        Found = LLVM::GlobalOp::create(
            B, Mod.getLoc(), ArrayTy, /*isConstant=*/true,
            LLVM::Linkage::Internal, SymName,
            StringAttr::get(Ctx, Text));
      }
      B.setInsertionPoint(Def);
      Value Addr = LLVM::AddressOfOp::create(
          B, Def->getLoc(), PtrTy, Found.getSymName());
      /* The const_char op's result is only consumed by the call site
       * we're about to rewrite. Replace uses with Addr so the op drops
       * to zero users after the call's erase; a later sweep deletes
       * the dead const_char. */
      Def->getResult(0).replaceAllUsesWith(Addr);
      return Addr;
    };

    /* Error-flag accessors: matlab_set_error / matlab_check_error /
     * matlab_clear_error. Used by try/catch and by the `error()`
     * builtin itself. */
    if (Name == "matlab_set_error" && Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_set_error", VoidTy, {});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_clear_error" && Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_clear_error", VoidTy, {});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_check_error" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_check_error",
                    IntegerType::get(Ctx, 32), {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* Rewrite @error(...) calls — whatever args the user passes we
     * ignore for v1 and just flip the error flag. */
    if (Name == "error") {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_set_error", VoidTy, {});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      /* Replace any result users with zero — error() has no real return. */
      for (auto R : Call->getResults())
        if (!R.use_empty()) {
          Value Z = LLVM::ConstantOp::create(
              B, Call->getLoc(), F64, B.getF64FloatAttr(0.0));
          R.replaceAllUsesWith(Z);
        }
      Call->erase();
      Changed = true;
      continue;
    }

    if (Name == "matlab_struct_new" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_struct_new", PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_struct_set_f64" ||
         Name == "matlab_struct_set_mat") &&
        Call->getNumOperands() == 3) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      Value Val = Call->getOperand(2);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = Name == "matlab_struct_set_mat";
      if (IsMat && Val.getType() != PtrTy) continue;
      if (!IsMat && Val.getType() != F64) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy, I64,
                                    IsMat ? (Type)PtrTy : (Type)F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Base, Ptr, LenV, Val});
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_struct_get_f64" ||
         Name == "matlab_struct_get_mat" ||
         Name == "matlab_struct_has_field") &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = Name == "matlab_struct_get_mat";
      Type Ret = IsMat ? (Type)PtrTy : (Type)F64;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, Ret, {PtrTy, PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Base, Ptr, LenV});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* Global / persistent scalar table accessors. The frontend emits
     * matlab.call_builtin @matlab_global_get_f64(i32) and
     * matlab.call_builtin @matlab_global_set_f64(i32, f64). */
    auto I32 = IntegerType::get(Ctx, 32);
    if (Name == "matlab_global_get_f64" &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == I32) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_global_get_f64", F64, {I32});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_global_set_f64" &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == I32 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_global_set_f64", VoidTy, {I32, F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1)});
      Call->erase();
      Changed = true;
      continue;
    }

    /* Multi-return dispatch (nargout > 1): today only eig has a two-
     * output variant [V, D] = eig(A). We emit two independent runtime
     * calls so each result can be consumed separately; the frontend
     * will have marked each LHS with a distinct result slot. */
    auto NA = Call->getAttrOfType<IntegerAttr>("nargout");
    if (NA && NA.getValue().getSExtValue() == 2 && Name == "eig" &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto FnV = rt("matlab_eig_V", PtrTy, {PtrTy});
      auto FnD = rt("matlab_eig_D", PtrTy, {PtrTy});
      auto CV = LLVM::CallOp::create(B, Call->getLoc(), FnV,
                                      ValueRange{Call->getOperand(0)});
      auto CD = LLVM::CallOp::create(B, Call->getLoc(), FnD,
                                      ValueRange{Call->getOperand(0)});
      Call->getResult(0).replaceAllUsesWith(CV.getResult());
      Call->getResult(1).replaceAllUsesWith(CD.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

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
      {"sum",        "matlab_sum",        1, "p"},
      {"prod",       "matlab_prod",       1, "p"},
      {"mean",       "matlab_mean",       1, "p"},
      {"min",        "matlab_min",        1, "p"},
      {"max",        "matlab_max",        1, "p"},
      {"size",       "matlab_size",       1, "p"},
      {"length",     "matlab_length",     0, "p"},
      {"numel",      "matlab_numel",      0, "p"},
      {"ndims",      "matlab_ndims",      0, "p"},
      {"isempty",    "matlab_isempty",    0, "p"},
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
      {"svd",        "matlab_svd",        1, "p"},
      {"eig",        "matlab_eig",        1, "p"},
      {"isequal",    "matlab_isequal",    0, "pp"},
      {"size",       "matlab_size_dim",   0, "pf"},   /* size(A, dim) */
      {"find",       "matlab_find",       1, "p"},
      {"matlab_empty_mat", "matlab_empty_mat", 1, ""},
    };

    // Pick the first entry with both a name match AND an arity match, so
    // overloaded builtins (e.g. size(A) vs size(A, dim)) route correctly.
    const Spec *S = nullptr;
    unsigned NOps = Call->getNumOperands();
    for (auto &E : Table) {
      if (E.MLName != Name) continue;
      if (E.ArgKinds.size() == NOps) { S = &E; break; }
    }
    /* Fallback: first name-match regardless of arity — keeps older single-
     * entry builtins working even when the call-site arity happens to
     * differ from our spec. The arity check inside the match logic below
     * will still reject mismatches safely. */
    if (!S) for (auto &E : Table) if (E.MLName == Name) { S = &E; break; }
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

  /* Sweep dead matlab.const_char ops whose only users were the struct
   * call sites we just rewrote. Run until fixed-point in case an
   * intermediate op we dropped frees up chains. */
  for (int R = 0; R < 4; ++R) {
    SmallVector<Operation *> Dead;
    Mod.walk([&](Operation *Op) {
      if (isMatlabOp(Op, "matlab.const_char") &&
          Op->getNumResults() == 1 &&
          Op->getResult(0).use_empty())
        Dead.push_back(Op);
    });
    if (Dead.empty()) break;
    for (Operation *Op : Dead) Op->erase();
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
    /* Comparisons: the runtime returns 0.0/1.0 matrices so logical
     * indexing A(A > 0) and similar patterns feed the same slice path. */
    {"matlab.gt",   "gt"},
    {"matlab.ge",   "ge"},
    {"matlab.lt",   "lt"},
    {"matlab.le",   "le"},
    {"matlab.eq",   "eq"},
    {"matlab.ne",   "ne"},
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

// Helper: does this value represent the colon sentinel (matlab.colon) or a
// null ptr already? Used while wrapping indices for the slice runtime.
static bool isColonSentinel(Value V) {
  Operation *D = V.getDefiningOp();
  if (!D) return false;
  if (D->getName().getStringRef() == "matlab.colon") return true;
  if (isa<LLVM::ZeroOp>(D)) return true;
  return false;
}

bool TensorLowering::rewriteSubscript() {
  // First, rewrite any matlab.end(base, dim) -> matlab_end_of_dim call and
  // matlab.colon -> llvm.mlir.zero. These need to happen before we try to
  // classify subscript operand types. (Bare-operand matlab.end without a
  // subscript context is left for later passes to reject cleanly.)
  SmallVector<Operation *> Ends, Colons;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.end") && Op->getNumOperands() == 2 &&
        Op->getOperand(0).getType() == PtrTy &&
        Op->getOperand(1).getType() == F64 &&
        Op->getNumResults() == 1 &&
        Op->getResult(0).getType() == F64)
      Ends.push_back(Op);
    else if (isMatlabOp(Op, "matlab.colon"))
      Colons.push_back(Op);
  });

  bool Changed = false;
  for (Operation *Op : Ends) {
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_end_of_dim", F64, {PtrTy, F64});
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Op->getOperand(0),
                                               Op->getOperand(1)});
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
    Op->erase();
    Changed = true;
  }
  for (Operation *Op : Colons) {
    B.setInsertionPoint(Op);
    Value Null = LLVM::ZeroOp::create(B, Op->getLoc(), PtrTy);
    Op->getResult(0).replaceAllUsesWith(Null);
    Op->erase();
    Changed = true;
  }

  // Now rewrite the subscript ops themselves.
  SmallVector<Operation *> Subs;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.subscript")) Subs.push_back(Op);
  });
  for (Operation *Op : Subs) {
    unsigned N = Op->getNumOperands();
    if (N < 2 || N > 3) continue;
    if (Op->getOperand(0).getType() != PtrTy) continue;

    // Classify each index.
    bool AllScalar = true;
    for (unsigned i = 1; i < N; ++i) {
      Type T = Op->getOperand(i).getType();
      if (T != F64) { AllScalar = false; break; }
    }

    B.setInsertionPoint(Op);
    Value Base = Op->getOperand(0);

    // All scalar + scalar f64 result => fast path, per-element access.
    if (AllScalar && Op->getNumResults() == 1 &&
        Op->getResult(0).getType() == F64) {
      if (N == 3) {
        auto Fn = rt("matlab_subscript2_s", F64, {PtrTy, F64, F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                        ValueRange{Base, Op->getOperand(1),
                                                   Op->getOperand(2)});
        Op->getResult(0).replaceAllUsesWith(NC.getResult());
      } else {
        auto Fn = rt("matlab_subscript1_s", F64, {PtrTy, F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                        ValueRange{Base, Op->getOperand(1)});
        Op->getResult(0).replaceAllUsesWith(NC.getResult());
      }
      Op->erase();
      Changed = true;
      continue;
    }

    // Slow path: any non-scalar index -> matlab_slice{1,2}.
    // Each index needs to reach the runtime as a ptr (row-vector of 1-based
    // indices) or null (colon). Convert:
    //   - f64 scalar  -> matlab_mat_from_scalar(x) : ptr
    //   - ptr         -> use as-is (range, index vector, or null sentinel)
    auto wrap = [&](Value V) -> Value {
      if (V.getType() == PtrTy) return V;
      if (V.getType() == F64) {
        auto Fn = rt("matlab_mat_from_scalar", PtrTy, {F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn, ValueRange{V});
        return NC.getResult();
      }
      /* Any other type means an operand we can't handle here — caller
       * will notice we didn't rewrite this subscript. */
      return Value{};
    };

    if (N == 3) {
      Value R = wrap(Op->getOperand(1));
      Value C = wrap(Op->getOperand(2));
      if (!R || !C) continue;
      auto Fn = rt("matlab_slice2", PtrTy, {PtrTy, PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                      ValueRange{Base, R, C});
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
    } else {
      Value I = wrap(Op->getOperand(1));
      if (!I) continue;
      auto Fn = rt("matlab_slice1", PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                      ValueRange{Base, I});
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
    }
    Op->erase();
    Changed = true;
  }
  (void)isColonSentinel;  // currently unused — kept for future expansion
  return Changed;
}

bool TensorLowering::rewriteSubscriptStore() {
  /* The frontend emits A(i, ..., k) = V as
   *   matlab.call_builtin @__subscript_store(%A, %i, ..., %k, %V)
   * Operand 0 is the base matrix; operands 1..N-1 are indices; operand N-1
   * is the RHS value. Dispatch to matlab_slice_store{1,2}[_scalar] based
   * on index count and RHS type. */
  SmallVector<Operation *> Stores;
  Mod.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto CA = Op->getAttrOfType<StringAttr>("callee");
    if (CA && CA.getValue() == "__subscript_store") Stores.push_back(Op);
  });

  bool Changed = false;
  for (Operation *Op : Stores) {
    unsigned N = Op->getNumOperands();
    /* Need base + at least one index + RHS => N >= 3. */
    if (N < 3) continue;
    Value Base = Op->getOperand(0);
    Value Rhs  = Op->getOperand(N - 1);
    unsigned NIdx = N - 2;
    /* Only 1-D and 2-D indexing wired up. */
    if (NIdx < 1 || NIdx > 2) continue;
    if (Base.getType() != PtrTy) continue;

    B.setInsertionPoint(Op);

    /* Wrap each index as ptr. */
    auto wrap = [&](Value V) -> Value {
      if (V.getType() == PtrTy) return V;
      if (V.getType() == F64) {
        auto Fn = rt("matlab_mat_from_scalar", PtrTy, {F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn, ValueRange{V});
        return NC.getResult();
      }
      return Value{};
    };

    Value I1 = wrap(Op->getOperand(1));
    if (!I1) continue;
    Value I2 = (NIdx == 2) ? wrap(Op->getOperand(2)) : Value{};
    if (NIdx == 2 && !I2) continue;

    bool RhsScalar = (Rhs.getType() == F64);
    bool RhsPtr    = (Rhs.getType() == PtrTy);
    if (!RhsScalar && !RhsPtr) continue;

    if (NIdx == 2) {
      if (RhsScalar) {
        auto Fn = rt("matlab_slice_store2_scalar", VoidTy,
                     {PtrTy, PtrTy, PtrTy, F64});
        LLVM::CallOp::create(B, Op->getLoc(), Fn,
                              ValueRange{Base, I1, I2, Rhs});
      } else {
        auto Fn = rt("matlab_slice_store2", VoidTy,
                     {PtrTy, PtrTy, PtrTy, PtrTy});
        LLVM::CallOp::create(B, Op->getLoc(), Fn,
                              ValueRange{Base, I1, I2, Rhs});
      }
    } else {
      if (RhsScalar) {
        auto Fn = rt("matlab_slice_store1_scalar", VoidTy,
                     {PtrTy, PtrTy, F64});
        LLVM::CallOp::create(B, Op->getLoc(), Fn,
                              ValueRange{Base, I1, Rhs});
      } else {
        auto Fn = rt("matlab_slice_store1", VoidTy,
                     {PtrTy, PtrTy, PtrTy});
        LLVM::CallOp::create(B, Op->getLoc(), Fn,
                              ValueRange{Base, I1, Rhs});
      }
    }

    /* The placeholder call produced a none-typed result that nobody reads,
     * but we still need to RAUW any (impossible) consumers before erasing. */
    for (auto R : Op->getResults()) {
      (void)R;
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
    Changed |= rewriteSubscriptStore();
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
