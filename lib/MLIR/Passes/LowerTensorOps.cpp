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
  bool Changed = false;
  SmallVector<Operation *> Allocs;
  Mod.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.alloc") || Op->getNumResults() != 1)
      return;
    Type T = Op->getResult(0).getType();
    /* Pick up tensor-typed slots, plus `none`-typed ones whose stores
     * are ptr-typed — cells / structs assigned via 'C = {...}' or
     * 's = matlab_struct_new()' land on a none slot that we want to
     * retype to ptr. */
    if (isTensorLike(T)) { Allocs.push_back(Op); return; }
    /* Ptr-typed allocs (struct slots from ensureStructSlot) also retype
     * to llvm.alloca so subsequent loads/stores go through llvm.* . */
    if (T == PtrTy) { Allocs.push_back(Op); return; }
    if (mlir::isa<NoneType>(T)) {
      bool AnyPtrStore = false;
      bool AnyF64Store = false;
      for (OpOperand &Use : Op->getResult(0).getUses()) {
        Operation *U = Use.getOwner();
        if (isMatlabOp(U, "matlab.store") && U->getNumOperands() == 2 &&
            U->getOperand(1) == Op->getResult(0)) {
          if (U->getOperand(0).getType() == PtrTy) AnyPtrStore = true;
          else if (mlir::isa<Float64Type>(U->getOperand(0).getType()))
            AnyF64Store = true;
        }
      }
      /* A none-typed slot whose only stores are f64 should be retyped
       * to f64 so the scalar-slot lowering can convert it to llvm.alloca
       * of f64. Only do this when no ptr store is also present — a
       * slot receiving both would be genuinely polymorphic and needs
       * the any-ptr fallback. */
      if (AnyPtrStore) Allocs.push_back(Op);
      else if (AnyF64Store) {
        /* Retype the alloc result and every matlab.load from it to
         * f64 in place. The stores are already f64. On the next pass
         * iteration, LowerScalarSlots will convert the whole slot to
         * an llvm.alloca of f64. */
        auto F64Ty = Float64Type::get(Ctx);
        Op->getResult(0).setType(F64Ty);
        for (OpOperand &Use : Op->getResult(0).getUses()) {
          Operation *U = Use.getOwner();
          if (isMatlabOp(U, "matlab.load") && U->getNumResults() == 1)
            U->getResult(0).setType(F64Ty);
        }
        Changed = true;
      }
    }
  });

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
    auto NewSlotOp = LLVM::AllocaOp::create(B, Alloc->getLoc(), PtrTy, PtrTy,
                                             One, /*alignment=*/0);
    // Propagate the matlab.alloc `name` attribute to the alloca so the
    // EmitC backend can emit readable variable names.
    if (auto NameAttr = Alloc->getAttrOfType<StringAttr>("name"))
      NewSlotOp->setAttr("matlab.name", NameAttr);
    Value NewSlot = NewSlotOp.getResult();

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

    /* rmfield(s, 'name') — route to matlab_struct_rmfield, returning
     * the same ptr so `s = rmfield(s, 'x')` keeps s working. */
    if (Name == "rmfield" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      Value NameV = Call->getOperand(1);
      int64_t Len = 0;
      auto fieldNameAddr0 = [&](Value N, int64_t &L) -> Value {
        Operation *Def = N.getDefiningOp();
        if (!isMatlabOp(Def, "matlab.const_char")) return Value{};
        auto VA = Def->getAttrOfType<StringAttr>("value");
        if (!VA) return Value{};
        StringRef Text = VA.getValue();
        L = (int64_t)Text.size();
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
          do { SymName = ("__matlab_str_f" + std::to_string(N++)); }
          while (Mod.lookupSymbol(SymName));
          Found = LLVM::GlobalOp::create(
              B, Mod.getLoc(), ArrayTy, /*isConstant=*/true,
              LLVM::Linkage::Internal, SymName,
              StringAttr::get(Ctx, Text));
        }
        B.setInsertionPoint(Def);
        Value Addr = LLVM::AddressOfOp::create(
            B, Def->getLoc(), PtrTy, Found.getSymName());
        Def->getResult(0).replaceAllUsesWith(Addr);
        return Addr;
      };
      Value Ptr = fieldNameAddr0(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_struct_rmfield", PtrTy, {PtrTy, PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Ptr, LenV});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* isfield(s, 'name') — route to matlab_struct_has_field. */
    if (Name == "isfield" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      Value NameV = Call->getOperand(1);
      int64_t Len = 0;
      auto fieldNameAddr0 = [&](Value N, int64_t &L) -> Value {
        Operation *Def = N.getDefiningOp();
        if (!isMatlabOp(Def, "matlab.const_char")) return Value{};
        auto VA = Def->getAttrOfType<StringAttr>("value");
        if (!VA) return Value{};
        StringRef Text = VA.getValue();
        L = (int64_t)Text.size();
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
          do { SymName = ("__matlab_str_f" + std::to_string(N++)); }
          while (Mod.lookupSymbol(SymName));
          Found = LLVM::GlobalOp::create(
              B, Mod.getLoc(), ArrayTy, /*isConstant=*/true,
              LLVM::Linkage::Internal, SymName,
              StringAttr::get(Ctx, Text));
        }
        B.setInsertionPoint(Def);
        Value Addr = LLVM::AddressOfOp::create(
            B, Def->getLoc(), PtrTy, Found.getSymName());
        Def->getResult(0).replaceAllUsesWith(Addr);
        return Addr;
      };
      Value Ptr = fieldNameAddr0(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_struct_has_field", F64, {PtrTy, PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Ptr, LenV});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

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
    /* Rewrite @error(...) calls. If the first arg is a const_char we
     * route through matlab_set_error_msg(ptr, len) so 'catch ME;
     * disp(ME.message)' gets back the user's text. Otherwise fall
     * back to matlab_set_error with no message. Extra args are
     * ignored in v1 (no printf-style formatting yet). */
    if (Name == "error") {
      B.setInsertionPoint(Call);
      Value MsgPtr;
      int64_t MsgLen = 0;
      if (Call->getNumOperands() >= 1) {
        Operation *Def = Call->getOperand(0).getDefiningOp();
        if (isMatlabOp(Def, "matlab.const_char")) {
          auto VA = Def->getAttrOfType<StringAttr>("value");
          if (VA) {
            StringRef Text = VA.getValue();
            MsgLen = (int64_t)Text.size();
            LLVM::GlobalOp Found;
            for (auto G : Mod.getOps<LLVM::GlobalOp>()) {
              if (!G.getConstant()) continue;
              auto Attr =
                  mlir::dyn_cast_or_null<StringAttr>(G.getValueAttr());
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
              do { SymName = ("__matlab_err_msg" + std::to_string(N++)); }
              while (Mod.lookupSymbol(SymName));
              Found = LLVM::GlobalOp::create(
                  B, Mod.getLoc(), ArrayTy, /*isConstant=*/true,
                  LLVM::Linkage::Internal, SymName,
                  StringAttr::get(Ctx, Text));
            }
            B.setInsertionPoint(Call);
            MsgPtr = LLVM::AddressOfOp::create(
                B, Call->getLoc(), PtrTy, Found.getSymName());
            Def->getResult(0).replaceAllUsesWith(MsgPtr);
          }
        }
      }
      if (MsgPtr) {
        auto Fn = rt("matlab_set_error_msg", VoidTy, {PtrTy, I64});
        Value LenV = LLVM::ConstantOp::create(
            B, Call->getLoc(), I64, B.getI64IntegerAttr(MsgLen));
        LLVM::CallOp::create(B, Call->getLoc(), Fn,
                              ValueRange{MsgPtr, LenV});
      } else {
        auto Fn = rt("matlab_set_error", VoidTy, {});
        LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      }
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

    /* Real string runtime. matlab_string_from_literal takes a
     * const_char arg; the others take matlab_string* pointers. We
     * materialise the literal's bytes as an llvm.mlir.global + len,
     * same pattern as the struct / cell field names. */
    if (Name == "matlab_string_from_literal" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1) {
      Value Ch = Call->getOperand(0);
      Operation *Def = Ch.getDefiningOp();
      if (!isMatlabOp(Def, "matlab.const_char")) continue;
      auto VA = Def->getAttrOfType<StringAttr>("value");
      if (!VA) continue;
      StringRef Text = VA.getValue();
      /* Reuse an existing __matlab_str* global or create one. */
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
        do { SymName = ("__matlab_str_s" + std::to_string(N++)); }
        while (Mod.lookupSymbol(SymName));
        Found = LLVM::GlobalOp::create(
            B, Mod.getLoc(), ArrayTy, /*isConstant=*/true,
            LLVM::Linkage::Internal, SymName,
            StringAttr::get(Ctx, Text));
      }
      B.setInsertionPoint(Call);
      Value Addr = LLVM::AddressOfOp::create(
          B, Call->getLoc(), PtrTy, Found.getSymName());
      Def->getResult(0).replaceAllUsesWith(Addr);
      int64_t Len = (int64_t)Text.size();
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_string_from_literal", PtrTy, {PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Addr, LenV});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_string_concat" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_string_concat", PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_string_disp" && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_string_disp", VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_string_len" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_string_len", F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* --- String-builtin dispatchers ------------------------------
     * All operate on matlab_string* values (the runtime wraps
     * "..." literals via matlab_string_from_literal). These are
     * the "frontend-called" builtin names (sprintf, upper, ...) —
     * distinct from the matlab_string_* internals above. */
    if ((Name == "upper" || Name == "lower" || Name == "strtrim") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      std::string Rn = "matlab_" + Name.str();
      B.setInsertionPoint(Call);
      auto Fn = rt(Rn, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "startsWith" || Name == "endsWith" ||
         Name == "contains") && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      std::string Rn = "matlab_" + Name.str();
      B.setInsertionPoint(Call);
      auto Fn = rt(Rn, F64, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != F64)
        Call->getResult(0).setType(F64);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "strcat" || Name == "strrep") &&
        Call->getNumResults() == 1 &&
        (Call->getNumOperands() == 2 || Call->getNumOperands() == 3)) {
      bool AllPtr = true;
      for (Value V : Call->getOperands())
        if (V.getType() != PtrTy) { AllPtr = false; break; }
      if (!AllPtr) continue;
      std::string Rn = "matlab_" + Name.str();
      SmallVector<Type, 4> Sig(Call->getNumOperands(), (Type)PtrTy);
      B.setInsertionPoint(Call);
      auto Fn = rt(Rn, PtrTy, Sig);
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "num2str" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_num2str", PtrTy, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* assert(cond) / assert(cond, msg). Void return — the frontend
     * drops any result. A false condition sets the error flag via
     * matlab_set_error_msg, so subsequent try/catch can pick it up.
     * Cond arrives as either f64 (e.g. `assert(v)` where v is a
     * scalar) or i1 (from a comparison like `assert(x == y)`); in
     * the i1 case we extend to f64 first. */
    if (Name == "assert" && Call->getNumOperands() >= 1) {
      auto I1 = IntegerType::get(Ctx, 1);
      Value Cond = Call->getOperand(0);
      if (Cond.getType() == F64 || Cond.getType() == I1) {
        B.setInsertionPoint(Call);
        if (Cond.getType() == I1) {
          Cond = arith::UIToFPOp::create(B, Call->getLoc(), F64, Cond);
        }
        if (Call->getNumOperands() == 1) {
          auto Fn = rt("matlab_assert", VoidTy, {F64});
          LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Cond});
          Call->erase();
          Changed = true;
          continue;
        }
        if (Call->getNumOperands() == 2 &&
            Call->getOperand(1).getType() == PtrTy) {
          auto Fn = rt("matlab_assert_msg", VoidTy, {F64, PtrTy});
          LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                ValueRange{Cond, Call->getOperand(1)});
          Call->erase();
          Changed = true;
          continue;
        }
      }
    }
    if (Name == "str2double" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_str2double", F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != F64)
        Call->getResult(0).setType(F64);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* sprintf(fmt)          -> matlab_sprintf_str
     * sprintf(fmt, v_f64)   -> matlab_sprintf_f64 */
    if (Name == "sprintf" && Call->getNumResults() == 1 &&
        Call->getNumOperands() >= 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      if (Call->getNumOperands() == 1) {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_sprintf_str", PtrTy, {PtrTy});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        if (Call->getResult(0).getType() != PtrTy)
          Call->getResult(0).setType(PtrTy);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      if (Call->getNumOperands() == 2 &&
          Call->getOperand(1).getType() == F64) {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_sprintf_f64", PtrTy, {PtrTy, F64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        if (Call->getResult(0).getType() != PtrTy)
          Call->getResult(0).setType(PtrTy);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* disp(ME.message) frontend-intercept routes here. */
    if (Name == "matlab_err_disp_message" && Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_err_disp_message", VoidTy, {});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->erase();
      Changed = true;
      continue;
    }

    /* Cell runtime. matlab_cell_new takes an f64 capacity hint and
     * returns ptr; set/get take (ptr, f64 index, value?) with f64 /
     * matrix-ptr value variants. Index is 1-based in the runtime. */
    if (Name == "matlab_cell_new" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_cell_new", PtrTy, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cell_set_f64" ||
         Name == "matlab_cell_set_mat") &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      bool IsMat = Name == "matlab_cell_set_mat";
      Value V = Call->getOperand(2);
      if (IsMat && V.getType() != PtrTy) continue;
      if (!IsMat && V.getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy, F64,
                                    IsMat ? (Type)PtrTy : (Type)F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1), V});
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cell_get_f64" ||
         Name == "matlab_cell_get_mat" ||
         Name == "matlab_cell_numel" ||
         Name == "matlab_iscell") &&
        Call->getNumResults() == 1) {
      B.setInsertionPoint(Call);
      Type Ret;
      SmallVector<Type, 2> Args;
      SmallVector<Value, 2> Ops;
      if (Name == "matlab_cell_get_mat") {
        if (Call->getNumOperands() != 2) continue;
        Ret = PtrTy;
        Args = {PtrTy, F64};
        Ops = {Call->getOperand(0), Call->getOperand(1)};
      } else if (Name == "matlab_cell_get_f64") {
        if (Call->getNumOperands() != 2) continue;
        Ret = F64;
        Args = {PtrTy, F64};
        Ops = {Call->getOperand(0), Call->getOperand(1)};
      } else {
        if (Call->getNumOperands() != 1 ||
            Call->getOperand(0).getType() != PtrTy) continue;
        Ret = F64;
        Args = {PtrTy};
        Ops = {Call->getOperand(0)};
      }
      auto Fn = rt(Name, Ret, Args);
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange(Ops));
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
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
         Name == "matlab_struct_get_child_struct" ||
         Name == "matlab_struct_has_field") &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsPtr = Name == "matlab_struct_get_mat" ||
                   Name == "matlab_struct_get_child_struct";
      Type Ret = IsPtr ? (Type)PtrTy : (Type)F64;
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

    /* REPL workspace accessors. Shape is the same as struct_* but
     * without a base ptr (the workspace is a singleton inside the
     * runtime). Used only when matlabc is invoked with -repl. */
    if ((Name == "matlab_ws_get_f64" || Name == "matlab_ws_get_mat") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1) {
      Value NameV = Call->getOperand(0);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = (Name == "matlab_ws_get_mat");
      Type Ret = IsMat ? (Type)PtrTy : (Type)F64;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, Ret, {PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Ptr, LenV});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_ws_set_f64" || Name == "matlab_ws_set_mat") &&
        Call->getNumOperands() == 2) {
      Value NameV = Call->getOperand(0);
      Value Val = Call->getOperand(1);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = (Name == "matlab_ws_set_mat");
      if (IsMat && Val.getType() != PtrTy) continue;
      if (!IsMat && Val.getType() != F64) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, VoidTy,
                   {PtrTy, I64, IsMat ? (Type)PtrTy : (Type)F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Ptr, LenV, Val});
      Call->erase();
      Changed = true;
      continue;
    }

    /* User-defined-class property accessors. Same shape as the struct
     * variants but the base is a matlab_obj* rather than matlab_struct*,
     * so the field name + length are materialised and passed identically;
     * the runtime delegates to the embedded struct table. */
    if ((Name == "matlab_obj_set_f64" || Name == "matlab_obj_set_mat") &&
        Call->getNumOperands() == 3) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      Value Val = Call->getOperand(2);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = Name == "matlab_obj_set_mat";
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
    if ((Name == "matlab_obj_get_f64" || Name == "matlab_obj_get_mat") &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsPtr = Name == "matlab_obj_get_mat";
      Type Ret = IsPtr ? (Type)PtrTy : (Type)F64;
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
    if (Name == "matlab_obj_new" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1) {
      Value Arg = Call->getOperand(0);
      auto I32 = IntegerType::get(Ctx, 32);
      if (Arg.getType() != I32) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_obj_new", PtrTy, {I32});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Arg});
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* File I/O. fopen takes two matlab_string* pointers (the frontend
     * wraps raw char/string literals for us); fclose / feof take an f64
     * file id; fgetl returns a matlab_string*. Sema leaves these
     * untyped, so we retype the call's result before RAUW to match the
     * runtime signature. */
    if (Name == "fopen" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_fopen", F64, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != F64)
        Call->getResult(0).setType(F64);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "fclose" || Name == "feof") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == F64) {
      llvm::StringRef Rn = (Name == "fclose") ? "matlab_fclose" : "matlab_feof";
      B.setInsertionPoint(Call);
      auto Fn = rt(Rn, F64, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != F64)
        Call->getResult(0).setType(F64);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "fgetl" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_fgetl", PtrTy, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* fread(fid, n) -> matlab_mat* (n-by-1). Binary reads: n doubles. */
    if (Name == "fread" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_fread", PtrTy, {F64, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* save(path, A) / load(path) — custom binary format, one matrix
     * per file. save takes a matlab_string path and a ptr matrix;
     * load takes a matlab_string path and returns a ptr matrix.
     * This is NOT MATLAB .mat-compatible — see runtime comments. */
    if (Name == "save" && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_save_mat", F64, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getNumResults() == 1) {
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
      }
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "load" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_load_mat", PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* fwrite(fid, A) — either matrix or scalar. Both variants return
     * the count of elements written as an f64. */
    if (Name == "fwrite" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == F64) {
      Type ArgT = Call->getOperand(1).getType();
      if (ArgT == PtrTy) {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_fwrite_mat", F64, {F64, PtrTy});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      if (ArgT == F64) {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_fwrite_f64", F64, {F64, F64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* 3-D array runtime: matlab_mat3 descriptor. The frontend emits
     * these directly on bindings tracked as 3-D (zeros/ones with 3
     * args). Each entry matches (ptr, ...) operand types. */
    if (Name == "matlab_subscript3_s" && Call->getNumOperands() == 4 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64 &&
        Call->getOperand(3).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_subscript3_s", F64, {PtrTy, F64, F64, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_subscript3_store" &&
        Call->getNumOperands() == 5 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64 &&
        Call->getOperand(3).getType() == F64 &&
        Call->getOperand(4).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_subscript3_store", VoidTy,
                   {PtrTy, F64, F64, F64, F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_size3_dim" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_size3_dim", F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_numel3" || Name == "matlab_ndims3") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
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

    /* Multi-return dispatch (nargout > 1). Each factorisation whose
     * MATLAB form returns multiple matrices (eig / qr / lu) is emitted
     * as two independent runtime calls sharing the input matrix; the
     * frontend will have marked each LHS with a distinct result slot.
     * This keeps the runtime ABI simple (one output per call) at the
     * cost of factoring the input twice; fine for the scripts the
     * compiler currently targets. */
    auto NA = Call->getAttrOfType<IntegerAttr>("nargout");
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      struct TwoRet { StringRef MLName, F0, F1; };
      static const TwoRet TwoReturns[] = {
        {"eig", "matlab_eig_V", "matlab_eig_D"},
        {"qr",  "matlab_qr_Q",  "matlab_qr_R"},
        {"lu",  "matlab_lu_L",  "matlab_lu_U"},
      };
      const TwoRet *T = nullptr;
      for (auto &E : TwoReturns)
        if (E.MLName == Name) { T = &E; break; }
      if (T) {
        B.setInsertionPoint(Call);
        auto F0 = rt(T->F0, PtrTy, {PtrTy});
        auto F1 = rt(T->F1, PtrTy, {PtrTy});
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), F0,
                                        ValueRange{Call->getOperand(0)});
        auto C1 = LLVM::CallOp::create(B, Call->getLoc(), F1,
                                        ValueRange{Call->getOperand(0)});
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->getResult(1).replaceAllUsesWith(C1.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
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
      {"zeros",      "matlab_zeros3",     1, "fff"},
      {"ones",       "matlab_ones",       1, "ff"},
      {"ones",       "matlab_ones3",      1, "fff"},
      {"eye",        "matlab_eye",        1, "ff"},
      {"magic",      "matlab_magic",      1, "f"},
      {"rand",       "matlab_rand",       1, "ff"},
      {"randn",      "matlab_randn",      1, "ff"},
      {"sum",        "matlab_sum",        1, "p"},
      {"sum",        "matlab_sum_dim",    1, "pf"},
      {"prod",       "matlab_prod",       1, "p"},
      {"prod",       "matlab_prod_dim",   1, "pf"},
      {"mean",       "matlab_mean",       1, "p"},
      {"mean",       "matlab_mean_dim",   1, "pf"},
      {"min",        "matlab_min",        1, "p"},
      {"max",        "matlab_max",        1, "p"},
      {"cumsum",     "matlab_cumsum",     1, "p"},
      {"cumsum",     "matlab_cumsum_dim", 1, "pf"},
      {"cumprod",    "matlab_cumprod",    1, "p"},
      {"cumprod",    "matlab_cumprod_dim",1, "pf"},
      {"sort",       "matlab_sort",       1, "p"},
      {"sortrows",   "matlab_sortrows",   1, "p"},
      {"unique",     "matlab_unique",     1, "p"},
      {"ismember",   "matlab_ismember",   1, "pp"},
      {"setdiff",    "matlab_setdiff",    1, "pp"},
      {"intersect",  "matlab_intersect",  1, "pp"},
      {"union",      "matlab_union",      1, "pp"},
      {"horzcat",    "matlab_horzcat",    1, "pp"},
      {"vertcat",    "matlab_vertcat",    1, "pp"},
      {"sub2ind",    "matlab_sub2ind",    0, "pff"},
      {"ind2sub",    "matlab_ind2sub",    1, "pf"},
      {"norm",       "matlab_norm",       0, "p"},
      {"trace",      "matlab_trace",      0, "p"},
      {"kron",       "matlab_kron",       1, "pp"},
      {"chol",       "matlab_chol",       1, "p"},
      {"pinv",       "matlab_pinv",       1, "p"},
      {"permute",    "matlab_permute",    1, "pp"},
      {"squeeze",    "matlab_squeeze",    1, "p"},
      {"flip",       "matlab_flip",       1, "p"},
      {"fliplr",     "matlab_fliplr",     1, "p"},
      {"flipud",     "matlab_flipud",     1, "p"},
      {"rot90",      "matlab_rot90",      1, "p"},
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
      {"asin",       "matlab_asin_m",     1, "p"},
      {"acos",       "matlab_acos_m",     1, "p"},
      {"atan",       "matlab_atan_m",     1, "p"},
      {"sinh",       "matlab_sinh_m",     1, "p"},
      {"cosh",       "matlab_cosh_m",     1, "p"},
      {"tanh",       "matlab_tanh_m",     1, "p"},
      {"log2",       "matlab_log2_m",     1, "p"},
      {"log10",      "matlab_log10_m",    1, "p"},
      {"sqrt",       "matlab_sqrt_m",     1, "p"},
      {"abs",        "matlab_abs_m",      1, "p"},
      {"sign",       "matlab_sign_m",     1, "p"},
      {"atan2",      "matlab_atan2_m",    1, "pp"},
      {"inv",        "matlab_inv",        1, "p"},
      {"det",        "matlab_det",        0, "p"},
      {"svd",        "matlab_svd",        1, "p"},
      {"eig",        "matlab_eig",        1, "p"},
      {"isequal",    "matlab_isequal",    0, "pp"},
      {"size",       "matlab_size_dim",   0, "pf"},   /* size(A, dim) */
      {"find",       "matlab_find",       1, "p"},
      {"matlab_empty_mat", "matlab_empty_mat", 1, ""},
    };

    // Pick the first entry with name + arity + TYPE match so overloaded
    // builtins (e.g. size(A) vs size(A, dim); sin(matrix) vs sin(scalar))
    // route correctly. If no Table entry fits the call-site's operand
    // types, S stays null and the code falls through to the scalar map
    // below (where sin(f64) -> matlab_sin_s etc. live).
    //
    // "Type match" accepts tensor-typed operands for ptr slots too:
    // on early pipeline iterations a matrix-producing literal or
    // builtin still has a tensor type, and we want to match and then
    // defer until retypeMatrixSlots converts the tensor to ptr on a
    // later iteration. Without this, sum(eye(4)) — where both ops
    // are inline — would never rewrite.
    const Spec *S = nullptr;
    unsigned NOps = Call->getNumOperands();
    auto argTypesMatch = [&](const Spec &E) -> bool {
      if (E.ArgKinds.size() != NOps) return false;
      for (unsigned i = 0; i < NOps; ++i) {
        char Kind = E.ArgKinds[i];
        Type Got = Call->getOperand(i).getType();
        if (Kind == 'f') {
          if (Got != F64) return false;
        } else { /* 'p' */
          if (Got != PtrTy && !isTensorLike(Got)) return false;
        }
      }
      return true;
    };
    for (auto &E : Table)
      if (E.MLName == Name && argTypesMatch(E)) { S = &E; break; }
    if (!S) {
      // Scalar variants of exp/log/sin/cos/tan/sqrt/abs when the arg is f64
      // already. Fall through to scalar-path below.
      static const llvm::StringMap<StringRef> Scalar = {
        {"exp", "matlab_exp_s"}, {"log", "matlab_log_s"},
        {"sin", "matlab_sin_s"}, {"cos", "matlab_cos_s"},
        {"tan", "matlab_tan_s"}, {"sqrt", "matlab_sqrt_s"},
        {"abs", "matlab_abs_s"},
        /* Trig/exp tail — scalar variants mirror their matrix forms
         * in the table above. */
        {"asin", "matlab_asin_s"}, {"acos", "matlab_acos_s"},
        {"atan", "matlab_atan_s"},
        {"sinh", "matlab_sinh_s"}, {"cosh", "matlab_cosh_s"},
        {"tanh", "matlab_tanh_s"},
        {"log2", "matlab_log2_s"}, {"log10", "matlab_log10_s"},
        {"sign", "matlab_sign_s"},
        /* Integer / type cast builtins — runtime is still f64, but
         * these truncate + saturate to the target dtype's range so
         * downstream arithmetic sees the value MATLAB would. */
        {"int8",   "matlab_int8_s"},   {"int16",  "matlab_int16_s"},
        {"int32",  "matlab_int32_s"},  {"int64",  "matlab_int64_s"},
        {"uint8",  "matlab_uint8_s"},  {"uint16", "matlab_uint16_s"},
        {"uint32", "matlab_uint32_s"}, {"uint64", "matlab_uint64_s"},
        {"double", "matlab_double_s"}, {"single", "matlab_single_s"},
        {"logical", "matlab_logical_s"},
      };
      /* Two-argument scalar: atan2(y, x). */
      if (Name == "atan2" && Call->getNumOperands() == 2 &&
          Call->getOperand(0).getType() == F64 &&
          Call->getOperand(1).getType() == F64) {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_atan2_s", F64, {F64, F64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      auto It = Scalar.find(Name);
      if (It == Scalar.end()) continue;
      if (Call->getNumOperands() != 1) continue;
      if (Call->getOperand(0).getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt(It->second, F64, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      /* Sema may have typed the call's result as a specific integer
       * width (si32 for int32, ui8 for uint8, etc.) while the runtime
       * returns f64. Since we stay f64 internally, retype the call's
       * result to f64 before replacing uses so downstream arith ops
       * don't see a type mismatch. */
      if (Call->getResult(0).getType() != F64)
        Call->getResult(0).setType(F64);
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
