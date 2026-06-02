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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "llvm/ADT/StringSet.h"

namespace matlab {
namespace mlirgen {

namespace {
using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

// Copy a `matlab.name` hint from an old op being rewritten onto its
// replacement, so the EmitPython backend can keep surfacing the
// user-source variable name (`x = np.linalg.solve(A, b)`) instead of
// falling back to a fresh `vN` id. No-op when `Old` carries no name or
// `New` already has one.
static void carryName(Operation *Old, Operation *New) {
  if (!Old || !New) return;
  auto NA = Old->getAttrOfType<StringAttr>("matlab.name");
  if (!NA) return;
  if (New->hasAttr("matlab.name")) return;
  New->setAttr("matlab.name", NA);
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

  // --- Complex literal (2i / 3j) ---------------------------------------
  // matlab.const_complex with a "value" string attribute lowers to a
  // matlab_complex_scalar(0.0, imag) call returning a ptr to a 1x1
  // matlab_mat_c.
  bool rewriteComplexLiterals();

  // --- Builtin calls -----------------------------------------------------
  bool rewriteBuiltinCalls();

  // --- Binary element-wise -----------------------------------------------
  bool rewriteBinaryOps();

  // --- Postfix unary (transpose / ctranspose as ops) ---------------------
  bool rewritePostfix();

  // --- disp(matrix) -----------------------------------------------------
  bool rewriteDispMatrix();

  // --- matlab_mat_truth(ptr) -> i8 — DAP/REPL cond coercion -------------
  bool rewriteMatTruth();

  // --- Coerce ptr-typed scf.if / scf.condition operands back to i1 ------
  // After rewriteBinaryOps replaces a matlab.lt/gt/etc. (originally i1)
  // with matlab_*_mm/ms/sm (returning ptr), any scf.if condition or
  // scf.condition first-operand whose type drifted from i1 to ptr would
  // fail SCF verification. Insert a matlab_mat_truth call + cmpi ne 0
  // at each such use site.
  bool fixupCondOperands();
  // #77: a scalar float arith op (arith.addf/subf/mulf/divf/negf, f64
  // result) can be left with a !llvm.ptr operand when an earlier scalar
  // lowering created it while the operand was f64 and a later pass retyped
  // that operand to a boxed-scalar matlab_mat* (e.g. `abs(y - ref)` where y
  // is a workspace scalar read as a ptr). Such an operand is always a 1x1
  // box, so unbox it to f64 via matlab_mat_to_scalar.
  bool unboxScalarArithOperands();

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

bool TensorLowering::rewriteComplexLiterals() {
  SmallVector<Operation *> Lits;
  Mod.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.const_complex")) Lits.push_back(Op);
  });
  bool Changed = false;
  for (Operation *Op : Lits) {
    auto VA = Op->getAttrOfType<StringAttr>("value");
    if (!VA) continue;
    /* The attribute value is the MATLAB source text — e.g. "2i", "3.5j",
     * "1.25e-3i". Strip the trailing i/j and parse the leading number
     * as the imaginary magnitude; real part is always 0 at a literal. */
    StringRef Txt = VA.getValue();
    if (Txt.empty()) continue;
    char Suffix = Txt.back();
    if (Suffix != 'i' && Suffix != 'j' && Suffix != 'I' && Suffix != 'J')
      continue;
    double Imag = 0.0;
    if (Txt.drop_back(1).getAsDouble(Imag)) continue;  /* couldn't parse */
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_complex_scalar", PtrTy, {F64, F64});
    auto Zero = LLVM::ConstantOp::create(B, Op->getLoc(), F64,
                                          B.getF64FloatAttr(0.0));
    auto Im = LLVM::ConstantOp::create(B, Op->getLoc(), F64,
                                        B.getF64FloatAttr(Imag));
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Zero, Im});
    if (Op->getNumResults() == 1 &&
        Op->getResult(0).getType() != PtrTy)
      Op->getResult(0).setType(PtrTy);
    carryName(Op, NC);
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
    Op->erase();
    Changed = true;
  }
  return Changed;
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
      carryName(Op, NC);
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
      Op->erase();
      Changed = true;
      continue;
    }
    int64_t Rows = 0, Cols = 0;
    SmallVector<Value, 16> Elts;
    if (!gatherLiteralElements(Op, Rows, Cols, Elts)) {
      /* H: Classdef array literal detection.  `[obj1; obj2; obj3]` of
       * classdef instances should build an object-array via the shipped
       * generic carrier (matlab_dlnet_oa_new + matlab_dlnet_oa_append),
       * NOT a matlab_vertcat (which interprets ptrs as matlab_mat* and
       * crashes).  Detect by walking operands; any operand whose
       * defining op is a func.call to a classdef method or a load from
       * a `matlab.class_id`-tagged alloc indicates an object array.
       *
       * We require ALL operands to be classdef instances to avoid
       * mixed-mode (matrix + classdef) literals — those are a user
       * error in MATLAB too.  The check is conservative; on a partial
       * match we fall through to the matrix-cat path. */
      auto isClassdefInstance = [&](Value V) -> bool {
        if (V.getType() != PtrTy) return false;
        Operation *D = V.getDefiningOp();
        if (!D) return false;
        if (auto Call = mlir::dyn_cast<mlir::func::CallOp>(D)) {
          /* Lookup the callee func.func; check matlab.class_name attr. */
          auto Sym = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
              D, Call.getCalleeAttr());
          if (Sym && Sym->hasAttr("matlab.class_name")) return true;
        }
        if (auto Load = mlir::dyn_cast<LLVM::LoadOp>(D)) {
          Operation *AllocOp = Load.getAddr().getDefiningOp();
          if (AllocOp && isMatlabOp(AllocOp, "matlab.alloc") &&
              AllocOp->hasAttr("matlab.class_id"))
            return true;
        }
        return false;
      };
      /* Recursively gather all leaf operands (skipping nested concat
       * ops). */
      std::function<bool(Operation *, SmallVector<Value> &)> gatherLeaves =
          [&](Operation *C, SmallVector<Value> &leaves) -> bool {
        for (Value V : C->getOperands()) {
          Operation *D = V.getDefiningOp();
          if (D && (isMatlabOp(D, "matlab.concat_row") ||
                    isMatlabOp(D, "matlab.concat_col"))) {
            if (!gatherLeaves(D, leaves)) return false;
          } else {
            leaves.push_back(V);
          }
        }
        return true;
      };
      SmallVector<Value> leaves;
      bool allClassdef = false;
      if (gatherLeaves(Op, leaves) && !leaves.empty()) {
        allClassdef = true;
        for (Value V : leaves) {
          if (!isClassdefInstance(V)) { allClassdef = false; break; }
        }
      }
      if (allClassdef) {
        B.setInsertionPoint(Op);
        auto NewFn = rt("matlab_dlnet_oa_new", PtrTy, {});
        Value arr = LLVM::CallOp::create(B, Op->getLoc(), NewFn, ValueRange{}).getResult();
        auto AppendFn = rt("matlab_dlnet_oa_append", PtrTy, {PtrTy, PtrTy});
        for (Value V : leaves) {
          arr = LLVM::CallOp::create(B, Op->getLoc(), AppendFn,
                                      ValueRange{arr, V}).getResult();
        }
        Op->getResult(0).replaceAllUsesWith(arr);
        Op->erase();
        Changed = true;
        continue;
      }
      /* Operands are not all f64 scalars — at least one is a matrix/vector
       * (e.g. `[x1 x2]` horzcat of column vectors, or `[a; b]` vertcat).
       * Fold the bracket concatenation via the runtime matlab_horzcat /
       * matlab_vertcat (pairwise left-fold), boxing any scalar operands
       * and recursing into nested concat rows. */
      std::function<Value(Operation *)> fold = [&](Operation *C) -> Value {
        bool isRow = isMatlabOp(C, "matlab.concat_row");
        Value acc;
        for (Value V : C->getOperands()) {
          Value piece;
          Operation *D = V.getDefiningOp();
          if (D && (isMatlabOp(D, "matlab.concat_row") ||
                    isMatlabOp(D, "matlab.concat_col"))) {
            piece = fold(D);
          } else if (V.getType() == F64) {
            B.setInsertionPoint(Op);
            auto Bx = rt("matlab_mat_from_scalar", PtrTy, {F64});
            piece = LLVM::CallOp::create(B, Op->getLoc(), Bx, ValueRange{V}).getResult();
          } else if (V.getType() == PtrTy) {
            piece = V;
          } else if (isTensorLike(V.getType())) {
            B.setInsertionPoint(Op);
            piece = mlir::UnrealizedConversionCastOp::create(B, Op->getLoc(), PtrTy, V)
                        .getResult(0);
          } else {
            return Value{};
          }
          if (!piece) return Value{};
          if (!acc) { acc = piece; continue; }
          B.setInsertionPoint(Op);
          auto Fn = rt(isRow ? "matlab_horzcat" : "matlab_vertcat", PtrTy, {PtrTy, PtrTy});
          acc = LLVM::CallOp::create(B, Op->getLoc(), Fn, ValueRange{acc, piece}).getResult();
        }
        return acc;
      };
      Value M = fold(Op);
      if (!M) continue;
      Op->getResult(0).replaceAllUsesWith(M);
      Op->erase();
      Changed = true;
      continue;
    }
    B.setInsertionPoint(Op);
    Value M = materializeMat(Op->getLoc(), Rows, Cols, Elts);
    // Carry forward the user-source variable name (set by SlotPromotion
    // when the literal was assigned into a named slot) onto the new
    // mat_from_buf call so the Python emitter can render
    // `A = np.array(...).reshape(...)` instead of an anonymous v0.
    if (auto NA = Op->getAttrOfType<StringAttr>("matlab.name"))
      if (Operation *Def = M.getDefiningOp())
        if (!Def->hasAttr("matlab.name"))
          Def->setAttr("matlab.name", NA);
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
      /* On a second LowerTensorOps sweep the matlab.const_char that
       * fed this call may already have been replaced by an
       * llvm.mlir.addressof — that's the materialised global. If
       * NameV is already a ptr-typed addressof of an existing
       * `__matlab_str_*` constant, just reuse it and recover the
       * length from the global's stored value. */
      if (auto Addr = mlir::dyn_cast_or_null<LLVM::AddressOfOp>(Def)) {
        if (auto G = mlir::SymbolTable::lookupNearestSymbolFrom<
                LLVM::GlobalOp>(Addr, Addr.getGlobalNameAttr())) {
          if (auto Val = mlir::dyn_cast_or_null<StringAttr>(
                  G.getValueAttr())) {
            LenOut = (int64_t)Val.getValue().size();
            return Addr.getResult();
          }
        }
      }
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

    /* Normalize 1-arg forms of 2-arg shape builtins: `eye(n)` is
     * `eye(n, n)`, same for zeros / ones / rand / randn. The runtime
     * only exposes 2-arg entries (matlab_eye(m, n) etc.), so rewrite
     * the call site rather than widen the ABI. Only matches when the
     * single operand is f64 — a 1-arg matrix form (e.g. `zeros(size(A))`)
     * is a different, not-yet-supported semantic. */
    if (Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == F64 &&
        (Name == "eye" || Name == "zeros" || Name == "ones" ||
         Name == "rand" || Name == "randn")) {
      Value N0 = Call->getOperand(0);
      B.setInsertionPoint(Call);
      mlir::OperationState S(Call->getLoc(), "matlab.call_builtin");
      S.addOperands({N0, N0});
      S.addTypes(Call->getResultTypes());
      for (auto A : Call->getAttrs()) S.addAttribute(A.getName(), A.getValue());
      Operation *New = B.create(S);
      for (unsigned i = 0; i < Call->getNumResults(); ++i)
        Call->getResult(i).replaceAllUsesWith(New->getResult(i));
      Call->erase();
      Changed = true;
      continue;
    }

    /* `diag(s)` with a scalar f64 `s` is `diag([s])`, a 1x1 matrix
     * with `s` on its single diagonal slot. Runtime matlab_diag takes
     * a matrix ptr, so box the scalar via matlab_mat_from_scalar(f64)
     * first. MATLAB accepts `diag(4) == 4` (scalar→1x1). */
    if (Name == "diag" && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == F64) {
      Value S0 = Call->getOperand(0);
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_mat_from_scalar", PtrTy, {F64});
      auto Box = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                       ValueRange{S0});
      mlir::OperationState St(Call->getLoc(), "matlab.call_builtin");
      St.addOperands({Box.getResult()});
      St.addTypes(Call->getResultTypes());
      for (auto A : Call->getAttrs()) St.addAttribute(A.getName(), A.getValue());
      Operation *New = B.create(St);
      for (unsigned i = 0; i < Call->getNumResults(); ++i)
        Call->getResult(i).replaceAllUsesWith(New->getResult(i));
      Call->erase();
      Changed = true;
      continue;
    }

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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
    /* §3.1: disp(tf) — Lowering.cpp routes a tf-pinned operand
     * through matlab_tf_disp(matlab_obj *) instead of the generic
     * matlab_disp_mat path. Same shape as matlab_string_disp: one
     * ptr operand, void result. */
    if (Name == "matlab_tf_disp" && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_tf_disp", VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase();
      Changed = true;
      continue;
    }
    /* Phase 1.1.C — typed int matrix disp. Lowering.cpp swaps the
     * callee on disp(typed_matrix) sites so the runtime entry hits
     * matlab_mat_i32_disp / matlab_mat_u8_disp directly. Mirror the
     * matlab_string_disp dispatch above; both consume one ptr operand
     * and return void. */
    if ((Name == "matlab_mat_i32_disp" || Name == "matlab_mat_u8_disp") &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase();
      Changed = true;
      continue;
    }
    /* Phase 1.1.G — cross-lane / to-double matrix casts. Lowering.cpp
     * picks the lane-aware callee when it sees `int32(uint8_matrix)` /
     * `uint8(int32_matrix)` / `double(typed_int_matrix)`; these all
     * take one ptr operand and return one ptr (the target descriptor). */
    if ((Name == "matlab_mat_i32_from_u8"  ||
         Name == "matlab_mat_u8_from_i32"  ||
         Name == "matlab_mat_i32_to_double" ||
         Name == "matlab_mat_u8_to_double") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* matlab_string_size_scalar() — emitted by the lowering's
     * size("...") fold for string scalars. Returns a fresh 1x2 row
     * vector [1 1] so downstream consumers (assignment to ans, disp,
     * arith) see a proper matlab_mat* instead of a misrouted call. */
    if (Name == "matlab_string_size_scalar" &&
        Call->getNumOperands() == 0 && Call->getNumResults() == 1) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_string_size_scalar", PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
        carryName(Call, NC);
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
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      /* General path: %s + vector args + 2+ values via the variadic
       * descriptor-array core (matlab_sprintf_vec).  Operand 0 is the format
       * (matlab_string*); operands 1+ are values.  The frontend `str_mask`
       * marks which are strings (kind 1 = matlab_string*, 0 = numeric mat).
       * Wait for every value operand to settle to ptr/f64 first. */
      {
        unsigned NOps = Call->getNumOperands();
        bool ready = true;
        for (unsigned i = 1; i < NOps; ++i) {
          Type T = Call->getOperand(i).getType();
          if (T != PtrTy && T != F64) { ready = false; break; }
        }
        if (ready) {
          auto Loc = Call->getLoc();
          auto I64 = B.getI64Type();
          auto I8 = B.getIntegerType(8);
          int64_t StrMask = 0;
          if (auto MA = Call->getAttrOfType<IntegerAttr>("str_mask"))
            StrMask = MA.getInt();
          B.setInsertionPoint(Call);
          unsigned NVals = NOps - 1;
          Value One = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
          Value ValsBuf = LLVM::AllocaOp::create(
              B, Loc, PtrTy, LLVM::LLVMArrayType::get(PtrTy, NVals), One, 0);
          Value KindsBuf = LLVM::AllocaOp::create(
              B, Loc, PtrTy, LLVM::LLVMArrayType::get(I8, NVals), One, 0);
          for (unsigned i = 1; i < NOps; ++i) {
            Value V = Call->getOperand(i);
            bool IsStr = (StrMask >> i) & 1;
            Value ValPtr;
            int8_t Kind;
            if (IsStr && V.getType() == PtrTy) {
              Kind = 1; ValPtr = V;
            } else if (V.getType() == F64) {
              Kind = 0;
              auto Fn = rt("matlab_mat_scalar", PtrTy, {F64});
              ValPtr = LLVM::CallOp::create(B, Loc, Fn, ValueRange{V}).getResult();
            } else {
              Kind = 0; ValPtr = V;   /* numeric matrix descriptor */
            }
            Value Idx = LLVM::ConstantOp::create(B, Loc, I64,
                                                 B.getI64IntegerAttr((int64_t)(i - 1)));
            Value VGep = LLVM::GEPOp::create(B, Loc, PtrTy, PtrTy, ValsBuf,
                                             ValueRange{Idx});
            LLVM::StoreOp::create(B, Loc, ValPtr, VGep);
            Value KGep = LLVM::GEPOp::create(B, Loc, PtrTy, I8, KindsBuf,
                                             ValueRange{Idx});
            LLVM::StoreOp::create(B, Loc,
                LLVM::ConstantOp::create(B, Loc, I8, B.getI8IntegerAttr(Kind)), KGep);
          }
          Value NV = LLVM::ConstantOp::create(B, Loc, I64,
                                              B.getI64IntegerAttr((int64_t)NVals));
          auto Fn = rt("matlab_sprintf_vec", PtrTy, {PtrTy, PtrTy, PtrTy, I64});
          auto NC = LLVM::CallOp::create(
              B, Loc, Fn, ValueRange{Call->getOperand(0), ValsBuf, KindsBuf, NV});
          if (Call->getResult(0).getType() != PtrTy)
            Call->getResult(0).setType(PtrTy);
          carryName(Call, NC);
          Call->getResult(0).replaceAllUsesWith(NC.getResult());
          Call->erase();
          Changed = true;
          continue;
        }
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cell_set_f64" ||
         Name == "matlab_cell_set_mat" ||
         Name == "matlab_cell_set_str") &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      bool IsMat = Name != "matlab_cell_set_f64";  /* mat + str take a ptr */
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
    /* Phase 1.3: 2-D cell ops. matlab_cell_new_2d takes (rows, cols)
     * f64 and returns ptr; set_*_2d takes (cell, r, k, value); get_*_2d
     * takes (cell, r, k); cell_size_dim (cell, dim) -> f64; concat_*
     * takes (cell, cell) -> cell ptr. */
    if (Name == "matlab_cell_new_2d" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_cell_new_2d", PtrTy, {F64, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cell_set_f64_2d" ||
         Name == "matlab_cell_set_mat_2d") &&
        Call->getNumOperands() == 4 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64) {
      bool IsMat = Name == "matlab_cell_set_mat_2d";
      Value V = Call->getOperand(3);
      if (IsMat && V.getType() != PtrTy) continue;
      if (!IsMat && V.getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy, F64, F64,
                                  IsMat ? (Type)PtrTy : (Type)F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1),
                                       Call->getOperand(2), V});
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cell_get_f64_2d" ||
         Name == "matlab_cell_get_mat_2d") &&
        Call->getNumResults() == 1 &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64) {
      B.setInsertionPoint(Call);
      bool IsMat = Name == "matlab_cell_get_mat_2d";
      auto Fn = rt(Name, IsMat ? (Type)PtrTy : (Type)F64,
                   {PtrTy, F64, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1),
                                                 Call->getOperand(2)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_cell_size_dim" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_cell_size_dim", F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cell_concat_row" ||
         Name == "matlab_cell_concat_col") &&
        Call->getNumResults() == 1 &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
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
      carryName(Call, NC);
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_videowriter_set_framerate" ||
         Name == "matlab_videowriter_set_quality") &&
        Call->getNumOperands() == 2) {
      Value Vw = Call->getOperand(0);
      Value Val = Call->getOperand(1);
      if (Vw.getType() != PtrTy) continue;
      B.setInsertionPoint(Call);
      /* The runtime setter takes a double; coerce an integer RHS. */
      if (auto IT = mlir::dyn_cast<IntegerType>(Val.getType())) {
        (void)IT;
        Val = LLVM::SIToFPOp::create(B, Call->getLoc(), F64, Val);
      } else if (Val.getType() != F64) {
        continue;   // unexpected RHS type — leave for a later pass / error
      }
      auto Fn = rt(Name, VoidTy, {PtrTy, F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Vw, Val});
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_struct_set_f64" ||
         Name == "matlab_struct_set_mat" ||
         Name == "matlab_struct_set_string") &&
        Call->getNumOperands() == 3) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      Value Val = Call->getOperand(2);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      /* A char-string field store (#79.2) carries a `matlab_string *`
       * (ptr) value — same call shape as `_set_mat`, distinct runtime
       * entry (stores with kind=3). Treat its value as ptr-typed but
       * keep the `_set_string` callee. */
      bool IsStr = Name == "matlab_struct_set_string";
      bool IsMat = Name == "matlab_struct_set_mat" || IsStr;
      /* Auto-promote `_f64` callee to `_mat` when the value operand
       * arrived as ptr — the AST-time dispatch in Lowering.cpp
       * picks the callee from the RHS type at lowering time, but
       * polymorphic flows (function args, class field stores
       * sourced from another function's return) settle their type
       * later in the pipeline. The runtime entries are
       * interchangeable on the type axis: `_mat` reads/writes the
       * matlab_mat * directly, `_f64` boxes a 1×1 — so promoting
       * is safe whenever the actual value is ptr. */
      if (!IsMat && Val.getType() == PtrTy) {
        Name = "matlab_struct_set_mat";
        IsMat = true;
      }
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* Phase 2: struct array runtime. _new takes no operands; _get and
     * _get_or_create take (arr, i) f64; _length / _numel take one
     * arg; _size_dim takes (arr, dim). */
    if (Name == "matlab_struct_arr_new" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_struct_arr_get" ||
         Name == "matlab_struct_arr_get_or_create") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_struct_arr_length" ||
         Name == "matlab_struct_arr_numel") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_struct_arr_size_dim" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* Workspace management commands: who / whos / clear take no
     * operands; clear_one takes a single const_char name. Delegate
     * directly.
     *
     * `matlab_dbg_keyboard_hook` is the runtime entry the lowerer
     * emits for a `keyboard` builtin call — same shape (no
     * operands, no result), so it slots into the same dispatch
     * arm. */
    if ((Name == "matlab_ws_who" || Name == "matlab_ws_whos" ||
         Name == "matlab_ws_clear" ||
         Name == "matlab_dbg_keyboard_hook" ||
         Name == "matlab_tic" || Name == "matlab_toc_print" ||
         Name == "matlab_pause_keypress") &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->erase();
      Changed = true;
      continue;
    }
    /* matlab_toc() — zero operands, single f64 result. Separate from
     * the void arm above because callers consume the elapsed value
     * (e.g. `t = toc()`). */
    if (Name == "matlab_toc" && Call->getNumOperands() == 0 &&
        Call->getNumResults() == 1) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_toc", F64, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* matlab_pause(seconds) — one f64 operand, no result. Mirrors the
     * tic/toc shape but accepts the sleep duration. */
    if (Name == "matlab_pause" && Call->getNumOperands() == 1) {
      Value Secs = Call->getOperand(0);
      if (Secs.getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_pause", VoidTy, {F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Secs});
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_ws_clear_one" && Call->getNumOperands() == 1) {
      Value NameV = Call->getOperand(0);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_ws_clear_one", VoidTy, {PtrTy, I64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Ptr, LenV});
      Call->erase();
      Changed = true;
      continue;
    }

    /* dbg(x) / dbg(x, "label"): the frontend emits
     *   matlab.call_builtin @matlab_dbg_* (file_char, line_i32,
     *                                      label_char, value)
     * Materialise both const_chars to (ptr, i64) pairs and emit
     * the 6-arg runtime call. */
    if ((Name == "matlab_dbg_f64" || Name == "matlab_dbg_mat") &&
        Call->getNumOperands() == 4) {
      auto I32 = IntegerType::get(Ctx, 32);
      Value FileV = Call->getOperand(0);
      Value LineV = Call->getOperand(1);
      Value LabelV = Call->getOperand(2);
      Value Val = Call->getOperand(3);
      if (LineV.getType() != I32) continue;
      bool IsMat = Name == "matlab_dbg_mat";
      if (IsMat && Val.getType() != PtrTy) continue;
      if (!IsMat && Val.getType() != F64) continue;
      int64_t FileLen = 0, LabelLen = 0;
      Value FilePtr = fieldNameAddr(FileV, FileLen);
      Value LabelPtr = fieldNameAddr(LabelV, LabelLen);
      if (!FilePtr || !LabelPtr) continue;
      B.setInsertionPoint(Call);
      Value FileLenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(FileLen));
      Value LabelLenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(LabelLen));
      auto Fn = rt(Name, VoidTy,
                   {PtrTy, I64, I32, PtrTy, I64,
                    IsMat ? (Type)PtrTy : (Type)F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{FilePtr, FileLenV, LineV,
                                       LabelPtr, LabelLenV, Val});
      Call->erase();
      Changed = true;
      continue;
    }

    /* Debug hook: injected by the lowerer at each statement when
     * -g is on. Two i32 operands (file_id, line); returns void. */
    if (Name == "matlab_dbg_hook" &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      auto I32 = IntegerType::get(Ctx, 32);
      Value FileV = Call->getOperand(0);
      Value LineV = Call->getOperand(1);
      if (FileV.getType() != I32 || LineV.getType() != I32) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_dbg_hook", VoidTy, {I32, I32});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{FileV, LineV});
      Call->erase();
      Changed = true;
      continue;
    }

    /* User-function frame entry: emitted by the lowerer at the top of
     * each user function body when -g is on. Single const_char operand
     * carrying the displayed name. We materialize the name as an
     * !llvm.ptr (via fieldNameAddr, same path as struct field names)
     * and call matlab_dbg_enter_frame(ptr, i64). */
    if (Name == "matlab_dbg_enter_frame" &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1) {
      Value NameV = Call->getOperand(0);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_dbg_enter_frame", VoidTy, {PtrTy, I64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Ptr, LenV});
      Call->erase();
      Changed = true;
      continue;
    }

    /* User-function frame exit: emitted before each func.return when
     * -g is on. No operands. */
    if (Name == "matlab_dbg_leave_frame" &&
        Call->getNumOperands() == 0 && Call->getNumResults() == 1) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_dbg_leave_frame", VoidTy, {});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      Call->erase();
      Changed = true;
      continue;
    }

    /* Class-name registration: emitted at the top of the script body
     * (DebugMode only) once per classdef so the runtime can resolve
     * class_id -> class name when the DAP server formats class
     * instances. Two operands: (i32 class_id, const_char name). */
    if (Name == "matlab_dbg_register_class" &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      auto I32 = IntegerType::get(Ctx, 32);
      Value ClsId = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      if (ClsId.getType() != I32) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_dbg_register_class", VoidTy, {I32, PtrTy, I64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{ClsId, Ptr, LenV});
      Call->erase();
      Changed = true;
      continue;
    }

    /* Per-frame Locals mirror: emitted by emitStore in DebugMode for
     * every store to a named slot. The first operand is a const_char
     * carrying the variable name; the second is the stored value.
     * We dispatch on the operand's lowered type — f64 routes to
     * matlab_dbg_frame_set_f64, !llvm.ptr (matrix descriptor) routes
     * to matlab_dbg_frame_set_mat. When the call carries a
     * `matlab.class_id` attribute (set by emitStore for slots whose
     * binding is pinned to a user classdef), the ptr operand is a
     * matlab_obj* and we route to matlab_dbg_frame_set_obj instead so
     * the runtime can stamp kind=2 and the DAP server keeps the class
     * identity. Operands that are still none-typed at this point
     * (scalar promotion hasn't completed yet) are punted to the next
     * iteration of the rewrite loop. */
    if (Name == "matlab_dbg_frame_set" &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      Value NameV = Call->getOperand(0);
      Value Val = Call->getOperand(1);
      mlir::Type VT = Val.getType();
      bool IsF64 = mlir::isa<mlir::Float64Type>(VT);
      bool IsPtr = mlir::isa<LLVM::LLVMPointerType>(VT);
      bool IsInt = mlir::isa<mlir::IntegerType>(VT);
      if (!IsF64 && !IsPtr && !IsInt) continue; /* still none-typed, retry */
      bool IsObj = IsPtr && Call->hasAttr("matlab.class_id");
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      /* Integer-typed mirror values (i1 from `r = age > 18`, i8/i16/i32
       * from fixed-point or integer arith) don't have a dedicated
       * runtime variant. Cast to f64 and reuse matlab_dbg_frame_set_f64
       * — bool maps to 0.0/1.0, narrow ints round-trip exactly, i64
       * loses precision past 2^53 but that's fine for a debug mirror.
       * Without this, the matlab.call_builtin survives all subsequent
       * passes and the func-to-llvm conversion fails on the residual
       * matlab.const_char operand. Guarded by
       * test/Debug/run_jit_userfn_tests.py (logical_return_gt_const). */
      if (IsInt) {
        auto IT = mlir::cast<mlir::IntegerType>(VT);
        if (IT.getWidth() == 1) {
          Value One = LLVM::ConstantOp::create(
              B, Call->getLoc(), F64, B.getF64FloatAttr(1.0));
          Value Zero = LLVM::ConstantOp::create(
              B, Call->getLoc(), F64, B.getF64FloatAttr(0.0));
          Val = LLVM::SelectOp::create(B, Call->getLoc(), Val, One, Zero);
        } else if (IT.isSigned() || IT.isSignless()) {
          Val = LLVM::SIToFPOp::create(B, Call->getLoc(), F64, Val);
        } else {
          Val = LLVM::UIToFPOp::create(B, Call->getLoc(), F64, Val);
        }
        IsF64 = true;
      }
      const char *Callee = IsF64 ? "matlab_dbg_frame_set_f64"
                                 : (IsObj ? "matlab_dbg_frame_set_obj"
                                          : "matlab_dbg_frame_set_mat");
      mlir::Type ValTy = IsF64 ? mlir::Type(F64) : mlir::Type(PtrTy);
      auto Fn = rt(Callee, VoidTy, {PtrTy, I64, ValTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Ptr, LenV, Val});
      Call->erase();
      Changed = true;
      continue;
    }

    /* REPL workspace accessors. Shape is the same as struct_* but
     * without a base ptr (the workspace is a singleton inside the
     * runtime). Used only when matlabc is invoked with -repl. */
    if ((Name == "matlab_ws_get_f64" || Name == "matlab_ws_get_mat" ||
         Name == "matlab_ws_get_string" || Name == "matlab_ws_get_sym" ||
         Name == "matlab_ws_get_symmat" || Name == "matlab_ws_get_handle") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1) {
      Value NameV = Call->getOperand(0);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = (Name == "matlab_ws_get_mat" ||
                    Name == "matlab_ws_get_string" ||
                    Name == "matlab_ws_get_sym" ||
                    Name == "matlab_ws_get_symmat" ||
                    Name == "matlab_ws_get_handle");
      Type Ret = IsMat ? (Type)PtrTy : (Type)F64;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, Ret, {PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Ptr, LenV});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* Scalar function-handle trampoline: matlab_call_handle_s{0,1,2,3}.
     * Operand 0 is the stored function pointer (ptr); the remaining
     * operands are f64 call arguments (none for s0, a zero-arg `@() ...`);
     * the result is f64.  Emitted by the lowering for `f(args)` where `f`
     * is a workspace-backed handle. */
    if ((Name == "matlab_call_handle_s0" || Name == "matlab_call_handle_s1" ||
         Name == "matlab_call_handle_s2" || Name == "matlab_call_handle_s3") &&
        Call->getNumResults() == 1 && Call->getNumOperands() >= 1) {
      Value Fn = Call->getOperand(0);
      if (Fn.getType() != PtrTy) continue;   /* wait for the ws load to lower */
      SmallVector<Type, 4> ArgTys;
      SmallVector<Value, 4> ArgVals;
      ArgTys.push_back(PtrTy);
      ArgVals.push_back(Fn);
      bool Ready = true;
      for (unsigned i = 1; i < Call->getNumOperands(); ++i) {
        Value A = Call->getOperand(i);
        if (A.getType() != F64) { Ready = false; break; }
        ArgTys.push_back(F64);
        ArgVals.push_back(A);
      }
      if (!Ready) continue;
      B.setInsertionPoint(Call);
      auto Fnc = rt(Name, F64, ArgTys);
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fnc, ArgVals);
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    if ((Name == "matlab_ws_set_f64" || Name == "matlab_ws_set_mat" ||
         Name == "matlab_ws_set_obj" || Name == "matlab_ws_set_string" ||
         Name == "matlab_ws_set_sym" || Name == "matlab_ws_set_symmat" ||
         Name == "matlab_ws_set_table" || Name == "matlab_ws_set_categorical" ||
         Name == "matlab_ws_set_datetime" || Name == "matlab_ws_set_duration" ||
         Name == "matlab_ws_set_struct" || Name == "matlab_ws_set_handle") &&
        Call->getNumOperands() == 2) {
      Value NameV = Call->getOperand(0);
      Value Val = Call->getOperand(1);
      /* Retarget f64-vs-mat based on the actual runtime type of Val
       * — the frontend chose between set_f64 / set_mat from Sema's
       * inferred type, but later passes may refine a binop result
       * from f64 to ptr (e.g. `x * 2` where x is a workspace matrix).
       * When the initial choice doesn't match the final value type,
       * flip to the correct variant so the store uses the right
       * runtime entry. The set_obj / set_string choices are sticky:
       * they carry Sema-pinned class / string-binding info, so we
       * trust the frontend and don't downgrade to set_mat even if
       * the value is a generic ptr. Same stickiness for set_table /
       * set_categorical / set_datetime / set_duration — the frontend
       * tagged the binding via the Phase-5 binding sets and the value
       * is always a pointer at this point. */
      bool IsObj = (Name == "matlab_ws_set_obj");
      bool IsString = (Name == "matlab_ws_set_string");
      bool IsSym = (Name == "matlab_ws_set_sym");
      bool IsSymmat = (Name == "matlab_ws_set_symmat");
      bool IsTable = (Name == "matlab_ws_set_table");
      bool IsCategorical = (Name == "matlab_ws_set_categorical");
      bool IsDatetime = (Name == "matlab_ws_set_datetime");
      bool IsDuration = (Name == "matlab_ws_set_duration");
      bool IsStruct   = (Name == "matlab_ws_set_struct");
      bool IsHandle   = (Name == "matlab_ws_set_handle");
      bool IsPtrSticky = IsObj || IsString || IsSym || IsSymmat ||
                         IsTable || IsCategorical || IsDatetime ||
                         IsDuration || IsStruct || IsHandle;
      bool IsMat;
      bool IsInt = mlir::isa<mlir::IntegerType>(Val.getType());
      if (IsPtrSticky) IsMat = true;
      else if (Val.getType() == PtrTy)      IsMat = true;
      else if (Val.getType() == F64)         IsMat = false;
      else if (IsInt)                         IsMat = false;
      else continue;   /* neither ptr nor f64 nor int yet — wait for another iter */
      if (IsPtrSticky && Val.getType() != PtrTy)
        continue; /* retry once Val lowers */
      /* Cast int → f64 for the workspace mirror. Same logic as
       * matlab_dbg_frame_set above: i1 from `x = age > 18` at script
       * scope (REPL/DAP) needs to flow through matlab_ws_set_f64.
       * Without this, the matlab.call_builtin would survive into
       * func-to-llvm conversion and the JIT would refuse the module.
       * Guarded by run_jit_userfn_tests.py (script_scope_bool_var). */
      if (IsInt) {
        auto IT = mlir::cast<mlir::IntegerType>(Val.getType());
        B.setInsertionPoint(Call);
        if (IT.getWidth() == 1) {
          Value One = LLVM::ConstantOp::create(
              B, Call->getLoc(), F64, B.getF64FloatAttr(1.0));
          Value Zero = LLVM::ConstantOp::create(
              B, Call->getLoc(), F64, B.getF64FloatAttr(0.0));
          Val = LLVM::SelectOp::create(B, Call->getLoc(), Val, One, Zero);
        } else if (IT.isSigned() || IT.isSignless()) {
          Val = LLVM::SIToFPOp::create(B, Call->getLoc(), F64, Val);
        } else {
          Val = LLVM::UIToFPOp::create(B, Call->getLoc(), F64, Val);
        }
      }
      /* Call fieldNameAddr AFTER the Val type check. fieldNameAddr
       * has a side effect (materialises a global + addressof and
       * replaces the const_char's uses with the addressof); once
       * that fires, subsequent calls can't find the original
       * const_char. So don't call it unless we're about to commit
       * to the rewrite. */
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      StringRef RuntimeName =
          IsHandle       ? "matlab_ws_set_handle"
          : (IsSymmat    ? "matlab_ws_set_symmat"
                         : (IsSym ? "matlab_ws_set_sym"
                              : (IsString      ? "matlab_ws_set_string"
                              : (IsObj         ? "matlab_ws_set_obj"
                              : (IsStruct      ? "matlab_ws_set_struct"
                              : (IsTable       ? "matlab_ws_set_table"
                              : (IsCategorical ? "matlab_ws_set_categorical"
                              : (IsDatetime    ? "matlab_ws_set_datetime"
                              : (IsDuration    ? "matlab_ws_set_duration"
                              : (IsMat         ? "matlab_ws_set_mat"
                                               : "matlab_ws_set_f64"))))))))));
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(RuntimeName, VoidTy,
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
    if ((Name == "matlab_obj_set_f64" || Name == "matlab_obj_set_mat" ||
         Name == "matlab_obj_set_string") &&
        Call->getNumOperands() == 3) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      Value Val = Call->getOperand(2);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = Name == "matlab_obj_set_mat";
      bool IsString = Name == "matlab_obj_set_string";
      /* Auto-promote `_f64` callee to `_mat` when the value operand
       * arrived as ptr — same reasoning as the matlab_struct_set
       * dispatch above. The class-method case especially needs this:
       * a `tf` constructor's `obj.Numerator = num` was lowered with
       * the f64 callee at AST time (because Sema couldn't see
       * through the param), but after LowerUserCalls retypes the
       * function signature to ptr the field-store now carries a
       * ptr-typed value. */
      if (!IsMat && !IsString && Val.getType() == PtrTy) {
        Name = "matlab_obj_set_mat";
        IsMat = true;
      }
      if (IsString && Val.getType() != PtrTy) continue;
      /* A matrix-typed property assigned a SCALAR (f64) value — e.g. a
       * 1x1 `[1]` passed to a constructor that stores it as a matrix
       * (local-level ssm A/B/C/D).  Box the scalar into a 1x1 matrix so
       * the property still reads back via matlab_obj_get_mat. */
      if (IsMat && Val.getType() == F64) {
        B.setInsertionPoint(Call);
        auto Box = rt("matlab_mat_from_scalar", PtrTy, {F64});
        Val = LLVM::CallOp::create(B, Call->getLoc(), Box, ValueRange{Val})
                  .getResult();
      }
      if (IsMat && Val.getType() != PtrTy) continue;
      if (!IsMat && !IsString && Val.getType() != F64) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      Type ValTy = (IsMat || IsString) ? (Type)PtrTy : (Type)F64;
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy, I64, ValTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Base, Ptr, LenV, Val});
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_obj_get_f64" || Name == "matlab_obj_get_mat" ||
         Name == "matlab_obj_get_string") &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsPtr = (Name == "matlab_obj_get_mat" ||
                    Name == "matlab_obj_get_string");
      Type Ret = IsPtr ? (Type)PtrTy : (Type)F64;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, Ret, {PtrTy, PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Base, Ptr, LenV});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* matlab_obj_disp_field(obj, "Name") — runtime-dispatched disp
     * routing emitted by Lowering's `disp(obj.Field)` site.  Same
     * shape as the obj_get/set helpers: ptr + name + len.  Void
     * return. */
    if (Name == "matlab_obj_disp_field" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      if (Base.getType() != PtrTy) continue;
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt("matlab_obj_disp_field", VoidTy, {PtrTy, PtrTy, I64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Base, Ptr, LenV});
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* Phase 4: containers.Map / dictionary runtime. */
    if (Name == "matlab_dict_new" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_dict_new", PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_dict_set_str_f64" ||
         Name == "matlab_dict_set_str_mat") &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      bool IsMat = Name == "matlab_dict_set_str_mat";
      Value V = Call->getOperand(2);
      if (IsMat && V.getType() != PtrTy) continue;
      if (!IsMat && V.getType() != F64) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy,
                                  IsMat ? (Type)PtrTy : (Type)F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1), V});
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_dict_set_num_f64" ||
         Name == "matlab_dict_set_num_mat") &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      bool IsMat = Name == "matlab_dict_set_num_mat";
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
    if ((Name == "matlab_dict_get_str_f64" ||
         Name == "matlab_dict_get_str_mat") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      bool IsMat = Name == "matlab_dict_get_str_mat";
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, IsMat ? (Type)PtrTy : (Type)F64,
                   {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_dict_get_num_f64" ||
         Name == "matlab_dict_get_num_mat") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      bool IsMat = Name == "matlab_dict_get_num_mat";
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, IsMat ? (Type)PtrTy : (Type)F64,
                   {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_dict_has_str" ||
         Name == "matlab_dict_remove_str") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_dict_has_num" ||
         Name == "matlab_dict_remove_num") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_dict_length" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_dict_length", F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* Phase 5.3: table runtime. */
    if (Name == "matlab_table_new" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_table_add_column" && Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(2).getType() == PtrTy) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      Value Col = Call->getOperand(2);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy, I64, PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Base, Ptr, LenV, Col});
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_table_get_column" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Base, Ptr, LenV});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_table_height" ||
         Name == "matlab_table_width" ||
         Name == "matlab_table_numel") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_table_size_dim" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_table_disp" && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase(); Changed = true; continue;
    }

    /* Phase 5.2: categorical runtime. */
    if (Name == "matlab_categorical_from_cell" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_categorical_length" ||
         Name == "matlab_categorical_numcats") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_categorical_iscategory" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_categorical_categories" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_categorical_disp" &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_categorical_eq" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }

    /* Phase 5.1: datetime / duration runtime. Constructors take 0/3/6
     * f64 args and return ptr; converters (duration_to_*) take ptr
     * and return f64; disp takes ptr and returns void; arithmetic
     * takes two ptrs and returns ptr (or a dur in the dt-dt case). */
    if (Name == "matlab_datetime_now" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_datetime_ymd" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {F64, F64, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1),
                                                 Call->getOperand(2)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_datetime_ymdhms" && Call->getNumResults() == 1 &&
        Call->getNumOperands() == 6) {
      bool ok = true;
      for (int i = 0; i < 6; ++i)
        if (Call->getOperand(i).getType() != F64) { ok = false; break; }
      if (!ok) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {F64, F64, F64, F64, F64, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1),
                                                 Call->getOperand(2),
                                                 Call->getOperand(3),
                                                 Call->getOperand(4),
                                                 Call->getOperand(5)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_duration_seconds" ||
         Name == "matlab_duration_minutes" ||
         Name == "matlab_duration_hours" ||
         Name == "matlab_duration_days" ||
         Name == "matlab_duration_years") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    /* Phase 5.4: vec unit constructors take a matlab_mat * (PtrTy)
     * and return a matlab_duration_vec *. */
    if ((Name == "matlab_duration_seconds_vec" ||
         Name == "matlab_duration_minutes_vec" ||
         Name == "matlab_duration_hours_vec"   ||
         Name == "matlab_duration_days_vec"    ||
         Name == "matlab_duration_years_vec")  &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    /* Phase 5.4: vec disp / length / size_dim and vec arithmetic.
     * Disp is void; length / size_dim return F64; arithmetic
     * combinations all take two PtrTy and return PtrTy. */
    if ((Name == "matlab_datetime_vec_disp" ||
         Name == "matlab_duration_vec_disp") &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_datetime_vec_length" ||
         Name == "matlab_duration_vec_length") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_datetime_vec_size_dim" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_datetime_add_duration_vec"     ||
         Name == "matlab_datetime_vec_add_duration"     ||
         Name == "matlab_datetime_vec_sub_duration"     ||
         Name == "matlab_datetime_vec_add_duration_vec" ||
         Name == "matlab_datetime_vec_sub_datetime_vec" ||
         Name == "matlab_datetime_vec_sub_datetime")    &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    /* Phase 5.4 (cont.) — timetable.
     *   matlab_timetable_new        ()       -> ptr
     *   matlab_timetable_disp       (ptr)    -> void
     *   matlab_timetable_height/_width/_numel(ptr) -> f64
     *   matlab_timetable_size_dim   (ptr,f64)-> f64
     *   matlab_timetable_set_row_times(ptr, ptr) -> void
     *   matlab_timetable_add_column (ptr, ptr, ptr) -> void
     *   matlab_table2timetable      (ptr, ptr) -> ptr
     */
    if (Name == "matlab_timetable_new" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 0) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_disp" &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_timetable_height" ||
         Name == "matlab_timetable_width"  ||
         Name == "matlab_timetable_numel") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_size_dim" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_set_row_times" &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1)});
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_add_column" &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(2).getType() == PtrTy) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      Value Col = Call->getOperand(2);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy, I64, PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Base, Ptr, LenV, Col});
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_table2timetable" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    /* Phase 5.4 (cont.) — accessors. _get_row_times takes a single
     * timetable ptr and returns the datetime_vec ptr. _get_column
     * takes (ptr, char*, i64) via the fieldNameAddr bridge.
     * _select_var matches the same shape. _select_rows_mat takes
     * (ptr, ptr) and returns ptr.                                 */
    if (Name == "matlab_timetable_get_row_times" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_timetable_get_column" ||
         Name == "matlab_timetable_select_var") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Base, Ptr, LenV});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_select_rows_mat" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    /* timerange + timerange-row subscript. */
    if (Name == "matlab_timerange_new" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      auto I32 = IntegerType::get(B.getContext(), 32);
      if (Call->getOperand(2).getType() == I32) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy, I32});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0),
                                                   Call->getOperand(1),
                                                   Call->getOperand(2)});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase(); Changed = true; continue;
      }
    }
    if (Name == "matlab_timetable_retime" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy) {
      auto I32 = IntegerType::get(B.getContext(), 32);
      if (Call->getOperand(1).getType() == I32 &&
          Call->getOperand(2).getType() == I32) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {PtrTy, I32, I32});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0),
                                                   Call->getOperand(1),
                                                   Call->getOperand(2)});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase(); Changed = true; continue;
      }
    }
    if (Name == "matlab_timetable_horzcat" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_datetime_vec_to_mat" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_movavg" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy) {
      auto I32 = IntegerType::get(B.getContext(), 32);
      if (Call->getOperand(1).getType() == I32 &&
          Call->getOperand(2).getType() == I32) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {PtrTy, I32, I32});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0),
                                                   Call->getOperand(1),
                                                   Call->getOperand(2)});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase(); Changed = true; continue;
      }
    }
    if (Name == "matlab_timetable_macd" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_fillmissing" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      auto I32 = IntegerType::get(B.getContext(), 32);
      if (Call->getOperand(1).getType() == I32) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {PtrTy, I32});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0),
                                                   Call->getOperand(1)});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase(); Changed = true; continue;
      }
    }
    if (Name == "matlab_timetable_summary" &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_head" &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy, F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1)});
      Call->erase(); Changed = true; continue;
    }
    if (Name == "matlab_timetable_synchronize" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 4 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      auto I32 = IntegerType::get(B.getContext(), 32);
      if (Call->getOperand(2).getType() == I32 &&
          Call->getOperand(3).getType() == I32) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy, I32, I32});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0),
                                                   Call->getOperand(1),
                                                   Call->getOperand(2),
                                                   Call->getOperand(3)});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase(); Changed = true; continue;
      }
    }
    if (Name == "matlab_timetable_select_rows_timerange" &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    /* matlab_timetable_set_description(ptr, char*, i64) via the
     * fieldNameAddr bridge — the RHS string literal arrives as a
     * matlab.const_char op. */
    if (Name == "matlab_timetable_set_description" &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      Value Base = Call->getOperand(0);
      Value NameV = Call->getOperand(1);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      B.setInsertionPoint(Call);
      Value LenV = LLVM::ConstantOp::create(
          B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
      auto Fn = rt(Name, VoidTy, {PtrTy, PtrTy, I64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Base, Ptr, LenV});
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_duration_to_seconds" ||
         Name == "matlab_duration_to_minutes" ||
         Name == "matlab_duration_to_hours" ||
         Name == "matlab_duration_to_days") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, F64, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_datetime_disp" ||
         Name == "matlab_duration_disp") &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, VoidTy, {PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0)});
      Call->erase(); Changed = true; continue;
    }
    if ((Name == "matlab_datetime_sub_datetime" ||
         Name == "matlab_datetime_add_duration" ||
         Name == "matlab_datetime_sub_duration" ||
         Name == "matlab_duration_add" ||
         Name == "matlab_duration_sub") &&
        Call->getNumResults() == 1 && Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase(); Changed = true; continue;
    }

    /* Phase 6.1 — Symbolic matrices (matlab_symmat_*). Separate from
     * matlab_sym_* because the prefix collides ("matlab_sym" + "mat_*"
     * vs "matlab_sym" + "_*"). Pattern: ptr/i64/f64 operands, return
     * is ptr (matrix or sym), i64 (rank), or void/none (disp/set).
     * The catch-all below handles every signature the runtime ships. */
    if (Name.starts_with("matlab_symmat_")) {
      bool AllReady = true;
      llvm::SmallVector<Type, 6> Sig;
      for (auto V : Call->getOperands()) {
        mlir::Type T = V.getType();
        if (T == PtrTy || T == F64 ||
            mlir::isa<mlir::IntegerType>(T)) {
          Sig.push_back(T == F64 ? F64
                                  : (mlir::isa<mlir::IntegerType>(T)
                                         ? (Type)I64 : (Type)PtrTy));
        } else { AllReady = false; break; }
      }
      if (!AllReady) continue;
      auto isVoidLike = [](mlir::Operation *Op) {
        if (Op->getNumResults() == 0) return true;
        return Op->getNumResults() == 1 &&
               mlir::isa<mlir::NoneType>(Op->getResult(0).getType());
      };
      bool Void = isVoidLike(Call);
      /* Result type: i64 for matlab_symmat_rank; ptr for everything
       * else that returns. */
      Type Ret = VoidTy;
      if (!Void) Ret = (Name == "matlab_symmat_rank") ? (Type)I64 : (Type)PtrTy;
      B.setInsertionPoint(Call);
      auto Fn = rt(Name, Ret, Sig);
      llvm::SmallVector<Value, 6> CallArgs;
      for (auto V : Call->getOperands()) {
        if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType()))
          if (IT.getWidth() != 64)
            V = LLVM::SExtOp::create(B, Call->getLoc(), I64, V);
        CallArgs.push_back(V);
      }
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, CallArgs);
      if (!Void) {
        if (Call->getResult(0).getType() != Ret)
          Call->getResult(0).setType(Ret);
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
      }
      Call->erase();
      Changed = true;
      continue;
    }

    /* Phase 6 — Symbolic Math Toolbox. All matlab_sym_* runtime entries
     * follow a small set of signatures determined by the callee suffix.
     * Rather than spelling each out (~25 entries), pattern-match the
     * prefix and dispatch on the suffix. The (name, len) shape used by
     * matlab_sym_named / _from_str / _str2sym needs the same const_char
     * materialisation as matlab_string_from_literal above. */
    if (Name.starts_with("matlab_sym_")) {
      llvm::StringRef Suf = Name.substr(strlen("matlab_sym_"));
      auto materialiseConstChar = [&](Value Ch, Value &OutPtr,
                                        Value &OutLen) -> bool {
        Operation *Def = Ch.getDefiningOp();
        if (!isMatlabOp(Def, "matlab.const_char")) return false;
        auto VA = Def->getAttrOfType<StringAttr>("value");
        if (!VA) return false;
        StringRef Text = VA.getValue();
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
        OutPtr = LLVM::AddressOfOp::create(
            B, Call->getLoc(), PtrTy, Found.getSymName());
        OutLen = LLVM::ConstantOp::create(
            B, Call->getLoc(), I64,
            B.getI64IntegerAttr((int64_t)Text.size()));
        return true;
      };
      /* Group A: (const char*, int64_t) → matlab_sym* */
      if (Suf == "named" || Suf == "from_str" || Suf == "str2sym") {
        if (Call->getNumOperands() != 1 || Call->getNumResults() != 1)
          continue;
        Value PtrA, LenA;
        if (!materialiseConstChar(Call->getOperand(0), PtrA, LenA))
          continue;
        auto Fn = rt(Name, PtrTy, {PtrTy, I64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{PtrA, LenA});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      /* Group A2: (matlab_sym*, const char*, int64_t) → matlab_sym*
       * — assume / assumeAlso. Same materialisation shape as Group A
       * but the const_char is the second operand. */
      if (Suf == "assume" || Suf == "assumeAlso") {
        if (Call->getNumOperands() != 2 || Call->getNumResults() != 1)
          continue;
        Value SymA = Call->getOperand(0);
        if (SymA.getType() != PtrTy) continue;
        Value PtrA, LenA;
        if (!materialiseConstChar(Call->getOperand(1), PtrA, LenA))
          continue;
        auto Fn = rt(Name, PtrTy, {PtrTy, PtrTy, I64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{SymA, PtrA, LenA});
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      /* Group B: scalar producers — (f64) → ptr, (i64) → ptr. */
      if (Suf == "from_double" && Call->getNumOperands() == 1 &&
          Call->getNumResults() == 1 &&
          Call->getOperand(0).getType() == F64) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {F64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      if (Suf == "from_i64" && Call->getNumOperands() == 1 &&
          Call->getNumResults() == 1 &&
          mlir::isa<mlir::IntegerType>(Call->getOperand(0).getType())) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, {I64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      /* Group C: matlab_sym_disp(ptr) → void. */
      if (Suf == "disp" && Call->getNumOperands() == 1 &&
          Call->getOperand(0).getType() == PtrTy) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, VoidTy, {PtrTy});
        LLVM::CallOp::create(B, Call->getLoc(), Fn,
                              ValueRange{Call->getOperand(0)});
        Call->erase();
        Changed = true;
        continue;
      }
      /* Group D: matlab_sym_double(ptr) → f64. */
      if (Suf == "double" && Call->getNumOperands() == 1 &&
          Call->getOperand(0).getType() == PtrTy &&
          Call->getNumResults() == 1) {
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, F64, {PtrTy});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0)});
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      /* Group F: void-returning matlab_symmat_disp / matlab_symmat_set.
       * Group E gates on NumResults == 1 so void calls fall through;
       * handle the symmat side-effect ops here. matlab.call_builtin
       * carries a NoneType result for void calls (the unregistered op
       * always has at least one result slot), so we accept either
       * "no result" or "single NoneType result". */
      auto isVoidResult = [](mlir::Operation *Op) {
        if (Op->getNumResults() == 0) return true;
        return Op->getNumResults() == 1 &&
               mlir::isa<mlir::NoneType>(Op->getResult(0).getType());
      };
      if (isVoidResult(Call)) {
        bool AllReady = true;
        llvm::SmallVector<Type, 6> Sig;
        for (auto V : Call->getOperands()) {
          mlir::Type T = V.getType();
          if (T == PtrTy || T == F64 ||
              mlir::isa<mlir::IntegerType>(T)) {
            Sig.push_back(T == F64 ? F64
                                    : (mlir::isa<mlir::IntegerType>(T)
                                           ? (Type)I64 : (Type)PtrTy));
          } else { AllReady = false; break; }
        }
        if (AllReady) {
          B.setInsertionPoint(Call);
          auto Fn = rt(Name, VoidTy, Sig);
          llvm::SmallVector<Value, 6> CallArgs;
          for (auto V : Call->getOperands()) {
            if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType()))
              if (IT.getWidth() != 64)
                V = LLVM::SExtOp::create(B, Call->getLoc(), I64, V);
            CallArgs.push_back(V);
          }
          LLVM::CallOp::create(B, Call->getLoc(), Fn, CallArgs);
          /* If the call had a NoneType result slot, leave it dangling —
           * any consumer would already have failed verification. The
           * matlab.call_builtin op itself goes away with the erase. */
          Call->erase();
          Changed = true;
          continue;
        }
      }
      /* Group E: catch-all for any other matlab_sym_* — derive arg
       * types from operands at the call site, output is ptr. Covers
       * add/sub/mul/div/pow/neg/eq/diff/diff_n/int/int_def/simplify/
       * expand/factor/subs/solve_one and the _d / d_ mixed-mode
       * variants without enumerating each. */
      if (Call->getNumResults() == 1) {
        bool AllReady = true;
        llvm::SmallVector<Type, 6> Sig;
        for (auto V : Call->getOperands()) {
          mlir::Type T = V.getType();
          if (T == PtrTy || T == F64 ||
              mlir::isa<mlir::IntegerType>(T)) {
            Sig.push_back(T == F64 ? F64
                                    : (mlir::isa<mlir::IntegerType>(T)
                                           ? (Type)I64 : (Type)PtrTy));
          } else {
            AllReady = false; break;
          }
        }
        if (!AllReady) continue;
        B.setInsertionPoint(Call);
        auto Fn = rt(Name, PtrTy, Sig);
        /* Cast any non-i64 integer operand up to i64 to match the C ABI. */
        llvm::SmallVector<Value, 6> CallArgs;
        for (auto V : Call->getOperands()) {
          if (auto IT = mlir::dyn_cast<mlir::IntegerType>(V.getType()))
            if (IT.getWidth() != 64)
              V = LLVM::SExtOp::create(B, Call->getLoc(), I64, V);
          CallArgs.push_back(V);
        }
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, CallArgs);
        carryName(Call, NC);
        if (Call->getResult(0).getType() != PtrTy)
          Call->getResult(0).setType(PtrTy);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* Phase 3: matlab_obj_clone takes a single ptr (the source obj)
     * and returns a fresh ptr. Used at value-class assignment sites
     * to give each binding its own copy. The operand type may still
     * be `none` at this point — class-instance slots haven't been
     * retyped yet — so we wait for the operand to settle into ptr
     * before lowering. */
    if (Name == "matlab_obj_clone" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1) {
      Value Arg = Call->getOperand(0);
      if (Arg.getType() != PtrTy) continue;
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_obj_clone", PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Arg});
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* matlab_mat_clone_cow — copy-on-assign deep clone of a numeric matrix
     * (ptr -> ptr), emitted for `B = A` so a later `B(i)=v` cannot mutate A's
     * shared buffer.  Same ptr-settle wait as matlab_obj_clone. */
    if (Name == "matlab_mat_clone_cow" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 1) {
      Value Arg = Call->getOperand(0);
      /* Scalars flow as f64 and need no clone (value semantics already) —
       * pass the operand through.  A real heap matrix is ptr-typed; wait for
       * it to settle to ptr before emitting the deep clone. */
      if (Arg.getType() != PtrTy) {
        if (mlir::isa<mlir::Float64Type>(Arg.getType())) {
          Call->getResult(0).replaceAllUsesWith(Arg);
          Call->erase();
          Changed = true;
        }
        continue;
      }
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_mat_clone_cow", PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Arg});
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* readtable(path) -> matlab_table*; readmatrix(path) -> matlab_mat*.
     * Path is a matlab_string* (PtrTy); the C runtime parses CSV /
     * delimited text and returns the appropriate descriptor. Sema
     * leaves these untyped — retype the result before RAUW. */
    if ((Name == "readtable" || Name == "readmatrix") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      llvm::StringRef Rn = (Name == "readtable") ? "matlab_readtable"
                                                  : "matlab_readmatrix";
      auto Fn = rt(Rn, PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* §17.5 #6 — cross-dialect composition. `y = mflowlink_run(path)`
     * lowers to the C runtime in runtime/runtime_mflowlink_call.cpp.
     * Same signature pattern as readmatrix: matlab_string* path in,
     * matlab_mat* row vector of final logged-signal values out. */
    if (Name == "mflowlink_run" &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_mflowlink_run", PtrTy, {PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy)
        Call->getResult(0).setType(PtrTy);
      carryName(Call, NC);
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
      carryName(Call, NC);
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
        carryName(Call, NC);
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
        carryName(Call, NC);
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
      carryName(Call, NC);
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
    /* Rank-4 scalar store: A(i,j,k,l) = v on a matlab_matN binding.
     * Routes to matlab_subscript4_pstore_s, which is N-D-aware.  Accepts
     * either PtrTy or tensor<*xf64> base (early-tracking lane). */
    if (Name == "matlab_subscript4_pstore_s" &&
        Call->getNumOperands() == 6 &&
        (Call->getOperand(0).getType() == PtrTy ||
         mlir::isa<mlir::TensorType>(Call->getOperand(0).getType())) &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64 &&
        Call->getOperand(3).getType() == F64 &&
        Call->getOperand(4).getType() == F64 &&
        Call->getOperand(5).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_subscript4_pstore_s", VoidTy,
                   {PtrTy, F64, F64, F64, F64, F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
      Call->erase();
      Changed = true;
      continue;
    }
    /* Rank>=5 scalar store: matlab_subscriptN_pstore_s(base, i1..iN, v).
     * Pack the N indices into a stack int64_t[] and call the variadic
     * runtime helper matlab_subscriptN_pstore_s(void*, int64_t nidx,
     * const int64_t*, double v), which is generic to 16 dims.  #93. */
    if (Name == "matlab_subscriptN_pstore_s" &&
        Call->getNumOperands() >= 7 &&
        Call->getOperand(0).getType() == PtrTy) {
      unsigned NOps = Call->getNumOperands();
      unsigned NIdx = NOps - 2;
      bool Ok = (Call->getOperand(NOps - 1).getType() == F64);
      for (unsigned k = 1; Ok && k <= NIdx; ++k)
        if (Call->getOperand(k).getType() != F64) Ok = false;
      if (Ok) {
        B.setInsertionPoint(Call);
        Location Loc = Call->getLoc();
        Value Base = Call->getOperand(0);
        Value Rhs = Call->getOperand(NOps - 1);
        Value One = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
        auto ArrayTy = LLVM::LLVMArrayType::get(I64, NIdx);
        Value Buf = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrayTy, One,
                                            /*alignment=*/0);
        for (unsigned k = 0; k < NIdx; ++k) {
          Value Iv = arith::FPToSIOp::create(B, Loc, I64,
                                              Call->getOperand(k + 1));
          Value Idx = LLVM::ConstantOp::create(B, Loc, I64,
                                                B.getI64IntegerAttr(k));
          Value ElemPtr = LLVM::GEPOp::create(B, Loc, PtrTy, I64, Buf,
                                               ValueRange{Idx});
          LLVM::StoreOp::create(B, Loc, Iv, ElemPtr);
        }
        Value NIdxV = LLVM::ConstantOp::create(B, Loc, I64,
                                                B.getI64IntegerAttr(NIdx));
        auto Fn = rt("matlab_subscriptN_pstore_s", VoidTy,
                     {PtrTy, I64, PtrTy, F64});
        LLVM::CallOp::create(B, Loc, Fn, ValueRange{Base, NIdxV, Buf, Rhs});
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* A(:, :, k) = scalar  -> matlab_subscript3_pstore_s(A, k, v). */
    if (Name == "matlab_subscript3_pstore_s" &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_subscript3_pstore_s", VoidTy, {PtrTy, F64, F64});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
      Call->erase();
      Changed = true;
      continue;
    }
    /* A(:, :, k) = M  -> matlab_subscript3_pstore_m(A, k, M). */
    if (Name == "matlab_subscript3_pstore_m" &&
        Call->getNumOperands() == 3 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_subscript3_pstore_m", VoidTy, {PtrTy, F64, PtrTy});
      LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
      Call->erase();
      Changed = true;
      continue;
    }
    /* A(:, :, k) read -> matlab_subscript3_slice(A, k) : 2-D plane. */
    if (Name == "matlab_subscript3_slice" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_subscript3_slice", PtrTy, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
      if (Call->getResult(0).getType() != PtrTy) Call->getResult(0).setType(PtrTy);
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* cat(3, A, B[, C]) -> matlab_cat3_{2,3} : slice-major matlab_mat3.
     * matlab_cat3_append folds N>2 planes (append one plane to a mat3). */
    /* cat(4, …) — image-batch stack via the matN row-major-extended layout. */
    if ((Name == "matlab_cat4_2" || Name == "matlab_cat4_3" ||
         Name == "matlab_cat4_4") &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      llvm::SmallVector<Type, 4> ArgTys(Call->getNumOperands(), PtrTy);
      auto Fn = rt(Name.str(), PtrTy, ArgTys);
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if ((Name == "matlab_cat3_2" || Name == "matlab_cat3_3" ||
         Name == "matlab_cat3_append") &&
        Call->getNumResults() == 1) {
      bool allp = true;
      for (auto O : Call->getOperands()) if (O.getType() != PtrTy) allp = false;
      if (allp) {
        B.setInsertionPoint(Call);
        SmallVector<Type, 3> ats(Call->getNumOperands(), PtrTy);
        auto Fn = rt(Name, PtrTy, ats);
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, Call->getOperands());
        if (Call->getResult(0).getType() != PtrTy) Call->getResult(0).setType(PtrTy);
        carryName(Call, NC);
        Call->getResult(0).replaceAllUsesWith(NC.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }
    if (Name == "matlab_size3_dim" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_size3_dim", F64, {PtrTy, F64});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      carryName(Call, NC);
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
      carryName(Call, NC);
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
      carryName(Call, NC);
      /* Forward persistent_name / persistent_fn so the AOT emitters
       * (EmitC / EmitPython / EmitTypeScript) can recognise the call
       * as a persistent access and lower to idiomatic per-language
       * code instead of a verbatim runtime call. The LLVM/JIT path
       * ignores these attrs. */
      if (auto PN = Call->getAttrOfType<StringAttr>("persistent_name"))
        NC->setAttr("persistent_name", PN);
      if (auto PF = Call->getAttrOfType<StringAttr>("persistent_fn"))
        NC->setAttr("persistent_fn", PF);
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
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                            ValueRange{Call->getOperand(0),
                                       Call->getOperand(1)});
      if (auto PN = Call->getAttrOfType<StringAttr>("persistent_name"))
        NC->setAttr("persistent_name", PN);
      if (auto PF = Call->getAttrOfType<StringAttr>("persistent_fn"))
        NC->setAttr("persistent_fn", PF);
      Call->erase();
      Changed = true;
      continue;
    }

    /* Persistent-array runtime ABI: isempty / get_ptr / set_ptr.
     * The frontend emits these as matlab.call_builtin; the SV
     * pipeline (Stage F's LowerPersistentFiArrays) needs them as
     * llvm.call BUT with persistent_name/persistent_fn preserved
     * so the per-element rewrite can build user-readable register
     * names (`<name>_<k>` instead of `buf<idx>_<k>`). Without
     * explicit handling here the standard MLIR conversion path
     * would convert the call but drop the unregistered attrs. */
    if (Name == "matlab_persistent_isempty" &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == I32 &&
        Call->getNumResults() == 1 &&
        Call->getResult(0).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_persistent_isempty", F64, {I32});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      if (auto PN = Call->getAttrOfType<StringAttr>("persistent_name"))
        NC->setAttr("persistent_name", PN);
      if (auto PF = Call->getAttrOfType<StringAttr>("persistent_fn"))
        NC->setAttr("persistent_fn", PF);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_persistent_get_ptr" &&
        Call->getNumOperands() == 1 &&
        Call->getOperand(0).getType() == I32 &&
        Call->getNumResults() == 1 &&
        Call->getResult(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_persistent_get_ptr", PtrTy, {I32});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0)});
      if (auto PN = Call->getAttrOfType<StringAttr>("persistent_name"))
        NC->setAttr("persistent_name", PN);
      if (auto PF = Call->getAttrOfType<StringAttr>("persistent_fn"))
        NC->setAttr("persistent_fn", PF);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (Name == "matlab_persistent_set_ptr" &&
        Call->getNumOperands() == 2 &&
        Call->getOperand(0).getType() == I32 &&
        Call->getOperand(1).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_persistent_set_ptr", VoidTy, {I32, PtrTy});
      auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      ValueRange{Call->getOperand(0),
                                                 Call->getOperand(1)});
      if (auto PN = Call->getAttrOfType<StringAttr>("persistent_name"))
        NC->setAttr("persistent_name", PN);
      if (auto PF = Call->getAttrOfType<StringAttr>("persistent_fn"))
        NC->setAttr("persistent_fn", PF);
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
        {"eig",   "matlab_eig_V",   "matlab_eig_D"},
        {"qr",    "matlab_qr_Q",    "matlab_qr_R"},
        {"lu",    "matlab_lu_L",    "matlab_lu_U"},
        {"schur", "matlab_schur_U", "matlab_schur_T"},
        /* [H, P] = hess(A) — H upper Hessenberg, P orthogonal with
         * P' A P = H. Order matches MATLAB's first-output-is-H. */
        {"hess",  "matlab_hess_H",  "matlab_hess_P"},
        /* [v, i] = min/max/sort(A) — value/sorted (output 0) + 1-based
         * index/permutation (output 1). */
        {"min",   "matlab_min",     "matlab_min_idx"},
        {"max",   "matlab_max",     "matlab_max_idx"},
        {"sort",  "matlab_sort",    "matlab_sort_idx"},
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

    /* ===== Wavelet Toolbox multi-return builtins ======================
     * Each output is an independent runtime call sharing the *coerced*
     * operands.  A const_char family/option arg is materialised into a
     * matlab_string* (the pde_table path, hoisted here); f64 stays f64;
     * tensor/ptr is bridged to llvm.ptr.  The runtime entry signature is
     * inferred from the coerced operand types, so a per-name table only
     * needs the operand count and the ordered output runtime names. */
    {
      struct WMR { StringRef name; unsigned nops; SmallVector<StringRef, 4> outs; };
      static const WMR wmret[] = {
        {"wfilters", 1, {"matlab_wavelet_wf_lod", "matlab_wavelet_wf_hid",
                         "matlab_wavelet_wf_lor", "matlab_wavelet_wf_hir"}},
        {"dwt",      2, {"matlab_wavelet_dwt_cA", "matlab_wavelet_dwt_cD"}},
        {"wavedec",  3, {"matlab_wavelet_wavedec_C", "matlab_wavelet_wavedec_L"}},
        {"wnoise",   3, {"matlab_wavelet_wnoise_x3", "matlab_wavelet_wnoise_xn3"}},
        {"cwt",      2, {"matlab_wavelet_cwt_mag", "matlab_wavelet_cwt_f"}},
        {"dwt2",     2, {"matlab_wavelet_dwt2_cA", "matlab_wavelet_dwt2_cH",
                         "matlab_wavelet_dwt2_cV", "matlab_wavelet_dwt2_cD"}},
        {"wavedec2", 3, {"matlab_wavelet_wavedec2_C", "matlab_wavelet_wavedec2_S"}},
      };
      bool wmatched = false;
      for (const auto &E : wmret) {
        if (Name != E.name) continue;
        if (Call->getNumOperands() != E.nops) continue;
        if (Call->getNumResults() == 0 ||
            Call->getNumResults() > E.outs.size()) continue;
        /* coerce each operand once. */
        bool ok = true;
        SmallVector<Value, 4> coerced;
        SmallVector<Type, 4> sig;
        SmallVector<Operation *, 2> deadLits;
        B.setInsertionPoint(Call);
        for (unsigned k = 0; k < E.nops; ++k) {
          Value V = Call->getOperand(k);
          Operation *Def = V.getDefiningOp();
          if (isMatlabOp(Def, "matlab.const_char")) {
            auto VA = Def->getAttrOfType<StringAttr>("value");
            if (!VA) { ok = false; break; }
            StringRef Text = VA.getValue();
            LLVM::GlobalOp Found;
            for (auto G : Mod.getOps<LLVM::GlobalOp>()) {
              if (!G.getConstant()) continue;
              auto At = mlir::dyn_cast_or_null<StringAttr>(G.getValueAttr());
              if (At && At.getValue() == Text) { Found = G; break; }
            }
            if (!Found) {
              OpBuilder::InsertionGuard IG(B);
              B.setInsertionPointToStart(Mod.getBody());
              auto ArrayTy = LLVM::LLVMArrayType::get(
                  IntegerType::get(Ctx, 8), static_cast<unsigned>(Text.size()));
              unsigned N = 0; std::string SymName;
              do { SymName = ("__matlab_str_w" + std::to_string(N++)); }
              while (Mod.lookupSymbol(SymName));
              Found = LLVM::GlobalOp::create(B, Mod.getLoc(), ArrayTy, true,
                  LLVM::Linkage::Internal, SymName, StringAttr::get(Ctx, Text));
            }
            Value Addr = LLVM::AddressOfOp::create(B, Call->getLoc(), PtrTy,
                                                   Found.getSymName());
            Value LenV = LLVM::ConstantOp::create(B, Call->getLoc(), I64,
                B.getI64IntegerAttr(static_cast<int64_t>(Text.size())));
            auto FnS = rt("matlab_string_from_literal", PtrTy, {PtrTy, I64});
            coerced.push_back(LLVM::CallOp::create(B, Call->getLoc(), FnS,
                                  ValueRange{Addr, LenV}).getResult());
            sig.push_back(PtrTy);
            deadLits.push_back(Def);
          } else if (V.getType() == F64) {
            coerced.push_back(V); sig.push_back(F64);
          } else if (V.getType() == PtrTy || isTensorLike(V.getType())) {
            if (V.getType() != PtrTy) {
              auto Cast = mlir::UnrealizedConversionCastOp::create(
                  B, Call->getLoc(), PtrTy, V);
              coerced.push_back(Cast.getResult(0));
            } else coerced.push_back(V);
            sig.push_back(PtrTy);
          } else { ok = false; break; }
        }
        if (!ok) continue;
        for (unsigned r = 0; r < Call->getNumResults(); ++r) {
          auto Fn = rt(E.outs[r], PtrTy, sig);
          auto Cr = LLVM::CallOp::create(B, Call->getLoc(), Fn, coerced);
          Call->getResult(r).replaceAllUsesWith(Cr.getResult());
        }
        Call->erase();
        for (Operation *D : deadLits) if (D->use_empty()) D->erase();
        Changed = true; wmatched = true;
        break;
      }
      if (wmatched) continue;
    }

    /* [X, Y] = meshgrid(x[, y]) / ndgrid(x[, y]): two ptr results, one
     * or two ptr inputs. Mirror the single-arg form (meshgrid(x) ==
     * meshgrid(x, x)) when only one operand was supplied — the runtime
     * entries accept y == NULL and re-use x. Two single-output calls
     * keep the runtime ABI uniform with the rest of the multi-return
     * builtins above. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Call->getNumResults() == 2 &&
        (Name == "meshgrid" || Name == "ndgrid") &&
        (Call->getNumOperands() == 1 || Call->getNumOperands() == 2)) {
      bool TypesOK = true;
      for (unsigned i = 0; i < Call->getNumOperands(); ++i)
        if (Call->getOperand(i).getType() != PtrTy &&
            !isTensorLike(Call->getOperand(i).getType())) {
          TypesOK = false; break;
        }
      if (TypesOK) {
        StringRef F0 = (Name == "meshgrid") ? StringRef("matlab_meshgrid_X")
                                            : StringRef("matlab_ndgrid_X");
        StringRef F1 = (Name == "meshgrid") ? StringRef("matlab_meshgrid_Y")
                                            : StringRef("matlab_ndgrid_Y");
        B.setInsertionPoint(Call);
        Value X = Call->getOperand(0);
        Value Y;
        if (Call->getNumOperands() == 2) Y = Call->getOperand(1);
        else Y = LLVM::ZeroOp::create(B, Call->getLoc(), PtrTy);
        auto Fn0 = rt(F0, PtrTy, {PtrTy, PtrTy});
        auto Fn1 = rt(F1, PtrTy, {PtrTy, PtrTy});
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn0,
                                        ValueRange{X, Y});
        auto C1 = LLVM::CallOp::create(B, Call->getLoc(), Fn1,
                                        ValueRange{X, Y});
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->getResult(1).replaceAllUsesWith(C1.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* [X, Y, Z] = peaks(N) — canonical MATLAB 3-D demo surface. Single
     * f64 argument, three ptr returns. Splits into three single-output
     * runtime entries mirroring the meshgrid pattern. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Name == "peaks" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 3 &&
        Call->getOperand(0).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fx = rt("matlab_peaks_X", PtrTy, {F64});
      auto Fy = rt("matlab_peaks_Y", PtrTy, {F64});
      auto Fz = rt("matlab_peaks_Z", PtrTy, {F64});
      SmallVector<Value, 1> CA{Call->getOperand(0)};
      auto Cx = LLVM::CallOp::create(B, Call->getLoc(), Fx, CA);
      auto Cy = LLVM::CallOp::create(B, Call->getLoc(), Fy, CA);
      auto Cz = LLVM::CallOp::create(B, Call->getLoc(), Fz, CA);
      Call->getResult(0).replaceAllUsesWith(Cx.getResult());
      Call->getResult(1).replaceAllUsesWith(Cy.getResult());
      Call->getResult(2).replaceAllUsesWith(Cz.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* ode_events — IVP solver with event detection. 5-result form
     * `[t, y, te, ye, ie] = ode_events(@f, tspan, y0, @evt)`.
     * 4 operands: ptr (f), ptr (tspan), f64 (y0), ptr (evt).
     * Splits into five matlab_ode_events_{t,y,te,ye,ie} runtime calls
     * sharing a thread-local cache. */
    if (NA && NA.getValue().getSExtValue() == 5 &&
        Name == "ode_events" &&
        Call->getNumOperands() == 4 && Call->getNumResults() == 5 &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy &&
        Call->getOperand(2).getType() == F64 &&
        Call->getOperand(3).getType() == PtrTy) {
      static const char *Suffixes[] = { "t", "y", "te", "ye", "ie" };
      B.setInsertionPoint(Call);
      for (int i = 0; i < 5; ++i) {
        std::string FnName =
            std::string("matlab_ode_events_") + Suffixes[i];
        auto Fn = rt(FnName, PtrTy, {PtrTy, PtrTy, F64, PtrTy});
        auto Ci = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        Call->getResult(i).replaceAllUsesWith(Ci.getResult());
      }
      Call->erase();
      Changed = true;
      continue;
    }

    /* IIR design — multi-return forms (lowpass + band variants).
     *
     *   [b, a] = butter(n, Wn)              LP   (Wn scalar)
     *   [b, a] = butter(n, [W1 W2])         BP   (Wn 2-elem vector)
     *   [b, a] = butter(n, Wn, 'high')      HP   (Wn scalar + 'high')
     *   [b, a] = butter(n, [W1 W2], 'stop') BS   (Wn 2-elem + 'stop')
     *   ...same shape pattern for cheby1 (extra Rp at position 1) and
     *      cheby2 (extra Rs at position 1).
     *
     * Bandpass / bandstop `[W1 W2]` is matched against the
     * `matlab.concat_row` defining op so we can extract two scalar
     * f64s and call the runtime entry directly. The optional
     * 'high' / 'stop' trailing string is parsed via the const_char op.
     */
    auto isIIRFamily = (Name == "butter" || Name == "cheby1" ||
                       Name == "cheby2");
    auto tryIIRDispatch = [&]() -> bool {
      if (!(NA && NA.getValue().getSExtValue() == 2 && isIIRFamily &&
            Call->getNumResults() == 2)) return false;
      bool isButter = (Name == "butter");
      int nopOk = Call->getNumOperands();
      bool nopShapeOk = isButter ? (nopOk == 2 || nopOk == 3)
                                  : (nopOk == 3 || nopOk == 4);
      if (!nopShapeOk) return false;
      if (Call->getOperand(0).getType() != F64) return false;
      Value RArg, WnArg, StrArg;
      if (isButter) {
        WnArg = Call->getOperand(1);
        if (nopOk == 3) StrArg = Call->getOperand(2);
      } else {
        if (Call->getOperand(1).getType() != F64) return false;
        RArg  = Call->getOperand(1);
        WnArg = Call->getOperand(2);
        if (nopOk == 4) StrArg = Call->getOperand(3);
      }
      bool wnIsScalar = (WnArg.getType() == F64);
      bool wnIsVec2   = false;
      Value W1, W2;
      if (auto Tt = mlir::dyn_cast<RankedTensorType>(WnArg.getType())) {
        if (Tt.getElementType().isF64() && Tt.getNumElements() == 2) {
          Operation *D = WnArg.getDefiningOp();
          if (!D || D->getName().getStringRef() != "matlab.concat_row" ||
              D->getNumOperands() != 2 ||
              D->getOperand(0).getType() != F64 ||
              D->getOperand(1).getType() != F64)
            return false;          /* variable [W1 W2] not yet wired */
          wnIsVec2 = true;
          W1 = D->getOperand(0);
          W2 = D->getOperand(1);
        }
      }
      if (!wnIsScalar && !wnIsVec2) return false;
      StringRef Tag;
      if (StrArg) {
        auto Tt = mlir::dyn_cast<RankedTensorType>(StrArg.getType());
        if (!Tt || !Tt.getElementType().isInteger(8)) return false;
        Operation *D = StrArg.getDefiningOp();
        if (!D || D->getName().getStringRef() != "matlab.const_char")
          return false;
        auto VA = D->getAttrOfType<StringAttr>("value");
        if (!VA) return false;
        Tag = VA.getValue();
      }
      enum { LP_T, HP_T, BP_T, BS_T } ft;
      if      (wnIsScalar && Tag.empty())   ft = LP_T;
      else if (wnIsVec2   && Tag.empty())   ft = BP_T;
      else if (wnIsScalar && Tag == "high") ft = HP_T;
      else if (wnIsVec2   && Tag == "stop") ft = BS_T;
      else                                  return false;
      const char *suf = (ft == LP_T) ? ""
                       : (ft == HP_T) ? "_hp"
                       : (ft == BP_T) ? "_bp"
                                      : "_bs";
      std::string Fb = "matlab_" + Name.str() + suf + "_b";
      std::string Fa = "matlab_" + Name.str() + suf + "_a";
      llvm::SmallVector<Value, 4> Args;
      llvm::SmallVector<Type, 4>  Sig;
      Args.push_back(Call->getOperand(0)); Sig.push_back(F64);
      if (!isButter) { Args.push_back(RArg); Sig.push_back(F64); }
      if (ft == LP_T || ft == HP_T) {
        Args.push_back(WnArg); Sig.push_back(F64);
      } else {
        Args.push_back(W1); Args.push_back(W2);
        Sig.push_back(F64); Sig.push_back(F64);
      }
      B.setInsertionPoint(Call);
      auto Fb_fn = rt(Fb, PtrTy, Sig);
      auto Fa_fn = rt(Fa, PtrTy, Sig);
      auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb_fn, Args);
      auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa_fn, Args);
      Call->getResult(0).replaceAllUsesWith(Cb.getResult());
      Call->getResult(1).replaceAllUsesWith(Ca.getResult());
      Call->erase();
      return true;
    };
    if (tryIIRDispatch()) { Changed = true; continue; }

    /* [b, a] = besself(n, Wo) — analog Bessel. Two f64 args, two ptr
     * results; splits into matlab_besself_{b,a}. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "besself" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fb = rt("matlab_besself_b", PtrTy, {F64, F64});
      auto Fa = rt("matlab_besself_a", PtrTy, {F64, F64});
      auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb,
                                      Call->getOperands());
      auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa,
                                      Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(Cb.getResult());
      Call->getResult(1).replaceAllUsesWith(Ca.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* [b, a] = iirnotch(w0, bw) / iirpeak(w0, bw) — DSP Tier-2 second-order
     * notch / peak biquad designers. Two f64 args, two ptr results;
     * splits into matlab_dsp_<name>_{b,a} (besself-style). */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        (Name == "iirnotch" || Name == "iirpeak") &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      std::string Fbn = "matlab_dsp_" + Name.str() + "_b";
      std::string Fan = "matlab_dsp_" + Name.str() + "_a";
      auto Fb = rt(Fbn, PtrTy, {F64, F64});
      auto Fa = rt(Fan, PtrTy, {F64, F64});
      auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, Call->getOperands());
      auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(Cb.getResult());
      Call->getResult(1).replaceAllUsesWith(Ca.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* [n, Wn] = buttord(Wp, Ws, Rp, Rs) / cheb1ord(...). 4 f64 args,
     * 2 f64 results. Splits into matlab_<name>_n / _Wn. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        (Name == "buttord" || Name == "cheb1ord" || Name == "cheb2ord") &&
        Call->getNumOperands() == 4 && Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64 &&
        Call->getOperand(2).getType() == F64 &&
        Call->getOperand(3).getType() == F64) {
      B.setInsertionPoint(Call);
      std::string Fn_n  = "matlab_" + Name.str() + "_n";
      std::string Fn_Wn = "matlab_" + Name.str() + "_Wn";
      auto Fnn  = rt(Fn_n,  F64, {F64, F64, F64, F64});
      auto Fnwn = rt(Fn_Wn, F64, {F64, F64, F64, F64});
      auto Cn  = LLVM::CallOp::create(B, Call->getLoc(), Fnn,
                                       Call->getOperands());
      auto Cwn = LLVM::CallOp::create(B, Call->getLoc(), Fnwn,
                                       Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(Cn.getResult());
      Call->getResult(1).replaceAllUsesWith(Cwn.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "freqz" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      /* b and a may arrive as ptr or tensor; defer until both are ptr
       * (matrix-slot lowering handles tensor → ptr). */
      auto t0 = Call->getOperand(0).getType();
      auto t1 = Call->getOperand(1).getType();
      if (t0 == PtrTy && t1 == PtrTy) {
        B.setInsertionPoint(Call);
        auto Fh = rt("matlab_freqz_h", PtrTy, {PtrTy, PtrTy, F64});
        auto Fw = rt("matlab_freqz_w", PtrTy, {PtrTy, PtrTy, F64});
        auto Ch = LLVM::CallOp::create(B, Call->getLoc(), Fh,
                                        Call->getOperands());
        auto Cw = LLVM::CallOp::create(B, Call->getLoc(), Fw,
                                        Call->getOperands());
        Call->getResult(0).replaceAllUsesWith(Ch.getResult());
        Call->getResult(1).replaceAllUsesWith(Cw.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* Helper for the §2.1 multi-LHS dispatchers below: take an operand
     * that should arrive as a matrix ptr but might be f64 (scalar
     * literal like `[1]` collapsed to scalar) or tensor<...xf64>; box
     * f64 → matlab_mat * via matlab_mat_from_scalar; pass tensor through
     * (the matrix-slot lowering converts to ptr later). Returns the
     * boxed/unmodified value, or nullopt if the operand type is not
     * acceptable. */
    auto boxAsPtr = [&](Value V) -> Value {
      Type T = V.getType();
      if (T == PtrTy || isTensorLike(T)) return V;
      if (T == F64) {
        auto Fn = rt("matlab_mat_from_scalar", PtrTy, {F64});
        B.setInsertionPoint(Call);
        return LLVM::CallOp::create(B, Call->getLoc(), Fn, {V}).getResult();
      }
      return Value{};
    };

    /* [mag, phase] = bode_tf(b, a, w) — 3 ptr operands, 2 ptr returns.
     * Splits into matlab_bode_tf_{mag,phase}. CST Tier 2.4 follow-on. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "bode_tf" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      if (V0 && V1 && V2) {
        B.setInsertionPoint(Call);
        auto Fm = rt("matlab_bode_tf_mag",   PtrTy, {PtrTy, PtrTy, PtrTy});
        auto Fp = rt("matlab_bode_tf_phase", PtrTy, {PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 3> CA{V0, V1, V2};
        auto Cm = LLVM::CallOp::create(B, Call->getLoc(), Fm, CA);
        auto Cp = LLVM::CallOp::create(B, Call->getLoc(), Fp, CA);
        Call->getResult(0).replaceAllUsesWith(Cm.getResult());
        Call->getResult(1).replaceAllUsesWith(Cp.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [mag, phase] = bode_ss(A, B, C, D, w) — 5 ptr operands, 2 ptr
     * returns. Splits into matlab_bode_ss_{mag,phase}. CST Tier 2.4. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "bode_ss" && Call->getNumOperands() == 5 &&
        Call->getNumResults() == 2) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      Value V3 = boxAsPtr(Call->getOperand(3));
      Value V4 = boxAsPtr(Call->getOperand(4));
      if (V0 && V1 && V2 && V3 && V4) {
        B.setInsertionPoint(Call);
        auto Fm = rt("matlab_bode_ss_mag",   PtrTy,
                     {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        auto Fp = rt("matlab_bode_ss_phase", PtrTy,
                     {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 5> CA{V0, V1, V2, V3, V4};
        auto Cm = LLVM::CallOp::create(B, Call->getLoc(), Fm, CA);
        auto Cp = LLVM::CallOp::create(B, Call->getLoc(), Fp, CA);
        Call->getResult(0).replaceAllUsesWith(Cm.getResult());
        Call->getResult(1).replaceAllUsesWith(Cp.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [Ad, Bd] = c2d(A, B, Ts) — same shape as bilinear (2 ptr + 1 f64,
     * 2 ptr returns). CST Tier 2.2. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "c2d" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_c2d_Ad", PtrTy, {PtrTy, PtrTy, F64});
        auto Fb = rt("matlab_c2d_Bd", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [A, B] = d2c(Ad, Bd, Ts) — ZOH discrete->continuous, inverse of c2d.
     * Same 2-ptr + scalar shape; routes to matlab_d2c_{A,B}. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "d2c" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_d2c_A", PtrTy, {PtrTy, PtrTy, F64});
        auto Fb = rt("matlab_d2c_B", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [Ad, Bd] = c2d_tustin(A, B, Ts) — Tustin (bilinear) discretisation.
     * Same shape as c2d above; routes to matlab_c2d_tustin_{Ad,Bd}. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "c2d_tustin" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_c2d_tustin_Ad", PtrTy, {PtrTy, PtrTy, F64});
        auto Fb = rt("matlab_c2d_tustin_Bd", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [A, B] = d2c_tustin(Ad, Bd, Ts) — inverse Tustin. Same shape. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "d2c_tustin" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_d2c_tustin_A", PtrTy, {PtrTy, PtrTy, F64});
        auto Fb = rt("matlab_d2c_tustin_B", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [L, P] = kalman(A, G, C, Qn, Rn) — gain + Riccati covariance.
     * Routes to matlab_kalman_L + matlab_kalman_P (or kalmd_* for the
     * discrete variant). Both helpers take the same 5-arg signature. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        (Name == "kalman" || Name == "kalmd") &&
        Call->getNumOperands() == 5 && Call->getNumResults() == 2) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      Value V3 = boxAsPtr(Call->getOperand(3));
      Value V4 = boxAsPtr(Call->getOperand(4));
      if (V0 && V1 && V2 && V3 && V4) {
        B.setInsertionPoint(Call);
        const char *lFn = (Name == "kalman") ? "matlab_kalman_L" : "matlab_kalmd_L";
        const char *pFn = (Name == "kalman") ? "matlab_kalman_P" : "matlab_kalmd_P";
        auto Fl = rt(lFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        auto Fp = rt(pFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 5> CA{V0, V1, V2, V3, V4};
        auto Cl = LLVM::CallOp::create(B, Call->getLoc(), Fl, CA);
        auto Cp = LLVM::CallOp::create(B, Call->getLoc(), Fp, CA);
        Call->getResult(0).replaceAllUsesWith(Cl.getResult());
        Call->getResult(1).replaceAllUsesWith(Cp.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [Acl, Bcl, Ccl] = feedback_ss / series_ss / parallel_ss
     * (A1, B1, C1, A2, B2, C2) — strictly-proper interconnection
     * primitives. Each routes to its own matlab_<name>_{A,B,C}. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        (Name == "feedback_ss" || Name == "series_ss" ||
         Name == "parallel_ss" || Name == "append_ss") &&
        Call->getNumOperands() == 6 && Call->getNumResults() == 3) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      Value V3 = boxAsPtr(Call->getOperand(3));
      Value V4 = boxAsPtr(Call->getOperand(4));
      Value V5 = boxAsPtr(Call->getOperand(5));
      if (V0 && V1 && V2 && V3 && V4 && V5) {
        B.setInsertionPoint(Call);
        std::string aFn = "matlab_" + std::string(Name) + "_A";
        std::string bFn = "matlab_" + std::string(Name) + "_B";
        std::string cFn = "matlab_" + std::string(Name) + "_C";
        auto Fa = rt(aFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        auto Fb = rt(bFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        auto Fc = rt(cFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 6> CA{V0, V1, V2, V3, V4, V5};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        auto Cc = LLVM::CallOp::create(B, Call->getLoc(), Fc, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->getResult(2).replaceAllUsesWith(Cc.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [AA, BB, Q, Z] = qz(A, B) — generalised Schur form of the matrix
     * pencil A − λ·B. Routes to four matlab_qz_{AA,BB,Q,Z} entries that
     * each recompute the full decomposition (same stateless pattern as
     * schur_U / schur_T). v1 path requires B invertible; the singular-
     * pencil case returns 0×0 from each entry (deferred). */
    if (NA && NA.getValue().getSExtValue() == 4 &&
        Name == "qz" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 4) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Faa = rt("matlab_qz_AA", PtrTy, {PtrTy, PtrTy});
        auto Fbb = rt("matlab_qz_BB", PtrTy, {PtrTy, PtrTy});
        auto Fq  = rt("matlab_qz_Q",  PtrTy, {PtrTy, PtrTy});
        auto Fz  = rt("matlab_qz_Z",  PtrTy, {PtrTy, PtrTy});
        SmallVector<Value, 2> CA{V0, V1};
        auto Caa = LLVM::CallOp::create(B, Call->getLoc(), Faa, CA);
        auto Cbb = LLVM::CallOp::create(B, Call->getLoc(), Fbb, CA);
        auto Cq  = LLVM::CallOp::create(B, Call->getLoc(), Fq,  CA);
        auto Cz  = LLVM::CallOp::create(B, Call->getLoc(), Fz,  CA);
        Call->getResult(0).replaceAllUsesWith(Caa.getResult());
        Call->getResult(1).replaceAllUsesWith(Cbb.getResult());
        Call->getResult(2).replaceAllUsesWith(Cq.getResult());
        Call->getResult(3).replaceAllUsesWith(Cz.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [Ar, Br, Cr] = balred(A, B, C, k) — k-state truncated balanced
     * realisation. Routes to matlab_balred_{A,B,C}, all of which take
     * the same (A, B, C, k) args. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Name == "balred" && Call->getNumOperands() == 4 &&
        Call->getNumResults() == 3 &&
        Call->getOperand(3).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      if (V0 && V1 && V2) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_balred_A", PtrTy, {PtrTy, PtrTy, PtrTy, F64});
        auto Fb = rt("matlab_balred_B", PtrTy, {PtrTy, PtrTy, PtrTy, F64});
        auto Fc = rt("matlab_balred_C", PtrTy, {PtrTy, PtrTy, PtrTy, F64});
        SmallVector<Value, 4> CA{V0, V1, V2, Call->getOperand(3)};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        auto Cc = LLVM::CallOp::create(B, Call->getLoc(), Fc, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->getResult(2).replaceAllUsesWith(Cc.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [K, S, e] = lqr(A, B, Q, R) — full continuous LQR shape:
     *   K = matlab_lqr(A, B, Q, R)
     *   S = matlab_care(A, B, Q, R)   (the Riccati solution)
     *   e = matlab_lqr_e(A, B, Q, R)  (closed-loop poles eig(A − B·K))
     * The 2-return shape [K, S] = lqr(...) is also handled. */
    if (NA && (NA.getValue().getSExtValue() == 2 ||
               NA.getValue().getSExtValue() == 3) &&
        (Name == "lqr" || Name == "dlqr") &&
        Call->getNumOperands() == 4 &&
        (Call->getNumResults() == 2 || Call->getNumResults() == 3)) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      Value V3 = boxAsPtr(Call->getOperand(3));
      if (V0 && V1 && V2 && V3) {
        B.setInsertionPoint(Call);
        const char *kFn  = (Name == "lqr") ? "matlab_lqr"  : "matlab_dlqr";
        const char *sFn  = (Name == "lqr") ? "matlab_care" : "matlab_dare";
        const char *eFn  = (Name == "lqr") ? "matlab_lqr_e" : "matlab_dlqr_e";
        auto Fk = rt(kFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy});
        auto Fs = rt(sFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 4> CA{V0, V1, V2, V3};
        auto Ck = LLVM::CallOp::create(B, Call->getLoc(), Fk, CA);
        auto Cs = LLVM::CallOp::create(B, Call->getLoc(), Fs, CA);
        Call->getResult(0).replaceAllUsesWith(Ck.getResult());
        Call->getResult(1).replaceAllUsesWith(Cs.getResult());
        if (Call->getNumResults() == 3) {
          auto Fe = rt(eFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy});
          auto Ce = LLVM::CallOp::create(B, Call->getLoc(), Fe, CA);
          Call->getResult(2).replaceAllUsesWith(Ce.getResult());
        }
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [X, K, L] = care(A, B, Q, R) / dare(...) — full Riccati shape:
     *   X = matlab_care/matlab_dare         (the stabilising solution)
     *   K = matlab_lqr/matlab_dlqr          (R⁻¹B'X gain, R = Schur complement
     *                                        for discrete)
     *   L = matlab_lqr_e/matlab_dlqr_e      (closed-loop poles eig(A − B·K) /
     *                                        eig(Ad − Bd·K))
     * The 2-return [X, K] form is also handled. */
    if (NA && (NA.getValue().getSExtValue() == 2 ||
               NA.getValue().getSExtValue() == 3) &&
        (Name == "care" || Name == "dare") &&
        Call->getNumOperands() == 4 &&
        (Call->getNumResults() == 2 || Call->getNumResults() == 3)) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      Value V3 = boxAsPtr(Call->getOperand(3));
      if (V0 && V1 && V2 && V3) {
        B.setInsertionPoint(Call);
        const char *xFn = (Name == "care") ? "matlab_care"  : "matlab_dare";
        const char *kFn = (Name == "care") ? "matlab_lqr"   : "matlab_dlqr";
        const char *lFn = (Name == "care") ? "matlab_lqr_e" : "matlab_dlqr_e";
        auto Fx = rt(xFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy});
        auto Fk = rt(kFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 4> CA{V0, V1, V2, V3};
        auto Cx = LLVM::CallOp::create(B, Call->getLoc(), Fx, CA);
        auto Ck = LLVM::CallOp::create(B, Call->getLoc(), Fk, CA);
        Call->getResult(0).replaceAllUsesWith(Cx.getResult());
        Call->getResult(1).replaceAllUsesWith(Ck.getResult());
        if (Call->getNumResults() == 3) {
          auto Fl = rt(lFn, PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy});
          auto Cl = LLVM::CallOp::create(B, Call->getLoc(), Fl, CA);
          Call->getResult(2).replaceAllUsesWith(Cl.getResult());
        }
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [num, den] = pade(τ, n) — Padé approximation of e^{-τs}. Two
     * f64 args, two ptr returns. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "pade" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_pade_num", PtrTy, {F64, F64});
      auto Fd = rt("matlab_pade_den", PtrTy, {F64, F64});
      SmallVector<Value, 2> CA{Call->getOperand(0), Call->getOperand(1)};
      auto Cn = LLVM::CallOp::create(B, Call->getLoc(), Fn, CA);
      auto Cd = LLVM::CallOp::create(B, Call->getLoc(), Fd, CA);
      Call->getResult(0).replaceAllUsesWith(Cn.getResult());
      Call->getResult(1).replaceAllUsesWith(Cd.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* [As, Bs, Cs] = sminreal(A, B, C) — structural minimal
     * realisation. Pure boolean-graph analysis on the structure
     * of A, B, C. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Name == "sminreal" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 3) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      if (V0 && V1 && V2) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_sminreal_A", PtrTy, {PtrTy, PtrTy, PtrTy});
        auto Fb = rt("matlab_sminreal_B", PtrTy, {PtrTy, PtrTy, PtrTy});
        auto Fc = rt("matlab_sminreal_C", PtrTy, {PtrTy, PtrTy, PtrTy});
        SmallVector<Value, 3> CA{V0, V1, V2};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        auto Cc = LLVM::CallOp::create(B, Call->getLoc(), Fc, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->getResult(2).replaceAllUsesWith(Cc.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [Ar, Br, Cr] = modred(A, B, C, elim, method_id) — modal
     * residualisation. method_id is 0 = Truncate, 1 = MatchDC. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Name == "modred" && Call->getNumOperands() == 5 &&
        Call->getNumResults() == 3 &&
        Call->getOperand(4).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      Value V2 = boxAsPtr(Call->getOperand(2));
      Value V3 = boxAsPtr(Call->getOperand(3));
      if (V0 && V1 && V2 && V3) {
        B.setInsertionPoint(Call);
        auto Fa = rt("matlab_modred_A", PtrTy,
                     {PtrTy, PtrTy, PtrTy, PtrTy, F64});
        auto Fb = rt("matlab_modred_B", PtrTy,
                     {PtrTy, PtrTy, PtrTy, PtrTy, F64});
        auto Fc = rt("matlab_modred_C", PtrTy,
                     {PtrTy, PtrTy, PtrTy, PtrTy, F64});
        SmallVector<Value, 5> CA{V0, V1, V2, V3, Call->getOperand(4)};
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        auto Cc = LLVM::CallOp::create(B, Call->getLoc(), Fc, CA);
        Call->getResult(0).replaceAllUsesWith(Ca.getResult());
        Call->getResult(1).replaceAllUsesWith(Cb.getResult());
        Call->getResult(2).replaceAllUsesWith(Cc.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [bb, aa] = thiran(D, n). Fractional-delay all-pass FIR. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "thiran" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == F64) {
      B.setInsertionPoint(Call);
      auto Fb = rt("matlab_thiran_b", PtrTy, {F64, F64});
      auto Fa = rt("matlab_thiran_a", PtrTy, {F64, F64});
      SmallVector<Value, 2> CA{Call->getOperand(0), Call->getOperand(1)};
      auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
      auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
      Call->getResult(0).replaceAllUsesWith(Cb.getResult());
      Call->getResult(1).replaceAllUsesWith(Ca.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* [num_r, den_r] = minreal(num, den, tol) — pole-zero
     * cancellation on the transfer-function form. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "minreal" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_minreal_tf_num", PtrTy, {PtrTy, PtrTy, F64});
        auto Fd = rt("matlab_minreal_tf_den", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Cn = LLVM::CallOp::create(B, Call->getLoc(), Fn, CA);
        auto Cd = LLVM::CallOp::create(B, Call->getLoc(), Fd, CA);
        Call->getResult(0).replaceAllUsesWith(Cn.getResult());
        Call->getResult(1).replaceAllUsesWith(Cd.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [bd, ad] = bilinear(b, a, fs). Splits into matlab_bilinear_{b,a}. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "bilinear" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fb = rt("matlab_bilinear_b", PtrTy, {PtrTy, PtrTy, F64});
        auto Fa = rt("matlab_bilinear_a", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        Call->getResult(0).replaceAllUsesWith(Cb.getResult());
        Call->getResult(1).replaceAllUsesWith(Ca.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* ---- Statistics Toolbox Tier-2 hypothesis tests --------------------- *
     * [h,p,ci,stats] = ttest/ttest2/vartest2/ztest/kstest(...)  (or the
     * [p,h,stats] order for the rank tests).  The compute symbol (out 0)
     * computes everything into a thread-local; the secondary outputs read
     * it back via matlab_stats_test_{o2,ci,stats}. */
    if (NA && (Name == "ttest" || Name == "ttest2" || Name == "vartest2" ||
               Name == "ztest" || Name == "kstest" || Name == "ranksum" ||
               Name == "signrank" || Name == "signtest") &&
        Call->getNumResults() >= 2 &&
        Call->getNumResults() == NA.getValue().getSExtValue()) {
      int nout = static_cast<int>(Call->getNumResults());
      int nin  = static_cast<int>(Call->getNumOperands());
      /* compute symbol (ttest picks 1-arg vs 2-arg). */
      std::string sym = std::string("matlab_stats_") + Name.str();
      if (Name == "ttest" && nin == 1) sym = "matlab_stats_ttest1";
      SmallVector<Type, 3> argTy(static_cast<size_t>(nin), PtrTy);
      SmallVector<Value, 3> CA;
      bool okBox = true;
      for (int i = 0; i < nin; ++i) {
        Value v = boxAsPtr(Call->getOperand(static_cast<unsigned>(i)));
        if (!v) { okBox = false; break; }
        CA.push_back(v);
      }
      if (okBox) {
        B.setInsertionPoint(Call);
        auto F0 = rt(sym, F64, argTy);
        Call->getResult(0).replaceAllUsesWith(
            LLVM::CallOp::create(B, Call->getLoc(), F0, CA).getResult());
        auto F1 = rt("matlab_stats_test_o2", F64, {});
        Call->getResult(1).replaceAllUsesWith(
            LLVM::CallOp::create(B, Call->getLoc(), F1, ValueRange{}).getResult());
        if (nout >= 3) {
          auto F2 = rt("matlab_stats_test_ci", PtrTy, {});
          Call->getResult(2).replaceAllUsesWith(
              LLVM::CallOp::create(B, Call->getLoc(), F2, ValueRange{}).getResult());
        }
        if (nout >= 4) {
          auto F3 = rt("matlab_stats_test_stats", PtrTy, {});
          Call->getResult(3).replaceAllUsesWith(
              LLVM::CallOp::create(B, Call->getLoc(), F3, ValueRange{}).getResult());
        }
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* [coeff,score,latent,tsquared,explained] = pca(X) — all ptr; the
     * compute (out 0) stashes the rest in a thread-local read by the
     * per-output symbols (tsquared is a stub empty matrix). */
    if (NA && Name == "pca" && Call->getNumResults() >= 2 &&
        Call->getNumResults() == NA.getValue().getSExtValue() &&
        Call->getNumOperands() == 1) {
      Value X = boxAsPtr(Call->getOperand(0));
      if (X) {
        int nout = static_cast<int>(Call->getNumResults());
        B.setInsertionPoint(Call);
        const char *syms[5] = {"matlab_stats_pca", "matlab_stats_pca_score",
                               "matlab_stats_pca_latent", "matlab_stats_pca_empty",
                               "matlab_stats_pca_explained"};
        for (int o = 0; o < nout && o < 5; ++o) {
          // Hold the operand in an OWNING SmallVector — a `ValueRange ar =
          // ValueRange{X}` view dangles once the temporary initializer_list
          // backing array is freed, so the call was reading freed memory for
          // operand #0 (it survived under libc++ but was clobbered under
          // libstdc++, producing "operand #0 does not dominate this use" on
          // the Linux build).
          SmallVector<Value, 1> ar;
          if (o == 0) ar.push_back(X);
          auto Fn = rt(syms[o], PtrTy,
                       ar.empty() ? ArrayRef<Type>{} : ArrayRef<Type>{PtrTy});
          Call->getResult(static_cast<unsigned>(o)).replaceAllUsesWith(
              LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange(ar)).getResult());
        }
        Call->erase(); Changed = true; continue;
      }
    }
    /* [idx,C,sumd,D] = kmeans(X, k) — idx + centroids + within-cluster
     * sums + point-to-centroid distances. */
    if (NA && Name == "kmeans" && Call->getNumResults() >= 2 &&
        Call->getNumResults() == NA.getValue().getSExtValue() &&
        Call->getNumOperands() == 2) {
      Value X = boxAsPtr(Call->getOperand(0));
      Value K = boxAsPtr(Call->getOperand(1));
      if (X && K) {
        int nout = static_cast<int>(Call->getNumResults());
        B.setInsertionPoint(Call);
        auto F0 = rt("matlab_stats_kmeans", PtrTy, {PtrTy, PtrTy});
        Call->getResult(0).replaceAllUsesWith(
            LLVM::CallOp::create(B, Call->getLoc(), F0, ValueRange{X, K}).getResult());
        const char *syms[3] = {"matlab_stats_km_C", "matlab_stats_km_sumd", "matlab_stats_km_D"};
        for (int o = 1; o < nout && o <= 3; ++o) {
          auto Fn = rt(syms[o - 1], PtrTy, {});
          Call->getResult(static_cast<unsigned>(o)).replaceAllUsesWith(
              LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{}).getResult());
        }
        Call->erase(); Changed = true; continue;
      }
    }

    /* HMM 2-output forms: [seq,states]=hmmgenerate, [pstates,logp]=
     * hmmdecode, [TRANS,EMIS]=hmmtrain.  Out 0 computes + stashes; out 1
     * reads the thread-local (logp is f64, the rest ptr). */
    if (NA && (Name == "hmmgenerate" || Name == "hmmdecode" || Name == "hmmtrain") &&
        Call->getNumResults() == 2 && NA.getValue().getSExtValue() == 2 &&
        Call->getNumOperands() == 3) {
      Value A0 = boxAsPtr(Call->getOperand(0));
      Value A1 = boxAsPtr(Call->getOperand(1));
      Value A2 = boxAsPtr(Call->getOperand(2));
      if (A0 && A1 && A2) {
        const char *compute = (Name == "hmmgenerate") ? "matlab_stats_hmmgenerate"
                            : (Name == "hmmdecode")   ? "matlab_stats_hmmdecode"
                                                      : "matlab_stats_hmmtrain";
        bool secondF64 = (Name == "hmmdecode");
        const char *reader = (Name == "hmmgenerate") ? "matlab_stats_hmm_states"
                           : (Name == "hmmdecode")   ? "matlab_stats_hmm_logp"
                                                     : "matlab_stats_hmm_emis";
        B.setInsertionPoint(Call);
        auto F0 = rt(compute, PtrTy, {PtrTy, PtrTy, PtrTy});
        Call->getResult(0).replaceAllUsesWith(
            LLVM::CallOp::create(B, Call->getLoc(), F0, ValueRange{A0, A1, A2}).getResult());
        auto F1 = rt(reader, secondF64 ? Type(F64) : Type(PtrTy), ArrayRef<Type>{});
        Call->getResult(1).replaceAllUsesWith(
            LLVM::CallOp::create(B, Call->getLoc(), F1, ValueRange{}).getResult());
        Call->erase(); Changed = true; continue;
      }
    }

    /* [z, p, k] = tf2zp(b, a). Splits into matlab_tf2zp_{z,p,k}. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Name == "tf2zp" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 3) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fz = rt("matlab_tf2zp_z", PtrTy, {PtrTy, PtrTy});
        auto Fp = rt("matlab_tf2zp_p", PtrTy, {PtrTy, PtrTy});
        auto Fk = rt("matlab_tf2zp_k", F64,   {PtrTy, PtrTy});
        SmallVector<Value, 2> CA{V0, V1};
        auto Cz = LLVM::CallOp::create(B, Call->getLoc(), Fz, CA);
        auto Cp = LLVM::CallOp::create(B, Call->getLoc(), Fp, CA);
        auto Ck = LLVM::CallOp::create(B, Call->getLoc(), Fk, CA);
        Call->getResult(0).replaceAllUsesWith(Cz.getResult());
        Call->getResult(1).replaceAllUsesWith(Cp.getResult());
        Call->getResult(2).replaceAllUsesWith(Ck.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* [b, a] = zp2tf(z, p, k). Splits into matlab_zp2tf_{b,a}. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "zp2tf" && Call->getNumOperands() == 3 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(2).getType() == F64) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      Value V1 = boxAsPtr(Call->getOperand(1));
      if (V0 && V1) {
        B.setInsertionPoint(Call);
        auto Fb = rt("matlab_zp2tf_b", PtrTy, {PtrTy, PtrTy, F64});
        auto Fa = rt("matlab_zp2tf_a", PtrTy, {PtrTy, PtrTy, F64});
        SmallVector<Value, 3> CA{V0, V1, Call->getOperand(2)};
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, CA);
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, CA);
        Call->getResult(0).replaceAllUsesWith(Cb.getResult());
        Call->getResult(1).replaceAllUsesWith(Ca.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* [b, a] = sos2tf(sos) — second-order sections → polynomial.
     * Splits into matlab_sos2tf_{b,a}. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "sos2tf" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 2) {
      Value V0 = boxAsPtr(Call->getOperand(0));
      if (V0) {
        B.setInsertionPoint(Call);
        auto Fb = rt("matlab_sos2tf_b", PtrTy, {PtrTy});
        auto Fa = rt("matlab_sos2tf_a", PtrTy, {PtrTy});
        auto Cb = LLVM::CallOp::create(B, Call->getLoc(), Fb, ValueRange{V0});
        auto Ca = LLVM::CallOp::create(B, Call->getLoc(), Fa, ValueRange{V0});
        Call->getResult(0).replaceAllUsesWith(Cb.getResult());
        Call->getResult(1).replaceAllUsesWith(Ca.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* findpeaks — 2-result form `[pks, locs] = findpeaks(x)` splits
     * into matlab_findpeaks_pks / _locs. Single-LHS `pks = ...` goes
     * through the regular Spec table below. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Name == "findpeaks" && Call->getNumOperands() == 1 &&
        Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fp = rt("matlab_findpeaks_pks",  PtrTy, {PtrTy});
      auto Fl = rt("matlab_findpeaks_locs", PtrTy, {PtrTy});
      auto Cp = LLVM::CallOp::create(B, Call->getLoc(), Fp,
                                      Call->getOperands());
      auto Cl = LLVM::CallOp::create(B, Call->getLoc(), Fl,
                                      Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(Cp.getResult());
      Call->getResult(1).replaceAllUsesWith(Cl.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* residue — partial-fraction expansion. 3-result form
     * `[r, p, k] = residue(b, a)`. 2 operands. Splits into
     * matlab_residue_{r,p,k}(b, a) — same eig_V/eig_D precedent as
     * the 2-result `[V, D] = eig(A)` path above. r and p are complex
     * column vectors; k is a real row vector (possibly empty).
     *
     * Operand promotion mirrors the single-result AutoBoxNames path:
     *   ptr      → pass through
     *   tensor   → defer (the matrix-slot lowering pass converts
     *              tensor → ptr in a later fixpoint iteration)
     *   f64      → auto-box via matlab_mat_from_scalar
     */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Name == "residue" &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 3) {
      auto okOrDefer = [&](mlir::Value V) -> int {
        /* 0 = pass-through ptr, 1 = needs box (f64), -1 = defer
         * (tensor not yet converted), -2 = type mismatch (skip). */
        auto T = V.getType();
        if (T == PtrTy) return 0;
        if (T == F64)   return 1;
        if (isTensorLike(T)) return -1;
        return -2;
      };
      int s0 = okOrDefer(Call->getOperand(0));
      int s1 = okOrDefer(Call->getOperand(1));
      if (s0 != -2 && s1 != -2 && s0 != -1 && s1 != -1) {
        B.setInsertionPoint(Call);
        auto FromScalar = rt("matlab_mat_from_scalar", PtrTy, {F64});
        Value B0 = (s0 == 1)
            ? LLVM::CallOp::create(B, Call->getLoc(), FromScalar,
                                    ValueRange{Call->getOperand(0)})
                  .getResult()
            : Call->getOperand(0);
        Value B1 = (s1 == 1)
            ? LLVM::CallOp::create(B, Call->getLoc(), FromScalar,
                                    ValueRange{Call->getOperand(1)})
                  .getResult()
            : Call->getOperand(1);
        static const char *Suffixes[] = { "r", "p", "k" };
        for (int i = 0; i < 3; ++i) {
          std::string FnName =
              std::string("matlab_residue_") + Suffixes[i];
          auto Fn = rt(FnName, PtrTy, {PtrTy, PtrTy});
          auto Ci = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                          ValueRange{B0, B1});
          Call->getResult(i).replaceAllUsesWith(Ci.getResult());
        }
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* pdepe — 1-D parabolic-elliptic PDE solver. Single-result (sol
     * matrix), 6 operands: (f64 m, ptr pdefn, ptr icfn, ptr bcfn,
     * ptr xmesh, ptr tspan). Routes directly to matlab_pdepe; single-
     * return so this lives outside the multi-return refinement path. */
    if (Name == "pdepe" &&
        Call->getNumOperands() == 6 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == F64 &&
        Call->getOperand(1).getType() == PtrTy &&
        Call->getOperand(2).getType() == PtrTy &&
        Call->getOperand(3).getType() == PtrTy &&
        Call->getOperand(4).getType() == PtrTy &&
        Call->getOperand(5).getType() == PtrTy) {
      B.setInsertionPoint(Call);
      auto Fn = rt("matlab_pdepe", PtrTy,
                   {F64, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy});
      auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                      Call->getOperands());
      Call->getResult(0).replaceAllUsesWith(C0.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    /* fzero — Optimization Toolbox Tier-1.1.  See
     * docs/optim_toolbox_roadmap.md.  Two operand shapes:
     *   x = fzero(@fn, x0)     — scalar guess        → matlab_optim_fzero
     *   x = fzero(@fn, [a b])  — sign-change bracket → matlab_optim_fzero_iv
     * Result is a scalar f64 root.  The handle is a ptr produced by
     * make_handle / anon-call lowering; the second arg's type
     * disambiguates the call shape. */
    if (Name == "fzero" &&
        Call->getNumOperands() == 2 && Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      mlir::Value Arg1 = Call->getOperand(1);
      mlir::Type Arg1Ty = Arg1.getType();
      B.setInsertionPoint(Call);
      if (Arg1Ty == F64) {
        auto Fn = rt("matlab_optim_fzero", F64, {PtrTy, F64});
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      if (Arg1Ty == PtrTy || isTensorLike(Arg1Ty)) {
        mlir::Value Coerced = Arg1;
        if (Arg1Ty != PtrTy) {
          auto Cast = mlir::UnrealizedConversionCastOp::create(
              B, Call->getLoc(), PtrTy, Arg1);
          Coerced = Cast.getResult(0);
        }
        auto Fn = rt("matlab_optim_fzero_iv", F64, {PtrTy, PtrTy});
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        mlir::ValueRange{
                                          Call->getOperand(0), Coerced});
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* linprog — Optimization Toolbox Tier-1.5.  See
     * docs/optim_toolbox_roadmap.md.  Two operand shapes:
     *   x = linprog(f, A, b)                    → matlab_optim_linprog3
     *   x = linprog(f, A, b, Aeq, beq, lb, ub)  → matlab_optim_linprog
     * Every argument is a matrix (a 0×0 `[]` stands for "absent").
     * Operands arrive as ptr or tensor; tensor operands are bridged to
     * ptr with builtin.unrealized_conversion_cast, exactly as the PDE
     * dispatch table does. */
    if (Name == "linprog" && Call->getNumResults() == 1 &&
        (Call->getNumOperands() == 3 || Call->getNumOperands() == 7)) {
      unsigned NArgs = Call->getNumOperands();
      bool ok = true;
      SmallVector<Value, 7> coerced;
      B.setInsertionPoint(Call);
      for (unsigned k = 0; k < NArgs; ++k) {
        Value V = Call->getOperand(k);
        Type T = V.getType();
        if (T == PtrTy) {
          coerced.push_back(V);
        } else if (isTensorLike(T) || mlir::isa<NoneType>(T)) {
          auto Cast = mlir::UnrealizedConversionCastOp::create(
              B, Call->getLoc(), PtrTy, V);
          coerced.push_back(Cast.getResult(0));
        } else if (T == F64) {
          /* A scalar argument (e.g. `beq = 3`) — box it into a 1×1
           * matrix descriptor so the runtime sees a uniform ptr ABI. */
          auto FromScalar = rt("matlab_mat_from_scalar", PtrTy, {F64});
          auto Boxed = LLVM::CallOp::create(B, Call->getLoc(), FromScalar,
                                            ValueRange{V});
          coerced.push_back(Boxed.getResult());
        } else {
          ok = false;
          break;
        }
      }
      if (ok) {
        const char *RtName =
            (NArgs == 3) ? "matlab_optim_linprog3" : "matlab_optim_linprog";
        SmallVector<Type, 7> ArgTys(NArgs, PtrTy);
        auto Fn = rt(RtName, PtrTy, ArgTys);
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn, coerced);
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* fmincon / quadprog / lsqlin / lsqnonlin — Optimization Toolbox
     * Tier-2.  See docs/optim_toolbox_roadmap.md §3.  Each carries
     * several call arities; the runtime ABI is a fixed-width ptr
     * vector, so the lowering maps the first N call operands to the
     * first N ABI slots and null-pads the rest.  Function handles
     * arrive already as ptr; matrix arguments arrive as ptr or tensor
     * (bridged to ptr); a scalar argument is boxed via
     * matlab_mat_from_scalar.  All single-result. */
    {
      struct OptimMultiArity {
        const char *name;
        const char *rt_name;
        unsigned abi_arity;
        SmallVector<unsigned, 4> valid_arities;
      };
      const SmallVector<OptimMultiArity, 9> optim_table = {
        {"fmincon",   "matlab_optim_fmincon",   9, {2, 4, 6, 8, 9}},
        {"quadprog",  "matlab_optim_quadprog",  8, {4, 6, 8}},
        {"lsqlin",    "matlab_optim_lsqlin",    8, {4, 6, 8}},
        {"lsqnonlin", "matlab_optim_lsqnonlin", 4, {2, 4}},
        /* Tier-3 — same multi-arity dispatch: first N call operands
         * map to the first N fixed-ABI slots, the rest are null-padded.
         * Function handles arrive as ptr; scalar args (e.g. coneprog's
         * `gamma`) are boxed via matlab_mat_from_scalar. */
        {"intlinprog",  "matlab_optim_intlinprog",  8, {4, 6, 8}},
        {"fminimax",    "matlab_optim_fminimax",    8, {2, 4, 6, 8}},
        {"fgoalattain", "matlab_optim_fgoalattain", 10, {4, 6, 8, 10}},
        {"coneprog",    "matlab_optim_coneprog",    11, {5, 7, 9, 11}},
        {"fseminf",     "matlab_optim_fseminf",     5, {3, 5}},
      };
      bool matched = false;
      for (const auto &E : optim_table) {
        if (Name != E.name) continue;
        unsigned nargs = Call->getNumOperands();
        bool arity_ok = false;
        for (unsigned a : E.valid_arities) if (a == nargs) arity_ok = true;
        if (!arity_ok || Call->getNumResults() != 1) break;

        B.setInsertionPoint(Call);
        SmallVector<Value, 9> args;
        bool ok = true;
        for (unsigned k = 0; k < E.abi_arity; ++k) {
          if (k < nargs) {
            Value V = Call->getOperand(k);
            Type T = V.getType();
            if (T == PtrTy) {
              args.push_back(V);
            } else if (isTensorLike(T) || mlir::isa<NoneType>(T)) {
              auto Cast = mlir::UnrealizedConversionCastOp::create(
                  B, Call->getLoc(), PtrTy, V);
              args.push_back(Cast.getResult(0));
            } else if (T == F64) {
              auto FromScalar = rt("matlab_mat_from_scalar", PtrTy, {F64});
              auto Boxed = LLVM::CallOp::create(B, Call->getLoc(), FromScalar,
                                                ValueRange{V});
              args.push_back(Boxed.getResult());
            } else {
              ok = false;
              break;
            }
          } else {
            /* Absent argument — pass a null ptr; the runtime's
             * mat_absent() treats it as "argument omitted". */
            args.push_back(LLVM::ZeroOp::create(B, Call->getLoc(), PtrTy));
          }
        }
        if (!ok) break;
        SmallVector<Type, 9> ArgTys(E.abi_arity, PtrTy);
        auto Fn = rt(E.rt_name, PtrTy, ArgTys);
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn, args);
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        Changed = true;
        matched = true;
        break;
      }
      if (matched) continue;
    }

    /* fsolve — Optimization Toolbox.  Two operand shapes:
     *   x = fsolve(@fn, x0)  with x0 scalar  → matlab_optim_fsolve_scalar
     *   x = fsolve(@fn, x0)  with x0 vector  → matlab_optim_fsolve
     * The handle is operand 0 (ptr); the x0 operand's type selects the
     * call shape. */
    if (Name == "fsolve" && Call->getNumOperands() == 2 &&
        Call->getNumResults() == 1 &&
        Call->getOperand(0).getType() == PtrTy) {
      Value X0 = Call->getOperand(1);
      Type X0Ty = X0.getType();
      B.setInsertionPoint(Call);
      if (X0Ty == F64) {
        auto Fn = rt("matlab_optim_fsolve_scalar", F64, {PtrTy, F64});
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                       Call->getOperands());
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
      if (X0Ty == PtrTy || isTensorLike(X0Ty)) {
        Value Coerced = X0;
        if (X0Ty != PtrTy) {
          auto Cast = mlir::UnrealizedConversionCastOp::create(
              B, Call->getLoc(), PtrTy, X0);
          Coerced = Cast.getResult(0);
        }
        auto Fn = rt("matlab_optim_fsolve", PtrTy, {PtrTy, PtrTy});
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                       ValueRange{Call->getOperand(0), Coerced});
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        Changed = true;
        continue;
      }
    }

    /* === PDE Toolbox — Tier-1 + Tier-2 function-form core. ===
     *
     * All entries are single-result, signature-rigid (no overloads at
     * the runtime ABI). Operand types match the runtime declarations
     * in runtime/runtime_pde.cpp. See docs/pde_toolbox_roadmap.md. */
    {
      struct PDEEntry {
        const char *name;
        const char *rt_name;
        Type result_ty;
        SmallVector<Type, 6> args;
      };
      const SmallVector<PDEEntry, 12> pde_table = {
        /* Tier-1 (2-D scalar). */
        {"pde_mesh_rect_tri",      "matlab_pde_mesh_rect_tri",
         PtrTy, {F64, F64, F64, F64, F64, F64}},
        {"pde_boundary_nodes_rect","matlab_pde_boundary_nodes_rect",
         PtrTy, {PtrTy}},
        {"pde_assemble_poisson_2d","matlab_pde_assemble_poisson_2d",
         PtrTy, {PtrTy, F64, F64, F64}},
        {"pde_apply_dirichlet",    "matlab_pde_apply_dirichlet",
         PtrTy, {PtrTy, PtrTy, F64}},
        /* Tier-2 (3-D linear elasticity). */
        {"pde_mesh_cuboid_tet",    "matlab_pde_mesh_cuboid_tet",
         PtrTy, {F64, F64, F64, F64, F64, F64}},
        {"pde_face_nodes",         "matlab_pde_face_nodes",
         PtrTy, {PtrTy, F64}},
        {"pde_assemble_elast_3d",  "matlab_pde_assemble_elast_3d",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_face_pressure_3d",   "matlab_pde_face_pressure_3d",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_apply_fixed_3d",     "matlab_pde_apply_fixed_3d",
         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"pde_reshape_disp_3d",    "matlab_pde_reshape_disp_3d",
         PtrTy, {PtrTy}},
        {"pde_von_mises_3d",       "matlab_pde_von_mises_3d",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"pde_node_von_mises_3d",  "matlab_pde_node_von_mises_3d",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"pde_peak_disp_3d",       "matlab_pde_peak_disp_3d",
         F64,   {PtrTy}},
        /* Struct field accessors (sidestep the f64-default field-access path). */
        {"pde_sys_K",              "matlab_pde_sys_K",
         PtrTy, {PtrTy}},
        {"pde_sys_F",              "matlab_pde_sys_F",
         PtrTy, {PtrTy}},
        {"pde_sys_M",              "matlab_pde_sys_M",
         PtrTy, {PtrTy}},
        {"pde_mesh_nodes",         "matlab_pde_mesh_nodes",
         PtrTy, {PtrTy}},
        {"pde_mesh_triangles",     "matlab_pde_mesh_triangles",
         PtrTy, {PtrTy}},
        {"pde_mesh_tets",          "matlab_pde_mesh_tets",
         PtrTy, {PtrTy}},
        {"pde_mesh_faces",         "matlab_pde_mesh_faces",
         PtrTy, {PtrTy}},
        /* Tier-3 transient + modal. */
        {"pde_assemble_transient_2d","matlab_pde_assemble_transient_2d",
         PtrTy, {PtrTy, F64, F64, F64}},
        {"pde_eigsmall",            "matlab_pde_eigsmall",
         PtrTy, {PtrTy, PtrTy, F64}},
        {"pde_step_forward_euler_2d","matlab_pde_step_forward_euler_2d",
         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        {"pde_init_uniform_2d",     "matlab_pde_init_uniform_2d",
         PtrTy, {PtrTy, F64, PtrTy}},
        /* Tier-4 nonlinear. */
        {"pde_solve_nonlinear_2d",  "matlab_pde_solve_nonlinear_2d",
         PtrTy, {PtrTy, F64, F64, F64, F64}},
        {"pde_result_solution",     "matlab_pde_result_solution",
         PtrTy, {PtrTy}},
        {"pde_result_num_iters",    "matlab_pde_result_num_iters",
         F64,   {PtrTy}},
        {"pde_result_resid",        "matlab_pde_result_resid",
         F64,   {PtrTy}},
        /* Geometry importers — single matlab_string* arg. */
        {"pde_load_stl",            "matlab_pde_load_stl",
         PtrTy, {PtrTy}},
        {"pde_load_glb",            "matlab_pde_load_glb",
         PtrTy, {PtrTy}},
        {"pde_save_stl",            "matlab_pde_save_stl",
         F64,   {PtrTy, PtrTy}},
        /* Sparse matrix runtime — runtime/runtime_sparse.cpp. */
        {"sparse",                  "matlab_sparse_from_triplets",
         PtrTy, {PtrTy, PtrTy, PtrTy, F64, F64}},
        {"speye",                   "matlab_sparse_eye",
         PtrTy, {F64}},
        {"spnnz",                   "matlab_sparse_nnz",
         F64,   {PtrTy}},
        {"sprows",                  "matlab_sparse_rows",
         F64,   {PtrTy}},
        {"spcols",                  "matlab_sparse_cols",
         F64,   {PtrTy}},
        {"spfull",                  "matlab_sparse_full",
         PtrTy, {PtrTy}},
        {"spdiag",                  "matlab_sparse_diag",
         PtrTy, {PtrTy}},
        {"sparse_matvec",           "matlab_sparse_matvec",
         PtrTy, {PtrTy, PtrTy}},
        {"pcg",                     "matlab_sparse_pcg",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"pcg_x",                   "matlab_sparse_pcg_x",
         PtrTy, {PtrTy}},
        {"pcg_flag",                "matlab_sparse_pcg_flag",
         F64,   {PtrTy}},
        {"pcg_relres",              "matlab_sparse_pcg_relres",
         F64,   {PtrTy}},
        {"pcg_iter",                "matlab_sparse_pcg_iter",
         F64,   {PtrTy}},
        /* Sparse FEM assembly. */
        {"pde_assemble_poisson_2d_sparse", "matlab_pde_assemble_poisson_2d_sparse",
         PtrTy, {PtrTy, F64, F64, F64}},
        {"pde_apply_dirichlet_sparse",     "matlab_pde_apply_dirichlet_sparse",
         PtrTy, {PtrTy, PtrTy, F64}},
        {"pde_assemble_elast_3d_sparse",   "matlab_pde_assemble_elast_3d_sparse",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_apply_fixed_3d_sparse",      "matlab_pde_apply_fixed_3d_sparse",
         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"pde_sys_K_sparse",               "matlab_pde_sys_K_sparse",
         PtrTy, {PtrTy}},
        /* Surface→tet voxelizer. */
        {"pde_voxelize_surface",           "matlab_pde_voxelize_surface",
         PtrTy, {PtrTy, F64}},
        /* femodel classdef façade kernel + setters. */
        {"pde_solve_femodel",              "matlab_pde_solve_femodel",
         PtrTy, {PtrTy}},
        {"pde_solve",                      "matlab_pde_solve",
         PtrTy, {PtrTy}},
        /* MATLAB-faithful legacy entry-point names that forward
         * to the same kernels: `solvepde(model)` and
         * `solvepdeeig(model)`.  Both reuse the unified
         * `matlab_pde_solve` dispatcher which looks at
         * `model.AnalysisType` to pick the right kernel. */
        {"solvepde",                       "matlab_pde_solve",
         PtrTy, {PtrTy}},
        {"solvepdeeig",                    "matlab_pde_solve",
         PtrTy, {PtrTy}},
        {"specifyCoefficients",            "matlab_pde_specify_coefficients",
         PtrTy, {PtrTy, F64, F64, F64}},
        {"applyBoundaryCondition",         "matlab_pde_apply_boundary_condition",
         PtrTy, {PtrTy, F64, F64}},
        /* Issue #28 — geometry + mesher surface.  The kwarg-bearing
         * forms receive the `'Name', value` positional pairs that the
         * parser lowers `Name=value` into; the runtime picks the values
         * it understands by key, so the fixed-arity match below tracks
         * exactly the shapes the gating examples emit. */
        {"multicuboid",                    "matlab_pde_multicuboid",
         PtrTy, {F64, F64, F64}},
        {"decsg",                          "matlab_pde_decsg",
         PtrTy, {PtrTy}},
        {"createpde",                      "matlab_pde_createpde",
         PtrTy, {}},
        {"geometryFromEdges",              "matlab_pde_geometry_from_edges",
         PtrTy, {PtrTy, PtrTy}},
        /* generateMesh(model, 'Hmax', h). */
        {"generateMesh",                   "matlab_pde_generate_mesh_kw",
         PtrTy, {PtrTy, PtrTy, F64}},
        /* generateMesh(model) — no kwargs. */
        {"generateMesh",                   "matlab_pde_generate_mesh",
         PtrTy, {PtrTy}},
        /* solve(model) / solve(model, 'FrequencyRange', [...]) — PDE
         * model solve (routed here only when arg0 is a PDE model; see
         * the `solve` sym-predicate carve-out in Lowering.cpp). */
        {"solve",                          "matlab_pde_solve",
         PtrTy, {PtrTy}},
        {"solve",                          "matlab_pde_solve_kw",
         PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* specifyCoefficients(model, m=,d=,c=,a=,f=) — 5 (key,val) pairs. */
        {"specifyCoefficients",            "matlab_pde_specify_coefficients_kw",
         PtrTy, {PtrTy, PtrTy, F64, PtrTy, F64, PtrTy, F64, PtrTy, F64, PtrTy, F64}},
        /* applyBoundaryCondition(model, "dirichlet", Edge=edges, u=val). */
        {"applyBoundaryCondition",         "matlab_pde_apply_bc_kw",
         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        /* interpolateSolution(R, x, y) → scalar. */
        {"interpolateSolution",            "matlab_pde_interpolate_solution",
         F64,   {PtrTy, F64, F64}},
        {"pde_kernel_mesh",                "matlab_pde_kernel_mesh",
         PtrTy, {PtrTy}},
        {"pde_kernel_u",                   "matlab_pde_kernel_u",
         PtrTy, {PtrTy}},
        {"pde_kernel_vm",                  "matlab_pde_kernel_vm",
         PtrTy, {PtrTy}},
        {"pde_set_material",               "matlab_pde_set_material",
         PtrTy, {PtrTy, PtrTy}},
        {"pde_set_face_fixed",             "matlab_pde_set_face_fixed",
         PtrTy, {PtrTy, F64}},
        {"pde_set_face_pressure",          "matlab_pde_set_face_pressure",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_generate_mesh",              "matlab_pde_generate_mesh",
         PtrTy, {PtrTy}},
        /* Geometry primitives. */
        {"pde_multicylinder",              "matlab_pde_multicylinder",
         PtrTy, {F64, F64, F64}},
        {"pde_multicylinder_hollow",       "matlab_pde_multicylinder_hollow",
         PtrTy, {F64, F64, F64, F64}},
        {"pde_multisphere",                "matlab_pde_multisphere",
         PtrTy, {F64, F64}},
        {"pde_translate",                  "matlab_pde_translate",
         PtrTy, {PtrTy, F64, F64, F64}},
        {"pde_rotate",                     "matlab_pde_rotate",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_scale",                      "matlab_pde_scale",
         PtrTy, {PtrTy, F64, F64, F64}},
        /* Tier-3 scalar AnalysisType. */
        {"pde_set_face_temperature",       "matlab_pde_set_face_temperature",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_face_heat",              "matlab_pde_set_face_heat",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_face_voltage",           "matlab_pde_set_face_voltage",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_face_charge",            "matlab_pde_set_face_charge",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_body_heat",              "matlab_pde_set_body_heat",
         PtrTy, {PtrTy, F64}},
        {"pde_set_body_charge",            "matlab_pde_set_body_charge",
         PtrTy, {PtrTy, F64}},
        {"pde_solve_thermal_steady",       "matlab_pde_solve_thermal_steady",
         PtrTy, {PtrTy}},
        {"pde_solve_electrostatic",        "matlab_pde_solve_electrostatic",
         PtrTy, {PtrTy}},
        {"pde_solve_magnetostatic",        "matlab_pde_solve_magnetostatic",
         PtrTy, {PtrTy}},
        {"pde_solve_dc_conduction",        "matlab_pde_solve_dc_conduction",
         PtrTy, {PtrTy}},
        {"pde_set_face_potential",         "matlab_pde_set_face_potential",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_face_current",           "matlab_pde_set_face_current",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_body_current",           "matlab_pde_set_body_current",
         PtrTy, {PtrTy, F64}},
        {"pde_solve_structural_transient", "matlab_pde_solve_structural_transient",
         PtrTy, {PtrTy}},
        {"pde_set_time_step",              "matlab_pde_set_time_step",
         PtrTy, {PtrTy, F64}},
        {"pde_set_num_steps",              "matlab_pde_set_num_steps",
         PtrTy, {PtrTy, F64}},
        {"pde_kernel_uhist",               "matlab_pde_kernel_uhist",
         PtrTy, {PtrTy}},
        {"pde_kernel_tlist",               "matlab_pde_kernel_tlist",
         PtrTy, {PtrTy}},
        {"pde_solve_structural_modal",     "matlab_pde_solve_structural_modal",
         PtrTy, {PtrTy}},
        {"pde_set_num_modes",              "matlab_pde_set_num_modes",
         PtrTy, {PtrTy, F64}},
        {"pde_kernel_freqs",               "matlab_pde_kernel_freqs",
         PtrTy, {PtrTy}},
        {"pde_eig_lanczos_si",             "matlab_pde_eig_lanczos_si",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"pde_eig_lanczos_si_full",        "matlab_pde_eig_lanczos_si_full",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"pde_eig_lambda",                 "matlab_pde_eig_lambda",
         PtrTy, {PtrTy}},
        {"pde_eig_phi",                    "matlab_pde_eig_phi",
         PtrTy, {PtrTy}},
        {"pde_solve_structural_frequency", "matlab_pde_solve_structural_frequency",
         PtrTy, {PtrTy}},
        {"pde_solve_structural_transient_modal",
         "matlab_pde_solve_structural_transient_modal",
         PtrTy, {PtrTy}},
        {"pde_set_rayleigh",               "matlab_pde_set_rayleigh",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_set_modal_results",          "matlab_pde_set_modal_results",
         PtrTy, {PtrTy, PtrTy}},
        {"pde_set_freq_list",              "matlab_pde_set_freq_list",
         PtrTy, {PtrTy, PtrTy}},
        {"pde_kernel_freqlist",            "matlab_pde_kernel_freqlist",
         PtrTy, {PtrTy}},
        {"pde_solve_harmonic_em",          "matlab_pde_solve_harmonic_em",
         PtrTy, {PtrTy}},
        {"pde_set_wave_number",            "matlab_pde_set_wave_number",
         PtrTy, {PtrTy, F64}},
        /* MINRES Krylov solver for symmetric indefinite sparse. */
        {"minres",                         "matlab_sparse_minres",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        /* ILU(0)-preconditioned GMRES(30) — production solver for
         * indefinite + nonsymmetric sparse systems. */
        {"sparse_gmres_ilu0",              "matlab_sparse_gmres_ilu0",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        /* T10 quadratic-tet (10-node) — mesh upgrade + assembly. */
        {"pde_mesh_quadratic",             "matlab_pde_mesh_quadratic",
         PtrTy, {PtrTy}},
        {"pde_assemble_elast_3d_t10",      "matlab_pde_assemble_elast_3d_t10",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_face_pressure_3d_t10",       "matlab_pde_face_pressure_3d_t10",
         PtrTy, {PtrTy, F64, F64}},
        {"pde_face_nodes_t10",             "matlab_pde_face_nodes_t10",
         PtrTy, {PtrTy, F64}},
        {"pde_node_von_mises_3d_t10",      "matlab_pde_node_von_mises_3d_t10",
         PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"pde_apply_fixed_3d_t10",         "matlab_pde_apply_fixed_3d_t10",
         PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* thermalTransient kernel + setters. */
        {"pde_solve_thermal_transient",    "matlab_pde_solve_thermal_transient",
         PtrTy, {PtrTy}},
        {"pde_set_initial_temperature",    "matlab_pde_set_initial_temperature",
         PtrTy, {PtrTy, F64}},
        {"pde_set_cell_temperature",       "matlab_pde_set_cell_temperature",
         PtrTy, {PtrTy, PtrTy}},
        {"pde_set_reference_temperature",  "matlab_pde_set_reference_temperature",
         PtrTy, {PtrTy, F64}},
        /* Tier-4: ROM + reconstructSolution + refineMesh / adaptmesh +
         * structuralStaticNL + multi-component PDEs. */
        {"pde_reduce",                     "matlab_pde_reduce",
         PtrTy, {PtrTy}},
        {"reduce",                         "matlab_pde_reduce",
         PtrTy, {PtrTy}},
        {"pde_reconstruct_solution",       "matlab_pde_reconstruct_solution",
         PtrTy, {PtrTy, PtrTy}},
        {"reconstructSolution",            "matlab_pde_reconstruct_solution",
         PtrTy, {PtrTy, PtrTy}},
        {"pde_refine_mesh",                "matlab_pde_refine_mesh",
         PtrTy, {PtrTy}},
        {"refineMesh",                     "matlab_pde_refine_mesh",
         PtrTy, {PtrTy}},
        {"pde_adapt_mesh",                 "matlab_pde_adapt_mesh",
         PtrTy, {PtrTy, F64}},
        {"adaptmesh",                      "matlab_pde_adapt_mesh",
         PtrTy, {PtrTy, F64}},
        {"pde_solve_structural_static_nl", "matlab_pde_solve_structural_static_nl",
         PtrTy, {PtrTy}},
        {"pde_set_multi_coeff",            "matlab_pde_set_multi_coeff",
         PtrTy, {PtrTy, F64, F64, F64, F64, F64, F64, F64, F64}},
        {"pde_solve_multi",                "matlab_pde_solve_multi",
         PtrTy, {PtrTy}},
        {"pde_multi_u",                    "matlab_pde_multi_u",
         PtrTy, {PtrTy}},
        {"pde_multi_v",                    "matlab_pde_multi_v",
         PtrTy, {PtrTy}},
        /* Tier-4 closure: Craig-Bampton + Total-Lagrangian +
         * Bey refinement + N-component PDEs. */
        {"pde_set_interface_face",         "matlab_pde_set_interface_face",
         PtrTy, {PtrTy, F64}},
        {"pde_reduce_craig_bampton",       "matlab_pde_reduce_craig_bampton",
         PtrTy, {PtrTy}},
        {"pde_solve_structural_static_tl", "matlab_pde_solve_structural_static_tl",
         PtrTy, {PtrTy}},
        {"pde_refine_mesh_bey",            "matlab_pde_refine_mesh_bey",
         PtrTy, {PtrTy}},
        {"refineMeshBey",                  "matlab_pde_refine_mesh_bey",
         PtrTy, {PtrTy}},
        {"pde_adapt_mesh_marked",          "matlab_pde_adapt_mesh_marked",
         PtrTy, {PtrTy, F64}},
        {"pde_set_multi_coeff_n",          "matlab_pde_set_multi_coeff_n",
         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"pde_solve_multi_n",              "matlab_pde_solve_multi_n",
         PtrTy, {PtrTy}},
        {"pde_multi_n_u",                  "matlab_pde_multi_n_u",
         PtrTy, {PtrTy, F64}},
        {"pde_assemble_poisson_3d_sparse", "matlab_pde_assemble_poisson_3d_sparse",
         PtrTy, {PtrTy, F64, F64, F64}},
        {"pde_apply_dirichlet_3d_sparse",  "matlab_pde_apply_dirichlet_3d_sparse",
         PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"pde_face_scalar_load_3d",        "matlab_pde_face_scalar_load_3d",
         PtrTy, {PtrTy, F64, F64}},
        /* Optimization Toolbox — Tier-1 single-signature solvers.
         * See docs/optim_toolbox_roadmap.md.  `fzero` and `linprog`
         * carry two operand shapes each and are dispatched by the
         * hand-rolled blocks above instead.  Objective handles arrive
         * as ptr (function-pointer materialised by anon-call lowering);
         * matrix args arrive as ptr or tensor and are bridged by the
         * loose-match coercion. */
        {"fminbnd",    "matlab_optim_fminbnd",    F64,   {PtrTy, F64, F64}},
        {"fminsearch", "matlab_optim_fminsearch", PtrTy, {PtrTy, PtrTy}},
        {"fminunc",    "matlab_optim_fminunc",    PtrTy, {PtrTy, PtrTy}},
        {"lsqnonneg",  "matlab_optim_lsqnonneg",  PtrTy, {PtrTy, PtrTy}},
        /* Tier-2 — `lsqcurvefit` has a single 4-arg shape; the model
         * handle and the three matrices all arrive as ptr (or are
         * coerced).  `fmincon` / `quadprog` / `lsqlin` / `lsqnonlin`
         * carry several arities and use the hand-rolled multi-arity
         * block below; `fsolve` has a scalar and an N-D form. */
        {"lsqcurvefit", "matlab_optim_lsqcurvefit", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy}},
        /* Tier-4 — problem-based expression-DAG builders.  Every
         * builder takes / returns a scalar node id (f64); `pb_var`
         * takes the variable-name string; `pb_solve` takes the
         * OptimizationProblem classdef object and returns the
         * named-variable solution struct. */
        {"matlab_optim_pb_var",   "matlab_optim_pb_var",   F64, {F64}},
        {"matlab_optim_pb_const", "matlab_optim_pb_const", F64, {F64}},
        {"matlab_optim_pb_add",   "matlab_optim_pb_add",   F64, {F64, F64}},
        {"matlab_optim_pb_sub",   "matlab_optim_pb_sub",   F64, {F64, F64}},
        {"matlab_optim_pb_neg",   "matlab_optim_pb_neg",   F64, {F64}},
        {"matlab_optim_pb_mul",   "matlab_optim_pb_mul",   F64, {F64, F64}},
        {"matlab_optim_pb_div",   "matlab_optim_pb_div",   F64, {F64, F64}},
        {"matlab_optim_pb_pow",   "matlab_optim_pb_pow",   F64, {F64, F64}},
        {"matlab_optim_pb_le",    "matlab_optim_pb_le",    F64, {F64, F64}},
        {"matlab_optim_pb_ge",    "matlab_optim_pb_ge",    F64, {F64, F64}},
        {"matlab_optim_pb_eq",    "matlab_optim_pb_eq",    F64, {F64, F64}},
        {"matlab_optim_pb_solve", "matlab_optim_pb_solve", PtrTy, {PtrTy}},
        {"matlab_optim_pb_solve_eqn", "matlab_optim_pb_solve_eqn",
         PtrTy, {PtrTy}},
        /* MPC Toolbox Tier-1 — loose-match dispatch (tolerates `none`
         * / tensor operands via unrealized_conversion_cast) so the
         * classdef-method-body calls work even when ym / r arrive as
         * matrix-literal tensors or unresolved-type slots. */
        {"matlab_mpc_construct", "matlab_mpc_construct", PtrTy,
         {PtrTy, PtrTy, F64, F64}},
        {"matlab_mpc_move",      "matlab_mpc_move",      PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_mpc_sim",       "matlab_mpc_sim",       PtrTy,
         {PtrTy, F64, PtrTy}},
        /* MPC Tier-2 §3.7 — mpcmove with mpcmoveopt override. */
        {"matlab_mpc_move_opt",  "matlab_mpc_move_opt",  PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        /* MPC Tier-3 §4.1 — adaptive mpcmove with per-tick plant. */
        {"matlab_mpc_move_adaptive", "matlab_mpc_move_adaptive", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        /* MPC Tier-3 §4.2 — time-varying mpcmove with stacked plants. */
        {"matlab_mpc_move_tv", "matlab_mpc_move_tv", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        /* MPC Tier-4 §5.4 — standalone active-set QP.  Both the
         * runtime-symbol name and the user-facing `mpcActiveSetSolver`
         * alias route via the same loose-match coercion. */
        {"matlab_mpc_active_set", "matlab_mpc_active_set", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"mpcActiveSetSolver",    "matlab_mpc_active_set", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy}},
        /* MPC Tier-4 §5.1/5.2/5.3 — explicit MPC. */
        {"matlab_mpc_generate_explicit", "matlab_mpc_generate_explicit",
         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64, PtrTy}},
        {"matlab_mpc_move_explicit",     "matlab_mpc_move_explicit",
         PtrTy, {PtrTy, PtrTy}},
        {"matlab_mpc_simplify_explicit", "matlab_mpc_simplify_explicit",
         PtrTy, {PtrTy, F64}},
        /* GPU Coder Tier-5 design-pattern helpers.  See
         * docs/gpu_coder_roadmap.md §6 / runtime/toolbox/gpu/runtime_gpu_helpers.cpp.
         * `gpucoder.<fn>` is folded by Parser to `gpucoder_<fn>`; we dispatch
         * via the user-facing flat name here.  Each runtime fn takes the
         * function-handle ptr as the first or appropriate operand (per the
         * MathWorks API shape).  Backends (T2-T4) override these at
         * emit-pass time with tiled / tree-reduce / bitonic kernels. */
        {"gpucoder_reduce",             "matlab_gpucoder_reduce",
         F64,   {PtrTy, PtrTy}},
        {"gpucoder_matrixMatrixKernel", "matlab_gpucoder_matmatkernel",
         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"stencilfun",                  "matlab_stencilfun",
         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"gpucoder_stencilKernel",      "matlab_stencilfun",
         PtrTy, {PtrTy, PtrTy, PtrTy}},  /* deprecated alias */
        {"gpucoder_sort",               "matlab_gpucoder_sort",
         PtrTy, {PtrTy}},
        /* Phase 4 of lapack_roadmap §4 — gpucoder.gemm(A,B) routes
         * through the runtime dispatcher (matlab_gpu_gemm) which picks
         * MPSMatrixMultiplication on Metal or falls back to the host
         * BLAS path. */
        {"gpucoder_gemm",               "matlab_gpu_gemm",
         PtrTy, {PtrTy, PtrTy}},
        /* (Removed: bare-name `rand/zeros/ones/eye/randn(n,m,'single')`
         * dispatch — it collided with 3-D `zeros(n,m,d)` because the
         * strict matcher coerces the literal-f64 third arg into ptr
         * via matlab_mat_from_scalar, breaking the matlab_zeros3 path.
         * `benchmark_gpu_backend.m` uses this form and is the only
         * affected fixture; rewrite it as `gpuArray.rand(n,n,'single')`
         * which parser-folds to gpuArray_rand and dispatches cleanly.) */
        /* gpuArray.<static> with the dtype-tag arg (`'single'` /
         * `'double'`) — loose-match coerces the const_char to a
         * matlab_string* via matlab_string_from_literal; the runtime
         * wrapper drops it and calls the underlying allocator. */
        {"gpuArray_rand",  "matlab_gpuArray_rand",  PtrTy, {F64, F64, PtrTy}},
        {"gpuArray_randn", "matlab_gpuArray_randn", PtrTy, {F64, F64, PtrTy}},
        {"gpuArray_zeros", "matlab_gpuArray_zeros", PtrTy, {F64, F64, PtrTy}},
        {"gpuArray_ones",  "matlab_gpuArray_ones",  PtrTy, {F64, F64, PtrTy}},
        {"gpuArray_eye",   "matlab_gpuArray_eye",   PtrTy, {F64, F64, PtrTy}},
        /* gather(g) — CPU lane: identity. */
        {"gather",         "matlab_gather",         PtrTy, {PtrTy}},
        /* gpuArray(X) constructor — transparent.  Loose-match accepts
         * any ptr-shaped input. */
        {"gpuArray",       "matlab_gpuArray_ctor",  PtrTy, {PtrTy}},
        /* existsOnGPU(g) — returns f64. */
        {"existsOnGPU",    "matlab_existsOnGPU",    F64, {PtrTy}},
        /* Bare-name `toc` zero-arg form — f64 elapsed time. */
        {"toc",            "matlab_toc",            F64, {}},
        /* gpuDevice() / gpuDevice(id) / gpuDeviceCount() — device
         * info + selection.  Strict 0/1-arg shapes. */
        {"gpuDeviceCount", "matlab_gpuDeviceCount", F64, {}},
        {"gpuDevice",      "matlab_gpuDevice_handle", PtrTy, {}},
        {"gpuDevice",      "matlab_gpuDevice_select", F64, {F64}},
        /* wait(gpuDevice) — synchronise.  CPU lane no-op. */
        {"wait",           "matlab_gpu_wait",       F64, {PtrTy}},
        /* arrayfun(@fn, X) — element-wise apply via the function-
         * handle ABI.  The anon body runs in the runtime (CPU) or as
         * an emitted kernel (Tier-2+).  Returns a fresh matrix. */
        {"arrayfun",       "matlab_arrayfun",       PtrTy, {PtrTy, PtrTy}},
        /* MPC Tier-4 §5.7 — Finite Control Set MPC. */
        {"matlab_mpc_move_finite", "matlab_mpc_move_finite",
         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        /* MPC Tier-5 — Nonlinear MPC. */
        {"matlab_nlmpc_move", "matlab_nlmpc_move",
         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        /* MPC Tier-6 §7.5/7.6 — review + sim-opt. */
        {"matlab_mpc_review", "matlab_mpc_review", PtrTy, {PtrTy}},
        {"matlab_mpc_sim_opt", "matlab_mpc_sim_opt", PtrTy,
         {PtrTy, F64, PtrTy, PtrTy}},
        /* Global Optimization Toolbox Tier-1 — stochastic global solvers.
         * Arg 0 is the objective-handle ptr; ga/particleswarm take nvars
         * (f64), simulannealbnd takes x0 (ptr); last arg is the hybrid
         * flag (f64). */
        {"matlab_gads_ga", "matlab_gads_ga", PtrTy,
         {PtrTy, F64, PtrTy, PtrTy, F64}},
        /* Tier-6 — ga with an optimoptions object (6th operand = carrier ptr). */
        {"matlab_gads_ga_opts", "matlab_gads_ga_opts", PtrTy,
         {PtrTy, F64, PtrTy, PtrTy, F64, PtrTy}},
        {"matlab_gads_particleswarm", "matlab_gads_particleswarm", PtrTy,
         {PtrTy, F64, PtrTy, PtrTy, F64}},
        {"matlab_gads_simulannealbnd", "matlab_gads_simulannealbnd", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        /* Global Optimization Toolbox Tier-2 — multi-start meta-solvers.
         * make_problem stashes (handle, x0, lb, ub) into the runtime
         * thread-local; run reads it back (multistart takes k, global-
         * search takes nothing). */
        {"matlab_gads_make_problem", "matlab_gads_make_problem", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_gads_multistart", "matlab_gads_multistart", PtrTy, {F64}},
        {"matlab_gads_globalsearch", "matlab_gads_globalsearch", PtrTy, {}},
        /* run(solver, problem [,k]) — runtime-dispatched (REPL-safe). */
        {"matlab_gads_run", "matlab_gads_run", PtrTy, {PtrTy, F64}},
        /* ---- Statistics and Machine Learning Toolbox Tier-1 ----
         * pde_table keyed by the user name → matlab_stats_* symbol; all
         * args are PtrTy so scalar literals get boxed (f64→1x1). */
        {"prctile",  "matlab_stats_prctile",  PtrTy, {PtrTy, PtrTy}},
        {"quantile", "matlab_stats_quantile", PtrTy, {PtrTy, PtrTy}},
        {"iqr",      "matlab_stats_iqr",      PtrTy, {PtrTy}},
        {"range",    "matlab_stats_range",    PtrTy, {PtrTy}},
        {"mode",     "matlab_stats_mode",     PtrTy, {PtrTy}},
        {"skewness", "matlab_stats_skewness", PtrTy, {PtrTy}},
        {"kurtosis", "matlab_stats_kurtosis", PtrTy, {PtrTy}},
        {"geomean",  "matlab_stats_geomean",  PtrTy, {PtrTy}},
        {"harmmean", "matlab_stats_harmmean", PtrTy, {PtrTy}},
        {"cov",      "matlab_stats_cov",      PtrTy, {PtrTy}},
        {"corr",     "matlab_stats_corr",     PtrTy, {PtrTy}},
        {"corrcoef", "matlab_stats_corrcoef", PtrTy, {PtrTy}},
        /* Distributions — 1-arg (defaults) and full-parameter forms. */
        {"normpdf",  "matlab_stats_normpdf1", PtrTy, {PtrTy}},
        {"normpdf",  "matlab_stats_normpdf",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"normcdf",  "matlab_stats_normcdf1", PtrTy, {PtrTy}},
        {"normcdf",  "matlab_stats_normcdf",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"norminv",  "matlab_stats_norminv1", PtrTy, {PtrTy}},
        {"norminv",  "matlab_stats_norminv",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"exppdf",   "matlab_stats_exppdf",   PtrTy, {PtrTy, PtrTy}},
        {"expcdf",   "matlab_stats_expcdf",   PtrTy, {PtrTy, PtrTy}},
        {"expinv",   "matlab_stats_expinv",   PtrTy, {PtrTy, PtrTy}},
        {"unifpdf",  "matlab_stats_unifpdf",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"unifcdf",  "matlab_stats_unifcdf",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"unifinv",  "matlab_stats_unifinv",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"normrnd",  "matlab_stats_normrnd",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"unifrnd",  "matlab_stats_unifrnd",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"exprnd",   "matlab_stats_exprnd",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* fitdist numeric cores (returned [params]; classdef populates). */
        {"matlab_stats_fit_normal",      "matlab_stats_fit_normal",      PtrTy, {PtrTy}},
        {"matlab_stats_fit_exponential", "matlab_stats_fit_exponential", PtrTy, {PtrTy}},
        /* Tier-2 hypothesis tests — single-output (h or p) expression form;
         * the multi-output [h,p,…] form is handled by the splitter above. */
        {"ttest",    "matlab_stats_ttest1",   F64, {PtrTy}},
        {"ttest",    "matlab_stats_ttest",    F64, {PtrTy, PtrTy}},
        {"ttest2",   "matlab_stats_ttest2",   F64, {PtrTy, PtrTy}},
        {"vartest2", "matlab_stats_vartest2", F64, {PtrTy, PtrTy}},
        {"ztest",    "matlab_stats_ztest",    F64, {PtrTy, PtrTy, PtrTy}},
        {"kstest",   "matlab_stats_kstest",   F64, {PtrTy}},
        {"ranksum",  "matlab_stats_ranksum",  F64, {PtrTy, PtrTy}},
        {"signrank", "matlab_stats_signrank", F64, {PtrTy, PtrTy}},
        {"signtest", "matlab_stats_signtest", F64, {PtrTy, PtrTy}},
        {"anova1",   "matlab_stats_anova1",   F64, {PtrTy}},
        /* fitdist alloc-then-populate (result discarded). */
        {"matlab_stats_fitdist_init", "matlab_stats_fitdist_init", PtrTy, {PtrTy, PtrTy, F64}},
        /* Tier-3 regression. */
        {"matlab_stats_fitlm_init",  "matlab_stats_fitlm_init",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_fitglm_init", "matlab_stats_fitglm_init", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_lm_predict",  "matlab_stats_lm_predict",  PtrTy, {PtrTy, PtrTy}},
        {"ridge",    "matlab_stats_ridge",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"regress",  "matlab_stats_regress",  PtrTy, {PtrTy, PtrTy}},
        /* Tier-4 — PCA + clustering (1-output / always-matrix forms). */
        {"pca",        "matlab_stats_pca",        PtrTy, {PtrTy}},
        {"kmeans",     "matlab_stats_kmeans",     PtrTy, {PtrTy, PtrTy}},
        {"pdist2",     "matlab_stats_pdist2",     PtrTy, {PtrTy, PtrTy}},
        {"pdist",      "matlab_stats_pdist",      PtrTy, {PtrTy}},
        {"squareform", "matlab_stats_squareform", PtrTy, {PtrTy}},
        {"silhouette", "matlab_stats_silhouette", PtrTy, {PtrTy, PtrTy}},
        /* Tier-6.2 — t-SNE non-linear embedding (closes carve-down). */
        {"tsne",       "matlab_stats_tsne",       PtrTy, {PtrTy}},
        /* Tier-5 — classification (alloc-then-populate inits + predict + confusionmat). */
        {"matlab_stats_fitknn_init",  "matlab_stats_fitknn_init",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_fitnb_init",   "matlab_stats_fitnb_init",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_fitlda_init",  "matlab_stats_fitlda_init",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_fittree_init", "matlab_stats_fittree_init", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_fitsvm_init",  "matlab_stats_fitsvm_init",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_fitecoc_init", "matlab_stats_fitecoc_init", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_clf_predict",  "matlab_stats_clf_predict",  PtrTy, {PtrTy, PtrTy}},
        {"confusionmat", "matlab_stats_confusionmat", PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_accuracy",   "matlab_stats_accuracy",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_precision",  "matlab_stats_precision",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_recall",     "matlab_stats_recall",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_fscore",     "matlab_stats_fscore",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_rocmetrics", "matlab_stats_rocmetrics", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_stats_aucroc",     "matlab_stats_aucroc",     PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* DL T6.3 user-facing names. */
        {"accuracy",   "matlab_stats_accuracy",   PtrTy, {PtrTy, PtrTy}},
        {"precision",  "matlab_stats_precision",  PtrTy, {PtrTy, PtrTy}},
        {"recall",     "matlab_stats_recall",     PtrTy, {PtrTy, PtrTy}},
        {"fScore",     "matlab_stats_fscore",     PtrTy, {PtrTy, PtrTy}},
        {"rocmetrics", "matlab_stats_rocmetrics", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"aucroc",     "matlab_stats_aucroc",     PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Tier-6 — ensembles. */
        {"matlab_stats_fitensemble_init", "matlab_stats_fitensemble_init", PtrTy, {PtrTy, PtrTy, PtrTy, F64, F64}},
        /* Tier-6 — HMM (1-output forms; multi-output via splitter). */
        {"hmmgenerate", "matlab_stats_hmmgenerate", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"hmmviterbi",  "matlab_stats_hmmviterbi",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"hmmdecode",   "matlab_stats_hmmdecode",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"hmmtrain",    "matlab_stats_hmmtrain",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Tier-6 — Bayesian optimization (objective handle at operand 0). */
        {"matlab_stats_bayesopt", "matlab_stats_bayesopt", PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* ===== Image Processing Toolbox Tier-1 + Tier-2 =====
         * Images are double matrices (M×N) or matlab_mat3 (M×N×3); string
         * args (filenames, fspecial/imnoise type) arrive as matlab_string*
         * (PtrTy) and are read in the runtime.  Multi-arity via the
         * scan-all-overloads matcher. */
        {"imread",     "matlab_image_imread",     PtrTy, {PtrTy}},
        {"imwrite",    "matlab_image_imwrite",    F64,   {PtrTy, PtrTy}},
        {"checkerboard", "matlab_image_checkerboard1", PtrTy, {PtrTy}},
        {"checkerboard", "matlab_image_checkerboard2", PtrTy, {PtrTy, PtrTy}},
        {"checkerboard", "matlab_image_checkerboard",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"im2double",  "matlab_image_im2double",  PtrTy, {PtrTy}},
        {"im2single",  "matlab_image_im2single",  PtrTy, {PtrTy}},
        {"im2uint8",   "matlab_image_im2uint8",   PtrTy, {PtrTy}},
        {"rgb2gray",   "matlab_image_rgb2gray",   PtrTy, {PtrTy}},
        {"im2gray",    "matlab_image_rgb2gray",   PtrTy, {PtrTy}},
        {"mat2gray",   "matlab_image_mat2gray",   PtrTy, {PtrTy}},
        {"imadd",      "matlab_image_imadd",      PtrTy, {PtrTy, PtrTy}},
        {"imsubtract", "matlab_image_imsubtract", PtrTy, {PtrTy, PtrTy}},
        {"immultiply", "matlab_image_immultiply", PtrTy, {PtrTy, PtrTy}},
        {"imdivide",   "matlab_image_imdivide",   PtrTy, {PtrTy, PtrTy}},
        {"imabsdiff",  "matlab_image_imabsdiff",  PtrTy, {PtrTy, PtrTy}},
        {"imcomplement", "matlab_image_imcomplement", PtrTy, {PtrTy}},
        {"imlincomb",  "matlab_image_imlincomb",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"imhist",     "matlab_image_imhist",     PtrTy, {PtrTy}},
        {"imadjust",   "matlab_image_imadjust1",  PtrTy, {PtrTy}},
        {"imadjust",   "matlab_image_imadjust",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"imadjust",   "matlab_image_imadjustg",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"stretchlim", "matlab_image_stretchlim", PtrTy, {PtrTy}},
        {"mean2",      "matlab_image_mean2",      F64,   {PtrTy}},
        {"std2",       "matlab_image_std2",       F64,   {PtrTy}},
        {"fspecial",   "matlab_image_fspecial1",  PtrTy, {PtrTy}},
        {"fspecial",   "matlab_image_fspecial2",  PtrTy, {PtrTy, PtrTy}},
        {"fspecial",   "matlab_image_fspecial",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"imgaussfilt","matlab_image_imgaussfilt1", PtrTy, {PtrTy}},
        {"imgaussfilt","matlab_image_imgaussfilt",  PtrTy, {PtrTy, PtrTy}},
        {"imboxfilt",  "matlab_image_imboxfilt1", PtrTy, {PtrTy}},
        {"imboxfilt",  "matlab_image_imboxfilt",  PtrTy, {PtrTy, PtrTy}},
        {"medfilt2",   "matlab_image_medfilt2_1", PtrTy, {PtrTy}},
        {"medfilt2",   "matlab_image_medfilt2",   PtrTy, {PtrTy, PtrTy}},
        {"ordfilt2",   "matlab_image_ordfilt2",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"stdfilt",    "matlab_image_stdfilt",    PtrTy, {PtrTy}},
        {"rangefilt",  "matlab_image_rangefilt",  PtrTy, {PtrTy}},
        {"histeq",     "matlab_image_histeq",     PtrTy, {PtrTy}},
        {"adapthisteq","matlab_image_adapthisteq",PtrTy, {PtrTy}},
        {"imsharpen",  "matlab_image_imsharpen",  PtrTy, {PtrTy}},
        {"imhistmatch","matlab_image_imhistmatch",PtrTy, {PtrTy, PtrTy}},
        {"imnoise",    "matlab_image_imnoise1",   PtrTy, {PtrTy}},
        {"imnoise",    "matlab_image_imnoise2",   PtrTy, {PtrTy, PtrTy}},
        {"imnoise",    "matlab_image_imnoise",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Tier-3 — geometric transforms. */
        {"imresize",   "matlab_image_imresize2",  PtrTy, {PtrTy, PtrTy}},
        {"imresize",   "matlab_image_imresize",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"imrotate",   "matlab_image_imrotate2",  PtrTy, {PtrTy, PtrTy}},
        {"imrotate",   "matlab_image_imrotate3",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"imrotate",   "matlab_image_imrotate",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"imcrop",     "matlab_image_imcrop",     PtrTy, {PtrTy, PtrTy}},
        {"imtranslate","matlab_image_imtranslate",PtrTy, {PtrTy, PtrTy}},
        {"imwarp",     "matlab_image_imwarp",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_image_fitgeo_init", "matlab_image_fitgeo_init", PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        /* Tier-4 — binarization + morphology. */
        {"graythresh", "matlab_image_graythresh", F64, {PtrTy}},
        {"otsuthresh", "matlab_image_otsuthresh", F64, {PtrTy}},
        {"imbinarize", "matlab_image_imbinarize",  PtrTy, {PtrTy}},
        {"imbinarize", "matlab_image_imbinarize2", PtrTy, {PtrTy, PtrTy}},
        {"im2bw",      "matlab_image_imbinarize",  PtrTy, {PtrTy}},
        {"im2bw",      "matlab_image_imbinarize2", PtrTy, {PtrTy, PtrTy}},
        {"strel",      "matlab_image_strel1",      PtrTy, {PtrTy}},
        {"strel",      "matlab_image_strel2",      PtrTy, {PtrTy, PtrTy}},
        {"strel",      "matlab_image_strel",       PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"imerode",    "matlab_image_imerode",     PtrTy, {PtrTy, PtrTy}},
        {"imdilate",   "matlab_image_imdilate",    PtrTy, {PtrTy, PtrTy}},
        {"imopen",     "matlab_image_imopen",      PtrTy, {PtrTy, PtrTy}},
        {"imclose",    "matlab_image_imclose",     PtrTy, {PtrTy, PtrTy}},
        {"imtophat",   "matlab_image_imtophat",    PtrTy, {PtrTy, PtrTy}},
        {"imbothat",   "matlab_image_imbothat",    PtrTy, {PtrTy, PtrTy}},
        {"imfill",     "matlab_image_imfill",      PtrTy, {PtrTy}},
        {"imfill",     "matlab_image_imfill2",     PtrTy, {PtrTy, PtrTy}},
        {"edge",       "matlab_image_edge1",       PtrTy, {PtrTy}},
        {"edge",       "matlab_image_edge",        PtrTy, {PtrTy, PtrTy}},
        {"bwareaopen", "matlab_image_bwareaopen",  PtrTy, {PtrTy, PtrTy}},
        /* Tier-5 — segmentation + region analysis. */
        {"bwlabel",    "matlab_image_bwlabel",     PtrTy, {PtrTy}},
        {"regionprops","matlab_image_regionprops", PtrTy, {PtrTy, PtrTy}},
        {"label2rgb",  "matlab_image_label2rgb",   PtrTy, {PtrTy}},
        {"bweuler",    "matlab_image_bweuler",     F64,   {PtrTy}},
        {"imsegkmeans","matlab_image_imsegkmeans", PtrTy, {PtrTy, PtrTy}},
        /* Tier-6 — transforms / quality / ROI / colour / block / deblur. */
        {"immse",      "matlab_image_immse",      F64,   {PtrTy, PtrTy}},
        {"psnr",       "matlab_image_psnr",       F64,   {PtrTy, PtrTy}},
        {"ssim",       "matlab_image_ssim",       F64,   {PtrTy, PtrTy}},
        {"rgb2hsv",    "matlab_image_rgb2hsv",    PtrTy, {PtrTy}},
        {"hsv2rgb",    "matlab_image_hsv2rgb",    PtrTy, {PtrTy}},
        {"rgb2ycbcr",  "matlab_image_rgb2ycbcr",  PtrTy, {PtrTy}},
        {"ycbcr2rgb",  "matlab_image_ycbcr2rgb",  PtrTy, {PtrTy}},
        {"rgb2lab",    "matlab_image_rgb2lab",    PtrTy, {PtrTy}},
        {"lab2rgb",    "matlab_image_lab2rgb",    PtrTy, {PtrTy}},
        {"dct2",       "matlab_image_dct2",       PtrTy, {PtrTy}},
        {"idct2",      "matlab_image_idct2",      PtrTy, {PtrTy}},
        {"radon",      "matlab_image_radon",      PtrTy, {PtrTy, PtrTy}},
        {"hough",      "matlab_image_hough",      PtrTy, {PtrTy}},
        {"houghpeaks", "matlab_image_houghpeaks", PtrTy, {PtrTy, PtrTy}},
        {"poly2mask",  "matlab_image_poly2mask",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"roifilt2",   "matlab_image_roifilt2",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"im2col",     "matlab_image_im2col",     PtrTy, {PtrTy, PtrTy}},
        {"col2im",     "matlab_image_col2im",     PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"deconvwnr",  "matlab_image_deconvwnr2", PtrTy, {PtrTy, PtrTy}},
        {"deconvwnr",  "matlab_image_deconvwnr",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"edgetaper",  "matlab_image_edgetaper",  PtrTy, {PtrTy, PtrTy}},
        /* distribution-object methods (runtime-dispatched; scalar args boxed). */
        {"matlab_stats_pd_pdf",    "matlab_stats_pd_pdf",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_pd_cdf",    "matlab_stats_pd_cdf",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_pd_icdf",   "matlab_stats_pd_icdf",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_stats_pd_random", "matlab_stats_pd_random", PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Global Optimization Toolbox Tier-3 — direct search (4-arg, no
         * hybrid; objective handle at operand 0). */
        {"matlab_gads_patternsearch", "matlab_gads_patternsearch", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy}},
        /* Global Optimization Toolbox Tier-4 — surrogate optimization
         * (fn, lb, ub, hybrid; objective handle at operand 0). */
        {"matlab_gads_surrogateopt", "matlab_gads_surrogateopt", PtrTy,
         {PtrTy, PtrTy, PtrTy, F64}},
        /* Global Optimization Toolbox Tier-5 — multiobjective (fn, nvars,
         * lb, ub; vector-objective handle at operand 0). */
        {"matlab_gads_gamultiobj", "matlab_gads_gamultiobj", PtrTy,
         {PtrTy, F64, PtrTy, PtrTy}},
        {"matlab_gads_paretosearch", "matlab_gads_paretosearch", PtrTy,
         {PtrTy, F64, PtrTy, PtrTy}},
        /* System Identification Toolbox Tier-1 — loose-match dispatch.
         * arx / ar populate a pre-allocated idpoly (result discarded).
         * sim / predict / poly2ss return matrices; compare / fpe / aic
         * / goodnessOfFit return scalar f64 metrics. */
        {"matlab_ident_arx", "matlab_ident_arx", PtrTy,
         {PtrTy, PtrTy, PtrTy}},
        {"matlab_ident_ar", "matlab_ident_ar", PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_ident_sim", "matlab_ident_sim", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_predict", "matlab_ident_predict", PtrTy,
         {PtrTy, PtrTy, F64}},
        {"matlab_ident_compare", "matlab_ident_compare", F64,
         {PtrTy, PtrTy}},
        {"matlab_ident_goodness", "matlab_ident_goodness", F64,
         {PtrTy, PtrTy}},
        {"goodnessOfFit", "matlab_ident_goodness", F64, {PtrTy, PtrTy}},
        {"matlab_ident_fpe", "matlab_ident_fpe", F64, {PtrTy}},
        {"matlab_ident_aic", "matlab_ident_aic", F64, {PtrTy}},
        {"matlab_ident_poly2ss_A", "matlab_ident_poly2ss_A", PtrTy, {PtrTy}},
        {"matlab_ident_poly2ss_B", "matlab_ident_poly2ss_B", PtrTy, {PtrTy}},
        {"matlab_ident_poly2ss_C", "matlab_ident_poly2ss_C", PtrTy, {PtrTy}},
        {"matlab_ident_poly2ss_D", "matlab_ident_poly2ss_D", PtrTy, {PtrTy}},
        /* Tier-2 — PEM estimators (populate idpoly in place) + pe/resid. */
        {"matlab_ident_armax", "matlab_ident_armax", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_ident_oe",    "matlab_ident_oe",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_ident_bj",    "matlab_ident_bj",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_ident_pe",    "matlab_ident_pe",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_resid", "matlab_ident_resid", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_iv4",   "matlab_ident_iv4",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_ident_delayest", "matlab_ident_delayest", F64, {PtrTy}},
        /* Tier-3 — subspace state-space (n4sid/ssest) + ss sim/compare. */
        {"matlab_ident_n4sid", "matlab_ident_n4sid", PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_ident_ssest", "matlab_ident_ssest", PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_ident_tfest", "matlab_ident_tfest", PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_ident_sim_ss", "matlab_ident_sim_ss", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_compare_ss", "matlab_ident_compare_ss", F64, {PtrTy, PtrTy}},
        {"matlab_ident_ss_A", "matlab_ident_ss_A", PtrTy, {PtrTy}},
        {"matlab_ident_ss_B", "matlab_ident_ss_B", PtrTy, {PtrTy}},
        {"matlab_ident_ss_C", "matlab_ident_ss_C", PtrTy, {PtrTy}},
        {"matlab_ident_ss_D", "matlab_ident_ss_D", PtrTy, {PtrTy}},
        /* Tier-4 — grey-box (4th operand is the structure-fn handle ptr). */
        {"matlab_ident_greyest", "matlab_ident_greyest", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_ident_nlgreyest", "matlab_ident_nlgreyest", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_ident_impulseest", "matlab_ident_impulseest", PtrTy,
         {PtrTy, PtrTy, F64}},
        {"matlab_ident_forecast", "matlab_ident_forecast", PtrTy,
         {PtrTy, PtrTy, F64}},
        {"matlab_ident_etfe", "matlab_ident_etfe", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_spa",  "matlab_ident_spa",  PtrTy, {PtrTy, PtrTy}},
        /* Tier-5 — EKF/UKF init (R may arrive as scalar f64 → boxed) + steps. */
        {"matlab_ident_ekf_init", "matlab_ident_ekf_init", PtrTy,
         {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_ident_ekf_predict", "matlab_ident_ekf_predict", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_ekf_correct", "matlab_ident_ekf_correct", PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_ident_ukf_predict", "matlab_ident_ukf_predict", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_ukf_correct", "matlab_ident_ukf_correct", PtrTy, {PtrTy, PtrTy, F64}},
        /* Tier-5 — recursive RLS estimators. */
        {"matlab_ident_rls_init",  "matlab_ident_rls_init",  PtrTy, {PtrTy, F64}},
        {"matlab_ident_rls_step",  "matlab_ident_rls_step",  PtrTy, {PtrTy, F64, PtrTy}},
        {"matlab_ident_rarx_init", "matlab_ident_rarx_init", PtrTy, {PtrTy, PtrTy}},
        {"matlab_ident_rarx_step", "matlab_ident_rarx_step", PtrTy, {PtrTy, F64, F64}},
        /* Tier-6 — regularized arx + parameter introspection. */
        {"matlab_ident_arx_reg",  "matlab_ident_arx_reg",  PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_ident_getcov",   "matlab_ident_getcov",   PtrTy, {PtrTy}},
        {"matlab_ident_getpvec",  "matlab_ident_getpvec",  PtrTy, {PtrTy}},
        {"matlab_ident_setpvec",  "matlab_ident_setpvec",  PtrTy, {PtrTy, PtrTy}},
        /* ===== Econometrics Toolbox — model-object methods =====
         * arima estimate populates a fresh object in place (fresh,
         * template, y); forecast/infer/simulate read the fitted model. */
        {"matlab_econ_arima_estimate", "matlab_econ_arima_estimate", PtrTy,
         {PtrTy, PtrTy, PtrTy}},
        {"matlab_econ_arima_forecast", "matlab_econ_arima_forecast", PtrTy,
         {PtrTy, F64, PtrTy}},
        {"matlab_econ_arima_infer", "matlab_econ_arima_infer", PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_econ_arima_simulate", "matlab_econ_arima_simulate", PtrTy,
         {PtrTy, F64}},
        {"matlab_econ_garch_estimate", "matlab_econ_garch_estimate", PtrTy,
         {PtrTy, PtrTy, PtrTy}},
        {"matlab_econ_garch_forecast", "matlab_econ_garch_forecast", PtrTy,
         {PtrTy, F64, PtrTy}},
        {"matlab_econ_garch_infer", "matlab_econ_garch_infer", PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_econ_garch_simulate", "matlab_econ_garch_simulate", PtrTy,
         {PtrTy, F64}},
        {"matlab_econ_varm_estimate", "matlab_econ_varm_estimate", PtrTy,
         {PtrTy, PtrTy, PtrTy}},
        {"matlab_econ_varm_forecast", "matlab_econ_varm_forecast", PtrTy,
         {PtrTy, F64, PtrTy}},
        {"matlab_econ_varm_simulate", "matlab_econ_varm_simulate", PtrTy,
         {PtrTy, F64}},
        {"matlab_econ_varm_irf", "matlab_econ_varm_irf", PtrTy,
         {PtrTy, F64}},
        {"matlab_econ_ssm_estimate", "matlab_econ_ssm_estimate", PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_econ_ssm_filter", "matlab_econ_ssm_filter", PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_econ_ssm_smooth", "matlab_econ_ssm_smooth", PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_econ_ssm_forecast", "matlab_econ_ssm_forecast", PtrTy,
         {PtrTy, F64, PtrTy}},
        {"matlab_econ_bayeslm_estimate", "matlab_econ_bayeslm_estimate", PtrTy,
         {PtrTy, PtrTy, PtrTy}},
        {"matlab_econ_bayeslm_forecast", "matlab_econ_bayeslm_forecast", PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_econ_dtmc_asymptotics", "matlab_econ_dtmc_asymptotics", PtrTy,
         {PtrTy}},
        {"matlab_econ_dtmc_simulate", "matlab_econ_dtmc_simulate", PtrTy,
         {PtrTy, F64}},
        /* ===== DSP System Toolbox =====
         * System-Object step/lifecycle entries.  The classdef method body
         * forwards the receiver `obj` (PtrTy) + the input frame (PtrTy
         * matrix); the runtime reads/writes the object's coefficient +
         * state properties and returns the output frame (PtrTy). */
        /* Tier-1 — core filters. */
        {"matlab_dsp_iir_step",    "matlab_dsp_iir_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_sos_step",    "matlab_dsp_sos_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_delay_step",  "matlab_dsp_delay_step",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_reset",       "matlab_dsp_reset",       PtrTy, {PtrTy}},
        {"matlab_dsp_init_state",  "matlab_dsp_init_state",  PtrTy, {PtrTy}},
        {"matlab_dsp_get_state",   "matlab_dsp_get_state",   PtrTy, {PtrTy}},
        /* Tier-3 — adaptive filters. */
        {"matlab_dsp_lms_step",    "matlab_dsp_lms_step",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dsp_rls_step",    "matlab_dsp_rls_step",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dsp_get_weights", "matlab_dsp_get_weights", PtrTy, {PtrTy}},
        /* Tier-4 — multirate + filter banks. */
        {"matlab_dsp_firdecim_step",   "matlab_dsp_firdecim_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_firinterp_step",  "matlab_dsp_firinterp_step",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_cicdecim_step",   "matlab_dsp_cicdecim_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_cicinterp_step",  "matlab_dsp_cicinterp_step",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_rateconv_step",   "matlab_dsp_rateconv_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_channelizer_step","matlab_dsp_channelizer_step",PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_synthesizer_step","matlab_dsp_synthesizer_step",PtrTy, {PtrTy, PtrTy}},
        /* Tier-5 — sources / sliding stats / detectors / spectral / buffering. */
        {"matlab_dsp_sine_step",      "matlab_dsp_sine_step",      PtrTy, {PtrTy}},
        {"matlab_dsp_nco_step",       "matlab_dsp_nco_step",       PtrTy, {PtrTy}},
        {"matlab_dsp_chirp_step",     "matlab_dsp_chirp_step",     PtrTy, {PtrTy}},
        {"matlab_dsp_movavg_step",    "matlab_dsp_movavg_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_movrms_step",    "matlab_dsp_movrms_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_movmax_step",    "matlab_dsp_movmax_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_movmin_step",    "matlab_dsp_movmin_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_movstd_step",    "matlab_dsp_movstd_step",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_peakfind_step",  "matlab_dsp_peakfind_step",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_dcblock_step",   "matlab_dsp_dcblock_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_zcd_step",       "matlab_dsp_zcd_step",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_spectest_step",  "matlab_dsp_spectest_step",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_asyncbuf_write", "matlab_dsp_asyncbuf_write", PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_asyncbuf_read",  "matlab_dsp_asyncbuf_read",  PtrTy, {PtrTy, F64}},
        /* Tier-6 — linalg + polish filter SOs. */
        {"matlab_dsp_levinson_step",  "matlab_dsp_levinson_step",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_notchpeak_step", "matlab_dsp_notchpeak_step", PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_lowpass_step",   "matlab_dsp_lowpass_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_highpass_step",  "matlab_dsp_highpass_step",  PtrTy, {PtrTy, PtrTy}},
        /* Tier-7 / Tier-8 — dsphdl.* simulation step entries + CORDIC. */
        {"matlab_dsphdl_fir_step",      "matlab_dsphdl_fir_step",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsphdl_biquad_step",   "matlab_dsphdl_biquad_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsphdl_sine_step",     "matlab_dsphdl_sine_step",     PtrTy, {PtrTy}},
        {"matlab_dsphdl_nco_step",      "matlab_dsphdl_nco_step",      PtrTy, {PtrTy}},
        {"matlab_dsphdl_firdecim_step", "matlab_dsphdl_firdecim_step", PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsphdl_cicdecim_step", "matlab_dsphdl_cicdecim_step", PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsphdl_latency",       "matlab_dsphdl_latency",       F64,  {PtrTy}},
        {"matlab_dsp_cordic_atan2",     "matlab_dsp_cordic_atan2",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_dsp_cordic_sqrt",      "matlab_dsp_cordic_sqrt",      PtrTy, {PtrTy}},
        /* ===== Curve Fitting Toolbox Tier-1 =====
         * fit/feval/coeffvalues/gof/output/disp all take the cfit object
         * (PtrTy) as the first operand; the model-tag string ('polyN')
         * arrives as a matlab.const_char and is coerced to matlab_string*
         * by the WantTy==PtrTy + const_char path below. */
        {"matlab_curvefit_fit",         "matlab_curvefit_fit",         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_curvefit_fit_opts",    "matlab_curvefit_fit_opts",    PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_curvefit_feval",       "matlab_curvefit_feval",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_curvefit_coeffvalues", "matlab_curvefit_coeffvalues", PtrTy, {PtrTy}},
        {"matlab_curvefit_gof",         "matlab_curvefit_gof",         PtrTy, {PtrTy}},
        {"matlab_curvefit_output",      "matlab_curvefit_output",      PtrTy, {PtrTy}},
        {"matlab_curvefit_disp",        "matlab_curvefit_disp",        PtrTy, {PtrTy}},
        /* Tier-3 — custom equations + postprocessing. */
        {"matlab_curvefit_fittype_init", "matlab_curvefit_fittype_init", PtrTy, {PtrTy, PtrTy}},
        {"matlab_curvefit_fit_custom",   "matlab_curvefit_fit_custom",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_curvefit_confint",        "matlab_curvefit_confint",        PtrTy, {PtrTy, F64}},
        {"matlab_curvefit_differentiate",  "matlab_curvefit_differentiate",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_curvefit_integrate",      "matlab_curvefit_integrate",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_curvefit_numcoeffs",      "matlab_curvefit_numcoeffs",      PtrTy, {PtrTy}},
        {"matlab_curvefit_formula",        "matlab_curvefit_formula",        PtrTy, {PtrTy}},
        /* Tier-4 — smooth (1-3 args) + csaps.  The method string (smooth's
         * 3rd arg) is coerced const_char→matlab_string by the path below. */
        {"smooth", "matlab_curvefit_smooth1", PtrTy, {PtrTy}},
        {"smooth", "matlab_curvefit_smooth2", PtrTy, {PtrTy, PtrTy}},
        {"smooth", "matlab_curvefit_smooth3", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"csaps",  "matlab_curvefit_csaps",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        /* Tier-5 — surface fitting. */
        {"matlab_curvefit_fit_surface", "matlab_curvefit_fit_surface", PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_curvefit_sfeval",      "matlab_curvefit_sfeval",      PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Tier-6 — ppform spline layer.  spline/pchip/ppmak/fnder/fnint are
         * ctor-and-populate (handled in Lowering); fnval/fnbrk are plain
         * matrix-returning builtins matched here by user name. */
        {"matlab_curvefit_spline_init", "matlab_curvefit_spline_init", PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_curvefit_ppmak_init",  "matlab_curvefit_ppmak_init",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_curvefit_fnder_init",  "matlab_curvefit_fnder_init",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_curvefit_fnint_init",  "matlab_curvefit_fnint_init",  PtrTy, {PtrTy, PtrTy}},
        {"fnval", "matlab_curvefit_fnval", PtrTy, {PtrTy, PtrTy}},
        {"fnbrk", "matlab_curvefit_fnbrk", PtrTy, {PtrTy, PtrTy}},
        /* ===== Wavelet Toolbox single-return builtins =====
         * Family / option strings (const_char) coerce to matlab_string*
         * via the WantTy==PtrTy + const_char path; scalar level/threshold
         * args stay f64; matrices/coefficient vectors bridge to llvm.ptr. */
        /* Tier-1 */
        {"idwt",    "matlab_wavelet_idwt",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"waverec", "matlab_wavelet_waverec", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"appcoef", "matlab_wavelet_appcoef", PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"detcoef", "matlab_wavelet_detcoef", PtrTy, {PtrTy, PtrTy, F64}},
        {"wrcoef",  "matlab_wavelet_wrcoef",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        {"upcoef",  "matlab_wavelet_upcoef",  PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"qmf",     "matlab_wavelet_qmf",     PtrTy, {PtrTy}},
        {"wmaxlev", "matlab_wavelet_wmaxlev", F64,   {PtrTy, PtrTy}},
        {"wextend", "matlab_wavelet_wextend", PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"wkeep",   "matlab_wavelet_wkeep",   PtrTy, {PtrTy, F64}},
        {"centfrq", "matlab_wavelet_centfrq", F64,   {PtrTy}},
        {"wentropy","matlab_wavelet_wentropy",F64,   {PtrTy, PtrTy}},
        {"wenergy", "matlab_wavelet_wenergy", PtrTy, {PtrTy, PtrTy}},
        {"dwtmode", "matlab_wavelet_dwtmode", PtrTy, {PtrTy}},
        /* Tier-2 */
        {"wthresh", "matlab_wavelet_wthresh", PtrTy, {PtrTy, PtrTy, F64}},
        {"thselect","matlab_wavelet_thselect",F64,   {PtrTy, PtrTy}},
        {"wnoisest","matlab_wavelet_wnoisest3",F64,  {PtrTy, PtrTy, F64}},
        {"wnoisest","matlab_wavelet_wnoisest1",F64,  {PtrTy}},
        {"wnoise",  "matlab_wavelet_wnoise_x", PtrTy, {F64, F64}},
        {"wden",    "matlab_wavelet_wden",    PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64, PtrTy}},
        {"wdenoise","matlab_wavelet_wdenoise3",PtrTy,{PtrTy, F64, PtrTy}},
        {"wdenoise","matlab_wavelet_wdenoise2",PtrTy,{PtrTy, F64}},
        {"wcompress","matlab_wavelet_wcompress",PtrTy,{PtrTy, F64, PtrTy}},
        {"measerr", "matlab_wavelet_measerr", F64,   {PtrTy, PtrTy}},
        /* Tier-3 */
        {"icwt",    "matlab_wavelet_icwt",    PtrTy, {PtrTy}},
        {"scal2frq","matlab_wavelet_scal2frq",PtrTy, {PtrTy, PtrTy, F64}},
        {"freq2scal","matlab_wavelet_freq2scal",PtrTy,{PtrTy, PtrTy, F64}},
        {"wcoherence","matlab_wavelet_wcoherence",PtrTy,{PtrTy, PtrTy}},
        /* Tier-4 */
        {"modwt",   "matlab_wavelet_modwt3",  PtrTy, {PtrTy, PtrTy, F64}},
        {"modwt",   "matlab_wavelet_modwt2",  PtrTy, {PtrTy, PtrTy}},
        {"imodwt",  "matlab_wavelet_imodwt2", PtrTy, {PtrTy, PtrTy}},
        {"imodwt",  "matlab_wavelet_imodwt1", PtrTy, {PtrTy}},
        {"modwtmra","matlab_wavelet_modwtmra2",PtrTy,{PtrTy, PtrTy}},
        {"modwtmra","matlab_wavelet_modwtmra1",PtrTy,{PtrTy}},
        {"modwtvar","matlab_wavelet_modwtvar",PtrTy, {PtrTy}},
        {"swt",     "matlab_wavelet_swt",     PtrTy, {PtrTy, F64, PtrTy}},
        {"iswt",    "matlab_wavelet_iswt",    PtrTy, {PtrTy, PtrTy}},
        {"idwt2",   "matlab_wavelet_idwt2",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"waverec2","matlab_wavelet_waverec2",PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"wcodemat","matlab_wavelet_wcodemat2",PtrTy,{PtrTy, F64}},
        {"wcodemat","matlab_wavelet_wcodemat1",PtrTy,{PtrTy}},
        /* Tier-5/6 */
        {"ewt",     "matlab_wavelet_ewt",     PtrTy, {PtrTy, F64}},
        {"vmd",     "matlab_wavelet_vmd",     PtrTy, {PtrTy, F64}},
        {"emd",     "matlab_wavelet_emd",     PtrTy, {PtrTy, F64}},
        {"matchingPursuit", "matlab_wavelet_omp", PtrTy, {PtrTy, PtrTy, F64}},
        {"waveletScattering", "matlab_wavelet_scatter", PtrTy, {PtrTy}},
        {"featureMatrix",     "matlab_wavelet_scatter", PtrTy, {PtrTy}},
        {"wpdec",   "matlab_wavelet_wpdec",   PtrTy, {PtrTy, F64, PtrTy}},
        {"wprec",   "matlab_wavelet_wprec",   PtrTy, {PtrTy, PtrTy}},
        {"wpcoef",  "matlab_wavelet_wpcoef",  PtrTy, {PtrTy, F64}},
        {"besttree","matlab_wavelet_besttree",PtrTy, {PtrTy}},
        {"wenergy", "matlab_wavelet_wenergy_wp", PtrTy, {PtrTy}},
        /* ===== Sensor Fusion and Tracking Toolbox =====
         * Init functions (constructor-intercept callees) and method runtime
         * symbols register here so the primary dispatcher can lower them. */
        {"matlab_fusion_quat_init_wxyz", "matlab_fusion_quat_init_wxyz", PtrTy,
         {PtrTy, F64, F64, F64, F64}},
        {"matlab_fusion_quat_init_mat",  "matlab_fusion_quat_init_mat",  PtrTy,
         {PtrTy, PtrTy}},
        {"matlab_fusion_quat_init_from_data", "matlab_fusion_quat_init_from_data", PtrTy,
         {PtrTy, PtrTy}},
        /* Algebra (operate on the N×4 Data matrices). */
        {"matlab_fusion_quat_mul_data",       "matlab_fusion_quat_mul_data",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_quat_conj_data",      "matlab_fusion_quat_conj_data",      PtrTy, {PtrTy}},
        {"matlab_fusion_quat_norm_data",      "matlab_fusion_quat_norm_data",      PtrTy, {PtrTy}},
        {"matlab_fusion_quat_normalize_data", "matlab_fusion_quat_normalize_data", PtrTy, {PtrTy}},
        {"matlab_fusion_quat_inverse_data",   "matlab_fusion_quat_inverse_data",   PtrTy, {PtrTy}},
        /* Conversions. */
        {"matlab_fusion_quat_to_eul",   "matlab_fusion_quat_to_eul",   PtrTy, {PtrTy}},
        {"matlab_fusion_eul_to_quat",   "matlab_fusion_eul_to_quat",   PtrTy, {PtrTy}},
        {"matlab_fusion_quat_to_rotm",  "matlab_fusion_quat_to_rotm",  PtrTy, {PtrTy, F64}},
        {"matlab_fusion_quat2rotm",     "matlab_fusion_quat2rotm",     PtrTy, {PtrTy}},
        {"matlab_fusion_rotm_to_quat",  "matlab_fusion_rotm_to_quat",  PtrTy, {PtrTy}},
        {"matlab_fusion_quat_rotatepoint", "matlab_fusion_quat_rotatepoint", PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_quat_rotateframe", "matlab_fusion_quat_rotateframe", PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_quat_slerp",    "matlab_fusion_quat_slerp",    PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_fusion_quat_dist",     "matlab_fusion_quat_dist",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_ecompass",      "matlab_fusion_ecompass",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_quat_parts",    "matlab_fusion_quat_parts",    PtrTy, {PtrTy}},
        /* T1.7 core gaps — generic math, surfaced under the user names too. */
        {"matlab_cross",   "matlab_cross",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dot",     "matlab_dot",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_deg2rad", "matlab_deg2rad", PtrTy, {PtrTy}},
        {"matlab_rad2deg", "matlab_rad2deg", PtrTy, {PtrTy}},
        {"matlab_normalize_vec", "matlab_normalize_vec", PtrTy, {PtrTy}},
        {"matlab_mvnrnd", "matlab_mvnrnd", PtrTy, {PtrTy, PtrTy, F64}},
        /* User-facing aliases (so the call site `cross(a,b)` matches here). */
        {"cross",    "matlab_cross",            PtrTy, {PtrTy, PtrTy}},
        {"dot",      "matlab_dot",              PtrTy, {PtrTy, PtrTy}},
        {"deg2rad",  "matlab_deg2rad",          PtrTy, {PtrTy}},
        {"rad2deg",  "matlab_rad2deg",          PtrTy, {PtrTy}},
        {"mvnrnd",   "matlab_mvnrnd",           PtrTy, {PtrTy, PtrTy, F64}},
        {"quat2eul", "matlab_fusion_quat_to_eul",  PtrTy, {PtrTy}},
        {"eul2quat", "matlab_fusion_eul_to_quat",  PtrTy, {PtrTy}},
        {"quat2rotm","matlab_fusion_quat2rotm",    PtrTy, {PtrTy}},
        {"rotm2quat","matlab_fusion_rotm_to_quat", PtrTy, {PtrTy}},
        {"slerp",    "matlab_fusion_quat_slerp",   PtrTy, {PtrTy, PtrTy, F64}},
        {"ecompass", "matlab_fusion_ecompass",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_quat_disp", "matlab_fusion_quat_disp", PtrTy, {PtrTy}},
        /* Sensor Fusion Tier-2 — motion / measurement / KF / EKF / UKF. */
        {"matlab_fusion_constvel",  "matlab_fusion_constvel",  PtrTy, {PtrTy, F64}},
        {"matlab_fusion_constacc",  "matlab_fusion_constacc",  PtrTy, {PtrTy, F64}},
        {"matlab_fusion_constturn", "matlab_fusion_constturn", PtrTy, {PtrTy, F64}},
        {"matlab_fusion_cvmeas",    "matlab_fusion_cvmeas",    PtrTy, {PtrTy}},
        {"matlab_fusion_cameas",    "matlab_fusion_cameas",    PtrTy, {PtrTy}},
        {"matlab_fusion_ctmeas",    "matlab_fusion_ctmeas",    PtrTy, {PtrTy}},
        {"constvel",  "matlab_fusion_constvel",  PtrTy, {PtrTy, F64}},
        {"constacc",  "matlab_fusion_constacc",  PtrTy, {PtrTy, F64}},
        {"constturn", "matlab_fusion_constturn", PtrTy, {PtrTy, F64}},
        {"cvmeas",    "matlab_fusion_cvmeas",    PtrTy, {PtrTy}},
        {"cameas",    "matlab_fusion_cameas",    PtrTy, {PtrTy}},
        {"ctmeas",    "matlab_fusion_ctmeas",    PtrTy, {PtrTy}},
        {"matlab_fusion_objdet_init", "matlab_fusion_objdet_init", PtrTy, {PtrTy, F64, PtrTy, PtrTy}},
        {"matlab_fusion_trackingkf_init",   "matlab_fusion_trackingkf_init",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_trackingkf_predict","matlab_fusion_trackingkf_predict",PtrTy, {PtrTy}},
        {"matlab_fusion_trackingkf_correct","matlab_fusion_trackingkf_correct",PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_trackingekf_init",   "matlab_fusion_trackingekf_init",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_trackingekf_predict","matlab_fusion_trackingekf_predict",PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_trackingekf_correct","matlab_fusion_trackingekf_correct",PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_trackingukf_init",   "matlab_fusion_trackingukf_init",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_trackingukf_predict","matlab_fusion_trackingukf_predict",PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_trackingukf_correct","matlab_fusion_trackingukf_correct",PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_initcvekf", "matlab_fusion_initcvekf", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_initctekf", "matlab_fusion_initctekf", PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Sensor Fusion Tier-3 — sensors + orientation filters + insfilterMARG. */
        {"matlab_fusion_imu_init", "matlab_fusion_imu_init", PtrTy, {PtrTy, F64, F64}},
        {"matlab_fusion_gps_init", "matlab_fusion_gps_init", PtrTy, {PtrTy, F64}},
        {"matlab_fusion_imu_step", "matlab_fusion_imu_step", PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_gps_step", "matlab_fusion_gps_step", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_ahrs_init",       "matlab_fusion_ahrs_init",       PtrTy, {PtrTy, F64}},
        {"matlab_fusion_imufilter_init",  "matlab_fusion_imufilter_init",  PtrTy, {PtrTy, F64}},
        {"matlab_fusion_compfilter_init", "matlab_fusion_compfilter_init", PtrTy, {PtrTy, F64}},
        {"matlab_fusion_ahrs_step",       "matlab_fusion_ahrs_step",       PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_imufilter_step",  "matlab_fusion_imufilter_step",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_compfilter_step", "matlab_fusion_compfilter_step", PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_insmarg_init",        "matlab_fusion_insmarg_init",        PtrTy, {PtrTy, F64}},
        {"matlab_fusion_insmarg_predict",     "matlab_fusion_insmarg_predict",     PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_fusion_insmarg_fuse_accel",  "matlab_fusion_insmarg_fuse_accel",  PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_insmarg_fuse_gps",    "matlab_fusion_insmarg_fuse_gps",    PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_allanvar", "matlab_fusion_allanvar", PtrTy, {PtrTy, F64}},
        {"allanvar",                "matlab_fusion_allanvar", PtrTy, {PtrTy, F64}},
        /* Sensor Fusion Tier-4 — waypointTrajectory + coordinate frames. */
        {"matlab_fusion_waypoint_init",   "matlab_fusion_waypoint_init",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_waypoint_lookup", "matlab_fusion_waypoint_lookup", PtrTy, {PtrTy, F64}},
        {"matlab_fusion_lla2ned",         "matlab_fusion_lla2ned",         PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_ned2lla",         "matlab_fusion_ned2lla",         PtrTy, {PtrTy, PtrTy}},
        {"lla2ned",                       "matlab_fusion_lla2ned",         PtrTy, {PtrTy, PtrTy}},
        {"ned2lla",                       "matlab_fusion_ned2lla",         PtrTy, {PtrTy, PtrTy}},
        /* Sensor Fusion Tier-5 — assignmunkres + trackerGNN. */
        {"matlab_fusion_assignmunkres",   "matlab_fusion_assignmunkres",   PtrTy, {PtrTy}},
        {"assignmunkres",                 "matlab_fusion_assignmunkres",   PtrTy, {PtrTy}},
        {"matlab_fusion_gnn_init",        "matlab_fusion_gnn_init",        PtrTy, {PtrTy, F64}},
        {"matlab_fusion_gnn_step",        "matlab_fusion_gnn_step",        PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_fusion_gnn_numconfirmed","matlab_fusion_gnn_numconfirmed",PtrTy, {PtrTy}},
        /* Sensor Fusion Tier-6 — covariance intersection + GOSPA / OSPA + RMSE + RTS smoother. */
        {"matlab_fusion_covint",  "matlab_fusion_covint",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"trackFuser",            "matlab_fusion_covint",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_fusion_gospa",   "matlab_fusion_gospa",   PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"trackGOSPAMetric",      "matlab_fusion_gospa",   PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_fusion_ospa",    "matlab_fusion_ospa",    PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"trackOSPAMetric",       "matlab_fusion_ospa",    PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_fusion_trackerror", "matlab_fusion_trackerror", PtrTy, {PtrTy, PtrTy}},
        {"trackErrorMetrics",        "matlab_fusion_trackerror", PtrTy, {PtrTy, PtrTy}},
        {"matlab_fusion_rts_smoother", "matlab_fusion_rts_smoother", PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"rtsSmoother",                "matlab_fusion_rts_smoother", PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* ===== Robotics System Toolbox — T1 tform conversions =============== */
        {"trvec2tform", "matlab_robotics_trvec2tform", PtrTy, {PtrTy}},
        {"tform2trvec", "matlab_robotics_tform2trvec", PtrTy, {PtrTy}},
        {"rotm2tform", "matlab_robotics_rotm2tform", PtrTy, {PtrTy}},
        {"tform2rotm", "matlab_robotics_tform2rotm", PtrTy, {PtrTy}},
        {"eul2tform",  "matlab_robotics_eul2tform",  PtrTy, {PtrTy}},
        {"tform2eul",  "matlab_robotics_tform2eul",  PtrTy, {PtrTy}},
        {"axang2rotm", "matlab_robotics_axang2rotm", PtrTy, {PtrTy}},
        {"rotm2axang", "matlab_robotics_rotm2axang", PtrTy, {PtrTy}},
        {"axang2tform","matlab_robotics_axang2tform",PtrTy, {PtrTy}},
        {"tform2axang","matlab_robotics_tform2axang",PtrTy, {PtrTy}},
        {"quat2tform", "matlab_robotics_quat2tform", PtrTy, {PtrTy}},
        {"tform2quat", "matlab_robotics_tform2quat", PtrTy, {PtrTy}},
        {"homtrans",   "matlab_robotics_homtrans",   PtrTy, {PtrTy, PtrTy}},
        {"wrapToPi",   "matlab_robotics_wrapToPi",   PtrTy, {PtrTy}},
        {"wrapTo2Pi",  "matlab_robotics_wrapTo2Pi",  PtrTy, {PtrTy}},
        {"vecnorm",    "matlab_robotics_vecnorm",    PtrTy, {PtrTy}},
        {"matlab_robotics_tform_mul", "matlab_robotics_tform_mul", PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_tform_inv", "matlab_robotics_tform_inv", PtrTy, {PtrTy}},
        /* Init / populator entries (callees of constructor intercepts). */
        {"matlab_robotics_se3_init",          "matlab_robotics_se3_init",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_so3_init",          "matlab_robotics_so3_init",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_tree_init",         "matlab_robotics_tree_init",         PtrTy, {PtrTy}},
        {"matlab_robotics_tree_addbody",      "matlab_robotics_tree_addbody",      PtrTy, {PtrTy, PtrTy, F64, F64, F64}},
        {"matlab_robotics_loadrobot",         "matlab_robotics_loadrobot",         PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_getTransform",      "matlab_robotics_getTransform",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_geometricJacobian", "matlab_robotics_geometricJacobian", PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_homeConfiguration", "matlab_robotics_homeConfiguration", PtrTy, {PtrTy}},
        {"matlab_robotics_randomConfiguration","matlab_robotics_randomConfiguration",PtrTy,{PtrTy}},
        {"matlab_robotics_ik_init",           "matlab_robotics_ik_init",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_ik_solve",          "matlab_robotics_ik_solve",          PtrTy, {PtrTy, PtrTy, PtrTy, F64, F64}},
        {"matlab_robotics_constraint_pose_init","matlab_robotics_constraint_pose_init",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"cubicpolytraj",  "matlab_robotics_cubicpolytraj",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"trapveltraj",    "matlab_robotics_trapveltraj",    PtrTy, {PtrTy, F64}},
        {"transformtraj",  "matlab_robotics_transformtraj",  PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_cubicpolytraj","matlab_robotics_cubicpolytraj",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_trapveltraj",  "matlab_robotics_trapveltraj",  PtrTy,{PtrTy, F64}},
        {"matlab_robotics_transformtraj","matlab_robotics_transformtraj",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_massMatrix",    "matlab_robotics_massMatrix",    PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_inverseDynamics","matlab_robotics_inverseDynamics",PtrTy,{PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_forwardDynamics","matlab_robotics_forwardDynamics",PtrTy,{PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_gravityTorque",  "matlab_robotics_gravityTorque",  PtrTy,{PtrTy, PtrTy}},
        {"matlab_robotics_velocityProduct","matlab_robotics_velocityProduct",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_centerOfMass",   "matlab_robotics_centerOfMass",   PtrTy,{PtrTy, PtrTy}},
        {"matlab_robotics_importrobot",    "matlab_robotics_importrobot",    PtrTy,{PtrTy, PtrTy}},
        {"matlab_robotics_gik_init",       "matlab_robotics_gik_init",       PtrTy,{PtrTy, PtrTy}},
        {"matlab_robotics_gik_solve",      "matlab_robotics_gik_solve",      PtrTy,{PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_constraint_position_init","matlab_robotics_constraint_position_init",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_constraint_orientation_init","matlab_robotics_constraint_orientation_init",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_collcyl_init",   "matlab_robotics_collcyl_init",   PtrTy,{PtrTy, F64, F64}},
        {"matlab_robotics_collcap_init",   "matlab_robotics_collcap_init",   PtrTy,{PtrTy, F64, F64}},
        {"matlab_robotics_gjk_collision",  "matlab_robotics_gjk_collision",  PtrTy,{PtrTy, PtrTy}},
        {"matlab_robotics_diffdrive_init", "matlab_robotics_diffdrive_init", PtrTy, {PtrTy, F64, F64}},
        {"matlab_robotics_diffdrive_derivative","matlab_robotics_diffdrive_derivative",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_unicycle_init", "matlab_robotics_unicycle_init", PtrTy, {PtrTy, F64}},
        {"matlab_robotics_unicycle_derivative","matlab_robotics_unicycle_derivative",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_bicycle_init", "matlab_robotics_bicycle_init", PtrTy, {PtrTy, F64}},
        {"matlab_robotics_bicycle_derivative","matlab_robotics_bicycle_derivative",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_ackermann_init", "matlab_robotics_ackermann_init", PtrTy, {PtrTy, F64}},
        {"matlab_robotics_ackermann_derivative","matlab_robotics_ackermann_derivative",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_occmap_init",    "matlab_robotics_occmap_init",    PtrTy, {PtrTy, F64, F64, F64}},
        {"matlab_robotics_occmap_set",     "matlab_robotics_occmap_set",     PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_robotics_occmap_get",     "matlab_robotics_occmap_get",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_occmap_check",   "matlab_robotics_occmap_check",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_prm_init",       "matlab_robotics_prm_init",       PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_robotics_prm_findpath",   "matlab_robotics_prm_findpath",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_pursuit_init",   "matlab_robotics_pursuit_init",   PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_robotics_pursuit_step",   "matlab_robotics_pursuit_step",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_collbox_init",   "matlab_robotics_collbox_init",   PtrTy, {PtrTy, F64, F64, F64}},
        {"matlab_robotics_collsphere_init","matlab_robotics_collsphere_init",PtrTy, {PtrTy, F64}},
        {"matlab_robotics_checkCollision", "matlab_robotics_checkCollision", PtrTy, {PtrTy, PtrTy}},
        {"matlab_robotics_rrt_init",       "matlab_robotics_rrt_init",       PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_robotics_rrt_plan",       "matlab_robotics_rrt_plan",       PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* ===== Navigation Toolbox Tiers 1–4 ============================= */
        {"matlab_nav_occmap_init",      "matlab_nav_occmap_init",      PtrTy, {PtrTy, F64, F64, F64}},
        {"matlab_nav_occmap_set",       "matlab_nav_occmap_set",       PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_nav_occmap_get",       "matlab_nav_occmap_get",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_occmap_check",     "matlab_nav_occmap_check",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_occmap_inflate",   "matlab_nav_occmap_inflate",   PtrTy, {PtrTy, F64}},
        {"matlab_nav_occmap_setgrid",   "matlab_nav_occmap_setgrid",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_ss_se2_init",      "matlab_nav_ss_se2_init",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_ss_dubins_init",   "matlab_nav_ss_dubins_init",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_ss_distance",      "matlab_nav_ss_distance",      PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_ss_interpolate",   "matlab_nav_ss_interpolate",   PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_ss_sample",        "matlab_nav_ss_sample",        PtrTy, {PtrTy}},
        {"matlab_nav_validator_init",   "matlab_nav_validator_init",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_validator_isstate","matlab_nav_validator_isstate",PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_validator_ismotion","matlab_nav_validator_ismotion",PtrTy,{PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_path_init",        "matlab_nav_path_init",        PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_path_length",      "matlab_nav_path_length",      PtrTy, {PtrTy}},
        {"matlab_nav_planner_init",     "matlab_nav_planner_init",     PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_nav_planner_plan",     "matlab_nav_planner_plan",     PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_shortenpath",      "matlab_nav_shortenpath",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_astar_init",       "matlab_nav_astar_init",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_astar_plan",       "matlab_nav_astar_plan",       PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_lidarscan_init",   "matlab_nav_lidarscan_init",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_matchscans",       "matlab_nav_matchscans",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_slam_init",        "matlab_nav_slam_init",        PtrTy, {PtrTy, F64, F64}},
        {"matlab_nav_slam_addscan",     "matlab_nav_slam_addscan",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_posegraph_init",   "matlab_nav_posegraph_init",   PtrTy, {PtrTy}},
        {"matlab_nav_posegraph_addrel", "matlab_nav_posegraph_addrel", PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_nav_posegraph_optimize","matlab_nav_posegraph_optimize",PtrTy,{PtrTy}},
        /* ===== Navigation Tiers 5–6 ===================================== */
        {"matlab_nav_vfh_step",         "matlab_nav_vfh_step",         PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_nav_mcl_init",         "matlab_nav_mcl_init",         PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_mcl_step",         "matlab_nav_mcl_step",         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_pf_initialize",    "matlab_nav_pf_initialize",    PtrTy, {PtrTy, F64, PtrTy, PtrTy}},
        {"matlab_nav_pf_predict",       "matlab_nav_pf_predict",       PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_pf_correct",       "matlab_nav_pf_correct",       PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_nav_pf_estimate",      "matlab_nav_pf_estimate",      PtrTy, {PtrTy}},
        {"matlab_nav_gnss_step",        "matlab_nav_gnss_step",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"gnssconstellation",           "matlab_nav_gnssconstellation",PtrTy, {F64}},
        {"matlab_nav_gnssconstellation","matlab_nav_gnssconstellation",PtrTy, {F64}},
        {"pseudoranges",                "matlab_nav_pseudoranges",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_pseudoranges",     "matlab_nav_pseudoranges",     PtrTy, {PtrTy, PtrTy}},
        {"receiverposition",            "matlab_nav_receiverposition", PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_receiverposition", "matlab_nav_receiverposition", PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_frenet_init",      "matlab_nav_frenet_init",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_frenet_g2f",       "matlab_nav_frenet_g2f",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_frenet_f2g",       "matlab_nav_frenet_f2g",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_trajgen_init",     "matlab_nav_trajgen_init",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_nav_trajgen_connect",  "matlab_nav_trajgen_connect",  PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        /* ===== Reinforcement Learning Toolbox Tier 1 (tabular) ========= */
        {"matlab_rl_gridworld_init",    "matlab_rl_gridworld_init",    PtrTy, {PtrTy}},
        {"matlab_rl_mdp_init",          "matlab_rl_mdp_init",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_cartpole_init",     "matlab_rl_cartpole_init",     PtrTy, {PtrTy}},
        {"matlab_rl_dqn_init",          "matlab_rl_dqn_init",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_dqn_train",         "matlab_rl_dqn_train",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_dqn_sim",           "matlab_rl_dqn_sim",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_pg_init",           "matlab_rl_pg_init",           PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_pg_train",          "matlab_rl_pg_train",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_pg_sim",            "matlab_rl_pg_sim",            PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_pendulum_init",     "matlab_rl_pendulum_init",     PtrTy, {PtrTy}},
        {"matlab_rl_ddpg_init",         "matlab_rl_ddpg_init",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_ddpg_train",        "matlab_rl_ddpg_train",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_ddpg_sim",          "matlab_rl_ddpg_sim",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_td3_init",          "matlab_rl_td3_init",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_td3_train",         "matlab_rl_td3_train",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_td3_sim",           "matlab_rl_td3_sim",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_ppo_init",          "matlab_rl_ppo_init",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_ppo_train",         "matlab_rl_ppo_train",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_ppo_sim",           "matlab_rl_ppo_sim",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_sac_init",          "matlab_rl_sac_init",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_sac_train",         "matlab_rl_sac_train",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_sac_sim",           "matlab_rl_sac_sim",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_countdown_init",    "matlab_rl_countdown_init",    PtrTy, {PtrTy}},
        {"matlab_rl_grpo_init",         "matlab_rl_grpo_init",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_grpo_train",        "matlab_rl_grpo_train",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_grpo_sim",          "matlab_rl_grpo_sim",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_trpo_init",         "matlab_rl_trpo_init",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_trpo_train",        "matlab_rl_trpo_train",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_trpo_sim",          "matlab_rl_trpo_sim",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_get_action",        "matlab_rl_get_action",        PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_get_maxq",          "matlab_rl_get_maxq",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_greedy_policy",     "matlab_rl_greedy_policy",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_obs_info",          "matlab_rl_obs_info",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_act_info",          "matlab_rl_act_info",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_table_init",        "matlab_rl_table_init",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_qvf_init",          "matlab_rl_qvf_init",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_agent_init",        "matlab_rl_agent_init",        PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_rl_get_critic",        "matlab_rl_get_critic",        PtrTy, {PtrTy, PtrTy}},
        {"matlab_rl_get_params",        "matlab_rl_get_params",        PtrTy, {PtrTy}},
        {"matlab_rl_train",             "matlab_rl_train",             PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_rl_sim",               "matlab_rl_sim",               PtrTy, {PtrTy, PtrTy}},
        /* ===== Deep Learning Toolbox Tiers 1-2 (dlarray + autodiff) ===== */
        {"matlab_dlnet_dlarray_init",   "matlab_dlnet_dlarray_init",   PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_extractdata",    "matlab_dlnet_extractdata",    PtrTy, {PtrTy}},
        {"matlab_dlnet_plus",           "matlab_dlnet_plus",           PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_minus",          "matlab_dlnet_minus",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_mtimes",         "matlab_dlnet_mtimes",         PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_times",          "matlab_dlnet_times",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_relu",           "matlab_dlnet_relu",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_sigmoid",        "matlab_dlnet_sigmoid",        PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_tanh",           "matlab_dlnet_tanh",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_softmax",        "matlab_dlnet_softmax",        PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_sum",            "matlab_dlnet_sum",            PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_mean",           "matlab_dlnet_mean",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_log",            "matlab_dlnet_log",            PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_exp",            "matlab_dlnet_exp",            PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_crossentropy",   "matlab_dlnet_crossentropy",   PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_mse",            "matlab_dlnet_mse",            PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_lstm",           "matlab_dlnet_lstm",           PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_transpose",      "matlab_dlnet_transpose",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_embed",          "matlab_dlnet_embed",          PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_gru",            "matlab_dlnet_gru",            PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_bilstm",         "matlab_dlnet_bilstm",         PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_lstmp",          "matlab_dlnet_lstmp",          PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        /* Tier C: rank-4 batched conv with autodiff support. */
        {"matlab_dlnet_conv2d_batch",  "matlab_dlnet_conv2d_batch", PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* Tier C: dlarray reshape (2-D / 4-D output) + pooling. */
        {"matlab_dlnet_reshape2",      "matlab_dlnet_reshape2",     PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_dlnet_reshape4",      "matlab_dlnet_reshape4",     PtrTy, {PtrTy, PtrTy, F64, F64, F64, F64}},
        {"matlab_dlnet_maxpool2d",     "matlab_dlnet_maxpool2d",    PtrTy, {PtrTy, PtrTy, F64, F64}},
        {"matlab_dlnet_avgpool2d",     "matlab_dlnet_avgpool2d",    PtrTy, {PtrTy, PtrTy, F64, F64}},
        /* BatchNorm + conv-with-bias/pad/stride + axis-aware softmax. */
        {"matlab_dlnet_batchnorm",     "matlab_dlnet_batchnorm",    PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_conv2d_full",   "matlab_dlnet_conv2d_full",  PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64, F64, F64, F64}},
        {"matlab_dlnet_softmax_dim",   "matlab_dlnet_softmax_dim",  PtrTy, {PtrTy, PtrTy, F64}},
        /* LayerNorm + BN inference (frozen-stats). */
        {"matlab_dlnet_layernorm",     "matlab_dlnet_layernorm",    PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_dlnet_batchnorm_eval","matlab_dlnet_batchnorm_eval",PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy}},
        /* GroupNorm + EMA-tracked BN training. */
        {"matlab_dlnet_groupnorm",     "matlab_dlnet_groupnorm",    PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        {"matlab_dlnet_batchnorm_train","matlab_dlnet_batchnorm_train",PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, PtrTy, F64}},
        /* InstanceNorm + RMSNorm. */
        {"matlab_dlnet_instancenorm",  "matlab_dlnet_instancenorm", PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_rmsnorm",       "matlab_dlnet_rmsnorm",      PtrTy, {PtrTy, PtrTy, PtrTy, F64}},
        /* Tape scoping (recommended between training iters to prevent
         * monotonic tape growth + slow dlgradient). */
        {"matlab_dltape_truncate",      "matlab_dltape_truncate",     PtrTy, {F64}},
        /* Functional optimizers (SGD-momentum + Adam + RMSProp). */
        {"matlab_dlnet_sgdmupdate",     "matlab_dlnet_sgdmupdate",    PtrTy, {PtrTy, PtrTy, PtrTy, F64, F64}},
        {"matlab_dlnet_adamupdate",     "matlab_dlnet_adamupdate",    PtrTy, {PtrTy, PtrTy, PtrTy, PtrTy, F64, F64, F64, F64, F64}},
        {"matlab_dlnet_rmspropupdate",  "matlab_dlnet_rmspropupdate", PtrTy, {PtrTy, PtrTy, PtrTy, F64, F64, F64}},
        /* Magnitude pruning. */
        {"matlab_dlnet_prune_mask",     "matlab_dlnet_prune_mask",    PtrTy, {PtrTy, F64}},
        /* Experiment-sweep harness: runExperiment(@fn, Grid) -> Nx1 results. */
        {"matlab_dlnet_run_experiment", "matlab_dlnet_run_experiment", PtrTy, {PtrTy, PtrTy}},
        /* T1.8 — image-data plumbing.  User-facing names go in this typed
         * Spec table (not the pde_table) so the const_char → matlab_string
         * promotion fires for the string-literal folder/path args. */
        {"matlab_dlnet_mkdir",          "matlab_dlnet_mkdir",          PtrTy, {PtrTy}},
        {"matlab_dlnet_imds_load",      "matlab_dlnet_imds_load",      PtrTy, {PtrTy}},
        {"matlab_dlnet_imds_count",     "matlab_dlnet_imds_count",     PtrTy, {PtrTy}},
        {"matlab_dlnet_imds_split",     "matlab_dlnet_imds_split",     PtrTy, {PtrTy, F64}},
        {"mkdir",           "matlab_dlnet_mkdir",      PtrTy, {PtrTy}},
        {"imageDatastore",  "matlab_dlnet_imds_load",  PtrTy, {PtrTy}},
        {"countEachLabel",  "matlab_dlnet_imds_count", PtrTy, {PtrTy}},
        {"splitEachLabel",  "matlab_dlnet_imds_split", PtrTy, {PtrTy, F64}},
        /* T3.4b — random rotate/scale/translate augmenter. */
        {"matlab_dlnet_augment_image",  "matlab_dlnet_augment_image",  PtrTy, {PtrTy, F64, F64, F64, F64, F64}},
        {"augmentImage",                "matlab_dlnet_augment_image",  PtrTy, {PtrTy, F64, F64, F64, F64, F64}},
        /* Tape-tracked shape concatenation — `[a; b]` / `[a b]` over
         * dlarray (matrix lane).  Backward slices the adjoint along the
         * concat axis. */
        {"matlab_dlnet_vertcat",        "matlab_dlnet_vertcat",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_horzcat",        "matlab_dlnet_horzcat",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        /* F: generic obj-array carrier — runtime-resident, handle-keyed. */
        {"matlab_dlnet_oa_new",         "matlab_dlnet_oa_new",         PtrTy, {}},
        {"matlab_dlnet_oa_append",      "matlab_dlnet_oa_append",      PtrTy, {PtrTy, PtrTy}},
        /* C: dlnetwork carrier — sequential layer-list driver. */
        {"matlab_dlnet_net_new",        "matlab_dlnet_net_new",        PtrTy, {}},
        {"matlab_dlnet_net_add_fc",     "matlab_dlnet_net_add_fc",     PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_net_add_relu",   "matlab_dlnet_net_add_relu",   PtrTy, {PtrTy}},
        {"matlab_dlnet_net_add_sigmoid","matlab_dlnet_net_add_sigmoid",PtrTy, {PtrTy}},
        {"matlab_dlnet_net_add_tanh",   "matlab_dlnet_net_add_tanh",   PtrTy, {PtrTy}},
        {"matlab_dlnet_net_add_softmax","matlab_dlnet_net_add_softmax",PtrTy, {PtrTy}},
        {"matlab_dlnet_net_predict",    "matlab_dlnet_net_predict",    PtrTy, {PtrTy, PtrTy}},
        /* T3.8 — GPU training dispatch toggle.  When `dlnetGpu(1)` is
         * on, dlnet's MTIMES forward + backward routes through
         * matlab_gpu_gemm (Metal-accelerated above 128³, CPU fallback
         * otherwise).  Solver-step (adamupdate / sgdmupdate / rmsprop)
         * stays on the host — it's bandwidth-bound elementwise, and
         * the single-device pattern keeps parameter updates host-side
         * by convention (matches PyTorch / TF). */
        {"matlab_dlnet_gpu_set",        "matlab_dlnet_gpu_set",        PtrTy, {F64}},
        /* H5 — ONNX inference-graph importer + programmatic builder. */
        {"onnxRead",            "matlab_onnx_read",            PtrTy, {PtrTy}},
        {"onnxRun",             "matlab_onnx_run",             PtrTy, {PtrTy, PtrTy}},
        {"onnxNewModel",        "matlab_onnx_new_model",       PtrTy, {}},
        {"onnxAddInit",         "matlab_onnx_add_init",        PtrTy, {PtrTy, PtrTy}},
        {"onnxSetInput",        "matlab_onnx_set_input",       PtrTy, {PtrTy, PtrTy}},
        {"onnxSetOutput",       "matlab_onnx_set_output",      PtrTy, {PtrTy}},
        {"onnxBeginNode",       "matlab_onnx_begin_node",      PtrTy, {PtrTy}},
        {"onnxNodeInput",       "matlab_onnx_node_input",      PtrTy, {PtrTy}},
        {"onnxNodeOutput",      "matlab_onnx_node_output",     PtrTy, {PtrTy}},
        {"onnxNodeAttrInt",     "matlab_onnx_node_attr_int",   PtrTy, {PtrTy, F64}},
        {"onnxNodeAttrFloat",   "matlab_onnx_node_attr_float", PtrTy, {PtrTy, F64}},
        {"onnxNodeAttrInts",    "matlab_onnx_node_attr_ints",  PtrTy, {PtrTy, PtrTy}},
        {"onnxEndNode",         "matlab_onnx_end_node",        PtrTy, {}},
        {"onnxSave",            "matlab_onnx_save",            PtrTy, {PtrTy}},
        /* Deep Learning Toolbox Phase 1 — small extra ops over dlarray. */
        {"matlab_dlnet_rdivide",        "matlab_dlnet_rdivide",        PtrTy, {PtrTy, PtrTy, PtrTy}},
        {"matlab_dlnet_sqrt",           "matlab_dlnet_sqrt",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_mean_dim",       "matlab_dlnet_mean_dim",       PtrTy, {PtrTy, PtrTy, F64}},
        {"matlab_dlnet_leakyrelu",      "matlab_dlnet_leakyrelu",      PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_gelu",           "matlab_dlnet_gelu",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_swish",          "matlab_dlnet_swish",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_softplus",       "matlab_dlnet_softplus",       PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_elu",            "matlab_dlnet_elu",            PtrTy, {PtrTy, PtrTy}},
        /* DL HDL Tier H1 — INT8 quantization (plain matrix in/out). */
        {"dlquantize",                  "matlab_dlnet_quantize",       PtrTy, {PtrTy}},
        {"matlab_dlnet_quantize",       "matlab_dlnet_quantize",       PtrTy, {PtrTy}},
        {"dlqscale",                    "matlab_dlnet_qscale",         PtrTy, {PtrTy}},
        {"matlab_dlnet_qscale",         "matlab_dlnet_qscale",         PtrTy, {PtrTy}},
        {"dlqclip",                     "matlab_dlnet_qclip",          PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_qclip",          "matlab_dlnet_qclip",          PtrTy, {PtrTy, PtrTy}},
        {"dlqcalibrate",                "matlab_dlnet_qcalibrate",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_qcalibrate",     "matlab_dlnet_qcalibrate",     PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_grad",           "matlab_dlnet_grad",           PtrTy, {PtrTy, PtrTy}},
        {"matlab_dlnet_reset",          "matlab_dlnet_reset",          PtrTy, {F64}},
      };
      bool matched = false;
      for (const auto &E : pde_table) {
        if (Name != E.name) continue;
        /* `continue` (not `break`) so a name with several entries of
         * different arities — e.g. normcdf(x) vs normcdf(x,mu,sigma) —
         * keeps scanning for the matching overload. */
        if ((size_t)Call->getNumOperands() != E.args.size()) continue;
        if (Call->getNumResults() != 1) continue;
        /* Loose match: PtrTy expected operands also accept tensor types
         * (the operand has come from a builtin whose type-inference
         * stamp is Array — still a runtime ptr at LLVM level).  We
         * insert an llvm.bitcast-style coercion via a no-op transfer
         * since llvm.ptr and tensor share the underlying ptr ABI. */
        bool ok = true;
        SmallVector<Value, 6> coerced;
        SmallVector<Operation *, 2> deadLits;   /* const_char ops to sweep */
        for (size_t k = 0; k < E.args.size(); ++k) {
          Value V = Call->getOperand(k);
          Type WantTy = E.args[k];
          /* String literal → matlab_string* — checked FIRST because a
           * matlab.const_char result is already typed PtrTy and would
           * otherwise be passed through as a raw global address. */
          if (WantTy == PtrTy && isMatlabOp(V.getDefiningOp(), "matlab.const_char")) {
            Operation *Def = V.getDefiningOp();
            auto VA = Def->getAttrOfType<StringAttr>("value");
            if (!VA) { ok = false; break; }
            StringRef Text = VA.getValue();
            LLVM::GlobalOp Found;
            for (auto G : Mod.getOps<LLVM::GlobalOp>()) {
              if (!G.getConstant()) continue;
              auto Attr = mlir::dyn_cast_or_null<StringAttr>(G.getValueAttr());
              if (Attr && Attr.getValue() == Text) { Found = G; break; }
            }
            if (!Found) {
              OpBuilder::InsertionGuard IG(B);
              B.setInsertionPointToStart(Mod.getBody());
              auto ArrayTy = LLVM::LLVMArrayType::get(
                  IntegerType::get(Ctx, 8), static_cast<unsigned>(Text.size()));
              unsigned N = 0; std::string SymName;
              do { SymName = ("__matlab_str_s" + std::to_string(N++)); }
              while (Mod.lookupSymbol(SymName));
              Found = LLVM::GlobalOp::create(B, Mod.getLoc(), ArrayTy,
                  /*isConstant=*/true, LLVM::Linkage::Internal, SymName,
                  StringAttr::get(Ctx, Text));
            }
            B.setInsertionPoint(Call);
            Value Addr = LLVM::AddressOfOp::create(B, Call->getLoc(), PtrTy,
                                                   Found.getSymName());
            Value LenV = LLVM::ConstantOp::create(B, Call->getLoc(), I64,
                B.getI64IntegerAttr(static_cast<int64_t>(Text.size())));
            auto FnS = rt("matlab_string_from_literal", PtrTy, {PtrTy, I64});
            coerced.push_back(
                LLVM::CallOp::create(B, Call->getLoc(), FnS,
                                     ValueRange{Addr, LenV}).getResult());
            deadLits.push_back(Def);
          } else if (V.getType() == WantTy) {
            coerced.push_back(V);
          } else if (WantTy == PtrTy && (isTensorLike(V.getType()) ||
                                          mlir::isa<NoneType>(V.getType()))) {
            /* Tensor / none → ptr: the underlying value is already an
             * llvm.ptr in the runtime ABI (tensor slots holding
             * matrix descriptors; none slots holding class-instance
             * pointers from the kwarg-ctor sugar).  Bridge via
             * builtin.unrealized_conversion_cast — the LLVM lowering
             * treats it as a noop when the runtime pointer types
             * line up. */
            B.setInsertionPoint(Call);
            auto Cast = mlir::UnrealizedConversionCastOp::create(
                B, Call->getLoc(), PtrTy, V);
            coerced.push_back(Cast.getResult(0));
          } else if (WantTy == PtrTy && V.getType() == F64) {
            /* f64 → ptr: box the scalar via matlab_mat_from_scalar.
             * Needed for matlab_mpc_* calls where SISO references
             * (e.g. `r = [1]`) arrive as plain f64 because Sema
             * scalar-promoted the 1×1 matrix. */
            B.setInsertionPoint(Call);
            auto FnBox = rt("matlab_mat_from_scalar", PtrTy, {F64});
            auto Box = LLVM::CallOp::create(B, Call->getLoc(), FnBox,
                                             ValueRange{V});
            coerced.push_back(Box.getResult());
          } else {
            ok = false; break;
          }
        }
        if (!ok) continue;   /* try the next same-name overload */
        B.setInsertionPoint(Call);
        auto Fn = rt(E.rt_name, E.result_ty, E.args);
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn, coerced);
        /* Replace the call's result with the new llvm.call result.
         * The original result type may be tensor while the new is ptr;
         * downstream uses (stores into tensor slots, loads with tensor
         * result type) accept the type mismatch because matlab.alloc /
         * load / store are unregistered ops with no operand-type
         * verification.  On the next iteration of the fixpoint,
         * retypeMatrixSlots sees ptr-typed stores and retypes the
         * slots to llvm.alloca, fixing up the loads too. */
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->erase();
        for (Operation *D : deadLits) if (D->use_empty()) D->erase();
        Changed = true;
        matched = true;
        break;
      }
      if (matched) continue;
    }

    /* Vector-y form: y0 is a ptr (matrix) instead of f64. Routes to the
     * matlab_ode{45,23}_v_* runtime entries which take a matrix RHS. */
    if (NA && (NA.getValue().getSExtValue() == 2 ||
               NA.getValue().getSExtValue() == 3) &&
        (Name == "ode45" || Name == "ode23" || Name == "ode23s") &&
        (Call->getNumOperands() == 3 || Call->getNumOperands() == 4) &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy &&
        Call->getOperand(2).getType() == PtrTy) {
      bool HaveOpts = (Call->getNumOperands() == 4);
      if (HaveOpts && Call->getOperand(3).getType() != PtrTy) {
        /* fall through */
      } else {
        const char *Suffix = HaveOpts ? "_opts" : "";
        std::string F0n = "matlab_" + Name.str() + "_v_t"     + Suffix;
        std::string F1n = "matlab_" + Name.str() + "_v_y"     + Suffix;
        std::string F2n = "matlab_" + Name.str() + "_v_stats" + Suffix;
        llvm::SmallVector<Type, 4> ArgTys = {PtrTy, PtrTy, PtrTy};
        if (HaveOpts) ArgTys.push_back(PtrTy);
        B.setInsertionPoint(Call);
        auto Fn0 = rt(F0n, PtrTy, ArgTys);
        auto Fn1 = rt(F1n, PtrTy, ArgTys);
        ValueRange Args = Call->getOperands();
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn0, Args);
        auto C1 = LLVM::CallOp::create(B, Call->getLoc(), Fn1, Args);
        Call->getResult(0).replaceAllUsesWith(C0.getResult());
        Call->getResult(1).replaceAllUsesWith(C1.getResult());
        if (NA.getValue().getSExtValue() == 3) {
          auto Fn2 = rt(F2n, PtrTy, ArgTys);
          auto C2 = LLVM::CallOp::create(B, Call->getLoc(), Fn2, Args);
          Call->getResult(2).replaceAllUsesWith(C2.getResult());
        }
        Call->erase();
        Changed = true;
        continue;
      }
    }
    /* [t, y] = ode45(@f, tspan, y0) / ode23: handle is a ptr (function
     * pointer materialised by LowerAnonCalls or rewriteMakeHandle), tspan
     * is a ptr (matrix), y0 is f64. Two single-output runtime entries
     * (matlab_ode45_t / matlab_ode45_y) share a thread-local cache so the
     * second call returns the paired column without re-integrating. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Call->getNumOperands() == 3 && Call->getNumResults() == 2 &&
        (Name == "ode45" || Name == "ode23" || Name == "ode23s") &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy &&
        Call->getOperand(2).getType() == F64) {
      StringRef F0 = (Name == "ode45") ? StringRef("matlab_ode45_t") :
                     (Name == "ode23s") ? StringRef("matlab_ode23s_t")
                                        : StringRef("matlab_ode23_t");
      StringRef F1 = (Name == "ode45") ? StringRef("matlab_ode45_y") :
                     (Name == "ode23s") ? StringRef("matlab_ode23s_y")
                                        : StringRef("matlab_ode23_y");
      B.setInsertionPoint(Call);
      auto Fn0 = rt(F0, PtrTy, {PtrTy, PtrTy, F64});
      auto Fn1 = rt(F1, PtrTy, {PtrTy, PtrTy, F64});
      ValueRange Args = Call->getOperands();
      auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn0, Args);
      auto C1 = LLVM::CallOp::create(B, Call->getLoc(), Fn1, Args);
      Call->getResult(0).replaceAllUsesWith(C0.getResult());
      Call->getResult(1).replaceAllUsesWith(C1.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* 4-arg form: `[t,y] = ode45(@f, tspan, y0, opts)` where opts is a
     * struct (ptr) carrying RelTol / AbsTol. Routes to the _opts runtime
     * entries which dereference the struct and override the defaults. */
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Call->getNumOperands() == 4 && Call->getNumResults() == 2 &&
        (Name == "ode45" || Name == "ode23" || Name == "ode23s") &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy &&
        Call->getOperand(2).getType() == F64 &&
        Call->getOperand(3).getType() == PtrTy) {
      StringRef F0 = (Name == "ode45") ? StringRef("matlab_ode45_t_opts") :
                     (Name == "ode23s") ? StringRef("matlab_ode23s_t_opts")
                                        : StringRef("matlab_ode23_t_opts");
      StringRef F1 = (Name == "ode45") ? StringRef("matlab_ode45_y_opts") :
                     (Name == "ode23s") ? StringRef("matlab_ode23s_y_opts")
                                        : StringRef("matlab_ode23_y_opts");
      B.setInsertionPoint(Call);
      auto Fn0 = rt(F0, PtrTy, {PtrTy, PtrTy, F64, PtrTy});
      auto Fn1 = rt(F1, PtrTy, {PtrTy, PtrTy, F64, PtrTy});
      ValueRange Args = Call->getOperands();
      auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn0, Args);
      auto C1 = LLVM::CallOp::create(B, Call->getLoc(), Fn1, Args);
      Call->getResult(0).replaceAllUsesWith(C0.getResult());
      Call->getResult(1).replaceAllUsesWith(C1.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    /* 3-return form: `[t, y, stats] = ode45(@f, tspan, y0[, opts])`.
     * The third result is a fresh matlab_struct* with nsteps/nfailed/
     * nfevals fields. All three calls share the cache, so only the
     * first solve runs. */
    if (NA && NA.getValue().getSExtValue() == 3 &&
        Call->getNumResults() == 3 &&
        (Name == "ode45" || Name == "ode23" || Name == "ode23s") &&
        Call->getOperand(0).getType() == PtrTy &&
        Call->getOperand(1).getType() == PtrTy &&
        Call->getOperand(2).getType() == F64 &&
        (Call->getNumOperands() == 3 ||
         (Call->getNumOperands() == 4 &&
          Call->getOperand(3).getType() == PtrTy))) {
      bool HaveOpts = (Call->getNumOperands() == 4);
      const char *Suffix = HaveOpts ? "_opts" : "";
      std::string F0n = "matlab_" + Name.str() + "_t"     + Suffix;
      std::string F1n = "matlab_" + Name.str() + "_y"     + Suffix;
      std::string F2n = "matlab_" + Name.str() + "_stats" + Suffix;
      llvm::SmallVector<Type, 4> ArgTys = {PtrTy, PtrTy, F64};
      if (HaveOpts) ArgTys.push_back(PtrTy);
      B.setInsertionPoint(Call);
      auto Fn0 = rt(F0n, PtrTy, ArgTys);
      auto Fn1 = rt(F1n, PtrTy, ArgTys);
      auto Fn2 = rt(F2n, PtrTy, ArgTys);
      ValueRange Args = Call->getOperands();
      auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn0, Args);
      auto C1 = LLVM::CallOp::create(B, Call->getLoc(), Fn1, Args);
      auto C2 = LLVM::CallOp::create(B, Call->getLoc(), Fn2, Args);
      Call->getResult(0).replaceAllUsesWith(C0.getResult());
      Call->getResult(1).replaceAllUsesWith(C1.getResult());
      Call->getResult(2).replaceAllUsesWith(C2.getResult());
      Call->erase();
      Changed = true;
      continue;
    }
    if (NA && NA.getValue().getSExtValue() == 2 &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 2 &&
        Call->getOperand(0).getType() == PtrTy) {
      /* [r, c] = size(A): two f64 results, one ptr input. Split into
       * two matlab_size_dim(A, 1) / matlab_size_dim(A, 2) calls —
       * cheaper than a multi-return runtime entry, and reuses the
       * existing size_dim primitive the single-return size(A, dim)
       * already goes through. */
      if (Name == "size") {
        B.setInsertionPoint(Call);
        auto Fn = rt("matlab_size_dim", F64, {PtrTy, F64});
        auto D1 = LLVM::ConstantOp::create(B, Call->getLoc(), F64,
                                            B.getF64FloatAttr(1.0));
        auto D2 = LLVM::ConstantOp::create(B, Call->getLoc(), F64,
                                            B.getF64FloatAttr(2.0));
        auto C0 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0), D1});
        auto C1 = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        ValueRange{Call->getOperand(0), D2});
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        if (Call->getResult(1).getType() != F64)
          Call->getResult(1).setType(F64);
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
      {"zeros",      "matlab_zeros4",     1, "ffff"},
      {"zeros",      "matlab_zeros5",     1, "fffff"},
      {"zeros",      "matlab_zeros6",     1, "ffffff"},
      {"ones",       "matlab_ones",       1, "ff"},
      {"ones",       "matlab_ones3",      1, "fff"},
      {"ones",       "matlab_ones4",      1, "ffff"},
      {"ones",       "matlab_ones5",      1, "fffff"},
      {"ones",       "matlab_ones6",      1, "ffffff"},
      {"eye",        "matlab_eye",        1, "ff"},
      {"magic",      "matlab_magic",      1, "f"},
      {"rand",       "matlab_rand",       1, "ff"},
      {"randn",      "matlab_randn",      1, "ff"},
      /* gpuArray.<static> with NO dtype tag — strict 2-arg shape. */
      {"gpuArray_rand",     "matlab_rand",              1, "ff"},
      {"gpuArray_randn",    "matlab_randn",             1, "ff"},
      {"gpuArray_zeros",    "matlab_zeros",             1, "ff"},
      {"gpuArray_ones",     "matlab_ones",              1, "ff"},
      {"gpuArray_eye",      "matlab_eye",               1, "ff"},
      {"gpuArray_linspace", "matlab_gpuArray_linspace", 1, "fff"},
      {"gpuArray_linspace", "matlab_gpuArray_linspace2",1, "ff"},
      {"sum",        "matlab_sum",        1, "p"},
      {"sum",        "matlab_sum_dim",    1, "pf"},
      {"sum",        "matlab_sum_dims",   1, "pp"},
      {"prod",       "matlab_prod",       1, "p"},
      {"prod",       "matlab_prod_dim",   1, "pf"},
      {"prod",       "matlab_prod_dims",  1, "pp"},
      {"mean",       "matlab_mean",       1, "p"},
      {"mean",       "matlab_mean_dim",   1, "pf"},
      {"mean",       "matlab_mean_dims",  1, "pp"},
      {"min",        "matlab_min",        1, "p"},
      {"min",        "matlab_min_mm",     1, "pp"},  /* min(A, B) elementwise */
      {"min",        "matlab_min_dim3",   1, "ppf"}, /* min(A, [], dim) */
      {"max",        "matlab_max",        1, "p"},
      {"max",        "matlab_max_mm",     1, "pp"},  /* max(A, B) elementwise */
      {"max",        "matlab_max_dim3",   1, "ppf"}, /* max(A, [], dim) */
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
      {"ipermute",   "matlab_ipermute",   1, "pp"},
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
      /* Phase 1.1.C — native int matrix casts.
       * Scalar forms (matlab_int32_s etc., handled by the Scalar map
       * above) operate on f64 and stay f64. Matrix forms produce
       * typed matlab_mat_i32* / matlab_mat_u8* descriptors. */
      {"int32",      "matlab_mat_i32_from_double", 1, "p"},
      {"uint8",      "matlab_mat_u8_from_double",  1, "p"},
      {"transpose",  "matlab_transpose",  1, "p"},
      {"ctranspose", "matlab_transpose",  1, "p"},
      {"diag",       "matlab_diag",       1, "p"},
      {"reshape",    "matlab_reshape",    1, "pff"},
      {"reshape",    "matlab_reshape3",   1, "pfff"},  /* reshape(A,m,n,p) */
      {"reshape",    "matlab_reshape4",   1, "pffff"}, /* reshape(A,d1,d2,d3,d4) */
      {"repmat",     "matlab_repmat",     1, "pff"},
      {"repmat",     "matlab_repmat3",    1, "pfff"},  /* repmat(A,r,c,p) */
      {"exp",        "matlab_exp_m",      1, "p"},
      {"log",        "matlab_log_m",      1, "p"},
      {"sin",        "matlab_sin_m",      1, "p"},
      {"cos",        "matlab_cos_m",      1, "p"},
      {"tan",        "matlab_tan_m",      1, "p"},
      {"asin",       "matlab_asin_m",     1, "p"},
      {"acos",       "matlab_acos_m",     1, "p"},
      {"atan",       "matlab_atan_m",     1, "p"},
      {"sind",       "matlab_sind_m",     1, "p"},
      {"cosd",       "matlab_cosd_m",     1, "p"},
      {"tand",       "matlab_tand_m",     1, "p"},
      {"asind",      "matlab_asind_m",    1, "p"},
      {"acosd",      "matlab_acosd_m",    1, "p"},
      {"atand",      "matlab_atand_m",    1, "p"},
      {"sinh",       "matlab_sinh_m",     1, "p"},
      {"cosh",       "matlab_cosh_m",     1, "p"},
      {"tanh",       "matlab_tanh_m",     1, "p"},
      {"log2",       "matlab_log2_m",     1, "p"},
      {"log10",      "matlab_log10_m",    1, "p"},
      {"sqrt",       "matlab_sqrt_m",     1, "p"},
      /* abs_c is polymorphic — accepts both real and complex. Routing
       * abs() through it keeps abs(complex) well-typed without a
       * separate dispatch entry while the real fast path still
       * collapses to the scalar math fn via the Scalar map above. */
      {"abs",        "matlab_abs_c",      1, "p"},
      {"sign",       "matlab_sign_m",     1, "p"},
      {"floor",      "matlab_floor_m",    1, "p"},
      {"ceil",       "matlab_ceil_m",     1, "p"},
      {"round",      "matlab_round_m",    1, "p"},
      {"fix",        "matlab_fix_m",      1, "p"},
      {"linspace",   "matlab_linspace",   1, "fff"},
      {"logspace",   "matlab_logspace",   1, "fff"},
      {"mod",        "matlab_mod_s",      0, "ff"},
      {"rem",        "matlab_rem_s",      0, "ff"},
      {"atan2",      "matlab_atan2_m",    1, "pp"},
      {"inv",        "matlab_inv",        1, "p"},
      {"det",        "matlab_det",        0, "p"},
      {"svd",        "matlab_svd",        1, "p"},
      {"eig",        "matlab_eig",        1, "p"},
      {"expm",       "matlab_expm",       1, "p"},
      {"logm",       "matlab_logm",       1, "p"},
      {"hess",       "matlab_hess",       1, "p"},
      {"schur",      "matlab_schur",      1, "p"},
      {"lyap",       "matlab_lyap",       1, "pp"},
      /* 3-arg form A·X + X·B + C = 0 — surfaces as `lyap(A, B, C)` in
       * MATLAB (same name, different arity, different equation). */
      {"lyap",       "matlab_sylvester",  1, "ppp"},
      {"dlyap",      "matlab_dlyap",      1, "pp"},
      {"lyapchol",   "matlab_lyapchol",   1, "pp"},
      {"care",       "matlab_care",       1, "pppp"},
      {"dare",       "matlab_dare",       1, "pppp"},
      {"icare",      "matlab_icare",      1, "pppp"},
      {"idare",      "matlab_idare",      1, "pppp"},
      /* 5-arg cross-term forms `care(A, B, Q, R, S)` / `dare(...)`. */
      {"care",       "matlab_care_5",     1, "ppppp"},
      {"dare",       "matlab_dare_5",     1, "ppppp"},
      {"icare",      "matlab_care_5",     1, "ppppp"},
      {"idare",      "matlab_dare_5",     1, "ppppp"},
      {"lqr",        "matlab_lqr",        1, "pppp"},
      {"dlqr",       "matlab_dlqr",       1, "pppp"},
      /* 5-arg cross-term LQR / DLQR. Wraps care_5 / dare_5 with the
       * matching gain-extraction algebra. */
      {"lqr",        "matlab_lqr_5",      1, "ppppp"},
      {"dlqr",       "matlab_dlqr_5",     1, "ppppp"},
      /* Output-weighted LQR. Model-object dispatch emits this. */
      {"lqry_ss",    "matlab_lqry_ss",    1, "pppppp"},
      {"ctrb",       "matlab_ctrb",       1, "pp"},
      {"obsv",       "matlab_obsv",       1, "pp"},
      {"place",      "matlab_place",      1, "ppp"},
      /* `acker(A, B, p)` — Ackermann's-formula pole placement. Same
       * runtime entry as `place`; the difference is purely
       * pedagogical (acker advertises SISO single-input). */
      {"acker",      "matlab_place",      1, "ppp"},
      {"isstable",   "matlab_isstable",   0, "p"},
      {"damp",       "matlab_damp",       1, "p"},
      {"hsvd",       "matlab_hsvd",       1, "ppp"},
      {"balreal_T",  "matlab_balreal_T",  1, "ppp"},
      {"balred_A",   "matlab_balred_A",   1, "pppf"},
      {"balred_B",   "matlab_balred_B",   1, "pppf"},
      {"balred_C",   "matlab_balred_C",   1, "pppf"},
      /* balred 1-return defaults to Ar (the more-useful default for
       * stability/eig analysis). The 3-return shape goes through the
       * dedicated splitter above. */
      {"balred",     "matlab_balred_A",   1, "pppf"},
      {"norm_h2",    "matlab_norm_h2",    0, "ppp"},
      {"dcgain_ss",  "matlab_dcgain_ss",  1, "pppp"},
      {"dcgain_tf",  "matlab_dcgain_tf",  1, "pp"},
      {"bandwidth_tf","matlab_bandwidth_tf",0,"pp"},
      {"stepinfo",   "matlab_stepinfo_struct", 1, "pp"},
      {"kalman_L",   "matlab_kalman_L",   1, "ppppp"},
      {"kalman_P",   "matlab_kalman_P",   1, "ppppp"},
      {"kalmd_L",    "matlab_kalmd_L",    1, "ppppp"},
      /* 1-return forms default to L (the gain — most-used output). The
       * 2-return [L, P] shape goes through the splitter above. */
      {"kalman",     "matlab_kalman_L",   1, "ppppp"},
      {"kalmd",      "matlab_kalmd_L",    1, "ppppp"},
      {"isstable_d", "matlab_isstable_d", 0, "p"},
      {"norm_h2_d",  "matlab_norm_h2_d",  0, "pppp"},
      /* c2d_tustin 1-return defaults to Ad (the more-useful default for
       * stability/eig analysis). The 2-return shape goes through the
       * dedicated splitter above. */
      {"c2d_tustin", "matlab_c2d_tustin_Ad", 1, "ppf"},
      /* d2c (ZOH) 1-return defaults to the continuous A matrix. */
      {"d2c",        "matlab_d2c_A",         1, "ppf"},
      /* d2c_tustin 1-return defaults to A. */
      {"d2c_tustin", "matlab_d2c_tustin_A",  1, "ppf"},
      {"gram_c",     "matlab_gram_c",     1, "pp"},
      {"gram_o",     "matlab_gram_o",     1, "pp"},
      {"step_ss",    "matlab_step_ss",    1, "ppppff"},
      /* step(sys, t) / step(tf, t) honouring a supplied time vector. */
      {"step_ss_t",  "matlab_step_ss_t",  1, "ppppp"},
      {"step_tf_t",  "matlab_step_tf_t",  1, "ppp"},
      /* §3.3 follow-ons — impulse / initial response. Same arg-shape
       * convention as step_ss; initial_ss carries x0 between (D, dt). */
      {"impulse_ss", "matlab_impulse_ss", 1, "ppppff"},
      {"initial_ss", "matlab_initial_ss", 1, "pppppff"},
      /* bode_ss 1-return form returns magnitude (the more-useful default
       * for plotting). The 2-return [mag, phase] = bode_ss(...) shape
       * goes through the dedicated splitter above. */
      {"bode_ss",    "matlab_bode_ss_mag",1, "ppppp"},
      /* Per-output bode entries for the model-object [mag,phase,wout]
       * multi-return splitter (Lowering.cpp). */
      {"bode_ss_mag",  "matlab_bode_ss_mag",  1, "ppppp"},
      {"bode_ss_phase","matlab_bode_ss_phase",1, "ppppp"},
      {"bode_tf_mag",  "matlab_bode_tf_mag",  1, "ppp"},
      {"bode_tf_phase","matlab_bode_tf_phase",1, "ppp"},
      {"lsim_ss",    "matlab_lsim_ss",    1, "pppppf"},
      /* §3.4 follow-ons — raw complex freqresp + nyquist (re/im
       * columns) + allmargin (1×4 row). freqresp / nyquist accept
       * either matrix-arg (ss / tf) or model-object call sites; the
       * model-object dispatch lives in Lowering.cpp. */
      {"freqresp_ss","matlab_freqresp_ss",1, "ppppp"},
      {"freqresp_tf","matlab_freqresp_tf",1, "ppp"},
      {"nyquist_ss", "matlab_nyquist_ss", 1, "ppppp"},
      {"nyquist_tf", "matlab_nyquist_tf", 1, "ppp"},
      {"allmargin_ss","matlab_allmargin_ss",1, "ppppp"},
      {"margin_ss_auto","matlab_margin_ss_auto",1,"pppp"},
      {"margin_tf_auto","matlab_margin_tf_auto",1,"pp"},
      {"gain_margin","matlab_gain_margin",0, "ppppp"},
      {"phase_margin","matlab_phase_margin",0,"ppppp"},
      {"bandwidth_ss","matlab_bandwidth_ss",0,"pppp"},
      {"getPeakGain_ss","matlab_getPeakGain_ss",0,"pppp"},
      {"pole",       "matlab_eig",        1, "p"},
      /* Generalised eig(A, B): 2-arg form routes to matlab_eig_gen
       * which returns the (possibly complex) spectrum of the pencil
       * A − λB. The 1-arg `eig(A)` keeps the existing dispatch. */
      {"eig",        "matlab_eig_gen",    1, "pp"},
      /* feedback_ss 1-return defaults to Acl (closed-loop A — most-
       * useful for stability/eig analysis). 3-return goes through the
       * dedicated splitter above. */
      {"feedback_ss","matlab_feedback_ss_A",1,"pppppp"},
      {"series_ss",  "matlab_series_ss_A", 1,"pppppp"},
      {"parallel_ss","matlab_parallel_ss_A",1,"pppppp"},
      {"append_ss",  "matlab_append_ss_A", 1,"pppppp"},
      /* §3.1 / §5.2 — direct-runtime-symbol passthroughs the class-
       * returning model-object short forms in Lowering.cpp emit
       * (c2d(sys, Ts), feedback / series / parallel / append /
       * blkdiag on ss model objects). Each Lowering site picks the
       * matching _A / _B / _C runtime entry directly so the result
       * fits into the ss(_,_,_,_) constructor call. */
      {"matlab_c2d_Ad",        "matlab_c2d_Ad",        1, "ppf"},
      {"matlab_c2d_Bd",        "matlab_c2d_Bd",        1, "ppf"},
      /* tf-object c2d / d2c (#27) — the tf class-pinned-first-arg sites in
       * Lowering.cpp emit these to round-trip (num, den) through the
       * tf2ss → discretise → ss2tf path. c2d takes Ts as an f64 (from the
       * call arg); d2c reads Ts off the model's Ts property (boxed 1×1). */
      {"matlab_c2d_tf_num",        "matlab_c2d_tf_num",        1, "ppf"},
      {"matlab_c2d_tf_den",        "matlab_c2d_tf_den",        1, "ppf"},
      {"matlab_c2d_tf_tustin_num", "matlab_c2d_tf_tustin_num", 1, "ppf"},
      {"matlab_c2d_tf_tustin_den", "matlab_c2d_tf_tustin_den", 1, "ppf"},
      {"matlab_d2c_tf_num",        "matlab_d2c_tf_num",        1, "ppp"},
      {"matlab_d2c_tf_den",        "matlab_d2c_tf_den",        1, "ppp"},
      /* Sys-form Kalman dispatcher — class-pinned-first-arg site in
       * Lowering.cpp extracts A/B/C off an `ss` and passes Ts (boxed
       * 1×1 matrix, matching the matrix-storage convention for class
       * scalar properties).  The dispatcher unboxes and picks the
       * continuous (Ts == 0) or discrete kernel. */
      {"matlab_kalman_sys_L",  "matlab_kalman_sys_L",  1, "pppppp"},
      /* MPC Toolbox Tier-1/2/3 — runtime entries called from inside
       * `mpc_classdefs.m`.  `construct` returns a dummy empty matrix
       * (caller discards). */
      {"matlab_mpc_construct",     "matlab_mpc_construct",     1, "ppff"},
      {"matlab_mpc_move",          "matlab_mpc_move",          1, "pppp"},
      {"matlab_mpc_move_opt",      "matlab_mpc_move_opt",      1, "ppppp"},
      {"matlab_mpc_move_adaptive", "matlab_mpc_move_adaptive", 1, "ppppppp"},
      {"matlab_mpc_move_tv",       "matlab_mpc_move_tv",       1, "ppppppp"},
      {"matlab_mpc_sim",           "matlab_mpc_sim",           1, "pfp"},
      /* MPC Tier-4 §5.4 — standalone KWIK active-set QP.  User-facing
       * `mpcActiveSetSolver(H, f, A, b)` aliases to the runtime
       * symbol. */
      {"matlab_mpc_active_set",    "matlab_mpc_active_set",    1, "pppp"},
      {"mpcActiveSetSolver",       "matlab_mpc_active_set",    1, "pppp"},
      /* MPC Tier-4 §5.1/5.2/5.3 — explicit MPC. */
      {"matlab_mpc_generate_explicit", "matlab_mpc_generate_explicit",
                                                                1, "ppppfp"},
      {"matlab_mpc_move_explicit",     "matlab_mpc_move_explicit",
                                                                1, "pp"},
      {"matlab_mpc_simplify_explicit", "matlab_mpc_simplify_explicit",
                                                                1, "pf"},
      /* MPC Tier-4 §5.7 — Finite Control Set MPC. */
      {"matlab_mpc_move_finite",       "matlab_mpc_move_finite",  1, "pppp"},
      /* MPC Tier-5 — Nonlinear MPC.  5th arg is the function-handle
       * void* (StateFcn). */
      {"matlab_nlmpc_move",            "matlab_nlmpc_move",       1, "ppppp"},
      /* MPC Tier-6 §7.5 — review() sanity diagnostic. */
      {"matlab_mpc_review",            "matlab_mpc_review",       1, "p"},
      /* MPC Tier-6 §7.6 — sim() with mpcsimopt override. */
      {"matlab_mpc_sim_opt",           "matlab_mpc_sim_opt",      1, "pfpp"},
      {"matlab_feedback_ss_A", "matlab_feedback_ss_A", 1, "pppppp"},
      {"matlab_feedback_ss_B", "matlab_feedback_ss_B", 1, "pppppp"},
      {"matlab_feedback_ss_C", "matlab_feedback_ss_C", 1, "pppppp"},
      {"matlab_series_ss_A",   "matlab_series_ss_A",   1, "pppppp"},
      {"matlab_series_ss_B",   "matlab_series_ss_B",   1, "pppppp"},
      {"matlab_series_ss_C",   "matlab_series_ss_C",   1, "pppppp"},
      {"matlab_parallel_ss_A", "matlab_parallel_ss_A", 1, "pppppp"},
      {"matlab_parallel_ss_B", "matlab_parallel_ss_B", 1, "pppppp"},
      {"matlab_parallel_ss_C", "matlab_parallel_ss_C", 1, "pppppp"},
      {"matlab_append_ss_A",   "matlab_append_ss_A",   1, "pppppp"},
      {"matlab_append_ss_B",   "matlab_append_ss_B",   1, "pppppp"},
      {"matlab_append_ss_C",   "matlab_append_ss_C",   1, "pppppp"},
      /* §5.1 — sminreal / modred direct runtime callees emitted by
       * the model-object short-form dispatch in Lowering.cpp. */
      {"matlab_sminreal_A", "matlab_sminreal_A", 1, "ppp"},
      {"matlab_sminreal_B", "matlab_sminreal_B", 1, "ppp"},
      {"matlab_sminreal_C", "matlab_sminreal_C", 1, "ppp"},
      {"matlab_modred_A",   "matlab_modred_A",   1, "ppppf"},
      {"matlab_modred_B",   "matlab_modred_B",   1, "ppppf"},
      {"matlab_modred_C",   "matlab_modred_C",   1, "ppppf"},
      {"matlab_mat_from_scalar", "matlab_mat_from_scalar", 1, "f"},
      /* bode_tf 1-return form returns magnitude (default for plotting).
       * The 2-return [mag, phase] = bode_tf(...) shape goes through the
       * dedicated splitter above. */
      {"bode_tf",    "matlab_bode_tf_mag",1, "ppp"},
      {"isequal",    "matlab_isequal",    0, "pp"},
      {"size",       "matlab_size_dim",   0, "pf"},   /* size(A, dim) */
      {"find",       "matlab_find",       1, "p"},
      {"matlab_empty_mat", "matlab_empty_mat", 1, ""},
      /* Complex builtins. Operand kind 'p' accepts either a matlab_mat*
       * or a matlab_mat_c* — the runtime side dispatches on the layout.
       * conj / fft / ifft / fft2 / ifft2 return complex (ptr); real /
       * imag / angle return a real matrix (also ptr but matlab_mat*). */
      /* complex(re, im): build a 1x1 matlab_mat_c from two scalars,
       * mirroring the literal `re + im*i` lowering at line ~361. */
      {"complex",    "matlab_complex_scalar", 1, "ff"},
      /* Matrix-arg variants for building a complex column from real
       * re/im columns (or broadcasting a scalar against a column).
       * Closes the `1i * real_col` ergonomics gap for ZPK / vector
       * fitting / signal-processing workflows. */
      {"complex",    "matlab_complex_mm",     1, "pp"},
      {"complex",    "matlab_complex_sm",     1, "fp"},
      {"complex",    "matlab_complex_ms",     1, "pf"},
      {"conj",       "matlab_conj_c",     1, "p"},
      {"real",       "matlab_real_c",     1, "p"},
      {"imag",       "matlab_imag_c",     1, "p"},
      {"angle",      "matlab_angle_c",    1, "p"},
      {"fft",        "matlab_fft_c",      1, "p"},
      {"ifft",       "matlab_ifft_c",     1, "p"},
      {"fft2",       "matlab_fft2_c",     1, "p"},
      {"ifft2",      "matlab_ifft2_c",    1, "p"},
      /* Convolution. Both operands are matrices (vector layout for conv). */
      {"conv",       "matlab_conv",       1, "pp"},
      {"conv2",      "matlab_conv2",      1, "pp"},
      /* Tier-C rank-4 batched conv: X(H,W,C,N) * W(kH,kW,C,K) -> Y(H',W',K,N). */
      {"conv2d_batch", "matlab_conv2d_batch", 1, "pp"},
      /* Batched 2-D matmul: A(M, K, B) * B(K, N, B) -> Y(M, N, B). */
      {"matmul3",      "matlab_matmul3",     1, "pp"},
      /* Full conv: + bias (K-vec), pad_h, pad_w, stride_h, stride_w. */
      {"conv2d_batch_full", "matlab_conv2d_batch_full", 1, "pppffff"},
      /* Tape-scoping convenience wrappers (user-facing names).
       * dlreset / dltape_truncate return ptr (empty matrix); dltape_size
       * returns scalar f64 (the current node count). */
      {"dlreset",          "matlab_dlnet_reset0",    1, ""},
      {"dltape_size",      "matlab_dltape_size",     0, "f"},
      {"dltape_truncate",  "matlab_dltape_truncate", 1, "f"},
      /* Functional optimizers — return a new W ptr; m / v / [adam's t]
       * are updated in place by the runtime. */
      {"sgdmupdate",       "matlab_dlnet_sgdmupdate", 1, "pppff"},
      {"adamupdate",       "matlab_dlnet_adamupdate", 1, "ppppfffff"},
      {"rmspropupdate",    "matlab_dlnet_rmspropupdate", 1, "pppfff"},
      /* Magnitude-based pruning helpers. */
      {"prune_mask",       "matlab_dlnet_prune_mask",    1, "pf"},
      {"mask_sparsity",    "matlab_dlnet_mask_sparsity", 0, "p"},
      /* Experiment harness — first arg is a function handle (passed
       * through as a raw ptr by LowerAnonCalls' retype path, mirroring
       * bayesopt). */
      {"runExperiment",    "matlab_dlnet_run_experiment", 1, "pp"},
      /* T1.8: numpartitions returns a scalar f64; the string-arg helpers
       * (mkdir/imageDatastore/countEachLabel/splitEachLabel) sit in the
       * typed Spec table above (line ~6168) so const_char → matlab_string
       * promotion can fire on the folder/path literal. */
      {"numpartitions",    "matlab_dlnet_imds_numfiles",  0, "p"},
      /* ONNX introspection — scalar f64 returns. */
      {"onnxNumNodes",     "matlab_onnx_num_nodes",       0, "p"},
      {"onnxNumInits",     "matlab_onnx_num_inits",       0, "p"},
      {"onnxOpset",        "matlab_onnx_opset",           0, "p"},
      /* T3.8 — GPU training dispatch toggle + introspection. */
      {"dlnetGpu",         "matlab_dlnet_gpu_set",        1, "f"},
      {"dlnetGpuActive",   "matlab_dlnet_gpu_get",        0, "f"},
      /* F: generic obj-array carrier (literal-free object arrays). */
      {"objArrayNew",      "matlab_dlnet_oa_new",         1, ""},
      {"objArrayAppend",   "matlab_dlnet_oa_append",      1, "pp"},
      {"objArrayLen",      "matlab_dlnet_oa_len",         0, "p"},
      {"objArrayGet",      "matlab_dlnet_oa_get",         1, "pf"},
      /* C: dlnetwork carrier — sequential layer driver. */
      {"dlnetwork",        "matlab_dlnet_net_new",        1, ""},
      {"addFC",            "matlab_dlnet_net_add_fc",     1, "ppp"},
      {"addRelu",          "matlab_dlnet_net_add_relu",   1, "p"},
      {"addSigmoid",       "matlab_dlnet_net_add_sigmoid",1, "p"},
      {"addTanh",          "matlab_dlnet_net_add_tanh",   1, "p"},
      {"addSoftmax",       "matlab_dlnet_net_add_softmax",1, "p"},
      {"netPredict",       "matlab_dlnet_net_predict",    1, "pp"},
      {"netNumLayers",     "matlab_dlnet_net_num_layers", 0, "p"},
      {"trainnet",         "matlab_dlnet_net_train",      0, "pppff"},
      /* im2col helper exposed to user code so callers can write their
       * own GEMM-based conv (e.g. depthwise, dilation, stride>1). */
      {"im2col_2d",    "matlab_im2col_2d",    1, "pff"},
      {"im2col_2d_pad","matlab_im2col_2d_pad",1, "pffffff"},
      /* Tier-1 builtins added alongside conv. filter is the IIR/FIR
       * difference equation (3 ptr args). The fftshift pair is
       * polymorphic on real/complex via the matlab_mat_c magic. */
      {"filter",     "matlab_filter",     1, "ppp"},
      {"any",        "matlab_any",        1, "p"},
      {"all",        "matlab_all",        1, "p"},
      {"tril",       "matlab_tril",       1, "p"},
      {"triu",       "matlab_triu",       1, "p"},
      {"fftshift",   "matlab_fftshift_c", 1, "p"},
      {"ifftshift",  "matlab_ifftshift_c",1, "p"},
      {"std",        "matlab_std",        1, "p"},
      {"var",        "matlab_var",        1, "p"},
      {"median",     "matlab_median",     1, "p"},
      {"diff",       "matlab_diff",       1, "p"},
      /* meshgrid/ndgrid one-arg form: meshgrid(x) == meshgrid(x, x).
       * The multi-return [X,Y]=... form has its own splitter above. */
      {"meshgrid",   "matlab_meshgrid_X", 1, "p"},
      {"ndgrid",     "matlab_ndgrid_X",   1, "p"},
      /* Tier-2: signal-processing, polynomial, numeric calculus. */
      {"xcorr",      "matlab_xcorr",      1, "pp"},
      {"polyval",    "matlab_polyval",    1, "pp"},
      {"polyfit",    "matlab_polyfit",    1, "ppf"},
      {"roots",      "matlab_roots",      1, "p"},
      {"poly",       "matlab_poly",       1, "p"},
      {"polyder",    "matlab_polyder",    1, "p"},
      {"polyint",    "matlab_polyint",    1, "p"},
      {"polyint",    "matlab_polyint_k",  1, "pf"},
      /* Tier-1 §2.1 IIR — single-return forms. The multi-return
       * `[b, a] = butter(...)` / `[b, a] = cheby1(...)` /
       * `[H, w] = freqz(...)` shapes are handled separately by the
       * dedicated multi-return dispatch above. The single-return
       * `B = butter(...)` form returns just b (matches MATLAB's
       * `[b, a] = butter(n, Wn); B = b;` shorthand). `H = freqz(...)`
       * returns just the complex H. */
      {"butter",     "matlab_butter_b",   1, "ff"},
      {"cheby1",     "matlab_cheby1_b",   1, "fff"},
      {"cheby2",     "matlab_cheby2_b",   1, "fff"},
      {"freqz",      "matlab_freqz",      1, "ppf"},
      /* Order helpers — single-LHS form returns just n (the order)
       * as a scalar f64. Multi-LHS `[n, Wn] = ...` is handled by the
       * dedicated multi-return dispatch above. */
      {"buttord",    "matlab_buttord_n",  0, "ffff"},
      {"cheb1ord",   "matlab_cheb1ord_n", 0, "ffff"},
      {"cheb2ord",   "matlab_cheb2ord_n", 0, "ffff"},
      /* §2.1 follow-on — analog↔digital + form conversions. Single-LHS
       * forms; multi-LHS `[bd, ad] = bilinear(...)`, `[z, p, k] = tf2zp(...)`,
       * `[b, a] = zp2tf(z, p, k)` are handled by the multi-return dispatch
       * above. */
      {"bilinear",   "matlab_bilinear_b", 1, "ppf"},
      {"freqs",      "matlab_freqs",      1, "ppp"},
      {"tf2zp",      "matlab_tf2zp_z",    1, "pp"},
      {"zp2tf",      "matlab_zp2tf_b",    1, "ppf"},
      {"besself",    "matlab_besself_b",  1, "ff"},
      {"tf2sos",     "matlab_tf2sos",     1, "pp"},
      {"sos2tf",     "matlab_sos2tf_b",   1, "p"},
      /* §2.2 FIR design + Savitzky-Golay. */
      {"fir1",       "matlab_fir1",       1, "ff"},
      /* DSP Tier-2 — equiripple / least-squares FIR design (single-return
       * b = the L=order+1 symmetric taps); notch/peak single-return num. */
      {"firpm",      "matlab_dsp_firpm",  1, "fpp"},
      {"firls",      "matlab_dsp_firls",  1, "fpp"},
      {"iirnotch",   "matlab_dsp_iirnotch_b", 1, "ff"},
      {"iirpeak",    "matlab_dsp_iirpeak_b",  1, "ff"},
      /* DSP Tier-5 — buffer(x,n) / buffer(x,n,p) frame segmenter. */
      {"buffer",     "matlab_dsp_buffer2", 1, "pf"},
      {"buffer",     "matlab_dsp_buffer",  1, "pff"},
      /* DSP HDL Tier-8 — element-wise CORDIC math (function-form). */
      {"cordic_atan2", "matlab_dsp_cordic_atan2", 1, "pp"},
      {"cordic_sqrt",  "matlab_dsp_cordic_sqrt",  1, "p"},
      {"sgolay",     "matlab_sgolay",     1, "ff"},
      {"sgolayfilt", "matlab_sgolayfilt", 1, "pff"},
      /* §2.5 close-the-loop filter helpers. */
      {"filtfilt",   "matlab_filtfilt",   1, "ppp"},
      {"sosfilt",    "matlab_sosfilt",    1, "pp"},
      {"impz",       "matlab_impz",       1, "ppf"},
      {"stepz",      "matlab_stepz",      1, "ppf"},
      {"grpdelay",   "matlab_grpdelay",   1, "ppf"},
      /* §3.4 transforms tail. */
      {"dct",        "matlab_dct",        1, "p"},
      {"idct",       "matlab_idct",       1, "p"},
      {"fwht",       "matlab_fwht",       1, "p"},
      {"hilbert",    "matlab_hilbert",    1, "p"},
      {"goertzel",   "matlab_goertzel",   1, "pf"},
      /* §3.1 nonparametric spectral. */
      {"periodogram", "matlab_periodogram", 1, "p"},
      {"pwelch",      "matlab_pwelch",      1, "ppf"},
      /* §3.3 time-frequency. */
      {"spectrogram", "matlab_spectrogram", 1, "ppf"},
      /* §3.2 linear prediction + parametric PSD. */
      {"levinson",    "matlab_levinson",    1, "pf"},
      {"lpc",         "matlab_lpc",         1, "pf"},
      {"aryule",      "matlab_aryule",      1, "pf"},
      {"arburg",      "matlab_arburg",      1, "pf"},
      {"pyulear",     "matlab_pyulear",     1, "pff"},
      {"pburg",       "matlab_pburg",       1, "pff"},
      /* §3.1 cross-spectral helpers. */
      {"cpsd",        "matlab_cpsd",        1, "pppf"},
      {"mscohere",    "matlab_mscohere",    1, "pppf"},
      {"tfestimate",  "matlab_tfestimate",  1, "pppf"},
      /* §4.3 pulse measurements — single-result forms. */
      {"findpeaks",   "matlab_findpeaks_pks", 1, "p"},
      {"rms",         "matlab_rms_s",         0, "p"},
      {"peak2peak",   "matlab_peak2peak_s",   0, "p"},
      {"peak2rms",    "matlab_peak2rms_s",    0, "p"},
      {"rssq",        "matlab_rssq_s",        0, "p"},
      {"medfilt1",    "matlab_medfilt1",      1, "pf"},
      {"hampel",      "matlab_hampel",        1, "pf"},
      {"envelope",    "matlab_envelope",      1, "p"},
      {"midcross",    "matlab_midcross",      1, "p"},
      {"risetime",    "matlab_risetime_s",    0, "p"},
      {"falltime",    "matlab_falltime_s",    0, "p"},
      {"dutycycle",   "matlab_dutycycle_s",   0, "p"},
      /* §4.3 pulse-statistics tail. */
      {"statelevels", "matlab_statelevels",   1, "p"},
      {"slewrate",    "matlab_slewrate_s",    0, "p"},
      {"pulseperiod", "matlab_pulseperiod_s", 0, "p"},
      {"pulsewidth",  "matlab_pulsewidth_s",  0, "p"},
      {"overshoot",   "matlab_overshoot_s",   0, "p"},
      {"undershoot",  "matlab_undershoot_s",  0, "p"},
      {"settlingtime","matlab_settlingtime_s",0, "pf"},
      /* §4.1 multirate. */
      {"upfirdn",     "matlab_upfirdn",       1, "ppff"},
      {"decimate",    "matlab_decimate",      1, "pf"},
      {"interp",      "matlab_interp",        1, "pf"},
      {"resample",    "matlab_resample",      1, "pff"},
      /* §4.2 waveform generators. */
      {"chirp",       "matlab_chirp",         1, "pfff"},
      {"sawtooth",    "matlab_sawtooth",      1, "pf"},
      {"square",      "matlab_square",        1, "pf"},
      {"gauspuls",    "matlab_gauspuls",      1, "pff"},
      {"rectpuls",    "matlab_rectpuls",      1, "pf"},
      {"tripuls",     "matlab_tripuls",       1, "pf"},
      {"sinc",        "matlab_sinc",          1, "p"},
      /* §4.4 alignment helpers. */
      {"xcov",        "matlab_xcov",          1, "pp"},
      {"finddelay",   "matlab_finddelay_s",   0, "pp"},
      {"dtw",         "matlab_dtw_s",         0, "pp"},
      {"interp1",    "matlab_interp1",    1, "ppp"},
      {"trapz",      "matlab_trapz",      1, "p"},
      {"trapz",      "matlab_trapz_xy",   1, "pp"},
      {"cumtrapz",   "matlab_cumtrapz",   1, "p"},
      {"gradient",   "matlab_gradient",   1, "p"},
      {"hamming",    "matlab_hamming",    1, "f"},
      {"hann",       "matlab_hann",       1, "f"},
      {"blackman",   "matlab_blackman",   1, "f"},
      /* Tier-1 windows tail (signal_toolbox_roadmap §2.3). */
      {"rectwin",        "matlab_rectwin",        1, "f"},
      {"triang",         "matlab_triang",         1, "f"},
      {"bartlett",       "matlab_bartlett",       1, "f"},
      {"barthannwin",    "matlab_barthannwin",    1, "f"},
      {"bohmanwin",      "matlab_bohmanwin",      1, "f"},
      {"parzenwin",      "matlab_parzenwin",      1, "f"},
      {"nuttallwin",     "matlab_nuttallwin",     1, "f"},
      {"blackmanharris", "matlab_blackmanharris", 1, "f"},
      {"flattopwin",     "matlab_flattopwin",     1, "f"},
      {"kaiser",         "matlab_kaiser",         1, "ff"},
      {"tukeywin",       "matlab_tukeywin",       1, "ff"},
      {"gausswin",       "matlab_gausswin",       1, "ff"},
      {"chebwin",        "matlab_chebwin",        1, "ff"},
      {"taylorwin",      "matlab_taylorwin",      1, "fff"},
      /* Tier-3: SVD-derived linalg + image-processing wrappers + interp2. */
      {"rank",       "matlab_rank",       0, "p"},
      {"cond",       "matlab_cond",       0, "p"},
      {"null",       "matlab_null",       1, "p"},
      {"orth",       "matlab_orth",       1, "p"},
      {"imfilter",   "matlab_imfilter",   1, "pp"},
      {"padarray",   "matlab_padarray",   1, "pp"},
      {"interp2",    "matlab_interp2",    1, "ppppp"},
      {"upsample",   "matlab_upsample",   1, "pf"},
      {"downsample", "matlab_downsample", 1, "pf"},
      /* === Propagation Models (PROP-Tier 1a/2a/2b/3) ===
       *
       * All entries are function-form, classdef-free; no string
       * selectors (numeric tags are used for env / model / climate
       * choices). See runtime/runtime_prop.cpp and docs/comm_toolbox_roadmap.md §3.
       */
      /* §3.1.5 geographic helpers */
      {"haversine",          "matlab_prop_haversine",     0, "ffff"},
      {"bearing",            "matlab_prop_bearing",       0, "ffff"},
      {"vincenty",           "matlab_prop_vincenty",      0, "ffff"},
      {"greatCircleDestLat", "matlab_prop_dest_lat",      0, "ffff"},
      {"greatCircleDestLon", "matlab_prop_dest_lon",      0, "ffff"},
      /* §3.1.1 ITU-R / NIST closed-form */
      {"fspl",                "matlab_prop_fspl",                0, "ff"},
      {"pathlossRain",        "matlab_prop_pathloss_rain",       0, "ffff"},
      {"pathlossGas",         "matlab_prop_pathloss_gas",        0, "fffff"},
      {"pathlossFog",         "matlab_prop_pathloss_fog",        0, "fff"},
      {"pathlossCloseIn",     "matlab_prop_pathloss_closein",    0, "fffff"},
      /* §3.1.2 cellular empirical */
      {"pathlossHata",        "matlab_prop_pathloss_hata",       0, "fffff"},
      {"pathlossCost231",     "matlab_prop_pathloss_cost231",    0, "fffff"},
      {"pathlossEgli",        "matlab_prop_pathloss_egli",       0, "ffff"},
      {"pathlossEcc33",       "matlab_prop_pathloss_ecc33",      0, "ffff"},
      {"pathlossSui",         "matlab_prop_pathloss_sui",        0, "fffff"},
      {"pathlossEricsson9999","matlab_prop_pathloss_ericsson9999",0,"fffff"},
      /* §3.1.3 / 3.1.4 Fresnel + diffraction */
      {"fresnelZoneRadius",   "matlab_prop_fresnel_zone_radius", 0, "ffff"},
      {"fresnelClearance",    "matlab_prop_fresnel_clearance",   0, "pfffff"},
      {"diffractionKnifeEdge","matlab_prop_diff_knife_edge",     0, "ffff"},
      {"diffractionBullington","matlab_prop_diff_bullington",    0, "pffff"},
      {"diffractionDeygout",  "matlab_prop_diff_deygout",        0, "pffff"},
      /* §3.2 Longley-Rice (ITM) */
      {"itmPathloss",         "matlab_prop_itm_pathloss",        0,
                              "pffffffffffff"},
      /* §3.3.1–3.3.4 terrain, LOS, link budget, single-TX coverage */
      {"terrainProfile",      "matlab_prop_terrain_profile",     1,
                              "pfffffffff"},
      {"losObstruction",      "matlab_prop_los_obstruction",     0, "pfff"},
      {"losClear",            "matlab_prop_los_clear",           0, "pfff"},
      {"linkBudget",          "matlab_prop_link_budget",         1,
                              "fffffffffffpffff"},
      {"coverageGrid",        "matlab_prop_coverage_grid",       1,
                              "fffffffpffffffffffff"},
      /* §3.4 directional patterns + mount + multi-site coverage */
      {"sectorPattern",       "matlab_prop_pat_sector",          0, "ffffff"},
      {"cosinePattern",       "matlab_prop_pat_cosine",          0, "ffffff"},
      {"gaussianPattern",     "matlab_prop_pat_gaussian",        0, "fffff"},
      {"isotropicPattern",    "matlab_prop_pat_isotropic",       0, "fff"},
      {"applyMountOrientation","matlab_prop_mount_to_local",     1, "ffff"},
      {"applyMountAz",        "matlab_prop_mount_az_local",      0, "ffff"},
      {"applyMountEl",        "matlab_prop_mount_el_local",      0, "ffff"},
      {"coverageGridMulti",   "matlab_prop_coverage_grid_multi", 1,
                              "pppffffffffffffff"},
      /* §3.5 PropagationModel classdef dispatcher.  Takes the model
       * kind as a `matlab_string *` (first arg, ptr) + the 6 site
       * scalars (tx/rx lat/lon/height) + frequency, dispatches to the
       * appropriate model in the runtime.  The classdef method
       * `pathloss(pm, rx, tx)` calls this with `pm.Kind` and the
       * site fields. */
      {"propPathlossDispatch", "matlab_prop_dispatch_pathloss",
                                0, "pfffffff"},
      /* `los(tx, rx)` site-aware LOS check — uses Earth-bulge math
       * over haversine distance (k=4/3 model).  Returns 1.0 / 0.0. */
      {"propLosSites",         "matlab_prop_los_sites",
                                0, "ffffff"},
      /* `sigstrength(rx, tx, pm)` — now a real MATLAB-side method
       * on RxSite (see rf_class_rxsite.m).  The inter-procedural
       * class-pinning pass in Resolver carries the call-site `pm`
       * pin into the method body's `pm` parameter, so the dispatch
       * to `pathloss(pm, rx, tx)` inside the body routes through
       * the PropagationModel method as expected.  The runtime
       * helper `matlab_prop_sigstrength` stays available as a back-
       * door for cases that don't go through Sema. */
      {"matlab_prop_sigstrength", "matlab_prop_sigstrength",
                                  0, "ppp"},
      /* `siteviewer(...)` — text-only stub.  Returns 0.  Lets
       * MathWorks tutorial code with `viewer = siteviewer;` calls
       * compile cleanly even though we have no GUI. */
      {"siteviewer", "matlab_prop_siteviewer_stub", 0, ""},
      /* Translate a PropagationModel.Kind string to the integer
       * model code used by coverageGrid / coverageGridMulti.
       * Called from the `coverage(tx, pm)` method body. */
      {"propKindToModelCode", "matlab_prop_kind_to_model_code", 0, "p"},
      /* antennaGain(ant, freq) — peak gain dBi.  Today returns the
       * textbook broadside value; full angle-dependent pattern
       * lookup lands with ANT-Tier-2 wire-MoM. */
      {"antennaGain", "matlab_prop_antenna_gain", 0, "pf"},
      /* === Communications Toolbox Tier-1 base layer ===
       * docs/comm_toolbox_roadmap.md §2. runtime/runtime_comm.cpp.
       */
      /* §2.2 rng — numeric-tag dispatch (string variants exposed
       * as rngDefault / rngShuffle named functions). */
      {"rng",          "matlab_comm_rng",          0, "f"},
      {"rngDefault",   "matlab_comm_rng_default", 0, ""},
      {"rngShuffle",   "matlab_comm_rng_shuffle", 0, ""},
      {"rngGet",       "matlab_comm_rng_get",     0, ""},
      {"rngSet",       "matlab_comm_rng_set",     0, "f"},
      /* §2.1 randi — 1/2/3 arg forms. */
      {"randi",        "matlab_comm_randi_s",     0, "f"},
      {"randi",        "matlab_comm_randi_nn",    1, "ff"},
      {"randi",        "matlab_comm_randi_mn",    1, "fff"},
      /* §2.3 randsrc / randerr. */
      {"randsrc",         "matlab_comm_randsrc",         1, "ffp"},
      {"randsrcWeighted", "matlab_comm_randsrc_weighted",1, "ffpp"},
      {"randerr",         "matlab_comm_randerr",         1, "fff"},
      /* §2.4 bit conversion. */
      {"int2bit", "matlab_comm_int2bit", 1, "pf"},
      {"bit2int", "matlab_comm_bit2int", 1, "pf"},
      {"de2bi",   "matlab_comm_de2bi",   1, "pf"},
      {"bi2de",   "matlab_comm_bi2de",   1, "p"},
      /* §2.5 awgn — polymorphic on real/complex via the magic-tag
       * sniff inside the runtime. The dispatch accepts a ptr arg
       * (matlab_mat OR matlab_mat_c). */
      {"awgn",   "matlab_comm_awgn",   1, "pf"},
      {"awgn",   "matlab_comm_awgn_p", 1, "pff"},
      /* §2.6 biterr / symerr — single-return forms return the BER
       * ratio (the second of MATLAB's [nerr, ratio] pair, since the
       * ratio is what almost every script consumes). The count-only
       * variants are named biterrCount / symerrCount for the rare
       * raw-integer use. */
      {"biterr",      "matlab_comm_biterr_ratio",    0, "pp"},
      {"biterrK",     "matlab_comm_biterr_ratio_k",  0, "ppf"},
      {"biterrCount", "matlab_comm_biterr_count",    0, "pp"},
      {"symerr",      "matlab_comm_symerr_ratio",    0, "pp"},
      {"symerrCount", "matlab_comm_symerr_count",    0, "pp"},
      /* === Communications Toolbox Tier-2 — digital modulation MVP ===
       * docs/comm_toolbox_roadmap.md §4. runtime/runtime_comm.cpp.
       * Numeric-tag dispatch: order = 0 binary / 1 Gray;
       * shape = 0 RRC ('sqrt') / 1 RC ('normal');
       * mod_code = 0 PAM, 1 PSK, 2 QAM, 3 DPSK, 4 FSK-coh, 5 FSK-nc. */
      {"qfunc", "matlab_comm_qfunc_s", 0, "f"},
      {"erfc",  "matlab_comm_erfc_s",  0, "f"},
      /* §4.1 PAM (real-line). */
      {"pammod",   "matlab_comm_pammod",   1, "pff"},
      {"pamdemod", "matlab_comm_pamdemod", 1, "pff"},
      /* §4.3 PSK (complex). 4 args: x, M, ini_phase, order. */
      {"pskmod",   "matlab_comm_pskmod",   1, "pfff"},
      {"pskdemod", "matlab_comm_pskdemod", 1, "pfff"},
      /* §4.5 FSK (complex, continuous-phase).
       *   fskmod(x, M, freqsep, nsamp, fs)
       *   fskdemod(y, M, freqsep, nsamp, fs, mode)
       * mode: 0 = coherent (signed correlation), 1 = noncoherent (|·|). */
      {"fskmod",   "matlab_comm_fskmod",   1, "pffff"},
      {"fskdemod", "matlab_comm_fskdemod", 1, "pfffff"},
      /* ANT-Tier-2 — straight thin-wire MoM (Pocklington / pulse-basis /
       * 2N×2N real-equivalent solve).  Half-wave dipole MVP. */
      {"antennaWireSolve",        "matlab_ant_wire_solve",        1, "ffff"},
      {"antennaWirePattern",      "matlab_ant_wire_pattern",      1, "fffff"},
      {"antennaWireSparameters",  "matlab_ant_wire_sparameters",  1, "fffp"},
      /* §4.2 QAM (complex). 4 args: x, M, order, unit_avg_power_flag. */
      {"qammod",     "matlab_comm_qammod",     1, "pfff"},
      {"qamdemod",   "matlab_comm_qamdemod",   1, "pfff"},
      {"qamdemodBit","matlab_comm_qamdemod_bit",1,"pfff"},
      {"qamdemodLlr","matlab_comm_qamdemod_llr",1,"pffff"},
      /* §4.6 generic constellation. Alphabet is a complex column. */
      {"genqammod",   "matlab_comm_genqammod",   1, "pp"},
      {"genqamdemod", "matlab_comm_genqamdemod", 1, "pp"},
      /* §4.7 pulse shaping. */
      {"rcosdesign", "matlab_comm_rcosdesign", 1, "ffff"},
      {"gaussdesign","matlab_comm_gaussdesign",1, "fff"},
      /* §4.8 berawgn closed-form curve. Args: EbN0_dB, M, mod_code. */
      {"berawgn",    "matlab_comm_berawgn_s",  0, "fff"},
      /* §4.9 scatterplot numeric form. */
      {"scatterplot","matlab_comm_scatterplot",1, "p"},
      /* eyediagram(x, n) — n × num_traces matrix where each
       * column is a consecutive n-sample slice of `x`.  Real and
       * complex inputs both supported. */
      {"eyediagram", "matlab_comm_eyediagram", 1, "pf"},
      /* === Communications Toolbox Tier-3 — channel coding ===
       * docs/comm_toolbox_roadmap.md §5. CRC function-form, the
       * `poly2trellis` / `convenc` / `vitdec` convolutional surface,
       * Hamming codes, and block interleavers. BCH / RS / gf and
       * LDPC / Turbo / Polar are deferred. */
      /* §5.1 CRC (function-form). */
      {"crcGenerate", "matlab_comm_crc_generate", 1, "pff"},
      {"crcCheck",    "matlab_comm_crc_check",    0, "pff"},
      {"crcStrip",    "matlab_comm_crc_strip",    1, "pf"},
      /* §5.2 convolutional codes. */
      {"poly2trellis","matlab_comm_poly2trellis", 1, "fp"},
      {"convenc",     "matlab_comm_convenc",      1, "pp"},
      {"vitdec",      "matlab_comm_vitdec",       1, "ppfff"},
      {"oct2dec",     "matlab_comm_oct2dec_s",    0, "f"},
      /* §5.3 Hamming. */
      {"hammgenParity",  "matlab_comm_hammgen_parity",  1, "f"},
      {"hammingEncode",  "matlab_comm_hamming_encode",  1, "pf"},
      {"hammingDecode",  "matlab_comm_hamming_decode",  1, "pf"},
      /* §5.5 block interleavers. */
      {"intrlv",   "matlab_comm_intrlv",   1, "pp"},
      {"deintrlv", "matlab_comm_deintrlv", 1, "pp"},
      /* === Communications Toolbox Tier-4 ===
       * Equalisation / sync / RF impairments
       * (docs/comm_toolbox_roadmap.md §6). */
      /* §6.1 adaptive equalisers (function-form). */
      {"lms", "matlab_comm_lms", 1, "ppff"},
      {"rls", "matlab_comm_rls", 1, "ppfff"},
      {"cma", "matlab_comm_cma", 1, "pfff"},
      {"dfe", "matlab_comm_dfe", 1, "ppfff"},
      /* §6.2 sync. */
      {"costasPll",      "matlab_comm_costas_pll",       1, "pfff"},
      {"symbolSyncMM",   "matlab_comm_symbol_sync_mm",   1, "pff"},
      {"preambleDetect", "matlab_comm_preamble_detect",  0, "pp"},
      /* §6.3 RF impairments. */
      {"phaseFreqOffset","matlab_comm_phase_freq_offset",1, "pff"},
      {"iqimbal",        "matlab_comm_iqimbal",          1, "pff"},
      {"memorylessNl",   "matlab_comm_memoryless_nl",    1, "pfffff"},
      {"phaseNoise",     "matlab_comm_phase_noise",      1, "pff"},
      /* Soft-decision Viterbi extension (Tier-3 follow-on parked
       * with the Tier-4 RF-impairment + soft-demod slice). */
      {"vitdecSoft",     "matlab_comm_vitdec_soft",      1, "ppff"},
      /* === Communications Toolbox Tier-5 — OFDM / fading / MIMO ===
       * docs/comm_toolbox_roadmap.md §7.  Function-form. */
      /* §7.1 OFDM. */
      {"ofdmmod",   "matlab_comm_ofdmmod",   1, "pff"},
      {"ofdmdemod", "matlab_comm_ofdmdemod", 1, "pff"},
      /* §7.2 fading channels. */
      {"rayleighChannel", "matlab_comm_rayleigh_channel", 1, "pppff"},
      {"ricianChannel",   "matlab_comm_rician_channel",   1, "pfppff"},
      /* §7.3 MIMO (Alamouti 2-Tx + ML detect). */
      {"ostbcEncode",  "matlab_comm_ostbc_encode",  1, "p"},
      {"ostbcCombine", "matlab_comm_ostbc_combine", 1, "pffff"},
      {"mlDetect",     "matlab_comm_ml_detect",     1, "pp"},
      /* === Communications Toolbox Tier-6 — spreading + source coding ===
       * docs/comm_toolbox_roadmap.md §8.  Function-form throughout. */
      /* §8.1 spreading sequences. */
      {"pnSequence",   "matlab_comm_pn_sequence",   1, "ffff"},
      {"goldSequence", "matlab_comm_gold_sequence", 1, "ffffff"},
      {"hadamard",     "matlab_comm_hadamard",      1, "f"},
      {"walshCode",    "matlab_comm_walsh_code",    1, "ff"},
      /* §8.2 source coding. */
      {"quantiz",      "matlab_comm_quantiz",       1, "ppp"},
      {"quantizApply", "matlab_comm_quantiz_apply", 1, "pp"},
      {"lloydsQuant",  "matlab_comm_lloyds_quant",  1, "ppff"},
      {"compandMu",    "matlab_comm_compand_mu",    1, "pfff"},
      {"compandA",     "matlab_comm_compand_a",     1, "pfff"},
      {"dpcmEncode",   "matlab_comm_dpcm_encode",   1, "ppp"},
      {"dpcmDecode",   "matlab_comm_dpcm_decode",   1, "pp"},
      /* === Communications Toolbox Tier-7 — modern channel codes ===
       * docs/comm_toolbox_roadmap.md §5.4.  Function-form. */
      {"polarEncode",   "matlab_comm_polar_encode",     1, "pf"},
      {"polarSCdecode", "matlab_comm_polar_sc_decode",  1, "ppf"},
      {"ldpcEncode",    "matlab_comm_ldpc_encode",      1, "pp"},
      {"ldpcDecodeMS",  "matlab_comm_ldpc_decode_ms",   1, "ppf"},
      {"turboEncode",   "matlab_comm_turbo_encode",     1, "ppp"},
      {"turboDecode",   "matlab_comm_turbo_decode",     1, "pppppf"},
      /* === RF Toolbox companion (RF-Tier-1 + RF-Tier-2) ===
       * docs/comm_toolbox_roadmap.md §9.  Function-form, 2-port
       * subset.  Touchstone v1 .s2p I/O + per-frequency closed-form
       * S-parameter analyses + Friis cascade for rfbudget. */
      /* §9.1.3 Touchstone I/O.  Reader returns a struct with S11/
       * S12/S21/S22 (complex columns), Frequencies (real column),
       * Z0 and NumPorts (scalars).  Writer emits .s2p in MA format. */
      {"touchstoneRead",      "matlab_rf_touchstone_read",       1, "p"},
      {"touchstoneWriteS2p",  "matlab_rf_touchstone_write_s2p",  0, "pppppppf"},
      /* Typed-getter helpers for the touchstoneRead return struct.
       * Needed because struct.S11 routes through matlab_struct_get_f64
       * by default; users call tsS11(data) / tsS12(data) / ... to get
       * the matrix-typed columns. */
      {"tsS11",     "matlab_rf_ts_s11",        1, "p"},
      {"tsS12",     "matlab_rf_ts_s12",        1, "p"},
      {"tsS21",     "matlab_rf_ts_s21",        1, "p"},
      {"tsS22",     "matlab_rf_ts_s22",        1, "p"},
      {"tsFreqs",   "matlab_rf_ts_freqs",      1, "p"},
      {"tsZ0",      "matlab_rf_ts_z0",         0, "p"},
      {"tsNumPorts","matlab_rf_ts_num_ports",  0, "p"},
      /* Generic multi-port S(i,j) getter — extracts S<i><j> field. */
      {"tsSij",     "matlab_rf_ts_sij",        1, "pff"},
      {"tsYij",     "matlab_rf_ts_yij",        1, "pff"},
      {"tsZij",     "matlab_rf_ts_zij",        1, "pff"},
      {"tsHij",     "matlab_rf_ts_hij",        1, "pff"},
      {"tsGij",     "matlab_rf_ts_gij",        1, "pff"},
      {"tsTij",     "matlab_rf_ts_tij",        1, "pff"},
      {"tsAbcdA",   "matlab_rf_ts_abcd_a",     1, "p"},
      {"tsAbcdB",   "matlab_rf_ts_abcd_b",     1, "p"},
      {"tsAbcdC",   "matlab_rf_ts_abcd_c",     1, "p"},
      {"tsAbcdD",   "matlab_rf_ts_abcd_d",     1, "p"},
      /* N-port S↔Y / S↔Z conversions.  Take the touchstoneRead struct
       * directly, return a parallel struct with Y_ij / Z_ij fields. */
      {"sparamS2yN",          "matlab_rf_s2y_n",                 1, "p"},
      {"sparamS2zN",          "matlab_rf_s2z_n",                 1, "p"},
      /* snp2smp port-extraction.  Args: data struct, port-list column,
       * target m-port count.  v1 assumes matched terminations at the
       * dropped ports. */
      {"snp2smp",             "matlab_rf_snp2smp",               1, "ppf"},
      /* Non-matched termination variant: per-dropped-port termination
       * impedance via the Schur-complement update. */
      {"snp2smpZ",            "matlab_rf_snp2smp_z",             1, "pppf"},
      /* Multi-port Touchstone writer.  Takes a data struct (any N) and
       * writes a Touchstone v1 .sNp file in MA format.  Auto-detects
       * port count from the struct's NumPorts field. */
      {"touchstoneWrite",     "matlab_rf_touchstone_write",      0, "pp"},
      /* N-port cascade (diagonal approximation for weakly-coupled
       * networks). */
      {"cascadeSparamsN",     "matlab_rf_cascade_n",             1, "pp"},
      /* Full Redheffer star-product N-port cascade (k = N/2 case).
       * Exact for arbitrarily-coupled networks of even port count. */
      {"cascadeSparamsNFull", "matlab_rf_cascade_n_full",        1, "pp"},
      /* 2-port cross-conversions: S↔H, S↔ABCD.  Per-frequency
       * closed-form expressions. */
      {"sparamS2h",           "matlab_rf_s2h",                   1, "ppppf"},
      {"sparamS2abcd",        "matlab_rf_s2abcd",                1, "ppppf"},
      /* Inverse cross-conversions: H→S, ABCD→S. */
      {"sparamH2s",           "matlab_rf_h2s",                   1, "ppppf"},
      {"sparamAbcd2s",        "matlab_rf_abcd2s",                1, "ppppf"},
      /* Additional cross-conversions: S↔G, S↔T, plus the Smith-chart
       * Γ ↔ Z helpers. */
      {"sparamS2g",           "matlab_rf_s2g",                   1, "ppppf"},
      {"sparamG2s",           "matlab_rf_g2s",                   1, "ppppf"},
      {"sparamS2t",           "matlab_rf_s2t",                   1, "pppp"},
      {"sparamT2s",           "matlab_rf_t2s",                   1, "pppp"},
      {"gamma2z",             "matlab_rf_gamma2z",               1, "pf"},
      {"z2gamma",             "matlab_rf_z2gamma",               1, "pf"},
      /* T / Pi matchingnetwork topologies.  q_target argument
       * controls the high-Q virtual-impedance level. */
      {"matchingnetworkT",    "matlab_rf_matchingnetwork_t",     1, "ffffff"},
      {"matchingnetworkPi",   "matlab_rf_matchingnetwork_pi",    1, "ffffff"},
      /* §9.1.2 S↔Y / S↔Z conversions (2-port). */
      {"sparamS2y",           "matlab_rf_s2y",                   1, "ppppf"},
      {"sparamS2z",           "matlab_rf_s2z",                   1, "ppppf"},
      /* §9.2.1 closed-form analyses (2-port). */
      {"gammaIn",             "matlab_rf_gamma_in",              1, "ppppff"},
      {"gammaOut",            "matlab_rf_gamma_out",             1, "ppppff"},
      {"vswr",                "matlab_rf_vswr_from_gamma",       1, "p"},
      /* type_code: 0=Gt, 1=Ga, 2=Gp.  Returns linear gain
       * (caller applies 10*log10 for dB). */
      {"powerGain",           "matlab_rf_power_gain",            1, "ppppffff"},
      {"stabilityK",          "matlab_rf_stability_k",           1, "pppp"},
      /* type: 0=mu1 (source-side), 1=mu2 (load-side). */
      {"stabilityMu",         "matlab_rf_stability_mu",          1, "ppppf"},
      {"s2tf",                "matlab_rf_s2tf",                  1, "ppppfff"},
      /* §9.2.2 cascade — 2-port T-parameter chain.  Args:
       * a.S11/12/21/22, b.S11/12/21/22.  Returns struct with
       * S11/S12/S21/S22 fields. */
      {"cascadeSparams2",     "matlab_rf_cascade2",              1, "pppppppp"},
      /* §9.2.3 rfbudget Friis cascade.  Args: gains_dB (col),
       * nfs_dB (col), ip3_dBm (col), p_in_dBm, bw_Hz.  Returns
       * struct (Gain_dB / NF_dB / IP3_in_dBm / OutputPower_dBm /
       * NoiseFloor_dBm / SNR_dB / ...). */
      {"rfbudgetFriis",       "matlab_rf_budget_friis",          1, "pppff"},
      /* §9.3.1 Vector Fitting (Gustavsen-Semlyen 1999).  Real-pole
       * MVP — fits measured H(jω) with Σ rⱼ/(s − pⱼ) + d.  Returns
       * a struct with Poles / Residues (real columns), D, Order,
       * FitError.  Args: freq (col), h_re (col), h_im (col),
       * nPoles, nIter. */
      {"rationalfit",         "matlab_rf_rationalfit",           1, "pppff"},
      /* §9.3.1 freqresp — evaluate the rational at frequencies.
       * Args: mdl_struct, freq (col).  Returns complex column. */
      {"freqresp",            "matlab_rf_freqresp",              1, "pp"},
      /* Typed-getter helpers for the rationalfit return struct.
       * Mirrors the touchstoneRead `tsS11` / `tsZ0` pattern. */
      {"rfPoles",             "matlab_rf_rf_poles",              1, "p"},
      {"rfResidues",          "matlab_rf_rf_residues",           1, "p"},
      {"rfD",                 "matlab_rf_rf_d",                  0, "p"},
      {"rfOrder",             "matlab_rf_rf_order",              0, "p"},
      {"rfFitError",          "matlab_rf_rf_fit_error",          0, "p"},
      /* §9.3.2 — time-domain RF.  timeresp(mdl, u, ts) drives the
       * fitted rational with the input signal `u`; s2tdr / s2tdt
       * fit then drive a unit step (TDR = reflection step, TDT =
       * transmission step). */
      {"timeresp",            "matlab_rf_timeresp",              1, "ppf"},
      {"s2tdr",               "matlab_rf_s2tdr",                 1, "ppfff"},
      {"s2tdt",               "matlab_rf_s2tdt",                 1, "ppfff"},
      /* §9.1.2 mixed-mode 4-port.  block_code: 0=dd, 1=dc, 2=cd,
       * 3=cc.  Args: 16 single-ended ptr columns + block code. */
      {"sparamS2smm",         "matlab_rf_s2smm",                 1,
                                                                  "ppppppppppppppppf"},
      /* §9.4.3 Smith chart numeric grid.  r_norm = constant-r ring
       * to draw (1.0 = matched), n_pts = points around the circle. */
      {"smithGrid",           "matlab_rf_smith_grid",            1, "ff"},
      {"smithRCircle",        "matlab_rf_smith_rcircle",         1, "p"},
      {"smithUnitCircle",     "matlab_rf_smith_unit",            1, "p"},
      /* §9.3.1 follow-on — passivity test.  Returns max |H(jω)|
       * over a dense log-spaced frequency sweep. */
      {"passivity",           "matlab_rf_passivity",             0, "pff"},
      /* §9.4.1 matchingnetwork — L-section auto-synthesis.  Args:
       * Re(Zs), Im(Zs), Re(Zl), Im(Zl), freq.  Returns a struct with
       * Topology / Q / L / C component values + return loss. */
      {"matchingnetwork",     "matlab_rf_matchingnetwork",       1, "fffff"},
      /* §9.3.3 transmission-line geometries.  Each takes geometry
       * parameters + a freq column + reference z0.  Returns the
       * 2-port S-parameter struct (S11/S12/S21/S22 complex columns). */
      {"rfckt_txline",        "matlab_rf_txline",                1, "fffpf"},
      {"rfckt_coaxial",       "matlab_rf_coaxial",               1, "ffffpf"},
      {"rfckt_microstrip",    "matlab_rf_microstrip",            1, "ffffpf"},
      {"rfckt_cpw",           "matlab_rf_cpw",                   1, "ffffpf"},
      {"rfckt_parallelplate", "matlab_rf_parallelplate",         1, "ffffpf"},
      {"rfckt_twowire",       "matlab_rf_twowire",               1, "ffffpf"},
      /* LC filter blocks.  topology: 0 = lowpass-tee, 1 = lowpass-pi,
       * 2 = highpass-tee, 3 = highpass-pi.  Args: topology, comp1,
       * comp2, freqs, z0. */
      {"rfckt_lcfilter",      "matlab_rf_lc_filter",             1, "fffpf"},
      /* 4-element LC bandpass / bandstop (codes 4..7).  Args:
       * topology, L1, C1, L2, C2, freqs, z0. */
      {"rfckt_lcfilter4",     "matlab_rf_lc_filter4",            1, "fffffpf"},
      /* RFCkt block analyze helpers — synthesize S-parameter structs
       * from scalar block properties (used by classdef analyze methods). */
      {"rfAnalyzeAmplifier",  "matlab_rf_analyze_amplifier",     1, "fpf"},
      {"rfAnalyzePassive",    "matlab_rf_analyze_passive",       1, "fpf"},
      {"rfAnalyzeSeries",     "matlab_rf_analyze_series",        1, "ffpf"},
      {"rfAnalyzeShunt",      "matlab_rf_analyze_shunt",         1, "ffpf"},
      /* §9.2.1 follow-on — conjugate-match Γ + group delay +
       * arbitrary-port s2tf + per-stage rfbudget + stability circles. */
      {"gammams",             "matlab_rf_gammams",               1, "pppp"},
      {"gammaml",             "matlab_rf_gammaml",               1, "pppp"},
      {"groupdelay",          "matlab_rf_groupdelay",            1, "pp"},
      {"s2tfPort",            "matlab_rf_s2tf_port",             1, "ppppfffff"},
      {"rfbudgetTable",       "matlab_rf_budget_table",          1, "pppff"},
      {"stabCircleLoad",      "matlab_rf_stab_circle_load",      1, "pppp"},
      {"stabCircleSource",    "matlab_rf_stab_circle_source",    1, "pppp"},
      /* MathWorks-faithful lowercase aliases.  Same runtime targets,
       * just additional registered names so tutorial-style code from
       * MathWorks docs runs verbatim. */
      {"s2y",     "matlab_rf_s2y",              1, "ppppf"},
      {"s2z",     "matlab_rf_s2z",              1, "ppppf"},
      {"s2h",     "matlab_rf_s2h",              1, "ppppf"},
      {"s2g",     "matlab_rf_s2g",              1, "ppppf"},
      {"s2abcd",  "matlab_rf_s2abcd",           1, "ppppf"},
      {"s2t",     "matlab_rf_s2t",              1, "pppp"},
      {"h2s",     "matlab_rf_h2s",              1, "ppppf"},
      {"g2s",     "matlab_rf_g2s",              1, "ppppf"},
      {"abcd2s",  "matlab_rf_abcd2s",           1, "ppppf"},
      {"t2s",     "matlab_rf_t2s",              1, "pppp"},
      {"rfbudget",    "matlab_rf_budget_friis",     1, "pppff"},
      {"rfwrite",     "matlab_rf_touchstone_write", 0, "pp"},
      {"sparameters", "matlab_rf_touchstone_read",  1, "p"},
      /* §9.3.1 follow-on — rationalfit pre-fit delay extraction +
       * post-fit passivity enforcement. */
      {"rfDelayEstimate",    "matlab_rf_delay_estimate",     0, "ppp"},
      {"rfApplyDelay",       "matlab_rf_apply_delay",        1, "pppf"},
      {"rfPassivityEnforce", "matlab_rf_enforce_passivity",  1, "pff"},
      /* Weighted Vector Fitting — per-frequency weight column scales
       * the LS rows so higher-weight frequencies dominate the fit. */
      {"rationalfitWeighted", "matlab_rf_rationalfit_w",     1, "ppppff"},
      /* §9.1.2 follow-on — newref re-reference to a new scalar z0. */
      {"newref",              "matlab_rf_newref",            1, "pf"},
      /* Redheffer star with arbitrary inner-connection port count. */
      {"cascadeSparamsNFullK", "matlab_rf_cascade_n_fullk",  1, "ppf"},
      /* §9.1.2 N-port S→ABCD and S→H (block-partitioned, even N). */
      {"sparamS2abcdN",        "matlab_rf_s2abcd_n",         1, "p"},
      {"sparamS2hN",           "matlab_rf_s2h_n",            1, "p"},
      /* §9.5 Verilog-A export — rfmodel.rational / RFRational.  Args:
       * rationalfit-struct-or-RFRational, filename string.  Writes a
       * parameterized .va module to disk. */
      {"writeVerilogA",        "matlab_rf_write_verilog_a",  0, "pp"},
      /* §9.5 Tier-2 — continuous rational filter export.
       *   writeVerilogATF(num, den, filename) — coefficients in
       *     descending power of s (MATLAB tf convention).
       *   writeVerilogAZPK(zeros, poles, k, filename) — zeros/poles
       *     as real or complex columns; complex-conjugate pairs fold
       *     into real-coefficient quadratic factors. */
      {"writeVerilogATF",      "matlab_rf_write_verilog_a_tf",     0, "ppp"},
      /* Scalar-fold shims for cases where `num = [1.0]` collapses to
       * an f64 instead of a 1×1 matrix at MIR. */
      {"writeVerilogATF",      "matlab_rf_write_verilog_a_tf_sm",  0, "fpp"},
      {"writeVerilogATF",      "matlab_rf_write_verilog_a_tf_ms",  0, "pfp"},
      {"writeVerilogATF",      "matlab_rf_write_verilog_a_tf_ss",  0, "ffp"},
      {"writeVerilogAZPK",     "matlab_rf_write_verilog_a_zpk",    0, "ppfp"},
      /* §9.5 Tier-3 — continuous SISO state-space export.
       *   writeVerilogASS(A, B, C, D, filename)
       * Emits one ddt(x[i]) contribution per state variable + the
       * output equation. */
      {"writeVerilogASS",      "matlab_rf_write_verilog_a_ss",     0, "pppfp"},
      /* Scalar-fold shim for 1-state systems where A / B / C collapse
       * to f64 at MIR (e.g. `A = [-1e6]` folds to scalar). */
      {"writeVerilogASS",      "matlab_rf_write_verilog_a_ss_fffd", 0, "ffffp"},
      /* §9.5 Tier-4 — analog source / comparator / Schmitt-trigger
       * Verilog-A export.  All take scalar parameters; no matrix args. */
      {"writeVerilogASource",     "matlab_rf_write_verilog_a_source",     0, "fffp"},
      {"writeVerilogAComparator", "matlab_rf_write_verilog_a_comparator", 0, "fffffp"},
      {"writeVerilogASchmitt",    "matlab_rf_write_verilog_a_schmitt",    0, "ffffp"},
      /* §9.5 Tier-5 — VCO via idtmod phase accumulation. */
      {"writeVerilogAVCO",        "matlab_rf_write_verilog_a_vco",        0, "fffp"},
      /* §9.5 Tier-6 — behavioral DAC (pure Verilog-A, analog-coded input). */
      {"writeVerilogADAC",        "matlab_rf_write_verilog_a_dac",        0, "ffffp"},
      /* §9.5 Tier-7 — compact analog components + sensor models. */
      {"writeVerilogADiode",      "matlab_rf_write_verilog_a_diode",      0, "ffp"},
      {"writeVerilogAOpAmp",      "matlab_rf_write_verilog_a_opamp",      0, "ffp"},
      {"writeVerilogARTD",        "matlab_rf_write_verilog_a_rtd",        0, "fffp"},
      {"writeVerilogAThermistor", "matlab_rf_write_verilog_a_thermistor", 0, "fffp"},
      /* §9.5 Tier-8 — white + flicker noise sources. */
      {"writeVerilogANoise",      "matlab_rf_write_verilog_a_noise",      0, "fffp"},
      /* §9.5 Tier-9 — lookup tables via $table_model.  Writes a .tbl
       * sidecar alongside the .va. */
      {"writeVerilogATable",      "matlab_rf_write_verilog_a_table",      0, "ppp"},
      /* §9.5 Tier-7 follow-on — composite RF / signal-chain blocks. */
      {"writeVerilogAAmplifier",  "matlab_rf_write_verilog_a_amplifier",  0, "fffp"},
      {"writeVerilogAAM",         "matlab_rf_write_verilog_a_am",         0, "ffp"},
      {"writeVerilogAIQMod",      "matlab_rf_write_verilog_a_iqmod",      0, "ffp"},
      /* ====================================================================
       * Financial Toolbox Tier-1: date arithmetic
       * ==================================================================*/
      {"yearfrac",   "matlab_yearfrac",   0, "ppf"},
      {"daysdif",    "matlab_daysdif",    0, "ppf"},
      {"daysadd",    "matlab_daysadd",    1, "pff"},
      {"daysact",    "matlab_daysact",    0, "pp"},
      {"days360",    "matlab_days360",    0, "pp"},
      {"days365",    "matlab_days365",    0, "pp"},
      {"busdate",    "matlab_busdate",    1, "pf"},
      {"isbusday",   "matlab_isbusday",   0, "p"},
      {"eomdate",    "matlab_eomdate",    1, "ff"},
      {"lweekdate",  "matlab_lweekdate",  1, "fff"},
      {"fweekdate",  "matlab_fweekdate",  1, "fff"},
      {"m2xdate",    "matlab_m2xdate",    0, "f"},
      {"x2mdate",    "matlab_x2mdate",    0, "f"},
      /* ====================================================================
       * Financial Toolbox Tier-1: cash flows + depreciation
       * ==================================================================*/
      {"pvfix",      "matlab_pvfix",      0, "fff"},
      {"fvfix",      "matlab_fvfix",      0, "fff"},
      {"pvvar",      "matlab_pvvar",      0, "pf"},
      {"fvvar",      "matlab_fvvar",      0, "pf"},
      {"irr",        "matlab_irr",        0, "p"},
      {"payper",     "matlab_payper",     0, "fff"},
      {"amortize",   "matlab_amortize",   1, "fff"},
      {"nomrr",      "matlab_nomrr",      0, "ff"},
      {"effrr",      "matlab_effrr",      0, "ff"},
      {"depstln",    "matlab_depstln",    1, "fff"},
      {"depsoyd",    "matlab_depsoyd",    1, "fff"},
      {"depfixdb",   "matlab_depfixdb",   1, "fff"},
      /* Financial Toolbox Tier-1: bond pricing + T-bills. */
      {"bndprice",   "matlab_bndprice",   0, "ffff"},
      {"bndyield",   "matlab_bndyield",   0, "ffff"},
      {"bnddurp",    "matlab_bnddurp",    1, "ffff"},
      {"bnddury",    "matlab_bnddury",    1, "ffff"},
      {"bndconvp",   "matlab_bndconvp",   0, "ffff"},
      {"accrfrac",   "matlab_accrfrac",   0, "ff"},
      {"prdisc",     "matlab_prdisc",     0, "ff"},
      {"prtbill",    "matlab_prtbill",    0, "ff"},
      {"ytbill",     "matlab_ytbill",     0, "ff"},
      {"beytbill",   "matlab_beytbill",   0, "ff"},
      /* Financial Toolbox Tier-1: returns + indicator function-form. */
      {"tick2ret",   "matlab_tick2ret",   1, "p"},
      {"ret2tick",   "matlab_ret2tick",   1, "p"},
      {"sma",        "matlab_sma",        1, "pf"},
      {"bolling",    "matlab_bolling",    1, "pff"},
      {"rsindex",    "matlab_rsindex",    1, "pf"},
      /* Financial Toolbox Tier-2: performance metrics. */
      {"sharpe",     "matlab_sharpe",     0, "pf"},
      {"sortino",    "matlab_sortino",    0, "pf"},
      {"inforatio",  "matlab_inforatio",  0, "pp"},
      {"tracking",   "matlab_tracking",   0, "pp"},
      {"maxdrawdown","matlab_maxdrawdown",0, "p"},
      {"lpm",        "matlab_lpm",        0, "pff"},
      {"portalpha",  "matlab_portalpha",  0, "ppf"},
      /* Financial Toolbox Tier-2: Black-Scholes Greeks + implied vol. */
      {"blsprice",   "matlab_blsprice",   0, "fffff"},
      {"blsdelta",   "matlab_blsdelta",   0, "fffff"},
      {"blsgamma",   "matlab_blsgamma",   0, "fffff"},
      {"blsvega",    "matlab_blsvega",    0, "fffff"},
      {"blsrho",     "matlab_blsrho",     0, "fffff"},
      {"blstheta",   "matlab_blstheta",   0, "fffff"},
      {"blslambda",  "matlab_blslambda",  0, "fffff"},
      {"blsimpv",    "matlab_blsimpv",    0, "fffff"},
      /* Financial Toolbox Tier-3: Portfolio classdef methods. The
       * first ptr arg is the Portfolio object (a class-pinned
       * matlab_obj*). Setters return the same obj for chaining. */
      {"setAssetMoments",         "matlab_portfolio_set_asset_moments",        1, "ppp"},
      {"setBounds",               "matlab_portfolio_set_bounds",               1, "ppp"},
      {"setBudget",               "matlab_portfolio_set_budget",               1, "pff"},
      {"setDefaultConstraints",   "matlab_portfolio_set_default_constraints",  1, "p"},
      {"estimateFrontier",        "matlab_portfolio_estimate_frontier",        1, "pf"},
      {"estimateFrontierByReturn","matlab_portfolio_estimate_frontier_by_return",1, "pf"},
      {"estimateMaxSharpeRatio",  "matlab_portfolio_estimate_max_sharpe",      1, "p"},
      {"estimatePortMoments",     "matlab_portfolio_estimate_port_moments",    1, "pp"},
      {"estimatePortReturn",      "matlab_portfolio_estimate_port_return",     0, "pp"},
      {"estimatePortRisk",        "matlab_portfolio_estimate_port_risk",       0, "pp"},
      {"estimateAssetMoments",    "matlab_portfolio_estimate_asset_moments",   1, "pp"},
      {"estimateBounds",          "matlab_portfolio_estimate_bounds",          1, "p"},
      {"estimateFrontierByRisk",  "matlab_portfolio_estimate_frontier_by_risk",1, "pf"},
      {"estimatePortFrontier",    "matlab_portfolio_estimate_frontier_points", 1, "pf"},
      {"plotFrontier",            "matlab_portfolio_plot_frontier",            1, "pf"},
      {"blacklitterman",          "matlab_blacklitterman",                     1, "ppppff"},
      {"blacklitterman",          "matlab_blacklitterman_q1",                  1, "pppfff"},
      {"riskparity",              "matlab_riskparity",                         1, "p"},
      {"riskbudget",              "matlab_riskbudget",                         1, "pp"},
      {"riskcontribution",        "matlab_riskcontribution",                   1, "pp"},
      /* Financial Toolbox Tier-4: regression with missing data. */
      {"ecmnmle",    "matlab_ecmnmle",    1, "p"},
      {"ecmncov",    "matlab_ecmncov",    1, "p"},
      {"mvnrmle",    "matlab_mvnrmle",    1, "pp"},
      {"capm",       "matlab_capm",       1, "ppf"},
      {"transprob",  "matlab_transprob",  1, "p"},
      {"cdsbootstrap","matlab_cdsbootstrap",1, "pppf"},
      {"cdsspread",  "matlab_cdsspread",  0, "ff"},
      {"cdsprice",   "matlab_cdsprice",   0, "fff"},
      {"fitmodel",   "matlab_creditscorecard_fitmodel",   1, "p"},
      {"probdefault","matlab_creditscorecard_probdefault",1, "pp"},
      {"score",      "matlab_creditscorecard_score",      1, "pp"},
      /* Financial Toolbox Tier-5: PortfolioCVaR / PortfolioMAD. The
       * shared estimateFrontier / estimatePortRisk / setDefaultConstraints
       * names route through the matlab_portfolio_* dispatchers (RiskKind
       * discriminant); these are the class-specific setters + readers. */
      {"setScenarios",        "matlab_portfoliocvar_set_scenarios",   1, "pp"},
      {"setProbabilityLevel", "matlab_portfoliocvar_set_prob_level",  1, "pf"},
      {"estimatePortVaR",     "matlab_portfoliocvar_estimate_port_var",0, "pp"},
      {"backtest",            "matlab_backtest",          1, "ppf"},
      {"backtestSummary",     "matlab_backtest_summary",  1, "p"},
      /* Financial Toolbox Tier-6: SDE Monte Carlo. */
      {"simByEuler",    "matlab_sde_sim_euler",    1, "pfff"},
      {"simBySolution", "matlab_sde_sim_solution", 1, "pfff"},
      {"haltonseq",     "matlab_haltonseq",        1, "ff"},
      {"optpricemc",    "matlab_optpricemc",       0, "pfff"},
      /* ================== Econometrics Toolbox Tier-1 ================== */
      /* Data prep. */
      {"price2ret",  "matlab_econ_price2ret",  1, "p"},
      {"ret2price",  "matlab_econ_ret2price",  1, "p"},
      {"hpfilter",   "matlab_econ_hpfilter",   1, "p"},
      {"hpfilter",   "matlab_econ_hpfilter_l", 1, "pf"},
      /* ACF / PACF. */
      {"autocorr",   "matlab_econ_autocorr",   1, "p"},
      {"autocorr",   "matlab_econ_autocorr_n", 1, "pf"},
      {"parcorr",    "matlab_econ_parcorr",    1, "p"},
      {"parcorr",    "matlab_econ_parcorr_n",  1, "pf"},
      {"crosscorr",  "matlab_econ_crosscorr",  1, "pp"},
      /* Diagnostic + comparison tests (return reject decision h, 0/1, at
       * the 5% level; the p-value/stat are available via the 2-output
       * forms wired in Lowering). */
      {"lbqtest",    "matlab_econ_lbqtest",    0, "p"},
      {"lbqtest",    "matlab_econ_lbqtest_n",  0, "pf"},
      {"archtest",   "matlab_econ_archtest",   0, "p"},
      {"archtest",   "matlab_econ_archtest_n", 0, "pf"},
      {"aicbic",     "matlab_econ_aic",        0, "ff"},
      {"aicbic",     "matlab_econ_aic_n",      0, "fff"},
      {"lratiotest", "matlab_econ_lratiotest", 0, "fff"},
      {"waldtest",   "matlab_econ_waldtest",   0, "pp"},
      {"lmtest",     "matlab_econ_lmtest",     0, "pp"},
      {"hac",        "matlab_econ_hac",        1, "pp"},
      {"fgls",       "matlab_econ_fgls",       1, "pp"},
      /* Unit-root + stationarity tests. */
      {"adftest",    "matlab_econ_adftest",    0, "p"},
      {"adftest",    "matlab_econ_adftest_n",  0, "pf"},
      {"pptest",     "matlab_econ_pptest",     0, "p"},
      {"kpsstest",   "matlab_econ_kpsstest",   0, "p"},
      {"lmctest",    "matlab_econ_lmctest",    0, "p"},
      {"vratiotest", "matlab_econ_vratiotest", 0, "p"},
      /* Tier-4 cointegration tests (function-form). */
      {"egcitest",   "matlab_econ_egcitest",   0, "p"},
      {"jcitest",    "matlab_econ_jcitest",    0, "p"},
      {"jcontest",   "matlab_econ_jcontest",   0, "p"},
      /* ===== Sensor Fusion Tier-1 — quaternion conversions ============
       * Pure matrix-in/matrix-out builtins (ret_code=1 = PtrTy/matrix). */
      {"quat2eul",   "matlab_fusion_quat_to_eul",   1, "p"},
      {"quat2rotm",  "matlab_fusion_quat2rotm",     1, "p"},
      {"eul2quat",   "matlab_fusion_eul_to_quat",   1, "p"},
      {"rotm2quat",  "matlab_fusion_rotm_to_quat",  1, "p"},
      /* slerp(M1, M2, t): two N×4 matrices + a scalar interpolation param. */
      {"slerp",      "matlab_fusion_quat_slerp",    1, "ppf"},
      /* ecompass(acc, mag): two 3-vectors → 1×4 quaternion data. */
      {"ecompass",   "matlab_fusion_ecompass",      1, "pp"},
      /* T1.7 core gaps surfaced here as plain generic builtins. */
      {"cross",      "matlab_cross",                1, "pp"},
      {"dot",        "matlab_dot",                  1, "pp"},
      {"deg2rad",    "matlab_deg2rad",              1, "p"},
      {"rad2deg",    "matlab_rad2deg",              1, "p"},
      {"mvnrnd",     "matlab_mvnrnd",               1, "ppf"},
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
    /* f32 is accepted at every 'f' slot — MATLAB `single()` / `double()`
     * casts lower to f32 results, and the runtime stays f64 internally,
     * so we widen with an fpext at the call site below.  Without this,
     * `gpuArray.linspace(single(-10), single(10), n)` and every other
     * builtin reached via a single() cast in its arg list fails dispatch.
     * (See issue #22.) */
    auto isFTagOK = [&](Type Got) -> bool {
      return Got == F64 || mlir::isa<Float32Type>(Got);
    };
    auto argTypesMatch = [&](const Spec &E) -> bool {
      if (E.ArgKinds.size() != NOps) return false;
      for (unsigned i = 0; i < NOps; ++i) {
        char Kind = E.ArgKinds[i];
        Type Got = Call->getOperand(i).getType();
        if (Kind == 'f') {
          if (!isFTagOK(Got)) return false;
        } else { /* 'p' */
          /* Accept Ptr, tensor<*xf64>, and `none` (untyped ptr — common
           * for dlgradient-returned gradients flowing into optimizers). */
          if (Got != PtrTy && !isTensorLike(Got) &&
              !mlir::isa<mlir::NoneType>(Got))
            return false;
        }
      }
      return true;
    };
    for (auto &E : Table)
      if (E.MLName == Name && argTypesMatch(E)) { S = &E; break; }

    /* Scalar-promotion fallback. If no strict match was found, try to
     * pick a Spec where the only mismatches are 'p' slots receiving
     * f64 values — those get auto-boxed via matlab_mat_from_scalar at
     * the call site below. Limited to an explicit allowlist so calls
     * like mean(5.0) still fall through to the scalar `Scalar` map
     * below instead of getting wrapped into a 1x1 matrix. The list
     * covers the Tier 1/2/3 builtins where scalar args are idiomatic
     * MATLAB (conv(u, gain), filter(b, 1, x), polyval(p, scalar), ...).
     * See docs/runtime.md "Scalar-arg overloads". */
    SmallVector<unsigned, 3> BoxIdx;
    SmallVector<unsigned, 3> UnboxIdx;
    if (!S) {
      static const llvm::StringSet<> AutoBoxNames = {
        "conv", "conv2", "filter", "xcorr",
        "polyval", "polyfit", "interp1", "interp2",
        "trapz", "cumtrapz", "imfilter", "padarray",
        /* §2.1 follow-on — analog↔digital + form conversions. */
        "bilinear", "freqs", "tf2zp", "zp2tf", "tf2sos", "sos2tf",
        /* CST Tier 1.4 — Lyapunov / Stein. Scalar `lyap([-1], [1])`
         * is a perfectly valid 1*1 invocation that we want to handle. */
        "lyap", "dlyap", "lyapchol",
        /* CST Tier 1.5 — algebraic Riccati. Same scalar-invocation rule. */
        "care", "dare", "icare", "idare",
        /* CST Tier 2 — LQR convenience wrappers; same scalar shape. */
        "lqr", "dlqr", "lqry_ss",
        /* CST Tier 3 — controllability/observability/place + characterization. */
        "ctrb", "obsv", "place", "damp", "hsvd",
        /* CST Tier 4 — balancing for model reduction. */
        "balreal_T", "balred", "balred_A", "balred_B", "balred_C",
        /* CST Tier 3 — H₂ system norm + DC gain + stepinfo. */
        "norm_h2", "dcgain_ss", "stepinfo",
        /* CST Tier 4.2 — Kalman / Kalmd steady-state gains. */
        "kalman", "kalmd", "kalman_L", "kalmd_L",
        /* MPC Tier-0 — sys-form Kalman dispatcher.  Scalar Qn / Rn
         * (e.g. SISO `kalman(sys, [1], [1])`) need the same auto-box
         * treatment as the matrix-form `kalman_L`. */
        "matlab_kalman_sys_L",
        /* MPC Tier-1 — `mpcmove(obj, st, ym, r)` and `sim(obj, T, r)`
         * may pass scalar `ym` / `r` / 1×1 matrix-literal references
         * (`r = [1]` is a SISO setpoint).  Same auto-box treatment so
         * those reach the runtime as proper matlab_mat *. */
        "matlab_mpc_move", "matlab_mpc_sim",
        /* MPC Tier-2 §3.7 — mpcmove with override. */
        "matlab_mpc_move_opt",
        /* MPC Tier-3 §4.1/§4.2 — adaptive + time-varying mpcmove. */
        "matlab_mpc_move_adaptive", "matlab_mpc_move_tv",
        /* MPC Tier-4 §5.4 — standalone active-set QP. */
        "matlab_mpc_active_set", "mpcActiveSetSolver",
        /* MPC Tier-4 §5.1/5.2/5.3 — explicit MPC. */
        "matlab_mpc_generate_explicit", "matlab_mpc_move_explicit",
        "matlab_mpc_simplify_explicit",
        /* MPC Tier-4 §5.7 — Finite Control Set MPC. */
        "matlab_mpc_move_finite",
        /* MPC Tier-5 — Nonlinear MPC. */
        "matlab_nlmpc_move",
        /* MPC Tier-6 §7.5/7.6 — review + sim-opt. */
        "matlab_mpc_review", "matlab_mpc_sim_opt",
        /* CST Tier 3 — discrete-time stability + H2. */
        "isstable_d", "norm_h2_d",
        /* CST Tier 2.2 — Tustin discretisation + inverse. */
        "c2d_tustin", "d2c_tustin",
        /* CST Tier 3.4 / 2.3 — gramians and SS step response. */
        "gram_c", "gram_o", "step_ss",
        /* CST Tier 2.4 — SISO bode_ss. */
        "bode_ss",
        /* CST Tier 2.3 follow-on + 2.4 — lsim, gain/phase margins,
         * TF-form bode. */
        "lsim_ss", "gain_margin", "phase_margin",
        "bandwidth_ss", "getPeakGain_ss",
        "feedback_ss", "series_ss", "parallel_ss", "append_ss", "bode_tf",
        /* `inv` on a 1*1 matrix is `1/x` — auto-box so `inv(R2)` works
         * uniformly for both 1*1 R (often the case in SISO LQR) and
         * larger matrices. */
        "inv",
        /* CST Tier 1.3 follow-on — matrix logarithm. Scalar `logm(4)` is
         * a valid 1*1 invocation that boxes to `[4]` for the runtime path. */
        "logm",
      };
      if (AutoBoxNames.contains(Name)) {
        for (auto &E : Table) {
          if (E.MLName != Name) continue;
          if (E.ArgKinds.size() != NOps) continue;
          bool can_box = true;
          SmallVector<unsigned, 3> idx;
          for (unsigned i = 0; i < NOps; ++i) {
            char K = E.ArgKinds[i];
            Type Got = Call->getOperand(i).getType();
            if (K == 'f') {
              if (!isFTagOK(Got)) { can_box = false; break; }
            } else { /* 'p' */
              if (Got == PtrTy || isTensorLike(Got)) {
                /* already matches strictly */
              } else if (Got == F64) {
                idx.push_back(i);
              } else {
                can_box = false; break;
              }
            }
          }
          if (can_box && !idx.empty()) {
            S = &E;
            BoxIdx = std::move(idx);
            break;
          }
        }
      }
    }

    /* Scalar-UNBOX fallback — the mirror of the autobox above. A workspace
     * scalar computed with element-wise ops (e.g. `snr_dB = EbNo + 10.*..`)
     * is typed as an array and reaches a builtin as a boxed matlab_mat*
     * (ptr) in an 'f' (scalar) slot. Pick a Spec where the only mismatches
     * are 'f' slots receiving a ptr, and unbox each via matlab_mat_to_scalar
     * at the call site. Allowlisted to builtins whose scalar args are
     * idiomatically computed this way, so a genuine matrix arg isn't
     * silently reduced to its first element. */
    if (!S) {
      static const llvm::StringSet<> AutoUnboxNames = {
        "awgn",
      };
      if (AutoUnboxNames.contains(Name)) {
        for (auto &E : Table) {
          if (E.MLName != Name) continue;
          if (E.ArgKinds.size() != NOps) continue;
          bool can_unbox = true;
          SmallVector<unsigned, 3> idx;
          for (unsigned i = 0; i < NOps; ++i) {
            char K = E.ArgKinds[i];
            Type Got = Call->getOperand(i).getType();
            if (K == 'f') {
              if (isFTagOK(Got)) {
                /* already matches strictly */
              } else if (Got == PtrTy) {
                idx.push_back(i);
              } else {
                can_unbox = false; break;
              }
            } else { /* 'p' */
              if (Got != PtrTy && !isTensorLike(Got) &&
                  !mlir::isa<mlir::NoneType>(Got)) {
                can_unbox = false; break;
              }
            }
          }
          if (can_unbox && !idx.empty()) {
            S = &E;
            UnboxIdx = std::move(idx);
            break;
          }
        }
      }
    }

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
        /* Degree-argument trigonometry, scalar forms. */
        {"sind", "matlab_sind_s"}, {"cosd", "matlab_cosd_s"},
        {"tand", "matlab_tand_s"}, {"asind", "matlab_asind_s"},
        {"acosd", "matlab_acosd_s"}, {"atand", "matlab_atand_s"},
        {"sinh", "matlab_sinh_s"}, {"cosh", "matlab_cosh_s"},
        {"tanh", "matlab_tanh_s"},
        {"log2", "matlab_log2_s"}, {"log10", "matlab_log10_s"},
        {"sign", "matlab_sign_s"},
        {"floor", "matlab_floor_s"}, {"ceil", "matlab_ceil_s"},
        {"round", "matlab_round_s"}, {"fix", "matlab_fix_s"},
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
      /* Two-argument scalar: atan2(y, x) / atan2d(y, x). */
      if ((Name == "atan2" || Name == "atan2d") &&
          Call->getNumOperands() == 2 &&
          Call->getOperand(0).getType() == F64 &&
          Call->getOperand(1).getType() == F64) {
        B.setInsertionPoint(Call);
        const char *RtName =
            (Name == "atan2d") ? "matlab_atan2d_s" : "matlab_atan2_s";
        auto Fn = rt(RtName, F64, {F64, F64});
        auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn,
                                        Call->getOperands());
        if (Call->getResult(0).getType() != F64)
          Call->getResult(0).setType(F64);
        carryName(Call, NC);
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
      carryName(Call, NC);
      Call->getResult(0).replaceAllUsesWith(NC.getResult());
      Call->erase();
      Changed = true;
      continue;
    }

    // Check argument count / types.
    if ((int)Call->getNumOperands() != (int)S->ArgKinds.size()) continue;
    SmallVector<Type, 3> ExpTys;
    /* Operand indices that need boxing skip the strict type check; we'll
     * materialize matlab_mat_from_scalar(f64) for each before the call.
     * BoxIdx is small (<= NOps, typically 1–3) so a linear scan beats
     * pulling in DenseSet. */
    auto BoxSet_count = [&](unsigned i) -> bool {
      for (unsigned x : BoxIdx) if (x == i) return true;
      return false;
    };
    auto UnboxSet_count = [&](unsigned i) -> bool {
      for (unsigned x : UnboxIdx) if (x == i) return true;
      return false;
    };
    bool OK = true;
    for (unsigned i = 0; i < S->ArgKinds.size(); ++i) {
      Type Exp = S->ArgKinds[i] == 'f' ? F64 : PtrTy;
      ExpTys.push_back(Exp);
      Type Got = Call->getOperand(i).getType();
      if (BoxSet_count(i)) continue;  /* will be boxed below */
      if (UnboxSet_count(i)) continue;  /* will be unboxed below */
      // Accept tensor-typed args where we expect ptr (we'll convert via a
      // subsequent retype — but only if the value is actually a ptr at
      // runtime). We'll be strict and require ptr now; tensor-typed inputs
      // come from allocs that our slot-retype handled, so by the time we
      // run this they should already be ptr.
      if (Exp == F64 && !isFTagOK(Got)) { OK = false; break; }
      if (Exp == PtrTy && Got != PtrTy) {
        /* Narrow opt-in: dlnet functional optimizers receive a tensor-typed
         * weight matrix from a zeros() alloc and a `none`-typed gradient
         * from dlgradient (upstream Sema doesn't infer ptr for either).
         * Accept tensor<*xf64> and `none` for those callees only —
         * broadening here would mis-route unrelated calls (e.g. linalg
         * multi-return helpers). */
        bool DlOptim = (Name == "adamupdate" || Name == "sgdmupdate" ||
                        Name == "rmspropupdate");
        if (DlOptim &&
            (mlir::isa<mlir::NoneType>(Got) || isTensorLike(Got))) {
          /* accept */
        } else {
          OK = false; break;
        }
      }
    }
    if (!OK) continue;

    Type ResTy = S->ResultKind == 0 ? F64 : PtrTy;
    B.setInsertionPoint(Call);
    SmallVector<Value, 3> CallOps;
    for (unsigned i = 0; i < Call->getNumOperands(); ++i) {
      Value V = Call->getOperand(i);
      if (BoxSet_count(i)) {
        /* If the boxed scalar arrived as f32 (e.g. `single(2.5)`),
         * widen to f64 before boxing — matlab_mat_from_scalar takes
         * a double. */
        if (mlir::isa<Float32Type>(V.getType())) {
          auto Ext = LLVM::FPExtOp::create(B, Call->getLoc(), F64, V);
          V = Ext.getResult();
        }
        auto FnBox = rt("matlab_mat_from_scalar", PtrTy, {F64});
        auto Box = LLVM::CallOp::create(B, Call->getLoc(), FnBox,
                                         ValueRange{V});
        CallOps.push_back(Box.getResult());
      } else if (UnboxSet_count(i)) {
        /* Boxed-scalar ptr in an 'f' slot — extract its scalar value. */
        auto FnUnbox = rt("matlab_mat_to_scalar", F64, {PtrTy});
        auto Ub = LLVM::CallOp::create(B, Call->getLoc(), FnUnbox,
                                       ValueRange{V});
        CallOps.push_back(Ub.getResult());
      } else {
        /* f32 -> f64 widening at every 'f' slot reached via a
         * single() / double() cast.  The runtime entry signature is
         * always (double, ...), so we extend before the call. */
        if (S->ArgKinds[i] == 'f' && mlir::isa<Float32Type>(V.getType())) {
          auto Ext = LLVM::FPExtOp::create(B, Call->getLoc(), F64, V);
          V = Ext.getResult();
        }
        CallOps.push_back(V);
      }
    }
    auto Fn = rt(S->RTName, ResTy, ExpTys);
    auto NC = LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{CallOps});
    carryName(Call, NC);
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
        N == "matlab.matldiv" || N == "matlab.matpow") {
      Binaries.push_back(Op); return;
    }
    for (auto &S : ElemSpecs)
      if (isMatlabOp(Op, S.MLName)) { Binaries.push_back(Op); return; }
  });

  bool Changed = false;
  for (Operation *Op : Binaries) {
    StringRef ML = Op->getName().getStringRef();
    Value A = Op->getOperand(0), BVal = Op->getOperand(1);
    /* #77: a matrix elementwise op with one matrix (ptr) operand needs its
     * scalar operand as f64 — the runtime `_ms`/`_sm` entries take a double.
     * A boxed comparison like `pred == (x > 0.5)` (pred a workspace scalar
     * boxed as matlab_mat*, the RHS a comparison) arrives as eq(ptr, i1),
     * which matched none of the mm/ms/sm shapes and survived unconverted.
     * Promote an i1/iN/f32 scalar operand to f64 when the other side is a
     * ptr so the _ms/_sm path matches. */
    {
      bool aPtr = A.getType() == PtrTy, bPtr = BVal.getType() == PtrTy;
      auto toF64Scalar = [&](Value V) -> Value {
        Type T = V.getType();
        if (T == F64 || T == PtrTy || isTensorLike(T)) return V;
        B.setInsertionPoint(Op);
        if (auto IT = mlir::dyn_cast<IntegerType>(T)) {
          if (IT.getWidth() == 1)
            return LLVM::UIToFPOp::create(B, Op->getLoc(), F64, V).getResult();
          return LLVM::SIToFPOp::create(B, Op->getLoc(), F64, V).getResult();
        }
        if (mlir::isa<Float32Type>(T))
          return LLVM::FPExtOp::create(B, Op->getLoc(), F64, V).getResult();
        return V;
      };
      if (aPtr && !bPtr) BVal = toF64Scalar(BVal);
      else if (bPtr && !aPtr) A = toF64Scalar(A);
    }
    Type AT = A.getType(), BT = BVal.getType();
    bool AP = AT == PtrTy, BP = BT == PtrTy;
    bool AF = AT == F64,    BF = BT == F64;
    /* `^` (matrix power): scalar base -> libm pow (no arith pow op exists, so
     * LowerScalarsToArith can't handle it); square-matrix base ^ scalar n ->
     * matlab_matpow. */
    if (ML == "matlab.matpow") {
      B.setInsertionPoint(Op);
      if (AF && BF) {
        auto Pf = rt("matlab_pow_scalar", F64, {F64, F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Pf, ValueRange{A, BVal});
        Op->getResult(0).replaceAllUsesWith(NC.getResult());
        Op->erase(); Changed = true;
      } else if (AP && BF) {
        auto Pf = rt("matlab_matpow", PtrTy, {PtrTy, F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Pf, ValueRange{A, BVal});
        Op->getResult(0).replaceAllUsesWith(NC.getResult());
        Op->erase(); Changed = true;
      }
      continue;
    }
    if (!AP && !BP) continue; // scalar-only — LowerScalarsToArith handled it

    B.setInsertionPoint(Op);
    LLVM::LLVMFuncOp Fn;
    SmallVector<Value, 2> Args;

    /* Phase 1.1.D: typed-int matrix lane (i32 / u8). Lowering attaches
     * a "dtype" StringAttr when either operand is a non-scalar Int32 /
     * UInt8 array. The runtime layer keeps separate descriptors and the
     * dispatch needs to pick matlab_mat_<lane>_<base>_<mm|ms|sm>. ms/sm
     * scalars are coerced from f64 via the public matlab_d_to_<lane>_sat
     * helpers so MATLAB's saturating cast semantics are preserved. */
    StringRef IntLane;
    if (auto DA = Op->getAttrOfType<StringAttr>("dtype"))
      IntLane = DA.getValue();

    auto coerceScalar = [&](Value V) -> Value {
      assert(!IntLane.empty());
      StringRef Helper = (IntLane == "i32") ? "matlab_d_to_i32_sat"
                                            : "matlab_d_to_u8_sat";
      Type IT = (IntLane == "i32") ? (Type)IntegerType::get(B.getContext(), 32)
                                   : (Type)IntegerType::get(B.getContext(), 8);
      auto Hf = rt(Helper.str(), IT, {F64});
      return LLVM::CallOp::create(B, Op->getLoc(), Hf, ValueRange{V})
          .getResult();
    };

    auto emitElem = [&](StringRef Base) {
      if (!IntLane.empty()) {
        std::string Pre = ("matlab_mat_" + IntLane + "_" + Base).str();
        Type IT = (IntLane == "i32")
                      ? (Type)IntegerType::get(B.getContext(), 32)
                      : (Type)IntegerType::get(B.getContext(), 8);
        if (AP && BP) {
          Fn = rt(Pre + "_mm", PtrTy, {PtrTy, PtrTy});
          Args = {A, BVal};
        } else if (AP && BF) {
          Fn = rt(Pre + "_ms", PtrTy, {PtrTy, IT});
          Args = {A, coerceScalar(BVal)};
        } else if (AF && BP) {
          Fn = rt(Pre + "_sm", PtrTy, {IT, PtrTy});
          Args = {coerceScalar(A), BVal};
        }
        return;
      }
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

    /* matmul/matdiv/matldiv on typed-int matrices fall back to f64 for
     * now (no native int LU solve / mtimes in the runtime); silence the
     * IntLane on those so the f64 branches below take effect. */
    if (ML == "matlab.matmul" || ML == "matlab.matdiv" ||
        ML == "matlab.matldiv")
      IntLane = "";

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
    carryName(Op, NC);
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
    carryName(Op, NC);
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
    carryName(Op, NC);
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
    B.setInsertionPoint(Op);
    /* #77: matlab_range takes f64 bounds, but a workspace scalar used as a
     * loop bound (`for i = 1:(nstate-1)`) can arrive as a boxed matlab_mat*
     * (ptr) or a non-f64 scalar. A range bound is scalar in MATLAB, so
     * coerce each operand to f64 — unboxing a ptr via matlab_mat_to_scalar
     * — instead of leaving the matlab.range unconverted. */
    auto toF64 = [&](Value V) -> Value {
      Type T = V.getType();
      if (T == F64) return V;
      if (T == PtrTy) {
        auto Fn = rt("matlab_mat_to_scalar", F64, {PtrTy});
        return LLVM::CallOp::create(B, Op->getLoc(), Fn, ValueRange{V})
            .getResult();
      }
      if (auto IT = mlir::dyn_cast<IntegerType>(T)) {
        if (IT.getWidth() == 1)
          return LLVM::UIToFPOp::create(B, Op->getLoc(), F64, V).getResult();
        return LLVM::SIToFPOp::create(B, Op->getLoc(), F64, V).getResult();
      }
      if (mlir::isa<Float32Type>(T))
        return LLVM::FPExtOp::create(B, Op->getLoc(), F64, V).getResult();
      return Value{};
    };
    SmallVector<Value, 3> Co;
    bool ok = true;
    for (unsigned i = 0; i < N; ++i) {
      Value C = toF64(Op->getOperand(i));
      if (!C) { ok = false; break; }
      Co.push_back(C);
    }
    if (!ok) continue;  /* an operand we can't coerce — leave for another pass */
    Value Start = Co[0];
    Value Step, End;
    if (N == 2) {
      End = Co[1];
      Step = LLVM::ConstantOp::create(B, Op->getLoc(), F64,
                                       B.getF64FloatAttr(1.0));
    } else {
      Step = Co[1];
      End  = Co[2];
    }
    auto Fn = rt("matlab_range", PtrTy, {F64, F64, F64});
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Start, Step, End});
    carryName(Op, NC);
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
    carryName(Op, NC);
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
    /* Accept N=2 (one index), N=3 (two indices), N=5 (four indices — Tier C
     * rank-4 fast path), and N>=6 (five-plus indices — rank>=5 variadic
     * path, #93). */
    if (N != 2 && N != 3 && N != 5 && N < 6) continue;
    /* Base may be PtrTy (matlab_mat / matlab_mat3 / matlab_matN pointer)
     * or `tensor<*xf64>` from the early-tracking lane.  Both reach the
     * runtime as a plain pointer. */
    if (Op->getOperand(0).getType() != PtrTy &&
        !mlir::isa<mlir::TensorType>(Op->getOperand(0).getType()))
      continue;

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
      if (N >= 6) {
        /* A(i,j,k,l,m[,...]) — rank>=5.  Pack the N-1 scalar indices into a
         * stack int64_t[] and call the variadic runtime helper
         * matlab_subscriptN_s(base, nidx, idx_ptr), which is generic to
         * 16 dims and falls back to lower-rank descriptors.  #93. */
        Location Loc = Op->getLoc();
        unsigned NIdx = N - 1;
        Value One = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(1));
        auto ArrayTy = LLVM::LLVMArrayType::get(I64, NIdx);
        Value Buf = LLVM::AllocaOp::create(B, Loc, PtrTy, ArrayTy, One,
                                            /*alignment=*/0);
        for (unsigned k = 0; k < NIdx; ++k) {
          Value Iv = arith::FPToSIOp::create(B, Loc, I64, Op->getOperand(k + 1));
          Value Idx = LLVM::ConstantOp::create(B, Loc, I64,
                                                B.getI64IntegerAttr(k));
          Value ElemPtr = LLVM::GEPOp::create(B, Loc, PtrTy, I64, Buf,
                                               ValueRange{Idx});
          LLVM::StoreOp::create(B, Loc, Iv, ElemPtr);
        }
        Value NIdxV = LLVM::ConstantOp::create(B, Loc, I64,
                                                B.getI64IntegerAttr(NIdx));
        auto Fn = rt("matlab_subscriptN_s", F64, {PtrTy, I64, PtrTy});
        auto NC = LLVM::CallOp::create(B, Loc, Fn,
                                        ValueRange{Base, NIdxV, Buf});
        Op->getResult(0).replaceAllUsesWith(NC.getResult());
      } else if (N == 5) {
        /* A(i,j,k,l) on a rank-4 (or higher with implicit trailing dims)
         * array.  Routes through matlab_subscript4_s, which is N-D-aware
         * and falls back to 2-D / 3-D when the descriptor is narrower. */
        auto Fn = rt("matlab_subscript4_s", F64,
                     {PtrTy, F64, F64, F64, F64});
        auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                        ValueRange{Base, Op->getOperand(1),
                                                   Op->getOperand(2),
                                                   Op->getOperand(3),
                                                   Op->getOperand(4)});
        Op->getResult(0).replaceAllUsesWith(NC.getResult());
      } else if (N == 3) {
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

    /* Slice (non-scalar index) forms for rank>=5 are out of scope (#93
     * covers scalar element access).  Leave the subscript unconverted
     * rather than mis-lower it to a 1-D/2-D slice. */
    if (N >= 6) continue;

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

/* After rewriteBinaryOps, scf.if / scf.condition operands that started
 * as matlab.gt/lt/eq/...:i1 may now be matlab_*_mm:!llvm.ptr. Coerce
 * each one back to i1 with a matlab_mat_truth(ptr) -> i8 call followed
 * by a cmp ne 0.
 *
 * Similarly arith.cmpf/cmpi can be left with a ptr operand when an
 * earlier scalar lowering folded matlab.gt(f64,f64) to arith.cmpf and
 * one of the operands later got retyped from f64 to ptr (e.g. abs of
 * a matrix). Rewrite those into a matlab_<base>_mm/_ms/_sm runtime
 * call, then matlab_mat_truth, then cmpi ne 0. Idempotent — runs in
 * the LowerTensorOps fixpoint. */
bool TensorLowering::fixupCondOperands() {
  auto I8 = IntegerType::get(Ctx, 8);
  bool Changed = false;

  auto matTruth = [&](Value V, Location L) -> Value {
    auto Fn = rt("matlab_mat_truth", I8, {PtrTy});
    auto Call = LLVM::CallOp::create(B, L, Fn, ValueRange{V});
    Value Zero = arith::ConstantOp::create(
        B, L, I8, B.getIntegerAttr(I8, 0));
    return arith::CmpIOp::create(
        B, L, arith::CmpIPredicate::ne, Call.getResult(), Zero);
  };

  auto coerce = [&](Operation *User, unsigned OpIdx) -> bool {
    Value V = User->getOperand(OpIdx);
    if (V.getType() != PtrTy) return false;
    B.setInsertionPoint(User);
    User->setOperand(OpIdx, matTruth(V, User->getLoc()));
    return true;
  };

  Mod.walk([&](Operation *Op) {
    if (auto If = dyn_cast<scf::IfOp>(Op)) {
      if (coerce(If, 0)) Changed = true;
    } else if (auto Cond = dyn_cast<scf::ConditionOp>(Op)) {
      if (coerce(Cond, 0)) Changed = true;
    }
  });

  /* Rewrite scalar arith.cmpf with a leaked ptr operand into a runtime
   * matrix comparison + truth coercion. The dialect rejects ptr
   * operands, so this MUST run before the LLVM conversion pipeline. */
  SmallVector<arith::CmpFOp> CmpFs;
  SmallVector<arith::CmpIOp> CmpIs;
  Mod.walk([&](Operation *Op) {
    if (auto Cf = dyn_cast<arith::CmpFOp>(Op)) {
      if (Cf.getLhs().getType() == PtrTy || Cf.getRhs().getType() == PtrTy)
        CmpFs.push_back(Cf);
    } else if (auto Ci = dyn_cast<arith::CmpIOp>(Op)) {
      if (Ci.getLhs().getType() == PtrTy || Ci.getRhs().getType() == PtrTy)
        CmpIs.push_back(Ci);
    }
  });

  auto baseForCmpF = [](arith::CmpFPredicate P) -> StringRef {
    using P_ = arith::CmpFPredicate;
    switch (P) {
      case P_::OEQ: case P_::UEQ: return "eq";
      case P_::ONE: case P_::UNE: return "ne";
      case P_::OLT: case P_::ULT: return "lt";
      case P_::OLE: case P_::ULE: return "le";
      case P_::OGT: case P_::UGT: return "gt";
      case P_::OGE: case P_::UGE: return "ge";
      default: return "";
    }
  };
  auto baseForCmpI = [](arith::CmpIPredicate P) -> StringRef {
    using P_ = arith::CmpIPredicate;
    switch (P) {
      case P_::eq:                       return "eq";
      case P_::ne:                       return "ne";
      case P_::slt: case P_::ult:        return "lt";
      case P_::sle: case P_::ule:        return "le";
      case P_::sgt: case P_::ugt:        return "gt";
      case P_::sge: case P_::uge:        return "ge";
    }
    return "";
  };

  auto rewriteCmpToMat = [&](Operation *Op, Value LHS, Value RHS,
                             StringRef Base) {
    if (Base.empty()) return;
    bool LP = LHS.getType() == PtrTy, RP = RHS.getType() == PtrTy;
    bool LF = LHS.getType() == B.getF64Type();
    bool RF = RHS.getType() == B.getF64Type();
    /* Only handle ptr/ptr, ptr/f64, f64/ptr — anything else we leave
     * for verification to surface. */
    if (!((LP && RP) || (LP && RF) || (LF && RP))) return;
    B.setInsertionPoint(Op);
    LLVM::LLVMFuncOp Fn;
    SmallVector<Value, 2> Args;
    if (LP && RP) {
      Fn = rt(("matlab_" + Base + "_mm").str(), PtrTy, {PtrTy, PtrTy});
      Args = {LHS, RHS};
    } else if (LP && RF) {
      Fn = rt(("matlab_" + Base + "_ms").str(), PtrTy, {PtrTy, B.getF64Type()});
      Args = {LHS, RHS};
    } else {
      Fn = rt(("matlab_" + Base + "_sm").str(), PtrTy, {B.getF64Type(), PtrTy});
      Args = {LHS, RHS};
    }
    auto Call = LLVM::CallOp::create(B, Op->getLoc(), Fn, Args);
    Value I1V = matTruth(Call.getResult(), Op->getLoc());
    Op->getResult(0).replaceAllUsesWith(I1V);
    Op->erase();
  };

  for (auto Cf : CmpFs) {
    rewriteCmpToMat(Cf, Cf.getLhs(), Cf.getRhs(),
                    baseForCmpF(Cf.getPredicate()));
    Changed = true;
  }
  for (auto Ci : CmpIs) {
    rewriteCmpToMat(Ci, Ci.getLhs(), Ci.getRhs(),
                    baseForCmpI(Ci.getPredicate()));
    Changed = true;
  }
  return Changed;
}

bool TensorLowering::unboxScalarArithOperands() {
  SmallVector<Operation *> Ops;
  Mod.walk([&](Operation *Op) {
    if (!isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::NegFOp>(Op))
      return;
    for (Value O : Op->getOperands())
      if (O.getType() == PtrTy) { Ops.push_back(Op); break; }
  });
  bool Changed = false;
  for (Operation *Op : Ops) {
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_mat_to_scalar", F64, {PtrTy});
    for (unsigned i = 0; i < Op->getNumOperands(); ++i) {
      if (Op->getOperand(i).getType() != PtrTy) continue;
      Value S = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                     ValueRange{Op->getOperand(i)})
                    .getResult();
      Op->setOperand(i, S);
    }
    Changed = true;
  }
  return Changed;
}

/* Lower matlab.call_builtin @matlab_mat_truth(ptr) -> i8 to a direct
 * LLVM call. Emitted by Lowerer::fixupIfCond when a scf.if / matlab.while
 * cond resolves to a matrix pointer (DAP/REPL workspace path).
 *
 * #77 JIT/-dap parity: fixupIfCond stamps the matlab_mat_truth wrapper
 * while the cond is still a ptr, but LowerScalarsToArith may afterwards
 * fold the producing op (e.g. `&&` of boxed scalars) down to a concrete
 * i1/iN/f64, leaving matlab_mat_truth with a non-ptr operand. A scalar's
 * truth is just `operand != 0`, so coerce it to i8 in-place rather than
 * calling the (ptr-typed) runtime helper. */
bool TensorLowering::rewriteMatTruth() {
  SmallVector<Operation *> Calls;
  Mod.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto CA = Op->getAttrOfType<StringAttr>("callee");
    if (!CA || CA.getValue() != "matlab_mat_truth") return;
    if (Op->getNumOperands() != 1) return;
    if (Op->getNumResults() != 1) return;
    Type OT = Op->getOperand(0).getType();
    if (OT != PtrTy && !mlir::isa<IntegerType>(OT) &&
        !mlir::isa<Float32Type, Float64Type>(OT))
      return;
    Calls.push_back(Op);
  });
  bool Changed = false;
  auto I8 = IntegerType::get(Ctx, 8);
  for (Operation *Op : Calls) {
    B.setInsertionPoint(Op);
    Value Arg = Op->getOperand(0);
    Type OT = Arg.getType();
    Value Truth;
    if (OT == PtrTy) {
      auto Fn = rt("matlab_mat_truth", I8, {PtrTy});
      Truth = LLVM::CallOp::create(B, Op->getLoc(), Fn, ValueRange{Arg})
                  .getResult();
    } else {
      // Scalar truth: (arg != 0) -> i1, widened to the i8 the consumers
      // (a cmpi ne 0 emitted by fixupIfCond) expect.
      Value NeZero;
      if (auto IT = mlir::dyn_cast<IntegerType>(OT)) {
        Value Z = arith::ConstantOp::create(B, Op->getLoc(), IT,
                                            B.getIntegerAttr(IT, 0));
        NeZero = arith::CmpIOp::create(B, Op->getLoc(),
                                       arith::CmpIPredicate::ne, Arg, Z);
      } else {
        Value Z = arith::ConstantOp::create(B, Op->getLoc(), OT,
                                            B.getFloatAttr(OT, 0.0));
        NeZero = arith::CmpFOp::create(B, Op->getLoc(),
                                       arith::CmpFPredicate::ONE, Arg, Z);
      }
      Truth = arith::ExtUIOp::create(B, Op->getLoc(), I8, NeZero);
    }
    Op->getResult(0).replaceAllUsesWith(Truth);
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
    Changed |= rewriteComplexLiterals();
    Changed |= rewriteBuiltinCalls();
    Changed |= rewriteLiterals();
    Changed |= rewriteBinaryOps();
    Changed |= rewritePostfix();
    Changed |= rewriteUnaryNeg();
    Changed |= rewriteRange();
    Changed |= rewriteSubscript();
    Changed |= rewriteSubscriptStore();
    Changed |= rewriteDispMatrix();
    Changed |= rewriteMatTruth();
    Changed |= fixupCondOperands();
    Changed |= unboxScalarArithOperands();
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
