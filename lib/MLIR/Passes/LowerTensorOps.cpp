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
    if (!gatherLiteralElements(Op, Rows, Cols, Elts)) continue;
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
         Name == "matlab_ws_get_symmat") &&
        Call->getNumOperands() == 1 && Call->getNumResults() == 1) {
      Value NameV = Call->getOperand(0);
      int64_t Len = 0;
      Value Ptr = fieldNameAddr(NameV, Len);
      if (!Ptr) continue;
      bool IsMat = (Name == "matlab_ws_get_mat" ||
                    Name == "matlab_ws_get_string" ||
                    Name == "matlab_ws_get_sym" ||
                    Name == "matlab_ws_get_symmat");
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
    if ((Name == "matlab_ws_set_f64" || Name == "matlab_ws_set_mat" ||
         Name == "matlab_ws_set_obj" || Name == "matlab_ws_set_string" ||
         Name == "matlab_ws_set_sym" || Name == "matlab_ws_set_symmat") &&
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
       * the value is a generic ptr. */
      bool IsObj = (Name == "matlab_ws_set_obj");
      bool IsString = (Name == "matlab_ws_set_string");
      bool IsSym = (Name == "matlab_ws_set_sym");
      bool IsSymmat = (Name == "matlab_ws_set_symmat");
      bool IsMat;
      bool IsInt = mlir::isa<mlir::IntegerType>(Val.getType());
      if (IsObj || IsString || IsSym || IsSymmat) IsMat = true;
      else if (Val.getType() == PtrTy)      IsMat = true;
      else if (Val.getType() == F64)         IsMat = false;
      else if (IsInt)                         IsMat = false;
      else continue;   /* neither ptr nor f64 nor int yet — wait for another iter */
      if ((IsObj || IsString || IsSym || IsSymmat) && Val.getType() != PtrTy)
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
          IsSymmat   ? "matlab_ws_set_symmat"
                     : (IsSym ? "matlab_ws_set_sym"
                              : (IsString ? "matlab_ws_set_string"
                                          : (IsObj ? "matlab_ws_set_obj"
                                                   : (IsMat ? "matlab_ws_set_mat"
                                                            : "matlab_ws_set_f64"))));
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
      carryName(Call, NC);
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
      {"min",        "matlab_min_mm",     1, "pp"},  /* min(A, B) elementwise */
      {"max",        "matlab_max",        1, "p"},
      {"max",        "matlab_max_mm",     1, "pp"},  /* max(A, B) elementwise */
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
      {"mod",        "matlab_mod_s",      0, "ff"},
      {"rem",        "matlab_rem_s",      0, "ff"},
      {"atan2",      "matlab_atan2_m",    1, "pp"},
      {"inv",        "matlab_inv",        1, "p"},
      {"det",        "matlab_det",        0, "p"},
      {"svd",        "matlab_svd",        1, "p"},
      {"eig",        "matlab_eig",        1, "p"},
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
    if (!S) {
      static const llvm::StringSet<> AutoBoxNames = {
        "conv", "conv2", "filter", "xcorr",
        "polyval", "polyfit", "interp1", "interp2",
        "trapz", "cumtrapz", "imfilter", "padarray",
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
              if (Got != F64) { can_box = false; break; }
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
    bool OK = true;
    for (unsigned i = 0; i < S->ArgKinds.size(); ++i) {
      Type Exp = S->ArgKinds[i] == 'f' ? F64 : PtrTy;
      ExpTys.push_back(Exp);
      Type Got = Call->getOperand(i).getType();
      if (BoxSet_count(i)) continue;  /* will be boxed below */
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
    SmallVector<Value, 3> CallOps;
    for (unsigned i = 0; i < Call->getNumOperands(); ++i) {
      Value V = Call->getOperand(i);
      if (BoxSet_count(i)) {
        auto FnBox = rt("matlab_mat_from_scalar", PtrTy, {F64});
        auto Box = LLVM::CallOp::create(B, Call->getLoc(), FnBox,
                                         ValueRange{V});
        CallOps.push_back(Box.getResult());
      } else {
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

/* Lower matlab.call_builtin @matlab_mat_truth(ptr) -> i8 to a direct
 * LLVM call. Emitted by Lowerer::fixupIfCond when a scf.if / matlab.while
 * cond resolves to a matrix pointer (DAP/REPL workspace path). */
bool TensorLowering::rewriteMatTruth() {
  SmallVector<Operation *> Calls;
  Mod.walk([&](Operation *Op) {
    if (!isMatlabOp(Op, "matlab.call_builtin")) return;
    auto CA = Op->getAttrOfType<StringAttr>("callee");
    if (!CA || CA.getValue() != "matlab_mat_truth") return;
    if (Op->getNumOperands() != 1) return;
    if (Op->getOperand(0).getType() != PtrTy) return;
    if (Op->getNumResults() != 1) return;
    Calls.push_back(Op);
  });
  bool Changed = false;
  auto I8 = IntegerType::get(Ctx, 8);
  for (Operation *Op : Calls) {
    B.setInsertionPoint(Op);
    auto Fn = rt("matlab_mat_truth", I8, {PtrTy});
    auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                    ValueRange{Op->getOperand(0)});
    Op->getResult(0).replaceAllUsesWith(NC.getResult());
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
