// Rewrites matlab.const_char and matlab.call_builtin @disp/@fprintf into
// LLVM-dialect globals + calls to the `matlab_runtime.c` shim. Also renames
// `func.func @script` to `@main` so the result is directly linkable.
//
// The produced module still contains func.func and arith ops; a follow-up
// pass pipeline (convert-arith-to-llvm, convert-func-to-llvm, etc.) finishes
// the lowering before LLVM IR translation.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace matlab {
namespace mlirgen {

namespace {

using namespace mlir;

bool isMatlabOp(Operation *Op, StringRef Name) {
  return Op && Op->getName().getStringRef() == Name;
}

//===----------------------------------------------------------------------===//
// Runtime function declarations
//===----------------------------------------------------------------------===//

LLVM::LLVMFuncOp getOrInsertRuntimeFunc(OpBuilder &B, ModuleOp M,
                                        StringRef Name, Type ResultTy,
                                        ArrayRef<Type> ArgTypes) {
  if (auto Existing = M.lookupSymbol<LLVM::LLVMFuncOp>(Name)) return Existing;
  OpBuilder::InsertionGuard G(B);
  B.setInsertionPointToStart(M.getBody());
  auto FnTy = LLVM::LLVMFunctionType::get(ResultTy, ArgTypes);
  auto F = LLVM::LLVMFuncOp::create(B, M.getLoc(), Name, FnTy);
  F.setLinkage(LLVM::Linkage::External);
  return F;
}

//===----------------------------------------------------------------------===//
// String globals
//===----------------------------------------------------------------------===//

// Pre-allocate a unique string global for each matlab.const_char value,
// so repeated occurrences of the same literal share storage.
struct StringGlobals {
  ModuleOp M;
  OpBuilder &B;
  int Counter = 0;

  StringGlobals(ModuleOp M, OpBuilder &B) : M(M), B(B) {}

  LLVM::GlobalOp getOrCreate(StringRef Text) {
    // Look for an existing global with the same value.
    for (auto G : M.getOps<LLVM::GlobalOp>()) {
      if (!G.getConstant()) continue;
      auto Attr = mlir::dyn_cast_or_null<StringAttr>(G.getValueAttr());
      if (Attr && Attr.getValue() == Text) return G;
    }
    OpBuilder::InsertionGuard Guard(B);
    B.setInsertionPointToStart(M.getBody());
    MLIRContext *Ctx = M.getContext();
    auto ArrayTy = LLVM::LLVMArrayType::get(
        IntegerType::get(Ctx, 8),
        static_cast<unsigned>(Text.size()));
    std::string Name = ("__matlab_str" + llvm::Twine(Counter++)).str();
    auto G = LLVM::GlobalOp::create(
        B, M.getLoc(), ArrayTy, /*isConstant=*/true,
        LLVM::Linkage::Internal, Name,
        StringAttr::get(Ctx, Text));
    return G;
  }
};

//===----------------------------------------------------------------------===//
// Per-op rewrites
//===----------------------------------------------------------------------===//

// Returns (ptrValue, length) for a matlab.const_char result value flowing into
// disp/fprintf. Creates the global + addressof if needed. Returns nullopt if
// the argument isn't a direct matlab.const_char (i.e. a dynamic string).
std::optional<std::pair<Value, int64_t>>
materializeStringArg(Value V, OpBuilder &B, StringGlobals &Strings) {
  Operation *Def = V.getDefiningOp();
  if (!isMatlabOp(Def, "matlab.const_char")) return std::nullopt;
  auto ValAttr = Def->getAttrOfType<StringAttr>("value");
  if (!ValAttr) return std::nullopt;
  StringRef Text = ValAttr.getValue();
  auto G = Strings.getOrCreate(Text);

  OpBuilder::InsertionGuard Guard(B);
  B.setInsertionPointAfter(Def);
  MLIRContext *Ctx = B.getContext();
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  Value Addr = LLVM::AddressOfOp::create(B, Def->getLoc(), PtrTy, G.getSymName());
  return std::make_pair(Addr, (int64_t)Text.size());
}

LogicalResult rewriteDispCall(Operation *Call, OpBuilder &B,
                              StringGlobals &Strings) {
  if (Call->getNumOperands() != 1) return failure();
  Value Arg = Call->getOperand(0);
  MLIRContext *Ctx = B.getContext();
  auto I64 = IntegerType::get(Ctx, 64);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto VoidTy = LLVM::LLVMVoidType::get(Ctx);

  B.setInsertionPoint(Call);
  ModuleOp M = Call->getParentOfType<ModuleOp>();

  if (auto Pair = materializeStringArg(Arg, B, Strings)) {
    auto [Ptr, Len] = *Pair;
    auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_disp_str", VoidTy,
                                     {PtrTy, I64});
    B.setInsertionPoint(Call);
    Value LenV = LLVM::ConstantOp::create(
        B, Call->getLoc(), I64, B.getI64IntegerAttr(Len));
    LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Ptr, LenV});
    Call->erase();
    return success();
  }

  if (Arg.getType() == F64) {
    auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_disp_f64", VoidTy, {F64});
    LLVM::CallOp::create(B, Call->getLoc(), Fn, ValueRange{Arg});
    Call->erase();
    return success();
  }

  // tensor<NxF64> (vector) / tensor<MxNxF64> (matrix) literal arguments:
  // walk the matlab.concat_* chain back to scalar defining ops, stack-alloc
  // a contiguous row-major buffer, fill it, and call the appropriate runtime
  // entry point.
  auto rankedTy = mlir::dyn_cast<RankedTensorType>(Arg.getType());
  if (!rankedTy || rankedTy.getElementType() != F64 ||
      !rankedTy.hasStaticShape()) return failure();

  auto Shape = rankedTy.getShape();
  int64_t Rows, Cols;
  llvm::SmallVector<Value, 16> Elements; // row-major

  Operation *Def = Arg.getDefiningOp();
  if (!Def) return failure();

  if (Shape.size() == 1) {
    // Row vector: the producer should be a matlab.concat_row.
    Rows = 1;
    Cols = Shape[0];
    if (!isMatlabOp(Def, "matlab.concat_row")) return failure();
    if ((int64_t)Def->getNumOperands() != Cols) return failure();
    for (Value V : Def->getOperands()) {
      if (V.getType() != F64) return failure();
      Elements.push_back(V);
    }
  } else if (Shape.size() == 2) {
    Rows = Shape[0];
    Cols = Shape[1];
    if (!isMatlabOp(Def, "matlab.concat_col")) return failure();
    if ((int64_t)Def->getNumOperands() != Rows) return failure();
    for (Value RowV : Def->getOperands()) {
      Operation *Row = RowV.getDefiningOp();
      if (!isMatlabOp(Row, "matlab.concat_row")) return failure();
      if ((int64_t)Row->getNumOperands() != Cols) return failure();
      for (Value E : Row->getOperands()) {
        if (E.getType() != F64) return failure();
        Elements.push_back(E);
      }
    }
  } else {
    return failure();
  }

  // Stack-allocate Rows*Cols doubles, then GEP + store each element.
  auto Loc = Call->getLoc();
  Value One = LLVM::ConstantOp::create(
      B, Loc, I64, B.getI64IntegerAttr(1));
  auto ArrayTy = LLVM::LLVMArrayType::get(F64, (unsigned)(Rows * Cols));
  Value BufPtr = LLVM::AllocaOp::create(
      B, Loc, PtrTy, ArrayTy, One, /*alignment=*/0);
  for (int64_t k = 0; k < Rows * Cols; ++k) {
    Value Idx = LLVM::ConstantOp::create(
        B, Loc, I64, B.getI64IntegerAttr(k));
    Value ElemPtr = LLVM::GEPOp::create(
        B, Loc, PtrTy, F64, BufPtr, ValueRange{Idx});
    LLVM::StoreOp::create(B, Loc, Elements[k], ElemPtr);
  }

  if (Shape.size() == 1) {
    auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_disp_vec_f64",
                                     VoidTy, {PtrTy, I64});
    Value NV = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(Cols));
    LLVM::CallOp::create(B, Loc, Fn, ValueRange{BufPtr, NV});
  } else {
    auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_disp_mat_f64",
                                     VoidTy, {PtrTy, I64, I64});
    Value MV = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(Rows));
    Value NV = LLVM::ConstantOp::create(B, Loc, I64, B.getI64IntegerAttr(Cols));
    LLVM::CallOp::create(B, Loc, Fn, ValueRange{BufPtr, MV, NV});
  }

  Call->erase();
  return success();
}

LogicalResult rewriteFprintfCall(Operation *Call, OpBuilder &B,
                                 StringGlobals &Strings) {
  unsigned NOps = Call->getNumOperands();
  if (NOps < 1 || NOps > 5) return failure();

  MLIRContext *Ctx = B.getContext();
  auto I64 = IntegerType::get(Ctx, 64);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto VoidTy = LLVM::LLVMVoidType::get(Ctx);

  auto FmtPair = materializeStringArg(Call->getOperand(0), B, Strings);
  if (!FmtPair) return failure();
  auto [FmtPtr, FmtLen] = *FmtPair;

  /* All extra args must be f64 for now — matching matlab_fprintf_f64_* . */
  for (unsigned i = 1; i < NOps; ++i)
    if (Call->getOperand(i).getType() != F64) return failure();

  B.setInsertionPoint(Call);
  ModuleOp M = Call->getParentOfType<ModuleOp>();
  Value FmtLenV = LLVM::ConstantOp::create(
      B, Call->getLoc(), I64, B.getI64IntegerAttr(FmtLen));

  SmallVector<Value, 6> Args{FmtPtr, FmtLenV};
  for (unsigned i = 1; i < NOps; ++i) Args.push_back(Call->getOperand(i));

  /* Pick the matching runtime symbol by arity. */
  StringRef Name;
  SmallVector<Type, 6> Sig{PtrTy, I64};
  switch (NOps) {
    case 1: Name = "matlab_fprintf_str";  break;
    case 2: Name = "matlab_fprintf_f64";   Sig.push_back(F64); break;
    case 3: Name = "matlab_fprintf_f64_2"; Sig.append({F64, F64}); break;
    case 4: Name = "matlab_fprintf_f64_3"; Sig.append({F64, F64, F64}); break;
    case 5: Name = "matlab_fprintf_f64_4"; Sig.append({F64, F64, F64, F64}); break;
    default: return failure();
  }
  auto Fn = getOrInsertRuntimeFunc(B, M, Name, VoidTy, Sig);
  LLVM::CallOp::create(B, Call->getLoc(), Fn, Args);
  Call->erase();
  return success();
}

void renameScriptToMain(ModuleOp M) {
  auto Script = M.lookupSymbol<func::FuncOp>("script");
  if (!Script) return;
  if (M.lookupSymbol<func::FuncOp>("main")) return;
  auto FnType = Script.getFunctionType();
  if (FnType.getNumInputs() != 0 || FnType.getNumResults() != 0) return;

  // Rewrite signature to `() -> i32` so the program's exit status is a clean
  // 0 rather than whatever leaked from the last register. Replace each
  // existing `func.return` with `func.return %c0 : i32`.
  MLIRContext *Ctx = M.getContext();
  OpBuilder B(Ctx);
  auto I32 = IntegerType::get(Ctx, 32);
  Script.setName("main");
  auto NewType = FunctionType::get(Ctx, {}, {I32});
  Script.setFunctionType(NewType);

  Script.walk([&](func::ReturnOp Ret) {
    B.setInsertionPoint(Ret);
    Value Zero = arith::ConstantOp::create(
        B, Ret.getLoc(), I32, B.getI32IntegerAttr(0));
    func::ReturnOp::create(B, Ret.getLoc(), ValueRange{Zero});
    Ret.erase();
  });
}

} // namespace

bool runLowerIO(ModuleOp M) {
  OpBuilder B(M.getContext());
  StringGlobals Strings(M, B);

  // Canonicalize single-operand matlab.concat_row whose result type matches
  // the operand type — these come from degenerate 1x1 matrix literals like
  // `[7]` and just pass the value through. Fold them away so subsequent
  // lowering (and the dead-code sweep) sees the scalar directly.
  {
    llvm::SmallVector<Operation *, 8> Trivial;
    M.walk([&](Operation *Op) {
      if (isMatlabOp(Op, "matlab.concat_row") && Op->getNumOperands() == 1 &&
          Op->getResult(0).getType() == Op->getOperand(0).getType())
        Trivial.push_back(Op);
    });
    for (Operation *Op : Trivial) {
      Op->getResult(0).replaceAllUsesWith(Op->getOperand(0));
      Op->erase();
    }
  }

  llvm::SmallVector<Operation *, 16> ToRewrite;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call_builtin")) {
      auto Callee = Op->getAttrOfType<StringAttr>("callee");
      if (!Callee) return;
      if (Callee.getValue() == "disp" ||
          Callee.getValue() == "fprintf" ||
          Callee.getValue() == "input") {
        ToRewrite.push_back(Op);
      }
    }
  });

  bool Changed = false;
  for (Operation *Op : ToRewrite) {
    auto Callee = Op->getAttrOfType<StringAttr>("callee");
    if (Callee.getValue() == "disp") {
      if (succeeded(rewriteDispCall(Op, B, Strings))) Changed = true;
    } else if (Callee.getValue() == "fprintf") {
      if (succeeded(rewriteFprintfCall(Op, B, Strings))) Changed = true;
    } else if (Callee.getValue() == "input") {
      /* input(prompt) numeric variant. Prompt must be a char literal. */
      if (Op->getNumOperands() != 1) continue;
      auto Pair = materializeStringArg(Op->getOperand(0), B, Strings);
      if (!Pair) continue;
      auto [Ptr, Len] = *Pair;
      MLIRContext *Ctx = B.getContext();
      auto I64 = IntegerType::get(Ctx, 64);
      auto F64 = Float64Type::get(Ctx);
      auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
      B.setInsertionPoint(Op);
      Value LenV = LLVM::ConstantOp::create(B, Op->getLoc(), I64,
                                             B.getI64IntegerAttr(Len));
      auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_input_num", F64,
                                       {PtrTy, I64});
      auto NC = LLVM::CallOp::create(B, Op->getLoc(), Fn,
                                      ValueRange{Ptr, LenV});
      Op->getResult(0).replaceAllUsesWith(NC.getResult());
      Op->erase();
      Changed = true;
    }
  }

  // Iteratively erase matlab.const_char / matlab.concat_* ops that became
  // dead once their disp consumers were rewritten. We loop because erasing a
  // concat_col makes its concat_row operands dead in the next sweep.
  for (;;) {
    llvm::SmallVector<Operation *, 16> Dead;
    M.walk([&](Operation *Op) {
      if ((isMatlabOp(Op, "matlab.const_char") ||
           isMatlabOp(Op, "matlab.concat_row") ||
           isMatlabOp(Op, "matlab.concat_col")) &&
          Op->use_empty())
        Dead.push_back(Op);
    });
    if (Dead.empty()) break;
    for (Operation *Op : Dead) { Op->erase(); Changed = true; }
  }

  renameScriptToMain(M);
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
