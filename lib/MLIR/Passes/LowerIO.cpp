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

  // Unhandled type — leave the op in place.
  return failure();
}

LogicalResult rewriteFprintfCall(Operation *Call, OpBuilder &B,
                                 StringGlobals &Strings) {
  if (Call->getNumOperands() < 1 || Call->getNumOperands() > 2)
    return failure();

  MLIRContext *Ctx = B.getContext();
  auto I64 = IntegerType::get(Ctx, 64);
  auto F64 = Float64Type::get(Ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(Ctx);
  auto VoidTy = LLVM::LLVMVoidType::get(Ctx);

  auto FmtPair = materializeStringArg(Call->getOperand(0), B, Strings);
  if (!FmtPair) return failure();
  auto [FmtPtr, FmtLen] = *FmtPair;

  B.setInsertionPoint(Call);
  ModuleOp M = Call->getParentOfType<ModuleOp>();
  Value FmtLenV = LLVM::ConstantOp::create(
      B, Call->getLoc(), I64, B.getI64IntegerAttr(FmtLen));

  if (Call->getNumOperands() == 1) {
    auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_fprintf_str", VoidTy,
                                     {PtrTy, I64});
    LLVM::CallOp::create(B, Call->getLoc(), Fn,
                         ValueRange{FmtPtr, FmtLenV});
    Call->erase();
    return success();
  }
  // Two operands: format + one f64 value.
  Value Num = Call->getOperand(1);
  if (Num.getType() != F64) return failure();
  auto Fn = getOrInsertRuntimeFunc(B, M, "matlab_fprintf_f64", VoidTy,
                                   {PtrTy, I64, F64});
  LLVM::CallOp::create(B, Call->getLoc(), Fn,
                       ValueRange{FmtPtr, FmtLenV, Num});
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

  llvm::SmallVector<Operation *, 16> ToRewrite;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.call_builtin")) {
      auto Callee = Op->getAttrOfType<StringAttr>("callee");
      if (!Callee) return;
      if (Callee.getValue() == "disp" ||
          Callee.getValue() == "fprintf") {
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
    }
  }

  // Collect matlab.const_char ops that have no more uses after the rewrites.
  llvm::SmallVector<Operation *, 16> DeadCharConsts;
  M.walk([&](Operation *Op) {
    if (isMatlabOp(Op, "matlab.const_char") && Op->use_empty())
      DeadCharConsts.push_back(Op);
  });
  for (Operation *Op : DeadCharConsts) { Op->erase(); Changed = true; }

  renameScriptToMain(M);
  return Changed;
}

} // namespace mlirgen
} // namespace matlab
