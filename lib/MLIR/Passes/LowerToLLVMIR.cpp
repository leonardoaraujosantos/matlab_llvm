// Runs the standard MLIR conversion pipeline to produce a module in the LLVM
// dialect, then translates it into LLVM IR textual form.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>

namespace matlab {
namespace mlirgen {

std::string lowerToLLVMIR(mlir::ModuleOp M) {
  mlir::MLIRContext *Ctx = M.getContext();

  // Ensure translation-to-LLVMIR hooks are registered on the context.
  mlir::registerBuiltinDialectTranslation(*Ctx);
  mlir::registerLLVMDialectTranslation(*Ctx);

  // Conversion pipeline: scf -> cf, then convert to llvm dialect.
  mlir::PassManager PM(Ctx);
  PM.addPass(mlir::createCanonicalizerPass());
  PM.addPass(mlir::createSCFToControlFlowPass());
  PM.addPass(mlir::createConvertControlFlowToLLVMPass());
  PM.addPass(mlir::createArithToLLVMConversionPass());
  PM.addPass(mlir::createConvertFuncToLLVMPass());
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(PM.run(M))) {
    std::cerr << "error: MLIR-to-LLVM conversion pipeline failed\n";
    return {};
  }

  // Translate to LLVM IR.
  llvm::LLVMContext LLVMCtx;
  auto LLVMModule = mlir::translateModuleToLLVMIR(M, LLVMCtx);
  if (!LLVMModule) {
    std::cerr << "error: translateModuleToLLVMIR failed\n";
    return {};
  }

  std::string Out;
  llvm::raw_string_ostream OS(Out);
  LLVMModule->print(OS, /*AAW=*/nullptr);
  OS.flush();
  return Out;
}

} // namespace mlirgen
} // namespace matlab
