// Runs the standard MLIR conversion pipeline to produce a module in the LLVM
// dialect, then translates it into LLVM IR textual form. Optionally attaches
// LLVM-dialect debug-info attributes (DICompileUnit / DIFile / DISubprogram)
// before translation so the resulting LLVM IR carries `!dbg` metadata that
// clang's downstream codegen turns into DWARF — making lldb / gdb able to
// step from the compiled binary back into the original `.m` source.

#include "matlab/MLIR/Passes/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <unordered_map>

namespace matlab {
namespace mlirgen {

namespace {

/* Walk every llvm.func in the module and stamp it with a
 * DISubprogram (and its enclosing DICompileUnit / DIFile). The
 * subprogram is attached as a fused location on the func op; the
 * MLIR-to-LLVM-IR translator (mlir::translateModuleToLLVMIR) reads
 * that attachment and emits real `!DISubprogram` metadata, then
 * threads `!DILocation` through every instruction whose location is
 * a `FileLineColLoc` parented by that fused scope.
 *
 * We pull the source file / line for each function from the first
 * `FileLineColLoc` we find on the function or anywhere inside its
 * body. Functions whose ops have no FileLineColLoc (purely-synthetic
 * helpers) get skipped — debug info is best-effort, the produced IR
 * still verifies and runs.
 *
 * Emission kind is LineTablesOnly: enough for source-level stepping
 * and per-line breakpoints, but skips the (much heavier) full DWARF
 * type-graph emission. Variable inspection (DW_TAG_variable etc.) is
 * orthogonal and not pursued today — DAP is the right surface for
 * MATLAB locals; lldb / gdb users get line tables.
 */
void attachDebugInfo(mlir::ModuleOp M) {
  mlir::MLIRContext *Ctx = M.getContext();

  /* Cache per-source-path so multi-file emit (when Sema starts
   * pulling in siblings) shares a single CU per file. */
  std::unordered_map<std::string, mlir::LLVM::DICompileUnitAttr> CUByFile;

  auto getCUForFile = [&](mlir::StringAttr filename) {
    auto It = CUByFile.find(filename.str());
    if (It != CUByFile.end()) return It->second;

    /* DIFileAttr wants the basename and the directory separately —
     * llvm::sys::path::filename / parent_path do the split portably. */
    llvm::StringRef Full = filename.getValue();
    llvm::StringRef BaseName = llvm::sys::path::filename(Full);
    llvm::StringRef Dir = llvm::sys::path::parent_path(Full);
    if (Dir.empty()) Dir = ".";
    auto File = mlir::LLVM::DIFileAttr::get(Ctx, BaseName, Dir);

    /* DistinctAttr is what makes the CU node distinct in the LLVM
     * `!metadata` graph; without it two CUs for the same file would
     * collapse and confuse the linker. */
    auto Id = mlir::DistinctAttr::create(mlir::UnitAttr::get(Ctx));

    /* No DWARF language code for MATLAB. DW_LANG_C is the closest
     * reasonable choice; lldb / gdb just need a recognisable language
     * to enable line stepping. */
    auto CU = mlir::LLVM::DICompileUnitAttr::get(
        Ctx, Id, llvm::dwarf::DW_LANG_C, File,
        mlir::StringAttr::get(Ctx, "matlabc"),
        /*isOptimized=*/false,
        mlir::LLVM::DIEmissionKind::LineTablesOnly,
        mlir::LLVM::DINameTableKind::Default,
        /*splitDebugFilename=*/mlir::StringAttr{});
    CUByFile[filename.str()] = CU;
    return CU;
  };

  /* Empty subroutine type — line-tables-only DWARF doesn't need the
   * type signature elaborated, and producing one for MATLAB's
   * dynamic-typed function shapes would be its own project. */
  auto EmptySubT = mlir::LLVM::DISubroutineTypeAttr::get(
      Ctx, llvm::dwarf::DW_CC_normal, /*types=*/{});

  M.walk([&](mlir::LLVM::LLVMFuncOp Fn) {
    /* Find a representative file/line for this function. Try the func
     * op's own location first; otherwise the first FileLineColLoc in
     * its body. */
    mlir::FileLineColLoc FLC =
        mlir::dyn_cast<mlir::FileLineColLoc>(Fn.getLoc());
    if (!FLC) {
      Fn.walk([&](mlir::Operation *Op) {
        if (auto L = mlir::dyn_cast<mlir::FileLineColLoc>(Op->getLoc())) {
          FLC = L;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
    }
    if (!FLC) return;

    auto CU = getCUForFile(FLC.getFilename());
    auto File = CU.getFile();

    auto NameAttr = Fn.getNameAttr();
    auto SpId = mlir::DistinctAttr::create(mlir::UnitAttr::get(Ctx));
    auto SP = mlir::LLVM::DISubprogramAttr::get(
        Ctx, SpId, CU, /*scope=*/CU, NameAttr, /*linkageName=*/NameAttr,
        File, /*line=*/FLC.getLine(), /*scopeLine=*/FLC.getLine(),
        mlir::LLVM::DISubprogramFlags::Definition,
        EmptySubT, /*retainedNodes=*/{}, /*annotations=*/{});

    /* Attach via a FusedLoc carrying the subprogram as metadata. The
     * translator looks for exactly this shape and uses it to root all
     * inner DILocations. */
    auto Fused = mlir::FusedLoc::get(Ctx, {Fn.getLoc()}, SP);
    Fn->setLoc(Fused);
  });
}

} // namespace

/* The MLIR LLVM-dialect translation rejects unknown parameter
 * attributes on llvm.func with "Unhandled parameter attribute
 * '<name>'". Our pipeline stamps `matlab.name` (and a few other
 * `matlab.*` arg attrs for fi-spec / array-shape metadata) on
 * func.func args/results so the EmitC / SystemVerilog backends can
 * render readable signatures. Those attrs ride the conversion to
 * llvm.func unchanged. The plain `-emit-llvm` translator tolerates
 * them, but the JIT (ExecutionEngine::create, used by `-repl` and
 * `-dap`) goes through a stricter path that errors on the same
 * input. Strip every `matlab.*` arg/result attr after the conversion
 * pipeline has run; the EmitC / SV emitters work off the source
 * func.func ops earlier, so this strip is invisible to them. */
void stripMatlabFuncAttrs(mlir::ModuleOp M) {
  M.walk([](mlir::LLVM::LLVMFuncOp Fn) {
    auto stripFromArrayAttr = [&](mlir::ArrayAttr Arr) -> mlir::ArrayAttr {
      if (!Arr) return Arr;
      llvm::SmallVector<mlir::Attribute> Filtered;
      Filtered.reserve(Arr.size());
      bool Changed = false;
      for (mlir::Attribute A : Arr) {
        auto Dict = mlir::dyn_cast<mlir::DictionaryAttr>(A);
        if (!Dict) {
          Filtered.push_back(A);
          continue;
        }
        llvm::SmallVector<mlir::NamedAttribute> Kept;
        Kept.reserve(Dict.size());
        for (mlir::NamedAttribute NA : Dict.getValue()) {
          if (NA.getName().getValue().starts_with("matlab.")) {
            Changed = true;
            continue;
          }
          Kept.push_back(NA);
        }
        if (Kept.size() == Dict.size())
          Filtered.push_back(A);
        else
          Filtered.push_back(mlir::DictionaryAttr::get(Fn.getContext(), Kept));
      }
      if (!Changed) return Arr;
      return mlir::ArrayAttr::get(Fn.getContext(), Filtered);
    };
    if (auto ArgAttrs = Fn.getArgAttrsAttr())
      Fn.setArgAttrsAttr(stripFromArrayAttr(ArgAttrs));
    if (auto ResAttrs = Fn.getResAttrsAttr())
      Fn.setResAttrsAttr(stripFromArrayAttr(ResAttrs));
  });
}

std::string lowerToLLVMIR(mlir::ModuleOp M, bool EmitDebugInfo) {
  mlir::MLIRContext *Ctx = M.getContext();

  // Ensure translation-to-LLVMIR hooks are registered on the context.
  mlir::registerBuiltinDialectTranslation(*Ctx);
  mlir::registerLLVMDialectTranslation(*Ctx);

  /* Drop uncalled classdef method bodies before the LLVM conversion.
   * Same posture as runReplInput (tools/matlabc/main.cpp): the
   * prelude-loaded classdef pulls every method into the TU, but
   * Sema only refines a method's param types when there's a call
   * site driving them.  An uncalled method body with `none`-typed
   * args survives the LowerScalars sweeps and trips the LLVM
   * translation step with `func.func` / `tensor<*xf64>` operands
   * that have no LLVM conversion.  Walking the SymbolTable to erase
   * dead class methods is safe — internal sibling calls keep the
   * transitive callee live, and non-classdef library functions
   * don't carry `matlab.class_name`. */
  {
    auto SymTbl = mlir::SymbolTable(M);
    bool Changed = true;
    while (Changed) {
      Changed = false;
      llvm::SmallVector<mlir::Operation *> Drop;
      M.walk([&](mlir::func::FuncOp F) {
        if (!F->hasAttr("matlab.class_name")) return;
        auto Sym = F.getSymNameAttr();
        if (auto Uses = SymTbl.getSymbolUses(Sym, M))
          if (Uses->empty()) Drop.push_back(F);
      });
      for (auto *Op : Drop) {
        SymTbl.erase(Op);
        Changed = true;
      }
    }
  }

  // Conversion pipeline: scf -> cf, then convert to llvm dialect.
  mlir::PassManager PM(Ctx);
  PM.addPass(mlir::createCanonicalizerPass());
  PM.addPass(mlir::createSCFToControlFlowPass());
  PM.addPass(mlir::createConvertControlFlowToLLVMPass());
  PM.addPass(mlir::createArithToLLVMConversionPass());
  PM.addPass(mlir::createConvertFuncToLLVMPass());
  PM.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (getenv("MATLABC_DUMP_PRE_PIPELINE")) {
    M.dump();
  }
  if (mlir::failed(PM.run(M))) {
    if (getenv("MATLABC_DUMP_ON_PIPELINE_FAIL")) {
      M.dump();
    }
    std::cerr << "error: MLIR-to-LLVM conversion pipeline failed\n";
    return {};
  }

  /* Stamp DI attrs after the conversion so we're working against
   * llvm.func ops (which the LLVMIR translator inspects), not the
   * original func.func ops that get rewritten by ConvertFuncToLLVMPass. */
  if (EmitDebugInfo)
    attachDebugInfo(M);

  /* Strip matlab.* arg/result attrs from llvm.func ops — see the
   * comment on stripMatlabFuncAttrs above. The plain `-emit-llvm`
   * path historically tolerated them but the JIT path doesn't, and
   * the conversion is needed unconditionally now to share the same
   * post-pipeline state across all three lowering callers. */
  stripMatlabFuncAttrs(M);

  if (getenv("MATLABC_DUMP_PRE_TRANSLATE")) {
    M.dump();
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
