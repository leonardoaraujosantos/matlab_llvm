#pragma once

#include <string>

namespace mlir { class ModuleOp; }

namespace matlab {
namespace mlirgen {

/// Intra-block slot promotion: matlab.alloc / matlab.load / matlab.store
/// chains that live in a single block are promoted to SSA values.
/// Returns true if anything changed.
bool runSlotPromotion(mlir::ModuleOp M);

/// Partial lowering of scalar matlab.* ops to arith.* / arith.constant.
/// Only rewrites ops whose operands and results are primitive MLIR types
/// (f32/f64/integer/i1). Array / tensor ops are left for later phases.
bool runLowerScalarsToArith(mlir::ModuleOp M);

/// Lowering of I/O-related matlab.* ops to external runtime calls.
///
///   matlab.const_char "txt" : () -> tensor<1xNxi8>
///       -> llvm.mlir.global + llvm.mlir.addressof producing !llvm.ptr.
///
///   matlab.call_builtin @disp(char_arg)
///       -> llvm.call @matlab_disp_str(ptr, i64)
///
///   matlab.call_builtin @disp(f64_arg)
///       -> llvm.call @matlab_disp_f64(f64)
///
///   matlab.call_builtin @fprintf(fmt_char, f64_arg)
///       -> llvm.call @matlab_fprintf_f64(fmt_ptr, fmt_len, f64)
///
///   matlab.call_builtin @fprintf(fmt_char)
///       -> llvm.call @matlab_fprintf_str(fmt_ptr, fmt_len)
///
/// Additionally renames `func.func @script` to `@main` so the resulting
/// module is directly linkable into an executable.
bool runLowerIO(mlir::ModuleOp M);

/// Outlines each matlab.parfor body into a private func.func and replaces
/// the parfor op with an llvm.call to matlab_parfor_dispatch, which spawns
/// one pthread per iteration. v1 supports bodies that only reference the
/// induction variable, arith constants and module-level symbols
/// (llvm.mlir.global, llvm.mlir.addressof). Returns the number of outlined
/// parfor loops; loops that can't be outlined are left in place so the
/// later LLVM conversion surfaces them.
unsigned runOutlineParfor(mlir::ModuleOp M);

/// Convert the whole module down to the LLVM dialect and translate to an
/// LLVM IR textual module. Returns empty string on failure.
std::string lowerToLLVMIR(mlir::ModuleOp M);

} // namespace mlirgen
} // namespace matlab
