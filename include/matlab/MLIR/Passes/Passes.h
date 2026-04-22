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

/// Convert every surviving `matlab.alloc` whose result type is a scalar
/// primitive (f32/f64/integer) into an `llvm.alloca`, rewriting all loads
/// and stores accordingly. Slot allocs with a `none` or non-scalar result
/// (cell/struct/tensor) are left for later phases. Must run after
/// SlotPromotion (which erases intra-block slots) and after type
/// refinement of user-defined functions.
bool runLowerScalarSlots(mlir::ModuleOp M);

/// Lowers every tensor-producing / tensor-consuming matlab.* op to an
/// llvm.call against the matrix runtime (matlab_zeros, matlab_add_mm,
/// matlab_transpose, matlab_disp_mat, ...). Tensor SSA values become
/// !llvm.ptr (pointer to a heap-allocated matlab_mat descriptor).
///
/// Also rewrites matlab.alloc whose result is a tensor type into
/// llvm.alloca of !llvm.ptr, with matlab.load/store converted accordingly,
/// so matrix-typed variables behave as pointer slots.
bool runLowerTensorOps(mlir::ModuleOp M);

/// Outlines each matlab.make_anon body into an llvm.func, replaces the
/// make_anon op with an llvm.mlir.addressof, and rewrites every
/// matlab.call_indirect through that handle into an llvm.call through a
/// function pointer. v1 scope: scalar f64 params, no captures of outer
/// values (anything more complex causes the pass to bail cleanly).
bool runLowerAnonCalls(mlir::ModuleOp M);

/// Lowers user-defined function calls: walks every matlab.call @fname(args)
/// in the module, and (a) retypes the target func.func's signature + entry
/// block arguments to match the call-site argument types when the original
/// signature was `none`-typed (single-site monomorphization), (b) rewrites
/// the matlab.call into a func.call with the refined types. Calls to
/// functions that the module doesn't declare are left in place.
///
/// Also updates the function's result types by inspecting its func.return
/// operand types, so user-defined functions return concrete values rather
/// than `none`. Callers that consumed the old `none` result have their uses
/// rewritten to the new typed value; consumers that can't accept the new
/// type (unlikely in our pipeline — the main consumers are matlab.*
/// unregistered ops that don't type-check) would error at verify time.
bool runLowerUserCalls(mlir::ModuleOp M);

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
