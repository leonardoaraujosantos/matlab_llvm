#pragma once

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace matlab {

class SourceManager;

namespace mlirgen {

/// Intra-block slot promotion: matlab.alloc / matlab.load / matlab.store
/// chains that live in a single block are promoted to SSA values.
/// Returns true if anything changed.
bool runSlotPromotion(mlir::ModuleOp M);

/// Partial lowering of scalar matlab.* ops to arith.* / arith.constant.
/// Only rewrites ops whose operands and results are primitive MLIR types
/// (f32/f64/integer/i1). Array / tensor ops are left for later phases.
bool runLowerScalarsToArith(mlir::ModuleOp M);

/// Rewrite Fixed-Point Designer (`fi`) ops into integer-shift sequences:
///   - `matlab.fi.const`  -> arith.constant (the stored integer)
///   - `matlab.fi.cast`   -> matlab_fi_quantize_{s,u} call (double → fi)
///                           or shift + saturate + truncate (fi → fi clamp)
///   - `matlab.add` / `matlab.sub` (fi-tagged) -> extend, align FLs by
///                          shift, integer add/sub, optional saturate
///   - `matlab.matmul` / `matlab.emul` (fi-tagged scalar) -> integer mul
///   - `matlab.neg` (fi-tagged) -> arith.subi 0
/// All other ops pass through unchanged. Runs after LowerScalarsToArith
/// (so non-fi scalar arith is already in arith.*) and before LowerSeqLoops
/// / LowerTensorOps. See docs/emit_fixed_point.md §7 for the rewrite
/// semantics.
bool runLowerFixedPoint(mlir::ModuleOp M);

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

/// Second-chance rewrite of matlab.call_indirect ops whose callee is an
/// llvm.mlir.addressof of a previously-outlined llvm.func. Intended to run
/// AFTER LowerTensorOps has retyped matrix operands (tensor -> ptr) so
/// that a call site with matrix captures can finally match its outlined
/// function's (ptr, f64, ...) signature.
bool runLowerAnonCallsPost(mlir::ModuleOp M);

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

/// Multi-callsite monomorphisation. When a user function is called with
/// distinct concrete arg signatures (e.g. sq(5) and sq([1 2 3])), clone
/// the function per signature and redirect each group of sites to its
/// specialisation. Runs AFTER LowerTensorOps so tensor-typed operands
/// have collapsed to !llvm.ptr — matrices of different shapes share
/// the ptr signature, so we only split when types genuinely differ
/// (f64 vs ptr). Also splits by call-site arity so functions called
/// with fewer args than declared get a per-arity clone with the
/// matching `nargin` value. Re-runs LowerUserCalls to retype the clones.
bool runMonomorphiseUserCalls(mlir::ModuleOp M);

/// Lower matlab.nargin / matlab.nargout placeholder ops to
/// arith.constant inside each enclosing func.func. The value comes
/// from the `matlab.nargin_value` / `matlab.nargout_value` attribute
/// on the function when present (set by the monomorphiser for
/// per-arity clones), falling back to the function's declared input
/// or result count.
bool runLowerNarginNargout(mlir::ModuleOp M);

/// Lowers sequential matlab.for (over a matlab.range) and matlab.while
/// into scf.while constructs so the MLIR conversion pipeline can finish
/// translation down to LLVM IR. Must run before LowerTensorOps (which
/// otherwise erases matlab.range) and after OutlineParfor.
bool runLowerSeqLoops(mlir::ModuleOp M);

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
///
/// When `EmitDebugInfo` is true, attach DICompileUnit / DIFile /
/// DISubprogram attributes so the resulting LLVM IR carries `!dbg`
/// metadata pointing back at the original `.m` source — i.e. clang + lldb /
/// gdb can step from the compiled binary into the user's `.m` file.
std::string lowerToLLVMIR(mlir::ModuleOp M, bool EmitDebugInfo = false);

/// Fold `scf.if` whose only effect is storing one of two externally-
/// defined SSA values into the same slot down to a single `arith.select`
/// plus one store. Chains nicely with Mem2RegLite so the slot disappears
/// when both arms are its only writers (classic `y = cond ? a : b;`).
/// Runs after LowerIO and before Mem2RegLite / EmitC.
bool runIfStoreToSelect(mlir::ModuleOp M);

/// Promote single-store `llvm.alloca` slots back to SSA so the EmitC
/// backend can emit them as ordinary C locals rather than a pointer-plus-
/// store-plus-load triple. Conservative: only fires on allocas whose
/// entire use set is one store (in the entry block, dominating all reads)
/// plus any number of plain loads. Runs after LowerIO and before EmitC
/// / LowerToLLVMIR.
bool runMem2RegLite(mlir::ModuleOp M);

/// Emit a self-contained C (or C++ when Cpp==true) source that reproduces
/// the semantics of the MLIR module. Expects the module to have been driven
/// through the same pipeline as -emit-llvm up to and including runLowerIO
/// (i.e. every matlab.* op has been replaced by arith / scf / func / llvm.*
/// ops and llvm.call sites against the matlab_* C runtime). The emitted
/// source is intended to be compiled and linked against runtime/matlab_runtime.c.
/// When NoLine==true, no `#line` directives are emitted (debug info is
/// lost — use for cleaner read-only output). When SM is non-null, MATLAB
/// `%` comments from the source are propagated into the generated source
/// as C++-style `//` comments above the corresponding statement. When
/// Doxygen==true, function-leading comment blocks are rendered as
/// `/** ... */` Doxygen blocks instead of `//` lines — useful when the
/// emitted C/C++ is consumed by codebases that extract API docs with
/// Doxygen. When CppAuto==true AND Cpp==true, call-result locals are
/// declared with `auto` rather than their explicit type — more concise
/// but loses the at-a-glance type hint. Returns empty string on failure.
std::string emitC(mlir::ModuleOp M, bool Cpp, bool NoLine = false,
                  bool Doxygen = false, bool CppAuto = false,
                  const matlab::SourceManager *SM = nullptr);

/// Emit a self-contained Python source that reproduces the semantics of the
/// MLIR module. Expects the module to have been driven through the same
/// pipeline as -emit-c up to and including runLowerIO / IfStoreToSelect /
/// Mem2RegLite. The emitted source imports `matlab_runtime` (a NumPy-backed
/// Python shim) and runs on CPython 3.10+. When NoLine==true, no line-
/// comment source markers are emitted. When SM is non-null, MATLAB `%`
/// comments from the source are propagated into the generated source as
/// `#` lines above the corresponding statement. Returns empty string on
/// failure.
std::string emitPython(mlir::ModuleOp M, bool NoLine = false,
                       const matlab::SourceManager *SM = nullptr);

/// Emit a self-contained TypeScript source that reproduces the semantics
/// of the MLIR module. Expects the same pipeline state as -emit-python.
/// The emitted file imports `matlab_runtime` (a numpy-ts-backed shim)
/// and `numpy_ts` (a minimal numpy-like API) and runs on Bun / `tsx` /
/// Node + `ts-node`. When NoLine==true, no line-comment source markers
/// are emitted. When SM is non-null, MATLAB `%` comments from the
/// source are propagated into the generated source as `//` lines above
/// the corresponding statement. Returns empty string on failure.
std::string emitTypeScript(mlir::ModuleOp M, bool NoLine = false,
                           const matlab::SourceManager *SM = nullptr);

/// Pattern info describing a canonical bounded for-loop shape that
/// `LowerSeqLoops` produces from `for i = init:end` (with optional
/// step). The same pattern is recognized by `HWLegalize` (Phase 2) and
/// `EmitSystemVerilog` so they agree on what to accept and what to
/// emit.
///
///   scf.while %iv = %init {
///     %cmp = arith.cmpf ole|oge, %iv, %end
///     scf.condition (%cmp) %iv : f64
///   } do {
///   ^bb0(%iv: f64):
///     ...body...
///     %next = arith.addf %iv, %step
///     scf.yield %next : f64
///   }
///
/// `IsDecreasing` is true when the comparison is `oge` (i.e. the loop
/// counts down from `init` to `end`). All three of `Init`, `End`,
/// `Step` are SSA values that — for Phase 2 — must be `arith.constant`
/// of `f64`. The caller checks the constant-ness; this struct just
/// records what was matched.
struct HWForLoopInfo {
  mlir::Value Init;
  mlir::Value End;
  mlir::Value Step;
  bool IsDecreasing = false;
  // The iv block-arg of the after-region. Used to verify that no body
  // op consumes the induction variable as a datapath value (Phase 2
  // restriction; relaxed in later phases).
  mlir::Value Iv;
};

/// Try to match the canonical for-loop pattern on an `scf.while`.
/// Returns true on success, populating Info. Phase 2 callers further
/// require that Info.Init / End / Step are `arith.constant`.
bool matchHWForLoop(mlir::Operation *WhileOp, HWForLoopInfo &Info);

/// Phase 4.5.4 — static fi-array lowering. Rewrites the canonical
///
///     v = fi(zeros(1, N), S, W, F);   % N constant
///     v(1) = ...;  v(2) = ...;        % constant-index writes
///     ... = v(1) + v(2) + ...;        % constant-index reads
///
/// pattern from runtime-call form (`matlab_mat_i64_zeros` +
/// `__subscript_store` + `matlab_mat_i64_subscript1_s`) into a
/// stack-allocated `llvm.alloca !llvm.array<N x iW>` with
/// `llvm.getelementptr` + `llvm.load` / `llvm.store` access.
/// Element width W is inferred from the typed-value operand of
/// every `__subscript_store` site (all must agree). Read sites
/// return i64 today; the rewrite truncates back to iW so the
/// subsequent fi-arithmetic chain stays integer-typed.
///
/// Phase 4.5.4 v1 scope:
///   - 1-D vectors only (rows = 1, cols = N constant)
///   - constant integer indices on both reads and writes
///   - element type uniform across all stores
///
/// Returns true on success (no diagnostic emitted on a non-match;
/// the runtime calls just stay in place and HWLegalize rejects
/// them downstream as before).
bool runLowerStaticFiArrays(mlir::ModuleOp M);

/// Pre-HWStateInfer normalization. The HDL Coder canonical
/// initializer is two separate guards:
///
///   if isempty(c)   c = INIT; end
///   if reset        c = INIT; end
///
/// but users frequently write the joined short-circuit form:
///
///   if isempty(c) || reset  c = INIT; end
///
/// which lowers to `matlab.short_or(isempty(idx), reset)` feeding
/// one scf.if. HWStateInfer's matcher requires the isempty result
/// to have exactly one use (a cmpf feeding the guard), so the
/// joined form fails legalization. This pass rewrites the joined
/// form into the canonical two-guard shape: clones the if-body,
/// drops the OR, and places one scf.if with the isempty operand
/// and another with the remaining operand. Both run the same body.
/// Returns true on success.
bool runSplitIsEmptyOr(mlir::ModuleOp M);

/// Phase 4.5.1 — slot-type inference for `matlab.alloc` ops still
/// typed `none` after the user-call iteration loop runs. Walks
/// every `matlab.alloc` whose result is `none`, looks at all
/// `matlab.store` ops whose addr operand is the alloc's result;
/// if every stored value's type agrees on the same scalar
/// primitive (integer or float), retypes the alloc's result and
/// every `matlab.load` reading from it. Idempotent.
///
/// LowerUserCalls' propagateScalarTypes does the same work, but
/// only for functions still reached via a `matlab.call` site —
/// once those collapse to `func.call`, the slot-retype logic
/// stops running. This pass picks up where that left off.
bool runRefineSlotTypes(mlir::ModuleOp M);

/// Phase 4.5.1 helper. Walks every `func.func` and patches its
/// declared result types from the body's `func.return` operand
/// types when the declared type was still `none`. Also refreshes
/// any `func.call` site whose result type now disagrees with the
/// callee's signature (creates a new typed `func.call` that
/// replaces the stale one).
///
/// Used both inline after the LowerScalarsToArith / LowerUserCalls
/// iteration loop (so the early verify check doesn't fail on
/// `make_handle("false") → arith.constant 1 : i1` rewrites that
/// haven't propagated to the function signature yet) and from the
/// SV pipeline as a final consistency pass before HWLegalize.
/// Returns true on success.
bool runRefineFuncSigs(mlir::ModuleOp M);

/// Phase 4.5.2 fixup. The Lowering pass places an
/// `unrealized_conversion_cast` on `scf.if` conditions whose source
/// type was `none` at lowering time (e.g. a load of a slot whose
/// type only refines after `LowerUserCalls`). After the
/// scalar-to-arith / user-call iteration loop has refined the
/// source to a concrete integer or float, this pass walks every
/// such cast and replaces it with the canonical `arith.cmpi ne,
/// src, 0` (integer) or `arith.cmpf one, src, 0.0` (float). Returns
/// true on success; emits an error per cast whose source still has
/// no concrete numeric type.
bool runRefineIfConds(mlir::ModuleOp M);

/// Phase 3 persistent-variable recognition. The MATLAB pattern:
///
///   persistent c;
///   if isempty(c)
///       c = <reset>;
///   end
///   ... user body that may read or write c via the runtime ABI ...
///   count = c;
///
/// lowers to a fixed runtime-call shape:
///
///   %ie = llvm.call @matlab_persistent_isempty(%idx) : (i32) -> f64
///   %c  = arith.cmpf one, %ie, 0.0
///   scf.if %c {
///     llvm.call @matlab_global_set_f64(%idx, %reset_init)  // reset value
///   }
///   ...
///   %v  = llvm.call @matlab_global_get_f64(%idx) : (i32) -> f64   // reads
///   ...
///   llvm.call @matlab_global_set_f64(%idx, %next_value)            // writes
///
/// `gatherHWPersistentState` recognizes this shape and returns one
/// HWPersistentInfo per persistent variable in `F`. Each persistent
/// becomes an inferable register in the emitted SystemVerilog.
struct HWPersistentInfo {
  // Integer index used by the runtime-call ABI (the 1st operand of
  // every isempty / get / set call site).
  int32_t Idx = 0;
  // User-visible variable name, taken from the `persistent_name`
  // string attr on each call site. Same across all sites.
  std::string Name;
  // The integer width and signedness of the register, inferred from
  // the typed value operand of each `_set_f64` site. Sites must
  // agree; mismatch is a hard error.
  unsigned Width = 0;
  bool Signed = true;
  // The reset-init value's MLIR Value, taken from the lone set call
  // inside the `isempty` then-region.
  mlir::Value ResetValue;
  // The scf.if op holding the `isempty` guard. The SV emitter skips
  // this op during always_comb body emission — the reset value is
  // routed into the always_ff's reset branch instead.
  mlir::Operation *IsEmptyGuard = nullptr;
  // Every `_get_f64` call site. The emitter renders the result of
  // each as a reference to the register's current-value signal.
  llvm::SmallVector<mlir::Operation *, 4> Gets;
  // Every `_set_f64` call site (excluding the one inside the
  // isempty guard). The emitter renders these as assignments to the
  // register's `_next` signal.
  llvm::SmallVector<mlir::Operation *, 4> Sets;
};

/// Walk a func.func and gather every recognized persistent variable.
/// Returns true if every persistent in `F` matched the canonical
/// shape; on failure, emits errors via mlir::emitError on the
/// offending op and returns false. `Out` is appended to.
bool gatherHWPersistentState(mlir::Operation *FuncOp,
                             llvm::SmallVectorImpl<HWPersistentInfo> &Out);

/// Synthesizability gate for the SystemVerilog backend (ASIC target).
/// Walks the post-LowerIO module and emits a source-located error for
/// every construct that cannot be mapped to inferable RTL — runtime
/// calls that survive lowering, recursion, dynamic-shape allocas,
/// `scf.while` (Phase 1 rejects all; later phases relax), function
/// values, `f64` arithmetic without an explicit fixed-point policy,
/// strings/cells/structs in datapath, and `inferred-latch` hazards
/// (an `always_comb`-shaped path where a result isn't fully assigned
/// on every branch).
///
/// Returns true if the module is synthesizable (no errors emitted).
/// Drives both `-check-synthesizable` (run-only-this-pass mode) and
/// `-emit-systemverilog` (gate-then-emit). See docs/emit_systemverilog.md
/// — "Synthesizability gate" for the per-phase coverage table.
bool runHWLegalize(mlir::ModuleOp M, const matlab::SourceManager *SM = nullptr);

/// Bit-width inference for the SystemVerilog backend. Walks the module
/// and verifies every SSA value carries a type that the SV emitter
/// can render: `i1` (→ `logic`), `i8/i16/i32/i64` (signed or unsigned
/// → `logic [W-1:0]`). Anything else (including `f64`) is rejected.
/// Phase 1 keeps the inference inline with verification — the
/// per-value `sv.type` attribute attachment is deferred to Phase 5
/// when fixed-point binary-point tracking needs it.
///
/// Returns true on success.
bool runHWBitWidthInfer(mlir::ModuleOp M,
                        const matlab::SourceManager *SM = nullptr);

/// Reset convention for the SystemVerilog backend. ASIC default is
/// `AsyncLow` (async-assert / sync-deassert active-low) — the typical
/// Synopsys flow convention. Sync variants are available for teams
/// that prefer sync-only reset trees. Stateless modules ignore this.
enum class HWResetKind { AsyncLow, SyncHigh, SyncLow };

/// Emit synthesizable SystemVerilog (ASIC target) from a module that
/// has been driven through the full lowering pipeline (same state as
/// `-emit-c` consumes — post-LowerIO, IfStoreToSelect, Mem2RegLite).
/// Produces one `module` per `func.func`. Combinational logic lands
/// in `always_comb`; persistent variables become `always_ff`-driven
/// registers with a `clk` + reset port pair added to the module's
/// signature. Returns empty string on failure.
std::string emitSystemVerilog(mlir::ModuleOp M,
                              const matlab::SourceManager *SM = nullptr,
                              HWResetKind Reset = HWResetKind::AsyncLow);

} // namespace mlirgen
} // namespace matlab
