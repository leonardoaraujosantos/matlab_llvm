# Examples — Run / Debug / REPL / DAP Status Report

Sweep date: **2026-06-02**.  matlabc built fresh from `main` (HEAD `6caf7dc`).

Scope: every `*.m` under `examples/` (376 files) exercised across three
execution axes and classified. This is the cross-axis companion to the
single-axis [`examples_status_report.md`](examples_status_report.md) (AOT-only).

| Axis | How it runs the example | Tool |
|---|---|---|
| **AOT (compile/run)** | `matlabc -emit-llvm` → `clang++` link vs `libMatlabRuntime` → run | `test/Examples/run_sweep.sh` |
| **DAP (whole-file JIT)** | `matlabc -dap` launches the file as one program, runs to `terminated` | `test/Debug/jit_parity_sweep.py` |
| **REPL (interactive, line-by-line)** | file body piped into `matlabc -repl` — each statement is its own JIT turn, values round-trip through the workspace struct | `test/Debug/repl_sweep.py` *(new this sweep)* |

The three axes share one parser/Sema/MLIR front; they differ only in **how
much of the program is in scope when each statement compiles**. AOT and DAP
compile the *whole file at once*; the interactive REPL compiles *one
statement at a time* and rehydrates earlier values from the workspace. That
single difference is the source of every divergence below.

---

## Summary

| Axis | In-scope | OK | Failing | Notes |
|---|---|---|---|---|
| **AOT** | 317 | **299** | 18 | all 18 are documented baseline entries — **0 regressions** |
| **DAP** | 269 | **269** | 0 | clean — full parity, nothing crashes/hangs/fails-to-launch |
| **REPL** | 269 | **226** | 43 | 23 CRASH + 20 ERROR — interactive incremental-compile divergence |

(Skip counts differ because the JIT axes additionally skip function-/classdef-only
files with no runnable top-level entry, symbolic/SymPP examples, interactive
`keyboard`/`input` demos, and missing-asset loaders — same rules in both JIT sweeps.)

**Headline:** compiled-and-run (AOT) and whole-file debug (DAP) are healthy.
The one thing "wrong" is the **interactive REPL**, which crashes or fails to
compile 43 examples that AOT and DAP run fine. The cause is structural, not a
regression — see below.

---

## Axis 1 — AOT compile/run: 299/317 OK, 0 regressions

All 18 failures are pre-listed in `test/Examples/known_failures.txt` and are
documented carve-outs, not new breakage:

| Cluster | Files | Class | Why |
|---|---|---|---|
| RL heavy training | `rl/pendulum_ddpg`, `pendulum_td3`, `pendulum_sac`, `cartpole_ppo`, `cartpole_trpo`, `countdown_grpo` | TIMEOUT | full multi-hundred-episode training runs past the 15 s/example budget; convergence is gated separately by `test/Run/rl_*.m` |
| PDE name=value | `pde/clamped_plate_pressure`, `poisson_disk`, `tuningfork_modal` | EMIT | parser gap: `femodel(AnalysisType="…")` name=value call syntax |
| GPU advanced | `gpu/benchmark_gpu_backend`, `run_validation_suite`, `test_gpuarray_stencil2d`, `test_parfor_gpu_batches`, `test_parfor_multigpu` | EMIT | deferred GPU Coder patterns + multi-file resolution (`docs/gpu_coder_roadmap.md`) |
| GPU function-only / Metal | `gpu/mandelbrot_gpu`, `test_gpuarray_arrayfun`, `test_gpuarray_axpy`, `test_gpuarray_gemm` | LINK | function-only files (no synthesized `main`) + Metal-host driver needing the Metal SDK |

Nothing that compiled+linked exited non-zero (RUNTIME = 0).

---

## Axis 2 — DAP whole-file JIT: 269/269 OK

`matlabc -dap` launches and runs every in-scope example to `terminated` with
no crash, hang, or launch failure. `jit_parity_known_issues.txt` is empty and
stays empty. This confirms the ReplMode + DebugMode JIT plumbing is at full
parity with AOT for *whole-file* execution — the state issues #77 / #105
closed.

---

## Axis 3 — Interactive REPL: 226/269 OK — the real "what's wrong"

Feeding a script into `matlabc -repl` line-by-line (as a user pasting it
would) diverges on **43 examples**. Two root causes, both downstream of
per-statement compilation:

> Caveat on the 226 "OK": the REPL is *lenient about runtime semantics* — an
> undefined name or a dimension mismatch prints nothing and still exits 0. So
> REPL-OK means "compiled + did not crash", **not** "numerically correct". The
> AOT lane remains the authority on run-to-exit-0; this axis adds only the
> interactive crash / compile-error signal. (This leniency is itself a minor
> finding — the REPL should surface those errors rather than swallow them.)

### 3a. CRASH (23) — handles/objects don't survive the cross-turn workspace round-trip

When a value produced in one REPL turn is consumed in a later turn, it is
serialized into the workspace struct and rehydrated on read. Two distinct
defects (see issue #116 for the code-cited root cause):

- **Anonymous function handles are never persisted at all.** The assignment
  `f = @(x) …` compiles to an *empty* `main` in ReplMode — `isLocalHandle`
  (`Lowering.cpp:347`) keeps the anon on the local-slot lane (correct for
  same-turn use, the `#77` fix) but that lane emits no workspace store, so
  `f` never enters the workspace (`whos` doesn't list it). A *named* handle
  (`g = @cos`) **does** emit `matlab_ws_set_handle` and survives — the
  divergence is anon-specific. On the next turn the absent `f` reads back as
  an empty matrix; a solver that invokes it as a closure jumps through a
  bogus pointer → SIGBUS (signal 10) / SIGSEGV (signal 11).
- **Table objects are persisted but untyped** (`whos` shows `Class ?`) and
  crash at a later use-site — a separate, smaller sub-bug (2 examples).

| Sub-cause | Examples | Signal |
|---|---|---|
| `h = @(…) …` defined, then **passed to a solver** in a later turn (`fminunc`/`fsolve`/`ga`/`patternsearch`/`surrogateopt`/`ode45`/`pdepe`/`nlmpc`/`greyest`/`ukf`) | `optim/*` (7), `globaloptim/*` (7), `ode_solver`, `heat_eq`, `mpc/pendulum_nlmpc`, `mpc/twin_rotor_nlmpc`, `ident/greybox_msd`, `ident/ukf_state_estimation`, `dlnet/dl_run_experiment` | 10 |
| `table`/`readtable`/`readmatrix` result round-trips through the workspace | `csv_stats`, `csv_table` | 10 / 11 |

This is the **same root class** the JIT-divergence work (#77/#105) chased — a
ReplMode `matlab_ws_get_*` read returns an untyped value that an AOT-shaped
path then mis-handles — but at *statement granularity*. The whole-file DAP
lane never hits it because the handle and its call site live in one
compilation unit, so the handle is never serialized.

Example (`optim/fminunc_rosenbrock.m`): `ros = @(x) …` then `r = fminunc(ros, [-1.2;1])` → RC 138 (SIGBUS).

### 3b. ERROR (20) — per-statement compile loses whole-file shape context

A single statement compiled in isolation can't see how a variable was
shaped earlier, so the AOT-shaped call-shape detectors back off and emit
`unsupported call shape` (or leave `unconverted matlab.* ops`). Whole-file
AOT/DAP resolve these because the defining op is visible.

| Failing builtin (per-statement) | Examples |
|---|---|
| `softmax`, `sigmoid`, `mse` (dlnet activations on rehydrated dlarray) | `dlnet/dl_cnn_bn`, `dl_cnn_classifier`, `dl_gru_sequence`, `dl_lstm_sequence`, `dl_siamese` |
| `__subscript_store` (5-arg N-D store on a rehydrated array) | `dlnet/dl_imagedatastore`, `dl_pretrained_inference`, `dl_real_image_pipeline`, `dlnet/matmul3_batched` |
| `setOccupancy` (3-arg) | `navigation/nav_rrt_plan`, `nav_mcl_localize` |
| `mpcmove`, `irf`, `summary` | `quadrotor/quadrotor_pid_mpc`, `econ/var_macro`, `finance/using_timetables_in_finance` |
| residual `matlab.subscript`/`matlab.matmul` left unconverted | `bank_account`, `persistent_counter`, `fi_fir_filter`, `images/channel_split`, `images/read_write_png`, `dlnet/dl_mha_forward` |

---

## What this means / suggested fixes

1. **AOT and DAP are healthy.** The compile/run and whole-file-debug stories
   need no work beyond the already-tracked carve-outs (RL timeouts, PDE
   name=value parser gap, GPU deferrals).

2. **The interactive REPL is the gap.** None of the 43 REPL failures is a
   regression vs the committed baselines — there has simply never been a
   per-example interactive-REPL gate, so this divergence was invisible.
   `test/Debug/repl_sweep.py` (added this sweep) makes it measurable.

3. **Highest-value REPL fix — persist anonymous handles across turns
   (✅ LANDED — fixes 13 of the 23 crashes).** For an `isLocalHandle`
   *capture-free* anon assigned in ReplMode, the assignment now *also* emits
   `matlab_ws_set_handle` (`Lowering.cpp`, in the `RhsIsHandle` block) — the
   binding still stays on the local-slot lane for same-turn calls, but the
   function pointer is additionally written to the workspace so a later
   turn's `Binding::IsHandle` / `matlab_ws_get_handle` (kind=13) read
   recovers it (the exact round-trip *named* handles already use; the
   per-session `g_ReplEngines` vector keeps the JIT'd anon code resident so
   the pointer stays valid). This took the REPL lane from 226→238 OK and
   crashes from 23→10. Guarded by `test/Repl/run_tests.sh`
   (`xturn_anon_handle_scalar`, `xturn_anon_handle_to_solver`).

   **Follow-ups:**
   - **Direct cross-turn handle call with a matrix arg (✅ LANDED — #119).**
     The kind=13 call trampolines were scalar-only; calling a recovered handle
     directly with a vector (`rastrigin(xlocal)`) lowered to a *subscript on
     the code pointer* — SIGSEGV, and when the garbage matrix dimension read
     off the code pointer was large, a multi-GB runaway allocation. Fixed by
     carrying the handle's return-kind in the kind=13 signature side-channel
     (`matlab_ws_set/get_handle_sig`), a Resolver hook that stamps
     `Binding::HandleRetKind`, and matrix trampolines
     (`matlab_call_handle_m{1,2}` → scalar, `_mm{1,2}` → matrix). Cleared the
     6 `globaloptim/*` crashes plus `ode_solver` and `lsqnonlin_curvefit`
     (REPL lane 238→246 OK, crashes 10→2). Residual corner: a matrix-returning
     anon whose body Sema can't type (e.g. `@(x) reshape(x,2,2)`) would
     misdispatch — noted in #119. Guarded by `xturn_anon_handle_matrix_arg_*`.
   - **Scalar-from-workspace-matrix → `tensor<*xi1>` in `if` (#120).** A 1×1
     builtin result round-tripped as a matrix stays tensor-typed; a scalar
     comparison on it fails `scf.if` verification. ReplMode-specific (AOT is
     fine). Surfaced in `optim/blade_pitch_opt` once the preceding crash was
     fixed.
   - **Captured anons** (`@(s) M*s`) need the closure environment serialized,
     not just the function pointer — the other open piece of #116.
   - **`table` round-trips** (`csv_*`, 2 crashes) — stored untyped, crash at
     use-site (#116).

4. **Second REPL fix — shape-carrying workspace reads (fixes ~20 errors).**
   Persist the inferred class/shape of a workspace variable alongside its
   value so a later statement's call-shape detector (softmax/`__subscript_store`/
   `setOccupancy`/…) sees the same type it would in whole-file compilation.
   Today the read returns a bare value and the detector backs off to
   `unsupported call shape`.

Both fixes converge on one mechanism: **the ReplMode workspace round-trip
must be kind- and shape-preserving.** That is the through-line of every
interactive-REPL divergence in this sweep.

---

## Reproducing

```bash
ninja -C build matlabc

# AOT (compile + link + run, gated vs baseline)
bash test/Examples/run_sweep.sh build/matlabc

# DAP whole-file JIT parity
python3 test/Debug/jit_parity_sweep.py "$PWD/build/matlabc" --timeout 20

# Interactive REPL (line-by-line) — NOTE: absolute matlabc path required
python3 test/Debug/repl_sweep.py "$PWD/build/matlabc" --timeout 20
```

---

## Changelog

- **2026-06-03 — #181 (✅ fixed).** `intersect` / `setdiff` / `union` of two
  **row** vectors returned **column** vectors — same orientation bug as #179
  but in the set operations. MATLAB returns a row vector only when *both*
  inputs are row vectors; otherwise a column. The shared `set_op` helper in
  `matlab_runtime.cpp` hard-coded `make_mat(u, 1)`, and the Python/TS shims
  reshaped to a column. All now branch on `both_row` (both inputs `rows == 1`).
  Mixed / column inputs still yield a column. Regression
  `test/Run/regress_setop_row_shape.m` (both-row → row, column/mixed → column;
  verified to produce `2x1` for the both-row case without the fix; runs on all
  backends). `test/Run/math_sort_set.m` expectations updated accordingly.

- **2026-06-03 — #179 (✅ fixed).** `unique()` of a **row** vector returned a
  **column** vector — orientation was not preserved. MATLAB returns a row
  vector for a row-vector input (and a column for a column vector or matrix);
  `sort` already preserved orientation, so the divergence was silent (wrong
  shape, right values). `matlab_unique` hard-coded `mat_alloc(u, 1)` with a
  comment incorrectly asserting "MATLAB's default is column for all unique()
  results"; the Python/TS shims likewise reshaped to a column. All three now
  branch on input orientation (`rows == 1` → `1 x u` row, else `u x 1` column).
  Regression `test/Run/regress_unique_row_shape.m` (row → `1x3`, column → `3x1`,
  matrix → `3x1`; verified to produce `3x1` for the row case without the fix;
  runs on all backends).

- **2026-06-03 — #177 (✅ fixed).** `numel(s)` / `length(s)` on a struct
  returned garbage — the struct ptr was read as a `matlab_mat`. The
  numel/length lowering had a cell special-case but no struct one. Added a
  scalar-struct case (`IsStruct` / `StructInitialised` / `StructBindings`,
  excluding struct arrays) that yields the constant `1`. Cell/matrix numel
  and struct arrays unaffected. Regression `test/Run/regress_numel_struct.m`
  (scalar + nested struct, guard use, cell/matrix; verified garbage without
  the fix; runs on all backends).

- **2026-06-03 — #175 (✅ fixed).** `end` inside cell brace-indexing
  (`c{end}`, `c{end-1}`) was left as an unconverted `matlab.end` — the cell
  read lowered its index without a `SubscriptCtx` base, and a cell `end` must
  resolve to `matlab_cell_numel` (not `matlab_end_of_dim`, which misreads the
  cell descriptor as a matrix). The cell read now pushes a cell-numel sentinel
  (`SubscriptCtx` dim `-1`) and the EndExpr lowering routes `-1` to
  `matlab_cell_numel`. Also extended the `CellMatElems` detection (tracking the
  cell element count) so `c{end}` of a matrix element picks `get_mat`.
  Regression `test/Run/regress_cell_end_index.m` (scalar `c{end}`/`end-1`,
  matrix element via `end`, downstream use; verified unconverted without the
  fix; runs on all backends).

- **2026-06-03 — #173 (✅ fixed).** `2.^x` (a digit immediately before `.^`)
  was mis-lexed as the float `2.` plus matrix-power `^` (→ unconverted
  matlab.matpow for a vector RHS), instead of `2 .^ x` (element-wise power).
  `Lexer::lexNumber` no longer consumes a trailing `.` when the next char is a
  dotted-operator char (`^ * / \ '`); and scalar `.^` (epow of two f64s) now
  routes to `matlab_pow_scalar` (mirroring the matpow scalar case) + a
  `pow_scalar` shim. Ordinary floats (`2.5`, `3.`, `2.*`/`2./`) unaffected.
  Regression `test/Run/regress_dotop_after_digit.m` (scalar `.^`, scalar`.^`vec,
  `2.*`/`2./`, float lexing, vec forms; verified unconverted ops without the
  fix; runs on all backends).

- **2026-06-03 — #169 (✅ fixed).** `num2str` of a vector/matrix failed with
  "unsupported call shape" — only the scalar f64 form was wired. Added a
  ptr-operand path → `matlab_num2str_mat` ("%g" per element, two spaces
  between row elements, newline between rows) + the python/ts shims. Scalar
  `num2str` unaffected. Regression `test/Run/regress_num2str_matrix.m`
  (row vector, decimals, char-concat label idiom; verified `unsupported call
  shape` without the fix; runs on all backends).
- **2026-06-03 — #171 (✅ fixed).** `mod` / `rem` of a vector/matrix failed
  with "unsupported call shape" — only the two-scalar form was wired. Added
  element-wise `_mm` / `_ms` / `_sm` variants (each element through the scalar
  `matlab_mod_s` / `matlab_rem_s`, so the MATLAB sign rules — mod follows the
  divisor, rem the dividend — are preserved) via an `ELT_BINARY` macro, the
  `pp`/`pf`/`fp` pde_table entries, and the python/ts shims. Two-scalar
  mod/rem unaffected. Regression `test/Run/regress_mod_rem_matrix.m`
  (vec/scalar, scalar/vec, vec/vec, negative sign rules; verified `unsupported
  call shape` without the fix).

- **2026-06-03 — #165 (✅ fixed).** Indexing a vector with a **same-length**
  index list returned the original order — `v(end:-1:1)` (reverse) and
  `v([3 2 1])` (reorder) both gave `[v]` unchanged. `matlab_slice1` (and the
  store path) treated any same-shape index as a logical mask. A genuine mask
  only holds 0/1, so a same-shape index with a value outside {0,1} is now an
  index list (gather/scatter in index order); all-0/1 same-shape stays a mask.
  New `idx_looks_like_mask` guard in `matlab_slice1` /
  `matlab_slice_store1[_scalar]` + the python/ts shims. Regression
  `test/Run/regress_reverse_index.m` (element-wise scalar disps; verified to
  return original order without the fix). Mask reads/stores (`v(v>2)`) and
  ascending/strided indexing unaffected.
- **2026-06-03 — #167 (✅ fixed).** `sort(x, 'ascend'|'descend')` (the 2-arg
  direction form) failed with "unsupported call shape" — only 1-arg `sort(x)`
  was wired. Refactored the ascending sort into a direction-parameterised core
  (`matlab_sort_impl`) and added `matlab_sort_dir(A, dir)` (sorts asc/desc by
  the first char of `dir`), wired via the string-arg table that materialises
  the direction `const_char` to a `matlab_string*`; added `sort_dir` to the
  python/ts shims. 1-arg `sort` / `[s,i]=sort` unaffected. Regression
  `test/Run/regress_sort_direction.m` (element-wise scalar disps; verified
  `unsupported call shape` without the fix; runs on all backends).

- **2026-06-03 — #152 (✅ fixed).** A scalar logical / comparison result
  displayed as `-1` instead of `1`: `disp(5>0)`, `disp(1|0)`, `disp(~0)` all
  printed `-1`. The LowerIO scalar-disp path widened an integer disp arg with
  `SIToFP`, switching to `UIToFP` only on an explicit `matlab.unsigned` tag —
  which an **i1 logical never carries**, so its `true` bit sign-extended to
  `-1.0`. An i1 is always a 0/1 logical, so it now zero-extends (`UIToFP`)
  unconditionally; wider ints keep the tag heuristic. Regression
  `test/Run/regress_logical_scalar_disp.m` (logical + comparison ops; verified
  `-1` without the fix; runs on all backends). The related case of a logical
  result *used in arithmetic* (`x=1|0; x+10` → unconverted `matlab.add`) is a
  separate lowering gap, filed as #161.
- **2026-06-03 — #161 (✅ fixed).** A scalar logical / comparison result (an
  i1) used in arithmetic failed to lower — `x=1|0; x+10`, `(5>0)*3`, `(5>0)/2`
  left the `matlab.add`/`sub`/`matmul`/`matdiv` unconverted (i1 operand
  mismatched the f64 other operand). MATLAB promotes a logical to double in
  arithmetic, so a new `promoteLogicalArith` helper widens an i1 operand to
  f64 (UIToFP — 0/1) when the other operand/result is float or both are
  logical, applied in `BinArithToArith` (+/-/.*/./), `ScalarMatMulToMulf`
  (`*`), and `ScalarMatDivToDivf` (`/`). Plain numeric arithmetic unaffected.
  Regression `test/Run/regress_logical_in_arithmetic.m` (verified unconverted
  ops without the fix; runs on all backends). Companion to #152 (the disp
  analog).
- **2026-06-02 — #147 (✅ fixed).** `isequal` on two **string** operands
  returned `0` even when equal: `matlab_isequal` reads `rows`/`cols`/`data`
  off its args as `matlab_mat*`, but a `matlab_string` has a different layout,
  so the comparison mis-read the strings. The frontend now routes a
  both-string `isequal` (detected via `isStringExpr`) to the `strcmp` path
  (#146 — length + byte compare); non-string `isequal` still uses
  `matlab_isequal`. Regression `test/Run/regress_isequal_strings.m` (strings,
  string vars, numeric arrays; verified to return `0` for equal strings
  without the fix; runs on all backends). The two-scalar form
  `isequal(5,5)` is a separate pre-existing `ff`-shape gap (#155).
- **2026-06-02 — #155 (✅ fixed).** `isequal(a,b)` with two scalar args failed
  (`unsupported call shape`): the pde_table had `isequal` only as `pp` (two
  matrix ptrs). Added an `ff` (two-scalar) entry → `matlab_isequal_2s`
  (returns a bare f64 0/1; isequal of scalars is genuinely scalar, so unlike
  max/min #153 it doesn't need a 1×1 ptr) + python/ts shims. Matrix `isequal`
  unaffected. Regression `test/Run/regress_isequal_two_scalars.m` (verified
  `unsupported call shape` without the fix; runs on all backends).

- **2026-06-02 — #156 / #157 (✅ fixed).** `disp(struct)` and `disp(cell)`
  SIGSEGV'd on **all** backends (AOT/-repl/-dap, rc 139): disp routed the
  struct/cell pointer to the polymorphic `matlab_disp_mat_f64`, which read it
  as a matrix descriptor (garbage rows/cols/data). The frontend now detects a
  struct- or cell-bound disp argument (`IsStruct`/`StructInitialised`/
  `StructBindings`/`CellBindings`) and routes it to new `matlab_disp_struct` /
  `matlab_disp_cell` runtime entries (registered in LowerTensorOps as void
  one-ptr calls) that print a field / element listing — `name: value`, with
  `[RxC double]` summaries for matrix members; not byte-exact to MATLAB but
  crash-free and deterministic. Plain numeric `disp` is unaffected (and was
  never broken — #157's "any disp" was the -dap manifestation of this same
  `matlab_disp_mat_f64` crash on a non-matrix). Regression
  `test/Run/regress_disp_struct_cell.m` (verified rc 139 without the fix).
  emit-python/typescript skipped (those shims lack the new entries / format
  differently; validated exactly on LLVM/C/C++).
- **2026-06-02 — #148 (✅ fixed).** An F64-returning string predicate
  (`contains` / `startsWith` / `endsWith` / `strcmp` / ...) used **directly**
  as an `if`/`while` condition failed to lower (`unrealized_conversion_cast`
  on llvm; "unsupported op" on emit-c/cpp/python/ts). These builtins are
  `none`-typed at MIR-to-MLIR lowering, so `fixupIfCond` leaves a
  verifier-placeholder cast on the `scf.if`; the result only refines to f64
  in the LowerTensorOps loop, and `runRefineIfConds` (which rewrites the
  placeholder to `arith.cmpf one, 0.0`) was only wired into the SV-emit
  pipeline. Added it to the shared emit full-pipeline (after the LowerTensorOps
  convergence), to `runJitSoftwareLowering` (JIT/REPL/-dap), and to
  `lowerToLLVMIR` (AOT). Regression `test/Run/regress_if_string_predicate.m`
  (verified to fail on both llvm and python without the fix; runs on all
  backends). Workaround previously was `if contains(...) == 1`.
- **2026-06-02 — #153 (✅ fixed).** `max(a,b)` / `min(a,b)` with two scalar
  args failed (`unsupported call shape`). The `LowerTensorOps` pde_table had
  `p` (reduction), `pp` (two matrices), `ppf` (dim) shapes for max/min but no
  `ff` (two scalars) or `pf`/`fp` (matrix-scalar broadcast). Added all three:
  the two-scalar form returns a **1×1 matrix** (matching the frontend's ptr
  typing for max/min, so `max(a,b)+c` and `max(max(1,2),3)` flow through the
  existing matrix paths — a bare-double return broke those), plus
  `matlab_max_2s/_ms/_sm`, `matlab_min_2s/_ms/_sm` runtime fns and python/ts
  shims. Reduction / element-wise / dim forms unaffected. Regression
  `test/Run/regress_max_min_two_scalars.m` (verified `unsupported call shape`
  without the fix).
- **2026-06-02 — #144 (✅ fixed).** Logical-mask indexed **assignment**
  (`v(v>2) = 0`) was wrong on all backends — it collapsed onto element 1
  (`[0 2 3 4]` instead of `[1 2 0 0]`). `matlab_slice_store1[_scalar]`
  interpreted the mask's 0/1 entries as literal linear indices (`0 → lin -1`,
  `1 → lin 0`). The mask *read* path (`matlab_slice1`) already treats an index
  the **same shape as A** as a logical mask; the store path didn't. Fixed by
  mirroring that same-shape heuristic (and column-major traversal) in both
  store entries and in the python/ts shims. Numeric-index assignment and mask
  reads are unaffected. Regression `test/Run/regress_logical_mask_store.m`
  (reduced to `sum()` scalars; verified to produce `9/18/9/19/...` without the
  fix). The explicit-`logical(...)`-variable index path (`__subscript_store`
  compile error) is a separate follow-up noted on #144.
- **2026-06-02 — #146 (✅ fixed).** `strcmp` / `strcmpi` were unimplemented
  (`undefined name`), so any program using them failed to compile. Wired them
  into the existing string-predicate family: added to the Resolver builtin
  list, extended the `contains`/`startsWith` lowering arm (2 string operands →
  F64), and added `matlab_strcmp` / `matlab_strcmpi` runtime fns (return **1.0
  if equal**, MATLAB sense — opposite of C; strcmpi case-insensitive) plus the
  python/ts shims. Regression `test/Run/regress_strcmp.m` (verified to fail
  with `undefined name 'strcmp'` without the fix; runs on all backends — no
  disp-formatting divergence). Two related pre-existing bugs found and filed
  separately: `isequal` returns 0 for equal strings (#147), and
  `if <string-predicate>(a,b)` directly as a condition fails to lower (#148,
  family-wide; workaround `== 1`). Single-quoted char-array args remain the
  same family-wide limitation.
- **2026-06-02 — #136 (✅ fixed).** `end` in single-subscript indexing resolved
  to `size(base, dim)` with `dim` = the 1-based subscript position, so a lone
  subscript used `dim=1` (row count). For a row vector that is `1`, not `numel`,
  so `v(end)` returned the **first** element instead of the last (and `v(end+1)`
  pointed at index 2). The lowering now pushes a sentinel `dim=0` for
  single-subscript indexing (both the store and load paths in `Lowering.cpp`),
  and `matlab_end_of_dim` maps `dim==0` → `matlab_numel` (mirrored in the
  Python/TS shims). Multi-subscript `A(end,end)` is unchanged (per-dimension).
  Together with #135 this makes the `v(end+1)=x` append idiom work. Regression
  `test/Run/regress_end_single_subscript.m` (verified to print first-element /
  garbage without the fix, across all backends).
- **2026-06-02 — #142 (✅ fixed).** A cell literal with a **matrix** element
  read back `0` on a brace index: `c = {1, 5, [1 2 3]}; disp(c{3})` printed `0`
  (and `sum(c{3})` failed to compile). The `CellIndex` read lowering picks the
  runtime entry from the Sema result type of `c{k}`, but Sema can't carry
  per-element cell types, so it defaulted to `matlab_cell_get_f64`, which
  returns `0` for a >1-element matrix slot (the stored `matlab_mat*` can't be
  unboxed to a scalar). The matrix is *stored* correctly (`cell_set_mat`,
  kind=1) — only read dispatch was wrong. Fixed by recording which cell-literal
  element indices are matrix/string-stored (`CellMatElems`, mirroring
  `MatStructFields`) and forcing `matlab_cell_get_mat` for a **constant-index**
  read of those slots; `matlab_cell_get_mat` is already kind-aware (boxes
  scalars, converts strings). Conservative — only provably-ptr elements
  (matrix literal / range / char / string / nested cell) are flagged, so a
  scalar slot is never mis-routed. Regression `test/Run/regress_cell_matrix_elem.m`
  (fails to compile without the fix — `sum(e{2})` sees a scalar). Variable-index
  reads and name-bound matrix elements remain follow-ups (tracked on #142).
- **2026-06-02 — #122 (✅ fixed).** `pde/poisson_disk.m` SIGSEGV under `-dap`.
  `matlab_pde_solve_femodel` read `MaterialProperties` via
  `matlab_struct_get_mat`, which returns a non-NULL **empty matrix** for a
  missing field; that empty matrix was reinterpreted as a `matlab_struct` and
  `struct_find_field` walked its garbage `nfields`/`names` → crash. The model
  only reached this structural fallback because its `Mesh` round-tripped empty
  under the `-dap` worker (~20% of runs). Fixed by gating `props` on
  `field_holds_struct` (runtime_pde.cpp). Deterministic regression in
  `test/Runtime/test_pde.c` (`test_femodel_missing_material_props`, verified
  to SIGSEGV without the fix). The intermittent empty-`Mesh` round-trip itself
  (silent wrong result, no longer a crash) is tracked as **#124**.
- **2026-06-02 — found while fixing #122: #123** (`pde/tuningfork_modal.m`
  SIGSEGVs via AOT — a #117 name=value `femodel` ctor builds a corrupt model;
  re-baselined in `known_failures.txt`) and **#124** (the `-dap` empty-`Mesh`
  round-trip race).
- **2026-06-02 — #116 table round-trip (✅ fixed).** The last 2 REPL crashes
  (`csv_stats`, `csv_table`). A `table` (`readtable`/`table(...)`) stored to the
  ReplMode workspace as kind=6 read back **untyped** on a later turn — the
  Resolver had no kind=6 case, so the binding was never re-stamped as a table.
  A cross-turn `height(T)`/`width(T)` then hit "unsupported call shape" and
  `disp(T)`/`T.col` crashed. (The workspace *read* already returned the table
  ptr via `matlab_struct_get_mat`'s kind=6 pass-through; only the binding's
  type was lost.) Fixed by adding `Binding::IsTable`, stamping it on the
  Resolver kind=6 lookup, and an `isTableBinding` helper so the Lowering
  dispatch sites (`height`/`width`/`T.col`/`disp`) treat a cross-turn
  `IsTable` binding as a table — same kind-preserving pattern as the #118/#119
  handle fix. Deterministic regression `test/Repl/run_tests.sh`
  (`xturn_table_height_width`), verified to fail without the fix. With this the
  REPL example lane reaches **248/269** (the residual non-OK are the per-statement
  shape ERRORs — separate #116 sub-bug — and the tracked PDE crashes).
- **2026-06-02 — #120 (✅ fixed).** A matrix-valued comparison used directly as
  an `if`/`elseif`/`while` condition (`if abs(v) < tol`) produced a
  `tensor<*xi1>` that `scf.if` rejected (`operand #0 must be 1-bit signless
  integer`). The condition was never reduced to MATLAB's "true iff every
  element is true". Not actually REPL-specific — it reproduces on the AOT
  `-emit-mlir` lane too whenever a comparison operand keeps its Sema array
  type (e.g. `abs(M)` returns `tensor<*xf64>`, so `< c` yields `tensor<*xi1>`;
  a workspace-backed 1×1 from `fminunc` hit the same path). Fixed by extending
  `Lowerer::fixupIfCond` to wrap a tensor-typed condition in `matlab_mat_truth`
  (the same reduction the matrix-pointer path already used; `LowerTensorOps::
  rewriteMatTruth` lowers it once the producing comparison becomes a
  `matlab_mat*`). Deterministic regression `test/Run/regress_matrix_if_cond.m`
  (+`.stdout`), verified to fail (`scf.if` tensor error) without the fix.
- **2026-06-02 — #123 (✅ root-cause SIGSEGV fixed).** `solve()` crashed when
  the model's `Geometry` field held a STRING path (`femodel(Geometry="fork.stl")`
  stores the path; the STL is never imported into a geometry struct). The
  rows/cols heuristic in `field_holds_struct` (runtime_pde.cpp) misread the
  matlab_string's `len` word as a struct pointer → `struct_find_field` walked
  it as `nfields`/`names` → SIGSEGV. Fixed by making `field_holds_struct`
  **kind-aware** (new `matlab_struct_field_kind`; accept only struct kinds
  1/2/12), plus NULL / NULL-name guards in `struct_find_field`. Deterministic
  regression `test/Runtime/test_pde.c` (`test_solve_string_geometry`), verified
  to SIGSEGV without the fix. `tuningfork_modal` still can't *complete* (its
  `fixtures/TuningFork.stl` asset isn't shipped, so `solve` returns empty and a
  later `RF.ModeShapes.Magnitude(:,7)` hits the empty-mat-as-obj crash **#128**),
  so it stays baselined; the original solve-path SIGSEGV is gone.
- **2026-06-02 — found while fixing #123: #128** (a property read off an
  empty/missing-field value — `RF.ModeShapes.Magnitude` — reinterprets the
  empty matrix as an obj in `matlab_obj_get_mat` → SIGSEGV; the obj-side of the
  empty-mat-as-struct class #122 closed on the struct side).
- **2026-06-02 — #128 (✅ fixed).** A property read off an empty/missing-field
  value — e.g. `RF.ModeShapes.Magnitude` where `RF.ModeShapes` is absent —
  SIGSEGV'd: `matlab_struct_get_mat` returns a non-NULL empty `matlab_mat`
  (rows==0, cols==0) for a missing field, and `matlab_obj_get_mat` then walked
  that empty matrix as an obj — its zero `rows` word lands at the struct
  `names` pointer offset, so `struct_find_field` dereferenced
  `((char**)NULL)[i]` at a near-NULL address. The obj-side of the
  empty-mat-as-struct class #122 closed on the PDE-struct side. Fixed by
  guarding `matlab_obj_get_mat`: an input whose `rows`/`cols` words are both 0
  (the empty-matrix sentinel; a real obj has non-NULL `names`/`kinds` there)
  returns empty instead of being walked. Deterministic regression
  `test/Runtime/test_struct_cell.c::test_obj_get_mat_on_empty`, verified to
  SIGSEGV without the fix.
- **2026-06-02 — #131 (✅ fixed).** A struct created/mutated by a field
  assignment (`s.x = v`) wasn't persisted to the ReplMode workspace, so a
  later REPL turn read an empty `s` and `s.x` came back `0` (silent wrong
  value — not in #116's enumerated crash/error set, and AOT/`-dap`/same-turn
  `-repl` are all correct). The field store wrote only into `s`'s local struct
  slot and emitted no `matlab_ws_set_struct`. Fixed in `Lowering.cpp` by
  persisting the (plain-struct) base to the workspace after a FieldAccess-LHS
  store in ReplMode — same store-side round-trip pattern as the anon-handle
  (#118) and table (#127) fixes. Deterministic regression
  `test/Repl/run_tests.sh` (`xturn_struct_field`, `xturn_struct_field_nested`),
  verified to read `0` without the fix. While investigating, also confirmed
  **#124 is deterministic, not a race** — `-dap` deterministically yields
  `FEM u(0) = 0.000000` (18/18) vs `-repl` `0.248873`; the issue was updated.
- **2026-06-02 — #133 (✅ fixed).** A struct array built by indexed
  field-assignment (`a(i).x = v`) wasn't persisted to the ReplMode workspace,
  so a cross-turn `a(i).x` / `length(a)` read an empty array (silent wrong
  value; AOT/same-turn correct). The array lived only in a local slot —
  `matlab_ws_set_struct_arr` / `_get_struct_arr` were referenced in a comment
  but never implemented. Fixed by adding the workspace ABI: new kind=14 +
  `matlab_ws_set_struct_arr` (read via `matlab_struct_get_mat`'s kind=14
  pass-through), `Binding::IsStructArray` stamped by the Resolver, and Lowering
  that persists the array after an `a(i).x=v` store and rehydrates it
  cross-turn — same kind-preserving pattern as handles (#118) / tables (#127) /
  plain structs (#131), distinct because struct arrays use `matlab_struct_arr*`.
  Deterministic regression `test/Repl/run_tests.sh`
  (`xturn_struct_array_field`, `xturn_struct_array_length`), verified empty
  without the fix.
- **2026-06-02 — #135 (✅ fixed).** A numeric matrix/vector didn't auto-grow on
  an out-of-bounds indexed assignment (`v(5) = 10` on a 1×3 left it 1×3, the
  write silently dropped) — inconsistent with cells / struct arrays, which do
  grow. AOT-level (all lanes). Fixed in `matlab_slice_store1_scalar`
  (runtime): an OOB linear index now grows the vector (col vector → rows,
  row/scalar/empty → cols), zero-filling the gap; a genuine 2-D matrix
  linear-OOB stays a no-op (MATLAB errors there). Regression
  `test/Run/regress_matrix_autogrow.m`, verified the write is dropped without
  the fix. (Skipped on emit-python / emit-typescript: their array shims can't
  grow a referenced array in place — a separate backend-codegen limitation;
  the emit-c/cpp lanes use the real runtime and grow correctly.) Found while
  fixing this: **#136** — `end` in single-subscript indexing resolves to
  `size(,1)` not `numel`, so `v(end)`/`v(end+1)` are wrong for row vectors
  (independent bug; together they make the `v(end+1)=x` append idiom work).
