# Examples — Compile/Execute Status Report

Scope: verify every `.m` under `examples/` (and subfolders) **compiles and
executes** through the LLVM path — *not* the SystemVerilog / Python / C /
C++ / TypeScript emitters. This is input for a follow-up fix roadmap.

## Method

For each `.m`: `matlabc -emit-llvm` → `clang++` link against the full runtime
object set (incl. plotting + cairo) → run with a 12 s timeout from the file's
own directory. This is the project's canonical AOT execute path (same as
`test/Run/run_tests.sh` / `/tmp/fastrun.sh`). Sweep date: 2026-05-22.

## Summary

Original sweep (2026-05-22, before fixes):

| Result | Count |
|---|---|
| **OK** (compile + link + run, exit 0) | **187 / 250** |
| EMIT (frontend / lowering error) | 56 |
| LINK (undefined symbols at AOT link) | 7 |
| TIMEOUT / runtime crash | 0 |

Re-sweep after this session's fixes (sym + mflowlink extras linked per the
recipes below):

| Result | Count |
|---|---|
| **OK** | **193 / 250** |
| EMIT | 53 |
| LINK | 4 |
| TIMEOUT / runtime crash | 0 |

The +6 OK are exactly the examples fixed this session: `bode_first_order`,
`step_response_siso`, `multi_series` (plot), `symbolic_demo`,
`quadrotor_derive_eom` (sym link), and `cross_dialect` (mflowlink link). The
remaining non-HDL failures are all documented deeper gaps:
**control (4)** — `lqr_double_integrator`, `kalman_tracker`, `c2d_zoh_demo`
(gap #2, c2d/d2c on a tf object), `tf_basic` (`tf('s')` + `matpow`);
**pde (3)** — `generateMesh`/`decsg`/`multicuboid` (gap #4);
**`mflow/blocks/clamp.m`** — a custom-block function body (F false positive,
runs via `custom_clamp.mflow`). The remaining **49 HDL** are out of LLVM-execute
scope (SV/cocotb targets).

**Clean folders (100% OK):** `antenna` (3), `comm` (19), `globaloptim` (7),
`ident` (7), `images` (7), `mpc` (5), `optim` (16), `rf` (10), `signal` (10),
`stats_ml` (9), `verilog_a` (24), plus 30/31 top-level examples.

All 63 failures are at **compile or link time** — nothing that compiled+linked
crashed or hung at runtime.

---

## Failures by category (for the roadmap)

### A. HDL examples — `examples/hdl/` (49 files, all fail) — *out of LLVM-execute scope*

These target `-emit-systemverilog` / `-emit-cocotb` (the SV path the user
excluded). They are not designed to run through the scalar LLVM execute path.
Root causes:

| # | Cause | Note |
|---|---|---|
| 27 | persistent fixed-point register store (`matlab_global_set_f64`, 2-arg) | the HDL register idiom `persistent reg; if isempty(reg); reg = fi(0,0,4,0); end` — the 2-arg persistent-set ABI lowers for SV but not the LLVM-execute path |
| 10 | multi-file reference (`undefined name 'foo'`) | `test_*.m` / `*_synth.m` drivers call a function defined in a *sibling* `.m`; single-file compilation can't see it |
| 7 | unconverted comparison ops (`matlab.eq/ne/gt/...`) | downstream of the persistent/fi lowering gaps |
| 4 | LINK undefined symbols | `aes_round`, `hamming74`, `median3`, `vector_processor` |
| 1 | array store (`__subscript_store`, 3-arg) | `sequential_processor.m` |

**Roadmap stance:** validate these via `-emit-systemverilog` + the cocotb
SIL harness (their actual target). Running them as software reference models
through LLVM would need (a) LLVM-path lowering of the persistent-fi register
set, and (b) multi-file/module linking — a sizeable, separate effort.

### B. Control System — `examples/control/` (6 files) — **real gaps**

| File | Issue | Suggested fix |
|---|---|---|
| `lqr_double_integrator.m`, `kalman_tracker.m` | `undefined name 'ssdata'` | add `ssdata` (extract A,B,C,D from a state-space model) |
| `tf_basic.m` | `undefined name 'tfdata'` | add `tfdata` (extract num/den from a tf model) |
| `c2d_zoh_demo.m` | `undefined name 'd2c'` | add `d2c` (discrete→continuous, inverse of `c2d`) |
| `step_response_siso.m` | `[y,t] = step(sys)` — 2-output multi-return unsupported | wire `step` multi-return splitter |
| `bode_first_order.m` | `[~, idx] = min(...)` — `~` ignore-output in a multi-return LHS not parsed (`unexpected '~' in expression`) | parser support for `~` LHS placeholder |

### C. PDE — `examples/pde/` (3 files) — **real parser gap**

`tuningfork_modal.m`, `poisson_disk.m`, `clamped_plate_pressure.m` all use the
**name=value call syntax**, e.g. `femodel(AnalysisType="structuralModal", …)`
→ `error: expected ')', found '='`. The parser only accepts the classic
`'Name', Value` form.

**Fix:** parser support for `Name=Value` function arguments (a newer MATLAB
syntax), lowering them to the existing name-value handling.

### D. Plotting — `examples/plot/` (1 file) — **real gap**

`multi_series.m` — `matlab_cell_set_mat: 3 arguments` — assigning a matrix
into a cell element (`c{i} = vec`) for a cell-array-of-series. The cell-set
with a matrix value shape isn't lowered.

### E. Symbolic Math — 2 files — **link-config gap (compiles fine)**

`symbolic_demo.m`, `quadrotor/quadrotor_derive_eom.m` — `-emit-llvm` succeeds;
the AOT link fails on `sympp::Symbol …` / `matlab_sym_*`. They need the
symbolic runtime (`runtime/toolbox/sym/runtime_sym.cpp.o`) **and** the
external **SymPP** library linked, which the standard example link set omits.
Not a compiler bug.

**Fix:** add `runtime_sym` + SymPP to the example AOT link recipe (the main
`matlabc` binary already links them).

### F. Flowchart fragments — 2 files — *not standalone-runnable* — **RESOLVED (false positive)**

Both run and are covered by flowchart CTest lanes (see the progress log below);
the sweep mis-flagged them by linking them as plain LLVM programs.

| File | Issue | How it actually runs |
|---|---|---|
| `mflowlink/cross_dialect.m` | needs the Flowchart libs (`mflowlink_run`) | `runtime/scripts/build_and_run.sh` detects `mflowlink_run`; tested by `test/Flowchart/CrossDialect` |
| `mflow/blocks/clamp.m` | a custom-block `function` body, not a script | embedded via `custom_clamp.mflow`; `-emit-matlab` tested by `test/Flowchart/EmitMatlab` |

---

## Priority suggestion for the fix roadmap

1. **Quick, high-value (common MATLAB idioms):**
   `~` ignore-output in multi-return LHS (B); `ssdata`/`tfdata`/`d2c`
   builtins (B); `Name=Value` argument parsing (C); cell `{i}=matrix` set (D).
2. **Medium:** `step` (and likely `impulse`/`lsim`) multi-return splitters (B).
3. **Config-only:** link `runtime_sym` + SymPP for symbolic examples (E).
4. **Out of current scope:** HDL examples (A) — exercise via SV/cocotb;
   flowchart fragments (F) — exercise via the mflow tooling.

---

## Progress log — fixes applied 2026-05-22 (suite 472 → 476, green throughout)

### DONE — general compiler/parser/runtime fixes (each has a regression test)

| Fix | What it covers | Regression test |
|---|---|---|
| `~` ignore-output in a multi-return LHS (`[~, idx] = …`) + the **Sema null-deref** that crashed `matlabc` on *any* `[~, …] = f()` | parser null LHS slot + `Resolver.cpp` null guard | `test/Run/multiret_tilde.m` |
| `min`/`max`/`sort` value+index 2nd output (`[v,i]=max(A)`) | `matlab_{min,max,sort}_idx` + `TwoReturns` splitter | `test/Run/multiret_tilde.m` |
| `Name=Value` call arguments (`f(Width=8)`) → `f('Width', 8)`; `==` stays a comparison | `Parser::parseArgList` | `test/Run/name_value_args.m` |
| `ssdata` / `tfdata` + **function-style class-method dispatch for multi-return** | `ss`/`tf` classdef methods + lowering | `test/Run/cst_data_extract.m` |
| CST model-object multi-return splitters: `[kest,L,P]=kalman(sys,Qn,Rn)`, `[Gm,Pm,Wcg,Wcp]=margin(sys)` | model-object path + `matlab_margin_ss_auto` | `test/Run/cst_multiret.m` |
| **D** cell-of-strings `{'a','b'}` (kind=3 string elements) → `legend({...})` works | `matlab_cell_set_str` + `cell_get_mat` char-code | `test/Run/cell_strings.m` |
| **E** symbolic AOT link recipe — `.requires-sym` marker links the prebuilt `WITH_SYM` `runtime_sym.o` + `libsympp` + GMP/MPFR (skips when SymPP absent). `symbolic_demo.m` / `quadrotor_derive_eom.m` run. | `run_tests.sh` / `fastrun.sh` `.requires-sym` | `test/Run/sym_basic.m` |
| **B3** `step` honours a supplied time vector + `[y,tout]` 2-output, on ss **and tf** (new `step_ss_t`/`step_tf_t`, tf via controllable-canonical `tf2ss`). | model-object multi-return path + step dispatch | `test/Run/step_multiret.m` |
| **scalar `^`** on the AOT path (`wn^2`) — `matlab.matpow(f64,f64)` → `matlab_pow_scalar`; `matrix^n` → `matlab_matpow`. Was a pre-existing gap (only the C/C++ emitters handled `^`). | LowerTensorOps matpow arm | `test/Run/step_multiret.m` |
| **B2** `d2c` — ZOH discrete→continuous, the explicit-matrix inverse of c2d (`[A,B]=d2c(Ad,Bd,Ts)` via `logm`). | LowerTensorOps d2c splitter + `matlab_d2c_{A,B}` | `test/Run/d2c_roundtrip.m` |
| **F** flowchart fragments — confirmed runnable + covered (not standalone `.m`): `cross_dialect.m` via `build_and_run.sh`; `clamp.m` custom block via `custom_clamp.mflow` emit-matlab. | n/a (false positive) | `test/Flowchart/CrossDialect`, `test/Flowchart/EmitMatlab` |

### TODO — remaining work (with real depth)

Original list items — all of B/C/D/E/F now done; only HDL (out of scope) and
the deeper per-example chains (see the fix plan below) remain:

- [x] **B2 — `d2c`** — DONE for explicit ss matrices (`[A,B]=d2c(Ad,Bd,Ts)`,
  ZOH inverse via `logm`). `c2d_zoh_demo.m` additionally needs c2d/d2c on a tf
  model object — see fix plan #2; the general (full-matrix) d2c needs the
  `logm` fix #5.
- [x] **B3 — `step` 2-output `[y, t] = step(sys, t)`** — DONE (ss + tf, honours
  the time grid). `step_response_siso.m` still needs **`stepinfo` to return a
  struct** (`S.RiseTime`): the runtime returns a 1×5 row and the existing
  `ctrl_stepinfo.m` test reads it positionally (`si(1,1)`), so switching to a
  struct is a convention change to make separately (would update that test).
- [x] **E — symbolic** (`symbolic_demo.m`, `quadrotor_derive_eom.m`): DONE.
  The `.requires-sym` test marker links the prebuilt `WITH_SYM` `runtime_sym.o`
  + `libsympp` (SymPP_DIR read from `CMakeCache.txt`) + GMP/MPFR, skipping when
  SymPP isn't built. Both examples run via the AOT path. (To run an example
  directly: link `build/.../runtime_sym.cpp.o` + `-L<SymPP>/src -lsympp` +
  `-lgmp -lmpfr` alongside the normal runtime objects.)
- [x] **F — flowchart fragments** (`mflowlink/cross_dialect.m`,
  `mflow/blocks/clamp.m`): RESOLVED — a false positive of the standalone-`.m`
  sweep (it linked them as plain LLVM programs). Both are runnable + already
  covered by flowchart CTest lanes:
  - `cross_dialect.m` runs via `runtime/scripts/build_and_run.sh` (which detects
    `mflowlink_run` and links `libMatlabFlowchart.a` + `runtime_mflowlink_call.cpp`);
    tested by `test/Flowchart/CrossDialect/run_tests.sh` (passes — checks the
    logged-signal banner + the stable −0.31 scope value, tolerating the
    near-zero noise entries).
  - `mflow/blocks/clamp.m` is a custom-block body (a `function`, no script), used
    by `examples/mflow/custom_clamp.mflow`; exercised via `-emit-matlab`
    (matches `test/Flowchart/EmitMatlab/custom_path.expected`).
- [ ] **A — HDL** (49 files): out of LLVM-execute scope (SV/cocotb targets).

**Deeper blockers revealed *after* fixing each first-error** (the sweep only
saw the first error per file; these surfaced once it was fixed):

| Example | Now blocked on | Depth |
|---|---|---|
| `control/bode_first_order` | **RUNS** — gap #3 (bode 3-output + margin/dcgain/bandwidth on tf) done | — |
| `control/step_response_siso` | **RUNS** — B3 step + gap #1 (stepinfo struct) done | — |
| `control/lqr_double_integrator` | `c2d(ss_obj, Ts, 'zoh')` — c2d on a model object + method string (gap #2) | CST feature |
| `control/kalman_tracker` | `c2d(ss_obj, Ts, 'zoh')` (kalman 3-output now works) (gap #2) | CST feature |
| `control/tf_basic` | `tf('s')` builder + `matpow` on a tf | CST feature |
| `control/c2d_zoh_demo` | `c2d`/`d2c` **on a tf model object** + `disp(tf)` (gap #2); the underlying ss `d2c`/`logm` now work | CST feature |
| `pde/*` (3) | `generateMesh` / `decsg` / `multicuboid` / `femodel` (gap #4) | PDE Toolbox |

### Known pre-existing limitations surfaced (NOT regressions, NOT yet fixed)

These were exposed while testing the fixes above; each fails identically on a
clean tree and is independent of this work:

- `m = c{i}; m(k)` — subscripting a **cell-element-result variable** does not
  lower for *any* element type (matrix, string, scalar).
- `disp(c{i})` routes a string element through the scalar `cell_get_f64` path
  (prints `0`); only `cell_get_mat` consumers (e.g. legend) see the string.
- passing a **string literal to a user function** (`f('Mode', 3)`) leaves the
  `const_char` unconverted — so user-defined name-value handlers don't work
  (builtins like `plot` do).
- `numel()` of a cell-element-result; `struct('a',1,'b',2)` (struct name-value
  construction) — both report "unsupported call shape".
- ~~`matlab_logm` returns an empty matrix for a full matrix whose real Schur
  form keeps a 2×2 block.~~ **FIXED** (gap #5) — `matlab_logm` now standardizes
  real-eigenvalue 2×2 Schur blocks (Givens triangularize), so it (and `d2c`)
  work for general real-eigenvalue matrices. Genuinely complex-eigenvalue
  blocks are still rejected (a complex-aware log is a separate item).

---

## Fix plan for the remaining deeper gaps

All of the reported B/C/D/E/F gaps are fixed (see the progress log). What
remains are the *deeper, per-example chains* that only surfaced after each
first-error fix. Each below is a concrete plan with the touch points and an
effort/risk estimate. Suggested order (cheap→expensive): **3 → 1 → 5 → 2 → 4**.

**Status:** #1, #3, #5 are **DONE** (suite 481) — `bode_first_order.m` and
`step_response_siso.m` now run end to end, and `d2c` works for general
real-eigenvalue systems. #2 and #4 remain.

### 1. `stepinfo` as a struct (`S.RiseTime`) — *small* — DONE (21a4297... see progress log)

- **Blocks:** `step_response_siso.m` (rise/settle/overshoot section).
- **Root cause:** `matlab_stepinfo(y,t)` returns a 1×5 row `[Rise, Settle, Over,
  Peak, PeakTime]`; `S.RiseTime` is a field access on a matrix → crash. The
  existing `test/Run/ctrl_stepinfo.m` reads it positionally (`si(1,1)`).
- **Plan:**
  1. Add `matlab_stepinfo_struct(y,t)` → `matlab_struct` with fields
     `RiseTime / SettlingTime / SettlingMin / SettlingMax / Overshoot /
     Undershoot / Peak / PeakTime` (reuse the current math +
     `matlab_struct_new` / `matlab_struct_set_f64`).
  2. Route `stepinfo` to the struct form and tag the result binding as a struct:
     add `"stepinfo"` to the `RhsIsStruct` known-builtin list
     (`lib/MLIR/Lowering.cpp` ~2334) so `S.RiseTime` lowers via
     `matlab_struct_get_f64`.
  3. Update `test/Run/ctrl_stepinfo.m` to read struct fields (the MATLAB-correct
     convention).
- **Risk:** intentionally changes `ctrl_stepinfo.m`'s expected output.

### 2. `c2d` / `d2c` on a tf model object + `disp(tf)` — *large*

- **Blocks:** `c2d_zoh_demo.m` (and the `c2d(ss_obj, Ts, 'zoh')` form in
  `lqr_double_integrator.m` / `kalman_tracker.m`).
- **Root cause:** `c2d` only dispatches `ss` with 2 args
  (`lib/MLIR/Lowering.cpp` ~7533). `c2d(tf, Ts, 'zoh')` falls to the generic
  builtin path and returns a non-tf, so `disp(Cd_zoh)` and everything
  downstream break. There is no discrete-model (sample-time) representation.
- **Plan (in order):**
  1. **`ss2tf` runtime** (`matlab_ss2tf_num` / `_den`): state space → num/den
     (den = `poly(eig(A))`; num = `C·adj(sI−A)·B + D`). Pairs with the existing
     `tf2ss_ccf`.
  2. **Sample time on the model:** add a `Ts` property to the `ss` / `tf`
     classdefs (0 = continuous); `c2d` sets it, `d2c` clears it.
  3. **c2d-on-model dispatch** (ss + tf), 2- and 3-arg with a method string:
     ss → `c2d_ss` / `c2d_tustin` → rebuild ss with `Ts`; tf → `tf2ss` → c2d →
     `ss2tf` → rebuild tf with `Ts`.
  4. **d2c-on-model**: the inverse (needs the `logm` fix #5 for a tf with an
     integrator pole at z = 1).
  5. **`disp(tf)` / `disp(ss)`**: pretty-print the transfer function in the
     s- or z-domain (extend the model-object display path).
- **Risk:** touches the model-object classdefs + display; do it as the staged
  sequence above so each step is testable.

### 3. `bode` 3-output + `margin` on a tf — *small–medium* — DONE

- **Blocks:** `bode_first_order.m`.
- **Root cause:** `bode` has 1-output (mag) and a 2-output `[mag,phase]`
  splitter; the 3-output `[mag,phase,wout]` form does not exist. The `margin`
  model-object splitter handles `ss` only (`matlab_margin_ss_auto`).
- **Plan:**
  1. **bode 3-output:** extend the `[mag,phase]=bode` multi-return splitter
     (LowerTensorOps) with a 3rd result `wout` = the supplied `w` (echo), or an
     auto log-spaced grid when `w` is omitted.
  2. **margin on tf:** in the model-object multi-return `margin` case
     (`lib/MLIR/Lowering.cpp`), add a `tf` branch — `tf2ss` →
     `matlab_margin_ss_auto`, or a dedicated `matlab_margin_tf_auto(num,den)`
     (build `w`, `freqresp_tf` → |H|/phase, interpolate the crossovers, mirroring
     `allmargin_ss`).

### 4. PDE mesh builders (`generateMesh` / `decsg` / `multicuboid` / `femodel`) — *large (own roadmap)*

- **Blocks:** `tuningfork_modal.m`, `poisson_disk.m`, `clamped_plate_pressure.m`.
- **Root cause:** undefined — the `Name=Value` fix lets the calls parse, but the
  PDE geometry/mesh subsystem isn't on this path.
- **Plan:** this is the PDE Toolbox Tier-2 mesher tracked in
  `docs/pde_toolbox_roadmap.md`: `decsg` (constructive-solid-geometry → 2-D
  geometry), `multicuboid` (3-D box geometry), `generateMesh` (Delaunay 2-D /
  tetrahedral 3-D mesher with `Hmax`), and the `femodel`/`generateMesh`
  plumbing. Multi-week; sequence under the PDE roadmap, not this gap list.

### 5. `matlab_logm` 2x2-real-block fix (generalizes `d2c`) — *medium (numerical)* — DONE

- **Blocks:** `d2c` on a full (non-diagonal) matrix; `logm` generally.
- **Root cause:** `francis_qr_` leaves a 2×2 block for a pair of *real*
  eigenvalues, and `matlab_logm` (`runtime/matlab_runtime.cpp` ~2455) rejects
  any non-zero subdiagonal (returns empty).
- **Plan:** after the Schur step in `matlab_logm`, **standardize 2×2 blocks**:
  for a block with real eigenvalues (discriminant ≥ 0) apply a Givens/Jacobi
  rotation that zeros the subdiagonal (triangularize), updating `U`; only
  genuinely complex pairs remain as 2×2 (flag those — a complex-aware log is a
  separate item). Parlett's recurrence then runs on the now-triangular `T`.
- **Payoff:** `d2c` works for general ss systems; `logm` (and `sqrtm`-style
  functions) become robust.
