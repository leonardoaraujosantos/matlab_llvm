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

| Result | Count |
|---|---|
| **OK** (compile + link + run, exit 0) | **187 / 250** |
| EMIT (frontend / lowering error) | 56 |
| LINK (undefined symbols at AOT link) | 7 |
| TIMEOUT / runtime crash | 0 |

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

Original list items not yet done:

- [ ] **B2 — `d2c`** (`c2d_zoh_demo.m`): inverse discretisation (needs a
  matrix-log / inverse-ZOH runtime). The `d2c_tustin` reverse map already
  exists; ZOH `d2c` does not.
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
| `control/bode_first_order` | `bode` 3-output `[mag,phase,wout]=bode(G,w)` + `margin` **on a `tf`** (have `margin` on `ss`) | CST feature |
| `control/lqr_double_integrator` | `c2d(ss_obj, Ts, 'zoh')` — c2d on a model object + method string | CST feature |
| `control/kalman_tracker` | `c2d(ss_obj, Ts, 'zoh')` (kalman 3-output now works) | CST feature |
| `control/tf_basic` | `tf('s')` builder + `matpow` on a tf | CST feature |
| `control/c2d_zoh_demo` | `c2d`/`d2c` **on a tf model object** (returning a discrete/continuous tf) + `disp(tf)` display; `d2c` exists for explicit ss matrices but `c2d(tf,Ts,'zoh')` returns a non-tf today | CST feature |
| `pde/*` (3) | `generateMesh` / `decsg` / `multicuboid` / `femodel` | PDE Toolbox |

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
- `matlab_logm` returns an empty matrix for a full matrix whose real Schur
  form keeps a 2×2 block (the Francis QR doesn't split a real-eigenvalue 2×2
  block into triangular). It works for diagonal / cleanly-deflating matrices.
  This limits `d2c` (ZOH) to those — the diagonal/decoupled case round-trips.
