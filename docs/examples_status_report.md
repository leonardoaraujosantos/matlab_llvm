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

### F. Flowchart fragments — 2 files — *not standalone-runnable*

| File | Issue |
|---|---|
| `mflowlink/cross_dialect.m` | needs `matlab_mflowlink_run` (mflowlink driver runtime); belongs to the `-emit-mflow-link-cpp` tooling |
| `mflow/blocks/clamp.m` | a `.mflow` computed-block body fragment (leaves `matlab.*` unconverted by design) — meant to be embedded in a flowchart, not run standalone |

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

## Progress — fixes applied 2026-05-22 (suite 472→ green throughout)

General compiler/parser fixes landed (each regression-tested):

- **`~` ignore-output in a multi-return LHS** (`[~, idx] = min(…)`) — parser
  now accepts the `~` placeholder; the multi-return lowering skips the null
  slot. Also fixed a **Sema null-deref** that crashed `matlabc` on any
  `[~, …] = f()`. Bonus: **`min`/`max`/`sort` value+index second output**
  (`[v, i] = max(A)`) via new `matlab_{min,max,sort}_idx` runtime + the
  `TwoReturns` splitter. Test `test/Run/multiret_tilde.m`.
- **`Name=Value` call arguments** (`f(Width=8, Name="x")`) — parsed and
  lowered to the classic `'Name', Value` pair. (`lib/Parse/Parser.cpp`.)
- **`ssdata` / `tfdata`** — added as `ss` / `tf` classdef methods, made
  reachable via **function-style class-method dispatch for multi-return**
  (`[A,B,C,D] = ssdata(sys)` → `ss.ssdata`). New general capability in the
  multi-return lowering path. Test `test/Run/cst_data_extract.m`.

**Important — fixing each reported *first error* often reveals a *deeper*,
previously-hidden blocker in the same example** (the sweep only saw the first
error). The targeted examples now need these further features:

| Example | Now blocked on | Depth |
|---|---|---|
| `control/bode_first_order` | `margin` (4-output gain/phase margin) | CST feature |
| `control/lqr_double_integrator` | `c2d(ss_obj, Ts, 'zoh')` — c2d on a model object + method string | CST feature |
| `control/kalman_tracker` | `kalman` 3-output multi-return on a model object | CST feature |
| `control/tf_basic` | `tf('s')` builder + `matpow` on a tf | CST feature |
| `control/c2d_zoh_demo` | `d2c` (inverse discretisation — matrix log) | CST runtime |
| `control/step_response_siso` | `step` 2-output `[y,t]` multi-return | CST feature |
| `pde/*` (3) | `generateMesh` / `decsg` / `multicuboid` / `femodel` | PDE Toolbox |

Still deferred (deeper than a quick gap):
- **D (`legend({...})` / cell-of-strings)** — cells don't track per-element
  type; a string element stored via `cell_set_mat` is read back as a
  char-matrix by `matlab_legend`. Needs cell string-element typing +
  `matlab_legend` string support. (A naive "store as `matlab_string`" turns
  the compile error into a `bad_alloc` — reverted.)
- **E (symbolic)** — `symbolic_demo` / `quadrotor_derive_eom` compile fine;
  they need `runtime_sym` + the external **SymPP** lib at AOT link (the
  example link set omits them; the `matlabc` binary itself links them).
- **F (flowchart fragments)** — not standalone-runnable `.m`.
