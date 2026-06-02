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

   **Still open (carved out of this fix, each tracked separately):**
   - **Direct cross-turn handle call with a matrix arg → SIGSEGV (#119).**
     The kind=13 call trampolines are scalar-only; calling a recovered handle
     directly with a vector (`rastrigin(xlocal)`) lowers to a subscript on the
     code pointer. This is what still crashes the 6 `globaloptim/*` demos
     (their solver calls now succeed; their `fprintf` self-reports don't).
     Needs the kind=13 ABI to carry arity + return-kind plus matrix
     trampolines.
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
