## Why

The mflowLink simulator is closer to Simulink's variable-step/stiff story than first
appears — but with real, named gaps. Today (`lib/Flowchart/MflowLinkSim.cpp`):

- **`ode45`** is a correct Dormand–Prince 5(4) adaptive integrator with an embedded
  error estimate and a factor-based step controller (grow/shrink, min-floor `1e-15`,
  reject-and-retry, RK4 fallback after 32 rejects).
- **`ode23`** is recognised but **aliases the DOPRI5 path** — it is *not* the native
  Bogacki–Shampine 3(2) pair MATLAB uses.
- **`ode15s`** is a genuine implicit solver: Backward Euler (**BDF1 only**) with a real
  Newton iteration, a forward-difference Jacobian, and a dense LU solve — but it is
  **fixed-step** (the user must pick `maxStep`), **order 1**, recomputes the full
  Jacobian every Newton iteration, and has **no mass-matrix / DAE** support.
- **`ode23s` / `ode23t` / `ode23tb`** are parsed but **fall through to RK4** — even
  though the runtime already ships a Rosenbrock `ode23s` for the MATLAB-language path
  (`runtime/matlab_runtime.cpp`, capability `ode-pde-solvers`).
- **No dense output**: signals are logged only at accepted step endpoints (zero-crossing
  bisection re-integrates with RK4); there is no Hermite interpolant and no `Refine`.

Several in-code comments are now stale (e.g. "ode45 builtins can be plumbed in once
Tier G lands" — Tier G shipped; the BDF1 caveat says "fixed-point iteration" but the code
uses Newton). This change closes the substantive solver gaps and corrects the record.

All solver paths already run identically in the interpreter (`-simulate`) and the compiled
`-emit-mflowlink-cpp` binary (both drive `MflowLinkSim`), so every addition here is
inherited by compiled models for free.

## What Changes

- **Native `ode23` (Bogacki–Shampine 3(2)).** Add the BS 3(2) tableau with its embedded
  2nd-order estimate; route `ode23` to it instead of aliasing DOPRI5.
- **Variable-order BDF for `ode15s` (orders 1–5).** Replace BDF1-only with a variable-order,
  variable-step BDF integrator carrying a short step/state history, error-controlled order
  and step selection (the standard NDF/BDF strategy). BDF1 stays the order-1 special case.
- **Variable-step control for the stiff lane.** Reuse the adaptive controller so `ode15s`
  (and the new stiff methods) pick their own step from `relTol`/`absTol`, with
  `maxStep`/`minStep` honoured — not a user-fixed step.
- **Wire `ode23s` (Rosenbrock-W) into mflowLink.** Adapt the runtime's Rosenbrock stiff
  solver to the `MflowLinkSim` derivative callback so stiff models get a one-step
  linearly-implicit option. `ode23t` (trapezoidal) / `ode23tb` (TR-BDF2) follow as the
  moderately-stiff members.
- **Jacobian reuse + optional analytic Jacobian.** Amortise the Jacobian across Newton
  iterations and across steps (refactor only when the step or convergence rate demands),
  and accept a model-supplied Jacobian hook where available.
- **Mass-matrix / index-1 DAE.** Support `M(t,y)·y' = f(t,y)` (constant and state-dependent
  `M`, including singular `M` for semi-explicit index-1 DAEs) in the implicit lane.
- **Dense output + `Refine`.** Add the DOPRI5/BDF interpolants so logging (and zero-crossing
  localisation) can produce sub-step samples without shrinking the step, exposed through a
  solver `refine` setting.
- **Correct the stale solver comments** and the `mflow_link_roadmap.md` §17.4 solver table.

No breaking changes: existing `ode45`/`ode15s`/fixed-step models keep their current numeric
behaviour (BDF1 remains reachable as `ode15s` order-1; `ode23` numerics change only because
it becomes the *correct* method, gated behind a documented note + regression goldens).

## Capabilities

### New Capabilities
- `mflow-variable-step-stiff-solvers`: the mflowLink variable-step / stiff ODE solver suite —
  native `ode23`, variable-order/variable-step `ode15s` (BDF1–5), `ode23s`/`ode23t`/`ode23tb`,
  Jacobian reuse + analytic-Jacobian hook, mass-matrix/index-1 DAE, dense output + `Refine`,
  and interpreter↔compiled parity.

### Modified Capabilities
- `flowchart-frontend`: the "mflowlink block model and simulation" requirement gains the
  variable-step/stiff solver scenarios (the simulator's solver behaviour lives here).

## Impact

- **Sim** (`lib/Flowchart/MflowLinkSim.cpp`, `include/matlab/Flowchart/MflowLinkSim.h`):
  new integrators (`bs32Step`, variable-order `bdfStep`, `rosenbrockStep`), a shared
  adaptive controller used by both explicit and implicit lanes, a step/state history buffer,
  dense-output interpolants, Jacobian cache, mass-matrix plumbing into `solveDense`.
- **Loader** (`include/matlab/Flowchart/Loader.h`): recognise `ode23s`/`ode23t`/`ode23tb`
  and a `refine` solver setting; `minStep`/mass-matrix params honoured.
- **Reuse**: the runtime Rosenbrock `ode23s` and the `ode-pde-solvers` numerics as the
  reference for the mflowLink port; the existing `solveDense` LU and zero-crossing bisection.
- **Tests**: `SimulateRun` stiff/variable-step fixtures (a stiff decay where BDF1 needs tiny
  steps but BDF-5 does not; a Van der Pol / Robertson-style stiff system; a mass-matrix DAE;
  a `Refine` densification check) plus `emit-mflowlink-cpp` byte-parity for each.
- **Docs**: `docs/mflow_link_roadmap.md` §17.4 solver table, `docs/dae_solver_roadmap.md`
  (mflowLink mass-matrix entry), `docs/mflowlink_blocks.md` solver-settings section.
