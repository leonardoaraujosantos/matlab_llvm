# Design — mflowLink variable-step / stiff solvers

## Context

`MflowLinkSim::stepMajor()` advances the continuous state `Y_` between discrete ticks by
calling a private derivative callback `derivAll(t, y, ydot)` (which runs `evalAll` and reads
the integrator-block derivative slice). The solver layer is a set of free functions taking a
member-function-pointer `Deriv` plus scratch workspaces:

- `rk4Substep` — fixed-step classic RK4 (the default / fallback).
- `dopri5Step` — Dormand–Prince 5(4); fills `YOut` (5th order) and `Err` (embedded 4th).
- `bdf1Step` — Backward Euler with Newton + forward-difference Jacobian + `solveDense` LU.

Selection flags (`AdaptiveSolver_`, `Implicit_`) are set once from `M_.Solver`. The adaptive
accept/reject controller lives inline in `stepMajor` (factor `0.9·norm^-0.2`, capped `[0.1, 5]`).

The whole layer is shared by `-simulate` and `-emit-mflowlink-cpp` (the emitted binary links
`libMatlabFlowchart.a`), so additions need no separate codegen.

## Goals / Non-goals

**Goals**: native `ode23`; variable-order+variable-step `ode15s` (BDF1–5); `ode23s`/`ode23t`/
`ode23tb`; Jacobian reuse + analytic hook; mass-matrix / index-1 DAE; dense output + `Refine`;
fix stale comments; interpreter↔compiled byte-parity; existing models unchanged.

**Non-goals**: fully-implicit `f(t,y,y')=0` higher-index DAE; automatic stiffness detection /
solver auto-switch (`ode15s` vs `ode45`); sparse Jacobians (dense LU is fine at mflowLink
state sizes); GPU/threaded integration.

## Decisions

### 1. One adaptive controller, two lanes
Extract the inline accept/reject logic from `stepMajor` into a small `StepController` helper
(error-norm in, accept + next-`h` out) and drive **both** the explicit (DOPRI5, BS32) and the
implicit (BDF, Rosenbrock) lanes through it. Keep the current factor controller
(`0.9·norm^(-1/(p+1))`, clamp `[0.1, 5]`, honour `maxStep`/`minStep`) but parameterise the
order `p` so each method uses its own exponent. This removes the "ode15s is fixed-step" gap
without inventing a new controller.

### 2. Variable-order BDF as the real `ode15s`
Carry a ring buffer of the last ≤5 accepted `(t, y)` pairs. Form the BDF-`k` predictor from
the backward-difference history and solve the implicit corrector with the existing Newton +
`solveDense`. Order/step selection: estimate the local error from the difference of
order-`k` and order-`(k-1)` corrections; raise the order when it is converging cleanly,
drop it on rejects (the standard NDF15 strategy, simplified). BDF1 is exactly `k=1`, so the
current `ode15s` numerics remain reachable and the existing fixed-step path is the `k=1`,
controller-disabled special case.

History reset: any discrete state change, zero-crossing reset, or solver restart clears the
history and restarts at order 1 (Simulink does the same after a reset).

### 3. Jacobian amortisation + analytic hook
Cache the Newton Jacobian and its LU factors; refactor only when (a) the step size changes
beyond a threshold, (b) Newton convergence stalls, or (c) the order changes — the classic
"keep the Jacobian as long as Newton is happy" rule. Add an optional `jacobian(t,y,J)`
derivative-side hook: when a model can supply ∂f/∂y analytically (e.g. linear `state_space`
blocks have a constant `A`), use it instead of finite differences. Default stays
forward-difference so nothing regresses.

### 4. Rosenbrock / TR family by adapting the runtime solver
`ode23s` (Rosenbrock-W, 2nd order, one Jacobian per step, no Newton loop) already exists for
the MATLAB-language path. Port its stage formulae against the `MflowLinkSim` `Deriv`/Jacobian
callbacks rather than re-deriving. `ode23t` (trapezoidal rule, non-dissipative) and `ode23tb`
(TR-BDF2) reuse the same Jacobian + `solveDense` machinery. These give the moderately-stiff
and "stiff but want to preserve oscillation" options.

### 5. Mass matrix / index-1 DAE
Generalise the implicit residual from `G(y) = y − y_old − h·f` to `G(y) = M·(y − y_old) −
h·f` with Jacobian `M − h·∂f/∂y`. `M` comes from a solver setting (constant matrix literal)
or, later, a state-dependent `M(t,y)` callback. A singular `M` (semi-explicit index-1 DAE)
is allowed because the implicit lane never inverts `M` alone — only `M − h·J`, which is
non-singular for small `h`. The explicit lanes reject a non-identity `M` with a sourced
error (mass matrices require an implicit method, as in MATLAB).

### 6. Dense output + Refine
DOPRI5 has a standard 5th-order dense-output interpolant (extra `b*(θ)` weights over the
existing stages — no new derivative evaluations). BDF interpolates from its history
polynomial. Expose `settings.solver.refine = k`: emit `k` evenly-spaced interpolated samples
across each accepted step. This also sharpens zero-crossing localisation (interpolate instead
of RK4 re-integration). Default `refine = 1` (endpoints only) keeps current output
byte-identical.

### 7. Numeric-change management for `ode23`
Switching `ode23` from DOPRI5-alias to the real BS 3(2) changes its samples. Gate with: a
clear changelog/doc note, fresh `SimulateRun` goldens, and a sanity check that BS32 and
DOPRI5 agree to tolerance on a smooth reference (so the change is "more correct", not a
regression). No silent behaviour change.

## Risks / Mitigations

- **Variable-order BDF is the hardest piece** → land it incrementally: (a) variable-step
  BDF1 first (controller only), (b) then BDF2, (c) then orders 3–5 + order selection. Each
  step is independently testable and BDF1 is always the fallback.
- **Jacobian caching bugs cause silent inaccuracy** → assert Newton convergence per step;
  on stall, force a refactor then a step cut; regression on a stiff system with a known
  analytic solution.
- **Compiled-mode divergence** → every solver fixture is added to the `emit-mflowlink-cpp`
  byte-parity list, so interpreter and compiled must match exactly.
- **DAE scope creep** → restrict to index-1 with a (possibly singular) mass matrix; higher
  index and fully-implicit form stay on `dae_solver_roadmap.md`.

## Migration

None for existing models. `ode45` and fixed-step are numerically unchanged; `ode15s` keeps
BDF1 reachable as order 1; only `ode23` numerics improve (documented + re-goldened). New
settings (`ode23s`/`ode23t`/`ode23tb`, `refine`, mass matrix) are opt-in.
