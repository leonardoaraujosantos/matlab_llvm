# Tasks — mflowLink variable-step / stiff solvers

Land incrementally; each slice is independently testable and BDF1/RK4 remain the fallback.

## 0. Record-correction (cheap, do first)
- [x] Fix stale solver comments in `MflowLinkSim.cpp` (the "Tier G not yet landed" note ~2830;
      the BDF1 "fixed-point iteration" caveat ~2956 — the code uses Newton).
- [x] Correct `docs/mflow_link_roadmap.md` §17.4 solver table (BDF1 *is* shipped; ode23 was an
      alias, now native; ode23s/t/tb fall through to RK4). *(PR #395 + ode23-native update)*

## 1. Shared step controller
- [~] The adaptive accept/reject loop now keys its step exponent off the method's embedded-error
      order (`1/(order+1)`), shared by DOPRI5 and BS32. A full `StepController` extraction (so the
      implicit lane can reuse it) lands with the variable-step BDF slice (task 3).
- [x] DOPRI5 numerics unchanged (exponent 0.2 preserved); `ode45` output byte-identical.

## 2. Native ode23 (Bogacki–Shampine 3(2)) — DONE
- [x] Add `bs32Step` (BS 3(2) tableau, FSAL, embedded 2nd-order error).
- [x] Route `ode23` to `bs32Step` (method enum `AdaptiveMethod_`) instead of aliasing DOPRI5.
- [x] Regression `ode23_decay.mflow` (y'=-y → e^-2; ode23 vs ode45 agreement) + emit-parity.

## 3. Variable-step BDF1 → variable-order BDF (ode15s)
- [x] **Variable-step BDF1** (`ImplicitAdaptive_`): under `variable_step`, `ode15s` sub-steps each
      major window with **step-doubling** (Richardson) error control — full step vs two half steps,
      advance with the half-step result, exponent 1/(order+1)=1/2, `maxStep` caps the step.
      Regression `ode15s_adaptive.mflow` (e^-2 within 1e-4 vs fixed-step's 0.013 error).
- [x] Keep fixed-step BDF1 reachable: `type: fixed_step` keeps the existing one-step path
      (stiff_bdf and the ode23s/t/tb fixtures are byte-identical).
- [ ] Variable-**order** BDF: add a `(t, y)` history ring buffer (≤5), BDF2, then orders 3–5
      with NDF-style order selection (the larger remaining piece).

## 4. Jacobian amortisation + analytic hook
- [ ] Cache the Newton Jacobian + LU factors; refactor only on step-change / Newton stall / order-change.
- [ ] Add an optional `jacobian(t, y, J)` derivative-side hook (constant `A` for linear blocks);
      default stays forward-difference.

## 5. Rosenbrock / TR family
- [x] `ode23s` modified Rosenbrock (2)3 (`rosenbrockStep`) against the `MflowLinkSim` Deriv
      callback: `W = I − h·d·J` (FD Jacobian) + `∂f/∂t` term, three back-substitutions via
      `solveDense`, embedded error estimate. `StiffMethod_` enum dispatches BDF1 vs Rosenbrock.
- [x] `Loader` already passes `ode23s` through; wiring it removes its RK4 fall-through.
      Regression `ode23s_stiff.mflow` (stiff plant at fixed h=0.1 RK4 would explode) + emit-parity.
- [x] `ode23t` (trapezoidal rule, non-dissipative) via `trapezoidalStep` on the same Newton +
      `solveDense` machinery; `StiffMethod::TRAPEZOIDAL`. Regression `ode23t_oscillator.mflow`.
- [x] **Bug fix (#398):** `derivative()` now settles all block outputs before reading
      derivatives — coupled multi-state implicit solvers (ode15s/ode23s/ode23t) diverged
      because a single eval pass read stale cross-coupled outputs into the FD Jacobian.
- [x] `ode23tb` (TR-BDF2) via `trBDF2Step`: trapezoidal sub-step (reuses `trapezoidalStep`) +
      BDF2 sub-step (Lagrange-derived non-uniform-mesh coefficients, consistency-checked
      α₀+α₁+α₂=0). `StiffMethod::TRBDF2`. Regression `ode23tb_stiff.mflow`. No named method
      now falls through to RK4.

## 6. Mass matrix / index-1 DAE
- [ ] Generalise the implicit residual to `M·(y − y_old) − h·f`, Jacobian `M − h·∂f/∂y`.
- [ ] Constant `M` from a solver setting; allow singular `M` (semi-explicit index-1).
- [ ] Explicit lanes reject non-identity `M` with a sourced error.

## 7. Dense output + Refine
- [ ] DOPRI5 dense-output interpolant (no extra derivative evals); BDF history-polynomial interp.
- [ ] `settings.solver.refine = k` → `k` interpolated samples per accepted step; default 1 = endpoints.
- [ ] Use the interpolant for zero-crossing localisation (replace RK4 re-integration).

## 8. Tests + parity
- [ ] `SimulateRun` fixtures: stiff scalar decay (BDF1 needs tiny `h`, BDF5 does not);
      Van der Pol / Robertson-style stiff system; a mass-matrix DAE; a `refine` densification check;
      a smooth system where ode23/ode45/ode15s all agree to tolerance.
- [ ] Add every solver fixture to the `emit-mflowlink-cpp` byte-parity list (interpreter == compiled).
- [ ] Regenerate any affected goldens; full flowchart ctest suite green.

## 9. Docs
- [ ] `docs/mflow_link_roadmap.md` §17.4 — solver suite status table.
- [ ] `docs/dae_solver_roadmap.md` — mflowLink mass-matrix / index-1 entry.
- [ ] `docs/mflowlink_blocks.md` — solver-settings section (algorithm list, `refine`, mass matrix).
