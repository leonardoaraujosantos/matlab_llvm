# Tasks — mflowLink variable-step / stiff solvers

Land incrementally; each slice is independently testable and BDF1/RK4 remain the fallback.

## 0. Record-correction (cheap, do first)
- [ ] Fix stale solver comments in `MflowLinkSim.cpp` (the "Tier G not yet landed" note ~2830;
      the BDF1 "fixed-point iteration" caveat ~2956 — the code uses Newton).
- [ ] Correct `docs/mflow_link_roadmap.md` §17.4 solver table (BDF1 *is* shipped; ode23 aliases
      DOPRI5; ode23s/t/tb fall through to RK4).

## 1. Shared step controller
- [ ] Extract the inline accept/reject logic from `stepMajor` into a `StepController` helper
      parameterised by method order `p` (error-norm in → accept + next-`h` out); honour
      `maxStep`/`minStep`.
- [ ] Re-route DOPRI5 through it; assert `ode45` output is byte-identical (pure refactor).

## 2. Native ode23 (Bogacki–Shampine 3(2))
- [ ] Add `bs32Step` (BS 3(2) tableau, FSAL, embedded 2nd-order error).
- [ ] Route `ode23` to `bs32Step` instead of aliasing DOPRI5.
- [ ] Doc note + fresh goldens; sanity-check BS32 vs DOPRI5 agree on a smooth reference.

## 3. Variable-step BDF1 → variable-order BDF (ode15s)
- [ ] Add a `(t, y)` history ring buffer (≤5) + reset on discrete change / zero-crossing / restart.
- [ ] Variable-step BDF1 via the shared controller (error from step-doubling or the BDF2 estimate).
- [ ] Add BDF2; then orders 3–5 with NDF-style order selection.
- [ ] Keep fixed-step BDF1 reachable (order 1, controller off) so current `ode15s` numerics persist.

## 4. Jacobian amortisation + analytic hook
- [ ] Cache the Newton Jacobian + LU factors; refactor only on step-change / Newton stall / order-change.
- [ ] Add an optional `jacobian(t, y, J)` derivative-side hook (constant `A` for linear blocks);
      default stays forward-difference.

## 5. Rosenbrock / TR family
- [ ] Port the runtime Rosenbrock `ode23s` against the `MflowLinkSim` Deriv/Jacobian callbacks.
- [ ] Add `ode23t` (trapezoidal) and `ode23tb` (TR-BDF2) on the same Jacobian + `solveDense`.
- [ ] Recognise the three algorithm names in `Loader.h`; remove the RK4 fall-through.

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
