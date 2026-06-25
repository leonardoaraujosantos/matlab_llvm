## Context

The repo already proves the two halves of this demo independently:
`examples/quadrotor/quadrotor_pid_mpc_3d.m` drives a nonlinear plant through a `sim3d`
scene graph (parented rotor actors, RK4 integration, one keyframe per control step,
`sim3d.export`), and `examples/control/{place_pole_assignment,lqr_via_care}.m` exercise
the CST design builtins (`place`, `ctrb`, `obsv`, `lqr`, `ss`) on linearized pendulum
plants. This change composes those two patterns into the canonical cart-pole and
double-inverted-pendulum demos. Constraint: stay additive — examples only, no runtime
or viewer changes — and keep interpreted/compiled output identical (no wall-clock,
no RNG).

## Goals / Non-Goals

**Goals:**
- Six self-contained programs under `examples/control/3d/`, each runnable via
  `matlabc -repl` and as a compiled script, producing one Babylon.js HTML player.
- Controllers designed on a **linearized** model (`ss` + `lqr`/`place`) but validated
  by simulating the **nonlinear** plant — the honest "design-linear, test-nonlinear"
  controls workflow.
- A README tour consistent with `examples/control/README.md` conventions.

**Non-Goals:**
- No new builtins, language features, MPC, or estimator/observer (full state assumed
  measurable). No swing-up control — all examples start near upright and stabilize.
- No automated numerical regression harness for the HTML output; correctness is the
  printed convergence diagnostics plus a parse/run smoke check.

## Decisions

**Plant models.** Cart-pole: 4-state `[x, ẋ, θ, θ̇]`, single force input `u` on the
cart, standard nonlinear EOM with cart mass `M`, pole mass `m`, half-length `l`,
gravity `g`. Double pendulum: 6-state `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]`, single cart force,
two-link nonlinear EOM derived via the manipulator form `M(q)q̈ + C(q,q̇)q̇ + G(q) = Bu`.
Both integrated with **RK4** at a fixed sub-step, mirroring the quadrotor example. The
EOM are written inline (no symbolic dependency) so the file is self-contained.

**Controller design path.** Linearize about upright by hand (constant `A`, `B` matrices
written literally in the file, derived in the header comment), wrap in `ss`, then:
- LQR examples → `K = lqr(A, B, Q, R)` (continuous) and feed `u = -K·x`.
- Pole-placement → check `rank(ctrb(A,B))`, then `K = place(A, B, P)` for an explicit
  stable pole set; print `obsv` rank as the dual diagnostic.
- PID (cart-pole) → classical `u = Kp·θ + Ki·∫θ + Kd·θ̇` with a slow outer cart-centering
  term, hand-tuned. This is genuinely classical and stabilizes the single pendulum.

**Double-pendulum PID — honest framing.** A single-input double inverted pendulum has
two unstable modes and one actuator; independent PID loops cannot robustly stabilize it.
The PID example therefore implements a documented best-effort full-state PD (a weighted
sum of position and both link-angle errors expressed through PID-style gains) and says
so plainly in the header and README. Rationale: the user asked for all three controllers
per plant; rather than omit it or ship something that silently diverges, we ship a
working full-state PD and label it accurately. Alternative considered — omit double PID:
rejected, breaks the symmetry the user requested.

**Scene graph.** Reuse the quadrotor parenting pattern: ground `plane`, cart `box`,
pole(s) as `cylinder` actors with `setParent(cart)` and a local `Translation` to the
hinge, so setting the cart `Translation` and each pole `Rotation` per step animates the
whole assembly. The cylinder's local offset places its base at the hinge; rotation about
the scene's pitch axis swings it. Continuous design + nonlinear sim run on the same fixed
`Ts`; one `w.run(Ts)` per control step records the keyframe.

**Determinism.** Fixed initial conditions, fixed step, no `rand`/`tic` — guarantees the
interpreted and compiled runs and the emitted HTML are byte-identical, matching the
existing sim3d examples' contract.

## Risks / Trade-offs

- **Double-pendulum PID may look sluggish or marginal** → documented as best-effort;
  LQR/place are presented as the correct tools, PID as the classical-control contrast.
- **Linearized gains vs nonlinear plant could diverge for large initial tilt** →
  start from small perturbations (a few degrees) and pick `Q`/`R` and pole sets with
  comfortable stability margin; headers state the basin-of-attraction caveat.
- **Cylinder hinge offset geometry is easy to get visually wrong** → verify the pole
  base sits at the cart top in the exported scene before finalizing; keep dimensions
  in the header comment.
- **Drift from CST builtin signatures** → mirror exactly the calls already used in
  shipped `examples/control/*.m` (`lqr(A,B,Q,R)`, `place(A,B,P)`, `ctrb(A,B)`,
  `obsv(A,C)`, `ss(A,B,C,D)`); smoke-run each example after writing.

## Migration Plan

Pure addition. No rollback concerns: deleting `examples/control/3d/` and reverting the
two README edits fully removes the change. No effect on builds, tests, or other examples.

## Open Questions

- None blocking. Whether to also add a thumbnail PNG (as the quadrotor example did) is a
  nice-to-have deferred to implementation.
