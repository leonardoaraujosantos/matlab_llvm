# examples/control/3d/

3-D-animated **inverted-pendulum** control demos. Each program designs a
stabilizing controller, integrates the **nonlinear** plant in the loop, and
records a keyframe timeline that exports to a self-contained Babylon.js HTML
player through the [`sim3d`](../../sim3d/README.md) command-line API.

These are the applied, animated counterparts to the text-only Control System
Toolbox tours in [`../`](../README.md): the same `lqr` / `place` / `ctrb` /
`obsv` / `ss` primitives, now stabilizing a body you can watch move.

## Run

Interpreted:

```sh
matlabc -repl < examples/control/3d/cartpole_lqr_3d.m
xdg-open cartpole_lqr_3d.html        # open the emitted player in a browser
```

Compiled (C or C++ — the runtime-handle object model backs `sim3d` and the
`ss` model objects in both lanes):

```sh
matlabc -emit-cpp examples/control/3d/cartpole_lqr_3d.m > cartpole_lqr_3d.cpp
c++ -std=c++20 -I runtime cartpole_lqr_3d.cpp build/libMatlabRuntime.a -lm -o cartpole_lqr_3d
./cartpole_lqr_3d                    # writes the same .html
```

Every program produces output identical across the interpreted and compiled
lanes (gated by `test/Differential/`), and writes one `<name>.html` next to the
working directory.

## The suite

Two plants × three controllers. The design model is **linearized about
upright**; the simulation that the animation shows is the **full nonlinear
plant** (the honest "design-linear, test-nonlinear" workflow). Full state is
assumed measured — no estimator.

| Plant | Controller | File | Emits |
|---|---|---|---|
| Cart-pole (single) | PID (classical) | `cartpole_pid_3d.m` | `cartpole_pid_3d.html` |
| Cart-pole (single) | LQR (state-space) | `cartpole_lqr_3d.m` | `cartpole_lqr_3d.html` |
| Cart-pole (single) | Pole placement | `cartpole_place_3d.m` | `cartpole_place_3d.html` |
| Double pendulum | PID / full-state PD | `double_pendulum_pid_3d.m` | `double_pendulum_pid_3d.html` |
| Double pendulum | LQR (state-space) | `double_pendulum_lqr_3d.m` | `double_pendulum_lqr_3d.html` |
| Double pendulum | Pole placement | `double_pendulum_place_3d.m` | `double_pendulum_place_3d.html` |

## Plant models

**Cart-pole** — a cart of mass `M` (single force input `u`) carrying a point
mass `m` on a massless rod of length `L`. State `X = [x; xdot; theta; thetadot]`
with `theta` measured from upright. Nonlinear EOM via the 2×2 mass matrix
`[M+m, m L cos; cos, L]`; open-loop has one real pole in the right half-plane
(the pole falls).

**Double inverted pendulum** — the cart carries a two-link chain of point
masses on massless rods. Absolute angles `th1, th2` from the upward vertical;
state `X = [x; th1; th2; xdot; th1dot; th2dot]` (6 states). Nonlinear EOM via
the 3×3 manipulator-form mass matrix `M(q) qddot = f(q, qdot, u)`. **Two**
unstable modes, **one** actuator — genuinely underactuated.

## Controllers

- **LQR** — `K = lqr(A, B, Q, R)` on the linearized `ss(A,B,C,D)` model, then
  `u = -K x` on the nonlinear plant. Penalise angle hard, cart lightly.
- **Pole placement** — confirm `rank(ctrb(A,B)) = n`, choose a stable pole set
  `P`, `K = place(A, B, P)`; the cart-pole demo also prints the `obsv` dual.
- **PID** — for the **single** cart-pole this is a genuine classical PID on the
  pole angle plus a slow cart-centering trim, hand-tuned (not synthesised from
  `(A,B)`). This is the canonical plant where PID shines.

### Honest note on double-pendulum "PID"

A single-input double inverted pendulum **cannot** be robustly stabilized by a
bank of independent PID loops — the two links fight each other through the one
shared input. `double_pendulum_pid_3d.m` therefore implements a best-effort
hand-tuned **full-state PD** (proportional + derivative on the cart and both
link angles, plus a small integral trim on the cart), which is effectively a
manually-chosen state-feedback gain. It is included so the PID / LQR /
pole-placement trio is complete for both plants, with this caveat stated in the
file header. For this plant, prefer the LQR or pole-placement versions.

## Scene construction

Each scene is a parented kinematic chain so links pivot about their **joints**,
not their centres:

```
cart (box, slides in x)
└── hub0 (pivot, rotates by th1)
    ├── link1 (thin box, length L1)
    └── hub1 (pivot, rotates by th2 − th1)   ← double pendulum only
        ├── link2 (thin box, length L2)
        └── bob   (tip mass)
```

Setting each hub's `Rotation = [0 pitch 0]` swings everything below it about the
joint. Cart position is the cart actor's `Translation`; one `world.run(Ts)` per
control step records the keyframe.

## Notes for the interpreter

- Build block matrices like `A = [zeros(3) eye(3); MG zeros(3)]` by
  **index-assignment** (`A = zeros(6); A(1:3,4:6) = eye(3); ...`) — 2×2 block
  *literals* do not concatenate reliably in `matlabc -repl`.
- Inside a matrix literal, separate columns with **commas** when an element is a
  parenthesised expression (`[0, (m1+m2)*g, 0]`, not `[0 (m1+m2)*g 0]`); a space
  immediately before `(` is mis-parsed.
- These programs use no local functions — `matlabc -repl` auto-prepends toolbox
  classdefs, so the dynamics are inlined per RK4 stage (the quadrotor-example
  convention).
