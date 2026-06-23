# Quadrotor — symbolic EOM + cascade PID/MPC flight control

A complete engineering problem on the matlab_llvm stack: derive a
quadrotor's equations of motion symbolically, then fly a 3-D figure-8
with a cascade controller — **linear MPC** for horizontal position and
**four PIDs** for attitude and altitude — and plot the result.

```
        position ref (figure-8)
                |
                v
        +---------------+   theta_cmd, phi_cmd   +----------------+
        |  MPC (MIMO)   | ---------------------> |  PID x4        |
        |  x, y track   |                        |  roll/pitch/   |
        |  + preview    |                        |  yaw/altitude  |
        +---------------+                        +----------------+
                ^                                        |
                | x, y                                   | u1..u4
                |                                         v
                |                              +----------------------+
                +----------------------------- | 6-DOF nonlinear plant|
                       full state              | (RK4 @ 200 Hz)       |
                                               +----------------------+
```

## Files

| File | What it does |
| --- | --- |
| `quadrotor_derive_eom.m` | Symbolic Math Toolbox derivation of the 6-DOF EOM + hover linearization. Prints the nonlinear translational/rotational accelerations and the decoupled linear plant used by the controller. |
| `quadrotor_pid_mpc.m` | Full closed-loop simulation: MIMO MPC (position) + 4 PIDs (attitude/altitude) driving the nonlinear plant, with reference previewing. Renders a 4-panel plot to `quadrotor_pid_mpc.png`. |
| `quadrotor_fp_derive.m` | **Richer** first-principles symbolic derivation: builds the ZYX rotation matrix column by column, the full nonlinear translational + rotational (Euler) dynamics, the rotor→wrench mixing matrix, then the hover linearizations feeding *all three* control layers. |
| `quadrotor_fp_flight.m` | **Animated** three-layer cascade flight demo (outer MPC + inner PID + inner MPC) flying a climbing figure-8. Renders a static 4-panel summary **and** a cool tilting-body 3-D animation (MP4) you can watch the attitude loop work in. |
| [`mflowlink/`](mflowlink/) | **Block-diagram** (drag-and-drop) versions of the cascade controller as `.mflow` signal-flow models: `quadrotor_pid.mflow` (cascade PID) and `quadrotor_mpc.mflow` (MPC outer + attitude PID). Run on the mflowLink simulator in interpreted and compiled mode; regression-gated in CI. See [`mflowlink/README.md`](mflowlink/README.md). |

## Animated demo — `quadrotor_fp_*`

The `quadrotor_fp_derive.m` / `quadrotor_fp_flight.m` pair is the cooler,
animation-first take on the same problem, runnable end-to-end through the
**JIT/REPL** with plotting + FFmpeg enabled:

```sh
cmake -B build -DMATLAB_LLVM_WITH_PLOT=ON -DMATLAB_LLVM_WITH_PLOT_FFMPEG=ON
cat examples/quadrotor/quadrotor_fp_flight.m | build/matlabc -repl /dev/stdin
```

It writes two artifacts:

- `/tmp/quadrotor_fp_summary.png` — 3-D path (ref vs actual), position vs
  time, attitude-PID command-vs-actual tracking, and the control inputs.
- `/tmp/quadrotor_fp_flight.mp4` — the quadrotor body **rendered with its
  true attitude** (the front arm is red so you can see it bank into the
  turns), trailing its flight path, over a ground shadow and the faint
  ghost of the reference figure-8, with a slowly orbiting camera.

**Three control layers** (the user-requested "PID for attitude **and** MPC
on the inner loop"):

| Layer | Controller | Plant model | Output |
| --- | --- | --- | --- |
| Outer | linear **MPC** (2-in/2-out) | `x_ddot=g·θ`, `y_ddot=-g·φ` | tilt cmd `θ_cmd, φ_cmd` |
| Inner | **PID ×3** | `φ_ddot=u2/Ix`, `θ_ddot=u3/Iy`, `ψ_ddot=u4/Iz` | torques `u2,u3,u4` |
| Inner | linear **MPC** | `z_ddot=du1/m` | thrust trim `du1` → `u1=m·g+du1` |

So the inner loop is a **PID and an MPC working together** — PIDs hold the
attitude the outer MPC asks for, while a second MPC tracks altitude. The
reference is a *climbing/descending* figure-8 (`z = 1.5 + 0.6·sin(0.4t)`)
so the altitude MPC has real work to do. Closed-loop RMS position error is
**~0.097 m** over a 12 s flight (this build, REPL path, single-setpoint
look-ahead — the AOT `quadrotor_pid_mpc.m` with full preview-matrix MPC
gets to 0.014 m).

## Control architecture

**Plant (12 states):** `[x y z  xdot ydot zdot  phi theta psi  p q r]`,
inputs `u1` (collective thrust) and `u2,u3,u4` (body torques).

**Outer loop — linear MPC (MIMO, 2-in / 2-out).**
Linearizing the translational dynamics about hover gives
`x_ddot = g·theta`, `y_ddot = -g·phi`. The MPC controls the
double-integrator pair `[x xdot y ydot]` with the commanded tilt
`[theta_cmd; phi_cmd]` as its two manipulated variables, subject to a
`±0.35 rad` tilt bound and a `±0.10 rad` per-step rate bound. It uses
**reference previewing** (Tier-6) — the upcoming `p = 12` steps of the
figure-8 are fed in as a `p×2` matrix so the controller leads the turns.

**Inner loop — 4 PIDs @ 200 Hz.**
Roll/pitch/yaw track the MPC's tilt commands; a fourth PID holds
altitude. The attitude loops are tuned ~5× faster than the MPC
(`wn ≈ 25 rad/s`) so the cascade has a clean timescale separation.

**Plant integration.** The true nonlinear 6-DOF rigid-body model is
integrated with classical RK4 at 10 sub-steps per MPC tick.

## Results

Tracking a figure-8 (`x = 2·sin(0.6t)`, `y = sin(1.2t)`, `z = 1`) for
10 s, the closed loop holds an RMS position error of **~0.014 m**. The
4-panel plot shows the 3-D path, position vs. time, commanded vs. actual
tilt, and the inner-loop inputs (thrust settles at the hover value
`m·g = 9.81 N`).

### Tuning notes — what moves the RMS

The error breaks down into a startup transient plus a small steady-state
lag. Measured contributions:

| Change | Full-run RMS |
| --- | --- |
| Start from rest on the ground | 0.24 m |
| **+ reference previewing** (12-step preview matrix) | 0.07 m (steady) |
| **+ start at trajectory trim** (airborne, path velocity matched) | **0.014 m** |
| Push the MPC harder (`Wy = 5`, `Wdu = 0.05`) | diverges (6.8 m) |
| Longer horizon `p = 20` | 0.015 m (no gain) |

Takeaways:

1. **The startup transient dominates.** Starting from rest while the
   reference is already moving at 1.2 m/s (and the quad must climb to
   1 m) forces a hard catch-up that saturates the tilt bound and pushes
   the vehicle out of the small-angle regime where the MPC's linear
   model is valid. Handing off at the trajectory's trim state removes
   ~95% of the error. (Equivalently, ramp the reference in from the
   vehicle's actual rest state.)
2. **Reference previewing is the next-biggest lever** — feeding the MPC
   the upcoming `p` setpoints lets it lead the turns instead of lagging.
3. **Don't over-tune the MPC weights.** The cascade relies on the MPC
   being slower than the inner attitude loop; raising the output weight
   `Wy` or cutting move-suppression `Wdu` makes the MPC command tilt
   faster than the PIDs can follow, and the loop goes unstable. The
   default weights are already near the sweet spot for this inner-loop
   bandwidth — if you want a faster MPC, first make the attitude PIDs
   faster.
4. **Horizon length past ~0.6 s doesn't help** here, since previewing
   already captures the relevant reference dynamics.

Further gains (not applied, diminishing returns below 0.014 m): a tilt
feedforward `θ_ff = ẍ_ref/g`, a faster inner loop to unlock a more
aggressive MPC, or a smaller `Ts`.

## Building & running

The derivation script needs the Symbolic Math Toolbox; the simulation
needs the Cairo plot backend. Build `matlabc` with both:

```sh
cmake -DMATLAB_LLVM_WITH_SYM=ON -DMATLAB_LLVM_WITH_PLOT=ON ...
```

Then compile/run either script through the usual `matlabc -emit-llvm`
→ `clang++` → link-runtime pipeline (link `runtime/toolbox/sym` and
`runtime/plot` plus their `gmp`/`mpfr`/`cairo` deps), or via the JIT.

A numeric-only gating copy (no plots, no symbolic deps) lives at
`test/Run/mpc_t6_quadrotor.m` and runs in the standard test harness.

## Compiler notes

A few patterns in the simulation are written around current
type-inference limits:

- The plant derivative is **inlined** (4 RK4 stages) rather than placed
  in a helper, because indexing a value passed across a user-function
  boundary does not yet lower.
- The RK4 state update is written **element-wise** (`S(i) = S(i) + ...`)
  because a loop-carried whole-vector reassignment (`S = S + ...`) does
  not yet lower.
- Actuator saturation via `if u > lim, u = lim; end` is omitted: a
  conditional scalar reassignment erases the scalar's inferred type. The
  controller is tuned so the limits aren't hit on this trajectory.
- `fprintf` integer fields use `%g` rather than `%d` (the runtime passes
  a double, which C's `%d` prints as 0).

### `quadrotor_fp_derive.m` — symbolic workaround

- **No bare unary minus on a sym function call.** `simplify(-sin(theta))`
  segfaults (the unary-minus lowering only routes a *name* operand to the
  symbolic negation runtime, so `-sin(theta)` falls through to numeric
  negation on a sym pointer — issue **#241**). The R31 = `-sin(theta)` entry is
  therefore written `simplify(sym(0) - sin(theta))`. Note `-g*sin(phi)` is
  fine: it parses as `(-g)*sin(phi)`, i.e. unary minus on the *symbol* `g`.

Verified end-to-end against SymPP 0.5.0
(`-DMATLAB_LLVM_WITH_SYM=ON`, GMP 6.3.0 / MPFR 4.2.1).

### `quadrotor_fp_flight.m` — REPL/plotting workarounds

These are written around current bugs (issues filed) and are the reason a
few lines look the way they do:

- **VideoWriter setup is packed onto one physical line** (`v = VideoWriter(...);
  v.FrameRate = 25; open(v); figure(2);`). Splitting it across lines makes
  the REPL segfault inside `start_encoder` — the writer handle doesn't
  survive across REPL submissions (issue **#236**). The render loop body can
  still be one statement per line.
- **`plot3` is only ever called with vector arguments.** A scalar coordinate
  (`plot3(cx, cy, cz, ...)` or even `plot3([cx], ...)`, which collapses to a
  scalar) fails MLIR verification (issue **#237**). The single front-rotor
  marker is therefore drawn as a degenerate 2-point segment, and the
  animation loop starts at `k = 3` so the trail/shadow slices always have
  ≥ 2 points.
- **Axis limits use `xlim`/`ylim`/`zlim`.** These now resolve on **both** the
  REPL and AOT/`-emit-*` front doors (issue **#238**, fixed); `zlim` actually
  pins the 3-D z-axis. (To *run* and get the PNG/MP4, use `-repl` or the
  `-emit-llvm` → `clang++` → link pipeline — a bare `matlabc file.m` only
  type-checks/lowers.)
- **PNG-per-frame export is avoided.** `saveas(gcf, sprintf(...))` with a
  computed filename silently writes nothing — only string literals save
  (issue **#239**) — so animation goes through VideoWriter, not a frame dump.
- **Outer/inner MPC use a single-setpoint look-ahead, not a preview matrix.**
  The `p×ny` preview-matrix form of `mpcmove` used by `quadrotor_pid_mpc.m`
  errors on the JIT/REPL path ("unsupported call shape", and tracking then
  degrades to ~1.9 m — issue **#240**), so this REPL-runnable demo feeds each
  MPC the reference `look` steps ahead as a single setpoint vector instead.
