## ODE / PDE Numerical Solvers

This page covers both the initial-value ODE solvers (`ode45`, `ode23`,
`ode23s`) and the 1-D parabolic-elliptic PDE solver (`pdepe`) that
sits on top of them via method-of-lines.

## Initial-Value ODE Solvers

`matlab_llvm` ships MATLAB-compatible initial-value problem (IVP)
solvers as runtime builtins. Three methods are first-class — no opt-in
flag, they land in the standard runtime alongside `eig` / `fft` /
`linspace`:

- **`ode45`** — Dormand–Prince 5(4). Default non-stiff solver.
- **`ode23`** — Bogacki–Shampine 3(2). Lower order, fewer stages.
- **`ode23s`** — Rosenbrock 2(3) (Shampine). **Stiff solver.** Uses one
  numerical-FD Jacobian per accepted step plus three linear solves; the
  implicit factor `(I − h·d·J)` absorbs stiff modes that would force
  tiny explicit steps. Use when `ode45` collapses to micro-steps or
  blows up on a stiff system.

## Quick start

Scalar `y` — the canonical "exponential decay with forcing":

```matlab
tspan = [0 10];
y0    = 1;
f     = @(t,y) -2*y + sin(t);
[t, y] = ode45(f, tspan, y0);
disp(y(end));        % -0.0498767  (analytic ≈ -0.04979)
```

Vector `y` — system of ODEs (linear oscillator):

```matlab
y0 = [1; 0];
[t, y] = ode45(@(t,yy) [(0 - yy(2)); yy(1)], [0 6.283185307179586], y0);
disp(y(end, 1));     % 0.999858  ≈ cos(2π)
disp(y(end, 2));     % 0.00108095 ≈ sin(2π)
```

`Y` is laid out N rows × D cols — `y(i, :)` is the state at `t(i)`.

Stiff system — Robertson reaction kinetics:

```matlab
f = @(t,y) [(0 - 0.04*y(1) + 1e4*y(2)*y(3));
            (0.04*y(1) - 1e4*y(2)*y(3) - 3e7*y(2)*y(2));
            (3e7*y(2)*y(2))];
[t, y, stats] = ode23s(f, [0 1], [1; 0; 0]);
disp(stats.nsteps);   % ~9 — ode45 would need thousands or diverge
```

## Call shapes

| Form | Notes |
|---|---|
| `[t, y] = ode45(@f, tspan, y0)` | Defaults: `RelTol = 1e-3`, `AbsTol = 1e-6`, `Refine = 4`. |
| `[t, y] = ode45(@f, tspan, y0, opts)` | `opts` is a struct built MATLAB-style (`opts.RelTol = 1e-9; …`). |
| `[t, y, stats] = ode45(@f, tspan, y0[, opts])` | `stats` is a struct with `nsteps` / `nfailed` / `nfevals`. |
| `tspan = [t0 tf]` | Adaptive grid + dense output (Refine sub-points per step). |
| `tspan = [t0 t1 … tN]` | Output at *exactly* those times via cubic-Hermite interpolation; `Refine` is ignored. |
| `tspan = [t1 t0]` with `t1 > t0` | Backward integration; the output grid runs from `t1` down to `t0`. |

`ode23` has the same surface; differences are method-internal (lower
order, fewer per-step function evaluations, Refine default = 1).

## odeset fields honoured

Build with field assignment, MATLAB-style:

```matlab
opts.RelTol      = 1e-9;
opts.AbsTol      = 1e-12;
opts.MaxStep     = 0.05;
opts.InitialStep = 1e-3;
opts.Refine      = 8;
opts.Stats       = 1;       % numeric flag, see deviation below
```

| Field | Effect | Default |
|---|---|---|
| `RelTol` | Per-component relative tolerance | `1e-3` |
| `AbsTol` | Per-component absolute tolerance | `1e-6` |
| `MaxStep` | Cap on adaptive step size (positive magnitude; backward direction is mirrored) | unbounded |
| `InitialStep` | Override the 1%-of-span heuristic for the first step | heuristic |
| `Refine` | Dense-output sub-points per accepted step (cubic Hermite) | `4` for `ode45`, `1` for `ode23` |
| `Stats` | Non-zero turns on the end-of-integration print (see deviation below) | off |

Other MATLAB `odeset` fields (`Events`, `OutputFcn`, `Jacobian`, `Mass`,
`NonNegative`, `NormControl`, …) are silently ignored — they currently
have no effect on the integration.

### Deviation: `Stats = 1`

MATLAB's canonical syntax is `opts.Stats = 'on'`. The frontend's
struct-set lowering doesn't yet pipe string values through
`matlab_struct_set_f64` cleanly, so we accept `opts.Stats = 1` as a
numeric flag instead. The print format (`N successful steps` /
`N failed attempts` / `N function evaluations`) matches MATLAB.

## Function-handle ABI

The user RHS is passed as an anonymous function (`@(t,y) …`). Two
flavours exist depending on the shape of `y0`:

| `y0` shape | Anon signature | Runtime ABI |
|---|---|---|
| Scalar (f64) | `@(t, y) -2*y + sin(t)` | `double f(double, double)` |
| Vector / matrix | `@(t, y) [-y(2); y(1)]` | `matlab_mat *f(double, matlab_mat *)` |

The compiler picks the path automatically. For vector `y` the
`LowerAnonCalls` pre-pass detects `ode45` / `ode23` sites with a
matrix-typed `y0` operand, traces the handle back to its `make_anon`,
and retypes the second block arg from `f64` to `ptr` before outlining.
The `LowerTensorOps` dispatch then routes to the `_v_*` runtime entries.

**Named-function handles with vector `y` are not supported.** A
`function dy = f(t,y) … end; ode45(@f, …)` form gets blocked by
`LowerUserCalls`'s signature-refinement gate, which rejects
`tensor<Nxf64>` ↔ `tensor<Nx1xf64>` shape mismatches. Use the anon form.

## Cache semantics

Each `[t, y] = ode45(...)` site lowers into paired `_t` / `_y` runtime
calls (and `_stats` for the 3-return form) sharing the same operands.
A thread-local cache slot keyed on `(handle, tspan, y0, tols, …)`
ensures only the first call actually integrates — the second and third
return the paired column / struct from the cache.

The cache is invalidated when any cache-key field changes (different
options, different y0, different RHS). Stats prints fire only on the
real-solve path, so calling with `Stats = 1` always prints exactly
once per integration.

## Events — root-finding during integration

Event detection lets the solver halt (or simply log) the moment a
user-supplied scalar function of `(t, y)` crosses zero. The wired
form is a 5-return builtin called `ode_events`:

```matlab
f   = @(t,y) -10;                 % constant velocity downward
y0  = 100;
evt = @(t,y) [y; 1; -1];          % [value; isterminal; direction]

[t, y, te, ye, ie] = ode_events(f, [0 20], y0, evt);
```

Semantics:

- `evt(t, y)` returns a 3×1 column `[value; isterminal; direction]`.
- A zero crossing of `value` triggers an event. `direction` filters:
  `+1` accepts only rising crossings, `-1` only falling, `0` either.
- `isterminal = 1` halts integration at the event; the integration
  arrays `t` and `y` end at the event point. `isterminal = 0` records
  the event and continues.
- `te`, `ye`, `ie` are the per-event time, state, and event-component
  index. `ie` is always `1` in v1 (single scalar event channel).
- The runtime uses bracket-then-bisect over the accepted DP45 step:
  cheap, robust, and within the step's local accuracy (Hermite-style
  refinement on `y` is good enough for the ball-drop class of tests).

### Deviation: non-MATLAB call shape

MATLAB takes events through `opts.Events = @evt` and reads the
`isterminal` / `direction` arrays from the event function's vector
return. Our struct-of-handles ABI is still TBD, so v1 exposes
events through a dedicated builtin `ode_events` rather than threading
them through `odeset`. Once the function-handle-in-struct ABI is
nailed down (same blocker as `OutputFcn` and `Mass`), `ode45` /
`ode23` / `ode23s` will route through the same event machinery via
`opts.Events`.

## Backend matrix

Bit-identical output across all three runtimes on the smoke ODEs.

| Backend | Status |
|---|:-:|
| `-emit-llvm` (and JIT — REPL / DAP) | ✅ |
| `-emit-c` / `-emit-cpp` | ✅ |
| `-emit-python` (the reference model) | ✅ |
| `-emit-typescript` | ✅ |
| `-emit-systemverilog` | n/a (out of HDL scope) |

## Examples

- [`examples/ode_solver.m`](../examples/ode_solver.m) — tour covering all
  call shapes (default tspan, odeset, user grid, backward, ode23 with
  MaxStep, 3-return stats, vector y).

## Tests

- [`test/Run/math_ode45_basic.m`](../test/Run/math_ode45_basic.m) —
  scalar `y`, the canonical user example.
- `math_ode45_dense.m` — Refine = 4 sample density and accuracy.
- `math_ode45_odeset.m` — `RelTol` / `AbsTol` round-trip vs analytic.
- `math_ode45_step_opts.m` — `MaxStep` and `InitialStep`.
- `math_ode45_refine.m` — `Refine = 1` / `4` / `8`.
- `math_ode45_backward.m` — backward-time round-trip.
- `math_ode45_user_grid.m` — `tspan = [t0 t1 … tN]` user grid.
- `math_ode45_stats.m` — `Stats = 1` summary.
- `math_ode45_three_return.m` — `[t, y, stats]` 3-return form.
- `math_ode45_vector.m` — system of ODEs via vector `y0`.
- `math_ode_events_ball.m` — ball-drop with terminal event,
  exercises bracket+bisect root-finding and integration halt.
- [`test/Runtime/test_ode.c`](../test/Runtime/test_ode.c) — 44 direct
  runtime checks (no JIT, no compiler frontend).

## 1-D Parabolic PDE — `pdepe`

MATLAB-compatible call shape:

```matlab
sol = pdepe(m, @pdefun, @icfun, @bcfun, xmesh, tspan)
```

Solves PDEs of the form

```
c(x, t, u, ∂u/∂x) ∂u/∂t = ∂/∂x f(x, t, u, ∂u/∂x) + s(x, t, u, ∂u/∂x)
```

on `xmesh = [a, x1, x2, …, b]` over `tspan`. The implementation is a
classic method-of-lines wrapper: spatial derivatives via finite
differences on the user mesh, the resulting interior ODE system handed
to `ode23s` (so stiff parabolic problems work without manual tuning).

### Quick start — 1-D heat equation

```matlab
% u_t = u_xx on [0, 1] with u(0,t) = u(1,t) = 0, u(x,0) = sin(πx).
% Analytic: u(x, t) = exp(-π²t) · sin(πx).

m       = 0;
pdefun  = @(x,t,u,dudx) [1; dudx; 0];          % c=1, f=du/dx, s=0
icfun   = @(x) sin(pi * x);
bcfun   = @(xl,ul,xr,ur,t) [ul; 0; ur; 0];     % Dirichlet zero

xmesh = linspace(0, 1, 21);
tspan = [0 0.1];
sol   = pdepe(m, pdefun, icfun, bcfun, xmesh, tspan);

% sol(end, 11) ≈ 0.3725 — analytic value at x = 0.5, t = 0.1.
```

### Scope

| | Status |
|---|:-:|
| `m = 0` (Cartesian) | ✅ |
| `m = 1` (cylindrical), `m = 2` (spherical) | ✅ — requires `xmesh(1) > 0` (axis-of-symmetry singularity not yet handled) |
| Scalar PDE (one component) | ✅ |
| System of PDEs (multi-component, MATLAB's `npde > 1`) | ❌ |
| Dirichlet BCs (`ql = qr = 0`) | ✅ |
| Neumann / Robin BCs (`ql ≠ 0`) | ✅ |
| Non-uniform mesh | ✅ — discretisation honours per-cell `dx` |
| Stiff parabolic problems (heat eq, diffusion) | ✅ — uses `ode23s` |
| `odeset` plumbed through to the time integrator | ❌ — uses `ode23s` defaults |

### How the BC is decoded

`bcfun` returns `[pl, ql, pr, qr]` such that `pl + ql·f = 0` at each
boundary (MATLAB's convention). The runtime keeps every mesh point in
the state vector and at each time-derivative call:

- **Dirichlet (`ql = 0`)**: snaps the boundary `u` to `g(t) = u_current − pl`
  (exploiting the standard linear form `pl = ul − g(t)`) and forces the
  boundary's `dU/dt = 0`. Re-snapped at each output time so any minor
  drift inside the integrator doesn't appear in `sol`.
- **Neumann / Robin (`ql ≠ 0`)**: derives the boundary flux as
  `f_boundary = −pl/ql` and uses it in a half-cell discretisation at
  the boundary node. Pure Neumann (`pl = 0, ql = 1`) gives the no-flux
  / insulated-wall condition; convective Robin (`pl = h(u − u_∞)`,
  `ql = 1`) and prescribed-flux Neumann all fall out of the same code
  path.

Mixed left/right BCs (Dirichlet at one end, Neumann at the other) are
handled.

### Cylindrical / spherical (`m = 1, 2`)

For non-Cartesian symmetries, the conservation form picks up `x^m`
factors:

```
c · ∂u/∂t = (1/x^m) · ∂/∂x [x^m · f(x, t, u, ∂u/∂x)] + s
```

The discretisation multiplies fluxes by `x^m` at midpoints and divides
the divergence by `x_i^m` at nodes (Skeel-Berzins integration). This
means `xmesh(1)` must be `> 0` — the axis-of-symmetry singularity at
`x = 0` (which MATLAB handles via L'Hôpital's rule + a Neumann zero
condition) is deferred to a follow-up.

Worked example (cylindrical Laplacian on annulus `r ∈ [1, 2]` with
Dirichlet `u(1) = 0, u(2) = 1`): the steady state `u(r) = log(r)/log(2)`
is recovered to ~2e-5 on a 21-point mesh.

### Output

Output `sol` is `N_t × N_x` — `sol(i, j)` is `u(t_i, x_j)`. MATLAB's
canonical 3-D layout `sol(t, x, k)` collapses to 2-D for our scalar-
PDE-only v1.

### What's next

`m ≠ 0`, Neumann/Robin BCs, and multi-component systems are the natural
follow-ups. Once `ode15s` lands (Phase 7.3), `pdepe` will route to it
for tighter tolerance on stiff problems.

## What's missing

Tracked separately in [`feature_status.md`](feature_status.md) and
[`roadmap.md`](roadmap.md). Highlights:

- Higher-order BDF stiff solver (`ode15s`) — `ode23s` is now shipped;
  `ode15s` is the next stiff-solver step (variable-order BDF + Newton
  iteration), more efficient than Rosenbrock on the very stiff end.
- Other stiff variants (`ode23t`, `ode23tb`, `ode15i` for DAEs).
- Non-stiff multistep (`ode113`) and high-order (`ode78`, `ode89`).
- BVP (`bvp4c`, `bvp5c`), DDE (`dde23`), DAE-with-mass-matrix.
- `Events` through `opts.Events` — the dedicated `ode_events` builtin
  ships today with bracket+bisect root-finding and the 5-return form;
  routing it through `odeset` is gated on the function-handle-in-
  struct ABI work that also unblocks `OutputFcn` / `Mass`.
- `OutputFcn` callback.
- Numerical PDE (`pdepe`, finite-element). With `ode23s` shipped, a
  method-of-lines `pdepe` wrapper becomes feasible; `ode15s` would make
  it more efficient on stiff parabolic problems. The shipped `pdsolve`
  family lives in the symbolic toolbox and returns closed-form
  expressions, not numerical grids — see [`sym.md`](sym.md).
- Vector `y` via *named* user functions (anon-only today; works for
  every solver).
