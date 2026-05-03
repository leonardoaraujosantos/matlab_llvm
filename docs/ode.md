## Initial-Value ODE Solvers

`matlab_llvm` ships MATLAB-compatible initial-value problem (IVP)
solvers as runtime builtins. Both `ode45` (Dormand–Prince 5(4)) and
`ode23` (Bogacki–Shampine 3(2)) are first-class — no opt-in flag, they
land in the standard runtime alongside `eig` / `fft` / `linspace`.

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
- [`test/Runtime/test_ode.c`](../test/Runtime/test_ode.c) — 44 direct
  runtime checks (no JIT, no compiler frontend).

## What's missing

Tracked separately in [`feature_status.md`](feature_status.md) and
[`roadmap.md`](roadmap.md). Highlights:

- Stiff solvers (`ode15s`, `ode23s`, `ode23t`, `ode23tb`, `ode15i`).
- Non-stiff multistep (`ode113`) and high-order (`ode78`, `ode89`).
- BVP (`bvp4c`, `bvp5c`), DDE (`dde23`), DAE-with-mass-matrix.
- `Events` (root-finding + 5-return form).
- `OutputFcn` callback.
- Numerical PDE (`pdepe`, finite-element). The shipped `pdsolve` family
  lives in the symbolic toolbox and returns closed-form expressions, not
  numerical grids — see [`sym.md`](sym.md).
- Vector `y` via *named* user functions (anon-only today).
