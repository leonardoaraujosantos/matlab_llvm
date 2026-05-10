# Control System Toolbox — Compatibility Roadmap

Scoped plan for what `matlab_llvm` (Runtime + Debug + REPL) needs to
ship in order to faithfully execute MATLAB Control System Toolbox
programs. Source: *Control System Toolbox User's Guide* (R2026a, 1982
pages, ~24 chapters across Linear System Modeling, Working with Linear
Models, Linear Analysis, Control Design, Control System Tuning,
Reliable Computations, and Examples).

The repo's overall compatibility target is a **practical numeric
subset** (see `feature_status.md`), so this doc inherits the same
posture: focus on the *programmable* surface (functions returning
arrays / structs / model objects), explicitly defer the GUI surface
(Control System Designer, Linear System Analyzer, Control System
Tuner, Model Reducer, PID Tuner, Compensator Editor, Linearizer apps),
the Simulink-coupled linearization path, and toolboxes that CST
*references but does not own* (Robust Control, System Identification,
Model Predictive Control, Adaptive Control, Resilient Control).

For shipped work, see [`feature_status.md`](feature_status.md). For
the cross-toolbox roadmap entries, see [`roadmap.md`](roadmap.md);
this doc is the per-toolbox companion, parallel to
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md).

---

## 0. Reading guide

- **Tier** = priority and dependency band, not strict order. Tier-1
  here is a **numeric-prerequisite tier** — almost no user-visible CST
  function lights up until those primitives land. Tier-2 is the first
  user-visible slice. This is different from the SPT roadmap, where
  Tier-1 was already user-visible, because CST sits on a heavier
  linear-algebra base.
- **Effort** is in the existing Phase 5.6.x cadence (one focused
  session ≈ a half-day; a "week" ≈ 5 sessions).
- **Status legend**: ✅ shipped · 🟡 partial · 🔵 not started ·
  🔴 deliberately deferred.
- **REPL / Debug** rows note display + DAP variable-inspector
  expectations. Unlike SPT (where outputs are mostly `matlab_mat *`
  and inherit the matrix display path), CST relies heavily on
  **structured model objects** (`tf`, `ss`, `zpk`, `frd`, `pid`).
  Those need new descriptor-aware rendering — flagged in §6.

---

## 1. Already shipped (Tier-0 baseline)

These are the matlab_llvm primitives that CST will sit on top of.
Locations are in `runtime/matlab_runtime.cpp` unless noted.

| Group | Functions / capabilities | Notes |
|---|---|---|
| OOP | `classdef`, single inheritance, `properties`, `methods`, constructors, static methods, operator overloading (`plus`, `minus`, `mtimes`, `mrdivide`, `eq`, …), `Dependent` properties with `get.Prop` / `set.Prop`, `enumeration` blocks | Phase 5 OOP arc. **All objects are handle-shaped today** (no value-class copy semantics) — relevant because MATLAB's `tf` / `ss` / `zpk` / `frd` are *value classes* in MATLAB. We can fake value semantics by making `+ - *` etc. always return fresh objects (CST functions never mutate in place). |
| Polynomial helpers | `roots`, `poly`, `polyder`, `polyint`, `polyint(p, k)`, `[r, p, k] = residue(b, a)` | All used by `tf` ↔ `ss` ↔ `zpk` conversions. `roots` uses Durand-Kerner so does not depend on `eig`. |
| Filter / DSP | `filter`, `filtfilt`, `sosfilt`, `freqz`, `impz`, `stepz`, `grpdelay`, FFT family | `freqz` is the *discrete* analog of `bode` for `tf`/`zpk` numerator/denominator vectors — useful as a fallback before the model-object machinery is in place. |
| ODE | `ode45`, `ode23`, `ode23s`, `ode_events`, `pdepe` | Stiff-capable. Underlies `lsim` / `initial` for non-LTI extensions; for LTI, closed-form `expm`-based simulation is preferred (Tier 1). |
| Linear algebra (scalar / dense) | `+ - .* ./ * /`, `\`, `inv`, `det`, `trace`, `transpose`, `'`, `chol` (sym), `lu`, `qr` (m≥n), `svd` (single-output σ), symmetric `eig` 1- and 2-return | **Critical gap for CST**: non-symmetric `eig` is currently *symmetrized* (`(A + A')/2`) — would give wrong poles for any non-self-adjoint plant. Must fix before any control workflow lights up. See Tier 1. |
| Complex | `conj`, `real`, `imag`, `angle`, complex `+ - .* ./ * /`, complex FFT | Required for `bode` / `nyquist` / `freqresp` (returns complex `H(jω)`). |
| Symbolic | SymPP-backed `sym` / `syms` with `laplace` / `ilaplace` etc. | Out-of-band convenience: a user can hand-derive `H(s)` symbolically then numericalize. CST does not consume `sym` directly — keep these lanes separate. |
| Containers | `struct`, struct arrays, `cell`, 2-D cell, `dictionary` / `containers.Map`, `table`, `datetime` / `duration`, `categorical` | `tf` / `ss` need a struct-or-classdef backing store; these all already work. Property metadata (e.g. `InputName` / `OutputName` / `TimeUnit`) maps onto cell-of-string + scalar string fields. |
| REPL / DAP | JIT REPL, workspace inspector, breakpoints, step, locals, `dbg(x)` | Will need new inspector entries for model objects (see §6). |

**What CST-specific code today**: zero. There are no `tf`, `ss`,
`zpk`, `bode`, `step`, `lsim`, `lyap`, `lqr`, `place`, `c2d` entries
in the runtime. The compatibility surface starts empty.

---

## 2. Tier 1 — numeric prerequisites (gates everything)

These are the linear-algebra primitives that virtually every CST
function calls into. Until they land, no user-visible CST function
can return correct answers in the general case. They are also useful
outside of CST (`expm` for matrix ODEs, `schur` for general eigen
problems, `lyap` for stochastic analysis, etc.) so the work is not
CST-specific deadweight.

This tier is roughly the size of the entire SPT Tier-1 + Tier-2 arc.
Plan it as such.

### 2.1 Full non-symmetric eigendecomposition ✅ (1+2-return shipped, real-eig path)

Today's `eig` symmetrizes `A` before diagonalizing — fast and
numerically clean for symmetric matrices, but **wrong for plants**.
A controller plant's `A` matrix is almost never symmetric.

**Scope**: real-Schur-form-based non-symmetric `eig` returning
complex eigenvalues / eigenvectors:
- `e = eig(A)` returns the (possibly complex) eigenvalues column
  vector.
- `[V, D] = eig(A)` returns `V * D = A * V` with complex `V`, `D`.
- Generalized: `eig(A, B)` (QZ algorithm) — gates `eig` of descriptor
  state-space models.

**Algorithm**: Hessenberg reduction (`hess`) → implicit shifted QR
with double-shift (Wilkinson) → real Schur form → eigenvalue extraction
from 1×1 / 2×2 diagonal blocks → eigenvector back-substitution. This
is "LAPACK `dgeev`" in spirit, hand-coded.

**Status**: 1-return `eig(A)` shipped — polymorphic real/complex
return via Francis double-shift implicit QR with deflation.
Symmetric inputs still take the Jacobi fast path. 2-return `[V, D] =
eig(A)` for non-symmetric A ✅ shipped via Schur-back-substitution
(real Schur form U' A U = T → solve T y_i = λ_i y_i for each
eigenvalue → v_i = U y_i, normalised). v1 path handles all-real
eigenvalues; complex conjugate pairs (2×2 quasi-blocks) need
complex back-substitution and return 0×0 today. The generalised
`eig(A, B)` would route through `qz` (already shipped 4-return for
B-invertible inputs) and is a small wrapper away.

**Effort**: ~2 sessions for the QR loop, ~1 session for eigenvector
back-substitution, ~1 session for QZ. ~1 week total.

**Gating tests** (additions to `test/Runtime/test_linalg.c`):
- 4×4 Jordan block with repeated complex eigenvalues
- companion matrix of a known polynomial: eigenvalues = roots
- well-known rotation matrix `[0 -1; 1 0]` → eigenvalues `±i`

### 2.2 Schur, Hessenberg, QZ 🟡 (schur + hess shipped, qz not)

Many CST routines (Lyapunov / Riccati solvers, balanced truncation)
use Schur or generalized Schur form directly, not as an internal step
of `eig`.

**Scope**:
- `[U, T] = schur(A)` real Schur form.
- `[U, T] = schur(A, 'complex')` complex Schur.
- `[H, P] = hess(A)`, `H = hess(A)` upper Hessenberg.
- `[AA, BB, Q, Z] = qz(A, B)` generalized Schur.

**Status**: `schur(A)` (1-return T) and `[U, T] = schur(A)` shipped
via the same Hessenberg + Francis-QR machinery as non-sym `eig`,
threading an orthogonal accumulator through both passes. `hess(A)`
(1-return) and `[H, P] = hess(A)` (2-return) shipped via the same
multi-return splitter pattern as `eig_V` / `eig_D`: `matlab_hess_H`
returns the upper Hessenberg form and `matlab_hess_P` returns the
orthogonal P with P' A P = H. `[AA, BB, Q, Z] = qz(A, B)`
✅ shipped (4-return splitter routing to `matlab_qz_{AA,BB,Q,Z}`).
v1 path: layered on `schur(B⁻¹·A)` + `qr(B·U)` — valid when B is
invertible. The singular-B path needs proper Hessenberg-Triangular
reduction + double-shift QZ iteration (Moler-Stewart 1973) and
returns 0×0 today; that's the gating piece for `zero(sys)` on the
Rosenbrock system matrix. Complex Schur is not separately needed
since the real form already exposes 1×1 / 2×2 blocks for complex
pairs.

**Effort**: 2-3 sessions on top of the Tier-2.1 QR machinery (most of
the work is shared).

### 2.3 Matrix exponential `expm` and inverse `logm` ✅ (both shipped)

`expm` is the **single most-called primitive in CST**. Used by:
- `c2d` zero-order-hold discretization (the canonical formula).
- `lsim` continuous-time exact simulation between samples.
- `initial` continuous-time evolution `x(t) = expm(A·t)·x0`.
- `lyapchol` and indirectly the time-domain branch of `gram`.
- LPV/LTV simulation (out of scope, but the building block reuses).

**Algorithm**: scaling-and-squaring with [13/13] Padé approximant
(Higham 2005). For `logm`, inverse scaling-and-squaring with Padé
approximant on `eye + (X − eye)`. Both need stable matrix solves; lean
on the existing LU.

**Effort**: ~1 week. `expm` alone unblocks the largest tier-2 chunk.

**Status**: `expm(A)` shipped via scaling-and-squaring with [13/13]
Padé (Higham 2005), bit-identical across all 5 emit lanes.
`logm(A)` ✅ shipped via Schur-then-Parlett-recurrence (Higham 2008
§11.4): real Schur T = U' A U, then F = log(T) computed with
F[i,i] = log(T[i,i]) on the diagonal and Parlett's commutativity
recurrence
`F[i,j] = (T[i,j](F[j,j]−F[i,i]) + Σ(T[i,k]F[k,j]−F[i,k]T[k,j])) /
(T[j,j]−T[i,i])` for the strict upper triangle, then log(A) =
U F U'. Returns 0×0 if the Schur form has 2×2 blocks (complex
eigenvalue pairs), non-positive diagonal, or repeated eigenvalues —
those failure paths await complex-arithmetic block log + Parlett's
block-form recurrence. Run-test `linalg_logm.m`.

**Gating tests**:
- `expm(zeros(n))` = `eye(n)`.
- `expm([0 1; -1 0]·t)` = rotation matrix.
- `expm(A + A')` symmetric path against eigendecomp.
- `logm(expm(A))` round-trip on stable `A`.

### 2.4 Lyapunov and Sylvester equations 🟡 (lyap + dlyap shipped)

**Scope**:
- `X = lyap(A, Q)` solves `A·X + X·A' + Q = 0` (continuous).
- `X = dlyap(A, Q)` solves `A·X·A' − X + Q = 0` (discrete).
- `X = lyap(A, B, C)` solves the Sylvester `A·X + X·B + C = 0`.
- `R = lyapchol(A, B)` Cholesky factor of the controllability gramian
  (used by balanced truncation).

**Algorithm**: Bartels-Stewart on top of Schur form. For `dlyap`,
the Smith iteration variant or Schur+back-substitution.

**Status**: `lyap(A, Q)` and `dlyap(A, Q)` shipped via vectorise +
dense LU on the n²·n² Kronecker matrix. `O(n^6)` cost; fine for
typical CST plants (n = 2..10). Bartels-Stewart on Schur form is the
proper large-plant follow-on (and `schur` is now shipped to gate it).
`lyapchol(A, B)` ✅ shipped — round-trip via `gram_c` then `chol`,
returns upper-triangular R with R'·R = Wc. The square-root
Hammarling solver (which avoids forming Wc explicitly) is the
proper large-plant follow-on. 3-arg Sylvester `lyap(A, B, C)` ✅
shipped via `matlab_sylvester(A, B, C)` solving A·X + X·B + C = 0
with the same vectorise + dense LU shape (size (n·m)² for A: n×n,
B: m×m); the dispatch table in `LowerTensorOps.cpp` maps `lyap`
arity-3 calls to this entry. Bartels-Stewart on Schur(A) and
Schur(B) is the large-plant follow-on. Run-tests
`linalg_lyapchol.m`, `linalg_sylvester.m`.

**Effort**: ~1 week, sits cleanly on Tier-2.2.

**Why this matters**: gates `gram`, `ctrb`/`obsv`-derived gramian
checks, balanced realizations, observer covariance propagation.

### 2.5 Algebraic Riccati equations ✅ (1+3+5-arg forms; icare / idare aliased)

**Scope**:
- `X = care(A, B, Q, R)` continuous-time algebraic Riccati
  `A'·X + X·A − X·B·R⁻¹·B'·X + Q = 0`. Returns the stabilizing solution
  `X` (1-return shipped). 3-return `[X, K, L] = care(A, B, Q, R)` ✅
  shipped — splits to `matlab_care` + `matlab_lqr` + `matlab_lqr_e`,
  giving the Riccati X, the LQ gain `K = R⁻¹B'X`, and the closed-loop
  spectrum `L = eig(A − B·K)` in one call. The 2-return `[X, K]` form
  drops L; useful when complex-poles handling isn't needed downstream.
- `X = dare(Ad, Bd, Q, R)` discrete analog ✅ (1-return + 3-return
  `[X, K, L]` shipped, K = `(R + B'XB)⁻¹B'XA`, L = `eig(Ad − Bd·K)`).
- 5-arg `care(A, B, Q, R, S)` ✅ shipped — Riccati with state-input
  cross term. Reduces to the 4-arg form by absorbing S into A_hat
  = A − B·R⁻¹·S' and Q_hat = Q − S·R⁻¹·S'. Same dispatch-table-
  multi-arity pattern as `lyap` (arity-2 vs arity-3 entries route
  to different runtime helpers). Same shape for `dare`, `icare`,
  `idare` 5-arg.
- 6-arg `care(A, B, Q, R, S, E)` (descriptor systems with
  generalised E·X·E' shape) — pending; needs the generalised
  Riccati QZ machinery on the symplectic pencil.
- `[X, K, L] = icare(...)` / `[X, K, L] = idare(...)` ✅ shipped as
  aliases to `care` / `dare` (1+5 arg). The numerical advantage
  shows up on ill-conditioned pencils; the proper Mehrmann-Voss
  structure-preserving QZ is the same generalised-Schur primitive
  that `zero(sys)` needs and ships together with the singular-B
  QZ path.

**Algorithm shipped**: `care` uses the matrix-sign Newton iteration
(Roberts 1980) on the Hamiltonian `H = [[A, -B R⁻¹ B']; [-Q, -A']]` —
no ordered Schur required. `dare` uses Newton-Kleinman iteration
(Hewer 1971) seeded from `X₀ = dlyap(Ad', Q)`; requires `Ad`
Schur-stable (typical after `c2d` of a damped continuous plant). The
direct symplectic-pencil approach via QZ is the textbook large-scale
algorithm; deferred until QZ is shipped.

**Why this matters**: gates `lqr` / `dlqr` (✅ shipped) / `lqi` /
`lqg` / `kalman` and the H₂ system `norm`. Without `care`/`dare` there
is no state-space optimal control.

### 2.6 Tier-1 closure summary

| Primitive | Effort | Status |
|---|---|---|
| Non-symmetric `eig` (2.1) | 1 wk | ✅ 1+2-return for real eigenvalues; complex pairs return 0×0 (deferred) |
| `schur` / `hess` / `qz` (2.2) | 0.5 wk | ✅ schur (1+2-ret), hess (1+2-ret), qz (4-ret, B-invertible path) |
| `expm` / `logm` (2.3) | 1 wk | ✅ both shipped (logm: real-Schur-then-Parlett, no 2×2 blocks) |
| `lyap` / `dlyap` / `lyapchol` (2.4) | 1 wk | ✅ all four shipped (lyap 2-arg, lyap 3-arg Sylvester, dlyap, lyapchol) |
| `care` / `dare` (2.5) | 1 wk | ✅ 1+3-return + 5-arg cross-term + icare/idare aliases |

**Tier-1 status**: numerical core complete enough to build the rest
of the toolbox. Logm, lyapchol, qz, the 2-return non-sym eig, and
generalised eig remain — all are individual follow-on slices, none
gating Tier-2/3/4 user-facing features that haven't already shipped.

**Total Tier-1**: ~4-5 weeks of focused sessions. Land in the order
above (each row depends on the rows above it).

**Demo-quality bypass**: a "skip Tier 1" path is possible — implement
`step` / `bode` / `c2d` directly via numerical ODE integration of the
state-space model and `polyval(num, jω) ./ polyval(den, jω)` for
transfer functions. This works for stable plants on bounded time
horizons but **silently returns garbage** for `lqr`, `place`,
`balred`, MIMO `norm`, anything that needs `expm`-accurate sampling,
or any unstable plant. Recommend against it; mention here only so the
trade-off is documented.

---

## 3. Tier 2 — minimum viable SISO control loop (~3 weeks)

This is the first user-visible CST slice: design a SISO controller,
simulate it, look at its frequency response. All Tier-2 items are
layered on Tier-1 primitives.

### 3.1 Model object constructors 🔵

MATLAB's `tf` / `ss` / `zpk` / `frd` / `pid` are value classes with
overloaded operators. Map them to matlab_llvm's `classdef`
(handle-shaped) and rely on the convention that arithmetic always
returns a fresh instance. No CST function in the practical subset
mutates a model object in place — value semantics fall out for free.

**Functions**:
- `tf(num, den)`, `tf(num, den, Ts)` — continuous + discrete.
- `tf(num, den, 'InputDelay', τ)`, `'OutputDelay'` — keep delays as a
  property; functions that ignore delays warn (matches MATLAB).
- `ss(A, B, C, D)`, `ss(A, B, C, D, Ts)`.
- `dss(A, B, C, D, E)` — descriptor (implicit) state-space; gated on
  Tier-1.2 QZ for analysis.
- `zpk(z, p, k)`, `zpk(z, p, k, Ts)`.
- `frd(response, freqs)`, `frd(response, freqs, Ts)`.
- `pid(Kp, Ki, Kd)`, `pid(Kp, Ki, Kd, Tf)`, `pidstd(Kp, Ti, Td)`.
- `pid2(Kp, Ki, Kd, Tf, b, c)` — 2-DOF PID.
- `tf('s')`, `tf('z', Ts)` — Laplace/Z handle for compositional
  syntax `s = tf('s'); G = 1/(s + 1)`.

**Internal storage**: each model is a classdef wrapping numeric
matrices in properties. `tf` stores cell-of-vector
`Numerator` / `Denominator` (so MIMO is a cell array). `ss` stores
`A` / `B` / `C` / `D` matrices. `zpk` stores `Z` / `P` / `K`. `pid`
stores the four/six gains. All carry `Ts` (0 = continuous), `TimeUnit`,
`InputName` / `OutputName`, `InputDelay` / `OutputDelay`,
`InternalDelay`, `Notes`, `UserData`.

**Why classdef and not opaque struct**: needed for `+`, `-`, `*`, `/`,
`'`, `==` operator overloads. The same machinery that makes `pid` work
makes `G + H`, `G * H`, `G / H` work — these are the canonical
composition idioms users expect.

**Operator overloads**:
| Operator | Semantics |
|---|---|
| `G + H` | parallel connection (sum) |
| `G - H` | parallel with sign flip |
| `G * H` | series (cascade) |
| `G \ H` | left-divide (rare; SISO-only practically) |
| `G / H` | right-divide |
| `G'` | Hermitian transpose (MIMO swap) |
| `inv(G)` | model inverse |
| `G == H` | structural equality on coefficients |

For mixed-type pairs (`tf + ss`, etc.), the rule MATLAB uses is the
"Recommended Working Representation" precedence ladder (§3.2 below).

**Effort**: 1 week. Most of the time is in coverage of all five
constructors plus the conversion table; the operator overloads are a
thin layer once conversions exist.

### 3.2 Conversions between types 🟡 (c2d ZOH shipped)

The CST UG dedicates an entire chapter (§5 Model Transformation) to
this. The five-way table (`tf` ↔ `ss` ↔ `zpk` ↔ `frd` ↔ `pid`) plus
continuous ↔ discrete is the connective tissue.

**Scope**:
- `tf(sys)`, `ss(sys)`, `zpk(sys)`, `frd(sys, freqs)`, `pid(sys)`
  — explicit form conversion.
- Implicit (auto) conversion under operator overloads, matching the
  MATLAB precedence: `frd` > `ss` > `zpk` > `tf` > `pid`.
- `c2d(sys, Ts, method)` — methods: `'zoh'` (default), `'foh'`,
  `'tustin'`, `'matched'`, `'impulse'`, `'least-squares'`. ZOH and
  Tustin are the two everyone uses.
- `d2c(sys, method)` — inverse of the above.
- `d2d(sys, Ts_new)` — resample a discrete model.
- `canon(sys, type)` — `'modal'`, `'companion'`. Builds canonical
  realizations.
- `ss2ss(sys, T)` — similarity transform.
- `balreal(sys)` — balanced realization (gates Tier-4 reduction).
- `prescale(sys)` — automatic state-space scaling (note: app-mode
  scaling is out of scope; the command-line variant is in scope).

**Algorithm notes**:
- `tf2ss` controllable canonical form is a direct construction;
  `ss2tf` for SISO is `det(sI − A) → C·adj(sI − A)·B + D·det(sI − A)`,
  evaluated via characteristic polynomial of `A` from `eig` /
  Faddeev-LeVerrier. **Numerically delicate** for high-order plants.
- `tf2zp` / `zp2tf` already exist (SPT); reuse.
- `c2d` ZOH: `Ad = expm(A·Ts)`, `Bd = ∫₀^Ts expm(A·τ)·B dτ`. The
  augmented-matrix trick (`expm([A B; 0 0]·Ts)`) gives both at once.
- `c2d` Tustin: bilinear `s = (2/Ts)·(z−1)/(z+1)` substitution into
  `A` / `B` / `C` / `D`. No `expm` needed.

**Status**: matrix-arg `c2d(A, B, Ts)` ZOH form shipped (Van Loan
augmented-matrix trick), 2-return splitter for `[Ad, Bd] = c2d(...)`.
Tustin / foh / impulse / matched-pole-zero / least-squares methods
are follow-ons. `d2c` (inverse direction) and `d2d` (resample) are
follow-ons. The `c2d(sys, Ts)` model-object form awaits §3.1.
`tf2ss` / `ss2tf` / `canon` / `ss2ss` follow §3.1 / §3.5 model
objects. `balreal` matrix-arg form shipped (Tier-4, see §5.1).

**Effort**: 1 week.

### 3.3 Time-domain simulation 🟡 (step_ss / lsim_ss shipped)

**Scope**:
- `[y, t, x] = step(sys, t)`, `step(sys)` (auto-time).
- `[y, t, x] = impulse(sys, t)`.
- `[y, t, x] = initial(sys, x0, t)`.
- `[y, t, x] = lsim(sys, u, t)`, `lsim(sys, u, t, x0)`.
- `S = stepinfo(y, t)` — overshoot, peak time, rise time, settling
  time. Reuses the SPT §4.3 `risetime` / `settlingtime` /
  `overshoot` / `undershoot` infrastructure.
- `S = lsiminfo(y, t)` — final value, peak, peak time, settling time.
- `gensig(type, period, T, Ts)` — pre-canned input signals (sine,
  square, pulse).
- `RespConfig` — initial conditions / amplitudes config object.

**Algorithm**:
- Continuous LTI: closed-form `x(t_{k+1}) = expm(A·dt)·x_k + ∫·B·u_k`.
  Same `expm` augmented-matrix trick as `c2d` ZOH.
- Discrete LTI: direct `x_{k+1} = A·x_k + B·u_k` recurrence. No
  primitives needed beyond matrix multiply.
- `step` / `impulse` are sugar over `lsim` with a unit step / unit
  impulse input.
- Auto-time: pick `T_final` from slowest pole; `Ts` from fastest pole
  (Nyquist-ish rule). Match MATLAB's heuristic loosely.

**Status**: matrix-arg `step_ss(A, B, C, D, dt, N)` and
`lsim_ss(A, B, C, D, u, dt)` shipped (ZOH discretisation +
recurrence; SISO step, MIMO lsim). `impulse` / `initial` / model-
object `step(sys)` / `stepinfo` / `lsiminfo` / `gensig` /
`RespConfig` are follow-ons.

**Effort**: 1 week. Bulk of the work is the auto-time heuristic and
making `stepinfo` / `lsiminfo` match MATLAB's settling-time
definitions.

**Gating tests**: SISO first-order (`tf(1, [τ 1])`), second-order
underdamped, MIMO 2×2 plant — verify against analytic answers.

### 3.4 Frequency-domain analysis 🟡 (bode_ss SISO + bode_tf + margins shipped)

**Scope**:
- `[mag, phase, w] = bode(sys, w)` / `bode(sys)`.
- `[mag, w] = bodemag(sys, w)`.
- `[re, im, w] = nyquist(sys, w)`.
- `[mag, phase, w] = nichols(sys, w)`.
- `[sv, w] = sigma(sys, w)` — singular values (MIMO H∞-relevant).
- `H = freqresp(sys, w)` — raw complex `H(jω)` evaluation.
- `evalfr(sys, f)` — evaluate at a specific complex `f`.
- `[Gm, Pm, Wcg, Wcp] = margin(sys)`.
- `S = allmargin(sys)` — all gain / phase / delay margins.
- `bandwidth(sys)` — −3 dB bandwidth.
- `getPeakGain(sys)` — peak ‖H(jω)‖.
- `getGainCrossover(sys, gain)`, `getPhaseCrossover(sys, phase)`.
- `dcgain(sys)` — limit `s → 0` (or `z → 1`).

**Algorithm**:
- For `tf`: `polyval(num, jω) ./ polyval(den, jω)` (already have
  `polyval`, complex arithmetic).
- For `ss`: solve `(jω·I − A)·X = B`, then `H = C·X + D`. One
  complex linear solve per frequency. For batches, factor `(jω·I − A)`
  via complex LU.
- For `zpk`: direct product form.
- `sigma` calls into complex SVD per frequency → needs complex SVD
  (currently only real σ-only) — minor extension.
- `margin`: find gain/phase crossovers via interpolation on the bode
  data; verify with high-density `freqresp` near crossovers for
  accuracy.

**Status**: matrix-arg `bode_ss(A, B, C, D, w)` (SISO; MIMO is a
follow-on) shipped via real 2n×2n decomposition of the complex
linear solve. `bode_tf(b, a, w)` shipped via complex Horner. 2-return
splitter for `[mag, phase] = bode_ss/bode_tf(...)` shipped.
`gain_margin(A,B,C,D,w)` and `phase_margin(A,B,C,D,w)` shipped (scan
the bode grid and interpolate the crossover; +Inf if no crossover).
`dcgain_ss(A, B, C, D)` shipped. `nyquist`, `nichols`, `sigma`
(needs MIMO bode + complex SVD), `bandwidth`, `getPeakGain`,
`allmargin` are follow-ons. Model-object forms `bode(sys)` etc.
follow §3.1.

**Effort**: 1.5 weeks. `bode` / `freqresp` are quick; `margin` /
`allmargin` need careful interpolation logic.

**Plotting**: this roadmap does **not** plan native plotting. `bode`
and friends return numeric `(mag, phase, w)` triples; users are
expected to dump them. A "no-display" mode (single-arg `bode(sys)`
that today would just plot) prints a small ASCII summary
("magnitude range [a, b] dB; phase range [c, d] deg over [w_min,
w_max] rad/s"). Phase-2 work could ship a hook into
`emit_python` to wrap matplotlib calls — out of scope here.

### 3.5 Pole / zero analysis 🟡 (isstable + damp shipped)

**Scope**:
- `p = pole(sys)` — closed-loop poles. Calls `eig(A)` for `ss`,
  `roots(den)` for `tf`.
- `z = zero(sys)` — transmission zeros. SISO: `roots(num)`. MIMO/`ss`:
  generalized eig of system matrix `[A B; C D]` vs `[I 0; 0 0]`
  pencil — needs Tier-1.2 QZ.
- `[wn, zeta, p] = damp(sys)` — natural freq, damping ratio per pole.
- `pzmap(sys)` — returns `(p, z)` (no plot).
- `iopzmap(sys)` — per I/O pair.
- `isstable(sys)` — boolean.
- `stabsep(sys)` — stable / unstable additive decomposition. Gates
  some Tier-4 reduction; stretch.

**Status**: matrix-arg `isstable(A)` shipped (continuous Hurwitz
check; marginally-stable returns 0 per MATLAB convention). `damp(A)`
shipped — 2-column `[wn, zeta]` form (MATLAB's full 4-column shape
`[wn, zeta, pole, time-const]` is a follow-on once we have a
4-return splitter). `pole(A) = eig(A)` is trivial via existing eig.
`zero(sys)` requires `qz` (generalised eig on the Rosenbrock matrix
`[A B; C D]` vs `[I 0; 0 0]`) and is gated on Tier-1.2 QZ.

**Effort**: 0.5 week (light, mostly thin wrappers).

### 3.6 Interconnections (SISO scope) 🔵

**Scope**:
- `series(sys1, sys2)` — same as `sys2 * sys1`.
- `parallel(sys1, sys2)` — same as `sys1 + sys2`.
- `[T, Ts] = feedback(sys1, sys2, sign)` — closed-loop. Default
  negative feedback. Returns the closed-loop `tf` / `ss` derived from
  `sys1 / (1 + sys2*sys1)` algebra.
- `append(sys1, sys2, ...)` — block-diagonal MIMO append (deferred to
  Tier-4 if MIMO connect lands then).

**Effort**: 0.5 week. Bulk is the feedback formula in state-space
(small Schur-complement-style algebra).

### 3.7 Discretization-aware response 🔵

**Scope**:
- `c2d` (covered in 3.2) — re-listed because it's the connective
  tissue between continuous design and discrete simulation. Most
  practical control work happens here.
- `dcgain`, `bandwidth`, `step`, `bode` for discrete sys. Already
  covered above; needs the `Ts ≠ 0` branch in each function.

### 3.8 PID design (basic) 🔵

**Scope**:
- `[C, info] = pidtune(sys, type)` — `type` ∈ {`'P'`, `'PI'`, `'PD'`,
  `'PID'`, `'PIDF'`, `'PI2'`, `'PD2'`, `'PID2'`, `'PIDF2'`}.
- `pidtune(sys, type, wc)` — target crossover frequency.
- `pidtuneOptions` — options struct (defer name-value initially).
- `getComponents(C2, ?)` — extract SISO components from a 2-DOF PID.

**Algorithm**: MATLAB's `pidtune` uses a robust frequency-response
loop-shaping approach (proprietary). A faithful but tractable
re-implementation: cross-over-driven Ziegler-Nichols / Astrom-Hagglund
tuning relations from the open-loop frequency response at the target
`wc`. Document divergence from MATLAB explicitly — `pidtune` outputs
will not be bit-identical.

**Effort**: 1 week if we stay with Astrom-Hagglund tuning; ≥3 weeks
if we want to chase MATLAB's actual H∞-flavored loop-shaping.

**Tier-2 closure status**: at the end of Tier 2, a user should be able
to:

```matlab
G = tf([1], [1 2 1]);          % plant
C = pidtune(G, 'PI');          % design
T = feedback(C*G, 1);          % close the loop
[y, t] = step(T, 0:0.01:10);   % simulate
[mag, phase, w] = bode(T);     % response
[Gm, Pm] = margin(C*G);        % margins
disp(stepinfo(y, t));          % overshoot / settling
```

Every line above lights up in Tier 2.

---

## 4. Tier 3 — state-space design (~3 weeks)

State-space optimal control. Sits cleanly on Tier-1.4 / 1.5
(Lyapunov / Riccati). Lights up the modern (post-1960s) control
workflow.

### 4.1 Linear-quadratic optimal control ✅ (1-return forms)

**Scope**:
- `K = lqr(A, B, Q, R)` — continuous LQR. Calls `care` ✅ (1-return
  shipped). `[K, S, P] = lqr(...)` 3-return form and 5-arg
  `lqr(A, B, Q, R, N)` cross-term are follow-ons.
- `K = dlqr(Ad, Bd, Q, R)` — discrete. Calls `dare` ✅ (1-return
  shipped).
- `[K, S, e] = lqry(sys, Q, R)` — output-weighted (`Q_x = C'·Q·C`,
  cross-term from `D`).
- `[K, S, e] = lqi(sys, Q, R)` — integral-action LQR (`A` augmented
  with integrator state).

**Effort**: 0.5 week (each is a thin wrapper over `care`/`dare`).

### 4.2 LQG and Kalman filter 🟡 (kalman_L + kalmd_L shipped)

**Scope**:
- `[kest, L, P] = kalman(sys, Q, R)` — continuous Kalman filter.
  Returns the estimator state-space model `kest` plus gain `L` and
  steady-state covariance `P`.
- `kalman(sys, Q, R, 'current')` / `'delayed'` — discrete time-update
  variants.
- `regulator = lqgreg(kest, K)` — assemble LQG regulator from
  estimator + LQ gain.
- `[reg, info] = lqg(sys, QXU, QWV)` — single-call LQG regulator
  synthesis. Convenience wrapper.
- `[reg, info] = lqgtrack(kest, K)` — tracking variant.

**Status**: matrix-arg `kalman_L(A, G, C, Qn, Rn)` and
`kalmd_L(Ad, G, C, Qn, Rn)` shipped — return just the steady-state
Kalman gain `L`. Implementation exploits LQR/Kalman duality:
`L = (lqr(A', C', G·Qn·G', Rn))'` (or `dlqr` for discrete). The
4-return shape `[kest, L, P] = kalman(sys, Q, R)` (estimator state-
space + gain + Riccati) and `lqgreg` / `lqg` / `lqgtrack` /
`current` / `delayed` variants are follow-ons.

**Effort**: 1 week (mainly bookkeeping; the heavy lifting is in
`care` / `dare`).

### 4.3 Pole placement 🟡 (SISO Ackermann shipped)

**Scope**:
- `K = place(A, B, p)` — multi-input pole placement (Kautsky-Nichols-
  Van Dooren robust algorithm).
- `K = acker(A, B, p)` — single-input Ackermann's formula. Convenient,
  but numerically poor for `n > 5`.
- `est = estim(sys, L, sensors, known)` — observer construction.

**Status**: matrix-arg `place(A, B, P)` shipped — SISO via
Ackermann's formula `K = [0…01]·ctrb⁻¹·α(A)`. Multi-input
Kautsky-Nichols-Van Dooren and the `estim` observer-construction
helper are follow-ons.

**Effort**: 1 week. Acker is half a session; `place` is the bulk
(Kautsky-Nichols algorithm is non-trivial — eigenstructure
assignment).

### 4.4 Controllability / observability / gramians ✅ (matrix-arg shipped)

**Scope**:
- `Cm = ctrb(sys)`, `ctrb(A, B)` — controllability matrix.
- `Om = obsv(sys)`, `obsv(A, C)` — observability matrix.
- `Wc = gram(sys, 'c')`, `Wo = gram(sys, 'o')` — gramians via Lyapunov
  (or Cholesky factor via `lyapchol` for numerical robustness).
- `[Abar, Bbar, Cbar, T, k] = ctrbf(sys)` — controllability staircase
  decomposition (gates `minreal`).
- `[Abar, Bbar, Cbar, T, k] = obsvf(sys)` — observability staircase.

**Status**: matrix-arg `ctrb(A, B)`, `obsv(A, C)`, `gram_c(A, B)`,
`gram_o(A, C)` all shipped. The structural-rank pair (`ctrb` /
`obsv`) and the energy-based pair (`gram_c` / `gram_o`) cover the
practical surface. `ctrbf` / `obsvf` staircase decompositions and
the model-object `gram(sys, 'c')` form are follow-ons.

**Effort**: 0.5 week.

### 4.5 System norms 🟡 (norm_h2 shipped)

**Scope**:
- `n = norm(sys)` — H₂ norm by default.
- `n = norm(sys, 2)` — explicit H₂ via `trace(B'·X·B)` where `X` solves
  the observability Lyapunov equation.
- `[n, fpeak] = norm(sys, Inf)` — H∞. Needs Boyd-Balakrishnan-Kabamba
  bisection on the Hamiltonian's purely-imaginary eigenvalues. Or
  Bruinsma-Steinbuch's two-step refinement.
- `[n, fpeak] = hinfnorm(sys, tol)` — explicit H∞ entry.

**Status**: matrix-arg `norm_h2(A, B, C)` shipped — `sqrt(trace(C ·
Wc · C'))` with `Wc = lyap(A, B B')`. Returns +Inf if A is not
Hurwitz. The H∞ norm (Boyd-Balakrishnan-Kabamba γ-bisection on
Hamiltonian eigenvalues, or Bruinsma-Steinbuch refinement) is a
follow-on.

**Effort**: 1 week — H∞ is the bulk; H₂ is a Lyapunov solve.

### 4.6 Stability / sensitivity analysis 🟡 (isstable shipped)

**Scope**:
- `[stabsys, unstabsys] = stabsep(sys)` — stable / unstable
  decomposition via ordered Schur. Gates Tier-4 reduction of unstable
  plants.
- `freqsep(sys, fc)` — slow / fast modal split.
- `[gain, info] = loopsens(sys)` — sensitivity / complementary
  sensitivity / loop-transfer / etc. (the "gang of six" or "gang of
  four").

**Status**: `isstable(A)` shipped (continuous Hurwitz check).
`stabsep` (ordered Schur stable/unstable split), `freqsep`, and
`loopsens` / `gangoffour` are follow-ons.

**Effort**: 1 week.

**Tier-3 closure status**: matrix-arg state-space optimal-control
workflow is now end-to-end usable. Users can do `K = lqr(A, B, Q, R);
L = kalman_L(A, G, C, Qn, Rn);` and assemble the LQG controller via
explicit state-space algebra. The model-object `lqgreg` / `lqg` and
the 3-return forms `[K, S, e] = lqr(...)` await §3.1 / multi-return
splitter follow-ons.

---

## 5. Tier 4 — model reduction & MIMO plumbing (~3 weeks)

CST's Chapter 6 (Model Simplification) and the MIMO connection
machinery. Useful once Tier-3 lights up because reduced-order
controllers and MIMO designs are the natural next step.

### 5.1 Model reduction 🟡 (balreal_T + balred_* + hsvd shipped)

**Scope**:
- `rsys = balred(sys, order)` — balanced truncation. Stable plants
  via Lyapunov gramians; unstable plants via `stabsep` first.
- `rsys = balreal(sys)` — balanced realization (already partly listed
  in 3.2 because `c2d`/`canon`/`balreal` are siblings).
- `rsys = modred(sys, elim, method)` — modal residualization
  (`'MatchDC'`) or truncation (`'Truncate'`).
- `[hsv, baldata] = hsvd(sys)` — Hankel singular values.
- `rsys = minreal(sys, tol)` — minimal realization (pole-zero
  cancellation). For `ss`: staircase-based via `ctrbf` / `obsvf`.
  For `tf` / `zpk`: tolerance-based pole-zero cancellation.
- `rsys = sminreal(sys)` — structural minimality (no numerical
  cancellation; faster).
- `R = reducespec(sys, method)` — task-based reduction object;
  defer (it is a wrapper over the above).

**Status**: matrix-arg `balreal_T(A, B, C)` shipped (Laub 1980
eigendecomposition variant — sym-eig + lyap stack; no Cholesky).
`balred_A(A, B, C, k)` / `balred_B` / `balred_C` shipped for k-state
balanced truncation; H∞ error bound `2·sum(HSV[k+1:n])`. `hsvd(A,
B, C)` shipped (sqrt(eig(Wc · Wo)) sorted descending). The full
4-return `[Ar, Br, Cr, hsv] = balreal/balred(...)` shapes need a
multi-return splitter; `modred` (modal residualization), `minreal`,
`sminreal`, and `reducespec` are follow-ons. Stable/unstable
pre-split via `stabsep` is the gating piece for unstable plants.

**Effort**: 2 weeks. `balred` is the bulk because the gramian path
plus stable/unstable decomposition is a multi-piece pipeline.

### 5.2 MIMO connection plumbing 🔵

**Scope**:
- `sys = connect(blocks, inputs, outputs)` — graph-style MIMO
  assembly.
- `S = sumblk('e = r - y', size)` — symbolic summation block (parses
  the equation string into a `ss` summing junction).
- `sys = append(sys1, sys2, ...)` — block-diagonal append.
- `sys = lft(sys1, sys2, nu, ny)` — linear fractional transformation.
- `sys = blkdiag(sys1, sys2)` — alias of `append`.

**Algorithm**: `connect` is the only non-trivial one — parse the
graph, identify summing junctions, build the closed-loop `ss` model
from concatenated `A` / `B` / `C` / `D` blocks plus the connection
matrix. Bookkeeping-heavy but no exotic primitives.

**Effort**: 1 week.

### 5.3 Time-delay handling 🔵

**Scope**:
- Internal delay representation on `ss` / `tf` (the property exists at
  Tier 2, but functions ignore it).
- `pade(τ, n)`, `pade(sys, n)` — Padé approximation (rational
  approximation of `e^{-τ s}`).
- `absorbDelay(sys)` — bake a discrete-time delay into extra states.
- `delayss(...)` — descriptor model with explicit delays.
- `thiran(τ, n)` — fractional-delay all-pass (signal-processing
  cousin).

**Effort**: 1 week. `pade` is a closed-form Padé recurrence; the
descriptor and delay-state machinery is the bulk.

**Tier-4 closure status**: a user can build a MIMO plant from blocks,
reduce it, design a controller, close the loop, and simulate — the
full classical-and-modern control workflow.

---

## 6. Tier 5 — passivity, sectors, advanced analysis (stretch)

Smaller-volume but conceptually important. Mostly thin layers over
Tier-1 / Tier-3 primitives.

### 6.1 Passivity and sectors 🔵

- `[ifc, dfc] = passivityplot(sys)` (numeric form: `getPassiveIndex`).
- `R = isPassive(sys)`.
- `[s, info] = getPassiveIndex(sys, type)`.
- `R = isParallelPassive(sys1, sys2)`.
- `[s, info] = getSectorIndex(sys, Q)` — sector-bounded analysis.
- `isPassive` for series / feedback / parallel interconnections.

**Effort**: 1 week. Underlying KYP-lemma test reduces to LMI
feasibility / Hamiltonian eigenvalue check on the imaginary axis (~
H∞ machinery).

### 6.2 Sensitivity and gang-of-N 🔵

- `[gang, info] = loopsens(P, C)` — sensitivity / complementary
  sensitivity / etc.
- `[L, P, T, S] = gangoffour(P, C)`.
- Already listed in 4.6; expand with plot-data-returning shape here.

### 6.3 Sparse-aware tail (large-scale, deferred) 🔴

CST §1 / §3 / §6 cover a *sparse* model branch (`sparss` / `mechss`)
for FE/PDE-derived structural models with up to ~10⁵ states. This
needs a sparse linear algebra stack (`sparse` matrix type, sparse LU,
sparse Lanczos). Out of scope unless a user demand surfaces; the
dense Tier-1–4 stack covers the practical control-of-mechatronics
surface. **Carved out** — see §10.

---

## 7. Tier 6 — control system tuning (heavy; deferred)

CST chapters 14–17 are about `systune` / `looptune` / `hinfstruct`
— **fixed-structure H∞ tuning**. These are the modern equivalent of
"shape your loop transfer function so it meets ten constraints
simultaneously". Unlike Tier-3 LQR, where one Riccati solve gives the
answer, `systune` is a **non-smooth nonlinear optimizer** over a
parameterized controller structure.

**Why this is heavy**:
- Requires a non-smooth optimizer (mainly an interior-point /
  trust-region method specialized for max-of-max objectives —
  Apkarian-Noll's `HIFOO`-style approach).
- Requires a `TuningGoal.*` constraint algebra (a dozen constraint
  types that combine into a single non-smooth objective).
- Requires `genss` / `genmat` (generalized models with named tunable
  blocks — `realp`, `tunablePID`, `tunableSS`, etc.) — a substantial
  symbolic-numeric data-flow layer.
- Requires named-block analysis (`getIOTransfer`, `getLoopTransfer`,
  `getSensitivity`, `getCompSensitivity`).

**Recommendation**: defer entirely. If tuning is requested, ship
`pidtune` (Tier-2 §3.8) and `hinfsyn` (Robust Control Toolbox; out of
scope here) before attempting `systune`. **Carved out** — see §10.

---

## 8. REPL / Debug-side work (cross-cutting)

Unlike SPT — where most outputs are `matlab_mat *` and inherit the
matrix display path — CST relies on **structured model objects**.
The REPL needs to render them; the DAP variable inspector needs to
expand them.

### 8.1 Model-object display

`disp(G)` for a `tf` / `ss` / `zpk` model has a canonical
multi-line formatted output:

```
G(s) =

         s + 2
  -------------------
   s^2 + 3 s + 5

Continuous-time transfer function.
```

For `ss`:
```
A =
       x1     x2
  x1   -1     1
  x2    0    -2

B =
       u1
  x1    0
  x2    1

C =
       x1   x2
   y1   1    0

D =
       u1
   y1   0

Continuous-time state-space model.
```

`zpk`:
```
              (s+2)
  K = 1 ----------
            (s+1)(s+3)

Continuous-time zero/pole/gain model.
```

Implementation: a `disp` method on each classdef calling into a
runtime helper `matlab_cst_disp_<type>` that does the formatting.
~0.5 week per type, batchable.

### 8.2 DAP variable inspector

For each model type, expose its top-level properties as expandable
children in the Locals panel:
- `tf`: `Numerator` / `Denominator` (cell-of-vector), `Variable`,
  `Ts`, `IODelay`, `InputDelay`, `OutputDelay`, `InternalDelay`,
  `InputName` / `OutputName`, `TimeUnit`, `Notes`, `UserData`.
- `ss`: `A` / `B` / `C` / `D` / `E`, `StateName`, `StateUnit`, plus
  the IO/time properties above.
- `zpk`: `Z` / `P` / `K`, plus the IO/time properties.
- `frd`: `ResponseData`, `Frequency`, `FrequencyUnit`, plus IO/time.
- `pid`: `Kp` / `Ki` / `Kd` / `Tf` / `b` / `c`, `Form`,
  `IFormula`, `DFormula`, plus IO/time.

Each child should render its underlying `matlab_mat *` (or scalar)
using the existing matrix renderer. ~1 session per type once the
inspector hook for classdef objects is in place — much of which the
existing Phase 5 OOP rendering already covers.

### 8.3 REPL JIT considerations

CST functions returning model objects are pure (no in-place mutation).
The existing JIT REPL caching strategy works; no new issues expected.
The one wrinkle: `tf('s')` / `tf('z')` is idiomatic
"variable-builder" syntax. Make sure the resulting handle survives
across REPL turns and composes correctly under operator overloads.

---

## 9. Suggested execution order

If user demand drives the order, expect this rough sequence (each row
unblocks the next; durations are focused-session estimates):

| Order | What | Effort | Status |
|---|---|---|---|
| 1 | Non-symmetric `eig` (Tier 1.1) | 1 wk | ✅ shipped (1-return) |
| 2 | `expm` / `logm` (Tier 1.3) | 1 wk | 🟡 expm ✅, logm pending |
| 3 | `schur` / `hess` / `qz` (Tier 1.2) | 0.5 wk | 🟡 schur ✅ + hess ✅, qz pending |
| 4 | `lyap` / `dlyap` / `lyapchol` (Tier 1.4) | 1 wk | 🟡 lyap ✅ + dlyap ✅, lyapchol pending |
| 5 | `care` / `dare` (Tier 1.5) | 1 wk | ✅ care + dare shipped (1-return + 3-return `[X, K, L]`) |
| 6 | Model object constructors (Tier 2.1) — `tf` / `ss` / `zpk` / `pid` | 1 wk | 🔵 not started — single biggest UX gap |
| 7 | Conversions + `c2d` / `d2c` (Tier 2.2) | 1 wk | 🟡 c2d ZOH (matrix-arg) shipped; d2c / Tustin / foh pending |
| 8 | Time-domain simulation (Tier 2.3) — `step` / `impulse` / `lsim` / `initial` / `stepinfo` | 1 wk | 🟡 step_ss + lsim_ss (matrix-arg) shipped; impulse / initial / stepinfo pending |
| 9 | Frequency-domain (Tier 2.4) — `bode` / `nyquist` / `sigma` / `freqresp` / `margin` | 1.5 wk | 🟡 bode_ss SISO + bode_tf + gain/phase margins shipped; MIMO bode + sigma + nyquist pending |
| 10 | Pole/zero + interconnections (Tier 2.5–2.6) | 1 wk | 🟡 isstable + damp shipped; zero + feedback / series / parallel pending |
| 11 | `pidtune` (Tier 2.8) | 1 wk | 🔵 pending (needs H∞ for MATLAB-faithful) |
| 12 | LQR + LQG + Kalman + place (Tier 3.1–3.3) | 1.5 wk | 🟡 lqr ✅ + dlqr ✅ + place SISO ✅ + kalman_L ✅ + kalmd_L ✅; lqgreg / lqg / 4-return [kest, L, P] pending |
| 13 | Gramians + ctrb/obsv + norms (Tier 3.4–3.5) | 1 wk | 🟡 ctrb + obsv + gram_c + gram_o + hsvd + norm_h2 + dcgain_ss shipped; norm_inf (H∞) pending |
| 14 | Model reduction (Tier 4.1) | 2 wk | 🟡 balreal_T + balred_{A,B,C} shipped; minreal / sminreal / modred / stabsep pending |
| 15 | MIMO `connect` / `sumblk` (Tier 4.2) | 1 wk | 🔵 pending (waits §3.1 model objects) |
| 16 | Padé / time delays (Tier 4.3) | 1 wk | 🔵 pending |
| 17 | Passivity / sectors (Tier 5.1) | 1 wk | 🔵 pending |

**Total**: ~18-20 weeks of focused sessions for Tier 1 → Tier 4
closure. Tier 5 is +1 week. Tier 6 (`systune`) is multi-month.

**Current state (2026-05-09)**: rows 1-5, 7, 8, 9, 12, 13, 14 all
🟡 partial-shipped via matrix-arg primitives. The big remaining
unlock is **row 6 (model objects)** — once `tf` / `ss` / `zpk`
classdefs land with operator overloads, the model-object forms of
rows 7–14 become 5-line wrappers each, and rows 11 / 15 become
reachable.

**MVP slice (~11 weeks)**: Order 1-11. Lights up the "PID design and
simulate" loop end-to-end. Most pedagogical control problems fit here.

---

## 10. Out of scope (carved out, by chapter / topic)

| Chapter / topic | What | Why out of scope |
|---|---|---|
| Throughout | All apps — Control System Designer, Linear System Analyzer, Control System Tuner, Model Reducer, PID Tuner, Compensator Editor, Linearizer | Interactive Qt apps; not a language feature |
| §1, §22 | Simulink linearization (`linearize`, `slLinearizer`, `slTuner`, batch trim/linearize) | Simulink is not in scope |
| §1 | LPV / LTV runtime simulation (`lpvss`, `ltvss`, scheduling-parameter functions) | Heavy time-varying state machinery; carve out unless a user requests gain-scheduling |
| §1, §6 | Sparse second-order models (`sparss`, `mechss`, sparse `linearize`) | Needs full sparse linear algebra stack (sparse LU, Lanczos); large-scale FE/PDE structural surface |
| §11 | Adaptive Control workflows | Adaptive Control Toolbox; separate product |
| §13 (parts) | System Identification entries (`idLTI`, `n4sid`, `tfest`, `ssest`) | System Identification Toolbox; separate product |
| §13 (parts) | Resilient / cyber-physical security examples | Reference applications, not CST primitives |
| §14–§17 | `systune` / `looptune` / `hinfstruct` / `TuningGoal.*` / `genss` / `realp` / `tunable*` | Non-smooth nonlinear optimization machinery; multi-month effort. Defer unless requested |
| §14 (parts) | H∞ synthesis (`hinfsyn`, `mixsyn`, `mu-tools`) | Robust Control Toolbox; separate product |
| §11 (MPC parts) | Model Predictive Control entries | MPC Toolbox; separate product |
| §22 | Interactive Scaling Tool (`prescale` GUI) | App; command-line `prescale` (no plot) is in scope |
| Plot customization | `bodeoptions`, `pzoptions`, plot tools, right-click menus | We do not ship native plotting. Functions return numeric data; visualization is delegated to the user (Python emit lane → matplotlib is a possible follow-up, not committed) |

---

## 11. Test corpus deltas

Mirror the SPT layout under `test/Run/` and `test/Runtime/`:

| Tier | New `test/Run/*.m` (rough count) | New `test/Runtime/test_*.c` |
|---|---|---|
| Tier 1 (linalg primitives) | ~6 (`linalg_eig_nonsym`, `linalg_expm`, `linalg_lyap`, `linalg_care`, `linalg_schur`, `linalg_qz`) | new `test/Runtime/test_linalg2.c` extending the existing one |
| Tier 2 (basic SISO loop) | ~8 (`ctrl_tf_basic`, `ctrl_ss_basic`, `ctrl_zpk_basic`, `ctrl_pid_basic`, `ctrl_step`, `ctrl_bode`, `ctrl_margin`, `ctrl_pidtune`) | new `test/Runtime/test_control.c` |
| Tier 3 (state-space design) | ~6 (`ctrl_lqr`, `ctrl_lqg`, `ctrl_kalman`, `ctrl_place`, `ctrl_gram`, `ctrl_norm`) | extend `test_control.c` |
| Tier 4 (reduction + MIMO) | ~5 (`ctrl_balred`, `ctrl_minreal`, `ctrl_connect`, `ctrl_sumblk`, `ctrl_pade`) | extend `test_control.c` |
| Tier 5 (passivity) | ~2 (`ctrl_passive`, `ctrl_loopsens`) | extend |

C / C++ / Python / TypeScript lanes must remain byte-identical, with
the same `.stdout-python` override convention SPT uses for numpy
bracket repr (matrix returns from CST will trigger this on the Python
lane).

**Display gating**: `disp(G)` for model objects produces multi-line
formatted output that must be byte-stable across lanes. The C lane is
canonical; TS and Python override files only land if the formatting
diverges (e.g., number-formatting precision differences). Plan for
overrides; they are easier to add than to retrofit.

---

## 12. Summary

CST compatibility is a **two-stage** project:

**Stage 1 (Tier 1, ~5 weeks) — ✅ SHIPPED**: numeric-prerequisite stack —
non-symmetric `eig` (1-return), `hess`, `schur`, `expm`, `lyap`/`dlyap`,
`care`/`dare`. Useful well beyond CST. With this in place, CST became a
relatively conventional descriptor + numerical-method layer on top.

**Stage 2 (Tiers 2–4) — ✅ matrix-arg form SHIPPED**: SISO design loop
(`lqr`/`dlqr` with 3-return `[K, S, e]`, `place`, `kalman`/`kalmd` with
2-return `[L, P]`); discretization (`c2d` ZOH, `c2d_tustin`,
`d2c_tustin`); time-domain (`step_ss`, `lsim_ss`, `stepinfo`);
frequency-domain (`bode_ss` SISO, `bode_tf`, gain/phase margins,
`bandwidth_ss`, `getPeakGain_ss`); state-space utilities (`ctrb`,
`obsv`, `gram_c`, `gram_o`, `isstable`/`isstable_d`, `damp`, `hsvd`,
`pole`, `dcgain_ss`, `norm_h2`/`norm_h2_d`); model reduction
(`balreal_T`, `balred` with 3-return `[Ar, Br, Cr]`); interconnection
(`feedback_ss`, `series_ss`, `parallel_ss`, `append_ss` — all matrix-
arg, strictly-proper, 3-return).

**Stage 3 (model objects) — 🟡 PARTIAL**: `tf` ships with
constructor + property reads + tf-vs-tf operator overloads
(`+`, `-`, `*`, `/`, unary `-`) + **scalar mixing**
(`G + 2`, `5 * G`, `G - 1`, `3 + G`, `G / 2`, …) + the **`s = tf
([1 0], 1)`** Laplace-variable composition idiom: write `J = (s +
2) / (s * s + 3 * s + 5)` and get `[1, 2] / [1, 3, 5]`. The
`tf('s')` char-literal sugar is a follow-on (char literals don't
survive the AST→MLIR lowering through the constructor body
today). `ss` / `zpk` / `frd` / `pid` classdefs are the remaining
shape — mechanical replicas of `tf` once the operator-overload +
scalar-mixing pattern is settled.

**Stage 3 — `tf` working with operator overloads (2026-05-09)**:
`tf(num, den)` constructor with `obj.Numerator` / `obj.Denominator`
read access AND tf-vs-tf operator overloads `+ - * / -` (binary
plus / minus / mtimes / mrdivide and unary minus) all ship in
`runtime/cst_classdefs.m`, auto-prepended by matlabc when
the user input mentions any of `tf` / `ss` / `zpk` / `pid` / `frd`
as a call target (see `tools/matlabc/main.cpp` `findCstPrelude` +
`userMentionsCstClass`). Several compiler fixes landed to make this
work:

- `LowerUserCalls.cpp` `runMonomorphiseUserCalls` and
  `runLowerUserCalls`: skip tensor-typed call-site args from
  bucketing / signature refinement *for class methods only*
  (gated by the `matlab.class_name` op attr). Tensor literals at
  call sites get boxed to ptr by a later `LowerTensorOps` sweep;
  refining the constructor sig to tensor early would lock in a
  signature that no post-boxing call site matches and leave
  stranded `matlab.call`s to a tensor-signature function.
  Non-class user functions keep the existing per-shape clone
  behaviour, so `sq([1 2 3])` and `sq(5)` still get separate clones
  with correctly-typed arithmetic bodies.
- `LowerTensorOps.cpp` field-set dispatch: when the AST-time choice
  of `_set_f64` callee turns out to receive a ptr-typed value at
  lowering time (because the function arg got refined to ptr by
  the LowerUserCalls retype), auto-promote the callee to `_set_mat`
  in place. The runtime `_set_mat` accepts ptr; `_set_f64` boxes a
  scalar — they're interchangeable on the type axis.
- `LowerTensorOps.cpp` `fieldNameAddr`: also accept
  `LLVM::AddressOfOp` as a valid input (not just
  `matlab.const_char`), so a second LowerTensorOps sweep can still
  resolve the field-name pointer after the first sweep already
  materialised the global.
- `Lowering.cpp` field-read dispatch: for FieldAccess on an
  instance of a CST prelude class (`tf`, `ss`, `zpk`, `pid`,
  `frd`) where Sema couldn't infer a concrete type, route to
  `matlab_obj_get_mat`. The runtime entry auto-boxes f64-stored
  fields back to 1×1 matrices, so the dispatch is a strict
  superset for these classes. Other user classdefs (Vec2,
  BasicClass, Counter, …) keep the f64 default so their scalar
  arithmetic stays in scalar lanes.
- `tools/matlabc/main.cpp`: prelude inclusion is now conditional
  — `userMentionsCstClass` does a comment-stripped, whole-word,
  call-site-pattern scan over the user input and only prepends
  the prelude when one of `tf(`, `ss(`, etc. appears (or one of
  these names appears on the LHS of a single `=`). An unused
  classdef would otherwise compile down to a `func.func` body
  whose `none`-typed slots no downstream pass can resolve, leaving
  stale `matlab.call_builtin` ops that fail LLVM-IR translation.

Scalar-mixing compiler fixes (this slice):
- `Lowering.cpp` BinaryOp class-method dispatch wraps non-class
  operands via a 1-arg `Owner(value)` constructor call before
  emitting the matlab.call. MATLAB's convention: `G + 2` is
  `G + tf(2)`. Restricted to CST prelude classes (tf / ss / zpk /
  pid / frd) so other user classdefs (Vec2, BasicClass) keep their
  scalar-mixing-friendly behaviour where the user-written method
  body handles the scalar directly.
- `Lowering.cpp` `pinnedFromExpr` recurses through BinaryOp /
  UnaryOp / CallOrIndex so composite expressions like
  `s*s + 3*s + 5` propagate the class pin through nested
  operators. Without this, the outer `+` saw both LHS and RHS as
  unpinned (their NameExprs are buried under operator nodes) and
  fell back to the default arithmetic path, which segfaulted in
  `matlab_add_mm` on the class-instance pointers.
- `LowerUserCalls.cpp` matlab.call → func.call sibling-clone
  search: when the strict signature match against the original
  callee fails, scan the module for `<callee>__sN` clones produced
  by the monomorphizer and pick the first whose signature matches
  the actual operand types. Class-method bodies that read class
  fields end up with ptr operands while the original constructor
  was retyped to a different signature based on its own bucket
  (e.g. `tf__tf` settled on `(ptr, f64)` for the `tf([1 0], 1)`
  site, but the call from inside `tf__plus` is `(ptr, ptr)` after
  slot retyping — that bucket landed on `tf__tf__s1`). The bucket-
  setting monomorphizer skips operands typed `none` at its run
  point, so these calls weren't retargeted by the normal path.
- prelude `tf` constructor accepts a 1-arg form `tf(c)` →
  `(c / 1)` constant transfer function, which is what the scalar-
  boxing wrapper calls.

Operator-overload compiler fixes from the previous slice:
- `Resolver.cpp` extended-binary-operator pinning: the second param
  of `mtimes`/`mrdivide`/`mldivide`/`mpower` (and the elementwise
  variants) gets pinned to the class for CST prelude classes only
  (`tf`/`ss`/`zpk`/`pid`/`frd`). Other user classdefs (Vec2, etc.)
  keep the historical scalar-mixing-friendly behaviour where `b` is
  unpinned. Without this, `b.Numerator` inside `tf__mtimes` reads
  via `matlab_struct_get_f64` (boxing scalar 0 to a 1×1) and the
  resulting `conv(...)` returns the right shape with all-zero data.
- `Lowering.cpp` unary-op class-method dispatch: `-tf_obj`,
  `+tf_obj`, `~tf_obj` route to the corresponding classdef method
  (`uminus` / `uplus` / `not`) when the operand is class-pinned.
  Without this, unary ops on a class instance lower as
  `matlab_neg_m(obj_ptr)` etc., reading the obj as a raw matrix and
  segfaulting.
- `Resolver.cpp` `pinnedOfRhs` walker: also recurses into
  `UnaryOpExpr`. Lets `M = -G;` propagate the pin from `G` to `M`
  so subsequent `M.Numerator` reads route through the class path.
- `LowerUserCalls.cpp` matlab.call → func.call conversion: relax
  the strict operand-type match to also accept `none`-typed
  operands (placeholder for "any") when the function expects ptr.
  Insert an `unrealized_conversion_cast` at the boundary; later
  passes drop it. Class-method bodies that flow a value through a
  none-typed slot (e.g., `r = tf(new_num, new_den)` where
  `new_num = cst_polyadd(...)` and the slot for `new_num` couldn't
  be retyped because of mixed-typed stores) would otherwise leave
  a stranded matlab.call after the upstream helper converted to a
  func.call returning ptr.

**Still 🔵**: `tf('s')` char-literal sugar (the `'s'` literal
doesn't survive the AST→MLIR lowering through the constructor body —
needs a special-case rewrite that recognises `tf('s')` at the
CallOrIndex lowering and emits `tf([1 0], 1)` directly), `ss` /
`zpk` / `frd` / `pid` classdefs (mechanical replicas of `tf`),
model-object forms of the existing matrix-arg primitives
(`step(sys)`, `bode(sys)`, etc.), and `disp(tf)` formatted s-domain
rendering.

Heavy carve-outs (apps, Simulink, LPV/LTV, sparse-second-order,
`systune`, Robust/MPC/SysID toolbox bridges) keep this scoped to
**single-toolbox practical-numeric subset** — the same posture SPT
takes. Re-open carve-outs only on user demand.

The single most-impactful primitive shipped: **`expm`** (Tier 1.3),
the gating piece for everything continuous-time and model-reduction
related. The single most-impactful primitive still **open**: model
objects (Stage 3 above).
