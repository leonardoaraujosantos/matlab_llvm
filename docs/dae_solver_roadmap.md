# Verilog-A Analog Simulation Roadmap — DAE core + MNA + full analog language

Scoped plan for what `matlab_llvm` (Sema + MLIR + Runtime + REPL/Debug
+ JIT) needs to ship a **full Verilog-A analog circuit simulator**:

1. a mass-matrix / fully-implicit **DAE solver core** (`ode15s`/`ode15i`),
2. a **Modified Nodal Analysis (MNA)** assembler that turns conservative
   device networks into a DAE,
3. the **complete Verilog-A analog operator + event language**,
4. the full set of **analyses** (DC / transient / AC / noise / RF/periodic),
5. **multi-discipline** (electrothermal, mechanical, …) and **structural
   hierarchy** (device → circuit), and
6. **numerical robustness + scale** for real circuits.

This is the *tractable, no-Simulink, no-Simscape-DSL* path to acausal
physical simulation. **Scope = full Verilog-A (the analog language).**
Verilog-AMS *digital / discrete-event* modeling (`@(posedge)`, `wreal`,
event-driven logic) is the explicit boundary — see Carve-outs (§13).

A first-class design goal is **maximum reuse of the existing runtime** —
GPU GEMM, complex-matrix ops, sparse triplet/iterative solvers, complex
FFT, and the `tf2ss`/`lsim` state-space lane all map directly onto pieces
of an analog simulator (§3 is the reuse map).

Companion docs:
- [`ode.md`](ode.md) — existing `ode45`/`ode23`/`ode23s` + `pdepe`.
- [`mflow_link_roadmap.md`](mflow_link_roadmap.md) — causal signal-flow
  simulator. **Behavioral** VA (single-input/output) lowers to mflowLink;
  **conservative** VA networks lower to MNA + these solvers. Note: the
  mflowLink *simulator's own* stiff solver gains mass-matrix / index-1
  support via OpenSpec `mflow-variable-step-stiff-solvers` — a sibling of
  this runtime DAE core (the simulator integrates block continuous-state;
  this core integrates MATLAB-language `ode15s`/`ode15i` IVPs/DAEs).
- [`acceleration_roadmap.md`](acceleration_roadmap.md),
  [`gpu_coder_roadmap.md`](gpu_coder_roadmap.md) — the GPU/CUDA lane reused
  in Tiers 7/10.
- [`control_toolbox_roadmap.md`](control_toolbox_roadmap.md) — `tf2ss`/
  `ss2tf`/`lsim` reused for the Laplace operators (Tier-4).
- [`pde_toolbox_roadmap.md`](pde_toolbox_roadmap.md) — `pdepe` already
  drives `ode23s_v`; same reuse precedent.

---

## 0. Reading guide

- **Status**: ✅ shipped · 🟡 partial · 🔵 not started.
- **Effort** in the existing cadence (session ≈ half-day; "week" ≈ 5).
- **Tiers 1–3 are the numerical core**; 4–5 the analog language; 6–7 the
  analyses; 8–9 multi-physics + circuits; 10 robustness/scale. Order is
  priority + dependency, not strict.
- Every tier must clear the **cross-mode execution contract** (§11) —
  AOT compile/run, debug/REPL, **and** JIT interpreted mode at parity.
  Hard gate, specified once in §11.
- §3 (**runtime reuse map**) is load-bearing: it's how "full Verilog-A"
  stays affordable.

---

## 1. Background — the numerical core, and what's out of scope

A **DAE** couples differential equations with algebraic constraints.
Two forms:
- **Mass-matrix / semi-explicit**: `M(t,y)·y' = f(t,y)`, `M` possibly
  **singular** (singular rows = pure constraints). → **`ode15s`**.
- **Fully implicit**: `f(t, y, y') = 0` (the charge/flux formulation of a
  circuit). → **`ode15i`**.

The **differential index** = how many constraint differentiations to
recover an ODE. BDF integrators (`ode15s`/`ode15i`) solve **index ≤ 1**
directly. Higher index needs **index reduction** (Pantelides + dummy
derivatives) — in MATLAB that's the **Symbolic Math Toolbox**
(`reduceDAEIndex`, `isLowIndexDAE`, `daeFunction`), *not* the numeric
core. Industrial Pantelides+tearing+BLT lives inside Simscape, unexposed.

> **MNA of realistic device networks is index ≤ 1.** A mass-matrix BDF
> integrator solves those directly. Index > 1 (ideal-source loops,
> inductor cutsets, ideal opamps) is handled in **Tier-10** by structural
> detection + a bounded Pantelides pass; until then, detect-and-error.

### MATLAB compatibility targets (for reference)

| MATLAB API | Product | Our tier |
|---|---|---|
| `ode15s`, `ode23t`, `ode23tb` (mass matrix) | base | T1 |
| `ode15i`, `decic` | base | T2 |
| `odeset('Mass','Jacobian','MassSingular',…)` | base | T1/T2 |
| `reduceDAEIndex`, `isLowIndexDAE`, `daeFunction` | Symbolic | T10 (bounded) |

---

## 2. What we already have — DAE solver reuse surface (`ode23s_v`)

The shipped stiff vector solver `rosen_solve_23s_v`
(`runtime/matlab_runtime.cpp:19977–20227`) already contains the hard
numerics. We extend a working stiff solver, not start from `ode45`.

| Block | Location | Reused by |
|---|---|---|
| `lu_factor_pp` (LU + partial pivot) | `:19988` | dense linear solves, all tiers |
| `lu_solve` | `:20018` | all tiers, verbatim |
| Central-FD Jacobian, column-by-column | `:20118–20132` | `ode15s` (∂f/∂y); `ode15i` (×2); MNA |
| Vector RHS ABI `ode_rhs_v_t` + `ode_v_call` | `:20036` | `ode15s` directly |
| Output `ode_v_push` / `ode_v_hermite`, grid/Refine | `:20175–20200` | all tiers, verbatim |
| Adaptive accept/reject + `MaxStep` | `:20173–20216` | all tiers |
| Stats + `ode_opts_resolve` | `:19000`, `:20219` | all tiers |
| Public-entry fan-out + multi-return split | `:19705–19821` | copy per solver |

The Rosenbrock W-method already factors `I − h·d·J` once per step with a
numerically-built Jacobian — ~70–80% of an implicit DAE integrator.

---

## 3. Runtime reuse map — how "full Verilog-A" stays affordable

The defining strategy: **most of an analog simulator is linear algebra,
spectral transforms, and state-space realization we already ship.** Each
simulator piece below names the existing runtime symbol it builds on.

| Simulator need | Reused runtime capability | Symbols | Tier |
|---|---|---|---|
| MNA matrix **assembly** | sparse triplet builder | `matlab_sparse_from_triplets`, `_diag`, `_eye` | T3 |
| MNA **solve at scale** (unsymmetric circuit matrices) | sparse iterative + ILU | `matlab_sparse_gmres_ilu`, `_matvec`, `_nnz` | T3/T10 |
| MNA **solve, small/dense** | dense LU | `lu_factor_pp` / `lu_solve` | T3 |
| `laplace_nd/zd/np/zp`, `zi_*` operators | transfer-fn → state-space realization | `tf2ss`, `ss2tf`, `matlab_lsim`, `matlab_tf` | T4 |
| `ddt` / `idt` / `idtmod` | the DAE `y'` itself + added integrator state | (DAE core) | T4 |
| **AC small-signal** (complex MNA over freq) | complex matrix ops; complex solve via real 2N block | `matlab_complex_mm`, `_ms`, `_sm` | T6 |
| **Noise / transient noise** | RNG + AWGN + spectral | `awgn`, `randn`, `matlab_fft_c` | T6 |
| **Harmonic balance / PSS / spectra** | complex FFT (time↔freq), dense block Jacobian | `matlab_fft_c`, `matlab_ifft_c`, `matlab_fftshift_c` | T7 |
| **HB dense Jacobian / Monte-Carlo / corner sweeps** | **GPU GEMM** + batch dispatch | `matlab_gpu_cuda_gemm_double`, `matlab_gpu_gemm`, `gpuArray`, `matlab_gpu_cuda_dispatch` | T7/T10 |
| Eigen-based **pole-zero / AC modal** | existing `eig`/`svd`/`qr` | (linalg lane) | T6 |
| Behavioral (causal) VA blocks | signal-flow simulator | mflowLink runtime | T4 |
| Math intrinsics (`exp/tanh/sqrt/log/pow/sin…`) | native runtime elementwise | (runtime) | T4 |

**Where GPU actually helps (honest):** the sparse transient/DC core is
latency-bound, so GPU GEMM does *not* speed it up. GPU pays off on the
**dense, batched** workloads: **harmonic-balance** block Jacobians,
**Monte-Carlo / corner / parameter sweeps** (embarrassingly-parallel
independent solves), and **behavioral-heavy dense MNA**. Tiers 7/10 route
those through `matlab_gpu_cuda_gemm_double` with graceful CPU fallback
when no device is present.

---

## 4. Tiers

### Tier-1 — `ode15s` (mass-matrix, index-1) 🔵
`[t,y] = ode15s(@f, tspan, y0, odeset('Mass',M))` solving `M·y'=f`, `M`
constant and possibly singular.
- **New numerics:** variable-order (1–5) **NDF/BDF** multistep predictor/
  corrector + step-history, replacing the single-step Rosenbrock stage
  block (`:20143–20162`). Newton corrector reusing `lu_factor_pp`/
  `lu_solve`/FD-Jacobian. Stamp `M` instead of `I`; **singular `M` → DAE
  for free**.
- **New options:** `Mass`, `MassSingular`, `Jacobian` in `ode_opts_resolve`.
- **Effort ~1.5 wk. Reuse ≈ 75%.** Acceptance: Robertson index-1 DAE vs MATLAB.

### Tier-2 — `ode15i` (fully implicit) + `decic` 🔵
`f(t,y,y')=0` + consistent ICs.
- **New:** residual ABI `ode_rhs_i_t (t,y,yp)` (mirrors `pdepe_rhs`); **two**
  FD Jacobians (∂/∂y, ∂/∂y'), iteration matrix `∂f/∂y + (β₀/h)·∂f/∂yp`;
  Newton + shared BDF controller (factor it out in T1). `decic` = one
  Newton solve at `t0`.
- **Effort ~1.5 wk. Reuse ≈ 65%.**

### Tier-3 — MNA assembler: transient + DC-op + sparse + limiting 🔵
The acausal core. **Not a new integrator** — an assembler emitting the
DAE residual + Jacobian for T1/T2.
- **Node map**: an unknown per `electrical` node + per current-branch
  (V-sources, inductors); ground = 0.
- **Stamper** → sparse triplets (`matlab_sparse_from_triplets`): linear
  R/C/L/sources (constant stamps) + nonlinear (diode `exp`, evaluated at
  the Newton iterate). **DC operating point** falls out (Newton at t=0).
- **Solve**: dense `lu_*` for small nets; `matlab_sparse_gmres_ilu` for
  scale.
- **Junction limiting / damped Newton** (`pnjlim`): mandatory — `diode.va`
  `exp()` overflows without it.
- **Index guard**: detect index > 1, error pointing at T10.
- **Effort ~2.5 wk. Integrator reuse 100%; assembly reuses sparse lane.**
  Acceptance: `rc_lowpass.va` (RC step vs analytic), `diode.va` rectifier
  (nonlinear + limiting), `resonant_bpf.va` (RLC via `ddt`).

### Tier-4 — Verilog-A analog operators 🔵
The continuous-time operator set. Each maps to reuse:
- `ddt(x)` → derivative — *native* (it's the DAE `y'`).
- `idt(x[,ic])`, `idtmod(x,off,mod)` → add an integrator **state** to the
  DAE (+ modulo wrap for `idtmod`, e.g. `vco.va` phase).
- `laplace_nd/zd/np/zp(...)`, `zi_nd/zp(...)` → **rational transfer fn →
  state-space via `tf2ss`**, companion states integrated inside the DAE.
  Reuses `tf2ss`/`ss2tf`/`lsim`. Covers `amplifier.va`, `biquad_butter.va`,
  `rf_rational.va` (11 `laplace_nd` uses in-tree).
- `transition(x,td,tr,tf)`, `slew(x,rate)` → filtered piecewise
  transitions (rate-limited state).
- Math/analog builtins: `exp/log/sqrt/pow/tanh/sin/cos/abs/min/max`,
  `$vt`, `$temperature`, `$abstime` — native runtime + simulator globals.
- `analog function`, `analog initial`, parameters + ranges +
  `$param_given`, `aliasparam`, genvar.
- **Effort ~2.5 wk.** Acceptance: `amplifier.va` (laplace), `vco.va`
  (idtmod), `biquad_butter.va`.

### Tier-5 — analog event engine 🔵
Discontinuity + event handling — required for switching/hysteretic devices.
- `cross(expr,dir[,tol])`, `above(expr)`, `timer(t,period)`,
  `last_crossing()` → **zero-crossing detection + step control to land on
  the event** + state reset / re-init. Builds on the existing adaptive
  stepper's event hook (`matlab_ode_events_*` precedent).
- `@(cross(...))`, `@(above(...))`, `@(timer(...))`, `@(initial_step)`,
  `@(final_step)` event blocks.
- `$bound_step`, `$discontinuity`, `$limit` solver hints.
- **Effort ~2.5 wk.** Acceptance: `comparator.va`, `schmitt.va`
  (7 `cross` + 6 `@(...)` uses in-tree).

### Tier-6 — analyses: DC sweep · AC small-signal · noise 🔵
- **DC sweep** — loop the DC-op Newton over a swept source/parameter.
- **AC small-signal** — linearize at the OP (reuse the assembled
  Jacobian), build the **complex** MNA `(G + jωC)·x = b`, solve per
  frequency. Reuse `matlab_complex_mm`/`_ms`/`_sm`; solve via the real 2N
  block `[G −ωC; ωC G]` so the existing real sparse/dense solver applies.
- **Noise analysis** — AC + device-noise PSD integration; the **only**
  place `white_noise()` / `flicker_noise()` / `noise_table()` produce
  output (covers `noise_thermal.va`, `noise_flicker.va`). **Transient
  noise** via `awgn`/`randn` stamped per step.
- `analysis("ac"|"dc"|"tran"|"noise")` VA function so models branch.
- **Effort ~3 wk.** Reuses complex + RNG/FFT lanes heavily.

### Tier-7 — RF / periodic-steady-state 🔵
The RF analyses behind the `.s1p` files (`dipole_1ghz.s1p`, `rf_rational.va`).
- **Harmonic balance (HB)** — frequency-domain Newton; **dense
  block-structured Jacobian** over harmonics × nodes. Time↔freq via
  `matlab_fft_c`/`ifft_c`; **the dense block GEMM goes to GPU**
  (`matlab_gpu_cuda_gemm_double`, CPU fallback).
- **PSS / PAC / PNOISE / PXF**, **S-parameter extraction** (multiport AC →
  Touchstone `.s1p/.s2p` write — reuses the existing RF Touchstone I/O).
- **Effort ~4 wk.** Headlines the GPU reuse.

### Tier-8 — multi-discipline / multi-physics 🔵
Generalize MNA from `electrical` to any conservative discipline over a
(potential, flow) nature pair.
- Discipline/nature definitions (`disciplines.vams`), `thermal`,
  `mechanical`, `rotational`, `magnetic`, `kinematic`.
- **Electrothermal coupling** (self-heating) — electrical + thermal nodes
  in one MNA system; `$temperature` feeds device models. (`rtd_pt100.va`,
  `thermistor_ntc.va`, the 2 `thermal` nodes in-tree.)
- **Effort ~2 wk** (the stamper is discipline-agnostic once natures exist).

### Tier-9 — structural netlist & hierarchy 🔵
"Circuit" vs "device": connect instances.
- Module **instantiation**, hierarchical flattening, ports (`inout`,
  vector/bus ports), parameter passing, `ground`.
- A netlist front-end (structural Verilog and/or a SPICE-deck reader)
  building the node map across many instances.
- **Effort ~3 wk.** Turns single-device models into simulatable circuits.

### Tier-10 — robustness & scale 🔵
What "any real circuit" needs numerically.
- **Convergence aids**: gmin stepping, source stepping, pseudo-transient,
  homotopy/continuation.
- **Sparse at scale**: full reliance on `matlab_sparse_gmres_ilu` +
  reordering; dense LU only below a node threshold.
- **GPU batch**: Monte-Carlo / corner / parameter sweeps as
  embarrassingly-parallel batched solves (`matlab_gpu_cuda_dispatch`).
- **Index-2 handling**: structural detection + a *bounded* Pantelides
  pass (the deferred reduction from §1), or reformulation hints.
- `.ic` / `uic` / nodeset, bypass, multirate/latency.
- **Effort ~4 wk** (incremental; ship as needed per circuit class).

---

## 11. Cross-mode execution contract (hard gate — every tier)

Identical behavior across AOT, REPL/Debug, and JIT — the same contract
every runtime builtin satisfies. Specifics for this work:

### 11.1 AOT compile/run (LLVM + C/C++/Python/TS)
- New entries in `runtime/matlab_runtime.cpp` (or a new
  `runtime/toolbox/circuit/runtime_circuit.cpp`) beside the `matlab_ode*`
  family. Sema **Resolver** registers `ode15s`/`ode15i`/`decic` + the
  analysis/`analog`/MNA entry points; **Lowering** rewrites
  `matlab.call_builtin` → C-ABI symbols. Strict-cast lane
  (`-Werror=old-style-cast` → `static_cast`).
- Multi-return follows the `matlab_ode45_t/_y/_stats` split-site pattern.
  The 3-arg `ode15i` residual handle is the only novel lowering item.
- Transpiled shims (`runtime/shim/*`) get the new signatures for parity.

### 11.2 JIT interpreted mode
- Runtime symbols resolve via the shared `DynamicLibrarySearchGenerator`
  against the runtime shared library — exactly as `runtime_signal` /
  `ode23s` / the **sparse** and **GPU** symbols already do. The solver,
  MNA, complex, sparse, FFT and GPU paths run bit-identically AOT vs JIT.
- **Function-handle ABI is the watch item**: `@f`/`@res` and VA `analog`
  callbacks must lower to JIT-invokable pointers (the existing
  `ode_rhs_v_t`/`pdepe_rhs` trampoline path). Add a JIT-mode test calling
  `ode15i` and an MNA solve with an anon residual.
- **GPU under JIT**: `matlab_gpu_cuda_*` are already resolved through the
  same generator; the JIT path must degrade to CPU GEMM when no device,
  and the GPU↔CPU result must match within tolerance under both modes.
- **REPL parity trap** (`jit_pipeline_divergence`): ReplMode `ws_get_mat`
  reads can defeat AOT-shaped detectors. Any result round-tripping a
  struct (solver `stats`, `odeset`, analysis descriptors) must be
  re-checked under `-repl`/`-dap` JIT, gated by the JIT-vs-AOT harness.

### 11.3 Debug / REPL display
- New `odeset` fields (`Mass`/`Jacobian`/`MassSingular`) and analysis
  descriptors render in the DAP variable inspector — extend the struct
  child-walker in `runtime_debug.cpp` (RF `RFCktAmplifier` / PDE
  `femodel` "typed bag of fields" precedent).
- Circuit/MNA descriptor (node map + stamps + last solution) gets a
  one-line REPL summary + DAP child-walker.

### 11.4 Determinism (cross-platform gate)
- FD-Jacobian step `√eps·max(|y|,1)`, BDF coeffs, sparse ILU drop
  tolerances are deterministic. **Transient-noise / Monte-Carlo use
  seeded RNG**; pin seeds so the full ctest gate (macOS + Linux) matches.
- Nonlinear (`diode`, HB) cases are libm-divergence points → acceptance
  tolerances, not bit-exact. GPU vs CPU GEMM → tolerance compare.
- Honor `ci_linux_gate`: c++20, no unordered-container iteration in
  golden output, numpy present.

---

## 12. Headline demos (gating examples)

- `examples/ode/dae_robertson.m` — index-1 stiff DAE via `ode15s` + `ode15i` (T1/T2).
- `examples/circuits/rc_step_response.m` — `rc_lowpass.va`, MNA → `ode15s` (T3).
- `examples/circuits/diode_rectifier.m` — `diode.va`, nonlinear MNA + limiting (T3).
- `examples/circuits/vco_pll.m` — `vco.va` `idtmod` + `laplace_nd` filter (T4).
- `examples/circuits/schmitt_trigger.m` — `schmitt.va` `cross`/event engine (T5).
- `examples/circuits/amp_ac_noise.m` — `amplifier.va` AC + `noise_*.va` noise analysis (T6).
- `examples/rf/mixer_hb_sparam.m` — harmonic balance + S-param, GPU-accelerated (T7).
- `examples/circuits/electrothermal_rtd.m` — `rtd_pt100.va` self-heating (T8).

## 13. Carve-outs

- **Verilog-AMS digital / mixed-signal**: `@(posedge)`, `wreal`,
  real-number modeling, event-driven logic, connect modules — a
  discrete-event simulator, a separate product. (This is the boundary of
  "full Verilog-A": the analog language is in scope; AMS-digital is not.)
- **Foundry compact-model libraries** (BSIM4/BSIM-CMG/HICUM/PSP/EKV as
  *bundled* models) — only what is hand-written or VA-imported; we are not
  shipping a PDK.
- **Layout / parasitic extraction, EM field solvers** (the `.s1p` is
  *consumed/produced*, not extracted from geometry — that's the Antenna/
  Propagation toolboxes).
- **Full Pantelides/tearing/BLT** — T10 ships a *bounded* index-2 pass
  only; arbitrary high-index reduction stays deferred.
- **Variable / state-dependent mass matrix** `M(t,y)` — constant `M` first.
- **GUI schematic capture / waveform viewer** — REPL + plot lane only.

## 14. Effort summary & build order

| Tier | Item | Effort | Key reuse |
|---|---|---|---|
| 1 | `ode15s` mass-matrix BDF | ~1.5 wk | `ode23s_v` LU/FD/output |
| 2 | `ode15i` + `decic` | ~1.5 wk | T1 controller |
| 3 | MNA transient + DC-op + sparse + limiting | ~2.5 wk | sparse triplets/GMRES + DAE core |
| 4 | VA analog operators | ~2.5 wk | `tf2ss`/`lsim`, native math, mflowLink |
| 5 | analog event engine | ~2.5 wk | `matlab_ode_events_*` stepper hook |
| 6 | DC sweep · AC · noise | ~3 wk | complex ops, FFT, AWGN/RNG |
| 7 | RF / HB / PSS / S-param | ~4 wk | **GPU GEMM** + complex FFT + Touchstone I/O |
| 8 | multi-discipline / electrothermal | ~2 wk | discipline-agnostic stamper |
| 9 | structural netlist / hierarchy | ~3 wk | (new front-end) |
| 10 | robustness & scale (convergence, sparse, GPU batch, index-2) | ~4 wk | sparse GMRES, GPU dispatch, bounded Pantelides |

**Order:** 1 → 2 → 3 (core + first real circuits) → 4 → 5 (full single-
device analog language) → 6 (the everyday analyses) → 8 (electrothermal,
cheap) → 9 (circuits) → 7 (RF) → 10 (scale, incremental).

**Minimal "useful analog simulator" cut:** T1+T2+T3+T4+T5+T6 (~14 wk) —
transient + DC + AC + noise of full single-device analog VA, the 90%.
T7–T10 extend to RF, multi-physics, multi-device circuits, and scale.
