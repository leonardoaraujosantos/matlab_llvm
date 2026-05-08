# Roadmap

Forward-looking work tracker for `matlab_llvm`. Organized by effort
horizon and dependency chain, not strict priority — what gets done next
depends on which items unblock real users.

For shipped work, see [`feature_status.md`](feature_status.md). For
detailed history of each backend, see the per-backend `emit_*.md`
docs.

---

## Conventions

- **Effort** is calendar time at one focused implementation session
  per stage (the existing Phase 5.6.x cadence). A "week" means
  ~5 sessions, not 40 hours.
- **Status** legend:
  - 🔵 not started
  - 🟡 in progress / partial
  - 🟢 done (kept here for context until rolled into `feature_status`)
- **Scope dependency** notes flag items that must land first.

---

## Recently shipped (data-container + multi-return arc)

A focused arc of nine phases closed since the last roadmap refresh.
Authoritative status is in [`feature_status.md`](feature_status.md);
roadmap-side summary:

| Phase | What | Gating test |
|---|---|---|
| 1.1 | Typed `int32` / `uint8` matrix runtime — saturating arith, comparisons, casts, REPL+DAP display, Python+TS parity | `int_matrix_binops.m`, `int_image_filter.m`, `int_pixel_math.m` |
| 1.2 | `varargout` (pure + mixed forms) and plain user-fn multi-return (was broken — both LHS got the same value) | `varargout_basic.m` |
| 1.3 | 2-D cell literals + `C{r,k}` indexing + `[A,B]` / `[A;B]` cell concat | `cell_2d.m` |
| 2 | Struct arrays (`s(i).x`) with auto-grow, `length`/`numel`/`size` | `struct_arr_basic.m` |
| 3 | OOP value-class copy-on-assign for non-`< handle` classes | `value_class_copy.m` |
| 4 | `containers.Map` / `dictionary` — string + numeric keys, f64 + matrix values | `dict_basic.m` |
| 5.1 | scalar `datetime` / `duration` with constructors, display, arithmetic | `datetime_basic.m` |
| 5.2 | 1-D `categorical` from string array — disp / length / iscategory / categories | `categorical_basic.m` |
| 5.3 | `table` — auto-named + `'VariableNames'`, dot column access, dynamic add, `height`/`width`/`disp` | `table_basic.m` |
| 6   | **Symbolic Math Toolbox** via [SymPP](https://github.com/leonardoaraujosantos/SymPP) — `syms`, `diff`, `int`, `simplify`, `expand`, `factor`, `subs`, `solve`, `vpa`, `taylor`, `limit`, `dsolve`, `pdsolve`, `laplace`/`fourier`/`ztrans` (+ inverses), `assume`, sym arithmetic dispatch on `+ - * / ^ ==`, sym-typed elementary functions, REPL JIT + DAP variable inspector, opt-in via `-DMATLAB_LLVM_WITH_SYM=ON` | `test/RunSym/sym_phase_a.m`, `sym_phase_b.m` |
| 6.1 | **Symbolic matrices + multi-eq + IVP + numeric solve** — new `matlab_symmat` opaque type (kind=8) with cross-TU REPL persistence + DAP rendering; `sym_matrix`, `sym_eye`, `sym_zeros`, `sym_det`, `sym_inv`, `sym_transpose`, `sym_trace`, `sym_rank`, `sym_linsolve`, `sym_dsolve_system`; fixed-arity multi-eq solvers `sym_solve_2x2` / `sym_solve_3x3` returning a symmat (one row per joint solution); `nsolve`, `vpasolve`, `dsolve_ivp` (1-cond), `apply_ivp` (1-cond), `checkodesol` | `test/RunSym/sym_phase_b1.m` |
| 6.2 | **Phase 6.2 ergonomics** — standard `[a 1; 2 b]` matrix literal syntax detects sym entries (no more explicit `sym_matrix(...)` constructor); variadic `sym_solve_sys([eq...], [var...])` for systems of any size via LLVM stack-array lowering; multi-condition `dsolve_ivp` / `apply_ivp` taking parallel sym vectors; `simplify` auto-chains `refine()` so assumptions propagate (`simplify(sqrt(y*y))` → `y` after `assume(y,'positive')`); `sym('pi')` / `sym('exp1')` resolve to SymPP singletons; LLVM ptr added to `RefineSlotTypes::isScalarPrim` so sym slots get type-promoted and Mem2Reg'd | `test/RunSym/sym_phase_b2.m` |
| 7   | **Initial-value ODE solvers** — `ode45` (Dormand–Prince 5(4)) and `ode23` (Bogacki–Shampine 3(2)) with adaptive FSAL step + cubic-Hermite dense output; scalar and **vector `y`** (system of ODEs via anon-handle retyping pre-pass in `LowerAnonCalls`); forward / backward / user-time-grid `tspan`; full odeset surface — `RelTol`, `AbsTol`, `MaxStep`, `InitialStep`, `Refine`, `Stats`; 2-return `[t, y]`, 3-return `[t, y, stats]`. C++ / Python / TypeScript runtimes bit-identical. See [`docs/ode.md`](ode.md). | `test/Run/math_ode45_*.m`, `test/Runtime/test_ode.c` |
| 7.1 | **Stiff solver — `ode23s`** (Rosenbrock 2(3), Shampine). One numerical-FD Jacobian per accepted step + LU-factored `(I − h·d·J)` shared across three linear back-solves; scalar y → division, vector y → in-place LU with partial pivoting in the runtime. Robertson kinetics solves in ~9 steps where `ode45` diverges. Mirrored in Python (`numpy.linalg.solve`) and TypeScript (custom LU). | `test/Run/math_ode23s_basic.m`, `math_ode23s_robertson.m` |
| 7.2 | **Numerical PDE — `pdepe`** (1-D parabolic-elliptic, method-of-lines). MATLAB-compatible call shape `pdepe(m, @pdefun, @icfun, @bcfun, xmesh, tspan)`. Coverage: Cartesian / cylindrical / spherical (`m = 0, 1, 2`); Dirichlet, Neumann, Robin BCs; non-uniform mesh; scalar PDE. Spatial finite-difference discretisation; resulting full-state ODE system handed to `ode23s_v` for stiff time integration. Heat-equation gating `u_t = u_xx` on a 21-point mesh recovers the analytic `exp(-π²t)·sin(πx)` to ~1e-3. Cylindrical Laplacian on annulus recovers the log-profile steady state to ~2e-5. Bit-identical across C++ / Python / TS. | `test/Run/math_pdepe_heat.m`, `math_pdepe_neumann.m`, `math_pdepe_radial.m` |
| 7.3 | **Event detection — `ode_events`.** 5-return builtin `[t, y, te, ye, ie] = ode_events(@f, tspan, y0, @evt)`. The event function returns a 3×1 column `[value; isterminal; direction]`; the integrator brackets each accepted DP45 step on `value`, then bisects within the bracket to localize the crossing. `direction` filters rising / falling / either; `isterminal = 1` halts integration at the event. Wired through Resolver + LowerTensorOps as a 5-result builtin (4 operands: rhs handle, tspan, y0, evt handle). Mirrored across C++ / Python / TS — ball-drop event reproduces to ~2e-13 across all three. Non-MATLAB call shape: routing through `opts.Events` is gated on the function-handle-in-struct ABI (same blocker as `OutputFcn` / `Mass`). | `test/Run/math_ode_events_ball.m` |

---

## Recently shipped (Signal Processing Toolbox arc, 2026-05-06 → 2026-05-07)

A second focused arc closing the practical SPT user surface across
Tier-1 (design / apply / inspect), Tier-2 (spectral / parametric /
time-frequency), and Tier-3 (multirate / measurements / alignment).
Drafted from the R2026a Signal Processing Toolbox User's Guide
(2048 pages, 27 chapters). Per-toolbox plan in
[`signal_toolbox_roadmap.md`](signal_toolbox_roadmap.md);
authoritative status in [`feature_status.md`](feature_status.md).

| Phase | What | Gating test |
|---|---|---|
| SPT 1 (§2.3) | **Windows tail** — 14 new windows (`rectwin`, `triang`, `bartlett`, `barthannwin`, `bohmanwin`, `parzenwin`, `nuttallwin`, `blackmanharris`, `flattopwin`, `kaiser`, `tukeywin`, `gausswin`, `chebwin`, `taylorwin`) plus retrofit of pre-existing `hamming`/`hann`/`blackman` into Python / TS for parity. Symmetric (non-periodic) form throughout. | `test/Run/sig_windows.m`, `test/Runtime/test_signal.c` |
| SPT 2 (§2.4) | **Polynomial helpers** — `roots` (Durand-Kerner / Weierstrass simultaneous iteration; the long-standing eig dependency the roadmap had flagged turned out to be moot — DK was already shipped), `poly` (repeated convolution by `[1, −r_i]`), `polyder`, `polyint`, `polyint(p, k)`, plus `[r, p, k] = residue(b, a)` distinct-pole expansion via cover-up rule (multi-return shape mirrors `[V, D] = eig`). | `test/Run/sig_poly.m`, `sig_residue.m` |
| SPT 3 (§2.1) | **IIR lowpass design** — `[b, a] = butter(n, Wn)`, `[b, a] = cheby1(n, Rp, Wn)`, `[b, a] = cheby2(n, Rs, Wn)` via bilinear-transform from analog prototypes (cheby2 needs the generalised `lowpass_from_analog_pz_` helper for j-axis zeros). `freqz(b, a, N)` + `[H, w] = freqz(...)`. Order helpers `[n, Wn] = buttord(...)` / `cheb1ord(...)`. Lowpass scope only — band variants attempted as a separate slice but discarded (peak normalisation off, deferred). | `test/Run/sig_iir.m`, `sig_iir_more.m` |
| SPT 4 (§2.2) | **FIR design** — `fir1(n, Wn)` windowed-sinc with default Hamming, `sgolay(k, f)` returning the projection matrix `B = V (V'V)⁻¹ V'`, `sgolayfilt(x, k, f)` applying it with proper boundary-row handling. | `test/Run/sig_fir.m` |
| SPT 5 (§2.5) | **Close-the-loop filter helpers** — `filtfilt(b, a, x)` forward-backward zero-phase (reflection padding + zero ICs; Gustafsson IC trick deferred), `sosfilt(sos, x)` cascade biquad, `impz`/`stepz`/`grpdelay` response inspection. All five share the internal `filter_flat_` direct-form-II-transposed helper. | `test/Run/sig_filt.m` |
| SPT 6 (§3.4) | **Transforms tail** — `dct`/`idct` (orthonormal, direct O(N²)), `fwht` (in-place butterfly, Hadamard ordering, divided by N), `hilbert` (FFT zero-negative-half + IFFT, returns `matlab_mat_c`), `goertzel` (single-bin DFT, 1×1 complex). | `test/Run/sig_xform.m` |
| SPT 7 (§3.1) | **Nonparametric spectral** — `periodogram(x)`, `pwelch(x, win, noverlap)`, `cpsd(x, y, win, noverlap)`, `mscohere(x, y, win, noverlap)`, `tfestimate(x, y, win, noverlap)`. Single-output, default `fs = 1`. cpsd / tfestimate return `matlab_mat_c`; the TS lane returns magnitude only (no native complex shape — same precedent as `roots` / `fft_c`). | `test/Run/sig_psd.m`, `sig_xspec.m` |
| SPT 8 (§3.2) | **Linear prediction + parametric PSD** — `levinson(r, p)` Levinson-Durbin recursion, `lpc(x, p)` (biased-autocorr + Levinson), `aryule(x, p)`, `arburg(x, p)` Burg forward+backward minimization, plus AR-based PSD `pyulear(x, p, N)` and `pburg(x, p, N)`. | `test/Run/sig_lp.m`, `sig_xspec.m` |
| SPT 9 (§3.3) | **Time-frequency** — `S = spectrogram(x, win, noverlap)` single-output `\|STFT\|²` per (freq, frame). 2-/3-return forms and `stft`/`istft` deferred. | `test/Run/sig_spec.m` |
| SPT 10 (§4.3) | **Pulse measurements core** — `findpeaks` (1-return + multi-return `[pks, locs]`, strict-monotonic), scalar reductions `rms`/`peak2peak`/`peak2rms`/`rssq`, signal cleanup `medfilt1`/`hampel`/`envelope`, pulse statistics `midcross`/`risetime`/`falltime`/`dutycycle` with auto-detected min/max state levels and 10/50/90% reference percentages. MinPeak* options + slewrate/pulseperiod/pulsewidth/overshoot/undershoot/settlingtime deferred. | `test/Run/sig_peaks.m`, `sig_pulse.m`, `sig_stat.m` |
| SPT 11 (§4.1) | **Real multirate** — `upfirdn(x, h, p, q)` (direct algorithm, no zero-stuffed buffer), `decimate(x, r)`, `interp(x, r)`, `resample(x, p, q)`. Replaces the toy `upsample`/`downsample` stubs. Default lowpass via `fir1`. Output lengths match MATLAB: `decimate` ⌈N/r⌉, `interp` N·r, `resample` ⌈N·p/q⌉. | `test/Run/sig_resample.m` |
| SPT 12 (§4.2) | **Waveform generators** — `chirp(t, f0, t1, f1)` linear method, `sawtooth(t, w)`, `square(t, duty)`, `gauspuls(t, fc, bw)`, `rectpuls(t, w)`, `tripuls(t, w)`, `sinc(x)`. | `test/Run/sig_wfmalign.m` |
| SPT 13 (§4.4) | **Alignment helpers** — `xcov(x, y)` mean-removed cross-correlation, `finddelay(x, y)` argmax of `\|xcorr\|`, `dtw(x, y)` dynamic time warping (scalar distance). `alignsignals` multi-return + `gccphat` deferred. | `test/Run/sig_wfmalign.m` |

Cumulative test deltas vs. `e812c3f` (pre-SPT baseline): **+17 run-tests** on the LLVM/C/C++/-strict lanes (172 → 189), **+17** on emit-python (161 → 178 + 11 skip), **+14** on emit-typescript (152 → 166, with 3 new skips for complex-valued tests where TS NDArray drops the imaginary part). All 42 ctest lanes regression-clean. ~80 new SPT runtime entries wired end-to-end.

---

Open follow-ups carried forward (still on the roadmap):

- **Phase 5.4 — `timetable`.** Builds on `table` + `datetime` row index.
- **Narrower / wider int lanes** — i8 / i16 / i64 / u16 / u32 / u64 matrix descriptors against the same template as Phase 1.1.
- **Full method-dispatch value semantics** for OOP — needs test-corpus migration to either rebind or `< handle`-annotate the existing class fixtures.
- **Heterogeneous table columns** — string / categorical / datetime columns alongside numeric.
- **Phase 6.3** — `matlabFunction(f, vars)` wrapping the SymPP-emitted Octave source into a callable function handle; assumption properties beyond SymPP's mask (`even`, `odd`, `prime`, `algebraic`, `complex` — needs SymPP-side phase); array-arg builtins `rsolve` / `groebner` / `pythagorean_triples` / `linear_diophantine` language-level wiring.
- **Phase 7.4 — Higher-order stiff solver.** `ode15s` (variable-order BDF + Newton iteration on top of the shipped FD-Jacobian / LU infrastructure from 7.1). Other stiff variants (`ode23t`, `ode23tb`, `ode15i` for DAEs) drop in cheaply once BDF lands. Will speed up `pdepe` on tight tolerances and very-stiff parabolic problems.
- **Phase 7.5 — `pdepe` extensions.** Multi-component systems (`npde > 1`); axis-of-symmetry handling for `xmesh(1) = 0` with `m > 0`; plumbing `odeset` (including `Events`) through to the time integrator. Then 2-D parabolic via `pdepe`-style on a tensor-product mesh.
- **Phase 7.6 — Events through `odeset`.** Bracket+bisect event detection ships today as the dedicated `ode_events` builtin (Phase 7.3); promoting it onto `opts.Events = @evt` for `ode45` / `ode23` / `ode23s` is gated on the function-handle-in-struct ABI work that also unblocks `OutputFcn` (live progress callback) and `Mass` (mass-matrix DAEs).
- **Vector `y` via *named* user functions** — currently anon-only; the LowerUserCalls signature-refinement gate rejects `tensor<Nxf64>` ↔ `tensor<Nx1xf64>` shape mismatches and needs widening.
- **Phase 8 — SV `state_display` regression** (commit `3622f10`): the const-fold pass eliminates assignments to output-port slots when the value comes from a persistent register read. A Phase-6.2 attempt to fix it via `LowerScalarSlots` cast insertion + EmitSV `arith.fptosi` rendering surfaced a cocotb timing divergence between the Python and SV references for `fir_asic_pipelined`; needs lockstep pipeline-equivalence work.
- **SPT §2.1 follow-on — IIR family completion (tail).** The big slice
  shipped: band variants (HP/BP/BS) of `butter`/`cheby1`/`cheby2` (the
  bandpass peak-normalisation bug from the previous attempt was the
  prewarp-vs-bilinear T-convention mismatch — `bilinear_pole_` now
  uses `(2+s)/(2-s)` so all four filter types reproduce scipy / MATLAB
  exactly); `besself` (analog Bessel-Thomson, MATLAB's norm='phase');
  standalone `bilinear` / `freqs`; `cheb2ord`; `tf2zp` / `zp2tf` /
  `tf2sos` / `sos2tf` form conversions. Still open: `ellip` +
  `ellipord` (Jacobi elliptic), the analog prototype builtins as
  standalone 3-return entries (`buttap` / `cheb1ap` / `cheb2ap` /
  `ellipap` / `besselap`), and the remaining state-space / zp-to-sos
  conversions (`tf2ss` / `ss2tf` / `zp2sos`).
- **SPT §2.2 follow-on — richer FIR design.** `fir2` (frequency sampling), `firls` (least-squares), `firpm` (Parks-McClellan / Remez exchange), `firrcos` (raised-cosine), `kaiserord`.
- **SPT §2.5 follow-on — strict Gustafsson `filtfilt`.** The
  lfilter_zi-based steady-state IC path that scipy uses by default
  (method='pad') shipped (constant signals now preserved exactly).
  The strict 1996 Gustafsson method (scipy's method='gust') uses an
  explicit edge-elimination linear system instead of padding; that's
  still open. Plus `phasez` / `zerophase` real-valued response helpers.
- **SPT Tier-2 follow-on — multitaper + STFT + subspace methods.** `dpss` (Slepian sequences via tridiagonal eig), `pmtm`, `stft`/`istft` (with COLA inversion for `istft`), `pspectrum`, `instfreq`, `instbw`, `czt` (chirp Z-transform via Bluestein), `cceps`/`rceps`/`icceps`, `pcov`/`pmcov`, `pmusic`/`peig`/`rootmusic`/`rooteig`, `prony`/`stmcb`.
- **SPT §4.1/§4.2/§4.4 follow-ons.** Polyphase decomposition (`polyphase`); chirp non-linear methods (quadratic / log / hyperbolic), `pulstran`, `diric`, `gmonopuls`, `vco`; `alignsignals` (multi-return), `gccphat`, `xcorr` scaling-option strings (`'biased'`/`'unbiased'`/`'normalized'`/`'coeff'`).
- **SPT §4.3 follow-on.** `findpeaks` name-value options (`MinPeakHeight`/`MinPeakDistance`/`MinPeakProminence`/`Threshold`/`SortStr`) — gated on Sema's name-value-arg parsing. The pulse-statistics tail (`statelevels`, `slewrate`, `pulseperiod`, `pulsewidth`, `overshoot`, `undershoot`, `settlingtime`) shipped in the §4.3 closure slice.
- **SPT Tier-4 — `digitalFilter` system object.** `designfilt`-style entry returning a filter handle that `filter`/`filtfilt`/`freqz` can polymorphically accept. Needs a new descriptor type alongside `matlab_dict` / `matlab_symmat`. Plus `dsp.SOSFilter` / `dsp.FIRFilter` HDL system objects for hooking the SystemVerilog backend onto the filter design path.
- **SPT Tier-4 — wavelets.** `cwt`, `dwt`/`idwt`, `wavedec`/`waverec`, `wvd` (Wigner-Ville), `fsst`/`ifsst` (Fourier synchrosqueezed). 2–3 weeks each, stand-alone algorithmic work.
- ~~**Sema follow-on — user functions shadow builtins.**~~ ✅ shipped in `5125af0` — `Scope::declare` now allows `Builtin → Function` and `Builtin → Class` promotion, so `function y = sin(x)` / `classdef filter` / etc. correctly override the same-name builtin in the current TU. The `square` user-fn fixture renamed to `squarem` in the SPT §4.2 slice (commit `39111c5`) can stay as-is — the rename is harmless and the fix works regardless.

---

## Near-term (~1 month)

### 1. HDL Verification with CocoTB 🔵

Wire generated SystemVerilog modules to a Python testbench harness
using [CocoTB](https://www.cocotb.org/). Each `examples/hdl/*.m`
module gets a paired `<name>_tb.py` that drives clk/rst, walks
through a handful of input vectors, and asserts the output matches
the MATLAB reference (run via the existing C/C++/Python emission of
the same source).

**Why it matters.** Today the SV pipeline is verified by
Verilator's lint pass (76 fixtures lint-clean) + Yosys generic
synth on the non-FSM subset. Lint catches
syntax / signedness / width issues but doesn't prove the RTL
*behaves* the same as the MATLAB source. CocoTB closes that gap:
the same MATLAB program is the golden reference and the
implementation under test.

**Scope.**
- New `test/EmitSVCocoTB/` directory mirroring `test/EmitSV/`.
- Helper script `just verify-cocotb <name>.m` that:
  1. Emits SV via `-emit-systemverilog`.
  2. Emits Python via `-emit-python` (the reference model).
  3. Runs CocoTB with a small driver that feeds the SV
     simulation and the Python model the same vectors per cycle.
  4. Diffs outputs cycle-by-cycle.
- CI lane `cocotb-tests` (gated on CocoTB + Verilator + Icarus
  presence; skip-if-missing rather than required).
- Per-module `*_tb.py` for the 8 `examples/hdl/` modules
  (`alu_16bit`, `mux_4to_1_16bit`, `counter_0_to_10`,
  `mealy_fsm`, `moore_fsm`, `vector_processor`,
  `sequential_processor`, `fir_asic_pipelined`).

**Out of scope (for v1).**
- UVM-style coverage / functional checking.
- Multi-clock testbenches.
- Proving timing closure.

**Dependencies.** None — both ends of the bridge already work.

**Effort.** ~1 week (harness + 8 testbenches + CI lane).

---

### 2. Tier 2: persistent fi-arrays in software backends 🔵

The remaining 3 of 8 `examples/hdl/` modules
(`fir_asic_pipelined`, `sequential_processor`, `vector_processor`)
use **persistent fi-arrays** —
`persistent buf; buf = fi(zeros(1, N), ...)` — which the SV path
lowers to N parallel registers via Stage F, but the C/C++/Python/TS
backends don't model.

**Why it matters.** Streaming / windowed signal-processing in pure
software is a real use case (FIR filters in C, sliding windows,
buffered DSP), not just an HDL idiom. Tier 2 unblocks all 8
HDL examples for software emission and makes the existing fi-array
support useful end-to-end.

**Scope.**
- C: `static T name[N] = {<init>};` at function entry; reads /
  writes through `name[k]`.
- C++: same.
- Python: `<fn>.<name> = [<init>] * N` at module scope; reads /
  writes through `<fn>.<name>[k]`.
- TS: `let <fn>_<name>: number[] = [<init>] * N;`.
- Recognize the `matlab_persistent_get_ptr → subscript1_s` and
  `matlab_persistent_set_ptr → array-of-stores` chains; suppress
  the runtime-call form and emit array indexing.

**Dependencies.** Tier 1 (shipped) recognizes the canonical
isempty pattern; Tier 2 extends the same recognition to the
array-typed persistent ABI.

**Effort.** ~3 days per backend × 4 backends = ~2 weeks.

---

### 3. SV codegen polish 🔵

The 8 HDL examples lint clean, but a few cosmetic / quality
issues remain that don't block synthesis but read awkwardly:

- **Storage-class literals on register width casts.**
  `count_reg <= 4'(8'sd0)` could just be `count_reg <= 4'sd0`.
  The wrap-cast is redundant when the source is a constant.
- **Saturate constant rendering.** Things like `64'sd68719476735`
  (= 2³⁶−1) read more naturally as `36'sh7FFF_FFFFF`. Cosmetic
  but DSP code is full of these.
- **`v0_1`, `v1_1`, ... synthetic intermediate names.** The
  saturate-clamp temps in `vector_processor` / `sequential_processor`
  / `fir_asic_pipelined` use compiler-generated names. Could
  derive semantic names from the surrounding context (e.g.
  `acc_clamped_1`, `prod_extended`) — much more readable RTL.
- **Comment-block placement on persistent declarations.** The
  source `% Estágio 0: Entradas` next to a `persistent delay_line`
  has no SV-side anchor right now (the declarations live in the
  prelude, not always_comb). Should attach to the prelude
  declaration block.

**Scope.** Each is independent; can be ordered by user impact or
done together.

**Effort.** ~2 days total.

---

### 4. SV codegen: pragma path for `-emit-c` / `-emit-cpp` 🔵

Today `% hdl: port(name, fi, signed, W, F)` pragmas are SV-only —
applying them to the C/C++/Python/TS pipelines would let function-
only `.m` files (no typed driver) compile to software too.

**Why it matters.** Asked-for already during the C/C++ audit
(`alu_16bit.m` standalone fails with `unsupported op: matlab.alloc`
because no driver pins types). Reusing the pragma machinery is
the smallest fix.

**Scope.** Lift the `IsSVPath` gate on `runApplyPortTypePragmas`
in `tools/matlabc/main.cpp`; verify nothing else gates on the
SV-only assumption.

**Effort.** ~30 min + regression check.

---

### 5. SPT follow-ons (highest-leverage gaps) 🔵

The SPT arc closed the practical "design / apply / inspect / measure"
surface; the highest-leverage open items, in priority order:

- **§2.1 IIR family completion (tail).** The bulk of §2.1 shipped:
  band variants HP/BP/BS for `butter`/`cheby1`/`cheby2`, `besself`,
  standalone `bilinear`/`freqs`, `cheb2ord`, `tf2zp`/`zp2tf` /
  `tf2sos`/`sos2tf` form conversions. Open: `ellip` + `ellipord`
  (Jacobi elliptic functions), the analog prototype builtins
  (`buttap`/`cheb1ap`/etc.) as standalone 3-return entries, and
  `tf2ss` / `ss2tf` / `zp2sos`. **Effort:** ~2 sessions.
- **§2.5 strict Gustafsson `filtfilt`.** The lfilter_zi-based
  steady-state IC path (scipy's pad-method default) shipped — constant
  signals now preserved exactly. The strict 1996 Gustafsson method
  (scipy's method='gust') uses an explicit edge-elimination linear
  system instead of padding; that and `phasez` / `zerophase` are
  still open. **Effort:** ~3 sessions.
- **§4.3 follow-on — `findpeaks` name-value options.** `MinPeakHeight`,
  `MinPeakDistance`, `MinPeakProminence`, `Threshold`, `SortStr`. The
  pulse-statistics tail (`slewrate`, `pulseperiod`, `pulsewidth`,
  `overshoot`, `undershoot`, `settlingtime`, `statelevels`) shares
  the same edge-detection scaffolding. **Effort:** ~3 sessions.
- **§3 multitaper + STFT.** `dpss` + `pmtm` for high-resolution
  spectral estimation; `stft` / `istft` (with COLA inversion) to close
  the time-frequency tail alongside the existing `spectrogram`.
  **Effort:** ~1 week.
These are independent and can land in any order; pick by what
unblocks real workflows.

The Sema "user-functions shadow builtins" follow-on that previously
appeared here landed in `5125af0` and is no longer open.

---

### 6. Runtime: arena allocator + leak audit 🟡

The C runtime currently uses `malloc`/`free` per matrix +
ref-counting on some paths. Two pain points:

- **Allocator pressure** in tight loops (e.g. `for i = 1:1000;
  A = A + B; end` allocates a fresh result matrix each iter).
- **No leak tracking surface.** Programs that genuinely leak
  (held refs in REPL workspace) are invisible until ASAN.

**Scope.**
- Per-call arena reset for the AOT-compiled paths.
- `MATLAB_RT_TRACE=1` env-var prints `alloc / free / leak`
  summary at exit.
- Optional: bump-allocator with explicit reset in JIT-mode for
  long REPL sessions.

**Effort.** ~1 week.

---

## Mid-term (~1–3 months)

### 7. Block language (visual nodes → AST → MLIR) 🟢

**v1 shipped.** The MatForge IDE now saves `.mflow` JSON files
that `matlabc` and `matlab-lsp` both consume. The implementation
chose graph → AST (rather than the originally planned graph →
MLIR direct), which got every existing backend — LLVM / C / C++ /
Python / TS / SV / fixed-point / hardware-report — for free, plus
a free `-emit-matlab` round-trip via the existing `formatAST`.

Five phases shipped:
- **1.** JSON loader + schema validation, byte-precise diagnostics
  (`-dump-flow`).
- **2.** Linear chain → AST: `variable`, `expression`, `display`,
  `input`, `assignment`, `constant`, `function_call`,
  `matrix_literal`.
- **3.** Structured control flow: `if` / `for` / `while` /
  `break` / `continue` / `return`, arbitrary nesting; refuses
  irreducible CFGs.
- **4.** Sub-flows lifted to top-level `Function`s;
  `function_definition` and `subflow_call` blocks.
- **4b.** `custom` blocks with three provenance modes: inline
  `source` / sibling `path` / `library_id` (resolved via
  `--block-path` + `MATFORGE_BLOCK_PATH`); function-insertion
  dedup; arity validation.
- **5.** Cross-backend round-trip lane (`.mflow` ≡
  round-tripped `.m` across C / C++ / Python / TS); `matlab-lsp`
  accepts `.mflow` URIs.

8 examples under `examples/mflow/` and 4 ctest lanes.
See [`flowchart_frontend.md`](flowchart_frontend.md) and the
shipped row in [`feature_status.md`](feature_status.md).

**Open follow-ups (v2 territory, not blocking).**
- Richer block library: `Delay (z⁻¹)`, `FIR`, `IIR (DF-II)`,
  `FSM (state diagram)`, `Counter`, `Accumulator` as primitive
  block kinds rather than custom blocks. Each becomes a small
  Phase-2/3-style render rule.
- Round-trip text ↔ blocks editing (currently one-way).
- 2-D / image-pipeline blocks — overlaps with item #7.

---

### 8. Improve HDL codegen: 2-D fi matrices + RAM inference 🔵

The biggest remaining SV scope gap. Today the pipeline supports
1-D fi arrays (shipped via Stage E + Stage F); 2-D matrices are
needed for image-processing pipelines and matrix-multiply HDL.

**Scope.**
- 2-D fi storage: `logic signed [W-1:0] mem [R][C]` declaration
  + 2-D subscript reads / writes.
- RAM inference for large 2-D persistents:
  `persistent buf; buf = fi(zeros(1, 1024), ...)` should infer
  a synth-tool-recognized SRAM block (`always_ff @(posedge clk)
  if (we) mem[addr] <= din;`) instead of 1024 parallel registers.
- Shape recognition: differentiate "small N for shift register
  → N parallel regs (Stage F today)" from "large N for
  data buffer → RAM block".

**Effort.** ~2 weeks.

---

### 9. SystemVerilog → MATLAB (reverse direction) 🔵

Take legacy synthesizable SV (or simple sequential RTL with
clocked persistent state) and lift it into MATLAB source for
verification, simulation, or porting.

**Why it matters.**
- HDL teams often have SV reference implementations and want
  to iterate on the algorithm in MATLAB (faster, with NumPy
  / matplotlib).
- Lets a designer take an existing IP block, lift it to MATLAB,
  modify it, and re-emit to SV via the existing forward path —
  closing the loop.

**Scope (v1).**
- Lex + parse a synthesizable SV subset:
  - `always_ff` (single clock + sync/async reset).
  - `always_comb` (combinational logic).
  - `unique case` and `if/else` chains.
  - `logic [N-1:0]` and `logic signed [N-1:0]` register declarations.
  - One-hot and binary-encoded `typedef enum` FSMs.
- Lift to a typed MATLAB AST:
  - SV register → `persistent` MATLAB var with
    `if isempty(_); _ = init; end` reset pattern (the same
    idiom Tier 1 recognizes).
  - `unique case (state)` → `switch state`.
  - Sized integer literals → `fi(_, signed, W, F)`.
- Output: pretty-printed MATLAB source via the existing
  formatter.

**Out of scope (for v1).**
- Verilog (only SystemVerilog).
- Multi-clock / CDC handling.
- Generate blocks / parameterized modules.
- Behavioral SV beyond the synthesizable subset.

**Dependencies.** None new — uses the existing AST + formatter.

**Effort.** ~3 weeks.

---

### 10. REPL: line editing + history + JIT cache 🟡

Today `matlabc -repl` is a minimal stdin loop. The major missing
ergonomics:

- **Readline** for history navigation (↑/↓), Ctrl-R search,
  Ctrl-A / Ctrl-E line motion.
- **Multi-line input** for `for ... end`, `function ... end`,
  `if ... end` blocks (today everything must be on one line).
- **Persistent JIT cache** keyed by hashed source so repeated
  function definitions don't re-JIT cold.
- **Tab completion** for variables in workspace + builtins.

**Effort.** ~1.5 weeks (most of it is editline / linenoise
integration; the rest is JIT cache wiring).

---

### 11. Improve HDL codegen: pipelining + retiming 🔵

Beyond the v1 stage-F register split, the pipeline doesn't
automatically rebalance critical paths. For DSP designs that need
to hit a target frequency, this matters.

**Scope.**
- `% hdl: target_freq(N_MHZ)` pragma.
- Compute critical-path estimate per always_comb block (op count
  × per-op latency table).
- Insert pipeline registers when the path exceeds budget.
- Already-shipped scaffolding: `-sv-input-pipeline=N` / `-sv-output-
  pipeline=N` for fixed-stage pipelining.

**Out of scope.** Sophisticated retiming (moving registers
across logic). Just insertion at safe boundaries.

**Effort.** ~2 weeks.

---

## Long-term / exploratory

### 12. MATLAB graphics / `plot` (limited) 🔵

For demos and tutorials. Render `plot(x, y)`, `bar(...)`,
`imagesc(...)` to PNG / SVG via a small wrapper around matplotlib
(Python path) or directly to PNG via stb_image_write (C path).
Not pixel-perfect MATLAB; just enough for quick visualization
of compiled programs.

**Effort.** ~1 week per output target.

---

### 13. `.mat` file save / load 🔵

Already documented in [`docs/save_load_compat.md`](save_load_compat.md).
Goal: read MATLAB v7.3 (HDF5-based) `.mat` files into the runtime
workspace and vice versa. Not a full MATLAB compatibility matrix;
just the common cases (`save('out.mat', '-v7.3')` followed by
`load('out.mat')` in another session).

**Effort.** ~2 weeks.

---

### 14. Toolbox stubs for symbolic / optimization 🔵

Single-file stubs that route to the equivalent open-source
library (`sympy` for Symbolic Math Toolbox, `scipy.optimize`
for Optimization Toolbox). Limited surface; just enough to make
common textbook MATLAB programs that use these toolboxes
compile and run.

**Effort.** Small per stub; total scope depends on which
toolbox.

---

## What's intentionally NOT on the roadmap

- **Full MATLAB language compatibility.** Pursuing this leads to
  toolbox dependencies and `.mat` file format edge cases that
  defeat the project's "self-contained, MathWorks-free" design
  goal.
- **GUI primitives** (`uicontrol`, `app designer` apps). The
  graphics roadmap entry above is rendering-only, no interaction.
- **Live Editor / `.mlx`** notebook format.
- **MEX file compatibility.** The runtime ABI is stable inside
  this project; cross-compatibility with MathWorks's MEX C
  interface is a separate engineering effort that brings little
  benefit since users on this stack want a MathWorks-free path.
- **Code obfuscation / encryption** (MathWorks `.p` files).

---

## Cross-cutting quality work

These don't fit a single roadmap slot but get folded into other
work as it lands:

- **Test corpus growth.** Original ≥150 run-tests + 50 SV goldens target met during the data-container arc; current corpus is **189 run-tests** + **77 SV goldens**. Next milestone: ≥300 run-tests as Tier-2 SPT follow-ons + multirate / waveform exercise more of the surface, and growing the SV-cocotb cycle-by-cycle lane (item #1) to cover all 39 HDL examples rather than just 8.
- **Formatter idempotency** verified by a fixed-point CI lane
  (parse → format → parse → format → identical).
- **Doc-up-to-dateness check** as a CI step (parse `feature_status.md`,
  verify every claimed `✅` has at least one test).
- **Performance benchmarks** baseline-tracked across releases —
  matrix-multiply / FIR / FFT / parfor reduction at a few sizes,
  recorded per commit.

---

## Update cadence

This file is updated at the end of each multi-week implementation
arc — most recently after the **Signal Processing Toolbox arc**
(SPT 1–13: windows, polynomial helpers, IIR/FIR design, close-the-loop,
transforms, spectral, LP, time-frequency, pulse measurements, multirate,
waveform generators, alignment — see "Recently shipped" above), prior
to that the data-container + multi-return arc (Phases 1.1 / 1.2 / 1.3 /
2 / 3 / 4 / 5.1 / 5.2 / 5.3), prior to that the SystemVerilog Phase 5.6
closure and the multi-backend persistent + isempty Tier 1.

Items get demoted from this roadmap to `feature_status.md` /
the relevant `emit_*.md` once shipped. Items get retired (no
demote) when the design has been superseded by a different
approach.
