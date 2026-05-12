# Verilog-A emission — design and tiered plan (2026-05-12)

Companion to `docs/emit_systemverilog.md`.  Where the SV backend
lowers a digital MATLAB subset into synthesizable RTL, this plan
scopes a **second analog/AMS backend** that lowers a *continuous*
MATLAB subset into Verilog-A (and, where the construct is mixed-
signal, Verilog-AMS).

The immediate motivation is to close one of the carved-out items
from `docs/rf_toolbox_plan.md` — `rfmodel.rational/writeVerilogA`
(behavioral Laplace export of a fitted S-parameter model).  But
the design generalizes well past RF: Control System Toolbox
`tf`/`zpk`/`ss`, Signal Integrity CTLE, behavioral ADC/DAC, PLL/
VCO/CDR, comparators, sensor models, transmission-line models,
analog signal sources, and noise models are all natural targets.

Per the existing two-backend split in `docs/emit_systemverilog.md`:

    MATLAB subset
    ├── Digital subset  →  SystemVerilog        (-emit-systemverilog)
    └── Analog subset   →  Verilog-A / -AMS     (-emit-verilog-a)

The mental rule is:

> If the MATLAB describes a continuous relationship between
> voltage, current, time, frequency, phase, noise, or physical
> state, **Verilog-A** is the right target.
> If it describes clocked logic, registers, FSMs, pipelines or
> bit-accurate arithmetic, **SystemVerilog** is the right target.

## 1. Goals

- Emit **vendor-neutral Verilog-A** (`.va`) consumable by
  ngspice (>=42 with the OpenVAF backend), Xyce (>=7.5), Cadence
  Spectre, Mentor Eldo, Synopsys CustomSim, Keysight ADS.
- **Pure Verilog-A only, initially.**  Mixed-signal Verilog-AMS
  (digital bit-bus ports, `connectmodule` / `connectrules`
  discipline resolution) is deferred to a separate later tier
  (§6 Tier-11) — the initial roadmap stays in one language.
- Keep the source MATLAB **fully executable** through every other
  backend.  An RC lowpass written for Verilog-A export should
  still produce numerical results via `matlabc -emit-llvm`, work
  in the REPL, step through under `matlabc -dap`, and plot via
  the existing plotting runtime.  Verilog-A emission is one more
  lane, not a separate sub-language.
- Close the RF-Toolbox `writeVerilogA` carve-out as the first
  shipped tier.

## 2. Non-goals

- Best-effort emission of arbitrary MATLAB scripts.  The emitter
  rejects anything outside the supported analog subset with a
  source-level diagnostic, exactly like the SV gate does today.
- Digital algorithms (FFT, FIR, image processing, ML, file I/O,
  string handling, large in-runtime matrix ops).  Those stay on
  the SV / C / Python / TS lanes.
- Full mixed-signal solver — no harmonic balance, no shooting,
  no envelope methods.  Just behavioral emission; simulation is
  the third-party simulator's job.
- Layout / parasitic extraction / PEX-aware modeling.  Out of
  scope.

## 3. What MATLAB constructs map to Verilog-A

The full landscape, grouped by physical class.  Each row in the
table below becomes a discrete row in the Tier-X implementation
plan in §6.

### 3.1 Continuous filters (rational transfer functions)

The cleanest map.  MATLAB already has the right surface area —
`tf`, `zpk`, `ss`, `butter('s')`, `cheby1('s')`, `bessel`, plus
the RF Toolbox `rfmodel.rational` from Vector Fitting.

MATLAB:
```matlab
s = tf('s');
H = 1 / (s^2/w0^2 + s/(Q*w0) + 1);
% or:
[num, den] = butter(2, 2*pi*1e6, 's');
% or (from RF Toolbox VF):
mdl = rationalfit(freqs, h_re, h_im, 6, 30);
```

Verilog-A:
```verilog-a
analog begin
    V(out) <+ laplace_nd(V(in), num_coefs, den_coefs);
end
```

For `rfmodel.rational` (sum-of-poles form `H(s) = D + Σ Cᵢ/(s − Aᵢ)`),
emit a `laplace_zp` or expanded numerator/denominator after
multiplying out the poles to real-coefficient form (Verilog-A
needs real coefficients; complex-pair poles fold to second-order
sections):

```verilog-a
V(out) <+ exp(-s*delay) * (D + S_section1(V(in)) + S_section2(V(in)) + ...);
```

Targets in MATLAB land:
- `tf(num, den)` — direct `laplace_nd`.
- `zpk(z, p, k)` — expand to num/den, or emit `laplace_zp(V(in), zeros, poles, k)`.
- `ss(A, B, C, D)` — emit per-state-variable `ddt(xᵢ) <+ ...` form (see §3.2).
- `butter` / `cheby1` / `cheby2` / `bessel` with `'s'` flag — already
  return analog-domain `(num, den)` pairs in the existing SPT
  runtime; route straight into `laplace_nd`.
- `rfmodel.rational` — fold complex pairs into real biquads,
  emit as sum of SOSs or as a single `laplace_zp`.

### 3.2 Differential equations / state-space

Any continuous system described by ODEs maps onto `ddt()`.

MATLAB:
```matlab
% dx/dt = -a*x + b*u
% y     = c*x
```

Verilog-A:
```verilog-a
analog begin
    ddt(x) <+ -a*x + b*V(in);
    V(out) <+ c*x;
end
```

For state-space `ss(A, B, C, D)` of order N:
```verilog-a
real x[0:N-1];
analog begin
    ddt(x[0]) <+ A00*x[0] + A01*x[1] + ... + B0*V(in);
    ddt(x[1]) <+ A10*x[0] + A11*x[1] + ... + B1*V(in);
    ...
    V(out) <+ C0*x[0] + C1*x[1] + ... + D*V(in);
end
```

The constraint is that the MATLAB source must look like
*equations*, not like *integration loops*.  We **do not** try to
lower `ode45(f, tspan, y0)` to Verilog-A — instead, the user
writes the right-hand side `f` and tags it with
`%#verilog-a state-space` or equivalent.  The emitter recognizes
the pattern.

### 3.3 Analog signal sources

MATLAB:
```matlab
y = A*sin(2*pi*f*t);
y = A*exp(-t/tau);
y = chirp(t, f0, t1, f1);
y = square(2*pi*f*t);
```

Verilog-A (using `$abstime` as the continuous time source):
```verilog-a
V(out) <+ A * sin(2*`M_PI*f*$abstime);
V(out) <+ A * exp(-$abstime/tau);
```

Useful as analog stimulus inside a testbench module the user
authors in MATLAB, ships through the existing pipeline (so they
can plot the stimulus, sanity-check it), and then emits as a
behavioral source module.

### 3.4 Comparators, Schmitt triggers, limiters

MATLAB:
```matlab
if vin > vth, y = 1; else y = 0; end
```

Verilog-A — naive form:
```verilog-a
V(out) <+ transition(V(in) > vth ? vh : vl, td, tr);
```

Verilog-A — recommended event-driven form (gives the simulator
explicit cross events, avoids the time-step backsearch tax):
```verilog-a
@(cross(V(in) - vth, +1)) state = 1;
@(cross(V(in) - vth, -1)) state = 0;
analog V(out) <+ transition(state ? vh : vl, td, tr);
```

Schmitt trigger: emit two `cross()` events at vhigh and vlow.

### 3.5 VCO / NCO / PLL components

Phase accumulation maps to `idtmod()`:

MATLAB:
```matlab
phase = phase + 2*pi*freq*dt;
y     = sin(phase);
```

Verilog-A:
```verilog-a
real phase;
analog begin
    phase = idtmod(2.0*`M_PI*freq, 0.0, 2.0*`M_PI);
    V(out) <+ amp * sin(phase);
end
```

For a charge-pump PLL the full loop becomes:
- VCO: as above (frequency input → phase output)
- Phase detector: a `cross()`-driven event comparing two phases
- Loop filter: a `laplace_nd` (often a PI: `1 + k/s`)

### 3.6 DAC behavioral models (pure Verilog-A)

DAC (analog-out) is pure Verilog-A:
```verilog-a
V(out) <+ transition(vref * code / ((1 << N) - 1), td, tr);
```

Quantization, INL/DNL, offset, gain error all expressible as
parameters of the same template.

ADC behavioral models (digital bit-bus outputs) need
Verilog-AMS — deferred to Tier-11.  In the meantime, an
"analog-level ADC" (one electrical port per bit driven by
`transition()` to `vh` / `vl`) can be expressed in pure
Verilog-A if a user really needs it; it's not bit-bus-clean but
it simulates anywhere.

### 3.7 Sensor models (phenomenological)

Linear or low-order-polynomial sensor responses fold straight
into Verilog-A:

```matlab
vout = sensitivity * pressure + offset;
r    = R0 * (1 + alpha*(T - T0));
```

```verilog-a
V(out) <+ sensitivity * V(pressure) + offset;
I(p, n) <+ V(p, n) / (R0 * (1 + alpha*($temperature - T0)));
```

Verilog-A's `$temperature` and `$abstime` are first-class — the
emitter recognizes MATLAB references to `T` (a parameter the
user has annotated as `%#verilog-a $temperature`) and rewrites
on the fly.

### 3.8 Compact circuit components

The classic Verilog-A demos — drop-in MATLAB equivalents:

| MATLAB                     | Verilog-A                                  |
|---|---|
| `i = C * diff(v) / diff(t)`| `I(p,n) <+ C * ddt(V(p,n));`               |
| `v = L * diff(i) / diff(t)`| `V(p,n) <+ L * ddt(I(p,n));`               |
| `i = v / R`                | `I(p,n) <+ V(p,n) / R;`                    |
| `i = Is * (exp(v/Vt) - 1)` | `I(p,n) <+ Is * (exp(V(p,n)/Vt) - 1);`     |
| `i = gm * vgs`             | `I(d,s) <+ gm * V(g,s);`                   |
| `vout = A * (vp - vn)`     | `V(out) <+ A * V(vp, vn);`                 |
| `vout = sat(A*vin)`        | `V(out) <+ tanh(A*V(in)) * vsat;`          |

### 3.9 Noise

MATLAB:
```matlab
y = signal + sigma * randn();
```

Verilog-A — **NOT** `randn()`-per-step (that would be wrong; the
simulator's noise analysis needs PSD, not samples):

```verilog-a
V(out) <+ signal + white_noise(sigma*sigma, "thermal");
V(out) <+ flicker_noise(K, alpha, "flicker_1_over_f");
```

The emitter requires the user to **annotate** `randn()` calls
with their noise type:
```matlab
y = signal + randn() * sigma;   %#verilog-a white_noise sigma^2 "thermal"
y = signal + randn() * K;       %#verilog-a flicker_noise K 1.0 "1_over_f"
```

Without an annotation, the emitter rejects with a diagnostic.
This is intentional: silent reinterpretation of stochastic
semantics is worse than a clear error.

### 3.10 Lookup tables / non-linear curves

`interp1` of a static table → `$table_model` (Verilog-A 2.4+):

```matlab
y = interp1(v_table, i_table, vin);
```

```verilog-a
I(out) <+ $table_model(V(in), "iv_curve.tbl", "1L,L");
```

For simulators without `$table_model` (older ngspice / Xyce),
fall back to piecewise-polynomial coefficients emitted inline.

## 4. What we explicitly do NOT emit to Verilog-A

The same rejection criteria as `docs/emit_systemverilog.md` for
*digital* — but specialized for analog:

- **Digital algorithms** — FFT, FIR filtering on samples,
  bit-accurate arithmetic, image processing, sorting, search.
  → use the SV / C / Python / TS lanes.
- **Strings, file I/O, plotting calls** — meaningless in
  Verilog-A.
- **Unbounded loops / recursion in runtime code** — Verilog-A is
  evaluated per-time-step.  Bounded for-loops over a fixed
  `parameter` are fine and unrolled.
- **In-runtime dynamic-size arrays / cells / structs / tables**
  — Verilog-A reals only.
- **`pause` / `input` / `eval` / `keyboard`** — same as SV.
- **`ode45(f, tspan, y0)`** — emit the *right-hand side* `f`
  with `ddt()`, not the solver loop itself.

The emitter rejects each of these with a precise diagnostic
pointing at the offending source line, matching the existing
`-check-synthesizable` gate pattern.

## 5. Backend architecture

### 5.1 Where it lives

Mirror the SV backend layout:

| Component                                | New path                                |
|---|---|
| Emitter                                  | `lib/MLIR/Emit/EmitVerilogA.cpp`        |
| Analog-subset gate (rejection diagnostics)| `lib/MLIR/Passes/CheckAnalog.cpp`       |
| CLI flag                                 | `-emit-verilog-a` (alias: `-emit-va`)   |
| Annotation parser                        | `lib/Sema/AnalogAnnotations.cpp`        |
| Lowering helpers (laplace, ddt patterns) | `lib/MLIR/Passes/LowerAnalog.cpp`       |
| RF Toolbox `writeVerilogA` runtime       | extend `runtime/runtime_rf.cpp` to call into the emitter as a runtime function |
| Golden tests                             | `test/EmitVA/<name>.{m,va.expected}`    |
| Diagnostic gate tests                    | `test/EmitVAFail/<name>.{m,err.expected}`|
| CTest lanes                              | `run-emit-va`, `run-emit-va-fail`, `run-emit-va-admslint` |

A second `-emit-verilog-ams` / `.vams.expected` lane lands with
Tier-11 (mixed-signal extensions); it's intentionally not in
the initial backend wiring.

### 5.2 Routing

The CLI flag activates a new MIR → Verilog-A walker.  The walker
is independent of the SV walker — analog and digital share the
same MIR but the analog walker rejects everything outside the
analog subset.

A single MATLAB file can target both backends only if its body
is in the *intersection* (e.g., pure algebraic expressions on
inputs to outputs).  In practice the file declares its target
via a top-of-file pragma:

```matlab
%#emit-target verilog-a
function vout = rc_lowpass(vin, R, C)
    persistent vout_state;
    if isempty(vout_state), vout_state = 0; end
    %#verilog-a ddt
    vout = (vin - vout_state) / (R*C);
    vout_state = vout;
end
```

The pragma is **only consumed by the Verilog-A emitter**.  Every
other backend ignores it (so the function still compiles + runs
on the LLVM / C / Python / TS lanes for unit testing).

### 5.3 Pattern-recognition rules

The emitter recognizes a small, explicit set of MATLAB patterns
and rewrites each to the corresponding Verilog-A primitive:

| MATLAB pattern                                       | Verilog-A primitive          |
|---|---|
| `tf(num, den)` returned + applied via `lsim`/`filter`-on-s | `laplace_nd(V(in), num, den)` |
| `zpk(z, p, k)`                                        | `laplace_zp(V(in), z, p, k)`  |
| `ss(A,B,C,D)` with state update `x = x + A*x*dt + B*u*dt` | unroll into `ddt(x[i]) <+ ...` |
| `y_state = y_state + rhs * dt` (annotated `ddt`)      | `ddt(y_state) <+ rhs`         |
| `if vin > vth ... else ... end` (annotated `cross`)   | `@(cross(V(in)-vth,+1)) ...`  |
| `sin(2*pi*f*$abstime)` / `cos(...)`                   | passthrough                   |
| `interp1(x, y, vin)` with constant `x,y`              | `$table_model` or piecewise   |
| `randn()` (annotated)                                 | `white_noise` / `flicker_noise`|
| `R0*(1 + alpha*(T - T0))` with `T` annotated `$temperature` | `$temperature` substitution |
| `C * diff(v)/diff(t)` (annotated `ddt`)               | `C * ddt(V(p,n))`             |

### 5.4 Discipline / port inference

The emitter needs to know which MATLAB inputs/outputs are
electrical, which are pure real signals, and which are
temperature / time / other natures.  Two paths:

1. **Pragma-driven** (default — simple, explicit):
   ```matlab
   %#verilog-a port vin electrical
   %#verilog-a port vout electrical
   %#verilog-a port temp $temperature
   function vout = rc_lowpass(vin, temp)
       ...
   end
   ```
2. **Naming convention fallback**: arguments named `v*` or `vin`
   /`vout` default to `electrical`; arguments named `i*` default
   to `electrical` (current branch); anything else stays `real`.
   Pragmas override.

### 5.5 Verilog-A language primer (what we emit)

The emitter targets Verilog-A 2.4 (LRM 2014).  Features used:

- Disciplines: `electrical`, `thermal`, `kinematic`, `rotational`
  (via `disciplines.vams`).
- Operators: `ddt(x)`, `idt(x)`, `idtmod(x, lo, hi)`,
  `absdelay(x, td)`, `delay(x, td)`.
- Transfer functions: `laplace_nd(x, num, den)`,
  `laplace_zd(x, z, d)`, `laplace_zp(x, z, p, k)`.
- Smoothing: `transition(x, td, tr)`, `slew(x, max_rise,
  max_fall)`.
- Events: `@(cross(expr, dir))`, `@(timer(t, period))`,
  `@(initial_step)`, `@(final_step)`.
- Noise: `white_noise(pwr, name)`,
  `flicker_noise(pwr, exp, name)`, `noise_table`.
- Built-in variables: `$abstime`, `$temperature`, `$vt(T)`,
  `$realtime`.
- Branches: `V(p, n)`, `I(p, n)`, contribution `<+`.

### 5.6 Why we do not auto-translate `ode45`

The RHS of an ODE *is* analog modeling.  The solver *loop* is
not — Verilog-A simulators have their own adaptive integrators.
Asking the emitter to "lower `ode45` to `ddt`" would mean
recognizing a numerical-method loop, undoing it, and recovering
the symbolic RHS — fragile and unnecessary.  The user is
expected to write the RHS as a separate function and tag it.

## 6. Tiered implementation plan

Effort estimates assume one focused implementation session ≈ 4h.

### Tier-1 — RF Toolbox `writeVerilogA` (closes carve-out)  ~4 sess

Target: `rfmodel.rational/writeVerilogA(mdl, filename)` from the
RF Toolbox closes its biggest carved-out item with a minimal,
self-contained Verilog-A emitter.

Steps:
1. `runtime/runtime_va_emit.cpp` — bare emitter that writes a
   `.va` file from a `(poles, residues, D, delay)` tuple.  Fold
   complex-conjugate pairs into real-coefficient biquads.
2. `writeVerilogA(mdl, filename)` runtime entry.  Output is a
   parameterized one-port module:
   ```verilog-a
   `include "disciplines.vams"
   module rfmodel_rational(in, out);
       electrical in, out;
       parameter real D = ...;
       parameter real delay = ...;
       analog begin
           V(out) <+ delay > 0.0
                   ? absdelay(D*V(in) + sos_sum(V(in)), delay)
                   : (D*V(in) + sos_sum(V(in)));
       end
   endmodule
   ```
3. `RFRational.writeVerilogA(filename)` classdef method.
4. Tests: `test/EmitVA/rf_rational_va_*.m` round-tripping the
   shipped VF fixtures through emit + (optionally) ngspice + (a
   `freqresp` cross-check that the emitted Verilog-A simulated
   in ngspice matches `freqresp(mdl, freqs)` to 1 % over band).
5. Doc update: `docs/rf_toolbox_plan.md` carved-out → shipped.

Test corpus delta: +5 fixtures.

### Tier-2 — Continuous filters via `tf` / `zpk` / `butter('s')`  ~3 sess

Target: any 1-input 1-output rational filter authored as `tf`
or `zpk` or returned from `butter('s')` / `cheby1('s')` etc.,
emits as `laplace_nd` or `laplace_zp`.

Steps:
1. CLI flag `-emit-verilog-a`.
2. MIR walker: recognize `tf(num, den)` + an apply pattern
   (`y = filter_tf(H, u)`) as the canonical input.
3. Emit module signature, parameter block, single `laplace_nd`
   contribution.
4. Tests: 2nd-order LP / HP / BP / BS, biquads, Butterworth-2,
   Chebyshev-3.
5. Add `examples/verilog_a/rc_lowpass.m`,
   `examples/verilog_a/butter_lp.m`,
   `examples/verilog_a/biquad.m`.

Test corpus delta: +8 fixtures.

### Tier-3 — State-space `ss(A,B,C,D)` → `ddt` array  ~3 sess

Target: continuous state-space objects emit one `ddt(x[i])`
contribution per state, plus the output equation.

Steps:
1. Recognize `ss(A,B,C,D)` constructed from constant matrices.
2. Emit `real x[0:N-1]`; per-state `ddt(...)`; output `V(out)
   <+ ...`.
3. Tests: 2nd-order LP via state-space, observer canonical form
   biquad.
4. Examples: `examples/verilog_a/ss_lp2.m`,
   `examples/verilog_a/ss_observer_biquad.m`.

Test corpus delta: +4 fixtures.

### Tier-4 — Sources, comparators, cross-events, transition  ~3 sess

Target: stimulus generators (`sin` / `exp` / `chirp` /
`square`), comparators, Schmitt triggers.

Steps:
1. Recognize `A*sin(2*pi*f*t)` patterns where `t` is annotated
   `$abstime`.
2. Recognize comparator `if vin > vth` patterns annotated
   `%#verilog-a cross`.
3. Emit `@(cross(...))` event blocks + `transition()`
   contributions.
4. Examples: `examples/verilog_a/sine_source.m`,
   `examples/verilog_a/comparator.m`,
   `examples/verilog_a/schmitt.m`.

Test corpus delta: +6 fixtures.

### Tier-5 — VCO / NCO / charge-pump PLL  ~2 sess

Target: phase-accumulator-based oscillators using `idtmod()`.

Steps:
1. Pattern: phase accumulator `phase = idtmod_rhs(2*pi*freq)`
   annotated `%#verilog-a idtmod 0 2*pi`.
2. Emit `phase = idtmod(2*`M_PI*V(in), 0, 2*`M_PI*1.0);`.
3. Examples: `examples/verilog_a/vco.m`,
   `examples/verilog_a/pll_charge_pump.m`.

Test corpus delta: +3 fixtures.

### Tier-6 — Behavioral DAC (pure Verilog-A)  ~2 sess

Target: parameterized DAC emitted as pure Verilog-A.

Steps:
1. `transition(vref * code / (2^N - 1), td, tr)` contribution
   with parameters for `N`, `vref`, `td`, `tr`, INL / DNL /
   gain error / offset.
2. Examples: `examples/verilog_a/dac_behav.m`,
   `examples/verilog_a/dac_inl_dnl.m`.

ADC (digital bit-bus output) is deferred to Tier-11
(Verilog-AMS).

Test corpus delta: +2 fixtures.

### Tier-7 — Compact components + sensor models  ~2 sess

Target: the §3.8 + §3.7 tables.

Steps:
1. Library of small examples (R, L, C, diode, transcondutor,
   op-amp, saturated op-amp, RTD, thermistor, photodiode).
2. Each is a 1-file MATLAB module with port annotations + the
   contribution equation.
3. Examples: `examples/verilog_a/{resistor, inductor, capacitor,
   diode, opamp_ideal, opamp_sat, rtd_pt100, thermistor,
   photodiode}.m`.

Test corpus delta: +9 fixtures.

### Tier-8 — Noise (white + flicker)  ~2 sess

Target: `randn()` calls annotated with noise type emit
`white_noise()` / `flicker_noise()`.

Steps:
1. Annotation parser for `%#verilog-a white_noise <pwr> "<name>"`.
2. Rejection diagnostic for un-annotated `randn()` in analog
   modules.
3. Examples: `examples/verilog_a/{noise_thermal, noise_flicker,
   noise_resistor}.m`.

Test corpus delta: +3 fixtures.

### Tier-9 — Lookup tables  ~2 sess

Target: `interp1` of constant tables emits `$table_model` (with
piecewise-polynomial fallback for older simulators).

Steps:
1. Recognize constant-table `interp1` calls.
2. Emit `$table_model(V(in), "<filename>.tbl", "1L,L")` or
   piecewise inline.
3. Side-emit the `.tbl` file alongside the `.va`.
4. Examples: `examples/verilog_a/{iv_curve_table,
   sensor_lookup}.m`.

Test corpus delta: +2 fixtures.

### Tier-10 — Polish + extras  ~3 sess

- `-emit-va-strict` option that runs ADMS / OpenVAF lint on the
  emitted file (similar to the Verilator lint pass in the SV
  lane).
- ngspice (or Xyce) co-simulation lane — `run-emit-va-cosim`
  CTest target: emit, simulate, compare against the MATLAB
  reference (using the in-tree `lsim` / `freqresp` /
  `timeresp` runtime).
- Doc: `docs/emit_verilog_a.md` user-facing reference (mirrors
  `docs/emit_systemverilog.md`).

Test corpus delta: optional cosim lane (~10 fixtures).

### Tier-11 — Verilog-AMS extensions (deferred)  ~4 sess

Target: lift the pure-VA constraint and emit Verilog-AMS where
the construct genuinely needs it.

Steps:
1. CLI flag `-emit-verilog-ams` (alias `-emit-vams`).
2. ADC with digital bit-bus output (`reg [N-1:0]` driven by
   `@(cross(V(clk)-vth,+1))` sample event).
3. `connectmodule` / `connectrules` discipline resolution for
   user-authored mixed-signal interconnects.
4. `test/EmitVAMS/<name>.{m,vams.expected}` golden lane and
   `run-emit-vams` CTest target.
5. Examples: `examples/verilog_a/adc_behav.m` (moves into
   `.vams.expected` here),
   `examples/verilog_a/comparator_digital_out.m`.

Test corpus delta: +3 `.vams` fixtures.  Gated separately from
Tiers 1–10 so the initial roadmap can land as pure VA.

### Total

~26 sessions of pure Verilog-A (Tiers 1–10) + ~4 sessions for
Verilog-AMS (Tier-11).  Tier-1 alone (~4 sess) closes the
RF-Toolbox `writeVerilogA` carve-out and unblocks shipping the
RF Toolbox at *100 % + 1*.

## 7. Examples layout — `examples/verilog_a/`

Each example is a single `.m` file that:

1. Compiles + runs through `matlabc -emit-llvm` and produces
   numerical output (so the user can sanity-check the
   continuous-time behavior with the existing plotting /
   integration runtime).
2. Works in the REPL: `matlabc -repl` → call the function with
   sample inputs → see numerical results.
3. Steps under `matlabc -dap` debug (set a breakpoint on the
   contribution line, inspect state variables).
4. Emits a Verilog-A module via `matlabc -emit-verilog-a` that
   passes ADMS / OpenVAF lint.

Proposed example set (final state — Tier-1 + Tier-2 + ... +
Tier-9):

| File                                           | Class                            |
|---|---|
| `rf_rational_writeva.m`                        | RF Toolbox VF model → .va        |
| `rc_lowpass.m`                                 | 1st-order RC LP via `ddt`        |
| `rlc_bandpass.m`                               | 2nd-order RLC via `ddt`          |
| `biquad.m`                                     | Biquad via `laplace_nd`          |
| `butter_lp.m`                                  | Butterworth-3 via `tf`+`laplace_nd` |
| `ss_lp2.m`                                     | State-space LP-2                 |
| `ss_observer_biquad.m`                         | Observer-canonical biquad        |
| `sine_source.m`                                | Behavioral sinusoid stimulus     |
| `chirp_source.m`                               | Chirp stimulus                   |
| `comparator.m`                                 | `cross()`-event comparator       |
| `schmitt.m`                                    | Schmitt trigger (dual `cross()`) |
| `vco.m`                                        | VCO via `idtmod`                 |
| `pll_charge_pump.m`                            | Charge-pump PLL (VCO + PD + LF)  |
| `dac_behav.m`                                  | Behavioral DAC                   |
| `dac_inl_dnl.m`                                | DAC with INL/DNL parameters      |
| `adc_behav.m` (Tier-11, AMS)                   | Behavioral ADC (`.vams`, deferred)|
| `resistor.m` / `capacitor.m` / `inductor.m`    | RLC primitives                   |
| `diode.m`                                      | Ideal diode                      |
| `opamp_ideal.m` / `opamp_sat.m`                | Op-amp + saturated op-amp        |
| `rtd_pt100.m`                                  | Pt-100 RTD using `$temperature`  |
| `thermistor.m`                                 | NTC thermistor                   |
| `photodiode.m`                                 | Light-dependent current source   |
| `noise_thermal.m` / `noise_flicker.m`          | Noise models                     |
| `iv_curve_table.m`                             | `interp1`-driven IV curve        |
| `ctle_3pole.m`                                 | Signal-integrity CTLE            |

For each, the repo carries:
- `examples/verilog_a/<name>.m` — the MATLAB source.
- `examples/verilog_a/<name>.stdout` — expected numerical output
  on the LLVM lane (sanity check).
- `examples/verilog_a/<name>.va.expected` — expected emitted
  Verilog-A.  (For `adc_behav` and similar mixed-signal
  examples, `.vams.expected`.)
- Skip-stamps for irrelevant lanes (`*.skip-emit-c`,
  `*.skip-emit-cpp`, `*.skip-emit-python`,
  `*.skip-emit-typescript`) — the analog subset is not the same
  as the C/C++/Python/TS supported subset, so most analog
  examples will skip emit-* lanes the same way RF Toolbox tests
  do today.

## 8. Test strategy

Three CTest lanes mirroring the SV pattern:

| Lane                      | What it does                                              |
|---|---|
| `run-emit-va`             | `matlabc -emit-verilog-a` byte-compare against `.va.expected` |
| `run-emit-va-fail`        | Negative tests: source that violates the analog subset    |
| `run-emit-va-admslint`    | Run `adms` / `openvaf` lint on each shipped `.va` (opt-in via `-DMATLAB_LLVM_WITH_VA_LINT=ON`) |
| `run-emit-va-cosim`       | Tier-10 optional: ngspice / Xyce cosim, compare against the in-tree `lsim` / `freqresp` / `timeresp` reference (opt-in via `-DMATLAB_LLVM_WITH_VA_COSIM=ON`) |
| `run-emit-vams` (Tier-11) | `matlabc -emit-verilog-ams` byte-compare against `.vams.expected` — deferred with Tier-11 |

The byte-compare lane is the load-bearing one — same model as
`test/EmitSV`.  The lint lane is best-effort and opt-in (some
contributors may not have OpenVAF locally).  The cosim lane is
nice-to-have polish.

## 9. Debug / REPL / plotting story

A key property of this design: **Verilog-A targeting does not
take the file off the rest of the pipeline.**

Workflow for a user authoring an RLC bandpass:

1. Write `examples/verilog_a/rlc_bandpass.m`.
2. Sanity-check numerically: `matlabc rlc_bandpass.m`
   (LLVM lane → real numerical output).
3. Set a breakpoint, step through: `matlabc -dap rlc_bandpass.m`.
4. Plot the time-domain response: enable
   `-DMATLAB_LLVM_WITH_PLOT=ON`, call `plot(t, vout)` inside
   the example.
5. Interactive tuning: `matlabc -repl` → call the function with
   different R/L/C values, replot.
6. Once happy with the numerical behavior, emit:
   `matlabc -emit-verilog-a rlc_bandpass.m -o rlc_bandpass.va`.
7. Drop the `.va` into Cadence Spectre / ngspice / Xyce and
   simulate it inside a SPICE netlist.

This is the same loop that has worked well for the SV backend:
the MATLAB stays executable, the emit step is a *projection*
onto a target language, never a replacement for the rest of
the compiler.

## 10. Relationship to the RF Toolbox carve-out

`docs/rf_toolbox_plan.md` lists `writeVerilogA` /
`rfmodel.rational/writeVA` as carved out.  Tier-1 of this plan
**closes that carve-out** with a minimal scope:

- A single runtime entry (`writeVerilogA(mdl, filename)` +
  `RFRational.writeVerilogA(filename)` method).
- Emits a parameterized `.va` consumable by Cadence / Spectre /
  ngspice / Xyce / ADS.
- Cross-checked against `freqresp(mdl, freqs)` via the optional
  cosim lane.

After Tier-1 lands, the RF Toolbox section in
`docs/feature_status.md` and `docs/rf_toolbox_plan.md` flips
that row from 🔴 carved-out to ✅ shipped.

The remaining RF Toolbox carve-outs (circuit envelope, harmonic
balance, RF Budget Analyzer app, Modelithics, IEEE P370, AMP
file format, Simulink RF Blockset) are unaffected by this plan
and stay carved out.

## 11. Relationship to other toolboxes

After Tier-2 + Tier-3 (continuous filters + state-space), the
Verilog-A backend automatically covers the Control System
Toolbox `tf` / `zpk` / `ss` model objects.  This means a user
who designs a controller via `lqr(A, B, Q, R)` followed by
`ss(A - B*K, B, eye(n), 0)` can immediately emit it as a
Verilog-A behavioral block for AMS simulation with a plant
model.  This is a meaningful side-deliverable that doesn't
require a separate Control-Toolbox-specific tier.

Similarly, after Tier-2, any SPT filter designed with
`butter('s')` / `cheby1('s')` / `cheby2('s')` / `bessel` is
directly exportable.

Tier-7 (sensor models) gives a foothold into general physical
modeling — the same MATLAB function that returns numerical
output can also be the canonical Verilog-A description of an
RTD, thermistor, or photodiode.

## 12. Status

**Plan only — nothing shipped yet.**

Tier-1 (RF Toolbox `writeVerilogA`) is the first concrete arc.
~4 sessions to close the RF-Toolbox carve-out.

## 13. Carved out (final)

- Verilog-A 2.4 features beyond what's listed in §5.5
  (`paramset`, `analog function` reuse libraries, complex
  branch contributions involving simultaneous `V` and `I`).
- Verilog-AMS in the initial scope — deferred to Tier-11.  The
  initial backend emits pure `.va` only.  Tier-11 lifts that
  constraint for ADC bit-bus outputs and user-authored mixed-
  signal interconnects (`connectmodule` / `connectrules`
  discipline resolution).
- ADMS-only extensions (Synopsys / Cadence proprietary
  extensions).
- HSPICE-only behavioral extensions.
- SPICE-format export (SPICE behavioral sources `B`/`E`/`G`/`H`)
  — Verilog-A subsumes this; only consider if a contributor
  needs SPICE-only flow.
- Geometry-aware models (substrate parameters, layout-dependent
  effects).  Out of scope.
