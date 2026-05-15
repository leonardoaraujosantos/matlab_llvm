# `examples/mflowlink/coder/` — Embedded Coder fixtures

Each `.mflow` here defines a **signal-flow subsystem** (a `Flow` with
`signal_inport` / `signal_outport` boundary tags) that the Embedded
Coder lane (`docs/embedded_coder_roadmap.md`) compiles into
standalone target source.

## Usage

```
matlabc -emit-python  stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-cpp     stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-c       stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-ts      stateless_mixer.mflow --subsystem stateless_mixer
# HDL — landing in Tier 5; rejects continuous blocks today
matlabc -emit-sv      stateless_mixer.mflow --subsystem stateless_mixer
```

Each emit lane produces a self-contained source file that exports a
function (or class, for stateful subsystems) named after the
subsystem.

## Fixtures (Tiers 1–3)

### Stateless (Tier 1)

| File | Block coverage |
|---|---|
| `stateless_mixer.mflow` | Gain · Sum · Saturation — the canonical "3-input mixer with clamping output" demo |
| `comparator_logic.mflow` | Compare-to-zero · Compare-to-constant · Logical AND — pure-boolean output |
| `threshold_switch.mflow` | Switch with threshold — picks between two data inputs based on a control signal |
| `math_fns.mflow` | Abs · Trig fn (sin) — exercises the `matlab_runtime` math entries on the Python side |

### Stateful (Tier 3)

| File | Block coverage |
|---|---|
| `unit_delay.mflow` | One-tick latch — minimal `z⁻¹`; output = previous input |
| `discrete_integrator.mflow` | Forward-Euler accumulator with `Ts = 0.1` and a 0.5× input gain |
| `discrete_pid.mflow` | Full PID controller — discrete integrator + unit delay + sum / gain / saturation; class-wrapped per target |

### Continuous → auto-discretised (Tier 4)

| File | Block coverage |
|---|---|
| `continuous_lowpass.mflow` | 1/(s+1) realised as Integrator + Sum feedback; auto-discretised to Forward Euler at the user-picked sample rate |

### SystemVerilog emit (Tier 5 + 5b)

| File | Block coverage |
|---|---|
| `scaled_sum_sv.mflow` | Stateless mixer (Gain · Sum) → synthesisable SV with Q16.16 fi ports |
| `matlab_fcn_sv.mflow` | `signal_matlab_fcn` block containing user-written synthesisable MATLAB (3·u1 + 5·u2) |
| `tapped_delay.mflow` | 3-tap shift register with multi-output — stateful subsystem → SV with `clk` / `rst_n` / `reset` + three `logic signed [31:0]` registers |
| `fir_4tap.mflow` | 4-tap FIR — Python/C/C++/TS class wrappers work; SV emit hits a pre-existing matlab_llvm fi-math width tracker bug (i32-vs-i64 mismatch between args and persistent reads when the same expression mixes both) — pass-through delay subsystems work, fi-multiply-on-persistent + add does not |

Usage:

```bash
matlabc -emit-sv scaled_sum_sv.mflow --subsystem scaled_sum
```

HDL-mode behaviour:

- **Continuous blocks are rejected** with a sourced error (no
  implicit auto-discretisation): the user replaces
  `signal_integrator` with `signal_discrete_integrator` /
  `signal_unit_delay` etc. in the source `.mflow` explicitly.
- **State emits as `persistent` variables** with `if isempty(...)
  || reset` initialisation that the SV pipeline lowers to
  clocked registers (Tier-5b — currently blocked on an
  isempty-vs-reset type mismatch; stateful subsystems still
  work for software targets).
- **Port types**: default `Q16.16 signed` (32-bit, 16 fractional);
  override per port with `--fi-spec u1=Q24.16` /
  `--fi-spec ctrl=UQ8.0` / `--fi-spec sig=Q32.24`.
- **`signal_matlab_fcn` bodies must be synthesisable** — the
  existing `-check-synthesizable` pass validates. Constants
  inside the body must use explicit `fi(...)` wrappers (the
  user writes `fi(3, 1, 32, 16) * u1`, not `3 * u1`).

Tier-5 carve-outs (separable follow-ups):

- ~~Stateful subsystems → SV~~ ✓ shipped 2026-05-15.
  `lib/MLIR/Passes/SplitIsEmptyOr.cpp` now recognises the
  post-`LowerScalarsToArith` shape (`arith.cmpf one, %ie, 0.0`
  feeding `arith.ori` with the reset operand), splits it the same
  way as the pre-lowering `matlab.short_or` pattern. Stateful
  subsystems with pure-passthrough delays (`tapped_delay.mflow`)
  emit synthesisable SV with proper `always_ff` + register
  declarations. **Side benefit**: the same fix unblocked **20
  pre-existing matlab_llvm SV tests** (`emit-sv-tests` went from
  47/77 → 67/77 passing; FSM-encoding sweep 6/10 → 10/10).
- **fi-math width tracker (Tier-5c, pre-existing matlab_llvm
  pipeline bug)**: expressions like `y = u + k*s` where `u` is a
  function arg (i32) and `s` is a persistent (lowered through a
  saturate path that produces i64) hit a `matlab.add(i32, i64) ->
  none` type mismatch at HWLegalize. Needs deeper SV pipeline
  work — the `runHWBitWidthInfer` pass should equalise widths
  before the legality check.
- `signal_saturation` → SV (bool-by-fi multiplication in the
  pure-arith form doesn't synthesise; workaround: replace with
  a `signal_matlab_fcn` block containing `if`/`elseif`/`else`).
- `signal_transfer_fcn` / `signal_state_space` /
  `signal_zero_pole` (continuous → discrete) — need bilinear or
  matrix-exponential discretization.

`signal_integrator` blocks get auto-discretised at codegen time —
no separate "discretizer" block needed. Sample period resolution:

1. `--target-rate <Ts>` CLI flag (explicit, wins)
2. block's `data.sample_time` / `params.Ts`
3. `settings.solver.maxStep` from the flow
4. **sourced error** if none of the above resolves

```bash
matlabc -emit-cpp continuous_lowpass.mflow \
    --subsystem continuous_lowpass --target-rate 0.05
```

The initial state value (`params.initialCondition`) is baked into
the class wrapper's default-init, so a freshly-constructed object
matches the simulator's t=0 snapshot.

Carve-outs (defer to a follow-up slice): `signal_transfer_fcn` /
`signal_state_space` / `signal_zero_pole` need bilinear /
matrix-exponential discretization; `signal_transport_delay` needs
a circular-buffer state.

Stateful subsystems emit a multi-return functional form
`[y, s_next] = step(u, s)` *and* a Tier-2 class wrapper that
holds the state slots as member fields and exposes a mutating
`step(u) → y`. The class shape is target-specific:

- Python: `class DiscretePid: def step(self, u): ...`
- C++:    `struct DiscretePid { double step(double u); };`
- C:      `typedef struct { ... } DiscretePid; double DiscretePid_step(DiscretePid *, double);`
- TS:     `class DiscretePid { step(u: number): number }`

Pass `--state-form=function` to opt out of the class wrapper and
emit only the bare functional form.

Each fixture is verified by the `flowchart-emit-subsystem-tests`
CTest lane:
- Python emit imported + entry function called with known inputs;
  outputs compared against analytic references.
- C++ emit compiled via `clang++ -O2`; printf'd outputs diffed
  against the same references.
- Stateful fixtures additionally run a class-form smoke test that
  drives the `step(...)` method for 10 ticks and checks the output
  sequence matches the analytic PID response (1.2 + 0.4·iacc with
  monotone-growing iacc).

## Adding new fixtures

1. Drop a `.mflow` here with a `signal_subsystem`-shaped `Flow`
   (one inport/outport per public arg).
2. Add a case to `CASES=( … )` in
   `test/Flowchart/EmitSubsystem/run_tests.sh` with the analytic
   reference values.
3. Run `ctest -R emit-subsystem` to verify.
