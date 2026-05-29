# Embedded Coder / mflowLink (block diagrams) — Tutorial

`matlabc` compiles `.mflow` **signal-flow** diagrams (the
`settings.kind = "signal_flow"` dialect) into standalone, AOT-compiled
code. Two emit shapes exist: **per-subsystem** — emit one named
`signal_subsystem` as a kernel function (Python / C / C++ / TypeScript)
or a SystemVerilog module — and **whole-diagram** — emit the entire
diagram (sources + subsystems + sinks + time loop) as a self-contained
driver, plus a **cocotb Software-in-the-Loop (SIL)** harness that wraps
the diagram around an SV DUT. Output is fully self-contained: no
dispatch table, no embedded `.mflow` JSON, no `MflowLinkSim` runtime
dependency (distinct from the `-emit-mflowlink-cpp` interpreter lane).

## Supported features

Block library (full catalogue in [`../mflowlink_blocks.md`](../mflowlink_blocks.md)):

- **Stateless**: `signal_gain`, `signal_sum`, `signal_product`,
  `signal_saturation`, `signal_abs`, `signal_math_fcn`,
  `signal_trig_fcn`, `signal_relop`, `signal_logical`,
  `signal_compare_to_zero` / `_constant`, `signal_switch`,
  `signal_multiport_switch`, `signal_constant`.
- **Stateful (discrete)**: `signal_unit_delay` (z⁻¹),
  `signal_discrete_integrator`, `signal_discrete_filter`,
  `signal_transport_delay` (→ tapped shift register).
- **Continuous → auto-discretised** (software targets):
  `signal_integrator`, `signal_transfer_fcn`, `signal_zero_pole`,
  `signal_state_space` — Forward Euler (default) or Tustin
  (`--discretize=tustin`); N-th-order, MIMO supported.
- **Sources / sinks** (whole-diagram): `signal_step`, `signal_sine`,
  `signal_ramp`, `signal_pulse`, `signal_chirp`, `signal_clock` /
  `signal_scope`, `signal_display`, `signal_to_workspace`.
- **Structure**: `signal_inport` / `signal_outport`,
  `signal_subsystem` (nested), `signal_matlab_fcn` (user MATLAB body),
  `signal_mux` / `signal_demux`, `signal_goto` / `signal_from`,
  enabled / triggered / if-action subsystems.

Emit targets: `-emit-{c,cpp,python,ts}` (per-subsystem and
whole-diagram), `-emit-sv` (per-subsystem only), `-emit-cocotb`
(whole-diagram SIL harness). Numerical equivalence is verified against
`matlabc -simulate` at the same sample rate by CTest.

## Build & emit

**Per-subsystem** — pass `--subsystem <name>` to emit one subsystem as
a kernel (or a class, for stateful subsystems):

```sh
matlabc -emit-python stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-c      stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-cpp    stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-ts     stateless_mixer.mflow --subsystem stateless_mixer
matlabc -emit-sv     scaled_sum_sv.mflow   --subsystem scaled_sum
```

SV port types default to `Q16.16 signed` (32-bit, 16 fractional);
override per port: `--fi-spec u1=Q24.16 --fi-spec ctrl=UQ8.0`.
Continuous-form blocks are **rejected** in SV mode — replace them with
discrete forms (`signal_discrete_integrator`, `signal_unit_delay`) in
the source `.mflow`.

**Whole-diagram** — omit `--subsystem` to emit the entire diagram
(sources + subsystems + sinks + time loop) as a standalone driver:

```sh
matlabc -emit-python whole_pid_loop.mflow > sim.py
matlabc -emit-c      whole_pid_loop.mflow > sim.c
matlabc -emit-cpp    whole_pid_loop.mflow > sim.cpp
matlabc -emit-ts     whole_pid_loop.mflow > sim.ts
```

Whole-diagram knobs: `--target-rate <Ts>` (base sample rate, overrides
`settings.solver.maxStep`), `--ticks <N>` (explicit tick count),
`--csv <path>` (sink log destination), `--decimation <N>` (log every
N-th tick).

**Cocotb SIL** — wrap the diagram around one (or more) SV DUT
subsystems:

```sh
matlabc -emit-cocotb cocotb_pid_sil.mflow --dut plant_dut
# knobs:
#   -cocotb-out=<dir>        output dir (default <stem>_cocotb)
#   -cocotb-tolerance=<t>    SIL assertion tolerance (default 1/65536, one Q16.16 LSB)
#   -cocotb-latency=<N>      pipeline depth: compare cycle k vs drive k-N
```

The SIL host-side reference **is** the per-subsystem Python emit, so a
SIL pass means the SV DUT matches the host reference within fi
quantisation noise. The harness samples DUT outputs **before** the
rising edge so the FF output reflects pre-edge state, matching MATLAB
unit-delay `y[k] = u[k-1]` semantics.

## Worked examples

(All fixtures under `examples/mflowlink/coder/`.)

### Stateless mixer (`stateless_mixer.mflow`)

The canonical Tier-1 demo: three inputs through `signal_gain` →
`signal_sum` → `signal_saturation` (a 3-input mixer with a clamping
output). A `signal_subsystem` named `stateless_mixer` with explicit
`signal_inport`/`signal_outport` boundary tags. Software targets emit a
pure-arith branch-free saturation; the SV target emits the saturation
rails as an explicit `if/elseif/else` that lowers to a 3-way mux.

### 4-tap FIR filter (`fir_4tap.mflow`)

```
u → [unit_delay d1] → [unit_delay d2] → [unit_delay d3]
u, d1, d2, d3 → gains (0.25 each) → sum(++++) → y
```

The `.mflow` is a `function`-kind flow with `signature.inputs = ["u"]`,
`signature.outputs = ["y"]`, three `signal_unit_delay` nodes
(`initialCondition: 0.0`), four `signal_gain` nodes (`gain: 0.25`), and
a `signal_sum` with `signs: "++++"`, wired by `data` edges. Emits as a
class wrapper (Python/C/C++/TS) and as SV (four unit-delay registers +
combinational tap sum + an `always_ff` block). Validated step response:
y[0..3] = 0.25 / 0.5 / 0.75 / 1.0, then steady-state 1.0 — the textbook
4-tap moving average.

### Discrete PID (`discrete_pid.mflow`)

A full PID controller: discrete integrator + unit delay + sum / gain /
saturation, class-wrapped per target. Stateful subsystems carry state
across `step(...)` calls inside the wrapper instance; the SV target
emits the state as `persistent` variables with `if isempty(...) ||
reset` init that the SV pipeline lowers to clocked registers.

### Continuous plant, auto-discretised (`tf_lowpass.mflow`, `tf_2nd_order.mflow`)

`tf_lowpass` realises 1/(s+1) as a `signal_transfer_fcn`; it discretises
under `--discretize=forward_euler` (default) or `--discretize=tustin`.
`tf_2nd_order` is 1/(s² + 0.4s + 1): Forward Euler uses a
controllable-canonical realisation, Tustin uses Direct Form II
Transposed (peak 1.527, matching the analytic value). `zp_plant.mflow`
(zero-pole-gain) and `ss_plant.mflow` / `mimo_state_space.mflow`
(state-space, including a 2-in/2-out decoupled plant) reuse the same
discretisation paths. These are software-target only — continuous
blocks are rejected by the SV lane.

### Whole-diagram PID loop (`whole_pid_loop.mflow`)

A complete closed loop: `signal_step` source → error `signal_sum`
(`signs: "+-"`) → PID → 1st-order `signal_transfer_fcn` plant →
`signal_scope` sinks, with a `settings.solver` block
(`fixed_step`, `ode4`, `stopTime: 4.0`). Emitting without `--subsystem`
synthesises the whole driver — a time loop over `stopTime/Ts` that
calls sources, subsystems, and sinks in topological order, latches
state across iterations, and dumps a CSV. The standalone
Python/C/C++/TS outputs run to completion and the CSV diffs against
`matlabc -simulate` at the same rate.

### Cocotb SIL bring-up (`cocotb_pid_sil.mflow`)

A step source → `plant_dut` subsystem (a Q16.16 gain) → scope. The
cocotb harness drives the SV-emitted `plant` DUT against the host-side
Python reference (itself the `-emit-python --subsystem` of the same
block) and asserts the SIL match within 1 Q16.16 LSB.
`cocotb_multi_dut_sil.mflow` extends this to multiple `--dut a,b,c`
DUTs instantiated side-by-side in a generated wrapper SV, each
lockstep-compared with per-DUT latency FIFOs;
`cocotb_host_helper_sil.mflow` exercises non-DUT helper subsystems run
host-side.

## Limitations & carve-outs

From [`../embedded_coder_roadmap.md`](../embedded_coder_roadmap.md):

- **No whole-diagram SV emit.** The SV lane stays per-subsystem — a
  whole `.mflow` model is bigger than one SV module should be;
  whole-diagram simulation lives on the host.
- **SV mode rejects continuous blocks** (no implicit
  auto-discretisation); the user substitutes discrete equivalents
  explicitly. `signal_matlab_fcn` bodies must be synthesisable, and
  constants inside them need explicit `fi(...)` wrappers
  (`fi(3, 1, 32, 16) * u1`, not `3 * u1`).
- **Cocotb SIL assumes one clock domain** — the DUT runs at the same
  `Ts` as the host loop; multi-clock SIL is out of scope.
- Discretisation changes the numerical answer versus `-simulate`'s
  continuous integration — cross-checks run at the **same rate**.
- Out of scope entirely: `-emit-mflowlink-cpp` replacement (that
  runtime-dispatch interpreter coexists), AUTOSAR / fixed-step
  real-time wrappers, variable-size signals, `signal_from_workspace`
  (host-side binding), Stateflow/Simscape inside a subsystem, and HIL
  beyond cocotb (board bring-up).

## See also

- Embedded Coder roadmap (tiers, CLI, demos): [`../embedded_coder_roadmap.md`](../embedded_coder_roadmap.md)
- mflowLink roadmap (signal-flow runtime + solver): [`../mflow_link_roadmap.md`](../mflow_link_roadmap.md)
- Per-block parameter catalogue: [`../mflowlink_blocks.md`](../mflowlink_blocks.md)
- `.mflow` schema reference: [`../flowchart_schema.md`](../flowchart_schema.md)
- Examples: `examples/mflowlink/coder/` (and its `README.md`), plus `examples/mflow/` for the control-flow dialect
