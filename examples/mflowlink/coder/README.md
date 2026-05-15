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

## Fixtures (Tier-1 — stateless)

| File | Block coverage |
|---|---|
| `stateless_mixer.mflow` | Gain · Sum · Saturation — the canonical "3-input mixer with clamping output" demo |
| `comparator_logic.mflow` | Compare-to-zero · Compare-to-constant · Logical AND — pure-boolean output |
| `threshold_switch.mflow` | Switch with threshold — picks between two data inputs based on a control signal |
| `math_fns.mflow` | Abs · Trig fn (sin) — exercises the `matlab_runtime` math entries on the Python side |

Each fixture is verified by the `flowchart-emit-subsystem-tests`
CTest lane:
- Python emit imported + entry function called with known inputs;
  outputs compared against analytic references.
- C++ emit compiled via `clang++ -O2`; printf'd outputs diffed
  against the same references.

## Adding new fixtures

1. Drop a `.mflow` here with a `signal_subsystem`-shaped `Flow`
   (one inport/outport per public arg).
2. Add a case to `CASES=( … )` in
   `test/Flowchart/EmitSubsystem/run_tests.sh` with the analytic
   reference values.
3. Run `ctest -R emit-subsystem` to verify.
