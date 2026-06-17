# Stateflow examples

State-chart `.mflow` files mirroring three MathWorks Stateflow tutorials.

## Lanes

| Goal | Command |
|---|---|
| Inspect resolved chart IR | `matlabc -dump-chart <file>.mflow` |
| Lower to readable MATLAB | `matlabc -emit-matlab <file>.mflow` |
| Lower to MIR / LLVM IR / C | `matlabc -emit-{mir,llvm,c} <(matlabc -emit-matlab <file>.mflow)` |
| Run via C++ interpreter | `matlabc -simulate <file>.mflow` |
| Debug + REPL via DAP | `matlabc -simulate --sim-dap <file>.mflow` |
| One-call REPL load | `matlabc -repl` then `>> loadStateChart('<file>.mflow')` (emits + sources in one step; the chart's `<name>_tick(...)` is then callable in the live REPL session) |
| Native execute as C | `matlabc -emit-matlab f.mflow \| matlabc -emit-c /dev/stdin > f.c && cc f.c -lm -o f && ./f` |
| **Synthesizable SystemVerilog** | `matlabc -emit-systemverilog <file>.mflow > module.sv` |

The software lowering uses persistent scalars + integer-indexed
regions (no struct(), no string literals) so matlabc's MATLAB →
LLVM lane lowers it cleanly through every emit-* mode. When the
mode is `-emit-systemverilog` / `-check-synthesizable` /
`-emit-hardware-report` / `-emit-cocotb`, the lowering switches to
HDL form: per-variable `if isempty(X), X = intW(0); end` reset
initialisers, integer-typed locals (cast through `int16(...)`),
no super-step inner loop (each chart_tick call advances at most
one transition per region — model one clock edge as one call).

## Moore vs Mealy

| Example | Style | Outputs depend on |
|---|---|---|
| `traffic_light_moore.mflow` | Moore | active state ONLY (entry actions assign red / yellow / green) |
| `vending_machine_mealy.mflow` | Mealy | active state + incoming event (the `select/dispense = 1` transition's cond action drives the output) |

Both produce verilator-clean SystemVerilog modules with the
standard `(clk, rst_n, inputs..., outputs...)` surface, an
`always_comb` block for the FSM next-state logic, and an
`always_ff @(posedge clk or negedge rst_n)` block for the
state registers.

In the lowering, the three roles map to the canonical RTL forms:

- **Moore outputs** (assigned by entry/during actions) → registered
  flip-flops; the value is state-resident and survives between ticks.
- **Mealy outputs** (assigned by a transition's condition/transition
  action) → combinational signals driven from state + input, reset to
  their default each tick (a one-cycle pulse). No registered entry
  action can clobber them.
- **Inputs** → plain input ports (combinational), never registers.

## pipelined_mac.mflow — a control FSM driving a pipelined datapath

The harder case: production RTL usually couples a small controller
with a *pipelined* datapath. `pipelined_mac.mflow` is a 2-state
controller (`Idle` / `Run`) gating a **3-stage arithmetic pipeline**
that computes `y = sample*3 + 7`, one result per clock, 3-cycle
latency. The pipeline registers (`s1..s3`) and a parallel valid-bit
shift register (`v1..v3`) are chart locals; `Run`'s `during` action
advances them once per clock:

```matlab
result = s3; valid_out = v3;     % drain the tail of the pipe
s3 = s2;     v3 = v2;            % stage 3 <- stage 2
s2 = s1 + 7; v2 = v1;            % stage 2 <- stage 1 (+ bias)
s1 = sample * 3; v1 = 1;         % stage 1 <- new input
```

Because each assignment reads the *previous* (registered) stage value
before overwriting it, the SV lowering emits a textbook pipeline
(`s3 <= s2; s2 <= s1 + 7; s1 <= sample*3; ...`) with `result` /
`valid_out` as registered Moore outputs — no combinational
input→output path, so it is timing-friendly for synthesis. `matlabc`
even strength-reduces `sample*3` to `(sample<<2) - sample`.

```
matlabc -emit-systemverilog examples/stateflow/pipelined_mac.mflow
matlabc -emit-cocotb        examples/stateflow/pipelined_mac.mflow -cocotb-out=build/pm
```

Note the one-clock-per-call SV model: `chart_tick` advances the FSM by
at most one transition per region per call, so a chart authored for
synthesis should keep its datapath in `during` actions (one clock =
one shift), exactly as here.

## get_started_create_chart.mflow

Reference: [Get Started: Create a Stateflow Chart](https://www.mathworks.com/help/stateflow/gs/get-started-create-chart.html)

Two-state battery model: `Charge` (default) ↔ `Discharge`, guarded
on the `isCharging` boolean input. Charge entry sets `sentPower = 0`
and the during action grows `charge` by 4 per super-step. Discharge
entry sets `sentPower = 3.5` and the during action drops `charge` by 3.

## get_started_hierarchy_chart.mflow

Reference: [Get Started: Add Hierarchy](https://www.mathworks.com/help/stateflow/gs/get-started-hierarchy-chart.html)

Extends the basic chart with substates:

- `Charge`: `FastCharge` (default, +4 per tick) → `SlowCharge` (+1)
  when `charge > 80`, then `Full` when `charge == 100`.
- `Discharge`: `Powered` (default, -3 per tick) → `Empty` (sets
  `sentPower = 0`) when `charge <= 3`.

Top-level transitions remain between `Charge` and `Discharge`.

## model_air_temperature_controller.mflow

Reference: [Model an Air Temperature Controller](https://www.mathworks.com/help/stateflow/ug/model-air-temperature-controler.html)

OR-decomposed root with `PowerOff` (default) and `PowerOn`.
`PowerOn` AND-decomposes into three parallel regions:

- `FAN1` (exec 1): `Off` ↔ `On` at the 120 °F threshold.
- `FAN2` (exec 2): `Off` ↔ `On` at the 150 °F threshold.
- `SpeedValue` (exec 3): leaf state whose during action sums the
  `in(FAN1_On) + in(FAN2_On)` predicates into the `airflow` output.

Drive it via DAP:

```
> stateChart/setLocal { name: "power", value: 1 }
> stateChart/setLocal { name: "temp",  value: 130 }
> stateChart/emit     { name: "tick" }
> stateChart/stepSuperStep
< airflow == 1   (FAN1 on)

> stateChart/setLocal { name: "temp",  value: 160 }
> stateChart/emit     { name: "tick" }
> stateChart/stepSuperStep
< airflow == 2   (FAN1 + FAN2 on)
```
