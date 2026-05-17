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
