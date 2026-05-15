# mflowLink Embedded Coder — Plan

A new code-gen lane that emits a single mflowLink **subsystem** as
standalone, AOT-compiled code in **Python / C / C++ / TypeScript /
SystemVerilog**. Separate from the existing
`-emit-mflowlink-cpp` lane, which bakes a *whole model* into a
standalone simulator built around the in-process `MflowLinkSim`
runtime; this lane emits per-block expressions **inline** — a
`signal_gain(2)` becomes `y = 2 * u;` literally, with no dispatch
table at runtime.

The mental rule:

> `-emit-mflowlink-cpp <model.mflow>` = ship the simulator.
> `-emit-{c,cpp,python,ts,sv} <model.mflow> --subsystem <name>` =
> ship the *kernel* of one subsystem.

Companion to:
- [`mflow_link_roadmap.md`](mflow_link_roadmap.md) — §17.5 #12
  "Real-Time / PIL / Embedded Coder" is the gap this doc closes
  (kernel level only; AUTOSAR / fixed-step RT wrappers are out of
  scope here).
- [`mflowlink_blocks.md`](mflowlink_blocks.md) — per-block
  parameter catalogue; the lowering reads the same `params` keys.
- [`verilog_a_plan.md`](verilog_a_plan.md) — the tiered-plan
  style this doc follows.

---

## 1. Goals

- Emit a single `signal_subsystem` as production code in
  **Python / C / C++ / TypeScript / SystemVerilog**.
- Output is **self-contained**: no link against `MflowLinkSim`, no
  embedded `.mflow` JSON, no dispatch table at runtime.
- Numerical equivalence against `matlabc -simulate` (at the
  same sample rate) verified per target by CTest.

## 2. Non-goals

- **Whole-model emit** — that's the existing `-emit-mflowlink-cpp`.
  This lane operates strictly *per-subsystem*.
- **AUTOSAR / fixed-step real-time wrappers** — Tier 6+ follow-up;
  the kernel is what this lane emits.
- **Variable-size signals** — fixed at codegen time.
- **`signal_from_workspace`** — workspace binding is host-side,
  can't AOT-compile.
- **Stateflow / Simscape inside a subsystem** — separate roadmaps.

## 3. Architecture — Path A: via MATLAB AST

```
.mflow signal_flow
  └─ signal_subsystem "MyCtrl" with inports/outports
     └─ SignalFlowLowering             (existing)
        ↓
        MflowLinkModel (subsystem subgraph)
        ↓
        SubsystemToMatlab              (NEW, lib/Flowchart/SubsystemToMatlab.cpp)
        ↓
        matlab::Function AST (synthesised)
        ↓
        existing -emit-{c,cpp,python,typescript,systemverilog}
```

**Why Path A and not direct per-target templates:**

- The matlab_llvm code-gen pipeline has 25 MLIR passes + 5 mature
  language emitters. Reuse, don't rebuild.
- HDL emit alone has scan-hw-pragmas + apply-port-type-pragmas +
  lower-fixed-point + hardware-legalize + synth-check — re-
  implementing per target = months of work.
- One bug fix in `LowerScalarsToArith` benefits every target.

**Constraint imposed by Path A:** every block kind in the
subsystem must be expressible as MATLAB code. Already true today —
`-emit-matlab` produces MATLAB from a whole model; the new pass
is its subsystem-only, function-shape sibling.

## 4. Subsystem boundary

The emit unit is a `signal_subsystem` with explicit `signal_inport`
+ `signal_outport` declarations. This makes the function signature
deterministic:

```
inports  u1, u2, ..., uN  → step's input args
outports y1, y2, ..., yM  → step's return tuple
internal blocks            → straight-line code in execution order
```

Flows without inport/outport boundaries are **not emittable** via
this lane — they're whole models, route to `-emit-mflowlink-cpp`.

Nested subsystems: flatten or emit as **local helper functions** in
the same target file. The outer subsystem boundary defines the
public API.

## 5. State shape per target

| Target | State carrier | Step signature |
|---|---|---|
| Python | `@dataclass` with mutable fields | `obj.step(u1, u2, ...) -> (y1, y2, ...)` |
| C | struct + free `step` function | `void step(State *s, const double in[], double out[])` |
| C++ | class, state in private members | `Output step(const Input &)` |
| TypeScript | class with `private state` | `step(u: number[]): number[]` |
| SystemVerilog | clocked module, state in registers | `always_ff @(posedge clk) ...` |

Functional alternative `[y, s'] = step(u, s)` available via
`--state-form=function` for testability / determinism workflows.
Default = **class/struct with mutating step** (idiomatic for
every target).

## 6. Per-block MATLAB lowering

The `SubsystemToMatlab` pass walks `MflowLinkModel.ExecOrder` and
emits one MATLAB statement per block. State-bearing blocks also
contribute a field to the synthesised `state` struct (one slot per
block id, accessed as `state.<id>`).

| Block kind | MATLAB lowering |
|---|---|
| `signal_constant` | `c = <value>;` (matrix literal preserved) |
| `signal_gain` | `y = K .* u;` |
| `signal_sum` | `y = u1 + u2;` (signs per `params.signs`) |
| `signal_product` | `y = u1 .* u2;` |
| `signal_abs` | `y = abs(u);` |
| `signal_saturation` | `y = max(L, min(U, u));` |
| `signal_math_fcn` / `signal_trig_fcn` | inline math fn |
| `signal_relop` / `signal_logical` / `signal_compare_to_*` | inline operator |
| `signal_dead_zone` | `y = (u > U) .* (u - U) + (u < L) .* (u - L);` |
| `signal_mux` / `signal_demux` / `signal_reshape` | array construct / slice / reshape |
| `signal_unit_delay` / `signal_zoh` | `y = state.<id>_d; state.<id>_d = u;` |
| `signal_discrete_integrator` | `state.<id>_acc = state.<id>_acc + Ts * u; y = state.<id>_acc;` (method per §17.5 #4) |
| `signal_discrete_filter` | direct-form-II difference equations from cached `num` / `den` |
| `signal_integrator` (continuous) | Tier 4: discretized via Backward Euler / Trapezoidal at `--target-rate` |
| `signal_transfer_fcn` / `signal_state_space` / `signal_zero_pole` | Tier 4: discretized state-space difference eqns |
| `signal_transport_delay` | circular buffer indexed by `Ts` |
| `signal_lookup_1d` / `signal_lookup_2d` | `interp1` / `interp2` |
| `signal_matlab_fcn` | inline `function_body` verbatim |
| `signal_switch` / `signal_multiport_switch` / `signal_merge` | ternary / array index |
| `signal_bus_creator` / `signal_bus_selector` | struct construction / field access |
| `signal_subsystem` (nested) | local MATLAB helper function in same TU |
| `signal_goto` / `signal_from` | contracted away during lowering (existing pass) |
| `signal_to_workspace` / `signal_scope` / `signal_display` / `signal_terminator` | **dropped** — sinks aren't part of the subsystem API |

Discrete `Ts` is taken from the subsystem's declared `sample_time`;
continuous blocks require `--target-rate Ts` on the CLI.

## 7. CLI surface

```
matlabc -emit-python  model.mflow --subsystem MyCtrl
matlabc -emit-cpp     model.mflow --subsystem MyCtrl
matlabc -emit-c       model.mflow --subsystem MyCtrl
matlabc -emit-ts      model.mflow --subsystem MyCtrl
matlabc -emit-sv      model.mflow --subsystem MyCtrl

Optional:
  --target-rate <Ts>             # sample period for discretizing continuous blocks
  --discretize-method <method>   # backward_euler (default) | trapezoidal | zoh
  --state-form <form>            # class (default) | function
  --fi-spec <port=Q.F>           # fixed-point spec per port (SV only; repeatable)
  --output <path>                # write to file instead of stdout
```

Reuses the existing `-emit-*` arg-parser; `--subsystem` is what
switches it from "compile <path>.m" to "compile the named
subsystem inside <path>.mflow."

## 8. Continuous-block discretization

Targets have no `ode45`. Continuous blocks
(`signal_integrator`, `signal_derivative`, `signal_transfer_fcn`,
`signal_state_space`, `signal_zero_pole`, `signal_transport_delay`)
must be discretized before emission.

- Software targets: default **Backward Euler** at the subsystem's
  declared `sample_time`; explicit `--target-rate Ts` overrides;
  hard error if neither is set.
- Opt-in: `--discretize-method={backward_euler,trapezoidal,zoh}`.
- Reuses the `signal_discrete_integrator` method machinery
  (shipped §17.5 #4) — generalise to transfer functions.

## 9. SystemVerilog specifics

HDL emit has tighter constraints than software:

- **Continuous blocks**: **hard-reject** with a sourced
  diagnostic. The user replaces `signal_integrator` etc. with
  `signal_discrete_integrator` / `signal_unit_delay` /
  `signal_discrete_filter` first. (Software targets auto-
  discretize via §8; HDL requires the discretization to be
  explicit and reviewable in the source `.mflow`.)
- **Variable-size signals**: unsupported (synth doesn't allow it).
- **`signal_matlab_fcn` body**: must pass the existing
  `-check-synthesizable` pass (no `while true`, no recursion,
  all arrays static-sized).
- **Fixed-point**: optional `--fi-spec port=Q.F` per port; the
  pass stamps `hdl.ports` attrs that the existing
  `runApplyPortTypePragmas` consumes. Without `--fi-spec`, ports
  emit as `real` (synthesizable but expensive — flagged in the
  Verilator lint lane).
- **Clock domain**: single `clk` + `rst` (sync, active-high) for
  the MVP; multi-clock is a Tier 6 carve-out.

## 10. Tier plan

Each tier ends with a working demo + a CTest lane.

### Tier 1 — Stateless Python emit  *(~3 days)*

- New pass `lib/Flowchart/SubsystemToMatlab.cpp` (~400 LOC).
- CLI plumbing in `tools/matlabc/main.cpp` (~50 LOC).
- Block coverage: Constant, Gain, Sum, Product, Abs, Saturation,
  Math/Trig fns, Relop/Logical/Compare, Mux/Demux/Reshape,
  Switch / Multiport Switch / Merge.
- Demo: `stateless_mixer.mflow` — a `signal_subsystem` with 3
  inports + 2 outports + Gain + Sum + Sat → emits a 6-line Python
  function.
- CTest: `flowchart-emit-subsystem-python` diffs Python output
  against `matlabc -simulate`.

### Tier 2 — C / C++ / TypeScript emit  *(✓ shipped 2026-05-15)*

- Same `SubsystemToMatlab` pass, switch the downstream emitter.
- All four software targets covered: Python, C, C++, TypeScript.
- Class-with-step shape: post-emit per-target wrapper appended by
  matlabc (`emitSubsystemClassWrapper`), placing the subsystem's
  state slots as member fields and the functional `step(...)` as
  the class's update method. Opt-out via `--state-form=function`.
- CTest: lane covers Python (import + tests) and C++ (clang++
  compile + run) for every fixture.

### Tier 3 — Stateful subsystems  *(✓ shipped 2026-05-15)*

- Unit Delay, ZOH, discrete integrator (Forward Euler) now lower
  to a multi-return functional form: signature gains one extra
  arg `s_<id>` per stateful block (current state) and one extra
  return `s_<id>_next` (next state). The emitted body reads the
  current state into the block's output variable, then computes
  the next state from the upstream input (Forward Euler for the
  integrator: `s + Ts · u`; latch for Unit Delay / ZOH).
- Type-anchor: each stateful read emits `<var> = <s_arg> + 0.0;`
  so pure-passthrough subsystems (one Unit Delay, no internal
  arithmetic) don't get collapsed to an empty body by the
  MLIR pipeline's dead-code pass.
- Class wrapper picks up state automatically (it lives as a
  member field; `step(u)` calls the functional form, latches
  the returned next-state into members, returns `y`).
- Demo: `examples/mflowlink/coder/discrete_pid.mflow` — full
  P + I + D + saturation controller with 50 ms-sampled
  Forward-Euler integrator + unit-delay derivative. 10-tick
  smoke test verifies numerical equivalence between the
  functional form and the auto-generated `DiscretePid` Python
  class. Carve-out: discrete filter (general N-order direct-form-II)
  defers to Tier 6 — its per-tap history is a vector slot, not a
  single scalar.

### Tier 4 — Continuous-block discretization  *(✓ partial 2026-05-15)*

- `signal_integrator` (continuous `dx/dt = u`) auto-discretised to
  Forward Euler at the chosen sample rate — same Lowering surface
  as the Tier-3 stateful blocks, just gated on the resolved `Ts`.
  Ts resolution order:
    1. `--target-rate <Ts>` CLI flag
    2. block's `data.sample_time` / `params.Ts`
    3. `settings.solver.maxStep`
    4. sourced error (no implicit default).
- Loop-breaker rule: every stateful block (Tier 3 + 4) drops its
  outgoing edges from the topo sort, and state reads get hoisted
  to the top of the function body — matches the simulator's
  "load Z_ first, then evalAll" tick shape and lets feedback paths
  through integrators / delays resolve cleanly.
- Per-block initial conditions (`params.initialCondition`) flow
  into the class wrapper's default-init so a fresh object matches
  the simulator's t=0 snapshot.
- Demo: `examples/mflowlink/coder/continuous_lowpass.mflow` —
  1/(s+1) plant realised as Integrator + Sum feedback. Class-form
  smoke test drives a unit step for 5 s at Ts=0.05 and checks the
  response stays within 1.5% of the analytic `1 - e^{-t}`.
- Carve-outs (separable follow-up slice):
  - `signal_transfer_fcn` / `signal_state_space` / `signal_zero_pole`
    need bilinear (Tustin) or matrix-exponential discretization
    — substantial linalg, separate slice.
  - `signal_transport_delay` needs a circular-buffer state slot
    (not a single scalar).

### Tier 5 — SystemVerilog emit  *(✓ partial 2026-05-15)*

Shipped:
- `--subsystem` against a `-emit-sv` mode routes through the
  HDL-aware `SubsystemToMatlab` path.
- **Hard-reject continuous blocks** with a sourced diagnostic
  (the user must replace with `signal_discrete_*` in the `.mflow`
  source). Tier-4 auto-discretisation stays software-target-only.
- **Programmatic `hdl.ports` stamping** in `tools/matlabc/main.cpp`
  — builds the `ArrayAttr` of `DictionaryAttr`s that
  `runApplyPortTypePragmas` consumes, no source-text comments
  needed. Default fi format is `Q16.16 signed` (32-bit, 16
  fractional); per-port overrides via repeatable
  `--fi-spec port=Q<W>.<F>` (signed) /
  `--fi-spec port=UQ<W>.<F>` (unsigned).
- **`persistent` state mode**: when `SubsystemEmitOptions.StateAsPersistent`,
  stateful blocks emit MATLAB `persistent <slot>; if isempty(<slot>)
  || reset; <slot> = fi(IC, sgn, W, F); end` — same pattern the
  static SV pipeline already lowers to clocked regs.
- **Numeric-literal fi wrapping**: ASTBuilder's `lit(V)` switches
  to `fi(V, sgn, W, F)` calls in HDL mode so SV-pipeline constant
  folding sees integer types instead of f64.
- **No driver script for SV**: the priming
  `__mflowlink_priming = subsystem(0, 0, ...)` call confused
  `arith.shli` (passes f64 zeros that don't match the fi'd args);
  SV mode skips it (port types come from `hdl.ports`, not from
  call-site inference).
- **`signal_matlab_fcn` support**: inline user MATLAB becomes a
  sibling local function. SV synth-check applies; user must use
  explicit `fi(...)` for constants inside the body.

Demos:
- `scaled_sum_sv.mflow` — stateless mixer (Gain · Sum) →
  synthesisable SV with Q16.16 fi ports.
- `matlab_fcn_sv.mflow` — `signal_matlab_fcn` block containing
  user-written synthesisable MATLAB.
- `tapped_delay.mflow` (Tier-5b) — 3-tap shift register with
  multi-output. Stateful subsystem → SV with `clk` / `rst_n` /
  `reset` + three `logic signed [31:0]` registers + the
  required `always_ff @(posedge clk or negedge rst_n)` block.
- `fir_4tap.mflow` — 4-tap FIR — Python/C/C++/TS emit + class
  wrappers work end-to-end; SV blocked on the fi-math width
  tracker bug (Tier-5c, see carve-outs).

Tier-5b (✓ shipped 2026-05-15):
- `lib/MLIR/Passes/SplitIsEmptyOr.cpp` extended to recognise the
  **post-`LowerScalarsToArith` shape**: `arith.cmpf one, %ie, 0.0`
  feeding `arith.ori` with the reset operand. Pre-existing
  implementation only matched the pre-lowering `matlab.short_or`
  shape, which the SV pipeline had already lowered by the time
  the split-pass ran. Side benefit: the same fix unblocked **20
  pre-existing matlab_llvm SV tests** (`emit-sv-tests` 47/77 →
  67/77; FSM-encoding sweep 6/10 → 10/10 — `axi_handshake.m`,
  `mealy_fsm.m`, `moore_fsm.m`, `cordic_pipe.m`, ...).
- HDL-mode dropped the `+ 0.0` software-target type anchor on
  Unit Delay / ZOH state updates — adding an f64 literal taints
  the persistent slot's fi typing and trips HWLegalize. HDL
  mode passes the input through directly.

Tier-5c (✓ shipped 2026-05-15):
- `lib/MLIR/Passes/LowerScalarsToArith.cpp::BinArithToArith` now
  sign-extends the narrower integer operand when a `matlab.add` /
  `matlab.sub` / `matlab.emul` sees mismatched widths.  Picks
  the wider integer as the result type. The common offender is
  `i32 function arg + i64 persistent-fetch` (the persistent goes
  through fi-multiply → fi-saturate which produces i64); the
  arith op used to bail with `result type none` and HWLegalize
  rejected it. Unblocks FIR / IIR / accumulators / any
  subsystem mixing pragma-typed args with persistent-fetch
  fi-multiplies. Demo `fir_4tap.mflow` emits clean SV with
  4 unit-delay registers + combinational tap sum + `always_ff`
  block; Python step response matches the 4-tap MA analytically.

Tier-5d open carve-outs (not yet shipped):
- `signal_saturation` → SV: bool-by-fi multiplication doesn't
  synthesise; workaround for HDL targets is to replace with a
  `signal_matlab_fcn` containing the if/elseif/else.
- Continuous `signal_transfer_fcn` / `signal_state_space` /
  `signal_zero_pole` discretization (bilinear / matrix-exp).
- Verilator + yosys CTest gated lane.
- `--fi-spec` flag stamps `hdl.ports` attrs.
- Synth-check + Verilator lint in the CTest lane (gated like
  `MATLAB_LLVM_WITH_VA_COSIM` — opt-in CMake flag).
- Demo: `fir_4tap.mflow` — 4-tap FIR filter subsystem → synth-
  clean SV that yosys-synth's without warnings; Verilator
  simulation matches `matlabc -simulate` waveform.

### Tier 6 — Nested + multirate + advanced  *(~1 week)*

- Nested `signal_subsystem` → local MATLAB helper function in
  same TU.
- `signal_matlab_fcn` body re-runs through the JIT-class
  refinement (§17.5 #8) so multi-return + indexing inside the
  body propagate to the emitted code.
- Multirate subsystems: emit per-rate `step` functions plus a
  scheduling preamble.

**Total to "every demo target works": ~4–5 weeks.**

## 11. Demo coverage

Under `examples/embedded_coder/`:

| Fixture | Exercises |
|---|---|
| `stateless_mixer.mflow` | Tier 1–2 — pure scalar fan-in/fan-out, no state |
| `discrete_pid.mflow` | Tier 3 — Unit Delay + discrete integrator state |
| `continuous_lowpass.mflow` | Tier 4 — Backward Euler discretization |
| `fir_4tap.mflow` | Tier 5 — synth-clean SV with Verilator cross-check |
| `nested_controller.mflow` | Tier 6 — outer subsystem calls inner subsystem |

Each fixture doubles as a CTest fixture under
`test/Flowchart/EmitSubsystem/`.

## 12. CTest lanes

- `flowchart-emit-subsystem-python` — Python execution diff vs `-simulate`
- `flowchart-emit-subsystem-c` — clang-compile + run + diff
- `flowchart-emit-subsystem-cpp` — clang++-compile + run + diff
- `flowchart-emit-subsystem-ts` — `tsc` + node run + diff
- `flowchart-emit-subsystem-sv` — Verilator lint + simulation diff
  (gated on `MATLAB_LLVM_EMIT_SUBSYSTEM_SV_COSIM=ON`)

Each lane re-uses the `compare CSV row-by-row` helper from
`test/Flowchart/SimulateRun/run_tests.sh`.

## 13. Risk register

| Risk | Mitigation |
|---|---|
| Discretization changes the numerical answer vs `-simulate` (which uses continuous integration) | Document explicitly; ship Backward Euler default + opt-in Trapezoidal; cross-check at the **same rate** by running `-simulate` with `solver.maxStep` clamped to `--target-rate` |
| `signal_matlab_fcn` bodies that don't synthesize for HDL | Reuse the existing `-check-synthesizable` pass + sourced diagnostic naming the offending construct |
| Subsystem contains an algebraic loop | Reject in the new emit lane — algebraic loops need a runtime solver, no closed-form code possible. Sourced diagnostic naming cycle members |
| Param-catalogue drift IDE ↔ codegen | Same `mflowlink_blocks.md` single-source pattern; CTest diff against IDE's `SignalFlowParamSpec` |
| Emit-* pipelines have MATLAB-program assumptions (workspace, REPL) baked in | The synthesised TU is a pure function — bypasses workspace; matches the static `-emit-*` (non-REPL) mode that already works for function files |
| State explosion in nested subsystems | Flatten or emit as local helpers; cap nesting depth in the recursion check |
| Multi-clock HDL emit | Tier 6 carve-out — MVP single `clk` + `rst` |

## 14. Carve-outs (deliberately not in scope)

- **Whole-model emit** — already covered by
  `-emit-mflowlink-cpp` (interpretive standalone)
- **Variable-size signals** — fixed at codegen time
- **`signal_from_workspace`** — workspace binding is host-side,
  can't AOT-compile
- **AUTOSAR / fixed-step real-time wrappers** — §17.5 #12
  follow-up; this lane just emits the kernel
- **Stateflow / Simscape inside a subsystem** — separate
  roadmaps (§17.5 #10 / #11)
- **Multi-clock HDL** — single `clk` + `rst` only for MVP
- **HDL ROM / RAM block inference** — Tier 6+ extension

## 15. Open questions

- **TypeScript runtime choice**: target `tsc` to Node, Deno, or
  browser-friendly ES modules? Default proposal: ES modules with
  `tsc --module esnext`; runs under Node or browser.
- **C++ output style**: header-only template or `.h` + `.cpp`
  split? Default: header-only for the kernel (one `MyCtrl.hpp`),
  matches embedded conventions.
- **Python typing**: emit `dataclass` for state + type hints, or
  plain class for max compatibility? Default: `dataclass` +
  hints (Python 3.10+ is the floor for matlab_llvm's existing
  emit-python).
- **Fixed-point spec format**: `--fi-spec u1=Q15.16` per port, or
  a single `--fi-default Q15.16` + per-port overrides? Default:
  per-port flag, repeatable.

---

## 16. Reference — relationship to the bigger §17.5 #12

`mflow_link_roadmap.md` §17.5 #12 reads:

> **Real-Time / PIL / Embedded Coder** *(production-grade
> codegen)*. Builds on Tier G but adds timing constraints,
> target-specific optimisations, AUTOSAR, fixed-step real-time
> hooks. Adjacent to but separate from the simulation surface.

This roadmap covers the **kernel** half of #12 — the per-
subsystem AOT-compiled code. The **wrapper** half (AUTOSAR task
binding, fixed-step task scheduling, PIL bring-up scaffolding,
target-specific cycle-accurate optimisation) is a follow-up that
*consumes* this lane's emitted kernels.

Together the two halves close §17.5 #12.
