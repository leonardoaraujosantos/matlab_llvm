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

Tier-5d (✓ saturation shipped 2026-05-15):
- `signal_saturation` → SV: HDL mode emits the if/elseif/else form
  (AST IfStmt node) instead of the pure-arith bool-by-fi form
  (which the SV synthcheck rejects). Software targets keep the
  compact arith form. `lowerBlock`'s return type widened from
  `AssignStmt*` to `Stmt*` so a single block can emit either
  an assign or a multi-statement if-block.
- Bonus side fix: `CmpToArith` now sign-extends the narrower
  operand for mismatched-width integer comparisons — mirrors
  the Tier-5c `BinArithToArith` equaliser. Helps stateful
  subsystems where a persistent fetch (i64-routed via fi-saturate)
  is compared against a rail constant (i32).
- Demo: `stateless_mixer.mflow` emits clean SV with saturation
  rails in the `always_comb` block. Discrete-PID-with-saturation
  still needs Tier-5e (state-update path keeps the persistent
  fetch at f64, requires routing through fptosi — separable
  follow-up).

Tier-5f (✓ shipped 2026-05-15):
- New pass `lib/MLIR/Passes/UnifyMixedWidthStores.cpp` runs
  between `runLowerScalarSlots` and `runHWLegalize` in the SV
  pipeline. Walks every `matlab.alloc` and, when its stores have
  mixed integer widths (e.g. i32 saturation rails + i64
  passthrough from a fi-saturate chain), sign-extends narrower
  stores to the widest store width via `arith.extsi`. Retypes
  the slot + every load + refreshes the enclosing function's
  return signature. The pass runs in tandem with a re-run of
  `runLowerScalarSlots` + `runMem2RegLite` so the now-typed
  alloc gets lowered to `llvm.alloca` and the single-writer
  scalar gets promoted.
- Closes the headline **full discrete PID + saturation → SV**
  case: `examples/mflowlink/coder/discrete_pid.mflow` emits a
  clean module with both state registers (s_iacc, s_prev_err),
  the saturation 3-way mux, the i64 accumulator chain, and the
  always_ff tick block.
- 16/16 emit-subsystem cases green; emit-sv-tests stays at
  67/77 (no regressions).

Tier-5g (✓ partial, shipped 2026-05-15):
- `signal_transfer_fcn` (1st-order strictly-proper only) now
  emits across all targets — software + HDL. Auto-discretised
  via Forward Euler at the subsystem's chosen `Ts` (CLI
  `--target-rate` / block `sample_time` / flow
  `settings.solver.maxStep`). Single state slot per TF
  (`s_<id>`), Forward-Euler difference equation
  `s_next = (1 - Ts*a0/a1) * s + (Ts*b0/a1) * u`.
- `signal_integrator` also drops out of the HDL hard-reject
  list now — same auto-discretization path (Tier 4) carries it
  through. Continuous integrator + sum-feedback subsystems
  emit clean SV with the integrator state register.
- Demo: `examples/mflowlink/coder/tf_lowpass.mflow` — 1/(s+1)
  directly via `signal_transfer_fcn`. Python step response
  matches `1 - e^{-t}` within 1% Forward-Euler accuracy;
  SV emits a single `logic signed [31:0] s_tf` register +
  `always_ff` tick block.
- 18/18 emit-subsystem cases green; 37/37 flowchart/runtime/
  frontend lanes green.

Tier-5h (✓ shipped 2026-05-15):
- **Higher-order TF** (any N-th order, strictly proper) via
  Forward Euler on the controllable canonical state-space
  realisation. Allocates N state slots per block; emits
  state-update difference equations + a y = Σ b_i*x_i output
  combination. Demo: `tf_2nd_order.mflow` — 2nd-order
  underdamped lowpass with peak overshoot 1.535 at t=3.21.
- **`signal_zero_pole`** — `resolveTFCoeffs` expands real-root
  zeros/poles + scalar gain into (num, den) polynomial form
  via `expandPoly`, then routes through the same TF path. Demo:
  `zp_plant.mflow` — 2/((s+1)(s+2)) monotonic step response,
  DC gain 1.0.
- **`signal_transport_delay`** — discretised as a length-N
  shift register (N = round(delay/Ts)). Each tap is one state
  slot; oldest tap feeds the output, newest tap takes the
  input. Demo: `transport_delay.mflow` — 4-tap shift register
  with verified delay-by-4-ticks step response.
- **`signal_state_space`** (SISO, D=0) — parses (A, B, C)
  matrices and discretises via Forward Euler:
  `x[k+1] = (I + Ts*A)*x[k] + Ts*B*u[k]; y = C*x[k]`.
  Demo: `ss_plant.mflow` — same 2nd-order plant as
  `tf_2nd_order.mflow` realised in canonical (A, B, C) form,
  with identical numerical response.
- **Multi-slot state infrastructure**: `StateSlot` extended
  with `LocalVar` (separate from `OutVar`) so a single block
  can carry multiple state registers. The state-read hoist,
  next-state expression collection, class-wrapper metadata,
  and the function-signature builder all generalise to
  "N slots per block".
- **Verilator lint gated CTest lane**: new
  `MATLAB_LLVM_WITH_EMIT_SUBSYSTEM_SV_COSIM` CMake flag
  (off by default) registers
  `flowchart-emit-subsystem-sv-verilator`. Runs
  `verilator --lint-only` over every SV-capable coder fixture;
  catches regressions the structural smoke tests can't. Skips
  cleanly when verilator isn't on PATH. **All 11 SV-capable
  demos pass Verilator lint** with the documented cosmetic
  warnings suppressed.

Tier-5i (✓ shipped 2026-05-15):
- **Bilinear (Tustin) discretisation** as an alternative to
  Forward Euler for `signal_integrator`, `signal_transfer_fcn`,
  `signal_zero_pole`, `signal_state_space` (SISO). New CLI flag
  `--discretize=forward_euler|tustin` (default `forward_euler`
  preserves Tier-5h behaviour). Tustin uses polynomial
  substitution `s = (2/Ts)·(z-1)/(z+1)` and Direct Form II
  Transposed; the SISO state-space case routes through a
  Faddeev-LeVerrier `(A,B,C) → (Num,Den)` conversion. Same N
  state slots as Forward Euler. Adds a direct-feedthrough term
  `n_n*u[k]` in the output equation, so any block kind under
  Tustin needs a SEPARATE state-read local (`x1_<id>`) instead
  of the legacy `LocalVar = OutVar` shape — see
  `needsSeparateLocal` in
  `lib/Flowchart/SubsystemToMatlab.cpp`. SV emit + Verilator
  lint all 11 SV-capable demos remain clean under Tustin.
  Demo: `tf_lowpass.mflow` driven with `--discretize=tustin`
  yields DF2T realisation (`y[k] = NumZ[0]·u[k] + v[k]`,
  `v_next = NumZ[1]·u[k] + 0.9512·y[k]` for `1/(s+1)` at
  Ts=0.05); 2nd-order peak 1.527 matches analytic ζ=0.2 ωₙ=1
  better than Forward Euler's 1.535.
- **MIMO state-space (vector-valued ports)** —
  `signal_state_space` now accepts B with P ≥ 1 columns and C
  with Q ≥ 1 rows. Per-port output variables (`<id>_y1` /
  `<id>_y2` / ...) tracked through a new `VarOfNodePort` map;
  `resolveInputExpr` consults it before falling back to the
  legacy `VarOfNode`. State update sums over all P inputs
  via `B[i,k]*u_k`; output equations emit one statement per
  output port. Tustin remains SISO-only — `--discretize=tustin`
  with a MIMO shape emits a sourced error (matrix bilinear
  needs a state-basis transform that this lowering doesn't
  do yet). Demo: `mimo_state_space.mflow` (2-in/2-out
  decoupled plant, A=diag(-1,-2), B=I, C=I).

Tier-5k (✓ shipped 2026-05-15):
- **SV fi-multiplication normalising shift fixed.** Every fi
  multiplication in `SubsystemToMatlab.cpp` now routes through a
  `fiMul` helper that wraps the result in `fi(prod, S, W, F)`.
  Sema infers the outer expression as Q<W>.<F>; the AST → MIR
  lowering emits a clamp-style `matlab.fi.cast` that
  LowerFixedPoint translates into `>>> Frac`. Combined with an
  outport wrap and an else-branch wrap inside `signal_saturation`,
  Sema now narrows every chain back to the declared port spec
  instead of accumulating widened FL through the expression tree.
  Input args also gain a `<arg> = fi(<arg>, S, W, F)` re-cast at
  the start of the body so Sema's Phase 5.6 Stage A.1
  `ParamFiSpec` mechanism pins their type — without it the SV
  pipeline would emit a malformed `fi(none, ...)` constructor
  cast at the outport that the SV emitter can't lower.
- **SV stateful local self-assignment fixed.** `exprFor` in
  `lib/MLIR/Passes/EmitSystemVerilog.cpp` now unwraps multi-use
  `arith.fptosi` / `arith.fptoui` of a persistent-get result to
  the register signal (previously only the single-use inline
  path did this). The canonical `d1 = d1` self-assignment shape
  in tapped-delay / FIR-style stateful subsystems now correctly
  reads as `d1 = s_d1`.
- **Behavioural cosim coverage**: all 14 SV-capable fixtures
  (was 4 before) pass `flowchart-emit-subsystem-sv-cosim` —
  bit-exact for pure-delay and stateless boolean shapes;
  within Q16.16 quantisation noise (~6e-4) for fi-arith chains.
  Worst-case errors: fir_4tap / unit_delay / transport_delay /
  comparator_logic / threshold_switch / stateless_mixer /
  tapped_delay = 0.000e+0; tf_lowpass / mimo_state_space /
  continuous_lowpass = 2.0e-4; tf_2nd_order / ss_plant = 2.5e-4;
  zp_plant = 6.0e-4; discrete_pid = 1.6e-4.

Tier-5j (✓ partial, shipped 2026-05-15):
- **Verilator behavioural cosim lane** —
  `test/Flowchart/EmitSubsystem/cosim.py` + `run_verilator_cosim.sh`
  compile each SV-capable fixture with `verilator --cc --exe
  --build`, drive deterministic stimulus through both the
  Verilator binary AND the Python emit's class wrapper, and
  compare per-tick outputs with tolerance. Gated under the
  existing `MATLAB_LLVM_WITH_EMIT_SUBSYSTEM_SV_COSIM` CMake flag
  as `flowchart-emit-subsystem-sv-cosim`. Skips cleanly when
  verilator isn't on PATH.
- Curated to **four fixtures whose SV emit is bit-exact today**:
  `unit_delay`, `transport_delay` (pure-delay state machines —
  state read IS the output, no fi-multiplication in the data
  path), `comparator_logic`, `threshold_switch` (pure stateless
  combinational with boolean outputs). The cosim decodes 1-bit
  output ports as plain booleans; wider ports as Q16.16 fi
  values. Handles both sequential (clk + rst_n + optional reset)
  and pure-combinational modules.
- Caught two pre-existing SV emit bugs the lint lane misses
  (carved out as Tier-5j follow-ups below):
    - **fi-multiplication missing the Q<W>.<F> normalising shift**
      — `fi(K, 1, 32, 16) .* x` lowers to `(x << log2(K_raw))`
      without the trailing `>>> 16`, so the wider intermediate's
      high bits truncate when stored into a 32-bit register.
      Affects every fixture with a Gain, Sum-of-products, TF,
      ZP, or state-space block — i.e. most numerically-interesting
      ones. Fix path: route fi-multiplications through an explicit
      `matlab.fi.cast` op so `LowerFixedPoint::rewriteFiCast`
      inserts the right shift. AST-level wrapping in
      `SubsystemToMatlab.cpp::fiMul` is the natural place, but
      the AST → MIR lowering currently emits a malformed cast
      (callee=`matlab_fi_quantize_s` on an `i32 → i32` cast)
      when the input type isn't yet inferred — needs a fi-type-
      aware AST builder or a post-codegen fixup pass.
    - **Stateful blocks' state-read hoist emits `local = local`
      (cosmetic self-assignment) instead of `local = state_reg`**
      in some shapes — visible in `tapped_delay` output of `d1 =
      d1; d2 = d2;` (suppressed today as the cosmetic Verilator
      UNOPTFLAT / ALWCOMBORDER warnings). The behavioural impact
      is that `local` reads uninitialised, so the output uses
      garbage instead of the latched register value. Fix path:
      a slot-promotion ordering or an explicit `matlab.load(s_*)
      → local` rewrite before the multi-output reassignment.

Tier-5j open carve-outs (not yet shipped):
- ~~SV fi-multiplication normalising shift~~ ✓ fixed in Tier-5k.
- ~~SV stateful local self-assignment~~ ✓ fixed in Tier-5k.

Tier-5l (✓ shipped 2026-05-15):
- **MIMO Tustin (matrix bilinear)** — `signal_state_space` under
  `--discretize=tustin` now supports MIMO too (was SISO-only).
  Implementation in `lib/Flowchart/SubsystemToMatlab.cpp`:
  new dense-matrix helpers (`matEye`, `matAdd`, `matSub`,
  `matMulOuter`, `matInverse` with partial pivoting) plus a
  `tustinSS` driver that computes `Ad = M(I+αA), Bd = α(I+Ad)MB,
  Cd = C, Dd = α·C·M·B` (α = Ts/2, M = (I−αA)⁻¹). The state
  transformation `z[k] = x[k] − α·M·B·u[k]` is folded into the
  emitted state update so the discrete equations are the
  standard `z[k+1] = Ad·z[k] + Bd·u[k], y[k] = Cd·z[k] + Dd·u[k]`
  shape. Direct-feedthrough `Dd·u` appears in the output
  equation as expected for Tustin. SISO state-space also routes
  through the same path (matrix Tustin reduces to scalar
  Tustin at N=P=Q=1). Demo: `mimo_tustin.mflow` — same 2-in/2-out
  decoupled plant as `mimo_state_space.mflow`, drive
  `--discretize=tustin` and verify per-port direct feedthrough
  (y1[0] = 0.0244 for u1=1, exactly matches α·M[0,0]·B[0,0])
  + decoupled DC gain (y1→1.0, y2→0.5) + no cross-leakage.
- **yosys generic synthesis lane** —
  `test/Flowchart/EmitSubsystem/run_yosys_synth.sh` runs
  `yosys -p "read_verilog -sv <sv>; synth -top <name>; stat"`
  over every SV-capable fixture, verifies it synthesises
  cleanly, parses the cell count from the stat dump, and
  enforces a per-fixture gate-count floor as a regression
  sentinel. Registered as `flowchart-emit-subsystem-sv-yosys`
  under the same `MATLAB_LLVM_WITH_EMIT_SUBSYSTEM_SV_COSIM`
  CMake flag. Skips cleanly when yosys isn't on PATH. 15/15
  fixtures synthesise (matlab_fcn_sv carved out — pre-existing
  emit issue where the user-fn's return type isn't fi-inferred).
  Per-fixture cell counts: unit_delay 32, threshold_switch 181,
  comparator_logic 47, stateless_mixer 478, scaled_sum 523,
  tapped_delay 96, transport_delay 128, fir_4tap 572,
  continuous_lowpass 1243, tf_lowpass 1355, mimo_state_space
  2722, zp_plant 3057, tf_2nd_order 4096, ss_plant 4109,
  discrete_pid 4727.

Tier-5l open carve-outs (not yet shipped):
- **Algebraic-loop detection for Tustin** — Tustin direct-
  feedthrough blocks placed in a feedback loop (e.g.
  Integrator → Sum → Integrator) produce an algebraic loop the
  loop-breaker can't break. Emitter currently produces an
  uninitialised OutVar read; should detect the cycle and
  surface a sourced error suggesting the user replace the
  manual Integrator+Sum subgraph with a single
  `signal_transfer_fcn`.

### Tier 6 — Nested + multirate + advanced  *(~1 week)*

Tier-6a (✓ shipped 2026-05-15) — nested subsystems for software
targets (Python / C / C++ / TypeScript):

- `signal_subsystem` blocks now resolve their `data.flow_id` to
  the inner flow and emit it as a sibling helper function in the
  same TU. Recursive lowering with a `NestedCtx` that caches each
  inner by `flow_id` so multiple references to the same subsystem
  share one function. Cycle detection in the recursion stack.
- State plumbing: each `signal_subsystem` block's slot count
  inherits from the inner's metadata (one outer slot per inner
  state arg, named `s_<outer_id>_<inner_arg>` to keep multiple
  instantiations unique). The outer's signature grows by the sum
  of inner state slots; the class wrapper's `step(u)` threads
  them through the multi-LHS call.
- Multi-output inner subsystems get per-port output variables
  via `VarOfNodePort` so downstream blocks can wire to `out1` /
  `out2` / ... independently.
- `buildSubsystemTU` flushes `Ctx.Pending` into the TU's
  `Functions` list in emission order (innermost first); Sema
  resolves the cross-function calls by name.
- `describeSubsystem` mirrors the recursion so the class
  wrapper's member fields align with the function signature.
- Demo: `nested_pid_filter.mflow` — outer subsystem wraps a
  `signal_subsystem` referencing an inner `lp_filter` flow
  (1/(s+1)) followed by gain 2. Step response within Forward-
  Euler tolerance of analytic `2·(1−e⁻ᵗ)`.

Tier-6a — HDL update (✓ shipped 2026-05-15):
- **Cross-function fi-type propagation in Sema** —
  `TypeInference::visitCallOrIndex` for `BindingKind::Function`
  used to return `Any` (with a TODO).  It now returns the
  callee's `OutputRefs[0]->InferredType` when the callee has
  been visited earlier in the TU walk. The Embedded Coder lane
  orders TU entries inner-first (helpers before the outer
  subsystem) so this fires naturally for nested subsystems;
  ordinary user `.m` files with forward references still fall
  back to `Any`.
- **HDL nested subsystems work.** The previous HDL-mode
  sourced error gate is removed. `outer_loop` SV emit
  instantiates `lp_filter u_lp_filter_0 (.clk(clk), .rst_n(rst_n),
  .u1(u1), .reset(reset), .y1(u_lp_filter_0_y1));` and uses
  the captured output downstream. Verilator lint + behavioural
  cosim + yosys generic synth all pass on the nested fixture.

Tier-6a — multi-instantiation (✓ already works):
- The matlab_llvm SV emitter renders each function call as a
  separate module instantiation (`lp_filter u_lp_filter_0(...);
  lp_filter u_lp_filter_1(...);`), each carrying its own
  register state automatically. Software mode also works — each
  `signal_subsystem` block gets its own state slots in the
  outer's signature (named `s_<outer_id>_<inner_arg>`), with
  the class wrapper holding all instances as separate member
  fields. Verified with a twin-LP demo: 64 DFFs (= 32 × 2
  separate state spaces) after `yosys synth`.

Tier-6b (✓ shipped 2026-05-15):
- **`signal_matlab_fcn` user-body fi-typing.** When the user-
  function is parsed and added to the TU, its formal args get a
  prepended `<arg> = fi(<arg>, S, W, F)` re-cast in HDL mode.
  Sema's Phase 5.6 Stage A.1 mechanism then pins the args' fi
  spec from the call sites; bare-int arithmetic inside the body
  (e.g. `u1 * 3 + u2 * 5`) inherits the fi type via the
  cross-function return-type propagation shipped in Tier-6a.
- The TU layout now pushes user functions BEFORE the outer Fn
  (alongside nested helpers) so TypeInference visits the user
  body first and the outer's call site can pull a typed return.
- `cosim.py` learned the multi-module shape: `parse_sv_ports`
  takes a `want_module` arg pointing at the outer's subsystem
  name; `verilator --top-module` is passed through so the
  testbench instantiates the right top. `matlab_fcn_sv` cosim
  passes bit-exact.
- Demo: `matlab_fcn_sv.mflow` — outer subsystem wraps a
  `signal_matlab_fcn` block whose body is
  `out = u1 * 3 + u2 * 5`. SV emit: outer module instantiates
  `poly_mac_mac_mac u_...` as a sub-module; the inner emits
  proper Q16.16 multiplications with the `>>> 16` normalising
  shift. yosys synth 686 cells.

Tier-6c (✓ partial, shipped 2026-05-15 — software targets):
- **Multirate subsystems for software targets.** Each stateful
  block can declare a per-block `sample_time` / `sampleTime` /
  `Ts` param. The emitter walks every block, finds the
  smallest positive period as the base rate (falls back to
  `settings.solver.maxStep`), computes per-block epoch =
  `round(period / base)`. Any epoch > 1 makes the subsystem
  multirate.
- Multirate subsystems get a hidden `_tick` state slot
  (initial 0) threaded through the function's args/returns
  alongside the regular `s_<id>` slots. The body increments
  `_tick_next = _tick + 1` at end-of-body so the counter
  advances each call.
- Each slow block's state-update is wrapped in
  `if mod(_tick, epoch) == 0 ... else <hold previous> ...
  end` so non-firing ticks preserve the current state.
  Fast blocks (epoch = 1) emit their state-update
  unconditionally and run every tick.
- Class wrapper picks up `_tick` automatically (it appears
  as a member field in `describeSubsystem`'s `StateArgNames`
  with initial value 0).
- Demo: `multirate_filters.mflow` — two `signal_unit_delay`
  blocks at base 0.01 and 5x-slower 0.05. Step-up test
  verifies the fast block latches immediately while the slow
  block holds the previous value until the next firing tick.

Tier-6c — HDL update (✓ shipped 2026-05-15):
- **HDL multirate.** Software emit uses a global `_tick` counter
  and `mod(_tick, epoch) == 0` to gate each slow block's state
  update. `mod` doesn't synthesise, so HDL emit uses a different
  shape: each slow block (epoch > 1) gets its OWN persistent
  `phase_<block>` counter that wraps at `epoch-1`. The state
  update gates on `phase == 0`; the counter advances via
  `if phase == epoch - 1; phase = 0; else phase = phase + 1`.
  Both branches synth to a clean 2-way mux + adder, and the
  per-block counters compose freely (multiple rate domains
  don't interfere). Demo: `multirate_filters.mflow` SV emit
  passes verilator lint, behavioural cosim (bit-exact vs Python),
  and yosys synth (60+ cells).

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
