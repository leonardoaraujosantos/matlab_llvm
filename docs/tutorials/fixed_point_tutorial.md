# Fixed-Point Designer (`fi`) — Tutorial

The Fixed-Point Designer lane adds MATLAB's `fi` numeric type: signed/unsigned
integers with an explicit word length and fraction length, arithmetic that does
the math with native integers and explicit shifts, and saturation/clamp
semantics. The same `numerictype` descriptors that drive C/C++/Python/TypeScript/
LLVM emission also feed the **SystemVerilog** synthesis lane and the **cocotb**
co-simulation path — so a fixed-point algorithm written once can be both run as
a bit-exact software reference and emitted as synthesizable RTL.

## Supported features

- **Constructors**: `fi(value, signed, WL, FL)` (e.g. `fi(0.75, 1, 16, 8)` for
  signed Q8.8), `fi(value, T)` with a `numerictype`, `numerictype(signed, WL, FL)`.
- **Array constructors**: `fi(zeros(1,N), ...)`, `fi([...], 1, 16, 0)` coefficient
  tables.
- **Scalar arithmetic**: `+`, `*` (and MAC accumulate), with native-int + shift
  lowering.
- **The clamp idiom**: `lhs(:) = rhs` holds the destination's numerictype (the
  shift-and-saturate write).
- **Vectors / shift registers**: concat `[x, delay(1:end-1)]`, slice
  `delay(1:k)`, element index `delay(i)`.
- **Display**: `disp(fi)` renders the real-world double; the DAP inspector shows
  `Q<int>.<frac>` + integer storage + real-world value.
- **Persistent fi arrays**: `persistent delay_line` survives across calls and
  REPL boundaries (the tapped-delay-line → SV regfile lane).
- **`double(fi)`** cast (used in the SV-emit pipeline).
- **HDL emission**: `-emit-systemverilog`, `-check-synthesizable`, with `% hdl:
  port(...)` and `% cocotb: latency(...)` pragmas.

## Build & run

```bash
build/matlabc -emit-llvm examples/fi_apply_gain.m > /tmp/fi_apply_gain.ll
clang++ -std=c++20 -O2 -Wno-override-module /tmp/fi_apply_gain.ll \
    build/libMatlabRuntime.a -ldl -lpthread -Wl,-dead_strip -o /tmp/fi_apply_gain
/tmp/fi_apply_gain
```

For the HDL track, emit RTL or run the synthesizability gate instead of linking:

```bash
build/matlabc -emit-systemverilog examples/dsp/fixedpoint_fir_hdl.m
build/matlabc -check-synthesizable examples/dsp/fixedpoint_fir_hdl.m
```

## Worked examples

### Applying a constant gain  (`examples/fi_apply_gain.m`)

The minimal Phase-1 surface: an explicit `(signed, WL, FL)` constructor, a
scalar multiply, and the `(:)` clamp idiom that holds the destination's spec.

```matlab
x = fi(0.75, 1, 16, 8);          % stored = 192  (Q8.8)
gain = fi(1.5, 1, 16, 8);        % stored = 384
y = fi(0, 1, 16, 8);
y(:) = x * gain;                 % real-world 1.125
disp(y);
```

`x * gain` multiplies the stored integers and rescales; `y(:) = ...` writes the
product back into `y` clamped to Q8.8. `disp(y)` prints the real-world `1.125`.

### Scalar FIR moving-average filter  (`examples/fi_fir_filter.m`)

Phase-3 surface: an `fi` array zero-init, a shift register built by concat, and a
multiply-accumulate loop with the accumulator held by `(:)`.

```matlab
h = [fi(0.25, 1, 16, 14), fi(0.25, 1, 16, 14), ...
     fi(0.25, 1, 16, 14), fi(0.25, 1, 16, 14)];
delay = fi(zeros(1, 4), 1, 16, 14);

for k = 1:4
    x = fi(1.0, 1, 16, 14);
    delay = [x, delay(1:3)];      % shift the tapped delay line
    acc = fi(0, 1, 16, 14);
    for i = 1:4
        acc(:) = acc + delay(i) * h(i);   % MAC, accumulator clamped to Q1.14
    end
    disp(acc);
end
```

The 4-tap moving average has impulse response 1/4 per tap, so the step response
settles at 1.0 and the partial sums print 0.25, 0.5, 0.75, 1.0 — bit-exact in
fixed point.

### Synthesizable fixed-point FIR (HDL headline)  (`examples/dsp/fixedpoint_fir_hdl.m`)

The form you reach for when the target is silicon: a `persistent` tapped-delay
line plus a constant-coefficient table, written so it lowers to synthesizable
SystemVerilog via the persistent-fi → SV regfile lane.

```matlab
function r = fir_filter_fi(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 12)
    % cocotb: latency(1)
    h = fi([1, 2, 3, 4, 3, 2, 1], 1, 16, 0);
    persistent delay_line;
    if isempty(delay_line)
        delay_line = fi(zeros(1, 7), 1, 16, 12);
    end
    delay_line = [fi(x, 1, 16, 12), delay_line(1:6)];
    p1 = delay_line(1) * h(1);
    % ... p2..p7 ...
    r = p1 + p2 + p3 + p4 + p5 + p6 + p7;
end
```

Run it three ways from the same source:

1. **Float reference** — link the script body (the `T = numerictype(1,16,12); y =
   fir_filter_fi(fi(0.5, T))` driver) and run it on the host.
2. **Hardware emit** — `matlabc -emit-systemverilog ...` produces a clocked SV
   module: the persistent delay line becomes N parallel registers, the
   coefficient table a static SV lookup. The `% hdl: port` directive pins the
   I/O fi types.
3. **Synthesizability gate** — `matlabc -check-synthesizable ...` confirms the
   body is RTL-legal.

The `% cocotb: latency(1)` pragma drives the cocotb SIL that checks the SV
matches the software fi reference cycle-by-cycle.

### DSP-HDL streaming FIR  (`examples/dsp/dsphdl_fir_stream.m`)

`dsphdl.FIRFilter` is the cycle-accurate hardware sibling of `dsp.FIRFilter`. In
MATLAB-side simulation both compute the same result; this example also exercises
the CIC + NCO down-converter chain and CORDIC `atan2`.

```matlab
b   = fir1(15, 0.25);
sim = dsp.FIRFilter('Numerator',  b);
hw  = dsphdl.FIRFilter('Numerator', b);
% ... 8-frame streaming run, state carried per frame via the handle classdef ...
fprintf('sim vs hw maxdiff = %.6f\n', max(abs(y_sim - y_hw)));
fprintf('hw FIR latency    = %.0f clock cycles\n', hw.getLatency());
```

The `sim` vs `hw` maxdiff is 0 (state-carry is bit-exact). The valid/ready
clocked-datapath SV emit for `dsphdl.*` objects is a documented follow-on — for
synthesizable fixed-point today, use the flat function form of
`fixedpoint_fir_hdl.m`.

## Limitations & carve-outs

- **Function-internal fi typing across user calls** is the biggest open UX gap
  (Tier-6); fi types propagate within a function but the cross-call typing is
  still being wired.
- Open Tier-6 follow-ons: 2-D fi matrices, the reductions tail
  (`prod`/`min`/`max`/`cumsum`/`dot`), fi `parfor` reductions, slope/bias
  scaling, complex fi, 3-D fi arrays.
- The `dsphdl.*` valid/ready/reset clocked datapath SV emit + its cocotb SIL is
  a follow-on; use the flat persistent-fi function form for synthesis today.
- Out of scope: the **Fixed-Point Tool** GUI, `coder.config('fixed-point', …)`
  MATLAB Coder integration, automatic **float→fixed conversion** (`fxpopt` /
  histogram-based scaling), HDL Coder app integration, and the Simulink Embedded
  MATLAB Function block.

## See also

- Implementation / lowering rules: [`../emit_fixed_point.md`](../emit_fixed_point.md)
- Roadmap: [`../fixed_point_toolbox_roadmap.md`](../fixed_point_toolbox_roadmap.md)
- Examples: `examples/fi_apply_gain.m`, `examples/fi_fir_filter.m`,
  `examples/dsp/fixedpoint_fir_hdl.m`, `examples/dsp/dsphdl_fir_stream.m`
