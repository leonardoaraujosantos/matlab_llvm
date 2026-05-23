# DSP System-Object → SystemVerilog bridge — design spec

**Status:** part shipped (SV emit works on the flat-fi form today), part open
(the `dsp.FIRFilter(...)` System-Object surface still goes through runtime
calls that the SV-emit lane rejects). This document captures the design of
the open piece — a new MLIR pass `LowerDspSystemObjects` — so a follow-on
PR can pick it up without re-doing the discovery.

This is **Category 1** from the DSP-Toolbox PR carve-down list:
- T6 §6.1–6.5 of [`dsp_toolbox_roadmap.md`](dsp_toolbox_roadmap.md): the
  fixed-point `dsp.FIRFilter` → emit-SV + cocotb SIL bridge.

## What works today (this PR)

A flat fi-typed FIR with a `persistent` shift register lowers cleanly to
synthesizable SystemVerilog through the shipped pipeline:

```matlab
function r = dsp_fixedpoint_fir(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 12)
    % cocotb: latency(1)
    h = fi([1, 2, 3, 4, 3, 2, 1], 1, 16, 0);
    persistent delay_line;
    if isempty(delay_line)
        delay_line = fi(zeros(1, 7), 1, 16, 12);
    end
    delay_line = [fi(x, 1, 16, 12), delay_line(1:6)];
    r = delay_line(1) * h(1) + delay_line(2) * h(2) + ... + delay_line(7) * h(7);
end
```

- `matlabc -emit-systemverilog` produces 112 lines of clean RTL: a clocked
  module with `clk`/`rst_n`/signed input/saturated 38-bit output, a 7-stage
  saturating MAC adder chain, persistent `delay_line[7]` shift register
  with `delay_line_next[7]` non-blocking update, constant-init coefficient
  table.
- Lint-clean under `verilator --lint-only -Wall`.
- Yosys-synthesizable.
- `matlabc -emit-cocotb` auto-generates the Makefile + ref Python + test
  harness; under cocotb 2.0.1 + verilator, 94 / 100 random vectors match
  (the 6 mismatches are saturation-edge divergences between the auto-
  generated Python ref and the SV — a project-wide "Python-emit gap" that
  `docs/emit_cocotb.md` tracks separately, not a DSP-specific issue).
- Gated as `test/EmitSV/dsp_fixedpoint_fir.m` + `.sv.expected`.
- Headline example: `examples/dsp/fixedpoint_fir_hdl.m`.

## What's still open: the SO surface

Users *would like* to write:

```matlab
function r = my_streaming_fir(x)
    %#codegen
    % hdl: port(x, fi, signed, 16, 12)
    persistent firFilt;
    if isempty(firFilt)
        firFilt = dsp.FIRFilter('Numerator', fi([1 2 3 4 3 2 1], 1, 16, 0));
    end
    r = firFilt(fi(x, 1, 16, 12));
end
```

…and have the SV emit produce the same `dsp_fixedpoint_fir` RTL. Today
this fails because the `dsp.FIRFilter` constructor lowers to
`matlab_obj_new` + `matlab_obj_set_*` + `matlab_dsp_iir_step` runtime
calls. `HWLegalize` (`lib/MLIR/Passes/HWLegalize.cpp:111–143`) explicitly
rejects every `matlab_*` call other than the persistent-state ABI
(`matlab_persistent_isempty` / `matlab_global_get_f64` / `_set_f64`).

## Proposed: `LowerDspSystemObjects` MLIR pass

A new pass that runs **before** `HWLegalize`, after
`LowerPersistentFiArrays`, in the `-emit-systemverilog` pipeline only.

### Pipeline insertion point

`tools/matlabc/main.cpp` around line 12468 (SV pipeline):

```cpp
mlirgen::runLowerPersistentFiArrays(M);
mlirgen::runLowerDspSystemObjects(M);    // NEW — Stage G
mlirgen::runRefineSlotTypes(M);
// ...
mlirgen::runHWLegalize(M);                // existing gate
```

### Recognition strategy

Walk every `func.func`. For each surviving `matlab_obj_new` call:

1. **Class ID check.** The call's `class_id` immediate operand must match
   a known DSP class (`dsp_FIRFilter`, `dsp_IIRFilter`, `dsp_BiquadFilter`,
   `dsp_SOSFilter`, `dsp_Delay`). Resolve via the registry the resolver
   already builds (`tools/matlabc/main.cpp` class-name → ID mapping).
2. **Guard check.** The `matlab_obj_new` must be inside an `scf.if` whose
   condition is `matlab_persistent_isempty(idx)` — i.e., the canonical
   "lazy init on first call" pattern. Same shape `LowerPersistentFiArrays`
   already recognizes for persistent arrays.
3. **Property collection.** Inside the same guard region, gather all
   `matlab_obj_set_mat`/`set_f64` calls on the new obj. For `dsp.FIRFilter`
   the keystone property is `Numerator` (matrix-typed); for `dsp.Delay`
   it's `Length` (scalar). Each property value must be a *compile-time
   constant* (an `fi(...)` literal or `zeros(...)` call).
4. **Step-call collection.** Walk the func for downstream uses of the
   `matlab_persistent_get_ptr(idx)` result. Each use must be either:
   - a `matlab_dsp_iir_step(obj, x)` call (the streaming step), or
   - a `matlab_obj_get_mat(obj, "State", ...)` read (introspection — rare
     in synthesizable code; reject and fall through to HWLegalize error).

### Rewrite

For a matched `dsp.FIRFilter` SO with constant `Numerator = [b₀,…,bₙ]`:

1. **Replace the SO persistent** (a single `matlab_obj_*` slot) with **N
   scalar persistents** for the delay line (`delay_line[0]`…`delay_line[N-1]`),
   exactly matching the shape `LowerPersistentFiArrays` already produces
   for a persistent fi-array.
2. **Replace the `if isempty` init** with N parallel
   `matlab_global_set_f64(idx*100 + k, 0.0)` calls (one per tap, zero-init).
3. **Replace each `matlab_dsp_iir_step(obj, x)`** with the flat MAC body:
   - Load each `delay_line[k]` via `matlab_global_get_f64(idx*100 + k)`.
   - Compute `r = Σ delay_line[k] * b[k]` (constants — Stage C
     `LowerStaticFiArrays` already handles literal coefficient tables).
   - Shift the delay line: `delay_line[k] = delay_line[k-1]` for k > 0,
     `delay_line[0] = x`. Emit via N parallel `matlab_global_set_f64`.

After this rewrite, the IR is bit-identical to what `LowerPersistentFiArrays`
produces for the flat fi-array `delay_line = [fi(x,…), delay_line(1:N-1)]`
pattern — so the remainder of the SV pipeline (`HWStateInfer`,
`HWBitWidthInfer`, `EmitSystemVerilog`) handles it without further change.

### Bail-out policy

If *any* check above fails (non-constant Numerator, missing isempty
guard, mixed step/property-read use, non-FIR SO), the pass **leaves the
IR unchanged**. `HWLegalize` then rejects the surviving runtime call
with the standard "no synthesizable form" error — same precedent as
every other Stage F/E/C pass (`LowerPersistentFiArrays:1–50`).

## Extending to other SOs

After `dsp.FIRFilter` is unblocked, the same pass body extends to:

| SO | Numerator class | Step call | Notes |
|---|---|---|---|
| `dsp.IIRFilter` | Numerator + Denominator | `matlab_dsp_iir_step` | TDF-II requires N state words for max(|num|,|den|) |
| `dsp.BiquadFilter` / `dsp.SOSFilter` | SOSMatrix (K×6) | `matlab_dsp_sos_step` | per-section z₁/z₂ state (2K total) |
| `dsp.Delay` | Length scalar | `matlab_dsp_delay_step` | trivial shift register, no MAC |
| `dsp.LMSFilter` | Length + StepSize | `matlab_dsp_lms_step` | adaptive — Weights also persistent; significantly more state |

The fundamental rewrite shape is the same for all: rewrite the obj
persistent + step-call pattern into N scalar persistents + a flat MAC /
shift / update body.

## Cocotb SIL integration

`matlabc -emit-cocotb` already works on the flat-fi form today
(see `examples/dsp/fixedpoint_fir_hdl_cocotb/` generated by
`matlabc -emit-cocotb examples/dsp/fixedpoint_fir_hdl.m`). The 6/100
random-vector mismatch under verilator is a documented saturation-edge
divergence between the auto-generated Python reference and the SV emit
output — a project-wide issue tracked in `docs/emit_cocotb.md` ("Python-
emit gaps") and orthogonal to this bridge.

Once the SO→SV bridge ships, `matlabc -emit-cocotb` on the
`dsp.FIRFilter`-based source will work transparently — the SO is rewritten
to flat-fi before the cocotb-emit pass sees the IR.

## Effort estimate

Per the DSP roadmap §6 budget: **~6 weeks** for the full SO→SV bridge
across all the filter SOs (FIR/IIR/Biquad/SOS/Delay). The first SO
(`dsp.FIRFilter` only) is **~2 weeks**: the pattern matcher (~400 LOC),
the rewrite builder (~200 LOC), the SV emit test (10 minutes), and the
cocotb regression-gate path. Subsequent SOs share the bulk of the
matcher and only add the per-class property/step recognition (~50 LOC
each).

## References

- `lib/MLIR/Passes/LowerPersistentFiArrays.cpp` — the closest precedent
  (rewrites persistent fi-arrays into N scalar persistents). The new
  pass uses the same shape.
- `lib/MLIR/Passes/HWLegalize.cpp:111–143` — the gate that currently
  rejects the SO runtime calls.
- `tools/matlabc/main.cpp:12436–12605` — the SV pipeline; insertion
  point for the new pass.
- `test/EmitSV/dsp_fixedpoint_fir.m` — the target SV shape (what the
  rewrite must produce).
- `examples/dsp/fixedpoint_fir_hdl.m` — the user-facing flat-fi example
  that emits SV today.
- `docs/emit_cocotb.md` — cocotb harness details + Python-emit-gap list.
