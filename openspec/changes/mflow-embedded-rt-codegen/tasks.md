# Tasks — mflowLink embedded / real-time codegen

Land incrementally; the existing flat emit (Tiers 1–7) is the kernel each slice wraps.

## 1. ERT entry-point contract (single-rate C)
- [ ] Add `--rt` / `--ert` to the whole-diagram C emit path (`-emit-c model.mflow --rt`).
- [ ] Generate `model.h`: `RT_MODEL` with `Model_DW` (state), `Model_U` (root inports/sources),
      `Model_Y` (root outports/logged sinks); fixed sizes at emit time.
- [ ] Emit `model_initialize` (zero + seed initial conditions), `model_step` (one step, reads
      `m->u`, writes `m->y`), `model_terminate` (hook).
- [ ] Keep the existing `simulate()` driver as the non-`--rt` default and the test oracle.
- [ ] `-emit-cpp --rt` variant (class wrapping the same struct/methods).

## 2. EmitRt parity test (the correctness contract)
- [ ] New `EmitRt` lane: emit the ERT trio for a stateful single-rate model, compile against a
      tiny generated harness that replays the model's source inputs, step N times, capture
      `Model_Y`, diff against `matlabc -simulate`.
- [ ] Byte-parity required (same contract as `emit-mflowlink-cpp`).

## 3. Fixed-step multirate scheduling
- [ ] Reuse Tier-6c sample-time inference; for multirate emit `model_step(m, tid)` + rate IDs.
- [ ] Generate an `rt_OneStep()` template with rate counters (base rate + sub-rates).
- [ ] Single-rate models collapse to `model_step(m)`; golden the rate-counter firing sequence.
- [ ] Extend `EmitRt` to a multirate stateful model.

## 4. Static / MISRA-leaning C profile
- [ ] `--rt-profile=misra`: no heap in the step path, all state in `RT_MODEL`, no VLAs/recursion,
      explicit fixed-point narrowing casts.
- [ ] Emit `/* MISRA deviation: <rule> — <reason> */` markers where unavoidable.
- [ ] Smoke test: grep the step path for `malloc(`/`calloc(`/`realloc(` → must be empty.

## 5. Whole-diagram SystemVerilog
- [ ] Compose per-subsystem SV modules into one top module; wire inter-block signals; single
      `clk`/`rst_n` + root IO ports.
- [ ] Reject continuous-time blocks with a sourced error (discretise first), as per-subsystem does.
- [ ] Verilator lint-clean; cosim against `-simulate` on a discrete model.

## 6. Target packaging
- [ ] `--emit-package <dir>`: generated sources + `model.h` + `build.mk` (compile/link recipe).
- [ ] Verify the package cross-compiles standalone (no runtime libs for the flat C path).

## 7. Buffer-in-place optimisation (best-effort, last)
- [ ] Single-use alias pass over the lowered AST: reuse a producer temporary when its signal is
      consumed exactly once and is not stateful/multi-fanout.
- [ ] Gate so numeric output is unchanged; assert `EmitRt` parity still byte-identical.

## 8. Docs
- [ ] `docs/embedded_coder_roadmap.md` — add the §17.5 #12 real-time / ERT tier (entry points,
      multirate scheduler, MISRA profile, whole-diagram SV, packaging).
- [ ] `docs/mflow_link_roadmap.md` — codegen section: the two lanes + the RT layer.
- [ ] `docs/mflowlink_blocks.md` — new "Real-time / embedded deployment" section with the
      `model_step` contract and the `--rt` / `--emit-package` flags.
