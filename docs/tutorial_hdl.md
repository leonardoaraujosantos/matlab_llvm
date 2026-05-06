# Tutorial: MATLAB → SystemVerilog → cocotb verification

This walks through the full HDL flow end-to-end with one running
example. By the end you'll have written a small DUT in MATLAB,
emitted synthesizable SystemVerilog, and verified it bit-exact
against a Python reference under cocotb — without touching SV by
hand.

If you just want to run the working examples, jump to
[Just-run-everything](#just-run-everything) at the bottom.

---

## What you need

- A built `matlabc` (one of: `cmake --build build` from a clone, or
  the binary from a release).
- `verilator` ≥ 5.0 — `brew install verilator` on macOS, your
  distro's package manager elsewhere.
- `cocotb` ≥ 2.0 — `pip install cocotb`.

The cocotb lane gates on both being installed; without them, the
SV emit still works but you can't run lockstep verification.

---

## Step 1 — Write a tiny DUT

Save this as `mux2.m`:

```matlab
function y = mux2(a, b, sel)
    %#codegen
    % hdl: port(a, fi, signed, 16, 0)
    % hdl: port(b, fi, signed, 16, 0)
    % hdl: port(sel, bool)
    if sel
        y = a;
    else
        y = b;
    end
end
```

Two things to notice:

- The `%#codegen` directive marks this as a code-gen target.
- The `% hdl: port(...)` comments declare the port shapes — `fi,
  signed, 16, 0` is a 16-bit signed integer (`fi` = MATLAB's
  Fixed-Point Designer numeric type), `bool` is a 1-bit signal.
  Without these pragmas, the SV emitter has no way to know the
  function's argument types.

The output `y`'s type is **inferred** from the assignment. For
clean control over the output port's signedness, add an output
pragma too — see [Output port pragmas](#output-port-pragmas)
below.

---

## Step 2 — Emit SystemVerilog

```sh
matlabc -emit-systemverilog mux2.m > mux2.sv
```

The result:

```sv
module mux2 (
    input  logic signed [15:0] a,
    input  logic signed [15:0] b,
    input  logic sel,
    output logic signed [15:0] y
);
    always_comb begin
        y = sel ? a : b;
    end
endmodule
```

(The simple if/else collapses to a ternary at emit time — the
`IfStoreToSelect` pass folds single-store if/else into
`arith.select` before SV emission.)

Two checks before going further:

```sh
# Lint with Verilator. -Wall + the suppress flags catch all the
# real bugs while ignoring style/cosmetic noise.
verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL \
    --top-module mux2 mux2.sv

# (Optional) sanity-check with Yosys generic synth.
yosys -p "read_verilog -sv mux2.sv; synth -top mux2"
```

Both should pass clean for any synthesizable design `matlabc`
emits — the SV golden lane runs both on every fixture in
`test/EmitSV/`.

---

## Step 3 — Verify against the Python reference

This is where cocotb takes over. One command:

```sh
matlabc -emit-cocotb mux2.m
```

Output:

```
matlabc: wrote CocoTB harness to mux2_cocotb (3 inputs, 1 outputs, combinational, 100 random vectors)
```

The harness directory is **self-contained**. Drop in and `make`:

```sh
cd mux2_cocotb && make
```

After ~5 seconds:

```
** TESTS=1 PASS=1 FAIL=0 SKIP=0 **
```

What just happened:

1. `matlabc -emit-cocotb` ran `-emit-systemverilog` to produce
   `mux2.sv` and `-emit-python` to produce `mux2_ref.py` (a
   reference model in Python with the same fi semantics as the
   MATLAB source).
2. It generated `test_mux2.py` — a cocotb testbench that drives
   100 random `(a, b, sel)` triples into both the SV simulation
   (via Verilator) and the Python reference, comparing outputs
   cycle-by-cycle.
3. `make` ran Verilator + cocotb + the harness; both DUT and
   reference returned identical values for every vector.

The output directory layout:

```
mux2_cocotb/
    mux2.sv             # the SV DUT
    mux2_ref.py         # the Python reference
    test_mux2.py        # cocotb testbench (drives + compares)
    cocotb_fi.py        # fi pack / unpack helpers
    matlab_runtime.py   # Python fi runtime (copied from source tree)
    Makefile            # Verilator + cocotb invocation
    coverage.txt        # per-port stats (after running `make`)
    dump.vcd            # waveform (after running `make`)
```

`coverage.txt` after the run summarises per-port min/max/mean +
histograms for narrow ports. `dump.vcd` is the Verilator trace —
open in GTKWave or Surfer to inspect cycle-by-cycle behaviour.

---

## Adding state (sequential designs)

Save this as `counter.m`:

```matlab
function n = counter(en, reset)
    %#codegen
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)
    %
    % 8-bit counter that increments while `en` is high.
    persistent count;
    if isempty(count) || reset
        count = uint8(0);
    end
    if en
        count = count + uint8(1);
    end
    n = count;
end
```

Three new ingredients:

1. **`persistent` declaration** — backs a `count_reg` flop in SV.
   Every persistent variable becomes a register.
2. **`if isempty(count) || reset`** — the canonical reset-init
   pattern. The init value (`uint8(0)`) becomes the SV register's
   reset value.
3. **`% cocotb: stimulus(reset, constant, 0)`** — pin `reset` low
   throughout the random-vector run. Without this, random `reset`
   values would fire often enough that the test sees more reset
   cycles than counts.

Emitting:

```sh
matlabc -emit-cocotb counter.m
cd counter_cocotb && make
```

The SV gains a `clk`/`rst_n` port and an `always_ff` block:

```sv
module counter (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    input  logic reset,
    output logic [7:0] n
);
    logic [7:0] count;
    logic [7:0] count_next;

    always_comb begin
        count_next = count;
        if (reset) count_next = 8'sd0;
        if (en) count_next = count + 8'sd1;
        n = count;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 8'sd0;
        else        count <= count_next;
    end
endmodule
```

The cocotb harness automatically detects the sequential shape and
adds clock generation + reset sequencing. No knobs needed for the
common case.

---

## Pipelined designs (`% cocotb: latency(N)`)

A staged pipeline in MATLAB is just chained persistent writes:

```matlab
function y = pipe2(x, reset)
    %#codegen
    % hdl: port(x, fi, signed, 16, 0)
    % hdl: port(reset, bool)
    % cocotb: stimulus(reset, constant, 0)

    persistent stage1;
    persistent stage2;
    if isempty(stage1) || reset; stage1 = int16(0); end
    if isempty(stage2) || reset; stage2 = int16(0); end

    stage1 = x;
    stage2 = stage1;   % auto-routed through pre-edge snapshot
    y = stage2;        % output read: post-edge value
end
```

No explicit snapshot pattern needed — the Python emitter
inserts the pre-edge captures automatically (see [Chained
persistent writes](#chained-persistent-writes--auto-handled-was-blocking-semantics)
under Common gotchas). The cocotb harness then compares
`ref(x[k])` against `DUT.y` at the same cycle (`L=0`), and
both report `x[k-1]` — bit-exact match.

`matlabc -emit-cocotb pipe2.m` still prints a latency hint based
on the persistent count:

```
hint: 2 `persistent` decls — pipelined; if outputs are registered, add `% cocotb: latency(1)` near the `% hdl: port(...)` lines, or pass `-cocotb-latency=1` on the CLI.
```

(For shift-register-style fi-arrays, the hint reads the array
length: `hint: 4-tap fi-array shift register — pipelined; ...`.)

The hint is **conservative** — under the snapshot semantics in
effect today, most designs work at `L=0` and the hint over-counts.
You can ignore it if your cocotb run passes without an explicit
pragma. Pass `% cocotb: latency(N)` only when:

- The design has additional pipeline registers introduced by the
  SV emit that aren't `persistent` declarations in MATLAB (rare).
- You want the harness to skip the first N cycles of comparison
  as warm-up — useful for designs where the initial output
  values are deliberately undefined.

---

## Building an FSM

State machines are the second common shape (after counters) you'll
hit. The emitter recognises a `switch`-on-persistent pattern and
lowers it to a clean `typedef enum` + `unique case`, which is what
synthesis tools want to see.

Save as `moore_fsm.m`:

```matlab
function [out_signal, state_display] = moore_fsm(input_bit, reset)
    %#codegen
    % hdl: port(input_bit, fi, unsigned, 8, 0)
    % hdl: port(reset, fi, unsigned, 8, 0)

    S0 = uint8(0);
    S1 = uint8(1);
    S2 = uint8(2);

    persistent current_state;
    if isempty(current_state) || reset
        current_state = S0;
    end

    switch current_state
        case S0
            if input_bit == 1; current_state = S1; end
        case S1
            if input_bit == 0; current_state = S2;
            else;              current_state = S0; end
        case S2
            if input_bit == 1; current_state = S1;
            else;              current_state = S0; end
        otherwise
            current_state = S0;
    end

    if current_state == S2
        out_signal = true;
    else
        out_signal = false;
    end
    state_display = current_state;
end
```

After `matlabc -emit-systemverilog moore_fsm.m`, the relevant SV:

```sv
typedef enum logic [1:0] {S0, S1, S2} current_state_t;

current_state_t current_state;
current_state_t current_state_next;

always_comb begin
    current_state_next = current_state;
    if (reset != 8'sd0) begin
        current_state_next = S0;
    end
    unique case (current_state)
        S0: begin
            if (input_bit == 1) current_state_next = S1;
        end
        S1: begin
            if (input_bit == 0) current_state_next = S2;
            else                current_state_next = S0;
        end
        S2: begin
            if (input_bit == 1) current_state_next = S1;
            else                current_state_next = S0;
        end
        default: current_state_next = S0;
    endcase
    out_signal    = (current_state == S2) ? 1'b1 : 1'b0;
    state_display = 8'(current_state);
end

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) current_state <= S0;
    else        current_state <= current_state_next;
end
```

What the emitter did automatically:

- **Discovered the enum.** The constant set `{S0=0, S1=1, S2=2}` is
  small and contiguous, so the persistent's storage type became a
  `typedef enum logic [1:0]` — width sized to fit the largest
  state. If you'd used non-contiguous values (`S0=0, S1=5, S2=9`),
  the emitter falls back to `logic [7:0]` and you lose the
  `unique case` synthesis benefit.
- **Generated `current_state_next`.** All persistent assignments
  inside the body write `current_state_next`; the `always_ff`
  registers it on the clock edge. This is the canonical
  next-state-logic / state-register split.
- **Made `case` `unique`.** Tells synth that the cases are
  mutually exclusive. The `otherwise` branch becomes the
  `default` (required for `unique` synthesis).
- **Output decoder is combinational.** `out_signal` and
  `state_display` are pure functions of `current_state` — no flop,
  no extra latency. That's Moore semantics. (For a Mealy machine,
  the same source pattern works but the output expression involves
  inputs, not just `current_state`.)

Verify under cocotb:

```sh
matlabc -emit-cocotb moore_fsm.m
cd moore_fsm_cocotb && make
```

Two pragmas worth adding for FSMs:

```matlab
% cocotb: cover(state_display, min_bins=3)
% cocotb: cover_pairs(state_display, min_pairs=4)
```

`cover` checks that random stimulus visited at least 3 distinct
values on `state_display` — catches the silent case where 100
random `input_bit` vectors only ever drive S0↔S1 and never reach
S2. `cover_pairs` checks **transitions**: at least 4 distinct
`(prev, curr)` state edges. A 3-state FSM can pass the bins gate
with `{S0, S1, S2}` while never traversing one specific edge
(`S2 → S1`, say) — the pairs gate catches that. For a complete
Moore exhaustive check, use `min_pairs = number of legal
transitions`.

A third gate, `% cocotb: cover_range(<port>)`, asserts every
value in a port's fi range was seen — useful for narrow inputs
(WL ≤ 8) where you want to gate on full-input-space coverage.
See [`emit_cocotb.md`](emit_cocotb.md) for the full syntax.

The Mealy variant lives in `examples/hdl/mealy_fsm.m`; a
computed-state form (state expression on the RHS instead of
`switch`) lives in `examples/hdl/computed_state_fsm.m`.

---

## Output port pragmas

Source pragmas can also declare output port types, not just
inputs:

```matlab
function crc = crc32(data_in, en, reset)
    %#codegen
    % hdl: port(data_in, bool)
    % hdl: port(en, bool)
    % hdl: port(reset, bool)
    % hdl: port(crc, fi, unsigned, 32, 0)   <-- output pragma
    ...
```

Without the output pragma, the SV emitter renders multi-bit
outputs as `signed` by default. With it, `crc` renders as
`output logic [31:0] crc` (unsigned). Important when the source
intent is clearly unsigned (CRC accumulators, hashes, byte
streams) — the cocotb harness's DUT-side decode aligns with the
Python reference's value rather than relying on the modulo-2^WL
fallback in `_eq`.

---

## Common gotchas

### Persistent reads need to be "anchored"

This snippet looks fine but breaks SV emission:

```matlab
persistent crc_reg;
if isempty(crc_reg); crc_reg = uint8(0); end
msb = bitand(bitshift(crc_reg, -7), uint8(1));   % FAILS
```

Inside the body, `crc_reg`'s read goes through a runtime
`matlab_global_get_f64` call (returns f64 from the runtime ABI).
The bitwise lowering needs both operands to be matching integer
types, which fails when one side is f64. Fix: snapshot the
register into a local with a `+ uint8(0)` (or equivalent typed-
zero) anchor:

```matlab
cur = crc_reg + uint8(0);   % anchored as i8
msb = bitand(bitshift(cur, -7), uint8(1));   % works
```

`crc_reg + uint8(0)` is free at synthesis (the synth tool folds
the `+0`), but in MLIR-land it forces the result through a typed
arithmetic op which anchors the value at i8. The `examples/hdl/`
fixtures all use this pattern.

### `if isempty(p) || reset` is the only supported init shape

`isempty(p)` alone works too. But anything more complex
(`isempty(p) && some_cond`, etc.) is rejected — the SV emitter
expects a single guarded init that becomes the register's reset
value.

### `bool` vs `fi(_, _, 1, 0)` for 1-bit ports

Use `% hdl: port(name, bool)` for any 1-bit signal (clock-enable,
ready, valid, etc.). It renders as `logic name` (single-bit,
unsigned). Declaring 1-bit ports as `fi(_, _, 1, 0)` works but
the compiler emits a hint suggesting `bool` instead.

### Chained persistent writes — auto-handled (was: blocking semantics)

**Status: auto-handled.** This section used to describe a real
divergence; now it documents what's behind the curtain so you
recognise the pattern when reading other people's code.

MATLAB uses blocking-assignment semantics:

```matlab
int1 = int1 + x;
int2 = int2 + int1;   % reads int1's just-written value (MATLAB)
int3 = int3 + int2;   % reads int2's just-written value (MATLAB)
```

The equivalent SV is non-blocking: `int1_next = int1 + x;
int2_next = int2 + int1;` — each `int2_next` reads `int1`'s
**pre-edge** register value, **not** `int1_next`. So a literal
MATLAB-to-SV translation produces a one-cycle pipeline shift
per chained stage between the Python reference and the SV DUT.

The Python emitter handles this automatically. Every persistent
read whose value flows to another persistent's write (rather
than to a function output) routes through a snapshot captured
at function entry:

```python
def f(x, reset):
    if isempty or reset:
        f.int1 = 0; f.int2 = 0; f.int3 = 0
    _int1_snap = f.int1
    _int2_snap = f.int2
    _int3_snap = f.int3
    f.int1 = _int1_snap + x
    f.int2 = _int2_snap + _int1_snap   # reads pre-edge int1, matches SV
    f.int3 = _int3_snap + _int2_snap   # reads pre-edge int2, matches SV
    return ...                          # output reads use post-edge directly
```

What's preserved (you don't need to do anything):

- Reads in next-state computations → use the snapshot. Matches
  SV's always_comb register reads.
- Reads in output assignments → use the post-edge actual storage.
  Matches what cocotb sees when sampling the output after the
  clock edge.
- Snapshots are captured **after** the `if isempty(p) || reset`
  arm, so on a reset cycle the snapshot reflects the post-reset
  value (matches SV's async-low reset propagation).

What this means for you:

- The "snapshot" workaround pattern (`int1_s = int1 + fi(0, 1,
  22, 0)`) is no longer required. Old code that uses it still
  works, but new code should use the natural blocking form.
- Old `% cocotb: latency(N)` pragmas added as workarounds for
  this divergence are now over-counts. The snapshot ref aligns
  with the DUT at `L=0` for most designs. Existing fixtures
  with `latency(N)` still pass — cocotb's L-cycle warm-up just
  skips comparison for the first N cycles, then both sides have
  identical state. New designs typically don't need a latency
  pragma.

What still needs care:

- **Mid-run reset cycles** (`reset=1` after the harness deassert
  and outside the multi-persistent-init lowering's range) aren't
  modelled in the Python ref today; the cocotb fixtures all pin
  `reset=0` after init. Don't drive `reset` high in stimulus
  pragmas.
- **Output reads of persistents** that haven't been written this
  call are read at post-edge — same value as pre-edge. No issue.
- **Output-routed reads** see the post-edge value. If you write
  `n = count + 1` (where `count` is persistent), `n` reflects
  `count + 1`, not the snapshot. That's the desired behaviour
  for output ports.

---

## Debugging cocotb mismatches

When cocotb fails, the harness prints a structured error block.
Reading it correctly is faster than diving for the VCD. Anatomy:

```
ERROR    cocotb.regression  #17 y: post=-32768 pre=42 ref=12345 args={'x': 12345}
ERROR    cocotb.regression    decoded: post=-32768 [signed 16b 0x8000] ref=12345 [signed 16b 0x3039]
ERROR    cocotb.regression    hint: saturation suspected: DUT pinned to -32768, ref outside [-32768..32767]
ERROR    cocotb.regression    trace: /path/to/dut_cocotb/dump.vcd (open in GTKWave / Surfer)
```

Field by field:

- **`#17`** — cycle number where the divergence happened. The
  harness keeps going past the first failure; this lets you spot
  whether failures cluster (state bug) or scatter (saturation
  edge cases, sign issue).
- **`y`** — the output port that mismatched.
- **`post=` / `pre=`** — the DUT's value sampled *after* and
  *before* the rising edge. If `pre` matches `ref` but `post`
  doesn't, the DUT is one cycle behind — that's the latency hint.
- **`ref=`** — the Python reference's value for the same input.
- **`args={...}`** — the input vector that produced this cycle.
  Reproducing by hand: feed these into the Python ref directly.
- **`decoded:`** — both values unpacked as fi-typed, with the raw
  bit pattern in hex. Useful when the decimal value lies but the
  bits agree (sign-interpretation case).
- **`hint:`** — the harness recognised the failure shape. Three
  hints exist today; each maps to a specific knob.

### Triage by hint

| Hint | What it means | Fix |
|---|---|---|
| `latency suspected: pre-edge sample matched ref; consider increasing latency by 1` | DUT lags one cycle behind the reference. Common when you add a register and forget to bump `% cocotb: latency(N)`. | Bump `% cocotb: latency(N)` by 1 (or pass `-cocotb-latency=N` on the CLI). The matlabc emit also prints a hint with a precise count when no latency is declared. |
| `sign-interpretation: bits match modulo 2^WL` | DUT and ref are bit-equivalent but the harness reads them with different signedness. The fallback `_eq` already handles this, so seeing this hint at all means a deeper width/sign disagreement. | Add an output port pragma — `% hdl: port(<out>, fi, unsigned, WL, 0)` (or `signed`) — to lock the SV port shape. |
| `saturation suspected: DUT pinned to <hi/lo>, ref outside [lo..hi]` | DUT is saturating; ref isn't. Either the DUT is using saturating arithmetic where the reference is using wrap, or vice-versa. | Check the source: are you mixing `fi(_, _, ..., 'OverflowAction', 'Saturate')` and plain ops? Either widen the result type (more headroom) or unify both sides on the same overflow policy. |

### When there's no hint

Three frequent shapes that don't auto-classify:

1. **First-cycle-only failures.** Cycle `#0` or `#1` mismatches
   alone almost always mean the persistent's reset value disagrees
   between SV and ref. Check the `if isempty(p) || reset` block —
   a `uint8(0)` init renders as `8'd0`, but `int16(0)` renders as
   `16'sd0`. Type-mismatch on the init constant propagates.
2. **Failures cluster on a specific input value.** Often a
   saturation edge case where the policy doesn't quite match. Grep
   the failing `args` value across cycles; if it repeats, it's
   data-dependent.
3. **Every cycle fails.** Almost always a missed pragma — output
   sign, latency, or stimulus. Re-read the SV emit and diff the
   port shapes against what `mux2_ref.py` expects.

### When the hint isn't enough — go to the VCD

Open `dump.vcd` in GTKWave or Surfer. The `trace:` line in the
error block prints the full path. What to look at:

- The mismatched output port at the failing cycle (`#17` in the
  example above).
- Walk *backward* one cycle at a time on the `*_next` combinational
  signals — that's where the wrong value first appears. Once it
  hits a flop, it sticks for a cycle.
- Compare the `current_state`/`*_reg` registers against what you
  expect from the args trail.

### Sweeping seeds

The default seed is `42` (baked in at emit time so golden runs
are byte-stable). The harness honours two env overrides without
re-emitting:

```sh
COCOTB_SEED=7    make           # replay one alternate seed
COCOTB_VECTORS=20 make          # shrink the run
COCOTB_SEED=7 COCOTB_VECTORS=20 make   # both
```

The first log line in the run echoes the values used:

```
INFO test  matlabc harness: seed=7 vectors=20
```

For coverage of the seed space — useful when chasing a flake or
just gaining confidence that the design isn't seed-sensitive —
the generated Makefile ships with a `sweep` target:

```sh
make sweep             # 20 seeds (1..20)
make sweep N=50        # 50 seeds
```

Output is one line per seed (`PASS` / `FAIL`) plus a summary. On
any failure the sweep exits 1 and prints the failing seed list:

```
sweep: 20 seeds on counter
  seed=1    PASS
  seed=2    PASS
  seed=3    FAIL
  ...
sweep: FAIL on seeds: 3 11
replay one with: COCOTB_SEED=<n> make
```

Replay one of those with `COCOTB_SEED=3 make` to get the full
mismatch dump. With a single failing seed pinned, shrink further
with `COCOTB_SEED=3 COCOTB_VECTORS=20 make` until the failing
cycle is the first one — then transcribe `args=...` from the
error block into a hand-written stimulus and step through in
GTKWave.

### Replay-from-trail

Every harness run drops `args_trail.jsonl` next to `coverage.txt`
and `dump.vcd`. One JSON record per cycle:

```
{"cycle": 0, "args": {"a": 9137, "b": -31129, "sel": 0}}
{"cycle": 1, "args": {"a": -18140, "b": 15497, "sel": 1}}
...
```

Two ways this is useful:

```sh
# 1. Pin a known-good or known-bad run as a regression. Save the
# trail, edit the source, replay deterministically — same inputs
# even after a re-emit:
cp args_trail.jsonl saved/repro_42.jsonl
make replay TRAIL=saved/repro_42.jsonl

# 2. Default-trail replay — reproduces whatever last ran:
make replay
```

`make replay` reads `args_trail.jsonl` (override with `TRAIL=…`)
and sets `COCOTB_REPLAY_ARGS=<file>` for the cocotb run. The
harness logs both the seed/vectors it would have used **and**
the trail it's actually driving:

```
INFO test  matlabc harness: seed=42 vectors=100
INFO test  matlabc harness: replaying 100 cycle(s) from .../args_trail.jsonl
```

Replay overrides random / stim / tester values cycle-by-cycle, so
the input sequence is bit-identical to the captured run regardless
of seed. That makes it the right tool for:

- **Permanent regression repros.** Save the failing trail as
  `regress_<bug>.jsonl`, commit it, replay on every CI run.
- **Diff-driven debugging.** Run twice with different source
  variants under the same trail; diff `coverage.txt` or the
  cocotb log to isolate the change in DUT behaviour.
- **Hand-edited minimal repros.** The trail is plain JSONL —
  trim it to the failing cycle plus enough warmup, edit args by
  hand if needed, replay.

The replay path doesn't need the seed sweep first; combine the
two when you have a flake — sweep to find a failing seed, capture
that run's `args_trail.jsonl`, then replay forever.

### Re-emitting from scratch

If anything in the harness looks stale (you edited the source but
the SV didn't change), nuke the directory and re-emit:

```sh
rm -rf moore_fsm_cocotb && matlabc -emit-cocotb moore_fsm.m
```

The harness directory is single-source-of-truth — there are no
side files outside it that need cleaning.

---

## Just-run-everything

The repository ships 39 reference HDL designs under
`examples/hdl/`. Run the cocotb sweep over all 39 modules in
parallel — every one verifies bit-exact:

```sh
ctest --test-dir build -R cocotb-tests        # CI form
# or directly:
bash test/EmitCocoTB/run_tests.sh build/matlabc
```

The runner walks every fixture in
[`test/EmitCocoTB/run_tests.sh`](../test/EmitCocoTB/run_tests.sh)
in parallel (defaults to 8 workers; override with
`COCOTB_PARALLEL=N`). Per-fixture latency lives in source as
`% cocotb: latency(N)`.

Or pick one to inspect by hand:

```sh
just emit-sv examples/hdl/fir_asic_pipelined.m         # write SV to a temp file
just verify-cocotb examples/hdl/fir_asic_pipelined.m   # full lockstep run
```

`just verify-cocotb` writes the harness, runs `make`, and streams
the cocotb output. Failures include the failing cycle, fi-decoded
values, canonical fault hints (saturation suspected, latency
suspected, sign-interpretation), and a path to `dump.vcd` for
waveform inspection.

---

## Where to go from here

| Topic | Doc |
|---|---|
| Full SystemVerilog supported subset (every pragma + every limitation) | [`sv_supported_subset.md`](sv_supported_subset.md) |
| SV backend architecture — passes, lowering pipeline, op handling | [`emit_systemverilog.md`](emit_systemverilog.md) |
| Cocotb design — port-list parser, stimulus pragmas, harness rendering | [`emit_cocotb.md`](emit_cocotb.md) |
| Fixed-point semantics — fi(...), saturation, quantization | [`emit_fixed_point.md`](emit_fixed_point.md) |
| Python emitter — fi runtime, persistent storage | [`emit_python.md`](emit_python.md) |

**Reference designs by category** (all in `examples/hdl/`):

- Combinational: `alu_16bit`, `mux_4to_1_16bit`, `popcount`,
  `priority_encoder`, `leading_zero_detector`, `barrel_shifter`,
  `median3`, `hamming74`, `aes_round`, `cordic_step`
- FSMs: `mealy_fsm`, `moore_fsm`, `computed_state_fsm`, `uart_rx`,
  `spi_master`, `i2c_bit_bang`
- Pipelined DSP: `fir_asic_pipelined`, `cic_decimator`,
  `cordic_pipe`, `sequential_processor`
- Arithmetic engines: `multi_cycle_mul`, `booth_mul`, `crc8`,
  `crc32`, `fnv1a`, `galois_lfsr`
- Memory / dataflow: `regfile`, `regfile_dyn` (runtime indexing),
  `fifo`, `async_fifo`, `axi_handshake`, `mmap_periph`,
  `sync_2ff`, `manchester_enc`
- Hierarchy: `hier_combinational`, `hier_sequential`

Every one compiles, lints clean under Verilator `-Wall`, and
verifies cycle-exact under cocotb. The integrator-chain
divergence that used to keep `cic_decimator` out of the sweep
is now auto-handled by the Python emitter's pre-snapshot
persistent-read pass — see [Chained persistent writes —
auto-handled](#chained-persistent-writes--auto-handled-was-blocking-semantics).
