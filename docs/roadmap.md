# Roadmap

Forward-looking work tracker for `matlab_llvm`. Organized by effort
horizon and dependency chain, not strict priority — what gets done next
depends on which items unblock real users.

For shipped work, see [`feature_status.md`](feature_status.md). For
detailed history of each backend, see the per-backend `emit_*.md`
docs.

---

## Conventions

- **Effort** is calendar time at one focused implementation session
  per stage (the existing Phase 5.6.x cadence). A "week" means
  ~5 sessions, not 40 hours.
- **Status** legend:
  - 🔵 not started
  - 🟡 in progress / partial
  - 🟢 done (kept here for context until rolled into `feature_status`)
- **Scope dependency** notes flag items that must land first.

---

## Near-term (~1 month)

### 1. HDL Verification with CocoTB 🔵

Wire generated SystemVerilog modules to a Python testbench harness
using [CocoTB](https://www.cocotb.org/). Each `examples/hdl/*.m`
module gets a paired `<name>_tb.py` that drives clk/rst, walks
through a handful of input vectors, and asserts the output matches
the MATLAB reference (run via the existing C/C++/Python emission of
the same source).

**Why it matters.** Today the SV pipeline is verified by
Verilator's lint pass (37 fixtures lint-clean). Lint catches
syntax / signedness / width issues but doesn't prove the RTL
*behaves* the same as the MATLAB source. CocoTB closes that gap:
the same MATLAB program is the golden reference and the
implementation under test.

**Scope.**
- New `test/EmitSVCocoTB/` directory mirroring `test/EmitSV/`.
- Helper script `just verify-cocotb <name>.m` that:
  1. Emits SV via `-emit-systemverilog`.
  2. Emits Python via `-emit-python` (the reference model).
  3. Runs CocoTB with a small driver that feeds the SV
     simulation and the Python model the same vectors per cycle.
  4. Diffs outputs cycle-by-cycle.
- CI lane `cocotb-tests` (gated on CocoTB + Verilator + Icarus
  presence; skip-if-missing rather than required).
- Per-module `*_tb.py` for the 8 `examples/hdl/` modules
  (`alu_16bit`, `mux_4to_1_16bit`, `counter_0_to_10`,
  `mealy_fsm`, `moore_fsm`, `vector_processor`,
  `sequential_processor`, `fir_asic_pipelined`).

**Out of scope (for v1).**
- UVM-style coverage / functional checking.
- Multi-clock testbenches.
- Proving timing closure.

**Dependencies.** None — both ends of the bridge already work.

**Effort.** ~1 week (harness + 8 testbenches + CI lane).

---

### 2. Tier 2: persistent fi-arrays in software backends 🔵

The remaining 3 of 8 `examples/hdl/` modules
(`fir_asic_pipelined`, `sequential_processor`, `vector_processor`)
use **persistent fi-arrays** —
`persistent buf; buf = fi(zeros(1, N), ...)` — which the SV path
lowers to N parallel registers via Stage F, but the C/C++/Python/TS
backends don't model.

**Why it matters.** Streaming / windowed signal-processing in pure
software is a real use case (FIR filters in C, sliding windows,
buffered DSP), not just an HDL idiom. Tier 2 unblocks all 8
HDL examples for software emission and makes the existing fi-array
support useful end-to-end.

**Scope.**
- C: `static T name[N] = {<init>};` at function entry; reads /
  writes through `name[k]`.
- C++: same.
- Python: `<fn>.<name> = [<init>] * N` at module scope; reads /
  writes through `<fn>.<name>[k]`.
- TS: `let <fn>_<name>: number[] = [<init>] * N;`.
- Recognize the `matlab_persistent_get_ptr → subscript1_s` and
  `matlab_persistent_set_ptr → array-of-stores` chains; suppress
  the runtime-call form and emit array indexing.

**Dependencies.** Tier 1 (shipped) recognizes the canonical
isempty pattern; Tier 2 extends the same recognition to the
array-typed persistent ABI.

**Effort.** ~3 days per backend × 4 backends = ~2 weeks.

---

### 3. SV codegen polish 🔵

The 8 HDL examples lint clean, but a few cosmetic / quality
issues remain that don't block synthesis but read awkwardly:

- **Storage-class literals on register width casts.**
  `count_reg <= 4'(8'sd0)` could just be `count_reg <= 4'sd0`.
  The wrap-cast is redundant when the source is a constant.
- **Saturate constant rendering.** Things like `64'sd68719476735`
  (= 2³⁶−1) read more naturally as `36'sh7FFF_FFFFF`. Cosmetic
  but DSP code is full of these.
- **`v0_1`, `v1_1`, ... synthetic intermediate names.** The
  saturate-clamp temps in `vector_processor` / `sequential_processor`
  / `fir_asic_pipelined` use compiler-generated names. Could
  derive semantic names from the surrounding context (e.g.
  `acc_clamped_1`, `prod_extended`) — much more readable RTL.
- **Comment-block placement on persistent declarations.** The
  source `% Estágio 0: Entradas` next to a `persistent delay_line`
  has no SV-side anchor right now (the declarations live in the
  prelude, not always_comb). Should attach to the prelude
  declaration block.

**Scope.** Each is independent; can be ordered by user impact or
done together.

**Effort.** ~2 days total.

---

### 4. SV codegen: pragma path for `-emit-c` / `-emit-cpp` 🔵

Today `% hdl: port(name, fi, signed, W, F)` pragmas are SV-only —
applying them to the C/C++/Python/TS pipelines would let function-
only `.m` files (no typed driver) compile to software too.

**Why it matters.** Asked-for already during the C/C++ audit
(`alu_16bit.m` standalone fails with `unsupported op: matlab.alloc`
because no driver pins types). Reusing the pragma machinery is
the smallest fix.

**Scope.** Lift the `IsSVPath` gate on `runApplyPortTypePragmas`
in `tools/matlabc/main.cpp`; verify nothing else gates on the
SV-only assumption.

**Effort.** ~30 min + regression check.

---

### 5. Runtime: arena allocator + leak audit 🟡

The C runtime currently uses `malloc`/`free` per matrix +
ref-counting on some paths. Two pain points:

- **Allocator pressure** in tight loops (e.g. `for i = 1:1000;
  A = A + B; end` allocates a fresh result matrix each iter).
- **No leak tracking surface.** Programs that genuinely leak
  (held refs in REPL workspace) are invisible until ASAN.

**Scope.**
- Per-call arena reset for the AOT-compiled paths.
- `MATLAB_RT_TRACE=1` env-var prints `alloc / free / leak`
  summary at exit.
- Optional: bump-allocator with explicit reset in JIT-mode for
  long REPL sessions.

**Effort.** ~1 week.

---

## Mid-term (~1–3 months)

### 6. Block language (visual nodes → AST → MLIR) 🟢

**v1 shipped.** The MatForge IDE now saves `.mflow` JSON files
that `matlabc` and `matlab-lsp` both consume. The implementation
chose graph → AST (rather than the originally planned graph →
MLIR direct), which got every existing backend — LLVM / C / C++ /
Python / TS / SV / fixed-point / hardware-report — for free, plus
a free `-emit-matlab` round-trip via the existing `formatAST`.

Five phases shipped:
- **1.** JSON loader + schema validation, byte-precise diagnostics
  (`-dump-flow`).
- **2.** Linear chain → AST: `variable`, `expression`, `display`,
  `input`, `assignment`, `constant`, `function_call`,
  `matrix_literal`.
- **3.** Structured control flow: `if` / `for` / `while` /
  `break` / `continue` / `return`, arbitrary nesting; refuses
  irreducible CFGs.
- **4.** Sub-flows lifted to top-level `Function`s;
  `function_definition` and `subflow_call` blocks.
- **4b.** `custom` blocks with three provenance modes: inline
  `source` / sibling `path` / `library_id` (resolved via
  `--block-path` + `MATFORGE_BLOCK_PATH`); function-insertion
  dedup; arity validation.
- **5.** Cross-backend round-trip lane (`.mflow` ≡
  round-tripped `.m` across C / C++ / Python / TS); `matlab-lsp`
  accepts `.mflow` URIs.

8 examples under `examples/mflow/` and 4 ctest lanes.
See [`flowchart_frontend.md`](flowchart_frontend.md) and the
shipped row in [`feature_status.md`](feature_status.md).

**Open follow-ups (v2 territory, not blocking).**
- Richer block library: `Delay (z⁻¹)`, `FIR`, `IIR (DF-II)`,
  `FSM (state diagram)`, `Counter`, `Accumulator` as primitive
  block kinds rather than custom blocks. Each becomes a small
  Phase-2/3-style render rule.
- Round-trip text ↔ blocks editing (currently one-way).
- 2-D / image-pipeline blocks — overlaps with item #7.

---

### 7. Improve HDL codegen: 2-D fi matrices + RAM inference 🔵

The biggest remaining SV scope gap. Today the pipeline supports
1-D fi arrays (shipped via Stage E + Stage F); 2-D matrices are
needed for image-processing pipelines and matrix-multiply HDL.

**Scope.**
- 2-D fi storage: `logic signed [W-1:0] mem [R][C]` declaration
  + 2-D subscript reads / writes.
- RAM inference for large 2-D persistents:
  `persistent buf; buf = fi(zeros(1, 1024), ...)` should infer
  a synth-tool-recognized SRAM block (`always_ff @(posedge clk)
  if (we) mem[addr] <= din;`) instead of 1024 parallel registers.
- Shape recognition: differentiate "small N for shift register
  → N parallel regs (Stage F today)" from "large N for
  data buffer → RAM block".

**Effort.** ~2 weeks.

---

### 8. SystemVerilog → MATLAB (reverse direction) 🔵

Take legacy synthesizable SV (or simple sequential RTL with
clocked persistent state) and lift it into MATLAB source for
verification, simulation, or porting.

**Why it matters.**
- HDL teams often have SV reference implementations and want
  to iterate on the algorithm in MATLAB (faster, with NumPy
  / matplotlib).
- Lets a designer take an existing IP block, lift it to MATLAB,
  modify it, and re-emit to SV via the existing forward path —
  closing the loop.

**Scope (v1).**
- Lex + parse a synthesizable SV subset:
  - `always_ff` (single clock + sync/async reset).
  - `always_comb` (combinational logic).
  - `unique case` and `if/else` chains.
  - `logic [N-1:0]` and `logic signed [N-1:0]` register declarations.
  - One-hot and binary-encoded `typedef enum` FSMs.
- Lift to a typed MATLAB AST:
  - SV register → `persistent` MATLAB var with
    `if isempty(_); _ = init; end` reset pattern (the same
    idiom Tier 1 recognizes).
  - `unique case (state)` → `switch state`.
  - Sized integer literals → `fi(_, signed, W, F)`.
- Output: pretty-printed MATLAB source via the existing
  formatter.

**Out of scope (for v1).**
- Verilog (only SystemVerilog).
- Multi-clock / CDC handling.
- Generate blocks / parameterized modules.
- Behavioral SV beyond the synthesizable subset.

**Dependencies.** None new — uses the existing AST + formatter.

**Effort.** ~3 weeks.

---

### 9. REPL: line editing + history + JIT cache 🟡

Today `matlabc -repl` is a minimal stdin loop. The major missing
ergonomics:

- **Readline** for history navigation (↑/↓), Ctrl-R search,
  Ctrl-A / Ctrl-E line motion.
- **Multi-line input** for `for ... end`, `function ... end`,
  `if ... end` blocks (today everything must be on one line).
- **Persistent JIT cache** keyed by hashed source so repeated
  function definitions don't re-JIT cold.
- **Tab completion** for variables in workspace + builtins.

**Effort.** ~1.5 weeks (most of it is editline / linenoise
integration; the rest is JIT cache wiring).

---

### 10. Improve HDL codegen: pipelining + retiming 🔵

Beyond the v1 stage-F register split, the pipeline doesn't
automatically rebalance critical paths. For DSP designs that need
to hit a target frequency, this matters.

**Scope.**
- `% hdl: target_freq(N_MHZ)` pragma.
- Compute critical-path estimate per always_comb block (op count
  × per-op latency table).
- Insert pipeline registers when the path exceeds budget.
- Already-shipped scaffolding: `-sv-input-pipeline=N` / `-sv-output-
  pipeline=N` for fixed-stage pipelining.

**Out of scope.** Sophisticated retiming (moving registers
across logic). Just insertion at safe boundaries.

**Effort.** ~2 weeks.

---

## Long-term / exploratory

### 11. MATLAB graphics / `plot` (limited) 🔵

For demos and tutorials. Render `plot(x, y)`, `bar(...)`,
`imagesc(...)` to PNG / SVG via a small wrapper around matplotlib
(Python path) or directly to PNG via stb_image_write (C path).
Not pixel-perfect MATLAB; just enough for quick visualization
of compiled programs.

**Effort.** ~1 week per output target.

---

### 12. `.mat` file save / load 🔵

Already documented in [`docs/save_load_compat.md`](save_load_compat.md).
Goal: read MATLAB v7.3 (HDF5-based) `.mat` files into the runtime
workspace and vice versa. Not a full MATLAB compatibility matrix;
just the common cases (`save('out.mat', '-v7.3')` followed by
`load('out.mat')` in another session).

**Effort.** ~2 weeks.

---

### 13. Toolbox stubs for symbolic / optimization 🔵

Single-file stubs that route to the equivalent open-source
library (`sympy` for Symbolic Math Toolbox, `scipy.optimize`
for Optimization Toolbox). Limited surface; just enough to make
common textbook MATLAB programs that use these toolboxes
compile and run.

**Effort.** Small per stub; total scope depends on which
toolbox.

---

## What's intentionally NOT on the roadmap

- **Full MATLAB language compatibility.** Pursuing this leads to
  toolbox dependencies and `.mat` file format edge cases that
  defeat the project's "self-contained, MathWorks-free" design
  goal.
- **GUI primitives** (`uicontrol`, `app designer` apps). The
  graphics roadmap entry above is rendering-only, no interaction.
- **Live Editor / `.mlx`** notebook format.
- **MEX file compatibility.** The runtime ABI is stable inside
  this project; cross-compatibility with MathWorks's MEX C
  interface is a separate engineering effort that brings little
  benefit since users on this stack want a MathWorks-free path.
- **Code obfuscation / encryption** (MathWorks `.p` files).

---

## Cross-cutting quality work

These don't fit a single roadmap slot but get folded into other
work as it lands:

- **Test corpus growth.** Aim for ≥150 run-tests + 50 SV goldens.
- **Formatter idempotency** verified by a fixed-point CI lane
  (parse → format → parse → format → identical).
- **Doc-up-to-dateness check** as a CI step (parse `feature_status.md`,
  verify every claimed `✅` has at least one test).
- **Performance benchmarks** baseline-tracked across releases —
  matrix-multiply / FIR / FFT / parfor reduction at a few sizes,
  recorded per commit.

---

## Update cadence

This file is updated at the end of each multi-week implementation
arc — most recently after the SystemVerilog Phase 5.6 closure,
the multi-backend persistent + isempty Tier 1, and the docs sync
that produced this file.

Items get demoted from this roadmap to `feature_status.md` /
the relevant `emit_*.md` once shipped. Items get retired (no
demote) when the design has been superseded by a different
approach.
