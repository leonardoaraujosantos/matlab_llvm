# C / C++ Emission

This document describes the `-emit-c` and `-emit-cpp` backends added on the
`emit_c_cpp` branch: what they do, where in the pipeline they run, the
design alternatives we considered, and why we picked the approach that
shipped.

## What it is

Two new driver flags on `matlabc`:

| Flag | Output |
|---|---|
| `-emit-c`   | Self-contained `.c` source |
| `-emit-cpp` | Self-contained `.c++` source |

Both produce code that links against the existing `runtime/matlab_runtime.c`
to form a standalone executable. No LLVM toolchain is required at the
compile step — just a plain C or C++ compiler:

```bash
matlabc -emit-c foo.m > foo.c
cc foo.c runtime/matlab_runtime.c -o foo -lm -lpthread
./foo
```

On the full test corpus (12 `examples/` programs + 95 `test/Run/*.m`
programs), the stdout of executables built through the C/C++ path matches
the stdout of the LLVM path byte-for-byte. See `just test-emitc`.

## Where in the pipeline this runs

Both flags reuse the same MLIR lowering pipeline as `-emit-llvm` up to and
including `runLowerIO`. After that, the driver branches:

- `-emit-llvm` runs the MLIR → LLVM conversion pipeline
  (`scf → cf`, `arith → llvm`, `func → llvm`, reconcile casts) and calls
  `mlir::translateModuleToLLVMIR`.
- `-emit-c` / `-emit-cpp` calls `emitC(ModuleOp, bool Cpp)`
  (in `lib/MLIR/Passes/EmitC.cpp`) which walks the module directly and
  prints C.

```
AST ──► MLIR ──► [SlotPromotion, LowerScalarsToArith, OutlineParfor,
                  LowerSeqLoops, LowerAnonCalls, LowerUserCalls,
                  LowerTensorOps, Monomorphise, LowerScalarSlots,
                  LowerIO]
                       │
                       ├──► [scf→cf, arith→llvm, func→llvm]
                       │          ───► translateModuleToLLVMIR ───► .ll
                       │
                       └──► emitC()  ───► .c / .c++
```

This is a deliberately early fork-point. At that snapshot, matrix /
struct / cell / error-flag / parfor / global / persistent semantics have
all been lowered to `llvm.call`s against the C runtime, but `scf.while`
and `scf.if` are still structural — much easier to print as C than
post-CFG-conversion basic blocks connected by `cf.br`.

## Closed set of ops the emitter handles

The only ops that survive into the snapshot are:

| Dialect | Ops |
|---|---|
| `func` | `func.func`, `func.call`, `func.return` |
| `arith` | `arith.constant`, `arith.{add,sub,mul,div}{f,i}`, `arith.{and,or,xor}i`, `arith.cmp{f,i}`, `arith.select`, casts (`sitofp`, `fptosi`, `extsi`, `trunci`, `extf`, `truncf`) |
| `scf` | `scf.while`, `scf.if`, `scf.condition`, `scf.yield` |
| `llvm` | `llvm.call`, `llvm.func` (decl + outlined defn), `llvm.alloca`, `llvm.load`, `llvm.store`, `llvm.mlir.global`, `llvm.mlir.addressof`, `llvm.mlir.constant`, `llvm.mlir.zero`, `llvm.getelementptr`, `llvm.return` |

Each op maps to a one-liner of C:

| MLIR | C output |
|---|---|
| `%c = arith.constant 1.0 : f64` | `double v0 = 1;` |
| `%s = arith.addf %a, %b` | `double v1 = v0 + v2;` |
| `%p = llvm.alloca %one x f64` | `double v3_slot = 0; void* v3 = &v3_slot;` |
| `%p = llvm.alloca %one x !llvm.array<N x f64>` | `double v3_slot[N] = {0}; void* v3 = v3_slot;` |
| `%r = llvm.getelementptr %base[%i]` | `void* v = (void*)(((double*)base) + i);` |
| `llvm.store %v, %p : f64, !llvm.ptr` | `*(double*)p = v;` |
| `%x = llvm.load %p : !llvm.ptr -> f64` | `double x = *(double*)p;` |
| `llvm.call @matlab_foo(%a, %b)` | `matlab_foo(a, b);` |
| `llvm.call %fp(%a)` (indirect) | `((double(*)(double))fp)(a);` |
| `scf.if %c { ... }` | `if (c) { ... }` |
| `scf.while {...} do {...}` | `while (1) { <before>; if (!cond) break; <after>; }` |
| `llvm.mlir.global @s = "..."` | `static const unsigned char s[N] = {...};` |

## Runtime ABI bridge

The runtime (`runtime/matlab_runtime.c`) is plain C with typed pointer
parameters: `matlab_disp_mat(matlab_mat *A)`, `matlab_struct_set_f64(matlab_struct *s, ...)`,
etc. MLIR after `LowerTensorOps` uses the opaque `!llvm.ptr` for all of
these, which we print as `void*`.

Mixing typed C params with `void*` callers causes C++ to reject the
implicit cast (and emits a warning on strict C, too). Rather than
generating per-callsite casts to the correct typed pointer, the emitter
takes a simpler route: **it emits its own runtime prototypes inline**,
using `void*` for every pointer parameter, inside `extern "C"` for C++:

```c
extern "C" {
extern void* matlab_matmul_mm(void*, void*);
extern void matlab_disp_mat(void*);
extern void matlab_disp_str(void*, int64_t);
// ... one per runtime symbol referenced in this module
}
```

At link time the C linkage resolves these `void*`-param declarations to
the runtime's typed definitions; the ABI is the same because all of these
pointer types have the same size and calling convention. The emitted file
doesn't `#include "matlab_runtime.h"`.

This approach has three properties we care about:

1. **No per-callsite cast clutter** — the emitted source reads like
   hand-written C.
2. **Same output works for both C and C++** — the only difference between
   `-emit-c` and `-emit-cpp` is the `extern "C"` wrap and the presence or
   absence of `<stdbool.h>`.
3. **Resilient to runtime refactors** — if you rename `matlab_mat*` or
   add a new pointer-typed entry point, the emitter doesn't need to know
   about the type, only the arity.

## SSA naming & control flow

### Value numbering

Every `mlir::Value` is assigned a stable `vN` identifier the first time
it's referenced, via a `DenseMap<Value, std::string>`. Producers declare
the local (`double v1 = ...;`); users reference it by name. Block
arguments are named the same way — when entering a new `func.func` /
`llvm.func`, each parameter is pre-bound to a fresh name used in the
signature and inside the body.

### `scf.if`

For each result of the `scf.if`, a mutable local is declared above the
`if`. Each `scf.yield` inside a branch becomes an assignment to the
corresponding local. Uses of the `scf.if`'s results downstream reference
those locals by name.

```
scf.if %c -> f64 {
  scf.yield %a : f64
} else {
  scf.yield %b : f64
}
```
becomes
```c
double v0 = 0;
if (c) {
  v0 = a;
} else {
  v0 = b;
}
```

### `scf.while`

For each iter-arg, a mutable local is declared above the loop, and the
before-block's block argument is bound to that same name (so reads
inside the before-region see the current iteration's value). The
`scf.condition` terminator becomes `if (!cond) break;`, and the
after-block's block argument is bound to the corresponding value
forwarded by `scf.condition`. The final `scf.yield` at the end of the
after-region becomes an assignment back into the iter-arg locals.

```
%r = scf.while (%iv = %start : f64) : (f64) -> f64 {
  %c = arith.cmpf ole, %iv, %end
  scf.condition(%c) %iv
} do {
^bb(%iv0: f64):
  ...
  %next = arith.addf %iv0, %step
  scf.yield %next
}
```
becomes
```c
double v0 = start;
while (1) {
  bool c = (v0 <= end);
  if (!c) break;
  ...
  double next = v0 + step;
  v0 = next;
}
```

This relies on each region being single-block. `LowerSeqLoops`,
`OutlineParfor`, and `LowerAnonCalls` all produce single-block regions
today, so the emitter never needs to handle `cf.br` / `cf.cond_br`.

### Indirect calls

`llvm.call` without a callee attribute is a function-pointer call —
produced by `LowerAnonCalls` for anonymous functions and
`LowerAnonCallsPost` for function handles with matrix captures. The
emitter casts the pointer through the correct function type built from
the call's argument/result types:

```c
double v3 = ((double(*)(double))fp)(arg);
```

## Design alternatives (the Option A vs B discussion)

When I scoped this work I considered three broad approaches. Here's the
full comparison; the branch ships Option A.

### Option A — Custom walker that prints C (chosen)

Walk the post-`LowerIO` MLIR module directly and print C. One pass, one
output file.

**Pros**
- Small. The final printer is ~600 LoC in a single file, mirrors
  `LowerToLLVMIR.cpp` in shape.
- Closed set of ops. After all the `Lower*` passes run, the IR uses
  maybe a dozen distinct op kinds — each becomes one case in the
  printer. Easy to audit, easy to extend when a new runtime feature
  lands.
- Full control over style. We can choose to emit mutable locals for
  `scf.while` iter-args (rather than threading them as function
  arguments or `setjmp`-style), choose `unsigned char` for string
  literals to avoid C++11 narrowing, and decide exactly which runtime
  prototypes to emit.
- Matches the runtime's shape. `runtime/matlab_runtime.c` is already
  plain C with uniform `matlab_*` function calls on opaque pointers.
  The MLIR at the snapshot point is *already* C-shaped. The emitter is
  almost a transliteration.
- Zero new MLIR dialect dependencies. No new `find_package`, no new
  `target_link_libraries`. Keeps the build minimal.

**Cons**
- We own the printer. If upstream MLIR invents new `arith` ops or an
  existing pass starts producing a new op kind, we have to teach the
  printer about it.
- No reuse of upstream infrastructure for diagnostics, formatting, or
  future refactors.

### Option B — Convert to the upstream `emitc` dialect, use `mlir-translate --mlir-to-cpp`

MLIR ships an `emitc` dialect whose ops map 1:1 to C statements, and a
`translateToCpp` pass that prints them. The approach would be: lower
our post-`LowerIO` module to `emitc`, then call the upstream translator.

**Pros**
- Upstream ownership. Bug fixes and improvements in the `emitc`
  translator come for free.
- Standard practice. Other MLIR projects emitting C (TFLite, IREE in
  some configurations) use this dialect.
- Potentially lower printer LoC if everything maps cleanly.

**Cons**
- Lots of conversion scaffolding we'd have to write ourselves:
  - `ConvertArithToEmitC` exists in upstream but covers only part of
    the `arith` ops we use.
  - `ConvertSCFToEmitC` does **not** exist — we'd write `scf.while` and
    `scf.if` to `emitc` ourselves, and it would look a lot like the
    printer we wrote anyway.
  - `ConvertLLVMToEmitC` does **not** exist — and we have meaningful
    amounts of `llvm.call`, `llvm.alloca`, `llvm.getelementptr`,
    `llvm.mlir.global`, `llvm.mlir.addressof` at the snapshot point.
- The `emitc` type system (`emitc.ptr`, `emitc.opaque<"matlab_mat*">`,
  `emitc.array`) needs to bridge the three opaque runtime pointer
  flavours (`matlab_mat*`, `matlab_struct*`, `matlab_cell*`). We'd need
  to pick an opaque-type convention and stick to it.
- New build dependencies. `MLIREmitCDialect` + `MLIRTargetCpp` + the
  conversion passes we'd write and register.
- Less control over output shape. `translateToCpp`'s style is set by
  upstream; if it prints `scf`-equivalent loops in a way we don't like
  (e.g. threading iter-args through function parameters, adding
  synthetic scope braces) we can't easily fix that without forking.
- Couples our backend to upstream MLIR's `emitc` evolution. Not
  inherently bad, but more moving parts than a 600-line walker.

### Option C — Hybrid

Use `emitc` for scalar/control-flow ops and a custom printer for runtime
calls + matrix pointers. In theory this gives the best of both.

In practice it was the worst: two sources of truth (our printer + the
upstream translator), two bugs to chase when output went wrong, and
bridging code between the two layers. Discarded early.

### The decision

Option A won on three fronts:

1. **Closed surface.** The set of ops at the snapshot point is small,
   stable, and fully under our control (our own lowering passes produce
   them). No runtime-driven growth.
2. **Low marginal cost.** The runtime is already C with `void*`
   pointers and uniform call shapes. The MLIR at the snapshot is the
   same shape. Translating one to the other is linear work.
3. **No new dependencies.** No new MLIR dialects pulled in, no new
   cmake targets, no new version pin.

Option B's main advantage — upstream maintenance — doesn't materialize
while the conversion layer we'd need doesn't exist upstream. If
`ConvertSCFToEmitC` and `ConvertLLVMToEmitC` land in a future LLVM
release, revisiting Option B becomes cheaper; until then, Option A
dominates.

## Files

| File | Purpose |
|---|---|
| `lib/MLIR/Passes/EmitC.cpp` | The emitter — all the case logic described above |
| `include/matlab/MLIR/Passes/Passes.h` | `emitC(ModuleOp, bool Cpp)` declaration |
| `tools/matlabc/main.cpp` | `-emit-c` / `-emit-cpp` flag parsing; reuses the `-emit-llvm` pipeline up through `runLowerIO`, then calls `emitC()` |
| `runtime/matlab_runtime.h` | Optional header with typed `matlab_*` prototypes — not used by the emitter itself (which inlines its own `void*` prototypes) but handy if you're writing C that links against the runtime by hand |
| `test/Run/run_tests_emitc.sh` | Per-`.m` runner: emit, compile with `cc`/`c++`, execute, diff stdout against the existing `.stdout` golden |
| `CMakeLists.txt` | Registers `EmitC.cpp` in `MatlabMLIR` and adds the `run-tests-emit-c` / `run-tests-emit-cpp` CTest targets |
| `justfile` | `emit-c` / `emit-cpp` / `compile-c` / `compile-cpp` / `test-emitc` recipes |

## Verification

| Suite | Tests | Result |
|---|--:|---|
| `frontend-tests` (Lexer, Parser, Sema, MIR, MLIR, Opt, Programs, Errors) | 77 | 77/77 pass |
| `run-tests`      (`-emit-llvm` + clang)             | 95 | 95/95 pass |
| `run-tests-emit-c`   (`-emit-c`  + `cc`, **new**)   | 95 | 95/95 pass |
| `run-tests-emit-cpp` (`-emit-cpp` + `c++`, **new**) | 95 | 95/95 pass |
| `run-tests-emit-c-strict`  (`-emit-c`  + `cc  -Wall -Wextra -Werror`) | 95 | 95/95 pass |
| `run-tests-emit-cpp-strict`(`-emit-cpp` + `c++ -Wall -Wextra -Werror`) | 95 | 95/95 pass |
| `emitc-fail-tests`  (fail-fast diagnostic contract)     | 1+ | pass |

`ctest --test-dir build` runs all seven targets end-to-end.

The strict lanes exempt `-Wunused-variable`, `-Wunused-but-set-variable`,
`-Wunused-parameter`, and `-Wunused-function`. The emitter produces one C
local per SSA value and a local per alloca's underlying storage, so in a
branch where a value isn't consumed it shows up as unused-but-declared.
That's deliberate. Everything else — implicit declarations, type
confusion, sign mismatches, missing returns, uninitialized use — must
still pass.

The `emitc-fail-tests` suite (under `test/EmitCFail/`) locks in the
fail-fast contract: for each `.m` that contains an op the emitter can't
handle, `matlabc -emit-c` must (a) exit non-zero and (b) include the
matching `.stderr` golden's text as a substring of stderr. Today the
suite covers `matlab.call_builtin` (e.g. `mod(x,y)`); new entries can
be added as new unsupported ops come up.

## Known limitations / future work

- **`scf.for`.** Not currently emitted by any pass on this branch, so
  the printer doesn't handle it. Adding it would be a single case —
  the shape is simpler than `scf.while`.
- **Multi-block regions / `cf.br`.** Same story: not produced by any
  current lowering pass. If a future pass starts producing CFG-shaped
  control flow, the printer would need to synthesise `goto` labels or
  the pass would need to stop lowering past `scf`.
- **Parfor determinism.** The emitted C path uses the same
  `matlab_parfor_dispatch` runtime entry as the LLVM path, so output
  ordering is equally nondeterministic across runs. The `.sorted`
  marker files already used by the LLVM test runner apply identically
  to the C/C++ test runners.
- **Non-UTF-8 string literals.** ASCII-safe strings are emitted as
  `"..."` C literals with `\n` / `\t` / `\r` escapes. Strings containing
  non-printable or non-ASCII bytes fall back to an `unsigned char[]`
  byte array so C++ narrowing conversions and escape ambiguity are
  avoided.

## Robustness

The emitter fails fast rather than producing broken output:

- **Unknown ops** hit a fallthrough that prints `/* UNSUPPORTED: <op> */`
  **and** flags the emission as failed. `matlabc -emit-c` exits non-zero
  with `error: emit-c: unsupported op in emitter: <mnemonic>` on stderr.
  This catches both the "undeclared identifier" case (op had a result
  downstream code referenced) and the silent-drop case (op had no
  results — a side-effecting op we forgot, which would otherwise miscompile).
- **Multi-result `func.func` / `llvm.func`** fails with a clear
  diagnostic rather than emitting invalid C. The printer only supports
  0- or 1-result functions; if a future pass produces multi-return
  signatures, the emitter refuses to continue.
- **Pre-emit `mlir::verify`** runs right before `emitC()` in the driver,
  so a malformed IR state is surfaced with a clear MLIR diagnostic
  rather than as a cryptic cc/c++ compile failure on the generated
  source.

## Readability features

- **MATLAB variable names flow through.** `matlab.alloc` carries a
  `name` StringAttr from the frontend. `LowerScalarSlots` and
  `LowerTensorOps` propagate it to the resulting `llvm.alloca` as a
  discardable `matlab.name` attribute. The emitter uses it to name C
  locals, so a MATLAB `total = 0; for i = 1:10; total = total + i; end`
  produces C locals called `total`, `i`, `total_p`, `i_p` (rather than
  `v0_slot`, `v1_slot`, etc.).
- **Function parameter names flow through.** AST→MLIR lowering attaches
  `matlab.name` as an arg attribute on each `func.func` parameter.
  `function y = fact(n)` emits `static double fact(double n)`.
- **Single-use SSA values inline into their user.** A pre-emission
  analysis marks pure single-use producers (constants, arith/cmp/select/
  casts, loads without intervening store/call, GEPs) as inlineable. At
  emission time, `exprFor(V)` lazily builds the C expression from the
  producer instead of declaring a `vN` local. `factorial.m` drops from
  ~90 emitted lines to 62 and the recursive branch reads
  `*(double*)y_p = ((*(double*)n_slot_p) * v7);` instead of a five-line
  sequence of temporaries. Constants always inline (any use count);
  other pure ops inline only when single-use and same-block to avoid
  duplicating work. The analysis is purely a printer optimization —
  it doesn't change the MLIR module, stdout, or the fail-fast contract.
- **`#line` directives** map every emitted statement back to the
  originating line in the `.m` source, so debuggers can step through
  the MATLAB program by its original lines. Requires the `SourceManager`
  to be threaded into `lowerToMLIR` (the driver does this automatically).
  Emits basenames, not absolute paths.
- **ASCII-safe quoted strings** for readable hand-inspection.
- **Per-section blank lines** between the runtime-extern block, the
  string-constant block, the forward-decl block, and the function
  bodies.
