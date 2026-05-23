# Examples sweep

Automated form of the manual sweep in `docs/examples_status_report.md`. It
compiles, links, and runs **every in-scope `examples/**/*.m`** through the
canonical AOT (LLVM) execute path and classifies each as `OK` / `EMIT` / `LINK`
/ `RUNTIME` / `TIMEOUT` / `SKIP`.

This is a **nightly** lane — it runs on the CI `schedule` (04:00 UTC) regardless
of whether there were any commits. It is **not** a per-PR gate (63+ examples are
known to fail at compile/link time, so it can't block PRs yet).

## What it checks

Examples carry no `.stdout` golden, so **OK = compile + link + run with exit 0**
— it verifies an example doesn't break the toolchain, not its numerics. Numeric
correctness of the headline examples is covered by the per-toolbox `test/Run/*.m`
lanes.

The recipe mirrors `test/Run/run_tests.sh`: the runtime is compiled once into
objects, then each example is `matlabc -emit-llvm` → `clang++`-linked → run from
its own directory under a timeout.

## Coverage guarantee

Every `.m` under `examples/` (recursively) produces exactly one result line
(`OK`/`EMIT`/`LINK`/`RUNTIME`/`TIMEOUT`/`SKIP`/`SKIPSYM`). The script asserts
`#result-lines == #.m-files` and aborts if any example is silently dropped, so
"we run all the examples" stays true. The report prints the reconciliation
(`Total = in-scope (run) + skipped`).

## Scope

`SKIP` (never a failure) covers paths that are not standalone LLVM-execute
programs — they have their own CI lanes:

| Prefix | Real target |
|---|---|
| `hdl/` | SystemVerilog / cocotb (`emit-sv-*`, `cocotb-tests`) |
| `mflow/`, `mflowlink/`, `stateflow/` | Flowchart dialects (`flowchart-*` lanes) |
| examples needing the Symbolic Math Toolbox | SKIPped when SymPP isn't linkable in the environment |

## Regression gate

Every non-SKIP failure is diffed against `known_failures.txt`:

- **failing & not in baseline → REGRESSION** → the script exits non-zero (turns
  the nightly lane red).
- **in baseline but now passing → STALE** → reported so you can prune the entry;
  does not fail the run.

## Usage

```bash
# sweep + gate on regressions (what CI runs)
bash test/Examples/run_sweep.sh build/matlabc

# after intentionally fixing/changing examples, regenerate the baseline
bash test/Examples/run_sweep.sh build/matlabc --update-baseline
```

`CLANG`, `CXX`, `CXXSTD`, `EXAMPLES_TIMEOUT`, `SYMPP_DIR`/`SYMPP_PREFIX` are
honored as environment overrides (same conventions as `test/Run/run_tests.sh`).

`known_failures.txt` is environment-portable: the bulk of entries are `EMIT`
(frontend/lowering) failures that are platform-independent. Sym-only and
cairo-only examples are SKIPped rather than listed, so the baseline matches
whether or not SymPP / cairo are present.
