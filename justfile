# matlab_llvm — development task runner.
#
# Install `just` (https://github.com/casey/just) and invoke commands as
# `just <target>`. All commands run from the repo root.
#
# Environment:
#   BUILD_DIR   output directory (default: build)
#   JOBS        parallel build jobs (default: all cores, via ninja)
#   CLANG       clang binary used by the Run suite (default:
#               /opt/homebrew/opt/llvm/bin/clang on macOS, clang on Linux)

BUILD_DIR := env_var_or_default("BUILD_DIR", "build")
JOBS      := env_var_or_default("JOBS", "")

# Show the available recipes.
default:
    @just --list

# Configure the build (CMake + Ninja). Re-run after CMakeLists.txt edits.
configure:
    cmake -S . -B {{BUILD_DIR}} -G Ninja

# Fast build. Implicitly re-runs CMake if needed.
build: configure
    cmake --build {{BUILD_DIR}} {{ if JOBS != "" { "-j " + JOBS } else { "" } }}

# Build without MLIR/LLVM (frontend only — useful on machines without
# Homebrew's `llvm` installed).
build-frontend:
    cmake -S . -B {{BUILD_DIR}} -G Ninja -DMATLAB_LLVM_WITH_MLIR=OFF
    cmake --build {{BUILD_DIR}}

# Run the full test suite: frontend goldens + Run execution tests.
test: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure

# Frontend goldens only (no linking / execution).
test-frontend: build
    ./test/run_tests.sh {{BUILD_DIR}}/matlabc

# Build-and-run tests only (requires MLIR build).
test-run: build
    ./test/Run/run_tests.sh {{BUILD_DIR}}/matlabc

# Regenerate all golden `.expected` / `.stdout` files. Use after an
# intentional output change.
update-goldens: build
    UPDATE=1 ./test/run_tests.sh $(pwd)/{{BUILD_DIR}}/matlabc

# Build a standalone executable from a .m file using the runtime shim.
# Example: `just compile examples/hello.m` produces ./hello.
compile FILE OUT="":
    ./runtime/build_and_run.sh {{FILE}} {{OUT}}

# Launch the JIT-backed REPL. Variables persist across input lines via
# a runtime-side workspace; blocks (if/for/while/...) auto-continue
# until their matching `end`.
repl: build
    {{BUILD_DIR}}/matlabc -repl

# Print a canonically-formatted version of a .m file to stdout.
# Example: `just format examples/factorial.m`
format FILE: build
    {{BUILD_DIR}}/matlabc -format {{FILE}}

# Build and run every program in examples/. Stops at the first failure.
examples: build
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/*.m; do
        name=$(basename "$f" .m)
        out="/tmp/ex_$name"
        echo "=== $name ==="
        ./runtime/build_and_run.sh "$f" "$out" >/dev/null
        "$out"
        echo
    done

# Show the token stream for a .m file.
tokens FILE: build
    ./{{BUILD_DIR}}/matlabc -dump-tokens {{FILE}}

# Show the parsed AST for a .m file.
ast FILE: build
    ./{{BUILD_DIR}}/matlabc -dump-ast {{FILE}}

# Show the Sema-annotated AST (bindings + inferred types).
sema FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-sema {{FILE}}

# Show the in-house MIR.
mir FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-mir {{FILE}}

# Show the MLIR module (pre-optimization).
mlir FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-mlir {{FILE}}

# Show the MLIR module after opt passes (SlotPromotion + scalar-to-arith).
mlir-opt FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-mlir -opt {{FILE}}

# Show the final LLVM IR (what clang links).
llvm FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-llvm {{FILE}}

# Emit self-contained C that links against runtime/matlab_runtime.c.
emit-c FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-c {{FILE}}

# Emit self-contained C++ (same semantics, extern "C" wrap around runtime).
emit-cpp FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-cpp {{FILE}}

# Emit self-contained Python that imports runtime/matlab_runtime.py.
emit-python FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-python {{FILE}}

# Emit self-contained TypeScript that imports runtime/matlab_runtime.ts
# (numpy-ts-backed shim). Run with bun / tsx / ts-node.
emit-typescript FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-typescript {{FILE}}

# Emit synthesizable SystemVerilog (ASIC target) from a .m file.
# Phase 1 — scalar combinational only. See docs/emit_systemverilog.md.
emit-sv FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-systemverilog {{FILE}}

# Multi-file SystemVerilog emit: typed driver + module file(s). Use
# this when the function lives in its own .m file (the MATLAB
# convention) and a separate driver provides the typed call site that
# the user-call refinement pipeline needs to fix port widths. Output
# goes to stdout — pipe to a file or use `compile-sv-multi` for the
# write+lint combo. Example:
#   just emit-sv-multi examples/hdl/alu_16bit_synth.m \
#                      examples/hdl/alu_16bit.m
emit-sv-multi DRIVER MODULE *EXTRA: build
    ./{{BUILD_DIR}}/matlabc -emit-systemverilog \
        {{DRIVER}} {{MODULE}} {{EXTRA}}

# Run the synthesizability gate on a .m file without producing
# output. Exit 0 means the source can be synthesized; non-zero with a
# diagnostic means it cannot.
check-sv FILE: build
    ./{{BUILD_DIR}}/matlabc -check-synthesizable {{FILE}}

# Emit + lint with Verilator. Requires `verilator` on PATH.
lint-sv FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    name=$(basename {{FILE}} .m)
    out=$(mktemp -d)/$name.sv
    ./{{BUILD_DIR}}/matlabc -emit-systemverilog {{FILE}} > "$out"
    verilator --lint-only -Wall --top-module "$name" "$out"
    echo "lint OK: $out"

# Run the SystemVerilog emission test suites (golden-diff + lint +
# synthesizability gate). Verilator is auto-detected; set
# `VERILATOR=` to skip the lint pass.
test-sv: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure -R "emit-sv-(fail-)?tests"

# Compile a .m file via the SystemVerilog emitter: writes ./<name>.sv
# and lints with Verilator when available. Parallel to `compile-c` /
# `compile-cpp` / `compile-python` / `compile-typescript`, except
# there's nothing to "run" — the .sv is the artifact, ready for
# downstream synthesis (DC, Genus, Yosys, ...) or simulation.
# Example: `just compile-sv examples/clamp.m` -> ./clamp.sv
compile-sv FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    name=$(basename {{FILE}} .m)
    out="./$name.sv"
    ./{{BUILD_DIR}}/matlabc -emit-systemverilog {{FILE}} > "$out"
    if command -v verilator >/dev/null 2>&1; then
        verilator --lint-only -Wall --top-module "$name" "$out"
        echo "built $out (lint OK)"
    else
        echo "built $out (verilator not on PATH; lint skipped)"
    fi

# Pre-synthesis hardware report (Phase 5.5). Walks the same post-
# pipeline IR `-emit-systemverilog` consumes and prints a Markdown
# summary per user function — operator counts, register widths,
# FSM state counts. Useful before invoking your downstream synth
# tool so you see the resource shape of each module first.
report-hw FILE: build
    ./{{BUILD_DIR}}/matlabc -emit-hardware-report {{FILE}}

# Multi-file SystemVerilog compile: typed driver + module file(s).
# Useful when a function lives in its own .m file (the MATLAB
# convention) and a separate driver provides the typed call site that
# the user-call refinement pipeline consumes. The first non-driver
# argument's basename is used as the output module name.
# Example:
#   just compile-sv-multi driver.m examples/hdl/mux_4to_1_16bit.m
compile-sv-multi DRIVER MODULE *EXTRA: build
    #!/usr/bin/env bash
    set -euo pipefail
    name=$(basename {{MODULE}} .m)
    out="./$name.sv"
    ./{{BUILD_DIR}}/matlabc -emit-systemverilog \
        {{DRIVER}} {{MODULE}} {{EXTRA}} > "$out"
    if command -v verilator >/dev/null 2>&1; then
        verilator --lint-only -Wall --top-module "$name" "$out"
        echo "built $out (lint OK)"
    else
        echo "built $out (verilator not on PATH; lint skipped)"
    fi

# Compile a .m file via the C emitter: produces ./<name> using cc.
# Example: `just compile-c examples/hello.m` -> ./hello
compile-c FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    name=$(basename {{FILE}} .m)
    src=$(mktemp -t mlc.XXXXXX).c
    ./{{BUILD_DIR}}/matlabc -emit-c {{FILE}} > "$src"
    cc -w "$src" runtime/matlab_runtime.c -o "./$name" -lm -lpthread
    rm -f "$src"
    echo "built ./$name"

# Compile a .m file via the C++ emitter.
compile-cpp FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    name=$(basename {{FILE}} .m)
    src=$(mktemp -t mlc.XXXXXX).cpp
    ./{{BUILD_DIR}}/matlabc -emit-cpp {{FILE}} > "$src"
    c++ -w -x c++ "$src" -x c runtime/matlab_runtime.c -o "./$name" -lm -lpthread
    rm -f "$src"
    echo "built ./$name"

# Emit and immediately run a .m file via python3.
# Example: `just compile-python examples/hello.m` -> prints hello
compile-python FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    src=$(mktemp -t mlp.XXXXXX).py
    ./{{BUILD_DIR}}/matlabc -emit-python {{FILE}} > "$src"
    PYTHONPATH=runtime python3 "$src"
    rm -f "$src"

# Emit and immediately run a .m file via a TypeScript runner (bun, tsx,
# or ts-node — first one found on PATH). Drops the emitted .ts into
# runtime/ so the relative imports of matlab_runtime / numpy_ts resolve.
# Example: `just compile-typescript examples/hello.m` -> prints hello
compile-typescript FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v bun  >/dev/null 2>&1; then runner="bun run";
    elif command -v tsx >/dev/null 2>&1; then runner="tsx";
    elif command -v ts-node >/dev/null 2>&1; then runner="ts-node --transpile-only";
    else echo "error: need bun, tsx, or ts-node on PATH" >&2; exit 1; fi
    src="runtime/__just_compile_$(basename {{FILE}} .m).ts"
    ./{{BUILD_DIR}}/matlabc -emit-typescript {{FILE}} > "$src"
    trap 'rm -f "$src"' EXIT
    (cd runtime && $runner "$(basename "$src")")

# Emit every program in examples/ to .py files under OUT (default /tmp/emit-python-examples).
# Useful for eyeballing the generated code across the whole corpus at once.
emit-python-examples OUT="/tmp/emit-python-examples": build
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{OUT}}"
    for f in examples/*.m; do
        name=$(basename "$f" .m)
        ./{{BUILD_DIR}}/matlabc -emit-python "$f" > "{{OUT}}/$name.py"
        echo "wrote {{OUT}}/$name.py"
    done

# Emit every program in examples/ to .ts files under OUT (default /tmp/emit-typescript-examples).
# Useful for eyeballing the generated code across the whole corpus at once.
emit-typescript-examples OUT="/tmp/emit-typescript-examples": build
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{OUT}}"
    for f in examples/*.m; do
        name=$(basename "$f" .m)
        ./{{BUILD_DIR}}/matlabc -emit-typescript "$f" > "{{OUT}}/$name.ts"
        echo "wrote {{OUT}}/$name.ts"
    done

# Run both C and C++ emission test suites (95 programs each).
test-emitc: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure \
        -R "(run-tests-emit-(c|cpp)(-strict)?|emitc-fail-tests)"

# Run the Python emission suite.
test-emitpython: build
    ./test/Run/run_tests_emitpython.sh ./{{BUILD_DIR}}/matlabc

# Run the TypeScript emission suite (uses bun/tsx/ts-node — first found).
test-emitts: build
    ./test/Run/run_tests_emitts.sh ./{{BUILD_DIR}}/matlabc

# Remove the build directory.
clean:
    rm -rf {{BUILD_DIR}}

# Wipe only the binary artifacts; keep the cmake cache.
rebuild:
    cmake --build {{BUILD_DIR}} --target clean
    cmake --build {{BUILD_DIR}}

# Line-count the project.
loc:
    @find include lib tools runtime test -type f \
        \( -name '*.cpp' -o -name '*.h' -o -name '*.c' -o \
           -name '*.def' -o -name '*.m' -o -name '*.sh' \) \
        | xargs wc -l | tail -1
