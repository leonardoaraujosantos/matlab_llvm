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
# Accepts both .m and .mflow inputs (the latter round-trips through
# the flowchart frontend before formatting).
# Example: `just format examples/factorial.m`
format FILE: build
    {{BUILD_DIR}}/matlabc -format {{FILE}}

# Same as `format`, but spelled `-emit-matlab` (the canonical name when
# the input is a `.mflow` flowchart). Identical output for `.m` inputs.
# Example: `just emit-matlab examples/mflow/factorial.mflow`
emit-matlab FILE: build
    {{BUILD_DIR}}/matlabc -emit-matlab {{FILE}}

# Show the parsed FlowDoc for a .mflow file (loader + validation only;
# no AST build). Useful for sanity-checking what the IDE saved.
# Example: `just dump-flow examples/mflow/factorial.mflow`
dump-flow FILE: build
    {{BUILD_DIR}}/matlabc -dump-flow {{FILE}}

# Emit a `.mflow` flowchart from any `.m` (or round-trip a `.mflow`).
# Output is in IDE-canonical JSON format (alphabetical keys,
# 2-space indent, blank-line empty arrays) so re-saves through the
# MatForge IDE produce minimal diffs.
# Example: `just emit-mflow examples/factorial.m > /tmp/factorial.mflow`
emit-mflow FILE: build
    {{BUILD_DIR}}/matlabc -emit-mflow {{FILE}}

# `emit-mflow` with `--preserve-layout`: copy `ui.position` from
# REF for every node id that matches the new emission. Use after
# editing the IDE canvas to keep your hand-placed positions stable
# across regenerations from the source `.m`.
# Example: `just emit-mflow-merge old.mflow examples/factorial.m`
emit-mflow-merge REF FILE: build
    {{BUILD_DIR}}/matlabc -emit-mflow --preserve-layout {{REF}} {{FILE}}

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

# Build and run every flowchart program in examples/mflow/ via the C
# emitter. Stops at the first failure. Parallel to `examples` but for
# the `.mflow` corpus.
mflow-examples: build
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/mflow/*.mflow; do
        name=$(basename "$f" .mflow)
        src=$(mktemp -t mlflow.XXXXXX).c
        out="/tmp/ex_${name}_mflow"
        echo "=== $name ==="
        ./{{BUILD_DIR}}/matlabc -emit-c "$f" > "$src"
        cc -w "$src" runtime/matlab_runtime.c -o "$out" -lm -lpthread
        rm -f "$src"
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

# Generate a CocoTB verification harness from a .m HDL file. The
# output directory is `<input-dir>/<stem>_cocotb/` and is fully
# self-contained (DUT, Python reference, harness, fi helpers,
# matlab_runtime.py, Makefile). See docs/emit_cocotb.md for the
# full feature description and roadmap. LATENCY is the HDL Verifier-
# style pipeline-latency parameter (cycles between input k applied
# and DUT output for input k surfacing); set it to the registered-
# output depth for pipelined DUTs (e.g., 2 for fir_asic_pipelined),
# 0 for combinational and FSM modules. VECTORS is the number of
# random stimulus vectors driven (default 100). Example:
#   just emit-cocotb examples/hdl/alu_16bit.m
#   just emit-cocotb examples/hdl/fir_asic_pipelined.m 2
emit-cocotb FILE LATENCY="0" VECTORS="100": build
    ./{{BUILD_DIR}}/matlabc -emit-cocotb \
        -cocotb-latency={{LATENCY}} \
        -cocotb-vectors={{VECTORS}} \
        {{FILE}}

# Generate a CocoTB harness AND run it. Equivalent to running
# `just emit-cocotb FILE LATENCY` followed by `make` inside the
# generated directory. Requires verilator + cocotb on PATH; falls
# through to the harness's own SIM= override if the user wants
# a different simulator (e.g. `SIM=icarus just verify-cocotb ...`).
# Example:
#   just verify-cocotb examples/hdl/alu_16bit.m
#   just verify-cocotb examples/hdl/fir_asic_pipelined.m 2
verify-cocotb FILE LATENCY="0" VECTORS="100": build
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v verilator >/dev/null 2>&1; then
        echo "verify-cocotb: verilator not on PATH (brew install verilator)" >&2
        exit 1
    fi
    if ! command -v cocotb-config >/dev/null 2>&1; then
        echo "verify-cocotb: cocotb not installed (pip install cocotb)" >&2
        exit 1
    fi
    ./{{BUILD_DIR}}/matlabc -emit-cocotb \
        -cocotb-latency={{LATENCY}} \
        -cocotb-vectors={{VECTORS}} \
        {{FILE}}
    name=$(basename {{FILE}} .m)
    parent=$(dirname {{FILE}})
    outdir="$parent/${name}_cocotb"
    cd "$outdir"
    # Use `${SIM:-verilator}` so the user can override the simulator
    # without editing the generated Makefile.
    SIM="${SIM:-verilator}" make

# Run -emit-cocotb + verify-cocotb across every supported
# examples/hdl/*.m and print a status table. Skips if verilator or
# cocotb is missing — matches the policy used for other optional
# dependencies. Per-module pipeline latency is hand-picked to match
# the registered-output depth (auto-detect lands in v3.4 — see
# docs/emit_cocotb.md). Modules whose ports the v1 harness can't
# drive (vector_processor) and whose semantics need v3 input-stability
# (sequential_processor) are surfaced as SKIP / KNOWN-MISMATCH.
test-cocotb: build
    #!/usr/bin/env bash
    set -uo pipefail
    if ! command -v verilator >/dev/null 2>&1; then
        echo "test-cocotb: verilator missing (skipping)"
        exit 0
    fi
    if ! command -v cocotb-config >/dev/null 2>&1; then
        echo "test-cocotb: cocotb missing (skipping)"
        exit 0
    fi
    # `<name>:<latency>:<expect>` — expect is one of:
    #   pass     — must pass; counts toward exit-code failure
    #   deferred — known-failing pending v3 (sequential_processor's
    #              input-stability gap, vector_processor's vector-port
    #              driving — see docs/emit_cocotb.md). Reported but
    #              doesn't fail the recipe.
    declare -a tests=(
        "alu_16bit:0:pass"
        "counter_0_to_10:0:pass"
        "fir_asic_pipelined:2:pass"
        "mealy_fsm:0:pass"
        "moore_fsm:0:pass"
        "mux_4to_1_16bit:0:pass"
        "vector_processor:0:pass"
        "sequential_processor:4:pass"
    )
    pass=0; fail=0; deferred=0; regressed=0
    for entry in "${tests[@]}"; do
        IFS=: read m L expect <<< "$entry"
        printf "  %-26s L=%s  " "$m" "$L"
        rm -rf "/tmp/test_cocotb/$m" 2>/dev/null || true
        mkdir -p "/tmp/test_cocotb"
        emit_log=$(./{{BUILD_DIR}}/matlabc -emit-cocotb \
                   -cocotb-out="/tmp/test_cocotb/$m" \
                   -cocotb-latency=$L \
                   examples/hdl/$m.m 2>&1)
        if [ $? -ne 0 ]; then
            if [ "$expect" = "deferred" ]; then
                echo "DEFERRED (emit not yet supported)"
                deferred=$((deferred+1))
            else
                echo "FAIL (emit failed)"
                regressed=$((regressed+1))
            fi
            continue
        fi
        run_log=$(cd "/tmp/test_cocotb/$m" && make 2>&1)
        ok=$(echo "$run_log" | grep -oE "TESTS=1 PASS=[0-9]+ FAIL=[0-9]+" | head -1)
        if echo "$ok" | grep -q "PASS=1 FAIL=0"; then
            echo "PASS"
            pass=$((pass+1))
        else
            if [ "$expect" = "deferred" ]; then
                echo "DEFERRED ($ok — expected, see docs/emit_cocotb.md)"
                deferred=$((deferred+1))
            else
                echo "FAIL ($ok) — REGRESSION"
                regressed=$((regressed+1))
            fi
        fi
    done
    echo
    echo "  cocotb: $pass passed, $deferred deferred (expected), $regressed regression(s)"
    [ $regressed -eq 0 ]

# Compile a .m file via the SystemVerilog emitter: writes ./<name>.sv
# and lints with Verilator when available. Parallel to `compile-c` /
# `compile-cpp` / `compile-python` / `compile-typescript`, except
# there's nothing to "run" — the .sv is the artifact, ready for
# downstream synthesis (DC, Genus, Yosys, ...) or simulation.
# Example: `just compile-sv examples/clamp.m` -> ./clamp.sv
compile-sv FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    base=$(basename {{FILE}}); name="${base%.*}"
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

# Compile a .m or .mflow file via the C emitter: produces ./<name>
# using cc. Example: `just compile-c examples/hello.m` -> ./hello
# (works equally for `examples/mflow/hello.mflow`).
compile-c FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    base=$(basename {{FILE}}); name="${base%.*}"
    src=$(mktemp -t mlc.XXXXXX).c
    ./{{BUILD_DIR}}/matlabc -emit-c {{FILE}} > "$src"
    cc -w "$src" runtime/matlab_runtime.c -o "./$name" -lm -lpthread
    rm -f "$src"
    echo "built ./$name"

# Compile a .m or .mflow file via the C++ emitter.
compile-cpp FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    base=$(basename {{FILE}}); name="${base%.*}"
    src=$(mktemp -t mlc.XXXXXX).cpp
    ./{{BUILD_DIR}}/matlabc -emit-cpp {{FILE}} > "$src"
    c++ -w -x c++ "$src" -x c runtime/matlab_runtime.c -o "./$name" -lm -lpthread
    rm -f "$src"
    echo "built ./$name"

# Emit and immediately run a .m or .mflow file via python3.
# Example: `just compile-python examples/hello.m` -> prints hello
compile-python FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    src=$(mktemp -t mlp.XXXXXX).py
    ./{{BUILD_DIR}}/matlabc -emit-python {{FILE}} > "$src"
    PYTHONPATH=runtime python3 "$src"
    rm -f "$src"

# Emit and immediately run a .m or .mflow file via a TypeScript runner
# (bun, tsx, or ts-node — first one found on PATH). Drops the emitted
# .ts into runtime/ so the relative imports of matlab_runtime / numpy_ts
# resolve.
# Example: `just compile-typescript examples/hello.m` -> prints hello
compile-typescript FILE: build
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v bun  >/dev/null 2>&1; then runner="bun run";
    elif command -v tsx >/dev/null 2>&1; then runner="tsx";
    elif command -v ts-node >/dev/null 2>&1; then runner="ts-node --transpile-only";
    else echo "error: need bun, tsx, or ts-node on PATH" >&2; exit 1; fi
    base=$(basename {{FILE}}); stem="${base%.*}"
    src="runtime/__just_compile_${stem}.ts"
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

# Run every flowchart (`.mflow`) ctest lane:
#   - flowchart-tests              (loader + validation, 9 fixtures)
#   - flowchart-emit-matlab-tests  (linear / control / sub-flows / custom, 17 fixtures)
#   - flowchart-cross-backend-tests (`.mflow` ≡ `.m` round-trip, 12 × 4 backends)
#   - flowchart-lsp-tests          (`matlab-lsp` accepts .mflow, 3 cases)
#   - flowchart-dap-tests          (`matlabc -dap` accepts .mflow, 3 cases)
#   - flowchart-emit-mflow-tests   (`.m`/`.mflow` → `.mflow` idempotency, 11 fixtures)
test-flowchart: build
    ctest --test-dir {{BUILD_DIR}} --output-on-failure -R "^flowchart-"

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

# ============================================================================
# Remote backend (server/) — FastAPI edge over the matlabc CLI.
# Requires `uv` (https://docs.astral.sh/uv/). See server/README.md and
# docs/remote_backend_plan.md.
# ============================================================================

# Install backend Python deps (incl. tests) into server/.venv via uv.
backend-install:
    cd server && uv sync --extra dev

# Enables Cairo plotting when present; the flag is sticky in the CMake cache,
# so plain `just build` keeps it afterwards (the Docker image always has it).
# Build matlabc with plotting for the backend (so /v1/plot works locally).
backend-build:
    #!/usr/bin/env bash
    set -euo pipefail
    if pkg-config --exists cairo 2>/dev/null; then
        echo "cairo found — building matlabc with /v1/plot support"
        cmake -S . -B {{BUILD_DIR}} -G Ninja -DCMAKE_BUILD_TYPE=Release -DMATLAB_LLVM_WITH_PLOT=ON
    else
        echo "cairo not found — /v1/plot disabled (install cairo, e.g. 'brew install cairo')"
        cmake -S . -B {{BUILD_DIR}} -G Ninja -DCMAKE_BUILD_TYPE=Release
    fi
    cmake --build {{BUILD_DIR}} --target matlabc {{ if JOBS != "" { "-j " + JOBS } else { "" } }}

# Build matlabc (with plotting) + serve the backend on :8000 (/docs, /healthz).
backend-up PORT="8000": backend-build
    cd server && MATLAB_BACKEND_MATLABC_BIN="{{justfile_directory()}}/{{BUILD_DIR}}/matlabc" \
        uv run uvicorn main:app --host 0.0.0.0 --port {{PORT}}

# Build matlabc (with plotting) + serve with auto-reload (for editing the server).
backend-dev PORT="8000": backend-build
    cd server && MATLAB_BACKEND_MATLABC_BIN="{{justfile_directory()}}/{{BUILD_DIR}}/matlabc" \
        uv run uvicorn main:app --reload --host 0.0.0.0 --port {{PORT}}

# Run the backend test suite (fake matlabc stub — no LLVM build needed).
backend-test:
    cd server && uv run --extra dev pytest -q

# Backend tests with a coverage report (target: >90%).
backend-cov:
    cd server && uv run --extra dev pytest -q --cov --cov-report=term-missing

# Live-server integration tests: boot uvicorn and hit it over real HTTP/WS.
# Uses build/matlabc when present (real compiler), else a fake stub.
backend-itest:
    cd server && uv run --extra dev pytest integration -q

# Same suite, against an already-deployed backend (e.g. the Coolify URL).
# Auth is resolved in this order: BACKEND_TOKEN env (static / pre-minted) →
# CYBERDYNE_USER + CYBERDYNE_PASS (logs in to CYBERDYNE_AUTH_URL, default
# https://auth.backend.coolify.cyberdynecorp.ai) → none.
# Usage:
#   just backend-test-remote URL=https://matlab-backend.coolify.cyberdynecorp.ai \
#       USER=leotest@test.com PASS='kugmet-5zozki-nuwJef'
#   just backend-test-remote URL=https://… TOKEN='ey…'
backend-test-remote URL TOKEN="" USER="" PASS="" AUTH="https://auth.backend.coolify.cyberdynecorp.ai" EXPECT_SANDBOX="":
    cd server && BACKEND_URL="{{URL}}" BACKEND_TOKEN="{{TOKEN}}" \
        CYBERDYNE_USER="{{USER}}" CYBERDYNE_PASS="{{PASS}}" \
        CYBERDYNE_AUTH_URL="{{AUTH}}" \
        EXPECT_SANDBOX="{{EXPECT_SANDBOX}}" \
        uv run --extra dev pytest integration -v
