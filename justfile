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
    UPDATE=1 ./test/run_tests.sh {{BUILD_DIR}}/matlabc

# Build a standalone executable from a .m file using the runtime shim.
# Example: `just compile examples/hello.m` produces ./hello.
compile FILE OUT="":
    ./runtime/build_and_run.sh {{FILE}} {{OUT}}

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
